"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py
"""

import os
import time
import math
import pickle
import sys
from contextlib import nullcontext
from dataclasses import dataclass, asdict, field
from ast import literal_eval

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter

from model import GPT2

# -----------------------------------------------------------------------------
# Configuration Classes
# -----------------------------------------------------------------------------

@dataclass
class SystemConfig:
    device: str = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    compile: bool = True # use PyTorch 2.0 to compile the model to be faster
    backend: str = 'nccl' # 'nccl', 'gloo', etc.
    out_dir: str = 'out'

@dataclass
class DatasetConfig:
    dataset: str = 'tinyshakespeare'
    num_workers: int = 4
    gradient_accumulation_steps: int = 30 # 16 * 30 = 480
    batch_size: int = 16
    block_size: int = 1024
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    checkpoint_interval: int = 100
    init_from: str = 'resume' # 'scratch' or 'resume' or 'gpt2*'

@dataclass
class TrainConfig:
    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5

@dataclass
class ModelConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    vocab_size: int = 50304 # default, will be adjusted
    block_size: int = 1024
    gradient_checkpointing: bool = True # Trade compute to save memory
    # Modernization flags
    use_rmsnorm: bool = False
    use_rope: bool = False
    use_swiglu: bool = False

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def get_latest_checkpoint(out_dir):
    """Finds the latest checkpoint file in out_dir."""
    if not os.path.exists(out_dir):
        return None
    
    # Check for specific files
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if os.path.exists(ckpt_path):
        return ckpt_path

    # Check for versioned files ckpt_{iter}.pt
    files = [f for f in os.listdir(out_dir) if f.startswith('ckpt_') and f.endswith('.pt')]
    if not files:
        return None
    
    # Extract version numbers
    def extract_iter(f):
        try:
            return int(f.split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            return -1
            
    files = sorted(files, key=extract_iter, reverse=True)
    if files:
        return os.path.join(out_dir, files[0])
    
    return None

class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size
        
    def __getitem__(self, idx):
        # We need to ensure we don't go out of bounds, though __len__ helps.
        # Ideally we'd just pick random indices like before for infinite stream, 
        # but Dataset usually maps index -> item.
        # To strictly replicate the previous "infinite random sampling" behavior without epochs:
        # We can make the dataset "infinite" or just very large, OR we can stick to the 
        # random sampling logic inside __getitem__ if we ignore idx, but that defeats the purpose of map-style.
        # Better approach for map-style:
        # Just return the slice at idx. The DataLoader with shuffle=True will handle randomness.
        # Note: Previous code did random sampling from *anywhere*. 
        # Configuring an IterableDataset is arguably better for "infinite" streaming, 
        # but Map-style with Shuffle is standard and easiest to implement for now.
        
        # However, for 'infinite' training loop style we used before, a standard epoch-based loader 
        # might require changing the loop structure (while loader: ...).
        # Let's keep the existing loop structure and make the loader infinite-cyclic or just re-create it.
        # Actually, standard practice for LLM pretraining is often IterableDataset.
        # But let's stick to simple Map-style with a large length or just wrapping it.
        
        # IMPORTANT: efficient slicing of memmap is fine.
        x = torch.from_numpy((self.data[idx:idx+self.block_size]).astype(np.int64))
        y = torch.from_numpy((self.data[idx+1:idx+1+self.block_size]).astype(np.int64))
        return x, y

@torch.no_grad()
def estimate_loss(model, ctx, train_data, val_data, dataset_config, system_config, device, eval_iters):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # manual random sampling for estimation
            ix = torch.randint(len(data) - dataset_config.block_size, (dataset_config.batch_size,))
            x = torch.stack([torch.from_numpy((data[i:i+dataset_config.block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+dataset_config.block_size]).astype(np.int64)) for i in ix])
            
            device_type = 'cuda' if 'cuda' in system_config.device else 'cpu'
            if device_type == 'cuda':
                x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
            else:
                x, y = x.to(device), y.to(device)
            
            with ctx:
                logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it, train_config):
    # 1) linear warmup for warmup_iters steps
    if it < train_config.warmup_iters:
        return train_config.learning_rate * it / train_config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > train_config.lr_decay_iters:
        return train_config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - train_config.warmup_iters) / (train_config.lr_decay_iters - train_config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return train_config.min_lr + coeff * (train_config.learning_rate - train_config.min_lr)

def parse_args():
    # default configs
    system = SystemConfig()
    dataset = DatasetConfig()
    train = TrainConfig()
    model = ModelConfig()
    
    # parse args
    for arg in sys.argv[1:]:
        if '=' not in arg:
            # assume it's a config file
            assert not arg.startswith('--')
            config_file = arg
            print(f"Overriding config with {config_file}:")
            with open(config_file) as f:
                print(f.read())
            # This part is tricky with dataclasses because exec would need to update the objects
            # For simplicity in this refactor, we'll skip arbitrary python file execution for now
            # and focus on CLI args, OR we can implement a simple parser for the file.
            # let's skip file exec for safety and clarity in this version or use a dict approach if needed.
            print("WARNING: Config file loading via exec() is disabled in this refactor. Use CLI arguments.")
        else:
            # assume it's a --key=value argument
            assert arg.startswith('--')
            key, val = arg.split('=')
            key = key[2:]
            
            try:
                attempt = literal_eval(val)
            except (ValueError, SyntaxError):
                attempt = val
                
            # try to find which config object owns this key
            found = False
            for conf in [system, dataset, train, model]:
                if hasattr(conf, key):
                    setattr(conf, key, attempt)
                    print(f"Overriding: {key} = {attempt}")
                    found = True
                    break
            if not found:
                 raise ValueError(f"Unknown config key: {key}")
                 
    return system, dataset, train, model

# -----------------------------------------------------------------------------
# Main Training Function
# -----------------------------------------------------------------------------

def main():
    system_cfg, dataset_cfg, train_cfg, model_cfg = parse_args()

    # DDP setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=system_cfg.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 
        seed_offset = ddp_rank 
        assert dataset_cfg.gradient_accumulation_steps % ddp_world_size == 0
        dataset_cfg.gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = system_cfg.device

    tokens_per_iter = dataset_cfg.gradient_accumulation_steps * ddp_world_size * dataset_cfg.batch_size * dataset_cfg.block_size
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        os.makedirs(system_cfg.out_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=system_cfg.out_dir)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True 
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[system_cfg.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Data Loader
    data_dir = os.path.join('data', dataset_cfg.dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    train_dataset = GPT2Dataset(train_data, dataset_cfg.block_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=dataset_cfg.batch_size, 
        shuffle=True, 
        num_workers=dataset_cfg.num_workers, 
        pin_memory=True,
        drop_last=True
    )
    # create an infinite iterator
    train_iter = iter(train_loader)

    # Model Init
    iter_num = 0
    best_val_loss = 1e9
    
    if dataset_cfg.init_from == 'scratch':
        print("Initializing a new model from scratch")
        model_cfg.vocab_size = 50257
        model = GPT2(model_cfg)
    elif dataset_cfg.init_from == 'resume':
        print(f"Resuming training from {system_cfg.out_dir}")
        ckpt_path = get_latest_checkpoint(system_cfg.out_dir)
        if ckpt_path is None:
            print(f"No checkpoint found in {system_cfg.out_dir}, starting from scratch")
            dataset_cfg.init_from = 'scratch'
            model = GPT2(model_cfg)
        else:
            print(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            # load model args from checkpoint
            checkpoint_model_args = checkpoint['model_args']
            # handle both dict and object (legacy compatibility)
            if isinstance(checkpoint_model_args, dict):
                for k, v in checkpoint_model_args.items():
                    if hasattr(model_cfg, k):
                        setattr(model_cfg, k, v)
            else:
                 for k, v in checkpoint_model_args.__dict__.items():
                    if hasattr(model_cfg, k):
                        setattr(model_cfg, k, v)
            
            model = GPT2(model_cfg)
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']
    elif dataset_cfg.init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {dataset_cfg.init_from}")
        override_args = dict(dropout=model_cfg.dropout)
        model = GPT2.from_pretrained(dataset_cfg.init_from, override_args)
        # update model_cfg with the loaded config
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
             setattr(model_cfg, k, getattr(model.config, k))

    # Crop block size if needed
    if dataset_cfg.block_size < model.config.block_size:
        model.crop_block_size(dataset_cfg.block_size)
        model_cfg.block_size = dataset_cfg.block_size

    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, betas=(train_cfg.beta1, train_cfg.beta2), weight_decay=train_cfg.weight_decay, fused=True)
    # extract optimizer state but don't load yet (to save memory during compile)
    optimizer_state = None
    if dataset_cfg.init_from == 'resume' and 'optimizer' in checkpoint:
        optimizer_state = checkpoint['optimizer']
    checkpoint = None
    torch.cuda.empty_cache() # free up memory from the checkpoint load

    # Compile
    if system_cfg.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)
        
        # Warmup compilation with a dummy step
        # This forces the compilation to happen NOW, before we load the optimizer state
        # preventing the memory spike of (Model + Optimizer + Compile Overhead)
        print("running warmup step for compilation...")
        dummy_x = torch.randint(0, 50257, (dataset_cfg.batch_size, dataset_cfg.block_size), device=device)
        dummy_y = torch.randint(0, 50257, (dataset_cfg.batch_size, dataset_cfg.block_size), device=device)
        with ctx:
            _, loss = model(dummy_x, dummy_y)
            loss.backward()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    # Restore optimizer state
    if optimizer_state is not None:
        print("loading optimizer state...")
        optimizer.load_state_dict(optimizer_state)
        optimizer_state = None

    # Wrap DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    raw_model = model.module if ddp else model

    # Training Loop
    t0 = time.time()
    
    while True:
        # Learning rate
        lr = get_lr(iter_num, train_cfg) if train_cfg.decay_lr else train_cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate
        if iter_num % dataset_cfg.eval_interval == 0 and master_process:
            losses = estimate_loss(model, ctx, train_data, val_data, dataset_cfg, system_config=system_cfg, device=device, eval_iters=dataset_cfg.eval_iters)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if master_process:
                writer.add_scalar("loss/train", losses['train'], iter_num)
                writer.add_scalar("loss/val", losses['val'], iter_num)
                writer.add_scalar("lr", lr, iter_num)
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
        
        if iter_num > 0 and iter_num % dataset_cfg.checkpoint_interval == 0 and master_process:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': asdict(model_cfg),
                'iter_num': iter_num,
                'best_val_loss': best_val_loss
            }
            print(f"saving checkpoint to {system_cfg.out_dir}")
            # save versioned checkpoint
            ckpt_name = f'ckpt_{iter_num}.pt'
            torch.save(checkpoint, os.path.join(system_cfg.out_dir, ckpt_name))

            # only keep last 5 checkpoints
            checkpoints = sorted([f for f in os.listdir(system_cfg.out_dir) if f.startswith('ckpt_') and f.endswith('.pt')],
                                key=lambda x: int(x.split('_')[1].split('.')[0]))
            while len(checkpoints) > 5:
                to_remove = checkpoints.pop(0) # remove oldest
                os.remove(os.path.join(system_cfg.out_dir, to_remove))
        
        if iter_num == 0 and dataset_cfg.eval_only:
            break
            
        # Forward/Backward
        for micro_step in range(dataset_cfg.gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == dataset_cfg.gradient_accumulation_steps - 1)
            
            # fetch next batch
            try:
                X, Y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                X, Y = next(train_iter)
                
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)

            with ctx:
                logits, loss = model(X, Y)
                loss = loss / dataset_cfg.gradient_accumulation_steps
            loss.backward()
            
        if train_cfg.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % dataset_cfg.log_interval == 0 and master_process:
            lossf = loss.item() * dataset_cfg.gradient_accumulation_steps
            if dt > 0:
                 mfu = -1.0 # placeholder
                 tokens_per_sec = (dataset_cfg.gradient_accumulation_steps * dataset_cfg.batch_size * dataset_cfg.block_size) / dt
                 print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, tok/sec {tokens_per_sec:.2f}")
                 writer.add_scalar("loss/step", lossf, iter_num)
                 writer.add_scalar("perf/tokens_per_sec", tokens_per_sec, iter_num)
            else:
                 print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
                 writer.add_scalar("loss/step", lossf, iter_num)
            
        iter_num += 1
        if iter_num > train_cfg.max_iters:
            break
            
    if master_process:
        writer.close()
            
    if ddp:
        destroy_process_group()

if __name__ == '__main__':
    main()
