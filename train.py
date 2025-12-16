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
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter

from model import GPT2
import config
import utils
import data_loader

# -----------------------------------------------------------------------------
# Main Training Function
# -----------------------------------------------------------------------------

def main():
    system_cfg, dataset_cfg, train_cfg, model_cfg = config.parse_args()

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

    train_dataset = data_loader.GPT2Dataset(train_data, dataset_cfg.block_size)
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
        ckpt_path = utils.get_latest_checkpoint(system_cfg.out_dir)
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
        lr = utils.get_lr(iter_num, train_cfg) if train_cfg.decay_lr else train_cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate
        if iter_num % dataset_cfg.eval_interval == 0 and master_process:
            losses = utils.estimate_loss(model, ctx, train_data, val_data, dataset_config=dataset_cfg, system_config=system_cfg, device=device, eval_iters=dataset_cfg.eval_iters)
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
