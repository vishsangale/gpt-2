
import sys
from dataclasses import dataclass
from ast import literal_eval
import torch

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
    use_rmsnorm: bool = True
    use_rope: bool = True
    use_swiglu: bool = True

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
