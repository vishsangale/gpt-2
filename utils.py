
import os
import math
import torch
import numpy as np

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
