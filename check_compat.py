import torch
from model import GPT2
from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    vocab_size: int = 50304
    block_size: int = 1024
    gradient_checkpointing: bool = True
    use_rmsnorm: bool = False
    use_rope: bool = False
    use_swiglu: bool = False

def check_compatibility(target_file='vanilla_keys.log', mode='save'):
    config = ModelConfig()
    # Ensure default config is "Vanilla"
    config.use_rope = False
    config.use_swiglu = False
    config.use_rmsnorm = False
    
    model = GPT2(config)
    keys = list(model.state_dict().keys())
    
    if mode == 'save':
        with open(target_file, 'w') as f:
            for k in keys:
                f.write(k + '\n')
        print(f"Saved {len(keys)} keys to {target_file}")
    elif mode == 'check':
        if target_file.endswith('.pt'):
            print(f"Loading checkpoint keys from {target_file}...")
            ckpt = torch.load(target_file, map_location='cpu')
            target_keys = list(ckpt['model'].keys())
            # filter _orig_mod prefix if present
            target_keys = [k.replace('_orig_mod.', '') for k in target_keys]
        else:
            with open(target_file, 'r') as f:
                target_keys = [line.strip() for line in f]
            
        if keys != target_keys:
            print("COMPATIBILITY FAIL!")
            print(f"Key order or content mismatch.")
            # detailed diff
            for i, (k1, k2) in enumerate(zip(keys, target_keys)):
                if k1 != k2:
                    print(f"First mismatch at index {i}:")
                    print(f"  Model: {k1}")
                    print(f"  Target: {k2}")
                    break
            if len(keys) != len(target_keys):
                print(f"Length mismatch: {len(keys)} vs {len(target_keys)}")
            exit(1)
        else:
            print("COMPATIBILITY PASS: Keys match exactly in order.")


if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['save', 'check'], default='check', nargs='?')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint file to verify against')
    args = parser.parse_args()
    
    target = args.checkpoint if args.checkpoint else 'vanilla_keys.log'
    check_compatibility(target_file=target, mode=args.mode)
