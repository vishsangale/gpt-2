import os
import sys
import argparse
import urllib.request
import numpy as np
import tiktoken
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DatasetConfig:
    name: str
    source_type: str  # 'url' or 'huggingface'
    source: str       # URL or HF dataset ID
    hf_name: Optional[str] = None # sub-dataset name for HF
    token_limit: Optional[int] = None
    val_ratio: float = 0.1

CONFIGS = {
    'tinyshakespeare': DatasetConfig(
        name='tinyshakespeare',
        source_type='url',
        source='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        val_ratio=0.1
    ),
    'fineweb': DatasetConfig(
        name='fineweb',
        source_type='huggingface',
        source='HuggingFaceFW/fineweb-edu',
        hf_name='sample-10BT',
        token_limit=100 * 1000 * 1000, # 100M tokens
        val_ratio=0.1
    )
}

def get_tokens_from_url(config: DatasetConfig, enc):
    data_dir = os.path.join(os.path.dirname(__file__), 'data', config.name)
    os.makedirs(data_dir, exist_ok=True)
    
    input_file_path = os.path.join(data_dir, 'input.txt')
    if not os.path.exists(input_file_path):
        print(f"Downloading {config.name} to {input_file_path}...")
        with open(input_file_path, 'w') as f:
            with urllib.request.urlopen(config.source) as response:
                f.write(response.read().decode())

    print(f"Reading and tokenizing {config.name}...")
    with open(input_file_path, 'r') as f:
        data = f.read()
    
    # TinyShakespeare is small enough to tokenize all at once
    tokens = enc.encode_ordinary(data)
    return tokens

def get_tokens_from_hf(config: DatasetConfig, enc):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library is required for huggingface datasets. pip install datasets")
        return []

    print(f"Loading {config.name} from HuggingFace (targeting {config.token_limit:,} tokens)...")
    dataset = load_dataset(config.source, name=config.hf_name, split="train", streaming=True)
    
    eot = enc._special_tokens['<|endoftext|>']
    all_tokens = []
    
    pbar = tqdm(total=config.token_limit, desc="Tokenizing")
    
    for doc in dataset:
        text = doc['text']
        tokens = enc.encode_ordinary(text)
        tokens.append(eot)
        all_tokens.extend(tokens)
        
        pbar.update(len(tokens))
        if len(all_tokens) >= config.token_limit:
            break
            
    pbar.close()
    return all_tokens[:config.token_limit]

def prepare_dataset(config: DatasetConfig):
    """Unified preparation function."""
    enc = tiktoken.get_encoding("gpt2")
    
    if config.source_type == 'url':
        train_tokens = get_tokens_from_url(config, enc)
        # Split logic for simple list/array
        req_token_count = len(train_tokens)
        split_idx = int(req_token_count * (1 - config.val_ratio))
        val_tokens = train_tokens[split_idx:]
        train_tokens = train_tokens[:split_idx]
        
    elif config.source_type == 'huggingface':
        # For HF we might want to split differently or stream, but here we gather all first for simplicity
        # matching the requested 'single function' style for data processing
        all_tokens = get_tokens_from_hf(config, enc)
        split_idx = int(len(all_tokens) * (1 - config.val_ratio))
        train_tokens = all_tokens[:split_idx]
        val_tokens = all_tokens[split_idx:]
        
    else:
        raise ValueError(f"Unknown source_type: {config.source_type}")

    print(f"train has {len(train_tokens):,} tokens")
    print(f"val has {len(val_tokens):,} tokens")

    data_dir = os.path.join(os.path.dirname(__file__), 'data', config.name)
    os.makedirs(data_dir, exist_ok=True)

    # export to bin files
    train_ids = np.array(train_tokens, dtype=np.uint16)
    val_ids = np.array(val_tokens, dtype=np.uint16)
    train_ids.tofile(os.path.join(data_dir, 'train.bin'))
    val_ids.tofile(os.path.join(data_dir, 'val.bin'))
    print(f"Saved binary files to {data_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare datasets for GPT-2 training.')
    parser.add_argument('--dataset', type=str, default='tinyshakespeare', choices=CONFIGS.keys(),
                        help='Which dataset to prepare')
    
    args = parser.parse_args()
    
    if args.dataset in CONFIGS:
        config = CONFIGS[args.dataset]
        prepare_dataset(config)
        # Flush buffers and force exit to avoid PyGILState_Release errors with some libraries
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    else:
        print(f"Unknown dataset: {args.dataset}")
