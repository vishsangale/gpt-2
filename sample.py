"""
Sample from a trained model
"""
import os
import argparse
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPT2

# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = 'out'
start = "\n"
num_samples = 3
max_new_tokens = 200
temperature = 0.8
top_k = 200
seed = 1337
device = 'cuda'
compile = False

parser = argparse.ArgumentParser(description='Sample from a trained model')
parser.add_argument('--init_from', type=str, default='resume', help="either 'resume' or 'gpt2' (e.g. gpt2-xl)")
parser.add_argument('--out_dir', type=str, default='out', help="ignored if init_from is not 'resume'")
parser.add_argument('--start', type=str, default="\n", help="start text or FILE:prompt.txt")
parser.add_argument('--num_samples', type=int, default=3, help="number of samples to draw")
parser.add_argument('--max_new_tokens', type=int, default=200, help="number of tokens generated in each sample")
parser.add_argument('--temperature', type=float, default=0.8, help="1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions")
parser.add_argument('--top_k', type=int, default=200, help="retain only the top_k most likely tokens, clamp others to have -inf probability")
parser.add_argument('--seed', type=int, default=1337, help="seed for random number generators")
parser.add_argument('--device', type=str, default='cuda', help="examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.")
parser.add_argument('--compile', action='store_true', help="use PyTorch 2.0 to compile the model to be faster")
args = parser.parse_args()

init_from = args.init_from
out_dir = args.out_dir
start = args.start
num_samples = args.num_samples
max_new_tokens = args.max_new_tokens
temperature = args.temperature
top_k = args.top_k
seed = args.seed
device = args.device
compile = args.compile

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
# -----------------------------------------------------------------------------

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

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = get_latest_checkpoint(out_dir)
    if ckpt_path is None:
        print(f"No checkpoint found in {out_dir}, cannot resume.")
        exit(1)
    
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = checkpoint['model_args']
    # create the model
    class SimpleConfig:
        def __init__(self, **kwargs):
            for k,v in kwargs.items():
                setattr(self, k, v)
    
    # handle dict vs object
    if isinstance(gptconf, dict):
         conf = SimpleConfig(**gptconf)
    else:
         conf = SimpleConfig(**gptconf.__dict__)
         
    model = GPT2(conf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT2.from_pretrained(init_from)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join(os.path.dirname(__file__), 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
