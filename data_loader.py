
import torch
import numpy as np

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
