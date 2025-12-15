# Training GPT-2 from Scratch: A Journey in Optimization

Reproducing the 124M parameter GPT-2 model is a rite of passage for LLM engineers. It sounds simple—implement the Transformer block, adding some embeddings, and hit train—but doing it *efficiently* is where the real engineering happens.

This project is a clean, optimized PyTorch implementation of GPT-2, trained on the TinyShakespeare dataset. We started with a basic implementation implementation and iteratively optimized it to achieve a **3.1x speedup** (from ~29k to ~92k tokens/sec) on a single GPU.

![Training Loss](assets/loss_curve.png)

---

## 🏗️ The Architecture

The core implementation (`model.py`) mirrors the original OpenAI 124M parameter model:
- **12 Layers**, **12 Heads**, **768 Embedding Dimension**.
- **Causal Self-Attention**: The heart of the model, masking future tokens.
- **Learned Positional Embeddings**: Standard GPT-2 style embedding (up to 1024 context length).
- **Weight Tying**: The embedding layer weights are shared with the final output projection head (`lm_head`).

We used **tiktoken** for the tokenizer, matching the standard GPT-2 vocabulary.

## 🚧 The Challenge: OOM and Slow Training

Our initial naive implementation ran into immediate bottlenecks on a 16GB VRAM GPU.
- **OOM Errors**: Trying to run a `batch_size` of 12 or even 8 caused CUDA Out of Memory errors.
- **Slow Throughput**: With a forced small batch size (4) and no optimizations, we were training at roughly **29,000 tokens/sec**.

We needed to do better.

## 🚀 The Usage Performance

We implemented a series of optimizations to unlock the hardware's potential:

### 1. Flash Attention 2 ⚡
The biggest win. We replaced the manual `(Q @ K.T) * scale -> Softmax -> @ V` implementation with `torch.nn.functional.scaled_dot_product_attention`. 
- **Impact**: drastically reduced memory usage (no $N^2$ attention matrix materialized) and improved speed.
- **Result**: Allowed us to increase batch size from 4 to 8, and later 16.

### 2. High-Performance Data Loading 🧵
We discovered our GPU was idling while the CPU prepared batches in the main thread.
- **Fix**: Implemented a custom `GPT2Dataset` and used `torch.utils.data.DataLoader` with `num_workers=4` and `pin_memory=True`.
- **Result**: Asynchronous data prefetching, keeping the GPU fed 100% of the time.

### 3. Fused AdamW 🔥
Standard PyTorch optimizers iterate through parameters in Python loops.
- **Fix**: Used `torch.optim.AdamW(..., fused=True)`.
- **Result**: Run the entire optimization step in a single fused CUDA kernel.

### 4. Vocabulary Padding 📏
Standard GPT-2 vocab is 50,257. This is an odd number that misaligns with GPU tile sizes.
- **Fix**: Padded the vocabulary to **50,304** (nearest multiple of 64).
- **Result**: More efficient matrix multiplications in the final layer.

### 5. `torch.compile` 🛠️
The final boss. We enabled PyTorch 2.0's graph compilation.
- **Hurdle**: Initially failed due to missing `Python.h` and `cuda.h` headers on the system.
- **Fix**: Installed `python3-dev` and `nvidia-cuda-toolkit`.
- **Result**: Another ~1.5x speedup by fusing elemental ops and reducing Python overhead.

## 📊 Final Results

| Configuration | Batch Size | Speed (tokens/sec) | Improvement |
|--------------|------------|--------------------|-------------|
| Initial | 4 | ~29,000 | 1.0x |
| + DataLoader & Optims | 4 | ~63,500 | 2.2x |
| + Flash Attn (Batch size inc.) | 8 | ~67,500 | 2.3x |
| **+ torch.compile** | **16** | **~92,000+** | **3.1x** |

We achieved a **3.1x speedup** overall.

## 💻 How to Run

### Setup
```bash
# Install dependencies
pip install torch numpy transformers datasets tiktoken tensorboard

# Install system headers for torch.compile (Ubuntu)
sudo apt install python3-dev nvidia-cuda-toolkit
```

### 1. Prepare Data
Download and tokenize the dataset:
```bash
python3 prepare_data.py
```

### 2. Train
Run the fully optimized training script:
```bash
python3 train.py \
    --batch_size=16 \
    --gradient_accumulation_steps=30 \
    --max_iters=100000 \
    --compile=True \
    --always_save_checkpoint=True
```

Monitor progress with TensorBoard:
```bash
tensorboard --logdir=out
```

### 3. Generate Text
Sample from your trained model:
```bash
python3 sample.py --out_dir=out --start="To be or not to be"
```

## License
MIT
