# How I Trained GPT-2 from Scratch and Sped it up by 3x

*By Vish Sangale*

**Code**: [https://github.com/vishsangale/gpt-2](https://github.com/vishsangale/gpt-2)

---

Training a Large Language Model (LLM) is often seen as a dark art reserved for those with clusters of H100s. But I wanted to see how far I could push a single consumer GPU (RTX 5080) to train a 124M parameter model—the original **[GPT-2 Small](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)**—from scratch.

Here is the story of how I started at a crawling **29,000 tokens/second** and optimized my way to over **92,000 tokens/second**, seeing a **3.1x speedup** through pure engineering.

## The Setup

I implemented the GPT-2 architecture in pure PyTorch:
- **12 Layers, 12 Heads, 768 Embedding Dim**
- **Tokenizer**: Tiktoken (OpenAI's BPE)
- **Dataset**: TinyShakespeare (A classic character modeling benchmark)

The goal was simple: get the loss down and the text generation up.

## The Bottleneck: "CUDA Out of Memory" 🛑

My first attempt was humble. I set the batch size to 12.
**Crash.** `CUDA Out of Memory`.

I lowered it to 8.
**Crash.**

I finally got it running at `batch_size=4`. It was painfully slow. The GPU utilization was spiky, and I was processing only ~29k tokens/sec. At this rate, training would take forever.

## The Optimization Journey 🚀

I didn't want to just "wait longer." I wanted to engineering my way out of the bottleneck. Here is what worked.

### 1. Flash Attention 2 ⚡
The standard attention mechanism computes an $N \times N$ matrix. For context length 1024, that's manageable, but it still eats massive VRAM.
I switched to `torch.nn.functional.scaled_dot_product_attention`, which uses **Flash Attention 2**.
- **Result**: Memory usage plummeted. I could instantly double my batch size from 4 to 8.

### 2. Feeding the Beast (Data Loading) 🍽️
I noticed `nvidia-smi` showed the GPU dropping to 0% utilization periodically. My Python script was building batches on the CPU *synchronously* while the GPU waited.
I wrote a custom `GPT2Dataset` and used Pytorch's `DataLoader` with `num_workers=4` and `pin_memory=True`.
- **Result**: The GPU stayed pinned at 99%. Throughput jumped to **63k tokens/sec**.

### 3. Fused AdamW 🔥
The optimizer updates 124 million parameters. Doing this in a Python for-loop is slow.
I flipped the `fused=True` flag in `torch.optim.AdamW`.
- **Result**: The entire optimizer step now runs as a single CUDA kernel.

### 4. `torch.compile` (The Final Boss) 🛠️
PyTorch 2.0 introduced `torch.compile`. It proved tricky—I had to debug missing system headers (`Python.h`, `cuda.h`) and install development toolkits.
But once it worked?
- **Result**: It fused operations and reduced overhead so much that I could **double the batch size again to 16**.

## The Result 📈

![Training Loss](assets/loss_curve.png)

We went from a fragile, slow script to a highly optimized training pipeline.

| Optimization Stage | Throughput (tok/sec) | Speedup |
|--------------------|----------------------|---------|
| Baseline (Batch 4) | 29,000 | 1.0x |
| + DataLoader | 63,500 | 2.2x |
| + Flash Attn | 67,500 | 2.3x |
| **+ Compile (Batch 16)** | **92,000+** | **3.1x** |

## Takeaway

You don't always need more GPUs. Sometimes you just need better engineering.

Check out the full code and try it yourself:
👉 **[vishsangale/gpt-2](https://github.com/vishsangale/gpt-2)**
