# Modernizing GPT-2: A Journey from 2019 to 2025

*How we injected state-of-the-art features into a classic architecture and what we learned about scaling.*

## Introduction
GPT-2 is a legendary model, effectively the "Hello World" of modern LLMs thanks to Andrej Karpathy's `nanoGPT`. But in the fast-moving world of AI, 2019 might as well be ancient history.

We set out to answer a simple question: **What happens if we take the classic GPT-2 architecture and inject the architectural improvements that power today's leading models like Llama 3?**

This post details our journey of implementing **RoPE**, **RMSNorm**, and **SwiGLU** into GPT-2, the backward-compatibility challenges we solved, and the surprising results we found when testing on datasets of different sizes.

---

## 1. The Upgrades: Why We Made Them

We focused on three key modernizations that have become standard in post-2023 LLMs.

### 🔄 Rotary Positional Embeddings (RoPE)
**The Old Way:** Standard GPT-2 uses learned absolute positional embeddings. The model learns a unique vector for position 0, position 1, etc. This doesn't scale well to longer contexts and fails to capture the *relative* distance between tokens effectively.

**The Upgrade:** We replaced this with **RoPE**. Instead of adding a vector, we *rotate* the query and key vectors in the attention mechanism based on their position. This allows the model to naturally understand "token A is 5 steps before token B" regardless of where they appear in the sequence.

### 📐 RMSNorm (Root Mean Square Normalization)
**The Old Way:** LayerNorm. It centers and scales the input.
**The Upgrade:** **RMSNorm**. It skips the centering step and only re-scales. It's computationally simpler and, empirically, often leads to more stable training at scale. It’s a small simplification that has helped models like Llama scale to massive sizes.

### 🧠 SwiGLU Activation
**The Old Way:** GeLU (Gaussian Error Linear Unit).
**The Upgrade:** **SwiGLU** (Swish Gated Linear Unit). This is one of the most impactful changes. It essentially gives the Feed-Forward Network (FFN) a "gate" implementation, increasing the model's capacity and expressivity. It requires slightly more parameters (due to the extra gate projection), but the performance capabilities per parameter are generally higher.

---

## 2. Challenge: The Backward Compatibility Trap
We didn't just want a new model; we wanted a unified codebase. We needed to ensure that we could still load old, vanilla GPT-2 checkpoints.

We implemented strict conditional logic in our `Block` and `MLP` classes:
```python
if config.use_rmsnorm:
    self.ln1 = RMSNorm(config.n_embd)
else:
    self.ln1 = nn.LayerNorm(config.n_embd)
```
We even wrote a `check_compat.py` script to verify that when these flags are disabled, the state dictionary keys match the vanilla model **exactly**. This allowed us to modernize the engine without breaking the car.

---

## 3. Experiment 1: The Overfitting Trap (Tiny Shakespeare)
Our first test was on the classic **Tiny Shakespeare** dataset. We anticipated the modernized model would crush the baseline.

**The Result:** It did... and it didn't.

| Metric | Vanilla GPT-2 | Modernized GPT-2 |
| :--- | :--- | :--- |
| **Train Loss** | 0.0707 | **0.0296** (Lower is better) |
| **Val Loss** | 7.7550 | **8.4237** (Higher?!) |

**What happened?**
The modernized model was *too good* for the data. The SwiGLU layers and improved attention (RoPE) gave the model significantly more expressivity. On a tiny dataset like Shakespeare, it didn't just learn the patterns; it memorized the text.

It learned faster (training loss plummeted), but it failed to generalize better (validation loss spiked). This was a textbook case of **overfitting due to over-parameterization relative to data size**.

---

## 4. Experiment 2: The Ablation Study (Who contributed what?)
We also ran a quick ablation to see which feature contributes most to early convergence (at Step 100).

| Feature | Val Loss (Step 100) | Verdict |
| :--- | :--- | :--- |
| **Vanilla** | 7.41 | Baseline. |
| **RoPE** | **7.35** | Winner. Better position handling helps immediately. |
| **SwiGLU**| 7.35 | Close second. More parameters/capacity. |
| **RMSNorm**| 7.36 | Solid, mostly for stability. |

It turns out **RoPE** gave us the biggest bang for our buck in the early stages, confirming that better positional understanding is critical for language modeling.

---

## 5. Experiment 3: Redemption (FineWeb)
To prove the architecture works, we needed a dataset that could withstand the power of the modernized model. We switched to **FineWeb**, a high-quality, massive web dataset.

**The Result:** Clear victory.

| Metric | Vanilla GPT-2 | Modernized GPT-2 |
| :--- | :--- | :--- |
| **Val Loss (Step 2000)** | 4.4223 | **4.0407** |

On the larger dataset, the overfitting vanished. The modernized model leveraged its superior architecture to learn more generalized patterns, achieving an **8.6% improvement** in validation loss over the baseline.

---

## Conclusion
Modernizing legacy architectures isn't just about pasting in new code. It's about understanding the relationship between **model expressivity** and **data scale**.

1.  **RoPE + SwiGLU + RMSNorm** makes the model a more efficient learner.
2.  On **small data**, this efficiency manifests as overfitting.
3.  On **large data**, it manifests as superior performance and generalization.

We now have a GPT-2 codebase that is backward compatible with 2019 checkpoints but capable of 2025 performance when fed the right data.

**Next Steps:** Scaling up further and investigating long-context performance with RoPE.
