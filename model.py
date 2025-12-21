import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint

# -----------------------------------------------------------------------------
# Modernization Modules
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.w2 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.dropout(self.c_proj(hidden))

def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis.detach()

def apply_rotary_emb(xq, xk, freqs_cis):
    # Ensure Q/K and freqs_cis are compatible dtypes for complex multiplication
    # We generally want to perform rotation in float32 for precision, then cast back
    
    # Reshape Q/K to expose rotation pairs
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Match sequence length
    freqs_cis = freqs_cis[:xq_.shape[2]] 
    
    # Ensure freqs_cis is on the same device as xq_
    freqs_cis = freqs_cis.to(xq_.device)

    # Broadcast to batch and num_heads (1, 1, T, hs/2)
    # Dimensions: xq_: (B, nh, T, hs/2)
    freqs_cis = freqs_cis.view(1, 1, freqs_cis.shape[0], freqs_cis.shape[1])
    
    # Apply rotation
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    # Cast back to original dtype
    return xq_out.type_as(xq), xk_out.type_as(xk)

# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    
    def forward(self, x, freqs_cis=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if getattr(self, 'use_rope', False) and freqs_cis is not None:
             q, k = apply_rotary_emb(q, k, freqs_cis)

        # flash attention
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        if getattr(config, 'use_swiglu', False):
            self.swiglu = SwiGLU(config)
        else:
            self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)
            self.gelu = nn.GELU()

    def forward(self, x):
        if hasattr(self, 'swiglu'):
            return self.swiglu(x)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        if getattr(config, 'use_rmsnorm', False):
            self.ln_1 = RMSNorm(config.n_embd)
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd)
            
        self.attn = CausalSelfAttention(config)
        self.attn.use_rope = getattr(config, 'use_rope', False)

        if getattr(config, 'use_rmsnorm', False):
            self.ln_2 = RMSNorm(config.n_embd)
        else:
            self.ln_2 = nn.LayerNorm(config.n_embd)
        
        self.mlp = MLP(config)

    def forward(self, x, freqs_cis=None):
        x = x + self.attn(self.ln_1(x), freqs_cis=freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if getattr(config, 'use_rope', False):
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = RMSNorm(config.n_embd) if getattr(config, 'use_rmsnorm', False) else nn.LayerNorm(config.n_embd),
            ))
            # No wpe for RoPE typically
            
            # Precompute freqs_cis
            head_dim = config.n_embd // config.n_head
            self.register_buffer('freqs_cis', precompute_freqs_cis(head_dim, config.block_size * 2))
        else:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = RMSNorm(config.n_embd) if getattr(config, 'use_rmsnorm', False) else nn.LayerNorm(config.n_embd),
            ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed a dict that contains keys that are not in the module..."
        # ~s o we disable it for now, but in the original paper weights are tied.~
        # ACTUALLY, weight tying is standard for GPT models.
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        
        if getattr(self.config, 'use_rope', False):
            # No wpe
            x = self.transformer.drop(tok_emb)
            # freqs_cis is registered as a buffer, so it is already on the correct device.
            # We just need to slice it to the current sequence length.
            freqs_cis = self.freqs_cis[:t]
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
            freqs_cis = None

        for block in self.transformer.h:
            if self.config.gradient_checkpointing:
                # When passing extra args to checkpoint, they are passed to the module
                if getattr(self.config, 'use_rope', False):
                     x = torch.utils.checkpoint.checkpoint(block, x, freqs_cis, use_reentrant=False)
                else:
                     x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                 if getattr(self.config, 'use_rope', False):
                     x = block(x, freqs_cis)
                 else:
                     x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                # In case we re-add bias buffer later; currently unused
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
    
    @classmethod
    def from_pretrained(cls, model_type):
        """
        Loads pretrained GPT-2 weights from huggingface
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config['bias'] = True # always True for GPT model checkpoints

        # create a from-scratch initialized minGPT model
        # using a simple namespace or class for config
        class Config:
            pass
        conf = Config()
        for k,v in config.items():
            setattr(conf, k, v)
        conf.dropout = 0.0 # meaningless for loading pretrained weights
        
        model = GPT2(conf)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
