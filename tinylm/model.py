"""A decoder-only transformer (GPT-style) written from scratch in PyTorch.

Data flow for a batch of token ids with shape (B, T):

    ids -> token embedding                      (B, T, dim)
        -> N x Block                            (B, T, dim)
        -> final RMSNorm
        -> output projection (tied to the embedding matrix)  (B, T, vocab)  = logits

Each Block is "pre-norm" with two residual branches:

    x = x + Attention(RMSNorm(x))      # tokens look at earlier tokens
    x = x + MLP(RMSNorm(x))            # each token is processed independently

Design choices (all standard in current small LMs, each explained where it is used):
  * RMSNorm instead of LayerNorm: cheaper, works as well.
  * Rotary position embeddings (RoPE): position is injected by rotating queries and keys, so
    attention scores depend on the RELATIVE distance between tokens. No learned position table.
  * SwiGLU MLP: a gated feed-forward layer, usually better than a plain GELU MLP at equal size.
  * No bias terms anywhere, and the output layer shares weights with the embedding ("weight tying"),
    which saves vocab*dim parameters. That matters a lot at 2M parameters.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """x / rms(x) * weight, where rms(x) = sqrt(mean(x^2)). Keeps activations at a stable scale.
    Computed in float32 for numerical safety even when the rest runs in float16."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def build_rope_cache(head_dim, context_len, theta):
    """Precompute cos/sin tables for rotary embeddings.

    Each pair of channels (i, i + head_dim/2) is treated as a 2D point and rotated by an angle
    position * freq_i, where freq_i = theta^(-2i/head_dim). Low channels rotate fast (fine position
    information), high channels rotate slowly (coarse position information).
    """
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half).float() / half))     # (half,)
    angles = torch.outer(torch.arange(context_len).float(), freqs)      # (T, half)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """x: (B, H, T, head_dim). Rotates channel pairs by the position-dependent angle.
    Because a dot product of two rotated vectors depends only on the difference of their angles,
    q.k then depends only on the relative position of the two tokens."""
    T = x.shape[2]
    cos = cos[:T][None, None].to(x.dtype)   # (1, 1, T, half)
    sin = sin[:T][None, None].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    For each token we compute a query q (what am I looking for), a key k (what do I contain) and a
    value v (what I pass on). Attention weights = softmax(q.k / sqrt(head_dim)) over EARLIER tokens
    only (the causal mask), and the output is the weighted sum of values. Several heads do this in
    parallel on different slices of the channels, so different heads can learn different relations.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.dim // cfg.n_head
        self.attn_impl = cfg.attn_impl
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)   # one matmul gives q, k and v
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)      # mixes the heads back together

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)  # each (B, H, T, hd)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        p = self.dropout if self.training else 0.0
        if self.attn_impl == "sdpa":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        else:
            # The same computation spelled out. Used in tests to check the fast path.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)          # (B, H, T, T)
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()      # True where attention is allowed
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = F.softmax(scores.float(), dim=-1).type_as(q)
            weights = F.dropout(weights, p=p)
            y = weights @ v                                                         # (B, H, T, hd)
        y = y.transpose(1, 2).reshape(B, T, C)   # concatenate heads
        return self.proj(y)


class SwiGLU(nn.Module):
    """Feed-forward layer: down( silu(gate(x)) * up(x) ). The silu(gate) part acts as a learned,
    per-channel on/off switch for the 'up' features."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(cfg.mlp_mult * cfg.dim)
        hidden = (hidden + 31) // 32 * 32   # round up to a multiple of 32 (friendly for GPU kernels)
        self.hidden = hidden
        self.gate = nn.Linear(cfg.dim, hidden, bias=False)
        self.up = nn.Linear(cfg.dim, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.dim, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.dim)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # weight tying: one matrix is both input and output embedding
        cos, sin = build_rope_cache(cfg.dim // cfg.n_head, cfg.context_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # Scale the layers that write into the residual stream by 1/sqrt(2*n_layer) so the stream's
        # variance does not grow with depth (each block adds two such writes).
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """idx: (B, T) token ids. Returns (logits (B, T, vocab), loss or None).
        Loss is next-token cross entropy: logits at position t are scored against token t+1,
        which the caller supplies as `targets` (already shifted)."""
        B, T = idx.shape
        if T > self.cfg.context_len:
            raise ValueError(f"sequence length {T} exceeds context_len {self.cfg.context_len}")
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def count_parameters(model):
    """Parameter counts. nn.Module.parameters() already counts the tied matrix once."""
    total = sum(p.numel() for p in model.parameters())
    emb = model.tok_emb.weight.numel()
    blocks = sum(p.numel() for p in model.blocks.parameters())
    return {"total": total, "embedding(tied with output)": emb, "transformer blocks": blocks,
            "final norm": model.norm_f.weight.numel(), "non-embedding": total - emb,
            "per block": blocks // len(model.blocks)}
