"""A small Llama-style Transformer: RoPE, GLU, RMSNorm and tied embeddings.

Adapted from scan_llm_bench
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

ROPE_THETA = 500_000
INIT_STD = 0.02


def glu_hidden_dim(d_model):
    """Width of the GLU's hidden layer: 8/3 of the model width, rounded up to x256."""
    return 256 * math.ceil(int(d_model * 8 / 3) / 256)


def training_flops(d_model, n_layers, n_tokens, vocab_size, seq_len):
    """FLOPs to train on `n_tokens` tokens, counted as in scan_llm_bench: the matmuls
    and attention of a forward pass, times 3 for forward + backward.
    """
    per_layer = (
        8 * d_model**2
        + 4 * seq_len * d_model
        + 6 * d_model * glu_hidden_dim(d_model)
    )
    forward_per_token = n_layers * per_layer + 2 * d_model * vocab_size
    return 3 * forward_per_token * n_tokens


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden_dim = glu_hidden_dim(dim)
        self.fc1 = nn.Linear(dim, 2 * self.hidden_dim, bias=False)
        self.fc2 = nn.Linear(self.hidden_dim, dim, bias=False)

    def forward(self, x):
        x, gate = self.fc1(x).split(self.hidden_dim, dim=-1)
        return self.fc2(F.silu(x) * gate)


def rope_frequencies(head_dim, seq_len):
    """cos/sin tables for rotary embeddings, shaped (1, seq_len, 1, head_dim // 2, 2)."""
    inv_freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), inv_freqs)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)[None, :, None]


def apply_rope(q, k, freqs):
    """Rotate consecutive pairs of q/k features by position-dependent angles."""
    qk = torch.cat([q, k], dim=2).unflatten(-1, (-1, 2)).float()
    cos, sin = freqs[..., 0], freqs[..., 1]
    rotated = torch.stack(
        [qk[..., 0] * cos - qk[..., 1] * sin, qk[..., 1] * cos + qk[..., 0] * sin], dim=-1
    ).flatten(3)
    return rotated.type_as(q).split(q.shape[2], dim=2)


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.w_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.w_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs):
        bsz, seq_len, dim = x.shape
        q, k, v = (t.view(bsz, seq_len, self.n_heads, -1) for t in self.w_qkv(x).split(dim, dim=2))
        q, k = apply_rope(q, k, freqs)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        return self.w_out(out.transpose(1, 2).reshape(bsz, seq_len, dim))


class Block(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = GLU(dim)

    def forward(self, x, freqs):
        x = x + self.attn(self.attn_norm(x), freqs)
        return x + self.mlp(self.mlp_norm(x))


class Transformer(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, n_layers, n_heads):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.register_buffer(
            "freqs", rope_frequencies(d_model // n_heads, seq_len), persistent=False
        )

        for module in self.modules():
            if isinstance(module, nn.Linear | nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
        for name, param in self.named_parameters():
            if name.endswith(("fc2.weight", "w_out.weight")):
                nn.init.normal_(param, mean=0.0, std=INIT_STD / math.sqrt(2 * n_layers))
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, tokens):
        x = self.embed_tokens(tokens)
        for layer in self.layers:
            x = layer(x, self.freqs[:, : tokens.shape[1]])
        return self.lm_head(self.out_norm(x))
