"""Vendored from PleIAs/CommonLingua (revision 43fe88d75e94b11283b66daccbbe4a73e7bc1361) under Apache 2.0.

Upstream: https://huggingface.co/PleIAs/CommonLingua/blob/main/model.py

ByteHybrid: byte-level language identification (CommonLingua v7.2.1).

Operates directly on raw UTF-8 bytes - no tokenizer required:

    raw bytes -> byte-embed + trigram-hash-embed (summed)
              -> 3 x depthwise Conv1D (k=15)
              -> 1 x bidirectional attention (RoPE, 4 heads)
              -> masked mean-pool
              -> classification head (334 logits)

The shipped checkpoint uses the ``base_ngram`` config: d_model=256, 4096 trigram
hash buckets x 64 dim, max_len=512 bytes. Total parameters ~ 2.35 M.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ByteNgramEmbed(nn.Module):
    """Rolling polynomial hash of byte trigrams into a fixed-size table.

    Hash collisions act as regularisation; the small table (4096 x 64)
    keeps parameter count bounded under arbitrary input distributions.
    """

    def __init__(self, num_buckets=4096, embed_dim=64, n=3):
        super().__init__()
        self.n = n
        self.num_buckets = num_buckets
        self.embed = nn.Embedding(num_buckets, embed_dim)

    def forward(self, byte_ids):
        B, T = byte_ids.shape
        clamped = byte_ids.clamp(max=255)
        padded = F.pad(clamped, (0, self.n - 1), value=0)
        h = torch.zeros(B, T, dtype=torch.long, device=byte_ids.device)
        for i in range(self.n):
            h = h * 257 + padded[:, i:i + T]
        return self.embed(h % self.num_buckets)


class ByteConvBlock(nn.Module):
    """Causal depthwise Conv1D + SwiGLU FFN, with residual + layernorm."""

    def __init__(self, d_model, kernel_size=15, expand=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model)
        self.norm2 = nn.LayerNorm(d_model)
        ffn = d_model * expand
        self.ffn_gate = nn.Linear(d_model, ffn, bias=False)
        self.ffn_up = nn.Linear(d_model, ffn, bias=False)
        self.ffn_down = nn.Linear(ffn, d_model, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm1(x).transpose(1, 2)
        x = F.pad(x, (self.pad, 0))
        x = F.silu(self.conv(x)).transpose(1, 2)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn_down(F.silu(self.ffn_gate(x)) * self.ffn_up(x))
        return residual + x


def _rope(q, k):
    head_dim = q.shape[-1]
    seq_len = q.shape[-2]
    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=q.device).float() / head_dim))
    t = torch.arange(seq_len, device=q.device)
    a = torch.outer(t, freqs)
    cos = a.cos().to(q.dtype)
    sin = a.sin().to(q.dtype)

    def rot(x):
        x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    return rot(q), rot(k)


class ByteAttnBlock(nn.Module):
    """Bidirectional self-attention with RoPE + SwiGLU FFN."""

    def __init__(self, d_model, n_heads=4, expand=2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        ffn = d_model * expand
        self.ffn_gate = nn.Linear(d_model, ffn, bias=False)
        self.ffn_up = nn.Linear(d_model, ffn, bias=False)
        self.ffn_down = nn.Linear(ffn, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        residual = x
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
        q, k = _rope(q, k)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        x = residual + self.out_proj(out)

        residual = x
        h = self.norm2(x)
        h = self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h))
        return residual + h


class ByteHybrid(nn.Module):
    """Byte-level classifier with optional trigram-hash augmentation."""

    def __init__(
        self,
        num_classes,
        d_model=256,
        n_conv=3,
        n_attn=1,
        n_heads=4,
        ffn_expand=2,
        max_len=512,
        conv_kernel=15,
        ngram_buckets=0,
        ngram_dim=64,
    ):
        super().__init__()
        self.max_len = max_len

        # Byte values 0-255 plus index 256 = padding token
        self.embed = nn.Embedding(257, d_model, padding_idx=256)

        self.ngram_embed = None
        if ngram_buckets > 0:
            self.ngram_embed = ByteNgramEmbed(ngram_buckets, ngram_dim, n=3)
            self.ngram_proj = nn.Linear(ngram_dim, d_model, bias=False)

        self.conv_layers = nn.ModuleList(
            [ByteConvBlock(d_model, conv_kernel, ffn_expand) for _ in range(n_conv)]
        )
        self.attn_layers = nn.ModuleList(
            [ByteAttnBlock(d_model, n_heads, ffn_expand) for _ in range(n_attn)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, byte_ids):
        pad_mask = byte_ids != 256
        x = self.embed(byte_ids)
        if self.ngram_embed is not None:
            x = x + self.ngram_proj(self.ngram_embed(byte_ids))
        for layer in self.conv_layers:
            x = layer(x)
        for layer in self.attn_layers:
            x = layer(x)
        x = self.final_norm(x)
        mask = pad_mask.unsqueeze(-1).to(x.dtype)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(x)


# Single shipped configuration. The checkpoint encodes which config it was
# trained with under the "config" key.
CONFIGS = {
    "base_ngram": dict(
        d_model=256, n_conv=3, n_attn=1, n_heads=4, conv_kernel=15,
        ngram_buckets=4096, ngram_dim=64,
    ),
}
