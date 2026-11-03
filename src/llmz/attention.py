"""attention 相關（第 4 章第 7～12 集）：RoPE、causal mask、scaled dot-product attention、multi-head self-attention。"""

import math

import torch
from torch import Tensor, nn

from llmz.functional import softmax
from llmz.layers import Linear


# ANCHOR: rope
class RoPE(nn.Module):
    """旋轉位置編碼：把向量兩兩一組當成平面上的點，第 i 個位置轉 i 倍的角度。"""

    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        freqs = theta ** (-torch.arange(0, d_k, 2) / d_k)  # (d_k/2,)：每一組轉速不同
        angles = torch.arange(max_seq_len)[:, None] * freqs[None, :]  # (max_seq_len, d_k/2)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """x 形狀 (..., T, d_k)，positions 形狀 (T,)。"""
        cos = self.cos[positions]  # (T, d_k/2)
        sin = self.sin[positions]
        x1, x2 = x[..., 0::2], x[..., 1::2]  # 相鄰兩個元素一組
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack((out1, out2), dim=-1).flatten(-2)  # 交錯放回去
# ANCHOR_END: rope


# ANCHOR: causal_mask
def causal_mask(seq_len: int, device: torch.device | None = None) -> Tensor:
    """(T, T) 的布林矩陣：第 i 列只有 j <= i 的位置是 True。"""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
# ANCHOR_END: causal_mask


# ANCHOR: sdpa
def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    """q (..., Tq, d)、k (..., Tk, d)、v (..., Tk, dv)；mask 為 False 的位置不准看。"""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (..., Tq, Tk)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return softmax(scores, dim=-1) @ v
# ANCHOR_END: sdpa


# ANCHOR: mha
class MultiHeadSelfAttention(nn.Module):
    """多頭 causal self-attention。輸入輸出都是 (B, T, d_model)。"""

    def __init__(self, d_model: int, n_heads: int, rope: RoPE | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)
        self.rope = rope

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        # (B, T, d_model) -> (B, T, H, d_head) -> (B, H, T, d_head)：每個 head 各自做 attention
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        if self.rope is not None:
            positions = torch.arange(T, device=x.device)
            q = self.rope(q, positions)
            k = self.rope(k, positions)
        out = scaled_dot_product_attention(q, k, v, causal_mask(T, x.device))
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)  # 把 head 接回去
        return self.out_proj(out)
# ANCHOR_END: mha
