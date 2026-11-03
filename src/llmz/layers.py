"""Transformer 的基本零件（第 4 章第 2～5 集）：Linear、Embedding、RMSNorm、SiLU、SwiGLU。

全書不用 torch.nn 的現成層，這裡是我們自己寫的版本；tests/test_layers.py 用 torch 內建函式驗證。
"""

import torch
from torch import Tensor, nn


# ANCHOR: linear
class Linear(nn.Module):
    """y = x @ W.T，沒有偏差項。weight 形狀 (out_features, in_features)。"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        std = (2 / (in_features + out_features)) ** 0.5
        w = torch.randn(out_features, in_features) * std
        self.weight = nn.Parameter(w.clamp(-3 * std, 3 * std))  # 截掉超過三個標準差的離群值

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T
# ANCHOR_END: linear


# ANCHOR: embedding
class Embedding(nn.Module):
    """查表：每個 token id 對應一列向量。weight 形狀 (num_embeddings, embedding_dim)。"""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        w = torch.randn(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(w.clamp(-3, 3))

    def forward(self, ids: Tensor) -> Tensor:
        return self.weight[ids]
# ANCHOR_END: embedding


# ANCHOR: rmsnorm
class RMSNorm(nn.Module):
    """把最後一個維度除以它的均方根，再乘上可學的縮放 weight。"""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()  # 平方和在 float32 算，避免低精度溢位
        rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(dtype)
# ANCHOR_END: rmsnorm


# ANCHOR: silu
def silu(x: Tensor) -> Tensor:
    """SiLU 啟用函式：x * sigmoid(x)。"""
    return x * torch.sigmoid(x)
# ANCHOR_END: silu


# ANCHOR: swiglu
class SwiGLU(nn.Module):
    """前饋網路：w2(silu(w1 x) * (w3 x))。"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w3 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))
# ANCHOR_END: swiglu


# ANCHOR: siluffn
class SiLUFFN(nn.Module):
    """沒有 gate 的前饋網路：w2(silu(w1 x))。第 6 章消融實驗用。"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))
# ANCHOR_END: siluffn


class Identity(nn.Module):
    """什麼都不做。第 6 章消融實驗拿掉 RMSNorm 時用。"""

    def forward(self, x: Tensor) -> Tensor:
        return x
