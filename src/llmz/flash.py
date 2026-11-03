"""FlashAttention 的純 PyTorch 版（第 8 章）：分塊計算、online softmax、backward 時重算機率。

只為了看懂演算法：Python 迴圈很慢，實際訓練請用融合好的 kernel（第 8 章第 7 集）。
"""

import math

import torch
from torch import Tensor


# ANCHOR: forward
def flash_attention_forward(
    q: Tensor, k: Tensor, v: Tensor, causal: bool = False, block_q: int = 64, block_k: int = 64
) -> tuple[Tensor, Tensor]:
    """q、k (..., T, d)，v (..., T, dv)。回傳輸出 O (..., T, dv) 與每一列分數的 logsumexp (..., T)。

    一次只算一塊 (block_q, block_k) 的分數，整個 (T, T) 的分數矩陣從來不會出現。
    """
    scale = 1 / math.sqrt(q.shape[-1])
    T = q.shape[-2]
    o = q.new_empty(*q.shape[:-1], v.shape[-1])
    lse = q.new_empty(q.shape[:-1])
    for i in range(0, T, block_q):
        qi = q[..., i : i + block_q, :] * scale
        rows = torch.arange(i, i + qi.shape[-2], device=q.device)[:, None]
        m = qi.new_full(qi.shape[:-1], float("-inf"))  # 目前看過的最大分數
        l = qi.new_zeros(qi.shape[:-1])  # 目前的 exp 加總
        acc = qi.new_zeros(*qi.shape[:-1], v.shape[-1])  # 目前的加權和，還沒除以 l
        k_end = i + qi.shape[-2] if causal else T  # causal：這一塊 query 看不到更後面的 key
        for j in range(0, k_end, block_k):
            s = qi @ k[..., j : j + block_k, :].transpose(-2, -1)  # (..., bq, bk)
            if causal:
                cols = torch.arange(j, j + s.shape[-1], device=q.device)[None, :]
                s = s.masked_fill(cols > rows, float("-inf"))
            m_new = torch.maximum(m, s.max(dim=-1).values)
            p = torch.exp(s - m_new[..., None])
            correction = torch.exp(m - m_new)  # 最大值變了，之前累積的要跟著縮放
            l = l * correction + p.sum(dim=-1)
            acc = acc * correction[..., None] + p @ v[..., j : j + block_k, :]
            m = m_new
        o[..., i : i + block_q, :] = acc / l[..., None]
        lse[..., i : i + block_q] = m + torch.log(l)
    return o, lse
# ANCHOR_END: forward


# ANCHOR: backward
def flash_attention_backward(
    q: Tensor, k: Tensor, v: Tensor, o: Tensor, lse: Tensor, do: Tensor,
    causal: bool = False, block_q: int = 64, block_k: int = 64,
) -> tuple[Tensor, Tensor, Tensor]:
    """已知 forward 的輸出 o、logsumexp 與 loss 對 o 的梯度 do，分塊算出 dq、dk、dv。"""
    scale = 1 / math.sqrt(q.shape[-1])
    T = q.shape[-2]
    d = (do * o).sum(dim=-1)  # (..., T)：softmax backward 要用的每列常數
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    for j in range(0, T, block_k):
        kj, vj = k[..., j : j + block_k, :], v[..., j : j + block_k, :]
        cols = torch.arange(j, j + kj.shape[-2], device=q.device)[None, :]
        i_start = j // block_q * block_q if causal else 0  # causal：更前面的 query 看不到這塊 key
        for i in range(i_start, T, block_q):
            qi, doi = q[..., i : i + block_q, :], do[..., i : i + block_q, :]
            s = qi @ kj.transpose(-2, -1) * scale
            if causal:
                rows = torch.arange(i, i + qi.shape[-2], device=q.device)[:, None]
                s = s.masked_fill(cols > rows, float("-inf"))
            p = torch.exp(s - lse[..., i : i + block_q, None])  # 用存下來的 logsumexp 重算機率
            dv[..., j : j + block_k, :] += p.transpose(-2, -1) @ doi
            dp = doi @ vj.transpose(-2, -1)
            ds = p * (dp - d[..., i : i + block_q, None]) * scale
            dq[..., i : i + block_q, :] += ds @ kj
            dk[..., j : j + block_k, :] += ds.transpose(-2, -1) @ qi
    return dq, dk, dv
# ANCHOR_END: backward


# ANCHOR: function
class FlashAttention(torch.autograd.Function):
    """把上面兩個函式接上 autograd：forward 只存 q、k、v、o、logsumexp，不存機率矩陣。"""

    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, causal: bool) -> Tensor:
        o, lse = flash_attention_forward(q, k, v, causal)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do: Tensor) -> tuple[Tensor, Tensor, Tensor, None]:
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = flash_attention_backward(q, k, v, o, lse, do, ctx.causal)
        return dq, dk, dv, None  # causal 不是張量，沒有梯度


def flash_attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = False) -> Tensor:
    return FlashAttention.apply(q, k, v, causal)
# ANCHOR_END: function
