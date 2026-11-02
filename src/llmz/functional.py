"""機率相關的基本運算（第 1 章第 13～16 集）：softmax、log-softmax、cross-entropy、perplexity。

全書不用 torch.nn.functional，這裡是我們自己寫的版本；
tests/test_functional.py 用 torch 內建函式驗證它們算得一樣。
"""

import torch
from torch import Tensor


# ANCHOR: softmax
def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """把一排任意實數變成機率分布：每個值在 0～1 之間，沿著 dim 加總為 1。"""
    x = x - x.max(dim=dim, keepdim=True).values  # 先減掉最大值，exp 才不會溢位
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)
# ANCHOR_END: softmax


# ANCHOR: logsumexp
def logsumexp(x: Tensor, dim: int = -1) -> Tensor:
    """log(sum(exp(x)))，先減掉最大值再算，不會溢位。"""
    m = x.max(dim=dim, keepdim=True).values
    return (m + torch.log(torch.exp(x - m).sum(dim=dim, keepdim=True))).squeeze(dim)
# ANCHOR_END: logsumexp


# ANCHOR: log_softmax
def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """log(softmax(x))。先減掉最大值，之後的每一步都在小數值上運算，不會溢位也不會損失精度。"""
    shifted = x - x.max(dim=dim, keepdim=True).values
    return shifted - torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True))
# ANCHOR_END: log_softmax


# ANCHOR: cross_entropy
def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """平均 cross-entropy。logits 形狀 (N, V)，targets 形狀 (N,)，放的是正確答案的索引。"""
    logp = log_softmax(logits, dim=-1)
    picked = logp[torch.arange(len(targets)), targets]  # 每一列只取正確答案那一格
    return -picked.mean()
# ANCHOR_END: cross_entropy


# ANCHOR: perplexity
def perplexity(loss: Tensor) -> Tensor:
    """perplexity 就是 cross-entropy 取 exp。"""
    return torch.exp(loss)
# ANCHOR_END: perplexity
