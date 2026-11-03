"""優化器與訓練用的小工具（第 5 章第 2～5 集）：SGD、AdamW、cosine 學習率排程、梯度裁剪。

全書不用 torch.optim 的現成優化器，只用 torch.optim.Optimizer 這個基底類別；
tests/test_optim.py 用 torch.optim.SGD／AdamW 驗證算出來一樣。
"""

import math
from collections.abc import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer


# ANCHOR: sgd
class SGD(Optimizer):
    """p = p - lr * grad。"""

    def __init__(self, params, lr: float):
        super().__init__(params, {"lr": lr})

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p -= group["lr"] * p.grad
# ANCHOR_END: sgd


# ANCHOR: adamw
class AdamW(Optimizer):
    """Adam 加上 decoupled weight decay。每個參數各自記住梯度的一階與二階動量。"""

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.95),
                 eps: float = 1e-8, weight_decay: float = 0.1):
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # 這個參數專屬的狀態字典，第一次是空的
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["t"] += 1
                t, m, v, g = state["t"], state["m"], state["v"], p.grad
                m.mul_(b1).add_((1 - b1) * g)  # 梯度的移動平均
                v.mul_(b2).add_((1 - b2) * g * g)  # 梯度平方的移動平均
                p *= 1 - lr * wd  # weight decay：參數先往 0 縮一點點，和梯度無關
                lr_t = lr * math.sqrt(1 - b2**t) / (1 - b1**t)  # 修正前幾步動量偏小的問題
                p -= lr_t * m / (torch.sqrt(v) + eps)
# ANCHOR_END: adamw


# ANCHOR: lr_cosine_schedule
def lr_cosine_schedule(step: int, lr_max: float, lr_min: float, warmup_steps: int, cosine_steps: int) -> float:
    """先線性 warmup 到 lr_max，再用 cosine 降到 lr_min，之後維持 lr_min。"""
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    if step > cosine_steps:
        return lr_min
    progress = (step - warmup_steps) / (cosine_steps - warmup_steps)
    return lr_min + 0.5 * (1 + math.cos(math.pi * progress)) * (lr_max - lr_min)
# ANCHOR_END: lr_cosine_schedule


# ANCHOR: clip_grad_norm
@torch.no_grad()
def clip_grad_norm_(params: Iterable[Tensor], max_norm: float, eps: float = 1e-6) -> Tensor:
    """把所有梯度看成一個大向量，長度超過 max_norm 就等比例縮小。回傳裁剪前的長度。"""
    grads = [p.grad for p in params if p.grad is not None]
    total_norm = torch.sqrt(sum((g * g).sum() for g in grads))
    if total_norm > max_norm:
        for g in grads:
            g *= max_norm / (total_norm + eps)
    return total_norm
# ANCHOR_END: clip_grad_norm
