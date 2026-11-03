"""Scaling laws（第 10 章）：計算預算、擬合 IsoFLOPs 曲線與冪律。"""

import numpy as np

from llmz.model import ModelConfig, TransformerLM
from llmz.utils import count_parameters


# ANCHOR: chinchilla_loss
def chinchilla_loss(n, d, e=1.69, a=406.4, b=410.7, alpha=0.34, beta=0.28):
    """Chinchilla 論文（Hoffmann et al. 2022）擬合的 L(N, D) = E + A/N^α + B/D^β。n、d 可以是陣列。"""
    return e + a / np.asarray(n, dtype=float) ** alpha + b / np.asarray(d, dtype=float) ** beta
# ANCHOR_END: chinchilla_loss


# ANCHOR: fit_isoflop
def fit_isoflop_minimum(n, loss) -> float:
    """同一個計算預算下的幾個 (參數量, loss)，對 log10(N) 擬合拋物線，回傳最低點的 N。"""
    c2, c1, _ = np.polyfit(np.log10(n), loss, 2)
    return float(10 ** (-c1 / (2 * c2)))
# ANCHOR_END: fit_isoflop


# ANCHOR: fit_power_law
def fit_power_law(x, y) -> tuple[float, float]:
    """擬合 y = k * x^a。取 log 之後是直線 log y = a log x + log k，回傳 (k, a)。"""
    a, log_k = np.polyfit(np.log(x), np.log(y), 1)
    return float(np.exp(log_k)), float(a)
# ANCHOR_END: fit_power_law


# ANCHOR: budget
def non_embedding_params(cfg: ModelConfig) -> int:
    """參數量扣掉 embedding 表：查表不用計算，lm_head 的矩陣乘法要算，所以留著。"""
    return count_parameters(TransformerLM(cfg)) - cfg.vocab_size * cfg.d_model


def steps_for_budget(flops: float, n_params: int, batch_size: int, context_length: int) -> int:
    """C = 6ND，所以 D = C / (6N) 個 token，再換算成訓練步數。"""
    return round(flops / (6 * n_params) / (batch_size * context_length))
# ANCHOR_END: budget
