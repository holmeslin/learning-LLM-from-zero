import numpy as np

from llmz.model import ModelConfig
from llmz.scaling import chinchilla_loss, fit_isoflop_minimum, fit_power_law, non_embedding_params, steps_for_budget


def test_fit_power_law_recovers_exact_parameters():
    x = np.array([1e3, 1e4, 1e5, 1e6])
    k, a = fit_power_law(x, 3.0 * x**0.5)
    assert np.isclose(k, 3.0) and np.isclose(a, 0.5)


def test_fit_isoflop_minimum_finds_vertex():
    n = np.array([1e6, 3e6, 1e7, 3e7, 1e8])
    loss = 2.0 + 0.3 * (np.log10(n) - np.log10(5e6)) ** 2
    assert np.isclose(fit_isoflop_minimum(n, loss), 5e6)


def test_chinchilla_isoflop_optimum_matches_closed_form():
    c = 1e21
    n = np.logspace(8, 11, 400)
    loss = chinchilla_loss(n, c / (6 * n))
    numeric = n[np.argmin(loss)]
    # 對 L(N, C/6N) 微分等於 0 的解：N_opt = G (C/6)^(β/(α+β))，G = (αA/(βB))^(1/(α+β))
    alpha, beta, a, b = 0.34, 0.28, 406.4, 410.7
    closed = (alpha * a / (beta * b)) ** (1 / (alpha + beta)) * (c / 6) ** (beta / (alpha + beta))
    assert abs(np.log(numeric / closed)) < 0.02


def test_budget_helpers():
    cfg = ModelConfig(vocab_size=100, d_model=16, n_layers=1, n_heads=2, d_ff=32)
    # 一層：q k v o 4×16×16、SwiGLU 3×16×32、兩個 RMSNorm 2×16；最後的 RMSNorm 16；lm_head 16×100
    assert non_embedding_params(cfg) == 4 * 256 + 3 * 512 + 32 + 16 + 1600
    assert steps_for_budget(6 * 1000 * 8192 * 10, 1000, 32, 256) == 10
