import torch

from llmz.inference import (
    KVCache,
    accept_or_resample,
    dequantize,
    forward_cached,
    generate_cached,
    quantize,
    speculative_generate,
)
from llmz.model import ModelConfig, TransformerLM

CFG = ModelConfig(vocab_size=50, context_length=32, d_model=32, n_layers=2, n_heads=4, d_ff=64)


def make_model(seed: int, cfg: ModelConfig = CFG) -> TransformerLM:
    torch.manual_seed(seed)
    return TransformerLM(cfg).eval()


def test_forward_cached_matches_full_forward():
    model = make_model(0)
    ids = torch.randint(0, 50, (2, 14))
    full = model(ids)
    cache = KVCache(CFG, 2, torch.device("cpu"))
    assert torch.allclose(forward_cached(model, ids[:, :6], cache), full[:, :6], atol=1e-5)  # prefill
    for t in range(6, 10):  # 一次一個
        assert torch.allclose(forward_cached(model, ids[:, t : t + 1], cache)[:, 0], full[:, t], atol=1e-5)
    assert torch.allclose(forward_cached(model, ids[:, 10:14], cache), full[:, 10:14], atol=1e-5)  # cache 不空、一次多個
    assert cache.length == 14


def test_generate_cached_matches_naive_greedy():
    model = make_model(0)
    ids = torch.randint(0, 50, (3, 5))
    naive = ids
    for _ in range(10):
        naive = torch.cat([naive, model(naive)[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(generate_cached(model, ids, 10), naive)


def test_speculative_greedy_matches_target_greedy():
    target = make_model(0)
    draft = make_model(1, ModelConfig(vocab_size=50, context_length=32, d_model=16, n_layers=1, n_heads=2, d_ff=32))
    ids = torch.randint(0, 50, (1, 4))
    out, accepted, proposed = speculative_generate(target, draft, ids, 20, k=3)
    assert torch.equal(out, generate_cached(target, ids, 20))
    assert 0 <= accepted <= proposed


def test_accept_or_resample_follows_target_distribution():
    torch.manual_seed(0)
    p = torch.tensor([0.5, 0.3, 0.15, 0.05])
    q = torch.tensor([0.1, 0.2, 0.3, 0.4])
    counts = torch.zeros(4)
    for _ in range(20000):
        _, t = accept_or_resample(p, q, torch.multinomial(q, 1).item())
        counts[t] += 1
    assert torch.allclose(counts / counts.sum(), p, atol=0.01)


def test_quantize_round_trip_error_within_half_step():
    torch.manual_seed(0)
    w = torch.randn(8, 64)
    for bits, group in [(8, None), (4, 16)]:
        q, scale = quantize(w, bits, group)
        assert q.dtype == torch.int8 and q.abs().max() <= 2 ** (bits - 1) - 1
        err = (dequantize(q, scale) - w).reshape(q.shape).abs()
        assert (err <= scale / 2 + 1e-6).all()
