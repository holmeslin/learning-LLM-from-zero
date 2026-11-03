import torch

from llmz.model import ModelConfig, TransformerLM
from llmz.utils import count_parameters

SMALL = ModelConfig(vocab_size=50, context_length=16, d_model=32, n_layers=2, n_heads=4, d_ff=64)


def test_lm_output_shape_and_causality():
    torch.manual_seed(0)
    model = TransformerLM(SMALL)
    ids = torch.randint(0, 50, (2, 10))
    logits = model(ids)
    assert logits.shape == (2, 10, 50)
    ids2 = ids.clone()
    ids2[:, 7:] = (ids2[:, 7:] + 1) % 50
    assert torch.allclose(logits[:, :7], model(ids2)[:, :7], atol=1e-5)


def test_parameter_count_matches_formula():
    model = TransformerLM(SMALL)
    d, ff, v, L = 32, 64, 50, 2
    per_layer = 4 * d * d + 3 * d * ff + 2 * d  # attention 四個投影、SwiGLU 三個矩陣、兩個 RMSNorm
    assert count_parameters(model) == v * d + L * per_layer + d + d * v


def test_ablation_flags_build_and_run():
    torch.manual_seed(0)
    ids = torch.randint(0, 50, (1, 8))
    for kw in [dict(norm="none"), dict(norm_position="post"), dict(rope=False), dict(ffn="silu")]:
        cfg = ModelConfig(**{**SMALL.__dict__, **kw})
        assert TransformerLM(cfg)(ids).shape == (1, 8, 50)
