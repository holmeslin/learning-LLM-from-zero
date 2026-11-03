import torch
import torch.nn.functional as F

from llmz.layers import Embedding, Linear, RMSNorm, SiLUFFN, SwiGLU, silu


def test_linear_is_x_times_w_transposed():
    torch.manual_seed(0)
    layer = Linear(5, 3)
    x = torch.randn(2, 4, 5)
    assert layer.weight.shape == (3, 5)
    assert torch.allclose(layer(x), x @ layer.weight.T)


def test_linear_init_std_and_truncation():
    torch.manual_seed(0)
    layer = Linear(400, 600)
    std = (2 / 1000) ** 0.5
    assert abs(layer.weight.std().item() - std) < 0.1 * std
    assert layer.weight.abs().max() <= 3 * std + 1e-6


def test_embedding_looks_up_rows():
    torch.manual_seed(0)
    emb = Embedding(10, 4)
    ids = torch.tensor([[3, 7], [0, 3]])
    out = emb(ids)
    assert out.shape == (2, 2, 4)
    assert torch.equal(out[0, 0], emb.weight[3]) and torch.equal(out[1, 1], emb.weight[3])


def test_rmsnorm_matches_torch():
    torch.manual_seed(0)
    norm = RMSNorm(8, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(8) + 0.5)
    x = torch.randn(3, 5, 8) * 10
    assert torch.allclose(norm(x), F.rms_norm(x, (8,), norm.weight, eps=1e-5), atol=1e-5)
    assert norm(x.half()).dtype == torch.float16


def test_silu_and_ffn_shapes():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 6)
    assert torch.allclose(silu(x), F.silu(x))
    assert SwiGLU(6, 16)(x).shape == (2, 3, 6)
    assert SiLUFFN(6, 16)(x).shape == (2, 3, 6)
