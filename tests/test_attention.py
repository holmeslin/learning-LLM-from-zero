import torch
import torch.nn.functional as F

from llmz.attention import MultiHeadSelfAttention, RoPE, causal_mask, scaled_dot_product_attention


def test_rope_keeps_length_and_only_depends_on_relative_position():
    torch.manual_seed(0)
    rope = RoPE(d_k=8, max_seq_len=32)
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    pos = torch.arange(4)
    assert torch.allclose(rope(q, pos).norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    a = rope(q, pos) @ rope(k, pos).transpose(-2, -1)
    b = rope(q, pos + 10) @ rope(k, pos + 10).transpose(-2, -1)
    assert torch.allclose(a, b, atol=1e-4)  # 整體平移 10 個位置，點積不變


def test_rope_matches_complex_number_reference():
    torch.manual_seed(0)
    rope = RoPE(d_k=6, max_seq_len=16, theta=100.0)
    x = torch.randn(2, 5, 6)
    pos = torch.tensor([0, 3, 4, 10, 15])
    freqs = 100.0 ** (-torch.arange(0, 6, 2) / 6)
    angles = pos[:, None] * freqs[None, :]
    ref = torch.view_as_complex(x.reshape(2, 5, 3, 2)) * torch.polar(torch.ones_like(angles), angles)
    assert torch.allclose(rope(x, pos), torch.view_as_real(ref).flatten(-2), atol=1e-5)


def test_causal_mask_is_lower_triangular():
    m = causal_mask(3)
    assert m.dtype == torch.bool
    assert torch.equal(m, torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]]).bool())


def test_sdpa_matches_torch_with_and_without_mask():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 5, 8), torch.randn(2, 3, 5, 4)
    assert torch.allclose(scaled_dot_product_attention(q, k, v), F.scaled_dot_product_attention(q, k, v), atol=1e-5)
    ours = scaled_dot_product_attention(q, k, v, causal_mask(5))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_mha_is_causal():
    torch.manual_seed(0)
    mha = MultiHeadSelfAttention(d_model=16, n_heads=4, rope=RoPE(4, 32))
    x = torch.randn(1, 6, 16)
    y = mha(x)
    x2 = x.clone()
    x2[:, 4:] = torch.randn(1, 2, 16)  # 改掉後兩個位置
    y2 = mha(x2)
    assert y.shape == (1, 6, 16)
    assert torch.allclose(y[:, :4], y2[:, :4], atol=1e-5)  # 前四個位置的輸出不受影響
    assert not torch.allclose(y[:, 4:], y2[:, 4:])
