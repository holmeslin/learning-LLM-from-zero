import pytest
import torch

from llmz.attention import causal_mask, scaled_dot_product_attention
from llmz.flash import flash_attention, flash_attention_forward
from llmz.functional import logsumexp


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("T", [7, 64, 100])  # 不整除、剛好一塊、好幾塊
def test_forward_matches_naive_attention(causal, T):
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, T, 16), torch.randn(2, 3, T, 16), torch.randn(2, 3, T, 8)
    o, lse = flash_attention_forward(q, k, v, causal, block_q=32, block_k=16)
    mask = causal_mask(T) if causal else None
    assert torch.allclose(o, scaled_dot_product_attention(q, k, v, mask), atol=1e-5)
    scores = q @ k.transpose(-2, -1) / 4
    if causal:
        scores = scores.masked_fill(~mask, float("-inf"))
    assert torch.allclose(lse, logsumexp(scores), atol=1e-5)


@pytest.mark.parametrize("causal", [False, True])
def test_backward_matches_autograd(causal):
    torch.manual_seed(0)
    T = 100
    q, k, v = (torch.randn(2, 2, T, 16, requires_grad=True) for _ in range(3))
    w = torch.randn(2, 2, T, 16)
    (flash_attention(q, k, v, causal) * w).sum().backward()
    ours = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None
    (scaled_dot_product_attention(q, k, v, causal_mask(T) if causal else None) * w).sum().backward()
    for a, t in zip(ours, (q, k, v)):
        assert torch.allclose(a, t.grad, atol=1e-4)

