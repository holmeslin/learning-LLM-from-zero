import torch

from llmz.functional import cross_entropy, log_softmax, logsumexp, perplexity, softmax


def test_softmax_matches_torch_and_sums_to_one():
    torch.manual_seed(0)
    x = torch.randn(4, 7) * 50  # 數值很大也不能溢位
    assert torch.allclose(softmax(x), torch.softmax(x, dim=-1), atol=1e-6)
    assert torch.allclose(softmax(x).sum(dim=-1), torch.ones(4))


def test_logsumexp_and_log_softmax_are_stable():
    x = torch.tensor([[1000.0, 1000.0], [0.0, -1000.0]])
    assert torch.allclose(logsumexp(x), torch.logsumexp(x, dim=-1))
    assert torch.allclose(log_softmax(x), torch.log_softmax(x, dim=-1))


def test_cross_entropy_matches_torch():
    torch.manual_seed(0)
    logits = torch.randn(5, 10)
    targets = torch.randint(0, 10, (5,))
    assert torch.allclose(cross_entropy(logits, targets), torch.nn.functional.cross_entropy(logits, targets))


def test_uniform_prediction_has_perplexity_equal_to_vocab_size():
    logits = torch.zeros(3, 8)  # 八個選項機率都一樣
    targets = torch.tensor([0, 3, 7])
    assert torch.isclose(perplexity(cross_entropy(logits, targets)), torch.tensor(8.0))
