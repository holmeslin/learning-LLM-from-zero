import torch

from llmz.functional import cross_entropy
from llmz.model import ModelConfig, TransformerLM
from llmz.profile import estimate_memory, saved_activation_bytes, timeit

SMALL = ModelConfig(vocab_size=50, context_length=16, d_model=32, n_layers=2, n_heads=4, d_ff=64)


def test_timeit_counts_calls_and_returns_seconds():
    calls = []
    t = timeit(lambda: calls.append(1), torch.device("cpu"), warmup=2, repeat=5)
    assert len(calls) == 7 and t >= 0


def test_saved_bytes_of_exp_is_its_output():
    w = torch.ones(10, requires_grad=True)
    assert saved_activation_bytes(lambda: torch.exp(w).sum()) == 40  # exp 的 backward 只需要輸出本身
    assert saved_activation_bytes(lambda: w.sum()) == 0
    assert saved_activation_bytes(lambda: (w * w).sum(), exclude=[w]) == 0


def test_checkpoint_saves_less_and_gives_same_gradients():
    ids = torch.randint(0, 50, (2, 9))

    def run(ckpt):
        torch.manual_seed(0)
        model = TransformerLM(ModelConfig(**{**SMALL.__dict__, "activation_checkpoint": ckpt}))
        loss_fn = lambda: cross_entropy(model(ids[:, :-1]).reshape(-1, 50), ids[:, 1:].reshape(-1))
        saved = saved_activation_bytes(loss_fn, exclude=model.parameters())
        loss_fn().backward()
        return saved, model.embed.weight.grad

    plain_bytes, plain_grad = run(False)
    ckpt_bytes, ckpt_grad = run(True)
    assert ckpt_bytes < plain_bytes
    assert torch.allclose(plain_grad, ckpt_grad, atol=1e-6)


def test_estimate_memory_is_16_bytes_per_param():
    assert sum(estimate_memory(1000).values()) == 16000
