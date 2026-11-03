import math

import torch
from torch import nn

from llmz.optim import SGD, AdamW, clip_grad_norm_, lr_cosine_schedule


def two_models():
    torch.manual_seed(0)
    a = nn.Linear(4, 3)
    b = nn.Linear(4, 3)
    b.load_state_dict(a.state_dict())
    return a, b


def run(model, opt, steps=5):
    torch.manual_seed(1)
    x = torch.randn(8, 4)
    for _ in range(steps):
        (model(x) ** 2).mean().backward()
        opt.step()
        opt.zero_grad()


def test_sgd_matches_torch():
    a, b = two_models()
    run(a, SGD(a.parameters(), lr=0.1))
    run(b, torch.optim.SGD(b.parameters(), lr=0.1))
    assert torch.allclose(a.weight, b.weight)


def test_adamw_matches_torch():
    a, b = two_models()
    run(a, AdamW(a.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1))
    run(b, torch.optim.AdamW(b.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1))
    assert torch.allclose(a.weight, b.weight, atol=1e-6)
    assert torch.allclose(a.bias, b.bias, atol=1e-6)


def test_cosine_schedule_shape():
    f = lambda s: lr_cosine_schedule(s, lr_max=1.0, lr_min=0.1, warmup_steps=10, cosine_steps=110)
    assert f(0) == 0 and f(5) == 0.5 and f(10) == 1.0
    assert math.isclose(f(60), 0.55)  # cosine 的中點
    assert f(110) == 0.1 and f(500) == 0.1


def test_clip_grad_norm_scales_down_only_when_too_big():
    p = nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([3.0, 4.0])
    norm = clip_grad_norm_([p], max_norm=1.0)
    assert math.isclose(norm.item(), 5.0)
    assert torch.allclose(p.grad, torch.tensor([0.6, 0.8]), atol=1e-5)
    p.grad = torch.tensor([0.3, 0.4])
    clip_grad_norm_([p], max_norm=1.0)
    assert torch.allclose(p.grad, torch.tensor([0.3, 0.4]))
