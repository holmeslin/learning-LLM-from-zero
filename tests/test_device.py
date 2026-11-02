import torch

from llmz.device import get_device


def test_get_device_returns_a_usable_device():
    device = get_device()
    assert device.type in {"cuda", "mps", "cpu"}
    assert (torch.ones(2, device=device) + 1).sum().item() == 4
