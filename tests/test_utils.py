import torch
from torch import nn

from llmz.utils import count_parameters


class TwoParams(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(3, 4))
        self.b = nn.Parameter(torch.zeros(4))


def test_count_parameters_sums_every_element():
    assert count_parameters(TwoParams()) == 3 * 4 + 4


def test_count_parameters_of_empty_module_is_zero():
    assert count_parameters(nn.Module()) == 0
