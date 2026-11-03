"""小工具（第 2 章第 7 集起）。"""

from torch import nn


# ANCHOR: count_parameters
def count_parameters(model: nn.Module) -> int:
    """模型裡所有參數的元素總數。"""
    return sum(p.numel() for p in model.parameters())
# ANCHOR_END: count_parameters
