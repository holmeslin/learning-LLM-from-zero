"""選擇執行裝置（第 1 章第 9 集）。"""

import torch


# ANCHOR: get_device
def get_device() -> torch.device:
    """有 GPU 就用 GPU：NVIDIA 用 cuda，Apple Silicon 用 mps，否則用 cpu。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
# ANCHOR_END: get_device
