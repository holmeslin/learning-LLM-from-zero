"""效能分析的小工具（第 7 章）：正確計時、量 backward 要存多少 activations、估算訓練記憶體。"""

import time
from collections.abc import Callable, Iterable

import torch


# ANCHOR: sync
def sync(device: torch.device) -> None:
    """等 GPU 把排隊的工作做完。CPU 本來就是同步的，什麼都不用做。"""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
# ANCHOR_END: sync


# ANCHOR: timeit
def timeit(fn: Callable[[], object], device: torch.device, warmup: int = 3, repeat: int = 10) -> float:
    """先跑 warmup 次不計時，再跑 repeat 次取平均，回傳每次的秒數。前後都要同步。"""
    for _ in range(warmup):
        fn()
    sync(device)
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    sync(device)
    return (time.perf_counter() - start) / repeat
# ANCHOR_END: timeit


# ANCHOR: saved_bytes
def saved_activation_bytes(fn: Callable[[], torch.Tensor], exclude: Iterable[torch.Tensor] = ()) -> int:
    """執行 fn（forward），回傳 autograd 為了 backward 存下來的張量總位元組數。exclude（例如參數）不算。"""
    skip = {t.untyped_storage().data_ptr() for t in exclude}
    seen: dict[int, int] = {}

    def pack(t: torch.Tensor) -> torch.Tensor:
        ptr = t.untyped_storage().data_ptr()
        if ptr not in skip:
            seen[ptr] = t.untyped_storage().nbytes()  # 同一塊記憶體只算一次
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn()
    return sum(seen.values())
# ANCHOR_END: saved_bytes


# ANCHOR: estimate_memory
def estimate_memory(n_params: int) -> dict[str, int]:
    """用 AdamW 訓練時，與 batch 無關的部分：參數、梯度、優化器狀態（float32，位元組）。"""
    return {"params": 4 * n_params, "grads": 4 * n_params, "optimizer": 8 * n_params}
# ANCHOR_END: estimate_memory
