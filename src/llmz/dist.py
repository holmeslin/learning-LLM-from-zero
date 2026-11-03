"""平行化（第 9 章）：在 CPU 上用 Gloo 起好幾個 process，模擬好幾張卡。"""

import os
import socket
import sys
from collections.abc import Callable, Iterable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn


# ANCHOR: launch
def _worker(rank: int, world_size: int, port: int, fn: Callable, args: tuple) -> None:
    saved = os.dup(1)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 1)  # Linux 版 Gloo 連線成功時會往 stdout 印一行訊息，先藏起來
        try:
            dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
        finally:
            os.dup2(saved, 1)
            os.close(saved)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def launch(fn: Callable, world_size: int, *args) -> None:
    """起 world_size 個 process，每個都執行 fn(rank, world_size, *args)，全部結束才回來。"""
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")  # 只走本機
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))  # 讓作業系統挑一個沒人用的 port
        port = s.getsockname()[1]
    mp.spawn(_worker, args=(world_size, port, fn, args), nprocs=world_size)
# ANCHOR_END: launch


# ANCHOR: print_in_order
def print_in_order(*args) -> None:
    """每個 rank 依序印出，輸出順序才固定。所有 rank 都要呼叫。"""
    for r in range(dist.get_world_size()):
        if r == dist.get_rank():
            print(f"[rank {r}]", *args, flush=True)
        dist.barrier()
# ANCHOR_END: print_in_order


# ANCHOR: ring_all_reduce
def ring_all_reduce(x: Tensor) -> Tensor:
    """就地把每個 rank 的 x 加總。x 切成 world_size 塊，每一步只和左右鄰居交換一塊。"""
    rank, n = dist.get_rank(), dist.get_world_size()
    chunks = list(x.view(-1).chunk(n))  # 都是 x 的 view，改它們就是改 x
    right, left = (rank + 1) % n, (rank - 1) % n

    def exchange(send_idx: int, recv_idx: int) -> Tensor:
        buf = torch.empty_like(chunks[recv_idx])
        req = dist.isend(chunks[send_idx], right)  # 先丟出去不等，才不會兩邊互等
        dist.recv(buf, left)
        req.wait()
        return buf

    for step in range(n - 1):  # reduce-scatter：做完後第 r 個 rank 擁有第 r+1 塊的總和
        recv_idx = (rank - step - 1) % n
        chunks[recv_idx] += exchange((rank - step) % n, recv_idx)
    for step in range(n - 1):  # all-gather：把加總好的塊沿著環傳一圈
        recv_idx = (rank - step) % n
        chunks[recv_idx].copy_(exchange((rank - step + 1) % n, recv_idx))
    return x
# ANCHOR_END: ring_all_reduce


# ANCHOR: naive_ddp
def broadcast_parameters(model: nn.Module) -> None:
    """讓每個 rank 的參數都和 rank 0 一樣。"""
    for p in model.parameters():
        dist.broadcast(p.data, src=0)


def all_reduce_gradients(model: nn.Module) -> None:
    """backward 之後呼叫：每個參數的梯度在所有 rank 之間取平均。"""
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad)
            p.grad /= dist.get_world_size()
# ANCHOR_END: naive_ddp


# ANCHOR: bucketed_ddp
class BucketedDDP:
    """backward 還在跑的時候，每湊滿一個 bucket 的梯度就丟出非同步 all-reduce。

    shortcut: 假設每個參數每次 backward 剛好得到一次梯度；有沒用到的參數或梯度累積時要改計數方式。
    """

    def __init__(self, model: nn.Module, bucket_mb: float = 25.0):
        self.model = model
        broadcast_parameters(model)
        self.buckets: list[list[nn.Parameter]] = [[]]
        size = 0
        for p in reversed(list(model.parameters())):  # backward 大致從最後一層往前算
            if size >= bucket_mb * 1e6:
                self.buckets.append([])
                size = 0
            self.buckets[-1].append(p)
            size += p.numel() * p.element_size()
        self.bucket_of = {p: i for i, bucket in enumerate(self.buckets) for p in bucket}
        self.ready = [0] * len(self.buckets)
        self.works: list[tuple[int, Tensor, dist.Work]] = []
        for p in model.parameters():
            p.register_post_accumulate_grad_hook(self._on_grad_ready)

    def _on_grad_ready(self, p: nn.Parameter) -> None:
        i = self.bucket_of[p]
        self.ready[i] += 1
        if self.ready[i] == len(self.buckets[i]):  # 這個 bucket 的梯度都算好了
            flat = torch.cat([q.grad.view(-1) for q in self.buckets[i]])
            self.works.append((i, flat, dist.all_reduce(flat, async_op=True)))

    def finish(self) -> None:
        """optimizer.step() 之前呼叫：等所有 all-reduce 做完，把平均後的梯度放回去。"""
        for i, flat, work in self.works:
            work.wait()
            flat /= dist.get_world_size()
            offset = 0
            for q in self.buckets[i]:
                q.grad.copy_(flat[offset : offset + q.numel()].view_as(q.grad))
                offset += q.numel()
        self.ready = [0] * len(self.buckets)
        self.works = []
# ANCHOR_END: bucketed_ddp


# ANCHOR: sharded_optimizer
class ShardedOptimizer:
    """每個 rank 只替自己負責的參數存優化器狀態、做更新，更新完再把參數 broadcast 給大家。"""

    def __init__(self, params: Iterable[nn.Parameter], optimizer_cls: type, **kwargs):
        self.params = list(params)
        load = [0] * dist.get_world_size()
        self.owner = [0] * len(self.params)
        for i in sorted(range(len(self.params)), key=lambda i: -self.params[i].numel()):
            r = load.index(min(load))  # 從大的參數開始，交給目前負責最少元素的 rank
            self.owner[i] = r
            load[r] += self.params[i].numel()
        mine = [p for p, r in zip(self.params, self.owner) if r == dist.get_rank()]
        self.optimizer = optimizer_cls(mine, **kwargs)

    def step(self) -> None:
        self.optimizer.step()
        for p, r in zip(self.params, self.owner):
            dist.broadcast(p.data, src=r)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None
# ANCHOR_END: sharded_optimizer
