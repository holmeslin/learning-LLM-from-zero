# FSDP 的想法

## 本集目標

理解 FSDP（Fully Sharded Data Parallel）：參數、梯度、優化器狀態全部切開存，每一層要用時才 all-gather 出完整參數，用完就丟；並用一個 Linear 層在兩個 rank 上示範。

## 正文

### 四種切法的記憶體

\\( \Psi \\) 個參數、\\( N \\) 個 rank、用 AdamW 和 float32，每個 rank 要存（不算 activations）：

| 做法 | 參數 | 梯度 | 優化器狀態 | 每個 rank |
| --- | --- | --- | --- | --- |
| DDP | \\( 4\Psi \\) | \\( 4\Psi \\) | \\( 8\Psi \\) | \\( 16\Psi \\) |
| ZeRO 1（切優化器狀態） | \\( 4\Psi \\) | \\( 4\Psi \\) | \\( 8\Psi/N \\) | |
| ZeRO 2（再切梯度） | \\( 4\Psi \\) | \\( 4\Psi/N \\) | \\( 8\Psi/N \\) | |
| ZeRO 3／FSDP（全部切） | \\( 4\Psi/N \\) | \\( 4\Psi/N \\) | \\( 8\Psi/N \\) | \\( 16\Psi/N \\) |

第 1 集的 70 億參數模型、8 張卡：

```python
psi, n = 7e9, 8
rows = {
    "DDP": 4 * psi + 4 * psi + 8 * psi,
    "ZeRO 1": 4 * psi + 4 * psi + 8 * psi / n,
    "ZeRO 2": 4 * psi + (4 * psi + 8 * psi) / n,
    "ZeRO 3 / FSDP": 16 * psi / n,
}
for name, size in rows.items():
    print(f"{name:14} {size / 1e9:6.1f} GB")
```

執行結果：

```text
DDP             112.0 GB
ZeRO 1           63.0 GB
ZeRO 2           38.5 GB
ZeRO 3 / FSDP    14.0 GB
```

只有全部切開，才能塞進一張 80 GB 的卡，還留空間給 activations。

### 要用時才拼起來

參數切開之後，forward 要算某一層時，每個 rank 手上只有這層參數的 \\( 1/N \\)。FSDP 的做法是：

1. **forward 到這一層**：all-gather 出完整的參數，算完這一層就把完整參數丟掉，只留自己那份。
2. **backward 到這一層**：再 all-gather 一次完整參數（forward 時丟了），算出完整的梯度。
3. **reduce-scatter 梯度**：每個 rank 只拿到自己那份參數的梯度總和，其他的丟掉。
4. **更新**：每個 rank 只更新自己那份參數，優化器狀態也只有那份。

任何時刻，完整的參數只存在「正在算的那一層」。通訊量是每層兩次 all-gather 加一次 reduce-scatter，比 DDP 的一次 all-reduce（= 一次 reduce-scatter + 一次 all-gather）多了一次 all-gather，約 1.5 倍。換來的是記憶體除以 \\( N \\)。實作上會在算這一層時，先 all-gather 下一層的參數，讓通訊和計算重疊（第 5 集的想法）。

### 一層的示範

用一個 `Linear(8, 6)` 的權重示範第 1～3 步。權重攤平成一維、切成兩半，每個 rank 只存一半：

```python
import torch
import torch.distributed as dist

from llmz.dist import launch, print_in_order


def run(rank, world_size):
    torch.manual_seed(0)
    full_weight = torch.randn(6, 8)  # 只用來產生初始值與對答案
    x = torch.randn(4, 8)
    shard = full_weight.view(-1).chunk(world_size)[rank].clone()  # 每個 rank 平常只存這一份
    my_x = x[rank * 2 : rank * 2 + 2]  # data parallel：每個 rank 看不同的資料

    # 1～2. 要用時 all-gather 出完整的權重
    gathered = torch.empty(full_weight.numel())
    dist.all_gather_into_tensor(gathered, shard)
    weight = gathered.view(6, 8).requires_grad_()
    loss = (my_x @ weight.T).pow(2).mean()
    loss.backward()

    # 3. reduce-scatter 梯度：每個 rank 只拿自己那份的總和，再除以 N 取平均
    my_grad = torch.empty_like(shard)
    dist.reduce_scatter_tensor(my_grad, weight.grad.view(-1))
    my_grad /= world_size
    del weight, gathered  # 完整的權重與梯度用完就丟

    # 對答案：單機看全部資料
    reference = full_weight.clone().requires_grad_()
    ((x[:2] @ reference.T).pow(2).mean() + (x[2:] @ reference.T).pow(2).mean()).div(2).backward()
    expected = reference.grad.view(-1).chunk(world_size)[rank]
    print_in_order(f"存 {shard.numel()} 個參數，梯度正確：{torch.allclose(my_grad, expected, atol=1e-6)}")


if __name__ == "__main__":
    launch(run, 2)
```

執行結果：

```text
[rank 0] 存 24 個參數，梯度正確：True
[rank 1] 存 24 個參數，梯度正確：True
```

- `dist.all_gather_into_tensor(out, x)`：all-gather 的另一種寫法，結果直接接成一個長張量，不用像第 3 集那樣準備串列。
- `dist.reduce_scatter_tensor(out, x)`：reduce-scatter 的對應寫法，把長張量 `x` 切成 \\( N \\) 份，第 \\( r \\) 份的總和放進 rank \\( r \\) 的 `out`。

每個 rank 平常只存 24 個參數，卻能算出和單機一樣的梯度。一個完整的 FSDP 就是對模型的每一層都這樣做，再加上第 6 集的分片優化器。PyTorch 內建的版本是 `torch.distributed.fsdp`，本書不展開。

## 重點整理

- ZeRO 1、2、3 依序把優化器狀態、梯度、參數切開；第 3 階段（FSDP）每個 rank 只存 \\( 16\Psi/N \\)。
- FSDP：每層 forward 與 backward 前 all-gather 完整參數，用完就丟；梯度用 reduce-scatter，每個 rank 只留自己那份。
- 通訊量約 DDP 的 1.5 倍，換到記憶體除以 \\( N \\)。
- `dist.all_gather_into_tensor`、`dist.reduce_scatter_tensor` 是拼成一個長張量的版本。
