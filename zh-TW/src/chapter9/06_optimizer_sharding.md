# optimizer state sharding

## 本集目標

發現 DDP 裡每個 rank 都存了一份完整而重複的優化器狀態，把它切開，每個 rank 只替一部分參數存狀態、做更新，再把更新後的參數傳給大家。

## 正文

### DDP 的重複

DDP 裡每個 rank 都有完整的參數、梯度與 AdamW 狀態，而且 all-reduce 之後它們**一模一樣**。AdamW 的兩個動量佔了 16 位元組裡的 8 位元組，\\( N \\) 個 rank 就存了 \\( N \\) 份相同的東西。

既然更新結果都一樣，何必每個人都算？讓每個 rank 只負責一部分參數：只替它們存 AdamW 狀態、只更新它們，更新完再 broadcast 給其他人。這就是 **ZeRO**（Zero Redundancy Optimizer）的第一階段，PyTorch 裡叫 `ZeroRedundancyOptimizer`。

```python,ignore
{{#include ../../../src/llmz/dist.py:sharded_optimizer}}
```

- 分配：從最大的參數開始，每個交給目前負責元素最少的 rank，讓大家的負擔差不多。標準模型的 embedding 和 `lm_head` 各有 512 萬個元素，其他參數小得多，照順序輪流分會很不平均。`sorted(..., key=...)` 依 `key` 算出的值排序，這裡排的是參數的索引，負號讓大的排前面。
- 每個 rank 用自己那份參數建一個真正的優化器（例如第 5 章的 `AdamW`），所以只有自己那份有狀態。
- `step()`：先更新自己負責的參數，再對每個參數從它的負責人 broadcast 一次，大家又一致了。
- `zero_grad()` 要清掉**所有**參數的梯度，不只是自己負責的。

`optimizer_cls` 傳進來的是類別本身（例如 `AdamW`，不加括號），`**kwargs` 是要轉交給它的參數（例如 `lr=1e-3`）。

### 每個 rank 省了多少

```python
import torch

from llmz.dist import ShardedOptimizer, all_reduce_gradients, broadcast_parameters, launch, print_in_order
from llmz.functional import cross_entropy
from llmz.model import ModelConfig, TransformerLM
from llmz.optim import AdamW


def run(rank, world_size):
    torch.manual_seed(0)
    model = TransformerLM(ModelConfig())
    broadcast_parameters(model)
    opt = ShardedOptimizer(model.parameters(), AdamW, lr=1e-3)

    torch.manual_seed(rank)
    ids = torch.randint(0, 10000, (2, 33))
    cross_entropy(model(ids[:, :-1]).reshape(-1, 10000), ids[:, 1:].reshape(-1)).backward()
    all_reduce_gradients(model)
    opt.step()
    opt.zero_grad()

    mine = opt.optimizer.param_groups[0]["params"]
    state_bytes = sum(s["m"].numel() * 4 + s["v"].numel() * 4 for s in opt.optimizer.state.values())
    print_in_order(f"負責 {len(mine)} 個參數、{sum(p.numel() for p in mine):,} 個元素，AdamW 狀態 {state_bytes / 1e6:.1f} MB")


if __name__ == "__main__":
    launch(run, 2)
```

執行結果：

```text
[rank 0] 負責 20 個參數、11,348,480 個元素，AdamW 狀態 90.8 MB
[rank 1] 負責 19 個參數、11,347,968 個元素，AdamW 狀態 90.8 MB
```

第 5 章的 `AdamW` 在 `state` 裡替每個參數存 `m`、`v` 兩個張量（還有步數 `t`，是整數），我們把 `m`、`v` 的位元組加起來。單機的 AdamW 狀態是 181.6 MB（第 7 章第 4 集），兩個 rank 各存一半左右。`tests/test_dist.py` 驗證了它和不切的 AdamW 更新出一模一樣的參數。

### 通訊多了多少

我們的寫法每一步做一次完整的 all-reduce（梯度），再 broadcast 一次所有參數，通訊量比 DDP 多。真正的 ZeRO 更聰明：既然每個 rank 只更新自己那份參數，它也只需要**自己那份的梯度總和**，所以把 all-reduce 換成 reduce-scatter；更新完再用 all-gather 把參數拼回來。第 3 集說過 all-reduce = reduce-scatter + all-gather，所以總通訊量和 DDP 一樣，記憶體卻省了。

### 再往下切

同樣的想法可以繼續：

- 梯度也只留自己那份（reduce-scatter 之後別人的那份就不需要了），這是 ZeRO 第二階段。
- 連參數本身也切開，每個 rank 平常只存一部分，要用的時候再拿，這是 ZeRO 第三階段，PyTorch 裡叫 FSDP，下一集的主題。

## 重點整理

- DDP 裡每個 rank 的優化器狀態都相同，是重複的；ZeRO 第一階段把它切開。
- `llmz.dist.ShardedOptimizer(params, AdamW, lr=...)`：每個 rank 只替自己負責的參數存狀態、做更新，再從負責人 broadcast 參數。
- 標準模型兩個 rank：每個 rank 的 AdamW 狀態從 181.6 MB 降到約一半。
- 正式做法用 reduce-scatter 梯度 + all-gather 參數，總通訊量和 DDP 一樣。
