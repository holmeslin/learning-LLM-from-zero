# naive DDP

## 本集目標

寫出最簡單的 data parallel：每個 rank 算不同的資料，backward 後把每個參數的梯度 all-reduce 取平均，並驗證結果和單機看整個 batch 一樣。

## 正文

### 想法

把一個 batch 切成 \\( N \\) 份，每個 rank 拿一份、各自 forward 與 backward。loss 是平均值，所以

\\[ \nabla \frac{1}{N} \sum_{r} L_r = \frac{1}{N} \sum_r \nabla L_r \\]

整個 batch 的梯度，就是每一份梯度的平均。每個 rank 算完自己的梯度後做一次 all-reduce 再除以 \\( N \\)，大家就都拿到整個 batch 的梯度。接著每個 rank 用同樣的梯度、同樣的優化器各自更新，參數就保持一模一樣，不需要再傳參數。

前提是一開始大家的參數就相同。每個 process 各自初始化，亂數不一定一樣，所以開始前從 rank 0 broadcast 一次。

這叫 **DDP**（Distributed Data Parallel）。最簡單的版本：

```python,ignore
{{#include ../../../src/llmz/dist.py:naive_ddp}}
```

`p.data` 是參數底下的張量本身，對它做 broadcast 不會被 autograd 記錄（和第 2 章的 `no_grad` 一樣的用意）。

### 驗證：和單機看整個 batch 一樣

每個 rank 故意用不同的種子初始化，再 broadcast；rank 0 另外建一個模型，單機看整個 batch，比較梯度：

```python
import torch

from llmz.dist import all_reduce_gradients, broadcast_parameters, launch, print_in_order
from llmz.functional import cross_entropy
from llmz.model import ModelConfig, TransformerLM

CFG = ModelConfig(vocab_size=50, context_length=8, d_model=32, n_layers=2, n_heads=4, d_ff=64)


def loss_of(model, ids):
    return cross_entropy(model(ids[:, :-1]).reshape(-1, 50), ids[:, 1:].reshape(-1))


def check(rank, world_size):
    torch.manual_seed(0)
    ids = torch.randint(0, 50, (4, 9))  # 每個 rank 都產生同一個 batch
    torch.manual_seed(100)
    single = TransformerLM(CFG)  # 單機版本，和 rank 0 的初始參數相同
    torch.manual_seed(100 + rank)
    model = TransformerLM(CFG)
    broadcast_parameters(model)

    per_rank = 4 // world_size
    loss_of(model, ids[rank * per_rank : (rank + 1) * per_rank]).backward()
    all_reduce_gradients(model)
    loss_of(single, ids).backward()
    same = all(torch.allclose(p.grad, q.grad, atol=1e-6) for p, q in zip(model.parameters(), single.parameters()))
    print_in_order("和單機的梯度相同：", same)


if __name__ == "__main__":
    launch(check, 2)
```

執行結果：

```text
[rank 0] 和單機的梯度相同： True
[rank 1] 和單機的梯度相同： True
```

兩個 rank 各看 2 筆，all-reduce 後的梯度和單機看 4 筆完全一致（只差浮點誤差）。

### 訓練迴圈

DDP 的訓練迴圈只比第 2 章的五步多一行：

```python,norun
loss = loss_of(model, my_part_of_batch)
loss.backward()
all_reduce_gradients(model)  # 新增的一行
opt.step()
opt.zero_grad()
```

資料怎麼分是程式自己決定的。第 5 章的 `get_batch` 是隨機取起點，最簡單的做法是每個 rank 用不同的種子（例如 `seed + rank`）取自己的 batch。

### 慢在哪裡

標準模型有 39 個參數張量，這個版本每一步要呼叫 39 次 all-reduce，而且要等整個 backward 做完才開始。我們在 M4 Max 的 CPU 上用兩個 rank、標準模型（長度 128、每個 rank batch 8）量了兩輪，每步時間的中位數：

| 版本 | 每步（兩輪） |
| --- | --- |
| 不通訊（各算各的，只當比較基準） | 698、823 ms |
| naive DDP | 1105、1166 ms |

通訊讓每一步多了三到四成的時間。下一集用兩個技巧把它降下來。CPU 上的計時跳動很大（第 7 章第 2 集），所以只看相對大小。

## 重點整理

- DDP：每個 rank 算不同的資料，backward 後 all-reduce 梯度再除以 \\( N \\)，就等於整個 batch 的梯度。
- 開始前從 rank 0 broadcast 參數；之後每個 rank 用相同的梯度更新，參數自動保持一致。
- `llmz.dist.broadcast_parameters`、`all_reduce_gradients`；訓練迴圈只多一行。
- naive 版每個參數一次 all-reduce、等 backward 全部做完才開始，通訊時間全部疊加在計算之後。
