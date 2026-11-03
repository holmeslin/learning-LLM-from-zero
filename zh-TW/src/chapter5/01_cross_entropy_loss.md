# cross-entropy loss 實作

## 本集目標

確認語言模型的 loss 就是第 1 章寫的 `cross_entropy`，知道 logits 怎麼攤平、答案怎麼對齊，以及訓練開始時 loss 應該是多少。

## 正文

### 已經寫好了

第 1 章第 16 集的 `llmz.functional.cross_entropy` 吃 `(N, V)` 的 logits 與 `(N,)` 的答案索引，回傳平均的 \\( -\log p_{\text{正確}} \\)。它內部先減最大值再算 log-softmax，所以模型輸出上千的 logits 也不會溢位。這就是全書的 loss，不用再寫。

### 對齊

模型吃 `(B, T)` 吐 `(B, T, V)`，位置 t 的 logits 猜 t+1。第 3 章第 14 集的 `get_batch` 會直接給對齊好的兩個張量（下面第 6 集），這裡先手動做一次：

```python
import torch

from llmz.functional import cross_entropy

torch.manual_seed(0)
B, T, V = 2, 4, 6
ids = torch.randint(0, V, (B, T + 1))
inputs, targets = ids[:, :-1], ids[:, 1:]
print(inputs)
print(targets)
logits = torch.randn(B, T, V)
loss = cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
print(loss)
```

執行結果：

```text
tensor([[2, 3, 5, 0],
        [3, 1, 1, 1]])
tensor([[3, 5, 0, 1],
        [1, 1, 1, 3]])
tensor(1.5623)
```

`targets` 就是 `inputs` 往左移一格。`reshape(-1, V)` 把 `(B, T, V)` 攤成 `(B×T, V)`，`targets.reshape(-1)` 攤成 `(B×T,)`，兩邊的順序一致（都是先第 0 段的 T 個位置、再第 1 段的），對得上。

### 起點

```python
import math

import torch

from llmz.functional import cross_entropy
from llmz.model import ModelConfig, TransformerLM

torch.manual_seed(0)
model = TransformerLM(ModelConfig(context_length=32))
ids = torch.randint(0, 10000, (4, 33))
logits = model(ids[:, :-1])
loss = cross_entropy(logits.reshape(-1, 10000), ids[:, 1:].reshape(-1))
print(round(loss.item(), 2), round(math.log(10000), 2))
```

執行結果：

```text
9.25 9.21
```

標準設定剛初始化的 loss 在 \\( \ln 10000 = 9.21 \\) 附近：每個 token 的機率都差不多是萬分之一。如果你的模型一開始 loss 遠大於這個數（例如 20），多半是初始化太大，logits 的差距一開始就很極端；這是檢查模型有沒有寫錯的第一個訊號。

訓練目標：把它從 9.2 壓到 1.4 左右（perplexity 從一萬降到 4）。

## 重點整理

- loss 就是 `llmz.functional.cross_entropy`，不用重寫。
- `targets = ids[:, 1:]`、`inputs = ids[:, :-1]`；logits `reshape(-1, V)`、targets `reshape(-1)`。
- 初始 loss ≈ \\( \ln V \\)；遠大於它代表初始化有問題。
