# causal mask

## 本集目標

寫出 `causal_mask`：讓位置 i 只能看 0 到 i，模型預測下一個 token 時才不會偷看答案。

## 正文

### 不准看未來

第 1 集說過，位置 t 的輸出是在猜 t+1 的 token，而 t+1 的 token 就在輸入裡。沒有限制的話，attention 會直接學會「看右邊那個」，loss 降到 0、什麼都沒學到。所以 attention 的分數矩陣 `(T, T)` 要遮掉右上半邊：第 i 列只留 j ≤ i。

```python,ignore
{{#include ../../../src/llmz/attention.py:causal_mask}}
```

`torch.tril` 保留矩陣的下三角（含對角線），其餘設 0；先做一個全 `True` 的布林矩陣再取下三角，`True` 代表准看。

```python
from llmz.attention import causal_mask

print(causal_mask(4))
```

執行結果：

```text
tensor([[ True, False, False, False],
        [ True,  True, False, False],
        [ True,  True,  True, False],
        [ True,  True,  True,  True]])
```

第 0 列只有自己、第 3 列看全部。對角線永遠是 `True`，所以每列至少有一個位置，上一集的 `nan` 不會發生。

### 接上 attention

```python
import torch

from llmz.attention import causal_mask, scaled_dot_product_attention
from llmz.functional import softmax

torch.manual_seed(0)
q = torch.randn(4, 8)
k = torch.randn(4, 8)
v = torch.randn(4, 2)
mask = causal_mask(4)
scores = (q @ k.T / 8 ** 0.5).masked_fill(~mask, float("-inf"))
print(softmax(scores, dim=-1))
out = scaled_dot_product_attention(q, k, v, mask)
print(out[0], v[0])
```

執行結果：

```text
tensor([[1.0000, 0.0000, 0.0000, 0.0000],
        [0.4929, 0.5071, 0.0000, 0.0000],
        [0.6221, 0.0318, 0.3461, 0.0000],
        [0.2032, 0.6235, 0.1025, 0.0708]])
tensor([0.1920, 0.5428]) tensor([0.1920, 0.5428])
```

權重矩陣的右上角全是 0：位置 0 只有一個選擇，權重 1；位置 3 分給四個。`out[0]` 和 `v[0]` 一模一樣，因為位置 0 只能看自己。

### 訓練與生成的一致

這個 mask 讓「一次把整段文字餵進去」和「一個一個 token 生成」算出來的東西相同：生成時位置 t 之後的 token 本來就不存在，訓練時用 mask 假裝它們不存在。所以訓練可以平行處理 T 個位置，生成時模型的行為又和訓練一致，第 6 章會用到這一點。

## 重點整理

- `causal_mask(T)`：`torch.tril` 做出的下三角布林矩陣，位置 i 只准看 j ≤ i。
- 沒有它模型會直接抄下一個 token。
- 對角線永遠准看，不會整列被遮。
- 訓練時平行處理所有位置，和逐個生成的結果一致。
