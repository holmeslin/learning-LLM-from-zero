# 數值穩定的 softmax

## 本集目標

確認第 1 章寫的 `softmax` 能處理 attention 會丟給它的兩種極端：很大的分數，以及用來遮住位置的 \\( -\infty \\)。

## 正文

### 大分數

第 1 章第 16 集寫的 `llmz.functional.softmax` 先減掉每一列的最大值再取 exp，所以分數上千也不會溢位。attention 的分數是點積，`d_head` 維的向量點積可以很大，這件事下一集還會再處理一次（除以 \\( \sqrt{d} \\)），但 softmax 本身先要站得住。

### 負無限大

第 10 集的 causal mask 會把「不准看」的位置分數設成 `-inf`。看看 softmax 怎麼處理：

```python
import torch

from llmz.functional import softmax

scores = torch.tensor([[2.0, 1.0, float("-inf"), float("-inf")],
                       [1000.0, float("-inf"), 999.0, float("-inf")]])
print(softmax(scores, dim=-1))
```

執行結果：

```text
tensor([[0.7311, 0.2689, 0.0000, 0.0000],
        [0.7311, 0.0000, 0.2689, 0.0000]])
```

`exp(-inf)` 是 0，所以被遮住的位置機率剛好是 0，其餘位置的機率照常加總為 1；減最大值時 `-inf - 1000` 還是 `-inf`，沒有問題。這就是我們「遮住」一個位置的方式：不是把它刪掉（形狀會變），而是讓它的機率變成 0。

### 整列都是 -inf 會怎樣

```python
import torch

from llmz.functional import softmax

print(softmax(torch.tensor([float("-inf"), float("-inf")])))
```

執行結果：

```text
tensor([nan, nan])
```

最大值是 `-inf`，`-inf - (-inf)` 是 `nan`，整列變 `nan`，接著毒死整個模型（第 1 章第 16 集說過 nan 會傳染）。所以 mask 有個鐵則：**每一列至少要留一個位置**。causal mask 永遠讓每個位置看到自己，自動滿足；第 11 章做 padding mask 時就得小心。

## 重點整理

- `llmz.functional.softmax` 減最大值再 exp，大分數不溢位。
- 遮位置的方法是把分數設成 `-inf`，softmax 後機率為 0，形狀不變。
- 一列全部 `-inf` 會變 `nan`；每列至少留一個位置。
