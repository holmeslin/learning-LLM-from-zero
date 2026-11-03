# 梯度裁剪

## 本集目標

寫出 `clip_grad_norm_`：把所有參數的梯度合起來看成一個向量，長度超過上限就等比例縮小，避免偶爾一個超大的梯度把模型推壞。

## 正文

### 偶爾會爆

訓練幾千步裡，總有幾步的梯度特別大：剛好抽到一批奇怪的資料、或 attention 的分數在某個位置飆高。AdamW 用 \\( \sqrt{v} \\) 除一下有緩衝，但 \\( v \\) 是移動平均，跟不上突然一次的暴衝。一步走太遠，loss 會突然跳上去，有時再也降不回來。

### 整體長度

做法：把**所有**參數的梯度攤平接成一個超長向量，算它的長度（第 1 章第 6 集的 `.norm()`，也就是平方和開根號）。長度超過 `max_norm` 就把每個梯度乘上 `max_norm / 長度`，方向不變、長度剛好等於上限；沒超過就不動。

```python,ignore
{{#include ../../../src/llmz/optim.py:clip_grad_norm}}
```

`sum((g * g).sum() for g in grads)` 是所有梯度的平方和，開根號就是整體長度。`g *= ...` 就地縮小，`@torch.no_grad()` 和第 2 集一樣。回傳裁剪前的長度，訓練時記進 log，看它什麼時候飆高很有用。

```python
import torch
from torch import nn

from llmz.optim import clip_grad_norm_

a = nn.Parameter(torch.zeros(2))
b = nn.Parameter(torch.zeros(1))
a.grad = torch.tensor([3.0, 4.0])
b.grad = torch.tensor([12.0])
norm = clip_grad_norm_([a, b], max_norm=1.0)
print(norm)
print(a.grad, b.grad)
print(torch.sqrt((a.grad ** 2).sum() + (b.grad ** 2).sum()))
```

執行結果：

```text
tensor(13.)
tensor([0.2308, 0.3077]) tensor([0.9231])
tensor(1.0000)
```

裁剪前長度 \\( \sqrt{9 + 16 + 144} = 13 \\)，每個梯度除以 13，裁剪後整體長度是 1，`a.grad` 和 `b.grad` 的比例沒變。

### 放在哪一步

在 `backward()` 之後、`step()` 之前：

```python,ignore
loss.backward()
grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad()
```

本書 `max_norm` 用 1.0。正常訓練時梯度長度多半在 0.2 到 1 之間，所以大部分步數不會被裁，只攔偶發的那幾步。

## 重點整理

- 所有梯度合成一個向量算長度，超過 `max_norm` 就等比例縮到剛好 `max_norm`，方向不變。
- `clip_grad_norm_(params, max_norm)` 回傳裁剪前的長度，拿來看訓練有沒有異常。
- 位置：`backward()` 之後、`step()` 之前；本書 `max_norm = 1.0`。
