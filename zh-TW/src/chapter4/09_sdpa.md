# scaled dot-product attention

## 本集目標

寫出 attention 的核心公式：query 和每個 key 算點積、除以 \\( \sqrt{d} \\)、softmax 變成權重、對 value 加權平均。

## 正文

### 三個角色

每個位置有三個向量：

- **query**（q）：我在找什麼。
- **key**（k）：我有什麼可以被找。
- **value**（v）：被找到之後我提供什麼。

位置 i 的輸出 = 用 \\( q_i \\) 和所有 \\( k_j \\) 算相似度（點積），softmax 成權重，再把所有 \\( v_j \\) 照權重加起來：

\\[ \text{attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V \\]

Q 是 `(T, d)`，\\( QK^\top \\) 是 `(T, T)`：第 i 列第 j 欄是位置 i 看位置 j 的分數。softmax 沿最後一個維度（每一列加總為 1），再乘 `(T, d_v)` 的 V 得到 `(T, d_v)`。

### 為什麼除以 \\( \sqrt{d} \\)

```python
import torch

torch.manual_seed(0)
d = 64
q = torch.randn(1000, d)
k = torch.randn(1000, d)
scores = q @ k.T
print(round(scores.std().item(), 1), round((scores / d ** 0.5).std().item(), 2))
```

執行結果：

```text
8.0 1.0
```

d 維的點積是 d 個乘積的和，標準差約 \\( \sqrt{d} \\)：d = 64 就是 8。分數的差距動輒幾十，softmax 之後幾乎永遠只有一個位置是 1、其他是 0，梯度極小，學不動。除以 \\( \sqrt{d} \\) 讓標準差回到 1 的量級。

### 程式

```python,ignore
{{#include ../../../src/llmz/attention.py:sdpa}}
```

`k.transpose(-2, -1)` 是第 1 章第 8 集的 `transpose`，只換最後兩個維度，前面的 batch、head 維度保持（所以 q、k、v 前面可以有任意維度，和 Linear 一樣）。`mask` 是布林張量，`masked_fill(~mask, -inf)` 把 `False` 的位置設成 `-inf`，`~` 是布林取反；上一集驗證過 softmax 之後它們會是 0。沒給 mask 就全部互看。

```python
import torch
import torch.nn.functional as F

from llmz.attention import scaled_dot_product_attention

torch.manual_seed(0)
q = torch.randn(1, 3, 4)
k = torch.randn(1, 3, 4)
v = torch.randn(1, 3, 2)
out = scaled_dot_product_attention(q, k, v)
print(out.shape)
print(out)
print(torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-6))
```

執行結果：

```text
torch.Size([1, 3, 2])
tensor([[[-0.3632,  0.2235],
         [-0.3095,  0.0426],
         [-0.4159,  0.1540]]])
True
```

3 個位置、每個 query 4 維、value 2 維，輸出 `(1, 3, 2)`：每個位置拿到一個 2 維的 value 加權平均。最後一行拿 PyTorch 內建的 `F.scaled_dot_product_attention` 比對，只用來驗證（全書的規則），第 8 章會再回來談它的高效實作。

## 重點整理

- query 找、key 被找、value 被取：\\( \text{softmax}(QK^\top/\sqrt{d})V \\)。
- 除以 \\( \sqrt{d} \\) 讓分數標準差回到 1，softmax 才不會飽和。
- `scaled_dot_product_attention(q, k, v, mask)`：`transpose(-2, -1)` 做 \\( K^\top \\)，`masked_fill(~mask, -inf)` 遮位置。
- 前面可以有任意 batch 維度。
