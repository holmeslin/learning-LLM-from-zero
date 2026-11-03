# backward 與 recompute

## 本集目標

推出 attention 的 backward 公式，寫成分塊的版本：不存機率矩陣，backward 時用 logsumexp 把它一塊一塊重算回來；再用 `torch.autograd.Function` 接上 autograd。

## 正文

### attention 的梯度

把一個 head 的 attention 拆成三步（省略 batch 與 head）：

\\[ S = \frac{QK^\top}{\sqrt{d}}, \quad P = \mathrm{softmax}(S), \quad O = PV \\]

backward 拿到的是 loss 對輸出的梯度 \\( dO \\)（形狀和 \\( O \\) 一樣），要算出 \\( dQ \\)、\\( dK \\)、\\( dV \\)。照第 1 章的鏈鎖律一步一步往回推：

- \\( O = PV \\) 是矩陣乘法：\\( dV = P^\top dO \\)，\\( dP = dO\, V^\top \\)。
- softmax 的 backward：\\( dS_{ij} = P_{ij} (dP_{ij} - D_i) \\)，其中 \\( D_i = \sum_j P_{ij} dP_{ij} \\)。
- \\( S = QK^\top / \sqrt{d} \\)：\\( dQ = dS\, K / \sqrt{d} \\)，\\( dK = dS^\top Q / \sqrt{d} \\)。

softmax 那一行看起來需要整列的 \\( P \\) 才能算 \\( D_i \\)，但它可以換個算法：

\\[ D_i = \sum_j P_{ij}\, (dO_i \cdot V_j) = dO_i \cdot \sum_j P_{ij} V_j = dO_i \cdot O_i \\]

\\( O \\) 是 forward 的輸出，本來就有。所以 \\( D \\) 一開始用 `(do * o).sum(-1)` 一次算好，之後每一塊都不用看整列。

### 重算機率

backward 每一小塊都需要那一塊的 \\( P \\)。我們沒存它，但存了每一列的 logsumexp \\( L_i \\)，而 softmax 正好就是

\\[ P_{ij} = \frac{e^{S_{ij}}}{\sum_k e^{S_{ik}}} = e^{S_{ij} - L_i} \\]

所以重算一小塊分數 \\( S \\)，再 `exp(s - lse)`，就得到那一塊精確的機率，不需要再做一次 online softmax。代價是多算一次 \\( QK^\top \\)，和第 7 章第 6 集的 activation checkpointing 一樣是用計算換記憶體；FlashAttention 的情況更划算，因為省下的搬運時間比多出的計算還多。

```python,ignore
{{#include ../../../src/llmz/flash.py:backward}}
```

迴圈順序和 forward 反過來：外層是 key 塊，內層是 query 塊。每一對小塊對 `dq`、`dk`、`dv` 各有一份貢獻，用 `+=` 加上去。causal 時，一個 key 塊只會被它所在位置之後的 query 塊看到，所以內層從 `j // block_q * block_q` 開始。

### 接上 autograd

第 2 章以來，backward 都是 autograd 自動推的。這次我們要自己提供 backward，寫法是繼承 `torch.autograd.Function`，定義兩個 `@staticmethod`：

```python,ignore
{{#include ../../../src/llmz/flash.py:function}}
```

- `forward(ctx, ...)`：正常計算，用 `ctx.save_for_backward` 存 backward 要的張量。這裡只存 `q`、`k`、`v`、`o`、`lse`，都跟 \\( T \\) 成正比。
- `backward(ctx, do)`：拿到輸出的梯度，回傳每個輸入的梯度，順序與 `forward` 的參數相同；`causal` 不是張量，回傳 `None`。
- 使用時呼叫 `FlashAttention.apply(...)`，不是直接呼叫 `forward`。

`autograd.Function` 是 autograd 的擴充介面，不是現成的層，本書的規則允許使用。

### 驗證梯度與記憶體

```python
import torch

from llmz.attention import causal_mask, scaled_dot_product_attention
from llmz.flash import flash_attention
from llmz.profile import saved_activation_bytes

torch.manual_seed(0)
q, k, v = (torch.randn(2, 4, 200, 32, requires_grad=True) for _ in range(3))
weights = torch.randn(2, 4, 200, 32)  # 讓 loss 對每個輸出位置的梯度都不一樣

(flash_attention(q, k, v, True) * weights).sum().backward()
ours = [t.grad.clone() for t in (q, k, v)]
for t in (q, k, v):
    t.grad = None
(scaled_dot_product_attention(q, k, v, causal_mask(200)) * weights).sum().backward()
print([torch.allclose(a, t.grad, atol=1e-4) for a, t in zip(ours, (q, k, v))])

for T in [256, 1024, 2048]:
    q, k, v = (torch.randn(1, 16, T, 32, requires_grad=True) for _ in range(3))
    saved = saved_activation_bytes(lambda: flash_attention(q, k, v, True).sum(), exclude=[q, k, v])
    print(f"T={T:5}  {saved / 1e6:.2f} MB")
```

執行結果：

```text
[True, True, True]
T=  256  0.54 MB
T= 1024  2.16 MB
T= 2048  4.33 MB
```

三個梯度都和 autograd 推出來的一樣。存下來的只有 `o`（`16 × T × 32` 個 float32）和 `lse`，長度加倍就加倍。和第 1 集比：

| 長度 | 第 4 章的 attention | 分塊版本 |
| --- | --- | --- |
| 256 | 8.5 MB | 0.54 MB |
| 1024 | 135.5 MB | 2.16 MB |
| 2048 | 541.5 MB | 4.33 MB |

從平方變成線性。長度 2048 時差了 125 倍，而且長度越長差越多。

## 重點整理

- attention backward：\\( dV = P^\top dO \\)、\\( dP = dO\,V^\top \\)、\\( dS = P \circ (dP - D) \\)、\\( dQ = dS\,K/\sqrt{d} \\)、\\( dK = dS^\top Q/\sqrt{d} \\)。
- \\( D_i = dO_i \cdot O_i \\)，不需要整列的 \\( P \\)。
- 用 forward 存下的 logsumexp 重算 \\( P = e^{S - L} \\)，不存機率矩陣，代價是多算一次 \\( QK^\top \\)。
- `torch.autograd.Function` 自訂 forward／backward，`ctx.save_for_backward` 存張量，用 `.apply` 呼叫。
- backward 要存的東西從 \\( O(T^2) \\) 降成 \\( O(T) \\)。
