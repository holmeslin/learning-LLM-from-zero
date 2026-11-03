# `einops` 速查

## 本集目標

認識 `einops` 套件：用一條字串描述 reshape、transpose、加總，比 `reshape(B, T, H, -1).transpose(1, 2)` 這種寫法好讀。本書正文不用它，但很多論文程式碼用。

## 正文

### 安裝

它在本書的開發相依裡（`uv sync` 會裝），正文的程式碼不依賴它。

### `rearrange`：reshape 與 transpose 一次做完

第 4 章第 11 集把 `(B, T, d_model)` 拆成 head 的寫法：

```python
import torch
from einops import rearrange

torch.manual_seed(0)
x = torch.randn(2, 5, 8)
a = x.reshape(2, 5, 4, 2).transpose(1, 2)
b = rearrange(x, "b t (h d) -> b h t d", h=4)
print(b.shape, torch.equal(a, b))
c = rearrange(b, "b h t d -> b t (h d)")
print(torch.equal(c, x))
```

執行結果：

```text
torch.Size([2, 4, 5, 2]) True
True
```

字串左邊是輸入的形狀、右邊是輸出，括號代表「這幾個維度合成一個」，`h=4` 告訴它怎麼拆。接回去就是把箭頭反過來。

### `reduce`：沿著某些維度加總或平均

```python
import torch
from einops import reduce

x = torch.arange(24.0).reshape(2, 3, 4)
print(torch.equal(reduce(x, "b t d -> b t", "mean"), x.mean(dim=-1)))
print(reduce(x, "b t d -> d", "max"))
```

執行結果：

```text
True
tensor([20., 21., 22., 23.])
```

輸出沒寫的維度就是被縮掉的維度，第三個引數說怎麼縮。

### `repeat`：broadcasting 寫成明的

```python
import torch
from einops import repeat

v = torch.tensor([1.0, 2.0, 3.0])
print(repeat(v, "d -> b d", b=2))
```

執行結果：

```text
tensor([[1., 2., 3.],
        [1., 2., 3.]])
```

### `einsum`

`einops.einsum` 和 `torch.einsum`（第 1 章第 8 集）一樣，只是維度可以用完整的名字而不是單一字母：

```python
import torch
from einops import einsum

torch.manual_seed(0)
q = torch.randn(2, 4, 8)
k = torch.randn(2, 6, 8)
a = einsum(q, k, "batch query d, batch key d -> batch query key")
print(a.shape, torch.allclose(a, q @ k.transpose(-2, -1)))
```

執行結果：

```text
torch.Size([2, 4, 6]) True
```

### 要不要用

好處是形狀寫在字串裡，一看就知道；壞處是多一個相依、`torch.compile` 偶爾要注意。本書選擇不用，讓讀者看到每個 reshape 與 transpose 實際在做什麼。讀別人的程式碼看到 `rearrange` 時，對照上面的表就能翻回來。

## 重點整理

- `rearrange(x, "b t (h d) -> b h t d", h=4)`：reshape 加 transpose 一行搞定。
- `reduce(x, "b t d -> b t", "mean")`：沒寫的維度被縮掉。
- `repeat(v, "d -> b d", b=2)`：明寫的 broadcasting。
- `einsum` 可以用完整的維度名稱。
