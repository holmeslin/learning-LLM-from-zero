# broadcasting

## 本集目標

搞懂兩個形狀不同的張量怎麼一起運算（broadcasting），學會 `dim` 與 `keepdim`，並看懂形狀對不上的錯誤訊息。

## 正文

### 你已經在用它了

第 2 集的 `x + 10` 就是 broadcasting：一個數字被「複製」到每個位置，跟整個張量相加。這一集把規則講清楚，因為 LLM 的程式碼裡到處都是它。

### 一列對整張表

```python
import torch

m = torch.arange(12).reshape(3, 4)
row = torch.tensor([0, 10, 20, 30])
print(m + row)
```

執行結果：

```text
tensor([[ 0, 11, 22, 33],
        [ 4, 15, 26, 37],
        [ 8, 19, 30, 41]])
```

`m` 是 `(3, 4)`，`row` 是 `(4,)`。PyTorch 把 `row` 複製成三列，和 `m` 的每一列相加。

### 規則

兩個張量要一起運算時，PyTorch 從**最後一個維度**開始往前比對，每一組維度要符合其中一個條件：

1. 兩邊一樣大，或
2. 其中一邊是 1（或根本沒有這個維度）。那一邊會被複製到跟另一邊一樣大。

剛才的例子：`(3, 4)` 對 `(4,)`。最後一維 4 對 4，符合；往前，`row` 沒有第二個維度，視為 1，複製成 3。所以結果是 `(3, 4)`。

### 一欄對整張表

想對「每一列」各加不同的數，`(3,)` 直接加會失敗（最後一維 4 對 3，對不上）。要先把它變成 `(3, 1)`：

```python
import torch

m = torch.arange(12).reshape(3, 4)
col = torch.tensor([0, 100, 200]).reshape(3, 1)
print(col)
print(m + col)
```

執行結果：

```text
tensor([[  0],
        [100],
        [200]])
tensor([[  0,   1,   2,   3],
        [104, 105, 106, 107],
        [208, 209, 210, 211]])
```

`(3, 4)` 對 `(3, 1)`：最後一維 4 對 1，把 1 複製成 4；前一維 3 對 3。結果 `(3, 4)`。

### 對不上就報錯

```python,error
import torch

m = torch.arange(12).reshape(3, 4)
col = torch.tensor([0, 100, 200])
print(m + col)
```

錯誤訊息：

```text
RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 1
```

訊息直接告訴你：在第 1 維，一邊是 4、一邊是 3，兩邊都不是 1，所以不能 broadcasting。這是 PyTorch 最常見的錯誤之一，看到就去檢查兩邊的 `.shape`。

### `dim` 與 `keepdim`

`.sum()` 預設把全部加總。給它 `dim` 就只沿著某個維度加：

```python
import torch

m = torch.arange(12).reshape(3, 4).float()
print(m.sum(dim=0))
print(m.sum(dim=1))
print(m.sum(dim=1, keepdim=True))
```

執行結果：

```text
tensor([12., 15., 18., 21.])
tensor([ 6., 22., 38.])
tensor([[ 6.],
        [22.],
        [38.]])
```

- `dim=0`：沿著第 0 維（列）加，把三列壓成一列，每一欄一個總和。
- `dim=1`：沿著第 1 維（欄）加，每一列一個總和。被加掉的那個維度消失，形狀從 `(3, 4)` 變 `(3,)`。
- `keepdim=True`：被加掉的維度保留成 1，形狀是 `(3, 1)`。

為什麼要 `keepdim`？因為接下來常常要「每一列除以該列的總和」，`(3, 1)` 才能和 `(3, 4)` broadcasting：

```python
import torch

m = torch.arange(12).reshape(3, 4).float()
p = m / m.sum(dim=1, keepdim=True)
print(p)
print(p.sum(dim=1))
```

執行結果：

```text
tensor([[0.0000, 0.1667, 0.3333, 0.5000],
        [0.1818, 0.2273, 0.2727, 0.3182],
        [0.2105, 0.2368, 0.2632, 0.2895]])
tensor([1., 1., 1.])
```

每一列現在加總都是 1。第 13 集的 softmax 做的就是這件事，只是先對每個數取 exp。

`dim=-1` 代表「最後一個維度」，本書之後幾乎都這樣寫，因為不管張量是二維還是三維，最後一維通常就是我們要處理的那一維。

## 重點整理

- broadcasting 從最後一個維度往前對：每一組要嘛一樣大，要嘛其中一邊是 1（或不存在），1 的那邊會被複製。
- `(3, 4)` 可以配 `(4,)`，也可以配 `(3, 1)`，但不能配 `(3,)`。
- `.sum(dim=1)` 沿著欄加，每列一個數，那個維度會消失；`keepdim=True` 保留成 1，方便接著 broadcasting。
- 看到 `The size of tensor a (4) must match the size of tensor b (3)`，就是形狀對不上。
