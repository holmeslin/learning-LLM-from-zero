# 索引與切片

## 本集目標

從張量裡取出一個元素、一段、一列或一欄，並且會用「索引張量」一次取出多個位置。

## 正文

### 一維：和 Python 串列一樣

```python
import torch

x = torch.arange(10, 20)
print(x)
print(x[0])
print(x[-1])
print(x[2:5])
print(x[:3])
print(x[7:])
```

執行結果：

```text
tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
tensor(10)
tensor(19)
tensor([12, 13, 14])
tensor([10, 11, 12])
tensor([17, 18, 19])
```

`torch.arange(10, 20)` 從 10 數到 19。索引從 0 開始，負數從後面數，切片 `a:b` 不含 `b`，省略就是到頭或到尾。這些規則和 Python 串列完全一樣，差別只在取出來的仍然是張量。

### 二維：用逗號分開列與欄

```python
import torch

m = torch.arange(12).reshape(3, 4)
print(m)
print(m[1])
print(m[1, 2])
print(m[:, 0])
print(m[:2, :2])
print(m[-1, -1])
```

執行結果：

```text
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
tensor([4, 5, 6, 7])
tensor(6)
tensor([0, 4, 8])
tensor([[0, 1],
        [4, 5]])
tensor(11)
```

- `m[1]`：第 1 列（從 0 數起，所以是第二橫排）。
- `m[1, 2]`：第 1 列第 2 欄那一格。
- `m[:, 0]`：所有列的第 0 欄，也就是第一直行。`:` 代表「這個維度全要」。
- `m[:2, :2]`：前兩列、前兩欄，左上角的 2×2。

印出來時 PyTorch 會把數字靠右對齊，所以個位數前面多了一個空格。

### 用索引張量一次取多個

中括號裡不只能放數字，也能放一個整數張量，一次取出好幾個位置：

```python
import torch

x = torch.tensor([10, 20, 30, 40])
idx = torch.tensor([3, 0, 0])
print(x[idx])
```

執行結果：

```text
tensor([40, 10, 10])
```

同一個位置可以重複取，順序照 `idx` 的順序。

二維時，給兩個索引張量，就是「第 i 個列索引配第 i 個欄索引」，每一對取一格：

```python
import torch

m = torch.arange(12).reshape(3, 4)
rows = torch.tensor([0, 1, 2])
cols = torch.tensor([3, 0, 2])
print(m[rows, cols])
```

執行結果：

```text
tensor([ 3,  4, 10])
```

取出的是 `m[0, 3]`、`m[1, 0]`、`m[2, 2]`。這個寫法後面算 loss 時會用到：每一列是一道題，`cols` 是每道題正確答案的位置，一行就把所有「正確答案那一格」撈出來。

### 索引也能用來修改

```python
import torch

x = torch.zeros(5)
x[0] = 1
x[2:4] = 7
print(x)
```

執行結果：

```text
tensor([1., 0., 7., 7., 0.])
```

## 重點整理

- 一維索引與切片的規則和 Python 串列相同；二維用逗號分開列與欄，`:` 代表整個維度。
- `m[1]` 是一整列，`m[:, 0]` 是一整欄，`m[1, 2]` 是一格。
- 中括號裡放整數張量可以一次取多個位置；二維給兩個索引張量就是一對一取格子。
- 索引可以放在等號左邊修改元素。
