# 第一個張量

## 本集目標

認識 PyTorch 最基本的東西：張量（tensor），並且知道張量之間的運算是「對應位置各算各的」。

## 正文

### 張量就是一排數字

LLM 從頭到尾都在處理數字：文字會先變成數字，模型的參數是數字，模型的輸出也是數字。PyTorch 用**張量**（tensor）裝這些數字。最簡單的張量就是一排數字：

```python
import torch

x = torch.tensor([1, 2, 3])
print(x)
```

執行結果：

```text
tensor([1, 2, 3])
```

`torch.tensor(...)` 把 Python 的串列變成張量。印出來時 PyTorch 會在外面包一層 `tensor(...)`，提醒你這不是普通的串列。

### 運算是逐元素進行

張量和一個數字做運算，會套用到每一個元素上：

```python
import torch

x = torch.tensor([1, 2, 3])
print(x + 10)
print(x * 2)
print(x * x)
print(x.sum())
```

執行結果：

```text
tensor([11, 12, 13])
tensor([2, 4, 6])
tensor([1, 4, 9])
tensor(6)
```

`x * x` 是每個元素自己乘自己，不是什麼特別的矩陣運算（矩陣乘法要用另一個符號，第 7 集會教）。`x.sum()` 把所有元素加起來，結果 `tensor(6)` 是一個「只有一個數字」的張量。

### 兩個張量之間

形狀相同的兩個張量做運算，也是對應位置各算各的：

```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([10, 20, 30])
print(a + b)
print(a * b)
```

執行結果：

```text
tensor([11, 22, 33])
tensor([10, 40, 90])
```

### 表格也是張量

兩層中括號就是一張表格，PyTorch 叫它二維張量：

```python
import torch

m = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(m)
print(m * 10)
print(m.sum())
```

執行結果：

```text
tensor([[1, 2, 3],
        [4, 5, 6]])
tensor([[10, 20, 30],
        [40, 50, 60]])
tensor(21)
```

### 把數字拿出來

`tensor(21)` 還是張量。要拿回普通的 Python 數字（例如想用 f-string 格式化、或存到檔案），用 `.item()`：

```python
import torch

m = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
total = m.sum().item()
print(total)
print(f"總和是 {total}，型別是 {type(total).__name__}")
```

執行結果：

```text
21
總和是 21，型別是 int
```

`.item()` 只能用在只有一個數字的張量上。

## 重點整理

- `torch.tensor([...])` 把串列變成張量；兩層中括號就是二維張量（表格）。
- 張量與數字、或兩個同形狀的張量做 `+ - * /`，都是對應位置各算各的。
- `.sum()` 把全部元素加總，結果仍是張量；`.item()` 把單一數字的張量變回 Python 數字。
