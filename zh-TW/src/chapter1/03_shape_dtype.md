# 形狀與 dtype

## 本集目標

看懂張量的形狀（shape）與元素型別（dtype），會用 `zeros`、`ones`、`arange` 建立張量，並用 `reshape` 改變形狀。

## 正文

### 形狀

每個張量都有形狀，告訴你每一個維度有幾個元素：

```python
import torch

x = torch.tensor([1, 2, 3])
m = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(x.shape)
print(m.shape)
```

執行結果：

```text
torch.Size([3])
torch.Size([2, 3])
```

`torch.Size([2, 3])` 唸作「2 列 3 欄」：第一個數字是列數，第二個是欄數。形狀裡有幾個數字，就是幾維。LLM 裡的張量常常是三維，例如「第幾句話、第幾個字、第幾個特徵」，讀法一樣，從外層數到內層。

讀錯誤訊息時，**形狀**是你第一個要看的東西。PyTorch 大部分的錯誤都是形狀對不上。

### 建立常用的張量

不用每個數字都手打：

```python
import torch

print(torch.zeros(2, 3))
print(torch.ones(3))
print(torch.arange(6))
```

執行結果：

```text
tensor([[0., 0., 0.],
        [0., 0., 0.]])
tensor([1., 1., 1.])
tensor([0, 1, 2, 3, 4, 5])
```

- `torch.zeros(2, 3)`：2 列 3 欄全部是 0。括號裡直接寫形狀。
- `torch.ones(3)`：3 個 1。
- `torch.arange(6)`：從 0 數到 5，和 Python 的 `range` 一樣不含結尾。

注意 `0.` 後面那個小數點：`zeros` 與 `ones` 給的是浮點數，`arange(6)` 給的是整數。這就是接下來要講的 dtype。

### reshape

同樣 6 個數字，可以排成不同的形狀：

```python
import torch

x = torch.arange(6)
print(x.reshape(2, 3))
print(x.reshape(3, 2))
print(x.reshape(-1, 3).shape)
```

執行結果：

```text
tensor([[0, 1, 2],
        [3, 4, 5]])
tensor([[0, 1],
        [2, 3],
        [4, 5]])
torch.Size([2, 3])
```

`reshape` 不改變數字，只改變排法，所以新形狀的元素總數必須和原來一樣。寫 `-1` 代表「這個維度你幫我算」：6 個數字、每列 3 個，那就是 2 列。

### dtype：整數還是浮點數

dtype 是張量裡每個元素的型別。最常碰到的兩種：

```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([1.0, 2.0, 3.0])
print(a.dtype)
print(b.dtype)
print(a.float().dtype)
print(a / 2)
```

執行結果：

```text
torch.int64
torch.float32
torch.float32
tensor([0.5000, 1.0000, 1.5000])
```

- 串列裡全是整數，就是 `int64`（64 位元整數）。
- 有小數點就是 `float32`（32 位元浮點數）。模型的參數、機率、loss 全部都是浮點數。
- `.float()` 把整數張量轉成 `float32`。
- 整數張量除以 2 會自動變成浮點數，PyTorch 印浮點數時預設顯示四位小數。

有些運算只接受浮點數。例如對整數張量取平均：

```python,error
import torch

a = torch.tensor([1, 2, 3])
print(a.mean())
```

錯誤訊息：

```text
RuntimeError: mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Long
```

訊息裡的 `Long` 就是 `int64` 的另一個名字。看到這類錯誤，先 `.float()` 再算就好。第 7 章會再認識更省記憶體的 `bfloat16`。

## 重點整理

- `.shape` 是形狀，`torch.Size([2, 3])` 唸作 2 列 3 欄；遇到錯誤先看形狀。
- `torch.zeros`、`torch.ones` 直接寫形狀；`torch.arange(n)` 產生 0 到 n-1。
- `reshape` 只改排法不改數字，元素總數要相同；`-1` 讓 PyTorch 自己算。
- `.dtype`：整數是 `int64`，浮點數是 `float32`；很多運算只接受浮點數，用 `.float()` 轉換。
