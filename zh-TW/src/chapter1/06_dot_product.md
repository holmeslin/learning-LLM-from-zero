# 向量與點積

## 本集目標

知道向量就是一維張量，會算兩個向量的點積，並理解點積代表「兩個向量有多像」。

## 正文

### 向量

一排數字就是一個向量。在 LLM 裡，每個 token（文字切成的小單位）都會變成一個向量，例如 512 個數字；模型要判斷兩個 token 有沒有關係，靠的就是向量之間的運算。這一集用三個數字的小向量練習。

### 點積：對應相乘再加總

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(a * b)
print((a * b).sum())
print(torch.dot(a, b))
print(a @ b)
```

執行結果：

```text
tensor([ 4., 10., 18.])
tensor(32.)
tensor(32.)
tensor(32.)
```

\\( 1 \times 4 + 2 \times 5 + 3 \times 6 = 32 \\)。三種寫法同一件事：手動 `*` 再 `.sum()`，或 `torch.dot`，或 `@`。本書之後都用 `@`，因為下一集它會長成矩陣乘法。

### 點積量的是「像不像」

兩個向量方向越接近，點積越大；垂直時是 0：

```python
import torch

cat = torch.tensor([1.0, 1.0, 0.0])
dog = torch.tensor([1.0, 0.8, 0.0])
car = torch.tensor([0.0, 0.0, 1.0])
print(cat @ dog)
print(cat @ car)
print(cat @ cat)
```

執行結果：

```text
tensor(1.8000)
tensor(0.)
tensor(2.)
```

`cat` 和 `dog` 的數字分布很像，點積大；`cat` 和 `car` 完全沒有重疊的成分，點積是 0。attention 判斷「這個字要看哪些字」用的就是點積（第 4 章）。

### 長度與 cosine 相似度

點積會受向量長度影響：把 `cat` 放大兩倍，點積也變兩倍，但「像不像」沒變。向量的長度用 `.norm()`：

```python
import torch

a = torch.tensor([3.0, 4.0])
b = torch.tensor([6.0, 8.0])
print(a.norm())
print(torch.sqrt((a * a).sum()))
print(a @ b)
print((a @ b) / (a.norm() * b.norm()))
```

執行結果：

```text
tensor(5.)
tensor(5.)
tensor(50.)
tensor(1.)
```

長度就是「自己和自己的點積再開根號」，\\( \sqrt{3^2 + 4^2} = 5 \\)。點積除以兩邊的長度，得到 cosine 相似度：`a` 與 `b` 方向完全相同，所以是 1；垂直是 0；方向相反是 -1。

LLM 裡常見的「除以某個長度」都是同一個想法：把尺度拿掉，只留下方向或相對大小。第 4 章的 attention 會把點積除以 \\( \sqrt{d} \\)，RMSNorm 會把向量除以自己的長度，到時候你會覺得眼熟。

## 重點整理

- 向量就是一維張量；點積是對應元素相乘再加總，`a @ b`。
- 點積越大代表兩個向量方向越接近，垂直是 0。
- `.norm()` 是向量長度；點積除以兩邊長度是 cosine 相似度，介於 -1 到 1。
