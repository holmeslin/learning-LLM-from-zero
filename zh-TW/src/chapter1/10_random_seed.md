# 隨機數與種子

## 本集目標

會產生三種常用的隨機張量，知道「種子」讓隨機結果可以重現，以及為什麼訓練 LLM 離不開隨機數。

## 正文

### 三種隨機張量

```python
import torch

torch.manual_seed(0)
print(torch.rand(3))
print(torch.randn(3))
print(torch.randint(0, 10, (5,)))
```

執行結果：

```text
tensor([0.4963, 0.7682, 0.0885])
tensor([0.3643, 0.1344, 0.1642])
tensor([6, 6, 9, 8, 6])
```

- `torch.rand(3)`：3 個介於 0 到 1 之間的均勻隨機數。
- `torch.randn(3)`：3 個「常態分布」的隨機數，大部分落在 -2 到 2 之間，平均 0。模型參數的初始值就是這樣來的。
- `torch.randint(0, 10, (5,))`：5 個 0 到 9 的隨機整數，注意形狀要用括號 `(5,)` 包起來。訓練時隨機挑資料會用到。

### 種子

隨機數其實是電腦照一條固定規則算出來的一長串數字，**種子**（seed）決定從哪裡開始算。設同一個種子，之後拿到的隨機數就一模一樣：

```python
import torch

torch.manual_seed(0)
a = torch.randn(3)
torch.manual_seed(0)
b = torch.randn(3)
print(a)
print(b)
print(torch.equal(a, b))
```

執行結果：

```text
tensor([ 1.5410, -0.2934, -2.1788])
tensor([ 1.5410, -0.2934, -2.1788])
True
```

`torch.equal` 檢查兩個張量是不是完全相同。注意上一段程式先呼叫了 `rand(3)` 才呼叫 `randn(3)`，所以那裡的 `randn` 結果和這裡不同：種子只決定起點，之後每一次呼叫都會往下取。

### 為什麼需要種子

訓練 LLM 有三個地方是隨機的：參數的初始值、每一步挑哪些資料、生成文字時抽哪個字。沒固定種子，同一支程式每次跑結果都不同，你就分不清「loss 變好是因為我改對了，還是運氣好」，也沒辦法重現 bug。

所以本書的慣例是：**每支用到隨機數的程式開頭都寫 `torch.manual_seed(0)`**，書上的執行結果就是這樣在 CPU 上算出來的。CPU 上同一個種子在每台機器都會給同樣的數字；mps 用的是另一套產生器，數字會不同，這不是錯。

### 看看 `randn` 長什麼樣

```python
import torch

torch.manual_seed(0)
x = torch.randn(10000)
print(x.mean())
print(x.std())
print((x.abs() < 2).float().mean())
```

執行結果：

```text
tensor(-0.0107)
tensor(1.0019)
tensor(0.9531)
```

一萬個 `randn`：平均接近 0，標準差（`std`，衡量分散程度）接近 1，大約 95% 的值落在 -2 到 2 之間。第 4 章初始化參數時，我們會刻意把標準差調成某個特定的大小，到時候這些數字會再出現。

最後一行是個常用小技巧：`x.abs() < 2` 得到一排 `True`/`False`，`.float()` 把它們變成 1 和 0，再取平均就是「符合條件的比例」。

## 重點整理

- `torch.rand` 是 0～1 均勻隨機，`torch.randn` 是平均 0、標準差 1 的常態隨機，`torch.randint(低, 高, 形狀)` 是隨機整數。
- `torch.manual_seed(n)` 固定起點，之後的隨機數每次都一樣；每次呼叫會繼續往下取。
- 本書每支用隨機數的程式都先 `torch.manual_seed(0)`，輸出在 CPU 上可重現；mps 的隨機數不同。
- `(條件).float().mean()` 算符合條件的比例。
