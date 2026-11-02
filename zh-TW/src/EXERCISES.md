# EXERCISES.md — 練習題規則與固定題庫

這份文件給 AI 助教使用，說明什麼時候可以出題、怎麼出、怎麼批改，並收錄第 1 章的固定題目。

本文件裡的題目分成兩種，請不要混在一起：

- **固定題**：寫在本文件「第 1 章」底下、狀態為 `可用` 的題目。
- **臨時題**：本文件沒有合適的固定題時，AI 依下方規則當場設計的題目。

臨時題只能在下方規則允許時出。固定題是用來對齊程度的，不是讓你延伸出一整套更難、更長、更像一般深度學習課程的題目。

## 出題原則

1. 出題前一定要知道讀者讀到第幾章第幾集（問法見 `GUIDE.md` 第 2 節）。
2. 先在本文件找讀者目前進度以內、離他最近的那一集條目，看它的狀態。
3. 狀態是 `可用`：從「題目」中挑**一題**給讀者。同一條目列了好幾題時，它們是彼此替代的選項，不是要一次做完的題組。
4. 狀態是 `不出題`：用一句話轉述條目裡的原因，再推薦往前最近的 `可用` 條目，或建議讀者繼續往下讀。
5. 找不到合適的固定題，或讀者明確想要超前的挑戰，才出臨時題。臨時題一定要說明它是臨時題，不能冒充本文件的題目。
6. 臨時題裡讀者需要理解或自己寫的部分，都要在目前進度以內（`GUIDE.md` 第 4 節）。讀者只是說「想要難一點」，不代表同意用還沒學的東西。
7. 固定句型（`GUIDE.md` 第 4.2 節）可以直接用，但不要把固定句型背後的原理當成練習目標。
8. 題目好像非用到還沒教的函式不可時，先想辦法改題目。真的改不掉，就把那段程式碼獨立放在一個區塊，標題寫「先照抄的部分（之後才會教，請不要修改）」，用一句話說明它的作用，然後清楚告訴讀者他只需要寫哪一段。
9. 張量一律用小尺寸（3～4 個元素），讓讀者能用手驗算答案。

## 固定題庫使用規則

1. 出題時只給「題目」。「練習目標」「批改重點」「參考答案」是給你用的，不要先貼給讀者。
2. 讀者卡住時，先給一個提示（「批改重點」裡的第一點通常就是），不要直接給答案。
3. 讀者交出答案後照 `GUIDE.md` 第 3.3 節批改；他要求看參考答案、或已經做對時，才貼參考答案。
4. 參考答案只是其中一種寫法；讀者的寫法能得到同樣輸出就算對。

## 臨時題規則

- 一題只練一個剛學的概念，程式碼在 10 行以內。
- 題目要說清楚輸入是什麼、預期印出什麼。
- 出題後附上你自己跑過（或手算過）的預期輸出，讓讀者能對答案。
- 不要出需要訓練、需要資料檔、需要 GPU 的題目；第 1 章讀者的題目只用 CPU 上的小張量。

---

## 第 1 章 張量與剛好夠用的數學

### 第 1 集 安裝 uv 與取得程式碼 — 狀態：不出題

這集只是裝工具與跑通 `uv run`。讀者如果跑不起來，照 `GUIDE.md` 第 3.4 節 debug。

### 第 2 集 第一個張量 — 狀態：可用

**題目**

1. 建立張量 `[2, 4, 6]`，印出它乘以 10 的結果，以及所有元素的總和。
2. 建立兩個張量 `[1, 2, 3]` 與 `[10, 20, 30]`，印出相加與相乘的結果。

**練習目標**：會用 `torch.tensor` 建立張量，知道運算是逐元素進行。

**批改重點**：有沒有 `import torch`；結果是不是張量（印出來有 `tensor(...)`）；相乘是逐元素，不是矩陣乘法。

**參考答案**（第 1 題）

```python
import torch

x = torch.tensor([2, 4, 6])
print(x * 10)
print(x.sum())
```

### 第 3 集 形狀與 dtype — 狀態：可用

**題目**

1. 用 `torch.arange` 做出 0 到 11 的整數，reshape 成 3 列 4 欄，印出張量與它的 `shape`。
2. 建立 `torch.tensor([1, 2, 3])`，印出 `dtype`；把它轉成浮點數再印一次 `dtype`。

**練習目標**：看懂 `shape`，知道整數與浮點數張量是不同的 `dtype`。

**批改重點**：`reshape(3, 4)` 的元素總數要對；轉浮點數用 `.float()` 或 `.to(torch.float32)` 都可以。

**參考答案**（第 1 題）

```python
import torch

x = torch.arange(12).reshape(3, 4)
print(x)
print(x.shape)
```

### 第 4 集 索引與切片 — 狀態：可用

**題目**

給定 `x = torch.arange(12).reshape(3, 4)`：印出第 2 列（索引 1）、最後一欄、以及左上角 2×2 的方塊。

**練習目標**：用逗號分開列與欄的索引，`:` 代表整個維度，負數索引從後面數。

**批改重點**：`x[1]`、`x[:, -1]`、`x[:2, :2]`；切片的結尾不包含。

**參考答案**

```python
import torch

x = torch.arange(12).reshape(3, 4)
print(x[1])
print(x[:, -1])
print(x[:2, :2])
```

### 第 5 集 broadcasting — 狀態：可用

**題目**

有一個 3×4 的張量 `x = torch.arange(12).reshape(3, 4)`，想把每一欄各減掉不同的數 `[0, 10, 20, 30]`。寫出來並印出結果；再說明如果要「每一列各減掉不同的數 `[0, 100, 200]`」，那個張量要先 reshape 成什麼形狀。

**練習目標**：知道 broadcasting 從最後一個維度往前對齊，形狀 `(4,)` 對得上 `(3, 4)` 的欄，`(3, 1)` 才對得上列。

**批改重點**：第二問要 `reshape(3, 1)`；直接用 `(3,)` 會報錯，可以請讀者試試看錯誤訊息。

**參考答案**

```python
import torch

x = torch.arange(12).reshape(3, 4)
print(x - torch.tensor([0, 10, 20, 30]))
print(x - torch.tensor([0, 100, 200]).reshape(3, 1))
```

### 第 6 集 向量與點積 — 狀態：可用

**題目**

`a = torch.tensor([1.0, 2.0, 3.0])`、`b = torch.tensor([4.0, 5.0, 6.0])`。先用 `*` 與 `.sum()` 算出點積，再用 `torch.dot` 算一次，確認兩者相同。

**練習目標**：點積就是「對應元素相乘再加總」。

**批改重點**：兩種算法結果都是 32；整數張量 `torch.dot` 也可以，但後面的章節都用浮點數，順便養成習慣。

**參考答案**

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print((a * b).sum())
print(torch.dot(a, b))
```

### 第 7 集 矩陣乘法 `@` — 狀態：可用

**題目**

`A` 是 2×3、`B` 是 3×2，都用 `torch.arange` 加 `reshape` 做出來。印出 `A @ B` 與它的形狀，再試著算 `A @ A`，觀察錯誤訊息並用一句話解釋為什麼不行。

**練習目標**：`(n, k) @ (k, m)` 得到 `(n, m)`，內側的維度要一樣。

**批改重點**：`A @ B` 是 2×2；`A @ A` 會報 `mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)`，因為 3 對不上 2。

**參考答案**（不含錯誤那行）

```python
import torch

A = torch.arange(6).reshape(2, 3)
B = torch.arange(6).reshape(3, 2)
print(A @ B)
print((A @ B).shape)
```

### 第 8 集 `einsum` — 狀態：可用

**題目**

用 `torch.einsum` 寫出：(1) 兩個向量的點積；(2) 一個 2×3 矩陣的轉置；(3) 一個 2×3 矩陣每一列的總和。每一個都用 `@`、`.T` 或 `.sum()` 驗證結果相同。

**練習目標**：看懂 `einsum` 的下標字串：重複出現的字母會被加總，輸出沒寫的字母也會被加總。

**批改重點**：`"i,i->"`、`"ij->ji"`、`"ij->i"`。

**參考答案**

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
M = torch.arange(6.0).reshape(2, 3)
print(torch.einsum("i,i->", a, a), torch.dot(a, a))
print(torch.einsum("ij->ji", M), M.T, sep="\n")
print(torch.einsum("ij->i", M), M.sum(dim=1))
```

### 第 9 集 device：cpu 與 mps — 狀態：不出題

這集是固定句型 `get_device()` 與 `.to(device)`，沒有可練的邏輯。讀者想確認環境時，請他印出 `get_device()` 的結果就好。

### 第 10 集 隨機數與種子 — 狀態：可用

**題目**

寫一支程式：設定 `torch.manual_seed(0)`，印出 `torch.randn(3)`；再設定一次同樣的種子，再印一次。觀察兩次是否相同，然後把第二次的種子改成 1，看看結果。

**練習目標**：同一個種子會產生同一串隨機數；種子不同結果就不同。

**批改重點**：種子要在**每次**產生之前重設；讀者的輸出與書上的 CPU 結果應該一致。

**參考答案**

```python
import torch

torch.manual_seed(0)
print(torch.randn(3))
torch.manual_seed(0)
print(torch.randn(3))
torch.manual_seed(1)
print(torch.randn(3))
```

### 第 11 集 導數與鏈鎖律 — 狀態：可用

**題目**

函數 \\( f(x) = x^3 \\)。用有限差分（\\( h = 10^{-4} \\)）估算 \\( x = 2 \\) 的導數，和手算的 \\( 3x^2 = 12 \\) 比較。

**練習目標**：導數就是「x 動一點點，f 動多少」的比值。

**批改重點**：公式是 `(f(x + h) - f(x)) / h`；結果大約 12.0006，不是剛好 12 是正常的。

**參考答案**

```python
h = 1e-4
x = 2.0
f = lambda x: x**3
print((f(x + h) - f(x)) / h)
```

### 第 12 集 梯度 — 狀態：可用

**題目**

\\( f(x, y) = x^2 + 3y \\)。在 \\( (x, y) = (1, 2) \\) 用有限差分分別估算對 x、對 y 的偏導數，組成梯度，和手算的 \\( (2x, 3) = (2, 3) \\) 比較。

**練習目標**：梯度是每個變數各自偏導數排成的向量；算某個變數的偏導時，其他變數不動。

**批改重點**：兩次差分各只動一個變數。

**參考答案**

```python
h = 1e-4
f = lambda x, y: x**2 + 3 * y
x, y = 1.0, 2.0
dx = (f(x + h, y) - f(x, y)) / h
dy = (f(x, y + h) - f(x, y)) / h
print(dx, dy)
```

### 第 13 集 softmax — 狀態：可用

**題目**

分數 `[2.0, 1.0, 0.1]`。不用任何內建的 softmax，自己用 `torch.exp` 與 `.sum()` 算出機率，印出來並確認總和是 1。再把三個分數都加上 100，看看結果有沒有變。

**練習目標**：softmax 把分數變成總和為 1 的正數；整體平移不改變結果。

**批改重點**：總和用 `.sum()` 印出來接近 1；加 100 之後結果不變（第 16 集會解釋為什麼這件事很重要）。

**參考答案**

```python
import torch

s = torch.tensor([2.0, 1.0, 0.1])
p = torch.exp(s) / torch.exp(s).sum()
print(p, p.sum())
p2 = torch.exp(s + 100) / torch.exp(s + 100).sum()
print(p2)
```

### 第 14 集 機率、log 機率與取樣 — 狀態：可用

**題目**

機率 `[0.7, 0.2, 0.1]`。(1) 印出每個機率的 log；(2) 設定種子 0，用 `torch.multinomial` 取樣 10 次（`replacement=True`），印出抽到的索引；(3) 用 `torch.argmax` 印出機率最大的索引。

**練習目標**：log 機率都是負數、機率越小越負；取樣會以機率決定出現頻率；argmax 永遠選最大的。

**批改重點**：`torch.multinomial(p, 10, replacement=True)`；10 次裡索引 0 應該出現最多次。

**參考答案**

```python
import torch

p = torch.tensor([0.7, 0.2, 0.1])
print(torch.log(p))
torch.manual_seed(0)
print(torch.multinomial(p, 10, replacement=True))
print(torch.argmax(p))
```

### 第 15 集 cross-entropy 與 perplexity — 狀態：可用

**題目**

模型對三個題目各給出四個選項的機率：

```python,ignore
p = torch.tensor([[0.7, 0.1, 0.1, 0.1],
                  [0.25, 0.25, 0.25, 0.25],
                  [0.1, 0.1, 0.1, 0.7]])
```

正確答案分別是索引 0、1、3。算出每一題的 \\( -\log p \\)、三題的平均 cross-entropy，以及 perplexity。

**練習目標**：cross-entropy 只看正確答案那一格的機率；猜得越準越小；perplexity 是它的 exp。

**批改重點**：取正確答案用 `p[torch.arange(3), targets]`；第二題是均勻猜，貢獻 `log 4`。

**參考答案**

```python
import torch

p = torch.tensor([[0.7, 0.1, 0.1, 0.1],
                  [0.25, 0.25, 0.25, 0.25],
                  [0.1, 0.1, 0.1, 0.7]])
targets = torch.tensor([0, 1, 3])
nll = -torch.log(p[torch.arange(3), targets])
print(nll)
loss = nll.mean()
print(loss, torch.exp(loss))
```

### 第 16 集 數值穩定：log-sum-exp — 狀態：可用

**題目**

分數 `[1000.0, 999.0, 998.0]`。(1) 用第 13 集的寫法算 softmax，觀察結果；(2) 先減掉最大值再算，印出結果；(3) 用 `llmz.functional` 的 `softmax` 驗證。

**練習目標**：`exp(1000)` 會溢位成 `inf`；先減最大值結果不變、但不會溢位。

**批改重點**：第一種會印出 `nan`；減掉最大值後三個機率約 `[0.6652, 0.2447, 0.0900]`。

**參考答案**

```python
import torch

from llmz.functional import softmax

s = torch.tensor([1000.0, 999.0, 998.0])
print(torch.exp(s) / torch.exp(s).sum())
shifted = s - s.max()
print(torch.exp(shifted) / torch.exp(shifted).sum())
print(softmax(s))
```
