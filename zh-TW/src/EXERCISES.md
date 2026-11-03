# EXERCISES.md — 練習題規則與固定題庫

這份文件給 AI 助教使用，說明什麼時候可以出題、怎麼出、怎麼批改，並收錄已出版各章的固定題目。

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
- 不要出需要資料檔、需要 GPU、要跑超過幾秒的題目；一律用 CPU 上的小張量。第 2 章起可以出「訓練幾步」的題目，但資料要在程式裡自己造。

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

---

## 第 2 章 autograd 與訓練迴圈

### 第 1 集 autograd — 狀態：可用

**題目**

1. `x = torch.tensor(3.0, requires_grad=True)`，算 `y = x ** 3 + 2 * x`，用 `backward()` 印出 `x.grad`，和手算的 \\( 3x^2 + 2 = 29 \\) 比較。
2. `w = torch.tensor([1.0, 2.0], requires_grad=True)`，算 `(w * w).sum()` 的梯度，再用第 1 章第 12 集的有限差分驗算。

**練習目標**：`requires_grad` → 計算 → `backward()` → `.grad` 這四步；梯度形狀和張量一樣。

**批改重點**：`backward()` 要對純量呼叫；忘記 `requires_grad=True` 會得到 `None`。

**參考答案**（第 1 題）

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 3 + 2 * x
y.backward()
print(x.grad)
```

### 第 2 集 手動梯度下降 — 狀態：可用

**題目**

用 autograd 對 \\( f(w) = (w - 5)^2 \\) 做 10 步梯度下降，從 `w = 0.0` 開始、學習率 0.3，每步印出 `w` 的值（四捨五入到小數點後三位）。最後 `w` 應該接近 5。

**練習目標**：更新寫在 `torch.no_grad()` 裡、更新後梯度歸零。

**批改重點**：少了 `no_grad` 會報錯或梯度跟丟；少了 `zero_()`（或設 `None`）梯度會越加越大、`w` 衝過頭。

**參考答案**

```python
import torch

w = torch.tensor(0.0, requires_grad=True)
for step in range(10):
    f = (w - 5) ** 2
    f.backward()
    with torch.no_grad():
        w -= 0.3 * w.grad
    w.grad.zero_()
    print(step, round(w.item(), 3))
```

### 第 3 集 `nn.Module` 與 `nn.Parameter` — 狀態：可用

**題目**

寫一個 `Quadratic(nn.Module)`，有三個參數 `a`、`b`、`c`（都從 0 開始），`forward(x)` 回傳 \\( ax^2 + bx + c \\)。建立模型後用 `named_parameters()` 印出每個參數的名字，再對 `torch.tensor([1.0, 2.0])` 算一次輸出。

**練習目標**：`super().__init__()`、`nn.Parameter`、`forward`、`model(x)`。

**批改重點**：忘記 `super().__init__()` 會在指定 `nn.Parameter` 時報錯；呼叫用 `model(x)` 不是 `model.forward(x)`。

**參考答案**

```python
import torch
from torch import nn


class Quadratic(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.c = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.a * x ** 2 + self.b * x + self.c


model = Quadratic()
for name, p in model.named_parameters():
    print(name, p.shape)
print(model(torch.tensor([1.0, 2.0])))
```

### 第 4 集 訓練迴圈固定句型 — 狀態：可用

**題目**

把第 4 集的程式改成 `x = torch.tensor([0.0, 1.0, 2.0])`、`y = torch.tensor([1.0, 3.0, 5.0])`（也就是 \\( y = 2x + 1 \\)），學習率 0.1，跑 20 步，每 5 步印一次 loss。用五步固定句型，順序不能變。

**練習目標**：forward → loss → `backward()` → `step()` → `zero_grad()`。

**批改重點**：`zero_grad()` 放在 `backward()` 之前會清掉剛算好的梯度；忘記 `zero_grad()` 梯度會累加。

**參考答案**

```python
import torch
from torch import nn


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.w * x + self.b


def mse(pred, target):
    return ((pred - target) ** 2).mean()


class SGD:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for p in self.params:
                p -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = None


x = torch.tensor([0.0, 1.0, 2.0])
y = torch.tensor([1.0, 3.0, 5.0])
model = LinearModel()
opt = SGD(model.parameters(), lr=0.1)
for step in range(1, 21):
    pred = model(x)
    loss = mse(pred, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    if step % 5 == 0:
        print(step, round(loss.item(), 4))
```

### 第 5 集 線性回歸完整範例 — 狀態：可用

**題目**

把第 5 集的資料改成 \\( y = -2x + 4 \\) 加雜訊 0.1（種子 0，100 個點），訓練 300 步，印出最後的 `w`、`b`。再把學習率分別改成 0.05 與 2.5 各跑一次，用一句話描述你看到的差別。

**練習目標**：整支程式能從頭寫出來；觀察學習率對收斂的影響。

**批改重點**：`w` 接近 -2、`b` 接近 4；0.05 時 300 步還沒收斂，2.5 時 loss 變成 `inf` 或 `nan`。

**參考答案**（只列資料那三行，其餘與第 5 集相同）

```python,ignore
torch.manual_seed(0)
x = torch.rand(100)
y = -2 * x + 4 + 0.1 * torch.randn(100)
```

### 第 6 集 `state_dict` 儲存與載入 — 狀態：可用

**題目**

建立第 3 集的 `LinearModel`，在 `no_grad()` 裡把 `w` 設成 1.5、`b` 設成 -0.5，存成 `model.pt`。再建一個新的 `LinearModel`，載入檔案，印出 `state_dict()` 確認參數一樣。

**練習目標**：`state_dict()`、`torch.save`、`torch.load`、`load_state_dict`。

**批改重點**：就地改參數要在 `no_grad()` 裡（`fill_` 或 `copy_` 都可以）；載入前要先建好同一個類別的模型。

**參考答案**

```python
import torch
from torch import nn


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.w * x + self.b


m = LinearModel()
with torch.no_grad():
    m.w.fill_(1.5)
    m.b.fill_(-0.5)
torch.save(m.state_dict(), "model.pt")
m2 = LinearModel()
m2.load_state_dict(torch.load("model.pt"))
print(m2.state_dict())
```

### 第 7 集 FLOPs 與記憶體估算 — 狀態：可用

**題目**

一個模型有 5000 萬個參數，要看 4000 萬個 token。(1) 用 AdamW 訓練時，參數、梯度、優化器狀態合計要多少 GB？(2) 用「訓練一個 token 約 6 × 參數量 FLOPs」估算總 FLOPs。(3) 用 `count_parameters` 數一個有 `nn.Parameter(torch.zeros(512, 2048))` 與 `nn.Parameter(torch.zeros(2048))` 的 `nn.Module` 有幾個參數。

**練習目標**：把估算公式實際算一次；會用 `count_parameters`。

**批改重點**：(1) 50M × 16 位元組 = 0.8 GB；(2) 6 × 50M × 40M = 1.2 × 10¹⁶；(3) 1,050,624。

**參考答案**

```python
import torch
from torch import nn

from llmz.utils import count_parameters

params = 50_000_000
print(params * 16 / 1e9, "GB")
print(f"{6 * params * 40_000_000:.1e}")


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(512, 2048))
        self.b = nn.Parameter(torch.zeros(2048))


print(count_parameters(M()))
```

---

## 第 3 章 Tokenizer：從 bytes 到 token

### 第 1 集 字串、Unicode 與 UTF-8 — 狀態：可用

**題目**

印出「台灣」兩個字的 UTF-8 bytes 與整數串列，再只取前 4 個 byte 用 `errors="replace"` decode，看看印出什麼。

**練習目標**：`encode`／`decode`、中文一個字 3 個 byte、半個字會變 `�`。

**批改重點**：6 個 byte；前 4 個 byte decode 出「台」加一個 `�`。

**參考答案**

```python
b = "台灣".encode("utf-8")
print(b, list(b))
print(b[:4].decode("utf-8", errors="replace"))
```

### 第 2 集 為什麼不用字元或單字 — 狀態：可用

**題目**

寫一段三句的英文（自己寫），分別算：不同字元的個數、字元總數、不同單字的個數（用 `split()`）、單字總數、UTF-8 byte 數。用一句話說明字元級與單字級各自的問題。

**練習目標**：詞彙表大小與序列長度的取捨。

**批改重點**：`set` 算不同的個數；說明要提到「序列長」與「詞彙表大／缺字」。

**參考答案**

```python
text = "I have a cat. The cat likes fish. Fish is good."
print(len(set(text)), len(text))
words = text.split()
print(len(set(words)), len(words))
print(len(text.encode("utf-8")))
```

### 第 3 集 BPE 的核心想法 — 狀態：可用

**題目**

語料是 `("h","u","g")` 出現 10 次、`("p","u","g")` 5 次、`("p","u","n")` 12 次、`("b","u","n")` 4 次。用 `Counter` 統計相鄰 pair，印出最多的那一對，並手動寫出合併它之後四個字各變成什麼。

**練習目標**：pair 次數要乘上字的出現次數；合併後字的 tuple 變短。

**批改重點**：`("u","n")` 16 次最多；合併後 `("p","un")`、`("b","un")`，另外兩個字不變。

**參考答案**

```python
from collections import Counter

words = Counter({("h", "u", "g"): 10, ("p", "u", "g"): 5, ("p", "u", "n"): 12, ("b", "u", "n"): 4})
pairs = Counter()
for word, n in words.items():
    for a, b in zip(word, word[1:]):
        pairs[(a, b)] += n
print(pairs.most_common(1))
```

### 第 4 集 pre-tokenization 與 GPT-2 regex — 狀態：可用

**題目**

用 `regex.findall(GPT2_PATTERN, ...)` 切 `"Don't stop! 123 apples, 4.5 kg."`，先自己猜會切成幾段、哪些帶空白，再執行確認。

**練習目標**：縮寫、數字、標點各自成段；空白黏在後面的字前面。

**批改重點**：`'t` 是獨立一段；`4.5` 會被切成 ` 4`、`.`、`5`。

**參考答案**

```python
import regex

from llmz.tokenizer import GPT2_PATTERN

print(regex.findall(GPT2_PATTERN, "Don't stop! 123 apples, 4.5 kg."))
```

### 第 5 集 特殊 token — 狀態：可用

**題目**

文字 `"a<|eot|>b<|pad|><|pad|>c"`，特殊 token 有 `<|eot|>` 與 `<|pad|>`。用 `split_on_special` 切開並印出；再說明如果不切開直接 `pretokenize`，`<|pad|>` 會變成什麼樣子（執行看看）。

**練習目標**：多個特殊 token 一起切；不切開會被 regex 拆爛。

**批改重點**：結果 `['a', 'b', '', 'c']`（兩個 `<|pad|>` 之間是空字串）。

**參考答案**

```python
from llmz.tokenizer import pretokenize, split_on_special

text = "a<|eot|>b<|pad|><|pad|>c"
print(split_on_special(text, ["<|eot|>", "<|pad|>"]))
print(pretokenize(text))
```

### 第 6 集 訓練 BPE：統計 pair — 狀態：可用

**題目**

對 `"hug hug hug pug pun pun bun"` 呼叫 `count_words`（沒有特殊 token），印出每個字與次數；再用 `count_pairs` 印出最多的三個 pair。注意句首的 `hug` 和後面帶空白的 ` hug` 是不同的字。

**練習目標**：`count_words` 把 pre-token 拆成單 byte 的 tuple；`count_pairs` 的次數有乘上字的次數。

**批改重點**：`(b'u', b'g')` 4 次、`(b'u', b'n')` 3 次。

**參考答案**

```python
from llmz.tokenizer import count_pairs, count_words

wc = count_words("hug hug hug pug pun pun bun", [])
for w, n in wc.items():
    print(n, w)
print(count_pairs(wc).most_common(3))
```

### 第 7 集 訓練 BPE：合併 — 狀態：可用

**題目**

對同一段 `"hug hug hug pug pun pun bun"` 跑 `train_bpe_naive`，詞彙表 260，印出四條合併規則與 `vocab[256]` 到 `vocab[259]`。先猜第一條是什麼再執行。

**練習目標**：`merges` 的順序、`vocab` 的編號規則。

**批改重點**：第一條是 `(b'u', b'g')`；vocab 256 是 `b'ug'`。

**參考答案**

```python
from llmz.tokenizer import train_bpe_naive

vocab, merges = train_bpe_naive("hug hug hug pug pun pun bun", 260, [])
print(merges)
print([vocab[i] for i in range(256, 260)])
```

### 第 8 集 訓練 BPE：加速與低資源做法 — 狀態：可用

**題目**

自己寫一段五句以上的英文，分別用 `train_bpe_naive(text, 300, [])` 與 `train_bpe(count_words(text, []), 300, [])` 訓練，確認 `vocab` 與 `merges` 完全相同。

**練習目標**：兩個版本結果一致；知道 `train_bpe` 的輸入是 `count_words` 的結果。

**批改重點**：直接比較兩個 tuple 印出 `True`。

**參考答案**

```python
from llmz.tokenizer import count_words, train_bpe, train_bpe_naive

text = "The sun was hot. The sand was hot. The sea was cool. We ran to the sea. The sea was fun."
print(train_bpe_naive(text, 300, []) == train_bpe(count_words(text, []), 300, []))
```

### 第 9 集 encode — 狀態：可用

**題目**

用 `"hug hug hug pug pun pun bun"` 訓練詞彙表 260 的 tokenizer（沒有特殊 token），encode `"hug bun"`，印出 id 與每個 id 對應的 bytes。先用第 7 集的合併規則手動推一次再執行。

**練習目標**：encode 照 rank 順序套用合併；句首與帶空白的字切法不同。

**批改重點**：`hug` 會合成一個 token（如果規則裡有），` bun` 要看 ` b`、`un` 有沒有被學到。

**參考答案**

```python
from llmz.tokenizer import Tokenizer, train_bpe_naive

vocab, merges = train_bpe_naive("hug hug hug pug pun pun bun", 260, [])
tok = Tokenizer(vocab, merges, [])
ids = tok.encode("hug bun")
print(ids, [vocab[i] for i in ids])
```

### 第 10 集 decode 與無效 bytes — 狀態：可用

**題目**

用只有 256 個 byte 的 tokenizer（`Tokenizer({i: bytes([i]) for i in range(256)}, [], [])`）encode「貓」，印出 id；分別 decode 前 1、2、3 個 id，看哪一個才印出「貓」。

**練習目標**：decode 先接 bytes 再轉字串；不完整的部分變 `�`。

**批改重點**：前兩次是 `�`，第三次是「貓」。

**參考答案**

```python
from llmz.tokenizer import Tokenizer

tok = Tokenizer({i: bytes([i]) for i in range(256)}, [], [])
ids = tok.encode("貓")
print(ids)
for k in [1, 2, 3]:
    print(tok.decode(ids[:k]))
```

### 第 11 集 串流 encode 大檔 — 狀態：可用

**題目**

用第 9 集的 tokenizer，把 `"hug\npug\nbun\n"` 用 `io.StringIO` 包成假檔案，`encode_iterable` 的結果用 `list` 收起來，和直接 `encode` 整段文字比較是否相同。再把 tokenizer `save` 成檔案、`load` 回來，確認 encode 結果一樣。

**練習目標**：逐行 encode 等於整段 encode；save／load 往返。

**批改重點**：兩個比較都印 `True`。

**參考答案**

```python
import io

from llmz.tokenizer import Tokenizer, train_bpe_naive

vocab, merges = train_bpe_naive("hug hug hug pug pun pun bun", 260, [])
tok = Tokenizer(vocab, merges, [])
text = "hug\npug\nbun\n"
print(list(tok.encode_iterable(io.StringIO(text))) == tok.encode(text))
tok.save("t.json")
print(Tokenizer.load("t.json").encode(text) == tok.encode(text))
```

### 第 12 集 中文語料的 BPE 示範 — 狀態：可用

**題目**

自己挑一首唐詩（公版）寫進程式，訓練詞彙表 300 的 tokenizer，印出前 5 條合併規則各自 decode（`errors="replace"`）後的樣子，數一數前 5 條裡有幾條是完整的字。

**練習目標**：中文 BPE 先學半個字再學整個字。

**批改重點**：前幾條大多是 `�`（兩個 byte）；完整的字要三個 byte 合完才出現。

**參考答案**

```python
from llmz.tokenizer import train_bpe_naive

poem = "床前明月光，疑是地上霜。舉頭望明月，低頭思故鄉。床前明月光，疑是地上霜。"
vocab, merges = train_bpe_naive(poem, 300, [])
for a, b in merges[:5]:
    print((a + b).decode("utf-8", errors="replace"))
```

### 第 13 集 壓縮率實驗 — 狀態：可用

**題目**

用第 13 集的 `story` 訓練詞彙表 400 的 tokenizer，自己寫兩句話：一句盡量用 `story` 裡出現過的字、一句盡量用沒出現過的字，算各自的 bytes/token，確認前者比較高。

**練習目標**：壓縮率只對相近的文字有效。

**批改重點**：沒看過的字會退到一個字母一個 token，bytes/token 接近 1。

**參考答案**

```python
from llmz.tokenizer import Tokenizer, train_bpe_naive

story = "Tom had a red ball. He liked to play with the ball in the park. The ball rolled into the pond."
vocab, merges = train_bpe_naive(story, 400, [])
tok = Tokenizer(vocab, merges, [])
for t in ["The ball was in the park.", "Quantum physics puzzles everyone."]:
    print(len(t.encode("utf-8")) / len(tok.encode(t)))
```

### 第 14 集 把 TinyStories 轉成 token 檔 — 狀態：不出題

這集是跑 `uv run tools/prepare_data.py`。讀者跑不起來時先看網路（下載要連到 huggingface.co）與 `data/` 目錄的權限；跑完的 token 數應該和書上一樣，不一樣多半是檔案不完整，刪掉 `data/` 重跑。
