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

---

## 第 4 章 Transformer 的零件

### 第 1 集 一張圖看懂 Transformer LM — 狀態：可用

**題目**

不用 `llmz`，只用 `torch.randint`、`torch.randn`、索引與 `@`，做出 B=3、T=4、vocab 20、d_model 8 的 `ids`、embedding 查表結果與 logits，印出三者的形狀。

**練習目標**：記住 `(B, T)` → `(B, T, d_model)` → `(B, T, vocab_size)`。

**批改重點**：`table[ids]` 的索引；最後乘的矩陣是 `(d_model, vocab_size)`。

**參考答案**

```python
import torch

torch.manual_seed(0)
ids = torch.randint(0, 20, (3, 4))
x = torch.randn(20, 8)[ids]
logits = x @ torch.randn(8, 20)
print(ids.shape, x.shape, logits.shape)
```

### 第 2 集 Linear（自己寫） — 狀態：可用

**題目**

建立 `Linear(6, 2)`，輸入 `torch.randn(4, 6)`，印出輸出形狀，並用 `torch.allclose` 確認輸出等於 `x @ layer.weight.T`。再把 1000 層 `Linear(64, 64)` 串起來跑一個 `torch.randn(10, 64)`，印出最後輸出的標準差（四捨五入到小數第二位），看它有沒有爆掉或消失。

**練習目標**：`Linear` 的形狀與初始化效果。

**批改重點**：1000 層後標準差應該仍在 1 的量級（不是 0 也不是 inf）；用 `for` 迴圈即可。

**參考答案**

```python
import torch

from llmz.layers import Linear

torch.manual_seed(0)
layer = Linear(6, 2)
x = torch.randn(4, 6)
print(layer(x).shape, torch.allclose(layer(x), x @ layer.weight.T))
h = torch.randn(10, 64)
for _ in range(1000):
    h = Linear(64, 64)(h)
print(round(h.std().item(), 2))
```

### 第 3 集 Embedding — 狀態：可用

**題目**

`Embedding(5, 3)`，用 `ids = torch.tensor([4, 4, 1])` 查表，確認前兩列相同、第三列不同；再印出 `count_parameters`。

**練習目標**：同一個 id 永遠拿到同一列；參數量是 `num × dim`。

**批改重點**：`torch.equal(out[0], out[1])` 為 `True`；參數 15。

**參考答案**

```python
import torch

from llmz.layers import Embedding
from llmz.utils import count_parameters

torch.manual_seed(0)
emb = Embedding(5, 3)
out = emb(torch.tensor([4, 4, 1]))
print(torch.equal(out[0], out[1]), torch.equal(out[0], out[2]))
print(count_parameters(emb))
```

### 第 4 集 RMSNorm — 狀態：可用

**題目**

做一個 `(3, 8)` 的張量，三列分別乘 0.01、1、100。過 `RMSNorm(8)` 之後，印出每一列的均方根（自己用 `(y * y).mean(dim=-1).sqrt()` 算），確認都接近 1。再把 `norm.weight` 在 `no_grad` 裡全部填成 2，看均方根變成多少。

**練習目標**：每列各自正規化；`weight` 是之後的縮放。

**批改重點**：三列都是 1.0；填 2 之後是 2.0。

**參考答案**

```python
import torch

from llmz.layers import RMSNorm

torch.manual_seed(0)
x = torch.randn(3, 8) * torch.tensor([[0.01], [1.0], [100.0]])
norm = RMSNorm(8)
y = norm(x)
print((y * y).mean(dim=-1).sqrt())
with torch.no_grad():
    norm.weight.fill_(2.0)
print((norm(x) ** 2).mean(dim=-1).sqrt())
```

### 第 5 集 SiLU 與 SwiGLU — 狀態：可用

**題目**

(1) 自己用 `torch.exp` 寫出 sigmoid，和 `torch.sigmoid` 在 `[-3, 0, 3]` 上比對。(2) 建立 `SwiGLU(8, 32)`，印出參數量並手算驗證是 \\( 3 \times 8 \times 32 \\)。

**練習目標**：sigmoid 的定義；SwiGLU 三個矩陣。

**批改重點**：`1 / (1 + torch.exp(-x))`；參數 768。

**參考答案**

```python
import torch

from llmz.layers import SwiGLU
from llmz.utils import count_parameters

x = torch.tensor([-3.0, 0.0, 3.0])
print(torch.allclose(1 / (1 + torch.exp(-x)), torch.sigmoid(x)))
print(count_parameters(SwiGLU(8, 32)), 3 * 8 * 32)
```

### 第 6 集 位置資訊與 RoPE 的想法 — 狀態：可用

**題目**

用第 6 集的 `rotate` 函式，固定 `q`、`k` 與角度 0.5，印出位置差 0、1、2、3 的點積各一個（取 (m, n) = (d, 0)），再印 (10, 7) 的點積，確認它等於差 3 的那個。

**練習目標**：旋轉後的點積只看位置差。

**批改重點**：(10, 7) 與 (3, 0) 相同。

**參考答案**

```python
import torch


def rotate(v, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((v[0] * c - v[1] * s, v[0] * s + v[1] * c))


theta = torch.tensor(0.5)
q, k = torch.tensor([1.0, 0.5]), torch.tensor([0.2, -1.0])
for d in range(4):
    print(d, round((rotate(q, d * theta) @ rotate(k, 0 * theta)).item(), 4))
print(round((rotate(q, 10 * theta) @ rotate(k, 7 * theta)).item(), 4))
```

### 第 7 集 RoPE 實作 — 狀態：可用

**題目**

`RoPE(d_k=4, max_seq_len=8)`，印出 `rope.cos` 的形狀與第 0 列（位置 0 的 cos 應該全是 1）。對一個 `torch.randn(1, 3, 4)` 用位置 `[0, 1, 2]` 旋轉，確認每個位置的向量長度不變，而且位置 0 的向量完全沒變。

**練習目標**：cos／sin 表的形狀；位置 0 不轉；長度不變。

**批改重點**：`rope.cos[0]` 全 1；`torch.allclose(out[0, 0], x[0, 0])`。

**參考答案**

```python
import torch

from llmz.attention import RoPE

torch.manual_seed(0)
rope = RoPE(4, 8)
print(rope.cos.shape, rope.cos[0])
x = torch.randn(1, 3, 4)
out = rope(x, torch.arange(3))
print(torch.allclose(out.norm(dim=-1), x.norm(dim=-1)), torch.allclose(out[0, 0], x[0, 0]))
```

### 第 8 集 數值穩定的 softmax — 狀態：可用

**題目**

對 `[[5.0, -inf, 3.0], [-inf, -inf, 1.0]]` 做 `llmz.functional.softmax`，先猜結果再執行。然後說明為什麼第二列不會是 `nan`。

**練習目標**：`-inf` 的位置機率 0；只要有一個不是 `-inf` 就安全。

**批改重點**：第二列是 `[0, 0, 1]`。

**參考答案**

```python
import torch

from llmz.functional import softmax

inf = float("inf")
print(softmax(torch.tensor([[5.0, -inf, 3.0], [-inf, -inf, 1.0]]), dim=-1))
```

### 第 9 集 scaled dot-product attention — 狀態：可用

**題目**

q、k、v 都是 `torch.randn(2, 3, 4)`（沒有 mask）。自己用 `@`、`transpose`、`softmax` 寫出 attention，和 `scaled_dot_product_attention` 的結果 `allclose` 比對。

**練習目標**：公式三步：分數、softmax、加權 value。

**批改重點**：要除以 `4 ** 0.5`；softmax 沿最後一維。

**參考答案**

```python
import torch

from llmz.attention import scaled_dot_product_attention
from llmz.functional import softmax

torch.manual_seed(0)
q, k, v = torch.randn(2, 3, 4), torch.randn(2, 3, 4), torch.randn(2, 3, 4)
mine = softmax(q @ k.transpose(-2, -1) / 4 ** 0.5, dim=-1) @ v
print(torch.allclose(mine, scaled_dot_product_attention(q, k, v), atol=1e-6))
```

### 第 10 集 causal mask — 狀態：可用

**題目**

用 `causal_mask(5)` 和 `scaled_dot_product_attention` 算一個 T=5 的 attention（q、k、v 隨機、d=4）。把 `v[4]` 改成全 0 再算一次，確認前四個位置的輸出完全沒變。

**練習目標**：mask 真的擋住了未來。

**批改重點**：比較 `out[:4]` 與 `out2[:4]`。

**參考答案**

```python
import torch

from llmz.attention import causal_mask, scaled_dot_product_attention

torch.manual_seed(0)
q, k, v = torch.randn(5, 4), torch.randn(5, 4), torch.randn(5, 4)
out = scaled_dot_product_attention(q, k, v, causal_mask(5))
v2 = v.clone()
v2[4] = 0
out2 = scaled_dot_product_attention(q, k, v2, causal_mask(5))
print(torch.allclose(out[:4], out2[:4]), torch.allclose(out[4], out2[4]))
```

### 第 11 集 multi-head attention — 狀態：可用

**題目**

建立 `MultiHeadSelfAttention(d_model=8, n_heads=2)` 與 `n_heads=4` 各一個，印出兩者的參數量，說明為什麼一樣。再餵 `torch.randn(1, 3, 8)` 印出輸出形狀。

**練習目標**：head 數不影響參數量。

**批改重點**：兩者都是 4 × 8 × 8 = 256。

**參考答案**

```python
import torch

from llmz.attention import MultiHeadSelfAttention
from llmz.utils import count_parameters

torch.manual_seed(0)
a, b = MultiHeadSelfAttention(8, 2), MultiHeadSelfAttention(8, 4)
print(count_parameters(a), count_parameters(b))
print(a(torch.randn(1, 3, 8)).shape)
```

### 第 12 集 把 RoPE 接進 attention — 狀態：可用

**題目**

建立有 RoPE 的 `MultiHeadSelfAttention(16, 4, rope=RoPE(4, 32))`。輸入 `x` 是 `torch.randn(1, 6, 16)`，把 `x` 的前兩個位置對調做成 `x2`，分別算輸出，確認 `out[0, 2:]` 與 `out2[0, 2:]`（第 2 個位置之後）**不**相同。再用沒有 RoPE 的版本做同樣的事，看結果是否相同，並用一句話解釋。

**練習目標**：RoPE 讓模型分得出順序；沒有 RoPE 時 causal attention 對前面 token 的排列不敏感。

**批改重點**：有 RoPE 時不同；沒有 RoPE 時位置 2 之後的輸出相同（它們看到的是同一組 key、value，只是順序不同）。

**參考答案**

```python
import torch

from llmz.attention import MultiHeadSelfAttention, RoPE

torch.manual_seed(0)
x = torch.randn(1, 6, 16)
x2 = x.clone()
x2[0, [0, 1]] = x[0, [1, 0]]
for rope in [RoPE(4, 32), None]:
    torch.manual_seed(1)
    attn = MultiHeadSelfAttention(16, 4, rope=rope)
    print(rope is not None, torch.allclose(attn(x)[0, 2:], attn(x2)[0, 2:], atol=1e-5))
```

### 第 13 集 pre-norm Transformer block — 狀態：可用

**題目**

用 `ModelConfig(d_model=16, n_heads=2, d_ff=32, context_length=8)` 建一個 `TransformerBlock`，餵 `torch.randn(1, 5, 16)`，確認輸出形狀。再用 `count_parameters` 和公式 \\( 4d^2 + 3d \cdot d_{ff} + 2d \\) 比對。

**練習目標**：block 不改形狀；知道參數在哪。

**批改重點**：1024 + 1536 + 32 = 2592。

**參考答案**

```python
import torch

from llmz.attention import RoPE
from llmz.model import ModelConfig, TransformerBlock
from llmz.utils import count_parameters

torch.manual_seed(0)
cfg = ModelConfig(d_model=16, n_heads=2, d_ff=32, context_length=8)
block = TransformerBlock(cfg, RoPE(8, 8))
print(block(torch.randn(1, 5, 16)).shape)
print(count_parameters(block), 4 * 16 * 16 + 3 * 16 * 32 + 2 * 16)
```

### 第 14 集 完整 Transformer LM — 狀態：可用

**題目**

用 `ModelConfig(vocab_size=30, context_length=8, d_model=16, n_layers=1, n_heads=2, d_ff=32)` 建模型，隨機一段 `ids`（形狀 `(2, 8)`），照書上的方式切出 `inputs`、`targets`，算 cross-entropy，確認它接近 \\( \ln 30 \\)。

**練習目標**：shift 一格；`reshape(-1, V)`；未訓練 loss ≈ ln V。

**批改重點**：`inputs = ids[:, :-1]`、`targets = ids[:, 1:]`；loss 在 3.4 附近（ln 30 = 3.40）。

**參考答案**

```python
import math

import torch

from llmz.functional import cross_entropy
from llmz.model import ModelConfig, TransformerLM

torch.manual_seed(0)
model = TransformerLM(ModelConfig(vocab_size=30, context_length=8, d_model=16, n_layers=1, n_heads=2, d_ff=32))
ids = torch.randint(0, 30, (2, 8))
logits = model(ids[:, :-1])
loss = cross_entropy(logits.reshape(-1, 30), ids[:, 1:].reshape(-1))
print(round(loss.item(), 2), round(math.log(30), 2))
```

### 第 15 集 參數量與 FLOPs — 狀態：可用

**題目**

把標準設定的 `d_model` 改成 768、`n_layers` 改成 6、`d_ff` 改成 2048，其餘不變。用第 15 集的公式算參數量，再用 `count_parameters` 驗證；接著算 forward 每個 token 的 FLOPs 與 41M token 訓練的總 FLOPs。

**練習目標**：把公式套到新設定上。

**批改重點**：先列每層 \\( 4d^2 + 3d \cdot d_{ff} + 2d \\)，再加 embedding、lm_head、最後的 norm。

**參考答案**

```python
from llmz.model import ModelConfig, TransformerLM
from llmz.utils import count_parameters

cfg = ModelConfig(d_model=768, n_layers=6, d_ff=2048)
V, d, L, ff, T = cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.d_ff, cfg.context_length
per_layer = 4 * d * d + 3 * d * ff + 2 * d
total = V * d + L * per_layer + d + d * V
print(total, count_parameters(TransformerLM(cfg)) == total)
forward = L * (2 * 4 * d * d + 4 * T * d + 2 * 3 * d * ff) + 2 * d * V
print(f"{forward / 1e6:.1f} MFLOPs/token, {3 * forward * 41_000_000 / 1e15:.1f} PFLOPs")
```

---

## 第 5 章 訓練

### 第 1 集 cross-entropy loss 實作 — 狀態：可用

**題目**

`ids = torch.tensor([[1, 2, 3, 4, 5]])`，手動切出 `inputs`、`targets`，用 `torch.zeros(1, 4, 8)` 當 logits（八個 token 全部同分）算 `cross_entropy`，確認結果是 \\( \ln 8 \\)。

**練習目標**：對齊與攤平；同分的 loss 是 ln V。

**批改重點**：`reshape(-1, 8)` 與 `reshape(-1)`；答案 2.0794。

**參考答案**

```python
import math

import torch

from llmz.functional import cross_entropy

ids = torch.tensor([[1, 2, 3, 4, 5]])
inputs, targets = ids[:, :-1], ids[:, 1:]
logits = torch.zeros(1, 4, 8)
print(cross_entropy(logits.reshape(-1, 8), targets.reshape(-1)), math.log(8))
```

### 第 2 集 從頭寫 SGD — 狀態：可用

**題目**

用 `llmz.optim.SGD` 重做第 2 章第 5 集的線性回歸（300 步、lr 0.5），確認最後的 `w`、`b` 和第 2 章一樣。然後印出 `opt.state_dict()["param_groups"]`，看裡面有什麼。

**練習目標**：基底類別版本的用法與第 2 章一致；認識 `state_dict` 的內容。

**批改重點**：`w` 2.931、`b` 2.033。

**參考答案**

```python
import torch
from torch import nn

from llmz.optim import SGD


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.w * x + self.b


torch.manual_seed(0)
x = torch.rand(100)
y = 3 * x + 2 + 0.1 * torch.randn(100)
model = LinearModel()
opt = SGD(model.parameters(), lr=0.5)
for _ in range(300):
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
print(round(model.w.item(), 3), round(model.b.item(), 3))
print(opt.state_dict()["param_groups"])
```

### 第 3 集 AdamW — 狀態：可用

**題目**

一個參數 `p = nn.Parameter(torch.tensor([1.0]))`，手動設 `p.grad = torch.tensor([0.5])`，用 `AdamW([p], lr=0.1, weight_decay=0.0)` 做一步。先手算第一步：\\( m = 0.1 \times 0.5 = 0.05 \\)、\\( v = 0.05 \times 0.25 = 0.0125 \\)、\\( m/\sqrt{v} = 0.447 \\)、修正係數 \\( \sqrt{1-0.95}/(1-0.9) = 2.236 \\)，所以更新量是 \\( 0.1 \times 2.236 \times 0.447 = 0.1 \\)，`p` 應該變成 0.9。再執行確認，並印出 `opt.state[p]`。

**練習目標**：把 AdamW 的公式手算一次；發現第一步的更新量剛好是 lr（不管梯度多大）。

**批改重點**：`p` 變成 0.9；`state` 裡 `t` 是 1、`m` 是 0.05、`v` 是 0.0125。

**參考答案**

```python
import torch
from torch import nn

from llmz.optim import AdamW

p = nn.Parameter(torch.tensor([1.0]))
opt = AdamW([p], lr=0.1, weight_decay=0.0)
p.grad = torch.tensor([0.5])
opt.step()
print(p, opt.state[p])
```

### 第 4 集 warmup + cosine 學習率排程 — 狀態：可用

**題目**

用 `lr_cosine_schedule`，設定 `lr_max=0.01`、`lr_min=0.001`、warmup 10 步、cosine 100 步，印出第 0、5、10、55、100、200 步的學習率，並說明第 55 步為什麼是 0.0055。

**練習目標**：三段的形狀。

**批改重點**：第 55 步是 cosine 的中點，\\( \cos(\pi/2) = 0 \\)，所以是兩端的平均。

**參考答案**

```python
from llmz.optim import lr_cosine_schedule

for s in [0, 5, 10, 55, 100, 200]:
    print(s, round(lr_cosine_schedule(s, 0.01, 0.001, 10, 100), 5))
```

### 第 5 集 梯度裁剪 — 狀態：可用

**題目**

兩個參數的梯度分別是 `[1.0, 1.0]` 與 `[1.0, 1.0]`，`max_norm` 設 1.0，先手算裁剪後每個元素是多少，再用 `clip_grad_norm_` 驗證。接著把 `max_norm` 改成 5.0，確認梯度沒被動。

**練習目標**：整體長度、等比例縮小、不超過就不動。

**批改重點**：長度 2，每個元素變 0.5；`max_norm` 5 時不變。

**參考答案**

```python
import torch
from torch import nn

from llmz.optim import clip_grad_norm_

a, b = nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(2))
a.grad, b.grad = torch.ones(2), torch.ones(2)
print(clip_grad_norm_([a, b], 1.0), a.grad, b.grad)
a.grad, b.grad = torch.ones(2), torch.ones(2)
print(clip_grad_norm_([a, b], 5.0), a.grad)
```

### 第 6 集 資料載入：隨機取 batch — 狀態：可用

**題目**

用 `np.arange(100, dtype=np.uint16)` 當資料，種子 0，`get_batch(batch_size=2, context_length=4)`，印出 `x`、`y`。再把 `context_length` 改成 99，觀察 `torch.randint` 的上限變成多少、會不會出錯。

**練習目標**：起點範圍與 `+1` 的關係。

**批改重點**：`context_length=99` 時起點只能是 0，`len(data) - context_length` 是 1，`randint(0, 1)` 永遠 0，不會出錯；100 才會出錯。

**參考答案**

```python
import numpy as np
import torch

from llmz.data import get_batch

data = np.arange(100, dtype=np.uint16)
torch.manual_seed(0)
x, y = get_batch(data, 2, 4, torch.device("cpu"))
print(x, y, sep="\n")
x, y = get_batch(data, 2, 99, torch.device("cpu"))
print(x[:, :3], y[:, -3:])
```

### 第 7 集 checkpoint — 狀態：可用

**題目**

用第 7 集的小模型存一個 checkpoint（步數 5），然後只載入模型、不傳優化器（`load_checkpoint(path, model2)`），確認回傳的步數還是 5、模型輸出一致，而新的優化器 `state` 是空的。

**練習目標**：`optimizer=None` 的用法；知道只載模型時優化器從零開始。

**批改重點**：`len(opt2.state) == 0`。

**參考答案**

```python
import torch

from llmz.model import ModelConfig, TransformerLM
from llmz.optim import AdamW
from llmz.train import load_checkpoint, save_checkpoint

cfg = ModelConfig(vocab_size=50, context_length=16, d_model=32, n_layers=1, n_heads=4, d_ff=64)
torch.manual_seed(0)
model = TransformerLM(cfg)
opt = AdamW(model.parameters())
save_checkpoint(model, opt, 5, "c.pt")
model2 = TransformerLM(cfg)
opt2 = AdamW(model2.parameters())
x = torch.randint(0, 50, (1, 6))
print(load_checkpoint("c.pt", model2), torch.allclose(model(x), model2(x)), len(opt2.state))
```

### 第 8 集 組裝訓練迴圈與 JSONL log — 狀態：可用

**題目**

把第 8 集的 smoke 設定改成 `max_steps=30`、`checkpoint_every=15`，跑一次 `train(cfg)`；再用**同一個** `cfg` 但 `max_steps=45` 跑第二次，觀察第二次是不是從第 30 步接續（看印出來的第一筆紀錄的 `step`）。

**練習目標**：`latest.pt` 自動接續。

**批改重點**：第二次印出的第一筆是 step 40 或 45（看 `log_every`），不是從 10 開始；`log.jsonl` 會累積兩次的紀錄。

**參考答案**

```python
import numpy as np

from llmz.model import ModelConfig
from llmz.train import TrainConfig, train

tokens = (np.arange(20000) % 64).astype(np.uint16)
tokens.tofile("train.bin")
tokens.tofile("valid.bin")
small = ModelConfig(vocab_size=64, context_length=16, d_model=32, n_layers=2, n_heads=4, d_ff=64)
base = dict(train_path="train.bin", valid_path="valid.bin", out_dir="smoke", model=small, batch_size=8,
            warmup_steps=5, lr_max=3e-3, log_every=15, eval_every=1000, eval_batches=1, checkpoint_every=15, device="cpu")
train(TrainConfig(max_steps=30, **base))
records = train(TrainConfig(max_steps=45, **base))
print([r["step"] for r in records])
```

### 第 9 集 MPS 注意事項 — 狀態：不出題

這集是實測數字與說明。讀者想知道自己機器的速度，請他跑 `uv run tools/train.py --out_dir runs/speed --max_steps 50 --log_every 10`（有 MPS 自動用 MPS），看 log 裡 `time` 欄位每 10 步差多少秒。

### 第 10 集 第一次完整訓練 TinyStories（40M tokens） — 狀態：不出題

這集是跑 `uv run tools/train.py --out_dir runs/base --compile`，幾十分鐘。讀者的 loss 曲線應該和書上接近（最後驗證 loss 差 0.05 以內都正常）。跑不動（記憶體不足）時建議 `--batch_size 16 --max_steps 10000`，token 總量相同。

### 第 11 集 畫 loss 曲線 — 狀態：可用

**題目**

自己造一個 `log.jsonl`（五行，`step` 與 `loss` 遞減），寫一支程式讀它、用 matplotlib 畫成 `my_loss.png`。不要用 `tools/plot_loss.py`，自己寫 `json.loads` 與 `plt.plot`。

**練習目標**：JSONL 的讀法、最基本的 matplotlib。

**批改重點**：逐行 `json.loads`；`plt.savefig` 之前不需要 `plt.show()`。

**參考答案**

```python
import json

import matplotlib.pyplot as plt

with open("log.jsonl", "w") as f:
    for step, loss in [(100, 5.0), (200, 3.2), (300, 2.5), (400, 2.1), (500, 1.9)]:
        f.write(json.dumps({"step": step, "loss": loss}) + "\n")
records = [json.loads(line) for line in open("log.jsonl")]
plt.plot([r["step"] for r in records], [r["loss"] for r in records], marker="o")
plt.xlabel("step")
plt.ylabel("loss")
plt.savefig("my_loss.png")
print("saved")
```

---

## 第 6 章 生成與實驗

### 第 1 集 greedy decoding — 狀態：可用

**題目**

照第 1 集的方式訓練會數數的小模型（150 步），然後從 `[60, 61, 62]` 開始 greedy 生成 6 個 token，看它會不會在 63 之後繞回 0。

**練習目標**：生成迴圈的四個步驟；`[0, -1]` 取最後位置。

**批改重點**：結果應是 `[60, 61, 62, 63, 0, 1, 2, 3, 4]`；餵模型時用 `ids[-16:]`。

**參考答案**

```python
import numpy as np
import torch

from llmz.generate import greedy_next
from llmz.model import ModelConfig, TransformerLM
from llmz.train import TrainConfig, train

tokens = (np.arange(20000) % 64).astype(np.uint16)
tokens.tofile("train.bin")
tokens.tofile("valid.bin")
cfg = TrainConfig(train_path="train.bin", valid_path="valid.bin", out_dir="smoke",
                  model=ModelConfig(vocab_size=64, context_length=16, d_model=32, n_layers=2, n_heads=4, d_ff=64),
                  batch_size=8, max_steps=150, warmup_steps=5, lr_max=3e-3, log_every=1000, eval_every=1000, checkpoint_every=1000, device="cpu")
train(cfg)
model = TransformerLM(cfg.model)
model.load_state_dict(torch.load("smoke/model.pt"))
model.eval()
ids = [60, 61, 62]
with torch.no_grad():
    for _ in range(6):
        ids.append(greedy_next(model(torch.tensor([ids[-16:]]))[0, -1]).item())
print(ids)
```

### 第 2 集 temperature — 狀態：可用

**題目**

logits `[1.0, 1.0, 1.0, 5.0]`。用 `llmz.functional.softmax` 印出溫度 0.5、1、2 時的機率；再用 `sample_next` 各抽 100 次（種子 0），數 token 3 被抽到幾次。

**練習目標**：溫度對分布的影響；抽樣次數和機率對得上。

**批改重點**：溫度越高 token 3 的次數越少；溫度 0.5 時幾乎 100 次。

**參考答案**

```python
import torch

from llmz.functional import softmax
from llmz.generate import sample_next

logits = torch.tensor([1.0, 1.0, 1.0, 5.0])
torch.manual_seed(0)
for t in [0.5, 1.0, 2.0]:
    n = sum(sample_next(logits, temperature=t).item() == 3 for _ in range(100))
    print(t, softmax(logits / t), n)
```

### 第 3 集 top-p 取樣 — 狀態：可用

**題目**

logits `[2.0, 1.0, 0.0, -1.0, -2.0]`。先算 softmax 的累積機率，手動判斷 `top_p=0.8` 會留下幾個 token，再用 `sample_next` 抽 200 次驗證只出現那些索引。

**練習目標**：「累積減自己 < p」的規則。

**批改重點**：機率約 `[0.64, 0.23, 0.09, 0.03, 0.01]`，累積 `[0.64, 0.87, ...]`；p=0.8 留前兩個。

**參考答案**

```python
import torch

from llmz.functional import softmax
from llmz.generate import sample_next

logits = torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0])
print(torch.cumsum(softmax(logits), dim=-1))
torch.manual_seed(0)
print(sorted(set(sample_next(logits, top_p=0.8).item() for _ in range(200))))
```

### 第 4 集 完整生成函式 — 狀態：可用

**題目**

載入 `runs/base/model.pt`（需要先跑完第 5 章第 10 集），用同一個開頭 `"One day, a little bird"`、同一個種子，分別用 `temperature=0`、`temperature=1.0`、`temperature=0.8, top_p=0.9` 各生成 80 個 token，比較三段文字的通順度與重複程度。沒有訓練好的模型時，改用第 1 集的數數模型，確認 `generate` 能跑（會輸出亂碼是正常的，因為 tokenizer 不對）。

**練習目標**：`generate` 的用法；三種取樣設定的差別。

**批改重點**：greedy 的那段會出現重複句型；這題沒有標準輸出。

**參考答案**（需要 `runs/base/model.pt` 與 `data/tokenizer.json`，不會被自動驗證）

```python,ignore
import torch

from llmz.device import get_device
from llmz.generate import generate
from llmz.model import ModelConfig, TransformerLM
from llmz.tokenizer import Tokenizer

device = get_device()
model = TransformerLM(ModelConfig()).to(device)
model.load_state_dict(torch.load("runs/base/model.pt", map_location=device))
model.eval()
tok = Tokenizer.load("data/tokenizer.json")
for kw in [dict(temperature=0), dict(temperature=1.0), dict(temperature=0.8, top_p=0.9)]:
    torch.manual_seed(0)
    print(kw, generate(model, tok, "One day, a little bird", max_new_tokens=80, **kw), sep="\n")
```

### 第 5～10 集 消融與掃描實驗 — 狀態：不出題

這幾集是跑 `tools/train.py` 的實驗，每個五分鐘以上，不適合當練習題。讀者想動手時，建議他挑一個書上沒做的變化（例如 `--n_layers 2` 或 `--d_model 256`）用同樣的 1000 步設定跑一次，和 `runs/base_1k` 比曲線，並提醒：一次只改一個東西、用 `tools/plot_loss.py` 把兩個 run 畫在同一張圖。

---

## 附錄一

### a MLX — 狀態：不出題

概念介紹，本書環境沒有裝 `mlx`。

### b `einops` 速查 — 狀態：可用

**題目**

把第 4 章第 11 集 `MultiHeadSelfAttention.forward` 裡的兩個 reshape／transpose（拆 head 與接回去）用 `rearrange` 改寫，對一個 `torch.randn(2, 5, 16)`（4 個 head）驗證結果與原本的寫法相同。

**練習目標**：`rearrange` 的字串對應到 reshape 與 transpose。

**批改重點**：`"b t (h d) -> b h t d"` 與 `"b h t d -> b t (h d)"`。

**參考答案**

```python
import torch
from einops import rearrange

torch.manual_seed(0)
x = torch.randn(2, 5, 16)
a = x.reshape(2, 5, 4, 4).transpose(1, 2)
b = rearrange(x, "b t (h d) -> b h t d", h=4)
print(torch.equal(a, b), torch.equal(rearrange(b, "b h t d -> b t (h d)"), x))
```

### c 常見錯誤 — 狀態：可用

**題目**

下面三段程式各會出什麼錯？先不要執行，寫下你猜的錯誤類型（形狀／device／dtype／autograd），再執行對答案。

```python,ignore
torch.ones(2, 3) + torch.ones(3, 2)
torch.arange(5).mean()
torch.ones(2, requires_grad=True).backward()
```

**練習目標**：看到訊息能分類。

**批改重點**：形狀（`size of tensor a (3) must match`）、dtype（`mean(): could not infer output dtype`）、autograd（`grad can be implicitly created only for scalar outputs`）。

**參考答案**：執行每一行各自觀察即可，沒有固定程式。

### d 對照 CS336 A1 — 狀態：不出題

純對照表。

### e 雲端 GPU 租用 — 狀態：不出題

流程說明，無法在本機練習。

## 第 7 章 效能分析與記憶體

### 第 1 集 記憶體階層 — 狀態：可用

**題目**

標準模型最後輸出的 logits 形狀是 `(32, 256, 10000)`、float32。不建立這個張量，算出它佔幾 MB，以及它是每一層進出的 `(32, 256, 512)` 張量的幾倍。

**練習目標**：位元組數 = 元素個數 × 每個元素的大小。

**批改重點**：327.68 MB；約 19.5 倍（就是 10000 / 512）。

**參考答案**

```python
logits_bytes = 32 * 256 * 10000 * 4
hidden_bytes = 32 * 256 * 512 * 4
print(logits_bytes / 1e6, "MB")
print(round(logits_bytes / hidden_bytes, 1))
```

### 第 2 集 正確計時 — 狀態：可用

**題目**

在 CPU 上用 `llmz.profile.timeit` 量 256 × 256 與 1024 × 1024 的矩陣乘法，印出每次的毫秒數與 GFLOPs。哪一個的 GFLOPs 比較高？

**練習目標**：用 `lambda` 包住要量的程式；FLOPs 公式 \\( 2n^3 \\)。

**批改重點**：時間每台機器不同。哪個 GFLOPs 高也不一定：大矩陣把固定開銷攤掉了，但 CPU 上小矩陣可能整個放進快取。重點是會用 `timeit` 與 \\( 2n^3 \\)。

**參考答案**

```python
import torch

from llmz.profile import timeit

device = torch.device("cpu")
for n in [256, 1024]:
    a = torch.randn(n, n)
    b = torch.randn(n, n)
    seconds = timeit(lambda: a @ b, device, warmup=2, repeat=10)
    print(n, f"{seconds * 1000:.3f} ms", f"{2 * n**3 / seconds / 1e9:.1f} GFLOPs")
```

### 第 3 集 `torch.profiler` — 狀態：可用

**題目**

用 `torch.profiler` 剖析 `llmz.layers.RMSNorm(512)` 對 `torch.randn(8, 128, 512)` 的 forward，印出統計表。表裡出現了哪幾種 `aten::` 運算？對照 `RMSNorm.forward` 的程式碼，各是哪一行做的？

**練習目標**：讀 profiler 的表；一行 Python 對應一到多個底層運算。

**批改重點**：主要是 `aten::mul` 兩次（平方、乘 weight）、`aten::mean`（裡面再呼叫 `sum` 與 `div_`）、`aten::add`（加 eps）、`aten::sqrt`、`aten::div`（`x / rms`）；另外有 `aten::to`、`copy_`、`empty` 之類的小運算，時間可以忽略。

**參考答案**

```python
import torch
from torch.profiler import ProfilerActivity, profile

from llmz.layers import RMSNorm

norm = RMSNorm(512)
x = torch.randn(8, 128, 512)
with torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as prof:
    norm(x)
print(prof.key_averages().table(sort_by="self_cpu_time_total", max_name_column_width=14))
```

### 第 4 集 記憶體估算 — 狀態：可用

**題目**

一個 1.24 億參數的模型用 AdamW 訓練，參數、梯度、優化器狀態共要幾 GB？再用 `saved_activation_bytes` 量 `llmz.layers.SwiGLU(512, 1344)` 對 `torch.randn(4, 256, 512)` 的 forward 存了幾 MB（排除參數），並說明大約是幾個 `(4, 256, 1344)` 張量的大小。

**練習目標**：`estimate_memory` 與 `saved_activation_bytes(fn, exclude=...)` 的用法。

**批改重點**：1.984 GB；SwiGLU 存了 29.6 MB，約 5.4 個 `(4, 256, 1344)` 張量（每個 5.5 MB）：w1、w3 的輸出、silu 的中間結果與相乘的結果，再加上輸入（2.1 MB）。

**參考答案**

```python
import torch

from llmz.layers import SwiGLU
from llmz.profile import estimate_memory, saved_activation_bytes

print(sum(estimate_memory(124_000_000).values()) / 1e9, "GB")
torch.manual_seed(0)
ffn = SwiGLU(512, 1344)
x = torch.randn(4, 256, 512)
saved = saved_activation_bytes(lambda: ffn(x).sum(), exclude=ffn.parameters())
print(saved / 1e6, "MB")
print(saved / (4 * 256 * 1344 * 4))
```

### 第 5 集 混合精度 autocast — 狀態：可用

**題目**

把 256、257、258、259 轉成 bfloat16 再轉回來，印出結果。為什麼 257 變了？再用 float16 試一次。

**練習目標**：bfloat16 只有 7 位元尾數，256 到 512 之間只能表示偶數。

**批改重點**：bfloat16 得到 256、256、258、260（257、259 捨入到最近的偶數）；float16 四個都不變。

**參考答案**

```python
import torch

values = torch.tensor([256.0, 257.0, 258.0, 259.0])
print(values.to(torch.bfloat16).float())
print(values.to(torch.float16).float())
```

### 第 6 集 activation checkpointing — 狀態：可用

**題目**

用 `ModelConfig(vocab_size=100, context_length=32, d_model=64, n_layers=n, n_heads=4, d_ff=172)`，batch 8，量 `n_layers` 為 2 與 6 時、開與不開 `activation_checkpoint` 的 activations 位元組數（排除參數）。checkpoint 之後每多一層多了幾位元組？和每一層的輸入大小比一比。

**練習目標**：checkpoint 之後每層只存輸入。

**批改重點**：開了 checkpoint，6 層減 2 層的差除以 4 正好是一個 `(8, 32, 64)` float32 張量，65536 位元組。

**參考答案**

```python
import torch

from llmz.functional import cross_entropy
from llmz.model import ModelConfig, TransformerLM
from llmz.profile import saved_activation_bytes

x = torch.randint(0, 100, (8, 33))
result = {}
for n in [2, 6]:
    for flag in [False, True]:
        torch.manual_seed(0)
        model = TransformerLM(ModelConfig(vocab_size=100, context_length=32, d_model=64, n_layers=n,
                                          n_heads=4, d_ff=172, activation_checkpoint=flag))
        fn = lambda: cross_entropy(model(x[:, :-1]).reshape(-1, 100), x[:, 1:].reshape(-1))
        result[n, flag] = saved_activation_bytes(fn, exclude=model.parameters())
print(result)
print((result[6, True] - result[2, True]) / 4, 8 * 32 * 64 * 4)
```

### 第 7 集 `torch.compile` — 狀態：不出題

實測需要 MPS 或 CUDA，CPU 上編譯又慢又不具代表性。讀者想練習時，請他照正文在自己的 Mac 上量 RMSNorm 的 eager 與 compiled，並用第 1 集的頻寬算出「搬幾趟」對不對得上。

### 第 8 集 arithmetic intensity — 狀態：可用

**題目**

算標準模型裡三個運算的 arithmetic intensity（float32）：`q_proj`（`(8192, 512) @ (512, 512)`）、`lm_head`（`(8192, 512) @ (512, 10000)`）、attention 的「分數 @ V」（一個 head，`(256, 256) @ (256, 32)`）。哪個在 M4 Max 上（分界點 37）卡在記憶體？

**練習目標**：intensity = FLOPs ÷ 讀寫位元組。

**批改重點**：約 124、230、13；分數 @ V 卡在記憶體，因為要讀整個 256 × 256 的分數矩陣，卻只輸出 32 維。

**參考答案**

```python
def matmul_intensity(m, k, n, bytes_per=4):
    return 2 * m * k * n / ((m * k + k * n + m * n) * bytes_per)


print(f"q_proj {matmul_intensity(8192, 512, 512):.0f}")
print(f"lm_head {matmul_intensity(8192, 512, 10000):.0f}")
print(f"scores @ V {matmul_intensity(256, 256, 32):.0f}")
```

## 第 8 章 FlashAttention 的原理

### 第 1 集 attention 的 O(T²) 記憶體 — 狀態：可用

**題目**

標準模型（batch 32、16 個 head、4 層）每一層的 attention 存了約兩個分數矩陣大小的 float32 資料。長度從 256 改成 1024 時，4 層的 attention 部分合計要存幾 GB？和第 7 章第 4 集量到的整個模型 3011 MB 比較。

**練習目標**：\\( B \cdot H \cdot T^2 \cdot 4 \\) 位元組，長度變 4 倍、大小變 16 倍。

**批改重點**：一個分數矩陣 2.15 GB，兩個乘 4 層約 17.2 GB，光這部分就是長度 256 時整個模型 activations 的 5 倍多。

**參考答案**

```python
one = 32 * 16 * 1024 * 1024 * 4
print(one / 1e9, "GB 一個分數矩陣")
print(2 * 4 * one / 1e9, "GB 四層")
print(round(2 * 4 * one / 3011e6, 1), "倍")
```

### 第 2 集 tiling — 狀態：可用

**題目**

把正文的分塊矩陣乘法改成支援不整除的情況：`A` 是 `(10, 7)`、`B` 是 `(7, 9)`、塊大小 4，驗證結果等於 `A @ B`。

**練習目標**：分塊迴圈的三層索引；切片超出範圍時自動截短。

**批改重點**：三層迴圈分別走 `A` 的列、`B` 的行、相乘的維度；累加器的形狀要用實際切到的大小，不能寫死 `(b, b)`。

**參考答案**

```python
import torch

torch.manual_seed(0)
A = torch.randn(10, 7)
B = torch.randn(7, 9)
b = 4
C = torch.zeros(10, 9)
for i in range(0, 10, b):
    for j in range(0, 9, b):
        acc = torch.zeros(A[i : i + b].shape[0], B[:, j : j + b].shape[1])
        for t in range(0, 7, b):
            acc += A[i : i + b, t : t + b] @ B[t : t + b, j : j + b]
        C[i : i + b, j : j + b] = acc
print(torch.allclose(C, A @ B, atol=1e-6))
```

### 第 3 集 online softmax — 狀態：可用

**題目**

用 online softmax 的方式，一次讀 3 個，算出 `[1.0, 3.0, 2.0, 6.0, 0.0, 4.0, 5.0]` 的 logsumexp（\\( m + \log \ell \\)），並用 `llmz.functional.logsumexp` 驗證。

**練習目標**：只維護 `m` 和 `l` 兩個數；最大值改變時乘上 `correction`。

**批改重點**：三段的最大值依序是 3、6、6；結果約 6.4578。

**參考答案**

```python
import torch

from llmz.functional import logsumexp

x = torch.tensor([1.0, 3.0, 2.0, 6.0, 0.0, 4.0, 5.0])
m = torch.tensor(float("-inf"))
l = torch.tensor(0.0)
for start in range(0, len(x), 3):
    s = x[start : start + 3]
    m_new = torch.maximum(m, s.max())
    l = l * torch.exp(m - m_new) + torch.exp(s - m_new).sum()
    m = m_new
print(m + torch.log(l), logsumexp(x))
```

### 第 4 集 純 PyTorch 分塊 attention forward — 狀態：可用

**題目**

用 `flash_attention_forward` 算長度 100 的 causal attention，分別用 `block_q=block_k=16`、`32`、`100`，確認三種結果彼此相同（`allclose`），也和第 4 章的 attention 相同。

**練習目標**：塊大小只影響計算順序，不影響結果。

**批改重點**：四者 `allclose`；`block=100` 時只有一塊，等於一次算完。

**參考答案**

```python
import torch

from llmz.attention import causal_mask, scaled_dot_product_attention
from llmz.flash import flash_attention_forward

torch.manual_seed(0)
q, k, v = (torch.randn(1, 2, 100, 8) for _ in range(3))
ref = scaled_dot_product_attention(q, k, v, causal_mask(100))
for b in [16, 32, 100]:
    o, _ = flash_attention_forward(q, k, v, True, block_q=b, block_k=b)
    print(b, torch.allclose(o, ref, atol=1e-5))
```

### 第 5 集 backward 與 recompute — 狀態：可用

**題目**

驗證 \\( D_i = \sum_j P_{ij}\, dP_{ij} \\) 和 \\( dO_i \cdot O_i \\) 相等：隨機產生 `q`、`k`、`v`（長度 6、4 維）和 `do`，算出完整的 `P`、`O = P @ v`、`dP = do @ v.T`，比較兩種算法。

**練習目標**：理解為什麼 backward 不需要整列的 `P` 也能算 \\( D \\)。

**批改重點**：`(P * dP).sum(-1)` 與 `(do * O).sum(-1)` 相等。

**參考答案**

```python
import torch

from llmz.functional import softmax

torch.manual_seed(0)
q, k, v, do = (torch.randn(6, 4) for _ in range(4))
P = softmax(q @ k.T / 2)
O = P @ v
dP = do @ v.T
print(torch.allclose((P * dP).sum(-1), (do * O).sum(-1), atol=1e-6))
```

### 第 6 集 Triton 一瞥 — 狀態：不出題

需要 NVIDIA 顯示卡。讀者有 GPU 而且想練習時，請他照正文的 RMSNorm kernel 寫一個「每列加上 1 再乘 2」的 kernel，和 PyTorch 的 `(x + 1) * 2` 比對。

### 第 7 集 `F.scaled_dot_product_attention` 現況 — 狀態：可用

**題目**

在 CPU 上用 `F.scaled_dot_product_attention` 和 `llmz.flash.flash_attention` 算同一組 causal attention（長度 300），對同一個 loss 做 backward，比較 `q` 的梯度是否相同。

**練習目標**：內建版本只拿來驗證；`grad` 要在兩次之間清掉。

**批改重點**：`allclose` 為 `True`；忘記 `q.grad = None` 會得到兩次相加的梯度。

**參考答案**

```python
import torch
import torch.nn.functional as F

from llmz.flash import flash_attention

torch.manual_seed(0)
q, k, v = (torch.randn(1, 2, 300, 16, requires_grad=True) for _ in range(3))
flash_attention(q, k, v, True).pow(2).sum().backward()
ours = q.grad.clone()
q.grad = None
F.scaled_dot_product_attention(q, k, v, is_causal=True).pow(2).sum().backward()
print(torch.allclose(ours, q.grad, atol=1e-4))
```

## 第 9 章 平行化：原理與 CPU 模擬

### 第 1 集 為什麼需要多卡 — 狀態：可用

**題目**

一個 130 億參數的模型用 AdamW 訓練，光參數、梯度、優化器狀態要幾 GB？至少要幾張 80 GB 的卡才放得下（假設可以完美切開）？

**練習目標**：每個參數 16 位元組。

**批改重點**：208 GB；至少 3 張（208 / 80 = 2.6，無條件進位）。

**參考答案**

```python
import math

size = 16 * 13e9
print(size / 1e9, "GB")
print(math.ceil(size / 80e9), "張")
```

### 第 2 集 Gloo 起 process — 狀態：可用

**題目**

用 `llmz.dist.launch` 起 3 個 process，每個 rank 印出 `rank ** 2`，輸出順序要固定。

**練習目標**：函式定義在最外層、主程式放在 `if __name__ == "__main__":`、用 `print_in_order`。

**批改重點**：輸出 0、1、4 依序出現；忘了 `__main__` 判斷會一直重複起 process 而出錯。

**參考答案**

```python
from llmz.dist import launch, print_in_order


def square(rank, world_size):
    print_in_order(rank**2)


if __name__ == "__main__":
    launch(square, 3)
```

### 第 3 集 集合通訊 — 狀態：可用

**題目**

4 個 rank，每個 rank 有 `torch.tensor([rank + 1.0])`。不用 `dist.all_reduce`，只用 `dist.all_gather` 算出四個值的總和，印出來並和 `dist.all_reduce` 的結果比較。

**練習目標**：all-gather 之後每個人都有全部的資料，可以自己加。

**批改重點**：總和 10；這樣做每個人收了 \\( N - 1 \\) 份完整資料，比 ring all-reduce 傳得多。

**參考答案**

```python
import torch
import torch.distributed as dist

from llmz.dist import launch, print_in_order


def run(rank, world_size):
    x = torch.tensor([rank + 1.0])
    out = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(out, x)
    total = sum(out)
    dist.all_reduce(x)
    print_in_order(total.item(), x.item())


if __name__ == "__main__":
    launch(run, 4)
```

### 第 4 集 naive DDP — 狀態：可用

**題目**

把正文的驗證改成 4 個 rank、batch 8（每個 rank 2 筆），確認梯度仍然和單機相同。

**練習目標**：每個 rank 取 `ids[rank * 2 : rank * 2 + 2]`；all-reduce 後除以 world size。

**批改重點**：四個 rank 都印 `True`。

**參考答案**

```python
import torch

from llmz.dist import all_reduce_gradients, broadcast_parameters, launch, print_in_order
from llmz.functional import cross_entropy
from llmz.model import ModelConfig, TransformerLM

CFG = ModelConfig(vocab_size=50, context_length=8, d_model=32, n_layers=2, n_heads=4, d_ff=64)


def loss_of(model, ids):
    return cross_entropy(model(ids[:, :-1]).reshape(-1, 50), ids[:, 1:].reshape(-1))


def check(rank, world_size):
    torch.manual_seed(0)
    ids = torch.randint(0, 50, (8, 9))
    torch.manual_seed(100)
    single = TransformerLM(CFG)
    torch.manual_seed(100 + rank)
    model = TransformerLM(CFG)
    broadcast_parameters(model)
    loss_of(model, ids[rank * 2 : rank * 2 + 2]).backward()
    all_reduce_gradients(model)
    loss_of(single, ids).backward()
    print_in_order(all(torch.allclose(p.grad, q.grad, atol=1e-6) for p, q in zip(model.parameters(), single.parameters())))


if __name__ == "__main__":
    launch(check, 4)
```

### 第 5 集 bucketing 與通訊重疊 — 狀態：可用

**題目**

建立標準模型的 `BucketedDDP`，bucket 大小分別設 5、25、100 MB，印出各有幾個 bucket。（只要建立就好，不用訓練；用 1 個 rank 即可。）

**練習目標**：bucket 大小與數量的取捨。

**批改重點**：bucket 越大數量越少；100 MB 大於全部梯度 90.8 MB，只剩 1 個。

**參考答案**

```python
import torch

from llmz.dist import BucketedDDP, launch
from llmz.model import ModelConfig, TransformerLM


def run(rank, world_size):
    for mb in [5, 25, 100]:
        torch.manual_seed(0)
        print(mb, "MB：", len(BucketedDDP(TransformerLM(ModelConfig()), bucket_mb=mb).buckets), "個 bucket")


if __name__ == "__main__":
    launch(run, 1)
```

### 第 6 集 optimizer state sharding — 狀態：可用

**題目**

不起 process，模擬 `ShardedOptimizer` 的分配方式：把標準模型的參數從大到小，每個交給目前負責元素最少的 rank，分給 4 個 rank，印出每個 rank 負責的元素數。

**練習目標**：大的先分、交給最少的人。

**批改重點**：四個數字很接近（約 565 萬到 570 萬）；如果照原本順序輪流分，差距會大很多。

**參考答案**

```python
from llmz.model import ModelConfig, TransformerLM

sizes = sorted((p.numel() for p in TransformerLM(ModelConfig()).parameters()), reverse=True)
load = [0] * 4
for n in sizes:
    load[load.index(min(load))] += n
print(load)
```

### 第 7 集 FSDP 的想法 — 狀態：可用

**題目**

用正文的表格公式，算 700 億參數的模型在 64 張卡上，ZeRO 1、2、3 每張卡各要幾 GB（不算 activations）。哪幾種放得進 80 GB？

**練習目標**：\\( 4\Psi + 4\Psi + 8\Psi/N \\) 等公式。

**批改重點**：ZeRO 1 約 568.8 GB、ZeRO 2 約 293.1 GB、ZeRO 3 約 17.5 GB；只有 ZeRO 3 放得進。

**參考答案**

```python
psi, n = 70e9, 64
print(f"{(8 * psi + 8 * psi / n) / 1e9:.1f}")
print(f"{(4 * psi + 12 * psi / n) / 1e9:.1f}")
print(f"{16 * psi / n / 1e9:.1f}")
```

### 第 8 集 tensor / pipeline parallelism — 狀態：可用

**題目**

pipeline 有 8 個階段。至少要幾個 micro-batch，bubble 的閒置比例才會低於 10%？

**練習目標**：\\( (p-1)/(m+p-1) \\)。

**批改重點**：\\( 7/(m+7) < 0.1 \\)，\\( m > 63 \\)，所以至少 64 個。

**參考答案**

```python
p = 8
m = 1
while (p - 1) / (m + p - 1) >= 0.1:
    m += 1
print(m)
```

### 第 9 集 通訊與計算的估算 — 狀態：可用

**題目**

用本機實測的 Gloo 頻寬 0.5 GB/s，以及 MPS 上約 10 TFLOPs 的計算速度（第 7 章第 1 集），算出每個 rank 每步至少要看多少 token，計算時間才會大於通訊時間（梯度用 float32，公式要改成什麼？）。

**練習目標**：比值公式的推導，位元組數從 2 換成 4。

**批改重點**：float32 時通訊量加倍，比值變成 \\( 0.75\, BT \cdot \text{BW} / F \\)，門檻是 \\( F / (0.75\,\text{BW}) \\)，約 2.7 萬個 token，是第 5 章一步 8192 個 token 的三倍多。

**參考答案**

```python
flops = 10e12
bandwidth = 0.5e9
print(f"{flops / (0.75 * bandwidth):,.0f} 個 token")
print(32 * 256, "個 token（第 5 章一步）")
```

## 第 10 章 Scaling laws

### 第 1 集 什麼是 scaling law — 狀態：可用

**題目**

用 Kaplan 的 \\( L(N) = (N_c/N)^{0.076} \\)（\\( N_c = 8.8 \times 10^{13} \\)），算出 loss 從 3.0 降到 2.5，參數量要變成幾倍。

**練習目標**：冪律的比例只和 loss 的比值有關。

**批改重點**：\\( N_2 / N_1 = (L_1 / L_2)^{1/0.076} = 1.2^{13.16} \approx 11 \\) 倍；和 \\( N_c \\) 無關。

**參考答案**

```python
alpha, n_c = 0.076, 8.8e13
n1 = n_c / 3.0 ** (1 / alpha)
n2 = n_c / 2.5 ** (1 / alpha)
print(f"{n1:.2e} → {n2:.2e}，{n2 / n1:.1f} 倍")
```

### 第 2 集 Chinchilla 與 IsoFLOPs — 狀態：可用

**題目**

用 `chinchilla_loss`，比較同樣 \\( 10^{23} \\) FLOPs 下兩種分配：100 億參數，以及 1000 億參數。各看幾個 token？哪一個 loss 比較低？

**練習目標**：\\( D = C / (6N) \\)；預算固定時大模型不一定比較好。

**批改重點**：100 億參數看約 1.7 兆個 token、1000 億參數只看約 1700 億個；100 億的 loss 較低。

**參考答案**

```python
from llmz.scaling import chinchilla_loss

c = 1e23
for n in [1e10, 1e11]:
    d = c / (6 * n)
    print(f"N = {n:.0e}  D = {d:.2e}  loss = {chinchilla_loss(n, d):.4f}")
```

### 第 3 集 擬合 IsoFLOPs 曲線 — 狀態：可用

**題目**

故意把掃描範圍放偏：\\( C = 10^{20} \\)，參數量只掃 \\( 10^{7} \\) 到 \\( 10^{8} \\) 的 9 個點（沒有雜訊）。用 `fit_isoflop_minimum` 擬合，和真正的最佳值（約 \\( 6.45 \times 10^8 \\)）比較，說明發生了什麼事。

**練習目標**：最低點落在掃描範圍外時，拋物線外推不可靠。

**批改重點**：9 個點的 loss 一路往下，最小的是範圍邊上的 \\( 10^8 \\)；拋物線的頂點約 \\( 2.9 \times 10^8 \\)，落在範圍外，和真正答案差了一倍多。正確的做法是往大的方向多掃幾個點。

**參考答案**

```python
import numpy as np

from llmz.scaling import chinchilla_loss, fit_isoflop_minimum

c = 1e20
n = np.logspace(7, 8, 9)
loss = chinchilla_loss(n, c / (6 * n))
print(np.round(loss, 4))
print(f"{fit_isoflop_minimum(n, loss):.2e}")
```

### 第 4 集 擬合冪律 — 狀態：可用

**題目**

不加雜訊，對 \\( 10^{18} \\) 到 \\( 10^{22} \\) 五個預算各用 `chinchilla_loss` 在密的格點上找到最佳 \\( N \\)，再用 `fit_power_law` 擬合指數，和 \\( \beta/(\alpha+\beta) \\) 比較。

**練習目標**：沒有雜訊時，擬合應該幾乎完全吻合推導。

**批改重點**：擬合出 0.4516，和 \\( 0.28 / 0.62 = 0.4516 \\) 一致到小數第四位。

**參考答案**

```python
import numpy as np

from llmz.scaling import chinchilla_loss, fit_power_law

budgets = np.array([1e18, 1e19, 1e20, 1e21, 1e22])
grid = np.logspace(6, 12, 20000)
best = [grid[np.argmin(chinchilla_loss(grid, c / (6 * grid)))] for c in budgets]
k, a = fit_power_law(budgets, np.array(best))
print(f"{a:.4f}", f"{0.28 / 0.62:.4f}")
```

### 第 5 集 Mac 上的迷你 scaling law — 狀態：可用

**題目**

用本集的結果表，把 \\( 3 \times 10^{14} \\) 的 d64 拿掉（只剩四個點）重新擬合。這個預算的最佳參數量、以及三個預算擬合出來的指數各變了多少？

**練習目標**：體會少數幾個點的擬合對單一資料點有多敏感。

**批改重點**：\\( 3 \times 10^{14} \\) 的頂點從 \\( 3.20 \times 10^6 \\) 移到 \\( 3.87 \times 10^6 \\)，差了兩成；指數只從 0.436 變成 0.434，因為三個點的直線主要由兩端的預算決定。中間的點不穩，指數看起來卻很穩，不代表指數可靠。

**參考答案**

```python
import numpy as np

from llmz.scaling import fit_isoflop_minimum, fit_power_law

n = {"d64": 746_816, "d128": 1_657_472, "d192": 3_248_448, "d256": 5_773_568, "d384": 14_461_824}
results = {
    1e14: {"d64": 2.6687, "d128": 2.5633, "d192": 2.5516, "d256": 2.7142},
    3e14: {"d128": 2.2566, "d192": 2.1331, "d256": 2.1350, "d384": 2.4697},
    1e15: {"d128": 2.0080, "d192": 1.8694, "d256": 1.8375, "d384": 1.9577},
}
best = [fit_isoflop_minimum([n[k] for k in losses], list(losses.values())) for losses in results.values()]
k, a = fit_power_law(np.array(list(results)), np.array(best))
print(f"{best[1]:.2e}", f"{a:.3f}")
```

### 第 6 集 限制與 μP 一瞥 — 狀態：可用

**題目**

Llama 3 8B 用了約 15 兆個 token 訓練。用 \\( C = 6ND \\) 算它的訓練計算量；如果改成每個參數 20 個 token 的 Chinchilla 最佳分配，同樣的計算量可以訓練多大的模型？

**練習目標**：\\( C = 6ND \\) 與 \\( D = 20N \\) 聯立。

**批改重點**：\\( C \approx 7.2 \times 10^{23} \\)；\\( 120 N^2 = C \\)，\\( N \approx 775 \\) 億。Meta 選擇了小得多的模型，換取推論便宜。

**參考答案**

```python
c = 6 * 8e9 * 15e12
print(f"{c:.1e} FLOPs")
print(f"{(c / 120) ** 0.5:.2e} 個參數")
```

## 第 11 章 推論

### 第 1 集 KV cache — 狀態：可用

**題目**

prompt 有 100 個 token，生成 156 個（剛好填滿 256 的 context）。沒有 cache、每一步重算整段時，模型一共處理幾個位置？是有 cache 時的幾倍？

**練習目標**：重算的總量是等差數列的和。

**批改重點**：\\( 100 + 101 + \cdots + 255 = 27690 \\) 個位置；有 cache 時 255 個（最後一個 token 不用再餵），約 109 倍。

**參考答案**

```python
p, n = 100, 156
naive = sum(range(p, p + n))
print(naive, p + n - 1, f"{naive / (p + n - 1):.0f} 倍")
```

### 第 2 集 prefill 與 decode — 狀態：可用

**題目**

標準模型的一個 `Linear(512, 1344)`（float32）只算讀權重的位元組，餵進 \\( T \\) 個 token 時 arithmetic intensity 是多少？\\( T \\) 至少多大才超過 M4 Max 的平衡點 37 FLOPs/位元組？

**練習目標**：intensity 約 \\( T/2 \\)，和矩陣大小無關。

**批改重點**：\\( 2nT / 4n = T/2 \\)；\\( T = 1 \\) 是 0.5，\\( T \ge 74 \\) 才超過 37。

**參考答案**

```python
n = 512 * 1344
for t in [1, 64, 74, 256]:
    print(t, 2 * n * t / (4 * n))
```

### 第 3 集 延遲與記憶體估算 — 狀態：可用

**題目**

用 Llama 3 8B（bfloat16）服務滿 8192 個 token 的序列。權重之外還有 16 GB 可以放 KV cache，能同時放幾條？如果它沒有用 GQA（32 個 KV head）呢？

**練習目標**：KV cache 的公式與 GQA 的效果。

**批改重點**：一條 1 GiB，16 GB 放得下 14 條；沒有 GQA 時一條 4 GiB，只放得下 3 條。

**參考答案**

```python
kv_seq = 2 * 32 * 8 * 128 * 2 * 8192
print(kv_seq / 2**30, "GiB")
print(int(16e9 // kv_seq), "條（8 個 KV head）")
print(int(16e9 // (kv_seq * 4)), "條（32 個 KV head）")
```

### 第 4 集 batching — 狀態：可用

**題目**

標準模型 decode 一步要讀全部權重，加上 batch 裡每條序列的 KV cache。cache 裡已有 128 個位置時，\\( B = 1 \\) 與 \\( B = 256 \\) 平均每生成一個 token 要讀多少位元組？cache 佔多少？

**練習目標**：權重被 batch 分攤，cache 不會。

**批改重點**：\\( B = 1 \\) 每個 token 92.88 MB，cache 只佔 2%；\\( B = 256 \\) 只要 2.45 MB，但 86% 是 cache。batch 大時瓶頸從權重移到 KV cache。

**參考答案**

```python
weights = 22_696_448 * 4
cache_seq = 2 * 4 * 16 * 32 * 4 * 128
for b in [1, 256]:
    per_token = (weights + b * cache_seq) / b
    print(f"B={b}：每個 token 讀 {per_token / 1e6:.2f} MB，其中 cache 佔 {cache_seq / per_token:.0%}")
```

### 第 5 集 speculative decoding — 狀態：可用

**題目**

假設每個草稿 token 都以機率 \\( \alpha = 0.8 \\) 被接受（彼此獨立），每輪平均產生 \\( (1-\alpha^{k+1})/(1-\alpha) \\) 個 token；草稿模型一步的成本是目標模型的 \\( c = 0.25 \\) 倍。加速倍數約是「每輪 token 數 ÷ \\( (1 + kc) \\)」，\\( k \\) 從 1 到 8 哪一個最好？

**練習目標**：\\( k \\) 太小猜得少，太大浪費在被拒絕的草稿上。

**批改重點**：\\( k = 3 \\) 最好，約 1.69 倍；\\( k = 4 \\) 的 1.68 倍差不多，之後下降。

**參考答案**

```python
alpha, c = 0.8, 0.25
for k in range(1, 9):
    tokens = (1 - alpha ** (k + 1)) / (1 - alpha)
    print(k, f"{tokens:.2f}", f"{tokens / (1 + k * c):.2f}")
```

### 第 6 集 量化初探 — 狀態：可用

**題目**

做一個每列大小差到一千倍的矩陣，比較「整個矩陣只用一個縮放」與「每列一個縮放」的 int8 量化，看最小與最大那一列的相對誤差。

**練習目標**：縮放的單位越大，小的值越容易被四捨五入成 0。

**批改重點**：只用一個縮放時，最小那一列全部變成 0，相對誤差 1.000；每列一個縮放時兩列都在 1% 以內。提示：把矩陣 `reshape(1, -1)` 就能讓 `quantize` 只算一個縮放。

**參考答案**

```python
import torch

from llmz.inference import dequantize, quantize

torch.manual_seed(0)
w = torch.randn(64, 256) * torch.logspace(-3, 0, 64)[:, None]  # 每一列的大小差到一千倍
per_tensor = quantize(w.reshape(1, -1), 8)  # 整個矩陣當成一列：只有一個縮放
per_channel = quantize(w, 8)
for name, (q, scale) in [("整個矩陣一個縮放", per_tensor), ("每列一個縮放", per_channel)]:
    w_hat = dequantize(q, scale).reshape(w.shape)
    rel = ((w_hat - w).abs().mean(dim=1) / w.abs().mean(dim=1))
    print(f"{name}：最小那一列的相對誤差 {rel[0]:.3f}，最大那一列 {rel[-1]:.4f}")
```

### 第 7 集 MoE 與 attention 替代方案 — 狀態：可用

**題目**

一個像 Mixtral 8x7B 的模型：32 層、`d_model` 4096、`d_ff` 14336 的 SwiGLU 專家，每層 8 個、每個 token 用 2 個。只算專家的參數，總共多少？每個 token 用到多少？

**練習目標**：MoE 的總參數與活躍參數。

**批改重點**：每個專家 1.76 億；總共約 451 億，每個 token 用約 113 億。加上 attention 與 embedding，就接近 Mixtral 公布的 467 億與 129 億。

**參考答案**

```python
d_model, d_ff, n_layers, n_experts, top_k = 4096, 14336, 32, 8, 2
expert = 3 * d_model * d_ff
print(f"每個專家 {expert:,}")
print(f"全部專家 {expert * n_experts * n_layers / 1e9:.1f}B，每個 token 用 {expert * top_k * n_layers / 1e9:.1f}B")
```

## 附錄二

### a CUDA 與 Triton 入門指引 — 狀態：可用

**題目**

用附錄二 a 的 CUDA 向量相加處理 1,000,000 個元素、每個 block 256 個 thread。要開幾個 block？最後一個 block 有幾個 thread 因為 `if (i < n)` 什麼都不做？全部一共幾個 warp？

**練習目標**：grid、block、warp 的換算，以及為什麼 kernel 要檢查邊界。

**批改重點**：3907 個 block（無條件進位）；最後一個 block 有 192 個 thread 閒著；共 31256 個 warp。沒有 `if (i < n)` 時，這 192 個 thread 會寫到陣列外面。

**參考答案**

```python
n, threads = 1_000_000, 256
blocks = (n + threads - 1) // threads
print(blocks, "個 block，", blocks * threads - n, "個 thread 閒著")
print(blocks * threads // 32, "個 warp")
```

### b 對照 CS336 A2、A3 — 狀態：不出題

純對照表。

## 第 12 章 評估

### 第 1 集 perplexity 的限制 — 狀態：可用

**題目**

同一段 600 bytes 的文字，模型 A 的 tokenizer 把它切成 150 個 token，總 \\( -\log p \\) 是 345 nat；模型 B 的 tokenizer 切成 100 個 token，總 \\( -\log p \\) 是 310 nat。算出兩個模型每個 token 的 perplexity 與 bits/byte。誰比較好？

**練習目標**：perplexity 的單位是每個 token，不同 tokenizer 要換成 bits/byte 才能比。

**批改重點**：A 的 perplexity 9.97、B 是 22.20，只看 perplexity 會選 A；但 bits/byte A 是 0.830、B 是 0.745，B 其實比較好。bits/byte 是總 \\( -\log p \\) 除以 byte 數，再除以 \\( \ln 2 \\)。

**參考答案**

```python
import math

models = {"A": (150, 345.0), "B": (100, 310.0)}  # (token 數, 總 -log 機率 nat)
for name, (n_tokens, nll) in models.items():
    print(f"模型 {name}：每個 token 的 perplexity {math.exp(nll / n_tokens):.2f}，{nll / 600 / math.log(2):.3f} bits/byte")
```

### 第 2 集 benchmark 與 few-shot — 狀態：可用

**題目**

四選一的題庫，隨機猜測的準確率是 25%，標準差是 \\( \sqrt{p(1-p)/n} \\)（\\( n \\) 是題數）。題數是 100、400、1000、4000 時標準差各是幾個百分點？至少要幾題，標準差才會小於等於 1 個百分點？

**練習目標**：題數決定分數的雜訊，100 題的題庫分不出差幾個百分點的兩個模型。

**批改重點**：100 題 4.33、400 題 2.17、1000 題 1.37、4000 題 0.68 個百分點；\\( 0.1875 / 0.01^2 = 1875 \\) 題。

**參考答案**

```python
p = 0.25  # 四選一亂猜的準確率
for n in [100, 400, 1000, 1875, 4000]:
    print(f"{n} 題：標準差 {(p * (1 - p) / n) ** 0.5 * 100:.2f} 個百分點")
```

### 第 3 集 loglikelihood 評多選題 — 狀態：可用

**題目**

兩個選項的總 log 機率是 \\( -5 \\)（1 個 token）與 \\( -6 \\)（4 個 token）。把分數改成 \\( \log p / \text{長度}^\alpha \\)，\\( \alpha = 0 \\) 是總和，\\( \alpha = 1 \\) 是每個 token 的平均。\\( \alpha \\) 從 0 開始每次加 0.05，第一次改選第二個選項是在哪裡？

**練習目標**：長度正規化是一個連續的旋鈕，不是兩種截然不同的算法；`pick_choice` 的 `lengths` 可以傳任何分母。

**批改重點**：\\( \alpha = 0.15 \\) 時第一次改選第二個。精確的門檻是 \\( \ln 1.2 / \ln 4 \approx 0.1315 \\)：在這之後 \\( -6/4^\alpha \\) 比 \\( -5 \\) 大。提示：把 `lengths ** alpha` 傳給 `pick_choice`。

**參考答案**

```python
import math

import torch

from llmz.evaluation import pick_choice

logprobs = torch.tensor([-5.0, -6.0])  # 第一個選項 1 個 token，第二個 4 個 token
lengths = torch.tensor([1.0, 4.0])
alphas = [i / 20 for i in range(21)]  # 0, 0.05, ..., 1
flip = next(a for a in alphas if pick_choice(logprobs, lengths**a) == 1)
print(f"alpha = {flip} 時第一次改選第二個選項（精確的門檻 {math.log(1.2) / math.log(4):.4f}）")
```

### 第 4 集 生成式評估與 LLM-as-judge — 狀態：可用

**題目**

寫一個比 `exact_match` 寬鬆的 `contains_match(prediction, reference)`：兩邊各自正規化（`normalize_answer`）之後，只要標準答案的字連續出現在預測裡就算對（要以整個字為單位，`cat` 不能算在 `scattered` 裡）。對下面 5 組（預測，標準答案）各印出 `exact_match` 與 `contains_match` 的結果，說明放寬之後多出來的「對」有沒有誤判：

1. `"A red ball."`、`"red ball"`
2. `"water, and then he drank it all"`、`"water"`
3. `"I do not like the park"`、`"like the park"`
4. `"scattered"`、`"cat"`
5. `"cold"`、`"cold water"`

**練習目標**：放寬評分規則換來的是另一種錯誤；exact match 太嚴，包含式又太鬆。

**批改重點**：第 1 組兩個都對；第 2 組只有 contains 對，而且是合理的（答案確實在裡面）；第 3 組 contains 也對，但這是誤判：句子的意思是不喜歡，包含式看不出否定；第 4、5 組兩個都錯。提示：正規化之後在兩邊各加一個空格再用 `in`，就能以整個字比對。

**參考答案**

```python
from llmz.evaluation import exact_match, normalize_answer


def contains_match(prediction: str, reference: str) -> bool:
    return f" {normalize_answer(reference)} " in f" {normalize_answer(prediction)} "  # 兩邊加空格：只算整個字


pairs = [
    ("A red ball.", "red ball"),
    ("water, and then he drank it all", "water"),
    ("I do not like the park", "like the park"),
    ("scattered", "cat"),
    ("cold", "cold water"),
]
for prediction, reference in pairs:
    print(f"{prediction!r} 對 {reference!r}：exact {exact_match(prediction, reference)}，contains {contains_match(prediction, reference)}")
```

### 第 5 集 資料污染 — 狀態：可用

**題目**

在隨機 token 組成的 5000 個 token 的「語料」裡，取一段長度 40 的 token 當題目，把正中間（第 20 個）token 換成語料裡不會出現的值 `99999`。分別用 \\( n = 13 \\)、8、5 查，印出 `contamination_rate`，再手算：為什麼是這些數字？接著對一段長度只有 20 的題目做同樣的事，\\( n = 13 \\) 的結果是多少？

**練習目標**：只改一個 token，就會讓包含它的至多 \\( n \\) 個 n-gram 都查不到；題目越短、\\( n \\) 越大，剩下能偵測的越少。

**批改重點**：長度 40 有 \\( 40 - n + 1 \\) 個 n-gram，包含中間那個 token 的有 \\( n \\) 個，所以比例是 \\( (28 - 13)/28 = 0.5357 \\)（\\( n = 13 \\)）、\\( 25/33 = 0.7576 \\)（\\( n = 8 \\)）、\\( 31/36 = 0.8611 \\)（\\( n = 5 \\)）。長度 20 時 \\( n = 13 \\) 只有 8 個 n-gram，每一個都包含中間那個 token，比例是 0：改一個 token 就整題逃過 13-gram 偵測。\\( n = 8 \\) 是 \\( 5/13 = 0.3846 \\)，\\( n = 5 \\) 是 \\( 11/16 = 0.6875 \\)。

**參考答案**

```python
import numpy as np

from llmz.evaluation import contamination_rate, ngram_index

rng = np.random.default_rng(0)
corpus = rng.integers(0, 10000, 5000)  # 假裝這是訓練集
for length in [40, 20]:
    edited = corpus[1000 : 1000 + length].copy()  # 從語料裡抄一段當題目
    edited[length // 2] = 99999  # 換掉正中間一個 token，這個值語料裡不會有
    for n in [13, 8, 5]:
        print(f"長度 {length}，n = {n}：{contamination_rate(edited, ngram_index(corpus, n), n):.4f}")
```

## 第 13 章 資料：從網頁到訓練集

### 第 1 集 Common Crawl 與 WARC — 狀態：可用

**題目**

WARC 檔裡每一筆記錄都是自己的一段 gzip，所以知道一筆記錄在檔案裡的起點（offset）與長度（length），就能只讀那一段，不必從頭讀。自己造一個 3 筆記錄的 `.warc.gz`（記錄內容自己編），記下每一段壓縮後的長度，寫一個函式 `read_at(path, offset, length)`：只讀檔案裡那一段 byte，用 `gzip.decompress` 解開，回傳記錄的原始 bytes。用它單獨讀出第 2 筆，印出它的 `WARC-Type`。

**練習目標**：每筆記錄是獨立的 gzip 段，所以能用 Range 只取一筆；Common Crawl 的索引就是這樣用的。

**批改重點**：第 2 筆的 offset 是前面所有段的壓縮後長度加起來，length 是它自己壓縮後的長度（不是解壓縮後的長度）；用 `f.seek(offset)` 與 `f.read(length)` 讀；讀出來的是 `WARC-Type: request`。提示：`gzip.compress` 回傳的 bytes 長度就是這一段在檔案裡佔的長度。常見錯誤：把整個檔案餵給 `gzip.decompress`，得到的是全部記錄接在一起。

**參考答案**

```python
import gzip


def read_at(path, offset, length):
    with open(path, "rb") as f:
        f.seek(offset)
        return gzip.decompress(f.read(length))  # 這一段自己就是完整的 gzip


members = []
for warc_type in ["warcinfo", "request", "response"]:
    body = f"toy {warc_type} content".encode()
    head = f"WARC/1.0\r\nWARC-Type: {warc_type}\r\nContent-Length: {len(body)}\r\n\r\n".encode()
    members.append(gzip.compress(head + body + b"\r\n\r\n"))
with open("toy.warc.gz", "wb") as f:
    f.write(b"".join(members))

offset, length = len(members[0]), len(members[1])  # 第 2 筆：前面各段的長度加起來，就是它的起點
record = read_at("toy.warc.gz", offset, length)
print(f"offset {offset}，length {length}")
print(record.split(b"\r\n")[1].decode())
```

### 第 2 集 HTML 轉文字 — 狀態：可用

**題目**

先執行 `uv run python tools/prepare_webtext.py` 取得 `data/cc/sample.warc.gz`。用 `html_to_text` 的 `skip_tags` 參數，量出 `select`、`nav`、`footer`、`aside` 各自單獨加進 `{"script", "style", "noscript", "template", "svg", "iframe", "title"}` 之後，在那 842 頁 HTML 上多丟了幾個字元。哪一個標籤丟得最多？四個數字加起來，和第 2 集「一起丟」多丟的 506,589 個字元一樣嗎？為什麼？

**練習目標**：用 `skip_tags` 參數做消融實驗，並看到標籤會巢狀、互相重疊。

**批改重點**：`select` 125,562、`nav` 149,723、`footer` 117,575、`aside` 170,507 個字元，`aside` 最多；四個加起來是 563,367，比一起丟的 506,589 多。因為標籤可以巢狀（例如 `footer` 裡面再放一個 `nav`）：同一段文字在兩個要丟的標籤裡面時，各自單獨量每個都算了一次，一起丟時只算一次。提示：`html_to_text(html, skip_tags=...)`，集合可以用 `base | {tag}` 加一個標籤。

**參考答案**

```python,norun
from llmz.pipeline import decode_html, html_to_text, iter_warc_records, split_http_response

htmls = []
for headers, body in iter_warc_records("data/cc/sample.warc.gz"):
    if headers["warc-type"] == "response":
        status, http_headers, payload = split_http_response(body)
        if status == 200 and "html" in http_headers.get("content-type", "").lower():
            htmls.append(decode_html(payload, http_headers["content-type"]))

base = {"script", "style", "noscript", "template", "svg", "iframe", "title"}
before = sum(len(html_to_text(h, base)) for h in htmls)
for tag in ["select", "nav", "footer", "aside"]:
    after = sum(len(html_to_text(h, base | {tag})) for h in htmls)
    print(f"{tag}：多丟 {before - after:,} 個字元")
```

### 第 3 集 語言辨識 — 狀態：可用

**題目**

第 3 集的 `train_langid` 預設 `n=3`、`alpha=0.1`，是隨手選的。把 `n` 換成 1、2、3、4，`alpha` 換成 0.01、0.1、1.0，共 12 組，各自量三個數字：60 句測試句的正確數；在 318 頁「一半以上是韓文、漢字或西里爾字母」的網頁上，整頁分類與 `predict_lines` 逐行投票各對幾頁（正解用書寫系統判斷，做法見第 3 集）。`n=1`（只看單一字元）夠用嗎？哪一個設定你會選？

**練習目標**：用 `NGramClassifier.train(samples, n, alpha)` 做超參數的比較，並看到「測試集全對」與「真實網頁上的表現」會給出不同的結論。

**批改重點**：測試句在 `n >= 3` 時全對（60/60），`n = 1` 是 59/60（`alpha = 1.0` 時 56/60）；可是在 318 頁網頁上，整頁分類反而是 `n = 1` 最好（`alpha = 0.01` 時 280 頁，`n = 3`、`alpha = 0.1` 是 245 頁），`alpha` 愈大愈差；逐行投票幾乎不受影響（303 到 311 頁）。可能的原因：網頁上的中文字多半在訓練資料裡只出現一兩次，`n` 愈大，能對上的片段愈少；整頁分類又很容易被認得的英文片段拉走；逐行投票把後面這個問題擋掉了。注意這只是在這一份資料、這個代理的正解上量到的，不要說成「`n = 1` 永遠比較好」；選哪一組沒有標準答案，要能說明依據。提示：`alpha` 愈小，「這個語言看過、另一個語言沒看過」的片段，證據愈強。常見錯誤：只看測試句的結果就選 `n = 3`。

**參考答案**

```python,norun
from llmz.pipeline import NGramClassifier, load_langid_samples, read_pages

SCRIPTS = {"han": (0x4E00, 0x9FFF), "kana": (0x3040, 0x30FF), "hangul": (0xAC00, 0xD7AF), "cyrillic": (0x0400, 0x04FF)}


def expected_language(text):
    letters = [ord(ch) for ch in text if ch.isalpha()]
    s = {name: sum(lo <= c <= hi for c in letters) / max(len(letters), 1) for name, (lo, hi) in SCRIPTS.items()}
    if s["hangul"] >= 0.5:
        return "ko"
    if s["han"] + s["kana"] >= 0.5:
        return "ja" if s["kana"] >= 0.05 else "zh"
    if s["cyrillic"] >= 0.5:
        return "ru"


train, test = load_langid_samples("train"), load_langid_samples("test")
pages = [(p.text, expected_language(p.text)) for p in read_pages("data/cc/pages.jsonl")]
pages = [(text, want) for text, want in pages if want]

print(" n  alpha  測試句  整頁  逐行")
for n in [1, 2, 3, 4]:
    for alpha in [0.01, 0.1, 1.0]:
        model = NGramClassifier.train(train, n, alpha)
        sentences = sum(model.predict(t)[0] == lang for lang, ts in test.items() for t in ts)
        whole = sum(model.predict(text)[0] == want for text, want in pages)
        lines = sum(model.predict_lines(text) == want for text, want in pages)
        print(f"{n:2d}  {alpha:5}  {sentences:3d}/60  {whole:3d}  {lines:3d}  (共 {len(pages)} 頁)")
```

M4 Max 上的輸出（每次相同）：

```text
 n  alpha  測試句  整頁  逐行
 1   0.01   59/60  280  311  (共 318 頁)
 1    0.1   59/60  272  310  (共 318 頁)
 1    1.0   56/60  245  311  (共 318 頁)
 2   0.01   59/60  257  309  (共 318 頁)
 2    0.1   59/60  250  310  (共 318 頁)
 2    1.0   59/60  223  306  (共 318 頁)
 3   0.01   60/60  248  309  (共 318 頁)
 3    0.1   60/60  245  309  (共 318 頁)
 3    1.0   60/60  213  304  (共 318 頁)
 4   0.01   60/60  246  308  (共 318 頁)
 4    0.1   60/60  240  309  (共 318 頁)
 4    1.0   60/60  207  303  (共 318 頁)
```

### 第 4 集 PII 遮蔽 — 狀態：可用

**題目**

Dolma 的規則是：一頁遮蔽了 6 個以上的 PII，整頁丟掉。在 817 頁網頁上，用 `mask_pii` 的次數，量門檻訂在「1、3、6、10 個以上」時，各丟幾頁、少了百分之幾的字元、帶走百分之幾的 PII。為什麼丟掉的頁數很少，字元卻少得比頁數的比例多很多？

**練習目標**：把「遮蔽」與「整頁丟掉」兩種處理接在一起，並量一個門檻的代價。

**批改重點**：1、3、6、10 個以上分別丟 190、51、14、4 頁（23.3%、6.2%、1.7%、0.5%），少 35.4%、17.2%、11.4%、8.4% 的字元，帶走 100%、68%、45%、34% 的 PII。丟掉的頁面比較長：門檻 6 個時，丟掉 1.7% 的頁面卻少了 11.4% 的字元，這 14 頁平均是全部頁面平均長度的約 6.7 倍（多半是長長的目錄或名單），所以字元的損失比例大於頁數的比例。提示：`mask_pii(text)` 回傳的字典，`sum(counts.values())` 是這一頁遮蔽了幾個。常見錯誤：用遮蔽後的文字長度算字元，結果和原文不一樣（標記比原文短）。

**參考答案**

```python,norun
from llmz.pipeline import mask_pii, read_pages

pages = list(read_pages("data/cc/pages.jsonl"))
results = [mask_pii(p.text) for p in pages]
n_pii = [sum(counts.values()) for _, counts in results]

for limit in [1, 3, 6, 10]:
    dropped = [i for i, n in enumerate(n_pii) if n >= limit]
    chars_lost = sum(len(pages[i].text) for i in dropped)
    pii_gone = sum(n_pii[i] for i in dropped)
    print(
        f"{limit:2d} 個以上整頁丟：丟 {len(dropped):3d} 頁（{len(dropped) / len(pages):.1%}），"
        f"少 {chars_lost / sum(len(p.text) for p in pages):.1%} 的字元，帶走 {pii_gone / sum(n_pii):.0%} 的 PII"
    )
```

### 第 5 集 有害內容分類 — 狀態：可用

**題目**

第 5 集的分類器是整句分類。Dolma 是逐句：把文件切成句子，只丟掉分數高於門檻的句子。寫 `drop_flagged(text, threshold)`，回傳 `(留下的文字, 丟掉幾句)`。用第 5 集的玩具分類器（可以縮小訓練資料，各 6 句）與一份 4 句的文件測試：三句正常、一句罵人（用 `blorf`）夾在中間。逐句丟與整頁丟，各剩幾個字元？

**練習目標**：把句子層級的處理接在分類器後面，並比較兩種粒度保留多少資料。

**批改重點**：文件的 4 句分數是 `-0.6`、`-1.0`、`1.32`、`-1.24`，門檻 `0.1` 丟掉第三句，留下 95 個字元；整頁丟則 0 個（文件原本 135 個）。用 `re.split(r"(?<=[.!?])\s+", text)` 切句；用 `" ".join` 接回去。提示：分數是 `score(text) = (ll["flagged"] - ll["ok"]) / 片段數`，不用 `predict` 的機率。常見錯誤：句子切在縮寫的句點（`Mr.`）而切錯，這個簡單的切法的限制，要先說出來。

**參考答案**

```python
import re

from llmz.pipeline import NGramClassifier, char_ngrams

FLAGGED = [
    "you are such a zorp, nobody wants you on this team",
    "what a blorf, you lost us the game again",
    "shut up you big snurgle and go away",
    "stupid zorp, you ruin every match",
    "go away you filthy snurgle, you are the worst",
    "what a pathetic blorf you are",
]
OK = [
    "good game everyone, that was a close match",
    "nice shot, you saved us in the last round",
    "can someone help me with the second level",
    "thanks for the invite, this team is fun",
    "the new map looks great, i love the colors",
    "well played, see you all tomorrow",
]
model = NGramClassifier.train({"flagged": FLAGGED, "ok": OK})


def score(text):
    ll = model.log_likelihoods(text)
    return (ll["flagged"] - ll["ok"]) / sum(char_ngrams(text, model.n).values())


def drop_flagged(text, threshold=0.1):
    """逐句分類：丟掉分數高於門檻的句子，回傳 (留下的文字, 丟掉幾句)。"""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [s for s in sentences if score(s) <= threshold]
    return " ".join(kept), len(sentences) - len(kept)


doc = (
    "Good game everyone. Thanks for the invite, this team is fun. "
    "What a pathetic blorf you are, go away. Well played, see you all tomorrow."
)
kept, dropped = drop_flagged(doc)
print(f"丟掉 {dropped} 句，留下：{kept}")
print(f"逐句丟：剩 {len(kept)} 個字元；整頁丟：剩 0 個字元（原本 {len(doc)} 個）")
```

### 第 6 集 Gopher 品質規則 — 狀態：可用

**題目**

Gopher 的七條規則只看文件的「形狀」。寫一個函式 `make_spam(n_words, seed)`：用一個自己編的詞表（約 30 個英文詞，其中至少要有兩個 Gopher 的停用詞），隨機抽詞組成一份 `n_words` 個詞的文件，每 10 個詞一行，讓 `gopher_check` 判定通過。接著把同一行約 16 個詞的文字重複 20 次，組成另一份文件，它也通過嗎？用規則的內容解釋這兩份文件為什麼能通過。

**練習目標**：理解品質規則是「形狀」的篩子：它擋得掉選單、標籤堆疊、數字表，但判斷不了內容通不通順，也看不出重複。

**批改重點**：兩份文件都通過（`(True, [])`）。隨機抽詞也會抽到 `the`、`and`，所以停用詞的條件過得了；詞數、平均詞長、符號比例、字母比例、行首行尾的條件，都只數「有多少」，不看詞排得通不通順。重複的行讓每一個比例都和原來的那一行一樣，沒有一條規則看得出「重複」。Gopher 的論文把重複另外放在「重複移除」這個階段，本章第 8、9 集的 exact dedup 與 MinHash 就是處理這件事。常見錯誤：詞表裡沒放停用詞，結果卡在 `stop_words`。

**參考答案**

```python
import random

from llmz.pipeline import gopher_check

WORDS = (
    "the of and garden river cheap window yellow market quickly paper silver engine planet "
    "orange bridge hollow ladder pencil forest candle wander mirror basket thunder velvet harbor copper meadow lantern whisper"
).split()


def make_spam(n_words, seed):
    rng = random.Random(seed)
    words = [rng.choice(WORDS) for _ in range(n_words)]
    return "\n".join(" ".join(words[i : i + 10]) for i in range(0, n_words, 10))  # 每 10 個詞一行


print(gopher_check(make_spam(100, 0)))

line = "buy cheap silver watches and the best bridge deals with free delivery to your door today"
repeated = "\n".join([line] * 20)
print(len(repeated.split()), gopher_check(repeated))
```

### 第 7 集 品質分類器 — 狀態：可用

**題目**

GPT-3 的論文用 `np.random.pareto(α) > 1 − document_score` 決定一份文件留不留，α = 9。假設 `document_score` 就是分類器給的機率 \\( s \\)。用 `np.random.default_rng(0).pareto(9, size=1_000_000)` 模擬，量出 \\( s = 0.99、0.9、0.5、0.1、0 \\) 的文件各自被留下的比例。numpy 的 `pareto` 是 Lomax 分布，\\( P(X > x) = (1 + x)^{-9} \\)，用它算出每個 \\( s \\) 的精確機率，和模擬比對。分數 0.9 的文件被留下的機率是多少？這和「分數大於 0.5 才留」的門檻有什麼不同？

**練習目標**：把分類器的分數變成留下的機率，看到 soft sampling 與硬門檻的差別，以及 α 怎麼調整「多嚴格」。

**批改重點**：模擬與精確值吻合，約 0.914、0.424、0.026、0.0031、0.0020（精確值是 \\( (2 - s)^{-9} \\)）。分數 0.9 的文件只有約 42% 被留下，分數 0 的文件也有 0.2% 的機會；硬門檻則是 0.5 以上全留、以下全丟。這樣的抽樣讓資料大部分來自高分文件，同時還收進一些不像正例的文件，不會整個縮到正例的樣子（論文說 α 取 9 是為了這個目的）。提示：留下的條件是 `x > 1 - s`，所以要算的是 \\( P(X > 1 - s) \\)。

**參考答案**

```python
import numpy as np

rng = np.random.default_rng(0)
x = rng.pareto(9, size=1_000_000)  # numpy 的 pareto 是 Lomax（Pareto II）分布：P(X > x) = (1 + x) ** -9
for s in [0.99, 0.9, 0.5, 0.1, 0.0]:
    simulated = (x > 1 - s).mean()
    exact = (1 + (1 - s)) ** -9  # P(X > 1 - s)
    print(f"分數 {s:.2f}：模擬留下 {simulated:.4f}，精確值 {exact:.4f}")
```

### 第 8 集 exact dedup — 狀態：可用

**題目**

`dedup_lines` 把出現在至少 `min_pages` 頁的行，每一份都拿掉。改成「每一行只保留在它第一次出現的那一頁，後面的頁面都拿掉」（同一頁自己重複的行不動），寫出 `dedup_lines_keep_first(pages)`。先執行 `uv run python tools/prepare_webtext.py` 取得 `data/cc/pages.jsonl`，比較它與 `dedup_lines(pages, 2)` 拿掉的行數、字元數與整頁丟掉的頁數，並看那 6 份一模一樣的頁面各留下幾份。

**練習目標**：「出現在很多頁就整個拿掉」與「重複的只留一份」是兩種不同的去重：前者適合每頁共用的框架，後者保留內容。

**批改重點**：`dedup_lines(pages, 2)` 拿掉 23,040 行（21.3%）、216,680 個字元（4.9%），整頁丟掉 10 頁，6 份一模一樣的頁面一份都不剩；`dedup_lines_keep_first` 拿掉 16,189 行（15.0%）、133,798 個字元（3.0%），整頁丟掉 8 頁，6 份一模一樣的頁面留下 1 份（第一份）。提示：邊掃邊記 `seen`（出現過的行的集合）。常見錯誤：每讀一行就更新 `seen`，結果同一頁自己重複的行被當成「前面出現過」而拿掉；要處理完這一頁才把這一頁的行加進 `seen`。

**參考答案**

```python,norun
from collections import Counter

from llmz.pipeline import Page, dedup_lines, read_pages


def dedup_lines_keep_first(pages):
    """每一行只留在它第一次出現的那一頁；同一頁自己重複的行不動。"""
    seen, kept = set(), []
    for page in pages:
        lines = page.text.split("\n")
        fresh = [line for line in lines if line and line not in seen]  # 前面的頁面都沒出現過的行
        seen.update(lines)  # 這一頁處理完才更新
        if fresh:
            kept.append(Page(page.url, "\n".join(fresh)))
    return kept


pages = list(read_pages("data/cc/pages.jsonl"))
n_lines = sum(len(p.text.split("\n")) for p in pages)
n_chars = sum(len(p.text) for p in pages)
for name, kept in [("dedup_lines(min_pages=2)", dedup_lines(pages, 2)), ("dedup_lines_keep_first", dedup_lines_keep_first(pages))]:
    kept_lines = sum(len(p.text.split("\n")) for p in kept)
    kept_chars = sum(len(p.text) for p in kept)
    print(
        f"{name}：拿掉 {n_lines - kept_lines:,} 行（{1 - kept_lines / n_lines:.1%}）、{n_chars - kept_chars:,} 個字元"
        f"（{1 - kept_chars / n_chars:.1%}），整頁丟掉 {len(pages) - len(kept)} 頁"
    )

same_text = Counter(p.text for p in pages)
copies = {p.url for p in pages if same_text[p.text] == 6}  # 那 6 份一模一樣的頁面
for name, kept in [("min_pages=2", dedup_lines(pages, 2)), ("keep_first", dedup_lines_keep_first(pages))]:
    print(f"{name}：6 份一模一樣的頁面留下 {sum(p.url in copies for p in kept)} 份")
```

### 第 9 集 MinHash + LSH — 狀態：可用

**題目**

簽章有 128 格時，\\( b \times r = 128 \\) 的組合只有 8 種。用 `lsh_probability` 印出每一種組合在 \\( J = 0.8 \\) 與 \\( J = 0.5 \\) 的候選機率：有沒有一種同時讓 \\( J = 0.8 \\) 的機率 >= 0.99、\\( J = 0.5 \\) 的機率 <= 0.1？如果簽章可以更長（\\( b \times r \le 512 \\)，\\( b, r \\) 不必是 128 的因數），符合條件的最短簽章是幾格？代價是什麼？

**練習目標**：S 曲線的位置由 \\( b, r \\) 的比例決定，陡不陡由 \\( b \times r \\) 的大小決定；要又陡又在對的位置，只能加長簽章。

**批改重點**：128 格裡沒有任何一組符合：\\( b=16, r=8 \\) 是 0.947 與 0.061，\\( J = 0.8 \\) 差一點；\\( b=32, r=4 \\) 是 1.000 與 0.873，\\( J = 0.5 \\) 太高。\\( b \times r \le 512 \\) 裡最短的是 \\( b=26, r=8 \\)，208 格（0.992 與 0.097）。代價：簽章是 128 格的 1.6 倍長（記憶體 1.6 倍：每份文件 1.6 KiB），`minhash_signature` 的時間也大約是 1.6 倍。提示：兩層迴圈枚舉 \\( b, r \\)，條件寫成函式。

**參考答案**

```python
from llmz.pipeline import lsh_probability


def ok(bands, rows):
    return lsh_probability(0.8, bands, rows) >= 0.99 and lsh_probability(0.5, bands, rows) <= 0.1


for bands in [1, 2, 4, 8, 16, 32, 64, 128]:
    rows = 128 // bands
    p80, p50 = lsh_probability(0.8, bands, rows), lsh_probability(0.5, bands, rows)
    print(f"128 格：b={bands:3d} r={rows:3d}：P(0.8)={p80:.3f}  P(0.5)={p50:.3f}  {'符合' if ok(bands, rows) else ''}")

size, bands, rows = min((b * r, b, r) for b in range(1, 513) for r in range(1, 513) if b * r <= 512 and ok(b, r))
print(f"最短的簽章：b={bands}, r={rows}，{size} 格；P(0.8)={lsh_probability(0.8, bands, rows):.3f}，P(0.5)={lsh_probability(0.5, bands, rows):.3f}")
```

### 第 10 集 組成資料管線 — 狀態：可用

**題目**

第 10 集說：會改文字的步驟（`dedup_lines`）要放在判斷文字的步驟（`keep_gopher`）前面。造一組 5 頁的網頁，每頁有同樣一行選單加上自己的內容，讓 `dedup_lines` 與 `keep_gopher` 兩種順序留下的頁數不同。用 `run_pipeline`（`size` 傳「詞數」）印出每一步之後的頁數與詞數，並且把最後的頁面再用 `gopher_check` 檢查一次。語言辨識與 `keep_gopher` 換順序，也會不同嗎？為什麼？

**練習目標**：用 `run_pipeline` 看見「改文字的步驟」與「判斷的步驟」不能交換，而只做判斷的步驟可以。

**批改重點**：每頁的內容是 44 個詞、共用的選單 12 個詞，總共 56 個詞，剛好過 Gopher 的 50 個詞的下限；拿掉選單之後只剩 44 個詞。先 `dedup_lines`：5 頁（285 個詞）→ 5 頁（220 個詞）→ 0 頁。先 Gopher：5 → 5（285）→ 5（220），最後的 5 頁用 `gopher_check` 再檢查，全部沒通過（詞數不到 50）。語言辨識與 `keep_gopher` 都只判斷、不改文字，頁面通不通過只看那一頁自己，所以換順序，留下的是同一批頁面。常見錯誤：選單只出現在 4 頁以下（`min_pages=5` 的門檻沒到，選單不會被拿掉）；兩種順序的內容與詞數，要用同一組頁面比，不要各造一組。

**參考答案**

```python
from llmz.pipeline import Page, dedup_lines, gopher_check, keep_gopher, run_pipeline

shared = "the shop is open and we have the best prices for you today"  # 每一頁都有的選單，12 個詞
pages = []
for i in range(5):
    body = " ".join([f"item{i}-{j}" for j in range(42)] + ["the", "and"])  # 每一頁自己的內容，44 個詞
    pages.append(Page(f"http://shop.example/{i}", f"{shared}\n{body}"))

orders = {
    "dedup_lines → Gopher": [("dedup_lines", dedup_lines), ("Gopher", keep_gopher)],
    "Gopher → dedup_lines": [("Gopher", keep_gopher), ("dedup_lines", dedup_lines)],
}
for name, steps in orders.items():
    kept, report = run_pipeline(pages, steps, size=lambda text: len(text.split()))
    print(f"{name}：" + " → ".join(f"{r.docs} 頁（{r.size} 個詞）" for r in report))
    print("  最後的頁面通過 Gopher：", [gopher_check(p.text)[0] for p in kept])
```

### 第 11 集 資料混合與合成資料 — 狀態：可用

**題目**

第 11 集談了混合權重，但沒有寫取樣器。寫 `mixed_batch(sources, weights, batch_size, context_length, rng)`：先用 `rng.multinomial` 決定 batch 裡每個來源各幾筆，再對各來源呼叫第 5 章的 `get_batch`，把結果接起來。造兩個假的 token 檔：大的來源 40 萬個 token（內容全是 1），小的來源 1 萬個 token（內容全是 2），這樣看第一個 token 就知道那一筆來自哪裡。小來源的權重分別是 10% 與 5%，訓練 1,000 步（batch 32、context 16），數出來自小來源的樣本有幾筆，以及小來源被看了幾遍。如果小來源最多只看 4 遍，權重最高可以是多少？

**練習目標**：把「權重」變成取樣器，並算出權重、資料量與重複次數的關係：權重越高，小來源重複越多遍。

**批改重點**：權重 10% 時有 3,287 / 32,000 筆來自小來源（10.27%），小來源被看了 5.26 遍；權重 5% 時有 1,539 筆（4.81%），看了 2.46 遍。看的遍數是「來自小來源的筆數 × context 長度 ÷ 小來源的 token 數」。最多看 4 遍的話，權重最高約 7.81%（\\( 4 \times 10000 \div (1000 \times 32 \times 16) \\)）。實際比例與權重不完全一樣：每一筆是獨立抽的，多出來的 0.27% 與少掉的 0.19% 是抽樣的隨機誤差（1,000 步、32,000 筆）。常見錯誤：用 `torch.multinomial` 逐筆抽，慢又難重現（`rng.multinomial` 一次決定各來源幾筆，種子固定，結果就固定）；算遍數時忘了乘 context 長度。

**參考答案**

```python
import numpy as np
import torch

from llmz.data import get_batch

big = np.ones(400_000, dtype=np.uint16)  # 大的來源：內容全是 1，方便看每一筆來自哪裡
small = np.full(10_000, 2, dtype=np.uint16)  # 小的來源：內容全是 2


def mixed_batch(sources, weights, batch_size, context_length, rng):
    """先決定 batch 裡每個來源各幾筆，再對各來源呼叫 get_batch，接起來。"""
    counts = rng.multinomial(batch_size, weights)
    parts = [get_batch(data, n, context_length, torch.device("cpu")) for data, n in zip(sources, counts) if n]
    return torch.cat([x for x, _ in parts]), torch.cat([y for _, y in parts])


rng = np.random.default_rng(0)
steps, batch_size, context_length = 1000, 32, 16
for weight in [0.1, 0.05]:
    from_small = 0
    for _ in range(steps):
        x, y = mixed_batch([big, small], [1 - weight, weight], batch_size, context_length, rng)
        from_small += int((x[:, 0] == 2).sum())
    samples = steps * batch_size
    tokens_seen = from_small * context_length
    print(f"權重 {weight:.0%}：{from_small:,} / {samples:,} 筆來自小來源（{from_small / samples:.2%}），"
          f"小來源被看了 {tokens_seen / len(small):.2f} 遍")

total_tokens = steps * batch_size * context_length
print(f"小來源最多看 4 遍：權重最高 {4 * len(small) / total_tokens:.2%}")
```

## 第 14 章 後訓練：SFT 與偏好學習

### 第 1 集 base model 與指令模型 — 狀態：可用

**題目**

用 `make_tasks(1000, seed=1, split="valid")`。先寫一個「永遠回答 `three`」的模型，用 `reward` 對五種題型各算正確率。再讓每種題型都回答「它自己的標準答案裡最常見的那一個」（用 `collections.Counter`），算每種題型的正確率與合計。第 3 集的 SFT 模型在同一批題目上是 755/1000，這個固定答案的基準線是多少？`color` 與 `who` 的基準線，和「在題目裡出現的候選之間亂猜」比起來，哪個高？

**練習目標**：評分要和基準線比。一個什麼都不看的固定答案，在有些題型上也會答對。

**批改重點**：永遠回答 `three` 時，加 8.5%、減 13.5%、數數量 26.0%，`color` 與 `who` 是 0%（答案不是數字）。每種題型回答它最常見的答案：`five` 22.3%、`one` 28.5%、`three` 26.0%、`purple` 12.1%、`Tom` 14.0%，合計 20.4%。SFT 的 75.5% 比它高 55 個百分點，主要是加、減、數數量從 20% 上下變成 100%。`color` 的固定答案只有 12.1%，比在兩個候選顏色之間亂猜的 50% 低很多；`who` 的 14.0% 也比三選一的 33% 低。差別在「知道答案一定在題目裡」，這是 SFT 學到的答案範圍（第 3 集）。

**參考答案**

```python
from collections import Counter

from llmz.posttrain import KINDS, make_tasks, reward

valid = make_tasks(1000, seed=1, split="valid")
print("永遠回答 three：")
for kind in KINDS:
    mine = [t for t in valid if t.kind == kind]
    print(f"  {kind:9s}{sum(reward(t, 'three') for t in mine) / len(mine):.1%}")

total = 0
for kind in KINDS:
    mine = [t for t in valid if t.kind == kind]
    best, count = Counter(t.answer for t in mine).most_common(1)[0]
    total += sum(reward(t, best) for t in mine)
    print(f"{kind:9s}最常見的答案 {best}：{count / len(mine):.1%}")
print(f"每種題型都回答它最常見的答案，合計 {total / len(valid):.1%}")
```

### 第 2 集 chat template 與 loss mask — 狀態：可用

**題目**

用第 3 章的 byte-level tokenizer（`Tokenizer(base_vocab([END]), [], [END])`），對 `make_tasks(1000, seed=0)` 的每一題用 `encode_chat` 編碼。算每題平均幾個 token、其中幾個算 loss、比例多少。再算：如果把 `Assistant:` 這個角色標記也算進 loss，比例變成多少？為什麼我們不這樣做？

**練習目標**：回覆只佔整串很小的一部分，mask 決定這一小部分以外的 token 不算 loss；角色標記是固定的、不需要學。

**批改重點**：每題平均 94.2 個 token（一個 byte 一個 token），算 loss 的 6.0 個（空格、答案的字母、`<|endoftext|>`），6.4%。`Assistant:` 是 10 個 byte，加進去變成 17.0%。不這樣做：`Assistant:` 在推論時是我們填的提示（`generation_prompt=True`），模型不用寫；它每一題都一樣，loss 很快變成 0，只是稀釋平均。

**參考答案**

```python
from llmz.posttrain import encode_chat, make_tasks, task_messages
from llmz.tokenizer import Tokenizer, base_vocab

END = "<|endoftext|>"
tokenizer = Tokenizer(base_vocab([END]), [], [END])
header = len(tokenizer.encode("Assistant:"))
total = trained = 0
tasks = make_tasks(1000, seed=0)
for task in tasks:
    ids, mask = encode_chat(tokenizer, task_messages(task, with_answer=True))
    total += len(ids)
    trained += sum(mask)
print(f"每題平均 {total / len(tasks):.1f} 個 token，算 loss 的 {trained / len(tasks):.1f} 個（{trained / total:.1%}）")
print(f"連 Assistant:（{header} 個 token）也算：{(trained + header * len(tasks)) / total:.1%}")
```

### 第 3 集 SFT 玩具版 — 狀態：可用

**題目**

用第 3 集的 smoke 設定（隨機初始化的小模型、byte-level tokenizer、60 步），只把 `n_train` 分別換成 16 與 1000，各訓練一次，印出最後的訓練 loss 與 `val_loss`。哪一個的訓練 loss 比較低？哪一個的 `val_loss` 比較低？為什麼？

**練習目標**：SFT 的資料夠不夠多會影響泛化；題目太少，模型把它們背下來。

**批改重點**：`n_train=16` 的訓練 loss 很低（約 0.03），`val_loss` 卻很高（約 2.8）：60 步 × 16 = 960 條樣本，平均每一題被看了 60 次，模型背下這 16 題的答案，沒學到規則；`n_train=1000` 的訓練 loss 較高（約 0.9），`val_loss` 低得多（約 1.2）。具體數字隨機器與 PyTorch 版本略有不同，但這個大小關係穩定。

**參考答案**

```python
from llmz.model import ModelConfig
from llmz.posttrain import SFTConfig, train_sft
from llmz.tokenizer import Tokenizer, base_vocab

END = "<|endoftext|>"
tokenizer = Tokenizer(base_vocab([END]), [], [END])
for n_train in [16, 1000]:
    cfg = SFTConfig(
        base_path="", out_dir=f"smoke_{n_train}",
        model=ModelConfig(vocab_size=257, context_length=256, d_model=64, n_layers=2, n_heads=4, d_ff=128),
        n_train=n_train, n_valid=20, batch_size=16, max_steps=60, lr_max=3e-3, lr_min=3e-4, warmup_steps=5,
        log_every=60, eval_every=60, device="cpu",
    )
    last = train_sft(cfg, tokenizer)[-1]
    print(f"n_train={n_train:4d}：訓練 loss {last['loss']:.2f}，val_loss {last['val_loss']:.2f}")
```

### 第 4 集 reward model 的想法 — 狀態：可用

**題目**

寫一個「只看回覆、不看題目」的獎勵模型：每個顏色一個分數（`nn.Parameter`，10 個數字，從 0 開始），用 `make_preferences(2000, seed=0, split="train", kinds=("color",))` 與 `bradley_terry_loss` 訓練 300 步（`AdamW`，`lr=0.05`）。算分數較高的是 `chosen` 的比例，訓練的 2000 對與 `make_preferences(1000, seed=1, split="valid", kinds=("color",))` 的 1000 對各算一次。這個比例會比 50% 高多少？為什麼？

**練習目標**：獎勵模型要看到提示。只看回覆的分數，在 `chosen` 與 `rejected` 的分布一樣的資料上，不可能比亂猜好。

**批改重點**：訓練集 53.6%，驗證集 50.7%，最後的 loss 是 0.6891，幾乎就是 \\( \log 2 = 0.6931 \\)。分數範圍只有 \\( -0.25 \\) 到 \\( 0.24 \\)。`color` 題的 `chosen` 是題目裡被問的那樣東西的顏色，`rejected` 是另一個，兩邊的顏色分布一樣（每個顏色當 `chosen` 與當 `rejected` 的機會均等），所以某個顏色「比較好」沒有道理；訓練集上的 3.6 個百分點是 300 步 `AdamW` 對 2000 對資料的雜訊擬合，驗證集上就沒了。要分辨哪個顏色對，得知道題目問的是哪一樣東西。

**參考答案**

```python
import torch
from torch import nn

from llmz.optim import AdamW
from llmz.posttrain import COLORS, bradley_terry_loss, make_preferences

train = make_preferences(2000, seed=0, split="train", kinds=("color",))
valid = make_preferences(1000, seed=1, split="valid", kinds=("color",))
index = {c: i for i, c in enumerate(COLORS)}


def ids(pairs):
    return torch.tensor([index[p.chosen] for p in pairs]), torch.tensor([index[p.rejected] for p in pairs])


scores = nn.Parameter(torch.zeros(len(COLORS)))  # 獎勵模型：每個顏色一個分數，不看題目
optimizer = AdamW([scores], lr=0.05, weight_decay=0.0)
train_chosen, train_rejected = ids(train)
for step in range(300):
    loss = bradley_terry_loss(scores[train_chosen], scores[train_rejected])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

for name, pairs in [("train", train), ("valid", valid)]:
    chosen, rejected = ids(pairs)
    rate = (scores[chosen] > scores[rejected]).float().mean().item()
    print(f"{name}：{len(pairs)} 對，分數較高的是 chosen 的比例 {rate:.1%}")
print("loss", round(loss.item(), 4), "（log 2 = 0.6931）")
print("分數範圍", round(scores.min().item(), 2), "到", round(scores.max().item(), 2))
```

### 第 5 集 DPO 推導與玩具版實作 — 狀態：可用

**題目**

policy 比 reference 多給 `chosen` 的 log 機率 2、少給 `rejected` 2（reference 兩個都是 \\( -3 \\)，policy 是 \\( -1 \\) 與 \\( -5 \\)）。對 \\( \beta \in \{0.01, 0.1, 1, 5\} \\)，用 `dpo_loss` 各算 loss，並用 `backward()` 算對 `policy_chosen` 的梯度。loss 隨 \\( \beta \\) 怎麼變？梯度呢？為什麼？

**練習目標**：\\( \beta \\) 同時放大隱含獎勵與壓縮 sigmoid 的有效範圍；loss 與梯度的大小不是單調的。

**批改重點**：loss 是 0.6733、0.5130、0.0181、0.0000：\\( \beta \\) 越大越小，因為隱含獎勵的差是 \\( \beta \times 4 \\)，sigmoid 越快飽和。梯度是 \\( -0.0049 \\)、\\( -0.0401 \\)、\\( -0.0180 \\)、\\( -0.0000 \\)：先變大再變小。\\( \beta \\) 小時，loss 幾乎是 \\( \log 2 \\)，梯度大約是 \\( -\beta \sigma(-4\beta) \approx -\beta/2 \\)，隨 \\( \beta \\) 增加；\\( \beta \\) 大時 sigmoid 已經飽和，這一對「已經做對了」，梯度趨近 0，policy 不再被往外推，停在離 reference 不遠的地方。這就是 \\( \beta \\) 控制「policy 可以離 reference 多遠」的方式。

**參考答案**

```python
import torch

from llmz.posttrain import dpo_loss

ref_chosen, ref_rejected = torch.tensor([-3.0]), torch.tensor([-3.0])
for beta in [0.01, 0.1, 1.0, 5.0]:
    policy_chosen = torch.tensor([-1.0], requires_grad=True)  # chosen 比 reference 高 2
    policy_rejected = torch.tensor([-5.0], requires_grad=True)  # rejected 比 reference 低 2
    loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
    loss.backward()
    print(f"beta {beta:5.2f}：loss {loss.item():.4f}　對 chosen 的梯度 {policy_chosen.grad.item():+.4f}")
```

## 第 15 章 後訓練：RL 與可驗證獎勵

### 第 1 集 policy gradient 直覺 — 狀態：可用

**題目**

吃角子老虎機的平均獎勵是 `[0, 3, 2, 5, 4]`，policy 的 logits 是 `[1.0, 0.0, 0.0, -1.0, 0.5]`。不要抽樣：對 5 個動作各算 \\( \pi(a)\, R(a)\, (\text{one\_hot}(a) - \pi) \\)，加起來，和對 `expected_reward` 做 `backward()` 得到的梯度比較。

**練習目標**：\\( \nabla J = \mathbb{E}[R \, \nabla \log \pi(a)] \\) 的期望值，真的就是把每個動作「機率乘上它的推力」加起來；抽樣只是用平均去逼近這個加總。

**批改重點**：兩個向量相同，約 `[-0.805, 0.149, 0.001, 0.164, 0.491]`。要能說出：softmax 的 \\( \nabla \log \pi(a) \\) 是 `one_hot(a) - π`；抽樣估計的平均會趨近這個加總。

**參考答案**

```python
import torch

from llmz.functional import softmax
from llmz.rl import expected_reward

means = torch.tensor([0.0, 3.0, 2.0, 5.0, 4.0])
logits = torch.tensor([1.0, 0.0, 0.0, -1.0, 0.5], requires_grad=True)

# 對照：直接對 expected_reward 做 backward
expected_reward(logits, means).backward()

# 不抽樣：把每個動作的 pi(a) * R(a) * (one_hot(a) - pi) 加起來
with torch.no_grad():
    probs = softmax(logits)
    by_hand = sum(probs[a] * means[a] * ((torch.arange(5) == a).float() - probs) for a in range(5))

print("手算的期望   ", [round(x, 3) for x in by_hand.tolist()])
print("autograd 梯度", [round(x, 3) for x in logits.grad.tolist()])
print("相同：", torch.allclose(by_hand, logits.grad, atol=1e-6))
```

### 第 2 集 REINFORCE 玩具版 — 狀態：可用

**題目**

`estimate_gradient` 的批次平均 baseline 把被抽到的樣本也算進平均，所以估計的平均會小 \\( (n-1)/n \\) 倍。改用 leave-one-out baseline：每個樣本的 baseline 是「其他 \\( n-1 \\) 個樣本獎勵的平均」。用 `sample_actions`、`bandit_rewards`、`reinforce_loss` 寫出 `loo_gradient`，取 \\( n=2 \\)、`noise_std=1.0`、policy 全 0，各重複 4000 次，比較「沒有 baseline」「批次平均」「leave-one-out」三種估計的平均（是精確梯度的幾倍）與變異加總。

**練習目標**：baseline 只要和被抽到的動作無關就不影響期望值；批次平均違反了這一點，只是影響很小（\\( n \\) 大時）。

**批改重點**：沒有 baseline 與 leave-one-out 的平均約是精確梯度的 1.0 倍，批次平均約 0.5 倍（\\( n=2 \\) 時 \\( (n-1)/n = 1/2 \\)）；變異加總約 4.5、0.8、3.2：leave-one-out 無偏，變異比沒有 baseline 小，但比批次平均大。數字隨隨機數種子略有變動（4000 次重複）。

**參考答案**

```python
import torch

from llmz.rl import bandit_rewards, estimate_gradient, expected_reward, reinforce_loss, sample_actions


def loo_gradient(logits, means, n, noise_std, generator):
    """leave-one-out baseline：每個樣本的 baseline 是「其他 n-1 個樣本獎勵的平均」。"""
    logits = logits.detach().requires_grad_()
    actions = sample_actions(logits, n, generator)
    rewards = bandit_rewards(means, actions, noise_std, generator)
    baselines = (rewards.sum() - rewards) / (n - 1)
    reinforce_loss(logits.expand(n, -1), actions, rewards - baselines).backward()
    return -logits.grad


means = torch.tensor([0.0, 3.0, 2.0, 5.0, 4.0])
logits = torch.zeros(5)
exact = torch.zeros(5, requires_grad=True)
expected_reward(exact, means).backward()
exact = exact.grad

n, repeats = 2, 4000
generator = torch.Generator().manual_seed(0)
runs = {
    "沒有 baseline": torch.stack([estimate_gradient(logits, means, n, False, 1.0, generator) for _ in range(repeats)]),
    "批次平均": torch.stack([estimate_gradient(logits, means, n, True, 1.0, generator) for _ in range(repeats)]),
    "leave-one-out": torch.stack([loo_gradient(logits, means, n, 1.0, generator) for _ in range(repeats)]),
}
for name, g in runs.items():
    scale = (g.mean(dim=0) @ exact / (exact @ exact)).item()  # 平均的方向上，是精確梯度的幾倍
    print(f"{name:14s} 平均是精確梯度的 {scale:.2f} 倍，變異加總 {g.var(dim=0).sum().item():.2f}")
```

### 第 3 集 GRPO 的 group-relative advantage — 狀態：可用

**題目**

一組 8 個回答、答對 \\( k \\) 個（\\( k = 1, \dots, 7 \\)），獎勵是 1 或 0。對每個 \\( k \\)，用 `group_advantages` 印出答對與答錯的回答的 advantage，不除與除以標準差兩種都印。\\( k = 4 \\) 時，答對與答錯的 advantage 大小相等嗎？\\( k = 1 \\) 時答對的 advantage 是 \\( k = 4 \\) 時的幾倍（兩種算法分別算）？

**練習目標**：group-relative advantage 讓稀有的成功拿到大的推力、常見的成功只拿到小的推力；除以標準差又把這個差距再拉開。

**批改重點**：\\( k = 4 \\) 時兩者大小相等（不除：±0.5；除：±0.935）；\\( k = 1 \\) 時答對的是 +0.875（不除）與 +2.475（除），分別是 \\( k = 4 \\) 時的 1.75 倍與 2.65 倍。答對與答錯的 advantage 加權（各自乘上人數）後總和為 0。

**參考答案**

```python
import torch

from llmz.rl import group_advantages

for k in range(1, 8):  # 8 個回答裡答對 k 個
    rewards = torch.tensor([[1.0] * k + [0.0] * (8 - k)])
    plain, normed = group_advantages(rewards, normalize_std=False)[0], group_advantages(rewards, normalize_std=True)[0]
    print(f"答對 {k} 個：答對的 advantage {plain[0]:+.3f}（不除）{normed[0]:+.3f}（除）；答錯的 {plain[-1]:+.3f}（不除）{normed[-1]:+.3f}（除）")
```

### 第 4 集 GRPO loss 實作與測試 — 狀態：可用

**題目**

對 `clip_eps` 取 0.1、0.2、0.5，把比值從 0.35 到 1.95（間隔 0.1，每個值的小數是 5，避開剛好等於 \\( 1 \pm \varepsilon \\) 的邊界）掃一遍。對 advantage 是 +1 與 -1 兩種情況，用 `grpo_loss`（一個 token、`old_logp = 0`）數出「梯度剛好是 0」的比值有哪些，範圍落在哪裡。再回答：把 `clip_eps` 設成很大（例如 100），結果會變成什麼？

**練習目標**：看懂 clip 是單側的：advantage 為正，只在比值高過 \\( 1+\varepsilon \\) 時關掉梯度；advantage 為負，只在低過 \\( 1-\varepsilon \\) 時關掉。

**批改重點**：\\( A=+1 \\) 時，梯度為 0 的是比值高於 \\( 1+\varepsilon \\) 的那些（\\( \varepsilon \\) 為 0.1、0.2、0.5 時各 9、8、5 個，範圍 1.15～1.95、1.25～1.95、1.55～1.95）；\\( A=-1 \\) 時是低於 \\( 1-\varepsilon \\) 的那些（6、5、2 個，範圍 0.35～0.85、0.35～0.75、0.35～0.45）。同一個比值在兩種 advantage 下不會同時梯度為 0。\\( \varepsilon \\) 很大時 clip 永遠碰不到，loss 變成沒有 clip 的 \\( -\rho A \\) 的平均，也就是只靠 \\( \rho \\) 修正的 policy gradient，一批 rollout 更新很多次就可能把模型推得很遠。

**參考答案**

```python
import math

import torch

from llmz.rl import grpo_loss


def grad_at(ratio, advantage, eps):
    logp = torch.tensor([[math.log(ratio)]], requires_grad=True)
    grpo_loss(logp, torch.zeros(1, 1), torch.tensor([advantage]), torch.ones(1, 1, dtype=torch.bool), clip_eps=eps).backward()
    return logp.grad.item()


ratios = [0.35 + 0.1 * k for k in range(17)]  # 0.35、0.45、...、1.95：避開 1 ± eps 剛好的邊界
for eps in [0.1, 0.2, 0.5]:
    for advantage in [+1.0, -1.0]:
        silent = [r for r in ratios if grad_at(r, advantage, eps) == 0.0]
        print(f"eps={eps} A={advantage:+.0f}：{len(silent)} 個比值梯度為 0，範圍 {min(silent):.2f}～{max(silent):.2f}")
```

### 第 5 集 RLVR：用可驗證獎勵訓練 SFT 模型 — 狀態：可用

**題目**

用 `rollout` 對第 14 章的 SFT 模型（`runs/sft/model.pt`）取樣：`color` 題與 `who` 題各 64 題、每題 8 個回答（`temperature=1.0`，`torch.manual_seed(0)`）。各自數出：平均獎勵、整組全對的題數、整組全錯的題數、有對有錯的題數。只有「有對有錯」的組 advantage 不是 0；如果每一題答對的機率都是亂猜的 1/2（`color`）與 1/3（`who`），有對有錯的組應該佔幾成？實際數字和它一樣嗎？

**練習目標**：RLVR 的訓練訊號只來自「組內有對有錯」的題目；整組全錯（或全對）的題目這一步等於白跑。

**批改重點**：若每題的答對機率都一樣是 \\( p \\)，一組 8 個全對或全錯的機率是 \\( p^8 + (1-p)^8 \\)：`color`（\\( p=1/2 \\)）約 0.8%、`who`（\\( p=1/3 \\)）約 3.9%，所以有對有錯的組應該有 99% 與 96% 左右。實際一次的結果（CPU，種子 0）：`color` 平均獎勵 0.47，全對 0、全錯 2、有對有錯 62（97%）；`who` 平均獎勵 0.33，全對 0、全錯 6、有對有錯 58（91%）。`who` 的全錯組比預期多：模型對每一題不是都剛好 1/3，有些題它偏向一個錯的名字，答對機率比 1/3 低很多。數字隨種子與機器略有不同。

**參考答案**

```python,norun
import torch

from llmz.model import ModelConfig, TransformerLM
from llmz.posttrain import make_tasks
from llmz.rl import rollout
from llmz.tokenizer import Tokenizer

tokenizer = Tokenizer.load("data/tokenizer.json")
model = TransformerLM(ModelConfig())
model.load_state_dict(torch.load("runs/sft/model.pt", map_location="cpu"))

for kind in ["color", "who"]:
    torch.manual_seed(0)
    tasks = make_tasks(64, seed=0, kinds=(kind,))
    rewards = rollout(model, tokenizer, tasks, group_size=8, max_new_tokens=4).rewards  # (64, 8)
    all_right = int((rewards.min(dim=1).values == 1).sum())
    all_wrong = int((rewards.max(dim=1).values == 0).sum())
    mixed = 64 - all_right - all_wrong  # 只有這些組的 advantage 不是 0
    print(f"{kind:5s} 平均獎勵 {rewards.mean():.2f}  整組全對 {all_right}  整組全錯 {all_wrong}  有對有錯 {mixed}（{mixed / 64:.0%}）")
```
