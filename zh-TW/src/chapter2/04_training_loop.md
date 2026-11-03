# 訓練迴圈固定句型

## 本集目標

寫出全書每一次訓練都長一樣的五步：forward → loss → `backward()` → `step()` → `zero_grad()`，並把「更新參數」與「梯度歸零」包成一個小小的 `SGD` 類別。

## 正文

### 先定義 loss

訓練需要一個數字告訴我們「模型現在猜得多差」，這個數字叫 **loss**（損失）。線性回歸最常用的是**均方誤差**（mean squared error，MSE）：預測值減正確值、平方、取平均。猜得越準，loss 越接近 0。

```python,ignore
def mse(pred, target):
    return ((pred - target) ** 2).mean()
```

第 1 章第 15 集的 cross-entropy 是語言模型用的 loss；這一章的 loss 只是換成了連續數值的版本，用法完全一樣：一個純量，可以 `backward()`。

### 把更新與歸零包起來

上一集的四個動作裡，「更新」與「歸零」要對每個參數做一次。把它們包成一個類別，叫做**優化器**（optimizer）：

```python,ignore
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
```

`SGD` 是 stochastic gradient descent（隨機梯度下降）的縮寫，就是第 1 章第 12 集那個「往梯度反方向走一小步」。`zero_grad` 這裡直接把 `.grad` 設成 `None`，下次 `backward()` 會重新建立，效果和 `zero_()` 一樣而且省一點記憶體。第 5 章會把它改寫成繼承 PyTorch 的 `Optimizer` 基底類別，再加上 AdamW；現在這十行就夠用。

### 五步固定句型

把 loss、模型、優化器接起來：

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


x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([3.0, 5.0, 7.0])
model = LinearModel()
opt = SGD(model.parameters(), lr=0.1)
for step in range(3):
    pred = model(x)            # 1. forward
    loss = mse(pred, y)        # 2. loss
    loss.backward()            # 3. backward
    opt.step()                 # 4. step
    opt.zero_grad()            # 5. zero_grad
    print(step, round(loss.item(), 4), round(model.w.item(), 4), round(model.b.item(), 4))
```

執行結果：

```text
0 27.6667 2.2667 1.0
1 0.3319 2.0178 0.8933
2 0.0053 2.0439 0.9076
```

資料是 \\( y = 2x + 1 \\)，三步之後 `w` 已經接近 2，`b` 往 1 走，loss 從 27.7 掉到 0.005。

迴圈裡那五行就是本書的**固定句型**，從現在起每個訓練程式都長這樣：

1. **forward**：`pred = model(x)`，模型算出預測。
2. **loss**：`loss = mse(pred, y)`，一個純量。
3. **backward**：`loss.backward()`，梯度寫進每個參數的 `.grad`。
4. **step**：`opt.step()`，用梯度更新參數。
5. **zero_grad**：`opt.zero_grad()`，清掉梯度，準備下一步。

之後模型會變成 Transformer、loss 變成 cross-entropy、`SGD` 變成 AdamW、`x` 變成一批 token，但這五行不會變。

## 重點整理

- loss 是一個純量，衡量模型猜得多差；線性回歸用 MSE。
- 優化器把「用梯度更新每個參數」與「梯度歸零」包成 `step()` 與 `zero_grad()`。
- 固定句型：forward → loss → `backward()` → `step()` → `zero_grad()`，全書通用。
- `p.grad = None` 與 `p.grad.zero_()` 效果一樣。
