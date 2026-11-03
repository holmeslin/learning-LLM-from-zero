# 線性回歸完整範例

## 本集目標

用固定句型從頭到尾訓練一個線性回歸：自己做資料、看 loss 下降、檢查學到的參數是否接近真實值。

## 正文

### 自己造資料

真實世界的資料我們不知道答案，練習時反過來：先決定答案 \\( y = 3x + 2 \\)，再加一點雜訊，看模型能不能把 3 和 2 找回來。

```python
import torch

torch.manual_seed(0)
x = torch.rand(100)
y = 3 * x + 2 + 0.1 * torch.randn(100)
print(x[:3])
print(y[:3])
```

執行結果：

```text
tensor([0.4963, 0.7682, 0.0885])
tensor([3.2274, 4.1350, 2.2426])
```

100 個 `x` 均勻落在 0 到 1 之間，`y` 是直線加上標準差 0.1 的雜訊（第 1 章第 10 集的 `torch.rand` 與 `torch.randn`）。種子固定，所以你的數字會和書上一樣。

### 完整程式

模型、loss、優化器照上一集，迴圈跑 300 步，每 50 步印一次：

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


torch.manual_seed(0)
x = torch.rand(100)
y = 3 * x + 2 + 0.1 * torch.randn(100)

model = LinearModel()
opt = SGD(model.parameters(), lr=0.5)
for step in range(1, 301):
    pred = model(x)
    loss = mse(pred, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}  w {model.w.item():.3f}  b {model.b.item():.3f}")
```

執行結果：

```text
step  50  loss 0.0081  w 2.889  b 2.055
step 100  loss 0.0079  w 2.930  b 2.034
step 150  loss 0.0079  w 2.931  b 2.033
step 200  loss 0.0079  w 2.931  b 2.033
step 250  loss 0.0079  w 2.931  b 2.033
step 300  loss 0.0079  w 2.931  b 2.033
```

### 讀懂輸出

- `w` 收斂到 2.93、`b` 到 2.03，很接近真實的 3 與 2。差一點是正常的：資料有雜訊，而且只有 100 個點。
- loss 停在 0.0079 不再下降。雜訊標準差是 0.1，平方是 0.01，loss 不可能比雜訊本身小，這就是「訓練好了」的訊號。
- 100 步之後數字幾乎不動，代表學習率 0.5 對這個問題剛好。你可以把 `lr` 改成 0.05 看它慢多少，改成 2.5 看它炸掉。

這支程式每一步都用全部 100 個點算 loss。第 5 章訓練語言模型時資料太多，一步只會隨機抽一小批（batch），這就是 SGD 名字裡「隨機」的由來。

### 想在 GPU 上跑

這個例子小到沒必要，但寫法先記著：`model.to(device)`、`x.to(device)`、`y.to(device)` 三行，其餘不變。MPS 上的浮點結果最後幾位可能和書上不同，不是錯。

## 重點整理

- 練習時先決定真實參數再加雜訊造資料，訓練完就能對答案。
- loss 降到雜訊的大小就不會再降，這是正常的終點，不是 bug。
- 學習率太小收斂慢、太大會發散；本例 0.5 剛好。
- 這支程式的結構（資料、模型、loss、優化器、五步迴圈）就是第 5 章訓練 LLM 的結構。
