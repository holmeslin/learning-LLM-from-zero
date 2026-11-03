# AdamW

## 本集目標

寫出 AdamW：每個參數各自記住梯度的移動平均與梯度平方的移動平均，用它們調整步伐，再加上 weight decay。這是訓練 LLM 的標準優化器。

## 正文

### SGD 的問題

SGD 對每個參數用同一個學習率。但不同參數的梯度大小差很多：embedding 裡罕見 token 的那一列梯度很小、attention 的某些矩陣梯度很大。一個學習率不可能同時適合所有人。

### Adam：每個參數自己的步伐

對每個參數維護兩個和它同形狀的張量：

- \\( m \\)：梯度的移動平均（動量，方向比單一步的梯度穩）。
- \\( v \\)：梯度**平方**的移動平均（這個參數的梯度通常多大）。

更新用 \\( m / \sqrt{v} \\)：梯度一向很大的參數除以大數、走小步；梯度一向很小的除以小數、走大步。效果是每個參數的步伐都被調到差不多的尺度。

\\[ m \leftarrow \beta_1 m + (1-\beta_1) g, \qquad v \leftarrow \beta_2 v + (1-\beta_2) g^2, \qquad \theta \leftarrow \theta - \eta_t \frac{m}{\sqrt{v} + \epsilon} \\]

\\( \beta_1 = 0.9 \\)、\\( \beta_2 = 0.95 \\) 是記多久的意思（越接近 1 記越久）。\\( \eta_t = \eta \sqrt{1-\beta_2^t} / (1-\beta_1^t) \\) 是修正：\\( m \\)、\\( v \\) 從 0 開始，前幾步會偏小，這個係數把它補回來，t 大了之後就趨近 \\( \eta \\)。

### W：weight decay

AdamW 的 W 是 weight decay：每一步把參數往 0 縮一點點 \\( \theta \leftarrow \theta (1 - \eta \lambda) \\)，\\( \lambda \\) 本書用 0.1。它讓參數不會無限制長大，是一種正則化。「decoupled」的意思是這一步**不**經過 \\( m / \sqrt{v} \\) 的調整，直接作用在參數上，和 Adam 的更新分開。

### 程式

```python,ignore
{{#include ../../../src/llmz/optim.py:adamw}}
```

`self.state[p]` 是基底類別給的、每個參數專屬的字典，第一次用到時是空的，我們放進 `t`、`m`、`v`。`mul_`、`add_` 是就地運算，不會每步配置新張量。順序照公式：先更新 `m`、`v`，weight decay，再做 Adam 的那一步。

```python
import torch
from torch import nn

from llmz.optim import AdamW


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
opt = AdamW(model.parameters(), lr=0.1, weight_decay=0.0)
for step in range(1, 201):
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}  w {model.w.item():.3f}  b {model.b.item():.3f}")
state = opt.state[model.w]
print(state["t"], abs(state["m"].item()) < 1e-3, round(state["v"].item(), 5))
```

執行結果：

```text
step  50  loss 0.0658  w 2.578  b 2.385
step 100  loss 0.0091  w 2.843  b 2.064
step 150  loss 0.0079  w 2.927  b 2.032
step 200  loss 0.0079  w 2.931  b 2.033
200 True 0.00041
```

第 2 章的線性回歸用 AdamW 一樣收斂。最後一行是 `w` 的狀態：步數 200、動量已經接近 0（收斂了，梯度很小）、平方動量 0.00041。這裡 `weight_decay=0`，因為兩個參數的正確答案不是 0，往 0 拉會拉偏；訓練 LLM 時幾千萬個參數裡沒有誰「應該」特別大，decay 才合理。

### 記憶體

每個參數多兩個同形狀的張量，第 2 章第 7 集說的「AdamW 每個參數 16 位元組」就是：參數 4、梯度 4、\\( m \\) 4、\\( v \\) 4。

`tests/test_optim.py` 用 `torch.optim.AdamW` 跑同樣的設定比對參數，一致。

## 重點整理

- Adam：每個參數記住梯度的移動平均 \\( m \\) 與平方的移動平均 \\( v \\)，更新 \\( m/(\sqrt{v}+\epsilon) \\)，讓每個參數的步伐尺度相當；\\( \eta_t \\) 修正起步偏小。
- W：每步把參數乘 \\( 1 - \eta\lambda \\)，與 Adam 的更新分開。
- 本書預設 \\( \beta = (0.9, 0.95) \\)、\\( \lambda = 0.1 \\)。
- 狀態放在 `self.state[p]`；每個參數多 8 位元組記憶體。
