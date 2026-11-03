# Linear（自己寫）

## 本集目標

寫出全書用最多次的零件 `Linear`：一個矩陣乘法，附上合理的初始化。

## 正文

### 就是 `x @ W.T`

```python,ignore
{{#include ../../../src/llmz/layers.py:linear}}
```

第 2 章第 3 集的 `nn.Module` 寫法：參數是一個 `(out_features, in_features)` 的矩陣，`forward` 做 `x @ W.T`。輸入最後一個維度是 `in_features`，前面不管有幾個維度都可以（`(B, T, in)` 進、`(B, T, out)` 出），因為矩陣乘法只看最後兩個維度，第 1 章第 7 集教過。

權重存成 `(out, in)` 而不是 `(in, out)`，是 PyTorch 的慣例，`state_dict` 才和別人的模型對得上。本書的 Linear 沒有偏差項（bias）：現代 LLM 幾乎都拿掉了，效果一樣、參數少一點。

```python
import torch

from llmz.layers import Linear
from llmz.utils import count_parameters

torch.manual_seed(0)
layer = Linear(4, 3)
x = torch.randn(2, 5, 4)
y = layer(x)
print(layer.weight.shape, y.shape)
print(torch.allclose(y, x @ layer.weight.T))
print(count_parameters(layer))
```

執行結果：

```text
torch.Size([3, 4]) torch.Size([2, 5, 3])
True
12
```

`torch.allclose` 檢查兩個張量每個元素是否「夠接近」（浮點數不要用 `==`）。

### 初始化為什麼重要

參數一開始是隨機的，但隨機的**大小**很重要。如果 W 的元素太大，`x @ W.T` 每經過一層就放大一次，幾十層之後數值爆掉；太小則一路縮到 0，梯度也跟著消失。目標是讓輸出和輸入大致**同樣大小**。

一個輸出元素是 `in_features` 個乘積的和，每項的大小約是 W 元素的標準差，所以和的標準差約是 \\( \sigma \sqrt{\text{in}} \\)。要讓它接近 1，\\( \sigma \\) 得是 \\( 1/\sqrt{\text{in}} \\) 的量級；反向傳播時同樣的道理對 `out` 成立。兩邊折衷：

\\[ \sigma = \sqrt{\frac{2}{\text{in} + \text{out}}} \\]

程式裡 `torch.randn(...) * std` 做出這個標準差，再 `.clamp(-3σ, 3σ)` 把偶爾出現的超大值截掉，免得一個離群值毀掉整層。

```python
import torch

from llmz.layers import Linear

torch.manual_seed(0)
layer = Linear(512, 512)
x = torch.randn(1000, 512)
print(round(layer.weight.std().item(), 4))
print(round(x.std().item(), 2), round(layer(x).std().item(), 2))
```

執行結果：

```text
0.0441
1.0 1.0
```

輸入的標準差是 1，經過一層之後還是 1 左右。疊很多層也不會爆。

## 重點整理

- `Linear(in, out)`：權重 `(out, in)`，`forward` 是 `x @ W.T`，沒有偏差項；前面任意維度都可以。
- 初始化標準差 \\( \sqrt{2/(\text{in}+\text{out})} \\)，截在三個標準差內，讓每層輸出和輸入大小相當。
- `torch.allclose` 比較浮點張量。
