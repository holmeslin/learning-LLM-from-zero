# SiLU 與 SwiGLU

## 本集目標

寫出 Transformer block 的第二個子層：前饋網路（FFN）。先認識啟用函式 SiLU，再組成帶 gate 的 SwiGLU。

## 正文

### 為什麼要非線性

把兩個 Linear 接起來 `W2 @ (W1 @ x)` 還是一個矩陣乘法，疊再多層都等於一層。要讓模型能表示複雜的對應，每層之間要夾一個**非線性**的函式，叫**啟用函式**（activation function）。

### SiLU

本書用 SiLU：\\( \text{silu}(x) = x \cdot \sigma(x) \\)，其中 \\( \sigma(x) = 1/(1+e^{-x}) \\) 是 sigmoid，把任何數壓到 0～1 之間。

```python,ignore
{{#include ../../../src/llmz/layers.py:silu}}
```

`torch.sigmoid` 是基本數學函式，和 `torch.exp` 一樣直接用。

```python
import torch

from llmz.layers import silu

x = torch.tensor([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0])
print(silu(x))
```

執行結果：

```text
tensor([-0.0719, -0.2384, -0.2689,  0.0000,  0.7311,  1.7616,  3.9281])
```

正數幾乎原樣通過（4 → 3.93），負數被壓向 0（-4 → -0.07），0 附近是平滑的曲線。比老式的 ReLU（負數直接變 0）多了一點平滑，訓練起來比較順。

### SwiGLU：加一個 gate

最簡單的 FFN 是 `w2(silu(w1(x)))`：先升到比較寬的 `d_ff` 維、過啟用函式、再降回 `d_model`。現在的 LLM 多半再加一條「gate」：另一個 Linear `w3` 的輸出和啟用後的結果逐元素相乘，讓模型可以逐維度決定要讓多少資訊通過。

\\[ \text{SwiGLU}(x) = W_2 \big( \text{silu}(W_1 x) \odot W_3 x \big) \\]

```python,ignore
{{#include ../../../src/llmz/layers.py:swiglu}}
```

```python
import torch

from llmz.layers import SwiGLU
from llmz.utils import count_parameters

torch.manual_seed(0)
ffn = SwiGLU(d_model=512, d_ff=1344)
x = torch.randn(2, 5, 512)
print(ffn(x).shape)
print(count_parameters(ffn), 3 * 512 * 1344)
```

執行結果：

```text
torch.Size([2, 5, 512])
2064384 2064384
```

三個矩陣，參數 \\( 3 \times d_{\text{model}} \times d_{\text{ff}} \\)。`d_ff` 怎麼選：沒有 gate 的 FFN 慣例用 \\( 4 d_{\text{model}} \\)；SwiGLU 多了一個矩陣，為了參數量差不多，用 \\( \frac{8}{3} d_{\text{model}} \\)，再湊到 64 的倍數讓矩陣乘法在 GPU 上對齊：\\( 512 \times 8/3 = 1365 \\)，取 1344。

FFN 對每個位置**各自**做同樣的事，位置之間不交換資訊；交換資訊是 attention 的工作。第 6 章第 8 集會把 gate 拿掉、換回最簡單的 FFN 比較看看。

## 重點整理

- 層與層之間要有非線性的啟用函式，本書用 SiLU：`x * sigmoid(x)`。
- SwiGLU：`w2(silu(w1(x)) * w3(x))`，三個矩陣，`d_ff` 取 \\( 8/3 \\) 倍 `d_model` 再湊到 64 的倍數（512 → 1344）。
- FFN 逐位置獨立運算，不混合位置。
