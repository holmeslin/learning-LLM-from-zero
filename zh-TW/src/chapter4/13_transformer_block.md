# pre-norm Transformer block

## 本集目標

把 RMSNorm、attention、SwiGLU 組成一個 block：兩個子層，各自加殘差連接，RMSNorm 放在子層之前。

## 正文

### 殘差連接

Block 不是 `x → attention → FFN`，而是：

\\[ x \leftarrow x + \text{attention}(\text{RMSNorm}(x)), \qquad x \leftarrow x + \text{FFN}(\text{RMSNorm}(x)) \\]

每個子層算出來的東西**加回**原本的 `x`，叫**殘差連接**（residual connection）。好處：梯度可以沿著「+ x」這條路直接流回前面的層，不用穿過每一層的矩陣，幾十層深的模型才訓練得動；而且每一層只需要學「要改什麼」，不用重新表示整個輸入。

### pre-norm

RMSNorm 放在子層**之前**（pre-norm），子層的輸入永遠是正規化過的；殘差那條路上完全沒有 norm，從第一層到最後一層是乾淨的加法。早期的 Transformer 把 norm 放在相加**之後**（post-norm），訓練起來不穩、需要很小心地調學習率，現在幾乎都改成 pre-norm，第 6 章第 6 集會比較。

### 設定與程式

模型的尺寸很多個數字，先集中在一個 `dataclass`（Python 的資料類別，自動產生 `__init__`）：

```python,ignore
{{#include ../../../src/llmz/model.py:config}}
```

預設值就是本書的標準設定。最後四個是第 6 章消融實驗的開關，這一章保持預設不用管。

```python,ignore
{{#include ../../../src/llmz/model.py:block}}
```

`forward` 的 `pre` 分支就是上面的兩條公式。`make_norm` 依設定給 `RMSNorm` 或什麼都不做的 `Identity`，`ffn` 依設定給 `SwiGLU` 或沒有 gate 的版本；預設下就是 RMSNorm 加 SwiGLU。

```python
import torch

from llmz.attention import RoPE
from llmz.model import ModelConfig, TransformerBlock
from llmz.utils import count_parameters

torch.manual_seed(0)
cfg = ModelConfig(d_model=32, n_heads=4, d_ff=64, context_length=16)
block = TransformerBlock(cfg, RoPE(8, 16))
x = torch.randn(2, 10, 32)
y = block(x)
print(y.shape)
print(round((y - x).abs().mean().item(), 3), round(x.abs().mean().item(), 3))
print(count_parameters(block), 4 * 32 * 32 + 3 * 32 * 64 + 2 * 32)
```

執行結果：

```text
torch.Size([2, 10, 32])
0.621 0.785
10304 10304
```

輸入輸出同形狀。第二行：輸出和輸入的差距大約和輸入本身同量級，剛初始化的 block 已經會「改一點」但不會把輸入沖掉。參數：attention 四個矩陣、SwiGLU 三個矩陣、兩個 RMSNorm 的 `weight`。

### 疊起來

一個 block 進去 `(B, T, d_model)` 出來還是 `(B, T, d_model)`，所以可以疊任意層，每層自己的參數。標準設定疊 4 層，下一集組成完整模型。

## 重點整理

- Block：`x = x + attn(norm(x))`，`x = x + ffn(norm(x))`。
- 殘差連接讓梯度有直通的路，深模型才訓練得動。
- pre-norm：norm 在子層之前，殘差路上沒有 norm；比 post-norm 穩。
- `ModelConfig` 集中所有尺寸，預設是標準設定；`TransformerBlock(cfg, rope)` 的參數 \\( 4d^2 + 3 d \cdot d_{\text{ff}} + 2d \\)。
