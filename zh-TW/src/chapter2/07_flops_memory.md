# FLOPs 與記憶體估算

## 本集目標

在動手訓練之前，用幾個乘法估出模型要多少記憶體、一步訓練要多少計算量，並把 `count_parameters` 放進 `llmz`。

## 正文

### 先數參數

一切估算從「有幾個參數」開始。把它寫成函式放進 `src/llmz/utils.py`：

```python,ignore
{{#include ../../../src/llmz/utils.py:count_parameters}}
```

`p.numel()` 是一個張量的元素個數，`sum` 把全部參數加起來。

```python
import torch
from torch import nn

from llmz.utils import count_parameters


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))


class Big(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(1000, 1000))


print(count_parameters(LinearModel()))
print(count_parameters(Big()))
```

執行結果：

```text
2
1000000
```

### 記憶體：每個參數不只佔一份

一個 `float32` 佔 4 個位元組（第 1 章第 3 集）。訓練時每個參數至少要放三份東西：

| 東西 | 大小 | 為什麼 |
| --- | --- | --- |
| 參數本身 | 4 位元組 | 模型 |
| 梯度 `.grad` | 4 位元組 | `backward()` 寫進來的，形狀和參數一樣 |
| 優化器狀態 | 0～8 位元組 | 本章的 `SGD` 沒有；第 5 章的 AdamW 每個參數再存兩個數 |

所以用 AdamW 訓練時，每個參數要 16 位元組。一百萬個參數是 16 MB，十億個參數是 16 GB，剛好吃光一台 16 GB 的 Mac，這還沒算中間結果。這就是為什麼本書的模型停在幾千萬參數。

```python
params = 1_000_000
print(params * 4 / 1e6, "MB 只放參數")
print(params * 16 / 1e6, "MB 用 AdamW 訓練")
```

執行結果：

```text
4.0 MB 只放參數
16.0 MB 用 AdamW 訓練
```

### 計算量：數乘加的次數

**FLOP**（floating point operation）是一次浮點運算，乘一次或加一次都算一個。矩陣乘法 \\( (m, k) @ (k, n) \\) 的每個輸出元素要 \\( k \\) 次乘法加 \\( k \\) 次加法，輸出有 \\( m \times n \\) 個，所以：

\\[ \text{FLOPs} = 2 \times m \times k \times n \\]

模型 forward 幾乎全是矩陣乘法，所以有個好記的近似：**forward 每個輸入大約花 2 × 參數量 個 FLOPs**（每個參數乘一次、加一次）。backward 要算對輸入與對參數兩份梯度，大約是 forward 的兩倍。合起來：

\\[ \text{訓練一個 token} \approx 6 \times \text{參數量} \text{ FLOPs} \\]

```python
params = 1_000_000
tokens = 40_000_000
flops = 6 * params * tokens
print(f"{flops:.1e} FLOPs")
print(f"{flops / 1e12:.0f} TFLOPs")
```

執行結果：

```text
2.4e+14 FLOPs
240 TFLOPs
```

一百萬參數的模型看 4000 萬個 token，要 240 TFLOPs。M4 Max 的 GPU 大約每秒做十幾 TFLOPs 的浮點運算，但實際上只能用到一部分，所以這會是幾分鐘的事；第 5 章的模型有幾千萬參數，就是幾十分鐘。先估再跑，就不會發生「跑了一晚才發現要三天」。

### 記憶體還有一塊：中間結果

`backward()` 需要 forward 過程中的中間結果（例如每一層的輸入），這些在 forward 時就得留著，叫 **activations**。它們和 batch 大小、序列長度成正比，模型小的時候反而是記憶體的大宗。第 7 章會專門估算，現在先知道：batch 開太大會爆記憶體，就是它。

## 重點整理

- `llmz.utils.count_parameters(model)` 把所有參數的 `numel()` 加總。
- 記憶體：參數、梯度各 4 位元組，AdamW 再加 8，共 16 位元組一個參數；activations 另計。
- 矩陣乘法 \\( (m, k) @ (k, n) \\) 是 \\( 2mkn \\) FLOPs；訓練一個 token 約 \\( 6 \times \\) 參數量。
- 先估再跑。

## 自我挑戰

這一章放進 `src/llmz/` 的只有 `src/llmz/utils.py` 的 `count_parameters`。把它的本體換成 `raise NotImplementedError`，執行：

```bash
uv run pytest tests/test_utils.py
```

自己寫回來讓測試通過。然後把第 5 集的線性回歸整支程式關掉書、從空白檔案默寫一次：資料、`LinearModel`、`mse`、`SGD`、五步迴圈，跑出一樣的輸出。這五步接下來每一章都會出現，現在就要寫到不用想。
