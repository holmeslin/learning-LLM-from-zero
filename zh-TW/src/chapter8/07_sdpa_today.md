# `F.scaled_dot_product_attention` 現況

## 本集目標

知道 PyTorch 內建的 `torch.nn.functional.scaled_dot_product_attention` 在 CPU、NVIDIA 顯示卡與 Mac 上各走哪一條路，並用實測比較它和本書的兩個版本。

## 正文

### 一個函式，好幾種實作

PyTorch 把 attention 包成一個函式：

```python,norun
import torch.nn.functional as F

out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

它的數學和第 4 章的 `scaled_dot_product_attention` 一樣，但底下會依硬體、dtype、形狀自動挑一個實作。`torch.nn.attention.SDPBackend` 列出了這些實作：

- `MATH`：照公式用一般運算拼起來，和第 4 章一樣會建出整個分數矩陣，什麼情況都能跑。
- `FLASH_ATTENTION`：FlashAttention-2 的 kernel，NVIDIA 顯示卡、半精度。
- `EFFICIENT_ATTENTION`：另一種同樣分塊、不存分數矩陣的 kernel，限制比較少。
- `CUDNN_ATTENTION`：NVIDIA cuDNN 函式庫提供的版本。

在 NVIDIA 顯示卡上，PyTorch 會自動挑能用的最快版本，也可以用 `with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):` 指定。本書照「從頭寫」的規則，正文與套件不用這個函式，這一集只拿它來比較。

### CPU：也是 FlashAttention

看 autograd 記下的 backward 名稱，就知道它走哪條路：

```python
import torch
import torch.nn.functional as F

from llmz.flash import flash_attention
from llmz.profile import saved_activation_bytes

torch.manual_seed(0)
q, k, v = (torch.randn(1, 16, 1024, 32, requires_grad=True) for _ in range(3))
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(out.grad_fn.name())
print(torch.allclose(out, flash_attention(q, k, v, True), atol=1e-5))
for fn in [lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), lambda: flash_attention(q, k, v, True)]:
    print(f"{saved_activation_bytes(lambda: fn().sum(), exclude=[q, k, v]) / 1e6:.2f} MB")
```

執行結果：

```text
ScaledDotProductFlashAttentionForCpuBackward0
True
2.16 MB
2.16 MB
```

PyTorch 2.9 在 CPU 上用的是 FlashAttention 式的實作，存下來的東西和我們第 5 集的版本一模一樣多：輸出加 logsumexp。

### Mac：訓練時還是存整個矩陣

在 MPS 上，同樣的呼叫（輸入需要梯度時）印出的 backward 名稱是 `UnsafeViewBackward0`，往回追 autograd 的圖會看到 `BmmBackward0`、`SafeSoftmaxBackward0`：它被拆成矩陣乘法加 softmax，softmax 的輸出（整個分數矩陣大小）會被存下來給 backward。

M4 Max 實測，batch 1、16 個 head、每個 head 32 維、causal，forward 加 backward 一次的時間，以及 forward 之後還佔著的記憶體：

| 長度 | 第 4 章 | 第 5 集（純 PyTorch 分塊） | `F.scaled_dot_product_attention` |
| --- | --- | --- | --- |
| 256 | 2.7 ms，9 MB | 12.6 ms，0.5 MB | 2.5 ms，6 MB |
| 1024 | 13.3 ms，138 MB | 138 ms，2.2 MB | 7.3 ms，88 MB |
| 4096 | 251 ms，2173 MB | 1815 ms，8.7 MB | 82 ms，1116 MB |
| 8192 | 1138 ms，8675 MB | 7443 ms，17 MB | 523 ms，4345 MB |

讀這張表：

- 內建版本最快，長度 4096 時比第 4 章快三倍。它把 softmax 等步驟融合了，搬運少了一些。
- 但它的記憶體仍然跟 \\( T^2 \\) 成正比：8192 時的 4345 MB 正好是一個 `(16, 8192, 8192)` 的 float32 矩陣。第 4 章存了兩個，它存一個。
- 我們的分塊版本記憶體最少，從頭到尾是線性的，但慢了一個數量級，原因在第 4 集說過。

所以截至 PyTorch 2.9，Mac 上的 PyTorch 沒有內建「又快、記憶體又是線性」的訓練用 attention；長文本訓練是 NVIDIA 顯示卡的強項。本書的標準設定長度只有 256，分數矩陣 134 MB 一層，還放得下，所以模型沿用第 4 章的寫法。

### FlashAttention 之後

FlashAttention 在 2022 年提出，之後的 FlashAttention-2（2023）改善了平行化與分工，FlashAttention-3（2024）針對 NVIDIA H100 的新硬體功能再加速。核心想法都沒變：分塊、online softmax、不存分數矩陣、backward 重算。你這一章寫的就是那個核心。

## 重點整理

- `F.scaled_dot_product_attention` 依硬體、dtype、形狀自動選實作：MATH、FLASH_ATTENTION、EFFICIENT_ATTENTION、CUDNN_ATTENTION。
- PyTorch 2.9 的 CPU 版是 FlashAttention 式的，存的東西和第 5 集一樣：輸出加 logsumexp。
- MPS 上訓練時它被拆成矩陣乘法加 softmax，比第 4 章快、記憶體少一半，但仍是 \\( O(T^2) \\)。
- 純 PyTorch 分塊版記憶體最省但最慢；真正又快又省要靠單一 kernel（NVIDIA 上的 FlashAttention）。

## 自我挑戰

這一章放進 `src/llmz/` 的是 `src/llmz/flash.py` 的 `flash_attention_forward`、`flash_attention_backward` 與 `FlashAttention`（`flash_attention`）。把前兩個函式的本體換成 `raise NotImplementedError`，執行：

```bash
uv run pytest tests/test_flash.py
```

自己寫回來讓測試通過。先寫 forward，用第 3 集的 online softmax；backward 照第 5 集的五個公式，記得 \\( D = \text{rowsum}(dO \circ O) \\)。

再試一題：把 `block_q`、`block_k` 改成 16、32、128，確認結果不變，並用第 7 章的 `timeit` 量量看在 MPS 上哪個最快。
