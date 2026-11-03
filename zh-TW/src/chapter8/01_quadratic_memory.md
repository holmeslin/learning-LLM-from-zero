# attention 的 O(T²) 記憶體

## 本集目標

算出並量出 attention 的分數矩陣隨序列長度怎麼長大，知道為什麼長文本的訓練會先卡在它。

## 正文

### 分數矩陣有多大

第 4 章第 9 集的 attention：

```python,ignore
{{#include ../../../src/llmz/attention.py:sdpa}}
```

`q @ k.transpose(-2, -1)` 的形狀是 `(B, H, T, T)`：每個 head、每個 query 位置對每個 key 位置都有一個分數。它的大小跟 \\( T^2 \\) 成正比，數學上寫成 \\( O(T^2) \\)，意思是「長度變成兩倍，它變成四倍」。

標準設定 batch 32、16 個 head，換不同長度算算看：

```python
for T in [256, 1024, 4096, 8192]:
    scores_bytes = 32 * 16 * T * T * 4
    print(f"T={T:5}  {scores_bytes / 1e9:7.2f} GB")
```

執行結果：

```text
T=  256     0.13 GB
T= 1024     2.15 GB
T= 4096    34.36 GB
T= 8192   137.44 GB
```

長度 256 時是 0.13 GB，第 7 章第 1 集算過。長度到 4096（現在的 LLM 很常見的長度），一層的一個分數矩陣就要 34 GB，是 16 GB 的 Mac 記憶體的兩倍。模型的其他部分，例如每一層進出的 `(B, T, d_model)`，只跟 \\( T \\) 成正比，同樣從 256 變 4096 只會大 16 倍。

### 量 backward 要存多少

用第 7 章第 4 集的 `saved_activation_bytes`，量一層 attention（batch 1、16 個 head、每個 head 32 維）為了 backward 存了多少：

```python
import torch

from llmz.attention import causal_mask, scaled_dot_product_attention
from llmz.profile import saved_activation_bytes

for T in [256, 512, 1024, 2048]:
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 16, T, 32, requires_grad=True) for _ in range(3))
    saved = saved_activation_bytes(
        lambda: scaled_dot_product_attention(q, k, v, causal_mask(T)).sum(), exclude=[q, k, v]
    )
    print(f"T={T:5}  {saved / 1e6:7.1f} MB  一個分數矩陣 {16 * T * T * 4 / 1e6:6.1f} MB")
```

執行結果：

```text
T=  256      8.5 MB  一個分數矩陣    4.2 MB
T=  512     33.9 MB  一個分數矩陣   16.8 MB
T= 1024    135.5 MB  一個分數矩陣   67.1 MB
T= 2048    541.5 MB  一個分數矩陣  268.4 MB
```

每次長度加倍，存的東西變四倍，而且大約是兩個分數矩陣的大小：`softmax` 裡 `exp` 的結果，以及 softmax 的輸出（`@ v` 的 backward 要用）。causal mask 是布林值，只佔一點點。

### 不只是記憶體，還有時間

第 7 章第 8 集算過，\\( QK^\top \\) 的 arithmetic intensity 只有 13，卡在記憶體。分數矩陣寫出去之後，softmax 還要讀一次、寫一次，`@ v` 再讀一次。一個 \\( T \times T \\) 的矩陣來回搬了好幾趟，真正的計算卻不多。

所以問題很清楚：**能不能不把整個分數矩陣寫到記憶體裡？** 一次只算一小塊、用完就丟，最後的結果還要和原本一模一樣。接下來兩集準備兩個工具，第 4 集組起來。

## 重點整理

- attention 的分數矩陣是 `(B, H, T, T)`，大小跟 \\( T^2 \\) 成正比；標準設定長度 4096 時一個就要 34 GB。
- 第 4 章的 attention 為了 backward 存了約兩個分數矩陣大小的資料，長度加倍就變四倍。
- 分數矩陣還要被 softmax、`@ v` 來回讀寫，時間也卡在搬資料。
- 目標：不把整個 \\( T \times T \\) 矩陣寫到記憶體，又得到一模一樣的結果。
