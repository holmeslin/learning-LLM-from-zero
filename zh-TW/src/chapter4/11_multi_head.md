# multi-head attention

## 本集目標

寫出 `MultiHeadSelfAttention`：用 Linear 從輸入做出 q、k、v，拆成多個 head 各自做 attention，再接回來。

## 正文

### self-attention

上一集的 q、k、v 是隨機給的。在 Transformer 裡它們都從同一個輸入 `x` 用三個 Linear 投影出來，所以叫 **self**-attention：每個位置拿自己的向量去當 query，看其他位置（的 key），取其他位置（的 value）。

### 為什麼要多個 head

一次 attention 只有一組權重：位置 i 對每個 j 一個數字。但一個 token 可能同時想看「前一個字」和「句子開頭的主詞」，一組權重顧不到。做法：把 `d_model` 維切成 `n_heads` 份，每份 `d_head = d_model / n_heads` 維，各自做一次 attention（各有自己的權重矩陣），最後把結果接回 `d_model` 維。標準設定 512 維、16 個 head，每個 head 32 維。

```python,ignore
{{#include ../../../src/llmz/attention.py:mha}}
```

形狀是這一集的重點：

1. `q_proj(x)` 是 `(B, T, d_model)`。
2. `reshape(B, T, n_heads, d_head)`：最後一維切成 head。
3. `transpose(1, 2)` 變 `(B, H, T, d_head)`：把 head 移到前面，這樣上一集的 `scaled_dot_product_attention` 會把 `B` 和 `H` 都當成 batch 維度，T 個位置對 T 個位置各自算，16 個 head 一次算完。
4. 算完 `transpose(1, 2)` 回 `(B, T, H, d_head)`，`reshape` 接回 `(B, T, d_model)`。
5. 最後再過一個 `out_proj`，讓不同 head 的結果混合。

`rope` 參數下一集才用，這裡先傳 `None`。

```python
import torch

from llmz.attention import MultiHeadSelfAttention
from llmz.utils import count_parameters

torch.manual_seed(0)
attn = MultiHeadSelfAttention(d_model=16, n_heads=4)
x = torch.randn(2, 6, 16)
y = attn(x)
print(y.shape)
print(count_parameters(attn), 4 * 16 * 16)
x2 = x.clone()
x2[:, 4:] = torch.randn(2, 2, 16)
y2 = attn(x2)
print(torch.allclose(y[:, :4], y2[:, :4]), torch.allclose(y[:, 4:], y2[:, 4:]))
```

執行結果：

```text
torch.Size([2, 6, 16])
1024 1024
True False
```

輸入輸出同形狀；參數是四個 `(d_model, d_model)` 的矩陣，和 head 數無關（head 只是把同一個矩陣的輸出切開）。最後兩行驗證 causal：改掉第 4、5 個位置的輸入，前四個位置的輸出一個數都沒變，後兩個變了。

### 為什麼 head 不會增加參數

16 個 head 各 32 維，`q_proj` 的 512 列裡每 32 列屬於一個 head；拆開來看是 16 個 `(32, 512)` 的小矩陣，合起來就是一個 `(512, 512)`。所以多 head 幾乎是免費的，只是把同樣的計算分成幾組。

## 重點整理

- self-attention：q、k、v 都由同一個輸入經三個 Linear 得到。
- multi-head：`d_model` 切成 `n_heads` 份各自 attention，形狀走 `(B, T, d)` → `(B, H, T, d_head)` → 回來，最後 `out_proj` 混合。
- `scaled_dot_product_attention` 把 B、H 都當 batch，一次算完所有 head。
- 參數 \\( 4 d_{\text{model}}^2 \\)，與 head 數無關。
