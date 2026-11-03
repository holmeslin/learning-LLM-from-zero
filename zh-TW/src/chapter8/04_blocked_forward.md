# 純 PyTorch 分塊 attention forward

## 本集目標

把 tiling 與 online softmax 組起來，寫出 FlashAttention 的 forward：一塊 query 對一塊 key 地算，從頭到尾不建出 \\( T \times T \\) 的分數矩陣，結果和第 4 章的 attention 一樣。

## 正文

### 兩層迴圈

外層迴圈一次拿 `block_q` 個 query，內層迴圈一次拿 `block_k` 個 key 和 value。每一對小塊做三件事：

1. 算這一小塊的分數 `qi @ kj.T`，形狀 `(block_q, block_k)`。
2. 用上一集的 online softmax 更新這塊 query 的 `m`、`l`、`acc`。
3. 分數用完就丟。

內層迴圈跑完，這塊 query 的輸出就是 `acc / l`。整段程式放在 `src/llmz/flash.py`：

```python,ignore
{{#include ../../../src/llmz/flash.py:forward}}
```

和上一集比，只多了幾件事：

- `...`（省略號）代表前面任意多個維度，所以 `(B, H, T, d)` 或 `(T, d)` 都能用，每個 batch、每個 head 一起算。
- 縮放 \\( 1/\sqrt{d} \\) 先乘在 `qi` 上，和乘在分數上一樣，但只要乘 `block_q × d` 個數。
- `q.new_empty`、`qi.new_zeros`、`qi.new_full` 建立和 `q` 同 dtype、同 device 的新張量。
- 除了輸出 `o`，還回傳每一列分數的 logsumexp `lse = m + log(l)`。它只有 `(..., T)` 這麼大，下一集的 backward 會用到。
- 最後一塊可能不滿 64 個（例如長度 200 = 64 + 64 + 64 + 8），切片會自動只拿剩下的，所以用 `qi.shape[-2]` 而不是 `block_q` 算位置。

### causal：跳過看不到的塊

causal attention 裡，第 \\( i \\) 個 query 只能看 \\( j \le i \\) 的 key。以塊來看，有三種情況：

- key 塊整個在 query 塊之後：全部看不到，**直接跳過不算**。這就是 `k_end = i + qi.shape[-2]`。
- key 塊整個在 query 塊之前：全部看得到，不用遮。
- 對角線上的塊：一部分看得到，用 `masked_fill` 把 `cols > rows` 的位置設成 \\( -\infty \\)。

跳過的塊幾乎佔一半，所以 causal 版本的計算量差不多是一般版本的一半。第 4 章的寫法則是先把整個矩陣算出來，再把一半遮掉。

每塊 query 的第一個 key 塊一定是第 0 塊，它包含位置 0，每個 query 都看得到，所以 `m` 在第一塊之後就不再是 \\( -\infty \\)，不會出現 \\( -\infty - (-\infty) \\) 這種無意義的運算。

### 驗證

```python
import torch

from llmz.attention import causal_mask, scaled_dot_product_attention
from llmz.flash import flash_attention_forward

torch.manual_seed(0)
q, k, v = (torch.randn(2, 4, 200, 32) for _ in range(3))
for causal in [False, True]:
    o, lse = flash_attention_forward(q, k, v, causal)
    ref = scaled_dot_product_attention(q, k, v, causal_mask(200) if causal else None)
    print(causal, tuple(o.shape), tuple(lse.shape), torch.allclose(o, ref, atol=1e-5))
```

執行結果：

```text
False (2, 4, 200, 32) (2, 4, 200) True
True (2, 4, 200, 32) (2, 4, 200) True
```

兩種情況都和第 4 章的版本相同（只差浮點誤差，加法的順序不一樣了）。

### 記憶體與速度

forward 過程中最大的暫存張量是一塊分數 `(B, H, 64, 64)`，跟 \\( T \\) 無關。輸出 `o` 和 `lse` 跟 \\( T \\) 成正比。分數矩陣不見了。

但這個純 PyTorch 版本很慢。M4 Max 的 MPS 上，batch 1、16 個 head、長度 1024 的 causal attention，forward 第 4 章的版本 4.7 ms，我們的分塊版本 66.5 ms。每一小塊都是好幾個獨立的小 kernel，Python 迴圈每一圈都要丟一批工作給 GPU，`m`、`l`、`acc` 每次更新也都寫回了記憶體。真正的 FlashAttention 是**一個** kernel：一塊 query 的 `m`、`l`、`acc` 從頭到尾待在晶片上，第 6 集會看到這種 kernel 怎麼寫。我們這個版本是用來看懂演算法、驗證數學，而不是用來加速的。

## 重點整理

- FlashAttention forward：外層一塊 query、內層一塊 key／value，每對小塊算分數、做 online softmax 更新、丟掉分數。
- 回傳輸出 `o` 與每列的 logsumexp `lse`，兩者都只跟 \\( T \\) 成正比。
- causal 時整塊看不到的 key 直接跳過，對角線上的塊才需要遮。
- 純 PyTorch 版結果正確但很慢；真正的加速要靠把整個迴圈寫成一個 GPU kernel。
