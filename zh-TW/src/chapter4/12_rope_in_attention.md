# 把 RoPE 接進 attention

## 本集目標

看懂 RoPE 在 multi-head attention 裡的位置：投影之後、算分數之前，只轉 q 和 k，每個 head 各自轉。

## 正文

### 接在哪裡

上一集的 `MultiHeadSelfAttention` 已經有這幾行：

```python,ignore
        if self.rope is not None:
            positions = torch.arange(T, device=x.device)
            q = self.rope(q, positions)
            k = self.rope(k, positions)
```

q、k 此時是 `(B, H, T, d_head)`，RoPE 對最後一維（`d_head`）做旋轉、用倒數第二維（T）的位置，前面的 B、H 一律照 broadcasting 處理，所以每個 head 各自轉、每段文字各自轉。`positions` 用 `torch.arange(T)`，在 `x` 所在的 device 上建立，才不會 CPU 與 GPU 混用（第 1 章第 9 集）。

`RoPE` 物件是在外面建好傳進來的，因為所有層共用同一張 cos／sin 表，沒必要每層各存一份；第 14 集的 `TransformerLM` 會建一個傳給每一層。

### 看得到位置了

拿一段「每個位置的 token 都一樣」的輸入來試。沒有 RoPE 時，每個 query 對每個 key 的分數都相同（向量都一樣）；有 RoPE 時，分數會隨距離變化：

```python
import torch

from llmz.attention import MultiHeadSelfAttention, RoPE

torch.manual_seed(0)
d_model, n_heads, T = 16, 2, 5
x = torch.randn(1, 1, d_model).expand(1, T, d_model)


def last_row_scores(attn):
    q = attn.q_proj(x).reshape(1, T, n_heads, -1).transpose(1, 2)
    k = attn.k_proj(x).reshape(1, T, n_heads, -1).transpose(1, 2)
    if attn.rope is not None:
        pos = torch.arange(T)
        q, k = attn.rope(q, pos), attn.rope(k, pos)
    scores = q @ k.transpose(-2, -1) / 8 ** 0.5
    return scores[0, 0, -1]  # 第 0 個 head，最後一個位置對每個 key 的分數


torch.manual_seed(0)
without = MultiHeadSelfAttention(d_model, n_heads, rope=None)
torch.manual_seed(0)
with_rope = MultiHeadSelfAttention(d_model, n_heads, rope=RoPE(d_model // n_heads, 32))
print(last_row_scores(without))
print(last_row_scores(with_rope))
```

執行結果：

```text
tensor([0.3538, 0.3538, 0.3538, 0.3538, 0.3538], grad_fn=<SelectBackward0>)
tensor([0.9112, 0.8126, 0.5215, 0.3025, 0.3538], grad_fn=<SelectBackward0>)
```

`expand` 把一個向量複製成 T 份（不真的複製記憶體，只是 broadcasting）。兩個 attention 用同一個種子建立，所以投影矩陣一模一樣。沒有 RoPE，最後一個位置看五個 key 的分數五個全同；有 RoPE，分數隨距離變化。注意最後一個（距離 0，自己看自己）和沒有 RoPE 時一模一樣：q 和 k 轉了同樣的角度，點積不變，這正是上一集說的「只和位置差有關」。這就是模型「分得出前一個字和前三個字」的來源。

### 不轉 value

value 不轉。位置資訊的用途只是決定權重（誰看誰看多少），value 是「被取走的內容」，轉了反而把內容弄亂。

## 重點整理

- RoPE 在投影之後、算分數之前套用，只轉 q 和 k，形狀 `(B, H, T, d_head)` 直接吃，每個 head 各自轉。
- `positions` 用 `torch.arange(T, device=x.device)`。
- 一個 `RoPE` 物件給所有層共用。
- 沒有 RoPE 時相同的 token 得到相同的分數；有 RoPE 時分數隨距離變化。
