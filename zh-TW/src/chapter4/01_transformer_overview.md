# 一張圖看懂 Transformer LM

## 本集目標

先看整體：token id 進去、每個位置對「下一個 token」的 logits 出來，中間是 embedding、幾層 Transformer block、最後一個 Linear。記住每一段的張量形狀，之後每一集只是填其中一格。

## 正文

### 從頭到尾

```text
ids        (B, T)            整數，第 3 章 tokenizer 的輸出
  │  Embedding                查表
  ▼
x          (B, T, d_model)    每個 token 一個向量
  │  Transformer block × n_layers
  │     x = x + attention(RMSNorm(x))      看前面的 token，混合資訊
  │     x = x + FFN(RMSNorm(x))            每個位置各自做一次非線性變換
  ▼
x          (B, T, d_model)
  │  RMSNorm → Linear(d_model, vocab_size)
  ▼
logits     (B, T, vocab_size) 每個位置對下一個 token 的分數
```

三個字母全書通用：**B** 是 batch 大小（一次處理幾段文字）、**T** 是序列長度（一段幾個 token）、**d_model** 是模型內部每個 token 的向量維度。本書的標準設定：d_model 512、4 層、16 個 head、vocab 10000、T 最長 256。

### 語言模型在做什麼

輸入 `ids[b, t]`，輸出 `logits[b, t]` 是一排 `vocab_size` 個分數，對應「第 t+1 個 token 會是誰」。第 1 章第 13 集的 softmax 把它變成機率，第 15 集的 cross-entropy 拿正確答案（就是 `ids[b, t+1]`）算 loss。所以一段 T 個 token 的文字，可以同時練 T 個「猜下一個」的題目，這是 Transformer 訓練有效率的關鍵。

注意第 t 個位置只能看第 0 到 t 個 token，不能偷看 t+1 以後的，否則答案就在輸入裡了。這由第 10 集的 causal mask 保證。

### 先用最簡單的零件走一遍形狀

用第 1 章的工具就能走完整條路（只是中間沒有 Transformer block）：

```python
import torch

torch.manual_seed(0)
B, T, vocab_size, d_model = 2, 5, 10, 4
ids = torch.randint(0, vocab_size, (B, T))
table = torch.randn(vocab_size, d_model)
x = table[ids]
print(ids.shape, x.shape)
W = torch.randn(d_model, vocab_size)
logits = x @ W
print(logits.shape)
print(logits[0, 0])
```

執行結果：

```text
torch.Size([2, 5]) torch.Size([2, 5, 4])
torch.Size([2, 5, 10])
tensor([ 1.3015,  0.0892, -2.5469,  4.0353,  0.1479,  0.1927,  1.4984,  1.7320,
        -1.1381, -1.4517])
```

`table[ids]` 是第 1 章第 4 集的「用索引張量取多個位置」，這就是 Embedding（第 3 集）。`x @ W` 是第 7 集的矩陣乘法，這就是最後的 Linear（下一集）。`logits[0, 0]` 是第 0 段文字第 0 個位置對 10 個 token 的分數，現在是亂數，第 5 章訓練後它會變得有意義。

### 本章的順序

| 集 | 零件 | 形狀變化 |
| --- | --- | --- |
| 2 | Linear | `(..., in)` → `(..., out)` |
| 3 | Embedding | `(B, T)` → `(B, T, d_model)` |
| 4 | RMSNorm | 不變 |
| 5 | SiLU、SwiGLU | 不變（中間經過 `d_ff`） |
| 6、7 | RoPE | 不變 |
| 8～12 | attention | 不變（中間拆成 `n_heads` 個 head） |
| 13 | Transformer block | 不變 |
| 14 | TransformerLM | `(B, T)` → `(B, T, vocab_size)` |

## 重點整理

- 語言模型：`(B, T)` 的 token id → `(B, T, vocab_size)` 的 logits，每個位置預測下一個 token。
- 流程：Embedding → n 層 block（attention 加殘差、FFN 加殘差，各自前面有 RMSNorm）→ RMSNorm → Linear。
- B 是 batch、T 是序列長度、d_model 是內部向量維度；本書標準設定 512、4 層、16 head。
- 第 t 個位置不能看 t+1 以後的 token。
