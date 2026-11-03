# Embedding

## 本集目標

寫出 `Embedding`：一張 `(vocab_size, d_model)` 的表，token id 查表拿到自己的向量。

## 正文

### 每個 token 一列

```python,ignore
{{#include ../../../src/llmz/layers.py:embedding}}
```

`forward` 只有一行：`self.weight[ids]`。這是第 1 章第 4 集的索引，`ids` 是什麼形狀，結果就多一個 `embedding_dim` 的維度。這張表是參數，訓練時梯度會流回對應的那幾列，所以每個 token 的向量是**學**出來的：意思相近的 token 向量會變得相近。

初始化用標準差 1 的常態分布、截在 ±3：和上一集不同，這裡不做矩陣乘法，每個向量自己就是一個 token 的表示，大小 1 剛好。

```python
import torch

from llmz.layers import Embedding

torch.manual_seed(0)
emb = Embedding(10, 4)
ids = torch.tensor([[3, 7], [0, 3]])
x = emb(ids)
print(x.shape)
print(torch.equal(x[0, 0], emb.weight[3]), torch.equal(x[1, 1], emb.weight[3]))
print(torch.equal(x[0, 0], x[1, 1]))
```

執行結果：

```text
torch.Size([2, 2, 4])
True True
True
```

兩段文字裡的 token 3 拿到的是同一列，不管它在哪個位置。「位置」的資訊要靠第 6、7 集的 RoPE 另外加進去。

### 大小

標準設定 vocab 10000 × d_model 512 = 512 萬個參數，是整個模型的四分之一左右（第 15 集會算全部）。最後一層 `Linear(d_model, vocab_size)` 的形狀剛好和它相同，有些模型讓兩者共用同一個矩陣來省參數，本書不共用，簡單一點。

### 為什麼不用 one-hot 乘矩陣

教科書常把 embedding 解釋成「one-hot 向量乘一個矩陣」：長度 vocab_size、只有一格是 1 的向量，乘 `(vocab_size, d_model)` 的矩陣，結果就是那一列。數學上一樣，但那是一萬個乘法只為了取一列，查表直接取就好。

## 重點整理

- `Embedding(num, dim)`：參數是 `(num, dim)` 的表，`forward` 是 `weight[ids]`，輸出比 `ids` 多一個 `dim` 維度。
- 同一個 token 在任何位置拿到同一列；位置資訊另外加。
- 標準設定下 embedding 有 512 萬個參數。
