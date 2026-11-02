# `einsum`

## 本集目標

看懂 `torch.einsum` 的下標字串，用它寫出點積、矩陣乘法、轉置、加總，以及三維張量的批次矩陣乘法。

## 正文

### 為什麼需要它

`@` 很好用，但張量一到三維以上，你就得開始想「哪一維要轉置、哪一維要對齊」。`einsum` 讓你直接用字母幫每個維度取名字，寫出來的式子就是你心裡想的那件事。LLM 的程式碼裡很常看到它。

### 規則

`torch.einsum("輸入下標,輸入下標->輸出下標", 張量, 張量)`：

1. 每個輸入張量給一串字母，一個字母對應一個維度。
2. 同一個字母出現在不同張量，代表那兩個維度要對齊（長度要一樣），對應位置相乘。
3. 箭頭右邊是輸出的維度；左邊有、右邊沒有的字母會被**加總**掉。

先看點積和它的反面：

```python
import torch

a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
print(torch.einsum("i,i->", a, b))
print(torch.einsum("i,j->ij", a, b))
```

執行結果：

```text
tensor(32.)
tensor([[ 4.,  5.,  6.],
        [ 8., 10., 12.],
        [12., 15., 18.]])
```

- `"i,i->"`：兩邊同一個 `i`，對應相乘；輸出沒有 `i`，所以加總。這就是點積。
- `"i,j->ij"`：字母不同，不對齊；輸出兩個都保留，得到一張 3×3 的表，每格是 `a[i] * b[j]`。

### 矩陣乘法、轉置、加總

```python
import torch

A = torch.arange(6.).reshape(2, 3)
B = torch.arange(6.).reshape(3, 2)
print(torch.einsum("ik,kj->ij", A, B))
print(A @ B)
print(torch.einsum("ij->ji", A))
print(torch.einsum("ij->i", A))
print(torch.einsum("ij->j", A))
```

執行結果：

```text
tensor([[10., 13.],
        [28., 40.]])
tensor([[10., 13.],
        [28., 40.]])
tensor([[0., 3.],
        [1., 4.],
        [2., 5.]])
tensor([ 3., 12.])
tensor([3., 5., 7.])
```

- `"ik,kj->ij"`：`k` 是內側維度，對齊後加總，剩下 `i` 和 `j`。和 `A @ B` 一模一樣。
- `"ij->ji"`：只有一個輸入，交換輸出字母的順序，就是轉置。
- `"ij->i"`：`j` 被加總，每列一個數，等於 `A.sum(dim=1)`。
- `"ij->j"`：`i` 被加總，等於 `A.sum(dim=0)`。

### 批次：很多句子一起算

LLM 一次會處理好幾個句子，張量就變成三維：`(句子數, token 數, 向量維度)`。要算每個句子裡 token 兩兩之間的點積：

```python
import torch

Q = torch.arange(12.).reshape(2, 3, 2)
K = torch.arange(12.).reshape(2, 3, 2)
scores = torch.einsum("bqd,bkd->bqk", Q, K)
print(scores.shape)
print(scores[0])
print(torch.equal(scores, Q @ K.transpose(1, 2)))
```

執行結果：

```text
torch.Size([2, 3, 3])
tensor([[ 1.,  3.,  5.],
        [ 3., 13., 23.],
        [ 5., 23., 41.]])
True
```

字母只是名字，但取得好就很好讀：`b` 是第幾個句子，`q` 和 `k` 是兩組 token 的位置，`d` 是向量維度。`"bqd,bkd->bqk"` 唸作「同一個句子裡，第 q 個和第 k 個 token 沿著 d 做點積」。`b` 兩邊都有、輸出也有，所以每個句子各算各的，不會混在一起。

用 `@` 寫同一件事得先 `K.transpose(1, 2)` 把後兩維對調，再靠三維的 `@` 自動對批次。結果一樣，但你得在腦中多轉一次。第 4 章的 attention 分數就是這個形狀：`scores[0]` 是第 0 個句子裡，3 個 token 彼此的相似度表。

### 什麼時候用哪個

二維、意圖明顯的，用 `@`。三維以上、或你想讓讀的人一眼看懂每個維度是什麼，用 `einsum`。本書兩種都會出現。社群裡還有一個更囉嗦但更好讀的套件 `einops`，附錄一會介紹。

## 重點整理

- `einsum` 用字母幫每個維度命名：同名對齊相乘，輸出沒寫的字母被加總。
- `"i,i->"` 是點積、`"ik,kj->ij"` 是矩陣乘法、`"ij->ji"` 是轉置、`"ij->i"` 是沿欄加總。
- 三維批次：`"bqd,bkd->bqk"` 讓每個句子各算各的 token 相似度表。
- `.transpose(1, 2)` 對調指定的兩個維度；二維時 `.T` 就夠。
