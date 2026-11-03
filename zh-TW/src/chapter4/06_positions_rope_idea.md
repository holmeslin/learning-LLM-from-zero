# 位置資訊與 RoPE 的想法

## 本集目標

理解為什麼 attention 本身看不到順序、為什麼要加位置資訊，以及 RoPE 的核心想法：把向量當成平面上的點、依位置旋轉，兩個向量的點積就只和它們的**距離**有關。

## 正文

### attention 分不出順序

第 3 集說過，Embedding 給同一個 token 同一個向量，不管它在第幾個位置。後面的 attention（第 9 集）是對這些向量算點積再加權平均，把輸入的順序打亂，每個位置算出來的結果也只是跟著換位置而已。所以「貓追狗」和「狗追貓」在模型眼裡會一樣，得另外把位置塞進去。

### 兩種做法

- **絕對位置**：準備一張 `(max_len, d_model)` 的表，第 t 個位置的向量加到第 t 個 token 的 embedding 上。簡單，但模型學到的是「第 37 個位置」這種絕對資訊；訓練時只看過 256 長，第 300 個位置就沒見過。
- **相對位置**：讓 attention 的分數只跟「兩個 token 差幾格」有關。語言的規律多半是相對的（「前一個字」「前三個字」），而且更容易推廣到沒看過的長度。

RoPE（rotary position embedding）是相對位置的一種做法，而且實作起來只是幾個乘法。

### 旋轉

把一個 2 維向量看成平面上的點，旋轉 \\( \theta \\) 角：

\\[ \begin{pmatrix} x' \\\\ y' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\\\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x \\\\ y \end{pmatrix} \\]

RoPE 的規則：位置 m 的向量轉 \\( m\theta \\)。關鍵性質：兩個點各自旋轉之後的點積，只跟它們**角度的差**有關，也就是只跟位置差 \\( m - n \\) 有關。驗證一下：

```python
import torch


def rotate(v, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((v[0] * c - v[1] * s, v[0] * s + v[1] * c))


theta = torch.tensor(0.3)
q = torch.tensor([1.0, 2.0])
k = torch.tensor([-0.5, 1.5])
for m, n in [(3, 1), (8, 6), (23, 21), (5, 1)]:
    score = rotate(q, m * theta) @ rotate(k, n * theta)
    print(m, n, round(score.item(), 4))
```

執行結果：

```text
3 1 3.4749
8 6 3.4749
23 21 3.4749
5 1 3.236
```

`torch.stack` 把兩個純量疊成一個 2 維向量。(3, 1)、(8, 6)、(23, 21) 差都是 2，點積完全相同；(5, 1) 差 4，就不同。所以只要在算 attention 分數之前，把 query 轉 \\( m\theta \\)、key 轉 \\( n\theta \\)，分數就自動帶有「差幾格」的資訊，模型不需要知道絕對位置。

### 高維怎麼辦

向量有 `d_head` 維（本書 32），不是 2 維。做法：兩個兩個一組，分成 `d_head / 2` 個平面，每個平面各自旋轉。而且每一組用**不同的轉速** \\( \theta_i \\)：第一組轉得快，分辨相鄰位置；最後一組轉得極慢，分辨很遠的距離。轉速的公式下一集寫程式時給。

## 重點整理

- attention 本身對順序沒感覺，位置資訊要另外加。
- 絕對位置用加法、不易推廣；相對位置讓分數只看距離。
- RoPE：位置 m 的向量旋轉 \\( m\theta \\)，旋轉後的點積只和 \\( m - n \\) 有關。
- 高維向量兩兩一組各自旋轉，每一組轉速不同。
