# 什麼是 scaling law

## 本集目標

認識冪律（power law）：為什麼它在 log-log 圖上是一條直線，以及 loss 怎麼隨著模型變大而下降。

## 正文

### 冪律

第 6 章換了很多設定，每次都要真的訓練才知道結果。如果 loss 和模型大小之間有固定的關係，我們就能用幾個小模型的結果預測大模型。2020 年 OpenAI 的 Kaplan 等人發現，在很大的範圍內，loss 和參數量 \\( N \\) 的關係是一個**冪律**（power law）：

\\[ L(N) = \left( \frac{N_c}{N} \right)^{\alpha_N} \\]

他們擬合出 \\( \alpha_N \approx 0.076 \\)、\\( N_c \approx 8.8 \times 10^{13} \\)（\\( N \\) 是不含 embedding 的參數量）。這種關係叫 **scaling law**。

### log-log 是直線

兩邊取 log：

\\[ \log L = \alpha_N \log N_c - \alpha_N \log N \\]

\\( \log L \\) 對 \\( \log N \\) 是一條直線，斜率是 \\( -\alpha_N \\)。所以 scaling law 的圖都畫在兩個軸都取 log 的座標上：看到直線就是冪律。

```python
import math

alpha, n_c = 0.076, 8.8e13
previous = None
for n in [1e6, 1e7, 1e8, 1e9, 1e10]:
    loss = (n_c / n) ** alpha
    slope = "" if previous is None else f"  斜率 {(math.log(loss) - math.log(previous[1])) / (math.log(n) - math.log(previous[0])):.3f}"
    print(f"N = {n:.0e}  loss = {loss:.3f}{slope}")
    previous = (n, loss)
```

執行結果：

```text
N = 1e+06  loss = 4.016
N = 1e+07  loss = 3.371  斜率 -0.076
N = 1e+08  loss = 2.830  斜率 -0.076
N = 1e+09  loss = 2.376  斜率 -0.076
N = 1e+10  loss = 1.994  斜率 -0.076
```

每次參數量變 10 倍，loss 都乘上同一個比例 \\( 10^{-0.076} \approx 0.84 \\)，斜率固定是 −0.076。參數量加倍，loss 只降約 5%（\\( 2^{-0.076} \approx 0.95 \\)）。這說明了為什麼模型要一直變大：每一點進步都很貴，但很穩定、可以預測。

### 不只參數量

Kaplan 等人也發現 loss 對資料量 \\( D \\)、對計算量 \\( C \\) 都是冪律，前提是另外兩個因素不是瓶頸（例如看資料量的影響時，模型要夠大）。這帶出一個問題：計算預算固定時，該把它花在更大的模型，還是更多的資料？下一集的 Chinchilla 回答了這個問題。

### 為什麼有用

假設你有一千萬美元的預算要訓練一個模型。scaling law 的做法是：

1. 用很小的預算訓練幾十個小模型，每個只花全部預算的一小部分。
2. 擬合 loss 與規模的冪律。
3. 外推到真正的預算，決定模型大小與資料量，並預測最後的 loss。

這一章的第 3～5 集就是在練習這三步。

## 重點整理

- 冪律 \\( y = k x^a \\)：在 log-log 座標上是斜率為 \\( a \\) 的直線。
- Kaplan 等人（2020）：\\( L(N) = (N_c/N)^{\alpha_N} \\)，\\( \alpha_N \approx 0.076 \\)，參數量加倍 loss 約降 5%。
- loss 對參數量、資料量、計算量都呈冪律。
- scaling law 讓我們用便宜的小實驗預測昂貴的大實驗。
