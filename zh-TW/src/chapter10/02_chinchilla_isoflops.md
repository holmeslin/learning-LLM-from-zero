# Chinchilla 與 IsoFLOPs

## 本集目標

理解計算預算 \\( C \approx 6ND \\) 下模型大小與資料量的取捨、IsoFLOPs 曲線的 U 形，以及 Chinchilla 的「每個參數約 20 個 token」。

## 正文

### 預算固定時的取捨

第 2 章第 7 集的估算：訓練的計算量 \\( C \approx 6ND \\)（\\( N \\) 個參數、\\( D \\) 個 token）。預算 \\( C \\) 固定時，\\( N \\) 和 \\( D \\) 一個變大另一個就變小：

- 模型太小：看再多資料，能力也有上限。
- 模型太大：資料只夠它看一點點，還沒學會預算就用完了。

中間有一個最好的分配。

### Chinchilla 的公式

2022 年 DeepMind 的 Hoffmann 等人訓練了四百多個模型，擬合出 loss 同時對 \\( N \\) 和 \\( D \\) 的公式：

\\[ L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta} \\]

\\( E \\) 是資料本身的不確定性，模型再大、資料再多也降不下去；後兩項分別是「模型不夠大」與「資料不夠多」造成的損失。他們的擬合值是 \\( E = 1.69 \\)、\\( A = 406.4 \\)、\\( B = 410.7 \\)、\\( \alpha = 0.34 \\)、\\( \beta = 0.28 \\)。

```python,ignore
{{#include ../../../src/llmz/scaling.py:chinchilla_loss}}
```

`np.asarray(..., dtype=float)` 讓它可以一次算一整個 numpy 陣列。

### IsoFLOPs 曲線

固定 \\( C \\)，掃過不同的 \\( N \\)，資料量就是 \\( D = C / (6N) \\)。把 loss 畫出來，這條「同樣 FLOPs 下的 loss」叫 **IsoFLOPs 曲線**：

```python
import numpy as np

from llmz.scaling import chinchilla_loss

budget = 1e21
for n in [1e8, 3e8, 1e9, 3e9, 1e10, 3e10]:
    d = budget / (6 * n)
    print(f"N = {n:.0e}  D = {d:.1e}  每個參數 {d / n:7.1f} 個 token  loss = {chinchilla_loss(n, d):.4f}")

n = np.logspace(8, 11, 1000)
loss = chinchilla_loss(n, budget / (6 * n))
best = n[np.argmin(loss)]
print(f"最佳 N = {best:.2e}，D = {budget / (6 * best):.2e}，每個參數 {budget / (6 * best) / best:.0f} 個 token")
```

執行結果：

```text
N = 1e+08  D = 1.7e+12  每個參數 16666.7 個 token  loss = 2.6198
N = 3e+08  D = 5.6e+11  每個參數  1851.9 個 token  loss = 2.4344
N = 1e+09  D = 1.7e+11  每個參數   166.7 個 token  loss = 2.3400
N = 3e+09  D = 5.6e+10  每個參數    18.5 個 token  loss = 2.3363
N = 1e+10  D = 1.7e+10  每個參數     1.7 個 token  loss = 2.4160
N = 3e+10  D = 5.6e+09  每個參數     0.2 個 token  loss = 2.5687
最佳 N = 1.82e+09，D = 9.13e+10，每個參數 50 個 token
```

loss 先降後升，是一個 U 形。最低點在 18 億參數，每個參數配 50 個 token。`np.logspace(8, 11, 1000)` 產生從 \\( 10^8 \\) 到 \\( 10^{11} \\)、在 log 座標上等距的 1000 個數；`np.argmin` 找出最小值的位置。

### Chinchilla 改變了什麼

Kaplan 等人（2020）的結論是：預算增加時，大部分應該花在加大模型，最佳參數量約隨 \\( C^{0.73} \\) 成長，資料量只隨 \\( C^{0.27} \\)。依照這個結論，當時的大模型都很大、資料相對少，例如 DeepMind 的 Gopher 有 2800 億參數，看了 3000 億個 token，每個參數約一個 token。

Hoffmann 等人用了三種方法估計最佳分配。前兩種直接從實驗找每個預算的最佳點（第 3 集的做法），結論是參數量與資料量應該**等比例**成長，大約各隨 \\( C^{0.5} \\)，換算下來每個參數約 20 個 token。第三種就是上面的公式，它給出的指數是 0.45 與 0.55（第 4 集會擬合出來），在 \\( 10^{21} \\) 時每個參數 50 個 token，和前兩種不完全一致；第 6 集會看到，後來有人重新擬合，得到的公式和前兩種方法比較接近。他們用和 Gopher 相同的計算量，訓練了 700 億參數、看 1.4 兆個 token 的 Chinchilla，結果全面勝過 Gopher。之後「每個參數約 20 個 token」就成了 **Chinchilla 最佳**（compute-optimal）的經驗法則。

兩份研究結論不同，一個主要原因是 Kaplan 等人的學習率排程沒有隨訓練長度調整：cosine 排程要在訓練結束時剛好降到最低，否則短的訓練沒降完、看起來比較差，偏向了「少資料」的結論。這呼應第 5 章第 4 集：排程的長度要配合總步數。

## 重點整理

- 預算 \\( C \approx 6ND \\) 固定時，模型太小或太大都不好，IsoFLOPs 曲線是 U 形。
- Chinchilla 公式：\\( L = E + A/N^\alpha + B/D^\beta \\)，`llmz.scaling.chinchilla_loss`。
- 最佳分配：Chinchilla 論文的前兩種方法得到參數量與資料量約各隨 \\( C^{0.5} \\) 成長、每個參數約 20 個 token；公式本身在 \\( 10^{21} \\) 時給出 50 個。
- Kaplan 的 \\( C^{0.73} \\) 偏向大模型，部分原因是學習率排程沒配合訓練長度。
