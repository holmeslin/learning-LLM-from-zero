# 擬合 IsoFLOPs 曲線（離線資料）

## 本集目標

拿到一組「每個預算訓練了幾個不同大小的模型」的資料，對每個預算擬合一條拋物線，找出最佳的模型大小。

## 正文

### 離線資料

真正做 scaling law 研究，資料來自幾十到幾百次訓練。我們先用 Chinchilla 的公式代替訓練，產生一組離線資料：5 個預算（\\( 10^{18} \\) 到 \\( 10^{22} \\) FLOPs），每個預算 9 個模型大小，loss 加上一點隨機雜訊，模擬真實實驗的誤差。

```python
import numpy as np

from llmz.scaling import chinchilla_loss

rng = np.random.default_rng(0)
budgets = [1e18, 1e19, 1e20, 1e21, 1e22]
runs = []  # (預算, 參數量, loss)
for c in budgets:
    center = (c / 120) ** 0.5  # 每個參數 20 個 token 時的大小，當作掃描的中心
    for n in center * np.logspace(-1, 1, 9):
        loss = chinchilla_loss(n, c / (6 * n)) + rng.normal(0, 0.01)
        runs.append((c, n, float(loss)))

for c, n, loss in runs[:9]:
    print(f"C = {c:.0e}  N = {n:.2e}  loss = {loss:.4f}")
```

執行結果：

```text
C = 1e+18  N = 9.13e+06  loss = 3.9887
C = 1e+18  N = 1.62e+07  loss = 3.7717
C = 1e+18  N = 2.89e+07  loss = 3.6370
C = 1e+18  N = 5.13e+07  loss = 3.5544
C = 1e+18  N = 9.13e+07  loss = 3.5313
C = 1e+18  N = 1.62e+08  loss = 3.5816
C = 1e+18  N = 2.89e+08  loss = 3.6896
C = 1e+18  N = 5.13e+08  loss = 3.8430
C = 1e+18  N = 9.13e+08  loss = 4.0448
```

- `np.random.default_rng(0)` 建立一個固定種子的 numpy 亂數產生器，`rng.normal(0, 0.01)` 抽一個平均 0、標準差 0.01 的雜訊。
- 掃描範圍以「每個參數 20 個 token」為中心，往兩邊各一個數量級（\\( 10^{-1} \\) 到 \\( 10^{1} \\) 倍）。真實研究也是這樣：先猜一個大概的位置，再往兩邊掃。

雜訊讓 loss 不再平滑，直接挑最小的那一點會受雜訊影響，而且只能挑到掃描過的大小。

### 對 log N 擬合拋物線

在最低點附近，IsoFLOPs 曲線對 \\( \log N \\) 很像一條拋物線 \\( y = c_2 x^2 + c_1 x + c_0 \\)。用全部 9 個點擬合拋物線，它的頂點 \\( x = -c_1 / (2 c_2) \\) 就是最佳的 \\( \log N \\)，雜訊被平均掉了，也能落在兩個掃描點之間。

```python,ignore
{{#include ../../../src/llmz/scaling.py:fit_isoflop}}
```

`np.polyfit(x, y, 2)` 用最小平方法擬合二次多項式，回傳 `(c2, c1, c0)`。

```python
import numpy as np

from llmz.scaling import chinchilla_loss, fit_isoflop_minimum

rng = np.random.default_rng(0)
budgets = [1e18, 1e19, 1e20, 1e21, 1e22]
for c in budgets:
    center = (c / 120) ** 0.5
    n = center * np.logspace(-1, 1, 9)
    loss = chinchilla_loss(n, c / (6 * n)) + rng.normal(0, 0.01, size=9)
    best = fit_isoflop_minimum(n, loss)
    grid = np.logspace(6, 12, 5000)
    truth = grid[np.argmin(chinchilla_loss(grid, c / (6 * grid)))]  # 沒有雜訊時的答案，只用來對照
    print(f"C = {c:.0e}  最小的點 {n[np.argmin(loss)]:.2e}  拋物線頂點 {best:.2e}  真正的最佳 {truth:.2e}")
```

執行結果：

```text
C = 1e+18  最小的點 9.13e+07  拋物線頂點 8.33e+07  真正的最佳 8.05e+07
C = 1e+19  最小的點 1.62e+08  拋物線頂點 2.34e+08  真正的最佳 2.28e+08
C = 1e+20  最小的點 9.13e+08  拋物線頂點 6.75e+08  真正的最佳 6.45e+08
C = 1e+21  最小的點 1.62e+09  拋物線頂點 1.80e+09  真正的最佳 1.82e+09
C = 1e+22  最小的點 9.13e+09  拋物線頂點 4.80e+09  真正的最佳 5.16e+09
```

`rng.normal(0, 0.01, size=9)` 一次抽 9 個雜訊。最後一欄是用沒有雜訊的公式在很密的格點上找到的真正答案，實際研究不會有，這裡只拿來對照。直接挑最小的點只能落在 9 個掃描大小之一，在 \\( 10^{22} \\) 時差了將近一倍；拋物線的頂點每個預算都在真正答案的一成以內。每個預算都得到一個最佳參數量，下一集把它們連起來。

### 為什麼是拋物線

拋物線只是在最低點附近的近似。離最低點很遠時曲線並不對稱（第 2 集的表：太小的那一側上升得比較慢）。所以掃描範圍不能太寬，而且最低點要落在範圍中間；如果最低的點在範圍的邊上，代表猜的中心不準，應該往那個方向多掃幾個點。

## 重點整理

- IsoFLOPs 資料：每個預算訓練好幾個大小，記下最後的 loss。
- 在最低點附近對 \\( \log N \\) 擬合拋物線，頂點就是這個預算的最佳參數量，`llmz.scaling.fit_isoflop_minimum`。
- 拋物線用到所有點，比直接挑最小值更不受雜訊影響。
- 掃描範圍要把最低點包在中間。
