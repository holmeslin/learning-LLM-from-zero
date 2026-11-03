# 擬合冪律

## 本集目標

把每個預算的最佳參數量連起來，擬合出 \\( N_{\text{opt}} \propto C^a \\)，並外推到更大的預算。

## 正文

### log-log 上的直線

第 1 集說過，冪律在 log-log 座標上是直線。所以擬合 \\( y = k x^a \\) 就是對 \\( (\log x, \log y) \\) 擬合一條直線：

```python,ignore
{{#include ../../../src/llmz/scaling.py:fit_power_law}}
```

`np.polyfit(..., 1)` 擬合一次多項式（直線），斜率就是指數 \\( a \\)，截距是 \\( \log k \\)。

### 從上一集的結果出發

```python
import numpy as np

from llmz.scaling import chinchilla_loss, fit_isoflop_minimum, fit_power_law

rng = np.random.default_rng(0)
budgets = np.array([1e18, 1e19, 1e20, 1e21, 1e22])
best_n = []
for c in budgets:
    n = (c / 120) ** 0.5 * np.logspace(-1, 1, 9)
    loss = chinchilla_loss(n, c / (6 * n)) + rng.normal(0, 0.01, size=9)
    best_n.append(fit_isoflop_minimum(n, loss))
best_n = np.array(best_n)
best_d = budgets / (6 * best_n)

k_n, a_n = fit_power_law(budgets, best_n)
k_d, a_d = fit_power_law(budgets, best_d)
print(f"N_opt = {k_n:.3g} × C^{a_n:.3f}")
print(f"D_opt = {k_d:.3g} × C^{a_d:.3f}")
print(f"兩個指數相加 {a_n + a_d:.3f}")

for c in [1e23, 1e24]:
    n, d = k_n * c**a_n, k_d * c**a_d
    print(f"C = {c:.0e}：N = {n:.2e}，D = {d:.2e}，每個參數 {d / n:.0f} 個 token，預測 loss {chinchilla_loss(n, d):.3f}")
```

執行結果：

```text
N_opt = 0.995 × C^0.441
D_opt = 0.167 × C^0.559
兩個指數相加 1.000
C = 1e+23：N = 1.36e+10，D = 1.23e+12，每個參數 90 個 token，預測 loss 2.005
C = 1e+24：N = 3.75e+10，D = 4.44e+12，每個參數 118 個 token，預測 loss 1.911
```

擬合出來的指數是 0.441 與 0.559。用公式推導的話，Chinchilla 公式的最佳分配是 \\( N_{\text{opt}} \propto C^{\beta/(\alpha+\beta)} \\)、\\( D_{\text{opt}} \propto C^{\alpha/(\alpha+\beta)} \\)，代入 \\( \alpha = 0.34 \\)、\\( \beta = 0.28 \\) 是 0.452 與 0.548。我們從 45 個含雜訊的點，擬合出只差 0.011 的指數。（`tests/test_scaling.py` 驗證了這個推導。）

兩個指數相加一定是 1：因為 \\( D_{\text{opt}} = C / (6 N_{\text{opt}}) \\)，\\( N \\) 長得慢，\\( D \\) 就要長得快來補。

### 外推

有了冪律，就能回答「預算是 \\( 10^{24} \\) FLOPs 時模型該多大」，不用真的去試。這是 scaling law 最有價值的地方，也是最危險的地方：我們的資料只到 \\( 10^{22} \\)，往外推兩個數量級，指數的一點點誤差會被放大。例如指數差 0.02，外推兩個數量級就差了 \\( 10^{0.04} \approx 1.1 \\) 倍；從 \\( 10^{18} \\) 推到 \\( 10^{24} \\)（六個數量級）就是 1.3 倍。

### 實務上的流程

1. 選 4 到 6 個預算，每個掃 5 到 10 個大小，用最小的計算量把 IsoFLOPs 曲線畫出來。
2. 每個預算擬合拋物線，得到最佳參數量。
3. 擬合 \\( N_{\text{opt}} \\) 與 \\( D_{\text{opt}} \\) 對 \\( C \\) 的冪律，外推到真正的預算。
4. 也可以擬合每個預算最低的 loss 對 \\( C \\) 的關係，預測大模型最後的 loss。

下一集用我們自己的小模型，在 Mac 上真的做一次。

## 重點整理

- 冪律在 log-log 上是直線，`llmz.scaling.fit_power_law(x, y)` 用一次多項式擬合，回傳 `(k, a)`。
- 從離線資料擬合出 \\( N_{\text{opt}} \propto C^{0.441} \\)、\\( D_{\text{opt}} \propto C^{0.559} \\)，和公式推導的 0.452、0.548 只差 0.011。
- 兩個指數相加是 1，因為 \\( C = 6ND \\)。
- 外推越遠，指數的小誤差放大越多。
