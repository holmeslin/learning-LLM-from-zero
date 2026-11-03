# warmup + cosine 學習率排程

## 本集目標

寫出 `lr_cosine_schedule`：訓練開始時學習率從 0 線性升到最大值（warmup），然後沿 cosine 曲線降到最小值。

## 正文

### 為什麼不固定

- **開頭要小**：第 3 集的 \\( m \\)、\\( v \\) 從 0 開始，前幾十步的估計很不準；而且剛初始化的模型梯度方向亂，大步走容易衝進壞的區域。所以前面幾百步讓學習率慢慢升上去，叫 **warmup**。
- **結尾要小**：接近好的解時，大步會在附近來回跳、停不下來，把學習率降下來才能走進去。

### cosine

warmup 之後用半個 cosine 曲線從 \\( \eta_{\max} \\) 平滑降到 \\( \eta_{\min} \\)：

\\[ \eta(t) = \eta_{\min} + \tfrac{1}{2}\Big(1 + \cos\big(\pi \cdot \tfrac{t - T_w}{T_c - T_w}\big)\Big)(\eta_{\max} - \eta_{\min}) \\]

\\( T_w \\) 是 warmup 步數、\\( T_c \\) 是 cosine 結束的步數（本書設成總步數）。\\( t = T_w \\) 時 cos 是 1，\\( \eta = \eta_{\max} \\)；\\( t = T_c \\) 時 cos 是 -1，\\( \eta = \eta_{\min} \\)。

```python,ignore
{{#include ../../../src/llmz/optim.py:lr_cosine_schedule}}
```

三段：warmup 線性、cosine、之後維持 \\( \eta_{\min} \\)。

```python
from llmz.optim import lr_cosine_schedule

for step in [0, 100, 200, 500, 1000, 2600, 4000, 5000, 6000]:
    lr = lr_cosine_schedule(step, lr_max=1e-3, lr_min=1e-4, warmup_steps=200, cosine_steps=5000)
    print(f"{step:5d} {lr:.6f}")
```

執行結果：

```text
    0 0.000000
  100 0.000500
  200 0.001000
  500 0.000991
 1000 0.000940
 2600 0.000550
 4000 0.000193
 5000 0.000100
 6000 0.000100
```

本書的標準設定：\\( \eta_{\max} = 10^{-3} \\)、\\( \eta_{\min} = 10^{-4} \\)、warmup 200 步、總共 5000 步。第 200 步到最大值，第 2600 步（cosine 的中點）剛好是兩者的平均 0.00055，第 5000 步到最小值。

### 怎麼接上優化器

第 2 集說過 `param_groups` 裡有 `lr`。每一步算出新的學習率，寫進去就好：

```python,ignore
lr = lr_cosine_schedule(step, ...)
for group in optimizer.param_groups:
    group["lr"] = lr
```

第 8 集的訓練迴圈就是這樣做。

## 重點整理

- warmup：前幾百步線性升到 \\( \eta_{\max} \\)，讓 Adam 的統計穩下來、初期不亂走。
- cosine：之後平滑降到 \\( \eta_{\min} \\)，結尾才走得進好的解。
- `lr_cosine_schedule(step, lr_max, lr_min, warmup_steps, cosine_steps)`；標準設定 \\( 10^{-3} \to 10^{-4} \\)、warmup 200、共 5000 步。
- 每步把算出的值寫進 `optimizer.param_groups[i]["lr"]`。
