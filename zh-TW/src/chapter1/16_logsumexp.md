# 數值穩定：log-sum-exp

## 本集目標

看到 `exp` 溢位變成 `inf` 和 `nan`，學會「先減掉最大值」這一招，寫出穩定的 softmax、log-softmax 與 cross-entropy，並把它們放進 `llmz`。

## 正文

### 分數大一點就爆

第 13 集的 softmax 直接對分數取 exp。真實模型的 logits 動輒幾十、上百：

```python
import torch

scores = torch.tensor([1000.0, 999.0, 998.0])
e = torch.exp(scores)
print(e)
print(e / e.sum())
```

執行結果：

```text
tensor([inf, inf, inf])
tensor([nan, nan, nan])
```

`float32` 最大只能到 \\( 3.4 \times 10^{38} \\)，\\( e^{89} \\) 就超過了，變成 `inf`（無限大）。`inf / inf` 是 `nan`（not a number），一旦出現 nan，之後所有運算都是 nan，訓練就整個壞掉。

### 先減掉最大值

第 13 集的性質二：所有分數同時加減同一個數，softmax 結果不變。那就把每個分數都減掉**最大值**：最大的那個變成 0，其他都是負數，exp 之後全部落在 0 到 1 之間，永遠不會溢位。

```python
import torch

scores = torch.tensor([1000.0, 999.0, 998.0])
shifted = scores - scores.max()
print(shifted)
e = torch.exp(shifted)
print(e / e.sum())
```

執行結果：

```text
tensor([ 0., -1., -2.])
tensor([0.6652, 0.2447, 0.0900])
```

答案和「理論上」直接算的一樣，只是這次算得出來。

### log-sum-exp

上一集說模型內部用 log 機率。softmax 的 log 是：

\\[ \log p_i = s_i - \log \sum_j e^{s_j} \\]

後面那一項 \\( \log \sum_j e^{s_j} \\) 叫 **log-sum-exp**，它有同樣的溢位問題，也用同一招解：把最大值 \\( m \\) 提出來，\\( \log \sum_j e^{s_j} = m + \log \sum_j e^{s_j - m} \\)。

```python
import torch

x = torch.tensor([1000.0, 999.0, 998.0])
m = x.max()
lse = m + torch.log(torch.exp(x - m).sum())
print(lse)
print(torch.logsumexp(x, dim=0))
log_p = x - lse
print(log_p)
print(torch.exp(log_p))
```

執行結果：

```text
tensor(1000.4076)
tensor(1000.4076)
tensor([-0.4076, -1.4076, -2.4076])
tensor([0.6652, 0.2447, 0.0900])
```

PyTorch 內建的 `torch.logsumexp` 算出一樣的數字，這裡只拿它來驗算，不拿來用。`x - lse` 就是 log 機率，exp 回去和上一段的 softmax 一致。

### 放進 `llmz`

把這一章學到的東西整理成函式，放在 `src/llmz/functional.py`。全部都支援 `dim`，用第 5 集的 `keepdim` 讓多列一起算：

```python,ignore
{{#include ../../../src/llmz/functional.py:softmax}}
```

這裡的 `x.max(dim=dim, keepdim=True)` 和前面的 `scores.max()` 差在多了 `dim`：有 `dim` 時它會回傳兩樣東西，每一列的最大值和那個最大值的位置，所以要用 `.values` 只拿最大值。`keepdim=True` 是第 5 集的老朋友，讓形狀對得上後面的 broadcasting。

```python,ignore
{{#include ../../../src/llmz/functional.py:logsumexp}}
```

```python,ignore
{{#include ../../../src/llmz/functional.py:log_softmax}}
```

`log_softmax` 沒有寫成 `x - logsumexp(x)`，而是先減最大值、再對減過的數字做所有運算。原因是 `float32` 在 1000 附近只能分辨到大約 0.00006，`1000 - 1000.4076` 這一步會損失精度；先減掉最大值，之後的數字都很小，就沒這個問題。這是寫測試時發現的，數值問題常常這樣藏在細節裡。

```python,ignore
{{#include ../../../src/llmz/functional.py:cross_entropy}}
```

```python,ignore
{{#include ../../../src/llmz/functional.py:perplexity}}
```

`cross_entropy` 就是上一集的流程，只是 softmax 換成了穩定版的 `log_softmax`。用起來像這樣：

```python
import torch

from llmz.functional import cross_entropy, log_softmax, perplexity, softmax

logits = torch.tensor([[1000.0, 999.0, 998.0],
                       [0.5, 2.5, 0.0]])
targets = torch.tensor([0, 1])
print(softmax(logits))
print(log_softmax(logits))
loss = cross_entropy(logits, targets)
print(loss)
print(perplexity(loss))
```

執行結果：

```text
tensor([[0.6652, 0.2447, 0.0900],
        [0.1112, 0.8214, 0.0674]])
tensor([[-0.4076, -1.4076, -2.4076],
        [-2.1967, -0.1967, -2.6967]])
tensor(0.3022)
tensor(1.3528)
```

第一列的分數上千也沒事了。這幾個函式在 `tests/test_functional.py` 裡有測試，用 `torch.softmax`、`torch.logsumexp` 與 `torch.nn.functional.cross_entropy` 比對結果：

```bash
uv run pytest tests/test_functional.py
```

第 4 章的模型會輸出 logits，第 5 章的訓練迴圈會用 `cross_entropy` 算 loss，就是這裡的這兩個函式。

## 重點整理

- `exp` 很容易溢位成 `inf`，接著變 `nan`，毒死整個計算。
- softmax 與 log-sum-exp 都先減掉最大值再算，結果不變、不會溢位：\\( \log \sum e^{x} = m + \log \sum e^{x - m} \\)。
- `log_softmax(x) = (x - m) - \log \sum e^{x - m}`，減過之後的每一步都在小數字上運算，精度也不會掉。
- `llmz.functional` 的 `softmax`、`logsumexp`、`log_softmax`、`cross_entropy`、`perplexity` 是本書之後一直會用的版本。

## 自我挑戰

這一章放進 `src/llmz/` 的函式有：

- `src/llmz/device.py`：`get_device`
- `src/llmz/functional.py`：`softmax`、`logsumexp`、`log_softmax`、`cross_entropy`、`perplexity`

把每個函式的本體（`def` 那一行以下、docstring 以外）全部刪掉，換成一行 `raise NotImplementedError`，然後執行：

```bash
uv run pytest tests/test_functional.py tests/test_device.py
```

看著測試一個一個失敗，再憑這一章學到的東西把它們寫回來，直到全部通過。寫不出來時回頭翻對應的那一集；真的卡住，`git checkout src/llmz/functional.py` 可以把原版拿回來。
