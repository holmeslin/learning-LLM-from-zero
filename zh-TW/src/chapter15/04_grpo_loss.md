# GRPO loss 實作與測試

## 本集目標

把 GRPO 的 clipped loss 寫成函式：token 層級的比值、clip、可選的長度正規化，再用測試確認「比值超出 clip 範圍時梯度是 0」。

## 正文

### 從 `reinforce_loss` 到 GRPO 的 loss

第 1 集的 `reinforce_loss` 是 \\( -A \log \pi(a) \\) 的平均，一個樣本一個動作。語言模型的一條回答有很多個 token，GRPO 的 loss 有三處不一樣：

1. **逐 token**：回答的機率是每個 token 的條件機率相乘，所以 log 機率是每個 token 的 log 機率相加。loss 在每個 token 上各算一項，同一條回答的 token 共用同一個 advantage（整條答對，每個 token 都被推高）。
2. **比值**：一批 rollout 會連續更新好幾次（下一集 `epochs=2`），第二次更新時，模型已經不是當初取樣的那個模型了。用「取樣時的模型」\\( \pi_{\text{old}} \\) 與「現在的模型」\\( \pi_\theta \\) 的比值 \\( \rho_t = \pi_\theta(o_t) / \pi_{\text{old}}(o_t) \\) 來修正。它對 log 機率的梯度是 \\( \nabla \rho = \rho \nabla \log \pi_\theta \\)，剛取樣完（\\( \rho = 1 \\)）時，和 REINFORCE 的 \\( A \nabla \log \pi \\) 一模一樣。
3. **clip**：比值離 1 太遠，就不再往同一個方向推，免得一批資料把模型推得太遠。

合起來（DeepSeekMath 論文的 GRPO，這裡省略它再加上去的 KL 懲罰項，也就是第 14 章第 5 集 RLHF 目標裡 \\( \beta \\) 倍的 KL（policy 與 reference 模型的距離）；\\( G \\) 條回答，第 \\( i \\) 條有 \\( |o_i| \\) 個 token）：

\\[ \mathcal{L} = -\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\big( \rho_{i,t} A_i,\ \text{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon) A_i \big) \\]

\\( \text{clip}(\rho, 1-\varepsilon, 1+\varepsilon) \\) 把 \\( \rho \\) 夾在 \\( [1-\varepsilon, 1+\varepsilon] \\) 之間，\\( \varepsilon \\) 預設 0.2。

### 每個 token 的 log 機率

比值要一個 token 一個 token 算。第 12 章的 `sequence_logprob` 把 token 的 log 機率加總成一個數字，這裡不加總：

```python,ignore
{{#include ../../../src/llmz/rl.py:token_logprobs}}
```

位置 \\( t \\) 的輸出預測第 \\( t+1 \\) 個 token，所以 `ids` 有 \\( T \\) 個 token、結果只有 \\( T-1 \\) 格；`mask[:, 1:]` 才和它對齊（第 0 個 token 沒有前文，永遠不算，和 `sequence_logprob` 一樣）。用隨機初始化的小模型驗證：加總之後和 `sequence_logprob` 一致。

```python
import torch

from llmz.evaluation import sequence_logprob
from llmz.model import ModelConfig, TransformerLM
from llmz.rl import token_logprobs

torch.manual_seed(0)
model = TransformerLM(ModelConfig(vocab_size=50, context_length=16, d_model=16, n_layers=1, n_heads=2, d_ff=32))
ids = torch.randint(0, 50, (3, 6))
mask = torch.tensor([[False, False, False, True, True, True]] * 3)  # 第 3、4、5 個 token 是「回答」

logp = token_logprobs(model, ids)
print(logp.shape)
print(torch.allclose(logp[:, 2:].sum(dim=1), sequence_logprob(model, ids, mask), atol=1e-5))  # token t 的 log 機率在第 t-1 格
```

執行結果：

```text
torch.Size([3, 5])
True
```

### clipped loss

```python,ignore
{{#include ../../../src/llmz/rl.py:grpo_loss}}
```

重點在三行：

- `ratio`：`logp - old_logp` 是 log 的比值，`exp` 之後就是 \\( \rho \\)。`old_logp` 是 rollout 之後、第一次更新之前，用同一個模型和同一個函式算一次（不帶梯度）。這樣第一次更新時 `ratio` 恰好是 1，不會因為取樣時和訓練時的算法略有不同，就一開始比值不是 1。
- `objective`：兩項取 `torch.minimum`，沒有 clip 的 \\( \rho A \\) 與夾過的。
- `per_answer`：只加總 `mask` 為 True 的 token；`length_norm=True` 再除以這條回答的 token 數，最後對 \\( N \\) 條回答平均。

### clip 到底擋掉了什麼

`min` 取比較小的那一項。advantage 為正時，我們希望 \\( \rho \\) 變大；為負時，希望它變小。用一個 token、`old_logp = 0`（所以 \\( \rho = e^{\text{logp}} \\)）、\\( \varepsilon = 0.2 \\)，對幾個比值看 loss 和梯度（loss 對 `logp` 的梯度；負的梯度會讓梯度下降把 `logp` 推高）：

```python
import math

import torch

from llmz.rl import grpo_loss


def loss_and_grad(ratio, advantage, eps=0.2):
    logp = torch.tensor([[math.log(ratio)]], requires_grad=True)
    loss = grpo_loss(logp, torch.zeros(1, 1), torch.tensor([advantage]), torch.ones(1, 1, dtype=torch.bool), clip_eps=eps)
    loss.backward()
    return loss.item(), logp.grad.item() + 0.0  # + 0.0：避免印出 -0.00


print("ratio   A=+1 的 loss 與梯度   A=-1 的 loss 與梯度")
for ratio in [0.5, 0.9, 1.0, 1.1, 1.5]:
    up, down = loss_and_grad(ratio, +1.0), loss_and_grad(ratio, -1.0)
    print(f"{ratio:.1f}      {up[0]:+.2f}  {up[1]:+.2f}           {down[0]:+.2f}  {down[1]:+.2f}")
```

執行結果：

```text
ratio   A=+1 的 loss 與梯度   A=-1 的 loss 與梯度
0.5      -0.50  -0.50           +0.80  +0.00
0.9      -0.90  -0.90           +0.90  +0.90
1.0      -1.00  -1.00           +1.00  +1.00
1.1      -1.10  -1.10           +1.10  +1.10
1.5      -1.20  +0.00           +1.50  +1.50
```

- **範圍內**（0.9、1.0、1.1）：梯度是 \\( -\rho A \\)，和沒有 clip 的 policy gradient 一樣。
- **\\( A = +1 \\)、\\( \rho = 1.5 \\)**：這個 token 的機率已經比取樣時高了 50%，超過 \\( 1+\varepsilon \\)，夾過的那項比較小，`min` 取它；它是常數，**梯度是 0**，不再繼續推高。
- **\\( A = -1 \\)、\\( \rho = 0.5 \\)**：機率已經降到一半，超過 \\( 1-\varepsilon \\)，同樣**梯度是 0**，不再繼續壓低。
- **方向反了的兩格**（\\( A=+1 \\)、\\( \rho = 0.5 \\)；\\( A=-1 \\)、\\( \rho = 1.5 \\)）：這個 token 本來該推高，機率卻掉下去了（或本來該壓低，機率卻升上去），`min` 取沒有 clip 的那項，梯度照給，把它拉回來。

所以 clip 不是「把比值夾住」，是「**往 advantage 想要的方向已經走夠遠，就關掉那一側的梯度**」；走錯方向的，永遠修正得到。

### 長度正規化與 mask

`length_norm=True` 時，每條回答的 token 先平均。兩條回答、同樣的 advantage 1.0，一條有 3 個 token、一條只有 1 個，看每個 token 的梯度：

```python
import torch

from llmz.rl import grpo_loss

mask = torch.tensor([[True, True, True], [True, False, False]])  # 第一條 3 個 token，第二條 1 個（後面是補的）
for length_norm in [True, False]:
    logp = torch.zeros(2, 3, requires_grad=True)
    grpo_loss(logp, torch.zeros(2, 3), torch.ones(2), mask, length_norm=length_norm).backward()
    print(f"length_norm={length_norm}")
    for row in logp.grad:
        print("  ", " ".join(f"{g:+.3f}" for g in row.tolist()))
```

執行結果：

```text
length_norm=True
   -0.167 -0.167 -0.167
   -0.500 +0.000 +0.000
length_norm=False
   -0.500 -0.500 -0.500
   -0.500 +0.000 +0.000
```

`length_norm=True`：每條回答的權重一樣（都是 1/2），長的那條每個 token 只分到 1/6，短的那條的唯一一個 token 分到 1/2。`length_norm=False`：每個 token 的權重一樣（1/2），長的那條總權重是短的 3 倍。哪個比較合理，第 6 集（Dr. GRPO）會回頭討論，現在先把它做成參數，可以切換。

補的位置（`mask` 為 False）的梯度是 0。為了讓它們連 `nan` 都不會出現，`ratio` 的那一行先用 `torch.where` 把被遮掉的位置換成 0，再做 `exp`。沒有這一步會怎樣：補的位置模型的 log 機率可能和 `old_logp` 差很多，`exp` 溢位成 `inf`，乘上 mask 的 0 變成 `nan`，而且 `nan` 的梯度會傳回整個模型：

```python
import torch

logp = torch.tensor([[0.0, 100.0]], requires_grad=True)  # 第二個位置被遮掉，但 exp(100) 在 float32 溢位成 inf
mask = torch.tensor([[True, False]])
(logp.exp() * mask).sum().backward()
print(logp.grad)
```

執行結果：

```text
tensor([[1., nan]])
```

### 用測試鎖住

`tests/test_rl.py` 把上面的每一條寫成測試。最重要的一條是「比值超出 clip 範圍、而且 clip 那一項比較小的時候，梯度是 0」（四種組合都測，也測了 `clip_eps` 放寬之後梯度會回來）：

```python
import math

import torch

from llmz.rl import grpo_loss


def grad_at(ratio, advantage, eps=0.2):
    logp = torch.tensor([[math.log(ratio)]], requires_grad=True)
    grpo_loss(logp, torch.zeros(1, 1), torch.tensor([advantage]), torch.ones(1, 1, dtype=torch.bool), clip_eps=eps).backward()
    return logp.grad.item()


for ratio, advantage in [(1.5, +1.0), (0.5, -1.0), (3.0, +2.0), (0.1, -3.0)]:
    assert grad_at(ratio, advantage) == 0.0  # 走過頭的一側：梯度是 0
for ratio, advantage in [(0.5, +1.0), (1.5, -1.0)]:
    assert math.isclose(grad_at(ratio, advantage), -ratio * advantage, rel_tol=1e-5)  # 走錯方向：照給
assert grad_at(1.5, +1.0, eps=0.6) != 0.0  # 範圍放寬到 [0.4, 1.6]，1.5 在裡面
print("通過")
```

執行結果：

```text
通過
```

```bash
uv run pytest tests/test_rl.py -k "clip or length_norm or masked or logprobs"
```

其餘的測試：`ratio = 1` 時 loss 是 \\( -\text{mean}(A) \\)；整組 advantage 都是 0 時梯度全是 0（呼應上一集：全對或全錯的組不提供梯度）；`length_norm` 的兩種手算結果；被遮掉的位置即使 `logp` 是 100 也沒有 `nan`。

## 重點整理

- GRPO 的 loss 是逐 token 的：每個 token 算 \\( \rho = \pi_\theta / \pi_{\text{old}} \\)，取 \\( \min(\rho A,\ \text{clip}(\rho, 1\pm\varepsilon) A) \\)，同一條回答的 token 共用同一個 advantage。`token_logprobs` 取每個 token 的 log 機率，`grpo_loss` 算 loss（省略原論文的 KL 懲罰項）。
- \\( \rho = 1 \\) 時梯度和 REINFORCE 一樣；比值在 \\( [1-\varepsilon, 1+\varepsilon] \\) 內時梯度是 \\( -\rho A \\)。
- clip 關掉的是「往 advantage 想要的方向已經走過頭」那一側的梯度（\\( A>0 \\) 且 \\( \rho>1+\varepsilon \\)、\\( A<0 \\) 且 \\( \rho<1-\varepsilon \\)）；走錯方向的永遠有梯度。
- `length_norm=True`：每條回答先對自己的 token 平均；`False`：每個 token 同權重。`mask` 把提示、補的位置遮掉，遮掉的位置先歸零再 `exp`，避免 `inf * 0 = nan`。
- `old_logp` 用同一個函式在更新前算一次，不帶梯度；`tests/test_rl.py` 驗證比值超出範圍時梯度為 0。
