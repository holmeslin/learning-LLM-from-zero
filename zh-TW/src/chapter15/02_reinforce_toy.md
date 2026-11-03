# REINFORCE 玩具版

## 本集目標

用上一集的梯度估計寫出完整的 REINFORCE 訓練迴圈，看到它在吃角子老虎機上收斂；再看為什麼要減掉 baseline：它不改變梯度的平均方向，卻讓估計安靜很多。

## 正文

### 完整的 REINFORCE

上一集的 `estimate_gradient` 抽樣、算出對 \\( J \\) 的梯度估計。把它放進迴圈，每一步往梯度方向走一小步，就是 REINFORCE（Williams 在 1992 年提出的演算法）：

```python,ignore
{{#include ../../../src/llmz/rl.py:train_bandit}}
```

和第 2 章第 2 集的手動梯度下降相比，只有兩個差別：方向相反（`+` 而不是 `-`，因為我們要讓 \\( J \\) 變大），以及梯度是抽樣估出來的、不是精確的。`lr` 是學習率，`n` 是每一步抽幾個樣本。`train_bandit` 每次從同一個種子重新開始，所以 `steps` 設不同值得到的是同一條訓練軌跡在不同時間點的狀態。

把獎勵加上標準差 1.0 的雜訊（機器 3 平均 5.0、機器 4 平均 4.0，單次拉的結果常常反過來），看 policy 會不會學到最好的機器：

```python
import torch

from llmz.functional import softmax
from llmz.rl import expected_reward, train_bandit

means = torch.tensor([0.0, 3.0, 2.0, 5.0, 4.0])  # 最好的是編號 3，平均 5.0
print("步數  編號 3 的機率  平均獎勵 J")
for steps in [0, 25, 50, 100, 200, 400]:
    logits = train_bandit(means, steps=steps, n=16, lr=0.1, baseline=True, noise_std=1.0, seed=0)
    print(f"{steps:4d}  {softmax(logits)[3].item():.3f}        {expected_reward(logits, means).item():.2f}")
```

執行結果：

```text
步數  編號 3 的機率  平均獎勵 J
   0  0.200        2.80
  25  0.522        3.94
  50  0.800        4.59
 100  0.928        4.85
 200  0.972        4.94
 400  0.988        4.98
```

走 200 步，policy 有 97% 的機率拉編號 3；平均獎勵從 2.8 升到 4.94，接近理論上限 5.0。這就是「只靠抽樣和分數」學到的 policy。

程式裡的 `baseline=True` 是這一集的主角。

### 為什麼要 baseline

上一集的估計是 \\( r \cdot \nabla \log \pi(a) \\)，獎勵本身當作推的力道。想像獎勵全部加 10（0 分變 10 分，5 分變 15 分）：每台機器的好壞順序完全沒變，可是現在每一個被抽到的動作都被用力往上推，只是有些推得稍微更用力。真正有用的訊息「這台比平均好多少」，被一個大常數蓋住了。

解法是讓力道變成「比平均好多少」：獎勵減掉一個 baseline \\( b \\)。

\\[ \nabla J = \mathbb{E}\big[ (R - b)\\, \nabla \log \pi(a) \big] \\]

只要 \\( b \\) 不依賴被抽到的動作，這個式子的平均值和沒減時一樣，因為 \\( \mathbb{E}[\\, b \cdot \nabla \log \pi(a)\\,] = b \sum_a \pi(a) \nabla \log \pi(a) = b \sum_a \nabla \pi(a) = b\\, \nabla 1 = 0 \\)：機率加總永遠是 1，微分是 0。但單次抽樣的結果會穩定很多，因為力道不再被大常數撐大。

`estimate_gradient` 的 `baseline=True` 用的是「這一批獎勵的平均」。為了量，我們固定 policy 在初始狀態（5 台機率一樣），反覆估計 1000 次梯度，看每個分量的變異（variance，`.var()`，也就是標準差 `.std()` 的平方）加總，以及 1000 次的平均：

```python
import torch

from llmz.rl import estimate_gradient, expected_reward


def show(t):
    return " ".join(f"{x:+.2f}" for x in t.tolist())


means = torch.tensor([0.0, 3.0, 2.0, 5.0, 4.0])
logits = torch.zeros(5)
exact = torch.zeros(5, requires_grad=True)
expected_reward(exact, means).backward()
print("精確梯度            ", show(exact.grad))
print("精確梯度 x 15/16    ", show(exact.grad * 15 / 16))

for offset in [0.0, 10.0]:
    for baseline in [False, True]:
        generator = torch.Generator().manual_seed(0)
        runs = torch.stack([estimate_gradient(logits, means + offset, 16, baseline, 1.0, generator) for _ in range(1000)])
        print(f"獎勵加 {offset:4.1f}，baseline={str(baseline):5s}  變異加總 {runs.var(dim=0).sum().item():5.2f}  平均 {show(runs.mean(dim=0))}")
```

執行結果：

```text
精確梯度             -0.56 +0.04 -0.16 +0.44 +0.24
精確梯度 x 15/16     -0.53 +0.04 -0.15 +0.41 +0.23
獎勵加  0.0，baseline=False  變異加總  0.55  平均 -0.56 +0.04 -0.16 +0.42 +0.26
獎勵加  0.0，baseline=True   變異加總  0.16  平均 -0.52 +0.04 -0.15 +0.41 +0.23
獎勵加 10.0，baseline=False  變異加總  8.34  平均 -0.57 +0.03 -0.17 +0.40 +0.32
獎勵加 10.0，baseline=True   變異加總  0.16  平均 -0.52 +0.04 -0.15 +0.41 +0.23
```

- 獎勵沒有加常數時，baseline 把變異從 0.55 降到 0.16，不到三分之一。
- 獎勵加 10 後，沒有 baseline 的變異暴增到 8.34，是原來的 15 倍；有 baseline 的結果和沒加 10 時**一模一樣**，因為 `r - r.mean()` 把加進去的常數整個消掉了。
- 四種情況的平均都接近精確梯度，baseline 沒有改變方向。有 baseline 的平均小一點，這是因為批次平均把自己也算進去了：16 個樣本的平均裡，被抽到的那個樣本佔了 1/16，所以估計的平均是精確梯度的 15/16 倍（上面第二行），方向沒變，只是縮小一點。

### 變異大會怎樣

變異大不只是數字難看。每一步的方向抖得厲害，policy 可能被早期的幸運樣本帶偏。同樣是獎勵加 10、學習率 0.1，各跑 20 個種子（`seed` 0 到 19），每次 200 步，看最後選到編號 3 的機率：

```python
import torch

from llmz.functional import softmax
from llmz.rl import train_bandit

means = torch.tensor([0.0, 3.0, 2.0, 5.0, 4.0]) + 10.0
for baseline in [False, True]:
    best = [softmax(train_bandit(means, steps=200, n=16, lr=0.1, baseline=baseline, noise_std=1.0, seed=seed))[3].item() for seed in range(20)]
    stuck = sum(p < 0.5 for p in best)
    print(f"baseline={str(baseline):5s}  最低 {min(best):.2f}  最高 {max(best):.2f}  編號 3 的機率小於 0.5 的種子：{stuck} 個")
```

執行結果：

```text
baseline=False  最低 0.19  最高 0.99  編號 3 的機率小於 0.5 的種子：2 個
baseline=True   最低 0.96  最高 0.98  編號 3 的機率小於 0.5 的種子：0 個
```

沒有 baseline 時，有些種子學壞了（最低只有 0.19）：獎勵全是正的，早期隨機拉到哪台，哪台就被往上推，推到機率很高之後，更常被抽到、更常被推，而編號 3 與編號 4 的平均只差 1，雜訊就有 1.0，一時的運氣足夠把 policy 帶到錯的地方。有 baseline 時，20 個種子的結果落在很窄的範圍。這個失敗率不穩定（種子換掉，數字就變），這裡的數字只代表這 20 個種子。

### baseline 的選擇

這一集用批次平均，簡單、不用額外的模型。也可以用別的 baseline，例如單獨學一個預測「這個情況平均能拿多少分」的模型（PPO 的 value 網路就是這個作法），或是只用「其他樣本」的平均（練習題）。下一集看 GRPO 怎麼選。

## 重點整理

- REINFORCE 的迴圈：抽樣、拿獎勵、算梯度估計、往上爬一步。`train_bandit` 在 5 台機器、獎勵有雜訊的設定下，200 步內讓最好那台的機率到 97%。
- 獎勵減掉一個與動作無關的 baseline，梯度的平均不變（\\( \sum_a \nabla \pi(a) = 0 \\)），變異卻可以大幅下降：獎勵加 10 時，批次平均 baseline 讓變異從 8.34 降到 0.16，而且結果與有沒有加 10 無關。
- 批次平均把自己也算進去，估計的平均小了 \\( (n-1)/n \\) 倍，方向不變。
- 沒有 baseline 時，獎勵全是正的，早期的運氣可能把 policy 帶去別台機器；沒有 baseline 的 20 個種子中有 2 個最後選編號 3 的機率小於 0.5（最低 0.19），有 baseline 的 20 個種子都在 0.96 以上；種子換掉數字會變，這只是這 20 個種子的結果。
