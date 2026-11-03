# policy gradient 直覺

## 本集目標

用一個只有 5 台機器的吃角子老虎機，看懂 policy gradient 的核心想法：不知道每台機器的平均獎勵，也能靠「抽樣、看獎勵、調整機率」讓 policy 越來越好。

## 正文

### 為什麼需要 RL

SFT 是「給你題目和標準答案，一個 token 一個 token 學」。可是有些任務只能判斷「對不對」：數學題的最終答案是不是 42、寫出來的程式能不能通過測試。標準答案可能有很多種寫法，我們也不想全部列出來。

這時換一個做法：讓模型自己寫一個答案，再由一個函式打分數，這個分數叫**獎勵**（reward）。答案拿到高分，就讓模型以後更常寫出這類答案。這就是強化學習（reinforcement learning，RL）。

RL 有幾個固定的名詞，先對應到語言模型上：

- **policy**（策略）：根據目前的情況決定要做什麼的規則。語言模型就是 policy：prompt 是情況，它輸出的整段回答是動作。
- **reward**（獎勵）：給一個動作打的分數，例如答對 1 分、答錯 0 分。
- 目標：調整 policy，讓**平均獎勵**越高越好。

語言模型太大，不適合第一次看這些想法。我們換成最小的 policy。

### 多臂吃角子老虎機

有 5 台機器（編號 0 到 4），每台拉一次會吐出一些獎勵，平均值各不相同，但你一開始不知道。你只有拉的機會。這個問題叫 multi-armed bandit。我們的 policy 是 5 個 logits 經過 softmax 變成機率：這一次拉哪台，照這個機率抽。

```python
import torch

from llmz.functional import softmax
from llmz.rl import bandit_rewards, expected_reward, sample_actions

means = torch.tensor([0.0, 3.0, 2.0, 5.0, 4.0])  # 5 台機器各自的平均獎勵，policy 不知道
logits = torch.zeros(5)  # 一開始每台的機率都一樣
print("機率", [round(p, 2) for p in softmax(logits).tolist()])
print("平均每次拿到", round(expected_reward(logits, means).item(), 2))

generator = torch.Generator().manual_seed(0)
actions = sample_actions(logits, 8, generator)
rewards = bandit_rewards(means, actions, noise_std=1.0, generator=generator)
print("拉了哪幾台", actions.tolist())
print("拿到的獎勵", [round(r, 2) for r in rewards.tolist()])
```

執行結果：

```text
機率 [0.2, 0.2, 0.2, 0.2, 0.2]
平均每次拿到 2.8
拉了哪幾台 [4, 3, 2, 4, 3, 3, 0, 1]
拿到的獎勵 [3.28, 4.6, 1.4, 4.18, 4.14, 6.1, -1.07, 3.12]
```

`torch.Generator().manual_seed(0)` 是一個自己帶著種子的隨機數來源（第 1 章第 10 集的 `torch.manual_seed(0)` 設的是全域那一個）。把它傳給抽樣的函式，同一個種子就抽到同樣的結果，也不會被其他程式用掉的隨機數影響。

下面是這一集用到的三個小函式：

```python,ignore
{{#include ../../../src/llmz/rl.py:sample_actions}}
```

```python,ignore
{{#include ../../../src/llmz/rl.py:bandit_rewards}}
```

```python,ignore
{{#include ../../../src/llmz/rl.py:expected_reward}}
```

- `sample_actions` 就是第 1 章第 14 集的 `torch.multinomial`，照 softmax 的機率抽動作。
- `bandit_rewards`：拉一次拿到的是這台機器的平均值，再加一點雜訊。真實的獎勵常常帶雜訊：同一個 prompt 寫兩次，可能一次答對、一次答錯。
- `expected_reward` 是目標，記作 \\( J \\)：每台機器的機率乘上它的平均獎勵，再加總。一開始每台機率都是 0.2，\\( J \\) 是 2.8；最好的是編號 3 的那台，平均 5.0。

我們要找的是讓 \\( J \\) 變大的 logits。

### 梯度往上爬，但我們沒有平均值

第 1 章學過梯度：它指出往哪個方向走，函式會增加最快。要讓 \\( J \\) 變大，就是沿著 \\( J \\) 對 logits 的梯度往上爬（gradient ascent，和梯度下降相反方向）。

這個玩具問題裡，平均值 `means` 我們寫在程式裡，所以可以直接對 `expected_reward` 做 `backward()`：

```python
import torch

from llmz.rl import expected_reward

means = torch.tensor([0.0, 3.0, 2.0, 5.0, 4.0])
logits = torch.zeros(5, requires_grad=True)
expected_reward(logits, means).backward()
print("J 對 5 個 logits 的梯度", [round(g, 2) for g in logits.grad.tolist()])
```

執行結果：

```text
J 對 5 個 logits 的梯度 [-0.56, 0.04, -0.16, 0.44, 0.24]
```

編號 1、3、4 的機器平均獎勵（3、5、4）高於 2.8，梯度是正的，logit 該往上推；編號 0、2 的低於 2.8，梯度是負的。

可是真實的問題裡，沒有人告訴我們 `means`，我們只能拉、只能看到拉到的獎勵。這時可以用抽樣估計梯度，關鍵是一個小技巧：

\\[ \nabla J = \sum_a \nabla \pi(a)\, R(a) = \sum_a \pi(a)\, R(a)\, \nabla \log \pi(a) = \mathbb{E}_{a \sim \pi}\big[ R(a)\, \nabla \log \pi(a) \big] \\]

\\( \nabla \\) 是對 logits 的梯度，\\( \pi(a) \\) 是 policy 選動作 \\( a \\) 的機率，\\( R(a) \\) 是它的平均獎勵。第二個等號用到 \\( \ln x \\) 的導數是 \\( 1/x \\) 與鏈鎖律（第 1 章第 11 集）：\\( \nabla \log \pi = \nabla \pi / \pi \\)，所以 \\( \nabla \pi = \pi \nabla \log \pi \\)。這樣每一項都帶著 \\( \pi(a) \\)，整個式子就是「照 policy 抽動作」的期望值 \\( \mathbb{E} \\)（照機率加權的平均），而期望值可以用抽樣平均來估計：

1. 照 policy 抽一個動作 \\( a \\)，拿到獎勵 \\( r \\)。
2. 算 \\( r \cdot \nabla \log \pi(a) \\)。
3. 抽很多個，取平均。

\\( \nabla \log \pi(a) \\) 是「讓 \\( a \\) 的機率變大」的方向，\\( r \\) 決定推多用力：獎勵高就大力推，獎勵低就輕推。對 softmax 來說，這個方向有簡單的形式：被選到那台的 logit 加 1，每一台再減去自己的機率，也就是 `one_hot(a) - π`。

### 用程式驗證

第 5 章的 loss 都是「越小越好」，所以我們把 \\( -r \cdot \log \pi(a) \\) 當成 loss，對它做 `backward()`，梯度正好是 \\( -r \cdot \nabla \log \pi(a) \\)。`reinforce_loss` 就是這個 loss，推高或推低由第三個參數 `advantages` 決定（這一集 advantage 就是獎勵本身，下一集會換成別的東西）：

```python,ignore
{{#include ../../../src/llmz/rl.py:reinforce_loss}}
```

它和第 1 章的 `cross_entropy` 長得很像：同樣用 `log_softmax` 再取選到的那一格。差別是每個樣本多乘上一個 `advantages`，而且前面有負號。如果 `advantages` 全是 1，它就是 cross-entropy：「選到什麼就讓它更可能」。

先看單一樣本：選了編號 3 的機器，拿到獎勵 6，logits 全 0（機率都是 0.2）。負號的 loss 梯度取負，應該是 `6 * (one_hot - 0.2)`：

```python
import torch

from llmz.rl import reinforce_loss


def show(t):
    return " ".join(f"{x:+.2f}" for x in t.tolist())


logits = torch.zeros(1, 5, requires_grad=True)
reinforce_loss(logits, torch.tensor([3]), torch.tensor([6.0])).backward()
print("往上爬的方向", show(-logits.grad[0]))
one_hot = torch.zeros(5)
one_hot[3] = 1.0
print("手算 6 * (one_hot - 0.2)", show(6 * (one_hot - 0.2)))
```

執行結果：

```text
往上爬的方向 -1.20 -1.20 -1.20 +4.80 -1.20
手算 6 * (one_hot - 0.2) -1.20 -1.20 -1.20 +4.80 -1.20
```

被選到的那台往上推 4.8，其他四台各往下推 1.2；總和是 0，因為 softmax 的機率加總永遠是 1，某一台變大，其他台就得變小。

接著把很多樣本平均。`estimate_gradient` 抽 `n` 個動作、拿獎勵，用 `reinforce_loss` 算梯度，再取負號，得到對 \\( J \\) 的梯度估計：

```python,ignore
{{#include ../../../src/llmz/rl.py:estimate_gradient}}
```

函式開頭的 `logits.detach().requires_grad_()`：`detach()` 回傳一個不連到舊計算圖的張量，`requires_grad_()` 是就地設定 `requires_grad=True`。這樣不管傳進來的 logits 之前有沒有追蹤梯度，每次呼叫都從乾淨的狀態開始，算完的 `logits.grad` 只屬於這一次。

`baseline` 參數留到下一集。這一集先用預設的 `False`，advantage 就是獎勵。拿它和剛剛的精確梯度比較，樣本數 `n` 越大越準：

```python
import torch

from llmz.rl import estimate_gradient, expected_reward


def show(t):
    return " ".join(f"{x:+.2f}" for x in t.tolist())


means = torch.tensor([0.0, 3.0, 2.0, 5.0, 4.0])
logits = torch.zeros(5)
exact = torch.zeros(5, requires_grad=True)
expected_reward(exact, means).backward()
print("精確的梯度", show(exact.grad))

generator = torch.Generator().manual_seed(0)
for n in [1, 10, 100, 10000]:
    estimate = estimate_gradient(logits, means, n, generator=generator)
    errors = torch.stack([(estimate_gradient(logits, means, n, generator=generator) - exact.grad).abs().max() for _ in range(200)])
    print(f"n={n:5d}  估計 {show(estimate)}  重複 200 次的平均誤差 {errors.mean().item():.2f}")
```

執行結果：

```text
精確的梯度 -0.56 +0.04 -0.16 +0.44 +0.24
n=    1  估計 -0.40 -0.40 +1.60 -0.40 -0.40  重複 200 次的平均誤差 2.43
n=   10  估計 -0.54 -0.24 +0.06 +0.46 +0.26  重複 200 次的平均誤差 0.56
n=  100  估計 -0.56 +0.16 -0.22 +0.34 +0.28  重複 200 次的平均誤差 0.20
n=10000  估計 -0.56 +0.03 -0.17 +0.46 +0.23  重複 200 次的平均誤差 0.02
```

每一列的「估計」是用同一個種子抽出來的單次結果；最後一欄是重複 200 次、每次取 5 個分量裡誤差最大的那一個，再平均。

只抽 1 個樣本時，估計幾乎只告訴你「剛剛抽到的那台往上推」，和精確梯度差很多，誤差平均 2.43；抽到 10000 個時，已經很接近，平均誤差 0.02。誤差大約隨著 \\( 1/\sqrt{n} \\) 變小：從 100 個樣本到 10000 個，樣本多 100 倍，誤差少約 10 倍（0.20 到 0.02）。

這就是 policy gradient：**不用知道獎勵怎麼來的，只要能抽樣、能拿到分數，就能估計出讓平均獎勵變大的方向。**獎勵可以是任何黑盒子：答案比對、單元測試、甚至人的評分。

### 接到語言模型

把 bandit 換成語言模型，只有名詞要換：

- 「5 台機器」變成「整段回答」。回答的機率是每個 token 條件機率的連乘，取 log 之後是連加，正是第 12 章第 3 集的 `sequence_logprob`。它沒有 `no_grad`，就是為了這裡對它做 `backward()`。
- 「拉一次」變成「用語言模型對 prompt 生成一段回答」（第 6、11 章的生成函式），再由獎勵函式打分。
- `reinforce_loss` 的 `logp` 換成整段回答的 log 機率，其餘不變。

第 4 集會做這件事。在那之前，先把 bandit 上的版本做好：下一集解決估計太「吵」的問題，第 3 集再看 GRPO 的做法。

## 重點整理

- RL 讓模型自己寫答案、由獎勵函式打分，不需要標準答案；policy 是根據情況決定動作的規則，語言模型本身就是 policy，目標是平均獎勵 \\( J \\) 越高越好。
- 只能抽樣、不知道平均獎勵時，用 \\( \nabla J = \mathbb{E}[R \, \nabla \log \pi(a)] \\) 估計梯度：抽動作、拿獎勵，獎勵越高就越用力把這個動作的機率往上推。
- softmax 的 \\( \nabla \log \pi(a) \\) 是 `one_hot(a) - π`；`reinforce_loss` 是 \\( -\text{advantage} \cdot \log \pi(a) \\) 的平均，梯度下降它就是讓平均獎勵往上爬。
- 抽的樣本越多，估計越接近精確梯度，誤差約隨 \\( 1/\sqrt{n} \\) 下降；單一樣本的估計非常吵。
- 語言模型上的做法一樣，只是動作是整段回答，log 機率是每個 token 的 log 機率相加。
