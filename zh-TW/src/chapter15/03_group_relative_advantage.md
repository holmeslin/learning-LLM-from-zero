# GRPO 的 group-relative advantage

## 本集目標

學會 GRPO 怎麼算 advantage：同一個 prompt 抽一組回答，用組內平均當 baseline，可選擇再除以組內標準差。看它在「答對機率不同」的 prompt 上怎麼分配學習訊號。

## 正文

### 上一集的 baseline 不夠用

上一集的 baseline 是「這一批獎勵的平均」，bandit 只有一個情況（5 台機器），夠用。語言模型的情況是 prompt，每個 prompt 難度不同：有的題目模型九成答得對，有的只有 1%。如果把所有 prompt 的回答放在一起算平均，簡單題的回答幾乎全部「比平均好」，困難題的回答幾乎全部「比平均差」，這個 baseline 沒有回答「這個 prompt 本來就答得對有多容易」。

比較理想的 baseline，是「這個 prompt 平均能拿幾分」。PPO 的做法是另外訓練一個 value 模型來預測這個數字，這個模型通常和 policy 差不多大，要多佔一份記憶體、多一份訓練。

GRPO（Group Relative Policy Optimization，DeepSeekMath 論文提出）換了一個便宜的做法：**同一個 prompt 直接抽一組（group）G 個回答，用這一組自己的獎勵平均當 baseline。**不需要 value 模型，因為這 G 個回答就是這個 prompt 的樣本。

### advantage 的算法

第 \\( i \\) 個回答的 advantage（它比同組平均好多少）：

\\[ A_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G) + \varepsilon} \\]

分子是上一集的「獎勵減 baseline」，baseline 是組內平均。分母把組內的獎勵尺度拉成 1，\\( \varepsilon \\) 是很小的數字（預設 \\( 10^{-6} \\)），避免整組獎勵都一樣、標準差是 0 的時候除以 0。

```python,ignore
{{#include ../../../src/llmz/rl.py:group_advantages}}
```

- `rewards` 的形狀是 `(P, G)`：P 個 prompt，每個 prompt 一組 G 個回答。平均與標準差都沿著 `dim=1`（組內）算。
- `std` 是樣本標準差（除以 \\( G-1 \\)，`torch.std` 的預設）。
- `normalize_std=False` 只減平均，不除標準差，這樣可以比較有沒有除的差別。

用一個具體的例子。三個 prompt、每個抽 8 個回答，答對 1 分、答錯 0 分：簡單的答對 7 個，困難的只答對 1 個，第三個全部答錯。比較三種算法的 advantage：

```python
import torch

from llmz.rl import group_advantages


def show(t):
    return " ".join(f"{x:+.2f}" for x in t.tolist())


rewards = torch.tensor([
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],  # 簡單：答對 7 個
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 困難：答對 1 個
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 全部答錯
])
methods = [
    ("全部 24 個回答一起減平均", rewards - rewards.mean()),
    ("每組減自己的平均", group_advantages(rewards, normalize_std=False)),
    ("再除以組內標準差", group_advantages(rewards, normalize_std=True)),
]
for name, adv in methods:
    print(name)
    for label, row in zip(["簡單", "困難", "全錯"], adv):
        print(f"  {label}  {show(row)}")
```

執行結果：

```text
全部 24 個回答一起減平均
  簡單  +0.67 +0.67 +0.67 +0.67 +0.67 +0.67 +0.67 -0.33
  困難  +0.67 -0.33 -0.33 -0.33 -0.33 -0.33 -0.33 -0.33
  全錯  -0.33 -0.33 -0.33 -0.33 -0.33 -0.33 -0.33 -0.33
每組減自己的平均
  簡單  +0.12 +0.12 +0.12 +0.12 +0.12 +0.12 +0.12 -0.88
  困難  +0.88 -0.12 -0.12 -0.12 -0.12 -0.12 -0.12 -0.12
  全錯  +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00
再除以組內標準差
  簡單  +0.35 +0.35 +0.35 +0.35 +0.35 +0.35 +0.35 -2.47
  困難  +2.47 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35
  全錯  +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00
```

- **全部一起減平均**：簡單題的 7 個正確回答和困難題那 1 個正確回答，拿到一樣的 +0.67，全錯那組的回答也被推低 0.33，即使這組根本沒有哪個回答比較好。
- **每組減自己的平均**：簡單題裡答對的只有 +0.12（本來就該答對），唯一答錯的是 -0.88；困難題唯一答對的拿到 +0.88，這個稀有的成功被大大地獎勵。全錯那組每一個 advantage 都是 0。
- **再除以標準差**：形狀不變，數字被放大。困難題的稀有成功從 +0.88 變成 +2.47，是不除的 2.8 倍：標準差越小（成功率越極端），分母越小，放得越大；簡單題裡唯一答錯的那個，也從 -0.88 放大成 -2.47。

「全錯的組 advantage 是 0」是 GRPO 的重要性質：全組答錯、或全組答對（獎勵都一樣），這組沒有「哪個回答比較好」的資訊，就不提供梯度。`eps` 就是為這種情況留的：分子是 0、分母是 `eps`，結果是 0，不會變成 `nan`。

### 一組要多大

困難題的一組如果全錯，就學不到東西。答對機率是 \\( p \\)，一組 \\( G \\) 個回答至少有一個答對的機率是 \\( 1 - (1-p)^G \\)：

```python
for p in [0.5, 0.1, 0.01]:
    chances = ", ".join(f"G={g}: {1 - (1 - p) ** g:.1%}" for g in [4, 8, 16, 64])
    print(f"答對機率 {p:4.2f}  組內至少一個答對的機率  {chances}")
```

執行結果：

```text
答對機率 0.50  組內至少一個答對的機率  G=4: 93.8%, G=8: 99.6%, G=16: 100.0%, G=64: 100.0%
答對機率 0.10  組內至少一個答對的機率  G=4: 34.4%, G=8: 57.0%, G=16: 81.5%, G=64: 99.9%
答對機率 0.01  組內至少一個答對的機率  G=4: 3.9%, G=8: 7.7%, G=16: 14.9%, G=64: 47.4%
```

答對機率 1% 的題目，一組 8 個回答只有 7.7% 的機會看到答對的，其餘 92.3% 的組 advantage 全是 0。G 越大，每一步的成本越高（要多生成 G 倍的回答），但學習訊號也越穩定，這是調整 GRPO 時最常見的取捨。

### 在玩具題上訓練

把 bandit 換成「有 prompt 的 bandit」：3 個 prompt，每個 prompt 各有一個 policy（一排 10 個候選答案的 logits），只有一個答案是對的。這就是可驗證獎勵最簡單的樣子：答案和標準答案一樣就是 1 分，不一樣是 0 分。

```python,ignore
{{#include ../../../src/llmz/rl.py:verifiable_rewards}}
```

```python,ignore
{{#include ../../../src/llmz/rl.py:train_groups}}
```

`train_groups` 每一步：每個 prompt 各抽一組 G 個回答（`sample_actions` 的 logits 形狀是 `(P, K)` 時，每一列各抽一組）、算 0／1 獎勵、用 `group_advantages` 算 advantage，再用上一集學的 `reinforce_loss` 往上爬。`expand` 與 `reshape` 把同一個 prompt 的 G 個回答接到同一列 logits 上。三個 prompt 的初始答對機率是 50%、10%、1%（簡單、中等、困難），每組 G=8。比較要不要除以標準差：

```python
import math

import torch

from llmz.functional import softmax
from llmz.rl import train_groups

correct = torch.tensor([2, 5, 7])  # 每個 prompt 的正確答案（共 10 個候選）
logits = torch.zeros(3, 10)
logits[torch.arange(3), correct] = torch.tensor([math.log(9), 0.0, math.log(9 / 99)])  # 答對機率 0.5、0.1、0.01


def accuracy(logits):
    return softmax(logits)[torch.arange(3), correct]


for normalize_std in [False, True]:
    print(f"normalize_std={normalize_std}（種子 0）")
    for steps in [0, 50, 100, 200, 400]:
        acc = accuracy(train_groups(logits, correct, group_size=8, steps=steps, lr=0.5, normalize_std=normalize_std, seed=0))
        print(f"  {steps:3d} 步  簡單 {acc[0]:.2f}  中等 {acc[1]:.2f}  困難 {acc[2]:.2f}")
    hard = [accuracy(train_groups(logits, correct, group_size=8, steps=400, lr=0.5, normalize_std=normalize_std, seed=s))[2].item() for s in range(3)]
    print("  困難題 400 步後，種子 0、1、2：", " ".join(f"{h:.2f}" for h in hard))
```

執行結果：

```text
normalize_std=False（種子 0）
    0 步  簡單 0.50  中等 0.10  困難 0.01
   50 步  簡單 0.83  中等 0.29  困難 0.01
  100 步  簡單 0.92  中等 0.72  困難 0.01
  200 步  簡單 0.97  中等 0.94  困難 0.01
  400 步  簡單 0.98  中等 0.98  困難 0.02
  困難題 400 步後，種子 0、1、2： 0.02 0.02 0.03
normalize_std=True（種子 0）
    0 步  簡單 0.50  中等 0.10  困難 0.01
   50 步  簡單 0.93  中等 0.77  困難 0.01
  100 步  簡單 0.97  中等 0.95  困難 0.02
  200 步  簡單 0.98  中等 0.99  困難 0.04
  400 步  簡單 0.99  中等 0.99  困難 0.98
  困難題 400 步後，種子 0、1、2： 0.98 0.96 0.98
```

- 簡單與中等的 prompt，兩種算法都學得起來，除以標準差的版本比較快（100 步時中等題 0.95 對 0.72）。
- 困難的 prompt 有一段平坦的時間：機率 1% 時，大部分的組全錯、advantage 是 0，policy 不動；平均要十幾步（機率 7.7%）才有一步碰巧答對一次，推一下。不除以標準差，每次的推力只有 +0.88，400 步後只到 0.02；除以標準差的推力是 +2.47，400 步時已經是 0.98。
- 這不代表除以標準差永遠比較好：它對稀有成功這一組，等於把步伐放大 2.8 倍，有點像把學習率調大。要不要除，第 6 集（Dr. GRPO）會再討論。

這些是用同一個學習率 0.5、固定種子的單次結果，不是在多個種子上平均。

## 重點整理

- GRPO 對同一個 prompt 抽一組 G 個回答，用組內獎勵的平均當 baseline，不需要像 PPO 那樣另外訓練 value 模型。
- `group_advantages`：`(r - 組內平均) / (組內標準差 + eps)`，`normalize_std=False` 則只減平均；每組的 advantage 平均為 0，除以標準差時組內標準差為 1。
- 全組獎勵相同（全對或全錯）時，advantage 是 0、沒有梯度；答對機率 1% 的題目，一組 8 個回答有 92.3% 的機會全錯。
- 除以標準差把稀有成功的 advantage 放大（8 個答對 1 個：+0.88 變 +2.47），玩具題上困難 prompt 學得快很多，但這等於放大了步伐，是否合理第 6 集再看。
- 在 3 個 prompt 的 toy RLVR 上，`train_groups` 搭配除以標準差，把答對機率從 50%、10%、1% 訓練到 0.98 以上（種子 0 的 400 步）；不除標準差時，答對機率 1% 的那題 400 步後仍只有 0.02。
