# reward model 的想法

## 本集目標

知道偏好資料長什麼樣子、reward model 做什麼，以及 Bradley–Terry 怎麼把「這個比那個好」變成一個可以訓練的 loss。實作 `bradley_terry_loss`，用程式產生偏好資料，量出 SFT 模型原本偏好正確回答的比例，那是下一集 DPO 的起點。

## 正文

### 比較，比寫標準答案容易

SFT 需要標準答案：每一題要有一個「應該這樣回答」的範本。很多時候這很難。兩段摘要哪一段寫得比較好？兩個回答哪一個比較有禮貌？要人寫出完美的答案很費力，但把兩個回答並排，說「這個比較好」容易得多。

所以後訓練的另一條路是收集**偏好資料**（preference data）：每一筆是一個提示、一個比較好的回答（`chosen`）、一個比較差的回答（`rejected`）。

我們的題目有標準答案，可以用程式「假裝」有人在比：`chosen` 是標準答案，`rejected` 是同一題的另一個候選。`color` 題的錯誤答案是題目裡的另一個顏色，`who` 題是另外兩個人名之一，加、減、數數量是別的數字：

```python,ignore
{{#include ../../../src/llmz/posttrain.py:preference_data}}
```

`make_preferences` 用和 `make_tasks` 一樣的 `seed`、`split`、`kinds`，所以同樣的參數產生的是同一批題目；`"train"` 與 `"valid"` 照舊不重複。

```python
from llmz.posttrain import make_preferences

shown = set()
for pair in make_preferences(50, seed=0):
    if pair.kind not in shown:
        shown.add(pair.kind)
        print(f"{pair.kind:9s}{pair.question}\n  chosen：{pair.chosen}　rejected：{pair.rejected}")
```

執行結果：

```text
color    Lily has a white apple and a black flower. What color is the flower?
  chosen：black　rejected：white
subtract Ben has four kites. He gives two to Tim. How many kites does Ben have left?
  chosen：two　rejected：seven
add      Ben has four books. He finds five more. How many books does Ben have now?
  chosen：nine　rejected：zero
count    Ben has a cup, a toy and a kite. How many things does Ben have?
  chosen：three　rejected：five
who      Mia has a kite, Ben has a flower and Tim has a ball. Who has the ball?
  chosen：Tim　rejected：Mia
```

`color` 與 `who` 的 `rejected` 是題目裡真的出現過的詞，不是亂湊的。分辨 `black` 與 `white` 哪個是花的顏色，要看題目才知道；只看回答分辨不出來。

### reward model：給回答打一個分數

**reward model**（獎勵模型）是一個函式 \\( r(x, y) \\)：輸入提示 \\( x \\) 與回答 \\( y \\)，輸出一個數字，越大代表越好。真實的做法是拿一個語言模型，把最後的輸出層從「詞彙表大小的 logits」換成「一個數字」，用偏好資料訓練（Ouyang 等人 2022 的 InstructGPT 就是請標註者把同一個提示的幾個回答排名，再用兩兩比較訓練獎勵模型）。有了它，要改進一個語言模型，就讓它多寫分數高的回答。

這本書不訓練獎勵模型：下一集的 DPO 會繞過它。但訓練獎勵模型的 loss，也就是這集要實作的東西，DPO 會原封不動地沿用。

### Bradley–Terry：分數的差變成勝率

人只給「哪個比較好」，沒有給分數，怎麼從比較得到分數？Bradley 與 Terry（1952）處理過一模一樣的問題：選手兩兩比賽，只知道誰贏。他們的模型是：每個選手有一個實力分數 \\( r \\)，A 贏過 B 的機率只由分數的**差**決定：

\\[ P(A \succ B) = \sigma(r_A - r_B), \qquad \sigma(x) = \frac{1}{1 + e^{-x}} \\]

\\( \sigma \\)（sigmoid）把任何實數壓到 0 到 1 之間。差 0 就是各半，差越大贏的機率越接近 1：

```python
import math

import torch

from llmz.posttrain import bradley_terry_loss

for diff in [-2, 0, 1, 2, 4]:
    print(f"分數差 {diff:+d}：好的贏的機率 {1 / (1 + math.exp(-diff)):.3f}")
```

執行結果：

```text
分數差 -2：好的贏的機率 0.119
分數差 +0：好的贏的機率 0.500
分數差 +1：好的贏的機率 0.731
分數差 +2：好的贏的機率 0.881
分數差 +4：好的贏的機率 0.982
```

訓練的目標是讓人選的那個贏的機率越大越好，也就是最大化 \\( \log \sigma(r_{\text{chosen}} - r_{\text{rejected}}) \\)，加上負號就是 loss：

```python,ignore
{{#include ../../../src/llmz/posttrain.py:bradley_terry}}
```

`-log σ(x)` 等於 `log(1 + e^-x)`，也就是 `logsumexp([0, -x])`，直接用第 1 章的 `logsumexp`，分數差再大或再小都不會溢位。

```python
import math

import torch

from llmz.posttrain import bradley_terry_loss

for margin in [-2.0, 0.0, 2.0, 10.0]:
    loss = bradley_terry_loss(torch.tensor([margin]), torch.tensor([0.0]))
    print(f"margin {margin:+5.1f}：loss {loss.item():.4f}")
print("log 2 =", round(math.log(2), 4))

chosen, rejected = torch.tensor([1.5, 0.2, 3.0]), torch.tensor([0.5, 0.7, -1.0])
shifted = bradley_terry_loss(chosen + 5, rejected + 5)
print("全部加 5 之後 loss 不變：", abs(shifted.item() - bradley_terry_loss(chosen, rejected).item()) < 1e-6)
```

執行結果：

```text
margin  -2.0：loss 2.1269
margin  +0.0：loss 0.6931
margin  +2.0：loss 0.1269
margin +10.0：loss 0.0000
log 2 = 0.6931
全部加 5 之後 loss 不變： True
```

三件事：

- 兩個分數一樣時 loss 是 \\( \log 2 \\)（猜一半一半）；`chosen` 比 `rejected` 高越多，loss 越接近 0；`chosen` 反而低，loss 就大。
- 只有**差**有意義。每個分數都加 5，loss 完全一樣。這件事下一集會用到。
- 梯度的方向對：`chosen` 的分數被往上推，`rejected` 的往下推（`tests/test_posttrain.py` 檢查）。

### 從勝負紀錄還原分數

用四個選手、2000 場比賽驗證：每場比賽的勝負依 Bradley–Terry 的機率隨機決定，我們只看勝負，用 `bradley_terry_loss` 和第 5 章的 `AdamW` 學出每個人的分數：

```python
import math
import random

import torch
from torch import nn

from llmz.optim import AdamW
from llmz.posttrain import bradley_terry_loss

true = [2.0, 1.0, 0.0, -1.0]  # 四個選手真正的實力（只有差有意義）
rng = random.Random(0)
games = []  # (贏的人, 輸的人)
for _ in range(2000):
    a, b = rng.sample(range(4), 2)
    a_wins = rng.random() < 1 / (1 + math.exp(-(true[a] - true[b])))  # Bradley–Terry：實力差決定贏的機率
    games.append((a, b) if a_wins else (b, a))
winners, losers = torch.tensor([w for w, _ in games]), torch.tensor([l for _, l in games])

scores = nn.Parameter(torch.zeros(4))  # 每個選手一個分數，從 0 開始
optimizer = AdamW([scores], lr=0.05, weight_decay=0.0)
for step in range(300):
    loss = bradley_terry_loss(scores[winners], scores[losers])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

learned = scores.detach() - scores.detach().mean()  # 平移不影響 loss，所以只看減掉平均之後的分數
target = torch.tensor(true) - torch.tensor(true).mean()
print("學到的分數：", [round(x, 1) for x in learned.tolist()])
print("真正的實力：", target.tolist())
print("名次一樣：", learned.argsort().tolist() == target.argsort().tolist())
```

執行結果：

```text
學到的分數： [1.5, 0.5, -0.6, -1.5]
真正的實力： [1.5, 0.5, -0.5, -1.5]
名次一樣： True
```

沒看到任何分數，只看勝負，學到的分數和真正的實力只差一點（2000 場比賽有運氣成分，不會剛好一樣）。獎勵模型做的是同一件事，只是「選手」換成（提示，回答），分數由語言模型算。

### 用 SFT 模型當評分器

我們沒有獎勵模型，但有一個現成的分數：語言模型給一個回答的 log 機率。`sequence_logprob`（第 12 章）算的就是它，只算回覆的 token。把 `chosen` 與 `rejected` 兩條對話疊成一批，各算一次：

```python,ignore
{{#include ../../../src/llmz/posttrain.py:pair_logprobs}}
```

- `pair_batch` 把每個 `Pair` 排成兩條對話，疊成 `(2B, T)`：前 `B` 列是 `chosen`，後 `B` 列是 `rejected`，一次 forward 就算完。
- `pair_logprobs` 把結果切成兩半。有梯度（訓練會用）。
- `preference_rate` 回答一個問題：模型給 `chosen` 的 log 機率比 `rejected` 高的 pair 佔幾成？平手不算。它是把 log 機率當成獎勵、用成對比較打分數。一次只算 `chunk` 個 pair，因為 logits 的大小是 `(2B, T, 10000)`，一次放 1000 個 pair 記憶體會爆。

用第 3 集存的 SFT 模型，量 1000 道 `"valid"` 題的偏好比例，並和第 3 集的答對題數放在一起：

```python,norun
import torch

from llmz.device import get_device
from llmz.model import ModelConfig, TransformerLM
from llmz.posttrain import KINDS, generate_replies, make_preferences, make_tasks, preference_rate, reward, task_messages
from llmz.tokenizer import Tokenizer

device = get_device()
tokenizer = Tokenizer.load("data/tokenizer.json")
pairs = make_preferences(1000, seed=1, split="valid")  # 和第 3 集同一批題目
tasks = make_tasks(1000, seed=1, split="valid")

model = TransformerLM(ModelConfig()).to(device)
model.load_state_dict(torch.load("runs/sft/model.pt", map_location=device))
model.eval()
replies = generate_replies(model, tokenizer, [task_messages(t) for t in tasks])
for kind in KINDS:
    rate = preference_rate(model, tokenizer, [p for p in pairs if p.kind == kind])
    hits = [reward(t, r) for t, r in zip(tasks, replies) if t.kind == kind]
    print(f"{kind:9s}偏好比例 {rate:.1%}　greedy 答對 {int(sum(hits))}/{len(hits)}")
```

M4 Max 上的輸出（MPS，某一次）：

```text
add      偏好比例 100.0%　greedy 答對 188/188
subtract 偏好比例 100.0%　greedy 答對 200/200
count    偏好比例 100.0%　greedy 答對 192/192
color    偏好比例 50.8%　greedy 答對 101/199
who      偏好比例 46.2%　greedy 答對 74/221
```

加、減、數數量：SFT 模型給正確答案的 log 機率每一題都比錯的高。`color` 是 50.8%，`who` 是 46.2%：亂猜的水準。這和第 3 集 greedy 答對的比例（51%、33%）一致，只是換了一種量法：greedy 只看模型最想寫哪個詞，偏好比例只問「正確的比錯誤的高嗎」，`who` 有兩個錯誤選項時，後者是一對一比較，亂猜是 50%。

下一集要做的事：用 `chosen`／`rejected` 這些成對資料，讓 `color` 與 `who` 的偏好比例上去。

## 重點整理

- 偏好資料是（提示、`chosen`、`rejected`）。比較兩個回答，比寫標準答案容易；我們用 `make_preferences`，把標準答案配上同一題的另一個候選。
- reward model 是 \\( r(x, y) \\)，給回答一個分數；真實做法是語言模型加一個輸出一個數字的頭，用偏好資料訓練。這本書不訓練它。
- Bradley–Terry：\\( P(\text{chosen} \succ \text{rejected}) = \sigma(r_{\text{chosen}} - r_{\text{rejected}}) \\)，loss 是它的 \\( -\log \\)（`bradley_terry_loss`）。分數一樣時是 \\( \log 2 \\)；只有差有意義，全部平移 loss 不變。
- `pair_batch`、`pair_logprobs`、`preference_rate`：把語言模型的 log 機率當分數，量「正確的比錯誤的高」的比例。M4 Max 實測：SFT 模型在加、減、數數量是 100%，`color` 50.8%、`who` 46.2%（亂猜是 50%）。
