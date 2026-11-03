# loglikelihood 評多選題

## 本集目標

自己寫出多選題的評分：對每個選項算 log 機率、挑最大的。比較「總 log 機率」與「除以長度」兩種算法，並量標準模型在小題庫上的準確率，以及 few-shot 有沒有幫助。

## 正文

### 把選擇題變成算機率

語言模型只會做一件事：給前面的 token，算下一個 token 的機率。MMLU、HellaSwag 這類選擇題沒有要模型「回答」，而是把每個選項分別接在題目後面，算模型給這段選項文字多大的機率，挑最大的。

一段文字的機率是每個 token 條件機率的連乘，取 log 之後變成連加：

\\[ \log p(\text{選項} \mid \text{題目}) = \sum_{t \in \text{選項}} \log p(x_t \mid x_{<t}) \\]

這個總和叫 loglikelihood（對數似然），和第 1 章的 cross-entropy 是同一個東西，只是不取平均，而且只算選項那幾個 token，題目的 token 不算。

### 帶 mask 的序列 log 機率

```python,ignore
{{#include ../../../src/llmz/evaluation.py:sequence_logprob}}
```

- `model(ids[:, :-1])`：位置 \\( t \\) 的輸出預測第 \\( t+1 \\) 個 token（第 4 章第 14 集），所以輸入少最後一個，答案是 `ids[:, 1:]`。
- `logp[batch, position, ids[:, 1:]]` 是第 1 章 `cross_entropy` 那個 `logp[torch.arange(N), targets]` 的批次版：三個索引張量的形狀是 `(B, 1)`、`(T-1,)`、`(B, T-1)`，broadcasting 之後得到 `(B, T-1)`，每個位置只取「正確答案那一格」的 log 機率。
- `mask` 和 `ids` 一樣大，`True` 的位置才算進總和。第 0 個 token 沒有前文可以預測，永遠不算。
- 這個函式沒有 `no_grad`：第 14 章要用它算 DPO 的 loss，需要對它 backward。評分時再從外面包 `no_grad`。

### 選項評分

```python,ignore
{{#include ../../../src/llmz/evaluation.py:choice_logprobs}}
```

- 題目與每個選項**分開** encode 再接起來，這樣所有選項共用同一段題目的 token。標準 tokenizer 的 pre-tokenization 讓空格跟著後面的單字（第 3 章第 4 集），所以選項開頭帶空格、分開 encode，通常和整段一起 encode 切出一樣的 token：在題庫的 420 個「題目＋選項」裡，兩種做法完全相同。
- 選項長短不一，短的往右補 0，湊成一個 `(C, L)` 的 batch 一次算完。causal mask 讓每個位置只看左邊（第 4 章第 10 集），所以右邊補的 0 不會影響前面的結果，mask 也不會把它們算進去。
- 回傳每個選項的總 log 機率，以及它有幾個 token，下一節要用。

### 總和還是平均

```python,ignore
{{#include ../../../src/llmz/evaluation.py:pick_choice}}
```

直接比總和有個偏見：每多一個 token 就多乘一個不超過 1 的機率，log 機率只會更負。同樣合理的兩個選項，長的幾乎一定輸給短的。常見的補救是除以長度，變成「每個 token 平均的 log 機率」。但平均也有偏見：它偏愛每個 token 都很好猜的長選項，例如把常見的片語接一大串。長度可以用 token 數，也可以用字元數或 byte 數，後者不會因為 tokenizer 把字切得多細而改變。

兩種算法沒有誰對誰錯，答案取決於題庫的選項。評測工具通常兩種都報，也就是為什麼 `predict_mcq` 有 `normalize` 參數：

```python,ignore
{{#include ../../../src/llmz/evaluation.py:predict_mcq}}
```

```python,ignore
{{#include ../../../src/llmz/evaluation.py:mcq_accuracy}}
```

### 先用玩具模型驗證

要確定寫對了，最好用一個機率我們完全知道的模型。`ToyLM` 只看「現在這個 token」，`probs[a][b]` 是 a 後面接 b 的機率，所以每個序列的 log 機率都能用手算：

```python
import math

import torch
from torch import nn

from llmz.evaluation import choice_logprobs, mcq_accuracy, pick_choice, predict_mcq, sequence_logprob
from llmz.model import ModelConfig


class ToyLM(nn.Module):
    """只看「現在這個 token」的語言模型：probs[a][b] 是 a 後面接 b 的機率。"""

    def __init__(self, probs):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(probs).log())
        self.cfg = ModelConfig(vocab_size=3, context_length=16)  # choice_logprobs 要知道長度上限

    def forward(self, ids):
        return self.logits[ids]  # 每個位置的 logits 就是「這個 token 後面接什麼」那一列


class ToyTokenizer:
    def encode(self, text):
        return ["abc".index(ch) for ch in text]  # a、b、c 各是一個 token


tokenizer = ToyTokenizer()

# 1. a 後面最可能接 b；b 後面最可能接 c
model = ToyLM([[0.1, 0.6, 0.3], [0.2, 0.2, 0.6], [0.5, 0.25, 0.25]])
ids = torch.tensor([[0, 1, 2]])
mask = torch.tensor([[False, True, True]])  # 只算 b、c 兩個 token
print(f"手算 log 0.6 + log 0.6 = {2 * math.log(0.6):.4f}")
print(f"sequence_logprob = {sequence_logprob(model, ids, mask).item():.4f}")

# 2. 題目是 "a"，三個選項
choices = ["b", "c", "bc"]
logprobs, n_tokens = choice_logprobs(model, tokenizer, "a", choices)
for choice, lp, n in zip(choices, logprobs, n_tokens):
    print(f"選項 {choice!r}：總和 {lp.item():.4f}，{n.item()} 個 token")
print("總和最大的是", choices[pick_choice(logprobs)])

# 3. 另一個模型：單獨一個 c 的機率比 bbbb 大，但 bbbb 平均每個 token 比較好猜
model = ToyLM([[0.2, 0.3, 0.5], [0.05, 0.9, 0.05], [0.3, 0.3, 0.4]])
choices = ["c", "bbbb"]
logprobs, n_tokens = choice_logprobs(model, tokenizer, "a", choices)
print("總和", [round(x, 4) for x in logprobs.tolist()], "->", choices[pick_choice(logprobs)])
print("平均", [round(x, 4) for x in (logprobs / n_tokens).tolist()], "->", choices[pick_choice(logprobs, n_tokens)])

# 4. 同樣的題目，正確答案是 bbbb：兩種算法的準確率不一樣
items = [{"context": "a", "choices": choices, "answer": 1}]
for normalize in ["none", "token"]:
    print(f"normalize={normalize}：預測 {predict_mcq(model, tokenizer, items, normalize=normalize)}（每題選的索引），"
          f"準確率 {mcq_accuracy(model, tokenizer, items, normalize=normalize)}")
```

執行結果：

```text
手算 log 0.6 + log 0.6 = -1.0217
sequence_logprob = -1.0217
選項 'b'：總和 -0.5108，1 個 token
選項 'c'：總和 -1.2040，1 個 token
選項 'bc'：總和 -1.0217，2 個 token
總和最大的是 b
總和 [-0.6931, -1.5201] -> c
平均 [-0.6931, -0.38] -> bbbb
normalize=none：預測 [0]（每題選的索引），準確率 0.0
normalize=token：預測 [1]（每題選的索引），準確率 1.0
```

第 1 段：手算與函式一致。第 2 段：`'bc'` 的總和是兩個 \\( \log 0.6 \\) 相加，比單獨的 `'b'` 更負。第 3、4 段：同一個模型，總和選 `c`，平均選 `bbbb`；正確答案是 `bbbb` 時，準確率從 0 變成 1。`tests/test_evaluation.py` 也是用這種玩具模型測試。

### 標準模型在題庫上的準確率

把真正的模型接上去，在 100 題的 `test` 上量三種算法，並用 `perf_counter` 看花多少時間：

```python,norun
import time

import torch

from llmz.device import get_device
from llmz.evaluation import load_mcq, mcq_accuracy
from llmz.model import ModelConfig, TransformerLM
from llmz.tokenizer import Tokenizer

device = get_device()
model = TransformerLM(ModelConfig()).to(device)
model.load_state_dict(torch.load("runs/base/model.pt", map_location=device))
model.eval()
tokenizer = Tokenizer.load("data/tokenizer.json")
test = load_mcq("test")

for normalize in ["none", "token", "byte"]:
    start = time.perf_counter()
    accuracy = mcq_accuracy(model, tokenizer, test, normalize=normalize)
    print(f"normalize={normalize:<5} 準確率 {accuracy:.0%}（{time.perf_counter() - start:.1f} 秒）")
print("隨機猜測 25%")
```

M4 Max 上的輸出（準確率每次相同，時間是某一次）：

```text
normalize=none  準確率 81%（3.8 秒）
normalize=token 準確率 88%（2.3 秒）
normalize=byte  準確率 86%（2.5 秒）
隨機猜測 25%
```

評分沒有取樣，同樣的模型與題庫每次的答案都一樣（CPU 與 MPS 也得到同樣的 81%、88%、86%）；不確定性只來自題目太少。三個數字都比隨機猜測的 25% 高出 12 個標準差以上（上一集算過，標準差約 4.3 個百分點），模型確實懂得挑出合理的結尾。

三種算法之間差了 2 到 7 個百分點，要小心解讀。總和與每個 token 的平均有 22 題選得不一樣，看看它們各自對在哪裡：

```python,norun
import torch

from llmz.device import get_device
from llmz.evaluation import choice_logprobs, load_mcq, predict_mcq
from llmz.model import ModelConfig, TransformerLM
from llmz.tokenizer import Tokenizer

device = get_device()
model = TransformerLM(ModelConfig()).to(device)
model.load_state_dict(torch.load("runs/base/model.pt", map_location=device))
model.eval()
tokenizer = Tokenizer.load("data/tokenizer.json")
test = load_mcq("test")

by_sum = predict_mcq(model, tokenizer, test)
by_token = predict_mcq(model, tokenizer, test, normalize="token")
different = [i for i in range(len(test)) if by_sum[i] != by_token[i]]
sum_right = sum(by_sum[i] == test[i]["answer"] for i in different)
token_right = sum(by_token[i] == test[i]["answer"] for i in different)
print(f"兩種算法選得不一樣的有 {len(different)} 題：只有總和答對 {sum_right} 題，只有平均答對 {token_right} 題，兩個都錯 {len(different) - sum_right - token_right} 題")

item = next(it for it in test if it["context"].startswith("It was a cold winter day"))
logprobs, n_tokens = choice_logprobs(model, tokenizer, item["context"], item["choices"])
print(item["context"])
for i, choice in enumerate(item["choices"]):
    mark = "  <- 答案" if i == item["answer"] else ""
    print(f"{'ABCD'[i]}.{choice}  總和 {logprobs[i]:.2f}，{n_tokens[i]} 個 token，平均 {logprobs[i] / n_tokens[i]:.2f}{mark}")
```

M4 Max 上的輸出：

```text
兩種算法選得不一樣的有 22 題：只有總和答對 6 題，只有平均答對 13 題，兩個都錯 3 題
It was a cold winter day. Ben put on his coat and
A. gloves before going out to play in the snow.  總和 -20.60，10 個 token，平均 -2.06  <- 答案
B. sunglasses and a straw hat.  總和 -30.36，6 個 token，平均 -5.06
C. swimsuit.  總和 -14.16，2 個 token，平均 -7.08
D. went to the beach to swim.  總和 -14.65，7 個 token，平均 -2.09
```

總和選了只有 2 個 token 的 `swimsuit.`：它每個 token 都很不合理（平均 -7.08），但只有兩個 token，總共的 -14.16 反而比正確答案的 -20.60 好。平均則選出正確的 A。

不過這 22 題裡平均答對 13 題、總和答對 6 題，其餘 3 題兩個都錯。如果兩種算法其實一樣好，這 19 題各自歸誰就像擲硬幣：

```python
from math import comb

n, k = 6 + 13, 13
p = sum(comb(n, i) for i in range(k, n + 1)) / 2**n
print(f"擲 {n} 次公平硬幣，同一面出現 {k} 次以上的機率：{p:.3f}")
```

執行結果：

```text
擲 19 次公平硬幣，同一面出現 13 次以上的機率：0.084
```

大約 8%，不算很小。所以只能保守地說：在這份題庫上，除以長度的算法看起來比較好（像上面的 `swimsuit.` 這種很短的錯誤選項，在總和裡佔便宜），但 100 題還不夠斷定。

### few-shot 有幫助嗎

最後比較 0-shot 與 few-shot。用上一集的 `build_prompt`，範例從 `dev` 取。除了 shot 數，也換掉挑的範例與分隔：

```python,norun
import torch

from llmz.device import get_device
from llmz.evaluation import load_mcq, mcq_accuracy
from llmz.model import ModelConfig, TransformerLM
from llmz.tokenizer import Tokenizer

device = get_device()
model = TransformerLM(ModelConfig()).to(device)
model.load_state_dict(torch.load("runs/base/model.pt", map_location=device))
model.eval()
tokenizer = Tokenizer.load("data/tokenizer.json")
test, dev = load_mcq("test"), load_mcq("dev")

settings = [
    ("0-shot", [], "\n\n"),
    ("1-shot（dev 第 1 題）", dev[:1], "\n\n"),
    ("1-shot（dev 最後 1 題）", dev[-1:], "\n\n"),
    ("3-shot（dev 前 3 題）", dev[:3], "\n\n"),
    ("3-shot（dev 後 3 題）", dev[-3:], "\n\n"),
    ("5-shot", dev, "\n\n"),
    ("3-shot（dev 前 3 題），用 <|endoftext|> 分隔", dev[:3], "<|endoftext|>"),
    ("3-shot（dev 前 3 題），用空格分隔", dev[:3], " "),
]
print("設定：總和／平均每個 token／平均每個 byte")
for name, shots, sep in settings:
    accuracies = [mcq_accuracy(model, tokenizer, test, shots=shots, normalize=n, sep=sep) for n in ("none", "token", "byte")]
    print(f"{name}：" + "／".join(f"{a:.0%}" for a in accuracies))
```

M4 Max 上的輸出：

```text
設定：總和／平均每個 token／平均每個 byte
0-shot：81%／88%／86%
1-shot（dev 第 1 題）：79%／86%／85%
1-shot（dev 最後 1 題）：75%／84%／85%
3-shot（dev 前 3 題）：77%／87%／87%
3-shot（dev 後 3 題）：76%／85%／85%
5-shot：71%／85%／83%
3-shot（dev 前 3 題），用 <|endoftext|> 分隔：77%／85%／88%
3-shot（dev 前 3 題），用空格分隔：74%／84%／84%
```

「總和」與「平均每個 token」這兩欄，所有 few-shot 的設定都比 0-shot 低：總和從 81% 降到 71%～79%，平均每個 token 從 88% 降到 84%～87%。「平均每個 byte」那欄有兩個設定（87%、88%）比 0-shot 的 86% 高一點，其餘持平或更低。同樣放 1 個範例，換一個範例差了 4 個百分點（79% 對 75%）。這些差距大多落在 4.3 個百分點的雜訊範圍內，能下的結論只有：對這個模型，few-shot 沒有穩定的好處，而且分數會隨著挑哪幾個範例、怎麼分隔而晃動。

為什麼？這裡只是推測，我們沒有驗證：標準模型只讀過一篇一篇的故事，沒見過「幾個不相干的故事接在一起」這種文件，也不懂「前面的例子是在示範格式」。GPT-3 的論文發現，模型越大，從範例得到的好處越多；2,270 萬個參數的小模型離那個程度還很遠。

## 重點整理

- 多選題評分 = 對每個選項算 \\( \log p(\text{選項} \mid \text{題目}) \\)，挑最大的。`sequence_logprob(model, ids, mask)` 只把 `mask` 為 `True` 的 token 加起來，沒有 `no_grad`，第 14 章會重用。
- `choice_logprobs` 把題目與選項分開 encode、補齊成一個 batch；`pick_choice` 比總和，或比「除以長度」（token 數、byte 數）之後的平均。
- 總和偏愛短選項，平均偏愛每個 token 都好猜的長選項；用哪一種會改變分數，報告時要寫清楚。
- 標準模型在 100 題的題庫上：總和 81%、平均每個 token 88%、平均每個 byte 86%，隨機猜測是 25%。兩種算法有 22 題不一樣，平均對 13 題、總和對 6 題，這個差距還不足以斷定誰比較好。
- 這個模型用 few-shot 沒有穩定的好處（總和與每個 token 平均兩種算法全部比 0-shot 低，總和最低到 71%），分數也隨挑的範例與分隔而晃動；結論要放在 ±4.3 個百分點的雜訊裡看。
