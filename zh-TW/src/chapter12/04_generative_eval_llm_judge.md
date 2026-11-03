# 生成式評估與 LLM-as-judge

## 本集目標

對沒有選項、要模型自己寫答案的題目，用 exact match 打分並看清楚它太嚴的地方；再讓模型當評審（LLM-as-judge），認識評審的偏誤，以及「兩個順序都問、一致才算」這個補救。

## 正文

### 沒有選項的題目

上一集把每個選項接在題目後面算機率，模型不需要「說」任何話。但 GSM8K 那類題目（第 2 集）是讓模型自己寫出答案，我們再判斷它對不對。判斷的方式大致有三種：

- **規則**：把模型的輸出和標準答案比，最常見的是 exact match（完全一樣才算對）；
- **執行**：模型寫的是程式，就直接跑單元測試；
- **評審**：請另一個模型（或同一個）讀答案、打分數或比較好壞。

這一集談第一種與第三種。

### exact match

```python,ignore
{{#include ../../../src/llmz/evaluation.py:exact_match}}
```

直接比兩段字串太嚴：多一個句點、大小寫不同就算錯。所以比之前先把兩邊「正規化」：轉成小寫、去掉標點、去掉冠詞 a／an／the、把連續的空白併成一個。這是常見的做法（例如問答資料集 SQuAD 的評分程式也是這樣），但只是約定，換一套規則分數就會不同。

```python
from llmz.evaluation import exact_match, normalize_answer

pairs = [
    (" The Park!", "park"),
    ("a red ball.", "Red  ball"),
    ("red ball", "ball"),
    ("water", "cold water and drank it all."),
    (" sad.", "sad and started to cry."),
]
for prediction, reference in pairs:
    print(f"{normalize_answer(prediction)!r} 對 {normalize_answer(reference)!r}：{exact_match(prediction, reference)}")
```

執行結果：

```text
'park' 對 'park'：True
'red ball' 對 'red ball'：True
'red ball' 對 'ball'：False
'water' 對 'cold water and drank it all'：False
'sad' 對 'sad and started to cry'：False
```

前兩組只差在格式，正規化之後一樣，算對。後三組內容不一樣就是錯，不管意思接不接近：`water` 和 `cold water and drank it all.` 差在說得多不多，`sad.` 和 `sad and started to cry.` 也是。exact match 不懂意思，只比字面。

### 同一批題目，換成生成

把上一集的題庫換成生成式的考法：只給題目，讓標準模型用 greedy 往下寫（最多 16 個 token），只留第一句，再和正確選項比 exact match。

```python,norun
import re

import torch

from llmz.device import get_device
from llmz.evaluation import exact_match, load_mcq, mcq_accuracy
from llmz.inference import generate_cached
from llmz.model import ModelConfig, TransformerLM
from llmz.tokenizer import Tokenizer

device = get_device()
model = TransformerLM(ModelConfig()).to(device)
model.load_state_dict(torch.load("runs/base/model.pt", map_location=device))
model.eval()
tokenizer = Tokenizer.load("data/tokenizer.json")
test = load_mcq("test")

hits, shown = 0, 0
for it in test:
    ids = torch.tensor([tokenizer.encode(it["context"])], device=device)
    text = tokenizer.decode(generate_cached(model, ids, 16)[0, ids.shape[1]:].tolist())  # greedy，最多 16 個 token
    end = re.search(r"[.!?\n]", text)
    prediction = text[: end.end()] if end else text  # 只留第一句
    gold = it["choices"][it["answer"]]
    hits += exact_match(prediction, gold)
    if not exact_match(prediction, gold) and shown < 6:  # 印出前 6 個沒對上的
        print(f"{it['context']}\n  模型寫：{prediction!r}\n  標準答案：{gold!r}")
        shown += 1
print(f"exact match：{hits}/{len(test)}")
print(f"同一批題目用 loglikelihood 選擇題評分：{mcq_accuracy(model, tokenizer, test, normalize='token'):.0%}")
```

M4 Max 上的輸出（greedy 沒有隨機性；CPU 與 MPS 都是 18/100）：

```text
Tom was very hungry, so he opened the fridge and ate a
  模型寫：' big piece of cheese.'
  標準答案：' sandwich.'
Anna was thirsty after running in the sun. She poured a glass of
  模型寫：' water on the ground and gave it to Ben.'
  標準答案：' cold water and drank it all.'
The cake was in the oven. When it was ready, the whole kitchen smelled
  模型寫：' so good.'
  標準答案：' sweet and yummy.'
Sue was making soup for her family. She cut up some carrots and put them
  模型寫：' in a big bowl.'
  標準答案：' in the pot.'
Sam spilled his milk on the table. His mom gave him a
  模型寫：' big hug and said, "You are a good boy, Tim.'
  標準答案：' towel to wipe it up.'
Lily lost her favorite toy, so she felt
  模型寫：' very sad.'
  標準答案：' sad and started to cry.'
exact match：18/100
同一批題目用 loglikelihood 選擇題評分：88%
```

同一個模型、同一批題目：選擇題 88%，exact match 18%。這兩個數字不是在量同一件事。選擇題只要模型把正確選項排在其他選項前面；生成式要它自己想出和標準答案一模一樣的字。

沒對上的 6 題裡，`so good.`、`in a big bowl.`、`very sad.` 都是說得通的結尾，只是不是那個標準答案；`water on the ground and gave it to Ben.` 就不太通。exact match 分不出這兩種：它只看字面，說得通但說法不同的答案全部算錯。所以 18% 只是「模型寫出和標準答案一樣的字」的比例，不是「模型寫出合理結尾」的比例。

評分的細節也是規則的一部分：上面「只留第一句」、「最多 16 個 token」、「怎麼正規化」，換一種做法分數就會不同。論文報 exact match 時，這些都要寫清楚。放寬規則（例如只要答案出現在輸出裡就算對）會換來另一種錯：模型寫一長串，剛好提到答案，就被判對。這是這集的練習。

### 請模型當評審

規則寫不出來的題目（寫一段摘要、回答開放的問題），常見的做法是請另一個語言模型來看。有兩種問法：

- **逐一打分**：給評分標準，請評審對每個回答打 1 到 5 分；
- **成對比較**：給兩個回答，請評審說哪個比較好。

成對比較比較容易讓評審給出穩定的答案，所以先做這個。我們沒有外部的 LLM 可以呼叫（本書不連外部 API），只用手上的標準模型示範機制：把兩個結尾排成 A、B 兩個選項，看模型接在 `Answer:` 後面寫 ` A` 還是 ` B` 的機率比較大。這個評審會很差，但正好讓偏誤看得一清二楚。

```python,ignore
{{#include ../../../src/llmz/evaluation.py:judge}}
```

- `judge_prompt` 組出提示。選項開頭帶空格，和題庫的 `choices` 一樣。
- `judge_once` 重用上一集的 `choice_logprobs`：選項是兩個標籤 ` A` 與 ` B`，挑機率大的。回傳的 0 或 1 是「排在第幾個」。
- `judge_pair` 把兩個答案各排前面問一次。兩次選到同一個答案才算數，不一致就回傳 `None`。

先用一個玩具評審看這幾個函式在做什麼。它不看題目，永遠只想說 `A`：

```python
import torch
from torch import nn

from llmz.evaluation import judge_once, judge_pair, judge_prompt
from llmz.model import ModelConfig


class CharTokenizer:
    def encode(self, text):
        return [ord(ch) for ch in text]  # 每個字元一個 token


class AlwaysA(nn.Module):
    """只會說 A 的評審：不管前面是什麼，下一個字都是 A 的機率最大。"""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))  # choice_logprobs 要靠參數找出 device
        self.cfg = ModelConfig(vocab_size=128, context_length=512)

    def forward(self, ids):
        logits = torch.zeros(*ids.shape, 128)
        logits[..., ord("A")] = 5.0
        return logits


story = "Tom was hungry, so he ate"
print(judge_prompt(story, " a pie.", " a rock."))
print("-----")
tokenizer, judge = CharTokenizer(), AlwaysA()
print("pie 排前面：選第", judge_once(judge, tokenizer, story, " a pie.", " a rock."), "個")
print("pie 排後面：選第", judge_once(judge, tokenizer, story, " a rock.", " a pie."), "個")
print("兩個順序都問：", judge_pair(judge, tokenizer, story, " a pie.", " a rock."))
```

執行結果：

```text
Which ending makes more sense?

Story: Tom was hungry, so he ate
A. a pie.
B. a rock.

Answer:
-----
pie 排前面：選第 0 個
pie 排後面：選第 0 個
兩個順序都問： None
```

玩具評審兩次都選排第一個的，所以不管 pie 排前排後都選第 0 個，兩個順序的答案互相矛盾，`judge_pair` 回傳 `None`：這個評審沒給出任何有用的資訊，被抓到了。

### 標準模型當評審

換成真的標準模型。每題拿題庫的正確結尾和一個隨機挑的錯誤結尾（固定種子）湊成一組，正確的排前面問一次、排後面問一次；再把標籤換成 `B`／`A`、`1`／`2`，看結果是跟著位置還是跟著標籤字走：

```python,norun
import random

import torch

from llmz.device import get_device
from llmz.evaluation import judge_once, judge_pair, load_mcq
from llmz.model import ModelConfig, TransformerLM
from llmz.tokenizer import Tokenizer

device = get_device()
model = TransformerLM(ModelConfig()).to(device)
model.load_state_dict(torch.load("runs/base/model.pt", map_location=device))
model.eval()
tokenizer = Tokenizer.load("data/tokenizer.json")
test = load_mcq("test")

# 每題一組：正確的結尾，對一個隨機挑的錯誤結尾
rng = random.Random(0)
pairs = []
for it in test:
    wrong = rng.choice([c for i, c in enumerate(it["choices"]) if i != it["answer"]])
    pairs.append((it["context"], it["choices"][it["answer"]], wrong))

for labels in [("A", "B"), ("B", "A"), ("1", "2")]:
    first_picked = right = 0
    for context, good, bad in pairs:
        gold_first = judge_once(model, tokenizer, context, good, bad, labels)  # 好的排前面：選 0 才對
        gold_second = judge_once(model, tokenizer, context, bad, good, labels)  # 好的排後面：選 1 才對
        first_picked += (gold_first == 0) + (gold_second == 0)
        right += (gold_first == 0) + (gold_second == 1)
    outcomes = [judge_pair(model, tokenizer, context, good, bad, labels) for context, good, bad in pairs]
    print(f"標籤 {labels[0]}／{labels[1]}：選了排第一的 {first_picked}/200 次，答對 {right}/200；"
          f"兩個順序一致 {sum(o is not None for o in outcomes)}/100 題")
```

M4 Max 上的輸出（沒有隨機性，CPU 與 MPS 相同）：

```text
標籤 A／B：選了排第一的 200/200 次，答對 100/200；兩個順序一致 0/100 題
標籤 B／A：選了排第一的 0/200 次，答對 100/200；兩個順序一致 0/100 題
標籤 1／2：選了排第一的 0/200 次，答對 100/200；兩個順序一致 0/100 題
```

- 標籤是 `A`／`B` 時，它 200 次都選排第一的；標籤對調成 `B`／`A`，它 200 次都選排第二的（也就是標籤 `A`）；用 `1`／`2` 則永遠選 `2`。**跟著走的是標籤字，不是位置**：這個模型永遠偏好某一個標籤字。
- 好的結尾在前、在後各 100 次，所以永遠只選一個標籤，剛好答對一半，100/200，和亂猜一樣。
- 兩個順序一致的是 0 題：`judge_pair` 把全部 100 題都擋下來，回傳 `None`。

同一個模型選結尾是有本事的（上一集 88%），但「在 A 和 B 之間選一個」這種問法它沒讀過：TinyStories 裡沒有這樣的格式，它也還不會照指令行事（第 14 章會正式介紹怎麼教它）。原因只是我們的推測，這裡只量到結果。

### 評審的偏誤

真正的 LLM 評審比這個玩具強得多，偏誤也不會這麼極端，但有些毛病是共通的。Zheng 等人（2023，MT-Bench 與 Chatbot Arena 的論文）整理出三種常見的偏誤：

- **位置偏誤**：偏好排在前面（或後面）的答案，和內容無關。
- **長度偏誤**：偏好比較長的答案，即使多出來的只是廢話。
- **自我偏好**：偏好和評審同一個模型寫出來的答案。

我們剛量到的是同一類問題的極端版本：評審的選擇和答案的好壞無關，只受提示的格式影響。補救的方式：

- **交換順序，一致才算**：就是 `judge_pair`。兩個順序給出不同的答案，代表評審的選擇受位置或標籤影響，這一題不採用（或當成平手）。
- **評審與被評的模型分開**：避開自我偏好。
- **拿人工標記的題目驗證評審**：先看評審和人給的答案有多常一致，再相信它對沒有標準答案的題目給的分數。
- 長度偏誤要在評分標準裡寫明「不要因為長就給高分」，或把比較的答案控制在差不多長。

沒有哪個補救能保證評審是對的，只能降低偏誤。LLM-as-judge 的分數是「這個評審的意見」，不是「答案的真實品質」。

## 重點整理

- 生成式題目沒有選項，可以用規則（exact match）、執行（跑測試）或評審（另一個模型）打分。
- `exact_match(prediction, reference)` 先正規化（小寫、去標點與冠詞 a／an／the、併空白）再比；格式差異不算錯，說法不同的答案全算錯。
- 標準模型在同一批 100 題上：選擇題 88%（每個 token 平均），生成式 exact match 18/100。exact match 只看字面，說得通但說法不同的結尾都算錯。
- LLM-as-judge 有逐一打分與成對比較兩種問法；`judge_prompt`、`judge_once`、`judge_pair` 用 `choice_logprobs` 比 ` A` 與 ` B` 的機率。
- 標準模型當評審，在三種標籤下答對率都是 100/200，選的是標籤字而不是內容；`judge_pair` 兩個順序都問、一致才算，把這 100 題全部擋下來。論文整理的偏誤有位置、長度與自我偏好三種。
