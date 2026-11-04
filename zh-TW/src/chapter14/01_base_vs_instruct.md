# base model 與指令模型

## 本集目標

看到只用預訓練練出來的 base model 面對一道題目時怎麼反應：它接著寫，不回答。知道指令模型和它差在後訓練，並準備好這一章會用的、答案可以自動檢查的題目。

## 正文

### base model 只會接著寫

第 5 章的訓練目標只有一個：預測下一個 token。這樣練出來的模型叫 **base model**（基礎模型）。給它一段文字，它就從這裡接著寫，寫出「像訓練資料的下文」。

我們的訓練資料是 TinyStories，裡面全是故事，沒有「一個問題後面跟著它的答案」這種文字。所以把問題丟給它，它把問題當成故事的開頭：

```python,norun
import torch

from llmz.generate import generate
from llmz.model import ModelConfig, TransformerLM
from llmz.posttrain import make_tasks
from llmz.tokenizer import Tokenizer

model = TransformerLM(ModelConfig())
model.load_state_dict(torch.load("runs/base/model.pt", map_location="cpu"))
model.eval()
tokenizer = Tokenizer.load("data/tokenizer.json")

valid = make_tasks(200, seed=1, split="valid")  # 下面會介紹 make_tasks
for kind in ["add", "color", "who"]:
    task = next(t for t in valid if t.kind == kind)
    print(task.question)
    print("  base model 接著寫：", repr(generate(model, tokenizer, task.question, 30, temperature=0)[len(task.question):]))
    print("  標準答案：", task.answer)
```

輸出（greedy 沒有隨機性，CPU 上跑）：

```text
Tim has two balls. He finds five more. How many balls does Tim have now?
  base model 接著寫： ' He likes balls. He thinks they are fun. He wants to play with them.\nHe runs to the tree and grabs the balls. He grabs'
  標準答案： seven
Max has a pink flower and a white ball. What color is the flower?
  base model 接著寫： '\n"Look, Lily, this is the flower. It is so pretty and pretty. Do you want to see it?" Max says.\n"'
  標準答案： pink
Ben has an apple, Anna has a cup and Tom has a kite. Who has the cup?
  base model 接著寫： '\n"Look, Anna, I have a cup of juice. It is very hot," Ben says. He looks at the cup and sees a picture'
  標準答案： Anna
```

它寫出來的故事很通順，這是它的本事。但它沒有回答：第一題沒有說 seven，第二題沒有說 pink。第 12 章第 4 集量到同樣的事：同一個模型選擇題有 88%，要它自己寫結尾只有 18/100 完全對上，把它當評審更是永遠只選同一個標籤字。那時候留下一句「第 14 章會正式介紹怎麼教它」，就是這一章的事。

### 一份可以自動打分的考卷

要量「有沒有照指令回答」，需要題目，也需要一個不靠人讀就能判斷對錯的方法。第 12 章的選擇題集只有 100 題，而且 SFT 還需要大量的訓練題。所以這一章改用程式依固定種子產生題目，答案是從題目裡的數字或名字直接算出來的，用字都是 TinyStories 風格的簡單英文：

```python,ignore
{{#include ../../../src/llmz/posttrain.py:task_makers}}
```

- `Task` 是一個 `NamedTuple`（第 13 章第 1 集見過）：題型 `kind`、使用者說的話 `question`、標準答案 `answer`。
- 五種題型：`add`（加法）、`subtract`（減法）、`count`（數有幾樣東西）、`color`（兩樣東西，問其中一樣的顏色）、`who`（三個人各有一樣東西，問誰有哪一樣）。數字都寫成英文單字（`seven`）。
- `article` 與 `some` 處理 `a`／`an` 與單複數，題目才不會出現 `a apple` 或 `one apples`。
- 每個 `make_*` 函式拿一個 `random.Random`（第 12 章第 5 集用過）決定名字、東西與數字，答案在同一個函式裡算出來。

```python,ignore
{{#include ../../../src/llmz/posttrain.py:task_api}}
```

- `make_tasks(n, seed)` 每次依 `seed` 產生同樣的 `n` 題。`kinds` 可以只選幾種題型。
- `split` 把題目分成 `"train"` 與 `"valid"`：看題目文字的 `zlib.crc32`（把一段 bytes 變成一個固定的整數，同樣的文字永遠得到同樣的數字，不像 Python 內建的 `hash`，每次執行都不同）除以 10 的餘數，餘 0 的是 `"valid"`，約佔十分之一。因為決定的是題目的文字，同一題不管用哪個種子產生，永遠在同一邊，`"train"` 與 `"valid"` 絕對不會有一模一樣的題目。第 12 章第 5 集看過題目混進訓練資料會怎樣，這裡從一開始就避免。
- `reward(task, reply)` 答對給 1.0、答錯給 0.0，比對用的就是第 12 章的 `exact_match`。第 15 章會用它當 RL 的獎勵，所以這個函式叫 `reward`。

```python
from llmz.posttrain import KINDS, make_tasks, reward

tasks = make_tasks(1000, seed=0)
for kind in KINDS:
    mine = [t for t in tasks if t.kind == kind]
    print(f"{kind:9s}{len(mine)} 題：{mine[0].question}  ->  {mine[0].answer}")

print("-----")
task = tasks[0]
for reply in [task.answer, f" {task.answer.capitalize()}.", f"{task.answer} and more", "I do not know"]:
    print(f"{reply!r:18s}{reward(task, reply)}")

print("-----")
train = {t.question for t in make_tasks(2000, seed=0, split="train")}
valid = {t.question for t in make_tasks(2000, seed=1, split="valid")}
print(f"train 有 {len(train)} 種題目，valid 有 {len(valid)} 種，兩邊重複的有 {len(train & valid)} 種")
```

執行結果：

```text
add      190 題：Ben has four books. He finds five more. How many books does Ben have now?  ->  nine
subtract 205 題：Ben has four kites. He gives two to Tim. How many kites does Ben have left?  ->  two
count    196 題：Ben has a cup, a toy and a kite. How many things does Ben have?  ->  three
color    214 題：Lily has a white apple and a black flower. What color is the flower?  ->  black
who      195 題：Mia has a kite, Ben has a flower and Tim has a ball. Who has the ball?  ->  Tim
-----
'black'           1.0
' Black.'         1.0
'black and more'  0.0
'I do not know'   0.0
-----
train 有 1959 種題目，valid 有 1730 種，兩邊重複的有 0 種
```

答案寫成 ` Black.` 也算對，因為 `exact_match` 先把兩邊正規化；多寫一句 `and more` 就不算了。這是這個獎勵的規則：只要回覆是一個字的答案，不多不少。

### 對 base model 考一次

把 200 道 `"valid"` 題丟給 base model，greedy 往下寫 16 個 token，留下第一句，用 `reward` 打分。第 12 章第 2 集說過 few-shot 可以讓模型看懂格式，所以也試一次：先放 3 題「題目加答案」當範例。

```python,norun
import re

import torch

from llmz.inference import generate_cached
from llmz.model import ModelConfig, TransformerLM
from llmz.posttrain import make_tasks, reward
from llmz.tokenizer import Tokenizer

model = TransformerLM(ModelConfig())
model.load_state_dict(torch.load("runs/base/model.pt", map_location="cpu"))
model.eval()
tokenizer = Tokenizer.load("data/tokenizer.json")
valid = make_tasks(200, seed=1, split="valid")


def first_sentence(text):
    text = text.strip()
    end = re.search(r"[.!?\n]", text)
    return text[: end.start()] if end else text


def reply(prompt, n=16):
    ids = tokenizer.encode(prompt)
    new = generate_cached(model, torch.tensor([ids]), n)[0, len(ids):].tolist()
    return first_sentence(tokenizer.decode(new))


def contains(task, text):
    return re.search(rf"\b{task.answer}\b", text, re.I) is not None


shots = make_tasks(3, seed=100)
prefix = "".join(f"Question: {s.question}\nAnswer: {s.answer}\n\n" for s in shots)
for name, make_prompt in [("只給題目", lambda t: t.question), ("3 個範例再給題目", lambda t: prefix + f"Question: {t.question}\nAnswer:")]:
    replies = [reply(make_prompt(t)) for t in valid]
    print(f"{name}：exact match {sum(reward(t, r) for t, r in zip(valid, replies)):.0f}/{len(valid)}，"
          f"答案的字有出現在回覆裡 {sum(contains(t, r) for t, r in zip(valid, replies))}/{len(valid)}")
```

輸出（greedy，CPU 上跑）：

```text
只給題目：exact match 0/200，答案的字有出現在回覆裡 17/200
3 個範例再給題目：exact match 0/200，答案的字有出現在回覆裡 12/200
```

200 題答對 0 題。就算把標準放寬到「答案的字有出現在回覆裡就算」，也只有 6% 到 9%，而且 17 題裡有 12 題是 `who`：它在故事裡剛好叫了那個人的名字（`"Look, Anna, I have a cup of juice`），不是答出來的。放了 3 個範例也沒有好處：這個 22M 參數、只讀過故事的模型，沒有在 context 裡學會新格式的本事（第 12 章第 2 集量到 few-shot 對它沒有穩定的好處，這裡是同一件事）。

### 指令模型多了什麼

**指令模型**（instruction-tuned model，ChatGPT 這類你會直接拿來對話的模型）和 base model 用的是同一種架構，同一種「預測下一個 token」的 loss。差別在資料與流程：預訓練之後，再接幾段**後訓練**：

| 階段 | 看什麼資料 | 教它什麼 | 本書 |
| --- | --- | --- | --- |
| 預訓練 | 大量文字 | 接著寫 | 第 5 章 |
| SFT | 一來一往的（指令，回覆） | 被問到就照格式回答 | 這一章第 2～3 集 |
| 偏好學習 | 成對的（好回答，壞回答） | 兩個回答裡，更像好的那個 | 這一章第 4～5 集 |
| RL | 能自動判斷對錯的獎勵 | 試錯，答對的做法加強 | 第 15 章 |

這一集到第 3 集只做第二列。SFT 的想法是：既然 base model 讀了文字就會模仿文字，那就拿一批「指令、回覆」排成一串的文字讓它讀，讀得夠多，它就會模仿這個格式。接下來兩集分別看這批文字怎麼排（chat template 與 loss mask），以及真的讀一次之後正確率變多少。

## 重點整理

- base model 只被訓練成預測下一個 token，面對問題只會當成故事的開頭接著寫。標準模型在 200 題上 exact match 是 0/200，放 3 個範例也是 0/200。
- 指令模型和 base model 架構與 loss 都一樣，差在後訓練：SFT（教格式）、偏好學習（比較好壞）、RL（用獎勵試錯）。
- `make_tasks(n, seed, split, kinds)` 依固定種子產生五種題型（加、減、數數量、顏色、誰有什麼）的題目，答案由題目的數字或名字直接算出；`"train"` 與 `"valid"` 用題目文字的 `zlib.crc32` 分開，所以不會有重複的題目。
- `reward(task, reply)` 用第 12 章的 `exact_match`：答對 1.0、答錯 0.0。這個「答案可以自動判斷」的設計，第 15 章的 RL 會直接拿來用。
