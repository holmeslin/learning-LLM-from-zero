# chat template 與 loss mask

## 本集目標

把一來一往的對話排成一串 token，並標出哪些 token 要算 loss：只有助理的回覆。用測試確認提示的部分沒有被算進去。

## 正文

### 把對話排成一串文字

語言模型只吃一串 token。一段對話有「誰說的」這個資訊，得靠約定排進這一串裡，這個約定叫 **chat template**。我們的約定很簡單：

```text
User: Tim has two balls. He finds five more. How many balls does Tim have now?
Assistant: seven<|endoftext|>
```

使用者的訊息是 `User: ` 加內容加換行；助理的回覆是 `Assistant:`、一個空格、內容，最後接 `<|endoftext|>`。

```python,ignore
{{#include ../../../src/llmz/posttrain.py:chat_template}}
```

`chat_pieces` 把訊息列表切成幾段文字，每段附一個 `True` 或 `False`：這一段要不要算 loss。`encode_chat` 把每一段分開 encode 再接起來，每個 token 的 mask 就是它那一段的標記。

```python
from llmz.posttrain import chat_pieces

messages = [
    {"role": "user", "content": "Tim has two balls. He finds five more. How many balls does Tim have now?"},
    {"role": "assistant", "content": "seven"},
]
for text, train in chat_pieces(messages):
    print(train, repr(text))
print("-----")
for text, train in chat_pieces(messages[:1], generation_prompt=True):
    print(train, repr(text))
```

執行結果：

```text
False 'User: Tim has two balls. He finds five more. How many balls does Tim have now?\n'
False 'Assistant:'
True ' seven<|endoftext|>'
-----
False 'User: Tim has two balls. He finds five more. How many balls does Tim have now?\n'
False 'Assistant:'
```

兩個模式：

- 訓練時給完整的對話。只有最後一段，也就是助理的回覆加結尾，是 `True`。
- 要讓模型回答時（`generation_prompt=True`）只給使用者的訊息，後面接一個 `Assistant:`，請模型從這裡開始寫。全部都是 `False`：這些是我們給的，不是模型要學的。`Assistant:` 這幾個 token 屬於提示：推論時它由我們填，模型不需要自己寫出來。

### 為什麼不加新的特殊 token

真實的模型多半會替「角色」與「一輪結束」新增專用的特殊 token。我們不加，只用兩樣已經有的東西：

- 角色標記 `User:`、`Assistant:` 就是一般文字，用第 3 章的 tokenizer 照常 encode。
- 結尾用第 3 章第 5 集的 `<|endoftext|>`。預訓練時它是「這份文件結束了」，我們把它當成「助理這一輪說完了」。

這樣詞彙表還是 10000 個，第 3 到第 5 章的 tokenizer 與模型原封不動沿用。新增 token 的話，embedding 與輸出層都要多幾列，新的那幾列是隨機初始化的，模型從沒看過。代價是角色標記不是專用的 token，是從一般字詞拼出來的，使用者的訊息裡如果剛好寫了一行 `Assistant:`，模型會分不出這是格式還是內容；專用的特殊 token 沒有這個問題。我們的題目是程式產生的，不會有這種情況。

結尾要算 loss。如果不教它在回覆的最後寫 `<|endoftext|>`，它寫完 `seven` 之後不知道要停，會接著編下去。

### 每個 token 的 mask

用第 3 章的 byte-level tokenizer（沒有 merges，一個 byte 一個 token）看每個 token 的 mask，一目了然：

```python
from llmz.posttrain import encode_chat
from llmz.tokenizer import Tokenizer, base_vocab

END = "<|endoftext|>"
tokenizer = Tokenizer(base_vocab([END]), [], [END])
ids, mask = encode_chat(tokenizer, [{"role": "user", "content": "2+3?"}, {"role": "assistant", "content": "5"}])
print(len(ids), "個 token，其中", sum(mask), "個算 loss")
print([(tokenizer.decode([i]), int(m)) for i, m in zip(ids, mask)])
```

執行結果：

```text
24 個 token，其中 3 個算 loss
[('U', 0), ('s', 0), ('e', 0), ('r', 0), (':', 0), (' ', 0), ('2', 0), ('+', 0), ('3', 0), ('?', 0), ('\n', 0), ('A', 0), ('s', 0), ('s', 0), ('i', 0), ('s', 0), ('t', 0), ('a', 0), ('n', 0), ('t', 0), (':', 0), (' ', 1), ('5', 1), ('<|endoftext|>', 1)]
```

最後三個：空格、`5`、`<|endoftext|>`。`Assistant:` 的冒號是 0，空格是 1：回覆從空格開始，因為真實的 tokenizer 裡 ` seven` 是一個帶空格的 token（和第 12 章第 3 集的選項一樣）。

換成真正的 tokenizer，用 20,000 道 `"train"` 題看：

```python,norun
from llmz.posttrain import chat_pieces, encode_chat, make_tasks, task_messages
from llmz.tokenizer import Tokenizer

tokenizer = Tokenizer.load("data/tokenizer.json")
tasks = make_tasks(20000, seed=0)
ids, mask = encode_chat(tokenizer, task_messages(tasks[0], with_answer=True))
print("|".join(tokenizer.decode([i]).replace("\n", "⏎") for i in ids))
print("算 loss 的 token：", [tokenizer.decode([i]) for i, m in zip(ids, mask) if m])

mismatch = total = trained = 0
for task in tasks:
    messages = task_messages(task, with_answer=True)
    ids, mask = encode_chat(tokenizer, messages)
    mismatch += ids != tokenizer.encode("".join(text for text, _ in chat_pieces(messages)))  # 整串一起 encode
    total += len(ids)
    trained += sum(mask)
print(f"分段 encode 與整串 encode 不一樣的：{mismatch}/{len(tasks)} 題")
print(f"每題平均 {total / len(tasks):.1f} 個 token，其中 {trained / len(tasks):.1f} 個算 loss（{trained / total:.1%}）")
```

輸出（沒有隨機性）：

```text
U|ser|:| Lily| has| a| white| apple| and| a| black| flower|.| What| color| is| the| flower|?|⏎|A|ss|ist|ant|:| black|<|endoftext|>
算 loss 的 token： [' black', '<|endoftext|>']
分段 encode 與整串 encode 不一樣的：0/20000 題
每題平均 29.5 個 token，其中 2.0 個算 loss（6.8%）
```

兩件事。第一，分段 encode 的結果和整串 encode 完全一樣：每一段的交界（換行與 `Assistant` 之間、冒號與空格之間）本來就是第 3 章第 4 集的 pre-tokenization 會切開的地方，所以分開 encode 不會改變 token 怎麼切，`ids` 和 `mask` 才對得上。第二，每題大約 30 個 token，算 loss 的只有答案與結尾 2 個，不到 7%。

### loss mask：只算助理說的話

SFT 的 loss 和預訓練一樣，是「每個 token 預測下一個 token 的 cross-entropy」。差別只有一個：有些位置不算。

```python,ignore
{{#include ../../../src/llmz/posttrain.py:sft_loss}}
```

- `pad_batch` 把一批長短不一的 `(ids, mask)` 疊成矩陣。短的往右補 0，補的位置 mask 是 `False`。模型是 causal 的，右邊補的 token 不會影響左邊，所以不用另外遮（第 12 章第 3 集的 `choice_logprobs` 也這樣做）。
- `sft_loss` 重用第 12 章的 `sequence_logprob`：它回傳每條序列裡 mask 為 `True` 的 token 的 log 機率總和（沒有 `no_grad`，就是為了這一章能 `backward()`）。加上負號、除以整批被算的 token 數，就是平均 loss。第 0 個 token 沒有前文可預測，永遠不算。

用一個機率可以手算的 bigram 模型，確認它和手算一致：

```python
import math

import torch
from torch import nn

from llmz.posttrain import pad_batch, sft_loss

# 三個符號 0、1、2 的 bigram「模型」：P[a][b] 是「現在是 a、下一個是 b」的機率，所以 loss 可以手算（和第 12 章第 3 集一樣）。
P = torch.tensor([[0.1, 0.6, 0.3], [0.2, 0.2, 0.6], [0.5, 0.25, 0.25]])


class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(P.log())

    def forward(self, ids):
        return self.logits[ids]  # 每個位置的 logits 只看這個位置的 token


# 兩條序列：前面兩個 token 是提示，後面是回覆。第二條短一個，要補一格。
ids, mask = pad_batch([([0, 1, 2, 0], [False, False, True, True]), ([2, 0, 1], [False, False, True])])
print(ids)
print(mask)

model = BigramLM()
only_reply = -(math.log(0.6) + math.log(0.5) + math.log(0.6)) / 3  # 1->2、2->0、0->1 三個被算的 token
everything = -(math.log(0.6) * 3 + math.log(0.5) * 2) / 5  # 再加上提示裡的 0->1、2->0
print(f"套 mask：{sft_loss(model, ids, mask).item():.4f}，手算 {only_reply:.4f}")
all_but_padding = torch.tensor([[True] * 4, [True] * 3 + [False]])
print(f"整串都算：{sft_loss(model, ids, all_but_padding).item():.4f}，手算 {everything:.4f}")
```

執行結果：

```text
tensor([[0, 1, 2, 0],
        [2, 0, 1, 0]])
tensor([[False, False,  True,  True],
        [False, False,  True, False]])
套 mask：0.5716，手算 0.5716
整串都算：0.5838，手算 0.5838
```

套 mask 與不套 mask 算出的 loss 不一樣，兩個都和手算一致。mask 只決定「哪些位置的預測錯誤算進 loss」，不是把提示藏起來：模型預測回覆的時候，整段提示還是它的 context，看得一清二楚。

為什麼只算助理的回覆？

- 提示是使用者給的，推論時我們會直接填好，不需要模型寫。把它算進 loss，模型就多花力氣學「怎麼寫出一道題目」。
- 回覆只佔 7%。不套 mask 的話，被平均的 token 有 93% 是提示，答案那 2 個 token 的比重很小。
- 第 3 集會做這個對照實驗，量出不套 mask 會差多少。

`tests/test_posttrain.py` 測的就是這幾件事：`chat_pieces` 只有助理的那一段是 `True`；`sft_loss` 等於手算的值；mask 外的 token 換掉，loss 不變；梯度只流到被算的那些位置。

## 重點整理

- chat template 是把訊息排成一串文字的約定。本書的格式是 `User: …\n` 與 `Assistant: …<|endoftext|>`，用一般文字當角色標記、沿用 `<|endoftext|>` 當結尾，不新增特殊 token，所以詞彙表與模型都不用改。
- `chat_pieces` 把訊息切成（文字，要不要算 loss）；`encode_chat` 分段 encode 並產生對齊的 `ids` 與 `mask`；`generation_prompt=True` 在最後接 `Assistant:` 請模型開始寫。
- 助理的回覆與結尾 `<|endoftext|>` 算 loss，提示不算；結尾不算的話模型不知道要停。標準 tokenizer 下一題約 30 個 token，只有 2 個（6.8%）算 loss。
- `pad_batch` 把序列往右補 0；`sft_loss(model, ids, mask)` 是 `sequence_logprob` 取負號、除以被算的 token 數；它和手算一致，mask 只決定誰算進 loss，不影響模型看到的 context。
