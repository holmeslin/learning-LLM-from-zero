# benchmark 與 few-shot

## 本集目標

知道 benchmark 由哪幾樣東西組成，認識本書附的小題庫，算出隨機猜測的分數與雜訊，並學會把 zero-shot 與 few-shot 的提示組出來。

## 正文

### benchmark 是什麼

benchmark（基準測試）就是一份固定的考卷加一套固定的評分方式：

- **題目**：事先收集好、不隨模型改變；
- **提示的格式**：題目怎麼排成一段文字餵給模型；
- **評分規則**：怎麼從模型的輸出判斷對錯，再算成一個分數。

三樣都固定，不同的人、不同的模型，分數才比得起來。論文與排行榜常見的有：MMLU（57 個科目的四選一題目）、HellaSwag（從四個句子裡選出最合理的續寫）、GSM8K（小學程度的數學應用題，要寫出解題過程再給答案）。前兩個是選擇題，模型只要給每個選項打分；GSM8K 沒有選項，得讓模型自己生成答案，第 4 集會談。這些題庫對 2,270 萬個參數、只讀過兒童故事的模型來說太難了：分數會和亂猜差不多，看不出任何差別。所以本書自己準備一份題庫。

### 本書的小題庫

`src/llmz/datasets/tinystories_mcq.jsonl` 有 105 題四選一，題目與選項全部是為本書原創、用 TinyStories 那種簡單英文寫的，授權和本書的程式碼一樣（CC0）。每題是一段故事的開頭，要選出最合理的結尾。檔案每行是一題 JSON，讀它的函式很短：

```python,ignore
{{#include ../../../src/llmz/evaluation.py:load_mcq}}
```

`split` 選 `"test"` 或 `"dev"`，不給就是全部。讀進來看看：

```python
from collections import Counter

from llmz.evaluation import load_mcq

test, dev = load_mcq("test"), load_mcq("dev")
print(len(test), len(dev))
item = dev[0]
print(item["context"])
for i, choice in enumerate(item["choices"]):
    mark = "  <- 答案" if i == item["answer"] else ""
    print(f"{'ABCD'[i]}.{choice}{mark}")
print(sorted(Counter(it["answer"] for it in test).items()))
```

執行結果：

```text
100 5
Mia felt very sleepy after a long day, so she went to
A. the park to play.
B. school.
C. bed and closed her eyes.  <- 答案
D. the store to buy a new hat for her dog.
[(0, 25), (1, 25), (2, 25), (3, 25)]
```

題庫分成兩份：`test` 有 100 題，用來打分；`dev` 有 5 題，只拿來當 few-shot 的範例（下面會說為什麼要分開）。每個選項開頭都帶一個空格，因為它是接在題目後面的下一段文字。正確答案的位置打散了，A、B、C、D 在 `test` 裡各 25 題，所以「永遠選 C」只能拿到 25%。

### 隨機猜測的分數與雜訊

四選一的題目亂猜，期望的準確率是 25%。但只有 100 題時，分數會上下跳多少？模擬一萬個隨機猜的人：

```python
import torch

from llmz.evaluation import load_mcq

test = load_mcq("test")
answers = torch.tensor([it["answer"] for it in test])
torch.manual_seed(0)
guesses = torch.randint(0, 4, (10000, len(test)))  # 一萬個人，每個人對 100 題各猜一個選項
scores = (guesses == answers).sum(dim=1)  # 每個人答對幾題
print(f"平均 {scores.float().mean().item():.1f} 題，標準差 {scores.float().std().item():.2f} 題")
print(f"最高 {scores.max().item()} 題；答對超過 33 題的有 {(scores > 33).sum().item()} 人")
print(f"公式 sqrt(n p (1-p))：{(len(test) * 0.25 * 0.75) ** 0.5:.2f} 題")
```

執行結果：

```text
平均 25.1 題，標準差 4.29 題
最高 41 題；答對超過 33 題的有 296 人
公式 sqrt(n p (1-p))：4.33 題
```

亂猜的標準差約 4.3 題，也就是 4.3 個百分點：一萬個人裡有 296 人（約 3%）僥倖拿到 34 題以上，最高的猜到 41 題。所以在 100 題的題庫上：

- 要比 25% 高出 2 個標準差（約 34%）以上，才算得上「看得出不是亂猜」；
- 兩個模型差幾個百分點，多半只是運氣。MMLU 這類題庫有上萬題，標準差只剩零點幾個百分點，差 1～2 個百分點才有意義。

### zero-shot 與 few-shot

同一題可以用不同的方式問模型：

- **zero-shot**：提示裡只有題目本身；
- **one-shot／few-shot**：題目前面先放 1 個或幾個「題目加正確答案」的範例。

GPT-3 的論文（Brown 等人，2020）把這種不更新任何參數、只靠提示裡的範例讓模型抓住格式與意圖的做法稱為 in-context learning。範例從 `dev` 取，不能從 `test` 取：如果範例本身就是要考的題目，模型等於先看過答案。

```python,ignore
{{#include ../../../src/llmz/evaluation.py:build_prompt}}
```

每個範例是題目接上正確的選項，範例之間與最後的題目之間用 `sep` 隔開（預設一個空行）。`shots` 是空的就是 zero-shot：

```python
from llmz.evaluation import build_prompt, load_mcq

test, dev = load_mcq("test"), load_mcq("dev")
print(build_prompt(test[0]["context"]))
print("-----")
print(build_prompt(test[0]["context"], dev[:2]))
```

執行結果：

```text
Tom was very hungry, so he opened the fridge and ate a
-----
Mia felt very sleepy after a long day, so she went to bed and closed her eyes.

Ben dropped his ice cream on the ground. He felt sad.

Tom was very hungry, so he opened the fridge and ate a
```

選項不放進提示。我們不是請模型「說出答案是 A 還是 C」，而是下一集要做的：把每個選項接在題目後面，看模型覺得哪一個最像接下來會出現的文字。範例也長得像一小段完整的故事，而不是考卷。

### 範例要放得下

模型一次最多看 256 個 token（`context_length`），範例越多，留給題目與選項的位置越少。用標準 tokenizer 算每個設定下最長的那一題（取字元最長的選項）有多少 token：

```python,norun
from llmz.evaluation import build_prompt, load_mcq
from llmz.tokenizer import Tokenizer

tokenizer = Tokenizer.load("data/tokenizer.json")
test, dev = load_mcq("test"), load_mcq("dev")
for k in range(len(dev) + 1):
    # 每一題「prompt + 最長的選項」的 token 數，取所有題目裡最大的
    longest = max(len(tokenizer.encode(build_prompt(it["context"], dev[:k]) + max(it["choices"], key=len))) for it in test)
    print(f"{k}-shot：最長的一題有 {longest} 個 token")
```

M4 Max 上的輸出（token 數每次相同）：

```text
0-shot：最長的一題有 27 個 token
1-shot：最長的一題有 48 個 token
2-shot：最長的一題有 63 個 token
3-shot：最長的一題有 86 個 token
4-shot：最長的一題有 102 個 token
5-shot：最長的一題有 121 個 token
```

5 個範例加最長的一題只有 121 個 token，遠低於 256，這份題庫全部放得下。真實的 benchmark 題目長得多（MMLU 的題目常有一整段文字），範例放幾個要看模型的 context 長度。

### 沒有唯一的答案

few-shot 的分數取決於放幾個範例、挑哪幾個、用什麼順序、用什麼分隔，這些全都會改變分數。所以論文報 benchmark 數字時，一定要同時寫出 shot 數與提示的格式；兩個數字只有在這些都相同時才能比。下一集會在真正的模型上量。

## 重點整理

- benchmark = 固定的題目、固定的提示格式、固定的評分規則；三樣都固定，分數才比得起來。
- 本書的題庫在 `llmz/datasets/tinystories_mcq.jsonl`：`test` 100 題打分、`dev` 5 題當 few-shot 範例，四選一，正確答案的位置各 25 題。
- 四選一亂猜是 25%；100 題時標準差約 4.3 個百分點，要高出約 9 個百分點才看得出不是亂猜，差幾個百分點的兩個模型不能說誰比較好。
- zero-shot 提示只有題目；few-shot 前面加幾個「題目加正確答案」的範例，範例要從 `dev` 取。`llmz.evaluation.build_prompt` 組出提示。
- few-shot 的分數取決於範例的數量、選擇、順序與分隔，報分數要同時寫出這些。
