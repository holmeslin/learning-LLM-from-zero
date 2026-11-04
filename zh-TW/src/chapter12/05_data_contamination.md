# 資料污染

## 本集目標

知道資料污染（benchmark 的題目早就出現在訓練資料裡）為什麼會讓分數虛高，自己寫出 n-gram 重疊偵測，在標準語料上實跑一次，並看清楚這個偵測抓得到什麼、抓不到什麼。

## 正文

### 什麼是資料污染

benchmark 的分數想回答的是「模型有沒有這個能力」。如果題目甚至答案已經在訓練資料裡，模型可以不靠能力、只靠背下來拿高分，分數就虛高了，這叫**資料污染**（data contamination）。

大模型的訓練資料多半是從網路上爬來的（第 13 章會正式介紹怎麼處理），而 benchmark 的題目一公開，就會出現在網頁、論壇與論文裡，有機會被爬進訓練資料。模型不是故意作弊，只是讀過而已。

### 用 n-gram 重疊偵測

最直接的檢查是看題目的文字有沒有出現在訓練資料裡。我們把題目切成每 \\( n \\) 個連續 token 的片段（n-gram），問：這些片段有多少比例也出現在訓練資料中？比例是 1，題目整段都在訓練資料裡；是 0，一個片段都沒出現過。

\\( n \\) 的大小是權衡。太小，「in the park」這種常見片語到處都有，什麼題目都會重疊；太大，題目只要被改掉幾個字就抓不到。GPT-3 的論文（Brown 等人，2020）檢查訓練資料與 benchmark 的重疊時用的是 13-gram，本書沿用 13 當預設值。

實作上有個問題：標準語料有 4,072 萬個 token，就有約 4,072 萬個 13-gram。把它們一個一個存進 Python 的 `set`（每個 n-gram 一個 tuple）又慢又占記憶體。我們改存雜湊值：每個 n-gram 算成一個 `uint64`，4,000 多萬個值只要幾百 MB。

```python,ignore
{{#include ../../../src/llmz/evaluation.py:contamination}}
```

- `ngram_hashes` 對每個位置的連續 `n` 個 token 算一個雜湊：`h = h * _MULT + t[i : i + len(h)]` 每做一次，所有位置同時多吃進一個 token；做 `n` 次，第 `j` 個位置就吃完了 `tokens[j : j + n]`。`uint64` 的乘法溢位時直接捨去高位，等於對 \\( 2^{64} \\) 取餘數。順序不同的 n-gram 會得到不同的值。
- `ngram_index` 把語料所有 n-gram 的雜湊排序、去重，只要建一次。
- `contamination_rate` 用 `np.searchsorted` 在排好序的 index 裡查每個雜湊在不在，回傳「在的比例」。
- 題目不到 `n` 個 token 就一個 n-gram 都沒有，沒辦法說它乾淨，所以回傳 `nan`，不是 0。

先用隨機的 token 看它怎麼運作：

```python
import numpy as np

from llmz.evaluation import contamination_rate, ngram_hashes, ngram_index

rng = np.random.default_rng(0)
corpus = rng.integers(0, 10000, 5000)  # 假裝這是訓練集：5000 個 token
index = ngram_index(corpus)  # n 預設 13
print(f"{len(corpus)} 個 token 有 {len(ngram_hashes(corpus))} 個 13-gram，index 裡有 {len(index)} 個不同的")

copied = corpus[1234:1274]  # 原封不動抄來的 40 個 token
unrelated = rng.integers(0, 10000, 40)  # 40 個新的 token
mixed = np.concatenate([corpus[100:120], unrelated[:20]])  # 前 20 個抄來的，後 20 個是新的
short = corpus[1234:1246]  # 只有 12 個 token
for name, tokens in [("抄來的", copied), ("不相干的", unrelated), ("一半抄來", mixed), ("太短的", short)]:
    print(f"{name}：{contamination_rate(tokens, index):.4f}")
```

執行結果：

```text
5000 個 token 有 4988 個 13-gram，index 裡有 4988 個不同的
抄來的：1.0000
不相干的：0.0000
一半抄來：0.2857
太短的：nan
```

`mixed` 有 40 個 token、28 個 13-gram，其中完全落在抄來那 20 個 token 裡的有 \\( 20 - 12 = 8 \\) 個，所以是 \\( 8/28 = 0.2857 \\)。

### 在標準語料上實跑

先檢查本書的題庫（`test` 與 `dev`，共 105 題）有沒有出現在標準語料的訓練集。每題把題目與正確選項接成一段文字，用 `n = 13`、8、5 各查一次。再用同一個 index，看驗證集與訓練集的重疊：第 5 章用驗證集 loss 判斷模型好壞，它和訓練集是不是真的互相獨立？

```python,norun
import numpy as np

from llmz.data import load_tokens
from llmz.evaluation import contamination_rate, load_mcq, ngram_index
from llmz.tokenizer import Tokenizer

tokenizer = Tokenizer.load("data/tokenizer.json")
train, valid = load_tokens("data/train.bin"), load_tokens("data/valid.bin")
# 題庫每一題當成一段文字：題目接上正確的選項
items = [tokenizer.encode(it["context"] + it["choices"][it["answer"]]) for it in load_mcq()]

print(f"{len(items)} 題，token 數 {min(map(len, items))}～{max(map(len, items))}")
print("n：太短沒有 n-gram 的題數／有重疊的題數／平均重疊比例")
for n in [13, 8, 5]:
    index = ngram_index(train, n)
    rates = np.array([contamination_rate(item, index, n) for item in items])
    checked = rates[~np.isnan(rates)]
    print(f"{n:>2}：{np.isnan(rates).sum()}／{(checked > 0).sum()}／{checked.mean():.3f}"
          f"（訓練集有 {len(index):,} 種 {n}-gram）")
    if n == 13:
        index13 = index

# 偵測器會不會響：從訓練集裡原封不動切一段下來，應該全部重疊
print("從訓練集切 40 個 token 當題目：", contamination_rate(train[1_000_000:1_000_040], index13))

# 驗證集對訓練集：整份驗證集有多少 13-gram 在訓練集裡出現過
print(f"驗證集的 13-gram 出現在訓練集的比例：{contamination_rate(valid, index13):.3f}")
eot = tokenizer.encode("<|endoftext|>")[0]  # 故事與故事之間的分隔 token
valid = np.asarray(valid)
stories = np.split(valid, np.flatnonzero(valid == eot) + 1)
rates = np.array([contamination_rate(s, index13) for s in stories])
sizes = np.array([max(len(s) - 12, 0) for s in stories])  # 每篇故事自己有幾個 13-gram
print(f"只算每篇故事自己裡面的 13-gram（拿掉跨 <|endoftext|> 的）：{np.nansum(rates * sizes) / sizes.sum():.3f}")
rates = rates[~np.isnan(rates)]
print(f"{len(rates):,} 篇故事：有任何一個 13-gram 重疊的占 {(rates > 0).mean():.1%}，"
      f"超過一半的有 {(rates > 0.5).sum()} 篇，最高的一篇是 {rates.max():.1%}")
```

M4 Max 上的輸出（沒有隨機性）：

```text
105 題，token 數 9～26
n：太短沒有 n-gram 的題數／有重疊的題數／平均重疊比例
13：18／0／0.000（訓練集有 38,118,707 種 13-gram）
 8：0／39／0.083（訓練集有 31,459,565 種 8-gram）
 5：0／101／0.490（訓練集有 17,541,210 種 5-gram）
從訓練集切 40 個 token 當題目： 1.0
驗證集的 13-gram 出現在訓練集的比例：0.082
只算每篇故事自己裡面的 13-gram（拿掉跨 <|endoftext|> 的）：0.057
27,630 篇故事：有任何一個 13-gram 重疊的占 84.8%，超過一半的有 2 篇，最高的一篇是 54.1%
```

建一個 13-gram 的 index 要排序 4,000 多萬個值（約 305 MB）。實測三次是 20.5、24.3、45.4 秒；當時機器上同時有別的工作在跑，數字偏高，沒有別的工作時會快不少，但沒有精確量過。

- **題庫是乾淨的**：在查得了的 87 題裡，沒有任何一題有 13-gram 出現在訓練集，這符合預期，因為題目是為本書原創的。偵測器本身會響：從訓練集切一段下來查，重疊比例是 1.0。
- **有 18 題查不了**：這些題目（題目加正確選項）不到 13 個 token，沒有 13-gram，`contamination_rate` 回傳 `nan`。要檢查它們得用比較小的 \\( n \\)。
- **\\( n \\) 越小，重疊越多，但意義越淡**：\\( n = 8 \\) 時 105 題有 39 題有重疊，\\( n = 5 \\) 時 101 題都有，平均有 49% 的 5-gram 在訓練集裡出現過。這些不是題目被抄走了，是 TinyStories 本來就大量重複的常見片語。題庫是原創的，所以我們知道 \\( n = 13 \\) 時的 0 才是對的；對別人的題庫，\\( n \\) 要選到「隨便一個沒污染的文字也不會大量重疊」的程度。
- **驗證集有 8.2% 的 13-gram 出現在訓練集**，84.8% 的故事至少有一個 13-gram 重疊。8.2% 是把整份驗證集當成一長串直接算的，其中有些 13-gram 橫跨 `<|endoftext|>`、把上一篇的結尾和下一篇的開頭接在一起；只算每篇故事自己裡面的，是 5.7%。我們沒有逐篇看原因（不放 TinyStories 的摘錄），只能推測是故事的固定開場與句型。但沒有一篇大半重疊：超過一半的只有 2 篇，最高 54.1%。所以第 5 章的驗證集 loss 不是對「完全沒見過的文字」量的，不過沒有整篇故事重複。「有任何一個 n-gram 重疊就算污染」在這種高度重複的語料上會判得太多，要設一個比例的門檻。第 13 章會正式介紹怎麼找出重複的文件。

### 污染讓分數虛高多少

偵測看到的是「重疊」，要知道它讓分數多出幾分，就得實際製造一份被污染的模型。做法：把 `test` 的前 50 題當成「洩漏」的題目，拿「題目接正確選項」微調標準模型 10 步（每步一批 50 條，AdamW，學習率 \\( 10^{-4} \\)），損失函數用第 3 集的 `sequence_logprob`（每個 token 的平均 \\( -\log p \\)）；然後分別量洩漏的 50 題與沒洩漏的後 50 題的準確率。

先在縮小版上看機制：隨機 token 的小模型，8 條序列洩漏進訓練，另外 8 條沒有。這是 CPU 上幾秒的 smoke 設定，也是 CI 跑的版本：

```python
import torch

from llmz.evaluation import sequence_logprob
from llmz.model import ModelConfig, TransformerLM
from llmz.optim import AdamW

torch.manual_seed(0)
cfg = ModelConfig(vocab_size=64, context_length=32, d_model=32, n_layers=1, n_heads=2, d_ff=64)
model = TransformerLM(cfg)
leaked = torch.randint(0, 64, (8, 20))  # 8 條隨機序列：洩漏進訓練資料的「題目」
clean = torch.randint(0, 64, (8, 20))  # 另外 8 條：沒洩漏
mask = torch.ones_like(leaked, dtype=torch.bool)  # 每個 token 都算
optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)


def per_token_loss(ids):
    return -sequence_logprob(model, ids, mask).sum().item() / (ids.shape[0] * (ids.shape[1] - 1))


print(f"訓練前：洩漏的 {per_token_loss(leaked):.1f}，沒洩漏的 {per_token_loss(clean):.1f}")
for step in range(1, 51):
    loss = -sequence_logprob(model, leaked, mask).sum() / mask[:, 1:].sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if step in (20, 50):
        print(f"訓練 {step} 步後：洩漏的 {per_token_loss(leaked):.1f}，沒洩漏的 {per_token_loss(clean):.1f}")
```

執行結果：

```text
訓練前：洩漏的 4.5，沒洩漏的 4.4
訓練 20 步後：洩漏的 2.0，沒洩漏的 4.6
訓練 50 步後：洩漏的 0.3，沒洩漏的 5.4
```

只在洩漏的那 8 條上訓練，它們的 loss 一路掉到接近 0（背下來了），沒洩漏的反而更高。換成標準模型與 `test` 題庫，機制是一樣的。M4 Max 上整個腳本單次實測 29 秒（含載入模型；當時機器上同時有別的工作在跑）：

```python,norun
import sys

import numpy as np
import torch

from llmz.device import get_device
from llmz.evaluation import contamination_rate, load_mcq, mcq_accuracy, ngram_index, sequence_logprob
from llmz.model import ModelConfig, TransformerLM
from llmz.optim import AdamW
from llmz.tokenizer import Tokenizer

device = get_device()
tokenizer = Tokenizer.load("data/tokenizer.json")
test = load_mcq("test")
seen, unseen = (test[50:], test[:50]) if "swap" in sys.argv else (test[:50], test[50:])  # 其中一半被「洩漏」進訓練資料


def to_batch(items):
    """每題「題目＋正確選項」變成一條序列，短的往右補 0，mask 標出真正的 token。"""
    seqs = [tokenizer.encode(it["context"] + it["choices"][it["answer"]]) for it in items]
    ids = torch.zeros(len(seqs), max(map(len, seqs)), dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, seq in enumerate(seqs):
        ids[i, : len(seq)], mask[i, : len(seq)] = torch.tensor(seq), True
    return ids.to(device), mask.to(device)


model = TransformerLM(ModelConfig()).to(device)
model.load_state_dict(torch.load("runs/base/model.pt", map_location=device))
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
ids, mask = to_batch(seen)

print("訓練步數：見過的 50 題（總和／每個 token 平均）｜沒見過的 50 題（總和／每個 token 平均）")
for step in range(11):
    if step > 0:  # 第 0 步是還沒訓練的基準
        model.train()
        loss = -sequence_logprob(model, ids, mask).sum() / mask[:, 1:].sum()  # 每個 token 的平均 -log 機率
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if step in (0, 1, 2, 5, 10):
        model.eval()
        a = [mcq_accuracy(model, tokenizer, seen, normalize=n) for n in ("none", "token")]
        b = [mcq_accuracy(model, tokenizer, unseen, normalize=n) for n in ("none", "token")]
        print(f"{step:>2}：{a[0]:.0%}／{a[1]:.0%}｜{b[0]:.0%}／{b[1]:.0%}")

# 偵測器看得到嗎：把洩漏的 50 題當成「語料」，再查兩組題目
def encode_item(it):
    return tokenizer.encode(it["context"] + it["choices"][it["answer"]])


index = ngram_index(np.concatenate([encode_item(it) for it in seen]))
for name, items in [("見過的", seen), ("沒見過的", unseen)]:
    rates = np.array([contamination_rate(encode_item(it), index) for it in items])
    print(f"{name} 50 題：重疊比例 1.0 的 {(rates == 1).sum()} 題，沒有 13-gram 無從判斷的 {np.isnan(rates).sum()} 題")
```

直接執行是前 50 題被洩漏，執行時加上 `swap`（`uv run python <檔名>.py swap`）就換成後 50 題。這個實驗不存模型，也不改動 `runs/base`。

M4 Max 上的輸出，兩次（兩次分別是前一半、後一半被洩漏；沒有取樣，CPU 上第一次的準確率也一樣）：

```text
# 前 50 題洩漏
訓練步數：見過的 50 題（總和／每個 token 平均）｜沒見過的 50 題（總和／每個 token 平均）
 0：84%／92%｜78%／84%
 1：88%／96%｜78%／84%
 2：90%／96%｜78%／84%
 5：100%／98%｜78%／84%
10：100%／100%｜78%／84%
見過的 50 題：重疊比例 1.0 的 40 題，沒有 13-gram 無從判斷的 10 題
沒見過的 50 題：重疊比例 1.0 的 0 題，沒有 13-gram 無從判斷的 8 題

# 後 50 題洩漏（swap）
訓練步數：見過的 50 題（總和／每個 token 平均）｜沒見過的 50 題（總和／每個 token 平均）
 0：78%／84%｜84%／92%
 1：86%／90%｜86%／92%
 2：88%／90%｜86%／92%
 5：94%／98%｜86%／92%
10：98%／100%｜84%／92%
見過的 50 題：重疊比例 1.0 的 42 題，沒有 13-gram 無從判斷的 8 題
沒見過的 50 題：重疊比例 1.0 的 0 題，沒有 13-gram 無從判斷的 10 題
```

- **洩漏的那半，分數從 78%～84%（總和）升到 98%～100%**，每個 token 平均的算法也從 84%～92% 升到 100%。**只花了 10 步、每步 50 條序列**。洩漏的題目只被看了這幾次，模型沒有變聰明，只是記住了這 50 題。
- **沒洩漏的那半幾乎不動**：第一次完全不變（78%／84%），第二次總和在 84%～86% 之間晃動（差一題）、每個 token 平均不變（92%）。50 題的雜訊約 \\( \sqrt{0.8 \times 0.2 / 50} \approx 5.7 \\) 個百分點，一題就是 2 個百分點。
- **偵測器也看得到洩漏**：被洩漏的 50 題裡，40 題（第二次是 42 題）的重疊比例是 1.0，沒洩漏的那半是 0。
- **但有 10 題（第二次是 8 題）是洩漏了卻查不到**：它們太短，湊不出一個 13-gram。洩漏的題目裡有 16%～20% 逃過了 \\( n = 13 \\) 的檢查。

這個實驗是極端版：拿題目本身去訓練。真實的污染常常只是題目混在龐大的訓練資料裡，被看過一兩次，效果沒有這麼大；但方向一樣，而且污染的是你不知道的部分。

### 偵測的侷限

n-gram 重疊只抓得到「原文照搬」：

- **太短的題目沒有 n-gram**（上面的 18 題與 10 題）。
- **改寫、翻譯、換掉幾個字就抓不到**。改掉一個 token，包含它的至多 \\( n \\) 個 n-gram 就都不重疊了：題目很長時，剩下的還抓得到大半，題目短時就剩沒幾個（練習會算）。
- **沒有訓練資料就沒法查**：這個偵測需要拿到訓練資料。模型只公開權重或只提供 API 時，只能用間接的方法，例如比較模型在原題與改寫題上的分數差。
- **沒查到不是沒污染**：它只回答「這段文字有沒有原樣出現過」。

### 為什麼新的 benchmark 會一直出現

題庫一公開，它的文字就會散布到網頁上，進入之後爬取的訓練資料；模型的分數也會逐漸飽和，接近滿分之後分不出模型的差別。所以評測的人會不斷出新的題庫、保留不公開的測試集、定期換題，並且像 GPT-3 論文那樣，在報告分數時附上污染分析。讀別人的分數時，值得問三件事：題庫公開多久了？有沒有污染檢查？檢查用的是什麼 \\( n \\)？

## 重點整理

- 資料污染是 benchmark 的題目已經在訓練資料裡，模型靠背下來拿高分，分數虛高。
- n-gram 重疊偵測：把題目切成每 \\( n \\) 個 token 的片段，算有多少比例出現在訓練資料中；\\( n \\) 預設 13（與 GPT-3 論文一致）。`ngram_hashes` 把每個 n-gram 算成 `uint64`，`ngram_index` 排序去重，`contamination_rate` 用 `searchsorted` 查，不到 \\( n \\) 個 token 回傳 `nan`。
- 在標準語料上實跑：105 題的題庫在 13-gram 上沒有任何重疊（18 題太短查不了）；\\( n \\) 變成 8 有 39 題重疊、變成 5 有 101 題，這是常見片語，不是被抄。驗證集有 8.2% 的 13-gram 在訓練集出現（拿掉橫跨故事邊界的是 5.7%），但沒有故事大半重疊（最高 54.1%）。
- 把 50 題洩漏進訓練 10 步，洩漏那半的準確率升到 98%～100%（原本 78%～84%），沒洩漏那半只在雜訊內變動；偵測器查得到其中的 40 或 42 題，查不到的是太短的 10 或 8 題。
- 偵測只抓得到原文照搬；改寫、翻譯、太短的題目、拿不到訓練資料時都抓不到。

## 自我挑戰

這一章放進 `src/llmz/` 的是 `src/llmz/evaluation.py` 的 `load_mcq`、`sequence_logprob`、`choice_logprobs`、`pick_choice`、`build_prompt`、`predict_mcq`、`mcq_accuracy`、`normalize_answer`、`exact_match`、`judge_prompt`、`judge_once`、`judge_pair`、`ngram_hashes`、`ngram_index`、`contamination_rate`。把它們的本體換成 `raise NotImplementedError`，執行：

```bash
uv run pytest tests/test_evaluation.py
```

自己寫回來讓測試通過。

再試一題：替 `exact_match` 加上多個標準答案的版本 `exact_match_any(prediction, references)`，prediction 對任何一個標準答案算對就算對。再試一題：改寫 `contamination_rate`，讓它額外回傳第一個重疊的 n-gram 在 `tokens` 裡的位置（沒有重疊就回傳 `None`），並說明為什麼這個位置在檢查「題目哪裡被抄走了」時有用。
