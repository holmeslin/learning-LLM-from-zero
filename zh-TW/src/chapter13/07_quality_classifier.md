# 品質分類器

## 本集目標

不手寫規則，改成給電腦兩堆例子：一堆「像標準語料」的文字、一堆「一般網頁」的文字，訓練一個雜湊 n-gram 加 logistic regression 的分類器，用它的分數替每份文件打「像不像標準語料」，並量出它在那 817 頁網頁上打的分數長什麼樣子。

## 正文

### 用例子定義「好」

上一集的規則只看形狀。還有另一條路：不去想「好文件長什麼樣子」，而是直接拿一批大家公認好的文字當正例、一批沒篩過的網頁當負例，訓練一個分類器去分，讓模型自己找出差別。

GPT-3 的論文（Brown 等人，2020，arXiv 2005.14165）附錄 A 就是這樣篩 Common Crawl 的：用 WebText、Wikipedia 與網路書籍語料當正例，沒篩過的 Common Crawl 當負例，訓練一個 logistic regression 分類器，特徵來自 Spark 的 tokenizer 與 HashingTF（把詞雜湊到固定數量的格子裡）。打完分數以後也不是設門檻，而是照 `np.random.pareto(α) > 1 − document_score` 這個條件決定留不留，α 取 9：分數越高，留下來的機率越大，分數低的也還有一點機會。論文說這樣選 α，是要「以高分的文件為主，但還是收一些不在正例分布裡的文件」。

我們做同樣的事，規模縮到一台筆電：

- 正例：標準語料（TinyStories）的故事。
- 負例：本章第 1、2 集抽出來的網頁。
- 特徵：每份文件前 256 個詞的每個詞與每對相鄰的詞，雜湊到固定數量的 bucket。
- 模型：logistic regression，用本書自己的 `AdamW` 訓練。

要先說清楚一件事：這樣訓練出來的分數，意思是「這份文件像不像**我們挑的正例**」，不是「這份文件好不好」。正例選什麼，「好」就是什麼。這一點到本集最後會看到它的後果。

### 雜湊 n-gram

特徵要把一份文件變成一串數字。最直接的做法：把所有出現過的詞與詞對列成一本字典，每個編一個號碼。但我們的訓練資料裡就有 43 萬種不同的 n-gram（下面會量），新資料裡還會出現沒見過的。**雜湊技巧**（hashing trick）換個想法：不建字典，直接算一個函式 \\( h \\)，把任何字串變成 0 到 \\( B - 1 \\) 之間的號碼，\\( B \\) 就是 bucket 的數量，不管看過多少種 n-gram，特徵的空間永遠是 \\( B \\) 格。

```python,ignore
{{#include ../../../src/llmz/pipeline.py:hash_ngrams}}
```

- 詞用 `\w+` 抓出來、轉小寫，所以標點與大小寫不影響。`words + [...]` 先放所有單詞（unigram），再放每一對相鄰的詞（bigram）。
- 函式 \\( h \\) 是 `zlib.crc32(...) % n_buckets`：CRC32 是標準函式庫裡一個固定的校驗碼。不用 Python 內建的 `hash()`，因為字串的 `hash()` 每次啟動程式都不一樣，存下來的權重換一次執行就對不上了。
- 兩個不同的 n-gram 可能落在同一個 bucket，這叫**碰撞**（collision）。它們會共用一個權重，下面會量它多常發生、有沒有影響。

```python
from llmz.pipeline import hash_ngrams

ids = hash_ngrams("The cat sat on the mat.", 1000)
print(len(ids), ids)  # 6 個詞 + 5 對相鄰的詞
print(ids == hash_ngrams("the CAT sat on the mat", 1000))  # 大小寫與標點不算
print(len(set(ids)), len(set(hash_ngrams("The cat sat on the mat.", 8))))  # bucket 越少，不同的 n-gram 越容易撞在一起
```

執行結果：

```text
11 [78, 832, 960, 192, 78, 202, 832, 18, 53, 770, 906]
True
9 4
```

11 個號碼只有 9 種。兩個 `the` 是同一個詞，號碼當然一樣（78）；另一個重複是真的碰撞：單詞 `cat` 和相鄰的一對 `the cat` 是不同的 n-gram，在 1000 個 bucket 裡剛好撞到同一個號碼 832。bucket 只有 8 個的時候，11 個 n-gram 只擠得進 4 個格子。

### Logistic regression

模型的結構是這樣：每個 bucket 一個權重 \\( w_b \\)，一份文件的分數是它所有 n-gram 的權重平均，再加一個 bias \\( c \\)。這個分數是 logit，過 sigmoid 就是「像標準語料」的機率：

\\[
z = \frac{1}{|d|} \sum_{g \in d} w_{h(g)} + c, \qquad p = \sigma(z) = \frac{1}{1 + e^{-z}}
\\]

\\( |d| \\) 是文件的 n-gram 數，取平均讓長文件與短文件的分數在同一個尺度上。這就是一維的 `Embedding`（第 4 章）再取平均：`self.weight[ids]` 查出每個 n-gram 的權重。

```python,ignore
{{#include ../../../src/llmz/pipeline.py:quality_classifier}}
```

- `forward(ids, mask)` 吃 `(B, L)` 的 bucket 號碼與 mask。文件長短不一，`pad_docs` 把它們補到同樣長，mask 是 0 的位置是補出來的，不算進平均。`clamp(min=1)` 讓空文件不會除以 0。
- `binary_cross_entropy` 是 logistic regression 的 loss。把 logit \\( z \\) 看成兩個類別的 logits \\( [0, z] \\)，softmax 之後第 1 類的機率正是 \\( \sigma(z) \\)，所以直接用第 1 章的 `cross_entropy` 就好。`tests/test_pipeline.py` 用 `F.binary_cross_entropy_with_logits` 驗證兩者一樣（只用在驗證）。
- `train_quality` 每一步抽一個 batch，**一半正例、一半負例**：我們有 2 萬篇故事，卻只有幾百頁網頁，不這樣抽的話模型幾乎只看得到故事。優化器是第 5 章的 `AdamW`。
- `score(texts)` 回傳每份文件的 logit，大於 0 就是機率過半。

用手算一次，確認「平均」與「padding 不算」：

```python
import torch

from llmz.pipeline import QualityClassifier

model = QualityClassifier(8)
with torch.no_grad():
    model.weight[3], model.weight[5], model.bias[0] = 2.0, -1.0, 0.5
ids = torch.tensor([[3, 5, 0], [5, 5, 5]])
mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])  # 第 1 篇只有 2 個 n-gram，最後一格是 padding
z = model(ids, mask)
print(z.tolist())  # 第 1 篇：(2 + -1) / 2 + 0.5；第 2 篇：(-1 -1 -1) / 3 + 0.5
print([round(p, 2) for p in torch.sigmoid(z).tolist()])
```

執行結果：

```text
[1.0, -0.5]
[0.73, 0.38]
```

### 玩具資料上的 smoke 訓練

先用一個原創的玩具資料確認它會學。正例是幾句溫和的故事句子，負例是幾句網頁的制式文字，每份文件隨機拼三句。訓練與驗證用不同的亂數種子拼出來，所以驗證的文件是沒看過的組合：

```python
import random

import torch

from llmz.pipeline import train_quality

GOOD = [
    "the little cat and the dog played in the sun",
    "a happy girl and her mom baked a warm cake",
    "the boy had a red ball and a kind friend",
]
BAD = [
    "click here to subscribe to our newsletter now",
    "terms of use privacy policy all rights reserved",
    "buy cheap pills online free shipping offer",
]


def make_docs(seed, n):
    rng = random.Random(seed)
    labels = [rng.randint(0, 1) for _ in range(n)]
    return [" ".join(rng.choice([BAD, GOOD][y]) for _ in range(3)) for y in labels], labels


texts, labels = make_docs(0, 200)
model, losses = train_quality(texts, labels, n_buckets=1 << 12, steps=60, batch_size=16)
print(f"loss：第 1 步 {losses[0]:.2f}，第 60 步 {losses[-1]:.2f}")

valid_texts, valid_labels = make_docs(1, 100)
correct = ((model.score(valid_texts) > 0).long() == torch.tensor(valid_labels)).float().mean()
print(f"沒看過的 100 份文件，答對 {correct:.0%}")

unseen = ["the girl and the dog had a warm cake", "subscribe now for free shipping and cheap pills"]
for text, p in zip(unseen, torch.sigmoid(model.score(unseen)).tolist()):
    print(f"{p:.2f}  {text}")
```

執行結果：

```text
loss：第 1 步 0.69，第 60 步 0.02
沒看過的 100 份文件，答對 100%
0.97  the girl and the dog had a warm cake
0.15  subscribe now for free shipping and cheap pills
```

loss 從 \\( \ln 2 \approx 0.69 \\)（還沒學，兩邊各一半）降到接近 0。這個玩具資料太好分，重點只是看流程跑得通；真的資料會難一點。

### 在真的資料上

設定：正例是 `data/train.txt` 隨機抽的 2 萬篇故事，負例是本章第 1、2 集抽出來的網頁。正例取自訓練集而不是 `data/valid.txt`：`valid.txt` 是第 5 章以來用來量模型、沒拿去訓練過任何東西的那一份，這一集要拿它當「沒看過的故事」來量分類器，就不能先拿它訓練分類器。網頁只有 817 頁，要小心兩件事。第一，同一個網站的頁面長得很像（一樣的選單、一樣的頁尾），如果同一個網站的頁面同時出現在訓練與驗證，分數會高估。所以我們用網址的主機名稱決定一頁屬於哪一份，同一個網站一定在同一邊。第二，驗證要用沒看過的資料：驗證故事來自 `data/valid.txt`，驗證網頁是留下來的 5 份之一。

先看 bucket 的數量有沒有影響。每個設定訓練 300 步（每步 64 篇，一半一半），種子 0 與 1 各一次：

```python,norun
import random
import time
import zlib
from urllib.parse import urlsplit

from llmz.pipeline import hash_ngrams, read_pages, train_quality


def read_stories(path, n, seed):
    docs = [d.strip() for d in open(path, encoding="utf-8").read().split("<|endoftext|>") if d.strip()]
    return random.Random(seed).sample(docs, n)


pages = list(read_pages("data/cc/pages.jsonl"))
hosts = [urlsplit(p.url).hostname for p in pages]
held = [zlib.crc32(h.encode()) % 5 == 0 for h in hosts]  # 同一個網站的頁面一定在同一邊
train_web = [p.text for p, h in zip(pages, held) if not h]
valid_web = [p.text for p, h in zip(pages, held) if h]
train_stories = read_stories("data/train.txt", 20000, 0)
valid_stories = read_stories("data/valid.txt", 2000, 1)
print(f"故事：訓練 {len(train_stories):,} 篇、驗證 {len(valid_stories):,} 篇")
print(f"網頁：訓練 {len(train_web)} 頁、驗證 {len(valid_web)} 頁（{len({x for x, h in zip(hosts, held) if h})} 個網站）")

start = time.perf_counter()
for text in train_stories:
    hash_ngrams(text, 1 << 16)
print(f"只把 {len(train_stories):,} 篇故事雜湊成號碼：{time.perf_counter() - start:.1f} 秒")

texts = train_stories + train_web
labels = [1] * len(train_stories) + [0] * len(train_web)
for bits in [8, 12, 16, 20]:
    for seed in [0, 1]:
        start = time.perf_counter()
        model, losses = train_quality(texts, labels, n_buckets=1 << bits, seed=seed)
        seconds = time.perf_counter() - start
        story_acc = (model.score(valid_stories) > 0).float().mean().item()
        web_acc = (model.score(valid_web) < 0).float().mean().item()
        print(f"n_buckets=2^{bits:<2d} 種子 {seed}：{seconds:4.1f} 秒，最後 20 步的平均 loss {sum(losses[-20:]) / 20:.4f}，驗證故事答對 {story_acc:.2%}，驗證網頁答對 {web_acc:.2%}")
```

M4 Max 上的輸出（準確率與 loss 每次相同；秒數是這一次的，包含雜湊的時間，另一次執行是 4.2～9.4 秒，別的程式同時在跑時會更慢）：

```text
故事：訓練 20,000 篇、驗證 2,000 篇
網頁：訓練 625 頁、驗證 192 頁（181 個網站）
只把 20,000 篇故事雜湊成號碼：2.9 秒
n_buckets=2^8  種子 0：10.1 秒，最後 20 步的平均 loss 0.0653，驗證故事答對 99.85%，驗證網頁答對 98.44%
n_buckets=2^8  種子 1： 8.7 秒，最後 20 步的平均 loss 0.0642，驗證故事答對 99.85%，驗證網頁答對 98.96%
n_buckets=2^12 種子 0：10.7 秒，最後 20 步的平均 loss 0.0102，驗證故事答對 100.00%，驗證網頁答對 100.00%
n_buckets=2^12 種子 1： 6.9 秒，最後 20 步的平均 loss 0.0104，驗證故事答對 100.00%，驗證網頁答對 100.00%
n_buckets=2^16 種子 0： 5.6 秒，最後 20 步的平均 loss 0.0038，驗證故事答對 100.00%，驗證網頁答對 100.00%
n_buckets=2^16 種子 1： 6.7 秒，最後 20 步的平均 loss 0.0040，驗證故事答對 100.00%，驗證網頁答對 100.00%
n_buckets=2^20 種子 0： 9.9 秒，最後 20 步的平均 loss 0.0029，驗證故事答對 100.00%，驗證網頁答對 100.00%
n_buckets=2^20 種子 1： 8.4 秒，最後 20 步的平均 loss 0.0031，驗證故事答對 100.00%，驗證網頁答對 100.00%
```

- 訓練很快：每個設定 4 到 11 秒，其中把 2 萬篇故事雜湊成號碼就要 2.9 秒。只有 300 步，模型只是一個 \\( B \\) 維的向量，CPU 就夠了，這一集不需要 GPU。
- bucket 只有 \\( 2^8 = 256 \\) 個時，驗證網頁有 1.0% 到 1.6% 被判成像故事（兩個種子）。\\( 2^{12} \\) 起驗證全對，再加大也沒有更好。
- 這個結果有點太漂亮。標準語料是給小孩看的簡單短篇故事，網頁是網頁，兩者的詞彙差得太遠，分得開不是什麼本事，後面會看到模型到底在看什麼。

碰撞呢？

```python,norun
import random
import zlib
from collections import Counter

from llmz.pipeline import WORD_RE, read_pages, train_quality


def read_stories(path, n, seed):
    docs = [d.strip() for d in open(path, encoding="utf-8").read().split("<|endoftext|>") if d.strip()]
    return random.Random(seed).sample(docs, n)


def ngrams(text):  # 和 hash_ngrams 一樣的 n-gram，只是不雜湊
    words = WORD_RE.findall(text.lower())[:256]
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


texts = read_stories("data/train.txt", 20000, 0) + [p.text for p in read_pages("data/cc/pages.jsonl")]
labels = [1] * 20000 + [0] * (len(texts) - 20000)
counts = Counter(g for t in texts for g in ngrams(t))
print(f"{len(counts):,} 種不同的 n-gram")
once = sum(n == 1 for n in counts.values())
top = sum(n for _, n in counts.most_common(5000))
print(f"{once / len(counts):.0%} 的 n-gram 只出現 1 次；出現次數最多的 5,000 種，占全部出現次數的 {top / sum(counts.values()):.0%}")
for bits in [16, 20]:
    buckets = Counter(zlib.crc32(g.encode()) % (1 << bits) for g in counts)
    shared = sum(buckets[zlib.crc32(g.encode()) % (1 << bits)] > 1 for g in counts)
    print(f"2^{bits} 個 bucket：{len(buckets):,} 個有人用，{shared / len(counts):.1%} 的 n-gram 和別的 n-gram 共用 bucket")

model, _ = train_quality(texts, labels, n_buckets=1 << 20)
weight = model.weight.detach()
name = {}  # bucket 反查回去：同一個 bucket 裡出現最多次的那個 n-gram
for g, n in counts.items():
    b = zlib.crc32(g.encode()) % (1 << 20)
    if b not in name or n > counts[name[b]]:
        name[b] = g
common = sorted((weight[b].item(), g) for b, g in name.items() if counts[g] >= 50)  # 只看出現 50 次以上的
print("權重最小（網頁）：", [g for _, g in common[:10]])
print("權重最大（故事）：", [g for _, g in common[::-1][:10]])
```

M4 Max 上的輸出（每次相同）：

```text
430,508 種不同的 n-gram
63% 的 n-gram 只出現 1 次；出現次數最多的 5,000 種，占全部出現次數的 75%
2^16 個 bucket：65,443 個有人用，99.9% 的 n-gram 和別的 n-gram 共用 bucket
2^20 個 bucket：352,912 個有人用，33.8% 的 n-gram 和別的 n-gram 共用 bucket
權重最小（網頁）： ['2', '1', '5', 'contact', '4', 'or', 'content', '10', '6', 'news']
權重最大（故事）： ['said', 'one day', 'he', 'smiled', 'they', 'she', 'once', 'upon a', 'and said', 'a time']
```

43 萬種 n-gram 擠進 65,536 個 bucket，幾乎每個 n-gram 都有「鄰居」，準確率卻一樣是 100%。從上面的第二行可以看出為什麼：大部分的 n-gram 只出現一次，共用 bucket 對它們的影響很小，真正撐起分類的是幾千種常見的 n-gram。反過來，bucket 少到 256 個，連常見的詞也擠在一起，才開始出錯。

雜湊不能反推，但可以從訓練文字的 n-gram 反查：把每個 n-gram 算一次 bucket，一個 bucket 裡出現最多次的就是它的名字。權重最大的是 `said`、`one day`、`he`、`smiled`、`they`、`she`、`once`，權重最小的是 `1`、`2`、`4`、`5`、`or`、`contact`、`content`、`news`。分類器學到的，是敘事（人稱代名詞、過去式的動詞、「從前從前」）對上網頁（數字、選單與導覽的字眼）。這就是它所謂的「像標準語料」。這份名單只列出現 50 次以上的 n-gram，權重的大小也只能當粗略的參考，不是嚴格的重要性排名。

### 網頁的分數長什麼樣子

最後回答這一集一開始的問題：817 頁的分數分布。如果每一頁都拿來訓練，再拿同一頁來打分，分數沒有意義。所以改成 5 份交叉：每一份輪流當驗證，用其他 4 份訓練，替這一份打分。每一頁的分數都來自一個沒看過它、也沒看過同一個網站的模型：

```python,norun
import random
import statistics
import time
import zlib
from urllib.parse import urlsplit

from llmz.pipeline import gopher_check, read_pages, train_quality


def read_stories(path, n, seed):
    docs = [d.strip() for d in open(path, encoding="utf-8").read().split("<|endoftext|>") if d.strip()]
    return random.Random(seed).sample(docs, n)


pages = list(read_pages("data/cc/pages.jsonl"))
fold = [zlib.crc32(urlsplit(p.url).hostname.encode()) % 5 for p in pages]  # 同一個網站一定在同一份
stories = read_stories("data/train.txt", 20000, 0)

start = time.perf_counter()
scores = [0.0] * len(pages)  # 每一頁的分數，來自「沒看過這一頁、也沒看過同一個網站」的模型
for k in range(5):
    train_web = [p.text for p, f in zip(pages, fold) if f != k]
    model, _ = train_quality(stories + train_web, [1] * len(stories) + [0] * len(train_web))
    rows = [i for i, f in enumerate(fold) if f == k]
    for i, score in zip(rows, model.score([pages[i].text for i in rows]).tolist()):
        scores[i] = score
print(f"訓練 5 個模型，共 {time.perf_counter() - start:.0f} 秒")

ranked = sorted(scores)
print("分數（logit）的最小值、10%、中位數、90%、最大值：", [round(ranked[round(f * (len(ranked) - 1))], 1) for f in [0, 0.1, 0.5, 0.9, 1]])
print(f"分數大於 0（機率過半）的有 {sum(s > 0 for s in scores)} 頁")
edges = [-100, -8, -6, -4, -2, 0, 100]
for lo, hi in zip(edges, edges[1:]):
    print(f"  [{lo:>4}, {hi:>3})：{sum(lo <= s < hi for s in scores):3d} 頁")

passed = [gopher_check(p.text)[0] for p in pages]
for name, flag in [("通過", True), ("沒通過", False)]:
    mine = [s for s, ok in zip(scores, passed) if ok == flag]
    print(f"Gopher {name}的 {len(mine)} 頁：分數的中位數 {statistics.median(mine):.1f}")
order = sorted(range(len(pages)), key=lambda i: -scores[i])
for share in [0.1, 0.25, 0.5]:
    top = order[: round(len(pages) * share)]
    print(f"分數最高的 {share:.0%}（{len(top)} 頁）裡，通過 Gopher 的有 {sum(passed[i] for i in top)} 頁")
```

M4 Max 上的輸出（分數與頁數每次相同，秒數是這一次的，另外兩次是 32、52 秒）：

```text
訓練 5 個模型，共 61 秒
分數（logit）的最小值、10%、中位數、90%、最大值： [-9.9, -7.1, -5.6, -3.8, 1.3]
分數大於 0（機率過半）的有 2 頁
  [-100,  -8)： 24 頁
  [  -8,  -6)：273 頁
  [  -6,  -4)：422 頁
  [  -4,  -2)： 80 頁
  [  -2,   0)： 16 頁
  [   0, 100)：  2 頁
Gopher 通過的 233 頁：分數的中位數 -4.5
Gopher 沒通過的 584 頁：分數的中位數 -5.8
分數最高的 10%（82 頁）裡，通過 Gopher 的有 69 頁
分數最高的 25%（204 頁）裡，通過 Gopher 的有 133 頁
分數最高的 50%（408 頁）裡，通過 Gopher 的有 170 頁
```

- **幾乎所有網頁都被判成「不像」**：817 頁裡只有 2 頁分數大於 0，中位數是 -5.6（機率約 0.4%）。這不是分類器壞了，是我們的正例太窄：標準語料是兒童故事，網頁幾乎都不像。如果拿這個分類器當 0.5 的門檻去篩，817 頁只剩 2 頁。
- **所以門檻要看分數的排序，不是看 0**。GPT-3 的做法正是用分數的高低決定「留的機率」，而不是一刀切。練習題會讓你模擬那條 Pareto 的規則。
- **它和 Gopher 的規則大致同向**：通過 Gopher 的 233 頁，分數的中位數 -4.5，沒通過的 584 頁是 -5.8；分數最高的 10%（82 頁）裡有 69 頁通過 Gopher，而全部頁面通過的比例只有 28.5%。這不意外：通過 Gopher 的多半是完整的英文段落，用字比較接近故事。
- **兩者也不一樣**：分數最高的 10% 裡仍有 13 頁沒通過 Gopher，通過 Gopher 的 233 頁裡也有 63 頁落在分數的後半段（前 50% 的 408 頁裡只有 170 頁通過）。分類器看的是「用字像不像正例」，規則看的是「形狀」。

### 存起來，給後面的管線用

`tools/train_quality.py` 用全部 817 頁訓練一個分類器，存成 `runs/quality/quality.pt`（`runs/` 不進 repo）。本章第 10 集組管線時會讀它。`--smoke` 是縮小的版本（2,000 篇故事、\\( 2^{12} \\) 個 bucket、100 步），在 CPU 上幾秒就跑完：

```bash
uv run python tools/train_quality.py --smoke --out runs/quality/quality_smoke.pt
uv run python tools/train_quality.py
```

M4 Max 上的輸出（秒數是這一次的：smoke 三次是 2.1～4.1 秒，完整的四次是 7.9～9.6 秒；其餘每次相同）：

```text
2,000 篇故事 + 817 頁網頁，4096 個 bucket，100 步：2.1 秒
loss 0.693 → 0.1068（最後 20 步的平均）
訓練資料上判成像標準語料的比例：故事 100.00%，網頁 0.12%
存到 runs/quality/quality_smoke.pt
20,000 篇故事 + 817 頁網頁，65536 個 bucket，300 步：8.9 秒
loss 0.693 → 0.0045（最後 20 步的平均）
訓練資料上判成像標準語料的比例：故事 100.00%，網頁 0.00%
存到 runs/quality/quality.pt
```

「訓練資料上」的數字只是確認模型有擬合，不是它的表現，表現看上一節交叉驗證的分數。

### 它真正量的是什麼

- **分數是「像正例的程度」**。把正例換成維基百科、書、或高品質的教學文字，同一批網頁會得到完全不同的排序。「品質」是我們挑的正例定義的，沒有客觀的品質。
- **篩完的資料會往正例的樣子縮**。只留下像正例的文件，剩下的就偏向正例的文體與主題：口語、方言、小眾的主題、其他語言的文件，容易得到低分。GPT-3 用 Pareto 隨機決定留不留，還能收進一些不像正例的文件。
- **它只看詞，不看意思**。一份文法完整但內容全錯的文章，分數可以很高；一份寫得很好但用了不同詞彙的專業文件，分數可能很低。
- **還是要和規則、語言辨識、去重一起用**。後面幾集（exact dedup、MinHash）處理重複，第 10 集把所有步驟接成一條管線。

## 重點整理

- 品質分類器用例子取代規則：正例是「像標準語料」的文字，負例是一般網頁，分數是「像正例的程度」，不是客觀的品質。GPT-3 用同樣的方法篩 Common Crawl，並用 Pareto 隨機決定留不留，而不是設門檻。
- 雜湊 n-gram：每個詞與每對相鄰的詞經 `zlib.crc32` 雜湊到 \\( B \\) 個 bucket，不用字典、特徵空間固定；不能用內建的 `hash()`（每次啟動都不同）。碰撞不可避免，但在我們的資料上，\\( 2^{12} \\) 個 bucket 以上，準確率就不再變好。
- `QualityClassifier` 是一維的 `Embedding` 取平均再加 bias，logit 過 sigmoid 是機率；loss 用第 1 章的 `cross_entropy`（把 logit 看成 \\( [0, z] \\)），優化器是 `AdamW`，每個 batch 一半正例一半負例。
- 在我們的資料上，分類器分得開故事與網頁（驗證全對），但 817 頁網頁幾乎都被判成「不像」（只有 2 頁分數大於 0）；分數的排序和 Gopher 的規則大致同向。用這個分數篩資料，要看排序、而不是 0 的門檻。
