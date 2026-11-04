# 語言辨識

## 本集目標

自己寫一個語言辨識器：把文字切成字元片段，數每個語言各看過哪些片段，再用 naive Bayes 把 log 機率加起來挑出最像的語言。訓練、存檔、讀檔，然後在上一集的 817 頁網頁上量各語言的比例，看它在哪裡出錯。

## 正文

### 為什麼要先認語言

網頁什麼語言都有。訓練資料通常只想留下目標語言：不同語言混在一起，模型的容量要分給它們；而且後面的品質規則（第 6 集）多半是照英文的習慣設計的。所以管線在轉完文字之後，第一道篩選是「這一頁是什麼語言」。

別人的做法：C4 用 `langdetect` 這個套件，只留下被判成英文的機率至少 0.99 的頁面（Raffel 等人 2020）；Dolma 保留英文分數至少 0.5 的文件（Soldaini 等人 2024）。本書自己寫，不到一百行。

### 切成字元片段

最直覺的做法是拿「字」當特徵：看到 `the`、`and` 就是英文。可是中文與日文不用空格分字，要先斷詞才有「字」可看；就算英文，一個字沒看過就沒有任何線索。比較省事的做法是直接看**連續的幾個字元**，叫 n-gram：`the` 裡有 `th`、`he`、`the`，`ción` 裡有 `ci`、`ió`、`ón`。每個語言常見的片段不一樣，這就夠分辨了。

```python,ignore
{{#include ../../../src/llmz/pipeline.py:char_ngrams}}
```

`casefold()` 是比 `lower()` 更徹底的轉小寫（例如德文的 `ß` 會變成 `ss`），讓大小寫不影響統計。`re.sub(r"[\W\d_]+", " ", ...)` 把標點、數字、底線、空白都換成空格，再用 `split()` 切成「段」。英文的一段是一個字，中文因為沒有空格，一段是兩個標點之間的一串字。每一段前後各補一個空格，這樣「字的開頭」與「字的結尾」也變成片段（` th` 是以 th 開頭、`he ` 是以 he 結尾）。`Counter` 是第 2 集用過的計數字典。

```python
from llmz.pipeline import char_ngrams

grams = char_ngrams("The cat", 3)
print(sorted(grams))
print(grams["the"], grams[" th"], grams["he "])
print(char_ngrams("貓咪很可愛", 2))
```

執行結果：

```text
[' c', ' ca', ' t', ' th', 'a', 'at', 'at ', 'c', 'ca', 'cat', 'e', 'e ', 'h', 'he', 'he ', 't', 't ', 'th', 'the']
1 1 1
Counter({'貓': 1, '咪': 1, '很': 1, '可': 1, '愛': 1, ' 貓': 1, '貓咪': 1, '咪很': 1, '很可': 1, '可愛': 1, '愛 ': 1})
```

### naive Bayes：把每個片段的 log 機率加起來

每個語言 \\( y \\) 有一張表，記錄訓練文字裡每個片段 \\( g \\) 出現了 \\( n_{y,g} \\) 次、全部共 \\( N_y \\) 個片段。一個片段在語言 \\( y \\) 裡的機率就是它出現的比例；一段文字有很多片段，假設它們互相獨立（這就是 naive，天真），整段文字的機率是各片段機率的連乘，取 log 之後（第 1 章：連乘會變得太小，所以改加 log）變成連加：

\\[ \log P(\text{文字} \mid y) = \sum_{g} c_g \log \frac{n_{y,g} + \alpha}{N_y + \alpha V}, \qquad \hat{y} = \arg\max_y \log P(\text{文字} \mid y) \\]

\\( c_g \\) 是片段 \\( g \\) 在這段文字裡出現幾次，\\( V \\) 是所有語言一共看過幾種片段。\\( \alpha \\) 是平滑：某個語言沒看過的片段，機率不能是 0（log 0 是負無限大，整段文字就被一個沒看過的片段判死刑），所以每個片段都先當作「看過 \\( \alpha \\) 次」。所有語言都沒看過的片段完全沒有線索，直接略過。先驗機率（每種語言一開始有多常見）我們假設都一樣，所以不用乘。

```python,ignore
{{#include ../../../src/llmz/pipeline.py:ngram_classifier}}
```

- `@classmethod` 是掛在類別上的方法：`NGramClassifier.train(...)` 不需要先有物件，直接回傳一個訓練好的物件。`load` 也是。
- `train` 只做一件事：把每個語言所有訓練文字的片段加總。沒有任何「學習」的迴圈，訓練就是數數。
- `log_likelihoods` 就是上面的公式。`predict` 挑分數最高的標籤，並把各語言的分數用 softmax（第 1 章）換成機率，這個機率叫後驗機率。`predict_lines` 等一下用。
- `save` 與 `load` 把計數表存成 JSON，所以訓練好的模型可以放進檔案，下次直接讀。

這個類別沒有一個字是語言專用的：標籤是任意的字串。第 5 集會把它拿去分類別的東西。

### 訓練資料

語言辨識需要每個語言的範例文字。我們用的是原創的短文：10 個語言（`en`、`zh`、`ja`、`ko`、`ru`、`de`、`fr`、`es`、`pt`、`it`）各 12 到 14 段生活用語與網站常見的字句（網站導覽列、隱私權聲明這類），再加每個語言 6 個沒有出現在訓練裡的句子當測試。`zh` 同時有繁體與簡體。檔案在 `src/llmz/datasets/langid_samples.jsonl`，一行一段文字。

```python,ignore
{{#include ../../../src/llmz/pipeline.py:langid}}
```

訓練、測試，再試幾個新句子：

```python
from llmz.pipeline import load_langid_samples, train_langid

model = train_langid()
test = load_langid_samples("test")
right = sum(model.predict(text)[0] == lang for lang, texts in test.items() for text in texts)
print(f"測試集 {right}/{sum(len(texts) for texts in test.values())} 句正確")

for text in ["Bonjour tout le monde", "今天的天氣真好", "Das Wetter ist heute schön", "これは日本語の文章です"]:
    label, posterior = model.predict(text)
    print(f"{text} -> {label} ({posterior:.3f})")

scores = model.log_likelihoods("Bonjour tout le monde")
print({label: round(s) for label, s in sorted(scores.items(), key=lambda kv: -kv[1])[:4]})
```

執行結果：

```text
測試集 60/60 句正確
Bonjour tout le monde -> fr (1.000)
今天的天氣真好 -> zh (1.000)
Das Wetter ist heute schön -> de (1.000)
これは日本語の文章です -> ja (1.000)
{'fr': -320, 'en': -341, 'pt': -375, 'es': -378}
```

分數是負的，越接近 0 越像。法文的句子，法文的分數最高，其次是英文、葡萄牙文與西班牙文。測試集的句子和訓練集的句子是同一種文體（短、乾淨、有標點），所以全對不代表什麼，要到真的網頁上才知道。

存檔與讀檔：

```python
import os

from llmz.pipeline import NGramClassifier, train_langid

model = train_langid()
model.save("langid.json")
print(f"{os.path.getsize('langid.json'):,} bytes")
loaded = NGramClassifier.load("langid.json")
text = "Hola, ¿cómo estás?"
print(loaded.predict(text)[0], loaded.predict(text) == model.predict(text))
```

執行結果：

```text
134,403 bytes
es True
```

### 在真的網頁上

上一集存下來的 817 頁（`data/cc/pages.jsonl`），整頁丟給 `predict`：

```python,norun
import time
from collections import Counter

from llmz.pipeline import read_pages, train_langid

pages = list(read_pages("data/cc/pages.jsonl"))
model = train_langid()

start = time.perf_counter()
preds = [model.predict(p.text) for p in pages]
seconds = time.perf_counter() - start

counts = Counter(label for label, _ in preds)
for label, n in counts.most_common():
    print(f"{label}  {n:4d} 頁  {n / len(pages):5.1%}")
print(f"{len(pages)} 頁，辨識 {seconds:.1f} 秒")

posteriors = [p for _, p in preds]
print(f"後驗機率 > 0.99 的有 {sum(p > 0.99 for p in posteriors)} 頁，> 0.9999 的有 {sum(p > 0.9999 for p in posteriors)} 頁")

weight = Counter()  # 改用每頁的字元數當權重：長的頁面占的比重大
for page, (label, _) in zip(pages, preds):
    weight[label] += len(page.text)
total = sum(weight.values())
print("按字元數計：", "  ".join(f"{label} {n / total:.1%}" for label, n in weight.most_common(4)))
```

M4 Max 上的輸出（辨識時間兩次執行是 3.5 與 3.9 秒，這裡貼第一次；其餘每次相同）：

```text
en   408 頁  49.9%
zh   100 頁  12.2%
ru    92 頁  11.3%
de    66 頁   8.1%
fr    39 頁   4.8%
ja    37 頁   4.5%
es    26 頁   3.2%
pt    22 頁   2.7%
ko    16 頁   2.0%
it    11 頁   1.3%
817 頁，辨識 3.5 秒
後驗機率 > 0.99 的有 812 頁，> 0.9999 的有 806 頁
按字元數計： en 51.3%  fr 14.5%  de 11.6%  ru 8.5%
```

英文大約占一半。按頁數與按字元數算，順序不一樣：少數幾個超長的頁面就能改變字元數的排行，所以管線裡「留下多少」要同時看頁數與字元數。

先看最後一行的前一行：**812 頁的後驗機率超過 0.99**，806 頁超過 0.9999。naive Bayes 假設每個片段互相獨立，一段文字有幾千個片段，每個片段都投一票，差一點點的證據會被乘幾千次，機率就被推到 1。所以 `predict` 回傳的機率不能當成「有多確定」；想要設門檻（像 Dolma 的 0.5），要換一個沒有被放大的分數，例如「每個片段平均的分數差」（第 5 集會用這個）。

三種錯法，下面都用實測的數字說話。

**錯法一：混了別種語言的頁面**。很多頁面的導覽列、按鈕、程式碼殘渣是英文，內文才是中文、日文、韓文。這時 `predict` 是整頁所有片段一起加分，一大段不認得的內文，輸給一小段熟悉的英文。為了知道真正的語言，我們用書寫系統當線索：一頁裡一半以上的字母是韓文、漢字或西里爾字母，答案幾乎就確定了（有假名算日文，沒有算中文，西里爾字母算 `ru`；烏克蘭文與塞爾維亞文也是西里爾字母，我們沒有那些標籤，所以放寬）。同時量一個改良版：`predict_lines`，一行一行分類，每一行的票數是它的字母數，得票最多的語言贏：

```python,norun
from collections import Counter

from llmz.pipeline import read_pages, train_langid

SCRIPTS = {"han": (0x4E00, 0x9FFF), "kana": (0x3040, 0x30FF), "hangul": (0xAC00, 0xD7AF), "cyrillic": (0x0400, 0x04FF)}


def script_shares(text):
    """每種文字（書寫系統）占所有字母的比例；不在上面四種裡的字母（拉丁字母等）不列。"""
    letters = [ord(ch) for ch in text if ch.isalpha()]
    return {name: sum(lo <= c <= hi for c in letters) / max(len(letters), 1) for name, (lo, hi) in SCRIPTS.items()}


def expected_language(text):
    """從書寫系統猜正解：只有一半以上的字母是韓文、漢字或西里爾字母的頁面才有答案，其他回傳 None。"""
    s = script_shares(text)
    if s["hangul"] >= 0.5:
        return "ko"
    if s["han"] + s["kana"] >= 0.5:
        return "ja" if s["kana"] >= 0.05 else "zh"
    if s["cyrillic"] >= 0.5:
        return "ru"  # 烏克蘭文、塞爾維亞文也是西里爾字母；我們沒有那些標籤，算 ru 對


pages = list(read_pages("data/cc/pages.jsonl"))
model = train_langid()
rows = [(p.text, expected_language(p.text)) for p in pages]
rows = [(text, want) for text, want in rows if want]
print(f"{len(rows)} 頁有一半以上是韓文、漢字或西里爾字母")
whole = Counter((want, model.predict(text)[0]) for text, want in rows)
lines = Counter((want, model.predict_lines(text)) for text, want in rows)
for name, c in [("整頁", whole), ("逐行投票", lines)]:
    right = sum(n for (want, got), n in c.items() if want == got)
    to_en = sum(n for (want, got), n in c.items() if got == "en")
    print(f"{name}：{right}/{len(rows)} 頁正確，{to_en} 頁被判成 en")
confusion = Counter((want, got) for (want, got), n in whole.items() for _ in range(n) if want != got)
print("整頁判錯的 (正解, 判成)：", dict(confusion.most_common(6)))

han = [ch for text, want in rows if want in ("zh", "ja") for ch in text if "\u4e00" <= ch <= "\u9fff"]
print(f"中文與日文的頁面共有 {len(han):,} 個漢字，只有 {sum(ch in model.vocab for ch in han) / len(han):.0%} 出現在訓練資料裡")
```

M4 Max 上的輸出（每次相同）：

```text
318 頁有一半以上是韓文、漢字或西里爾字母
整頁：245/318 頁正確，67 頁被判成 en
逐行投票：309/318 頁正確，5 頁被判成 en
整頁判錯的 (正解, 判成)： {('zh', 'en'): 54, ('ja', 'en'): 11, ('zh', 'pt'): 2, ('zh', 'it'): 2, ('zh', 'de'): 2, ('ko', 'en'): 2}
中文與日文的頁面共有 222,587 個漢字，只有 40% 出現在訓練資料裡
```

這 318 頁的整頁準確率只有 77%，錯的幾乎都是被判成 `en`，其中中文與日文最多。原因有兩個。一是訓練資料太小：網頁裡的漢字只有 40% 出現在我們的訓練資料裡，沒看過的字被略過、完全沒有投票；英文的片段幾乎每個都看過很多次，每個都投出有分量的一票。二是整頁的片段全部加在一起，導覽列、按鈕上那幾個英文字，就能把票數拉過去。逐行投票讓每一行各自決定：導覽列的行字母數少，內文的行字母數多，所以正確率提高到 97%（318 頁裡錯了 9 頁）。這就是 `predict_lines` 存在的原因。

**錯法二：沒有標籤的語言**。我們只認得 10 個語言，其他語言的頁面也會被硬塞進其中一個，因為分類器沒有「都不是」這個選項。用只有越南文與波蘭文才常用的字母當線索（占全頁字母 3% 以上就當作是那個語言）：

```python,norun
from collections import Counter

from llmz.pipeline import read_pages, train_langid

MARKERS = {
    "越南文": "ăơưđạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ",
    "波蘭文": "ąęłńśźż",
}
pages = list(read_pages("data/cc/pages.jsonl"))
model = train_langid()
for name, marker in MARKERS.items():
    rows = []
    for page in pages:
        letters = [ch for ch in page.text.lower() if ch.isalpha()]
        if sum(ch in marker for ch in letters) >= 0.03 * max(len(letters), 1):
            rows.append(page.text)
    whole = Counter(model.predict(t)[0] for t in rows)
    lines = Counter(model.predict_lines(t) for t in rows)
    print(f"{name}：{len(rows)} 頁；整頁 {dict(whole.most_common(4))}；逐行投票 {dict(lines.most_common(4))}")
```

M4 Max 上的輸出（每次相同）：

```text
越南文：11 頁；整頁 {'en': 11}；逐行投票 {'en': 10, 'de': 1}
波蘭文：18 頁；整頁 {'en': 16, 'it': 1, 'es': 1}；逐行投票 {'en': 16, 'de': 1, 'es': 1}
```

越南文 11 頁全部被判成英文，波蘭文 18 頁有 16 頁。這兩種語言都用拉丁字母，帶重音的字母我們沒看過（略過），剩下的 ASCII 片段看起來比較像英文。逐行投票幫不上忙，因為整頁的每一行都是同一個語言。如果管線的規則是「留下被判成英文的頁面」，這 27 頁就會混進英文資料。

**錯法三：太短的文字**。一個只有幾個字的導覽列，片段太少，分數差很小。把 60 個測試句子都砍成只看前面 `k` 個字元，看正確率掉多少：

```python
from llmz.pipeline import load_langid_samples, train_langid

model = train_langid()
test = load_langid_samples("test")
for k in [3, 5, 8, 12, 20, 100]:
    right = sum(model.predict(text[:k])[0] == lang for lang, texts in test.items() for text in texts)
    print(f"只看前 {k:3d} 個字元：{right}/60 句正確")
```

執行結果：

```text
只看前   3 個字元：42/60 句正確
只看前   5 個字元：45/60 句正確
只看前   8 個字元：51/60 句正確
只看前  12 個字元：53/60 句正確
只看前  20 個字元：56/60 句正確
只看前 100 個字元：60/60 句正確
```

（`k = 100` 比所有句子都長，等於整句。）只看 3 個字元時三成的句子判錯，到 20 個字元才接近全對。`predict` 只有在「完全沒有看過的片段」時才回傳 `("unknown", 0.0)`，其他情況照樣挑一個，所以太短的頁面要自己設下限（例如少於幾十個字母就不判斷）。

### 這個版本做不到的事

- **沒有「都不是」**：要認更多語言，就加那個語言的範例文字；或另外量「這一頁有多少比例的片段是看過的」，太低就當作 `unknown`（練習）。
- **只靠 1 到 3 個字元的片段**：同一語系的語言（西班牙文與葡萄牙文、塞爾維亞文與克羅埃西亞文）很難分，繁體與簡體中文也沒有分開（兩個都是 `zh`）。
- **訓練資料很小**：正式的語言辨識工具用的是大量真實文字，涵蓋上百種語言，也更能處理混合內容。這一集的重點是每個決定都看得見。

## 重點整理

- 語言辨識是管線的第一道篩選：C4 只留下 `langdetect` 英文機率至少 0.99 的頁面，Dolma 留下英文分數至少 0.5 的文件。
- 用 1 到 3 個字元的片段（n-gram）當特徵，不需要斷詞；`char_ngrams` 把標點與數字當分隔、每一段前後補空格。
- naive Bayes：每個語言一張片段計數表，文字的分數是各片段 log 機率的總和（加 \\( \alpha \\) 平滑，所有語言都沒看過的片段略過），挑最高的；訓練就是數數，`save` 與 `load` 存成 JSON。
- naive Bayes 的後驗機率太極端：817 頁裡 812 頁超過 0.99，不能當信心。
- 在 817 頁上（M4 Max）：英文占 49.9%；整頁分類在以非拉丁字母為主的 318 頁上只對 77%（多半被判成 `en`），逐行投票提高到 97%；沒有標籤的語言（越南文 11 頁全部、波蘭文 18 頁中的 16 頁）會被判成英文。
