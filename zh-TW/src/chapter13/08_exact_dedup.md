# exact dedup

## 本集目標

把「出現在很多頁的行」認出來並拿掉：用 64 bit 的指紋代表一整行，掃一遍數出每一行出現在幾頁，再量這一步在真的網頁文字上拿掉多少。

## 正文

### 為什麼要去重

網路上同一段文字常常出現很多次：每一頁都有的選單與頁尾、被轉載好幾次的文章、同一個網站的鏡像。重複的內容有兩個壞處。一是浪費：同樣的文字讀好幾遍，花了計算量，卻沒有多學到什麼。二是模型容易把重複太多次的片段原封不動背下來。Lee 等人 2022 年的論文〈Deduplicating Training Data Makes Language Models Better〉就是做這件事：把訓練資料去重之後，模型逐字吐出訓練資料的頻率明顯變低。

去重可以在不同的層級做。這一集做最單純的一種：逐行的 exact dedup，兩行一字不差（exact）才算重複。第 2 集把網頁轉成「一行一個區塊」，就是為了這一步。下一集再處理「幾乎一樣、但不是一字不差」的整份文件。

### 數每一行出現在幾頁

做法是掃兩遍：第一遍數每一行出現在幾頁裡；第二遍把出現在至少 `min_pages` 頁的行拿掉，一行都不剩的頁面整頁丟掉。數的是「幾頁」，不是「幾次」，同一頁自己重複的行只算一頁：一行在同一頁重複很多次（例如清單裡每一項都有的同一個詞），不代表它是跨頁共用的框架。

```python,ignore
{{#include ../../../src/llmz/pipeline.py:dedup_lines}}
```

用四頁原創的小網頁走一遍。前三頁是同一個網站，共用選單與頁尾；第四頁是別的網站，剛好抄了第二頁的一句話：

```python
from llmz.pipeline import Page, count_lines, dedup_lines, line_hash

pages = [
    Page("http://fern.example/1", "Home | Shop | About\nHow to repot a fern\nUse a pot one size bigger.\nCopyright 2026 Fern Shop"),
    Page("http://fern.example/2", "Home | Shop | About\nWatering your fern\nWater when the top soil is dry.\nCopyright 2026 Fern Shop"),
    Page("http://fern.example/3", "Home | Shop | About\nLight for ferns\nBright, indirect light is best.\nCopyright 2026 Fern Shop"),
    Page("http://bread.example/1", "Notes on bread\nWater when the top soil is dry.\nBake at 220 degrees."),
]

counts = count_lines(pages)
for line in ["Home | Shop | About", "Water when the top soil is dry.", "Bake at 220 degrees."]:
    print(counts[line_hash(line)], "頁：", line)

for min_pages in [3, 2]:
    print(f"\nmin_pages={min_pages}")
    for page in dedup_lines(pages, min_pages):
        print(page.url, page.text.split("\n"))
```

執行結果：

```text
3 頁： Home | Shop | About
2 頁： Water when the top soil is dry.
1 頁： Bake at 220 degrees.

min_pages=3
http://fern.example/1 ['How to repot a fern', 'Use a pot one size bigger.']
http://fern.example/2 ['Watering your fern', 'Water when the top soil is dry.']
http://fern.example/3 ['Light for ferns', 'Bright, indirect light is best.']
http://bread.example/1 ['Notes on bread', 'Water when the top soil is dry.', 'Bake at 220 degrees.']

min_pages=2
http://fern.example/1 ['How to repot a fern', 'Use a pot one size bigger.']
http://fern.example/2 ['Watering your fern']
http://fern.example/3 ['Light for ferns', 'Bright, indirect light is best.']
http://bread.example/1 ['Notes on bread', 'Bake at 220 degrees.']
```

`min_pages=3` 只拿掉三頁都有的選單與頁尾，內容都留著。`min_pages=2` 連那句兩個網站都有的句子也拿掉了，兩份都沒了：「出現不只一次」不一定是框架，可能是正文被轉載。門檻訂得越低，越容易誤傷；訂得越高，漏掉的框架越多。後面用真的網頁看這個取捨。

### 用指紋代替整行文字

一行文字可以很長，十億行放不進記憶體。所以計數表的鍵不用整行文字，用一個固定大小的 64 bit 整數，也就是這一行的指紋：

```python,ignore
{{#include ../../../src/llmz/pipeline.py:line_hash}}
```

`blake2b` 是標準函式庫 `hashlib` 裡的雜湊函式，`digest_size=8` 要它輸出 8 個 byte，`int.from_bytes` 再把這 8 個 byte 讀成一個整數。雜湊函式把任意長度的 bytes 變成固定長度的一串 bytes，同樣的輸入永遠得到同樣的輸出，不同的輸入幾乎不會撞在一起。不直接用 Python 內建的 `hash()`，因為字串的 `hash()` 每次啟動 Python 都不一樣（為了安全，這是刻意的），算出來的數字不能存檔，也不能拿到另一個程序比對。

指紋固定大小，所以可以放進 numpy 陣列、寫成檔案、依指紋分成很多份交給很多台機器數。代價是「幾乎」不會撞在一起，不是「絕對」不會。一個 64 bit 的指紋有 \\( 2^{64} \\) 種可能，\\( n \\) 種不同的行裡，預期有 \\( \binom{n}{2} / 2^{64} \\) 對撞在一起（生日問題）：

```python
for n in [74_283, 10**8, 10**10, 10**12]:
    expected = n * (n - 1) / 2 / 2**64
    print(f"{n:>17,} 種不同的行：預期有 {expected:.2g} 對撞在一起")
```

執行結果：

```text
           74,283 種不同的行：預期有 1.5e-10 對撞在一起
      100,000,000 種不同的行：預期有 0.00027 對撞在一起
   10,000,000,000 種不同的行：預期有 2.7 對撞在一起
1,000,000,000,000 種不同的行：預期有 2.7e+04 對撞在一起
```

74,283 種不同的行是下面真實網頁文字的量，撞在一起的機率可以忽略。到了一百億（\\( 10^{10} \\)）行，預期就會撞到好幾對，要加長到 128 bit。我們的規模用 64 bit 綽綽有餘。

### 在真的網頁上

用第 2 集存好的 `data/cc/pages.jsonl`（需要先執行 `tools/prepare_webtext.py`），在不同的 `min_pages` 下量拿掉多少：

```python,norun
import time

from llmz.pipeline import count_lines, dedup_lines, read_pages

pages = list(read_pages("data/cc/pages.jsonl"))
n_lines = sum(len(p.text.split("\n")) for p in pages)
n_chars = sum(len(p.text) for p in pages)

start = time.perf_counter()
counts = count_lines(pages)
seconds = time.perf_counter() - start
print(f"{len(pages)} 頁、{n_lines:,} 行、{n_chars:,} 個字元；{len(counts):,} 種不同的行，數一遍 {seconds:.2f} 秒")

for min_pages in [2, 3, 5, 10, 20]:
    kept = dedup_lines(pages, min_pages)
    kept_lines = sum(len(p.text.split("\n")) for p in kept)
    kept_chars = sum(len(p.text) for p in kept)
    print(
        f"min_pages={min_pages:2d}：拿掉 {n_lines - kept_lines:6,} 行（{1 - kept_lines / n_lines:.1%}）、"
        f"{n_chars - kept_chars:7,} 個字元（{1 - kept_chars / n_chars:.1%}），整頁丟掉 {len(pages) - len(kept)} 頁"
    )
```

M4 Max 上的輸出（數一遍的時間兩次執行都是 0.06 秒；其餘每次相同）：

```text
817 頁、108,198 行、4,413,284 個字元；74,283 種不同的行，數一遍 0.06 秒
min_pages= 2：拿掉 23,040 行（21.3%）、216,680 個字元（4.9%），整頁丟掉 10 頁
min_pages= 3：拿掉 14,974 行（13.8%）、 91,063 個字元（2.1%），整頁丟掉 6 頁
min_pages= 5：拿掉 10,413 行（9.6%）、 53,892 個字元（1.2%），整頁丟掉 6 頁
min_pages=10：拿掉  6,272 行（5.8%）、 29,058 個字元（0.7%），整頁丟掉 0 頁
min_pages=20：拿掉  4,183 行（3.9%）、 12,913 個字元（0.3%），整頁丟掉 0 頁
```

行數拿掉了 3.9% 到 21.3%，字元卻只少了 0.3% 到 4.9%：被拿掉的行大多很短。對 `min_pages=5` 拿掉的 10,413 行再看一眼它們是什麼樣的行、在頁面的哪裡：

```python,norun
from llmz.pipeline import count_lines, line_hash, read_pages

pages = list(read_pages("data/cc/pages.jsonl"))
counts = count_lines(pages)

removed = []  # (這一行, 它在頁面裡的位置)
for page in pages:
    lines = page.text.split("\n")
    for i, line in enumerate(lines):
        if counts[line_hash(line)] >= 5:
            where = "頭 5 行" if i < 5 else "尾 5 行" if i >= len(lines) - 5 else "中間"
            removed.append((line, where))

print(f"min_pages=5 拿掉的 {len(removed):,} 行：")
print(f"  最多 3 個字元：{sum(len(line) <= 3 for line, _ in removed) / len(removed):.0%}")
print(f"  至少 50 個字元：{sum(len(line) >= 50 for line, _ in removed) / len(removed):.1%}")
print(f"  一個字母都沒有（數字、符號）：{sum(not any(c.isalpha() for c in line) for line, _ in removed) / len(removed):.0%}")
for where in ["頭 5 行", "尾 5 行", "中間"]:
    print(f"  在頁面的{where}：{sum(w == where for _, w in removed) / len(removed):.0%}")
```

M4 Max 上的輸出（每次相同）：

```text
min_pages=5 拿掉的 10,413 行：
  最多 3 個字元：60%
  至少 50 個字元：0.9%
  一個字母都沒有（數字、符號）：51%
  在頁面的頭 5 行：7%
  在頁面的尾 5 行：5%
  在頁面的中間：88%
```

我們原本預期拿掉的是頁首的選單與頁尾，結果頭尾各 5 行加起來只占 12%。大多數是夾在正文中間的短行：一半連一個字母都沒有，是數字與分隔符號，像是頁碼與清單編號；它們出現在很多頁，只是因為很多頁都用得到同樣的數字。拿掉它們多半無傷，但不保證每次都對：如果清單或表格裡的「1」被拿掉，那一列就少了一格。

再看整頁被丟掉的：`min_pages=5` 的 6 頁是同一份只有一行的頁面的 6 份複本，`min_pages=2` 的 10 頁裡有 8 頁是兩組完全相同的頁面（6 份與 2 份）。每一份的每一行都出現在不只一頁，所以每一份都被拿光，一份都沒留下。逐行去重只問「這一行出現在幾頁」，不知道這幾頁是不是同一份內容的複本。整份重複的處理，留一份才對，這是下一集的事。

### 這個版本做不到的事

- **差一點點的行**：「3 comments」與「5 comments」、帶日期或計數器的行，每一頁都不一樣，exact 的比對抓不到。
- **整份重複**：上面看到的，全部拿光，而不是留一份。要留一份，就得先知道哪些頁面是同一份。
- **正文裡的洞**：從文章中間拿走一行，前後文可能接不起來。T5 論文（Raffel 等人，2020）處理 C4 的做法是以三個連續句子為單位，重複的片段只留一份。本書不做這麼細。
- **記憶體**：計數表整個放在記憶體。十億行以上的規模，要依指紋把行分成很多份，一份一份數。

## 重點整理

- 逐行 exact dedup：兩行一字不差才算重複。掃兩遍：先數每一行出現在幾頁，再拿掉出現在至少 `min_pages` 頁的行，一行都不剩的頁面整頁丟掉；同一頁自己重複的行只算一頁。
- 用 `hashlib.blake2b` 的 64 bit 指紋代替整行文字，大小固定、可以存檔與分片；不用內建 `hash()`，因為它每次啟動都不一樣。64 bit 在十億行以上的規模會有撞到的可能。
- 在這 817 頁上，`min_pages=5` 拿掉 9.6% 的行、1.2% 的字元；被拿掉的行 60% 不超過 3 個字元，頁首頁尾只占 12%，其餘是夾在正文中間的數字與短行。
- `min_pages` 太低會誤傷被轉載的正文，也會把整份重複的頁面兩份都拿光；整份重複與「差一點點」的近似重複，交給下一集的 MinHash。
