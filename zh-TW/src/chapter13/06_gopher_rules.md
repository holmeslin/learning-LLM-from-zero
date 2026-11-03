# Gopher 品質規則

## 本集目標

實作 DeepMind 在 Gopher 論文裡用來篩網頁的七條品質規則，讓函式回傳「有沒有通過」與「沒通過哪幾條」，再用它量出：我們那 817 頁網頁，每一條規則各刪掉多少。

## 正文

### 先用規則，再談模型

上一集留下來的 817 頁，什麼都有：正常的文章、一整頁的選單、商品價目表、一串關鍵字堆起來的垃圾頁。要把它們分出高低，最便宜的做法是寫規則：數詞數、看平均詞長、看有幾行以「…」結尾。每條規則都簡單到一眼看得懂，哪份文件為什麼被刪，也講得出理由。

Gopher 是 DeepMind 在 2021 年發表的 2800 億參數語言模型。它的論文（Rae 等人，*Scaling Language Models: Methods, Analysis & Insights from Training Gopher*，arXiv 2112.11446）附錄 A.1.1 說明了他們怎麼篩 MassiveWeb 這個網頁資料集：先濾掉非英文的文件、轉成文字，然後套用幾條「簡單、容易理解」的規則。論文講得很清楚的有這七條：

| 規則名稱 | 沒通過的條件 |
| --- | --- |
| `word_count` | 詞數不在 50 到 100,000 之間 |
| `mean_word_length` | 平均詞長不在 3 到 10 個字元之間 |
| `symbol_ratio` | `#` 的數量除以詞數大於 0.1，或「…」（含 `...`）的數量除以詞數大於 0.1 |
| `bullet_lines` | 超過 90% 的行以項目符號開頭 |
| `ellipsis_lines` | 超過 30% 的行以「…」結尾 |
| `alpha_words` | 少於 80% 的詞含有至少一個字母 |
| `stop_words` | `the`、`be`、`to`、`of`、`and`、`that`、`have`、`with` 這 8 個英文詞，出現的不到 2 個 |

論文沒有交代的細節要自己決定：「詞」怎麼切、哪些符號算項目符號。下面是我們的選擇。

### `gopher_check`

```python,ignore
{{#include ../../../src/llmz/pipeline.py:gopher_check}}
```

- **詞**就是用空白切出來的一段（`text.split()`），**行**是非空的一行。上一集的 `html_to_text` 已經讓每個區塊自成一行，這裡剛好用得上。
- 每條規則各自判斷，所以一份文件可以同時沒通過好幾條。回傳 `(not failed, failed)`：第一個值是「全部通過」，第二個是沒通過的規則名稱，依上面表格的順序。
- 邊界照論文的字面：「不在 50 到 100,000 之間」，所以剛好 50 個詞通過、49 個沒通過；「大於 0.1」，所以剛好 0.1 通過；「少於 80%」，所以剛好 80% 通過。測試 `tests/test_pipeline.py` 對每條規則都試了邊界的兩側。
- 項目符號我們認 `•●▪‣·-*` 這幾個（`line.startswith(BULLETS)` 的 `startswith` 可以直接吃一個 tuple，只要有一個符合就是 `True`）。`-` 與 `*` 也算是一種取捨：Markdown 的清單用它們。
- `stop_words` 先用 `w.strip(string.punctuation).lower()` 把詞兩邊的標點拿掉、轉小寫，再和那 8 個詞取交集。同一個停用詞出現一百次也只算一個。
- 空字串也能算：比例的分母用 `max(..., 1)`，空字串會沒通過 `word_count`、`mean_word_length`、`alpha_words` 與 `stop_words`，不會除以 0。

### 試用

六份原創的小文件，各自踩到不同的規則：

```python
from llmz.pipeline import gopher_check

docs = {
    "正常的段落": (
        "The garden behind our house is small, but my sister and I spend most of the summer there.\n"
        "We water the tomatoes in the morning, and in the evening we sit on the steps to watch the bees that come to the lavender.\n"
        "Last week she found a snail on the fence and wanted to keep it, so we built it a little home out of an old flowerpot.\n"
        "It has not left the pot yet, and I think it likes the shade."
    ),
    "網站選單": "Home\nAbout us\nProducts\nBlog\nContact\nLogin\nCart (0)\nPrivacy\nTerms\nSitemap",
    "標籤堆疊": " ".join([f"#{w}" for w in "sale deal cheap best price new top hot free gift offer shop buy now today only limited".split()] * 3)
    + " the best deals and the best offers with free shipping to you",
    "搜尋結果摘要": "\n".join(
        f"{s} and the rest of the story that you will have to read to find out…"
        for s in ["Review of the new phone:", "How to bake bread at home", "The history of the bicycle", "Ten places to visit in spring"]
    ),
    "價目表": "\n".join(f"Item {i}: 12.50 USD, 3 pcs, 40 x 60 cm, code 1{i}3{i}" for i in range(1, 9)),
    "中文文章": "我們家後面的小花園不大，但是整個夏天，我和姊姊大部分的時間都待在那裡。早上我們幫番茄澆水，傍晚就坐在台階上，看蜜蜂飛到薰衣草上。",
}
for name, text in docs.items():
    passed, failed = gopher_check(text)
    print(f"{name}：{len(text.split())} 個詞，{'通過' if passed else '沒通過'} {failed}")
```

執行結果：

```text
正常的段落：83 個詞，通過 []
網站選單：12 個詞，沒通過 ['word_count', 'stop_words']
標籤堆疊：63 個詞，沒通過 ['symbol_ratio']
搜尋結果摘要：82 個詞，沒通過 ['ellipsis_lines']
價目表：96 個詞，沒通過 ['alpha_words', 'stop_words']
中文文章：1 個詞，沒通過 ['word_count', 'mean_word_length', 'stop_words']
```

選單太短而且沒有英文的停用詞；標籤堆疊的 `#` 太多；搜尋結果的每一行都以「…」收尾；價目表一半的詞是數字，還缺停用詞。最後一行最值得看：中文整段沒有空白，被當成「一個詞」，所以詞數不夠、平均詞長是整段的長度、也沒有任何一個英文停用詞。這不是中文文章不好，而是這組規則是為英文寫的，論文也是先過濾掉非英文的文件才套用它們。實際的管線會把語言辨識（本章第 3 集）放在前面。

### 在真的網頁上

把 817 頁都跑一遍：每條規則各刪掉幾頁、有幾頁「只有這一條」沒通過（如果拿掉這條規則，就會多留下這幾頁）。我們還沒做語言辨識，所以再用一個粗糙的代替品：文字裡的字母有 99% 以上是 ASCII 的頁面，大概是英文。最後把規則也拿去量本書一直在用的標準語料（TinyStories），看它的故事會被刪掉多少。

```python,norun
from collections import Counter

from llmz.pipeline import gopher_check, read_pages

RULES = ["word_count", "mean_word_length", "symbol_ratio", "bullet_lines", "ellipsis_lines", "alpha_words", "stop_words"]


def report(name, failed_lists):
    print(f"{name}：{len(failed_lists)} 份，沒通過至少一條的 {sum(map(bool, failed_lists))} 份")
    for rule in RULES:
        n = sum(rule in failed for failed in failed_lists)
        alone = sum(failed == [rule] for failed in failed_lists)
        print(f"  {rule:16s} {n:5d} 份沒通過，其中只有這一條沒通過的 {alone} 份")
    print("  同時沒通過幾條：", sorted(Counter(map(len, failed_lists)).items()))


pages = list(read_pages("data/cc/pages.jsonl"))
failed_lists = [gopher_check(p.text)[1] for p in pages]
report("網頁", failed_lists)
kept = sum(len(p.text) for p, failed in zip(pages, failed_lists) if not failed)
print(f"通過的頁面佔全部 {sum(len(p.text) for p in pages):,} 個字元的 {kept / sum(len(p.text) for p in pages):.1%}")


def ascii_share(text):  # 文字裡的字母有多少比例是 ASCII：粗糙地代替語言辨識
    letters = [c for c in text if c.isalpha()]
    return sum(c.isascii() for c in letters) / max(len(letters), 1)


latin = [failed for p, failed in zip(pages, failed_lists) if ascii_share(p.text) >= 0.99]
report("ASCII 字母占 99% 以上的網頁", latin)

stories = [d.strip() for d in open("data/valid.txt", encoding="utf-8").read().split("<|endoftext|>") if d.strip()]
report("標準語料的故事", [gopher_check(s)[1] for s in stories])
```

M4 Max 上的輸出（檔案固定，每次相同）：

```text
網頁：817 份，沒通過至少一條的 584 份
  word_count         138 份沒通過，其中只有這一條沒通過的 16 份
  mean_word_length    98 份沒通過，其中只有這一條沒通過的 2 份
  symbol_ratio        14 份沒通過，其中只有這一條沒通過的 0 份
  bullet_lines         0 份沒通過，其中只有這一條沒通過的 0 份
  ellipsis_lines       8 份沒通過，其中只有這一條沒通過的 0 份
  alpha_words        122 份沒通過，其中只有這一條沒通過的 23 份
  stop_words         539 份沒通過，其中只有這一條沒通過的 284 份
  同時沒通過幾條： [(0, 233), (1, 325), (2, 194), (3, 54), (4, 11)]
通過的頁面佔全部 4,413,284 個字元的 45.2%
ASCII 字母占 99% 以上的網頁：273 份，沒通過至少一條的 110 份
  word_count          55 份沒通過，其中只有這一條沒通過的 13 份
  mean_word_length     2 份沒通過，其中只有這一條沒通過的 0 份
  symbol_ratio         6 份沒通過，其中只有這一條沒通過的 0 份
  bullet_lines         0 份沒通過，其中只有這一條沒通過的 0 份
  ellipsis_lines       6 份沒通過，其中只有這一條沒通過的 0 份
  alpha_words         34 份沒通過，其中只有這一條沒通過的 15 份
  stop_words          81 份沒通過，其中只有這一條沒通過的 31 份
  同時沒通過幾條： [(0, 163), (1, 59), (2, 34), (3, 11), (4, 6)]
標準語料的故事：27630 份，沒通過至少一條的 1 份
  word_count           1 份沒通過，其中只有這一條沒通過的 1 份
  mean_word_length     0 份沒通過，其中只有這一條沒通過的 0 份
  symbol_ratio         0 份沒通過，其中只有這一條沒通過的 0 份
  bullet_lines         0 份沒通過，其中只有這一條沒通過的 0 份
  ellipsis_lines       0 份沒通過，其中只有這一條沒通過的 0 份
  alpha_words          0 份沒通過，其中只有這一條沒通過的 0 份
  stop_words           0 份沒通過，其中只有這一條沒通過的 0 份
  同時沒通過幾條： [(0, 27629), (1, 1)]
```

- 817 頁裡有 584 頁（71%）至少沒通過一條，通過的 233 頁。通過的 233 頁只占頁數的 28.5%，卻有 45.2% 的文字：通過的頁面比較長。
- 最大的一刀是 `stop_words`，539 頁沒通過，比其他規則加起來還多。這是因為多數頁面根本不是英文。限定在 ASCII 字母占 99% 以上的 273 頁，它降到 81 頁，沒通過至少一條的剩 110 頁，通過的有 163 頁。在這 273 頁裡，`word_count`（55 頁）與 `alpha_words`（34 頁）才是接下來最常刪的。
- `bullet_lines` 一頁也沒刪：`html_to_text` 把每個 `<li>` 變成一行、但不留項目符號，文字裡本來就沒有 `•`。如果你的文字抽取會保留符號，這條才會派上用場。`symbol_ratio` 與 `ellipsis_lines` 各刪 14 與 8 頁，而且沒有一頁是「只有這一條」，它們抓到的頁面，別的規則都已經抓到了。
- 大多數沒通過的頁面只踩一條規則（325 頁），其次是兩條（194 頁）。規則之間重疊很多：一頁很短的選單，會同時缺詞數、缺停用詞、平均詞長也怪。
- 標準語料的 27,630 個故事，只有 1 個沒通過，原因是不到 50 個詞。這是個好兆頭：規則對乾淨的英文文章不會誤傷。

### 這組規則的限制

- **它只看形狀，不看內容**。一段把英文詞隨機排起來、但有停用詞、詞長正常、沒有符號的亂碼，會通過全部七條；把同一行重複幾十次也會通過。下一集的分類器看得到哪些詞常一起出現，重複的內容則留給本章後面的 exact dedup 與 MinHash。練習題會讓你親手造一份通過的垃圾。
- **門檻是經驗值**。Gopher 的 50 詞與 3 到 10 字元，是針對它們的英文網頁資料決定的。換一種語言、換一種文體（詩、程式碼、清單、對話），好文件也會被刪掉。拿到新資料，先像上面那樣量一次每條規則刪多少，再決定要不要照用。
- **英文限定**。`stop_words` 是整組規則裡最強的一條，同時也最偏英文。

## 重點整理

- Gopher 的七條規則：詞數 50～100,000、平均詞長 3～10、`#` 與「…」各自的符號比例不超過 0.1、項目符號開頭的行不超過 90%、「…」結尾的行不超過 30%、至少 80% 的詞含字母、8 個英文停用詞至少出現 2 個。出處是 Rae 等人 2021 的附錄 A.1.1。
- `gopher_check(text)` 回傳 `(是否全部通過, 沒通過的規則名稱)`；詞用空白切、行用非空的一行。每條規則各自判斷，邊界照論文的字面（「大於」「少於」「不在……之間」）。
- 在我們的 817 頁上，584 頁至少沒通過一條；`stop_words` 一條就刪 539 頁，因為多數頁面不是英文，限定在大概是英文的 273 頁後降到 81 頁。標準語料的 27,630 個故事只有 1 個沒通過。
- 這些規則只看形狀：亂碼可以全部通過，非英文會被誤刪，門檻也是針對英文網頁調的。下一集改用資料來定義「好」，訓練一個品質分類器。
