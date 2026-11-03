# 組成資料管線

## 本集目標

把前九集的步驟接成一條管線：決定它們的順序，量出每一步留下多少頁與多少 token，最後輸出成第 3 章格式的 token 檔，交給第 5 章的訓練程式讀。

## 正文

### 每一步：一批頁面進、一批頁面出

前面各集的函式長得不一樣：`dedup_lines` 吃一批頁面，`gopher_check` 吃一頁的文字、回傳通過與否，`mask_pii` 吃一段文字、回傳遮蔽後的文字與計數。要接起來，先把它們都變成同一種形狀：一批 `Page` 進、一批 `Page` 出，丟掉的頁面不出現，改過的頁面是新的 `Page`。`dedup_lines` 與 `drop_near_duplicates` 本來就是這個形狀，其餘的用幾個小函式包起來：

```python,ignore
{{#include ../../../src/llmz/pipeline.py:standard_steps}}
```

`keep_ascii` 多出來的原因，後面看到真的網頁再說。接起來只要一個函式：

```python,ignore
{{#include ../../../src/llmz/pipeline.py:run_pipeline}}
```

`run_pipeline` 依序執行每一步，每一步之後記一筆 `StepReport`：名稱、剩幾頁、剩下的文字有多大。大小預設是字元數；要量 token，就傳 `size=lambda text: len(tokenizer.encode(text))`。

先用九頁原創的假網頁走一遍。每一頁都有同樣兩行選單與頁尾；除了五篇英文文章，還有一頁文章加了一個字的複本、一頁德文、一頁越南文、一頁只有選單、一頁只有一句話。文章用手填的權重當「品質分類器」：出現 `library`、`garden`、`bakery`、`bread`、`wind`、`rain` 這幾個詞就加分，不訓練：

```python
import torch

from llmz.pipeline import (
    Page,
    QualityClassifier,
    dedup_lines,
    drop_near_duplicates,
    hash_ngrams,
    keep_ascii,
    keep_by_quality,
    keep_gopher,
    keep_language,
    mask_pii_pages,
    run_pipeline,
    train_langid,
)

NAV = "Home | About | Contact\nCopyright 2026 Example Press"
LIBRARY = (
    "The old library at the end of the street has a small reading room with a large window that faces the garden. "
    "Every morning the librarian opens the door and sets out the newspapers, and a few regular visitors come in to read "
    "while the kettle warms in the corner. People have said that the room is the quietest place in town."
)
BAKERY = (
    "Our bakery opens before sunrise, and the smell of warm bread fills the whole street by the time the first customers arrive. "
    "We make every loaf by hand with flour from a mill that is only a few miles away, and we have never used anything that "
    "the neighbors could not pronounce. Questions about an order? Write to baker@example.com and we will answer the same day."
)
BIKE = (
    "To fix a flat tire on the road, take the wheel off, pull one side of the tire away from the rim, and find the hole in the "
    "tube before you do anything else. Patch it with glue and a small piece of rubber, wait until the glue is dry, and put "
    "everything back with just enough air that the tire keeps its shape."
)
WEATHER = (
    "A small weather station on the roof records the temperature, the wind, and the rain every ten minutes, and it sends the "
    "numbers to a screen in the hallway. Over the years the students have learned that the wind from the north is the one "
    "that brings the coldest mornings, and that the rain usually comes before the clouds look dark."
)
GERMAN = (
    "Die alte Bibliothek am Ende der Straße hat einen kleinen Lesesaal mit einem großen Fenster, das zum Garten zeigt. "
    "Jeden Morgen öffnet die Bibliothekarin die Tür und legt die Zeitungen aus, und einige Stammgäste kommen herein, "
    "um zu lesen, während das Wasser in der Ecke warm wird. Man sagt, dass der Raum der ruhigste Ort der Stadt ist."
)
VIETNAMESE = (
    "Thư viện cũ ở cuối con phố có một phòng đọc nhỏ với cửa sổ lớn nhìn ra khu vườn. Mỗi buổi sáng, cô thủ thư mở cửa và bày "
    "báo ra bàn, vài người khách quen vào đọc trong khi ấm nước nóng dần ở góc phòng. Mọi người nói rằng đó là nơi yên tĩnh nhất "
    "trong thị trấn, và không ai muốn thay đổi điều đó. Vào những ngày mưa, bọn trẻ theo bố mẹ đến mượn sách tranh và nghe cô đọc truyện."
)
pages = [
    Page("http://a.example/library", f"{NAV}\n{LIBRARY}"),
    Page("http://a.example/bakery", f"{NAV}\n{BAKERY}"),
    Page("http://a.example/bike", f"{NAV}\n{BIKE}"),
    Page("http://a.example/weather", f"{NAV}\n{WEATHER}"),
    Page("http://a.example/library?ref=mail", f"{NAV}\n{LIBRARY.replace('quietest', 'calmest')}"),
    Page("http://b.example/bibliothek", f"{NAV}\n{GERMAN}"),
    Page("http://c.example/thu-vien", f"{NAV}\n{VIETNAMESE}"),
    Page("http://a.example/empty", NAV),
    Page("http://a.example/thanks", f"{NAV}\nThanks for visiting! Call 555-010-1234."),
]

quality = QualityClassifier(1 << 12)
with torch.no_grad():
    for word in ["library", "garden", "bakery", "bread", "wind", "rain"]:
        quality.weight[hash_ngrams(word, 1 << 12)] = 1.0

steps = [
    ("dedup_lines", dedup_lines),
    ("drop_near_duplicates", drop_near_duplicates),
    ("語言：en", keep_language(train_langid())),
    ("ASCII 字母 >= 90%", keep_ascii(0.9)),
    ("Gopher 規則", keep_gopher),
    ("品質分類器（Pareto，α=2）", keep_by_quality(quality, alpha=2)),
    ("遮蔽 PII", mask_pii_pages),
]
kept, report = run_pipeline(pages, steps)
for row in report:
    print(f"{row.name}：{row.docs} 頁，{row.size} 個字元")
print([p.url for p in kept])
print("遮蔽後：", [line for p in kept for line in p.text.split("\n") if "<EMAIL>" in line][0][-50:])
```

執行結果：

```text
輸入：9 頁，2864 個字元
dedup_lines：8 頁，2397 個字元
drop_near_duplicates：7 頁，2075 個字元
語言：en：6 頁，1739 個字元
ASCII 字母 >= 90%：5 頁，1362 個字元
Gopher 規則：4 頁，1323 個字元
品質分類器（Pareto，α=2）：3 頁，1016 個字元
遮蔽 PII：3 頁，1006 個字元
['http://a.example/library', 'http://a.example/bakery', 'http://a.example/weather']
遮蔽後：  Write to <EMAIL> and we will answer the same day.
```

每一步各擋下一種頁面：選單與頁尾在 9 頁都出現（`min_pages=5`），拿掉之後只有選單的那頁一行不剩，整頁丟掉；加了一個字的複本留給 `drop_near_duplicates`；德文由語言這一步擋下，越南文沒有「都不是」的選項，被判成英文，由 `keep_ascii` 擋下（越南文的字母大半有重音，不是 ASCII）；只有一句話的頁面詞數不到 50，沒通過 Gopher 規則；剩下的四篇文章，品質分類器依分數排名次再抽樣，這次留下三篇（`bike` 沒有任何一個加分的詞，名次最低，沒被抽中）；最後那一頁的 email 變成 `<EMAIL>`。

### 順序

七個步驟的順序有三個考慮，每個都用 817 頁量過。

**只看頁面、不改文字的步驟，順序不影響結果**。語言與 Gopher 規則都只是判斷，通過就留下，不通過就丟掉，不碰文字，所以先做哪一個，留下的頁面都一樣。第 6 集說 Gopher 的規則是為英文寫的，所以本章先辨識語言，讓 Gopher 只看英文；但從結果看，換過來也行，差在時間：語言辨識的時間跟頁數成正比，先用便宜的 Gopher 規則把頁數砍掉，辨識就快很多（下面表裡的 `GLDN`）。

**會改文字的步驟，要放在判斷文字的步驟前面**。`dedup_lines` 拿掉每頁共用的行，頁面變短，Gopher 的詞數規則（至少 50 個詞）看到的是拿掉之後的文字。把順序換過來，Gopher 先判斷，頁面帶著選單與頁尾通過，之後選單被拿掉，頁面可能只剩幾十個詞，卻已經過了關。下表每一列是一種順序（`D` 是 `dedup_lines`、`N` 是 `drop_near_duplicates`、`L` 是語言、`G` 是 Gopher 規則），印出每一步之後剩幾頁，並且把最後的頁面再用 Gopher 規則檢查一次：

```python,norun
import time

from llmz.pipeline import (
    dedup_lines,
    drop_near_duplicates,
    gopher_check,
    keep_gopher,
    keep_language,
    read_pages,
    run_pipeline,
    train_langid,
)

pages = list(read_pages("data/cc/pages.jsonl"))
steps = {
    "D": ("dedup_lines", dedup_lines),
    "N": ("drop_near_duplicates", drop_near_duplicates),
    "L": ("語言：en", keep_language(train_langid())),
    "G": ("Gopher 規則", keep_gopher),
}  # D：逐行去重，N：近似重複，L：語言，G：Gopher 規則

for order in ["DNLG", "DLGN", "NDLG", "LGDN", "LGND", "GLDN"]:
    start = time.perf_counter()
    kept, report = run_pipeline(pages, [steps[c] for c in order])
    seconds = time.perf_counter() - start
    failing = sum(not gopher_check(p.text)[0] for p in kept)  # 最後的文字再用 Gopher 的規則檢查一次
    print(f"{order}：{' → '.join(str(r.docs) for r in report)} 頁；最後的 {len(kept)} 頁裡有 {failing} 頁沒通過 Gopher（{seconds:.1f} 秒）")

lg, _ = run_pipeline(pages, [steps["L"], steps["G"]])
gl, _ = run_pipeline(pages, [steps["G"], steps["L"]])
print("先 L 再 G，與先 G 再 L，留下的是同一批頁面：", {p.url for p in lg} == {p.url for p in gl})
```

M4 Max 上的輸出（頁數每次相同；時間是這一次的，當時機器上還有別的程式在跑，所以只能看出 `GLDN` 大約快一半）：

```text
DNLG：817 → 811 → 808 → 318 → 180 頁；最後的 180 頁裡有 0 頁沒通過 Gopher（16.0 秒）
DLGN：817 → 811 → 319 → 180 → 180 頁；最後的 180 頁裡有 0 頁沒通過 Gopher（12.6 秒）
NDLG：817 → 802 → 802 → 316 → 179 頁；最後的 179 頁裡有 0 頁沒通過 Gopher（13.6 秒）
LGDN：817 → 328 → 183 → 183 → 183 頁；最後的 183 頁裡有 3 頁沒通過 Gopher（13.0 秒）
LGND：817 → 328 → 183 → 182 → 182 頁；最後的 182 頁裡有 3 頁沒通過 Gopher（13.0 秒）
GLDN：817 → 233 → 183 → 183 → 183 頁；最後的 183 頁裡有 3 頁沒通過 Gopher（6.8 秒）
先 L 再 G，與先 G 再 L，留下的是同一批頁面： True
```

前三列（`DNLG`、`DLGN`、`NDLG`）去重在 Gopher 前面，最後的頁面全部通過 Gopher；後三列（`LGDN`、`LGND`、`GLDN`）去重在後面，最後各有 3 頁沒通過，是頁面被拿掉共用的行之後才變短的。最後一行確認了第一點：`L` 與 `G` 換過來，留下的是同一批頁面（`GLDN` 與 `LGDN` 都是 183 頁）。

**去重的順序，在整條管線裡差得很小**。第 9 集單獨量 `drop_near_duplicates`：直接用在 817 頁上丟掉 15 頁，先做 `dedup_lines` 只丟 3 頁。放進管線，先 `dedup_lines` 的 `DNLG` 與先近似重複的 `NDLG`，結果是 180 頁與 179 頁，只差 1 頁。如果去重放在最後（`LGDN`），兩個去重步驟都找不到東西（183 → 183 → 183）：第 9 集找到的那些重複，到去重這一步已經被語言或 Gopher 規則擋掉了。單獨量一步看到的差別，會比放進整條管線之後大得多。

我們選 `DNLG`：先 `dedup_lines`，再 `drop_near_duplicates`，然後語言、Gopher。代價是去重要處理 817 頁，而不是語言與 Gopher 之後的 180 頁；這個規模上無所謂（第 9 集量過，`drop_near_duplicates` 在 817 頁上要 0.5～1.2 秒）。規模大到去重變得很貴的時候，會想先用便宜的規則把頁數砍下來，就要接受上面「最後有幾頁沒通過」的後果，並在去重之後再檢查一次 Gopher 的規則。

其他兩個位置不用量就能決定。遮蔽 PII（`mask_pii_pages`）放最後：它只是把一個詞換成另一個詞，不決定頁面去留，放最後只需要處理留下來的頁面。有害內容分類（第 5 集）本來會放在 Gopher 規則之後、品質分類器之前，本書的管線沒有這一步。

### 品質分類器的分數怎麼用

第 7 集的 `quality.pt` 把標準語料的故事當正例、817 頁網頁當負例，最後只有 2 頁網頁的分數大於 0。在本章去重、語言、Gopher 之後剩下的 164 頁上，分數是這個樣子：

```python,norun
import numpy as np
import torch

from llmz.pipeline import (
    QualityClassifier,
    dedup_lines,
    drop_near_duplicates,
    keep_ascii,
    keep_gopher,
    keep_language,
    pareto_keep,
    read_pages,
    run_pipeline,
    train_langid,
)
from llmz.tokenizer import Tokenizer

tokenizer = Tokenizer.load("data/tokenizer.json")
state = torch.load("runs/quality/quality.pt", weights_only=True)
quality = QualityClassifier(state["weight"].numel())
quality.load_state_dict(state)

pages = list(read_pages("data/cc/pages.jsonl"))
before_quality = [
    ("dedup_lines", dedup_lines),
    ("drop_near_duplicates", drop_near_duplicates),
    ("語言：en", keep_language(train_langid())),
    ("ASCII 字母", keep_ascii(0.9)),
    ("Gopher 規則", keep_gopher),
]
survivors, _ = run_pipeline(pages, before_quality)  # 品質分類器之前的所有步驟
scores = quality.score([p.text for p in survivors])
prob = torch.sigmoid(scores).numpy()
tokens = np.array([len(tokenizer.encode(p.text)) for p in survivors])
print(f"{len(survivors)} 頁，{tokens.sum():,} 個 token")
print(f"分數（logit）：最小 {scores.min():.1f}，中位數 {scores.median():.1f}，最大 {scores.max():.1f}")
print(f"機率：中位數 {np.median(prob):.4f}，最大 {prob.max():.3f}，超過 0.5 的有 {int((prob > 0.5).sum())} 頁，超過 0.01 的有 {int((prob > 0.01).sum())} 頁")

raw = [int((np.random.default_rng(seed).pareto(9, len(prob)) > 1 - prob).sum()) for seed in range(5)]
print(f"\nGPT-3 的規則（document_score 直接用機率）、α=9：種子 0～4 各留下 {raw} 頁")

print("\n改用名次（pareto_keep）：")
for alpha in [2, 4, 9]:
    expected = len(prob) * (1 - 2 ** (1 - alpha)) / (alpha - 1)
    keeps = [pareto_keep(scores, alpha, seed) for seed in range(5)]
    print(f"α={alpha}：理論上留下 {expected:.1f} 頁；種子 0～4 各留下 {[int(k.sum()) for k in keeps]} 頁、{[int(tokens[k].sum()) for k in keeps]} 個 token")
```

M4 Max 上的輸出（每次相同）：

```text
164 頁，471,310 個 token
分數（logit）：最小 -7.5，中位數 -5.5，最大 -1.6
機率：中位數 0.0041，最大 0.163，超過 0.5 的有 0 頁，超過 0.01 的有 40 頁

GPT-3 的規則（document_score 直接用機率）、α=9：種子 0～4 各留下 [0, 1, 0, 0, 1] 頁

改用名次（pareto_keep）：
α=2：理論上留下 82.0 頁；種子 0～4 各留下 [89, 86, 80, 84, 87] 頁、[288356, 282161, 291498, 306437, 328058] 個 token
α=4：理論上留下 47.8 頁；種子 0～4 各留下 [55, 52, 37, 53, 55] 頁、[108004, 228585, 131058, 201199, 209552] 個 token
α=9：理論上留下 20.4 頁；種子 0～4 各留下 [26, 22, 17, 23, 24] 頁、[62830, 47164, 43176, 117709, 118540] 個 token
```

164 頁裡沒有一頁的機率超過 0.5，最高的只有 0.163，中位數 0.0041。第 7 集的練習把 GPT-3 的規則 `np.random.pareto(α) > 1 − document_score` 用在機率上：機率 0.0041 的頁面，α=9 時留下的機率是 \\( (2 - 0.0041)^{-9} \approx 0.2\% \\)。實測 164 頁在五個種子下只留下 0 到 1 頁。分類器沒有壞，是我們的「好」只有兒童故事，網頁對它來說都不像。

所以改用名次：把 164 頁依分數由低到高排，名次的百分位 \\( u \\)（最高分是 1，最低約 0）當作 `document_score`。名次 \\( u \\) 的頁面留下的機率是 \\( (2-u)^{-\alpha} \\)，整體約留下 \\( (1 - 2^{1-\alpha})/(\alpha-1) \\)：\\( \alpha=2 \\) 約一半，\\( \alpha=4 \\) 約 29%，\\( \alpha=9 \\) 約 12%。α 現在直接決定「留下多少」，不再取決於分類器的分數落在哪裡。

```python,ignore
{{#include ../../../src/llmz/pipeline.py:pareto_keep}}
```

用 10 萬個分數剛好等於名次的文件，看每一段名次的保留率，與理論值比：

```python
import numpy as np

from llmz.pipeline import pareto_keep

alpha, n = 4, 100_000
keep = pareto_keep(np.arange(n), alpha=alpha, seed=0)  # 分數 = 名次，由低到高
print(f"整體：留下 {keep.mean():.3f}，理論 {(1 - 2 ** (1 - alpha)) / (alpha - 1):.3f}")
for lo in [0.0, 0.2, 0.4, 0.6, 0.8]:
    hi = lo + 0.2
    exact = ((2 - hi) ** (1 - alpha) - (2 - lo) ** (1 - alpha)) / ((alpha - 1) * (hi - lo))  # (2-u)^-α 在這一段的平均
    print(f"名次 {lo:.0%}～{hi:.0%}：留下 {keep[int(lo * n) : int(hi * n)].mean():.3f}，理論 {exact:.3f}")
```

執行結果：

```text
整體：留下 0.291，理論 0.292
名次 0%～20%：留下 0.077，理論 0.077
名次 20%～40%：留下 0.124，理論 0.121
名次 40%～60%：留下 0.202，理論 0.200
名次 60%～80%：留下 0.351，理論 0.357
名次 80%～100%：留下 0.702，理論 0.702
```

最高的 20% 留下 7 成左右，最低的 20% 還有 8%：大部分資料來自高分的頁面，低分的也有機會，資料不會整個縮到「像故事」的樣子。這比一刀切的門檻柔和，代價是結果帶著隨機：上面五個種子在 α=4 時留下 37 到 55 頁，token 數從 10.8 萬到 22.9 萬，差一倍。頁數比 token 數穩定，因為頁面的長度差很多，一頁長的頁面進來或不進來就差很多 token。本章用種子 0，是這五個裡 token 最少的一個。

還有一件事要確認：`quality.pt` 是用這 817 頁當負例訓練的，現在拿它替這些頁面排名次，等於拿訓練資料打分。第 7 集的交叉驗證，是每一頁由沒看過它、也沒看過同一個網站的模型打分。兩種分數的名次差多少：

```python,norun
import random
import zlib
from urllib.parse import urlsplit

import numpy as np
import torch

from llmz.pipeline import (
    QualityClassifier,
    dedup_lines,
    drop_near_duplicates,
    keep_ascii,
    keep_gopher,
    keep_language,
    read_pages,
    run_pipeline,
    train_langid,
    train_quality,
)

pages = list(read_pages("data/cc/pages.jsonl"))
steps = [
    ("dedup_lines", dedup_lines),
    ("drop_near_duplicates", drop_near_duplicates),
    ("語言：en", keep_language(train_langid())),
    ("ASCII 字母", keep_ascii(0.9)),
    ("Gopher 規則", keep_gopher),
]
survivors, _ = run_pipeline(pages, steps)
state = torch.load("runs/quality/quality.pt", weights_only=True)
quality = QualityClassifier(state["weight"].numel())
quality.load_state_dict(state)
in_sample = quality.score([p.text for p in survivors]).numpy()  # 看過這些頁面（當負例）的模型打的分

# 第 7 集的 5 份交叉驗證：同一個網站一定在同一份，每一頁由「沒看過它」的模型打分
stories = [d.strip() for d in open("data/train.txt", encoding="utf-8").read().split("<|endoftext|>") if d.strip()]
stories = random.Random(0).sample(stories, 20000)
fold = lambda page: zlib.crc32(urlsplit(page.url).hostname.encode()) % 5
cross = {}
for k in range(5):
    train_web = [p.text for p in pages if fold(p) != k]
    model, _ = train_quality(stories + train_web, [1] * len(stories) + [0] * len(train_web))
    held_out = [p for p in survivors if fold(p) == k]
    cross.update(zip((p.url for p in held_out), model.score([p.text for p in held_out]).tolist()))
held_out_scores = np.array([cross[p.url] for p in survivors])

rank = lambda x: x.argsort().argsort()
print(f"{len(survivors)} 頁：兩種分數的名次相關係數 {np.corrcoef(rank(in_sample), rank(held_out_scores))[0, 1]:.3f}")
top = len(survivors) // 5
overlap = set(np.argsort(-in_sample)[:top]) & set(np.argsort(-held_out_scores)[:top])
print(f"各自分數最高的前 20%（{top} 頁）重疊 {len(overlap)} 頁")
```

M4 Max 上的輸出（每次相同；訓練 5 個模型大約 1 分鐘）：

```text
164 頁：兩種分數的名次相關係數 0.961
各自分數最高的前 20%（32 頁）重疊 29 頁
```

名次相關係數 0.961，最高的 32 頁有 29 頁重疊。用 `quality.pt` 排名次，和沒看過資料的模型差不多，所以可以拿來用。

### 在真的網頁上

把七個步驟接起來，大小用 token 量：

```python,norun
import functools
import os
import time

import torch

from llmz.data import get_batch, load_tokens, tokenize_to_file
from llmz.pipeline import (
    QualityClassifier,
    dedup_lines,
    drop_near_duplicates,
    keep_ascii,
    keep_by_quality,
    keep_gopher,
    keep_language,
    mask_pii_pages,
    read_pages,
    run_pipeline,
    train_langid,
    write_corpus_text,
)
from llmz.tokenizer import Tokenizer

tokenizer = Tokenizer.load("data/tokenizer.json")
state = torch.load("runs/quality/quality.pt", weights_only=True)
quality = QualityClassifier(state["weight"].numel())
quality.load_state_dict(state)

pages = list(read_pages("data/cc/pages.jsonl"))
steps = [
    ("dedup_lines（min_pages=5）", dedup_lines),
    ("drop_near_duplicates", drop_near_duplicates),
    ("語言：en", keep_language(train_langid())),
    ("ASCII 字母 >= 90%", keep_ascii(0.9)),
    ("Gopher 規則", keep_gopher),
    ("品質分類器（Pareto，α=4）", keep_by_quality(quality, alpha=4)),
    ("遮蔽 PII", mask_pii_pages),
]  # 有害內容分類（第 5 集）本來在品質分類器前面；本書的管線沒有這一步

count_tokens = functools.cache(lambda text: len(tokenizer.encode(text)))  # 文字沒變的頁面不用重新 encode
start = time.perf_counter()
kept, report = run_pipeline(pages, steps, size=count_tokens)
seconds = time.perf_counter() - start

for row in report:
    print(f"{row.name}：{row.docs} 頁（{row.docs / report[0].docs:.1%}），{row.size:,} 個 token（{row.size / report[0].size:.1%}）")
print(f"整條管線 {seconds:.0f} 秒")

n_pages = write_corpus_text(kept, "data/cc/web.txt")
n_tokens = tokenize_to_file(tokenizer, "data/cc/web.txt", "data/cc/web.bin")
print(f"data/cc/web.bin：{n_pages} 頁、{n_tokens:,} 個 token，{os.path.getsize('data/cc/web.txt') / n_tokens:.2f} bytes/token")

data = load_tokens("data/cc/web.bin")  # 第 3 章的讀法
x, y = get_batch(data, 4, 256, torch.device("cpu"))  # 第 5 章的取 batch
print(data.dtype, tuple(x.shape), tuple(y.shape), bool((x[:, 1:] == y[:, :-1]).all()))
```

M4 Max 上的輸出（頁數與 token 數每次相同；「整條管線」的秒數三次執行是 14～16 秒）：

```text
輸入：817 頁（100.0%），4,733,998 個 token（100.0%）
dedup_lines（min_pages=5）：811 頁（99.3%），4,669,430 個 token（98.6%）
drop_near_duplicates：808 頁（98.9%），4,661,595 個 token（98.5%）
語言：en：318 頁（38.9%），1,955,349 個 token（41.3%）
ASCII 字母 >= 90%：262 頁（32.1%），617,073 個 token（13.0%）
Gopher 規則：164 頁（20.1%），471,310 個 token（10.0%）
品質分類器（Pareto，α=4）：55 頁（6.7%），108,004 個 token（2.3%）
遮蔽 PII：55 頁（6.7%），107,923 個 token（2.3%）
整條管線 16 秒
data/cc/web.bin：55 頁、108,088 個 token，2.98 bytes/token
uint16 (4, 256) (4, 256) True
```

一步一步看：

- **去重**兩步一共拿掉 9 頁（1.1%），token 少了 1.5%，後者還包含被拿掉的共用的行。
- **語言**留下 318 頁（38.9%），token 占 41.3%：頁數與 token 數的比例不一樣，因為頁面有長有短。
- **ASCII 字母**拿掉 56 頁，卻拿掉 68% 的 token（195 萬到 62 萬）：這 56 頁平均每頁約 2.4 萬個 token，留下的 262 頁平均約 2,400。為什麼要多這一步：把這一行拿掉重跑，最後是 61 頁、659,920 個 token，其中一頁以捷克文為主的頁面就有約 39.6 萬個 token，占 60%。捷克文沒有標籤（第 3 集），被判成英文，又長、形狀又合 Gopher 的規則，一路通過。`keep_ascii` 是個便宜又粗糙的補丁，只適用「要留下英文」的管線；比較正式的做法，是讓語言辨識有「都不是」這個選項（第 3 集提過：量一頁有多少比例的片段是訓練時看過的，太低就當作不認得）。
- **Gopher 規則**：262 頁剩 164 頁。
- **品質分類器**：164 頁剩 55 頁（33.5%）、token 剩 22.9%（種子 0）。
- **PII**：頁數不變，token 少了 81 個。

最後是 55 頁，占輸入的 6.7%，108,088 個 token（輸入的 2.3%）。`web.bin` 的 token 比最後一列的 107,923 多 165 個：每一頁多一個 `<|endoftext|>`，加兩個換行。2.98 bytes/token 比第 3 章標準語料的 4.12 低：tokenizer 是在兒童故事上訓練的，同樣長度的網頁文字要用多 38% 的 token。

這個規模不是拿來訓練一個像樣的模型：10.8 萬個 token，是標準語料的 0.27%。重點是管線的每一步都看得見、算得出。

### 輸出成 token 檔

`write_corpus_text` 把每一頁後面接一行 `<|endoftext|>`，寫成第 3 章 `train.txt` 的樣子；再交給第 3 章的 `tokenize_to_file`，就是 uint16 的 token 檔：

```python,ignore
{{#include ../../../src/llmz/pipeline.py:write_corpus_text}}
```

頁面文字裡剛好有 `<|endoftext|>` 這個字串的話，要先拿掉：不然 tokenizer 會把它當成真的結束符號，那一頁在中間就斷了。用假網頁，加上沒有任何 merge 的 byte 級 tokenizer，確認第 5 章的資料載入讀得進去，也真的能訓練（CPU 幾秒）：

```python
import math
import os
import tempfile

import torch

from llmz.data import get_batch, load_tokens, tokenize_to_file
from llmz.model import ModelConfig
from llmz.pipeline import Page, write_corpus_text
from llmz.tokenizer import Tokenizer, base_vocab
from llmz.train import TrainConfig, train

EOT = "<|endoftext|>"
tokenizer = Tokenizer(base_vocab([EOT]), [], [EOT])  # 256 個 byte，加上 <|endoftext|> 是第 256 號
pages = [Page(str(i), f"page number {i % 3} says hello to the garden.\nIt even mentions {EOT} by accident.") for i in range(30)]

with tempfile.TemporaryDirectory() as folder:
    text, tokens = os.path.join(folder, "web.txt"), os.path.join(folder, "web.bin")
    print("頁數：", write_corpus_text(pages, text))
    print("token 數：", tokenize_to_file(tokenizer, text, tokens))
    data = load_tokens(tokens)
    print("dtype：", data.dtype, "；結束符號：", int((data == 256).sum()), "個")  # 頁面裡的那個被拿掉了
    x, y = get_batch(data, 4, 16, torch.device("cpu"))
    print("batch：", tuple(x.shape), tuple(y.shape))

    cfg = TrainConfig(
        train_path=tokens, valid_path=tokens, out_dir=os.path.join(folder, "run"),
        model=ModelConfig(vocab_size=257, context_length=16, d_model=32, n_layers=1, n_heads=4, d_ff=64),
        batch_size=8, max_steps=30, warmup_steps=3, lr_max=3e-3, log_every=30, eval_every=30, eval_batches=2,
        checkpoint_every=30, device="cpu",
    )
    records = train(cfg)
print("驗證 loss 比亂猜好得多：", records[-1]["val_loss"] < math.log(257) - 1)
```

執行結果（某一次）：

```text
頁數： 30
token 數： 2190
dtype： uint16 ；結束符號： 30 個
batch： (4, 16) (4, 16)
{'step': 30, 'loss': 3.8083, 'lr': 0.0001, 'grad_norm': 1.11, 'time': 0.2, 'val_loss': 3.6743}
驗證 loss 比亂猜好得多： True
```

換成真的 `data/cc/web.bin`：最後 10% 當驗證，標準模型的設定與 batch，練 50 步，順便量標準模型（只看過故事）在網頁上的 loss：

```python,norun
import torch

from llmz.data import load_tokens
from llmz.device import get_device
from llmz.model import ModelConfig, TransformerLM
from llmz.train import TrainConfig, evaluate, train

data = load_tokens("data/cc/web.bin")
cut = int(len(data) * 0.9)  # 最後 10% 當驗證
data[:cut].tofile("data/cc/web_train.bin")
data[cut:].tofile("data/cc/web_valid.bin")
cfg = TrainConfig(
    train_path="data/cc/web_train.bin", valid_path="data/cc/web_valid.bin", out_dir="runs/web_smoke",
    max_steps=50, warmup_steps=5, log_every=10, eval_every=25, eval_batches=10, checkpoint_every=50,
)  # 標準模型、標準的 batch，只練 50 步

torch.manual_seed(0)
device = get_device()
base = TransformerLM(ModelConfig())
base.load_state_dict(torch.load("runs/base/model.pt", weights_only=True))
base_loss = evaluate(base.to(device), load_tokens(cfg.valid_path), cfg, device)
print(f"標準模型（只看過故事）在 web_valid.bin 上的 loss：{base_loss:.2f}")
train(cfg)
```

M4 Max 上的輸出（loss 每次相同；時間是第一次的，第二次 19.8 秒）：

```text
標準模型（只看過故事）在 web_valid.bin 上的 loss：7.39
{'step': 10, 'loss': 6.897, 'lr': 0.0009728616793536588, 'grad_norm': 0.655, 'time': 4.9}
{'step': 20, 'loss': 6.2985, 'lr': 0.0007750000000000001, 'grad_norm': 0.73, 'time': 9.4}
{'step': 30, 'loss': 5.4602, 'lr': 0.00047185832004988137, 'grad_norm': 0.474, 'time': 13.7}
{'step': 40, 'loss': 5.4449, 'lr': 0.00020528000059645996, 'grad_norm': 0.391, 'time': 17.7}
{'step': 50, 'loss': 5.0565, 'lr': 0.0001, 'grad_norm': 0.414, 'time': 21.7, 'val_loss': 5.5402}
```

只看過故事的標準模型，在網頁上的 loss 是 7.39（亂猜是 \\( \ln 10000 = 9.21 \\)），新模型只在 10 萬個 token 的網頁上練 50 步（約 22 秒，看了這個檔案約 4 遍），驗證 loss 就到 5.54。資料的領域決定模型會什麼，這是下一集的起點。

### 看一下輸出

管線的每一步都回報頁數與 token 數，但那些數字不會告訴你輸出裡有什麼。上面抓到捷克文那一頁，是因為去看了最大的一頁。最後再做幾個便宜的檢查：

```python,norun
import numpy as np

from llmz.data import load_tokens
from llmz.evaluation import ngram_hashes
from llmz.tokenizer import Tokenizer

tokenizer = Tokenizer.load("data/tokenizer.json")
eot = tokenizer.special_to_id["<|endoftext|>"]
data = np.asarray(load_tokens("data/cc/web.bin"))

ends = np.flatnonzero(data == eot)  # 每一頁到結束符號為止
sizes = np.diff(ends, prepend=-1) - 1  # 每一頁的 token 數（不含結束符號）
print(f"{len(sizes)} 頁，{len(data):,} 個 token；每頁 token 數的中位數 {np.median(sizes):.0f}，最大 {sizes.max():,}（占 {sizes.max() / sizes.sum():.1%}）")

grams = ngram_hashes(data)  # 第 12 章第 5 集：每連續 13 個 token 一個雜湊值
print(f"13-gram 共 {len(grams):,} 個，不重複的占 {len(np.unique(grams)) / len(grams):.1%}")

stories = np.asarray(load_tokens("data/train.bin")[: len(data)])  # 標準語料開頭同樣長的一段，對照
grams = ngram_hashes(stories)
print(f"對照：標準語料開頭 {len(stories):,} 個 token，13-gram 不重複的占 {len(np.unique(grams)) / len(grams):.1%}")
```

M4 Max 上的輸出（每次相同）：

```text
55 頁，108,088 個 token；每頁 token 數的中位數 1189，最大 8,974（占 8.3%）
13-gram 共 108,076 個，不重複的占 94.0%
對照：標準語料開頭 108,088 個 token，13-gram 不重複的占 98.9%
```

最大的一頁占 8.3%，不再有一頁獨大。13-gram 有 6% 是重複的，標準語料同樣長度的一段是 1.1%。第 8、9 集沒有處理這些：`dedup_lines` 只拿掉出現在至少 5 頁的行，同一頁自己重複的內容與只出現在 2 到 4 頁的行都還在，`drop_near_duplicates` 看的是整份文件。

### 這條管線做不到的事

- **只有 817 頁**：這是一個 WARC 檔的前 32 MiB。去重的計數表、品質分類器的 `score`（一次把所有頁面補齊成同樣長）都放在記憶體裡，網路規模要分段、分機器做。
- **沒有有害內容那一步**：第 5 集說過，要做就要先決定誰來標註、門檻訂在哪裡。
- **語言辨識只認 10 種語言，`keep_ascii` 是補丁**：捷克文與越南文這種沒有標籤的拉丁字母語言，要靠更好的語言辨識才解決得了。
- **tokenizer 是給故事用的**：2.98 對 4.12 bytes/token，同樣長度的網頁文字，token 數多 38%。
- **品質分數代表「像兒童故事」**：不是客觀的品質（第 7 集）。換一組正例，同一批網頁的排名完全不同。
- **只有品質那一步是隨機的**：同樣的種子，每次結果一樣；不同的種子，token 數可以差一倍。

## 重點整理

- 每一步都是「一批頁面進、一批頁面出」；`run_pipeline` 依序執行，回報每一步之後的頁數與大小（字元或 token）。
- 順序：只判斷的步驟（語言、Gopher）互換不影響結果；會改文字的 `dedup_lines` 要放在判斷文字的 Gopher 規則前面；去重的順序在整條管線裡差很小，單獨量一步會誇大差別。
- 品質分類器的機率擠在 0 附近，GPT-3 的 Pareto 規則改用分數的名次：\\( \alpha \\) 決定留下多少（約 \\( (1 - 2^{1-\alpha})/(\alpha-1) \\)），低分的頁面仍有機會。
- 817 頁、4,733,998 個 token 經過七步，剩 55 頁、108,088 個 token；輸出用 `write_corpus_text` 加第 3 章的 `tokenize_to_file`，第 5 章的 `load_tokens`、`get_batch`、`train` 讀得進去。
- 看輸出：語言辨識沒有「都不是」，一頁捷克文就占了 60% 的 token，是看最大的一頁才抓到的。
