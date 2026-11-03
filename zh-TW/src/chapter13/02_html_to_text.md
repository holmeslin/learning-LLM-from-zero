# HTML 轉文字

## 本集目標

把網頁的 bytes 變成乾淨的純文字：先用對的編碼解成字串，再用標準函式庫的 `html.parser` 去掉 script、style 與導覽，每個區塊留成一行，最後存成後面幾集共用的檔案。

## 正文

### 先把 bytes 解成字串

上一集拆出來的 `payload` 是 bytes。bytes 要用「編碼」才能變回文字，而網頁用的編碼五花八門。同一串 bytes，用錯編碼就是亂碼：

```python
from llmz.pipeline import decode_html

data = "你好，世界".encode("gbk")  # 老的中文網站常用 GBK
print(data)
print(data.decode("utf-8", errors="replace"))
print(data.decode("gbk"))
print(decode_html(data, "text/html; charset=gb2312"))  # 從 HTTP 標頭知道編碼
print(decode_html(b'<meta charset="gbk">' + data))  # 標頭沒寫，從網頁自己寫的 <meta> 知道
```

執行結果：

```text
b'\xc4\xe3\xba\xc3\xa3\xac\xca\xc0\xbd\xe7'
��ã�����
你好，世界
你好，世界
<meta charset="gbk">你好，世界
```

`errors="replace"` 把解不開的 byte 換成 U+FFFD（�，Unicode 專門給這種情況的替代字元），程式不會出錯，但原本的字已經丟了。所以要先知道網頁用的是哪一種編碼。`decode_html` 的做法照瀏覽器的順序：

```python,ignore
{{#include ../../../src/llmz/pipeline.py:decode_html}}
```

1. HTTP 標頭 `Content-Type: text/html; charset=...` 寫的編碼。
2. 網頁最前面 2048 個 byte 裡 `<meta charset="...">`（或老式的 `<meta http-equiv="Content-Type" content="...; charset=...">`）寫的編碼。用 `rb"..."` 的位元組正規表示式去找，因為還沒解碼，只能在 bytes 上找；編碼名稱都是 ASCII，不受影響。
3. 都沒寫，或寫的名稱 Python 不認得（`LookupError`）、或用它解不開（`UnicodeDecodeError`），就換下一個；最後當 UTF-8，壞掉的 byte 變成 U+FFFD。

中間那張 `CHARSET_ALIASES` 表是從真的網頁學來的：很多網頁寫 `gb2312`、`shift_jis`、`iso-8859-1`，內容其實用了它們的擴充版（`gbk`、`cp932`、`cp1252`），瀏覽器也是這樣處理。Python 的 `gb2312` 解碼器只認得 GB 2312 裡的字，碰到擴充的字就丟錯誤，所以我們直接換成擴充版的解碼器。`for ... else` 是第 11 章用過的寫法：`for` 迴圈沒有被 `break` 打斷，才執行 `else`。

對真的檔案量一量。先看網頁從哪裡知道編碼，再看解完之後有多少頁還留著 U+FFFD，以及這張別名表救回了幾頁：

```python,norun
from collections import Counter

import llmz.pipeline as pipeline
from llmz.pipeline import CHARSET_RE, decode_html, iter_warc_records, split_http_response

pages = []  # (HTTP 標頭的 Content-Type, 網頁的 bytes)
for headers, body in iter_warc_records("data/cc/sample.warc.gz"):
    if headers["warc-type"] == "response":
        status, http_headers, payload = split_http_response(body)
        if status == 200 and "html" in http_headers.get("content-type", "").lower():
            pages.append((http_headers["content-type"], payload))

where = Counter()
for content_type, payload in pages:
    if CHARSET_RE.search(content_type.encode()):
        where["HTTP 標頭有寫"] += 1
    elif CHARSET_RE.search(payload[:2048]):
        where["只有 meta 有寫"] += 1
    else:
        where["都沒寫"] += 1
print(len(pages), "頁 HTML：", dict(where))

for name in ["有別名表", "沒有別名表"]:
    texts = [decode_html(payload, content_type) for content_type, payload in pages]
    broken = [t.count("\ufffd") for t in texts if "\ufffd" in t]
    print(f"{name}：{len(broken)} 頁含 U+FFFD，最多的一頁有 {max(broken)} 個")
    pipeline.CHARSET_ALIASES.clear()  # 第二輪不用別名表
```

M4 Max 上的輸出（檔案固定，每次相同）：

```text
842 頁 HTML： {'只有 meta 有寫': 168, 'HTTP 標頭有寫': 653, '都沒寫': 21}
有別名表：11 頁含 U+FFFD，最多的一頁有 1025 個
沒有別名表：14 頁含 U+FFFD，最多的一頁有 21843 個
```

842 頁裡有 168 頁的編碼只寫在 `<meta>` 裡，21 頁兩邊都沒寫，所以光看 HTTP 標頭不夠。別名表讓含有 U+FFFD 的頁面從 14 頁降到 11 頁，最嚴重的一頁從 21,843 個降到 1,025 個。剩下的有些是宣告 UTF-8、但內容裡夾了幾個壞 byte，也有標頭與 `<meta>` 寫的編碼互相矛盾、兩個都解不開的頁面，這種我們只能接受。

### `html.parser`：邊讀邊通知

字串有了，要找出裡面的文字。HTML 的標籤可以巢狀、可以漏寫結尾，不要用字串取代或正規表示式去拆。標準函式庫的 `html.parser.HTMLParser` 負責把 HTML 從頭讀到尾，遇到什麼就呼叫對應的方法：遇到開始標籤呼叫 `handle_starttag`，遇到結束標籤呼叫 `handle_endtag`，遇到標籤之間的文字呼叫 `handle_data`。我們繼承它、改寫這三個方法就行。先讓它把看到的東西印出來。`class Show(HTMLParser)` 表示 `Show` 繼承 `HTMLParser`：它有的功能 `Show` 都有，我們只改寫其中三個方法。

```python
from html.parser import HTMLParser


class Show(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("開始", tag)

    def handle_endtag(self, tag):
        print("結束", tag)

    def handle_data(self, data):
        print("文字", repr(data))


parser = Show()
parser.feed("<p>Fish &amp; chips <b>now</b></p><script>x = 1 < 2</script>")  # 餵 HTML 進去
parser.close()  # 告訴它沒有更多了
```

執行結果：

```text
開始 p
文字 'Fish & chips '
開始 b
文字 'now'
結束 b
結束 p
開始 script
文字 'x = 1 < 2'
結束 script
```

兩件事值得注意。`&amp;` 在 `handle_data` 收到之前就已經換成 `&` 了（`HTMLParser` 預設會把 `&amp;`、`&nbsp;` 這類實體換成字元）。`<script>` 裡面的 `x = 1 < 2` 也是以「文字」的身份送來：剖析器不會幫你分辨哪些文字是給人看的，這要我們自己決定。

### 三個決定

`html_to_text` 靠三個決定把網頁變成文字：

```python,ignore
{{#include ../../../src/llmz/pipeline.py:html_to_text}}
```

1. **哪些標籤整個丟掉**（`SKIP_TAGS`）：`script`、`style` 是程式與樣式，不是文字；`title`、`select`（下拉選單）、`nav`（導覽）、`footer`（頁尾）、`aside`（側欄）是每頁都有的框架。`skip` 是計數器，進入這些標籤 +1、離開 -1，大於 0 的時候收到的文字都不要，所以 `nav` 裡面再包 `ul`、`a` 也一起被丟掉。
2. **哪些標籤換行**（`BLOCK_TAGS`）：段落、標題、清單項目、表格的格子、`<br>` 這些在瀏覽器上自成一塊的標籤，開始與結束的地方各插一個換行；`<b>`、`<a>` 這種行內標籤不換行，所以 `<b>red</b>ball` 還是同一個字。
3. **空白怎麼處理**：HTML 的原始碼裡，空格、換行、縮排都只算「一個空白」。最後逐行用 `" ".join(line.split())` 把連續的空白（包含 `&nbsp;` 變來的 `\xa0`）合成一個空格，再丟掉空行。

用一頁原創的小網頁試試，第二個是什麼都不丟（`skip_tags=set()`）的結果：

```python
from llmz.pipeline import html_to_text

page = """<html>
<head><title>Mia's Garden - Home</title><style>p { color: red; }</style></head>
<body>
<nav><a href="/">Home</a> | <a href="/about">About</a></nav>
<h1>The red ball</h1>
<p>Mia had a <b>red</b> ball.<br>She played   with it
every day.</p>
<script>window.track("page_view");</script>
<ul><li>park</li><li>pond</li></ul>
<footer>Copyright 2026 &mdash; all rights reserved</footer>
</body></html>"""

print(html_to_text(page))
print("-----")
print(html_to_text(page, skip_tags=set()))
```

執行結果：

```text
The red ball
Mia had a red ball.
She played with it
every day.
park
pond
-----
Mia's Garden - Homep { color: red; }
Home | About
The red ball
Mia had a red ball.
She played with it
every day.
window.track("page_view");
park
pond
Copyright 2026 — all rights reserved
```

一行一個區塊的格式是刻意的：本章第 8 集要「逐行」找出重複的頁首頁尾，第 6 集的品質規則也要看「有幾行」。

### 在真的網頁上

三種丟法，在真的 842 頁 HTML 上各留下多少文字？

```python,norun
import statistics
import time

from llmz.pipeline import SKIP_TAGS, decode_html, html_to_text, iter_warc_records, split_http_response

htmls = []
for headers, body in iter_warc_records("data/cc/sample.warc.gz"):
    if headers["warc-type"] == "response":
        status, http_headers, payload = split_http_response(body)
        if status == 200 and "html" in http_headers.get("content-type", "").lower():
            htmls.append(decode_html(payload, http_headers["content-type"]))

CODE_TAGS = {"script", "style", "noscript", "template", "svg", "iframe", "title"}
for name, skip in [("什麼都不丟", set()), ("只丟 script、style 之類", CODE_TAGS), ("再丟 select、nav、footer、aside", SKIP_TAGS)]:
    print(f"{name}：{sum(len(html_to_text(h, skip)) for h in htmls):,} 個字元")

start = time.perf_counter()
texts = [html_to_text(h) for h in htmls]
seconds = time.perf_counter() - start
lengths = [len(t) for t in texts]
empty = [h for h, t in zip(htmls, texts) if not t]
print(f"{len(htmls)} 頁，轉換 {seconds:.1f} 秒，每頁文字量的中位數 {statistics.median(lengths):,.0f} 個字元")
print(f"沒有文字的 {len(empty)} 頁，其中 HTML 不到 300 個字元的 {sum(len(h) < 300 for h in empty)} 頁")
```

M4 Max 上的輸出（轉換時間兩次執行是 3.1 與 2.7 秒，這裡貼第一次；其餘每次相同）：

```text
什麼都不丟：20,368,432 個字元
只丟 script、style 之類：4,919,873 個字元
再丟 select、nav、footer、aside：4,413,284 個字元
842 頁，轉換 3.1 秒，每頁文字量的中位數 1,516 個字元
沒有文字的 25 頁，其中 HTML 不到 300 個字元的 18 頁
```

最大的一刀是 script 與 style：什麼都不丟時，四分之三的「文字」其實是程式碼與樣式表。再丟 `select`、`nav`、`footer`、`aside`，又少了約一成。沒有文字的頁面多半是空殼：HTML 本身只有幾行，或文字大概是 JavaScript 在瀏覽器裡才畫出來的，我們不跑 JavaScript，看不到。

### 存起來，後面幾集共用

接下來幾集要一再處理這些文字，所以轉一次存起來。模組裡剩下三個函式：

```python,ignore
{{#include ../../../src/llmz/pipeline.py:iter_pages}}
```

`iter_pages` 把上一集的 `iter_warc_records`、`split_http_response` 與這一集的 `decode_html`、`html_to_text` 接起來：只留 `response`、狀態碼 200、`Content-Type` 有 `html` 的記錄，產生 `Page(網址, 文字)`。`Page` 是 `typing.NamedTuple`：像 tuple 一樣能解包，又可以用 `page.url`、`page.text` 取值。

```python,ignore
{{#include ../../../src/llmz/pipeline.py:write_pages}}
```

存成 JSON Lines：一頁一行 JSON，`{"url": ..., "text": ...}`。`json.dumps` 會把文字裡的換行跳脫成 `\n`，所以一頁永遠只佔一行；`ensure_ascii=False` 讓中文維持原樣、不要變成 `你`。讀回來的時候用 `for line in f` 逐行讀，不用 `read().splitlines()`，因為後者連 U+2028 這類字元也當成換行，會把一頁切成兩半。用兩頁原創的假資料走一遍：

```python
import json

from llmz.pipeline import Page, read_pages, write_pages

pages = [Page("http://example.com/a", "第一行\n第二行"), Page("http://example.com/b", "only one line")]
print(write_pages(pages, "pages.jsonl"), "頁")
with open("pages.jsonl", encoding="utf-8") as f:
    print(f.read(), end="")
print(list(read_pages("pages.jsonl")) == pages)
```

執行結果：

```text
2 頁
{"url": "http://example.com/a", "text": "第一行\n第二行"}
{"url": "http://example.com/b", "text": "only one line"}
True
```

真的網頁：執行上一集的腳本 `tools/prepare_webtext.py`，它在下載之後接著做這件事（沒有文字的頁面不存）：

```bash
uv run python tools/prepare_webtext.py
```

M4 Max 上的輸出（第一次執行，時間看你的網路；之後再執行，檔案都已經存在，會直接跳過）：

```text
data/cc/sample.warc.gz: 33,554,432 bytes，7 秒
data/cc/pages.jsonl: 842 個 HTML 頁面，留下 817 頁有文字的，6,345,494 bytes，3 秒
```

結果在 `data/cc/pages.jsonl`（不進 repo）。本章後面的各集（語言辨識、個資、品質、去重）都從這個檔案讀：`read_pages("data/cc/pages.jsonl")`。

### 這個版本做不到的事

- **JavaScript 才畫出來的內容**：看不到，上面那些空殼頁面就是。
- **沒關起來的標籤**：`<nav>` 之類忘了寫結尾，`skip` 一直大於 0，後面的文字全被吃掉。真的網頁裡很少見（這個檔案的 842 頁沒有因此變成空頁），遇到再換成會修復 HTML 的剖析器。`<pre>` 的縮排也會被壓掉。
- **沒有用 `nav` 標籤的選單與頁尾**：很多網站的選單是一堆 `div`，這個版本分不出來，它們會留在文字裡。本章第 8 集會用「逐行找重複」把這種每頁都有的行去掉。
- **內容的好壞**：留下來的文字什麼都有，廣告、垃圾、不當的內容也在裡面。網頁開頭那幾行常是選單，後面才是正文；真的網路上不是每一頁都值得訓練，本章後面會一步一步處理。

實務上會用專門的「主要內容抽取」工具（例如 resiliparse、trafilatura），它們比這個版本聰明很多，但依賴多、細節多。本書只用標準函式庫，是為了讓每一個決定都看得見。

## 重點整理

- bytes 要先用對的編碼解成字串：先信 HTTP 標頭的 `charset`，再信網頁前 2048 byte 的 `<meta>`，都沒有就當 UTF-8（壞的 byte 變成 U+FFFD）；網頁寫的舊名稱要換成擴充版（`gb2312` → `gbk` 等）。
- `html.parser.HTMLParser` 邊讀 HTML 邊呼叫 `handle_starttag`、`handle_endtag`、`handle_data`；它不分辨文字給誰看，要自己決定。
- `html_to_text` 有三個決定：`SKIP_TAGS` 整個丟掉（script、style、nav、footer 等，用計數器處理巢狀）、`BLOCK_TAGS` 的邊界換行、空白合成一個空格。結果是「一行一個區塊」。
- 在這個檔案的 842 頁 HTML 上，什麼都不丟時四分之三是程式碼與樣式；`iter_pages`、`write_pages`、`read_pages` 把轉好的 `Page(url, text)` 存成 `data/cc/pages.jsonl`，一頁一行 JSON，後面幾集共用。
