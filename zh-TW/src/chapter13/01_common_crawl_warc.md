# Common Crawl 與 WARC

## 本集目標

知道 Common Crawl 是什麼、一個 WARC 檔裡有哪幾種記錄，用 HTTP Range 只下載一個 WARC 檔的前 32 MiB，再用標準函式庫自己把記錄一筆一筆讀出來。

## 正文

### 訓練資料從網頁來

前兩部我們用的是 TinyStories：別人整理好、一行一個故事的文字檔。可是大型語言模型的訓練資料，大宗是網頁。自己去爬整個網路要花很多力氣，所以多數人從 [Common Crawl](https://commoncrawl.org) 開始：一個非營利組織，定期爬一次網路，把原始資料放在公開的網站上讓人下載。

每一次爬取叫一個 crawl，名字像 `CC-MAIN-2026-43`（名字裡有年份，這一個是 2026 年 10 月爬的，寫這一章時最新）。每個 crawl 底下有幾個清單檔，列出它所有的檔案路徑；`CC-MAIN-2026-43` 的 `warc.paths.gz` 列了 100,000 個 WARC 檔，每個大約 0.9 GB。同一批網頁有三種格式：

| 格式 | 內容 | 這個 crawl 的第 0 個檔 |
| --- | --- | --- |
| WARC | 原始資料：HTTP 請求、HTTP 回應（含完整的 HTML）、一些 metadata | 923,804,210 bytes |
| WAT | 從 WARC 抽出的 metadata（標頭、連結等） | 155,793,054 bytes |
| WET | 已經抽好的純文字 | 64,395,252 bytes |

WET 看起來最省事，但「怎麼把 HTML 變成文字」是別人幫你決定的：哪些內容算導覽、段落怎麼斷，你看不到也改不了。本書從 WARC 開始，下一集自己做這個轉換，知道每個決定換掉了什麼。

### 只下載前 32 MiB

整個 WARC 檔將近 1 GB，我們只要一小段。做法和第 3 章下載 TinyStories 一樣：HTTP 的 Range 標頭，`llmz.data.download` 直接可用。本書固定用這一個檔案：

- crawl：`CC-MAIN-2026-43`
- 檔案：`crawl-data/CC-MAIN-2026-43/segments/1791170053823.21/warc/CC-MAIN-20261005105623-20261005135623-00000.warc.gz`，也就是這個 crawl 的 `warc.paths.gz` 裡的第一個
- 下載：前 32 MiB（33,554,432 bytes），約佔整個檔案的 3.6%
- 驗證：下載下來的檔案 SHA-256 是 `ab4b69e56c59215216881cbdd5834360a0345cb5fab0dd6bd59304cf3f284575`（Common Crawl 的檔案寫好之後不會再改，所以每個人的前 32 MiB 都一樣）

```python,norun
from llmz.data import download

url = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-43/segments/1791170053823.21/warc/CC-MAIN-20261005105623-20261005135623-00000.warc.gz"
n = download(url, "data/cc/sample.warc.gz", 32 * 1024 * 1024)
print(f"{n:,} bytes")
```

M4 Max 上的輸出（某一次，時間看你的網路）：

```text
33,554,432 bytes
```

伺服器有照規矩回應，可以用 `curl` 看：要前 100 個 byte，回的是狀態碼 206（Partial Content），`content-range` 的斜線後面是整個檔案的大小。

```bash
curl -s -D - -o /dev/null -r 0-99 https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-43/segments/1791170053823.21/warc/CC-MAIN-20261005105623-20261005135623-00000.warc.gz
```

```text
HTTP/2 206
content-length: 100
content-range: bytes 0-99/923804210
```

（只列出相關的三行。）下載、以及下一集的轉換，都包在一支腳本裡，檔案已經存在就跳過：

```bash
uv run python tools/prepare_webtext.py
```

Common Crawl 是公共資源，練習時請不要反覆下載同一個檔案。下載的檔案放在 `data/cc/`，不進 repo，書裡也只放統計數字，不貼網頁的內容：網頁的著作權屬於各個網站。

### 一個 `.warc.gz` 是很多段 gzip 接在一起

WARC（Web ARChive）是存網頁用的檔案格式，有 ISO 標準（ISO 28500）。檔案由很多筆**記錄**一個接一個組成，每一筆長這樣：

```text
WARC/1.0                         <- 版本
WARC-Type: response              <- 標頭：每行「名稱: 值」，以 \r\n 結尾
WARC-Target-URI: http://...
Content-Length: 15957
（其他標頭）
                                 <- 空行：標頭結束
（Content-Length 個 byte 的內容）
\r\n\r\n                         <- 記錄結尾
```

Common Crawl 的檔案裡有四種記錄（`WARC-Type`）：

- `warcinfo`：整個檔案只有一筆，在最前面，描述這次 crawl（用什麼軟體爬的）。
- `request`：爬蟲送出的 HTTP 請求。
- `response`：伺服器回的 HTTP 回應，標頭加上網頁本體，我們要的 HTML 在這裡。
- `metadata`：爬蟲額外記下的資訊，例如抓取花了幾毫秒。

每抓一個網址，依序出現 `request`、`response`、`metadata` 三筆。

重點是壓縮的方式：**每一筆記錄各自壓成一段 gzip，整個檔案就是這些段直接接在一起**。這有兩個好處。第一，檔案從中間切斷，前面完整的段仍然解得開，所以只下載前 32 MiB 就能用。第二，只要知道某一筆記錄在檔案裡的起點與長度（Common Crawl 另外提供索引），就能用 Range 只抓那一筆。

我們自己造一個三筆記錄的小 WARC 來看。內容全是原創的假資料：

```python
import gzip

from llmz.pipeline import iter_warc_records, split_http_response


def make_record(warc_type: str, body: bytes, uri: str = "") -> bytes:
    head = f"WARC/1.0\r\nWARC-Type: {warc_type}\r\n"
    if uri:
        head += f"WARC-Target-URI: {uri}\r\n"
    head += f"Content-Length: {len(body)}\r\n\r\n"
    return gzip.compress(head.encode() + body + b"\r\n\r\n")  # 每筆記錄自己壓成一段


members = [
    make_record("warcinfo", b"software: toy crawler\r\n"),
    make_record("request", b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n", "http://example.com/"),
    make_record("response", b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<p>Hello</p>", "http://example.com/"),
]
with open("toy.warc.gz", "wb") as f:
    f.write(b"".join(members))  # 直接接在一起，就是一個 .warc.gz

for headers, body in iter_warc_records("toy.warc.gz"):
    print(headers["warc-type"], headers.get("warc-target-uri", "-"), len(body))

status, http_headers, payload = split_http_response(body)  # 迴圈最後一筆：response
print(status, http_headers["content-type"], payload)

# 第三段只下載了開頭 20 個 byte，像是 Range 下載在檔案中間被切斷
with open("cut.warc.gz", "wb") as f:
    f.write(members[0] + members[1] + members[2][:20])
print("切斷後讀到", len(list(iter_warc_records("cut.warc.gz"))), "筆")
```

執行結果：

```text
warcinfo - 23
request http://example.com/ 37
response http://example.com/ 56
200 text/html b'<p>Hello</p>'
切斷後讀到 2 筆
```

### 讀出每一筆記錄

上面呼叫的 `iter_warc_records` 是這一章資料管線模組 `src/llmz/pipeline.py` 的第一個函式：

```python,ignore
{{#include ../../../src/llmz/pipeline.py:iter_warc_records}}
```

它是一個生成器（第 3 章第 11 集用過的 `yield`），一次產生一筆記錄，所以 32 MiB 的檔案不會整個攤開在記憶體裡：

- `gzip.open` 是標準函式庫的 gzip 讀取器，遇到「很多段接在一起」的檔案會自己一段一段往下讀，我們只管把它當成一個大檔案。
- 先讀版本那一行，再一行一行讀標頭，直到空行。標頭名稱一律轉成小寫，因為 WARC 與 HTTP 的標頭名稱都不分大小寫。
- 標頭裡的 `Content-Length` 說內容有幾個 byte，照這個數字讀，不要靠找空行：內容本身（網頁、PDF）裡面也有空行。
- 檔案被切斷時，最後一段 gzip 不完整，gzip 模組會丟出 `EOFError`（剛好只剩下一個 byte 時是 `BadGzipFile`，兩種都接）。我們把它當成「到此為止」，丟掉最後那筆，已經讀到的不受影響。這和第 3 章 `cut_at_last_document` 的想法一樣：切在中間的尾巴不要。

`response` 記錄的內容，本身又是一份完整的 HTTP 回應：狀態行、標頭、空行、網頁本體。`split_http_response` 把它拆開：

```python,ignore
{{#include ../../../src/llmz/pipeline.py:split_http_response}}
```

HTTP 回應的第一行是 `HTTP/1.1 200 OK`，空格後面第二個欄位是狀態碼。Common Crawl 存回來的網頁本體已經是解壓縮之後的內容：伺服器原本可能用 gzip、br 或 zstd 壓縮，`X-Crawler-Content-Encoding` 標頭記著原本的壓縮方式，但這個檔案裡 `payload` 直接就是 HTML 的 bytes，不用自己解壓縮。

### 這個檔案裡有什麼

對真的檔案數一數：每種記錄幾筆、HTTP 狀態碼、網頁的類型、網頁來自幾個不同的主機。

```python,norun
from collections import Counter
from urllib.parse import urlsplit

from llmz.pipeline import iter_warc_records, split_http_response

types, statuses, content_types, hosts = Counter(), Counter(), Counter(), Counter()
example = None
for headers, body in iter_warc_records("data/cc/sample.warc.gz"):
    types[headers["warc-type"]] += 1
    if headers["warc-type"] == "response":
        example = example or headers
        status, http_headers, payload = split_http_response(body)
        statuses[status] += 1
        content_types[http_headers.get("content-type", "").split(";")[0].strip().lower() or "（沒寫）"] += 1
        hosts[urlsplit(headers["warc-target-uri"]).hostname] += 1

print(f"{sum(types.values()):,} 筆記錄：{dict(types)}")
print("HTTP 狀態碼：", dict(statuses))
print("Content-Type：", dict(content_types.most_common()))
print(f"{len(hosts)} 個不同的主機，最多的一個有 {hosts.most_common(1)[0][1]} 筆 response")
print("response 的 WARC 標頭：", list(example))
```

M4 Max 上的輸出（檔案固定，每次相同）：

```text
2,579 筆記錄：{'warcinfo': 1, 'request': 860, 'response': 859, 'metadata': 859}
HTTP 狀態碼： {200: 859}
Content-Type： {'text/html': 842, 'application/pdf': 10, '（沒寫）': 2, 'application/xml': 1, 'text/plain': 1, 'text/xml': 1, 'application/x-tar': 1, 'application/octet-stream': 1}
817 個不同的主機，最多的一個有 3 筆 response
response 的 WARC 標頭： ['warc-type', 'warc-date', 'warc-record-id', 'content-length', 'content-type', 'warc-warcinfo-id', 'warc-concurrent-to', 'warc-ip-address', 'warc-target-uri', 'warc-protocol', 'warc-payload-digest', 'warc-block-digest', 'warc-identified-payload-type']
```

整理一下：

- 讀到 2,579 筆完整的記錄；最後一筆在第 32 MiB 處被切斷，沒讀進來。
- 859 筆 `response`，狀態碼全是 200，所以我們不必處理轉址或錯誤頁。
- `text/html` 有 842 筆（98%），其餘是 PDF、XML、純文字、壓縮檔等，下一集只轉 HTML。
- 859 筆 response 來自 817 個不同的主機，最多的一個主機也只有 3 筆：一個 WARC 檔裡的網頁來自整個網路的各個角落，不是幾個網站的連續頁面，所以前 32 MiB 就是一份不錯的小樣本。

## 重點整理

- Common Crawl 定期爬網路並公開原始資料；一個 crawl 有幾萬個 WARC 檔，另有 WAT（metadata）與 WET（純文字）。本書固定用 `CC-MAIN-2026-43` 的第 0 個 WARC 檔的前 32 MiB。
- WARC 檔由很多筆記錄組成（`warcinfo`、`request`、`response`、`metadata`），每筆記錄是自己的一段 gzip，所以 HTTP Range 切在中間時，前面完整的記錄還讀得出來。
- `iter_warc_records(path)` 逐筆產生 `(WARC 標頭, 內容)`，標頭名稱小寫，照 `Content-Length` 讀內容，遇到被切斷的尾巴就停。
- `response` 的內容是一份 HTTP 回應，`split_http_response` 拆成 `(狀態碼, 標頭, 網頁的 bytes)`；Common Crawl 已經幫我們解開 `Content-Encoding`。
- 這個檔案的前 32 MiB 有 2,579 筆記錄、859 筆 response、842 頁 HTML，來自 817 個不同的主機。
