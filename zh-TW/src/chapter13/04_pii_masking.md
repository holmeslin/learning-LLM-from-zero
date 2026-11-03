# PII 遮蔽

## 本集目標

用正規表示式找出網頁文字裡的 email、IP 與電話號碼，換成固定的標記（`<EMAIL>`、`<IP>`、`<PHONE>`），同時回傳各遮蔽了幾個。然後在 817 頁網頁上量遮蔽了多少、哪些是誤殺。

## 正文

### 為什麼要遮蔽

PII 是 personally identifiable information（個人可識別資訊）的縮寫：email、電話、地址、身分證字號、帳號……網頁上到處都是。模型看過的東西，有可能在某個提示下被原樣吐出來；一個真實人物的電話號碼，不應該出現在訓練資料裡，更不應該被模型背起來。

所以管線在訓練前，要把這些資訊處理掉。處理的方式有兩種：整頁丟掉，或只把那一小段換成標記。Dolma（Soldaini 等人 2024）的做法是兩個一起用：用正規表示式找出 email、電話與 IP，換成標記；但一頁裡有 6 個以上的 PII，就整頁丟掉。本集做第一種：**遮蔽**（mask）。換成標記而不是刪掉，句子的結構還在，模型看到的是「請寄信到 `<EMAIL>`」，學得到「這裡會有一個 email」，但學不到是哪一個。

### 三個正規表示式

正規表示式（regular expression）第 3 章第 4 集已經用過，這裡再多用一個功能：**look-around**（前後斷言）。`(?<!X)` 的意思是「前面不可以是 X」，`(?!X)` 是「後面不可以是 X」，它們只檢查旁邊的字元，不會把它們算進比對的結果裡。我們用它們避免「在一長串數字中間切一段」。

```python,ignore
{{#include ../../../src/llmz/pipeline.py:mask_pii}}
```

- **email**：`@` 前面是英數字與 `._%+-`，後面是用點隔開的網域，最後一段是至少兩個英文字母。句尾的句號不會被吃進去，因為最後一段只收字母。
- **IPv4**：四段 0 到 255 的數字用點隔開。`_OCTET` 一次只比對一段（`25[0-5]`、`2[0-4]\d`、`1\d\d`、或一到兩位數）；`256.1.1.1` 因此不算。前面不能是字母數字或點，後面不能是字母數字、也不能是「點加數字」：`1.2.3.4.5` 不算，句尾的 `1.2.3.4.` 算。
- **電話**最麻煩，因為各國格式五花八門。我們的規則是三件事一起：長得像（`+` 開頭的一串數字，或幾組數字用空格、點、減號隔開）、數字加起來有 9 到 15 個（`mask_pii` 裡檢查；E.164 規定國際電話號碼最長 15 位），而且不像別的東西：開頭像日期（`2026-11-03`、`03.11.2026`）、緊貼著別的數字組（一長串數字的表格）、後面接 `:數字`（時間）或 `.數字`（小數）的都不算。
- `re.ASCII` 讓 `\w` 與 `\d` 只算 ASCII，這樣「電話0912-000-111」貼在中文旁邊也認得出來；沒有它，中文字也算 `\w`，前面的 `(?<![\w.+-])` 會擋掉它。
- `pattern.sub(函式, text)` 對每個比對呼叫函式，用回傳值取代；函式裡順便數數，所以 `mask_pii` 一趟就得到遮蔽後的文字與次數。處理順序是 email、IP、電話：`198.51.100.77` 長得像電話，要先被 IP 規則拿走。

試試看。範例都是保留給文件用的假資料（`example.com`、RFC 5737 保留的 IP、555 開頭的號碼），後半是不該被遮蔽的反例：

```python
from llmz.pipeline import mask_pii

text = """Contact Mia.Lee+news@example.com or call +1 (555) 010-1234.
台灣：(02) 2345-6789、0912-000-111；伺服器 192.0.2.1，備援 198.51.100.77。"""
masked, counts = mask_pii(text)
print(masked)
print(counts)

print("--- 不該被遮蔽的 ---")
for text in [
    "2026-11-03 12:30:45",
    "version 1.2.3, build 10.0.0.1.5",
    "Price: 1,234,567 dollars",
    "3.14159265358979",
    "10 11 12 13 14 15 16 17",
    "user@localhost, @name",
]:
    print(mask_pii(text)[0] == text, text)
```

執行結果：

```text
Contact <EMAIL> or call <PHONE>.
台灣：<PHONE>、<PHONE>；伺服器 <IP>，備援 <IP>。
{'email': 1, 'phone': 3, 'ip': 2}
--- 不該被遮蔽的 ---
True 2026-11-03 12:30:45
True version 1.2.3, build 10.0.0.1.5
True Price: 1,234,567 dollars
True 3.14159265358979
True 10 11 12 13 14 15 16 17
True user@localhost, @name
```

### 在真的網頁上

在 817 頁網頁上全部遮蔽一次，量每一類遮蔽了多少：

```python,norun
import ipaddress
import time
from collections import Counter

from llmz.pipeline import IPV4_RE, mask_pii, read_pages

pages = list(read_pages("data/cc/pages.jsonl"))

start = time.perf_counter()
results = [mask_pii(p.text) for p in pages]
seconds = time.perf_counter() - start

total, with_pii = Counter(), Counter()
for _, counts in results:
    for kind, n in counts.items():
        total[kind] += n
        with_pii[kind] += n > 0
for kind in ["email", "phone", "ip"]:
    print(f"{kind:5s} 遮蔽 {total[kind]:4d} 次，分布在 {with_pii[kind]:3d} 頁")
print(f"至少有一個的頁面：{sum(sum(c.values()) > 0 for _, c in results)}/{len(pages)}，耗時 {seconds:.1f} 秒")

per_page = sorted((sum(c.values()) for _, c in results), reverse=True)
print(f"最多的一頁遮蔽了 {per_page[0]} 個；遮蔽數最多的 10 頁合計 {sum(per_page[:10])} 個，占全部的 {sum(per_page[:10]) / sum(per_page):.0%}")

ips = [m.group() for p in pages for m in IPV4_RE.finditer(p.text)]
print(f"IP {len(ips)} 個，其中 {sum(ipaddress.ip_address(x).is_private for x in ips)} 個是私有位址（127.0.0.1、192.168.x.x 之類）")
```

M4 Max 上的輸出（耗時兩次執行是 0.4 與 0.8 秒，這裡貼第一次；其餘每次相同）：

```text
email 遮蔽  327 次，分布在 124 頁
phone 遮蔽  256 次，分布在 123 頁
ip    遮蔽    4 次，分布在   3 頁
至少有一個的頁面：190/817，耗時 0.4 秒
最多的一頁遮蔽了 132 個；遮蔽數最多的 10 頁合計 238 個，占全部的 41%
IP 4 個，其中 4 個是私有位址（127.0.0.1、192.168.x.x 之類）
```

817 頁裡有不到四分之一（190 頁）至少有一個，而且很集中：遮蔽數最多的 10 頁就占了 41%，多半是店家目錄、聯絡窗口的列表。這就是 Dolma 另外加「一頁 6 個以上就整頁丟掉」的原因：這樣的頁面本來就是一張通訊錄，不是文章。email 遮蔽完之後，用同樣的規則再找一次，一個都找不到。

IP 只有 4 個，而且全是私有位址：網頁上的 IP 多半是教學文章裡的 `127.0.0.1` 與 `192.168.x.x`，本來就不指向任何人。遮蔽它們不傷害資料，但也說明這個規則在乎的是「長得像」，不是「真的是個資」。

### 誤殺有多少

email 與 IP 的格式很固定，規則很準。電話不是。隨機抽 50 個被遮蔽的電話（`random.Random(0).sample`，固定種子），把前後文印出來（數字換成 9，避免把真人的號碼印出來），人工逐個看過：

```python,norun
import random
import re

from llmz.pipeline import PHONE_RE, read_pages

found = []
for page in read_pages("data/cc/pages.jsonl"):
    for m in PHONE_RE.finditer(page.text):
        if 9 <= sum(c.isdigit() for c in m.group()) <= 15:  # 和 mask_pii 的條件一樣
            found.append(re.sub(r"\d", "9", page.text[max(0, m.start() - 30) : m.end() + 12]).replace("\n", "⏎"))
print(len(found))
for i, s in enumerate(random.Random(0).sample(found, 50)):
    print(i, s)
```

遮蔽了 256 個電話，抽到的 50 個裡有 45 個是電話（聯絡方式、傳真、店家目錄），5 個不是：一個技術標準編號、一個帳號、一個印尼盾金額、一個澳洲公司編號（ABN，格式是 `99 999 999 999`）、一個看不出是什麼的數字。不是電話的比例大約一成；這是人工判斷、50 個樣本，只能當粗略的估計。（輸出裡有真實網頁的文字，所以書上不貼。）

### 這個版本做不到的事

- **姓名、地址、帳號**：這些沒有固定格式，正規表示式辦不到。實務上會用專門的 PII 偵測工具（也是模型）。
- **沒有分隔的電話**：`0912000111` 這種十碼連在一起的，我們不遮蔽，因為和任何十位數字（訂單編號、時間戳）分不開。`+` 開頭的才算。
- **IPv6**：沒做。完整的八組寫法很少見，簡寫（`::1`）又很容易和別的東西搞混。
- **誤殺**：電話約一成，上面看到了。email 規則也可能把 `logo@2x.png` 這種檔名當成 email；這個檔案的 817 頁沒有出現，但規則上會。
- **遮蔽不等於匿名**：留下的網址、使用者名稱、留言內容，湊在一起還是能認出一個人。PII 遮蔽是減少風險，不是消除。

## 重點整理

- PII 是個人可識別資訊；訓練前要處理，因為模型可能把看過的 email 與電話原樣吐出來。Dolma 的做法：用正規表示式遮蔽 email、電話、IP，一頁有 6 個以上就整頁丟掉。
- `mask_pii` 用三個正規表示式，依序遮蔽 email、IPv4、電話，回傳遮蔽後的文字與各類次數；用 look-around（`(?<!...)`、`(?!...)`）避免在一長串數字中間切一段，電話另外要求 9 到 15 個數字，並排除日期、時間與小數。
- 在 817 頁網頁上（M4 Max）：遮蔽 327 個 email、256 個電話、4 個 IP，共 190 頁至少有一個，最多的 10 頁占 41%；4 個 IP 全是私有位址；隨機抽 50 個電話，人工判斷 45 個是電話（誤殺約一成）。
- 這個版本不處理姓名與地址、沒有分隔的電話與 IPv6；遮蔽不等於匿名。
