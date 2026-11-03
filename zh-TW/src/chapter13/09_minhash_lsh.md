# MinHash + LSH

## 本集目標

找出「幾乎一樣、但不是一字不差」的文件：把每份文件變成 shingle 集合，用 MinHash 壓成 128 個數字的簽章，再用 LSH 分桶，只比可能相似的候選對，最後用精確的 Jaccard 驗證。

## 正文

### 近似重複

上一集的 exact dedup 只認得一字不差的行。可是網路上更常見的是「幾乎一樣」：轉載的文章多了一行來源、同一份模板換了標題與日期、網站的兩個網址顯示同一篇內容但廣告不同。這些文件逐行比對時，大部分的行會被認出來，但整份文件有多相似，上一集的做法沒有回答。

這一集要回答兩件事：兩份文件「有多相似」怎麼用一個數字表示；以及有一大堆文件的時候，怎麼不用兩兩比對就找出相似度高的。

### shingle 與 Jaccard

把一份文件看成一個集合，集合的元素是文件裡每一段連續 5 個字，叫 shingle（原意是屋頂的瓦片，一片壓著一片）。兩份文件的相似度用 Jaccard 相似度表示，也就是兩個集合的交集大小除以聯集大小：

\\[ J(A, B) = \frac{|A \cap B|}{|A \cup B|} \\]

完全一樣是 1，沒有任何共同的 shingle 是 0。程式把每個 shingle 換成 64 bit 的指紋（用上一集的 `line_hash`），集合就是一個整數陣列：

```python,ignore
{{#include ../../../src/llmz/pipeline.py:shingles}}
```

`jaccard` 用 numpy 的 `np.intersect1d` 求兩個陣列的交集，`assume_unique=True` 告訴它兩邊本來就沒有重複的元素。先用兩句只差一個字的句子試試：

```python
from llmz.pipeline import jaccard, shingles

a = "the quick brown fox jumps over the lazy dog near the old river bank"
b = "the quick brown fox jumps over the sleepy dog near the old river bank"
c = "a slow green turtle walks under the tall bridge by the quiet lake shore"
sa, sb, sc = shingles(a), shingles(b), shingles(c)

print("shingle 數：", sa.size, sb.size, sc.size)
print(f"a 與 b（差一個字）：{jaccard(sa, sb):.3f}")
print(f"a 與 c（完全不同）：{jaccard(sa, sc):.3f}")
```

執行結果：

```text
shingle 數： 10 10 10
a 與 b（差一個字）：0.333
a 與 c（完全不同）：0.000
```

14 個字的句子有 10 個 shingle。一個字會出現在最多 5 個 shingle 裡，所以改一個字，a 的 10 個 shingle 就有 5 個變了：5 個共有、聯集 15 個，Jaccard 掉到 0.33。長的文件沒有這麼劇烈，道理相同：改一個字就影響 5 個 shingle。用 400 個字的文件，在均勻分布的位置改幾個字，看 Jaccard 掉多少（最後一欄是下一節 MinHash 的估計，先不用管它）：

```python
import numpy as np

from llmz.pipeline import jaccard, minhash_signature, shingles

rng = np.random.default_rng(0)
words = list(rng.choice([f"w{i}" for i in range(5000)], 400))  # 原創的假文件：從 5000 個假字隨機抽 400 個
original = shingles(" ".join(words))
signature = minhash_signature(original)

print("改了幾個字  Jaccard  MinHash 估計")
for edits in [1, 2, 4, 8, 20, 40]:
    edited = list(words)
    for k in range(edits):
        edited[k * 400 // edits + 5] = f"new{k}"  # 改的位置平均分散在文件裡
    other = shingles(" ".join(edited))
    estimate = (signature == minhash_signature(other)).mean()
    print(f"{edits:10d}  {jaccard(original, other):7.3f}  {estimate:12.3f}")
```

執行結果：

```text
改了幾個字  Jaccard  MinHash 估計
         1    0.975         0.984
         2    0.951         0.945
         4    0.904         0.898
         8    0.817         0.828
        20    0.597         0.586
        40    0.329         0.289
```

400 個字裡改 8 個字（2%），Jaccard 就掉到 0.82；改 40 個字（10%）只剩 0.33。所以「Jaccard >= 0.8」在這樣的文件上，相當於改動不超過 2% 左右的字。本集用 0.8 當近似重複的門檻。

### 兩兩比的問題

有了相似度，最直接的做法是每一對都算一次。\\( N \\) 份文件有 \\( N(N-1)/2 \\) 對：817 頁是 333,336 對，用一點小技巧（大小差一倍以上的對不可能達到 0.5，直接跳過）在這個檔案上只要 1.5 秒。可是文件數每多 10 倍，對數多 100 倍；十億（\\( 10^9 \\)）份文件約 \\( 5 \times 10^{17} \\) 對，每一對還要比兩個幾百個元素的集合，不可能。

要解決兩件事：每份文件別用整個集合表示（太大），用一個固定長度的小簽章；也別每一對都比，只比「很可能相似」的。前一件事是 MinHash，後一件是 LSH。

### MinHash：把集合壓成一串數字

想像有一個把所有可能的 shingle 隨機重新排隊的方法（隨機排列）。對任一個集合，看它的元素在新的隊伍裡誰排最前面，這個元素就是這個集合的 MinHash。

兩個集合 A 與 B 合在一起的聯集裡，排最前面的元素只有一個。它如果在交集裡，A 與 B 的 MinHash 就是同一個元素；如果只在 A 或只在 B，兩個 MinHash 就不同。隨機排列下，聯集裡每個元素排最前面的機會均等，所以

\\[ P(\text{MinHash}(A) = \text{MinHash}(B)) = \frac{|A \cap B|}{|A \cup B|} = J(A, B) \\]

兩個集合的 MinHash 相同的機率，剛好就是 Jaccard 相似度。一次排列只能得到「相同」或「不同」，所以多做幾次：用 \\( k \\) 種不同的隨機排列，每種各得到一個 MinHash，連起來就是 \\( k \\) 個數字的簽章（signature）。兩份簽章裡「同一格相同」的比例，就是 \\( J \\) 的估計。每一格是一次獨立的試驗，所以估計的標準差是 \\( \sqrt{J(1-J)/k} \\)，格數越多越準。

```python,ignore
{{#include ../../../src/llmz/pipeline.py:minhash_signature}}
```

隨機排列不可能真的把 \\( 2^{64} \\) 個可能的 shingle 排一遍，所以用 \\( x \mapsto a x + b \\) 當排列：每份文件用同一組 \\( a, b \\)（同一個 `seed`），每個集合取每種排列下的最小值。`a` 是奇數時這是 \\( 2^{64} \\) 上的一一對應（`uint64` 乘法超過 64 bit 的部分自動捨去，就是 mod \\( 2^{64} \\)）。這不是嚴格的「隨機排列」，不過 shingle 的指紋本來就像隨機數，實測誤差和理論的標準差吻合。用兩個集合 \\( J = 0.5 \\)，換不同的 `seed` 重複 100 次，看估計怎麼隨 \\( k \\) 變：

```python
import numpy as np

from llmz.pipeline import jaccard, minhash_signature

rng = np.random.default_rng(0)
ids = rng.integers(0, 2**64, 200, dtype=np.uint64)
a, b = ids[:150], ids[50:]  # 交集 100 個、聯集 200 個
J = jaccard(a, b)
print("精確的 Jaccard：", J)

for k in [16, 64, 256, 1024]:
    estimates = np.array([(minhash_signature(a, k, seed) == minhash_signature(b, k, seed)).mean() for seed in range(100)])
    print(f"k={k:4d}：估計的平均 {estimates.mean():.3f}，標準差 {estimates.std():.3f}（理論 {(J * (1 - J) / k) ** 0.5:.3f}）")
```

執行結果：

```text
精確的 Jaccard： 0.5
k=  16：估計的平均 0.497，標準差 0.114（理論 0.125）
k=  64：估計的平均 0.493，標準差 0.064（理論 0.062）
k= 256：估計的平均 0.501，標準差 0.029（理論 0.031）
k=1024：估計的平均 0.498，標準差 0.016（理論 0.016）
```

估計的平均都在 0.5 附近，沒有偏高或偏低；標準差跟著 \\( 1/\sqrt{k} \\) 縮小，與理論的差不多。要把誤差減半，格數要變成 4 倍。我們用 128 格：\\( J = 0.5 \\) 時標準差約 0.044，簽章只有 1 KiB（128 個 8 byte），不管文件多長。

### LSH：只比同一個桶裡的

簽章有了，還是要找出簽章相似的對，但不想每一對都比。LSH（locality-sensitive hashing）的做法是：把 128 格切成 \\( b \\) 段（band），每段 \\( r \\) 格，\\( b \times r = 128 \\)。每一段當成一個鍵，把整段相同的文件放進同一個桶（bucket）。只要有任何一段，兩份文件的 \\( r \\) 格全部相同，這兩份就是候選對。

```python,ignore
{{#include ../../../src/llmz/pipeline.py:lsh_candidates}}
```

相似度 \\( s \\) 的兩份文件，一格相同的機率是 \\( s \\)，一整段 \\( r \\) 格都相同是 \\( s^r \\)，\\( b \\) 段裡至少有一段相同的機率是

\\[ P(\text{候選}) = 1 - (1 - s^r)^b \\]

```python,ignore
{{#include ../../../src/llmz/pipeline.py:lsh_probability}}
```

這條 S 形的曲線決定「多相似的對會被找到」。看幾種 \\( b, r \\) 的組合（都是 \\( b \times r = 128 \\)）：

```python
from llmz.pipeline import lsh_probability

configs = [(128, 1), (32, 4), (16, 8), (8, 16)]
print("相似度  " + "  ".join(f"b={b:3d},r={r:2d}" for b, r in configs))
for s in [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    print(f"{s:6.2f}  " + "  ".join(f"{lsh_probability(s, b, r):10.3f}" for b, r in configs))
```

執行結果：

```text
相似度  b=128,r= 1  b= 32,r= 4  b= 16,r= 8  b=  8,r=16
  0.20       1.000       0.050       0.000       0.000
  0.40       1.000       0.564       0.010       0.000
  0.50       1.000       0.873       0.061       0.000
  0.60       1.000       0.988       0.237       0.002
  0.70       1.000       1.000       0.613       0.026
  0.80       1.000       1.000       0.947       0.204
  0.90       1.000       1.000       1.000       0.806
  0.95       1.000       1.000       1.000       0.990
```

每一欄都是從 0 爬到 1 的 S 形，爬得最快的位置大約在 \\( (1/b)^{1/r} \\)：\\( b=16, r=8 \\) 約 0.71，\\( b=8, r=16 \\) 約 0.88。\\( r \\) 越大，一段要全部相同越難，門檻右移；\\( b \\) 越大，機會越多，門檻左移。我們要找 \\( J \ge 0.8 \\) 的對，所以選 \\( b=16, r=8 \\)：\\( J=0.8 \\) 有 94.7% 成為候選，\\( J \ge 0.9 \\) 幾乎一定；\\( J=0.5 \\) 只有 6%。

這條曲線是理論。用人造的集合實測：每個相似度各做 300 對，看有多少成為候選：

```python
import numpy as np

from llmz.pipeline import lsh_candidates, lsh_probability, minhash_signature

rng = np.random.default_rng(0)
only = 30  # 每個集合各有 30 個自己獨有的元素
for s in [0.5, 0.6, 0.7, 0.8, 0.9]:
    shared = round(2 * only * s / (1 - s))  # J = shared / (shared + 2 * only)
    hits = 0
    for _ in range(300):
        ids = rng.integers(0, 2**64, shared + 2 * only, dtype=np.uint64)
        a, b = ids[: shared + only], np.concatenate([ids[:shared], ids[shared + only :]])
        signatures = np.stack([minhash_signature(a), minhash_signature(b)])
        hits += bool(lsh_candidates(signatures, bands=16, rows=8))  # 非空的集合就是這一對成了候選
    print(f"相似度 {s}：理論 {lsh_probability(s, 16, 8):.3f}，實測 {hits / 300:.3f}")
```

執行結果：

```text
相似度 0.5：理論 0.061，實測 0.050
相似度 0.6：理論 0.237，實測 0.243
相似度 0.7：理論 0.613，實測 0.620
相似度 0.8：理論 0.947，實測 0.943
相似度 0.9：理論 1.000，實測 1.000
```

實測與理論相差在 300 對的取樣誤差內（約 ±0.03）。

### 驗證，再去掉

LSH 只保證「相似的對很可能被放進同一個桶」，沒有保證放在同一個桶的就相似，也可能漏掉一些。所以候選對還要逐對算精確的 Jaccard，低於門檻的丟掉。這樣分工：LSH 負責不漏掉相似的對（寧可多給候選），驗證負責不冤枉。

```python,ignore
{{#include ../../../src/llmz/pipeline.py:verify_pairs}}
```

驗證過的 `(i, j)` 再交給 `drop_near_duplicates`：近似重複的頁面只留排在最前面的那一份。

```python,ignore
{{#include ../../../src/llmz/pipeline.py:drop_near_duplicates}}
```

配對依 `i` 由小到大處理：`i` 是否被丟掉，已經由排在它前面的文件決定了。`i` 如果已經被丟掉，就不能當 `j` 的代表，所以三份文件 A、B、C 裡，A 與 B 相似、B 與 C 相似、A 與 C 卻不夠相似時，留下 A 與 C。

用五頁假網頁走一遍：第 3 頁是第 1 頁改了兩個字，第 4 頁是第 1 頁的原樣複本，其餘是各自獨立的隨機文字：

```python
import numpy as np

from llmz.pipeline import Page, drop_near_duplicates, lsh_candidates, minhash_signature, shingles, verify_pairs

rng = np.random.default_rng(0)
vocab = [f"w{i}" for i in range(2000)]


def random_page(url, n_words=200):
    return Page(url, " ".join(rng.choice(vocab, n_words)))


base = random_page("http://a.example/base")
words = base.text.split()
words[50], words[120] = "edited", "changed"
pages = [
    base,
    random_page("http://b.example/other"),
    Page("http://c.example/near", " ".join(words)),
    Page("http://d.example/copy", base.text),
    random_page("http://e.example/last"),
]

sets = [shingles(page.text) for page in pages]
signatures = np.stack([minhash_signature(s) for s in sets])
candidates = sorted(lsh_candidates(signatures, bands=16, rows=8))
print("候選對：", candidates)
for i, j, similarity in verify_pairs(candidates, sets, threshold=0.8):
    print(f"{pages[i].url} 與 {pages[j].url}：Jaccard {similarity:.3f}")
print("留下：", [page.url for page in drop_near_duplicates(pages)])
```

執行結果：

```text
候選對： [(0, 2), (0, 3), (2, 3)]
http://a.example/base 與 http://c.example/near：Jaccard 0.903
http://a.example/base 與 http://d.example/copy：Jaccard 1.000
http://c.example/near 與 http://d.example/copy：Jaccard 0.903
留下： ['http://a.example/base', 'http://b.example/other', 'http://e.example/last']
```

### 在真的網頁上

用 `data/cc/pages.jsonl` 的 817 頁。先做兩件事：算 shingle 與簽章，再用兩兩比對求「標準答案」，看 LSH 在不同的 \\( b, r \\) 下找回多少 Jaccard >= 0.8 的對：

```python,norun
import time

import numpy as np

from llmz.pipeline import lsh_candidates, lsh_probability, minhash_signature, read_pages, shingles, verify_pairs

pages = list(read_pages("data/cc/pages.jsonl"))

start = time.perf_counter()
sets = [shingles(page.text) for page in pages]
t_shingle = time.perf_counter() - start
start = time.perf_counter()
signatures = np.stack([minhash_signature(s, 128) for s in sets])
t_signature = time.perf_counter() - start
print(f"{len(pages)} 頁，每頁 shingle 數的中位數 {np.median([s.size for s in sets]):.0f}")
print(f"shingle {t_shingle:.1f} 秒，128 個數字的簽章 {t_signature:.1f} 秒")

# 標準答案：兩兩算精確的 Jaccard。J <= 小集合 / 大集合，所以大小差超過一倍的對不可能 >= 0.5，直接跳過
start = time.perf_counter()
order = sorted((i for i in range(len(sets)) if sets[i].size), key=lambda i: sets[i].size)
pairs = (
    (min(i, j), max(i, j))
    for a, i in enumerate(order)
    for j in order[a + 1 :]
    if sets[i].size >= 0.5 * sets[j].size  # order 由小到大排，所以後面的 j 只會更大
)
truth = verify_pairs(pairs, sets, 0.5)
print(f"兩兩比對 {time.perf_counter() - start:.1f} 秒：Jaccard >= 0.5 的有 {len(truth)} 對")
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
    print(f"  >= {threshold}：{sum(s >= threshold for _, _, s in truth)} 對")

true_pairs = {(i, j) for i, j, s in truth if s >= 0.8}
print("\n候選對（128 個數字切成 bands 段、每段 rows 個）：")
for bands, rows in [(128, 1), (32, 4), (16, 8), (8, 16)]:
    candidates = lsh_candidates(signatures, bands, rows)
    found = len(candidates & true_pairs)
    print(
        f"  bands={bands:3d} rows={rows:2d}：候選 {len(candidates):7,} 對，找回 {found}/{len(true_pairs)} 對 Jaccard >= 0.8 的，"
        f"理論上 J=0.8 的機率 {lsh_probability(0.8, bands, rows):.3f}"
    )
```

M4 Max 上的輸出（兩次執行的時間範圍：shingle 0.4～0.5 秒、兩兩比對 1.1～1.4 秒，這裡貼第一次；其餘每次相同）：

```text
817 頁，每頁 shingle 數的中位數 187
shingle 0.5 秒，128 個數字的簽章 0.1 秒
兩兩比對 1.4 秒：Jaccard >= 0.5 的有 57 對
  >= 0.5：57 對
  >= 0.6：50 對
  >= 0.7：39 對
  >= 0.8：37 對
  >= 0.9：19 對
  >= 0.99：16 對

候選對（128 個數字切成 bands 段、每段 rows 個）：
  bands=128 rows= 1：候選     710 對，找回 37/37 對 Jaccard >= 0.8 的，理論上 J=0.8 的機率 1.000
  bands= 32 rows= 4：候選     112 對，找回 37/37 對 Jaccard >= 0.8 的，理論上 J=0.8 的機率 1.000
  bands= 16 rows= 8：候選      51 對，找回 37/37 對 Jaccard >= 0.8 的，理論上 J=0.8 的機率 0.947
  bands=  8 rows=16：候選      32 對，找回 31/37 對 Jaccard >= 0.8 的，理論上 J=0.8 的機率 0.204
```

333,336 對裡，Jaccard >= 0.8 的有 37 對。其中 16 對 >= 0.99：有一組 6 份一模一樣的頁面（兩兩 15 對），加上另一組 2 份的（1 對）；其他 21 對是「差一點」的頁面。

候選越寬鬆，要驗證的對越多：\\( b=128, r=1 \\) 有 710 對候選，其中 673 對驗證後不到 0.8；\\( b=16, r=8 \\) 只剩 51 對候選，卻一樣找回全部 37 對；\\( b=8, r=16 \\) 候選最少（32 對），漏了 6 對。和 S 曲線的預測一致：\\( r \\) 越大，門檻越往右移。表裡「理論上 J=0.8 的機率」只對 J 剛好是 0.8 的對成立，真實的對有一半以上 >= 0.9（19 對），所以 \\( b=8, r=16 \\) 實際找回 31/37，比 20% 這個數字好得多。

這個規模看不出 LSH 的好處：兩兩比對（扣掉小集合跳過的對）只要 1.1～1.4 秒。LSH 把「每一對都比」換成「簽章線性掃一遍、只驗證候選」，簽章與分桶的工作量和文件數成正比（桶不大的時候），兩兩比對卻和文件數的平方成正比。

### 在真的網頁上：去掉多少頁

整個流程交給 `drop_near_duplicates`，預設 `bands=16, rows=8`，門檻 0.8。換三種門檻，也換兩種輸入：原始的頁面，與先做上一集的 `dedup_lines(pages, 5)` 之後的頁面；最後多印一個對照，用最寬鬆的 `bands=128, rows=1`（候選幾乎不會漏，結果等同兩兩比對）：

```python,norun
import time

from llmz.pipeline import dedup_lines, drop_near_duplicates, read_pages

pages = list(read_pages("data/cc/pages.jsonl"))

for name, source in [("原始", pages), ("先 dedup_lines(min_pages=5)", dedup_lines(pages, 5))]:
    print(f"{name}（{len(source)} 頁）：")
    for threshold in [0.5, 0.8, 0.9]:
        start = time.perf_counter()
        kept = drop_near_duplicates(source, threshold)  # 預設 bands=16, rows=8
        seconds = time.perf_counter() - start
        loose = drop_near_duplicates(source, threshold, bands=128, rows=1)  # 最寬鬆的設定，當作對照
        print(
            f"  threshold={threshold}：丟掉 {len(source) - len(kept)} 頁（{seconds:.1f} 秒），"
            f"bands=128、rows=1 丟掉 {len(source) - len(loose)} 頁"
        )
```

M4 Max 上的輸出（時間三次執行的範圍是 0.5～1.2 秒，這裡貼第一次；其餘每次相同）：

```text
原始（817 頁）：
  threshold=0.5：丟掉 22 頁（0.6 秒），bands=128、rows=1 丟掉 28 頁
  threshold=0.8：丟掉 15 頁（0.8 秒），bands=128、rows=1 丟掉 15 頁
  threshold=0.9：丟掉 8 頁（1.2 秒），bands=128、rows=1 丟掉 8 頁
先 dedup_lines(min_pages=5)（811 頁）：
  threshold=0.5：丟掉 7 頁（0.8 秒），bands=128、rows=1 丟掉 12 頁
  threshold=0.8：丟掉 3 頁（0.9 秒），bands=128、rows=1 丟掉 3 頁
  threshold=0.9：丟掉 2 頁（0.9 秒），bands=128、rows=1 丟掉 2 頁
```

三件事：

- **門檻 0.8 與 0.9，LSH 沒有漏**：預設設定丟掉的頁數與最寬鬆的相同（15 與 15、8 與 8）。門檻 0.5 時預設設定少丟了 6 頁（22 與 28）：\\( b=16, r=8 \\) 是照 0.8 調的，J 在 0.5 到 0.7 的對只有 6% 到 61% 的機會成為候選。門檻變了，\\( b, r \\) 也要跟著變。
- **丟掉的是「後面那一份」**：817 頁丟掉 15 頁，是 37 對相似的頁面裡每一組只留排在最前面的一份。
- **順序會改變答案**：先拿掉每頁共用的行，近似重複從 15 頁降到 3 頁。那 15 頁裡有 5 頁是上面那 6 份一模一樣的頁面，被 `dedup_lines` 整組拿光了（811 頁裡一頁都沒有），有 7 頁在拿掉共用的行之後不再相似：其中 6 頁的網址是同一個樣式，是同一個模板產生的頁面，原本 41 行大多是共用的，拿掉之後只剩 1 到 14 行，各自不同。MinHash 是看整份文件的 shingle，模板占了大半，頁面就顯得很像。剩下的 3 頁是真的內容重複。本章後面把步驟組成管線時，順序是要決定的事。

簽章的估計準不準，用上面 57 對 Jaccard >= 0.5 的頁面直接比：簽章裡相同的比例，與精確的 Jaccard 差多少。換四個 `seed`（四組不同的隨機排列）：

```python,norun
import numpy as np

from llmz.pipeline import jaccard, minhash_signature, read_pages, shingles

pages = list(read_pages("data/cc/pages.jsonl"))
sets = [shingles(page.text) for page in pages]

# Jaccard >= 0.5 的頁面對（大小差超過一倍的不可能）
order = sorted((i for i in range(len(sets)) if sets[i].size), key=lambda i: sets[i].size)
pairs = [
    (i, j)
    for a, i in enumerate(order)
    for j in order[a + 1 :]
    if sets[i].size >= 0.5 * sets[j].size and jaccard(sets[i], sets[j]) >= 0.5
]
exact = np.array([jaccard(sets[i], sets[j]) for i, j in pairs])
print(f"{len(pairs)} 對")

for seed in range(4):
    signatures = {i: minhash_signature(sets[i], 128, seed) for pair in pairs for i in pair}
    estimate = np.array([(signatures[i] == signatures[j]).mean() for i, j in pairs])
    error = estimate - exact
    print(f"seed={seed}：估計 - 精確 的平均 {error.mean():+.3f}，絕對值的平均 {np.abs(error).mean():.3f}，最大 {np.abs(error).max():.3f}")
```

M4 Max 上的輸出（每次相同）：

```text
57 對
seed=0：估計 - 精確 的平均 +0.022，絕對值的平均 0.029，最大 0.089
seed=1：估計 - 精確 的平均 -0.008，絕對值的平均 0.024，最大 0.087
seed=2：估計 - 精確 的平均 +0.013，絕對值的平均 0.025，最大 0.086
seed=3：估計 - 精確 的平均 +0.010，絕對值的平均 0.024，最大 0.085
```

單一一對的誤差平均 0.02 多、最大 0.09 左右，與前面 \\( \sqrt{J(1-J)/128} \\)（\\( J=0.5 \\) 時 0.044）相符。平均誤差的正負號隨 `seed` 變：這 57 對裡有很多對共用同一份內容（例如那 6 份一模一樣的頁面，兩兩 15 對用的是同樣 6 份簽章），不是 57 次獨立的試驗，所以單一 `seed` 的平均沒有互相抵消，不代表估計有偏。

### 這個版本做不到的事

- **用空白分字**：日文與沒有空格的中文會分得很粗，shingle 變得稀疏。要改用字元 n-gram，把 `shingles` 裡的 `words` 換成字元的串列就行；這份網頁裡的中文頁多數還是有空格與標點分開的字，所以先不改。
- **大桶**：一桶裡有 \\( m \\) 份文件就產生 \\( m(m-1)/2 \\) 對。6 份一模一樣的頁面是 15 對，不要緊；同一個模板如果有 10 萬份複本，就是約 50 億對。桶很大的時候要改成只跟桶裡的一份代表比。
- **留第一份**：`drop_near_duplicates` 留下的是排在最前面的，不看哪一份品質比較好。
- **邊界上的漏網之魚**：\\( J=0.8 \\) 的對有 5.3% 不會成為候選。要更保險就加大 \\( b \\)（或減小 \\( r \\)），代價是候選變多、簽章變長。
- **簽章的大小**：每份文件 1 KiB，十億份文件約 1 TB。每一段的分桶互相獨立，大規模時按段（以及桶的鍵）分給很多台機器。
- **段落級的重複**：兩份很不一樣的長文件，共用同一個長段落，整份的 Jaccard 很低，抓不到。Lee 等人（2022）用 suffix array 找重複的子字串處理這件事，本書不做。

## 重點整理

- 近似重複：把文件看成「連續 5 個字」（shingle）的集合，相似度用 Jaccard（交集 / 聯集）。改一個字會讓 5 個 shingle 不同，400 個字改 8 個字，Jaccard 就掉到 0.82。
- MinHash：隨機排列下兩個集合的最小元素相同的機率剛好是 Jaccard；用 128 種排列得到 128 個數字的簽章，相同格的比例就是估計，標準差 \\( \sqrt{J(1-J)/k} \\)。簽章大小固定，與文件長度無關。
- LSH：128 格切成 \\( b \\) 段、每段 \\( r \\) 格，某一段整段相同的文件成為候選，機率 \\( 1-(1-s^r)^b \\) 是 S 形，\\( r \\) 越大門檻越右移。實測與理論吻合；候選要再用精確的 Jaccard 驗證。
- 在 817 頁上，Jaccard >= 0.8 的有 37 對，預設的 \\( b=16, r=8 \\) 只比 51 對就全部找回，丟掉 15 頁；先做 `dedup_lines` 的話，只剩 3 頁。步驟的順序會改變每一步看到的東西。
