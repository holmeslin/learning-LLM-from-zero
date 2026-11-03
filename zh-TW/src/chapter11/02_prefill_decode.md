# prefill 與 decode

## 本集目標

知道帶 cache 的生成分成 prefill 與 decode 兩個階段，量出它們的速度差多少，並用第 7 章的 arithmetic intensity 解釋為什麼。

## 正文

### 兩個階段

```python,ignore
{{#include ../../../src/llmz/inference.py:generate_cached}}
```

- **prefill**（預填）：把整段 prompt 一次餵進去，一次 forward 算完所有位置，填滿 cache，順便得到第一個新 token。
- **decode**（解碼）：之後每一步只餵上一步選出的那一個 token，cache 長度加 1。

`pick_next` 處理一整個 batch：temperature 為 0 時每一列取 `argmax`；否則每一列各自交給第 6 章的 `sample_next` 抽樣，再用 `torch.stack` 疊回 `(B, 1)`。`generate_cached` 吃的是 token id 的張量、回傳的也是張量，沒有碰 tokenizer，方便第 4、5 集組合使用。

使用者感受到的延遲也分兩段：送出 prompt 到看到第一個字的時間（time to first token，主要是 prefill），以及之後每個字出現的間隔（主要是一步 decode）。

### 量量看

在 M4 Max 上用標準模型量，每次呼叫前後都同步、取 60 次的中位數。prefill 量不同的 prompt 長度 \\( T \\)；decode 量 cache 裡已有 64 個位置時，生成一個 token 的時間：

```python,norun
import statistics
import time

import torch

from llmz.device import get_device
from llmz.inference import KVCache, forward_cached
from llmz.model import ModelConfig, TransformerLM
from llmz.profile import sync

device = get_device()
model = TransformerLM(ModelConfig()).to(device)
model.load_state_dict(torch.load("runs/base/model.pt", map_location=device))
model.eval()


def median_time(fn, n=60):
    for _ in range(5):
        fn()
    times = []
    for _ in range(n):
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


for T in [1, 16, 64, 128, 256]:
    x = torch.randint(0, 10000, (1, T), device=device)
    t = median_time(lambda: forward_cached(model, x, KVCache(model.cfg, 1, device)))
    print(f"prefill T={T}：{t * 1000:.2f} ms，每秒 {T / t:.0f} 個 token")

cache = KVCache(model.cfg, 1, device)
forward_cached(model, torch.randint(0, 10000, (1, 64), device=device), cache)
x = torch.randint(0, 10000, (1, 1), device=device)


def decode_step():
    cache.length = 64  # 每次都從同一個長度開始，量的是同一件事
    forward_cached(model, x, cache)


t = median_time(decode_step)
print(f"decode 一步：{t * 1000:.2f} ms，每秒 {1 / t:.0f} 個 token")
```

`statistics.median` 取中位數，比平均值不受偶爾一次特別慢的影響。把 `get_device()` 換成 `torch.device("cpu")` 就是 CPU 的數字。兩次執行的結果：

| | MPS | CPU |
| --- | --- | --- |
| prefill \\( T=1 \\) | 6.9～7.0 ms | 16.5 ms |
| prefill \\( T=16 \\) | 9.0～9.1 ms | 34.5 ms |
| prefill \\( T=64 \\) | 6.7～12.6 ms | 54.2 ms |
| prefill \\( T=128 \\) | 9.9～11.9 ms | 102.2 ms |
| prefill \\( T=256 \\) | 9.0～12.1 ms | 142.6 ms |
| decode 一步 | 7.5～7.6 ms | 11.0 ms |

在 MPS 上，處理 1 個 token 和 256 個 token 的時間都在 7～12 ms 之間：prefill 每秒能處理 2 萬多個 token，decode 每秒只有 130 個左右，差了一百多倍。CPU 上 prefill 的時間隨 \\( T \\) 明顯增加，但處理 256 個 token 也只是 decode 一步的 13 倍，平均每個 token 便宜得多。

（第 1 集生成 200 個 token 花 0.77 秒，一步不到 4 ms，比這裡的 7.5 ms 快。差別在於這裡每一步都同步，量的是單獨一步的延遲；連續生成時 GPU 的佇列裡一直有工作，不用每一步都等。）

### 為什麼差這麼多

回到第 7 章的 arithmetic intensity：每讀一個位元組做幾個 FLOP。

一個 `Linear` 有 \\( n \\) 個參數（float32，\\( 4n \\) 個位元組）。餵進 \\( T \\) 個 token 時，矩陣乘法做 \\( 2nT \\) 個 FLOP，權重只要讀一次：

\\[
\text{intensity} \approx \frac{2nT}{4n} = \frac{T}{2}
\\]

- **decode**：\\( T = 1 \\)，每讀一個位元組只做 0.5 個 FLOP。第 7 章量到 M4 Max 的平衡點是每位元組 37 個 FLOP，decode 遠低於它，算力幾乎全在等資料。
- **prefill**：\\( T = 256 \\)，intensity 128，超過平衡點，變成算力決定速度。

所以 prefill 是 **compute-bound**，decode 是 **memory-bound**。對我們這麼小的模型，連讀權重都很快（下一集會算出只要多少時間），時間主要花在啟動一個個小運算的固定開銷上，這也是 MPS 上 1 個 token 和 256 個 token 時間差不多的原因。不管哪一種，decode 一次只處理一個 token 都很浪費。第 4 集的 batching 就是讓 decode 一次處理很多個序列，把 \\( T \\) 換成 batch 大小 \\( B \\)。

## 重點整理

- 帶 cache 的生成分兩段：prefill 一次處理整段 prompt；decode 每一步只處理一個新 token。
- `llmz.inference.generate_cached(model, ids, max_new_tokens, temperature, top_p)` 吃 `(B, T)` 的 id 張量，`pick_next` 一次替整個 batch 選下一個 token。
- 使用者看到的延遲：第一個 token 主要看 prefill，之後每個 token 看一步 decode。
- M4 Max 的 MPS 上，prefill 每秒處理 2 萬多個 token，decode 每秒 130 個左右。
- decode 的 arithmetic intensity 約 0.5 FLOP/位元組，是 memory-bound；prefill 約 \\( T/2 \\)，是 compute-bound。
