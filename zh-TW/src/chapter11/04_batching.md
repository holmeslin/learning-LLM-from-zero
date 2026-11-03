# batching

## 本集目標

讓 decode 一次替很多條序列各生成一個 token，量出 batch 變大時每秒能生成的 token 數怎麼變化，並認識實際服務時 batching 遇到的問題。

## 正文

### 同一份權重，多條序列一起用

上一集算過，decode 一步的時間主要花在讀權重（以及固定開銷），真正的計算很少。如果一次餵 \\( B \\) 條序列，權重一樣只讀一次，計算變成 \\( B \\) 倍，intensity 從 0.5 變成 \\( B/2 \\)。在變成 compute-bound 之前，一步的時間幾乎不變，每秒生成的 token 數卻是 \\( B \\) 倍。

`generate_cached` 本來就吃 `(B, T)` 的張量，`KVCache` 也有 batch 這一維，不用改任何程式。先確認一起生成和分開生成的結果一樣：

```python
import torch

from llmz.inference import generate_cached
from llmz.model import ModelConfig, TransformerLM

torch.manual_seed(0)
cfg = ModelConfig(vocab_size=50, context_length=32, d_model=32, n_layers=2, n_heads=4, d_ff=64)
model = TransformerLM(cfg).eval()
prompts = torch.randint(0, 50, (3, 5))  # 三條長度都是 5 的 prompt

together = generate_cached(model, prompts, 10)
for i in range(3):
    alone = generate_cached(model, prompts[i : i + 1], 10)
    print(i, torch.equal(together[i : i + 1], alone), together[i, 5:].tolist())
```

執行結果：

```text
0 True [14, 14, 14, 14, 14, 14, 38, 46, 40, 47]
1 True [0, 27, 0, 16, 14, 14, 14, 14, 20, 20]
2 True [14, 14, 20, 15, 9, 15, 48, 2, 41, 3]
```

batch 裡的每一條序列各自做 attention、各自查自己的 cache，互不干擾。

### 量量看

和第 2 集一樣取 60 次的中位數，cache 裡已有 64 個位置，量 decode 一步：

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


for B in [1, 4, 16, 64, 256, 512]:
    cache = KVCache(model.cfg, B, device)
    forward_cached(model, torch.randint(0, 10000, (B, 64), device=device), cache)
    x = torch.randint(0, 10000, (B, 1), device=device)

    def decode_step():
        cache.length = 64
        forward_cached(model, x, cache)

    t = median_time(decode_step)
    print(f"B={B}：一步 {t * 1000:.2f} ms，每秒 {B / t:.0f} 個 token，cache {cache.nbytes() / 1e6:.1f} MB")
```

MPS 是兩次執行的範圍，CPU 是一次：

| \\( B \\) | MPS 一步 | MPS 每秒 token | CPU 一步 | CPU 每秒 token | KV cache |
| --- | --- | --- | --- | --- | --- |
| 1 | 7.5～7.6 ms | 133 | 11.0 ms | 91 | 4.2 MB |
| 4 | 7.8～8.1 ms | 494～512 | 17.0 ms | 236 | 16.8 MB |
| 16 | 9.5～9.9 ms | 1,624～1,680 | 41.8 ms | 383 | 67.1 MB |
| 64 | 8.8～10.4 ms | 6,167～7,289 | 95.5 ms | 670 | 268.4 MB |
| 256 | 10.6～11.1 ms | 23,011～24,267 | 266.0 ms | 963 | 1,073.7 MB |
| 512 | 17.9～19.3 ms | 26,596～28,540 | 444.5 ms | 1,152 | 2,147.5 MB |

在 MPS 上，\\( B \\) 從 1 增加到 256，一步只從 7.5 ms 變成 11 ms，每秒生成的 token 數多了 170～180 倍。到 512 一步的時間才明顯增加，開始被算力限制。CPU 的算力小得多，\\( B = 4 \\) 一步的時間就變長了，但每秒的 token 數還是一路增加，512 時是 \\( B = 1 \\) 的 12 倍多。

代價是記憶體：每條序列都有自己的 KV cache，滿 256 個位置時每條 4.2 MB（上一集算的 4 MiB）。\\( B = 512 \\) 時 cache 要 2.1 GB，是權重（91 MB）的 23 倍。大模型的服務能同時接多少使用者，常常是被 KV cache 的記憶體卡住，而不是算力。

### 實際服務時的問題

上面的實驗故意讓每條 prompt 一樣長、生成一樣多的 token。真實的請求不會這樣：

- **長度不同**：prompt 長短不一，要補齊（padding）成一樣長，並用 mask 讓補上的位置不被看到。補得越多，浪費越多。
- **結束時間不同**：有的回答 10 個 token 就結束，有的要 1000 個。如果等整個 batch 都結束才換下一批，早結束的位置一直空轉。**continuous batching** 在每一步都檢查：有序列結束就馬上把它換成排隊中的新請求。
- **cache 的空間**：每條序列最後會多長事先不知道，照最長的長度預先配置（像我們的 `KVCache`）會浪費很多記憶體。**PagedAttention**（vLLM 用的方法）把 cache 切成固定大小的小塊，像作業系統管理記憶體分頁一樣，用到才配置。

這些都是推論伺服器（例如 vLLM、SGLang）的工作，本書不實作。

## 重點整理

- decode 一次處理 \\( B \\) 條序列，權重只讀一次，intensity 約 \\( B/2 \\)；變成 compute-bound 之前，一步的時間幾乎不變。
- `generate_cached` 與 `KVCache` 本來就支援 batch，一起生成和分開生成的結果相同。
- M4 Max 的 MPS 上，\\( B = 256 \\) 每秒生成兩萬多個 token，是 \\( B = 1 \\) 的 170～180 倍；\\( B = 512 \\) 開始被算力限制。
- 每條序列都有自己的 KV cache，batch 大時 cache 遠大於權重，常常是同時服務人數的上限。
- 實際服務要處理長度不同（padding）、結束時間不同（continuous batching）與 cache 空間（PagedAttention）。
