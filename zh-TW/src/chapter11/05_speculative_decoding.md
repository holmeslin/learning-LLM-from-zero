# speculative decoding

## 本集目標

用一個小的草稿模型先猜幾個 token，再讓大模型一次驗證；理解為什麼這樣產生的文字和大模型自己生成的分布完全相同，並量出在 Mac 上什麼時候會變快、什麼時候反而變慢。

## 正文

### 想法

第 2 集量過：處理 1 個 token 和處理好幾個 token，一次 forward 的時間差不多。decode 一次只算一個 token，大部分的時間都浪費了。

**speculative decoding**（推測解碼）利用這一點：

1. 用一個便宜的**草稿模型**（draft model）一個一個猜出 \\( k \\) 個 token。
2. 把這 \\( k \\) 個 token 一次餵給**目標模型**（target model），一次 forward 就得到它在這 \\( k \\) 個位置上的分布，外加第 \\( k+1 \\) 個位置的分布。
3. 從第一個開始逐一檢查：接受就留下，第一次拒絕時從修正過的分布重抽一個 token，後面的草稿全部丟掉。如果 \\( k \\) 個都接受，再從第 \\( k+1 \\) 個位置的分布多抽一個。

每一輪目標模型只做一次 forward，卻能產生 1 到 \\( k+1 \\) 個 token。

### 接受的規則

設目標模型在某個位置的分布是 \\( p \\)，草稿模型是 \\( q \\)，草稿依 \\( q \\) 抽到 \\( x \\)：

- 以 \\( \min(1, p(x)/q(x)) \\) 的機率接受 \\( x \\)；
- 拒絕的話，從 \\( \max(0, p - q) \\) 正規化之後的分布重抽。

```python,ignore
{{#include ../../../src/llmz/inference.py:accept_or_resample}}
```

這個規則保證最後的 token 恰好服從 \\( p \\)：草稿模型猜得比目標模型保守的 token（\\( q(x) \le p(x) \\)）一定接受；猜得太多的 token 只接受 \\( p(x)/q(x) \\) 的比例，少掉的那一部分機率，由重抽時的 \\( \max(0, p - q) \\) 補回來。所以草稿模型再爛，也只會讓速度變慢，**不會改變生成的分布**。

`to_probs` 在 temperature 為 0 時回傳 one-hot：這時接受的規則變成「草稿和目標的 argmax 一樣就接受，不一樣就換成目標的 argmax」，結果和目標模型自己做 greedy 一模一樣。

用兩個簡單的分布驗證：

```python
import torch

from llmz.inference import accept_or_resample

torch.manual_seed(0)
p = torch.tensor([0.5, 0.3, 0.15, 0.05])  # 目標模型
q = torch.tensor([0.1, 0.2, 0.3, 0.4])  # 很差的草稿模型
counts = torch.zeros(4)
n_accept = 0
for _ in range(20000):
    ok, t = accept_or_resample(p, q, torch.multinomial(q, 1).item())
    counts[t] += 1
    n_accept += ok
print("結果的分布", [round(c, 3) for c in (counts / counts.sum()).tolist()])
print("接受率", n_accept / 20000)
```

執行結果：

```text
結果的分布 [0.507, 0.303, 0.144, 0.046]
接受率 0.4948
```

草稿模型的分布和目標差很多，只有大約一半被接受（理論值是 \\( \sum_x \min(p(x), q(x)) = 0.1 + 0.2 + 0.15 + 0.05 = 0.5 \\)），但結果的分布和 \\( p \\) 一致。

### 完整的流程

```python,ignore
{{#include ../../../src/llmz/inference.py:speculative}}
```

- `tokens` 是已經確定的 token。兩個模型各有一個 cache，cache 裡永遠是 `tokens` 的前一段；`feed` 把還沒進 cache 的部分餵進去，回傳它們的 logits。
- 草稿模型第一次可能要餵好幾個 token（上一輪多出來的），之後每次一個。
- 目標模型一次餵進「還沒進 cache 的已確定 token」加上 \\( k \\) 個草稿，最後 \\( k+1 \\) 個位置的分布就是驗證要用的 `ps`。這就是第 1 集「cache 不空、一次餵好幾個 token」的情況。
- 這一輪新確定了 `n_ok` 個草稿，加上一個重抽或多送的 token。被拒絕的草稿已經寫進 cache 了，把 `length` 退回 `n + n_ok`，那些格子之後會被覆蓋。預先配置的 cache 讓「退回」只是改一個整數。
- `for ... else`：`else` 只在迴圈沒有被 `break` 時執行，也就是 \\( k \\) 個草稿全部接受的時候。

用兩個沒訓練過的小模型確認 greedy 時結果和目標模型一樣：

```python
import torch

from llmz.inference import generate_cached, speculative_generate
from llmz.model import ModelConfig, TransformerLM

torch.manual_seed(0)
target = TransformerLM(ModelConfig(vocab_size=50, context_length=64, d_model=32, n_layers=2, n_heads=4, d_ff=64)).eval()
draft = TransformerLM(ModelConfig(vocab_size=50, context_length=64, d_model=16, n_layers=1, n_heads=2, d_ff=32)).eval()
ids = torch.randint(0, 50, (1, 4))
out, accepted, proposed = speculative_generate(target, draft, ids, 30, k=3)
print("和目標模型的 greedy 一樣：", torch.equal(out, generate_cached(target, ids, 30)))
print(f"接受 {accepted} / {proposed} 個草稿")
```

執行結果：

```text
和目標模型的 greedy 一樣： True
接受 2 / 84 個草稿
```

兩個隨機模型幾乎沒有共識，草稿大多被拒絕，但結果仍然和目標模型完全一樣。

### 用真的模型量

目標模型是第 5 章的標準模型（2270 萬參數，val loss 1.69）。草稿模型借用第 10 章 scaling law 實驗訓練好的兩個小模型：\\( 10^{15} \\) 預算的 d128（294 萬參數，val loss 2.01）與 d256（834 萬參數，val loss 1.84）。它們用同一個 tokenizer，可以直接配對。

```python,norun
import json
import time

import torch

from llmz.device import get_device
from llmz.inference import generate_cached, speculative_generate
from llmz.model import ModelConfig, TransformerLM
from llmz.profile import sync
from llmz.tokenizer import Tokenizer

device = get_device()


def load(path, cfg):
    model = TransformerLM(cfg).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()


target = load("runs/base/model.pt", ModelConfig())
draft_cfg = ModelConfig(**json.load(open("runs/scaling/1e+15_d128/config.json"))["model"])
draft = load("runs/scaling/1e+15_d128/model.pt", draft_cfg)
tokenizer = Tokenizer.load("data/tokenizer.json")
ids = torch.tensor([tokenizer.encode("Once upon a time, there was a little girl named Lily.")], device=device)


def timed(fn):
    fn()  # 暖身
    sync(device)
    start = time.perf_counter()
    result = fn()
    sync(device)
    return time.perf_counter() - start, result


t, ref = timed(lambda: generate_cached(target, ids, 200))
print(f"目標模型自己：{t:.2f} 秒")
for k in [2, 4, 6]:
    t, (out, accepted, proposed) = timed(lambda: speculative_generate(target, draft, ids, 200, k=k))
    print(f"k={k}：{t:.2f} 秒，接受 {accepted}/{proposed}，和目標模型一樣：{torch.equal(out, ref)}")
```

`ModelConfig(**json.load(...)["model"])` 從訓練時存的 `config.json` 讀回模型的設定。生成 200 個 token、greedy（temperature 0）的結果：

| | CPU | MPS |
| --- | --- | --- |
| 目標模型自己 | 3.05 秒 | 0.70 秒 |
| d128 草稿自己 | 0.80 秒 | 0.46 秒 |
| d256 草稿自己 | 1.35 秒 | 0.80 秒 |
| d128 草稿，\\( k = 2 \\) | **2.24 秒** | 4.07 秒 |
| d128 草稿，\\( k = 4 \\) | 2.31 秒 | 4.34 秒 |
| d128 草稿，\\( k = 6 \\) | 2.22 秒 | 5.22 秒 |
| d256 草稿，\\( k = 2 \\) | 3.35 秒 | 4.38 秒 |
| d256 草稿，\\( k = 4 \\) | 3.78 秒 | 5.08 秒 |

所有 greedy 的組合，生成的 200 個 token 都和目標模型自己生成的完全相同。接受率與每一輪平均產生的 token 數（200 除以輪數）：

| 草稿 | \\( k \\) | 接受率 | 每輪 token |
| --- | --- | --- | --- |
| d128 | 2 | 124/156 = 0.79 | 2.6 |
| d128 | 4 | 144/224 = 0.64 | 3.6 |
| d128 | 6 | 158/276 = 0.57 | 4.3 |
| d256 | 2 | 125/154 = 0.81 | 2.6 |
| d256 | 4 | 148/208 = 0.71 | 3.8 |
| d256 | 6 | 162/252 = 0.64 | 4.8 |

temperature 0.8 時接受率比較低（d128、\\( k = 4 \\) 在兩次執行中是 0.47 與 0.55），因為抽樣讓兩個模型更常選到不同的 token。

### 為什麼有時變快、有時變慢

一輪的成本大約是「草稿模型 \\( k \\) 步 + 目標模型 1 步」，換來每輪 2.6～4.8 個 token。要划算，草稿模型的一步必須比目標模型便宜很多。

- **CPU、d128**：草稿一步約 4 ms，目標約 15 ms，便宜將近 4 倍，所以快了 1.3～1.4 倍。
- **CPU、d256**：草稿一步約 7 ms，只便宜 2 倍多，再加上抽樣、比對這些 Python 的額外工作，就沒有變快。
- **MPS**：反而慢了 6～7 倍。每猜一個 token 都要 `.item()` 把結果拿回 CPU 才能決定下一步，GPU 的佇列被打斷，每一步都要等同步（第 2 集量過，同步的一步是 7.5 ms，連續生成時不到 4 ms）。但同步只解釋了一部分：\\( k = 2 \\) 時一輪三次 forward，同步的成本約 22 ms，一輪實測卻約 52 ms（4.07 秒 ÷ 78 輪）；其餘花在每一步建立張量、在 10000 個詞上抽樣與比對這些小工作，我們沒有再細分。在 MPS 上，三個模型都小到被固定開銷主導，草稿模型並沒有比較便宜。

實際使用時，目標模型通常有幾百億參數，草稿模型小了幾十倍，而且整個流程在 GPU 上完成，不在每一步回到 Python。提出這個方法的兩篇論文（Leviathan 等人與 Chen 等人，2023）在大模型上量到 2～3 倍的加速。

## 重點整理

- speculative decoding：草稿模型猜 \\( k \\) 個 token，目標模型一次 forward 驗證，每輪產生 1 到 \\( k+1 \\) 個 token。
- 以 \\( \min(1, p/q) \\) 接受、從 \\( \max(0, p-q) \\) 重抽，結果恰好服從目標模型的分布；temperature 0 時和目標模型的 greedy 完全一樣。
- `llmz.inference.speculative_generate` 用兩個 KV cache，被拒絕的草稿只要把 `length` 退回去。
- 在 CPU 上用 d128 當草稿快了 1.3～1.4 倍；d256 太貴沒有變快；MPS 上因為每一步都要同步，反而慢了 6～7 倍。
- 要變快，草稿模型的一步必須比目標模型便宜很多，接受率也要夠高。
