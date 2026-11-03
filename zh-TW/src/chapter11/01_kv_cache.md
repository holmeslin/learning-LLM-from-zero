# KV cache

## 本集目標

找出第 6 章生成迴圈重複計算的部分，用 KV cache 把每一層算過的 K、V 存起來，讓每一步只算新的 token，並驗證結果和原本完全一樣。

## 正文

### 重算了多少

第 6 章第 4 集的 `generate` 每一步都把整段序列餵進模型，只取最後一個位置的 logits。prompt 有 13 個 token、生成 200 個時，模型一共處理了多少個位置？

```python
prompt, new = 13, 200
naive = sum(range(prompt, prompt + new))
print(f"每一步重算全部：{naive} 個位置")
print(f"每個位置只算一次：{prompt + new - 1} 個位置")
```

執行結果：

```text
每一步重算全部：22500 個位置
每個位置只算一次：212 個位置
```

`sum(range(a, b))` 是 \\( a + (a+1) + \cdots + (b-1) \\)：第一步處理 13 個位置，第二步 14 個，依此類推。生成越長，浪費越多，總量跟長度的平方成正比。

### 哪些東西可以存

回想第 4 章的 attention：每個位置算出自己的 q、k、v，第 \\( t \\) 個位置用自己的 q 去和第 \\( 1 \\) 到 \\( t \\) 個位置的 k 算分數，再加權平均它們的 v。因為 causal mask，**前面位置的 k 與 v 永遠不會因為後面多了 token 而改變**：第 5 個位置的 k 只由前 5 個 token 決定。

所以生成新 token 時：

- 新位置的 q、k、v 要算；
- 舊位置的 k、v 和上一步一模一樣，存起來直接用；
- 舊位置的 q 不需要了，它們的輸出上一步已經用過。

每一層各存一份 K 與 V，就是 **KV cache**（KV 快取）。之後每一步只餵一個新 token，模型對它算 q、k、v，把 k、v 接到 cache 後面，再用 q 對整個 cache 做 attention。

### 程式：預先配置好的 cache

```python,ignore
{{#include ../../../src/llmz/inference.py:kv_cache}}
```

所有層的 K 放在一個張量裡，形狀 `(n_layers, B, H, max_len, d_head)`，V 也一樣。一開始就照 `context_length` 配置好，`length` 記錄目前填到哪裡。預先配置的好處是：之後不用一直接長張量，記憶體用量一開始就知道（第 3 集會算），要「退回」幾個位置也只要改 `length`（第 5 集會用到）。

### 帶 cache 的 attention

```python,ignore
{{#include ../../../src/llmz/inference.py:attention_cached}}
```

和第 4 章的 `MultiHeadSelfAttention.forward` 對照著看，差別有三個：

- **位置**：新的 \\( T \\) 個 token 接在 cache 裡 \\( P \\) 個位置後面，RoPE 要用 \\( P, P+1, \dots, P+T-1 \\)，不是從 0 開始。
- **寫進 cache**：新算出來的 k、v 放到第 \\( P \\) 到 \\( P+T-1 \\) 格。
- **mask**：新的第 \\( i \\) 個 token 在整段序列的位置是 \\( P+i \\)，它看得到位置 \\( 0 \\) 到 \\( P+i \\)。`torch.arange(max_len) <= positions[:, None]` 用 broadcasting 一次做出 `(T, max_len)` 的布林矩陣；cache 裡還沒填的位置（大於 \\( P+T-1 \\)）自然都是 False。

這裡故意對**整個** cache 做 attention，用 mask 遮掉還沒用到的格子，而不是只取前 \\( P+T \\) 格。我們第一次寫的版本取的是 `cache.k[layer, :, :, : P + T]`，結果在 MPS 上每一步要 15～29 ms，改成固定長度後只要 3～4 ms。原因是 MPS 每遇到一個沒見過的張量形狀，就要重新準備一次計算圖；只取前 \\( P+T \\) 格時，每一步的形狀都不一樣。多算的幾格 attention 很便宜，固定形狀省下的時間多得多。

### 帶 cache 的 forward

```python,ignore
{{#include ../../../src/llmz/inference.py:forward_cached}}
```

它重用模型自己的零件（`model.embed`、每一層的 `norm1`、`attn` 的投影、`ffn`、`norm_final`、`lm_head`），只是把 attention 換成上面的版本。為了簡短，只支援標準的 pre-norm。

有一個限制：cache 裡的 k 已經用 RoPE 依位置轉過了，不能像第 6 章那樣只保留最後 `context_length` 個 token 往前滑，否則位置全部要重算。所以 prompt 加上生成的長度不能超過 `context_length`（我們的模型是 256），超過就報錯。

### 驗證：和完整的 forward 一樣

用一個沒訓練過的小模型，先一次餵 6 個 token，再一個一個餵，每一步都和「整段重算」的結果比：

```python
import torch

from llmz.inference import KVCache, forward_cached
from llmz.model import ModelConfig, TransformerLM

torch.manual_seed(0)
cfg = ModelConfig(vocab_size=50, context_length=32, d_model=32, n_layers=2, n_heads=4, d_ff=64)
model = TransformerLM(cfg).eval()
ids = torch.randint(0, 50, (1, 10))
full = model(ids)  # 整段一次算

cache = KVCache(cfg, 1, torch.device("cpu"))
logits = forward_cached(model, ids[:, :6], cache)
print(6, torch.allclose(logits, full[:, :6], atol=1e-5), "cache 長度", cache.length)
for t in range(6, 10):
    logits = forward_cached(model, ids[:, t : t + 1], cache)
    print(t + 1, torch.allclose(logits[:, 0], full[:, t], atol=1e-5), "cache 長度", cache.length)
```

執行結果：

```text
6 True cache 長度 6
7 True cache 長度 7
8 True cache 長度 8
9 True cache 長度 9
10 True cache 長度 10
```

每一步都一樣，只差浮點數運算順序造成的極小誤差。`tests/test_inference.py` 還測了 cache 不是空的時候一次餵好幾個 token 的情況，第 5 集會用到。

### 用訓練好的模型比速度

`generate_cached`（下一集細看）用 KV cache 生成；沒有 cache 的版本就是每一步把整段餵進去：

```python,norun
import torch

from llmz.device import get_device
from llmz.inference import generate_cached
from llmz.model import ModelConfig, TransformerLM
from llmz.profile import timeit
from llmz.tokenizer import Tokenizer

device = get_device()
model = TransformerLM(ModelConfig()).to(device)
model.load_state_dict(torch.load("runs/base/model.pt", map_location=device))
model.eval()
tokenizer = Tokenizer.load("data/tokenizer.json")
ids = torch.tensor([tokenizer.encode("Once upon a time, there was a little girl named Lily.")], device=device)


@torch.no_grad()
def generate_naive(ids, n):
    for _ in range(n):
        ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
    return ids


print("結果相同：", torch.equal(generate_naive(ids, 200), generate_cached(model, ids, 200)))
for name, fn in [("沒有 cache", generate_naive), ("KV cache", lambda ids, n: generate_cached(model, ids, n))]:
    t = timeit(lambda: fn(ids, 200), device, warmup=2, repeat=5)
    print(f"{name}：{t:.2f} 秒，每秒 {200 / t:.0f} 個 token")
```

M4 Max 上的輸出（某一次）：

```text
結果相同： True
沒有 cache：5.45 秒，每秒 37 個 token
KV cache：0.77 秒，每秒 260 個 token
```

兩種方法都用 greedy，生成的 200 個 token 完全相同，KV cache 快了 7 倍。

## 重點整理

- 沒有 cache 時每一步重算整段序列，總計算量跟長度的平方成正比。
- causal attention 裡舊位置的 k、v 不會變，存起來就是 KV cache；每一步只算新 token 的 q、k、v。
- `llmz.inference.KVCache` 預先配置 `(n_layers, B, H, max_len, d_head)` 的 K、V；`forward_cached` 只餵新的 token，RoPE 的位置從 `cache.length` 接著算。
- 對整個 cache 做 attention、用 mask 遮掉沒用到的格子，形狀固定，MPS 上一步從 15～29 ms 降到 3～4 ms。
- cache 的 k 已經轉過位置，不能往前滑，總長度不能超過 `context_length`。
- 在 M4 Max 上生成 200 個 token：沒有 cache 5.45 秒，有 cache 0.77 秒，結果相同。
