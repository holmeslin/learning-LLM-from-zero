# `torch.compile`

## 本集目標

知道 `torch.compile` 為什麼能讓模型變快：把好幾個逐元素運算融合成一個 kernel，少搬幾趟資料；並用 RMSNorm 實測。

## 正文

### 一行一個 kernel

平常執行 PyTorch 程式叫 **eager 模式**：Python 每執行一行運算，就丟一個 kernel 給 GPU，算完的結果寫回記憶體，下一行再讀出來。看第 4 章的 RMSNorm：

```python,ignore
{{#include ../../../src/llmz/layers.py:rmsnorm}}
```

輸入是 `(32, 256, 512)` 的 float32，16.8 MB。`x * x` 讀一次、寫一次；`mean` 再讀一次；`x / rms` 讀、寫；`* self.weight` 再讀、寫。整個張量大小的讀寫加起來大約七趟，117 MB。可是 RMSNorm 真正需要的只有兩趟：讀一次輸入、寫一次輸出。

每個運算本身的計算量很小，時間幾乎都花在搬資料（第 1 集）。多搬的五趟就是浪費。

### 融合

`torch.compile(fn)` 會在第一次呼叫時追蹤 `fn` 執行了哪些運算，整理成一張計算圖，再交給編譯器產生新的 kernel。連續的逐元素運算會被**融合**（fuse）成一個 kernel：每個元素讀進來後，平方、加總、除、乘全部在晶片上做完，只寫一次結果。

```python,norun
import torch

from llmz.layers import RMSNorm
from llmz.profile import timeit

device = torch.device("mps")
norm = RMSNorm(512).to(device)
x = torch.randn(32, 256, 512, device=device)
fast = torch.compile(norm)
with torch.no_grad():
    eager = timeit(lambda: norm(x), device, repeat=500)
    compiled = timeit(lambda: fast(x), device, repeat=500)
print(f"eager {eager * 1e6:.0f} µs，compiled {compiled * 1e6:.0f} µs")
```

M4 Max 實測三次：eager 585～596 µs，compiled 85～96 µs，快了六倍多。用第 1 集的頻寬算：搬 117 MB 要 420 µs，搬 33.5 MB 要 120 µs，eager 的時間大致就是搬七趟的時間；compiled 比 120 µs 還快一些，因為 16.8 MB 的張量有一部分留在晶片的快取裡，不用每次都從記憶體讀。

`torch.no_grad()` 是第 2 章教過的：這裡只量 forward，不需要 autograd 記錄。

### 整個模型

`tools/train.py` 的 `--compile` 對整個模型做同一件事。第 5 章第 9 集量過：標準設定每步從 442 ms 降到 276 ms，快了 1.6 倍。模型裡的矩陣乘法本來就是一個大 kernel，編譯幫不上忙；省下來的是 RMSNorm、SwiGLU 的 `silu` 與乘法、RoPE、softmax 這些逐元素運算。

### 代價

- **第一次很慢**：RMSNorm 的第一次呼叫花了 0.41 秒（eager 不到 1 ms），整個模型要幾秒鐘。訓練幾千步完全划得來，只跑幾次的程式就不必編譯。
- **形狀改變會重新編譯**：編譯出來的 kernel 是針對某個形狀的。本書訓練時 batch 和長度固定，不會遇到；第 6 章的生成每一步長度都不一樣，長度一變就會觸發重新編譯（PyTorch 之後會改編一個允許形狀變動的版本），只生成幾十個 token 的話划不來。
- **出錯比較難查**：錯誤訊息會穿過編譯器，比 eager 模式長很多。先用 eager 把程式寫對，最後再加 `torch.compile`。

## 重點整理

- eager 模式一個運算一個 kernel，每個中間結果都寫回記憶體再讀出來。
- `torch.compile` 把連續的逐元素運算融合成一個 kernel，少搬資料；矩陣乘法本身不會變快。
- M4 Max 實測：RMSNorm 從約 590 µs 降到約 90 µs；整個模型每步快 1.6 倍。
- 代價：第一次呼叫要編譯、形狀改變會重新編譯、錯誤訊息較難讀。
