# `torch.profiler`

## 本集目標

用 `torch.profiler` 看出訓練一步裡每種運算各花了多少時間，找出最值得優化的地方。

## 正文

### 為什麼需要 profiler

`timeit` 告訴我們一步花多久，但不告訴我們花在哪裡。profiler（效能剖析器）會記錄執行過程中呼叫的每一個底層運算與時間，最後整理成一張表。

PyTorch 內建的是 `torch.profiler.profile`。它是一個 `with` 區塊，區塊裡執行的東西都會被記錄。`record_function("名字")` 可以在表裡標出自己的段落。

### 剖析一步訓練

為了讓它在 CPU 上幾秒鐘跑完，我們用縮小的模型：2 層、長度 128、batch 8。和第 2 集一樣，先跑一步 warmup 再開始記錄。

```python
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from llmz.functional import cross_entropy
from llmz.model import ModelConfig, TransformerLM

torch.manual_seed(0)
model = TransformerLM(ModelConfig(context_length=128, n_layers=2))
x = torch.randint(0, 10000, (8, 129))


def train_step():
    with record_function("forward"):
        logits = model(x[:, :-1])
        loss = cross_entropy(logits.reshape(-1, 10000), x[:, 1:].reshape(-1))
    with record_function("backward"):
        loss.backward()


train_step()  # warmup，不記錄
with profile(activities=[ProfilerActivity.CPU]) as prof:
    train_step()

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=8, max_name_column_width=14))
for event in prof.key_averages():
    if event.key in ("forward", "backward"):
        print(event.key, f"{event.cpu_time_total / 1000:.0f} ms")
```

執行結果（某一次）：

```text
--------------  ------------  ------------  ------------  ------------  ------------  ------------  
          Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
--------------  ------------  ------------  ------------  ------------  ------------  ------------  
      aten::mm        37.56%      58.144ms        37.57%      58.162ms       1.292ms            45  
     aten::mul         8.08%      12.502ms         8.10%      12.544ms     140.946us            89  
     aten::exp         7.18%      11.109ms         7.18%      11.109ms       3.703ms             3  
aten::maske...         5.32%       8.237ms         5.32%       8.237ms       2.059ms             4  
       forward         5.04%       7.809ms        54.88%      84.969ms      84.969ms             1  
    aten::add_         4.38%       6.788ms         4.38%       6.788ms     119.095us            57  
aten::sigmo...         4.28%       6.628ms         4.28%       6.628ms       3.314ms             2  
     aten::max         3.66%       5.663ms         3.66%       5.663ms       1.888ms             3  
--------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 154.823ms

forward 85 ms
backward 70 ms
```

時間每台機器不同，但排序大致一樣。

### 怎麼讀這張表

每一列是一種運算，`aten::` 開頭的是 PyTorch 底層的 C++ 運算（`aten::mm` 是矩陣乘法、`aten::exp` 是指數）。幾個欄位：

- **Self CPU**：這個運算本身花的時間，不含它呼叫的其他運算。
- **CPU total**：包含它呼叫的其他運算。`forward` 和 `backward` 是我們用 `record_function` 標的段落，表格最後兩行另外印出它們的 total，就是整段的時間。
- **# of Calls**：被呼叫了幾次。

我們依 Self CPU 排序，最上面就是真正花時間的地方。看幾件事：

1. 矩陣乘法 `aten::mm` 排第一，這符合第 4 章第 15 集的估算：模型的 FLOPs 幾乎都在矩陣乘法。
2. 照第 2 章第 7 集的估算，`backward` 應該是 `forward` 的兩倍。我們在 M4 Max 的 CPU 上跑了幾次，比例從 0.8 倍到 1.8 倍都有：這麼小的模型，逐元素運算和固定開銷佔了大半，時間不照 FLOPs 走，CPU 上的時間本身也會跳動。估算是起點，最後還是要量。
3. `aten::mul`、`aten::exp`、`aten::div`、`aten::add_` 這些逐元素運算（RMSNorm、SwiGLU、RoPE、softmax 的零件）加起來也佔了兩成以上。它們的 FLOPs 很少，卻花了明顯的時間，因為它們卡在搬資料（第 1 集）。這就是第 8 章 FlashAttention 和第 7 集 `torch.compile` 要解決的問題。

### 在 GPU 上

NVIDIA 顯示卡上要加 `ProfilerActivity.CUDA` 才看得到每個 kernel 在 GPU 上的時間。`ProfilerActivity` 沒有 MPS 這個選項，在 Mac 上它只記錄 CPU 端丟工作的時間；PyTorch 另有 `torch.mps.profiler`，要搭配 Xcode 的 Instruments 看，本書不展開。在 Mac 上用 `timeit` 一段一段量比較實際。

profiler 也能匯出時間軸：`prof.export_chrome_trace("trace.json")`，用 Chrome 瀏覽器打開 `chrome://tracing` 或 <https://ui.perfetto.dev> 載入，能看到每個運算在時間軸上的位置。

## 重點整理

- `with profile(activities=[ProfilerActivity.CPU]) as prof:` 記錄區塊裡的所有運算。
- `record_function("名字")` 標出自己的段落；`prof.key_averages().table(sort_by=...)` 印出統計表。
- Self CPU 是運算本身的時間，依它排序找最慢的運算。
- 訓練一步：矩陣乘法最多、backward 約 forward 的兩倍；softmax 類的運算 FLOPs 少但不便宜。
