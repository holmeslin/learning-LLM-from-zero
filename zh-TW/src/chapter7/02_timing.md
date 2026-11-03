# 正確計時（同步、warmup）

## 本集目標

寫出一個量 GPU 程式不會騙人的計時函式：先暖身、前後同步、重複多次取平均。

## 正文

### GPU 是非同步的

在 MPS 或 CUDA 上執行 `c = a @ b`，Python 只是把「做矩陣乘法」這個工作丟進 GPU 的佇列就回來了，GPU 在背後慢慢算。直接用 `time.perf_counter()` 量，量到的只是「丟工作」的時間。

```python,norun
import time

import torch

device = torch.device("mps")
a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)
for _ in range(3):
    a @ b
torch.mps.synchronize()

start = time.perf_counter()
c = a @ b
t_no_sync = time.perf_counter() - start
torch.mps.synchronize()
t_sync = time.perf_counter() - start
print(f"沒同步 {t_no_sync * 1000:.2f} ms，同步後 {t_sync * 1000:.2f} ms")
```

M4 Max 實測：

```text
沒同步 0.25 ms，同步後 14.24 ms
```

差了 50 倍。`torch.mps.synchronize()`（CUDA 是 `torch.cuda.synchronize()`）會等佇列裡的工作全部做完才回來，所以計時的起點和終點都要同步。CPU 本來就是一行做完才做下一行，不用同步。

我們把它包成一個不分 device 的函式，放進 `src/llmz/profile.py`：

```python,ignore
{{#include ../../../src/llmz/profile.py:sync}}
```

### 第一次特別慢

同一個 2048 × 2048 矩陣乘法連跑四次，每次都同步（M4 Max 實測）：

```text
第 0 次 15.68 ms
第 1 次 9.62 ms
第 2 次 8.84 ms
第 3 次 9.64 ms
```

第一次多花了六毫秒：GPU 要先準備這個形狀的 kernel（在 GPU 上跑的一小段程式）、配置記憶體。這些只發生一次，訓練五千步時根本看不到，所以計時前要先跑幾次不算，叫 **warmup**（暖身）。

### 重複多次取平均

單次的時間會受到時脈、其他程式的影響而跳動，量太短也會被誤差淹沒。用 `torch.compile` 過的 RMSNorm（第 7 集）在 M4 Max 上實測，每次只跑 50 遍時三次結果是 325、117、180 µs；改成跑 500 遍，三次是 96、89、85 µs，穩定多了。

把這三件事放在一起：

```python,ignore
{{#include ../../../src/llmz/profile.py:timeit}}
```

`fn` 是一個不帶參數的函式，用 `lambda` 包住要量的程式碼就行。注意只在迴圈前後同步，中間不同步：我們要量的是 GPU 連續工作的速度，每次都同步會把「丟工作」的空檔也算進去。

```python
import torch

from llmz.profile import timeit

device = torch.device("cpu")
a = torch.randn(512, 512)
b = torch.randn(512, 512)
seconds = timeit(lambda: a @ b, device, warmup=3, repeat=20)
print(f"{seconds * 1000:.3f} ms")
print(f"{2 * 512**3 / seconds / 1e9:.1f} GFLOPs")
```

執行結果（某一次）：

```text
0.276 ms
971.0 GFLOPs
```

第二行用第 2 章第 7 集的公式：矩陣乘法 \\( (m, k) @ (k, n) \\) 是 \\( 2mkn \\) 個 FLOPs，除以時間就是每秒做了幾次運算。GFLOPs 是每秒十億次。

### 量訓練的一步

訓練一步的 `fn` 就是第 2 章的五步：

```python,norun
def one_step():
    loss = cross_entropy(model(x[:, :-1]).reshape(-1, vocab), x[:, 1:].reshape(-1))
    loss.backward()
    opt.step()
    opt.zero_grad()

print(timeit(one_step, device, warmup=3, repeat=10))
```

本章後面的「每步幾毫秒」都是這樣量的：標準模型、batch 32、長度 256。每次量的結果會差個百分之五到十，例如同樣的 fp32 設定在不同時間量到 442、477、494 ms，所以我們只比較同一次執行裡的數字。

## 重點整理

- GPU 是非同步的，計時前後都要 `sync(device)`，否則量到的只是丟工作的時間。
- 第一次執行特別慢，先 warmup 幾次不計時。
- 重複多次取平均，量太少會跳動。
- `llmz.profile.timeit(fn, device, warmup, repeat)` 把這三件事包起來，回傳每次的秒數。
