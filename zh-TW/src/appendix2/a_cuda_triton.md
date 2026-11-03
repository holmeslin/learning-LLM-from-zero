# CUDA 與 Triton 入門指引（選讀：需 GPU）

## 本集目標

有 NVIDIA 顯示卡的時候，知道 GPU 怎麼組織執行緒與記憶體、CUDA 與 Triton 的 kernel 各長什麼樣子、在 CUDA 上量時間與記憶體的工具，以及一條從本書走到自己寫 kernel 的學習路線。

## 正文

這一集的程式需要 NVIDIA 顯示卡與 CUDA，本書的環境沒有，所以沒有經過本書的測試流程執行，也沒有實測數字。租機器的方法見附錄一 e。

### GPU 的組織

第 7 章第 1 集的記憶體階層，在 NVIDIA 顯示卡上是這樣：

- **HBM**（顯示記憶體）：幾十 GB，所有張量平常放在這裡。資料中心等級的卡頻寬每秒好幾 TB，但相對於算力仍然是瓶頸。
- **SM**（streaming multiprocessor）：一張卡有一百多個 SM，每個 SM 有自己的運算單元、一小塊 **shared memory**（共享記憶體，每個 SM 一兩百 KB）與大量**暫存器**。shared memory 和暫存器就是第 8 章說的「晶片上」。

程式這一邊分三層：

- **thread**（執行緒）：最小的執行單位。每 32 個 thread 綁成一個 **warp**，同一個時間做同一個指令。
- **block**（執行緒區塊）：一群 thread（常見 128～1024 個），被排到同一個 SM 上，可以共用 shared memory、互相等待。
- **grid**：一個 kernel 啟動的所有 block。block 之間不能直接溝通，只能透過 HBM。

第 8 章 FlashAttention 的「一塊 query」，在 CUDA 裡就是一個 block；`m`、`l`、`acc` 放在暫存器與 shared memory，所以不必寫回 HBM。

### CUDA 的 kernel

最簡單的例子：兩個向量相加。CUDA C++ 的 kernel 是從**一個 thread** 的角度寫的：

```cuda
__global__ void add_kernel(const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 我是全部 thread 裡的第幾個
    if (i < n) out[i] = x[i] + y[i];                // 最後一個 block 可能超出範圍
}

torch::Tensor add(torch::Tensor x, torch::Tensor y) {
    auto out = torch::empty_like(x);
    int n = x.numel();
    int threads = 256;
    add_kernel<<<(n + threads - 1) / threads, threads>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
```

- `__global__` 表示這是在 GPU 上執行、由 CPU 啟動的 kernel。
- `blockIdx.x`、`blockDim.x`、`threadIdx.x` 是第幾個 block、每個 block 幾個 thread、block 裡的第幾個 thread。
- `<<<block 數, 每個 block 的 thread 數>>>` 啟動 kernel。

PyTorch 可以在執行時編譯這段程式，變成一個 Python 函式：

```python,norun
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = open("add.cu").read()  # 上面那段 CUDA 程式
ext = load_inline(
    name="add_ext",
    cpp_sources="torch::Tensor add(torch::Tensor x, torch::Tensor y);",
    cuda_sources=cuda_source,
    functions=["add"],
)
x = torch.randn(1_000_000, device="cuda")
y = torch.randn(1_000_000, device="cuda")
print(torch.allclose(ext.add(x, y), x + y))
```

`load_inline` 會呼叫 NVIDIA 的編譯器 `nvcc`，第一次要等一陣子。真正的 kernel（矩陣乘法、attention）還要自己把資料搬進 shared memory、安排每個 thread 讀哪些位置才不會互相衝突、使用 Tensor Core 這種專門做小矩陣乘法的單元，難度高很多。

### Triton 的 kernel

同樣的向量相加用 Triton 寫，是從**一個 program（一整塊資料）**的角度：

```python,norun
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)  # 這個 program 負責的一整塊位置
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    return out
```

CUDA 的 `if (i < n)` 變成 Triton 的 `mask`；一個 block 裡的 thread 怎麼分工，交給編譯器決定。第 8 章第 6 集的 RMSNorm kernel 是同樣的寫法。Triton 的取捨是：容易寫、通常能達到手寫 CUDA 的大部分速度，但無法控制最底層的細節。

### 在 CUDA 上量時間與記憶體

本書的工具在 CUDA 上照樣能用：`get_device()` 會選 `cuda`，`llmz.profile.sync` 會呼叫 `torch.cuda.synchronize()`，`timeit` 與 `torch.profiler` 都不用改。CUDA 另外有幾個 MPS 沒有的工具：

```python,norun
import torch

# 只量 GPU 上的時間：在 GPU 的佇列裡插兩個時間戳記
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
y = torch.randn(4096, 4096, device="cuda") @ torch.randn(4096, 4096, device="cuda")
end.record()
torch.cuda.synchronize()
print(f"{start.elapsed_time(end):.2f} ms")

# 記憶體的最高點
torch.cuda.reset_peak_memory_stats()
y = torch.randn(4096, 4096, device="cuda") @ torch.randn(4096, 4096, device="cuda")
print(f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")

# 記錄每一次配置與釋放，存成檔案後拿到 https://pytorch.org/memory_viz 看
torch.cuda.memory._record_memory_history()
y = torch.randn(4096, 4096, device="cuda") @ torch.randn(4096, 4096, device="cuda")
torch.cuda.memory._dump_snapshot("memory.pickle")
```

- `torch.cuda.Event` 量的是 GPU 執行的時間，不含 Python 的開銷。
- `max_memory_allocated` 是第 7 章第 4 集估算的訓練記憶體最直接的驗證方法。
- memory snapshot 會畫出每個張量什麼時候配置、什麼時候釋放，找記憶體最高點時很好用。

命令列工具：`nsys`（Nsight Systems）看整個程式的時間軸，搭配 `torch.cuda.nvtx.range("名稱")` 在時間軸上標出區段；`ncu`（Nsight Compute）深入分析單一 kernel 的頻寬與算力使用率。另外，設定環境變數 `TORCH_LOGS=output_code` 執行用了 `torch.compile` 的程式，可以看到它產生的 Triton kernel，這是學 Triton 很好的範例來源。

### 學習路線

1. 照附錄一 e 把本書的程式搬到 CUDA 上，跑 `uv run pytest`，再用 `timeit` 比較第 8 章的 `flash_attention`、第 4 章的 attention 與 `F.scaled_dot_product_attention`。
2. 照 Triton 官方教學（<https://triton-lang.org> 的 Tutorials）依序做：向量相加、融合的 softmax、矩陣乘法、layer norm。每做一個就和 PyTorch 的結果比對、量時間。
3. 把第 8 章第 4 集的 `flash_attention_forward` 改寫成 Triton kernel（第 8 章第 6 集描述了對應關係），用 `flash_attention_forward` 的輸出驗證，再挑戰 backward。
4. 做 CS336 Assignment 2（下一集的對照表）。
5. 想深入 CUDA C++：NVIDIA 的 CUDA C++ Programming Guide，以及《Programming Massively Parallel Processors》這本教科書。

## 重點整理

- NVIDIA 顯示卡：HBM 容量大、頻寬相對不足；每個 SM 有 shared memory 與暫存器，就是「晶片上」。
- 執行單位：32 個 thread 一個 warp，一群 thread 一個 block（排到同一個 SM），所有 block 組成 grid。
- CUDA kernel 從一個 thread 的角度寫，用 `blockIdx`、`threadIdx` 算位置；`torch.utils.cpp_extension.load_inline` 可以在 Python 裡編譯它。
- Triton kernel 從一整塊資料的角度寫，用 `mask` 處理邊界，thread 的分工交給編譯器。
- CUDA 上的工具：`torch.cuda.Event`、`max_memory_allocated`、memory snapshot、`nsys`、`ncu`、`TORCH_LOGS=output_code`。
