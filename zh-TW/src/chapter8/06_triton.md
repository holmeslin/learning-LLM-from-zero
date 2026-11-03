# Triton 一瞥（選讀：需 GPU）

## 本集目標

知道 Triton 是什麼、一個 Triton kernel 長什麼樣子，以及第 4 集的分塊 forward 要怎麼變成真正快的 kernel。這一集需要 NVIDIA 顯示卡才能執行，Mac 上只能讀。

## 正文

### 為什麼要自己寫 kernel

第 4 集的純 PyTorch 版本慢，是因為每一行運算都是一個獨立的 kernel：算一小塊分數、寫回記憶體；`exp`、寫回；`@ v`、寫回……。要讓一塊 query 的 `m`、`l`、`acc` 從頭到尾待在晶片上，整個內層迴圈必須寫在**同一個** kernel 裡。PyTorch 的運算做不到這件事，`torch.compile`（第 7 章第 7 集）也不一定能把這麼複雜的迴圈融合好。

NVIDIA 顯示卡的 kernel 傳統上用 CUDA C++ 寫，要自己管每個執行緒、每塊共享記憶體，很難寫。**Triton** 是 OpenAI 開源的語言，用 Python 語法寫 GPU kernel，以「一整塊資料」為單位思考，執行緒的細節交給編譯器。`torch.compile` 在 NVIDIA 顯示卡上產生的 kernel 就是 Triton。

### 一個 Triton kernel

用第 4 章的 RMSNorm 當例子：每一列（最後一個維度）各自算均方根再相除。Triton 的寫法是讓每個 **program**（一個 kernel 的執行單位，可以想成一小群一起工作的執行緒）處理一列：

```python,norun
import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)  # 我是第幾個 program，就處理第幾列
    cols = tl.arange(0, BLOCK)  # 0, 1, ..., BLOCK-1
    mask = cols < n_cols  # BLOCK 是 2 的次方，可能比一列長，多出來的不讀不寫
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)  # 整列讀進晶片
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    rms = tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    tl.store(out_ptr + row * n_cols + cols, x / rms * w, mask=mask)  # 只寫一次


def rmsnorm(x, weight, eps=1e-5):
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    out = torch.empty_like(x2)
    n_rows, n_cols = x2.shape
    rmsnorm_kernel[(n_rows,)](x2, weight, out, n_cols, eps, BLOCK=triton.next_power_of_2(n_cols))
    return out.reshape(x.shape)
```

本書的環境沒有 NVIDIA 顯示卡，這段程式沒有經過本書的測試流程執行；`triton` 也不在本書的相依套件裡。在有 NVIDIA 顯示卡的 Linux 上，用 `uv add triton` 安裝後，可以拿第 4 章的 `RMSNorm` 比對結果。

幾個要點：

- `@triton.jit` 標記這是 kernel。參數裡的張量會變成指向第一個元素的**指標**（`x_ptr`），要自己用「起點 + 位移」算位置。
- `tl.load`／`tl.store` 一次讀寫一整塊，`mask` 擋掉超出範圍的位置。讀進來的 `x` 在晶片上，平方、加總、除、乘都不會寫回記憶體。
- `kernel[(n_rows,)](...)` 是啟動 kernel：方括號裡是要開幾個 program，這裡一列一個。
- `BLOCK: tl.constexpr` 是編譯時就確定的常數，Triton 會為不同的值各編一個版本。

和第 7 章第 7 集的 eager 寫法比，這個 kernel 每個元素只讀一次、寫一次，就是 `torch.compile` 融合後的效果，只是這次是手寫的。

### FlashAttention 的 Triton 版本長什麼樣

把第 4 集的 `flash_attention_forward` 對照過來：

- 外層迴圈 `for i in range(0, T, block_q)` 不見了，變成「開 `T / block_q` × batch × head 個 program」，每個 program 處理一塊 query。
- program 一開始用 `tl.load` 把自己那塊 `qi` 讀進晶片，建好 `m`、`l`、`acc`。
- 內層迴圈 `for j in range(...)` 原封不動留在 kernel 裡：每一圈 `tl.load` 一塊 `kj`、`vj`，用 `tl.dot` 做小矩陣乘法，更新 `m`、`l`、`acc`。
- 迴圈結束，`tl.store` 寫出這塊的 `o` 和 `lse`。

整個 \\( T \times T \\) 的分數一個位元組都沒寫到記憶體，`q`、`o` 各讀寫一次，`k`、`v` 被每個 query 塊讀一次。這正是 FlashAttention 快的原因。有 GPU 的話，很值得照第 4、5 集的程式自己用 Triton 寫一次。

## 重點整理

- 把整個內層迴圈寫在同一個 kernel 裡，`m`、`l`、`acc` 才能留在晶片上；PyTorch 的運算層級做不到。
- Triton 用 Python 語法寫 GPU kernel，以一塊資料為單位：`tl.program_id`、`tl.arange`、`tl.load`／`tl.store` 加 `mask`、`kernel[grid](...)`。
- FlashAttention 的 Triton 版：一個 program 處理一塊 query，內層迴圈讀 key／value 塊，只寫出 `o` 與 `lse`。
- Triton 在 NVIDIA、AMD 等顯示卡上執行，Mac 上沒有。
