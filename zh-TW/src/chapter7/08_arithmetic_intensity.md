# arithmetic intensity

## 本集目標

用「每搬一個位元組做幾次運算」判斷一個運算是卡在計算還是卡在記憶體，並解釋本章看到的幾個現象。

## 正文

### 定義

**arithmetic intensity**（運算強度）是一個運算的 FLOPs 除以它至少要搬的位元組數：

\\[ \text{intensity} = \frac{\text{FLOPs}}{\text{讀寫的位元組數}} \\]

機器也有一個對應的比例：每秒能算的 FLOPs 除以每秒能搬的位元組。用第 1 集 M4 Max 的實測：

```python
peak_flops = 10.34e12  # float32 矩陣乘法，第 5 集的表
bandwidth = 279e9  # 複製張量，第 1 集
print(f"{peak_flops / bandwidth:.0f} FLOPs/byte")
```

執行結果：

```text
37 FLOPs/byte
```

一個運算的 intensity 低於 37，搬資料的時間比計算長，它**卡在記憶體**；高於 37 才有機會**卡在計算**，用滿 GPU。

### 算幾個例子

兩個長度 \\( n \\) 的 float32 向量相加：\\( n \\) 次加法，讀兩個向量、寫一個，共 \\( 12n \\) 位元組。

兩個 \\( n \times n \\) 矩陣相乘：\\( 2n^3 \\) FLOPs（第 2 章第 7 集），讀兩個、寫一個矩陣，共 \\( 12n^2 \\) 位元組，intensity 是 \\( n/6 \\)，矩陣越大越高。

```python
def matmul_intensity(m, k, n, bytes_per=4):
    flops = 2 * m * k * n
    moved = (m * k + k * n + m * n) * bytes_per
    return flops / moved


print(f"向量相加 {1 / 12:.3f}")
for n in [256, 1024, 4096]:
    print(f"{n} x {n} 矩陣乘法 {matmul_intensity(n, n, n):.0f}")
# 標準模型的 SwiGLU 第一個 Linear：(32*256, 512) @ (512, 1344)
print(f"SwiGLU w1 {matmul_intensity(32 * 256, 512, 1344):.0f}")
# attention 的 Q @ K 轉置，一個 head：(256, 32) @ (32, 256)
print(f"Q @ K^T {matmul_intensity(256, 32, 256):.0f}")
```

執行結果：

```text
向量相加 0.083
256 x 256 矩陣乘法 43
1024 x 1024 矩陣乘法 171
4096 x 4096 矩陣乘法 683
SwiGLU w1 177
Q @ K^T 13
```

- 逐元素運算的 intensity 遠低於 1，永遠卡在記憶體。這就是第 3 集 profiler 裡 `mul`、`exp` 花那麼多時間、第 7 集 `torch.compile` 能讓它們快六倍的原因：融合不會減少計算，只會減少搬運。
- 模型裡的 Linear 都在 100 以上，卡在計算，換更快的演算法沒用，只能換更快的硬體或更低的精度。
- attention 的 \\( QK^\top \\) 只有 13：每個 head 只有 32 維，乘出來的 256 × 256 分數矩陣卻很大，寫出去比算它還花時間。後面的 softmax 再把它讀寫一遍。第 8 章 FlashAttention 就是要讓這個大矩陣不寫回記憶體。

### roofline

把兩件事合起來：一個運算能達到的速度最多是

\\[ \min(\text{peak FLOPs},\ \text{intensity} \times \text{頻寬}) \\]

畫成圖，橫軸 intensity、縱軸速度，是一條先斜上升、到 37 之後變平的線，形狀像屋頂，叫 **roofline 模型**。斜的部分卡在記憶體，平的部分卡在計算。

它是上限，不是預測。第 5 集量到 256 × 256 的矩陣乘法只有 1.23 TFLOPs：intensity 43 已經在屋頂平的那一段，照理能接近 10 TFLOPs。差距來自固定開銷，一個這麼小的 kernel 只要幾微秒就算完，丟工作給 GPU 的時間反而佔了大半。roofline 告訴你努力的方向（少搬資料，或讓運算夠大），實際還是要量。

### 本章回顧

| 手段 | 針對 | M4 Max 實測 |
| --- | --- | --- |
| bfloat16 autocast | 矩陣乘法的計算量、activations 大小 | 每步快 13%，記憶體少約 1 GB |
| activation checkpointing | activations 大小 | 每步慢 9%，記憶體少 2.15 GB |
| `torch.compile` | 逐元素運算的資料搬運 | 每步快 1.6 倍 |

## 重點整理

- arithmetic intensity = FLOPs ÷ 讀寫的位元組數；M4 Max 的分界點約 37 FLOPs/byte。
- 逐元素運算遠低於 1，卡在記憶體；大矩陣乘法 intensity 約 \\( n/6 \\)，卡在計算。
- attention 的 \\( QK^\top \\) 只有約 13，加上 softmax 的讀寫，是 FlashAttention 要解決的問題。
- roofline：可達速度 = min(peak, intensity × 頻寬)，是上限不是預測。

## 自我挑戰

這一章放進 `src/llmz/` 的是 `src/llmz/profile.py` 的 `sync`、`timeit`、`saved_activation_bytes`、`estimate_memory`，以及 `src/llmz/model.py` 裡 `TransformerLM.forward` 的 `activation_checkpoint` 分支。把它們的本體換成 `raise NotImplementedError`（`forward` 只要拿掉 checkpoint 分支），執行：

```bash
uv run pytest tests/test_profile.py
```

自己寫回來讓測試通過。

如果你有 NVIDIA 顯示卡：在 `TrainConfig` 加一個 `autocast` 開關，照第 5 集的寫法把 forward 和 loss 包進 `torch.autocast`，用 `timeit` 量 float32 與 bfloat16 每步的時間，和本章 M4 Max 的 13% 比一比。
