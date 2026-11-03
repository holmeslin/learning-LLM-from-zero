# 通訊與計算的估算

## 本集目標

學會估算 data parallel 每一步的通訊時間與計算時間，判斷通訊會不會成為瓶頸，並實測 Gloo 在本機的 all-reduce 速度。

## 正文

### 實測本機的 all-reduce

量 25 MB 的 all-reduce，比較 Gloo 內建的版本和第 3 集自己寫的 ring all-reduce：

```python
import time

import torch
import torch.distributed as dist

from llmz.dist import launch, ring_all_reduce


def bench(rank, world_size):
    x = torch.randn(25_000_000 // 4)  # 25 MB
    for name, fn in [("dist.all_reduce", dist.all_reduce), ("ring_all_reduce", ring_all_reduce)]:
        fn(x)  # warmup
        dist.barrier()
        start = time.perf_counter()
        for _ in range(5):
            fn(x)
        dist.barrier()
        seconds = (time.perf_counter() - start) / 5
        if rank == 0:
            print(f"{name:16} {seconds * 1000:6.1f} ms  {x.numel() * 4 / seconds / 1e9:.2f} GB/s")


if __name__ == "__main__":
    launch(bench, 2)
```

執行結果（某一次）：

```text
dist.all_reduce    45.7 ms  0.55 GB/s
ring_all_reduce    31.9 ms  0.78 GB/s
```

這裡的 GB/s 是「資料大小 ÷ 時間」，叫 **algorithm bandwidth**。我們在 M4 Max 上另外量過 90.8 MB（標準模型梯度的大小）的 all-reduce：Gloo 內建 163～213 ms，自己寫的 ring 140～147 ms，約 0.5～0.65 GB/s。這比第 7 章量到的記憶體頻寬 279 GB/s 慢了幾百倍：Gloo 是走 TCP 網路協定在 process 之間搬資料，即使在同一台機器上。

### 通訊時間

第 3 集算過，ring all-reduce 每個 rank 送出 \\( 2(N-1)/N \cdot S \\) 位元組。梯度大小是 \\( S = \Psi \times \\) 每個數的位元組數。卡數一多，\\( 2(N-1)/N \\) 接近 2，所以

\\[ t_{\text{comm}} \approx \frac{2 \cdot S}{\text{頻寬}} \\]

和卡數幾乎無關。

### 計算時間

每個 rank 一步看 \\( B \cdot T \\) 個 token（自己那一份），計算量是 \\( 6 \Psi B T \\) FLOPs（第 2 章第 7 集）：

\\[ t_{\text{compute}} = \frac{6 \Psi B T}{\text{每秒 FLOPs}} \\]

### 誰是瓶頸

兩者相除，\\( \Psi \\) 消掉了（梯度用 bfloat16，每個數 2 位元組）：

\\[ \frac{t_{\text{compute}}}{t_{\text{comm}}} = \frac{6 \Psi B T / F}{4 \Psi / \text{BW}} = \frac{1.5\, B T \cdot \text{BW}}{F} \\]

要讓計算時間大於通訊時間（第 5 集的重疊才藏得住通訊），每個 rank 每步至少要看

\\[ B T > \frac{F}{1.5\, \text{BW}} \\]

個 token。代入幾組數字，\\( F \\) 用第 1 集的假設：H100 的 990 TFLOPs 用到 40%：

```python
flops = 0.4 * 990e12
for name, bandwidth in [("400 GB/s", 400e9), ("50 GB/s", 50e9), ("5 GB/s", 5e9)]:
    print(f"頻寬 {name:9}：每個 rank 每步至少 {flops / (1.5 * bandwidth):9,.0f} 個 token")
```

執行結果：

```text
頻寬 400 GB/s ：每個 rank 每步至少       660 個 token
頻寬 50 GB/s  ：每個 rank 每步至少     5,280 個 token
頻寬 5 GB/s   ：每個 rank 每步至少    52,800 個 token
```

頻寬高時，每個 rank 一步看幾百個 token 就能把通訊藏在計算後面；頻寬只有 5 GB/s 時要五萬多個 token。這就是為什麼：

- 同一台機器內的卡用專門的高速連線（NVIDIA 叫 NVLink），機器之間用專門的高速網路。
- 卡越快（\\( F \\) 越大），對頻寬的要求越高；只換更快的卡、不換網路，效益會被通訊吃掉。
- data parallel 的每個 rank 不能分到太小的 batch，否則通訊會蓋過計算。這也限制了能用的總卡數：總 batch 大小有上限（第 6 章第 10 集看過 batch 太大的效果）。

### 算一下本機

本機的 Gloo 約 0.5 GB/s，標準模型的梯度是 float32 的 90.8 MB，一次 all-reduce 約 0.16～0.21 秒（上面的實測）；而 MPS 上加 `--compile` 一步只要 0.28 秒（第 5 章第 9 集）。如果真的用兩個 process 在一台 Mac 上做 data parallel，通訊就佔掉了大半時間，而且兩個 process 搶同一顆 GPU，不會比較快。CPU 模擬只是為了學會寫法；真正的多卡加速要有多張卡與快速的連線。

## 重點整理

- ring all-reduce 的通訊時間約 \\( 2S / \text{BW} \\)，和卡數幾乎無關。
- 計算時間 \\( 6\Psi BT / F \\)；兩者的比值是 \\( 1.5\, BT \cdot \text{BW} / F \\)，和模型大小無關。
- 每個 rank 每步的 token 數要大於 \\( F / (1.5\,\text{BW}) \\)，通訊才藏得住；卡越快越需要高頻寬。
- 本機 Gloo 實測約 0.5～0.65 GB/s，比記憶體頻寬慢幾百倍，CPU 模擬只用來學寫法。

## 自我挑戰

這一章放進 `src/llmz/` 的是 `src/llmz/dist.py` 的 `launch`、`print_in_order`、`ring_all_reduce`、`broadcast_parameters`、`all_reduce_gradients`、`BucketedDDP`、`ShardedOptimizer`。保留 `launch` 與 `print_in_order`，把其他函式與方法的本體換成 `raise NotImplementedError`，執行：

```bash
uv run pytest tests/test_dist.py
```

自己寫回來讓測試通過。建議順序：`ring_all_reduce`（先用 2 個 rank 想清楚每一步送哪一塊）、naive DDP、`ShardedOptimizer`、最後 `BucketedDDP`。

再試一題：把 `ShardedOptimizer` 改成真正的 ZeRO 第一階段：用 `dist.reduce_scatter_tensor` 取代梯度的 all-reduce、用 `dist.all_gather_into_tensor` 取代參數的 broadcast（提示：先把所有參數攤平接成一個長張量，再平均切給每個 rank）。
