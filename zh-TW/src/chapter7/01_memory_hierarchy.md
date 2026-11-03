# 記憶體階層（GPU 與 Apple Silicon）

## 本集目標

知道 GPU 的資料放在哪幾層記憶體、每一層多快，以及為什麼「搬資料」常常比「算」更花時間。

## 正文

### 算得快，搬得慢

GPU 有成千上萬個計算單元，每秒能做十幾兆次浮點運算。但每次運算的輸入都得先從記憶體搬進來，算完再寫回去。如果搬的速度跟不上，計算單元就只能閒著等。

記憶體分好幾層，越靠近計算單元越快、越小：

| 層 | 大小（大約） | 特性 |
| --- | --- | --- |
| 暫存器 | 每個計算單元幾 KB | 最快，程式碼控制不到 |
| 晶片上的 SRAM（快取、shared memory） | 每組計算單元幾百 KB | 快，但小 |
| 顯示卡記憶體（NVIDIA 叫 HBM） | 幾十 GB | 張量平常放的地方，頻寬每秒幾 TB |
| CPU 主記憶體 | 幾十到幾百 GB | 要經過 PCIe 傳到顯示卡，最慢 |

我們寫 `x = torch.randn(..., device="cuda")` 時，`x` 放在顯示卡記憶體；每做一個運算，GPU 就把需要的部分讀進晶片、算完、寫回去。

### Apple Silicon：一整塊共用

Mac 沒有獨立顯示卡。CPU 和 GPU 共用同一塊記憶體（Apple 叫 unified memory），所以上表最後一層「CPU 傳到顯示卡」不存在，`x.to("mps")` 不用經過 PCIe。但 GPU 讀寫這塊記憶體還是有頻寬上限：36 GB 版本的 M4 Max，Apple 規格標示 410 GB/s。

我們實際量一下（M4 Max 實測，第 2 集會教怎麼量才準）：

| 量什麼 | 結果 |
| --- | --- |
| 複製一個 256 MB 的張量（讀一次、寫一次） | 279 GB/s |
| 4096 × 4096 的 float32 矩陣乘法 | 10.3 TFLOPs |

TFLOPs 是每秒一兆次浮點運算。把兩個數字放在一起看：一秒能搬 279 GB，也就是大約 700 億個 float32；同一秒能算 10 兆次。**搬一個數的時間，夠算 150 次。**

```python
flops_per_second = 10.3e12
bytes_per_second = 279e9
floats_per_second = bytes_per_second / 4  # 一個 float32 佔 4 位元組
print(round(flops_per_second / floats_per_second))
```

執行結果：

```text
148
```

### 這對我們的模型代表什麼

如果一個運算對每個讀進來的數只做一兩次計算（例如兩個張量相加、RMSNorm 裡的平方），計算單元大部分時間在等資料，這種運算叫**卡在記憶體**（memory-bound）。矩陣乘法每個數會被重複用很多次，比較有機會**卡在計算**（compute-bound）。第 8 集會把這個比例正式算出來。

NVIDIA 的資料中心顯示卡也是一樣的故事，只是兩個數字都大得多：H100 的規格表寫 HBM 3.35 TB/s、bfloat16 矩陣運算約 990 TFLOPs，比例更懸殊。所以 FlashAttention（第 8 章）這類技術的核心都是「少搬資料」。

### 張量有多大

要估算搬多少，先要會算張量的位元組數：元素個數乘上每個元素的大小。

```python
import torch

x = torch.zeros(32, 256, 512)
print(x.numel(), x.element_size())
print(x.numel() * x.element_size() / 1e6, "MB")

# 標準模型一層 attention 的分數矩陣：batch 32、16 個 head、256 × 256
scores_bytes = 32 * 16 * 256 * 256 * 4
print(scores_bytes / 1e6, "MB")
```

執行結果：

```text
4194304 4
16.777216 MB
134.217728 MB
```

`(32, 256, 512)` 是標準模型每一層進出的形狀，16.8 MB；attention 分數矩陣卻有 134 MB，是它的 8 倍，而且跟序列長度的平方成正比。第 4 集與第 8 章會一再回到這個數字。

## 重點整理

- 記憶體由快到慢：暫存器、晶片上的 SRAM、顯示卡記憶體、CPU 主記憶體；越快越小。
- Apple Silicon 的 CPU 與 GPU 共用記憶體；M4 Max 實測搬資料 279 GB/s、矩陣乘法 10.3 TFLOPs。
- 搬一個 float32 的時間夠算約 150 次，所以很多運算卡在記憶體而不是計算。
- 張量位元組數 = `numel() * element_size()`。
