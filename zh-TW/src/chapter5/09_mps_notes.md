# MPS 注意事項（TF32、compile）

## 本集目標

知道在 Apple Silicon 上訓練和在 NVIDIA 顯示卡上有哪些不同：沒有 TF32、`torch.compile` 可用而且值得開、速度與記憶體的實測數字。

## 正文

### 三種 device 的速度

標準設定、batch 32 × 256，M4 Max（36 GB）實測每一步的時間：

| device | 每步 | 每秒 token | 5000 步 |
| --- | --- | --- | --- |
| CPU | 2420 ms | 3,400 | 3 小時 22 分 |
| MPS | 442 ms | 18,500 | 37 分 |
| MPS + `torch.compile` | 276 ms | 29,700 | 23 分 |

GPU 記憶體用了 5.5 GB，16 GB 的機器放得下。CPU 慢五到九倍，但同樣的程式一行不改就能跑，只是要等。較小的晶片（M1、M2 的非 Max 版本）GPU 核心數只有四分之一到一半，推估再慢兩到四倍，沒有實測。

### TF32

NVIDIA 顯示卡有一種叫 TF32 的矩陣乘法模式：把 float32 的尾數砍到 10 位元來換速度，PyTorch 上用 `torch.backends.cuda.matmul.allow_tf32 = True`（或 `torch.set_float32_matmul_precision("high")`）打開。MPS **沒有**這個東西，設了也沒效果；Apple GPU 的 float32 就是完整精度。真正要快要靠第 7 章的半精度（bfloat16），MPS 支援。

### `torch.compile`

`torch.compile(model)` 把模型的 Python 程式碼轉成融合過的 kernel：把連續的小運算（例如 RMSNorm 裡的平方、平均、開根號、除法）合成一個，少跑幾趟記憶體。PyTorch 2.9 的 MPS 後端支援它，上表看到快了 1.6 倍。代價是第一次 forward 要編譯幾秒鐘，模型形狀變了會重新編譯（本書 batch 和長度固定，不會）。

用法就是 `train.py` 裡那一行：`forward = torch.compile(model)`，之後用 `forward(x)` 代替 `model(x)`；參數還是 `model` 的，優化器與 checkpoint 不受影響。命令列加 `--compile` 就開。

```python,norun
import torch

from llmz.model import ModelConfig, TransformerLM

model = TransformerLM(ModelConfig()).to("mps")
fast = torch.compile(model)
x = torch.randint(0, 10000, (32, 256), device="mps")
print(torch.allclose(fast(x), model(x), atol=1e-4))
```

編譯後的結果和原本的只差浮點誤差。

### 其他差異

- 隨機數：`torch.manual_seed` 在 CPU 上保證每台機器一樣，MPS 上的 `randn` 不保證；本書的 `get_batch` 在 CPU 上抽起點再搬過去，所以每一步拿到的資料跨機器一致，但模型初始化在 CPU 做完才 `.to(device)`，也一致；只有運算的末位誤差會不同。
- 計時：GPU 是非同步的，`time.time()` 量到的可能只是「把工作丟出去」的時間，量之前要 `torch.mps.synchronize()`。第 7 章第 2 集專門講。
- 記憶體：Apple Silicon 的 CPU 與 GPU 共用同一塊記憶體，`x.to("mps")` 不是真的複製到另一張卡，但 PyTorch 還是會各管各的，省不了。

## 重點整理

- M4 Max 實測：CPU 2.4 秒一步、MPS 0.44 秒、MPS + compile 0.28 秒；標準訓練 23 到 37 分鐘。
- MPS 沒有 TF32；要快靠 bfloat16（第 7 章）。
- `torch.compile` 在 MPS 上可用，約 1.6 倍，命令列 `--compile`。
- 計時前 `torch.mps.synchronize()`；資料與初始化在 CPU 做，跨機器一致。
