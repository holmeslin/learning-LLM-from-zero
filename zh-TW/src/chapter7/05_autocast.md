# 混合精度 autocast

## 本集目標

認識 float16 與 bfloat16 兩種半精度，學會用 `torch.autocast` 讓矩陣乘法用半精度算、參數維持 float32，並看它在 M4 Max 上省了多少。

## 正文

### 兩種 16 位元浮點數

float32 用 32 個位元存一個數：1 位元正負號、8 位元指數（決定能表示多大多小）、23 位元尾數（決定多精確）。16 位元的版本有兩種切法：

| dtype | 指數 | 尾數 | 最大值 | 1 附近的間隔 |
| --- | --- | --- | --- | --- |
| float32 | 8 | 23 | \\( 3.4 \times 10^{38} \\) | \\( 1.2 \times 10^{-7} \\) |
| float16 | 5 | 10 | 65504 | 0.00098 |
| bfloat16 | 8 | 7 | \\( 3.4 \times 10^{38} \\) | 0.0078 |

float16 精確一點、範圍小很多；bfloat16 的範圍和 float32 一樣，但只有 7 位元尾數，很粗。實際試試：

```python
import torch

for value in [1 + 1 / 256, 70000.0, 1e-8]:
    half = torch.tensor(value, dtype=torch.float16).item()
    brain = torch.tensor(value, dtype=torch.bfloat16).item()
    print(value, half, brain)
```

執行結果：

```text
1.00390625 1.00390625 1.0
70000.0 inf 70144.0
1e-08 0.0 1.0011717677116394e-08
```

- \\( 1 + 1/256 \\)：float16 存得下，bfloat16 直接變成 1。
- 70000：超過 float16 的最大值，變成 `inf`；bfloat16 存成 70144，不準但沒爆。
- \\( 10^{-8} \\)：float16 太小存不下，變成 0；bfloat16 沒問題。

訓練時梯度常常是很小的數，用 float16 會變成 0（叫 underflow），所以 float16 訓練要搭配「先把 loss 乘上一個大數、更新前再除回來」的技巧（PyTorch 的 `GradScaler`）。bfloat16 範圍夠大，不需要這招，這是它在 LLM 訓練裡成為主流的原因。粗糙的尾數靠下面的「混合」來補。

### 混合精度：哪裡用半精度

全部換成 bfloat16 會出問題：訓練後段學習率降到 1e-4 時，AdamW 每步對參數的更新量只有參數值的千分之幾，比 bfloat16 能分辨的相對差距（0.4%～0.8%）還小，加上去常常等於沒加。所以**混合精度**的做法是：

- 參數、梯度、優化器狀態維持 float32。
- forward 裡的矩陣乘法用半精度算（最花時間、也最不怕誤差）。
- 加總、softmax、norm 這類容易累積誤差的運算用 float32。

`torch.autocast` 自動做這件事。在它的 `with` 區塊裡，矩陣乘法會先把輸入轉成指定的 dtype 再算：

```python
import torch

from llmz.layers import Linear

torch.manual_seed(0)
linear = Linear(4, 3)
x = torch.randn(2, 4)
with torch.autocast("cpu", dtype=torch.bfloat16):
    y = linear(x)
    z = x + 1
print(linear.weight.dtype, y.dtype, z.dtype)
```

執行結果：

```text
torch.float32 torch.bfloat16 torch.float32
```

參數還是 float32，`@` 的輸出變成 bfloat16，加法沒被改。第 4 章的 `RMSNorm` 一開始就先 `x.float()` 再算平方和，就是為了這種情況。

訓練迴圈裡只要把 forward 和 loss 包進去，backward 放在外面：

```python,norun
with torch.autocast(device.type, dtype=torch.bfloat16):
    logits = model(x[:, :-1])
    loss = cross_entropy(logits.reshape(-1, vocab).float(), x[:, 1:].reshape(-1))
loss.backward()
opt.step()
opt.zero_grad()
```

logits 出來是 bfloat16，`.float()` 轉回 float32 再算 cross-entropy，log-sum-exp 才不會失準。

### M4 Max 實測

先看矩陣乘法本身，n × n 乘 n × n，每秒做幾兆次運算（TFLOPs）：

| dtype | n = 256 | n = 1024 | n = 4096 |
| --- | --- | --- | --- |
| float32 | 1.23 | 7.60 | 10.34 |
| bfloat16 | 1.44 | 7.29 | 11.41 |
| float16 | 0.90 | 7.61 | 7.87 |

大矩陣上 bfloat16 只比 float32 快一成，float16 反而更慢。再看標準模型訓練一步（batch 32、長度 256，同一個 batch 重複訓練 13 步後的 loss）：

| 設定 | 每步 | MPS 配置的記憶體 | loss |
| --- | --- | --- | --- |
| float32 | 477 ms | 6.55 GB | 2.315 |
| autocast bfloat16 | 414 ms | 5.62 GB | 2.315 |
| autocast float16（沒有 GradScaler） | 403 ms | 5.48 GB | 2.464 |

bfloat16 快了 13%、省了將近 1 GB，loss 和 float32 一樣。float16 沒做 loss scaling，loss 明顯變差。

### 為什麼 Mac 上效果有限

NVIDIA 從 V100 開始有專門算半精度矩陣乘法的電路（Tensor Core），半精度比 float32 快好幾倍，混合精度幾乎是必開的。M4 Max 的 GPU 沒有這麼大的落差，所以只快一成多。省下的記憶體來自 activations：存下來給 backward 用的矩陣乘法輸入與結果變成 bfloat16，只佔一半。用上一集的 `saved_activation_bytes` 在 CPU 上量 batch 4，從 376.8 MB 降到 276.1 MB，少了 27%；沒減半，是因為 softmax、RMSNorm 這些維持 float32 的部分沒變。

所以本書的 `tools/train.py` 沒有加 autocast 的選項：在 Mac 上多一個開關換 13%，不如第 7 集的 `torch.compile`。如果你在 NVIDIA 顯示卡上跑，就值得加，見本章的自我挑戰。

## 重點整理

- float16：精度較好、範圍小（最大 65504），訓練要搭配 loss scaling；bfloat16：範圍同 float32、精度粗，LLM 訓練的主流。
- 混合精度：參數、梯度、優化器狀態維持 float32，矩陣乘法用半精度。
- `with torch.autocast(device.type, dtype=torch.bfloat16):` 包住 forward 與 loss，backward 在外面；logits 先 `.float()` 再算 loss。
- M4 Max 實測：bfloat16 每步快 13%、記憶體少約 1 GB；NVIDIA 顯示卡上效果大得多。
