# 量化初探

## 本集目標

把權重從 32 位元的浮點數存成 8、4 甚至更少位元的整數，量出記憶體省了多少、loss 變差多少，並知道為什麼要「分組」。

## 正文

### 為什麼要量化

第 3 集算過，decode 一步至少要把所有權重讀一遍。權重的位元組少一半，下限就少一半，能放進記憶體的模型也大一倍。**量化**（quantization）把權重用更少的位元表示：float32 是 4 個位元組，int8 是 1 個，int4 只要半個。

### 對稱量化

最簡單的做法：一組權重找出最大的絕對值 \\( m \\)，令 \\( s = m / q_{\max} \\)，每個權重除以 \\( s \\) 再四捨五入成整數：

\\[
q = \text{round}(w / s), \quad \hat{w} = q \cdot s
\\]

8 位元時 \\( q_{\max} = 127 \\)，整數落在 \\(-127\\) 到 \\(127\\)；4 位元時 \\( q_{\max} = 7 \\)。存的時候只要存整數 \\( q \\) 和每組一個縮放 \\( s \\)，用的時候乘回去（**反量化**，dequantize）。四捨五入的誤差最多半格，也就是 \\( s/2 \\)。

```python,ignore
{{#include ../../../src/llmz/inference.py:quantize}}
```

- `group_size` 決定「一組」多大。預設 `None` 是整列一組（per-channel，每個輸出通道一個縮放）；給 32 就是每 32 個權重一組。分組後形狀變成 `(out, in / g, g)`，`amax(dim=-1, keepdim=True)` 對每一組取最大的絕對值。
- `clamp(min=1e-12)` 防止整組都是 0 時除以 0。
- `quantize_model_` 是**模擬量化**：把每個 `Linear` 的權重換成量化再還原的值，模型照常用 float32 計算，所以 loss 反映量化的誤差，但不會變快。真正省記憶體與時間，要把整數存起來、在矩陣乘法的 kernel 裡一邊讀一邊反量化，那是 llama.cpp、GPTQ、AWQ 這類工具做的事。函式名稱結尾的底線和 `add_` 一樣，表示就地修改模型。

### 離群值與分組

權重裡偶爾會有特別大的值。用一個人造的例子看它的影響：

```python
import torch

from llmz.inference import dequantize, quantize

torch.manual_seed(0)
w = torch.randn(64, 256) * 0.02  # 大小和訓練好的權重差不多
w[0, 0] = 1.0  # 第 0 列放一個離群值
q, scale = quantize(w, 8)
print(q.dtype, tuple(q.shape), tuple(scale.shape))
print([round(x, 4) for x in w[1, :4].tolist()], "→", q[1, 0, :4].tolist())

for bits, group in [(8, None), (4, None), (4, 32)]:
    q, scale = quantize(w, bits, group)
    err = (dequantize(q, scale) - w).abs()
    print(f"{bits} bit，group={group}：第 0 列平均誤差 {err[0].mean():.5f}，其他列 {err[1:].mean():.5f}")
```

執行結果：

```text
torch.int8 (64, 1, 256) (64, 1, 1)
[-0.0008, -0.016, -0.0099, -0.0072] → [-1, -25, -15, -11]
8 bit，group=None：第 0 列平均誤差 0.00196，其他列 0.00012
4 bit，group=None：第 0 列平均誤差 0.01479，其他列 0.00219
4 bit，group=32：第 0 列平均誤差 0.00339，其他列 0.00162
```

第 0 列的縮放被那個 1.0 撐大：4 位元時一格是 \\( 1.0 / 7 \approx 0.14 \\)，而一般的權重只有 0.02 左右，幾乎全部被四捨五入成 0，平均誤差 0.0148，和權重本身差不多大。每 32 個一組時，離群值只影響它自己那一組，第 0 列的誤差降到 0.0034。代價是每組多存一個縮放：float32 的縮放每 32 個權重 4 個位元組，平均每個權重多 1 位元。

### 量化訓練好的模型

對標準模型的所有 `Linear`（1757 萬個參數，float32 是 70.3 MB）做模擬量化，在驗證集上用同樣的 100 批算 loss。embedding 表只是查表，不做矩陣乘法，這裡沒有量化。

```python,norun
import torch

from llmz.data import load_tokens
from llmz.device import get_device
from llmz.inference import quantize_model_
from llmz.model import ModelConfig, TransformerLM
from llmz.train import TrainConfig, evaluate

device = get_device()
cfg = TrainConfig(eval_batches=100)
valid = load_tokens(cfg.valid_path)

for bits, group in [(None, None), (8, None), (4, None), (4, 64), (4, 32), (3, None), (3, 32), (2, 32)]:
    model = TransformerLM(ModelConfig()).to(device)
    model.load_state_dict(torch.load("runs/base/model.pt", map_location=device))
    nbytes = 70_287_360 if bits is None else quantize_model_(model, bits, group)
    torch.manual_seed(0)  # 每次抽同樣的驗證 batch
    print(f"bits={bits} group={group}：{nbytes / 1e6:.2f} MB，val loss {evaluate(model, valid, cfg, device):.4f}")
```

| 位元 | 分組 | `Linear` 權重 | val loss |
| --- | --- | --- | --- |
| 32（float） | | 70.29 MB | 1.6917 |
| 8 | 整列 | 17.70 MB | 1.6918 |
| 4 | 整列 | 8.91 MB | 1.7187 |
| 4 | 64 | 9.88 MB | 1.7089 |
| 4 | 32 | 10.98 MB | 1.7056 |
| 3 | 整列 | 6.71 MB | 1.8728 |
| 3 | 32 | 8.79 MB | 1.7747 |
| 2 | 32 | 6.59 MB | 3.1316 |

- **8 位元**：記憶體變成四分之一，loss 只差 0.0001，幾乎無損。
- **4 位元**：整列一組差了 0.027；分組越細越好，32 個一組只差 0.014。
- **3 位元**：整列一組差了 0.18，分組後 0.08，開始明顯變差。
- **2 位元**：只剩 \\(-1, 0, 1\\) 三種值，loss 爆到 3.13，模型壞掉了。

int8 幾乎是免費的，4 位元分組是目前在個人電腦上跑大模型最常見的選擇。再往下就要更聰明的方法，例如一邊量化一邊調整還沒量化的權重，抵銷前面四捨五入造成的輸出誤差（GPTQ），或依照 activation 的大小找出重要的通道，先把它們放大再量化，讓它們的相對誤差變小（AWQ）。

只量化權重（weight-only）是最容易的一種。把 activation 也量化，矩陣乘法才能整個用整數算，但 activation 的離群值比權重嚴重得多，要更多技巧；本書不談。

## 重點整理

- 量化用更少位元存權重；decode 是 memory-bound，位元組少就快，也放得下更大的模型。
- 對稱量化：每組用最大絕對值算縮放 \\( s = m / q_{\max} \\)，\\( q = \text{round}(w/s) \\)，誤差最多 \\( s/2 \\)。
- `llmz.inference.quantize(w, bits, group_size)` 回傳 int8 整數與縮放；`dequantize` 乘回去；`quantize_model_` 做模擬量化並回傳實際儲存要的位元組。
- 一個離群值會撐大整組的縮放，分組讓它只影響自己那一組，代價是多存縮放。
- 標準模型：int8 的 loss 只差 0.0001；int4 每 32 個一組差 0.014；2 位元就壞了。
