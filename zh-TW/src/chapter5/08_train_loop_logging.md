# 組裝訓練迴圈與 JSONL log

## 本集目標

把前面七集組成 `train(cfg)`：設定用 dataclass 集中、每一步做完整的流程、定期印 loss 與驗證 loss、存成 JSONL、存 checkpoint，並用 smoke 設定在 CPU 上幾秒鐘跑通。

## 正文

### 設定

```python,ignore
{{#include ../../../src/llmz/train.py:config}}
```

預設值就是第 10 集完整訓練的設定。`model` 欄位是第 4 章的 `ModelConfig`，`field(default_factory=...)` 是 dataclass 放另一個物件當預設值的寫法。`device` 空字串代表用第 1 章第 9 集的 `get_device()`。

### 驗證 loss

訓練時的 loss 是模型在**它正在學的那一批**上的表現，會比真實程度樂觀。另外準備一個模型沒練過的驗證集（第 3 章第 14 集的 `valid.bin`），定期抽幾批算 loss，這個數字才是我們真正關心的。

```python,ignore
{{#include ../../../src/llmz/train.py:evaluate}}
```

`model.eval()`／`model.train()` 切換模式：本書的模型沒有訓練與推論行為不同的層（例如 dropout），所以現在沒差，但這是慣例，先養成習慣。整個函式用 `@torch.no_grad()`，不記錄圖，省記憶體也快。

### 主迴圈

```python,ignore
{{#include ../../../src/llmz/train.py:train}}
```

逐段看：

1. 固定種子、決定 device、建立輸出目錄、用 `memmap` 打開兩個 token 檔。
2. 建模型搬到 device，建 AdamW；`latest.pt` 存在就載回來接續（上一集）。`cfg.compile` 為真時用 `torch.compile` 包住模型，下一集講。
3. 迴圈每一步：算學習率寫進 `param_groups`（第 4 集）→ `get_batch`（第 6 集）→ forward → `cross_entropy`（第 1 集）→ `backward()` → 裁剪（第 5 集）→ `step()`（第 3 集）→ `zero_grad()`。粗體那五個就是第 2 章的固定句型，一個都沒少。
4. 每 `log_every` 步記一筆：步數、loss、學習率、梯度長度、經過的秒數；每 `eval_every` 步多記驗證 loss。一筆一行 JSON 寫進 `log.jsonl`（JSON Lines：每行一個 JSON 物件，隨時可以 `tail`、也好畫圖），同時印到螢幕。
5. 每 `checkpoint_every` 步存 `latest.pt`；結束存 `model.pt` 與 `config.json`。

### smoke 設定

全書規則：每個會訓練的東西都要有幾秒鐘能在 CPU 跑完的版本。資料用「0、1、2、…、63、0、1、…」這種一看就有規律的序列，模型縮到幾萬個參數，60 步：

```python
import numpy as np

from llmz.model import ModelConfig
from llmz.train import TrainConfig, train

tokens = (np.arange(20000) % 64).astype(np.uint16)
tokens.tofile("train.bin")
tokens.tofile("valid.bin")
cfg = TrainConfig(
    train_path="train.bin", valid_path="valid.bin", out_dir="smoke",
    model=ModelConfig(vocab_size=64, context_length=16, d_model=32, n_layers=2, n_heads=4, d_ff=64),
    batch_size=8, max_steps=60, warmup_steps=5, lr_max=3e-3,
    log_every=20, eval_every=60, eval_batches=2, checkpoint_every=60, device="cpu",
)
records = train(cfg)
print(len(records), records[-1]["step"], records[-1]["val_loss"] < 1.0)
```

執行結果（某一次）：

```text
{'step': 20, 'loss': 1.9337, 'lr': 0.0024995480642206634, 'grad_norm': 0.944, 'time': 0.2}
{'step': 40, 'loss': 0.8336, 'lr': 0.0009476482311472649, 'grad_norm': 0.586, 'time': 0.3}
{'step': 60, 'loss': 0.6704, 'lr': 0.0001, 'grad_norm': 0.571, 'time': 0.4, 'val_loss': 0.7033}
3 60 True
```

60 步之後驗證 loss 從 \\( \ln 64 = 4.16 \\) 降到 1 以下：模型學會了「下一個是 +1」。你的 loss 數字會和書上略有不同（CPU 上的浮點加總順序會隨機器而異），但趨勢一樣。`tests/test_train.py` 跑的就是這個設定。

## 重點整理

- `TrainConfig` 集中所有訓練超參數，預設是標準設定；`train(cfg)` 回傳 log 紀錄。
- 每一步：排程改 lr → `get_batch` → forward → loss → `backward()` → 裁剪 → `step()` → `zero_grad()`。
- 驗證 loss 用模型沒練過的資料算，才是真實程度；`evaluate` 用 `no_grad` 與 `eval()`。
- log 用 JSONL（一行一個 JSON），每 1000 步存 `latest.pt`，重跑自動接續。
- smoke 設定：假資料、小模型、60 步、CPU 幾秒鐘。
