# checkpoint

## 本集目標

寫出 `save_checkpoint` 與 `load_checkpoint`：模型參數、優化器狀態、目前步數一起存檔，訓練中斷後能從原地接著跑。

## 正文

### 只存模型不夠

第 2 章第 6 集存的是 `model.state_dict()`。要**接著練**，還需要兩樣：AdamW 的 \\( m \\)、\\( v \\)（第 3 集）和目前是第幾步（第 4 集的學習率排程要靠它）。少了優化器狀態，接續時相當於重新 warmup；少了步數，學習率會從頭來。

```python,ignore
{{#include ../../../src/llmz/train.py:checkpoint}}
```

三樣東西包成一個字典，`torch.save` 一次存。`optimizer.state_dict()` 是第 2 集說的基底類別功能，把 `state` 與 `param_groups` 一起打包。`map_location="cpu"` 讓在 GPU 上存的檔在任何機器都載得回來，載回之後模型 `.to(device)` 即可。

```python
import torch

from llmz.model import ModelConfig, TransformerLM
from llmz.optim import AdamW
from llmz.train import load_checkpoint, save_checkpoint

cfg = ModelConfig(vocab_size=50, context_length=16, d_model=32, n_layers=1, n_heads=4, d_ff=64)
torch.manual_seed(0)
model = TransformerLM(cfg)
opt = AdamW(model.parameters(), lr=1e-3)
x = torch.randint(0, 50, (2, 8))
model(x).sum().backward()
opt.step()
opt.zero_grad()
save_checkpoint(model, opt, step=123, path="ckpt.pt")

model2 = TransformerLM(cfg)
opt2 = AdamW(model2.parameters(), lr=1e-3)
step = load_checkpoint("ckpt.pt", model2, opt2)
print(step)
print(torch.allclose(model(x), model2(x)))
print(opt2.state[model2.embed.weight]["t"])
```

執行結果：

```text
123
True
1
```

載回來的模型對同樣輸入算出同樣結果，優化器也記得 embedding 已經更新過 1 步。

### 什麼時候存

第 8 集的訓練迴圈每 1000 步存一次到 `runs/<名稱>/latest.pt`（覆蓋舊的），結束時另外把純模型存成 `model.pt` 給第 6 章生成用。重新執行同樣的指令時，迴圈會先看 `latest.pt` 在不在，在就載回來從那一步接著跑。

標準設定的 checkpoint 大小：參數 2270 萬 × 4 位元組，加上 \\( m \\)、\\( v \\) 各一份，約 270 MB。

## 重點整理

- checkpoint = 模型 `state_dict` + 優化器 `state_dict` + 步數，一個字典 `torch.save`。
- `load_checkpoint(path, model, optimizer)` 回傳步數；`map_location="cpu"` 讓跨裝置載入沒問題。
- 每 1000 步覆蓋 `latest.pt`，結束存 `model.pt`；重跑同指令自動接續。
