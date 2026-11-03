# greedy decoding

## 本集目標

用訓練好的模型生成文字的最簡單方法：每次都選 logits 最高的那個 token，接在後面，再算一次。

## 正文

### 一次一個 token

語言模型只會做一件事：給一段 token，算出「下一個是誰」的分數。生成就是重複這件事：

1. 把目前的序列餵進模型，拿**最後一個位置**的 logits（形狀 `(V,)`）。
2. 從這 V 個分數挑一個 token。
3. 接到序列後面，回到 1。

直到碰到 `<|endoftext|>`、或長度到上限。這一集先用最簡單的挑法：**greedy**，永遠選最大的。

```python,ignore
{{#include ../../../src/llmz/generate.py:greedy_next}}
```

第 1 章第 14 集的 `torch.argmax`，一行。

### 用一個會數數的模型試

還沒載入第 5 章的模型之前，先用第 5 章第 8 集的 smoke 設定訓練一個「下一個是 +1」的小模型，看生成怎麼跑：

```python
import numpy as np
import torch

from llmz.generate import greedy_next
from llmz.model import ModelConfig, TransformerLM
from llmz.train import TrainConfig, train

tokens = (np.arange(20000) % 64).astype(np.uint16)
tokens.tofile("train.bin")
tokens.tofile("valid.bin")
cfg = TrainConfig(
    train_path="train.bin", valid_path="valid.bin", out_dir="smoke",
    model=ModelConfig(vocab_size=64, context_length=16, d_model=32, n_layers=2, n_heads=4, d_ff=64),
    batch_size=8, max_steps=150, warmup_steps=5, lr_max=3e-3, log_every=1000, eval_every=1000, checkpoint_every=1000, device="cpu",
)
train(cfg)
model = TransformerLM(cfg.model)
model.load_state_dict(torch.load("smoke/model.pt"))
model.eval()

ids = [10, 11, 12]
with torch.no_grad():
    for _ in range(8):
        logits = model(torch.tensor([ids[-16:]]))[0, -1]
        ids.append(greedy_next(logits).item())
print(ids)
```

執行結果（某一次）：

```text
{'step': 150, 'loss': 0.0554, 'lr': 0.0001, 'grad_norm': 0.076, 'time': 4.5, 'val_loss': 0.0522}
[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
```

模型從 10、11、12 接下去數到 20。每一圈：`torch.tensor([ids[-16:]])` 把目前的序列包成 `(1, T)`（batch 1，最多餵 `context_length` 個），`[0, -1]` 取第 0 段、最後一個位置的 logits，`greedy_next` 挑一個，`.item()` 變成 Python 整數接上去。`model.eval()` 與 `no_grad` 是第 5 章第 8 集 `evaluate` 的慣例，生成時一律這樣。

### greedy 的問題

greedy 是確定的：同樣的開頭永遠生出同樣的句子。而且它每一步只看眼前最大的，容易卡進重複：「and then he said and then he said……」。真實的文字每一步都有好幾個合理的下一個字，下一集改成照機率**抽**。

## 重點整理

- 生成 = 重複「餵序列 → 取最後位置的 logits → 挑一個 token → 接上去」，直到結束 token 或長度上限。
- greedy：`torch.argmax`，確定、常重複。
- 餵模型時只給最後 `context_length` 個 token；用 `model.eval()` 與 `torch.no_grad()`。
