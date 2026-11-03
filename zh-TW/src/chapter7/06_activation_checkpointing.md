# activation checkpointing

## 本集目標

用「backward 時重算」換記憶體：每個 Transformer block 只留輸入，中間結果到 backward 時再算一次，並實測省了多少、慢了多少。

## 正文

### 用計算換記憶體

第 4 集量到標準模型一層要存 658 MB 的 activations，但每一層的輸入只有 `(32, 256, 512)`，16.8 MB。大部分是 block 內部的中間結果：attention 分數、softmax 的輸出、SwiGLU 的三個中間張量……

**activation checkpointing**（也叫 gradient checkpointing）的想法是：forward 時每個 block 只存輸入，內部的中間結果算完就丟。backward 走到這個 block 時，拿存下來的輸入重新跑一次這個 block 的 forward，把中間結果算回來，再做 backward。

代價是每個 block 的 forward 要跑兩次。訓練一步大約是 forward 1 份、backward 2 份，多跑一次 forward 就是多三分之一的計算。

### 加進模型

PyTorch 提供 `torch.utils.checkpoint.checkpoint(fn, x)`：執行 `fn(x)`，但不存 `fn` 裡面的中間結果，backward 時自動重算。它不是現成的層，而是 autograd 的工具，本書的規則允許用。我們在 `ModelConfig` 加一個開關 `activation_checkpoint`，`TransformerLM.forward` 改成：

```python,ignore
{{#include ../../../src/llmz/model.py:lm}}
```

`use_reentrant=False` 是 PyTorch 建議的新版實作，舊版有些限制，新程式一律用 `False`。`self.training` 是 `Module` 內建的旗標，預設是 `True`；呼叫 `model.eval()` 會把它設成 `False`，推論時這樣做，不需要 backward，也就不用 checkpoint。

### 結果一樣，記憶體少很多

```python
import torch

from llmz.functional import cross_entropy
from llmz.model import ModelConfig, TransformerLM
from llmz.profile import saved_activation_bytes

torch.manual_seed(0)
x = torch.randint(0, 10000, (4, 257))
for flag in [False, True]:
    torch.manual_seed(1)  # 兩個模型的初始參數相同
    model = TransformerLM(ModelConfig(activation_checkpoint=flag))

    def loss_fn():
        logits = model(x[:, :-1])
        return cross_entropy(logits.reshape(-1, 10000), x[:, 1:].reshape(-1))

    saved = saved_activation_bytes(loss_fn, exclude=model.parameters())
    loss_fn().backward()
    grad = model.blocks[0].attn.q_proj.weight.grad
    print(flag, f"{saved / 1e6:.1f} MB", f"{grad.abs().sum().item():.4f}")
```

執行結果：

```text
False 376.8 MB 19.1185
True 55.7 MB 19.1185
```

梯度一模一樣（重算的過程和第一次完全相同），存下來的 activations 只剩不到六分之一。batch 32 在 CPU 上量：

| 層數 | 不 checkpoint | checkpoint |
| --- | --- | --- |
| 4 | 3011.1 MB | 445.4 MB |
| 8 | 5643.8 MB | 512.6 MB |

checkpoint 之後每多一層只多 16.8 MB（那一層的輸入），剩下的大多是 logits 那 328 MB。backward 時一次只重算一個 block，所以任何時刻最多再多一層的 658 MB。

### M4 Max 實測

標準設定、batch 32，訓練一步：

| 設定 | 每步 | MPS 配置的記憶體 |
| --- | --- | --- |
| 不 checkpoint | 494 ms | 5.47 GB |
| checkpoint | 538 ms | 3.32 GB |

慢了 9%，比理論上的三分之一少：lm_head 和 loss 不在 block 裡，不會重算；優化器更新也不受影響。記憶體少了 2.15 GB。

### 什麼時候開

標準模型在 16 GB 的 Mac 上不開也放得下，所以預設關閉。當你想把 batch、長度或層數加大而記憶體不夠時，再用命令列開：

```bash
uv run python tools/train.py --activation_checkpoint
```

大型模型訓練幾乎都會開，而且常常只對部分層開，在記憶體與速度之間取平衡。

## 重點整理

- activation checkpointing：forward 只存每個 block 的輸入，backward 時重跑該 block 的 forward 把中間結果算回來。
- `torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)`；`ModelConfig(activation_checkpoint=True)` 或 `--activation_checkpoint` 開啟。
- 梯度完全相同；標準模型 batch 32 的 activations 從 3011 MB 降到 445 MB。
- M4 Max 實測：每步慢 9%，記憶體少 2.15 GB。
