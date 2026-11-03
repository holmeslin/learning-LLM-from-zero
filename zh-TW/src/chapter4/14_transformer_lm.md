# 完整 Transformer LM

## 本集目標

把 Embedding、n 層 block、最後的 RMSNorm 與 Linear 組成 `TransformerLM`，從 token id 算出 logits，並用第 1 章的 cross-entropy 算出第一個（還沒訓練的）loss。

## 正文

### 組裝

```python,ignore
{{#include ../../../src/llmz/model.py:lm}}
```

`nn.ModuleList` 是 `nn.Module` 的串列版：放進去的每一層的參數都會被登記，`parameters()`、`.to(device)`、`state_dict()` 都看得到（普通的 Python list 不會）。RoPE 建一個給所有層共用。`forward` 就是第 1 集那張圖：查表、逐層、norm、投影到 `vocab_size`。迴圈裡 `if self.cfg.activation_checkpoint` 那個分支是第 7 章第 6 集省記憶體用的，預設關閉，現在把它當成只有 `x = block(x)` 這一行。

```python
import torch

from llmz.model import ModelConfig, TransformerLM
from llmz.utils import count_parameters

torch.manual_seed(0)
cfg = ModelConfig(vocab_size=50, context_length=16, d_model=32, n_layers=2, n_heads=4, d_ff=64)
model = TransformerLM(cfg)
ids = torch.randint(0, 50, (2, 10))
logits = model(ids)
print(logits.shape)
print(count_parameters(model))
print(logits[0, 0, :5])
```

執行結果：

```text
torch.Size([2, 10, 50])
23840
tensor([ 0.0474,  0.3298, -1.1284, -1.2346,  0.2434], grad_fn=<SliceBackward0>)
```

`(2, 10)` 進、`(2, 10, 50)` 出。最後一行是第 0 段第 0 個位置對前 5 個 token 的分數，現在是亂數。

### 第一個 loss

訓練目標：位置 t 的 logits 要猜中 `ids[t+1]`。所以輸入拿 `ids[:, :-1]`，答案拿 `ids[:, 1:]`，兩個都是 `(B, T-1)`。第 1 章第 16 集的 `cross_entropy` 吃 `(N, V)` 與 `(N,)`，把 batch 和位置攤平就行：

```python
import torch

from llmz.functional import cross_entropy, perplexity
from llmz.model import ModelConfig, TransformerLM

torch.manual_seed(0)
cfg = ModelConfig(vocab_size=50, context_length=16, d_model=32, n_layers=2, n_heads=4, d_ff=64)
model = TransformerLM(cfg)
ids = torch.randint(0, 50, (2, 11))
inputs, targets = ids[:, :-1], ids[:, 1:]
logits = model(inputs)
loss = cross_entropy(logits.reshape(-1, 50), targets.reshape(-1))
print(round(loss.item(), 3), round(perplexity(loss).item(), 1))
import math
print(round(math.log(50), 3))
```

執行結果：

```text
3.991 54.1
3.912
```

`reshape(-1, 50)` 的 `-1` 讓 PyTorch 自己算那一維（這裡是 2 × 10 = 20）。還沒訓練的模型 loss 接近 \\( \ln 50 = 3.91 \\)，也就是均勻亂猜的程度（第 1 章第 15 集：perplexity 接近詞彙表大小）。第 5 章的工作就是把這個數字壓下去：標準語料 vocab 10000，起點是 \\( \ln 10000 \approx 9.2 \\)，訓練好大約 1.4。

### 用標準設定建一個

```python
from llmz.model import ModelConfig, TransformerLM
from llmz.utils import count_parameters

model = TransformerLM(ModelConfig())
print(count_parameters(model))
```

執行結果：

```text
22696448
```

兩千兩百多萬個參數，下一集算清楚它們在哪裡、要多少記憶體與計算量。

## 重點整理

- `TransformerLM(cfg)`：Embedding → `nn.ModuleList` 的 n 層 block → RMSNorm → Linear 到 `vocab_size`。
- `nn.ModuleList` 讓串列裡的層被當成子模組管理。
- 訓練的輸入是 `ids[:, :-1]`、答案是 `ids[:, 1:]`，logits 攤平成 `(N, V)` 交給 `cross_entropy`。
- 未訓練的 loss ≈ \\( \ln(\text{vocab\_size}) \\)。
