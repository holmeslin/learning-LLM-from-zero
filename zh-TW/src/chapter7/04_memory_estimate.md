# 記憶體估算：參數、梯度、優化器狀態、activations

## 本集目標

把訓練時的記憶體拆成四塊，各自估出或量出大小，知道 batch 開大時爆掉的是哪一塊。

## 正文

### 四塊記憶體

第 2 章第 7 集說過，用 AdamW 訓練時每個參數要 16 位元組：參數 4、梯度 4、AdamW 的兩個動量 8。這三塊跟 batch 大小無關，寫成函式：

```python,ignore
{{#include ../../../src/llmz/profile.py:estimate_memory}}
```

```python
from llmz.model import ModelConfig, TransformerLM
from llmz.profile import estimate_memory
from llmz.utils import count_parameters

n = count_parameters(TransformerLM(ModelConfig()))
memory = estimate_memory(n)
for name, size in memory.items():
    print(f"{name:10} {size / 1e6:6.1f} MB")
print(f"{'total':10} {sum(memory.values()) / 1e6:6.1f} MB")
```

執行結果：

```text
params       90.8 MB
grads        90.8 MB
optimizer   181.6 MB
total       363.1 MB
```

標準模型只要 363 MB。但第 5 章在 MPS 上訓練時用了 5.5 GB 以上，差的大部分是第四塊：**activations**，forward 時為了 backward 而留下來的中間結果。

### 量出 autograd 存了什麼

activations 很難一個一個數，但可以直接量。PyTorch 的 autograd 每次要為 backward 存一個張量時，都會經過一個可以掛鉤子的地方：`torch.autograd.graph.saved_tensors_hooks(pack, unpack)`。在這個 `with` 區塊裡，每存一個張量就呼叫一次 `pack`，我們在 `pack` 裡記下它的大小就行：

```python,ignore
{{#include ../../../src/llmz/profile.py:saved_bytes}}
```

同一塊記憶體可能被好幾個運算存起來（例如同一個輸入被兩個運算用到），所以用 `untyped_storage().data_ptr()`（這塊記憶體的位址）當鍵，只算一次。

還有一件事：`Linear` 的 backward 要用到 `weight`，所以 autograd 也會把參數存起來。參數已經算在第一塊了，`exclude` 讓我們把它們排除。

先用一個小例子確認它量得對：

```python
import torch

from llmz.profile import saved_activation_bytes

w = torch.ones(10, requires_grad=True)
print(saved_activation_bytes(lambda: torch.exp(w).sum()))
print(saved_activation_bytes(lambda: (w * w).sum()))
print(saved_activation_bytes(lambda: w.sum()))
print(saved_activation_bytes(lambda: (w * w).sum(), exclude=[w]))
```

執行結果：

```text
40
40
0
0
```

\\( e^w \\) 的導數就是 \\( e^w \\) 自己，所以 `exp` 只存輸出，10 個 float32 是 40 位元組。`w * w` 對兩邊的導數都要用到 `w`，存了兩次同一個張量，去掉重複後還是 40。`sum` 的導數全是 1，什麼都不用存。最後一行把 `w` 排除，就是 0。

### 標準模型

用 batch 4 量標準模型（batch 32 要 3 GB，太佔記憶體）：

```python
import torch

from llmz.functional import cross_entropy
from llmz.model import ModelConfig, TransformerLM
from llmz.profile import saved_activation_bytes

torch.manual_seed(0)
model = TransformerLM(ModelConfig())
x = torch.randint(0, 10000, (4, 257))


def loss_fn():
    logits = model(x[:, :-1])
    return cross_entropy(logits.reshape(-1, 10000), x[:, 1:].reshape(-1))


print(f"{saved_activation_bytes(loss_fn, exclude=model.parameters()) / 1e6:.1f} MB")
```

執行結果：

```text
376.8 MB
```

batch 4 就已經比參數、梯度、優化器狀態加起來還多。我們在 CPU 上改不同設定量了幾組（數字在任何機器上都一樣）：

| batch | 長度 | 層數 | activations |
| --- | --- | --- | --- |
| 32 | 256 | 4 | 3011.1 MB |
| 16 | 256 | 4 | 1505.8 MB |
| 32 | 128 | 4 | 1237.0 MB |
| 32 | 256 | 8 | 5643.8 MB |

從表上讀出三個規律：

- **跟 batch 成正比**：batch 減半，activations 剛好減半。
- **跟層數成正比**：4 層變 8 層多了 2633 MB，一層大約 658 MB。剩下不屬於任何一層的大約 378 MB，主要是 logits：\\( 32 \times 256 \times 10000 \times 4 \\) 位元組 = 328 MB。
- **長度減半，省得比一半還多**：長度 128 只要 1237 MB，不到 3011 的一半。因為 attention 分數矩陣（第 1 集的 134 MB）跟長度的**平方**成正比，長度減半它就只剩四分之一。

### 加起來對不對得上

batch 32 的估算是 363 MB + 3011 MB ≈ 3.4 GB。M4 Max 上實際量到 MPS 配置了 5.5～6.6 GB（`torch.mps.driver_allocated_memory()`，依量的時機不同）。多出來的是 backward 過程中暫時算出來的東西（每個 activation 的梯度，用完就丟）、運算之間的暫存張量，以及 PyTorch 為了下次重用而先留著不還的記憶體。所以估算告訴我們「誰是大宗、怎麼隨設定變化」，實際要多少還是要上機器看。

在 16 GB 的 Mac 上，系統和其他程式先佔掉幾 GB，batch 32 的標準設定還放得下；batch 128 就要 12 GB 以上的 activations，會很吃緊。下兩集介紹的混合精度與 activation checkpointing，就是從 activations 這塊下手。

## 重點整理

- 訓練記憶體四塊：參數、梯度、優化器狀態（AdamW 共 16 位元組一個參數，`estimate_memory`）與 activations。
- `saved_tensors_hooks` 可以攔下 autograd 存的每個張量；`saved_activation_bytes(fn, exclude=model.parameters())` 用它量出 activations 的位元組數。
- 標準模型 batch 32：參數那三塊 363 MB，activations 3011 MB。
- activations 跟 batch、層數成正比；attention 的部分跟長度的平方成正比。
