# device：cpu 與 mps

## 本集目標

知道張量住在哪個 device，用固定句型 `get_device()` 把運算搬到 GPU（Mac 上叫 mps），並看懂 device 不合的錯誤訊息。

## 正文

### 張量住在哪裡

到目前為止我們的張量都在 CPU 上算。訓練模型時要做非常多的矩陣乘法，GPU 擅長這件事，快很多（第 5 章會實測）。Apple Silicon 的 GPU 在 PyTorch 裡叫 **mps**（Metal Performance Shaders），NVIDIA 的叫 **cuda**。

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
print(x.device)
print(torch.backends.mps.is_available())
```

執行結果（某一次）：

```text
cpu
True
```

張量預設建立在 `cpu`。第二行在 Apple Silicon 的 Mac 上是 `True`，在沒有 GPU 的機器上是 `False`，所以這段的輸出每台機器不一樣。

### 固定句型：`get_device()`

本書的套件 `llmz` 裡有一個小函式，幫你挑可用的 device，優先順序是 cuda、mps、cpu：

```python,ignore
{{#include ../../../src/llmz/device.py:get_device}}
```

用法照抄即可：

```python
import torch

from llmz.device import get_device

device = get_device()
x = torch.tensor([1.0, 2.0, 3.0]).to(device)
y = torch.zeros(3, device=device)
print(x.device)
print((x * 2).device)
print((x * 2).to("cpu"))
```

執行結果（某一次）：

```text
mps:0
mps:0
tensor([2., 4., 6.])
```

- `from llmz.device import get_device`：從本書套件載入這個函式。`uv sync` 已經把 `llmz` 裝好了，所以可以直接 import。
- `.to(device)`：把張量複製一份到那個 device 上。建立時也可以直接指定 `device=device`。
- 在 mps 上的張量做運算，結果還是在 mps 上；要印出來或存檔時，用 `.to("cpu")` 搬回來。`.item()` 在任何 device 上都能用。
- `mps:0` 的 `0` 是編號，只有一顆 GPU 時永遠是 0。

### 不同 device 不能一起算

一個在 GPU、一個在 CPU 的張量放在一起運算會報錯。這段在 Mac 上會出錯；在只有 CPU 的機器上，`get_device()` 也回傳 cpu，兩個就住在同一處，不會出錯：

```python,norun
import torch

from llmz.device import get_device

a = torch.tensor([1.0, 2.0, 3.0]).to(get_device())
b = torch.tensor([1.0, 2.0, 3.0])
print(a + b)
```

錯誤訊息（在 Mac 上）：

```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!
```

這是 PyTorch 的第三常見錯誤（前兩名是上面兩集的形狀錯誤）。解法永遠是：把其中一個 `.to(...)` 到另一邊。第 5 章訓練時，資料與模型都要搬到同一個 device。

### 本書的輸出都用 CPU

GPU 算浮點數的順序和 CPU 不完全一樣，結果的最後幾位小數可能不同。為了讓你能逐字對答案，**本書所有「執行結果」都是在 CPU 上算的**，之後只有真的要訓練時才會用 `get_device()`。你在 mps 上跑，看到小數點後第四位不一樣，不是你寫錯。

## 重點整理

- 張量預設在 `cpu`；Apple Silicon 的 GPU 叫 `mps`，NVIDIA 的叫 `cuda`。
- 固定句型：`from llmz.device import get_device`、`device = get_device()`、`x.to(device)`。
- 運算結果留在輸入所在的 device；不同 device 的張量不能一起算，錯誤訊息是 `Expected all tensors to be on the same device`。
- 本書的執行結果都在 CPU 上產生，GPU 的結果末位可能不同。
