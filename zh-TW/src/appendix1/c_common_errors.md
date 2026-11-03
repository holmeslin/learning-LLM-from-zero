# 常見錯誤（shape、device、dtype）

## 本集目標

把第一部最常撞到的錯誤訊息排成一張表：看到哪一行，就知道是哪一類問題、該去看哪一集。每個都附一段故意出錯的程式，訊息是 PyTorch 2.9 的實際輸出。

## 正文

### 形狀

```python,error
import torch

print(torch.ones(3) + torch.ones(2))
```

錯誤訊息：

```text
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0
```

逐元素運算的兩邊形狀對不上，broadcasting 也救不了（第 1 章第 5 集）。先印兩邊的 `.shape`，想清楚哪一維應該是 1 或相同；常見的修法是 `reshape(-1, 1)` 或 `keepdim=True`。

```python,error
import torch

print(torch.ones(2, 3) @ torch.ones(2, 3))
```

錯誤訊息：

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)
```

矩陣乘法內側維度要相同（第 1 章第 7 集）。多半是忘了 `.T` 或 `transpose(-2, -1)`，或者 Linear 的權重放成 `(in, out)`。

```python,error
import torch

print(torch.arange(6).reshape(4, 2))
```

錯誤訊息：

```text
RuntimeError: shape '[4, 2]' is invalid for input of size 6
```

`reshape` 前後元素總數要一樣（第 1 章第 3 集）。multi-head 的 `reshape(B, T, n_heads, d_head)` 算錯 `d_head` 時就是它。

### device

```python,norun
import torch

from llmz.device import get_device

x = torch.ones(3).to(get_device())
print(x + torch.ones(3))
```

錯誤訊息（在有 MPS 的機器上）：

```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!
```

一個在 GPU、一個在 CPU（第 1 章第 9 集）。模型 `.to(device)` 之後，每一批資料也要 `.to(device)`；在模型裡建新張量時指定 `device=x.device`（第 4 章第 12 集的 `torch.arange(T, device=x.device)`）。這段在只有 CPU 的機器上不會出錯，所以沒有被自動驗證。

### dtype

```python,error
import torch

print(torch.tensor([1, 2, 3]).mean())
```

錯誤訊息：

```text
RuntimeError: mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Long
```

整數張量不能算平均（第 1 章第 3 集）。`torch.tensor([1, 2, 3])` 是 `int64`，先 `.float()`。`torch.exp`、`torch.log` 會自動把整數轉成浮點數，但 `.mean()`、`.std()` 不會。

```python,error
import torch

w = torch.randn(5, 3)
print(w[torch.tensor([1.0, 2.0])])
```

錯誤訊息：

```text
IndexError: tensors used as indices must be long, int, byte or bool tensors
```

拿浮點數張量當索引（第 1 章第 4 集）。`get_batch` 的 `astype(np.int64)` 就是在避免這個：從 uint16 檔案讀出來的 id 要先轉成整數型別才能查 embedding。

```python,error
import torch

emb = torch.randn(10, 4)
print(emb[torch.tensor([3, 10])])
```

錯誤訊息：

```text
IndexError: index 10 is out of bounds for dimension 0 with size 10
```

索引超出範圍。embedding 的表有 `vocab_size` 列，id 最大只能到 `vocab_size - 1`；tokenizer 的詞彙表和模型的 `vocab_size` 對不上時會出現。

### autograd

```python,error
import torch

w = torch.ones(3, requires_grad=True)
(w * 2).backward()
```

錯誤訊息：

```text
RuntimeError: grad can be implicitly created only for scalar outputs
```

`backward()` 只能對純量呼叫（第 2 章第 1 集），先 `.sum()` 或 `.mean()`。

```python,error
import torch

w = torch.ones(3, requires_grad=True)
w -= 0.1 * w
```

錯誤訊息：

```text
RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
```

直接就地改一個要求梯度的參數（第 2 章第 2 集）。更新參數要包在 `with torch.no_grad():` 裡。

### `nn.Module`

```python,error
import torch
from torch import nn


class M(nn.Module):
    def __init__(self):
        self.w = nn.Parameter(torch.zeros(3))


M()
```

錯誤訊息：

```text
AttributeError: cannot assign parameters before Module.__init__() call
```

忘了 `super().__init__()`（第 2 章第 3 集）。

```python,error
import torch
from torch import nn


class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(3))


class B(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(4))


B().load_state_dict(A().state_dict())
```

錯誤訊息：

```text
	size mismatch for w: copying a param with shape torch.Size([3]) from checkpoint, the shape in current model is torch.Size([4]).
```

載入的 checkpoint 和模型的形狀對不上（第 2 章第 6 集）。多半是建模型時的 `ModelConfig` 和訓練時不同：載入 `runs/<名稱>/config.json` 裡的設定來建。鍵名對不上（多一個、少一個）會是 `Missing key(s)` 或 `Unexpected key(s)`。

### 不是錯誤訊息的錯誤

| 現象 | 常見原因 | 看哪裡 |
| --- | --- | --- |
| loss 一開始就是 `nan` | softmax 沒減最大值、mask 把整列遮掉、學習率太大 | 第 1 章第 16 集、第 4 章第 8 集、第 6 章第 9 集 |
| loss 一開始遠大於 \\( \ln V \\) | 初始化太大 | 第 4 章第 2 集、第 5 章第 1 集 |
| loss 完全不動 | 忘了 `optimizer.step()`、學習率是 0、`requires_grad` 沒開 | 第 2 章第 4 集 |
| loss 越來越大 | 忘了 `zero_grad()`、學習率太大 | 第 2 章第 2 集、第 6 章第 9 集 |
| 訓練 loss 很低、驗證 loss 很高 | 過擬合，資料太少或看太多輪 | 第 5 章第 8 集 |
| 前幾個位置的輸出會被後面的 token 影響 | causal mask 沒接上、或 mask 的 True/False 反了 | 第 4 章第 10 集 |
| `ModuleNotFoundError: No module named 'torch'` | 沒用 `uv run` | 第 1 章第 1 集 |

## 重點整理

- 形狀：`size of tensor a must match`、`cannot be multiplied`、`shape is invalid`，先印 `.shape`。
- device：`Expected all tensors to be on the same device`，資料與新張量都要在 `x.device`。
- dtype：整數不能 `.mean()`、浮點數不能當索引、索引不能超過 `vocab_size - 1`。
- autograd：`backward()` 要純量、更新參數要 `no_grad`。
- `nn.Module`：先 `super().__init__()`；`state_dict` 形狀與鍵名要完全一致。
