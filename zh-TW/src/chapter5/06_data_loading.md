# 資料載入：隨機取 batch

## 本集目標

寫出 `get_batch`：從幾千萬個 token 的檔案裡隨機切出一批固定長度的序列，順便做好往後移一格的答案。

## 正文

### 不需要切成「句子」

語言模型的訓練資料不用對齊故事的開頭。`train.bin` 是四千萬個 token 排成一列，每一步只要隨機挑 `batch_size` 個起點，各取 `context_length + 1` 個連續的 token：前 `context_length` 個當輸入、後 `context_length` 個當答案。故事之間有 `<|endoftext|>`，模型自己會學到那之後是新的開始。

```python,ignore
{{#include ../../../src/llmz/data.py:get_batch}}
```

- `torch.randint(0, len(data) - context_length, (batch_size,))`：起點，最大值確保取 `context_length + 1` 個不會超出檔尾。用 `torch.randint` 而不是 NumPy 的隨機數，是為了讓第 1 章第 10 集的 `torch.manual_seed` 一次控制全部。
- `data[i : i + T + 1]` 從 `np.memmap` 切片，只有這一段真的被讀進記憶體（第 3 章第 14 集）。`np.stack` 疊成 `(B, T+1)`。
- `astype(np.int64)`：檔案裡是 uint16，模型的索引要 `int64`。`torch.from_numpy` 轉成張量。
- 最後一行切成輸入與答案，搬到 device。

```python
import numpy as np
import torch

from llmz.data import get_batch

data = np.arange(1000, dtype=np.uint16)
torch.manual_seed(0)
x, y = get_batch(data, batch_size=3, context_length=5, device=torch.device("cpu"))
print(x)
print(y)
print(x.dtype, torch.equal(y, x + 1))
```

執行結果：

```text
tensor([[944, 945, 946, 947, 948],
        [454, 455, 456, 457, 458],
        [683, 684, 685, 686, 687]])
tensor([[945, 946, 947, 948, 949],
        [455, 456, 457, 458, 459],
        [684, 685, 686, 687, 688]])
torch.int64 True
```

假資料是 0、1、2、…，所以 `y` 剛好是 `x + 1`，一眼看出對齊沒錯。

### 一輪是多少

標準語料 40,720,162 個 token，一個 batch 32 × 256 = 8192 個，5000 步看 4096 萬個，差不多一輪。隨機取起點不保證每個 token剛好看一次，但統計上接近。

## 重點整理

- `get_batch(data, batch_size, context_length, device)`：隨機起點、`memmap` 切片、`np.stack`、轉 `int64`，回傳往後移一格的 `(inputs, targets)`。
- 起點隨機數用 `torch.randint`，由 `torch.manual_seed` 控制。
- 5000 步 × 8192 個 token ≈ 標準語料一輪。
