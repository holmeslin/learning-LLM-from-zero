# `torch.distributed` + Gloo 在 CPU 起兩個 process

## 本集目標

寫出一個起好幾個 process、每個都加入同一個通訊群組的小工具，認識 rank 與 world size。

## 正文

### process、rank、world size

第 3 章第 8 集用 `multiprocessing.Pool` 把 BPE 的統計分給好幾個 process。那時每個 process 各算各的，算完把結果交回主程式。多卡訓練不一樣：每個 process 從頭到尾都在，訓練到一半要和其他 process 交換梯度，所以它們得先**加入同一個群組**。

- **world size**：群組裡有幾個 process（幾張卡）。
- **rank**：自己是第幾個，從 0 到 world size − 1。程式靠 rank 決定「我要處理哪一份資料」。

每個 process 有自己獨立的記憶體，就像每張卡有自己的顯示卡記憶體。一個 process 改了自己的張量，別人看不到，要交換只能透過通訊。

### 起 process 並加入群組

放進 `src/llmz/dist.py`：

```python,ignore
{{#include ../../../src/llmz/dist.py:launch}}
```

- `mp.spawn(fn, args, nprocs)`：`torch.multiprocessing` 的函式，起 `nprocs` 個新的 Python process，第 \\( i \\) 個執行 `fn(i, *args)`，全部結束才回來。新的 process 會重新 import 主程式，所以主程式一樣要放在 `if __name__ == "__main__":` 裡（第 3 章第 8 集），而要執行的函式必須定義在檔案的最外層。
- `dist.init_process_group("gloo", init_method="tcp://127.0.0.1:port", rank, world_size)`：用 Gloo 後端加入群組。大家約好在本機的某個 port 碰面，等到 `world_size` 個都到齊才往下走。
- port 用 `socket` 綁定 port 0 讓作業系統挑一個空的，這樣同時跑好幾個範例也不會撞到。
- `GLOO_SOCKET_IFNAME` 告訴 Gloo 只走本機的網路介面（Mac 叫 `lo0`、Linux 叫 `lo`），不設的話它會印一行警告。
- 結束前 `destroy_process_group()` 離開群組。

### 第一個多 process 程式

好幾個 process 同時 `print`，輸出的順序每次不同。`print_in_order` 讓大家輪流印：

```python,ignore
{{#include ../../../src/llmz/dist.py:print_in_order}}
```

`dist.barrier()` 是最簡單的通訊：每個 process 走到這裡都要停下來等，等大家都到了才一起往下走。所以 rank 0 印完、大家過了第一個 barrier，才輪到 rank 1。

```python
import torch

from llmz.dist import launch, print_in_order


def hello(rank, world_size):
    x = torch.tensor([rank * 10.0])
    print_in_order("world_size =", world_size, "x =", x.tolist())


if __name__ == "__main__":
    launch(hello, 4)
```

執行結果：

```text
[rank 0] world_size = 4 x = [0.0]
[rank 1] world_size = 4 x = [10.0]
[rank 2] world_size = 4 x = [20.0]
[rank 3] world_size = 4 x = [30.0]
```

四個 process 執行同一個函式，只有 `rank` 不同，所以 `x` 不同。這就是所有多卡程式的樣子：**一份程式、很多份資料，靠 rank 區分**。

### 換成真正的 GPU

在有好幾張 NVIDIA 卡的機器上，改三個地方就是真正的多卡程式：後端從 `"gloo"` 換成 `"nccl"`、每個 process 用 `torch.cuda.set_device(rank)` 選自己的卡、張量放到 `"cuda"`。一般不用 `mp.spawn` 而用 `torchrun` 指令來起 process，它會用環境變數告訴每個 process 自己的 rank。通訊函式的寫法完全不變。

## 重點整理

- 多卡訓練：world size 個 process 各執行同一份程式，用 rank（0 到 world size − 1）區分。
- `llmz.dist.launch(fn, world_size, *args)`：用 `mp.spawn` 起 process、`init_process_group("gloo", ...)` 加入群組，每個執行 `fn(rank, world_size, *args)`。
- 每個 process 的記憶體獨立，只能透過通訊交換資料；`dist.barrier()` 讓大家互相等。
- `print_in_order` 讓輸出順序固定；GPU 上把 `gloo` 換成 `nccl`，寫法不變。
