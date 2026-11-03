# bucketing 與通訊重疊

## 本集目標

用兩個技巧讓 DDP 變快：把很多小梯度打包成幾個 bucket 一起傳，並且在 backward 還在算的時候就開始傳已經算好的梯度。

## 正文

### 兩個問題

上一集的 naive DDP 有兩個浪費：

1. **次數太多**：每次通訊都有固定的啟動成本，39 個參數就付 39 次。把很多梯度接成一個長張量一次傳，固定成本只付一次。
2. **等太久**：backward 是從最後一層往前算的，`lm_head` 的梯度最早算好，embedding 最晚。naive 版等全部算完才開始傳，這段時間網路閒著；算好的梯度其實可以馬上開始傳，和還在進行的 backward **重疊**。

把參數依 backward 的順序（大致是倒過來）分成幾個 **bucket**，每個 bucket 大約 25 MB（PyTorch 的 DDP 預設也是 25 MB）。一個 bucket 裡的梯度都算好了，就把它們接起來丟出一個非同步的 all-reduce，backward 繼續算下一層。

### 兩個新工具

- `p.register_post_accumulate_grad_hook(fn)`：替參數 `p` 掛一個鉤子，backward 把梯度寫進 `p.grad` 之後就呼叫 `fn(p)`。我們用它知道「這個參數的梯度好了」。
- `dist.all_reduce(x, async_op=True)`：丟出通訊就馬上回來，回傳一個 `Work` 物件；之後呼叫 `.wait()` 才等它做完。和第 3 集的 `isend` 同一個道理。

```python,ignore
{{#include ../../../src/llmz/dist.py:bucketed_ddp}}
```

- 建構時從 rank 0 broadcast 參數，再把參數倒過來依序裝進 bucket，裝滿 `bucket_mb` 就開下一個。
- 每個參數的梯度好了，鉤子把那個 bucket 的計數加一；計數等於 bucket 的參數數量，就用 `torch.cat` 把梯度接成一個長張量，丟出非同步 all-reduce。
- `finish()` 在 `opt.step()` 之前呼叫：等每個 all-reduce 做完，除以 \\( N \\)，再切回去複製到各參數的 `.grad`，最後把計數歸零給下一步用。

類別開頭的 `shortcut:` 註解寫明了這個簡化版的限制：它假設每次 backward 每個參數剛好收到一次梯度。PyTorch 真正的 DDP 會處理沒用到的參數、梯度累積等情況。

### 看到重疊

用標準模型、兩個 rank，記下每個 bucket 丟出 all-reduce 的那一刻，還有幾個參數的梯度沒算出來：

```python
import torch

from llmz.dist import BucketedDDP, launch
from llmz.functional import cross_entropy
from llmz.model import ModelConfig, TransformerLM


class TracedDDP(BucketedDDP):
    def _on_grad_ready(self, p):
        before = len(self.works)
        super()._on_grad_ready(p)
        if len(self.works) > before:  # 剛丟出一個 all-reduce
            waiting = sum(q.grad is None for q in self.model.parameters())
            self.log.append((self.works[-1][0], waiting))


def run(rank, world_size):
    torch.manual_seed(0)
    model = TransformerLM(ModelConfig())
    ddp = TracedDDP(model)
    ddp.log = []
    ids = torch.randint(0, 10000, (2, 33))
    cross_entropy(model(ids[:, :-1]).reshape(-1, 10000), ids[:, 1:].reshape(-1)).backward()
    ddp.finish()
    if rank == 0:
        for i, bucket in enumerate(ddp.buckets):
            mb = sum(p.numel() for p in bucket) * 4 / 1e6
            print(f"bucket {i}：{len(bucket):2} 個參數，{mb:5.1f} MB")
        for i, waiting in ddp.log:
            print(f"bucket {i} 丟出 all-reduce 時，還有 {waiting} 個參數沒有梯度")


if __name__ == "__main__":
    launch(run, 2)
```

執行結果：

```text
bucket 0： 4 個參數， 26.0 MB
bucket 1：19 個參數， 27.7 MB
bucket 2：16 個參數， 37.1 MB
bucket 0 丟出 all-reduce 時，還有 35 個參數沒有梯度
bucket 1 丟出 all-reduce 時，還有 16 個參數沒有梯度
bucket 2 丟出 all-reduce 時，還有 0 個參數沒有梯度
```

39 個參數變成 3 次通訊。bucket 0 丟出去時還有 35 個參數的梯度沒算，bucket 1 丟出去時還有 16 個，這兩次通訊都和剩下的 backward 同時進行；只有最後一個 bucket 要等 backward 結束。最後一個 bucket 特別大，因為 embedding 一個就佔了 20.5 MB。

### 實測

M4 Max 的 CPU、兩個 rank、標準模型（長度 128、每個 rank batch 8），每步時間兩輪的中位數：

| 版本 | 通訊次數 | 每步（兩輪） |
| --- | --- | --- |
| 不通訊（比較基準） | 0 | 698、823 ms |
| naive DDP | 39 | 1105、1166 ms |
| 一個 bucket（全部接在一起，backward 後才傳） | 1 | 975、927 ms |
| 25 MB 的 bucket，邊算邊傳 | 3 | 851、882 ms |

打包省下一段，重疊再省一段，兩輪的排序都一樣。CPU 上計算與通訊搶同一批核心，重疊的效果打了折扣；在 GPU 上，計算在 GPU、通訊走網路卡，兩者真的能同時進行，重疊的效果更明顯。

### bucket 要多大

bucket 太小，次數變多，回到 naive 的問題；太大，第一個 bucket 要等很久才湊滿，能重疊的時間變少，最極端就是「一個 bucket」那一列。25 MB 是 PyTorch 的預設值，實務上會依模型與網路調整。

## 重點整理

- bucketing：把很多小梯度接成一個長張量一次 all-reduce，省下每次通訊的固定成本。
- 重疊：`register_post_accumulate_grad_hook` 知道梯度何時算好，`all_reduce(..., async_op=True)` 丟出去不等，backward 繼續；`finish()` 再一起 `.wait()`。
- `llmz.dist.BucketedDDP(model, bucket_mb)`：標準模型 39 個參數變 3 個 bucket，前兩個在 backward 進行中就開始傳。
- M4 Max CPU 實測：naive 約 1.1 秒一步，bucket 加重疊約 0.87 秒，接近不通訊的 0.7～0.8 秒。
