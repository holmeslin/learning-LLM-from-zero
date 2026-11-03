# all-reduce 等集合通訊

## 本集目標

認識四個最常用的集合通訊（broadcast、all-reduce、all-gather、reduce-scatter），並自己用點對點的 send／recv 寫出 ring all-reduce。

## 正文

### 集合通訊

**集合通訊**（collective communication）是群組裡每個 process 都要一起呼叫的通訊函式，少一個人呼叫，其他人就會一直等。四個主角：

| 函式 | 做完之後 |
| --- | --- |
| `dist.broadcast(x, src)` | 每個人的 `x` 都變成 rank `src` 的 `x` |
| `dist.all_reduce(x)` | 每個人的 `x` 都變成所有人的 `x` 加總 |
| `dist.all_gather(out_list, x)` | 每個人的 `out_list` 都拿到所有人的 `x`，依 rank 排好 |
| `dist.reduce_scatter(out, in_list)` | 把所有人的 `in_list` 對應位置加總，第 \\( r \\) 份交給 rank \\( r \\) |

它們都是**就地**修改傳進去的張量，不回傳新張量。

```python
import torch
import torch.distributed as dist

from llmz.dist import launch, print_in_order


def demo(rank, world_size):
    x = torch.tensor([float(rank)])
    dist.broadcast(x, src=2)
    print_in_order("broadcast     ", x.tolist())

    x = torch.tensor([float(rank), 1.0])
    dist.all_reduce(x)
    print_in_order("all_reduce    ", x.tolist())

    out = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(out, torch.tensor([rank * 10.0]))
    print_in_order("all_gather    ", [t.item() for t in out])

    pieces = [torch.tensor([rank * 10.0 + i]) for i in range(world_size)]  # 每個人準備 world_size 份
    mine = torch.zeros(1)
    dist.reduce_scatter(mine, pieces)
    print_in_order("reduce_scatter", mine.tolist())


if __name__ == "__main__":
    launch(demo, 3)
```

執行結果：

```text
[rank 0] broadcast      [2.0]
[rank 1] broadcast      [2.0]
[rank 2] broadcast      [2.0]
[rank 0] all_reduce     [3.0, 3.0]
[rank 1] all_reduce     [3.0, 3.0]
[rank 2] all_reduce     [3.0, 3.0]
[rank 0] all_gather     [0.0, 10.0, 20.0]
[rank 1] all_gather     [0.0, 10.0, 20.0]
[rank 2] all_gather     [0.0, 10.0, 20.0]
[rank 0] reduce_scatter [30.0]
[rank 1] reduce_scatter [33.0]
[rank 2] reduce_scatter [36.0]
```

- broadcast 之後大家都是 rank 2 的值 2。
- all-reduce 之後大家都是 \\( 0 + 1 + 2 = 3 \\) 和 \\( 1 + 1 + 1 = 3 \\)。
- all-gather 之後大家都有 `[0, 10, 20]`。
- reduce-scatter：rank \\( k \\) 準備的第 \\( i \\) 份是 \\( 10k + i \\)。第 \\( r \\) 份在三個人手上加起來是 \\( 0 + 10 + 20 + 3r \\)，交給 rank \\( r \\)，所以是 30、33、36。

一個重要的關係：**all-reduce = reduce-scatter + all-gather**。先把加總分散到各人手上（每人負責一份），再把各人手上的那份傳給所有人。

### ring all-reduce

all-reduce 最直接的做法是大家把資料都傳給 rank 0，rank 0 加總後再傳回去。rank 0 要收 \\( N - 1 \\) 份、送 \\( N - 1 \\) 份，它的網路會變成瓶頸，人越多越慢。

**ring all-reduce** 讓 \\( N \\) 個人排成一圈，每個人只和右邊的鄰居交換，而且每次只交換資料的 \\( 1/N \\)：

1. 把資料切成 \\( N \\) 塊。
2. **reduce-scatter 階段**，走 \\( N - 1 \\) 步：每一步把手上的某一塊傳給右邊，同時從左邊收一塊、加到自己對應的那塊上。走完後，每個人都有某一塊的完整總和。
3. **all-gather 階段**，再走 \\( N - 1 \\) 步：把加總好的塊沿著圈傳一輪，覆蓋掉舊的。

用點對點通訊寫出來：`dist.isend(t, dst)` 把 `t` 送給 `dst`，不等對方收到就先回來，回傳一個可以 `.wait()` 的物件；`dist.recv(t, src)` 等著從 `src` 收資料放進 `t`。如果每個人都先用會等待的 `send` 送給右邊，大家會同時卡住等別人收，所以送的那一半用 `isend`。

```python,ignore
{{#include ../../../src/llmz/dist.py:ring_all_reduce}}
```

`x.view(-1).chunk(n)` 切出來的每一塊都是 `x` 的 view（第 1 章第 3 集），對 `chunks[i]` 就地加或複製，就是在改 `x`。最後一塊可能比較小，例如 10 個數切 3 塊是 4、4、2。

```python
import torch
import torch.distributed as dist

from llmz.dist import launch, print_in_order, ring_all_reduce


def check(rank, world_size):
    x = torch.arange(10.0) * (rank + 1)
    expected = x.clone()
    dist.all_reduce(expected)
    ring_all_reduce(x)
    print_in_order(x.tolist(), torch.equal(x, expected))


if __name__ == "__main__":
    launch(check, 4)
```

執行結果：

```text
[rank 0] [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0] True
[rank 1] [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0] True
[rank 2] [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0] True
[rank 3] [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0] True
```

每個 rank 都得到 \\( (1 + 2 + 3 + 4) \times [0, 1, \dots, 9] \\)，和 `dist.all_reduce` 一模一樣。

### 每個人傳多少

資料大小 \\( S \\) 位元組，兩個階段各走 \\( N - 1 \\) 步，每步傳 \\( S / N \\)，所以每個人送出

\\[ 2 \cdot \frac{N - 1}{N} \cdot S \\]

不管幾個人，都不到 \\( 2S \\)。人多了，每個人的通訊量幾乎不變，這是 ring all-reduce 好用的原因。代價是要走 \\( 2(N-1) \\) 步，每一步都有固定的延遲，人很多、資料很小時就不划算；NCCL 這類函式庫會依情況選擇 ring 或其他演算法。

## 重點整理

- 集合通訊每個 process 都要呼叫，就地修改張量：broadcast、all-reduce、all-gather、reduce-scatter。
- all-reduce = reduce-scatter + all-gather。
- ring all-reduce：切成 \\( N \\) 塊，每步只和鄰居交換一塊；`isend` 不等待、`recv` 等待，避免互相卡住。
- 每個人送出 \\( 2(N-1)/N \cdot S \\) 位元組，幾乎不隨人數增加。
