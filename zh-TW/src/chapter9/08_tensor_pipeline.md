# tensor / pipeline parallelism（原理）

## 本集目標

理解另外兩種切法：把一層的矩陣切給好幾個 rank（tensor parallel），以及把不同層放在不同 rank（pipeline parallel），並用兩個 rank 驗證 tensor parallel 的 FFN 算出一樣的結果。

## 正文

### tensor parallel：切矩陣

data parallel 與 FSDP 都是「每個 rank 看不同的資料、算完整的模型」。tensor parallel 反過來：每個 rank 看**同樣的資料**，但只算每一層的一部分。

以 FFN 為例（用第 4 章 SiLU 版的寫法，\\( W_1 \\) 形狀 `(d_ff, d_model)`、\\( W_2 \\) 形狀 `(d_model, d_ff)`）：

\\[ y = \mathrm{silu}(x W_1^\top)\\, W_2^\top \\]

1. 把 \\( W_1 \\) 依**輸出**維度切成兩半（`d_ff` 那一維）。每個 rank 算出中間結果 `h` 的一半欄位。silu 是逐元素的，各算各的那一半不需要交換。
2. 把 \\( W_2 \\) 依**輸入**維度切成兩半（也是 `d_ff` 那一維），每個 rank 用自己那一半的 `h` 乘自己那一半的 \\( W_2 \\)，得到 `y` 的**部分和**。
3. 一次 all-reduce 把部分和加起來，每個 rank 都拿到完整的 `y`。

第一個矩陣按欄切（column parallel）、第二個按列切（row parallel），中間完全不通訊，整個 FFN 只要一次 all-reduce。attention 也能這樣切：每個 rank 負責幾個 head。

```python
import torch
import torch.distributed as dist

from llmz.dist import launch, print_in_order
from llmz.layers import silu


def run(rank, world_size):
    torch.manual_seed(0)
    d_model, d_ff = 8, 12
    x = torch.randn(3, d_model)  # 每個 rank 拿同樣的輸入
    w1 = torch.randn(d_ff, d_model)
    w2 = torch.randn(d_model, d_ff)
    full = silu(x @ w1.T) @ w2.T  # 單機答案

    part = d_ff // world_size
    my_w1 = w1[rank * part : (rank + 1) * part]  # 依輸出維度切
    my_w2 = w2[:, rank * part : (rank + 1) * part]  # 依輸入維度切
    y = silu(x @ my_w1.T) @ my_w2.T  # 部分和
    dist.all_reduce(y)
    print_in_order(f"存 {my_w1.numel() + my_w2.numel()} 個參數，結果正確：{torch.allclose(y, full, atol=1e-5)}")


if __name__ == "__main__":
    launch(run, 2)
```

執行結果：

```text
[rank 0] 存 96 個參數，結果正確：True
[rank 1] 存 96 個參數，結果正確：True
```

每個 rank 只存一半的參數，算一半的 FLOPs。代價是**每一層**的 forward 都要一次 all-reduce（backward 也一次），傳的是 activations，大小是 `(B, T, d_model)`。這比 data parallel 每步一次的梯度通訊頻繁得多，所以 tensor parallel 只在連線最快的同一台機器內使用（通常 8 張卡）。

### pipeline parallel：切層

另一種切法是把層分給不同的 rank：rank 0 放第 1～8 層、rank 1 放第 9～16 層……。forward 時 rank 0 算完把 activations 送給 rank 1（第 3 集的 send／recv），backward 時梯度再反方向送回來。每個 rank 只存自己那幾層，通訊只發生在相鄰的 rank 之間，量也不大，適合跨機器。

問題在於等待：rank 1 要等 rank 0 算完才能開始，rank 0 要等後面的 backward 傳回來才能算自己的 backward。如果一次只送一整個 batch，任何時刻只有一個 rank 在工作。解法是把 batch 切成 \\( m \\) 個 **micro-batch**，像生產線一樣接連送進去：rank 0 算完第一個 micro-batch 交出去，馬上算第二個。

即使這樣，生產線開頭與結尾還是有人閒著，這段閒置叫 **bubble**。\\( p \\) 個階段、\\( m \\) 個 micro-batch 時，閒置的比例大約是

\\[ \frac{p - 1}{m + p - 1} \\]

```python
p = 4
for m in [1, 4, 16, 64]:
    print(f"{m:2} 個 micro-batch：閒置 {(p - 1) / (m + p - 1):.0%}")
```

執行結果：

```text
 1 個 micro-batch：閒置 75%
 4 個 micro-batch：閒置 43%
16 個 micro-batch：閒置 16%
64 個 micro-batch：閒置 4%
```

micro-batch 越多，bubble 越小，但每個 micro-batch 變小，矩陣乘法的效率會變差（第 7 章第 8 集），activations 也要多存幾份。實務上會用更聰明的排程（例如 forward 與 backward 交錯）把 bubble 再壓低，本書不展開。

### 三種一起用

大模型訓練常說的「3D parallelism」就是三者並用，例如 1024 張卡：每台機器 8 張卡做 tensor parallel，8 台機器串成 pipeline，再複製 16 份做 data parallel（\\( 8 \times 8 \times 16 = 1024 \\)），data parallel 那一維通常還會用 ZeRO 或 FSDP 切優化器狀態。

## 重點整理

- tensor parallel：每個 rank 看同樣的資料、存每層的一部分；FFN 的 \\( W_1 \\) 按輸出切、\\( W_2 \\) 按輸入切，一次 all-reduce 加起部分和。
- tensor parallel 每層都要通訊 activations，只適合同一台機器內的快速連線。
- pipeline parallel：不同的 rank 放不同的層，相鄰 rank 之間傳 activations；用 micro-batch 減少閒置，bubble 約 \\( (p-1)/(m+p-1) \\)。
- 大規模訓練把 data、tensor、pipeline 三種並用。
