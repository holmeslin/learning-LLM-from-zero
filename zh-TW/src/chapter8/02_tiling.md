# tiling

## 本集目標

理解分塊計算（tiling）：把大矩陣切成小塊，一次只把幾塊放進快速的晶片記憶體裡算，減少來回搬資料的量。

## 正文

### 晶片上的小記憶體

第 7 章第 1 集說過，GPU 晶片上有一小塊很快的 SRAM，每組計算單元只有幾百 KB；張量平常放在大而慢的顯示卡記憶體。kernel 的寫法決定資料要在兩者之間搬幾趟。

以矩陣乘法 \\( C = AB \\)（都是 \\( n \times n \\)）為例。最直接的做法是：每算一個 \\( C_{ij} \\)，就把 \\( A \\) 的第 \\( i \\) 列和 \\( B \\) 的第 \\( j \\) 行讀進來，做 \\( n \\) 次乘加。每個輸出要讀 \\( 2n \\) 個數，總共讀 \\( 2n^3 \\) 個數。

### 切成小塊

改成一次處理一塊 \\( b \times b \\) 的輸出：把 \\( A \\) 的一條 \\( b \times b \\) 小塊和 \\( B \\) 對應的小塊讀進晶片，算完它們對這塊輸出的貢獻，加到累加器裡，再換下一對小塊。

```python
import torch

torch.manual_seed(0)
n, b = 8, 4
A = torch.randn(n, n)
B = torch.randn(n, n)
C = torch.zeros(n, n)
for i in range(0, n, b):  # 輸出的第幾塊列
    for j in range(0, n, b):  # 輸出的第幾塊行
        acc = torch.zeros(b, b)  # 這塊輸出的累加器，放在晶片上
        for t in range(0, n, b):  # 沿著相乘的維度一塊一塊加
            acc += A[i : i + b, t : t + b] @ B[t : t + b, j : j + b]
        C[i : i + b, j : j + b] = acc  # 算完才寫回去，只寫一次
print(torch.allclose(C, A @ B, atol=1e-6))
```

執行結果：

```text
True
```

結果和 `A @ B` 一樣。差別在搬運：每讀進一個 \\( b \times b \\) 小塊，它會被用 \\( b \\) 次（和另一塊的 \\( b \\) 行或列相乘），所以總讀取量從 \\( 2n^3 \\) 降到 \\( 2n^3 / b \\)。

```python
n, b = 4096, 64
naive_reads = 2 * n**3
tiled_reads = 2 * n**3 // b
print(f"{naive_reads:.2e} → {tiled_reads:.2e}，少了 {naive_reads // tiled_reads} 倍")
```

執行結果：

```text
1.37e+11 → 2.15e+09，少了 64 倍
```

塊越大，重複使用越多，但塊要放得進晶片上的 SRAM。實際的矩陣乘法 kernel 都是這樣分塊的，這也是大矩陣乘法能卡在計算而不是卡在記憶體的原因。

### 套到 attention 上

attention 的兩個矩陣乘法 \\( QK^\top \\) 和 \\( PV \\)（\\( P \\) 是 softmax 後的機率）都能分塊。真正的麻煩在中間的 softmax：要算第 \\( i \\) 列的機率，得先知道整列分數的最大值與 exp 總和，而這需要看過整列 \\( T \\) 個分數。如果只拿到一小塊分數，還不能算出這一塊的機率。

如果 softmax 也能「看一塊、更新一次」，整個 attention 就能一塊一塊算：算出一小塊分數、更新 softmax、乘上對應的 \\( V \\) 小塊、累加，分數用完就丟。下一集就是這個工具。

## 重點整理

- tiling：把計算切成放得進晶片 SRAM 的小塊，每塊讀進來後重複使用很多次。
- 分塊的矩陣乘法結果不變，讀取量從 \\( 2n^3 \\) 降到 \\( 2n^3 / b \\)。
- attention 的兩個矩陣乘法都能分塊，卡住的是需要看完整列才能算的 softmax。
