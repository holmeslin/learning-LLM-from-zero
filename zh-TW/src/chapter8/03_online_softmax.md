# online softmax

## 本集目標

學會一邊讀分數、一邊更新 softmax 的技巧：只記住目前的最大值、exp 總和與加權和，看完所有分塊後得到和一次算完一模一樣的結果。

## 正文

### 一次算完的版本

只看 attention 的一列：一個 query 對 \\( T \\) 個 key 的分數 \\( s_1, \dots, s_T \\)，以及每個 key 對應的 value。這一列的輸出是

\\[ o = \sum_j \frac{e^{s_j - m}}{\ell} v_j, \quad m = \max_j s_j, \quad \ell = \sum_j e^{s_j - m} \\]

減掉最大值 \\( m \\) 是第 1 章第 16 集的數值穩定技巧。問題是 \\( m \\) 和 \\( \ell \\) 都要看完整列才知道。

### 邊看邊修正

假設我們先看了前半段，算出了這段的最大值 \\( m \\)、總和 \\( \ell \\)、還沒除以 \\( \ell \\) 的加權和 \\( a = \sum e^{s_j - m} v_j \\)。接著看到下一段，新的最大值變成 \\( m' \\)。之前每一項都是用 \\( e^{s_j - m} \\) 算的，現在要換成 \\( e^{s_j - m'} \\)，兩者只差一個共同的倍數：

\\[ e^{s_j - m'} = e^{s_j - m} \cdot e^{m - m'} \\]

所以不用回頭重算，把舊的 \\( \ell \\) 和 \\( a \\) 都乘上 \\( e^{m - m'} \\)，再加上新一段的貢獻就好。全部看完後，\\( o = a / \ell \\)。

```python
import torch

from llmz.functional import softmax

torch.manual_seed(0)
scores = torch.tensor([0.5, -1.0, 2.0, 0.3, 3.5, -2.0, 1.0, 0.0, 5.0, -0.5])
values = torch.randn(10, 4)

m = torch.tensor(float("-inf"))  # 目前看過的最大分數
l = torch.tensor(0.0)  # 目前的 exp 總和
a = torch.zeros(4)  # 目前的加權和，還沒除以 l
for start in range(0, 10, 4):  # 每次看 4 個：0~3、4~7、8~9
    s = scores[start : start + 4]
    m_new = torch.maximum(m, s.max())
    correction = torch.exp(m - m_new)
    p = torch.exp(s - m_new)
    l = l * correction + p.sum()
    a = a * correction + p @ values[start : start + 4]
    m = m_new
    print(f"看到第 {start + len(s) - 1} 個  max={m.item():.3f}  總和={l.item():.3f}")

print(torch.allclose(a / l, softmax(scores) @ values, atol=1e-6))
print(torch.allclose(m + torch.log(l), torch.logsumexp(scores, dim=0)))
```

執行結果：

```text
看到第 3 個  max=2.000  總和=1.456
看到第 7 個  max=3.500  總和=1.441
看到第 9 個  max=5.000  總和=1.326
True
True
```

`torch.maximum(a, b)` 逐元素取兩者中較大的那個（`.max()` 則是在一個張量裡找最大值）。最大值在每一段都變大了一次（2.0、3.5、5.0），每次都靠 `correction` 把之前的總和縮小。三段看完，結果和一次算完的 `softmax(scores) @ values` 一樣。第一段開始時 \\( m = -\infty \\)，`correction` 是 \\( e^{-\infty} = 0 \\)，正好把一開始的 0 乘掉，不用特別處理。

最後一行順便算出了這一列分數的 logsumexp：\\( m + \log \ell \\)（第 1 章第 16 集）。這裡用 `torch.logsumexp` 只是為了驗證。第 5 集的 backward 會用到這個數字。

### 一次一整塊 query

上面是一個 query。attention 有很多 query，每個 query 各有自己的 \\( m \\)、\\( \ell \\)、\\( a \\)，彼此獨立，所以可以把一塊 query 疊在一起算：\\( m \\) 和 \\( \ell \\) 變成向量，\\( a \\) 變成矩陣，`p @ values` 變成矩陣乘法。下一集就是這樣寫的。

## 重點整理

- online softmax 只記三樣東西：目前的最大值 \\( m \\)、exp 總和 \\( \ell \\)、加權和 \\( a \\)。
- 最大值從 \\( m \\) 變成 \\( m' \\) 時，舊的 \\( \ell \\) 與 \\( a \\) 乘上 \\( e^{m - m'} \\) 就修正好了。
- 看完後 \\( a / \ell \\) 就是 softmax 加權的結果，\\( m + \log \ell \\) 是這一列的 logsumexp。
- 每個 query 互相獨立，可以一整塊一起更新。
