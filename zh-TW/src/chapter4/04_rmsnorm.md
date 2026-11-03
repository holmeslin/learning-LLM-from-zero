# RMSNorm

## 本集目標

寫出 `RMSNorm`：把每個 token 的向量除以它的均方根，讓每一層的輸入大小固定，深的模型才訓練得起來。

## 正文

### 問題：大小會飄

上一集 Linear 的初始化讓輸出大小和輸入差不多，但那只是「差不多」，而且訓練幾千步之後參數早就不是初始值了。幾十層疊下來，有的 token 向量變成 100、有的變成 0.01，後面的層就很難對所有輸入都做對。解法：在每個子層之前，把向量**重新縮放到固定大小**。

### 均方根

一個向量 \\( x \\) 的均方根（root mean square）：

\\[ \text{rms}(x) = \sqrt{\frac{1}{d} \sum_i x_i^2 + \epsilon} \\]

把 \\( x \\) 除以它，結果的均方根就是 1。\\( \epsilon \\) 是個很小的數（\\( 10^{-5} \\)），防止 \\( x \\) 全為 0 時除以 0。最後再乘一個可學的向量 `weight`（初始全 1），讓模型需要的話可以把某些維度放大縮小回來。

```python,ignore
{{#include ../../../src/llmz/layers.py:rmsnorm}}
```

`dim=-1` 是最後一個維度，也就是 `d_model`：每個 token 自己算自己的均方根，和 batch 裡其他 token 無關。`keepdim=True` 讓形狀對得上後面的除法（第 1 章第 5 集）。開頭先轉成 `float32`、結尾轉回原本的 dtype：第 7 章用半精度訓練時，平方和在低精度下會溢位，所以這一步固定用高精度算。

```python
import torch

from llmz.layers import RMSNorm

torch.manual_seed(0)
norm = RMSNorm(4)
x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                  [100.0, 0.0, 0.0, 0.0]])
y = norm(x)
print(y)
print(torch.sqrt((x * x).mean(dim=-1)))
print(torch.sqrt((y * y).mean(dim=-1)))
```

執行結果：

```text
tensor([[0.3651, 0.7303, 1.0954, 1.4606],
        [2.0000, 0.0000, 0.0000, 0.0000]], grad_fn=<MulBackward0>)
tensor([ 2.7386, 50.0000])
tensor([1.0000, 1.0000], grad_fn=<SqrtBackward0>)
```

兩列的均方根原本差三十幾倍，正規化之後都是 1（第二列是 0.99998，因為 \\( \epsilon \\) 的關係）。注意它**沒有**減掉平均值，也不要求元素加總為 0，只管大小，所以叫 RMSNorm 而不是更早的 LayerNorm；少一步、效果一樣，現在的 LLM 幾乎都用它。

### 放在哪裡

第 13 集的 block 會在 attention 之前、FFN 之前各放一個 RMSNorm（pre-norm），模型最後出 logits 之前再放一個。第 6 章第 5、6 集會做實驗：拿掉它、或移到子層之後，看訓練會怎樣。

## 重點整理

- RMSNorm 把每個 token 的向量除以自己的均方根（沿 `d_model`），再乘可學的 `weight`。
- 目的是固定每個子層輸入的大小，深的模型才穩。
- 不減平均值；平方和用 `float32` 算。
- 位置：每個子層之前，以及輸出 logits 之前。
