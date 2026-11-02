# cross-entropy 與 perplexity

## 本集目標

用「正確答案的 log 機率」定義模型猜得多差（cross-entropy loss），會對一整批題目算出平均 loss，並理解 perplexity 在說什麼。

## 正文

### 猜對的機率越大，loss 越小

模型對三個候選字給了機率 `[0.7, 0.2, 0.1]`。如果正確答案是第 0 個字，它猜得不錯；如果是第 2 個字，它猜得很差。我們要一個數字衡量「有多差」，而且越差越大。答案是：**正確答案那一格的機率，取負 log**。

```python
import torch

p = torch.tensor([0.7, 0.2, 0.1])
print(-torch.log(p[0]))
print(-torch.log(p[2]))
```

執行結果：

```text
tensor(0.3567)
tensor(2.3026)
```

上一集說過 log 機率 ≤ 0，加個負號就變成 ≥ 0 的數字：猜對的機率是 1 時 loss 是 0（完美），機率越接近 0 loss 越大（可以大到無限）。其他兩格的機率完全不管，只看正確答案那一格。這個數字叫 **cross-entropy**，也叫 negative log-likelihood（負對數概似）。

### 一次算一整批

訓練時一次有很多題。每一列是一題的機率分布，`targets` 是每題的正確答案索引。用第 4 集「兩個索引張量」的技巧，一行撈出每題正確答案那一格：

```python
import torch

p = torch.tensor([[0.7, 0.1, 0.1, 0.1],
                  [0.25, 0.25, 0.25, 0.25],
                  [0.1, 0.1, 0.1, 0.7]])
targets = torch.tensor([0, 1, 3])
picked = p[torch.arange(3), targets]
print(picked)
nll = -torch.log(picked)
print(nll)
print(nll.mean())
```

執行結果：

```text
tensor([0.7000, 0.2500, 0.7000])
tensor([0.3567, 1.3863, 0.3567])
tensor(0.6999)
```

第二題模型完全沒把握（四個選項各 0.25），貢獻的 loss 是 \\( -\log 0.25 = \log 4 \approx 1.386 \\)。三題平均 0.70，這就是這一批的 loss。公式：

\\[ L = -\frac{1}{N} \sum_{i=1}^{N} \log p_i[y_i] \\]

\\( p_i[y_i] \\) 是第 i 題在正確答案 \\( y_i \\) 那一格的機率。

### 從分數開始算

模型給的其實是分數（logits），所以完整流程是：softmax 變機率、撈出正確答案那格、取負 log、平均：

```python
import torch

def softmax(x, dim=-1):
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)

logits = torch.tensor([[2.0, 1.0, 0.1],
                       [0.5, 2.5, 0.0]])
targets = torch.tensor([0, 1])
p = softmax(logits)
loss = -torch.log(p[torch.arange(2), targets]).mean()
print(p)
print(loss)
```

執行結果：

```text
tensor([[0.6590, 0.2424, 0.0986],
        [0.1112, 0.8214, 0.0674]])
tensor(0.3069)
```

兩題的正確答案機率分別是 0.659 和 0.821，都算猜得不錯，平均 loss 0.31。訓練的目標就是調整參數，讓這個數字越小越好。

### perplexity：平均在幾個字裡猜

loss 是個 log 尺度的數字，不太直觀。把它 exp 回去，得到 **perplexity**（困惑度）：

```python
import torch

uniform_loss = -torch.log(torch.tensor(0.25))
print(uniform_loss)
print(torch.exp(uniform_loss))
print(torch.exp(torch.tensor(0.3069)))
```

執行結果：

```text
tensor(1.3863)
tensor(4.)
tensor(1.3592)
```

四個選項完全不知道選哪個，loss 是 \\( \log 4 \\)，perplexity 剛好是 4：「模型像是在 4 個字裡面亂猜」。剛才的 loss 0.3069 對應 perplexity 1.36：「像是在 1.36 個字裡猜」，很有把握。

訓練語言模型時，剛開始參數是隨機的，對詞彙表裡每個字的機率差不多，所以起始 loss 大約是 \\( \log(\text{詞彙表大小}) \\)：詞彙表一萬個字就是 \\( \log 10000 \approx 9.2 \\)，perplexity 一萬。第 5 章你會看著它從 9 點多一路往下掉。

## 重點整理

- cross-entropy loss 只看正確答案那一格的機率，取負 log；機率 1 是 0，機率越小越大。
- 一批題目用 `p[torch.arange(N), targets]` 撈出每題正確答案的機率，取負 log 後平均。
- 完整流程：logits → softmax → 撈正確答案 → 負 log → 平均。
- perplexity 是 `exp(loss)`，可以讀成「平均像在幾個字裡猜」；完全不知道時等於詞彙表大小。
