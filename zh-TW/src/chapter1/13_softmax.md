# softmax

## 本集目標

把一排任意的分數變成機率分布：自己用 `exp` 和 `sum` 寫出 softmax，並認識它的三個性質。

## 正文

### 模型吐出來的是分數

語言模型最後一步，是對詞彙表裡**每一個** token 給一個分數，分數越高代表模型越覺得下一個字是它。這些分數叫 **logits**，可以是任何實數，有正有負。但我們想要的是「每個字的機率」：全部是正數、加起來是 1。把分數變成機率的函式就叫 softmax。

### 兩步：取 exp，再除以總和

```python
import torch

scores = torch.tensor([2.0, 1.0, 0.1])
e = torch.exp(scores)
p = e / e.sum()
print(e)
print(p)
print(p.sum())
```

執行結果：

```text
tensor([7.3891, 2.7183, 1.1052])
tensor([0.6590, 0.2424, 0.0986])
tensor(1.0000)
```

- `torch.exp` 對每個分數算 \\( e^{x} \\)。不管分數是正是負，結果一定是正數。
- 除以總和之後，三個數加起來是 1。顯示成 `1.0000` 而不是 `1.`，是因為浮點數加總差了千萬分之一左右的誤差，正常。

寫成公式就是：

\\[ p_i = \frac{e^{s_i}}{\sum_j e^{s_j}} \\]

分子是「我的 exp」，分母是「大家的 exp 加總」。

### 三個性質

把 softmax 包成函式，試幾件事：

```python
import torch

def softmax(x):
    e = torch.exp(x)
    return e / e.sum()

scores = torch.tensor([2.0, 1.0, 0.1])
print(softmax(scores))
print(softmax(scores + 10))
print(softmax(scores * 3))
print(softmax(scores / 3))
```

執行結果：

```text
tensor([0.6590, 0.2424, 0.0986])
tensor([0.6590, 0.2424, 0.0986])
tensor([0.9495, 0.0473, 0.0032])
tensor([0.4450, 0.3188, 0.2362])
```

1. **順序不變，差距放大**。分數 2.0 和 1.0 只差 1，機率卻差了 \\( e \approx 2.7 \\) 倍。這是 exp 的效果。
2. **整體平移沒差**。所有分數同時加 10，機率完全一樣，因為分子分母同時多乘了 \\( e^{10} \\)，約掉了。下一集的「取樣」不受影響，第 16 集還會靠這個性質救命。
3. **放大分數更「尖」，縮小更「平」**。乘 3 之後第一名拿走 95%；除以 3 之後三個差不多。第 6 章生成文字時會用一個叫 temperature 的數字來除，就是在調這個。

### 一次處理很多列

模型一次會對很多個位置各算一排分數，形狀是 `(位置數, 詞彙表大小)`。要對**每一列**各做一次 softmax，就是第 5 集的 `dim` 與 `keepdim`：

```python
import torch

def softmax(x, dim=-1):
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)

scores = torch.tensor([[2.0, 1.0, 0.1],
                       [0.0, 0.0, 0.0]])
p = softmax(scores)
print(p)
print(p.sum(dim=-1))
```

執行結果：

```text
tensor([[0.6590, 0.2424, 0.0986],
        [0.3333, 0.3333, 0.3333]])
tensor([1.0000, 1.0000])
```

`dim=-1` 是最後一維，也就是詞彙表那一維。每一列各自加總為 1；第二列三個分數都一樣，所以機率均分。

### 還差一步

這個版本有一個致命問題：分數稍微大一點，`exp` 就會爆掉。第 16 集會修好它，修好的版本才會放進 `llmz`。

## 重點整理

- softmax 把一排分數（logits）變成機率：先 `exp` 全部變正，再除以總和讓加總為 1。
- 三個性質：順序不變但差距被放大；整體加同一個數結果不變；放大分數變尖、縮小變平。
- 多列時用 `e / e.sum(dim=-1, keepdim=True)`，每一列各自是一個機率分布。
