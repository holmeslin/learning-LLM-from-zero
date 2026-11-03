# top-p 取樣

## 本集目標

只在「機率最高的那一小群 token」裡抽，把尾巴那些機率很低、通常是胡說的 token 砍掉。

## 正文

### 尾巴的問題

詞彙表一萬個 token，每一步模型給前幾個合理的選擇高機率，剩下九千多個各分到一點點。單獨看每個都小，加起來卻可能有 10%、20%：十步裡總有一兩步會抽到一個完全不合理的字，接下來的句子就歪了。溫度調低可以壓尾巴，但也會把前幾名的差距拉大，變得單調。

### 只留累積到 p 的那群

**top-p**（也叫 nucleus sampling）：把 token 照機率由大到小排，從最大的開始累加，加到剛好超過 p 為止，只在這群裡抽（機率重新正規化）。p = 0.9 的意思是「只看能湊到九成機率的那幾個」。模型很確定時這群可能只有一兩個 token，不確定時可能有幾十個，大小自動調整，這是它比「固定取前 k 個」好的地方。

```python,ignore
{{#include ../../../src/llmz/generate.py:sample_next}}
```

中間那段：

1. `torch.sort(descending=True)` 把機率由大到小排，同時拿到原本的索引。
2. `torch.cumsum` 累加。
3. `cumulative - sorted_probs < top_p`：這個 token **之前**的累積還沒到 p，就留下。寫成「減掉自己」是為了讓剛好跨過 p 的那一個也被留下，而且第一個永遠會留（它之前的累積是 0）。
4. `torch.zeros_like(probs).scatter(-1, 索引, 值)` 把留下的機率放回原本的位置，其餘是 0；再除以總和正規化。

```python
import torch

from llmz.functional import softmax
from llmz.generate import sample_next

torch.manual_seed(0)
logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0])
probs = softmax(logits)
print(probs)
print(torch.cumsum(probs, dim=-1))
for p in [0.5, 0.9, 1.0]:
    draws = [sample_next(logits, top_p=p).item() for _ in range(200)]
    print(p, sorted(set(draws)))
```

執行結果：

```text
tensor([0.6364, 0.2341, 0.0861, 0.0317, 0.0117])
tensor([0.6364, 0.8705, 0.9567, 0.9883, 1.0000])
0.5 [0]
0.9 [0, 1, 2]
1.0 [0, 1, 2, 3, 4]
```

累積機率：第一個 token 0.64，前兩個 0.87，前三個 0.96。p = 0.5 時只有 token 0 留下（它之前的累積是 0 < 0.5，第二個之前的累積 0.64 已超過）；p = 0.9 時留三個（前兩個累積 0.87 < 0.9，所以第三個也留）；p = 1 全留。

### 怎麼選

本書生成 TinyStories 用 temperature 0.8、top-p 0.9 左右：有變化，又不太會出怪字。這兩個參數沒有標準答案，下一集生成時自己調著看。

## 重點整理

- top-p：機率排序、累加到 p，只在這群裡抽；群的大小隨模型的確定程度自動調整。
- `torch.sort`／`torch.cumsum`／`scatter` 三步；「累積減自己 < p」保證第一個一定留、跨過 p 的那個也留。
- 常用設定：temperature 0.8、top-p 0.9。
