# temperature

## 本集目標

把「選最大」改成「照機率抽」，並用溫度控制機率分布的平或尖：溫度低接近 greedy，溫度高更隨機。

## 正文

### 照機率抽

logits 經過 softmax 是機率（第 1 章第 13 集），照機率抽一個就是第 14 集的 `torch.multinomial`。模型說「下一個字 60% 是 cat、30% 是 dog」，就六成的時候出 cat、三成 dog，文字會有變化，也不會卡在同一句。

### 溫度

抽之前先把 logits 除以**溫度** \\( \tau \\)：

\\[ p_i = \frac{e^{s_i / \tau}}{\sum_j e^{s_j / \tau}} \\]

- \\( \tau = 1 \\)：原本的分布。
- \\( \tau < 1 \\)：分數的差距被放大，大的更大，分布變尖，越接近 greedy。
- \\( \tau > 1 \\)：差距縮小，分布變平，低分的 token 也有機會。

```python
import torch

from llmz.functional import softmax

logits = torch.tensor([3.0, 2.0, 1.0, -2.0])
for t in [0.3, 1.0, 3.0]:
    print(t, softmax(logits / t))
```

執行結果：

```text
0.3 tensor([9.6437e-01, 3.4403e-02, 1.2273e-03, 5.5719e-08])
1.0 tensor([0.6623, 0.2436, 0.0896, 0.0045])
3.0 tensor([0.4134, 0.2962, 0.2123, 0.0781])
```

溫度 0.3 時第一個 token 拿走 96%，溫度 3 時四個差不多平均。

### 程式

`sample_next` 同時處理溫度與下一集的 top-p：

```python,ignore
{{#include ../../../src/llmz/generate.py:sample_next}}
```

這一集只看第一行和最後一行：除以溫度、softmax、`multinomial` 抽一個。中間的 `top_p` 區塊下一集講，`top_p=1.0` 時不會進去。

```python
import torch

from llmz.generate import sample_next

torch.manual_seed(0)
logits = torch.tensor([3.0, 2.0, 1.0, -2.0])
for t in [0.3, 1.0, 3.0]:
    draws = [sample_next(logits, temperature=t).item() for _ in range(100)]
    print(t, [draws.count(i) for i in range(4)])
```

執行結果：

```text
0.3 [98, 2, 0, 0]
1.0 [75, 18, 7, 0]
3.0 [37, 40, 15, 8]
```

抽 100 次，每個 token 被抽到的次數：溫度低幾乎都是 token 0，溫度高四個都常出現，和上面的機率一致。

### 溫度 0

`generate`（第 4 集）把 `temperature=0` 當成 greedy 的開關，因為除以 0 不行，而溫度趨近 0 的極限就是永遠選最大。

## 重點整理

- 抽樣：softmax 後 `torch.multinomial`，文字有變化、不卡重複。
- 溫度：logits 先除以 \\( \tau \\)；小於 1 變尖（接近 greedy），大於 1 變平（更隨機）。
- `sample_next(logits, temperature, top_p)`；`temperature=0` 代表 greedy。
