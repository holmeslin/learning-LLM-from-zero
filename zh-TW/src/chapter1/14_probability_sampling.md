# 機率、log 機率與取樣

## 本集目標

會從機率分布裡抽出一個結果（取樣）或直接選最大的（argmax），並理解為什麼模型都用 log 機率而不是機率。

## 正文

### 從機率分布抽一個字

有了機率，模型要怎麼決定下一個字？一種做法是照機率**抽**：機率 0.7 的字大約七成的時候被抽到。`torch.multinomial` 就是在做這件事：

```python
import torch

p = torch.tensor([0.7, 0.2, 0.1])
torch.manual_seed(0)
samples = torch.multinomial(p, 20, replacement=True)
print(samples)
print((samples == 0).float().mean())
```

執行結果：

```text
tensor([2, 1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
tensor(0.7500)
```

`torch.multinomial(p, 20, replacement=True)` 依照 `p` 抽 20 次，回傳抽到的索引；`replacement=True` 代表每次抽完放回去，可以重複。20 次裡索引 0 出現了 15 次，四分之三，和機率 0.7 差不多（次數不會每次都剛好，抽越多次越接近）。

生成文字時，模型每一步就是對整個詞彙表做一次這種抽樣，抽到哪個字就寫哪個字（第 6 章）。

### 或是直接選最大的

另一種做法：永遠選機率最高的，叫 argmax（「最大值的位置」）：

```python
import torch

p = torch.tensor([0.1, 0.6, 0.3])
print(torch.argmax(p))
print(p[p.argmax()])
```

執行結果：

```text
tensor(1)
tensor(0.6000)
```

`argmax` 回傳的是**索引**（第 1 個），不是機率值本身。這種每一步都選最大的生成方式叫 greedy，結果固定但容易一直重複同樣的話，第 6 章會比較。

### 機率相乘會小到消失

一句話的機率，是每個字的機率相乘。十個字每個機率 0.1，整句就是 \\( 0.1^{10} \\)。字一多，數字就小到電腦存不下：

```python
import torch

p = torch.tensor(0.1)
print(p ** 10)
print(p ** 30)
print(p ** 50)
```

執行結果：

```text
tensor(1.0000e-10)
tensor(1.0000e-30)
tensor(0.)
```

`float32` 最小只能表示到大約 \\( 10^{-45} \\)，\\( 0.1^{50} \\) 直接變成 0。一旦變 0，後面不管怎麼算都是 0，什麼資訊都沒了。真實的句子有幾百個字，這問題一定會發生。

### 用 log 把乘變加

解法：不存機率，存機率的 **log**。log 有一個性質：\\( \log(a \times b) = \log a + \log b \\)。相乘變相加，數字就不會縮到消失：

```python
import torch

p = torch.tensor([0.7, 0.2, 0.1])
print(torch.log(p))
print(torch.log(p).sum())
print(torch.log(p.prod()))
print(50 * torch.log(torch.tensor(0.1)))
```

執行結果：

```text
tensor([-0.3567, -1.6094, -2.3026])
tensor(-4.2687)
tensor(-4.2687)
tensor(-115.1293)
```

- `torch.log` 是自然對數（以 \\( e \\) 為底）。機率在 0 到 1 之間，log 之後一定 **≤ 0**：機率越接近 1，log 越接近 0；機率越小，log 越負。
- 三個 log 相加，等於三個機率相乘（`.prod()`）再取 log。
- 五十個 0.1 相乘是 \\( 10^{-50} \\)，存不下；但它的 log 是 -115，一個普通的數字。

所以模型內部算的都是 **log 機率**（log-probability）。要回到機率時，`torch.exp` 一下就好：\\( e^{\log p} = p \\)。下一集的 loss 就直接定義在 log 機率上。

## 重點整理

- `torch.multinomial(p, n, replacement=True)` 依機率抽 `n` 個索引；`torch.argmax(p)` 回傳機率最大的索引。
- 機率相乘會小到 `float32` 存不下（變成 0）。
- `torch.log` 把相乘變相加；log 機率一定 ≤ 0，越接近 0 代表機率越大。
- 模型內部用 log 機率運算，需要時用 `torch.exp` 變回機率。
