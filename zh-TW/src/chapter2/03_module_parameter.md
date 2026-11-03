# `nn.Module` 與 `nn.Parameter`

## 本集目標

把模型的參數裝進一個 `nn.Module` 類別，讓 PyTorch 幫我們管理「這個模型有哪些參數」。

## 正文

### 參數散落各處的問題

上一集只有一個 `w`。真正的模型有幾十、幾百個張量，各自 `requires_grad`、各自更新、各自歸零，很快就會漏掉一個。PyTorch 的做法是把模型寫成一個**類別**（Python 的 `class`，這裡第一次用到：它把資料和函式包成一個東西），繼承 `nn.Module`，參數用 `nn.Parameter` 宣告：

```python
import torch
from torch import nn


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.w * x + self.b


model = LinearModel()
print(model)
print(model(torch.tensor([1.0, 2.0, 3.0])))
```

執行結果：

```text
LinearModel()
tensor([0., 0., 0.], grad_fn=<AddBackward0>)
```

逐行看：

- `class LinearModel(nn.Module)`：我們的模型是一種 `nn.Module`。
- `__init__` 第一行一定要 `super().__init__()`，讓 `nn.Module` 先把自己準備好。
- `nn.Parameter(...)` 把一個張量宣告成參數：它自動 `requires_grad=True`，而且會被登記在這個模型底下。
- `forward` 寫「輸入進來怎麼算出輸出」。呼叫時寫 `model(x)`，不要直接呼叫 `model.forward(x)`，PyTorch 會在中間做一些額外的事。

這個模型算的是 \\( y = wx + b \\)，一條直線，`w` 是斜率、`b` 是截距，兩個都從 0 開始。

### 模型知道自己有哪些參數

`nn.Module` 最大的好處：它會幫你列出所有 `nn.Parameter`，不管藏在幾層底下。

```python
import torch
from torch import nn


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.w * x + self.b


model = LinearModel()
for name, p in model.named_parameters():
    print(name, p)
```

執行結果：

```text
w Parameter containing:
tensor(0., requires_grad=True)
b Parameter containing:
tensor(0., requires_grad=True)
```

`model.parameters()` 只給參數本身，`model.named_parameters()` 連名字一起給。下一集的更新與歸零就靠 `for p in model.parameters():` 一次處理全部，再也不會漏。

### 之後還會用到的兩個好處

- `model.to(device)` 一次把所有參數搬到 GPU（第 1 章第 9 集的 `device`），不用一個一個 `.to`。
- 第 6 集會看到 `model.state_dict()`，把所有參數打包成一個字典存檔。

一個 `nn.Module` 裡面也可以放別的 `nn.Module`（第 4 章的 Transformer 就是一層包一層），參數一樣會被自動收集。

## 重點整理

- 模型寫成繼承 `nn.Module` 的類別，`__init__` 先 `super().__init__()`，參數用 `nn.Parameter` 宣告（自動 `requires_grad=True`）。
- `forward` 定義輸入到輸出的計算；呼叫時寫 `model(x)`。
- `model.parameters()`／`model.named_parameters()` 列出全部參數，不會漏。
- `model.to(device)` 一次搬全部參數。
