# `state_dict` 儲存與載入

## 本集目標

用 `state_dict()` 把模型參數存成檔案，之後用 `load_state_dict()` 載回來，模型就能接著用。

## 正文

### 參數就是一個字典

`nn.Module` 把所有參數整理成一個字典，鍵是參數名稱、值是張量，叫 `state_dict`：

```python
import torch
from torch import nn


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(2.9314))
        self.b = nn.Parameter(torch.tensor(2.0332))

    def forward(self, x):
        return self.w * x + self.b


model = LinearModel()
print(model.state_dict())
```

執行結果：

```text
OrderedDict({'w': tensor(2.9314), 'b': tensor(2.0332)})
```

這裡直接把上一集訓練出來的 `w`、`b` 填進去，省得再跑 300 步。

### 存檔與載回

`torch.save` 把任何張量組成的東西寫成檔案，`torch.load` 讀回來：

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


trained = LinearModel()
with torch.no_grad():
    trained.w.fill_(2.9314)
    trained.b.fill_(2.0332)
torch.save(trained.state_dict(), "linear.pt")

fresh = LinearModel()
print(fresh(torch.tensor([0.5])))
fresh.load_state_dict(torch.load("linear.pt"))
print(fresh(torch.tensor([0.5])))
print(trained(torch.tensor([0.5])))
```

執行結果：

```text
tensor([0.], grad_fn=<AddBackward0>)
tensor([3.4989], grad_fn=<AddBackward0>)
tensor([3.4989], grad_fn=<AddBackward0>)
```

`fresh` 一開始參數全 0，預測是 0；`load_state_dict` 之後和 `trained` 算出一樣的值。檔案叫什麼都行，習慣用 `.pt`。`fill_` 是就地把張量填成同一個數，因為參數有 `requires_grad`，改數值要像第 2 集一樣包在 `no_grad()` 裡。

### 兩個規則

- `load_state_dict` 要求字典裡的鍵與形狀和模型完全對得上，多一個、少一個、形狀不同都會報錯。這是好事：載錯模型會立刻知道。
- 存的是**參數**，不是模型的程式碼。載入時要先用同一個類別建出模型。

第 5 章的 checkpoint 除了模型，還會把優化器的狀態和目前第幾步一起存，用的也是 `torch.save` 與 `state_dict`。

## 重點整理

- `model.state_dict()` 是「參數名稱 → 張量」的字典。
- `torch.save(model.state_dict(), 路徑)` 存檔，`model.load_state_dict(torch.load(路徑))` 載回。
- 載入前要先用同一個類別建好模型；鍵與形狀必須完全一致。
- 就地改參數數值（`fill_`、`-=`）要在 `torch.no_grad()` 裡做。
