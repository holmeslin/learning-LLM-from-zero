# 從頭寫 SGD（`Optimizer` 基底）

## 本集目標

把第 2 章的簡易 `SGD` 改寫成繼承 `torch.optim.Optimizer` 的版本，學會 `param_groups` 與 `state` 這兩個之後 AdamW 會用到的東西。

## 正文

### 為什麼要繼承

第 2 章第 4 集的 `SGD` 只有一個串列和一個 `lr`。PyTorch 的 `Optimizer` 基底類別多給三樣東西：

- `param_groups`：參數分組，每組可以有自己的超參數（學習率、weight decay……），第 8 集會每一步改學習率，就是改這裡。
- `state`：每個參數專屬的狀態字典，AdamW 要記住動量就放這裡。
- `state_dict()`／`load_state_dict()`：把上面兩樣存檔、載回，第 7 集的 checkpoint 需要。

本書的規則是不用 `torch.optim` 的現成優化器，但這個基底類別只負責管帳，算法全是我們自己寫的，可以用。

```python,ignore
{{#include ../../../src/llmz/optim.py:sgd}}
```

- `super().__init__(params, {"lr": lr})`：第二個引數是預設的超參數字典，基底類別會把 `params` 包成一個 group，再把這些超參數放進去。
- `step` 兩層迴圈：每個 group、裡面每個參數。`p.grad is None` 的參數（這一步沒用到）跳過。
- `@torch.no_grad()` 裝飾器讓整個 `step` 都在 `no_grad` 裡執行，和第 2 章第 2 集的 `with torch.no_grad():` 一樣，只是寫在函式上。

### 用法不變

```python
import torch
from torch import nn

from llmz.optim import SGD


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.w * x + self.b


x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([3.0, 5.0, 7.0])
model = LinearModel()
opt = SGD(model.parameters(), lr=0.1)
for step in range(3):
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(step, round(loss.item(), 4), round(model.w.item(), 4), round(model.b.item(), 4))
print(opt.param_groups[0]["lr"], len(opt.param_groups[0]["params"]))
```

執行結果：

```text
0 27.6667 2.2667 1.0
1 0.3319 2.0178 0.8933
2 0.0053 2.0439 0.9076
0.1 2
```

前三行和第 2 章第 4 集一模一樣。最後一行看 `param_groups`：一個 group，`lr` 0.1，兩個參數。`zero_grad()` 是基底類別給的，預設把 `.grad` 設成 `None`，和我們第 2 章自己寫的一樣。

`tests/test_optim.py` 拿 `torch.optim.SGD` 跑同樣的步數比對參數，完全相同。

## 重點整理

- 繼承 `torch.optim.Optimizer`：`super().__init__(params, 預設超參數字典)`，基底類別提供 `param_groups`、`state`、`zero_grad()`、`state_dict()`。
- `step` 走訪 `param_groups` 裡每個參數，用 `@torch.no_grad()` 包住。
- 用法和第 2 章的簡易版完全相同。
