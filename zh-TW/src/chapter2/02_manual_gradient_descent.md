# 手動梯度下降

## 本集目標

把第 1 章的五步梯度下降改成用 autograd 算梯度，學會 `torch.no_grad()` 與 `.grad.zero_()` 這兩個訓練迴圈裡一定會出現的東西。

## 正文

### 梯度會累加

上一集最後提到 `.grad` 是累加的。先親眼看一次：

```python
import torch

w = torch.tensor([1.0, 2.0, -3.0], requires_grad=True)
(w ** 2).sum().backward()
print(w.grad)
(w ** 2).sum().backward()
print(w.grad)
w.grad.zero_()
print(w.grad)
```

執行結果：

```text
tensor([ 2.,  4., -6.])
tensor([  4.,   8., -12.])
tensor([0., 0., 0.])
```

第二次 `backward()` 之後梯度變成兩倍，因為新算出來的 \\( 2w \\) 加到了舊的上面。`w.grad.zero_()` 把它清成 0；結尾的底線是 PyTorch 的慣例，代表「直接改這個張量本身」。所以訓練時每一步算完梯度、更新完參數之後，都要把梯度歸零，不然下一步會用到上一步的殘留。

### 更新參數時不要被記錄

第 1 章第 12 集的更新是 `w = w - lr * grad`。改用 autograd 之後有個麻煩：`w` 標了 `requires_grad`，這一行本身也會被記進圖裡，而且會做出一個新的 `w`，原本的梯度就跟丟了。我們要的是「就地改 `w` 的數值，而且這一步不要記錄」，寫法是把更新包在 `torch.no_grad()` 裡：

```python
import torch

w = torch.tensor([1.0, 2.0, -3.0], requires_grad=True)
lr = 0.1
for step in range(5):
    f = (w ** 2).sum()
    f.backward()
    with torch.no_grad():
        w -= lr * w.grad
    w.grad.zero_()
    print(step, round(f.item(), 4), round((w ** 2).sum().item(), 4))
```

執行結果：

```text
0 14.0 8.96
1 8.96 5.7344
2 5.7344 3.67
3 3.67 2.3488
4 2.3488 1.5032
```

每一行印出這一步開始時的 \\( f \\) 和更新後的 \\( f \\)，數字和第 1 章第 12 集手算梯度的版本完全一樣：從 14 降到 1.5。差別只在 `grad = 2 * w` 那一行不見了，換成 `f.backward()` 之後讀 `w.grad`。

`with torch.no_grad():` 區塊裡的運算不會被記錄、也不會產生 `grad_fn`，`w -= ...` 是就地減（和 `zero_()` 一樣直接改原張量），所以 `w` 還是同一個張量，`.grad` 也還在。

### 四個動作

整理一下迴圈裡的順序：

1. 算函數值（之後叫 forward）。
2. `backward()` 算梯度。
3. 在 `no_grad()` 裡用梯度更新參數。
4. 把梯度歸零。

順序 3 和 4 可以對調（先歸零再更新就錯了，梯度沒了），其他不能動。下一集先把參數裝進 `nn.Module`，第 4 集再把這四步定成全書的固定句型。

## 重點整理

- `.grad` 累加，每一步更新完要 `w.grad.zero_()`；結尾底線代表就地修改。
- 更新參數要寫在 `with torch.no_grad():` 裡，這一步才不會被記進圖，`w` 也保持是同一個張量。
- 迴圈順序：算函數值 → `backward()` → `no_grad` 裡更新 → 梯度歸零。
