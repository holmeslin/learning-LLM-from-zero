# autograd：`requires_grad` 與 `backward()`

## 本集目標

讓 PyTorch 自動算梯度：在張量上標 `requires_grad=True`，對結果呼叫 `backward()`，梯度就會出現在 `.grad` 裡。

## 正文

### 第 1 章的梯度，交給 PyTorch 算

第 1 章第 12 集我們對 \\( f(w) = \sum w_i^2 \\) 用有限差分算出梯度 \\( 2w \\)。同一個函數，這次讓 PyTorch 算：

```python
import torch

w = torch.tensor([1.0, 2.0, -3.0], requires_grad=True)
f = (w ** 2).sum()
print(f)
f.backward()
print(w.grad)
```

執行結果：

```text
tensor(14., grad_fn=<SumBackward0>)
tensor([ 2.,  4., -6.])
```

三個步驟：

1. 建立張量時加 `requires_grad=True`，告訴 PyTorch「我之後要對它求梯度」。
2. 照平常的方式算出 `f`。注意印出來多了一個 `grad_fn=<SumBackward0>`：PyTorch 記住了 `f` 是怎麼算出來的（最後一步是 `sum`）。
3. 對結果呼叫 `f.backward()`，梯度就被寫進 `w.grad`，形狀和 `w` 一樣。

結果 `[2, 4, -6]` 正是 \\( 2w \\)，和第 1 章有限差分的答案一樣，但這次沒有任何「推一點點再算一次」，而且不管 `w` 有三個還是三億個元素，都只要呼叫一次 `backward()`。

### 它怎麼做到的：記下每一步，再用鏈鎖律倒推

第 1 章第 11 集的鏈鎖律說：外層函數的導數乘以內層函數的導數。autograd 就是把這件事自動化：你每做一個運算，PyTorch 就在背後記下「這個結果是由哪些張量、經過什麼運算得到的」，串成一張圖。`backward()` 從結果出發，沿著這張圖倒著走，一邊走一邊把每一步的導數乘起來，最後落到每個標了 `requires_grad` 的張量上。

用一個鏈鎖律的例子驗證：\\( y = (3x + 1)^2 \\)，在 \\( x = 2 \\) 的導數是 \\( 2(3x + 1) \cdot 3 = 42 \\)。

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = (3 * x + 1) ** 2
y.backward()
print(x.grad)
```

執行結果：

```text
tensor(42.)
```

### 會傳染的 `requires_grad`

只要運算裡有一個張量 `requires_grad=True`，結果就自動也是 `True`，而且帶著 `grad_fn`。這就是圖一路往下串的方式：

```python
import torch

a = torch.tensor([1.0, 2.0], requires_grad=True)
b = a * 3
print(b.requires_grad, b)
```

執行結果：

```text
True tensor([3., 6.], grad_fn=<MulBackward0>)
```

兩件事先記著，下一集會用到：

- `backward()` 只能對**純量**（單一個數字）呼叫，所以我們總是先把結果 `sum()` 或 `mean()` 成一個數。訓練時的 loss 本來就是一個數，剛好。
- `.grad` 是**累加**的：再呼叫一次 `backward()`，新的梯度會加在舊的上面，不會覆蓋。

### 為什麼要先學有限差分

第 1 章的有限差分慢，但它是「梯度」的定義本身。autograd 算出來的東西你永遠可以用有限差分驗算，之後自己寫新的層時，這招會救你很多次。

## 重點整理

- `torch.tensor(..., requires_grad=True)` 標記要求梯度的張量；之後的運算結果會自動帶 `requires_grad` 與 `grad_fn`。
- 對純量結果呼叫 `.backward()`，梯度寫進每個來源張量的 `.grad`，形狀與該張量相同。
- 原理是鏈鎖律：PyTorch 記下每一步運算，從結果倒著把導數乘回去。
- `.grad` 會累加，不會自動歸零。
