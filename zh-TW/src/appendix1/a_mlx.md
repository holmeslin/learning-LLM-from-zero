# MLX：Mac 上的替代方案

## 本集目標

知道 MLX 是什麼、和 PyTorch + MPS 差在哪裡、什麼時候值得換過去。本書不用它，但在 Mac 上做 LLM 遲早會碰到。

## 正文

### 是什麼

MLX 是 Apple 自己出的陣列計算框架，專門為 Apple Silicon 設計。API 長得像 NumPy 加上一點 PyTorch（有 `mlx.core` 的陣列、`mlx.nn` 的層、`mlx.optimizers`），但底層直接用統一記憶體與 Metal。

### 和 PyTorch + MPS 的差異

| 項目 | PyTorch + MPS | MLX |
| --- | --- | --- |
| 記憶體 | 有 `.to("mps")`，CPU 與 GPU 各管各的 | 沒有 device 的概念，陣列就在統一記憶體裡，CPU 與 GPU 都能直接用 |
| 執行 | 立即執行（eager），`torch.compile` 可選 | 延遲執行（lazy）：寫了不算，呼叫 `mx.eval()` 或用到結果才一起算 |
| 梯度 | autograd 記錄圖、`backward()` | 函數式：`mx.grad(f)` 回傳「算梯度的函數」 |
| 生態 | 幾乎所有論文程式碼、Hugging Face 都是 PyTorch | 小很多，但 Mac 上推論常見的工具（例如 mlx-lm）都在 |
| 速度 | 本書實測訓練 0.28 秒一步（compile） | 同樣的模型通常快一些到快不少，尤其是推論 |

本書第 5 章第 9 集的 MPS 注意事項（沒有 TF32、`synchronize`）在 MLX 都不存在，因為它根本不是「把 CUDA 的做法搬到 Mac」，而是從 Mac 的硬體出發設計的。

### 什麼時候換

- 你只在 Mac 上跑，而且在意速度與記憶體，尤其是推論：值得試。
- 你要跟別人的程式碼、預訓練模型、雲端 GPU 接軌：留在 PyTorch。
- 學習階段：留在 PyTorch。本書每個零件的數學在 MLX 一模一樣，學會了換框架只是改 API。

### 對照

本書的零件在 MLX 裡長這樣（示意，不是完整程式）：

```python,ignore
import mlx.core as mx
import mlx.nn as nn

x = mx.array([1.0, 2.0, 3.0])          # torch.tensor
y = mx.softmax(x)                       # 我們的 softmax
w = mx.random.normal((4, 3))            # torch.randn
loss_and_grad = nn.value_and_grad(model, loss_fn)  # 取代 loss.backward()
loss, grads = loss_and_grad(model, batch)
optimizer.update(model, grads)          # 取代 optimizer.step()
mx.eval(model.parameters())             # lazy：到這裡才真的算
```

想試的話另外開一個專案裝 `mlx`，不要加進本書的環境。

## 重點整理

- MLX 是 Apple 的陣列框架：統一記憶體、延遲執行、函數式梯度。
- 在 Mac 上常比 PyTorch + MPS 快，但生態小；學習與接軌用 PyTorch。
- 數學一樣，換框架只是改 API。
