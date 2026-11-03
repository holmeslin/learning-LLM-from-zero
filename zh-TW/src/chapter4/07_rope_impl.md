# RoPE 實作

## 本集目標

把上一集的旋轉寫成 `RoPE` 模組：預先算好每個位置、每一組的 cos 與 sin，`forward` 對 `(..., T, d_k)` 的張量做旋轉。

## 正文

### 轉速

第 i 組（i 從 0 到 \\( d_k/2 - 1 \\)）的轉速：

\\[ \theta_i = \Theta^{-2i / d_k} \\]

\\( \Theta \\) 是個大常數，本書用 10000。i = 0 時轉速 1（每個位置轉 1 弧度），i 越大轉速越接近 0。位置 m 的第 i 組轉 \\( m \theta_i \\)。

### 預先算好

```python,ignore
{{#include ../../../src/llmz/attention.py:rope}}
```

`__init__` 先做 `freqs`（每組的轉速）、再做 `angles`（每個位置 × 每組轉速，形狀 `(max_seq_len, d_k/2)`），然後存 cos 和 sin。這兩張表不是參數（不用學、沒有梯度），但要跟著模型一起 `.to(device)`，所以用 `register_buffer` 登記成 **buffer**：`nn.Module` 會像管參數一樣管它，但 `parameters()` 不會列出來。`persistent=False` 讓它不進 `state_dict`（反正載入時重算就有）。

`forward` 把最後一個維度拆成偶數位 `x1` 與奇數位 `x2`（第 1 章第 4 集的切片 `0::2` 是「從 0 開始每隔 2 個」），套上一集的旋轉公式，再用 `torch.stack(..., dim=-1)` 加 `flatten(-2)` 交錯放回原來的位置。`positions` 是每個 token 的位置，正常情況就是 `torch.arange(T)`。

```python
import torch

from llmz.attention import RoPE

torch.manual_seed(0)
rope = RoPE(d_k=8, max_seq_len=64)
print(rope.cos.shape)
q = torch.randn(1, 4, 8)
k = torch.randn(1, 4, 8)
pos = torch.arange(4)
print(torch.allclose(rope(q, pos).norm(dim=-1), q.norm(dim=-1)))
a = rope(q, pos) @ rope(k, pos).transpose(-2, -1)
b = rope(q, pos + 10) @ rope(k, pos + 10).transpose(-2, -1)
print(torch.allclose(a, b, atol=1e-5))
print(a[0, 3])
```

執行結果：

```text
torch.Size([64, 4])
True
True
tensor([ 1.5755,  5.1467, -0.5972, -2.3062])
```

三件事：cos 表是 `(64, 4)`（64 個位置、4 組）；旋轉不改變向量長度（`.norm(dim=-1)` 是第 1 章第 6 集的長度，每個位置各算一個）；把所有位置整體往後移 10 格，query 與 key 的點積矩陣完全不變，這就是上一集的相對位置性質。最後一行是位置 3 的 query 對四個 key 的分數，下一集之後它們會進 softmax。

`tests/test_attention.py` 另外用複數乘法寫了一個對照版本驗證這個實作（旋轉就是乘一個單位複數）。

### 用在哪裡

RoPE 只套在 attention 的 query 和 key 上，value 不轉，因為位置資訊只需要影響「誰看誰」的分數，不需要改變被看的內容。第 12 集把它接進 multi-head attention 時會看到。

## 重點整理

- 第 i 組轉速 \\( \Theta^{-2i/d_k} \\)，\\( \Theta = 10000 \\)；位置 m 轉 \\( m\theta_i \\)。
- cos、sin 表用 `register_buffer` 存：跟著 `.to(device)`，但不是參數、不進 `state_dict`。
- `forward` 拆偶數位與奇數位、旋轉、`stack` + `flatten` 放回。
- 不改向量長度；整體平移位置，點積不變。只套在 query 與 key。
