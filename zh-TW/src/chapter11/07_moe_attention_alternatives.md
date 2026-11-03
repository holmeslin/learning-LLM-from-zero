# MoE 與 attention 替代方案

## 本集目標

認識兩類從架構上降低推論成本的方法：讓每個 token 只用到一部分參數的 MoE，以及讓 KV cache 不再隨長度增加的 attention 替代方案。

## 正文

### MoE：參數很多，每個 token 只用一點

前面幾集的成本都和「每個 token 要讀、要算的參數量」有關。**mixture of experts（MoE，混合專家）**把每一層的前饋網路換成好幾個「專家」（各自是一個前饋網路），再加一個小小的 **router**（路由器）：每個 token 由 router 打分數，只送給分數最高的 \\( k \\) 個專家，輸出是這幾個專家的加權和。

模型的總參數可以很多（知識容量大），但每個 token 實際用到的只有一小部分（計算便宜）。例如 Mistral AI 的 Mixtral 8x7B 每層 8 個專家、每個 token 用 2 個，總參數 467 億，每個 token 只用到 129 億；DeepSeek-V3 總參數 6710 億，每個 token 用 370 億。

用第 4 章的零件寫一個小的 MoE 層：

```python
import torch
from torch import nn

from llmz.functional import softmax
from llmz.layers import Linear, SwiGLU


class MoE(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, top_k):
        super().__init__()
        self.router = Linear(d_model, n_experts)
        self.experts = nn.ModuleList([SwiGLU(d_model, d_ff) for _ in range(n_experts)])
        self.top_k = top_k

    def forward(self, x):  # x (N, d_model)：N 個 token
        probs = softmax(self.router(x), dim=-1)  # (N, n_experts)
        weights, chosen = torch.topk(probs, self.top_k, dim=-1)  # 每個 token 分數最高的 top_k 個專家
        weights = weights / weights.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            token_idx, slot = torch.where(chosen == e)  # 哪些 token 選了專家 e，是它的第幾個選擇
            if len(token_idx) > 0:
                out[token_idx] += weights[token_idx, slot, None] * expert(x[token_idx])
        return out, chosen


torch.manual_seed(0)
moe = MoE(d_model=64, d_ff=128, n_experts=8, top_k=2)
x = torch.randn(1000, 64)
out, chosen = moe(x)
print("輸出形狀", tuple(out.shape))
print("每個專家分到的 token 數", torch.bincount(chosen.flatten(), minlength=8).tolist())

per_expert = sum(p.numel() for p in moe.experts[0].parameters())
print(f"專家的參數共 {8 * per_expert:,}，每個 token 只用 {2 * per_expert:,}")
```

執行結果：

```text
輸出形狀 (1000, 64)
每個專家分到的 token 數 [283, 272, 202, 206, 210, 270, 276, 281]
專家的參數共 196,608，每個 token 只用 49,152
```

- `torch.topk(probs, k, dim=-1)` 回傳最大的 \\( k \\) 個值與它們的位置，形狀都是 `(N, k)`。
- `torch.where(chosen == e)` 回傳所有 True 的座標：第一個是 token 的編號，第二個是它在 `chosen` 裡的第幾欄，用來取對應的權重。
- `torch.bincount` 數每個整數出現幾次，`minlength=8` 確保 8 個專家都有一格。
- 每個專家只處理選了它的 token，計算量是一般前饋網路的 \\( k \\) 倍（這裡 2 倍），參數卻是 8 倍。

MoE 的難處：

- **負載不平衡**：上面隨機初始化的 router，各專家分到的 token 數就不平均。訓練時 router 容易越來越偏好少數幾個專家，其他專家學不到東西。常見的做法是在 loss 裡加一項懲罰不平衡的**輔助 loss**。
- **記憶體**：每個 token 只算一部分，但所有專家都要放在記憶體裡。推論時省的是計算與每步要讀的權重（batch 小時），不是放模型的空間。
- **平行化**：專家常分散在不同的卡上（expert parallelism），token 要在卡之間傳來傳去，通訊量大。

### KV cache 會一直長大

第 3 集算過，Llama 3 8B 每個 token 的 KV cache 是 128 KiB。把 context 拉長到 128K 個 token（Llama 3.1 就是這樣），一條序列的 cache 就是 16 GiB，比權重還大。attention 的 cache 跟長度成正比，每生成一個 token 要讀的 cache 也跟長度成正比。下面幾種方法從不同的角度處理它：

- **GQA／MQA**（第 3 集）：好幾個 query head 共用 k、v，cache 縮小幾倍，但仍然隨長度增加。
- **壓縮 K、V**：DeepSeek 的 multi-head latent attention（MLA）把 k、v 先投影到一個低維的向量再存，用的時候再投影回來，cache 小得多。
- **sliding window attention**：每個 token 只看最近的 \\( W \\) 個 token，cache 最多存 \\( W \\) 個位置。Mistral 7B 用了 \\( W = 4096 \\)。更遠的資訊只能靠一層一層往上傳遞間接看到。
- **線性 attention 與 state space model**：把 softmax attention 換成可以寫成遞迴的形式，每一步只更新一個固定大小的狀態，像 RNN 一樣。cache 不再隨長度增加，生成每個 token 的成本也固定。代表是 Mamba 系列。代價是固定大小的狀態裝不下所有細節，要精確找出很久以前的某一句話時，表現通常不如 attention。

現在常見的折衷是**混合架構**：大部分的層用線性 attention 或 sliding window，每隔幾層放一層完整的 attention。

這一章的推論技巧（KV cache、batching、speculative decoding、量化）對這些架構大多一樣適用。第三部會進入評估、資料與對齊。

## 重點整理

- MoE 把前饋網路換成多個專家，router 讓每個 token 只用 top-\\( k \\) 個；總參數大，每個 token 的計算小。
- MoE 的難處：負載不平衡（用輔助 loss）、所有專家都要放進記憶體、專家分散時通訊量大。
- `torch.topk` 取最大的幾個值與位置；`torch.where` 找出所有 True 的座標。
- attention 的 KV cache 隨長度增加；GQA／MQA、MLA 縮小它，sliding window 限制它的長度，線性 attention 與 state space model 把它換成固定大小的狀態。

## 自我挑戰

這一章放進 `src/llmz/` 的是 `src/llmz/inference.py` 的 `KVCache`、`attention_cached`、`forward_cached`、`pick_next`、`generate_cached`、`to_probs`、`accept_or_resample`、`speculative_generate`、`quantize`、`dequantize`、`quantize_model_`。把它們的本體換成 `raise NotImplementedError`（`KVCache` 換 `__init__`），執行：

```bash
uv run pytest tests/test_inference.py
```

自己寫回來讓測試通過。

再試一題：替 `generate_cached` 加上 `stop_id` 參數。batch 裡某一條序列生成了 `stop_id` 之後，它後面的位置都填 `stop_id`，其他序列照常生成；所有序列都停了就提早結束。
