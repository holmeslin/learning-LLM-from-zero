# 參數量與 FLOPs

## 本集目標

用第 2 章第 7 集的方法，算清楚標準設定的模型參數在哪裡、訓練一步要多少 FLOPs、記憶體要多少，知道第 5 章那場訓練為什麼要幾十分鐘。

## 正文

### 參數在哪裡

| 零件 | 公式 | 標準設定 |
| --- | --- | --- |
| Embedding | \\( V \times d \\) | 10000 × 512 = 5,120,000 |
| 每層 attention | \\( 4 d^2 \\) | 4 × 512² = 1,048,576 |
| 每層 SwiGLU | \\( 3 d \cdot d_{\text{ff}} \\) | 3 × 512 × 1344 = 2,064,384 |
| 每層兩個 RMSNorm | \\( 2d \\) | 1,024 |
| 最後 RMSNorm | \\( d \\) | 512 |
| lm_head | \\( d \times V \\) | 5,120,000 |

```python
from llmz.model import ModelConfig, TransformerLM
from llmz.utils import count_parameters

cfg = ModelConfig()
V, d, L, ff = cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.d_ff
per_layer = 4 * d * d + 3 * d * ff + 2 * d
total = V * d + L * per_layer + d + d * V
print(per_layer, total)
print(count_parameters(TransformerLM(cfg)) == total)
print(round(2 * V * d / total, 2))
```

執行結果：

```text
3113984 22696448
True
0.45
```

兩千兩百七十萬個參數，其中 Embedding 和 lm_head 合起來佔 45%：小模型的詞彙表很「貴」。4 層 block 佔剩下的 55%。

### 一個 token 的 FLOPs

第 2 章第 7 集：矩陣乘法 \\( (m, k) @ (k, n) \\) 是 \\( 2mkn \\) FLOPs，所以每個 token 經過一個有 P 個參數的矩陣是 \\( 2P \\) FLOPs。逐項算 forward：

| 零件 | 每個 token 的 FLOPs |
| --- | --- |
| Embedding | 0（查表） |
| 每層 attention 投影 | \\( 2 \times 4d^2 \\) |
| 每層 attention 分數 \\( QK^\top \\) 與 \\( AV \\) | \\( 2 \times 2 \times T \times d \\)（跟序列長度 T 有關） |
| 每層 SwiGLU | \\( 2 \times 3 d \cdot d_{\text{ff}} \\) |
| lm_head | \\( 2 \times d V \\) |

```python
from llmz.model import ModelConfig

cfg = ModelConfig()
V, d, L, ff, T = cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.d_ff, cfg.context_length
per_layer = 2 * 4 * d * d + 4 * T * d + 2 * 3 * d * ff
forward = L * per_layer + 2 * d * V
print(f"forward 每個 token {forward / 1e6:.1f} MFLOPs，其中 attention 分數佔 {L * 4 * T * d / forward:.1%}")
train = 3 * forward
tokens = 5000 * 32 * 256
print(f"訓練 {tokens / 1e6:.0f}M 個 token 共 {train * tokens / 1e15:.1f} PFLOPs")
```

執行結果：

```text
forward 每個 token 37.2 MFLOPs，其中 attention 分數佔 5.6%
訓練 41M 個 token 共 4.6 PFLOPs
```

每個 token forward 約 3700 萬 FLOPs，差不多是「2 × 非 embedding 參數」（第 2 章第 7 集的估法：2 × 1760 萬 = 3500 萬），attention 分數的部分在 T = 256 時只佔 5.6%；T 長到幾千時它才會變成主角，那是第 8 章的事。訓練（forward + backward ≈ 3 倍）4100 萬個 token 要 4.6 PFLOPs。M4 Max 的 GPU 實際跑這種小模型大約每秒 3～5 TFLOPs，算起來十五到二十五分鐘；第 5 章實測看看。

### 記憶體

參數 2270 萬個，AdamW 訓練每個 16 位元組：363 MB。activations 另算：batch 32 × T 256 = 8192 個 token，每層要留下 attention 分數 `(32, 16, 256, 256)` 與幾個 `(8192, d_ff)` 的中間結果，大約幾百 MB 到 1 GB，16 GB 的機器沒問題。詳細估算放在第 7 章。

## 重點整理

- 標準設定 2270 萬參數，Embedding + lm_head 佔 45%，每層 310 萬。
- forward 每 token ≈ 2 × 非 embedding 參數 FLOPs，attention 分數項正比於 T，短序列可忽略；訓練 ≈ 3 × forward。
- 4100 萬個 token 的訓練約 4.6 PFLOPs，M4 Max 二十分鐘量級。
- 參數加優化器狀態 363 MB，activations 幾百 MB。

## 自我挑戰

這一章放進 `src/llmz/` 的東西：

- `src/llmz/layers.py`：`Linear`、`Embedding`、`RMSNorm`、`silu`、`SwiGLU`、`SiLUFFN`
- `src/llmz/attention.py`：`RoPE`、`causal_mask`、`scaled_dot_product_attention`、`MultiHeadSelfAttention`
- `src/llmz/model.py`：`TransformerBlock`、`TransformerLM`

建議順序：`Linear`、`Embedding`、`RMSNorm`、`silu`、`SwiGLU` 的 `forward` 先清掉重寫（`__init__` 的初始化可以留著）；再做 `causal_mask` 與 `scaled_dot_product_attention`；然後 `RoPE.forward`（最容易寫錯的是偶數位、奇數位的拆法）；最後 `MultiHeadSelfAttention.forward` 的那串 reshape 與 transpose。每做完一個就跑：

```bash
uv run pytest tests/test_layers.py tests/test_attention.py tests/test_model.py
```

`test_model.py` 的參數量公式與 causal 測試會在所有零件都對了之後才通過。
