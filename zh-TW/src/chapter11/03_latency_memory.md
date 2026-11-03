# 延遲與記憶體估算

## 本集目標

用模型的設定估算推論要多少記憶體（權重加 KV cache），以及 decode 一步最快能多快；認識用來縮小 KV cache 的 GQA。

## 正文

### 兩塊記憶體

推論不用梯度與優化器狀態（第 7 章訓練記憶體四塊裡的兩塊），只剩：

- **權重**：參數量乘每個參數的位元組。
- **KV cache**：每一層、每個位置各存一個 k 與一個 v。每個 token 要

\\[
2 \times \text{層數} \times \text{KV head 數} \times d_{\text{head}} \times \text{每個數的位元組}
\\]

乘上序列長度與 batch 大小，就是整個 cache。

### decode 一步的下限

上一集說過 decode 是 memory-bound：每生成一個 token，所有權重都要從記憶體讀一遍。所以一步至少要「權重位元組 ÷ 記憶體頻寬」這麼久。這也是第 10 章第 6 集說「推論成本跟模型大小成正比」的原因。

### 算算看

用第 7 章量到的 M4 Max 頻寬 279 GB/s，比較本書的標準模型與 Meta 的 Llama 3 8B。Llama 3 8B 的設定取自它的 `config.json`：32 層、`d_model` 4096、32 個 query head、8 個 KV head、`d_ff` 14336、詞彙表 128256、最長 8192 個 token，以 bfloat16 儲存。

```python
def count_params(vocab, d_model, n_layers, n_heads, n_kv_heads, d_ff):
    d_head = d_model // n_heads
    attn = 2 * d_model * d_model + 2 * d_model * n_kv_heads * d_head  # q、o 與 k、v
    ffn = 3 * d_model * d_ff  # SwiGLU
    return 2 * vocab * d_model + n_layers * (attn + ffn + 2 * d_model) + d_model  # embedding、lm_head、RMSNorm


bandwidth = 279e9
models = {
    "標準模型（float32）": dict(cfg=(10000, 512, 4, 16, 16, 1344), bytes_per=4, context=256),
    "Llama 3 8B（bfloat16）": dict(cfg=(128256, 4096, 32, 32, 8, 14336), bytes_per=2, context=8192),
}
for name, m in models.items():
    vocab, d_model, n_layers, n_heads, n_kv_heads, d_ff = m["cfg"]
    n = count_params(*m["cfg"])
    weights = n * m["bytes_per"]
    kv_token = 2 * n_layers * n_kv_heads * (d_model // n_heads) * m["bytes_per"]
    step = weights / bandwidth
    print(name)
    print(f"  參數 {n:,}，權重 {weights / 1e9:.3f} GB")
    print(f"  decode 一步至少 {step * 1000:.2f} ms，每秒最多 {1 / step:.0f} 個 token")
    print(f"  KV cache 每個 token {kv_token / 1024:.0f} KiB，滿 {m['context']} 個 token {kv_token * m['context'] / 2**20:.0f} MiB")
```

執行結果：

```text
標準模型（float32）
  參數 22,696,448，權重 0.091 GB
  decode 一步至少 0.33 ms，每秒最多 3073 個 token
  KV cache 每個 token 16 KiB，滿 256 個 token 4 MiB
Llama 3 8B（bfloat16）
  參數 8,030,261,248，權重 16.061 GB
  decode 一步至少 57.56 ms，每秒最多 17 個 token
  KV cache 每個 token 128 KiB，滿 8192 個 token 1024 MiB
```

`count_params` 用標準模型的設定算出 22,696,448，和第 4 章一樣，表示公式沒寫錯；同一個公式算 Llama 3 8B 得到約 80 億，和它的名字相符。

幾個觀察：

- **標準模型**：讀權重只要 0.33 ms，但第 1 集實測一步將近 4 ms，差了十倍以上。這麼小的模型，時間花在啟動運算、Python 迴圈這些固定開銷上，頻寬根本沒用滿。
- **Llama 3 8B**：權重 16 GB，M4 Max 每秒最多生成 17 個 token 左右，這是任何實作都超不過的上限。換成頻寬 3 TB/s 以上的資料中心 GPU，上限就是每秒一兩百個 token。
- **KV cache**：Llama 3 8B 每個 token 128 KiB，一條 8192 個 token 的序列要 1 GiB。同時服務 16 個這樣的使用者，cache 就和權重一樣大。

### GQA：少存一點 K、V

Llama 3 8B 有 32 個 query head，卻只有 8 個 KV head：每 4 個 query head 共用同一組 k、v。這叫 **grouped-query attention（GQA）**。如果像本書的模型一樣每個 head 都有自己的 k、v（32 個 KV head），KV cache 會變成 4 倍，每個 token 512 KiB。極端的情況是所有 query head 共用一組，叫 multi-query attention（MQA）。

GQA 讓 cache 變小，decode 時要讀的資料就變少；論文的實驗顯示，模型品質幾乎沒有損失。實作上只要讓 `k_proj`、`v_proj` 輸出比較少的 head，做 attention 前把每組 k、v 複製給對應的幾個 query head。本書的模型沒有用 GQA。

## 重點整理

- 推論的記憶體是權重加 KV cache；KV cache 每個 token \\( 2 \times \text{層數} \times \text{KV head 數} \times d_{\text{head}} \times \text{位元組} \\)。
- decode 一步至少要「權重位元組 ÷ 頻寬」：標準模型 0.33 ms，Llama 3 8B 在 M4 Max 上約 58 ms，每秒最多約 17 個 token。
- 標準模型實測一步將近 4 ms，遠大於下限，小模型被固定開銷主導。
- Llama 3 8B 一條 8192 個 token 的序列，KV cache 要 1 GiB。
- GQA 讓好幾個 query head 共用一組 k、v，Llama 3 8B 用 8 個 KV head，cache 是不共用時的四分之一。
