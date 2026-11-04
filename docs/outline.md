# 從零開始學 LLM — 全書目錄規劃（已定案）

Python 3.13 · PyTorch 2.9 · 繁中 · 會 Python 但沒碰過深度學習 · 一集一個小概念 · 16 GB Apple Silicon 可跑 · 對照 CS336 Spring 2026

全書規則：
- 從頭寫：不用 `torch.nn` 現成層與 `torch.optim`，只用 `Module`、`Parameter`、容器與 `Optimizer` 基底；Part 0 的線性回歸就開始自己寫
- 相依只有 torch、numpy、regex、pytest、matplotlib；`einops` 只在附錄
- 每個訓練集附 smoke 設定（CPU 幾分鐘內跑完，CI 跑的就是它）；完整訓練是章末任務，時間標「M4 Max 實測，較小晶片更慢」
- 全書標準語料：TinyStories train 檔的固定切片（大小以 40M token 看過約一輪為準，tokenize 後驗證），讓每個數字可重現且塞得進 16 GB
- 測試全部原創（CS336 A1 測試是 MIT，不能進 CC0 程式碼樹）

# 第一部 從零到訓練出一個小 LLM（對照 CS336 A1）

## 第 1 章 張量與剛好夠用的數學
01 安裝 uv 與取得程式碼 · 02 第一個張量 · 03 形狀與 dtype · 04 索引與切片 · 05 broadcasting · 06 向量與點積 · 07 矩陣乘法 `@` 與它的意義 · 08 `einsum` · 09 device：cpu 與 mps · 10 隨機數與種子 · 11 導數與鏈鎖律（直覺）· 12 梯度 · 13 softmax · 14 機率、log 機率與取樣 · 15 cross-entropy 與 perplexity · 16 數值穩定：log-sum-exp

## 第 2 章 autograd 與訓練迴圈
01 autograd：`requires_grad` 與 `backward()` · 02 手動梯度下降 · 03 `nn.Module` 與 `nn.Parameter` · 04 訓練迴圈固定句型（forward → loss → backward → step → zero_grad）· 05 線性回歸完整範例 · 06 `state_dict` 儲存與載入 · 07 FLOPs 與記憶體估算

## 第 3 章 Tokenizer：從 bytes 到 token
01 字串、Unicode 與 UTF-8 · 02 為什麼不用字元或單字 · 03 BPE 的核心想法 · 04 pre-tokenization 與 GPT-2 regex · 05 特殊 token · 06 訓練 BPE：統計 pair · 07 訓練 BPE：合併 · 08 訓練 BPE：加速與低資源做法 · 09 encode · 10 decode 與無效 bytes · 11 串流 encode 大檔 · 12 中文語料的 BPE 示範 · 13 壓縮率實驗 · 14 把 TinyStories 轉成 token 檔（`np.memmap`、uint16）

## 第 4 章 Transformer 的零件
01 一張圖看懂 Transformer LM · 02 Linear（自己寫）· 03 Embedding · 04 RMSNorm · 05 SiLU 與 SwiGLU · 06 位置資訊與 RoPE 的想法 · 07 RoPE 實作 · 08 數值穩定的 softmax · 09 scaled dot-product attention · 10 causal mask · 11 multi-head attention · 12 把 RoPE 接進 attention · 13 pre-norm Transformer block · 14 完整 Transformer LM · 15 參數量與 FLOPs

## 第 5 章 訓練
01 cross-entropy loss 實作 · 02 從頭寫 SGD（`Optimizer` 基底）· 03 AdamW · 04 warmup + cosine 學習率排程 · 05 梯度裁剪 · 06 資料載入：隨機取 batch · 07 checkpoint · 08 組裝訓練迴圈與 JSONL log · 09 MPS 注意事項（TF32、compile）· 10 第一次完整訓練 TinyStories（40M tokens）· 11 畫 loss 曲線

## 第 6 章 生成與實驗
01 greedy decoding · 02 temperature · 03 top-p 取樣 · 04 完整生成函式 · 05 消融：拿掉 RMSNorm · 06 消融：post-norm · 07 消融：NoPE · 08 消融：SiLU 取代 SwiGLU · 09 學習率掃描 · 10 批次大小

## 附錄一
a MLX：Mac 上的替代方案 · b `einops` 速查 · c 常見錯誤（shape、device、dtype）· d 對照 CS336 A1 測試清單（純文字對照）· e 雲端 GPU 租用

# 第二部 系統、規模與推論（對照 CS336 A2、A3，L5–L11）

## 第 7 章 效能分析與記憶體
01 記憶體階層（GPU 與 Apple Silicon）· 02 正確計時（同步、warmup）· 03 `torch.profiler` · 04 記憶體估算：參數、梯度、優化器狀態、activations · 05 混合精度 autocast · 06 activation checkpointing · 07 `torch.compile` · 08 arithmetic intensity

## 第 8 章 FlashAttention 的原理
01 attention 的 O(T²) 記憶體 · 02 tiling · 03 online softmax · 04 純 PyTorch 分塊 attention forward · 05 backward 與 recompute · 06 Triton 一瞥（選讀：需 GPU）· 07 `F.scaled_dot_product_attention` 現況

## 第 9 章 平行化（原理 + CPU 模擬）
01 為什麼需要多卡 · 02 `torch.distributed` + Gloo 在 CPU 起兩個 process · 03 all-reduce 等集合通訊 · 04 naive DDP · 05 bucketing 與通訊重疊 · 06 optimizer state sharding · 07 FSDP 的想法 · 08 tensor / pipeline parallelism（原理）· 09 通訊與計算的估算

## 第 10 章 Scaling laws
01 什麼是 scaling law · 02 Chinchilla 與 IsoFLOPs · 03 擬合 IsoFLOPs 曲線（離線資料）· 04 擬合冪律 · 05 Mac 上的迷你 scaling law（3–4 個小模型）· 06 限制與 μP 一瞥

## 第 11 章 推論
01 KV cache · 02 prefill 與 decode · 03 延遲與記憶體估算 · 04 batching 原理 · 05 speculative decoding 原理 · 06 量化初探 · 07 MoE 與 attention 替代方案（敘述）

## 附錄二
a CUDA 與 Triton 入門指引（選讀）· b 對照 CS336 A2、A3

# 第三部 評估、資料與對齊（對照 CS336 A4、A5，L12–L17）

## 第 12 章 評估
01 perplexity 的限制 · 02 benchmark 與 few-shot · 03 loglikelihood 評多選題 · 04 生成式評估與 LLM-as-judge · 05 資料污染

## 第 13 章 資料：從網頁到訓練集
01 Common Crawl 與 WARC · 02 HTML 轉文字 · 03 語言辨識 · 04 PII 遮蔽 · 05 有害內容分類 · 06 Gopher 品質規則 · 07 品質分類器 · 08 exact dedup · 09 MinHash + LSH · 10 組成資料管線 · 11 資料混合與合成資料（敘述）

## 第 14 章 後訓練：SFT 與偏好學習
01 base model 與指令模型 · 02 chat template 與 loss mask · 03 SFT 玩具版 · 04 reward model 的想法 · 05 DPO 推導與玩具版實作

## 第 15 章 後訓練：RL 與可驗證獎勵
01 policy gradient 直覺 · 02 REINFORCE 玩具版 · 03 GRPO 的 group-relative advantage · 04 GRPO loss 實作與測試 · 05 RLVR：用可驗證獎勵訓練 SFT 模型 · 06 Dr. GRPO、GSPO 一瞥 · 07 vLLM rollout（選讀：需 GPU）

## 附錄三
a 對照 CS336 A4、A5
