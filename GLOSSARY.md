# 從零開始學 LLM

為會寫 Python、沒碰過深度學習的讀者撰寫的繁體中文 LLM 教學書；本檔統一書籍編排、AI 助教規格與正文使用的名詞。

## 書籍結構

**部**：
全書最上層的分組；第一部從 PyTorch 基礎到訓練出一個小 LLM，第二部是系統、規模與推論，第三部是評估、資料與對齊。
_Avoid_: 篇、卷

**章**：
一個主題的集合，例如「第 3 章 Tokenizer」。
_Avoid_: 單元、課

**集**：
一個檔案，只教一個小概念，固定包含「本集目標」「正文」「重點整理」。
_Avoid_: 節、課、lesson

**附錄**：
屬於主線閱讀順序的補充章節；附錄一要在進入第二部之前讀完。

**選讀**：
標了「選讀」的集需要 GPU 或外部服務才能實際執行，沒有也不影響後面的內容。

## 教學規格

**固定句型**：
書中要求讀者先照抄、之後才解釋原理的寫法，例如 `uv run playground/xxx.py`、訓練迴圈的五個步驟。
_Avoid_: 樣板、boilerplate

**固定題庫**：
`EXERCISES.md` 裡預先寫好的練習題，涵蓋所有已出版的章。

**臨時題**：
AI 助教依讀者目前進度現場出的題目，必須明說不是固定題庫裡的題目。

**自我挑戰**：
每章最後一集結尾的練習：把 `src/llmz/` 裡本章的函式本體清掉，自己寫到 `uv run pytest` 通過。

**smoke 設定**：
把模型、資料與步數縮到 CPU 幾分鐘內能跑完的設定，`tests/` 與 `tools/checkcode.py` 執行的是它。
_Avoid_: 迷你設定、debug 模式

**標準語料**：
TinyStories 訓練檔的固定切片，全書的訓練數字都以它為準（第 3 章定義）。

## 保留英文的核心術語

第一次出現加一句中文釋義，之後直接用英文：token、tokenizer、BPE、embedding、attention、logits、softmax、loss、batch、checkpoint、RoPE、RMSNorm、SwiGLU、KV cache、scaling law、SFT、DPO、GRPO。

## 台灣用語對照

| 書中用語 | 不要用 | 英文 |
| --- | --- | --- |
| 資料、資料集 | 數據、數據集 | data, dataset |
| 機率 | 概率 | probability |
| 取樣 | 採樣、采样 | sampling |
| 推論 | 推理 | inference |
| 最佳化、優化器 | 優化（動詞） | optimize, optimizer |
| 正規化 | 歸一化、標準化 | normalization（RMSNorm、LayerNorm） |
| 正則化 | 正規化 | regularization（weight decay、dropout） |
| 微調 | 精調 | fine-tuning |
| 詞彙表 | 詞表 | vocabulary |
| 向量、矩陣、張量 | 矢量 | vector, matrix, tensor |
| 維度 | 維數 | dimension |
| 記憶體 | 內存 | memory |
| 顯示卡、GPU | 顯卡 | GPU |
| 效能 | 性能 | performance |
| 函式 | 函數（數學函數除外） | function |
| 變數 | 變量 | variable |
| 迴圈 | 循環 | loop |
| 字元 | 字符 | character |
| 位元組 | 字節 | byte |
| 字串 | 字符串 | string |
| 預設 | 默認 | default |
| 程式 | 程序 | program |
| 網路 | 網絡 | network |
| 梯度下降 | 梯度下降法 | gradient descent |
| 學習率 | 學習速率 | learning rate |
| 過擬合 | 過度擬合 | overfitting |
| 權重 | 權值 | weight |
| 偏差項 | 偏置 | bias |
| 交叉熵、cross-entropy | 交叉熵損失函數 | cross-entropy（正文用英文） |
| 區塊 | 塊 | block（Transformer block） |
| 專案 | 項目 | project |
| 套件 | 包、庫 | package |
| 啟用 | 激活 | activation（activation function 寫「啟用函式」） |
