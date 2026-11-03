# 批次大小

## 本集目標

固定看過的 token 總數，比較 batch 8、32、128 的差別：小 batch 步數多但每步吵，大 batch 每步穩但步數少，而且速度差很多。

## 正文

### 固定 token 總數

基準 batch 32 × 1000 步 = 819 萬個 token。要公平，batch 8 就要 4000 步、batch 128 只有 250 步：

```bash
uv run tools/train.py --out_dir runs/bs8 --compile --max_steps 4000 --eval_every 400 --batch_size 8
uv run tools/train.py --out_dir runs/bs128 --compile --max_steps 250 --eval_every 25 --warmup_steps 50 --batch_size 128
```

batch 128 只有 250 步，warmup 照預設 200 步的話還沒升到最高就要開始降了，所以改成 50。這也是個提醒：改 batch 常常要連帶改排程。

### 結果

TBD_TABLE_BS

![batch 大小](images/batch_size.png)

TBD_TEXT_BS

### 速度

TBD_TEXT_BS_SPEED

### 真正的限制

本書的模型小，batch 128 × 256 塞得進 GPU；大模型的 batch 受記憶體限制（第 7 章），而且多卡訓練時 batch 是分攤到所有卡的（第 9 章），所以實務上 batch 大小常常是「硬體塞得下多少」決定的，再回頭調學習率。

## 重點整理

- 比較 batch 大小要固定 token 總數：`batch × steps` 相同。
- TBD_SUMMARY_BS
- 改 batch 要連帶考慮 warmup 與學習率。

## 自我挑戰

這一章放進 `src/llmz/` 的是 `src/llmz/generate.py`：`greedy_next`、`sample_next`、`generate`。把三個函式本體清掉自己寫回來：

```bash
uv run pytest tests/test_generate.py
```

`sample_next` 的 top-p 那段是最容易寫錯的地方（「累積減自己」的邊界）。然後用你的 `generate` 和 `runs/base/model.pt` 生成幾個故事，確認它們通順。

最後是整個第一部的總挑戰：把 `src/llmz/` 整個目錄的函式本體全部清掉，從 `tests/` 開始，一個模組一個模組寫回來，直到 `uv run pytest` 全綠，再跑一次完整訓練。做到這裡，你已經從零寫出並訓練了一個語言模型。
