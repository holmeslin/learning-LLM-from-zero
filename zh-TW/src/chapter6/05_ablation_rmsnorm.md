# 消融：拿掉 RMSNorm

## 本集目標

學會消融實驗（ablation）的做法：改一個零件、其他全部不動、比 loss 曲線。第一個實驗把 RMSNorm 整個拿掉，看第 4 章第 4 集說的「沒有它訓練不穩」是真是假。

## 正文

### 實驗設定

完整訓練要半小時，六個實驗跑不完。本章所有實驗都用縮短的設定：1000 步、其餘和標準設定相同（820 萬個 token，cosine 排程在第 1000 步降到最低）。先跑一個縮短版的基準，之後每個實驗都和它比：

```bash
uv run tools/train.py --out_dir runs/base_1k --compile --max_steps 1000 --eval_every 100
```

`--eval_every 100` 讓驗證 loss 密一點，曲線才看得清楚。M4 Max 上每個實驗約五分鐘。

### 拿掉 RMSNorm

`ModelConfig` 的 `norm="none"` 會把 block 裡與最後的 RMSNorm 全換成 `Identity`（第 4 章第 13 集）：

```bash
uv run tools/train.py --out_dir runs/no_rmsnorm --compile --max_steps 1000 --eval_every 100 --norm none
uv run tools/train.py --out_dir runs/no_rmsnorm_lr3e-4 --compile --max_steps 1000 --eval_every 100 --norm none --lr_max 3e-4 --lr_min 3e-5
```

第二行把學習率降到三分之一再試一次，看不穩定能不能用小步伐救回來。

### 結果

TBD_TABLE_NORM

![拿掉 RMSNorm 的 loss 曲線](images/ablation_norm.png)

TBD_TEXT_NORM

### 怎麼讀消融

- 只改一個東西。同時改兩個，就不知道是誰的功勞。
- 種子、資料順序、步數全部一樣（`tools/train.py` 的 `seed` 預設 0，`get_batch` 的起點由它決定，所以每個實驗看到的 batch 序列相同）。
- 看驗證 loss 的**最後幾個點**，也看曲線中間有沒有抖動或跳起來。
- 差 0.01 不算差，是 batch 抽樣的雜訊；差 0.05 以上才值得相信。

## 重點整理

- 消融：改一個零件、其他不動、比曲線；本章用 1000 步的縮短設定，基準是 `runs/base_1k`。
- TBD_SUMMARY_NORM
- 差距小於 0.02 當成雜訊。
