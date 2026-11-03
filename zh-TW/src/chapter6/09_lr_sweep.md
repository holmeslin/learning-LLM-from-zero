# 學習率掃描

## 本集目標

同一個模型用四個學習率各跑一次，找出太小、剛好、太大、發散的界線，知道學習率為什麼是最重要的超參數。

## 正文

### 四個值

`lr_min` 跟著 `lr_max` 一起縮放（維持十分之一）：

```bash
uv run tools/train.py --out_dir runs/lr_3e-4 --compile --max_steps 1000 --eval_every 100 --lr_max 3e-4 --lr_min 3e-5
uv run tools/train.py --out_dir runs/lr_3e-3 --compile --max_steps 1000 --eval_every 100 --lr_max 3e-3 --lr_min 3e-4
uv run tools/train.py --out_dir runs/lr_1e-2 --compile --max_steps 1000 --eval_every 100 --lr_max 1e-2 --lr_min 1e-3
```

基準 `runs/base_1k` 是 \\( 10^{-3} \\)。四個值每個差三倍左右，這是掃學習率的慣例：差距太小看不出差別，太大會漏掉最好的區間。

### 結果

TBD_TABLE_LR

![學習率掃描](images/lr_sweep.png)

TBD_TEXT_LR

### 怎麼選

一般的經驗：先掃三倍一格，找到最好的那格之後，發散的那個值除以二到三，常常就是最好的。學習率和模型大小、batch 大小、訓練長度都有關，換了設定要重掃；第 10 章的 scaling law 會談怎麼從小模型推到大模型。

## 重點整理

- 掃學習率用三倍一格；本書的四個值 \\( 3\times10^{-4} \\)、\\( 10^{-3} \\)、\\( 3\times10^{-3} \\)、\\( 10^{-2} \\)。
- TBD_SUMMARY_LR
- 最好的學習率通常在「發散的值」往下兩三倍的地方；換設定要重掃。
