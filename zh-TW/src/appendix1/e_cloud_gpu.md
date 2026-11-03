# 雲端 GPU 租用

## 本集目標

知道什麼時候需要租 GPU、租來之後怎麼把本書的程式搬上去跑，以及避免浪費錢的幾個習慣。本書第一部全部不需要它。

## 正文

### 什麼時候需要

- 第一部：不需要。M4 Max 23 分鐘、16 GB 的 Mac 一小時內、CPU 幾小時，都跑得完。
- 第二部的 Triton（第 8 章第 6 集）與附錄二的 CUDA：標了「選讀」，需要 NVIDIA 顯示卡。
- 第三部的 vLLM rollout（第 15 章第 7 集）：選讀，需要 GPU。
- 想把模型放大十倍、資料放大十倍：需要。

### 租什麼

按小時計費的 GPU 主機有很多家，共同點是給你一台裝好 NVIDIA 驅動與 CUDA 的 Linux 機器，用 ssh 連進去。挑的時候看三件事：

1. **顯示記憶體**：本書的模型 24 GB 綽綽有餘；放大十倍的模型要 40～80 GB。
2. **CUDA 版本**：要和 PyTorch 2.9 支援的版本相容（目前是 12.x）。
3. **價格與計費方式**：按分鐘或按小時、關機要不要錢、硬碟另計與否。價格變動快，這裡不列。

### 搬上去

本書的 `pyproject.toml` 讓 Linux 預設裝 **CPU 版**的 PyTorch（CI 用，省下載）。在 GPU 機器上要改成 CUDA 版：把 `[tool.uv.sources]` 那段的 `pytorch-cpu` 改成 PyTorch 官方的 CUDA 索引（網址在 <https://pytorch.org/get-started/locally/>），然後：

```bash
git clone <你的 repo>
cd learning-LLM-from-zero
uv sync
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run tools/prepare_data.py
uv run tools/train.py --out_dir runs/base --compile
```

`get_device()` 會自動選 `cuda`，其他程式一行不用改。NVIDIA 顯示卡上可以多做兩件 MPS 做不到的事：`torch.set_float32_matmul_precision("high")` 打開 TF32（第 5 章第 9 集），以及第 7 章的 bfloat16 混合精度；兩者加起來通常再快幾倍。

### 省錢的習慣

- 先在本機用 smoke 設定跑通，確認程式沒錯再上雲；雲端除錯最貴。
- 用 `tmux` 或 `nohup` 跑訓練，ssh 斷線不會中斷。
- checkpoint 開著（本書每 1000 步），機器被收回也能接續。
- 跑完立刻把 `runs/` 拉回本機、關機。閒置的 GPU 和在跑的一樣貴。
- 先算 FLOPs（第 4 章第 15 集）估時間，再決定租幾小時。

## 重點整理

- 第一部不需要 GPU；選讀的 Triton、CUDA、vLLM 與放大實驗才需要。
- 租用前看顯示記憶體、CUDA 版本、計費方式。
- 搬上去：改 `pyproject.toml` 的 torch 索引、`uv sync`、`prepare_data`、`train`，`get_device()` 自動用 `cuda`。
- 先本機 smoke、用 tmux、開 checkpoint、跑完就關。
