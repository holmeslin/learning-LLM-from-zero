[![CC BY-NC-ND 4.0](https://img.shields.io/badge/content-CC%20BY--NC--ND%204.0-blue)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![CC0 1.0](https://img.shields.io/badge/code-CC0%201.0-blue)](https://creativecommons.org/publicdomain/zero/1.0/)

# 從零開始學 LLM

給會寫 Python、沒碰過深度學習的人的 LLM 教學（PyTorch 2.9，一台 16 GB Apple Silicon Mac 就能跑），請閱讀[線上版](https://holmeslin.github.io/learning-LLM-from-zero/zh-TW/)。

課綱參考 Stanford [CS336: Language Modeling from Scratch](https://cs336.stanford.edu)；文字與程式碼皆為原創。

## 本機建置

需要 [uv](https://docs.astral.sh/uv/) 與 [mdBook](https://github.com/rust-lang/mdBook) 0.5.4：

```bash
uv sync                                      # 安裝 Python 3.13、PyTorch 與 llmz 套件
uv run pytest                                # 測試 src/llmz
uv run python tools/checkcode.py zh-TW/src   # 執行並驗證書中所有 Python 範例
cd zh-TW && mdbook serve                     # 本機預覽
```

跟著書訓練模型：

```bash
uv run tools/prepare_data.py                        # 下載標準語料、訓練 BPE、轉 token 檔（約一分鐘）
uv run tools/train.py --out_dir runs/base --compile # 完整訓練（M4 Max 約 23 分鐘）
uv run tools/plot_loss.py runs/base -o loss.png     # 畫 loss 曲線
```

## 程式碼怎麼放

- `src/llmz/`：全書逐章累積的套件（tokenizer、模型、優化器……），書中用 mdBook 的 `{{#include}}` 以 anchor 引入片段，所以書上看到的程式碼就是 `tests/` 驗證過的版本。
- `zh-TW/src/`：mdBook 內容。每個 ```` ```python ```` 區塊都會被 `tools/checkcode.py` 實際執行。
- `tools/`：`checkcode.py`（驗證書中範例）、`prepare_data.py`、`train.py`、`plot_loss.py`；第 10 章 `scaling_sweep.py`、`plot_scaling.py`；第 13 章 `prepare_webtext.py`（網頁文字）、`train_quality.py`（品質分類器）；第 14 章 `sft.py`、`dpo.py`；第 15 章 `rlvr.py`。
- `playground/`、`data/`、`runs/`：讀者的練習檔、語料與訓練輸出，都在 `.gitignore` 裡。

### 範例程式碼標註

書中 ```` ```python ```` 區塊預設會存成 `main.py` 執行，必須以結束碼 0 結束；區塊後緊接「執行結果：」加 ```` ```text ```` 區塊時，stdout 必須與之完全一致（寫「執行結果（某一次）：」則只執行不比對）。可在 info string 加上標註：

| 標註 | 意義 |
| --- | --- |
| `python,ignore` | 不驗證（片段、`{{#include}}` 引入的程式碼） |
| `python,norun` | 只檢查語法，不執行（會跑很久或需要外部資源） |
| `python,error` | 預期以非 0 結束碼結束（故意示範錯誤） |
| `python,stdin=3\n5` | 執行時餵入 stdin（`\n` 代表換行，值內不可有空白） |

`checkcode` 也會檢查每個 `{{#include 路徑:anchor}}` 指到的檔案與 anchor 都存在。

## 授權

除非另有標示，本書的文字、圖片與非程式碼內容以 [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) 授權。

本書中的程式碼（含 `src/`、`tests/`、`tools/`）以 [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) 釋出，可自由複製、修改與使用。
