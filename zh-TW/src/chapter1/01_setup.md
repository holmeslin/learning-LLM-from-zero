# 安裝 uv 與取得程式碼

## 本集目標

裝好 uv，下載本書的程式碼，用 `uv run` 跑出第一支會 `import torch` 的程式。

## 正文

### 安裝 uv

本書用 [uv](https://docs.astral.sh/uv/) 管理 Python 版本與套件。它會幫我們下載正確的 Python、安裝 PyTorch，並且讓每支程式都在同一個環境裡執行，不用自己管虛擬環境。

macOS 上有 Homebrew 的話，打開終端機執行：

```bash
brew install uv
```

沒有 Homebrew，或是用 Linux、Windows，照 uv 官方的安裝說明做：<https://docs.astral.sh/uv/getting-started/installation/>。

裝完後關掉終端機再開一次，確認裝好了：

```bash
uv --version
```

### 取得本書的程式碼

本書所有程式碼都放在同一個 repo 裡，書上出現的每一段都是從這裡拿出來、被測試驗證過的。先把它抓下來：

```bash
git clone https://github.com/holmeslin/learning-LLM-from-zero.git
cd learning-LLM-from-zero
uv sync
```

`uv sync` 會讀 `pyproject.toml`，下載 Python 3.13 與 PyTorch 2.9，並把本書的套件 `llmz` 裝進專案環境。第一次要抓幾百 MB，花幾分鐘是正常的；之後再執行就只會檢查、不會重抓。

### 第一支程式

在 repo 裡建立一個 `playground` 資料夾，你自己的練習檔都放這裡（它被 `.gitignore` 忽略，不會弄亂 repo）：

```bash
mkdir -p playground
```

新增檔案 `playground/hello.py`，內容如下：

```python
import torch

print("PyTorch", torch.__version__)
```

執行：

```bash
uv run playground/hello.py
```

執行結果（某一次）：

```text
PyTorch 2.9.1
```

看到版本號就成功了。`import torch` 把 PyTorch 載進來，`torch.__version__` 是它的版本字串。你的版本號後面可能多一段文字（例如 `+cpu`），沒關係。

### 固定句型：寫檔、`uv run`

之後本書的每個範例都照這個流程：

1. 在 `playground/` 新增一個 `.py` 檔，把範例貼進去（建議用打的，比較記得住）。
2. 在 repo 根目錄執行 `uv run playground/<檔名>.py`。

一定要用 `uv run`，不要直接打 `python playground/hello.py`。直接用 `python` 會用到你電腦裡其他的 Python，它沒有裝 PyTorch，你會看到：

```text
ModuleNotFoundError: No module named 'torch'
```

`uv run` 的意思是「用這個專案的環境來跑」，它會自動選對 Python 與套件。

### 沒有 Mac 也可以

本書以 Apple Silicon 的 Mac 為主要環境，但所有程式碼在 Linux 與 Windows 的 CPU 上也能跑，只是訓練會慢一些。本書的設定讓 Linux 預設安裝 CPU 版的 PyTorch；有 NVIDIA 顯示卡、想用 CUDA 的讀者，可以把 `pyproject.toml` 裡 `[tool.uv.sources]` 那段刪掉再 `uv sync`。

## 重點整理

- 用 uv 管環境：`git clone` 本書 repo 之後 `uv sync` 一次就好。
- 自己的練習檔放在 `playground/`，用 `uv run playground/<檔名>.py` 執行。這是全書的固定句型。
- 不要直接用 `python` 執行，會找不到 `torch`。
- 本書以 PyTorch 2.9、Python 3.13 為準。
