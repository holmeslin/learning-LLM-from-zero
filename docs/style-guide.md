# 寫作風格指南

撰寫或修改 `zh-TW/src/` 內容前請先讀完。名詞定義見根目錄 `GLOSSARY.md`，全書目錄見 `docs/outline.md`。

## 讀者與語氣

- 讀者**會寫一點 Python**（變數、函式、串列、迴圈、`import`），但沒碰過深度學習、PyTorch，數學只剩高中程度。
- 繁體中文原生撰寫（台灣用語：程式、函式、變數、資料、機率、取樣、推論、記憶體、顯示卡、效能、最佳化），不是翻譯腔。用語對照表見 `GLOSSARY.md`。
- 口語、友善、短句，用「我們」「你」。不要說「這很簡單」「顯然」「不過如此」。
- 不寫成維基百科或論文。先講直覺，再上程式碼，公式放最後、而且旁邊一定有能跑的程式碼。
- 中英文之間加半形空格；程式碼、識別字、指令用反引號。
- 不參考、不改寫、不翻譯其他教材的文字：CS336 的講義與作業手冊只當課綱與引用來源；datawhalechina/diy-llm（CC BY-NC-SA）與 learning-rust-from-zero（CC BY-NC-ND）一個字都不能用。程式碼也一樣：本書程式碼是 CC0，不能混入 MIT 授權的 CS336 程式碼與測試。

## 術語

- 核心術語保留英文：token、tokenizer、embedding、attention、logits、softmax、loss、batch、checkpoint、RoPE、KV cache……第一次出現時加一句台灣中文釋義，之後直接用英文。
- 已有定論的中文照用：模型、訓練、參數、梯度、機率、取樣、推論、微調、資料集、詞彙表。
- 數學用 MathJax：行內寫 `\\( x \\)`，獨立一行寫 `\\[ ... \\]`。一集最多兩三個公式。

## 每一集的格式

檔名 `##_slug.md`（附錄 `a_slug.md`），標題與 `SUMMARY.md` 的標題一致：

```markdown
# <與 SUMMARY 相同的標題>

## 本集目標

一兩句話：讀完這集你會什麼。

## 正文

用 `###` 小標分段。每段一個重點，短範例 + 白話解說。

## 重點整理

- 3～5 條，條列本集真正教了什麼（AI 助教靠這段判斷進度）。
```

- 一集只教**一個**小概念。篇幅大約 40～150 行 markdown；第 1 章偏短。
- 每章的 `README.md` 只有 `# <章名>` 加一兩段導言，說明這章要學什麼、為什麼重要，不列目錄。
- 每章**最後一集**的「重點整理」之後加一節 `## 自我挑戰`：列出本章放進 `src/llmz/` 的函式，請讀者把函式本體換成 `raise NotImplementedError`，自己寫回來，直到 `uv run pytest tests/test_<模組>.py` 通過。
- 每章 `README.md` 的導言最後一段註明對應的 CS336 講次或作業（例如「對應 CS336 Assignment 1 的 tokenizer 部分」），只給連結與一句話，不轉述內容。

## 進度邊界（最重要）

- 範例與說明**只能使用 `SUMMARY.md` 中排在本集之前已教過的概念**，加上下方「固定句型」。
- 不得不提到後面的內容時，用一句話帶過並標明「第 X 章會正式介紹」，不展開。
- 不要「順便」介紹本集主題以外的語法或函式。
- Python 本身的語法（函式、類別、串列、字典、`for`、`with`）視為讀者已會，不受進度邊界限制，但第一次用到類別時仍要用一句話提醒。

## 固定句型

以下寫法在正式解釋之前就可以照抄使用，結構保持不變，只改檔名與變數名稱：

| 首次出現 | 固定句型 | 正式解釋 |
| --- | --- | --- |
| 第 1 章第 1 集 | 取得程式碼與執行：`git clone` 本書 repo、`uv sync`、在 `playground/` 寫 `.py` 檔、`uv run playground/<檔名>.py` | 不另外解釋 |
| 第 1 章第 1 集 | `import torch` | 第 1 章 |
| 第 1 章第 9 集 | `from llmz.device import get_device` 與 `device = get_device()`、`.to(device)` | 第 1 章第 9 集 |
| 第 2 章第 4 集 | 訓練迴圈：forward → loss → `backward()` → `step()` → `zero_grad()` | 第 2 章 |

## 程式碼區塊

- 一律 PEP 8（4 個空格縮排），變數與函式名稱用英文，字串與註解可以用中文。
- ```` ```python ```` 區塊必須是**完整可執行的腳本**，會被 `tools/checkcode.py` 存成 `main.py` 在空目錄執行，結束碼必須為 0。`from llmz...` 可以直接 import，因為 `uv sync` 會安裝本套件。
- 輸出區塊前一行固定寫「執行結果：」，`checkcode` 會**逐字比對** stdout。輸出不固定時（device 名稱、版本號、時間、記憶體、沒固定種子的隨機數）寫「執行結果（某一次）：」，只執行不比對。
- 書上的「執行結果：」一律在 **CPU、固定種子**下產生；MPS 的最後幾位數可能不同，文中要提醒讀者。
- 標註 `python,ignore`／`python,norun`／`python,error`／`python,stdin=...` 的意義見 `README.md`。不要用 `python,ignore` 逃避驗證：能寫成完整腳本就寫成完整腳本。
- 終端機指令用 ```` ```bash ````，不會被驗證。
- 故意示範錯誤：用 `python,error`，並在區塊後寫「錯誤訊息：」加 ```` ```text ```` 貼出 traceback 的**最後一行**（手動貼，用 PyTorch 2.9 的實際訊息）。
- 程式輸出用 ```` ```text ```` 區塊呈現。

## 從頭寫的規則

- 模型、loss、優化器的程式碼**不用** `torch.nn` 的現成層、`torch.nn.functional`、`torch.optim`；只能用 `torch.nn.Module`、`torch.nn.Parameter`、容器（`ModuleList`、`ModuleDict`）與 `torch.optim.Optimizer` 基底類別。
- torch 頂層函式（`torch.softmax`、`torch.logsumexp`）**只能用來驗證**自己寫的版本，不能在正文或套件裡當實作；驗證用途要在文字裡說明。
- 相依只有 torch、numpy、regex、pytest、matplotlib；`einops` 只在附錄介紹，正文用 `torch.einsum`。

## 套件與 include

- 跨集共用、會被測試的程式碼放在 `src/llmz/`，一個模組對應一個主題（`functional.py`、`tokenizer.py`、`layers.py`、`attention.py`、`model.py`、`optim.py`、`train.py`……）。
- 在原始檔用 `# ANCHOR: name` 與 `# ANCHOR_END: name` 標出片段，markdown 用 ```` ```python,ignore ```` 包住 `{{#include ../../../src/llmz/<模組>.py:name}}` 引入，路徑相對於該 markdown。
- anchor 名稱用英文小寫，同一檔案內不重複，不留沒被引用的 anchor；anchor 行與相鄰程式碼之間空一行。
- 被 include 的函式，正文要另有一個可執行的 ```` ```python ```` 區塊 `from llmz.<模組> import name` 實際呼叫並印出結果，讓讀者看到它怎麼用。
- 每個會訓練的範例都要有 smoke 設定（CPU 幾分鐘內完成），`tests/` 與 `checkcode` 跑的是它；完整訓練是章末任務，時間標「M4 Max 實測，較小的晶片會更慢」。
- 測試全部原創，用 torch 內建函式或手算的小例子驗證。

## 正確性

- 以 PyTorch 2.9、Python 3.13 為準；MPS 上沒有 TF32；`torch.compile` 在 2.9.1 的 MPS 上實測可用（Inductor 的 Metal 後端），加速幅度要實測後寫，不要憑印象。
- 書中的數字（loss、時間、記憶體）都要實測，寫明環境與設定。
- 不確定 API 行為時實際寫程式驗證，不要憑印象。
- 完成後執行 `uv run pytest` 與 `uv run python tools/checkcode.py zh-TW/src/<章節目錄>`，全部通過才算完成；`cd zh-TW && mdbook build` 不能有 warning。
- 寫完一章後更新：`SUMMARY.md`（把 `- [標題]()` 換成真正的連結）、`GUIDE.md` 的「已出版與未出版的章節」表與第 4 節進度表、`EXERCISES.md`。
