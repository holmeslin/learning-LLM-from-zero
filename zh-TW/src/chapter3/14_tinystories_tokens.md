# 把 TinyStories 轉成 token 檔

## 本集目標

定義全書的標準語料，用一支腳本下載、訓練 BPE、把文字轉成 uint16 的 token 檔，第 5 章訓練直接讀它。

## 正文

### 標準語料

TinyStories 是一套由 GPT-4 產生的英文童話故事資料集，用字簡單、故事很短，小模型也學得出通順的句子。全書的訓練數字都以下面這份切片為準：

| 項目 | 定義 | 實際大小 |
| --- | --- | --- |
| 訓練集 `data/train.txt` | `TinyStoriesV2-GPT4-train.txt` 的前 160 MiB，截短到最後一個完整的 `<|endoftext|>` | 167,771,202 bytes，40,720,162 個 token |
| 驗證集 `data/valid.txt` | 整個 `TinyStoriesV2-GPT4-valid.txt` | 22,502,601 bytes，5,461,883 個 token |
| tokenizer | 在訓練集上訓練，vocab 10000，特殊 token `<|endoftext|>` | `data/tokenizer.json`，9,743 次合併 |

4000 萬個 token 的大小是刻意選的：第 5 章的模型看它一輪就是一次完整訓練，M4 Max 上幾十分鐘。整個 TinyStories 有 2.2 GB、五億多個 token，本書用不到。

資料集的授權是 CDLA-Sharing-1.0，所以 repo 裡不放任何一段它的文字，`data/` 在 `.gitignore` 裡，每個人自己下載。

### 只下載需要的部分

不用抓整個 2.2 GB：HTTP 有 Range 標頭，可以只要檔案的前面那一段。`src/llmz/data.py`：

```python,ignore
{{#include ../../../src/llmz/data.py:download}}
```

標準函式庫的 `urllib` 就夠了；`Range: bytes=0-N` 請伺服器只給前 N 個 byte，一次讀 1 MB 寫進檔案。切在 160 MiB 的地方多半是一篇故事的中間，所以再把尾巴砍到最後一個 `<|endoftext|>`：

```python,ignore
{{#include ../../../src/llmz/data.py:cut_at_last_document}}
```

### 寫成 uint16

詞彙表 10000 小於 65536，每個 id 用 2 個 byte 的無號整數（uint16）就裝得下，4000 萬個 token 只要 80 MB：

```python,ignore
{{#include ../../../src/llmz/data.py:tokenize_to_file}}
```

`array.array("H")` 是標準函式庫裡「一串 16 位元無號整數」的容器，比 Python 串列省十幾倍記憶體；`extend` 直接吃第 11 集的生成器，一行一行 encode 進去；`tofile` 把二進位內容原樣寫出。

讀回來用 NumPy 的 `memmap`：

```python,ignore
{{#include ../../../src/llmz/data.py:load_tokens}}
```

`np.memmap` 把檔案映射成陣列，`data[i]` 或 `data[a:b]` 才真的去讀那一段，整個檔案不會進記憶體。第 5 章每一步從它隨機切一批出來訓練。

小例子走一遍：

```python
import numpy as np

from llmz.data import load_tokens, tokenize_to_file
from llmz.tokenizer import Tokenizer, train_bpe_naive

text = "one fish two fish<|endoftext|>\nred fish blue fish<|endoftext|>\n"
with open("tiny.txt", "w", encoding="utf-8") as f:
    f.write(text)
vocab, merges = train_bpe_naive(text, 280, ["<|endoftext|>"])
tok = Tokenizer(vocab, merges, ["<|endoftext|>"])
n = tokenize_to_file(tok, "tiny.txt", "tiny.bin")
data = load_tokens("tiny.bin")
print(n, data.dtype, data.shape)
print(data[:5])
print(tok.decode(data[:5].tolist()))
```

執行結果：

```text
12 uint16 (12,)
[267 260 270 260 256]
one fish two fish<|endoftext|>
```

`data[:5].tolist()` 把 NumPy 的整數轉回 Python 的 `int`，`decode` 才查得到 `vocab`。

### 整支腳本

把下載、截短、平行統計、`train_bpe`、存 tokenizer、轉檔串起來，就是 `tools/prepare_data.py`。執行：

```bash
uv run tools/prepare_data.py
```

執行結果（M4 Max 實測，較小的晶片會更慢；下載速度看網路）：

```text
train.txt: 167,771,202 bytes，8 秒
valid.txt: 22,502,601 bytes，3 秒
pre-tokenize：25,553 種不同的 pre-token，3 秒
train_bpe：9,743 次合併，13 秒
valid.bin：5,461,883 個 token，4.12 bytes/token，3 秒
train.bin：40,720,162 個 token，4.12 bytes/token，19 秒
```

一分鐘內完成。已經存在的檔案會跳過，中斷了重跑就好。之後 `data/` 裡有五個檔案：`train.txt`、`valid.txt`、`tokenizer.json`、`train.bin`、`valid.bin`，第 5 章只需要後三個。

## 重點整理

- 標準語料：TinyStories 訓練檔前 160 MiB（40,720,162 個 token）＋整個驗證檔（5,461,883 個 token），vocab 10000。
- `download(url, path, max_bytes)` 用 HTTP Range 只抓前面一段；`cut_at_last_document` 砍掉不完整的尾巴。
- `tokenize_to_file` 用 `array.array("H")` 逐行 encode 寫成 uint16；`load_tokens` 用 `np.memmap` 讀，不進記憶體。
- `uv run tools/prepare_data.py` 一分鐘做完全部；`data/` 不進 git。

## 自我挑戰

這一章放進 `src/llmz/` 的函式有：

- `src/llmz/tokenizer.py`：`pretokenize`、`split_on_special`、`count_words`、`count_pairs`、`merge_word`、`base_vocab`、`train_bpe_naive`、`count_words_parallel`、`train_bpe`，以及 `Tokenizer` 的 `encode_word`、`encode`、`decode`、`encode_iterable`、`save`、`load`
- `src/llmz/data.py`：`download`、`cut_at_last_document`、`tokenize_to_file`、`load_tokens`

建議的順序：先把 `merge_word`、`count_pairs`、`train_bpe_naive` 換成 `raise NotImplementedError` 自己寫回來，再做 `encode_word` 與 `decode`，最後挑戰 `train_bpe`（增量更新那段最難）。每做完一個就跑：

```bash
uv run pytest tests/test_tokenizer.py tests/test_data.py
```

全部通過後，重新跑一次 `uv run tools/prepare_data.py`（先刪掉 `data/tokenizer.json` 與兩個 `.bin`），token 數應該和書上一模一樣。
