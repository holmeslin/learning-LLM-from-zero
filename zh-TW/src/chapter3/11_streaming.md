# 串流 encode 大檔

## 本集目標

用 `yield` 一行一行 encode，不把整個檔案讀進記憶體；並把訓練好的 tokenizer 存成檔案重複使用。

## 正文

### 一次讀一行

標準語料的訓練檔 160 MB，整個讀成一個字串再 `encode` 也不是不行，但真正的語料是幾十 GB，就塞不下了。做法是把檔案當成「一行一行的 iterable」，每一行各自 encode，用 `yield` 一個一個吐出 id：

```python,ignore
{{#include ../../../src/llmz/tokenizer.py:encode_iterable}}
```

`yield` 讓函式變成生成器（generator）：呼叫它不會立刻跑完，而是每被要一個值才算一個。`yield from` 把一整個串列裡的東西逐個吐出去。Python 開檔案得到的 `f` 本身就是「一行一行」的 iterable，直接丟進去就行。

```python
import io

from llmz.tokenizer import Tokenizer, train_bpe_naive

text = "low low low lower lowest<|endoftext|>new newer newest"
vocab, merges = train_bpe_naive(text, 265, ["<|endoftext|>"])
tok = Tokenizer(vocab, merges, ["<|endoftext|>"])
f = io.StringIO("low lower\nnewest low\n")
print(list(tok.encode_iterable(f)))
```

執行結果：

```text
[258, 264, 114, 10, 263, 262, 259, 10]
```

`io.StringIO` 把字串假裝成一個檔案，方便示範；真的檔案就是 `with open(path) as f:`。10 是換行字元 `\n` 的 byte 值，每一行結尾都有一個。

一行一行切有個小代價：換行前後的字不會被合併在一起。GPT-2 的 regex 本來就把換行當成獨立的片段（第 4 集），所以結果和整個檔案一次 encode **完全一樣**，`tests/test_tokenizer.py` 有驗證這件事。

### 存檔與載入

訓練一次就好，之後要用就載入。`vocab` 裡的 bytes 不能直接寫進 JSON，用 latin-1 把每個 byte 一對一變成一個字元（latin-1 剛好有 256 個字元，和 256 種 byte 對得上）：

```python,ignore
{{#include ../../../src/llmz/tokenizer.py:save_load}}
```

```python
from llmz.tokenizer import Tokenizer, train_bpe_naive

text = "low low low lower lowest<|endoftext|>new newer newest"
vocab, merges = train_bpe_naive(text, 265, ["<|endoftext|>"])
tok = Tokenizer(vocab, merges, ["<|endoftext|>"])
tok.save("small.json")
back = Tokenizer.load("small.json")
print(back.encode("lowest") == tok.encode("lowest"))
```

執行結果：

```text
True
```

標準語料的 tokenizer 存成 `data/tokenizer.json` 大約 330 KB，第 14 集會做。

## 重點整理

- `encode_iterable(lines)` 用 `yield from` 一行一行吐出 id，記憶體用量和檔案大小無關。
- 開檔得到的 `f` 本身就是逐行的 iterable；逐行 encode 的結果和整檔 encode 相同。
- `save(path)`／`Tokenizer.load(path)` 用 JSON 存 vocab、merges、特殊 token；bytes 用 latin-1 轉成字元。
