# decode 與無效 bytes

## 本集目標

寫出 `decode`：把 id 串列變回文字，並處理模型可能吐出「半個字」的情況。

## 正文

### 接起來再 decode

```python,ignore
{{#include ../../../src/llmz/tokenizer.py:decode}}
```

每個 id 查 `vocab` 得到 bytes，全部接起來，最後一次 `decode("utf-8")`。注意**不是**一個 token 一個 token 各自 decode：一個中文字的 3 個 byte 可能分散在兩個 token 裡，要先接好再轉。

```python
from llmz.tokenizer import Tokenizer, train_bpe_naive

text = "low low low lower lowest<|endoftext|>new newer newest"
vocab, merges = train_bpe_naive(text, 265, ["<|endoftext|>"])
tok = Tokenizer(vocab, merges, ["<|endoftext|>"])
ids = tok.encode("lowest newer<|endoftext|>low")
print(tok.decode(ids))
print(tok.decode([108, 111]))
```

執行結果：

```text
lowest newer<|endoftext|>low
lo
```

第一行是往返：encode 再 decode 回到原文，特殊 token 也原樣回來。第二行直接給兩個 byte 的 id（`l` 是 108、`o` 是 111），印出 `lo`。

### 半個字

這個 tokenizer 沒看過中文，所以中文每個 byte 各是一個 token：

```python
from llmz.tokenizer import Tokenizer, train_bpe_naive

text = "low low low lower lowest<|endoftext|>new newer newest"
vocab, merges = train_bpe_naive(text, 265, ["<|endoftext|>"])
tok = Tokenizer(vocab, merges, ["<|endoftext|>"])
ids = tok.encode("你好")
print(ids, [tok.vocab[i] for i in ids])
print(tok.decode(ids[:2]))
```

執行結果：

```text
[228, 189, 160, 229, 165, 189] [b'\xe4', b'\xbd', b'\xa0', b'\xe5', b'\xa5', b'\xbd']
�
```

只拿前兩個 id 去 decode，是「你」的前 2 個 byte，不是合法 UTF-8。第 1 集說過的 `errors="replace"` 讓它變成 `�` 而不是整個程式崩潰。第 6 章生成文字時，模型一個 token 一個 token 吐，隨時可能停在半個字上，這個參數非常重要。

### 往返一定成立嗎

`decode(encode(text)) == text` 對任何文字都成立，因為最差的情況也只是退到單一 byte，而 byte 接回去永遠是原本的 bytes。反過來 `encode(decode(ids)) == ids` **不一定**成立：ids 可能是訓練時不會產生的切法（例如上面把 `lo` 拆成兩個 byte，encode 回去會先合併）。

## 重點整理

- `decode(ids)`：所有 token 的 bytes 先接起來，再一次 `decode("utf-8", errors="replace")`。
- 一個字元的 bytes 可能跨 token，所以不能逐個 token decode。
- 半個字會變成 `�`，程式不會崩潰。
- `decode(encode(t)) == t` 永遠成立；反方向不一定。
