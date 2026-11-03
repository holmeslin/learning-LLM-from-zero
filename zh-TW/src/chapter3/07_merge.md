# 訓練 BPE：合併

## 本集目標

寫出 `merge_word` 與教學版的 `train_bpe_naive`，在小語料上跑出一份詞彙表與合併規則。

## 正文

### 合併一個字裡的 pair

```python,ignore
{{#include ../../../src/llmz/tokenizer.py:merge_word}}
```

從左到右掃，遇到連續兩個元素剛好是要合併的 pair，就用接起來的 bytes 取代它們、跳兩格；否則照抄、跳一格。

```python
from llmz.tokenizer import merge_word

print(merge_word((b"l", b"o", b"w"), (b"l", b"o")))
```

執行結果：

```text
(b'lo', b'w')
```

### 起始詞彙表

詞彙表是「id → bytes」的字典。先放 256 個 byte，再放特殊 token：

```python,ignore
{{#include ../../../src/llmz/tokenizer.py:base_vocab}}
```

### 訓練迴圈（教學版）

```python,ignore
{{#include ../../../src/llmz/tokenizer.py:train_bpe_naive}}
```

每一輪：統計 pair → 挑最多的（次數相同取較大的）→ 記進 `merges` → 詞彙表多一個 token → 用 `merge_word` 更新每一個字。直到詞彙表達到 `vocab_size`，或者沒有 pair 可合併（語料太小）。

`merges` 是一個**有順序**的串列，第 9 集的 encode 會照同樣的順序套用，所以它和 `vocab` 一樣重要，都要存下來。

```python
from llmz.tokenizer import train_bpe_naive

text = "low low low lower lowest<|endoftext|>new newer newest"
vocab, merges = train_bpe_naive(text, 265, ["<|endoftext|>"])
print(len(vocab))
print(vocab[256], vocab[257], vocab[264])
for pair in merges:
    print(pair)
```

執行結果：

```text
265
b'<|endoftext|>' b'ow' b' lowe'
(b'o', b'w')
(b'l', b'ow')
(b' ', b'low')
(b'n', b'e')
(b'ne', b'w')
(b's', b't')
(b'new', b'e')
(b' low', b'e')
```

詞彙表 265 = 256 個 byte + 1 個特殊 token + 8 次合併。看合併的順序：先 `ow`，再 `l` + `ow` 成 `low`，再加上前面的空白成 ` low`。`new` 也一樣三步黏起來。這就是第 3 集說的「常見的字會自己黏成一個 token」。

### 為什麼叫教學版

這個版本每一輪都把所有字的 pair **重新統計一次**。語料小的時候無所謂；標準語料有兩萬多種字、要做近萬次合併，每輪重算加起來是二十億次迴圈，Python 得跑好幾分鐘。下一集只改統計的方式，結果一模一樣，快二十倍以上。

## 重點整理

- `merge_word(word, pair)`：從左到右把相鄰的 pair 換成接起來的 bytes。
- `base_vocab(special_tokens)`：256 個 byte 加特殊 token。
- `train_bpe_naive(text, vocab_size, special_tokens)` 回傳 `(vocab, merges)`；`merges` 有順序，encode 要照同樣順序套用。
- 教學版每輪重新統計所有 pair，語料大就太慢。
