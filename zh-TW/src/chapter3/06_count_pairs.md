# 訓練 BPE：統計 pair

## 本集目標

把第 3 集的手算變成兩個函式：`count_words` 統計每個 pre-token 出現幾次，`count_pairs` 統計相鄰 pair 出現幾次。

## 正文

### 先數字，不要數 pair

160 MB 的語料有四千萬個 pre-token，但**不同**的 pre-token 只有兩萬多種（TinyStories 的用字很簡單）。所以先把每種 pre-token 出現幾次數好，之後所有統計都只對這兩萬多種做，次數用乘的。

```python,ignore
{{#include ../../../src/llmz/tokenizer.py:count_words}}
```

每個 pre-token 用第 5 集的 `split_on_special` 與第 4 集的 `pretokenize` 切出來之後，拆成「一個 byte 一個元素」的 tuple：`tuple(bytes([b]) for b in word)`。這就是第 3 集手算時 `("l", "o", "w")` 的 byte 版，只是元素是 `b'l'` 這種單一 byte 的 bytes，之後合併時兩個 bytes 可以直接用 `+` 接起來。

```python
from llmz.tokenizer import count_words

text = "low low low lower lowest<|endoftext|>new newer newest"
word_counts = count_words(text, ["<|endoftext|>"])
for word, n in word_counts.items():
    print(n, word)
```

執行結果：

```text
1 (b'l', b'o', b'w')
2 (b' ', b'l', b'o', b'w')
1 (b' ', b'l', b'o', b'w', b'e', b'r')
1 (b' ', b'l', b'o', b'w', b'e', b's', b't')
1 (b'n', b'e', b'w')
1 (b' ', b'n', b'e', b'w', b'e', b'r')
1 (b' ', b'n', b'e', b'w', b'e', b's', b't')
```

句首的 `low` 和句中的 ` low`（帶空白）是不同的 pre-token，這是第 4 集說過的。

### 再數 pair

```python,ignore
{{#include ../../../src/llmz/tokenizer.py:count_pairs}}
```

和第 3 集一模一樣：`zip(word, word[1:])` 列出相鄰 pair，一個字出現 n 次每個 pair 就加 n。

```python
from llmz.tokenizer import count_pairs, count_words

text = "low low low lower lowest<|endoftext|>new newer newest"
word_counts = count_words(text, ["<|endoftext|>"])
pairs = count_pairs(word_counts)
print(pairs.most_common(4))
```

執行結果：

```text
[((b'l', b'o'), 5), ((b'o', b'w'), 5), ((b' ', b'l'), 4), ((b'w', b'e'), 4)]
```

`(b'l', b'o')` 和 `(b'o', b'w')` 都是 5 次。照第 3 集的規則，次數相同取較大的 pair，`b'o' > b'l'`，所以第一個合併的是 `(b'o', b'w')`。下一集來做合併。

## 重點整理

- `count_words(text, special_tokens)`：切開特殊 token、pre-tokenize、每個 pre-token 拆成單 byte 的 tuple，回傳 `Counter`。
- 不同的 pre-token 只有幾萬種，之後的統計都在這個 `Counter` 上做，次數用乘的。
- `count_pairs(word_counts)`：相鄰 pair 的次數。
- 次數相同時取較大的 pair。
