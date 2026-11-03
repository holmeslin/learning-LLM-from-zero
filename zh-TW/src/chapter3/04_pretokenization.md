# pre-tokenization 與 GPT-2 regex

## 本集目標

在 BPE 之前先用一條正規表示式把文字切成小片段（pre-token），讓合併只在片段內發生，不會黏出 `cat.` 這種跨單字、跨標點的 token。

## 正文

### 為什麼要先切

如果對整段文字直接做 BPE，`t` 和 `.` 相鄰的次數很多，就會黏出 `t.`，接著 `cat.`、`cat,`……同一個字配上不同標點各佔一個 token，詞彙表就浪費了。解法：先把文字切成「一個單字」「一個標點」這種小片段，BPE 只在片段裡面合併。

### GPT-2 的規則

切法用正規表示式描述。本書照 GPT-2 的規則，寫在 `src/llmz/tokenizer.py`：

```python,ignore
{{#include ../../../src/llmz/tokenizer.py:pattern}}
```

這條用的是 `regex` 套件而不是標準函式庫的 `re`，因為需要 `\p{L}`（任何語言的字母）、`\p{N}`（任何數字）。由左到右五種情況：

| 片段 | 意思 |
| --- | --- |
| `'(?:[sdmt]\|ll\|ve\|re)` | 英文縮寫的尾巴：`'s`、`'t`、`'ll`…… |
| ` ?\p{L}+` | 可有可無的一個空白，接一串字母 |
| ` ?\p{N}+` | 可有可無的一個空白，接一串數字 |
| ` ?[^\s\p{L}\p{N}]+` | 可有可無的一個空白，接一串標點符號 |
| `\s+(?!\S)`、`\s+` | 剩下的空白 |

```python
import regex

from llmz.tokenizer import GPT2_PATTERN

print(regex.findall(GPT2_PATTERN, "Hello, world! I'm 25 years old."))
```

執行結果：

```text
['Hello', ',', ' world', '!', ' I', "'m", ' 25', ' years', ' old', '.']
```

注意空白黏在**後面那個字的前面**：`' world'` 而不是 `'world '`。這樣「句首的字」和「句中的字」是不同的 pre-token，模型可以分辨；而標點永遠自己一組。

### 轉成 bytes

上一集說 BPE 的最小單位是 byte，所以每個片段切出來後立刻 `encode` 成 bytes：

```python,ignore
{{#include ../../../src/llmz/tokenizer.py:pretokenize}}
```

```python
from llmz.tokenizer import pretokenize

print(pretokenize("Hello, world!"))
```

執行結果：

```text
[b'Hello', b',', b' world', b'!']
```

`regex.finditer` 和 `findall` 一樣，但一次給一個結果，不會先把整個串列建出來，處理大檔比較省記憶體。

### 空白與中文

```python
import regex

from llmz.tokenizer import GPT2_PATTERN

print(regex.findall(GPT2_PATTERN, "  two  spaces\n\nnew line 你好嗎？"))
```

執行結果：

```text
[' ', ' two', ' ', ' spaces', '\n', '\n', 'new', ' line', ' 你好嗎', '？']
```

連續空白被拆開，換行各自一個片段。中文沒有空白分詞，`\p{L}+` 會把整串連續的中文字當成一個片段，BPE 得在片段裡自己學出詞的邊界，第 12 集會看到效果。

## 重點整理

- pre-tokenization 先把文字切成小片段，BPE 只在片段內合併，避免 `cat.` 這種 token。
- GPT-2 的 regex 五種情況：縮寫、字母串、數字串、標點串、空白；空白黏在後面的字前面。
- 用 `regex` 套件才有 `\p{L}`、`\p{N}`。
- `pretokenize(text)` 回傳每個片段的 UTF-8 bytes。
