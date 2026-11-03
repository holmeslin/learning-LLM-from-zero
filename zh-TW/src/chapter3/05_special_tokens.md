# 特殊 token

## 本集目標

知道 `<|endoftext|>` 這種特殊 token 是「不能被拆、也不能跟別人合併」的整體，訓練前要先用它把文字切開。

## 正文

### 文件的邊界

訓練語料是幾百萬篇小故事接在一起的一個大檔。模型要知道一篇故事在哪裡結束、下一篇從哪裡開始，否則會學到「上一篇的結尾接著下一篇的開頭」這種沒意義的規律。做法是在每篇之間放一個標記：

```text
Once upon a time.<|endoftext|>The end.<|endoftext|>
```

`<|endoftext|>` 就是一個**特殊 token**：它在詞彙表裡佔一個 id，但不是從統計學來的，而是我們手動加的。生成時模型吐出它就代表「寫完了」（第 6 章）。

### 不切開會怎樣

直接把它丟給上一集的 `pretokenize`：

```python
from llmz.tokenizer import pretokenize

doc = "Once upon a time.<|endoftext|>The end.<|endoftext|>"
print(pretokenize(doc))
```

執行結果：

```text
[b'Once', b' upon', b' a', b' time', b'.<|', b'endoftext', b'|>', b'The', b' end', b'.<|', b'endoftext', b'|>']
```

它被 regex 拆成了 `.<|`、`endoftext`、`|>` 三塊，而且句號還跟 `<|` 黏在一起。BPE 接著會學出一堆跟 `<|` 有關的合併，全是垃圾。

### 先切開

所以統計之前，先用特殊 token 把文字切成段，每段分開做 pre-tokenization：

```python,ignore
{{#include ../../../src/llmz/tokenizer.py:split_on_special}}
```

`re.escape` 把 `<|endoftext|>` 裡的 `|` 這種對 regex 有意義的符號跳脫掉；`"|".join` 把多個特殊 token 用「或」接起來；`sorted(..., key=len, reverse=True)` 讓比較長的特殊 token 先比對（如果同時有 `<|a|>` 和 `<|a|><|a|>`，要先認出長的）。這裡用標準函式庫的 `re` 就夠了。

```python
from llmz.tokenizer import split_on_special

doc = "Once upon a time.<|endoftext|>The end.<|endoftext|>"
print(split_on_special(doc, ["<|endoftext|>"]))
```

執行結果：

```text
['Once upon a time.', 'The end.', '']
```

切開之後每一段再各自 `pretokenize`，特殊 token 本身不參與統計，所以永遠不會跟任何東西合併。最後一段是空字串，因為檔案以 `<|endoftext|>` 結尾，不影響統計。

### 詞彙表裡的位置

本書的詞彙表順序固定：id 0～255 是 256 個 byte，接著是特殊 token（`<|endoftext|>` 是 256），之後才是 BPE 學出來的合併。第 7 集會看到這個起始詞彙表怎麼建。

## 重點整理

- 特殊 token（如 `<|endoftext|>`）標記文件邊界，是手動加進詞彙表的整體，不拆、不合併。
- 不先切開，regex 會把它拆爛，BPE 會學到垃圾。
- `split_on_special(text, special_tokens)` 用 `re.escape` + `re.split` 把文字切成不含特殊 token 的段落。
- 詞彙表順序：256 個 byte → 特殊 token → 合併出來的 token。
