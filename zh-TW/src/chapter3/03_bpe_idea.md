# BPE 的核心想法

## 本集目標

用一個手算得出來的小例子，理解 BPE 的一句話定義：反覆把「最常相鄰出現的兩個 token」合併成一個新 token。

## 正文

### 一句話

BPE（byte-pair encoding）的訓練只有一個動作，重複做：

> 找出目前所有相鄰 pair 裡出現最多次的那一對，把它合併成一個新 token，加進詞彙表。

做幾次，詞彙表就多幾個 token。要詞彙表大小 10000，就從 256 個 byte 開始做大約 9700 次。

### 手算一次

先不管 byte，用字母當最小單位。假設語料裡只有四個字，各出現幾次：

```python
from collections import Counter

words = Counter({
    ("l", "o", "w"): 5,
    ("l", "o", "w", "e", "r"): 2,
    ("n", "e", "w", "e", "r"): 6,
    ("w", "i", "d", "e", "r"): 3,
})
pairs = Counter()
for word, n in words.items():
    for a, b in zip(word, word[1:]):
        pairs[(a, b)] += n
print(pairs.most_common(3))
```

執行結果：

```text
[(('e', 'r'), 11), (('w', 'e'), 8), (('l', 'o'), 7)]
```

每個字先拆成一個一個字母的 tuple，`Counter` 是 Python 標準函式庫的計數字典。`zip(word, word[1:])` 把相鄰兩個配成 pair，一個字出現 n 次，裡面的每個 pair 就加 n。

`("e", "r")` 出現 11 次（`lower` 2 次、`newer` 6 次、`wider` 3 次），是第一個要合併的 pair。合併之後 `lower` 變成 `("l", "o", "w", "er")`，`er` 從此是一個 token。再統計一次，下一個最多的是 `("w", "er")` 8 次……一直做下去。

### 次數相同怎麼辦

兩個 pair 次數一樣時要有個固定規則，結果才能重現。本書的規則：**取比較大的那一對**（Python 比較 tuple 的順序）。

```python,ignore
best = max(pairs, key=lambda p: (pairs[p], p))
```

`key` 回傳 `(次數, pair)`，`max` 先比次數，次數相同再比 pair 本身。

### 為什麼有效

常見的字（`the`、` and`）幾輪之後就會黏成一個 token；罕見的字永遠黏不完整，留成幾段。詞彙表的內容完全由語料的統計決定，不需要人工編字典，而且任何輸入都切得出來，因為最差也能退到單一 byte。

接下來四集把這個想法變成真正能處理 160 MB 文字的程式：先切成小片段（第 4、5 集），再統計與合併（第 6、7 集），最後加速（第 8 集）。

## 重點整理

- BPE 訓練 = 重複「找最常見的相鄰 pair → 合併成新 token」。
- 統計時每個字拆成 tuple，`zip(word, word[1:])` 列出相鄰 pair，次數乘上字的出現次數。
- 次數相同時取較大的 pair，結果才可重現。
- 詞彙表由統計決定，常見的字會黏成一個 token，罕見的字留成幾段。
