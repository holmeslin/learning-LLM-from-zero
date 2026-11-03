# encode

## 本集目標

寫出 `Tokenizer` 類別與 `encode`：把一段文字變成 token id 串列，方法是對每個 pre-token 照訓練時的順序套用合併規則。

## 正文

### 把 vocab 與 merges 裝進類別

```python,ignore
{{#include ../../../src/llmz/tokenizer.py:tokenizer_init}}
```

- `rank`：每條合併規則的順序編號。第 7 集說過 `merges` 有順序；encode 時遇到好幾個可以合併的 pair，要先合併**最早學到**（rank 最小）的那一個，才會和訓練時的結果一致。
- `token_to_id`：`vocab` 反過來，bytes → id，encode 最後一步要用。
- `special_to_id`：特殊 token 的字串 → id。
- `cache`：同一個 pre-token 第二次出現就直接查表。TinyStories 四千萬個 pre-token 只有兩萬多種，快取讓 encode 快好幾倍。

### 一個 pre-token 怎麼切

```python,ignore
{{#include ../../../src/llmz/tokenizer.py:encode_word}}
```

把 pre-token 拆成單 byte，然後重複：列出所有相鄰 pair，找 rank 最小的那一對；如果連它都不在 `rank` 裡（`get` 回傳預設值 `len(self.rank)`），代表沒有規則可以套用了，停止；否則用第 7 集的 `merge_word` 合併。最後把每一段 bytes 換成 id。

### 整段文字

```python,ignore
{{#include ../../../src/llmz/tokenizer.py:encode}}
```

和第 5 集的 `split_on_special` 幾乎一樣，差在 regex 外面多了一對括號：`re.split` 的 pattern 有括號時，被切掉的那段（也就是特殊 token 本身）也會留在結果裡。這樣我們才能遇到特殊 token 就直接給它的 id，其他段落照常 pre-tokenize 再 `encode_word`。

```python
from llmz.tokenizer import Tokenizer, train_bpe_naive

text = "low low low lower lowest<|endoftext|>new newer newest"
vocab, merges = train_bpe_naive(text, 265, ["<|endoftext|>"])
tok = Tokenizer(vocab, merges, ["<|endoftext|>"])
ids = tok.encode("lowest newer<|endoftext|>low")
print(ids)
print([tok.vocab[i] for i in ids])
print(tok.encode_word(b"lower"))
```

執行結果：

```text
[258, 101, 262, 32, 263, 114, 256, 258]
[b'low', b'e', b'st', b' ', b'newe', b'r', b'<|endoftext|>', b'low']
[258, 101, 114]
```

對照第 7 集的 `merges`：`lowest` 切成 `low` + `e` + `st`（` low` 帶空白的版本和句首的 `low` 不同，所以 `lowe` 這條規則用不上）；`<|endoftext|>` 是 256；`newer` 切成 `newe` + `r`。訓練語料太小，切法看起來有點怪，但規則就是這樣：有什麼合併規則就套什麼。

## 重點整理

- `Tokenizer(vocab, merges, special_tokens)` 預先建好 `rank`、`token_to_id`、`special_to_id` 與快取。
- `encode_word`：拆成單 byte，反覆合併 rank 最小的 pair，直到沒有規則可用。
- `encode`：用帶括號的 `re.split` 保留特殊 token，特殊 token 直接給 id，其餘 pre-tokenize 後逐個 `encode_word`。
- 套用順序必須和訓練時一致，所以用 rank 而不是「隨便挑一個能合併的」。
