# 壓縮率實驗

## 本集目標

用「每個 token 平均代表幾個 byte」衡量 tokenizer 的好壞，看它怎麼隨詞彙表大小、語料是否相近而變化，並知道標準語料 tokenizer 的實測數字。

## 正文

### 一個數字：bytes per token

同一段文字，token 越少代表 tokenizer 壓得越好：序列短、模型一次能看更多內容、算得也快。指標就是文字的 byte 數除以 token 數。

```python
from llmz.tokenizer import Tokenizer, train_bpe_naive

story = """Tom had a red ball. He liked to play with the ball in the park.
One day the ball rolled into the pond. Tom was sad.
A big dog jumped into the pond and got the ball. Tom was happy again.
He said thank you to the dog, and they played in the park all day.
"""


def bytes_per_token(tok, text):
    return len(text.encode("utf-8")) / len(tok.encode(text))


tests = ["The dog and the ball were in the park.", "The astronaut repaired the telescope.", "子曰：學而時習之"]
for vocab_size in [256, 300, 400]:
    vocab, merges = train_bpe_naive(story, vocab_size, [])
    tok = Tokenizer(vocab, merges, [])
    print(vocab_size, round(bytes_per_token(tok, story), 2), [round(bytes_per_token(tok, t), 2) for t in tests])
```

執行結果：

```text
256 1.0 [1.0, 1.0, 1.0]
300 2.34 [2.53, 1.23, 1.0]
400 3.72 [2.53, 1.23, 1.0]
```

四個欄位：詞彙表大小、在訓練文字本身的 bytes/token、以及三句測試文字的 bytes/token。

- 詞彙表 256（零次合併）每個 byte 一個 token，永遠是 1.0。
- 詞彙表越大，訓練文字壓得越好（2.34 → 3.72）。
- 跟訓練文字用字相近的句子（dog、ball、park）壓到 2.53；用字完全不同的句子（astronaut、telescope）只有 1.23，幾乎退回一個字母一個 token；中文更是 1.0，因為訓練文字裡一個中文 byte 都沒有。

**tokenizer 只對它看過的那種文字有效**。拿英文 tokenizer 切中文、拿童話 tokenizer 切程式碼，序列都會長好幾倍。

### 標準語料的實測

第 14 集的 `tools/prepare_data.py` 在 TinyStories 訓練集上訓練 vocab 10000 的 tokenizer，M4 Max 實測：

| 文字 | bytes/token |
| --- | --- |
| TinyStories 訓練集本身（40,720,162 個 token） | 4.12 |
| TinyStories 驗證集 | 4.12 |
| 一句自己寫的童話風英文 | 4.36 |
| 《論語》一段 | 1.00 |

驗證集和訓練集一樣是 4.12，代表 tokenizer 沒有「背」訓練集，童話的用字就是這麼集中。詞彙表裡最長的 token 是 ` enthusiastically`、` accomplishment`、` disappointment` 這種帶空白的整個長字，最早的幾次合併是 ` t`、`he`、` a`、` s`、` w`、`nd`、` the`。

### 速度

encode 的速度也是指標。驗證集 22.5 MB、546 萬個 token，用第 11 集的 `encode_iterable`：

| 項目 | 數值 |
| --- | --- |
| 時間 | 2.9 秒 |
| 吞吐量 | 每秒約 190 萬個 token（7.7 MB） |

純 Python 能這麼快，靠的是第 9 集的快取：兩萬五千種 pre-token 各只真正算一次，其餘都是查表。照這個速度，整個 2.2 GB 的 TinyStories 大約五分鐘；幾百 GB 的網路語料就得用 Rust 寫的 tokenizer 加上多機平行了，這是第 13 章的事。

## 重點整理

- bytes/token 衡量壓縮率：越高序列越短；詞彙表越大越高，但只對相近的文字有效。
- 沒看過的字退回單 byte，中文在英文 tokenizer 上是 1.0。
- 標準語料 tokenizer：4.12 bytes/token，encode 每秒約 190 萬個 token。
- 快取讓純 Python 的 encode 夠快，因為不同的 pre-token 只有兩萬多種。
