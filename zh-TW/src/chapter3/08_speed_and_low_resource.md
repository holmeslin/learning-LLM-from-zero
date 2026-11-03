# 訓練 BPE：加速與低資源做法

## 本集目標

用兩招讓 BPE 訓練在 160 MB 語料上幾十秒內完成、記憶體只要幾百 MB：pre-tokenization 用多個 process 平行做，合併時只更新受影響的 pair。

## 正文

### 第一招：平行統計

統計 pre-token 是「把檔案讀過一遍、regex 切、丟進 `Counter`」，每一段文字互不相干，可以拆給多個 CPU 核心同時做。切的位置要對齊 `<|endoftext|>`，才不會把一篇故事切成兩半：

```python,ignore
{{#include ../../../src/llmz/tokenizer.py:count_words_parallel}}
```

`_count_chunk` 負責一塊：用 `seek` 跳到起點、讀到終點、decode、呼叫第 6 集的 `count_words`。主函式先算好每一塊的邊界（從每 8 MB 處往後找最近的 `<|endoftext|>`），再用標準函式庫的 `multiprocessing.Pool` 把這些工作分給 `num_workers` 個 process，回來的 `Counter` 用 `update` 加總。

用 `Pool` 的程式一定要放在 `if __name__ == "__main__":` 底下，不然 macOS 上每個子 process 會重新執行整個檔案。

### 第二招：只更新受影響的 pair

合併 `(b'l', b'ow')` 時，只有**含有這個 pair 的字**會變，其他字的 pair 次數完全不動。所以：

1. 一開始統計一次所有 pair 的次數，並記下「每個 pair 出現在哪些字裡」。
2. 每一輪挑出最多的 pair 之後，只對含有它的字：先把這個字舊的 pair 次數全部扣掉，`merge_word` 之後再把新的 pair 加回來。

```python,ignore
{{#include ../../../src/llmz/tokenizer.py:train_bpe}}
```

輸入從 `text` 變成 `word_counts`（平行統計的結果）。`where` 是「pair → 字的索引集合」，`pairs` 是次數；兩者同步增減。挑最多的那一行和教學版一樣，所以結果**完全相同**：

```python
from llmz.tokenizer import count_words, train_bpe, train_bpe_naive

text = "low low low lower lowest<|endoftext|>new newer newest"
special = ["<|endoftext|>"]
naive = train_bpe_naive(text, 265, special)
fast = train_bpe(count_words(text, special), 265, special)
print(naive == fast)
```

執行結果：

```text
True
```

`tests/test_tokenizer.py` 也在一段更長的文字上做同樣的比對。

### 實測

標準語料（第 14 集會定義：TinyStories 訓練檔的前 160 MiB）在 M4 Max 上，用 `tools/prepare_data.py`：

| 步驟 | 時間 | 備註 |
| --- | --- | --- |
| 平行統計 pre-token（14 個 process） | 3 秒 | 單一 process 要 26 秒 |
| `train_bpe` 到 vocab 10000（9,743 次合併） | 13 秒 | 教學版做到 vocab 1000 就要 28 秒（正式版 0.9 秒），推算到 10000 要六分鐘 |

記憶體只要放 25,553 種 pre-token 和它們的 pair，幾百 MB，16 GB 的機器綽綽有餘。關鍵不是程式寫得多巧，而是**統計的對象是「不同的字」而不是「每一個字」**：四千萬個 pre-token只有兩萬五千種。

### 還可以更快

挑最多的 pair 那一行每輪要掃一次所有 pair，用 heap 可以再快，但程式會複雜一截；對本書的語料大小沒必要。真正的 tokenizer 函式庫（例如 Hugging Face 的 tokenizers）用 Rust 寫，同樣的事快上百倍，原理一樣。

## 重點整理

- 平行統計：按 `<|endoftext|>` 切塊，`multiprocessing.Pool` 分給多個 process，`Counter.update` 加總；要放在 `if __name__ == "__main__":` 底下。
- 增量更新：記住每個 pair 在哪些字裡，合併時只扣掉與加回受影響的字的 pair。
- `train_bpe(word_counts, vocab_size, special_tokens)` 結果與 `train_bpe_naive` 完全相同，標準語料 13 秒。
- 省資源的關鍵是只處理「不同的 pre-token」。
