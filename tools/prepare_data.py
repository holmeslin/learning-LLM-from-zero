"""準備全書的標準語料（第 3 章第 14 集）：

1. 下載 TinyStories 訓練檔的前 160 MiB（HTTP Range，不用抓整個 2.2 GB）與整個驗證檔到 data/
2. 把訓練檔截短到最後一個完整文件
3. 在訓練檔上訓練 vocab 10000 的 BPE，存成 data/tokenizer.json
4. 把兩個檔案轉成 uint16 的 token 檔 data/train.bin、data/valid.bin

已經存在的檔案會跳過，所以中斷後重跑即可。
"""

import os
import time

from llmz.data import cut_at_last_document, download, tokenize_to_file
from llmz.tokenizer import Tokenizer, count_words_parallel, train_bpe

BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
TRAIN_BYTES = 160 * 1024 * 1024  # 標準語料：訓練檔的前 160 MiB，切在最後一個 <|endoftext|>
SPECIAL_TOKENS = ["<|endoftext|>"]
VOCAB_SIZE = 10000


def main() -> None:
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/train.txt"):
        t0 = time.time()
        n = download(BASE_URL + "TinyStoriesV2-GPT4-train.txt", "data/train.txt", TRAIN_BYTES)
        n = cut_at_last_document("data/train.txt")
        print(f"train.txt: {n:,} bytes，{time.time() - t0:.0f} 秒")
    if not os.path.exists("data/valid.txt"):
        t0 = time.time()
        n = download(BASE_URL + "TinyStoriesV2-GPT4-valid.txt", "data/valid.txt")
        print(f"valid.txt: {n:,} bytes，{time.time() - t0:.0f} 秒")

    if not os.path.exists("data/tokenizer.json"):
        t0 = time.time()
        counts = count_words_parallel("data/train.txt", SPECIAL_TOKENS, num_workers=os.cpu_count() or 4)
        t1 = time.time()
        print(f"pre-tokenize：{len(counts):,} 種不同的 pre-token，{t1 - t0:.0f} 秒")
        vocab, merges = train_bpe(counts, VOCAB_SIZE, SPECIAL_TOKENS)
        Tokenizer(vocab, merges, SPECIAL_TOKENS).save("data/tokenizer.json")
        print(f"train_bpe：{len(merges):,} 次合併，{time.time() - t1:.0f} 秒")

    tokenizer = Tokenizer.load("data/tokenizer.json")
    for split in ["valid", "train"]:
        if not os.path.exists(f"data/{split}.bin"):
            t0 = time.time()
            n = tokenize_to_file(tokenizer, f"data/{split}.txt", f"data/{split}.bin")
            size = os.path.getsize(f"data/{split}.txt")
            print(f"{split}.bin：{n:,} 個 token，{size / n:.2f} bytes/token，{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
