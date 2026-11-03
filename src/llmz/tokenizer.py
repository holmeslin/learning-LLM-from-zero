"""BPE tokenizer（第 3 章）：從 bytes 訓練出詞彙表與合併規則，再把文字變成 token id。

全書不用 tiktoken 或 transformers，這裡是我們自己寫的版本；
tests/test_tokenizer.py 用手算的小例子與往返（encode 再 decode）驗證。
"""

import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from multiprocessing import Pool

import regex

# ANCHOR: pattern
# GPT-2 的 pre-tokenization 規則：縮寫、字母串、數字串、標點串、空白
GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# ANCHOR_END: pattern


# ANCHOR: pretokenize
def pretokenize(text: str) -> list[bytes]:
    """照 GPT-2 的規則把文字切成小片段，每一片段轉成 UTF-8 bytes。"""
    return [m.group().encode("utf-8") for m in regex.finditer(GPT2_PATTERN, text)]
# ANCHOR_END: pretokenize


# ANCHOR: split_on_special
def split_on_special(text: str, special_tokens: list[str]) -> list[str]:
    """用特殊 token 把文字切開，回傳不含特殊 token 的各段文字。"""
    if not special_tokens:
        return [text]
    pattern = "|".join(re.escape(s) for s in sorted(special_tokens, key=len, reverse=True))
    return re.split(pattern, text)
# ANCHOR_END: split_on_special


# ANCHOR: count_words
def count_words(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    """統計每個 pre-token 出現幾次。每個 pre-token 先拆成一個一個 byte 的 tuple。"""
    counts: Counter[tuple[bytes, ...]] = Counter()
    for chunk in split_on_special(text, special_tokens):
        for word in pretokenize(chunk):
            counts[tuple(bytes([b]) for b in word)] += 1
    return counts
# ANCHOR_END: count_words


# ANCHOR: count_pairs
def count_pairs(word_counts: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
    """統計所有相鄰 pair 出現幾次（一個字出現 n 次，裡面的 pair 就各算 n 次）。"""
    pairs: Counter[tuple[bytes, bytes]] = Counter()
    for word, n in word_counts.items():
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] += n
    return pairs
# ANCHOR_END: count_pairs


# ANCHOR: merge_word
def merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """把 word 裡每一個相鄰的 pair 合併成一個 token。"""
    merged = pair[0] + pair[1]
    out: list[bytes] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)
# ANCHOR_END: merge_word


# ANCHOR: base_vocab
def base_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """起始詞彙表：256 個 byte，再加上特殊 token。"""
    vocab = {i: bytes([i]) for i in range(256)}
    for s in special_tokens:
        vocab[len(vocab)] = s.encode("utf-8")
    return vocab
# ANCHOR_END: base_vocab


# ANCHOR: train_bpe_naive
def train_bpe_naive(
    text: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """教學版 BPE 訓練：每一輪重新統計所有 pair，合併最常見的那一對。"""
    vocab = base_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []
    word_counts = count_words(text, special_tokens)
    while len(vocab) < vocab_size:
        pairs = count_pairs(word_counts)
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))  # 次數相同時取 bytes 較大的那一對
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]
        word_counts = Counter({merge_word(w, best): n for w, n in word_counts.items()})
    return vocab, merges
# ANCHOR_END: train_bpe_naive


# ANCHOR: count_words_parallel
def _count_chunk(args: tuple[str, int, int, list[str]]) -> Counter[tuple[bytes, ...]]:
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return count_words(text, special_tokens)


def count_words_parallel(
    path: str, special_tokens: list[str], num_workers: int = 8, chunk_bytes: int = 8_000_000
) -> Counter[tuple[bytes, ...]]:
    """把大檔切成幾塊，每塊在第一個特殊 token 處對齊，用多個 process 同時統計。"""
    boundary = special_tokens[0].encode("utf-8")
    with open(path, "rb") as f:
        size = f.seek(0, 2)
        cuts = [0]
        pos = chunk_bytes
        while pos < size:
            f.seek(pos)
            found = f.read(chunk_bytes).find(boundary)  # 往後找最近的文件邊界
            if found < 0:
                break
            cuts.append(pos + found)
            pos = cuts[-1] + chunk_bytes
        cuts.append(size)
    jobs = [(path, a, b, special_tokens) for a, b in zip(cuts, cuts[1:])]
    total: Counter[tuple[bytes, ...]] = Counter()
    with Pool(num_workers) as pool:
        for counts in pool.imap_unordered(_count_chunk, jobs):
            total.update(counts)
    return total
# ANCHOR_END: count_words_parallel


# ANCHOR: train_bpe
def train_bpe(
    word_counts: Counter[tuple[bytes, ...]], vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """正式版 BPE 訓練：pair 的次數只在受影響的字上增減，不每輪重算。結果與教學版完全相同。"""
    vocab = base_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []
    words = list(word_counts)
    counts = list(word_counts.values())
    pairs: Counter[tuple[bytes, bytes]] = Counter()
    where: dict[tuple[bytes, bytes], set[int]] = {}  # 每個 pair 出現在哪些字裡
    for i, word in enumerate(words):
        for pair in zip(word, word[1:]):
            pairs[pair] += counts[i]
            where.setdefault(pair, set()).add(i)
    while len(vocab) < vocab_size and pairs:
        best = max(pairs, key=lambda p: (pairs[p], p))
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]
        for i in list(where[best]):
            old, n = words[i], counts[i]
            for pair in zip(old, old[1:]):  # 這個字舊的 pair 全部退掉
                pairs[pair] -= n
                where[pair].discard(i)
                if pairs[pair] <= 0:
                    del pairs[pair]
            new = merge_word(old, best)
            words[i] = new
            for pair in zip(new, new[1:]):  # 再把新的 pair 加回來
                pairs[pair] += n
                where.setdefault(pair, set()).add(i)
    return vocab, merges
# ANCHOR_END: train_bpe


class Tokenizer:
    """用訓練好的 vocab 與 merges 做 encode／decode。"""

    # ANCHOR: tokenizer_init
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.rank = {pair: i for i, pair in enumerate(merges)}  # 越早學到的合併越優先
        self.token_to_id = {token: i for i, token in vocab.items()}
        self.special_to_id = {s: self.token_to_id[s.encode("utf-8")] for s in self.special_tokens}
        self.cache: dict[bytes, list[int]] = {}
    # ANCHOR_END: tokenizer_init

    # ANCHOR: encode_word
    def encode_word(self, word: bytes) -> list[int]:
        """對一個 pre-token 依序套用 merges：每次合併目前 rank 最小（最早學到）的 pair。"""
        if word in self.cache:
            return self.cache[word]
        parts = [bytes([b]) for b in word]
        while len(parts) > 1:
            pairs = list(zip(parts, parts[1:]))
            best = min(pairs, key=lambda p: self.rank.get(p, len(self.rank)))
            if best not in self.rank:
                break
            parts = list(merge_word(tuple(parts), best))
        ids = [self.token_to_id[p] for p in parts]
        self.cache[word] = ids
        return ids
    # ANCHOR_END: encode_word

    # ANCHOR: encode
    def encode(self, text: str) -> list[int]:
        """文字 → token id 串列。特殊 token 原樣對應到它自己的 id。"""
        ids: list[int] = []
        if not self.special_tokens:
            pieces = [text]
        else:
            pattern = "(" + "|".join(re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True)) + ")"
            pieces = re.split(pattern, text)  # 加括號讓特殊 token 也留在結果裡
        for piece in pieces:
            if piece in self.special_to_id:
                ids.append(self.special_to_id[piece])
            else:
                for word in pretokenize(piece):
                    ids.extend(self.encode_word(word))
        return ids
    # ANCHOR_END: encode

    # ANCHOR: decode
    def decode(self, ids: list[int]) -> str:
        """token id 串列 → 文字。拼不成合法 UTF-8 的 bytes 會變成 �。"""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
    # ANCHOR_END: decode

    # ANCHOR: encode_iterable
    def encode_iterable(self, lines: Iterable[str]) -> Iterator[int]:
        """一行一行 encode，用 yield 一個一個吐出 id，整個檔案不用一次讀進記憶體。"""
        for line in lines:
            yield from self.encode(line)
    # ANCHOR_END: encode_iterable

    # ANCHOR: save_load
    def save(self, path: str) -> None:
        """存成 JSON。bytes 不能直接進 JSON，用 latin-1 把每個 byte 一對一變成字元。"""
        data = {
            "vocab": {str(i): t.decode("latin-1") for i, t in self.vocab.items()},
            "merges": [[a.decode("latin-1"), b.decode("latin-1")] for a, b in self.merges],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        vocab = {int(i): t.encode("latin-1") for i, t in data["vocab"].items()}
        merges = [(a.encode("latin-1"), b.encode("latin-1")) for a, b in data["merges"]]
        return cls(vocab, merges, data["special_tokens"])
    # ANCHOR_END: save_load
