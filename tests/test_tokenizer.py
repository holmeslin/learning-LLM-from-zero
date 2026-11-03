from collections import Counter

from llmz.tokenizer import (
    Tokenizer,
    count_pairs,
    count_words,
    merge_word,
    pretokenize,
    split_on_special,
    train_bpe,
    train_bpe_naive,
)

SPECIAL = ["<|endoftext|>"]
SAMPLE = (
    "The cat sat on the mat. The cat saw a bat.<|endoftext|>"
    "A small dog ran to the park and the dog was happy!<|endoftext|>"
    "你好，世界。你好嗎？<|endoftext|>"
)


def test_pretokenize_follows_gpt2_rules():
    assert pretokenize("Hello, world! I'm 25.") == [b"Hello", b",", b" world", b"!", b" I", b"'m", b" 25", b"."]


def test_split_on_special_removes_the_special_tokens():
    assert split_on_special("a<|endoftext|>b<|endoftext|>", SPECIAL) == ["a", "b", ""]
    assert split_on_special("abc", []) == ["abc"]


def test_count_pairs_and_merge_word_by_hand():
    words = Counter({(b"l", b"o", b"w"): 2, (b"l", b"o", b"g"): 1})
    assert count_pairs(words) == Counter({(b"l", b"o"): 3, (b"o", b"w"): 2, (b"o", b"g"): 1})
    assert merge_word((b"l", b"o", b"w"), (b"l", b"o")) == (b"lo", b"w")
    assert merge_word((b"a", b"a", b"a"), (b"a", b"a")) == (b"aa", b"a")


def test_naive_training_first_merge_is_the_most_frequent_pair():
    vocab, merges = train_bpe_naive("ab ab ab ac", 257, [])
    assert merges == [(b"a", b"b")]
    assert vocab[256] == b"ab"


def test_fast_training_matches_naive():
    naive = train_bpe_naive(SAMPLE, 300, SPECIAL)
    fast = train_bpe(count_words(SAMPLE, SPECIAL), 300, SPECIAL)
    assert naive == fast
    assert len(fast[0]) == 300 and len(fast[1]) == 300 - 257


def test_encode_decode_roundtrip_with_special_and_unicode():
    vocab, merges = train_bpe_naive(SAMPLE, 320, SPECIAL)
    tok = Tokenizer(vocab, merges, SPECIAL)
    text = "The cat saw 你好 🐱<|endoftext|>unseen wörds!"
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert tok.special_to_id["<|endoftext|>"] in ids
    assert all(i < len(vocab) for i in ids)


def test_encode_iterable_matches_encode():
    vocab, merges = train_bpe_naive(SAMPLE, 300, SPECIAL)
    tok = Tokenizer(vocab, merges, SPECIAL)
    lines = ["The cat\n", "sat on\n", "the mat.\n"]
    assert list(tok.encode_iterable(lines)) == tok.encode("".join(lines))


def test_decode_of_partial_utf8_gives_replacement_char():
    tok = Tokenizer({i: bytes([i]) for i in range(256)}, [], [])
    assert tok.decode([0xE4, 0xBD]) == "�"


def test_save_and_load(tmp_path):
    vocab, merges = train_bpe_naive(SAMPLE, 300, SPECIAL)
    tok = Tokenizer(vocab, merges, SPECIAL)
    tok.save(tmp_path / "tok.json")
    back = Tokenizer.load(tmp_path / "tok.json")
    assert back.vocab == vocab and back.merges == merges and back.special_tokens == SPECIAL
    assert back.encode(SAMPLE) == tok.encode(SAMPLE)
