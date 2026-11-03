import numpy as np

from llmz.data import cut_at_last_document, load_tokens, tokenize_to_file
from llmz.tokenizer import Tokenizer, train_bpe_naive

SPECIAL = ["<|endoftext|>"]


def test_tokenize_to_file_roundtrips_through_memmap(tmp_path):
    text = "one fish two fish<|endoftext|>red fish blue fish<|endoftext|>\n"
    vocab, merges = train_bpe_naive(text, 280, SPECIAL)
    tok = Tokenizer(vocab, merges, SPECIAL)
    src = tmp_path / "a.txt"
    src.write_text(text, encoding="utf-8")
    n = tokenize_to_file(tok, src, tmp_path / "a.bin")
    data = load_tokens(tmp_path / "a.bin")
    assert data.dtype == np.uint16 and len(data) == n
    assert tok.decode(data.tolist()) == text


def test_cut_at_last_document_drops_the_partial_tail(tmp_path):
    p = tmp_path / "t.txt"
    p.write_bytes(b"doc one<|endoftext|>\ndoc two<|endoftext|>\ndoc thr")
    n = cut_at_last_document(p)
    assert p.read_bytes() == b"doc one<|endoftext|>\ndoc two<|endoftext|>\n"
    assert n == len(p.read_bytes())
