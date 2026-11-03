import gzip
import math
import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from llmz.data import get_batch, load_tokens, tokenize_to_file
from llmz.model import ModelConfig
from llmz.pipeline import (
    Page,
    QualityClassifier,
    StepReport,
    binary_cross_entropy,
    count_lines,
    decode_html,
    dedup_lines,
    drop_near_duplicates,
    gopher_check,
    hash_ngrams,
    html_to_text,
    iter_pages,
    iter_warc_records,
    jaccard,
    keep_ascii,
    keep_by_quality,
    keep_gopher,
    keep_language,
    line_hash,
    lsh_candidates,
    lsh_probability,
    mask_pii_pages,
    minhash_signature,
    pareto_keep,
    read_pages,
    run_pipeline,
    shingles,
    split_http_response,
    train_langid,
    train_quality,
    verify_pairs,
    write_corpus_text,
    write_pages,
)
from llmz.tokenizer import Tokenizer, base_vocab
from llmz.train import TrainConfig, train

EOT = "<|endoftext|>"


def warc_record(warc_type: str, body: bytes, **headers: str) -> bytes:
    """測試用：組一筆 WARC 記錄（標頭、空行、內容、結尾的兩個 CRLF），並壓成獨立的一段 gzip。"""
    fields = {"WARC-Type": warc_type, **{k.replace("_", "-"): v for k, v in headers.items()}}
    fields["Content-Length"] = str(len(body))
    head = "WARC/1.0\r\n" + "".join(f"{k}: {v}\r\n" for k, v in fields.items())
    return gzip.compress(head.encode() + b"\r\n" + body + b"\r\n\r\n")


def http_response(payload: bytes, content_type: str = "text/html", status: str = "200 OK") -> bytes:
    head = f"HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nServer: toy\r\n\r\n"
    return head.encode() + payload


def test_iter_warc_records_reads_each_record(tmp_path):
    path = tmp_path / "a.warc.gz"
    path.write_bytes(
        warc_record("warcinfo", b"software: toy\r\n")
        + warc_record("response", b"hello\r\n\r\nworld", WARC_Target_URI="http://example.com/a")
    )
    records = list(iter_warc_records(path))
    assert [r[0]["warc-type"] for r in records] == ["warcinfo", "response"]  # 標頭名稱一律小寫
    assert records[0][1] == b"software: toy\r\n"
    assert records[1][0]["warc-target-uri"] == "http://example.com/a"
    assert records[1][1] == b"hello\r\n\r\nworld"  # 內容裡的空行不會被誤當成標頭結尾


def test_iter_warc_records_stops_cleanly_on_truncated_file(tmp_path):
    rng = random.Random(0)  # 內容要壓不下去，切在中間才真的少掉資料
    first, second, third = (warc_record("response", rng.randbytes(400)) for _ in range(3))
    for cut in [1, 5, len(third) // 2, len(third) - 20]:  # 第三筆被切在 gzip 標頭裡、中間、快結束的地方
        path = tmp_path / f"cut{cut}.warc.gz"
        path.write_bytes(first + second + third[:cut])
        assert len(list(iter_warc_records(path))) == 2
    (tmp_path / "boundary.warc.gz").write_bytes(first + second)  # 剛好切在記錄之間
    assert len(list(iter_warc_records(tmp_path / "boundary.warc.gz"))) == 2


def test_split_http_response():
    status, headers, payload = split_http_response(http_response(b"<p>hi</p>\r\n\r\nmore", "text/html; charset=GBK"))
    assert status == 200
    assert headers["content-type"] == "text/html; charset=GBK"  # 標頭名稱小寫，值保持原樣
    assert payload == b"<p>hi</p>\r\n\r\nmore"
    assert split_http_response(http_response(b"", status="404 Not Found"))[0] == 404
    assert split_http_response(b"garbage")[0] == 0  # 不像 HTTP 回應


def test_decode_html_follows_header_then_meta_then_utf8():
    text = "你好，世界"
    assert decode_html(text.encode("gbk"), "text/html; charset=gbk") == text
    assert decode_html(text.encode("big5"), "text/html; charset=Big5") == text  # 大小寫不拘
    meta = b'<html><head><meta charset="gbk"></head><body>' + text.encode("gbk")
    assert text in decode_html(meta, "text/html")
    old_style = b'<meta http-equiv="Content-Type" content="text/html; charset=big5">' + text.encode("big5")
    assert text in decode_html(old_style, "text/html")
    # 標頭寫的名稱，網頁實際用的是它的擴充版（瀏覽器也這樣處理）：解碼器本身認得的名稱不夠用
    assert decode_html("①あ".encode("cp932"), "text/html; charset=Shift_JIS") == "①あ"
    assert decode_html("镕".encode("gbk"), "text/html; charset=gb2312") == "镕"
    assert decode_html(b"it\x92s", "text/html; charset=iso-8859-1") == "it\u2019s"
    assert decode_html(b"\xa1", "text/html; charset=windows-874") == "\u0e01"
    assert decode_html(b"\xef\xbb\xbf" + text.encode(), "text/html; charset=utf-8") == text  # 開頭的 BOM 不留
    assert decode_html(text.encode(), "text/html") == text  # 都沒寫就當 UTF-8
    assert decode_html(text.encode(), "text/html; charset=no-such-codec") == text  # 不認得的名稱不能讓程式出錯
    assert decode_html(b"ok \xff\xfe ok", "text/html") == "ok \ufffd\ufffd ok"  # 壞掉的 byte 變成 U+FFFD


def test_html_to_text_drops_non_prose_and_keeps_paragraphs():
    html = """<html><head><title>Toy title</title><style>p { color: red }</style></head>
    <body>
      <nav><ul><li><a href="/">Home</a></li><li>Menu</li></ul></nav>
      <script>var x = "<p>not text</p>";</script>
      <h1>First   heading</h1>
      <p>One <b>bold</b>word &amp; an&nbsp;entity &lt;3.</p>
      <div>line<br>break</div>
      <ul><li>item a</li><li>item b</li></ul>
      <footer>Contact us</footer>
    </body></html>"""
    assert html_to_text(html) == "First heading\nOne boldword & an entity <3.\nline\nbreak\nitem a\nitem b"
    # skip_tags 可以換：只丟 script 與 style 時，導覽與頁尾的字都會留下來
    assert html_to_text(html, skip_tags={"script", "style"}).splitlines()[:3] == ["Toy title", "Home", "Menu"]


def test_html_to_text_survives_messy_html():
    assert html_to_text("") == ""
    assert html_to_text("just text") == "just text"
    assert html_to_text("<p>unclosed <b>bold<p>next</i></div>") == "unclosed bold\nnext"
    assert html_to_text("<script src='a.js'/><p>after self-closing script</p>") == "after self-closing script"
    assert html_to_text("<nav>unclosed nav<p>swallowed") == ""  # 已知的限制：沒關的 nav 會吃掉後面全部


def test_iter_pages_keeps_only_successful_html_responses(tmp_path):
    page = "<p>你好，世界</p>".encode("gbk")
    path = tmp_path / "mix.warc.gz"
    path.write_bytes(
        warc_record("warcinfo", b"software: toy\r\n")
        + warc_record("request", b"GET / HTTP/1.1\r\n", WARC_Target_URI="http://a.example/")
        + warc_record("response", http_response(page, "text/html; charset=gbk"), WARC_Target_URI="http://a.example/")
        + warc_record("response", http_response(b"%PDF", "application/pdf"), WARC_Target_URI="http://b.example/x.pdf")
        + warc_record("response", http_response(b"<p>gone</p>", status="404 Not Found"), WARC_Target_URI="http://c.example/")
        + warc_record("response", http_response(b"<p>second</p>"), WARC_Target_URI="http://d.example/")
        + warc_record("metadata", b"fetchTimeMs: 3\r\n")
    )
    assert list(iter_pages(path)) == [Page("http://a.example/", "你好，世界"), Page("http://d.example/", "second")]


def test_write_and_read_pages_roundtrip(tmp_path):
    pages = [Page("http://a.example/", "第一行\n第二行"), Page("http://b.example/", 'quote " and \\ and \u2028 inside')]
    path = tmp_path / "pages.jsonl"
    assert write_pages(pages, path) == 2
    assert path.read_text(encoding="utf-8").count("\n") == 2  # 一頁一行，文字裡的換行被跳脫
    assert list(read_pages(path)) == pages


# ---- 第 13 章第 6 集：Gopher 品質規則 ----


def words_doc(words: list[str], per_line: int = 10) -> str:
    return "\n".join(" ".join(words[i : i + per_line]) for i in range(0, len(words), per_line))


def plain_words(n: int) -> list[str]:
    """n 個「正常」的詞：含 the、and 兩個停用詞，平均詞長 3.875，全是字母。"""
    base = ["the", "dog", "and", "that", "bird", "walked", "to", "market"]
    return [base[i % len(base)] for i in range(n)]


def failed_rules(words: list[str], per_line: int = 10) -> list[str]:
    passed, failed = gopher_check(words_doc(words, per_line))
    assert passed == (not failed)
    return failed


def test_gopher_passes_ordinary_text():
    assert gopher_check(words_doc(plain_words(80))) == (True, [])


def test_gopher_word_count_boundaries():
    assert failed_rules(plain_words(49)) == ["word_count"]
    assert failed_rules(plain_words(50)) == []
    assert failed_rules(plain_words(100_000), per_line=100) == []
    assert failed_rules(plain_words(100_001), per_line=100) == ["word_count"]


def test_gopher_mean_word_length_boundaries():
    def mean_len(total_chars: int) -> list[str]:  # 50 個詞：the、and 加 47 個 10 字元的詞，最後一個詞補到總字元數
        return ["the", "and"] + ["a" * 10] * 47 + ["a" * (total_chars - 6 - 470)]

    assert failed_rules(mean_len(500)) == []  # 平均剛好 10
    assert failed_rules(mean_len(501)) == ["mean_word_length"]
    assert failed_rules(["the", "and"] + ["cat"] * 48) == []  # 平均剛好 3
    assert failed_rules(["the", "and"] + ["cat"] * 47 + ["at"]) == ["mean_word_length"]


def test_gopher_symbol_ratio_boundaries():
    for symbol, name in [("#", "hash"), ("...", "ellipsis"), ("…", "ellipsis")]:
        words = plain_words(60)
        for i in range(6):  # 6 / 60 = 0.1：不算「大於」
            words[i] = symbol + words[i]
        assert failed_rules(words) == [], name
        words[6] = symbol + words[6]  # 7 / 60 > 0.1
        assert failed_rules(words) == ["symbol_ratio"], name


def test_gopher_symbol_ratio_hash_and_ellipsis_are_counted_separately():
    words = plain_words(60)
    for i in range(6):
        words[i] = "#" + words[i]
        words[i + 6] = words[i + 6] + "…"  # 各 6 個：合起來 0.2，但各自都沒超過 0.1
    assert failed_rules(words, per_line=1000) == []  # 全部放同一行，不觸發逐行規則


def test_gopher_bullet_lines_boundaries():
    lines = ["the dog and that bird walked to market again today"] * 10  # 一行 10 個詞，共 100 個詞
    assert gopher_check("\n".join(["• " + l for l in lines[:9]] + lines[9:])) == (True, [])  # 9/10 = 0.9：不算「超過」
    assert gopher_check("\n".join("• " + l for l in lines)) == (False, ["bullet_lines"])
    assert gopher_check("\n".join("- " + l for l in lines)) == (False, ["bullet_lines"])


def test_gopher_ellipsis_lines_boundaries():
    lines = ["the dog and that bird walked to market again today"] * 10
    assert gopher_check("\n".join([l + "..." for l in lines[:3]] + lines[3:])) == (True, [])  # 3/10 = 0.3：不算「超過」
    assert gopher_check("\n".join([l + "…  " for l in lines[:4]] + lines[4:])) == (False, ["ellipsis_lines"])  # 行尾空白不影響


def test_gopher_alphabetic_word_boundaries():
    words = plain_words(50)
    for i in range(10):  # 40/50 = 0.8 的詞有字母：剛好達標
        words[i] = "1234"
    assert failed_rules(words) == []
    words[10] = "1234"  # 39/50
    assert failed_rules(words) == ["alpha_words"]


def test_gopher_stop_word_boundaries():
    words = ["cat"] * 60
    assert failed_rules(words) == ["stop_words"]
    assert failed_rules(["the"] * 30 + words) == ["stop_words"]  # 同一個停用詞出現再多次也只算一個
    assert failed_rules(["The,"] + ["and."] + words) == []  # 不分大小寫，標點不算
    assert failed_rules(["the"] + ["cat"] * 59 + ["With"]) == []


def test_gopher_reports_every_failed_rule():
    passed, failed = gopher_check("")  # 空字串不能除以 0
    assert not passed
    assert failed == ["word_count", "mean_word_length", "alpha_words", "stop_words"]
    assert gopher_check("12 34 56")[1] == ["word_count", "mean_word_length", "alpha_words", "stop_words"]


# ---- 第 13 章第 7 集：品質分類器 ----


def test_hash_ngrams_is_stable_across_processes():
    assert hash_ngrams("hello", 10**6) == [60870]  # zlib.crc32(b"hello") = 907060870（公開的校驗值）；Python 內建的 hash() 每次執行都不一樣，不能用


def test_hash_ngrams_makes_unigrams_then_bigrams():
    ids = hash_ngrams("a b c", 1 << 20)
    assert ids[:3] == [hash_ngrams(w, 1 << 20)[0] for w in "abc"]
    assert ids[3:] == [hash_ngrams("a b", 1 << 20)[-1], hash_ngrams("b c", 1 << 20)[-1]]
    assert len(ids) == 5
    assert hash_ngrams("a b", 1 << 20)[-1] != hash_ngrams("b a", 1 << 20)[-1]  # 順序不同，bigram 就不同
    assert all(0 <= i < 7 for i in hash_ngrams("some text with several words", 7))  # 一定落在 [0, n_buckets)


def test_hash_ngrams_normalizes_and_truncates():
    assert hash_ngrams("The  CAT, sat!", 1000) == hash_ngrams("the cat sat", 1000)  # 小寫、標點與空白不算
    assert hash_ngrams("", 1000) == [] and hash_ngrams("?!", 1000) == []
    assert len(hash_ngrams("w " * 1000, 1000)) == 256 + 255  # 只看前 256 個詞
    assert len(hash_ngrams("w " * 1000, 1000, max_words=10)) == 10 + 9


def test_quality_classifier_averages_the_weights_of_a_documents_ngrams():
    model = QualityClassifier(8)
    with torch.no_grad():
        model.weight[3], model.weight[5], model.bias[...] = 2.0, -1.0, 0.5
    ids = torch.tensor([[3, 5, 0], [5, 5, 5]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])  # 第 1 篇最後一格是 padding，不能算進去
    assert torch.allclose(model(ids, mask), torch.tensor([(2 - 1) / 2 + 0.5, -1 + 0.5]))


def test_binary_cross_entropy_matches_torch():
    torch.manual_seed(0)
    logits, labels = torch.randn(32) * 3, torch.randint(0, 2, (32,))
    assert torch.allclose(binary_cross_entropy(logits, labels), F.binary_cross_entropy_with_logits(logits, labels.float()), atol=1e-6)
    assert torch.isfinite(binary_cross_entropy(torch.tensor([200.0, -200.0]), torch.tensor([0, 1])))  # 很離譜的 logits 也不會溢位


GOOD = ["the little cat and the dog played in the sun", "a happy girl and her mom baked a warm cake", "the boy had a red ball and a kind friend"]
BAD = ["click here to subscribe to our newsletter now", "terms of use privacy policy all rights reserved", "buy cheap pills online free shipping offer"]


def toy_texts(seed: int, n: int) -> tuple[list[str], list[int]]:
    rng = random.Random(seed)
    texts, labels = [], []
    for _ in range(n):
        label = rng.randint(0, 1)
        parts = [rng.choice([BAD, GOOD][label]) for _ in range(3)]
        texts.append(" ".join(parts))
        labels.append(label)
    return texts, labels


def test_train_quality_converges_on_separable_toy_data():
    texts, labels = toy_texts(0, 200)
    model, losses = train_quality(texts, labels, n_buckets=1 << 12, steps=60, batch_size=16, lr=0.1)
    assert losses[0] > 0.6 > 0.1 > losses[-1]  # 一開始約 ln 2 = 0.693，結束時接近 0
    test_texts, test_labels = toy_texts(1, 100)
    scores = model.score(test_texts)
    assert ((scores > 0).long() == torch.tensor(test_labels)).all()  # 沒看過的組合也分得開
    unseen = model.score(["the girl and the dog had a warm cake", "subscribe now for free shipping and cheap pills"])
    assert unseen[0] > 0 > unseen[1]


def test_train_quality_is_deterministic_and_balances_classes():
    texts, labels = toy_texts(0, 100)
    a, losses_a = train_quality(texts, labels, n_buckets=256, steps=10, batch_size=8)
    b, losses_b = train_quality(texts, labels, n_buckets=256, steps=10, batch_size=8)
    assert torch.equal(a.weight, b.weight) and losses_a == losses_b
    texts, labels = toy_texts(0, 400)  # 負例 90 篇、正例只留 5 篇
    neg = [t for t, l in zip(texts, labels) if l == 0][:90]
    pos = [t for t, l in zip(texts, labels) if l == 1][:5]
    skewed, _ = train_quality(neg + pos, [0] * 90 + [1] * 5, n_buckets=256, steps=30, batch_size=8)
    assert skewed.score([GOOD[0]]) > 0  # 每個 batch 一半一半，正例沒有被淹沒


def test_line_hash_is_a_stable_64_bit_number():
    assert line_hash("Contact us") == line_hash("Contact us")
    assert line_hash("Contact us") != line_hash("Contact us ")  # exact：差一個空格就是不同的行
    assert 0 <= line_hash("") < 2**64 and 0 <= line_hash("你好") < 2**64


def test_count_lines_counts_pages_not_occurrences():
    pages = [Page("a", "Home\nBody a\nHome"), Page("b", "Home\nBody b"), Page("c", "Body c")]
    counts = count_lines(pages)
    assert counts[line_hash("Home")] == 2  # 同一頁重複兩次只算一頁
    assert counts[line_hash("Body a")] == 1 and counts[line_hash("nope")] == 0


def test_dedup_lines_removes_every_copy_of_lines_on_many_pages():
    menu, footer = "Home | Shop | About", "Copyright 2026"
    pages = [Page(f"http://x/{i}", f"{menu}\nbody {i}\nmore {i}\n{footer}") for i in range(4)]
    kept = dedup_lines(pages, min_pages=3)
    assert [p.text for p in kept] == [f"body {i}\nmore {i}" for i in range(4)]
    assert [p.url for p in kept] == [p.url for p in pages]  # 網址與順序不變
    assert dedup_lines(pages, min_pages=5) == pages  # 門檻比頁數高就什麼都不動
    assert dedup_lines(pages[:1], min_pages=1) == []  # 一行都不剩的頁面整頁丟掉


def test_dedup_lines_keeps_lines_repeated_inside_one_page():
    pages = [Page("a", "x\nx\nx\ny"), Page("b", "z")]
    assert dedup_lines(pages, min_pages=2) == pages  # 重複只發生在同一頁，不算「很多頁都有」
    assert dedup_lines([Page("a", "")], min_pages=2) == []  # 空頁面不留


def test_shingles_are_unique_hashes_of_word_ngrams():
    got = shingles("The  quick brown\nfox JUMPS over the lazy dog", n=5)
    assert got.dtype == np.uint64 and got.size == 5  # 9 個字 → 5 個長度為 5 的視窗
    assert np.array_equal(got, shingles("the quick brown fox jumps over the lazy dog", n=5))  # 大小寫與空白不影響
    assert shingles("a b c d e a b c d e", n=5).size == 5  # 6 個視窗，第 1 個與第 6 個相同，集合只留 5 個
    assert shingles("only three words", n=5).size == 1  # 比 n 短的文件，整段當一個 shingle
    assert shingles("", n=5).size == 0 and shingles(" \n ", n=5).size == 0


def test_one_changed_word_destroys_n_shingles():
    words = [f"w{i}" for i in range(30)]
    edited = words[:15] + ["changed"] + words[16:]
    a, b = shingles(" ".join(words), n=5), shingles(" ".join(edited), n=5)
    assert a.size == b.size == 26 and np.intersect1d(a, b).size == 26 - 5  # 含第 15 個字的 5 個視窗都變了
    assert jaccard(a, b) == 21 / 31


def test_jaccard_exact():
    a, b = np.array([1, 2, 3, 4], dtype=np.uint64), np.array([3, 4, 5], dtype=np.uint64)
    assert jaccard(a, b) == 2 / 5 and jaccard(a, a) == 1.0 and jaccard(a, np.array([9], dtype=np.uint64)) == 0.0
    assert jaccard(np.array([], dtype=np.uint64), np.array([], dtype=np.uint64)) == 0.0  # 空集合不是任何東西的重複


def random_pair(rng, shared: int, only: int):
    """兩個集合各有 only 個自己獨有的元素、共有 shared 個；Jaccard = shared / (shared + 2 * only)。"""
    ids = rng.integers(0, 2**64, shared + 2 * only, dtype=np.uint64)
    return ids[: shared + only], np.concatenate([ids[:shared], ids[shared + only :]])


def test_minhash_signature_estimates_jaccard():
    rng = np.random.default_rng(0)
    errors = []
    for shared, only in [(10, 90), (100, 100), (300, 33), (190, 5), (60, 270)]:  # J ≈ 0.05、0.33、0.82、0.95、0.10
        a, b = random_pair(rng, shared, only)
        exact = jaccard(a, b)
        for seed in range(10):
            sa, sb = minhash_signature(a, 256, seed), minhash_signature(b, 256, seed)
            errors.append(float((sa == sb).mean()) - exact)
    errors = np.array(errors)  # 估計的標準差是 sqrt(J(1-J)/256)，最大約 0.031
    assert abs(errors.mean()) < 0.01  # 沒有系統性偏高或偏低
    assert np.abs(errors).max() < 0.1  # 約 3.2 倍標準差


def test_minhash_signature_properties():
    rng = np.random.default_rng(1)
    a, b = random_pair(rng, 50, 100)
    sig = minhash_signature(a, 64, 0)
    assert sig.shape == (64,) and sig.dtype == np.uint64
    assert np.array_equal(sig, minhash_signature(a[::-1].copy(), 64, 0))  # 集合沒有順序
    assert np.array_equal(sig, minhash_signature(a, 64, 0))  # 同一個 seed 同一組隨機排列
    assert not np.array_equal(sig, minhash_signature(a, 64, 1))
    assert (minhash_signature(b, 64, 0) == sig).mean() < 0.2  # J = 1/5 左右
    assert minhash_signature(a).shape == (128,)  # 預設 128 個排列


def test_minhash_signature_of_empty_set_does_not_crash():
    sig = minhash_signature(shingles(""), 8)
    assert sig.shape == (8,) and (sig == np.iinfo(np.uint64).max).all()


def test_lsh_probability_is_the_s_curve():
    assert lsh_probability(0.5, bands=1, rows=1) == 0.5
    assert lsh_probability(0.5, bands=2, rows=3) == pytest.approx(1 - (1 - 0.5**3) ** 2)
    assert lsh_probability(0.0, 16, 8) == 0.0 and lsh_probability(1.0, 16, 8) == 1.0
    curve = [lsh_probability(s / 10, 16, 8) for s in range(11)]
    assert curve == sorted(curve)  # 越相似，越容易成為候選
    assert lsh_probability(0.9, 16, 8) > 0.99 and lsh_probability(0.3, 16, 8) < 0.01


def test_lsh_candidates_bucket_by_band():
    sigs = np.array([[1, 2, 3, 4], [1, 2, 9, 9], [7, 7, 3, 4], [1, 5, 5, 5]], dtype=np.uint64)
    assert lsh_candidates(sigs, bands=2, rows=2) == {(0, 1), (0, 2)}  # 0 與 1 的第 1 段相同；0 與 2 的第 2 段相同
    assert lsh_candidates(sigs, bands=4, rows=1) == {(0, 1), (0, 2), (0, 3), (1, 3)}  # 每段只要 1 格相同就算，寬鬆得多
    assert lsh_candidates(sigs, bands=1, rows=4) == set()  # 整條簽章都要一樣
    with pytest.raises(ValueError):
        lsh_candidates(sigs, bands=3, rows=2)  # bands * rows 要等於簽章長度


def planted_near_duplicates(rng, n_other: int, n_pairs: int, n_edits: int, size: int = 300):
    """n_other 份互不相干的集合，加上 n_pairs 對「複本改掉 n_edits 個元素」的集合；複本排在最後。"""
    others = [rng.integers(0, 2**64, size, dtype=np.uint64) for _ in range(n_other)]
    bases = [rng.integers(0, 2**64, size, dtype=np.uint64) for _ in range(n_pairs)]
    copies = []
    for base in bases:
        copy = base.copy()
        copy[rng.choice(size, n_edits, replace=False)] = rng.integers(0, 2**64, n_edits, dtype=np.uint64)
        copies.append(copy)
    return others + bases + copies, [(n_other + i, n_other + n_pairs + i) for i in range(n_pairs)]


def test_lsh_finds_planted_near_duplicates_and_little_else():
    rng = np.random.default_rng(0)
    sets, planted = planted_near_duplicates(rng, n_other=200, n_pairs=30, n_edits=9)  # J = 291/309 = 0.94
    sigs = np.stack([minhash_signature(s, 128) for s in sets])
    assert lsh_candidates(sigs, bands=16, rows=8) == set(planted)  # 全部找回，而且沒有多的

    sets, planted = planted_near_duplicates(rng, n_other=0, n_pairs=40, n_edits=100)  # J = 200/400 = 0.5
    sigs = np.stack([minhash_signature(s, 128) for s in sets])
    found = lsh_candidates(sigs, bands=16, rows=8) & set(planted)
    assert len(found) < 0.3 * 40  # 理論上只有 6% 會成為候選：不夠相似的對本來就不該被找出來


def test_verify_pairs_checks_exact_jaccard():
    rng = np.random.default_rng(1)
    sets, planted = planted_near_duplicates(rng, n_other=2, n_pairs=2, n_edits=300 // 10)
    sets[-1] = sets[-1].copy()  # 第 2 對再多改 60 個元素，Jaccard 掉到約 0.67
    sets[-1][:60] = rng.integers(0, 2**64, 60, dtype=np.uint64)
    candidates = {(2, 4), (3, 5), (0, 1)}  # 兩對複本，外加一對不相干的假候選
    got = verify_pairs(sorted(candidates), sets, threshold=0.8)
    assert [(i, j) for i, j, _ in got] == [(2, 4)]
    assert got[0][2] == pytest.approx(jaccard(sets[2], sets[4])) and got[0][2] > 0.8


def words_page(rng, n_words: int, url: str) -> Page:
    vocab = [f"word{i}" for i in range(2000)]
    return Page(url, " ".join(rng.choice(vocab, n_words)))


def test_drop_near_duplicates_keeps_the_first_of_each_group():
    rng = np.random.default_rng(2)
    base = words_page(rng, 200, "http://x/base")
    edited = base.text.split()
    edited[50], edited[120] = "edited", "changed"  # 2 個字 → 10 個 shingle 不同，Jaccard 約 0.9
    near = Page("http://x/near", " ".join(edited))
    other = words_page(rng, 200, "http://x/other")
    pages = [base, other, near, Page("http://x/copy", base.text), words_page(rng, 150, "http://x/last")]
    kept = drop_near_duplicates(pages, threshold=0.8)
    assert [p.url for p in kept] == ["http://x/base", "http://x/other", "http://x/last"]
    assert drop_near_duplicates([]) == []


def test_drop_near_duplicates_threshold_and_chains():
    rng = np.random.default_rng(3)
    a = words_page(rng, 400, "a")
    b_words = a.text.split()
    b_words[100] = "x1"  # a 與 b：1 個字不同
    b = Page("b", " ".join(b_words))
    c_words = list(b_words)
    c_words[300] = "x2"  # b 與 c：再 1 個字不同；a 與 c 兩個字不同
    c = Page("c", " ".join(c_words))
    assert [p.url for p in drop_near_duplicates([a, b, c], threshold=0.9)] == ["a"]  # 三份兩兩 Jaccard 都 > 0.9
    assert [p.url for p in drop_near_duplicates([a, b, c], threshold=0.97)] == ["a", "c"]  # b 與 a 太像被丟掉；c 與 a 的 Jaccard 低於門檻，留下


def test_run_pipeline_reports_every_step_in_order():
    pages = [Page("a", "one two three"), Page("b", "four five"), Page("c", "six")]
    drop_short = lambda ps: [p for p in ps if len(p.text.split()) > 1]
    shout = lambda ps: [Page(p.url, p.text.upper()) for p in ps]
    kept, report = run_pipeline(pages, [("太短的丟掉", drop_short), ("大寫", shout)])
    assert kept == [Page("a", "ONE TWO THREE"), Page("b", "FOUR FIVE")]
    assert report == [StepReport("輸入", 3, 25), StepReport("太短的丟掉", 2, 22), StepReport("大寫", 2, 22)]  # 預設的大小是字元數
    _, report = run_pipeline(pages, [("太短的丟掉", drop_short)], size=lambda text: len(text.split()))
    assert [(r.docs, r.size) for r in report] == [(3, 6), (2, 5)]  # size 可以換成別的量法，例如 token 數
    assert run_pipeline([], [("x", drop_short)])[1] == [StepReport("輸入", 0, 0), StepReport("x", 0, 0)]


EN_PARAGRAPH = (
    "The old library at the end of the street has a small reading room with a large window that faces the garden. "
    "Every morning the librarian opens the door and sets out the newspapers, and a few regular visitors come in to read "
    "while the kettle warms in the corner. People have said that the room is the quietest place in town, and nobody "
    "wants to change that. On rainy days the children come with their parents to borrow picture books and to listen to "
    "a story that the librarian reads aloud with a different voice for every character."
)
DE_PARAGRAPH = (
    "Die alte Bibliothek am Ende der Straße hat einen kleinen Lesesaal mit einem großen Fenster, das zum Garten zeigt. "
    "Jeden Morgen öffnet die Bibliothekarin die Tür und legt die Zeitungen aus, und einige Stammgäste kommen herein, "
    "um zu lesen, während das Wasser in der Ecke warm wird. Man sagt, dass der Raum der ruhigste Ort der Stadt ist, "
    "und niemand möchte das ändern. An regnerischen Tagen kommen die Kinder mit ihren Eltern, um Bilderbücher zu leihen."
)


def test_standard_steps_filter_and_mask_in_pipeline_order():
    pages = [
        Page("en", EN_PARAGRAPH),
        Page("de", DE_PARAGRAPH),
        Page("short", "Home | About | Contact"),
        Page("mail", EN_PARAGRAPH + " Write to Mia at mia@example.com for a card."),
    ]
    steps = [("語言", keep_language(train_langid())), ("Gopher", keep_gopher), ("PII", mask_pii_pages)]
    kept, report = run_pipeline(pages, steps)
    assert [(r.name, r.docs) for r in report] == [("輸入", 4), ("語言", 3), ("Gopher", 2), ("PII", 2)]  # 德文、導覽列各被一步擋下
    assert [p.url for p in kept] == ["en", "mail"]
    assert kept[1].text.endswith("Write to Mia at <EMAIL> for a card.")
    assert keep_language(train_langid(), "de")(pages)[0].url == "de"


def test_pareto_keep_depends_on_rank_and_seed_only():
    scores = np.random.default_rng(0).normal(size=1000)
    keep = pareto_keep(scores, alpha=4, seed=3)
    assert keep.dtype == bool and keep.shape == (1000,)
    assert keep[scores.argmax()]  # 名次最高的百分位是 1，門檻 1 - 1 = 0，Pareto 的樣本一定大於 0
    assert (keep == pareto_keep(scores, alpha=4, seed=3)).all()  # 同樣的種子，同樣的結果
    assert (keep != pareto_keep(scores, alpha=4, seed=4)).any()
    assert (keep == pareto_keep(np.exp(scores) * 100 - 7, alpha=4, seed=3)).all()  # 分數的單位、平移都不影響，只看名次
    assert (keep == pareto_keep(torch.tensor(scores), alpha=4, seed=3)).all()  # score() 回傳的是 tensor
    assert pareto_keep([], alpha=4).shape == (0,)


@pytest.mark.parametrize("alpha", [2.0, 4.0, 9.0])
def test_pareto_keep_matches_the_exact_keep_probability(alpha):
    # 百分位 u 的文件留下的機率是 P(X > 1 - u) = (2 - u)^(-alpha)；在 [a, b] 區間裡取平均要積分：
    def mean_keep(a, b):
        return ((2 - b) ** (1 - alpha) - (2 - a) ** (1 - alpha)) / ((alpha - 1) * (b - a))

    n = 200_000
    keep = pareto_keep(np.arange(n), alpha=alpha, seed=0)
    for a, b in [(0, 1), (0, 0.1), (0.45, 0.55), (0.9, 1)]:  # 全部、最低的 10%、中間、最高的 10%
        got = keep[int(a * n) : int(b * n)].mean()
        assert got == pytest.approx(mean_keep(a, b), abs=0.015), (alpha, a, b)  # 每一段約 2 萬篇，抽樣的標準差 0.003
    assert mean_keep(0, 1) == pytest.approx((1 - 2 ** (1 - alpha)) / (alpha - 1))  # 整體留下的比例：alpha=4 約 29%、alpha=9 約 12%


def test_keep_by_quality_prefers_high_scoring_pages():
    texts, labels = toy_texts(0, 200)
    model, _ = train_quality(texts, labels, n_buckets=1 << 12, steps=60, batch_size=16, lr=0.1)
    test_texts, test_labels = toy_texts(1, 400)
    pages = [Page(str(i), t) for i, t in enumerate(test_texts)]
    kept = keep_by_quality(model, alpha=4, seed=0)(pages)
    kept_labels = [test_labels[int(p.url)] for p in kept]
    assert 0.2 < len(kept) / len(pages) < 0.4  # alpha=4 留下約 29%
    assert 0.7 < sum(kept_labels) / len(kept) < 0.9  # 完全分得開時，好的占 0.47 / (0.47 + 0.11) = 80%：大部分是好的，分數低的也有機會
    assert keep_by_quality(model)([]) == []


def byte_tokenizer() -> Tokenizer:
    """測試用：沒有任何 merge 的 tokenizer，每個 byte 一個 token，<|endoftext|> 是 256。"""
    return Tokenizer(base_vocab([EOT]), [], [EOT])


def test_write_corpus_text_then_chapter3_tokenize_gives_a_loadable_token_file(tmp_path):
    pages = [Page("a", "first page\nsecond line"), Page("b", f"oops {EOT} inside"), Page("c", "last")]
    assert write_corpus_text(pages, tmp_path / "web.txt") == 3
    n_tokens = tokenize_to_file(byte_tokenizer(), tmp_path / "web.txt", tmp_path / "web.bin")
    data = load_tokens(tmp_path / "web.bin")
    assert data.dtype == np.uint16 and len(data) == n_tokens
    assert (data == 256).sum() == 3  # 一頁一個結束符號；頁面文字裡原本就有的那一個被拿掉了，不會讓文件提早結束
    assert byte_tokenizer().decode(data.tolist()) == (
        f"first page\nsecond line\n{EOT}\noops  inside\n{EOT}\nlast\n{EOT}\n"
    )  # 跟第 3 章的 train.txt 同一個樣子：文件、結束符號、換行
    x, y = get_batch(data, 4, 8, torch.device("cpu"))
    assert x.shape == y.shape == (4, 8) and (x[:, 1:] == y[:, :-1]).all()  # 第 5 章的資料載入讀得進來


def test_the_token_file_trains_with_the_chapter5_loop(tmp_path):
    pages = [Page(str(i), f"page number {i % 3} says hello to the garden") for i in range(30)]
    write_corpus_text(pages, tmp_path / "web.txt")
    tokenize_to_file(byte_tokenizer(), tmp_path / "web.txt", tmp_path / "web.bin")
    cfg = TrainConfig(
        train_path=str(tmp_path / "web.bin"), valid_path=str(tmp_path / "web.bin"), out_dir=str(tmp_path / "run"),
        model=ModelConfig(vocab_size=257, context_length=16, d_model=32, n_layers=1, n_heads=4, d_ff=64),
        batch_size=8, max_steps=30, warmup_steps=3, lr_max=3e-3, log_every=10, eval_every=30, eval_batches=2,
        checkpoint_every=30, device="cpu",
    )  # fmt: skip
    records = train(cfg)
    assert records[-1]["val_loss"] < math.log(257) - 1  # 比亂猜（ln 257 = 5.55）好得多


def test_keep_ascii_catches_languages_the_classifier_has_no_label_for():
    czech = "Příliš žluťoučký kůň úpěl ďábelské ódy, a nikdo z lidí v té malé vesnici nechápal, proč to dělá. " * 3
    pages = [Page("en", EN_PARAGRAPH), Page("cs", czech), Page("mixed", "naïve café résumé coöperate " + EN_PARAGRAPH), Page("none", "1234 5678 !!!")]
    assert [p.url for p in keep_ascii()(pages)] == ["en", "mixed"]  # 沒有字母的頁面算 0%
    assert [p.url for p in keep_ascii(1.0)(pages)] == ["en"]
    assert [p.url for p in keep_ascii(0.0)(pages)] == ["en", "cs", "mixed", "none"]
    assert keep_ascii()([]) == []
