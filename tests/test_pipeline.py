import gzip
import random

import numpy as np
import pytest

from llmz.pipeline import (
    Page,
    count_lines,
    decode_html,
    dedup_lines,
    drop_near_duplicates,
    html_to_text,
    iter_pages,
    iter_warc_records,
    jaccard,
    line_hash,
    lsh_candidates,
    lsh_probability,
    minhash_signature,
    read_pages,
    shingles,
    split_http_response,
    verify_pairs,
    write_pages,
)


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
