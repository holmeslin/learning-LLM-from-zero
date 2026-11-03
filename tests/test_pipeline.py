import gzip
import random

import torch
import torch.nn.functional as F

from llmz.pipeline import (
    Page,
    decode_html,
    QualityClassifier,
    binary_cross_entropy,
    gopher_check,
    hash_ngrams,
    html_to_text,
    iter_pages,
    iter_warc_records,
    read_pages,
    split_http_response,
    train_quality,
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
