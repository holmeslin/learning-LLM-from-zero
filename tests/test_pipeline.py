import gzip
import random

from llmz.pipeline import (
    Page,
    decode_html,
    html_to_text,
    iter_pages,
    iter_warc_records,
    read_pages,
    split_http_response,
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
