"""資料管線（第 13 章）：從網頁原始資料（Common Crawl 的 WARC）到乾淨的文字。

這個模組之後每一集會再加一個步驟，每一步都是「一份文件進、一份文件（或沒有）出」的小函式。
"""

import gzip
import json
import re
from collections.abc import Iterable, Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import NamedTuple


class Page(NamedTuple):
    url: str
    text: str


def _parse_headers(lines: Iterable[str]) -> dict[str, str]:
    """"Name: value" 的每一行變成字典；HTTP 與 WARC 的標頭名稱都不分大小寫，所以一律轉小寫。"""
    headers = {}
    for line in lines:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return headers


# ANCHOR: iter_warc_records
def iter_warc_records(path: str | Path) -> Iterator[tuple[dict[str, str], bytes]]:
    """逐筆讀 .warc.gz，產生 (WARC 標頭, 內容)。檔案被截斷時，丟掉最後那筆不完整的，然後正常結束。

    shortcut: 用 Content-Length 讀固定長度，不檢查 WARC 版本；不是 gzip 的檔案也只會得到 0 筆而不是錯誤，要嚴格時先檢查前 2 個 byte。
    """
    with gzip.open(path, "rb") as f:  # 一個 .warc.gz 是很多段 gzip 接在一起，gzip 模組會自己一段一段接著讀
        try:
            while f.readline().startswith(b"WARC/"):  # 版本那一行，例如 WARC/1.0
                lines = []
                while line := f.readline().strip():  # 空行就是標頭的結尾
                    lines.append(line.decode("utf-8"))
                headers = _parse_headers(lines)
                length = int(headers["content-length"])
                body = f.read(length)
                f.read(4)  # 每筆記錄最後有 \r\n\r\n
                if len(body) < length:  # 檔案剛好在這筆中間結束
                    return
                yield headers, body
        except (EOFError, gzip.BadGzipFile):  # 最後一段 gzip 還沒結束檔案就沒了（切在下一段的第 1 個 byte 時是 BadGzipFile）
            return
# ANCHOR_END: iter_warc_records


# ANCHOR: split_http_response
def split_http_response(body: bytes) -> tuple[int, dict[str, str], bytes]:
    """WARC 的 response 記錄內容是一整份 HTTP 回應。回傳 (狀態碼, 標頭, 網頁的 bytes)；看起來不像 HTTP 就回傳狀態碼 0。"""
    head, sep, payload = body.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status_line = lines[0].split()
    if not sep or len(status_line) < 2 or not status_line[0].startswith("HTTP/") or not status_line[1].isdigit():
        return 0, {}, b""
    return int(status_line[1]), _parse_headers(lines[1:]), payload
# ANCHOR_END: split_http_response


# ANCHOR: decode_html
CHARSET_RE = re.compile(rb"charset\s*=\s*[\"']?([\w.:-]+)", re.IGNORECASE)
# 網頁寫的是舊名稱，實際用的是它的擴充版；瀏覽器也是這樣讀，Python 的同名解碼器不夠用
CHARSET_ALIASES = {"gb2312": "gbk", "shift_jis": "cp932", "euc-kr": "cp949", "iso-8859-1": "cp1252", "windows-874": "cp874"}


def decode_html(payload: bytes, content_type: str = "") -> str:
    """bytes 轉文字：先信 HTTP 標頭的 charset，再信網頁前 2048 bytes 裡 <meta> 的 charset，都沒有就當 UTF-8。

    宣告的編碼不認得、或解不開時換下一個；最後用 UTF-8 解，壞掉的 byte 變成 U+FFFD。開頭的 BOM 會拿掉。
    """
    for source in (content_type.encode("latin-1", "replace"), payload[:2048]):
        if m := CHARSET_RE.search(source):
            name = m.group(1).decode("ascii").lower()
            try:
                text = payload.decode(CHARSET_ALIASES.get(name, name))
                break
            except (LookupError, UnicodeDecodeError):
                pass
    else:
        text = payload.decode("utf-8", errors="replace")
    return text.removeprefix("\ufeff")
# ANCHOR_END: decode_html


# ANCHOR: html_to_text
SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "iframe", "title", "select", "nav", "footer", "aside"}
BLOCK_TAGS = {
    "address", "article", "blockquote", "br", "dd", "div", "dl", "dt", "fieldset", "figcaption", "figure", "form",
    "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "ol", "p", "pre", "section", "table", "td",
    "th", "tr", "ul",
}  # fmt: skip


class _TextExtractor(HTMLParser):
    def __init__(self, skip_tags: set[str]):
        super().__init__()  # 預設 convert_charrefs=True：&amp; 與 &nbsp; 在 handle_data 之前就換好了
        self.skip_tags = skip_tags
        self.parts: list[str] = []
        self.skip = 0  # 目前在幾層 skip_tags 裡面

    def handle_starttag(self, tag, attrs):
        self.skip += tag in self.skip_tags
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.skip_tags and self.skip:
            self.skip -= 1
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(html: str, skip_tags: set[str] = SKIP_TAGS) -> str:
    """HTML 轉純文字：去掉 skip_tags 裡的內容，每個區塊（段落、標題、清單項目）一行，行內的空白合成一個空格。

    shortcut: 沒關起來的 <nav> 之類會吃掉後面的全部文字，<pre> 的縮排也會被壓掉；要處理時改用能修復 HTML 的剖析器。
    """
    parser = _TextExtractor(skip_tags)
    parser.feed(html)
    parser.close()
    lines = (" ".join(line.split()) for line in "".join(parser.parts).split("\n"))
    return "\n".join(line for line in lines if line)
# ANCHOR_END: html_to_text


# ANCHOR: iter_pages
def iter_pages(path: str | Path) -> Iterator[Page]:
    """WARC 檔裡每個成功（狀態碼 200）的 HTML 回應，轉成 Page(網址, 文字)。"""
    for headers, body in iter_warc_records(path):
        if headers.get("warc-type") != "response":
            continue
        status, http_headers, payload = split_http_response(body)
        content_type = http_headers.get("content-type", "")
        if status == 200 and "html" in content_type.lower():
            yield Page(headers.get("warc-target-uri", ""), html_to_text(decode_html(payload, content_type)))
# ANCHOR_END: iter_pages


# ANCHOR: write_pages
def write_pages(pages: Iterable[Page], path: str | Path) -> int:
    """一頁一行的 JSON：{"url": ..., "text": ...}。文字裡的換行會被跳脫，所以一行就是一頁。回傳寫了幾頁。"""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for page in pages:
            f.write(json.dumps(page._asdict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_pages(path: str | Path) -> Iterator[Page]:
    with open(path, encoding="utf-8") as f:  # 用檔案物件逐行讀；str.splitlines 會在 U+2028 之類的字元也切開
        for line in f:
            yield Page(**json.loads(line))
# ANCHOR_END: write_pages
