"""資料管線（第 13 章）：從網頁原始資料（Common Crawl 的 WARC）到乾淨的文字。

這個模組之後每一集會再加一個步驟，每一步都是「一份文件進、一份文件（或沒有）出」的小函式。
"""

import gzip
import json
import re
import string
import zlib
from collections.abc import Iterable, Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor

from llmz.functional import cross_entropy
from llmz.optim import AdamW


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


# ANCHOR: gopher_check
GOPHER_STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}
BULLETS = ("•", "●", "▪", "‣", "·", "-", "*")
ELLIPSES = ("...", "…")


def gopher_check(text: str) -> tuple[bool, list[str]]:
    """Gopher 的品質規則（Rae 等人 2021，MassiveWeb）。回傳 (有沒有全部通過, 沒通過的規則名稱)。

    「詞」就是用空白切出來的一段；「行」就是非空的一行。每條規則各自判斷，所以一份文件可以同時沒通過好幾條。

    shortcut: 用空白切詞，中文、日文整段連在一起會被當成一個詞；這些規則是為英文寫的，其他語言要先過語言辨識，或換一組規則。
    """
    words = text.split()
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    n_words = max(len(words), 1)  # 空字串也要能算：比例一律除以至少 1
    n_lines = max(len(lines), 1)
    failed = []
    if not 50 <= len(words) <= 100_000:
        failed.append("word_count")
    if not 3 <= sum(len(w) for w in words) / n_words <= 10:
        failed.append("mean_word_length")
    if text.count("#") / n_words > 0.1 or sum(text.count(e) for e in ELLIPSES) / n_words > 0.1:  # 兩種符號各自算
        failed.append("symbol_ratio")
    if sum(line.startswith(BULLETS) for line in lines) / n_lines > 0.9:
        failed.append("bullet_lines")
    if sum(line.endswith(ELLIPSES) for line in lines) / n_lines > 0.3:
        failed.append("ellipsis_lines")
    if sum(any(c.isalpha() for c in w) for w in words) / n_words < 0.8:
        failed.append("alpha_words")
    if len(GOPHER_STOP_WORDS & {w.strip(string.punctuation).lower() for w in words}) < 2:
        failed.append("stop_words")
    return not failed, failed
# ANCHOR_END: gopher_check


# ANCHOR: hash_ngrams
WORD_RE = re.compile(r"\w+")


def hash_ngrams(text: str, n_buckets: int, max_words: int = 256) -> list[int]:
    """文字變成一串 bucket 編號：前 max_words 個詞（小寫）的每個詞、每一對相鄰的詞，各自雜湊到 [0, n_buckets)。

    用 zlib.crc32，同一個字串永遠得到同一個數字；Python 內建的 hash() 每次啟動都不一樣，存下來的權重就對不上了。
    """
    words = WORD_RE.findall(text.lower())[:max_words]
    grams = words + [f"{a} {b}" for a, b in zip(words, words[1:])]
    return [zlib.crc32(g.encode()) % n_buckets for g in grams]
# ANCHOR_END: hash_ngrams


# ANCHOR: quality_classifier
class QualityClassifier(torch.nn.Module):
    """hashed n-gram 的 logistic regression：每個 bucket 一個權重，文件的分數是它所有 n-gram 的權重的平均，加上 bias。"""

    def __init__(self, n_buckets: int = 1 << 16):
        super().__init__()
        self.n_buckets = n_buckets
        self.weight = torch.nn.Parameter(torch.zeros(n_buckets))
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, ids: Tensor, mask: Tensor) -> Tensor:
        """ids、mask 形狀 (B, L)；mask 是 0 的位置是 padding，不算。回傳每篇的 logit，形狀 (B,)。"""
        return (self.weight[ids] * mask).sum(-1) / mask.sum(-1).clamp(min=1) + self.bias  # 就是一維的 Embedding，再取平均

    @torch.no_grad()
    def score(self, texts: list[str]) -> Tensor:
        """每篇文件的 logit：大於 0 表示模型認為「像標準語料」的機率超過一半；torch.sigmoid(score) 是機率。

        shortcut: 一次把所有文件補齊成同樣長，記憶體跟文件數成正比；幾萬篇以上要分段呼叫。
        """
        return self(*pad_docs([hash_ngrams(t, self.n_buckets) for t in texts]))


def pad_docs(docs: list[list[int]]) -> tuple[Tensor, Tensor]:
    """每篇長度不一的 bucket 編號補齊成 (B, L)，並給一個標示真正內容的 mask。"""
    ids = torch.zeros(len(docs), max(max(map(len, docs), default=0), 1), dtype=torch.long)
    mask = torch.zeros(ids.shape)
    for i, doc in enumerate(docs):
        ids[i, : len(doc)] = torch.tensor(doc, dtype=torch.long)
        mask[i, : len(doc)] = 1
    return ids, mask


def binary_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    """logistic regression 的 loss。把 logit z 看成兩個類別的 logits [0, z]，就是第 1 章的 cross_entropy。"""
    return cross_entropy(torch.stack([torch.zeros_like(logits), logits], dim=-1), labels)


def train_quality(
    texts: list[str],
    labels: list[int],
    n_buckets: int = 1 << 16,
    steps: int = 300,
    batch_size: int = 64,
    lr: float = 0.1,
    weight_decay: float = 0.01,
    seed: int = 0,
) -> tuple[QualityClassifier, list[float]]:
    """labels：1 是「像標準語料」的正例，0 是一般網頁的負例。每個 batch 一半正例一半負例，因為兩邊的數量通常差很多。

    回傳訓練好的分類器與每一步的 loss。
    """
    docs = [hash_ngrams(t, n_buckets) for t in texts]  # 只雜湊一次，訓練時一直重用
    y = torch.tensor(labels)
    pos, neg = (y == 1).nonzero().flatten(), (y == 0).nonzero().flatten()
    gen = torch.Generator().manual_seed(seed)
    model = QualityClassifier(n_buckets)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    for _ in range(steps):
        half = batch_size // 2
        idx = torch.cat([pos[torch.randint(len(pos), (half,), generator=gen)], neg[torch.randint(len(neg), (half,), generator=gen)]])
        ids, mask = pad_docs([docs[i] for i in idx.tolist()])
        loss = binary_cross_entropy(model(ids, mask), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, losses
# ANCHOR_END: quality_classifier
