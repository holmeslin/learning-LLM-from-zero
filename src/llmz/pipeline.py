"""資料管線（第 13 章）：從網頁原始資料（Common Crawl 的 WARC）到乾淨的文字。

這個模組之後每一集會再加一個步驟，每一步都是「一份文件進、一份文件（或沒有）出」的小函式。
"""

import gzip
import json
import math
import re
from collections import Counter
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


# ANCHOR: char_ngrams
def char_ngrams(text: str, n: int = 3) -> Counter[str]:
    """文字的 1～n 個字元的片段（n-gram）計數。大小寫不分；標點、數字、空白只當分隔；每一段前後各補一個空格。

    「段」在英文是一個字，在中文是兩個標點之間的一串字（中文沒有空格）。補空格讓「字的開頭與結尾」也成為特徵。
    """
    grams: Counter[str] = Counter()
    for chunk in re.sub(r"[\W\d_]+", " ", text.casefold()).split():
        padded = f" {chunk} "
        for k in range(1, n + 1):
            grams.update(g for i in range(len(padded) - k + 1) if (g := padded[i : i + k]) != " ")
    return grams
# ANCHOR_END: char_ngrams


# ANCHOR: ngram_classifier
class NGramClassifier:
    """字元 n-gram 的 naive Bayes：每個標籤各有一張「片段出現幾次」的表，用 log 機率加總打分。"""

    def __init__(self, counts: dict[str, Counter[str]], n: int = 3, alpha: float = 0.1):
        self.counts, self.n, self.alpha = counts, n, alpha
        self.totals = {label: sum(c.values()) for label, c in counts.items()}
        self.vocab = set().union(*counts.values())

    @classmethod
    def train(cls, samples: dict[str, Iterable[str]], n: int = 3, alpha: float = 0.1) -> "NGramClassifier":
        counts = {label: Counter() for label in samples}
        for label, texts in samples.items():
            for text in texts:
                counts[label].update(char_ngrams(text, n))
        return cls(counts, n, alpha)

    def log_likelihoods(self, text: str) -> dict[str, float]:
        """每個標籤各一個分數：文字裡每個片段的 log P(片段 | 標籤) 加總。所有標籤都沒看過的片段直接略過。

        P(片段 | 標籤) = (這個標籤看過幾次 + alpha) / (這個標籤總共看過幾個片段 + alpha × 詞彙表大小)；alpha 讓沒看過的片段不會是 0。
        """
        grams = {g: c for g, c in char_ngrams(text, self.n).items() if g in self.vocab}
        extra = self.alpha * len(self.vocab)
        return {
            label: sum(c * math.log((counts[g] + self.alpha) / (self.totals[label] + extra)) for g, c in grams.items())
            for label, counts in self.counts.items()
        }

    def predict(self, text: str) -> tuple[str, float]:
        """回傳 (分數最高的標籤, 它的後驗機率)；後驗是各標籤分數的 softmax，先驗機率假設每個標籤一樣。

        shortcut: 沒有任何看過的片段時回傳 ("unknown", 0.0)。naive Bayes 把每個片段當成互相獨立，
        文字一長後驗就會被推到幾乎是 1，所以這個機率不能當成「有多確定」；要信心時改看每個片段的平均分數差。
        """
        scores = self.log_likelihoods(text)
        if not any(scores.values()):
            return "unknown", 0.0
        best = max(scores, key=scores.get)
        return best, 1 / sum(math.exp(s - scores[best]) for s in scores.values())

    def predict_lines(self, text: str) -> str:
        """一行一行分類再投票，每一行的票數是它的字母數；回傳得票最多的標籤。

        一頁裡夾著別種語言的導覽列、程式碼時，整頁一起分類很容易被它們拉走；逐行投票讓大宗的語言贏。
        """
        votes: Counter[str] = Counter()
        for line in text.splitlines():
            label, _ = self.predict(line)
            votes[label] += sum(ch.isalpha() for ch in line)
        votes.pop("unknown", None)
        return votes.most_common(1)[0][0] if votes else "unknown"

    def save(self, path: str | Path) -> None:
        data = {"n": self.n, "alpha": self.alpha, "counts": self.counts}
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NGramClassifier":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({label: Counter(c) for label, c in data["counts"].items()}, data["n"], data["alpha"])
# ANCHOR_END: ngram_classifier


# ANCHOR: langid
LANGID_PATH = Path(__file__).parent / "datasets" / "langid_samples.jsonl"


def load_langid_samples(split: str = "train", path: str | Path = LANGID_PATH) -> dict[str, list[str]]:
    """原創的多語短文，回傳 {語言代碼: [文字, ...]}。split 是 "train" 或 "test"。"""
    samples: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["split"] == split:
                samples.setdefault(row["lang"], []).append(row["text"])
    return samples


def train_langid(n: int = 3, alpha: float = 0.1) -> NGramClassifier:
    return NGramClassifier.train(load_langid_samples("train"), n, alpha)
# ANCHOR_END: langid


# ANCHOR: mask_pii
# re.ASCII：\w 與 \d 只算 ASCII，這樣「電話0912-000-111」貼在中文旁邊也認得出來
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}", re.ASCII)
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"  # 0～255，不接受 01 這種前導零
IPV4_RE = re.compile(rf"(?<![\w.]){_OCTET}(?:\.{_OCTET}){{3}}(?!\w|\.\d)", re.ASCII)
# 電話：「+」開頭的連續數字，或（國碼）、區碼、再加 2～4 組 2～4 位數，用空格、點或減號隔開（括號可有可無）。
# 不算電話：開頭像日期（2026-11-03、03.11.2026）、前後緊貼著別的數字組（一長串數字的表格）、後面接「:數字」（時間）或「.數字」（小數）
PHONE_RE = re.compile(
    r"(?<![\w.+-])(?<!\d[ .-])(?!\d{4}[-./]\d{1,2}[-./]\d|\d{1,2}[-./]\d{1,2}[-./]\d{4}\b)"
    r"(?:\+\d{9,15}|(?:\+\d{1,3}[ .-]?)?\(?\d{1,4}\)?(?:[ .-]\(?\d{2,4}\)?){2,4})(?![\w-]|[:. ]\d)",
    re.ASCII,
)
PII_PATTERNS = [("email", EMAIL_RE, "<EMAIL>"), ("ip", IPV4_RE, "<IP>"), ("phone", PHONE_RE, "<PHONE>")]


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """把 email、IPv4 與電話號碼換成 <EMAIL>、<IP>、<PHONE>。回傳 (遮蔽後的文字, 各類遮蔽了幾個)。

    電話另外要求數字總共 9～15 個（E.164 的上限是 15），日期與短的數字串因此不會被誤遮蔽。
    shortcut: 只認 IPv4 與有分隔的電話格式（沒有分隔的 0912000111 不算），不處理姓名與地址；要更準時換成專門的 PII 偵測工具。
    """
    counts = dict.fromkeys(("email", "phone", "ip"), 0)

    def replacer(kind: str, token: str):
        def sub(m: re.Match) -> str:
            if kind == "phone" and not 9 <= sum(c.isdigit() for c in m.group()) <= 15:
                return m.group()
            counts[kind] += 1
            return token

        return sub

    for kind, pattern, token in PII_PATTERNS:  # ip 要在 phone 前面：198.51.100.77 長得像電話
        text = pattern.sub(replacer(kind, token), text)
    return text, counts
# ANCHOR_END: mask_pii
