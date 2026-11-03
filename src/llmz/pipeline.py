"""資料管線（第 13 章）：從網頁原始資料（Common Crawl 的 WARC）到乾淨的文字。

這個模組之後每一集會再加一個步驟，每一步都是「一份文件進、一份文件（或沒有）出」的小函式。
"""

import gzip
import json
import math
import re
import string
import zlib
from collections import Counter
from collections.abc import Iterable, Iterator
from hashlib import blake2b
from html.parser import HTMLParser
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

import numpy as np
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


# ANCHOR: line_hash
def line_hash(line: str) -> int:
    """一行文字的 64 bit 指紋。不用內建的 hash()：字串的 hash() 每次啟動 Python 都不一樣，存下來或跨程序比對都不行。"""
    return int.from_bytes(blake2b(line.encode(), digest_size=8).digest())
# ANCHOR_END: line_hash


# ANCHOR: dedup_lines
def count_lines(pages: Iterable[Page]) -> Counter[int]:
    """每一行（用指紋代表）出現在幾頁裡；同一頁重複的行只算一頁。

    shortcut: 計數表整個放在記憶體；十億行以上要依指紋分成很多份，一份一份數。
    """
    return Counter(h for page in pages for h in {line_hash(line) for line in page.text.split("\n")})


def dedup_lines(pages: list[Page], min_pages: int = 5) -> list[Page]:
    """拿掉出現在至少 min_pages 頁的行（每一份都拿掉），再丟掉一行都不剩的頁面。空行也一起丟。"""
    counts = count_lines(pages)
    kept = []
    for page in pages:
        lines = [line for line in page.text.split("\n") if line and counts[line_hash(line)] < min_pages]
        if lines:
            kept.append(Page(page.url, "\n".join(lines)))
    return kept
# ANCHOR_END: dedup_lines


# ANCHOR: shingles
def shingles(text: str, n: int = 5) -> np.ndarray:
    """文件變成「連續 n 個字」的集合（shingle），每個 shingle 用 64 bit 指紋代表。比 n 個字還短的文件，整段當一個 shingle。

    shortcut: 用空白分字，不寫空白的語言（日文、沒有空格的中文）會分得很粗；要改成字元 n-gram 時，把 words 換成字元的串列。
    """
    words = text.lower().split()
    grams = {" ".join(words[i : i + n]) for i in range(max(1, len(words) - n + 1))} if words else set()
    return np.array([line_hash(gram) for gram in grams], dtype=np.uint64)


def jaccard(x: np.ndarray, y: np.ndarray) -> float:
    """兩個集合（元素不重複的陣列）的 Jaccard 相似度：交集大小 / 聯集大小。兩個都是空集合時回傳 0。"""
    shared = np.intersect1d(x, y, assume_unique=True).size
    return shared / max(1, x.size + y.size - shared)
# ANCHOR_END: shingles


# ANCHOR: minhash_signature
def minhash_signature(hashes: np.ndarray, num_perm: int = 128, seed: int = 0) -> np.ndarray:
    """集合變成 num_perm 個數字：第 i 個是「第 i 種隨機排列下，集合裡最小的那個元素」。

    排列用 x -> a*x + b（mod 2^64，uint64 乘法溢位時自動捨去高位），a 是奇數，所以是一一對應。
    兩個集合的簽章在某一格相同的機率，就是它們的 Jaccard 相似度。
    shortcut: 一次算出 (num_perm, 集合大小) 的整個矩陣；一份文件有幾十萬個 shingle 時要切塊。
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2**64, num_perm, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 2**64, num_perm, dtype=np.uint64)
    return (hashes[None, :] * a[:, None] + b[:, None]).min(axis=1, initial=np.iinfo(np.uint64).max)  # 空集合：全是最大值
# ANCHOR_END: minhash_signature


# ANCHOR: lsh_probability
def lsh_probability(similarity: float, bands: int, rows: int) -> float:
    """相似度為 similarity 的兩份文件，在「bands 段、每段 rows 格」的設定下成為候選對的機率：至少有一段整段相同。"""
    return 1 - (1 - similarity**rows) ** bands
# ANCHOR_END: lsh_probability


# ANCHOR: lsh_candidates
def lsh_candidates(signatures: np.ndarray, bands: int, rows: int) -> set[tuple[int, int]]:
    """簽章矩陣 (文件數, bands * rows)。把每一列切成 bands 段，某一段整段相同的文件放進同一個桶，同桶的兩兩配對就是候選對 (i, j)，i < j。

    shortcut: 同一桶有 m 份文件就產生 m(m-1)/2 對（例如幾百份一模一樣的頁面，或全部是空的文件）；桶很大時要改成只跟桶裡的代表比。
    """
    if signatures.shape[1] != bands * rows:
        raise ValueError(f"簽章有 {signatures.shape[1]} 格，但 bands * rows = {bands * rows}")
    pairs = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = {}
        for doc, part in enumerate(signatures[:, band * rows : (band + 1) * rows]):
            buckets.setdefault(part.tobytes(), []).append(doc)
        for docs in buckets.values():
            pairs.update(combinations(docs, 2))
    return pairs
# ANCHOR_END: lsh_candidates


# ANCHOR: verify_pairs
def verify_pairs(pairs: Iterable[tuple[int, int]], sets: list[np.ndarray], threshold: float) -> list[tuple[int, int, float]]:
    """候選對只是「可能相似」，逐對算精確的 Jaccard，留下 >= threshold 的，回傳 (i, j, Jaccard)。"""
    scored = ((i, j, jaccard(sets[i], sets[j])) for i, j in pairs)
    return [(i, j, similarity) for i, j, similarity in scored if similarity >= threshold]
# ANCHOR_END: verify_pairs


# ANCHOR: drop_near_duplicates
def drop_near_duplicates(pages: list[Page], threshold: float = 0.8, bands: int = 16, rows: int = 8) -> list[Page]:
    """近似重複的頁面只留排在最前面的那一份。「近似」是 shingle 集合的 Jaccard 相似度 >= threshold。"""
    if not pages:
        return []
    sets = [shingles(page.text) for page in pages]
    signatures = np.stack([minhash_signature(s, bands * rows) for s in sets])
    dropped = set()
    for i, j, _ in verify_pairs(sorted(lsh_candidates(signatures, bands, rows)), sets, threshold):
        if i not in dropped:  # i 自己已經被丟掉的話，它就不能當 j 的代表；(i, j) 依 i 由小到大處理，所以 i 的狀態此時已經定案
            dropped.add(j)
    return [page for k, page in enumerate(pages) if k not in dropped]
# ANCHOR_END: drop_near_duplicates
