"""資料檔處理（第 3 章第 14 集起）：下載、轉成 token 檔、讀回來、隨機取 batch。"""

import array
import urllib.request

import numpy as np
import torch
from torch import Tensor

from llmz.tokenizer import Tokenizer


# ANCHOR: download
def download(url: str, path: str, max_bytes: int | None = None) -> int:
    """下載到 path；給 max_bytes 時只要檔案的前面那一段（HTTP Range），回傳實際位元組數。"""
    req = urllib.request.Request(url)
    if max_bytes is not None:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    with urllib.request.urlopen(req) as resp, open(path, "wb") as f:
        total = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            total += len(chunk)
    return total
# ANCHOR_END: download


# ANCHOR: cut_at_last_document
def cut_at_last_document(path: str, boundary: str = "<|endoftext|>") -> int:
    """把檔案截短到最後一個完整文件（最後一個 boundary 之後的殘缺部分丟掉），回傳新長度。"""
    marker = boundary.encode("utf-8")
    with open(path, "rb") as f:
        data = f.read()
    end = data.rfind(marker)
    if end < 0:
        raise ValueError(f"找不到 {boundary!r}")
    end += len(marker)
    with open(path, "wb") as f:
        f.write(data[:end])
        f.write(b"\n")
    return end + 1
# ANCHOR_END: cut_at_last_document


# ANCHOR: tokenize_to_file
def tokenize_to_file(tokenizer: Tokenizer, text_path: str, out_path: str) -> int:
    """一行一行 encode，把 id 以 uint16 寫進 out_path，回傳 token 總數。"""
    ids = array.array("H")  # 無號 16 位元整數，一個 token 佔 2 位元組
    with open(text_path, encoding="utf-8") as f:
        ids.extend(tokenizer.encode_iterable(f))
    with open(out_path, "wb") as f:
        ids.tofile(f)
    return len(ids)
# ANCHOR_END: tokenize_to_file


# ANCHOR: load_tokens
def load_tokens(path: str) -> np.memmap:
    """用 memmap 打開 token 檔：像陣列一樣索引，但不會整個讀進記憶體。"""
    return np.memmap(path, dtype=np.uint16, mode="r")
# ANCHOR_END: load_tokens


# ANCHOR: get_batch
def get_batch(data: np.ndarray, batch_size: int, context_length: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """從 token 檔隨機切 batch_size 段，每段 context_length 個 token；targets 是 inputs 往後移一格。"""
    starts = torch.randint(0, len(data) - context_length, (batch_size,))
    chunks = np.stack([data[i : i + context_length + 1] for i in starts.tolist()])  # (B, T+1)
    chunks = torch.from_numpy(chunks.astype(np.int64))
    return chunks[:, :-1].to(device), chunks[:, 1:].to(device)
# ANCHOR_END: get_batch
