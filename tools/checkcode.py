"""checkcode：抽出書中所有 ```python 程式碼區塊並執行驗證。

用法：uv run python tools/checkcode.py zh-TW/src

規則（詳見 README.md）：
- 每個 ```python 區塊會存成 main.py，用目前的直譯器在空目錄裡執行，必須以結束碼 0 結束。
- 區塊後面緊接著「執行結果：」加 ```text 區塊時，stdout 必須與它完全一致；
  寫「執行結果（某一次）：」就只執行、不比對。
- info string 標註（逗號分隔，值裡不能有空白）：
  python,ignore   不驗證（片段）
  python,norun    只檢查語法，不執行
  python,error    預期會以非 0 結束碼結束（故意示範錯誤）
  python,stdin=3\n5   執行時餵入 stdin，\n 代表換行
- 另外檢查每個 {{#include 路徑:anchor}} 指到的檔案與 anchor 都存在。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

TIMEOUT = 300  # 秒；訓練類範例要用 smoke 設定，幾分鐘內跑完
FENCE_RE = re.compile(r"^(`{3,})(.*)$")
INCLUDE_RE = re.compile(r"\{\{#include\s+([^}:\s]+)(?::([^}\s]+))?\s*\}\}")
RESULT_LINE = "執行結果："


@dataclass
class Block:
    file: Path
    line: int
    attrs: dict[str, str] = field(default_factory=dict)
    code: str = ""
    expected: str | None = None  # 「執行結果：」後面的 text 區塊內容


def parse_attrs(info: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for a in info.split(",")[1:]:
        k, _, v = a.partition("=")
        attrs[k] = v.replace("\\n", "\n")
    return attrs


def check_include(md: Path, n: int, rel: str, anchor: str | None) -> list[str]:
    target = (md.parent / rel).resolve()
    if not target.is_file():
        return [f"{md}:{n}: include 找不到檔案 {rel}"]
    if anchor and not anchor[0].isdigit():
        text = target.read_text(encoding="utf-8")
        if f"ANCHOR: {anchor}" not in text or f"ANCHOR_END: {anchor}" not in text:
            return [f"{md}:{n}: {rel} 裡沒有 anchor {anchor}"]
    return []


def collect(root: Path) -> tuple[list[Block], list[str]]:
    """回傳所有 python 區塊（含預期輸出），以及 include 檢查的錯誤。"""
    blocks: list[Block] = []
    errors: list[str] = []
    for md in sorted(root.rglob("*.md")):
        blocks_here, errors_here = collect_file(md)
        blocks += blocks_here
        errors += errors_here
    return blocks, errors


def collect_file(md: Path) -> tuple[list[Block], list[str]]:
    blocks: list[Block] = []
    errors: list[str] = []
    lines = md.read_text(encoding="utf-8").splitlines()
    state = "outside"  # outside / python / expected / other
    marker = ""  # 目前 fence 的反引號，巢狀的四反引號才對得上
    indent = 0
    buf: list[str] = []
    cur: Block | None = None
    pending: Block | None = None  # 剛結束的 python 區塊，等著看有沒有「執行結果：」
    expect_text = False  # 已看到「執行結果：」，下一個 ```text 就是預期輸出
    n = 0
    for n, raw in enumerate(lines, 1):
        s = raw.strip()
        for im in INCLUDE_RE.finditer(raw):  # include 一律寫在 fence 裡，所以每一行都要看
            errors += check_include(md, n, im.group(1), im.group(2))
        if state == "outside":
            m = FENCE_RE.match(s)
            if m:
                marker, info = m.group(1), m.group(2).strip()
                indent = len(raw) - len(raw.lstrip())
                buf = []
                if info == "python" or info.startswith("python,"):
                    cur = Block(md, n, parse_attrs(info))
                    state = "python"
                    pending, expect_text = None, False
                elif info == "text" and expect_text:
                    state = "expected"
                else:
                    state = "other"
                    pending, expect_text = None, False
                continue
            if pending is not None:
                if s == "":
                    continue
                if s == RESULT_LINE and not expect_text:
                    expect_text = True
                    continue
                pending, expect_text = None, False
            continue
        # 在 fence 裡
        if s == marker:
            content = "\n".join(line[indent:] if line[:indent].strip() == "" else line for line in buf)
            if state == "python" and cur is not None:
                cur.code = content + "\n"
                blocks.append(cur)
                pending, cur = cur, None
            elif state == "expected" and pending is not None:
                pending.expected = content
                pending = None
            expect_text = False
            state = "outside"
            continue
        buf.append(raw)
    if state != "outside":
        errors.append(f"{md}:{n}: 程式碼區塊沒有結束")
    return blocks, errors


def check(b: Block) -> str | None:
    """驗證一個區塊，回傳錯誤訊息；None 表示通過。"""
    if "norun" in b.attrs:
        try:
            compile(b.code, str(b.file), "exec")
        except SyntaxError as e:
            return f"語法錯誤：{e}"
        return None
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
        "PYTHONIOENCODING": "utf-8",
    }
    with tempfile.TemporaryDirectory() as d:
        Path(d, "main.py").write_text(b.code, encoding="utf-8")
        try:
            p = subprocess.run(
                [sys.executable, "-I", "-X", "utf8", "main.py"],
                cwd=d,
                input=b.attrs.get("stdin", ""),
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return f"逾時（{TIMEOUT} 秒）"
    if "error" in b.attrs:
        return None if p.returncode != 0 else "預期會出錯，卻正常結束"
    if p.returncode != 0:
        return f"結束碼 {p.returncode}\n{p.stdout}{p.stderr}"
    if b.expected is not None and p.stdout.rstrip("\n") != b.expected.rstrip("\n"):
        return f"輸出與書上不一致\n---- 書上 ----\n{b.expected}\n---- 實際 ----\n{p.stdout.rstrip()}\n----"
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: checkcode <dir>", file=sys.stderr)
        return 2
    blocks, errors = collect(Path(argv[1]))
    todo = [b for b in blocks if "ignore" not in b.attrs]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        results = list(ex.map(check, todo))
    failures = [f"{b.file}:{b.line}: {r}" for b, r in zip(todo, results) if r] + errors
    for f in failures:
        print("FAIL", f)
    print(f"{len(todo)} blocks checked, {len(failures)} failed, {len(blocks) - len(todo)} ignored")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
