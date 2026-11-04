"""準備第 13 章的網頁文字（第 13 章第 1、2 集）：

1. 下載 Common Crawl 一個 WARC 檔的前 32 MiB（HTTP Range，不用抓整個檔案）到 data/cc/sample.warc.gz
2. 解析 WARC，把每個成功的 HTML 回應轉成純文字，存成 data/cc/pages.jsonl（一頁一行，沒有文字的頁面丟掉）

已經存在的檔案會跳過，所以中斷後重跑即可。網頁內容不進 repo（data/ 在 .gitignore 裡）。
"""

import os
import time

from llmz.data import download
from llmz.pipeline import iter_pages, write_pages

CRAWL = "CC-MAIN-2026-43"  # 2026 年 10 月的 crawl
WARC_FILE = "crawl-data/CC-MAIN-2026-43/segments/1791170053823.21/warc/CC-MAIN-20261005105623-20261005135623-00000.warc.gz"
URL = "https://data.commoncrawl.org/" + WARC_FILE
SAMPLE_BYTES = 32 * 1024 * 1024
WARC_PATH, PAGES_PATH = "data/cc/sample.warc.gz", "data/cc/pages.jsonl"


def main() -> None:
    os.makedirs("data/cc", exist_ok=True)
    if not os.path.exists(WARC_PATH):
        t0 = time.time()
        n = download(URL, WARC_PATH, SAMPLE_BYTES)
        print(f"{WARC_PATH}: {n:,} bytes，{time.time() - t0:.0f} 秒")
    if not os.path.exists(PAGES_PATH):
        t0 = time.time()
        n_html = 0

        def non_empty_pages():
            nonlocal n_html
            for page in iter_pages(WARC_PATH):
                n_html += 1
                if page.text:
                    yield page

        n_pages = write_pages(non_empty_pages(), PAGES_PATH)
        size = os.path.getsize(PAGES_PATH)
        print(f"{PAGES_PATH}: {n_html} 個 HTML 頁面，留下 {n_pages} 頁有文字的，{size:,} bytes，{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
