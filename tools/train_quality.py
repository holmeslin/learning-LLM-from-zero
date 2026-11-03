"""訓練第 13 章第 7 集的品質分類器：正例是標準語料的故事，負例是 data/cc/pages.jsonl 的全部網頁。

用法：
    uv run python tools/train_quality.py            # 完整：2 萬篇故事、2^16 個 bucket、300 步
    uv run python tools/train_quality.py --smoke    # smoke：2,000 篇故事、2^12 個 bucket、100 步，CPU 幾秒
    uv run python tools/train_quality.py --out runs/quality/quality.pt

存的是 QualityClassifier 的 state_dict；用的時候：
    sd = torch.load(path, weights_only=True); model = QualityClassifier(sd["weight"].numel()); model.load_state_dict(sd)
先執行 tools/prepare_webtext.py（網頁）與 tools/prepare_data.py（標準語料）。
"""

import argparse
import os
import random
import time

import torch

from llmz.pipeline import QualityClassifier, read_pages, train_quality


def read_stories(path: str, n: int, seed: int) -> list[str]:
    docs = [d.strip() for d in open(path, encoding="utf-8").read().split("<|endoftext|>") if d.strip()]
    return random.Random(seed).sample(docs, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="縮小設定，CPU 幾秒")
    ap.add_argument("--out", default="runs/quality/quality.pt")
    args = ap.parse_args()
    n_stories, n_buckets, steps = (2000, 1 << 12, 100) if args.smoke else (20000, 1 << 16, 300)

    start = time.perf_counter()
    stories = read_stories("data/train.txt", n_stories, 0)
    pages = [p.text for p in read_pages("data/cc/pages.jsonl")]
    model, losses = train_quality(stories + pages, [1] * len(stories) + [0] * len(pages), n_buckets=n_buckets, steps=steps)
    print(f"{len(stories):,} 篇故事 + {len(pages)} 頁網頁，{n_buckets} 個 bucket，{steps} 步：{time.perf_counter() - start:.1f} 秒")
    print(f"loss {losses[0]:.3f} → {sum(losses[-20:]) / 20:.4f}（最後 20 步的平均）")
    print(f"訓練資料上判成像標準語料的比例：故事 {(model.score(stories) > 0).float().mean():.2%}，網頁 {(model.score(pages) > 0).float().mean():.2%}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    reloaded = QualityClassifier(n_buckets)
    reloaded.load_state_dict(torch.load(args.out, weights_only=True))
    assert torch.equal(reloaded.score(pages[:20]), model.score(pages[:20]))
    print("存到", args.out)


if __name__ == "__main__":
    main()
