"""把一個或多個 runs/<名稱>/log.jsonl 畫成 loss 曲線（第 5 章第 11 集）。例：

    uv run tools/plot_loss.py runs/base
    uv run tools/plot_loss.py runs/base_1k runs/no_rmsnorm runs/post_norm -o ablation.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt


def read_log(run_dir: str) -> list[dict]:
    with open(os.path.join(run_dir, "log.jsonl")) as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("-o", "--out", default="loss.png")
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--x", choices=["step", "tokens"], default="step", help="橫軸用步數或看過的 token 數")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run_dir in args.runs:
        records = read_log(run_dir)
        name = os.path.basename(run_dir.rstrip("/"))
        scale = 1
        if args.x == "tokens":  # 一步看 batch_size × context_length 個 token，從 config.json 讀
            with open(os.path.join(run_dir, "config.json")) as f:
                cfg = json.load(f)
            scale = cfg["batch_size"] * cfg["model"]["context_length"]
        ax.plot([r["step"] * scale for r in records], [r["loss"] for r in records], alpha=0.4, label=f"{name} train")
        val = [r for r in records if "val_loss" in r]
        ax.plot([r["step"] * scale for r in val], [r["val_loss"] for r in val], marker="o", label=f"{name} val")
    ax.set_xlabel(args.x)
    ax.set_ylabel("loss")
    if args.ymax:
        ax.set_ylim(top=args.ymax)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
