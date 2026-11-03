"""把 tools/scaling_sweep.py 的結果畫成 IsoFLOPs 曲線與最佳參數量的冪律（第 10 章第 5 集）。例：

    uv run tools/plot_scaling.py runs/scaling/results.jsonl -o scaling.png
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from llmz.scaling import fit_isoflop_minimum, fit_power_law

COLORS = ["#6da7ec", "#2a78d6", "#104281"]  # 同一個色相由淺到深：預算由小到大


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("-o", "--out", default="scaling.png")
    args = parser.parse_args()
    with open(args.results) as f:
        runs = [json.loads(line) for line in f]
    budgets = sorted({r["budget"] for r in runs})

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    best = []
    for budget, color in zip(budgets, COLORS):
        sub = sorted((r for r in runs if r["budget"] == budget), key=lambda r: r["n_params"])
        n = np.array([r["n_params"] for r in sub])
        loss = np.array([r["val_loss"] for r in sub])
        n_opt = fit_isoflop_minimum(n, loss)
        best.append(n_opt)
        c2, c1, c0 = np.polyfit(np.log10(n), loss, 2)
        x = np.linspace(np.log10(n[0]), np.log10(n[-1]), 100)
        left.plot(10**x, c2 * x**2 + c1 * x + c0, color=color, linewidth=2)
        left.plot(n, loss, "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=2,
                  label=f"C = {budget:.0e}")
        left.axvline(n_opt, color=color, linewidth=1, linestyle=":")
    left.set_xscale("log")
    left.set_xlabel("non-embedding parameters N")
    left.set_ylabel("validation loss")
    left.set_title("IsoFLOPs")
    left.legend(frameon=False, loc="lower left")

    k, a = fit_power_law(np.array(budgets), np.array(best))
    c = np.logspace(np.log10(budgets[0]) - 0.2, np.log10(budgets[-1]) + 0.2, 50)
    right.plot(c, k * c**a, color="#888888", linewidth=2, label=f"N_opt = {k:.3g} C^{a:.3f}")
    for budget, n_opt, color in zip(budgets, best, COLORS):
        right.plot(budget, n_opt, "o", color=color, markersize=9, markeredgecolor="white", markeredgewidth=2)
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("compute C (FLOPs)")
    right.set_ylabel("optimal N")
    right.set_title("Optimal model size")
    right.legend(frameon=False)

    for ax in (left, right):
        ax.grid(alpha=0.25, which="both")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
