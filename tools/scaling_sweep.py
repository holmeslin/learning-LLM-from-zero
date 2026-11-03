"""第 10 章第 5 集：在幾個計算預算下各訓練幾個大小不同的模型，最後的 val loss 寫進 runs/scaling/results.jsonl。

    uv run tools/scaling_sweep.py                 # 全部（M4 Max 約一小時）
    uv run tools/scaling_sweep.py --budgets 1e14  # 只跑一個預算

已經寫進 results.jsonl 的組合會跳過，中斷後重跑會接著做。
"""

import argparse
import json
import os

from llmz.model import ModelConfig
from llmz.scaling import non_embedding_params, steps_for_budget
from llmz.train import TrainConfig, train

SIZES = {  # 名稱: (d_model, n_layers, d_ff)，每個 head 32 維
    "d64": (64, 2, 192),
    "d128": (128, 2, 320),
    "d192": (192, 3, 512),
    "d256": (256, 4, 704),
    "d384": (384, 6, 1024),
}
PLAN = {  # 預算越大，最佳的模型越大，所以挑的大小跟著往上移
    1e14: ["d64", "d128", "d192", "d256"],
    3e14: ["d64", "d128", "d192", "d256", "d384"],  # 第一次跑完最低點在 d256 邊上，往大的方向補一個
    1e15: ["d128", "d192", "d256", "d384"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", type=float, nargs="*", default=list(PLAN))
    parser.add_argument("--out_dir", default="runs/scaling")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "results.jsonl")
    done = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            done = {(r["budget"], r["name"]) for r in map(json.loads, f)}

    for budget in args.budgets:
        for name in PLAN[budget]:
            if (budget, name) in done:
                continue
            d_model, n_layers, d_ff = SIZES[name]
            model = ModelConfig(d_model=d_model, n_layers=n_layers, n_heads=d_model // 32, d_ff=d_ff)
            n = non_embedding_params(model)
            steps = steps_for_budget(budget, n, 32, model.context_length)
            cfg = TrainConfig(
                out_dir=os.path.join(args.out_dir, f"{budget:.0e}_{name}"), model=model, max_steps=steps,
                lr_max=3e-3, lr_min=3e-4, warmup_steps=max(20, steps // 20), log_every=100,
                eval_every=steps, eval_batches=100, checkpoint_every=steps, compile=True,
            )
            records = train(cfg)
            result = {"budget": budget, "name": name, "n_params": n, "tokens": steps * 32 * model.context_length,
                      "steps": steps, "val_loss": records[-1]["val_loss"], "time": records[-1]["time"]}
            print(result)
            with open(results_path, "a") as f:
                f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
