"""從命令列訓練（第 5 章第 10 集起）。例：

    uv run tools/train.py --out_dir runs/base
    uv run tools/train.py --out_dir runs/no_rmsnorm --norm none --max_steps 1000
"""

import argparse
from dataclasses import fields

from llmz.model import ModelConfig
from llmz.train import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser()
    for f in fields(TrainConfig):
        if f.name == "model":
            continue
        kind = type(f.default) if f.default is not None else str
        if kind is bool:
            parser.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        elif kind is tuple:
            parser.add_argument(f"--{f.name}", type=float, nargs=2, default=f.default)
        else:
            parser.add_argument(f"--{f.name}", type=kind, default=f.default)
    for f in fields(ModelConfig):
        if type(f.default) is bool:
            parser.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    args = vars(parser.parse_args())
    model_cfg = ModelConfig(**{f.name: args.pop(f.name) for f in fields(ModelConfig)})
    args["betas"] = tuple(args["betas"])
    train(TrainConfig(model=model_cfg, **args))


if __name__ == "__main__":
    main()
