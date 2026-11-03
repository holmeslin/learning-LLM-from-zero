"""第 14 章第 5 集：從 SFT 模型（runs/sft/model.pt）開始 DPO，存成 runs/dpo/model.pt。例：

    uv run tools/dpo.py                                   # 標準設定
    uv run tools/dpo.py --out_dir runs/dpo_beta1 --beta 1.0   # 換 beta

縮小版（smoke）見第 5 集與 tests/test_posttrain.py。
"""

import argparse
from dataclasses import fields

from llmz.model import ModelConfig
from llmz.posttrain import DPOConfig, train_dpo


def main() -> None:
    parser = argparse.ArgumentParser()
    for f in fields(DPOConfig):
        if f.name == "model":
            continue
        if type(f.default) is bool:
            parser.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    for f in fields(ModelConfig):
        if type(f.default) is bool:
            parser.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    args = vars(parser.parse_args())
    model_cfg = ModelConfig(**{f.name: args.pop(f.name) for f in fields(ModelConfig)})
    train_dpo(DPOConfig(model=model_cfg, **args))


if __name__ == "__main__":
    main()
