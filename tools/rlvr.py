"""第 15 章第 5、6 集：從 SFT 模型（runs/sft/model.pt）開始做 RLVR，存成 runs/rlvr/model.pt。例：

    uv run tools/rlvr.py                                  # 標準設定（M4 Max 上的時間見第 5 集）
    uv run tools/rlvr.py --out_dir runs/rlvr_seed1 --seed 1   # 另一個種子
    uv run tools/rlvr.py --no-normalize_std --no-length_norm --length_divisor 4   # 第 6 集：Dr. GRPO（把標準差與回答長度的除法拿掉）
    uv run tools/rlvr.py --seq_ratio                          # 第 6 集：GSPO（整條回答共用一個比值）

縮小版（smoke）見第 5 集與 tests/test_rl.py。
"""

import argparse
from dataclasses import fields

from llmz.model import ModelConfig
from llmz.rl import RLVRConfig, train_rlvr


def main() -> None:
    parser = argparse.ArgumentParser()
    for f in fields(RLVRConfig):
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
    train_rlvr(RLVRConfig(model=model_cfg, **args))


if __name__ == "__main__":
    main()
