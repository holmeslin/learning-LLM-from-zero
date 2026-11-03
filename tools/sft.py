"""第 14 章第 3 集：從標準模型開始 SFT。例：

    uv run tools/sft.py                                   # 標準設定，存到 runs/sft（M4 Max 上幾分鐘）
    uv run tools/sft.py --out_dir runs/sft_nomask --no-mask   # 整串都算 loss 的對照
    uv run tools/sft.py --base_path "" --d_model 64 --n_layers 2 --n_heads 4 --d_ff 128 --vocab_size 10000 --max_steps 60 --device cpu  # smoke
"""

import argparse
from dataclasses import fields

from llmz.model import ModelConfig
from llmz.posttrain import SFTConfig, train_sft


def main() -> None:
    parser = argparse.ArgumentParser()
    for f in fields(SFTConfig):
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
    train_sft(SFTConfig(model=model_cfg, **args))


if __name__ == "__main__":
    main()
