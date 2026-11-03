import json

import numpy as np
import torch

from llmz.data import get_batch
from llmz.model import ModelConfig, TransformerLM
from llmz.optim import AdamW
from llmz.train import TrainConfig, load_checkpoint, save_checkpoint, train

SMALL = ModelConfig(vocab_size=64, context_length=16, d_model=32, n_layers=2, n_heads=4, d_ff=64)


def make_data(tmp_path):
    # 可預測的序列：0,1,2,...,63,0,1,... 模型幾步就能學會「下一個是 +1」
    tokens = (np.arange(20000) % 64).astype(np.uint16)
    for name in ["train.bin", "valid.bin"]:
        tokens.tofile(tmp_path / name)


def test_get_batch_shapes_and_shift():
    data = np.arange(1000, dtype=np.uint16)
    torch.manual_seed(0)
    x, y = get_batch(data, batch_size=4, context_length=8, device=torch.device("cpu"))
    assert x.shape == (4, 8) and y.shape == (4, 8) and x.dtype == torch.int64
    assert torch.equal(y, x + 1)


def test_smoke_training_lowers_loss_and_writes_log(tmp_path):
    make_data(tmp_path)
    cfg = TrainConfig(train_path=str(tmp_path / "train.bin"), valid_path=str(tmp_path / "valid.bin"),
                      out_dir=str(tmp_path / "run"), model=SMALL, batch_size=8, max_steps=60, warmup_steps=5,
                      lr_max=3e-3, log_every=10, eval_every=30, eval_batches=2, checkpoint_every=30, device="cpu")
    records = train(cfg)
    assert records[0]["loss"] > records[-1]["loss"] * 1.5
    assert records[-1]["val_loss"] < 3.0
    lines = (tmp_path / "run" / "log.jsonl").read_text().strip().split("\n")
    assert json.loads(lines[-1])["step"] == 60
    assert (tmp_path / "run" / "model.pt").exists() and (tmp_path / "run" / "latest.pt").exists()


def test_checkpoint_roundtrip_and_resume(tmp_path):
    torch.manual_seed(0)
    model = TransformerLM(SMALL)
    opt = AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 64, (2, 8))
    model(x).sum().backward()
    opt.step()
    save_checkpoint(model, opt, 7, tmp_path / "ckpt.pt")
    model2 = TransformerLM(SMALL)
    opt2 = AdamW(model2.parameters(), lr=1e-3)
    assert load_checkpoint(tmp_path / "ckpt.pt", model2, opt2) == 7
    assert torch.allclose(model(x), model2(x))
    assert opt2.state[model2.embed.weight]["t"] == 1
