"""訓練迴圈（第 5 章第 7～8 集）：設定、checkpoint、評估、主迴圈與 JSONL log。"""

import json
import os
import time
from dataclasses import asdict, dataclass, field

import torch

from llmz.data import get_batch, load_tokens
from llmz.device import get_device
from llmz.functional import cross_entropy
from llmz.model import ModelConfig, TransformerLM
from llmz.optim import AdamW, clip_grad_norm_, lr_cosine_schedule


# ANCHOR: config
@dataclass
class TrainConfig:
    train_path: str = "data/train.bin"
    valid_path: str = "data/valid.bin"
    out_dir: str = "runs/base"
    model: ModelConfig = field(default_factory=ModelConfig)
    batch_size: int = 32
    max_steps: int = 5000  # 32 × 256 × 5000 ≈ 4100 萬個 token，標準語料看一輪
    lr_max: float = 1e-3
    lr_min: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    log_every: int = 25
    eval_every: int = 250
    eval_batches: int = 20
    checkpoint_every: int = 1000
    seed: int = 0
    device: str = ""  # 空字串代表用 get_device()
    compile: bool = False
# ANCHOR_END: config


# ANCHOR: checkpoint
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, path: str) -> None:
    """模型參數、優化器狀態、目前步數一起存，之後才能從中斷的地方接著練。"""
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
    """載回參數（與優化器狀態），回傳存檔時的步數。"""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]
# ANCHOR_END: checkpoint


# ANCHOR: evaluate
@torch.no_grad()
def evaluate(model: torch.nn.Module, data, cfg: TrainConfig, device: torch.device) -> float:
    """在驗證集上隨機抽 eval_batches 批，回傳平均 loss。"""
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = get_batch(data, cfg.batch_size, cfg.model.context_length, device)
        logits = model(x)
        losses.append(cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)
# ANCHOR_END: evaluate


# ANCHOR: train
def train(cfg: TrainConfig) -> list[dict]:
    """主迴圈。每一步：取 batch → forward → loss → backward → 裁剪 → step → zero_grad。回傳 log 紀錄。"""
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device) if cfg.device else get_device()
    os.makedirs(cfg.out_dir, exist_ok=True)
    train_data = load_tokens(cfg.train_path)
    valid_data = load_tokens(cfg.valid_path)

    model = TransformerLM(cfg.model).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr_max, betas=cfg.betas, weight_decay=cfg.weight_decay)
    step = 0
    resume = os.path.join(cfg.out_dir, "latest.pt")
    if os.path.exists(resume):
        step = load_checkpoint(resume, model, optimizer)
    forward = torch.compile(model) if cfg.compile else model

    records: list[dict] = []
    log_file = open(os.path.join(cfg.out_dir, "log.jsonl"), "a")
    t0 = time.time()
    while step < cfg.max_steps:
        step += 1
        lr = lr_cosine_schedule(step, cfg.lr_max, cfg.lr_min, cfg.warmup_steps, cfg.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_data, cfg.batch_size, cfg.model.context_length, device)
        logits = forward(x)
        loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        if step % cfg.log_every == 0 or step == cfg.max_steps:
            rec = {"step": step, "loss": round(loss.item(), 4), "lr": lr,
                   "grad_norm": round(grad_norm.item(), 3), "time": round(time.time() - t0, 1)}
            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                rec["val_loss"] = round(evaluate(model, valid_data, cfg, device), 4)
            records.append(rec)
            log_file.write(json.dumps(rec) + "\n")
            log_file.flush()
            print(rec)
        if step % cfg.checkpoint_every == 0 or step == cfg.max_steps:
            save_checkpoint(model, optimizer, step, resume)
    log_file.close()
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model.pt"))
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    return records
# ANCHOR_END: train
