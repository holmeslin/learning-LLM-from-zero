"""強化學習入門（第 15 章）：用多臂吃角子老虎機（bandit）看 policy gradient、REINFORCE 與 group-relative advantage。

這裡的 policy 是一排 logits 經過 softmax；第 4 集起，同一個 loss 會換成真正的語言模型。
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import NamedTuple

import torch
from torch import Tensor, nn

from llmz.device import get_device
from llmz.functional import log_softmax, softmax
from llmz.inference import KVCache, forward_cached
from llmz.model import ModelConfig, TransformerLM
from llmz.optim import AdamW, clip_grad_norm_
from llmz.posttrain import END, accuracy, encode_chat, make_tasks, pad_batch, reward, task_messages
from llmz.tokenizer import Tokenizer


# ANCHOR: sample_actions
def sample_actions(logits: Tensor, n: int, generator: torch.Generator | None = None) -> Tensor:
    """照 softmax(logits) 的機率抽 n 個動作。logits 形狀 (K,) 回傳 (n,)；形狀 (P, K) 則每一列各抽 n 個，回傳 (P, n)。"""
    return torch.multinomial(softmax(logits), n, replacement=True, generator=generator)
# ANCHOR_END: sample_actions


# ANCHOR: bandit_rewards
def bandit_rewards(means: Tensor, actions: Tensor, noise_std: float = 0.0, generator: torch.Generator | None = None) -> Tensor:
    """每個動作 k 的平均獎勵是 means[k]；拉一次拿到平均值再加上標準差 noise_std 的高斯雜訊。"""
    noise = torch.randn(actions.shape, generator=generator) * noise_std
    return means[actions] + noise
# ANCHOR_END: bandit_rewards


# ANCHOR: expected_reward
def expected_reward(logits: Tensor, means: Tensor) -> Tensor:
    """policy 的目標 J：每個動作的機率乘上它的平均獎勵，加總。"""
    return (softmax(logits) * means).sum(dim=-1)
# ANCHOR_END: expected_reward


# ANCHOR: reinforce_loss
def reinforce_loss(logits: Tensor, actions: Tensor, advantages: Tensor) -> Tensor:
    """logits (N, K)、actions (N,)、advantages (N,) -> 純量。advantage 為正就把選到的動作的機率推高，為負就推低。"""
    logp = log_softmax(logits, dim=-1)[torch.arange(len(actions)), actions]  # 和第 1 章 cross_entropy 一樣，只取選到的那一格
    return -(advantages * logp).mean()
# ANCHOR_END: reinforce_loss


# ANCHOR: estimate_gradient
def estimate_gradient(logits: Tensor, means: Tensor, n: int, baseline: bool = False, noise_std: float = 0.0,
                      generator: torch.Generator | None = None) -> Tensor:
    """REINFORCE 對 J 的梯度估計 (K,)：抽 n 個動作、拿獎勵，用 reinforce_loss 算梯度後取負號（因為 J 要往上爬）。

    baseline=True 時，advantage 是「獎勵減去這一批獎勵的平均」；否則 advantage 就是獎勵本身。
    shortcut: 批次平均把自己也算進去，估計的平均縮小 (n-1)/n 倍；要完全無偏時改用 leave-one-out（第 2 集練習）。
    """
    logits = logits.detach().requires_grad_()
    actions = sample_actions(logits, n, generator)
    rewards = bandit_rewards(means, actions, noise_std, generator)
    advantages = rewards - rewards.mean() if baseline else rewards
    reinforce_loss(logits.expand(n, -1), actions, advantages).backward()
    return -logits.grad
# ANCHOR_END: estimate_gradient


# ANCHOR: train_bandit
def train_bandit(means: Tensor, steps: int, n: int, lr: float, baseline: bool = True, noise_std: float = 0.0, seed: int = 0) -> Tensor:
    """從全 0 的 logits 開始，每一步用 n 個樣本估計梯度、往上爬一步（gradient ascent），回傳最後的 logits。"""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.zeros(len(means))
    for _ in range(steps):
        logits = logits + lr * estimate_gradient(logits, means, n, baseline, noise_std, generator)
    return logits
# ANCHOR_END: train_bandit


# ANCHOR: group_advantages
def group_advantages(rewards: Tensor, normalize_std: bool = True, eps: float = 1e-6) -> Tensor:
    """rewards (P, G)：P 個 prompt、每個 prompt 的一組 G 個回答的獎勵。回傳同樣形狀的 advantage：組內獎勵減掉組內平均，
    normalize_std=True 時再除以組內標準差（樣本標準差，除以 G-1；全組獎勵相同時標準差是 0，分子也是 0，所以 advantage 是 0）。"""
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    return centered / (rewards.std(dim=1, keepdim=True) + eps) if normalize_std else centered
# ANCHOR_END: group_advantages


# ANCHOR: verifiable_rewards
def verifiable_rewards(correct: Tensor, actions: Tensor) -> Tensor:
    """correct (P,)：每個 prompt 的正確答案；actions (P, G)：每個 prompt 的 G 個回答。答對 1，答錯 0。"""
    return (actions == correct[:, None]).float()
# ANCHOR_END: verifiable_rewards


# ANCHOR: train_groups
def train_groups(logits: Tensor, correct: Tensor, group_size: int, steps: int, lr: float,
                 normalize_std: bool = True, seed: int = 0) -> Tensor:
    """logits (P, K)：P 個 prompt 各自的 policy。每一步：每個 prompt 抽一組 group_size 個回答、用 0/1 獎勵算 group advantage、往上爬一步。"""
    generator = torch.Generator().manual_seed(seed)
    P, K = logits.shape
    logits = logits.clone()
    for _ in range(steps):
        logits.requires_grad_()
        actions = sample_actions(logits.detach(), group_size, generator)  # (P, G)
        advantages = group_advantages(verifiable_rewards(correct, actions), normalize_std)
        flat_logits = logits[:, None, :].expand(P, group_size, K).reshape(-1, K)  # 同一個 prompt 的回答共用同一列 logits
        reinforce_loss(flat_logits, actions.reshape(-1), advantages.reshape(-1)).backward()
        logits = (logits - lr * logits.grad).detach()  # loss 是 -J，往下降就是 J 往上爬
    return logits
# ANCHOR_END: train_groups


# ANCHOR: token_logprobs
def token_logprobs(model: nn.Module, ids: Tensor) -> Tensor:
    """ids (N, T) -> (N, T-1)：每個位置「實際接下來的那個 token」的 log 機率（位置 t 預測 token t+1）。
    和第 12 章的 sequence_logprob 是同一件事，只是不加總：GRPO 的 ratio 要一個 token 一個 token 算。沒有 no_grad，可以直接 backward。"""
    logp = log_softmax(model(ids[:, :-1]), dim=-1)  # (N, T-1, V)：位置 t 預測 token t+1
    batch, position = torch.arange(ids.shape[0], device=ids.device)[:, None], torch.arange(ids.shape[1] - 1, device=ids.device)
    return logp[batch, position, ids[:, 1:]]  # broadcasting 之後每個 token 只取自己那一格
# ANCHOR_END: token_logprobs


# ANCHOR: grpo_loss
def grpo_loss(logp: Tensor, old_logp: Tensor, advantages: Tensor, mask: Tensor, clip_eps: float = 0.2, length_norm: bool = True) -> Tensor:
    """GRPO 的 clipped policy-gradient loss（省略 KL 項）。

    logp、old_logp、mask：(N, T-1)，和 token_logprobs 對齊（mask 用 rollout 的 mask[:, 1:]）；advantages：(N,)，每條回答一個數字。
    old_logp 是 rollout 當時的 policy 算的，不帶梯度。ratio = exp(logp - old_logp)；每個 token 取 min(ratio * A, clip(ratio) * A)。
    length_norm=True：每條回答先對自己的 token 平均（原始 GRPO）；False：token 加總，長的回答權重大。最後對 N 條回答平均。
    """
    ratio = torch.where(mask, logp - old_logp, torch.zeros_like(logp)).exp()  # 被遮掉的位置先歸零再 exp：exp(100) = inf，inf * 0 會變成 nan
    advantages = advantages[:, None]
    objective = torch.minimum(ratio * advantages, ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages)  # (N, T-1)
    per_answer = (objective * mask).sum(dim=1)
    if length_norm:
        per_answer = per_answer / mask.sum(dim=1).clamp(min=1)  # shortcut: 沒有任何 token 的回答分母用 1（分子也是 0）；rollout 至少會有一個 token
    return -per_answer.mean()
# ANCHOR_END: grpo_loss


# ANCHOR: rollout
@torch.no_grad()
def sample_batch(model: TransformerLM, ids: Tensor, max_new_tokens: int, temperature: float = 1.0) -> Tensor:
    """ids (B, T) -> (B, T + max_new_tokens)：和第 11 章的 generate_cached 同一套 KV cache、prefill 加 decode，差在抽樣。
    pick_next 一列一列抽（為了 top_p），MPS 上每一列都是好幾次小運算，rollout 的時間幾乎都花在這裡（第 5 集量過）；
    這裡整個 batch 一次抽，不支援 top_p。"""
    cache = KVCache(model.cfg, ids.shape[0], ids.device)
    out = [ids]
    logits = forward_cached(model, ids, cache)[:, -1]  # prefill
    for _ in range(max_new_tokens):
        next_ids = torch.multinomial(softmax(logits / temperature, dim=-1), 1)  # (B, 1)
        out.append(next_ids)
        if len(out) <= max_new_tokens:
            logits = forward_cached(model, next_ids, cache)[:, -1]  # decode
    return torch.cat(out, dim=1)


class Rollout(NamedTuple):
    ids: Tensor  # (N, T)：提示加回答，N = 題數 * group_size，同一題的 group_size 個回答排在一起；短的往右補 0
    mask: Tensor  # (N, T)：回答的 token 是 True（從回答第一個 token 到第一個 <|endoftext|>，含它）；提示、結尾之後、補的都是 False
    rewards: Tensor  # (題數, group_size)：每個回答的獎勵


def rollout(model: nn.Module, tokenizer, tasks: list, group_size: int, max_new_tokens: int = 8, temperature: float = 1.0) -> Rollout:
    """每一題抽 group_size 個回答，並且把 token id 留下來（posttrain.generate_replies 只回傳字串，不夠 GRPO 用）。
    長度相同的提示放在同一個 batch（KV cache 的要求，和 generate_cached 一樣），最後照原本的順序排回去。
    結尾 <|endoftext|> 也算回答的一部分：模型要學會「什麼時候停」。"""
    device = next(model.parameters()).device
    end_id = tokenizer.special_to_id[END]
    prompts = [encode_chat(tokenizer, task_messages(task), generation_prompt=True)[0] for task in tasks for _ in range(group_size)]
    by_length: dict[int, list[int]] = {}
    for i, prompt in enumerate(prompts):
        by_length.setdefault(len(prompt), []).append(i)
    examples, rewards = [None] * len(prompts), [0.0] * len(prompts)
    for length, rows in by_length.items():
        out = sample_batch(model, torch.tensor([prompts[i] for i in rows], device=device), max_new_tokens, temperature)
        for i, new in zip(rows, out[:, length:].tolist()):
            n = new.index(end_id) + 1 if end_id in new else len(new)  # 回答的 token 數（含結尾）
            examples[i] = (prompts[i] + new, [False] * length + [True] * n + [False] * (len(new) - n))
            rewards[i] = reward(tasks[i // group_size], tokenizer.decode(new[: n - 1] if end_id in new else new))
    ids, mask = pad_batch(examples)
    return Rollout(ids, mask, torch.tensor(rewards).reshape(len(tasks), group_size))
# ANCHOR_END: rollout


# ANCHOR: train_rlvr
@dataclass
class RLVRConfig:
    sft_path: str = "runs/sft/model.pt"  # 從第 14 章的 SFT 模型開始；空字串 = 不載入，隨機初始化（smoke 用）
    tokenizer_path: str = "data/tokenizer.json"
    out_dir: str = "runs/rlvr"
    model: ModelConfig = field(default_factory=ModelConfig)
    kinds: str = "color,who"  # 逗號分隔的題型：SFT 之後還有進步空間的兩種
    prompts_per_step: int = 16  # 每一步抽幾題
    group_size: int = 8  # 每題抽幾個回答
    max_new_tokens: int = 4  # SFT 模型的回覆是「一個字加結尾」，2 個 token；留一點餘裕
    temperature: float = 1.0  # 取樣要照 policy 本來的機率（1.0），ratio 才有意義
    epochs: int = 2  # 同一批 rollout 更新幾次；第 1 次 ratio 恆為 1，之後才會碰到 clip
    max_steps: int = 600
    lr: float = 3e-5  # 固定，不排程
    clip_eps: float = 0.2
    normalize_std: bool = True  # group_advantages 要不要除以組內標準差
    length_norm: bool = True  # grpo_loss 要不要對每條回答的 token 平均
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 50
    n_valid: int = 500  # 驗證題（color 與 who 的 valid 題，固定種子 1，和 RL 的種子無關：不同種子的結果才能比）
    seed: int = 0
    device: str = ""  # 空字串代表用 get_device()


def train_rlvr(cfg: RLVRConfig, tokenizer=None) -> list[dict]:
    """RLVR 主迴圈：rollout -> 0/1 獎勵 -> group advantage -> 對同一批 rollout 更新 epochs 次。回傳 log 紀錄。"""
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device) if cfg.device else get_device()
    os.makedirs(cfg.out_dir, exist_ok=True)
    tokenizer = tokenizer or Tokenizer.load(cfg.tokenizer_path)
    kinds = tuple(cfg.kinds.split(","))
    train_tasks = make_tasks(cfg.max_steps * cfg.prompts_per_step, seed=cfg.seed, split="train", kinds=kinds)
    valid_tasks = make_tasks(cfg.n_valid, seed=1, split="valid", kinds=kinds)

    model = TransformerLM(cfg.model).to(device)
    if cfg.sft_path:
        model.load_state_dict(torch.load(cfg.sft_path, map_location=device))
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    records, log_file, t0 = [], open(os.path.join(cfg.out_dir, "log.jsonl"), "w"), time.time()

    def log(rec: dict) -> None:
        records.append(rec)
        log_file.write(json.dumps(rec) + "\n")
        log_file.flush()
        print(rec)

    log({"step": 0, "val_accuracy": accuracy(model, tokenizer, valid_tasks, cfg.max_new_tokens)})
    for step in range(1, cfg.max_steps + 1):
        tasks = train_tasks[(step - 1) * cfg.prompts_per_step : step * cfg.prompts_per_step]
        out = rollout(model, tokenizer, tasks, cfg.group_size, cfg.max_new_tokens, cfg.temperature)
        ids, mask = out.ids.to(device), out.mask.to(device)[:, 1:]  # mask 跟著 token_logprobs 少一格
        advantages = group_advantages(out.rewards, cfg.normalize_std).reshape(-1).to(device)
        with torch.no_grad():
            old_logp = token_logprobs(model, ids)  # rollout 當時的 policy 的 log 機率，更新時不變
        for _ in range(cfg.epochs):
            logp = token_logprobs(model, ids)
            loss = grpo_loss(logp, old_logp, advantages, mask, cfg.clip_eps, cfg.length_norm)
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        ratio = (logp.detach() - old_logp).exp()  # 最後一次更新前的 ratio
        rec = {"step": step, "reward": round(out.rewards.mean().item(), 4), "loss": round(loss.item(), 4) + 0.0,  # + 0.0：避免印出 -0.0
               "clip_frac": round((((ratio - 1).abs() > cfg.clip_eps) & mask).sum().item() / mask.sum().item(), 4),
               "mixed_groups": round((out.rewards.std(dim=1) > 0).float().mean().item(), 3),  # 組內有對有錯：只有這些組有梯度
               "resp_len": round(mask.sum().item() / len(ids), 2), "grad_norm": round(grad_norm.item(), 3), "time": round(time.time() - t0, 1)}
        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            rec["val_accuracy"] = accuracy(model, tokenizer, valid_tasks, cfg.max_new_tokens)
        if step % cfg.log_every == 0 or step == cfg.max_steps or "val_accuracy" in rec:
            log(rec)
        else:
            records.append(rec)
    log_file.close()
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model.pt"))
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    return records
# ANCHOR_END: train_rlvr
