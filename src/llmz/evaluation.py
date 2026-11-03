"""評估（第 12 章）：帶 mask 的序列 log 機率、多選題評分、few-shot 提示、準確率。"""

import json
from pathlib import Path

import torch
from torch import Tensor, nn

from llmz.functional import log_softmax

MCQ_PATH = Path(__file__).parent / "datasets" / "tinystories_mcq.jsonl"


# ANCHOR: load_mcq
def load_mcq(split: str | None = None, path: Path = MCQ_PATH) -> list[dict]:
    """讀本書附的多選題集。每題是 {"split", "context", "choices", "answer"}；split 是 "dev"（5 題，few-shot 範例）或 "test"（100 題）。"""
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [it for it in items if split is None or it["split"] == split]
# ANCHOR_END: load_mcq


# ANCHOR: sequence_logprob
def sequence_logprob(model: nn.Module, ids: Tensor, mask: Tensor) -> Tensor:
    """ids (B, T)、mask (B, T) -> (B,)：每條序列裡 mask 為 True 的 token 的 log 機率總和。

    mask[:, t] 為 True 代表「第 t 個 token 要算進去」。第 0 個 token 沒有前文可以預測，永遠不算。
    沒有 no_grad：之後（第 14 章）要對它做 backward。
    """
    logp = log_softmax(model(ids[:, :-1]), dim=-1)  # (B, T-1, V)：位置 t 預測 token t+1
    batch, position = torch.arange(ids.shape[0], device=ids.device)[:, None], torch.arange(ids.shape[1] - 1, device=ids.device)
    token_logp = logp[batch, position, ids[:, 1:]]  # (B, T-1)：broadcasting 之後每個 token 只取自己那一格
    return (token_logp * mask[:, 1:]).sum(dim=1)
# ANCHOR_END: sequence_logprob


# ANCHOR: choice_logprobs
@torch.no_grad()
def choice_logprobs(model: nn.Module, tokenizer, context: str, choices: list[str]) -> tuple[Tensor, Tensor]:
    """每個選項接在 context 後面的 log 機率總和 (C,)，以及每個選項的 token 數 (C,)。

    context 與選項分開 encode 再接起來；選項開頭的空格要自己帶（" park"），token 的切法才和正常文字一樣。
    shortcut: context 太長就從左邊截掉，few-shot 放幾個範例要自己控制；要精確處理時改成丟出錯誤。
    """
    device = next(model.parameters()).device
    ctx = tokenizer.encode(context)
    opts = [tokenizer.encode(c) for c in choices]
    room = model.cfg.context_length - max(len(o) for o in opts)
    if room < 1:
        raise ValueError("選項比 context_length 還長")
    ctx = ctx[-room:]
    ids = torch.zeros(len(opts), len(ctx) + max(len(o) for o in opts), dtype=torch.long)  # 短的往右補 0
    mask = torch.zeros_like(ids, dtype=torch.bool)  # 只有選項的 token 是 True
    for i, o in enumerate(opts):
        ids[i, : len(ctx) + len(o)] = torch.tensor(ctx + o)
        mask[i, len(ctx) : len(ctx) + len(o)] = True
    logprobs = sequence_logprob(model, ids.to(device), mask.to(device))  # causal：右邊補的 0 不影響左邊
    return logprobs.cpu(), mask.sum(dim=1)
# ANCHOR_END: choice_logprobs


# ANCHOR: pick_choice
def pick_choice(logprobs: Tensor, lengths: Tensor | None = None) -> int:
    """選分數最高的選項。lengths 是 None 時比總 log 機率，否則比「每單位長度的平均 log 機率」。"""
    score = logprobs if lengths is None else logprobs / lengths
    return int(score.argmax())
# ANCHOR_END: pick_choice


# ANCHOR: build_prompt
def build_prompt(context: str, shots: list[dict] = (), sep: str = "\n\n") -> str:
    """few-shot 提示：先放幾題「題目加正確答案」，最後才是要考的題目。shots 是空的就是 0-shot。"""
    solved = [shot["context"] + shot["choices"][shot["answer"]] for shot in shots]
    return sep.join(solved + [context])
# ANCHOR_END: build_prompt


# ANCHOR: predict_mcq
def predict_mcq(model: nn.Module, tokenizer, items: list[dict], shots: list[dict] = (),
                normalize: str = "none", sep: str = "\n\n") -> list[int]:
    """對每一題挑出分數最高的選項。normalize 是 "none"（總 log 機率）、"token"（除以選項的 token 數）或 "byte"（除以位元組數）。"""
    if normalize not in ("none", "token", "byte"):
        raise ValueError(f"不認得的 normalize：{normalize}")
    predictions = []
    for it in items:
        logprobs, n_tokens = choice_logprobs(model, tokenizer, build_prompt(it["context"], shots, sep), it["choices"])
        lengths = None  # "none"
        if normalize == "token":
            lengths = n_tokens
        elif normalize == "byte":
            lengths = torch.tensor([len(c.encode()) for c in it["choices"]])
        predictions.append(pick_choice(logprobs, lengths))
    return predictions
# ANCHOR_END: predict_mcq


# ANCHOR: mcq_accuracy
def mcq_accuracy(model: nn.Module, tokenizer, items: list[dict], shots: list[dict] = (),
                 normalize: str = "none", sep: str = "\n\n") -> float:
    """答對的題數除以總題數。"""
    predictions = predict_mcq(model, tokenizer, items, shots, normalize, sep)
    return sum(p == it["answer"] for p, it in zip(predictions, items)) / len(items)
# ANCHOR_END: mcq_accuracy
