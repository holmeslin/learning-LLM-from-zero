"""從訓練好的模型生成文字（第 6 章第 1～4 集）：greedy、temperature、top-p。"""

import torch
from torch import Tensor

from llmz.functional import softmax
from llmz.model import TransformerLM
from llmz.tokenizer import Tokenizer


# ANCHOR: greedy_next
def greedy_next(logits: Tensor) -> Tensor:
    """永遠選分數最高的 token。logits (V,) -> 純量 id。"""
    return torch.argmax(logits)
# ANCHOR_END: greedy_next


# ANCHOR: sample_next
def sample_next(logits: Tensor, temperature: float = 1.0, top_p: float = 1.0) -> Tensor:
    """依機率抽一個 token。temperature 控制分布多平，top_p 只在機率最高的那一小群裡抽。"""
    probs = softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative - sorted_probs < top_p  # 累積到 top_p 之前的那些都留下（至少留第一個）
        probs = torch.zeros_like(probs).scatter(-1, sorted_ids[keep], sorted_probs[keep])
        probs = probs / probs.sum()
    return torch.multinomial(probs, 1)[0]
# ANCHOR_END: sample_next


# ANCHOR: generate
@torch.no_grad()
def generate(model: TransformerLM, tokenizer: Tokenizer, prompt: str, max_new_tokens: int = 100,
             temperature: float = 1.0, top_p: float = 1.0, stop_token: str = "<|endoftext|>") -> str:
    """從 prompt 開始，一次生成一個 token，碰到 stop_token 或達到上限就停。"""
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt)
    stop_id = tokenizer.special_to_id.get(stop_token)
    context_length = model.cfg.context_length
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-context_length:]], device=device)  # 只餵最後 context_length 個
        logits = model(x)[0, -1]  # 最後一個位置對下一個 token 的 logits
        if temperature == 0:
            next_id = greedy_next(logits)
        else:
            next_id = sample_next(logits, temperature, top_p)
        if next_id.item() == stop_id:
            break
        ids.append(next_id.item())
    return tokenizer.decode(ids)
# ANCHOR_END: generate
