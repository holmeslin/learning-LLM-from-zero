"""Transformer 語言模型（第 4 章第 13～14 集）。"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from llmz.attention import MultiHeadSelfAttention, RoPE
from llmz.layers import Embedding, Identity, Linear, RMSNorm, SiLUFFN, SwiGLU


# ANCHOR: config
@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    n_layers: int = 4
    n_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0
    # 下面四個是第 6 章消融實驗用的開關，預設就是標準的 Transformer
    norm: str = "rms"  # "rms" 或 "none"
    norm_position: str = "pre"  # "pre" 或 "post"
    rope: bool = True
    ffn: str = "swiglu"  # "swiglu" 或 "silu"
    activation_checkpoint: bool = False  # 第 7 章第 6 集：activation checkpointing
# ANCHOR_END: config


def make_norm(cfg: ModelConfig) -> nn.Module:
    return RMSNorm(cfg.d_model) if cfg.norm == "rms" else Identity()


# ANCHOR: block
class TransformerBlock(nn.Module):
    """一層：attention 加殘差，再前饋網路加殘差；RMSNorm 放在每個子層之前（pre-norm）。"""

    def __init__(self, cfg: ModelConfig, rope: RoPE | None):
        super().__init__()
        self.norm_position = cfg.norm_position
        self.norm1 = make_norm(cfg)
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, rope)
        self.norm2 = make_norm(cfg)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff) if cfg.ffn == "swiglu" else SiLUFFN(cfg.d_model, cfg.d_ff)

    def forward(self, x: Tensor) -> Tensor:
        if self.norm_position == "pre":
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
        else:  # post-norm：先加殘差再 norm
            x = self.norm1(x + self.attn(x))
            x = self.norm2(x + self.ffn(x))
        return x
# ANCHOR_END: block


# ANCHOR: lm
class TransformerLM(nn.Module):
    """token id (B, T) -> 每個位置對下一個 token 的 logits (B, T, vocab_size)。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = Embedding(cfg.vocab_size, cfg.d_model)
        d_head = cfg.d_model // cfg.n_heads
        rope = RoPE(d_head, cfg.context_length, cfg.rope_theta) if cfg.rope else None
        self.blocks = nn.ModuleList([TransformerBlock(cfg, rope) for _ in range(cfg.n_layers)])
        self.norm_final = make_norm(cfg)
        self.lm_head = Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, ids: Tensor) -> Tensor:
        x = self.embed(ids)
        for block in self.blocks:
            if self.cfg.activation_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)  # 第 7 章第 6 集：不存中間結果，backward 時重算
            else:
                x = block(x)
        return self.lm_head(self.norm_final(x))
# ANCHOR_END: lm
