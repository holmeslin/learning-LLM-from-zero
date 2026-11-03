"""推論（第 11 章）：KV cache、帶 cache 的生成、speculative decoding、權重量化。"""

import torch
from torch import Tensor

from llmz.attention import MultiHeadSelfAttention, scaled_dot_product_attention
from llmz.functional import softmax
from llmz.generate import sample_next
from llmz.layers import Linear
from llmz.model import ModelConfig, TransformerLM


# ANCHOR: kv_cache
class KVCache:
    """每一層的 K、V 預先配置好，形狀 (n_layers, B, H, max_len, d_head)；length 是目前存了幾個位置。"""

    def __init__(self, cfg: ModelConfig, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32):
        shape = (cfg.n_layers, batch_size, cfg.n_heads, cfg.context_length, cfg.d_model // cfg.n_heads)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0

    def nbytes(self) -> int:
        return 2 * self.k.numel() * self.k.element_size()
# ANCHOR_END: kv_cache


# ANCHOR: attention_cached
def attention_cached(attn: MultiHeadSelfAttention, x: Tensor, cache: KVCache, layer: int) -> Tensor:
    """和 MultiHeadSelfAttention.forward 一樣，但 x 只有新的 T 個位置，前面 P 個位置的 K、V 從 cache 讀。"""
    B, T, _ = x.shape
    P = cache.length
    q = attn.q_proj(x).reshape(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
    k = attn.k_proj(x).reshape(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
    v = attn.v_proj(x).reshape(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
    positions = torch.arange(P, P + T, device=x.device)  # 新的位置接在 cache 後面
    if attn.rope is not None:
        q = attn.rope(q, positions)
        k = attn.rope(k, positions)
    cache.k[layer, :, :, P : P + T] = k
    cache.v[layer, :, :, P : P + T] = v
    # 對整個 cache 做 attention，還沒用到的位置遮掉：形狀固定，MPS 才不用每一步重新編譯
    mask = torch.arange(cache.k.shape[3], device=x.device) <= positions[:, None]  # (T, max_len)
    out = scaled_dot_product_attention(q, cache.k[layer], cache.v[layer], mask)
    return attn.out_proj(out.transpose(1, 2).reshape(B, T, attn.n_heads * attn.d_head))
# ANCHOR_END: attention_cached


# ANCHOR: forward_cached
@torch.no_grad()
def forward_cached(model: TransformerLM, ids: Tensor, cache: KVCache) -> Tensor:
    """只餵還沒進 cache 的 token (B, T)，回傳它們的 logits (B, T, V)，cache.length 加 T。

    shortcut: 只支援 pre-norm（標準設定）；要支援 post-norm 時照 TransformerBlock.forward 補上另一個分支。
    """
    if model.cfg.norm_position != "pre":
        raise ValueError("forward_cached 只支援 pre-norm")
    T = ids.shape[1]
    if cache.length + T > model.cfg.context_length:
        raise ValueError(f"超過 context_length {model.cfg.context_length}：cache 裡的位置已經用 RoPE 轉過，不能往前滑")
    x = model.embed(ids)
    for i, block in enumerate(model.blocks):
        x = x + attention_cached(block.attn, block.norm1(x), cache, i)
        x = x + block.ffn(block.norm2(x))
    cache.length += T
    return model.lm_head(model.norm_final(x))
# ANCHOR_END: forward_cached


# ANCHOR: generate_cached
def pick_next(logits: Tensor, temperature: float = 0.0, top_p: float = 1.0) -> Tensor:
    """logits (B, V) -> 下一個 token (B, 1)。temperature 0 用 greedy，否則每一列各自抽。"""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    return torch.stack([sample_next(row, temperature, top_p) for row in logits])[:, None]


@torch.no_grad()
def generate_cached(model: TransformerLM, ids: Tensor, max_new_tokens: int,
                    temperature: float = 0.0, top_p: float = 1.0) -> Tensor:
    """ids (B, T) -> (B, T + max_new_tokens)。先 prefill 整段 prompt，之後每一步只餵一個新 token。"""
    cache = KVCache(model.cfg, ids.shape[0], ids.device)
    next_ids = pick_next(forward_cached(model, ids, cache)[:, -1], temperature, top_p)  # prefill
    out = [ids, next_ids]
    for _ in range(max_new_tokens - 1):  # decode
        next_ids = pick_next(forward_cached(model, next_ids, cache)[:, -1], temperature, top_p)
        out.append(next_ids)
    return torch.cat(out, dim=1)
# ANCHOR_END: generate_cached


# ANCHOR: accept_or_resample
def to_probs(logits: Tensor, temperature: float) -> Tensor:
    """temperature 0 時是 one-hot（全部機率給最大的那個），否則是 softmax(logits / temperature)。"""
    if temperature == 0:
        return torch.zeros_like(logits).scatter(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
    return softmax(logits / temperature, dim=-1)


def accept_or_resample(p: Tensor, q: Tensor, token: int) -> tuple[bool, int]:
    """token 是依草稿分布 q 抽出來的。以 min(1, p/q) 的機率接受，否則從 max(0, p - q) 重抽；結果恰好服從 p。"""
    if torch.rand(()).item() < (p[token] / q[token]).item():
        return True, token
    residual = (p - q).clamp(min=0)
    return False, torch.multinomial(residual / residual.sum(), 1).item()
# ANCHOR_END: accept_or_resample


# ANCHOR: speculative
@torch.no_grad()
def speculative_generate(target: TransformerLM, draft: TransformerLM, ids: Tensor, max_new_tokens: int,
                         k: int = 4, temperature: float = 0.0) -> tuple[Tensor, int, int]:
    """ids (1, T)。每一輪草稿模型先猜 k 個 token，目標模型一次 forward 驗證。回傳 (結果, 接受的草稿數, 草稿總數)。"""
    tokens = ids[0].tolist()
    goal = len(tokens) + max_new_tokens
    if goal + k > target.cfg.context_length:
        raise ValueError("prompt + max_new_tokens + k 超過 context_length")
    t_cache, d_cache = KVCache(target.cfg, 1, ids.device), KVCache(draft.cfg, 1, ids.device)
    accepted = proposed = 0

    def feed(model: TransformerLM, cache: KVCache, seq: list[int]) -> Tensor:  # 把還沒進 cache 的部分餵進去
        return forward_cached(model, torch.tensor([seq[cache.length :]], device=ids.device), cache)[0]

    while len(tokens) < goal:
        n = len(tokens)
        drafts, qs = [], []
        for _ in range(k):  # 1. 草稿模型一個一個猜
            q = to_probs(feed(draft, d_cache, tokens + drafts)[-1], temperature)
            drafts.append(torch.multinomial(q, 1).item())
            qs.append(q)
        ps = to_probs(feed(target, t_cache, tokens + drafts)[-(k + 1) :], temperature)  # 2. 一次驗證：(k+1, V)
        n_ok = 0
        for i, t in enumerate(drafts):  # 3. 從頭逐一決定接受或重抽，第一次拒絕就停
            ok, t = accept_or_resample(ps[i], qs[i], t)
            tokens.append(t)
            if not ok:
                break
            n_ok += 1
        else:
            tokens.append(torch.multinomial(ps[k], 1).item())  # 全部接受：目標模型多送一個
        accepted += n_ok
        proposed += k
        t_cache.length = min(t_cache.length, n + n_ok)  # 被拒絕的草稿之後的 K、V 作廢
        d_cache.length = min(d_cache.length, n + n_ok)
    return torch.tensor([tokens[:goal]], device=ids.device), accepted, proposed
# ANCHOR_END: speculative


# ANCHOR: quantize
def quantize(w: Tensor, bits: int = 8, group_size: int | None = None) -> tuple[Tensor, Tensor]:
    """對稱量化 (out, in) 的權重：每一列每 group_size 個一組（預設整列一組），
    除以「最大絕對值 / qmax」後四捨五入成 [-qmax, qmax] 的整數。回傳 (整數 q, 縮放 scale)。"""
    out_features, in_features = w.shape
    g = group_size or in_features
    groups = w.reshape(out_features, in_features // g, g)
    qmax = 2 ** (bits - 1) - 1
    scale = (groups.abs().amax(dim=-1, keepdim=True) / qmax).clamp(min=1e-12)  # 整組都是 0 時避免除以 0
    q = torch.round(groups / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale


def dequantize(q: Tensor, scale: Tensor) -> Tensor:
    return (q.float() * scale).reshape(q.shape[0], -1)


def quantize_model_(model: TransformerLM, bits: int = 8, group_size: int | None = None) -> int:
    """把每個 Linear 的權重換成「量化再還原」的值（模擬量化），回傳這些權重實際儲存要的位元組。

    shortcut: 4 bit 的整數也放在 int8 裡、不打包；位元組照 bits/8 算，真的要省記憶體時兩個 4 bit 擠一個位元組。
    """
    nbytes = 0
    for m in model.modules():
        if isinstance(m, Linear):
            q, scale = quantize(m.weight.data, bits, group_size)
            m.weight.data = dequantize(q, scale).to(m.weight.dtype)
            nbytes += q.numel() * bits // 8 + scale.numel() * scale.element_size()
    return nbytes
# ANCHOR_END: quantize
