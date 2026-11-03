import torch

from llmz.generate import generate, greedy_next, sample_next
from llmz.model import ModelConfig, TransformerLM
from llmz.tokenizer import Tokenizer, train_bpe_naive


def test_greedy_picks_argmax():
    assert greedy_next(torch.tensor([0.1, 2.0, -1.0])).item() == 1


def test_temperature_zero_like_and_top_p_restricts_support():
    torch.manual_seed(0)
    logits = torch.tensor([3.0, 2.0, 1.0, -5.0, -5.0])
    low_t = [sample_next(logits, temperature=0.05).item() for _ in range(50)]
    assert set(low_t) == {0}  # 溫度很低時幾乎永遠選最大的
    top = [sample_next(logits, top_p=0.5).item() for _ in range(200)]
    assert set(top) <= {0, 1}
    hot = [sample_next(logits, temperature=5.0).item() for _ in range(300)]
    assert len(set(hot)) >= 4  # 溫度高時連很低分的也會被抽到


def test_generate_respects_context_and_stop_token():
    torch.manual_seed(0)
    text = "a b c d e f g<|endoftext|>a b c d e f g<|endoftext|>"
    vocab, merges = train_bpe_naive(text, 270, ["<|endoftext|>"])
    tok = Tokenizer(vocab, merges, ["<|endoftext|>"])
    model = TransformerLM(ModelConfig(vocab_size=270, context_length=4, d_model=16, n_layers=1, n_heads=2, d_ff=32))
    assert generate(model, tok, "a b", max_new_tokens=0) == "a b"
    out = generate(model, tok, "a b", max_new_tokens=6, temperature=0)  # context_length 4 < 8，會截掉前面
    assert out.startswith("a b")
    torch.manual_seed(0)
    assert generate(model, tok, "a b", max_new_tokens=6, temperature=0) == out  # greedy 是確定的
