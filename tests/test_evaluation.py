import math
from collections import Counter

import torch
from torch import nn

from llmz.evaluation import (
    build_prompt,
    choice_logprobs,
    load_mcq,
    mcq_accuracy,
    pick_choice,
    predict_mcq,
    sequence_logprob,
)
from llmz.functional import cross_entropy
from llmz.model import ModelConfig, TransformerLM

# 三個符號 0、1、2 的 bigram「模型」：P[a][b] 是「現在是 a、下一個是 b」的機率。
# 機率分布是我們自己定的，所以每個序列的 log 機率都可以手算。
P = torch.tensor([[0.1, 0.6, 0.3], [0.2, 0.2, 0.6], [0.5, 0.25, 0.25]])


class CharTokenizer:
    """只認得 a、b、c 的 tokenizer：a=0、b=1、c=2。"""

    def encode(self, text: str) -> list[int]:
        return ["abc".index(ch) for ch in text if ch in "abc"]  # 空白和換行直接忽略


class BigramLM(nn.Module):
    """logits[b, t] = log P[ids[b, t]]：下一個 token 只看現在這一個。"""

    def __init__(self, probs: torch.Tensor):
        super().__init__()
        self.logits = nn.Parameter(probs.log())
        self.cfg = ModelConfig(vocab_size=3, context_length=16)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.logits[ids]


def test_sequence_logprob_matches_hand_computation():
    model = BigramLM(P)
    ids = torch.tensor([[0, 1, 2], [2, 0, 1]])
    mask = torch.tensor([[False, True, True], [False, False, True]])
    got = sequence_logprob(model, ids, mask)
    # 第一條：log P[0][1] + log P[1][2]；第二條只算最後一格：log P[0][1]
    want = torch.tensor([math.log(0.6) + math.log(0.6), math.log(0.6)])
    assert torch.allclose(got, want, atol=1e-6)


def test_sequence_logprob_ignores_first_position_and_masked_out_tokens():
    model = BigramLM(P)
    ids = torch.tensor([[0, 1, 2]])
    only_last = sequence_logprob(model, ids, torch.tensor([[False, False, True]]))
    with_first = sequence_logprob(model, ids, torch.tensor([[True, False, True]]))  # 第 0 格沒有前文，不能算
    assert torch.equal(only_last, with_first)
    assert sequence_logprob(model, ids, torch.zeros(1, 3, dtype=torch.bool)).item() == 0.0


def test_sequence_logprob_has_gradient_toward_the_masked_tokens():
    model = BigramLM(P)
    ids = torch.tensor([[0, 1, 2]])
    sequence_logprob(model, ids, torch.tensor([[False, False, True]])).sum().backward()
    # 只有「1 之後接 2」這一格被算進去：它的 logit 要變大，其他 logit 的梯度加起來是 0
    grad = model.logits.grad
    assert grad[1, 2] > 0 and grad[0].abs().sum() == 0 and grad[2].abs().sum() == 0


def test_sequence_logprob_with_all_ones_mask_is_negative_cross_entropy():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=50, context_length=16, d_model=16, n_layers=1, n_heads=2, d_ff=32)
    model = TransformerLM(cfg).eval()
    ids = torch.randint(0, 50, (3, 10))
    got = sequence_logprob(model, ids, torch.ones(3, 10, dtype=torch.bool))
    logits = model(ids[:, :-1])
    for b in range(3):  # 每條序列自己的 cross-entropy 乘上 token 數，就是總 log 機率
        ce = cross_entropy(logits[b], ids[b, 1:])
        assert torch.allclose(got[b], -ce * 9, atol=1e-4)


def test_choice_logprobs_scores_each_choice_given_the_context():
    model, tok = BigramLM(P), CharTokenizer()
    logprobs, n_tokens = choice_logprobs(model, tok, "a", ["b", "c", "bc"])  # 長短不同，內部要補齊
    want = [math.log(0.6), math.log(0.3), math.log(0.6) + math.log(0.6)]  # "bc"：a→b、b→c
    assert torch.allclose(logprobs, torch.tensor(want), atol=1e-6)
    assert n_tokens.tolist() == [1, 1, 2]


def test_pick_choice_finds_the_most_likely_continuation():
    model, tok = BigramLM(P), CharTokenizer()
    for context, choices, answer in [("a", ["c", "b", "a"], 1), ("b", ["a", "b", "c"], 2), ("c", ["b", "c", "a"], 2)]:
        logprobs, _ = choice_logprobs(model, tok, context, choices)
        assert pick_choice(logprobs) == answer


# 另一個分布：在 a 後面，單獨一個 "c"（機率 0.5）比 "bbbb"（0.3 × 0.9³）更可能，
# 但 "bbbb" 每個 token 平均更可能。
P_LONG = torch.tensor([[0.2, 0.3, 0.5], [0.05, 0.9, 0.05], [0.3, 0.3, 0.4]])


def test_length_normalization_changes_the_choice():
    model, tok = BigramLM(P_LONG), CharTokenizer()
    logprobs, n_tokens = choice_logprobs(model, tok, "a", ["c", "bbbb"])
    assert pick_choice(logprobs) == 0  # 總和：log 0.5 = -0.69 > log(0.3 × 0.729) = -1.52
    assert pick_choice(logprobs, n_tokens) == 1  # 每個 token：-0.69 < -0.38


def test_build_prompt_puts_solved_examples_before_the_question():
    shot = {"context": "a b", "choices": [" c", " d"], "answer": 1}
    assert build_prompt("x y") == "x y"
    assert build_prompt("x y", [shot]) == "a b d\n\nx y"
    assert build_prompt("x y", [shot, shot], sep="|") == "a b d|a b d|x y"


def item(context: str, choices: list[str], answer: int) -> dict:
    return {"context": context, "choices": choices, "answer": answer}


def test_mcq_accuracy_counts_correct_predictions():
    model, tok = BigramLM(P), CharTokenizer()
    items = [item("a", ["b", "c"], 0), item("b", ["a", "c"], 1), item("c", ["b", "a"], 0)]  # 最後一題標錯答案
    assert predict_mcq(model, tok, items) == [0, 1, 1]
    assert math.isclose(mcq_accuracy(model, tok, items), 2 / 3)


def test_mcq_accuracy_with_length_normalization():
    model, tok = BigramLM(P_LONG), CharTokenizer()
    items = [item("a", ["c", "bbbb"], 1)]
    assert mcq_accuracy(model, tok, items) == 0.0
    assert mcq_accuracy(model, tok, items, normalize="token") == 1.0
    assert mcq_accuracy(model, tok, items, normalize="byte") == 1.0


def test_few_shot_examples_reach_the_model():
    class FirstTokenLM(BigramLM):
        """下一個 token 的分布只由整段序列的第一個 token 決定：prompt 最前面放什麼會改變結果。"""

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return self.logits[ids[:, :1]].expand(-1, ids.shape[1], -1)

    model, tok = FirstTokenLM(P), CharTokenizer()
    items = [item("a", ["b", "c"], 0)]
    shot = item("b", ["b"], 0)
    assert predict_mcq(model, tok, items) == [0]  # 從 a 開頭：b 比 c 可能
    assert predict_mcq(model, tok, items, shots=[shot]) == [1]  # 從 b 開頭：c 比 b 可能


def test_packaged_question_set_is_well_formed():
    test, dev = load_mcq("test"), load_mcq("dev")
    assert (len(test), len(dev)) == (100, 5)
    assert len(load_mcq()) == 105 and len({it["context"] for it in load_mcq()}) == 105  # 題目不重複
    for it in test + dev:
        assert len(it["choices"]) == 4 and len(set(it["choices"])) == 4
        assert 0 <= it["answer"] < 4
        assert all(c.startswith(" ") for c in it["choices"])  # 選項開頭帶空格，接在 context 後面才是正常文字
    assert Counter(it["answer"] for it in test) == {0: 25, 1: 25, 2: 25, 3: 25}  # 正確答案的位置沒有偏向
