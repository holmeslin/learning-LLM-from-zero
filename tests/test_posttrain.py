import json
import math
from dataclasses import replace

import torch
from torch import nn

from llmz.model import ModelConfig, TransformerLM
import re

from llmz.posttrain import (KINDS, DPOConfig, Pair, SFTConfig, Task, accuracy, bradley_terry_loss, chat_pieces, dpo_loss, encode_chat, generate_replies, make_preferences,
                            make_tasks, pad_batch, pair_batch, pair_logprobs, preference_rate, reward, sft_loss, task_messages, train_dpo, train_sft, wrong_answers)
from llmz.tokenizer import Tokenizer, base_vocab

END = "<|endoftext|>"
TOK = Tokenizer(base_vocab([END]), [], [END])  # 沒有 merges：一個 byte 一個 token，再加上 <|endoftext|>，所以每個 token 在哪裡一目了然
QA = [{"role": "user", "content": "2 + 3?"}, {"role": "assistant", "content": "five"}]


def test_chat_pieces_marks_only_assistant_reply_as_trainable():
    assert chat_pieces(QA) == [("User: 2 + 3?\n", False), ("Assistant:", False), (" five" + END, True)]


def test_generation_prompt_ends_at_assistant_header_and_trains_nothing():
    pieces = chat_pieces(QA[:1], generation_prompt=True)
    assert pieces == [("User: 2 + 3?\n", False), ("Assistant:", False)]


def test_encode_chat_mask_covers_exactly_the_reply_and_end_token():
    ids, mask = encode_chat(TOK, QA)
    assert TOK.decode(ids[:-1]) == "User: 2 + 3?\nAssistant: five"  # 最後一個是 <|endoftext|>
    assert ids[-1] == TOK.special_to_id[END]
    assert len(ids) == len(mask)
    assert TOK.decode([i for i, m in zip(ids, mask) if m][:-1]) == " five"
    assert sum(mask) == len(" five") + 1  # 回覆的每個 token 加上結尾


def test_encode_chat_equals_encoding_the_joined_text_in_one_go():
    # 一段一段 encode 再接起來，要和整串一起 encode 一樣：mask 才會對得上 ids
    ids, _ = encode_chat(TOK, QA)
    assert ids == TOK.encode("".join(text for text, _ in chat_pieces(QA)))


def test_multi_turn_trains_every_assistant_turn():
    messages = QA + [{"role": "user", "content": "6 - 1?"}, {"role": "assistant", "content": "five"}]
    ids, mask = encode_chat(TOK, messages)
    assert sum(mask) == 2 * (len(" five") + 1)
    assert not mask[0] and mask[-1]


# 三個符號 0、1、2 的 bigram「模型」（和 test_evaluation 一樣）：P[a][b] 是「現在是 a、下一個是 b」的機率，所以 loss 可以手算。
P = torch.tensor([[0.1, 0.6, 0.3], [0.2, 0.2, 0.6], [0.5, 0.25, 0.25]])


class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(P.log())
        self.cfg = ModelConfig(vocab_size=3, context_length=16)

    def forward(self, ids):
        return self.logits[ids]


def test_pad_batch_pads_right_and_never_trains_the_padding():
    ids, mask = pad_batch([([0, 1, 2], [False, True, True]), ([2, 0], [False, True])])
    assert ids.tolist() == [[0, 1, 2], [2, 0, 0]]
    assert mask.tolist() == [[False, True, True], [False, True, False]]


def test_sft_loss_is_mean_negative_log_prob_of_masked_tokens():
    ids, mask = pad_batch([([0, 1, 2], [False, True, True]), ([2, 0], [False, True])])
    # 被算進去的是三個 token：0→1、1→2、2→0；其他位置（第 0 個 token、填充）都不算
    want = -(math.log(0.6) + math.log(0.6) + math.log(0.5)) / 3
    assert abs(sft_loss(BigramLM(), ids, mask).item() - want) < 1e-6


def test_sft_loss_ignores_tokens_outside_the_mask():
    model = BigramLM()
    mask = torch.tensor([[False, False, True, True]])
    a = sft_loss(model, torch.tensor([[0, 1, 2, 2]]), mask)
    b = sft_loss(model, torch.tensor([[2, 1, 2, 2]]), mask)  # 只改最前面那個沒被算的 token
    assert torch.equal(a, b)


def test_sft_loss_backward_gives_gradient_only_to_rows_that_were_trained():
    model = BigramLM()
    ids, mask = torch.tensor([[0, 1, 2]]), torch.tensor([[False, False, True]])  # 只算 1→2
    sft_loss(model, ids, mask).backward()
    assert model.logits.grad[1].abs().sum() > 0  # 「現在是 1」那一列有梯度
    assert model.logits.grad[0].abs().sum() == 0 and model.logits.grad[2].abs().sum() == 0


# 下面幾個測試用「從題目文字自己算答案」檢查 make_tasks 給的答案：數字詞與規則都是這裡另外寫的，不是從產生器抄來的。
WORDS = "zero one two three four five six seven eight nine ten".split()


def number(word: str) -> int:
    return WORDS.index(word)


def tasks_of(kind: str, n: int = 200):
    return [t for t in make_tasks(n * len(KINDS) * 2, seed=3) if t.kind == kind][:n]


def test_make_tasks_is_deterministic_and_seed_dependent():
    assert make_tasks(40, seed=1) == make_tasks(40, seed=1)
    assert make_tasks(40, seed=1) != make_tasks(40, seed=2)
    assert {t.kind for t in make_tasks(200, seed=0)} == set(KINDS)


def test_make_tasks_can_be_limited_to_some_kinds():
    tasks = make_tasks(100, seed=0, kinds=("color", "who"))
    assert {t.kind for t in tasks} == {"color", "who"}
    assert make_tasks(5, seed=0, kinds=("add",))[0].kind == "add"


def test_train_and_valid_questions_never_overlap():
    train = {t.question for t in make_tasks(3000, seed=0, split="train")}
    valid = {t.question for t in make_tasks(1000, seed=0, split="valid")}
    assert valid and not train & valid
    assert len(train) > 1000  # 題目的組合夠多，不是同幾題重複


def test_add_answers_are_the_sum():
    tasks = tasks_of("add")
    assert len(tasks) == 200
    for t in tasks:
        a, b = [number(w) for w in re.findall(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b", t.question)]
        assert t.answer == WORDS[a + b], t.question


def test_subtract_answers_are_the_difference():
    for t in tasks_of("subtract"):
        a, b = [number(w) for w in re.findall(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b", t.question)]
        assert b < a and t.answer == WORDS[a - b], t.question


def test_count_answers_are_the_number_of_items():
    for t in tasks_of("count"):
        items = re.search(r"has (.*)\. How many", t.question).group(1)
        assert t.answer == WORDS[len(re.findall(r"\ban? [a-z]+", items))], t.question


def test_color_answers_are_the_color_of_the_asked_thing():
    for t in tasks_of("color"):
        asked = re.search(r"What color is the (\w+)\?", t.question).group(1)
        assert t.answer == re.search(rf"an? (\w+) {asked}\b", t.question).group(1), t.question


def test_who_answers_are_the_owner_of_the_asked_thing():
    for t in tasks_of("who"):
        asked = re.search(r"Who has the (\w+)\?", t.question).group(1)
        assert t.answer == re.search(rf"(\w+) has an? {asked}\b", t.question).group(1), t.question


def test_questions_use_a_and_an_correctly():
    for t in make_tasks(1000, seed=0):
        assert not re.search(r"\ba [aeiou]", t.question) and not re.search(r"\ban [^aeiou\W]", t.question), t.question


def test_reward_is_exact_match_after_normalising():
    t = make_tasks(1, seed=0)[0]
    assert reward(t, t.answer) == reward(t, f" {t.answer.upper()}. ") == 1.0
    assert reward(t, t.answer + " and more") == reward(t, "") == 0.0


def test_task_messages_adds_the_answer_as_assistant_turn():
    t = make_tasks(1, seed=0)[0]
    assert task_messages(t) == [{"role": "user", "content": t.question}]
    assert task_messages(t, with_answer=True)[1] == {"role": "assistant", "content": t.answer}


def tiny_model_and_tokenizer():
    torch.manual_seed(0)
    model = TransformerLM(ModelConfig(vocab_size=257, context_length=128, d_model=32, n_layers=2, n_heads=4, d_ff=64)).eval()
    return model, Tokenizer(base_vocab([END]), [], [END])


def naive_greedy(model, tok, messages, steps):
    ids = torch.tensor([encode_chat(tok, messages, generation_prompt=True)[0]])
    for _ in range(steps):
        ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
    return ids[0, -steps:].tolist()


def test_generate_replies_batches_prompts_of_different_lengths_and_keeps_the_order():
    model, tok = tiny_model_and_tokenizer()
    prompts = [task_messages(t) for t in make_tasks(6, seed=0)]
    assert len({len(encode_chat(tok, p, True)[0]) for p in prompts}) > 1  # 長度確實不一樣
    replies = generate_replies(model, tok, prompts, max_new_tokens=5)
    for prompt, reply in zip(prompts, replies):
        assert reply == tok.decode(naive_greedy(model, tok, prompt, 5))


def test_generate_replies_stops_at_the_end_token():
    model, tok = tiny_model_and_tokenizer()
    prompt = task_messages(make_tasks(1, seed=0)[0])
    first, second, third = naive_greedy(model, tok, prompt, 3)
    tok.special_to_id = {END: third}  # 假裝模型第三個寫出來的 token 就是結尾
    assert generate_replies(model, tok, [prompt], max_new_tokens=8) == [tok.decode([first, second])]


def test_generate_replies_can_sample_a_group_for_one_prompt():
    model, tok = tiny_model_and_tokenizer()
    group = generate_replies(model, tok, [task_messages(make_tasks(1, seed=0)[0])] * 4, max_new_tokens=6, temperature=1.0)
    assert len(group) == 4


def test_accuracy_is_the_mean_reward_of_greedy_replies():
    model, tok = tiny_model_and_tokenizer()
    tasks = make_tasks(4, seed=0)
    replies = generate_replies(model, tok, [task_messages(t) for t in tasks], max_new_tokens=4)
    assert accuracy(model, tok, tasks, max_new_tokens=4) == 0.0  # 沒訓練過的模型不可能剛好答對
    graded = [Task(t.kind, t.question, r) for t, r in zip(tasks[:2], replies)] + tasks[2:]
    assert accuracy(model, tok, graded, max_new_tokens=4) >= 0.5  # 前兩題的標準答案換成模型自己的回答，一定對


def smoke_config(tmp_path, **changes) -> SFTConfig:
    # 縮小版：隨機初始化的小模型、每個 byte 一個 token、幾十步，CPU 上幾秒鐘
    cfg = dict(base_path="", out_dir=str(tmp_path / "sft"), model=ModelConfig(vocab_size=257, context_length=256, d_model=64, n_layers=2, n_heads=4, d_ff=128),
               n_train=1000, n_valid=20, batch_size=16, max_steps=60, lr_max=3e-3, lr_min=3e-4, warmup_steps=5,
               log_every=20, eval_every=60, device="cpu")
    return SFTConfig(**(cfg | changes))


def test_smoke_sft_lowers_the_masked_loss_and_writes_the_checkpoint(tmp_path):
    records = train_sft(smoke_config(tmp_path), Tokenizer(base_vocab([END]), [], [END]))
    assert records[-1]["val_loss"] < 2.0  # 沒訓練的模型約 ln 257 = 5.5；回覆只有二十幾種答案，60 步就降到 2 以下
    assert 0.0 <= records[-1]["val_accuracy"] <= 1.0
    run = tmp_path / "sft"
    assert json.loads((run / "log.jsonl").read_text().splitlines()[-1])["step"] == 60
    model = TransformerLM(smoke_config(tmp_path).model)
    model.load_state_dict(torch.load(run / "model.pt"))  # 之後（第 14、15 章後面的集數）要載回來用


def test_unmasked_loss_trains_on_the_question_too(tmp_path):
    # 不套 mask：整串都算 loss，一開始的 loss 平均了很多「猜題目文字」的 token，數字和只算回覆的不一樣
    masked = train_sft(smoke_config(tmp_path, max_steps=1, out_dir=str(tmp_path / "a")), Tokenizer(base_vocab([END]), [], [END]))
    unmasked = train_sft(smoke_config(tmp_path, max_steps=1, out_dir=str(tmp_path / "b"), mask=False), Tokenizer(base_vocab([END]), [], [END]))
    assert masked[0]["loss"] != unmasked[0]["loss"]


# ---- 第 14 章第 4、5 集：偏好學習 ----


def test_bradley_terry_loss_is_log_2_when_both_rewards_are_equal():
    assert abs(bradley_terry_loss(torch.tensor([3.0, -1.0]), torch.tensor([3.0, -1.0])).item() - math.log(2)) < 1e-6


def test_bradley_terry_loss_matches_the_hand_calculation_and_shrinks_as_the_margin_grows():
    # margin 1：-log sigmoid(1) = log(1 + e^-1) = 0.3132616875；平均兩筆（第二筆 margin 0 是 log 2）
    want = (0.3132616875182228 + math.log(2)) / 2
    assert abs(bradley_terry_loss(torch.tensor([2.0, 5.0]), torch.tensor([1.0, 5.0])).item() - want) < 1e-6
    losses = [bradley_terry_loss(torch.tensor([m]), torch.tensor([0.0])).item() for m in [-2.0, 0.0, 2.0, 6.0]]
    assert losses == sorted(losses, reverse=True)


def test_bradley_terry_loss_gradient_raises_the_chosen_reward_and_lowers_the_rejected_one():
    chosen, rejected = torch.tensor([0.0], requires_grad=True), torch.tensor([0.0], requires_grad=True)
    bradley_terry_loss(chosen, rejected).backward()
    assert abs(chosen.grad.item() + 0.5) < 1e-6 and abs(rejected.grad.item() - 0.5) < 1e-6  # margin 0：-sigmoid(0)、+sigmoid(0)


def test_bradley_terry_loss_does_not_overflow_with_huge_margins():
    loss = bradley_terry_loss(torch.tensor([1000.0, -1000.0]), torch.tensor([-1000.0, 1000.0]))
    assert torch.isfinite(loss) and abs(loss.item() - 1000.0) < 1e-3  # 一個贏很多（0）、一個輸很多（2000），平均 1000


def tensor(*values, grad=False):
    return torch.tensor(list(values), requires_grad=grad)


def test_dpo_loss_is_log_2_when_the_policy_has_not_moved_from_the_reference():
    # policy == reference：不管 log 機率是多少，兩個 log 比值都是 0
    assert abs(dpo_loss(tensor(-1.0, -4.0), tensor(-3.0, -2.0), tensor(-1.0, -4.0), tensor(-3.0, -2.0), beta=0.3).item() - math.log(2)) < 1e-6


def test_dpo_loss_is_log_2_when_chosen_and_rejected_are_the_same_reply():
    same = tensor(-2.5, -0.5)
    assert abs(dpo_loss(same, same, tensor(-3.0, -1.0), tensor(-3.0, -1.0), beta=0.1).item() - math.log(2)) < 1e-6
    assert abs(dpo_loss(tensor(-1.0), tensor(-3.0), tensor(-2.0), tensor(-2.0), beta=0.0).item() - math.log(2)) < 1e-6  # beta = 0 也是 log 2


def test_dpo_loss_matches_the_hand_calculation():
    # policy 比 reference：chosen 多 1、rejected 少 1，差 2；beta 0.5 -> sigmoid 裡是 1；loss = log(1 + e^-1)
    assert abs(dpo_loss(tensor(-1.0), tensor(-3.0), tensor(-2.0), tensor(-2.0), beta=0.5).item() - 0.3132616875182228) < 1e-6


def test_dpo_loss_gradient_raises_chosen_and_lowers_rejected_with_size_beta_over_2():
    pc, pr = tensor(-2.0, grad=True), tensor(-2.0, grad=True)
    dpo_loss(pc, pr, tensor(-2.0), tensor(-2.0), beta=0.5).backward()
    assert abs(pc.grad.item() + 0.25) < 1e-6 and abs(pr.grad.item() - 0.25) < 1e-6  # -beta/2、+beta/2


def test_dpo_loss_is_higher_when_the_reference_already_liked_the_chosen_reply_more():
    # 同樣的 policy，reference 本來就更偏好 chosen -> policy 相對 reference 的進步比較少 -> loss 比較高
    policy = (tensor(-1.0), tensor(-3.0))
    fresh = dpo_loss(*policy, tensor(-1.0), tensor(-1.0), beta=0.5)
    already = dpo_loss(*policy, tensor(-1.0), tensor(-3.0), beta=0.5)
    assert already > fresh


def test_wrong_answers_are_other_candidates_from_the_question():
    for task in make_tasks(300, seed=4, kinds=("color", "who")):
        wrong = wrong_answers(task)
        assert wrong and task.answer not in wrong
        assert all(re.search(rf"\b{w}\b", task.question) for w in wrong), task.question  # 候選都在題目裡
        assert len(wrong) == (1 if task.kind == "color" else 2)
    assert set(wrong_answers(make_tasks(1, seed=0, kinds=("add",))[0])) == set(WORDS) - {make_tasks(1, seed=0, kinds=("add",))[0].answer}


def test_make_preferences_pairs_the_correct_answer_with_a_wrong_one_deterministically():
    pairs = make_preferences(300, seed=2)
    assert pairs == make_preferences(300, seed=2) and pairs != make_preferences(300, seed=3)
    for pair, task in zip(pairs, make_tasks(300, seed=2)):
        assert (pair.kind, pair.question, pair.chosen) == (task.kind, task.question, task.answer)
        assert pair.rejected in wrong_answers(task)
    who = [p for p in pairs if p.kind == "who"]
    assert len({p.rejected for p in who}) > 5  # 兩個錯的名字都會被挑到，不是永遠同一個位置


def test_make_preferences_train_and_valid_never_overlap_and_kinds_can_be_limited():
    train = {p.question for p in make_preferences(2000, seed=0, split="train")}
    valid = {p.question for p in make_preferences(500, seed=0, split="valid")}
    assert valid and not train & valid
    assert {p.kind for p in make_preferences(100, seed=0, kinds=("color", "who"))} == {"color", "who"}


def test_pair_batch_puts_chosen_rows_first_and_trains_only_the_reply():
    pairs = [Pair("color", "Q one?", "red", "blue"), Pair("color", "Q two longer?", "green", "pink")]
    ids, mask = pair_batch(TOK, pairs)
    assert ids.shape[0] == 4
    for row, reply in enumerate(["red", "green", "blue", "pink"]):  # 前兩列 chosen、後兩列 rejected
        assert TOK.decode([i for i, m in zip(ids[row].tolist(), mask[row].tolist()) if m][:-1]) == " " + reply


def test_pair_logprobs_matches_scoring_each_reply_separately():
    model, tok = tiny_model_and_tokenizer()
    pairs = make_preferences(5, seed=0)  # 題目長度不同 -> 要補 0，補的部分不能影響答案
    chosen, rejected = pair_logprobs(model, *pair_batch(tok, pairs))
    assert chosen.shape == rejected.shape == (5,)
    for i, pair in enumerate(pairs):
        for got, reply in [(chosen[i], pair.chosen), (rejected[i], pair.rejected)]:
            ids, mask = encode_chat(tok, task_messages(Task(pair.kind, pair.question, reply), with_answer=True))
            want = sum(torch.log_softmax(model(torch.tensor([ids]))[0, t - 1], -1)[ids[t]] for t in range(len(ids)) if mask[t])
            assert abs(got.item() - want.item()) < 1e-3


def test_preference_rate_is_the_fraction_of_pairs_where_chosen_has_the_higher_logprob():
    model, tok = tiny_model_and_tokenizer()
    pairs = make_preferences(12, seed=1)
    with torch.no_grad():
        chosen, rejected = pair_logprobs(model, *pair_batch(tok, pairs))
    want = (chosen > rejected).sum().item() / len(pairs)
    assert preference_rate(model, tok, pairs, chunk=5) == want  # 分批算（5、5、2）答案不變
    assert preference_rate(model, tok, [Pair(p.kind, p.question, p.chosen, p.chosen) for p in pairs]) == 0.0  # 兩邊相同：平手不算贏


def dpo_smoke_config(tmp_path, **changes) -> DPOConfig:
    cfg = dict(sft_path="", out_dir=str(tmp_path / "dpo"), model=ModelConfig(vocab_size=257, context_length=256, d_model=64, n_layers=2, n_heads=4, d_ff=128),
               n_train=16, n_valid=20, batch_size=16, max_steps=40, lr_max=1e-3, lr_min=1e-4, warmup_steps=2, beta=0.1,
               log_every=1, eval_every=40, device="cpu")
    return DPOConfig(**(cfg | changes))


def test_smoke_dpo_starts_at_log_2_then_learns_the_pairs(tmp_path):
    records = train_dpo(dpo_smoke_config(tmp_path), Tokenizer(base_vocab([END]), [], [END]))
    assert abs(records[0]["loss"] - math.log(2)) < 1e-4  # 第 1 步：policy 就是 reference
    assert records[-1]["loss"] < 0.3  # 16 題反覆看 40 步，背得起來
    last = records[-1]
    assert last["chosen_gain"] > 0 > last["rejected_gain"]  # chosen 的 log 機率比 reference 高、rejected 比較低
    assert last["train_preference"] > records[0]["train_preference"] and 0.0 <= last["val_preference"] <= 1.0
    model = TransformerLM(dpo_smoke_config(tmp_path).model)
    model.load_state_dict(torch.load(tmp_path / "dpo" / "model.pt"))


def test_dpo_starts_from_the_sft_checkpoint(tmp_path):
    torch.manual_seed(7)
    cfg = dpo_smoke_config(tmp_path, lr_max=1e-12, lr_min=1e-13, max_steps=3)
    start = TransformerLM(cfg.model).state_dict()
    torch.save(start, tmp_path / "sft.pt")
    train_dpo(replace(cfg, sft_path=str(tmp_path / "sft.pt")), Tokenizer(base_vocab([END]), [], [END]))
    end = torch.load(tmp_path / "dpo" / "model.pt")
    assert all(torch.allclose(start[k], end[k], atol=1e-8) for k in start)  # 學習率小到幾乎沒動：載入的是我們給的檢查點，不是另一組隨機初始化
