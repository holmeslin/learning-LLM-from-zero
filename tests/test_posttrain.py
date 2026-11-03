import json
import math

import torch
from torch import nn

from llmz.model import ModelConfig, TransformerLM
import re

from llmz.posttrain import KINDS, SFTConfig, Task, accuracy, chat_pieces, encode_chat, generate_replies, make_tasks, pad_batch, reward, sft_loss, task_messages, train_sft
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
