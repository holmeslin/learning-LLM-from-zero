import json
import math

import pytest
import torch
from torch import nn

import llmz.rl
from llmz.functional import softmax
from llmz.inference import generate_cached
from llmz.model import ModelConfig, TransformerLM
from llmz.posttrain import Task
from llmz.rl import (
    RLVRConfig,
    bandit_rewards,
    estimate_gradient,
    expected_reward,
    group_advantages,
    grpo_loss,
    importance_ratio,
    reinforce_loss,
    rollout,
    sample_actions,
    sample_batch,
    token_logprobs,
    train_bandit,
    train_groups,
    train_rlvr,
    verifiable_rewards,
)
from llmz.tokenizer import Tokenizer, base_vocab


def test_expected_reward_is_probability_weighted_mean():
    logits = torch.zeros(2)  # 兩個動作機率各 0.5
    assert math.isclose(expected_reward(logits, torch.tensor([1.0, 3.0])).item(), 2.0, rel_tol=1e-6)


def test_sample_actions_follow_the_policy():
    logits = torch.log(torch.tensor([0.7, 0.2, 0.1]))
    actions = sample_actions(logits, 20000, torch.Generator().manual_seed(0))
    freq = torch.bincount(actions, minlength=3) / 20000
    assert torch.allclose(freq, softmax(logits), atol=0.02)


def test_bandit_rewards_mean_and_noise():
    means = torch.tensor([0.0, 5.0])
    actions = torch.ones(20000, dtype=torch.long)
    rewards = bandit_rewards(means, actions, noise_std=2.0, generator=torch.Generator().manual_seed(0))
    assert abs(rewards.mean().item() - 5.0) < 0.1
    assert abs(rewards.std().item() - 2.0) < 0.1
    assert torch.equal(bandit_rewards(means, torch.tensor([0, 1, 1]), noise_std=0.0), torch.tensor([0.0, 5.0, 5.0]))


def test_reinforce_loss_gradient_pushes_the_chosen_action_up():
    # 手算：logits 全 0 時 pi = 1/3；選了動作 1、advantage = 2，
    # loss = -A * log pi(1)，對 logits 的梯度是 -A * (one_hot(1) - pi) = [2/3, -4/3, 2/3]
    logits = torch.zeros(1, 3, requires_grad=True)
    reinforce_loss(logits, torch.tensor([1]), torch.tensor([2.0])).backward()
    assert torch.allclose(logits.grad, torch.tensor([[2 / 3, -4 / 3, 2 / 3]]), atol=1e-6)


def test_reinforce_loss_negative_advantage_pushes_it_down_and_averages_samples():
    logits = torch.zeros(2, 3, requires_grad=True)
    reinforce_loss(logits, torch.tensor([0, 2]), torch.tensor([-1.0, 0.0])).backward()
    # 第二個樣本 advantage 是 0，沒有梯度；第一個被平均除以 2：-(−1) * (one_hot(0) - 1/3) / 2
    assert torch.allclose(logits.grad[0], torch.tensor([1 / 3, -1 / 6, -1 / 6]), atol=1e-6)
    assert torch.equal(logits.grad[1], torch.zeros(3))


MEANS = torch.tensor([0.2, 0.5, 0.35, 0.8, 0.6])


def exact_gradient(logits: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    logits = logits.clone().requires_grad_()
    expected_reward(logits, means).backward()  # 已知平均獎勵時，J 可以直接對 logits 微分
    return logits.grad


def test_estimate_gradient_matches_exact_gradient_with_many_samples():
    logits = torch.tensor([0.5, -0.3, 0.0, 0.2, -0.1])
    want = exact_gradient(logits, MEANS)
    for baseline in [False, True]:  # baseline 只降低變異，不改變期望值
        got = estimate_gradient(logits, MEANS, 200000, baseline=baseline, generator=torch.Generator().manual_seed(0))
        assert torch.allclose(got, want, atol=0.01)


def test_baseline_cuts_variance_when_rewards_have_a_large_offset():
    logits, means = torch.zeros(5), MEANS + 10.0  # 獎勵全是正的，而且很大
    variance = {}
    for baseline in [False, True]:
        g = torch.Generator().manual_seed(2)
        runs = torch.stack([estimate_gradient(logits, means, 8, baseline=baseline, noise_std=0.5, generator=g) for _ in range(500)])
        variance[baseline] = runs.var(0).sum().item()
    assert variance[True] < variance[False] / 20


def test_train_bandit_converges_to_the_best_arm():
    for baseline in [False, True]:
        logits = train_bandit(MEANS, steps=300, n=16, lr=1.0, baseline=baseline, noise_std=0.3, seed=0)
        assert logits.argmax().item() == 3  # 平均獎勵最高的動作
        assert softmax(logits)[3] > 0.9


def test_group_advantages_have_zero_mean_per_group():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0], [2.0, 4.0, 6.0, 8.0], [0.0, 0.0, 1.0, 1.0]])
    for normalize_std in [False, True]:
        assert torch.allclose(group_advantages(rewards, normalize_std).mean(dim=1), torch.zeros(3), atol=1e-6)


def test_group_advantages_with_std_have_unit_std_per_group():
    torch.manual_seed(0)
    rewards = torch.randn(6, 8) * torch.tensor([0.1, 1.0, 5.0, 20.0, 0.5, 3.0])[:, None] + 7  # 每組的尺度與平均都不同
    adv = group_advantages(rewards, normalize_std=True)
    assert torch.allclose(adv.std(dim=1), torch.ones(6), atol=1e-4)


def test_group_advantages_hand_computed():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # 平均 0.25；樣本標準差（除以 G-1）是 0.5
    assert torch.allclose(group_advantages(rewards, normalize_std=False), torch.tensor([[0.75, -0.25, -0.25, -0.25]]))
    assert torch.allclose(group_advantages(rewards, normalize_std=True), torch.tensor([[1.5, -0.5, -0.5, -0.5]]), atol=1e-4)


def test_group_advantages_of_identical_rewards_are_zero():
    rewards = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [3.5, 3.5, 3.5]])  # 全對、全錯：標準差 0，不能變成 nan 或很大的數
    for normalize_std in [False, True]:
        assert torch.equal(group_advantages(rewards, normalize_std), torch.zeros(3, 3))


def test_verifiable_rewards_are_one_for_the_correct_answer():
    actions = torch.tensor([[2, 0, 2], [1, 1, 3]])  # 兩個 prompt，每個 3 個回答
    assert torch.equal(verifiable_rewards(torch.tensor([2, 1]), actions), torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]))


def test_train_groups_learns_the_correct_answers():
    correct = torch.tensor([3, 0, 7])
    for normalize_std in [False, True]:
        logits = train_groups(torch.zeros(3, 10), correct, group_size=8, steps=300, lr=0.5, normalize_std=normalize_std)
        assert torch.equal(logits.argmax(dim=1), correct)
        assert softmax(logits)[torch.arange(3), correct].min() > 0.9


# ---- 第 15 章第 4 集：token 層級的 clipped loss ----

P = torch.tensor([[0.1, 0.6, 0.3], [0.2, 0.2, 0.6], [0.5, 0.25, 0.25]])  # 三個符號的 bigram「模型」：P[a][b] 是 a 的下一個是 b 的機率


class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(P.log())

    def forward(self, ids):
        return self.logits[ids]


def test_token_logprobs_are_the_log_probability_of_each_next_token():
    got = token_logprobs(BigramLM(), torch.tensor([[0, 1, 2, 0], [2, 2, 1, 1]]))
    want = torch.tensor([[P[0, 1], P[1, 2], P[2, 0]], [P[2, 2], P[2, 1], P[1, 1]]]).log()  # 位置 t 預測 token t+1，所以少一格
    assert got.shape == (2, 3)
    assert torch.allclose(got, want, atol=1e-6)


def clipped_grad(ratio: float, advantage: float, eps: float = 0.2):
    """單一 token：old_logp = 0，所以 ratio = exp(logp)。回傳 (loss, 對 logp 的梯度)。"""
    logp = torch.tensor([[math.log(ratio)]], requires_grad=True)
    loss = grpo_loss(logp, torch.zeros(1, 1), torch.tensor([advantage]), torch.ones(1, 1, dtype=torch.bool), clip_eps=eps)
    loss.backward()
    return loss.item(), logp.grad.item()


def test_gradient_is_zero_when_the_ratio_is_outside_the_clip_range_and_the_clip_is_the_smaller_term():
    # advantage 為正、ratio 已經高過 1+eps：再推高沒有好處，被 clip 住；advantage 為負、ratio 已經低過 1-eps：同理
    for ratio, advantage, want_loss in [(1.5, +1.0, -1.2), (0.5, -1.0, +0.8), (3.0, +2.0, -2.4), (0.1, -3.0, +2.4)]:
        loss, grad = clipped_grad(ratio, advantage)
        assert grad == 0.0
        assert math.isclose(loss, want_loss, rel_tol=1e-5)  # loss 是 -(clip 之後的 ratio * A)


def test_gradient_flows_when_the_ratio_moved_the_wrong_way():
    # advantage 為正、ratio 掉到 1-eps 以下：min 取沒有 clip 的那項，要把它拉回來；advantage 為負、ratio 衝到 1+eps 以上同理
    for ratio, advantage in [(0.5, +1.0), (1.5, -1.0)]:
        loss, grad = clipped_grad(ratio, advantage)
        assert math.isclose(loss, -ratio * advantage, rel_tol=1e-5)
        assert math.isclose(grad, -ratio * advantage, rel_tol=1e-5)  # d(-ρA)/d logp = -ρA，因為 dρ/d logp = ρ


def test_gradient_inside_the_clip_range_is_the_plain_policy_gradient():
    loss, grad = clipped_grad(1.1, 2.0)
    assert math.isclose(loss, -2.2, rel_tol=1e-5)
    assert math.isclose(grad, -2.2, rel_tol=1e-5)
    _, grad = clipped_grad(1.0, 1.0)  # ratio = 1（剛 rollout 完、還沒更新）：梯度就是 -A，和 REINFORCE 一樣
    assert math.isclose(grad, -1.0, rel_tol=1e-6)


def test_clip_range_follows_eps():
    assert clipped_grad(1.5, 1.0, eps=0.2)[1] == 0.0
    assert clipped_grad(1.5, 1.0, eps=0.6)[1] != 0.0  # 範圍放寬到 [0.4, 1.6]，1.5 就在裡面


MASK2 = torch.tensor([[True, True], [True, False]])  # 第一條序列 2 個 token、第二條 1 個（第二個位置是補的或結尾之後）


def test_length_norm_averages_tokens_within_each_sequence_first():
    logp = torch.zeros(2, 2, requires_grad=True)
    advantages = torch.tensor([1.0, 0.5])
    loss = grpo_loss(logp, torch.zeros(2, 2), advantages, MASK2, length_norm=True)
    loss.backward()
    assert math.isclose(loss.item(), -(1.0 + 0.5) / 2, rel_tol=1e-6)  # 每條序列先對自己的 token 平均，再對序列平均
    assert torch.allclose(logp.grad, torch.tensor([[-0.25, -0.25], [-0.25, 0.0]]))  # 長的序列每個 token 的權重比較小


def test_without_length_norm_every_token_counts_the_same():
    logp = torch.zeros(2, 2, requires_grad=True)
    loss = grpo_loss(logp, torch.zeros(2, 2), torch.tensor([1.0, 0.5]), MASK2, length_norm=False)
    loss.backward()
    assert math.isclose(loss.item(), -(2 * 1.0 + 0.5) / 2, rel_tol=1e-6)  # token 數乘上 advantage，再對序列平均
    assert torch.allclose(logp.grad, torch.tensor([[-0.5, -0.5], [-0.25, 0.0]]))


def test_masked_positions_have_zero_gradient_even_if_their_ratio_would_overflow():
    logp = torch.tensor([[0.0, 100.0]], requires_grad=True)  # 被遮掉的位置 exp(100) 在 float32 是 inf；inf * 0 不能變成 nan
    loss = grpo_loss(logp, torch.zeros(1, 2), torch.tensor([1.0]), torch.tensor([[True, False]]))
    loss.backward()
    assert math.isclose(loss.item(), -1.0, rel_tol=1e-6)
    assert torch.equal(logp.grad, torch.tensor([[-1.0, 0.0]]))


def test_a_group_with_zero_advantage_gives_zero_gradient():
    logp = torch.randn(3, 4, requires_grad=True)
    grpo_loss(logp, logp.detach() + 0.05, torch.zeros(3), torch.ones(3, 4, dtype=torch.bool)).backward()
    assert torch.equal(logp.grad, torch.zeros(3, 4))


def test_ratio_one_loss_is_the_negative_mean_advantage_when_every_sequence_has_tokens():
    logp = torch.randn(4, 3)
    advantages = torch.tensor([1.0, -2.0, 0.5, 3.0])
    loss = grpo_loss(logp, logp.clone(), advantages, torch.ones(4, 3, dtype=torch.bool))
    assert math.isclose(loss.item(), -advantages.mean().item(), rel_tol=1e-6)


# ---- 第 15 章第 5 集：rollout 與 RLVR ----

END = "<|endoftext|>"
TOK = Tokenizer(base_vocab([END]), [], [END])  # 一個 byte 一個 token，再加上 <|endoftext|>
TINY = ModelConfig(vocab_size=257, context_length=128, d_model=32, n_layers=1, n_heads=2, d_ff=64)
SHORT = Task("add", "2 + 3?", "five")
LONG = Task("add", "ten minus four plus one equals?", "seven")  # 提示的長度和 SHORT 不一樣：rollout 要能把不同長度的提示一起處理


def scripted_generate(model, ids, max_new_tokens, temperature=1.0):
    """假的 sample_batch：同一個 prompt 的第 r 個回答是 r 決定的：第 0 個答 five 並寫結尾（後面還接著亂碼），第 1 個答 nine 並寫結尾，
    LONG 題的每個回答都寫滿 max_new_tokens 個 token、不寫結尾。"""
    rows = []
    for r, row in enumerate(ids.tolist()):
        text = TOK.decode(row)
        if "2 + 3" in text:
            new = TOK.encode([" five", " nine"][r % 2]) + [TOK.special_to_id[END]] + TOK.encode("xyz")
        else:
            new = TOK.encode(" abcdefghij")
        rows.append(row + (new + TOK.encode("q" * 8))[:max_new_tokens])
    return torch.tensor(rows)


def test_rollout_keeps_ids_masks_the_response_and_scores_each_reply(monkeypatch):
    monkeypatch.setattr(llmz.rl, "sample_batch", scripted_generate)
    out = rollout(tiny_model(), TOK, [LONG, SHORT], group_size=2, max_new_tokens=8)
    assert out.ids.shape == out.mask.shape == (4, len("User: ten minus four plus one equals?\nAssistant:") + 8)  # 補齊到最長的提示
    assert torch.equal(out.rewards, torch.tensor([[0.0, 0.0], [1.0, 0.0]]))  # (題數, 每題的回答數)，順序和傳進來的 tasks 一樣
    prompt_len = [len(TOK.encode(f"User: {t.question}\nAssistant:")) for t in (LONG, LONG, SHORT, SHORT)]
    for row, n in enumerate(prompt_len):
        assert not out.mask[row, :n].any()  # 提示不算
    long_row, five_row, nine_row = out.mask[0], out.mask[2], out.mask[3]
    assert long_row.sum() == 8  # 沒寫結尾：整段都算
    assert five_row.sum() == len(" five") + 1  # " five" 加上結尾；結尾之後的亂碼不算
    assert nine_row.sum() == len(" nine") + 1
    end_at = prompt_len[2] + len(" five")
    assert out.ids[2, end_at] == TOK.special_to_id[END] and five_row[end_at] and not five_row[end_at + 1]
    assert TOK.decode(out.ids[2, prompt_len[2] : end_at].tolist()) == " five"
    assert not out.mask[2, end_at + 1 :].any()  # 補齊的地方也是 False


def test_rollout_scores_replies_with_the_reward_function_it_is_given(monkeypatch):
    monkeypatch.setattr(llmz.rl, "sample_batch", scripted_generate)
    out = rollout(tiny_model(), TOK, [SHORT], group_size=2, max_new_tokens=8, reward_fn=lambda task, reply: float(len(reply)))
    assert torch.equal(out.rewards, torch.tensor([[5.0, 5.0]]))  # " five"、" nine" 各 5 個字元；預設的 reward 會給 [1, 0]


def tiny_model():
    torch.manual_seed(0)
    return TransformerLM(TINY)


def test_sample_batch_at_a_tiny_temperature_is_greedy_decoding_from_chapter_11():
    model = tiny_model()
    ids = torch.tensor([TOK.encode("User: 2 + 3?\nAssistant:")] * 3)
    assert torch.equal(sample_batch(model, ids, 6, temperature=1e-4), generate_cached(model, ids, 6, temperature=0.0))


def test_sample_batch_draws_each_row_independently_from_softmax():
    model = tiny_model()
    ids = torch.tensor([TOK.encode("User: 2 + 3?\nAssistant:")] * 4000)
    first = sample_batch(model, ids, 1)[:, -1]
    probs = softmax(model(ids[:1])[0, -1])  # 4000 列都是同一個提示，第一個抽到的 token 的頻率要接近 softmax
    freq = torch.bincount(first, minlength=257) / 4000
    assert (first != first[0]).any() and (freq - probs).abs().max() < 0.03


def test_rollout_samples_with_a_real_model_and_the_logprobs_line_up():
    model = tiny_model()
    out = rollout(model, TOK, [SHORT, LONG], group_size=3, max_new_tokens=5, temperature=1.0)
    assert out.ids.shape[0] == 6 and out.rewards.shape == (2, 3)
    logp = token_logprobs(model, out.ids)  # (N, T-1)：和 mask[:, 1:] 對齊
    assert logp.shape == out.mask[:, 1:].shape
    assert (logp[out.mask[:, 1:]] < 0).all()


def smoke_config(tmp_path, **changes) -> RLVRConfig:
    cfg = dict(sft_path="", out_dir=str(tmp_path / "rlvr"), model=TINY, prompts_per_step=4, group_size=4, max_new_tokens=4,
               max_steps=3, epochs=2, lr=1e-3, eval_every=3, n_valid=8, device="cpu")
    return RLVRConfig(**(cfg | changes))


def test_smoke_rlvr_runs_updates_the_policy_and_writes_the_checkpoint(tmp_path):
    cfg = smoke_config(tmp_path)
    records = train_rlvr(cfg, TOK)
    assert [r["step"] for r in records] == [0, 1, 2, 3]  # 第 0 步是還沒訓練的驗證正確率
    assert all(0.0 <= r["reward"] <= 1.0 and 0.0 <= r["clip_frac"] <= 1.0 and math.isfinite(r["loss"]) for r in records[1:])
    assert 0.0 <= records[-1]["val_accuracy"] <= 1.0
    run = tmp_path / "rlvr"
    assert json.loads((run / "log.jsonl").read_text().splitlines()[-1])["step"] == 3
    model = TransformerLM(TINY)
    model.load_state_dict(torch.load(run / "model.pt"))


def test_first_update_of_a_rollout_is_on_policy_so_nothing_is_clipped(tmp_path):
    records = train_rlvr(smoke_config(tmp_path, epochs=1, max_steps=2), TOK)
    assert all(r["clip_frac"] == 0.0 for r in records[1:])  # 只更新一次：ratio 恆為 1


def test_rlvr_raises_the_reward_it_is_given(tmp_path):
    # 隨機初始化的模型答不對任何一題（獎勵全是 0，沒有訊號）。換一個一開始約一半會拿到的 0/1 獎勵：回覆的第一個字元是 ASCII
    ascii_first = lambda task, reply: float(reply[:1] != "" and ord(reply[0]) < 128)
    cfg = smoke_config(tmp_path, prompts_per_step=4, group_size=8, max_steps=10, epochs=2, lr=1e-2, eval_every=10, max_new_tokens=2)
    records = train_rlvr(cfg, TOK, reward_fn=ascii_first)[1:]
    first, last = sum(r["reward"] for r in records[:3]) / 3, sum(r["reward"] for r in records[-3:]) / 3
    assert 0.3 < first < 0.7 and last > first + 0.2  # 種子 0～4 實測：開頭 0.49～0.59，結尾 0.85～0.93


# ---- 第 15 章第 6 集：序列層級的比值（GSPO）與固定的長度除數（Dr. GRPO） ----


def loss_and_grad(logp_values, old_values, advantages, mask=None, **kwargs):
    logp = torch.tensor(logp_values, requires_grad=True)
    mask = torch.ones_like(logp, dtype=torch.bool) if mask is None else mask
    loss = grpo_loss(logp, torch.tensor(old_values), torch.tensor(advantages), mask, **kwargs)
    loss.backward()
    return loss.item(), logp.grad


def test_sequence_ratio_of_one_token_answers_equals_the_token_ratio():
    # 長度為 1：序列比值 = exp(那一個 token 的 log 比值)，和 token 層級完全一樣，連梯度（含 clip）都一樣
    logp, old = [[0.4], [-0.9], [0.1], [-0.3]], [[0.0], [0.0], [0.0], [0.0]]
    advantages = [1.0, 1.0, -2.0, -0.5]  # 比值 1.49、0.41、1.11、0.74：clip 的每一種情況都有
    token = loss_and_grad(logp, old, advantages, seq_ratio=False)
    seq = loss_and_grad(logp, old, advantages, seq_ratio=True)
    assert math.isclose(token[0], seq[0], rel_tol=1e-6)
    assert torch.allclose(token[1], seq[1], atol=1e-7)


def test_sequence_ratio_is_the_geometric_mean_of_the_token_ratios():
    # 兩個 token 的比值是 e^0.1 與 e^-0.1（都在 clip 範圍內）：幾何平均是 1，所以 loss = -A * 1，兩個 token 的梯度一樣大（-0.5）
    loss, grad = loss_and_grad([[0.1, -0.1]], [[0.0, 0.0]], [1.0], seq_ratio=True)
    assert math.isclose(loss, -1.0, rel_tol=1e-6)
    assert torch.allclose(grad, torch.tensor([[-0.5, -0.5]]), atol=1e-6)
    # token 層級：每個 token 的梯度是 -A * rho / 2，比值大的 token 被推得比較用力
    _, grad = loss_and_grad([[0.1, -0.1]], [[0.0, 0.0]], [1.0], seq_ratio=False)
    assert torch.allclose(grad, torch.tensor([[-0.5 * math.exp(0.1), -0.5 * math.exp(-0.1)]]), atol=1e-6)


def test_sequence_ratio_blocks_or_passes_the_whole_answer_together():
    # A = +1、第一個 token 的比值 e^0.5（超過 1.2）、第二個是 e^-0.1
    # token 層級：只有第一個 token 被擋住（梯度 0），第二個照常。序列比值 e^0.2 = 1.221 > 1.2：整條一起被擋住
    _, token = loss_and_grad([[0.5, -0.1]], [[0.0, 0.0]], [1.0], seq_ratio=False)
    _, seq = loss_and_grad([[0.5, -0.1]], [[0.0, 0.0]], [1.0], seq_ratio=True)
    assert token[0, 0] == 0.0 and token[0, 1] != 0.0
    assert torch.equal(seq, torch.zeros(1, 2))
    # 反過來：第一個 token e^0.3 超過 1.2，序列比值 e^0.15 = 1.162 在範圍內：token 層級擋住一個，序列層級兩個都有梯度
    _, token = loss_and_grad([[0.3, 0.0]], [[0.0, 0.0]], [1.0], seq_ratio=False)
    _, seq = loss_and_grad([[0.3, 0.0]], [[0.0, 0.0]], [1.0], seq_ratio=True)
    assert token[0, 0] == 0.0
    assert torch.allclose(seq, torch.full((1, 2), -0.5 * math.exp(0.15)), atol=1e-6)


def test_sequence_ratio_ignores_padding_positions():
    # 第二條只有一個 token；被遮掉的位置即使 log 比值是 100 也不能影響平均，也不能有梯度或 nan
    mask = torch.tensor([[True, True], [True, False]])
    loss, grad = loss_and_grad([[0.2, -0.2], [0.1, 100.0]], [[0.0, 0.0], [0.0, 0.0]], [1.0, 1.0], mask, seq_ratio=True)
    assert math.isclose(loss, -(1.0 + math.exp(0.1)) / 2, rel_tol=1e-6)
    assert torch.allclose(grad, torch.tensor([[-0.25, -0.25], [-0.5 * math.exp(0.1), 0.0]]), atol=1e-6)


def test_importance_ratio_is_the_same_number_on_every_token_of_an_answer_when_sequence_level():
    logp, old = torch.tensor([[0.3, 0.1, 0.5], [0.0, 0.0, 0.0]]), torch.zeros(2, 3)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    assert torch.allclose(importance_ratio(logp, old, mask), torch.exp(torch.tensor([[0.3, 0.1, 0.0], [0.0, 0.0, 0.0]])))
    seq = importance_ratio(logp, old, mask, seq_ratio=True)
    assert torch.allclose(seq, torch.tensor([[math.exp(0.2)] * 3, [1.0] * 3]), atol=1e-6)


def test_length_divisor_replaces_the_per_answer_length_when_per_answer_averaging_is_off():
    # 兩條回答 2 個與 1 個 token，advantage 1.0 與 0.5；token 加總之後都除以同一個 4，不管各自多長
    loss, grad = loss_and_grad([[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [1.0, 0.5], MASK2, length_norm=False, length_divisor=4.0)
    assert math.isclose(loss, -(2 * 1.0 / 4 + 0.5 / 4) / 2, rel_tol=1e-6)
    assert torch.allclose(grad, torch.tensor([[-0.125, -0.125], [-0.0625, 0.0]]), atol=1e-7)
    _, plain = loss_and_grad([[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [1.0, 0.5], MASK2, length_norm=False)
    assert torch.allclose(grad * 4, plain, atol=1e-7)  # 只差一個常數倍：沒有哪一條回答被特別照顧


def test_length_divisor_does_nothing_while_per_answer_averaging_is_on():
    a = loss_and_grad([[0.1, 0.2], [0.3, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [1.0, -1.0], MASK2, length_norm=True)
    b = loss_and_grad([[0.1, 0.2], [0.3, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [1.0, -1.0], MASK2, length_norm=True, length_divisor=7.0)
    assert a[0] == b[0] and torch.equal(a[1], b[1])


def test_smoke_rlvr_with_sequence_ratio_and_a_fixed_length_divisor(tmp_path):
    cfg = smoke_config(tmp_path, seq_ratio=True, length_norm=False, length_divisor=4.0, normalize_std=False)
    records = train_rlvr(cfg, TOK)
    assert [r["step"] for r in records] == [0, 1, 2, 3]
    assert all(math.isfinite(r["loss"]) and 0.0 <= r["clip_frac"] <= 1.0 for r in records[1:])
    one_epoch = train_rlvr(smoke_config(tmp_path, seq_ratio=True, epochs=1), TOK)
    assert all(r["clip_frac"] == 0.0 for r in one_epoch[1:])  # 只更新一次：序列比值也恆為 1
