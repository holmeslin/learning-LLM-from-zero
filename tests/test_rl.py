import math

import torch

from llmz.functional import softmax
from llmz.rl import (
    bandit_rewards,
    estimate_gradient,
    expected_reward,
    group_advantages,
    reinforce_loss,
    sample_actions,
    train_bandit,
    train_groups,
    verifiable_rewards,
)


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
