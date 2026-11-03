"""強化學習入門（第 15 章）：用多臂吃角子老虎機（bandit）看 policy gradient、REINFORCE 與 group-relative advantage。

這裡的 policy 是一排 logits 經過 softmax；第 4 集起，同一個 loss 會換成真正的語言模型。
"""

import torch
from torch import Tensor

from llmz.functional import log_softmax, softmax


# ANCHOR: sample_actions
def sample_actions(logits: Tensor, n: int, generator: torch.Generator | None = None) -> Tensor:
    """照 softmax(logits) 的機率抽 n 個動作。logits 形狀 (K,) 回傳 (n,)；形狀 (P, K) 則每一列各抽 n 個，回傳 (P, n)。"""
    return torch.multinomial(softmax(logits), n, replacement=True, generator=generator)
# ANCHOR_END: sample_actions


# ANCHOR: bandit_rewards
def bandit_rewards(means: Tensor, actions: Tensor, noise_std: float = 0.0, generator: torch.Generator | None = None) -> Tensor:
    """每個動作 k 的平均獎勵是 means[k]；拉一次拿到平均值再加上標準差 noise_std 的高斯雜訊。"""
    noise = torch.randn(actions.shape, generator=generator) * noise_std
    return means[actions] + noise
# ANCHOR_END: bandit_rewards


# ANCHOR: expected_reward
def expected_reward(logits: Tensor, means: Tensor) -> Tensor:
    """policy 的目標 J：每個動作的機率乘上它的平均獎勵，加總。"""
    return (softmax(logits) * means).sum(dim=-1)
# ANCHOR_END: expected_reward


# ANCHOR: reinforce_loss
def reinforce_loss(logits: Tensor, actions: Tensor, advantages: Tensor) -> Tensor:
    """logits (N, K)、actions (N,)、advantages (N,) -> 純量。advantage 為正就把選到的動作的機率推高，為負就推低。"""
    logp = log_softmax(logits, dim=-1)[torch.arange(len(actions)), actions]  # 和第 1 章 cross_entropy 一樣，只取選到的那一格
    return -(advantages * logp).mean()
# ANCHOR_END: reinforce_loss


# ANCHOR: estimate_gradient
def estimate_gradient(logits: Tensor, means: Tensor, n: int, baseline: bool = False, noise_std: float = 0.0,
                      generator: torch.Generator | None = None) -> Tensor:
    """REINFORCE 對 J 的梯度估計 (K,)：抽 n 個動作、拿獎勵，用 reinforce_loss 算梯度後取負號（因為 J 要往上爬）。

    baseline=True 時，advantage 是「獎勵減去這一批獎勵的平均」；否則 advantage 就是獎勵本身。
    shortcut: 批次平均把自己也算進去，估計的平均縮小 (n-1)/n 倍；要完全無偏時改用 leave-one-out（第 2 集練習）。
    """
    logits = logits.detach().requires_grad_()
    actions = sample_actions(logits, n, generator)
    rewards = bandit_rewards(means, actions, noise_std, generator)
    advantages = rewards - rewards.mean() if baseline else rewards
    reinforce_loss(logits.expand(n, -1), actions, advantages).backward()
    return -logits.grad
# ANCHOR_END: estimate_gradient


# ANCHOR: train_bandit
def train_bandit(means: Tensor, steps: int, n: int, lr: float, baseline: bool = True, noise_std: float = 0.0, seed: int = 0) -> Tensor:
    """從全 0 的 logits 開始，每一步用 n 個樣本估計梯度、往上爬一步（gradient ascent），回傳最後的 logits。"""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.zeros(len(means))
    for _ in range(steps):
        logits = logits + lr * estimate_gradient(logits, means, n, baseline, noise_std, generator)
    return logits
# ANCHOR_END: train_bandit


# ANCHOR: group_advantages
def group_advantages(rewards: Tensor, normalize_std: bool = True, eps: float = 1e-6) -> Tensor:
    """rewards (P, G)：P 個 prompt、每個 prompt 的一組 G 個回答的獎勵。回傳同樣形狀的 advantage：組內獎勵減掉組內平均，
    normalize_std=True 時再除以組內標準差（樣本標準差，除以 G-1；全組獎勵相同時標準差是 0，分子也是 0，所以 advantage 是 0）。"""
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    return centered / (rewards.std(dim=1, keepdim=True) + eps) if normalize_std else centered
# ANCHOR_END: group_advantages


# ANCHOR: verifiable_rewards
def verifiable_rewards(correct: Tensor, actions: Tensor) -> Tensor:
    """correct (P,)：每個 prompt 的正確答案；actions (P, G)：每個 prompt 的 G 個回答。答對 1，答錯 0。"""
    return (actions == correct[:, None]).float()
# ANCHOR_END: verifiable_rewards


# ANCHOR: train_groups
def train_groups(logits: Tensor, correct: Tensor, group_size: int, steps: int, lr: float,
                 normalize_std: bool = True, seed: int = 0) -> Tensor:
    """logits (P, K)：P 個 prompt 各自的 policy。每一步：每個 prompt 抽一組 group_size 個回答、用 0/1 獎勵算 group advantage、往上爬一步。"""
    generator = torch.Generator().manual_seed(seed)
    P, K = logits.shape
    logits = logits.clone()
    for _ in range(steps):
        logits.requires_grad_()
        actions = sample_actions(logits.detach(), group_size, generator)  # (P, G)
        advantages = group_advantages(verifiable_rewards(correct, actions), normalize_std)
        flat_logits = logits[:, None, :].expand(P, group_size, K).reshape(-1, K)  # 同一個 prompt 的回答共用同一列 logits
        reinforce_loss(flat_logits, actions.reshape(-1), advantages.reshape(-1)).backward()
        logits = (logits - lr * logits.grad).detach()  # loss 是 -J，往下降就是 J 往上爬
    return logits
# ANCHOR_END: train_groups
