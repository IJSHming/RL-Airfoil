"""PPO (Proximal Policy Optimization) 强化学习智能体：
用于对黑盒优化器进行 Meta-Optimization (元优化) 控制。
包含 Actor-Critic 网络架构、GAE 优势估计、PPO 截断损失与自适应训练更新。
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    """正交初始化神经网络权重"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """PPO 策略与价值双头 MLP 网络"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Critic (价值网络)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        # Actor (策略网络)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        # 可学习动作对数标准差
        self.actor_logstd = nn.Parameter(torch.full((action_dim,), -0.5, dtype=torch.float32))

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state)

    def get_action_and_value(
        self, state: torch.Tensor, action: torch.Tensor = None, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean = torch.tanh(self.actor_mean(state))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if deterministic:
            action = action_mean
        elif action is None:
            action = probs.sample()

        action_clipped = torch.clamp(action, -1.0, 1.0)
        log_prob = probs.log_prob(action_clipped).sum(-1)
        entropy = probs.entropy().sum(-1)
        value = self.critic(state)
        return action_clipped, log_prob, entropy, value


class PPOAgent:
    """PPO 训练与推理封装器"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.20,
        ent_coef: float = 0.01,
        vf_coef: float = 0.50,
        max_grad_norm: float = 0.50,
        update_epochs: int = 8,
        minibatch_size: int = 64,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.device = torch.device(device)

        self.network = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """根据当前状态输出连续动作向量"""
        self.network.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _, _, _ = self.network.get_action_and_value(state_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def update(self, rollouts: List[Dict[str, Any]]) -> Dict[str, float]:
        """使用收集到的 rollout 经验进行 PPO 策略与价值网络更新"""
        if len(rollouts) == 0:
            return {}

        self.network.train()

        states = torch.tensor(np.array([t["state"] for t in rollouts], dtype=np.float32), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([t["action"] for t in rollouts], dtype=np.float32), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array([float(t["reward"]) for t in rollouts], dtype=np.float32), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([t["next_state"] for t in rollouts], dtype=np.float32), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array([float(bool(t["done"])) for t in rollouts], dtype=np.float32), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            values = self.network.get_value(states).squeeze(-1)
            next_values = self.network.get_value(next_states).squeeze(-1)

            # GAE (Generalized Advantage Estimation) 计算
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(len(rewards))):
                nextnonterminal = 1.0 - dones[t]
                nextval = next_values[t]
                delta = rewards[t] + self.gamma * nextval * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

            _, old_log_probs, _, _ = self.network.get_action_and_value(states, actions)

        # 归一化优势项
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO 多 Epoch Mini-batch 更新
        dataset_size = len(states)
        batch_size = min(self.minibatch_size, dataset_size)

        total_pg_loss, total_v_loss, total_ent = 0.0, 0.0, 0.0
        n_updates = 0

        for _ in range(self.update_epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    states[mb_idx], actions[mb_idx]
                )
                log_ratio = new_log_probs - old_log_probs[mb_idx]
                ratio = torch.exp(log_ratio)

                mb_advantages = advantages[mb_idx]

                # Policy Clipped Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                v_loss = 0.5 * ((new_values.squeeze(-1) - returns[mb_idx]) ** 2).mean()

                # Entropy Loss
                entropy_loss = entropy.mean()

                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_ent += entropy_loss.item()
                n_updates += 1

        return {
            "policy_loss": total_pg_loss / max(n_updates, 1),
            "value_loss": total_v_loss / max(n_updates, 1),
            "entropy": total_ent / max(n_updates, 1),
        }

    def save(self, path: str):
        """保存模型权重"""
        torch.save({
            "network": self.network.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }, path)

    def load(self, path: str):
        """加载模型权重"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
