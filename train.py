"""RL-Meta-Airfoil 强化学习元优化器训练脚本：
在 10 个雷诺数 (1e6~1e7) 与 11 个攻角 (0~10度) 构成的气动网格上，
对 SPSA、Py-BOBYQA、GA、PSO、CMA-ES、TuRBO 训练各自专属的 PPO Meta-Policy，
并保存各优化器的模型权重。
"""

import argparse
import os
import sys
import time
from typing import List, Tuple
import numpy as np
import torch

from src.evaluator import AerodynamicEvaluator
from src.optimizers import OPTIMIZER_CONFIGS
from src.ppo import PPOAgent


def get_training_grid() -> List[Tuple[float, float]]:
    """生成 10 个雷诺数 x 11 个攻角 = 110 个训练工况点"""
    re_values = np.linspace(1e6, 1e7, 10)
    alpha_values = np.linspace(0.0, 10.0, 11)
    grid = []
    for re in re_values:
        for alpha in alpha_values:
            grid.append((float(re), float(alpha)))
    return grid


def train_single_optimizer(
    opt_name: str,
    n_episodes: int = 100,
    device: str = "cpu",
    output_dir: str = "weights",
):
    print(f"\n=======================================================")
    print(f" 开始训练 Meta-RL 优化器: {opt_name}")
    print(f" 状态维度: {OPTIMIZER_CONFIGS[opt_name]['state_dim']} | 动作维度: {OPTIMIZER_CONFIGS[opt_name]['action_dim']}")
    print(f" 训练 Episode 轮数: {n_episodes} (每 Episode 消耗 300 次 NFE)")
    print(f"=======================================================")

    cfg = OPTIMIZER_CONFIGS[opt_name]
    agent = PPOAgent(
        state_dim=cfg["state_dim"],
        action_dim=cfg["action_dim"],
        lr=3e-4,
        device=device,
    )

    grid = get_training_grid()
    rng = np.random.RandomState(42)
    # 打乱训练工况顺序
    sampled_indices = rng.choice(len(grid), size=n_episodes, replace=(n_episodes > len(grid)))

    start_time = time.time()
    recent_improvements = []

    for ep in range(n_episodes):
        cond_idx = sampled_indices[ep]
        reynolds, alpha = grid[cond_idx]

        # 实例化对应工况的气动评估器
        evaluator = AerodynamicEvaluator(reynolds=reynolds, alpha=alpha)

        # 实例化该优化器并在 Meta-RL 闭环下运行
        opt_class = cfg["class"]
        optimizer = opt_class(
            evaluator=evaluator,
            max_evals=300,
            seed=int(1000 + ep),
            is_meta=True,
            rl_policy=lambda s: agent.select_action(s, deterministic=False),
        )

        res = optimizer.run()
        transitions = optimizer.transitions

        # 使用该轮采集到的经验更新 PPO 策略
        loss_dict = agent.update(transitions)

        gain = optimizer.best_score - optimizer.initial_score
        recent_improvements.append(gain)
        if len(recent_improvements) > 10:
            recent_improvements.pop(0)

        avg_gain_10 = np.mean(recent_improvements)

        if (ep + 1) % 10 == 0 or (ep + 1) == n_episodes or ep == 0:
            elapsed = time.time() - start_time
            print(
                f"[{opt_name:9s}] Ep {ep+1:3d}/{n_episodes} | "
                f"Re: {reynolds/1e6:.1f}M, AoA: {alpha:4.1f}° | "
                f"Init L/D: {optimizer.initial_score:6.2f} -> Best L/D: {optimizer.best_score:6.2f} "
                f"(+{gain:5.2f}, 10-Ep Avg: +{avg_gain_10:5.2f}) | "
                f"Loss(PG: {loss_dict.get('policy_loss', 0.0):.3f}, V: {loss_dict.get('value_loss', 0.0):.2f}) | "
                f"Time: {elapsed:5.1f}s"
            )

    os.makedirs(output_dir, exist_ok=True)
    weight_file = os.path.join(output_dir, f"meta_ppo_{opt_name.lower().replace('-', '_')}.pt")
    agent.save(weight_file)
    print(f">>> {opt_name} Meta-Policy 训练完成，权重保存至: {weight_file}")
    return agent


def main():
    parser = argparse.ArgumentParser(description="RL-Meta-Airfoil PPO Training")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes per optimizer")
    parser.add_argument(
        "--optimizers",
        type=str,
        default="SPSA,Py-BOBYQA,GA,PSO,CMA-ES,TuRBO",
        help="Comma-separated list of optimizers to train",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (cuda/cpu)",
    )
    parser.add_argument("--output_dir", type=str, default="weights", help="Directory to save model weights")
    args = parser.parse_args()

    target_optimizers = [opt.strip() for opt in args.optimizers.split(",") if opt.strip()]

    print(f"RL-Meta-Airfoil PPO Training Initialized")
    print(f"Device: {args.device} | Episodes: {args.episodes}")
    print(f"Target Optimizers: {target_optimizers}")

    total_start = time.time()
    for opt_name in target_optimizers:
        if opt_name not in OPTIMIZER_CONFIGS:
            print(f"Warning: Unknown optimizer '{opt_name}', skipping...")
            continue
        train_single_optimizer(
            opt_name=opt_name,
            n_episodes=args.episodes,
            device=args.device,
            output_dir=args.output_dir,
        )

    print(f"\n=======================================================")
    print(f" 全部优化器 Meta-RL 训练完毕！总耗时: {time.time()-total_start:.1f} 秒")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
