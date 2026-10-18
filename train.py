"""RL-Meta-Airfoil 强化学习元优化器高效并行训练脚本：
充分利用 HPC 集群 10 核 CPU 并行采集气动经验 + 集中式 PPO 梯度更新。
采用纯 NumPy 无锁策略推理彻底杜绝多进程 OpenMP/CUDA 死锁。
在 10 个雷诺数 (1e6~1e7) 与 11 个攻角 (0~10度) 构成的 110 个气动网格工况上，
为 SPSA、Py-BOBYQA、DE、PSO、CMA-ES、TuRBO 训练各自专属的 Meta-Policy 神经网络权重。
"""

import argparse
import concurrent.futures
import os
import sys
import time
from typing import Any, Dict, List, Tuple
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


def numpy_policy_forward(s: np.ndarray, w: Dict[str, np.ndarray], deterministic: bool = False) -> np.ndarray:
    """纯 NumPy 无锁连续前向推理，彻底杜绝 OpenMP/CUDA 多进程死锁"""
    s_clean = np.nan_to_num(s, nan=0.0, posinf=5.0, neginf=-5.0)
    h1 = np.tanh(s_clean @ w["actor_mean.0.weight"].T + w["actor_mean.0.bias"])
    h2 = np.tanh(h1 @ w["actor_mean.2.weight"].T + w["actor_mean.2.bias"])
    mean = np.tanh(h2 @ w["actor_mean.4.weight"].T + w["actor_mean.4.bias"])
    if deterministic:
        return np.clip(mean, -1.0, 1.0)
    logstd = np.clip(w["actor_logstd"], -3.0, 0.5)
    std = np.exp(logstd)
    noise = np.random.randn(*mean.shape)
    action = mean + std * noise
    return np.clip(action, -1.0, 1.0)


def worker_rollout(task_tuple: Tuple[str, float, float, int, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    """单核 CPU 独立工作进程：执行 300 步黑盒气动优化，采集并返回转移经验。"""
    opt_name, reynolds, alpha, seed, weights = task_tuple

    evaluator = AerodynamicEvaluator(reynolds=reynolds, alpha=alpha)
    cfg = OPTIMIZER_CONFIGS[opt_name]
    opt_class = cfg["class"]

    def rl_action_fn(s: np.ndarray) -> np.ndarray:
        return numpy_policy_forward(s, weights, deterministic=False)

    optimizer = opt_class(
        evaluator=evaluator,
        max_evals=300,
        seed=seed,
        is_meta=True,
        rl_policy=rl_action_fn,
    )
    res = optimizer.run()

    return {
        "transitions": optimizer.transitions,
        "initial_score": float(optimizer.initial_score),
        "best_score": float(optimizer.best_score),
        "gain": float(optimizer.best_score - optimizer.initial_score),
        "reynolds": float(reynolds),
        "alpha": float(alpha),
    }


def train_single_optimizer(
    opt_name: str,
    n_episodes: int = 110,
    num_workers: int = 10,
    device: str = "cpu",
    output_dir: str = "weights",
):
    print(f"\n=======================================================")
    print(f" 开始高效并行训练 Meta-RL 优化器: {opt_name}")
    print(f" 状态维度: {OPTIMIZER_CONFIGS[opt_name]['state_dim']} | 动作维度: {OPTIMIZER_CONFIGS[opt_name]['action_dim']}")
    print(f" 并行工作进程: {num_workers} 核 CPU | 集中梯度更新设备: {device}")
    print(f" 训练 Episode 轮数: {n_episodes} (覆盖 10x11 = 110 气动网格工况)")
    print(f"=======================================================")

    cfg = OPTIMIZER_CONFIGS[opt_name]
    state_dim = cfg["state_dim"]
    action_dim = cfg["action_dim"]

    lr = 2e-4 if opt_name == "PSO" else 3e-4
    epochs = 4 if opt_name == "PSO" else 8
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        update_epochs=epochs,
        device=device,
    )

    grid = get_training_grid()
    rng = np.random.RandomState(42)
    # 随机打乱工况顺序，保证全面而均匀地遍历 110 种气动网格
    shuffled_indices = rng.permutation(len(grid))
    sampled_indices = [int(shuffled_indices[i % len(grid)]) for i in range(n_episodes)]

    # 将 110 个 episode 划分为若干批次 (每批 num_workers 个并发)
    batch_size = max(1, num_workers)
    n_batches = int(np.ceil(n_episodes / batch_size))

    start_time = time.time()
    total_transitions_collected = 0
    all_gains = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for b in range(n_batches):
            b_start = b * batch_size
            b_end = min((b + 1) * batch_size, n_episodes)
            current_batch_indices = sampled_indices[b_start:b_end]

            # 导出当前网络参数为纯 NumPy 数组分发给工作进程
            numpy_weights = {k: v.detach().cpu().numpy() for k, v in agent.network.state_dict().items()}

            tasks = []
            for ep_offset, cond_idx in enumerate(current_batch_indices):
                ep_global = b_start + ep_offset
                reynolds, alpha = grid[cond_idx]
                seed = int(1000 + ep_global)
                tasks.append((opt_name, reynolds, alpha, seed, numpy_weights))

            # 10 核 CPU 并行收集 10 个工况的完整 300 步优化经验
            batch_results = list(executor.map(worker_rollout, tasks))

            # 汇集本批次的所有转移经验
            batch_transitions = []
            batch_gains = []
            for r in batch_results:
                batch_transitions.extend(r["transitions"])
                batch_gains.append(r["gain"])
                all_gains.append(r["gain"])

            total_transitions_collected += len(batch_transitions)

            # 集中式更新 PPO 策略网络
            loss_dict = agent.update(batch_transitions)

            recent_avg_gain = float(np.mean(all_gains[-min(20, len(all_gains)):]))
            elapsed = time.time() - start_time
            print(
                f"[{opt_name:9s}] Batch {b+1:2d}/{n_batches:2d} (Ep {b_end:3d}/{n_episodes}) | "
                f"Transitions: +{len(batch_transitions):4d} (Tot: {total_transitions_collected:5d}) | "
                f"Batch Gain: +{np.mean(batch_gains):5.2f} (20-Ep Avg: +{recent_avg_gain:5.2f}) | "
                f"Loss(PG: {loss_dict.get('policy_loss', 0.0):.3f}, V: {loss_dict.get('value_loss', 0.0):.2f}) | "
                f"Time: {elapsed:5.1f}s"
            )

    os.makedirs(output_dir, exist_ok=True)
    weight_file = os.path.join(output_dir, f"meta_ppo_{opt_name.lower().replace('-', '_')}.pt")
    agent.save(weight_file)
    print(f">>> {opt_name} Meta-Policy 训练完成，权重保存至: {weight_file}")

    return agent


def main():
    parser = argparse.ArgumentParser(description="RL-Meta-Airfoil Multi-Core Parallel PPO Training")
    parser.add_argument(
        "--episodes",
        type=int,
        default=110,
        help="Number of training episodes per optimizer (default: 110 covering the full 10x11 grid)",
    )
    parser.add_argument(
        "--optimizers",
        type=str,
        default="SPSA,Py-BOBYQA,DE,PSO,CMA-ES,TuRBO",
        help="Comma-separated list of optimizers to train",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of CPU workers for parallel rollout collection (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for PPO training updates (cuda/cpu)",
    )
    parser.add_argument("--output_dir", type=str, default="weights", help="Directory to save model weights")
    args = parser.parse_args()

    target_optimizers = [opt.strip() for opt in args.optimizers.split(",") if opt.strip()]

    print(f"RL-Meta-Airfoil Multi-Core Parallel PPO Training Initialized")
    print(f"Device: {args.device} | Parallel Workers: {args.num_workers} | Episodes: {args.episodes}")
    print(f"Target Optimizers: {target_optimizers}")

    total_start = time.time()
    for opt_name in target_optimizers:
        if opt_name not in OPTIMIZER_CONFIGS:
            print(f"Warning: Unknown optimizer '{opt_name}', skipping...")
            continue
        train_single_optimizer(
            opt_name=opt_name,
            n_episodes=args.episodes,
            num_workers=args.num_workers,
            device=args.device,
            output_dir=args.output_dir,
        )

    print(f"\n=======================================================")
    print(f" 全部优化器 Meta-RL 并行训练完毕！总耗时: {time.time()-total_start:.1f} 秒")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
