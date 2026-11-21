"""RL-Meta-Airfoil 强化学习元优化器高效并行训练脚本 (Paired Control-Variate PPO + 90/20 Internal Validation):
严格遵守科研诚信原则：
1. 训练与内部验证工况均严格限制于 10x11 = 110 个离线网格工况 (Re in {1..10}e6, alpha in {0..10} deg)，
   彻底隔离测试工况 (Re=3.5e6, alpha=3.5 deg)；
2. 引入同种子配对 Vanilla 控制变量 (Common Random Numbers, CRN)：每个训练回合在相同 (Re, alpha, seed) 下
   对比 RL-Meta 与 Vanilla 增益，构造跨工况尺度不变的相对增量 + 终局超额增益奖励，彻底消除跨工况难度方差；
3. 将 110 网格按 90 训练 / 20 内部验证分层切分，周期性在 20 验证工况上评测确定性策略相对 Vanilla 的平均超额增益，
   择优保存验证集最优检查点 (从零策略基线 0.0% 起步，杜绝末端抖动退化快照)；
4. 采用 spawn 多进程上下文 + 单线程 BLAS + 纯 NumPy 策略前向推理，杜绝 CUDA/OpenMP 死锁与线程争抢。
"""

import argparse
import concurrent.futures
import copy
import multiprocessing
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from typing import Any, Dict, List, Tuple
import numpy as np
import torch

from src.evaluator import AerodynamicEvaluator
from src.optimizers import OPTIMIZER_CONFIGS
from src.ppo import PPOAgent


def get_training_grid() -> List[Tuple[float, float]]:
    """生成 10 个雷诺数 x 11 个攻角 = 110 个离线网格工况 (严格排除测试点 Re=3.5e6, alpha=3.5°)"""
    re_values = np.linspace(1e6, 1e7, 10)
    alpha_values = np.linspace(0.0, 10.0, 11)
    grid = []
    for re in re_values:
        for alpha in alpha_values:
            grid.append((float(re), float(alpha)))
    return grid


def split_train_val_grid(grid: List[Tuple[float, float]], n_val: int = 20) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """将 110 个离线网格工况按固定种子均匀切分为 90 训练 + 20 内部验证 (零测试集接触)"""
    rng = np.random.RandomState(2026)
    perm = rng.permutation(len(grid))
    val_set = set(int(i) for i in perm[:n_val])
    train_grid = [grid[i] for i in range(len(grid)) if i not in val_set]
    val_grid = [grid[i] for i in range(len(grid)) if i in val_set]
    return train_grid, val_grid


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


def worker_paired_rollout(task_tuple: Tuple[str, float, float, int, Dict[str, np.ndarray], bool]) -> Dict[str, Any]:
    """单核独立工作进程：在相同 (Re, alpha, seed) 下执行同种子配对 Vanilla 与 RL-Meta 对比采集。"""
    opt_name, reynolds, alpha, seed, weights, deterministic = task_tuple
    cfg = OPTIMIZER_CONFIGS[opt_name]
    opt_class = cfg["class"]

    # 1. 同种子 Vanilla 控制变量基线 (严格仅在离线训练/验证网格内运行)
    ev_v = AerodynamicEvaluator(reynolds=reynolds, alpha=alpha, max_evals=300)
    opt_v = opt_class(evaluator=ev_v, max_evals=300, seed=seed, is_meta=False)
    opt_v.run()
    v_init = float(opt_v.initial_score)
    v_best = float(opt_v.best_score)
    v_gain = float(v_best - v_init)
    scale_denom = max(v_gain, 15.0)

    # 2. 同种子 RL-Meta 策略采集
    ev_m = AerodynamicEvaluator(reynolds=reynolds, alpha=alpha, max_evals=300)

    def rl_action_fn(s: np.ndarray) -> np.ndarray:
        return numpy_policy_forward(s, weights, deterministic=deterministic)

    opt_m = opt_class(
        evaluator=ev_m,
        max_evals=300,
        seed=seed,
        is_meta=True,
        rl_policy=rl_action_fn,
    )
    opt_m.run()

    m_init = float(opt_m.initial_score)
    m_best = float(opt_m.best_score)
    m_gain = float(m_best - m_init)
    excess_abs = float(m_best - v_best)
    excess_rel = float(excess_abs / max(abs(v_best), 20.0))

    transitions = opt_m.transitions
    if len(transitions) > 0:
        v_hist = opt_v.history_best
        has_step_idx = ("eval_count" in transitions[0]) and (len(v_hist) > 0)
        if has_step_idx:
            # 精确势能差分轨迹配对塑造 (Ng et al. 1999): 每步即时奖励 = (m_t - m_{t-1}) - (v_t - v_{t-1})，
            # 彻底消除长时域 (T=285) GAE 折扣衰减对终局配对信号的淹没
            prev_v_idx = max(int(transitions[0]["eval_count"]) - 2, 0)
            prev_v_val = float(v_hist[min(prev_v_idx, len(v_hist) - 1)])
            for tr in transitions:
                raw_delta_m = float(tr["reward"])
                cur_v_idx = max(int(tr["eval_count"]) - 1, 0)
                cur_v_val = float(v_hist[min(cur_v_idx, len(v_hist) - 1)])
                raw_delta_v = cur_v_val - prev_v_val
                prev_v_val = cur_v_val
                tr["reward"] = float((raw_delta_m - 0.8 * raw_delta_v) / scale_denom) * 2.5
                tr["done"] = False
            transitions[-1]["done"] = True
            paired_excess_norm = float(np.clip((m_gain - v_gain) / scale_denom, -1.5, 1.5))
            transitions[-1]["reward"] += 1.0 * paired_excess_norm
        else:
            for tr in transitions:
                raw_delta = float(tr["reward"])
                tr["reward"] = float(raw_delta / scale_denom)
                tr["done"] = False
            transitions[-1]["done"] = True
            paired_excess_norm = float(np.clip((m_gain - v_gain) / scale_denom, -1.5, 1.5))
            transitions[-1]["reward"] += 2.0 * paired_excess_norm

    return {
        "transitions": [] if deterministic else transitions,
        "vanilla_best": v_best,
        "meta_best": m_best,
        "gain": m_gain,
        "excess_abs": excess_abs,
        "excess_rel": excess_rel,
    }


def evaluate_on_val_grid(
    executor: concurrent.futures.ProcessPoolExecutor,
    opt_name: str,
    val_grid: List[Tuple[float, float]],
    numpy_weights: Dict[str, np.ndarray],
    n_seeds_per_cond: int = 2,
) -> Tuple[float, float, int]:
    """在 20 个内部验证工况上以多种子配对评测确定性策略相对 Vanilla 的平均超额增益 (零测试集接触)"""
    tasks = [
        (opt_name, reynolds, alpha, 5000 + s_idx * 100 + idx, numpy_weights, True)
        for s_idx in range(n_seeds_per_cond)
        for idx, (reynolds, alpha) in enumerate(val_grid)
    ]
    res_list = list(executor.map(worker_paired_rollout, tasks))
    mean_excess_abs = float(np.mean([r["excess_abs"] for r in res_list]))
    mean_excess_pct = float(np.mean([r["excess_rel"] for r in res_list]) * 100.0)
    wins = int(sum(1 for r in res_list if r["excess_abs"] > 1e-6))
    return mean_excess_abs, mean_excess_pct, wins


def train_single_optimizer(
    opt_name: str,
    n_episodes: int = 450,
    num_workers: int = 10,
    device: str = "cpu",
    output_dir: str = "weights",
    val_every: int = 5,
):
    print(f"\n=======================================================")
    print(f" 开始配对基线纯净 Meta-RL 并行训练: {opt_name}")
    print(f" 状态维度: {OPTIMIZER_CONFIGS[opt_name]['state_dim']} | 动作维度: {OPTIMIZER_CONFIGS[opt_name]['action_dim']}")
    print(f" 并行工作进程: {num_workers} 核 CPU (spawn) | 梯度更新设备: {device}")
    print(f" 训练回合: {n_episodes} (90 训练工况 / 20 内部验证工况，严格排除测试点 3.5M/3.5°)")
    print(f"=======================================================")

    cfg = OPTIMIZER_CONFIGS[opt_name]
    state_dim = cfg["state_dim"]
    action_dim = cfg["action_dim"]

    torch.manual_seed(42)
    np.random.seed(42)
    base_lr = 3e-4
    min_lr = 5e-5
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=base_lr,
        ent_coef=0.005,
        update_epochs=8,
        device=device,
    )

    full_grid = get_training_grid()
    train_grid, val_grid = split_train_val_grid(full_grid, n_val=20)

    rng = np.random.RandomState(42)
    sampled_indices: List[int] = []
    while len(sampled_indices) < n_episodes:
        sampled_indices.extend([int(idx) for idx in rng.permutation(len(train_grid))])
    sampled_indices = sampled_indices[:n_episodes]

    batch_size = max(1, num_workers)
    n_batches = int(np.ceil(n_episodes / batch_size))

    start_time = time.time()
    total_transitions_collected = 0
    best_val_pct = 0.0
    best_state_dict = copy.deepcopy(agent.network.state_dict())
    best_batch = 0

    mp_ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as executor:
        for b in range(n_batches):
            frac = b / max(n_batches - 1, 1)
            agent.set_lr(base_lr * (1.0 - frac) + min_lr * frac)

            b_start = b * batch_size
            b_end = min((b + 1) * batch_size, n_episodes)
            current_batch_indices = sampled_indices[b_start:b_end]

            numpy_weights = {k: v.detach().cpu().numpy() for k, v in agent.network.state_dict().items()}

            tasks = []
            for ep_offset, cond_idx in enumerate(current_batch_indices):
                ep_global = b_start + ep_offset
                reynolds, alpha = train_grid[cond_idx]
                seed = int(1000 + ep_global)
                tasks.append((opt_name, reynolds, alpha, seed, numpy_weights, False))

            batch_results = list(executor.map(worker_paired_rollout, tasks))

            batch_transitions = []
            batch_excess_abs = []
            for r in batch_results:
                batch_transitions.extend(r["transitions"])
                batch_excess_abs.append(r["excess_abs"])

            total_transitions_collected += len(batch_transitions)
            loss_dict = agent.update(batch_transitions)

            elapsed = time.time() - start_time
            should_val = ((b + 1) % val_every == 0) or (b + 1 == n_batches)
            val_msg = ""
            if should_val:
                eval_weights = {k: v.detach().cpu().numpy() for k, v in agent.network.state_dict().items()}
                val_abs, val_pct, val_wins = evaluate_on_val_grid(executor, opt_name, val_grid, eval_weights)
                star = ""
                if val_pct > best_val_pct:
                    best_val_pct = val_pct
                    best_state_dict = copy.deepcopy(agent.network.state_dict())
                    best_batch = b + 1
                    star = " ★BEST"
                val_msg = f" | Val40: {val_abs:+.2f} ({val_pct:+.2f}%, {val_wins}/40){star}"

            if should_val or b == 0:
                print(
                    f"[{opt_name:9s}] Batch {b+1:2d}/{n_batches:2d} (Ep {b_end:3d}/{n_episodes}) | "
                    f"TotTr: {total_transitions_collected:5d} | "
                    f"TrainΔvsV: {np.mean(batch_excess_abs):+6.2f} | "
                    f"Loss(PG:{loss_dict.get('policy_loss', 0.0):+.3f}, V:{loss_dict.get('value_loss', 0.0):.3f})"
                    f"{val_msg} | {elapsed:5.1f}s",
                    flush=True,
                )

    agent.network.load_state_dict(best_state_dict)
    os.makedirs(output_dir, exist_ok=True)
    weight_file = os.path.join(output_dir, f"meta_ppo_{opt_name.lower().replace('-', '_')}.pt")
    agent.save(weight_file)
    print(f">>> {opt_name} 训练完成! 选用内部验证最佳检查点 (Batch {best_batch}, Val20={best_val_pct:+.2f}%) -> {weight_file}")
    return agent


def main():
    parser = argparse.ArgumentParser(description="RL-Meta-Airfoil Paired-Baseline Parallel PPO Training")
    parser.add_argument("--episodes", type=int, default=450, help="Training episodes per optimizer (default: 450)")
    parser.add_argument("--turbo_episodes", type=int, default=350, help="Training episodes for TuRBO (default: 350)")
    parser.add_argument("--optimizers", type=str, default="SPSA,Py-BOBYQA,DE,PSO,CMA-ES,TuRBO")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="weights")
    parser.add_argument("--val_every", type=int, default=5)
    args = parser.parse_args()

    target_optimizers = [opt.strip() for opt in args.optimizers.split(",") if opt.strip()]
    total_start = time.time()
    for opt_name in target_optimizers:
        if opt_name not in OPTIMIZER_CONFIGS:
            print(f"Warning: Unknown optimizer '{opt_name}', skipping...")
            continue
        eps = args.turbo_episodes if opt_name == "TuRBO" else args.episodes
        train_single_optimizer(
            opt_name=opt_name,
            n_episodes=eps,
            num_workers=args.num_workers,
            device=args.device,
            output_dir=args.output_dir,
            val_every=args.val_every,
        )

    print(f"\n=======================================================")
    print(f" 全部优化器配对基线 Meta-RL 并行训练完毕！总耗时: {time.time()-total_start:.1f} 秒")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
