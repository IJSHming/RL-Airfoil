"""RL-Meta-Airfoil 10 随机种子对比基准测试脚本：
在统一工况 (Re=3.5e6, AoA=3.5度) 下，
针对 6 种优化器分别使用 10 个独立随机种子运行 Vanilla 与 RL-Meta 对比实验，
完整记录 300 步升阻比轨迹与最终翼型几何形状，并持久化为 JSON 与 NPZ 数据文件。
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np
import torch

from src.airfoil_utils import theta_to_coordinates
from src.evaluator import AerodynamicEvaluator
from src.optimizers import OPTIMIZER_CONFIGS
from src.ppo import PPOAgent

# 10 个标准独立随机种子
BENCHMARK_SEEDS = [42, 123, 256, 777, 1024, 2024, 3407, 5555, 8888, 9999]


def run_benchmark(
    weights_dir: str = "weights",
    output_dir: str = "results",
    reynolds: float = 3.5e6,
    alpha: float = 3.5,
    max_evals: int = 300,
    seeds: List[int] = BENCHMARK_SEEDS,
    device: str = "cpu",
):
    print(f"\n=======================================================")
    print(f" 开始 RL-Meta vs Vanilla 严格独立对比测试")
    print(f" 评估工况: Re = {reynolds/1e6:.1f}M, AoA = {alpha:.1f}°")
    print(f" 优化器集合: {list(OPTIMIZER_CONFIGS.keys())}")
    print(f" 测试随机种子数: {len(seeds)} 个 ({seeds})")
    print(f" 每轮评估预算: {max_evals} 次 NFE")
    print(f"=======================================================\n")

    evaluator = AerodynamicEvaluator(reynolds=reynolds, alpha=alpha)

    # 存储全部实验数据
    results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    summary_table = []

    for opt_name, cfg in OPTIMIZER_CONFIGS.items():
        results[opt_name] = {"Vanilla": [], "RL-Meta": []}

        # 加载 Meta-PPO 权重
        weight_path = os.path.join(weights_dir, f"meta_ppo_{opt_name.lower().replace('-', '_')}.pt")
        meta_agent = None
        if os.path.exists(weight_path):
            meta_agent = PPOAgent(
                state_dim=cfg["state_dim"],
                action_dim=cfg["action_dim"],
                device=device,
            )
            meta_agent.load(weight_path)
            print(f"[{opt_name}] 成功加载已训练权重: {weight_path}")
        else:
            print(f"[{opt_name}] 警告: 未找到权重文件 {weight_path}，使用未训练基线智能体测试")
            meta_agent = PPOAgent(
                state_dim=cfg["state_dim"],
                action_dim=cfg["action_dim"],
                device=device,
            )

        OptClass = cfg["class"]

        vanilla_final_scores = []
        meta_final_scores = []

        # 遍历 10 个随机种子
        for seed_idx, seed in enumerate(seeds):
            # 1. 运行 Vanilla 基准
            v_opt = OptClass(
                evaluator=evaluator,
                max_evals=max_evals,
                seed=seed,
                is_meta=False,
            )
            v_res = v_opt.run()
            v_det = evaluator.evaluate_detailed(v_res["best_x"])
            v_coords = theta_to_coordinates(v_res["best_x"])

            v_record = {
                "seed": seed,
                "best_score": float(v_res["best_score"]),
                "history": [float(x) for x in v_res["history"]],
                "best_x": [float(x) for x in v_res["best_x"]],
                "coordinates": {k: v.tolist() for k, v in v_coords.items()},
                "detailed": {
                    "cl": float(v_det["cl"]),
                    "cd": float(v_det["cd"]),
                    "cm": float(v_det["cm"]),
                    "t_max": float(v_det["t_max"]),
                    "x_t_max": float(v_det["x_t_max"]),
                    "area": float(v_det["area"]),
                    "is_feasible": bool(v_det["is_feasible"]),
                },
            }
            results[opt_name]["Vanilla"].append(v_record)
            vanilla_final_scores.append(v_res["best_score"])

            # 2. 运行 RL-Meta
            m_opt = OptClass(
                evaluator=evaluator,
                max_evals=max_evals,
                seed=seed,
                is_meta=True,
                rl_policy=lambda s: meta_agent.select_action(s, deterministic=True),
            )
            m_res = m_opt.run()
            m_det = evaluator.evaluate_detailed(m_res["best_x"])
            m_coords = theta_to_coordinates(m_res["best_x"])

            m_record = {
                "seed": seed,
                "best_score": float(m_res["best_score"]),
                "history": [float(x) for x in m_res["history"]],
                "best_x": [float(x) for x in m_res["best_x"]],
                "coordinates": {k: v.tolist() for k, v in m_coords.items()},
                "detailed": {
                    "cl": float(m_det["cl"]),
                    "cd": float(m_det["cd"]),
                    "cm": float(m_det["cm"]),
                    "t_max": float(m_det["t_max"]),
                    "x_t_max": float(m_det["x_t_max"]),
                    "area": float(m_det["area"]),
                    "is_feasible": bool(m_det["is_feasible"]),
                },
            }
            results[opt_name]["RL-Meta"].append(m_record)
            meta_final_scores.append(m_res["best_score"])

            print(
                f"  [{opt_name:9s}] Seed {seed_idx+1:2d}/{len(seeds)} (seed={seed:4d}) | "
                f"Vanilla L/D: {v_res['best_score']:6.2f} vs RL-Meta L/D: {m_res['best_score']:6.2f} "
                f"(Diff: {m_res['best_score'] - v_res['best_score']:+6.2f})"
            )

        v_mean, v_std = np.mean(vanilla_final_scores), np.std(vanilla_final_scores)
        m_mean, m_std = np.mean(meta_final_scores), np.std(meta_final_scores)
        improvement = m_mean - v_mean
        pct_imp = (improvement / max(abs(v_mean), 1e-4)) * 100.0

        summary_table.append({
            "optimizer": opt_name,
            "vanilla_mean": v_mean,
            "vanilla_std": v_std,
            "meta_mean": m_mean,
            "meta_std": m_std,
            "gain": improvement,
            "pct_gain": pct_imp,
        })

        print(
            f"--> [{opt_name:9s}] 10-Seed Summary | "
            f"Vanilla: {v_mean:.2f} ± {v_std:.2f} | "
            f"RL-Meta: {m_mean:.2f} ± {m_std:.2f} | "
            f"提升: {improvement:+.2f} ({pct_imp:+.1f}%)\n"
        )

    # 保存实验数据
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "condition": {"reynolds": reynolds, "alpha": alpha, "max_evals": max_evals},
                "seeds": seeds,
                "summary": summary_table,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # 同时保存高紧凑 npz 格式供绘图快速读取
    npz_data = {}
    for opt_name in OPTIMIZER_CONFIGS.keys():
        v_hist = np.array([r["history"] for r in results[opt_name]["Vanilla"]])
        m_hist = np.array([r["history"] for r in results[opt_name]["RL-Meta"]])
        npz_data[f"{opt_name}_vanilla_history"] = v_hist
        npz_data[f"{opt_name}_meta_history"] = m_hist

    npz_path = os.path.join(output_dir, "benchmark_results.npz")
    np.savez_compressed(npz_path, **npz_data)

    print(f"=======================================================")
    print(f" 对比基准测试完成！数据已保存至:")
    print(f" - JSON 完整格式: {json_path}")
    print(f" - NPZ 紧凑格式:  {npz_path}")
    print(f"=======================================================")

    print(f"\n{'算法':<12} | {'Vanilla 升阻比':<18} | {'RL-Meta 升阻比':<18} | {'绝对提升':<10} | {'相对提升'}")
    print("-" * 75)
    for row in summary_table:
        print(
            f"{row['optimizer']:<12} | "
            f"{row['vanilla_mean']:6.2f} ± {row['vanilla_std']:5.2f}      | "
            f"{row['meta_mean']:6.2f} ± {row['meta_std']:5.2f}      | "
            f"{row['gain']:+7.2f}    | "
            f"{row['pct_gain']:+6.1f}%"
        )
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="RL-Meta-Airfoil Benchmark")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Directory with saved PPO weights")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save benchmark results")
    parser.add_argument("--reynolds", type=float, default=3.5e6, help="Test Reynolds number")
    parser.add_argument("--alpha", type=float, default=3.5, help="Test angle of attack (degrees)")
    parser.add_argument("--max_evals", type=int, default=300, help="Max evaluations budget")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run PPO inference (cuda/cpu)",
    )
    args = parser.parse_args()

    run_benchmark(
        weights_dir=args.weights_dir,
        output_dir=args.output_dir,
        reynolds=args.reynolds,
        alpha=args.alpha,
        max_evals=args.max_evals,
        seeds=BENCHMARK_SEEDS,
        device=args.device,
    )


if __name__ == "__main__":
    main()
