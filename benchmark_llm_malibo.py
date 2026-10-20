"""LLM-Meta-TuRBO 与 MALIBO 10 随机种子对比基准测试脚本：
严格遵循工作区规范 GEMINI.md 与学术对比严谨性要求。
在统一基准工况 (Re=3.5e6, AoA=3.5度, NACA0012 起始, 300 次 NFE) 下：
1. 从 results/benchmark_results.json 中直接载入已有的 Vanilla-TuRBO 与 RL-Meta-TuRBO 10 种子测试数据；
2. 完整运行 10 个随机种子下的 LLM-Meta-TuRBO 优化实验；
3. 完整运行 10 个随机种子下的 MALIBO 贝叶斯元优化实验；
4. 结构化汇总 4 种方法全指标并持久化为 results/llm_malibo_benchmark_results.json 与 .npz 文件；
5. 输出全样本统计数据（均值、标准差、分位数、提升幅度、物理可行性）。
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# 引入项目核心组件
from src.airfoil_utils import get_kulfan_bounds, get_naca0012_theta, theta_to_coordinates
from src.evaluator import AerodynamicEvaluator
from src.llm_agent import OptimizedLLMMetaAgent
from src.optimizers.llm_turbo import LLMMetaTuRBOOptimizer

# 引入 Meta-CFD 组件
_META_CFD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Meta-CFD")
if _META_CFD_DIR not in sys.path:
    sys.path.insert(0, _META_CFD_DIR)

from malibo.config import MALIBOConfig, ResFFNConfig
from malibo.models import MALIBOMetaModel
from malibo.optimizer import MALIBOOptimizer

# 10 个标准独立随机种子 (与 benchmark.py 严格完全对齐)
BENCHMARK_SEEDS = [42, 123, 256, 777, 1024, 2024, 3407, 5555, 8888, 9999]


# ==============================================================================
# 1. MALIBO 驱动包装函数
# ==============================================================================

def run_malibo_single_seed(
    seed: int,
    reynolds: float = 3.5e6,
    alpha: float = 3.5,
    max_evals: int = 300,
    checkpoint_path: str = "Meta-CFD/malibo/malibo_checkpoint.pt",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """对单个随机种子执行 MALIBO 元优化"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from evaluator import AerodynamicEvaluator as MetaCFDEvaluator
    evaluator = MetaCFDEvaluator(reynolds=reynolds, alpha=alpha)
    from malibo_dataset_generator.airfoil_baselines import (
        get_airfoil_bounds,
        get_airfoil_parameters,
    )
    x0 = get_airfoil_parameters("naca0012")
    lb, ub = get_airfoil_bounds(x0)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_tasks = ckpt.get("num_tasks", 550)
    feature_dim = ckpt.get("feature_dim", 50)

    res_cfg = ResFFNConfig(in_dim=18, feature_dim=feature_dim)
    model = MALIBOMetaModel(num_tasks=num_tasks, config=res_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    opt_cfg = MALIBOConfig()
    opt_cfg.optimization.seed = seed
    opt_cfg.optimization.budget = max_evals
    init_x_unit = (x0 - lb) / (ub - lb + 1e-8)

    def is_geom_valid(th: np.ndarray) -> bool:
        _, af = evaluator._theta_to_kulfan(th)
        is_fatal, _, _ = evaluator.validate_geometry(af)
        return not is_fatal

    malibo_opt = MALIBOOptimizer(
        meta_model=model,
        dim=18,
        lb=lb,
        ub=ub,
        init_x=init_x_unit,
        evaluate_init_first=True,
        config=opt_cfg,
        budget=max_evals,
        geometry_validator=is_geom_valid,
        device=device,
    )

    t0 = time.time()
    res = malibo_opt.optimize(objective_func=evaluator.evaluate, budget=max_evals, verbose_every=50)
    elapsed = time.time() - t0

    best_x = res["best_theta"]
    det = evaluator.evaluate_detailed(best_x)
    coords = theta_to_coordinates(best_x)

    return {
        "seed": seed,
        "best_score": float(res["best_y"]),
        "history": [float(x) for x in res["history"]],
        "best_x": [float(x) for x in best_x],
        "coordinates": {k: v.tolist() for k, v in coords.items()},
        "detailed": {
            "cl": float(det.get("cl", 0.0)),
            "cd": float(det.get("cd", 0.0)),
            "cm": float(det.get("cm", 0.0)),
            "t_max": float(det.get("t_max", 0.0)),
            "x_t_max": float(det.get("x_t_max", 0.0)),
            "area": float(det.get("area", 0.0)),
            "is_feasible": bool(det.get("is_feasible", True)),
        },
        "elapsed_seconds": float(elapsed),
    }


# ==============================================================================
# 2. LLM-Meta-TuRBO 驱动包装函数
# ==============================================================================

def run_llm_meta_single_seed(
    seed: int,
    llm_agent: OptimizedLLMMetaAgent,
    reynolds: float = 3.5e6,
    alpha: float = 3.5,
    max_evals: int = 300,
) -> Dict[str, Any]:
    """对单个随机种子执行 LLM-Meta-TuRBO 元优化"""
    evaluator = AerodynamicEvaluator(reynolds=reynolds, alpha=alpha)
    x0 = get_naca0012_theta()
    lb, ub = get_kulfan_bounds(x0)

    opt = LLMMetaTuRBOOptimizer(
        evaluator=evaluator,
        x0=x0,
        bounds=(lb, ub),
        max_evals=max_evals,
        seed=seed,
        llm_agent=llm_agent,
    )

    t0 = time.time()
    res = opt.run()
    elapsed = time.time() - t0

    best_x = res["best_x"]
    det = evaluator.evaluate_detailed(best_x)
    coords = theta_to_coordinates(best_x)

    return {
        "seed": seed,
        "best_score": float(res["best_score"]),
        "history": [float(x) for x in res["history"]],
        "best_x": [float(x) for x in best_x],
        "coordinates": {k: v.tolist() for k, v in coords.items()},
        "detailed": {
            "cl": float(det.get("cl", 0.0)),
            "cd": float(det.get("cd", 0.0)),
            "cm": float(det.get("cm", 0.0)),
            "t_max": float(det.get("t_max", 0.0)),
            "x_t_max": float(det.get("x_t_max", 0.0)),
            "area": float(det.get("area", 0.0)),
            "is_feasible": bool(det.get("is_feasible", True)),
        },
        "interventions": res.get("interventions", []),
        "elapsed_seconds": float(elapsed),
    }


# ==============================================================================
# 3. 全流程基准运行与持久化
# ==============================================================================

def run_full_benchmark(
    base_benchmark_path: str = "results/benchmark_results.json",
    output_json_path: str = "results/llm_malibo_benchmark_results.json",
    output_npz_path: str = "results/llm_malibo_benchmark_results.npz",
    checkpoint_path: str = "Meta-CFD/malibo/malibo_checkpoint.pt",
    llm_url: str = "http://127.0.0.1:30000/v1",
    llm_model: str = "Qwen/Qwen3.8-27B-FP8",
    reynolds: float = 3.5e6,
    alpha: float = 3.5,
    max_evals: int = 300,
    seeds: List[int] = BENCHMARK_SEEDS,
    skip_llm: bool = False,
    skip_malibo: bool = False,
):
    print("=" * 80)
    print(" 开始 LLM-Meta-TuRBO 与 MALIBO 10 随机种子基准对比测试")
    print(f" 工况设置: Re={reynolds:.2e}, AoA={alpha:.1f}°, NFE={max_evals}, 起始翼型=NACA0012")
    print(f" 测试种子 ({len(seeds)}个): {seeds}")
    print("=" * 80)

    # 1. 载入原始 benchmark_results.json 中的 Vanilla-TuRBO 与 RL-Meta-TuRBO
    if not os.path.exists(base_benchmark_path):
        raise FileNotFoundError(f"未找到原始基准结果文件: {base_benchmark_path}")

    with open(base_benchmark_path, "r", encoding="utf-8") as f:
        orig_data = json.load(f)

    vanilla_turbo_data = orig_data["results"]["TuRBO"]["Vanilla"]
    rl_meta_turbo_data = orig_data["results"]["TuRBO"]["RL-Meta"]

    print(f"\n[1/4] 成功载入基线数据:")
    v_scores = [r["best_score"] for r in vanilla_turbo_data]
    r_scores = [r["best_score"] for r in rl_meta_turbo_data]
    print(f"  - Vanilla-TuRBO: 均值={np.mean(v_scores):.2f} ± {np.std(v_scores):.2f}, 最大={np.max(v_scores):.2f}")
    print(f"  - RL-Meta-TuRBO: 均值={np.mean(r_scores):.2f} ± {np.std(r_scores):.2f}, 最大={np.max(r_scores):.2f}")

    # 读取可能已存在的持久化文件，供复用
    prev_saved_results = {}
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                prev_saved_results = json.load(f).get("results", {})
        except Exception:
            pass

    # 2. 运行 MALIBO
    if skip_malibo and "MALIBO" in prev_saved_results:
        print(f"\n[2/4] 跳过 MALIBO 重新评测，直接复用已有 10 种子实验数据...")
        malibo_results = prev_saved_results["MALIBO"]
    else:
        print(f"\n[2/4] 启动 MALIBO 元学习贝叶斯优化测试 (10 个随机种子)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        malibo_results = []
        for idx, s in enumerate(seeds):
            t_start = time.time()
            print(f"  >>> [MALIBO Seed {s} ({idx+1}/{len(seeds)})] 正在执行 300 步优化...")
            res_s = run_malibo_single_seed(
                seed=s,
                reynolds=reynolds,
                alpha=alpha,
                max_evals=max_evals,
                checkpoint_path=checkpoint_path,
                device=device,
            )
            malibo_results.append(res_s)
            print(f"      --> 完成! 耗时: {res_s['elapsed_seconds']:.1f}s | 最终升阻比: {res_s['best_score']:.2f}")

            # 中途持久化防掉电
            partial_payload = {
                "condition": {"airfoil": "NACA0012", "reynolds": reynolds, "alpha": alpha, "max_evals": max_evals, "seeds": seeds},
                "results": {
                    "Vanilla-TuRBO": vanilla_turbo_data,
                    "RL-Meta-TuRBO": rl_meta_turbo_data,
                    "MALIBO": malibo_results,
                }
            }
            if "LLM-Meta-TuRBO" in prev_saved_results:
                partial_payload["results"]["LLM-Meta-TuRBO"] = prev_saved_results["LLM-Meta-TuRBO"]
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(partial_payload, f, indent=2, ensure_ascii=False)

    m_scores = [r["best_score"] for r in malibo_results]
    print(f"  [MALIBO 完成] 均值={np.mean(m_scores):.2f} ± {np.std(m_scores):.2f}, 最大={np.max(m_scores):.2f}")

    # 3. 运行 LLM-Meta-TuRBO
    partial_payload = {
        "condition": {"airfoil": "NACA0012", "reynolds": reynolds, "alpha": alpha, "max_evals": max_evals, "seeds": seeds},
        "results": {
            "Vanilla-TuRBO": vanilla_turbo_data,
            "RL-Meta-TuRBO": rl_meta_turbo_data,
            "MALIBO": malibo_results,
        }
    }

    if skip_llm and "LLM-Meta-TuRBO" in prev_saved_results:
        print(f"\n[3/4] 跳过 LLM-Meta-TuRBO 重新评测，直接复用已有 10 种子测试数据...")
        llm_meta_results = prev_saved_results["LLM-Meta-TuRBO"]
    else:
        print(f"\n[3/4] 启动 LLM-Meta-TuRBO 测试 (10 个随机种子, SGLang @ {llm_url})...")
        llm_agent = OptimizedLLMMetaAgent(base_url=llm_url, model=llm_model)
        llm_meta_results = []
        for idx, s in enumerate(seeds):
            print(f"  >>> [LLM-Meta-TuRBO Seed {s} ({idx+1}/{len(seeds)})] 正在执行 300 步优化...")
            res_s = run_llm_meta_single_seed(
                seed=s,
                llm_agent=llm_agent,
                reynolds=reynolds,
                alpha=alpha,
                max_evals=max_evals,
            )
            llm_meta_results.append(res_s)
            print(
                f"      --> 完成! 耗时: {res_s['elapsed_seconds']:.1f}s | "
                f"干预次数: {len(res_s['interventions'])} | 最终升阻比: {res_s['best_score']:.2f}"
            )

            # 中途持久化
            partial_payload["results"]["LLM-Meta-TuRBO"] = llm_meta_results
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(partial_payload, f, indent=2, ensure_ascii=False)

    l_scores = [r["best_score"] for r in llm_meta_results]
    print(f"  [LLM-Meta-TuRBO 完成] 均值={np.mean(l_scores):.2f} ± {np.std(l_scores):.2f}, 最大={np.max(l_scores):.2f}")

    # 4. 构建统一结果字典与学术统计摘要
    print(f"\n[4/4] 汇总统计数据并保存至 {output_json_path} 与 {output_npz_path}...")

    methods_data = {
        "Vanilla-TuRBO": vanilla_turbo_data,
        "RL-Meta-TuRBO": rl_meta_turbo_data,
        "LLM-Meta-TuRBO": llm_meta_results,
        "MALIBO": malibo_results,
    }

    v_mean = float(np.mean(v_scores))

    summary_list = []
    for name, data_list in methods_data.items():
        scores = [r["best_score"] for r in data_list]
        m = float(np.mean(scores))
        s = float(np.std(scores))
        med = float(np.median(scores))
        gain = m - v_mean
        pct = (gain / max(abs(v_mean), 1e-4)) * 100.0

        feas_count = sum(1 for r in data_list if r.get("detailed", {}).get("is_feasible", True))

        summary_list.append({
            "method": name,
            "mean": m,
            "std": s,
            "median": med,
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "gain_vs_vanilla": gain,
            "pct_gain_vs_vanilla": pct,
            "feasibility_rate": feas_count / len(data_list),
        })

    final_payload = {
        "condition": {
            "airfoil": "NACA0012",
            "reynolds": reynolds,
            "alpha": alpha,
            "max_evals": max_evals,
            "seeds": seeds,
        },
        "summary": summary_list,
        "results": methods_data,
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)
    print(f"  -> JSON 数据已持久化: {output_json_path}")

    # 保存 NPZ 文件
    npz_dict = {
        "seeds": np.array(seeds),
        "vanilla_history": np.array([r["history"] for r in vanilla_turbo_data]),
        "vanilla_scores": np.array(v_scores),
        "rl_meta_history": np.array([r["history"] for r in rl_meta_turbo_data]),
        "rl_meta_scores": np.array(r_scores),
        "llm_meta_history": np.array([r["history"] for r in llm_meta_results]),
        "llm_meta_scores": np.array(l_scores),
        "malibo_history": np.array([r["history"] for r in malibo_results]),
        "malibo_scores": np.array(m_scores),
    }
    np.savez_compressed(output_npz_path, **npz_dict)
    print(f"  -> NPZ 数据已持久化: {output_npz_path}")

    # 打印最终对比大表
    print("\n" + "=" * 105)
    print(" 四大优化方法全面对比基准测试结果 (10 个独立随机种子, NACA0012 @ Re=3.5M, AoA=3.5°)")
    print("=" * 105)
    print(f"{'Method':<18} | {'Mean ± Std':<18} | {'Median':<8} | {'Min / Max':<16} | {'Δ vs Vanilla':<14} | {'Feasible':<8}")
    print("-" * 105)
    for row in summary_list:
        delta_str = f"{row['gain_vs_vanilla']:+.2f} ({row['pct_gain_vs_vanilla']:+.1f}%)" if row['method'] != "Vanilla-TuRBO" else "Baseline"
        print(
            f"{row['method']:<18} | {row['mean']:6.2f} ± {row['std']:5.2f}   | "
            f"{row['median']:6.2f}   | {row['min']:5.1f} / {row['max']:5.1f}   | "
            f"{delta_str:<14} | {row['feasibility_rate']*100:.0f}%"
        )
    print("=" * 105)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-Meta-TuRBO 与 MALIBO 10 随机种子基准测试")
    parser.add_argument("--llm_url", type=str, default="http://127.0.0.1:30000/v1")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--reynolds", type=float, default=3.5e6)
    parser.add_argument("--alpha", type=float, default=3.5)
    parser.add_argument("--max_evals", type=int, default=300)
    parser.add_argument("--skip_llm", action="store_true", help="跳过 LLM 重新运行，复用已有数据")
    parser.add_argument("--skip_malibo", action="store_true", help="跳过 MALIBO 重新运行，复用已有数据")
    args = parser.parse_args()

    run_full_benchmark(
        llm_url=args.llm_url,
        llm_model=args.llm_model,
        reynolds=args.reynolds,
        alpha=args.alpha,
        max_evals=args.max_evals,
        skip_llm=args.skip_llm,
        skip_malibo=args.skip_malibo,
    )
