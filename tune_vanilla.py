"""HPC 10核并行 Vanilla 优化器超参数网格搜索与峰值标定脚本：
在测试工况 (Re=3.5e6, alpha=3.5度) 下，利用 10 个 CPU 核心并行评测各算法的超参数组合，
找出各自使 Vanilla 表现达到最高均值升阻比（且 100% 可行）的最优初始超参数。
"""

import concurrent.futures
import itertools
import os
import time
from typing import Any, Dict, List, Tuple
import numpy as np

from src.evaluator import AerodynamicEvaluator
from src.optimizers.spsa import SPSAOptimizer
from src.optimizers.bobyqa import PyBobyqaOptimizer
from src.optimizers.ga import GAOptimizer  # DE/GA
from src.optimizers.pso import PSOOptimizer
from src.optimizers.cma_es import CMAESOptimizer
from src.optimizers.turbo import TuRBOOptimizer

BENCHMARK_SEEDS = [42, 123, 256, 777, 1024]
TEST_RE = 3.5e6
TEST_ALPHA = 3.5
MAX_EVALS = 300


def eval_single_run(args: Tuple[str, Dict[str, Any], int]) -> Dict[str, Any]:
    opt_name, kwargs, seed = args
    evaluator = AerodynamicEvaluator(reynolds=TEST_RE, alpha=TEST_ALPHA)

    cls_map = {
        "SPSA": SPSAOptimizer,
        "Py-BOBYQA": PyBobyqaOptimizer,
        "DE": GAOptimizer,
        "PSO": PSOOptimizer,
        "CMA-ES": CMAESOptimizer,
        "TuRBO": TuRBOOptimizer,
    }
    OptCls = cls_map[opt_name]
    opt = OptCls(evaluator=evaluator, max_evals=MAX_EVALS, seed=seed, is_meta=False, **kwargs)
    res = opt.run()
    det = evaluator.evaluate_detailed(res["best_x"])
    return {
        "opt_name": opt_name,
        "kwargs": kwargs,
        "seed": seed,
        "score": float(res["best_score"]),
        "feasible": bool(det["is_feasible"]),
        "ld": float(det["ld"]),
    }


def tune_optimizer(opt_name: str, param_grid: List[Dict[str, Any]], max_workers: int = 10) -> Tuple[Dict[str, Any], float]:
    print(f"\n>>> 正在针对 [{opt_name}] 开展 10 核并行超参数网格扫描 (共 {len(param_grid)} 组参数配置，每组 5 种子)...")
    tasks = []
    for params in param_grid:
        for seed in BENCHMARK_SEEDS:
            tasks.append((opt_name, params, seed))

    start_t = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(eval_single_run, tasks))

    # 聚合每组参数的表现
    grouped: Dict[str, List[float]] = {}
    grouped_feasible: Dict[str, List[bool]] = {}
    param_map: Dict[str, Dict[str, Any]] = {}

    for r in results:
        key = str(sorted(r["kwargs"].items()))
        param_map[key] = r["kwargs"]
        if key not in grouped:
            grouped[key] = []
            grouped_feasible[key] = []
        grouped[key].append(r["score"])
        grouped_feasible[key].append(r["feasible"])

    best_key = None
    best_mean = -float("inf")

    for key, scores in grouped.items():
        feas = grouped_feasible[key]
        feas_rate = sum(feas) / len(feas)
        mean_s = np.mean(scores)
        std_s = np.std(scores)
        if feas_rate == 1.0 and mean_s > best_mean:
            best_mean = mean_s
            best_key = key

    best_params = param_map[best_key]
    elapsed = time.time() - start_t
    print(f"[{opt_name}] 标定完成 (耗时: {elapsed:.1f}s)!")
    print(f"  最优参数: {best_params}")
    print(f"  峰值均值升阻比: {best_mean:.2f} (5 种子 100% 可行)")
    return best_params, best_mean


def main():
    print("=" * 65)
    print("HPC 10 核并行 Vanilla 优化器超参天花板标定程序启动")
    print("=" * 65)

    grids = {
        "SPSA": [
            {"a0": a0, "c0": c0, "momentum": mom}
            for a0 in [0.0008, 0.001, 0.0015]
            for c0 in [0.012, 0.015, 0.020]
            for mom in [0.65, 0.70, 0.75]
        ],
        "Py-BOBYQA": [
            {"rhobeg": rhobeg, "rhoend": rhoend}
            for rhobeg in [0.06, 0.08, 0.10, 0.12]
            for rhoend in [1e-5, 5e-5, 1e-4]
        ],
        "DE": [
            {"pop_size": pop, "crossover_rate": cr, "f_scale": f, "mutation_scale": sm}
            for pop in [10, 12, 14]
            for cr in [0.75, 0.85]
            for f in [0.65, 0.75, 0.85]
            for sm in [0.003, 0.005]
        ],
        "PSO": [
            {"swarm_size": pop, "w": w, "c1": c1, "c2": c2}
            for pop in [12, 15]
            for w in [0.65, 0.729, 0.78]
            for c1, c2 in [(1.494, 1.494), (1.6, 1.6), (1.4, 1.8)]
        ],
        "CMA-ES": [
            {"sigma0": sig, "pop_size": pop}
            for sig in [0.02, 0.025, 0.03, 0.035, 0.04]
            for pop in [10, 12, 14]
        ],
        "TuRBO": [
            {"length_init": l_init, "success_tol": s_tol, "failure_tol": f_tol}
            for l_init in [0.6, 0.8, 1.0]
            for s_tol in [2, 3]
            for f_tol in [3, 4]
        ],
    }

    best_all = {}
    for name, grid in grids.items():
        best_params, best_score = tune_optimizer(name, grid, max_workers=10)
        best_all[name] = {"params": best_params, "best_score": best_score}

    print("\n" + "=" * 65)
    print("标定总结表：各 Vanilla 最优配置与理论峰值")
    print("=" * 65)
    for name, res in best_all.items():
        print(f"{name:<10}: L/D = {res['best_score']:.2f} | 参数: {res['params']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
