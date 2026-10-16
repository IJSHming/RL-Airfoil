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
from scipy.linalg import solve_triangular
import torch

# 引入项目核心组件
from src.airfoil_utils import get_kulfan_bounds, get_naca0012_theta, theta_to_coordinates
from src.evaluator import AerodynamicEvaluator
from src.optimizers.base import BaseAirfoilOptimizer
from src.optimizers.turbo import matern52_kernel

# 引入 references/Meta-CFD 组件
_META_CFD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "references", "Meta-CFD")
if _META_CFD_DIR not in sys.path:
    sys.path.insert(0, _META_CFD_DIR)

from llm_agent import LLMMetaAgent
from malibo.config import MALIBOConfig, ResFFNConfig
from malibo.models import MALIBOMetaModel
from malibo.optimizer import MALIBOOptimizer

# 10 个标准独立随机种子 (与 benchmark.py 严格完全对齐)
BENCHMARK_SEEDS = [42, 123, 256, 777, 1024, 2024, 3407, 5555, 8888, 9999]


# ==============================================================================
# 1. 结构化 LLM 客户端包装 (加入高效推理 prompt 与 token 约束)
# ==============================================================================

# ==============================================================================
# 1. 结构化 LLM 客户端包装 (加入宏观气动方向引导与高效推理约束)
# ==============================================================================

def convert_directional_strategy(x_base: np.ndarray, strat: Dict[str, Any]) -> np.ndarray:
    """将 LLM 输出的宏观气动方向策略转换为 18 维 Kulfan 参数偏移量 delta"""
    su = float(strat.get("upper_camber_shift", 0.0))
    sl = float(strat.get("lower_camber_shift", 0.0))
    st = float(strat.get("thickness_shift", 0.0))
    sle = float(strat.get("le_radius_shift", 0.0))

    delta = np.zeros(18, dtype=np.float64)
    # 上表面 0..7: 钟形正弦加权增加弯度与吸力峰 (控制 30%~50% 弦长吸力区)
    u_weights = np.sin(np.pi * (np.arange(8) + 0.5) / 8.0)
    delta[0:8] = su * u_weights + st * 0.5

    # 下表面 8..15: 线性后部加载抬升 (越靠近后缘抬升越明显，增强后部加载与下表面反弯)
    l_weights = (np.arange(8) + 1.0) / 8.0
    delta[8:16] = sl * l_weights - st * 0.5

    # 前缘钝度半径 16
    delta[16] = sle
    return delta


LLM_SYSTEM_PROMPT = """You are a senior aerodynamicist and Bayesian Optimization meta-controller.
Your goal is to maximize the penalized lift-to-drag ratio (L/D) of an airfoil at Re=3.5e6, alpha=3.5 deg starting from NACA 0012.

Key Aerodynamic Insights:
1. Symmetric NACA 0012 has 0 camber, resulting in low lift (CL ~ 0.39) and modest L/D (~65.8).
2. High-performance transonic/subsonic airfoils require:
   - Moderate positive upper camber (upper_camber_shift in [0.05, 0.12]): enhances suction peak without premature shock.
   - Aft reflex and rear loading (lower_camber_shift in [0.04, 0.10]): raises lower surface towards the trailing edge, reducing pressure drag CD.
   - Thickness control (thickness_shift in [-0.02, 0.03]): keeps t_max within the optimal structural range [11%, 16%].
   - Leading edge radius (le_radius_shift in [0.0, 0.03]): softens suction collapse at alpha=3.5 deg.

Strategy per Optimization Phase:
- Early Phase (steps <= 35, low CL): Be BOLD and AGGRESSIVE. Strongly increase upper camber and lower reflex to break symmetry, expand trust region.
- Mid Phase (steps 36-100): Airfoil has camber; gently fine-tune thickness distribution and suction peak position.
- Late Transition (steps 101-150): Micro-tune and slightly contract trust region for smooth local Bayesian convergence.
- After step 150: The optimization will transition entirely to pure Bayesian local exploitation.

Output ONLY a valid JSON object matching this schema:
{
  "thought": "1-2 sentence engineering rationale (under 40 words).",
  "length_scale_factor": float in [0.8, 1.8],
  "prob_perturb": float in [0.25, 0.75],
  "exploration_weight": float in [0.8, 1.8],
  "directional_strategy": {
    "upper_camber_shift": float in [-0.05, 0.15],
    "lower_camber_shift": float in [-0.05, 0.15],
    "thickness_shift": float in [-0.03, 0.04],
    "le_radius_shift": float in [-0.03, 0.03]
  }
}
"""


class OptimizedLLMMetaAgent(LLMMetaAgent):
    """针对 Qwen3.8-27B 宏观气动方向决策优化的元控制器智能体"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:30000/v1",
        model: str = "Qwen/Qwen3.8-27B-FP8",
        api_key: str = "EMPTY",
        timeout: float = 120.0,
    ):
        super().__init__(base_url=base_url, model=model, api_key=api_key, timeout=timeout)

    def build_user_prompt(self, state: Dict[str, Any]) -> str:
        step = state["step"]
        best_cl = state.get("best_cl", 0.39)
        best_ld = state.get("best_ld", 65.8)

        if step <= 35 and best_cl < 0.48:
            phase_str = (
                "EARLY EXPLORATION (Aggressive Camber Needed): Baseline NACA 0012 is symmetric. "
                "You MUST apply bold positive upper camber shift (+0.07 ~ +0.12) and lower reflex (+0.05 ~ +0.10) "
                "to break symmetry and boost lift."
            )
        elif step <= 100:
            phase_str = (
                f"MID REFINEMENT: Airfoil already developed positive camber (CL={best_cl:.3f}, L/D={best_ld:.1f}). "
                "Apply gentle micro-adjustments (shifts in [-0.03, +0.03]) to optimize suction peak and reduce drag without over-thickening."
            )
        else:
            phase_str = (
                f"PRE-CONVERGENCE ALIGNMENT: L/D={best_ld:.1f}. Apply very small micro-tuning (shifts in [-0.02, +0.02]) "
                "and slightly contract trust region (length_scale_factor in [0.8, 1.0]) for smooth Bayesian convergence."
            )

        prompt = f"""=== Current Optimization State ===
- Phase: {phase_str}
- Airfoil: NACA0012 base | Re: {state['reynolds']:.2e} | Alpha: {state['alpha']} deg
- Progress: Step {step} / {state['max_evals']} (Remaining: {state['budget_left']})
- Initial L/D: {state['init_ld']:.2f} | Current Best L/D: {state['best_ld']:.2f}
- Aerodynamics: CL = {best_cl:.4f}, CD = {state.get('best_cd', 0.0):.5f}, t_max = {state.get('best_t_max', 0.12)*100:.2f}% (Target: 11%~15%)
- TuRBO State: TR Length L = {state['tr_length']:.4f}, Stagnation = {state['stagnation_count']}, Successes = {state['success_counter']}/{state['success_tolerance']}

Concisely decide meta-parameters and directional strategy. Output final JSON:"""
        return prompt

    def decide_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_prompt = self.build_user_prompt(state)

        step = state.get("step", 16)
        is_early = (step <= 35 and state.get("best_cl", 0.39) < 0.48)
        default_action = {
            "thought": "Fallback to default TuRBO dynamics.",
            "length_scale_factor": 1.4 if is_early else 1.0,
            "prob_perturb": 0.6 if is_early else 0.5,
            "exploration_weight": 1.4 if is_early else 1.0,
            "directional_strategy": {
                "upper_camber_shift": 0.09 if is_early else 0.01,
                "lower_camber_shift": 0.07 if is_early else 0.01,
                "thickness_shift": 0.01,
                "le_radius_shift": 0.01,
            },
        }

        try:
            if self.client is not None:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": LLM_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    top_p=0.90,
                    max_tokens=2048,
                )
                raw_text = response.choices[0].message.content or ""
                import re
                clean = re.sub(r"```json\s*", "", raw_text)
                clean = re.sub(r"```\s*", "", clean).strip()
                match = re.search(r"\{.*\}", clean, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    scale_L = float(np.clip(data.get("length_scale_factor", 1.0), 0.8, 1.8))
                    prob_p = float(np.clip(data.get("prob_perturb", 0.5), 0.2, 0.8))
                    exp_w = float(np.clip(data.get("exploration_weight", 1.0), 0.8, 1.8))
                    strat = data.get("directional_strategy", {})
                    return {
                        "thought": str(data.get("thought", "")),
                        "length_scale_factor": scale_L,
                        "prob_perturb": prob_p,
                        "exploration_weight": exp_w,
                        "directional_strategy": {
                            "upper_camber_shift": float(np.clip(strat.get("upper_camber_shift", 0.0), -0.05, 0.15)),
                            "lower_camber_shift": float(np.clip(strat.get("lower_camber_shift", 0.0), -0.05, 0.15)),
                            "thickness_shift": float(np.clip(strat.get("thickness_shift", 0.0), -0.04, 0.05)),
                            "le_radius_shift": float(np.clip(strat.get("le_radius_shift", 0.0), -0.03, 0.03)),
                        },
                    }
            return default_action
        except Exception as e:
            print(f"        [LLM Warning] Request encountered '{type(e).__name__}: {e}'. Falling back to safe default.")
            return default_action


# ==============================================================================
# 2. LLM-Meta-TuRBO 优化器实现 (严格对齐 TuRBO 基准超参，结合前中期定向元引导)
# ==============================================================================

class LLMMetaTuRBOOptimizer(BaseAirfoilOptimizer):
    """基于大语言模型宏观定向元控制的高维信赖域贝叶斯优化器 (LLM-Meta-TuRBO)。
    初始超参与 Vanilla-TuRBO、RL-Meta-TuRBO 严格对齐：
    n_init = 15, L_init = 0.8, L_min = 0.5**7, L_max = 1.6, success_tol = 3, failure_tol = 4.
    
    关键机制升级：
    1. 前中期严格集中干预 (仅在 step <= 150 触发)，后期完全交由 TuRBO-1 进行高精度纯贝叶斯收敛；
    2. 宏观方向偏移量 (Directional Strategy) 注入与多尺度工业几何可行性前置检验；
    3. 汤普森候选采样方向性漂移倾斜，兼顾全局气动先验与局部贝叶斯不确定性探索。
    """

    def __init__(
        self,
        evaluator: AerodynamicEvaluator,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        max_evals: int = 300,
        seed: int = 42,
        n_init: int = 15,
        length_min: float = 0.5**7,
        length_max: float = 1.6,
        length_init: float = 0.8,
        success_tol: int = 3,
        failure_tol: int = 4,
        llm_agent: Optional[OptimizedLLMMetaAgent] = None,
        milestone_steps: Tuple[int, ...] = (16, 35, 65, 105, 140),
        stag_threshold: int = 12,
        min_intervention_interval: int = 20,
    ):
        super().__init__(evaluator, x0, bounds, max_evals, seed, is_meta=True)
        self.n_init = n_init
        self.L_min = length_min
        self.L_max = length_max
        self.L_init = length_init
        self.base_L = length_init
        self.L = length_init
        self.success_tol = success_tol
        self.failure_tol = failure_tol

        self.n_success = 0
        self.n_failure = 0
        self.temperature = 1.0
        self.prob_perturb = min(20.0 / len(self.x0), 1.0)
        self.llm_agent = llm_agent

        self.milestone_steps = set(milestone_steps)
        self.stag_threshold = stag_threshold
        self.min_intervention_interval = min_intervention_interval
        self.last_intervention_step = 0
        self.guide_direction_unit: Optional[np.ndarray] = None
        self.guide_direction_steps_left: int = 0
        self.intervention_history: List[Dict[str, Any]] = []

    def get_state(self) -> np.ndarray:
        return self.get_base_state()

    def apply_action(self, action: np.ndarray):
        pass

    def run(self) -> Dict[str, Any]:
        d = len(self.x0)
        dim_range = self.upper_bounds - self.lower_bounds

        def to_unit(x):
            return (x - self.lower_bounds) / dim_range

        def from_unit(u):
            return self.lower_bounds + u * dim_range

        X_train = []
        Y_train = []

        # 1. 初始化基准 NACA0012
        score0 = self.evaluate(self.x0)
        self.initial_score = score0
        X_train.append(to_unit(self.x0))
        Y_train.append(score0)

        # 初始微扰散点 (共 n_init 个点)
        for _ in range(1, self.n_init):
            if self.eval_count >= self.max_evals:
                break
            noise = self.rng.uniform(-0.03, 0.03, size=d)
            x_cand = np.clip(self.x0 + noise, self.lower_bounds, self.upper_bounds)
            score = self.evaluate(x_cand)
            X_train.append(to_unit(x_cand))
            Y_train.append(score)

        # 2. TuRBO 主优化循环
        while self.eval_count < self.max_evals:
            score_before = self.best_score
            step = self.eval_count + 1

            # 判断是否触发 LLM 元优化干预 (严格限定 step <= 150 前中期)
            need_intervention = (
                (step <= 150)
                and (
                    (step in self.milestone_steps)
                    or (
                        self.stagnation_counter >= self.stag_threshold
                        and (step - self.last_intervention_step) >= self.min_intervention_interval
                    )
                )
            )

            if need_intervention and self.llm_agent is not None:
                self.last_intervention_step = step
                det = self.evaluator.evaluate_detailed(self.best_x)

                state_dict = {
                    "airfoil_name": "NACA0012",
                    "reynolds": self.evaluator.Re,
                    "alpha": self.evaluator.alpha,
                    "step": step,
                    "max_evals": self.max_evals,
                    "budget_left": self.max_evals - self.eval_count,
                    "dim": d,
                    "init_ld": self.initial_score,
                    "best_ld": self.best_score,
                    "best_cl": det.get("cl", 0.0),
                    "best_cd": det.get("cd", 0.0),
                    "best_t_max": det.get("t_max", 0.0),
                    "stagnation_count": self.stagnation_counter,
                    "tr_length": self.L,
                    "success_counter": self.n_success,
                    "success_tolerance": self.success_tol,
                }

                action = self.llm_agent.decide_action(state_dict)
                self.intervention_history.append({"step": step, "action": action})

                scale_L = action["length_scale_factor"]
                self.base_L = float(np.clip(self.base_L * scale_L, self.L_min, self.L_max))
                self.prob_perturb = action["prob_perturb"]
                self.temperature = action["exploration_weight"]

                strat = action.get("directional_strategy", {})
                delta_dir = convert_directional_strategy(self.best_x, strat)

                # 自适应多尺度工业几何可行性前置检验 (1.0 -> 0.5 -> 0.25)
                cand_to_eval = None
                chosen_delta = None
                for s_fac in [1.0, 0.5, 0.25]:
                    trial_x = np.clip(self.best_x + s_fac * delta_dir, self.lower_bounds, self.upper_bounds)
                    try:
                        _, k_af = self.evaluator._theta_to_kulfan(trial_x)
                        is_fatal, _, metrics = self.evaluator.validate_geometry(k_af)
                        is_feasible, total_penalty, _, _ = self.evaluator.calculate_penalties(metrics)
                        t_max = metrics.get("t_max", 0.12)
                        if not is_fatal and is_feasible and total_penalty == 0.0 and 0.09 <= t_max <= 0.19:
                            cand_to_eval = trial_x
                            chosen_delta = s_fac * delta_dir
                            break
                    except Exception:
                        pass

                if cand_to_eval is not None and chosen_delta is not None:
                    # 设定方向引导向量 (在 unit 空间)
                    u_best = to_unit(self.best_x)
                    u_cand = to_unit(cand_to_eval)
                    diff_u = u_cand - u_best
                    norm_u = np.linalg.norm(diff_u)
                    if norm_u > 1e-4:
                        self.guide_direction_unit = diff_u / norm_u
                        self.guide_direction_steps_left = 20

                    # 直接作为高优先级候选点进行一次真实求值
                    cand_score = self.evaluate(cand_to_eval)
                    X_train.append(to_unit(cand_to_eval))
                    Y_train.append(cand_score)

                    print(
                        f"      [LLM-Meta @ Step {step:3d}] Scale: {scale_L:.2f}, L: {self.base_L:.4f}, "
                        f"Perturb: {self.prob_perturb:.2f}, Temp: {self.temperature:.2f} | "
                        f"Dir Eval: {cand_score:.2f} (Best: {self.best_score:.2f}) | "
                        f"Thought: {action.get('thought', '')[:60]}...",
                        flush=True,
                    )

                    if cand_score > score_before + 1e-3:
                        self.n_success += 1
                        self.n_failure = 0
                    else:
                        self.n_failure += 1
                        self.n_success = 0

                    if self.eval_count >= self.max_evals:
                        break
                    continue

                print(
                    f"      [LLM-Meta @ Step {step:3d}] Scale: {scale_L:.2f}, L: {self.base_L:.4f}, "
                    f"Perturb: {self.prob_perturb:.2f}, Temp: {self.temperature:.2f} | "
                    f"Thought: {action.get('thought', '')[:60]}...",
                    flush=True,
                )

            self.L = self.base_L

            # 3. 高斯过程建模与 Thompson 候选采样
            X_arr = np.array(X_train, dtype=np.float64)
            Y_arr = np.array(Y_train, dtype=np.float64)
            Y_gp = np.maximum(Y_arr, 0.0)
            y_mean = float(np.mean(Y_gp))
            y_std = float(max(np.std(Y_gp), 1e-4))
            Y_norm = (Y_gp - y_mean) / y_std

            n_pts = len(X_arr)
            K = matern52_kernel(X_arr, X_arr, lengthscale=0.30) + 1e-4 * np.eye(n_pts, dtype=np.float64)
            try:
                L_chol = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-2 * np.eye(n_pts, dtype=np.float64)
                L_chol = np.linalg.cholesky(K)

            z_chol = solve_triangular(L_chol, Y_norm, lower=True)
            alpha = solve_triangular(L_chol.T, z_chol, lower=False)

            x_best_unit = to_unit(self.best_x)
            tr_lb = np.clip(x_best_unit - self.L / 2.0, 0.0, 1.0)
            tr_ub = np.clip(x_best_unit + self.L / 2.0, 0.0, 1.0)

            n_candidates = 800
            cand_samples = self.rng.uniform(tr_lb, tr_ub, size=(n_candidates, d))
            mask = self.rng.rand(n_candidates, d) > self.prob_perturb
            mask[self.rng.rand(n_candidates) < 0.1, :] = False
            cand_samples[mask] = np.broadcast_to(x_best_unit, (n_candidates, d))[mask]

            # 若当前存在活跃的方向引导向量，将前 350 个候选沿引导方向倾斜采样 (仅在 step <= 150)
            if (
                self.guide_direction_unit is not None
                and self.guide_direction_steps_left > 0
                and step <= 150
            ):
                n_dir = 350
                drift = self.rng.uniform(0.1 * self.L, 1.0 * self.L, size=(n_dir, 1))
                cand_samples[:n_dir] = np.clip(
                    x_best_unit + drift * self.guide_direction_unit + self.rng.normal(0, 0.02 * self.L, size=(n_dir, d)),
                    0.0,
                    1.0,
                )
                self.guide_direction_steps_left -= 1

            K_star = matern52_kernel(cand_samples, X_arr, lengthscale=0.30)
            mu = K_star @ alpha
            v = solve_triangular(L_chol, K_star.T, lower=True)
            var = np.clip(1.0 - np.sum(v**2, axis=0), 1e-6, None)
            sigma = np.sqrt(var)

            z_sample = self.rng.randn(n_candidates)
            thompson_samples = mu + self.temperature * sigma * z_sample
            best_cand_idx = int(np.argmax(thompson_samples))

            next_x_unit = cand_samples[best_cand_idx]
            next_x = from_unit(next_x_unit)

            new_score = self.evaluate(next_x)
            X_train.append(next_x_unit)
            Y_train.append(new_score)

            # 更新 TuRBO 状态
            if new_score > score_before + 1e-3:
                self.n_success += 1
                self.n_failure = 0
            else:
                self.n_failure += 1
                self.n_success = 0

            if self.n_success >= self.success_tol:
                self.base_L = min(self.base_L * 2.0, self.L_max)
                self.n_success = 0
            elif self.n_failure >= self.failure_tol:
                self.base_L = self.base_L / 2.0
                if self.base_L < self.L_min * 1.5:
                    self.base_L = self.L_init
                self.n_failure = 0

        while len(self.history_best) < self.max_evals:
            self.history_best.append(self.best_score)

        return {
            "best_x": self.best_x,
            "best_score": self.best_score,
            "history": self.history_best[:self.max_evals],
            "interventions": self.intervention_history,
        }


# ==============================================================================
# 3. MALIBO 驱动包装函数
# ==============================================================================

def run_malibo_single_seed(
    seed: int,
    reynolds: float = 3.5e6,
    alpha: float = 3.5,
    max_evals: int = 300,
    checkpoint_path: str = "references/Meta-CFD/malibo/malibo_checkpoint.pt",
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
# 4. LLM-Meta-TuRBO 驱动包装函数
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
# 5. 全流程基准运行与持久化
# ==============================================================================

def run_full_benchmark(
    base_benchmark_path: str = "results/benchmark_results.json",
    output_json_path: str = "results/llm_malibo_benchmark_results.json",
    output_npz_path: str = "results/llm_malibo_benchmark_results.npz",
    checkpoint_path: str = "references/Meta-CFD/malibo/malibo_checkpoint.pt",
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
