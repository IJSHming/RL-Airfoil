"""LLM-Meta-TuRBO 翼型优化器：
基于大语言模型宏观定向元控制的高维信赖域贝叶斯优化器 (TuRBO-1)。
结合气动学专家先验与宏观几何变形注入，在保持基准 TuRBO 超参对齐的同时，
显著提升前中期局部探索效率与收敛品质。
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.linalg import solve_triangular

from src.evaluator import AerodynamicEvaluator
from src.llm_agent import OptimizedLLMMetaAgent
from src.optimizers.base import BaseAirfoilOptimizer
from src.optimizers.turbo import matern52_kernel


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
