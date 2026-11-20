"""LLM-Meta-TuRBO 翼型优化器：
基于本地大语言模型停滞自适应元控制的高维信赖域贝叶斯优化器 (TuRBO-1)。
与 3D 风力机叶片实验保持完全一致的工程与科研公平性设计：
1. 与 Vanilla-TuRBO / RL-Meta-TuRBO 严格对齐基础超参:
   n_init = 15, L_init = 0.8, L_min = 0.5**7, L_max = 1.6, success_tol = 3, failure_tol = 4;
2. 仅在第 16 步执行一次 post_init 初始审视，之后完全由真实停滞计数器触发 (stag_threshold = 10,
   min_intervention_interval = 18, step <= 180)，消除硬编码多节点同步阶跃；
3. LLM 的 length_scale_factor 作为临时乘子叠加在 TuRBO 原生 base_L 状态机上 (12 步衰减)，不破坏原生状态机；
4. 采用正交平滑物理基函数 (对称中弦弯度、后弦弯度、对称厚度、前缘半径)，零硬编码特定构型斜坡；
5. 宏观方向偏移仅经由评估器自身解析几何校验进行步长回退 (1.0/0.5/0.25，零气动求值，无额外隐藏厚度过滤)，
   且仅在 30% 信赖域候选切片内注入有界方向倾斜 (严格裁剪于当前信赖域 [tr_lb, tr_ub] 内)。
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.linalg import solve_triangular

from src.evaluator import AerodynamicEvaluator
from src.llm_agent import OptimizedLLMMetaAgent
from src.optimizers.base import BaseAirfoilOptimizer
from src.optimizers.turbo import matern52_kernel


def convert_directional_strategy(strat: Optional[Dict[str, Any]]) -> np.ndarray:
    """将 LLM 输出的对称正交宏观气动策略转换为 18 维 Kulfan 参数偏移量 delta."""
    delta = np.zeros(18, dtype=np.float64)
    if not strat or not isinstance(strat, dict):
        return delta

    s_camber = float(strat.get("camber_shift", 0.0))
    s_aft = float(strat.get("aft_camber_shift", 0.0))
    s_thick = float(strat.get("thickness_shift", 0.0))
    s_le = float(strat.get("le_radius_shift", 0.0))

    # 8 阶 Bernstein 坐标位置 u in (0, 1)
    u = (np.arange(8, dtype=np.float64) + 0.5) / 8.0
    # 1. 对称中弦钟形包络 (峰值在 50% 弦长，两端光滑趋零)
    bell = np.sin(np.pi * u)
    # 2. 光滑后弦加载包络 (峰值在 ~67% 弦长，前缘趋零避免破坏吸力峰)
    aft_bell = (u ** 2) * np.sin(np.pi * u)
    aft_bell = aft_bell / max(float(np.max(aft_bell)), 1e-8)

    camber_profile = s_camber * bell + s_aft * aft_bell
    thick_profile = 0.5 * s_thick * bell

    delta[0:8] = camber_profile + thick_profile
    delta[8:16] = camber_profile - thick_profile
    delta[16] = s_le
    delta[17] = 0.0
    return delta


class LLMMetaTuRBOOptimizer(BaseAirfoilOptimizer):
    """基于大语言模型停滞自适应元控制的高维信赖域贝叶斯优化器 (LLM-Meta-TuRBO)."""

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
        milestone_steps: Tuple[int, ...] = (16,),
        stag_threshold: int = 10,
        min_intervention_interval: int = 18,
    ):
        super().__init__(evaluator, x0, bounds, max_evals, seed, is_meta=True)
        self.n_init = int(n_init)
        self.L_min = float(length_min)
        self.L_max = float(length_max)
        self.L_init = float(length_init)
        self.base_L = float(length_init)
        self.scale_L_mult = 1.0
        self.scale_L_steps_left = 0
        self.L = float(length_init)
        self.success_tol = int(success_tol)
        self.failure_tol = int(failure_tol)

        self.n_success = 0
        self.n_failure = 0
        self.temperature = 1.0
        self.prob_perturb = min(20.0 / len(self.x0), 1.0)
        self.llm_agent = llm_agent

        self.milestone_steps = set(milestone_steps)
        self.stag_threshold = int(stag_threshold)
        self.min_intervention_interval = int(min_intervention_interval)
        self.last_intervention_step = 0
        self.guide_direction_unit: Optional[np.ndarray] = None
        self.guide_direction_steps_left: int = 0
        self.intervention_history: List[Dict[str, Any]] = []

    def get_state(self) -> np.ndarray:
        return self.get_base_state()

    def apply_action(self, action: np.ndarray):
        pass

    def _adjust_trust_region(self, improved: bool) -> None:
        """更新 TuRBO 原生信赖域状态机，并按步衰减临时 LLM 尺度乘子."""
        if improved:
            self.n_success += 1
            self.n_failure = 0
        else:
            self.n_success = 0
            self.n_failure += 1

        if self.n_success >= self.success_tol:
            self.base_L = min(self.base_L * 2.0, self.L_max)
            self.n_success = 0
        elif self.n_failure >= self.failure_tol:
            self.base_L = self.base_L / 2.0
            if self.base_L < self.L_min * 1.5:
                self.base_L = self.L_init
                self.scale_L_mult = 1.0
                self.scale_L_steps_left = 0
            self.n_failure = 0

        if self.scale_L_steps_left > 0:
            self.scale_L_steps_left -= 1
            if self.scale_L_steps_left == 0:
                self.scale_L_mult = 1.0

        self.L = float(np.clip(self.base_L * self.scale_L_mult, self.L_min, self.L_max))

    def run(self) -> Dict[str, Any]:
        d = len(self.x0)
        dim_range = self.upper_bounds - self.lower_bounds

        def to_unit(x):
            return (x - self.lower_bounds) / dim_range

        def from_unit(u):
            return self.lower_bounds + u * dim_range

        X_train: List[np.ndarray] = []
        Y_train: List[float] = []

        # 1. 初始化基准 NACA0012 (与 Vanilla-TuRBO / RL-Meta-TuRBO 比特级对齐)
        score0 = self.evaluate(self.x0)
        self.initial_score = score0
        X_train.append(to_unit(self.x0))
        Y_train.append(score0)

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

            is_milestone = (step in self.milestone_steps)
            is_stagnation = (
                self.stagnation_counter >= self.stag_threshold
                and (step - self.last_intervention_step) >= self.min_intervention_interval
            )
            need_intervention = (step <= 180) and (is_milestone or is_stagnation)

            if need_intervention and self.llm_agent is not None:
                self.last_intervention_step = step
                trigger_reason = "post_init" if (is_milestone and not is_stagnation and step == 16) else "stagnation"
                det = self.evaluator.evaluate_detailed(self.best_x)

                state_dict = {
                    "airfoil_name": "NACA0012",
                    "reynolds": self.evaluator.Re,
                    "alpha": self.evaluator.alpha,
                    "step": step,
                    "trigger_reason": trigger_reason,
                    "max_evals": self.max_evals,
                    "budget_left": self.max_evals - self.eval_count,
                    "dim": d,
                    "init_ld": self.initial_score,
                    "best_ld": self.best_score,
                    "best_cl": det.get("cl", 0.0),
                    "best_cd": det.get("cd", 0.0),
                    "best_cm": det.get("cm", 0.0),
                    "best_t_max": det.get("t_max", 0.0),
                    "stagnation_count": self.stagnation_counter,
                    "tr_length": self.L,
                    "success_counter": self.n_success,
                    "success_tolerance": self.success_tol,
                }

                action = self.llm_agent.decide_action(state_dict)
                inter_item: Dict[str, Any] = {
                    "step": step,
                    "trigger_reason": trigger_reason,
                    "action": action,
                }
                if hasattr(self.llm_agent, "interaction_log") and len(self.llm_agent.interaction_log) > 0:
                    last_log = self.llm_agent.interaction_log[-1]
                    inter_item["user_prompt"] = last_log.get("user_prompt")
                    inter_item["raw_response"] = last_log.get("raw_response")
                self.intervention_history.append(inter_item)

                # 临时叠加信赖域乘子 (12 步后衰减回 1.0)，不破坏原生 base_L
                scale_L = float(action["length_scale_factor"])
                self.scale_L_mult = scale_L
                self.scale_L_steps_left = 12
                self.L = float(np.clip(self.base_L * self.scale_L_mult, self.L_min, self.L_max))
                self.prob_perturb = float(action["prob_perturb"])
                self.temperature = float(action["exploration_weight"])

                strat = action.get("directional_strategy")
                delta_dir = convert_directional_strategy(strat)

                # 仅使用评估器自身的解析几何可行性校验进行多尺度步长回退 (0 次气动求值，无额外隐藏过滤)
                cand_to_eval = None
                if np.linalg.norm(delta_dir) > 1e-8:
                    for s_fac in (1.0, 0.5, 0.25):
                        trial_x = np.clip(self.best_x + s_fac * delta_dir, self.lower_bounds, self.upper_bounds)
                        try:
                            _, k_af = self.evaluator._theta_to_kulfan(trial_x)
                            is_fatal, _, metrics = self.evaluator.validate_geometry(k_af)
                            is_geom_ok, total_penalty, _, _ = self.evaluator.calculate_penalties(metrics)
                            if not is_fatal and is_geom_ok and total_penalty == 0.0:
                                cand_to_eval = trial_x
                                break
                        except Exception:
                            pass

                if cand_to_eval is not None:
                    u_best = to_unit(self.best_x)
                    u_cand = to_unit(cand_to_eval)
                    diff_u = u_cand - u_best
                    norm_u = float(np.linalg.norm(diff_u))
                    if norm_u > 1e-4:
                        self.guide_direction_unit = diff_u / norm_u
                        self.guide_direction_steps_left = 15

                    cand_score = self.evaluate(cand_to_eval)
                    X_train.append(u_cand)
                    Y_train.append(cand_score)

                    print(
                        f"      [LLM-Meta @ Step {step:3d} ({trigger_reason})] Scale: {scale_L:.2f}, L: {self.L:.4f}, "
                        f"Perturb: {self.prob_perturb:.2f}, Temp: {self.temperature:.2f} | "
                        f"Dir Eval: {cand_score:.2f} (Best: {self.best_score:.2f}) | "
                        f"Thought: {action.get('thought', '')[:60]}...",
                        flush=True,
                    )

                    improved = (cand_score > score_before + 1e-3)
                    self._adjust_trust_region(improved)

                    if self.eval_count >= self.max_evals:
                        break
                    continue

                print(
                    f"      [LLM-Meta @ Step {step:3d} ({trigger_reason})] Scale: {scale_L:.2f}, L: {self.L:.4f}, "
                    f"Perturb: {self.prob_perturb:.2f}, Temp: {self.temperature:.2f} | "
                    f"Thought: {action.get('thought', '')[:60]}...",
                    flush=True,
                )

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

            # 仅在 30% 候选切片内注入有界 LLM 方向性倾斜采样，并严格裁剪在信赖域 [tr_lb, tr_ub] 内
            if (
                self.guide_direction_unit is not None
                and self.guide_direction_steps_left > 0
                and step <= 195
            ):
                n_dir = min(240, n_candidates // 3)
                drift = self.rng.uniform(0.05 * self.L, 0.65 * self.L, size=(n_dir, 1))
                dir_cands = (
                    x_best_unit
                    + drift * self.guide_direction_unit
                    + self.rng.normal(0.0, 0.03 * self.L, size=(n_dir, d))
                )
                cand_samples[:n_dir] = np.clip(dir_cands, tr_lb, tr_ub)
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

            improved = (new_score > score_before + 1e-3)
            self._adjust_trust_region(improved)

        while len(self.history_best) < self.max_evals:
            self.history_best.append(self.best_score)

        return {
            "best_x": self.best_x,
            "best_score": self.best_score,
            "history": self.history_best[:self.max_evals],
            "interventions": self.intervention_history,
        }
