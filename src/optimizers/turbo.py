"""TuRBO (Trust Region Bayesian Optimization) 翼型优化器：
支持 Vanilla TuRBO 与 RL-Meta-TuRBO（自适应动态信赖域空间尺度与采样探索温度）。
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from src.optimizers.base import BaseAirfoilOptimizer
from src.evaluator import AerodynamicEvaluator


def matern52_kernel(x1: torch.Tensor, x2: torch.Tensor, lengthscale: float = 0.25, outputscale: float = 1.0) -> torch.Tensor:
    """Matérn 5/2 协方差核函数计算"""
    dist = torch.cdist(x1 / lengthscale, x2 / lengthscale)
    sqrt5 = 2.23606797749979
    val = (1.0 + sqrt5 * dist + (5.0 / 3.0) * dist**2) * torch.exp(-sqrt5 * dist)
    return outputscale * val


class TuRBOOptimizer(BaseAirfoilOptimizer):
    """高维信赖域贝叶斯优化器 (TuRBO-1)。高斯过程代理模型 + 汤普森采样，严格 300 次 NFE。"""

    def __init__(
        self,
        evaluator: AerodynamicEvaluator,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        max_evals: int = 300,
        seed: int = 42,
        is_meta: bool = False,
        rl_policy: Optional[Any] = None,
        n_init: int = 15,
        length_min: float = 0.5**7,
        length_max: float = 1.6,
        length_init: float = 0.8,
        success_tol: int = 3,
        failure_tol: int = 4,
    ):
        super().__init__(evaluator, x0, bounds, max_evals, seed, is_meta, rl_policy)
        self.n_init = n_init
        self.L_min = length_min
        self.L_max = length_max
        self.L_init = length_init
        self.L = length_init
        self.success_tol = success_tol
        self.failure_tol = failure_tol

        self.n_success = 0
        self.n_failure = 0
        self.temperature = 1.0
        self.mean_gp_uncertainty = 1.0

    def get_state(self) -> np.ndarray:
        """TuRBO 状态空间 (10维) = 6维通用状态 + 4维 TuRBO 专有状态"""
        base = self.get_base_state()

        log_L = np.clip(np.log(max(self.L / self.L_init, 1e-4)) / 2.0, -2.0, 2.0)
        norm_succ = min(self.n_success / 5.0, 2.0)
        norm_fail = min(self.n_failure / 5.0, 2.0)
        norm_unc = min(self.mean_gp_uncertainty / 5.0, 2.0)

        specific = np.array([log_L, norm_succ, norm_fail, norm_unc], dtype=np.float32)
        return np.concatenate([base, specific])

    def apply_action(self, action: np.ndarray):
        """Action (2维) -> [信赖域尺度乘子, 汤普森采样探索温度]"""
        scale_L = float(np.exp(np.clip(action[0], -1.5, 1.5)))
        self.L = float(np.clip(self.L * scale_L, self.L_min, self.L_max))
        self.temperature = float(np.exp(np.clip(action[1], -1.2, 1.2)))

    def run(self) -> Dict[str, Any]:
        d = len(self.x0)
        dim_range = self.upper_bounds - self.lower_bounds

        # 归一化搜索空间 [0, 1]^d 便于 GP 建模
        def to_unit(x):
            return (x - self.lower_bounds) / dim_range

        def from_unit(u):
            return self.lower_bounds + u * dim_range

        # 1. 初始化采样点
        X_train = []
        Y_train = []

        # 基准 NACA0012
        score0 = self.evaluate(self.x0)
        self.initial_score = score0
        X_train.append(to_unit(self.x0))
        Y_train.append(score0)

        # 初始微扰散点
        for _ in range(1, self.n_init):
            if self.eval_count >= self.max_evals:
                break
            noise = self.rng.uniform(-0.03, 0.03, size=d)
            x_cand = np.clip(self.x0 + noise, self.lower_bounds, self.upper_bounds)
            score = self.evaluate(x_cand)
            X_train.append(to_unit(x_cand))
            Y_train.append(score)

        # 2. TuRBO 主循环
        while self.eval_count < self.max_evals:
            score_before = self.best_score
            state = self.get_state()
            action = None
            if self.is_meta and self.rl_policy is not None:
                action = self.rl_policy(state)
                self.apply_action(action)

            # 转换为 PyTorch 张量
            X_tensor = torch.tensor(np.array(X_train), dtype=torch.float64)
            Y_arr = np.array(Y_train)
            # 目标标准化
            y_mean = float(np.mean(Y_arr))
            y_std = float(max(np.std(Y_arr), 1e-4))
            Y_norm = torch.tensor((Y_arr - y_mean) / y_std, dtype=torch.float64).unsqueeze(1)

            # 计算核矩阵并进行 Cholesky 分解
            K = matern52_kernel(X_tensor, X_tensor, lengthscale=0.30) + 1e-4 * torch.eye(len(X_tensor), dtype=torch.float64)
            try:
                L_chol = torch.linalg.cholesky(K)
                alpha = torch.cholesky_solve(Y_norm, L_chol)
            except Exception:
                # 若矩阵病态，加入更大对角正则
                K += 1e-2 * torch.eye(len(X_tensor), dtype=torch.float64)
                L_chol = torch.linalg.cholesky(K)
                alpha = torch.cholesky_solve(Y_norm, L_chol)

            # 生成信赖域边界 (围绕当前最优解 x_best)
            x_best_unit = to_unit(self.best_x)
            tr_lb = np.clip(x_best_unit - self.L / 2.0, 0.0, 1.0)
            tr_ub = np.clip(x_best_unit + self.L / 2.0, 0.0, 1.0)

            # 候选点生成 (1000 个带坐标扰动的局部候选)
            n_candidates = 800
            cand_samples = self.rng.uniform(tr_lb, tr_ub, size=(n_candidates, d))
            # TuRBO 核心坐标扰动技巧：并非所有维度都变动
            prob_perturb = min(20.0 / d, 1.0)
            mask = self.rng.rand(n_candidates, d) > prob_perturb
            mask[self.rng.rand(n_candidates) < 0.1, :] = False  # 保证至少 10% 候选全维度扰动
            cand_samples[mask] = np.broadcast_to(x_best_unit, (n_candidates, d))[mask]

            X_cand = torch.tensor(cand_samples, dtype=torch.float64)

            # 计算候选点的后验均值与方差
            with torch.no_grad():
                K_star = matern52_kernel(X_cand, X_tensor, lengthscale=0.30)
                mu = (K_star @ alpha).squeeze(1).numpy()
                v = torch.linalg.solve_triangular(L_chol, K_star.T, upper=False)
                var = torch.clamp(1.0 - torch.sum(v**2, dim=0), min=1e-6).numpy()
                sigma = np.sqrt(var)

            self.mean_gp_uncertainty = float(np.mean(sigma) * y_std)

            # 汤普森采样选择最佳候选
            # 引入 RL 调节的探索温度 temperature
            z = self.rng.randn(n_candidates)
            thompson_samples = mu + self.temperature * sigma * z
            best_cand_idx = int(np.argmax(thompson_samples))

            next_x_unit = cand_samples[best_cand_idx]
            next_x = from_unit(next_x_unit)
            new_score = self.evaluate(next_x)

            X_train.append(next_x_unit)
            Y_train.append(new_score)

            # 更新信赖域长度
            if new_score > score_before + 1e-3:
                self.n_success += 1
                self.n_failure = 0
            else:
                self.n_failure += 1
                self.n_success = 0

            if self.n_success >= self.success_tol:
                self.L = min(self.L * 2.0, self.L_max)
                self.n_success = 0
            elif self.n_failure >= self.failure_tol:
                self.L = max(self.L / 2.0, self.L_min)
                self.n_failure = 0

            # 记录经验
            if self.is_meta and action is not None:
                next_state = self.get_state()
                reward = (self.best_score - score_before) / 10.0 - 0.01
                done = (self.eval_count >= self.max_evals)
                self.transitions.append({
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done,
                })

        while len(self.history_best) < self.max_evals:
            self.history_best.append(self.best_score)

        return {
            "best_x": self.best_x,
            "best_score": self.best_score,
            "history": self.history_best[:self.max_evals],
            "transitions": self.transitions,
        }
