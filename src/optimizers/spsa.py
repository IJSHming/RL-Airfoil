"""SPSA (Simultaneous Perturbation Stochastic Approximation) 翼型优化器：
支持 Vanilla SPSA 与 RL-Meta-SPSA（动态步长与摄动尺度调节）。
采用 Spall 经典标准准则，并基于理论衰减调度曲线进行 RL 锚定弹性缩放。
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from src.optimizers.base import BaseAirfoilOptimizer
from src.evaluator import AerodynamicEvaluator


class SPSAOptimizer(BaseAirfoilOptimizer):
    """SPSA 优化器。每次迭代消耗 2 次评估。总迭代轮次约为 150 代 (300 NFE)。"""

    def __init__(
        self,
        evaluator: AerodynamicEvaluator,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        max_evals: int = 300,
        seed: int = 42,
        is_meta: bool = False,
        rl_policy: Optional[Any] = None,
        a0: float = 0.0008,
        c0: float = 0.020,
        alpha: float = 0.602,
        gamma: float = 0.101,
        momentum: float = 0.75,
    ):
        super().__init__(evaluator, x0, bounds, max_evals, seed, is_meta, rl_policy)
        self.a0 = a0
        self.c0 = c0
        self.alpha_exp = alpha
        self.gamma_exp = gamma
        self.momentum = momentum

        # 动态控制量
        self.current_a = a0
        self.current_c = c0
        self.last_grad = np.zeros_like(self.x0)
        self.current_grad = np.zeros_like(self.x0)
        self.m = np.zeros_like(self.x0)
        self.iter_k = 0

    def get_state(self) -> np.ndarray:
        """SPSA 状态空间 (10维) = 6维通用状态 + 4维 SPSA 专有动态特征"""
        base = self.get_base_state()

        log_a = np.clip(np.log(max(self.current_a / self.a0, 1e-4)) / 2.0, -2.0, 2.0)
        log_c = np.clip(np.log(max(self.current_c / self.c0, 1e-4)) / 2.0, -2.0, 2.0)
        grad_norm = min(np.linalg.norm(self.current_grad) / 50.0, 2.0)

        # 连续两步梯度方向余弦相似度
        norm_prod = (np.linalg.norm(self.current_grad) * np.linalg.norm(self.last_grad)) + 1e-8
        grad_cos = np.dot(self.current_grad, self.last_grad) / norm_prod
        grad_cos = np.clip(grad_cos, -1.0, 1.0)

        specific = np.array([log_a, log_c, grad_norm, grad_cos], dtype=np.float32)
        return np.concatenate([base, specific])

    def apply_action(self, action: np.ndarray, vanilla_a: float = None, vanilla_c: float = None):
        """Action (2维) -> 锚定 Spall 理论衰减基准进行动态弹性调节，彻底废除无界指数累乘"""
        if vanilla_a is None:
            vanilla_a = self.a0 / (1.0 + self.iter_k / 20.0) ** self.alpha_exp
        if vanilla_c is None:
            vanilla_c = self.c0 / (1.0 + self.iter_k) ** self.gamma_exp

        scale_a = float(np.exp(np.clip(action[0], -1.0, 1.0) * 0.40))
        scale_c = float(np.exp(np.clip(action[1], -1.0, 1.0) * 0.40))

        self.current_a = float(np.clip(vanilla_a * scale_a, 1e-5, 0.008))
        self.current_c = float(np.clip(vanilla_c * scale_c, 1e-4, 0.040))

    def run(self) -> Dict[str, Any]:
        self.evaluate_initial()
        current_x = self.x0.copy()
        d = len(current_x)

        while self.eval_count < self.max_evals:
            self.iter_k += 1
            # 标准 Vanilla 调度衰减 (Spall 准则)
            vanilla_a = self.a0 / (1.0 + self.iter_k / 20.0) ** self.alpha_exp
            vanilla_c = self.c0 / (1.0 + self.iter_k) ** self.gamma_exp

            if not self.is_meta:
                self.current_a = vanilla_a
                self.current_c = vanilla_c

            state = self.get_state()
            action = None
            if self.is_meta and self.rl_policy is not None:
                action = self.rl_policy(state)
                self.apply_action(action, vanilla_a, vanilla_c)

            # 伯努利随机摄动向量 Delta in {-1, +1}^d
            delta = self.rng.choice([-1.0, 1.0], size=d)
            x_plus = current_x + self.current_c * delta
            x_minus = current_x - self.current_c * delta

            score_before = self.best_score
            f_plus = self.evaluate(x_plus)
            if self.eval_count >= self.max_evals:
                break
            f_minus = self.evaluate(x_minus)

            # 随机梯度估计 (目标是最大化升阻比)并进行梯度范数截断保护
            self.last_grad = self.current_grad.copy()
            diff = f_plus - f_minus
            grad_raw = (diff / (2.0 * self.current_c)) * delta
            gnorm = float(np.linalg.norm(grad_raw))
            if gnorm > 200.0:
                self.current_grad = grad_raw * (200.0 / max(gnorm, 1e-8))
            else:
                self.current_grad = grad_raw

            # 梯度平滑与动量更新
            self.m = self.momentum * self.m + (1.0 - self.momentum) * self.current_grad
            current_x = np.clip(
                current_x + self.current_a * self.m,
                self.lower_bounds,
                self.upper_bounds,
            )

            # 记录 RL 转移
            if self.is_meta and action is not None:
                next_state = self.get_state()
                done = (self.eval_count >= self.max_evals)
                # 跨工况自适应归一化稠密奖励
                rel_gain = (self.best_score - score_before) / max(self.initial_score, 15.0)
                reward = rel_gain * 20.0 - 0.002
                if done:
                    reward += max((self.best_score - self.initial_score) / 50.0, 0.0) * 0.1
                self.transitions.append({
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done,
                })

        # 补全至 300 步轨迹
        while len(self.history_best) < self.max_evals:
            self.history_best.append(self.best_score)

        return {
            "best_x": self.best_x,
            "best_score": self.best_score,
            "history": self.history_best[:self.max_evals],
            "transitions": self.transitions,
        }
