"""SPSA (Simultaneous Perturbation Stochastic Approximation) 翼型优化器：
支持 Vanilla SPSA 与 RL-Meta-SPSA（动态步长与摄动尺度调节）。
遵循 Spall (1998) 经典权威准则，进行参数空间跨度自适应比例缩放与初始梯度校准。
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
        c_rel: float = 0.03,
        step_rel: float = 0.03,
        alpha: float = 0.602,
        gamma: float = 0.101,
        momentum: float = 0.0,
    ):
        super().__init__(evaluator, x0, bounds, max_evals, seed, is_meta, rl_policy)
        self.c_rel = c_rel
        self.step_rel = step_rel
        self.alpha_exp = alpha
        self.gamma_exp = gamma
        self.momentum = momentum
        self.span = self.upper_bounds - self.lower_bounds

        # 动态控制量
        self.current_a = step_rel
        self.current_c = c_rel * self.span
        self.last_vanilla_a = step_rel
        self.last_grad = np.zeros_like(self.x0)
        self.current_grad = np.zeros_like(self.x0)
        self.m = np.zeros_like(self.x0)
        self.iter_k = 0

    def get_state(self) -> np.ndarray:
        """SPSA 状态空间 (10维) = 6维通用状态 + 4维 SPSA 专有动态特征 (无上限饱和，含近期可行率)"""
        base = self.get_base_state()

        log_a_ratio = float(np.clip(np.log(max(self.current_a / max(self.last_vanilla_a, 1e-8), 1e-3)), -2.0, 2.0))
        feas_rate = self.get_recent_feasibility_rate(window=12)
        log_grad_norm = float(np.clip(np.log10(1.0 + np.linalg.norm(self.current_grad)) / 4.0, 0.0, 2.0))

        # 连续两步梯度方向余弦相似度
        norm_prod = (np.linalg.norm(self.current_grad) * np.linalg.norm(self.last_grad)) + 1e-8
        grad_cos = float(np.dot(self.current_grad, self.last_grad) / norm_prod)
        grad_cos = float(np.clip(grad_cos, -1.0, 1.0))

        specific = np.array([log_a_ratio, feas_rate, log_grad_norm, grad_cos], dtype=np.float32)
        return np.concatenate([base, specific])

    def apply_action(self, action: np.ndarray, vanilla_a: float, vanilla_c: np.ndarray):
        """Action (2维) -> 纯净 RL 策略动态调节步长乘子与摄动尺度乘子 (action=0 时精确退化为 Vanilla SPSA)"""
        a1 = float(np.clip(action[0], -1.0, 1.0))
        a2 = float(np.clip(action[1], -1.0, 1.0)) if len(action) > 1 else 0.0
        scale_a = float(np.exp(a1 * 0.45))
        scale_c = float(np.exp(a2 * 0.35))

        self.current_a = float(np.clip(vanilla_a * scale_a, 1e-6, 0.50))
        self.current_c = np.clip(vanilla_c * scale_c, 1e-4 * self.span, 0.25 * self.span)

    def run(self) -> Dict[str, Any]:
        self.evaluate_initial()
        current_x = self.x0.copy()
        d = len(current_x)
        span = self.span

        # 初始 Spall 梯度探测与 a_base 校准
        c_vec = self.c_rel * span
        delta0 = self.rng.choice([-1.0, 1.0], size=d)
        f_p = self.evaluate(current_x + c_vec * delta0)
        f_m = self.evaluate(current_x - c_vec * delta0)
        diff0 = abs(f_p - f_m) + 1e-4
        a_base = self.step_rel * 2.0 * self.c_rel / max(diff0, 0.5)

        while self.eval_count < self.max_evals:
            self.iter_k += 1
            # 标准 Vanilla 调度衰减 (Spall 准则)
            vanilla_a = a_base / (1.0 + self.iter_k / 20.0) ** self.alpha_exp
            vanilla_c = c_vec / (1.0 + self.iter_k) ** self.gamma_exp
            self.last_vanilla_a = vanilla_a

            if self.is_meta and self.rl_policy is not None:
                state = self.get_state()
                action = self.rl_policy(state)
                self.apply_action(action, vanilla_a, vanilla_c)
            else:
                self.current_a = vanilla_a
                self.current_c = vanilla_c
                state = self.get_state()
                action = None

            # 伯努利随机摄动向量 Delta in {-1, +1}^d
            delta = self.rng.choice([-1.0, 1.0], size=d)
            x_plus = current_x + self.current_c * delta
            x_minus = current_x - self.current_c * delta

            score_before = self.best_score
            f_plus = self.evaluate(x_plus)
            if self.eval_count >= self.max_evals:
                break
            f_minus = self.evaluate(x_minus)

            # 相对坐标梯度估计
            self.last_grad = self.current_grad.copy()
            diff = f_plus - f_minus
            grad = (diff / (2.0 * self.current_c)) * delta
            self.current_grad = grad

            # 步长更新与物理约束边界截断
            step = self.current_a * span * grad
            step = np.clip(step, -0.10 * span, 0.10 * span)
            current_x = np.clip(
                current_x + step,
                self.lower_bounds,
                self.upper_bounds,
            )

            # 记录经验 (保存原始最优增量，交由训练端按同工况配对基线做无偏控制变量归一化)
            if self.is_meta and action is not None:
                next_state = self.get_state()
                done = (self.eval_count >= self.max_evals)
                self.transitions.append({
                    "state": state,
                    "action": action,
                    "reward": float(self.best_score - score_before),
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
