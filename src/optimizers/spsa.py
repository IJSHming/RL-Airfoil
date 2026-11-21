"""SPSA (Simultaneous Perturbation Stochastic Approximation) 翼型优化器：
支持 Vanilla SPSA 与 RL-Meta-SPSA（动态步长、摄动尺度与步长信任域裁剪界调节）。
遵循 Spall (1998) 经典权威准则，进行参数空间跨度自适应比例缩放与初始梯度校准。
"""

from typing import Any, Dict, List, Optional, Tuple
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
        max_step_rel: float = 0.10,
        momentum: float = 0.0,
    ):
        super().__init__(evaluator, x0, bounds, max_evals, seed, is_meta, rl_policy)
        self.c_rel = c_rel
        self.step_rel = step_rel
        self.alpha_exp = alpha
        self.gamma_exp = gamma
        self.max_step_rel = max_step_rel
        self.momentum = momentum
        self.span = self.upper_bounds - self.lower_bounds

        # 动态控制量
        self.current_a = step_rel
        self.current_c = c_rel * self.span
        self.current_max_step_rel = max_step_rel
        self.last_vanilla_a = step_rel
        self.last_raw_rel_step = 0.0
        self.recent_clip_fracs: List[float] = []
        self.last_grad = np.zeros_like(self.x0)
        self.current_grad = np.zeros_like(self.x0)
        self.m = np.zeros_like(self.x0)
        self.iter_k = 0

    def get_state(self) -> np.ndarray:
        """SPSA 状态空间 (10维) = 6维通用状态 + 4维量纲无关的 SPSA 专有动态特征"""
        base = self.get_base_state()

        # 1. Vanilla 调度步长 a_k 的对数尺度 (反映 Spall 初始探测校准后的量级差异)
        log_vanilla_a = float(np.clip((np.log10(max(self.last_vanilla_a, 1e-8)) + 3.5) / 1.5, -2.0, 2.0))
        # 2. 近期评估可行率 (捕捉不可行悬崖区域)
        feas_rate = self.get_recent_feasibility_rate(window=12)
        # 3. 近期步长裁剪饱和率 (反映步长是否频繁撞击信任域裁剪边界)
        clip_rate = float(np.mean(self.recent_clip_fracs[-8:])) if self.recent_clip_fracs else 0.0
        # 4. 裁剪前中位相对步长幅值 (对单一极小 span 维度鲁棒，反映真实步长过小或过冲)
        log_rel_step = float(np.clip(np.log10(1.0 + 20.0 * self.last_raw_rel_step) / 2.0, 0.0, 2.0))

        specific = np.array([log_vanilla_a, feas_rate, clip_rate, log_rel_step], dtype=np.float32)
        return np.concatenate([base, specific])

    def apply_action(self, action: np.ndarray, vanilla_a: float, vanilla_c: np.ndarray):
        """Action (3维) -> 纯净 RL 策略动态调节 (a_k, c_k, max_step_k) 乘子 (action=0 时精确退化为 Vanilla SPSA)"""
        a1 = float(np.clip(action[0], -1.0, 1.0))
        a2 = float(np.clip(action[1], -1.0, 1.0)) if len(action) > 1 else 0.0
        a3 = float(np.clip(action[2], -1.0, 1.0)) if len(action) > 2 else 0.0
        scale_a = float(np.exp(a1 * 1.2))
        scale_c = float(np.exp(a2 * 0.8))
        scale_clip = float(np.exp(a3 * 1.4))

        self.current_a = float(np.clip(vanilla_a * scale_a, 1e-8, 1.0))
        self.current_c = np.clip(vanilla_c * scale_c, 1e-4 * self.span, 0.25 * self.span)
        self.current_max_step_rel = float(np.clip(self.max_step_rel * scale_clip, 0.005, 0.50))

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
                self.current_max_step_rel = self.max_step_rel
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

            # 步长更新与动态信任域/物理约束边界截断
            raw_step = self.current_a * span * grad
            rel_raw = np.abs(raw_step) / np.maximum(span, 1e-8)
            self.last_raw_rel_step = float(np.median(rel_raw))
            max_step = self.current_max_step_rel * span
            self.recent_clip_fracs.append(float(np.mean(np.abs(raw_step) > max_step)))
            step = np.clip(raw_step, -max_step, max_step)
            current_x = np.clip(
                current_x + step,
                self.lower_bounds,
                self.upper_bounds,
            )

            # 记录经验 (附带 eval_count 以启用精确势能差分轨迹配对奖励，消除长时域 GAE 折扣衰减)
            if self.is_meta and action is not None:
                next_state = self.get_state()
                done = (self.eval_count >= self.max_evals)
                self.transitions.append({
                    "state": state,
                    "action": action,
                    "reward": float(self.best_score - score_before),
                    "next_state": next_state,
                    "done": done,
                    "eval_count": int(self.eval_count),
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
