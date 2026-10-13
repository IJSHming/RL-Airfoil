"""Py-BOBYQA (Bound Optimization BY Quadratic Approximation) 翼型优化器：
支持 Vanilla BOBYQA 与 RL-Meta-BOBYQA（动态信赖域半径自适应伸缩）。
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pybobyqa
from src.optimizers.base import BaseAirfoilOptimizer
from src.evaluator import AerodynamicEvaluator


class PyBobyqaOptimizer(BaseAirfoilOptimizer):
    """Py-BOBYQA 翼型优化器。基于信赖域二次逼近，单步评估，总预算严格限制为 300 次。"""

    def __init__(
        self,
        evaluator: AerodynamicEvaluator,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        max_evals: int = 300,
        seed: int = 42,
        is_meta: bool = False,
        rl_policy: Optional[Any] = None,
        rhobeg: float = 0.10,
        rhoend: float = 0.0001,
    ):
        super().__init__(evaluator, x0, bounds, max_evals, seed, is_meta, rl_policy)
        self.rhobeg = rhobeg
        self.rhoend = rhoend

        # BOBYQA 专有跟踪状态
        self.current_delta = rhobeg
        self.current_rho = rhobeg
        self.current_ratio = 1.0
        self.model_grad_norm = 0.0

    def get_state(self) -> np.ndarray:
        """Py-BOBYQA 状态空间 (10维) = 6维通用状态 + 4维 BOBYQA 信赖域专有动态特征"""
        base = self.get_base_state()

        log_delta = np.clip(np.log(max(self.current_delta / self.rhobeg, 1e-4)) / 2.0, -2.0, 2.0)
        log_rho = np.clip(np.log(max(self.current_rho / self.rhoend, 1e-4)) / 5.0, -2.0, 2.0)
        norm_ratio = np.clip(self.current_ratio / 2.0, -2.0, 2.0)
        grad_norm = min(self.model_grad_norm / 20.0, 2.0)

        specific = np.array([log_delta, log_rho, norm_ratio, grad_norm], dtype=np.float32)
        return np.concatenate([base, specific])

    def apply_action(self, action: np.ndarray, controller_delta: float, controller_rho: float) -> float:
        """Action (1维) -> 锚定当前控制器 delta 施加弹性缩放，消除无界累乘漂移"""
        scale_delta = float(np.exp(np.clip(action[0], -1.0, 1.0) * 0.35))
        new_delta = float(np.clip(
            controller_delta * scale_delta,
            controller_rho * 1.01,
            self.rhobeg * 4.0,
        ))
        self.current_delta = new_delta
        return new_delta

    def run(self) -> Dict[str, Any]:
        self.evaluate_initial()

        # 定义最小化目标函数 (取负升阻比)
        def objective(x):
            if self.eval_count >= self.max_evals:
                return -self.best_score
            val = self.evaluate(x)
            return -val

        orig_tr_step = pybobyqa.controller.Controller.trust_region_step
        optimizer_instance = self

        def hooked_trust_region_step(controller_self, params):
            if optimizer_instance.eval_count >= optimizer_instance.max_evals:
                return orig_tr_step(controller_self, params)

            # 更新当前控制器的参数观测
            optimizer_instance.current_delta = float(controller_self.delta)
            optimizer_instance.current_rho = float(controller_self.rho)

            # RL-Meta 介入调节
            action = None
            if optimizer_instance.is_meta and optimizer_instance.rl_policy is not None:
                state = optimizer_instance.get_state()
                score_before = optimizer_instance.best_score
                action = optimizer_instance.rl_policy(state)
                # 动态锚定调节 delta
                controller_self.delta = optimizer_instance.apply_action(
                    action,
                    float(controller_self.delta),
                    float(controller_self.rho),
                )

            d, gopt, H, gnew, crvmin = orig_tr_step(controller_self, params)
            optimizer_instance.model_grad_norm = float(np.linalg.norm(gopt))

            # 记录经验 (跨工况自适应归一化稠密奖励)
            if optimizer_instance.is_meta and action is not None:
                next_state = optimizer_instance.get_state()
                done = (optimizer_instance.eval_count >= optimizer_instance.max_evals)
                rel_gain = (optimizer_instance.best_score - score_before) / max(optimizer_instance.initial_score, 15.0)
                reward = rel_gain * 30.0 - 0.002
                if optimizer_instance.best_score > 150.0 and score_before <= 150.0:
                    reward += 1.0  # 突破 150 基准
                if optimizer_instance.best_score > 175.0 and score_before <= 175.0:
                    reward += 2.0  # 突破 175 卓越水平
                if done:
                    reward += max((optimizer_instance.best_score - optimizer_instance.initial_score) / 50.0, 0.0) * 0.25
                optimizer_instance.transitions.append({
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done,
                })

            return d, gopt, H, gnew, crvmin

        # 安装动态钩子
        pybobyqa.controller.Controller.trust_region_step = hooked_trust_region_step

        np.random.seed(self.seed)
        try:
            pybobyqa.solve(
                objective,
                self.x0.copy(),
                bounds=(self.lower_bounds, self.upper_bounds),
                scaling_within_bounds=True,
                maxfun=self.max_evals,
                rhobeg=self.rhobeg,
                rhoend=self.rhoend,
                user_params={"init.random_initial_directions": True},
                do_logging=False,
                print_progress=False,
            )
        except Exception:
            pass
        finally:
            # 还原原始方法
            pybobyqa.controller.Controller.trust_region_step = orig_tr_step

        while len(self.history_best) < self.max_evals:
            self.history_best.append(self.best_score)

        return {
            "best_x": self.best_x,
            "best_score": self.best_score,
            "history": self.history_best[:self.max_evals],
            "transitions": self.transitions,
        }
