"""协方差矩阵自适应进化策略 (CMA-ES) 翼型优化器：
支持 Vanilla CMA-ES 与 RL-Meta-CMA-ES（自适应步长乘子与探索强度动态调节）。
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import cma
from src.optimizers.base import BaseAirfoilOptimizer
from src.evaluator import AerodynamicEvaluator


class CMAESOptimizer(BaseAirfoilOptimizer):
    """CMA-ES 优化器。基于 cma 库，种群规模约 12，总评估严格限制为 300 次。"""

    def __init__(
        self,
        evaluator: AerodynamicEvaluator,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        max_evals: int = 300,
        seed: int = 42,
        is_meta: bool = False,
        rl_policy: Optional[Any] = None,
        sigma0: float = 0.035,
        pop_size: int = 10,
    ):
        super().__init__(evaluator, x0, bounds, max_evals, seed, is_meta, rl_policy)
        self.sigma0 = sigma0
        self.pop_size = pop_size

        # CMA-ES 状态量
        self.current_sigma = sigma0
        self.path_ratio = 1.0
        self.cond_c = 1.0

    def get_state(self) -> np.ndarray:
        """CMA-ES 状态空间 (9维) = 6维通用状态 + 3维 CMA-ES 协方差与步长特征"""
        base = self.get_base_state()

        log_sigma = np.clip(np.log(max(self.current_sigma / self.sigma0, 1e-4)) / 2.0, -2.0, 2.0)
        norm_path = min(self.path_ratio / 2.0, 2.0)
        norm_cond = np.clip(np.log10(max(self.cond_c, 1.0)) / 4.0, 0.0, 2.0)

        specific = np.array([log_sigma, norm_path, norm_cond], dtype=np.float32)
        return np.concatenate([base, specific])

    def apply_action(self, action: np.ndarray, es: cma.CMAEvolutionStrategy):
        """Action (1维) -> 动态调节当前进化策略的全局步长 sigma (前中期防塌陷，后期精细收敛)"""
        scale_sigma = float(np.exp(np.clip(action[0], -1.0, 1.0) * 0.45))
        # 前中期 (eval < 220) 强制维持至少 0.006 步长防止过早陷入局部极小
        min_allowed_sigma = 0.006 if self.eval_count < 220 else 1e-4
        new_sigma = float(np.clip(es.sigma * scale_sigma, min_allowed_sigma, 0.18))
        es.sigma = new_sigma
        self.current_sigma = new_sigma

    def run(self) -> Dict[str, Any]:
        self.evaluate_initial()

        inopts = {
            "bounds": [self.lower_bounds.tolist(), self.upper_bounds.tolist()],
            "popsize": self.pop_size,
            "maxfevals": self.max_evals,
            "seed": self.seed,
            "verbose": -9,
        }

        es = cma.CMAEvolutionStrategy(self.x0.copy(), self.sigma0, inopts)

        while not es.stop() and self.eval_count < self.max_evals:
            # 提取 CMA 内部状态
            self.current_sigma = float(es.sigma)
            try:
                # 进化路径范数与期望长度比值
                if hasattr(es, "ps") and es.ps is not None:
                    d = len(self.x0)
                    chiN = d**0.5 * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d**2))
                    self.path_ratio = float(np.linalg.norm(es.ps) / chiN)
                # 协方差矩阵条件数
                if hasattr(es, "D") and es.D is not None:
                    self.cond_c = float(np.max(es.D) / max(np.min(es.D), 1e-12)) ** 2
            except Exception:
                pass

            score_before = self.best_score
            state = self.get_state()
            action = None
            if self.is_meta and self.rl_policy is not None:
                action = self.rl_policy(state)
                self.apply_action(action, es)

            # 采样候选解群体
            X = es.ask()
            fitnesses_to_tell = []
            for candidate in X:
                if self.eval_count >= self.max_evals:
                    fitnesses_to_tell.append(-self.best_score)
                else:
                    score = self.evaluate(candidate)
                    fitnesses_to_tell.append(-score)

            es.tell(X, fitnesses_to_tell)

            # 记录经验 (跨工况自适应归一化稠密奖励与巅峰里程碑激励)
            if self.is_meta and action is not None:
                next_state = self.get_state()
                done = bool(self.eval_count >= self.max_evals or bool(es.stop()))
                rel_gain = (self.best_score - score_before) / max(self.initial_score, 15.0)
                reward = rel_gain * 30.0 - 0.002
                if self.best_score > 205.0 and score_before <= 205.0:
                    reward += 1.0  # 突破 205 门槛里程碑
                if self.best_score > 218.0 and score_before <= 218.0:
                    reward += 2.0  # 冲刺 218+ 巅峰大奖
                if done:
                    reward += max((self.best_score - self.initial_score) / 50.0, 0.0) * 0.25
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
