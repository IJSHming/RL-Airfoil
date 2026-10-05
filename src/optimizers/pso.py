"""粒子群优化算法 (Particle Swarm Optimization, PSO) 翼型优化器：
支持 Vanilla PSO 与 RL-Meta-PSO（动态惯性权重 w、个体认知 c1 与社会协同 c2 调节）。
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from src.optimizers.base import BaseAirfoilOptimizer
from src.evaluator import AerodynamicEvaluator


class PSOOptimizer(BaseAirfoilOptimizer):
    """PSO 翼型优化器。粒子群规模 15，共迭代 20 代 (300 NFE)。"""

    def __init__(
        self,
        evaluator: AerodynamicEvaluator,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        max_evals: int = 300,
        seed: int = 42,
        is_meta: bool = False,
        rl_policy: Optional[Any] = None,
        swarm_size: int = 15,
        w: float = 0.729,
        c1: float = 1.494,
        c2: float = 1.494,
    ):
        super().__init__(evaluator, x0, bounds, max_evals, seed, is_meta, rl_policy)
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # 跟踪统计特征
        self.swarm_diversity = 0.0
        self.mean_vel_norm = 0.0

    def get_state(self) -> np.ndarray:
        """PSO 状态空间 (11维) = 6维通用状态 + 5维 PSO 群体动力学特征"""
        base = self.get_base_state()

        norm_w = self.w
        norm_c1 = self.c1 / 2.0
        norm_c2 = self.c2 / 2.0
        norm_div = min(self.swarm_diversity / 0.5, 2.0)
        norm_v = min(self.mean_vel_norm / 0.1, 2.0)

        specific = np.array([norm_w, norm_c1, norm_c2, norm_div, norm_v], dtype=np.float32)
        return np.concatenate([base, specific])

    def apply_action(self, action: np.ndarray):
        """Action (3维) -> 动态调节 [惯性权重 w, 认知系数 c1, 社会系数 c2]"""
        # action in [-1, 1]^3
        a1, a2, a3 = action[0], action[1], action[2]
        self.w = float(np.clip(0.20 + 0.75 * ((a1 + 1.0) / 2.0), 0.1, 0.95))
        self.c1 = float(np.clip(0.50 + 2.00 * ((a2 + 1.0) / 2.0), 0.2, 3.0))
        self.c2 = float(np.clip(0.50 + 2.00 * ((a3 + 1.0) / 2.0), 0.2, 3.0))

    def run(self) -> Dict[str, Any]:
        d = len(self.x0)
        positions = np.zeros((self.swarm_size, d), dtype=np.float64)
        velocities = np.zeros((self.swarm_size, d), dtype=np.float64)
        pbest_positions = np.zeros((self.swarm_size, d), dtype=np.float64)
        pbest_scores = np.full(self.swarm_size, -float("inf"), dtype=np.float64)

        v_max = 0.10 * (self.upper_bounds - self.lower_bounds)

        # 初始化第 0 号粒子为 NACA0012
        positions[0] = self.x0.copy()
        pbest_positions[0] = self.x0.copy()
        score0 = self.evaluate(positions[0])
        pbest_scores[0] = score0
        self.initial_score = score0

        # 初始化其余粒子
        for i in range(1, self.swarm_size):
            if self.eval_count >= self.max_evals:
                break
            noise = self.rng.uniform(-0.03, 0.03, size=d)
            pos = np.clip(self.x0 + noise, self.lower_bounds, self.upper_bounds)
            positions[i] = pos
            pbest_positions[i] = pos.copy()
            velocities[i] = self.rng.uniform(-v_max, v_max, size=d)
            score = self.evaluate(pos)
            pbest_scores[i] = score

        # 群体迭代
        while self.eval_count < self.max_evals:
            # 统计群体动力学特征
            centroid = np.mean(positions, axis=0)
            self.swarm_diversity = float(np.mean(np.linalg.norm(positions - centroid, axis=1)))
            self.mean_vel_norm = float(np.mean(np.linalg.norm(velocities, axis=1)))

            score_before = self.best_score
            state = self.get_state()
            action = None
            if self.is_meta and self.rl_policy is not None:
                action = self.rl_policy(state)
                self.apply_action(action)

            # 更新粒子速度与位置
            gbest = self.best_x.copy()
            for i in range(self.swarm_size):
                if self.eval_count >= self.max_evals:
                    break
                r1 = self.rng.uniform(0.0, 1.0, size=d)
                r2 = self.rng.uniform(0.0, 1.0, size=d)

                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (pbest_positions[i] - positions[i])
                    + self.c2 * r2 * (gbest - positions[i])
                )
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                positions[i] = np.clip(
                    positions[i] + velocities[i],
                    self.lower_bounds,
                    self.upper_bounds,
                )

                score = self.evaluate(positions[i])
                if score > pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_positions[i] = positions[i].copy()

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
