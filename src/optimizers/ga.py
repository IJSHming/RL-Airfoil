"""遗传算法 (Genetic Algorithm, GA) 翼型优化器：
支持 Vanilla GA 与 RL-Meta-GA（自适应动态交叉率、变异率与变异尺度）。
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from src.optimizers.base import BaseAirfoilOptimizer
from src.evaluator import AerodynamicEvaluator


class GAOptimizer(BaseAirfoilOptimizer):
    """实数编码遗传算法 (Real-Coded GA)。种群规模 15，进化 20 代 (300 NFE)。"""

    def __init__(
        self,
        evaluator: AerodynamicEvaluator,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        max_evals: int = 300,
        seed: int = 42,
        is_meta: bool = False,
        rl_policy: Optional[Any] = None,
        pop_size: int = 15,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.15,
        mutation_scale: float = 0.02,
    ):
        super().__init__(evaluator, x0, bounds, max_evals, seed, is_meta, rl_policy)
        self.pop_size = pop_size
        self.pc = crossover_rate
        self.pm = mutation_rate
        self.sigma_m = mutation_scale
        self.initial_sigma_m = mutation_scale

        # 跟踪种群统计量
        self.pop_fitness_std = 0.0
        self.pop_diversity = 0.0

    def get_state(self) -> np.ndarray:
        """GA 状态空间 (11维) = 6维通用状态 + 5维 GA 种群动态特征"""
        base = self.get_base_state()

        norm_pc = self.pc
        norm_pm = self.pm * 2.0
        log_sigma = np.clip(np.log(max(self.sigma_m / self.initial_sigma_m, 1e-4)) / 2.0, -2.0, 2.0)
        norm_fit_std = min(self.pop_fitness_std / 20.0, 2.0)
        norm_div = min(self.pop_diversity / 0.5, 2.0)

        specific = np.array([norm_pc, norm_pm, log_sigma, norm_fit_std, norm_div], dtype=np.float32)
        return np.concatenate([base, specific])

    def apply_action(self, action: np.ndarray):
        """Action (3维) -> 动态调节 [交叉概率 pc, 变异概率 pm, 变异步长 sigma_m]"""
        # action in [-1, 1]^3
        a1, a2, a3 = action[0], action[1], action[2]
        self.pc = float(np.clip(0.4 + 0.58 * ((a1 + 1.0) / 2.0), 0.3, 0.99))
        self.pm = float(np.clip(0.02 + 0.38 * ((a2 + 1.0) / 2.0), 0.01, 0.50))
        scale_sigma = float(np.exp(np.clip(a3, -1.5, 1.5)))
        self.sigma_m = float(np.clip(self.initial_sigma_m * scale_sigma, 0.001, 0.08))

    def run(self) -> Dict[str, Any]:
        d = len(self.x0)
        population = np.zeros((self.pop_size, d), dtype=np.float64)
        fitnesses = np.zeros(self.pop_size, dtype=np.float64)

        # 初始第 0 号个体为基准 NACA0012
        population[0] = self.x0.copy()
        fitnesses[0] = self.evaluate(population[0])
        self.initial_score = fitnesses[0]

        # 初始化其余种群个体
        for i in range(1, self.pop_size):
            if self.eval_count >= self.max_evals:
                break
            noise = self.rng.uniform(-0.03, 0.03, size=d)
            ind = np.clip(self.x0 + noise, self.lower_bounds, self.upper_bounds)
            population[i] = ind
            fitnesses[i] = self.evaluate(ind)

        # 世代进化循环
        while self.eval_count < self.max_evals:
            # 统计当前种群状态
            self.pop_fitness_std = float(np.std(fitnesses))
            centroid = np.mean(population, axis=0)
            self.pop_diversity = float(np.mean(np.linalg.norm(population - centroid, axis=1)))

            score_before = self.best_score
            state = self.get_state()
            action = None
            if self.is_meta and self.rl_policy is not None:
                action = self.rl_policy(state)
                self.apply_action(action)

            # 精英保留 (保留第 1 名)
            best_idx = np.argmax(fitnesses)
            next_pop = [population[best_idx].copy()]
            next_fit = [fitnesses[best_idx]]

            # 锦标赛选择与重组变异
            while len(next_pop) < self.pop_size and self.eval_count < self.max_evals:
                # 锦标赛选择父代
                def tournament_select():
                    candidates = self.rng.choice(self.pop_size, size=2, replace=False)
                    return population[candidates[0]] if fitnesses[candidates[0]] > fitnesses[candidates[1]] else population[candidates[1]]

                p1 = tournament_select()
                p2 = tournament_select()

                # 交叉操作 (SBX / 混合交叉)
                if self.rng.rand() < self.pc:
                    alpha = self.rng.uniform(-0.1, 1.1, size=d)
                    child = alpha * p1 + (1.0 - alpha) * p2
                else:
                    child = p1.copy()

                # 变异操作 (高斯扰动)
                for gene_idx in range(d):
                    if self.rng.rand() < self.pm:
                        child[gene_idx] += self.rng.normal(0, self.sigma_m)

                child = np.clip(child, self.lower_bounds, self.upper_bounds)
                fit = self.evaluate(child)
                next_pop.append(child)
                next_fit.append(fit)

            population = np.array(next_pop)
            fitnesses = np.array(next_fit)

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
