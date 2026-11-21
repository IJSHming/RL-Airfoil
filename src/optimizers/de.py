"""差分进化算法 (Differential Evolution, DE) 翼型优化器：
采用连续空间实数编码 current-to-best 差分变异与二项交叉 (Binomial Crossover)。
遵循 Storn & Price (1997) 权威经典准则，采用物理边界自适应种群散布初始化 (0.20 * span)，
严格确保物理可行性，支持标准 Vanilla DE 与自适应 RL-Meta-DE (动态调度变异率 F 与交叉率 CR)。
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from src.optimizers.base import BaseAirfoilOptimizer
from src.evaluator import AerodynamicEvaluator


class DEOptimizer(BaseAirfoilOptimizer):
    """先进差分进化算法。种群规模 12，迭代约 25 代 (300 NFE)。"""

    def __init__(
        self,
        evaluator: AerodynamicEvaluator,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        max_evals: int = 300,
        seed: int = 42,
        is_meta: bool = False,
        rl_policy: Optional[Any] = None,
        pop_size: int = 12,
        crossover_rate: float = 0.80,
        f_scale: float = 0.70,
        span_scale: float = 0.20,
    ):
        super().__init__(evaluator, x0, bounds, max_evals, seed, is_meta, rl_policy)
        self.pop_size = pop_size
        self.base_cr = crossover_rate
        self.base_f = f_scale
        self.span_scale = span_scale

        self.CR = crossover_rate
        self.F = f_scale
        self.span = self.upper_bounds - self.lower_bounds

        # 跟踪种群统计量
        self.pop_fitness_std = 0.0
        self.pop_diversity = 0.0

    def get_state(self) -> np.ndarray:
        """DE 状态空间 (11维) = 6维通用状态 + 5维 DE 种群演化特征 (以近期可行率替换冗余停滞特征)"""
        base = self.get_base_state()

        norm_cr = float(self.CR)
        norm_f = float(self.F / 1.2)
        feas_rate = self.get_recent_feasibility_rate(window=12)
        norm_fit_std = float(min(self.pop_fitness_std / 20.0, 2.0))
        norm_div = float(min(self.pop_diversity / 0.25, 2.0))

        specific = np.array([norm_cr, norm_f, feas_rate, norm_fit_std, norm_div], dtype=np.float32)
        return np.concatenate([base, specific])

    def apply_action(self, action: np.ndarray):
        """Action (2维) -> 纯净 RL 策略动态调节差分进化超参 [F, CR] (action=0 时精确退化为 Vanilla DE)"""
        a1 = float(np.clip(action[0], -1.0, 1.0))
        a2 = float(np.clip(action[1], -1.0, 1.0)) if len(action) > 1 else 0.0
        self.F = float(np.clip(self.base_f * np.exp(a1 * 0.35), 0.35, 1.05))
        self.CR = float(np.clip(self.base_cr + a2 * 0.15, 0.50, 0.95))

    def run(self) -> Dict[str, Any]:
        d = len(self.x0)
        population = np.zeros((self.pop_size, d), dtype=np.float64)
        fitnesses = np.zeros(self.pop_size, dtype=np.float64)

        # 初始第 0 号个体为基准 NACA0012
        population[0] = self.x0.copy()
        fitnesses[0] = self.evaluate(population[0])
        self.initial_score = fitnesses[0]
        span = self.span

        # 初始种群边界自适应散布 (采用拒绝采样严格保证 100% 物理与气动可行)
        for i in range(1, self.pop_size):
            if self.eval_count >= self.max_evals:
                break
            for _ in range(50):
                noise = self.rng.uniform(-self.span_scale, self.span_scale, size=d) * span
                ind = np.clip(self.x0 + noise, self.lower_bounds, self.upper_bounds)
                score = self.evaluate(ind)
                if score > 0.0:
                    population[i] = ind
                    fitnesses[i] = score
                    break
            else:
                population[i] = self.x0.copy()
                fitnesses[i] = self.initial_score

        # 世代演化循环
        while self.eval_count < self.max_evals:
            # 统计当前种群状态
            valid_fits = [f for f in fitnesses if f > 0.0]
            self.pop_fitness_std = float(np.std(valid_fits)) if len(valid_fits) > 1 else 0.0
            centroid = np.mean(population, axis=0)
            self.pop_diversity = float(np.mean(np.linalg.norm(population - centroid, axis=1)))

            score_before = self.best_score
            state = self.get_state()
            action = None
            if self.is_meta and self.rl_policy is not None:
                action = self.rl_policy(state)
                self.apply_action(action)
            else:
                self.CR = self.base_cr
                self.F = self.base_f

            # 遍历种群中的每个个体进行差分重组与变异演化
            for i in range(self.pop_size):
                if self.eval_count >= self.max_evals:
                    break

                # 随机选择 2 个不同于 i 的个体
                candidate_indices = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2 = self.rng.choice(candidate_indices, size=2, replace=False)

                # current-to-best 差分变异引导向量
                best_idx = int(np.argmax(fitnesses))
                v = (
                    population[i]
                    + self.F * (population[best_idx] - population[i])
                    + self.F * (population[r1] - population[r2])
                )

                # 二项交叉 (Binomial Crossover)
                j_rand = self.rng.randint(d)
                crossover_mask = (self.rng.rand(d) < self.CR)
                crossover_mask[j_rand] = True
                trial = np.where(crossover_mask, v, population[i])
                trial = np.clip(trial, self.lower_bounds, self.upper_bounds)

                # 评估试探向量
                trial_fit = self.evaluate(trial)

                # 贪婪选择与保优更新 (只有当试探解适应度更高且有效时替代)
                if trial_fit >= fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = trial_fit

            # 记录 RL 经验 (保存原始最优增量，交由训练端按同工况配对基线做无偏控制变量归一化)
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

        while len(self.history_best) < self.max_evals:
            self.history_best.append(self.best_score)

        return {
            "best_x": self.best_x,
            "best_score": self.best_score,
            "history": self.history_best[:self.max_evals],
            "transitions": self.transitions,
        }
