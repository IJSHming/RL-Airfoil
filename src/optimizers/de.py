"""差分进化算法 (Differential Evolution, DE) 翼型优化器：
采用连续空间实数编码 current-to-best 差分变异与二项交叉 (Binomial Crossover)，
并以超参数扫描峰值 (pop=12, CR=0.85, F=0.85, sigma_m=0.003) 为基准，
支持 Vanilla DE 与 RL-Meta-DE（自适应动态交叉率 CR、差分步长 F 与变异步长 sigma_m 弹性调节）。
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
        crossover_rate: float = 0.70,
        f_scale: float = 0.50,
        mutation_scale: float = 0.001,
    ):
        super().__init__(evaluator, x0, bounds, max_evals, seed, is_meta, rl_policy)
        self.pop_size = pop_size
        self.base_cr = crossover_rate
        self.base_f = f_scale
        self.base_sigma_m = mutation_scale

        self.CR = crossover_rate
        self.F = f_scale
        self.sigma_m = mutation_scale

        # 跟踪种群统计量
        self.pop_fitness_std = 0.0
        self.pop_diversity = 0.0

    def get_state(self) -> np.ndarray:
        """DE 状态空间 (11维) = 6维通用状态 + 5维 DE 种群演化特征"""
        base = self.get_base_state()

        norm_cr = float(self.CR)
        norm_f = float(self.F / 1.2)
        base_sig = max(self.base_sigma_m, 1e-4)
        log_sigma = float(np.clip(np.log(max(self.sigma_m / base_sig, 1e-4)) / 2.0, -2.0, 2.0))
        norm_fit_std = float(min(self.pop_fitness_std / 20.0, 2.0))
        norm_div = float(min(self.pop_diversity / 0.5, 2.0))

        specific = np.array([norm_cr, norm_f, log_sigma, norm_fit_std, norm_div], dtype=np.float32)
        return np.concatenate([base, specific])

    def apply_action(self, action: np.ndarray):
        """Action (3维) -> 动态调节差分进化超参 [CR, F, sigma_m] 从默认基准向黄金探索空间跃迁"""
        a1, a2, a3 = float(action[0]), float(action[1]), float(action[2])
        cr_target = float(np.clip(self.base_cr + 0.25 * np.clip(a1, -1.0, 1.0), 0.50, 0.96))
        f_target = float(np.clip(self.base_f + 0.50 * np.clip(a2, -1.0, 1.0), 0.35, 1.15))
        scale_sigma = float(np.exp(np.clip(a3, -1.0, 1.0) * 2.2))
        sig_target = float(np.clip(self.base_sigma_m * scale_sigma, 1e-4, 0.020))

        # 动量平滑滤波
        self.CR = float(0.4 * self.CR + 0.6 * cr_target)
        self.F = float(0.4 * self.F + 0.6 * f_target)
        self.sigma_m = float(0.4 * self.sigma_m + 0.6 * sig_target)

    def run(self) -> Dict[str, Any]:
        d = len(self.x0)
        population = np.zeros((self.pop_size, d), dtype=np.float64)
        fitnesses = np.zeros(self.pop_size, dtype=np.float64)

        # 初始第 0 号个体为基准 NACA0012
        population[0] = self.x0.copy()
        fitnesses[0] = self.evaluate(population[0])
        self.initial_score = fitnesses[0]

        # 初始化其余种群个体 (小幅度高斯微扰探索周围可行域)
        for i in range(1, self.pop_size):
            if self.eval_count >= self.max_evals:
                break
            noise = self.rng.uniform(-0.02, 0.02, size=d)
            ind = np.clip(self.x0 + noise, self.lower_bounds, self.upper_bounds)
            population[i] = ind
            fitnesses[i] = self.evaluate(ind)

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
                self.sigma_m = self.base_sigma_m

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

                # 局部高斯微扰增强逃逸能力
                trial += self.rng.normal(0.0, self.sigma_m, size=d)
                trial = np.clip(trial, self.lower_bounds, self.upper_bounds)

                # 评估试探向量
                trial_fit = self.evaluate(trial)

                # 贪婪选择与保优更新 (只有当试探解适应度更高且有效时替代)
                if trial_fit >= fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = trial_fit

            # 记录 RL 转移 (跨工况自适应归一化稠密奖励与突破里程碑激励)
            if self.is_meta and action is not None:
                next_state = self.get_state()
                done = (self.eval_count >= self.max_evals)
                rel_gain = (self.best_score - score_before) / max(self.initial_score, 15.0)
                reward = rel_gain * 30.0 - 0.002
                if self.best_score > 105.0 and score_before <= 105.0:
                    reward += 1.0  # 冲出 105 局部停滞
                if self.best_score > 130.0 and score_before <= 130.0:
                    reward += 1.5  # 突破 130 平台
                if self.best_score > 160.0 and score_before <= 160.0:
                    reward += 2.5  # 冲向 160+ 高升阻比
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
