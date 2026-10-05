"""黑盒优化器基础抽象类：
统一管理 300 次 NFE 评估计数、气动评估调用、状态提取与历史轨迹追踪。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from src.airfoil_utils import get_kulfan_bounds, get_naca0012_theta
from src.evaluator import AerodynamicEvaluator


class BaseAirfoilOptimizer(ABC):
    """所有气动外形黑盒优化器的基础类。
    
    统一约束：
    1. 严格 300 次函数评估预算 (NFE = max_evals = 300)。
    2. 统一初始形状为 NACA0012。
    3. 统一记录完整的历史收敛轨迹与可行解状态。
    4. 提供通用的 RL 状态提取与动态动作应用接口。
    """

    def __init__(
        self,
        evaluator: AerodynamicEvaluator,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        max_evals: int = 300,
        seed: int = 42,
        is_meta: bool = False,
        rl_policy: Optional[Any] = None,
    ):
        self.evaluator = evaluator
        self.x0 = get_naca0012_theta() if x0 is None else np.asarray(x0, dtype=np.float64)
        if bounds is None:
            self.lower_bounds, self.upper_bounds = get_kulfan_bounds(self.x0)
        else:
            self.lower_bounds, self.upper_bounds = bounds

        self.max_evals = int(max_evals)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.is_meta = bool(is_meta)
        self.rl_policy = rl_policy

        # 运行指标跟踪
        self.eval_count = 0
        self.best_x = self.x0.copy()
        self.best_score = -float("inf")
        self.initial_score = 0.0

        # 历史记录 (每步记录历史最优升阻比)
        self.history_best: List[float] = []
        self.history_evaluated_scores: List[float] = []
        self.history_feasible: List[bool] = []
        self.recent_best_window: List[float] = []
        self.stagnation_counter = 0

        # RL 经验缓存 (供训练时收集轨迹)
        self.transitions: List[Dict[str, Any]] = []

    def evaluate(self, x: np.ndarray) -> float:
        """评估单个翼型候选点。如果预算耗尽，直接返回当前最优并打断。"""
        if self.eval_count >= self.max_evals:
            return self.best_score

        x_clipped = np.clip(x, self.lower_bounds, self.upper_bounds)
        score = self.evaluator.evaluate(x_clipped)
        self.eval_count += 1

        is_feasible = (score > 0.0)
        self.history_feasible.append(is_feasible)
        self.history_evaluated_scores.append(score)

        if score > self.best_score:
            self.best_score = score
            self.best_x = x_clipped.copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        self.history_best.append(self.best_score)
        self.recent_best_window.append(self.best_score)
        if len(self.recent_best_window) > 20:
            self.recent_best_window.pop(0)

        return score

    def evaluate_initial(self):
        """初始评估基准形状 NACA0012"""
        score = self.evaluate(self.x0)
        self.initial_score = score
        return score

    def get_base_state(self) -> np.ndarray:
        """提取通用的优化动态基础状态特征 (6维)：
        1. 归一化评估进度 eval_count / max_evals
        2. 归一化雷诺数 (log10(Re) - 6.0) / 1.0 (映射 [1e6, 1e7] 到 [0, 1])
        3. 归一化攻角 alpha / 10.0 (映射 [0, 10] 到 [0, 1])
        4. 相对初始解的提升 (best_score - initial_score) / 50.0
        5. 近期窗口提升率 (best_score - window_min) / 20.0
        6. 停滞步数比例 min(stagnation / 30.0, 1.0)
        """
        progress = self.eval_count / max(self.max_evals, 1)
        re_norm = (np.log10(max(self.evaluator.Re, 1e6)) - 6.0) / 1.0
        alpha_norm = self.evaluator.alpha / 10.0
        gain_norm = (self.best_score - self.initial_score) / 50.0
        recent_gain = (self.best_score - self.recent_best_window[0]) / 20.0 if self.recent_best_window else 0.0
        stag_norm = min(self.stagnation_counter / 30.0, 1.0)

        return np.array([progress, re_norm, alpha_norm, gain_norm, recent_gain, stag_norm], dtype=np.float32)

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """返回包含通用特征与该算法专有动态特征的完整状态向量"""
        pass

    @abstractmethod
    def apply_action(self, action: np.ndarray):
        """将 RL 策略输出的连续动作映射应用到优化器的控制参数"""
        pass

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """运行优化直至消耗满 300 次评估"""
        pass
