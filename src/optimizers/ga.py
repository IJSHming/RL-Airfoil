"""差分进化 / 遗传算法兼容别名模块。
统一指向 DEOptimizer，保持学术命名严谨性并向前兼容。
"""

from src.optimizers.de import DEOptimizer

# 别名兼容
GAOptimizer = DEOptimizer

__all__ = ["DEOptimizer", "GAOptimizer"]
