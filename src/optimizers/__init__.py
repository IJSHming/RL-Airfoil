from src.optimizers.base import BaseAirfoilOptimizer
from src.optimizers.spsa import SPSAOptimizer
from src.optimizers.bobyqa import PyBobyqaOptimizer
from src.optimizers.de import DEOptimizer
from src.optimizers.ga import GAOptimizer
from src.optimizers.pso import PSOOptimizer
from src.optimizers.cma_es import CMAESOptimizer
from src.optimizers.turbo import TuRBOOptimizer

OPTIMIZER_CONFIGS = {
    "SPSA": {
        "class": SPSAOptimizer,
        "state_dim": 10,
        "action_dim": 2,
        "desc": "Simultaneous Perturbation Stochastic Approximation",
    },
    "Py-BOBYQA": {
        "class": PyBobyqaOptimizer,
        "state_dim": 10,
        "action_dim": 1,
        "desc": "Bound Optimization BY Quadratic Approximation",
    },
    "DE": {
        "class": DEOptimizer,
        "state_dim": 11,
        "action_dim": 3,
        "desc": "Differential Evolution (current-to-best/1/bin)",
    },
    "PSO": {
        "class": PSOOptimizer,
        "state_dim": 11,
        "action_dim": 3,
        "desc": "Particle Swarm Optimization",
    },
    "CMA-ES": {
        "class": CMAESOptimizer,
        "state_dim": 9,
        "action_dim": 1,
        "desc": "Covariance Matrix Adaptation Evolution Strategy",
    },
    "TuRBO": {
        "class": TuRBOOptimizer,
        "state_dim": 10,
        "action_dim": 2,
        "desc": "Trust Region Bayesian Optimization",
    },
}

# 向前兼容 GA 别名
OPTIMIZER_CONFIGS["GA"] = OPTIMIZER_CONFIGS["DE"]

__all__ = [
    "BaseAirfoilOptimizer",
    "SPSAOptimizer",
    "PyBobyqaOptimizer",
    "DEOptimizer",
    "GAOptimizer",
    "PSOOptimizer",
    "CMAESOptimizer",
    "TuRBOOptimizer",
    "OPTIMIZER_CONFIGS",
]
