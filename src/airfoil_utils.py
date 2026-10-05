"""翼型参数化与几何工具库：
提供 NACA0012 初始参数生成、18维 Kulfan 参数上下界、以及翼型表面点云坐标提取。
"""

from typing import Dict, Tuple
import numpy as np
import aerosandbox as asb


def get_naca0012_theta() -> np.ndarray:
    """获取 NACA0012 的标准 18 维 Kulfan (CST) 参数表示。
    
    前 8 维：上表面权重 upper_weights
    中间 8 维：下表面权重 lower_weights
    第 17 维：前缘权重 leading_edge_weight
    第 18 维：后缘厚度 TE_thickness
    """
    af = asb.Airfoil("naca0012")
    k_af = af.to_kulfan_airfoil(n_weights_per_side=8)
    theta = np.concatenate([
        np.asarray(k_af.upper_weights, dtype=np.float64),
        np.asarray(k_af.lower_weights, dtype=np.float64),
        [float(k_af.leading_edge_weight)],
        [float(k_af.TE_thickness)],
    ])
    return theta


def get_kulfan_bounds(base_theta: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """获取 18 维 Kulfan 参数的可行探索上下界。
    
    保证在工程物理与气动可行域范围内探索，避免不必要的物理塌陷。
    """
    if base_theta is None:
        base_theta = get_naca0012_theta()

    lower_bounds = np.zeros(18, dtype=np.float64)
    upper_bounds = np.zeros(18, dtype=np.float64)

    # 1. 上表面权重 (通常为正值，厚度与吸力面控制)
    lower_bounds[0:8] = np.maximum(0.01, base_theta[0:8] - 0.20)
    upper_bounds[0:8] = base_theta[0:8] + 0.25

    # 2. 下表面权重 (通常为负值，下表面与弯度控制)
    lower_bounds[8:16] = base_theta[8:16] - 0.25
    upper_bounds[8:16] = np.minimum(0.05, base_theta[8:16] + 0.20)

    # 3. 前缘权重 (微调前缘曲率，微弱正负扰动)
    lower_bounds[16] = -0.05
    upper_bounds[16] = 0.05

    # 4. 后缘厚度 (工程最小到最大允许范围)
    lower_bounds[17] = 0.0010
    upper_bounds[17] = 0.0060

    return lower_bounds, upper_bounds


def theta_to_coordinates(theta: np.ndarray, n_points: int = 150) -> Dict[str, np.ndarray]:
    """从 18 维 Kulfan 向量生成连续的上/下表面坐标点集用于高保真绘图。
    
    返回:
        {
            "x_upper": x_u, "y_upper": y_u,
            "x_lower": x_l, "y_lower": y_l,
            "x_closed": x_all, "y_closed": y_all
        }
    """
    theta = np.asarray(theta, dtype=np.float64)
    u_w = theta[0:8]
    l_w = theta[8:16]
    le_w = float(theta[16]) if len(theta) > 16 else 0.0
    te_t = float(theta[17]) if len(theta) > 17 else 0.0025

    kulfan_af = asb.KulfanAirfoil(
        upper_weights=u_w,
        lower_weights=l_w,
        leading_edge_weight=le_w,
        TE_thickness=te_t,
    )

    # 使用余弦分布加密前缘和后缘
    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(beta))

    try:
        coords_u = kulfan_af.upper_coordinates(x)
        coords_l = kulfan_af.lower_coordinates(x)
        yu = coords_u[:, 1]
        yl = coords_l[:, 1]
    except Exception:
        yu = np.zeros_like(x)
        yl = np.zeros_like(x)

    # 闭合翼型环路：从后缘上表面 -> 前缘 -> 后缘下表面
    x_closed = np.concatenate([coords_u[::-1, 0], coords_l[1:, 0]])
    y_closed = np.concatenate([yu[::-1], yl[1:]])

    return {
        "x_upper": x,
        "y_upper": yu,
        "x_lower": x,
        "y_lower": yl,
        "x_closed": x_closed,
        "y_closed": y_closed,
    }
