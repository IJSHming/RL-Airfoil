"""2D 翼型高保真水密 CFD STL 表面网格导出脚本 (RL-Airfoil)。

从 `results/llm_malibo_benchmark_results.json` 读取与论文 Fig. 4 完全一致的中位数 (Median) 代表性翼型，
结合 NACA 0012 基准翼型，生成 100% 水密 (Watertight, Closed 2-Manifold) 3D 拉伸 STL 网格，
存放于 `cfd_models/` 目录，供高保真 RANS/URANS CFD 气动验证使用。

坐标系与拉伸规范:
- X 轴: 弦向坐标 x in [0.0, 1.0] m (单位弦长 c = 1.0 m);
- Y 轴: 弯度与厚度法向坐标 y (保留精确 Kulfan CST 后缘有限厚度 TE_thickness);
- Z 轴: 准二维展向拉伸段 z in [0.0, 0.1] m (展长 b = 0.1 m = 0.1c, 适配 OpenFOAM / Fluent / STAR-CCM+ 2D/对称面网格);
- 端盖三角剖分: 沿弦向采用精确上下表面逐条带 (strip) 三角化，严格杜绝大弯度凹型翼型端面自交与法向翻转。
"""

import json
import os
from typing import Any, Dict, List, Tuple

import aerosandbox as asb
import numpy as np
import trimesh

from src.airfoil_utils import get_naca0012_theta
from src.evaluator import AerodynamicEvaluator


def build_extruded_airfoil_mesh(
    theta: np.ndarray,
    chord: float = 1.0,
    span: float = 0.1,
    n_chord: int = 200,
) -> trimesh.Trimesh:
    """将 18 维 Kulfan CST 翼型沿 Z 轴拉伸为 100% 水密闭合 2-Manifold 三角网格。"""
    theta = np.asarray(theta, dtype=np.float64)
    k_af = asb.KulfanAirfoil(
        upper_weights=theta[0:8],
        lower_weights=theta[8:16],
        leading_edge_weight=float(theta[16]),
        TE_thickness=float(theta[17]) if len(theta) > 17 else 0.0,
    )

    beta = np.linspace(0.0, np.pi, n_chord)
    x_c = 0.5 * (1.0 - np.cos(beta))
    yu = k_af.upper_coordinates(x_c)[:, 1]
    yl = k_af.lower_coordinates(x_c)[:, 1]

    # 强制前缘精确闭合于 (0, 0) 并防止数值误差导致上下表面反转
    yu[0] = 0.0
    yl[0] = 0.0
    te_gap = float(yu[-1] - yl[-1])
    is_blunt_te = te_gap > 1e-7

    x_phys = x_c * chord
    yu_phys = yu * chord
    yl_phys = yl * chord

    # 构造每个展向截面 z0=0 和 z1=span 的独立顶点索引表:
    # 顶点排列: 0 = LE(0,0), 1..n_chord-1 = 上表面 i=1..n_chord-1,
    #           n_chord..2*n_chord-2 = 下表面 i=1..n_chord-1 (若尖尾缘则去掉最后一个重复 TE 顶点)
    if is_blunt_te:
        n_sec = 2 * n_chord - 1
        sec_xy = np.zeros((n_sec, 2), dtype=np.float64)
        sec_xy[0] = [x_phys[0], 0.0]
        sec_xy[1:n_chord, 0] = x_phys[1:]
        sec_xy[1:n_chord, 1] = yu_phys[1:]
        sec_xy[n_chord:, 0] = x_phys[1:]
        sec_xy[n_chord:, 1] = yl_phys[1:]

        def u_idx(i: int) -> int:
            return 0 if i == 0 else i

        def l_idx(i: int) -> int:
            return 0 if i == 0 else (n_chord - 1 + i)

        # 逆时针 (CCW) 边界环路索引: LE -> 下表面(1..n-1) -> TE上(n-1) -> 上表面(n-2..1)
        ccw_loop = [0] + [l_idx(i) for i in range(1, n_chord)] + [u_idx(i) for i in range(n_chord - 1, 0, -1)]
    else:
        n_sec = 2 * n_chord - 2
        sec_xy = np.zeros((n_sec, 2), dtype=np.float64)
        sec_xy[0] = [x_phys[0], 0.0]
        sec_xy[1:n_chord, 0] = x_phys[1:]
        sec_xy[1:n_chord, 1] = yu_phys[1:]
        sec_xy[n_chord:, 0] = x_phys[1:-1]
        sec_xy[n_chord:, 1] = yl_phys[1:-1]

        def u_idx(i: int) -> int:
            return 0 if i == 0 else i

        def l_idx(i: int) -> int:
            if i == 0:
                return 0
            if i == n_chord - 1:
                return n_chord - 1
            return n_chord - 1 + i

        ccw_loop = [0] + [l_idx(i) for i in range(1, n_chord - 1)] + [n_chord - 1] + [u_idx(i) for i in range(n_chord - 2, 0, -1)]

    # 拼装 z=0 与 z=span 两层顶点
    v_bot = np.column_stack([sec_xy, np.zeros(n_sec, dtype=np.float64)])
    v_top = np.column_stack([sec_xy, np.full(n_sec, span, dtype=np.float64)])
    vertices = np.vstack([v_bot, v_top])
    off = n_sec

    faces: List[List[int]] = []

    # 1. 侧面蒙皮 (CCW 环路沿 +Z 拉伸 -> 外法向朝外)
    m = len(ccw_loop)
    for k in range(m):
        a = ccw_loop[k]
        b = ccw_loop[(k + 1) % m]
        faces.append([a, b, off + b])
        faces.append([a, off + b, off + a])

    # 2. 精确逐条带端盖三角化 (完全位于翼型内部, 绝对零自交)
    # 顶面 (z = span, 法向 +Z, CCW):
    # 前缘三角形: (LE, l_1, u_1)
    faces.append([off + 0, off + l_idx(1), off + u_idx(1)])
    # 中间条带四边形 i -> i+1:
    last_strip_end = n_chord - 1 if is_blunt_te else n_chord - 2
    for i in range(1, last_strip_end):
        u0, u1 = off + u_idx(i), off + u_idx(i + 1)
        l0, l1 = off + l_idx(i), off + l_idx(i + 1)
        faces.append([l0, l1, u1])
        faces.append([l0, u1, u0])
    if not is_blunt_te:
        # 尖尾缘闭合三角形: (l_{n-2}, TE, u_{n-2})
        faces.append([off + l_idx(n_chord - 2), off + u_idx(n_chord - 1), off + u_idx(n_chord - 2)])

    # 底面 (z = 0, 法向 -Z, CW = 顶面反向):
    faces.append([0, u_idx(1), l_idx(1)])
    for i in range(1, last_strip_end):
        u0, u1 = u_idx(i), u_idx(i + 1)
        l0, l1 = l_idx(i), l_idx(i + 1)
        faces.append([l0, u1, l1])
        faces.append([l0, u0, u1])
    if not is_blunt_te:
        faces.append([l_idx(n_chord - 2), u_idx(n_chord - 2), u_idx(n_chord - 1)])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.int64), process=True)
    trimesh.repair.fix_normals(mesh)
    if mesh.volume < 0:
        mesh.invert()
    assert mesh.is_watertight, "生成的 2D 翼型 STL 未闭合 (not watertight)!"
    assert mesh.is_volume, "生成的 2D 翼型 STL 不是合法闭合实体 (not a valid volume)!"
    return mesh


def main():
    data_path = "results/llm_malibo_benchmark_results.json"
    out_dir = "cfd_models"
    os.makedirs(out_dir, exist_ok=True)

    with open(data_path, "r", encoding="utf-8") as f:
        blob = json.load(f)

    cond = blob["condition"]
    reynolds = float(cond.get("reynolds", 3.5e6))
    alpha = float(cond.get("alpha", 3.5))
    evaluator = AerodynamicEvaluator(reynolds=reynolds, alpha=alpha, max_evals=10)

    x0 = get_naca0012_theta()
    det0 = evaluator.evaluate_detailed(x0)

    items: List[Tuple[str, str, str, np.ndarray, Dict[str, Any]]] = [
        (
            "airfoil_naca0012_baseline.stl",
            "Baseline NACA 0012",
            "N/A",
            x0,
            {
                "ld": float(det0["ld"]),
                "cl": float(det0["cl"]),
                "cd": float(det0["cd"]),
                "cm": float(det0["cm"]),
                "t_max": float(det0["t_max"]),
                "te_thickness": float(x0[17]),
            },
        )
    ]

    method_specs = [
        ("Vanilla-TuRBO", "airfoil_turbo_median.stl", "TuRBO-1 (Median)"),
        ("RL-Meta-TuRBO", "airfoil_rl_meta_turbo_median.stl", "RL-Meta-TuRBO (Median)"),
        ("LLM-Meta-TuRBO", "airfoil_llm_meta_turbo_median.stl", "LLM-Meta-TuRBO (Median)"),
        ("MALIBO", "airfoil_malibo_median.stl", "MALIBO (Median)"),
    ]

    for m_key, fname, desc in method_specs:
        runs = blob["results"][m_key]
        scores = np.array([r["best_score"] for r in runs], dtype=np.float64)
        # 与 plot_llm_malibo.py 完全一致的中位数选择逻辑: argsort(scores)[len(scores)//2]
        med_idx = int(np.argsort(scores)[len(scores) // 2])
        rep = runs[med_idx]
        theta = np.array(rep["best_x"], dtype=np.float64)
        det = rep["detailed"]
        items.append(
            (
                fname,
                desc,
                str(rep["seed"]),
                theta,
                {
                    "ld": float(rep["best_score"]),
                    "cl": float(det["cl"]),
                    "cd": float(det["cd"]),
                    "cm": float(det["cm"]),
                    "t_max": float(det["t_max"]),
                    "te_thickness": float(theta[17]),
                },
            )
        )

    md_lines = [
        "# 2D Airfoil Geometry Models for CFD Aerodynamic Validation (RL-Airfoil)",
        "",
        "This directory contains 100% watertight (closed 2-manifold) extruded 2D airfoil STL surface meshes corresponding to the exact **Representative Median Runs** plotted in the paper (`figures/llm_malibo/llm_malibo_airfoil_shapes.pdf`), ready for high-fidelity RANS CFD validation.",
        "",
        "## Coordinate System & Flow Conditions",
        "- **X-axis**: Chordwise coordinate $x \\in [0.0, 1.0]\\,\\text{m}$ (unit chord $c = 1.0\\,\\text{m}$).",
        "- **Y-axis**: Airfoil camber and thickness normal direction (preserving exact Kulfan CST finite trailing-edge thickness $t_{\\text{TE}}$).",
        "- **Z-axis**: Quasi-2D spanwise extrusion $z \\in [0.0, 0.1]\\,\\text{m}$ (extrusion span $b = 0.1\\,\\text{m} = 0.1c$, directly compatible with OpenFOAM 1-cell-thick 2D RANS / ANSYS Fluent / STAR-CCM+ symmetry-plane setups).",
        f"- **Operating Condition**: Incompressible subsonic flow at $Re = {reynolds:.1e}$, angle of attack $\\alpha = {alpha:.1f}^\\circ$ (e.g., sea-level air $\\rho = 1.225\\,\\text{{kg/m}}^3$, $\\mu = 1.789 \\times 10^{{-5}}\\,\\text{{Pa}}\\cdot\\text{{s}}$, $U_\\infty \\approx 51.1\\,\\text{{m/s}}$ for $c = 1.0\\,\\text{{m}}$).",
        "- **Chordwise Discretization**: 200 cosine-clustered nodes per surface (leading-edge & trailing-edge refinement) with analytical strip-triangulated end-caps (guaranteed zero self-intersection even for high-camber profiles).",
        "",
        "## Model Inventory (Exact Paper Median Runs)",
        "",
        "| File Name | Design | Selected Seed | NeuralFoil $L/D$ | $C_L$ | $C_D$ | $C_M$ | $t_{\\max}/c$ | $t_{\\text{TE}}/c$ | Watertight | Volume ($\\text{m}^3$) | Surface Area ($\\text{m}^2$) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for fname, desc, seed_str, theta, info in items:
        mesh = build_extruded_airfoil_mesh(theta, chord=1.0, span=0.1, n_chord=200)
        stl_path = os.path.join(out_dir, fname)
        mesh.export(stl_path)
        print(
            f"[{desc:<26}] Seed={seed_str:>4} | L/D={info['ld']:6.2f} | "
            f"Watertight={mesh.is_watertight} | Vol={mesh.volume:.6f} m^3 | Area={mesh.area:.4f} m^2 -> {stl_path}"
        )
        md_lines.append(
            f"| `{fname}` | {desc} | `{seed_str}` | {info['ld']:.2f} | {info['cl']:.4f} | "
            f"{info['cd']:.5f} | {info['cm']:.4f} | {info['t_max']*100:.2f}% | {info['te_thickness']*100:.3f}% | "
            f"{mesh.is_watertight} | {mesh.volume:.6f} | {mesh.area:.4f} |"
        )

    md_lines.extend([
        "",
        "## CFD Meshing Recommendations",
        "- **Turbulence Model**: SST $k$-$\\omega$ (with $\\gamma$-$Re_\\theta$ transition model recommended to match NeuralFoil/XFoil natural transition physics at $Re = 3.5 \\times 10^6$).",
        "- **First Cell Wall Thickness**: For $Re = 3.5 \\times 10^6$ and $c = 1.0\\,\\text{m}$, target $y^+ \\le 1$ with $y_1 \\approx 6.5 \\times 10^{-6}\\,\\text{m}$ ($0.0065\\,\\text{mm}$), growth ratio $1.15\\text{--}1.20$, 35–45 inflation layers.",
        "- **Far-field Domain**: C-grid or O-grid with radius $\\ge 20c$ ($20\\,\\text{m}$) and spanwise `symmetry` / `empty` boundary conditions on $z = 0$ and $z = 0.1\\,\\text{m}$.",
    ])

    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"已生成 2D CFD 模型说明文档: {readme_path}")


if __name__ == "__main__":
    main()
