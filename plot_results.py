"""RL-Meta-Airfoil 专业可视化绘图脚本：
从基准测试结果文件 (JSON / NPZ) 中读取数据，
独立绘制并输出出版级高分辨率图表：
1. 10 次独立实验的平均收敛曲线与置信带 (convergence_curves)
2. 10 次实验最终升阻比的均值柱状对比图与提升幅度 (bar_comparison)
3. 10 次实验最终升阻比分布的 Boxplot 箱线图 (boxplot_comparison)
4. 10 次实验中取中位数（第 5 名）的最优翼型几何形状对照图 (airfoil_shapes)
5. 2x2 四宫格全景汇总大图 (comprehensive_overview)
同时保存为 300 DPI PNG 与 PDF 矢量图。
"""

import argparse
import json
import os
import sys
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

# 设置专业学术绘图风格
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "mathtext.fontset": "stix",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc",
    "figure.autolayout": False,
})

OPTIMIZER_ORDER = ["SPSA", "Py-BOBYQA", "GA", "PSO", "CMA-ES", "TuRBO"]
COLOR_VANILLA = "#2563EB"   # 科技蓝
COLOR_META = "#DC2626"      # 活力红
COLOR_NACA = "#4B5563"      # 沉稳灰


def load_data(data_path: str = "results/benchmark_results.json"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到测试数据文件: {data_path}，请先运行 benchmark.py 进行测试！")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def plot_convergence_curves(data: dict, output_dir: str):
    """图 1: 10 次独立实验的平均收敛曲线 (Mean ± Std)"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=False)
    axes = axes.flatten()

    results = data["results"]
    max_evals = data["condition"]["max_evals"]
    steps = np.arange(1, max_evals + 1)

    for idx, opt_name in enumerate(OPTIMIZER_ORDER):
        ax = axes[idx]
        v_hists = np.array([r["history"] for r in results[opt_name]["Vanilla"]])
        m_hists = np.array([r["history"] for r in results[opt_name]["RL-Meta"]])

        v_mean, v_std = np.mean(v_hists, axis=0), np.std(v_hists, axis=0)
        m_mean, m_std = np.mean(m_hists, axis=0), np.std(m_hists, axis=0)

        # 绘制均值实线
        ax.plot(steps, v_mean, label="Vanilla", color=COLOR_VANILLA, linewidth=2.2)
        ax.fill_between(steps, v_mean - v_std, v_mean + v_std, color=COLOR_VANILLA, alpha=0.18)

        ax.plot(steps, m_mean, label="RL-Meta", color=COLOR_META, linewidth=2.2)
        ax.fill_between(steps, m_mean - m_std, m_mean + m_std, color=COLOR_META, alpha=0.18)

        # 标记最终均值
        ax.scatter([max_evals], [v_mean[-1]], color=COLOR_VANILLA, s=45, zorder=5)
        ax.scatter([max_evals], [m_mean[-1]], color=COLOR_META, s=45, zorder=5)

        gain = m_mean[-1] - v_mean[-1]
        pct = (gain / max(abs(v_mean[-1]), 1e-4)) * 100.0
        ax.set_title(f"{opt_name} (Δ: {gain:+.1f} / {pct:+.1f}%)", fontsize=13, fontweight="bold", pad=8)
        ax.grid(True)
        ax.set_xlim(0, max_evals)

        if idx in [0, 3]:
            ax.set_ylabel("Best Penalized L/D ($C_L / C_D$)", fontsize=11, fontweight="bold")
        if idx >= 3:
            ax.set_xlabel("Function Evaluations (NFE)", fontsize=11, fontweight="bold")

        ax.legend(loc="lower right", fontsize=10)

    fig.suptitle(
        f"Convergence Curves across 10 Independent Seeds (Re={data['condition']['reynolds']/1e6:.1f}M, AoA={data['condition']['alpha']}°)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = os.path.join(output_dir, "convergence_curves.png")
    pdf_path = os.path.join(output_dir, "convergence_curves.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"已生成收敛曲线图: {png_path} & {pdf_path}")


def plot_bar_comparison(data: dict, output_dir: str):
    """图 2: 10 次实验最终升阻比的均值柱状对比图与提升幅度"""
    summary = {row["optimizer"]: row for row in data["summary"]}

    x = np.arange(len(OPTIMIZER_ORDER))
    width = 0.36

    v_means = [summary[opt]["vanilla_mean"] for opt in OPTIMIZER_ORDER]
    v_stds = [summary[opt]["vanilla_std"] for opt in OPTIMIZER_ORDER]
    m_means = [summary[opt]["meta_mean"] for opt in OPTIMIZER_ORDER]
    m_stds = [summary[opt]["meta_std"] for opt in OPTIMIZER_ORDER]

    fig, ax = plt.subplots(figsize=(12, 6.5))

    rects1 = ax.bar(
        x - width / 2,
        v_means,
        width,
        yerr=v_stds,
        capsize=5,
        label="Vanilla",
        color=COLOR_VANILLA,
        edgecolor="#1E3A8A",
        linewidth=1.2,
        alpha=0.88,
    )
    rects2 = ax.bar(
        x + width / 2,
        m_means,
        width,
        yerr=m_stds,
        capsize=5,
        label="RL-Meta",
        color=COLOR_META,
        edgecolor="#991B1B",
        linewidth=1.2,
        alpha=0.88,
    )

    ax.set_ylabel("Final Penalized L/D ($C_L / C_D$)", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Final L/D Comparison: Vanilla vs RL-Meta across 6 Optimizers (10 Seeds, NFE=300)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(OPTIMIZER_ORDER, fontsize=12, fontweight="bold")
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, axis="y", alpha=0.6)

    # 标注数值与百分比增益
    for i, opt in enumerate(OPTIMIZER_ORDER):
        gain = summary[opt]["gain"]
        pct = summary[opt]["pct_gain"]
        top = max(v_means[i] + v_stds[i], m_means[i] + m_stds[i])
        color_text = "#047857" if gain >= 0 else "#B91C1C"
        ax.annotate(
            f"{gain:+.1f}\n({pct:+.1f}%)",
            xy=(x[i] + width / 2, m_means[i]),
            xytext=(x[i], top + 4.0),
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=color_text,
            arrowprops=dict(arrowstyle="->", color=color_text, lw=1.2),
        )

    y_max = max(max(v_means), max(m_means)) * 1.22
    ax.set_ylim(0, y_max)
    plt.tight_layout()

    png_path = os.path.join(output_dir, "bar_comparison.png")
    pdf_path = os.path.join(output_dir, "bar_comparison.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"已生成柱状对比图: {png_path} & {pdf_path}")


def plot_boxplot_comparison(data: dict, output_dir: str):
    """图 3: RL-Meta 与 Vanilla 最终升阻比分布的 Boxplot 箱线图"""
    results = data["results"]

    fig, ax = plt.subplots(figsize=(13, 6.5))

    positions = []
    box_data = []
    colors = []

    for i, opt in enumerate(OPTIMIZER_ORDER):
        v_scores = [r["best_score"] for r in results[opt]["Vanilla"]]
        m_scores = [r["best_score"] for r in results[opt]["RL-Meta"]]

        pos_v = i * 2.5 + 0.8
        pos_m = i * 2.5 + 1.6
        positions.extend([pos_v, pos_m])
        box_data.extend([v_scores, m_scores])
        colors.extend([COLOR_VANILLA, COLOR_META])

    bplot = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white", markersize=6),
        medianprops=dict(color="#111827", linewidth=2.0),
    )

    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.70)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.3)

    # 散点抖动展示真实点分布
    for idx, (pos, vals) in enumerate(zip(positions, box_data)):
        jitter = np.random.RandomState(42).uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(pos + jitter, vals, color=colors[idx], edgecolor="#111111", s=32, alpha=0.9, zorder=5)

    ax.set_ylabel("Final Penalized L/D ($C_L / C_D$)", fontsize=13, fontweight="bold")
    ax.set_title("Distribution of Final L/D across 10 Independent Seeds (Boxplot & Data Points)", fontsize=14, fontweight="bold", pad=15)

    tick_pos = [i * 2.5 + 1.2 for i in range(len(OPTIMIZER_ORDER))]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(OPTIMIZER_ORDER, fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.6)

    # 自定义图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLOR_VANILLA, edgecolor="#333333", alpha=0.7, label="Vanilla"),
        Patch(facecolor=COLOR_META, edgecolor="#333333", alpha=0.7, label="RL-Meta"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="black", markersize=7, label="Mean Value"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=11)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "boxplot_comparison.png")
    pdf_path = os.path.join(output_dir, "boxplot_comparison.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"已生成箱线图: {png_path} & {pdf_path}")


def plot_airfoil_shapes(data: dict, output_dir: str):
    """图 4: 10 次实验中取中位数（第 5 名）的最优翼型几何形状对照图"""
    from src.airfoil_utils import get_naca0012_theta, theta_to_coordinates

    naca0012_coords = theta_to_coordinates(get_naca0012_theta())

    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True, sharey=True)
    axes = axes.flatten()

    results = data["results"]

    for idx, opt in enumerate(OPTIMIZER_ORDER):
        ax = axes[idx]

        # 排序寻找第 5 名 (中位数解，按 10 种子从高到低排序索引 4)
        v_runs = sorted(results[opt]["Vanilla"], key=lambda r: r["best_score"], reverse=True)
        m_runs = sorted(results[opt]["RL-Meta"], key=lambda r: r["best_score"], reverse=True)

        v_median = v_runs[4]  # 第 5 名
        m_median = m_runs[4]  # 第 5 名

        v_coords = v_median["coordinates"]
        m_coords = m_median["coordinates"]

        # 绘制初始 NACA0012 基准
        ax.plot(
            naca0012_coords["x_closed"],
            naca0012_coords["y_closed"],
            linestyle="--",
            color=COLOR_NACA,
            linewidth=1.8,
            label="Initial NACA 0012",
        )

        # 绘制 Vanilla 最优解
        ax.plot(
            v_coords["x_closed"],
            v_coords["y_closed"],
            color=COLOR_VANILLA,
            linewidth=2.2,
            label=f"Vanilla (5th L/D: {v_median['best_score']:.1f})",
        )

        # 绘制 RL-Meta 最优解
        ax.plot(
            m_coords["x_closed"],
            m_coords["y_closed"],
            color=COLOR_META,
            linewidth=2.4,
            label=f"RL-Meta (5th L/D: {m_median['best_score']:.1f})",
        )

        ax.set_aspect("equal", "box")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_title(f"{opt} Airfoil Geometry (Median / 5th Rank)", fontsize=12, fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.16, 0.20)
        ax.legend(loc="upper right", fontsize=9.5)

        if idx in [0, 2, 4]:
            ax.set_ylabel("y / c", fontsize=11, fontweight="bold")
        if idx >= 4:
            ax.set_xlabel("x / c", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Optimized Airfoil Geometries Comparison (Median 5th Rank of 10 Runs)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = os.path.join(output_dir, "airfoil_shapes.png")
    pdf_path = os.path.join(output_dir, "airfoil_shapes.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"已生成翼型几何对照图: {png_path} & {pdf_path}")


def plot_comprehensive_overview(data: dict, output_dir: str):
    """图 5: 2x2 四宫格全景汇总大图"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    results = data["results"]
    max_evals = data["condition"]["max_evals"]
    steps = np.arange(1, max_evals + 1)
    summary = {row["optimizer"]: row for row in data["summary"]}

    # Panel A: 整体平均收敛轨迹对比
    ax_a = fig.add_subplot(gs[0, 0])
    all_v_hists = []
    all_m_hists = []
    for opt in OPTIMIZER_ORDER:
        v_h = np.mean([r["history"] for r in results[opt]["Vanilla"]], axis=0)
        m_h = np.mean([r["history"] for r in results[opt]["RL-Meta"]], axis=0)
        all_v_hists.append(v_h)
        all_m_hists.append(m_h)
        ax_a.plot(steps, v_h, color=COLOR_VANILLA, alpha=0.35, linewidth=1.2)
        ax_a.plot(steps, m_h, color=COLOR_META, alpha=0.35, linewidth=1.2)

    grand_v_mean = np.mean(all_v_hists, axis=0)
    grand_m_mean = np.mean(all_m_hists, axis=0)
    ax_a.plot(steps, grand_v_mean, color=COLOR_VANILLA, linewidth=3.2, label="Vanilla Grand Mean")
    ax_a.plot(steps, grand_m_mean, color=COLOR_META, linewidth=3.2, label="RL-Meta Grand Mean")
    ax_a.set_title("(A) Grand Average Convergence Trajectory (NFE=300)", fontsize=13, fontweight="bold")
    ax_a.set_xlabel("Evaluations", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Penalized L/D", fontsize=11, fontweight="bold")
    ax_a.legend(loc="lower right", fontsize=11)
    ax_a.grid(True)

    # Panel B: 柱状图对比
    ax_b = fig.add_subplot(gs[0, 1])
    x = np.arange(len(OPTIMIZER_ORDER))
    width = 0.35
    v_means = [summary[opt]["vanilla_mean"] for opt in OPTIMIZER_ORDER]
    m_means = [summary[opt]["meta_mean"] for opt in OPTIMIZER_ORDER]
    ax_b.bar(x - width/2, v_means, width, label="Vanilla", color=COLOR_VANILLA, alpha=0.85)
    ax_b.bar(x + width/2, m_means, width, label="RL-Meta", color=COLOR_META, alpha=0.85)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(OPTIMIZER_ORDER, fontsize=11, fontweight="bold")
    ax_b.set_title("(B) Final L/D Performance Comparison", fontsize=13, fontweight="bold")
    ax_b.set_ylabel("Final L/D", fontsize=11, fontweight="bold")
    ax_b.legend(loc="upper left", fontsize=11)
    ax_b.grid(True, axis="y")

    # Panel C: 箱线图对比
    ax_c = fig.add_subplot(gs[1, 0])
    positions = []
    box_data = []
    colors = []
    for i, opt in enumerate(OPTIMIZER_ORDER):
        v_scores = [r["best_score"] for r in results[opt]["Vanilla"]]
        m_scores = [r["best_score"] for r in results[opt]["RL-Meta"]]
        positions.extend([i*2.2 + 0.7, i*2.2 + 1.4])
        box_data.extend([v_scores, m_scores])
        colors.extend([COLOR_VANILLA, COLOR_META])

    bp = ax_c.boxplot(box_data, positions=positions, widths=0.55, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax_c.set_xticks([i*2.2 + 1.05 for i in range(len(OPTIMIZER_ORDER))])
    ax_c.set_xticklabels(OPTIMIZER_ORDER, fontsize=11, fontweight="bold")
    ax_c.set_title("(C) Seed Robustness & Distribution (10 Seeds)", fontsize=13, fontweight="bold")
    ax_c.set_ylabel("Final L/D", fontsize=11, fontweight="bold")
    ax_c.grid(True, axis="y")

    # Panel D: 代表性翼型几何对照 (以 CMA-ES 与 TuRBO 为例)
    ax_d = fig.add_subplot(gs[1, 1])
    from src.airfoil_utils import get_naca0012_theta, theta_to_coordinates
    naca_c = theta_to_coordinates(get_naca0012_theta())
    ax_d.plot(naca_c["x_closed"], naca_c["y_closed"], "--", color=COLOR_NACA, label="NACA 0012 (Init)")

    turbo_v = sorted(results["TuRBO"]["Vanilla"], key=lambda r: r["best_score"], reverse=True)[4]
    turbo_m = sorted(results["TuRBO"]["RL-Meta"], key=lambda r: r["best_score"], reverse=True)[4]
    ax_d.plot(turbo_v["coordinates"]["x_closed"], turbo_v["coordinates"]["y_closed"], color=COLOR_VANILLA, label=f"Vanilla (L/D: {turbo_v['best_score']:.1f})")
    ax_d.plot(turbo_m["coordinates"]["x_closed"], turbo_m["coordinates"]["y_closed"], color=COLOR_META, label=f"RL-Meta (L/D: {turbo_m['best_score']:.1f})")
    ax_d.set_aspect("equal", "box")
    ax_d.set_title("(D) Airfoil Geometry Profiling (TuRBO 5th Median)", fontsize=13, fontweight="bold")
    ax_d.set_xlabel("x / c", fontsize=11, fontweight="bold")
    ax_d.set_ylabel("y / c", fontsize=11, fontweight="bold")
    ax_d.legend(loc="upper right", fontsize=10.5)
    ax_d.grid(True, linestyle=":")

    fig.suptitle("RL-Meta-Airfoil Comprehensive Meta-Optimization Benchmark Summary", fontsize=16, fontweight="bold", y=0.98)

    png_path = os.path.join(output_dir, "comprehensive_overview.png")
    pdf_path = os.path.join(output_dir, "comprehensive_overview.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"已生成全景汇总大图: {png_path} & {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="RL-Meta-Airfoil Visualizer")
    parser.add_argument("--data_file", type=str, default="results/benchmark_results.json", help="Path to benchmark results JSON")
    parser.add_argument("--output_dir", type=str, default="figures", help="Output directory for generated figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_data(args.data_file)

    print("开始生成所有对比图表...")
    plot_convergence_curves(data, args.output_dir)
    plot_bar_comparison(data, args.output_dir)
    plot_boxplot_comparison(data, args.output_dir)
    plot_airfoil_shapes(data, args.output_dir)
    plot_comprehensive_overview(data, args.output_dir)
    print(f"\n全部图表生成完毕！文件保存在: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
