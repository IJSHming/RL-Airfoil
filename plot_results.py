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

def get_optimizer_order(results: dict) -> List[str]:
    return ["SPSA", "Py-BOBYQA", "DE", "PSO", "CMA-ES", "TuRBO"]


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
    optimizer_order = get_optimizer_order(results)

    for idx, opt_name in enumerate(optimizer_order):
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
    """图 2: 柱状图对比：
    1. 生成 6 张独立的优化器 10 随机种子 Vanilla vs RL-Meta 详细对照柱状图
    2. 生成 1 张 2x3 六合一全景图 (bar_comparison.png)
    """
    results = data["results"]
    summary = {row["optimizer"]: row for row in data["summary"]}
    optimizer_order = get_optimizer_order(results)

    # 1. 生成 6 张独立的单优化器柱状图
    for opt_name in optimizer_order:
        v_runs = results[opt_name]["Vanilla"]
        m_runs = results[opt_name]["RL-Meta"]
        n_seeds = len(v_runs)
        x = np.arange(n_seeds)
        width = 0.38

        v_scores = [r["best_score"] for r in v_runs]
        m_scores = [r["best_score"] for r in m_runs]
        seed_labels = [f"S{i+1}\n({r['seed']})" for i, r in enumerate(v_runs)]

        fig, ax = plt.subplots(figsize=(11, 5.5))
        rects1 = ax.bar(
            x - width / 2,
            v_scores,
            width,
            label=f"Vanilla (Mean: {np.mean(v_scores):.2f})",
            color=COLOR_VANILLA,
            edgecolor="#1E3A8A",
            linewidth=1.2,
            alpha=0.88,
        )
        rects2 = ax.bar(
            x + width / 2,
            m_scores,
            width,
            label=f"RL-Meta (Mean: {np.mean(m_scores):.2f})",
            color=COLOR_META,
            edgecolor="#991B1B",
            linewidth=1.2,
            alpha=0.88,
        )

        # 均值水平参考辅助线
        ax.axhline(np.mean(v_scores), color=COLOR_VANILLA, linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axhline(np.mean(m_scores), color=COLOR_META, linestyle="--", linewidth=1.5, alpha=0.7)

        # 标注柱上数值
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        gain = np.mean(m_scores) - np.mean(v_scores)
        pct = (gain / max(abs(np.mean(v_scores)), 1e-4)) * 100.0
        ax.set_ylabel("Final Penalized L/D ($C_L / C_D$)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Independent Seeds (NFE=300)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"{opt_name}: 10 Independent Seeds Comparison (Vanilla vs RL-Meta | Mean Gain: {gain:+.2f}, {pct:+.1f}%)",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(seed_labels, fontsize=10)
        ax.set_ylim(0, max(max(v_scores), max(m_scores)) * 1.18)
        ax.grid(True, axis="y", alpha=0.6)
        ax.legend(loc="upper right", fontsize=11)
        plt.tight_layout()

        safe_name = opt_name.lower().replace("-", "_")
        single_png = os.path.join(output_dir, f"bar_comparison_{safe_name}.png")
        single_pdf = os.path.join(output_dir, f"bar_comparison_{safe_name}.pdf")
        plt.savefig(single_png, dpi=300, bbox_inches="tight")
        plt.savefig(single_pdf, bbox_inches="tight")
        plt.close()

    # 2. 生成 2x3 六合一全景图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
    axes = axes.flatten()

    for idx, opt_name in enumerate(optimizer_order):
        ax = axes[idx]
        v_runs = results[opt_name]["Vanilla"]
        m_runs = results[opt_name]["RL-Meta"]
        n_seeds = len(v_runs)
        x = np.arange(n_seeds)
        width = 0.38

        v_scores = [r["best_score"] for r in v_runs]
        m_scores = [r["best_score"] for r in m_runs]
        seed_labels = [f"S{i+1}" for i in range(n_seeds)]

        rects1 = ax.bar(x - width / 2, v_scores, width, label="Vanilla", color=COLOR_VANILLA, alpha=0.85)
        rects2 = ax.bar(x + width / 2, m_scores, width, label="RL-Meta", color=COLOR_META, alpha=0.85)

        ax.axhline(np.mean(v_scores), color=COLOR_VANILLA, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.axhline(np.mean(m_scores), color=COLOR_META, linestyle="--", linewidth=1.2, alpha=0.8)

        gain = np.mean(m_scores) - np.mean(v_scores)
        pct = (gain / max(abs(np.mean(v_scores)), 1e-4)) * 100.0
        ax.set_title(f"{opt_name} (Δ: {gain:+.1f} / {pct:+.1f}%)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(seed_labels, fontsize=9)
        ax.grid(True, axis="y", alpha=0.5)
        ax.set_ylim(0, max(max(v_scores), max(m_scores)) * 1.18)
        if idx in [0, 3]:
            ax.set_ylabel("Final L/D ($C_L / C_D$)", fontsize=10, fontweight="bold")
        if idx >= 3:
            ax.set_xlabel("Seeds", fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"10-Seed Detailed Performance Comparison across 6 Optimizers (Vanilla vs RL-Meta)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = os.path.join(output_dir, "bar_comparison.png")
    pdf_path = os.path.join(output_dir, "bar_comparison.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"已生成 6 张单优化器柱状图与综合柱状图: {png_path}")


def plot_boxplot_comparison(data: dict, output_dir: str):
    """图 3: RL-Meta 与 Vanilla 最终升阻比分布的 Boxplot 箱线图"""
    results = data["results"]
    optimizer_order = get_optimizer_order(results)

    fig, ax = plt.subplots(figsize=(13, 6.5))

    positions = []
    box_data = []
    colors = []

    for i, opt in enumerate(optimizer_order):
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
        showfliers=False,  # 取消默认空心离群点，避免与全样本实心散点重叠
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

    tick_pos = [i * 2.5 + 1.2 for i in range(len(optimizer_order))]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(optimizer_order, fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.6)

    # 智能预留顶部空间，防止图例遮挡最高箱体
    all_scores = [s for sublist in box_data for s in sublist]
    y_min = max(0, min(all_scores) - 15)
    y_max = max(all_scores) + 35
    ax.set_ylim(y_min, y_max)

    # 自定义图例 (横向排列于顶部，避免遮挡数据箱体)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLOR_VANILLA, edgecolor="#333333", alpha=0.7, label="Vanilla"),
        Patch(facecolor=COLOR_META, edgecolor="#333333", alpha=0.7, label="RL-Meta"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="black", markersize=7, label="Mean Value"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", ncol=3, fontsize=11, framealpha=0.9)

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
    optimizer_order = get_optimizer_order(results)

    for idx, opt in enumerate(optimizer_order):
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
    optimizer_order = get_optimizer_order(results)

    # Panel A: 整体平均收敛轨迹对比
    ax_a = fig.add_subplot(gs[0, 0])
    all_v_hists = []
    all_m_hists = []
    for opt in optimizer_order:
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
    x = np.arange(len(optimizer_order))
    width = 0.35
    v_means = [summary[opt]["vanilla_mean"] for opt in optimizer_order]
    m_means = [summary[opt]["meta_mean"] for opt in optimizer_order]
    ax_b.bar(x - width/2, v_means, width, label="Vanilla", color=COLOR_VANILLA, alpha=0.85)
    ax_b.bar(x + width/2, m_means, width, label="RL-Meta", color=COLOR_META, alpha=0.85)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(optimizer_order, fontsize=11, fontweight="bold")
    ax_b.set_title("(B) Final L/D Performance Comparison", fontsize=13, fontweight="bold")
    ax_b.set_ylabel("Final L/D", fontsize=11, fontweight="bold")
    max_b_val = max(max(v_means), max(m_means))
    ax_b.set_ylim(0, max_b_val * 1.25)  # 留出充足顶部空间，防止图例遮挡柱体
    ax_b.legend(loc="upper left", ncol=2, fontsize=10.5, framealpha=0.9)
    ax_b.grid(True, axis="y")

    # Panel C: 箱线图对比
    ax_c = fig.add_subplot(gs[1, 0])
    positions = []
    box_data = []
    colors = []
    for i, opt in enumerate(optimizer_order):
        v_scores = [r["best_score"] for r in results[opt]["Vanilla"]]
        m_scores = [r["best_score"] for r in results[opt]["RL-Meta"]]
        positions.extend([i*2.2 + 0.7, i*2.2 + 1.4])
        box_data.extend([v_scores, m_scores])
        colors.extend([COLOR_VANILLA, COLOR_META])

    bp = ax_c.boxplot(box_data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax_c.set_xticks([i*2.2 + 1.05 for i in range(len(optimizer_order))])
    ax_c.set_xticklabels(optimizer_order, fontsize=11, fontweight="bold")
    ax_c.set_title("(C) Seed Robustness & Distribution (10 Seeds)", fontsize=13, fontweight="bold")
    ax_c.set_ylabel("Final L/D", fontsize=11, fontweight="bold")
    ax_c.grid(True, axis="y")

    # Panel D: 代表性翼型几何对照 (以 CMA-ES 与 TuRBO 为例) - 按要求取消图例显示
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
    # Subplot D: 取消图例显示 (per user request)
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
    parser.add_argument("--output_dir", type=str, default="figures/rl_meta", help="Output directory for generated figures")
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
