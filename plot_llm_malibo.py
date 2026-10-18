"""出版级专业可视化绘图脚本 (plot_llm_malibo.py)：
从 results/llm_malibo_benchmark_results.json / .npz 中读取数据，
针对四大优化方法（Vanilla-TuRBO, RL-Meta-TuRBO, LLM-Meta-TuRBO, MALIBO）生成 4 张高清学术图表：
1. 机翼形状对比 (figures/llm_malibo_airfoil_shapes.png / .pdf)
2. 10 次实验柱状图对比 (figures/llm_malibo_bar_comparison.png / .pdf)
3. 箱线图分布对比 (figures/llm_malibo_boxplot_comparison.png / .pdf)
4. 平均收敛曲线对比 (figures/llm_malibo_convergence_curves.png / .pdf)
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# 专业学术绘图风格
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "mathtext.fontset": "stix",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "legend.framealpha": 0.94,
    "legend.edgecolor": "#cccccc",
    "figure.autolayout": False,
})

# 专属配色方案 (四色专业学术对比)
COLOR_VANILLA = "#2563EB"   # 科技蓝 (Vanilla-TuRBO)
COLOR_RL_META = "#D97706"   # 琥珀橙 (RL-Meta-TuRBO)
COLOR_LLM_META = "#059669"  # 翡翠绿 (LLM-Meta-TuRBO)
COLOR_MALIBO = "#DC2626"    # 活力红 (MALIBO)
COLOR_BASELINE = "#6B7280"  # 沉稳灰 (NACA 0012)

COLORS = {
    "Vanilla-TuRBO": COLOR_VANILLA,
    "RL-Meta-TuRBO": COLOR_RL_META,
    "LLM-Meta-TuRBO": COLOR_LLM_META,
    "MALIBO": COLOR_MALIBO,
}

LABELS = {
    "Vanilla-TuRBO": "TuRBO",
    "RL-Meta-TuRBO": "RL-Meta-TuRBO",
    "LLM-Meta-TuRBO": "LLM-Meta-TuRBO",
    "MALIBO": "MALIBO",
}


def load_benchmark_data(data_path: str = "results/llm_malibo_benchmark_results.json") -> dict:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到基准数据文件: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================================================================
# 1. 机翼形状对比图 (Airfoil Shapes Comparison)
# ==============================================================================

def plot_airfoil_shapes(data: dict, output_dir: str = "figures"):
    """绘制中位数（第 5 名）代表性翼型几何剖面与关键物理指标标注"""
    os.makedirs(output_dir, exist_ok=True)
    results = data["results"]
    methods = ["Vanilla-TuRBO", "RL-Meta-TuRBO", "LLM-Meta-TuRBO", "MALIBO"]

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)

    # 载入基准 NACA0012 形状
    from src.airfoil_utils import get_naca0012_theta, theta_to_coordinates
    naca_theta = get_naca0012_theta()
    naca_coords = theta_to_coordinates(naca_theta)

    import matplotlib.patches as mpatches

    line_baseline = ax.plot(
        naca_coords["x_closed"],
        naca_coords["y_closed"],
        color=COLOR_BASELINE,
        linestyle=":",
        linewidth=1.8,
        label="Baseline NACA 0012 (L/D: 65.8)",
        zorder=2,
    )[0]
    ax.fill(naca_coords["x_closed"], naca_coords["y_closed"], color=COLOR_BASELINE, alpha=0.06)

    method_lines = {}
    for method in methods:
        data_list = results[method]
        scores = [r["best_score"] for r in data_list]
        med_idx = int(np.argsort(scores)[len(scores) // 2])
        rep_run = data_list[med_idx]

        coords = rep_run["coordinates"]
        x_c = np.array(coords["x_closed"])
        y_c = np.array(coords["y_closed"])

        color = COLORS[method]
        label = f"{LABELS[method]} (L/D: {rep_run['best_score']:.1f})"

        line_style = "-"
        lw = 2.2 if method == "MALIBO" else (2.0 if method == "LLM-Meta-TuRBO" else 1.8)
        line = ax.plot(x_c, y_c, color=color, linestyle=line_style, linewidth=lw, label=label, zorder=4)[0]
        method_lines[method] = line

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(-0.10, 0.18)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlabel("Normalized Chord Location ($x/c$)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalized Camber & Thickness ($y/c$)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Optimized Airfoil Geometry Profiles Comparison (Representative Median Runs)\n"
        f"Flow Conditions: $Re = 3.5 \\times 10^6, \\alpha = 3.5^\\circ$ | 300 NFE Budget",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    # 布局 2 列图例 (Matplotlib 按列优先填充)：
    # 第 1 列 (0, 1, 2): Baseline, TuRBO, 空白占位
    # 第 2 列 (3, 4, 5): RL-Meta-TuRBO, LLM-Meta-TuRBO, MALIBO
    empty_proxy = mpatches.Patch(color="none", label="")
    legend_handles = [
        line_baseline,
        method_lines["Vanilla-TuRBO"],
        empty_proxy,
        method_lines["RL-Meta-TuRBO"],
        method_lines["LLM-Meta-TuRBO"],
        method_lines["MALIBO"],
    ]
    legend_labels = [h.get_label() for h in legend_handles]
    ax.legend(legend_handles, legend_labels, loc="upper right", ncol=2, fontsize=10.5, framealpha=0.45)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "llm_malibo_airfoil_shapes.png")
    pdf_path = os.path.join(output_dir, "llm_malibo_airfoil_shapes.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"[图表保存] 机翼形状对比已输出: {png_path} & {pdf_path}")


# ==============================================================================
# 2. 10 次独立实验柱状图对比 (Bar Comparison)
# ==============================================================================

def plot_bar_comparison(data: dict, output_dir: str = "figures"):
    """绘制 10 个独立测试种子的分组柱状图以及总体 Mean ± Std 概览"""
    os.makedirs(output_dir, exist_ok=True)
    results = data["results"]
    seeds = data["condition"]["seeds"]
    n_seeds = len(seeds)
    methods = ["Vanilla-TuRBO", "RL-Meta-TuRBO", "LLM-Meta-TuRBO", "MALIBO"]

    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=300)

    x = np.arange(n_seeds)
    width = 0.19

    for i, method in enumerate(methods):
        scores = [r["best_score"] for r in results[method]]
        pos = x + (i - 1.5) * width
        rects = ax.bar(
            pos,
            scores,
            width,
            label=LABELS[method],
            color=COLORS[method],
            alpha=0.88,
            edgecolor="#222222",
            linewidth=0.8,
            zorder=3,
        )

    # 标记基线初始 L/D 参考线
    ax.axhline(65.8, color=COLOR_BASELINE, linestyle=":", linewidth=1.5, label="Initial NACA 0012 (65.8)", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=11, fontweight="bold")
    ax.set_xlabel("Independent Evaluation Seeds", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("Optimal Penalized Lift-to-Drag Ratio ($L/D$)", fontsize=12, fontweight="bold")
    ax.set_ylim(130, 270)
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=1)

    cond = data["condition"]
    ax.set_title(
        f"Comparative Benchmark across 10 Independent Seeds\n"
        f"Flow Conditions: $Re = {cond['reynolds']/1e6:.1f}\\times 10^6, \\alpha = {cond['alpha']}^\\circ$ | NFE = {cond['max_evals']}",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.legend(loc="upper left", ncol=3, fontsize=10.5, framealpha=0.92)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "llm_malibo_bar_comparison.png")
    pdf_path = os.path.join(output_dir, "llm_malibo_bar_comparison.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"[图表保存] 10 次实验柱状图对比已输出: {png_path} & {pdf_path}")


# ==============================================================================
# 3. 箱线图对比 (Boxplot Comparison)
# ==============================================================================

def plot_boxplot_comparison(data: dict, output_dir: str = "figures"):
    """绘制四大优化方法的性能分布箱线图 (含散点抖动抖动与均值菱形)"""
    os.makedirs(output_dir, exist_ok=True)
    results = data["results"]
    methods = ["Vanilla-TuRBO", "RL-Meta-TuRBO", "LLM-Meta-TuRBO", "MALIBO"]

    fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=300)

    score_data = [[r["best_score"] for r in results[m]] for m in methods]
    means = [np.mean(s) for s in score_data]

    box = ax.boxplot(
        score_data,
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="#111111", linewidth=2.0),
        flierprops=dict(marker="o", markersize=6, alpha=0.7),
        zorder=3,
    )
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels([LABELS[m] for m in methods])

    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.65)
        patch.set_edgecolor(COLORS[method])
        patch.set_linewidth(1.8)

    # 叠加数据散点 (Jitter) 与 均值标记 (Diamond)
    for i, (m, scores) in enumerate(zip(methods, score_data)):
        x_jitter = np.random.normal(i + 1, 0.04, size=len(scores))
        ax.scatter(x_jitter, scores, color=COLORS[m], edgecolor="#111111", s=55, alpha=0.85, zorder=5)
        mean_val = np.mean(scores)
        ax.scatter(i + 1, mean_val, marker="D", color="#ffffff", edgecolor=COLORS[m], s=75, linewidth=2.0, zorder=6)
        ax.text(
            i + 1,
            mean_val + 3.5,
            f"Mean: {mean_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#111111",
            zorder=7,
        )

    ax.set_ylim(155, 258)
    ax.set_ylabel("Final Best Penalized Lift-to-Drag Ratio ($L/D$)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=1)
    cond = data["condition"]
    ax.set_title(
        f"Statistical Performance Distribution across 10 Independent Seeds (Boxplot + Jitter)\n"
        f"$Re = {cond['reynolds']/1e6:.1f}\\times 10^6, \\alpha = {cond['alpha']}^\\circ$ | 300 NFE",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    plt.tight_layout()
    png_path = os.path.join(output_dir, "llm_malibo_boxplot_comparison.png")
    pdf_path = os.path.join(output_dir, "llm_malibo_boxplot_comparison.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"[图表保存] 箱线图对比已输出: {png_path} & {pdf_path}")


# ==============================================================================
# 4. 平均收敛曲线对比 (Convergence Curves Comparison)
# ==============================================================================

def plot_convergence_curves(data: dict, output_dir: str = "figures"):
    """绘制 10 次实验全程 300 步的平均收敛轨迹与 ±1 Std 置信带"""
    os.makedirs(output_dir, exist_ok=True)
    results = data["results"]
    max_evals = data["condition"]["max_evals"]
    steps = np.arange(1, max_evals + 1)
    methods = ["Vanilla-TuRBO", "RL-Meta-TuRBO", "LLM-Meta-TuRBO", "MALIBO"]

    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=300)

    for method in methods:
        hists = np.array([r["history"] for r in results[method]])
        mean = np.mean(hists, axis=0)
        std = np.std(hists, axis=0)
        color = COLORS[method]
        label = f"{LABELS[method]} (Final: {mean[-1]:.1f} ± {std[-1]:.1f})"

        lw = 2.6 if method == "MALIBO" else (2.4 if method == "LLM-Meta-TuRBO" else 2.0)
        ax.plot(steps, mean, label=label, color=color, linewidth=lw, zorder=4)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15, zorder=3)

    # 阶段标注 (前中期与后期)
    ax.axvline(100, color="#888888", linestyle=":", linewidth=1.2)
    ax.text(50, 80, "Early-Mid Phase\n(Fast Exploration)", ha="center", fontsize=10, style="italic", color="#555555")
    ax.axvline(200, color="#888888", linestyle=":", linewidth=1.2)
    ax.text(150, 80, "Mid-Late Phase\n(Exploitation)", ha="center", fontsize=10, style="italic", color="#555555")
    ax.text(250, 80, "Final Convergence\n(High-Precision)", ha="center", fontsize=10, style="italic", color="#555555")

    ax.set_xlim(1, max_evals)
    ax.set_ylim(60, 260)
    ax.set_xlabel("Number of Function Evaluations (NFE)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Best Penalized Lift-to-Drag Ratio ($L/D$)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)

    cond = data["condition"]
    ax.set_title(
        f"Mean Convergence Trajectories across 10 Seeds (Mean ± Std Shaded Band)\n"
        f"Flow Conditions: $Re = {cond['reynolds']/1e6:.1f}\\times 10^6, \\alpha = {cond['alpha']}^\\circ$ | Initial Airfoil: NACA 0012",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.legend(loc="upper left", fontsize=10.5, framealpha=0.88)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "llm_malibo_convergence_curves.png")
    pdf_path = os.path.join(output_dir, "llm_malibo_convergence_curves.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"[图表保存] 收敛曲线对比已输出: {png_path} & {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="生成四大方法对比的 4 张出版级高清图表")
    parser.add_argument("--data", type=str, default="results/llm_malibo_benchmark_results.json")
    parser.add_argument("--output_dir", type=str, default="figures/llm_malibo")
    args = parser.parse_args()

    data = load_benchmark_data(args.data)
    plot_airfoil_shapes(data, args.output_dir)
    plot_bar_comparison(data, args.output_dir)
    plot_boxplot_comparison(data, args.output_dir)
    plot_convergence_curves(data, args.output_dir)
    print("\n>>> 所有 4 张学术对比图表 (PNG + PDF) 已全部成功生成！")


if __name__ == "__main__":
    main()
