# RL-Meta-Airfoil: 强化学习元优化黑盒翼型气动外形设计

本项目基于深度强化学习（PPO），针对 6 种经典与前沿黑盒优化算法（SPSA、Py-BOBYQA、GA、PSO、CMA-ES、TuRBO）构建了**动态超参数元优化控制系统（Learning to Optimize, L2O）**。

优化目标为**最大化升阻比（$C_L / C_D$）**，同时受到最大相对厚度、结构梁最小厚度、后缘防鱼尾单调收缩、前缘钝度半径、后缘闭合楔角、截面面积、抗凹陷凸性等全套工业级强物理几何约束。参数化采用 **Kulfan (CST) 18 维表示**，优化初始基准统一为 **NACA 0012**，评估预算严格统一为 **300 次函数评估（NFE = 300）**。

---

## 1. 核心对比实验数据 (10 个独立随机种子)

在统一测试工况（$Re=3.5 \times 10^6, \alpha=3.5^\circ$, 初始 NACA 0012 升阻比约为 $65.80$, 严格预算 NFE=300）下，Vanilla 基准与 RL-Meta 的 10 随机种子独立实验统计结果如下：

| 优化算法 | 算法类型 | Vanilla 基准升阻比 ($L/D$) | RL-Meta 强化升阻比 ($L/D$) | 绝对提升 ($\Delta$) | 相对增益 (%) | 10 种子可行率 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TuRBO** | 高维贝叶斯信赖域 (GP + TS) | $191.36 \pm 34.40$ | **$212.09 \pm 15.69$** | **$+20.73$** | **$+10.8\%$** | **100% (10/10)** |
| **CMA-ES** | 协方差矩阵自适应演化策略 | $193.85 \pm 11.55$ | **$203.96 \pm 15.98$** | **$+10.12$** | **$+5.2\%$** | **100% (10/10)** |
| **PSO** | 粒子群群智能优化 | **$191.34 \pm 15.50$** | $186.31 \pm 16.71$ | $-5.02$ | $-2.6\%$ | **100% (10/10)** |
| **SPSA** | 随机逼近一阶模拟 (Spall标准) | **$193.21 \pm 21.47$** | $134.85 \pm 24.10$ | $-58.36$ | $-30.2\%$ | **100% (10/10)** |
| **GA** | 进化算法 (差分变异+二项交叉) | **$178.35 \pm 29.34$** | $151.93 \pm 20.77$ | $-26.43$ | $-14.8\%$ | **100% (10/10)** |
| **Py-BOBYQA** | 二次逼近无导数信赖域 | **$136.19 \pm 30.95$** | $126.46 \pm 11.20$ | $-9.74$ | $-7.2\%$ | **100% (10/10)** |

> **关键实验结论**：
> 1. **高容量全局代理与高阶策略获益最显著**：贝叶斯优化（TuRBO）在 Meta-RL 动态自适应信赖域与采样温度的调控下，升阻比从 191.36 跃升至 **212.09 (+10.8%)**，且标准差减半（方差显著降低，稳定性大幅提升）；CMA-ES 全局步长自适应同样取得稳定正增益（**+5.2%**，均值突破 203）。
> 2. **100% 工业几何可行性保证**：通过在 `BaseAirfoilOptimizer` 与 `AerodynamicEvaluator` 中部署严格可行解门禁拦截与梯度范数截断保护，**所有 10 轮种子下优化生成的最终翼型均 100% 满足工业几何约束**（最大相对厚度、结构梁厚度、光滑度与单调性校验全部通过，彻底杜绝局部波浪与“鱼尾”变形）。
> 3. **Spall 准则调优与 GA 高阶算子升级**：SPSA 在应用 Spall 经典衰减标准与微步长后，基准升阻比由原先的 ~90 飙升至 **193.21**；GA 在由简单高斯变异重构为 `current-to-best` 连续差分演化算子后，基准由原先的 110 跃升至 **178.35**（最高单次达 228.41）。
> 4. **Py-BOBYQA 随机化恢复**：通过配置 `init.random_initial_directions: True` 并注入独立种子，成功消除了原先 10 种子 0 方差的确定性问题。

---

## 2. 状态与动作空间设计

各优化器共享通用优化动态基础特征（评估进度 $t/300$、归一化雷诺数、攻角、历史相对收益、近期提升率与停滞比例，共 6 维），并结合各优化器内部专有动力学特征：

- **SPSA (10维 State / 2维 Action)**: 状态增加步长对数、扰动对数、梯度范数、前后步梯度余弦夹角；动作动态微调学习率乘子 $\mu_a$ 与扰动步长乘子 $\mu_c$。
- **Py-BOBYQA (10维 State / 1维 Action)**: 状态增加信赖域半径 $\Delta$、收敛半径 $\rho$、二次模型梯度范数；动作动态调节信赖域自适应伸缩乘子 $\kappa_\Delta$。
- **GA (11维 State / 3维 Action)**: 状态增加交叉率 $CR$、差分步长 $F$、变异尺度 $\sigma_m$、种群适应度方差、种群中心多样性；动作调节交叉率 $CR$、差分引导权重 $F$ 与局部微扰尺度 $\sigma_m$。
- **PSO (11维 State / 3维 Action)**: 状态增加粒子群分散度、平均速度范数；动作动态调节惯性权重 $w$、个体认知权重 $c_1$ 与社会协同权重 $c_2$。
- **CMA-ES (9维 State / 1维 Action)**: 状态增加全局步长 $\sigma$、进化路径长度比值、协方差矩阵条件数；动作动态调节全局步长乘子 $\kappa_\sigma$。
- **TuRBO (10维 State / 2维 Action)**: 状态增加局部信赖域尺度 $L$、连续成功/失败计数、GP 后验不确定度；动作调节信赖域尺度 $\kappa_L$ 与汤普森采样探索温度 $\tau$。

---

## 3. 代码仓库结构

```bash
RL-Airfoil/
├── src/
│   ├── evaluator.py          # 工业级气动与几何约束评估器 (NeuralFoil + AeroSandbox + 严格可行门禁)
│   ├── airfoil_utils.py      # NACA0012 初始参数、18维 Kulfan 边界与闭合坐标生成
│   ├── ppo.py                # PyTorch PPO Actor-Critic 双头网络、GAE 与数值防溢出模块
│   └── optimizers/           # 6 大黑盒优化器实现 (支持 Vanilla 与 RL-Meta 模式)
│       ├── base.py           # 优化器基类 (严格可行解检查、300 NFE 计数、状态提取)
│       ├── spsa.py           # SPSA 优化器 (Spall准则超参 + 梯度范数裁剪保护)
│       ├── bobyqa.py         # Py-BOBYQA 优化器 (随机初始方向 + 动态信赖域钩子)
│       ├── ga.py             # 遗传算法 (current-to-best 连续差分变异 + 二项交叉)
│       ├── pso.py            # 粒子群算法 (可行解个人历史最优 pbest 防护)
│       ├── cma_es.py         # CMA-ES 优化器 (官方 cma 库 + 动态步长调节)
│       └── turbo.py          # TuRBO 信赖域贝叶斯优化 (GP + Cholesky 汤普森采样)
├── train.py                  # PPO 元优化器训练脚本 (110 气动网格工况，支持全排列遍历)
├── benchmark.py              # 10 随机种子独立对比实验脚本 (记录 300 步完整历史与最终形状)
├── plot_results.py           # 独立出版级绘图脚本 (读取结果文件输出 6 张单优化器柱状图与全景图)
├── weights/                  # 训练保存的 6 个 Meta-PPO 策略权重 (.pt)
├── results/                  # 10 随机种子完整评估轨迹与最终几何数据 (JSON & NPZ)
├── figures/                  # 高清科研对比图表 (300 DPI PNG 与 PDF 矢量图)
│   ├── bar_comparison_spsa.png / .pdf       # SPSA 10 种子具体对比柱状图
│   ├── bar_comparison_py_bobyqa.png / .pdf  # Py-BOBYQA 10 种子具体对比柱状图
│   ├── bar_comparison_ga.png / .pdf         # GA 10 种子具体对比柱状图
│   ├── bar_comparison_pso.png / .pdf        # PSO 10 种子具体对比柱状图
│   ├── bar_comparison_cma_es.png / .pdf     # CMA-ES 10 种子具体对比柱状图
│   ├── bar_comparison_turbo.png / .pdf      # TuRBO 10 种子具体对比柱状图
│   ├── bar_comparison_6panels.png           # 2x3 六合一柱状图全景对比
│   ├── convergence_curves.png / .pdf        # 10 种子平均收敛曲线与置信带
│   ├── boxplot_comparison.png / .pdf        # 10 种子最终升阻比分布箱线图与真实数据散点
│   ├── airfoil_shapes.png / .pdf            # 10 种子取中位数（第5名）的最优翼型几何形状对照
│   └── comprehensive_overview.png / .pdf    # 2x2 全景大图汇总
└── README.md
```

---

## 4. 运行说明

### 环境安装
```bash
uv pip install aerosandbox neuralfoil py-bobyqa cma botorch gpytorch torch matplotlib pandas scipy
```

### 1. 训练 PPO 元优化器 (遍历全部 110 种网格工况)
```bash
python train.py --episodes 110 --device cuda
```

### 2. 运行 10 随机种子独立对比实验
```bash
python benchmark.py --reynolds 3.5e6 --alpha 3.5 --max_evals 300 --device cuda
```

### 3. 生成独立图表
```bash
python plot_results.py --data_file results/benchmark_results.json --output_dir figures
```
