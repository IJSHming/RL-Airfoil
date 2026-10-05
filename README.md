# RL-Meta-Airfoil: 强化学习元优化黑盒翼型气动外形设计

本项目基于深度强化学习（PPO），针对 6 种经典与前沿黑盒优化算法（SPSA、Py-BOBYQA、GA、PSO、CMA-ES、TuRBO）构建了**动态超参数元优化控制系统（Learning to Optimize, L2O）**。

优化目标为**最大化升阻比（$C_L / C_D$）**，同时受到最大相对厚度、结构梁最小厚度、后缘防鱼尾单调收缩、前缘钝度半径、后缘闭合楔角、截面面积、抗凹陷凸性等全套工业级强物理几何约束。参数化采用 **Kulfan (CST) 18 维表示**，优化初始基准统一为 **NACA 0012**，评估预算严格统一为 **300 次函数评估（NFE = 300）**。

---

## 1. 核心对比实验数据 (10 个独立随机种子)

在统一测试工况（$Re=3.5 \times 10^6, \alpha=3.5^\circ$, NFE=300）下，Vanilla 基准与 RL-Meta 的 10 随机种子独立实验统计结果如下：

| 优化算法 | 算法类型 | Vanilla 基准升阻比 ($L/D$) | RL-Meta 强化升阻比 ($L/D$) | 绝对提升 ($\Delta$) | 相对增益 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **SPSA** | 随机逼近一阶模拟 | $90.08 \pm 20.60$ | **$133.02 \pm 15.98$** | **$+42.94$** | **$+47.7\%$** |
| **Py-BOBYQA** | 二次拟合局部信赖域 | $161.07 \pm 0.00$ | **$174.97 \pm 0.00$** | **$+13.90$** | **$+8.6\%$** |
| **GA** | 进化算法 (实数编码) | $110.14 \pm 8.11$ | **$125.19 \pm 7.54$** | **$+15.05$** | **$+13.7\%$** |
| **PSO** | 群智能优化 | $182.96 \pm 18.14$ | **$184.67 \pm 15.27$** | **$+1.72$** | **$+0.9\%$** |
| **CMA-ES** | 协方差自适应演变 | $193.13 \pm 15.44$ | $179.93 \pm 6.51$ | $-13.20$ | $-6.8\%$ |
| **TuRBO** | 高维贝叶斯信赖域 | $218.79 \pm 10.61$ | **$230.95 \pm 13.26$** | **$+12.15$** | **$+5.6\%$** |

---

## 2. 状态与动作空间设计

各优化器共享通用状态特征（评估进度 $t/300$、归一化雷诺数、攻角、历史相对收益、近期提升率与停滞比例，共 6 维），并结合各优化器专有算法内部动力学特征：

- **SPSA (10维 State / 2维 Action)**: 状态增加学习率对数、扰动步长对数、梯度范数、前后梯度余弦夹角；动作调节梯度步长乘子 $\mu_a$ 与扰动步长乘子 $\mu_c$。
- **Py-BOBYQA (10维 State / 1维 Action)**: 状态增加信赖域半径 $\Delta$、收敛半径 $\rho$、二次模型梯度范数；动作调节信赖域自适应伸缩乘子 $\kappa_\Delta$。
- **GA (11维 State / 3维 Action)**: 状态增加种群适应度方差、种群中心多样性；动作调节交叉率 $p_c$、变异率 $p_m$ 与高斯变异步长 $\sigma_m$。
- **PSO (11维 State / 3维 Action)**: 状态增加粒子群分散度、平均速度范数；动作动态调节惯性权重 $w$、个体认知权重 $c_1$ 与社会协同权重 $c_2$。
- **CMA-ES (9维 State / 1维 Action)**: 状态增加全局步长 $\sigma$、进化路径长度比值、协方差矩阵条件数；动作动态调节全局步长乘子 $\kappa_\sigma$。
- **TuRBO (10维 State / 2维 Action)**: 状态增加局部信赖域尺度 $L$、连续成功/失败计数、GP 后验不确定度；动作调节信赖域尺度 $\kappa_L$ 与汤普森采样探索温度 $\tau$。

---

## 3. 代码仓库结构

```bash
RL-Airfoil/
├── src/
│   ├── evaluator.py          # 工业级气动与几何约束评估器 (NeuralFoil + AeroSandbox)
│   ├── airfoil_utils.py      # NACA0012 初始参数、18维 Kulfan 边界与坐标生成
│   ├── ppo.py                # PyTorch PPO Actor-Critic 双头神经网络与 GAE 模块
│   └── optimizers/           # 6 大黑盒优化器实现 (支持 Vanilla 与 RL-Meta 模式)
│       ├── base.py           # 优化器基类 (300 NFE 计数、特征提取与状态追踪)
│       ├── spsa.py           # SPSA 优化器
│       ├── bobyqa.py         # Py-BOBYQA 优化器
│       ├── ga.py             # 遗传算法 (Real-Coded GA)
│       ├── pso.py            # 粒子群算法 (PSO)
│       ├── cma_es.py         # CMA-ES 优化器
│       └── turbo.py          # TuRBO 信赖域贝叶斯优化
├── train.py                  # PPO 元优化器训练脚本 (110 气动网格工况)
├── benchmark.py              # 10 随机种子独立对比实验脚本
├── plot_results.py           # 独立出版级绘图脚本 (读取数据文件生成图表)
├── weights/                  # 训练保存的 6 个 Meta-PPO 策略权重 (.pt)
├── results/                  # 10 随机种子完整评估轨迹与最终几何数据 (JSON & NPZ)
├── figures/                  # 5 组高清科研对比图表 (300 DPI PNG 与 PDF 矢量图)
│   ├── comprehensive_overview.png / .pdf
│   ├── convergence_curves.png / .pdf
│   ├── bar_comparison.png / .pdf
│   ├── boxplot_comparison.png / .pdf
│   └── airfoil_shapes.png / .pdf
└── README.md
```

---

## 4. 运行说明

### 环境安装
```bash
uv pip install aerosandbox neuralfoil py-bobyqa cma botorch gpytorch torch matplotlib pandas scipy
```

### 1. 训练 PPO 元优化器
```bash
python train.py --episodes 100 --device cuda
```

### 2. 运行 10 种子对比实验
```bash
python benchmark.py --reynolds 3.5e6 --alpha 3.5 --max_evals 300 --device cuda
```

### 3. 生成科研对比图表
```bash
python plot_results.py --data_file results/benchmark_results.json --output_dir figures
```
