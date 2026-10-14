# RL-Meta-Airfoil: 强化学习元优化黑盒翼型气动外形设计

本项目基于深度强化学习（PPO），针对 6 种经典与前沿黑盒优化算法（SPSA、Py-BOBYQA、DE、PSO、CMA-ES、TuRBO）构建了**动态超参数元优化控制系统（Learning to Optimize, L2O）**。

优化目标为在典型亚音速/跨音速巡航工况（$Re=3.5 \times 10^6, \alpha=3.5^\circ$）下**最大化升阻比（$C_L / C_D$）**，同时受到最大相对厚度、结构梁最小厚度、后缘防鱼尾单调收缩、前缘钝度半径、后缘闭合楔角、截面面积、抗凹陷凸性等全套工业级强物理几何约束。参数化采用 **Kulfan (CST) 18 维连续参数表示**，优化初始基准统一为 **NACA 0012**，评估预算严格统一为 **300 次函数评估（NFE = 300）**。

针对基准测试设定的严谨性与学术说服力，本项目**全面贯彻原作者对于多维异质空间“尺度敏感标准经验准则”与开源权威包官方默认参数**，所有 Vanilla 基准均采用无人工干预的通用配置，并在此强基准上证明 Meta-RL 的自适应超额赋能能力。

---

## 1. 核心对比实验数据 (10 个独立随机种子)

在统一基准工况（$Re = 3.5 \times 10^6, \alpha = 3.5^\circ$、严格 300 次 NFE）下，初始基准 NACA 0012 升阻比为 **$65.80$**。各优化器在消耗满严格 300 次评估预算后的 10 个独立随机种子（`Seed: 42, 123, 256, 777, 1024, 2024, 3407, 5555, 8888, 9999`）基准对比实验数据如下：

| 优化算法 | 算法类型分类与权威基准机制 | Vanilla 坚实基准 ($L/D$) | RL-Meta 强化升阻比 ($L/D$) | 绝对提升 ($\Delta$) | 相对增益 (%) | 10 种子可行率 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SPSA** | 随机逼近一阶梯度模拟 (Spall 跨度自适应比例) | $186.93 \pm 21.08$ | **$212.64 \pm 13.36$** | **$+25.71$** | **$+13.8\%$** | **100% (10/10)** |
| **Py-BOBYQA** | 二次逼近无导数信赖域 (Powell 官方默认信赖域) | $136.19 \pm 30.95$ | **$180.11 \pm 25.69$** | **$+43.92$** | **$+32.2\%$** | **100% (10/10)** |
| **DE** | 差分进化算法 (边界自适应散布 + Storn-Price) | $181.90 \pm 13.87$ | **$194.80 \pm 15.48$** | **$+12.90$** | **$+7.1\%$** | **100% (10/10)** |
| **PSO** | 粒子群优化 (Clerc-Kennedy 临界压缩因子) | $195.24 \pm 13.52$ | **$200.19 \pm 19.45$** | **$+4.94$** | **$+2.5\%$** | **100% (10/10)** |
| **CMA-ES** | 协方差自适应演化 (Hansen 官方默认步长) | $189.19 \pm 11.69$ | **$203.04 \pm 17.61$** | **$+13.84$** | **$+7.3\%$** | **100% (10/10)** |
| **TuRBO** | 高维贝叶斯信赖域 (Eriksson 默认信赖域) | $195.73 \pm 12.32$ | **$204.41 \pm 20.90$** | **$+8.68$** | **$+4.4\%$** | **100% (10/10)** |

### 核心科研与实验结论
1. **全员强基准，拒绝“稻草人”**：各优化器的 Vanilla 表现均处于 **136 ~ 195** 的高水平气动优化区间（SPSA 达 186.93，DE 达 181.90，PSO 达 195.24），收敛曲线呈现出活跃且自然的上升历程，彻底杜绝了“基准过于弱小像水平线”的学术瑕疵；
2. **严格遵从奠基文献推荐准则**：
   - **PSO**：严格采用 Clerc & Kennedy (2002) 奠基收敛因子解（$\chi = 0.72984, c_1 = c_2 = 1.49618$），速度限幅 $v_{\max} = 0.10 \times \text{span}$，种群规模 $S = 10 + \lfloor 2\sqrt{D} \rfloor = 15$；
   - **SPSA**：严格遵循 Spall (1998) 尺度敏感准则，扰动与各维度几何跨度成比例绑定（$c_k = 0.03 \cdot \Delta \theta$），消除后缘与型线间的尺度失配；
   - **Py-BOBYQA**：严格遵循 Powell (2009) 官方库初始默认信赖域半径（$\rho_{\text{beg}} = 0.10$）；
   - **DE**：严格遵循 Storn & Price (1997) 经典准则，初始种群按各维度实际几何跨度自适应散布（$0.20 \times \text{span}$），拒绝采样保证 100% 物理有效；
   - **CMA-ES**：遵循 Hansen (2006) 官方 `cma` 默认步长（$\sigma_0 = 0.05$）并增设前中期步长下限防护；
   - **TuRBO**：遵循 Eriksson et al. (2019) 官方信赖域尺度设定（$L_0 = 0.8$）。
3. **RL-Meta 斩获全场最高升阻比**：在极其强大的 Vanilla 基准之上，RL-Meta 全部实现稳健的正向超额增益。其中 **SPSA 均值暴涨至 212.64 (+25.71, +13.8%)**，超越 TuRBO 与 CMA-ES，一跃成为全场表现最顶尖的优化器；
4. **顶尖梯队全员破 200**：PSO（200.19）、CMA-ES（203.04）、TuRBO（204.41）、SPSA（212.64）四款优化器在 RL-Meta 驱动下均成功跨越 200 升阻比大关；
5. **100% 航空工程物理合规**：全场 120 组测试的最优设计可行率保持 **100%**，无任何物理自交或后缘畸变。

---

## 2. 状态与动作空间设计

各优化器共享通用优化动态基础特征（评估进度 $t/300$、归一化雷诺数、攻角、历史相对收益、近期提升率与停滞比例，共 6 维），并结合各优化器内部专有动力学特征：

- **SPSA (10维 State / 2维 Action)**: 状态增加步长相对归一化值、扰动相对归一化值、梯度范数、连续两步梯度方向余弦夹角；动作动态微调学习率乘子 $\mu_a$ 与扰动步长乘子 $\mu_c$。
- **Py-BOBYQA (10维 State / 1维 Action)**: 状态增加信赖域半径 $\Delta$、收敛半径 $\rho$、二次模型梯度范数；动作动态调节信赖域自适应伸缩乘子 $\kappa_\Delta$。
- **DE (11维 State / 3维 Action)**: 状态增加交叉率 $CR$、变异比例 $F$、停滞比例、种群适应度方差、种群中心多样性；动作动态调节交叉率 $CR$ 与差分步长 $F$。
- **PSO (11维 State / 3维 Action)**: 状态增加粒子群分散度、平均速度范数；动作在 Clerc-Kennedy 压缩因子临界特征根守卫下，动态调节惯性动量 $w$ 与社会/认知权重比例。
- **CMA-ES (9维 State / 1维 Action)**: 状态增加全局步长 $\sigma$、进化路径长度比值、协方差矩阵条件数；动作动态调节全局步长乘子 $\kappa_\sigma$，实施前中期步长下限防护。
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
│       ├── spsa.py           # SPSA 优化器 (Spall跨度比例扰动 + 停滞回溯重构)
│       ├── bobyqa.py         # Py-BOBYQA 优化器 (官方默认信赖域 + 动态信赖域钩子)
│       ├── de.py             # 差分进化算法 (边界自适应散布 + current-to-best/1/bin)
│       ├── pso.py            # 粒子群算法 (Clerc-Kennedy 压缩因子特征值守卫)
│       ├── cma_es.py         # CMA-ES 优化器 (官方 cma 库 + 步长下限保护)
│       └── turbo.py          # TuRBO 信赖域贝叶斯优化 (GP + Cholesky 汤普森采样)
├── train.py                  # PPO 元优化器高效训练脚本 (10核 CPU 并发采样 + GPU 集中梯度更新)
├── benchmark.py              # 10 随机种子独立对比实验脚本 (记录 300 步完整历史与最终形状)
├── plot_results.py           # 出版级绘图脚本 (输出 6 张单优化器柱状图、收敛曲线、箱线图与全景大图)
├── weights/                  # 训练保存的 6 个 Meta-PPO 策略权重 (.pt)
├── results/                  # 10 随机种子完整评估轨迹与最终几何数据 (JSON & NPZ)
├── figures/                  # 高清科研对比图表 (300 DPI PNG 与 PDF 矢量图)
│   ├── bar_comparison_spsa.png / .pdf       # SPSA 10 种子具体对比柱状图
│   ├── bar_comparison_py_bobyqa.png / .pdf  # Py-BOBYQA 10 种子具体对比柱状图
│   ├── bar_comparison_de.png / .pdf         # DE 10 种子具体对比柱状图
│   ├── bar_comparison_pso.png / .pdf        # PSO 10 种子具体对比柱状图
│   ├── bar_comparison_cma_es.png / .pdf     # CMA-ES 10 种子具体对比柱状图
│   ├── bar_comparison_turbo.png / .pdf      # TuRBO 10 种子具体对比柱状图
│   ├── bar_comparison.png / .pdf            # 2x3 六合一柱状图全景对比
│   ├── convergence_curves.png / .pdf        # 10 种子平均收敛曲线与置信带
│   ├── boxplot_comparison.png / .pdf        # 10 种子最终升阻比分布箱线图与真实数据散点
│   ├── airfoil_shapes.png / .pdf            # 10 种子取中位数（第5名）的最优翼型几何形状对照
│   └── comprehensive_overview.png / .pdf    # 2x2 全景大图汇总
└── README.md
```

---

## 4. 运行复现说明

### 1. 环境安装
```bash
uv pip install aerosandbox neuralfoil py-bobyqa cma botorch gpytorch torch matplotlib pandas scipy
```

### 2. 训练 PPO 元优化器 (覆盖 110 种网格工况，10 核并发)
```bash
python train.py --num_workers 10 --device cuda
```

### 3. 运行 10 随机种子独立基准测试
```bash
python benchmark.py --num_workers 10
```

### 4. 生成全套出版级高清图表
```bash
python plot_results.py
```
