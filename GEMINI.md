# RL-Airfoil 项目开发与科研实验规范 (Workspace Rules)

本规范适用于 RL-Meta（6 种经典黑盒优化器）、LLM-Meta-TuRBO 与严格原论文版 MALIBO（Pan et al., ICML 2024）的统一科研公平性、学术诚信与算力调度约束。

## 1. 科研诚信与公平对比铁律 (Research Integrity & Fair Benchmarking)
1. **严格零隐藏启发式（RL-Meta 公平性）**：
   - 所有 RL-Meta 优化器的 `apply_action` 必须满足零动作恒等性（当 PPO 输出 `action = 0` 时，算法与超参精确退化为对应的 Vanilla 基线）。
   - 严禁仅在 `is_meta=True` 分支内暗置未赋予 Vanilla 的启发式特权（如停滞精英回溯重启、硬编码退火调度、停滞温度硬跳变或额外步长地板）；所有奖励函数须采用跨工况无标度相对增益，严禁硬编码针对测试工况尺度的绝对阈值。
2. **严格零提示词泄漏（LLM-Meta 诚实性）**：
   - 严禁在 LLM 系统/用户提示词中硬编码特定翼型的最优偏移数值区间、指令性偏移命令或目标厚度区间；所有几何偏移通道须采用对称边界，且触发机制采用 `post_init`（Step 16）+ 真实停滞自适应触发。
   - 严禁在 LLM 服务离线或解析异常时静默回退人工启发式字典。
3. **严格原论文对齐（MALIBO 基准忠实度）**：
   - MALIBO（`Meta-CFD/malibo`）必须严格忠实于 Pan et al. (ICML 2024, `MALIBO.pdf`) 原文：4×64 ResFFN（ELU, $d=50$）、精确 Eq. (2) LFBO 损失、排序 KS+Cov 先验正则、固定 $\gamma=1/3$ Laplace 汤普森采样、70/30 早停残差 GBDT 及全局 5,120 点均匀随机采集搜索（严禁私自添加信赖域、稀疏坐标掩码或免费几何预筛器）。
4. **严格排除测试工况**：
   - RL 训练网格（110 工况）与 MALIBO 元训练集（250 任务）均必须严格排除测试工况（$Re = 3.5 \times 10^6, \alpha = 3.5^\circ$）。

## 2. 进程间通信与推理无锁化 (Lock-Free Inter-Process Safety)
- 在使用 `ProcessPoolExecutor` 开展 CPU 并行评估时，统一使用 `multiprocessing.get_context("spawn")` 与纯 NumPy 前向推理（`numpy_policy_forward`），彻底杜绝 CUDA/OpenMP fork 死锁。

## 3. 硬件算力分工 (10-Core CPU + NVIDIA L40S GPU)
- **10 核 CPU**：承担 `NeuralFoil ('xxxlarge')` 气动评估并发采集与 10 种子测试。
- **L40S GPU**：承担 PPO 集中式策略梯度更新、MALIBO 元网络训练/推理以及本地 SGLang (`Qwen/Qwen3.8-27B-FP8`) 大模型推理。
