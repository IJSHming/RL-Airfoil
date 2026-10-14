# RL-Airfoil 项目开发与科研实验规范 (Workspace Rules)

本规范专为后续 LLM-Meta（TuRBO 专用优化器）与 MALIBO（Meta-learning for Likelihood-free Bayesian Optimization）阶段定制，作为代码编写、任务调度与算力分配的强制性行为准则。

## 1. 物理约束与局部探索真实性 (Physical Constraints & Realistic Exploration)
1. **气动与几何强约束前置校验**：
   - 候选翼型生成必须严格遵从 `AerodynamicEvaluator` 的工业级几何与物理约束（最大厚度、结构梁站位厚度、后缘防鱼尾单调递减、前缘钝度半径、抗凹陷凸性等）。
   - 在 TuRBO 候选采样或 MALIBO 无似然代理采样生成时，必须采用拒绝采样或边界投影，杜绝表面自交叉等物理塌陷解进入气动求解器或破坏代理模型连续性。
2. **基准局部探索真实性**：
   - 优化空间探索必须围绕基准构型（NACA 0012 等）保持合理的局部物理尺度，严禁使用脱离工程实际的盲目全局离散采样而导致评价预算被边界惩罚浪费。

## 2. 进程间通信与推理无锁化 (Lock-Free Inter-Process Safety)
1. **严格规避多进程 CUDA/OpenMP 死锁**：
   - 在使用 `ProcessPoolExecutor` 或 `multiprocessing` 开展多并发评估时，子进程严禁直接继承父进程已初始化的 CUDA 上下文。
   - 子进程中的轻量前向推理或代理评估优先采用纯 NumPy 解耦（`numpy_policy_forward`）或统一采用 `spawn` 启动模式。
2. **架构解耦**：
   - 数据采集（CPU 并发）与模型前向/梯度更新（GPU 加速）严格解耦，杜绝子进程内部的锁竞争。

## 3. 长程任务自省与进程监控 (Task Self-Inspection - User: c250089)
1. **主动监测与自省机制**：
   - 在运行超过 30 秒的长程训练、批次测试或 TuRBO/MALIBO 实验时，必须周期性（约每隔 1 分钟或批次交替时）通过命令行主动检查进程健康状态。
2. **固定用户与监控指令**：
   - 监控对象严格锁定当前用户 `c250089`，使用以下指令查看资源占用：
     ```bash
     top -b -n 1 -u c250089 | head -n 25
     ```
   - 若出现多个进程 CPU 占用与 GPU 占用均为 0% 的假死状态，必须立即自省中断并诊断修复。

## 4. 10 核 CPU 与 L40S GPU 强大算力最大化利用准则 (Hardware Acceleration Strategy)
1. **硬件特性认知**：
   - 本机配备 **10 核 CPU** 与 **NVIDIA L40S GPU (48GB 显存、Ada 架构强大张量核心)**，具有充沛的多核与并行加速潜力。
2. **算力精准分工**：
   - **10 核 CPU 满载**：负责 NeuralFoil/AeroSandbox 气动评估的多任务并行求解（通过 10 工作进程并发）。
   - **L40S GPU 极致加速**：
     - TuRBO 的高斯过程（GP）拟合、边际似然优化（MLL）、高维协方差 Cholesky 分解以及 Thompson 候选采样批量计算优先放于 GPU 显存执行；
     - 后续 LLM-Meta 的大模型推理、嵌入计算与 MALIBO 的无似然神经网络代理模型训练/推理统一在 L40S 上高效调度。
3. **并发 GPU 调度安全**：
   - 涉及多任务共享 GPU 时，优先采用主进程集中批处理，或多进程安全分配 GPU 算力，确保 10 核 CPU 与 L40S 均处于饱满利用状态且绝对不发生死锁。
