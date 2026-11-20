# 实验索引

以下目录保留历史运行包里的生成器、参数和输入。复算论文统计量请直接从 [CPU 复现说明](../docs/reproduction.md) 开始。重新生成图像还需对应的模型权重、推理依赖及包内约定的数据布局。

| 运行包 | 对应的论文问题 | 原包入口 |
|---|---|---|
| R88 | 在 40 个源图的恢复型编辑上建立原生配置与两个交叉回放单元，比较空间分配和逐步幅度 | [协议](R88_AnalysisClosure_Fast/protocol.json) · [运行说明](R88_AnalysisClosure_Fast/README.md) |
| R91 | 检查恢复型结果的种子差异、空间图结构及保护下界 | [协议](R91_ICME_Minimal/protocol.json) · [运行说明](R91_ICME_Minimal/README.md) |
| R92 | 用每个源图和种子的单个标量校准原生积分，比较标量近似与逐步回放；40 个源图 × 2 个种子新增 80 个输出 | [协议](A_R92_ScalarCalibration/protocol.json) · [运行说明](A_R92_ScalarCalibration/README.txt) |
| SAM / FlowEdit，首批 300 | 在 50 个源图上运行 6 个核心设置，包括默认配置、常数对照和回放 | [协议](A_DoseLens_Pilot_W1/protocol.json) · [运行说明](A_DoseLens_Pilot_W1/RUN.txt) |
| SAM / FlowEdit，追加 600 | 在同一队列扩展强度与窗口扫描，沿用首批 300 的目录和既有输出。共形成每源 18 个输出、合计 900 个输出的图册 | [增量启动脚本](A_DoseLens_Scan600/run_scan.sh) |

追加 600 的脚本假设 AutoDL 上已有首批核心结果。实验包保留当时的 Linux 路径及模型加载约定。论文中的“五档扫描 750 个运行点”包括复用的默认点；900 是整套 SAM 图册的输出数。两者统计对象不同。
