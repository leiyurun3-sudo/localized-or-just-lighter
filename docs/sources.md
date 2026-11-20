# 代码与数据来源

- **基准与图像。** 本文采用 PIE-Bench 源图、编辑提示及区域掩码。原始图像的权利归相应权利人，标注条款与出处保留在 [数据说明](../experiments/R91_ICME_Minimal/DATA_LICENSE.txt) 中。
- **推理实现。** SAM-Flow 源码、固定提交号和实验适配内容见 [UPSTREAM.json](../experiments/A_DoseLens_Pilot_W1/UPSTREAM.json)。原作者 MIT 声明见 [LICENSE.upstream](../experiments/A_DoseLens_Pilot_W1/LICENSE.upstream)。模型权重按相应发布方条款另行获取。
- **论文与归因。** 主文引用列在 [references.tex](../paper/references.tex)。文献审计对 24 篇候选工作逐篇记录比较对象、原文位置与判定。其中 15 篇进入区域组件比较的分母。[审计数据](../reproduction/literature/control_audit.csv)
- **逐图分数。** [证据索引](../reproduction/EVIDENCE_INDEX.md) 对应到原始记录、哈希、指标计算版本与可取得的输出。不同年代的恢复型运行具有不同的原生图像覆盖，补充材料首页列出分母。
- **人工评测。** [离线评测协议](../human-evaluation/README.md) 预先规定样本、作答和汇总规则。v21 尚未收集独立评分。
