# Localized, or Just Lighter?

**空间掩码带来的背景保持收益，有多少来自把编辑放对位置，又有多少来自整体改得更轻？** 这篇分析论文在 FLUX.1-dev 上研究两种免训练图像编辑算子，用逐步幅度回放分析组件，用等响应比较评估完整配置。

[阅读主文（6 页）](manuscript/A_ICME_Main.pdf) · [补充材料（53 页）](manuscript/A_ICME_Supplement.pdf) · [900 张输出图册](reproduction/Atlas_900_outputs.pdf) · [下载审阅包](manuscript/A_ICME_Review_Supplement.zip)

## 两个编辑器揭示了什么

| 比较 | 观测结果 | 如何解读 |
|---|---|---|
| 恢复型校正，40 个源图 | 默认系数下，空间减均匀的背景 LPIPS 差为 **+0.0362**。在两个参考幅度序列下分别回放，均为 **−0.0100**（四位小数）。 | 默认配置下均匀校正更强。控制实际幅度后，空间分配的背景保持收益才显现。 |
| SAM-Flow，两分支校准 | 对每张图分别拟合两个空间常数，在有背景像素的 42 张图上保留 **84.5%** 的默认背景收益。在全部 50 张图上，平均 directional CLIP 从 **0.0841** 降至 **0.0521**。 | 仅调节幅度就能获得很大一部分背景收益，同时会改变编辑响应。 |
| SAM-Flow，在相同响应下比较 | 默认点的编辑区/背景 MAE 比为 **4.00**（空间）和 **2.11**（均匀）。对默认及以下四档，在响应区间重叠的 40 张图上，平均背景 LPIPS 差为 **+0.0263**，95% CI **[0.0169, 0.0370]**。 | 这里按均匀减空间计差，正值有利于空间分配。匹配同图 directional CLIP 后，保留同组默认差距的 **21.3%**。 |

40 张恢复型源图与 50 张 SAM 源图属于两个队列。SAM 扫描使用 seed 0。加入 1.25 扩展档后，41 张图有共同响应区间，保留比例为 32.4%。正文分别报告两个扫描范围及其分母。

## 从哪里开始

| 想做什么 | 入口 |
|---|---|
| 理解问题、实验设计和数字如何互相印证 | [中文研究导读](docs/research-guide.md) |
| 用已归档分数复算均值、置信区间与敏感性结果 | [CPU 复现说明](docs/reproduction.md) · [完整证据索引](reproduction/EVIDENCE_INDEX.md) |
| 查看四张主图的图稿、绘图说明及数据 | [图表目录](figures/) |
| 阅读、修改、编译两篇 PDF | [主文 TeX](paper/main.tex) · [补充 TeX](paper/supplement.tex) · [编译说明](docs/reproduction.md#编译论文) |
| 追溯实验实现、设置和生成数量 | [实验索引](experiments/README.md) · [文献逐篇编码](reproduction/literature/control_audit.csv) |
| 查看版本材料和原生输出 | [v21 版本下载](https://github.com/leiyurun3-sudo/localized-or-just-lighter/releases/tag/v21) |
| 准备最后的人工盲评 | [评测说明](human-evaluation/README.md) · [统计口径](human-evaluation/统计口径与回填.md) |

从仓库根目录运行以下两行即可复算主要点估计。Python 标准库即可执行。

```bash
cd reproduction
python reproduce.py
```

**版本状态**　v21 为投稿准备稿。文献编码、轨迹测量和像素指标均有记录。人工指令成功率尚待独立评分者完成，仓库提供的是评测方案与空白问卷。数据、第三方代码与模型的来源见[来源说明](docs/sources.md)。
