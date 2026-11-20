# 四张主图的可编辑图稿

[图 1](archive/v19_cross.svg) · [图 2](archive/v19_protocols.svg) · [图 3](archive/v19_changes.svg) · [图 4](archive/v19_selectivity.svg)。这些 SVG 保留可编辑文字、曲线和独立图层，可直接在 Inkscape 中打开。论文排版使用的 PDF 在 [paper/figures](../paper/figures/)。

## 重画

使用 Python 3.12，从仓库根目录执行。安装依赖后可离线重画，全程使用 CPU。

```bash
python -m pip install -r figures/requirements.txt
python redraw.py
```

单独重画图 3、4 可执行 `python redraw.py 3 4`。结果写入 `figures/generated/`。脚本优先使用排版所用的 Arial，缺少时依次使用 Liberation Sans、Matplotlib 自带的 DejaVu Sans。替代字体的字宽可能略有差异。

## 每张图用到什么

| 图 | 作图入口 | 绘图输入与读法 |
|---|---|---|
| 1 · 两种算子的背景变化 | [fig1.py](../scripts/fig1.py) | [cells.json](../data/figures/cells.json) 保存两组未舍入的四格 LPIPS；键的第一位表示分配方式，第二位表示幅度曲线来源。纵向位置来自实测值，左侧箭头为机制示意。花朵源图、保护图、狗的源图和编辑结果与图 2、3 共用。 |
| 2 · 回放与匹配响应 | [fig2.py](../scripts/fig2.py) | [profiles.json](../data/figures/profiles.json) 保存恢复算子的 28 步 RMS、推离算子第 2–27 步的双分支记录、两条扫描曲线、双常数点及重叠区间的分段端点。[flower_gate.npy](../data/figures/flower_gate.npy) 是原始 64 × 64 保护权重。 |
| 3 · 三个源图的局部对照 | [fig3.py](../scripts/fig3.py) | 从归档 SVG 的具名图层读取 3 张源图、12 张编辑结果及 3 张掩膜。脚本按 [cases.json](../data/figures/cases.json) 的原图坐标重新裁出 96 × 96 局部，按像素重算 RGB 变化图和区域 MAE 比值。LPIPS、方向 CLIP 和罐盖的五档匹配差来自归档测量值。 |
| 4 · 操作点与选择性 | [fig4.py](../scripts/fig4.py) | [operating_points.csv](../data/figures/operating_points.csv) 给出各设置及其置信区间。[data](../data/figures/) 下的 `four_setting_*` 和 `five_setting_*` 分别保存逐源差值与汇总。面板 (c) 的分类点取五档扫描；底部同时绘出四档 40 源和五档 41 源的 U − S 汇总。 |

请保留 `archive/`。脚本将图 2、3 已嵌入的 PNG 图层作为像素输入，重画时无需另外下载原图。统计量的计算入口见 [复现索引](../reproduction/EVIDENCE_INDEX.md)。

## 来源与许可

源图、编辑指令和掩膜来自 Ju 等人的 PIE-Bench。原图版权归各自权利人；标注采用 CC BY-NC-SA 4.0。图中使用了局部裁切、缩放和掩膜轮廓，模型输出与像素变化图由本研究生成。[项目与论文](https://github.com/cure-lab/PnPInversion) · [随数据提供的许可原文](../data/figures/PIE_Bench_dataset_license.txt) · [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

本目录的作图代码和数值表随本仓库提供，后续复用以仓库作者明确授予的许可为准。第三方软件各自适用其许可证。
