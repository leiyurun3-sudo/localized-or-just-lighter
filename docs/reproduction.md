# 复现与编译

## 用 CPU 复算

在仓库根目录执行。点估计只需 Python 标准库，置信区间需要 NumPy，映射敏感性脚本还使用 SciPy。项目此前在 Python 3.12 上验证了整套审阅包命令。

```bash
cd reproduction
python reproduce.py
python -m pip install numpy scipy
python reproduce.py --ci
python analysis/v21_bounded/bounded_mrc.py --output rerun/bounded
python analysis/sam_seed_repeat/compare_seeds.py analysis/sam_seed_repeat/seed1.csv --out seed_comparison.json
```

根脚本打印 JSON；四档结果在 SAM 部分的 `scales_at_or_below_default`。最后一条比较 seed 0 与 seed 1 的四档结果。seed 2 的 48 条记录见 [seed2_partial.csv](../reproduction/analysis/sam_seed_repeat/seed2_partial.csv)，覆盖 12 个源图的四个核心配置。逐图常数拟合、留一档插值、固定状态下的掩码检验与跨队列检查各有输入表和运行入口，见[证据索引](../reproduction/EVIDENCE_INDEX.md)。区域 LPIPS 的两种口径在补充材料 §1.1 定义。

## 编译论文

从仓库根目录进入 `paper/`。保留同级 `analysis/`，部分 TeX 文件使用相对路径引用其中的数值。使用 Tectonic 的默认 PDF 输出，第一次编译需要能访问 TeX 资源包或已有缓存。

```bash
cd paper
tectonic main.tex
tectonic supplement.tex
```

生成的 `main.pdf` 和 `supplement.pdf` 在当前目录。用于上传的独立 PDF 与匿名 ZIP 已另行处理图片体积，归档在 [manuscript](../manuscript/)。需要直接打包好的 TeX 工程、恢复型映射的可选输入或逐图原生输出时，可从 [v22 完整包](https://github.com/leiyurun3-sudo/localized-or-just-lighter/archive/refs/tags/v22-complete.zip) 下载。压缩包的 `downloads/` 内含900 图扫描、seed 1 的完整输出和 seed 2 的部分输出。前两组按原始目录展开。

## 提交前材料

[匿名补充 ZIP](../manuscript/A_ICME_Review_Supplement.zip) 是供审稿的独立包，含 54 页补充材料、10 页图册和复现文件。作者侧另有 [CMT 填报稿](CMT-draft.txt)、[模板核验记录](template-check.md) 和 [38 项修改记录](revision-checklist.md)。
