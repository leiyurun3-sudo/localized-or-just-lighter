# ICME 2027 第 30 项 · 模板核验（2026-11-19）

**获取模板无需登录 ICME 官网。** ICME 初投稿的 Word、LaTeX 两个按钮与 SPS 作者资源页的模板按钮均指向 `bit.ly/SPSpapertemplate`。终点是 IEEE Signal Processing Society 在 Overleaf 发布的 v1.1。匿名访问可读取完整示例主文件和发布者 PDF；实测 “Open as Template” 返回 HTTP 302 到 Overleaf 的 `/login`。若要取到发布者 ZIP 并逐字节核对其中的封装类，当前核实的官方路径是在 Overleaf 中打开该模板，选择 Download Source。[ICME 初投作者页](https://2027.ieeeicme.org/authors/initial-author-instructions) · [SPS 作者资源页](https://signalprocessingsociety.org/index.php/events/conference-workshop-author-resources) · [SPS 发布页](https://www.overleaf.com/latex/templates/ieee-signal-processing-society-conference-and-workshop-template/xdqvgfmfghwj) · [Overleaf 官方导出方法](https://www.overleaf.com/blog/627-tip-of-the-week-how-to-download-an-overleaf-template)

## 已验证的范围

| 对象 | 本轮证据 |
|---|---|
| 本地 SPS 封装类 `paper/ieeeconf.cls` | 实际从 v21 的 paper 目录取样，版本为 2026/11/02 v1.1，SHA-256 为 `77950d4bb10945d218da497a94db5c880545ee5f0d9358f1cf0fbeac2d74ddc8`。公开发布页未提供该文件的独立下载或校验值，官方 ZIP 成员的字节身份仍待核实。 |
| 底层类与参考文献样式 | 重新下载 CTAN 的 IEEEtran 包，本地 `IEEEtran.cls`（v1.8b）和 `IEEEtran.bst` 均与包内对应文件逐字节相同。[CTAN 原包](https://mirrors.mit.edu/CTAN/macros/latex/contrib/IEEEtran.zip) |
| 其他 2027 会议 | 实际解包 ICASSP 的公开 ZIP，内含 `spconf.sty` 和 `IEEEbib.bst`，成员清单中没有 SPS 的 `ieeeconf.cls`。ISBI 2027 作者页链接至 2025 站点下的 2024/07 ZIP，本轮未取得其成员字节。[ICASSP Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php) · [ISBI 作者页](https://biomedicalimaging.org/2027/papers/) |

## 官方示例复编

重新下载发布页的 `ieeeconf.tex` 与三页 PDF，以本地 SPS 类和 **pdfTeX 1.40.27** 原样编译活动示例（9 pt、具名）。参考 PDF 的生成器也是 pdfTeX 1.40.27。缺少的两项依赖都取自发布者的公开内容：从官方 PDF 第 2 页单独提取 `fig1.pdf` 的嵌入矢量对象和原始边界；从官方 TeX 注释中取出四条排版好的参考文献作为 `.bbl`。未拿到原始独立 `fig1.pdf` 容器或 `refs.bib`。页面正文、公式、表格、题头和分栏全部由 TeX 重新排版。[官方参考 PDF](https://www.overleaf.com/latex/templates/ieee-signal-processing-society-conference-and-workshop-template/xdqvgfmfghwj.pdf)

| 页 | 提取词数 | 词框四坐标最大偏差（PDF pt） | 300 dpi RGB 相异像素 / 总像素 |
|---|---:|---:|---:|
| 1 | 883 | 0 | 0 / 8,415,000 |
| 2 | 810 | 0 | 0 / 8,415,000 |
| 3 | 270 | 0 | 0 / 8,415,000 |

双方页面均为 612 × 792 pt。逐页词序、逐行文本、文字起点及字体名称一致。原 PDF 与复编 PDF 的文件哈希不同；逐页渲染哈希相同。复编验证了发布者示例用到的版式，不能从输出反推类文件的每个字节。

另将同一官方例子的类选项从 `[9pt]` 单独改为 `[doubleanonymous]`，实编默认 10 pt 双匿名样例。首页为 “Anonymous submission”和“Author names and affiliations withheld for review”，出现连续边栏行号。PDF 作者元数据为 “Author names withheld for review”。ICME 作者页指定的第一行为 “Anonymous ICME Submission”。SPS 发布页只提供具名参考 PDF，双匿名结果是独立运行记录。

**离线复核。** 在安装了相关宏包的 TeX Live 2025 环境进入 `demo/`，对 `ieeeconf.tex` 连续运行三次 `pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape ieeeconf.tex`，保留已附的 `ieeeconf.bbl`。回到本目录执行 `python compare.py`（PyMuPDF、NumPy），可重算逐页词框和像素差。`sources.json` 保存来源、原件摘要及 ZIP 成员，`layout-comparison.json` 保存每页渲染摘要，`build-record.txt` 摘录最后一次编译所载入的类。
