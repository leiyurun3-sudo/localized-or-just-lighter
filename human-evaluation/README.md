# 离线编辑成功盲评 / Offline edit-success assessment

本仓库的三份空白离线问卷可从 [v21 下载区](https://github.com/leiyurun3-sudo/localized-or-just-lighter/releases/tag/v21) 取得。以下操作说明针对协调者的本地完整套件。仓库提供评分程序，计分所需的私有编号对照与随机化记录由协调者单独保管。

本目录是协调者保存的用户套件。三份 ZIP 各含 50 个源图各自的 3 个默认输出，共 150 项。三份问卷使用相同图像、独立随机顺序和独立匿名编号。保留本目录在投稿文件之外。当前状态为待收集，尚无人工评分。

## 分发与作答 / Administration

1. 为每位独立评分者分配一份不同的 ZIP。两人时使用任意两份，三人时各用一份。协调者单独保管 `organizer/` 中的配置密钥、随机化种子和提示词对照表。
2. 评分者完整解压自己的 ZIP，在桌面浏览器中打开解压后的 `index.html`。可断网使用。每题展示完整的 512 × 512 原图和输出，以及同一份中英评分目标。点击“原始指令与模型输入”可查看未经改写的英文描述。
3. 核对变化是否出现在指定对象、数量、位置、方向或风格上。可辨认的目标变化记 1，否则记 0。目标以实际源 / 目标提示对为准。无关的画质或背景变化不参与此二元判定。可另勾选判断有疑虑；暂时跳过的项目留空。各人独立作答。
4. 关闭网页前导出 JSON。可在同一份问卷恢复部分备份。浏览器本地存储可能会随浏览器设置或移动文件而清除，导出的 JSON 可跨设备恢复。协调者收回每人的最终导出并分别保存。

Give each independent rater a different ZIP. Extract all files and open `index.html` offline. Rate whether the requested target change is visible on its specified subject, independent of unrelated image quality. A skipped item is missing. An uncertainty flag accompanies a binary vote. Raters should work independently and return their exported JSON, with one final export per person.

## 固定解释 / Frozen prompt policy

网页保存了逐源固定中英解释。说明文字与生成时实际使用的描述发生冲突时，以提示对为准。两处直接冲突分别涉及水獭周围的心形变色（原指令误写乌鸦）和女孩站在房屋前方（原指令将前后关系写反）。范围、风格和拼写注释另列。协调者可用 `organizer/prompt_decisions.csv` 查看全部 50 个源图的原始措辞与评分目标。

The supplied prompt pair resolves contradictory instruction text. The same fixed clarification appears for every configuration and rater. Two direct contradictions concern the otter's heart decorations and the girl/house spatial relation. The organizer ledger retains the original English wording and labels scope, spelling and style clarifications separately. The protocol was fixed before collecting ratings.

## 汇总真实作答 / Score returned responses

在本目录运行，Python 3 仅用标准库。将示例中的输入文件名替换为实际回收的 JSON。三人时在命令中加入第三份导出。

```text
python organizer/score.py R01-ratings.json R02-ratings.json --out summary.json
```

配置 `uniform`、`two_constant`、`sam_native` 对应 Uniform、Two constants、Spatial。程序按问卷和匿名项目编号连接密钥，各人读到的同一源图和配置会被匹配起来。重复提交同一评分者或错误问卷会报错。

- 每配置每人报告成功数、已评数、缺失数、有疑虑数及成功率（成功数 / 已评数）。每配置应评 50 项。
- 每配置另报所有已提交评分者共同完成的项目数量，以及这些项目上的赞成票比例。两人分歧时，两张有效票均计入。缺失保持缺失，有疑虑标记保留原票。
- 两两 Cohen’s κ 使用该对评分者共同完成的项目，同时输出重叠数、2 × 2 计数和直接一致率。三人时另在三人共同完成的项目上计算 Fleiss’ κ。既分配置也汇总全部 150 项。重叠为空或 κ 的分母为零时返回 `null`。

The scorer reports each rater's successes over observed items and a vote-based success rate on complete overlap across submitted raters. Pairwise Cohen's kappa uses each pair's overlap. With three returns, Fleiss' kappa uses items rated by all three. Counts and observed agreement accompany kappa. Undefined statistics are JSON null. Run `python organizer/score.py --check-key` to check the assignment design without any responses.
