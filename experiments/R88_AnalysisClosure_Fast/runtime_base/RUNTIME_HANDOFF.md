# R88 runtime → BPaper

2026-11-04。本次只做一次静态接线核对；未执行 materialize、GPU 或测试套件，未修改父层文件。

## 接线结论

| 臂 | engine 分支 | profile |
|---|---|---|
| REQ | roi_uniform / local_coefficient：0.25 P c | 自身记录 |
| GLB_R | r86_arm=GLOBAL / r86_fixed_profile | REQ |
| U25 | uniform / none：0.25 c；无 route_* / E/P | 自身记录 |
| REQ_U | r86_arm=CORR / r86_fixed_profile | U25 |
| SHUF_R | CORR / wrong_prompt_histmatch | REQ |
| REV_R | CORR / reverse | REQ |

- run.py 的执行顺序满足两个 anchor 先于各自依赖臂；U25 不触发探针、地图保存或地图产物校验。
- SHUF_R 的 probe_target_prompt 仅传给 donor 探针；编辑、token alignment 和正确 E/P 仍使用 target_prompt。probe_donor_case 随 job 保存，实际源图始终为当前 case。
- plan.name=R88_AnalysisClosure、R88_PREFLIGHT_PLAN_SHA256、R88_QUEUE_PROJECT 与 runtime 一致；runtime 不依赖类别数量，40 源/9 类不改变接线。
- anchor NFE 读取 record.nfe_standalone；顶层已移除 job 的固定期望值。本次同时删除模板残留的 expected_nfe_standalone=97，并将顶层复制的 stats_plan.json 加入精简交付清单。

## 缓存必须保持的等价条件

1. **参考轨迹**：同源像素/预处理、分辨率、编码潜变量与 token IDs、seed、完整 sigma 网格、gamma、inv_substeps、strength、模型权重和数值精度。反演固定 empty prompt / guidance=0；缓存状态交给每臂时 clone，避免原位污染。
2. **源探针共享**：仅复用同一 reference state、sigma/node、source_prompt embeddings、guidance、IDs、模型/精度下的 native memory-off source field。正确目标与 donor 可共享这部分；不能跨 source/seed/reference 混用，也不能用任何臂当前 z_i 的 source field 替代。
3. **目标与 E 缓存**：当前外层 key 含 reference group、source_prompt、正确 target_prompt、guidance、response_sigmas；内层 responses[(role, prompt)] 分开 correct/donor。保持相同三节点、每节点 channel RMS → q10/q90 → clip → 三图平均；缓存原始 E，donor 不覆盖正确 E/P。U25 跳过探针时保留现有缓存。
4. **图分配**：SHUF_R 按 P_donor 升序稳定秩放置正确 P 的排序原值；并列按 row-major index。直接使用精确 packed P，避免 P→route→P 的舍入。REV_R 始终为 1.25-P，范围 [0.25,1]，通常不保直方图；两种图都不改 memory gate。
5. **源 KV**：若继续缓存，按 reference state/node、源条件、guidance、token 布局、捕获 block、processor/精度保存完整 K/V bank，并保持 [0,7) capture/apply 语义。native probe 与 capture 使用不同 attention 执行路径，不能仅凭相同输入直接替换；跳过 capture 前必须恢复对应 bank。各臂当前 z_i 的 source correction 和 target forward 继续独立计算。
6. **profile 与计数**：REQ/U25 保持各自原生轨迹，回放仍逐步匹配所选 anchor 的实际 RMS。nfe_incremental 记录真实新增调用；nfe_standalone 按单臂独立执行成本加回跨臂复用节省，不能重复累计同一调用。当前 donor 另算 6 NFE；共享 source probe 后重算 distinct calls，不能机械相加两份 6。anchor 成本继续读取记录，不写死常数。

父层建议（未改）：collect_delivery 的 160-anchor guard，以及 generation/evaluation 的 480-output guard 可由 plan.jobs 推导，保持与动态 planned 一致。40 源/两种子/六臂当前数量匹配。源码运行入口用顶层 run.py；历史 start.sh/run_all.sh 不作为本轮入口。
