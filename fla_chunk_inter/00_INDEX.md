# fla_chunk_inter 迁移与调优学习档案

## 一句话：这个算子教会我什么

CV 中继型反传算子在 910B3 上的完整证据链交付：NK-merge / 1-late flag / C 侧对角化三级
结构性重构（19556→6868us，-64.9%），一次**交付期工具链缺陷的定位-证伪-结构修复
全程**（规划器注入 scratch 的 UB 打包随上下文翻转 → 坏打包即 Illegal instruction；
5 个 scratch 依赖 op 全消），以及**分设环境（CPU 编辑/NPU 远程执行）下的迁移重验证**
（精度/性能/双口径画像跨环境确认，attempt1 基线在异环境上下文崩溃——D-1 布局彩票
的独立复现，反证 D-2 修复的环境鲁棒性）。

## 阅读顺序（约 4.5 h）

1. ref_gpu.py（30min）→ 迁移源头：Triton 语义权威（chunk_kda_bwd_kernel_inter，
   bwd inter-chunk dq/dk/dw/dg 单 kernel）；ref_gpu_tilelang_gpu.py 是上游 TileLang
   CUDA 版迁移起点 + 测试样例（bench main() 全 fp32）
2. DESIGN.md（60min）→ Developer 模式单 kernel 设计（fp16 hi/lo 3-gemm 拆分、内存
   预算表）+ §12-§15 四轮结构级增补（R2 NK-merge/1-late flag、R3 限制清单、D-2
   scratch 消除、R4 C 侧对角化收敛）
3. debug_log.md（50min）→ 核心坑：fp32 gemm_v0 损坏、同 UB 源双 L1 目的 sync-point
   失配、reduce_2d [64,64] 损坏、mul_add_dst 受限指令、D-1 交付期缺陷定位（8 次证伪
   + "路径敏感"归因教训）、D-2 结构修复、R4 收敛、**R-EV 分设环境重验证（含长构建
   需心跳保活的平台教训）**
4. perf_tuning/fla_chunk_inter_final_summary.md（50min）→ 三场战役 + 交付修复 +
   R4 收敛全景 + R-EV 重验证记录；perf_log.md 逐轮 A/B 数据
5. history_version/（30min）→ 25 个快照：首次实现 → R1 fixed-core/bDK=64 → R2
   NK-merge/1-late flag → R3 失败候选 → D-2 scratch 消除 → R4 三采纳
6. perf_tuning/fla_chunk_inter_prof_final_npu.tar.gz（终态轮，解压后用 MindStudio
   Insight 打开）→ 图形化复盘终态 Timeline 与 pipe 占用，对照 final_summary 瓶颈
   归因；fla_chunk_inter_opprof_final_npu_opprof/ 为 CSV 数字口径（Excel/脚本查看），
   fla_chunk_inter_profiling_final_npu/summary.txt 为聚合摘要——Insight 打不开
   OPPROF 属正常（格式不可混用）。基线轮（attempt1）在分设环境上下文崩溃无法采集
   PROF，基线性能沿用原环境实测（19556.3us）

## 三大关键决策

1. **fp16 hi/lo 3-gemm 拆分**（fp32 gemm_v0 工具链损坏强制）：每操作数拆 hi+lo、
   三 gemm fp32 累加（~2^-22/操作数对）——精度政策决定中继字节，是一切带宽核算前提。
2. **NK-merge 任务重构造 + 1-late flag 配对**（R2，合计约 -42%）：任务从
   (b,chunk,head,dk-block) 合并为 (b,chunk,head)（-18.5%）；drain(j) 消费 j-1 的
   中继、等待 j-1 的 flag，消除同迭代锁步（V flag 等待 4896→0us，-29.4%）。
3. **交付期缺陷按证据链结构修复而非绕过**（用户决策）：消除全部 5 个 reduce/
   broadcast scratch 依赖 op（+11% 代价）→ R4 收敛回收 86.5% 至 6868us（vs 修复前
   +1.5%）；分设环境重验证全绿且 attempt1 基线在异环境崩溃——结构修复的环境鲁棒性
   得到独立复现证明。

## 交付物

- 分支 feat/fla-chunk-inter-pr（origin，基于 upstream/ascendc_pto@557b11b8，
  commit **e18a3b1d**（ruff format 门禁修订版））：examples_experiment/
  dsa_sparse_finetune/{example_fla_chunk_inter.py, test_example_fla_chunk_inter.py}
- PR: 待用户网页创建（文稿见本目录 PR_DESCRIPTION.md）
- 性能：do_bench 19556.3 → **6868.3 us（-64.9%）**；分设环境（DevEnvC_DCbyi_npu）
  重验证 6884.7us（+0.24% 噪声内）、msprof Task Duration 6673.2us、满频/20 核均衡；
  精度 --level all 全绿（ratio 全 1.0000，与原环境逐位一致）
- 工具链上游反馈四项（debug_log/DESIGN §14.3）：planner 指针序确定性、无锚
  head-set 丢弃、fused 列广播乘缺失、pip 旧构建兼容性

## 遗留疑问 / 未走通的路

- scratch-free 形态地板 ≈ 6868us：dgm 中继回读（C 侧无 gather）、门控 2 转置
  （broadcast 为红线）、drain 树折叠语义限制——上游修 planner 指针序后可回收
  ~500us 级（debug_log next_hint）
- pip 旧版 tilelang（d75300ed）对本 kernel 族基础结构即崩——PR 注明最低工具链
  版本要求（当前源码构建）
- threads=2 三项上游修复（R3 限制清单 §3.1-1）解锁 40 AIV 后可达带再评估
- 分设平台教训：后台 job 不算平台"活动"，长构建/长验证需 CPU 侧周期性轻量 exec
  心跳防 30 分钟空闲自动关机（R-EV 实证：首次构建被中断，增量续跑 + 心跳后成功）
- 多会话共用 journal worktree 时未跟踪归档文件可能被并行 git 操作清除——归档后
  立即 commit + push；journal 路径一律绝对路径
