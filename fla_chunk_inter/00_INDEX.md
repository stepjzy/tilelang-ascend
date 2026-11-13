# fla_chunk_inter 迁移与调优学习档案（两轮）

## 一句话：这个算子教会我什么

CV 中继型反传算子在 910B3 上的两轮完整证据链：结构优化全家桶（NK-merge /
1-late flag / C 侧对角化 / 装载批处理，19556→6741.8us，-65.5%）、交付期工具链
缺陷的定位-证伪-结构修复（规划器注入 scratch 的 UB 打包随上下文翻转即崩 →
scratch-free 红线成为第二轮设计的**首版硬约束**）、以及分设环境（CPU 编辑/
NPU 远程执行）下按新 migration skill 的全量重迁移实证（含 flag slot 碰撞这类
只有重做一遍才会暴露的隐患）。

## 阅读顺序（约 5.5 h）

1. ref_gpu.py（30min）→ 迁移源头：Triton 语义权威；ref_gpu_tilelang_gpu.py
   是上游 TileLang CUDA 版迁移起点 + 测试样例
2. DESIGN.md（60min，第二轮终版设计）→ scratch-free 红线首版强制 + 领域知识
   注入后的结构七件套；DESIGN_round1.md（30min）→ 第一轮设计及其 §12-§15
   结构演进史（D-1/D-2 缺陷定位与修复全程——两轮 DESIGN 对照读价值最大）
3. debug_log.md（60min）→ 两轮全部 Attempt：第一轮 6 坑（fp32 gemm_v0 损坏/
   sync-point 失配/reduce_2d 损坏/mul_add_dst 受限/D-1 交付缺陷/D-2 修复）+
   第二轮新发现（flag slot 10-workspace 碰撞、UB razor 边界三连撞、重排家族
   6 变体证伪、分设环境心跳教训）
4. perf_tuning/fla_chunk_inter_final_summary.md（50min）→ 第一轮四场战役 +
   第二轮章节（R2.0-R2.6）；fla_chunk_inter_r2_perf_log.md → R2 逐轮 A/B
5. perf_tuning/fla_chunk_inter_r2_pipeline_view.png（10min，直接看图）→
   三面板流水图：launch 时间轴/逐核双墙锁步/逐流水分解——终态瓶颈归因的
   图形化呈现（数据来自 r2 OPPROF + PROF，MindStudio Insight 替代物）
6. history_version/（30min）→ 26 快照两轮演进链
7. perf_tuning/fla_chunk_inter_r2_prof_{baseline,final}.tar.gz（按需）→
   MindStudio Insight 打开的终态/基线 Timeline（PROF 目录，Insight 唯一可用
   格式）；fla_chunk_inter_r2_opprof_iter{0,1}/ 为 CSV 数字口径

## 三大关键决策（第二轮视角）

1. **scratch-free 作为首版硬约束而非交付期补丁**：第一轮 attempt1 用
   reduce/broadcast 在沙箱好布局下全绿、交付期才崩（D-1 八项证伪定位到
   op 家族层）；第二轮设计直接内建替代形式（折叠树/对角化/行展开），
   Stage 2 首版即过且环境鲁棒——**缺陷知识前置到设计层的价值实证**。
2. **"首版即终态结构"的领域知识注入**：第二轮 Stage 2 从零重写的首版即
   达 6810.9us（vs 第一轮首版 19556us），因为七件套结构作为合法先例进入
   设计；Stage 3 战役则聚焦真正的新空间（装载批处理 -2.40% 采纳 + 重排
   家族 6 变体全谱证伪）——**重迁移的正确姿势是复用结论而非复用代码**。
3. **按证据链收官而非硬扛**：iter2/3/4 三连击（+4.29%/+0.70%/±0.00%）
   触发中止条件，双墙锁步地板核算闭合（AIC 6452/AIV 6449 逐核差全为正）
   → 用户先看流水图（r2_pipeline_view.png + PROF）后决策交付——检查点
   五选一流程的完整走通。

## 交付物

- 分支 feat/fla-chunk-inter-pr（origin，基于 upstream/ascendc_pto@557b11b8，
  commit **1d30c897**（描述↔代码双向一致性审计修正版，见 debug_log Attempt PR-CI3~CI6）：
  **examples_experiment/kda/FLA_KDA/**{fla_chunk_inter.py,
  test_fla_chunk_inter.py}——按新版 skill「交付路径+文件名镜像规则」
  （实现 = <源基名>.py、测试 = test_<源基名>.py，不另加前缀）
  （来源 examples/kda/FLA_KDA/fla_chunk_inter.py → 合入 examples_experiment/
  kda/FLA_KDA/）；CI 全绿
- PR: 待用户网页创建（文稿见本目录 PR_DESCRIPTION.md：19556→6741.8us
  前后比对 + 可达值依据，不含目标对比）
- 性能：第二轮终态 **6741.8us**（do_bench 3 次独立；OPPROF 6502.7 /
  PROF 6657.2 交叉验证）；vs 第二轮 Stage 2 基线 6810.9 = -1.02%；vs 首次
  正确实现基线 19556.3 = -65.5%
- 精度：两环境全量绿（第二轮 44 精度检查 ratio 下界 0.999939；L2 13/13
  拒绝；Boundary 17 PASS + 6 文档化告警）

## 遗留疑问 / 未走通的路

- 3400us 原指标 1.98× 未达：双墙锁步地板（第二轮 OPPROF 复验），突破需
  上游三项联合解锁（fp32 gemm / threads=2 vid / V↔C 直连）——其中
  **fp32 gemm 是决定性单项**（消 3× gemm 放大 + 中继字节减半）
- 两个未试的结构方向（分析已归档，天花板 ~5.5ms 仍不达标故未投入）：
  [do;dv] 行堆叠单 gemm（C 侧 -200~400us）与 DV 维任务切分 + 双缓冲
  （V 侧三引擎重叠，风险：UB razor + flag 预算 + 合并协议）
- pip 旧版 tilelang（d75300ed）对本 kernel 族基础结构即崩——PR 注明最低
  工具链版本要求
- 分设平台教训（已反馈 skill-journal）：长任务需心跳保活（后台 job 不算
  平台活动）；多会话 journal worktree 未跟踪文件有被并行 git 操作清除风险
  （归档后立即 commit+push）
