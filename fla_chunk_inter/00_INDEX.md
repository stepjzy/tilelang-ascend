# fla_chunk_inter 迁移与调优学习档案

## 一句话：这个算子教会我什么

CV 中继型反传算子在 910B3 上的完整证据链交付：NK-merge / 1-late flag / C 侧对角化三级
结构性重构（19556→6868us，-64.9%），以及一次**交付期工具链缺陷的定位-证伪-结构修复
全程**（规划器注入 scratch 的 UB 打包随导入上下文翻转 → 坏打包即 Illegal instruction；
5 个 scratch 依赖 op 全消 + pip 旧版构建双变量归因修正）。

## 阅读顺序（约 4 h）

1. ref_gpu.py（30min）→ 迁移源头：Triton 语义权威（chunk_kda_bwd_kernel_inter，
   bwd inter-chunk dq/dk/dw/dg 单 kernel）；ref_gpu_tilelang_gpu.py 是上游 TileLang
   CUDA 版迁移起点 + 测试样例（bench main() 全 fp32）
2. DESIGN.md（60min）→ Developer 模式单 kernel 设计（fp16 hi/lo 3-gemm 拆分、内存
   预算表）+ §12-§15 四轮结构级增补（R2 NK-merge/1-late flag、R3 限制清单、D-2
   scratch 消除、R4 C 侧对角化收敛）
3. debug_log.md（50min）→ 核心坑：fp32 gemm_v0 损坏、同 UB 源双 L1 目的 sync-point
   失配、reduce_2d [64,64] 损坏、mul_add_dst 受限指令、**D-1 交付期缺陷定位（含
   8 次证伪实验与"路径敏感"归因的教训）**、D-2 结构修复、R4 收敛
4. perf_tuning/fla_chunk_inter_final_summary.md（50min）→ 三场战役 + 交付修复 +
   R4 收敛全景；perf_log.md 逐轮 A/B 数据
5. history_version/（30min）→ 25 个快照：首次实现 → R1 fixed-core/bDK=64 → R2
   NK-merge/1-late flag → R3 失败候选 → D-2 scratch 消除 → R4 三采纳

## 三大关键决策

1. **fp16 hi/lo 3-gemm 拆分**（fp32 gemm_v0 工具链损坏强制）：每操作数拆 hi+lo、
   三 gemm fp32 累加（~2^-22/操作数对）——精度政策决定中继字节，是一切带宽核算前提。
2. **NK-merge 任务重构造 + 1-late flag 配对**（R2，合计约 -42%）：任务从
   (b,chunk,head,dk-block) 合并为 (b,chunk,head)（-18.5%）；drain(j) 消费 j-1 的
   中继、等待 j-1 的 flag，消除同迭代锁步（V flag 等待 4896→0us，-29.4%）。
3. **交付期缺陷按证据链结构修复而非绕过**（用户决策）：规划器注入 scratch 的 UB
   打包次序随导入上下文（路径/模块文本/工具链构建）翻转，坏打包=设备崩溃；8 项
   候选修复证伪后定位到 op 家族层——消除全部 5 个 reduce/broadcast scratch 依赖
   op（+11% 性能代价），再经 R4 一轮收敛（C 侧 ones-gemm 对角化 + row_expand 门控
   + 中继读融合）回收 86.5% 代价至 6868us（vs 修复前 6767 仅 +1.5%）。
   "pip 旧版构建崩溃"与"真布局翻转"双变量归因的 bisect 方法是可复用的教训。

## 交付物

- 分支 feat/fla-chunk-inter-pr（origin，基于 upstream/ascendc_pto@557b11b8，
  commit ec3092d3）：examples_experiment/dsa_sparse_finetune/{example_fla_chunk_inter.py,
  test_example_fla_chunk_inter.py}
- PR: 待用户网页创建（文稿见本目录 PR_DESCRIPTION.md）
- 性能：do_bench 19556.3 → **6868.3 us（-64.9%）**；精度全绿（L0/L1 40 条
  ratio=1.0000，valrange_l dg 7.812e-3 ≤ 1e-2）；交付件在 worktree 真实路径 +
  异长度路径双验证通过（布局鲁棒）
- 工具链上游反馈四项（debug_log/DESIGN §14.3）：planner 指针序确定性、无锚
  head-set 丢弃、fused 列广播乘缺失、TIR while 死循环

## 遗留疑问 / 未走通的路

- scratch-free 形态地板 ≈ 6868us：dgm 中继回读（C 侧无 gather）、门控 2 转置
  （broadcast 为红线）、drain 树折叠语义限制——上游修 planner 指针序后可回收
  ~500us 级（debug_log next_hint）
- pip 旧版 tilelang（d75300ed）对本 kernel 族基础结构即崩——PR 注明最低工具链
  版本要求（当前源码构建）
- threads=2 三项上游修复（R3 限制清单 §3.1-1）解锁 40 AIV 后可达带再评估
- 多会话共用 journal worktree 时未跟踪归档文件可能被并行会话 git 操作清除——
  归档后立即 commit + push（本次 history_version/perf_tuning 曾被清除后重建）
