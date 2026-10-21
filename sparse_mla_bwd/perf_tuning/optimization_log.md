# sparse_mla_bwd 优化点清单（tilelang-perf-optimization Step 3）

算子类型判定（Step 2）：**混合型**（生成代码同时含 IS_ASCEND_AIC 的 gemm_v0 链与
IS_ASCEND_AIV 的 tile/atomic 链；OpBasicInfo Op Type = mix, Block Dim 20 / Mix 40）。

## Part A 优化点清单

[#1] **两遍结构的第二遍 V 侧 gather 消除（ws_kv 按块暂存）**（参考:
  optimization-guide.md §2.8 稀疏访存 gather 模式 + 本仓 fwd iter-2 先例；
  performance-antipatterns.md L124/L146 同族）：**适用（本轮已实施）** —
  msprof 实证 AIC 86% 时间等 ws_kv flag（id8 28.4ms + id10 37.9ms），V 侧
  MTE3 39.8ms@3.09GB/s 微写为关键路径；两遍结构把逐行 gather 做了两遍，
  其中第二遍纯冗余（数据与第一遍逐位相同）。stash [grid, NI+1, BN, DDT]
  + pass-2 CV 配对 dummy 写（scratch 槽）。实测 k2 79371→55691us（-29.9%），
  dQ 逐位一致。约束（combinecv.cc L358-395）：每 workspace 语句数 V=C 且
  按序配对；**跨核 flag 预算 ≤16 对**（本轮平台发现，见 [#4]）。

[#2] **pass-1 gather 批量 staging（fwd iter-2 3:3 形态）**（参考:
  optimization-guide.md §2.8 优化后形态 + fwd perf iter-2 生产先例）：
  **适用但本轮被平台约束阻塞** — 剩余 MTE3 22.6ms@4GB/s 中 pass-1 微写
  ~30MB 仍是主项之一。two-batch 3:3 需要 17 个同步对（flag id 16 溢出，
  见 [#4]）；single-batch [64,576] 整 2D 写触发编译器 SEGV（OptimizeForTarget
  内，"Vec loop times (=1) not enough to catch up cube loop times (=4)" 族）。
  iter-1 路线：先合并 ws_p/ws_dpb hi+lo 通道为单条 3D 语句（省 2 对 → 15
  对），再上 3:3。探针前置：2D UB plane 目的切片（p_hilo[0,:,:]）合法性、
  [2,M,BN] 3D L1 作 GEMM operand（q_all[s] 同款 dim-0 切片先例）。

[#3] **dKV 原子散射路径**（参考: optimization-guide.md §2.5/带宽分层；
  DESIGN §6.5 方案 2）：**适用（iter-1 主攻）** — 剩余 V 侧最大流量项：
  ws_dkv 读回 60MB + fp32 原子写 60MB / AIV（考核 shape 下 2.4GB 全局原子
  流量，每行平均被 ~448 个 token 贡献）。当前无分离测量——iter-1 先做
  var_noatomic 差分（删 C 侧 G5/G6 ws_dkv 写 + V 侧 scatter 段，保持配对
  0:0）定量，再决定：原子吞吐实测 vs 方案 2（per-core 私有 + 归约）vs
  段聚合。注意 pto 后端 atomic 语义差异（DESIGN R5）。

[#4] **跨核 flag 预算（平台约束，非优化点——约束记录）**：硬件跨核 flag
  id 上限 16（0..15）。原始实现 14 对、iter-0 采纳版 16 对（贴线）。第 17
  对的 id 16 混叠到 flag 0（ws_dkv 生产者每块置位）→ 配对等待立即释放 →
  读到陈旧 workspace（NI=1 被同值掩盖、NI=4 dQ 28%/dkv 99% 损坏）。
  tilelang 侧 sync_flag_id_++ 无上限检查（ascend_combinecv.cc L111）——
  任何语句数重构必须复核对数 ≤16。

[#5] **V 侧 P/dP 链逐行算子向量化**（参考: performance-antipatterns.md
  L124 + fwd iter-3 先例；vector-practices/vector_reduce_pass_fusion.md）：
  **适用（iter-2 候选）** — `for h in range(v_block)` 两处 mask/bias 行乘
  （32 行/块/遍 ×2 遍）+ hi/lo 拆分链 ~10 op；fwd 同款改造后 softmax 链
  448 行算子 → ~74 组级 op（-21%）。bwd 当前 vec 仅 1.8ms（3%），收益
  上限有限，排在 [#2][#3] 之后。

[#6] **hi/lo 双通道 plain-bf16 回退**（DESIGN §13.1 缓解分支；任务书已知
  线索 2）：**低优先** — C 侧 cube 纯算仅 2.6ms（4.7%），GEMM 非瓶颈；
  hi/lo 的 V 侧代价仅 ws_p/dpb 4×8KB/块（3.3MB 总量，<3% MTE3）。回退
  收益 <1ms 且必须过域 B 0.999 收紧门限——性价比不足，暂缓。

[#7] **k3 postprocess**（810us vs 预算 80us）：**适用（次要）** — 4.5MB
  fp32 读 + 2.25MB bf16 写 @ 8.3GB/s（rows_per_aiv=16 窄行 + launch 开销）。
  cast 行宽 16→64（fwd/indexer 先例）预计 → ~200us。占比如今 1.4%，iter-2
  顺手项。

[#8] **Host 侧轻量化**（optimization-guide.md §2.12）：**基本不适用** —
  dQ/dKV 分配+清零 ~190us（已测）、2×launch marshal 在 k2 执行期隐藏；
  do_bench 全接口口径下 host 占比 <1%。prepare_token_indices 已有内容键
  缓存。

## [ORDER-PLAN] 实施顺序（iter-1 起）

1. [#3] 原子散射差分定量（var_noatomic）— 前置依赖: 无 — 理由: 剩余
   最大流量项，决定 iter-1+ 的架构走向（原子 vs 方案 2）。
2. [#2] 批量 gather（via [#4] 预算腾挪：ws_p/dpb 3D 合并）— 前置依赖:
   [#3] 的测量结果排序 — 理由: 两个 ~10ms 级项按实测收益排序实施。
3. [#5] P/dP 链向量化 — 前置依赖: [#2]（UB 布局变动后再腾 bcast 槽）。
4. [#7] k3 cast 行宽 — 独立小项。
5. [#6] hi/lo 回退 — 仅当 [#2][#3] 后 C 侧成为瓶颈时重估。

iter-0 已实施: [#1]（采纳，-29.9%）。[#4] 为平台约束记录，贯穿所有后续项。

## iter-1 结果（2026-10-15，详见 perf_log.md Iteration 1）

- [#3] 差分定量完成：dkv 路径边际 22776us（40.6%），**原子溢价仅 1367us（2.4%）**
  → 方案 2（私有+归约）**否决**；真瓶颈=逐行串行发射（PIPE_ALL+GM 标量读+微读+握手）。
- 采纳：gather full+mini staging（fwd 3:3 的 2:2 形态，320 微写+256 握手/块→2 语句）
  + scatter 载荷瘦身（idx_ub 源 pos + 段微读合并）→ 全接口 56364→32586us（**-42.2%**，
  pto -42.6%），flag 对数不变 16。精度全量门禁 pass。
- [#2] 部分落地（gather 侧）；scatter 侧 3:3 被第 17 对 flag 卡死——省对路径全部受阻：
  (a) 3D-of-4D region T.copy 静默截断（双后端，smoke 实证）；(b) 640 宽 stash 编译
  SEGV（NI=4）；(c) gemm_v0 双操作数同 dtype。三条平台约束已固化 kernel docstring。
- [#5]（P/dP 链向量化）未动——vec 仅 1.8ms（5.5%），优先级维持 iter-2+。
- [#7]（k3 行宽）未动——810us 池，iter-2 顺手项。
- iter-2 排序（按新画像）：1) ws_dkv 读回批量（前置=解 −1 flag：修 pass 侧 640-SEGV
  或等效）；2) k3 行宽 16→32；3) 锁步结构评估（AUTO 栈封死，fwd iter-4 RC1-RC5；
  混合模式=架构级，需 Orchestrator 决策）。

## iter-2 结果（2026-10-15，详见 perf_log.md Iteration 2）

- **新机制发现（源码级）**：combinecv 跨核配对只遍历 cube 侧 workspace 名单 →
  纯 V 侧 GM transit（命名避开 "workspace*"）**零 flag 对、无语句数约束**；
  同核 RAW/WAR 由通用 auto-sync 按缓冲名依赖分析覆盖。此机制解锁了 [#2 单遍化]
  的一个此前被否决的子路线。
- 采纳：A) P/dP_raw V-only fp32 stash（pst_p/pst_dp）——pass-2 G1/G2 重算、mask
  链、ws_s/ws_dp 读全部删除，s/dp flag 2→1（16→14 对），dQ 逐位一致（k2 −8.6%）；
  B) k3 行宽 16→32（逐位一致）；C) mask/bias 向量化（fwd iter-3 形，bit-exact）。
  合计全接口 32586→28541us（**−12.4%**，pto −7.9%），精度全量门禁 pass。
- [#2 遗留 scatter 侧（ws_dkv 批量读回）]**结构性证伪**：UB 上限 196352B 且
  规划器不跨 symbolic 任务循环回收 pass-1 槽位 → 147KB / 73KB staging 均
  "available: 0"；即便 flag 解锁（P-stash 已省 2 对，15 对预算内可容纳 3 点形态）
  也放不下。C 侧按段拆分受 L0C 累加器复用限制；按行半拆分破坏 vid 数据就绪
  对称性（按序配对语义）。**逐行 scatter 微读模式在本编译栈冻结**。
- [#5] 向量化已落地（vec 1.79→0.99ms）。[#7] k3 已落地（810us 池收干）。
- iter-3+ 排序：无剩余 AUTO 栈内高价值项——**[DESIGN_ERROR] 候选判定成立**
  （证据：AIC 等 flag 73%、AIV 38%、读量 ~147MB/AIV ≈ 12ms 带宽下限 > 目标
  7680us；本轮 6+ 条受阻路线各有复现签名）。Expert/混合模式 + 数据流级读量
  削减为唯一可达路径，提请 Orchestrator 决策。
