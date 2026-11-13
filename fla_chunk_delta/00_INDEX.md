# fla_chunk_delta 迁移与调优学习档案

## 一句话：这个算子教会我什么

串行递推型算子的性能地板不在带宽或算力，而在**跨核往返延迟**——两轮 msprof 证据链
（含一次对第一轮归因的实测证伪）如何把"墙"从 barrier 成本修正为往返等待，以及
golden 量化点契约如何反过来封锁 kernel 层优化空间。

## 阅读顺序（约 3 h）

1. ref_gpu.py（30min）→ 迁移源头：Triton chunk_gated_delta_rule fwd/bwd，
   每 (b,h,i_v) 链 fp32 状态 [DK,DV] 按 chunk 串行递推，g/gk 双门控
2. DESIGN.md（40min）→ Developer 模式 fixed-core 链交织 + 状态驻 UB + 4 relay
   交换 + gemm slab 合并的完整设计；§10 roofline 与风险分级
3. debug_log.md（30min）→ 核心坑：threads=2 vid 分裂下 gather/broadcast 不可用
   （源码级验证 + vid-safe 等价实现）；DESIGN golden 转录顺序错误的上游对拍修正；
   R11 fwd 2-chain UB 284KB 溢出崩溃（planner lifetime 盲区）
4. perf_tuning/fla_chunk_delta_final_summary.md（30min）→ 16180.5→6077.3us
   （-62.5%）怎么来的；两轮归因修正史（barrier 假设被 R12 census 证伪 →
   真因是串行往返等待）；地板核算 6.0ms ≈ 实测
5. history_version/（20min）→ 被否路线：fwd 3 链批（+3.6%）、bwd BV=128
   （L0C 溢出）、q~ 列 region UB 写（nan）、fwd 2-chain（崩溃 507015）、
   ws5/ws6 合并（破坏 golden 封锁的 scale 折叠→精度 FAIL）
6. perf_tuning/fla_chunk_delta_prof_r13/**（终态轮，PROF 目录直接用 MindStudio
   Insight 打开）→ 图形化复盘 AIC/AIV 双侧 41-51% wait_id* 互等、Cube 仅
   6.6-8.4%、block 4-9 尾批提前完成；对照 final_summary 地板核算验证归因；
   fla_chunk_delta_prof_r1/** 为基线轮；OPPROF_*/ CSV 为数字口径（Excel/脚本）
7. perf_tuning/fla_chunk_delta_{fwd,bwd}_kernel_source_*.c → 生成 C 源码证据链
   （Select 风暴消除前后、barrier 计数）

## 三大关键决策

1. **状态驻 UB fp32 + L2 relay workspace 交换**（否决 L0C 驻留：与 GEMM 累加器
   争用且 128KB 上限装不下 bwd 双缓冲）——递推状态跨 chunk 存活，UB 192KB 刚好
   容纳 [128,128] fp32 + ping-pong。
2. **threads=1 prep kernel 根除标量提取链**：Stage-2 实现的每 chunk 192 次标量
   迭代（S-pipe flag + PIPE_ALL barrier）是最大单项开销（AIV scalar 79-84%），
   预计算门控/衰减 tile 后 fwd -62~64%。二轮再把 q~ 外移 prep（bwd -7.5%）。
3. **性能目标 3400us 的证据链判定**：两轮 13 次迭代 + 双口径画像 + 地板核算
   （fwd main ~2.5ms ≈ 实测 2.57、bwd ~2.9ms ≈ 实测 3.04）证明达标需同时减半
   往返数与延迟——分别被 golden cast 点契约（实测折叠 dh0 ratio 0.9601 FAIL）
   与同步工具链粒度封锁；用户裁定按实测值交付并在 PR 如实披露。

## 交付物

- 分支 `feat/fla-chunk-delta-pr`（origin，commit e7df98af，含 CI smoke main）
- 合入路径：`examples_experiment/kda/FLA_KDA/{fla_chunk_delta,test_fla_chunk_delta}.py`
  （镜像来源 tile-ai/tilelang examples/kda/FLA_KDA/fla_chunk_delta.py）
- PR：待用户创建（PR_DESCRIPTION.md 为正本文稿）

## 遗留疑问 / 未走通的路

- AUTO_SYNC 粒度：每 V-op 语句间 PIPE_V barrier（~0.10us/条 vs chunk_o 无阻塞流
  0.04-0.05）——工具链级改进项，已记 skill-journal
- fwd 2-chain BV=128：UB 284KB>196KB + planner lifetime 重叠盲区 + 跨 pipe WAR
  漏保护（aicore exception 507015）——需工具链修复后重试
- golden 量化点契约：`qt=bf16(q·e^g)` 未缩放、`vn16` fp16 cast 固化了上游
  cast 顺序，封锁中继合并类优化（精度语义正确，优化空间代价）
- NCHAIN=128/20 核尾批 ~12% 浪费：3 链批实测更差，整除结构（140/120 链）未试
