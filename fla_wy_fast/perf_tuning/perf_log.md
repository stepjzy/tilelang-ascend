# perf_log — fla_wy_fast（Stage 3 性能调优跨轮日志）

## Iteration 1 — 2026-10-22T05:55:00+08:00
- bottleneck_type: compute（Vector 核串行行循环的指令下发开销，非带宽）
- optimization: 8 项尝试，4 项采纳——
  1. `T.tile.broadcast` + 全量 tile 指令替换 6 个 kernel 的 `T.serial(64)` 行循环
     （fwd-u/-wkg、k1/k2/v1/v2，含 fwd-wkg kg 列广播链 axis=0）；
  2. host Msk 常量 per-(device,chunk) 缓存（去掉计时区内 CPU tril + H2D）；
  3. wrapper 内部 `torch.npu.synchronize()` 移除（对齐上游 do_bench 语义，
     上游算子 wrapper 内部无 sync）；
  4. kernel 文档同步更新（广播方案、cast-on-copy 工具链限制）。
  回滚 4 项：A cast 链 prologue hoist（中性，软流水已重叠）、dbeta scatter 移
  da（净 +0.04ms）、k2 CSE（UB 溢出）、kg div 变体（持平）。
- baseline_time: 23.151 ms（fwd 8.745 + bwd 14.406，Stage 2 口径含 wrapper sync）
- candidate_time: 13.084 ms（同口径）/ 11.719 ms（上游对齐口径，fwd 4.783 + bwd 6.935）
- improvement: -43.5%（同口径）/ -45.8%（上游口径）
- precision: pass（--level all 退出码 0；L0/L1 8/8 PRECISION_PASS，L2 7/7 正确
  拒绝，Boundary 5/6 PASS + 1 预期内 WARN——与基线行为一致的 fp16 拆分范围限制）
- adopted: yes
- rollback_reason: n/a（4 项子尝试按规则各自回滚，见 optimization_log.md）
- next_hint: 进一步收益需工具链修复——多 gemm 组 L0 staging、sync-point 计数
  （z+da 合并）、UB↔L1 片上直连、GM↔UB bf16↔fp32 cast-on-copy、JIT launch
  路径每 call 重建 Analyzer（~1ms）；算子层剩余空间 <10%。
