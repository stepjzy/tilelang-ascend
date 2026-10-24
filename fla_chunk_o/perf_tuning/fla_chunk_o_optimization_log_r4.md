# fla_chunk_o R4 优化记录（host 分发绕过轮）

> R4 输入状态：R3 终态（3-chunk + 协议地板论证），官方均值 1.492ms，精度
> 全过。用户决策"继续调优"，主攻方向由编排层给定：host 分发绕过 /
> 4-chunk 批处理 / 终态画像。本轮 P0 Host 侧优化（skill 固定优先级）+
> 结构性上限实证。前轮记录：optimization_log.md（R1）、_r2.md、_r3.md。

## Part A 优化点清单

- [#R4-1] [Host 侧 JIT 分发开销消除 — P0 Host 侧]（参考:
  optimization-guide.md §2.12 系（Host 侧预处理内化；本条的变体：消除的
  不是数据搬运而是**调度路径本身的每调用重建开销**），skill SKILL.md
  "固定优先级：先静态分析，再 P0 Host 侧优化"）：**适用** — 分解实测
  795.7us/调用（Analyzer 重建 289.4 + cython 逐参重解析 ~292 + 5×
  torch.empty 64.8 + stream 查询 35.1），其中必要成本仅 11.5us（JIT 缓存
  2.4 + lib.call 9.1）；wrapper（算子文件内）可实现缓存式低层直呼。
  R1-R3 期间未做的原因：彼时瓶颈在 kernel 侧（3.285→1.50），分发被设备
  饱和掩盖至首 rep；R3 终态后成为官方均值最大可寻址项。
- [#R4-2] [更深任务批处理（4-chunk）]（参考: optimization-guide.md §2.9
  Fixed Core 模式 + R3-4 3-chunk 先例）：**适用性存疑→实证不可行** —
  R3 实测 2→3 chunk 仅 -13us（摊薄边际递减：1/2-1/3=0.167 vs
  1/3-1/4=0.083，预计 3→4 约一半）；UB 预算前置检查发现 tilelang A2/A3
  硬上限 196352B（memory_planning.cc:35）而 3-chunk 已 100% 打满 →
  DK=DV=128 编译期段错误（faulthandler 实证 TVM OptimizeForTarget 内
  C 层 CHECK 崩溃）。**结构性封死，弃用**。
- [#R4-3] [终态画像复查]（skill Step 1 纪律：画像先行）：**适用** —
  R3 消融图谱后 kernel 结构变化（R3-4 采纳 3-chunk）+ R4 新增 host 旁路，
  需确认无新可寻址 stall。结果：无（双侧占满墙，协议地板复现）。
- [禁止重试清单核对] threads=2 裸切 / PTO target / L1 同名 ping-pong /
  纯自动 flag 2-late drain / 深度节流 lag 6/10 / hi/lo 行打包 / K-concat
  单 gemm / bf16 域掩码 / gate 链融合 / 去 hi/lo 双 gemm / block_DV=64 /
  O 写回迁 V / C 侧全批装载上移 / Select 地址链 / v in-body vs 预取 —
  **全部保持禁止**（R1-R3 证伪，本轮无新前提）。

## [ORDER-PLAN] 实施顺序：
1. [#R4-3] 终态画像与分发分解（并行采集基线） — 前置依赖: [无] — 理由:
   画像先行纪律 + 分解数据决定 #R4-1 的实现设计。
2. [#R4-1] host 分发绕过 — 前置依赖: [#R4-3 ✅] — 理由: 分解确认浪费
   构成（Analyzer/cython 杂项/分配/stream），旁路设计据此定型；P0 Host
   侧优先于一切 kernel 侧改动。
3. [#R4-2] 4-chunk 批处理 — 前置依赖: [#R4-1 ✅] — 理由: 在最终 host
   口径上做 A/B；前置检查（UB/L1/环安全/flag ID 空间）先行。

## 实施记录

[ORDER-CHECK] 准备实施: [#R4-3 画像/分解] | 前置依赖: [无] | 结论: ✅
[RESULT-R4-3a] 分解（fla_chunk_o_dispatch_decompose_r4.py）: 全 wrapper
  795.7us = Analyzer 289.4 + cython 杂项 292 + alloc5 64.8 + stream 35.1 +
  JIT 2.4 + 余量；投影快路径 132us。
[RESULT-R4-3b] msprof（msprof_output_r4）: 单_launch 1391.1us；AIC cube
  12.7% / mte2 38% / relay 等待 ~733us（重叠）；AIV throttle 等待 1114.6us
  （lag-2 设计使然）；无新可寻址 stall。

[ORDER-CHECK] 准备实施: [#R4-1 host 分发绕过] | 前置依赖: [#R4-3 ✅] | 结论: ✅
[IMPL-R4-1] 已阅读 optimization-guide.md §2.12 系 + tilelang/jit/
  adapter/cython/cython_wrapper.pyx L75-203（forward 全路径），关键约束:
  (a) 输出必须 fresh 分配（上游语义不别名）；(b) workspace 为内部暂存可
  复用（同流串行 + write-then-read）；(c) stream/device 入缓存键；(d)
  全静态 kernel 无 dynamic symbolics（adapter.dynamic_symbolic_map={}，
  探针实证）。
[KERNEL-REUSE-AUDIT] 依据: §2.12（host 侧，不触碰 kernel 调用域） |
  维度及结论: 调用域不变（同 spec 同 shape 集合；spec 键失配自动回退公共
  路径重新特化） | 结论: reuse（kernel 二进制零变化）
[SELF-CHECK] 本次 Edit 只涉及 [#R4-1]（fla_chunk_o.py wrapper 段 + ctypes
  import + _WS_IDX/_OUT_IDX 常量；kernel 体零改动）
[RESULT-R4-1] 官方 A/B: 1.492 → **1.416**（-76us，-5.1%）；分发 795.7 →
  163.4us；首 rep 暴露 0.83 → 0.13ms；精度 --level all 11/11 + L2 10/10 +
  Boundary 全过（含双 stream 交错 / 21+ 次复用 / 非连续回退）。

[ORDER-CHECK] 准备实施: [#R4-2 4-chunk] | 前置依赖: [#R4-1 ✅] | 结论: ✅
  （前置检查：UB 估 232KB<256KB（当时按 256KB 估——后被证伪：实际上限
  196352B）；L1 352KB<512KB；环安全 NSLOT=12；flag ID 空间探针通过
  （THROTTLE 48→64 行为一致 → ≥65 ID 可用））
[IMPL-R4-2] 生成器 gen_r4_body.py（4-chunk 参数化）+ 头部改造（staging×4、
  L1×4、NSLOT 12、THROTTLE 64、NB/K_MAX 公式）。
[RESULT-R4-2] (1,128,2,64,64) 编译+精度 PASS；(2,256,4,128,128) **编译期
  段错误 exit 139**。根因: **UB 硬上限 196352B，3-chunk 已 100% 打满**
  （kernel_source_r3_final.c L42 = 196352 == memory_planning.cc:35 常量）。
  4-chunk staging +36864B/vid 溢出 → C 层 CHECK 崩溃穿 FFI。**弃用回滚**
  （md5 校验恢复 r4_iter1）。

## R4 结论
- 采纳 1 项（#R4-1，-76us）；弃用 1 项（#R4-2，结构性）；记录 1 项
  （#R4-3，无新发现）。
- 终态：官方均值 **1.417ms**（公共路径口径 1.492），稳态 ~1.406，msprof
  设备侧 1391.1us。**目标 1320us 在当前同步协议架构下不可达（实测封闭：
  设备稳态 > 目标）**→ [PERF_INCOMPLETE]，剩余差距全部为工具链级（relay
  合并被 zN 布局阻断 + AUTO_SYNC 每 op 同步指令流 + UB 硬上限）。
