# fla_chunk_o Stage 3 最终交付报告（fla_chunk_o_final_summary.md）

> 设备：**Ascend 910B3**（64GB HBM，20 AI Cube + 40 AI Vector cores，L2 192MB）
> CANN 9.2.0 ｜ tilelang editable @ 2f22c6ac ｜ 日期：2026-10-22
> 计时口径：**与 test_fla_chunk_o.py --level perf 完全一致**（NPU 事件，
> warmup=10，rep=10，均值，仅设备侧 kernel 时间，不含张量构造/H2D-D2H/JIT/
> 框架开销；wrapper 内无 sync，对齐上游 do_bench）。

## 1. 基线 → 最终对比

| 项 | 基线（Stage 2 交付） | 最终（Stage 3） | 改善 |
|---|---|---|---|
| bench shape (1, 8192, 64, 128, 128) bf16+fp32 gate | 3.285 ms | **1.638–1.658 ms**（4 次独立官方运行） | **-49.5～-50.2%** |
| 稳态（去掉首 rep 冷启动） | ~3.20 ms | ~1.55-1.59 ms | -51% |
| 每（chunk, b, h）任务时间（×20 核） | 8.0 us | 4.0 us | -50% |
| **目标 < 1320 us** | 超 2.49× | **超 1.24× → 未达标** | — |

最终 per-rep（1.638 ms 运行）：[2.294, 1.506, 1.542, 1.546, 1.637, 1.562,
1.484, 1.577, 1.586, 1.644]（首 rep 为冷启动）。同会话 4 次独立官方运行：
1.638 / 1.646 / 1.648 / 1.658 ms（±2% 设备噪声；取 1.65 ms 为代表值，
对应 -49.7%）。

## 2. 各优化手段及贡献（同会话 A/B 消融，逐项实测）

| # | 优化手段 | 前 → 后 (ms) | 贡献 | 采纳 |
|---|---------|-------------|------|-----|
| 1 | threads=2（含 3D 视图修复 vid 落维 bug） | 3.285 → 3.264 | 中性（结构基础） | ✅ |
| 2 | i_k Pipelined 循环展开为直线全 DK（大 DMA、中继 4→2、gemm 4×K64→2×K128、flag 减半） | 3.264 → 2.993 | **-8.3%** | ✅ |
| 3 | Fixed-Core grid（20 核 + 核内任务循环）+ 显式小 workspace（~2.9MiB 常驻 L2，替代 ~768MiB/launch 自动注入）+ Msk hoist | 2.993 → 2.245 | **-25.0%** | ✅ |
| 4 | 循环内 if 全消除（branch-free Select + 跨步任务映射）—— AUTO_SYNC 对每个 IfThenElse 插 2 次 PipeBarrier<PIPE_ALL> | 2.245 → 2.126 | -5.3% | ✅ |
| 5 | 早期 a_mask 中继 + intra-gemm 先行（与 V 的 gate 链重叠）+ 下任务 A/Q/G loads 预取 | 2.126 → 1.859 | **-12.6%** | ✅ |
| 6 | 手动 cross-core flag（ID 按奇偶）+ 输出 drain 延迟 2 任务（打破 AUTO_CV_SYNC FIFO 配对造成的 V/C 锁步）+ NSLOT=4 + 2 peeled 填充 | 1.859 → 1.691 | **-9.0%** | ✅ |
| 7 | o 中继 bf16 化（Cube 侧 fixpipe 完成 fp32→bf16，V 侧免 cast） | 1.691 → 1.670 | -1.2% | ✅ |
| 8 | ws1/ws2 中继读批量化（MTE2 latency 重叠） | 1.670 → 1.603 | -4.0% | ✅ |
| 9 | L1 ping-pong + h/v 预取（2 变体） | 1.603 → 1.621 / 1.685 | 无收益 | ❌ 回滚 |
| 10 | 手动 flag ID 3/4 → 12/13（消除与自动 flag 0..8 的潜在冲突；安全修正） | 持平 | 0 | ✅ |

根因链条：3.285 →（展开）2.99 →（fixed-core）2.25 →（去 if）2.13 →
（重叠微结构）1.86 →（破锁步）1.69 →（bf16 中继 + 读批量化）1.60-1.64。

## 3. 瓶颈定位方法与关键证据

1. **生成代码审查**（get_kernel_source，perf_tuning/fla_chunk_o_kernel_source_*.c）：
   - 基线：Pipelined(num_stages=2) 实际未软件流水（单缓冲顺序循环）；
     全部 V↔C 交换经 4 个自动注入 GM workspace（+152KB/块逻辑流量）；
     每块 ~10 次跨核 flag。
   - 中途：循环内 if 被 PipeBarrier<PIPE_ALL> 包围（每迭代 ~8 次全流水排空）。
2. **差分探针**（fla_chunk_o_probe_breakdown.py / _sides.py）：skeleton
   （纯搬运+中继+同步）2.245ms = 完整 kernel 的 75% → **延迟/同步型而非
   带宽/计算型瓶颈**。
3. **msprof op**（msprof_output*/）：Block Dim 20 / Mix Block Dim 40（实证
   910B3 = 20 cube + 40 vector）；锁步期两核 ~80% 时间互等 flag
   （AIC wait_id0 13.2/16.7us 等 V 中继；AIV wait_id3 15.0/17.3us 等 FIX）；
   最终态：AIC scalar 53%、mte2 29%、cube 7%；AIV vec 13%、mte2 28%。
4. **HBM 参照**：同族 fla_wy_fast D2D 实测 1263GB/s（r+w）。

## 4. 有效带宽与目标差距（如实）

| 口径 | 数值 |
|---|---|
| 唯一 IO（960MiB）÷ 1.638ms | **597 GB/s = 实测上限的 47%** |
| 逻辑流量（唯一 + 中继 ~120KB/任务）÷ 1.638ms | ~715 GB/s |
| 1320us 目标隐含（唯一 IO） | 745 GB/s = 59%（**未达**） |
| IO 硬下限（960MiB ÷ 1263GB/s） | 797 us |

**结论：目标未达成，但非"物理不可达"**——HBM 仅 47% 利用（非带宽墙），
差距来自本工具链 CV 中继模型的结构性同步开销（见 §5）。若按 IO 下限
797us 与同族单 kernel 实证 81% 带宽利用率推算，物理可达下限 ~1.0ms，
当前 1.638ms 距其 1.6×——剩余差距是工程/工具链问题，不是硬件物理极限。

## 5. 未达标根因论证（结构性，按影响排序）

1. **每任务跨核中继的固有串行链**：本算子每个 (chunk, b, h) 任务需要
   3 次 V→C 中继（a_mask、qg_hi、qg_lo——精度回退方案强制 hi/lo 双份）
   + 1 次 C→V（o）+ 3 gemm + 5 次 GM→L1 装载。AIC 串行链实测 ~4.0us/任务，
   其中 mte2（88KB）1.35us、gemm+L0 staging ~0.7us、FIX 0.15us，其余
   ~1.4-1.8us 为**工具链每 op 同步开销**（MTE2_M/M_FIX 握手、屏障、flag、
   DMA 描述符、地址 Select 链）——msprof AIC scalar 活动占 53%。
2. **AUTO_CV_SYNC 的 FIFO flag 配对**：自动 flag 按语句出现序严格配对，
   V 对中继的第 i 次读必须等 C 的第 i 次写 → 跨任务流水被强制锁步。
   已用手动 flag（混合模式）为 o 路径绕过；hi/lo/a 三对仍受此模型约束
   （V 领先 ~2 任务后等待≈0，但配对语义不允许更深的重排）。
3. **UB 容量阻塞任务批处理**：当前每 vid UB 占用 184KB（96%）。2-chunk
   任务批处理（每 token 的 flag/循环/drain 开销减半，估算可达 ~1.25ms）
   需要 [2BS,·] 级别的工作缓冲，实测溢出（~350KB/vid）。
4. **9 个必备 op × 每 op 开销的下限**：即使所有传输/计算全部重叠，
   9 op 的串行下发+同步 ≈ 1.4-1.8us/任务 → ~0.6-0.75ms 纯开销，
   加 2.0-2.4us 纯 pipe 工作 → 理论最优 ~1.05-1.25ms（与目标同量级，
   需要更深的 op 融合/更少的中继，受 1-3 限制）。

## 6. 已尝试无收益/被阻塞的优化清单

| 项 | 结果 |
|---|---|
| threads=2 裸切换 | 精度损坏（vid 落维 bug）→ 3D 视图修复后中性 |
| PTO target（pipe FIFO） | 3.104 ms，劣于 ascendc（2.993）→ 弃 |
| L1 ping-pong + h/v 预取（2 变体） | 无收益/劣化（保守屏障抵消）→ 回滚 |
| 纯自动 flag 的 2-late drain | 死锁（FIFO 配对实证）→ 手动 flag 替代 |
| 2-chunk 任务批处理 | UB 溢出（184→350KB/vid）→ 弃 |
| hi/lo 单中继（行打包 [hi;lo]） | vid 行切分使装配布局跨步，region gemm 不可表达 → 弃 |
| K-concat 单 gemm（[hi\|lo]@[h;h]） | 列打包无法用连续 cast 构造 → 弃 |
| bf16 域掩码 | T.tile.mul 不支持 bf16 → 弃 |
| gate 链算子融合（scale 折入 ln2 等） | op 数不减（数学等价变形均 2 op）→ 弃 |
| 移除 hi/lo 双 gemm | 精度 FAIL（0.988 < 0.99，Stage 2 已证）→ 保留 |
| block_DV=64 变体 | DESIGN §10.4-A 已否（+384MiB IO）→ 未试 |

## 7. 精度复验（最终门禁）

`python test_fla_chunk_o.py --level all` → **Test Passed!（exit 0）**
- L0 2/2、L1 9/9 全部 `[PRECISION_PASS]`（ratio=1.0000；含 bench shape、
  DK=96/DV=48 非二次幂、质数轴、值域、非默认 scale）
- L2 10/10 正确拒绝（S%64≠0 ×3、DK 32/80、DV 100/144、chunk32、dtype ×2）
- Boundary 3 PASS + 2 个 Stage 2 已记录的预期 WARN（bnd_inf_g：hi/lo 在
  inf 下的固有局限；bnd_g_dbound：超出精度标准可认证域——两者与基线
  行为一致，非本轮调优引入）
- bf16 fixpipe 中继转换（iteration 7）经全量门禁验证精确
- 每轮采纳前均过 l0_repr 快速精度门；全量复验在 iteration 7 后与最终
  交付前各执行一次

## 8. 交付文件清单（perf_tuning/ 下，fla_chunk_o_ 前缀）

| 文件 | 内容 |
|---|---|
| `fla_chunk_o_final_summary.md` | 本报告 |
| `fla_chunk_o_perf_log.md` | 跨轮结构化日志（含全部迭代数据） |
| `fla_chunk_o_optimization_log.md` | skill 规范的优化点清单/ORDER-PLAN/RESULT |
| `fla_chunk_o_baseline_iter1.json` | 基线记录（口径/shape/分析） |
| `fla_chunk_o_bench_iter.py` | 同口径快速 A/B bench + 快速精度门 |
| `fla_chunk_o_probe_breakdown.py` / `_probe_sides.py` | 差分探针（瓶颈定位） |
| `fla_chunk_o_dump_kernel_source.py` / `_dump_fa_source.py` | 生成代码转储（含 threads=2 先例验证） |
| `fla_chunk_o_kernel_source_iter{1,1_threads2,2_unrolled,3_fixedcore,4_noif,6_manualflag}.c` | 各阶段生成代码证据 |
| `fla_chunk_o_impl_iter{1_before,1b,2,3,4,5,6,7,8}.py` / `fla_chunk_o_impl_final.py` | 各方案转折点快照 + 最终版 |
| `msprof_output{,2,3}/` | 三次 msprof op 画像数据 |

主实现：`../fla_chunk_o.py`（接口不变：fla_chunk_o(q,v,g,A,h,scale,chunk_size)；
测试 `../test_fla_chunk_o.py` 未改动）。

## 9. 对上游/后续的建议

1. 工具链：AUTO_CV_SYNC 支持按 slot/ID 配对（解除 FIFO 锁步）；IfThenElse
   的 PIPE_ALL 屏障按实际冲突插；MTE2_M/M_FIX 握手批量化。
2. 算子层：若允许混合/Expert 模式全量手动同步 + 更省 UB 的 hi/lo 表示
   （如 fp16），2-chunk 批处理 + 深度流水有 ~1.25ms 的可达空间。
3. 验收口径注意：perf_bench 首 rep 含 ~0.7ms 冷启动（均值含首 rep，
   与 Stage 2 基线同口径对比）。

---

# Round 2 增补（2026-10-23，用户决策"继续调优"后第二轮调优）

> R2 起点 1.690 ms（= R1 终态复现）→ **最终 1.504/1.511/1.541 ms（代表
> ~1.51，稳态 ~1.42）**，R2 内 **-10.7%**；相对原始基线 3.285 ms 累计
> **-54%**。目标 < 1320 us **仍未达标**（超 14%）→ [PERF_INCOMPLETE]。
> 完整迭代数据：fla_chunk_o_perf_log.md R2 章节 / fla_chunk_o_optimization_log_r2.md。

## R2 有效优化（同会话 A/B 消融，逐项实测）

| # | 手段 | 前 → 后 (ms) | 贡献 | 采纳 |
|---|------|-------------|------|-----|
| 1 | C 侧 fixpipe 直写 strided O（o_relay + V drain 全链消灭） | 1.690 → 1.652 | -2.2%（结构价值为主） | ✅ |
| 2 | Select-free 主循环 + Select 尾 | → 1.670 | 0 | ✅ |
| 3 | 2-chunk 批处理（staging 双份） | → 1.675 | 0 | ✅ |
| 4 | L1 操作数按名 ping-pong + B 装载上移 | → 1.667 | 0 | ✅ |
| 5 | **v 中继经 V 侧**（strided v 读移至 AIV 余量，AIC 全连续读）+ V 链就地压缩 | → 1.563 | **-6.2%** | ✅ |
| 6 | O 写回迁 V（o_relay + V 排水） | → 1.759 | **+11% 劣化** | ❌ 回滚 |
| 7 | o_frag 按块奇偶 ping-pong | → 1.568 | 0 | ✅ |
| 8 | 节流加深至 2 批次 + NSLOT=8 + peel-batch | → **1.530** | **-2.4%（稳态 -3.9%）** | ✅ |

## R2 关键实证（工具链事实修正与新增）

1. **strided GM 读比连续读慢 ~3×**（AIC MTE2：16 vs 48GB/s）——v 的
   64×256B 行块装载占 AIC 关键路径 ~0.44us/任务（探针实测 -180us）。
   中继连续化（经 V）净赚 104us。
2. **跨核 flag 协议 ~470ns/次**（V 预置后仍存在）：4 relay/任务 ≈
   名义 1.9us/任务，大部分与 MTE2/gemm 重叠；1 批滞后的节流曾使
   hi/lo 中继"恰好赶上"C（491us 等待），2 批滞后修复。
3. **fla_wy_fast 的"L0C→GM(bf16) fixpipe 直存损坏"记录不在本路径复现**
   （copy_l0c_to_gm + realDstN=stride 实测精确，ratio=1.0000）。
4. **R1 的"直写 O vs 中继中性"在 V 有负载时反转为"直写严格优"**。
5. **R1 ping-pong 失败根因定位**：AUTO_SYNC 的 WAR 分析按 buffer_name
   （HasDataDependency, ascend_sync_insert.cc）——同名共享缓冲强制
   M_MTE2 串行；按名 ping-pong（_a/_b）解除。但解除后暴露 MTE2 管线
   本身才是墙（72% busy），故持平。
6. **就地 tile.op（dst==src）逐位精确**（文档已明示可别名）。

## R2 后剩余瓶颈（最终归因，msprof_output_r2c）

1. **AIC MTE2 88KB/任务 @48GB/s = 1.82us/任务（50% busy）**，其中 56KB
   为 CV 中继（hi/lo 32KB 精度强制、a_mask 8KB、v 16KB）。
2. **中继合并被 L1 zN 分形布局阻断**：单 flag 单读需 region gemm 操作数，
   zN 16 列带布局使 region 逻辑偏移 ≠ 物理偏移（列/行堆叠均断裂）。
3. **跨核 flag 协议开销**（~470ns × 4/任务，部分隐藏）。
4. **AUTO_SYNC 相邻 set/wait 标记对**阻止 MTE2 队列前瞻（FIFO 饥饿）。
5. HBM 47-59% 利用（非墙）；IO 下限 797us < 目标隐含值——差距是
   工具链/结构性的，非物理不可达。

## 对上游/后续的建议（R2 更新）

1. 工具链：zN 对齐的 L1 region 操作数（或 per-buffer L1 布局控制）→
   可合并 4 relay 为 1（估 -0.5~1.5us/任务）；跨核 flag 批量化/廉价化；
   AUTO_SYNC marker 拆分（set 先发、wait 后置）解锁 MTE2 前瞻。
2. 算子层：若精度政策允许（单 cast qg 的 0.988 vs 0.99 门限），hi/lo
   单中继可再省 16KB/任务中继字节（当前被精度门禁强制）。
3. 验收口径：官方均值含首 rep 冷启动（~+0.08-0.09ms）；稳态 ~1.42。

---

# Round 3 增补（2026-10-23，画像驱动 + 地板核算，用户决策第三轮调优）

> R3 起点 1.513 ms（R2 终态复现）→ **最终 1.486/1.510/1.497 ms（代表 ~1.498，
> 稳态 ~1.40）**，R3 内 **-15us（-1.0%）**；相对原始基线 3.285 ms 累计 **-54.4%**。
> 目标 < 1320 us **仍未达成**（超 13.5%）→ [PERF_INCOMPLETE]。
> 本轮方法：生成代码审查 + 画像 + **消融图谱**（逐成分移除测真实贡献）。

## R3 有效变更（同会话 A/B，逐项实测）

| # | 手段 | 前 → 后 (ms) | 贡献 | 采纳 |
|---|------|-------------|------|-----|
| 1 | **跨核 flag ID 冲突修复（12→40→48）**：auto pass 的 flag ID 从 0 逐站点分配（0-31/0-47），与手动节流 ID 12 冲突（每 ID 单计数器、双向共享）→ 节流失效 + 潜在竞态 | 1.513 → 1.501 | -0.8% | ✅（**正确性修复**） |
| 2 | V 体重构：v-relay 上移体首 + prefetch 拆分（中体/体尾） + v 早装载 | 1.501 → 1.497 | ~0 | ✅（交付时间严格不劣化） |
| 3 | **3-chunk 任务批处理**（staging×3、L1×3、NSLOT=9、peel=2 整批） | 1.497 → **1.484**（稳态 1.417→1.40） | -0.9%（稳态 -1.3%，分布不重叠） | ✅ |
| 4 | C 侧全批装载上移 | → 1.478 | 0 | ❌ 弃用 |

## R3 关键实证（新增工具链事实）

1. **auto flag ID 与用户 ID 冲突是实际缺陷**（ascend_combinecv.cc L373-400，
   global_sync_flag_id 从 0 递增、不感知用户 ID；官方文档确认跨核 flag 每 ID
   单计数器）。修复后 relay 等待 780→570us（多与 MTE2 重叠，性能净变化小）。
2. **跨核 flag 计数器容量实测 < ~10 pending**（8 裸 set + 2 peel set 死锁；
   文档标 15，2-AIV 计数更紧）——**跨核流水深度硬上限**，深度节流不可达。
3. **V 是生产受限**（6 批节流信用无变化）——节流深度/lag 无关紧要。
4. **消融图谱：每任务计算几乎免费**——V 全部数学 0 / V 装载 0 / V 拷贝 ~0 /
   C 全部 gemm -30us / C h-装载 -13us / C FIX -19us / 单 relay 往返 -33us。
   **墙 = 每任务 ~3.4us 管线协议地板**（4 relay 往返 × ~0.1us + 每站点同步
   指令流），relay 数量下限 4/任务被 L1 zN 分形布局锁死（hi/lo 合并/K-concat
   单 gemm 被 region 物理偏移阻断，R2 结论重新推导确认）。
5. 流程教训：快速精度门 shape（l0_repr）N_MAIN=0 **不覆盖主循环**（本轮曾因
   快照事故漏检，已修复）；主循环覆盖 shape（如 (1,2048,16,128,128)）已纳入
   常规检查；消融变体必须独立文件名。

## 地板核算表（R3 终态，官方均值口径分解）

| 项 | 数值 | 说明 |
|---|---|---|
| kernel 稳态（9 rep 均值） | **~1.40 ms** | 实测（1.386-1.444 波动；消融证明其中计算/装载/拷贝 < 80us，其余为 4 relay 往返/任务 + 每站点同步协议的管线地板） |
| host 分发（每次调用） | **~0.999 ms** | 编排层实测（~0.6ms TVM Arith Analyzer 重建 + ~0.4ms Cython forward 14 张量签名适配；torch.empty 仅 0.065ms）；框架层，算子 PR 不可修 |
| 首 rep 暴露 | +0.85~1.06 ms | 首 rep 2.24-2.46 - 稳态 ≈ 分发在设备空闲时全暴露 |
| **合成预估** = 稳态 + 分发/10 | 1.40 + 0.10 = **1.50 ms** | **与实测官方均值 1.486-1.510 吻合 ✓** |
| 官方均值框架地板（若 kernel < 分发） | ≈ max(kernel, 0.999) + 0.10 ≈ **1.10 ms** | kernel 压到 1.0ms 以下时分发逐 rep 暴露——框架地板仍低于 1.32 目标，**目标未被框架封锁** |
| 达标所需 kernel 稳态 | **≤ ~1.22 ms** | 1.32 - 0.10；当前 1.40 → 需再 -180us（-13%） |

## R3 后剩余差距归因（kernel 侧 -180us 的构成分析）

1. **relay 往返协议（最大可寻址池）**：4 往返/任务 × ~0.1us ≈ 165us 总量。
   合并到 3 往返（hi/lo 单 relay + K-concat 单 gemm）估 -40~-70us——**被 zN
   布局阻断**（工具链）；这是"kernel 可优化"类中最大单项。
2. **每站点同步指令流**：每任务 ~15-20 个同步站点（marker 对/屏障/flag）×
   每站点 ~0.1us —— 需 AUTO_SYNC 更粗粒度同步或 Expert 模式全手动（R1 估
   ~0.6-0.75ms 纯开销的组成部分）——工具链/重写级。
3. **每批固定开销**：3-chunk 后残余 ~0.1us/任务 —— 4-chunk 边际（UB 预算
   临界 + 收益递减，未试）。
4. **host 分发地板 0.10ms（均值口径贡献）**：框架层（Arith Analyzer 重建），
   上游反馈项——"框架/工具链配合"类。

**结论**：目标 1320us 需要 kernel 稳态 ≤1.22ms；剩余 -180us 中约 -40~-70us
（relay 合并）被 zN 布局阻断、其余主体为同步协议地板（AUTO_SYNC 每 op 开销）
——属"框架/工具链配合"类差距；纯算子侧手段（批处理/装载重排/节流）已在
R1-R3 穷尽（本轮消融图谱证明其边际趋零）。

---

# Round 4（R4）增补 — host 分发绕过 + 结构性上限实证（2026-10-24）

> 用户 R3 检查点决策"继续调优"。R4 起点 **1.492 ms**（官方均值；稳态 ~1.411；
> 首 rep 暴露 ~0.83ms）。主攻：host 分发绕过（est -80~90us）+ 4-chunk 批处理
> （est -10~20us）+ 终态画像。**终态 1.417 ms（-75us，-5.0%）**，累计
> 3.285 → 1.417（**-56.9%**）。目标 1320us 仍未达成（超 7.3%）→
> [PERF_INCOMPLETE]，但本轮把"不可达"的论证从推断升级为**实测封闭**：
> **设备侧单_launch 1391.1us（msprof）> 1320us——即使 host 分发为零成本，
> 官方均值也不可能低于设备稳态，目标在本同步协议架构下不可达。**

## R4 有效变更（同会话 A/B，逐项实测）

| # | 手段 | 前 → 后 (ms) | 贡献 | 采纳 |
|---|------|-------------|------|-----|
| 1 | **host 分发绕过 _FastLaunch**（缓存编译实例 + workspace 复用 + lib.call ABI 直呼，护栏+自检+回退） | 1.492 → **1.416** | **-76us（-5.1%）**；分发 795.7→163.4us | ✅ |
| 2 | 4-chunk 批处理 | 编译期段错误 | 0（DK=DV=128） | ❌ UB 硬上限（见下） |
| 3 | 终态 msprof 画像 | — | 无新可寻址项；确认协议地板 | 记录 |

## R4 结构性实证（新工具链事实）

1. **公共 JIT 分发对全静态 kernel 的纯浪费构成**（fla_chunk_o_dispatch_
   decompose_r4.py 实测，每次调用）：Analyzer 重建 289.4us（仅符号 shape
   需要，本 kernel 无）+ cython 逐参 shape 重解析/校验 ~292us + 5×
   torch.empty 64.8us + stream 查询 35.1us = **795.7us**；其中真正必要的是
   JIT 缓存查找 2.4us + lib.call 入队 9.1us。上游 ascendc_pto（43b0b697）
   对 tilelang/jit/ 零改动，无现成快路径。
2. **tilelang A2/A3 UB 硬上限 = 196352B**（memory_planning.cc:35 /
   codegen_ascend.cc:40），且溢出表现为**段错误**（TVM OptimizeForTarget
   内 C 层 CHECK 穿 FFI，faulthandler 实证）而非干净报错。3-chunk kernel
   的 ascend_ub 恰好 100.0% 打满该值 → 4-chunk staging（+36864B/vid）不可
   行；staging 峰值期全部活跃不可释放，Vector 缓冲不可迁 L1（内存层级）。
   **4-chunk 及更深批处理对本算子族在此工具链下结构性封死**（A5 的 262144B
   预算是另一回事）。
3. **跨核 flag ID 空间实测 ≥65 个不同 ID**：3-chunk kernel 上 THROTTLE
   48→64 探针行为与 48 完全一致（bench 1.420/精度 PASS）→ 无 6-bit 回绕，
   ID 0-64 互不混叠（CANN 文档 ISASI 标 0-10，显著保守）。R3-1 的 ID 冲突
   修复在 4-chunk 场景依然有效（THROTTLE=64 > auto 0-63）。
4. **设备侧协议地板复测**：msprof 单_launch 1391.1us；AIC cube 12.7%/
   mte2 38%/scalar 24.4%（relay 等待 ~733us 与 MTE2 重叠）；AIV throttle
   等待 1114.6us（lag-2 设计使然）/vec 20.5%。双侧占满墙时间，无空闲
   bubble——R3 消融图谱结论（每任务 ~3.4us 协议地板 = 4 relay 往返 + 每
   站点同步指令流）在终态复现。

## 地板核算表（R4 终态更新，官方均值口径分解）

| 项 | R3 终态 | **R4 终态** | 说明 |
|---|---|---|---|
| kernel 稳态（设备侧） | ~1.40 ms | **~1.406 ms**（bench 9-rep 均值；msprof 单_launch 1391.1us） | R3 已证协议地板：计算/装载/拷贝合计 <80us，其余为 4 relay 往返/任务 + 每站点同步指令流 |
| host 分发（每次调用） | ~1.00 ms（公共路径） | **公共 0.80 / 快路径 0.163 ms**（R4 实测 795.7/163.4us；首 rep 暴露 0.83→0.13ms） | 快路径构成：launch 本体 51.3 + stream 37.6 + 输出 empty 13 + 杂项 |
| 首 rep 暴露 | +0.83~1.06 ms | **+0.11~0.17 ms**（首 rep 1.513-1.566） | |
| **合成预估** = 稳态 + 暴露/10 | 1.50（实测 1.492 ✓） | **1.406 + 0.014 ≈ 1.42**（实测 1.414-1.421 ✓） | 模型持续吻合 |
| 公共路径合成（交付文件关闭旁路） | — | 1.406 + 0.08 ≈ **1.49**（R4 基线实测 1.492 ✓） | 两口径均可由同一模型解释 |
| **零 host 成本下限** | — | **= 设备稳态 ~1.391-1.406 ms > 1.320 目标** | **目标在当前同步协议架构下不可达**（超出 71-86us，全部在设备侧） |
| 达标所需 | kernel ≤1.22 ms | **kernel ≤1.32 - 0.014 ≈ 1.31 ms**（快路径口径） | 需设备侧再 -90us：relay 合并（-40~-70us）被 zN 布局阻断 + 同步指令流地板——工具链级 |

## host 分发绕过的 PR 可接受性结论（两口径如实报告）

**结论：可交付（含旁路），理由**：
1. 使用的均为非下划线框架属性（kernel.adapter / adapter.params /
   adapter.lib）+ 生成库自身的 extern "C" call ABI（cython 适配器调用的同
   一入口）——in-repo 先例：examples/torch_tl_ascend/compile_tl_op/
   flash_attention.py 直接访问 kernel.adapter.lib_generator。
2. 三重护栏保证最坏退化不劣化：(a) 每调用全量输入校验；(b) arm 时 bitwise
   自检（公共结果 == 两次快路径结果）；(c) 任何异常（含上游重构）回退公共
   路径。自检失败则该键永久 poison，只损失性能不损失正确性。
3. 输出张量语义与上游 wrapper 一致（每次 fresh 分配，不别名）；workspace
   复用仅限内部暂存（同流串行安全，stream/device 入键隔离）。
**若评审仍不接受**：删除 _FastLaunch 块（约 120 行，与公共路径干净隔离）
即回退公共路径，交付口径数字为 **1.492 ms**（R4 基线实测）；框架层浪费
（per-call Analyzer 重建 ~0.6-0.8ms）作为上游反馈项记录（R4 工具链发现
#1，perf_log R4 章节）。

## R4 后状态与移交

- 纯算子侧（批处理/装载重排/节流/协议）：R1-R4 穷尽且 R4 补充了结构性
  封死证据（UB 上限 → 4-chunk 不可行；flag 空间探针 → 无 ID 瓶颈；
  msprof → 无新 stall）。
- 剩余差距 -97us（1.417 → 1.320）全部在设备侧同步协议（relay 往返被 zN
  布局阻断合并 + AUTO_SYNC 每 op 同步指令流）——"框架/工具链配合"类。
- host 侧残余（~163us 快路径，其中 stream 查询 37.6us 可经 torch_npu 私有
  raw getter 降至 0.8us，未用）——对均值贡献已 <5us，忽略。

# Round 5 增补（同步协议架构轮，2026-10-24，用户决策"继续第五轮调优"）

## R5 任务与结论一句话

主攻同步协议架构（pass_configs 旋钮清点 / 站点削减 / 手动 flag 协议 /
_FastLaunch 微优化），**主线 1-4 全部穷尽，零采纳，终态 = R4 终态
（1.417ms，md5 一致）**；本轮以 6 组真实 A/B + 源码级证据封闭了 Developer
模式内最后一批性能假设，剩余差距（~100us）判定为 AUTO_SYNC 协议架构地板
→ **[EXPERT_MODE_REQUIRED]**（需用户授权整体手动同步重写才能继续）。

## R5 各手段及贡献（μs，同会话 A/B 实测）

| 手段 | 结果 | 实测 |
|---|---|---|
| --cce-auto-sync=off（编译器同步层关闭） | 无效 | 1.426 vs 基线 1.421（噪声内） |
| --cce-remove-auto-sync=true | 无效 | 1.426 |
| -O3 | 无效 | 1.435 |
| relay 单 batch flag（12 检查→1，改名逃逸 AUTO_CV_SYNC） | **+36% 劣化** | 1.934；设备侧反而 1346.8（-87）但 cube_wait 98%——批粒度锁步 |
| hi/lo K=256 融合 + 9 站点手动 flag | 分析性放弃 | H 堆叠 WAW barrier +250us 量级 > 收益 ~45us（源码级推导，未实施） |
| _FastLaunch stream 查询私有 API 快路径 | 无墙钟收益 | 1.430/1.417（host 163→127us 隐藏于设备后）；按规则回滚 |

## pass_configs / 同步旋钮全量清点结论（主线 1）

- **TileLang TVM 层 28 键**：唯二同步相关增量发现——TL_ASCEND_AUTO_SYNC_VS
  是主 pass 的**补充**（V→V、S↔Others 补插，只加不减，与我们方向相反）；
  其余键（tail mask / vector mask reuse / PTO pipe copy 等）对本 kernel
  无同步语义。
- **Bisheng 编译器层**：--cce-auto-sync（ascendc 默认 on）/
  --cce-remove-auto-sync / -O3 三旋钮 A/B 全部无效——**编译器层不是第二
  同步层，417 站点 100% 来自 AscendSyncInsert 显式调用**。
  --cce-auto-sync-log 旗标本身触发编译器崩溃（上游反馈项）。

## R5 核心实证（否定性，封闭剩余假设空间）

1. **flag 检查开销理论否定**：R4 msprof 的"AIC relay 等待 733us"=
   12×459ns×133 的预置检查流，但 iter2 快发射 launch（flag 预置时）仅
   -20us → **墙临界份额 ~3%**。"批量 flag/廉价协议"类优化收益上限实测
   封闭。
2. **渐进触发是承重结构**：单 batch flag 把 C 的 12 次装载绑定到 V 的
   MTE3 全批排空后 → 批内 chunk 级 C/V 交错（C 消费 chunk-c 重叠 V 生产
   chunk-c+1）被摧毁，+36%。任何站点/flag 粒度粗化不可行。
3. **融合装载的 WAW 隐性成本**：同 buffer 名静态切片双写（[H;H] 堆叠）
   触发名字级 WAW PipeBarrier<MTE2>（无切片重叠分析）→ hi/lo K=256 融合
   净负。
4. **V 侧大空闲再确认**：AIV throttle 等待 1114.6us（R4）+ 本轮 iter2
   设备侧 1346.8 快于基线——V 是被节流方，任何 V 侧优化不进墙。
5. **C 侧工作几乎免费**（R3 消融 + 本轮复核）：gemm 30us + H 装载 13us +
   FIX 19us + 单 relay 往返 33us ≈ 95us；墙的 ~93% 是协议结构本身。

## 地板核算表更新（R5 终态 = R4 终态，分解细化为三层）

| 层 | 构成 | 量级 | 可移除性（R5 实证） |
|---|---|---|---|
| C/V 实际工作 | gemm+装载+FIX+relay 数据搬运 | ~95us | 已最小（R1-R4 优化后） |
| relay 站点协议 | 12 站点/批 × ~11us（含渐进触发结构） | ~132us | **不可粗化**（iter2 +36%）；不可合并（zN region 禁区 + WAW） |
| 每 op 同步指令流 | AscendSyncInsert 名字级依赖屏障 + 边界配对 + flag 检查 | ~1.2ms | **Developer 模式内不可关闭**（编译器层无效；V 侧屏障无墙收益；C 侧屏障 pass 强制） |
| host/launch | 快路径 163us（隐藏）+ 首 rep 暴露 ~0.13ms | 均值 <15us | stream 快路径已证无墙收益 |

**达标所需**：设备侧再 -100us，全部落在第三层 → 只能整体替换同步协议
（TL_ASCEND_AUTO_SYNC=False + 全手动 barrier/flag），即 Expert 级同步重写。

## [EXPERT_MODE_REQUIRED] 论证（模式红线，请示用户授权）

- **判定**：Developer 模式（AUTO_SYNC pass 按名依赖分析 + 每 op 屏障）
  内的全部性能杠杆已在 R1-R5 穷尽：R1-R4 优化 3.285→1.417ms（-56.9%），
  R5 以 6 组 A/B + 源码证据封闭最后假设（编译器层无效 / flag 合并劣化 /
  融合净负 / V 侧无墙 / host 隐藏）。
- **Expert 路线预估**：TL_ASCEND_AUTO_SYNC=False + 手动同步重写——C 侧
  每批 ~45 条同步指令（4 组 M_FIX/FIX_M/MTE2_M 配对 + 7 M-barrier +
  12 flag wait）可压至 ~10-15 条（消同名 WAW 伪屏障：gemm 间 M 屏障×6、
  H 堆叠 WAW、循环顶 WAR 配对合并）；V 侧 ~96 条压至 ~15（48 个 V 屏障
  是名字级伪依赖，V 管线硬件按序）。**若协议成本随指令数近线性（消融
  旁证：单 relay 往返 11us、C 全 gemm 仅 30us），设备侧理论空间
  -300~-600us → 800-1100us 区间，可越 1320 目标**。
- **风险（如实）**：R2-R5 手动协议 4 次尝试 3 次失败（2-late drain 死锁、
  深节流计数器死锁、单 flag 锁步劣化）——全手动 hazard 分析正确性风险
  高；预估区间不确定性大（协议成本未必线性）；重写量级 = kernel 主体
  （~700 行）。
- **替代路线（不需授权）**：上游工具链两项——zN 对齐 region gemm 操作数
  （解封 relay 合并 -40~-70us）与跨核批量 flag 原语（解封检查开销）；
  均已记录于 R4/R5 工具链发现。

## R5 产出文件

| 文件 | 内容 |
|---|---|
| fla_chunk_o_optimization_log_r5.md | R5 完整优化记录（清点/A/B/源码证据） |
| r5_final_precision.log | 终态全量精度（exit=0） |
| msprof_output_r5_base/ | 基线（=终态）设备侧画像 1434.2us |
| msprof_output_r5_iter2/ | 单 flag 变体画像（cube_wait 98.1% 证据） |
| history_version/fla_chunk_o_perf_r5_*.py | 起点/iter2 前/iter3 前/iter4 快照 |
