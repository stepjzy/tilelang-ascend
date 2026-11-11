# fla_chunk_delta Stage-3 Performance Tuning — Final Summary (Round 2)

**日期**: 2026-11-11 | **执行**: tilelang-op-perf-tuner (Stage 3, round-2 深挖) | **状态**: 本轮 kernel 层手段穷尽（证据链完备），未达 3400us 目标，交回 orchestrator 裁定

> Round-1 结果（2026-11-11 早）：16180.5 → 7095.9us（-56.1%）。Round-2 由用户否决
> "kernel 层穷尽" 结论后启动，本轮全部优化以流水图/生成源码证据立项（无盲试）。

## 1. 结果总览

| 指标 | Round-2 入口 | Round-2 终态 | 本轮变化 | Stage-2 累计 | 目标 |
|------|-------------|-------------|---------|-------------|------|
| fwd bench (us) | 2538.3 | 2551.3 / 2549.1（双复测） | ~持平（fwd 代码仅 iter8 后未变；跨会话散布 ±3%） | **-62.4%** | — |
| bwd bench (us) | 4559.8 | 3518.6 / 3517.6 | **-22.8%** | **-61.6%** | — |
| **合计 (us)** | **7098.1** | **6069.9 / 6066.7（均值 6068，跨会话范围 6067-6284）** | **-14.5%** | **-62.5%** | **< 3400** |
| 精度 | all pass | **all pass（--level all --bench exit 0；L0 6/6 + L1 10/10 + L2 10/10 + Boundary 5/5；max_abs 与 Stage-2 记录逐位一致）** | — | pass | pass |

bench 口径：test --bench 哨兵（do_bench warmup=20 rep=10 mean，设备侧 NPU events），
B=1,S=8192,H=64,DK=128,DV=128,cs=64，fwd fp16 / bwd bf16，wrapper 默认分支（含 prep launch）。
终态 kernel sha256: e5c6f1540f08cf9c61dc80df8f1a87e69557661c6e98de043d13925adcc48322。

## 2. 本轮（Iteration 8-13）每轮手段与贡献

| 轮 | 手段 | fwd/bwd/合计 (us) | 采纳 |
|----|------|------------------|------|
| 8 | 测量口径修正（bench_entry fwd-main/dump-src BV=64 硬编码→_pick_bv；真终态 fwd BV=128 源码证据） | 2538.3 / 4559.8 / 7098.1（基线复测） | 基线固化 |
| 9 | **bwd Select 风暴消除 + 单链尾批**（NFULL=NCHAIN//(2·CE) 满批可证在界→纯算术索引，condval if/else 内联风暴 68+24/迭代清零；余数 R 改受卫单链尾批，12 核跳过整块而非 redirect 复制链） | 2538.7 / 3929.4 / 6468.1（bwd **-13.8%**） | ✓ |
| 10 | **bwd q~ 外移到 prep**（QT=bf16(fp32(q)·e^g)，位等价算术链；主核删 13 V 语句/链·chunk + 2 次 ws3/ws3b 中继握手；C 侧一条 [BS,DK] QT 直读替代两条 ws3 readback） | 2649.1 / 3635.1 / 6284.2（该会话整体偏慢 ~3%；bwd 同会话 **-7.5%**） | ✓ |
| 11 | ws5/ws6 L0C 累加合并 + fwd 2-chain BV=128（CE=16 无尾批，往返减半） | 分析否决 / 运行崩溃（aicore 507015） | ✗（证据见 §4） |
| 12 | 前导 MTE2 装载前置 + row_expand scratch 按半/按链拆名 | 2627.2 / 3593.3 / 6220.5（-1.0%，噪声内） | ✗ 回滚（诊断价值见 §5） |
| 13 | 终态验证（全量门禁 + bench×2 + 双口径 msprof + 源码 census） | 2551.3 / 3518.6 / **6069.9**（复测 6066.7） | 终态 |

## 3. 本轮采纳清单（含证据）

1. **bwd select-storm 消除 + 单链尾批**（iter9）：round-1 终态源码证据——每条 tl-op 前内联
   2-8 个 `condval` if/else（AIV 68 + AIC 24 块/迭代，Select-redirect 的索引表达式逐语句
   逐迭代重算），且尾批让 12 核跑 redirect 复制链。满批在界性数学证明
   （max tid = 2·NFULL·CE − 1 ≤ NCHAIN − 1）；生成 C 验证 condval=0，尾批折叠为
   `if (cid < 8)`。**bwd -630.4us（-13.8%）**。
2. **bwd q~ 外移**（iter10）：V 侧 13 语句/链·chunk 的 q~ 路径（q_half 双读、双 cast、
   双 row_expand、双 ws3/ws3b 中继、GE 读）整体移入 threads=1 prep（每 (b,chunk) 每 head
   8 语句，gate 用已在 UB 的 gt_t 行——数值位等价）；主核 Q3→QT、GE/ws3/ws3b 槽位删除。
   **bwd 同会话 -7.5%**（跨会话 3635/3518 口径）。

## 4. 本轮被否清单（证据）

| 方案 | 结果 | 证据 |
|------|------|------|
| qdo_scale 折入 QT gate | bwd 3538.7（-9.9%）但 **l0_bwd_opts dh0 ratio=0.9601 FAIL** | golden 在**未缩放**点量化 `qt=bf16(q·e^g)`（test golden 原文）；折叠使 bf16 舍入点移动，bwd 递推反馈（dH→dv2→wdv2→dH）放大：dh0 max_abs 2.384e-07→2.284e-03、dh 3.906e-03→1.562e-02。回退后 L0 max_abs 逐位复原 |
| ws5/ws6 L0C 累加合并（一读回一代两） | 分析否决 | dH = D·dH + scale·qdo − wdv2 的合并要求 scale·qdo 在 L0C 内——即把 scale 折入 bf16 q~ 操作数（上一行实测 FAIL）；无 scale-free 形式（scale(qdo+wdv2) 需 wdv2 单独读回；1/scale 折 dv2 引入非 2 幂不精确乘法） |
| fwd 2-chain BV=128（CE=16，零尾批、往返减半） | **aicore exception 507015**（l0_fwd_repr 起全部 2-chain 用例） | 生成 C（…_r11.c）：2-chain fwd UB 需求 ~284KB/AIV > 196KB → memory planner 启用 lifetime 重叠分配（h16_ub_b 与 vn_ub_b **同址 90496**，kh_ub/vn_ub_b/h16_ub_b 相互重叠）→ **AscendSyncInsert 对物理重叠的不同名跨 pipe WAR 漏插保护**（Sub 写 vn_ub_b 时 prologue 的 MTE3 中继仍在读同字节）→ 竞态崩溃。共享 L0C frag（96KB）不救——错在 UB 重叠；仅同名共享仍需 ~203KB > 196KB。k_g 外移可省 gate 路径装下 UB，但 fwd golden 在 cast point 3 量化 `vn16=(vn·gate).to(fp16)`——同类 golden 耦合（已实测会 FAIL） |
| 前导装载前置 + scratch 拆名 | **-1.0%（噪声内），回滚** | 有诊断价值：bwd PIPE_V 25→14/迭代（拆名机制按推导生效）但 wall 中性 → **barrier drain 成本被实测证伪**（round-1 的 ~0.1us/stmt 是含 issue+stall 的上界估计） |
| 并行 scan 算法重排（H(t)=A(t)H(t-1)+b(t) 的 Blelloch） | 分析否决 | 组合矩阵 [128,128] 每 chain 127+ 次 compose + 下扫——gemm/语句量 ×3-4 于本已 issue+latency-bound 的结构（cube 仅 6.6-8.4% 利用率但 wall 由往返决定）；chunk_o 已验证最优 1.15us/gemm 不可达 |

## 5. 穷尽判定证据（终态 msprof r13，均已取回 CPU）

- **fwd main 2573.2us（真 BV=128 口径，bench_entry 已修正）**：PipeUtilization——
  AIC wait_id0（等 V 的 ws1）**51%** + wait_id2（ws3）**45%**；AIV wait_id1（ws2）**49%**
  + wait_id3（ws4）**47%**——**双侧各花一半 wall 互等对方**（串行 H 递推 ping-pong）；
  fixpipe 16-19%；cube **6.6%**；GM 带宽 5.8-8.7%。
- **bwd main 3044.7us**：AIC wait_id0（ws1）**41%** + fixpipe 30% + 分散 readback 等待；
  cube 8.4%；GM 8.5-10.7%。AIV wait_id3（ws2）29% 等。
- **iter12 诊断（barrier 证伪）**：PIPE_V barrier 25→14/迭代 wall 不变——剩余成本不是
  barrier drain，是语句发射 + 跨核往返本身。
- **生成 C barrier 计数**：fwd 11 PIPE_V/chunk（…_r13.c）；bwd 25/迭代（iter12 拆名可
  降至 14，中性）。

## 6. 地板核算（终态口径）

- fwd main ≈ 512 chunks × 2 次串行跨核往返 × ~2.4us ≈ **2.5ms**（实测 2.57——已在地板上；
  AIC 双等待和 2515us/2573 = 98% 可解释为两段等待+间隙发射）
- bwd main ≈ 512 iters × ~2.5 次往返 × ~2.2us + fixpipe 30% ≈ **2.9ms**（实测 3.04）
- prep 0.38 + wrapper（do_bench 事件+输出分配+launch）~0.15 → **合计 ~6.0ms ≈ 实测 6068 ✓**
- 达到 3400 需 fwd/bwd main 各 ~1.7ms：即**同时**减半往返数（fwd 2-chain：UB 不可行，
  实测崩溃证据；scan：语句量爆炸，分析证据）与往返延迟（操作数折叠合并中继：
  golden 量化点耦合，实测 FAIL 证据）——均不在 kernel 层可达范围。
- 物理口径：带宽利用率 ~10%（6068 vs 带宽地板 1.61ms）；gemm 利用率 ~7-8%；
  chunk_o 已验证最优 1.15us/gemm vs 本算子 fwd 2.5 / bwd 1.85——差距即串行递推的结构税。

**结论：round-2 以证据驱动又压缩 14.5%（累计 -62.5%），并证伪了第一轮遗留的两大假设
（barrier drain 成本、3318.8us 口径）。当前 1.78× 差距的约束 = 串行递推跨核往返 +
golden 量化点耦合 + UB/同步工具链盲区，均属 kernel 层之外。**

## 7. 产物清单（均已取回 CPU，examples_experiment/dsa_sparse_finetune/perf_tuning/）

| 产物 | 路径 |
|------|------|
| R8 真终态 fwd BV=128 源码（第一轮口径缺陷的修正证据） | fla_chunk_delta_fwd_kernel_source_final_bv128.c |
| R9/R11/R12/R13 生成源码（condval 归零 / 崩溃现场 / 拆名 census / 终态） | fla_chunk_delta_{fwd,bwd}_kernel_source_r{9,11,12,13}.c |
| 终态画像（op fwd/bwd main 真 BV 口径 + bwd wrapper(prep) + app 级 PROF） | fla_chunk_delta_prof_r13/（OPPROF×3 + PROF app） |
| 每轮基线/终态 json | fla_chunk_delta_baseline_iter{8,13_terminal}.json |
| 迭代日志（Iteration 8-13） | fla_chunk_delta_perf_log.md |
| bench/msprof 入口（BV 口径已修正） | fla_chunk_delta_bench_entry.py |
| 采集脚本 | fla_chunk_delta_profile_r13.sh |
| 历史快照 | ../history_version/fla_chunk_delta_perf2_r{8_base,9,10,12_probe,13_terminal}.py |

注：终态 main-only 口径（fwd 2573.2 / bwd 3044.7 / prep 382.6）与 bench 哨兵
（2551.3 / 3518.6 / 6069.9）自洽（main+prep+wrapper ≈ bench）。
