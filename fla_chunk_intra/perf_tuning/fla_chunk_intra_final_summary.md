# fla_chunk_intra Stage 3 终态报告（R1 final summary）

> **【2026-11-06 重验证补录 / 2026-11-07 更新】** 本报告数字为旧环境（本地直连 910B3 主机，x86 host）实测。
> 分设架构（CPU 唯一真源 / NPU `DevEnvC_DCbyi_npu`：1×910B3 + 22 vCPU ARM host）重验证结论见
> `reval/reval_summary.md`：精度 90/90 复现；**纯 kernel 时间（msprof Task Duration）fwd 10395.1 +
> bwd 793.0 = 11188.1us**（在本文验收建议 ≈11.5ms 带内，设备侧无退化）。
> 【11-06 记录】新环境旧基线（463c234c，无 #1948）do_bench 12105.3us——bwd 链曾被新环境 host
> 分发地板（1450us/调用 > 设备 793us）逐 rep 暴露。
> 【11-07 更新】**该 host 分发地板已被基线更新消除**（4805777d 含上游 #1948 懒构造 TVM Analyzer）：
> host 地板 1450→700us，bwd do_bench 1676→1055.7us，合计 12105→**11343.0us**，与旧环境 11442.8
> 持平（-0.9%）；PR 分支已同步同基线（551b85cd）。详见 `reval/reval2_summary.md`。
> **PR 性能主口径经用户决策（2026-11-06）为纯 kernel 时间**，do_bench 保留为上游测试样例
> 对齐口径的补充披露。以下原文未改动。

日期：2026-11-03。终态实现快照：`history_version/fla_chunk_intra_perf_r1_final.py`。
迭代日志：`fla_chunk_intra_perf_log.md`；优化点清单：`fla_chunk_intra_optimization_log.md`；
起点基线：`fla_chunk_intra_baseline.md` + `fla_chunk_intra_baseline_r1_iter0.json`。

## 1. 终态实测（官方口径：do_bench NPU 事件，warmup=20 / rep=10 / mean，仅设备侧，3 次独立运行）

fwd shape (1,8192,64,128) bf16 in / fp32 gate；bwd shape (1,8192,8,128) fp32。

| kernel | run1 | run2 | run3 | 代表值（均值） | 基线 | Δ |
|---|---|---|---|---|---|---|
| fwd k_inter | 3559.1 | 3568.6 | 3565.2 | 3564.3 | 5164.1 | **-31.0%** |
| fwd k_subst | 2704.0 | 2553.7 | 2711.9 | 2656.5 | （自 k_g 拆出）| — |
| fwd k_g | 1195.8 | 1208.1 | 1202.8 | 1202.2 | （自 k_g 拆出）| — |
| fwd k_g2 | 1084.3 | 1079.1 | 1087.4 | 1083.6 | 1576.2 | **-31.3%** |
| fwd k_y | 1233.7 | 1230.1 | 1242.5 | 1235.4 | 2404.8 | **-48.6%** |
| fwd k_ai | 1140.2 | 1130.4 | 1144.7 | 1138.4 | 1852.6 | **-38.6%** |
| **fwd chain（6 kernel）** | 10762.1 | 10621.9 | 10792.0 | **10725.3** | 16451.4 | **-34.8%** |
| bwd k_stage | 644.4 | 686.9 | 687.2 | 672.8 | 1195.1 | **-43.7%** |
| bwd k_gemm | 808.9 | 875.5 | 851.9 | 845.4 | 2088.2 | **-59.5%** |
| **bwd chain（2 kernel）** | 1286.9 | 1280.9 | 1338.8 | **1302.2** | 3290.5 | **-60.4%** |
| **fwd + bwd 合计** | 12049.0 | 11902.8 | 12130.8 | **12027.5** | 19741.9 | **-39.1%** |

终态工件（md5 2d0004e0，lint 后）复核运行：fwd chain 10604.0 / bwd chain 1278.2 / 合计 **11882.1us**
——4 次运行带 11882-12131，代表值 ~12.0ms 成立（-39.1% 维持）。

（k_g 基线 5672.3 拆为 k_subst+k_g 两kernel后合计 3858.7，**-32.0%**。）

## 2. 优化手段与各自贡献（A/B 全部同会话实测，3% 噪声阈值）

| # | 手段 | A/B 数据（us） | 采纳 | 精度 |
|---|---|---|---|---|
| 1 | fwd k_inter 锚点消去超级 GEMM：逐 s2 锚点在乘积内精确相消（qg_i·rhs_j = q_i k_j 2^{g_i−g_j}），9 组操作数/18 gemm → 3 组/6 gemm（A 侧 48 行——block(0,*) 乘积全弃）；β 行缩放移 V 尾段 fp32 精确乘 | 5136.5 → 3517.8（**-31.5%**）| ✅ | L0 全过 |
| 2 | bwd 中点锚点超级 GEMM：gn_s/gn'_s 锚点在 D_s/V_s 后置缩放内相消 → 每 matrix role 一组共享操作数，16 逻辑 gemm→4（48→12 gemm_v0/半），staging 210→128KB/cid，exp 元素 −84%，MdA/MdK 全量 [64,64]（KMAX 列窗冗余） | k_stage 1202→648（−46.1%）+ k_gemm 2108→820（−61.1%）= **−55.5%** | ✅ | L0 全过，bwd max_abs 普遍改善 |
| 3 | fwd 预分裂 ws 直读链：五个 ws 转 bf16 hi/lo 对（字节数不变），下游 C 段 GM→L1 直读消中继；I+ 折叠为 V 尾段 fp32 精确加法（Y=Aid+G2@Aid、Ai=Y+G@Y） | fwd chain 14854.8 → 12152.7（**-18.2%**；k_y −48.9/k_ai −37.6/k_g2 −33.3/k_g −7.2）| ✅ | L0 全过 |
| 4 | k_g 前代 ILP 交织（per-s 临时 + 单循环） | 5266.7 → 5397.4（**+2.5% 劣化**）| ❌ 回滚 | 数值逐位一致 |
| 5 | k_g 拆分：纯 V 前代 kernel（k_subst 直写 AidHi/Lo）+ 无 relay G gemm kernel（全直读，AIC 跨块自由流动） | 融合 5269.8 → 拆分 3764.4（**-28.5%**）| ✅ | 数值逐位一致 |
| 6 | k_g2 threads=2 试点 | 1083.3 → 1596.2（**+47% 劣化**）| ❌ 回滚 | 数值逐位一致 |

过程中两处工具链深层发现（实证，已入 debug_log 语义）：
- **0×NaN=NaN**（iter3）：ws 未写区域的垃圾 NaN 会穿过 gemm 操作数的精确零（Cube MAC），毒化
  全下游——k_inter 现显式零写 Akkbp rows 0-16，ws 全域确定性写出。
- **同 buffer 兼作 gemm A/B 操作数读出垃圾**（iter1 探针）与 **[64,64] transpose_A 可用**（iter2）：
  zN 布局按消费侧模板，一 buffer 一形态。

## 3. 终态精度（--level all，2 次全跑 + lint 后复验）

**90 PASS / 0 FAIL / 0 WARN**（L0 23 + L1 36 + L2 12 正确拒绝 + Boundary 19），exit 0。
相对起点**精度增强**：bnd_bwd_decay 由 WARN（ratio 0.5/NaN）转 **PASS**（中点锚点把 fp16 操作数
指数包络从 2^18.24 压到 ≤2^12.2，消除旧溢出）；bnd_fwd_decay max_abs 7.45e-9。
ruff check 全过（4 处死变量清除后复验逐位一致）。

## 4. 瓶颈归因终局（msprof_intra3_*，per-block 均值 us）

| kernel | C 占用 | C 生产性 | V (vec/mte/scalar) | 主导墙 |
|---|---|---|---|---|
| k_inter 3564 | 7.4 | 2.7 | 10.7 (3.4/2.4/3.8) | C 锁步（等 V 操作数 relay，wait_id6）+ V op 发射 |
| k_subst 2657 | — | — | 12.4 (7.8/2.0/3.4) | **串行前代链内 RAW（vector_stall 主导）= 语义固有**；ΣV/40AIV=2.54ms ≈ 实测 2.66-2.82 ✓ 贴地板 |
| k_g/g2/y/ai ~4660 | ~1.9 | ~1.0 | ~3.0 | C→V frag relay 依赖等待（wait_id0 2.7-4.3）+ 块粒度 |
| k_stage 673 | — | — | 7.5 (3.9/5.1/1.1) | V 串行链 + GM 写 |
| k_gemm 845 | 10.4 | ~10 | 18.0 (5.2/13.7/2.3) | V mte2（staging+Fragscr 读）= 传输墙 |

## 5. 地板核算与剩余差距分类

### 流量/算力地板（终态结构）
- **fwd**：GM 流量 ≈ 538KB/program × 8192 ≈ 4.4GB（IO 94 + ws r+w ~200 + relay ~244 per 块）
  → @1263GB/s（100%）≈ 3.5ms；@~600GB/s（实测有效）≈ 7.4ms。
  实测 fwd 10.73 = k_subst（V 算力地板 2.54ms，不与流量重叠）+ 其余 8.07ms（其余流量 7.3ms@600
  的 1.10×——**贴线**）。
- **bwd**：GM 流量 ≈ 784KB/cid × 1024 ≈ 803MB → @600GB/s ≈ 1.34ms；实测 1.30ms——**已在传输地板上**
  （staging 131MB 多数 L2 命中）。
- 唯一 IO 物理下限 826us（1044MB ÷ 1263GB/s）对当前结构不可达（见下）。

### 剩余差距分类
**kernel 层（本仓可做，未做）**：
1. k_inter fixed-core 任务循环 + 显式 L2 常驻 relay（inter R1-iter2 同构，-11.3% 家族实证）：
   压 C 锁步（当前 C 占用 7.4 vs 生产性 2.7），bound → max(ΣC_prod/20≈1.1, ΣV/40≈2.2) ≈ 2.2-2.5ms
   → 潜在 **−1.0~1.3ms**。未做原因：×6 kernel 重写 + 手动 flag/planner 高风险（inter iter6 教训），
   本轮预算内优先了低风险高确定性项。
2. 小 kernel（k_g/g2/y/ai）frag 回传 L0C→GM 直写 + 手动 FIX flag（bwd 先例）：省 auto-relay 开销
   不省依赖等待 → 潜在 −0.3~0.6ms。
   两项合计潜在 −1.3~1.9ms → **~10.1-10.7ms 可达带**。

**工具链层（本仓不可解，需上游）**：
1. C/V 多跳限制强制 fwd 6-kernel 拆分 → 624MiB 级 ws 往返（DESIGN §9.5 已论证 1.34-1.65ms 逻辑
   下限来源）；单体融合需多 C 段支持。
2. fp32 gemm_v0 操作数损坏 → 全链 hi/lo 3× gemm + 全 ws 分裂对（操作数字节 ×2 vs fp32 直用）。
3. V 侧 per-op 发射开销（~100-200ns/op）：k_subst 的 336 小 op 串行 = 2.5ms——前代行递归语义串行
   + 无 V 侧批量行解算原语；Neumann gemm 化需 kernel 内多 C 段（同 1）。
4. threads=2 对本算子族实证封堵（iter6，+47%——vid 拆分开销 > 双 AIV 收益；与 inter 家族结论互证）。

## 6. 重定目标建议

- **建议目标值：fwd+bwd kernel 合计 ≈ 10.5ms**（口径：do_bench NPU 事件、warmup=20 rep=10 mean、
  仅设备侧、fwd (1,8192,64,128) bf16 / bwd (1,8192,8,128) fp32）。
- 依据：当前实测 12.03ms + 已识别且家族实证的 kernel 层剩余项（§5 潜在 −1.3~1.9ms）外推
  10.1-10.7ms；10.5 取带中值，验收余量 ±0.5ms。
- 边界说明：proto 原 540us 目标维持物理不可达结论（826us 唯一 IO 下限的 0.65×，需 1.53× 实测
  r+w 带宽）；**<4.2ms 的任何目标都隐含工具链修复前提**（fp32 gemm / 多 C 段 / 任务循环 planner），
  建议在目标条款中显式标注为上游依赖项而非算子层承诺。
- bwd 单独口径建议：≤1.35ms（已在地板上，验收值 = 实测 + 噪声余量）。

## 7. 文件与可复现

- 终态实现：`fla_chunk_intra.py`（6 fwd + 2 bwd JIT kernel）；快照链 `history_version/fla_chunk_intra_perf_r1_iter{0,2,3,4,5}_before.py`。
- bench：`perf_tuning/bench_attempt2.py`（官方口径）；A/B：`fla_chunk_intra_ab_r1_iter{1,2,3}.py` + `/tmp/opencode/probe/ab_iter{5,6}.py`。
- 画像：`msprof_intra_*`（基线）/ `msprof_intra2_*`（iter3 后）/ `msprof_intra3_*`（终态 fwd），
  解析 `parse_intra_msprof.py`。
- 探针（/tmp/opencode/probe）：intra_p1（k_inter Akkbp 对拍）、p3c-p3g（0×NaN 定位与注入复现）。
- 主仓未 commit；ref_gpu/ 未改动；DESIGN 语义未变（无 [DESIGN_ERROR]——ws 契约与 kernel 数为
  实现层，模块 docstring 已同步）。


---

# R2 终态（2026-11-03，用户决策：R2 再调一轮后交付）

## R2.1 实测（官方口径 ×4，设备故障恢复后采集）

| kernel | run1 | run2 | run3 | run4 | 代表值 |
|---|---|---|---|---|---|
| fwd k_inter | 3138.9 | 3138.8 | 3147.0 | 3133.9 | 3138.6 |
| fwd k_subst | 2540.7 | 2550.4 | 2536.6 | 2551.4 | 2544.8 |
| fwd k_g | 1206.1 | 1207.3 | 1196.6 | 1212.6 | 1205.7 |
| fwd k_g2 | 1067.8 | 1064.9 | 1068.0 | 1064.3 | 1066.3 |
| fwd k_y | 1224.9 | 1228.8 | 1238.6 | 1228.3 | 1230.2 |
| fwd k_ai | 1134.2 | 1130.0 | 1138.0 | 1131.9 | 1133.5 |
| **fwd chain（6 kernel）** | 10148.2 | 10170.2 | 10135.1 | 10183.8 | **10159.3** |
| **bwd chain（2 kernel）** | 1303.6 | 1285.6 | 1272.4 | 1272.2 | **1283.5** |
| **合计** | 11451.8 | 11455.9 | 11407.5 | 11456.0 | **11442.8** |

- 相对 R1 终态 12027.5：**-4.9%**；相对 Stage-2 基线 19774.3：**-42.1%**。
- 精度：--level all **90 PASS / 0 FAIL / 0 WARN**（R2-#1 数值逐位一致；ruff clean）。
- R2 采纳：k_inter fixed-core 任务循环 + 显式 L2 relay（k_inter 3564→3139，-11.9%）。
- R2 回滚/放弃：k_y frag 协议试点（-1.1~-1.6% 子阈值，小 kernel 已在 max(C,V) 重叠极限）；
  drain mask 乘（估算 -0.25ms；UB 溢出→半行拆分→planner 确定性挂死→设备 ffts 锁存故障，
  按纪律止损放弃）。

## R2.2 可达带判定：**未进带**（11443 > 10700）

R1 估的 10.1-10.7ms 带偏乐观，两处修正（实证）：
1. inter 家族 -11.3% 不可外推：inter kernel 是 C 占用主导（锁步可消），intra k_inter 在锁步消除后
   显露 **V 侧地板**（fixed-core msprof：wall 3030us = 每 core V 工作和；produce 4.5us + drain 2.9us
   /task × 8192 / 20 AIV）——fixed-core 只兑现 -0.43ms（vs 估 -1.0~1.3）。
2. 小 kernel frag 协议换手不消依赖等待（-1.1~-1.6% 实测）。

## R2.3 地板核算（路线穷尽判据）

**kernel 稳态**（4 次运行带 11408-11456，do_bench 仅设备侧，host 分发不在口径内）：
- k_subst 2545us：**V 语义地板**（前代行递归链内 RAW 固有串行；40 AIV 满载，ΣV/40 = 实测值）。
- k_inter 3139us：**每 core V bound**（fixed-core 后锁步已消；20 AIV——threads=2 双重证伪不可用 40）。
- k_g/g2/y/ai 4636us：各在 **max(C,V) 跨块重叠极限**（frag 数据依赖固有）。
- bwd 1284us：**803MB@~600GB/s 传输地板**（实测 1.27-1.30 vs 模型 1.34）。
- fwd 合成地板：流量 4.4GB@600GB/s ≈ 7.3ms + k_subst V 地板 2.5ms（不重叠）≈ **9.8ms** → 实测 fwd
  10.16 = 合成地板的 1.04×（贴线）。

**「kernel 仍可优化」类（本仓可做但本轮受阻）**：
- k_inter drain mask 乘（~-0.25ms）：被 planner UB 预算（192KB 贴线）+ 语句重排确定性挂死
  （R1-iter6a 同源第三实证）阻断——**需 planner 别名/屏障鲁棒性改进后重试**。

**「需工具链配合」类（本仓不可解）**：
1. 每 kernel 单 C 段 → 强制 8-kernel 拆分：fwd GM 流量 ~4.4GB vs 唯一 IO 706MB（**826us 物理下限
   在现工具链不可达**——需多 C 段/单体融合支持）。
2. fp32 gemm_v0 损坏 → 全链 hi/lo 3× gemm + ws 分裂对。
3. V per-op 发射开销 ~50ns/op：k_subst 的 336 op 串行链 = 语义+发射地板。
4. threads=2 vid 拆分开销 > 双 AIV 收益（+47%，两轮双重证伪）。

## R2.4 验收建议值

**≈11.5ms**（实测 11443 + 噪声余量；4 次运行带 11408-11456）。口径：do_bench NPU 事件、warmup=20
rep=10 mean、仅设备侧、fwd (1,8192,64,128) bf16 + bwd (1,8192,8,128) fp32。
**≤10.7ms 需前置条件**：planner 鲁棒性修复（~-0.3ms 级重试项）或工具链结构项（多 C 段/fp32 gemm/
任务循环 planner 支持——收益一个数量级更大）；540us proto 目标维持物理不可达结论不变。

## R2.5 过程事故记录（详见 perf_log R2-#3 与 debug_log G18-G21）

- head-set 必须为 prologue 首语句（否则 CombineCV 归类 V 段 → 死锁）。
- prim_func 体内局部变量使 region/shape 符号化（"size must be same"/PTO 非常量形状）。
- planner 语句重排第三实证（挂死形态）。
- **设备 ffts 锁存故障**：挂死 kernel 级联设备级错误（跨进程 ~40 分钟自愈；容器内无复位权限；
  纯 torch/简单 kernel 探针可先行判定设备状态——本轮曾据此误判好代码挂死）。
