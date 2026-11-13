# fla_wy_fast 流水图分析（baseline，msprof 双口径）

> 采集：2026-11-09，远程 910B3（DevEnvC_DCbyi_npu，基线 463c234c），分设环境经 cannlab。
> 交付版实现（sha f220d20a）。口径：msprof op 逐核 OPPROF（8 核各 1 次正式采集，warm-up=10）
> + 应用级 PROF（Insight 时间线，120 次 launch 位置映射）。
> 产物：`perf_tuning/fla_wy_fast_prof/baseline/{app_baseline, opprof/, profiling/}`。

## 1. 时间线 vs do_bench 交叉验证（数据可信性）

| 核 | 时间线均值(us) | do_bench(ms) | 逐 block 拆解 |
|---|---|---|---|
| u | 1977 | 1.98（Σ4.78） | AIV 4.45 / AIC 3.79 |
| wkg | 2827 | 2.83 | AIV 6.34 / AIC 5.22 |
| k1 | 1315 | 1.32（Σ6.94） | AIV 4.72 / AIC 6.01 |
| k2 | 1714 | 1.71 | AIV 9.45 / AIC 7.08 |
| v1 | 1228 | 1.23 | AIV 4.84 / AIC 6.11 |
| v2 | 1255 | 1.25 | AIV 9.94 / AIC 7.03 |
| z | 764 | 0.76 | AIV 4.74 / AIC 6.20 |
| da | 769 | 0.77 | AIV 4.71 / AIC 6.02 |

两口径逐核吻合（<3%）。墙模型：**TaskDur ≈ blocks × max(AIV,AIC per-block) / (16~20 通道)**。

## 2. 逐 pipe 归因（per-block 时间占比；ratio 分母=launch 总长，已换算）

| 核 | AIV 侧 | AIC 侧 | 关键腿 |
|---|---|---|---|
| u | **scal 61%** / vec 30% / mte2·3 22% | scal 56% / mte2 26% / cube 5% / wait_ib 1.04us | AIV(scal) |
| wkg | **scal 63%** / vec 40% | scal 49% / mte2 22% / cube 5% | AIV(scal) |
| k1 | **scal 99%** / mte2 47% / vec 39% | mte2 19% / scal 11% / cube 6%（可见项仅 ~40%，余为跨核等待） | AIC(等 AIV) |
| k2 | scal 59% / **mte2 55%** | （AIC 解析待补） | AIV |
| v1 | **scal 85%** | 同 k1 形态 | AIC(等 AIV) |
| v2 | scal 59% / **mte2 59%** | scal 40% / mte2 17% / wait_ib 1.15us | AIV |
| z | **scal 90%** | 同 k1 形态 | AIC(等 AIV) |
| da | **scal 82%** | 同 k1 形态 | AIC(等 AIV) |

**统一结论：AIV 侧 scalar pipe 是全部 8 个 kernel 的最终瓶颈**——fwd/k2/v2 直接 AIV 长腿；
k1/v1/z/da 是 AIC 等 AIV 供数（AIC 可见 pipe 项仅占 ~40%，其余为跨核等待）。

## 3. 对 Stage 3 结论的修正（重要）

- Stage 3 判"带宽主导、k1/k2 逻辑带宽 1020GB/s 近饱和"——**msprof 证伪**：k1 实际 GM
  流量 ≈ 448MB / 1300us = **345GB/s**（实测上限的 27%）；u = 321MB/1934us = 166GB/s。
  带宽远未饱和，墙是 **per-block scalar 控制开销 × block 数量**。
- Stage 3 的 broadcast 化优化（-9.21ms）有效的原因由此得到解释：它削减的正是
  scalar 指令条数（64 行循环 → 全量 tile 指令）。
- 判定流程对照（profiling skill §4.1）：u 实际 1934us vs 搬运理论 254us → 差距 7.6×
  （>50% 档）→ 瓶颈在模型之外 = scalar/控制开销，已定位。

## 4. 可寻址优化项（画像驱动，候选）

1. **多 chunk 合并 per block（降 block 数）**：现在每 block 只处理 1 个 64×D chunk，
   prologue 的 beta gather 链 / A 载入 / flag 初始化等 scalar 固定开销逐 block 重复。
   一个 block 处理连续 2~4 chunk（外层 serial 循环包住现有 Pipelined 体）可摊薄。
   fwd 8192→4096 blocks 理论省约一半固定 scalar。**风险**：debug_log 记录的
   "serial 循环内 V→C 交换同步坏"坑——外层 chunk 循环包住整个 Pipelined 体的形态
   需实验验证（工具链行为未知）。
2. **gather 链常量化（削 per-block scalar）**：beta 的
   arith_progression+reinterpretcast+gather 链每 block 一套（纯 scalar）；host 预计算
   成 GM 常量后整块 T.copy 载入，gather 链全消。代价 +极小 IO，收益 scalar。
3. **AIC wait_ib ~1us（u/wkg/v2）**：指令缓冲等待，间接信号，优先级低。

## 5. 穷尽判定（对照 migration skill 判据）

- ~~流水图无可寻址 stall~~ → **不成立**：已寻址到 scalar pipe 主导 + 两个候选优化项；
- 结构性重排已试：broadcast 化（已采纳）、hoist/CSE/scatter（已试无收益）、
  kernel 合并（工具链锁死）；
- 地板核算：带宽地板 1.97ms、host 分发地板稳态隐藏（实测 b−a ≈ −0.06ms）。
- **结论：kernel 层路线未穷尽**。scalar 优化预估可再收 20~40%（11.7ms → ~7-9ms 量级，
  估算非实测）；1370us 物理不可达不变（唯一 IO 带宽地板 1.97ms）。

## 6. 归档产物清单

- `app_baseline/PROF_*/`（75MB，含 mindstudio_profiler.db）——MindStudio Insight
  时间线复盘用（基线轮必采 ✓）
- `opprof/{u,wkg,k1,k2,v1,v2,z,da}/OPPROF_*/`——逐核 CSV（agent 读数口径）
- `profiling/<kid>/summary.txt + metadata.json + CSV`——perf_summary.py 归档摘要
- 时间线逐 launch 数据：`app_baseline/PROF_*/mindstudio_profiler_output/task_time_*.csv`
  （位置映射脚本内联在会话记录中）
