# fla_wy_fast Stage 3 优化记录（optimization_log）

> 环境：910B3 (64GB, 20 AI Cores) / CANN 9.2.0 / tilelang 0.1.4+1d752539。
> 计时口径：NPU event，warmup=20，rep=10，均值，仅设备侧（upstream do_bench 等价）。
> per-kernel 计时：bench_per_kernel.py（独立 kernel 闭环，rep 间无 sync 暴露）。

## Step 1-2: 基线与算子类型

- 算子类型：**CV 混合型**（生成代码同时含 `ASCEND_IS_AIC` 与 `ASCEND_IS_AIV`，
  KERNEL_TYPE_MIX_AIC_1_1）。
- 分 kernel 基线（2026-10-22，调优前）：

| kernel | 时间 ms | us/块 (×20核) | 逻辑 GM 流量 | 唯一 HBM 流量 |
|---|---|---|---|---|
| HBM 参照 (D2D copy 512MiB×2) | 0.848 | — | 1 GiB | **1263-1266 GB/s (r+w)** |
| fwd-u | 3.789 | 9.2 | ~1.4 GB (含 workspace 中继) | ~321 MiB |
| fwd-wkg | 4.238 | 10.3 | ~1.7 GB | ~705 MiB |
| bwd-k1 | 2.311 | 11.3 | ~1.35 GB | ~577 MiB |
| bwd-k2 | 2.588 | 12.6 | ~1.8 GB | ~1.09 GB |
| bwd-v1 | 2.248 | 11.0 | ~1.0 GB | ~385 MiB |
| bwd-v2 | 4.332 | 21.1 | ~1.0 GB | ~450 MiB |
| bwd-z | 0.808 | 3.9 | ~260 MiB | 192 MiB |
| bwd-da | 0.803 | 3.9 | ~260 MiB | 192 MiB |

**瓶颈定位结论（数据支撑）**：
1. **D2D 拷贝实测 1263-1266 GB/s (r+w)** —— DESIGN §9.5 假设的 392GB/s 标称与实测
   硬件不符（该数字为【推断】未校准，DESIGN 自己也标注"Stage 2 需 STREAM 实测
   校准"）。"两方向 350-400GB/s 已近 HBM 饱和"的 Stage 2 结论**不成立**：把
   workspace 中继全算 HBM 流量，fwd-u 理想 ~1.1ms vs 实测 3.789ms（3.4×）；按
   唯一流量算仅 ~0.27ms（14×）。瓶颈不在 HBM 带宽。
2. **每块执行延迟主导**：无串行行循环的 z/da 单发 kernel 3.9us/块；串行行循环
   最多的 v2 21.1us/块——强相关。
3. **生成代码证据**（fwd-u `get_kernel_source()`）：`T.serial(64)` 行循环与
   in-place 2D `T.Parallel` 行广播**都** lower 成 64 次标量循环（`bt.GetValue(i)`
   + 每行一条 `Muls` repeat=1 瘦指令）——反模式清单"Vector Core 内逐元素/逐行
   for loop 计算"。fwd-wkg 的 kg 循环 64行×4ops×2slice = 512 瘦指令/块，为
   全算子最大热点。
4. A_l1 是 AIC 侧 `copy_gm_to_l1` 直达（不经 workspace），fwd 的 A 循环内重复读
   代价小（L1 双缓冲）；V 侧 gemm 操作数经 GM workspace 中转（AIV 写 →
   CrossCore flag → AIC MTE2 读）。

## Step 3: Part A 优化点清单

[#1] [Vector 行循环 → broadcast + 整 tile 指令]（performance-antipatterns.md
     L127-155 "Vector Core 内逐元素/逐行 for loop 计算"; optimization-guide.md
     §2.6 L381）: **适用** — 6 个 kernel 存在 `for i in T.serial(64)` 行级 tile
     操作（fwd-u bt mul / fwd-wkg kbg / k1 kbg / k2 kbg+dk2 / v1 vb / v2 dv）。
     写法：`T.tile.broadcast(bt2d, bt, axis=1)`（行向量 [BS]→[BS,bD]）+ 一条
     2D `tile.mul`。先例：mha_sink_fwd_bhsd.py:431/606、opt_gdn_solve_tril.py:61。

[#2] [fwd-wkg kg 列广播链 → broadcast + 2D tile 链]（optimization-guide.md
     §2.6; antipatterns "正交轴串行化" L429）: **适用** — kg 的 64 行×4 ops 串行
     循环改为 `gnL = gn*LN2`(1D) → `broadcast(gn2d, gnL, axis=0)` → `sub(gn2d,
     gn2d, gL)` → `exp` → `mul(kg, k, gn2d)`，6 条胖指令替代 256+ 瘦指令。

[#3] [bwd k2/v2 A cast 链提 prologue]（optimization-guide.md §2.4）: **适用
     （受限版）** — A 与 slice 无关但在循环内每 slice 重算 fp32→fp16 hi/lo
     cast 链；完整 hoist（含 relay）触发 pipeline pass prologue 交换错版本化
     bug；仅 cast 链（纯 V 工作）提 prologue、relay 留循环内。

[#4] [z+da 合并 / dbeta scatter 外移 / threads=2]: **不适用/暂缓** — z+da 合并
     复现 debug_log #6 V→C→V→C→V sync-point 编译错；dbeta scatter 外移做了
     消融（见 #5）；threads=2 被 gather 链 vid 切分 bug 阻塞。

[#5] [dbeta scatter 从 v2 移至 da]: **消融后回滚** — v2 的 MTE3→MTE2 barrier
     实测成本 ~0（流水线吸收），da 无循环可隐藏新增 load+scatter → 净 +0.04ms。

[#6] [k2 公共子表达式 Xg = X·2^g]: **UB 溢出回滚** — r_tmpk 生命周期延长与
     dk2_ub 重叠 +16KB，DK=128 时 memory-planning 溢出（k2 已在 UB 边缘）。

[#7] [GM↔UB cast-on-copy 直载直存]: **工具链不支持** — codegen 生成
     `copy_gm_to_ub<bfloat16_t>` 收 `LocalTensor<float>` 的非法模板调用（探针
     复现，probe_cast.py）；bf16↔fp32 跨 dtype GM 拷贝未实现两步 lowering。

[#8] [fwd-wkg kg div 变体 (k·2^gn)/2^g]: **无收益回滚** — 1D exp(64)+div(4096)
     与 sub+exp(4096) 等价耗时，div 与 exp 在 Vector 核同价。

[#9] [host 侧 msk 常量缓存]（optimization-guide.md §2.12 Host 轻量化）:
     **适用** — bwd wrapper 每次调用 CPU tril + H2D（可能同步的 pageable 拷贝）
     在计时区内；改为模块级 per-(device,chunk) 缓存。

[#10] [wrapper 内部 sync 移除]（口径对齐）: **适用** — 上游 do_bench 只在 rep
     循环外 sync，上游算子 wrapper 内部无 sync；Stage 2 内部 sync 把 JIT 分发
     延迟（~1ms/launch，cython wrapper 每次重建 TVM Analyzer）暴露进设备侧
     计时。移除后对齐上游语义。

[KERNEL-REUSE-PRECHECK] [#1][#2][#3] 不扩大 kernel 调用域（shape/batch/dtype/
尾块/stage 不变，仅核内指令重排）；[#2] 的 broadcast 对 fp32 [64]→[64,64] 有
dtype coverage 测试与 mha_sink（fp32）先例 → unknown/false: none →
plan: direct-reuse。

## Part B [ORDER-PLAN]（实际执行序）

1. [#1a] fwd-u bt 广播 → 2. [#1b] fwd-wkg kbg 广播 → 3. [#2] kg 列广播链
4. [#1c] k1 / [#1d] v1 → 5. [#1e] k2 / [#1f] v2 → 6. [#3] A hoist（回滚）
→ 7. [#9] msk 缓存 → 8. [#5] scatter 外移（回滚）→ 9. [#6] k2 CSE（回滚）
→ 10. [#7] 直载直存（工具链阻塞）→ 11. [#8] div 变体（回滚）→ 12. [#10] sync 移除

## Step 4-5: 逐项实施与验证记录

[ORDER-CHECK]/[IMPL-#N]/[RESULT-#N] 汇总（每步 L0 全过后再 bench，全量
--level all 在批次末复验）：

```
[RESULT-#1a] fwd-u bt 广播        | 精度 pass | 3.789 -> 2.023 ms | -47%
[RESULT-#1b] fwd-wkg kbg 广播     | 精度 pass | 4.238 -> 3.118 ms | -26%
[RESULT-#2]  fwd-wkg kg 列广播链  | 精度 pass | 3.118 -> 2.971 ms | -4.7%
             （exp 数学吞吐为固定成本，循环开销仅小头）
[RESULT-#1c] bwd-k1 kbg 广播      | 精度 pass | 2.311 -> 1.313 ms | -43%
[RESULT-#1d] bwd-v1 vb 广播       | 精度 pass | 2.248 -> 1.229 ms | -45%
[RESULT-#1e] bwd-k2 kbg+dk2 广播  | 精度 pass | 2.588 -> 1.797 ms | -31%
[RESULT-#1f] bwd-v2 dv 广播       | 精度 pass | 4.332 -> 1.258 ms | -71%
[RESULT-#3]  k2/v2 A cast 链 hoist| 精度 pass(min shape) | v2 1.258 -> 1.278
             （中性偏差；Pipelined 软件流水已重叠不变量 cast 链。k2 DK=128
              UB 溢出编译失败 → 一并回滚）
[RESULT-#9]  msk 常量缓存 (host)  | 精度 pass | BWD 端到端 8.635 -> 7.663 ms
[RESULT-#5]  dbeta scatter 移 da  | 精度 pass | v2+da 2.061 -> 2.099 ms（净负）
             → 回滚
[RESULT-#6]  k2 CSE Xg            | 编译失败（UB 溢出 r_tmpk +16KB）→ 回滚
[RESULT-#7]  GM↔UB 直载直存       | 编译失败（工具链模板错配）→ 回滚，记录
[RESULT-#8]  kg div 变体          | 精度 pass | 2.971 -> 2.987（持平）→ 回滚
[RESULT-#10] wrapper sync 移除    | 精度 pass | 端到端 13.084 -> 11.719 ms
```

## 最终 per-kernel 状态（2026-10-22，调优后）

| kernel | 基线 ms | 最终 ms | Δ | us/块 |
|---|---|---|---|---|
| fwd-u | 3.789 | 1.870 | -51% | 4.6 |
| fwd-wkg | 4.238 | 2.819 | -33% | 6.9 |
| bwd-k1 | 2.311 | 1.321 | -43% | 6.5 |
| bwd-k2 | 2.588 | 1.802 | -30% | 8.8 |
| bwd-v1 | 2.248 | 1.224 | -46% | 6.0 |
| bwd-v2 | 4.332 | 1.257 | -71% | 6.1 |
| bwd-z | 0.808 | 0.825 | +2%（噪声） | 4.0 |
| bwd-da | 0.803 | 0.807 | 噪声 | 3.9 |
| **合计（设备和）** | 16.12 | 11.93 | **-26%** | — |

## 端到端（bench_fla_wy_fast.py 口径）

| 口径 | 基线 | 最终 | Δ |
|---|---|---|---|
| Stage 2 口径（wrapper 内含 sync） | 23.151 | 13.084 | **-43.5%** |
| 上游对齐口径（wrapper 无 sync） | 21.636 | **11.719** | **-45.8%** |

（fwd 4.783 / bwd 6.935，上游口径；基线同口径复测 fwd 8.013 / bwd 13.622）

## 剩余瓶颈归因（最终）

1. **逻辑 GM 流量放大**（kernel 拆分重读 + C/V workspace 中继）：大 kernel 逻辑
   带宽 600-1020 GB/s（实测 D2D 上限的 48-81%），k1/k2 已贴近；消除需工具链
   修复（单 Pipelined 体限制、多 gemm 组 L0 损坏、UB↔L1 片上直连缺失）。
2. **z/da 单发 kernel 延迟受限**（~4us/块，逻辑带宽仅 25%）：无循环可流水，
   合并被 sync-point bug 阻塞。
3. **host JIT 分发 ~1ms/launch**（cython wrapper 每次重建 TVM Analyzer +
   stream 查询）：上游口径下被队列深度掩盖，但 8 kernel 串行场景仍是下限。
4. **UB 容量边缘**（k2）：CSE/A-hoist 均被 memory-planning 溢出阻塞。
