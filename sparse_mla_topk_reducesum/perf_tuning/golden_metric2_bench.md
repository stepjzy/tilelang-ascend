# golden_metric2_bench — 指标 2 精测（golden 配置纯 kernel + 220us 判定）

日期：2026-10-09 ｜ kernel 终态 #19-p2+#22（git status 未跟踪、无 Phase-4 R2 残留 ✓）
工具：/tmp/opencode/golden_metric2_bench.py（graph-replay 主口径）+ prof_golden.py（msprof）+ golden_e2e_graph.py（graph-captured e2e 分解）

## 0. 测量协议

- **数据构造**：与 test_sparse_mla_topk_reducesum.py 逐行同源——seed=42 randn bf16 → `.npu().bfloat16()`；offsets=[0,1023,2048]（varlen 1023+1025）；topk_indices = arange(128) 全行重复 [S,1,128]；lse 由 fp32 参考流程生成（值不影响 kernel 计时，但保持同源）。
- **token_indices 预构建**（不计时，与历史锚点口径一致）——纯 kernel 执行。
- **主口径 graph-replay**：batch 内 N=1 call 捕获单 graph，batch=25 replay 事件计时，iters=20，**2 组独立测量**（各自完整 warmup+build-cache 预热）。
- **副口径 msprof**：Task Duration（独立进程 12 call）。
- **graph-captured e2e**：kernel + 归一化（`sum(dim=-2)` + `rs / rs.sum(dim=-1,keepdim=True)`）捕获进同一 graph，消除 host dispatch——分离归一化 GPU 成本。
- **eager e2e**：单次事件计时（含 tilelang host dispatch ~1.2ms + torch eager op dispatch）。
- **历史锚点口径差异声明**：kernel docstring 披露的 426.7us@BI128 / 520us@BI32 为 **randint clamp 散射数据**（bench_graph_smtk.py 构造 `randint→clamp(max=s_col)`）；本轮 golden 数据为 **arange 全行重复**（全任务读相同 128 行 = 高度连续/L2 极热）——**数据形态差异已知（~1.8×），历史锚点直接对照需注明**。

## 1. 主口径结果（graph-replay，2 组独立测量）

| 配置 | run1 mean | run2 mean | **均值** | min | max | **vs 220us** |
|------|-----------|-----------|---------|-----|-----|-------------|
| **BI=128（调优档）** | 359.9 | 369.4 | **364.7** | 356.8 | 370.9 | **FAIL（+65.8%）** |
| **BI=32（默认档）** | 476.2 | 474.4 | **475.3** | 473.5 | 491.8 | **FAIL（+116.0%）** |

run-to-run 漂移：BI=128 2.6%、BI=32 0.4%（在已文档化 ±3% 带内）。

## 2. 副口径交叉（msprof Task Duration）

| 配置 | msprof | graph-replay 均值 | 比值 |
|------|--------|-----------------|------|
| BI=128 | 396.1us | 364.7us | 1.086 |
| BI=32 | 479.6us | 475.3us | 1.009 |

BI=128 比值 1.086 在已知 profiler 开销范围（1.06-1.14）内 ✓。

## 3. 辅助口径：端到端（kernel + 接口归一化）

| 配置 | graph-captured e2e | graph-captured kernel | **归一化 GPU 成本** | eager e2e（host-bound） |
|------|--------------------|--------------------|-------------------|----------------------|
| BI=128 | 395.4us | 356.4us | **~39.0us** | 1557.4us |
| BI=32 | 508.9us | 471.3us | **~37.6us** | 1688.0us |

- **归一化 GPU 成本 ≈ 39us**（>20us 阈，单独拆出）：为 2 个小 op（sum(dim=-2) + div）在 [2048,1,1,128] fp32 上的 kernel launch 开销主导（实际计算 <1us；43us 独立测量含 graph replay 侧 ~4us）。
- **eager e2e 1557-1688us** 为 host dispatch 主导（tilelang host API ~1.2ms floor + torch eager 2 op dispatch ~100us each）——非 GPU 计算成本，已文档化（Phase 1-4 eager 口径失效结论）。

## 4. 与历史锚点对照

| 配置 | 历史锚点（randint 数据） | 本轮实测（arange golden 数据） | 偏差 | 解释 |
|------|------------------------|---------------------------|------|------|
| BI=128 | 426.7us | 364.7us | **-14.5%** | **数据形态差异**：golden arange 索引全任务读相同 128 连续行（batch0: rows 0-127, batch1: rows 1023-1150），L2 全命中 + MTE2 burst 连续——vs randint clamp 散射行读（随机 1152B 行）。已实证 arange vs randint 差异 ~1.8×（分析轮 F6） |
| BI=32 | 520us | 475.3us | **-8.6%** | 同上（BI=32 粒度更细、局部性增益更小） |

偏差 >5% 的原因 = **数据形态**（arange 连续 vs randint 散射），非 kernel 回退或测量漂移——两组独立测量带（356.8-370.9）远小于与锚点的差距。

## 5. 判定汇总

| 指标 | BI=128 | BI=32 | 判定 |
|------|--------|-------|------|
| 纯 kernel（graph-replay 主口径） | **364.7us** | **475.3us** | 均 **FAIL** |
| vs 220us 差距 | +144.7us（+65.8%） | +255.3us（+116.0%） | — |
| kernel + 归一化（graph-captured） | 395.4us | 508.9us | 均 FAIL |
| 与 220us 的结构差距 | 1.66× | 2.16× | Phase 1-4 拼装地板 415-430@D512/randint 散射数据——**arange golden 数据形态更友好（-14.5%），但 220us 仍不可达** |

## 6. 不确定性声明

1. **BI=128 run-to-run 漂移 2.6%**（359.9↔369.4）：在已文档化 ±3% 带内；2 组独立测量 + graph-captured 交叉（356.4us）确认区间 356-396us。
2. **msprof 比值 1.086**（BI=128）：profiler 开销含在 Task Duration 中，已知系统性偏高（1.06-1.14 带）。
3. **lse 值不影响 kernel 计时**（kernel 仅加载 lse 做 fp32 减法；数据内容无关）。本轮使用参考流程生成的 lse（与 golden 同源），但计时等价于任意有效 fp32 [S,H] 张量。
4. **归一化 GPU 成本 39us 的构成**：2 个 kernel launch（sum + div）各 ~20us，实际 GPU 计算 <1us（[2048,1,1,128] fp32 = 262K 元素）。在 graph 模式下 launch 侧开销为主要成分；eager 模式下额外叠加 host dispatch。
5. **历史锚点不可直接对照**：数据形态差异（arange vs randint）>14%，偏差原因为已知数据局部性效应而非 kernel 性能回退。
