# panel_metric2_bench — 指标 2 终验（panel 路径 + generic 回归 + 差距归因）

日期：2026-10-09 ｜ kernel 终态 = panel-mode 实现版（1351 行；panel builder + host 分发 `panel_mode="shared"`；通用路径零改动）
工具：/tmp/opencode/panel_final_bench.py（graph-replay 主口径 2 组独立）+ prof_panel.py（msprof）

## 0. 测量协议

- **数据构造**：golden 同源——seed=42 randn bf16、offsets=[0,1023,2048]、topk_indices=arange(128) 全行重复 [S,1,128]、lse fp32 [S,H]（值不影响计时）。token_indices 预构建（不计时）。
- **panel 分发验证**：`_panel_gate_ok() → ok=True reason=ok` ✓；`panel vs generic(BI128) torch.equal=True` ✓（精度等价）；panel kernel 经 `tl_sparse_mla_topk_reducesum_panel()` 直调（绕过 host dispatch，纯 kernel 计时）。
- **主口径 graph-replay**：batch 内 N=1 call 捕获单 graph，batch=25 replay 事件计时，iters=20，**2 组独立测量**。
- **副口径 msprof**：Task Duration（独立进程 12 call）。

## 1. 主口径结果

### Panel 路径（golden 配置，panel_mode="shared"）

| run | mean | min | max |
|-----|------|-----|-----|
| 1 | 143.1 | 141.8 | 144.3 |
| 2 | 143.5 | 142.7 | 144.0 |
| **均值** | **143.3** | 141.8 | 144.3 |

**vs 220us：PASS（余量 1.53×）**

### Generic 路径回归复测

| 配置 | 本轮 | 前轮锚点（同会话系列） | 偏差 | developer 粗测 | 与 developer 偏差 |
|------|------|---------------------|------|----------------|------------------|
| BI=128 | 378.9 | 364.7 | +3.9% | 384.9 | -1.6% |
| BI=32 | 471.4 | 475.3 | -0.8% | 473.1 | -0.4% |

BI128 偏差 +3.9% 在已知跨会话漂移带 ±3-5% 内（D512/BI128 配置历史漂移带 356.8-396.1us 含 msprof）。BI32 偏差 -0.8% 在带内 ✓。

### msprof 交叉

| 路径 | msprof | graph-replay | 比值 | 判定 |
|------|--------|-------------|------|------|
| panel | 147.1 | 143.3 | **1.027** | ✓（低于典型 1.06-1.14 带——panel kernel 无 profiler 敏感段） |

## 2. 汇总判定表

| 路径 | 配置 | graph-replay mean | vs 220us | 提速 vs generic BI128 | 提速 vs generic BI32 |
|------|------|------------------|----------|---------------------|---------------------|
| **panel** | panel_mode="shared" | **143.3us** | **PASS** | **2.64×** | **3.29×** |
| generic | BI=128 | 378.9us | FAIL (+72%) | 1.00× | 1.24× |
| generic | BI=32 | 471.4us | FAIL (+115%) | 0.80× | 1.00× |

## 3. 差距归因：实测 143.3 vs 设计预算 60-80us

### 3.1 msprof pipe 分解（panel 路径）

**AIC（C 侧）= 141.1us**：

| 分量 | busy (us) | 占 aic_time | 设计 §10.1 预估 | 差距 |
|------|-----------|------------|----------------|------|
| cube (gemm) | 28.6 | 20% | 28.6 | **0 ✓ 精确匹配** |
| **MTE1 (L1→L0)** | **73.2** | **52%** | **~0（假设被 L0 乒乓隐藏）** | **+73.2 ← 最大设计盲区** |
| MTE2 (GM→L1, Q+panel) | 48.6 | 34% | ~11.8→~0 暴露 | +37（Q 读率 38.8GB/s ≠ 设计假设 160GB/s clean L2） |
| FIX (L0C→GM) | 20.0 | 14% | ~18→10-14 暴露 | +6 ✓ 接近 |
| wait F_PANEL | 4.9 | 3% | — | ✓ 小（panel 仅 288KB） |
| scalar_mte1_stall | 106.4 | 75% | — | 标量被 MTE1 串行链阻塞 |

**AIV（V 侧）= 141.1us**：

| 分量 | busy (us) | 占 aiv_time | 设计 §10.2 预估 | 差距 |
|------|-----------|------------|----------------|------|
| vec (exp/mul/reduce) | 41.6 | 30% | 39.2 | +2.4 ✓ |
| MTE2 (ws_acc 读) | 41.8 | 30% | ~18→5-8 暴露 | +34（重叠不充分） |
| MTE3 (输出+panel 写) | 18.7 | 13% | ~6 | +13（含 panel gather + 输出） |
| scalar_wait (cross-flag) | **54.8** | **39%** | ~5-10 | **+45 ← 第二大设计低估** |
| — F_PANEL wait (id4) | 20.3 | 14% | — | 每 task 握手 ×102 |
| — F_PDONE wait (id5) | 21.3 | 15% | — | 每 task 握手 ×102 |

### 3.2 差距构成表（63-83us 的来源）

| 排名 | 缺口分量 | 量级 (us) | 根因 | 设计预判偏差 |
|------|---------|----------|------|------------|
| **1** | **MTE1 (L1→L0) 未隐藏** | **~73** | gemm 操作数装载的串行依赖链（flag 序）阻止与 cube 重叠；设计假设 L0 乒乓完全隐藏 | **设计 §10.1 未单列 MTE1——假设 gemm_v0 内部 L0 ping-pong 自动隐藏** |
| **2** | **cross-flag 耦合（V 侧 wait）** | **~42** | 每 task F_PANEL/F_PDONE 握手 ×102 task/核 = 204 次跨核 flag 往返；设计按"每组 4 flag × 8 组 = 5-10us"估算 | **设计 §10.3 取历史 coupling 带低段 5-10us——实际 per-task 频率远高于 per-group** |
| 3 | MTE2 Q 读率 | ~37 | 设计假设 clean L2 160GB/s；实际 38.8GB/s（Q 37.7MB 与 V/AIC 争带宽 + 18KB/task 小块） | 设计 §10.1 Q 读率取值过高 |
| 4 | V ws_acc 读暴露 | ~34 | 设计假设与 V 计算流水重叠后暴露 5-8us；实际 41.8us busy + 重叠不足 | 设计 §10.2 重叠假设过乐观 |
| 5 | V panel gather + 输出 | ~13 | 设计 ~6us；实际含 MTE3 输出写（1MB）+ panel 组装 | 设计低估输出成本 |
| — | cube / FIX / vec / launch | ≈0 | 精确匹配或接近 | ✓ |

### 3.3 归因结论

143.3us ≈ 设计各分量（60-80us）+ **MTE1 串行 73us** + **cross-flag 42us** − 部分重叠抵消（~30us）。设计的三大分析盲区：
1. **MTE1 (L1→L0) 完全被忽略**（假设 gemm 内部 L0 乒乓隐藏；实际 flag 依赖链阻止）
2. **per-task cross-flag 频率低估 20×**（5-10us → 42us：per-group 估算 vs per-task 实际）
3. **Q 读 GM→L1 率假设过高**（160 → 39GB/s）

## 4. roofline 数据版

| 路径 | AI (FLOP/B) | 实测吞吐 (TFLOPS) | 落墙区段 |
|------|------------|------------------|---------|
| **panel** | **81** | **31.6** | **MTE1 串行区**（L1→L0 装载 52% AIC busy 为主导） |
| generic BI128 | 4.8 | 12.0 | C-ws 脏 burst 区（302MB 往返主导） |
| generic BI32 | 4.8 | 9.6 | C-ws 区 |

panel AI = 4.53 GFLOP / 55.8 MB（Q 37.7 + ws_acc 16.8 + panel 0.3 + 输出 1）= 81 FLOP/B——比 generic 提升 17×，但 MTE1 L1→L0 装载成为新瓶颈（非字节墙而是 L0 装载串行化）。

## 5. 不确定性声明

1. **graph-replay run-to-run 漂移 0.3%**（143.1↔143.5）：远小于已知 ±3% 带——panel kernel 短且无 V/C 均衡敏感区。
2. **msprof 比值 1.027**：低于典型 1.06-1.14 带——可能因 panel kernel 无 profiler 敏感的 DMA 排空段；147.1us 为保守上界。
3. **generic BI128 回归 +3.9%**：在 D512/BI128 已知漂移带内（356-396us 含 msprof，4 会话 8 组测量带）。非 kernel 回退（panel-generic equivalence torch.equal=True 逐字节验证）。
4. **MTE1 73.2us 的"未隐藏"判定**：基于 mte1_busy 52% + mte1_stall 75% 的 pipe 交叉证据；无独立 MTE1-only 测量。可能部分 stall 来自 MTE2（MTE1 等 L1 数据就绪），但量级不超过 MTE2 busy 48.6us。
5. **cross-flag 42us**：来自 V 侧 scalar_wait id4/id5 的 msprof 计数；包含所有 F_PANEL/F_PDONE 等待（含首尾填充/排空）。
6. **设计预算 60-80us 引用自 DESIGN_panel_mode.md §10.3**（设计期预判，±10% cube 率 + coupling 无独立测量）。
