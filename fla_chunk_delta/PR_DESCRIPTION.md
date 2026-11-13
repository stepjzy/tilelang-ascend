为 Ascend NPU 新增 `fla_chunk_delta`（KDA gated delta rule 的 chunk 状态递推，前向 + 反向双 kernel）算子，迁移自上游 tilelang `examples/kda/FLA_KDA/fla_chunk_delta.py`（Triton 实现）。

- **Kernel 架构**：`@tilelang.jit` 装饰的前向/反向两个主 kernel（`T.Kernel(threads=2)`，Block Dim 20 / Mix Block Dim 40）+ 各自配套的 `threads=1` 门控预计算 kernel（Block Dim 20 / Mix 20）。Developer 模式（`alloc_shared`/`alloc_fragment` + 自动同步），主 kernel 采用 fixed-core 任务循环：B×H×NV 条递推链 striped 分配到 20 个 AI Core，核内 `T.serial` 逐 chunk 递推，2 链满批 + 受卫单链尾批。
- **计算结构**：每条链维护 fp32 状态 `H`/`dH` ∈ R^{DK×DV}，按 chunk（BT=64）串行递推。前向：先存当前状态快照 `h[t]`，`v_new = u − w @ H`，经 token 标量门控 `exp(g_last − g)` 与 per-channel 衰减 `exp(gk_last)` 更新后执行 `H += k^T @ v_new`；反向按 chunk 逆序：`dv2 = dv + (k @ dH) ⊙ exp(g_last − g)`，`dH = D·dH + (q̃^T @ do)·scale − w^T @ dv2`。`USE_G`/`USE_GK`/`USE_INITIAL_STATE`/`STORE_FINAL_STATE`/`SAVE_NEW_VALUE`/`USE_EXP2` 均为编译期分支。
- **数据流与内存管理**：门控量（`exp(g_last − g)`、衰减 tile、`q̃ = q·exp(g)`）在 prep kernel 中预计算为整 tile（fp32，硬件 transpose + 行广播展开），主 kernel 只消费不重算；GM → UB/L1 → L0 的搬运全部经 `T.copy`/`T.gemm_v0` 合法路径；递推状态常驻 UB（fp32；反向 2 链批处理配 A/B 双状态缓冲），跨核链间交换经 L2 上的 relay workspace（16 槽 ring）；K 维按 64 行 slab 合并 GEMM（前向每 chunk 2 次、反向 3 次 gemm_v0，半精度操作数）。
- **关键优化策略**：① 门控预计算 kernel 消除主 kernel 内每 chunk 192 次标量迭代（前向 7020.5 → 4219.4 μs，-39.9%；反向同幅 -36.9%）；② relay ring 化（NSLOT=16，跨链 WAR 解耦的使能结构，单独实测约持平）+ MTE3 队列排序让中继写提前于 HBM 快照写（前向再 -11.1%）；③ 链批结构 + 源码序重排（2 链满批 + 受卫单链尾批，消除每迭代 90+ 条标量分支的 Select 风暴）：前向 -29.6%、反向 -13.8%；④ `q̃` 外移至 prep（主 kernel 每链每 chunk 少 13 条 Vector 语句与 2 次中继握手，反向再 -7.5%）；⑤ 前向单链 + BV=128 全值宽（实测与 BV=64 持平，为更大 DV 与值域余量保留）；⑥ 反向 `q̃^T@do` 与 `w^T@dv2` 共享 readback、L0C 分时复用。
- **数值精度处理**：递推状态全程 fp32；per-chunk 衰减积 `exp(g_last)·exp(gk_last)` 合并为单次 `exp(g_last + gk_last)`（相对差 ~1 ulp，远小于输出量化步长）；输出按上游语义在 GEMM 前降至 fp16/bf16（`v_new`、`q̃` 的 cast 点与上游 Triton 实现逐位对齐）；`ht`/`dh0` 保持 fp32。
- **硬件与实现约束**：定长序列（S % 64 == 0）、DK ∈ {64, 128}（192/256 属上游声明域但本实现直接拒绝，有负向测试锁定）、DV 为 64 的倍数、H ≤ 248（prep 门控/衰减 staging 行上限）、chunk_size=64；fwd 输入 fp16、bwd 输入 bf16（`g`/`gk` 恒 fp32）；threads=2 下 gather/broadcast 不可用（vid 分裂），标量门控链全部走预计算整 tile 路径。
- **测试验证**：PyTorch fp64 重写参考实现为 golden（独立 token 级递推对拍 0 diff）；L0 门槛 6 用例 + L1 功能 10（尾批/参数分支/值域）+ L2 负向 10（非法 shape/dtype 正确拒绝，含 DK=96/256、DV=100、S 不整除、非法 dtype）+ Boundary 5（零输入/exp 下溢/inf/nan 结构/值域边界），31 个用例覆盖全部编译期分支与三档 dtype 容差；算子文件自带 CI smoke 入口（L0 门 shape 的 fwd/bwd 全 wrapper 调用与 shape/dtype/finite 检查，输出 `[SMOKE_PASS]`/`Test Passed!`，对齐 bench_test.sh 判定口径）。

## 精度验证 ✅

**测试配置**：`(B, S, H, DK, DV)` 覆盖 `(1,128,2,64,64)` 至 `(2,256,4,128,128)`；fwd fp16 / bwd bf16（g/gk fp32）；分支覆盖 h0/ht/dh0/save_new_value=False/use_exp2/scale

| **测试层级/类别** | **用例数** | **检查项数** | **测试结果** | **精度指标** | **最大绝对误差** |
| --- | --- | --- | --- | --- | --- |
| L0 门槛（规则 shape） | 6 | 13 | 全部 PASS | matched_ratio = 1.0000 | h 2.441e-04 / v_new 1.953e-03 / ht 1.288e-05 |
| L1 功能（尾块/分支/值域） | 10 | 22 | 全部 PASS | matched_ratio ≥ 0.9999 | dh 3.906e-03 / dv2 9.766e-04 / dh0 2.384e-07 |
| L2 负向（非法输入拒绝） | 10 | — | 全部正确拒绝 | — | — |
| Boundary（特殊值） | 5 | 10 | 全部 PASS | matched_ratio = 1.0000 | 0.000e+00 |
| **总计** | **31** | **45** | **全部通过** | — | — |

- **精度阈值**（按输出 dtype 分档混合容差）：fp16 档 atol 2⁻¹⁴ / rtol 2⁻⁹ / max_abs_error_limit 1e-1 / required_matched_ratio 0.99；bf16 档 atol 2⁻¹⁰ / rtol 2⁻⁶ / 1e0 / 0.99；fp32 档（ht/dh0）atol 2⁻¹⁶ / rtol 2⁻¹⁰ / 1e-2 / 0.99
- **通过条件**：`|x−y| ≤ atol + rtol·|y|` 且 max_abs_error ≤ limit 且 matched_ratio ≥ 0.99
- **参考实现**：PyTorch fp64 重写（chunk 级递推，cast 点与上游 Triton 实现一致）

## 性能测试

**测试平台**：Ascend 910B3（20 AIC + 40 AIV，1800 MHz）
**测试配置**：B=1, S=8192, H=64, DK=128, DV=128, chunk_size=64；fwd 输入/输出 fp16、bwd 输入/输出 bf16（g/gk fp32），g + gk 完整门控，默认分支（无 h0/ht，save_new_value=True）

| **性能指标** | **测试结果** |
| --- | --- |
| fwd（do_bench，NPU event） | **2560.5 μs** |
| bwd（do_bench，NPU event） | **3516.8 μs** |
| fwd + bwd 合计 | **6077.3 μs** |
| msprof Task Duration（fwd main / bwd main） | 2573.2 / 3044.7 μs |
| msprof Task Duration（fwd prep / bwd prep） | 32.9 / 382.6 μs |
| 最小 / 最大延迟（跨会话复测） | 6066.7 / 6284 μs（合计） |
| Block Dim / Mix Block Dim | 20 / 40（main），20 / 20（prep） |
| Op Type | mix |

> 预热 20 次后重复 10 次取平均，仅计设备侧 kernel 执行时间（NPU event 计时，
> 不含张量构造、H2D/D2H、JIT 编译与框架调度开销）；msprof 单次采集为佐证口径。

### ⚠️ 性能差距说明

| **平台/方案** | **运行延迟** | **性能差距** |
| --- | --- | --- |
| Ascend NPU（本次 PR） | **6077.3 μs**（fwd+bwd 合计，实测） | — |
| 性能目标 | 3400 μs（fwd+bwd 合计） | **1.79×** |

实测值未达目标，差距的构成已用 msprof 管线画像定位：

- **串行递推的跨核往返等待是主要构成**：chunk 状态链式依赖使每链每 chunk 必需
  2-2.5 次跨核同步往返（实测 ~2.2-2.4 μs/次）；终态画像中 AIC/AIV 双侧各有
  41-51% 的 wall time 处于跨核 flag 互等，而 Cube 计算管线占用仅 6.6-8.4%、
  GM 有效带宽 ~430-500 GB/s（实测上限 1263 GB/s）——带宽与算力均非瓶颈。
- **延迟地板核算**：fwd main ~2.5 ms（实测 2.57）/ bwd main ~2.9 ms（实测
  3.04）+ prep 0.38 + wrapper ~0.15 ≈ 6.0 ms，与实测 6077 μs 吻合，即当前实现
  已贴近该结构下的延迟地板。
- 达标需要同时将往返次数与单次往返延迟各减半：往返次数的压缩受限于上游
  cast 点语义（操作数折叠会改变量化位置，实测引起 dh0 精度回退）；单次延迟
  的压缩依赖核内细粒度同步/无阻塞调度能力，属工具链层改进项。
- 相对初始实现（16180.5 μs），当前版本经门控预计算 kernel、relay ring、
  满批分解、`q̃` 外移等优化累计降低 **62.5%**。
