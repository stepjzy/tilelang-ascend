为 Ascend NPU 新增 `sparse_mla_bwd`（DSA 稀疏 MLA 反传：dQ GEMM 链 + P/dP 向量链 + dKV 随机行散射）算子。参考 GPU 实现迁移，交付于 `examples_experiment/dsa/`（kernel + test），并登记 `ci/operator_test_manifest.yaml`。

- **Kernel 架构**：Developer 模式 pair-loop 主路径（零 Expert API：`alloc_shared` / `alloc_fragment` + workspace 命名自动握手 + 全部 4 项 `pass_configs`，编译器对该结构生成 C 领先的软件流水），并预留 Expert 手动 flag 升级路径。前向结果 k1 已折叠入本算子；k2 主循环采用 pair-loop 生产/消费错位结构；k3 为 postprocess kernel。Op Type = mix，Block Dim 20 / Mix 40。**AscendC（默认）与 PTO（显式 target）双后端**均通过编译、运行与精度验证；性能以 AscendC 考核，PTO 如实报告。
- **计算结构**：两遍结构——pass-1 V 侧 gather KV 行并计算 P / dP 链（hi/lo 双通道拆分），pass-2 C 侧完成 dQ 相关 GEMM；dKV 通过 fp32 GM 原子散射写出（考核 shape 下每行平均被 ~448 个 token 贡献）。散射策略基于随机行散射寻址能力的查证选型。
- **数据流与内存管理**：跨核中转经 GM workspace：`ws_kv` 按块暂存 `[grid, NI+1, BN, DDT]`（NI 个逐块 stash 槽 + scratch 配对槽）；P / dP 链以纯 V 侧 fp32 stash 中转（不占用跨核 flag 预算）；`ws_dkv` 为 fp32 原子散射缓冲。C 侧 GEMM 操作数 L1 驻留、fp32 累加 L0C；UB 段序约 184 KB 分时复用。
- **关键优化策略**（累计 79395 → 27388 μs，**-65.5%**）：（1）`ws_kv` 按块暂存消除两遍结构中第二遍的冗余 gather（profiling 实测 AIC 86% 时间在等待该中转，消除后 -29.1%）；（2）gather 整 2D 批量写 + scatter 载荷改读 UB 索引副本（消除同步 GM 标量读），-42.2%；（3）P/dP 链 V-only fp32 stash（pass-2 重算与 mask 链读全部删除）+ postprocess 行宽 16→32 + mask/bias 向量化，-16.6%。dQ 各步优化后与优化前逐位一致。
- **数值精度处理**：bf16 GEMM + fp32 累加；P/dP 链 hi/lo 双通道拆分保精度；dKV fp32 原子散射（AscendC / PTO 的 atomic 语义差异已适配）。
- **硬件与实现约束**（实测定位并固化于 kernel docstring）：跨核 flag 对数上限 16——第 17 对的 flag id 会混叠到 flag 0，导致读到陈旧 workspace（NI=1 时被同值掩盖，NI=4 时 dQ 28% / dkv 99% 损坏）；3D-of-4D region 的 `T.copy` 静默截断；超宽 stash 触发编译器崩溃；`copy_ub_to_ub` 非整行 region 需整形容器中转；UB 196352 B 上限且规划器不跨 symbolic 任务循环回收槽位。
- **测试验证**：分层测试（L0 / L1 / L2 / Boundary，双后端）+ 接口覆盖检查 24 项 + 显式 delta、越界、-1、重复索引、varlen、kv_group=2 专项用例，全量通过。

## 精度验证 ✅

**测试配置**：主 shape B=1, S=2048, H=64, HKV=1, DQKV=576（D=512, D_tail=64）, DV=512, topk=512, bf16；输入按参考口径生成（seed=0 randperm 因果采样），(o, lse) 由测试内 torch fallback 计算。

| **测试层级/类别** | **用例数** | **检查项数** | **测试结果** | **精度指标** | **最大绝对误差** |
| --- | --- | --- | --- | --- | --- |
| L0 门槛（双后端） | 9 | 9 | ✅ 9/9 PASS | 混合容差双门限 | — |
| L1 功能（双后端，含 delta / 越界 / -1 / 重复 / varlen / kvgroup2 专项） | 15 | 15 | ✅ 15/15 PASS | 混合容差双门限 | — |
| L2 异常（正确拒绝） | 9 | 9 | ✅ PASS | — | — |
| Boundary | 4 | 4 | ✅ PASS（WARN 与基线逐位一致） | — | — |
| 接口覆盖检查 | — | 24 | ✅ 24 PASS / 0 MISS / 0 N/A | — | — |

- **精度阈值**：混合容差双门限（atol / rtol / max_abs_error_limit / required_matched_ratio 按 dtype 派生）
- **通过条件**：分层全 PASS + 覆盖检查 0 MISS + boundary WARN 与基线逐位一致
- **参考实现**：PyTorch 参考（`ref_gpu/sparse_mla_bwd.py` 语义 + 测试内 torch fallback）

## 性能测试

**测试平台**：Ascend 910B3（A2）
**测试配置**：全接口口径 `sparse_mla_bwd(..., delta=None)`（k2 + k3 两段发射）；主 shape 同上；`tilelang.profiler.do_bench(warmup=250, rep=100)` 均值。

| **性能指标** | **测试结果** |
| --- | --- |
| 全接口（AscendC，考核路径） | **27387.6 μs**（同窗口基线复核 27211.3 μs；会话级漂移 ±2% 已显式记录） |
| 初始实现全接口（AscendC / PTO） | 79395.3 / 80004.2 μs —— 累计 **-65.5%** |
| PTO 全接口 | 与 AscendC 同向改善（如 -42.6% 时点实测），终态未单独计时（考核路径已判定） |
| Block Dim / Mix Block Dim | 20 / 40 |
| Op Type | mix |

> 计时口径：do_bench warmup=250 / rep=100 均值，全接口（两段发射）；全程 NPU 空闲串行（计时前经 npu-smi 确认）；性能结论均以同窗口新旧版本交替对照为准，排除会话漂移误判。

### ⚠️ 性能差距说明

| **平台/方案** | **运行延迟** | **性能差距** |
| --- | --- | --- |
| Ascend 910B3（本次提交，AscendC 全接口） | **27387.6 μs** | — |
| 性能目标（全接口） | < 7680 μs | **未达标，3.6×**（如实披露） |

- 缺口根因（实测证据完整）：瓶颈在 V 侧逐行串行发射结构（每行的发射、事件与屏障开销，而非带宽——fp16 / bf16 中转减半实测均为负收益佐证）；实测读量下限 ~12 ms/AIV + V 管道功 ~19 ms，超过目标的 2.5×。字节级（中转减宽）、事务级（批量发射）、结构级（scatter 批量读回，受 UB 上限与槽位回收限制）、时序级（跨块错位流水，实测收益 0.77% 在噪声带内）四个方向的优化空间均已实测探明并触顶。
- 弥合路径（架构级，超出本次变更范围）：数据通路级削减——transit 融合 / C 侧驻留 / 精度门控 bf16 中转，需配合 Expert 手动同步实现。
