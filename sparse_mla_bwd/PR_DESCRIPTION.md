# PR Description — sparse_mla_bwd（草稿：分支 feat/sparse-mla-bwd-pr @ fc62b3f4，PR 未创建）

为 Ascend NPU 新增 `sparse_mla_bwd`（DSA 稀疏 MLA 反传：dQ GEMM 链 + P/dP 向量链 + dKV 随机行散射）算子。迁移自主仓 GPU 实现（`ref_gpu/sparse_mla_bwd.py`），交付于 `examples_experiment/dsa/`（kernel 1114 行 + test 189 行），已登记 `ci/operator_test_manifest.yaml`。

- **Kernel 架构**：Developer 形态 pair-loop 主路径 + Expert fallback（DESIGN rev3）。k1 已折叠入 bwd；k2 主循环以 indexer_bwd 同款 pair-loop 生产/消费错位结构实现，**零 Expert API**（`alloc_shared`/`alloc_fragment` + workspace 命名自动握手 + 4-True `pass_configs`，AUTO 栈对该结构编译出 C 领先软件流水）；k3 postprocess 亦 Developer 形态。Expert 手动 flag 仅为锁步复现时的升级路径（升级判据：复测 msprof AIC 等 flag 占比 > 30%，§4.2）。Op Type = mix，Block Dim 20 / Mix 40。
- **计算结构**：双后端硬性要求（ascendc 默认 + pto 显式），均须通过编译 + 运行 + 精度验证；性能以 ascendc 考核、pto 如实报告不设门槛。两遍结构：pass-1 V 侧 gather KV 行 + P/dP 链（hi/lo 拆分），pass-2 C 侧 GEMM（dQ = dO·Vᵀ 链）；dKV 经 fp32 GM 原子散射（每行平均被 ~448 个 token 贡献）。
- **数据流与内存管理**：跨核 transit 走 GM workspace（`ws_kv` 按块暂存 `[grid, NI+1, BN, DDT]` + scratch 槽配对 dummy 写；`ws_p`/`ws_dpb` hi+lo 通道；`ws_dkv` fp32 原子散射缓冲）；C 侧 GEMM 操作数 L1 驻留、fp32 累加 L0C；UB 段序 ~184 KB 分时。dKV 散射策略经随机行散射寻址能力查证后选型（DESIGN §6 专题）。
- **关键优化策略**（调优轨迹 79395→27388 μs，**-65.5%**）：（1）iter-0 `ws_kv` 按块暂存——删两遍结构的第二遍冗余 gather（msprof 实证 AIC 86% 时间等 flag），-29.1%；（2）iter-1 gather full+mini staging（320 微写+256 握手/块 → 2 语句）+ scatter 载荷瘦身（pos 改读 idx_ub UB 副本消除同步 GM 标量读），-42.2%；（3）iter-2 纯 V 侧 GM transit 零 flag 机制（源码级发现：combinecv 只遍历 cube 侧名单）+ P/dP_raw V-only fp32 stash + k3 行宽 16→32 + mask/bias 向量化，-12.4%；（4）iter-3b 微调 -4.8%；（5）iter-4 2-token stagger 消 id14——实测 -0.77% < 3% 止损阈按规则回滚（重叠型等待证伪）。
- **数值精度处理**：bf16 GEMM + fp32 累加；P/dP 链 hi/lo 拆分保精度；dKV fp32 原子散射（ascendc/pto atomic 语义差异已查证，DESIGN R5）；dQ 各轮优化后逐位一致验证。
- **硬件与实现约束**（本轮平台发现，已固化 kernel docstring）：跨核 flag 对数 ≤ 16（第 17 对 id 16 混叠 flag 0 → 读陈旧 workspace，NI=1 同值掩盖 / NI=4 dQ 28% dkv 99% 损坏）；3D-of-4D region `T.copy` 静默截断；640 宽 stash 编译 SEGV（NI=4）；`copy_ub_to_ub` 非整行 region 线性化 bug（中转必须整形容器）；UB 196352 B 上限且规划器不跨 symbolic 任务循环回收槽位。
- **测试验证**：分层测试（L0/L1/L2/Boundary，双后端）+ 覆盖门禁 + 显式 delta / oob / neg1 / repeat / varlen / kvgroup2 专项用例。

## 精度验证 ✅

**测试配置**：主 shape B=1, S=2048, H=64, HKV=1, DQKV=576（D=512, D_tail=64）, DV=512, topk=512, bf16；输入 = 主仓口径生成（seed=0 randperm 因果采样），(o, lse) = 测试内 torch fallback。

| **测试层级/类别** | **用例数** | **检查项数** | **测试结果** | **精度指标** | **最大绝对误差** |
| --- | --- | --- | --- | --- | --- |
| L0 门槛（双后端） | 9 | 9 | ✅ 9/9 PASS | 混合容差双门限 | — |
| L1 功能（双后端，含显式 delta/oob/neg1/repeat/varlen/kvgroup2） | 15 | 15 | ✅ 15/15 PASS | 混合容差双门限 | — |
| L2 异常（正确拒绝） | 9 | 9 | ✅ PASS | — | — |
| Boundary | 4 | 4 | ✅ PASS（WARN 与 stock 逐位一致） | — | — |
| 覆盖门禁（coverage_check） | — | 24 | ✅ 24 PASS / 0 MISS / 0 N/A | — | — |

- **精度阈值**：混合容差双门限（atol / rtol / max_abs_error_limit / required_matched_ratio 按 dtype 派生）；校准层口径见 DESIGN §12（rev3 修订后）
- **通过条件**：`--level all` "Test Passed!"（分层全 PASS + 覆盖门禁 0 MISS + boundary WARN 与基线逐位一致）
- **参考实现**：主仓 PyTorch 参考（`ref_gpu/sparse_mla_bwd.py` 语义 + 测试内 torch fallback）

## 性能测试

**测试平台**：Ascend 910B3（A2）
**测试配置**：全接口口径 `sparse_mla_bwd(..., delta=None)`（k2 + k3 两段发射）；主 shape 同上；`tilelang.profiler.do_bench(warmup=250, rep=100)` mean。

| **性能指标** | **测试结果** |
| --- | --- |
| 全接口（ascendc，考核路径） | **27387.6 μs**（iter-3b 采纳基线；同窗口 stock 复核 27211.3 μs；会话级漂移 ±2% 显式记录） |
| 基线（Stage 3 iter-0 实测） | 79395.3 μs（pto 80004.2 μs）——累计 **-65.5%** |
| pto 全接口（如实报告，不设门槛） | iter-1 时点 -42.6% 同向改善（终态未单测，考核路径已判定） |
| Block Dim / Mix Block Dim | 20 / 40 |
| Op Type | mix |

> 计时口径：do_bench warmup=250 rep=100 mean，全接口（两段发射）；全程 NPU 空闲串行（每次计时前 npu-smi 确认）；采纳结论一律以同窗口背靠背 A/B（stock↔current 交替）为准。

### ⚠️ 性能差距说明

| **平台/方案** | **运行延迟** | **性能差距** |
| --- | --- | --- |
| Ascend 910B3（本次 PR，ascendc 全接口） | **27387.6 μs** | — |
| 任务书性能目标（全接口） | < 7680 μs | **未达标，3.6×**（如实披露） |

- 缺口根因（完整 [DESIGN_ERROR] 证据链，见 `perf_tuning/`）：AUTO 栈四条路线全部触墙——字节级（fp16/bf16 槽双减宽均负收益：瓶颈是每行/每 chunk 的发射、事件、屏障而非带宽）、事务级（V 侧微事务发射结构为地板）、结构级（iter-2 剩余 scatter 批量读回被 UB 上限 + 槽位回收封死）、时序级（iter-4 stagger 重叠等待证伪）。实测读量下限 ~12 ms/AIV + V 管道功 ~19 ms > 目标 2.5×。
- 弥合路径（未实施，属架构级）：数据流级削减——transit 融合 / C 侧驻留 / 精度门控 bf16 transit（需 Orchestrator 裁决升级 Expert/混合模式）。
