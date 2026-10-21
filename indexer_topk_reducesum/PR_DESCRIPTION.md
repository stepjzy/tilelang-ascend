# PR Description — indexer_topk_reducesum（草稿：分支 feat/indexer-topk-reducesum-pr @ 4132da0f，PR 未创建）

为 Ascend NPU 新增 `indexer_topk_reducesum`（DSA 稀疏微调 lightning indexer 前向：topk 概率质量头归约，GEMM + relu + 加权归约 + 因果 mask + topk + softmax 融合）算子。迁移自主仓 GPU 实现（`ref_gpu/indexer_topk_reducesum.py`），交付于 `examples_experiment/dsa/`（kernel 1447 行 + test 135 行），已登记 `ci/operator_test_manifest.yaml`。

- **Kernel 架构**：双 kernel（fast/generic）+ host 分派（indexer_bwd 双路径先例）。Developer 模式，4-True `pass_configs`（`TL_ASCEND_AUTO_CV_COMBINE` / `AUTO_CV_SYNC` / `AUTO_SYNC` / `MEMORY_PLANNING`，DESIGN §4.7 锁定）。fast 路径 persistent GRID=20（910B3 cube core 数），任务粒度 QG=8 token/组，striped 轮询均衡因果三角负载；generic 路径为全语义兜底（merge_sort，仅 fp32）。Op Type = mix（MIX_AIC_1_2，AIC cube/mte1/mte2/fix + AIV vec/mte2/mte3）。
- **计算结构**：`(m,h)` 展平 GEMM：`[QG*H=512, D] × [D, BN=32]` per 组 per 块（8320 次大矩阵，固定开销摊薄 16×；数学恒等：logits[(u,h),n] = Q[u,h,:]·K[n,:]，GEMM 累加顺序变化仅在 fp32 内）。编译期上界 NBLK=CAP/BN + `if k_st <= t_max` 守卫（空转块仅 1 次 if 判断，守卫内不发射 GEMM/拷贝）。因果 mask 仅在 straddle 块执行（每 token 仅 ~1 块跨越因果对角，其余块全有效免 mask 或全无效免整条链）。s≤0 时保留 per-block 融合 Muls 回退分支（sm_scale 为编译期常量，Python 分支零运行时成本）。
- **数据流与内存管理**：中转 `ws_logits`（bf16）跨核 GM workspace；转置 GEMM `[BN, QG*H]` 布局使 relay 读 32×512B burst、fix 写 32×1KB burst（原 64B burst，mte2 有效带宽 12.6→85.5GB/s）；score_row fill(-inf) + topk pad 语义（idx>i_u→-1 修正，与 GPU kernel L180-183 同款）。
- **关键优化策略**：（1）V 侧向量链融合为 half-tile 大 op——基线每 token 13 条向量指令 + ~12 条 PipeBarrier_V，融合后指令数与 barrier 数 ÷4，逐位等价；（2）straddle-only 因果 mask（arith_progression + compare + select 向量化，gqa_sink 先例）；（3）sm_scale 移位到 per-task [CAP] 行乘（sum 后乘，与 golden 逐位一致，省 NBLK×[QG2H,BN] 次 Muls）；（4）score_2d [QG2,BN] 中转 + fill(-inf) + 融合 4 行 store（copy_ub_to_ub dstGap）；（5）host 低开销化：单同步 + numpy 构表 + 内容键缓存 + `_CachedLauncher`（full interface 4776→1594us，-66.4%）。
- **数值精度处理**：数值基线 = 复刻 golden 数值路径（bf16 logits round 中转 cast + fp32 头累加 + sm_scale sum 后乘 + fp32 softmax），比 GPU kernel（全程 fp32 logits、sm_scale 预乘 q）更贴近 golden；残差仅剩 golden 独有的 bf16 乘积 round（相对误差 ≤ 2⁻⁸）。
- **硬件与实现约束**：generic 的 merge_sort 仅支持 fp32（ascend_tile.py:525）；compare 要求 src 256B 对齐（H*BN*4，H≥8 时满足）；4-True pass_configs 配置锁定；topk/merge_sort tmp 显式注入备用（InjectTmpBuffer 理论风险）。
- **测试验证**：分层测试（L0/L1/L2/Boundary）+ 覆盖门禁 + s≤0 回退分支补测；AscendC 与 PTO 双后端（`--level all --backend both`）。

## 精度验证 ✅

**测试配置**：主 shape B=1, S=2048, H=64, D=128, topk=64, bf16, offsets=[0, 2048]（l0_main，主仓测试输入原样复刻，seed 42）；全量分层 `--level all`。

| **测试层级/类别** | **用例数** | **检查项数** | **测试结果** | **精度指标** | **最大绝对误差** |
| --- | --- | --- | --- | --- | --- |
| L0 门槛 | 24 | 24 | ✅ PASS | 全过 | — |
| L1 功能 | 42 | 42 | ✅ PASS | 全过 | — |
| L2 异常（正确拒绝） | 13 | 13 | ✅ PASS | — | — |
| Boundary | 4 | 4 | ✅ PASS | — | — |
| s≤0 回退分支补测（sm_scale=-0.5/0.0） | — | — | ✅ PASS | mask 域索引 matched 1.0000 | max_abs ~6e-8 |
| 覆盖门禁（coverage_check） | — | 24 | ✅ 24 PASS / 0 MISS / 0 N/A | — | — |
| PTO 后端全量（--level all --backend both） | — | 132 | ✅ 132 PRECISION_PASS / 0 FAIL / 0 WARN | — | — |

- **精度阈值**：bf16/fp32 混合容差（DESIGN §9.3，与 golden 数值路径对齐论证）
- **通过条件**：分层全 PASS + 覆盖门禁 0 MISS
- **参考实现**：PyTorch golden（DESIGN §9.1，主仓数值路径）

## 性能测试

**测试平台**：Ascend 910B3（A2）
**测试配置**：l0_main：B=1, S=2048, H=64, D=128, topk=64, bf16, offsets=[0, 2048]

| **性能指标** | **测试结果** |
| --- | --- |
| fast 纯 kernel（torch.npu Event，5 warmup + 20 iters，均值） | **1142.7 μs**（3 次复测 1142.8 / 1142.7 / 1145.3 / 1161.5） |
| msprof fast Task Duration（交叉验证） | 1129.9 μs |
| PTO fast 纯 kernel（清态 4 次独立进程复测） | **1925.5 μs**（1925.5 / 1926.9 / 1930.4 / 1932.7，簇宽 0.4%；较 PTO 基线 6554.1 μs **-70.7%**） |
| full interface（AscendC） | 1594-1617 μs（host 开销 ~430 μs） |
| generic 纯 kernel（不考核） | 3104-3455 μs（kernel 未动，持平） |
| Block Dim | fast persistent GRID=20 |
| Op Type | mix（MIX_AIC_1_2） |

> AscendC 计时口径：torch.npu Event，5 warmup + 20 iters；msprof Task Duration 读数 1129.9 μs 与事件计时一致（-1.1%）。全程 NPU 空闲串行。调优轨迹：基线 2022.1/2002.9 μs → 终态 1142.7 μs（**-43.5%**），三轮采纳（V 链融合 -10.7% → 转置 GEMM 宽 burst -35.5% → host 低开销化）。

### ⚠️ 性能差距说明

| **平台/方案** | **运行延迟** | **性能差距** |
| --- | --- | --- |
| Ascend 910B3（本次 PR，fast，AscendC） | **1142.7 μs** | — |
| 任务书性能目标（fast 纯 kernel） | < 1428 μs | **达标**（低于目标 20%） |

- 主仓 GPU kernel 存在但未做同口径实测对比，无 GPU 基线数据（如实说明，不引用估计值）。
- 剩余优化空间（未做，见 final_summary）：barrier 税（估算 ~30-40% vec 相邻时间）、AIC 2-ahead + relay ping-pong 软件流水、topk 段审计。
