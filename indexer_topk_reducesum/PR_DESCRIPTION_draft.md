为 Ascend NPU 新增 `indexer_topk_reducesum`（DSA 稀疏微调 lightning indexer 前向：topk 概率质量头归约，融合 GEMM + relu + 加权归约 + 因果 mask + topk + softmax）算子。参考 GPU 实现迁移，交付于 `examples_experiment/dsa/`（kernel + test），并登记 `ci/operator_test_manifest.yaml`。

- **Kernel 架构**：双 kernel（fast / generic）+ host 侧分派。Developer 模式，启用全部 4 项 `pass_configs`（`TL_ASCEND_AUTO_CV_COMBINE` / `AUTO_CV_SYNC` / `AUTO_SYNC` / `MEMORY_PLANNING`）。fast 路径为 persistent kernel（grid = 20，对应 910B3 cube core 数），每任务处理 QG=8 个 token，striped 轮询均衡因果三角负载；generic 路径为全语义兜底（merge_sort 实现 topk，fp32）。Op Type = mix（AIC GEMM/MTE + AIV 向量链，MIX_AIC_1_2）。
- **计算结构**：`(m, h)` 展平 GEMM：`[QG*H=512, D] × [D, BN=32]` per 组 per K 块——以大矩阵摊薄 GEMM 固定开销（相比逐 token 小矩阵约 16×），数学恒等 `logits[(u,h), n] = Q[u,h,:]·K[n,:]`，累加顺序变化仅发生在 fp32 内，被容差覆盖。K 块数采用编译期上界 `NBLK=CAP/BN` + 越界守卫（空转块只做一次判断，不发射 GEMM 与拷贝，结果与精确边界一致）。因果 mask 仅在跨越对角线的块上执行（每 token 约 1 块，其余块全有效免 mask 或全无效免整条链）。`sm_scale` 为编译期常量：`s>0` 时移位到归约后行乘，`s≤0` 时切换 per-block 融合路径（Python 分支，零运行时成本）。
- **数据流与内存管理**：AIC 与 AIV 间经 GM workspace `ws_logits`（bf16）中转；GEMM 采用转置布局 `[BN, QG*H]`，使中转读写均为 512B / 1KB 宽 burst（原始 64B 窄行布局下 MTE 有效带宽仅 12.6 GB/s，转置后 85.5 GB/s）；无效 token 行以 `-inf` 填充（topk pad 语义），输出侧 `idx > i_u → -1` 修正。
- **关键优化策略**：（1）V 侧向量链融合为 half-tile 大算子——原始每 token 13 条向量指令 + ~12 条 PipeBarrier，融合后指令数与 barrier 数降至 1/4，逐位等价；（2）straddle-only 因果 mask（等差数列 + compare + select 向量化）；（3）`sm_scale` 移位至 per-task 行乘，省去每块 `[QG2H, BN]` 次乘法且与参考实现逐位一致；（4）`score_2d [QG2, BN]` 中转 + 4 行融合 store（`copy_ub_to_ub` dstGap）；（5）host 侧单同步 + numpy 构表 + 参数内容缓存，full interface 从 4776 μs 降至 1594 μs。
- **数值精度处理**：bf16 GEMM + fp32 累加；logits 中转保留 bf16 round、`sm_scale` 归约后乘、softmax 全 fp32——数值路径与 PyTorch 参考逐环节对齐（较 GPU 版的"fp32 logits + sm_scale 预乘"更贴近参考），残差仅剩参考独有的 bf16 乘积 round（相对误差 ≤ 2⁻⁸）。
- **硬件与实现约束**：generic 路径 merge_sort 仅支持 fp32；compare 源需 256B 对齐（`H*BN*4`，H≥8 满足）；topk/merge_sort 临时缓冲支持显式注入。
- **测试验证**：分层测试（L0 门槛 / L1 功能 / L2 异常 / Boundary）+ 接口覆盖检查 24 项，AscendC 与 PTO 双后端全量通过，另含 `sm_scale ≤ 0` 回退路径专项补测。

## 精度验证 ✅

**测试配置**：主 shape B=1, S=2048, H=64, D=128, topk=64, bf16, offsets=[0, 2048]（参考实现测试输入复刻，seed 42）；全量分层测试。

| **测试层级/类别** | **用例数** | **检查项数** | **测试结果** | **精度指标** | **最大绝对误差** |
| --- | --- | --- | --- | --- | --- |
| L0 门槛 | 24 | 24 | ✅ PASS | 全过 | — |
| L1 功能 | 42 | 42 | ✅ PASS | 全过 | — |
| L2 异常（正确拒绝） | 13 | 13 | ✅ PASS | — | — |
| Boundary | 4 | 4 | ✅ PASS | — | — |
| s≤0 回退分支补测（sm_scale=-0.5/0.0） | — | — | ✅ PASS | mask 域索引 matched 1.0000 | max_abs ~6e-8 |
| 双后端全量（--level all --backend both） | — | 132 | ✅ 132 PRECISION_PASS / 0 FAIL / 0 WARN | — | — |

- **精度阈值**：bf16/fp32 混合容差（与参考数值路径逐环节对齐）
- **通过条件**：分层全 PASS + 接口覆盖检查 0 MISS
- **参考实现**：PyTorch golden（参考实现数值路径）

## 性能测试

**测试平台**：Ascend 910B3（A2）
**测试配置**：B=1, S=2048, H=64, D=128, topk=64, bf16, offsets=[0, 2048]；fast 路径

| **性能指标** | **测试结果** |
| --- | --- |
| fast 纯 kernel（torch.npu Event，5 warmup + 20 iters，均值） | **1142.7 μs**（3 次复测：1142.8 / 1142.7 / 1145.3 / 1161.5） |
| msprof Task Duration（交叉验证） | 1129.9 μs |
| PTO 后端 fast 纯 kernel（4 次独立进程复测） | **1925.5 μs**（1925.5 / 1926.9 / 1930.4 / 1932.7，簇宽 0.4%） |
| full interface（含 host 开销 ~430 μs） | 1594-1617 μs |
| generic 纯 kernel（通用兜底路径） | 3104-3455 μs（不作为性能路径） |
| Block Dim | 20（persistent） |
| Op Type | mix |

> AscendC 以 torch.npu Event 计时（5 warmup + 20 iters），msprof 交叉验证读数一致（-1.1%）；全程 NPU 空闲串行。相对初始实现的 2022.1 μs，当前 fast 路径累计优化 **-43.5%**（向量链融合、转置 GEMM 宽 burst、host 低开销化）。

### ⚠️ 性能差距说明

| **平台/方案** | **运行延迟** | **性能差距** |
| --- | --- | --- |
| Ascend 910B3（本次提交，fast，AscendC） | **1142.7 μs** | — |
| 性能目标（fast 纯 kernel） | < 1428 μs | **达标**（低于目标 20%） |

- GPU 参考实现未做同口径实测对比，无 GPU 基线数据（不引用估计值）。
- 仍有优化空间：向量链 barrier 开销、AIC 领先 2 对的 relay 软件流水、topk 段审计。
