为 Ascend NPU 新增 `fla_wy_fast`（KDA/Kimi Delta Attention chunked WY 表示，前向 + 反向）算子。

- **Kernel 架构**：前向 2 个 + 反向 6 个共 8 个 `@tilelang.jit` kernel，Developer 模式（`alloc_shared`/`alloc_fragment` + 自动 C/V 划分与同步，`threads=1`），grid 一维化为 NT×B×H（benchmark shape 下前向 8192 / 反向 4096 block）；pass_configs 开启 AUTO_CV_COMBINE / AUTO_CV_SYNC / AUTO_SYNC / MEMORY_PLANNING。
- **计算结构**：前向 `u = A_c·(β⊙v)`、`w = A_c·(β⊙k⊙2^g)`、`kg = k·2^(g_n−g)`；反向 `dA = −strict_tril(A_cᵀ·S·A_cᵀ)`（其中 `S = strict_tril(dw·K_bgᵀ + du·V_bᵀ)`），`dk2/dv/dbeta/dg2` 按链式法则逐项展开。`2^g` 以 `exp(g·ln2)` 实现（exp2 在昇腾后端无映射）。反向因单 kernel 仅支持一个 cross-core 流水体而拆为 k1/k2/v1/v2/z/da 六段，跨段累加量经 GM 交接。
- **数据流与内存管理**：GM→UB→L1→L0A/L0B→L0C 逐级搬运；`β[B,S,H]` 经 flat 视图 + gather 链加载（子 32B strided DMA 不可靠）；dbeta 散列经 GM scratch 中转并附显式 MTE3→MTE2 flag 同步；strict-tril 常量按 (device, chunk) 在 host 侧缓存。
- **关键优化策略**：以 `T.tile.broadcast`（axis=0/1）+ 全量 tile 向量指令替换逐行串行循环（覆盖 6 个 kernel，单 kernel -26%~-71%，合计 -9.21ms）；strict-tril 常量 host 缓存（-0.97ms）；wrapper 移除内部同步以对齐上游计时口径（-1.37ms）。整体 23.151ms → 11.719ms（-49.4%）。
- **数值精度处理**：fp32 `gemm_v0` 在当前工具链上不可用（隔离用例即产出垃圾值），前向以 bf16 hi/lo 双 gemm 拆分（相对误差 ~2⁻¹⁶，bf16 门限 2⁻⁶）、反向以 fp16 hi/lo 三 gemm 模拟 fp32（~2⁻²²，fp32 门限 2⁻¹⁰）；拆分方案均先经 CPU 模拟按精度门验证后落地。已知限制：反向操作数 |x| > 65504（fp16 表示范围）时精度退化，该边界已入测试并文档化。
- **硬件与实现约束**：要求 S % 64 == 0、chunk_size = 64、DK/DV ∈ {32, 64, 96, 128}（流水体恰好 2 迭代的 block 规则，其余 16 倍数显式拒绝并测试覆盖）；DK/DV = 128 时反向部分 kernel 的 UB 占用接近满额。
- **测试验证**：PyTorch chunk-loop golden 对拍，混合容差双门限判定；L0 门槛 4 用例 + L1 功能 4 用例（含两组 benchmark shape 与 D=96）全部通过，L2 负向 7 用例全部正确拒绝，Boundary 6 用例 5 通过 + 1 预期内 WARN（fp16 拆分范围限制）。

## 精度验证 ✅

**测试配置**：fwd (1,8192,64,128,128) bf16 + fp32 gate；bwd (1,32768,8,128,128) fp32；另含最小 shape 与 D=96 变体

| **测试层级/类别** | **用例数** | **检查项数** | **测试结果** | **精度指标** | **最大绝对误差** |
| --- | --- | --- | --- | --- | --- |
| L0 门槛（规则 shape） | 4 | 16 | 全部通过 | matched_ratio ≥ 0.99 双门限 | fwd 2.50e-1 / bwd 2.44e-3 |
| L1 功能（benchmark shape + D=96） | 4 | 16 | 全部通过 | 同上 | fwd 5.00e-1 / bwd 4.40e-3 |
| L2 负向（非法输入） | 7 | — | 全部正确拒绝 | 抛异常即通过 | — |
| Boundary（合法特殊值） | 6 | 19 | 5 通过 + 1 预期内 WARN | 同上 | 1 例 WARN：g=40 超 fp16 拆分范围（文档化限制） |
| **总计** | **21** | **51** | **阻塞用例全部通过** | — | — |

- **精度阈值**：bf16（atol 2⁻¹⁰，rtol 2⁻⁶，max_abs_limit 1e0）；fp32（atol 2⁻¹⁶，rtol 2⁻¹⁰，max_abs_limit 1e-2）；matched_ratio ≥ 0.99
- **通过条件**：逐元素 `|actual − golden| ≤ atol + rtol·|golden|`，且整体 matched_ratio ≥ 0.99 且 max_abs_error ≤ limit
- **参考实现**：PyTorch chunk-loop golden（语义对齐上游 Triton 参考实现）

## 性能测试

**测试平台**：Ascend 910B3（20 AI Cores，64GB HBM），CANN 9.2.0
**测试配置**：fwd (1,8192,64,128,128) bf16 + fp32 gate；bwd (1,32768,8,128,128) fp32

| **性能指标** | **测试结果** |
| --- | --- |
| NPU 事件计时（前向 kernel） | **4783 μs** |
| NPU 事件计时（反向 6 kernel 串行） | **6935 μs** |
| **前向 + 反向合计** | **11719 μs** |

> 预热 20 次后重复 10 次取平均；NPU 事件仅圈住设备侧 kernel 执行，wrapper 无内部同步（对齐上游 do_bench 口径，不含张量构造、H2D/D2H 搬运、JIT 编译与框架调度开销）。grid：fwd 8192 / bwd 4096 block，threads=1。

### ⚠️ 性能差距说明

| **平台/方案** | **运行延迟** | **性能差距** |
| --- | --- | --- |
| Ascend NPU（本次 PR） | **11719 μs** | — |
| 性能目标（前向 + 反向合计，GPU 级基线口径） | 1370 μs | 8.55× |

性能目标 1370 μs 隐含有效带宽 1.82 TB/s，为本机实测 D2D 带宽上限（1263 GB/s，读+写）的 1.44 倍——该目标在 910B3 上物理不可达（按唯一必需 IO 2.49 GB 与实测带宽推得理论下限约 1.97 ms）。当前 11.719 ms 距实测带宽下限 5.95×，剩余差距的构成为：kernel 拆分导致的输入重复读、C/V 片上无直连路径时的 workspace 中继流量、单发 kernel 固有延迟、host 侧 JIT 分发开销（约 1 ms/launch）。以上均为工具链层限制（fp32 gemm 不可用、单 kernel 单 cross-core 流水体、多 gemm 组 L0 staging 损坏、sync-point 计数失配等）；算子层优化已收敛（多轮无收益尝试后停止），逐项消融数据完整保留于调优记录。
