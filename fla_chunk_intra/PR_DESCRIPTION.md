为 Ascend NPU 新增 `fla_chunk_intra`（KDA / Kimi Delta Attention intra-chunk 前向融合 inter+solve_tril 与反向梯度）算子，语义对齐上游 tilelang `examples/kda/FLA_KDA/fla_chunk_intra.py`（Triton/FLA 参考实现）。

- **Kernel 架构**：8 个 `@tilelang.jit` kernel（fwd 6：`k_inter` → `k_subst` + `k_g` → `k_g2` → `k_y` → `k_ai` 链；bwd 2：`k_stage` → `k_gemm`）。Developer 模式（`alloc_shared`/`alloc_fragment` + 自动同步），threads=1，`pass_configs` 开启 AUTO_CV_COMBINE / AUTO_CV_SYNC / AUTO_SYNC / MEMORY_PLANNING。fwd 包装函数返回 `(Aqk, Akk)` bf16；bwd 包装函数返回 `(dq2, dk2, db2, dg2)` fp32，host 后处理（`db2.sum(0)+db`、dg2 的 reverse chunk-local cumsum）归调用方，与上游 kernel/测试边界一致。
- **计算结构**：fwd 按 (chunk, b×h) 计算 off-diagonal Aqk/Akk 块（衰减 GEMM）、对角块前代求解 `Ai_ss = (I + strict_tril(A_s))^{-1}`（纯 V 行递归）、以及合并块逆 `Ai = (I+G)(I+G²)D⁻¹`（符号折叠恒等式，连续 GEMM）；Aqk 对角与块上三角显式写 0。bwd 把上游逐列顺序循环重表述为仅依赖 (i, j) 的掩码 GEMM：前序块 `dq2_pre/dk2_pre`、对角统一掩码、转置侧 `dkt`，尾段做 post-β 缩放与梯度累加。
- **数据流与内存管理**：跨 kernel GM workspace 采用预分裂 bf16 hi/lo 对（与 fp32 等字节），消费 kernel 的 C 段直接 GM→L1 读取（无 UB→L1 relay）；bwd 由纯 V 的 staging kernel 把全部 GEMM 操作数 tile 写入 GM scratch（中点锚点衰减定界），Cube 侧整 tile Nd2Nz 读取，结果经 L0C→GM 直写（fp32）+ 手动 FIX 跨核 flag 交接，V 侧读回做尾段；`[B,S,H]` beta 载入使用 flat-view span + arith_progression + reinterpretcast + tile.gather 链（sub-32B stride DMA 不可靠）。
- **关键优化策略**（基线 19.77ms → 最终 11.44ms，-42.1%，全部 A/B 同会话实测）：
  1. fwd k_inter 锚点消去超级 GEMM（逐 s2 锚点在乘积内精确相消，18 → 6 gemm，该 kernel -31.5%）；
  2. bwd 中点锚点超级 GEMM staging（16 → 4 逻辑 gemm，exp 元素计算 -84%，bwd 整体 -55.5%）；
  3. fwd 预分裂 hi/lo workspace 直读链（消费 kernel 直读消中继，I+ 折叠为 V 尾段精确 fp32 加法，fwd chain -18.2%）；
  4. k_g 拆分为纯 V 前代 kernel + 无 relay GEMM kernel（该阶段 -28.5%）；
  5. k_inter fixed-core 任务循环 + 显式 L2 常驻 relay（该 kernel -11.9%）。
- **数值精度处理**：fwd 以 bf16 hi/lo 拆分操作数在 L0C 累加（bf16 输出门限 2⁻⁶ 内）；bwd 全链 fp16 hi/lo 3-gemm 模拟 fp32 GEMM（gemm_v0 的 fp32 操作数在当前工具链损坏）；累加与 I+ 折叠在 fp32 域；`2^x` 以 `exp(x·ln2)` 实现（exp2 无昇腾后端映射）。已知限制：操作数超出 fp16 范围（|x| > 65504）会损失精度；中点锚点把 bwd 操作数指数包络压至 ≤2^12.2，常规衰减域内无溢出。
- **硬件与实现约束**：S 需被 chunk_size(64) 整除、chunk_size 被 sub_chunk_size(16) 整除；输入需 contiguous；fwd dtype 契约固定（q/k/beta bf16，gk/Akk_diag fp32），bwd 全 fp32；H=1 时 db2 走平坦视图单行直写（32B 槽散布约定下 [64,1] 散写仅在 H≥2 自补偿）。跨核 ffts 事件仅 16 槽且 11-14 系统保留（CANN 9.2.0 `GetffstMsg` `flagId & 0xf`），故 bwd 以 kernel 边界作 V→C 同步、尾段用手动 FIX flag（ID 9/10）。
- **测试验证**：交付测试含 4 个代表性用例（L0 最小门槛 + L1 benchmark shape × fwd/bwd），PyTorch fp64 golden（fwd 为单位下三角块求解；bwd 为统一掩码直接式，host 后处理两侧同构）。输入构造对齐上游测试样例：fwd `gk = logsigmoid(randn)` 后 chunk 内 cumsum、`Akk_diag = ones`；bwd 全 randn（g 未经 cumsum）。完整开发期分层套件（L0/L1/L2/Boundary 含正确拒绝与非有限域）共 90 项检查全部通过。

## 精度验证 ✅

**测试配置**：fwd `B=1, S=8192, H=64, DK=128, chunk_size=64, sub_chunk_size=16`（bf16 in/out, fp32 gate/accum）；bwd `B=1, S=8192, H=8, DK=128`（全 fp32）；及 L0 最小 shape `(1,64,2,128)`

| **测试层级/类别** | **用例数** | **检查项数** | **测试结果** | **精度指标** | **最大绝对误差** |
| --- | --- | --- | --- | --- | --- |
| L0 fwd 最小门槛 (1,64,2,128) | 1 | 2（Aqk/Akk） | PASS | matched_ratio=1.0000 | 7.63e-06 |
| L0 bwd 最小门槛 (1,64,2,128) | 1 | 5（dq2/dk2/dg2/db/dg） | PASS | matched_ratio ≥ 0.9998 | 3.05e-04 |
| L1 fwd benchmark shape | 1 | 2（Aqk/Akk） | PASS | matched_ratio=1.0000 | 1.25e-01（bf16 档内） |
| L1 bwd benchmark shape | 1 | 5（dq2/dk2/dg2/db/dg） | PASS | matched_ratio ≥ 0.9999 | 6.41e-04 |
| **总计** | **4** | **14** | **全部通过** | — | — |

- **精度阈值**：bf16 输出 atol 2⁻¹⁰、rtol 2⁻⁶、max_abs_error_limit 1e0、matched_ratio ≥ 0.99；fp32 输出 atol 2⁻¹⁶、rtol 2⁻¹⁰、max_abs_error_limit 1e-2、matched_ratio ≥ 0.99
- **通过条件**：双门限——逐元素 `|actual − golden| ≤ atol + rtol·|golden|` 的 matched_ratio ≥ 0.99，且整体 max_abs_error ≤ limit；inf/nan 结构比对
- **参考实现**：PyTorch fp64 重写 golden（fwd `torch.linalg.solve_triangular` 单位下三角块求解 + 块逆展开交叉校验；bwd 统一掩码直接式 + 上游逐列循环直译交叉校验，两方向 golden 双路互拍一致）
- **环境复验**：同一完整测试套件在分设环境（910B3 + 22 vCPU ARM host）全量重跑：90 项检查全部通过（0 FAIL / 0 WARN），与上表一致

## 性能测试

**测试平台**：Ascend 910B3
**测试配置**：fwd `B=1, S=8192, H=64, DK=128` bf16 in / fp32 gate；bwd `B=1, S=8192, H=8, DK=128` fp32

### 纯 Kernel 时间（主口径：msprof 算子级 Task Duration）

| **性能指标** | **测试结果** |
| --- | --- |
| fwd（6 kernel 合计） | **10395.1 μs**（k_inter 3041.7 / k_subst 2822.3 / k_g 1170.4 / k_g2 1055.6 / k_y 1187.8 / k_ai 1117.3） |
| bwd（2 kernel 合计） | **793.0 μs**（k_stage 221.2 / k_gemm 571.8） |
| **fwd + bwd 全部 kernel 合计** | **11188.1 μs** |
| threads | 1（Developer 模式默认 C/V 划分） |

> 计时方式：`msprof op` 算子级采集的 Task Duration——设备侧硬件计数器记录的单 kernel 实际执行时长，不含任何 host 提交、同步或框架开销。相对首个收敛实现（19741.9 μs，do_bench 口径）整体 **-42%**。

### 上游测试样例对齐口径（补充：do_bench 事件计时）

| 环境（NPU 同型号 1×910B3，host 不同） | fwd chain | bwd chain | 合计 |
| --- | --- | --- | --- |
| x86 host | 10159.3 μs | 1283.5 μs | **11442.8 μs**（4 次 11408–11456） |
| 22 vCPU ARM host（分设环境） | 10429.2 μs | 1676.1 μs | **12105.3 μs**（4 次 11994–12187） |

> NPU 事件计时，预热 20 次、重复 10 次取平均（对齐上游 tilelang 测试样例 `do_bench`）。两环境 kernel 代码相同、NPU 同型号：差异集中在 bwd——分设环境 host 每次链调用需 ~1.45ms（JIT 适配层提交 + 运行时开销），超过 bwd 的设备工作量（793 μs），事件窗口逐次包含设备等待 host 提交的空隙（均值由 host 提交速率主导）；fwd 设备工作量 10.4ms 远大于 host 提交速率，不受影响（+2.6%）。纯 kernel 时间（主口径）两环境一致，设备侧无退化。

### ⚠️ 性能差距说明

| **平台/方案** | **运行延迟** | **性能差距** |
| --- | --- | --- |
| Ascend 910B3（本次 PR，纯 kernel 实测） | **11188.1 μs** | — |
| 唯一 IO 带宽下限（1044MB ÷ 实测 1263GB/s D2D，估算） | 826 μs | 实测为下限的 13.5× |
| 当前结构合成地板（4.4GB GM 流量@600GB/s + 前代 V 语义地板 2.5ms + bwd 传输地板，估算） | ~11 ms | 实测 ≈ 合成地板的 1.02× |

本算子接口契约决定的最小数据搬运量为 1044MB（fwd 706MB，其中 gk fp32 256MiB 为接口契约 + bwd 338MB），910B3 实测 HBM 读写上限 1263GB/s 对应 826us 物理下限；要达到该下限需要单 kernel 多 C 段（或单体融合）支持。当前工具链每个 kernel 仅支持一个 C 段，fwd 被迫拆为 6 kernel、GM 流量放大约 6×（~4.4GB vs 706MB 唯一 IO）；fp32 `gemm_v0` 操作数损坏迫使全链 hi/lo 拆分（GEMM 数 ×3）；前代求解的行递归串行依赖是 V 侧语义地板（~2.5ms）；threads=2 实测 +47%（两方向独立测量）。以上构成剩余差距的主要成分，属工具链/结构性约束而非 kernel 层可实现项；bwd 方向（793.0μs 纯 kernel）已低于其 803MB@~600GB/s ≈ 1.34ms 的传输地板估计（片上缓存命中更优）。上述下限/地板均为按实测带宽与实测流量的估算值，非实测延迟。
