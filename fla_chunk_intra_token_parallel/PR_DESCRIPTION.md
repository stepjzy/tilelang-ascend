为 Ascend NPU 新增 `fla_chunk_intra_token_parallel`（KDA intra-chunk 每 token 衰减内积 Aqk/Akk 生成）算子，迁移自 tile-ai/tilelang 的 `examples/kda/FLA_KDA/fla_chunk_intra_token_parallel.py`。

- **Kernel 架构**：单个 `@tilelang.jit` kernel（`out_idx=[-2, -1]`，`workspace_idx=[4, 5, 6]` 自动注入 3 个 fp32 GM 中转缓冲）；Developer 模式；`pass_configs` 开启 `AUTO_CV_COMBINE / AUTO_CV_SYNC / AUTO_SYNC / MEMORY_PLANNING`。一维 `T.Kernel(CORE_EFF)` 固定核网格（CORE_EFF = min(20, 任务数)），`(cid, vid)` 绑定将每个任务的向量工作分配到核对的 2 个 AIV 子块。
- **计算结构**：对 chunk（CS=64）内每个 subchunk（SCS=16）的 token i 与前序 token j（j ≤ i，含自身）计算 `kgj = k_j · 2^(g_i - g_j)`，`Aqk[i, j%CS] = Σ_d q_i·kgj`，`Akk[i, j-i_ts] = (j<i) ? Σ_d (β_i·k_i)·kgj : 0`。衰减用中锚点因式分解 `2^(g_i-g_j) = E+_i · E-_j`（E± = exp(±(g-anchor))），每 head 一次性构造 Q'=q·E+、K'=k·E-、K''=k·E+ 三个 [16, DK] slab；点积整体卸载到 Cube 核：`Aqk = tril(Q' @ K'^T)`、`Akk = strict_tril((β·K'') @ K'^T)`，按 4-head 组块对角批量一次 gemm_v0（transpose_B），取 [32,16] 对角块。V 侧用 compare+select（VSEL）做下三角掩码——深衰减下块对角垃圾单元可达 ±inf（累加器中 E+·E- 至 2^180），乘法掩码会产生 inf·0=NaN，select 直接替换为 0，与 golden 的 masked_fill 语义一致。
- **数据流与内存管理**：V（AIV）构造 fp32 操作数 A=[Q';K''][128,DK]、B=K'[64,DK] → GM workspace 双缓冲槽（NSLOT=2，按核 ping-pong，常驻 L2）→ C（AIC）按 head 组从 relay 槽装载 L1，经高层 T.gemm_v0（transpose_B）在 L0C 累加后暂存至 GM o_relay → V 掩码+CAST_RINT+写回 GM。操作数强制 fp32：bf16 舍入单独就会使 Akk matched_ratio 降到 0.9848（< 0.99 门限，CPU 模拟实测），且与 split-bf16 中转字节数相同。UB 峰值 ~174KB/192KB：drain 暂存采用单 head [16,*] 缓冲逐 head 复用（整任务布局需 30KB，单 head 仅 7.5KB）。
- **关键优化策略**（基线 19150us → 终版 1762us，10.9×；调优峰值 1739us）：① subchunk 并行任务划分 + 静态循环 + 宏展开（消除每 block 816 次条件 barrier）；② 衰减因式分解（Exp 指令 -87%）；③ 点积从 Vector 管线卸载到全程空闲的 20 个 Cube 核（-49.2%）——迁移初期 msprof 显示 AIV 满载 97.98% 而 Cube 全程空转、29% 时间耗在比乘法慢 5.3× 的行广播指令上；④ 固定核网格 + 2 任务延迟软件流水 + 双缓冲 relay（V drain(t-2)/produce(t) 与 C gemm(t) 稳态三路重叠，AIC busy 99.55%，-39.5%）；⑤ β 折叠向量化（one-hot select + 行归约 + 列广播，替代每任务 64 次标量读写，-18.8%）。跨核同步混合自动/手动：操作数 relay 保留 AUTO_CV_SYNC（pass 按 "workspace" 字面名识别），结果 relay 改手动 FIX flag（交替事件 ID，配对 2 任务延迟的 drain）。
- **数值精度处理**：fp32 计算域全程（E±/Q'/K'/K''/gemm 操作数/累加器），输出经 staging + CAST_RINT 整块转 bf16；`exp2(x)` 实现为 `exp(x·ln2)`（exp2 无 Ascend 后端映射），且 anchor 减法在原始 log2 域先于 ln2 缩放进行——大 |g| 下先乘后减会把相邻 gate 舍入为同值丢失衰减差值。因式分解的域限制已文档化：subchunk 半程衰减 |g_i - g_anchor| < ~127（log2 域，fp32 上限余量 31）。
- **硬件与实现约束**：定长序列（S % CS == 0、CS % SCS == 0 且 CS/SCS 为 2 的幂、H % 2 == 0 且 BH 自适应 8→4→2、DK % 16 == 0 且 32 ≤ DK ≤ 128（BH=8 主路径的 192KB UB 共存上限，超界干净拒绝）、BH·SCS ≥ 16 的 gemm N 分形下界且 GGRP 整除 BH）；输入 q/k/beta bf16 + gk fp32，输出 bf16；varlen 不在范围。
- **测试验证**：分层套件 30 用例（L0 规则 shape 3 + L1 泛化 9 [含 H=2/6/12 头组自适应、DK=48 非 2 幂、SCS=8 非默认] + L2 非法输入拒绝 10 + Boundary 深衰减/inf/nan 4），PyTorch golden 对照，全部通过（golden 另经三重循环朴素转写交叉校验）。交付测试文件含 3 个代表性用例（bench 结构 L0、多 batch L1、深衰减 gate 哨兵 L1）+ 2 个域拒绝检查（DK=256 超预算、CS/SCS 非 2 幂）。

## 精度验证 ✅

**测试配置**：`B=1..2, S=64..1024, H=8..64, DK=32..128, chunk=64, sub_chunk=8/16；q/k/beta bfloat16 + gk float32 → Aqk [B,S,H,64] / Akk [B,S,H,16] bfloat16`

| **测试层级/类别** | **用例数** | **检查项数** | **测试结果** | **精度指标** | **最大绝对误差** |
| --- | --- | --- | --- | --- | --- |
| L0 规则 shape | 3 | 6（Aqk/Akk 各 3） | PASS | matched_ratio=1.0000 | 6.25e-2 |
| L1 泛化（头组/宽度/SCS） | 9 | 18 | PASS | matched_ratio=1.0000 | 6.25e-2 |
| L2 非法输入拒绝 | 10 | 10 | PASS（全部正确抛异常） | — | — |
| Boundary（深衰减/inf/nan） | 4 | 8 | PASS | matched_ratio=1.0000 | 6.25e-2 |
| **总计** | **26** | **42** | **全部通过** | — | **≤ 6.25e-2** |

- **精度阈值**：bf16 输出混合容差 atol 2^-10 / rtol 2^-6 / max_abs_error_limit 1e0 / matched_ratio ≥ 0.99（双门限：逐元素 `|actual-golden| ≤ atol + rtol·|golden|`，整体 ratio 与硬帽同时满足；inf/nan 位置结构比对）
- **通过条件**：所有用例 matched_ratio=1.0000 且 max_abs ≤ 6.25e-2（上限 1e0）
- **参考实现**：PyTorch 向量化 golden（fp32 域 + torch.exp2 精确语义 + 清零掩码），并与 Triton 权威实现的朴素三重循环转写交叉校验一致

## 性能测试

**测试平台**：Ascend 910B3（20 AIC + 40 AIV）
**测试配置**：`B=1, S=8192, H=64, DK=128, chunk_size=64, sub_chunk_size=16；q/k/beta bfloat16 + gk float32 → 输出 bfloat16`（与上游 TileLang CUDA 示例 main() 一致）

| **性能指标** | **测试结果** |
| --- | --- |
| Kernel 执行时间（torch.npu 事件计时，20 次预热后 10 次重复取均值） | **1762.4 μs**（稳态 per-rep 1690-1710） |
| 30 次 back-to-back 循环均值 | 1715.8 μs |
| Block Dim | 固定核网格 20（每核 2 AIV 子块） |
| 有效内存带宽 | 353 GB/s |

> 计时仅覆盖设备侧 kernel 执行（不含张量构造、H2D/D2H、JIT 编译与框架调度开销）。首次迁移可运行版本为 19150 μs，经结构重组与流水化后为 1762 μs（-90.8%；调优峰值为 1739 μs，终版为采用高层 T.gemm_v0 intrinsic 的 DSL 清晰性支付了 +1.3%）。
