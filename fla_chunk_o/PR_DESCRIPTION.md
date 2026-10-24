为 Ascend NPU 新增 `fla_chunk_o`（KDA / chunked gated linear attention 前向输出 O）算子。

- **Kernel 架构**：单 `@tilelang.jit` kernel（`fla_chunk_o_kernel`，`out_idx=[10]`、
  `workspace_idx=[6,7,8,9]`），Developer 模式 + 一处手动跨核 throttle flag，
  `pass_configs` 开启 `TL_ASCEND_AUTO_SYNC / TL_ASCEND_AUTO_CV_SYNC /
  TL_ASCEND_AUTO_CV_COMBINE / TL_ASCEND_CVT_CASE_BF16`。一维
  `T.Kernel(CORE_EFF=20, threads=2)` fixed-core grid + 核内任务循环：8192 个逻辑
  (chunk, b, h) 任务按 `tid = cid + t * CORE_EFF` 条带化映射到 20 个 Cube 核，
  每核单 block 常驻摊薄派发/TPipe 初始化开销。
- **计算结构**：逐 64-token chunk 计算
  `o = (q·scale ⊙ 2^g) @ h[c] + tril_incl_diag(A) @ v`（`scale = DK^-0.5`，
  `2^g` 以 `exp(g·ln2)` 实现——`exp2` 无 Ascend 后端映射）。inter 项按全 DK
  展开（q/g 每任务一次装载 [64, DK]，hi/lo 两支 K=DK gemm）；intra 项为
  64×64×128 gemm。host wrapper 校验 `S % 64 == 0`、`DK ∈ {64,96,128}`、
  `DV` 为 16 的倍数且 ≤128，域外显式拒绝。
- **数据流与内存管理**：q/v/g/A/h 只读消费（A/h 由上游算子生产）。Q/G/A/O 经
  零拷贝 3D 视图 `[B, S, H*D]` 消费（vid 行分裂落在 S 维）；V/Hstate 由 Cube 侧
  GM→L1 直接消费。V→C 四路中继（qg_hi/qg_lo/a_mask/v）经显式命名的 L2 常驻
  workspace（[20, NSLOT, ...] slot ring，数 MiB）交换，替代自动注入的
  ~768MiB/launch workspace。累加在 L0C fp32，输出由 Cube 侧 fixpipe 直写
  strided O 行（fp32→bf16 转换随 FIX 指令完成）。
- **关键优化策略**（B=1,S=8192,H=64,DK=DV=128,chunk=64 基准：3.285ms → **1.417ms，-57%**）：
  fixed-core 任务循环 + L2 常驻中继（消除每 block 派发与 workspace 注入）；
  3-chunk 批处理软件流水（每批同步脚手架摊薄 3 倍，Vector 落后 Cube 两个批次）；
  分支无关尾处理（越界迭代经 Select 重定向至本核首任务，循环内零 IfThenElse——
  AUTO_SYNC 会在每个 IfThenElse 周围排空全管线）；v 中继走 Vector 侧
  （strided 16GB/s → 连续 48GB/s）；按名 ping-pong 的 L1 操作数解除同名
  WAR 串行；host 快速发射路径缓存编译实例与中继 workspace（通用 JIT 适配层
  每次调用重建参数校验与 TVM Arith Analyzer，~0.8ms host 开销在队列为空时
  完全暴露，快路径降至 ~0.16ms，官方均值再 -75us）。
- **数值精度处理**：qg 拆分为 bf16 hi/lo 两支 gemm 累加（~2^-18 相对误差）——
  单次 bf16 cast（~2^-9）在相消元素（|o| ≪ |o_inter|）上实测 matched_ratio
  0.988，低于 bf16 门禁 0.99；gate 链全 fp32 域经 `T.tile.exp/mul`；
  bf16↔fp32 转换一律 UB 片内两步 copy 链（GM↔UB cast-on-copy 代码生成损坏）。
- **硬件与实现约束**：910B3（20 Cube + 40 Vector 核）；threads=2 使全部 Vector
  核参与；UB 预算 96%（3-chunk staging 为当前上限，4-chunk 超 196352B 硬上限）；
  手动 flag ID 从 48 起编（auto pass 不感知用户 ID，冲突会静默破坏握手）；
  wrapper 无内部 synchronize（上游 do_bench 语义），Msk 常量按 (device, chunk)
  缓存。
- **测试验证**：交付文件内含 2 个代表用例（L0 最小门禁 (1,128,2,64,64) 与
  L1 基准 shape (1,8192,64,128,128)），PyTorch golden（fp32 域 chunk 循环，
  `torch.exp2` 精确语义）。开发期分层套件另覆盖非对齐/质数/退化 shape、
  数值范围、特殊值与域外拒绝。

## 精度验证 ✅

**测试配置**：`B=1..2, S=64..8192, H=1..64, DK=48..128, DV=16..128, chunk=64，bf16（gate fp32）`

| **测试层级/类别** | **用例数** | **检查项数** | **测试结果** | **精度指标** | **最大绝对误差** |
| --- | --- | --- | --- | --- | --- |
| L0 门槛（规则 shape） | 2 | 2 | PASS | matched_ratio=1.0000 | 1.250e-01 |
| L1 功能（含基准 shape/非对齐/质数/数值范围/非默认 scale） | 9 | 9 | PASS | matched_ratio=1.0000 | 1.000e+00（极值范围用例，硬帽内） |
| L2 域外拒绝（S 非整除/DK 域外/DV 域外/fp16/gate bf16 等） | 10 | 10 | PASS（全部正确拒绝） | — | — |
| Boundary 特殊值（zero/inf/nan/dbound） | 5 | 5 | 3 PASS + 2 记录性告警 | 告警详见下 | inf 门控 NaN 结构差 / 超范围量级 |
| **总计** | **26** | **26** | **PASS（门禁项全过）** | — | — |

- **精度阈值**：bf16 混合容差——逐元素 `|actual-golden| ≤ 2^-10 + 2^-6·|golden|`，
  整体 `matched_ratio ≥ 0.99` 且 `max_abs_error ≤ 1e0`
- **通过条件**：L0/L1 全部 matched_ratio=1.0000（远超 0.99 门禁）
- **参考实现**：PyTorch chunk 循环 fp32 golden（`fla_chunk_o_ref`）
- **Boundary 告警说明**（非门禁项）：`g=±inf` 门控下 hi/lo 拆分产生
  `inf-inf=NaN`（golden 为 ±inf，位置结构一致）；`g=±120` 时 `|o|~1e33` 超出
  bf16 1e0 绝对误差硬帽的可认证域（ratio 0.9999）。两者均为 hi/lo 精度方案
  在极端值域的固有边界，正常值域内不出现。

## 性能测试

**测试平台**：Ascend 910B3
**测试配置**：`B=1, S=8192, H=64, DK=128, DV=128, chunk_size=64`，输入输出 bf16（gate fp32，累加 fp32）

| **性能指标** | **测试结果** |
| --- | --- |
| NPU 事件计时（10 次预热后 10 次重复取均值，仅设备侧） | **1417 μs**（快发射路径）/ 1492 μs（通用 JIT 发射路径） |
| 稳态单次（rep 2-10） | ~1400 μs |
| Block Dim / Mix Block Dim | 20 blocks（fixed-core）× threads=2 |
| Op Type | 单 kernel（Triton 参考为同构单 kernel） |

> 计时口径对齐上游 `do_bench`：预热 10 次、同步后连续 10 次重复、逐次
> NPU 事件计时取均值，不含张量构造/H2D-D2H/JIT 编译/框架调度。两种发射路径
> 差异仅为 host 侧参数适配开销在首次重复中的暴露（快路径已含完整参数校验、
> 首次发射 bitwise 自检与失败自动回退通用路径）。

### ⚠️ 性能差距说明

| **平台/方案** | **运行延迟** | **性能差距** |
| --- | --- | --- |
| Ascend 910B3（本次 PR，快发射路径） | **1417 μs** | — |
| 性能目标（纯 Kernel 合计） | < 1320 μs | 超 7.5% |

实测设备侧单次执行 1391μs（msprof），已高于 1320μs 目标——即使 host 发射
开销为零，该 shape 下也无法达标。剩余差距的构成为工具链同步协议开销：每任务
约 3.4μs 的 relay 往返与逐操作同步指令流（计算/装载合计 <80μs，IO 带宽利用
47-59% 非墙）。进一步压缩需要 zN 对齐的 L1 region 操作数（合并 4 路中继）、
跨核 flag 批量化原语或全手动同步模式，均超出本算子可控范围；上述实测依据
已随学习档案归档，可作为上游工具链改进的参考。
