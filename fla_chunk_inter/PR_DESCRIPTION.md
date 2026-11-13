为 Ascend NPU 新增 `fla_chunk_inter`（KDA chunked 反传 inter-chunk 梯度）算子。

- **Kernel 架构**：`@tilelang.jit` 单融合 CV kernel（`out_idx=[19..22]`，
  `workspace_idx=[8..18]` 绝对索引，23 张量签名），Developer 模式
  （`alloc_shared/alloc_fragment` + `TL_ASCEND_AUTO_CV_COMBINE/AUTO_CV_SYNC/
  AUTO_SYNC/MEMORY_PLANNING` 四件套 pass_configs）。`T.Kernel(CORE_EFF<=20,
  threads=1)` fixed-core grid + 单 `T.serial` produce/drain 任务循环：8192 个逻辑
  (chunk, b, h) 任务按 `tid = cid + j * CORE_EFF` 条带化摊到 20 个常驻核，越界
  迭代 Select-redirect 幂等重算。
- **计算结构**：对每个 (b, chunk, head) 任务，在 DK 维逐 block 展开（trace-time
  常量展开）：`dq = (do @ h^T) * scale * 2^g`；`dk = (v @ dh^T) * 2^(gn - g)`
  （gn = chunk 末行累积 log2 衰减）；`dw = -(dv @ h^T)`（负号折叠进 dv 操作数，
  Cube 侧直存 GM）；`dg = q*dq - k*dk + m_last * pp`，`pp = 2^gn * Σ_v(h*dh)
  + Σ_t(dk*k)`，其中 `Σ_v(h*dh)` 由 `diag(h @ dh^T)` 的 Cube 侧 gemm 组产出。
  DK 与 DV 为静态整除约束，全部循环边界编译期静态。
- **数据流与内存管理**：GM → UB（V 侧 fp32 装载 + hi/lo 拆分 + fp16 中继）→
  GM 显式中继 workspace（10 个 fp16 操作数中继 + 1 个融合 fp32 drain 中继
  dq|dk|diag 单读回，L2 常驻）→ L1（gemm 操作数，A 操作数跨 DK block 复用）→
  L0C（fp32 累加 fragment）→ GM。片上预算：UB 声明集经 lifetime 复用压入
  196 KiB 硬上限，L0C 累加器 4 fragment。
- **关键优化策略**：
  - **fixed-core grid + 显式 L2 常驻中继**：逻辑任务条带化 + 小容量中继
    workspace，中继字节 L2 命中（~97%）；
  - **NK-merge 任务重构造**：任务从 (chunk, head, dk-block) 合并为 (chunk,
    head)，无 DK 轴的 v/do/dv 单次装载拆分中继，消除 NK 倍重复读；
  - **1-late flag 配对**：drain(j) 消费任务 j-1 的中继、等待 j-1 的 flag
    （prologue head-set 预充 j=0；手动 flag 用专用 slot 10——10 个操作数中继
    使自动 flag 分配器占用 0..9，11..14 系统保留，10 是唯一无碰撞用户槽），
    Vector 侧从不因同迭代 Cube 消费链停摆；
  - **装载批处理**：三个 GM 装载连发（MTE2 突发）+ split 链集中后置，装载流
    连续化（同会话 A/B -2.40%，装载流 mte2 stall 775.7 → 9.7 us）；
  - **block_DK 形状自适应**：DK%64==0 且 DV>=128 时翻倍 DK tile（性能参数，
    两路径覆盖全域）。
- **数值精度处理**：本工具链 fp32 `gemm_v0` 操作数损坏，全部 GEMM 操作数拆
  fp16 hi/lo 对（x = hi + lo），每逻辑 GEMM 变 hi@hi + hi@lo + lo@hi，fp32 L0C
  累加（相对误差 ~2^-22/操作数对）；exp2 无后端实现，2^x = exp(x*ln2)，ln2
  预折进 g/gn（舍入差 ~2e-7 相对，rtol 预算下 4 个量级）；2^g 门控不进 GEMM
  （fp16 操作数 65504 溢出防线）。
- **硬件与实现约束**（wrapper 确定性拒绝）：S % 64 == 0（尾块无 h/dh 状态，
  参考实现未定义域）、DK % 32 == 0、DV 为 16 的倍数且 >= 64、chunk_size == 64、
  全 fp32、输入连续。**实现零使用 `T.reduce_sum` / `T.tile.broadcast`**：这
  两类 op 会触发内存规划器注入共享 scratch，其 UB 打包随编译上下文翻转且存在
  非法放置（设备 Illegal instruction）；归约以折叠树替代、行广播以转置+行展开
  指令替代、末行注入为单行加、对角和经 Cube 侧 gemm + gather 提取（example
  内置维护注释，请勿重新引入）。
- **测试验证**：分层测试套件（L0 门槛 2 用例 + L1 功能 9 用例含 scale/值域/
  非对齐 shape + L2 负向 13 用例 + Boundary 5 用例）；交付示例
  `test_fla_chunk_inter.py` 取代表性 L0 最小门槛 shape 与 bench shape 两用例
  + 一个域外输入（S % 64 != 0）确定性拒绝检查，PyTorch golden 对比，
  4 输出独立混合容差判定（含 golden 非有限位置 NaN/Inf 结构比对）。

## 精度验证 ✅

**测试配置**：B∈{1,2}、S∈{64..8192}、H∈{1..64}、DK∈{32,64,96,128}、DV∈{64..128}、
chunk_size=64、全 fp32；golden = PyTorch 重写参考实现（einsum 轴序经显式
per-(b,h) matmul 循环逐位交叉验证）

| **测试层级/类别** | **用例数** | **检查项数** | **测试结果** | **精度指标** | **最大绝对误差** |
| --- | --- | --- | --- | --- | --- |
| L0 门槛（规则 shape） | 2 | 8 | 全部通过 | matched_ratio = 1.0000 ×8 | 1.083e-3（dg @ (2,256,4,128,128)） |
| L1 功能（含不规则/值域/scale） | 9 | 36 | 全部通过 | ratio 下界 0.999939 | 2.037e-3（值域极值用例 dg） |
| L2 负向（域外确定性拒绝） | 13 | 13 | 全部正确拒绝 | —（不比精度） | — |
| Boundary（Inf/NaN/零/溢出） | 5 | 23 | 17 通过 + 6 文档化告警 | 17 检查 ratio = 1.0000 | 告警为 2^(gn-g) 有限巨大值与 >65504 操作数出 fp16 包络（结构一致，设计内预期） |
| **总计** | **29** | **80** | **通过** | — | — |

- **精度阈值**：fp32 混合容差——atol 2^-16（1.53e-5）、rtol 2^-10（9.77e-4）、
  max_abs_error_limit 1e-2、required_matched_ratio 0.99（双门限：逐元素
  `|actual-golden| <= atol + rtol*|golden|` 且整体 matched_ratio 与 max_abs
  同时达标）
- **通过条件**：L0/L1 全部 ratio >= 0.99 且 max_abs <= 1e-2；golden 非有限
  位置做结构比对（非有限位置逐一重合且 NaN/Inf 模式相等）；L2 验证域外输入
  被 ValueError 确定性拒绝
- **参考实现**：PyTorch golden（fp32 域 + `torch.exp2` 语义，与上游 Triton
  `chunk_kda_bwd_dqkwg` 对齐验证）

## 性能测试

**测试平台**：Ascend 910B3（20 AIC + 40 AIV，HBM r+w 1263 GB/s）
**测试配置**：B=1, S=8192, H=64, DK=128, DV=128, chunk_size=64, scale=1.0，全 fp32
（对齐上游 `examples/kda/chunk_bwd_dqkwg.py` benchmark 入口）

| **性能指标** | **测试结果** |
| --- | --- |
| NPU 事件计时（20 次预热 + 10 次重复取均值，仅设备侧 kernel 执行时间） | **6741.8 μs**（3 次独立运行 6734.4 / 6741.3 / 6749.8） |
| msprof op 单 launch 设备侧（CSV 口径交叉验证） | 6502.7 μs |
| 应用级 msprof 11 launch 均值（Insight 口径） | 6657.2 μs |
| 3 次独立运行范围 | 6734.4 – 6749.8 μs |
| Block Dim | 20 blocks（fixed-core，每 block 1 AIC + 1 AIV，threads=1），8192 逻辑任务条带化 |
| Op Type | Cube+Vector 混合融合 kernel |

> 计时口径对齐上游 KDA 示例 `do_bench`：预热 20 次后重复 10 次取平均，仅计
> 设备侧 kernel 执行时间，不含张量构造 / H2D-D2H / JIT 编译 / 框架调度开销。
> kernel 合计口径：本算子为单 kernel，wrapper 内无其他设备侧操作。

### 性能优化效果

| **版本** | **运行延迟（do_bench 口径）** | **相对基线** |
| --- | --- | --- |
| 基线（首次正确实现：逐 dk-block 任务 + 逐 block 自动中继，未含结构优化） | 19556 μs | — |
| 终版（fixed-core + L2 常驻中继 + NK-merge + 1-late flag + Cube 侧对角化 + 装载批处理 + 门控向量化 + 融合中继读回） | **6741.8 μs** | **-65.5%（2.9×）** |

关键手段各自的同会话 A/B 实测贡献（相对各自起点）：block_DK 自适应 -31.1%、
1-late flag 配对 -29.4%、NK-merge 任务重构造 -18.5%、fixed-core + L2 常驻中继
-11.3%、装载批处理 -2.40%、dgkn Cube 侧对角化 -4.5%、门控向量化 -2.2%、中继
读融合 -2.1%。

### 当前配置下的可达值分析（客观依据）

终态贴当前实现形态的结构地板，msprof 终版画像归因（8192 任务 / 20 核 /
满频 1800MHz）：

| 项 | 实测值 | 说明 |
| --- | --- | --- |
| AIV 串行链（mte2/vec/mte3） | 2880 + 2238 + 1453 = 6571 ≈ 墙 | 三引擎和≈墙 = 零重叠签名：UB 196 KiB 容量强制逐操作数链式 |
| AIC busy | 5507 us（85.3% 墙） | cube 937 + mte1 1233 + mte2 1259 + fixpipe 979 + scalar 1256；在等 V 中继 |
| 双墙锁步 | aic 6452 / aiv 6449，逐核差 +2~4us 全为正 | 两侧核同步进退——语序重排类优化的结构盲区（6 变体实测零收益） |
| V-MTE2 | 56.7 GB/s（引擎界上沿），416.3 KB/任务结构必需字节 | hi/lo 拆分后中继 + 门控装载，占墙 44.7% |
| HBM 唯一 IO | 3.758 GB / 6502.7 us = 45.8%；L2 命中 ~97% | 唯一 IO 不是瓶颈，中继流量被 L2 吸收 |

- **当前形态地板 ≈ 6740 μs**：V 侧串行链（UB 196 KiB 硬上限）与 C 侧 busy
  5507 μs 双墙锁步共同锚定；语序重排空间经 6 个变体实测证伪（区间交叠/劣化）。
- **即使 V 侧三引擎完美重叠，C 侧 busy 5507 μs 仍在目标之上**——C busy 由
  12 gemm/任务（fp32 gemm 损坏 → hi/lo 3-gemm 放大）锚定，此项为工具链上游
  修复项（fp32 gemm 修复将同时消 3× gemm 数与中继字节减半，可达带重开）。
- **更远的地板**：threads=2（40 AIV）受 vid 切分语义限制（本工具链
  `ascend_vid_reduction.cc` 的 broadcast/跨 vid 归约缺口）不可用；修复后装载
  带宽翻倍。HBM 唯一 IO 物理界 2980 μs@100% / 3673 μs@81%（同平台实证最佳
  效率）。

