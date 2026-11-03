为 Ascend NPU 新增 `fla_chunk_inter`（KDA chunked 反传 inter-chunk 梯度）算子。

- **Kernel 架构**：`@tilelang.jit` 单融合 CV kernel（`out_idx=[20,21,22,23]`，
  `workspace_idx=[8..19]` 绝对索引，24 张量签名），Developer 模式
  （`alloc_shared/alloc_fragment` + `TL_ASCEND_AUTO_CV_COMBINE/AUTO_CV_SYNC/
  AUTO_SYNC/MEMORY_PLANNING` 四件套 pass_configs）。`T.Kernel(CORE_EFF<=20,
  threads=1)` fixed-core grid + 单 `T.serial` produce/drain 任务循环：8192 个逻辑
  (chunk, b, h) 任务按 `tid = cid + j * CORE_EFF` 条带化摊到 20 个常驻核。
- **计算结构**：对每个 (b, chunk, head) 任务，在 DK 维逐 block 展开：
  `dq = (do @ h^T) * scale * 2^g`；`dk = (v @ dh^T) * 2^(gn - g)`（gn = chunk
  末行累积 log2 衰减）；`dw = -(dv @ h^T)`（负号折叠进 dv 操作数，epilogue 直存
  GM）；`dg = q*dq - k*dk + m_last * pp`，`pp = 2^gn * sum_v(h*dh) + sum_t(dk*k)`，
  其中 `sum_v(h*dh)` 由 Cube 侧 `h @ dh^T` 对角化产出。DK 与 DV 为静态整除约束，
  全部循环边界编译期静态。
- **数据流与内存管理**：GM → UB（V 侧 fp32 装载 + hi/lo 拆分）→ GM 显式中继
  workspace（workspace_idx 自动分配，L2 常驻）→ L1（gemm 操作数，A 操作数跨 DK
  block 复用）→ L0C（fp32 累加 fragment）→ GM。drain 侧 dq/dk/dgkn 三中继融合为
  单次 48KB 读。片上预算：UB ≤ 196 KiB 硬上限内规划，L1 双 hi/lo 操作数对 +
  L0C 累加器。
- **关键优化策略**：
  - **fixed-core grid + 显式 L2 常驻中继**：逻辑任务条带化 + 小容量中继
    workspace，中继字节 L2 命中（-11.3%）；
  - **block_DK 形状自适应**：DK%64==0 且 DV>=128 时翻倍 DK tile，任务数减半、
    V 侧每任务字节 -40%（-31.1%）；
  - **NK-merge 任务重构造**：任务从 (chunk, head, dk-block) 合并为 (chunk, head)，
    无 DK 轴的 v/do/dv 单次装载拆分中继，消除 NK 倍重复读（-18.5%）；
  - **1-late flag 配对**：drain(j) 消费任务 j-1 的中继、等待 j-1 的 flag（一次性
    head-set 预充），消除同迭代锁步暴露的 V 侧等待 4896 → 0 us（-29.4%）；
  - **dgkn Cube 侧对角化**：`sum_v(h*dh)` 改为复用 dq/dk gemm 组已在 L1 的
    hi/lo 操作数做 `h @ dh^T` 三个小 gemm，取对角经小中继回传 drain（V 侧
    -4.5%）；
  - **门控向量化与中继读融合**：gn 门控行展开用转置+行广播指令（无 scratch
    形态，-2.2%）；drain 三中继融合单读（-2.1%）。
- **数值精度处理**：本工具链 fp32 `gemm_v0` 损坏，全部 GEMM 操作数拆 fp16 hi/lo
  对（x = hi + lo），每 GEMM 变 hi@hi + hi@lo + lo@hi，fp32 L0C 累加（相对误差
  ~2^-22/操作数对，含 dgkn 对角化路径）；exp2 无后端实现，2^x = exp(x*ln2)，ln2
  预折进 g/gn；2^g 门控不进 GEMM（fp16 操作数 65504 溢出防线）。
- **硬件与实现约束**（wrapper assert 确定性拒绝）：S % 64 == 0（尾块在参考实现中
  未定义）、DK % 32 == 0、DV 为 16 的倍数且 >= 64、chunk_size == 64、全 fp32。
  一个 UB 源只能中继到单个 L1 目的（workspace pass 按源名解析 V→C 中继写），故
  dw 组共享 dq 组的 h 中继。**实现不使用 `T.reduce_sum` / `T.tile.broadcast`**：
  本工具链这两类 op 会触发规划器注入共享 scratch，其 UB 打包次序随编译上下文
  翻转且存在非法放置（设备 Illegal instruction）；归约以折叠树/对角化替代、行
  广播以转置+行展开指令替代（example 内置维护注释，请勿重新引入）。
- **测试验证**：分层测试套件（L0 门槛 2 用例 + L1 功能 8 用例含 scale/值域/非
  二次幂 shape + L2 负向 8 用例 + Boundary 5 用例）；交付示例
  `test_example_fla_chunk_inter.py` 取代表性 L0 最小门槛 shape 与 bench shape
  两用例，PyTorch golden 对比，4 输出独立混合容差判定。

## 精度验证 ✅

**测试配置**：B∈{1,2}、S∈{64..8192}、H∈{1..64}、DK∈{32,64,96,128}、DV∈{64..128}、
chunk_size=64、全 fp32；golden = PyTorch 重写参考实现（einsum 轴序经显式 matmul
循环交叉验证）

| **测试层级/类别** | **用例数** | **检查项数** | **测试结果** | **精度指标** | **最大绝对误差** |
| --- | --- | --- | --- | --- | --- |
| L0 门槛（规则 shape） | 2 | 8 | 全部通过 | matched_ratio = 1.0000 ×8 | 1.221e-3（dg @ (2,256,4,128,128)） |
| L1 功能（含不规则/值域/scale） | 8 | 32 | 全部通过 | matched_ratio = 1.0000 ×32 | 7.812e-3（值域 ×3.0 极值用例 dg） |
| L2 负向（域外确定性拒绝） | 8 | 8 | 全部正确拒绝 | —（不比精度） | — |
| Boundary（Inf/NaN/零/溢出） | 5 | 20 | 4 通过 + 1 文档化告警 | 4 用例 ratio = 1.0000 | 告警为 ±65504 操作数极限用例（结构一致，非阻塞） |
| **总计** | **23** | **68** | **通过** | — | — |

- **精度阈值**：fp32 混合容差——atol 2^-16（1.53e-5）、rtol 2^-10（9.77e-4）、
  max_abs_error_limit 1e-2、required_matched_ratio 0.99（双门限：逐元素
  `|actual-golden| <= atol + rtol*|golden|` 且整体 matched_ratio 与 max_abs 同时达标）
- **通过条件**：L0/L1 全部 ratio >= 0.99 且 max_abs <= 1e-2（实测全部 1.0000）；
  L2 验证域外输入被 assert 确定性拒绝
- **参考实现**：PyTorch golden（fp32 域 + `torch.exp2` 语义，与上游 Triton
  `chunk_kda_bwd_dqkwg` 对齐验证）

## 性能测试

**测试平台**：Ascend 910B3（20 AIC + 40 AIV，HBM 实测 r+w 带宽 1263 GB/s）
**测试配置**：B=1, S=8192, H=64, DK=128, DV=128, chunk_size=64, scale=1.0，全 fp32
（对齐上游 `examples/kda/chunk_bwd_dqkwg.py` benchmark 入口）

| **性能指标** | **测试结果** |
| --- | --- |
| NPU 事件计时（20 次预热 + 10 次重复取均值，仅设备侧 kernel 执行时间） | **6868.3 μs**（5 次独立运行 6871.9 / 6873.0 / 6859.0 / 6855.8 / 6881.8） |
| 5 次独立运行范围 | 6855.8 – 6881.8 μs |
| Block Dim | 20 blocks（fixed-core，每 block 1 AIC + 1 AIV，threads=1），8192 逻辑任务条带化 |
| Op Type | Cube+Vector 混合融合 kernel |

> 计时口径对齐上游 KDA 示例 `do_bench`：预热 20 次后重复 10 次取平均，仅计
> 设备侧 kernel 执行时间，不含张量构造 / H2D-D2H / JIT 编译 / 框架调度开销。
> kernel 合计口径：本算子为单 kernel，wrapper 内无其他设备侧操作。

### 性能优化效果

| **版本** | **运行延迟（do_bench 口径）** | **相对基线** |
| --- | --- | --- |
| 基线（首次正确实现，逐 dk-block 任务 + 逐 block 自动中继） | 19556 μs | — |
| 终版（fixed-core + L2 常驻中继 + block_DK 自适应 + NK-merge + 1-late flag + Cube 侧对角化 + 门控向量化 + 中继读融合） | **6868 μs** | **-64.9%（2.85×）** |

关键手段各自的同会话 A/B 实测贡献（相对各自起点）：block_DK 自适应 -31.1%、
1-late flag 配对 -29.4%、NK-merge 任务重构造 -18.5%、fixed-core + L2 常驻中继
-11.3%、dgkn Cube 侧对角化 -4.5%、门控向量化 -2.2%、中继读融合 -2.1%。

### 当前配置下的可达值分析（客观依据）

终态贴当前实现形态的结构地板，msprof 终版画像归因（8192 任务 / 20 AIV）：

| 项 | 实测值 | 说明 |
| --- | --- | --- |
| V-MTE2 busy（装载） | ~3219 μs @ 53 GB/s/AIV | 416 KB/任务对，结构必需字节（hi/lo 拆分使 fp32 原字节翻倍） |
| 立即优化空间 | — | V 侧 flag 等待 0 us、装载零冗余（NK-merge 后） |
| Cube 侧 | 全程隐藏于 V 链之后 | 对角化后 C 侧承担 dgkn 三 gemm |

- **当前形态地板 ≈ 6868 μs**，由三段无 scratch 结构的实测成本构成：dgkn 对角
  中继回读（Cube 侧无 gather 原语，~291 μs）、gn 门控的转置+行展开（行广播指令
  为唯一无 scratch 形态，~328 μs）、drain 折叠树（~170 μs）。
- **若上游工具链修复内存规划器对注入 scratch 的打包确定性**（指针序迭代 → 名称
  确定性排序），`reduce_sum`/`broadcast` 的直接形态可用，上述三段合计 ~500 μs 级
  可回收（同上下文实测过 6767 μs 形态）。
- **更远的地板**：threads=2（40 AIV）受 vid 切分语义限制（本工具链
  `ascend_vid_reduction.cc` 的 broadcast/跨 vid 归约缺口）不可用；修复后装载带宽
  翻倍、V 链地板减半，但随即受 Cube 消费链约束。HBM 唯一 IO（3.76 GB）物理界
  2975 μs@100% / 3673 μs@81%（同平台实证最佳效率）。
- **结论**：本工具链与 910B3 架构下，该算子当前实现形态的现实可达值 ≈ 6868 μs，
  本实现已贴线；进一步逼近 HBM 物理界依赖上游工具链解锁（规划器 scratch 打包
  确定性、threads=2 vid 语义、V↔C 片上直连三项，均可独立修复）。
