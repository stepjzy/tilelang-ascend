# fla_chunk_intra_token_parallel — Stage 3 跨轮性能日志（perf_log）

> 计时口径（每轮一致）：torch.npu.Event(enable_timing=True)，warmup=20 / rep=10 / mean，
> 仅设备侧 kernel 时间（对齐 ref_gpu/test_utils_kda.py do_bench 口径的 NPU 等价实现；
> loop 模式（单事件对包 N 次连发 / N）交叉验证 host 分发污染，两口径差 <1% = 非分发受限）。
> 基准 shape：(1, 8192, 64, 128)，CS=64，SCS=16，scale=1.0（DESIGN §9.4 perf_bench 哨兵）。
> 唯一 IO 621,409,280 B；实测 D2D r+w 上限 1263 GB/s（fla_wy_fast journal 标定）→ 纯带宽下限 492 us。
> 沙箱 NPU 为多算子共享：每轮计时前用 `ps` + `npu-smi` 确认无外部进程（fla_chunk_o 会话的
> probe_sides.py 等），per-rep 方差 <0.1% 作为无争抢旁证。

## Iteration 1 — 2026-10-22T21:40:00+08:00

- bottleneck_type: sync + 指令下发开销（barrier/标量散写风暴，非带宽非纯算力）
- optimization: **热循环重构（DESIGN R5/R6 杠杆）**——消灭标量散写 + 运行时 if + 行批量 tile 化 + h-major 布局 + per-head 聚合搬运
  - 诊断（生成 C 代码静态剖析，fla_chunk_intra_token_parallel_kernel_source_iter1.c）：
    基线每 block ~2700 次 barrier（每个 IfThenElse 前后各 1 次 PIPE_ALL × 816、每个标量
    SetValue 带 V_S flag 对 + PIPE_ALL × 1088、同管线相邻 tile op 间 PIPE_V × ~700、j>i 空迭代
    ×240）+ 1088 次标量 UB 写 + 81 次小 DMA；估算纯向量计算仅 ~1.9ms，开销占 ~90%。
  - 重构内容：
    1. i 轴 parse 期展开：`T.macro`（i 为字面量参数）+ 16 个 `if {i} < SCS:` 守卫调用点
       （捕获的 Python int 使 if 在 parse 期求值为真分支，不生成 TIR If）；kernel body 零运行时分支。
    2. per-(h,i) 行批量 tile：j∈[0,i] 整批 broadcast(g_i)→sub→exp→kgj→broadcast(q_i)→mul→
       reduce_sum(dim=-1) **直写 staging region**（aqk_s[h,i,w:w+i+1] 连续 1D 段）——标量散写 1088→0，
       tile op 950→~690，reduce 调用 272→124（8.5× 减少），j 空迭代浪费 100% 消除。
    3. h-major 计算域 [BH_V,SCS,DK] + per-head 2D strided 聚合载入（[16,DK]/burst 256B/512B，
       48+1 次 DMA→12+1）；写回 per-head [SCS,CS]/[SCS,AKK_W] 2D strided（32→8 次）。
    4. β 后折乘改为归约前标量 Muls（ascend_muls BufferLoad 平坦索引路径，
       beta[i,h] @ 垫宽 [SCS,16] fp32 的 flat i*16+h）：60 次/块（每次附带 1 次 PIPE_ALL——
       向量化 β 广播因 4B 粒度地址无法 32B 对齐而不可行，见下方工具链发现）。
    5. i=0 对角快路径：exp(0)==1 精确成立，Aqk_diag = rowsum(q_0⊙k_0)（跳过 broadcast/sub/exp）。
  - 实施中修复的 4 个工具链/对齐问题（详见 optimization_log）：
    a) region point 维保留 extent-1 前导维（[1,BH_V] vs [BH_V,1] broadcast 形状失配）；
    b) VEC 指令源地址必须 32B 对齐（β 标量广播 4B 粒度 → aicore 507015）；
    c) UB 内 copy 两侧行宽均须 ≥32B（β 16B 行紧凑 copy → aicore exception）；
    d) copy lowering 丢弃中间维 point 切片的物理行距（[:, h, :] 按连续读 → staging 翻转 h-major；
       窄行 MTE3 源 32B pitch → SCS=8 时 Akk staging 垫宽到 16 列）。
- baseline_time: 19150.1 us（events mean；per-rep 19079-19083，方差 <0.03%；loop 19100.0）
- candidate_time: 9095.4 us（events mean；per-rep 9004-9011；loop 9027.8）
- improvement: **-52.5%（2.11×）**；有效带宽 32→68 GB/s（实测上限的 2.6%→5.4%）
- precision: pass（`--level all` exit 0：L0 3/3 + golden 自检 + L1 9/9 + L2 10/10 正确拒绝 +
  Boundary 4/4 [BOUNDARY_PASS]，全部 ratio=1.0000，max_abs ≤ 6.25e-2，与 Stage 2 基线一致）
- adopted: **yes**
- rollback_reason: n/a
- next_hint:
  - **[#3] AUTO_SYNC=False + 手动最小同步**（optimization-guide §2.2.0 决策表：纯 AIV 流水必须关）：
    当前每 block 仍有 ~690 次 tile-op 间 PIPE_V barrier + 60 次 muls 标量读 PIPE_ALL。
    生成代码已确认 barrier 由 ascend_sync_insert.cc 的 GetRequiredSyncType 对同管线相邻访问
    无条件插入；关闭后需手动补 mte2→v（载入后）/ v→mte3（写回前）flag 对并全量复验。
    预期这是下一个最大杠杆（若 barrier 仍占 ~80%，潜在再 2-4×）。
  - **[#4] 因式分解式（e_i·e'_j 锚定）**：-30% elem-ops、exp 69.6K→16K/块（-77%）；需 mid-anchor
    （(g_first+g_last)/2，subchunk 通道衰减 ≤~250 时安全）+ Boundary 重审 + 域收窄文档化。
  - β 向量化折乘（消除 60 次 muls 标量读）：需 [BH_V,1] 方向的 32B 对齐 β 布局（per-(i,h) 标量
    32B slot 的填充=scatter，不可用 2D DMA 表达）；候选：T.transpose 建 h-major β + per-h 折乘 +
    select 掩码保对角（NaN-safe）。
  - reduce_2d 的 tmp/延迟机制（M1StepDelay/WaitForVectorData）在 staging 直写模式下的成本未单独
    标定；若 [#3] 后 reduce 占比显性化，评估 reduce 与 mul 的重组。
  - 每轮计时前确认 NPU 空闲（共享沙箱；fla_chunk_o 会话间歇性跑 probe）。

## Iteration 2 — 2026-10-22T22:10:00+08:00

- bottleneck_type: sync（本轮假设，**实测证伪**）→ 重新分类：向量指令流本身（issue/归约 helper
  开销，compute/issue-bound，待 iter3 分解剖析定论）
- optimization: **[#3] AUTO_SYNC=False + 手动最小同步（4 对 set/wait flag）+ [#3b] β 标量读
  barrier 消除**（上一轮 next_hint 的最高优先候选，预期 2-4×）
  - 实施（生成代码已验证：4 组 flag 对 = V_MTE2/MTE2_V/V_S/V_MTE3，**PipeBarrier 总数 0**，
    原 ~608 V-V + 60 ALL barrier/block 全部消除；`--level all` 全绿 pass）：
    1. pass_configs AUTO_SYNC: False；kernel body 手动插入 4 对跨管线 flag：
       v→mte2（β fill→β 载入 WAR）/ mte2→v（载入→首个 V 消费 RAW）/
       v→s（beta_f 稳定后标量读）/ v→mte3（bf16 cast→写回 RAW）。
       V-V 同管线 barrier 依据 §2.2.1"V pipe 单条指令链天然串行"+ opt_gdn_solve_tril
       in-place 先例全部省略（精度全量通过证实）。
    2. β 标量读改 PrimExpr 路径：同块预读局部 `b{i} = beta_f[{i}, h] * 1.0` 传入宏——
       raw BufferLoad src1 走前端 access_ptr 路径（codegen 无条件 PipeBarrier<PIPE_ALL> +
       GetValue，**该 barrier 在 codegen 层发射，关 AUTO_SYNC 消不掉**）；预读局部走
       CoerceScalarArg 路径（提升为 `float b{i} = beta_f.GetValue(...)` 局部，零 barrier）。
    3. 探针实证的三态 lowering（probe/probe2，值得 skill journal 沉淀）：
       a) 直连调用 + 预读局部：无 barrier（hoisted GetValue 局部）；
       b) **内联表达式作宏参数：被宏机制折叠回 raw BufferLoad → barrier 路径回归**（坑）；
       c) 索引语义：前端 BufferLoad 分支取 indices[0] 当**平坦偏移**（quirk），PrimExpr 路径走
          OffsetOf 正规扁平化——`beta_f[i*16+h, 0]` 在两条路径下指向不同元素！正确写法
          `beta_f[i, h]`（OffsetOf = i*16+h，与 quirk 路径同元素）。本轮曾因此产生越界索引
          （生成 GetValue(h*16+256)），L0 前静态检查发现并修复。
  - **结果：性能假设证伪** — barrier 风暴移除仅 -2.3%：
    - baseline（复测确认）：9077.3 us（per-rep 8999-9015，方差 <0.2%）
    - candidate：8871.7 us（复测 8863.0，per-rep 8775-8790 稳定）→ **-2.3%**
    - 推论：~670 barrier/block × 4096 block ≈ 2.7M 次 barrier 总成本仅 ~210 us（~75 ns/次，
      或大部分已被流水掩盖）——**iter1 "~90% sync 开销" 的估算对当前结构不成立**（该估算
      源自基线版的标量散写+IfThenElse 结构，iter1 重构后剩余 barrier 已不是主要开销）。
    - 重新归因（待 iter3 剖析验证）：每 vid-work-item ~690 条向量指令流 × ~63 ns/条 ≈ 43 us
      （204.8 item/核 × 43.3 us = 8.87 ms ✓ 对账吻合）。疑点排序：
      (i) reduce_2d_v2 helper 内部 Delay/pipe_barrier 机制（124 次/item，成本未标定）；
      (ii) 小 repeat（2-16）指令的 per-instruction 固定开销（每个 tile op 另发
           SetMaskNorm/SetVectorMask 配置对，~1700 条标量侧配置/item）；
      (iii) exp 吞吐（571M exp，估算仅占 ~8%）。
- baseline_time: 9077.3 us（iter1 采纳版复测，与 9095.4 记录一致）
- candidate_time: 8871.7 us（确认复测 8863.0 us；loop 口径 8803.7/8804.8）
- improvement: **-2.3%**（< 3% 噪声阈值）
- precision: pass（`--level all` exit 0：L0 3/3 + golden 自检 + L1 9/9 + L2 10/10 +
  Boundary 4/4，ratio=1.0000 全绿，max_abs 与 Stage 2 基线一致量级）
- adopted: **no**
- rollback_reason: 改进 -2.3% 低于 3% 采纳阈值（精度通过、改进真实但不足阈值；已回滚到
  impl_iter2_before.py = iter1 采纳版，回滚后 L0 复验 pass）
- next_hint:
  - **[#6] 先剖析再优化（iter3 第一优先）**：构造分解 probe 标定三块成本——(a) 纯 mul/broadcast
    指令链（无 reduce）、(b) reduce_2d 单独成本（124 次/item × 内部 Delay/barrier）、(c) exp。
    数据决定后续路径；避免再次基于估算押注单一杠杆。
  - 若 reduce_2d 主导：评估 reduce 重组（减少调用次数/换 WholeReduceSum 类原语/两行窗口合并）。
  - 若 per-instruction 开销主导：指令合并方向——增大单条指令 repeat（行窗口拼接）、减少
    SetMask* 配置对（同 mask 模式指令相邻排布是否省配置？查 codegen SetMaskNorm/SetMaskCount
    发射条件）、或 j-major 重排。
  - **[#4] 因式分解重估**：若 issue-bound，elem-op -30% 收益有限（指令数不变）；exp -77%
    缩短的是指令 repeat 而非指令数。优先级让位于剖析。
  - 工具链沉淀（本轮 3 个发现待 journal 化）：宏参数内联标量表达式被折叠回 BufferLoad（坑）；
    BufferLoad 标量 src1 的 indices[0]-as-flat-offset quirk vs PrimExpr OffsetOf 语义差异；
    codegen β 标量 PIPE_ALL 与 AUTO_SYNC 无关（须走 PrimExpr 路径消除）。
  - 附注：perf_tuning/fla_chunk_intra_token_parallel_kernel_source_iter1.c 为 iter1 开发中期
    过期产物（beta_f 还是 64 元素旧布局），barrier 剖析结论按结构仍成立（op 序列同构），
    索引细节以本轮 iter2 dump + 源码为准。

## Iteration 3 — 2026-10-22T23:05:00+08:00

- bottleneck_type: compute/issue（向量指令流，68.2% 实测占比；**本轮先分解剖析后定论**）
- optimization: **[#7] 因式分解式重组（mid-anchor，DESIGN §10.4-C + 分解剖析数据驱动）**
  - **分解剖析**（本轮第一优先，fla_chunk_intra_token_parallel_decompose_iter3.py 差分探针：
    P_dma/P_vec/P_full 与采纳版完全同构的 block/DMA/缓冲结构，bench shape）：
    | 成分 | 耗时 us | 占比 |
    |------|---------|------|
    | P_dma（DMA+写回+setup+调度 floor） | 828.2 | 9.1% |
    | 向量链（broadcast/sub/exp/mul/muls） | 6181.3 | **68.2%** |
    | reduce_2d（124 次/block） | 2055.8 | 22.7% |
    → iter2 的"向量指令流 issue-bound"归因被数据证实；reduce per-call ~40ns·item 无异常放大；
    DMA+调度 9.1%（含纯带宽下限 492us）非瓶颈。数据落盘 decomposition_iter3.json。
  - 重组：e_ij = 2^(g_i-g_j) = E+_i × E-_j（anchor a = SCS//2）。每 h 一次性计算
    E- → K'=k⊙E-（kgj 缓冲）、E+ → Q'=q⊙E+（q_f slab in-place 覆写，diag 先消费原始行）、
    K''=k⊙E+（e 缓冲 in-place 链）共 8 条 [16,128] 整行大 repeat 指令；per-i 链
    11 条 → 7 条（bcast+mul+reduce ×2 + muls）。**零新增 UB**（e/kgj/q_f 缓冲复用链）。
  - 静态验证（kernel_source_iter3.c，每 h 迭代）：非_reduce 指令 **137 → 85（-38%）**，
    Exp 15 → 2（**-87%**），Sub 15 → 2，Broadcast 45 → 31，Mul 46 → 34，Muls 16 不变，
    reduce_2d 31 次不变，PipeBarrier 169 → 131 随动。
  - 溢出域重审（因式分解的域变化，DBOUND 实测）：-12/token × 15 = -180 subchunk 跨度下
    mid-anchor 单侧最大 96（log2 域）< 127 fp32 上限，且 j ≤ i 保证 E+/E- 大值不同现于
    同一乘积 → 无 inf×0/inf×inf；bnd_dbound 实测 ratio=1.0000 通过。安全域文档化进
    kernel docstring：subchunk 半程衰减 |g_i - g_a| < ~127（余量 31）。
- baseline_time: 9071.4 us（events mean；per-rep 8999-9017；loop 9022.9；与 iter1/iter2 记录一致）
- candidate_time: **6913.9 us**（复测 6917.3；per-rep 6844-6856 方差 <0.2%；loop 6868.0/6868.5）
- improvement: **-23.8%（1.31×）**；有效带宽 69 → 90 GB/s（上限的 5.4% → 7.1%）
  - 对账：新向量链 ≈ 6917 - 828 - 2056 = 4033us（原 6181，**-34.8%**，介于指令数 -38% 与
    元素量 -31% 之间，与 issue+执行混合成本模型自洽 ✓）
- precision: pass（`--level all` exit 0：L0 3/3 + golden 自检 + L1 9/9 [含 SCS=8/DK=48/H=2/6/12]
  + L2 10/10 正确拒绝 + Boundary 4/4，全部 ratio=1.0000，max_abs 与 Stage 2 基线一致量级）
- adopted: **yes**
- rollback_reason: n/a
- next_hint:
  - 剩余结构（bench shape，6917us）：向量链 4033（58%）+ reduce 2056（30%）+ DMA/setup 828（12%）。
  - **[下轮最大杠杆候选] reduce_2d 30%**：124 次/h 的 M=1..16 变行数调用。方向：(a) 拼接
    diag+行 reduce 的窗口（M 从 1 起步的调用合并为更大 M）；(b) 研究 reduce_2d 的 plan 选择
    （kSum M 行 × N=128 的 vcgadd 步数与 Delay 链，src/tl_templates/ascend/reduce_2d_v2.h
    Select() cost4 模型）；(c) 手动 vcgadd 第一步压缩 + 小尾巴合并（绕过 helper）。
  - 向量链 58% 的进一步压缩空间有限（85 条/h 已接近该结构下限；除非换 GEMM 卸载路径——
    DESIGN §10.4-D 研究性选项，3 项工具链风险，需用户决策）。
  - muls 标量读的 PIPE_ALL barrier（codegen 层，60 次/block）若 reduce 侧解决后仍显性，
    可重估 iter1 next_hint 的 β 向量化（T.transpose h-major β + per-h 折乘）。
  - 若两者均无 >3% 空间：连续无提升计数将达 2（iter2 回滚 + 假设下轮），Stage 3 可收敛。

## Iteration 4 — 2026-10-23T01:09:56+08:00

- bottleneck_type: reduce_2d（30% 占比复核成立）——但全部重组路径被分析 + 实测双重否决
  （结构性成本，非可优化实现缺陷）
- optimization: **reduce_2d 重组专项：三方向全量分析 + split-merge 探针证伪（kernel 零改动）**
  - (a) 窗口合并/拼接：**dst 连续性墙**——reduce_2d v2 dst 必须是 M 连续元素（codegen 只传
    基址偏移、helper packed 写），Aqk/Akk 下三角输出在矩形 staging 内不可能跨行连续；
    零填充方形方案 cost4 8384 vs 5834（+44%，StepCost4 8m 按行收费）排除；unpack 被 32B
    MTE 粒度 + 无向量 gather 原语封锁。
  - (b) plan 研究（Select/cost4 离线求值，reduce2d_plan_dump.cxx：planner 纯 constexpr 可
    host 编译）：M=2..16/N=128/Clear/ARZ=false 统一 3 步 plan，**已是模型最优**；step0
    per-row 是 vcgadd repeat-stride 寻址限制（行内 even/odd 交错与行距不可兼得），≤64 宽
    列切片可单指令 batch（Scan 路径，cost4 4/row vs 8/row）。
  - (c) 手动第一步压缩：DSL 无裸 vcgadd 部分归约原语；flat [M*16,8] 视图 cost4 3× 劣化；
    3D [M,16,8] 前端取 M=extent[1] 对 A>1 语义不成立。
  - **split-merge（(b) 衍生唯一实测候选）**：每窗口 reduce 拆 [M,64]@128 列切片×2，第二片
    clear=False merge（无 combine op；仓内测试先例验证过该语义）。模型 +27% vs emitter
    指令计数 -50% 矛盾 → 差分探针（splitreduce_iter4.py，与主 kernel 同构）实测裁决：
    **7982.0 vs 6907.3 us = +15.6%，证伪**（数值 allclose 1e-2 PASS；per-call 固定开销
    ~30ns × 调用数 31→62/h 支配指令节省）。模型方向连续第二次被实测证实（iter2 屏障、
    本轮 split）。
  - 次要 β 向量化：per-row 标量广播源 4B 行宽违反 VEC 32B 对齐（iter1 aicore 507015 同墙），
    标量 Muls 1 op vs 向量路径 3 op 在 ~58ns/op 档净负，排除。iter2 desync 叠加预期
    ~-2.6%（屏障实测 ~1.5ns/个）低于阈值且需重做手动 flag 分析，不实施。
- baseline_time: 6923.0 us（本轮 events mean；per-rep 6846-6852 方差 <0.2% 除首 rep 7579
  外；loop 6868.1；探针会话 V_cur 复测 6907.3 一致）
- candidate_time: n/a（无 kernel 改动；探针候选 7982.0 us 证伪后未上库）
- improvement: 0%（本轮无候选达到实施门槛；连续无提升计数 1）
- precision: pass（kernel 零改动，md5 9168a76b 与 impl_iter4_before.py 一致；`--level all`
  复验 exit 0：13 PRECISION_PASS + 14 BOUNDARY_PASS，ratio=1.0000 全绿，
  logs_iter4/precision_standng.log）
- adopted: no（本轮无改动可采纳）
- rollback_reason: n/a —— kernel 零改动（探针与 planner dump 均在 perf_tuning 沙箱内），
  无需回滚
- next_hint:
  - **Stage 3 收敛建议（首选）**：三成分均达结构下限——向量链 58%（85 指令/h，iter3 已压
    -38%）、reduce 30%（256 dot-product 行/h × 下三角输出布局在 reduce_2d v2 + 非A3
    ARZ=false 全屏障平台下的结构性成本）、DMA 12%（含 492us 纯带宽下限）。剩余理论空间
    均超红线或需用户决策：GEMM 卸载（DESIGN §10.4-D 研究性选项，Q'·K'^T 走 Cube，3 项
    工具链风险）；A3 平台 ARZ=true repeat-zero 延迟替代屏障（硬件依赖）；reduce_2d helper
    本体优化（src/tl_templates 共享基础设施，超算子文件红线）。
  - 若强制继续：唯一未实测候选为 iter2 desync 叠加（预期 ~-2.6% < 3% 阈值），不建议；
    连续无提升 2 次即满足中断上报条件。
  - 工具链沉淀候选（本轮 3 项，见 optimization_log Iteration 4）：planner 离线求值方法、
    128 宽行寻址限制、per-call 固定开销支配性。

## Iteration 5 — 2026-10-23T01:42:51+08:00

- bottleneck_type: sync（残余 auto 屏障 + β 标量 PIPE_ALL，**本轮实测裁决**：-2.61% 证实
  其非主要成本，"极尽全力"证据链闭环）
- optimization: **[#12 实施] iter2 desync 方案叠加到 iter3 因式分解结构**（最后一个可自主
  执行候选；iter2/iter4 两次证明模型预测不可靠，按规程实测裁决）
  - 反模式检查（本轮 Step 3，对照 performance-antipatterns.md 全清单）：唯一命中
    "冗余全局同步"（同管线相邻 tile op 间 ~131 PipeBarrier_V/h + β 标量读 60 次
    PIPE_ALL/block，被 h 循环放大）= 本轮目标；其余条目（launch core/逐行 for loop/指令
    拼接/tile 过小/CV overlap/流水双 buffer/逐块归约/正交轴/alloc_var 误分核）已在前四轮
    解决、否决或不适用。检查点要求（关闭自动同步后须生成代码 + 精度双重确认）已执行。
  - 实施（在 iter3 结构上重新推导同步点，非照搬 iter2 位置）：
    1. pass_configs AUTO_SYNC: False；
    2. 4 对手动跨管线 flag（eventId 0，每 block 内 set/wait 即时配对）：
       - v→mte2：fill(beta_h)[V] → β GM 载入[MTE2]（WAR；iter3 的 aqk_s/akk_s fill 已移至
         cast 之后，该对只耦合 beta_h，位置不受影响）；
       - mte2→v：全部 GM 载入（Q/K/GK h 循环 + β）→ 首个 V 消费（cast q_f/k_f、β cast
         copy、g2 muls）（RAW）；
       - v→s：β_f cast copy[V] → h 循环内标量预读[S]（RAW；copy_ub_to_ub 跨 dtype 实为
         AscendC::Cast = V pipe，common.h L360-377 源码实证，故单对 V_S 覆盖）；
       - v→mte3：bf16 cast[V] → 写回 DMA[MTE3]（RAW）。
    3. β 标量读改 PrimExpr 预读局部：`b{i} = beta_f[i, h] * 1.0` 守卫内提升、按名传宏
       （2D 索引 OffsetOf = i*16+h；`* 1.0` 包裹强制 CoerceScalarArg 路径零 barrier——raw
       BufferLoad 会被 codegen EmitSelectedRawScalar 无条件加 PIPE_ALL，与 AUTO_SYNC 无关，
       iter2 发现 (a)/(b) 三态 lowering 复用）。
    4. V pipe 同管线 in-place 链（e 缓冲 E-→E+→K'' 三段复用、q_f 覆写 Q'）依赖 §2.2.1
       "V pipe 天然串行" + opt_gdn_solve_tril 先例（§2.13 "dst 需交替" 冲突警告按 iter2
       证据权重裁定不采信）；精度门禁兜底验证通过。
  - 生成代码静态验证（kernel_source_iter5.c）：SetFlag/WaitFlag 各 4、**PipeBarrier 总数
    0**（iter3 的 131/h V-V 屏障 + 60/block β PIPE_ALL 全部消除）、15 个 hoisted
    `beta_f.GetValue(h_1 + 16*i)` 索引正确、热循环 Broadcast/Mul/Muls/reduce_2d 指令序列
    与 iter3 同构（纯同步差分，无计算路径变化）。
- baseline_time: 6923.7 us（events mean；per-rep 6847-6855 + 首rep 7576；loop 6869.3；
  与 iter3/iter4 记录 6913.9-6923.0 一致，基线确认）
- candidate_time: 6742.8 us（events mean；per-rep 6658-6671 + 首rep 7433，方差 <0.2%；
  loop 6687.4）
- improvement: **-2.61%**（loop 口径 -2.65%）—— 低于 3% 采纳阈值，**回滚**
  - 对账：iter4 模型预期 ~-2.6%（~588 可移除屏障/item × ~1.5ns）与实测吻合——本轮为模型
    预期首次与实测一致，但结论仍按实测口径（iter2 -2.3% vs 预期 2-4×、iter4 模型+27% vs
    实测+15.6% 方向相反，两次教训）。屏障在 iter3 结构上仍非主要成本（向量链 issue/执行
    主导），desync 技术成功（barrier 131/h→0）但收益天花板 ~2.6%。
- precision: pass（desync 版 `--level all`：13 PRECISION_PASS + 14 BOUNDARY_PASS（含 L2
  正确拒绝），ratio=1.0000 全绿，max_abs 与 Stage 2 基线一致量级；SCS=8 守卫结构 / H=2,6,12
  自适应 BH / bnd_dbound 因式分解域哨兵均通过，logs_iter5/precision_all.log）
- adopted: **no**
- rollback_reason: 改进 -2.61% 低于 3% 采纳阈值（改进真实、精度通过但不足阈值）。已回滚
  impl_iter5_before.py（md5 9168a76b = iter3 采纳版逐字节复原）；回滚后复测确认：
  `--level all` 13 PRECISION_PASS + 0 fail（logs_iter5/precision_rollback_confirm.log）+
  bench 6915.3 us（rollback_confirm_iter5.json，基线恢复）。desync 候选完整保留：
  impl_iter5_desync_candidate.py + kernel_source_iter5.c（零 PipeBarrier 结构样本）。
- **Stage 3 收敛判定（consecutive_no_improvement = 2：iter4 零可实施候选 + iter5 低于阈值）**
  - 最终性能：**~6915-6924 us @ (1,8192,64,128)**（events mean 口径；vs Stage 2 交付基线
    19150.1 us 累计 **-63.9%（2.77×）**；有效带宽 32→90 GB/s = 实测上限 1263 GB/s 的
    7.1%；距纯带宽下限 492us 14.1×）
  - 三成分结构下限证据链（decomposition_iter3.json + iter4/iter5 专项否决）：
    - 向量链 ~4033us（58%）：85 指令/h（iter3 因式分解 -38%）；进一步压缩只剩 GEMM 卸载
      （研究性路径）；
    - reduce_2d ~2056us（30%）：iter4 三方向全量否决（dst 连续性墙 / planner 已是模型最优
      / split-merge +15.6% 实测证伪）；
    - DMA/setup ~828us（12%）：含 492us 纯带宽下限，剩余空间 <336us 且受 3-active-dim
      GM 布局限制；
    - sync 残余：iter5 实测消除全部屏障仅 -2.6%（屏障成本 ~1.5ns/个、被流水掩盖，
      iter2/iter5 双重实证）。
  - 剩余不可自主执行选项清单（均超红线或需用户决策，Stage 3 就此收敛）：
    1. **GEMM 卸载研究性路径**（DESIGN §10.4-D）：Q'·K'^T 走 Cube 管线，3 项工具链风险，
       需用户决策后立项；
    2. **A3 平台 ARZ=true**：reduce_2d repeat-zero 延迟替代屏障，硬件平台依赖；
    3. **reduce_2d helper 本体优化**：src/tl_templates 共享基础设施，超出算子文件红线。
- next_hint: n/a（Stage 3 收敛；如需突破 6920us 量级，由用户在上述 3 项中决策后重启）

## Iteration 6 — 2026-10-23T12:55:00+08:00（用户点名：msprof 流水图分析 + 编译器限制 vs 优化位置判定）

- bottleneck_type: **向量链内 AscendC::Broadcast 指令成本（实测占 kernel 29%）**——优化位置已精确定位，
  但消除手段在当前工具链不可达（DSL 无 stride-0 源二元乘原语）；其余结构维度（ILP/屏障/DMA 重叠/
  block 划分）全部实测证伪。**判定：找到了优化位置，但被编译器/工具链封锁；kernel 文件零改动。**
- optimization: **Part A msprof 流水图剖析 + 三轮差分标定探针（kernel 零改动，全部在 perf_tuning 沙箱）**
  - **msprof 采集**（prof_iter6/，--application 稳态 20 次连发，PipeUtilization +
    ArithmeticUtilization 双 metric 组；msprof 解析步骤需 LD_LIBRARY_PATH=/usr/local/sqlite3/lib
    否则 metric 解析静默失败——工具链坑）：
    | 指标（稳态 20 次均值） | 数值 | 口径 |
    |---|---|---|
    | task wall | 6914.4 us | ascend_task.db（与 bench 6916 交叉吻合）|
    | AIV 核占用 | **97.97%** | aiv_total_time/wall；aiv_total_cycles 487.2M = 40 核×1.8GHz×wall×97.9%（40 AIV 全程满载，204.8 items/核无空隙）|
    | **Vector pipe busy** | **80.0%** | aiv_vec_time 5529.8us |
    | Scalar pipe | 17.1%（重叠） | SetMask 配置 + 循环控制 |
    | MTE1/MTE2/MTE3 | 0 / 4.7% / 2.2% | MTE2+MTE3 busy 480.6us ≈ 492us 纯带宽下限（DMA 本身满效率且远非瓶颈）|
    | 向量 FLOP | 4.917G/launch | PMU aiv_vector_fops；10.08 FLOP/cycle/核 持续，vec busy 期间 12.35 |
    | Cube 核（AIC） | 67.5us（空转 scalar） | MIX_AIC_1_2 的 AIC 子块——**20 个 Cube 核全程空闲** |
    | AI Core 频率 | 1800 MHz | msprof freq 事件（800MHz 空闲态切换） |
  - **ILP 交错探针**（ilp_cal_iter6.py，7 变体同构差分；nosync=纯依赖链无屏障，仅计时）：
    P0_dma 830.3 / P1 串行链+AUTO 4879.5 / P2 串行链无屏障 4777.1（屏障仅 -2.6%，三重复证）/
    **P3 交错独立链（h-pair 双缓冲、RAW 距离≥2）4663.3（仅再 -2.9%）** / P6 交错+reduce 7097.9
    （比采纳版差 +2.8%）→ **910B3 向量管不会重叠交错排放的独立指令**，链上 ~102 cyc/op 为固有
    串行成本，非调度损失。P5 交错+AUTO 数值校验失败系探针 op 序 bug（Aqk reduce 晚于 Akk mul
    覆写 pq，不影响计时结论）。
  - **向量峰值/分指令标定**（veccal_iter6.py，5 变体）：**Mul [16,128] repeat-32 = 48.9 cyc/op
    （42 dst-elem/cycle）**；per-i 小 repeat（2-30）mul 同为 ~48.2 cyc/op（平坦）；repeat-128 大块
    16.1 elem/cycle（大 repeat 反而更差）；**Broadcast [16,128] = 260.8 cyc/op（7.9 elem/cycle，
    比 Mul 慢 5.3×/元素）**；exp 探针被 DCE（无效，由链差分夹逼为小项）。
  - **广播占比直接实测**（bcastshare_iter6.py）：链上真实 124 次广播（31/h×4，行分布 [1..16]×128）
    = **1986.5us = 链的 50.3% = 全 kernel 的 28.7%**，平均 140.8 cyc/op，与 260.8@16rows 线性
    吻合（**~16.2 cyc/行**）；链减广播 = 1310.7us（mul/muls ~48 cyc/op 平坦区 + exp/sub 小项）。
  - **广播消除路径逐一否决（工具链证据链）**：
    1. DSL `binary_op` 强制 size 相等（ascend_tile.py:1070-1072 assert）——区域操作数无形状广播；
    2. `T.tile.mul` 的 stride-0 源（srcRepeatStride=0）无 DSL 原语；C++ helper `tl::ascend::mul_mask`
       已存在（common.h:1611-1624，全 stride 暴露）但无 intrinsic 注册；
    3. raw `T.call_extern("handle", "tl::ascend::mul_mask<...>", ...)` 走 CopyCodegen copy 约定，
       未注册名直接生成 `not implemented yet`（codegen_ascend.cc CopyCodegen else 分支）；
    4. `row_expand_mul`（PTO-only，RowExpandMulCodegen LOG(FATAL)）与 `row_expand_mul_experiment`
       （brcb+mul_mask，AscendC 可用）均为**每折叠行（256B）标量**语义，β 折叠进 K'' 需 β 列
       [16,1]→[32] 预展开（窄列 gather 撞 4B/32B 对齐墙，iter1 aicore 507015 同源）——死局；
    5. `brcb` = 标量→32B 块 lane 展开（测试 golden + EmitSelectedRawRowExpand 实证），**非行复制**；
    6. MTE copy 无复制语义（β "point→region" 先例实为自然 GM 窗口 strided copy）；`gatherb` 按
       32 位元素 gather，偏移表自身需行复制（循环依赖）。
  - 结论回答用户判定问题：**"优化位置找到了"**（Broadcast 占 29%，指令级 5.3× 成本差），
    **"但当前编译器/工具链不支持修复"**——需要新增 `T.tile` 原语（暴露 stride-0 源二元乘 /
    mul_mask），属 tilelang-ascend-tile-api skill 流程（前端+codegen+helper+测试），超本轮
    算子文件红线（只碰 fla_chunk_intra_token_parallel_*）。预期收益（元素吞吐模型）：消除广播
    → 链 -50% → 全 kernel 约 -29%（~4900us 量级）。
- baseline_time: 6902.7 us（本轮 session events mean，ilp_cal 内复测；msprof task wall 6914.4us
  交叉一致；与 iter3-5 记录 6913.9-6923.7 一致，基线确认）
- candidate_time: n/a（无 kernel 改动；快筛探针 P3/P6 均未达实施门槛：P6 +2.8% 劣化）
- improvement: 0%（本轮无可实施候选；连续无提升计数 3：iter4 + iter5 + iter6——但本轮为
  "定位+封锁证据"而非"无候选"：优化位置明确、封锁点明确）
- precision: pass（kernel 零改动，md5 9168a76b 与 iter3 采纳版逐字节一致，无需复验）
- adopted: no（无改动可采纳）
- rollback_reason: n/a（kernel 零改动）
- 剩余行动项（均超本轮红线，按优先级）：
  1. **新增 stride-0 源二元乘 tile 原语**（tilelang-ascend-tile-api skill；C++ helper mul_mask
     已就绪，需注册 intrinsic + codegen + 测试）：预期 -29%（最大单杠杆，本轮证据链完整）；
  2. **GEMM 卸载**（DESIGN §10.4-D）：msprof 实证 20 Cube 核全程空闲，每头本质为 16×16×128
     下三角 matmul——Cube 侧承接 Q'·K'^T 后向量侧仅剩 reduce/E±，需用户决策（3 项工具链风险）；
  3. A3 平台 ARZ=true（reduce_2d repeat-zero 替代内部 pipe_barrier，硬件依赖）；
  4. reduce_2d helper 本体优化（共享基础设施红线）。
- 产物：
  - prof_iter6/{run_profile.py, out_pipe/, out_arith/, msprof_summary_iter6.json}（msprof 双 metric
    组稳态数据 + 提炼摘要）
  - fla_chunk_intra_token_parallel_ilp_cal_iter6.{py,json}（ILP 交错 7 变体裁决）
  - fla_chunk_intra_token_parallel_veccal_iter6.{py,json}（向量峰值/分指令成本标定）
  - fla_chunk_intra_token_parallel_bcastshare_iter6.{py,json}（广播占比直接实测）
  - 工具链坑（journal 候选）：msprof 解析需 LD_LIBRARY_PATH 含 /usr/local/sqlite3/lib；
    @tilelang.jit kernel 体缺 `return main` → func.script() NoneType（报错具有误导性）；
    closure factory 内定义 jit kernel 可用但首版探针因缺 return 失败排查耗时。

## Iteration 7 — 2026-10-23T16:10:00+08:00（用户指令：继续尝试优化——GEMM 卸载研究性路径立项执行）

- bottleneck_type: **向量链 broadcast+reduce_2d（iter6 定位 29%+30%，DSL 无 stride-0 原语被封锁）→ Cube GEMM 卸载**
  （DESIGN §10.4-D；msprof iter6 实证 20 Cube 核全程空闲）
- optimization: **GEMM 卸载主路径落地（fp32 操作数 + block-diagonal 批量 gemm_v0 + 显式 GM workspace 中转 + select 掩码）**
  - 数学（iter3 因式分解版已有）：per-head Aqk_h = tril(Q'_h @ K'_h^T)、Akk_h = strict_tril((β⊙K'')_h @ K'^T)；
    β 在 V 侧折进 K''（per-row 标量 muls，生产版 beta_f[i*16+h, 0] BufferLoad 模式），Akk 无需 GEMM 后折乘。
  - **GEMM 结构（选定）**：GGRP=4 head/group 的 block-diagonal 批量——A=[Q';K''] 四头堆叠 [128,DK]、B=K' 四头堆叠
    [64,DK]，一次 gemm_v0(transpose_B) → C[128,64]，取对角 [32,16] 块（4× FLOP 浪费换 4× 更少调用：ggrp=1 实测
    +168us 劣化证实调用开销主导）；fp32 操作数（bf16 舍入单致 Akk matched-ratio 0.9848<0.99，CPU 模拟证实为
    bf16 固有；fp32 与 split-bf16 中转字节数相同但单 gemm 调用 + 精度与 iter3 版逐位可比）。
  - 数据通路：V 侧 per-head 装载+构造（q/k/gk 行内装载，峰值 UB ~124KB）→ fp32 累计缓冲 a_agg/b_agg →
    GM workspace_a/b（MTE3）→ C 侧 GM→L1（MTE2）→ gemm → L0C→GM workspace_r（fixpipe）→ V 侧读回
    select 掩码 → staging → 写回。AUTO_CV_SYNC 生成跨核 flag。
  - **掩码**：arith_progression + broadcast(axis 0/1) + compare + select(VSEL_TENSOR_SCALAR_MODE)——上三角垃圾在
    深衰减下可 ±inf（E+·E- 至 2^180），乘法掩码 inf×0=NaN，select 替换为 0 与 golden masked_fill 语义一致
    （bnd_dbound ratio=1.0000 通过）。
  - 本轮修复的 4 个工具链/结构问题（探针 8 轮迭代定位，详见 optimization_log）：
    1. **AUTO_CV_SYNC 按 buffer 名字面量 "workspace" 识别**（ascend_combinecv.cc FetchWorkspaceName）——
       非 "workspace" 命名静默无跨核同步（WsA/WsB 命名 → 竞态全零）；
    2. **memory planner 对 region-dst 写不延长 liveness**——跨循环累计缓冲（a_agg/b_agg）必须先整块 fill
       锚定，否则被 planner 与循环内 bc_row/e 空间复用，fp32 字节被按 bf16/fp32 错读（ws_b NaN 调查链）；
    3. T.copy(src, dst) 实参序（结果读回两处写反 → 全零输出）；
    4. 转置域 β 折乘链（transpose+broadcast+mul+select+transpose）存在可复现 23-cell/head 归零
       （a_m 行 3-4；掩码终态 dump 正确、链路隔离验证正确、fresh mask/VECTOR_MASK_REUSE=False 均无效，
       根因未定位）→ 改用 β→K'' 折乘彻底绕开该链。
- baseline_time: 6922.8 us（iter3 采纳版复测 events mean；per-rep 6839-6852+首rep 7616；loop 6870.1；
  与 iter3-6 记录 6913.9-6923.7 一致；baseline_iter7.json）
- candidate_time: **3517.7 us**（events mean；steady per-rep 3440-3452+首rep 4181；loop 3464.6；
  candidate_iter7.json；探针轮 ggrp4-fp32 3454.8 / ggrp1-fp32 3622.9 对照）
- improvement: **-49.2%（1.97×，events）；loop 口径 -49.6%**；有效带宽 90→177 GB/s（上限 1263 GB/s 的
  7.1%→14.0%）；vs Stage 2 基线 19150.1 累计 **-81.6%（5.44×）**；距纯带宽下限 492us 7.1×（原 14.1×）
- precision: pass（`--level all` exit 0：**13 PRECISION_PASS + 14 BOUNDARY_PASS / 0 FAIL**（logs_iter7/
  precision_all.log）——L0 3/3 + golden 自检 + L1 9/9（含 SCS=8/DK=48/H=2,6,12 自适应 BH→GGRP 2/4）+
  L2 10/10 正确拒绝 + Boundary 4/4（zero/inf_q/nan_beta/dbound 全 ratio=1.0000）；
  探针轮 randn/dbound/infq 三场景 vs 采纳版逐位对比 Aqk bit=1.0000）
- adopted: **yes**（md5 6d919539；快照 impl_iter7_candidate.py + 备份 impl_iter7_before.py=iter3 版 9168a76b）
- rollback_reason: n/a
- **Part C 效率构成（msprof 稳态 20 次，prof_iter7/msprof_summary_iter7.json）**：wall 3504.4us；
  | 维度 | iter6（纯向量） | iter7（GEMM 卸载） |
  |---|---|---|
  | AIV 核占用 | 97.97% | 96.32% |
  | AIV Vector pipe | 80.0%（5530us） | **30.3%（1063us）** |
  | AIV Scalar pipe | 17.1% | 20.8% |
  | AIV MTE2/MTE3 | 4.7%/2.2% | 17.5%/9.6%（workspace 中转 1.5GB r+w） |
  | AIC（Cube）busy | 67.5us 空转 | **80.07%（2806us；cube_utilization 79.99%）** |
  | AIC mac/scalar/mte1/mte2/fixpipe | — | 7.1%/19.4%/3.6%/7.4%/4.0% |
  向量链 4033→1063us（-74%，broadcast/reduce_2d 全部消失）；新增 AIC 流水 2806us（其中纯 mac 仅 7%，
  gemm 编排/装载/同步主导）+ 中转 DMA 显性化——**新结构瓶颈 = 每块 C/V 串行化**（workspace 同步 + L1/L0
  装载编排），AIC 80% busy 中非 mac 占 73 个百分点。
- next_hint:
  - **跨块流水**（最大剩余杠杆）：当前 V(构造)→ws→C(gemm)→ws→V(后处理) 在块内串行，AIV/AIC 互等；
    双缓冲 workspace + 放松同步（或 Expert 模式 cc_sync 核间流水，flash_attn_bhsd_cc_sync 先例）可让
    块 n+1 的 V 与块 n 的 C 重叠——AIC 非 mac 占比 73% 表明编排而非算力是墙，理论可再 -30~40%。
  - 中转带宽：fp32 操作数 384KB/block r+w ≈ 1.5GB；bf16+split-A/B 同字节但 3 gemm 调用（调用开销实测
    +168us/4调用，净负）——维持 fp32。
  - GGRP=8 单调用对 fp32 不可行（L0A 128KB>64KB 单步预算）；kL0Size=64 降 K 步可试（2 K-步 ping-pong，
    A 64KB→32KB 槽 ×2——需模板 static_assert 复核，预期收益 <100us）。
  - β 标量 muls（64/vid·block ≈ 180-360us）：广播化需 [16,1] 列源（32B 对齐墙，iter1 同源）——除非新增
    tile 原语（stride-0 源二元乘，iter6 行动项 #1 同族）。
  - 工具链 journal 候选（本轮 4 项）：workspace 命名同步识别、region-dst liveness 缺陷、转置域 β 链
    23-cell 归零（未定位根因，需最小复现立项）、bf16 GEMM 操作数精度门限数据。
- 产物：fla_chunk_intra_token_parallel_{gemprobe,maskdiag,mismatch,failclass,bfsim,fp32forensics}_iter7.py
  （探针/诊断链）、gemmprobe_iter7_ggrp{4_fp32,1_fp32}.json、baseline/candidate_iter7.json、
  prof_iter7/{run_profile.py,run_msprof.sh,msprof_summary_iter7.json}、logs_iter7/precision_all.log、
  impl_iter7_{before,candidate}.py

## Iteration 8 — 2026-10-23T16:55:00+08:00（用户指令：继续尝试优化——跨块 C/V 流水，iter7 next_hint 首选路径）

- bottleneck_type: **块内 V(构造)→ws→C(gemm)→ws→V(后处理) 三段串行**（iter7 判定：AIC 80.07% busy 但 mac 仅
  7.1%，编排等待主导；每核组的相邻块互锁步——V 在 C gemm 期间空转、C 在 V 构造/后处理期间空转）
- optimization: **固定核网格 + 任务跨块软件流水（2-任务延迟 drain + 双缓冲中转 + 手动/自动混合跨核 flag）**
  （fla_chunk_o iter6 已验证模板，同工具链家族；iter7 计算体逐语句保留，仅重排调度结构）
  - **网格重构**：`T.Kernel(CORE_EFF=min(20,NBLK))` 固定核（20 核组），跨步任务映射 `tid = cid + t*CORE_EFF`，
    `TPT = ceil(NBLK/CORE_EFF)`（bench 形状 NBLK=4096 → 每核 205 任务）；越界任务 branch-free Select 重定向到
    本核首个任务（cid）——重定向是逐位相同的重计算、垃圾 drain 会被真值覆盖，循环内零 IfThenElse（每个
    IfThenElse 经 AUTO_SYNC 展开为 2× PipeBarrier<PIPE_ALL>）。
  - **流水时序**（单循环体三相位，V/C 由 AUTO_CV_COMBINE 按 op 类型拆分到两核）：
    ```
    迭代 t:  V 核                                C 核
    ─────    ──────────────────────────          ─────────────────────
    prologue produce(0)→ws[0] ─auto flag→  gemm(0)→o_relay[0]; set FIX(3)
             [垃圾 drain(2%NSLOT)，无 wait，后覆写]
             produce(1)→ws[1] ─auto flag→  gemm(1)→o_relay[1]; set FIX(4)
             [垃圾 drain(3%NSLOT)，同上]
    loop t:  wait FIX(t-2) → drain(t-2)          （gemm(t-1) 尾部 ∥ V drain(t-2)）
             从 o_relay[t%2] 读回+select+写回
             produce(t)→ws[t%2] ─auto flag→       wait relay(t) → gemm(t)→o_relay[t%2]
                                                  set FIX(t%2) ∥ V 下一迭代 drain(t-1)…
    ```
    稳态：V 领先 C 至多 2 个迭代；V 的 drain 等待的 FIX(t-2) 早已置位 → 两核均不因对方**当前**相位停转，
    墙钟 ≈ max(ΣV, ΣC) 而非 ΣV+ΣC。peeled 迭代 0/1 各带尾部垃圾 drain（无 wait）——避免 t=0 死锁（produce
    必须先于首个 drain）并保持每个中转 buffer 两侧访问 site 数平衡（2 peeled + 1 loop，pass 为每对 site
    生成独立 flag ID，msprof 日志实证 id 3/4/5 各自配对）。
  - **同步策略（精确控制 C/V 边沿）**：
    - ws_a/ws_b（V→C 操作数中转）：保留 "workspace" 命名 + AUTO_CV_SYNC 自动 flag——produce 与 consume 在
      **同一循环迭代**内，pass 的 FIFO 同迭代配对恰好就是所需边沿（C gemm(t) 等 V produce(t)）；
    - 结果中转 **workspace_r → 改名 o_relay**（脱离 "workspace" 字面量 → 无自动 flag）：C→V 手动
      `T.set_cross_flag("FIX", 3+t%2)` / `T.wait_cross_flag(3+t%2, "V")`——**交替事件 ID** 让延迟 2 的 drain
      与迭代 t-2 的 FIX 配对；若用自动 FIFO flag 会同迭代配对 → V drain(t) 等 FIX(t) 等 V 自身 produce(t)
      → 死锁（fla_chunk_o 已证）。flag ID 3 起（0-2 留给自动 pass）。
  - **双缓冲 NSLOT=2 的 WAR 安全性**（无额外 flag，传递序保证）：(a) V produce(t+2) 覆写 C gemm(t) 已读的
    ws 槽——V produce(t+2) 在 V 流内晚于 drain(t)，后者等 FIX(t) 手动 flag，而 C 置该 flag 时 gemm(t) 的
    L1 装载已完成；(b) C FIX(t+2) 覆写 V drain(t) 已读的 o_relay 槽——FIX(t+2) 晚于 C 等 produce(t+2) 的
    自动 relay flag，而 V 的 produce(t+2) MTE3 在 MTE3 单 pipe 内晚于 drain(t) 的 MTE2→Vector→MTE3 链。
  - **掩码提升**：lt/slt 掩码（arith_progression+双 broadcast+双 compare）按核计算一次（iter7 每块重算
    4096 次）；β 装载/staging fills 留在各相位内。
  - **中转容量**：workspace 形状 [NBLK,...]→[CORE_EFF,NSLOT,...]（bench 形状 ~640MiB→~5MiB，常驻 L2；
    每任务中转流量不变）；jit 装饰器 workspace_idx=[4,5,6] 不变。
- baseline_time: 3513.3 us（iter7 采纳版本轮复测 events mean，steady per-rep 3438-3457；loop 3466.2；与
  iter7 记录 3517.7/3464.6 一致；baseline_iter8.json）
- candidate_time: **2125.0 us**（events mean；steady per-rep 2064-2074+首rep 2630；loop 2091.5；
  candidate_iter8.json；探针轮 2166.2/2084.1）
- improvement: **-39.5%（events）/ -39.7%（loop）**，命中 iter7 预测带 -30~40%；有效带宽 177→293 GB/s
  （ceiling 的 14.0%→23.2%）；vs Stage 2 基线 19150.1 累计 **-88.9%（9.0×）**；距纯带宽下限 492us 4.3×（原 7.1×）
- precision: pass（`--level all` exit 0 "Test Passed!"：**13 PRECISION_PASS + 14 BOUNDARY_PASS / 0 FAIL**（logs_iter8/
  precision_all.log）——L0 3/3 + golden 自检 + L1 9/9（含 SCS=8/DK=48/H=2,6,12）+ L2 10/10 正确拒绝 +
  Boundary 4/4（zero/inf_q/nan_beta/dbound 全 ratio=1.0000）；探针另做 5 次重复运行逐位一致（竞态快检）+
  bench 30 连发 per-rep 离散 <1%）
- adopted: **yes**（md5 dafbcb35；快照 impl_iter8_candidate.py + 备份 impl_iter8_before.py=iter7 版 6d919539）
- rollback_reason: n/a
- **Part C 流水效果（msprof 稳态 20 次，prof_iter8/msprof_summary_iter8.json）**：wall 2064.0us（与 bench steady
  2069 交叉吻合）；
  | 维度 | iter7（串行） | iter8（流水） |
  |---|---|---|
  | AIC busy（% of wall） | 80.07%（2805.9us） | **99.55%（2054.8us）** |
  | AIC mac | 7.07%（247.6us） | 12.11%（250.0us） |
  | AIC scalar | 19.40% | 30.62%（632.1us） |
  | AIC mte2 / fixpipe | 7.37% / 4.04% | 12.79% / 7.03% |
  | AIV Vector pipe | 30.34%（1063.2us） | 50.66%（1045.7us） |
  | AIV scalar / mte2 / mte3 | 20.85%/17.55%/9.62% | 26.44%/26.68%/15.82% |
  各 pipe 绝对时间几乎不变（工作量相同：vec 1063→1046、mac 248→250、AIC scalar 680→632us）——**墙钟 -41%
  完全来自 C/V 相位重叠**（AIC busy 80→99.55% = Cube 核饱和成为新临界路径；AIV Vector 50.7% 意味着 V 核
  有 ~49% 空闲裕量）。AIC 非 mac 仍占其忙时 87.9%（scalar 30.6% = gemm 调用编排 2 次/任务×4096）。
- 本轮工具链发现（journal 候选）：
  1. **`T.wait_cross_flag(flag, "")` 在 910B3 抛 InternalError**——CombineCV 的 AscendResourceScopeVerifier 按
     第 2 参 pipe 归属资源（ascend_combinecv.cc:840-844 → ResourceForPipeArgument；"V"→kVector），空串判
     kExplicit 报 "must be inside T.Scope"；语言层 docstring 却称 pipe 参数 "仅 A5 支持，其他架构必须留空"
     ——文档与 pass 行为矛盾，正确用法是传执行核角色（"V"/"FIX"）；
  2. AUTO_CV_SYNC 对 "Vec loop times (N) not enough to catch up cube loop times (2N)" 告警（V 单写 vs C 侧
    NG=2 双读同 buffer）：pass 在计数匹配深度（TPT 层）放置 flag，单 wait 覆盖整 buffer 读写——iter7 即
    存在的同构模式，良性（msprof sync 点 dump + 精度/稳定性实证）。
- next_hint:
  - **AIC gemm 编排为新临界路径**（AIC 99.55% busy、scalar 30.6%、mac 仅 12.1%）：减少 gemm 调用数——
    GGRP=8（fp32 受 L0A 64KB 单 K 步预算封锁，A 槽 128KB）需 kL0Size=64 双 K 步（每步 A 槽 64KB 恰好
    满预算，2 步 ping-pong）——模板 static_assert 复核后值得探针（iter7 估 <100us，但现在调用开销占比
    更高，实际收益可能放大）；
  - **V 核裕量利用**（Vector pipe 50.7%，~1000us 空闲）：下一任务 GM 预取入 UB（fla_chunk_o 先例，MTE2
    延迟藏进 drain）或吸收更多构造侧工作；
  - AIC 侧 mte2+fixpipe ~20%（ws 中转 512KB/任务）：bf16 操作数可减半但精度已被 iter7 否决（0.9848<0.99）；
    L2 常驻后延迟已非主项；
  - 距纯带宽下限 4.3×：主要余量在 AIC 编排与 gemm 调用粒度，而非 DMA 带宽。
- 产物：fla_chunk_intra_token_parallel_pipeline_probe_iter8.py（探针：2 形状正确性+5 次重复逐位稳定）、
  baseline_rebench_iter8.py + baseline/candidate_iter8.json + probe_iter8.json（探针轮）、
  prof_iter8/{run_profile.py,run_msprof.sh,parse_summary.py,out_pipe/,msprof_summary_iter8.json}、
  logs_iter8/precision_all.log、impl_iter8_{before,candidate}.py

## Iteration 9 — 2026-10-23T18:35:00+08:00（用户指令：继续尝试优化——iter8 next_hint 三方向逐一立项，全部否决，收敛论证）

- bottleneck_type: **AIC 编排为临界路径**（iter8 定量：AIC busy 99.55%，scalar 30.6%/632us、mac 12.1%/250us、
  mte2 12.8%/264us、fixpipe 7.0%/145us、mte1 6.2%/128us；wall≈10.0us/task，各 pipe 时间和仅 ~6.9us，
  差值 ~3.1us/task 为 gemm_v0 模板内部 flag 链与 pass flag 的串行往返延迟）
- optimization: **三候选立项执行，全部实测/论证否决，无采纳**（iter8 next_hint 方向 1/2/3 全覆盖）
  - **候选 A：GGRP=8 + kL0Size=64 双 K 步（gemm 调用 2→1/任务）——静态断言死路 + 成本核算否决**。
    (i) 模板实测（fla_chunk_intra_token_parallel_ggrp8probe → src/tl_templates/ascend/common.h:1351
    static_assert）：gemm_v0 的 L0A/L0B ping-pong 预算 = 64KB/kNumSteps，**任何多 K 步调用每槽仅 32KB**
    ——fp32 M=256（GGRP=8）单步 128KB>64KB、双步 64KB>32KB，kL0Size=32 四步 32KB≤32KB 虽可编译，
    但 (ii) 成本核算否决：iter7 ggrp1-vs-ggrp4 实测（3622.9 vs 3454.8us，+6 调用/任务）定价单次 gemm
    调用边际成本仅 **~0.14us**，而 GGRP=8 块对角浪费使 mac×2（+250us）+ fixpipe 字节×2（+~70us）
    + 每步 PipeBarrier×2 —— 净负。iter8 next_hint 的"调用开销占比更高收益放大"预期被证伪。
  - **候选 B：分组 ping-pong L1/L0C 槽（消跨组 WAR/RAW 等待）——三重独立封锁**。
    (i) **AUTO_CV_SYNC site-count 奇偶约束**：pass 按 buffer 对 cube/vec 的访问"语句位点数"严格配对
    （分组加载 = ws_a cube 6 位点 vs vec 3 位点 → "Mismatch in sync points" 编译错）；V 侧按 g 对齐拆分
    produce 需交错 head→group 映射 + 双份 MTE3，代价>收益。(ii) **zN 分形布局封锁 region 操作数**：
    L1 gemm 操作数为 zN 分形布局（common.h LayoutL1=zN），flat 偏移的 region 切片（A_l1[16384]）≠
    子块分形基址 —— 实测确定性强腐蚀（Aqk 0.87/Akk 0.54，5 次重复逐位一致）；gemm 操作数必须整 buffer。
    (iii) **模板内部 FIX_M 等待封锁结构重叠**：gemm_v0 prologue 自带 WaitFlag(FIX_M)（等所有先前
    fixpipe 完成），分 C_l0c_g0/g1 名字只能省 pass 生成的冗余 flag 对（~1-2%），mma(g1)∥fixpipe(g0)
    的结构性重叠不可达。B 系列（含 A_l1 NG-wide 单拷贝 + region 操作数变体、C_l0c 分裂变体）全部回滚。
  - **候选 C：bf16 o_relay（fixpipe 字节减半）——精度位级全过、性能 -4.7% 否决**。
    实现：o_relay/V 侧 r_ub 转 bf16，fixpipe fp32→bf16 在途转换，V 侧读回后 CAST_NONE 转 fp32 再
    in-place select（VSEL 910B3 拒绝 bf16 操作数；跨 dtype GM→UB 拷贝不支持）→ fp32 staging 的
    CAST_RINT 对精确可表示值为恒等。**精度：fixpipe 舍入与 CAST_RINT 位级相同**——randn/dbound/infq
    + H=2/6/12 + DK=48/32 全部 bit=1.000000（vs iter8 采纳版逐位对比），golden ratio 全 1.0000；
    SCS=8 的 bf16 relay 行宽 16B<32B DMA 对齐（实测腐蚀）→ SCS%16!=0 回退 fp32 relay 后同样位级全过。
    **性能：2229.9us vs 同会话 iter8 复测 2129.8us = -4.7% 劣化**（fixpipe 转换模式开销 + V 侧每头
    +2 cast 的 flag/barrier 编排超出 V 裕量）→ 按门禁否决回滚。
  - **V 核 GM 预取（next_hint 方向 2）：论证 0 收益**——wall=AIC 串行链（busy 99.55%），稳态 V 领先
    C 至多 2 任务、C 从不因 V 停转；V 侧任何削减不降 wall。方向 2 关闭。
- baseline_time: 2139.7 us（iter8 采纳版本轮复测 events mean，steady per-rep 2062-2074+首rep 2799；
  loop 2089.6；baseline_iter9.json；与 iter8 记录 2125.0/2091.5 一致，噪声 <0.7%）
- candidate_time: 候选 C 2229.9 us（同会话对照 2129.8）——**-4.7% 劣化，未达门禁**；候选 A/B 未过编译/正确性关
- improvement: **0%（无采纳）**；累计维持 19150.1→2125.0（-88.9%，9.0×）；距纯带宽下限 492us 4.3×
- precision: pass（回滚态全量门禁，logs_iter9/{precision_l0,l1,l2,boundary}.log 分级执行——单进程
  --level all 两次遭宿主 memcg 静默 OOM 杀，分级跑全绿）：**L0 3/3 + golden 自检 + L1 9/9（含
  SCS=8/DK=48/H=2,6,12）+ L2 10/10 正确拒绝 + Boundary 4/4（全 ratio=1.0000）= 13 PRECISION_PASS +
  14 BOUNDARY_PASS / 0 FAIL**，与 iter8 门禁完全一致
- adopted: **no**（三候选全否决；主文件回滚至 iter8 采纳版 md5 dafbcb35 已验证）
- rollback_reason: 候选 C 实测 -4.7% 劣化（<3% 门禁且为负）；候选 A 模板 static_assert 死路 + 边际
  调用成本 0.14us 证伪收益模型；候选 B 三重封锁（site-count/zN region/模板 FIX_M）残值 ~1-2% 不值风险
- **收敛论证（DSL 层已达平台期）**：
  1. wall = AIC 串行链 10.0us/task，其中必要数据搬运（MTE2 1.29 + MTE1 0.62 + mac 1.22 + fixpipe
     0.71 ≈ 3.8us）+ 模板/pass flag 编排与往返延迟（~6.2us）——后者主体在 gemm_v0 C++ 模板内
     （~20 flag 类操作/调用 × 2 调用/任务，框架代码，红线禁改）；
  2. 调用数削减（GGRP=8）被 L0A ping-pong 预算结构性封锁且边际收益 0.14us/调用 < 浪费项；
  3. 组间重叠被模板内部 FIX_M 等待封锁；操作数粒度被 zN 分形布局封锁；
  4. 中转字节削减（bf16 relay）实测负收益（转换/编排开销 > 字节节省）；
  5. V 侧裕量（49%）不可兑现为 wall（AIC 饱和）；
  6. 剩余可达路径均 <3% 门禁。**进一步提速需框架级改动**（gemm_v0 模板 flag 编排瘦身 / L0A 预算 /
     region-精确依赖分析 / 跨 dtype DMA），超出算子层红线，建议另立框架 issue 而非继续算子迭代。
- 本轮工具链发现（journal 候选，含 2 项正反数据点）：
  1. **gemm_v0 L0A/L0B ping-pong 预算公式**（common.h:1344 kL0Budget=64KB/(steps>1?2:1)）——文档级
     约束：多 K 步调用槽减半，fp32 M>128 不可达；建议在 gemm_v0 docstring 补充该预算规则；
  2. **L1 gemm 操作数 zN 分形布局使 region 切片非法**（flat 偏移≠子块分形基址，确定性静默腐蚀）——
     建议在 gemm_v0 断言操作数为整 buffer 或文档明示；
  3. **AUTO_CV_SYNC workspace 位点数奇偶约束**（cube/vec 语句位点严格配对，ascend_combinecv.cc:380）；
  4. **VSEL 910B3 拒绝 bfloat16**（Select dst 需 half；bf16 select 需先 CAST_NONE 转 fp32）；
  5. **跨 dtype GM→UB 拷贝不支持**（copy_gm_to_ub<T> 单模板参数）；UB→UB 跨 dtype cast-copy 支持；
  6. **fixpipe fp32→bf16 舍入与 CAST_RINT 位级相同**（正面数据点：中转 dtype 减半的精度风险为零，
     未来框架修复转换开销后可直接复用本轮精度结论）；
  7. **T.copy(src,dst) vs T.tile.cast(dst,src) 参数序相反**（iter7 已 journal，本轮再犯：回退路径
     拷贝写反致 SCS=8 腐蚀，生成码读出定位）——建议语言层统一或 lint；
  8. trace 限制三则：prim_func 内列表推导式分配（ForFrame 不可迭代）、if 块内名字跨块不可见、
     `for x in range(n)` 落为 TIR 循环（Var 索引）而非 Python 展开；
  9. 宿主 memcg 静默 OOM：多形状连编（~8+ 次 bisheng）会杀进程且无 traceback——长门禁建议
     PYTHONUNBUFFERED=1 + 分级/分批执行（本轮 logs_iter9 即此法）。
- 产物：fla_chunk_intra_token_parallel_bf16relay_probe_iter9.py（候选 C 位级/性能探针，分 stage）、
  {ggrp8_probe,pingpong_probe,bf16relay_probe}_iter9*.log（三候选证据链）、
  fla_chunk_intra_token_parallel_kernel_source_iter9_scs8.c（SCS=8 生成码诊断）、
  baseline_iter9.json + candidate_iter9.json、logs_iter9/{precision_l0,l1,l2,boundary}.log、
  impl_iter9_before.py（=iter8 采纳版备份，回滚源）

## Iteration 10 — 2026-10-24T03:40:00+08:00（用户指令：框架级优化继续调——削减 C 侧 gemm 编排开销，路径 A/B 全探索，无采纳，收敛于 V 侧手动化缺口）

- bottleneck_type: **gemm_v0 模板 + pass 的本地 flag 编排**（iter8/9 定量：每 task 2 次 gemm_v0 = pass 外围 16 条 + 模板内部 28 条 + 6 PipeBarrier；wall 10us/task 中 pipe 忙碌和 6.9us、空转 3.1us = 串行等待边界 ~12/task 的 pipe 往返延迟）
- **Step 1 编排构成量化表**（iter10 dump fla_chunk_intra_token_parallel_kernel_source_iter10.c，C 侧主循环每 task = 2 gg × 下列每 gg 序列）：

  | 组成（每 gg = 每 gemm 调用） | 条数 | 阻塞 M 流？ | 语义 | 冗余判定 |
  |---|---|---|---|---|
  | pass: SetFlag/WaitFlag<M_MTE2>(7) 对 + PipeBarrier<MTE2> | 3 | 否（立即过） | copy_gm_to_l1 前 | 冗余 |
  | pass: copy_gm_to_l1 A(64KB) + B(32KB) | 2 | — | MTE2 数据搬运 | 必要（1.29us/task 的一半） |
  | pass: SetFlag/WaitFlag<FIX_M>(0) 对 | 2 | 是（等上轮 fixpipe） | L0C WAR | 必要 |
  | pass: SetFlag/WaitFlag<MTE2_M>(1) 对 + PipeBarrier<M> | 3 | 是（等 MTE2 完成） | L1 RAW | 必要但与模板 MTE2_MTE1 语义重叠 |
  | 模板 prologue: MTE2_MTE1 对 | 2 | 否 | MTE1 等 L1 | 与 pass MTE2_M 重叠 |
  | 模板 prologue: FIX_M(0) 对 | 2 | 否（立即过） | 与 pass FIX_M(0) 背靠背重复 | 冗余 |
  | 模板 prologue: SetFlag<M_MTE1>(0/1) priming | 2 | 否 | slot 释放预置 | slot1 单步不用→1 条冗余 |
  | 模板循环: WaitFlag<M_MTE1>(0) + copy_l1_to_l0a/l0b + MTE1_M 对 + PipeBarrier<M> + mma + SetFlag<M_MTE1>(0) | 4+3 数据 | 是（MTE1_M 等 L0 加载） | L0 RAW/WAR | 必要 |
  | 模板 epilogue: WaitFlag<M_MTE1>(0/1) + MTE1_MTE2 对 + M_FIX 对 | 6 | 否（立即过/队列内） | 事件闭环 | slot1 的 1 条冗余 |
  | pass: SetFlag/WaitFlag<M_FIX>(2) 对 + PipeBarrier<FIX> | 3 | 否（FIX 队列内） | 与模板 epilogue M_FIX(0) 重复 | 冗余 |
  | pass: copy_l0c_to_gm(32KB) | 1 | — | fixpipe | 必要 |
  | **合计/调用** | **38 条 flag + 6 barrier + 5 数据** | M 流阻塞点 4 | | 冗余 ~14 条 |
  | 每 task ×2 调用 + CrossCore 3（wait WS×2 + set FIX） | **76 flag + 12 barrier** | | | |

- optimization: **路径 A 全探索（kernel 侧绕过/重写编排），4 个候选全部执行到底，全部否决或未达门禁**
  - **候选 A1（mma 分解）：T.copy(L1→L0A/L0B) + T.mma 替代 gemm_v0**（探针 mmadecompose_probe：模板整体消失，pass 对 6 个 op 插 16 条+3barrier/调用 = -27% flag；两形状 bit 级 exact vs iter8 采纳版）。**bench 2139.1us vs baseline 2139.7（iter9 复测）= 0% 打平否决**——实证 iter9 的论断：wall 卡在串行等待**边界数**（mma 分解不减少边界，12/task 不变），flag 条数 -27% 无收益。**关键方法论收获：flag 指令条数不是瓶颈，边界数才是**。
  - **候选 A2（完全手动 flag + 双缓冲流水，mha_sink_fwd_bhsd 先例模式）**：关 AUTO_SYNC/AUTO_CV_SYNC + ws 改名 a_relay/b_relay + C 侧 T.Scope("C") 手写握手（E_L1/E_L0/E_L0C 事件 ID + priming）+ L1/L0C 双槽 + V 侧手动 set_cross_flag("MTE3")。**两个独立死锁**：① slot 化 flag 序列死锁（mini 隔离复现：packed [2,*] 双槽版 + 独立 buffer 分支版 + bisheng on/off 均死锁——与 mha 表面同构但存在未定位的硬件语义细节）；② 大 kernel AICore timeout。
  - **候选 A3（E2 模板忠实手写 = 正确性基线）**：逐条转写 gemm_v0 模板内部序列（prologue 对×2 + priming×2 + 循环 + epilogue 闭环，22 条/调用）+ 手动跨核 WS flag + V 侧靠 bisheng auto-sync（cce_auto_sync 默认 on）兜底。**mini C-only 环境双 dtype 场景 bit 级全过（bisheng on/off 都过）**——证明模板序列自含正确、bisheng 与手写本地 flag 兼容（推翻死锁假说②的本地归因）。**大 kernel 版（e2manual）无死锁（死锁②根因确认为 slot 化序列）但精度腐蚀非确定性**（Aqk 0.868 / Akk 0.535，5 次重复不稳定 → **V 侧数据竞态**：AUTO_SYNC=False 下 tilelang 不再插 V↔MTE2/MTE3 同步，bisheng auto-sync 对生成 C++ 的补漏不足）。**未过精度关否决**。
  - **候选 B（框架模板瘦身）未启动**：路径 A 的探索已证明——header-only 模板瘦身最多省模板内 28 条中的 ~3 条（slot1 管理），pass 外围 16 条需改 .cc（rebuild + 并行会话风险），且 A1 实证 flag 条数削减无 wall 收益——**B 的预期收益同样受边界数瓶颈封顶，无立项价值**。
- **本轮工具链决定性发现（journal 候选，含 3 项正反数据点）**：
  1. **wall 瓶颈是等待边界数而非 flag 指令数**（A1 实证：-27% flag = 0% wall）——任何不引入跨步重叠的编排瘦身（含框架级）收益趋零；**唯一有效方向 = 双缓冲软件流水**（预计 -40~60%），但被下条封锁；
  2. **mha_sink_fwd_bhsd 的手动流水模式不可直接移植**：其正确性实证依赖 bisheng auto-sync 兜底（cce_auto_sync 默认 on）——**tilelang AUTO_SYNC=False 关闭后，V 侧 MTE2↔V/MTE3↔V 同步无人负责**（VS pass 只覆盖 V→V barrier 和 S↔other，ascend_sync_insert_vs.cc:897-904 ShouldSync 对 MTE2/V 组合返回 false；且 VS=True 会强制 --cce-auto-sync=off 使 V 侧彻底裸奔）——**Expert 模式混核 kernel 必须手动补全 V 侧全部 pipe 边界**（约 12 对/task），这是路径 A 的真实工作量所在；
  3. **gemm_v0 模板内部序列自含正确**（E2 手写在 bisheng off 下 bit 级通过）——可作为手动编排的正确性基线模板复用；
  4. **T.Kernel JIT 调用语义**：@tilelang.jit 函数调用返回 JITKernel 而非执行（须 kern = f(...) 后再 kern(args...)）——本轮 mini 系列前 8 个实验全部因此无效（"全错"实为从未执行），**debug 前先验证 kernel 真的跑了**（输出哨兵/statistics）；
  5. **[2,*] 双槽 L1/L0C packed buffer 的 flat-offset 切片在 zN fractal 布局下腐蚀**（iter9 结论强化：独立 buffer + if 分支是唯一合法双槽形态）；
  6. **T.Tensor([40,...]) 4D 整行切法 + cid*2+gg 索引静态可证**，而 it*2+gg 复合 TIR Var 索引过不了 BufferRegion 检查（iter9 教训 8 的补充：取模/已知范围变量才行）；
  7. **cross-core flag 硬语义**：V 侧每 vid set 同一 ID ×2 + C 侧 wait ×1 的形态在 iter8 auto 版实证正确（本轮 e2manual 沿用同形态无死锁）。
- baseline_time: 2139.7 us（iter9 复测值沿用，未重测——本轮无候选进入 bench 门禁；与 iter8 记录 2125.0 噪声带内一致）
- candidate_time: A1 mma 分解 2139.1 us（打平，events mean；candidate json 见 fla_chunk_intra_token_parallel_mmadecompose_iter10.json）；A2/A3 未过正确性关
- improvement: **0%（无采纳）**；累计维持 19150.1→2125.0（-88.9%，9.0×）；距带宽下限 492us 4.3×
- precision: n/a（无候选进入全量门禁；A1 探针两形状 bit exact 但性能打平自动否决；A3 探针 0.868/0.535 竞态腐蚀）
- adopted: **no**（主文件 md5 dafbcb35 未动，= iter8 采纳版）
- rollback_reason: A1 性能 0%（编排瘦身方向被边界数瓶颈证伪）；A2 死锁（slot 化序列）；A3 V 侧竞态腐蚀（bisheng 兜底不足）；B 未立项（同 A1 逻辑封顶）
- 产物：fla_chunk_intra_token_parallel_impl_iter10_before.py（=iter8 版备份）、mmadecompose/e2manual/minic 系列探针 ×12、
  {mmadecompose,e2manual,manualflag,minic*}_iter10*.py/.c/.log、kernel_source_iter10.c（编排量化基线）、
  dump_source_iter10.py、logs_iter10/（16 个实验日志）、fla_chunk_intra_token_parallel_mmadecompose_iter10.json
- next_hint: **唯一剩余的 wall 级优化路径 = C 侧双缓冲流水（-40~60% 潜力），其完整实现清单已探明**：
  1. C 侧：E2 模板序列为基线 + L1/L0C 独立双槽 buffer（if s==0/else 分支，勿用 packed [2,*]）+ slot 化事件 ID 的死锁根因定位（对照 mha 逐指令 diff，重点：M_MTE1 priming 的 per-gg 闭环 vs 跨-task 交替；建议用 T.printf 二分定位卡死的 wait）；
  2. **V 侧手动同步（本轮缺口，工作量 ~12 对 flag/task）**：每 GM→UB copy 前 wait_flag("V","MTE2") + 后 set_flag("MTE2","V")；每 UB→GM copy 前 wait_flag("V","MTE3") + 后 set_flag("MTE3","V")；UB 复用 WAR 对——参照 mha V 侧模式（其正确性在 bisheng 兜底下实证，手动补全后可关 bisheng 或保留双保险）；
  3. 通过后重跑 A1 的 bench 对照（验证"边界数→wall"模型：双缓冲应把 12 边界/task 压到 ~6）。
  风险提示：步骤 1 的死锁 debug 不可控（本轮 4 个变体未定位），建议先用 mini 框架（本轮已建，注意 JIT 二段调用）+ T.printf 逐 wait 打点。

## Iteration 11 — 2026-10-24T05:20:00+08:00（用户指令：iter10 next_hint——C 侧双缓冲流水 mma(g1)∥fixpipe(g0)，预期 10→6us/task；结构落地+量化验证成功，wall 打平否决，收敛于 V 侧工作负载瓶颈）
- bottleneck_type: **V 侧每 task 工作负载（drain+produce ≈ 10.34us/task，其中 beta 折叠 2.61us）**——iter10 "12 串行边界" 模型修正：边界并非全在 C 侧，V 侧与 C 侧在 iter8 流水线下同为 ~10us 的 co-binding 约束；C 侧减半后 V 侧独占瓶颈
- **Step 1 结构落地（独立双槽 E2，全部验证通过）**：
  - **槽布局**：L0C 双独立槽 C_l0c_0/C_l0c_1（偶/奇 head group， planner 实测 offset 0/32KB 不叠加）；**L1 单整块** [NG*AM, DK]（非双槽！）；L0A/L0B 单槽（fp32 A tile [128,128] 恰占满 64KB L0A，双槽容量不可行）；C 侧 E2 序列分解 op（full-slab GM→L1 1 次 + L1 行切片→L0A/B + mma + L0C→GM），**全 pass 生成 flag，零手写本地 flag**
  - **两个决定性结构约束（本轮探明）**：① AUTO_CV_SYNC 要求 cube/vec 侧 workspace 同步点数严格相等（ascend_combinecv.cc:379 FATAL）→ C 侧每 section 只能 1 次 ws_a+1 次 ws_b 拷贝 → 整块 slab 加载 + 偶/奇两条 TIR 循环（`for g in T.serial((NG+1)//2)` 偶组→槽 0 / `NG//2` 奇组→槽 1，静态槽缓冲，NG∈{1,2,4,8} 全覆盖，无 IfThenElse）；② pass 的 event ID 0-7 轮转分配**无视手写 flag ID**（ascend_sync_insert.cc:1175）→ AUTO_SYNC=True 下手写本地 flag 必撞 ID——这正是 iter10 A2/A3 被迫关 AUTO_SYNC 进而 V 侧裸奔的根因链条
  - mini 探针链全绿：closure-Python-int 静态 if 内联 ✓（staticif）；NG=2 静态切片 bit-ratio 1.000000 ✓（minic11）；NG=4 动态 L1 行切片 bit-ratio 1.000000 ✓（minic11d）；kernel_source_iter11.c 编排确认：每 task 1 次 slab MTE2 边界 + mma(g0) 的 FIX_M 只等 fixpipe(t-1,g0)（早边界）+ **mma(g1) 前 FIX_M 完全消失**（FIX 管线 barrier 传递序）+ fixpipe 与下一 mma 并行发射
  - **C 侧隔离计时（minic11 timing，bench shape tile × 20 核 × 200 task）：4.93us/task**（iter8/A1 单槽 ~10us）——**C 侧链路如设计减半**，"边界数→wall" 假设在 C 侧成立
- **wall 打平的量化归因（本轮核心产出，三探针差分）**：
  | 探针 | per-task | 结论 |
  |---|---|---|
  | 全 kernel（candidate） | 10.07us | = 2130.0us/205task，与 baseline 持平 |
  | V-ONLY（C 侧逐组序列剥离，ws slab 拷贝+FIX flag 保留维持点位/死锁平衡） | **10.34us** | **V 侧独占 wall**，C 4.93us 完全被隐藏 |
  | V-ONLY-NOBETA（16×标量 Muls 折叠换 1 次全 tile mul，仅计时） | **7.73us** | **beta 折叠 = 2.61us/task**（64 次 scalar GetValue+小 Muls，~25% wall） |
  | C-ONLY（minic11） | 4.93us | C 侧已达标（目标 6us 以内） |
  - V 侧剩余 ~7.7us ≈ DMA（relay 写出 96KB + 读入 ~73KB/task ≈ 5-6us）+ 向量链（exp×8、cast×8、~40 tile op、~10 V barrier）
- **beta 折叠向量化尝试（compare+select+reduce_sum+broadcast 五 op/head 替 32 op/head，数学 bit 等价）被两堵工具链墙阻断**：
  1. **列抽取无合法路径**：[SCS,16] 取列 h —— strided UB→UB 拷贝 4B 行 → aicore EE9999（32B 块粒度）；binary op 列切片被显式拒绝（ascend_tile.py:1030 "column-offset slices produce wrong results or aicore exception 507015"，实测复现）；broadcast 从 strided region 读**静默丢步长读成连续**（错值不报错，footgun）；N=1 gm→ub 拷贝硬件/helper 层损坏（首元素复制到所有行，copyn1 探针实证；现行 N=4 拷贝合法）
  2. **UB 预算耗尽**：planner（MEMORY_PLANNING=True）LinearScan + **ExtendKillIndex 把 peeled-0 触碰过的全部缓冲的 kill 延伸到主循环尾** → drain+produce 缓冲全量共存 195.4KB/196.352KB，余量仅 960B；折叠需 eq_m 256B + beta_sel 1KB + beta_col 64B + 比较源（j_grid 读入循环 → liveness +1KB）≈ 2.4KB，超支 ~1.4KB（两次布局尝试 akk_h 分配失败实证）
  - **解锁路径已探明未实施**：drain 暂存重构（aqk_s [4,16,64]fp32 16KB + aqk_h [4,16,64]bf16 8KB → per-head 复用 [16,64]，释放 ~18-22KB）代价 ~+8 op/task（~0.35us），净收益估 **−1.2~1.6us/task ≈ wall −9~12%**——超出本轮 scope，留 iter12
- baseline_time: **2131.4 us**（iter8 md5 dafbcb35 今日复测 events mean，loop 2086.6；与 iter8 记录 2125.0 / iter9 复测 2139.7 同噪声带，稳态 rep 2066-2070）
- candidate_time: **2130.0 us**（events mean，loop 2085.5；candidate json 见 fla_chunk_intra_token_parallel_candidate_iter11.json，md5 d24a758e）
- improvement: **+0.07% ≈ 0%（噪声级，无采纳）**；累计维持 19150.1→2131.4（-88.9%，9.0×）；距带宽下限 492us 4.3×
- precision: candidate `--level l0` 3/3 + golden 自检 PASS（ratio 1.0000）；mini 双形状 bit 级 exact；全量门禁未达（性能门禁自动否决，无采纳可能）；回滚后主文件 md5 dafbcb35 = iter8 采纳版，L0 复测 PASS
- adopted: **no**（主文件回滚至 iter8，md5 dafbcb35）
- rollback_reason: **wall 0%——V 侧 co-binding**：C 侧链路 10→4.93us 的实打实改进被 V 侧 10.34us 完全隐藏（max(V,C) 流水周期不变）；>3% 门禁未过
- 产物：impl_iter11_before.py（=iter8 备份）、impl_iter11_candidate.py（C 侧双槽流水完整实现，iter12 直接复用）、kernel_source_iter11.c（双槽编排基线）、minic11/minic11d/betafold/betareload/copyn1/staticif/vonly/vonly_nobeta 探针 ×8 + minic11_source + timing 脚本、candidate/baseline json ×2；中断会话遗留 dump_a1/a1_source_iter11.c 已消化（A1 编排检视目的达成）
- next_hint: **唯一 wall 级方向 = V 侧工作负载削减（本轮已量化到 op 级）**：
  1. **iter12 主攻：drain 暂存重构（释放 18-22KB UB）→ 落地 beta 折叠向量化**（五 op 序列各环节均已单独探针验证：compare 张量-标量 ✓ 256B 对齐规则用 [SCS,16] 满足、select VSEL_TENSOR_SCALAR_MODE ✓、reduce_sum dim=-1 [M,N]→[M,1] mha 先例 ✓、broadcast axis=1 ✓ 本 kernel mask hoist 既有；除 -0.0 角点外 bit 等价，-0.0 过 Akk 容差门禁）；预期 wall −9~12%（~2130→~1870）
  2. V 侧 DMA 地板：relay 96KB 写出/task 为精度强制 fp32（iter7 已否决 bf16），V 地板 ~5-6us/task → 无 relay 重构的 wall 地板 ~1250-1450us
  3. journal 候选（3 项工具链缺陷实证）：N=1 gm→ub 损坏；broadcast-from-strided-region 静默错值；planner ExtendKillIndex 使 peeled+loop 缓冲全量共存（UB 碎片化根因）
  4. 收敛判定：若 iter12 落地后 wall ~1870（-12%），下一档需 relay 结构重构（L2 常驻已做、bf16 被精度封死）→ 边际收益递减，建议 1870 附近收敛（~10.2×，距带宽下限 3.8×）

## Iteration 12 — 2026-10-24T05:41:28+08:00（用户指令：继续压榨性能——iter11 next_hint：drain 暂存重构释放 UB → 落地 beta 折叠五 op 向量化，预期 wall −9~12%；实际 **−18.8%（2141.6→1739.2us）**，超预期达成并采纳）
- bottleneck_type: **V 侧每 task 工作负载（10.34 → ~8.2-8.4us/task）**：beta 折叠 2.61us（64 次标量 GetValue+小 Muls）被五 op 向量化折叠替换为 ~0.4-0.7us；V 仍独占墙（C 侧 4.93us 持续被隐藏）
- **基线选择：iter11 候选（d24a758e）而非 iter8**——其 C 侧双槽流水已单独过 L0/bit-exact/bench（2130.0us 持平），且五 op 折叠脚手架已写入（仅因 UB 预算无法编译）；实测确认该文件状态在 bench 形状即 `akk_h required: 2048, new memory available: 384` 分配失败（ascend_memory_planning.cc:767），与 iter11 记录一致
- **Step 1 drain 暂存重构（UB 预算表，bench 形状 SCS=16/DK=128/BH_V=4/CS=64，单位 B）**：
  | 项目 | iter11 候选（编译失败态） | iter12 |
  |---|---|---|
  | produce 链+累加器+mask+r_ub（含五 op scratch 2,368：eq_m 256+beta_sel 1024+beta_col 64+j16b 1024） | 166,272 | 166,336（+idx16 64） |
  | drain 暂存 | aqk_s 16,384 + akk_s 4,096 + aqk_h 8,192 + akk_h 2,048 = **30,720** | 4,096+1,024+2,048+512 = **7,680**（[BH_V,SCS,*] → per-head [SCS,*] 复用） |
  | ExtendKillIndex 全量共存峰值 | **196,992 > 196,352 预算 → akk_h 分配失败** | **174,016 ≤ 196,352（余量 22,336）** ✓ |
  - 重构语义：fill/select/cast/GM 写回全部移入 h 循环内对单头 [SCS,*] 缓冲执行（同元素级 op 作用于同值，cleared-row 零由 per-head fill 重建）→ bit 等价；代价 ~+12 个小 tile op/task（per-head fill/cast 替代整任务一次）
  - 附带优化：j16b 的 [SCS,16] 广播提升到 mask hoist 区（每 task 省 3 个小广播；其 planner interval 不变——ExtendKillIndex 本就从 peeled-0 延伸到循环尾）
- **Step 2 beta 折叠五 op 向量化落地**（iter11 探明的 one-hot 路径：compare(eq_m, j16b, h) → select(beta_sel, eq_m, beta_f, 0) → reduce_sum(→beta_col, dim=-1) → broadcast(bc_row, axis=1) → mul(e, e, bc_row)）：kernel_source_iter12.c 实证 lowering——**GetValue 0（标量折叠彻底消失）**、CompareScalar×3 site / Select VSEL_TENSOR_SCALAR_MODE×3 / reduce_2d kSum [16,16]→[16,1]×3 / Broadcast [16,1]→[16,128]×3 / 全 tile Mul×3
- **Step 2 附带修复（iter11 遗留隐性缺陷，全量门禁抓获）**：原五 op 代码 `broadcast(j16b, idx, axis=0)` 中 j16b 为固定宽 16 列而 idx 长 SCS——仅在 SCS=16 巧合成立，SCS=8 编译期 DiagnosticError（Broadcast dimension mismatch: src[1]=8 != dst[1]=16）；iter11 候选从未编译成功故未暴露。修复：新增 idx16 [16]（arith_progression 0..15）作广播源；SCS=8 编译+精度全过（[8,16] eq_m 128B < 256B 对齐规则的担忧未成真）
- baseline_time: **2141.6 us**（iter8 md5 dafbcb35 本轮同会话复测 events mean，loop 2091.6；与 iter8 记录 2125.0 / iter11 复测 2131.4 同噪声带）
- candidate_time: **1739.2 us**（events mean，loop 1688.7，稳态 rep 1661-1676；candidate json 见 fla_chunk_intra_token_parallel_candidate_iter12.json，md5 a618cc21）
- improvement: **−18.79%（2141.6→1739.2，超 3% 门禁采纳；亦超 iter11 预估 −9~12%）**；累计 19150.1→1739.2（**−90.9%，11.0×**）；距带宽下限 492us 3.53×；358 GB/s = 实测 D2D 上限 28.3%
- precision: `--level all` **exit 0 + Test Passed**：L0 3/3 + golden 自检、L1 9/9（含 H=2/6/12 泛化、DK=48、scs8）、L2 6/6、Boundary 4/4（zero/inf_q/nan_beta/dbound 深衰减）全 ratio=1.0000（首轮门禁 l1_scs8 编译失败即上述 idx16 缺陷，修复后二轮全过）
- adopted: **yes**（主文件 = iter12 候选，md5 a618cc211fa5782ba4a56b6c6a074a25；iter8 回滚点保留于 impl_iter11_before.py / impl_iter8_candidate.py dafbcb35）
- rollback_reason: N/A（无需回滚）
- **V 侧 wall 分解变化（per task，207 个流水槽/核）**：10.34（beta 2.61 标量 + DMA ~5-6 + 向量链 ~1.7-2.1）→ **~8.16（loop 1688.7/207）~8.40（events 1739.2/207）** = beta ~0.4-0.7（五 op）+ DMA ~5-6 + 向量链 ~1.7-2.1 + drain 重构开销 ~+0.3 − j16b hoist ~−0.05；净削 **−1.94~2.18us/task**
- 产物：impl_iter12_before.py（=iter11 候选 d24a758e 备份）、impl_iter12_candidate.py（采纳版 a618cc21）、kernel_source_iter12.c（采纳版 lowering 证据）、compilecheck_iter12.py / dumpsource_iter12.py 辅助脚本、baseline/candidate json ×2
- next_hint: **收敛判定（依 iter11 预案，达成并超出 ~1870 目标）**：
  1. V 侧 ~8.2-8.4 仍独占墙（C 4.93 隐藏），剩余构成 = DMA 地板 ~5-6us（relay 96KB fp32 写出 + ~73KB 读入；bf16 被 iter7 精度封死、L2 常驻已做）+ 语义强制向量链 ~1.7-2.1us（exp×8/cast×8 为算法本体）+ 五 op ~0.4-0.7
  2. 无 relay 结构重构的下一档理论空间仅剩向量链削减（~−0.5-1us，−6~12%，需 fuse cast+mul / 减少 barrier 等高投入低回报改造）；再往下必须 relay 结构重构（无合法压缩路径，iter11 已封死）→ 边际收益递减
  3. **建议收敛于 1739.2us（11.0×，距带宽下限 3.53×）**；若继续，唯一 wall 级方向是 relay 结构重构（改变 fp32 relay 语义/布局的 research 级改造），预期收益 ~1250-1450us 地板区
  4. journal 候选补充：iter11 遗留五 op 代码的 broadcast 源宽度隐性缺陷（[SCS,16] dst 配 SCS 长源，仅 SCS=16 巧合成立）——从未编译的候选代码必须过全形状门禁后才能作为后续基线

## Iteration 13 — 2026-10-24T17:30:00+08:00（用户指令：relay 结构重构 research 级三路径评估；1/2 不可行 + 3 实施后遭同步 pass 结构耦合阻断，全证据链判定 relay 重构在本工具链不可行，正式收敛 1739.2us）
- bottleneck_type: **V 侧 DMA 暴露（relay 重构研究，iter12 next_hint 唯一 wall 级方向，预期地板 1250-1450us）**：msprof 稳态 20 次（prof_iter13/msprof_summary_iter13.json，中断会话遗留、本轮消化）wall 1660.7us = 8.02us/task；AIV pipe busy = vec 940.2us（4.54/task）+ scalar 652.7（3.15）+ mte2 578.2（**2.79**）+ mte3 292.2（1.41），pipe 合计 11.9us/task vs wall 8.02 → 平均并发仅 1.48，暴露 DMA ≈ 2.5-3.5us/task；AIC busy 99.5%（含 flag 自旋，C 临界 4.93 隐藏）
- **三路径评估表**：
  | 路径 | 可行性 | 预期收益 | 证据 |
  |---|---|---|---|
  | 1 消除中转（AIC/AIV 共享 UB/L1） | **不可行（硬件）** | —（若可行 −2~3us/task） | 910B3 MIX_AIC_1_2 为**分离模式**：CANN CrossCoreSetFlag 文档明示 AIC/AIV 是 AI Core 内独立子核、私有 Local Memory，通信仅 GM；tilelang WorkspaceReduction pass `needs_gm_workspace_ = (platform != "A5")`（ascend_workspace_reduction.cc:137）——UB↔L1/L0C 虚拟通道在 A2/A3 一律经 GM 分解；`copy_ub_to_l1` C++ helper 仅支持 half（common.h:415，fp32 操作数被 iter7 精度门封死）；CombineCV 同步只识别 GM copy + "workspace" 命名（FetchWorkspaceName）——同步机制本身与 GM 耦合 |
  | 2 压缩 relay 字节（ggrp=1 重估 + A 行布局） | **不可行（结构 + 实测劣化）** | 0（仅 C 侧变化） | ggrp 只影响 C 侧（produce 96KB/task 字节与 ggrp 无关；drain 每头读 [32,16] 对角块与 ggrp 无关）；C 4.93us 隐藏于 V 8.02 下，ggrp=1 只增 C 侧序列数（2→8 组）→ 探针实测 **1892.6us（events）/1856.3（loop），较基线 +153.4us = +8.8% 回归**（ggrp1_probe_iter13，sanity allclose 通过），旧编排 +168us 劣化在新编排下**未翻转**；A 行布局：A/B 操作数每行都参与有效对角块计算（浪费仅在 C 输出 87.5%，为 C 侧 fixpipe 字节、已被隐藏）→ V 侧确无字节可省 |
  | 3 DMA 重叠（produce/drain load-compute 流水） | **量化成立、实施被 pass 阻断** | 理论 −1.5~2.2us/task（1200-1350us），实测不可达 | 暴露机制实锤：iter12 lowering 中每 h 迭代头部 `SetFlag<V_MTE2>/WaitFlag<V_MTE2>+PipeBarrier<PIPE_MTE2>`（单槽 staging 的 WAR）→ head h+1 的 MTE2 loads 被迫等 head h 向量链排空，produce loads（~2.2us/task）+ drain o_relay 读完全暴露；实施见下 |
- **路径 3 实施（未采纳）**：A/B 两组独立命名 staging（q/k/g2_row_a/_b + r_ub_a/_b，UB +18,432B ≤ iter12 余量 22,336B）+ BH_V∈{4,2,1} Python 静态分发，调度 = 奇槽 loads 发于偶链前、下偶槽 loads 发于奇链前（loadB(1)→chainA(0)→loadA(2)→chainB(1)→…），L0/编译全过、dump 逐 op 验证调度/头偏移/staging 行区间/beta fold lowering 全部正确、C 侧与 iter12 **逐字节一致**
- **失败证据链（为何 NO-GO）**：(2,256,16,128) 上 **Aqk 逐位精确、Akk 确定性损坏 19,180 cell（≈14.6%）且 100% 集中于 t=1（peeled-1）中转的任务**（t=0/2/3 站点全过），3 次重复计数不变（非竞态、纯确定性）；二分矩阵：V1（仅 produce 流水化）败于 t=1、V2（仅 drain 流水化）同样败、V5p（peeled-1 串行化重排）仍败且 Aqk 亦坏、V9p（loop 串行化）扩散至全站点、BH_V=2（H=12）同样败、NBLK≤CORE_EFF 小形状过（t=1 无真实任务）→ **结论：AUTO_SYNC/AUTO_CV_SYNC 的同步生成与 `for h in T.serial` 循环结构强耦合（循环承载依赖经 2-迭代合成展开分析、flag FIFO 配对按 buffer 粒度——dump 证实同 buffer 双槽仍插假 WAR），任何直线化/交错重排均确定性破坏 NSLOT=2 跨核 WAR 传递链，且损坏无法在 DSL 层修复**；候选快照 impl_iter13_candidate.py（md5 16862eb0）
- baseline_time: **1739.6 us**（回滚后本轮复测 events mean，loop 1687.6；与 iter12 记录 1739.2/1688.7 同噪声带；baseline_iter13.json）
- candidate_time: **N/A（未过精度门禁，不进入正式 bench）**——路径 2 探针 1892.6us（劣化）、路径 3 候选精度失败
- improvement: **0%（无采纳）**；维持 1739.2us（累计 19150.1→1739.2 = **−90.9%，11.0×**；距带宽下限 492us 3.53×；357 GB/s = 实测 D2D 上限 28.3%）
- precision: 回滚态全量门禁 `--level all` **exit 0 + Test Passed**：27 PASS / 0 FAIL（13 PRECISION + 14 BOUNDARY，logs_iter13/fla_chunk_intra_token_parallel_precision_all.log），与 iter12 记录一致
- adopted: **no（主文件回滚至 iter12 采纳版，md5 复核 a618cc211fa5782ba4a56b6c6a074a25）**
- rollback_reason: **路径 3 候选确定性 Akk 损坏（t=1 槽位 100% 任务，Aqk 位精确的 K''-半区选择性损坏）**——同步 pass 与 h-loop 结构耦合，4 种调度变体全部失败，DSL 层无修复路径
- next_hint: **正式收敛于 1739.2us（11.0×）**——relay 重构三路径证据链闭合：
  1. 剩余理论空间需 pass 级/工具链级配合：① 同步 pass 支持 region 粒度依赖（消除同 buffer 双槽假 WAR）；② 跨核 WAR 传递链对直线化代码的正确性（t=1 损坏根因未定位，需最小复现立项）；③ A5 融合核平台（UB↔L0C 直通，`platform != "A5"` 分支）可消除中转——均超出算子层 scope
  2. 若必须继续：T.serial 配对循环形态（`for hp in T.serial(BH_V//2)` 内含双链+符号头索引）保留循环承载依赖分析，是唯一未试的 DSL 形态——但 4 变体皆败后预期同样受累，不建议
  3. journal 候选：① WorkspaceReduction A2/A3 GM-bounce 文档化（relay 地板的架构根源）；② AUTO_SYNC buffer 粒度 WAR/RAW 假边（同 buffer 不相交 region 仍串行，dump 证据）；③ 直线化重排后跨核协议确定性损坏（t=1 隔离、Aqk/Akk 选择性、二分矩阵）；④ ggrp=1 在 E2 编排下仍 +8.8%（调用开销假设更新：序列编排开销 > fixpipe 字节节省）
- 产物：impl_iter13_before.py（=iter12 采纳版备份）/ impl_iter13_candidate.py（失败候选 16862eb0）、ggrp1_probe_iter13.{py,json} + bench 脚本、v1/v2/v5p/v9p 二分探针 + bisect{,2,3,4}_iter13 日志、kernel_source_iter13.c / v1_source_iter13.c / iter12ref_src_2256.c（对照 lowering 证据）、baseline_iter13.json、logs_iter13/（precision_all + bench + 探针日志 ×6）、prof_iter13/（中断会话 msprof 数据，本轮消化）
