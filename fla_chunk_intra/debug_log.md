# debug_log — fla_chunk_intra（Stage 2 实现记录）

## Attempt 1 — 2026-11-02
- mode: first_impl
- classification: precision_fail（fwd 全过；bwd 数值未收敛——工具链深层交互，证据链见下）
- fail_category: other（bwd：跨核同步/工具链交互；非设计错误）
- test_level: l0（fwd 4 用例全过；bwd 未过）
- coverage: L0:7（fwd 4 PASS / bwd 3 FAIL）；L1/L2/Boundary 未扩展（L0 bwd 未收敛前不扩展）
- boundary_warnings: none
- changes:
  - `fla_chunk_intra.py`：6 JIT kernel（fwd 5 + bwd 1，Developer 模式 threads=1）+ 2 host wrapper
  - `test_fla_chunk_intra.py`：PyTorch fp64 golden（fwd solve_triangular 版 + fwd block-recursion 交叉校验 + bwd 统一掩码版 + bwd 上游逐列循环直译交叉校验，四路 golden 全部互拍一致）+ check_precision（precision-standard §4.1 原样）+ L0 七用例 + do_bench（NPU 事件，warmup=20 rep=10 mean）
- error_summary: 见下方「踩坑记录」
- design_error_reason: none（DESIGN 的算法结构、代数恒等式、API 映射均正确；发现的 4 处 §9.1 golden 转写笔误属文档转写错误，已按 Triton 语义权威修正，非设计层错误）
- rollback: no
- backup_path: history_version/fla_chunk_intra_impl_s2_attempt1.py
- instrumentation_cleaned: yes（所有调试探针均在 /tmp，主文件无插桩）
- next_hint: bwd 按「BWD v3 重构方案」执行（见下）

### 实测结果（首次可信实测）

| 项 | 值 | 口径 |
|---|---|---|
| perf_fwd_bench (1,8192,64,128) | **16.441 ms**（5-kernel 链合计） | do_bench：NPU 事件，warmup=20 rep=10 mean，仅设备侧 kernel 时间（对齐 ref_gpu/test_utils_kda.py） |
| perf_bwd_bench (1,8192,8,128) | **1.997 ms**（单 kernel；⚠️ 数值未收敛，仅结构口径，不作为验收数据） | 同上 |

fwd 16.44ms vs DESIGN §9.5 现实预期 2-4ms：超 4-8×，推测前代求逆与 k_inter 的 V 侧串行 tile op 数主导（DESIGN 对 vector op 吞吐估计偏乐观）——Stage 3 调优对象。ws 中转 624MiB + IO 结构与设计一致。

### 精度结果（L0）

| 用例 | 结果 | Aqk | Akk |
|---|---|---|---|
| l0_fwd_min (1,64,2,128) | **PASS** | ratio=1.000000 max_abs=7.63e-06 | ratio=1.000000 max_abs=4.88e-04 |
| l0_fwd_repr (2,256,4,128) | **PASS** | ratio=1.000000 max_abs=1.95e-03 | ratio=1.000000 max_abs=3.91e-03 |
| l0_fwd_dk64 (1,128,4,64) | **PASS** | ratio=1.000000 max_abs=1.95e-03 | ratio=1.000000 max_abs=1.95e-03 |
| l0_fwd_bench_repr (1,1024,64,128) | **PASS** | ratio=1.000000 max_abs=3.13e-02 | ratio=1.000000 max_abs=6.25e-02 |
| l0_bwd_min / repr / dk64 | **FAIL** | 输出整体错误（dq2 ratio≈0.0001） | — |

### 踩坑记录（按发现顺序，全部实证）

**G0. DESIGN §9.1 golden 转写笔误（4 处，Stage 1 文档转写错误，非设计错误）**
1. M 组装行：`akk * bf[...].unsqueeze(-1)` 维度错且多了 permute——正确为 `akk * bf[...].unsqueeze(2)`（无 permute）。
2. `Akk_out[:, s0:s0+CS, :, :] = Ai.permute(0, 2, 3, 1)` → 正确 `permute(0, 2, 1, 3)`。
3. bwd (A)+(B) einsum `"bjh,bjhk->bhk"` → 正确 `"bhj,bjhk->bhk"`（`dA_c[:, i]` 是 [B,H,j]）。
4. bwd (D)+(E) 的 `m_up` 需 `unsqueeze(-1)` 对齐 j 维。
交叉校验锁定：fwd golden 与「逐块前代 + 6 块递推」直译版 0 误差互拍；bwd golden 与上游逐列循环直译版 fp64 一致（~1e-14）。

**G1. `@T.prim_func` 是 TVM AST 解析器，不是 Python trace（segfault 级）**
kernel 体内禁止：Python 列表/列表推导/元组循环/调用 Python 辅助函数——解析器直接段错误（无报错）。s/s2 静态展开必须**字面展开**，hi/lo split 链必须**内联**。

**G2. 跨核 workspace relay 按 buffer 名索引（ascend_workspace_reduction.cc `src_to_workspace_map_`/`dst_to_workspace_map_`）**
**每个 UB→L1（V→C）拷贝必须独占专属 UB 源 buffer；每个 L0C→UB（C→V）拷贝必须独占专属 UB 目的 buffer**。复用一个 temp 会导致多个 relay 塌缩到同一 GM workspace → CombineCV 报 `Mismatch in sync points ... cube has 1, vec has N`（实证：p10 探针 shared 失败 / unique 通过）。纯 V 的 UB↔UB temp 可自由复用。这是 ★fla_wy_fast 每操作数独立 temp 的真实原因。

**G3. `T.tile.add/sub` 等 tile op 的 dst region 展平连续性约束**
非整行/非整块的块对角 region（如 `aid_st[0:16, 0:16]` of [64,64]）做 tile.add 会前端报错。修复：+I 移到 [16,16] 整块域（新增 Meye 常量 host 缓存），放置用 region COPY（合法）。

**G4. ⭐最重要：UB 2D region 作为 UB→GM 拷贝的源（region 列宽 < buffer 列宽）时数据错位**
`T.copy(aid_st[0:16, 0:16], Aidws[...])` 生成 `copy_ub_to_gm<float, 16, 16>(..., aid_st[0], 128, 16, 16)` —— codegen 把 region 列数当作源行距 → 源按**打包连续**读取（行距 16），而实际行距 64 → 行 r 内容落到输出行 4r（实证：Aidws 行 1/2/3 内容出现在行 4/8/12，diff 恰为恒等差）。修复：**所有 UB→GM 窄 region 读写一律走「打包整块 buffer 中转」**（region 先 copy 到 [16,R] 整块 UB buffer，再整块写 GM；GM→UB 窄 region 写同理）。fwd 全链修复后 L0 4 用例全过。fwd 内部的 UB↔UB region 拷贝无此问题（codegen 传 buffer 行距 ✓）。

**G5. fwd 链路验证方法论**
分段对拍（Akkbp → AiDiag → G → G² → Y → Ai 逐 stage 与 fp64 golden 对拍）将 fwd 失效定位到 G4 的三个窄 region 写。bwd 侧同样先做 stage 对拍。

**G6. bwd 现状（未收敛，证据链完整）**
- 分块 staging（Kdec/Qdec/KBdec、MdA、锚点）**逐项数值验证正确**（p22 探针，误差 ~e-6 级）。
- gemm+relay 机制在**隔离探针中全组合正确**（M16/K32/K64/K128 × 有无 region × Pipelined(2) 循环内 × grid=2 全 PASS，误差 e-6 级）。
- 但完整 bwd kernel（或裁剪到 s=0-only 的版本）输出整体错误（dq2 ratio≈0.0001，无错位模式、非半区交换）。
- **已发现确定的破坏向量**：CANN 9.2.0 dav_c220 `GetffstMsg` 中 `flagId & 0xf`——**跨核 flag ID 仅 16 槽且 11-14 为系统保留（SYNC_AIC=11/AIV=12/AIC_AIV=13/ALL=14）**；完整 bwd 生成 **40 个** flag ID（28 UB→L1 + 12 C→V），16-39 全部截断碰撞 → 循环结构下（V 迭代 1 写 vs C 迭代 0 读真并发）必然数据竞态。fwd 24 个 flag 也截断，但 fwd 是直线 phase 分离结构（V 相全部写完 C 相才开始读），靠 MTE3 流速差幸存（实证通过）——该脆弱性已记录。
- 简单探针 14 个 flag 仍正确 → flag 数量非唯一因素；真实 bwd 的深交互（staging 计算链 + 大量 relay + 尾段）中还有未分离的第二因子（候选：prologue 携带 buffer 的 relay 版本化、多 relay 的 flag 配对时序）。
- 附带发现：bwd `num_stages=2` 触发 UB 规划溢出（软件流水 stage 拆分拉长 buffer 区间，LinearScan 超 192KB）→ 已改 `num_stages=1`（编译通过；12 个 fp32 staging 收敛为 1 个共享 + mda/mdk 就地乘后预算可容纳）。

### BWD v3 重构方案（下一 attempt 的执行蓝图）

目标：把跨核 exchange 数压到 ≤ ~11 个 flag（ID 0-10），消除截断/保留位碰撞：
1. **操作数 slab 打包**：每 s 的 3 个 B 操作数（Kdec/Qdec/KBdec）叠成 [3·64, BK] fp16 slab（hi/lo 各一）→ 每 s 2 个 UB→L1 exchange（替代 6）；4 s 共 8 个。
2. **MdA/MdK 四段纵向叠放**：mda_hi|mda_lo|mdk_hi|mdk_lo 叠成 [256, 64] fp16 → 1 个 exchange；gemm A 操作数用行偏移 region（hi: [s·16:+16, 0:KMAX]，lo: [64+s·16:...]）。
3. **尾段去 relay**：C 侧直接 `T.copy(frag, fragscr_gm[...])`（★k1 已验证的 L0C→GM fp32 直写，零 flag）+ **1 对手动跨核 flag**（`T.set_cross_flag("FIX", 9 + i_kk%2)` / `T.wait_cross_flag(..., "V")`，★token_parallel 同款，ID 从 9 起避开自动分配的 0-8）+ V 侧从 GM 读回 12 个 frag 结果做尾段。
4. 总 flag：8 + 1 + 2 = 11（ID 0-10）✓。
5. 若仍失败：按输出拆 2 kernel（DESIGN R1/R6 预授权回退：kernel-1 = staging+gemm+frag→GM（~5 flag），kernel-2 = 纯 V 尾段（0 flag），重读 K/G/Q/dq/dk/dg ≈ +200MB）。
6. 保留：num_stages=1、共享 odec_f staging、就地 mda/mdk、字面展开。

### 其他环境/纪律记录
- 主仓未 commit；ref_gpu/ 未修改；探针全部在 /tmp/opencode/probe。
- do_bench 用 `torch.npu.Event(enable_timing=True)`；wrapper 无内部 sync（★口径教训）。
- bench 内存：fwd 峰值 ~1.4GB（含 5×128MB ws + IO），910B3 64GB 无压力。

## Attempt 2 — 2026-11-03
- mode: precision_fix（执行上轮「BWD v3 重构方案」；方案在实施中被两个深层发现推翻重构为 v3-2k）
- classification: precision_pass（fwd 4/4 + bwd 3/3 L0 全绿；L1 9/9；L2 12 拒绝全对；Boundary 5 case 过 + 2 WARN（非阻塞）；覆盖门禁 20 PASS/0 MISS）
- fail_category: none
- test_level: all（L0+L1+L2+Boundary）
- coverage: L0:7（fwd 4 + bwd 3，全 PASS）；L1:9（3 fwd + 2 bwd + 4 valrange，全 PASS）；L2:12（全部正确拒绝）；Boundary:7（5 PASS + 2 WARN）
- boundary_warnings: bnd_bwd_decay dk2/dg2（ratio=0.5, max_abs=nan）——极端衰减 g=−0.38/token 使 Kdec 操作数指数达 +18.24，fp16 hi/lo 溢出为 inf，与掩码零 MdA 相乘得 NaN；golden（fp64 直接式）为有限值 → 结构比对不匹配 → WARN（DESIGN §10.3 R9 已知 fp16 域限制，非阻塞）
- changes:
  - `fla_chunk_intra.py`：bwd 从 1-kernel 重构为 **2-kernel v3-2k**：`fla_chunk_intra_bwd_k_stage`（纯 V，直线，零跨核 flag：MdA/MdK 行/列 tile + Kdec/Qdec/KBdec 全部 hi/lo 分裂后写 GM scratch OpscrM[GRID,1280,64]/Opscr[GRID,2,1376,BK]）+ `fla_chunk_intra_bwd_k_gemm`（Pipelined(2) 体 = 单 C 段 + 单 V 段：C 侧 56 个 per-tile GM→L1 Nd2Nz 读 + 48 gemm_v0 + 12 个 L0C→GM 直写 Fragscr + set FIX 9+i_kk%2；V 侧 wait FIX + 读回 frag + 修正尾段）；跨核 ffts flag 总数 = 2（ID 9/10）
  - **修正 attempt-1 遗留数学 bug**：尾段 D_s/V_s 后置缩放未应用（dk2pre 链的 db/dk2b/dg2 缺 ×D_s；dkt 缺 ×V_s——attempt-1 算了 v_s 但从未使用）
  - db2 scatter：H≥2 保留 ★fla_wy_fast 链（db2d 加宽为 [BS,8] 防越界踩踏）；H=1 走平坦 [B,S*H] 视图单行 256B 直写（db2 改 host 分配）
  - `test_fla_chunk_intra.py`：L1/L2/Boundary 场景 B 扩展（L1_CASES 9 条 + valrange 变换、L2_CASES 12 条拒绝、BOUNDARY_CASES 7 条；COVERAGE_CATEGORY=Fusion）
  - `proto_fla_chunk_intra.yaml`：仅修 YAML 语法（preamble 散文行加 # 注释，内容零改动）——原文件不可解析导致 checker 降级文本扫描、把 bfloat16 误判为 float16
  - `perf_tuning/bench_attempt2.py`：两方向分 kernel bench 脚本
  - `history_version/fla_chunk_intra_impl_s2_attempt2_bwd_v3.py`：收敛版快照
- error_summary: 见下「踩坑记录 G7-G12」
- design_error_reason: none（v3 slab 方案被工具链 zN 布局约束推翻属实现层发现，回退路径在 DESIGN R1/R6 预授权内）
- rollback: no
- backup_path: history_version/fla_chunk_intra_impl_s2_attempt1.py（attempt-1 原状）；history_version/fla_chunk_intra_impl_s2_attempt2_bwd_v3.py（本轮收敛版）
- instrumentation_cleaned: yes（全部探针在 /tmp/opencode/probe，主文件无插桩）
- next_hint: Stage 3 性能调优（bwd v3-2k 的 GM 中转流量 ~4×于 DESIGN 估计——tile 合并/超级 gemm 化是首选；另见「遗留」）

### 实测结果（do_bench：NPU 事件，warmup=20 rep=10 mean，仅设备侧；本轮数值已收敛，为有效验收数据）

| kernel | shape | 耗时 |
|---|---|---|
| fwd k_inter | (1,8192,64,128) bf16 | 5150.8 us |
| fwd k_g | 同上 | 5673.4 us |
| fwd k_g2 | 同上 | 1584.5 us |
| fwd k_y | 同上 | 2405.6 us |
| fwd k_ai | 同上 | 1849.6 us |
| **fwd 合计**（5 kernel 链，分项和 16663.8 / 链式 16462.2） | | **~16.46 ms** |
| bwd k_stage | (1,8192,8,128) fp32 | 1214.2 us |
| bwd k_gemm | 同上 | 2107.7 us |
| **bwd 合计**（2 kernel 链，分项和 3321.9 / 链式 3312.1） | | **~3.31 ms** |
| **fwd + bwd 总计** | | **19774.3 us** |

- fwd 与 attempt-1 实测一致（16.44 → 16.46 ms，kernel 未动）。
- bwd 3.31 ms：含 staging GM 中转 ~1.2GB（k_stage 写 + k_gemm 读），是 DESIGN 单 kernel 估计的主要超支来源（见「遗留」）；attempt-1 的 1.997 ms 系错误数值上的结构口径，不作数。
- proto 540 us 目标仍物理不可达（DESIGN §9.5 [DESIGN_ERROR] 论证维持；工具链强制拆分使其更远）。

### 踩坑记录（Attempt 2 新增，全部实证）

**G7. ⭐zN tile 形状约束（推翻 v3 slab 方案的根因）**：隐式 relay 写入 L1 的是 **zN 分形布局、形状由拷贝决定**（整 buffer 拷贝 = buffer 形状；Nd2Nz），而 gemm_v0 读的是 **zN tile、形状由 gemm 模板决定**、基址为 region 的 row-major 扁平偏移。二者列带 stride（roundUp16(行数)·16）只在「拷贝 tile == gemm 操作数 tile」时一致；对 whole-relayed buffer 做行偏移/窄列 region 读读到错位 zN → 纯垃圾（v3_probe 实证：A 操作数错 2.8e1，B 操作数恰 N=16 宽例外正确）。**attempt-1 的全部 region 操作数本来就错**（即使 flag 完美也错）。修复：每个 gemm 操作数 = 独立整块 L1 buffer + 精确同形 GM→L1 Nd2Nz 读。

**G8. ⭐跨核 ffts flag 16 槽硬截断（G6 的机制确认 + 预算解）**：CANN 9.2.0 `GetffstMsg` 硬编码 `flagId & 0xf`，仅 16 槽且 11-14 系统保留（SYNC_AIC/AIV/AIC_AIV/ALL）→ 用户可用 ID 0-10。实证：flagid_probe2（5 relay 站点 → ID 0-4 全对数值 PASS）vs 完整 bwd 40 站点（ID 16-39 全部截断碰撞）。**普适约束：隐式 relay 每站点一个 ID，多 relay 算子设计期必须核算 ≤11。**

**G9. V→C 手动握手不可表达于混 C/V Pipelined 体**：`set_cross_flag("MTE3", id)`（V 侧 GM 写后发射的正确管道）分类为 kExplicit 需 T.Scope("V") 包装，但 Pipelined 体内的 T.Scope 块被 CrossCorePipeline 的 LoopAnalyzer 按 StmtInfo 展平丢弃（AttrStmt 丢失）→ PipelinePlanning "Can't handle the body" 或 CombineCV verifier "must be inside T.Scope"（bisect 探针实证）。**解：拆 kernel，kernel 边界即 V→C 同步**；C→V 方向 set("FIX")+wait(...,"V") 两边可分类，实测有效（v3_2k_probe：数值 1e-6，flag 恰 1 对）。

**G10. T.copy 参数序是 (src, dst)**：v3_probe 曾把读回写成 T.copy(f_ub, Fr[...])（实际反向写 Fragscr），"Fr vs Out 一致"的校验因两边都是 f_ub 拷贝而假通过——双向拷贝场景必须用内容校验（golden 对拍）而非自洽校验。

**G11. ⭐窄行 GM/UB 拷贝的 32B 槽约定（H=1 db2 散布 bug）**：[64,1]-region 的 GM→UB 读把 4B burst 散布到 32B 间隔的 UB 槽（MTE「窄行坐 32B 槽」约定，issue-#1682 类），配套的 [64,1] UB→GM 写从同样 32B 槽读回——★链自补偿**仅当最终 GM 行距 >4B（H≥2）**；H=1（连续片）写侧变 flat 单 burst → 错配 → db2 全为元素 0 附近垃圾；且散布写会越界踩踏相邻 UB（db2d 加宽 [BS,8]=2KB 后自包含）。修复：H=1 走 [B,S*H] 平坦视图单行 256B 直写（db2 host 分配 + 零拷贝视图）；H≥2 保留 ★链。L1 的 (1,64,1,64)/(2,64,1,64)/(1,64,1,128) 由此修复。

**G12. Pipeline planner 多写约束**：Pipelined 体内同一 buffer 的 in-place sub（copy 写 + sub 再写同 buffer）触发 "Multiple writes to overlapping buffer regions"（stage 0/3 冲突）——分裂链必须分离目的（lo 单独 buffer）。attempt-1 的代码模式本就如此，本轮探针确认了边界。

### BWD 收敛关键步骤（时间序）
1. v3_probe（slab 打包 + region gemm + FIX flag）：FAIL 2.8e1 → 定位 G7（zN 列带 stride 失配：A 读错、N=16 的 B 恰好对）
2. gemm_region/shape_probe 复核：region-dst 拷贝+同形 gemm 读 PASS、全形状（含 N=16 B / [64,16] transposed-A）经隐式 relay PASS → 确认合法用法是「整块 L1 buffer + 同形填充」
3. v3pp/v3-2k 探针：单 kernel 显式 GM staging + T.Scope("V") MTE3 flag → 撞 G9；改 2-kernel → PASS（1e-6，flag 恰 9/10）
4. 重构落盘 → l0_bwd_min 首跑即 PASS（ratio=1.000000）→ L0 3/3
5. 全量首跑（all_run3）：L1 (1,64,1,64) db_post FAIL（ratio 0.0156, max 419）→ G11（H=1 散布链）→ db2f 平坦视图修复 → 全绿（all_run5 EXIT_CODE=0）

### 遗留问题（Stage 3 输入）
- **bwd GM 流量**：v3-2k 的 staging 中转（OpscrM 160MB + Opscr 344MB + Fragscr 96MB @bench GRID=1024，读+写 ~1.2GB）远超 DESIGN §5.5 估计的 ~1MB relay——bwd 实测会显著慢于单 kernel 设计；tile 合并（block-diag + 3-in-1 超级 gemm，本轮已推导出可行布局但 L1 预算 448KB 需精算）是首选优化
- **bnd_bwd_decay WARN**：fp16 操作数在 chunk 总衰减 >~13 (log2) 时溢出为 inf、与掩码零相乘得 NaN（DESIGN R9 已知限制）；若上游真实域需要更大衰减，需 operand 域缩放或 fp32 中转
- **fwd 24 flag 仍未瘦身**（直线 phase 分离幸存，G6 已记录脆弱性）；Stage 3 可用同款「整块 L1 + 同形读」思路消除

## Stage 3（R1 性能调优，2026-11-03）——工具链发现补遗

**G13. ⭐0×NaN=NaN 穿越 gemm 操作数精确零**：ws（或任何 GM buffer）未写区域的垃圾若为 NaN 位模式，
gemm 另一侧操作数的**精确 0**（块对角/掩码零）在 Cube MAC 上不能屏蔽它（0×NaN=NaN），NaN 沿 K 维
和下游链全量毒化。表现为**间歇性**精度失败（取决于分配历史——torch.empty 复用内存的位模式），
L0 可连续复现也可连续干净。修复：生产者 kernel 显式零写全部未用区域（本仓：k_inter 零写
Akkbp rows 0-16），ws 全域确定性写出。设计期规则：**ws 契约必须覆盖每个字节**。

**G14. 同一 L1 buffer 兼作 gemm A 与 B 操作数读出垃圾**：zN 布局由消费侧模板决定，一个 L1 buffer
只承载一种形态；A、B 两侧需要同一数据时必须双 buffer 双 GM→L1 读（k_g2 的 ga_*/gb_* 模式）。

**G15. [64,64] transpose_A gemm_v0 可用**（[64,16] 先例同机制扩展，iter2 首跑即过）；M=48 的
A 操作数 [48,128] 亦可用（iter1）。

**G16. threads=2 对本算子族全面封堵（含纯消费 V 尾段）**：k_g2（V 侧仅 split/GM 写/frag 读，
inter 家族三大破坏因子均不在场）实测 +47% 劣化——vid 拆分开销本身大于双 AIV 收益。与 inter
R1-iter3 结论互证：该方向无需再试。

**G17. V→C auto-relay 造成块级锁步；C 段只读 GM 的 kernel 实现跨块重叠**（msprof：relay 站点在的
kernel AIC 占用≈等待；纯直读 kernel wall≈(C+V)/2）。解法（本轮 iter5）：把 V 侧生产拆成独立纯 V
kernel 直写 ws，消费 kernel 全直读——字节数不变（relay r+w = ws r+w），锁步 flag 消失。

## Stage 3 R2（2026-11-03）——工具链发现补遗（续）

**G18. ⭐prologue 中 set_cross_flag 必须先于一切 V 语句（fixed-core head-set）**：fixed-core 循环
kernel 的循环前 head-set（`T.set_cross_flag("FIX", 41)`）若前面有 tile.fill/tile.copy 等 V 语句，
会被 CombineCV 归类进 V 段 → C 侧永不置位 → drain(0) 的 wait 死锁（aicore 超时）。**head-set 必须
是 prologue 第一条语句**。inter 终态配方（LastRow load 在 set 前）未踩坑是因为其语句组合不同——
不要假设序无关。

**G19. ⭐prim_func 体内局部变量使 region/shape 符号化**：`half = (BS-BC)//2` 这类**体内**赋值是
TIR 符号变量，region 索引（`q_f[0:half]`）与 alloc shape（`[half, BK]`）引用它会得到非常量形状
→ 前端 "size must be same" / PTO "physical buffer shape must be constant"。**派生常量必须定义在
外层 Python 函数**（CORE_EFF 先例），体内按常量解析。

**G20. ⚠️planner 语句结构脆弱性第三实证（挂死形态）**：A 侧 split 临时半行化（region 拷贝/sub
数量翻倍）触发**确定性** aicore 挂死（无报错、3/3 复现）——与 R1-iter6a（数据损坏）、inter R1-iter6
（劣化）同源的 memory-planner 别名 + 名级 auto-sync 组合风险。**任何语句结构重排（即使语义不变）
都必须完整 A/B + 多轮 L0 门禁，且挂死可能级联设备故障**。

**G21. ⚠️设备级 ffts 锁存故障与恢复**：planner 挂死类 kernel（aicore 超时被 timeout 杀死）可使
设备进入 ffts 错误锁存态（"timeout or trap"，跨进程持续，纯 torch NPU op 与简单 tilelang kernel
可部分可用→全恢复，全程 ~40 分钟自愈）。容器内 npu-smi reset 不可用（"cannot be executed on a
VM or container"；sudo 可用但复位属宿主操作）。**操作纪律：遇 EZ9999/ffts 级联先跑纯 V 探针
（k_stage）+ examples/ 已知良好 kernel 判定设备状态，避免在故障态上误判代码**（本轮 R1-final
好代码被误判一次）。
