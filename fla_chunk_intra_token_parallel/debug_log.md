# debug_log — fla_chunk_intra_token_parallel（沙箱多算子共存，按算子前缀隔离）

## Attempt 1 — 2026-10-22T20:35:00+08:00
- mode: first_impl
- classification: precision_pass
- fail_category: none
- test_level: all（L0 先行收敛 1 轮修复后通过；随后场景 B 扩展 + 覆盖门禁 + 全量）
- coverage: L0:3（+1 golden 自检） L1:9 L2:10 Boundary:4；coverage_check 20 PASS / 0 MISS / 0 N/A（exit 0）
- boundary_warnings: none（L2 10/10 正确拒绝，Boundary 4/4 精度通过，均 [BOUNDARY_PASS]）
- changes:
  - 新建 `fla_chunk_intra_token_parallel.py`：Developer 模式纯 Vector kernel（§3.3 伪代码落地：subchunk-parallel 一维 grid + (cid,vid)、静态 i/j 循环 + guard、exp(x·ln2) 替代 exp2、fp32 计算域、fp32 staging + CAST_RINT、UB↔UB 跨 dtype cast 链）+ host 包装（域断言 + BH 8→4→2 自适应）。
  - 新建 `test_fla_chunk_intra_token_parallel.py`：golden（§9.1 草案 + 2 处 bug 修复）+ naive 三重循环交叉校验 + check_precision（precision-standard §4.1 逐字）+ L0 三用例（§9.2 逐字）+ 场景 B L1/L2/Boundary + COVERAGE_MANIFEST。
  - **golden 草案修复 1（语义）**：`kgj = k_sub.unsqueeze(1) * e`（草案为 unsqueeze(2)，会把 k 的 token 轴对齐到 e 的 i 轴，算出 q_i·k_i·2^(g_i−g_j)，与 Triton 权威 kgj=k_j·2^(g_i−g_j) 矛盾）。经 naive 交叉校验确认。
  - **golden 草案修复 2（形状）**：masked_fill 掩码需 `[1,SCS,SCS,1]`（草案 `[SCS,SCS]` 无法与 `[B,SCS,SCS,H]` 广播，运行时崩溃）。
  - **实现层对伪代码的必要修正（均非设计层）**：
    1. 载入 dst 必须是 region（`q_h[t*BH_V:(t+1)*BH_V, :]`）；草案的标量 dst `q_h[t*BH_V, 0]` 两侧均无 extent，copy 推断直接断言失败。
    2. staging 用 3D `[SCS, BH_V, CS/SCS]`（元素布局与草案 2D `[SCS, BH_V*CS]` 同构）：草案"单次 2D 行间 gap T.copy 写回"在 TIR region 模型下不可表达——3 个 active 维（S,H,CS）的 region 在 2D DMA 模型里外维被丢弃（静默截断，仅写 1 个 token）或 extent 超出 H 维（越界）。改为 per-token flat burst（`aqk_h[t,:,:] → Aqk[bb, i_ts+t, h0v, 0]`，每次 [BH_V,CS] 连续段 512B/128B packed flat copy）。
    3. **beta 窄行 DMA 修复（本轮唯一精度 bug）**：beta 载入为 [SCS,BH_V]=8B/行 窄行 DMA，MTE 窄行落在 32B slot（common.h "narrow UB rows sit in 32B slots" 约定），packed [SCS,BH_V] 声明与 DMA 实际落位错位 → beta_f 全错 → Akk 非零值全错（首跑 Akk ratio≈0.54 = 恰好零元素占比 136/256，Aqk 完美佐证隔离）。修复：staging 垫宽为 `[SCS,16]`（32B 行距）+ 列区 region `[:, 0:BH_V]` 作 dst + fill 清零 + flat 256 元素 cast。
    4. beta 列提取用标量散写（`beta_col[h,0]=beta_f[i,h]`）；草案的 `T.copy(beta_f[i,0:BH_V], beta_col[:,0])` 触发 copy extent 逐维 max（[1,BH_V]×[BH_V,1]→[BH_V,BH_V]）不可用。
- error_summary: 首跑 L0：Aqk 3 用例全过（ratio=1.0000），Akk 全错（ratio≈0.54，max_abs 45~97）；根因 beta 窄行 DMA 32B slot 错位（见 changes.3），修复后 L0/L1/L2/Boundary 全绿。另：首跑前 golden 自检抓出 masked_fill 形状崩溃（修复 2）。
- design_error_reason: none。发现的问题均为实现层（golden 草案笔误 + TIR region/DMA 细节），设计的核心决策（subchunk 分块、静态循环+guard、exp(x·ln2)、cast 链、staging+散写）全部成立；R3（reduce_sum dim=-1+clear 组合）、R8（标量散写）首轮即验证通过，R4（UB 138KB/192KB）BH=8 编译通过。
- rollback: no
- backup_path: n/a（first_impl，无 history_version 备份义务）
- instrumentation_cleaned: n/a（未使用临时插桩；问题通过 Aqk/Akk 症状隔离 + lowering 源码分析定位，未改 kernel 调试）
- next_hint:
  - Stage 3（perf_tuner）：perf_bench 哨兵 (1,8192,64,128) 未跑（按任务书本轮不做性能）；DESIGN §9.5 已论证 370us 目标物理不可达（需 1.33× 实测带宽上限），标定后按替代路线重定口径。
  - 输出写回为 16+16 次/block 的 flat burst（512B/128B），载入为 16×3 次行拷；Stage 3 可评估 3D 聚合搬运与 (i+1) 行批量 tile 化（DESIGN R5/R6）。
  - L1 已含 BH=4/BH=2 头组自适应路径与 SCS=8 泛化路径；若未来放开 CS≠64 需注意 staging 随 CS 线性增长（CS=128 时 UB 峰值接近上限）。
  - beta 窄行 32B slot 约定建议沉淀到 tilelang-api-best-practices（GM→UB 窄行 DMA 的 dst 声明必须按 32B 行距垫宽）。
