# fla_chunk_intra Stage 3 调优日志（perf_log）

基线口径：do_bench（NPU 事件，warmup=20 / rep=10 / mean，仅设备侧），fwd shape (1,8192,64,128) bf16 / bwd shape (1,8192,8,128) fp32，同会话 A/B（旧形态自 history_version 快照模块导入）。噪声阈值 3%。精度门：每轮 L0 全量（bench 配置 + L0 代表双过），终版 --level all。

## Iteration 1 — 2026-11-03T01:20+08:00（fwd k_inter 锚点消去超级 GEMM）
- bottleneck_type: transfer+compute（k_inter AIC 占用 12.0us/块，其中生产性仅 5.2：V 侧 scalar 发射 8.6us/块 ~110 tile op + UB→L1 中继 84KB/块 18 站点；msprof_intra_inter）
- optimization: [#1] 锚点在乘积内精确相消（qg_i·rhs_j = q_i k_j 2^{g_i-gn}·2^{gn-g_j}），9 组锚定操作数 → 3 组全宽操作数（Q̃=q·2^g [48,128]、ktp=k·2^g [48,128]、K̃=k·2^{-g} [64,128]，A 侧仅 rows16-64 乘积有效），18→6 gemm_v0（[48,128]@[128,64] + [48,128]@[128,64] transpose_B）；β 行缩放移 V 尾段 fp32 精确乘（精度优于旧 bf16 操作数内折叠）；中继 84→48KB/块；V op ~110→~60
- 过程工程事实：① A 侧误用 K̃(2^{-g}) 致乘积 2^{-g_i-g_j}~2^{+100} 垃圾（Akkbp 探针 1e8-1e9 精确锁定）→ A 侧改独立操作数 ktp=k·2^{+g}；② UB LinearScan 192KB 贴线溢出（3×32KB f32 操作数 + 64KB hi/lo + 32KB relay 对）→ e_f 就地复用（exp 链 dst==src1 交换律已验证模式）+ A 侧操作数缩 48 行（block(0,*) 乘积全弃）→ 峰值 ~178KB
- baseline_time: k_inter 5133.8/5139.1 us（同会话 A/B ×2，与基线 5164.1 一致）
- candidate_time: k_inter 3518.7/3516.8 us
- improvement: **-31.5%**（k_inter 单核）/ fwd chain 16451→14821 us（-9.9%）
- precision: pass（L0 7/7；Aqk/Akk ratio 全 1.000000；Akk bench max_abs 6.25e-2→1.25e-1，limit 1e0 内 8× 余量——尾段 fp32 β 缩放 + M=48 gemm 的舍入路径差异；Akkbp 新旧差 4.1e-4 = hi/lo 舍入级）
- adopted: yes
- rollback_reason: 无
- next_hint: [#2] bwd 中点锚点超级 GEMM（k_stage+k_gemm 重写：staging 210→128KB/cid、48→12 gemm_v0/半、exp 元素 -84%）

## Iteration 2 — 2026-11-03T02:20+08:00（bwd 中点锚点超级 GEMM）
- bottleneck_type: transfer（k_gemm mte1_stall 26.6us/块 = GM staging 读 + k_stage V 串行链 vector_stall 35 + mte3 25.7；msprof_intra_{stage,gemm}）
- optimization: [#2] gn_s/gn'_s 锚点在 D_s/V_s 后置缩放内精确相消 → 每 matrix role 一组共享操作数（16 逻辑 gemm→4）：MdA/MdK 全量 [64,64] hi/lo（KMAX 列窗冗余——Mlo 已掩码 j>i≥16s）+ 每半 K̃=k·2^{a-g}/Q̃=q·2^{g-a}/KB~=βk·2^{g-a}（a=g[行32] chunk 中点锚，fp16 操作数指数 ≤ 半 chunk 衰减 2^12.2 < 旧 envelope 2^18.24——bnd 溢出域反而收缩）；staging 210→128KB/cid；exp 元素 -84%（每半 ~20 块 [16,BK] → 2 块 [64,BK]）；V 尾段 8 条 per-s D/V 链 → 2 条共享 ea/eam [64,BK] + per-s 3 mul
- 过程工程事实：[64,64] transpose_A 首跑即过（[64,16] 先例同 zN 机制）；Pipelined 体语句结构未动（FIX 9/10 flag 预算不变）
- baseline_time: k_stage 1202.4 / k_gemm 2107.6 us（同会话 A/B ×2 mean；与基线 1195/2088 一致）
- candidate_time: k_stage 648.5 / k_gemm 819.8 us
- improvement: k_stage **-46.1%** / k_gemm **-61.1%** / bwd 合计 3298→1468 us（**-55.5%**）
- precision: pass（L0 7/7 全绿；bwd max_abs 普遍改善：l0_bwd_min dg2 1.2e-4→7.6e-5、dg_post 3.7e-4→1.5e-4；新旧输出差 2.4e-4/7.3e-4/6.1e-4 = hi/lo 舍入级）
- adopted: yes
- rollback_reason: 无
- next_hint: [#3] fwd 预分裂 ws 直读链（k_g2/y/ai 的 A/B 操作数中继 96KB/块 → GM→L1 直读 48KB；I+ 折叠进 V 尾段 fp32 加法——Y=Aid+G2@Aid、Ai=Y+G@Y 比旧 (I+X) 先分裂后乘精度更好；G 需零填 rows 0-16——G@G 的 c<16 列会放大 Akkbp 垃圾行）

## Iteration 3 — 2026-11-03T03:05+08:00（fwd 预分裂 ws 直读链）
- bottleneck_type: transfer（k_g2/y/ai 每次从 ws 读 fp32 后重 split + 48KB/块 中继 r+w（96KB 流量/块）+ V 侧 staging op；k_g 等 C 8.6us/块）
- optimization: [#3] 五个 ws（Akkbp/Aid/G/G2/Y）转 bf16 hi/lo 对（字节数不变），下游 k_g/k_g2/k_y/k_ai 的 C 段 GM→L1 直读（中继清零，bwd k_gemm 已验证形态）；I+ 折叠进 V 尾段 fp32 精确加法（Y=Aid+G2@Aid、Ai=Y+G@Y——比旧 (I+X) 先分裂后乘精度更好，Akk 新旧差 3e-8）；Aid fp32 由 hi+lo 尾段重建
- 过程工程事实（本轮两个深层发现，全部实证）:
  1. ⭐0×NaN=NaN：Akkbp ws 未写的 rows 0-16 垃圾为 NaN 时，aid 块对角精确零在 Cube MAC 上**不能**屏蔽（NaN 注入探针：G rows16+ 被毒化 18432 元素→全链 NaN；干净垃圾时侥幸通过=间歇性，L0 dk64 连续 3 次 ratio 0.625 复现）→ k_inter 显式零写 rows 0-16（ws 全域确定性写出），k_g 原 zero16 暂存移除
  2. UB 192KB 精算：H=64 的 beta gather span（[48×64]bf16=6KB）使 k_inter 峰值超线 ~1KB → A 侧 split 临时独立为 [48,128]（省 16KB）
- baseline_time: fwd chain 14854.8 us（同会话 A/B；k_inter 3524.9/k_g 5676.9/k_g2 1577.0/k_y 2417.2/k_ai 1848.4）
- candidate_time: fwd chain 12152.7 us（k_inter 3562.4/k_g 5267.8/k_g2 1052.3/k_y 1235.5/k_ai 1154.1）
- improvement: **-18.2%**（fwd chain）/ 累计 fwd 16451→12153（-26.1%）；k_y -48.9%、k_ai -37.6%、k_g2 -33.3%、k_g -7.2%、k_inter +1.1%（密集 hi/lo 写 +10KB/块，噪声级）
- precision: pass（L0 23/23 全绿；dk64 修复后 ratio 1.0/max_abs 4.9e-4——预分裂链精度反升）
- adopted: yes
- rollback_reason: 无
- next_hint: k_g 5268us 现占 fwd 43%（串行前代 V 链 + 每 op 屏障）→ [#4] 4 路 s 链 ILP 交织（per-s 临时 buffer 消跨链 WAR 屏障）；bwd 1468us 贴 2-kernel 结构地板（~1.3ms @600GB/s）

## Iteration 4 — 2026-11-03T03:40+08:00（k_g 前代 ILP 交织，回滚）
- bottleneck_type: compute（k_g V vec 6.8us/块 + vector_stall 8.2（逐 op 屏障）+ C wait_id0 9.7（等 aid relay）；msprof_intra2_g）
- optimization: [#4] 4 条 s 前代链单循环交织（per-s 临时 buffer a_row0..3/tmp0..3/red0..3/new0..3，假设消跨链 WAR 屏障后 V 管线可 4 路重叠）
- baseline_time: k_g 5268.9/5264.4 us（同会话 A/B ×2；iter3 采纳态）
- candidate_time: k_g 5395.4/5399.4 us
- improvement: **+2.4~2.6%（劣化）**
- precision: pass（L0 全过；G 新旧逐位一致 0.000e+00——纯调度效应）
- adopted: no
- rollback_reason: 两轮一致劣化。证伪假设：串行循环下共享临时不产生跨链屏障（循环本就顺序执行，auto-sync 的 WAR 点在循环内已就地串行），交织无屏障可消；反而 4× 临时集（+8.8KB UB）扰动 memory-planner 布局。教训：屏障结构收益需 msprof 证据先行（vector_stall 8.2 的来源是链内 RAW 屏障而非跨链 WAR），跨链 ILP 在名级 auto-sync 下不可兑现
- next_hint: k_g 剩余墙 = 链内逐行 RAW（语义固有，前代行递归）+ C 等 V（9.7us）——Developer 模式内无进一步手段；候选 = fixed-core 任务循环结构（inter R1-iter2 同构，L2 常驻中继 + 块间 C/V 重叠）或接受为地板

## Iteration 5 — 2026-11-03T04:10+08:00（k_g 拆分：纯 V 前代 kernel + 无 relay G gemm kernel）
- bottleneck_type: sync（k_g C 占用 12.4us/块中 9.7us 等 aid relay flag——V→C relay 块级锁步；对照：C 段只读 GM 的 k_y/k_ai/k_g2 实现跨块重叠 wall≈(C+V)/2；msprof_intra2_g vs _y 证据）
- optimization: [#5] 旧融合 k_g 拆为 k_subst（纯 V：4×14 行递归 + aid staging + split + AidHi/Lo 全量写）+ k_g'（C 段全直读 AidHi/Lo + AkkbpHi/Lo → 3 gemm_v0 → frag relay → G split/写）。AidHi/Lo 本就是 ws 契约（iter3），零新 ws；relay r+w 字节 = ws r+w 字节（等量），锁步 flag 消失，AIC 跨块自由流动
- baseline_time: fused k_g 5266.5/5273.0 us（同会话 A/B ×2；iter3 采纳态）
- candidate_time: k_subst 2564.0 + k_g' 1201.0 = 3765.0 us
- improvement: k_g 阶段 **-28.5%**（fwd chain 12153→~10650 预期）
- precision: pass（L0 全过；AidHi/G 新旧逐位一致 0.000e+00——纯调度重排，数学路径不变）
- adopted: yes
- rollback_reason: 无
- next_hint: fwd 剩余：k_inter 3.5ms（V 侧 op 发射 + C 锁步 wait_id6）+ k_subst 2.56ms（纯 V 串行链=语义固有地板）+ k_g'/g2/y/ai ~4.6ms（frag relay 锁步 + 小 kernel 开销）；同款拆分对 k_inter 的收益待精算（其 V/C 本就 ~2× 重叠，拆分增 kernel 边界）

## Iteration 6 — 2026-11-03T04:35+08:00（k_g2 threads=2 试点，回滚）
- bottleneck_type: compute（4 个小 kernel 的 V 尾段仅用 ~20 AIV——mix block 每块 1 AIV，40 AIV 半闲置；假设 threads=2 唤醒双 AIV）
- optimization: [#6] k_g2 threads=2（V 尾段 = split + GM 写 + frag relay 读——无 broadcast/reduce/relay-writer，inter 家族 threads=2 三大破坏因子均不在场）
- baseline_time: k_g2 1084.8/1081.7 us（threads=1，同会话 A/B ×2；iter5 采纳态）
- candidate_time: k_g2 1599.9/1592.5 us
- improvement: **+47%（劣化）**
- precision: pass（G2 逐位一致 0.000e+00）
- adopted: no
- rollback_reason: 两轮一致 +47% 劣化。threads=2 的 vid 拆分开销（C→V frag relay 的双 vid 消费 + 调度）远超 V 半闲置损失——与 inter 家族 threads=2 结论一致（该方向对本算子族全面封堵，即使 V 侧语义安全）。已回滚至 iter5 采纳态（md5 一致）
- next_hint: 终态固化：--level all 全量 + 官方口径 bench ×3 + 终态 msprof + 地板核算

# R2（用户决策：再调一轮后交付，验收目标 ≈10.5ms ±0.5）

## R2 Iteration 1 — 2026-11-03T09:50+08:00（k_inter fixed-core 任务循环 + 显式 L2 relay）
- bottleneck_type: sync（k_inter C 占用 7.4us/块 vs 生产性 2.7——per-block auto-relay 锁步 wait_id6；msprof_intra3_inter）
- optimization: [R2-#1] inter 家族终态配方移植：T.Kernel(20) + 单 T.serial(L=BASE+2) 条纹循环（task=cid+j×20，过界 Select 重定向 cid 幂等重算）；6 组 V→C 操作数 relay 改 (CORE,NSLOT=4) 显式 workspace（8.3MB L2 常驻 vs auto per-cid ~640MB）；2 组 C→V 结果 relay 改手动交替 FIX flag（40+j%2→槽 8/9）+ 1-late drain（wait 40+(j+1)%2 ↔ set 40+(j-1)%2）+ 循环前 head-set(41) 预充；bet 走 (NSLOT,48) UB 环（唯一 loop-carried V 状态）
- 过程工程事实：① bet 环读作 broadcast src 时 [expr,region] 被物化为 [1,48]（inter 的环读是 mul/add 操作数语境无此问题）→ 先 copy 到 [48] 纯 buffer 再 broadcast；② A/B 脚本变量命名 bug（Lo 克隆误名 h_ref）制造了 72 的假损坏信号——|lo−hi|≈hi 量级，定位后修正；kernel 本身 19+ 次（锤击 12 + 探针/复刻 7）逐位一致且确定性
- baseline_time: k_inter 3554.1/3553.4 us（同会话 A/B ×2 两轮会话；R1 终态代表值 3564.3）
- candidate_time: k_inter 3122.8/3125.0 us
- improvement: **-12.1%**（k_inter 单核；fwd chain 预期 10725→~10182）
- precision: pass（L0 23/23 ×3 次 + L1 全过；新旧输出逐位一致 0.000e+00——纯调度重排）
- adopted: yes
- rollback_reason: 无
- next_hint: [R2-#2] 小 kernel（k_g/k_g2/k_y/k_ai）frag 回传 L0C→GM 直写 + 手动 FIX flag（bwd k_gemm 先例）；预期合计进 10.1-10.7 带

## R2 Iteration 2 — 2026-11-03T11:00+08:00（k_y frag 回传协议试点，回滚）
- bottleneck_type: sync（小 kernel 的 frag C→V auto-relay 依赖等待 wait_id0 2.7-4.3us/块）
- optimization: [R2-#2a] k_y 试点：frag 回传改 L0C→GM 直写（YScr [GRID,64,64] fp32）+ 手动 FIX flag（bwd k_gemm 先例），替代 auto C→V relay
- baseline_time: k_y 1227.2/1225.4 us（同会话 A/B ×2）
- candidate_time: k_y 1207.7/1212.4 us
- improvement: **-1.1~-1.6%（低于 3% 噪声阈值）**
- precision: pass（YHi/YLo 新旧逐位一致 0.000e+00）
- adopted: no（按规则回滚）
- rollback_reason: 小 kernel 已在 max(C,V) 跨块重叠极限（R1 msprof：wall/block ≈ (C+V)/2），协议换手不消依赖等待（frag 数据依赖固有）；字节等量（relay r+w = scratch r+w）。R1 估的 -0.3~0.6ms 偏乐观，方向证伪后不扩展到 k_g/g2/ai
- next_hint: [R2-#2b] k_inter drain 块放置暂存 → 常量 mask 乘（估算 -0.25ms）

## R2 Iteration 3 — 2026-11-03T10:15~12:30（k_inter drain mask 乘，放弃——planner 脆弱性 + 设备故障）
- optimization: [R2-#2b] drain 的 6+6 次 [16,16] 块放置暂存改 0/1 块下三角常量 mask 乘（1.0/−1.0/0×x 逐位精确；−14 V op/task ≈ −0.25ms 预估）
- 过程工程事实（全部实证，逐条记录）:
  1. 宿主 mask helper 语义 bug ×2：cols 参数传维度而非块数 → [48,1024]；修后 helper 又被传 rows=3 → [3,64]（sum=0）——kernel 按声明形状读扁平元素 = 乱码，ratio 0.88 定位
  2. prologue 语句序敏感：[fill, load(mask), set(41)] 与 [load, fill, set] 均 aicore 超时挂死（set_cross_flag 被 CombineCV 分类进 V 段 → head-set flag 永不置位 → drain(0) wait 死锁）；**head-set 提到 prologue 最前（先于一切 V 语句）后通过**——inter 终态配方未记录此序约束
  3. UB 192KB 溢出：双 mask（+24KB）与单 0/1 mask（+12KB）在 bench shape H=64（beta gather span 6KB）均溢出 ~17KB → A 侧 split 临时半行化 [24,128]（−24KB）
  4. ⚠️**体内局部变量使 region/shape 符号化**：`half = (BS−BC)//2` 写在 prim_func 体内 → TIR 符号变量 → 前端 "size must be same" + PTO "physical buffer shape must be constant"；**必须在外层 Python 函数定义**（CORE_EFF 同款）后常量解析 ✓
  5. ⚠️半行 split 语句结构触发**确定性 aicore 挂死**（3/3 复现；R1-iter6a planner 别名/屏障类）——放弃
  6. ⚠️**设备级故障事故**：调试期间的挂死 kernel 使设备进入 ffts 锁存错误态（aicore/aivector "timeout or trap"，跨进程持续 ~40 分钟；纯 torch op 与 examples/ gemm 部分可用→全恢复），容器内无法 npu-smi reset（"cannot be executed on a VM or container"；sudo 可用但驱动工具需宿主）。**教训：planner 挂死类实验后若见 EZ9999/ffts 级联，先跑 device_check 探针再继续，避免在故障态上误判代码问题**（本轮 R1-final 好代码被误判挂死一次）
- baseline_time: k_inter 3139 us（R2-#1 采纳态）
- candidate_time: 未获有效数据（UB 溢出→半行拆分→挂死→设备故障，未能完成 A/B）
- improvement: 无（放弃）
- precision: L0 曾通过（双 mask + head-set-first 序）；无终态
- adopted: no（放弃，回滚至 R2-#1 态，md5 ef253da1d630b062e4f4fe1269bef269）
- rollback_reason: 收益预估仅 −0.25ms，代价 = UB 溢出 + planner 确定性挂死 + 设备故障风险；按「失败当场记录当场回滚，不恋战」纪律止损。该微优化归类为「需工具链配合（planner 别名/屏障鲁棒性 + UB 预算）」
- next_hint: R2 终态固化（4 次 bench 带 + --level all + 终态快照）

## R2 终态汇总 — 2026-11-03T14:30+08:00

- 终版 kernel 时间（4 次独立官方口径运行，设备恢复后）：fwd chain 10148.2/10170.2/10135.1/10183.8 → **10159.3**；bwd chain 1303.6/1285.6/1272.4/1272.2 → **1283.5**；**合计 11451.8/11455.9/11407.5/11456.0 → 11442.8**（带内散布 48us）
- 相对 R1 终态 12027.5：**-4.9%**（k_inter 3564→3139 = -11.9%）；相对 Stage-2 基线 19774.3：**-42.1%**
- 精度：--level all **90 PASS / 0 FAIL / 0 WARN**，exit 0；ruff clean；终态快照 history_version/fla_chunk_intra_perf_r2_final.py（md5 ef253da1…）
- **可达带判定：未进带**（11443 > 10700 带上沿）
- 采纳链：R2-#1 k_inter fixed-core 任务循环 + 显式 L2 relay + 手动 1-late FIX flag（k_inter 单核 -12.1%）
- 回滚/放弃链：R2-#2a k_y frag 协议（-1.1~-1.6% 子阈值）；R2-#2b drain mask（planner 脆弱 + 设备故障止损）
- 地板核算与归因：见 fla_chunk_intra_final_summary.md R2 节
