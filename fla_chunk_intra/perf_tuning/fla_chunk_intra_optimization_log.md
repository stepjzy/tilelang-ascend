# fla_chunk_intra Stage 3 优化记录（optimization_log）

基线：见 `fla_chunk_intra_baseline_r1_iter0.json`（fwd chain 16451us / bwd chain 3291us / 合计 19742us，
attempt-2 数据 0.2% 内复现）。画像：`msprof_intra_{inter,g,g2,y,ai,stage,gemm}/`，解析
`parse_intra_msprof.py`。

## 画像结论（per-block 均值 us；wall×20/8192 = 每块墙钟 fwd ≈ 12.5-14.0，bwd ≈ 22.5-39.4）

| kernel | wall us | AIC 占用 | AIC 生产性(cube+mte1+mte2+fix) | AIV 生产性 | 主导等待 |
|---|---|---|---|---|---|
| k_inter | 5111 | 12.0 | 5.2 (mte2 中继读 2.6, scalar 3.2) | vec 5.7 + mte2 3.6, scalar 8.6(op 发射) | ~3.6 gap/锁步 |
| k_g | 5739 | 13.4 | **1.1** | **vec 7.7**(串行前代) + scalar 4.6 | aic wait_id0 8.6(等 V 产操作数), aiv vector_stall 9.0 |
| k_g2 | 1606 | 3.3 | 0.9 | 1.9 | aiv wait_id4 5.2(等 C 回传) |
| k_y | 2327 | 4.4 | 1.1 | 2.5 | wait_id4 3.1 + wait_id0 2.5 |
| k_ai | 1815 | 3.8 | 0.9 | 2.2 | wait_id4 3.2 |
| k_stage | 1152 | — | — | **vec 26.9 + mte3 25.7**, vector_stall 35 | 串行 V 链 |
| k_gemm | 2017 | 38.5 | cube 3.0 + mte1 4.6 + **mte2 24.0** + fix 4.0 | 6.0+8.1+4.0 | **mte1_stall 26.6**(GM staging 读), wait_id9/10 33.5(V 等 FIX) |

归因：
- fwd = transfer+sync 混合：每 kernel V→C→V 锁步（AIC wait_id0 占 AIC 占用 65% in k_g），
  auto-relay（UB→L1 workspace 中转）静态 ~576KB r+w/program ≈ 4.7GB fwd 总流量（DESIGN 模型外），
  V 侧 scalar 发射流（k_inter 8.6us/块）与串行前代（k_g vec 7.7us/块）为 V 墙。
- bwd = transfer：k_gemm 的 GM staging 读（mte1_stall 26.6/块）+ k_stage 的 V 串行链（vector_stall 35）
  与 GM 写（mte3 25.7）。静态 staging 流量 ~678MB。

## Part A 优化点清单

[#1] fwd k_inter 锚点消去超级 GEMM（参考: optimization-guide.md §2.13/反模式"tile size 过小"；
     antipatterns.md L272-306）: **适用** — Aqk[s2]=scale·(q⊙2^g)@(k⊙2^{−g})^T 与
     Akkbp=K̃B@K̃^T 的逐 s2 锚点在乘积内精确相消（2^{g_i−gn}·2^{gn−g_j}=2^{g_i−g_j}），9 组操作数
     （qg/kgp/rhs ×3 s2, 18 gemm_v0, ~100 V op）塌缩为 3 组全宽操作数（Q̃/K̃/K̃B, 6 gemm_v0,
     ~60 V op），中继字节 84→48KB/块。bf16 域（±2^127）无溢出风险；bwd bnd 类极端衰减 fwd 侧
     gk≤0（logsigmoid）且上三角块乘积（≤2^117·16<2^128 fp32）不溢出、不写出。丢弃块产物无污染。
[#2] bwd k_stage/k_gemm chunk 中点锚点超级 GEMM（同 §2.13；guide §2.4 L324-351 减少重复载入）:
     **适用** — dq2_pre_s/dk2_pre_s/dkt_s 的 gn_s/gn'_s 锚点在 D_s/V_s 后置缩放内相消
     （dq2_s[i]=2^{g_i−a}(MdA@K̃)[i], K̃=k⊙2^{a−g_j}；dkt[i]=2^{a−g_i}(MdA^T@Q̃+MdK^T@K̃B)[i]，a=chunk 中点行 32）。
     16 逻辑 gemm/半 → 4（48→12 gemm_v0）；staging 210→128KB/cid；exp 元素数 -84%；
     KMAX 窗口列对 MdA 无约束（MdA[i,j]=0 for j>i≥16s 早已被 Mlo 掩码，KMAX_s>16(s+1)>i 恒成立）。
     锚点 a=g[行32] 使 fp16 操作数指数 ≤ 半 chunk 衰减（bnd −0.38/token → ≤2^{11.9}，与现 envelope
     2^{11.8} 等价，bnd WARN 不扩大）。KERNEL-REUSE-PRECHECK: 候选不扩调用域（同 shape/dtype/grid，
     仅 GM staging 布局与 gemm 形状变化——bwd k_gemm 的 [64,64] transpose_A 为唯一新形态，
     [64,16] transpose_A 已验证（G7 修复链），[64,64] 走同机制，需探针确认 → plan: repair-kernel-first（先单独验证）。
[#3] fwd 预分裂 ws + C 侧直读链（参考: guide §2.4 减少重复载入/一次驻留；antipatterns.md L298
     "workspace 访存量 < L2 时更可能命中"）: **适用** — k_g2/k_y/k_ai 每次从 ws 读 fp32 后重 split
     （48KB 中继 r+w/块 ×3 kernel）。改为上游 kernel 直接写 hi/lo bf16 ws 对（同字节数），下游
     C 段 GM→L1 直读（bwd k_gemm 已验证形态），消除中继 r+w（96KB→48KB 单读/块）；
     I+ 折叠进 V 尾段 fp32 加法（Y=Aid+G2@Aid, Ai=Y+G@Y——比现 (I+G2) 先 split 后乘**精度更好**，
     fp32 精确加 vs bf16 hi/lo 的 I 部分舍入）。Akkbp/Aid/G/G2/Y 五个 ws 全部转 hi/lo 对。
     frag 回传改 L0C→GM 直写 + GM→UB 读回（bwd Fragscr 先例，零 flag）替代 C→V 中继。
[#4] k_g 串行前代 ILP 交织（antipatterns.md L127-153 "逐行 for loop"）: **部分适用** — 4 个 s 的
     前代存在真实迭代依赖（该条注明"不适用"于不可等价重排场景），但 4 条 s 链相互独立，
     逐行交织（i 外层、s 内层）暴露 4 路 ILP，压 aiv vector_stall（9.0us/块）。语句重排在
     planner+auto-sync 下高风险（inter R1-iter6 教训），同 V 管线内重排风险较低，实测裁决。
[#5] fixed-core(20) + 显式 L2 常驻中继 workspace（guide §2.9 L487；inter R1-iter2 同构 -11.3%）:
     **适用但延后** — fwd 5 kernel 全部重构为任务循环 + 手动 flag（24 flag 截断问题在循环结构下
     变为真竞态，需 inter 式交替 ID 方案），工作量/风险最大，在前 3 项落地后评估剩余预算再决定。
[#6] bwd OpscrM 跨半重读消除（guide §2.4）: **低优先** — Md tile 32KB ≤ L1/2、T=2 满足条件，但
     OpscrM 总量 56MB < L2 192MB，第二次读大概率 L2 命中（msprof L2 数据可证）；且环外提载
     需"直线 C 段 + Pipelined 体"混用（模块注记 1 未验证形态）。除非 #2 后画像仍显示 mte1_stall 主导。

[ORDER-PLAN] 实施顺序：
1. [#1] fwd k_inter 超级 GEMM — 前置依赖: 无 — 理由: fwd 最大单项、独立 kernel 可单独 A/B、
   精度风险低（bf16 域 + 同相对舍入路径）
2. [#2] bwd 中点锚点超级 GEMM — 前置依赖: 无 — 理由: bwd 最大单项（staging -40% + gemm 48→12）；
   与 #1 代数同源，独立验证
3. [#3] fwd 预分裂 ws 直读链 — 前置依赖: [#1] — 理由: 改 ws 契约需在 k_inter 新形态上叠加
   （Akkbp 输出转 hi/lo）；k_g/k_g2/k_y/k_ai 连锁改动作为一个 ws 契约项整体验证
4. [#4] k_g ILP 交织 — 前置依赖: [#3] — 理由: 在 #3 后 k_g 的 V 相结构稳定后再做语句级重排
5. [#5] fixed-core 重构 — 前置依赖: [#1][#3] — 理由: 结构基板稳定后评估；预算不足则放弃并在
   终局归因中说明
