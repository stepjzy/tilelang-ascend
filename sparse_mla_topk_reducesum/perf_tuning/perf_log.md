
---

# 指标 2 精测（2026-10-09 晚；mode=perf_analysis 只读；golden 配置 220us 判定）
报告 = perf_tuning/golden_metric2_bench.md。数据形态 = golden arange（全任务读相同 128 连续行）——
与历史锚点（randint 散射）不可直接对照（差 -14.5%/-8.6% 为数据局部性效应）。
- **graph-replay 主口径（2 组独立）**：**BI=128 = 364.7us（FAIL +65.8%）/ BI=32 = 475.3us（FAIL +116.0%）**
- msprof 交叉：396.1/479.6us（比值 1.086/1.009，已知 profiler 开销带内 ✓）
- graph-captured e2e 分解：归一化 GPU 成本 ~39us（2 op kernel launch 主导，计算 <1us）；
  eager e2e 1557-1688us = host dispatch 主导（~1.2ms floor + 2 op ~200us，非 GPU 成本）
- 与历史锚点偏差 -14.5%/-8.6%：**全部由 arange vs randint 数据形态解释**（连续行读 L2 全命中 + burst 连续 vs 散射行读；分析轮已证 1.8× 差异）
- **结论：golden 配置 220us 不可达**（BI=128 364.7us = 1.66×；结构地板来自 ws 往返 + 行读引擎，Phase 4 拼装账已定谳）

---

# golden roofline 深度分析（2026-10-09 晚；mode=perf_analysis 只读；指标 2 上限 + 瓶颈归因）
报告 = perf_tuning/golden_roofline.md。核心发现：
- **theoA = 47.6us**（算法最小：V 向量 39.2 + launch 8.4；KV 源仅 288KB/256 行全 L2）——golden 数据形态下算法本身纯计算型
- **架构税 7.7-10.9×**（G1-G4 实测/theoA）；**G1 的 90% = C 侧 ws 302MB GM 往返（数据形态无关的架构强制成本）**
- **G1-G3 差距 77us = 耦合解耦红利**（V arange 提前完成 → C 侧 F0 等待从 105→28us），非 V 提速——arange/randint 源行池相同（均 256 行 L2 热），差异仅在描述符顺序
- **220us 不可达论证**：C 引擎 328us > 220us（即使 V 完全解耦、coupling=0）；消除 ws 往返的全部路径已判死（R2/R5/反转/m_tile，Phase 1-4）
- 数据形态特化路径（新）：arange/共享模式检测 → 256 行 KV 面板 L1 常驻 + 小 dense GEMM → theoA 级 48us → 220 PASS——但为模式特化（通用索引不成立），需 leader 判断交付边界

---

# 指标 2 终验（panel 路径）（2026-10-09 深夜；mode=perf_analysis 只读）
报告 = perf_tuning/panel_metric2_bench.md。核心结果：
- **panel 路径 graph-replay 2 组均值 = 143.3us（141.8-144.3）→ vs 220us PASS（余量 1.53×）**
- msprof 交叉 147.1us（比值 1.027 ✓ 低于典型 profiler 开销带——panel 无敏感 DMA 段）
- generic 回归：BI128 378.9（+3.9% 漂移带内）/ BI32 471.4（-0.8% ✓）；panel-generic torch.equal=True 逐字节等价验证 ✓
- **提速**：2.64× vs generic BI128 / 3.29× vs generic BI32
- **差距归因（143.3 vs 设计 60-80）**：三大设计盲区——① MTE1(L1→L0) 73.2us 未被 gemm L0 乒乓隐藏（52% AIC busy，flag 串行链阻止重叠）；② per-task cross-flag 42us（设计按 per-group 5-10us 估算，实际 ×102 task 频率）；③ Q 读率 38.8 ≠ 160GB/s（争带宽）
- panel AI = 81 FLOP/B（vs generic 4.8）——MTE1 串行化成为新瓶颈（非字节墙）

---
# 指标 2 终验（auto 缺省分发）（2026-10-09 深夜；mode=perf_analysis 只读）
报告 = perf_tuning/auto_metric2_bench.md。核心结果：
- **auto 缺省 + golden arange graph-replay 2 组均值 = 175.7us（174.5-178.7）→ vs 220us PASS（余量 1.25×）**
- auto 三连发 = detect(~15-17) + panel_guarded(145.3) + generic_guarded 空载(~8-10) + graph 开销(~3-5)
- **auto vs shared 开销 = 30.4us**（设计预算 12-14，gap 17-19——detect 的 1MB indices 读未被设计计入）
- **auto + random 分价 = ~21us（+4.6% vs generic 显式 441.7 参照）**——可接受
- **auto + 元数据 miss (topk=256) = 零开销**（586.8 vs 589.1 = -0.4% ✓）
- shared 显式回归 +1.4% / generic BI128 -2.9% / BI32 -0.1%（均在漂移带内 ✓）
- panel-generic torch.equal=True + gate/detect/violations 分发验证全过 ✓
- msprof 交叉不适用（3 kernel 同名聚合平均 12.98us 被空载拉低——如实披露，判定以 graph-replay 为唯一口径）

---

# generic 路径深度流水剖析（2026-10-10；mode=perf_analysis 只读；random 索引 pipe 级重构）
报告 = perf_tuning/generic_pipeline_analysis.md。核心发现：
- **双侧串行链共门控**（修正旧"耦合 105µs"归因）：C 链 4.25µs/块（MTE2 3.25 贴 L2 dirty 46GB/s 天花板 + gemm_v0 队列粒度 flag 链暴露 ~1.0µs）≈ V 链 4.26µs/块（MTE2 行 2.84 + MTE3 1.01 + vec 0.41 严格交替，引擎和/墙钟=1.004）；**跨核 flag 本身近似免费**（F0 wait 18.6µs，F1 176.8µs 被 bulk 排空吸收）
- **C 侧根因（源码级）**：common.h gemm_v0 头/尾 MTE2↔MTE1 队列粒度排空 flag → MTE2(n+1) 无法与 MTE1(n) 重叠；mte1_stall 394µs=scalar 停在发射队列满（引擎 flag 链指示器）
- **SIDE_PAIR 移除 spike**（/tmp 1 行副本）：精度 4/4 torch.equal；wall 不变（+0.2% 噪声）→ **C 绑死实证**；但 V overlap 停 1.04 → 第二把锁 = post 段 wait_cross_flag(F1) 标量阻塞（跨核 flag 无法入队），发射流被锁 C 节拍
- **可攻击组合 P1+P2+P2b**（框架 gemm_v0 per-slot flag + 算子 SIDE_PAIR=False + LEAD=3）：441→~375-395µs（-11~-15%）；**单侧修复 ≈0**（spike 实证 V 侧；C 侧对称推断）
- 封死重申：DEPTH（8 flag id 用满）/ 握手批量（无暴露等待可批）/ burst 整形（已最优+无 gather-DMA）/ 任务重排（±0.8%）/ V-C 再平衡（内存层级强制）——**新结论：全部可回收 bubble 在各侧内部，非跨核间隙**
- **天花板**：random BI128 完美流水极限 ~350-370µs（C MTE2 引擎时间 332µs 不可调度消除）；**220µs 差 1.6-1.8×，generic 路径架构性不可达**（与 golden_roofline 架构税结论独立再确认）
- 意外发现：① PipeUtilization.csv 粒度=每核一行（60=20AIC+40AIV）；② random 税分解 = 两侧 MTE2 各 +16%（C 45.4 vs arange 52.7GB/s 的 L2 slice 争用 + V 44.4 vs 36.4ns/行）；③ BI32=V 引擎重叠 1.30 的自然实验但 V scalar 91% 发射瓶颈 + C 链共门控 → 差 15%

---

# AIV/AIC 联合瓶颈分析与方向裁决（2026-10-11；mode=perf_analysis 只读；P1 证伪后重构）
报告 = perf_tuning/joint_bottleneck_analysis.md。核心发现：
- **联合瓶颈 = 共享 L2 服务带宽**：官方 arange 口径总流量 ~995MB，实测聚合上限 ~2.96-3.0TB/s（真实 kernel 自身即最高观测）→ 带宽地板 ~333µs，当前贴地运行（会话带 335.7-381.5，进程热态 ±7%）
- **旧门控真实身份 = 隐式带宽仲裁器且近最优**：flag 节拍聚合 2.96 vs 自由跑 2.57TB/s（K3 骨架直接测量，C duty 99.7%@87.3GB/s、V 23.5→45.3ns/行双侧退化）——P1 回退的物理机制闭环
- **"V 引擎和≈墙钟"消解**：busy 是 L2 服务占用投影非引擎极限（K2 solo 23.5ns/行证伪饱和说）；"46GB/s dirty 天花板"证伪（K4 脏≈干净 ~128-130GB/s solo）
- **K5 决定性证伪**：128 行拷贝→1 条 bulk（同字节）-1.6%——L2 数据路径绑定，描述符/粒度/连续性整形对 220 全部无效
- **P2 官方口径实测 -1.53%**（配对双轮）+ 精度 5/5 exact + V overlap 1.031→1.102 + C 读率 55.3→59.8（机制=V 锯齿平滑→仲裁间隙利用↑）→ **唯一已验证可采纳候选**
- 方向裁决：P2 ✅ > P2b（LEAD=3，-1~-3% 需 spike）> 半释放门控 ❌（解门控谱系两端已测：全仲裁最优/全解最差，中间无已知更优点）> ws 读率攻击 ❌（K1/K4/K5 架构性贴顶）> 连续性快路径 不推荐（K5 同字节证伪，仅=P2b 特化版）
- 结构探针：topk=64/BI=64 流量减半仅 -3.5%（每块 3.16≈3.28µs）→ 每块/每 task 固定开内地板；H=32 +19.2%（链半暴露）；S=4096 +108.6% 超线性（Q/输出足迹压 L2 容量）
- **220 结论**：需 4.52TB/s（1.52× 观测上限）或流量 -35%（消除路径全封死+K5 数据路径绑定）→ 官方口径不可达；现实区间 333-360µs；P2+P2b 后 ~322-350µs
- 开放角度（不预判硬件下界）：L2 服务率对混合比的未扫描点 / L2 驻留提示原语 / 算法层流量消除（超算子权限）——如实列出未穷尽

---

# P2 交付态终验（2026-10-11；mode=perf_analysis 只读；多实例 + msprof 前后对比）
报告 = perf_tuning/p2_final_bench.md。核心结果：
- **精度 3/3 torch.equal**（官方 arange + 随机尾形；g_l 修复对官方口径行为不变直接验证）
- **BI128 多实例终测（5×2 交错）**：base 364.3 → P2 363.1µs（-0.32%，本轮分辨率地板内）；**跨会话 3/3 方向一致非正（-1.56 / -1.53 / -0.32%）**，量级带 -0.3~-1.6%（点估计 ~-1%），采信依据 = 方向一致 + Stage 2 方差收窄（29.6→7.8µs）+ 机制链
- **BI32 控制臂 +0.04%** ✓（NI≥2 档 TIR 同路径零影响再证）
- **msprof 前后（req 7）**：V overlap **0.983→1.112**（正式化，MTE3 busy +20% 重叠显形签名）；C ws+Q 读率 **60.5→67.9GB/s（+12%）**；V F1 wait -17.7%；C FIX -20.4%；**MTE1/Cube/Vector 逐位不变（确定性锚）**；**每核 spread 10.5→4.1µs**（方差收窄的核级复现）；msprof 相对差 -8.5% 禁用于排名（重申），引擎级作机制证据
- **220 距离**：交付官方终值 363.1µs（G6 锚带 355.2-363.1）= 1.65×，差 143µs；构成 = 带宽地板 ~333（架构性）+ 偏置带 10-30µs + 调度残余 0-5%（P2b 已回退）；**363.1 恰落在联合分析预测带（343-363）上沿**——generic 官方口径已贴联合带宽地板 + 偏置带
- 交付态 kernel 本轮零改动（工作树 diff 仅 P2 交付差异，git 核验）
