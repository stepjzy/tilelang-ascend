# fla_chunk_inter Stage 3 优化记录（optimization_log.md）

> 基线：19469.7 us（do_bench warmup=20/rep=10/mean/设备侧，同会话复测；调度基线 19556.3 us）
> 目标：< 3400 us（纯 kernel 时间）。shape: B=1,S=8192,H=64,DK=128,DV=128,chunk=64,scale=1.0,fp32。
> 算子类型（Step 2 实证）：**混合型**（生成代码 KERNEL_TYPE_MIX_AIC_1_1；`main_kernel<<<32768>>>`，
> 每 block = 1 AIC + 1 AIV；IS_ASCEND_AIC / IS_ASCEND_AIV 分支均在）。
> 生成代码静态画像（perf_tuning/fla_chunk_inter_kernel_source_iter0.c）：
> - 每任务 ~150 同步站点（AIC 侧 2 iter × ~33 + epilogue；AIV 侧 2 iter × ~37 + epilogue）
> - Pipelined(num_stages=2) **实际未软件流水**（生成嵌套普通 for，单缓冲顺序执行）——家族结论复现
> - 12 个 auto workspace 中继全部按 `cid` 独占槽位：**总足迹 ≈ 4.5GB**（workspace_1-4 各 256MB、
>   workspace_5-10 各 512MB、workspace_11/12 各 256MB）→ 远超 L2 192MB，中继 9GB 写+读击穿 L2 驻留
> - scale=1.0 时 `Muls 1.0` 未折叠（每任务 1 个冗余向量 op）
> - 唯一 HBM IO 3.76GB；含中继的有效流量 ~13.3GB → 19.5ms 对应 ~680GB/s 有效带宽 = 峰值的 54%
>   → **中继 HBM 流量 + 每任务同步协议**双重主导（非纯带宽墙、非纯协议墙）

## Part A 优化点清单（适用性判定）

[#1] i_v 循环全展开为直线单发（bDV=DV，K=DV gemm，中继 20→10、gemm 18→9、装载 10→5、
  dgkn 免累加）（参考: optimization-guide.md 循环展开/家族 fla_chunk_o iter2 "Pipelined 循环
  展开为直线全 DK（大 DMA、中继 4→2、gemm 4×K64→2×K128、flag 减半）-8.3%"；gemm_v0 kL0Size=128
  内部切块 → K=DV≤128 单 chunk、K>128 亦安全）：**适用** — 家族同构实证；op 数/task 128→79（-38%）、
  同步站点约 -40%；直线代码中 auto 中继 = 家族已验证形态；UB 峰值（liveness 复用后 ~128KB）< 196KB。
[#2] fixed-core grid + 显式 (CORE_NUM, NSLOT) L2 常驻中继 workspace + threads=2 + T.serial
  条纹任务循环 + drain-pipelined（参考: performance-antipatterns.md "launch core 数" 关注项 A
  （32768 任务按任务数 launch，命中反模式）；家族 fla_chunk_o iter3 -25%、全旅程 -50%；DESIGN §9.5
  阶梯①）：**适用** — 反模式直接命中；workspace 足迹 4.5GB → 20×NSLOT×~160KB ≈ 6.4MB 常驻 L2，
  消除 ~9GB 中继 HBM 流量；block 派发 32768 → 20。
[#3] bDK=64（任务数减半 16384、v/do/dv 重读 4×→2×、h/dh 中继总字节 -0）（参考: optimization-guide.md
  tile size 扩大；DESIGN §9.5 阶梯②）：**适用（依赖 [#2] 落地后的 UB 重新核算）** — UB 峰值
  Phase h/dh ~160KB 临界（需削峰验证）；L0C 3×[64,64]f32=48KB ✓；L1 160KB ✓。
[#4] threads=2（40 AIV 全唤醒，Vector 行切分）（参考: DESIGN §9.5 阶梯③；家族交付形态）：
  **适用** — 与 [#2] 同构（家族 fixed-core 即 threads=2）；单独在 32768-block 形态下为中性（家族实测），
  并入 [#2] 实施。
[#5] scale==1.0 折叠（trace 期 if，删除 Muls 1.0 冗余 op）：**适用** — 1 op/task，零风险；
  scale≠1.0 路径保留原乘法（L1 l1_scale 用例覆盖）。
[#6] host 分发 _FastLaunch 绕过（参考: 家族 R4，公共 JIT 分发 ~0.8ms/调用 → 快路径 163us）：
  **暂缓** — 当前 kernel 时间 ~19.5ms >> 分发 ~0.8ms，均值口径贡献 ~0.08ms；待 kernel 时间逼近
  1ms 量级再核算（家族地板核算方法已备）。
[#7] dq/dw gemm 组 M-concat 合并（[do_hi;dvn_hi]@h，gemm 9→6）：**不适用（本轮）** — 需 UB 行切片
  装配（T.copy 到 [128,DV] 切片目的，未验证 API）；收益仅 3 gemm + 1 relay/task；风险收益比差，
  列为后备。
[#8] 中继 bf16 化 / K-concat 单 gemm / hi-lo 行打包：**不适用** — 家族实证被 zN 分形布局或精度门禁
  阻断（fla_chunk_o final_summary §6），不重复尝试。
[#9] 每 tile op 微融合（mul_add_dst / axpy）：**不适用** — epilogue 门控链无 mul+add 相邻对
  （exp 链主导），无可融合形态。
[#10] msprof 深度画像（逐 pipe 占用率 / scalar stall / 对侧空闲率）：**按需** — 直觉瓶颈已由生成
  代码静态画像证实（中继足迹 + 站点数）；进入"深挖阶段"（直觉项证伪后）再强制执行。

[KERNEL-REUSE-PRECHECK]（[#1]/[#2]/[#3] 共同）：
candidate: 循环结构 / grid 结构 / tiling 参数重构
expanded_domain: 无新增（shape 域、dtype、接口签名均不变；wrapper assert 原样）
compatibility_dimensions: 接口签名不变（out_idx=[9,10,11,12] 绝对索引语义不变）；精度机制不变
  （fp16 hi/lo 3-gemm 拆分、exp(x·ln2)、LastRow 常量、dw 符号折叠）；L2/L1/L0C/UB 容量逐项重算
unknown_or_false: none
plan: repair-kernel-first（每轮 l0 快速精度门 + 终版 --level all）

## Part B [ORDER-PLAN] 实施顺序

[ORDER-PLAN] 实施顺序：
1. [#1] i_v 全展开直线化 — 前置依赖: 无 — 理由: 家族同构实证、风险最低、直接砍 38% op/站点；
   且其产物（直线任务体）是 [#2] fixed-core 任务循环的嵌入体，先行落地使 [#2] 的 A/B 干净。
2. [#2+#4] fixed-core + 显式 L2 中继 workspace + threads=2 + drain-pipelined — 前置依赖: [#1 ✅] —
   理由: 反模式关注项 A 的正解；-50% 家族最大杠杆；[#1] 的直线任务体直接作为 T.serial 循环体。
3. [#3] bDK=64 — 前置依赖: [#2 ✅] — 理由: 任务数减半在 fixed-core 形态下=每核循环次数减半；
   UB 削峰需在 [#2] 结构上重算（数量依赖：[#2] 改变 buffer 数量/布局）。
4. [#5] scale 折叠 — 前置依赖: 无（随 [#1] 顺带，属同一次 Edit 的相邻行内微调，单独记录）。
5. [#10] msprof 深挖 — 前置依赖: [#1]-[#3] 后仍有差距时强制执行（调优纪律：无画像不立优化项）。
6. [#6] host 分发核算 — 前置依赖: kernel 时间逼近 1ms 量级时（家族方法）。
