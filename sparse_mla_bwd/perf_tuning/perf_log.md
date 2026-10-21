# sparse_mla_bwd perf_tuning log (Stage 3)

目标：全接口 latency < 7680us @ B=1,S=2048,H=64,HKV=1,DQKV=576,DV=512,topk=512,bf16。
口径：`sparse_mla_bwd(..., delta=None)` 全接口（bwd k2 + postprocess k3 两段发射），
`tilelang.profiler.do_bench(fn, warmup=250, rep=100)` mean（主仓同款）；
输入 = 主仓口径生成（seed=0 randperm 因果采样）；(o,lse) = 测试内 torch fallback。
基线（Stage 3 iter-0 实测）：ascendc 79395us / pto 80004us——考核路径取 ascendc。

---

## Iteration 0 — 2026-10-14（基线画像 + 候选 A：ws_kv 按块暂存）

- bottleneck_type: **transfer（V 侧 MTE 微事务碎片化）+ sync（两遍结构放大）**
- optimization: |
    候选 A（已采纳）：ws_kv 按块暂存。`workspace_kv` 变 `[grid, NI+1, BN, DDT]`
    （NI 个逐块 stash 槽 + 1 个 scratch 槽）；pass-1 gather 按 (cid, i) 写入槽 i；
    **pass-2 的 V 侧逐行 gather 整段删除**（两遍结构下第二遍 gather 与第一遍数据
    逐位相同，纯冗余——msprof：AIC 86% 时间等 ws_kv flag，id8 28.4ms（pass-1）+
    id10 37.9ms（pass-2）），C 侧 pass-2 从 stash 槽 bulk MTE2 重读（C 侧空闲，
    cube 纯算仅 2.6ms/3.4%）。AUTO_CV_SYNC 语句配对以 2 条 scratch 槽 dummy 写
    补齐（4:4）；显式 delta 单遍路径保留原 gather（槽位索引化，2:2）。
    候选 B（批量 staging gather，fwd iter-2 3:3 形态）**本轮受阻未采纳**：
    (a) two-batch 3:3 需 17 个同步对——第 17 对的跨核 flag id 16 **溢出 16 上限**，
    混叠 flag 0（ws_dkv 生产者每块置位）→ 配对等待立即释放 → pass-2 读陈旧 ws_s
    （NI=1 被同值掩盖，NI=4 dQ 28%/dkv 99% 损坏，双后端同源）；
    (b) single-batch [64,576] 整 2D 写（保持 16 对）触发编译器 SEGV
    （OptimizeForTarget 内，"Vec loop 1 vs cube 4" 族警告后崩溃）。
    iter-1 路线：合并 ws_p/ws_dpb hi+lo 通道为单条 3D 语句（省 2 对 → 15 对）后
    再上 3:3。
- baseline_time: 全接口 ascendc 79395.3 us / pto 80004.2 us（k2 = 79371us，99.5%）
- candidate_time: 全接口 ascendc **56363.8 us** / pto **57210.7 us**（k2 = 55691us）
- improvement: **-29.1% / -28.5%**（远超 3% 噪声阈；xcheck 事件计时闭合 56260/57351）
- precision: **pass**（--level all "Test Passed!"：L0 9/9 + L1 15/15 双后端 kernel
  fwd 模式 + 覆盖门禁 24 PASS/0 MISS/0 N/A；valrange ratio-WARN 与 Stage-2 逐位
  一致。过程中 L0-4 显式 delta 腿曾 FAIL——首版无条件删了 pass-2 gather，而
  delta_internal=False 无 pass-1（单遍结构）；修复后与 Stage-2 数值一致。
  正确性证据：dQ 与改前逐位一致，dkv 在 stock 自身原子序噪声带内）
- adopted: **yes**（新基线 56364/57211）
- rollback_reason: -
- next_hint: |
    剩余画像（msprof，AIV 关键路径 54.9ms）：MTE3 22.6ms@4GB/s（pass-1 gather 微写
    ~30MB + **dkv 原子散射 60MB + ws_dkv 读回 60MB**）+ MTE2 17.4ms@8GB/s（KV 行读
    30MB + ws 读回 ~115MB）+ scalar 4.3ms + vec 1.8ms。iter-1 优先级：
    1) var_noatomic 差分定量原子散射真实成本（删 C 侧 G5/G6 ws_dkv 写 + V 侧
       scatter 段，配对 0:0）——决定走原子优化还是 §6.5 方案 2（私有 + 归约）；
    2) 候选 B 复活（ws_p/dpb 3D 合并腾 flag 预算 → 15 对 → fwd 3:3 形态）；
    3) P/dP 链 h-loop 向量化（fwd iter-3 同款，~64 行算子/块/遍）。
    平台约束（贯穿）：跨核 flag 对数 ≤16；V 侧 GM 微事务有效带宽 3-8GB/s
    （fwd 同病）；单条无内循环的大 2D 写可触发编译器 SEGV。
- consecutive_no_improvement: 0（首轮即 -29.1%）

### 事故与测量有效性记录（如实）

- 候选 B two-batch 版的损坏定位耗时较长：初判为同步配对语义问题，静态比对生成
  代码（flag 位置/槽位寻址/MTE 依赖屏障均正确）后由 **flag id 计数**（A=16 对
  通过、B=17 对损坏）+ NI=1 掩蔽效应闭环定位为硬件 id 上限——该约束已固化到
  模块 docstring 与 optimization_log [#4]。
- L0-4 显式 delta 腿首测 FAIL 为本会话编辑失误（变体 smoke 未覆盖
  delta_internal=False 分支）——全量门禁拦截，修复后复跑全量通过。教训已记：
  后续每轮变体验证必须含显式 delta 路径。
- 全程 NPU 空闲串行（每次计时前 npu-smi 确认；本轮无并行会话冲突、无误杀进程）。

## Iteration 1 — 2026-10-15（差分定量 + gather full+mini staging + scatter 载荷瘦身）

- bottleneck_type: **transfer（V 侧逐行串行发射路径——GM 标量读 + 微事务 + 逐段握手）**
- optimization: |
    诊断（差分矩阵，/tmp/opencode/b3i1/）：var_noatomic 删 C 侧 G5/G6+V 侧 scatter 段
    → dkv 散射路径边际成本 22776us（40.6% k2）；var_writescatter 仅原子→普通散写
    → **原子溢价仅 1367us（2.4%）**——§6.5 方案 2（私有+归约）否决，真瓶颈是每行
    3×PIPE_ALL + Indices.GetValue 同步 GM 标量读 + 5 段微读 + 5 次 MTE2→MTE3 握手
    的全串行发射（26112 行/AIV）。
    采纳（两项，flag 对数不变仍 16）：
    A. gather full+mini staging：逐行 GM 读入 kv_stage [HALF, DDT] 后整 2D MTE3
       一次写出 + 幂等末行重写配对 C 侧 tail 读（fwd iter-2 3:3 的 2:2 形态，
       [R,C] 整 2D UB→GM 写双后端探针通过；320 微写+256 握手/块 → 2 语句）。
    B. scatter 载荷瘦身：pos 改读 idx_ub（mask 链的 int32 UB 副本，与 Indices
       逐位相等，消除同步 GM 标量读）+ 4×512B 段微读合并为 1 次 [dim] 2KB 读；
       语句数与配对不变（段读/尾读 vs G5/G6）。
    受阻未采纳（iter-2 候选，平台约束已固化 docstring）：3:3 分批 scatter staging
    （需第 17 对 flag）；ws_p/dpb hi+lo 3D 合并（T.copy 3D-of-4D 区域静默截断为
    首 2D 平面，双后端）；G3/G4 五段融合省 flag 对（640 宽 stash 在 NI=4 触发
    编译器 SEGV，"Vec loop times not enough" 族）；fp32 P/dP 直传（gemm_v0 要求
    两操作数同 dtype）。
- baseline_time: 全接口 ascendc 56363.8 us / pto 57210.7（iter-0 采纳基线；本会话
  k2 事件计时复核 56145.7，闭合 +0.8%）
- candidate_time: 全接口 ascendc **32586.0 us** / pto **32823.9**（do_bench 口径；
  xcheck 32564.4/32877.7 闭合 <0.1%；k2 事件计时 31181.3/32983.6）
- improvement: **-42.2% / -42.6%**（远超 3% 噪声阈）
- precision: **pass**（--level all "Test Passed!" 44×PRECISION_PASS，L0 9/9 + L1 15/15
  阻塞级 + L2/Boundary 记录级；boundary nan_kv/dbound WARN 与 iter-0 逐位一致；
  变体 smoke 12/12 双后端双 delta 路径 vs stock sim 1.0000000）
- adopted: **yes**（新基线 32586/32824）
- rollback_reason: -
- consecutive_no_improvement: 0
- next_hint: |
    iter-1 后画像（msprof，k2=32624us）：AIV MTE2 12.2ms@10.9GB/s（ws_dkv 读回
    60MB 仍逐行 2 读/行 + KV 行读 30MB + ws_s/dp 26MB + ws_dq 29MB）+ MTE3 6.0ms
    @15.8GB/s + scalar 3.7ms + vec 1.8ms；AIC 等 stash flag id8 11.3ms + id10
    10.1ms，AIV 等 C 侧 flag ~13.2ms——**锁步往返（每块 6+ 次 flag round trip ×
    1632 块访问/AIV）成为新的结构性主项**。iter-2 优先级：
    1) ws_dkv 读回批量（需 3:3 → 先解决 −1 flag：候选=修 pass 侧 640-SEGV 或
       q_all 5 段化只动 q 侧 L1 布局）；
    2) 锁步结构（fwd iter-4 RC1-RC5 结论：AUTO 栈封死，需混合模式手动 flag——
       架构级，触发 [DESIGN_ERROR] 评估前先做完 1) 与 k3 行宽 16→32）；
    3) k3 postprocess 行宽（810us 池，rows_per_aiv 16→32 顺手续 UB 预算）。
- 事故与测量有效性记录（如实）：
    - var_gs 首版 smoke dq sim 0.44：融合回退时误删任务序言的
      acc_dqt_ub 清零（T.tile.fill）——dq 尾累积器带垃圾初值；全量 smoke 拦截，
      修复后 12/12。教训：结构性回退必须 diff 校对每个 alloc/fill 的存活。
    - var_merge（bf16 hi/lo 3D 合并）静默截断 bug 由 smoke dkv sim 0.999996/
      frac 36% 暴露（dQ 逐位一致 + dkv 半损坏 = lo 通道截断的特征签名）——
      生成源码定位为 copy_ub_to_gm 形状 (128,32) 只拷 4096/8192 元素。
    - 640 宽 stash 编译 SEGV 在 NI=4 复现两次（var_gfs/var_gf），NI=1 通过——
      与 iter-0 candidate-B SEGV 同族；已固化约束。

## Iteration 2 — 2026-10-15（P/dP_raw V-only stash 消灭 pass-2 重算 + k3 行宽 + 向量化）

- bottleneck_type: **sync（锁步 flag 往返——pass-2 s/dp 等待链 + AIC GEMM 重算发射）+ transfer（AIV 读带宽饱和）**
- optimization: |
    平台判定先行（本轮两条硬结论，源码+复现签名级）：
    P1. UB 可用上限 = 196352B（ASCEND_SHARED_MEM_SIZE），iter-1 采纳版已顶满——
        ws_dkv 批量读回 staging 在**任意 flag 预算**下不可行：[64,576] fp32（147456B）
        与 [32,576] fp32（73728B，恰等于活跃区间不相交的 kv_stage 槽位大小）均报
        "Memory allocation failed ... new memory available: 0"。根因：线性扫描规划器
        不跨 symbolic extent 的任务循环回收 pass-1 槽位。**iter-1 遗留的 3:3 设想
        即便解决第 17 对 flag 也会死在 UB**（追溯性证伪）。
    P2. combinecv 源码级新机制发现：跨核配对只遍历 cube 侧 workspace 名单
        （FetchWorkspaceName 只认名字含 "workspace" 的 buffer）→ **纯 V 侧 GM
        transit（命名避开 workspace*）零 flag 消耗、无语句数约束**；同核
        MTE3→MTE2 RAW / MTE2→MTE3 WAR 由通用 auto-sync pass 按缓冲名依赖分析
        覆盖（ascend_sync_insert.cc L1369-1395 RAW/WAR/WAW 检测）。
    采纳（三项，分阶段验证）：
    A. P/dP_raw V-only GM stash（P2 机制）：pass-1 把 post-exp P 与 raw dP
       （fp32）暂存 pst_p/pst_dp [grid, NI, head_kv, BN]；pass-2 直读短链
       （dP = P*(dP_raw−Δ)*sm_scale），**删除 pass-2 的 G1/G2 GEMM 重算**
       （Stage-2 R11 +39.5% 发射负债）+ pass-2 mask 链 + pass-2 ws_s/ws_dp 读；
       s/dp flag 对各 2→1（总 16→14）。dQ 与改前逐位一致（fp32 暂存保值）。
       单项 k2 32505.5→29726.6（−8.6%）。
    B. k3 postprocess 行宽 16→32（UB 110KB 预算内；输出逐位一致，nan 感知
       微验双后端 5 种行数）。
    C. mask/bias 施加向量化（fwd iter-3 同形，bcast_h [8,BN] axis=0 组广播 +
       NGRP 区域算子，bit-exact）。
    受阻未采纳（证据固化 docstring + baseline_iter2.json）：ws_dkv 全量/3:3/
       seg-only staging（P1）；C 侧按段/按行半拆分（L0C 累加器复用 / vid
       数据就绪不对称破坏按序配对）。
- baseline_time: 全接口 ascendc 32586.0 / pto 32823.9（iter-1 采纳基线；
  本会话 stock 复核 31898.1，漂移 −2.1%；k2 事件计时复核 32505.5 闭合）
- candidate_time: 全接口 ascendc **28541.0** / pto **30226.5**（do_bench；
  xcheck 28587.1/30338.3 闭合 <0.2%；k2 事件 28427.3/30151.7）
- improvement: **−12.4% / −7.9%**（vs 同会话 stock 复核亦 −10.5%，远超 3% 噪声阈）
- precision: **pass**（--level all "Test Passed!" 44×PRECISION_PASS，L0 9/9 +
  L1 15/15 阻塞级双后端；valrange-m/asym ratio-WARN 与 iter-1 记录逐位一致
  （0.9479/0.3011）；boundary WARN 与 stock 运行逐位一致；3 处 dkv max 在已记录
  原子序噪声带内；变体 smoke 24/24 双后端双 delta 路径 vs stock sim 1.0000000）
- adopted: **yes**（新基线 28541/30226）
- rollback_reason: -
- consecutive_no_improvement: 0
- next_hint: |
    iter-2 后画像（msprof，task 28549us）：AIC 等 flag 20.4ms/73%（其中 19.8ms 等
    V 侧 kv stash 生产 flag id7/id9）；AIV 等 flag 10.7ms/38%；AIV MTE2 10.8ms
    @12.2GB/s **读带宽饱和**（读量 ~147MB/AIV：KV 30 + ws_s/dp 13 + pst 13 +
    ws_dq 30 + ws_dkv 读回 60 + idx 1 ≈ 12ms 带宽下限）+ MTE3 5.6ms + vec 0.99ms
    （1.79→0.99，向量化+mask 链去重生效）+ scalar 2.7ms。**[DESIGN_ERROR] 候选
    判定已具备**：目标 7680us 与当前 28541us 差 3.72×，AUTO 栈内剩余项全部触墙
    （锁步 RC1-RC5 + flag≤16 + 语句数配对 + UB 规划器 + 读量带宽下限——当前
    数据流即使零同步开销也 >12ms）。Expert/混合模式预期：手动 flag 可重叠 AIC
    20.4ms 空闲 + 手动放置 73KB scatter staging → 约 19-21ms；达 7680us 还需
    数据流级读量削减（transit 融合 / C 侧 L0C 驻留累加（L0C 容量封顶）/ 精度
    门控的 bf16 transit）——均为架构级决策，提请 Orchestrator 裁决是否回退设计。
- 事故与测量有效性记录（如实）：
    - 首版 staging（[64,576] 全量）与 3:3 版（[32,576]×2，需第 17 对 flag 的
      幂等重写形态已写好）均在内存规划阶段失败（"available: 0"×2）——失败
      签名直接固化为本轮平台结论 P1；未消耗 NPU 时间于无效运行。
    - k3 微验首报 MISMATCH 为 NaN≠NaN 的比较假警报（两侧均 NaN）——改用
      nan 感知比较后 10/10 BIT-IDENTICAL；顺带证实 bf16 张量不支持 NPU
      位运算（比较需落 CPU）。
    - pto 侧 var_a→var_full k2 +841us（向量化在 pto 侧疑似微亏），但 pto 全接口
      30226.5 仍优于 iter-1 pto 基线 32823.9（−7.9%）；考核路径 ascendc 不受影响，
      如实记录。
    - 全程 NPU 空闲串行（每次计时前 npu-smi 确认；无并行会话冲突）。

## Iteration 3b — 2026-10-16（收获轮：榨取全部合法优化，5 项独立门控实验）

- bottleneck_type: **transfer+sync（V 侧 MTE 微事务发射/同步结构受限——字节级优化实测无效）**
- optimization: |
    本轮按 iter-3 画像优先级执行 5 项独立门控实验（每项独立备份/回滚），净采纳 1 项：
    A. 【采纳】P1b 探针删除（v1）：rev3 fix#4 的 produce 段尾 UB→UB 探针
       (cast_bf16→lse_2d)。生成源码证据（kernel_src.c rev3）：P1b 只 lower 成
       PIPE_V 屏障——cast_bf16 的访问是 V 读 vs MTE3 读（读读不冲突），lse_2d
       仅 V 管道使用（同队列有序），无任何跨管道保护需求；其 927×24KB≈22MB
       尾部 UB 流量 + 每迭代 V 屏障是纯开销。P1a（MTE3_MTE2）/P2/P3（MTE3_V）
       保留（静态论证各自的 WAR/RAW 保护对象均不可省）。
    B. 【回退·平台约束】scatter 微读 2→1 合并：撞 AUTO_CV_SYNC 语句配对
       （workspace_dkv: cube 2 条 G5/G6 写 vs vec 1 点）——per-row 语句数=C 侧
       写点数的硬约束，微读合并被封死。
    C. 【回退·平台约束】dq accumulate 双槽交替（s_ub/dp_ub ping-pong 解耦
       V_MTE2/MTE2_V 事件链）：trace-time 分支破坏 combinecv 循环归并
       （workspace_dq: cube 2 vs vec 3）。
    D. 【回退·负收益】dkv 贡献槽 fp16 档（env SPARSE_MLA_BWD_DKV_SLOT=fp16，
       三档参数化保留）：精度过门（L0 9/9，dkv ratio 0.9963-1.0000 vs bf16 档
       0.965-0.983——fp16 中间粒度有效性得证）；但性能 +1425us：dkv 读回的真
       瓶颈是每行微事务发射/同步结构（事务数不变），字节减半无效且 per-row
       2×UB cast 反增 V 串行开销。
    E. 【回退·负收益】dq 槽 bf16（env SPARSE_MLA_BWD_DQ_SLOT=bf16，保留）：
       精度过门（L0 9/9，dq ratio 0.9983-1.0000，Δ 恒等式后 dq 余量充分）；
       性能 −683us：bulk 读字节减半的收益 < per-chunk UB→UB cast + 2×PIPE_V
       屏障的 V 临界路径开销。**发现并修复平台 bug**：copy_ub_to_ub 对非整行
       region（srcStride != N）按线性拷贝（tail 段 91% 损坏、s0/s1 逐位正确的
       签名定位；R1 证据 2D 形态）——中转必须用整形容器（dqt_bf16）。
    F. 【不可行】KV gather 批量化：T.gather 仅支持 UB 内 gather（源码确认），
       GM 侧索引 gather 无原语；fwd 的 gather 同为逐行 MTE2 形态（无可复用的
       批量模式）。AUTO 栈内封死，未消耗 NPU 时间。
    G. 【回退·负收益】pst 复活（env SPARSE_MLA_BWD_PST=1，保留）：iter-2 P2
       机制（非 workspace 命名零 flag + 配对 16→14）在 rev3 下精度过门
       （bit-exact 转发，L0 9/9）；性能 +672us——画像预期的"省重读 1.16ms"
       在 stagger 结构下被同尺寸 pst 读回抵消，M3 写 13.2MB 净增，省下的仅
       vec 重算 0.3ms。rev2 两遍结构下 pst 的收益前提在 rev3 已不存在。
- baseline_time: 全接口 ascendc 28762.2 us（iter-3 画像 11:12 复测口径）
- candidate_time: 全接口 ascendc **27387.6 us** / pto **29196.5**（最终态；
  会话漂移 ±2%：早窗口 27339/27382，晚窗口 28060/28443——同窗口背靠背 A/B
  stock 28230.9 vs current 27387.6 = **−843.3us / −3.0%**；早窗口 vs 画像
  基线 −1423us / −4.9%；两窗口同向）
- improvement: **−4.8% / −3.5%**（vs 28762 口径 / pto 30226.5 iter-2 基线）
- precision: **pass**（--level all "Test Passed!"：L0 9/9 + L1 15/15 阻塞级
  双后端 + L2 9 正确拒绝 + boundary 记录级 WARN 与 stock 逐位一致（valrange-asym
  dq ratio 0.3011 与 iter-2 记录逐位一致）；dq vs stock 3 case bit-identical；
  dkv vs stock max diff 2.0 << stock 自身原子序噪声带 16.0（数值等价）；
  变体 L0 门禁：fp16 槽 9/9、bf16 dq 槽 9/9、pst 9/9 全部独立验证）
- adopted: **yes**（v1 = P1b 删除；三档 env 选项代码保留为已验证态，默认全关
  = v1 行为；最终文件 = v2 存档）
- rollback_reason: D/E/G 三项精度全过门但性能负收益按规则回退（默认关闭）；
  B/C 撞平台约束回退；F 无原语不可行
- consecutive_no_improvement: 0（本轮 −4.8%）
- next_hint: |
    本轮实证强化的结构性结论（供 Orchestrator 终局裁决）：
    1. **V 侧微事务/发射结构是地板**：fp16 槽（字节 −50%）与 bf16 dq 槽
       （bulk 字节 −50%）双双负收益——带宽不是瓶颈，每行/每 chunk 的发射、
       事件、屏障才是。AUTO 栈内字节级优化空间已穷尽。
    2. 平台约束新增两条（固化）：per-row workspace 语句数 = C 侧写点数
       （微读合并且配对冲突）；copy_ub_to_ub 非整行 region 线性化 bug
       （中转必须整形容器）。
    3. 剩余可想的都是架构级（Expert 手动 flag + stagger 2-token 消 id14、
       transit 融合、C 侧驻留）——画像 §6 已判 AUTO 栈封死（RC1-RC5）。
       与 7680us 目标差 3.6×，[DESIGN_ERROR] 证据链完整（iter-2 提请、
       iter-3 画像强化、本轮 5 项实验佐证）。
- 事故与测量有效性记录（如实）：
    - bf16 dq 槽首版 tail 中转走 cast_bf16[:, 0:64] region——L0 全 FAIL
      （ascendc sim 0.92 / pto 全绿的双后端分化签名 + 数值手术定位 tail 段
      91% bad、s0/s1 逐位一致）→ copy_ub_to_ub 线性化 bug 实锤，换 dqt_bf16
      整形容器修复后 9/9。pto/ascendc 分化源于 helper 实现差异。
    - 会话级漂移 ±2%（同代码 27339→28443）超过 3% 噪声阈的判定：全部采纳
      结论以同窗口背靠背 A/B 为准（stock↔current 交替，两窗口同向 −843/−1423）。
    - 全程 NPU 空闲串行（npu-smi 确认；无并行会话、无误杀进程）。

## Iteration 4 — 2026-10-16（2-token stagger 消 id14：首可行变体 +0.77% < 3% 止损，回滚）

- bottleneck_type: **sync（id14 零距离同 token 生产链依赖——实测为重叠型等待，非临界）**
- optimization: |
    纸面核算先行（baseline_iter4.json paper_audit 节，逐槽对照 DESIGN §8.3/§8.4）：
    结构 = V 段序 produce(pt)/P-chain(pt−1)/consume(pt−2) 三段错位（C consume 滞后
    produce 2 task）；混合槽深 ws_kv/s/dp=3（ws_kv 末读 G3(p) 距写 gather(p) 2 task，
    深度 2 剩同迭代跨核 WAR 不安全）、PD/DQ/DKV=2（flag 同 token + NI 迭代覆写距）；
    q/dO L1 单 task 槽 + G5/G5'/G6 GM 直读（3 parity 槽需 616KB > L1 512KB）；
    UB 段序同构不变（~184KB 分时）；L1 416KB（81%）；GM ~148MB（74% L2）；flag
    16 对不变（各 workspace 语句数逐条不变）。
    实现一次性通过全量门禁（L0 9/9 + L1 15/15 + L2 9 + boundary，双后端，WARN 与
    历史逐位一致）——纸面核算的槽位安全/依赖闭合全部兑现。
    同窗口背靠背 A/B（A,B,A,B 交替，do_bench warmup=250 rep=100）：iter4
    27009.5/26995.0 vs stock 27221.3/27201.2 → **−209us / −0.77%**（方向一致
    但 << 3% 噪声阈）。
    归因（数据反推，未再消耗 NPU 时间做画像）：① id14 的 2.26ms 在原画像中属
    "重叠等待"池（AIV busy 27.77ms 含 8.49ms 重叠 flag 等待）——消除重叠等待
    不缩短临界路径，2.26ms 是 profiling 上界而非可兑现收益；② 流水线头 id8
    （V gather→C G1 同迭代 0 距离，18.4ms）未被本轮 stagger 触及——整体节拍
    不变；③ 新增开销：G5 直读 +55MB/AIC C 侧 MTE1、P-chain 段 mask 链重算
    （idx 读 + 12 tile ops/块）、流水深度 +2 task 的 prologue/epilogue 尾部——
    三项与 ① 的真实可兑现部分基本对冲。
- baseline_time: 全接口 ascendc 27211.3 us（同窗口 stock 复核；iter3b 采纳基线
  27387.6，窗口漂移 −0.6%）
- candidate_time: 全接口 ascendc **27002.2 us**（两轮 27009.5/26995.0）
- improvement: **−0.77%（< 3% 噪声阈）**
- precision: **pass**（--level all "Test Passed!"：L0 9/9 + L1 15/15 阻塞级双后端
  + L2 9 正确拒绝 + boundary 记录级 WARN 与 stock 逐位一致——首版即过，含显式
  delta / oob / neg1 / repeat / varlen / kvgroup2 用例）
- adopted: **no（按止损规则回滚：stagger 首个可行变体收益 <3% 即停止）**
- rollback_reason: 收益 −0.77% 在噪声带内且远低于 3% 采纳阈；保留需付 L2 工作集
  113→148MB（56%→74%）与 C 侧 MTE1 +35% 的持续风险，不值得
- 存档: history_version/sparse_mla_bwd_perf_iter4_v1_stagger.py（可用实现，供后续
  Expert 手动 flag 轮参考其段结构与槽深核算）；sparse_mla_bwd.py 已恢复 iter3b
  采纳态（md5 76682f43）
- consecutive_no_improvement: 1
- next_hint: |
    1. 止损已触发（任务书规则：stagger 首个可行变体 <3% 即停）——AUTO 栈内
       stagger 加深路线关闭，且本轮实测把"消 id14=−2.26ms"的画像上界证伪为
       重叠型（可兑现 <0.8%），候选 3 剩余部分（Expert 手动 flag 消 id8/锁步、
       V 8.5ms 等待池结构化压缩）需架构级改动，属 [DESIGN_ERROR] 裁决域。
    2. 结构性证据链收口：iter-2 提请、iter-3 画像强化（读量下限 ~12ms +
       V 管道功 ~19ms > 目标 2.5×）、iter-3b 字节级穷尽（D/E/G 负收益、F 无
       原语）、iter-4 时序级穷尽（stagger 重叠等待证伪）——AUTO 栈四条路线
       （字节/事务/结构/时序）全部触墙，28.4ms vs 7680us 差 3.6× 的缺口只能由
       数据流级削减（transit 融合 / C 侧驻留 / 精度门控 bf16 transit）弥合。
- 事故与测量有效性记录（如实）:
    - A/B 两轮四点方向一致（A 两轮均优于 B 两轮），排除窗口漂移误判；全程 NPU
      空闲串行（npu-smi 确认）。
    - 首版即过全量门禁（无调试循环）——纸面核算→实现的转化率证据，槽深/边界
      guard/探针位置论证有效；该实现作为可用资产存档而非丢弃。
    - pto 侧性能未测（考核路径 ascendc 已判定；精度双后端已过）——如实记录。
