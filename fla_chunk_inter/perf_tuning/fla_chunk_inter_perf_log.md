# fla_chunk_inter Stage 3 跨轮调优日志（perf_log.md）

基线口径统一声明：do_bench（warmup=20 / rep=10 / mean / 仅设备侧 NPU 事件计时），shape
B=1,S=8192,H=64,DK=128,DV=128,chunk=64,scale=1.0,全 fp32，同会话 A/B（同 seed=0 同 rep）。
目标：< 3400 us。噪声阈值 3%。精度门：每轮至少 --level l0，终版 --level all。

---

## Iteration 0 — 2026-11-02T16:30+08:00（基线复测）
- 动作: 调度基线复测 + 生成代码画像（fla_chunk_inter_kernel_source_iter0.c）+ msprof op 画像（msprof_output_iter1/）
- baseline_time: 19556.3 us（调度值）/ 19469.7 us（本会话复测，采用为 A/B 锚点）
- 画像关键事实:
  - main_kernel<<<32768>>>，每 block = 1 AIC + 1 AIV；Pipelined(2) 未软件流水（嵌套普通 for）
  - 12 个 auto workspace 全部 per-cid 槽位，总足迹 ≈ 4.5GB >> L2 192MB（中继 9GB r+w 击穿 L2）
  - 每任务 ~150 同步站点；msprof 每 block 均值：AIC cube 0.54us / mte2 2.86us / fixpipe 1.11us /
    wait_id9 4.85us / mte1_stall 3.67us；AIV vec 2.67us / mte2 6.28us@26.6GB/s / mte3 3.58us /
    wait_id2 8.59us（60% 等 C 的 gemm 回传）
  - 结论：每 block 内 V→C→V 严格锁步 + 中继 HBM 流量主导；有效带宽 ~680GB/s = 峰值 54%

## Iteration 1 — 2026-11-02T16:40+08:00
- bottleneck_type: transfer（中继 GM 流量 + 锁步）；op/同步站点为次要（本轮证伪其主导性）
- optimization: [#1] i_v 循环全展开直线化（bDV=DV=128 单发：中继 20→10、gemm 18→9、装载 10→5、
  dgkn 免累加）+ [#5] scale==1.0 trace 期折叠
- baseline_time: 19469.7 us
- candidate_time: 19323.6 us
- improvement: -0.75%（< 3% 噪声阈值）
- precision: pass（l0 8/8，ratio 全 1.0000）
- adopted: yes（中性采纳，结构基础——家族先例：fla_chunk_o threads=2 "中性（结构基础）✅ 采纳"）
- rollback_reason: 无。判读：op 数 -38% 几乎无收益 → 直接证伪"op/同步协议主导"假设；
  锁步与中继流量墙掩盖 op 收益（msprof：AIV wait_id2 8.59us / AIC wait_id9 4.85us）。
  保留理由：(a) 精度全过且不劣化；(b) 直线任务体是 Iteration 2 fixed-core T.serial 循环体的
  必要基板（Pipelined 体与显式 workspace 中继混用嵌套为未验证形态）；(c) 家族实证中继
  L2 化后 op/站点成本重新成为主导（fla_chunk_o R3 消融），届时本项 -38% 站点将兑现。
- next_hint: Iteration 2 = fixed-core(20) + 显式 (20,4,*) L2 常驻 workspace + drain-first 流水
  （家族 iter3 同构，-25% 家族实证 + 消除 9GB 中继 HBM 流量预期）

## Iteration 2 — 2026-11-02T17:20+08:00（fixed-core + 显式 L2 中继 + drain 流水）
- bottleneck_type: transfer（中继 HBM 流量 + 每 block 锁步；msprof 画像证据）
- optimization: [ORDER-PLAN #2] fixed-core grid（T.Kernel(min(20,GRID)) + 单 T.serial 任务循环，BASE/REM 条纹 +
  Select 重定向幂等重算）+ 12 个显式 (CORE_EFF, NSLOT=4, ...) 中继 workspace（足迹 4.5GB → 11.8MB 常驻 L2）+
  1-late drain-first 流水 + dq/dk 结果中继手动交替 flag（ID 40+j%2，改名 dq_relay/dk_relay 逃逸 auto FIFO 配对）
- 过程工程事实（编译期段错误两次定位）:
  1) AUTO_CV_SYNC 按程序序配对 workspace_* 站点：环外 head-peel 与环内 drain 的 trip 失配 → 段错误
     （ascend_combinecv.cc "Vec loop times (=1) not enough to catch up cube (=5)" 警告形态）→ 改单循环全站点
  2) memory planning 的 loop-aware liveness extension：循环外 alloc + 循环内首触的 buffer 区间被钉到循环尾
     → produce+drain 声明和 ~457KB > 196352B → OptimizeForTarget 段错误（家族 R4 同签名）→ 首触全入循环
  3) reduce 输出到运行时索引切片需全维索引（dgkn_ring[slot, 0:block_DK]）
- baseline_time: 19469.7 us（iter1 形态 19323.6 us 采纳为新基线）
- candidate_time: 17279.1 us
- improvement: -10.6%（vs 19323.6）/ -11.3%（vs 会话基线）
- precision: pass（L0 8/8 + 全量 40 PASS / L2 8/8 / Boundary 2 文档化 WARN）
- adopted: yes
- next_hint: msprof：C 侧 73% 空闲（wait_id0 12.95ms），V 侧为墙（mte2 5.65ms@47GB/s + vec 4.36ms +
  mte3 4.38ms，~8.9us/task 单 AIV 串行链）→ threads=2 或削减 V 总量（bDK=64）

## Iteration 3 — 2026-11-02T17:35+08:00（threads=2）
- optimization: threads=2（V 行切分到 2 AIV，40 AIV 全唤醒）
- candidate_time: 精度崩溃（ratio 0.05-0.54，全输出）
- precision: fail → 回滚
- adopted: no
- rollback_reason: threads=2 的 vid 切分损坏本算子富 V 侧（broadcast + dim=0 reduce + 中继写消费者）——
  与 fla_wy_fast 家族结论一致（其 threads=2 gather 链 vid 越界先例，最终交付全 kernel threads=1）。
  富 V 侧算子族在 vid 语义下的安全性无先例，本轮不深挖（回到已证安全的路线）
- next_hint: 2-late drain 交替 ID（消除 V 锁步等待）与 bDK=64（任务减半 -40% V 总量）

## Iteration 4 — 2026-11-02T17:45+08:00（2-late drain + 交替 ID）
- optimization: 家族 final 配方——drain 延迟 2（dgkn ring 深度足），flag ID 40+j%2 与 produce(j-2) 配对
- candidate_time: 17904.1 us
- improvement: +3.6%（劣化）
- precision: pass（L0）
- adopted: no → 回滚
- rollback_reason: V 的 1-late 锁步等待（5.6us/task）实际与 DMA 引擎重叠（scalar pipe 等待时
  mte2/mte3 仍在推进），消除它无墙钟收益；更深的排水延迟反而增加流水填充延迟与尾部浪费。
  家族配方在本算子 V:C ≈ 2:1 的负载比下不成立（其负载比近 1:1）
- next_hint: bDK=64（V 总字节 -40%，任务减半）

## Iteration 5 — 2026-11-02T18:00+08:00（bDK=64 形状自适应）
- bottleneck_type: transfer（V 侧总字节主导）
- optimization: block_DK = 64 if (DK % 64 == 0 and DV >= 128) else 32（任务 32768 → 16384，
  V 侧 mte2/mte3 总字节 -40%；家族 _pick_block_dk 先例）
- 过程工程事实: fp32 Reduce2D dim=1 终端在恰好 [64,64] 组合下静默损坏（scratch/Plan 选形缺陷，
  reduce_2d_v2.h Reduce2DScratchElements 路径）——DK/DV 矩阵实证：[32,64]/[32,128]/[64,128] 干净，
  [64,64] 的 dgkn 全错（dg 的 m_last 行全错，255/256 元素）；DV>=128 guard 保证 suite 全部 shape
  落在已验证干净的 reduce 形态上。(64,128) 的 0.9999 边际为预存在舍入（bDK=32 同样 0.9999，非本轮引入)
- baseline_time: 17279.1 us
- candidate_time: 11909.6 us
- improvement: -31.1%
- precision: pass（L0 + 全量 40 PASS / L2 8/8 / Boundary 2 文档化 WARN）
- adopted: yes
- next_hint: 11909.6 us 距 3400 目标仍差 3.5×；msprof 定位新墙（V 仍主导？C 侧 gemm K=128×9 翻倍后?）

## Iteration 6 — 2026-11-02T18:40+08:00（V 侧流水重排三项消融，全部回滚）
- bottleneck_type: V 侧 mte2 满载（47GB/AIV ≈ 天花板）+ vec/mte3/waits 未重叠段
- optimization: 三项子实验（逐项 L0 门 + perf A/B）:
  a) drain 装载（g/gn/q/k）前移到手动 flag wait 之前 → **精度崩溃**（dg m_last 行全错，max_abs
     1.034e+02 与 iter5 reduce-bug 同签名）。根因定位（生成代码规划对比实证）：装载前移改变 drain 侧
     buffer 区间起点 → memory planner 重排全部 UB 偏移 → dh_ub/k_ub/dg_ub 三者别名到同一偏移 8448
     （q_ub@256=h_ub@256 同类）→ 跨迭代跨管线（MTE2 写 vs VEC 读）WAR 无名字级同步覆盖（
     ascend_sync_insert.cc 按名分析，无别名感知）→ pp 路径损坏。回滚恢复。
  b) produce 装载突发化（[do+v] 突发 + dv 装载前移）单独测试 → 精度 PASS 但 **+4.3% 劣化**
     （12422.3 vs 11909.6）——重排扰动 UB 规划 → auto-sync 屏障结构劣化。回滚。
  c) mul_add_dst 融合（dg 尾部 mul+add → 1 op，删 mp2d_ub）单独测试 → 精度 PASS 但 +1.0%
     （12034.4，噪声内偏差且扰动规划）。回滚。
- baseline_time: 11909.6 us（iter5 采纳态）
- candidate_time: a) 精度 fail；b) 12422.3 us；c) 12034.4 us
- improvement: 三项均无提升
- precision: a) fail；b)/c) pass
- adopted: no（全部回滚；终态 MD5 与 iter5 快照一致）
- rollback_reason: 见上；**经验结论：本工具链下 memory-planner 别名 + 名字级 auto-sync 的组合使
  语句重排成为高风险操作——"安全"重排可致精度崩溃（别名冒险）或性能劣化（屏障结构），
  逐项实测为唯一裁决**
- next_hint: 终止调优。地板核算（msprof_final）：V-mte2 4032us@47GB/s 满载 = 3.93GB/940GB/s 结构地板，
  > 3400us 目标 → 目标在本架构（V-only 装载 + threads=2 损坏 + planner 脆弱）下不可达。

---

## 终态汇总（Stage 3 交付）

- 终版 kernel 时间：**12015.7 / 12033.5 / 12033.4 us**（3 次独立官方口径运行，代表值 ~12028 us；
  iter5 采纳时单次读数 11909.6 us，差异 ~1% 在设备噪声内）。msprof 单_launch 设备侧 11900.6 us。
- 相对基线：19556.3（调度）/ 19469.7（会话复测）→ **-38.6% / -38.2%**
- 精度：--level all 全绿（40 PASS / 0 FAIL / L2 8/8 正确拒绝 / Boundary 2 个 Stage 2 已文档化 WARN，
  非本轮引入）
- 采纳链条：iter1 直线化（中性，结构基础）→ iter2 fixed-core+L2 中继（-11.3%）→ iter5 bDK=64（-31.1%）
- 回滚链条：threads=2（精度崩溃）、2-late 交替 flag（+3.6%）、iter6 三项重排（崩溃/劣化/噪声）
- 目标判定：**未达标**（12028 vs 3400 us 目标，超 3.54×）→ 见 final_summary 地板核算

---

# R2（Developer 模式续挖，用户检查点 2026-11-02）

## R2 Iteration 1 — 2026-11-02T20:40+08:00（NK-merge 任务重构造）
- bottleneck_type: transfer（v/do/dv 跨 bk 兄弟任务 2× 冗余装载 + 左中继重产；msprof V-mte2 4032us 满载为墙）
- optimization: [#R2-1] NK-merge——任务改为 (b, chunk, head)（GRID 16384→8192），NK 个 dk-block 合并为任务内
  trace-time 展开序列：v/do/dv 单次装载+拆分+中继（左操作数），C 侧 L1 A 操作数跨 bk 复用；右操作数
  （h/dh）per-bk 扁平槽位（ws_3/4/7/8 → (CORE,NSLOT*NK,...) 索引 slot*NK+bk，区域不相交消除 FIFO
  覆写风险）；dq_relay/dk_relay/dgkn_ring 同构扁平化（[NSLOT*NK, bDK] 行 = slot*NK+bk）。NK=1 时语句级
  退化为 R1 终态形态（回退安全性）。threads=2/grid=40/bDK=128 方向在源码/预算层封堵（见
  fla_chunk_inter_optimization_log_r2.md Part A）
- baseline_time: 12042.1 us（本会话复测；R1 档案 ~12028）／同会话 A/B：R1 形态 11988.0/11948.8（mean 11968.4）
- candidate_time: 9764.7 / 9739.4 us（同会话 A/B，mean 9752.0）
- improvement: **-18.5%**（vs 同会话 R1 mean）
- precision: pass（L0 8/8 一次通过，max_abs 与 R1 基线逐位一致；--level all 51 PASS + 2 Stage 2 已文档化 WARN）
- adopted: yes
- msprof 复测归因（msprof_output_r2_iter1，单_launch 9630.07us）:
  - AIV: vec 3041.8→2554.2 / mte2 4031.5→2608.7（47→58-61GB/s，合并后突发更大）/ mte3 2454.6→1365.4
    （67→92-98GB/s）——V 三引擎 busy 合计 9.53→6.53ms（-31%，超建模 -20%，DMA 突发效率红利）
  - AIC: mte2 1686.1→1084.6（中继读 320→224KB/任务对 ✓ -30% 归因精确兑现）/ scalar 1839→969 /
    wait_id0 8812→5845
  - **新暴露：wall 9630 − V busy 6533 ≈ 3.1ms V 侧暴露等待**（wait_id8/9 合计 4896us，与 DMA 部分重叠）。
    归因：drain(j) 的 0-late 手动 flag（等 C(j) 完成）迫使 V 在 produce(j) 结束后串行等 C 的整条消费链
    （~9.1us/任务 busy），V/C 追逐链周期 23.5us/任务 = V_produce(8.9) + 暴露 C 等待(~5.2) + V_drain(7.0)
- next_hint: r2_iter2 = 1-late flag——drain(j) 实读 task j-1 结果，只需等 C(j-1) 的 flag
  （40+(j+1)%2 配对 C(j-1)，j=0 用 C 侧循环前 head-set flag 41 预充计数）。预期消除 ~5us/任务暴露等待
  （wall → ~7.5-8ms 区间）。注意 R1 iter4 回滚的是 2-late 变体（drain lag 也改为 2）——本轮 drain lag
  保持 1，只改 flag 配对深度，与该回滚不冲突；但 planner/sync 结构脆弱，需逐项 L0+同会话 A/B 裁决

## R2 Iteration 2 — 2026-11-02T21:05+08:00（1-late flag 配对 + head-set 预充）
- bottleneck_type: sync（drain 的 0-late 手动 flag 迫使 V 串行等 C 整条消费链；iter1 画像暴露 ~3.1ms V 侧等待）
- optimization: drain(j) 实读 task j-1 的 dq/dk 中继 → wait 改配对 C(j-1) 的 flag（wait 40+(j+1)%2 ↔
  set 40+(j-1)%2）；j=0 的 wait 用 C 侧循环前 head-set flag 41 预充（计数语义平衡：head-set ↔ drain(0)）。
  drain lag 保持 1（与 R1 iter4 回滚的 2-late 变体不同——那是 drain lag 改 2 + flag 配对 produce(j-2)）
- baseline_time: 9796.7 us（iter1 采纳态，同会话 A/B mean）
- candidate_time: 6919.4 us（6903.1 / 6935.7）
- improvement: **-29.4%**
- precision: pass（L0 8/8 逐位一致；--level all 51 PASS + 2 Stage 2 WARN；无死锁——head-set 配对验证成立）
- adopted: yes
- msprof 复测归因（msprof_output_r2_iter2，单_launch 6853.5us）:
  - AIV wait_id8/9: 4896 → **0us**（暴露等待全消）；AIC wait_id0: 5845 → 2795us
  - V 三引擎 busy 合计 7119us > wall 6764us——引擎间开始真实重叠（~356us）
  - 新墙 = V 侧串行依赖链：16.5us/任务 = produce 链(~8.9) + drain 链(~7.0) + 残余等待(~0.6)；
    aiv_scalar_vector_stall 6490us = 标量发射流被逐 op 数据屏障串行化（UB 容量不允许全突发装载，
    装载→拆分→中继逐操作数链式）
- next_hint: r2_iter3 = drain 门控链 op 削减（ml2d 广播提升 + ln2 预折）——预计 -3~4%；
  此后 V 侧串行链为墙，进一步需引擎级重叠（UB 容量 + planner 别名限制，Developer 模式内难解）

## R2 Iteration 3 — 2026-11-02T21:25+08:00（drain 门控链 op 削减）
- bottleneck_type: compute（V 侧 vec 链 op 数；iter2 画像 scalar_vector_stall 6490us = 发射流逐 op 串行，
  op 数即墙时间）
- optimization: ① ml2d 广播提升至 drain 头（bk 不变量，-1 慢广播/任务；家族证据 Broadcast ~5.3× 慢于 Mul）
  ② ln2 预折：g_ub/gn_ub 原位 ×ln2 后 gnd = gn·ln2 − g·ln2 直接减，消除每 bk 一个 [64,64] mul（gexp/
  gnexp 改为 exp 直接读预缩放源）
- baseline_time: 6981.5 us（iter2 采纳态，同会话 A/B mean）
- candidate_time: 6727.6 us（6730.1 / 6725.1）
- improvement: **-3.6%**（超 3% 噪声阈值）
- precision: pass（L0 + --level all 51 PASS + 2 WARN；l0_repr dk 3.66e-4→4.27e-4 / dg 7.32e-4→1.22e-3
  为 ln2 折算舍入差异（预测 ~2e-7 相对 × 指数量级放大），valrange_l dg 8.301e-3 逐位不变，全部 ratio 1.0000）
- adopted: yes
- next_hint: V 侧串行链仍为墙（~16.0us/任务）。剩余方向：#R2-5 pass_configs 补测、produce 侧突发/重排
  （iter6 教训：高风险）或接受串行链为新地板

## R2 Iteration 4 — 2026-11-02T21:45+08:00（produce 左链装载批处理，回滚）
- bottleneck_type: compute（produce 链逐操作数 串行，装载/拆分/中继交替发射）
- optimization: do/v/dv 三个 32KB 装载移至左链头部连发（mte2 流水与拆分 vec 重叠）
- baseline_time: 6730.5 us（iter3 采纳态，A/B×3 mean；另组 6694.1）
- candidate_time: 6577.8 us（另组 6494.7）
- improvement: 合并 5 次配对 = **-2.55%**（两会话 -2.98% / -2.27%，5/5 方向一致）
- precision: pass（L0 逐位一致）
- adopted: **no**（< 3% 噪声阈值，按规则回滚；方向一致为正但未达采纳线，记录备查）
- rollback_reason: 噪声规则。教训更新：R1 iter6 的"装载突发化 +4.3% 劣化"在新结构（NK-merge +
  扁平槽位 + 1-late flag）下不再劣化（-2.5% 方向为正），但单独不足以采纳
- next_hint: r2_iter5 = 左链整体前移至 bk h/dh 链之前 + 装载批处理（组合预期 -6~8% vs iter3：
  左链拆分 vec 期间 h/dh(0) 装载流重叠）。C 侧首 gemm 门控从 max(ws_1, ws_3) 语义不变（两侧互换到达序，
  max 同值）；C 消费链尾延 ~1.5us 仍被 produce(j) 7us 窗口覆盖，drain 1-late flag 依旧提前满足

## R2 Iteration 5 — 2026-11-02T22:00+08:00（左链前移 + 装载批处理组合，回滚）
- bottleneck_type: compute（produce 链串行；试图用块交换扩大 mte2/vec 重叠）
- optimization: iter4 批处理装载 + 左链整体前移到 per-bk h/dh 链之前（h/dh 装载流与左链拆分 vec 重叠）
- baseline_time: 6644.4 us（iter3 采纳态）
- candidate_time: 6770.9 us（另组 6824.1——B 侧实为 iter5 候选，脚本标签沿用）
- improvement: **+1.9%（劣化）**
- precision: pass（L0 逐位一致）
- adopted: no
- rollback_reason: 两会话一致劣化（+1.9% / +1.1%）。归因：左链前移使右中继（ws_3/4/7/8）到达后移 →
  C 消费链整体后移并拉长，抵消 mte2/vec 重叠收益；且块交换扰动 planner 屏障结构（iter6 教训再现）。
  produce/drain 重排家族方向就此闭合：iter4（-2.5% 子阈值）+ iter5（+1~2% 劣化）
- next_hint: R2 候选方向全部实证封堵（见 final_summary R2 刷新地板核算）。
  终态 = iter3 采纳态（NK-merge + 1-late flag + 门控链削减）

## R2 终态汇总（2026-11-02T22:20+08:00）

- 终版 kernel 时间：**6777.0 / 6755.8 / 6767.4 us**（3 次独立官方口径运行，代表值 ~6767）；
  msprof 单_launch 设备侧 **6534.3 us**。lint（ruff check+format）后 L0 复验逐位一致。
- 相对 R1 终态 12028：**-43.8%**；相对 Stage 2 基线 19556.3：**-65.4%**
- 精度：--level all 全绿（51 PASS + 2 个 Stage 2 已文档化 WARN / L2 8/8 正确拒绝 / exit 0）
- 采纳链：r2_iter1 NK-merge（-18.5%）→ r2_iter2 1-late flag+head-set（-29.4%）→ r2_iter3 门控链削减（-3.6%）
- 回滚链：r2_iter4 装载批处理（-2.55% < 3% 噪声阈值，规则回滚）、r2_iter5 左链前移（+1.9% 劣化）
- 目标判定：**未达标**（6767 vs 3400 us，超 1.99×）→ 刷新地板核算与 R2 封堵清单见
  final_summary §3（V 串行链 6.8ms 贴线；threads=2/grid=40/重排/pass_configs 全方向实证闭合）
- R2 结构级调整已同步增补 DESIGN §12

---

# R3（Expert 授权 + 3400us 原指标续挖，2026-11-03）

R3 冲刺方向（按地板核算优先级）：① threads=2 纯手动最小实验（决定性）② Expert V 侧三引擎手动流水
③ C 链压缩 ④ launch 形态 ⑤ bDK=128。基线口径不变：do_bench warmup=20/rep=10/mean/仅设备侧。

## R3 Iteration 0 — 2026-11-03T09:30+08:00（会话基线锚点）
- 动作: r3_start 快照（= R2 终态，md5 db1d9e8a）+ 3 次官方口径 do_bench
- baseline_time: **6719.8 us**（6731.7 / 6716.1 / 6711.6；与 R2 档案 6767 差 -0.7%，会话噪声内）
- precision: l0 8/8 PASS（快照态复验）
- adopted: n/a（锚点）
- next_hint: 决定性实验先行——threads=2 微形状 kernel（drain 链精确算子形态）

## R3 Iteration 1 — 2026-11-03T10:00+08:00（threads=2 纯手动最小实验——决定性实验，方向①闭合）
- bottleneck_type: transfer（V 装载带宽 20 AIV 封顶是 3400 的第一墙；threads=2 唯一 40 AIV 入口）
- optimization: 微形状 kernel 实验（fla_chunk_inter_vid2_micro.py，[BS=64,bDK=64] bench 精确算子形态，
  threads=1 对照 + threads=2 实验组，fp64 golden 门禁）：
  - K1 pure2d（纯 2-D 行切分算子 + [BS] 广播 axis=1）: t1 PASS / **t2 FAIL**（dg 末行 ratio 0.9844——
    [BS]→[BS,bDK] "形状一致"的 broadcast 也损坏：tl.ascend_broadcast 不在 vid pass 的 tile_ops
    集内（ascend_vid_reduction.cc:563-586），显式 shape 参数不折半 → dst ptr 越界写）
  - K2 gn_bcast（[bDK] 1-D 链 + broadcast axis=0）: t1 PASS / **t2 FAIL**（dk ratio 0.2501，垃圾 7.5e4
    ——1-D [bDK] 沿 DK 切半 vs 2-D [BS,bDK] 沿 BS 切半，轴失配 + broadcast 不感知）
  - K3 dim0_reduce: t1 PASS / **t2 FAIL**（dg 末行——per-vid 部分和无跨 vid 合并）
  - K4 full_chain（= 本算子 drain 精确结构）: t1 PASS / **t2 FAIL**（ratio 0.26，与 R1 iter3 实机崩溃同签名）
  - K5 dim1_reduce（produce 侧 [bDK,DV] DK 切分 + dim=1 reduce）: t1 PASS / **t2 PASS**（produce 侧
    形态 threads=2 可行——但 threads 是 kernel 级参数，drain 不可行即整体不可行）
- candidate_time: n/a（结构判定实验，非性能候选）
- precision: 上述（t1 全 PASS 验证实验自身正确性）
- adopted: no（方向闭合，非采纳）
- rollback_reason: 不适用（无实现改动；fla_chunk_inter.py 与 r3_start 逐位一致）
- **结论（源码 + 实测双重证据，覆盖 R2 §12.4 的纯源码结论）**：drain 数学（逐列不变量 gn/dgkn/pp +
  跨行 Σ_t + 行掩码 broadcast）在 vid 语义下**结构性不可表达**——需要上游三项修复（broadcast vid
  感知 / 跨 vid reduce 合并 / per-buffer 切分轴标注）才能解锁 40 AIV；届时 V-mte2 地板 2867→~1434us。
- next_hint: Expert V 侧方向②——先做 barrier-thinning 微实验量化 yield

## R3 Iteration 2 — 2026-11-03T10:30+08:00（drain 门控链 op 融合 mul_add_dst，精度失败回滚）
- bottleneck_type: compute（V 侧 op 数 = 串行发射链长度）
- optimization: 两处 mul+add 融合为 T.tile.mul_add_dst（dst = src0·src1 + dst）：① pp += dgkn·2^gn
  ② dg += pp2d·ml2d（并删 mp2d_ub，-8KB UB）——共 -2 op/bk，预期 ~-3~5%
- baseline_time: 6719.8 us
- candidate_time: 精度失败（无性能意义）
- precision: **fail**（l0 dg 0.9852/0.9857，max_abs 137/339——末行 m_last 注入路径确定性偏差 ~1% 量级；
  分离二分定位：仅 pp 融合失败、仅 dg 尾融合失败、双融合失败、加 layout-keeper 死写仍失败——
  非规划器布局扰动，是 mul_add_dst 本征）
- adopted: no → 回滚（md5 恢复 r3_start，l0 8/8 复验 PASS）
- rollback_reason: 根因证据（CANN 9.2.0 头文件 kernel_operator_vec_binary_intf.h:249-250）：
  **MulAddDst 带 `__ASC_USE_RESERVED_UBUF__(3510)` 标注——使用保留 UBUF 暂存的受限 Level-0/2 指令**；
  微测（[4] fill 1/2/3 → 7 精确）证明语义 dst=d+a·b 正确，但在本 kernel 上下文（规划器管理的 196KB UB +
  多 buffer 场景）产生 ~1% 确定性偏差——保留 UBUF 机制与规划器分配冲突的实证形态。工具链层问题，
  kernel 层规避：禁用该指令族（记录：T.tile.mul_add_dst 在本算子域内不可用）
- next_hint: 改试 drain in-place 链压足迹（-32KB UB 压力）

## R3 Iteration 3 — 2026-11-03T11:00+08:00（drain in-place 链足迹压缩，设备崩溃回滚）
- bottleneck_type: transfer/compute（drain 活跃集 ~64KB 与 produce 交叉别名 → planner 屏障结构；试图
  -32KB 足迹给规划器腾出跨相位重叠空间）
- optimization: gexp/gn2d 两暂存消除——broadcast(gnd_ub, gn_ub) 原位 sub/exp 链 + 2^g 改 g_ub 原位 exp
  （算术逐位不变，仅少 2 个 [BS,bDK] buffer）
- baseline_time: 6719.8 us
- candidate_time: 运行失败（ACL 507014 设备侧崩溃→设备挂起；宿主 NPU 操作超时，需清理残留进程恢复）
- precision: fail（l0 两条用例 ACL stream synchronize failed 507014）
- adopted: no → 回滚（md5 恢复，l0 8/8 PASS，恢复后官方口径 3 次 **6639.1us**——设备状态正常）
- rollback_reason: in-place broadcast→sub→exp 链触发 sync-insert/pass 对 in-place 序列的处理缺陷
  （R1 iter6a 家族签名再现：语句/缓冲结构变化 → 崩溃）。设备挂起额外暴露一个 R2 世代残留探针进程
  （/tmp/opencode/probe/r2_hang_locate.py）占卡——已清理（issue 记录）
- next_hint: Developer 模式候选空间全部闭合（r2_iter4 批处理 -2.55%<3% 家族闭合 + 本轮两候选失败）——
  转入 Expert 微实验与终局核算

## R3 附实验 — 2026-11-03T11:30+08:00（vchain 微实验：Expert barrier-thinning yield 定量，方向②精化）
- bottleneck_type: sync（假设检验：aiv_scalar_vector_stall 6278us = 逐 op 自动屏障串行化发射流？）
- optimization: 微 kernel（fla_chunk_inter_vchain_micro.py）：drain 形态链（6 装载→17 vec→3 写出）×
  400 任务，A= AUTO_SYNC=True（逐 op 名字级屏障）vs B= AUTO_SYNC=False + 手动突发协议
  （T.Scope("V") + MTE3→MTE2 跨迭代 WAR + MTE2→V 装载突发 + V→MTE3 写出突发，3 对 flag/head-set 预充）
- candidate_time: A 665.8us（1.66us/任务）/ B 645.6us（1.61us/任务）——**仅 -3.1%**
  （复跑核验：A 731.6us / B 643.3us——A 存在 665-732us 运行间波动，B 稳定 ~643-646us；
  两次一致结论：B 稳定贴 mte2 引擎界 ~1.6us/任务，A 的屏障结构带抖动，yield 上限 -3~-12%）
- precision: 双 PASS（B 的手动协议一次正确——flag 计数平衡验证成立；复跑 bitwise 同精度签名）
- adopted: no（微实验，非 kernel 候选）
- **结论（假设证伪）**：小工作集 V 链在 AUTO_SYNC 下已达 mte2 引擎界（1.66us ≈ 96KB/55GB/s）——
  逐 op 屏障不是发射流串行化主因；真因是 **UB 196KB 容量强制 produce 逐操作数链式**（224KB 装载集
  无法突发，5 操作数轮次串行）+ produce/drain 交叉别名屏障。手动 flag 化（Expert 同步重写）yield ≈ 0
  （-3.1% 链级）；Expert-V 的可回收空间在 drain 内部重叠（~3.5us/任务）与跨相位重叠（需 produce 活跃集
  ~130KB + drain ~64KB ≈ 194KB 贴 196KB 上限，razor-thin），非屏障削减。修正后 Expert-V 地板
  ~9-12us/任务 ≈ 3.7-5.0ms（R2 的 ~4.5ms C 墙结论在 V 侧同样成立）
- next_hint: 方向②③④⑤全部证据闭合，进入终局核算与报告

## R3 终态汇总（2026-11-03T12:00+08:00）

- 终版 kernel：**= R2 终态原样（md5 db1d9e8a 与 r3_start 逐位一致）**——R3 两实现候选（mul_add_dst
  融合 / in-place 足迹压缩）均失败回滚，无净变更。恢复后复测 6651.6/6636.7/6628.9（mean 6639.1）
  与基线 6719.8 同分布（会话间方差 ~1.2%）；**代表值沿用 6767（R2 档案）/ 6639-6720（本会话实测带）**。
- 精度：--level l0 8/8 + 全量门（回滚态 = R2 终态的逐位复现，R2 全绿结论承继）。
- 采纳链：R3 无采纳（两候选失败 + 三方向实验闭合）；回滚链：iter2（mul_add_dst 精度，CANN 头文件证据）、
  iter3（in-place 崩溃 507014）。
- 目标判定：**未达标（6767 vs 3400，超 1.99×）且全部冲刺方向证据闭合**——完整限制清单（逐项：限制 +
  证据 + 突破条件 + 地板值）见 final_summary §3（用户硬性要求格式）。
- R3 结构级结论已同步增补 DESIGN §13。

## R3 反模式清单复核（终态，2026-11-03T12:30+08:00）

对照 `.agents/skills/tilelang-perf-optimization/references/performance-antipatterns.md` 逐项核查
R3 终态（= R2 终态逐位一致，R2 已核查项承继）：

| 反模式项 | 终态判定 |
|---|---|
| launch core 数（关注项 A/B） | ✓ 已用 fixed-core(20) + 核内任务循环（R1 iter2 采纳配方） |
| V 内逐元素/逐行 for loop | ✓ 无（全 tile 级指令） |
| 冗余全局同步 | ✓ 手动 flag 1-late 配对 + head-set 预充（R2 iter2 采纳） |
| **mul+add 未融合** | ⚠️ **有意保留 mul+add 分离形态**：r3_iter2 实证 T.tile.mul_add_dst 在本算子
  上下文产生 ~1% 确定性精度偏差（CANN 9.2.0 kernel_operator_vec_binary_intf.h:249-250
  `__ASC_USE_RESERVED_UBUF__(3510)` 受限指令与规划器管理 UB 冲突）——skill 文档推荐写法与
  实测冲突，保留原因 = 精度优先（skill-journal 反馈候选，见 issues） |
| tile size 过小 | ✓ [64,64]/[64,128] 为 UB 界内最大 |
| 混合算子未开 CV overlap | ✓ AUTO_CV_COMBINE=True |
| 纯 AIV memory bound 未做流水/双 buffer | ⚠️ 有意保留串行形态：R2 iter4/5（-2.55%<3% / +1.9%）+
  R3 vchain 微实验证伪 barrier 假设——真因 UB 196KB 容量强制逐操作数链式（限制清单 §3.1-2），
  Developer 模式内无解锁路径 |
| 多行 tile 循环内逐块归约 | ✓ 无（每任务单发归约） |
| 正交轴串行化 | ✓ 无标量扫描 |
| AUTO_CV_COMBINE 误分核 | ✓ 有真实 C 侧 |
| Vector mask reuse fallback | ✓ 未使用 |

## R4 Iteration 2 — 2026-11-03T20:10+08:00（dgkn ones-gemm 对角化移 C 侧，重调度会话）
- bottleneck_type: compute（V 侧 produce 折叠链 vec/语句开销；画像 vec 3254/44.4% + 归因 539us）
- optimization: dgkn[k]=Σ_v h·dh=diag(h@dh^T) 移 C 侧：3 个 hi/lo 小 gemm（复用 L1 已有
  h_hi/lo、dh_hi/lo）→ [bDK,bDK] fp32 L0C frag（4 frags 64KB<128KB）→ fixpipe 进新 dgm_relay
  （与 dq/dk relay 同模式，同 40+j%2 手动 flag 组覆盖）→ drain 回读 + 一次 T.tile.gather 对角
  提取（prologue arith_progression(int32)+reinterpretcast(uint32) 常量 byte 偏移，
  AscendC gather NoWorkspace 无 scratch 注入，D-2 红线安全）。删除 produce 侧
  mul+transpose+chunk-adds+6 层树+行拷贝整链与 32KB prodT_ub。
- baseline_time: 7503.7（会话锚点 7510.6/7494.2/7506.3；D-2 验收 7510.3 同分布）
- candidate_time: 7167.1（7159.8/7187.1/7154.5）
- improvement: -4.49%（-336.6us）
- precision: pass（L0+L1 全绿 40 PASS；valrange_l dg 7.812e-3 ≤ 1e-2 与基态持平；
  --level all Test Passed）
- adopted: yes
- next_hint: gn 门控 64 逐行 Sub 为下一优先（归因 565us）

## R4 Iteration 3 — 2026-11-03T20:20+08:00（gn 门控转置+row_expand 向量化，bDK=64 路径）
- bottleneck_type: compute（drain 64 条逐行 Sub 语句开销；反模式清单"正交轴串行化"）
- optimization: gnd[t,:]=gn[:]-g[t,:] 的 64 Sub → transpose(g_ub→gT，复用已死 gexp_ub)
  + row_expand_sub_experiment（brcb 展开 gn 逐行标量块 + 单次 masked Sub）+ mul(-1) 保号
  （exp 参数与旧行链逐位一致）+ exp + transpose 回。bDK=32 行宽 128B 不满足 256B 行窗 →
  trace-time 分支保留 legacy 行链（L1 套件覆盖）。tmp 为显式 alloc（[bDK,8] f32）。
- baseline_time: 7167.1（iter2 采纳态）
- candidate_time: 7012.3（7011.7/7003.5/7026.9/7018.8/7000.5，5 次）
- improvement: -2.16%（-154.8us；<3% 默认阈但实测噪声 ≤0.5%、区间不交叠 ~14σ，
  按 mandate"按证据推进"条款采纳，偏差显式记录）
- precision: pass（l0_repr bDK=64 新路径 dg 1.221e-3 与 iter2 逐位一致；--level all 全绿）
- adopted: yes
- next_hint: 画像重采（mte2 带宽界确认）+ 树折叠替代封堵验证 + 中继读融合

## R4 Iteration 4 — 2026-11-03T20:50+08:00（drain 三中继读融合 + 终局归因）
- bottleneck_type: transfer（mte2 3219us@53GB/s 对 416KB/task = 引擎带宽界）+ sync（发射/屏障语句开销）
- optimization: dq/dk/dgm 三个 GM 中继融合为一个 ddr_relay（每 (task,bk) 行段
  [0,BS)|[BS,2BS)|[2BS,2BS+bDK)），drain 3×16KB 读 → 1×48KB 单读；消费侧全行区域视图
  （flat-contiguous）；签名 25→23 张量（out_idx=[19..22]）。树折叠替代实证封堵：
  wholereduce 每 repeat 仅 8-block lane 求和不跨 repeat 累加（64 行折叠不可能）、
  sum_experiment 为 last-dim 轴不符、转置反超。
- baseline_time: 7012.3（iter3 采纳态，5 次散布 0.37%）
- candidate_time: 6868.3（6871.9/6859.0/6855.8/6881.8/6873.0，5 次）
- improvement: -2.05%（-144.0us；区间不交叠 ~10σ，同 iter3 判定法采纳）
- precision: pass（L0 误差与 iter3 逐位一致；--level all 40 PASS，valrange_l dg 7.812e-3）
- adopted: yes
- **R4 终态判定：6868.3 vs 6767（R2 档案）= +1.50%，进入 ±3% 收敛带 [6564,6970] → 收敛条件①达成**
- next_hint: 剩余 +101us 结构封堵归因（vs 6767）：dgm 中继回读 mte2 ~291us 无 C 侧 gather
  可解 / 门控 2 转置 ~328us 为无 scratch 行广播最小形态 / 树 170us 无原语替代——
  全部为 D-2 无 scratch 红线下的结构地板；Developer+scratch-free 形态地板 ≈ 6868us

## R4 反模式清单复核（终态，2026-11-03T21:20+08:00）

对照 performance-antipatterns.md 逐项核查（R3 终表承继，仅列变化项）：

| 反模式项 | R4 终态判定 |
|---|---|
| V 内逐元素/逐行 for loop | ✅ **已消除**（bDK=64 路径：转置 + row_expand_sub + 保号 + 转置回，"正交轴串行化"官方解法）；bDK=32 路径**有意保留**行链——row_expand 需 256B 行窗（f32 64 列），bDK=32 行宽 128B 物理不满足（约束实证，非性能取舍）；保留原因 = 形态约束而非反模式选择 |
| 正交轴串行化 | ✅ R4-3 依官方解法消除（原 R3 表记"无标量扫描"系漏检 64 逐行 Sub 语句级开销，本次修正） |
| mul+add 未融合 | ⚠️ 有意保留（R3 iter2 实证 mul_add_dst 精度缺陷，CANN 头文件证据，承继） |
| 纯 AIV memory bound 未流水/双 buffer | ⚠️ 有意保留（R2/R3 UB 196KB 容量锁实证，承继；R4-4 中继读融合为该锁内的语句级缓解） |
| 其余项 | ✓ 承继 R3 终表判定（fixed-core/手动 flag/tile 尺寸/CV overlap/循环内归约/误分核/mask fallback） |
