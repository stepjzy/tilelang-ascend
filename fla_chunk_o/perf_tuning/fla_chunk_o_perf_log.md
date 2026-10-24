# perf_log — fla_chunk_o（Stage 3 性能调优跨轮日志）

> 设备：Ascend 910B3（64GB HBM，20 AI Cube + 40 AI Vector cores）/ CANN 9.2.0 /
> tilelang editable @ 2f22c6ac。计时口径 = test_fla_chunk_o.py --level perf
> （NPU 事件，warmup=10，rep=10，均值，仅设备侧，与上游 do_bench 一致）。
> bench shape：B=1, S=8192, H=64, DK=128, DV=128（GRID = 8192 任务）。
> 迭代期快速 A/B 用 perf_tuning/fla_chunk_o_bench_iter.py（同口径 + l0_repr
> 快速精度门）；每轮采纳前跑 --level all 全量门禁。

## Iteration 1 — 2026-10-22（threads=2 + 3D 视图修复）
- bottleneck_type: sync/latency（生成代码证据：Pipelined 未软件流水、4 个 GM
  workspace 中继 +152KB/块、~10 次跨核 flag/块；带宽仅 25% 利用 → 非带宽瓶颈）
- optimization: threads=1→2（DESIGN R9）。直接切换精度 FAIL（ratio=0.2733, NaN）
  —— 根因：vid-reduction 把子块行偏移加到 GM 维 (gm_dims - ub_dims)，4D bshd
  切片落到 H 维（读错 head）。修复：Q/G/A/O 改为零拷贝 3D 视图 [B,S,H*D]，
  使 S 行维落在正确位置（flash_attn_bshd_developer 先例验证 threads=2 机制本身）。
- baseline_time: 3.285 ms（稳态 3.20）
- candidate_time: 3.264 ms（稳态 3.19）
- improvement: -0.6%（噪声内）
- precision: pass（ratio=1.0000）
- adopted: yes（作为后续结构的基础；单独看性能中性——V 侧减半但 AIC 侧不变，
  发现双侧锁步迹象）
- rollback_reason: n/a
- next_hint: 瓶颈在串行延迟链而非 V 吞吐；消除未流水的 i_k 循环。

## Iteration 2 — i_k 循环展开为直线全 DK
- bottleneck_type: sync/latency
- optimization: 去掉 Pipelined 循环：q/g 单次 [64,DK] 大 DMA（行 256B/512B）、
  h 连续 32KB 单 DMA、V 链一趟、中继 4→2 次（写+读各半）、gemm 4×K=64 →
  2×K=128、跨核 flag 减半。
- baseline_time: 3.264 ms
- candidate_time: 2.993 ms（稳态 2.90）
- improvement: -8.3%
- precision: pass
- adopted: yes
- next_hint: 中继/flag 仍多；probe 分解显示 skeleton（纯 DMA+中继+同步）占 75%。

## Probe 分解（iteration 2 结构上）
- skeleton（loads+relays+o 链，无向量计算无 gemm）：2.245 ms
- +向量计算：2.359 ms（+0.11 —— 向量计算便宜）
- +gemm/FIX（完整）：2.993 ms（+0.63）
- 结论：**延迟/同步型瓶颈**（每块 5+ 次串行暴露的 MTE2 latency）。

## Iteration 3 — Fixed-Core 重构（antipatterns 关注项 A）
- bottleneck_type: sync/latency（8192 块按任务数 launch 的每块调度开销 +
  auto 注入 ~768MiB workspace）
- optimization: grid=20（cube 核数）+ 核内 T.serial 任务循环 + 显式小 workspace
  （workspace_idx，[20,2,...] ≈ 2.9MiB 常驻 L2）+ Msk hoist。
- baseline_time: 2.993 ms
- candidate_time: 2.245 ms
- improvement: -25.0%
- precision: pass
- adopted: yes
- next_hint: 生成代码显示循环内 if 守卫被 AUTO_SYNC 用 PipeBarrier<PIPE_ALL>
  包围（每迭代 ~8 次全流水排空）。

## Iteration 4 — 消除循环内 if（branch-free Select + 跨步任务映射）
- bottleneck_type: sync（PipeBarrier<PIPE_ALL> × IfThenElse）
- optimization: 跨步映射 tid = cid + t*CORE_EFF（Select 回退目标 cid 恒在界内）
  + Select 替代全部 if 守卫（PT 内零 IfThenElse）。
- baseline_time: 2.245 ms
- candidate_time: 2.126 ms
- improvement: -5.3%
- precision: pass
- adopted: yes
- next_hint: msprof 显示两核 ~80% 时间互相等 flag（锁步）；早期 a_mask 中继 +
  loads 预取。

## Iteration 5 — 早期 a_mask 中继 + intra-gemm-先行 + loads 预取
- bottleneck_type: sync（V 的 gate 链与 C 的 gemm 完全串行）
- optimization: V 侧 mask 链后立即中继 a_mask（C 的 intra gemm 只需它 + GM v，
  与 V 的 gate 链重叠）；A/Q/G loads 移到 produce 尾部（MTE2 latency 藏于 drain）。
- baseline_time: 2.126 ms
- candidate_time: 1.859 ms
- improvement: -12.6%
- precision: pass
- adopted: yes
- next_hint: msprof 复测：V 仍 43% 等 FIX flag —— V 的下一 produce 被 drain
  （等 AIC 的 FIX）阻塞，锁步仍在。

## Iteration 6 — 手动 flag + 2 任务延迟 drain（打破锁步）
- bottleneck_type: sync（AUTO_CV_SYNC 的 flag 按出现序 FIFO 配对 → V 的读必须
  等 C 的同序写 → 每任务 V_produce + C_gemm + V_drain 全串行）
- optimization: o 中继改名 o_relay（脱离自动 flag）+ 手动 T.set_cross_flag/
  T.wait_cross_flag（ID 12/13 按奇偶配对）+ drain 延迟 2 任务（NSLOT=4）+
  2 个 peeled 填充迭代（保 flag 位点数平衡 + 避免 t=0 死锁）。中途验证：
  纯自动 flag 的 2-late drain 死锁（FIFO 配对实证）；wait_cross_flag 的 pipe
  参数必须是 "V"（ResourceForPipe 只认 V/M/MTE1/FIX）。
- baseline_time: 1.859 ms
- candidate_time: 1.691 ms
- improvement: -9.0%
- precision: pass
- adopted: yes
- next_hint: AIC 成新瓶颈（scalar 53%、mte2 29%）。

## Iteration 7 — o 中继 bf16 化（fixpipe 转换）
- bottleneck_type: transfer（AIC FIX 32KB fp32 + V drain 读 16KB/vid + cast）
- optimization: o_relay 改 bf16 —— Cube 侧 fixpipe 直接完成 fp32→bf16 转换
  （fla_wy_fast 记录的 "L0C→GM(bf16) fixpipe 损坏" 适用于直存输出张量，
  本中继路径实测精确）；V drain 免 cast、读减半。
- baseline_time: 1.691 ms
- candidate_time: 1.670 ms
- improvement: -1.2%（噪声边缘；FIX 非当时瓶颈）
- precision: pass（含 --level all 全量 11/11 PASS）
- adopted: yes（减少 AIC/V 两侧字节数，为后续让路）
- next_hint: AIC 的 ws 读延迟暴露。

## Iteration 8 — ws1/ws2 读批量化
- bottleneck_type: sync（hi/lo 中继读被 hi-gemm 隔开，各自 MTE2 latency 暴露）
- optimization: 两个 inter 中继读背靠背发出（lo flag 仅落后 hi flag 一个
  MTE3 burst），latency 重叠。
- baseline_time: 1.670 ms
- candidate_time: 1.603 ms（同会话复测 1.646-1.662，稳态 ~1.58-1.59）
- improvement: -4.0%
- precision: pass
- adopted: yes
- next_hint: 手动 flag ID 3/4 与自动 flag 0..8 潜在冲突 → 移到 12/13（安全性
  修正，性能不变）。

## Iteration 9 — L1 ping-pong + h/v 预取（尝试，回滚）
- optimization: strip-mine 循环 + L1 操作数双缓冲 + h/v 预取（尝试把 0.8us
  h/v MTE2 传输移出 AIC 串行链）。两个变体（尾置预取/中置预取）分别
  1.621 / 1.685 ms —— 无收益甚至劣化（工具链在预取与 gemm 间插入的保守
  屏障抵消重排序收益）。
- adopted: no（回滚到 iteration 8 + flag ID 安全修正）
- rollback_reason: 无收益（-0/+5%），复杂度不值得

## 迭代 10（最终安全修正）
- manual flag ID 3/4 → 12/13（自动 flag 占 0..8：3 ws × 3 位点 = 9 对；
  消除潜在 ID 冲突）。性能不变（1.646/1.648 ms 两连测）。

## 最终状态（官方口径 test_fla_chunk_o.py --level perf）
- **1.638–1.658 ms**（4 次独立官方运行：1.638 / 1.646 / 1.648 / 1.658，
  ±2% 设备噪声；代表值 1.65 ms）（per-rep [2.294, 1.506, 1.542, 1.546, 1.637, 1.562, 1.484,
  1.577, 1.586, 1.644]；稳态 ~1.55-1.59）
- 基线 3.285 ms → **-50.2%**（2.005×）
- 目标 < 1.320 ms：**未达标**（超 24%）
- 精度：--level all 全量 PASS（L0 2/2 + L1 9/9 PRECISION_PASS，L2 10/10
  正确拒绝，Boundary 3 PASS + 2 个 Stage 2 已记录的预期 WARN）

## 剩余瓶颈归因（最终 msprof + 分析）
1. AIC 串行链 ~3.9us/任务：9 个必备 op（h、v、ws3 读、intra gemm、ws1 读、
   ws2 读、hi gemm、lo gemm、FIX）× 每 op 的工具链同步开销（MTE2_M/M_FIX
   握手 + 屏障 + flag），scalar 活动占 53%。
2. AUTO_CV_SYNC flag FIFO 配对限制（已用手动 flag 绕过 o 路径；hi/lo/a 三对
   自动 flag 仍在同迭代内配对，但由于 V 领先 ~2 任务，等待 ≈0）。
3. UB 容量（184KB/vid，96% 占用）阻塞 2-chunk 任务批处理（本可将每 token
   的 flag/循环开销减半，估算可达 ~1.25ms）。
4. HBM 带宽 640GB/s（50% 利用）—— 非当前瓶颈，但 1320us 目标隐含 745GB/s
   （59%），留给同步开销的余量极小。

---

# ROUND 2（2026-10-23，用户决策"继续调优"，预算 10 轮，噪声阈值 3%）

> R2 起点：1.690 ms（官方 --level perf，与 R1 终态 1.638-1.658 一致，噪声内）。
> 主线：UB 腾挪→2-chunk 批处理；hi/lo/a 手动 flag；msprof 新线索。
> R2 全程以生成代码审查 + 探针 + msprof 三重证据驱动（详见
> fla_chunk_o_optimization_log_r2.md）。

## R2 Iteration 1 — C 侧 fixpipe 直写 O（消灭 o_relay + V drain 全链）
- bottleneck_type: transfer（V drain MTE 流量 32KB/任务 + 2 组队列标记 + 6 条 V 地址链）
- optimization: FIX 指令 copy_l0c_to_gm 直接写 strided O 行（realDstN=H*DV 由
  compute_strideN 推导，src/op/ascend.cc 实证）；手动 flag 12/13 转为纯节流；
  o_relay/o_half/drain 标记全消。**工具链事实修正**：fla_wy_fast 记录的
  "L0C→GM(bf16) 直存损坏"不在本路径复现（fp32→bf16 fixpipe 转换精确）。
- baseline_time: 1.690 ms
- candidate_time: 1.652 ms（稳态 1.582 持平）
- improvement: -2.2%（<3% 噪声阈值，但省的全是 V 侧非瓶颈工作 + 8KB/vid UB +
  o_relay GM 环路，结构价值）
- precision: pass（l0_repr ratio=1.0000）
- adopted: yes（结构基础；生成代码证据 realDstN=8192 正确）
- next_hint: Select 地址链（AIC 11 条/迭代未 CSE）。

## R2 Iteration 2 — Select-free 主循环 + Select 尾部剥离
- optimization: t ∈ [2,407] 主循环纯算术地址（406/410 迭代零 if-else 链），
  尾部 4 迭代保留 Select；peel 恒合法去 Select。
- baseline_time: 1.652 → candidate_time: 1.670/1.607（复测 1.670/1.590）
- improvement: 0（持平——证明 condval 链非主要开销；AIC scalar 53% 主要是
  flag/MTE2 队列等待与 DMA 描述符）
- precision: pass
- adopted: yes（严格更少 scalar 工作；为批处理提供结构）
- rollback_reason: n/a

## R2 Iteration 3 — 2-chunk 任务批处理（R1 主线 1，UB 经 R2-1 腾挪后重启）
- optimization: 每迭代消费任务对 (2k, 2k+1)：staging 双份（+28KB/vid，恰入
  R2-1 腾出的余量）；fp32 链顺序复用；节流简化为单 flag 12（每批 1 set/1 wait，
  全局严格交替）；NSLOT=4 配对批 {0,1}/{2,3}。
- baseline_time: 1.670 → candidate_time: 1.675/1.601
- improvement: 0（持平——生成代码显示工具链在 A/B 块间插入同名 WAR 的
  M_MTE2 串行，MTE2 队列与 M 流水仍严格轮替）
- precision: pass
- adopted: yes（作为 ping-pong 的结构载体）
- 关键发现：HasDataDependency（ascend_sync_insert.cc）按 buffer_name 判
  WAR → 按名 ping-pong 可解除串行（R1 ping-pong 失败根因推断为同名单缓冲）。

## R2 Iteration 4 — L1 操作数按名 ping-pong（_a/_b 独立 buffer）
- optimization: A_l1/V_l1/H_l1/qg_*_l1 各拆 _a/_b（L1 176KB/512KB）；B 块
  h 装载上移至 A 块 gemm 组后（无 WAR 等待，实测生成代码确认 M_MTE2 消失）。
- baseline_time: 1.675 → candidate_time: 1.667/1.593
- improvement: 0（持平——MTE2/M 串行解除了，但暴露真正的墙：见下）
- precision: pass
- adopted: yes
- next_hint: msprof 必须做——四个结构变体全部持平说明真实瓶颈未触及。

## 【R2 关键 msprof 证据】（msprof_output_r2，1437us 口径）
- AIC mte2 = 1150us（72%）/任务 2.8us（88KB → 31.5GB/s/核）；scalar 30%；
  cube 11%；fix 13%。AIV vec 18%、mte2 44%、scalar 34%。
- 探针（v 装载换连续读）：1.667→1.487（-180us）→ **strided v 装载
  （64×256B、行距 16KB）是 AIC 关键路径大头**。

## R2 Iteration 5 — v 中继经 V 侧 + V 链就地压缩（R2-3+R2-6 合并）
- optimization: (a) gate 链就地化（tile.mul/exp/sub dst==src，消灭
  gL/gexp/qg/qg_lo_f32 4 buffer，-64KB/vid——文档明示可别名，实测逐位精确）；
  (b) AIV 用其余量读 strided v（3D 视图防 vid 落维）→ 连续中继 ws4 →
  AIC 只读连续 16KB（88KB 全连续，~48GB/s）。
- baseline_time: 1.667 → candidate_time: 1.563/1.499（in-body v 装载版）
- improvement: **-6.2%（-104us）**
- precision: pass（l0_repr ratio=1.0000；就地链数值等价）
- adopted: yes
- 中途失败：lo 中继源误用 fp32 q_ub 直写 GM bf16 → 编译错（GM↔UB
  cast-on-copy 已知缺陷）→ 经 qg_lo_ub UB 内 cast 修复。
- v 预取变体（藏延迟于节流等待）：1.579/1.509 持平（ws4 flag 延迟非问题），
  保留预取版（结构更优）。

## R2 Iteration 6 — O 写回迁 V 侧（R2-7，尝试，回滚）
- optimization: C FIX→o_relay 连续 + 手动 12/13 + V 排水写 strided O
  （R1 已证模式，目标省 C 侧 strided FIX ~0.2us/任务）。
- baseline_time: 1.579 → candidate_time: **1.759/1.684（+180us 劣化）**
- improvement: -11% 劣化
- precision: pass（结构正确，性能劣化）
- adopted: no（回滚）
- rollback_reason: V 排水链（2×flag 等待 + 2×MTE2 读 + 2×strided MTE3 写 +
  标记对）加在已 61% scalar 的 V 上，代价 ≫ C 侧节省。**R1"直写 vs 中继
  中性"结论修正：V 侧有负载（v 中继）时直写 O 严格更优。**

## R2 Iteration 7 — o_frag 双缓冲（R2-8）
- optimization: L0C 累加器按块奇偶 ping-pong（2×32KB≤128KB），消除
  FIX-A→intra-B 的 o_frag WAR（M_FIX/FIX_M 握手对消失）。
- baseline_time: 1.572 → candidate_time: 1.568/1.489
- improvement: 0（持平——该 WAR 不在临界路径）
- precision: pass
- adopted: yes（严格更少的同步点）

## 【R2 第二份 msprof 证据】（msprof_output_r2b，1437us 口径）
- AIC mte2=748us（50%，全连续 48GB/s）、scalar 459（31%）；AIV scalar 61%
  （含节流等待）。**AIC 跨核 flag 等待合计 ~777us**（8 个 ID × ~470ns/次），
  其中 hi/lo 中继（id2/id4）491us 最大。
- AIV wait_id12（节流）= 1338us/1434（93%）→ V 极度空闲。
- 根因：节流 1 批滞后使 V 的 batch-k 中继"恰好赶上"C 消费——gate 链
  （exp 慢）在 C 到达时仍在路上。

## R2 Iteration 8 — 节流加深至 2 批次滞后 + NSLOT=8（R2-9）
- optimization: V 的 batch-k 生产改由 C 的 FIX-A(k-2) 门控（新增 peel-batch
  种子管线 + flag 计数配对）；NSLOT 4→8（3 个在飞批次）；主循环 Select-free
  边界重算（K_MAX）。V 领先 2 批次 → 所有 relay flag 预先置位。
- baseline_time: 1.568 → candidate_time: 1.530/1.431
- improvement: **-2.4%（稳态 -3.9%，-58us）**
- precision: pass
- adopted: yes
- next_hint: 剩余 flag 等待为协议固有开销（~470ns/次×4/任务）且大部分与
  MTE2/gemm 重叠——合并 flag 被 zN 分形布局阻断（region 操作数不安全）。

## R2 最终状态（官方口径 test_fla_chunk_o.py --level perf）
- **1.504 / 1.511 / 1.541 ms**（3 次独立运行，代表值 ~1.51，稳态 ~1.42）
- R2 起点 1.690 → **-10.7%**；R1 终态 1.65 → -8.5%；原始基线 3.285 → **-54%**
- 目标 < 1.320 ms：**未达标**（超 14%）
- 精度：--level all 全量 PASS（L0 2/2 + L1 9/9 ratio=1.0000，L2 10/10 正确
  拒绝，Boundary 3 PASS + 2 个 Stage 2 已记录预期 WARN）
- 最终 msprof（msprof_output_r2c，1373us）：AIC mte2 704（50%）/scalar
  635（45%，含与管线重叠的等待）/cube 176（12%）/fix 174（12%）/mte1 205（14%）；
  AIV scalar 80%（节流等待，V 空闲充裕）

## R2 剩余瓶颈归因（最终）
1. **AIC MTE2 88KB/任务 @ ~48GB/s = 1.82us/任务（50% busy）**——其中 56KB
   为 CV 中继字节（hi/lo 32KB 精度强制、a_mask 8KB 语义强制、v 16KB 连续化
   结构强制）。中继合并（单 flag 单读）被 L1 zN 分形布局阻断：列/行 region
   操作数在 zN 物理布局下不可表达（gemm_v0 虽接受 BufferRegion，但跨列
   region 的物理偏移与分形带布局不符）。
2. **跨核 flag 协议开销 ~470ns/次 × 4 次/任务**——V 预置后仍存在的固有
   往返；与 MTE2 工作部分重叠，临界路径贡献 < 名义值。
3. **AUTO_SYNC 的相邻 set/wait 标记对**阻止 MTE2 队列前瞻发射（FIFO 饥饿
   模式）——工具链层（需 marker 拆分或队列前瞻）。
4. HBM ~47-59% 利用（非墙）；IO 下限 797us 仍低于目标隐含值。

# ============ ROUND 3 (R3) — 画像驱动 + 地板核算（2026-10-23）============

> R3 起点：**1.513 ms**（官方 --level perf；R2 终态 1.504/1.511/1.541 复现）。
> 稳态 ~1.41。目标 < 1320 us。预算 10 轮，噪声阈值 3%。
> 本轮核心方法：**消融图谱**（ablation atlas）——逐成分移除测真实贡献。

## R3 Iteration 1 — 跨核 flag ID 冲突修复（12→40，后续随站点数升至 48）
- bottleneck_type: sync（节流协议被计数器混叠破坏）
- optimization: 源码级发现：ascend_combinecv.cc L373-400 的 auto flag ID 从 0
  逐站点递增分配（4 ws × 8 site = ID 0-31），**不知晓用户手动 ID**——R2 的手动
  节流 ID 12 与 auto 的 loop chunk-A ws2 站点冲突（官方文档确认每 ID 单计数器，
  双向共享）。冲突使 V 的节流 wait 可被自身 relay set 释放、C 的 relay wait 可被
  自身手动 set 满足——节流语义混乱 + 潜在正确性隐患。修复：THROTTLE_FLAG_ID=40
  （3-chunk 后站点 48 个 → 升至 48）。**发现方式**：r2c 画像 C 侧 relay 等待
  ~780us 与"设计的 lag-2 预置"矛盾 → 生成代码审查 → 源码+文档双实锤。
- baseline_time: 1.513 → candidate_time: 1.501
- improvement: -0.8%（噪声内；relay 等待 780→570us，多与 MTE2 重叠）
- precision: pass（l0_repr）
- adopted: yes（**正确性修复**——竞态隐患必须消除；性能收益为副）
- next_hint: 修复后 relay 等待仍 ~570us → V 交付迟到的根因另在。

## 【R3 裁决实验 A：V 是节流受限还是生产受限】
- 方法：NSLOT 8→32 + 8 个裸 set（内核起始即发，FIX 管线空 → 立即生效）给 V
  ~10 批领先信用。
- 结果：**设备死锁（exit 124）**——跨核 flag 计数器容量实测 **pending < ~10**
  （文档"最多 15 次"在 2-AIV 计数下更紧）。EXTRA=4（lag~6 信用）：正常运行
  但 **1.516 ms 无变化**。
- **结论：V 是生产受限**（给 6 批信用也跑不前）——节流深度/lag 参数无关紧要，
  V 的串行生产链就是墙。同时确认：计数器容量硬约束跨核流水深度上限。

## 【R3 裁决实验 B：消融图谱（ablation atlas）】
- 方法：主循环内逐成分移除（精度必坏、只测性能；peel/tail 保留维持 flag 计数）。
- 结果（steady，vs 1.42 基线）：
  | 移除成分 | 稳态 | Δ |
  |---|---|---|
  | V 侧全部数学（mul/exp/sub ×12/批） | 1.424 | **0** |
  | V 侧全部 prefetch 装载（8 GM→UB/批） | 1.423 | **0** |
  | V 侧 cast 拷贝（×12/批） | 1.442 | ~0（结构扰动） |
  | C 侧全部 gemm（×9/批） | 1.393 | -30us |
  | C 侧 Hstate 装载（×2/批） | 1.411 | -13us |
  | C 侧 FIX 直写（×3/批） | 1.396 | -19us |
  | a_mask relay 往返（双侧站点） | 1.366 | **-33us（单 relay 往返实测价）** |
  | hi/lo relay 往返（双侧站点） | 1.500 | +80（结构污染，弃用） |
- **结论：每任务计算/装载/拷贝几乎免费（合计 <80us）；墙 = 每任务 ~3.4us 的
  管线协议地板**：4 次 relay 往返（flag set+deliver+wait+MTE3/MTE2，实测每个
  ~0.08-0.13us）+ 每站点同步指令流。relay 数量下限 4/任务被 L1 zN 分形布局
  锁死（hi/lo 合并与 K-concat 单 gemm 均被 region 物理偏移阻断，R2 结论本轮
  重新推导确认）。

## R3 Iteration 2 — V 体重构（v-relay 上移 + prefetch 拆分）
- optimization: (a) v-relay ×2 上移到体首（仅依赖 staging，无 V-pipe 数学）；
  (b) next-A 的 4 个 prefetch 移到 chunk-A 链后（中体），next-B 留体尾——
  顶部 MTE2_V 等待从 72KB 排空（~2.3us 暴露）缩到 36KB。
- baseline_time: 1.501 → candidate_time: 1.489
- improvement: -0.8%（远小于预期）
- precision: pass
- adopted: yes（每个 relay 的交付时间严格不劣化）
- next_hint: 顶部 MTE2_MTE3 等待仍暴露体尾 prefetch 排空。

## R3 Iteration 3 — v 装载上移（R3-3）
- optimization: v 装载（每 chunk 8KB）从体尾移到 mask-A 后（排空藏于 gate-A
  计算 ~1.5us）；中体/体尾 prefetch 只留 a/q/g（28KB，各有 ≥1.5us 隐藏窗）。
- baseline_time: 1.489 → candidate_time: 1.497
- improvement: 0（噪声内）
- precision: pass
- adopted: yes（结构上消除了所有暴露排空窗口；实测确认 MTE2 排空非墙）
- next_hint: 消融图谱指向每批固定同步开销 → 3-chunk 批处理摊薄。

## R3 Iteration 4 — 3-chunk 任务批处理（R3-4，编排层主线 1）
- optimization: 每流水步消费 3 个跨步任务 (3k, 3k+1, 3k+2)：staging ×3 套
  （+36KB/vid）、L1 操作数 ×3 套（264KB<512KB）、NSLOT=9（环安全：lag≤2 时
  3d≡m mod 9 无解）、peel = 2 个完整 3-chunk 批（节流计数保持 lag-2 配对）、
  o_frag 按奇偶 ping-pong（chunk 0/2→a, 1→b）、THROTTLE_FLAG_ID 40→48
  （auto 站点增至 48 个）。每批固定同步（节流 wait/set、循环控制、地址链、
  顶部 WAR 对）摊到 3 任务。
- baseline_time: 1.497 → candidate_time: **1.481/1.490/1.480（代表 1.484，
  稳态 1.394-1.402）**
- improvement: mean -0.9%，steady -1.3%（7+ 次运行分布不重叠，真实但小）
- precision: pass（--level all 11/11 + L2 10/10 + Boundary；bench shape 与
  DK=96/DV=48 主循环覆盖 shape 均验证）
- adopted: yes
- **中途事故（已修复）**：FIX 消融实验后误将消融版覆盖快照 → 一度主循环无 FIX
  写（bench shape 零值、l1_bench FAIL 0.026）→ 从生成器重建 + seed2102 复验
  PASS。教训：消融变体必须用独立文件名，不得覆盖 good 快照。
- next_hint: 每批固定开销远小于预估（R2 时代 1→2 批化也是 0 收益的旧结论在新
  瓶颈域下部分复现）——每任务 relay 协议是地板。

## R3 Iteration 5 — C 侧全批装载上移（R3-5，尝试，弃用）
- optimization: C 体 15 个 MTE2 装载全部上移体首背靠背入队（per-ws FIFO 序
  保持），gemm 链后置——攻击 MTE2 饥饿（busy 43-50%）。
- baseline_time: 1.484 → candidate_time: 1.475-1.480（含一次误用 FIX 消融版
  的测量，修正后复测持平）
- improvement: 0（V 生产受限定步，C 侧装载顺序无关——C 只是在不同位置等 flag）
- precision: pass
- adopted: no（弃用，最小改动原则保留原 3-chunk 结构）

## R3 最终状态（官方口径 test_fla_chunk_o.py --level perf）
- **1.486 / 1.510 / 1.497 ms**（3 次独立运行，代表 ~1.498；快速门复验
  1.476/1.400）——R3 起点 1.513 → **-15us（-1.0%）**；R1 基线 3.285 → **-54.4%**
- 稳态 1.394-1.418（代表 ~1.40）
- 目标 < 1.320 ms：**未达成**（超 13.5%）→ [PERF_INCOMPLETE]
- 精度：--level all 全量 PASS（11/11 + L2 10/10 + Boundary 3 PASS + 2 个
  Stage 2 已记录预期 WARN）
- 消融图谱 + 裁决实验详见 optimization_log_r3.md；地板核算见 final_summary.md R3 章节

## R3 工具链级发现（新增，供上游参考）
1. **auto 跨核 flag ID 与用户手动 ID 冲突**（combinecv global_sync_flag_id 从 0
   递增、不感知用户 ID）——本 kernel 曾因此节流失效 + 潜在竞态。建议 pass 预留
   用户 ID 段或提供查询 API。
2. **跨核 flag 计数器容量实测 < ~10 pending**（8 裸 set + 2 peel set 即死锁；
   文档标 15）——2-AIV 计数下更紧；跨核流水深度硬上限。
3. **relay 合并（4→3 往返/任务）仍被 L1 zN 分形布局阻断**（region 物理偏移 ≠
   逻辑偏移，行/列堆叠均断裂）——zN 感知的 region 拷贝/操作数可再省 ~40-50us。
4. **每任务 ~3.4us 管线协议地板**：全部计算/装载/拷贝消融合计 <80us；剩余为
   4 relay 往返 × ~0.1us + 每站点同步指令流——Developer 模式 AUTO_SYNC 的
   每 op 同步开销（R1 已记录）在 relay 密集结构下的下限体现。

# ============ ROUND 4 (R4) — host 分发绕过 + 4-chunk 尝试 + 终态画像 ============
# （2026-10-24，用户 R3 检查点决策"继续调优"）

> R4 起点：**1.492 ms**（官方 --level perf，3 次复现 1.484/1.490/1.501；稳态
> ~1.411；首 rep 2.17-2.28 → 暴露 ~0.83ms）。目标 < 1320 us。预算 10 轮。
> 主攻方向（按预估收益）：主线 1 host 分发绕过（est -80~90us 均值）、
> 主线 2 4-chunk 批处理（est -10~20us）、主线 3 终态新画像。

## R4 Iteration 1 — host 分发绕过（_FastLaunch 缓存式低层直呼）
- bottleneck_type: other（host 分发；设备侧 R3 已证协议地板）
- optimization: 公共 JIT 分发路径（JITKernel.__call__ → CythonKernelWrapper
  .forward）对本全静态 kernel 每次调用重复：**tvm.arith.Analyzer 重建
  289us**（仅符号 shape simplify 需要——本 kernel 无符号 shape）+ cython 层
  shape 逐参重解析/校验 ~292us + 5×torch.empty 65us + stream 查询 35us，
  合计 **795.7us/调用**（JIT 缓存查找仅 2.4us，lib.call 入队仅 9.1us；
  分解脚本 fla_chunk_o_dispatch_decompose_r4.py）。wrapper 内新增
  _FastLaunch：按 (shape-spec, device, stream) 键缓存编译实例句柄 + 4 个
  relay workspace（内部 write-then-read 暂存、同流串行复用安全；输出保持
  每次 fresh torch.empty 不别名）+ 预建 argtypes，直接走生成库的
  extern "C" call ABI（与 cython 适配器同一入口）。护栏：(a) 每调用全量
  输入 shape/dtype/contiguity/device 校验；(b) arm 时 bitwise 自检
  （公共结果 == 两次复用同一 workspace 的快路径结果）；(c) 任何异常/护栏
  失败回退公共路径（上游重构最坏退化为原性能，不会错果）。
- baseline_time: 1.492 → candidate_time: **1.416**（1.413/1.416/1.418；
  稳态 1.401-1.407；首 rep 1.513-1.566）
- improvement: **-76us（-5.1%，均值）**；首 rep 暴露 0.83 → **0.13ms**；
  分发 795.7 → **163.4us**（其中 entry.launch 本体 51.3us + stream 查询
  37.6us + 输出 empty 13us + wrapper 杂项）
- precision: pass（--level all：L0+L1 11/11 ratio=1.0000 + L2 10/10 正确
  拒绝 + Boundary 4 PASS/2 既有预期 WARN；另验证双 stream 交错/21+ 次复用/
  非连续输入回退公共路径同报错）
- adopted: yes
- next_hint: 上游 tilelang-ascend ascendc_pto（43b0b697）对 tilelang/jit/
  零改动——无现成快路径可复用；torch_npu._C._npu_getCurrentRawStream 比公共
  current_stream() 快 45×（0.8 vs 36us）但属私有 API，未用（记录备查）。

## R4 Iteration 2 — 4-chunk 任务批处理（尝试，编译期放弃）
- bottleneck_type: sync（每批固定同步开销摊薄——R3 已证边际趋零的方向）
- optimization（未落地）: 生成器参数化 4-chunk（gen_r4_body.py：staging×4、
  L1×4、NSLOT 12、o_frag 按奇偶 a/b/a/b、THROTTLE 64）。前置检查全部通过：
  (a) UB 预算 +36864B/vid（v 8K+a 4K+q 8K+g 16K @DK=DV=128）按 256KB UB 估
  算放得下；(b) L1 4 套 352KB < 512KB；(c) 环安全 4d≡m (mod 12) 对 d≤2、
  m∈[0,3] 无解；(d) **flag ID 空间探针**：3-chunk kernel 上 THROTTLE 48→64
  实测 bench 1.420/精度 PASS 与 48 完全一致 → ID 64 与 0-47 不混叠，
  ≥65 个不同跨核 flag ID 可用（文档 ISASI 范围 0-10 明显保守，CANN
  atlasascendc_api_07_0273）。
- 结果：小 shape（DK=64/DV=64）编译+精度 PASS；**DK=DV=128 shape 编译期
  段错误（exit 139，faulthandler 定位 TVM OptimizeForTarget 内）**。
- 根因（实锤）：**tilelang A2/A3 UB 硬上限 = 196352B**
  （src/transform/ascend_memory_planning.cc:35 ASCEND_SHARED_MEM_SIZE；
  src/target/codegen_ascend.cc:40 ASCEND_A2A3_UB_SIZE）。3-chunk kernel 的
  ascend_ub 恰好 = 196352B（**100.0% 打满**，kernel_source_r3_final.c L42）
  → 第 4 块 staging +36864B 溢出 → memory-planning pass C 层 CHECK 崩溃
  （穿 FFI 段错误而非干净 Python 报错）。staging 无可释放：管线峰值期全部
  活跃（3 在飞 chunk + 下批 prefetch 原地双重占用）；Vector 侧缓冲不可迁
  L1（内存层级：L1 为 Cube 专属）。
- adopted: no（回滚 r4_iter1 快照，md5 校验）
- 记录：4-chunk 对本算子族在 tilelang 192KB 级 UB 预算下不可行；3-chunk
  设计已精确饱和 UB。结构性上限（除非每 chunk staging 缩小——IO 唯一性
  决定不可能——或 A5 平台 262144B 预算）。

## R4 Iteration 3 — 终态 msprof 画像（主线 3）
- 结论：**无 R3 消融未覆盖的新可寻址 stall**。单_launch 1391.1us
  （OpBasicInfo；Block Dim 20 / Mix 40）。AIC：cube 12.7%/mte1 14.8%/mte2
  38.0%/fix 10.0%/scalar 24.4%，relay 等待合计 ~733us（mod-16 归并口径，
  与 MTE2 重叠）；AIV：**throttle 等待（id0=48 mod 16）1114.6us**（lag-2
  节流设计使然），vec 20.5%/mte2 37.4%/mte3 17.1%，vector_stall 293us。
  双侧占满墙时间（C 1329 / V 1319-1322 vs 墙 1391）——无空闲 bubble，
  计算/搬运各 <40% busy，**墙 = 同步协议地板**（R3 结论复现，非新增项）。
  msprof 数据：perf_tuning/msprof_output_r4/。

## R4 最终状态（官方口径 test_fla_chunk_o.py --level perf）
- **1.415 / 1.414 / 1.421 ms**（终态复测 3 次，代表 ~1.417；快路径 A/B
  批次 1.413/1.416/1.418）——R4 起点 1.492 → **-75us（-5.0%）**；
  R1 基线 3.285 → **-56.9%（累计）**
- 稳态 1.401-1.410（~1.406）；首 rep 1.513-1.566（暴露 ~0.13ms）
- 公共路径口径（交付文件关闭旁路的等价数字）：~1.492（R4 基线即公共路径）
- 目标 < 1320 us：**未达成**（超 7.3%）→ [PERF_INCOMPLETE]
- 精度：--level all 全量 PASS（11/11 + L2 10/10 + Boundary，exit=0；
  perf_tuning/r4_final_precision.log）
- 分发绕过的 PR 可接受性评估：见 final_summary R4 章节（结论：可交付，
  带护栏与回退；两口径数字均已如实报告）

## R4 工具链级发现（新增，供上游参考）
1. **公共 JIT 分发对静态 kernel 每调用重建 tvm.arith.Analyzer（289us）**
   （cython_wrapper.pyx L95，Analyzer() 无条件构造，仅符号 shape simplify
   使用）+ cython 层逐参 shape 重解析/校验 ~292us——合计 ~0.6-0.8ms/调用
   纯浪费。建议：Analyzer 惰性构造（首个符号 shape 才建）或 wrapper 级
   缓存；上游 ascendc_pto（43b0b697）尚无此修复。
2. **A2/A3 UB 预算硬上限 196352B 且 memory-planning 溢出为段错误**
   （非干净报错）——建议 pass 层显式校验并抛 Python 可捕获异常；文档化
   192KB 级 UB 预算（区别于硬件 256KB）。
3. **跨核 flag ID 空间实测 ≥65 个不同 ID**（文档 ISASI 标 0-10，实测
   0-64 互不混叠）——R3-1 的 ID 冲突问题真正的安全边界比文档宽得多。
4. torch_npu 私有 `torch_npu._C._npu_getCurrentRawStream` 0.8us vs 公共
   `torch.npu.current_stream().npu_stream` 36us（45×）——上游反馈项。

# Round 5（同步协议架构轮，用户决策"继续第五轮调优"）

> 主攻方向：同步协议架构本身（pass_configs 旋钮全量清点 / 站点削减 /
> 手动 flag 协议 / _FastLaunch 微优化）。模式红线：Developer 模式内深化
> 允许，整体 Expert 切换需上报。预算 10 轮，噪声阈值 3%。

## R5 基线（本轮起点）
- 官方 bench：1.421 ms（per-rep 1.381-1.580，稳态 ~1.40）；md5 与
  r4_final 一致（58c43db0）
- msprof 设备侧：1434.2us（msprof_output_r5_base）
- 快照：history_version/fla_chunk_o_perf_r5_start.py

## R5 Iteration 1 — 编译器层同步旋钮（主线 1，3 个 A/B）
- bottleneck_type: sync
- optimization: ① compile_flags=["--cce-auto-sync=off"]（bisheng 编译器
  内存操作自动同步层关闭，ascendc 默认 on——本轮清点发现的双层同步结构
  假设）；② +"--cce-remove-auto-sync=true"（编译器冗余 flag 消除）；
  ③ +"-O3"。旗标到达性实证（编译报错命令行可见）；kernel 源零改动。
- baseline_time: 1.421 → candidate: 1.426 / 1.426 / 1.435
- improvement: 0（全部噪声内）
- precision: pass（reference check ratio=1.0000）
- adopted: no（全部回滚）
- **结论：编译器层 auto-sync 对本 kernel 无实际插入——417 个同步站点
  100% 来自 TVM AscendSyncInsert pass 显式调用**。pass_configs 全量清点
  （28 键）：TL_ASCEND_AUTO_SYNC_VS 是主 pass 的补充（只加不减，方向
  相反）；其余键无同步语义。主线 1 关闭。

## R5 Iteration 2 — relay 单 batch flag 协议（主线 3 核心假设）
- bottleneck_type: sync
- optimization: workspace_1..4 改名 relay_hi/lo/a/v（逃逸 AUTO_CV_SYNC
  纯名字匹配，ascend_combinecv.cc L201 实证）+ 单一 RELAY_READY flag
  （ID 49；V 侧体尾 set_cross_flag("MTE3")（需 T.Scope("V") 包裹——
  ResourceForPipe 只认 V/M/MTE1/FIX），C 侧体首 wait_cross_flag(49,
  "MTE1")）。稳态 lag-2 下 flag 预置，12 检查 → 1 检查。
- baseline_time: 1.421 → candidate_time: **1.934（+36% 劣化）**
- improvement: -36%
- precision: pass（ratio=1.0000，bitwise 跨调用稳定）
- adopted: no（回滚）
- **关键实证**：msprof 设备侧反而 1346.8us（-87us）但 AIC cube_wait
  98.1%（基线 52.9%）——单 flag 把 C 的 12 次装载绑定到 V 的 MTE3 全批
  排空后，**批内 chunk 级 C/V 交错被摧毁，退化为批粒度锁步**；逐 launch
  1.36-1.96 剧烈波动。快发射 launch（flag 预置）仅 -20us → **flag 检查
  开销（459ns×12×133=733us 管线口径）的墙临界份额 ~3%**，远小于锁步
  损失。逐 relay 渐进触发是承重结构。主线 3 的"flag 合并"方向证伪。

## R5 Iteration 3 — hi/lo K=256 融合 + 9 站点手动 flag（分析性放弃）
- bottleneck_type: sync
- optimization（未落地）: qg_hi/qg_lo 融合为 relay_hilo [.., BS, 2DK]
  （K-concat），C 侧单装载 + 单 K=256 gemm（[hi|lo]@[H;H] 分块恒等），
  9 站点手动 flag。收益预算（R3 消融定标）：-3 站点（~33us）-3 gemm
  issue -3 M-barrier（~10us）；代价：H 双读 +13us 且**同 buffer 名 WAW
  使 AUTO_SYNC 在 H 堆叠双装载间插 PipeBarrier<MTE2>（按名判依赖无切片
  分析，ascend_sync_insert.cc L1485）→ C 标量流 +250us 量级停顿**。
  V 侧预堆叠 relay 变体净 ~-10~-20us（+3 hh 站点抵消 -3 站点）。
- 结果：净负（直接变体）/ 低于噪声风险比（relay 变体）→ **弃用**。zN
  region 操作数（单装载多 gemm 切片）为 R2 证伪禁区不适用排除。

## R5 Iteration 4 — _FastLaunch stream 查询快路径（主线 4）
- bottleneck_type: host
- optimization: torch_npu._C._npu_getCurrentRawStream 私有绑定（0.8us vs
  公共 37.6us）+ 导入期/首用交叉校验 + 永久回退护栏；中途修复一个
  UnboundLocalError（wrapper 内对 global 赋值）。
- baseline_time: 1.421 → candidate: 1.430 / 1.417（稳态 ~1.403-1.405）
- improvement: 0（**host 分发 163→~127us 但完全隐藏于设备时间后，
  官方口径无感**——与任务书预期一致"低于噪声门"）
- precision: pass
- adopted: no（按"无变化回滚"规则；私有 API 风险面无对价收益）
- 记录：实现+护栏已验证可行（history_version/fla_chunk_o_perf_r5_iter4_
  stream.py 备查），若上游接受私有 API 可直接复用。

## R5 最终状态
- **1.421 / 1.417 ms**（终态 == R4 终态，md5 校验一致；本轮零采纳）
- 精度：--level all 全量 PASS（exit=0，24 PASS + 2 既有预期 WARN；
  perf_tuning/r5_final_precision.log）
- 主线 1-4 全部穷尽；连续 4 轮无净提升 → 按中止条件停轮
- **[EXPERT_MODE_REQUIRED]**：见 final_summary R5 章节论证

## R5 新增工具链/结构实证（供上游与后续轮参考）
1. **bisheng --cce-auto-sync / --cce-remove-auto-sync / -O3 对
   AscendSyncInsert 显式同步流无影响**（到达性实证）——编译器层不是第二
   同步层；--cce-auto-sync-log 旗标本身触发编译器 ERR99999 崩溃。
2. **AUTO_CV_SYNC 的跨核 flag 注入是纯名字匹配**（"workspace" 子串，
   combinecv.cc FetchWorkspaceName）——改名即逃逸，但对多写单读的融合
   buffer 不安全（按名+顺序配对）。
3. **ResourceForPipe 只认 V/M/MTE1/FIX**——手动 set_cross_flag("MTE3")
   需 T.Scope("V") 包裹才能进 V 段（否则 AscendResourceScopeVerify 拒绝）。
4. **跨核 relay flag 的检查开销墙临界份额 ~3%**（iter2 快发射实测 -20us
   vs 管线口径 733us）——"批量 flag"类优化的理论收益上限被实测否定。
5. **relay 逐站点渐进触发（每个 set 紧随各自 MTE3 copy）是批内 C/V 交错
   的承重结构**——任何粒度粗化（单批 flag）以 +36% 劣化代价摧毁交错。
6. **同 buffer 名静态切片双写触发名字级 WAW barrier**（无切片重叠分析）
   ——L1/GM 融合装载设计的隐性成本源（hi/lo K-fusion 因此净负）。

## R5 最终状态（官方口径复测 3 次）
- **1.425 / 1.417 / 1.424 ms**（代表 ~1.422，稳态 ~1.40）——与 R4 终态
  （1.415/1.414/1.421）会话噪声内一致；本轮零采纳，交付文件与 R4 终态
  md5 相同（58c43db00e485cf4ceb1b7e85c5a037f）
- 精度：--level all 全量 PASS（exit=0；r5_final_precision.log）
- 中止原因：主线 1-4 穷尽 + 连续 4 轮无净提升（中止条件 #2）
- 判定：**[EXPERT_MODE_REQUIRED]**（论证见 final_summary R5 章节）
