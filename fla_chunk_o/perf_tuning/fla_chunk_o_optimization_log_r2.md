# fla_chunk_o Stage 3 ROUND 2 优化记录（optimization_log）

> 环境：Ascend 910B3 (64GB HBM, 20 AI Cores 可见) / CANN 9.2.0 / tilelang editable @ 2f22c6ac。
> R2 起点：1.690 ms（官方 --level perf，与 R1 终态 1.638-1.658 一致，噪声内）；目标 < 1320 us。
> 计时口径：NPU event，warmup=10，rep=10，均值，仅设备侧。快速 A/B 用
> perf_tuning/fla_chunk_o_bench_iter.py（同口径 + l0_repr 快速精度门）。

## R2 Step 1-2: 基线与算子类型（沿用 R1 结论）

- R2 基线（本会话实测，2026-10-23）：**1.690 ms**（稳态 ~1.58），per-rep
  [2.691, 1.571, 1.608, 1.483, 1.603, 1.623, 1.567, 1.598, 1.541, 1.610]。
- 算子类型：CV 混合型（R1 已判，KERNEL_TYPE_MIX_AIC_1_2）。
- R1 终态 msprof 归因：AIC scalar 53% / mte2 29% / cube 7%；AIV vec 13% / mte2 28%。
  → 瓶颈 = AIC 串行链（每任务 ~4.0us：mte2 1.35 + gemm 0.7 + FIX 0.15 + 同步/scalar
  1.4-1.8）。scalar 53% 含 MTE2 队列标记等待（暴露的搬运延迟）+ flag/屏障/地址链。

## R2 生成代码新证据（fla_chunk_o_kernel_source_r2_start.c）

1. **condval 地址链未 CSE**：AIC 循环体 11 条、AIV 循环体 12 条
   `if (tt*5 + cid/4 < K) condval = ...; else condval = cid;`（同一 Select 表达式
   在每个 T.copy 地址站点被复制一份）——纯 scalar 分支链，每任务 ~23 条。
2. **o 中继路径全链**（可整体消除）：C FIX→o_relay(GM 16KB 写) → V MTE2 读回
   o_half(8KB/vid) → V MTE3 写 O(8KB/vid)，伴随 MTE3_MTE2/MTE2_MTE3 标记对 ×2、
   手动 flag 等待、6 条 V 侧 drain 地址 condval 链。
3. **UB 15 buffer 静态偏移无生命周期复用**（MEMORY_PLANNING 未做跨语句复用）：
   per-vid 172032B / 196352B（87.6%），L1 余 436KB 空闲。
4. `copy_l0c_to_gm` 的 realDstN=行 stride（src/op/ascend.cc compute_strideN L153-178）：
   strided O 直写可生成正确参数；R1-7 已证该指令的 fp32→bf16 fixpipe 转换精确。

## R2 Part A 优化点清单

[#R2-1] [C 侧 fixpipe 直写 O（消灭 o_relay + V drain 全链）]（新生成代码证据 2/4；
      antipatterns "CV 中转"思想的极限化——把最后一跳中继也去掉）:
      **适用** — C 的 FIX 直接写 O[bb, s0:s0+64, bh*DV:...]（strideN=H*DV 由
      compute_strideN 正确推导，实证见证据 4）；V 侧保留 wait_cross_flag(12/13)
      作为节流（V ≤2 任务领先，NSLOT=4 安全性论证不变）；o_relay/o_half/
      drain 标记对/6 条 drain 地址链全部消失。预期 -0.3~-0.6us/任务（V drain
      MTE 流量 -32KB/任务 + 2 组标记 + C 侧不变）。
      风险：fla_wy_fast "L0C→GM(bf16) fixpipe 直存损坏" 记录（§8-7）——但 R1-7
      已证本工具链同指令（fp32→bf16 + 连续 stride）精确；差异仅 stride。
      plan: repair-kernel-first（先 l0_repr 快速精度门 + bench；损坏即回滚）。
[#R2-2] [Select-free 主循环 + Select 尾部剥离]（证据 1）:
      **适用** — 主循环 t ∈ [2,407]（prefetch t+1 ≤ 408 对全核合法）用纯算术
      地址（零 Select）；尾部 t ∈ [408,411] 4 个 peeled 迭代保留 Select 回退；
      peel-0/1 本就恒合法（cid, cid+20 < GRID）去 Select。消除 ~96% 迭代的
      ~23 条分支链/迭代（双核）。
[#R2-3] [V 链就地化压缩 UB]（证据 3；tile.mul/sub 文档明示 dst 可别名 src，
      ascend_tile.py L1134/L1158）:
      **适用** — gL=Muls(g_ub,g_ub,LN2)（已证模式 q_ub×scale）；Exp 就地
      g_ub；qg=Mul(q_ub,q_ub,g_ub)（dst==src0）；lo=Sub(q_ub,q_ub,hi_f32)
      （dst==src0）→ 消灭 gL_ub/gexp_ub/qg_ub/qg_lo_f32 4 个 buffer
      （-64KB/vid，172→108KB/vid）。数值逐元素等价（同 op 同操作数），精度
      应逐位不变。作为 #R2-4 的 UB 腾挪前置。
[#R2-4] [2-chunk 任务批处理]（R1 主线 1，UB 溢出阻塞解除后重启）:
      **适用** — 每迭代处理任务对 (2k, 2k+1)：staging 双份（+28KB/vid，由
      #R2-3 腾出）；fp32 计算链顺序复用；任务 B 的 h/v MTE2 延迟藏于任务 A
      的 gemm 段；迭代级开销（循环控制/屏障合并）减半。NSLOT=4 配对批
      {0,1}/{2,3} 无冲突；节流 flag 12/13 每批 2 set/2 wait 平衡。
      预期 -0.4~-0.8us/任务（MTE2 暴露减半 + 迭代开销摊薄）。
[#R2-5] [hi/lo/a 手动 flag 化]（R1 主线 2）:
      **暂缓（分析性降级）** — R1 "两核 80% 互等" 是手动 o flag 之前的证据；
      现结构 V 领先 ≤2 任务，C 的 CrossCoreWaitFlag(8/2/5) 实测等待 ≈0
      （R1 next_hint 已记录"V 领先后等待≈0"）；手动化不减少 V 侧
      V_MTE3 WAR 标记对（那些是 UB 缓冲防覆写必需），也不减少 C 侧等待
      指令数。预期收益 ≈0。仅当 #R2-1..4 落地后 msprof 仍显示 flag 等待
      占比显著时重估。
[#R2-6] [gate 链融合 / bf16 掩码 / K-concat / hi-lo 打包 / block_DV=64 / 去 hi-lo
      双 gemm / L1 ping-pong / PTO / threads 裸切换]: **不适用** — R1 已证伪，
      禁止重试清单（除非新前提：#R2-1 消灭 drain 后 V 侧结构剧变，届时重估
      V 侧相关项）。

## R2 Part B [ORDER-PLAN]

1. [#R2-1] C 直写 O — 前置依赖: 无 — 理由: 一行改动级试错、潜在收益最大、
   且简化后续所有迭代的 V 侧结构。
2. [#R2-2] Select-free 主循环 — 前置依赖: 无（与 R2-1 独立，叠加上）— 理由:
   纯重结构、低风险；放在 R2-1 后避免地址链基线漂移干扰归因。
3. [#R2-3] V 链就地化 — 前置依赖: 无 — 理由: UB 腾挪 enabler，精度应逐位不变。
4. [#R2-4] 2-chunk 批处理 — 前置依赖: #R2-3（UB 预算）— 理由: R1 已证无腾挪
   直接批处理溢出（~350KB/vid）。
5. [#R2-5] 手动 flag 三对 — 条件触发（msprof 证据）。

## R2 Step 4-5: 逐项实施与验证记录（RESULT 汇总，每步 l0_repr 快速精度门）

```
[ORDER-CHECK] 准备实施: [#R2-1] C 侧 fixpipe 直写 O | 前置依赖: 无 | 结论: ✅
[RESULT-R2-1] 直写 O（realDstN=H*DV）| 精度 pass | 1.690 -> 1.652 ms | -2.2%
              （稳态持平；采纳为结构基础：-8KB/vid UB、o_relay 环路全消、
              V 侧 6 条地址链删除；工具链事实修正：直存损坏记录不复现）
[RESULT-R2-2] Select-free 主循环 + Select 尾 | 精度 pass | 1.652 -> 1.670 ms | 0
              （condval 链非主要开销——生成代码确认主循环零 if 链后持平）
[ORDER-CHECK] 准备实施: [#R2-4] 2-chunk 批处理 | 前置依赖: R2-1(UB) ✅ | 结论: ✅
[RESULT-R2-4] 2-chunk 批处理（staging 双份 +28KB/vid）| 精度 pass | 1.670 ->
              1.675 ms | 0（同名 WAR 的 M_MTE2 串行在块间重现）
[RESULT-R2-5] L1 按名 ping-pong + B 装载上移 | 精度 pass | 1.675 -> 1.667 | 0
              （M_MTE2 确认消失——暴露 MTE2 管线本身 72% busy 为真瓶颈）
[RESULT-probe] v 装载换连续读（计时探针）| 1.667 -> 1.487 ms | -11%（实证
              strided 64×256B v 读是 AIC 关键路径大头）
[ORDER-CHECK] 准备实施: [#R2-3+#R2-6] V 链就地化 + v 中继 | 结论: ✅
[RESULT-R2-6] 就地 gate 链（mul/exp/sub dst==src）+ AIV 读 strided v → 连续
              ws4 中继 | 精度 pass | 1.667 -> 1.563 ms | -6.2% ✅
              （中途：lo 中继 fp32 直写 GM bf16 → 编译错，cast-on-copy 缺陷
              重确认，经 qg_lo_ub 修复；v 预取变体持平，保留预取版）
[RESULT-R2-7] O 写回迁 V（o_relay + V 排水）| 精度 pass | 1.579 -> 1.759 |
              +11% 劣化 → 回滚（V 排水链代价 ≫ C 侧 strided FIX 节省；
              R1"中性"结论在 V 有负载时反转为"直写严格优"）
[RESULT-R2-8] o_frag 按块奇偶 ping-pong | 精度 pass | 1.572 -> 1.568 | 0
              （该 WAR 不在临界路径；采纳——同步点严格更少）
[RESULT-R2-9] 节流加深 2 批次 + NSLOT=8 + peel-batch | 精度 pass(--level all) |
              1.568 -> 1.530 ms | -2.4%（稳态 -3.9%）✅
              （V 预置所有 relay flag；777us flag 等待中的 JIT 交付部分回收）
```

## R2 最终结果

- 官方 perf_bench：**1.690 → 1.504/1.511/1.541 ms（代表 ~1.51，-10.7%）**；
  稳态 ~1.42。原始基线 3.285 → **-54%**。
- 目标 < 1.320 ms 未达成（超 14%）；[PERF_INCOMPLETE]。
- 精度：--level all 全量 PASS（与 R1 交付同等覆盖，Boundary 2 WARN 为
  Stage 2 已记录预期行为）。

## R2 未走通的路（新增，供工具链修复参考）

1. **L1 zN 分形布局阻断中继合并**：4 relay → 1 单 flag 单读需要打包 L1
   buffer + 列/行 region gemm 操作数；zN 的 16 列带物理布局使 region 的
   逻辑偏移 ≠ 物理偏移（行堆叠同样跨带断裂）。gemm_v0 前端虽接受
   BufferRegion，跨列 region 不安全。若工具链支持 zN 对齐的 region 操作数
   或 per-buffer 显式 L1 布局，可再省 ~0.5-1.5us/任务。
2. **跨核 flag 协议 ~470ns/次**（V 预置后仍存在）：4 relay × 470ns/任务；
   与 MTE2 部分重叠。需硬件/运行时层面的批量 flag 或更廉价协议。
3. **AUTO_SYNC 相邻 set/wait 标记对**：SetFlag 与 WaitFlag 相邻发射使
   scalar 立即阻塞，MTE2 队列无法前瞻装载（FIFO 饥饿）——批处理/ping-pong
   均无法绕过（装载早于 marker 即被 marker 覆盖等待）。
4. **O 写回迁 V 在 V 有负载时劣化**（+11%）：排水链的 flag+标记+双搬运转
   嫁到 V 侧的代价超过 C 侧 strided FIX（37GB/s vs 连续 ~2×）节省。
5. **strided GM 读速率 ~16GB/s vs 连续 ~48GB/s**（AIC MTE2）：bshd 布局的
   64×256B 行块（v）与 32×{128,256,512}B（A/q/g，V 侧）读慢 3 倍——v 已
   用中继连续化；A/q/g 在 V 侧非临界路径，未处理。
