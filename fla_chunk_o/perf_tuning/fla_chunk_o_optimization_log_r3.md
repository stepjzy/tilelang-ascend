# fla_chunk_o Stage 3 ROUND 3 优化记录（optimization_log）

> 环境：Ascend 910B3 (64GB HBM, 20 AI Cores) / CANN 9.2.0 / tilelang editable @ 2f22c6ac。
> R3 起点：**1.513 ms**（官方 --level perf，2026-10-23 实测；R2 终态复现 1.504/1.511/1.541 一致）。
> 稳态 ~1.41 ms。目标 < 1320 us（官方均值口径）。预算 10 轮，噪声阈值 3%。
> 快速 A/B：perf_tuning/fla_chunk_o_bench_iter.py（同口径 + l0_repr 精度门）。

## R3 Step 1-2: 基线与算子类型（沿用 R1/R2 结论）

- R3 基线（本会话实测）：1.513 ms（per-rep [2.412, 1.412, 1.399, 1.406, 1.404,
  1.401, 1.443, 1.420, 1.423, 1.412]；首 rep +1.0ms = host 分发地板暴露，
  与编排层核算吻合：mean ≈ steady + dispatch/10 = 1.41 + 0.10）。
  会话首跑 2.288 ms 为前会话残留设备态污染（AIV 计数器卡 97%），二次复跑恢复。
- 算子类型：CV 混合型（KERNEL_TYPE_MIX_AIC_1_2，R1 判定沿用）。

## R3 新证据（本轮核心发现，全部源码/官方文档级）

### E1. 【flag ID 冲突——R2 遗留缺陷】
- 生成代码（fla_chunk_o_kernel_source_r3_start.c，R2 终态同构）：
  - AIC 循环体：`CrossCoreWaitFlag(12)`（auto ws2 lo-A relay 读）与
    `CrossCoreSetFlag<0x2, PIPE_FIX>(12)`（手动节流 set）**同 ID**；
  - AIV 循环体：`CrossCoreWaitFlag(12)`（手动节流 wait）与
    `CrossCoreSetFlag<2, PIPE_MTE3>(12)`（auto ws2 lo-A relay 写）**同 ID**。
- 根因（src/transform/ascend_combinecv.cc L373-400）：auto pass 的
  `global_sync_flag_id` 从 0 逐 sync-point 递增，**不知晓用户手动 flag ID**。
  当前 4 workspace × 8 chunk-site = 32 个 sync point → auto 占用 ID 0-31，
  与手动 ID 12 冲突（R2-10 移到 12 时 auto 仅 0-8，此后站点数增长无人复查）。
- 官方文档（CrossCoreSetFlag, CANN 9.0.0-beta1）："每个ID对应一个用于控制
  同步的计数器"——**每 ID 单计数器**（非按方向分离），冲突使手动节流与
  auto relay 共享计数器 → 节流语义混乱（V 的节流 wait 可被自身 relay set
  提前释放 / C 的 relay wait 可被 C 自身手动 set 满足）。
- 画像吻合（msprof_output_r2c，AIC 稳态站点等待，msprof id 按 mod-16 归并）：
  id4={hi-A,a-A}=312.5us + id12={lo-A,v-A}=309.9us + id5={hi-B,a-B}=90.3 +
  id13={lo-B,v-B}=61.8 ≈ **780us relay 等待（1.9us/batch）**——设计的
  "V 领先 2 批 → flag 预置" 完全未发生，V 实际 just-in-time 交付。
  AIV wait_id12=1342us（98.5% 节流空转）与"V 交付慢"并存 → 计数器混乱
  的节流既没让 V 领先、也没让 V 全速——纯浪费。

### E2. 【MTE2 FIFO 饥饿——结构实锤】
- 生成代码 AIC 循环体：8 个 `CrossCoreWaitFlag` 与 `copy_gm_to_l1` 严格
  交错（wait 在 copy 前），scalar 每个 wait 处阻塞时 MTE2 队列仅含先前的
  1-2 个 copy → MTE2 无法前瞻（R2 已知"AUTO_SYNC 相邻 set/wait 标记对
  阻止前瞻"，本轮在 cross-wait 站点上再次实锤）。
- 画像：AIC mte2 busy 704us/1373 墙钟（**49.5% duty，starved 非带宽墙**；
  88KB/task @ ~102GB/s 等效 — 全连续读 + L2 命中），scalar_mte2_stall
  646.5us。

### E3. 【节流深度不足】
- 即使修复 E1，lag-2 的交付裕量仅 ~0.3us（C: FIX-A(k-2)→FIX-B→batch k-1
  ≈ 3.8us vs V: FIX drain+flag ~1us + 生产 ~2.5us）——方差即等待。
  lag-3 裕量 ~10us。NSLOT=8 环安全分析：lag L 的 V 写 (2c+2L)%8 与 C 未来
  读槽位冲突条件 2L≡m (mod 8), m∈[1,2L-1] → L=3 安全（6∉[1,5]），
  L=4 不安全（8≡0 冲突 batch c 槽位）。

### E4. 编排层地板核算（框架层，不可修，写入 final_summary）
- host 分发 ~0.999ms/调用（~0.6ms TVM Arith Analyzer 重建 + ~0.4ms
  Cython forward）；mean ≈ steady + dispatch/10（1.41+0.10=1.51 ✓）。

## R3 Part A 优化点清单

[#R3-1] [手动节流 flag ID 冲突修复（12→40）]（证据 E1）:
      **适用** — auto pass 占 0-31；40 = 32+8 裕量（3-chunk 批处理将增至
      ~44-48 站点，届时需复查）。预期恢复设计的 lag-2 节流：V 领先 2 批、
      relay flag 预置 → C 的 780us relay 等待大部分消失（部分与 MTE2
      重叠，净收益 A/B 实测）。官方文档确认 ID 为 uint16_t；>15 的 ID
      在 msprof wait_id 列不可见（归并计数），不影响功能（auto 16-31
      已在用）。风险：>31 的硬件支持未证实——编译+精度门+bench 立即验证。
[#R3-2] [C 侧全批装载上移]（证据 E2）:
      **适用** — C 体 10 个 MTE2 copy（h/v/a/hi/lo × 2 chunk）全部上移到
      体首、背靠背入队（per-workspace 站点序不变：ws4-A,ws3-A,ws1-A,
      ws2-A,ws4-B,ws3-B,ws1-B,ws2-B → 各 ws 的 A→B 相对序保持，FIFO
      配对不变）；首 gemm 前仅有的阻塞点是 cross-wait（R3-1 修复后预置
      → no-op）。MTE2 队列满载 176KB/batch 连续排空。体首 M_MTE2(7)
      排空对继续保护 L1 WAR（上批 gemm 读 vs 本批装载）。
      依赖 #R3-1（relay flag 未预置时上移反而更早阻塞）。
[#R3-3] [节流 lag 2→3]（证据 E3）:
      **适用** — peel-fill1 的 C 段补一个 set（使全局 set 序 s0,s_f1,s_b1,
      s_2..；w_k↔s_{k-3}）。计数器语义下 lag-3 成立；NSLOT=8 环安全（E3
      分析）。为 R3-2 的更早 relay 读取（体首）提供交付裕量。
      依赖 #R3-1（冲突未修复时 lag 调整无效）。
[#R3-4] [3-chunk 任务批处理]（编排层主线 1，R2-4 结构延伸）:
      **条件触发** — R3-1/2/3 落地后 msprof 复测：若每批固定开销（节流
      协议 + 循环控制 + 标记组）仍显著（scalar 非 mte2-stall 部分占比
      高），3-chunk 摊薄再评估；若墙已移至 MTE2 busy 本身（~704us 级），
      批处理不改变 MTE2 字节 → 无收益，跳过。UB 需 +1 staging 集
      （~72KB/vid，R2-6 腾挪余量待编译验证）。
[#R3-5] [V 侧结构]（编排层主线 4）:
      **暂缓** — AIV wait 98.5% 空转，V 非墙；仅当 R3-1 修复后 V 交付
      仍慢（C relay 等待未收敛）时重估（V 的 mte2 31GB/s strided 读、
      gate 链串行化、MTE3 drain 延迟）。
[R2 已证伪清单] O 写回迁 V / hi-lo 单中继 / K-concat / bf16 掩码 /
      gate 融合 / 去 hi-lo 双 gemm / block_DV=64 / PTO / threads 裸切换 /
      L1 同名单缓冲：**禁止重试**（R2 log §未走通）。

## R3 Part B [ORDER-PLAN]

1. [#R3-1] flag ID 冲突修复 — 前置依赖: 无 — 理由: 一行级试错、信息量
   最大（裁决计数器共享假设）、且是后续一切节流语义的前提。
2. [#R3-2] C 侧全批装载上移 — 前置依赖: #R3-1 — 理由: relay flag 预置
   是上移不劣化的前提；攻击 646us mte2-stall。
3. [#R3-3] lag 2→3 — 前置依赖: #R3-1 — 理由: 上移后 C 在体首即取全部
   relay，交付裕量需加深；与 R3-2 可同会话 A/B 但独立归因。
4. [#R3-4] 3-chunk 批处理 — 条件触发（R3-1..3 后画像复判）。
5. [#R3-5] V 侧结构 — 条件触发。

## R3 Step 4-5: 逐项实施与验证记录

```
[ORDER-CHECK] 准备实施: [#R3-1] 手动节流 flag ID 12→40 | 前置依赖: 无 | 结论: ✅
[IMPL-R3-1] 已阅读 src/transform/ascend_combinecv.cc L373-400（ID 分配）+
    CrossCoreSetFlag 官方文档（计数器语义）；关键约束: auto ID 0-31 占用、
    手动须在其外、每 ID 单计数器。
[RESULT-R3-1] flag 12→40（4 set + 2 wait 站点）| 精度 pass | 1.513 → 见下
```
（逐项结果随迭代追加）

## R3 结果汇总（RESULT 补全）

```
[RESULT-R3-1] flag ID 12→40（冲突修复）| 精度 pass | 1.513 → 1.501 | -0.8%
              （relay 等待 780→570us；正确性修复为主，采纳）
[ORDER-CHECK] 准备实施: [#R3-2a] 深度节流裁决探针 | 结论: ✅
[RESULT-R3-2a] 8 裸 set（lag~10 信用）| 设备死锁——计数器容量 <~10 pending；
              EXTRA=4（lag~6）| 1.516 | 0 → **裁决：V 生产受限**；NSLOT=32
              单独 | 1.519 | 0（回归 NSLOT=8）
[RESULT-R3-ablation] 消融图谱：V 数学 0 / V 装载 0 / V 拷贝 ~0 / C gemm -30 /
              C h-装载 -13 / C FIX -19 / a-relay 往返 -33 —— **每任务计算
              几乎免费，relay 往返 + 同步协议 = 每任务 ~3.4us 地板**
[RESULT-R3-2] v-relay 上移 + prefetch 拆分 | 精度 pass | 1.501 → 1.489 | -0.8% 采纳
[RESULT-R3-3] v 装载上移（早装载）| 精度 pass | 1.489 → 1.497 | 0 采纳（结构）
[ORDER-CHECK] 准备实施: [#R3-4] 3-chunk 批处理 | 前置: R3-1 ✅ | 结论: ✅
[RESULT-R3-4] 3-chunk 批处理（staging×3 +36KB/vid、L1×3、NSLOT=9、
              THROTTLE_FLAG_ID→48）| 精度 pass（--level all 全量）|
              1.497 → 1.484（稳态 1.417→1.40，7+ 次运行分布不重叠）| 采纳
              （中途 FIX-消融快照事故已修复并复验 seed2102 PASS）
[RESULT-R3-5] C 侧全批装载上移 | 1.484 → ~1.478 | 0 → 弃用（V 定步，C 装载
              顺序无关；最小改动保留原结构）
```

## R3 最终
- 官方 perf：1.486/1.510/1.497（代表 ~1.498，稳态 ~1.40）；R3 内 -15us（-1.0%）
- [PERF_INCOMPLETE]：目标 1320us 未达（超 13.5%）
- 采纳清单：R3-1（flag 冲突修复，正确性）、R3-2/3（V 体重构，中性）、R3-4（3-chunk，小幅）
- 弃用：R3-5（C 装载上移，0）
- 未走通（本轮新增）：深度节流（计数器容量死锁）、relay 合并（zN 布局，R2 结论复认）、
  4-chunk（UB 预算 + 3-chunk 边际收益递减）
