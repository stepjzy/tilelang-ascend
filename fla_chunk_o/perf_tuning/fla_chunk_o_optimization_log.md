# fla_chunk_o Stage 3 优化记录（optimization_log）

> 环境：Ascend 910B3 (64GB HBM, 20 AI Cores 可见) / CANN 9.2.0 / tilelang editable @ 2f22c6ac。
> 计时口径：NPU event，warmup=10，rep=10，均值，仅设备侧（与 test_fla_chunk_o.py --level perf、
> 上游 do_bench 完全一致）。快速迭代用 perf_tuning/bench_iter.py（同口径 + l0_repr 快速精度门）。

## Step 1-2: 基线与算子类型

- 基线（本会话实测，2026-10-22）：**3.285 ms**（稳态 ~3.20 ms），per-rep
  [4.029, 3.203, 3.217, 3.199, 3.189, 3.207, 3.195, 3.195, 3.208, 3.204]。
  Stage 2 移交值 3.263 ms，复现一致（差 <1%，噪声内）。
- 算子类型：**CV 混合型**（生成代码同时含 `ASCEND_IS_AIC` 与 `ASCEND_IS_AIV`，
  `KERNEL_TYPE_MIX_AIC_1_1`；kernel_source_iter1.c:10/45/70）。

### 瓶颈定位（生成代码证据 + 定量分析，kernel_source_iter1.c）

1. **V↔C 交换全部经 GM workspace 中继**（4 个）：qg_hi→ws1、qg_lo→ws2、
   a_mask→ws3（**循环内写 2 次、AIC 只读 1 次**）、o_frag(fp32)→ws4。
   每块中继 152KB，8192 块 ≈ **+1194MiB 逻辑流量**（唯一 IO 960MiB 的 2.3×）。
   中继数据 L2 命中概率高（20 并发块 × 152KB ≈ 3MB 工作集 << 192MB L2），
   HBM 压力 ≈ 960MiB/3.2ms ≈ 300GB/s（25%）→ **非 HBM 带宽瓶颈**。
2. **Pipelined(num_stages=2) 实际未软件流水**：生成代码是单缓冲顺序循环
   （`for i in 0..2`，固定 buffer offset，无 stage 轮转），每次迭代的 q/g MTE2
   加载延迟被 `WaitFlag<MTE2_V>` 完全暴露（load issue → wait → V 计算 → 3×MTE3 中继写）。
3. **跨核 flag 往返密集**：每块 ~10 次 CrossCoreSet/WaitFlag（4 V→C 中继 + 1 C→V），
   每次 ~100-200ns 量级；加上每块 10 次 MTE2/MTE3/V HardEvent 对。
4. **V 侧指令链长**（每块）：A 掩码链 5 op（含 2 次 fp32↔bf16 cast）+ 循环 ×2 的
   gate 链（4 op）+ hi/lo 分裂链（4 op）+ 输出链（2 op）≈ 830KB UB 流量/块。
5. **DMA 粒度**：q/A 每 slice 行 128B（64 行，行距 8-16KB），g 256B，v/o 256B；
   bshd 布局决定，无法改布局，但**可以减少 DMA 次数**（合并不切片的搬运）。

结论：**瓶颈类型 = sync/延迟型（非带宽）**。每块 7.8us 中，暴露的 MTE 延迟 +
跨核 flag 往返 + 未流水的 V 链占主导。达标需每块 ≤3.22us。

## Step 3: Part A 优化点清单

[#1] [threads=1→2]（DESIGN R9；antipatterns "launch core 数"；先例
     examples/developer_mode/flash_attn_bshd_developer.py:47 —— 同为 Developer 模式
     CV 融合 + PASS_CONFIGS 四件套 + Pipelined + gemm_v0 + V→C copy，threads=2 交付）:
     **适用** — AscendVidReduction 将 UB buffer 首维减半、每 block 拆 2 个 AIV
     sub-block（vid 0/1）并行处理，V 侧链路时间期望 ~减半；本算子无 gather 链
     （fla_wy_fast threads=2 阻塞根因），风险低于先例。
[KERNEL-REUSE-PRECHECK]
candidate: threads=2（无 reference 章节，DESIGN R9 + flash_attn_bshd_developer 先例）
expanded_domain: 无（grid/shape/dtype 均不变，仅 block 内并行度变化）
compatibility_dimensions: UB 减半后峰值 ~72KB<192KB ✓；gemm 操作数 M=64 需双 vid
  拼回（workspace vid offset 机制自动处理）；out_idx/out 索引不变
unknown_or_false: vid 拼回后 gemm 数据正确性（需 L0 实测）
plan: repair-kernel-first（先 L0 精度验证再采纳入基线）

[#2] [i_k 循环展开为直线全 DK 代码]（optimization-guide §2.5 小数据块合并载入 +
     §2.2 Double Buffer 的替代——工具链 Pipelined 失效后以展开换大块 DMA）:
     **适用** — 去掉 Pipelined 循环后：q [64,128] 单次 DMA（行 256B）替代 2×[64,64]
     （行 128B）；g [64,128] 512B 行；h [128,128] 连续 32KB 单 DMA；V 链一趟；
     中继 2 次（hi/lo 全宽）替代 4 次；gemm 2 次 K=128 替代 4 次 K=64（Cube 效率↑）；
     跨核 flag 减半。UB 峰值估算 ~136-144KB < 192KB（lifetime 复用后）。
     风险：无循环的直线 V→C 交换无逐字先例（现有先例：循环体内 V→C ✓、循环后
     C→V ✓）；编译错或 mis-sync 则回退循环版。
[KERNEL-REUSE-PRECHECK]
candidate: i_k 展开直线化
expanded_domain: 无（同 shape/dtype；仅循环结构消除）
compatibility_dimensions: UB 峰值 136-144KB ✓（MEMORY_PLANNING lifetime 复用）；
  L1 峰值 qg_hi+lo 32KB + H 32KB + A 8 + V 16 = 88KB < 512KB ✓；L0C 32KB ✓
unknown_or_false: 直线 V→C 交换的 AUTO_CV_SYNC 行为（先例仅覆盖循环内/循环后）
plan: repair-kernel-first

[#3] [A 掩码 bf16 域化 + Msk 常量 bf16 化]（optimization-guide §2.7 指令融合——
     消除 cast 链）：**适用** — A⊙M 的 M 为 0/1 常量，bf16 乘法精确无舍入；
     省去 a_f32 fp32 域 2 次 cast + 1 次 fp32 mul（~72KB UB 流量/块），Msk 载入
     16KB→8KB。前提：T.tile.mul 支持 bf16 操作数（需查 API/试编译）。
     Msk 是 kernel 内部常量输入（DESIGN §11.5），dtype 变更不影响算子契约。
[KERNEL-REUSE-PRECHECK]
candidate: bf16 掩码
expanded_domain: Msk 内部常量 dtype fp32→bf16（非算子契约输入）
compatibility_dimensions: tile.mul bf16 支持性 unknown（先例均为 fp32 域）
unknown_or_false: tile.mul bf16 lowering
plan: repair-kernel-first（先最小 shape 试编译+精度）

[#4] [Fixed Core 模式：grid=物理核数 + 核内 T.serial 任务循环]
     （antipatterns "关注项 A：任务数高于物理 AI Core 数，按任务数 launch"）:
     **适用（待 #1-#3 后评估）** — 8192 块按任务数 launch 放大块调度/TPipe
     init/地址计算开销，且 workspace 随 block_num 膨胀（~768MiB）；Fixed Core
     后 workspace 按核数分配（L2 必驻）。风险：T.serial 包裹 CV 体的同步点机制
     （AttachSyncStmt 的 stage_loop 条件同步）行为未知；尾块越界需防护。
     plan: repair-kernel-first

[#5] [PTO target]（target="pto"；examples/gemm/example_gemm_pto_developer.py、
     sparse_flash_attn_gqa_pto_developer.py 先例）：**备选** — A2 平台 PTO pipe
     仍经 GM FIFO（TPush.hpp pushVec2GMFiFo），但以 FFTS 硬件同步 + RingFIFO
     替代 flag ping-pong，生产者/消费者解耦（LocalSlotNum=2）。若 #1-#4 后仍
     未达标再试。plan: repair-kernel-first

[#6] [block_DV=64 变体]：**不适用** — DESIGN §10.4-A 已论证 +384MiB 重复读
     （q/g/A ×2），带宽敏感算子反向优化；仅 UB 溢出时降级（当前不触发）。

[#7] [num_stages 微调]：**不适用** — extent=2 下 num_stages=2 已是唯一合理值；
     且实测 Pipelined 未生成 stage 轮转（见瓶颈 2），调 num_stages 无意义；
     #2 展开后无循环，N/A。

[#8] [Host 侧优化]（optimization-guide §2.12 P0）：**已达标** — wrapper 零
     view/reshape/contiguous/aclnn；Msk per-(device,chunk) 缓存 ✓；无内部
     sync ✓（Stage 2 已按上游口径交付）。P0 无剩余项。

[#9] [Vector 反模式扫描]（antipatterns "Vector Core 内逐元素/逐行 for loop"）:
     **无** — 全部 T.tile.* buffer 级指令，无逐元素/逐行循环（Stage 2 交付形态）。

[#10] [CV overlap / pass_configs]（antipatterns "AIC/AIV 混合算子未开启 CV
     overlap"；optimization-guide §2.10）：**已开启** — 四件套全 True；
     §2.10 关自动同步属最后手段（手动 flag 管理风险高），仅在 #1-#5 全部
     失败后考虑。

## Part B [ORDER-PLAN] 实施顺序

1. [#1] threads=2 — 前置依赖: 无 — 理由: 单行改动、V 侧吞吐直接 ×2 期望，
   先拿最大且最便宜的收益；先 dump 生成代码确认 vid 拼回机制再精度验证。
2. [#2] i_k 展开直线化 — 前置依赖: #1（在 threads=2 基线上叠加）— 理由: 消除
   未流水的循环暴露延迟 + 减半中继次数与 flag 往返；#1 采纳后 UB 减半，展开的
   UB 峰值压力减半，顺序耦合。
3. [#3] bf16 掩码 — 前置依赖: #2（在直线化后叠加）— 理由: 消除 A 链 cast；
   独立小改动，放最后单独消融。
4. [#4] Fixed Core — 前置依赖: #1-#3 结果 — 理由: 结构改动最大，仅当仍有
   per-block 开销 dominating 时才值得。
5. [#5] PTO — 前置依赖: #1-#4 — 理由: 换 codegen 路径，验证成本高，兜底手段。

## 消融纪律

每项单独 A/B（bench_iter.py），精度门 = l0_repr 快速精度（ratio≥0.99）；
采纳时跑 --level all 全量复验；回滚项记录原因。基线 3.285ms（本会话）。

## Step 4-5: 逐项实施与验证记录（RESULT 汇总，每步 l0_repr 快速精度门 +
> 采纳时 --level all 全量；perf 为 bench_iter 同口径 ms）

```
[ORDER-CHECK] 准备实施: [#1] threads=2 | 前置依赖: 无 | 结论: ✅
[RESULT-#1a] threads=2 裸切换      | 精度 FAIL(ratio=0.2733,NaN) | 根因: vid 偏移
             落 H 维（读错 head）→ 3D 视图 [B,S,H*D] 修复
[RESULT-#1b] threads=2 + 3D 视图   | 精度 pass | 3.285 -> 3.264 ms | -0.6%（中性，
             作为结构基础保留；揭示锁步：V 减半无收益 → 串行链主导）
[ORDER-CHECK] 准备实施: [#2] i_k 展开直线化 | 前置依赖: #1 ✅ | 结论: ✅
[RESULT-#2]  i_k 展开全 DK          | 精度 pass | 3.264 -> 2.993 ms | -8.3%
[RESULT-probe] skeleton/vec/full 分解: 2.245/2.359/2.993 ms —— 非计算瓶颈
[ORDER-CHECK] 准备实施: [#4] Fixed-Core | 前置依赖: #2 ✅ | 结论: ✅
[RESULT-#4]  fixed-core + 显式 ws   | 精度 pass | 2.993 -> 2.245 ms | -25.0%
[RESULT-#4b] 循环内 if 消除(Select) | 精度 pass | 2.245 -> 2.126 ms | -5.3%
             （AUTO_SYNC 对每个 IfThenElse 前后各插一次 PipeBarrier<PIPE_ALL>，
              生成代码实证；消除后 grep PIPE_ALL = 0）
[RESULT-#5]  早期 a 中继+intra 先行+loads 预取 | 精度 pass | 2.126 -> 1.859 | -12.6%
[RESULT-#6]  手动 flag + 2-late drain | 精度 pass | 1.859 -> 1.691 ms | -9.0%
             （中途实证：纯自动 flag 的 2-late drain 死锁——flag 按出现序 FIFO
              配对；o_relay 改名脱离自动 flag + 手动 ID 按奇偶配对 + 2 peeled）
[RESULT-#7]  o_relay bf16（fixpipe 转换）| 精度 pass(--level all 11/11) | 1.691 ->
             1.670 ms | -1.2%（采纳：省两侧字节；fla_wy_fast 的 fixpipe 损坏
             记录不适用于本中继路径）
[RESULT-#8]  ws1/ws2 读批量化        | 精度 pass | 1.670 -> 1.603 ms | -4.0%
[RESULT-#9]  L1 ping-pong + h/v 预取 | 精度 pass | 1.603 -> 1.621/1.685 ms | 无收益
             → 回滚（工具链保守屏障抵消重排序）
[RESULT-#10] 手动 flag ID 3/4 → 12/13 | 精度 pass | 1.646/1.648 ms | 持平（安全
             修正：自动 flag 占 0..8，消除 ID 冲突风险）
```

## 最终结果

- 官方 perf_bench：**3.285 → 1.638 ms（-50.2%）**；稳态 ~1.55-1.59 ms。
- 目标 < 1.320 ms 未达成（超 24%）；详见 fla_chunk_o_final_summary.md 的
  根因论证与已尝试无收益清单。

## 未走通的路（本轮新增，供工具链修复参考）

1. **AUTO_CV_SYNC flag FIFO 配对**：V 侧对中继 buffer 的第 i 次读必须等
   C 侧第 i 次写 —— 跨任务流水被强制锁步；手动 flag（gdn 先例）可绕过但
   属混合模式。
2. **AUTO_SYNC 的 IfThenElse → PipeBarrier<PIPE_ALL>**：条件执行的中继守卫
   代价极高，逼迫 branch-free Select 化（Select 又引入 scalar condval 链）。
3. **threads=2 vid 偏移对 bshd 4D 切片落在 H 维**（读错 head，静默损坏）；
   3D 视图规避。
4. **wait_cross_flag 的 pipe 参数在 A2 上仅影响资源分类**（codegen 只用
   arg0），且 ResourceForPipe 只认 "V"/"M"/"MTE1"/"FIX"。
5. **中继路径的 fixpipe fp32→bf16 转换实测精确**（与 fla_wy_fast 的直存
   损坏记录并存——损坏限定于直存输出张量路径）。
6. **UB 96% 占用阻塞 2-chunk 任务批处理**（本可再省 ~50% 每 token 开销）。
