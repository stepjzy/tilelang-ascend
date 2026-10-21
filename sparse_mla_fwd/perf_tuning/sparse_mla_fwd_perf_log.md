# sparse_mla_fwd 性能调优日志

目标：DESIGN §9.4 哨兵 < 24288 us（B=1,S=4096,H=128,HKV=1,DQK=576,DV=512,topk=1024,bf16）。
口径：纯 kernel NPU 事件计时（warmup5 + 20 轮），NPU 空闲串行，`source set_env.sh`。
基线（Stage 2 C2 固化）：ascendc 138551±205 us / pto 133774±297 us（接口级 137856/134576 us）。

---

## Iteration 1 — 2026-10-14（画像 + 快赢 A/B）

- bottleneck_type: **transfer（gather/中转事务碎片化）+ sync（C/V 粗粒度串行）复合**；V 侧小算子发射开销（issue-bound）为次因
- 优化实施：无（三项快赢 A/B 全部否决，主文件零改动，见下）
- precision: pass（未改 kernel，无需复验；L0/L1 全绿状态继承 Stage 2）
- adopted: no（本轮为画像轮，基线不变）
- rollback_reason: 三项快赢均无收益（BI64 -81%、ws_o bf16 ascendc 编译失败且 pto 无收益、grid40 噪声级）
- consecutive_no_improvement: 1

### 1. 画像方法

变体 kernel（/tmp/opencode/t3i1/，ascendc 后端，同口径计时）：
- `var_full`：原样拷贝（复核 139407±347 us vs C2 138551±205，+0.6% 噪声级，方法学成立）
- `var_nogather`：成对删除 V 侧 gather 写段 + C 侧 ws_kv 读段（AUTO_CV_SYNC 语句配对保持 0:0）
- topk 缩放：topk=128（NI=1）vs 1024（NI=8）同 shape 输入（indices 前缀切片），线性外推每块边际成本
- 弃用 `var_gatheronly`（成对删除 gemm/softmax 后死锁 aicore timeout——过度裁剪破坏同步图，不作数）

### 2. 画像矩阵（ascendc 纯 kernel，us）

| 变体 | topk | mean±std | 说明 |
|------|------|----------|------|
| var_full | 1024 | 139407±347 | 基线复核 |
| var_full | 128 | 19263±160 | NI=1 |
| var_nogather | 1024 | 38700±168 | 去 gather+ws_kv 中转 |
| var_nogather | 128 | 6666±159 | NI=1 去 gather |

### 3. 定量分解（线性边际，每 task 8 块 + 固定项）

| 分段 | 每块边际 (us) | 总量 (ms) | 占比 |
|------|--------------|-----------|------|
| gather 段（V 逐行 gather + ws_kv 写 + C 段读 + C/V 同步放大） | 12587 | 100.7 | **72.3%** |
| 非 gather 段（QK 5 gemm + PV 4 gemm + softmax 448 逐行小算子/块 + ws_s/p/o 中转 + C/V 串行） | 4576 | 36.6 | 26.3% |
| 固定项（task 循环 / Q 装载 / epilogue 写回） | — | ~2.1 | 1.5% |

核算：8×12587 + 8×4576 + 2100 ≈ 139.4 ms ✓（与实测闭合）

关键回答（本轮任务四问）：
- **① gather+中转 ≈ 100.7 ms（72%）**。机制：每行 6 个微事务（1 次 1152B GM 读 + 5 次段写 256B×4+128B）× 512 行/task/AIV，全部串行发射，MTE 事务碎片化 + 每行标量寻址。
- **② gemm 串行时间不是主项**：纯算 3.66 ms/核@峰值（design §9.5），nogather 段 38.7 ms 中 gemm 占小头；剩余由 V 侧逐行小算子（softmax 链 448 个 [128]-元素行算子/块/AIV ≈ 735K 次/核，~45ns/次的发射开销）与 C/V 串行构成。
- **③ C/V 重叠度 ≈ 0（全串行）**：msprof PipeUtilization（kernel 137.4ms，20 AIC + 40 AIV 全 block 均值）——**AIC 91% 时间在等 V 侧跨核 flag**（wait_id0=ws_kv gather 99.5ms/72% + wait_id3=ws_p 26.0ms/19%），cube 纯算仅 4.3ms（3.1%，与 §9.5 预算 3.66ms 吻合）；AIV 仅 15% 时间等 C 侧。combinecv.cc:527 Warning "Cube loop times (32) not enough to catch up vec (512)"（C 侧 s-环 4×8=32 vs V 侧 j-环 64×8=512，每 task）——同步点无法落在细粒度循环层，退化为每 i-块级 C↔V 往返，V 是唯一关键路径。
- **④ 中转流量与有效带宽**：聚合 GM 流量 ≈ 38 GB（gather 读 4.6 + ws_kv 读写 9.2 + ws_s 4.2 + ws_p 2.1 + ws_o 16.8 + Q 0.6 + O 0.5 GB）→ 聚合有效带宽 ≈ **273 GB/s**（HBM ~1.6TB/s 的 17%）；V 侧单 AIV 539MB/139.4ms ≈ **3.9 GB/s vs 26 GB/s 实测锚点（15% MTE 效率）**——结构性发射效率问题，非带宽饱和。

### 4. 快赢 A/B 结果（全部否决，主文件未动）

| A/B | 配置 | 结果 | 判定 |
|-----|------|------|------|
| BI=64 vs 128 | N_SEG=8,NI=16 vs 4,8 | 252524±165 us（**+81% 恶化**） | 否决——块数翻倍、gemm 减半、事务更碎；BI=128 保持 |
| ws_o bf16（设计档 1） | C 侧 L0C fp32→GM bf16 + V 侧直读 | ascendc **编译失败**（`copy_gm_to_ub` 无 dtype cast 重载，tmpx94imxil.cpp:267）；pto 135360±206 us（vs pto 基线 133774，**+1.2% 噪声级**） | 否决——当前 issue-bound 非带宽-bound，档 1 收益前提不成立；待 iter-2 消除发射瓶颈后若回到带宽-bound 再启（需加 bf16 UB 中转槽绕过 GM→UB cast 限制） |
| grid=40 vs 20 | 40 block（2:1 超订 AIC） | 137384±265 us（-1.5%，<3% 噪声阈） | 否决——910B3 20 AIC/40 AIV 已被 grid=20+vid∈{0,1} 物理吃满，超订仅排队无并行收益 |

### 5. 反模式对照（performance-antipatterns.md）

| 命中项 | 位置 | 严重度 |
|--------|------|--------|
| Vector Core 逐行 for loop（L124） | softmax 链 `for h in range(v_block)` ×3 + `for d: for h:` alpha 缩放（448 行算子/块/AIV） | 高（非 gather 段主因） |
| 稀疏访存逐元素 Gather（optimization-guide §2.8 优化前形态） | gather j-loop：单 buffer 逐行 1 读 + 5 段写 | 高（gather 段主因） |
| AIC/AIV 未流水（L303） | C/V 每 i-块 4 次往返全串行（combinecv 警告佐证） | 高（全程放大器） |
| launch core 数（L60-92） | persistent grid=20 + striped 4096 task（末核偏差 1%） | 无问题 |
| tile size 过小（L267） | UB/L1 利用率低（事务粒度 128-1152B） | 中（随 §2.8 一并解决） |

### 6. msprof PipeUtilization（var_full，ascendc，kernel=main_kernel，137947us，Block Dim 20 / Mix 40）

`msprof op --kernel-name=main --aic-metrics=PipeUtilization --launch-count 1 --launch-skip-before-match 5`（稳态单次 launch，输出 /tmp/opencode/t3i1/msprof_out/）：

| 侧 | 指标 | 值 | 解读 |
|----|------|-----|------|
| AIC×20 | cube 纯算 | 4.3ms（3.1%） | gemm 非瓶颈，与 §9.5 预算 3.66ms 吻合 |
| AIC×20 | scalar wait_id0（ws_kv flag） | **99.5ms（72%）** | 等 V 侧 gather 完成——与画像 ① 的 100.7ms 逐项闭合 |
| AIC×20 | scalar wait_id3（ws_p flag） | 26.0ms（19%） | 等 V 侧 softmax 段（逐行小算子） |
| AIC×20 | mte1/mte2/fixpipe/scalar | 2.4/3.5/2.8/7.7% | C 侧自身负载极低 |
| AIV×40 | **MTE3（写）占用** | **51% = 69.5ms @ 2.15 GB/s** | gather 段 320 个微写/task/AIV 的碎片化代价（对照 MTE2 活动带宽 19.3GB/s——写路径带宽能力并非 2GB/s，是事务太小） |
| AIV×40 | vec / scalar | 10.3% / 13.9% | softmax 逐行小算子 + 逐行标量寻址开销 |
| AIV×40 | wait_id2+id4（等 C） | 8.7+11.9ms（15%） | V 侧几乎不等 C——单向依赖，C 完全可被流水掩盖 |

结论：V 侧 MTE3 碎片写（69.5ms@2.15GB/s）+ V 侧标量/行算子开销是唯一关键路径；C 侧 91% 空转等 flag。P0（gather 整段写出）直接攻 69.5ms 主项；P2（C/V 流水）在 P0/P1 之后只需掩盖 C 的 ~5ms。

### 7. 迭代 2 方向排序（本轮只论证不实施）

1. **P0 gather 批量 staging**（optimization-guide §2.8 优化后形态 + reducesum #15v2 先例）：V 侧 j-loop 仅保留 64 次 GM 行读入 `kv_stage [HALF, DDT]` UB 多槽，块尾整段 MTE3 写出（5 个 [64, DS] 段写 vs 现在 320 个微写）；语句配对不变（V 2 语句 ↔ C 2 语句）。预期 gather 段 100.7 → ~10-15 ms。前置探针：2D UB 列段切片 `kv_stage[:, s*DS:(s+1)*DS]` 作 T.copy 源（R4 只证过 3D 前导维切片与 1D 列切片，需 /tmp 最小复现）。
2. **P1 softmax broadcast 向量化**（反模式 L124 替代写法）：m_i/m_prev/mask broadcast 到 [v_block, BI] 后整 tile 运算，448 行算子 → ~15 个 2D 算子；UB 需腾 ~32KB 临时（acc_s_ub 复用槽重排）。预期非 gather 段 36.6 → ~12-18 ms。与 P0 独立可并行开发。
3. **P2 C/V 跨块流水**（indexer_bwd pair-GEMM 软件流水先例，sparse_mla_fwd.py 同族）：ws_kv/ws_s/ws_p/ws_o 加 ring 维 [grid, 2, ...]，V 领先 1 块产出、C 消费上一块，把串行和折叠为 max(V,C)。前置风险：AUTO_CV_SYNC 自动 flag 配对在 ring 化后是否保持 1 块松弛需探针（indexer_bwd 用同名 workspace 前缀配对已验证可行）。
4. 预期路径：P0+P1 后 ≈ 15-18 + 2 ≈ 17-20 ms；P2 再折叠 → **< 24288 us 达标**（与 DESIGN §9.5 21-23 ms 贴线估算一致且余量更大）。
5. **混合模式升级触发判据（用户已预批）**：P0/P1/P2 逐一落地并排除实现退化后，若仍 > 24288 us（尤其 AUTO_CV_SYNC 在 ring 化下无法给出 1 块松弛、或 C 侧 L0A/L0B 手动 staging + 细粒度 flag 成为必要）→ 升级混合模式（ascendc 专项 fast path + pto 保精度路径）。

### 8. 本轮产物

- 画像变体与驱动：/tmp/opencode/t3i1/{var_full,var_nogather,var_wsobf16,var_gatheronly}.py + bench.py + logs/
- 基线备份：history_version/sparse_mla_fwd_perf_iter1.py（主文件零改动，备份即基线）
- 基线 JSON：perf_tuning/sparse_mla_fwd_baseline_iter1.json
- 主文件 sparse_mla_fwd.py：**未改动**（三项 A/B 均未采纳，无回滚需求）

next_hint: 迭代 2 先做 P0 探针（2D UB 列段切片源合法性）+ P0 实施；P1 并行推进；P2 依赖 P0/P1 落地后的剩余串行形态再定 ring 粒度。

### 9. 事故与测量有效性记录（如实）

- **误杀并行会话进程（16:58）**：清理 gatheronly 死锁残留时，将 PID 1232605 误判为残留并 kill -9——实际是**并行 bwd 会话**的 `test_indexer_topk_reducesum.py --level all`（教训：kill 前应先 `ps -p <pid>` 确认命令行）。该会话本轮测试被中断，需 Orchestrator 知会其重跑验证。
- **本轮测量污染风险评估**：16:12 基线复核前有 idle 确认（idle_check.log）；16:31-16:57 的 A/B 链未逐次重查（失误）。但数据自洽性支持结论有效：① var_full 139407 vs C2 空闲基线 138551（+0.6%）；② msprof 设备侧 task duration 137947 独立吻合（-1.0%，AICore 争用会同时抬高两者）；③ 全部运行 std <0.3%（争用通常呈双峰抖动）；④ 变体间线性关系严格闭合。判定：并发进程大概率处于 CPU 侧阶段（golden/输入构造）或晚于计时窗口启动，本轮结论可用；迭代 2 起恢复"每次计时前 npu-smi 检查"纪律。

---

## Iteration 2 — 2026-10-14（P0：gather 批量 staging）

- bottleneck_type: transfer（gather 写路径碎片化——迭代 1 结论，本轮消除主项）
- optimization: V 侧 gather 批量 staging + 整 2D MTE3 写出（320 碎片写 → 2 大写 + 1 幂等小写）
- baseline_time: 138551 us（ascendc C2 口径）/ 133774 us（pto）
- candidate_time: 81700 us（ascendc）/ 78565 us（pto）
- improvement: **-41.1% / -41.3%**（std 270/308 < 0.4%，远超 3% 噪声阈）
- precision: pass（L0 4/4 + L1 48/0 + L2 4/4 + Boundary 16 过 2 WARN[与基线逐位一致]，双后端）
- adopted: **yes**（新基线 81700/78565）
- rollback_reason: 无
- consecutive_no_improvement: 0（重置）

### 1. 前置探针（/tmp/opencode/t3i2/，双后端）

| 探针 | 结论 |
|------|------|
| ① 切片形式（probe1/1b/1c） | **ascendc：2D UB 列段切片作 T.copy 源损坏**（B/F 形式，j=0 行正确其余 garbage——与 R1 同类的 region offset 缺陷；pto 全形式正确→共享前端判定不成立，为 ascendc codegen 侧）。**可用形式（双后端 PASS）**：整 2D buffer 作源（→2D 列段 GM dest）、1D 行切片源（kv_stage[j, c0:c1]）、2D 行 dest staging（kv_stage[j, :]）、整 2D GM→UB 列段 dest。取模区域索引（j%KB）亦损坏（probe2 row1d，弃用） |
| ② 带宽（probe2，grid=20×40 AIV，NBLK=64） | frag（生产形态 320 微写/块/AIV）**61.14 us/块/AIV**（与实测算子 61.4 us/块闭合，方法学互证）；batch32（2×[32,576] 整 2D 写）**17.11 us（3.6×，8.6GB/s rw）**；分解：indices 链 8.83 + 64 行读增量 7.34（~115ns/op）+ 写 ~1-5 |

### 2. 结构变化（F6b 迭代，语句配对 2:2 → 3:3）

```
[V] gather（每 i-块，每 AIV）                    [C] ws_kv 消费
  for j in 32: KV[rows_int[..j], g,:] → kv_stage[j,:]   wait(flag0)
  T.copy(kv_stage, ws_kv[cid, half:half+32, :])  set →   read ws_kv[0:16,0:16] → l1_dummy   ← 语句1（dummy，仅依赖 batch0）
  for j in 32: KV[rows_int[..32+j], g,:] → kv_stage[j,:] wait(flag1)
  T.copy(kv_stage, ws_kv[cid, half+32:half+64, :]) set →  for s: ws_kv[0:BI, s*DS:..] → v_all[s]  ← 语句2（s-loop，flag1=两批齐）
  T.copy(kv_stage[31,:], ws_kv[cid, half+63, :])  set →   ws_kv[0:BI, 512:576] → tail_l1     ← 语句3（幂等重写，配 tail 读）
```

**关键发现（本轮最大坑，记入知识面）**：AUTO_CV_SYNC 按序配对的语义是**每条 C 语句只能依赖其配对（及更早）的 V 语句**。首版 2:2（V: batch0/batch1 整 2D 写 ↔ C: s-loop/tail 读）导致 C 的 s-loop 读（需全部 128 行）在 flag0（仅 batch0）后即启动，与 batch1 写竞态——双后端同源确定性 97% 损坏（smoke 8.2M mismatch，两后端逐位同值）。修复 = 上述 3:3（C 前置 dummy 读 + V 尾部幂等单行重写，各 +1 语句）；生成代码核验：wait(0)→dummy、wait(1)→s-loop、wait(2)→tail 顺序正确。代价：每块 +1 对跨核 flag。

### 3. UB 预算表（生成代码实测，ascendc）

| buffer | 大小 | 偏移 | 说明 |
|--------|------|------|------|
| acc_o_all | 128KB | 0 | 常驻（跨 i-块累加） |
| 1D 链（m/sumexp/idx/mask 等） | ~4.6KB | 131072-134912 | 常驻或链期 |
| **kv_stage [32,576] bf16** | **36KB** | **135168** | **与 acc_s_ub 同址（planner 生命期别名）** |
| acc_s_ub / cast_bf16 | 32KB/16KB | 135168/172288 | softmax 期活跃（gather 期死） |
| tmp_ub / sumexp_i | 16KB/256B | 172288/188672 | |
| **峰值** | **188928B ≤ 196352B** | | 7.4KB 余量，无 overflow 警告 |

取舍记录：全量 staging [64,576]=72KB 不可行（gather 期可别名池仅 ~50KB：acc_s_ub+cast_bf16+小件；acc_o_all 常驻不可让）。GKB=32 两批 + MEMORY_PLANNING 生命期别名（skill 指南 §2.11 明示支持；sync insert 在 planning 之后按地址插屏障，生成代码核验 MTE3_MTE2 链在位）。BI=64 时 GKB=16 同构（新增 assert HALF%2==0 即 BI%4==0）。

### 4. 精度复验（改后全量）

| 层 | 结果 |
|----|------|
| L0（small+main × 双后端） | 4/4 PASS，matched_ratio=1.0000，max_abs 1.562e-2/1.431e-6 与 Stage 2 基线逐位一致（数值轮廓不变） |
| L1（48 例含三组 gather 关键） | l1_repeat 0.9997 / l1_neg_oob 0.9998 / l1_short_segments 1.0000，双后端全过 |
| L2 + Boundary | 4/4 + 16 过 + bnd_dbound 2 WARN（0.9312/0.9315，与 C2 基线逐位一致，非阻塞） |

语义六注意点核验：clamp 寻址保护 ✓（rows_int 链未动）、mask 置 0 贡献 ✓（mask 链未动）、Indices 任意分布 ✓（逐行标量寻址保留）、重复索引 ✓（l1_repeat）、-1/越界 ✓（l1_neg_oob）、首行自注意 ✓（mask 数学未动）。

### 5. 计时（DESIGN §9.4 哨兵，纯 kernel NPU 事件 warmup5+20，NPU 空闲串行）

| 后端 | iter1 基线 | iter2 | Δ | interface |
|------|-----------|-------|---|-----------|
| ascendc | 138551±205 | **81700±270**（min 81328/max 82126） | **-41.1%** | 81952±380 |
| pto | 133774±297 | **78565±308**（min 78140/max 79176） | **-41.3%** | 78949±334 |

vs 24288 us 预算：OVER 336.4%/323.5%（原 570.5%/550.8%）。测量前 npu-smi 空闲确认（本机 idle 基线 AICore 读数 ~88% 无进程；忙时 96%+ 有进程——session 起始即观测到该 driver 读数特性）。

### 6. 分解更新（估算，下轮重画像校准）

旧：gather 100.7 + 非 gather 36.6 + 固定 2.1 = 139.4ms。新实测 81.7ms；probe 折算 gather ≈ 17.11us/块 × 1640 块/AIV ≈ 28ms（若成立则非 gather ≈ 51ms，与旧 36.6 不闭合——差额疑为 C/V 同步放大随块耗时变化的重分布 + 第 3 对 flag + 真实 kernel 链比 probe 链多 ~9 个向量 op）。诚实口径：gather 段 ∈ [28, 43]ms，**非 gather 段（softmax 448 逐行小算子/块 + 链 + C/V 串行）已成为或即将成为主项**。

### 7. 事故与测量有效性记录（如实）

- **l1 首跑被 cgroup OOM 静默 SIGKILL**（17:57-17:59，日志无 traceback 中断；同窗 dmesg 有 rsync/apt-get OOM 事件；触发条件：与并行 bwd 会话 diag_l1.py 编译期叠加）。处置：等其退出后串行重跑 → 全过。**本轮未 kill 任何非本会话进程**（上轮教训执行）。
- bwd 会话 diag_l1.py 两度运行（18:00/18:06，timeout 600 包装）期间未启动计时；perf 计时前确认其退出 + npu-smi 无进程。

### 8. 本轮产物

- 探针与日志：/tmp/opencode/t3i2/{probe1_slices,probe1b_isolate,probe1c_fallback,probe2_bw,probe2b_decompose}.py + logs/ + kernel_src_ascendc.cpp（3:3 版生成代码）
- 主文件 sparse_mla_fwd.py：gather 块重构（3:3 语句 + kv_stage + l1_dummy），ruff check 过（format 与仓库现状一致）
- 备份：history_version/sparse_mla_fwd_perf_iter2.py（改前版本=iter1 基线）
- 基线 JSON：perf_tuning/sparse_mla_fwd_baseline_iter2.json

next_hint: 迭代 3 优先重画像（var_nogather 同法差分 + msprof）定位新主项，预期排序：P1 softmax broadcast 向量化（448 行算子/块 → ~15 个 2D 算子，反模式 L124）≈ 可观；indices/mask 链精简（~16 个 [128] 向量 op/块，probe 链实测 8.83us/块）；P2 C/V 跨块流水（ring 化 ws_*，吸收 C 侧等待）；gather 读侧（64×1152B MTE2 ~7.3us/块）与 staging ping-pong（MTE2/MTE3 重叠，需防取模索引坑）次之。

---

## Iteration 3 — 2026-10-14（P1：softmax/行算子向量化 + mask 链精简）

- bottleneck_type: compute（V 侧逐行小算子发射开销——本轮消除主项）
- optimization: softmax 链 448 行算子/块 → ~90 个组级 broadcast/2D 算子（T.tile.broadcast axis=0/1 + 行组区域算子）；mask 链 11 → 8 op（min 形态 + 2-op bias）；reduce 拆组（16KB workspace → 2KB）
- baseline_time: 81700 us（ascendc）/ 78565 us（pto）（iter-2 采纳基线；本轮 var_full 复测 81753±254，+0.06% 噪声级，口径闭合）
- candidate_time: **64437±211 us（ascendc）/ 75635±194 us（pto）**
- improvement: **-21.1% / -3.7%**（std 211/194，远超 3% 噪声阈；pto 增益小——其后端逐行算子本就较便宜或 broadcast 较贵，下轮画像复核）
- precision: **pass**（--level all "Test Passed!"：L0 4/4 双后端 max_abs 1.562e-2/1.431e-6 与基线逐位一致；L1 48/0；L2 4/4；Boundary 16 过 + 6 WARN = bnd_extreme ×4（1.28e+02/1.92e+02/0.9999/1.0000，与 C1 定性逐位一致）+ bnd_dbound ×2（0.9312/0.9315，与 iter-2 逐位一致），WARN 口径零恶化）
- adopted: **yes**（新基线 64437/75635）
- rollback_reason: 无
- consecutive_no_improvement: 0（重置）

### 1. 重画像（先画像后调优，var 链 ascendc 同口径 warmup5+20）

| 变体 | 均值±std (us) | 差分含义 |
|------|--------------|---------|
| var_full（iter-2 基线复核） | 81753±254 | 与 iter-2 记录 81700±270 闭合 |
| var_nogather（成对删 gather+ws_kv 3:3→0:0） | 38360±185 | gather 段 = **43393 us（53.1%）** |
| var_nosm（+删 softmax 逐行链 448/块 + 尾 div 256/task） | 16598±169 | softmax 行算子链 = **21762 us（26.6%）** |
| var_nomask（+删 mask 1D 链 11 op/块） | 16552±160 | mask 链 = 46 us（噪声内；probe2b 的 8.83us/块 kernel 级未复现） |
| 残差（nomask 本底） | 16552 | Q 装载 + gemm + ws_s/p/o 中转 + softmax 状态 op + epilogue + task 循环 |

四问回答：① softmax 逐行链 21.8ms（26.6%）——本轮主项，已消除至 ~4.5ms 残差；② mask/indices 链 ~0（46us，噪声）——上轮 probe 估算在 kernel 级不成立，精简仅保留语义等价收益；③ gather 段 43.4ms（53.1%）成为绝对主项（C 侧等 flag 随之）；④ 新分解闭合：43.4 + 16.5 + 4.5 ≈ 64.4ms ✓（与候选实测 64437 逐项吻合）。

### 2. 前置探针（/tmp/opencode/t3i3/，双后端）

| 探针 | 结论 |
|------|------|
| probe_broadcast（全尺寸 [64,128] bcast temp 方案） | 语义全对但 **UB 放不下**：acc_o_all 128K + acc_s_ub 32K + [64,128] fp32 32K = 196,608B > 196,352B 限（三大 buffer 互斥，结构性不可行） |
| probe2（probe_broadcast 失败定位） | **平台坑①：整 3D GM→UB T.copy 只拷第一个 2D chunk**（ascendc 生成 `copy_gm_to_ub<128,64>` 仅 8192 元素；pto 同症状 d=1 全零）——探针 staging 专用坑，主 kernel 无此形态（2D region 拷贝安全） |
| probe2 变体 A/B/C 复盘 | 排除 div/3D-region/临时复用嫌疑——全为 staging 坑①所致 |
| probe3（最终形态，8 行组 [8,128] temp） | **PASS**：axis=0/axis=1 broadcast（含 1D 切片源）、2D 行组区域 in-place mul/sub、3D 中间维区域 mul、临时覆写、整 2D exp——双后端 × 3 重复全部 bit-exact |

### 3. 实施形态（三次编译失败后落定的 UB 预算约束）

- 反模式闭环：本轮消除 performance-antipatterns.md L124「Vector Core 内逐元素/逐行 for loop 计算」（iter-1 §5 表格主项），替代写法即文档 L135「broadcast + 整 tile 计算」；L146「broadcast 需额外 UB 空间」的代价警示恰为本节 4KB 组级 temp 约束的由来（skill 参考文档预判命中）。iter-1 §5 其余命中项（gather 碎片化、C/V 未流水）仍开放，转入迭代 4。
- bcast_h [EG=8 行, BI] fp32（4KB @BI=128）——**尺寸是 UB 预算倒逼**：16KB 版 planner 溢出（"tmp_ub required 16384, available 7680"——reduce workspace 16KB 活跃于任何跨越 reduce 的 buffer 生命期，尾部仅 7.4KB）；8KB 版差 512B；4KB 版通过。
- **平台坑②：VEC UB 对齐**——1D 切片 broadcast 源必须 32B 对齐：EG=4（v_block=32 即 kv_group=2）时 m_prev[g*4:...] 偏移 16B，双后端 aicore exception "UB address not aligned"（iso 复现）；EG=8 固定行组 + NGRP=v_block//8 修复，l1_kvgroup2 恢复且与备份逐位一致。
- reduce_max/reduce_sum 拆为 NGRP 组调用（region src + 1D 切片 dst）——压 tmp 16KB→2KB，腾出 bcast_h 空间；逐元素语义不变（每行 max/sum 在自身行组内）。
- mask 链 min 形态：mask = clamp01(min(idx+1, s_i+1-idx))，11→8 op（数学等价证明：窗口内两条件 ≥1、窗外任一 ≤0，min-clamp 精确；大 |idx| fp32 舍入鲁棒性与旧形态一致）；bias = mask*(-BIGC)+BIGC（2 op，值恒等 {0, BIGC}）。
- 尾 div 保留逐 h（每 task 一次，频次 1/8，收益 ~7% 链时——不动）。
- 算子数/块：softmax 448 → ~74 组级 op（18 mask + 16 m_i + 40 alpha；含 broadcast）；语义逐元素序不变 → **输出与基线逐位一致**（L0/L1/Boundary max_abs 全部逐位复现；smoke 1274-mismatch 为该 config 容差误报，备份 kernel 同值 1274，已验证）。

### 4. 精度复验（--level all，双后端）

56 PRECISION_PASS（L0 8 + L1 48）/ 0 FAIL；16 BOUNDARY_PASS；6 WARN（bnd_extreme ×4 + bnd_dbound ×2，全部与基线逐位一致，口径零恶化）。语义六注意点：clamp 寻址 ✓、mask 置 0 贡献 ✓（l1_neg_oob 0.9998 同基线）、Indices 任意分布 ✓、重复索引 ✓（l1_repeat 0.9997 同基线）、-1/越界 ✓、首行自注意 ✓。kv_group=2/H=64/BI=64/sm_scale=0.5/valrange 全过（EG=8 重排后 v_block=32/16 路径修复验证）。

### 5. 计时（DESIGN §9.4 哨兵，纯 kernel NPU 事件 warmup5+20，NPU 空闲串行确认）

| 后端 | iter-2 基线 | iter-3 | Δ | vs 24288 us |
|------|-----------|--------|---|------------|
| ascendc | 81700±270 | **64437±211** | **-21.1%** | OVER 165.3%（原 236.4%） |
| pto | 78565±308 | **75635±194** | **-3.7%** | OVER 211.3%（原 223.5%） |

测量纪律：bwd 并行会话 19:15 L0 测试全程等待未并行计时（未 kill 任何进程）；每次计时前 npu-smi 确认无进程。

### 6. 迭代 4 方向排序（按新画像）

1. **P2 C/V 跨块流水（ring 化）**：gather 段 43.4ms = 新总量的 **67.3%**，绝对主项。ws_kv/ws_s/ws_p/ws_o 加 [grid, 2, ...] ring 维，V 领先 1 块产出、C 消费上一块，串行和折叠为 max(V,C)。前置探针：AUTO_CV_SYNC 在 ring 化后的配对语义（indexer_bwd 同名前缀先例）。若成，理论下限 ≈ max(gather 43.4, C 侧 ~16.5) + 1 块松弛 ≈ 45ms 级，需配合下述压缩 gather 本体。
2. **gather 段本体压缩**：43.4ms 中 probe 折算批量写 ~17ms/全核，读侧 64×1152B MTE2 ~7.3us/块 ≈ 3ms/核 + 3:3 flag 往返放大（iter-1 msprof: C 侧 91% 等 flag 的结构在 gather 段内依旧）——ring 化后残余为纯带宽项，届时 ws_o bf16（16.8→8.4GB，档 1）重新入池。
3. **pto 增益异常排查**（-3.7% vs ascendc -21.1%）：pto 的 broadcast/行组形态可能有更优写法（row_expand_*_experiment 族 pto 原生 TROWEXPANDMUL 单指令），若 pto 为交付主后端则优先。
4. 预期路径：P2 + gather 压缩后 ≈ 30-40ms；若 ring 化被同步语义卡死（AUTO_CV_SYNC 无法 1 块松弛）→ 触发用户已预批的混合模式升级判据（ascendc 专项 fast path + pto 保精度路径）。

### 7. 事故与测量有效性记录（如实）

- smoke_modified.py（t3i2 遗留脚本）对 S=128/H=128/TOPK=128/seed3 配置**容差误报**：备份 kernel 同样 1274-mismatch/max_abs 1.562e-02——已用备份对照排除回归，改以 L0/L1 为准（教训：smoke 容差未对该 config 校准，沿用需先跑备份基线）。
- 三次编译/运行失败（16KB bcast_h planner 溢出 → 8KB 差 512B → EG=4 对齐崩溃）全部收敛为可复现平台约束（UB 预算 + VEC 32B 对齐），已固化在 kernel 注释与本日志。
- alpha 循环重排（d-outer）由 smoke 误报触发，最终验证两种循环序均与基线逐位一致——重排保留（probe3 同构形态），注释已如实更正。
- 未 kill 任何非本会话进程；bwd 会话测试窗口主动等待。

### 8. 本轮产物

- 探针与日志：/tmp/opencode/t3i3/{probe_broadcast,probe2_div,probe3_patterns}.py + logs/ + kernel_src_iter3_ascendc.cpp + iso_kvgroup2.py + var_{full,nogather,nosm,nomask}.py + bench.py
- 主文件 sparse_mla_fwd.py：softmax 向量化 + mask 链 min 形态 + reduce 拆组（ruff check/format 过；注释即文档）
- 备份：history_version/sparse_mla_fwd_perf_iter3.py（改前版本 = iter-2 采纳基线）
- 基线 JSON：perf_tuning/sparse_mla_fwd_baseline_iter3.json（含画像矩阵与候选结果）

next_hint: 迭代 4 主攻 P2 ring 化（gather 段 43.4ms/67.3%）+ AUTO_CV_SYNC ring 配对探针；次选 gather 读侧批量与 ws_o bf16 复评；pto 增益异常排查视交付后端优先级。

## Iteration 4 — 2026-10-14（P2 ring 探针：AUTO_CV_SYNC 配对语义 + 平台根因链；次选 ws_o bf16 复评）

- bottleneck_type: sync（C/V 逐块 lockstep 无法在 AUTO_CV_SYNC 栈上解除——本轮探针结论）
- optimization: 探针主导（ring 未落地）；次选 ws_o bf16（读侧 UB hop 绕开 copy_gm_to_ub 无 cast）已评估未采纳
- baseline_time: 64437 us（ascendc）/ 75635 us（pto）（iter-3 采纳基线，主文件未动）
- candidate_time: ws_o bf16 变体 63683±286（ascendc，-1.17%）/ 72023±252（pto，-4.78%）
- improvement: ascendc -1.17%（< 3% 噪声阈，不采纳）；pto -4.78%（真实但非主后端）
- precision: 主 kernel 未改动（iter-3 pass 状态延续）；ws_o bf16 变体 L0 8/8 双后端 pass（out max_abs 3.125e-02，基线 1.562e-02）
- adopted: **no**（ascendc 增益低于噪声阈 + 精度余量代价；基线 64437/75635 保留）
- rollback_reason: 无需回滚（主文件本轮零改动，md5 与 history_version/sparse_mla_fwd_perf_iter4.py 一致）
- consecutive_no_improvement: 1

### 1. ring 探针（/tmp/opencode/t3i4/，gen_probes.py 模板生成 27 个探针模块，双后端）

判据：ring 调度与 lockstep 参照（同 slot 布局）输出**逐位一致** + 3 重复稳定 + host golden allclose + 不挂死；NI ∈ {8,5,3,2,1}（偶/奇/极小）。
**结论：RING 被 AUTO_CV_SYNC + AscendMemoryPlanning 栈的五层独立机制封死**（全部源码级验证，非猜测）：

| # | 根因 | 源码证据 | 探针证据 |
|---|------|---------|---------|
| RC1 | **flag 硬件预算 16**：`CrossCoreSetFlag` → `GetffstMsg(flagId & 0xf)`（dav_c220/kernel_operator_sync_impl.h L124-127）；AutoInsertCrossCoreSync 按静态语句对分配全局递增 id | 生成代码 flag id 集合：主 kernel 6 个 ✓；rb4-NI3 36 个 → 截断别名 → 大面积损坏（maxdiff 48+，88% 元素错） |
| RC2 | **配对按静态 TIR 语句 + 循环乘数匹配**（FindTargetLoopDepth "loop times"）：V/C 语句须在同名同 extent 循环巢内才逐语句轮转；循环外生产者 × 循环内消费者 → 首迭代后 flag 陈旧 | V/C 双独立循环（不同 extent）→ 整循环级挂载 → 死锁（pass WARNING "loop times not enough" + 源码走读）；[p,sl] 共享巢 → 中段对逐语句轮转 ✓ |
| RC3 | **planner 单区间生命期**（每 buffer 一个 [首触,末触] 区间；循环体只记录一次；循环外触达把 kill 延伸到循环尾）：环形交错使 kv_stage 区间跨 softmax scratch → iter-2 的 kv_stage↔acc_s_ub 别名被撤销 → UB 217.6KB > 196.35KB | ra2（交错 8 行批）/ra4（sm 相位→gather 相位）/rb3（相位 scratch 分离）全部 UB 超限（"Memory allocation failed for kv_stage required 36864, available 11520"）；lock0 同布局别名成立（生成布局 kv_stage@131072 == acc_s_ub@131072）证明区间模型差异是环形结构性的 |
| RC4 | **planner 两个 bug**：`findReusableBlock` 空 `free_blocks` 取 `back()`（UB 超限→段错误而非清晰报错）；`buffer_map_` 存指针后 `std::sort` 失效（悬垂，潜伏） | gdb 栈：`AscendMemoryPlanner::LinearScanAllocator::allocate` SIGSEGV（ra2/ra4）；修复 RB3 后转为干净 LOG(FATAL) |
| RC5 | **Mod 区域索引 = 硬件 aicore exception**（独立于 kv_stage 竞态复现；t3i2 坑再确认）→ 3+ 槽环（循环槽位算术）不可表达 | lock(Mod) 在 RowBlock 修复后仍 `aicore exception`（rtDeviceSynchronizeWithTimeout） |

**正面成果**（对主 kernel 有直接价值）：
- **RowBlock/PIPE_ALL 负载模式发现**：gather 行地址含 UB 标量读（`indices_ub[...]` GetValue）→ `is_sliced` → 核内 pass 在每行拷贝前插 `PipeBarrier<PIPE_ALL>`（排干上一迭代 MTE3 对 kv_stage 的读）。纯算术地址只得 `PIPE_MTE2` → 跨迭代 kv_stage WAR 竞态（NTASK>GRID 必现 NaN，NTASK==GRID 干净——探针二分证据）。**主 kernel 现有 rows_int 寻址是正确性的隐性依赖，重构 gather 时必须保留**（已在本节固化）。
- lock0（lockstep+RowBlock）NI∈{8,5,3,1} 双后端全过；ra4-NI2（无循环 ring 形态）通过——flag 装得下时逐语句轮转语义本身成立。
- rc（clamp-ring，RowBlock 宿主侧夹断越界 gather，静态语句数不变，12 个 flag ≤ 16）通过语句数检查——仅被 RC3（UB 别名）卡住。

### 2. 混合升级判据（用户已预批 Expert 手动 flag 为终极解）——**触发**

证据链：ring 的理论收益（串行和 → max(V,C)，≈45ms 级）需要 (a) ≥2 槽轮转 flag、(b) 跨迭代 flag 轮转、(c) 32KB staging 与 softmax scratch 的 UB 别名。AUTO 栈五层机制（RC1-RC5）中任一单独成立即封死该路径；Expert 模式手动 set_flag/wait_flag（每迭代显式轮转 flag id、手动 UB 布局）可同时绕开 RC1（flag 显式分配 ≤16 轮转复用）、RC2（无静态配对约束）、RC3（手动地址布局无 planner 依赖）。**建议迭代 5 起启用混合模式升级**（ascendc 专项 fast path），或先做 planner/pass 侧修复（RC4 两 bug 为纯 bug，RC1/RC3 为设计约束）。

### 3. 次选方向评估：ws_o bf16 复评（档 1，16.8→8.4GB transit）

- iter-1 阻塞（ascendc `copy_gm_to_ub` 无 cast）**原样复现**于直连路径；绕法：GM→UB 同型读入 cast_bf16 + UB→UB cast（`indices_ub→idx_f` 先例）→ **双后端编译通过、L0 8/8 pass**；C 侧 `copy_l0c_to_gm` fp32→bf16 cast 写直连可用。
- 计时（哨兵 20 轮）：ascendc 63683±286（**-1.17%**，< 3% 阈）；pto 72023±252（**-4.78%**）。
- **不采纳**：主后端增益低于噪声阈；精度余量代价（L0 out max_abs 1.562e-02 → 3.125e-02，bf16 加数舍入）。pto 增益留档——若 pto 为交付主后端可复评（配合 L1/Boundary 全量门禁）。
- 原因分析：ws_o transit 在 lockstep 下与 gemm/flag 等待重叠度高，字节减半只回收 ~0.75ms；额外 UB→UB cast hop（32KB×4/块）部分抵消。ring 化（若混合模式解锁）后 transit 占比上升，届时复评价值更高。

### 4. 测量纪律

NPU 空闲（npu-smi 无进程）串行确认；未 kill 任何非本会话进程；bwd 并行会话无干扰窗口；20 轮 ± std 如实记录；主文件零改动（md5 校验）。

### 5. 本轮产物

- 探针：/tmp/opencode/t3i4/{gen_probes.py, probe_run.py, probe_{lock0,lock,ra2,ra2loop,ra4,rb3,rb4,rc}_n{1,2,3,5,8}.py, diag_lock0.py, build_*.py, sparse_mla_fwd_wsobf16.py, bench_v.py, kernel_src_{main_ni8, probe_flat_t8, rb4_n3}.cpp, logs/（15+ 日志含 gdb 栈）}
- 备份：history_version/sparse_mla_fwd_perf_iter4.py（改前版本 = iter-3 采纳基线；本轮最终未改动主文件）
- 基线 JSON：perf_tuning/sparse_mla_fwd_baseline_iter4.json（含 RC1-RC5 全证据与 ws_o bf16 评估）

next_hint: 迭代 5 优先级：(1) 混合模式升级（用户已预批）——Expert 手动 flag 实现 2 槽 ring（rc 结构为蓝本：12 flag 语义已验证，手动同步绕开 RC1/RC2/RC3）；(2) 或先修 pass 侧（RC4 两 bug 可独立修复，RC3 多区间生命期为特性级）；(3) pto row_expand 形态（pto -4.78% 说明其后端形态敏感）；(4) gather 本体压缩（读侧 64×1152B MTE2 为散布读，上限有限）。

## Iteration 5 — 2026-10-15（混合模式升级：Expert 手动 flag ring fast path）

- bottleneck_type: sync（C/V 逐块 lockstep——iter-4 结论；本轮以 Expert 手动 flag ring 解除一层，V 侧串行成为新主项）
- optimization: 新增 `tl_sparse_mla_fwd_fast`（Expert 模式，ascendc 专项）+ host fast/generic 分派（indexer_bwd 形态）。DEPTH=2 ring 槽 × 4 workspace、LEAD=1 前瞻 gather、4 族 × 2 槽 = 8 个 cross-flag（F0 gather / F1A ws_s / F2 ws_p / F1B ws_o）、手动 UB 布局（194,560B 总和式，无 planner）、C 侧零手动 flag（gemm_v0 内建 4 类 head/tail flag 覆盖 L1/L0C 全部 hazard，reducesum 先例）。V body(n) = [gather(n+1); post(n)]，C body(n) = [QK; PV]，扁平块流 n=k*NI+t，全部 set/wait 位于有效块段内 1:1（§7.2 防悬挂）。
- baseline_time: 64437 us（ascendc iter-3 采纳基线）/ 75635 us（pto，本轮未动沿用）
- candidate_time: **58763±196 us（fast ascendc，20 轮；首样 57964±241）**；同会话 portable 复测 70206±265（+8.9% vs 记录基线——机器态漂移，见 §4）
- improvement: **vs 记录基线 -8.8%；同会话 A/B -16.3%**（两样本 -15.8%/-16.3%，std<0.5%）
- precision: **pass**（--level all "Test Passed!"：L0 8/8 + L1 48/0 = 56 PRECISION_PASS 0 FAIL；L2 4/4；Boundary 16 PASS + 6 WARN = bnd_extreme×4 + bnd_dbound×2，与 iter-3 基线逐位一致；fast 路径 oracle 6 用例 × 3 重复全部 bit-exact vs portable）
- adopted: **yes**（ascendc 新基线 58763；pto 75635 沿用）
- rollback_reason: 无
- consecutive_no_improvement: 0（重置）
- vs 24288 us 预算: OVER 141.9%（原 165.3%），剩余 2.42×（原 2.65×）

### 1. 结构与绕障映射（iter-4 RC1-RC5 → Expert 解法）

| AUTO 栈障碍 | Expert fast 绕法 |
|------------|-----------------|
| RC1 flag 预算 16（静态语句对全局递增 id） | 显式 4 族×2 槽=8 id 轮转复用，≤16 ✓ |
| RC2 静态语句配对 + 循环乘数匹配 | 手动 set/wait_cross_flag，无静态配对约束 |
| RC3 planner 单区间生命期（别名撤销→UB 超限） | 无 MEMORY_PLANNING：缓冲区总和式手动布局 194,560B ≤ 196,352B |
| RC4 planner 两 bug | 不经过 planner |
| RC5 Mod 区域索引硬件异常 | 槽位索引全部为 DEPTH=2 取模——**注意：与 t3i2/t3i4 的 Mod 区域索引坑不同层级**：坑在 TIR BufferRegion 的 mod 索引表达式，fast 版的 `slot*BI` / `n%DEPTH` 是 TIR 算术（标量），进入 copy 的 region 偏移为线性表达式，非 mod 索引 region ✓ |

slots 安全链（DEPTH=2/LEAD=1，全部经 flag 传递序推导，日志可查源码注释）：ws_kv 槽 m 由 bulk(m+2)@V body(m+1) 重写，程序序后于 post(m) 的 F1A(m) wait ⟹ C 已消费；ws_s/ws_p/ws_o 同理经 F 链 + V 串行程序序；kv_stage 单槽以显式 mte3→mte2 对（批间 + gather 头）覆盖跨 gather WAR（gather(n+1) 先于 post(n) 的 F1A wait，故 bulk(n) 排空无 flag 链可依，必须显式）。

### 2. 实现期平台发现（/tmp/opencode/t3i5/ 探针链）

| # | 发现 | 证据 |
|---|------|------|
| P1 | **单 cast 槽位 + [全部 cast; 全部 store] 批处理 = 所有 store 读到末组内容**（确定性 7/8 损坏，末组正确；fast 首版 Output 84% mismatch / Lse 逐位精确的根因） | probe_a_only（tagged 值：OutA 全部 = 组 7 数据）；修复=逐组 INTERLEAVE cast+store（mte3→v / v→mte3 相邻对） |
| P2 | 2D 行段切片 src + fp32→bf16 cast 拷贝**可用**（t3i2 损坏的是列段 src，行段安全） | probe_min form F bit-exact |
| P3 | `copy_ub_to_gm` 单模板类型（无 dtype cast）——UB(fp32)→GM(bf16) 直连不可表达，必须 UB→UB cast 槽 | common.h L272 模板签名 + 探针编译错误 |
| P4 | 整 3D GM→UB 拷贝只拷首个 2D chunk（t3i3 坑①再确认） | probe_cast_forms 生成代码 `copy_gm_to_ub<128,64>` 单 chunk |
| P5 | Expert 探针必须自带全部相邻 set/wait 对——零 flag 探针全形态 garbage（MTE2/V/MTE3 无任何自动序） | probe_cast_forms 首版 vs probe_min 对照 |
| P6 | AUTO_CV_SYNC 以 buffer 名含 "workspace" 子串配对——Expert fast 版沿用 ws_* 命名（无子串）避免双重同步 | ascend_combinecv.cc FetchWorkspaceName L196 |
| P7 | reducesum 的 bos if/else 分支教训 load-bearing：select 包裹 GM 标量进 tile.add 触发 AddsAndMuls codegen 崩 | fast 版按 if/else 写法一次通过 |

### 3. 精度复验（--level all，fast 分派生效）

56 PRECISION_PASS / 0 FAIL（L0 8：l0_small/main × 双后端，ascendc 走 fast——max_abs 1.562e-02/1.431e-06 与基线逐位一致；L1 48：in-domain 配置 l1_short_segments/l1_ni2/l1_tail1_prime/l1_ni1 的 ascendc 走 fast，其余（kv_group=2/heads=64/BI=64）回退 portable）。语义六注意点：clamp 寻址 ✓（rows_ub int32 向量预 clamp，reducesum 形态）、mask 置 0 ✓、任意分布 ✓、重复计贡献 ✓（l1_repeat）、-1/越界 ✓（l1_neg_oob）、首行 ✓。fast 域 oracle：NI∈{1,2,8}×{short/neg_oob/repeat/single}×3 重复全 bit-exact（含主 shape 3 重复）。

### 4. 计时与测量有效性（同口径 warmup5+20 NPU 事件，计时前 npu-smi 无进程确认）

| 项 | 值 |
|----|-----|
| fast ascendc（候选） | **58763±196**（min 58576 / max 59550；首样 57964±241） |
| portable ascendc（同会话复测） | 70206±265（首样 68822±230）——**vs 记录基线 64437 漂移 +8.9%** |
| A/B（同会话） | **-16.3%**（两样本 -15.8%/-16.3%） |
| vs 记录基线 | -8.8%（取候选 58763） |

漂移说明（如实）：portable kernel 代码零改动（builder 未触碰，仅新增 fast 路径与分派），两次同会话复测均 68.8-70.2ms 高于昨日记录 64.4ms，std 均 <0.5%（非双峰争用形态）——判为机器态（时钟/温度）漂移。采纳判定取双口径保守值：同会话 A/B -16.3% 与 vs 记录基线 -8.8% 均 >> 3% 阈 → 采纳。**新基线记 58763（fast）**，iter-6 起所有对比以同会话 A/B 为准（学：跨日绝对值不可比，本轮起记录同会话 portable 参照值）。

### 5. 分解更新（估算，iter-6 msprof 校准）

fast 稳态 ~35.8us/块（58763/1640）vs 估算 V 串行 ~23us/块（gather ~13 + post ~10）——差值 ~12.8us/块 ≈ 21ms，归因候选：(a) 同步对数量 ~36 对/块（gather 12 + post 24 含 p-store 逐组 16 对 + ws_o 8 对）的发射/往返；(b) kv_stage 4 批串行 mte3→mte2 停顿（每批 ~0.5-1us）；(c) p-store 逐组 cast→store 往返 8 次/块。iter-6 先 msprof 定位再动手。

### 6. 迭代 6 方向排序

1. **msprof fast 重画像**（PipeUtilization + wait 分布）：区分 (a) 同步往返 vs (b) MTE 排空停顿 vs (c) 真实带宽，排序下述各项。
2. **p-store ping-pong cast 槽**（cast_g×2 轮转）：消除逐组 mte3→v/v→mte3 往返（~8 对/块）；UB 需 +2KB——sumexp_i 并入 t_ub[0:64] 可腾（生命期不相交，见 iter-3 §3）。
3. **gather 批内 ping-pong**（kv_stage [2,8,576] 8 批轮转）：行读与 bulk 重叠，消除 3× 批间停顿。
4. **ws_o bf16 复评**（iter-4 留档 -1.17%）：ring 下 ws_o 读在 V 关键路径上（4×32KB/块），transit 减半收益重估。
5. 若上述合计仍 >45ms 级且 msprof 显示 V MTE 带宽为硬下限 → gather 读侧结构性改造（如 8 批 × 8 行的 MTE2 合并、或 indices 预排序假设——需用户批准语义边界）。

### 7. 本轮产物

- 主文件 sparse_mla_fwd.py：+`tl_sparse_mla_fwd_fast`（Expert ring）+`_fast_gate_ok`+分派（ruff check/format 过）；portable 路径零改动
- 备份：history_version/sparse_mla_fwd_perf_iter5.py（改前 = iter-3 采纳版，md5 与 iter4 备份一致）
- 探针与日志：/tmp/opencode/t3i5/{fast_proto,probe_cast_forms,probe_min,probe_a_only,bench_fast}.py + logs/（含 level_all.log 全量门禁、bench_fast×2）
- 基线 JSON：perf_tuning/sparse_mla_fwd_baseline_iter5.json

next_hint: 迭代 6 主攻同步往返削减（p-store ping-pong → gather ping-pong），先 msprof 校准 §5 分解假设；ws_o bf16 在 ring 形态下复评；跨日绝对值不可比，一律同会话 A/B。

## Iteration 6 — 2026-10-15（msprof 画像分解 + V 侧流水化三候选全灭：AIV flag 阻塞标记 / UB 微布局敏感性两条硬规则）

- bottleneck_type: sync（V 侧 HardEvent 往返 31.6% + C 侧 gemm_v0 pipe 级全同步 56.5% mte1_stall——但两者在当前语义下均已证明不可进一步流水化，见 §3-§4）
- optimization: 本轮以 msprof 画像分解为纲，按画像实施 4 个候选 A/B，**全部否决回滚**：D=epilogue div 向量化（非 bit-exact）、C=gather 半槽 ping-pong 流水（C2 合法版 +18.7%）、B=p-store ping-pong（B2 +9.3%）、B-lite=宽批 cast/store（+9.1%）。最终实现零改动（md5 与 iter-5 采纳版一致 d3b46163c6132a3f177144418d345cb1）。
- baseline_time: 58763 us（iter-5 记录）/ 75635 us（pto 记录）
- candidate_time: **52321±172 us（fast ascendc 本会话纯 kernel，ab_run 口径；接口级 52621±141）**——vs 记录基线 -11.0% 为机器态漂移（同向于 iter-5 的 portable 漂移 +8.9%，跨日绝对值不可比再次证实）；同会话参照：portable ascendc 纯 kernel 63985±240、**pto 纯 kernel 71683±262 / 接口 72069±281**（vs pto 记录 75635 漂移 -5.2%）；本轮 4 个候选 56301~62068 us 全部劣于同会话 V0
- improvement: 0%（无采纳；同会话 A/B 口径下各候选 -7.6% ~ -18.7%）
- precision: **pass**（回滚后 --level all 全量：L0 8/8 + L1 48/0 + L2 4/4 + Boundary 16 PASS + 6 WARN，与 iter-5 基线一致；fast 域 oracle bit-exact 含 NaN 感知比较——见 §6 检查器修正）
- adopted: **no**（全部回滚；实现保持 iter-5 采纳版）
- rollback_reason: 4 候选全部性能劣化或精度失败（§3 A/B 表）
- consecutive_no_improvement: 1
- vs 24288 us 预算: OVER 115.4%（52321 本会话口径），剩余 2.15×（跨日漂移修正后估计仍在 ~2.4×）

### 1. msprof 画像分解（本轮主交付；msprof op PipeUtilization，20 核均值，单 launch 56120us profiled wall ≈ 34.2us/块）

| 侧 | 管道 | 占比 | 折算/块 | 说明 |
|----|------|------|--------|------|
| C (cube0) | cube(mma) | 7.7% | 2.6us | 9 gemm 纯算力极小 |
| C | mte1 (L1→L0) | 5.8% | 2.0us | 576KB @ ~288GB/s |
| C | **mte2 (GM→L1)** | **34.7%** | 11.9us | 176KB/块（ws_kv 144KB 5 次 256B 窄行 strided + ws_p 32KB）@ ~15GB/s 有效 |
| C | fixpipe (L0C→GM) | 17.9% | 6.1us | 320KB/块（ws_s 64KB + ws_o 256KB fp32）@ 49.9GB/s |
| C | scalar | 18.9% | 6.5us | |
| C | **scalar_mte1_stall** | **56.5%** | 19.4us | gemm_v0 内部 pipe 级全同步的等待（§4） |
| C | cross-flag wait_id0 | 0.2% | ~0 | **C 从不等 V** |
| V (vector0/1) | vec | 17.6% | 6.0us | softmax 链（iter-3 已向量化） |
| V | **mte2 (GM→UB)** | **31.6%** | 10.8us | 232KB/块（KV 行读 72KB 散射 1152B + ws_s 32KB + ws_o 128KB fp32）@ 20.5GB/s |
| V | mte3 (UB→GM) | 17.6% | 6.0us | 88KB/块 @ 16.3GB/s |
| V | scalar | 24.2% | 8.3us | |
| V | **scalar_wait (HardEvent)** | **31.6%** | 10.8us | 相邻 [set;wait] 对的往返延迟（~36 对/块） |
| V | scalar_vector_stall | 15.5% | 5.3us | GetValue 等 VEC（gather 行读 S/V 往返） |
| V | cross-flag wait_id1 | 0.0% | ~0 | **V 也几乎不等 C** |

**35.8us/块分解结论：C 侧自缚 34.2us / V 侧自缚 34.2us / C-V 重叠率 ≈ 100%（两侧 cross-flag 等待均≈0，ring DEPTH=2+LEAD=1 已充分）**。iter-5 §5 的 12.8us/块差值不是同步空转，而是 V 侧 HardEvent 往返（10.8us）+ C 侧慢速 strided mte2 + 双侧 scalar 开销。**关键路径：V 侧略绑定**（C2/B2 实验中 V 侧开销增减 1:1 传导至 wall）。

### 2. 画像驱动的候选池重排（实施前分析）

- ~~C 侧 L1 双缓冲~~：读 gemm_v0 源码（common.h L1306-1370）发现内建 flag 为 **pipe 级全同步**——head [SetFlag/WaitFlag MTE2_MTE1] 等 MTE2 全排空、tail [MTE1_MTE2] 等 MTE1 全排空、[FIX_M] 等 mma 完成后才返回——L1 双缓冲无法解除 copy(n+1)↔PV(n) 串行。**死路**。
- ~~C 侧 QK 合并（K=576 单 gemm）~~：与 PV 共享 v_all 的 R1 列切片约束冲突（[128,576] 与 4×[128,128] 布局不可同 buffer）；双存则 KV mte2 翻倍。**死路**。
- ~~ws_o bf16（C fix -2.6us + V mte2 -3us）~~：V 侧需 16KB bf16 staging，UB 余量 1.75KB，且本轮证实 UB 微布局敏感（§3 B2b）。**本轮死路，iter-7 候选**。
- 实施队列：D（div 向量化）→ C（gather 流水）→ B（p-store ping-pong）→ B-lite（宽批）。

### 3. A/B 结果表（全部同会话，V0=52321±172 为分母；精度 quick check = fast vs portable ascendc bit-exact×4 shape + 主 shape）

| 候选 | 结果 | us | vs V0 | 判定 |
|------|------|-----|-------|------|
| D epilogue div 向量化（broadcast+2D div 替 256 行 div） | bit-exact 失败：Divs(标量) vs Div(向量) fp32 末位差 → 95/39M 元素 1 bf16 ULP（Lse 逐位一致） | — | — | **否决**（纯重构必须 bit-exact） |
| P1 gather 延迟 release 标记（无预置版前体） | vector core timeout（死锁） | — | — | 挂死 |
| P1c 旧代码 + 仅 2 个悬空预置 set | vector core timeout | — | — | **定位根因**（§4 R-AIV-FLAG） |
| C2 gather 半槽流水（合法 ledger：头排水对 + hb<NHB-2 才 set release，无悬空无预置） | bit-exact ✓ 但 62068±188 | 62068 | **+18.7%** | **否决**（8 行小 bulk + 倍增 marker 开销吞掉重叠收益） |
| B2 p-store ping-pong（合法 ledger 版） | bit-exact ✓ 但 57197±247 | 57197 | +9.3% | **否决** |
| B-lite 宽批 cast/store（16 行批，纯相邻对，无新 marker） | bit-exact ✓ 但 57067±173 | 57067 | +9.1% | **否决** |
| B2b 仅 sumexp_i→t_ub[64:128] 合并（其余全 V0） | bit-exact ✓ 但 56301±216 | 56301 | **+7.6%** | **定位毒源**（§4 R-UB-LAYOUT） |

### 4. 硬件/编译器规则发现（本轮最有价值产出，后续轮次与同仓算子均适用）

- **R-AIV-FLAG（AIV set_flag = 源管道阻塞标记）**：AIV 侧 `set_flag` 在源管道（MTE3/MTE2）指令流插入标记，**标记被 wait_flag 消费前该管道停转**。推论：(a) 悬空/预置 set = 永久停转（P1c：旧代码+2 个预置 set 即挂死 vector core）；(b) 延迟消费 = 停转窗口 = 消费间隔（C2 +18.7%）；(c) **只有相邻 [set;wait] 全排空对是安全高效的**——细粒度 slot 流水（cast/store 重叠、行读/bulk 重叠）在 AIV 上结构性不可表达。mha_sink_fwd_bhsd 的 C 侧延迟标记惯用法（SIG+side 跨迭代）不可移植到 AIV。
- **R-UB-LAYOUT（V 侧 UB 微布局极度敏感）**：仅把 sumexp_i（256B，布局尾 buffer，零偏移移动、零 flag 变化、同操作数）合并进 t_ub[64:128] 即 **+7.6%**。机制未定位（银行冲突/tail-mask 代码路径均可能）；工程结论：**fast 路径 UB 布局冻结，任何腾挪需单独 A/B**。
- **R-DIV（标量/向量除法末位不一致）**：`T.tile.div(dst, a, scalar)` 与 broadcast+向量 div 的 fp32 结果偶发末位差（Divs vs Div 指令），bf16 舍入放大为 1 ULP——**逐行标量 div 不能向量化**（bit-exact 门禁下）。
- R-WIDE-CAST：[16,128] 宽批 cast/store 比逐 [8,128] 慢 ~1.5%（B-lite vs B2b 差值）——往返次数减半的收益 < 宽形代码路径损失。

### 5. 测量有效性

- 本会话机器态：V0 复测 52321 vs 昨日记录 58763（-11.0%）——与 iter-5 portable 漂移（+8.9%）同为跨日态漂移，std<0.5% 非争用形态。**iter-7 起所有 A/B 必须同会话**（本轮已执行）。
- 各候选 std 173-247us（<0.5%），A/B 差值（4-10ms）远超噪声阈。

### 6. 检查器修正（ab_run.py，/tmp/opencode/t3i6/）

生成器含 10% -1 + 10% 越界索引 → ~7% 行全遮蔽 → Output=0/0=NaN、Lse=-inf（双路径同值）。`torch.equal` 不能判 NaN==NaN——quick check 已改 NaN 感知逐位比较（`(a==b)|both_nan`）；主 shape 及全部 stress 用例 fast vs portable 非 NaN 元素全部逐位一致。

### 7. 产物与备份

- 主文件零改动（md5 = iter-5 采纳版）；备份 history_version/sparse_mla_fwd_perf_iter6.py（改前状态）
- A/B 基建：/tmp/opencode/t3i6/{ab_run.py, prof_fast.py, dump_src.py, diag_v0.py} + logs/{ab_V0,D,C,P1,P1c,C2,B2,Blite,B2b}.log + msprof_fast/ msprof_op_fast/（画像 CSV）+ fast_kernel_src{,_C}.cc
- 基线 JSON：perf_tuning/sparse_mla_fwd_baseline_iter6.json

next_hint: 迭代 7 剩余合法杠杆排序：(1) **ws_o bf16 的 UB 重构**——GKB=8 缩 kv_stage 至 9KB + m1_ub/idx_f 合并 + sumexp 合并共腾 ~11.8KB 仍差 ~4.6KB，唯一途径是 kv_stage/o-stage 显式共享布局（TIR 单 buffer 重构）或 acc_s_ub 生命期切分，且必须先过 R-UB-LAYOUT 单项 A/B；(2) C 侧 strided mte2 11.9us/块（176KB@15GB/s）——ws_kv 段-major 布局会毁 V bulk，暂无合法解；(3) 若 (1)(2) 均不可行，剩余 2.15-2.4× gap 需要设计层变更（CV 分工重构 / workspace 往返模式消除）——届时标 [DESIGN_ERROR] 回 Stage 1（本轮已试 4 案 + 结构分析排除 3 案，满足≥3 案门槛，但先给 iter-7 一次 UB 重构机会）。

---

## Iteration 7 — 2026-10-15（DESIGN_ERROR 判定：架构性能地板证明 + 设计层证据包）

- bottleneck_type: other（架构性能地板——三硬规则 + 平台约束锁死当前形态的相位下限，实现层无剩余路径）
- optimization: 无代码改动（纯分析判定轮；ws_o bf16 UB 重构实证正式跳过，见 §1 处置说明）
- baseline_time: 52321±172 us（fast ascendc 本会话 ab_run 口径 = iter-6 V0）
- candidate_time: 无候选（判定轮，无新实验，全部引用在案数据）
- improvement: 不适用（0%）
- precision: pass（实现零改动，iter-5 采纳版状态延续，md5 d3b46163c6132a3f177144418d345cb1；零改动无复验需求）
- adopted: no（零改动；本轮交付 = [DESIGN_ERROR] 判定 + rev2 证据包）
- rollback_reason: 不适用（零改动，无回滚需求）
- consecutive_no_improvement: 2
- vs 24288 us 预算: OVER 115.4%（52321 口径），剩余 2.15×；架构地板 52-56ms（§2 推导）

### 1. 判定结论与依据

**结论 [DESIGN_ERROR]：当前架构（bf16 GM workspace 中转 + L0C 128KB + AIV flag 语义 + UB 冻结布局）的性能地板
52-56ms，为目标 24288us 的 2.15-2.31×；实现层合法杠杆已枚举穷尽（iter-6 实证否决 4 案 + 分析排除 3 案，满足 ≥3 案
设计错误门槛），最后剩余杠杆 ws_o bf16 的理论上限 ~44ms 仍差目标 ~1.8×——无实现层路径可达标，需回 Stage 1 rev2
重构数据通路（突破方向证据包见 §3）。**

依据链（全部在案）：
1. 同会话基线 52321±172 us vs 目标 24288 us（2.15×）——iter-6 §3 V0（ab_run 同会话口径）。
2. 迭代 6 画像：ring 重叠已 100%（C 侧 cross-flag wait 0.2% / V 侧 0.0%）——调度层零剩余收益；C/V 各自缚
   34.2us/块平齐，wall = max(C,V) 已取满，双侧同时贴各自平台墙（§2 逐相位归属表）。
3. 三硬规则（iter-6 §4）：R-AIV-FLAG（AIV set_flag = 源管道阻塞标记 → slot 级流水结构性不可表达，锁 V 侧
   HardEvent 10.8us/块 + ~36 相邻对纪律）；R-UB-LAYOUT（256B 腾挪即 -7.6% → UB 布局冻结）；R-DIV（bit-exact
   门禁下 epilogue 标量 div 不可向量化）。
4. 方案门槛：iter-6 实证否决 D/C2/B2/Blite（+7.6%~+18.7% 或非 bit-exact）+ 分析排除 C 侧 L1 双缓冲（gemm_v0
   pipe 级全同步）/ QK 合并（R1 列切片冲突）/ ws_o bf16-当轮（UB 16KB 放不下）共 7 案；迭代 1-6 累计另有 6 案
   否决留档（iter-1 三快赢、iter-4 ws_o bf16 旧架构 -1.17% 等）。

**ws_o bf16 UB 重构（最后剩余杠杆）——正式跳过实证，双重理由：**
- 执行不可得：该实证连续两次调度截断，无法产出可信数据（如实记录，非结果推测）。
- 决策无价值：即使收益全额兑现（iter-6 §2 相位口径：C fix -2.6us + V mte2 -3us ≈ -5.6us/块，-16%），上限
  ~44ms 仍差目标 ~1.8×（44/24.288 = 1.81）——最优情形仍失败，实证不改变判定方向。
- 叠加在案风险：iter-4 旧架构实测 -1.17%（<3% 噪声阈不采纳）+ 精度余量代价（out max_abs 1.562e-2 → 3.125e-2）；
  R-UB-LAYOUT 要求任何腾挪单独 A/B（B2b 256B 腾挪 -7.6% 先例）；UB 余量仅 1.75KB（194,560/196,352），需
  kv_stage/o-stage TIR 单 buffer 显式共享布局或 acc_s_ub 生命期切分（iter-6 next_hint (1)）。
- 处置：留档为 rev2 方向 (a) 的从属项——L0C 驻留 acc_o 从根上消除 ws_o 往返（16.8→2.1GB），严格优于 bf16 化减半
  （16.8→8.4GB），且不触碰 R-UB-LAYOUT 冻结区。

### 2. 当前架构性能地板推导

**地板公式：34.2us/块 × ~1638 块/核 ≈ 56ms 理论墙。** 块数 = 4096 task × NI=8 = 32768 块 / 20 核 = 1638.4 块/核
均值（关键核 205 task × 8 = 1640）→ 55.9-56.1ms。闭合验证：msprof 单 launch profiled wall 56120us（= 34.2 × 1640
✓）与 ab_run 实测 52321us（无 profiler 开销口径，-6.7%）两级测量夹逼确认地板带 **52-56ms**。双侧 flag 等待 ≈0 →
地板 = max(C,V) 自缚时间，架构内调度无杠杆；唯一出路是降低相位本体——被下表逐项锁死：

| 相位（iter-6 §1 口径） | us/块 | 锁死该相位的硬规则 / 平台约束 |
|---|---|---|
| V mte2 10.8（KV 行读 72KB 散射 + ws_s 32KB + **ws_o 128KB fp32**） | 10.8 | ① bf16 UB→L1 禁直连（common.h copy_ub_to_l1 static_assert half-only，DESIGN L534/L302）→ gather 必经 GM ws_kv；② L0C 128KB（DESIGN L300）→ acc_o [128,512] 256KB 无法驻留 → ws_o 逐块读回 |
| V mte3 6.0（ws_kv bulk 72KB + ws_p 16KB） | 6.0 | bf16 禁直连（ws_kv 行-major 布局为 V bulk 写与 C strided 读的折衷，段-major 毁 V bulk——iter-6 next_hint (2)） |
| V HardEvent 10.8（~36 相邻 flag 对/块） | 10.8 | **R-AIV-FLAG**：仅相邻 [set;wait] 全排空对安全 → 逐组 cast/store、逐批 gather、逐 chunk ws_o 读的往返不可折叠不可流水（C2 +18.7% 实证） |
| V vec 6.0 + scalar 8.3 | 14.3 | iter-3 已向量化（剩余为语义必要算子）；R-DIV 锁 epilogue 256 行标量 div（kernel L953-955） |
| C mte2 11.9（ws_kv 144KB 5 次窄行 strided + ws_p 32KB） | 11.9 | bf16 禁直连 → C 侧 GM 读回 ws_kv，@ ~15GB/s 有效（窄行 strided） |
| C fixpipe 6.1（ws_s 64KB + **ws_o 256KB fp32**） | 6.1 | L0C 128KB → acc_o 放逐 V-UB → 每块 PV 后必须 flush（kernel L718-723） |
| C gemm 4.6 + mte1_stall 19.4（重叠相位） | 4.6 | gemm_v0 内建 flag 为 pipe 级全同步（common.h L1306-1370，iter-6 §2）→ L1 双缓冲不可表达（已排除） |
| C scalar 6.5 | 6.5 | 寻址/循环开销（块数放大项，见方向 b） |
| flag 预算 16（GetffstMsg &0xf，iter-4 RC1） | — | 现用 4 族 × DEPTH=2 = 8 ids；非当前墙（重叠已满），但与 R-AIV-FLAG 叠加封锁 slot 级流水与更深 ring 的表达空间 |

**地板构成小结**：C 侧 34.2us 中 ~18.0us（mte2 11.9 + fix 6.1）是 bf16 禁直连 + L0C 容量两项平台约束的直接代价；
V 侧 34.2us 中 ~11-12us（ws_o 读回 ~6.0 + ws_kv bulk 写等）同源、10.8us 为 R-AIV-FLAG 代价、其余为语义必要算子与
散射读。两侧同时贴墙且已 100% 重叠 → **架构地板 ~34.2us/块 ≈ 56ms 成立**，且每一项均有明确归属（上表），无"再调
调度/再挤流水"的剩余空间——这正是 [DESIGN_ERROR] 的完整证明结构。

### 3. 设计层突破方向证据包（供 Stage 1 rev2）

> 流量口径（当前 fast 形态逐项重算，32768 块 / 4096 task）：KV gather 读 4.83 + ws_kv rw 9.66 + ws_s rw 4.29 +
> ws_p rw 2.15 + ws_o rw 16.8 + Q 0.60 + Output 0.54 ≈ **38.7GB**（与 iter-1 §3-④ "≈38GB" / DESIGN L464 闭合）。

**a. M=32 head 分片 + L0C 跨块 acc_o 累加（ws_o 16.8 → 2.1GB）**
- 机制：PV 累加器从 V-UB（acc_o_all 128KB，kernel L666）改为 L0C 驻留跨块 init=False 链（L718-719 已有块内形态），
  任务尾单次 flush [128,512] fp32（w+r = 2×1.07GB = 2.1GB）。驻留编排两形态：(i) head 分片流——M=32 → acc_o
  [32,512] 64KB + acc_s [32,BI] 16KB = 80KB < 128KB 单分片全任务驻留；(ii) d-outer 块序重排——不分片，acc_o
  [128,128] chunk（现 acc_o_l0c 同形 L664）按 DV 段外层跨块累加，P 需全块驻留（ws_p 扩 per-task 全 NI 槽，或
  +3.2GB ws_p 重读）。
- ⚠ 设计题 1（驻留编排）：4 分片 × 64KB = 256KB > L0C 128KB——分片 pass 化（pass 边界 GM park/unpark ≈ 2.1GB，
  与任务尾 flush 等价）+ pass 间 KV 复读（+14.5GB ws_kv；(c) 下 KV 池仅 4.7MB L2 常驻可缓解）vs (ii) 的 P 驻留
  代价——结构性张力，rev2 核心设计题。
- ⚠ 设计题 2（alpha rescale）：L0C 内 acc_o 无法被 V 侧 rescale → (1) 对角 gemm：acc_o ← diag(alpha)·acc_o +
  P·V（+4.2 MFLOP/块 ≈ +12% cube，2.6→~2.9us/块，7.7% 利用率余量足；alpha 经小 buffer V→C 传递，复用 F2 族）；
  或 (2) 数学重排：块内 m 固定不 rescale、任务尾统一校正（FlashMLA split-merge 式，需防 exp 溢出）。
- 平台约束核对：L0C 80KB ✓；L1 增量小（q 分片 37KB + v_all 147KB 不变 < 现役 328KB 实证包络 ✓）；gemm_v0
  init=False 跨块累加已验证（L711）；flag 预算 ✓（与 (c) 组合则 F0 族 2 ids 释放）。
- 精度风险：(1) 路径 acc_o 全程 fp32 L0C、rescale 亦 fp32——不劣于现状（现路径多 4 次 GM 往返舍入）；(2) 路径
  改变累加次序 → 需重基准（bit-exact 门禁豁免评估）。
- 依据：kernel L663-664（L0C 贴线 64+64KB）/L666/L654（ws_o fp32）/L718-723（逐块 flush）/L936-950（V 读回累加）/
  L952-976（epilogue）；iter-6 §1；DESIGN L300。

**b. BI=256 大块（固定项摊销；未上 (a) 时 ws_o 16.8 → 8.4GB）**
- 机制：NI 8→4、总块数 32768→16384；块数-scaling 项（V HardEvent 10.8 + V scalar 8.3 + C scalar 6.5 = 25.6us/块
  双侧合计，关键路径份额 ~17us/块）总量减半；ws_o 每块 [128,512] 与 BI 无关 → 未上 (a) 时总量减半。
- 预期：单独使用墙 ~46-48ms（固定项 -14ms 被 traffic 相位随块尺寸放大 1.45-2× 抵消）；组合价值 = 固定项摊销——
  (a)+(c) 后固定项成为新主项时为必要组成（见 e 表注 4）。
- 平台约束核对（贴线精算）：L1 = v_all [2,256,256]bf16 262KB + tail [256,64] 33KB = **295KB**；q_all [2,128,256]
  131KB + qt [128,64] 16KB = **147KB**；+ p_l1 [128,256] 66KB → 合计 ~508KB。L0C：acc_s [128,256] = 128KB 单独
  贴线 → **必须与 (a) 联动**（M=32：32+64=96KB ✓）。⚠ L1 上限本仓未核验（现役实证下限 328KB 可用；508KB 需探针，
  若 910B3 L1 < 508KB 则缩 BI=192 或不入）。UB 全新布局需整体重排（R-UB-LAYOUT：单独 A/B）。
- 精度风险：无语义变化（块内行数不影响逐行 online softmax 数学）；gemm 分形 [128,256]/[256,256] ✓。
- 依据：iter-6 §1（HardEvent/scalar 相位）；iter-1 §4（BI=64 恶化 +81%——块尺寸单调性反向验证）；kernel L622；
  DESIGN L300（BI=256 L0C 贴线预警）。

**c. fp16 直连档（消 ws_kv 9.66GB + F0 族；DESIGN §9.5 档 3 原案）**
- 机制：V 侧 gather 后 UB 内 bf16→fp16 cast → `copy_ub_to_l1` 片上直连（common.h:358-361 static_assert
  half-only——fp16 合法、bf16 被禁，即 §2 表"bf16 禁直连"的正面解）；Nd2Nz 直落 L1 分形布局（免 mte1 ND→NZ
  变换）。消除：ws_kv rw 9.66GB + F0 cross-flag 族（2 ids）+ C 侧 strided mte2 11.9us/块 主项 + V 侧 bulk mte3 写段。
- 平台约束核对：copy_ub_to_l1 half-only ✓（本仓源码级确认）；gemm_v0 fp16×fp16→fp32 ✓；Q 需同 fp16（gemm 无
  混合 dtype）→ 输入契约变化（host 预 cast 604MB 或接口层 fp16 档）；PTO 走 generic（原案限定 ascendc fast path）。
- 精度风险：fp16 尾数 11bit > bf16 8bit（逐元素更精）但值域窄（max 65504 / min 正规 ~6.1e-5）→ **值域谓词 +
  generic 回退成对为硬门禁**（原案 L686 + §11.4）：host 侧 KV（4.7MB 全量扫，代价可忽略）与 Q（604MB）|x| 上界
  预检；下溢（|x|<6e-5 → subnormal/0）是否入谓词需 rev2 界定；P∈[0,1] 无溢出；输出仍 bf16。
- 预期流量：38.7 → 29.0GB（-25%）；C 侧 mte2 11.9 → ~2-3us（仅剩 ws_p），mte1_stall 亦减（Nd2Nz 免变换）。
- 依据：DESIGN_sparse_mla_fwd.md L686（档 3 原案全文）/L534（static_assert 行号 + gqa_sink 消 ws 先例"All 6 GM
  workspace buffers eliminated"）/L302/L330；kernel L743-755/L813-832（bulk 写 + F0）/L701-707（C 侧读 + F0 wait）。

**d. deepseek_v4 大 tile 模式参考（结构对标）**
- 机制：S2_BASE=512 KV 大 tile + M_CHUNK=16 行块 softmax（行粒度 online softmax 与大 tile 解耦，与本算子
  v_block/EG 行组同构）+ L1 KV ring 3 槽（对照本算子 GM DEPTH=2 ring——ring 从 GM 层上移 L1 层）+ L0A/L0B 双缓冲
  （对照 gemm_v0 pipe 级全同步——operand 级流水替代全排空）。
- 实证口径：同家族（MLA decode 前向）91 TFLOPS，对照本算子有效算力需求 ~48 TFLOPS（DESIGN L7）——结构余量证据。
- ⚠ 来源核对：本仓已检索（examples_experiment/{deepseek_mla,deepseek_nsa} + 全局 rg）**无 deepseek_v4/S2_BASE
  命中**——为上游/外部参考，rev2 需获取源码核对参数与可移植性（如实记录）。
- 平台约束核对：L1 3 槽 ring = 3×147KB = 441KB 贴线（同 (b) L1 未核验风险）；L0A/L0B 双缓冲需 Expert 手动
  operand staging 绕过 gemm_v0 内建同步（fast path 已是 Expert，可行性中等，需探针）；topk 逐行 gather 为本算子
  独有（deepseek_v4 为连续/paged KV）→ 需保留 V 侧 gather + (c) 直连组合。
- 精度风险：纯结构参考，无直接精度迁移。依据：任务给定参数（上列）+ 本仓检索无命中（⚠）。

**e. 流量-带宽可达性汇总（组合可行性判定）**

| 组合 | 总 GM 流量 | 24288us 所需有效带宽 | vs L2 ~1TB/s 口径¹ | vs 实测聚合 0.74TB/s² |
|---|---|---|---|---|
| 当前架构（fast iter-5 采纳版） | 38.7GB | 1.59TB/s | 超 59% | 超 115% |
| a（L0C 驻留 acc_o） | 24.0GB | 0.99TB/s | 贴线（±1%） | 超 34% |
| b（BI=256） | 30.3GB | 1.25TB/s | 超 25% | 超 69% |
| c（fp16 直连） | 29.0GB | 1.20TB/s | 超 20% | 超 62% |
| a+b | 24.0GB³ | 0.99TB/s | 贴线 | 超 34% |
| **a+c** | **14.3GB** | **0.59TB/s** | **余 41%** | **余 20% ✓** |
| **a+b+c** | **14.3GB³** | **0.59TB/s** | **余 41%** | **余 20% ✓** |

¹ L2 ~1TB/s 为 DESIGN L689 估算口径（workspace 工作集 ~20MB 常驻 L2，吸收率未实测——rev2 需双后端实测确认）。
² 实测聚合有效带宽 = 38.7GB / 52.321ms = 0.74TB/s（当前架构已达，作可达性下界锚点）。
³ b 的 ws_o 减半被 a 的消除吸收；其余 per-block workspace（ws_kv/ws_s/ws_p）在 b 下总量不变。
⁴ 流量可达是必要非充分条件：固定项（R-AIV-FLAG 相邻对纪律，~17us/块关键路径口径）在 a+c 下仍 ~20-28ms ≥ 目标线
  → **(b) 为必要组成**；a+b+c 后固定项 ~14ms + 流量 0.59TB/s（低于实测锚点 20%）——两侧首次同时低于目标线，
  端到端余量需 rev2 按相位精算。

### 4. 本轮产物与状态

- perf_log 本章节（DESIGN_ERROR 判定 + 地板推导 + a-e 证据包 + 可达性表）；debug_log.md 追加 [perf-iter7] 一行
- 主文件 sparse_mla_fwd.py：**零改动**（md5 = iter-5 采纳版 d3b46163c6132a3f177144418d345cb1，与 iter-6 备份一致）
- 无新增基线 JSON / 无实现备份（零改动轮，iter-6 备份链延续有效）
- 精度状态：pass 延续（零改动无复验需求；iter-6 回滚后 --level all 全量绿在案）

next_hint: Orchestrator 裁决回 Stage 1 rev2：以 **(a)+(c) 为流量主轴 + (b) 固定项摊销 + (d) 结构参考** 重设计
数据通路（L0C 驻留 acc_o + fp16 UB→L1 直连 + 大块/L1 ring）；本实现（fast 52.3ms 同会话 / 便携 64-70ms）冻结为
实现层最优参考基线，rev2 后作对照。
