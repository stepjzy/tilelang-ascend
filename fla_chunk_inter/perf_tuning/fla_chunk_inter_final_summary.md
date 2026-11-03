# fla_chunk_inter Stage 3 R3 最终交付报告（fla_chunk_inter_final_summary.md）

> 设备：**Ascend 910B3**（20 AIC + 40 AIV，L2 192MB，HBM 实测带宽 1263 GB/s r+w，同族标定）
> tilelang 仓库内 editable 构建 ｜ R3 战役日期：2026-11-03 ｜ R1/R2 历史见 git 历史与 perf_log 前段
> 计时口径：**do_bench（warmup=20 / rep=10 / mean / 仅设备侧 NPU 事件，不含张量构造/H2D-D2H/JIT/框架开销）**；
> msprof op 单_launch 设备侧时间交叉验证。
> R3 用户指令：按原指标 **<3400us** 继续调优；Expert 路线授权使用；**若仍未达标，必须逐项列出
> "是什么限制"（限制描述 + 证据（文件:行）+ 突破条件 + 地板值）**——本报告 §3 按此硬性要求交付。

## 0. R3 结论速览

| 项 | R2 终态（= R3 起点） | **R3 终态** | 变化 |
|---|---|---|---|
| do_bench 代表值 | ~6767 us | **~6767 us（R2 档案）/ 6639-6720 us（R3 会话实测带）** | **无净变更** |
| msprof 单_launch 设备侧 | 6534.3 us | 6534.3 us（kernel 未变，R2 画像承继） | 0 |
| vs 目标 < 3400 us | 超 1.99× | **超 1.99× → 仍未达标** | — |
| 精度 | --level all 全绿 | **--level all Test Passed（exit 0）本会话复验** | — |
| 采纳链 | — | **R3 零采纳**（两实现候选失败回滚；五个方向全部证据闭合） | — |

**R3 kernel 与 R2 终态逐位一致（md5 db1d9e8a）**。R3 的产出是**决定性证据**而非实现变更：
threads=2 结构性闭合（微实验 + 源码双证）、Expert barrier-thinning 假设证伪（-3.1%）、
mul_add_dst 指令禁用证据（CANN 头文件）、in-place 链崩溃证据、launch 形态源码闭合、
bDK=128 预算复核——合成**完整限制清单（§3，用户硬性要求格式）**。

## 1. R3 战役记录（详见 perf_log R3 段）

| 轮次 | 内容 | 结果 |
|---|---|---|
| r3_iter0 | 会话基线锚点（3 次官方口径） | 6719.8 us（与 R2 档案 -0.7%，噪声内） |
| r3_iter1 | **threads=2 纯手动最小实验**（fla_chunk_inter_vid2_micro.py，5 子链 ×2 线程形态，fp64 golden 门禁） | **方向闭合**：drain 数学 4/4 子链 t2 损坏（含"形状一致"的 [BS] axis=1 broadcast），produce 侧 1/1 可行但 threads 是 kernel 级参数 |
| r3_iter2 | drain 门控链 mul+add → mul_add_dst 融合 ×2 处（-2 op/bk） | **精度失败回滚**（dg 0.9852/0.9857 ~1% 确定性偏差；CANN 9.2.0 `kernel_operator_vec_binary_intf.h:249-250` MulAddDst 带 `__ASC_USE_RESERVED_UBUF__(3510)` 受限标注；分离二分排除规划器扰动） |
| r3_iter3 | drain in-place 链（broadcast→sub→exp 原位，-32KB UB 足迹） | **设备崩溃回滚**（ACL 507014 → 设备挂起；sync-insert 对 in-place 序列缺陷，R1 iter6a 家族签名；恢复后 6639.1us 复验正常） |
| R3 附 | **vchain 微实验**（fla_chunk_inter_vchain_micro.py：drain 形态链 ×400 任务，AUTO_SYNC vs 手动突发协议） | **假设证伪**：A 665.8-731.6us / B 643.3-645.6us（双跑核验，yield -3~-12%，B 稳定贴 mte2 引擎界 1.6us/任务）——逐 op 屏障非串行化主因，真因 UB 容量 + 交叉别名；手动协议精度双 PASS（flag 计数平衡验证成立） |

设备事件记录：iter3 崩溃后设备挂起，清理两个占卡进程（含一个 **R2 世代残留探针**
`/tmp/opencode/probe/r2_hang_locate.py`）后恢复；恢复后全量门 + 3 次 do_bench 复验正常。

## 2. 精度复验（R3 终版门禁，回滚态 = R2 终态逐位复现）

`python test_fla_chunk_inter.py --level all` → **Test Passed!（exit 0）**
- L0 2/2 + L1 8/8 全 `[PRECISION_PASS]`；L2 8/8 正确拒绝；Boundary 3 PASS + 2 个 Stage 2 已文档化 WARN
- kernel 与 R2 终态 md5 一致 → R2 全绿结论承继 + 本会话全量复跑实证

## 3. 完整限制清单（用户硬性要求：逐项 限制 + 证据 + 突破条件 + 地板值）

### 3.1 逐项清单

**【限制 1】threads=2 的 vid 语义对本算子 drain 数学结构性破坏 —— 40 AIV 唯一入口锁死**
- 限制描述：threads=2 时 `AscendVidReduction` 无条件将**每个** shared.ub buffer 第 0 维折半、GM 访问在
  （gm 维数 − ub 维数）轴加 vid 偏移。本算子 drain 的三类算子形态不可表达：① 1-D [bDK] 逐列不变量
  （gn/dgkn/pp）沿 DK 切半 vs 2-D [BS,bDK] 沿 BS 切半——唯一桥接 `T.tile.broadcast` 不在折半集内，
  显式 shape 参数不折半 → dst 越界写/尺寸失配；② dim=0 reduce（pp = Σ_t dk·k）产 per-vid 部分和，
  **无跨 vid 合并机制**；③ [1,bDK] 载入的 vid 偏移 = vid×dim0/2 落在 S 轴 → 读相邻 chunk。
- 证据（实测，本会话 `perf_tuning/fla_chunk_inter_vid2_micro.py`）：
  K1（纯 2-D + [BS]→[BS,bDK] axis=1 broadcast，"形状一致"）：t1 PASS / **t2 FAIL**（dg 末行 ratio 0.9844）；
  K2（[bDK] 1-D + axis=0 broadcast）：t1 PASS / **t2 FAIL**（dk ratio 0.2501，max_abs 7.5e4）；
  K3（dim=0 reduce）：t1 PASS / **t2 FAIL**（末行部分和）；
  K4（= drain 精确结构）：t1 PASS / **t2 FAIL**（ratio 0.26，与 R1 iter3 实机崩溃同签名）；
  K5（produce 侧 [bDK,DV] DK 切分 + dim=1 reduce）：t1/t2 **双 PASS**（produce 形态可行，但 threads 为
  kernel 级参数，drain 不可行即整体不可行）。
- 证据（源码）：`src/transform/ascend_vid_reduction.cc` L184-204（dim-0 无条件折半）、L475-487
  （GM 偏移轴 = indices.size()−ub_dims，trailing 对齐规则）、L563-586（tile_ops 折半集**不含**
  tl.ascend_broadcast）、L628-729（ascend_reduce 仅 M 折半，无跨 vid 合并；Analyze 阶段 L68-89 仅定
  workspace 除数）；`tilelang/engine/phase.py` L51-55（vid pass 在 CombineCV 之前，作用于全函数）。
- 突破需要什么（**上游修复项 ×3，需组合**）：① `tl.ascend_broadcast` vid 感知（显式 shape 参数随
  buffer 折半，或编译期越界检查）；② dim=0 reduce 的跨 vid 合并协议（部分和经 GM 合并或硬件规约树）；
  ③ per-buffer 免切分/切分轴标注（1-D 列向量按声明轴切或不切）。任一单项修复不充分。
- 解锁后地板：V-mte2 2867 → **~1434 us**（40 AIV，装载带宽翻倍）；V 链地板 **~3.5 ms 级**——
  但随即受限制 3 约束（C 墙 ~4.5ms），**需与限制 3 联合解锁**方可达标。

**【限制 2】UB 196KB 容量强制 produce 逐操作数串行 —— Expert 同步重写无法解锁**
- 限制描述：produce 装载集 224KB/任务（do/v/dv 96KB + h/dh 128KB fp32 域）> 单 AIV UB，装载→拆分→
  中继只能逐操作数轮次链式（5 轮串行），mte2/vec/mte3 引擎无法在 produce 内重叠。
- 证据：`src/transform/ascend_memory_planning.cc`（UB 硬上限 196352B）；R2 iter2 画像
  `aiv_scalar_vector_stall 6490us`；**本会话 vchain 微实验**（`fla_chunk_inter_vchain_micro.py`）：
  同一 drain 形态链 AUTO_SYNC 下 1.66 us/任务 ≈ mte2 引擎界（96KB/55GB/s = 1.8us）——逐 op 屏障不是
  串行化主因（手动突发协议仅 -3.1%，双 PASS）；真因 = UB 容量链式 + produce/drain 交叉别名屏障。
- 突破需要什么：**架构物理上限**（单 AIV UB 196KB 为硬件常数）。任务内更细切分已被 reduce_2d
  [64,64] 损坏（R1 iter5 实证）与中继协议复杂度封死；唯一路径 = threads=2（限制 1，需上游修复）。
- 地板：Expert-V 最佳（drain 内部重叠 ~3.5us/任务 + razor-thin 跨相位重叠，produce 活跃集 ~130KB +
  drain ~64KB ≈ 194KB 贴 196KB 上限）= **~9-12 us/任务 ≈ 3.7-5.0 ms**。

**【限制 3】C 消费链 10.9us/任务 —— V↔C 无片上直连，GM 中继架构强制**
- 限制描述：C 侧每任务 busy 4481us（cube 703 + mte1 staging 925 + mte2 中继读 1208 + fixpipe 662 +
  scalar 地址计算 983）= 10.9us/任务；V 侧任何提速后 C 即成墙。
- 证据：R2 final 画像（`msprof_output_r2_final`，kernel 未变，承继有效）；中继架构为工具链事实
  （DESIGN §8.6/§4.4：V↔C 交换全部经 GM workspace，无片上直连路径——fla_wy_fast final_summary §4
  实证）；18 gemm/任务为精度强制（fp32 gemm_v0 损坏 → fp16 hi/lo 3-gemm 拆分，家族域内无替代）；
  中继字节已最小（NK-merge 后 224KB/任务对，fp16 hi+lo = fp32 原字节）。
- 突破需要什么（上游修复项）：V↔C 片上直连（跨核共享存储或 L1 直达 DMA）——工具链立项级；
  或 C 波次化手动流水（HISA 式，`examples/HISA/paged_block_sparse_mqa_attn_expert.py` 模式）可压向
  max-engine **~2.9 us/任务**，但属全量 Expert 重写（L1 2 深槽 448KB/512KB razor + 手动 flag 协议，
  家族 4 次尝试 3 失败先例，见 §3.2 限制 6）。
- 地板：不波次化 ~10.9us/任务 = **4.47 ms 墙**；波次化理论 ~1.2-2.9us/任务 + 协议开销。

**【限制 4】launch 形态锁死 —— grid=40 无效**
- 限制描述：block 形态仅两种（KERNEL_TYPE_MIX_AIC_1_1 / MIX_AIC_1_2，由 cv_ratio 决定），每 block
  1 AIC + 1/2 AIV；无 AIV-only block、无稀疏 C 配比 → grid=40 为两波串行 20（每波仍 20 AIV）。
- 证据（源码）：`src/target/codegen_ascend.cc:858-860`。
- 突破需要什么：上游新增 AIV-only 或非共享 vid 的 1:2 形态——解锁后即 threads=2 路径（归入限制 1）。
- 地板：无独立地板（路径并入限制 1）。

**【限制 5】目标隐含 88% HBM 唯一 IO 效率 > 家族实证最佳 81%**
- 限制描述：唯一 IO 3.758GB ÷ 3400us = 1105GB/s = 峰值 1263GB/s 的 88%。
- 证据：DESIGN §9.5（IO 明细与家族标定 k1/k2 81% 最佳）；R2 §4 带宽核算 555GB/s=44%（do_bench 口径）。
- 突破需要什么：**架构物理上限**（88% 效率无家族先例；当前架构下中继流量与唯一 IO 共享内存系统）。
- 地板：**3673 us @81%** / 2975 us @100%（绝对物理界）。

**【限制 6】工具链指令/规划器缺陷族（多项独立上游修复）**
- ① `T.tile.mul_add_dst` 在规划器管理的 UB 上下文不可用：CANN 9.2.0
  `kernel_operator_vec_binary_intf.h:249-250` `__ASC_USE_RESERVED_UBUF__(3510)` 受限标注；本会话实测
  ~1% 确定性偏差（l0 dg 0.9852/0.9857，分离二分排除布局扰动；微测语义本身正确）→ op 融合路线封死一块。
- ② in-place broadcast→sub→exp 链崩溃（ACL 507014 设备挂起，r3_iter3 实证）。
- ③ reduce_2d dim=1 恰好 [64,64] 静默损坏（R1 iter5，DK/DV 矩阵实证；DV≥128 guard 防御中）。
- ④ memory-planner 别名 + 名字级同步使合法语句重排高风险：三战役八例（R1 iter6 a/b/c、R2 iter4/5、
  R3 iter3 崩溃 + iter2/iter3 布局扰动对照）。
- 各为独立上游修复项；④修复后 produce/drain 跨相位重叠与批处理装载家族可重开（R2 iter4: -2.55%
  方向为正未达线）。

### 3.2 可达性合成（各路径地板对照）

| 配置 | 可达地板 | 依据 |
|---|---|---|
| Developer 当前（终态 6767/6639-6720） | **已贴线**（V 串行链地板 6.5-6.8ms） | R2 §3.1 + R3 复测 |
| + threads=2（**若**限制 1 三项上游修复） | ~4.4 ms（C 墙） | 限制 1 + 限制 3 |
| Expert-V（手动同步 + UB 手工装配，单侧） | **3.7-5.0 ms** | vchain 微实验 + UB 核算（限制 2） |
| Expert-V + Expert-C（双侧手写波次，razor-edge 理论） | 2.9-3.4 ms | max(V-mte2 2867, C-max-engine ~1208+) + 协议开销；需 UB 194KB + L1 448KB + 手动协议全部 razor 成立 |
| 绝对物理界（HBM 唯一 IO 100%） | 2975 us | §9.5 |

**判定：3400us 落在"双侧全 Expert 手写波次"的理论 razor 边缘（2.9-3.4ms）之内、但其现实置信区间
（3.7-5.0ms）之外**。任何单项解锁（threads=2 或 C 波次化或 V 重叠）单独均不足；需 ≥2 项联合
（threads=2 上游修复 + C 波次化/片上直连，或 V+C 双侧全手写 razor 装配）。后者为家族最高风险区
（手动协议 4 试 3 败）且理论余量 0-15%，在本战役内有界工程量下不具可信成功概率——如实披露，
不做破坏精度/正确性的达标动作（家族政策）。

## 4. 有效带宽与目标差距（如实，承继 R2 + R3 无变化）

| 口径 | R3 终态 |
|---|---|
| 唯一 IO（3.758 GB）÷ do_bench 均值（6639-6767） | 555-568 GB/s = 44-45% |
| 逻辑流量（唯一 + L2 吸收中继）÷ 6534us | ~1.29 TB/s ≈ 102%（L2 承担主要中继） |
| 3400us 目标隐含（唯一 IO） | 1105 GB/s = 88%（> 家族实证最佳 81%） |

## 5. R3 产出文件清单（perf_tuning/，fla_chunk_inter_ 前缀）

| 文件 | 内容 |
|---|---|
| fla_chunk_inter_final_summary.md | 本报告（R3 整体刷新，覆盖 R2 版；R2 数据保留于 git 历史） |
| fla_chunk_inter_perf_log.md | 跨轮日志（追加 R3 段：iter0 锚点 / iter1-3 / vchain 附实验 / 终态） |
| fla_chunk_inter_baseline_r3_iter0.json | R3 会话基线（6719.8us + 恢复后复测带 6639.1） |
| fla_chunk_inter_vid2_micro.py | **决定性实验 1**：threads=2 vs drain 链（5 子链 ×2 形态，fp64 golden） |
| fla_chunk_inter_vchain_micro.py | **决定性实验 2**：AUTO_SYNC vs 手动突发协议 yield 定量（-3.1%） |
| fla_chunk_inter_ab_r3_base.py / _recheck.py | 会话基线 / 设备恢复后复验 A/B 脚本 |
| ../history_version/fla_chunk_inter_perf_r3_start.py | R3 起点快照（= R2 终态，md5 db1d9e8a） |
| ../history_version/fla_chunk_inter_perf_r3_iter{1,2,3}_before.py | 各子轮改前快照（iter1/2/3 均回滚 → 工作区 = start） |

主实现：`../fla_chunk_inter.py`（与 R2 终态逐位一致，接口不变：
`fla_chunk_inter_bwd(q,k,v,g,h,dv,do,dh,scale,chunk_size)`）；测试未改动。
DESIGN 增补 §13（R3 结构级结论 + 限制清单索引）。

## 6. 对上游的建议（优先级序，R3 修订）

1. **threads=2 vid 语义三项修复**（限制 1：broadcast vid 感知 / 跨 vid reduce 合并 / per-buffer 切分轴
   标注）——解锁 40 AIV，V-mte2 地板 2867→1434us；与 2 联合后目标重入可达域。
2. **V↔C 片上直连路径**（限制 3）——消 C 侧 mte1/mte2 中继 staging（2.1ms/8192 任务对）；
   或 C 波次化调度支持（手动 flag 协议的正确性护栏）。
3. **memory-planner 别名感知同步**（限制 6-④）——解锁 produce/drain 跨相位重叠与批处理装载家族
   （R2 iter4 -2.55% 方向证据）；三战役八例实证支撑立项。
4. `T.tile.mul_add_dst` 保留 UBUF 机制与规划器共存修复（限制 6-①）；in-place tile 链 sync-insert
   修复（6-②）；reduce_2d [64,64] 修复（6-③，R1 遗留）；跨 dtype GM→UB 转换 DMA（消拆分链，
   R2 建议 ③ 重申）。

---

**判定：目标 3400 us 未达成（R3 终态 ~6767us do_bench 档案值 / 6639-6720us 会话带，与 R2 终态逐位一致，
超 1.99×）→ [PERF_INCOMPLETE]。根因分类"工具链/架构限制"（§3.1 六项清单，逐项含证据与突破条件）：
绝对 HBM 物理界（2975-3673us）低于目标，但每条通往它的路径（threads=2 / Expert 双侧手写 / C 直连）
均被≥2 项独立上游修复或 razor-edge 手工装配条件卡死——单点修复不改变可达域，联合修复方可。**

---

# R4 增补（2026-11-03，收敛战役终版，替换 R3 §0 结论速览的时效部分）

## R4.0 结论速览

- **终版 kernel：6868.3 us**（bench (1,8192,64,128,128)，do_bench warmup=20/rep=10/
  mean/仅设备侧，5 次独立：6871.9/6859.0/6855.8/6881.8/6873.0）
- vs D-2 修复基态 7510.3（3 次独立同口径）：**-642.0us / -8.55%**（回收修复代价的 86.5%）
- vs R2 档案基线 6767：**+101.3us / +1.50%——在 ±3% 收敛带 [6564, 6970] 内，R4 收敛**
- vs proto 目标 3400：仍超 2.02×（该目标在结构地板下不可达，见 R3 §3 限制清单 +
  R4 §15.2 结构地板归因——本节不重复）
- 精度：--level all 全绿（L0+L1 40 PASS ratio 全 1.0000；valrange_l dg 7.812e-3
  ≤ 帽 1e-2 与 D-2 基态持平；bd_dbound 为设计预期 WARN 非阻塞）

## R4.1 采纳链（3 项，逐项 A/B 实测）

| 迭代 | 优化 | A/B（同会话同口径） | 判定 |
|------|------|--------------------|------|
| R4-2 | dgkn ones-gemm 对角化移 C 侧（3 hi/lo gemm + dgm 中继 + drain 一次 gather 对角提取；删 produce 侧 mul/transpose/树整链） | 7503.7→7167.1（-4.49%） | 采纳 |
| R4-3 | gn 门控向量化（转置 + row_expand_sub brcb 单 Sub + 保号，bDK=64 路径；bDK=32 保留行链） | 7167.1→7012.3（-2.16%） | 采纳* |
| R4-4 | drain 三中继读融合（dq\|dk\|dgm → 单 ddr_relay 行段，3×16KB→1×48KB 单读） | 7012.3→6868.3（-2.05%） | 采纳* |

\* R4-3/R4-4 幅度 < 3% 默认噪声阈，但同会话独立均值散布 ≤0.5%、候选/基线区间
完全不交叠（~10-14σ）——按 mandate"无硬性数字门禁，按证据推进"条款采纳，
偏差已在 perf_log/optimization_log_r4 显式标注。

回滚记录：无（三项全部一次通过精度门禁；R4-4 过程中一次编辑损坏即时从快照恢复，
未产生错误测试结果）。

## R4.2 剩余差距结构归因（+101us，全部实证封堵，详 DESIGN §15.2）

1. dgm 中继回读 ~291us（mte2 带宽界）：C 侧无 gather、fixpipe 仅连续写、gemm 无法
   紧凑产对角（置换分析）——三死路实证。
2. 门控 2 次转置 ~328us：broadcast 触发 scratch 注入为 D-1/D-2 红线；转置+brcb 为
   无 scratch 行广播实测最小形态。
3. drain 树 ~170us：wholereduce 8-block/repeat 不跨 repeat 累加（codegen 语义实证）、
   sum_experiment 轴不符、转置反超。
4. mte2 字节地板 416KB/task（输入+中继全为结构必需，53GB/s 引擎界）。
→ **Developer + scratch-free 双约束形态地板 ≈ 6868us**；6767 为允许 scratch 注入
op 时的形态（工具链修复 planner 指针序后可回收 ~500us 级，见 debug_log next_hint）。

## R4.3 产出文件（R4 新增）

- perf_tuning/fla_chunk_inter_optimization_log_r4.md（Part A + ORDER-PLAN + 逐项记录）
- perf_tuning/fla_chunk_inter_perf_log.md R4 段（iter2/3/4 结构化记录）
- perf_tuning/msprof_output_r4_iter3/ + msprof_r4_iter3.run.log（iter3 终态画像）
- history_version/fla_chunk_inter_perf_r4_iter{2,3,4}_{before,after}.py（快照链）
- DESIGN §15（R4 结构级增补）；debug_log Attempt R4（含死路与过程事故记录）
- 终版 kernel：examples_experiment/dsa_sparse_finetune/fla_chunk_inter.py
  （md5 703d5914b29fa844788e4b4b3e68b983）

## R4.4 G3 交付门（终态）

- 交付件重生成：tilelang-fla-chunk-inter-pr/examples_experiment/dsa_sparse_finetune/
  example_fla_chunk_inter.py（终态 kernel 代码 + R4 版清洁 docstring，**保留 MAINTENANCE
  NOTE 与"勿重新引入 reduce/broadcast"警示**）；test 文件接口未变无需同步。
- worktree 实测（source 主仓 set_env.sh 后于该目录）：
  `python test_example_fla_chunk_inter.py` → **"Kernel Output Match!" exit 0**
  （l0_min bDK=32 路径 + l1_bench (1,8192,64,128,128) bDK=64 路径全过；
  worktree 解析其自身 repo 源码构建 0.1.4+557b11b8，非崩溃旧 pip 版 d75300ed）。
- 红线终检：终态 kernel 两 shape 类（bDK=64/32）生成 C grep
  tmp_ub/Broadcast/reduce_2d/reduce_sum/sharedTmp/scratch/WholeReduce 均 **0 命中**。
- 交付件保持未跟踪工作树文件（agent 不 commit，用户自行提 PR）。
