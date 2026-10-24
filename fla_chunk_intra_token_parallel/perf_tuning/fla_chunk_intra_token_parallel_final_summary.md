# fla_chunk_intra_token_parallel 性能调优最终总结（final_summary）

- 日期：2026-10-24（三轮调优 iter1-12 后更新）
- 状态：**算子+DSL 层收敛于 1739.2us**（用户裁定持续压榨后，唯一剩余方向为 research 级
  relay 结构重构，见"收敛证据链"）
- 最终 kernel：**iter12 采纳版**（drain 暂存重构 + beta 向量化 + iter11 C 侧双槽流水基座，
  md5 a618cc211fa5782ba4a56b6c6a074a25）

## 计时口径（与上游 do_bench 对齐）

- benchmark shape：B=1, S=8192, H=64, DK=128, chunk_size=64, sub_chunk_size=16；
  输入 q/k/beta bf16 + gk fp32（chunk 内 cumsum，logsigmoid 构造）→ 输出 Aqk [B,S,H,64] /
  Akk [B,S,H,16] 均 bf16，scale=1.0（对齐上游 tilelang 测试样例 main()）
- 计时方法：torch.npu Event（enable_timing），**warmup=20 后 rep=10 取均值**，仅设备侧
  kernel 执行时间（不含张量构造/H2D-D2H/JIT/框架开销）；loop 模式（30 次 back-to-back）
  交叉验证，两口径差 <1%
- 平台：Ascend 910B3（20 AIC + 40 AIV，AI Core 1800MHz），实测 D2D 带宽上限 1263 GB/s
  （fla_wy_fast 调优期实测标定）

## 基线 → 最终（两轮调优全轨迹）

| 阶段 | 延迟（events mean） | 有效带宽 | 带宽利用率 | 状态 |
|------|---------------------|----------|------------|------|
| Stage 2 交付基线（iter0，纯 Vector） | 19150.1 us | 32 GB/s | 2.6% | — |
| iter1 热循环重构 | 9095.4 us | 68 GB/s | 5.4% | 采纳 |
| iter2 AUTO_SYNC=False 手动同步 | 8871.7 us | — | — | 回滚（-2.3% < 3%） |
| iter3 因式分解式重组（mid-anchor） | 6913.9 us | 90 GB/s | 7.1% | 采纳 |
| iter4 reduce_2d 三路径重组 | 无候选达标 | — | — | 结构性否决 |
| iter5 desync 叠加 iter3 | 6742.8 us | — | — | 回滚（-2.61% < 3%） |
| **一轮收敛点（6916.4us）→ 用户要求流水图深挖** | | | | |
| iter6 msprof 流水图分析 | 定位瓶颈，无 kernel 改动 | — | — | 分析轮 |
| iter7 **GEMM 卸载到空闲 Cube**（gemm_v0 ggrp=4 fp32） | 3517.7 us | 177 GB/s | 14.0% | 采纳 |
| iter8 **跨块 C/V 软件流水**（固定核网格 + NSLOT=2 双缓冲 + 手动/自动混合 flag） | **2125.0 us** | 293 GB/s | 23.2% | 采纳 |
| iter9 GGRP=8+kL0Size=64 / L1-L0C ping-pong / bf16 relay | 全否决 | — | — | 回滚 |
| iter10 mma 分解/手动 flag/E2 模板（编排分析轮） | 定论：瓶颈为等待**边界数**非 flag 条数（-27% 条数=0% wall）；A2 死锁/A3 V 侧竞态 | — | — | 分析+证伪 |
| iter11 C 侧双缓冲流水（L0C 双槽+零手写 flag） | 2130.0us（wall 0%） | — | — | 回滚（C 侧 10→4.93us 被 V 侧 10.34us 隐藏；V 侧归因表产出） |
| iter12 **drain 暂存重构 + beta 折叠五 op 向量化** | **1739.2 us**（复核 1737.8，稳态 rep 1666-1680；loop 1693.3） | 357 GB/s | 28.3% | 采纳 |
| **最终（独立复核 2026-10-24）** | **1737.8 us** | 357 GB/s | 28.3% | 收敛 |

**累计：19150.1 → 1739.2 us（-90.9%，11.0×）**；距纯带宽下限 492us 由 38.9× 缩至 **3.53×**。

## 主要优化手段及贡献

1. **热循环重构（iter1，-52.5%）**：T.macro 字面量参数 + parse 期 bool 守卫实现 i 轴静态
   展开（消灭 816 次 if-barrier）；per-(h,i) 行批量 tile 化 + reduce_sum 直写 staging
   （标量散写 1088→0）；h-major 计算域 + per-head 2D strided 聚合搬运（DMA 81→21 次/block）。
2. **因式分解式重组（iter3，-23.8%）**：e_ij = 2^(g_i-g_j) = E+_i × E-_j（mid-anchor），
   每 h 8 条 [16,128] 整行大 repeat 指令，per-i 链 11→7 op；溢出域经 DBOUND 实测重审。
3. **GEMM 卸载到空闲 Cube（iter7，-49.2%）**：iter6 msprof 发现 20 个 AIC 全程空转且 29%
   时间是行广播（Broadcast 指令 5.3× 慢于 Mul）；V 侧构造 A=[Q';K'']/B=K' 经 GM workspace
   中转，C 侧 gemm_v0(transpose_B) 批量算对角 [32,16] 块，V 侧 compare+select(VSEL) 掩码
   回写。bf16 操作数被 CPU 模拟精度否决（ratio 0.9848<0.99），fp32 直传。
4. **跨块 C/V 软件流水（iter8，-39.5%）**：固定核网格 T.Kernel(min(20,NBLK)) + 跨步任务
   映射（branch-free 越界重定向）；单循环三相位 V drain(t-2)→produce(t) ∥ C gemm(t)，
   2 任务延迟流水；NSLOT=2 双缓冲 workspace，o_relay 手动 FIX flag（交替 ID）避免自动
   FIFO 同迭代配对死锁。msprof 证实 AIC busy 80.07%→99.55%（C/V 重叠兑现）。
5. **drain 暂存重构 + beta 向量化（iter12，-18.8%，基于 iter11 C 侧双槽基座）**：drain
   暂存从整任务 [BH_V,SCS,*] 改 per-head [SCS,*] 复用（30,720B→7,680B，UB 总占用
   196,992→174,016B，余量 22.3KB）；beta 折叠 64 标量对 → one-hot 五 op 向量链
   （compare/select/reduce_sum/broadcast/mul，kernel_source 实证 GetValue=0）；C 侧
   L0C 双独立槽 + E2 手写序列实现 fixpipe(g0)∥mma(g1)∥下任务预取三路并行（C 侧
   10→4.93us/task）。V 侧 wall 10.34→~8.2us/task。附带修复 iter11 broadcast 源宽
   隐性缺陷（SCS=8 门禁抓获）。

## 最终流水状态（iter12 版）

- wall = V 侧 ~8.2us/task（beta ~0.5 + DMA 地板 ~5-6（relay 96KB fp32/task，精度强制）
  + 语义向量链 ~1.7-2.1（exp/cast 为算法本体）+ 重构开销 ~0.3）
- C 侧 4.93us/task 持续隐藏（V 侧独占墙）；AIC busy 99.55%

## 收敛证据链（算子层穷尽，iter4/5/9 三次专项否决 + iter6/9 框架封锁论证）

| 路径 | 否决证据 |
|------|----------|
| reduce_2d 重组（iter4） | dst 连续性墙 + planner 离线求值已最优 + split-merge 实测 +15.6% 证伪 |
| 手动同步/desync（iter2/5） | 双重实证屏障全消仅 ~2.6% 天花板（被流水掩盖） |
| ILP 交错（iter6） | 910B3 向量管不重叠独立指令（实测 -2.9% 链级 / +2.8% 交错劣化） |
| 行广播消除（iter6） | DSL binary_op 强制同尺寸、stride-0 mul 无原语（C++ helper mul_mask 存在但未注册）、brcb/gatherb/row_expand 六路全封锁——**需框架新增 tile 原语（tilelang-ascend-tile-api 流程）**；该 29% 已被 iter7 GEMM 卸载路径整体取代 |
| GGRP=8 + kL0Size=64（iter9） | common.h:1351 static_assert：L0A ping-pong 预算 64KB/kNumSteps，多步每槽 32KB 装不下 fp32 M=256；kL0Size=32 变体 mac×2+fixpipe×2 净负 |
| L1/L0C 分组 ping-pong（iter9） | 三重封锁：AUTO_CV_SYNC 位点数奇偶编译错 + L1 zN 分形布局 region 操作数静默腐蚀（Aqk 0.87）+ gemm_v0 模板内部 FIX_M 等待封死结构重叠 |
| bf16 workspace relay（iter9） | 精度位级全过（fixpipe 舍入 ≡ CAST_RINT）但实测 -4.7%（转换/编排开销 > 字节节省） |
| V 核 GM 预取（iter9） | AIC busy 99.55% = wall，稳态 C 从不等 V，预取零收益 |
| gemm 编排 flag 瘦身（iter10，含框架级） | A1 mma 分解 bit exact 但 0% wall——瓶颈是等待**边界数**（~12/task）非 flag 条数（-27% 条数=0%）；header-only 框架瘦身上限被封顶 |
| C 侧双缓冲流水（iter11） | C 侧 10→4.93us/task 技术达成但 wall 0%——被 V 侧 10.34us 完全隐藏（max(V,C) 不变）；A2 slot 化死锁、A3 V 侧竞态根因闭环 |
| beta 向量化直达（iter11） | UB 余量仅 960B（planner ExtendKillIndex 全量共存）+ 列抽取三路工具链缺陷（N=1 gm→ub 损坏/strided EE9999/broadcast 静默丢步长）——经 iter12 drain 重构绕开 |

**剩余提速空间（iter12 判定）**：
1. V 侧 DMA 地板 ~5-6us/task（relay 96KB fp32/task）——bf16 relay 被 iter9 实测精度否决
   （-4.7% 劣化）+ 精度位级虽过但转换开销反噬；L2 常驻已做
2. 语义强制向量链（exp/cast）为算法本体，不可消除
3. **唯一 wall 级方向 = relay 结构重构（research 级）**：理论地板 ~1250-1450us
   （-15~28%），无合法压缩路径已探明（bf16 封死、L2 已做、跨 dtype DMA 工具链缺失），
   高投入低回报
4. 框架级立项（gemm_v0 模板编排、stride-0 广播原语、region 依赖分析、跨 dtype DMA）
   对本算子 wall 收益趋零（C 侧已隐藏），优先级让位于 relay 重构

**剩余提速空间全部为框架级立项**（超出算子文件红线，建议另立 issue）：
1. gemm_v0 模板 flag 编排瘦身（scalar 30.6% 主导，wall ≈ 10us/task 中 ~6us 编排）
2. stride-0 广播乘原语（mul_mask 注册 intrinsic）
3. region 精确依赖分析（解锁 L1/L0C ping-pong）
4. 跨 dtype GM→UB DMA 转换（当前 UB 内两段式 cast 的开销源头）

## 与性能目标的差距（如实披露）

- 目标：纯 Kernel 总耗时 < 370 us（用户验收基准）
- 实测最终：**1739.2 us（复核 1737.8，稳态 rep 1666-1680），为目标值的 4.7×；距纯带宽
  下限 492 us 尚有 3.53×**
- 根因：① IO 契约下限 593 MiB → 370us 隐含 1.68 TB/s = 实测 D2D 上限的 1.33%（设计期
  论证物理不可达，即使 gk 改 bf16 也需 386us@100% 带宽）；② wall = V 侧 ~8.2us/task
  （DMA relay 地板 ~5-6us 精度强制 + 语义向量链），C 侧 4.93us 已隐藏；唯一 wall 级
  剩余方向为 research 级 relay 结构重构（理论地板 ~1250-1450us，仍达不到 370us）。

## 精度验收（最终版独立复核 2026-10-24）

- `--level all` exit 0：**13 PRECISION_PASS + 14 BOUNDARY_PASS / 0 FAIL**
  （L0 3/3 + golden 自检 + L1 9/9 含 H=2/6/12、DK=48/32、SCS=8 泛化 + L2 10/10 正确拒绝
  + Boundary 4/4 含深衰减/inf/nan 用例），全部 ratio=1.0000，max_abs ≤ 6.25e-2（cap 1e0）
- bf16 输出混合容差：atol 2^-10 / rtol 2^-6 / max_abs_limit 1e0 / matched_ratio ≥ 0.99

## 产物索引

- 基线/候选 JSON：`fla_chunk_intra_token_parallel_{baseline,candidate}_iter{1..12}.json`
- 分解剖析：`..._decomposition_iter3.json`、iter6 veccal/bcastshare/ilp_cal、iter7/8 msprof
  快照（`prof_iter6/`、`prof_iter7/`、`msprof_summary_iter{6,7}.json`）
- 探针：`..._splitreduce_iter4.py`、`..._{ilp_cal,veccal,bcastshare}_iter6.{py,json}`、
  iter10/11 编排 dump 与 mini 复现（`..._a1_source_iter11.c` 等）
- 生成代码：`..._kernel_source_iter{1,2,3,5,12}.c`
- 迭代日志：`..._perf_log.md`（Iteration 1-12 全记录）、`..._optimization_log.md`
  （含 30+ 项工具链发现，skill journal 候选清单）
- 回滚备份：`..._impl_iter{1..12}_before.py`、`..._impl_iter5_desync_candidate.py`、
  `..._impl_iter{7,8,11}_candidate.py`
