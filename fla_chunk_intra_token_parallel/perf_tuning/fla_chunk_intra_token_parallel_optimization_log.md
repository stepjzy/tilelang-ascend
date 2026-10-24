# fla_chunk_intra_token_parallel — Stage 3 优化记录（optimization_log）

> 算子类型（Step 2 判定）：**纯 Vector（AIV）**。`get_kernel_source()` 无 IS_ASCEND_AIC，主体在
> `if ASCEND_IS_AIV` 内（见 fla_chunk_intra_token_parallel_kernel_source_iter1.c）。
> 计时口径：torch.npu.Event(enable_timing=True)，warmup=20 / rep=10 / mean，仅设备侧
> （对齐 ref_gpu/test_utils_kda.py do_bench；task 指令允许的 NPU 等价实现，loop 模式交叉验证
> host 分发污染）。基线实测见 baseline_iter1.json。

## 基线诊断（Step 1-2，2026-10-22）

- baseline: **19150 us** @ (1,8192,64,128)；有效带宽 32 GB/s = 实测上限 1263 GB/s 的 2.6%；
  纯带宽下限 492 us（39×差距）。per-rep 方差 <0.03%，events≈loop（非 host 分发受限）。
- 生成 C 代码静态剖析（fla_chunk_intra_token_parallel_kernel_source_iter1.c）：
  - 每对标量散写 `aqk_s.SetValue` ×8（含 SetFlag<V_S>+WaitFlag+PipeBarrier<ALL>）
    → 136 对/block ≈ **1088 次标量写 + 864 组 V_S flag 对**
  - `ascend_sync_insert.cc`（AUTO_SYNC）对**同管线相邻访问插入 PipeBarrier_V**
    （GetRequiredSyncType: same-pipeline → PipeBarrier_<pipe>），~700 次/block
  - **每个 IfThenElse 前后各插 PipeBarrier_ALL**（VisitStmt_(IfThenElseNode)），当前热循环
    i/j guard 结构 ≈ **816 次 ALL barrier/block**
  - j>i 空迭代仍付 2 次 ALL barrier（120 次/block）
  - 载入 49 次 DMA/block（48 行拷 + beta）、写回 32 次/block
- 估算：纯向量计算 ≈ 1.9-2 ms（20 核并行）；barrier/标量/issue 开销 ≈ 17 ms（~90%）。
  **瓶颈分类：sync + 标量散写 + 指令下发（issue）开销主导**（compute-overhead bound，
  非带宽、非纯算力）。

## Part A 优化点清单（Step 3）

[#1] **热循环重构：消灭标量散写 + 运行时 if + 行批量 tile 化**（参考: optimization-guide.md §2.6
指令向量化、§2.7 指令融合；performance-antipatterns.md "Vector Core 内逐元素/逐行 for loop
计算"、"正交轴串行化"）：**适用** — 标量散写（aqk_s/akk_s SetValue ×1088/block）与 if-guard
（816 ALL barrier/block）是 barrier 风暴主因。改法：i 轴 Python 展开（`for i in range(SCS)`，
trace 期常量，消除运行时 if），per-(h,i) 对 j∈[0,i] 整批 tile 化：
`broadcast(g_i/q_i/k_i 行) → sub/exp/mul → reduce_sum(dim=-1) 直接写入 staging region`
（aqk_s[i,h,w:w+i+1] 连续 1D run），β 经 reduce 后按行 broadcast 折乘（Akk_ij = β_i × Σ k_i⊙kgj，
数学恒等，fp32 舍入路径变化 ≪ bf16 门禁）。op 数 950→~640，标量写 1088→0，运行时 if 0。
反模式"有意保留项"：无（本轮全部消除）。

[#2] **h-major 布局 + per-head 2D 聚合搬运**（参考: optimization-guide.md §2.11 扫描轴批量搬运；
DESIGN §10 R6）：**适用** — [#1] 的 per-(h,i) 2D tile（g2[h,0:i+1,:] 等）要求 [BH_V,SCS,DK]
h-major 连续布局；随之载入从 48 次行拷改为 12 次 per-head 2D strided 拷（[16,DK] burst，
GM 侧行距 16KB，先例：现行 beta 行间 gap 拷已验证 strided+strided 可行），写回 32 次 flat burst
改为 8 次 per-head 2D strided（burst 128B/32B ≥32B 对齐）。DMA 事务 81→21 次/block。

[#3] **AUTO_SYNC=False + 手动最小同步**（参考: optimization-guide.md §2.2.0 决策表、§2.10）：
**暂不实施（iteration 2 候选）** — 决策表明示纯 AIV 流水需 AUTO_SYNC=False，但本轮先隔离变量
（[#1][#2] 后 V-V barrier ~640 次/block 仍在，若实测仍显著再动同步配置，需手动补
mte2→v / v→mte3 flag 并全量精度复验）。

[#4] **因式分解式（e_i·e'_j 锚定，-30% ops / -77% exp）**（参考: DESIGN §10.4-C、§9.5）：
**暂不实施（后续迭代候选）** — 溢出域变化（subchunk 内衰减 >~246 时 inf×0=NaN vs 直接式恒良
定义），需 Boundary 重审 + 域收窄文档化；且 exp 是否为剩余瓶颈需 [#1][#2] 落地后实测确认。

[#5] **Fixed Core / launch 形状**（参考: optimization-guide.md §2.9）：**暂不实施** — 4096 逻辑
block / 20 核属正常调度（生成代码 `<<<4096>>>`），基线瓶颈已定位为 barrier 非调度；[#1][#2]
后再评估。

[#6] 归约遍数融合（vector_reduce_pass_fusion.md）：**不适用** — 归约维 DK=128 单遍可容纳
（C1/C2/C3 不满足，X1 命中：一次装入 UB）。

[#7] 指令融合 mul+add（axpy/mul_add_dst）：**不适用** — 热路径为 mul→reduce 结构，无 mul+add
拼接模式。

[KERNEL-REUSE-PRECHECK]
candidate: [#1][#2] 计算重构 + h-major 布局
expanded_domain: 无新增调用域（同 shape/dtype 契约；SCS/BH 自适应路径不变，DK 32/48/128 均为
32B 对齐列宽）
compatibility_dimensions: broadcast axis=0（[1,DK]→[R,DK]，AscendC::Broadcast dim=2 axis=0 支持，
前端已验证校验路径）；reduce_sum dst 为 3D buffer 的 1D region（前端签名支持 BufferRegion）；
per-head 2D strided copy（GM point-H region → UB 连续，先例：beta 行间 gap 拷）
unknown_or_false: none（均为前端签名内构造，编译+L0 首验）
plan: direct-reuse（kernel 内部重构，host 接口不变）

## Part B [ORDER-PLAN] 实施顺序

1. [#1] 热循环重构 — 前置依赖: 无 — 理由: 标量散写+if-barrier 为最大瓶颈源（~17ms/19ms）
2. [#2] h-major 布局 + 聚合搬运 — 前置依赖: [#1] — 理由: per-(h,i) 2D tile 需 h-major 连续
   布局；布局与批量 tile 化耦合，作为同一候选实现（[#1] 的 region 均以 [#2] 布局表达）
3. [#3] AUTO_SYNC=False — 前置依赖: [#1][#2] — 理由: 隔离变量，待新结构实测后评估剩余
   V-V barrier 占比
4. [#4] 因式分解式 — 前置依赖: [#1][#2][#3] — 理由: 需先确认 exp 是否仍为瓶颈 + Boundary
   重审义务

## 实施记录

[ORDER-CHECK] 准备实施: [#1]+[#2] 联合（计算重构 + h-major 聚合搬运） | 前置依赖: 无 | 结论: ✅

## Iteration 1 实施记录（[#1]+[#2] 联合）

[IMPL-#1#2] 已阅读 optimization-guide.md §2.6/§2.7/§2.11（扫描轴批量搬运）、§2.2.0（同步决策表）、
performance-antipatterns.md（逐行 for loop / launch core / 纯 AIV 未流水 各条目）、
vector-practices/vector_reduce_pass_fusion.md（C1-C4 适用条件，判定不适用）、
reduce_ascend.py / ascend_tile.py 前端签名与校验逻辑、ascend_sync_insert.cc（barrier 插入规则）、
codegen_ascend.cc EmitSelectedRawScalar（ascend_muls 标量路径）、
tl_templates/ascend/common.h copy_gm_to_ub/copy_ub_to_gm（32B slot 语义）、
reduce_2d_v2.h（tmp/parity 机制）。关键约束: tile op 前端只接受静态 extent（符号 extent 断言失败）；
binary op region 必须平坦连续（部分列切片被拒）；VEC 源/目地址 32B 对齐；MTE 两侧行宽 ≥32B；
copy lowering 丢弃中间维 point 切片行距。

[SELF-CHECK] 本次 Edit 只涉及 [#1][#2]（kernel body 重构 + 布局，host 接口/pass_configs 不变）

[RESULT-#1#2] 优化点: 热循环重构（标量散写/if-barrier 消灭 + 行批量 tile 化 + h-major 聚合搬运）
| 精度: pass（--level all exit 0，ratio=1.0000 全绿）| 性能: 9095.4 us | 对比: **-52.5% vs 基线 19150.1 us**

### 实施过程的关键技术发现（供后续轮次与 skill journal 沉淀）

1. **`for i in range(N)` + `if`（cumsum_kda 先例模式）在 tvmscript 中生成 TIR 循环 + 运行时分支**，
   AUTO_SYNC pass 对每个 IfThenElse 前后各插一次 PipeBarrier<PIPE_ALL>（ascend_sync_insert.cc
   VisitStmt_(IfThenElseNode)），对每个标量 SetValue 插 SetFlag/WaitFlag<V_S> + PIPE_ALL——
   这是基线 ~2700 barrier/block 的结构性来源。**`T.macro`（字面量参数）+ parse 期 bool 的
   `if {i} < SCS:` 守卫调用点**可实现零运行时分支的静态展开（parser visit_if 对
   isinstance(condition, bool) 走真分支）。
2. **VEC 指令的 UB 源/目地址必须 32B 对齐**：per-(i,h) 标量 β 广播（beta_f[i,h,:] 4B 粒度地址）
   → aicore 507015 "UB address accessed by the VEC instruction is not aligned"。
   ascend_muls 的 BufferLoad 平坦索引路径（GetValue）是标量读的合法替代，但每次调用附带
   PipeBarrier<PIPE_ALL>（codegen_ascend.cc EmitSelectedRawScalar）。
3. **UB 内 copy 两侧行宽均须 ≥32B**（MTE burst 32B 语义）：β 紧凑 copy（[SCS,BH_V] 4 fp32=16B 行
   dst）→ aicore exception。β 保留垫宽 [SCS,16] 布局（32B 行）+ 平坦索引 i*16+h 读取。
4. **copy lowering 丢弃中间维 point 切片的物理行距**：写回 src `aqk_h[:, h, :]`（3D 中维 point）
   被按连续 flat 读（模板 srcN=末维 extent、srcStride=0）→ 全部输出值错位。修复：staging 翻转
   h-major [BH_V,SCS,CS]，per-head 切片 [h,:,:] 连续。
5. **窄行 MTE3（UB→GM）源行必须落在 32B pitch**：SCS=8 时 Akk staging 8 bf16=16B 行距 → 写回
   按 32B/burst 推进错位（Akk 全错）。修复：Akk staging 垫宽 max(SCS,16) 列 + 写回读 [0:SCS) 列区。
6. reduce_sum 支持 dst 为 3D buffer 的 1D region（[point,point,run] 平坦连续段）+ real_shape
   嵌入校验（_shape_can_embed_expected_shape 跳过前导 1）——staging 直写（零标量散写）可行。
7. broadcast 前端：3D region 折叠到末 2 维（前导 point 维保留 extent-1）→ [1,BH_V] vs [BH_V,1]
   形状失配；1D→2D/2D→2D（axis=0/1）均支持，等形退化走 Muls×1 拷贝路径。

### 结果文件

- 基线：perf_tuning/fla_chunk_intra_token_parallel_baseline_iter1.json（19150.1 us / 32 GB/s / 2.6%）
- 候选：perf_tuning/fla_chunk_intra_token_parallel_candidate_iter1.json（9095.4 us / 68 GB/s / 5.4%）
- 本轮备份（回滚基线）：perf_tuning/fla_chunk_intra_token_parallel_impl_iter1_before.py
- 生成代码剖析样本：perf_tuning/fla_chunk_intra_token_parallel_kernel_source_iter1.c（候选版）

---

# Iteration 2（2026-10-22）

## 基线复测（Step 1）

- baseline 复测：**9077.3 us**（events mean；per-rep 8999-9015，方差 <0.2%；loop 9023.5）
  —— 与 iter1 记录 9095.4 us 一致（-0.2%，噪声内），基线确认。
- 算子类型（Step 2，沿用）：纯 Vector（AIV），生成代码主体在 `if ASCEND_IS_AIV` 内。
- 生成代码静态剖析（iter1 候选版 kernel_source_iter1.c）：h_1 循环体内
  **152 次 PipeBarrier<PIPE_V> + 15 次 PipeBarrier<PIPE_ALL>**，×4 次 h 循环
  = **~608 次 V-V barrier + 60 次 ALL barrier / block**（4096 block ≈ 2.5M + 245K 次）。
  barrier 来源定位：
  1. ascend_sync_insert.cc GetRequiredSyncType：同管线 RAW/WAR/WAW → 无条件
     `PipeBarrier_<pipe>`（V-V 608 次）；
  2. codegen_ascend.cc EmitSelectedRawScalar：β 标量源（access_ptr 路径）每次
     `PipeBarrier<PIPE_ALL> + GetValue`（60 次，**codegen 层发射，与 AUTO_SYNC 无关**）；
  3. 边界 MTE2/MTE3 flag 对（少量，合理）。
- **瓶颈分类（Step 2）：sync（V-V barrier 风暴 + β 标量 ALL barrier）**，与 iter1 判定一致。

## Part A 优化点清单（Step 3）

[#3] **AUTO_SYNC=False + 手动最小同步**（参考: optimization-guide.md L106-L125 §2.2.0 决策表、
L230-L262 Vector 三路 flag 时序、L272-L287 §2.2.1 buffer 分配规则；api-schedule-sync.md §4
同步原语；ascend_sync_insert.cc L68-72[L72]——AUTO_SYNC=False 时 pass 直接返回不插入）：
**适用（本轮主优化）** — 决策表明示纯 AIV Vector 算子必须 AUTO_SYNC=False。同管线 V-V barrier
的正确性论证：
  a) optimization-guide §2.2.1："V pipe 单条指令链天然串行，同 buffer 读写安全"；
  b) 仓内先例 opt_gdn_solve_tril.py（AUTO_SYNC=False）存在长链连续 V op + in-place
     `sub(o_ub[i,:], o_ub[i,:], red_ub)`（dst==src）且无任何 V-V barrier，测试通过；
  c) reduce_2d_v2.h 内部自带顺序保证（WaitForVectorData → pipe_barrier(PIPE_V)/Delay 内嵌）。
  §2.13 "dst 需交替" 警告与 §2.2.1 冲突，按证据权重采信 §2.2.1（先例 + 决策表），
  若精度失败则首选排查此点。手动同步方案（每 block 4 对 set/wait，共 8 条指令）：
  - `v→mte2`：fill(beta_h)[V] 完成后 β GM 载入[MTE2] 才能写 beta_h[:, 0:BH_V]（WAR）；
  - `mte2→v`：全部 GM 载入完成后首个 V 计算才开始（RAW）；
  - `v→s`：beta_f[V 写] 稳定后标量读（cumsum 先例模式）；
  - `v→mte3`：cast 到 bf16[V] 完成后写回[MTE3] 才能读 aqk_h/akk_h（RAW）。
  跨 block UB 复用安全性：块间由 runtime 串行化（opt_gdn/moe_token_permute/mha_sink 等
  AUTO_SYNC=False 算子均依赖此语义）。

[#3b] **β 标量读消除（PIPE_ALL barrier 清零）**——[#3] 的伴随修改：codegen 的
EmitSelectedRawScalar 对 access_ptr 标量源无条件发 PipeBarrier<PIPE_ALL>（L3010-3017，
已读源码确认），关 AUTO_SYNC 无法消除。方案：前端 binary_op 对 src1 的分派
（ascend_tile.py L1073-1085）——raw BufferLoad → access_ptr 路径（barrier）；
**非 BufferLoad 的 PrimExpr → CoerceScalarArg 路径（纯 PrintExpr，无 barrier）**。
改法：h 循环顶部把 beta_f[i*16+h, 0] 以 `* 1.0` 算术包裹读入 TIR 标量局部
（parse 期 guard `if i < SCS` 保 SCS<16 域不越界），热循环 T.tile.mul 传 PrimExpr——
每次 Akk 行省 1 次 ALL barrier + 1 次内联标量往返（60 → 0 次 ALL barrier/block，
标量读保留但无 barrier，探针已验证 lowering 形态）。
反模式对照：performance-antipatterns.md "冗余全局同步"（同步被循环次数放大）——直接命中，
本轮消除。

[#4] **因式分解式（e_i·e'_j 锚定）**（参考: DESIGN §10.4-C、§9.5）：**本轮不实施** —
溢出域变化（subchunk 内衰减 >~246 → inf×0=NaN vs 直接式恒良定义）需 Boundary 重审 +
域收窄文档化；[#3] 落地后 exp/算术占比需重新标定再决定优先级。留作 iter3 候选。

[#5] **num_stages/双缓冲/调度形状**：**本轮不实施** — [#3] 关闭自动同步后 MTE2/V/MTE3
三队列跨 block 自然重叠（4096 block/20 核，队列 per-core 持久）；显式双缓冲为后续候选
（若 [#3] 后仍有 load/compute 空泡）。

[KERNEL-REUSE-PRECHECK]
candidate: [#3]+[#3b]（同步配置 + β 标量路径，kernel 内部改写）
expanded_domain: 无新增调用域（host 接口/shape/dtype 契约不变；SCS/BH 自适应路径不变）
compatibility_dimensions: pass_configs AUTO_SYNC=False（Developer 模式合法配置，
opt_gdn/mha_sink/moe 先例）；T.set_flag/wait_flag 前端签名（ascend.py L233-270，
_pipe 含 "s"）；标量 PrimExpr src1（ascend_tile.py binary_op PrimExpr 分支）；
event 对 V_MTE2/MTE2_V/V_S/V_MTE3 均在 GetEventMapping 表内（operation_config.h L378-417）
unknown_or_false: none（探针 probe_iter2.c 已验证全部三点 lowering 形态）
plan: direct-reuse（kernel 内部改写，host 不变）

## Part B [ORDER-PLAN] 实施顺序

1. [#3]+[#3b] 联合（AUTO_SYNC=False + 手动 4 对 flag + β PrimExpr 标量）— 前置依赖: 无 —
   理由: 二者耦合于同一次 kernel body 重写（#3b 的 v→s flag 本身是 #3 手动同步方案的一环）；
   探针已消除 lowering 不确定性；改写后先 L0 快验再全量。
2. [#4] — 前置依赖: [#3] — 理由: 需 [#3] 后的实测瓶颈占比 + Boundary 重审义务。
3. [#5] — 前置依赖: [#3] — 理由: [#3] 已带跨 block 队列重叠，显式流水按需追加。

## 实施记录

[ORDER-CHECK] 准备实施: [#3]+[#3b] 联合 | 前置依赖: 无 | 结论: ✅

[IMPL-#3#3b] 已阅读 optimization-guide.md §2.2.0/§2.2.1/Vector 三路 flag 时序（L106-287）、
api-schedule-sync.md §4（set_flag/wait_flag 签名与 Set/Wait 配对铁律）、
performance-antipatterns.md（冗余全局同步/纯 AIV 未流水/逐元素 for loop 各条目）、
ascend_sync_insert.cc L68-72/L1590-1610（GetRequiredSyncType 同管线规则 + False 短路）、
operation_config.h GetEventMapping（V_MTE2/MTE2_V/V_S/V_MTE3 合法）、
codegen_ascend.cc EmitSelectedRawScalar（access_ptr→PIPE_ALL 无条件）+ CoerceScalarArg
（PrimExpr→PrintExpr）+ VisitExpr_(BufferLoadNode)（标量上下文 GetValue）、
ascend_tile.py binary_op（src1 BufferLoad vs PrimExpr 分派）、
opt_gdn_solve_tril.py / opt_gdn_chunk_cumsum.py（AUTO_SYNC=False 先例：连续 V 链无 barrier、
v→s 标量读模式）。关键约束: 每 set 必须有配对 wait（含初始/最终 token）；标量 PrimExpr
须用 `* 1.0` 包裹（raw BufferLoad 会被前端转 access_ptr → barrier 路径）；SCS<16 时标量
局部必须 parse 期 guard（beta_f 垫宽固定 16，越界读风险）。

[SELF-CHECK] 本次 Edit 只涉及 [#3]+[#3b]（PASS_CONFIGS AUTO_SYNC + kernel body 手动同步 +
β 标量路径；host 包装/宏结构/test 不变）

[RESULT-#3#3b] 优化点: AUTO_SYNC=False + 4 对手动 flag + β PrimExpr 标量（生成代码零
PipeBarrier）| 精度: pass（--level all exit 0，ratio=1.0000 全绿）| 性能: 8871.7 us
（复测 8863.0）| 对比: **-2.3% vs 基线 9077.3 us —— 低于 3% 阈值，已回滚**

实施过程的 3 个关键技术发现（probe/probe2 实证，供 skill journal 沉淀）：

1. **β 标量 src1 的三态 lowering**：(a) 同块预读局部（`b = beta_f[i, h] * 1.0` 传名）→
   CoerceScalarArg 路径，hoisted `float b = GetValue(...)`，零 barrier；(b) 内联表达式作
   **宏参数** → 被宏机制折叠回 raw BufferLoad → 前端 access_ptr 路径 → codegen
   PipeBarrier<PIPE_ALL> 回归（该 barrier 在 EmitSelectedRawScalar 无条件发射，与
   AUTO_SYNC 无关）；(c) raw BufferLoad 直接传参 → 同 (b)。唯一无 barrier 形态是 (a)。
2. **标量索引语义分裂**：前端 binary_op 的 BufferLoad 分支把 `indices_1[0]`（第一索引）
   当**平坦偏移**传给 codegen（`beta_f[i*16+h, 0]` → GetValue(i*16+h) 的 quirk 语义）；
   PrimExpr 路径走 `buffer.OffsetOf(indices)` 正规扁平化（`beta_f[i*16+h, 0]` →
   GetValue((i*16+h)*16)，**越界错位**）。跨路径迁移时必须改写为 `beta_f[i, h]`。
   （本轮曾产生此 bug，生成代码静态检查（GetValue 索引突变为 h*16+256）发现，L0 前修复。）
3. **性能假设证伪**：移除 ~670 barrier/block（生成代码静态确认为 0）仅 -2.3%（~210 us 总
   收益，~75 ns/barrier 或已被流水掩盖）——"sync 主导"结论对 iter1 采纳版结构不成立。
   对账：690 向量指令流/vid-item × ~63 ns × 204.8 item/核 = 8.87 ms ✓。真实瓶颈待 iter3
   分解剖析（reduce_2d 内部 Delay/pipe_barrier 124 次/item、小 repeat 指令 per-instr 开销
   ~1700 条 SetMask* 配置/item、exp ~8% 估算）。

回滚记录：fla_chunk_intra_token_parallel.py 已恢复为 impl_iter2_before.py（iter1 采纳版，
逐字节一致），回滚后 L0 复验 pass。候选版生成代码保留于
fla_chunk_intra_token_parallel_kernel_source_iter2.c（零 PipeBarrier 结构样本）。

---

# Iteration 3（2026-10-22）

## 分解剖析（Step 1-2，本轮第一优先）

- baseline 复测：**9071.4 us**（events mean；per-rep 8999-9017 稳定；loop 9022.9）——与 iter1/iter2
  记录一致（9095.4 / 9077.3，噪声内）。
- **差分探针**（fla_chunk_intra_token_parallel_decompose_iter3.py，模块级独立 kernel，与采纳版
  完全同构的 block/DMA/缓冲结构）：
  - P_dma（仅 loads+casts+fills+writeback）：**828.2 us（9.1%）**
  - P_vec（+完整热循环向量链，无 reduce）：**7009.5 us** → 向量链 = **6181.3 us（68.2%）**
  - P_full（采纳版，= P_vec + 124 reduce_2d/block）：**9065.2 us** → reduce = **2055.8 us（22.7%）**
- **瓶颈分类（实测定论）：向量指令流 issue/执行（compute/issue-bound，68.2% 主导）**；
  iter2 的重新归因被数据证实。reduce_2d 次之（22.7%，~40ns/调用·item，per-call 机制无异常）；
  DMA+调度 floor 仅 9.1%（含纯带宽下限 492us）——IO/调度非瓶颈。
- 生成代码静态剖析（kernel_source_iter3.c，每 h 迭代）：45 broadcast + 15 sub + 15 exp +
  46 mul + 16 muls = 137 条非 reduce 指令（行窗口 1..16 递增，元素量 1126 行×128/h）。

## Part A 优化点清单（Step 3）

[#7] **因式分解式重组（mid-anchor，DESIGN §10.4-C，剖析数据驱动）**（参考: DESIGN §9.5/§10.4-C；
optimization-guide.md §2.6 指令向量化、§2.7 指令融合、§2.13 归约模式；
performance-antipatterns.md "多行 tile 循环内逐块归约"关注项——下三角结构无法整体循环外归约，
以削减 per-i 指令数为主）：**适用（本轮主优化）** —
  - 数学：e_ij = 2^(g_i-g_j) = E+_i × E-_j，E+_i = 2^(g_i-g_a)，E-_j = 2^(g_a-g_j)，
    anchor a = SCS//2。Q'_i = q_i ⊙ E+_i、K'_j = k_j ⊙ E-_j、K''_i = k_i ⊙ E+_i 一次性计算
    （[16,128] 整行大 repeat），per-i 链退化为 broadcast + mul + reduce（Aqk）与
    broadcast + mul + muls + reduce（Akk）。
  - 指令数：每 h 非_reduce 137 → 85 条（**-38%**）；元素量 -31%；exp 元素量 136 行 → 32 行/h（**-76%**）。
  - UB：零新增（E-→K' 复用 kgj 缓冲、Q' 覆写 q_f slab（diag 后）、K'' 覆写 e（in-place mul））。
  - 溢出域重审（DBOUND 实测值）：用例 -12/token × 15 = -180 subchunk 跨度，mid-anchor 单侧
    最大 96（log2 域）< 127 fp32 上限；且 j ≤ i 保证 E+/E- 大值不同时同乘积 → 无 inf×0。
    安全域文档化：subchunk 半程衰减 |g_i-g_a| < ~127。
  - 精度：e_ij 由 1 次 exp 变 2 次 exp + 2 次乘法舍入（fp32 相对误差 ~1e-7 级 ≪ bf16 门限
    6.25e-2）；diag 快路径保留（原始 q_f/k_f，零精度损失）。

[#8] reduce_2d 重组（跨 i 合并调用/换原语）：**不适用** — 下三角 (i,j) 对的 pq 矩阵依赖 i
（共同 j 前缀对不同 i 值不同），无法合并为循环外单次归约；124 次/h 的实测均摊成本
（~40ns/调用·item）无异常放大，重组空间已被结构性封死。

[#9] DMA 聚合（13 load+8 write/block → 更少）：**不适用（本轮）** — P_dma 实测 828us（9.1%），
其中纯带宽下限 492us，DMA 侧总优化空间 <336us 且需改 GM 布局（3-active-dim 限制），
优先级远低于 68.2% 的向量链。

[#10] tile size 扩大（BH/SCS 上调）：**不适用** — UB 峰值 ~191.5KB/191.75KB 已打满
（5 个 [4,16,128] 主缓冲 + staging + scratch），无扩余量。

[KERNEL-REUSE-PRECHECK]
candidate: [#7] 因式分解式重组（kernel 内部计算链重写，host 接口/shape/dtype 契约不变）
expanded_domain: 无新增调用域（SCS/BH 自适应路径不变；DK 32/48/128 均兼容——anchor 行
  切片与 [16,DK] 整行 op 的 32B 对齐在 DK%16==0 域内不变）
compatibility_dimensions: T.tile.broadcast（[1,DK]→[R,DK] axis=0，前端已验证）；in-place
  mul dst==src1（AUTO_SYNC=True V pipe 串行安全，主 kernel mul(g2,g2,LN2)/exp(e,e) 先例）；
  reduce_sum dst 1D region（不变）；宏字面量展开（不变）
unknown_or_false: none（均为已验证前端构造，L0 首验兜底）
plan: direct-reuse（kernel 内部重构）

## Part B [ORDER-PLAN] 实施顺序

1. [#7] 因式分解式重组 — 前置依赖: 无 — 理由: 剖析数据指向向量链 68.2% 为最大杠杆，
   [#7] 是唯一 -38% 指令数的结构改动；reduce/DMA 侧无空间（[#8][#9] 已判定）

## 实施记录

[ORDER-CHECK] 准备实施: [#7] 因式分解式重组 | 前置依赖: 无 | 结论: ✅

[IMPL-#7] 已阅读 optimization-guide.md §2.6（L381-400）/§2.7（L402-426）/§2.13 归约模式
（L1024-1131，含 V pipe in-place 安全声明）、performance-antipatterns.md（多行 tile 逐块归约/
逐行 for loop/指令融合 各条目）、DESIGN §9.5（性能预算）/§10.4-C（因式分解选项与溢出域）、
reduce_2d_v2.h（StepCost4/WaitForVectorData 机制，确认 reduce 无 per-call 异常）、
test_fla_chunk_intra_token_parallel.py L431-438（DBOUND 衰减构造 -12×pos）。
关键约束: exp 输入 |x×ln2| < 88.7（fp32 上限）；anchor 行切片 [a:a+1] 的 32B 对齐
（DK%16==0 域内成立）；diag 必须在 q_f 覆写前执行；E- 先于 E+ 计算（e 缓冲三段复用
E- → E+ → K''）；SCS<2 时一次性链与热循环同 guard 跳过。

[SELF-CHECK] 本次 Edit 只涉及 [#7]（kernel docstring + 宏重构 + kernel body 计算链；
host wrapper/pass_configs/test 不变）

[RESULT-#7] 优化点: 因式分解式重组（mid-anchor E+/E- 拆分 + 一次性 Q'/K'/K'' 大 repeat slab +
per-i 链 11→7 op）| 精度: pass（--level all exit 0，ratio=1.0000 全绿，bnd_dbound 域内通过）
| 性能: 6913.9 us（复测 6917.3，per-rep 方差 <0.2%）| 对比: **-23.8% vs 基线 9071.4 us —— 采纳**

### 结果文件

- 基线：perf_tuning/fla_chunk_intra_token_parallel_baseline_iter3.json（9071.4 us / 69 GB/s / 5.4%）
- 分解剖析：perf_tuning/fla_chunk_intra_token_parallel_decomposition_iter3.json
  （P_dma 828.2 / 向量链 6181.3 / reduce 2055.8 us）
- 候选：perf_tuning/fla_chunk_intra_token_parallel_candidate_iter3.json（6913.9 us / 90 GB/s / 7.1%）
- 本轮备份（回滚基线）：perf_tuning/fla_chunk_intra_token_parallel_impl_iter3_before.py（iter1 采纳版）
- 生成代码剖析样本：perf_tuning/fla_chunk_intra_token_parallel_kernel_source_iter3.c（候选版，
  464 行；基线版结构数据见 decomposition json 的 static_census 字段）

### 本轮关键技术发现（供 skill journal 沉淀）

1. **TVM script parser 的闭包/bool-参数 if 静默失败**：`@T.prim_func` 函数体内引用外层闭包
   自由变量（工厂模式 `with_vec`）或 JIT 参数直接作 if 条件（`if WITH_VEC:`）时 parse
   静默返回 None PrimFunc（无 stderr），在 jit cache 的 `func.script()` 处才以
   AttributeError 暴露——差分探针必须用模块级独立 kernel 定义（iter2 探针同款模式），
   变体切换用独立函数而非参数开关。
2. **差分分解剖析方法论**（P_dma/P_vec/P_full 三探针，与主 kernel 同构 block/DMA/缓冲）：
   一轮实验即把"DMA/向量链/reduce"三成分定量化（9.1%/68.2%/22.7%），推翻/证实跨轮假设
   （iter2 的 issue-bound 归因被证实，iter1 的"sync ~90%"估算再次确认只对基线版结构成立）。
   tile op 的命令式 buffer 写语义保证删除 reduce 调用不会 DCE 上游向量链（探针差分有效）。
3. **因式分解的溢出域安全判据**（mid-anchor）：j ≤ i（下三角）保证 E+ 大值（i < a 侧）与
   E- 大值（j > a 侧）永不同现于同一乘积；唯一 NaN 路径 inf×0 需要 half-span 衰减同时
   > 127（上溢）与 > 149（下溢）——数学上不可达（127 < 149）。安全域收敛为单一条件：
   subchunk 半程衰减 < ~127。DBOUND 用例（-12/token，half-span 96）余量 31。
4. **in-place mul 链的缓冲编排**（UB 零新增实现因式分解）：e 缓冲三段复用（E- → E+ → K''，
   每段被下一步消费后覆写）、q_f slab 覆写为 Q'（diag 先消费原始行）——依赖 AUTO_SYNC=True
   的 V pipe 串行 in-place 安全语义（optimization-guide §2.13 V pipe buffer 分离）。

# Iteration 4（2026-10-23）

## reduce_2d 重组专项分析（Step 3，iter3 next_hint 首选杠杆）

本轮基线：6923.0 us（baseline_iter4.json，per-rep 方差 <0.2%；探针会话内 V_cur 复测 6907.3
一致）。时间结构沿用 decomposition_iter3：向量链 4033us（58%）+ reduce 2056us（30%）+
DMA/setup 828us（12%）。

### Part A 优化点清单

[#8] reduce_2d 窗口合并/拼接（next_hint (a)）：**不适用（结构性墙，分析级否决）** —
  reduce_2d v2 的 dst 必须是 M 个连续元素（codegen ReduceOpCodegen 只传 dst 基址偏移，
  helper 以 packed 布局写 M 输出）；Aqk/Akk 输出为下三角（row i 用 [w, w+i+1) / [0, i)），
  任意跨 i 合并都要求输出连续，而下三角在矩形 staging 内不可能连续。零填充成方形
  （[256,128] 单调用/侧）则 step0 按行收费（StepCost4 8m 项）：浪费行 136+120 →
  cost4 8384 vs 当前 5834（+44%）。packed→方形 unpack 被 32B MTE 粒度（<8 元素行 <32B）
  与无向量 gather 原语封锁。
[#8b] reduce_2d plan 研究（next_hint (b)）：**完成，证实 plan 已最优** — planner 纯
  constexpr 可 host 求值（reduce2d_plan_dump.cxx 离线枚举）：M=2..16/N=128/Clear=true/
  ARZ=false 统一 3 步 plan（128→16→2→0），cost4 ≈ 64 + 13·M（M%8≠0 另加 bump）。
  step0 每行一条 vcgadd 是寻址限制而非 planner 失误：repeat-stride 只能二选一行走"128 宽
  行内 even/odd 块交错"或"行距"，故 128 宽行无法 batch；≤64 宽行（列切片，Scan 路径）
  可单指令 repeat=M batch 全部行——cost4 模型 8/row（per-row body）vs 4/row（batched
  scan）的定价差即源于此。
[#9] 手动 vcgadd 第一步压缩（next_hint (c)）：**不适用** — DSL 无裸 vcgadd 部分归约
  原语（narrow fallback 要求 n≤64 且 clear=True，fp32 合法 plan 时升级回 v2 helper）；
  flat [M*16, 8] 视图 plan cost4=944（M=16，劣化 3×）；3D [M,16,8] 路径前端取
  M=extent[1]，对 A>1 只写 M 输出，语义不成立。
[#10] split-merge 列切片（(b) 衍生最优候选）：**实测证伪** — [M,64]@PR=128 列切片×2
  （clear=True + clear=False merge，无 combine op；仓内测试先例
  test_fp32_sliced_max_supports_merge_and_wide_rows）。cost4 模型 aqk 侧 3744 vs 2939
  （+27%），但 emitter 指令计数 ~95 vs ~190（-50%）——矛盾交由差分探针裁决：
  **7982.0 vs 6907.3 us = +15.6%，证伪**（数值正确 allclose 1e-2 PASS，clear=False
  merge 机制本身可用）。per-call 固定开销（~30ns × 调用数 31→62/h）支配 batch 化
  指令节省。
[#11] β 向量化（次要 hint）：**不适用** — per-row 标量广播源 [16,1] 4B 行宽违反 VEC 源
  32B 对齐（iter1 aicore 507015 同墙）；per-h β 列抽取（beta_f [16,16] 列 stride 64B）
  无 gather 原语、MTE 4B 粒度不可表达；标量 Muls（1 op）替换为 fill+broadcast+mul
  （3 op）在实测 ~58ns/tile-op 档位净负（+120 op/item ≈ +7us/item vs 屏障节省 ~0.3us）。
[#12] iter2 desync 叠加 iter3 基础：**预期 <3%，不实施** — iter2 实测屏障 ~1.5ns/个
  （完全流水化）；iter3 基础可移除 auto 屏障 ~588/item ≈ -2.6%，低于采纳阈值，且需对
  因式分解 in-place 链重做手动 flag 分析（高风险低回报）。

### Part B [ORDER-PLAN] 实施顺序

1. [#10] split-merge — 前置依赖: 无 — 理由: 模型（+27%）与 emitter 指令计数（-50%）
   结论相反，唯一需实测裁决的候选；按 iter3 "先剖析再优化" 教训走探针先行。
→ 探针证伪后本轮无 kernel 实施项（[#8][#8b][#9][#11][#12] 分析级否决）。

## 实施记录

[ORDER-CHECK] 准备实施: [#10] split-merge 差分探针 | 前置依赖: 无 | 结论: ✅（探针级，
  kernel 零改动）

[IMPL-#10] 已阅读 optimization-guide.md §2.13（L1024-1131 归约模式铁律：只用 dim=-1、
  循环外归约）、performance-antipatterns.md（多行 tile 逐块归约条目）、
  vector-practices/vector_reduce_pass_fusion.md（适用条件 C1-C4/X1-X3：本算子归约维
  128 可一次入 UB、单遍扫描 → 归约遍数融合不适用）、reduce_2d_v2.h 全文（planner/
  emitter/M1 叶）、reduce_2d_vector_delay.h（ChooseWait：ARZ=false → 全部
  VectorBarrier）、reduce_2d_m1.h（M1 叶路径）、codegen_ascend.cc L1740-1890
  （ReduceOpCodegen：v2 路径 physical_row/平台 ARZ 判定、narrow fallback 约束）、
  ascend_vector_instruction_selection.cc（kReduceNarrow→kReduceFp32V2 升级路径）、
  reduce_ascend.py（前端 reduce：region/real_shape/physical_row 附加条件）、
  test_tilelang_ascend_language_narrow_reduce.py + reduce2d_v2.py（列切片 + clear=False
  merge 语义仓内先例）。关键约束: 切片基址 32B 对齐（64 fp32 = 256B ✓）；
  physical_row=128 附加要求全 extent 常量（i 字面量展开保证）；M=1 保留 M1 叶；
  clear=False 不可走 narrow fallback（fp32 v2 存在合法 merge plan）。

[SELF-CHECK] 本次只新增 perf_tuning 探针文件（splitreduce_iter4.py / reduce_plan2_iter4.cxx），
  kernel 与 test 零改动（md5 9168a76b 与 impl_iter4_before.py 一致）。

[RESULT-#10] 优化点: split-merge reduce（[M,64]×2 clear/merge）| 精度: 探针级 allclose
  (rtol=atol=1e-2) PASS | 性能: 7982.0 us | 对比: **+15.6% vs 6907.3 —— 证伪不采纳**
  （kernel 未动，无需回滚）

### 结果文件

- 本轮基线：fla_chunk_intra_token_parallel_baseline_iter4.json（6923.0 us）
- planner 离线 dump：reduce2d_plan_dump.cxx + fla_chunk_intra_token_parallel_reduce_plan2_iter4.cxx
  （当前/两阶段/切片/merge 各 shape 的 plan、cost4、scratch）
- 差分探针：fla_chunk_intra_token_parallel_splitreduce_iter4.py / .json（+15.6%）/
  logs_iter4/splitreduce_probe.log
- 精度复验（零改动状态）：logs_iter4/precision_standng.log（exit 0，13 PRECISION_PASS +
  14 BOUNDARY_PASS，ratio=1.0000 全绿）

### 本轮关键技术发现（供 skill journal 沉淀）

1. **reduce2d_v2 planner 可离线求值**：planner 为纯 constexpr（非 __CCE__ 部分），host
   g++ 直接 include 头文件即可枚举任意 (M, N, SrcRowStride, Clear, ARZ) 的 plan/cost4/
   scratch——优化决策无需先写 NPU 探针（本轮 (a)(c) 两方向由 cost4 模型直接否决，仅
   模型与 emitter 计数矛盾的 split-merge 需要实测裁决，且模型方向再次被证实）。
2. **128 宽行的 per-row step0 是 vcgadd 寻址限制**：repeat-stride 无法同时行走行内
   even/odd 块交错与行距 → 128 宽行每行一条指令；≤64 宽行单指令 batch 全部行。此为
   cost4 模型 8/row（per-row body）vs 4/row（batched scan）定价差的物理来源。
3. **per-call 固定开销支配小窗口 reduce**：实测 split（调用数×2、指令数约减半）+15.6%，
   与 cost4 模型方向一致——reduce_2d 的调用数本身是主要成本项之一，但下三角输出的
   dst 连续性要求（codegen 只传基址 + helper packed 写）使跨窗口合并在算子侧不可表达。

# Iteration 5（2026-10-23）

## desync 叠加实测裁决（iter4 [#12] 的实施，最后一个可自主候选）

本轮基线：6923.7 us（baseline_iter5.json，per-rep 方差 <0.2%，与 iter3/iter4 记录一致）。
目标：iter2 desync 方案（当时在 iter1 结构上实测 -2.3% 回滚）重新实施到 iter3 因式分解
结构，实测叠加效果，完成"极尽全力"证据链（模型预期 ~-2.6%，但模型已被 iter2/iter4 两次
证伪，必须实测）。

### Part A 优化点清单（Step 3）

[#12] **AUTO_SYNC=False + 手动 flag 对 + β PrimExpr 预读（iter2 方案叠加 iter3 结构）**
（参考: optimization-guide.md §2.2.0 决策表、§2.2.1 V pipe 串行、§2.13 in-place 警告冲突
裁定；api-schedule-sync.md §4；performance-antipatterns.md "冗余全局同步"——本轮唯一命中
条目，其余 10 条已在前四轮解决/否决/不适用）：**适用（本轮唯一候选）** —
iter3 结构残余 ~131 PipeBarrier_V/h + 60 β 标量 PIPE_ALL/block。同步点按 iter3 缓冲/调度
结构重新推导（非照搬 iter2）：
  - v→mte2（fill beta_h → β 载入，WAR）/ mte2→v（全部 GM 载入 → 首个 V 消费，RAW）/
    v→s（β_f cast copy → 标量预读，RAW）/ v→mte3（bf16 cast → 写回，RAW），eventId 0；
  - copy_ub_to_ub 跨 dtype = AscendC::Cast（V pipe，common.h L360-377 源码实证）→
    β_f 写者在 V pipe，单对 V_S 覆盖全部标量读；
  - β 预读局部 `b{i} = beta_f[i, h] * 1.0`（2D 索引 OffsetOf = i*16+h；`* 1.0` 强制
    PrimExpr/CoerceScalarArg 路径；按名传宏——内联表达式会被宏机制折叠回 raw BufferLoad
    → PIPE_ALL 回归，iter2 三态 lowering 发现）；
  - V pipe in-place 链（e: E-→E+→K''、q_f→Q'）依赖 §2.2.1 串行语义（§2.13 冲突按
    iter2 证据权重 + opt_gdn 先例裁定；精度门禁兜底）。

[KERNEL-REUSE-PRECHECK]
candidate: [#12] desync 叠加（同步配置 + kernel body flag/预读改写）
expanded_domain: 无新增调用域（host 接口/shape/dtype 契约不变；SCS/BH 自适应路径不变）
compatibility_dimensions: AUTO_SYNC=False（Developer 模式合法配置，opt_gdn/mha_sink/moe
  先例）；T.set_flag/wait_flag 前端签名（ascend.py L233-272，_pipe 含 "s"）；V_S 事件在
  GetEventMapping 表内（operation_config.h L401-402）；标量 PrimExpr src1（ascend_tile.py
  binary_op 分派，iter2 探针已验证 lowering 形态）
unknown_or_false: none（iter2 已在全同机制上探针+全量验证，本轮仅结构位置适配）
plan: direct-reuse（kernel 内部改写，host 不变）

## 实施记录

[ORDER-CHECK] 准备实施: [#12] desync 叠加 | 前置依赖: 无 | 结论: ✅

[IMPL-#12] 已阅读 optimization-guide.md §2.2.0（L106-125 决策表）/§2.2.1（L279-291 V pipe
单缓冲安全）/§2.13（L1117-1131 in-place 与 dst 交替警告）、api-schedule-sync.md §4（Set/Wait
配对铁律，L151-179）、performance-antipatterns.md 全清单（冗余全局同步条目 L157-199 含
"关闭自动同步必须生成代码 + 精度双确认"检查点）、iter2 生成代码样本
（kernel_source_iter2.c 的 4 flag 对 + hoisted GetValue 形态）、common.h copy_ub_to_ub
（L359-377，跨 dtype 走 AscendC::Cast = V pipe）。关键约束: 每 set 必须有配对 wait（同
block 内即时配对，无残留 token）；β 预读必须 2D 索引 + `* 1.0` 包裹 + 按名传宏；预读须在
parse 期 guard 内（SCS<16 域保护）；iter3 的 fill 位置（cast 后）与 iter2（β 载入前）
不同——v→mte2 对只耦合 beta_h 不受影响。

[SELF-CHECK] 本次 Edit 只涉及 [#12]（PASS_CONFIGS AUTO_SYNC + 4 对 flag + β 预读局部 +
_rows_factored 宏签名；host 包装/test 不变；diff 纯同步相关，热循环计算链零变化）

[RESULT-#12] 优化点: desync 叠加（生成代码 PipeBarrier 131/h + 60/block → 0）| 精度:
pass（--level all：13 PRECISION_PASS + 14 BOUNDARY_PASS，ratio=1.0000 全绿）| 性能:
6742.8 us | 对比: **-2.61% vs 基线 6923.7 us —— 低于 3% 阈值，已回滚**
（回滚后复测：精度 13 PASS + bench 6915.3 us 确认复原）

### 结果文件

- 基线：fla_chunk_intra_token_parallel_baseline_iter5.json（6923.7 us）
- 候选：fla_chunk_intra_token_parallel_candidate_iter5.json（6742.8 us，-2.61%）
- 回滚确认：fla_chunk_intra_token_parallel_rollback_confirm_iter5.json（6915.3 us）
- 本轮备份（回滚基线）：fla_chunk_intra_token_parallel_impl_iter5_before.py（md5 9168a76b）
- desync 候选保留：fla_chunk_intra_token_parallel_impl_iter5_desync_candidate.py
- 生成代码样本：fla_chunk_intra_token_parallel_kernel_source_iter5.c（零 PipeBarrier）
- 精度日志：logs_iter5/precision_all.log（desync 版）+ precision_rollback_confirm.log
- bench 日志：logs_iter5/baseline.log / candidate_bench.log / rollback_bench.log

### 本轮关键技术发现（供 skill journal 沉淀）

1. **desync 收益与计算结构无关、与屏障绝对数近似线性**：iter1 结构 -2.3%（~670 barrier/
   block）、iter3 结构 -2.61%（~588/item）——两次独立实测都在 -2.3%~-2.6% 区间，印证
   屏障 ~1.5ns/个被流水掩盖的定性结论；"零 PipeBarrier"的技术成功不等于 wall-clock 收益，
   纯 Vector 算子同步开销天花板可据此预估。
2. **copy_ub_to_ub 跨 dtype 的管线归属**：common.h 实现为 AscendC::Cast（V pipe）而非
   MTE——AUTO_SYNC=False 手动同步分析中 UB 内跨 dtype copy 的生产者/消费者都按 V pipe
   处理，无需 MTE1 相关 flag 对（iter2 结构隐式正确，本轮源码级显式确认）。
3. **iter2 三态 lowering 发现的跨结构可移植性**：β 标量 PrimExpr 预读（2D 索引 + `* 1.0`
   包裹 + 按名传宏）在 iter3 宏结构上原样可用；守卫内提升的预读局部对 SCS<16 域同样成立
   （l1_scs8 通过）。

## Stage 3 收敛（consecutive_no_improvement = 2）

iter4（零可实施候选）+ iter5（-2.61% < 3% 阈值回滚）→ Stage 3 正式收敛。最终性能
~6915-6924 us（vs 19150.1 us 基线 -63.9% / 2.77×；带宽利用率 7.1%）。三成分结构下限
证据链与剩余不可自主选项（GEMM 卸载 / A3 ARZ / helper 本体优化）见 perf_log.md
Iteration 5 收敛判定。
