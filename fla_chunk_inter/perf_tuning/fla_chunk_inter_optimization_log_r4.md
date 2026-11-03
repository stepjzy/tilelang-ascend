# fla_chunk_inter 性能优化日志 — R4（收敛战役，2026-11-03 起）

R4 目标：从交付修复后基态 7510.3us（3 次独立 do_bench：7508.8/7511.6/7510.5，D-2 G5 验收）
收敛回 ~6767us 量级（交付修复前水平；修复代价 +743us）。

会话锚点（重调度会话前、同一验证基态 kernel）：7510.6/7494.2/7506.3us，均值 7503.7us
（r4_session_baseline.log，2026-11-03T19:01）。

终态画像（msprof_output_r4_current，11/3 19:02，7323.6us task duration）：
- V 侧（关键路径）：vec 3254us(44.4%) + mte2 2888us(39.4%) + mte3 1433us(19.6%)；
  aiv_scalar 71% busy 且 stall_by_ub 7008us(96%) → 语句流/UB 等待主导。
- C 侧：cube 仅 9.6%（702us），mte1 12.6%，scalar_wait 3300-3600us → 大量余量。
- 结论：减 V 侧语句数与向量遍数、把可移的规约/广播类工作移到 C 侧。

## Part A 优化点清单（Step 3）

[#1] dgkn ones-gemm 对角化移 C 侧（produce 侧 V 链整体删除）
（参考: performance-antipatterns.md L127「Vector Core 内逐元素/逐行 for loop 计算」；
vector-practices/vector_reduce_pass_fusion.md 归约遍数融合思想；本次新增源码核查
tilelang/language/ascend_tile.py gather/reinterpretcast/arith_progression + 
src/transform/allocate_tmp_buffer.cc:758-762 gather 在 AscendC 目标 NoWorkspace）
[适用] — 归因 -539us（基线 json regression_attribution）。dgkn[k]=Σ_v h[k,v]·dh[k,v]
= diag(h@dh^T)；C 侧 L1 已有 h_hi/lo、dh_hi/lo（dk 组操作数）→ 3 个 hi/lo 小 gemm
→ [bDK,bDK] fp32 frag（L0C 4 frags = 64KB < 128KB ✓）→ T.copy frag→dgm_relay GM
（与 dq_relay/dk_relay 同模式）→ drain 侧 mte2 回读 + T.tile.gather 对角提取
（offset buffer 由 prologue arith_progression(int32)+reinterpretcast(uint32) 一次构建，
cann-bench/gather 生产同款）。删除 V 侧 mul+transpose+chunk-adds+6 层树+行拷贝整链
（含 32KB prodT_ub）。**红线核查**：gather/reinterpretcast/arith_progression 均不在
allocate_tmp_buffer.cc 的 AscendC requires_workspace 集合（gather 显式 NoWorkspace），
无 planner 注入 scratch → D-2 红线安全。精度：dgkn 误差类从 fp32 元素链变为 hi/lo
gemm 类（~2^-22 相对，与 dq/dk 同类）；门禁 = valrange_l dg max_abs ≤ 1e-2 实测。

[#2] gn 门控 64 逐行 Sub → 转置 + row_expand_sub_experiment（bDK=64 路径）
（参考: performance-antipatterns.md L429-454「正交轴串行化」— 文档给出的正解就是
transpose 向量化；ascend_tile.py:2983 row_expand_sub_experiment = brcb + 一次
masked Sub，FA expert_v3 生产先例）
[适用] — 归因 -565us。gnd[t,:]=gn[:]-g[t,:] 的 64 条 Sub 语句（每条 [bDK] 256B，
语句开销主导）替换为：transpose(g_ub→gT) + row_expand_sub(gT,gn,tmp)（brcb 展开
gn 为逐行标量块）+ mul(-1)（保号：得到 gn-g；exp 参数与现实现逐位一致）+
transpose(gndT→gnd)。约束核查：dst/src0 需 256B 行（f32 64 列）→ 仅 bDK=64 路径
可用；bDK=32（DV<128 或 DK%64≠0）保留旧 64-Sub 路径（trace-time 分支，L1 套件
覆盖）。tmp 为显式 alloc_shared（[bDK,8] f32）非注入。gT 复用已死的 gexp_ub 缓冲
（dq 乘完后死亡），UB 净增仅 tmp 2KB。预期收益打折：新增 2 次 [64,64] 转置
（produce 链归因推算单次 ~1.6us）→ 净省估计 -170~-300us，实测裁决。

[#3] 折叠树指令精简（~170us 边际）
[暂缓] — 候选①已删除 produce 侧整树；drain 侧 dkk 树为 6 条语句（每层单次
masked Add），无可替代原语（wholereduce 已 deprecated 且语义不符；inner 维折叠
被 binary_op 区域连续性约束拒绝）。若①②采纳后仍差 >3%，再评估。

[#4] 画像驱动补查
[部分适用] — aiv_scalar 71%/stall_by_ub 96% 指向语句数削减（①②直接作用）；
C 侧 wait 3.3-3.6ms 余量足够吸收①新增的 3 gemm/(task,bk)（cube 702→~935us、
mte1/fixpipe +33%，仍远低于 7.3ms 墙）。无其他独立大项。

## [ORDER-PLAN] 实施顺序

1. [#1] dgkn ones-gemm 对角化 — 前置依赖: 无 — 理由: 归因最大、与②独立、
   先行验证 gather/relay 通路与精度门禁。
2. [#2] gn 门控 row_expand 化 — 前置依赖: [#1] — 理由: 同处 drain 代码域，
   ① 采纳后再叠加，避免归因混淆；② 的 gT 复用（gexp_ub）不受①影响
   （①不动 g/gexp 生命期）。
3. [#3]/[#4] — 前置依赖: [#1][#2] — 理由: 边际项，按①②后的残余差距决定。

迭代编号说明：r4_iter1_before.py 为中断会话产物（对应未完成候选①脚手架），
本轮从 iter2 重新计数（沙箱已恢复 = D-2 验证基态）。

## Iteration 记录

（以下按 6 子步骤逐步追加）

### Iteration 2 — [#1] dgkn ones-gemm 对角化移 C 侧

[IMPL-#1] 已阅读 ascend_tile.py L1848-1906（gather：offset=byte 偏移 uint32 buffer，
count=min(dst,off)，无 tmp 参数时 AscendC 不注入 scratch）/ L398-431（arith_progression：
int32 支持 ascendc）/ ascend.py L618（reinterpretcast）/ allocate_tmp_buffer.cc L758-762
（ascend_gather 在 AscendC 目标 NoWorkspace）/ cann-bench/gather.py L62-80（offset 构建生产
范式），关键约束: gather 的 src/dst 同 dtype、offset 为 uint32 byte 偏移；dgm [bDK,bDK]
frag→GM relay 与 dq_frag→dq_relay 同模式；4 frags 64KB < L0C 128KB。

[KERNEL-REUSE-AUDIT] 依据: performance-antipatterns.md L127 + 硬件容量表 L30-63 |
维度及结论: L0C 64KB<128KB ✓ / UB 净减（删 prodT_ub 32KB + dgkn_ring，加 dgm_ub 16KB
+ 小 buffer）✓ / L2 relay +2.6MB（~27MB 总量，192MB 容量）✓ / flag 组复用 40+j%2
（dgm_relay 写在 C 组 flag set 之前、读在 drain wait 之后）✓ / 无新 scratch 注入
（gather NoWorkspace；arith_progression/reinterpretcast 不在 requires_workspace 集）✓
| 结论: reuse

[SELF-CHECK] 本次 Edit 只涉及 [#1]（含配套 docstring 与索引移位，无其他行为变化）

[RESULT-#1] 优化点: dgkn ones-gemm 对角化移 C 侧 | 精度: L0+L1 全绿 40 PASS，
valrange_l dg 7.812e-03 ≤ 帽 1e-2（与基态 7.8e-3 持平，hi/lo 误差贡献不显著），
--level all Test Passed（bd_dbound WARN 为设计预期非阻塞，ratio 0.9999 同基态类）
| 性能: 7159.8/7187.1/7154.5 us，均值 7167.1（do_bench warmup=20/rep=10/mean/
仅设备侧，3 次独立）| 对比: vs 会话锚点 7503.7（7510.6/7494.2/7506.3）= -336.6us
即 -4.49%，超 3% 噪声阈 → 采纳（实现归因的 62%，差额为新增 relay 回读 mte2 +
drain gather + C 侧 3 gemm 的真实开销）

### Iteration 3 — [#2] gn 门控 64 逐行 Sub → 转置 + row_expand_sub 向量化（bDK=64 路径）

[ORDER-CHECK] 准备实施: [#2] | 前置依赖: [#1 ✅] | 结论: ✅

[IMPL-#2] 已阅读 ascend_tile.py L2762-3007（row_expand_sub_experiment：dst/src0 需
256B 行宽（f32 64 列）、行数 8..248 且 %8==0、物理行 stride 匹配且 1..255 block、
src1 标量线性 [R]/[1,R]/[R,1]、显式 tmp=[rows*8] 元素、dtype 限 fp16/fp32）+
codegen_ascend.cc L2064-2153（brcb 展开 + PipeBarrier + 单次 {op}_mask 终端）+
testing test_tilelang_ascend_language_elementwise.py L5736-5760（tmp 路径 UT）+
FA expert_v2/v3 生产用法（L209-225 无 tmp packed 路径）。关键约束: bDK=32 行宽
128B 不满足 256B 行窗 → 保留 legacy 64-Sub 路径（trace-time 分支）。

[KERNEL-REUSE-AUDIT] 依据: performance-antipatterns.md L429-454（正交轴串行化的
转置向量化正解）| 维度及结论: gT 复用已死 gexp_ub（dq 乘后死亡，线性生命期扩展）✓ /
tmp 显式 alloc（非注入）✓ / 负号折叠保留 exp 参数逐位一致（mul(-1) 在 exp 前）✓ /
转置 32B 对齐（64×4B=256B）✓ / 无新 scratch 注入（row_expand 不在
requires_workspace 集）✓ | 结论: reuse

[SELF-CHECK] 本次 Edit 只涉及 [#2]（bDK=64 门控路径 + gexp_tmp 声明；legacy 路径
与 ① 的代码未动）

[RESULT-#2] 优化点: gn 门控向量化 | 精度: L0 全绿（l0_repr bDK=64 新路径 dg
1.221e-03 与 iter2 完全一致 → 逐位等价实证；l0_min bDK=32 legacy 路径不变），
--level all 40 PASS Test Passed（bd_dbound 预期 WARN）| 性能: 5 次独立 do_bench
7011.7/7003.5/7026.9/7018.8/7000.5 us，均值 7012.3 | 对比: vs iter2 基线 7167.1
（7159.8/7187.1/7154.5，同会话）= -154.8us 即 -2.16%。
**采纳判定说明**：幅度 < 3% 默认噪声阈，但本 bench 口径实测噪声（独立均值散布）
仅 ≤0.5%（候选 5 次散布 0.37%，基线 3 次散布 0.45%），两组区间完全不交叠
（候选 max 7026.9 < 基线 min 7154.5，差 127.6us，~14σ 分离）→ 按 mandate
"无硬性数字门禁，按证据推进"条款采纳；此偏差在最终报告中显式标注。

### Iteration 4 — [#4-画像归因] drain 三中继读融合（dq|dk|dgm → 单次回读）

[ORDER-CHECK] 准备实施: [#4] | 前置依赖: [#1 ✅ #2 ✅] | 结论: ✅

[IMPL-#4] 已阅读 iter3 画像（msprof_output_r4_iter3：V 三管线 busy 6955 ≈ 墙 6823；
mte2 3219us@~53GB/s 对 416KB/task = **带宽界**，vec 2279 语句/延迟主导，mte3 1457@76GB/s；
aiv_scalar 67% 且 vector_stall 5092 → 语句数是 vec 侧次级杠杆）+ ③ 树折叠替代实证封堵
（wholereduce 每 repeat 仅 8-block lane 求和不跨 repeat 累加，64 行折叠不可能；
sum_experiment 为 last-dim 归约轴不符 + 转置成本反超树）。关键约束: 融合不改字节量，
收益来自 3 次独立读 → 1 次的发射/屏障开销削减（每任务 -4 条 mte2 语句 + 名字级同步）。

[KERNEL-REUSE-AUDIT] 依据: iter3 画像 + binary_op 区域连续性约束 | 维度及结论:
融合中继区域视图（[0:BS]/[BS:2BS]/[2BS:2BS+bDK] 全行区域 = flat-contiguous）✓ /
gather src 支持 BufferRegion ✓ / C 侧 frag→GM 深区域拷贝（行偏移 2BS·bDK·4B=32B 对齐）✓ /
签名 25→23 张量（3 中继→1），out_idx=[19..22] workspace_idx=[8..18] ✓ / 手动 flag 组
不变（C 在 flag set 前写完全部三段）✓ | 结论: reuse

[SELF-CHECK] 本次 Edit 只涉及 [#4]（融合中继 + drain 消费区域化 + 索引移位；
中途一次编辑损坏已从 iter4_before 快照恢复重做，恢复后 md5 校验一致）

[RESULT-#4] 优化点: drain 中继读融合 | 精度: L0 全绿（误差数字与 iter3 逐位一致），
--level all 40 PASS Test Passed（bd_dbound 预期 WARN），valrange_l dg 7.812e-03 ≤ 1e-2
| 性能: 5 次独立 do_bench 6871.9/6859.0/6855.8/6881.8/6873.0 us，均值 6868.3
| 对比: vs iter3 基线 7012.3（5 次，散布 0.37%）= -144.0us 即 -2.05%（区间不交叠：
候选 [6855.8,6881.8] vs 基线 [7000.5,7026.9]，~10σ；同 iter3 判定法按证据采纳，偏差显式记录）
| **vs 目标 6767：+1.50%，进入 ±3% 收敛带 [6564, 6970] → R4 收敛条件①达成**

### R4 候选空间终态归因（画像+实证）

- ① dgkn ones-gemm 对角化：已采纳（-336.6us）。剩余 vs 旧 reduce_sum 形态 +~203us =
  dgm 中继回读的 mte2 带宽成本（16KB/(task,bk) ≈ 291us mte2 时间）——**结构封堵**：
  gather 仅读 UB、fixpipe 仅连续写、gemm 无法紧凑产出对角（三条独立死路已证），
  D-2 红线禁 reduce_sum。
- ② gn 门控向量化：已采纳（-154.8us）。剩余 vs 旧 broadcast 形态 +~410us = 2 次转置
  (~1.6us/次·[64,64])——**结构封堵**：broadcast 触发 scratch 注入为红线；转置是
  无 scratch 行广播的最小形态。
- ③ 树折叠精简：**实证封堵**（wholereduce 语义不可能 64 路折叠；sum_experiment 轴不符；
  转置反超）。drain 树 6 语句保留。
- ④ 中继读融合：已采纳（-144.0us）。mte2 字节地板 = 输入加载+中继回读（h/dh 128KB、
  do/v/dv 96KB、G/Q/K 64KB、dq/dk/dgm 48KB / task，53GB/s 引擎界）——全部为结构必需。
- 终态 6868.3us vs 6767（R2 档案）= +1.50%（带内）；vs R3 会话地板带 6639-6720
  差异 ~2.2-3.4%（含会话间方差 ±1.2%，R3 同代 kernel 实测 6639-6720）。
