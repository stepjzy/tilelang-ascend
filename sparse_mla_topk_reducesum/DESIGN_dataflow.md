# DESIGN_dataflow.md — P3 序列级去重驻留（distinct-row panel residency）数据流重构

> **route**: P3（通用算子数据流重构——AIV gather → GM ws_kv → AIC gemm 三腿 302MB×2 消除路线）
> **baseline**: git `22bc68a8`（P2 交付态，官方 BI128 = 363.1µs，全量回归 70 PASS，带宽地板 ~333µs）
> **target**: 官方 BI128 < **220µs**，**完全通用**（任意合法索引正确；无契约 / 无内容假设 / 优雅降级）
> **mode**: first_design（新路线筛选 + 方案定型，直接进入实现）
> 参照锚点：git `0f11b3cc`（Plan-A panel 机制：L1 驻留 staging / 8-flag / 组调度 / 143.3µs 实测）；
> perf_tuning/{joint_bottleneck_analysis, p2_final_bench, golden_roofline, panel_metric2_bench}.md

---

## 0. 结论速览

| 项 | 结论 |
|---|---|
| **选定方案** | **P3 序列级去重驻留**：每 (seq, kv_group) 构造"被引行并集 panel"（M=topk 定宽），panel 经 ws_panel 单次 GM 中转驻留每核 L1；C 侧**转置 gemm**（panel×Qᵀ → accT [topk, H]）；V 侧 **pos 计数映射 + gatherb 块选列**；大 P_seq / 非共享输入经**设备侧资格判定 + 推测双发射**回退现通用路径 |
| **ws 流量** | 302MB×2（ws_kv 双腿）→ **~12MB**（panel staging 20 核×2 组×147KB×2 方向）；全 kernel 流量 995MB → **~110MB**（9×） |
| **官方预期带** | **175–200µs**（点估计 ~185µs = 143.3 panel 锚 + prep 10–15 + 双发射 8–12 + 选列 10–20）；证伪线 220µs（§6.2 预注册） |
| **正确性域** | panel kernel 正确域 = {P_seq ≤ topk}（**维度性质，非内容契约**）；调度系统（panel + 回退）对**任意合法输入**正确（重复/乱序/-1/越界/空序列/G>1/RH>1） |
| **与被否决 Panel 的区别** | 无索引共享契约、无内容假设、kernel 内 pos 按行独立计算（非 j↔列恒等直映射）、违约→性能回退而非正确性风险、API 零变化（§5.9 强制章节逐条论证） |
| **硬件新查证** | L2 cache/persistence hint、multicast DMA、MTE burst 修饰符：**仓内无 API，标"未证实，不得依赖"**；新增可用原语：`T.tile.gatherb`（UB 32B 块索引 gather，仓内已测）、`T.tile.sort/merge_sort`、`T.tile.atomic_add`（仅连续区域，非散射）、`gather_mask` 定式抽取（§3） |

---

## 1. 基线流量账本与任务定义

**算子与计算语义（不变式——本重构纯改数据流，语义/接口/精度域零变化）**：
`sparse_mla_topk_reducesum`——varlen packed 布局下，对每 token 行 s、每组 g、每 topk 位置
idx = Indices[s,g,j]：`ReduceSum[s,g,r,j] = Σ_h exp((Q[s,h,:]·KV_main[bos+idx] + Q_tail·KV_tail)·sm_scale
− Lse[s,h]) · 1(idx ≤ s_i ∧ idx ≥ 0)`（causal 无效位贡献 0；-1/越界经 clamp+mask 安全丢弃；
REPLICATE_H 分片沿 h 维展开）。适用场景：DSA 稀疏注意力 topk 概率质量的 head 归约（MLA decode，
bf16 Q/KV + fp32 累加输出）。语义细节以 proto.yaml formula 与 DESIGN_generic_p1.md §1 为准。

### 1.1 官方口径数据形态（唯一性能口径，逐字锁定）

`test_sparse_mla_topk_reducesum.py::test_kernel`（L434-446）逐字：seed=42、
q [2048, 16, 576] bf16、kv [2048, 1, 576] bf16（ΣS_kv = ΣS = 2048）、
offsets=[0, 1023, 2048]（2 个 varlen 序列：1023 + 1025）、
**topk_indices = arange(128) 逐行重复 2048 次**（全行共享同一索引向量）、topk=128、BI=128。

⇒ 每 (seq, g) 的被引行并集 = **128 行**（P_seq = 128 = topk），全批 distinct KV 行 = 256 行（147KB×2）。

### 1.2 基线（generic 路径）带宽账本（joint_bottleneck_analysis §1.1，引用）

| 流量腿 | 字节 | 路径 |
|---|---|---|
| V 行读（gather 源，2048 块 × 128 行 × 1152B） | 302MB | L2 命中（arange 全热） |
| V bulk 写（ws_kv 中转） | 302MB | L2 脏行写 |
| C ws 读回（ws_kv → L1） | 302MB | L2 脏行读 |
| Q 读 + ws_acc×2 + 输出 + idx/lse | ~95MB | 混合 |
| **合计** | **~995MB** | 聚合服务率上限 ~2.96TB/s ⇒ 带宽地板 ~333µs |

基线 363.1µs 已贴地板（联合分析结论：时序/调度侧仅剩 0-5%，P2 已收 1.5%，P2b +3.23% 确证回退）。
**220µs 需流量 ≤651MB（-35%）或聚合 4.52TB/s（1.52× 观测上限）——唯一现实出路 = 流量消除。**

---

## 2. 用户五问逐答（全部带证据）

### Q1：ws_kv 为何 302MB×2？——按槽位中转的机制成因

**答案：W1（写腿）= 262,144 个行拷贝槽位 × 1152B/行 = 302MB，其中 262,144 = ΣS × topk = 2048 × 128——通用路径对每个（查询行 s × topk 槽位 j）独立搬运一整行 KV（1152B = 576 bf16），跨行/跨槽位零去重；读腿对称 302MB。机制成因是三级结构叠加：**

1. **引擎分工强制**（AIV gather / AIC gemm 分离）：索引化行读只能在 AIV（MTE2 标量基址行拷贝，`sparse_mla_topk_reducesum.py` L350-353 `T.copy(KV[kv_row, g, :], ...)` 模式），gemm 只能在 AIC（L0A/L0B 分形操作数）。
2. **内存层级强制**：UB↔L1 无直拷路径（api-kernel-memory.md §1 workspace 机制原文："由于 Ascend 硬件限制，UB 和 L1 不能直接互通，必须通过 Global Memory 中转"；§3 搬运路径表无 UB↔L1 行）⇒ AIV 采集的数据必须 GM ws_kv round-trip 才能到 AIC。
3. **槽位粒度设计**（本设计要消除的冗余）：中转单位 = 块（BI×DDT=147KB/块，`ws_kv: [grid_size, DEPTH*BI, DDT]` L215），块内容 = 该块 topk 个索引的行拼装——**逐块独立、无跨块去重**。同一 KV 行被 20 个核、每核多个块重复搬运。

**证据**：kernel 源码 L215/L284/L589-594（ws_kv 布局与双腿拷贝）；joint_bottleneck_analysis §1.1 表（三腿 302MB 逐腿列出）；api-kernel-memory.md §1/§3（层级强制）。

### Q2：强制 vs 冗余分解——"每（去重行×消费核）一次"是架构强制，"按槽位"是设计冗余（官方 1024×）

**逐条流分解（每 kernel 调用）：**

| 分量 | 强制性 | 论证 |
|---|---|---|
| AIV→AIC 数据必须过 GM | **架构强制** | UB↔L1 直拷不存在（Q1 证据 2）；L0C→UB fixpipe 无 API（A2 判死，joint_bottleneck §3.2）；GM→L0 直通无（同） |
| 同一行被多个消费核各自搬运 | **架构强制**（当前核划分下） | 无 multicast DMA（§3 查证：仓内无 API）；ws 每核私有（perf_log："ws 每核私有"）；L2 吸收读侧但写侧/服务带宽仍逐核发生 |
| **每（去重行 × 消费核）搬运一次** | **设计可达下限**（panel 路径实现） | = 每 (b,g) 组 panel 单次 staging/核：20 核 × 2 组 × 128 行 × 1152B ≈ 5.9MB/方向——比基线降 51× |
| **每（行 × topk 槽位）搬运一次（基线现状）** | **纯设计冗余** | 无任何架构理由要求逐槽位重复；官方口径冗余度 = 262,144 / 256 distinct = **1024×** |

**流分解（官方口径，读侧）**：基线行读 = 262,144 次（每 (s,j) 一次）；panel 下限 = 每核每组一次 staging 读 = 20×2×128 = 5,120 次（51×↓）；二者之差（~257K 次 = 98%）全部为槽位粒度冗余。**结论：Q2 的二分成立——"每（去重行×消费核）"是 AIV/AIC 分离 + 无组播下的强制项；"按槽位 vs 按去重行"的 51×（每方向）/ 1024×（vs distinct 全局）倍数是可消除的设计冗余。**

### Q3：可用机制清单（逐项查证，含"官方支持的其他传输机制"必查项）

**（a）已判死机制（引用既有证据，本轮不重试）：**

| 机制 | 判死证据 |
|---|---|
| 任务重组 / 数据依赖分桶 | golden_roofline.md L110：分桶 ≥100µs；generic_pipeline_analysis P6：任务→core 重排 ≤1% 无收益；perf_log：任务重排 ±0.8% 维持判死 |
| AIC GM 散射读（直采） | R5：100ns/行 ND→zN，含连续同率——分布无关判死（joint_bottleneck §3.2） |
| 手工流水 / 解门控 | P1 谱系：R2 +12.7%、P1 +4~6%（ablation 三会话确证） |
| Q 反转（Q 来到 KV 侧） | Q 重载墙 158-248µs/核 + 散射原子 202ns/op + 分桶 ≥100µs（golden_roofline L110） |
| 同字节重塑（bulk 化/描述符整形） | K5：128 行拷贝→1 bulk（同字节）-1.6% 噪声级——数据路径绑定（joint_bottleneck §2 裁决表） |
| LEAD/event/buffer-depth 参数搜索 | P2b +3.23% 三轮确证；DEPTH 8 flag id 用满（用户明令禁止重搜） |

**（b）本轮必查项——传输机制 API 查证（grep tilelang/language/ascend.py、ascend_tile.py、src/tl_templates/ascend/common.h、api-kernel-memory.md）：**

| 待查机制 | 查证结果 | 结论 |
|---|---|---|
| L2 cache policy / persistence hint | tilelang/language 与 tl_templates 中无任何 cache policy/persist 相关 API（grep 仅命中无关的 Broadcast 向量算子/persistent 循环原语） | **未证实，不得依赖** |
| multicast / 广播 DMA | 无 multicast API；common.h L1643 `Broadcast` 是**向量计算指令**（UB 内张量广播），非 DMA 组播 | **未证实，不得依赖** |
| MTE burst / 修饰符控制 | DataCopyPad 参数面仅 srcGap/dstGap（行间 gap）；api-kernel-memory §3 明确"不支持列方向（行内）strided access"——搬运模型 = 行内连续 + 行间 gap，无 burst/hint 修饰符 | **未证实，不得依赖**；且此约束直接封死"acc 列切片 T.copy"选列设计（§5.4 规避） |
| AIC 侧 L1 索引化访问 | L1 消费仅经 gemm_v0 内部 mte1 分形装载（ascend.py L418-462：A/B 必须 L1、C 必须 L0C）；无 L1 gather/索引寻址原语 | **不存在**（AIC 不能按索引采行——R5 判死的根因） |
| UB↔L1 直拷 | api-kernel-memory §1/§3：无此路径（workspace 强制中转） | **已证无** |
| L0C→UB fixpipe | A2 判死（joint_bottleneck §3.2） | **已证无** |
| GM→L0 直通 | 同上 | **已证无** |

**（c）本轮新查证的可用正向原语（设计支点，全部仓内实证）：**

| 原语 | 语义 | 仓内证据 |
|---|---|---|
| `T.tile.gatherb(dst, src0, offset, repeat_time, dst_blk_stride, dst_rep_stride)` | **UB 内 32B 块索引 gather**：offset 按字节寻址 src0，每 entry 搬 32B 块到 dst 连续位置 | testing/python/language/test_tilelang_ascend_language_elementwise.py L2622-2706（含 golden 语义：offset=字节偏移、uint32、每 entry 32B） |
| `T.tile.sort(dst, src, actual_num)` | 降序排序，输出 (value, index) 交织对，fp32（int32 需 cast，≤2^24 精确），任意 num 内部 sort32+merge | ascend_tile.py L434-484；examples/deepseek_v4/lightning_indexer.py L952 实战 |
| `T.tile.merge_sort` | 2/3/4 路归并（降序交织对，fp32，blockLen ≤ 4095） | ascend_tile.py L487-533；lightning_indexer L968-997 |
| `T.tile.gather_mask(dst, src, "P0101"/"P1010")` | 定式奇偶抽取（从交织对中抽 value/index lane） | ascend_tile.py L671-729；lightning_indexer L1062-1064 |
| `T.tile.atomic_add(dst_GM, src_local)` | **仅连续区域** atomic 加（dst=GM region，extent 合并）——**不支持散射地址** | ascend_tile.py L313-349（API 形状直接排除散射计数）；gqa_sink L437/472 实战 |
| `T.tile.arith_progression / select / clamp / broadcast(双轴)` | 等差数列、三态选择、夹取、双轴广播 | ascend_tile.py L398/L770；当前 kernel L632（axis=0）/L616（axis=1） |
| GM 标量读 / UB 标量读（GetValue） | `TokenIndices[s_row, 0]`（L307-309）、`rows_ub[row_j]`（L352） | 当前 kernel 逐字先例 |
| gemm_v0 操作数可换位 | A[M,K]×B[N,K]ᵀ→C[M,N]，A/B 仅要求 L1——**panel 作 A（M=topk）、Q 作 B（N=H）形状合法** | ascend.py L418-525（M/N/K 由 buffer 形状推导，n_actual%16 校验仅 N 轴） |

**（d）任务重组再评估（Q3 要求引用判死证据后裁决）**：分桶 ≥100µs（golden_roofline L110）+ 任务重排 ±0.8%（P6）⇒ 任何"按数据依赖重排任务流"的方案（含 §4 候选 4 的 panel 分块×任务过滤）维持判死。**本设计不重组任务流**——任务枚举序与基线逐字节同构（persistent 轮询 pid = k×grid+cid），仅改变每个任务的**数据供给方式**。

### Q4：D+tail 布局审计——确认最优，无废字节

**确认（不推翻）**：ws_kv 采用 D+D_tail 合并布局 [grid, DEPTH×BI, DDT=576]（kernel L215，rev1 已把 DMA 从 2×DDT 双 ws 减半为单 ws），C 侧单段 gemm K=576（L288 `T.gemm_v0(q_l1, kv_l1, ..., transpose_B=True)`）**整行 1152B 全部进入 K 累加**——主段 512 与尾段 64 在同一 dot-product 内，无 padding、无废字节、无第二次 gemm 调用开销。K=576 = 4×128+64 与 kL0Size=128 的 k 步进（kL0Size=576 单步实测 +2.9% 反向证据，perf_log iter3）。**结论：1152B/行 = 全部被消费的有效载荷，布局已最优；K5 已证同字节重塑无效——布局轴无剩余收益，流量消除只能走"减少搬运次数"（P3）。**

### Q5：通用驻留策略（核心）——选定"序列级去重驻留"，完整设计见 §5

**主候选验证结论（orchestrator 预判成立，含三处设计修正）：**

1. **"每序列 panel = 被引行并集 + L1 驻留 + V 按映射选列"方向正确**——流量账本：302MB×2 → ~12MB（20 核 × 2 组 × 147KB × 2 方向 staging），全 kernel 995MB → ~110MB。
2. **修正一（M 定宽 = topk，非 P_seq 运行时宽）**：gemm_v0 的 M/N/K 由 L1 buffer 形状在**编译期**推导（ascend.py L449-462），分形/mte1 循环编译期展开——**运行时 M 不支持**。M 取 Pmax=448 定宽时官方口径 acc 腿膨胀至 118MB + gemm 53µs（vs M=128 的 33MB/15µs），预期 ~240µs **超 220 目标**；M=topk 定宽时官方 P_seq=128=topk ⇒ **零计算膨胀**（gemm 同形 [topk,576]×[576→H]，FLOP 与基线逐块相同）。⇒ **panel kernel 取 M=topk 编译期定宽，资格阈值为 P_seq ≤ topk**。
3. **修正二（pos_table 改为运行时计数映射，无 GM 查表）**：任务书的"bitmap 扫描 + 行前缀和 + pos_table"需 262K 元素级散射写（bitmap/count 置位）——**仓内无散射原语**（atomic_add 仅连续区域 §2-Q3c；MTE 4B 小拷贝 262K×~25ns/40AIV ≈ 165µs 判死；S-pipe 标量 GM 写同量级判死）⇒ **pos 映射改为主 kernel V 侧每任务向量计数：pos_j = Σ_c 1{panel_rows[c] > idx_j}（compare-reduce，§5.4）**——panel_rows（降序）驻 UB，全部用已证原语（broadcast 双轴 + clamp 比较 + reduce_sum），零散射、零 GM 查表。
4. **修正三（资格判定 day-1 取保守档）**：P_seq 的精确设备侧计算需全量排序去重（sort/merge_sort 可行但吞吐未实测，风险表 R3）；day-1 资格判定 = **逐组"全行与首行逐元素相等"检测**（detect kernel 同构，~10-15µs）——数学上 sound：共享 ⇒ 并集 = unique(row0)，|unique(row0)| ≤ topk ⇒ P_seq ≤ topk 恒成立。**panel kernel 本体按通用 pos 机制实现（正确域 = {P_seq ≤ topk} 任意分布），day-1 资格只是保守子集；tier-2（sort 去重资格）零 kernel 改动升级覆盖（§5.7.3）。**

**大 P_seq 处理（设计重点，orchestrator 警告项）**：over-subscription 分析成立——acc 腿 = tasks×H×P×4B，P_seq=2048 时 268MB **反超 ws_kv**（且 gemm 膨胀 P/topk 倍）⇒ 大 P_seq panel 无收益；panel 分块×任务过滤又涉分桶（≥100µs 判死）⇒ **结构化回退是唯一正解**：P_seq > topk（day-1 判据：非全共享）→ 设备侧 Violations 标量 + 推测双发射回退现通用路径（§5.7）。回退判定机制裁决见 §5.7.2（区分"契约特化三模式"与"正确性通用的性能自适应"）。

---

## 3. 硬件可行性核实清单（三类限制区分，逐项证据出处）

| # | 机制 | 类别 | 结论 | 证据出处 |
|---|---|---|---|---|
| 1 | UB↔L1 直拷 | 硬件架构限制 | 无——GM workspace 中转强制 | api-kernel-memory.md §1（workspace 机制原文）/§3（路径表） |
| 2 | L0C→UB fixpipe | 硬件/编译器限制 | 无 API（A2 判死） | joint_bottleneck_analysis §3.2 |
| 3 | GM→L0 直通 | 硬件架构限制 | 无 | 同上 |
| 4 | AIC GM 散射读 | 硬件架构限制 | 100ns/行 ND→zN，分布无关 | R5（joint_bottleneck §3.2 引用） |
| 5 | AIC L1 索引化访问 | 硬件架构限制 | 无原语（L1 仅经 gemm mte1 分形消费） | ascend.py gemm_v0 契约 L418-462 |
| 6 | 散射 atomic / 散射 GM 写 | 编译器/API 限制 | atomic_add 仅连续区域；无 scatter API | ascend_tile.py L313-349 |
| 7 | T.copy 列方向 strided | 编译器/API 限制 | 不支持（行内连续模型，静默错位） | api-kernel-memory.md §3「多维切片的硬件限制」 |
| 8 | gemm 运行时 M/N | 编译器限制 | M/N/K 编译期定形（本设计 M=topk 的根因） | ascend.py L449-462 形状推导 |
| 9 | L2 cache policy/persistence | 未证实 | 仓内无 API——**不得依赖** | §2-Q3b grep 全负 |
| 10 | multicast DMA | 未证实 | 仓内无 API（Broadcast 是向量算子非 DMA） | §2-Q3b |
| 11 | MTE burst/修饰符 | 未证实 | DataCopyPad 仅 srcGap/dstGap | §2-Q3b |
| 12 | T.tile.gatherb / sort / atomic(连续) / clamp / broadcast | **可用（正向）** | 仓内已测/实战 | §2-Q3c 表 |
| 13 | cross-flag 预算 | 可用 | 8 个合法 id（A2 FFTS 4bit）恰满 panel 协议 | 当前 kernel L141-146；0f11b3cc §7.1/§7.2 配平表 |

**设计含义**：(1)-(8) 封死一切"绕过 GM 中转/列切片/散射/AIC 直采"路径 ⇒ 唯一出路 = **减少经过 GM 的次数**（去重驻留）；(9)-(11) 不可依赖 ⇒ 方案不得建立在其上；(12) 提供 V 侧选列与 prep 构造的全部原语。

---

## 4. 候选筛选表与选定方案

| # | 候选 | 减量潜力（官方口径） | 可行性（证据） | 预期收益 | 风险 | 裁决 |
|---|---|---|---|---|---|---|
| 1 | **P3 序列级去重驻留（M=topk + 转置 gemm + gatherb 选列 + 双发射回退）** | 995→~110MB（9×） | 全机制已证：staging/8-flag/组调度 = 0f11b3cc 143.3µs 实测；gatherb/sort/compare-reduce = 仓内已测原语；转置 gemm = API 形状合法（spike-1 验证） | **-45~-52%（363→175-200µs）** | 转置 gemm 分形方向未实证（R1，首日 spike；备选链 §8） | **✅ 选定** |
| 2 | P3' M=Pmax=448 定宽（覆盖 P∈(topk,448] 部分共享域） | 同上但 acc 腿 118MB + gemm 53µs | 同上 | 官方 ~240µs **超 220 目标**（流量/算力双膨胀，§2-Q5 修正一） | 目标失守 | ❌ 拒绝（M=topk 为官方目标最优；其增量域由 tier-2 另行评估） |
| 3 | P3'' over-subscription 大 P_seq panel | P=2048 时 acc 268MB 反超 ws_kv | 算术判死 | 负 | — | ❌ 拒绝（Q5 警告确认） |
| 4 | panel 分块 × 任务过滤 | 理论可行 | 分桶 ≥100µs 判死（golden_roofline L110） | 负 | — | ❌ 维持判死 |
| 5 | ws_acc bf16 | acc 腿 49→25MB（-24MB，总量 -2.4%） | 可行 | ~-1% 量级 | §9.3 fp32 1e-4/1e-3 判据下 exp(bf16 logits) 精度风险；且不触 302MB 主腿 | ❌ 不立项（收益小、精度风险不对称） |
| 6 | 槽内去重（块内 idx 重复消除） | 典型 topk 无重复 ⇒ ~0 | gather_mask 定式不可数据去重 | ~0 | — | ❌ 拒绝（orchestrator 预判"收益 ~0"确认） |
| 7 | L2 persistence / multicast / MTE 修饰符 | 理论大 | **未证实**（§3 #9-11） | 不可依赖 | 违反"只信证据"纪律 | ❌ 标未证实，不立项 |
| 8 | 布局再优化（D+tail 已合并） | 0 | Q4 确认最优；K5 同字节重塑 -1.6% 噪声 | 0 | — | ❌ 已贴顶 |
| 9 | 任务重组/负载再平衡 | — | P6 ±0.8%、分桶 ≥100µs | ≈0/负 | — | ❌ 维持判死 |
| 10 | 连续性快路径（arange 检测→bulk） | ~-1-3% | 联合分析裁决表：与 P2b 同收益 + 特化邻域 | 小 | 特化嫌疑 | ❌ 已裁决不推荐 |

**选定 = 候选 1**。筛选依据链：流量消除是唯一量级出路（§1.2）→ 消除点必须在"槽位粒度"（Q2）→ 机制面只有"每核每组一次 staging"可达（Q3）→ M=topk 保官方零膨胀 + 双发射保通用性（Q5）。

---

## 5. 选定方案完整设计（P3）

### 5.0 编程模式与总体架构

**编程模式：Expert（继承基线，用户既定）**——显式 `T.alloc_L1/alloc_L0C/alloc_ub` + GM workspace 中转（ws_panel/ws_acc）+ 显式双流 `T.Scope("C")/T.Scope("V")` + 手动 cross-flag；pass_configs 不设置（与基线及 examples/sparse_flash_attention 同构）。prep kernel 为 V-only Expert（detect 同构）。

**总体架构：同流三连发（0f11b3cc auto 机制复用，语义重构为性能谓词）**

```
host 入口 sparse_mla_topk_reducesum(q, kv, topk_indices, lse, offsets, ...)  # API 逐字不变
  ├─ host 静态门控（编译期事实，无 D2H）：
  │    topk ≤ 128 ∧ 2·topk·DDT·2 + H·DDT·2 + 4096 ≤ 524288（L1 预算，Plan-A G1 镜像）
  │    ∧ 支持域 assert 全过 → 编译 prep + panel_guarded + generic_guarded 三 kernel
  │    否则 → 仅 generic（= 基线行为逐字节，topk=256 / kvdb1 等域外配置零影响）
  └─ 每调用（graph 可捕获，无 D2H）：
       Violations = torch.zeros(1, int32)        # 每调用新建（0f11b3cc §8.4 生命周期不变式）
       prep(Indices, Offsets, TokenIndices, Violations, PanelRows)     # kernel 边界 = grid 屏障
       panel_guarded(..., Violations, ...)       # 入口守卫 Violations[0]==0 才执行
       generic_guarded(..., Violations, ...)     # 入口守卫 Violations[0]!=0 才执行（= 现基线 kernel + 守卫）
```

- `generic_guarded` = 现 P2 交付态 kernel **逐语句不动** + 入口 `ntask = Select(Violations[0] != 0, raw, 0)` 守卫（0f11b3cc `_reducesum_generic_body(guarded=True)` 已验证形态——守卫不进入时零 flag op、循环零次，无悬挂，§7.5 论证）。
- 双发射的开销 = 被跳过 kernel 的空跑 launch（auto 实测透明性代价 +12-14µs 中 launch 部分，~8-12µs）；正确性不依赖守卫以外的任何时序。

### 5.1 数据规格与新增张量

接口（proto.yaml）**零变化**：q/kv/topk_indices/lse/offsets/token_indices 输入、ReduceSum 输出、attr 全集不变（§9 交付清单核对）。新增均为**内部 workspace**（不进 proto）：

| 张量 | shape | dtype | 用途 |
|---|---|---|---|
| `PanelRows` | [G×B, topk] | int32 | 每 (b,g) 组的 panel 行号表（降序、含重复无害、-1/越界值保留经 clamp 寻址） |
| `Violations` | [1] | int32 | 资格谓词（0 = panel 可走；>0 = 回退） |
| `ws_panel` | [grid, 2×topk, DDT] | bf16 | V→C panel GM 中转双缓冲（0f11b3cc 同形） |
| `ws_acc` | [grid, 2, topk, H_pb] | fp32 | C→V score 中转双槽（**[P, H] 转置布局**，§5.3） |

### 5.2 prep kernel（资格判定 + panel 构造，V-only 40 lane）

**架构参照**：`tl_sparse_mla_topk_reducesum_detect`（0f11b3cc L1393-1497，V-only、lane = cid×2+vid ∈ [0,40)、(seq,lane) 整除分区、链式相邻行比较、条件 atomic）。**新增职责**：每组的 panel 构造（首行排序）。

```
phase-1（全 lane 并行，detect 同构）：逐组 (b,g) 扫描
  对 seq b 的行区间 [bos, eos)、组 g 的索引块 Indices[s, g, :]：
    lane 分区比较相邻行（链式：row[i+1] vs row[i]，chunk C=min(256, 16384//GST) 行窗预载）
    任何不等 → 本 lane 记 violation
  语义（soundness 精确性）：链式相等 ⟺ 组内全行逐元素相等（无漏报路径——
    与 detect 的契约检测同一条数学性质，但此处仅作性能谓词，见 §5.7.2 论证）
phase-2（每组单 lane，lane = gb % 40）：panel 构造
  row0 = Indices[bos, g, :]                    # [topk] int32
  row0_f = cast fp32（≤2^24 精确，262K 域内）    # T.copy cast（当前 kernel idx_f_ub 同款）
  T.tile.sort(dst[2·topk], row0_f, topk)        # 降序 (value, index) 对
  vals = gather_mask(dst, "P1010")              # 抽 value lane（lightning L1062 同款）
  PanelRows[gb, :] = cast int32 写出            # 降序、含重复（重复列无害，§5.8.2）
phase-3：if violation: atomic_add(Violations, 1)  # 连续 [1] 区域 atomic（API 合法形态）
```

- **空序列**（S_b=0）：无任务无组，phase-1 `ncmp ≤ 0` 自然跳过（detect 同款）；单行序列：ncmp=0，共享平凡成立。
- **G>1**：组索引块按 (b,g) 双层枚举；Indices 列 g 的行块行距 G×topk（MTE2 行拷贝逐行发射，行内连续合法）。
- **耗时预估**：phase-1 = detect 同构（1MB 索引流扫描 + 向量比较链，auto 实测全链 175.7 vs shared 143.3 的差值含 detect ~10-15µs）；phase-2 = 每组一次 128 元素排序（微秒级）；**合计 ~10-15µs**（spike-3 校准）。
- **PanelRows 语义不变式（正确性根基）**：包含该组**全部**索引值（含 -1/越界/被 mask 的值——不做因果过滤，见 §5.8.4 论证）。

### 5.3 panel kernel——C 侧（组调度 + 转置 gemm）

**任务枚举与组调度**：与 Plan-A / 基线完全同构——persistent 轮询 `pid = k×grid + cid`，`total_tasks = ΣS × RH × G`；C 侧每 task 1 次 GM 标量读 `b_c = TokenIndices[s_row, 0]`，与 `prev_g/prev_b` 比较得组键切换（0f11b3cc L1015-1036 逐字复用；packed 布局保证 (g,b) 键沿核任务流字典序单调）。

**buffer（L1/L0C）**：
```python
q_l1     = T.alloc_L1([H_pb, DDT], dtype)          # 18KB（H=16/576/bf16）
panel_l1 = T.alloc_L1([2, topk, DDT], dtype)       # 双槽 ping-pong 294KB（L1 524KB 预算内，host 门控镜像 assert）
accT_l0c = T.alloc_L0C([topk, H_pb], accum_dtype)  # 8KB ≤ 128KB
```

**每 task 主体（组内）**：
```python
T.copy(Q[s_row, H0:H1, :], q_l1)                            # [H, DDT] 整行切片（基线同款）
# 转置 gemm（本设计核心改动，spike-1）：
#   C[c, h] = Σ_k panel[c,k]·Q[h,k]  ==  基线 acc[h, j↦c] 的同一 dot-product
T.gemm_v0(panel_l1[pslot], q_l1, accT_l0c, transpose_B=True, init=True)
#   M=topk（panel 作 A 进 L0A，kL0Size=128 分块 [128,128]×2B=32KB ≤ L0A 64KB）
#   N=H_pb（q 作 B 进 L0B [16,128]×2B=4KB；N%16==0 分形合法）
#   K=DDT=576（k 步进 4×128+64，与基线逐块 gemm 完全相同 ⇒ fp32 L0C 累加顺序相同）
T.wait_cross_flag(F_WSF + slot)                             # ws_acc 槽可写（V 读毕上上 task）
T.copy(accT_l0c, ws_acc[cid, slot])                         # FIX [topk, H] 8KB 连续
T.set_cross_flag("FIX", F1_ACC + slot)
```

**组切换**：`pslot_c = 1 - pslot_c`；`T.wait_cross_flag(F_PANEL + pslot_c)` → `T.copy(ws_panel[cid, pslot_c×topk:(pslot_c+1)×topk, :], panel_l1[pslot_c])` → `T.set_cross_flag("MTE2", F_PDONE + pslot_c)`（0f11b3cc L1023-1033 逐字复用）。

**位等价性论证（logits）**：panel 列 c 的行 = idx_j 经映射的 KV 行 ⇒ gemm 元素 (c,h) 与基线 (h,j) 是**同一对 576 向量的内积**，且 K 轴分块序（kL0Size=128）与累加序相同 ⇒ **logits 逐位一致**（Plan-A §9.1 等价性论证对转置方向同样成立——M/N 交换不改变逐元素 K 累加序）。

### 5.4 panel kernel——V 侧（staging + pos 计数映射 + gatherb 选列）

**组调度与 staging（0f11b3cc L1095-1200 复用）**：prologue 静态展开扫描 task 0..LA-1（LA=32）组键 + 主循环前瞻 k+LA（`if k+LA < ntask` 守卫，防幻影组 F_PANEL 错位死锁——§7.6 红字教训逐条继承）；组首现时：
```python
panel_rows_ub ← PanelRows[gb, :]            # [topk] int32 一次性载入（512B）
rows_f_ub ← cast fp32                       # 比较链用（每组一次）
for c in range(topk):                       # staging 行采集（每组每核一次，非每 task）
    row = clamp(panel_rows_ub[c], 0, ΣS_kv-1)   # -1/越界 → 合法行（垃圾列，永不选中+mask 兜底）
    T.copy(KV[row, g, :], panel_stage_ub[c - vid·HALF, :])   # 每 AIV 半区，行拷贝模式=基线 gather 同款
T.copy(panel_stage_ub, ws_panel[cid, pslot_v×topk + vid·HALF : ...])   # 单条 bulk
T.set_cross_flag("MTE3", F_PANEL + pslot_v)
```
（`panel_stage_ub [HALF, DDT]` 复用由相邻 mte3→mte2 对保护——Plan-A 同款。）

**每 task post（选列核心，全新段）**：
```python
T.wait_cross_flag(F1_ACC + slot)
T.copy(ws_acc[cid, slot, :, :], accT_ub)          # (a) 全片 [topk, H] 连续读 8KB（列切片被 §3#7 封死 ⇒ 全读+UB 内选）
T.copy(Indices[s_row, g, t·topk ...], idx_ub)     # (b) 本行原始索引 [HALF]（mask 链需要；基线同款载入）
idx_f ← cast fp32
# (c) pos 计数映射（比较计数，降序表）：pos_j = Σ_c 1{rows[c] > idx_j}
T.tile.broadcast(rows2d, rows_f_ub, axis=1)       #   [topk] → [CHUNK, HALF] 分块处理（UB 预算，§5.6）
T.tile.broadcast(idx2d, idx_f, axis=0)            #   [HALF] → [CHUNK, HALF]
T.tile.sub(d, rows2d, idx2d); T.tile.clamp(d, d, 0, 1)   # 1{rows > idx}（clamp 比较链，基线 mask 同款算术）
T.tile.reduce_sum(pos_acc, d, dim=0, clear=False) #   累加跨 chunk → pos_f [HALF] ∈ [0, topk)
# (d) gatherb 偏移构造：offset_j = pos_j × H×4（字节）；H=16 ⇒ 每列 2×32B 块（lo: +0, hi: +32）
off_lo ← pos_f × (H·4);  off_hi ← off_lo + 32     # 向量乘加（fp32 → int32 cast-copy）
# (e) gatherb 选列（两次调用，h 半区各一；spike-2 参数化直抄 testing L2622-2706）：
T.tile.gatherb(sel_lo, accT_ub, off_lo, ...)      #   [HALF] 个 32B 块 ← accT_ub 行首块
T.tile.gatherb(sel_hi, accT_ub, off_hi, ...)      #   [HALF] 个 32B 块 ← accT_ub 行次块
# (f) score 链（[HALF, H] 定向，与基线逐元素同值）：
#   acc = sel × sm_scale − lse2d_T（lse [H] bcast axis=0 → [HALF, H]）
#   exp(acc) × mask2d_T（m1/m2 由原始 idx 构造，算术链与基线 L626-632 逐字同款，bcast axis=1）
T.tile.reduce_sum(rs_ub, acc, dim=1, clear=True)  # (g) h 归约 → [HALF]
T.copy(rs_ub, ReduceSum[s_row, g, r, vid·HALF : ...])
T.set_cross_flag("MTE2", F_WSF + slot)            # ws_acc 槽释放
```

- **(a) 全片读的代价**：每 AIV 读全 [topk, H]（8KB）vs 基线 [H, HALF]（4KB）——+4KB/task/AIV，合计 +16MB（计入了 §6.1 流量账本）。备选（64 条 64B GM 行散射读 ~1.3-2µs/task）时间上同量级但机制面弱（无 gatherb 复用性），不取。
- **(c) 的语义**：降序表中 pos = 严格大于计数 = 该值的秩——重复值映射到首个出现列（§5.8.2 正确性）；不在表中的值（仅可能为无效索引，§5.8.4 不变式保证）映射到任意合法列，由 mask 归零。
- **(g) 的位差声明**：h 归约的加数遍历序（[HALF,H] axis=1 vs 基线 [H,HALF] axis=0）可能相差 ≤1ulp——**测试判据 = 对参考混合容差 + 同路径 3-run 逐位**（§7），跨路径不要求逐位相等。
- **scores 布局 [P,H] vs [H,P] 的访址代价（任务书要求项）**：[H,P] 列切片被 DataCopyPad 行内连续模型封死（§3#7，静默错位）；[P,H] 使列 = 连续 64B 行 ⇒ gatherb 32B 块寻址成立 ⇒ **[P,H]（转置 gemm 产物）是选列机制的唯一合法布局**，其代价（(a) 全片读 +4KB/task/AIV）已入账并接受。

### 5.5 Cross-flag 协议（8 id 预算复用，0f11b3cc §7.1/§7.2 配平表逐字沿用）

| flag | id | 方向 | 语义 | 计数配平 |
|---|---|---|---|---|
| F_PANEL[2] | 0,1 | V→C set@MTE3 | 组 panel 已落 ws_panel[槽]（mode-2 quorum） | 1:1 无 init |
| F_PDONE[2] | 2,3 | C→V set@MTE2 | ws_panel[槽] 已被 C 消费（L1 装载完成） | V 运行期 wait=组数-1 + 尾部 destroy×1/槽 |
| F1_ACC[2] | 4,5 | C→V set@FIX | ws_acc[槽] score 就绪 | 1:1 |
| F_WSF[2] | 6,7 | V→C set@MTE2 | ws_acc[槽] 可重写（gqa_sink pretend-free 同款） | V init×2 + 每 task set；C 每 task wait + 尾部 destroy×2 |

守卫早退分支（Violations≠0 时 panel 整体不进入）：C 循环零次 + 尾部 destroy 跳过、V 整体不进入（F_WSF 不 init、组调度零触碰）——**无悬挂/无半更新**（0f11b3cc §7.5 论文级论证逐字适用）。ntask==0 的核同理论证。全部 cross-flag set/wait 的 guard 条件与 vid 无关 ⇒ 两 AIV 恒同分支（mode-2 quorum 前提）。

### 5.6 内存层级规划与 buffer 预算

**数据通路（GM → L1/UB → L0）**：
```
KV[gm] ──MTE2 行采集──▶ panel_stage_ub[ub] ──MTE3──▶ ws_panel[gm] ──MTE2──▶ panel_l1[L1] ──mte1──▶ L0A
Q[gm] ──MTE2──▶ q_l1[L1] ──mte1──▶ L0B
L0C(accT) ──FIX──▶ ws_acc[gm] ──MTE2──▶ accT_ub[ub] ──gatherb──▶ sel[ub] ──V 链──▶ rs_ub ──MTE3──▶ ReduceSum[gm]
Indices[gm] ──MTE2──▶ idx_ub；PanelRows[gm] ──MTE2──▶ panel_rows_ub；Lse[gm] ──MTE2──▶ lse_ub
```
（无跨级违规：全路径 GM↔UB↔(GM)↔L1↔L0，workspace 中转遵守 §3#1。）

| 存储 | 占用 | 上限 | 余量 |
|---|---|---|---|
| L1（C 侧） | panel_l1 2×topk×DDT×2B = 294KB + q_l1 18KB ≈ **312KB** | 524KB | ✓（host 门控 assert 镜像） |
| L0C | [topk, H_pb]×4B = 8KB | 128KB | ✓ |
| L0A/L0B | kL0Size=128 分块：[topk,128]×2B=32KB / [H,128]×2B=4KB | 64KB/64KB | ✓ |
| UB（V 侧） | panel_stage 73.7KB + accT 8KB + sel 2×2KB + pos/cmp chunk 8KB（[CHUNK=128, HALF] 分块）+ idx/mask/lse 链 ~12KB ≈ **106-110KB** | 192KB | ✓（cmp 矩阵分块是预算关键） |
| GM 新增 | PanelRows G×B×topk×4B（官方 1KB）+ Violations 4B | — | 微量 |

### 5.7 回退机制与判定（大 P_seq / 非共享输入）

**5.7.1 判定谓词（设备侧，无 D2H）**：`Violations[0] == 0` ⟺ 全部 (b,g) 组逐行共享（day-1 谓词）。soundness：共享 ⇒ P_seq = |unique(row0)| ≤ topk ⇒ panel kernel 前提成立。**漏报不可能**（链式相邻比较数学精确）；误报（把可 panel 的输入判回 generic）只损失性能不损失正确性——day-1 的误报域 = {非逐元素共享但 P_seq ≤ topk} 的输入（如集合共享的乱序行、部分共享并集 ≤ topk），由 tier-2 收紧（5.7.3）。

**5.7.2 机制裁决：推测双发射（复用 0f11b3cc/00a93f81 已验证实现），及与被禁"Auto 分发"的区分**：

| 维度 | 被否决的 panel/auto（0f11b3cc） | 本设计 |
|---|---|---|
| 分发依据 | **数据内容契约**（行间索引共享——语义假设） | **数据维度谓词**（P_seq ≤ topk——去重后尺寸的性能判定；day-1 由共享检测 sound 地蕴含） |
| 违约后果 | 契约破坏 = 语义错误（依赖 detect+逐任务校验兜底防错果） | 无"违约"概念——谓词为假 = 走 generic，**两路径对任意输入各自正确** |
| API 面 | `panel_mode="auto"/"shared"/"generic"` 三模式参数 | **零参数零模式**（用户接口逐字不变） |
| kernel 正确域 | {行间共享}（内容域，kernel 内 j↔列恒等直映射） | {P_seq ≤ topk}（尺寸域，kernel 内 pos 按行独立计数映射——任意行序/重复/部分共享在域内均正确） |
| 检测 kernel 职责 | 契约校验（错果防线） | 性能路由（错果不可能——判错方向只有性能损失） |

**论证**：用户明令禁止的是"契约特化的三模式"（正确性建立在索引共享假设上、违约即错果的那套）；本设计是"正确性通用的性能自适应路径"——正确性由双路径各自的一般性保证（panel kernel 的 pos 机制无任何行间假设；generic kernel 是现基线），设备侧谓词只决定**性能**。该区分与任务书"正确性通用的缓存复用属允许范围"的界定一致。**选择双发射而非 kernel 内分支的原因**：两路径 UB staging 无法共存（panel_stage 73.7KB + 基线 kv_gather_ub 147KB > 192KB，TIR buffer 静态分配）；0f11b3cc 已验证守卫 kernel 形态与三连发 graph 兼容性（§8.4）。

**5.7.3 tier-2 升级路径（零 kernel 改动）**：将 prep 的资格判定从"逐行共享检测"升级为"全量排序去重 + P_seq 精确计数"（sort/merge_sort 分块 + 归并树，PanelRows = 归并前缀 [topk]，§2-Q5 修正二/三）——panel kernel **不改一行**（pos 计数机制天然支持集合共享/部分共享/任意行序），覆盖域从 {逐行共享} 扩到 {P_seq ≤ topk}。代价：prep ~10-15µs → 排序吞吐待实测（风险 R3）。**day-1 不上**（官方口径与全部共享域已被 day-1 覆盖；sort 吞吐未证实前进实现违反预注册纪律）。

**5.7.4 域外配置静态门控**：topk > 128 或 L1 预算超限（kvdb1 类 DDT=1024）→ host 编译期直接只发 generic（现基线行为，零影响）——Plan-A G1-G5 域检查的编译期子集（内容检查全部去除）。

### 5.8 正确性论证（逐项）

**5.8.1 官方口径（全行共享 arange）**：P_seq = 128 = topk，panel = 降序 arange；每行 pos 映射相同；logits 逐位 = 基线（§5.3 K 序论证）；score 链逐元素同值 ⇒ 混合容差内必然通过（§7 判据）。

**5.8.2 重复索引（行内重复 / 行间重复值）**：panel_rows 含重复列（sort 保留）——降序计数映射下重复值 x 的 pos = #{> x}（指向首个 x 列），重复列本身永不位于任何合法 idx 的映射目标 ⇒ 重复列是**死列**（浪费一列 gemm，无正确性影响）；两个 j 指向同一 idx ⇒ 同 pos 同列 ⇒ 输出相同值（语义正确：同一 KV 行的 score）。

**5.8.3 乱序（行内顺序任意）**：pos 按值计数与行内位置无关 ⇒ 任意排列正确（本机制相对 Plan-A 直映射的本质泛化点）。

**5.8.4 -1 / 越界**：三重防线——(i) PanelRows **保留全部原始值**（prep 不做有效性过滤——不变式：panel ⊇ 组内出现过的所有索引值，因此**任何**本行索引值都在表中、pos 映射到自身列，不会误选他列）；(ii) staging 采集时 clamp 到 [0, ΣS_kv-1]（仅保护寻址，与基线 §5.4 语义、参考实现 clamp 行为逐字一致——列内容 = KV[clamp(idx)] 与参考读 KV[clamp(idx)] 相同）；(iii) mask 链 m1·m2 由**原始** idx 构造（1{idx ≤ s_i}·1{idx ≥ 0}，基线算术逐字），无效索引贡献恒 0。全 -1 行：panel=[-1,...]，全列 clamp 行 0 垃圾，全 mask=0，输出精确 0（ref_reducesum_raw 的 0/0 域同构）。

**5.8.5 空序列 / 单行序列**：prep `ncmp ≤ 0` 跳过；无任务 ⇒ 组调度零触碰（§5.5 无悬挂论证）；[0,0,1023,2048] 空首段 offsets 回归用例覆盖。

**5.8.6 G>1**：组 = (b, g)，每组建独立 panel（PanelRows 按 gb 索引）；任务流 (g,b) 键单调（packed 布局）⇒ 组调度正确（Plan-A 同构支持 G>1）；g_l 组号缺陷（P2 修复的 LEAD 窗口缺陷）不适用——panel 路径无 LEAD 前瞻窗口，组键前瞻用纯 pid 算术。

**5.8.7 REPLICATE_H > 1（H_pb=64 分片）**：task 含 r 维，同 (s,g) 的 r 分片共享 panel（组键不含 r）；accT [topk, 64] = 32KB，gatherb 每列 8×32B 块（off 链 ×8 展开）——机制同构泛化。

**5.8.8 P_seq = 0**（组内全无效索引）：PanelRows = 全 -1/垃圾值排序，panel 照常 staging（clamp 合法行）+ gemm 照常（M=topk 定宽）——全列死列，全 mask=0，输出 0。**无 N=0 空矩阵路径**（M 定宽的价值）。

**5.8.9 竞态面**：ws_panel/ws_acc 槽复用由 F_PDONE/F_WSF 显式握手（不依赖时序推断）；panel_stage_ub/sel/accT 等 UB 复用由相邻 set/wait 对保护（#15v2/#22 教训逐条继承：尾体保护对无条件存在、位置关键）；L0-D7 竞态族 3-run 逐位验证。

### 5.9 与被否决 Panel 特化的区别（强制章节——为何属"通用缓存复用"而非"数据内容特化"）

| # | 被否决 Panel 特化（0f11b3cc，用户已禁） | 本设计（P3） | 通用性依据 |
|---|---|---|---|
| 1 | **索引共享契约**：panel 语义 = "每 (seq,g) 所有行共享同一 topk 向量"（DESIGN_panel_mode §1.1 契约条款） | **无契约**：panel 语义 = "被引行并集，宽度 ≤ topk"——正确域是**尺寸谓词**而非内容关系 | 正确性域 {P_seq ≤ topk} 与"行间是否共享"无逻辑蕴含（乱序/重复/部分共享在域内均正确，pos 机制按行独立） |
| 2 | **内容假设**：j↔panel 列恒等直映射（kernel 假定所有行同序） | **零内容假设**：pos_j = Σ 1{rows > idx_j} 逐行独立计算；mask 用原始 idx | 映射由数据推出而非由假设空缺 |
| 3 | **任意输入正确性**：违约输入靠 detect + 逐任务校验链**防错果**（三层防御存在的原因 = 会错） | **结构性正确**：双路径各自一般正确，谓词判错只影响性能；panel kernel 对域内任意分布（重复/乱序/-1/越界/空组）数学正确（§5.8） | 不存在"需要防御的违约" |
| 4 | **API 特化**：panel_mode 三模式参数 + host 门控 + detect 分发 | **API 零变化**：无新参数、无模式、无用户可见分发（proto.yaml 不动） | 接口层无特化面 |
| 5 | **降级语义**：shared 声明错误 = 错误结果（运行期校验拦截） | **优雅降级**：非 eligible → generic（性能回退，结果不变） | 降级是性能路径而非正确性路径 |
| 6 | 选列机制：契约直映射（j = 列） | **pos_table/计数映射**（值→秩，按行计算） | §5.8.3/5.8.2/5.8.4 的泛化来源 |

**结论**：本设计利用的是"**任意合法索引都成立的去重缓存复用**"（同一 KV 行在同组内只搬一次——对任何输入分布都成立的物理事实），而非"假设输入具有某种结构"——属任务书允许的"正确性通用的缓存复用"。

---

## 6. 性能预算与证伪条件（预注册，沿用项目纪律）

### 6.1 官方口径流量账本（panel 路径）

| 流量腿 | 基线（generic） | P3（panel 路径） | 依据 |
|---|---|---|---|
| V 行读（gather 源） | 302MB | **0.29MB**（staging 读 20核×2组×128行×1152B） | §5.4 staging 每组每核一次 |
| V bulk 写 ws | 302MB | **5.9MB**（ws_panel 写） | 同上 |
| C ws 读回 | 302MB | **5.9MB**（ws_panel → L1） | 同上 |
| Q 读 | 37.7MB | 37.7MB | 不变 |
| ws_acc 写+读 | 33MB | **49MB**（FIX 8KB + 双 AIV 全片读 2×8KB / task） | §5.4 (a) 全片读代价 |
| 输出 + idx/lse/PanelRows | ~20MB | ~13MB | idx 每行仍读（mask 需要） |
| **合计** | **~995MB** | **~110MB（9×↓）** | 带宽地板 110MB/2.96TB/s ≈ 37µs ⇒ **非带宽约束** |

### 6.2 时序预算与预期带

| 分量 | 估计 | 依据 |
|---|---|---|
| prep kernel | 10–15µs | detect 同构扫描（auto 实测 detect ~10-15µs 类）+ 每组 128 元素小排序 |
| 双发射空跑 launch | 8–12µs | auto 透明性代价 +12-14µs（detect+launch 合计）的 launch 分量 |
| panel kernel 主体 | ~143µs | Plan-A 143.3µs 实测锚（结构不变：MTE1 串行 73 + flag 42 + 引擎 ~60；msprof 分解 panel_metric2_bench §3.1） |
| 选列增量（V 侧） | 10–20µs | pos 计数 ~5-10（[128,64]×4-5 向量遍/task/AIV，V 引擎有 40%+ slack）+ gatherb ~2-3 + 全片读增量 ~5-8 |
| **合计预期带** | **175–200µs**（点估计 ~185µs） | vs 基线 363.1µs = **-45~-52%**；vs 220 目标 = 余量 9-20% |

**证伪条件（预注册）**：
1. 官方 BI128 graph-replay 多实例终值 **> 220µs** ⇒ 路线证伪（分解归因后允许一次定向迭代：若 prep > 35µs → 攻 prep；若选列 > 40µs → 攻选列/备选链；若 panel 主体 > 165µs → 查 staging/握手回归——迭代后仍 > 220 则按用户规则回退基线）；
2. 精度门任何一项 FAIL（§7 判据）⇒ 立即停性能迭代转精度归因；
3. 全量回归（70 用例）任何 FAIL ⇒ 回退；
4. 竞态族 3-run 任意非逐位 ⇒ 回退。
**计测纪律**：graph-replay 纯 kernel 主口径（batch=25/iters=10/warmup=5，官方 G6 逐字）+ 多实例交错扫描（±0.8-4% 分辨率地板，p2_final_bench §1 协议）+ msprof 机制证据（禁排名）；同会话配对为准。

---

## 7. 验证方案（Golden + L0 门槛测试计划；L1/L2/Boundary 由 Stage 2 场景 B 扩展）

### 7.1 Golden 函数（草案，PyTorch 参考实现——沿用现有测试参考，零改动）

- `ref_mla_topk_softmax(q, kv, topk_indices, offsets)`（test 文件 L~120-234）：逐序列 `exp((Q·KV_main + Q_tail·KV_tail)·scale − Lse)` + causal mask + head 归约——**Renormalized 参考实现**；
- `ref_reducesum_raw(...)`（L287-304）：原始 §1.3 公式参考（无归一化）——覆盖 0/0 域（全 -1 行）与 clamp+mask 语义逐字对齐（"Out-of-range indices are clamped for the gather and masked exactly like the kernel"）。
- 判定链：`check_vs_ref`（混合容差：err_ratio ≤ 1e-2 ∧ abs_err ≤ 1e-3 ∧ NaN=0 ∧ matched_ratio ≥ 0.99）+ `check_equal`（torch.equal，仅同路径多 run / 跨配置稳定性）。

### 7.2 L0-D 门槛测试计划（本设计新增族；规则 shape——panel 路径无 block_I 维，generic 路径沿用 BI 整除域）

| 用例 | 级别 | Shape / 配置 | dtype | 索引分布 / 覆盖点 | 判据 |
|---|---|---|---|---|---|
| **L0-D1 官方 golden 逐字**（panel 路径锚定） | L0 | S=2048/H=16/D=512/tail=64/topk=128/offsets=[0,1023,2048]/seed=42，arange 全行重复 | 固定组合 | 官方口径；**路径断言：Violations==0（panel 被走）** | golden 判据 + §9.3 混合容差 |
| L0-D2 同 shape 非共享（回退正确性） | L0 | 同 D1 shape，随机 causal seed=123 | 固定组合 | **路径断言：Violations>0（generic 被走）**；同 shape 双路径对照 | golden + 混合容差 |
| L0-D3 重复索引共享（死列语义） | L0 | 同 D1 shape，index_vec = [5,5,7,7,...] 含行内重复（共享） | 固定组合 | panel 路径；重复列 pos 映射（§5.8.2）；P_seq < topk | golden + 混合容差 |
| L0-D4 乱序集合共享（day-1 回退域） | L0 | 同 D1 shape，各行 = 同一向量的随机排列（集合共享、逐元素不等） | 固定组合 | **day-1 谓词回退 generic**（tier-2 机会记录）；正确性必须成立 | golden + 混合容差 + Violations>0 断言 |
| L0-D5 -1/越界共享 | L0 | 同 D1 shape，index_vec 混入 -1、ΣS_kv-1、越界值 | 固定组合 | panel 路径 clamp staging + mask 三重防线（§5.8.4）；附全 -1 行用 raw 参考 | golden + 混合容差；全 -1 行 vs raw 参考精确 0 |
| L0-D6 大 P_seq 回退 + G>1 | L0 | S=2048/H=32/**G=2**/topk=128，full_range 随机（P_seq ≫ topk） | 固定组合 | generic 路径 + 多组 Violations 聚合 + 无挂死 | golden + 混合容差 + Violations>0 |
| L0-D7 竞态压测族 | L0 | ①官方 shape ×3 run；②短流 S=64/D∈{128,256}/topk∈{32,64} ×3；③混合路径交替（共享/非共享交替调用）×3 | 固定组合 | flag 握手/槽复用/staging 竞态（§5.8.9） | 3-run 两两 torch.equal（逐位）+ 每 run vs ref |
| L0-D8 分组/分片扫描（panel 路径） | L0 | 共享索引 × offsets∈{单段/[0,1023,2048]/[0,0,1023,2048] 空首段} × G∈{1,2} × H∈{16,32}；RH>1（H=128）≥1 例 | 固定组合 | 组调度 (b,g) 键、空序列、REPLICATE 分片共享 panel（§5.8.5-5.8.7） | golden + 混合容差 |
| **L0-D9 性能锚定（非精度门）** | L0-perf | 官方 golden 逐字 × BI128，graph-replay 多实例（§6.2 纪律） | 固定组合 | 预期带 175-200µs；**220µs 证伪线**；分解报告（prep/launch/panel/选列） | §6.2 预注册协议 |

路径断言机制：host 侧模块级 `LAST_DISPATCH`（记录 Violations 张量引用，仅测试读——热路径零 D2H；graph-replay 下测试在 replay 外同步读取）。

### 7.3 精度标准（§9.3 逐字继承——语义与 dtype 域未变）

| 数据 | dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|---|---|---|---|---|---|
| ReduceSum（唯一输出） | float32 | 1e-4 | 1e-2 | 1e-3 | 0.99 |
| Q/KV 输入（经 fp32 L0C 累加，比较恒作用于 fp32 输出） | bfloat16 | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |
| Indices/Offsets/TokenIndices/PanelRows/Violations 输入 | int32 | 0 | 0 | 0 | 1.0 |

辅助全局判据：err_ratio = RMS(err)/RMS(ref) ≤ 1e-2、NaN 计数 = 0；特殊值位置敏感契约（isnan/±Inf mask 先严格比较，再对有限值套容差）沿用。**跨路径（panel vs generic）不要求逐位相等**（h 归约序 ≤1ulp 声明，§5.4）；同路径多 run 逐位稳定为强制。

### 7.4 覆盖维度（供 Stage 2 场景 B 扩展基线）

D-DTYPE-{bf16,fp32,int32}（固定组合域内）；D-SHAPE-{ALIGNED,TAIL-1,TAIL-MID,PRIME,EDGE}（generic 路径 S 维任意/varlen 已支持；panel 路径 topk 维 M=topk 定宽无尾块——TAIL 声明 COVERAGE_NA 理由：M/N/K 全编译期常数）；D-VALRANGE-{S,M,L,ASYM}；D-PARAM-{dim_v,kv_group,sm_scale,block_I,grid_size}（block_I 仅 generic 路径）；D-SPECIAL-*（-1/越界/全 -1/空段）；D-EXC-*（域外 topk/shape 拒绝）；**新增 D-DISPATCH-{PANEL,GENERIC,MIXED}**（路径路由维度，L0-D1/D2/D6 锚定）。

---

## 8. 风险表与首日 spike

| # | 风险 | 等级 | 缓解 / 备选 |
|---|---|---|---|
| R1 | **转置 gemm**（panel 作 A 进 L0A、M=topk/N=16 方向）分形装载/mte1 行为未实证 | **高** | **spike-1 首日**：最小 kernel `gemm_v0(panel[M,K], q[N,K], C[M,N], transpose_B=True)` 与非转置结果 torch.equal（K 序位等价论证验证）。失败备选链（机制全已证）：非转置 gemm acc [H, topk] + V 全片读 [H, topk] + reduce-first（h 归约先行 → rs_all [topk] → GM scratch 落地 → 逐 j GM 标量读选值，TokenIndices 模式）——代价 +20-30µs，预期带退至 ~200-215µs（仍在 220 内为设计底仓） |
| R2 | gatherb stride/offset 参数化错误（[topk,H] 源 → [HALF,8]×2 目的） | 中 | spike-2：testing L2622-2706 参数化直抄改造；失败走 R1 备选链（其不依赖 gatherb） |
| R3 | prep 开销超预算（sort 硬件吞吐未实测；扫描链比 detect 多 PanelRows 写） | 中 | spike-3 计时；超 25µs 时 phase-2 排序移至 panel kernel V 侧 prologue（组首现时排序，与 staging 重叠——不改变正确性，只改执行位置） |
| R4 | detect 漏报（soundness 破坏 → panel 错果） | 中 | 链式相邻比较数学精确无漏报路径；L0-D 近邻分布（单元素差、跨组差）hammered；D2/D4/D6 路径断言锁定 |
| R5 | 守卫早退分支 flag 配平（悬挂/半更新） | 中 | 0f11b3cc §7.5 早退论证 + ntask==0 核路径双重复用；L0-D6/D7 无挂死验证 |
| R6 | UB 预算超限 | 低 | cmp 矩阵 [CHUNK, HALF] 分块（§5.6 已入账 106-110KB/192KB）；编译期 assert 镜像 |
| R7 | 官方终值 >220（选列/prep 超预算） | 中 | §6.2 证伪条件预注册 + 分解归因迭代；R1 备选链保底 200-215 |
| R8 | 跨路径 ulp 差异误判回归 | 低 | §7.3 判据显式声明（vs ref 容差 + 同路径逐位） |
| R9 | PanelRows 通道 prep→panel 竞态（同流 kernel 边界） | 低 | kernel 边界 = 隐式 grid 屏障（stream 序）；L0-D7 混合路径压测 |

**首日 spike 顺序**（依风险依赖链）：spike-1（转置 gemm 位等价）→ spike-2（gatherb 参数化）→ spike-4（panel kernel 端到端 shared 直跑，对照 143.3µs 锚 + 选列分解）→ spike-3（prep 计时）→ spike-5（三连发 + graph-replay 兼容 + 空跑开销）。spike-1/2 任一失败即启用 R1 备选链重估（预期带仍 <220 时继续，否则回报 orchestrator 重裁决）。

---

## 9. 交付清单与同类实现引用

| 交付物 | 状态 |
|---|---|
| `examples/dsa_sparse_finetune/DESIGN_dataflow.md` | 本文档 |
| `examples/dsa_sparse_finetune/proto.yaml` | **核对不动**：接口零变化（inputs q/kv/topk_indices/lse/offsets/token_indices、outputs reducesum、attrs dim_v/kv_group/sm_scale/block_I/grid_size、dtype 全集与 §7.3 一致）——P3 全部改动在 kernel 内部与 workspace，不进 proto；block_I 语义收窄为"generic 路径分块参数"记入实现注释（非接口变更） |
| Stage 2 输入 | 本文档 §5（机制）+ §7.2（L0-D 族）+ §8（spike 顺序与备选链）+ §6.2（证伪条件） |
| 历史保全 | git 22bc68a8（基线）/ 0f11b3cc（Plan-A 参照）不动；实验失败按用户规则回退基线 |

**同类实现引用（全部实读）**：
- `examples/dsa_sparse_finetune/sparse_mla_topk_reducesum.py`（基线 kernel：gather/flag/mask 链先例，L215/L284/L350-353/L616-632）
- git `0f11b3cc:examples/dsa_sparse_finetune/sparse_mla_topk_reducesum.py` L812-1366（panel body：staging/8-flag/组调度/LA 前瞻）、L1393-1497（detect kernel：prep 架构）、guarded/generic_guarded 形态
- `testing/python/language/test_tilelang_ascend_language_elementwise.py` L2622-2706（gatherb 语义与参数化 golden）
- `examples/lightning_indexer/example_lightning_indexer.py` L116-121、`examples/deepseek_v4/lightning_indexer.py` L952-997/L1062-1064（sort/merge_sort/topk/gather_mask 实战）
- `examples/attention_sink/example_gqa_sink_bhsd_bhsd.py` L162-178/L437-472（arith_progression、atomic_add 实战）
- `examples/gemm/example_gemm_intrinsic.py` L28-103（flag 全生命周期协议先例）
- `.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-kernel-memory.md`（§1 workspace 强制/§3 搬运路径与切片限制）
