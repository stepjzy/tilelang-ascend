# gather 组合优化 spike：五候选可行性验证 + 微基准（MB-0..4）— 2026-10-12

> **route**: Stage A spike_bench（可行性验证 + 微基准；**不整合完整 kernel**——胜者整合属下一阶段）
> **baseline**: git `22bc68a8`（P2 交付态，官方 BI128 = 363.1µs；kernel/test 全程零改动 ✓）
> **靶子**: P3 证伪的精确病灶——搬运机制成立（995→118MB）但无契约选值链 + panel 地板 291µs → e2e 894µs
> **环境**: 910B3（编译 arch dav-2201）、CANN 9.2.0、torch 2.7.1 + torch_npu；脚本 `/tmp/opencode/mb/`
> **测量纪律**: 全部真机；每项 ≥3 轮取中位；关键对照（MB-4）同会话交错配对（A/B/A/B）；
> 微基准量级效应 >10%，单实例口径，跨会话漂移 ±3-21% 已观测并如实标注（配对结论不受影响）。

---

## 0. 一句话总裁决

**五条候选全部"机制可行"（没有一条死于不可逾越的硬件墙——每一堵墙都找到了已实测验证的桥），
选值链从 1.87µs/task 降到 0.19-0.23µs/task；但五候选合起来也到不了 220µs——
因为本 spike 证伪了 P3 的根因归因：瓶颈从来不是 36 条选值指令，而是每 task 的
装载/flag 流水地板（V 侧 ~5.9µs、C 侧 ~5.4µs 串行口径），选值指令只是恰好超出地板阴影的那一截。**

---

## 1. P3 根因归因的证伪（本 spike 最重要发现）

P3 报告根因 1 称："≈36 条串行向量指令/task/AIV，实测 ~5.5µs/task（≈153ns/issue）"。
**该 153ns/instr 是把整个 5.5µs 除以 36 导出的循环论证**。实测拆解（MB-0，P2 生产同款
2-槽环预取结构 + P3 逐字 flag 对，Expert 模式）：

| 每 task 分量（V 侧，102 task/核 ×20 核） | 实测 |
|---|---|
| loads(idx 256B + accT 8KB + lse 64B) + 3 组 flag 对 + cast + store（**empty 基线**） | **5.94µs** |
| P3 完整选值链（cast + reduce-first + 等值矩阵 + k 归一 + div + mask，30 条 V 指令） | +1.87µs（合计 7.81） |
| gather 选值链（cast + reduce-first + **1 条 vgather** + mask，16 条 V 指令） | **+0.23µs**（合计 6.18） |

- **P3 链的真实成本 ≈ 1.87µs / 30 指令 ≈ 62ns/指令**，不是 153ns。
- 5.5µs 中约 **3.6-4µs 是装载/同步结构地板**，链只是超出地板"阴影"的部分：
  16 条指令的 gather 链完全藏进阴影（+0.23µs），30 条的等值矩阵链超出 ~1.9µs。
- **指令级重排无效**（MB-0c）：两条完全独立流的指令交错（p3_ilv2）= 7.75 vs 7.81µs/task
  （-0.8%，噪声级）→ V pipe 对独立链无可观测重叠，**减指令数是唯一杠杆**。

---

## 2. 五候选逐条：可行性证据 + 微基准

### C1 框架 int32-offset 元素 Gather 泛化 —— **可行，且现在就能用（零框架改动的路径已存在）**

**封装链定位（offset uint32 硬编码点）**：
- CANN 层：`kernel_operator_vec_gather_intf.h` L67-100 —— `AscendC::Gather` 的
  `srcOffset` 硬类型 `const LocalTensor<uint32_t>&`（Level 0/2 三个重载全部如此）。
- dav_c220 实现层：`kernel_operator_vec_gather_impl.h` L57-60 —— 数据 dtype
  SupportType 含 **float**（元素 gather 的值类型从来不是问题）；
  L31-35 `Gatherb` 的 SupportType 仅 **uint16/uint32**（P3 spike-2 结论复核属实）。
- tilelang 暴露层：`tilelang/language/ascend_tile.py` L1792-1850（docstring "Must be uint32"，
  无强制检查）；codegen 直通 `AscendC::Gather`（`src/target/codegen_ascend.cc` GatherCodegen）。
- **P3 的真正死因**：P3 尝试 `T.tile.cast`（int32→uint32 **数值**转换）——dav_c220 vconv
  无该实现（本 spike C2b 再次实证：编译失败）。**正确出路是位视图不是数值转换**：
  `T.reinterpretcast`（`tilelang/language/ascend.py` L618；codegen L3148-3162 生成
  `LocalTensor<uint32_t> x = y.ReinterpretCast<uint32_t>()` —— **零运行时指令的 C++ 别名**）。
- **现成证据链**：`examples/cann-bench/gather/gather.py` L72-80 已用
  `mul(int32)→reinterpretcast(uint32)→gather(fp32 值)` 完整链，本机复跑 **20/20 PASS**
  （含 fp32 值 × int32/int64 索引 7 个用例）——**P3 调研漏掉了这条已验证路径**。

**MB-1（fp32 值 × 设备侧 int32 偏移，128 槽位选 64 值，每 task 成本）**：

| 变体 | 每 task | 链成本（扣 empty） |
|---|---|---|
| g_full：cast + reduce-first + gather + mask（C1+C4 形态完整替代链） | 6.18µs | **0.23µs** |
| g_selonly：add + gather（隔离 gather 本身） | 5.67µs | ≈0（藏入地板） |
| g_dep1 / g_dep2：1/2 条严格串行依赖 gather | 5.66 / 5.44µs | ≈0（藏入地板） |

正确性：g_full 位级/数值全对（含 NaN 计数 0）；单条 vgather（64 元素=1 repeat，
`elementCountSingleRepeat=64`，intf_impl.h L204-210）在依赖链中的延迟远小于任务地板。

**框架 spike diff（人体工学改进，非必需）**：`src/target/codegen_ascend.cc`
GatherCodegen +16/-1 行——offset buffer dtype 为 int32 时自动发射
`.ReinterpretCast<uint32_t>()`。已实施→增量重建→**int32 偏移直传 gather 编译通过且数值正确**；
uint32 既有用户码字不变（cann-bench 20/20 + g_full 复测通过）→ **已回退**，
patch 存 `/tmp/opencode/mb/c1_spike_gather_int32_offset.patch`（可整体重放）。
uint32 既有用户零影响论证：cast 仅在 offset dtype==int32 分支发射，而该形态此前**编译必败**
（无存量用户）；uint32 路径生成的 C++ 逐字节不变。

### C2 fp32→uint32 bitcast + gatherb 选值 —— **机制位保真验证通过；元素粒度形态被 C1 支配（判死因：块粒度）**

- **C2a 位保真（真机实测 OK）**：fp32（含 +inf/-inf/NaN 特殊位型）→ reinterpretcast u32 视图
  → gatherb 块置换 → u32 直存，**逐位相等**（`/tmp/opencode/mb/mb_gatherb.py` k_bitcast_roundtrip）。
- **C2b（死因实证）**：`T.copy` fp32→uint32 跨 dtype 是**数值转换**（1.5→1）且
  dav_c220 vconv **无 fp32→uint32 实现（编译失败）**——位保真必须走 reinterpretcast。
- **元素选值形态判死**：gatherb 是 **32B 块粒度**（intf.h L40-54；仓内测试
  `test_..._elementwise.py` L2652-2689 实证 offset=字节偏移、每 repeat 搬 8 块、
  dstRepStride 单位=块）——选 4B 元素需 8× 过采样 + 块内 lane 抽取（又回到一次元素 gather
  或 3-4 条移位/mask 指令），**严格劣于 C1 的单条 vgather**。C2 的价值全部体现在 C3 的
  块对齐形态里。
- **附带新发现（API 限制）**：reinterpretcast 的 f32 视图别名**不能直接当 `T.copy` 的存储源**
  （实测返回垃圾值/错位数据）；别名只能被 V 算子消费（mul/sub/exp/reduce 实测全部正确）
  或以原宽 buffer 直存。整合时需注意。

### C3 gatherb 块 gather（uint16 索引？→ 实为 uint32 offset；[P,H] 行选）—— **可行，select-first 全链 0.19µs/task**

- 索引 dtype 澄清：offset 恒为 `LocalTensor<uint32_t>`（CANN 签名硬类型）；
  P3 spike-2 说的 uint16/uint32 是**数据** dtype 支持面。字节偏移语义（仓内测试 + 本 spike
  probe_gb2 三例全过）。
- [P,H] 布局下每行 H=16 fp32 = 64B = **2 块**；槽位选值 = 每 task 一次 gatherb
  （repeat=16 × 8 块 = 128 块 = 4KB，dstRepStride=8 连续落）——"槽位选值=2 次 gatherb"的
  原设问不成立：**一次调用搬全部 64 行**。
- **MB-3（select-first：gatherb 选行 → exp/mul/sub [64,16]（一半元素量）→ reduce → mask）**：
  **0.19µs/task**（6.16 vs empty3 5.97，同会话配对扣减），正确性 OK（置换 pos 表，位精确）。select-first 相比
  P3 的 select-after 还省一半 exp/reduce 元素量——但那部分本来就在地板阴影里，故与 C1
  的 0.23µs 在噪声内打平。
- **布局连带（对 C5 的反作用）**：C3 要求输出 **[P,H]**（行块连续）；若 gemm 出 [H,P]，
  列跨步 512B 非块连续，gatherb 不可用（元素 gather 仍可用偏移表吸收跨步）。

### C4 组内 Position 表缓存 —— **可行且近乎免费；另有"恒等映射"零成本退化形态（域口径需 design 裁决）**

- P3 是每 task 重算等值矩阵；C4 = pos 表按 (seq,g) 组算一次、组内全部 task 复用
  （idx 变更检测invalidate 可保留通用域：共享输入恒命中，失配走逐 task 重算回退分支）。
- **组 prologue 实测**（每核 102 组）：sort 路线（cast+sort[128]+gather_mask）**5.57µs/组**；
  count 矩阵路线（[128×128] compare+reduce）**6.71µs/组**。官方 shape 每组 ~51 task/核
  → 摊销 **0.11-0.13µs/task**，可忽略。
- **叠加算术（正交性）**：C4 是 C1/C3 的使能器（提供 uint32 字节偏移表 pos×4），
  与选值机制正交；C1+C4 = 每 task 1 gather + mask（16 V 指令，0.23µs）；
  C3+C4 = 每 task 1 gatherb + 半量 exp/reduce（18 V 指令，0.19µs）。
- **退化形态（design 裁决项，非本 spike 职权）**：day-1 资格域 = 组内全行相等（prep 已验证）
  ⇒ 若 panel 按 row0 **原序**构造，pos ≡ 恒等映射，**零成本**——但这把 panel kernel 的
  正确域从 {P_seq ≤ topk}（维度性质）收窄到资格门本身（P3 设计刻意保持 kernel 通用域、
  门只作性能回退）。属算法设计口径选择，三类限制标注见 §5。

### C5 非转置 gemm（panel 地板修复）—— **证伪：方向差异 ≤4-5%，gemm 本身完全不可见；地板真凶是 Q 装载/握手**

P3 报告嫌疑主因（"转置 gemm L0A [128×128]×2B=32KB 恰满乒乓槽预算 → mte1↔mma 重叠余量归零"）
的机制核对：`common.h` L1274-1276 `kL0Budget = 64KB/(kNumSteps>1?2:1)`，K=576/kL0Size=128
→ 5 步 → 预算 32KB/槽——转置臂 L0A 32KB **恰满**、非转置臂（A=q [16,576]，Plan-A 形态，
git 0f11b3cc L1033 同款）L0A 4KB、panel 32KB 移 L0B。**机制存在，但实测不成立**：

**MB-4（P3 地板变体结构：C 侧 Q 装载+gemm+FIX+跨核握手，V 侧 panel 供给+空转握手；
同会话交错配对，6/6 轮）**：

| 臂 | 中位 µs | 配对结论 |
|---|---|---|
| T_full（转置，P3 形态） | 616.9 | 基准 |
| N_full（非转置，Plan-A 形态） | 591.9 | **-4.1%**（6 轮配对 5 胜 1 负，均值 -5%） |
| T_nogemm / N_nogemm（去 gemm） | 592.6 / 591.0 | **= full**：gemm 完全藏入流水 |
| T_qgemm / N_qgemm（Q 装载+gemm，无 FIX/握手） | 551.4 / 550.9 | **两方向相等** |
| qonly（仅 Q 装载 18KB GM→L1） | 554.6 | **= qgemm：gemm 增量为零** |

- 输出正确性：T vs N 转置视图 6/6 相等；对 torch golden rel=4.4e-7/7.8e-7（末写者口径）；
  最小 gemm 探针 rel=4.1e-7。
- **解读**：[128,576]×[576,16] gemm 仅 2.36 MFLOP，两种方向都被流水完全隐藏；
  **panel 地板的真凶是每 task 18KB Q 小块 GM→L1 装载的串行化（qonly=5.4µs/task 未流水口径）
  + FIX/握手**——291µs 地板的归因应从"L0A 预算"改判为"每 task 装载/握手粒度"。
  与 §1 的 V 侧地板发现**同构**：两侧的每 task 流水开销才是共同大象。
- N_full 的 -4% 不足以称为"地板修复"（修复假设需要 ~2×），且若采用 C3 则必须保留
  转置 [P,H] 输出——**C5 与 C3 存在布局冲突，与 C1 兼容**。

---

## 3. MB 数据总表（紧凑）

| 基准 | 变体 | 每 task/组 | 链成本（扣地板） | 正确性 |
|---|---|---|---|---|
| MB-0a | p3_full（P3 等值矩阵链） | 7.81µs | **1.87µs** | 306/306 task 全对 |
| MB-0b | p3_selonly（仅选值） | 6.46µs | ~1.0µs | 对 |
| MB-0c | p3_ilv2（双流交错） | 7.75µs | ≈p3_full（-0.8%） | 对（重排无效） |
| MB-0 地板 | empty（loads+flags+cast+store） | **5.94µs** | — | — |
| MB-1a | g_full（C1+C4 完整链） | 6.18µs | **0.23µs** | 对（nan=0） |
| MB-1b | g_selonly（gather 隔离） | 5.67µs | ≈0 | 对 |
| MB-1c | g_dep1/2（串行依赖 gather） | 5.66/5.44µs | ≈0 | 对 |
| C4-a | grp_sort（sort 路线） | 5.57µs/**组** | 摊销 0.11µs/task | 对 |
| C4-b | grp_count（count 矩阵） | 6.71µs/**组** | 摊销 0.13µs/task | 对 |
| MB-2 | C2a 位保真 / C2b 值转换 | — | — | **位精确 OK / 编译死** |
| MB-3 | c3_full（gatherb select-first） | 6.16µs | **0.19µs** | 对（位精确） |
| MB-4 | T/N_full（gemm 方向） | 616.9/591.9µs 全核 | **-4.1%** | 6/6+golden 对 |
| MB-4 | nogemm / qgemm / qonly | ≈592/551/555µs | gemm 增量≈0 | — |

（指令清单：p3_full 30 条 V 指令/task = 4 Broadcast+4 Maxs+4 Mins+3 Muls+3 Adds+3 Mul+3 reduce_sum+2 Sub+cast+Exp+Abs+Div；g_full 16 条；c3_full 18 条。同步：每 task 6 flag 操作 + 4 MTE 拷贝，三变体同构。）

## 4. 依赖链分析（153ns/issue 的解剖）

1. **153ns/issue 不存在**：30 指令链实测 1.87µs → 62ns/指令；5.5µs 的其余 ~3.6µs 是
   loads+flag 地板（P3 把地板份额误记到指令头上）。
2. **V pipe 无跨链重叠**（ilv2 实证）→ 重排/软件流水**不是**杠杆；**减指令数**是唯一杠杆，
   且收益呈"阴影阈值"形态：≤~16-20 条指令的链完全藏入地板（增量≈0），
   P3 的 30 条超出阈值 ~1.9µs，gather 形态（16-18 条）回到阈值内。
3. **到达 220 所需的 V post ≤1.4µs/task（P3 报告口径）在本架构下已由 C1+C4 达成**
   （0.23µs）——但**这不充分**：地板本身 5.94µs/task 才是 e2e 的绑定项（见 §6）。

## 5. 三类限制标注表（用户 req 7 预备）

| # | 限制事实 | 归类 | 证据 | 桥/规避 |
|---|---|---|---|---|
| 1 | AscendC::Gather offset 硬类型 uint32 | **AscendC·TileLang API** | CANN intf.h L98-100；int32 直传编译失败（本 spike 复现） | reinterpretcast 位视图（实测位保真）/ C1 spike diff（已验证） |
| 2 | int32→uint32 **数值**转换（vconv）无实现 | **AscendC·TileLang API**（底层 dav_c220 指令面缺失） | C2b 编译失败；P3 spike-2 CAST_NONE/RINT 同因 | 同上（位视图绕开数值转换） |
| 3 | gatherb 数据 dtype 仅 uint16/uint32（fp32 静默无效） | **硬件指令限制**（dav_c220 GatherbImpl SupportType） | dav_c220 impl.h L31-35；P3 spike-2 全零复现 | bitcast 别名（C2a 位保真实测通过） |
| 4 | gatherb 32B 块粒度（元素选值 8× 过采样） | **硬件指令限制** | intf.h L40-54 + 仓内测试字节偏移语义 | 块对齐形态（C3 行选）/ 元素选值走 vgather（C1） |
| 5 | uint32 tile 算术仅 fill | **硬件指令限制**（P3 spike-2；本 spike间接复核） | offset 算术必须 int32 域完成 | mul/add 在 int32 域 + 位视图（cann-bench 模式） |
| 6 | V pipe 独立链不可重叠 | **硬件指令限制**（流水行为） | MB-0c ilv2 -0.8% | 无（减指令数） |
| 7 | T.copy 跨 dtype = 数值转换（非 bitcast）；f32 别名不可作存储源 | **AscendC·TileLang API** | C2b + C2 调试实测 | 存原始宽 buffer / 别名仅作 V 算子源 |
| 8 | 每 task 小块 MTE2 装载串行开销（18KB Q ≈5.4µs 未流水；V 侧 loads+flags ≈5.9µs） | **算法设计限制**（任务粒度/流水设计，非硬件墙） | MB-4 qonly；MB-0 empty | 多 task 批量装载/握手摊销（下一阶段设计项） |
| 9 | pos 表恒等退化 vs P3 通用正确域 {P_seq≤topk} | **算法设计限制**（域口径选择） | §2-C4 分析 | 保留逐 task 失配回退分支（invalidate）可兼得 |
| 10 | C3 需 [P,H] 输出 × C5 产出 [H,P] | **算法设计限制**（布局耦合） | §2-C3/C5 | C1 与两者都兼容；C3+C5 互斥 |

**分布**：硬件指令 3 项（#3/4/6，全部找到桥）；API 4 项（#1/2/5/7，全部找到桥）；
算法设计 3 项（#8/9/10——其中 #8 是 220 的真绑定项，属可设计规避，非墙）。

## 6. 排序、整合建议与组合叠加算术

**排序**（按实测价值/风险）：
1. **C1**（元素 gather，经 reinterpretcast；spike diff 可选）——立即可用，选值链 1.87→0.23µs
2. **C4**（pos 表组内缓存）——C1/C3 使能器，摊销 0.11µs/task；恒等退化形态留 design 裁决
3. **C3**（gatherb select-first）——0.19µs，与 C1 打平但绑死 [P,H] 布局 + bitcast 复杂度；
   仅在"进一步压 exp/reduce 元素量"或"C1 路径受阻"时启用
4. **C5**（非转置 gemm）——证伪为地板修复（-4%）；仅当无 C3 时可顺手取
5. **C2**（元素形态）——被 C1 支配；保留其 bitcast 验证结论供 C3 使用

**组合叠加算术**（e2e = prep 29.3 + launch 25.3 + max(C 侧, V 侧)×102.4 task/核）：

| 组合 | V 侧/task | C 侧/task | 预期 e2e（官方 BI128 口径） | vs P3 894 | vs 220 |
|---|---|---|---|---|---|
| P3 原状 | 7.81µs | ~5.8-6.0µs | ≈894（实测复核模型 ✓） | 1.0× | 4.1× |
| **C1+C4**（推荐整合形态） | 6.18µs | ~5.9µs(T) | **≈660-690µs** | 1.3× | **~3.0×** |
| C1+C4+C3 | 6.16µs | ~5.9µs | ≈660-690µs | 1.3× | ~3.0× |
| C1+C4+C5 | 6.18µs | ~5.7µs(N) | ≈660µs | 1.35× | ~3.0× |

**五候选全集的天花板 ≈ 660-690µs**：选值已免费，但 V 侧地板（loads+flags 5.94µs/task）
与 C 侧 Q 装载串行（5.4µs/task 未流水）成为新绑定——它们不在五候选的攻击面内。

**220 的路径（本 spike 数据支撑的下一阶段设计项，非本阶段职权）**：
- 批量粒度：accT 装载按 4 task 合批（32KB 一次 MTE2）+ 握手按批摊销 → V 地板 5.9→~2-2.5µs/task；
  C 侧 Q 装载同思路流水化 → ~2.2-3µs/task；
- 叠加 prep/launch（54.6µs）后估算 **e2e ≈ 300-350µs**；
- 再进 220 需压缩 launch 25.3µs（双发射空跑）与 prep 29.3µs，或更深的合批（8 task）。
  **结论口径：220 在本硬件 API 面上没有被本次 spike 发现的不可逾越墙——
  它是"每 task 装载/握手粒度"的设计问题；五候选解决了选值机制，下一阶段必须解决粒度。**

## 7. 框架 spike diff 清单

| 文件 | 行数 | 内容 | 状态 |
|---|---|---|---|
| `src/target/codegen_ascend.cc` GatherCodegen | +16/-1 | int32 offset 自动 `ReinterpretCast<uint32_t>()` | **已验证（int32 直传 OK；uint32 回归 cann-bench 20/20 + g_full OK）→ 已回退**；patch: `/tmp/opencode/mb/c1_spike_gather_int32_offset.patch` |

工作树 = P2 基线（kernel/test 零改动；build/ 已从回退后源码重建，二进制=源码一致）。

## 8. 复现与工件

- 脚本：`/tmp/opencode/mb/{harness,mb_sel,mb_gatherb,mb_gemm}.py` + 探针
  `probe_{vid,gather,race,gb2,gb3,gb4,gemm,int32off,fix2,dbg,autosync}.py` + `final_gemm.py`；
  日志 `log_{sel2,gb,gemm}.txt`。
- 关键复现命令：`source set_env.sh && python /tmp/opencode/mb/mb_sel.py empty p3_full g_full`
  （MB-0/1）；`python /tmp/opencode/mb/mb_gatherb.py`（C2/C3）；`python /tmp/opencode/mb/final_gemm.py`（MB-4 配对）。
- 调试期发现（过程记录）：① tilelang JIT 输出由 adapter 分配并返回（调用方只传输入）；
  ② 紧邻 GM→UB 装载与消费者之间存在时序竞态（P2 用槽位环预取规避——本 spike 全部 bench
  采用同构环结构后 306/306 task 稳定全对；裸紧邻结构在 NTASK=102 形态可复现首 task 读旧值）；
  ③ AUTO_SYNC 对同 pipe 依赖逐对插 `PipeBarrier<PIPE_V>`（`ascend_sync_insert.cc` L1404-1413），
  生产 Expert kernel 实测无需（P2/P3 先例 + 本 spike 无 barrier 链 306/306 对）。
