# DESIGN_panel_mode — sparse_mla_topk_reducesum 共享索引面板（panel-mode）快速路径变体设计

日期：2026-10-09 ｜ 模式：first_design（变体设计——为已终态算子设计"共享索引面板"快速路径）
基线：kernel 终态 #19-p2+#22（`sparse_mla_topk_reducesum.py`，765 行，**只读参考**；原 DESIGN.md 已删除，kernel docstring 为架构真相源）
直接依据：`perf_tuning/golden_roofline.md` §5.2（方案蓝图）+ `perf_tuning/golden_metric2_bench.md`（G1=364.7us FAIL 现状）
语义声明：**语义零变化，纯性能特化**。数学公式与现 kernel 完全一致——
`ReduceSum[s,g,r,j] = Σ_h exp((Q[s,h]·KV_main + Q_tail·KV_tail)*sm_scale - Lse[s,h]) * causal_mask(idx<=s_i 且 idx!=-1)`

---

## 0. 概述

### 0.1 问题与机会

golden_metric2 实测（graph-replay 主口径）：BI=128 = **364.7us（vs 220us 判定线 FAIL +65.8%）**。roofline 归因：
G1 的 **90%（328us）= C 侧 ws_kv 302MB GM 往返**——通用索引架构下每个 (task, topk 块) 都要把
gather 好的 KV 行经 `ws_kv` GM 中转给 C，该成本与数据形态无关、架构强制。

golden 数据（arange 全行重复索引）满足一个更强的索引模式契约：**对每个 (sequence, g)，所有行共享
完全相同的索引向量 P**。此时每个 (seq, g) 只有一个 128 行的共享 KV 面板（panel），把它一次性载入
L1 常驻，C 流对每个 task 做一次小 dense GEMM `Q[16,576] × panel[128,576]^T`，即可**完全跳过
ws_kv 302MB 往返**（面板 staging 仅 ~11.8MB 全核总量，51× 缩减）。预期 364.7 → 60-90us（现实
估计，含 ws_acc 34MB 往返；theoA 理想 47.6us），220us 判定 PASS 余量 2.4-3.7×。

### 0.2 交付边界与不变量

- 同文件（`sparse_mla_topk_reducesum.py`）新增 `tl_sparse_mla_topk_reducesum_panel` kernel builder
  与 host 分发；**通用路径（`tl_sparse_mla_topk_reducesum` 及 host 入口默认行为）零修改**（仅新增
  分支），已终态的 #19-p2+#22 结构不回炉。
- 输出接口不变：`ReduceSum [ΣS, kv_group, REPLICATE_H, topk] float32`；接口层归一化仍在交付边界外。
- 不满足契约/门控的输入**必须**回落通用路径或显式报错（ErrFlag），禁止沉默出错。

---

## 1. 模式契约（特化触发条件，精确定义）

### 1.1 契约定义

对每个 (sequence b, kv_group g)：所有行 s ∈ seq(b) 的索引向量逐元素相等（含顺序）：

```
∀ s, s' ∈ seq(b):  Indices[s, g, 0..topk) == Indices[s', g, 0..topk)   （记公共向量为 P[b,g]）
```

panel 定义：`panel[b,g] = KV[bos_b + P[b,g][j], g, :] for j in 0..topk`（topk × DDT 行集，
P 含 -1/越界值时行号 clamp 到 [0, ΣS_kv-1] 仅保护寻址，贡献由 causal mask 置 0——与现 kernel
§5.4 语义一致）。causal mask（`P[j] <= s_i && P[j] != -1`）**逐 task 逐列**计算，与共享与否
无关——契约只约束索引描述符，不改变逐行 mask 语义。

契约是**数据内容性质**，host 分发只能靠调用方声明 + kernel 内运行时校验兜底（§8）。

### 1.2 panel 预算门控（容量实算，全部经出处核实）

| 存储单元 | 容量 | 出处 |
|---|---|---|
| L1 | 524,288 B | `tilelang-api-best-practices/references/api-kernel-memory.md` §2（出处 `$ASCEND_HOME_PATH/.../Ascend910B3.ini`）；`examples/attention_sink/.../example_gqa_sink_fwd_varlen.py` L201 注释交叉验证 |
| L0A / L0B | 65,536 B 各 | 同上 |
| L0C | 131,072 B | 同上 |
| UB | 196,608 B | 同上（现 kernel L162 注释 `170*1024` 预算同源） |

**每核 L1 panel 足迹（双缓冲 2 槽 + q_l1）**：

```
footprint = 2 × topk × DDT × 2B  (panel_l1 × 2)  +  H_per_block × DDT × 2B  (q_l1)  +  对齐余量
golden (topk=128, DDT=576, H=16): 2×128×576×2 + 16×576×2 = 294,912 + 18,432 = 313,344 B ≈ 306KB ≤ 512KB ✓（60%）
```

通式门控（host 侧纯元数据计算）：

```
G1(容量):  2×topk×DDT×2 + H_per_block×DDT×2 + 4KB 余量 ≤ 524,288
G2(L0B):   topk × kL0Size × 2 ≤ 65,536     # kL0Size=128 → topk ≤ 256（gemm_v0 K 分块 staging，
                                           #   tilelang/language/ascend.py L444-454 kL0Size 语义）
G3(L0C):   H_per_block × topk × 4 ≤ 131,072  # 16×topk×4 → topk ≤ 2048
```

golden 配置三点全过（G2: 128×128×2=32KB ✓，G3: 8KB ✓）。v1 域取保守值 **topk ≤ 128**（golden 域；
topk ∈ (128,256] 需 kL0Size=64 + 单缓冲 panel 变体，列为后续扩展，§11）；topk > 128 → 回退通用路径。

### 1.3 任务→核映射分析与摊销门控（轮询 vs 连续分块）

沿用现 kernel 的 **persistent 轮询（striped）映射**：`pid = k×grid_size + cid`，`ntask = ceildiv(total−cid, grid)`。
沿每核任务路径，(g, b) 键**字典序单调**（pid 升 → g 不减；g 内 s_row 不减 → b 不减，packed 布局保证），
故 panel 按序装载、无乱序回跳。

| 映射 | 每核触及 panel 数 | golden | 短 seq 密集（B 大） |
|---|---|---|---|
| 轮询（现 kernel，v1 采用） | min(B×G, ntask_core) | 2（batch0/batch1 各一） | 趋近 ntask_core → 摊销崩溃 → 需门控 |
| 连续分块（gqa_sink `task_range` 同款） | ≈ 跨界核 2 个、其余 1 个 | 1-2 | 每核 ~(chunk/seq_len)+1 → 容忍度高 |

**v1 决策：轮询**。理由：(a) golden（B=2）两映射 panel 数几乎相同（2 vs 1-2），性能无差；
(b) 与现 kernel 任务分解逐字节同构，#19-p2+#22 的全部流水/守护证明直接继承，改动面最小；
(c) 连续分块作为后续扩展（放宽 B×G 域），列 §11。

摊销门控（host 侧可算，offsets 在 `prepare_token_indices` 中本就 `.cpu().tolist()`，零额外同步）：

```
G4(组数):  B × kv_group ≤ 8                     # 轮询映射下每核 panel 装载数上界；
                                           #   golden B×G=2 ✓。每组装载成本 ~5-8us（§10），8 组 ≈ 40-64us 仍
                                           #   ≪ 通用路径 365us，安全垫充足
G5(摊销):  ΣS × REPLICATE_H × G / (B × G) = ΣS×RH/B ≥ 16
                                           # 每 panel 平均任务数；panel staging 流量 ≤ 通用 ws 流量 1/16
                                           #   golden: 2048/2 = 1024 ✓✓
```

### 1.4 域检查（沿用现 kernel 支持域）

dtype 固定组合（Q/KV=bfloat16、Lse/ReduceSum=float32、Indices/Offsets/TokenIndices=int32）、
dim/tail_dim 幂等、topk 偶且 ≥32、head_kv 域（16≤h≤64 幂等 或 %64==0）、padded_H≠head_kv 时
kv_group==1——全部沿用现 kernel 编译期 assert；**任一不满足 → 回退通用路径**（通用路径域 ⊇ panel 域）。

---

## 2. 编程模式选型

**Expert（沿用现 kernel 同模式）**：显式 `T.alloc_L1 / T.alloc_L0C / T.alloc_ub` + `T.Scope("C")/T.Scope("V")`
双流 + 手动 `T.set_flag/wait_flag`（核内）+ `T.set_cross_flag/wait_cross_flag`（核间，mode=2 quorum：
`tilelang/language/ascend.py` L191-210 "between AICs and AIVs within the same group"）。pass_configs 不设置。

理由：(a) 现 kernel 即 Expert 且已终态，变体必须同构以复用其全部同步纪律与实证教训（HISA 悬挂、
#15v2 相邻对、#22 尾体竞态）；(b) panel 常驻 L1 + 跨组调度需要显式 L1 生命周期控制，Developer 自动
规划不表达"组粒度常驻"；(c) 语义/结构零漂移降低评审与回归成本。

---

## 3. 特化 kernel 架构

### 3.1 数据流总图

```
                    ┌────────────────────── AIC（C 流, Scope("C")）──────────────────────┐
                    │ per (组变更): wait F_PANEL[pslot] → copy ws_panel[cid,pslot] →       │
                    │              panel_l1[pslot] (MTE2 144KB) → set F_PDONE[pslot]      │
                    │ per task k:   b(k) GM 标量读(组检测) → wait F_WSF[k%D] →            │
                    │              copy Q[s_row,H0:H1] → q_l1 → gemm_v0(q_l1,             │
                    │              panel_l1[pslot], acc_l0c, transpose_B=True, init=True) │
                    │              → copy acc_l0c → ws_acc[cid,k%D] (FIX) → set F1[k%D]   │
                    └───────▲───────────────────────────────────────┬────────────────────┘
                     F_PANEL/F_PDONE                                     F1（score 就绪）
                    ┌───────┴───────────────────────────────────────▼────────────────────┐
                    │ AIV×2（V 流, Scope("V")）                                          │
                    │ prologue: init-set F_WSF×2；扫 task 0..LA-1 组键（LA=32 次 GM 标量 │
                    │   读）→ 按序 gather 新组: P=Indices[s_row,g,:] → rows=clamp(P+bos) │
                    │   → 分块行读 KV → panel_stage_ub → ws_panel[cid,slot] (MTE3) →     │
                    │   set F_PANEL[slot]（复用槽前 wait F_PDONE[slot]）                  │
                    │ per task k: lookahead 组检测(b(k+LA) 1 次 GM 标量读,新组→gather) → │
                    │   wait F1[k%D] → read ws_acc → acc_ub → mask 链(缓存 P + s_i) →    │
                    │   exp → reduce_sum → store ReduceSum → set F_WSF[k%D]              │
                    │   + 逐任务契约校验: Indices[s_row,g,半区] vs 缓存 P → 违约          │
                    │     atomic_add(ErrFlag, 1)（不沉默出错）                            │
                    └────────────────────────────────────────────────────────────────────┘
```

- **panel 载入（GM→L1，每组每核一次）**：V 侧 gather（AIV MTE2 行读为已证最快路径，26.1GB/s×40
  生产校准；AIC 直采已判死——R5 spike r_aic=100ns/行）→ `ws_panel`(GM) → C 整块拷入 L1。UB↔L1 不能
  直拷（api-kernel-memory.md §1/§3，硬件架构），GM 中转为强制路径，但流量仅 2×panel_bytes×组数×核数。
  golden 全核总量 ≈ 11.8MB（写 5.9 + 读 5.9）vs 消除的 302MB。
- **C 流（AIC）**：Q 读 37.7MB（L2 热，每 task 18KB，与 gemm 的 M/MTE1 重叠）；单 gemm
  `[16,576]×[576,topk]`（K 序与现 kernel 逐块 gemm 完全一致 → **logits 逐位一致**，§9 等价性依据）；
  ws_acc GM 往返**保留**（A2 无 L0C→UB fixpipe，API 级缺失——roofline §5.1 m_tile 判死结论），
  16.8MB 写 + 16.8MB 读计入现实地板。
- **V 流（AIV）**：mask 链与现 kernel 同构（m1/m2 clamp 算术链），P 按 (b,g) 组缓存（m2 = 1{P≠−1}
  组内只算一次，m1 = 1{P[j]≤s_i} 每 task 用新 s_i 重算）；exp/head 归约/输出同现 kernel。

### 3.2 与现 kernel 的关系

- 新增 `tl_sparse_mla_topk_reducesum_panel(...)`（同文件、独立 `@tilelang.jit`，out_idx/workspace_idx
  声明新 ws 集合）+ host 入口 `sparse_mla_topk_reducesum(..., panel_mode=...)` 分发；
- 通用路径 builder 与默认调用行为**零修改**（`_BUILD_CACHE` 键扩展 panel 维度）；
- 测试 `test_sparse_mla_topk_reducesum.py` 扩展 panel 用例（§9），golden 判据不变。

### 3.3 循环与调度结构

- **T.Kernel(grid_size, is_npu=True) 一维 persistent grid**（现 kernel 同；三维 Kernel 本项目不支持）。
- C/V 双 scope 各一个 **`T.serial(ntask)`** 每 task 主循环（运行时界 `ntask = max(0, ceildiv(total−cid, grid))`
  ——静态上限 + 运行时界的合规模式；**不用 T.Pipelined**：其循环界必须静态，ntask 动态不满足，
  ascend-constraints §1 强制检测项）。
- V prologue 为**编译期静态展开** `for i in range(LA)` + 每步 `if i < ntask` 运行时守卫（§6.1；
  现 kernel prologue `for s in range(LEAD)` + guard 同模式）。
- 行级循环 `for j in range(topk/2)`（gather 行读）静态展开，MTE2 描述符流水发射（现 kernel
  `for j in range(HALF)` 同款）；**无 T.Parallel**（Expert 模式手动 scope，元素级并行由 V pipe 内建）。

---

## 4. 数据规格与内存规划（Expert 显式层级）

### 4.1 输入/输出（与现 kernel 逐字段一致）

| 张量 | shape | dtype | 说明 |
|---|---|---|---|
| Q | [ΣS, heads, DDT] | bfloat16 | DDT = dim + tail_dim（golden 576） |
| KV | [ΣS_kv, kv_group, DDT] | bfloat16 | |
| Indices | [ΣS, G, topk] | int32 | panel 契约对象；值域 [-1, ΣS_kv) |
| Lse | [ΣS, heads] | float32 | |
| Offsets | [batch+1] | int32 | cu_seqlens |
| TokenIndices | [ΣS, 2] | int32 | host 预构建（`prepare_token_indices`，既有） |
| **ReduceSum**（out） | [ΣS, G, RH, topk] | float32 | 不变 |
| ws_panel（新 ws） | [grid, 2, topk, DDT] | bfloat16 | panel GM 中转双缓冲；golden ≈ 5.9MB |
| ws_acc（新 ws） | [grid, 2, H_per_block, topk] | float32 | DEPTH=2 槽；golden ≈ 320KB |
| ErrFlag（新 ws） | [1] | int32 | 契约违约计数（host 预置 0，调用后检查） |

（通用路径 ws_kv 不再被 panel 路径使用；两 builder 各自 workspace_idx 独立。）

### 4.2 片上存储实算（golden 配置，per core / per AIV）

| 缓冲 | 层级 | shape | 字节 | 容量占比 |
|---|---|---|---|---|
| panel_l1 ×2 槽 | L1 | [2, topk, DDT] bf16 | 294,912 | 56.2% of 512KB |
| q_l1 | L1 | [16, 576] bf16 | 18,432 | 3.5% |
| **L1 合计** | | | **313,344 ≈ 306KB** | **59.7%** ✓ |
| acc_s_l0c | L0C | [16, topk] fp32 | 8,192 | 6.3% of 128KB ✓ |
| L0B staging（gemm_v0 内部，kL0Size=128） | L0B | [topk, 128] bf16 | 32,768 | 50% of 64KB ✓（与现 kernel BI=128 同形，实证） |
| panel_stage_ub | UB | [topk/2, DDT] bf16/半区 | 73,728 | 37.5% of 192KB |
| acc_ub/lse2d/mask2d/杂项 | UB | — | ≈ 15,000 | ≈ 7.6% |
| **UB 合计** | | | **≈ 89KB** | **≈ 45%** ✓ |

对齐核查（decision-tree §3）：DDT=576 与行 1152B 均 32B 倍数（UB/L1 32B 对齐 ✓）；L0A/B 512B、
L0C 64B 由 gemm_v0 内部分形管理（M=16≥16、K=576 整除 16 满足 zN fractal——现 kernel L201-203 同论证）。
T.copy 切片核查：全部切片的列维均为 buffer 最内维（ws_panel/ws_acc/ReduceSum 的 topk/行维切片），
无 4D 列 strided 场景（api-kernel-memory.md §3 限制不触发）。

### 4.3 [REORDER-COST] 审计（panel 路径的 scatter→staging 重排）

```text
path: panel-gather（KV 散射行 → ws_panel 连续 → L1）
gm_passes: panel 数据 GM 读(KV 行)+GM 写(ws_panel)+GM 读(→L1) = 3 × panel_bytes × 组数 × 核数
           = golden 3×144KB×2×20 ≈ 17.3MB…（实际读 5.9/写 5.9/读 5.9，C 侧读后常驻，共 11.8MB 净流量 + KV 源行读 288KB L2 热）
           vs 通用路径 ws_kv 往返 604MB（写 302+读 302）—— 51× 缩减
dma_transactions: 每 (组,核): 128 条 MTE2 行读(1152B) + 2 条 MTE3 bulk(72KB×半区) + 1 条 C 侧 MTE2(144KB)
                  golden 全核 ≈ 40 组装 × 131 ≈ 5.2K 条（vs 通用 26.2K×2 行描述符 + 824 块拷贝）
gm_scalar_accesses: C 侧 1/task + V 侧 2/task + V prologue 33/核 ≈ 3×102+33 ≈ 340/核（≈ numel 级？否：
                    ΣS=2048，占比 17%，且全部为流水化标量读，先例 gqa_sink 2/task 生产验证）
address_div_mod_per_element: pid 算术 per task（同现 kernel），无逐元素 div/mod
active_cores / serial_tasks_per_core: 20 核 / 102 task（golden）
largest_case_timeout_gate: golden S=2048/topk=128，单 case 预算 < 220us（设计预判 60-90us，§10）
result: pass
```

---

## 5. Tiling 策略

| 维度 | 块 | 依据 |
|---|---|---|
| gemm M | H_per_block=16（padded_H/RH 逻辑沿用） | 现 kernel 同；L0A fractal M≥16 |
| gemm N | **topk 整宽单 gemm**（topk ≤ 128 v1 域） | L0B staging topk×kL0×2 ≤ 64KB；golden 128×128×2=32KB；与现 kernel BI=128 gemm 完全同形（已验证配置） |
| gemm K | DDT=576，kL0Size=128（gemm_v0 内部 4.5 步推进，余数步内部处理——现 kernel 实证） | K 序一致 ⇒ logits 逐位一致（等价性证据） |
| topk > 128 | v1 不支持（host 门控回退）；扩展方案 = panel_l1 行切片 BufferRegion 多 gemm（gemm_v0 接受 BufferRegion，ascend.py L432-437） | §11 风险表 |
| V 侧 topk 半分 | 每 AIV 处理 topk/2 列（vid 半区，同现 kernel HALF 机制） | ws_acc/acc_ub/mask 全部按半区切 |
| Q 行 | 每 task 整行 [H, DDT] 连续切片，无 gather | 同现 kernel |
| 非整除 | 沿用现 kernel 域约束（topk%BI==0 等编译期 assert），panel 路径无新增尾块维度（topk 即 gemm N，无块尾） | — |

（GEMM 非整除检测：M=16/N=topk/K=576 均满足分形与对齐；576 % 128 = 64 ≠ 0 的 K 余数步由 gemm_v0
内部处理，现 kernel L266-269 同配置生产验证，非新增风险。）

---

## 6. panel 调度协议（C/V 组序列一致性）

### 6.1 组序列与单调性

- C 侧检测：每 task 1 次 GM 标量读 `b(k) = TokenIndices[s_row(k), 0]`（g(k) 纯 pid 算术），与
  `prev_key` 比较；变更 ⇒ `pslot ^= 1; wait F_PANEL[pslot]; copy ws_panel[cid,pslot] → panel_l1[pslot];
  set F_PDONE[pslot]`。**C-scope GM 标量读先例**：`examples/attention_sink/example_gqa_sink_fwd_varlen/
  example_gqa_sink_fwd_varlen.py` L268-270（`q_seqlens[bz]`/`kv_seqlens[bz]` 在 Scope("C") 内每 task 读，
  生产性能 1.06× GPU 基线）。
- V 侧供给：prologue 静态展开扫描 task i=0..LA-1（LA=32）的组键（LA 次 GM 标量读，一次性），
  按首现顺序 gather 各组（组 g → 槽 g%2，槽复用前 `wait F_PDONE[槽]`）；主循环每 task 检查
  `b(k+LA)`（1 次标量读 + 缓存比较；`k+LA ≥ ntask` 时 clamp 寻址 + guard 跳过——现 kernel pid_n
  clamp 同款），新组 ⇒ gather。
  **prologue 每步必须带 `if i < ntask` 运行时守卫**（与现 kernel prologue `if s < nblk` 同模式）：
  无守卫时 clamp 产生的幻影行会"检测"出本核任务流中不存在的组 → V 多 set 一次 F_PANEL → 与 C 的
  wait 序列错位 → 死锁/错数据。主循环 lookahead 的 `k+LA < ntask` 守卫同理（已述）。`last_gathered`
  组键仅可在守卫内更新。
- **一致性论证**：C 的 wait 序列 = tasks 0..ntask−1 的组首现序列；V 的 set 序列 = prologue
  （tasks 0..LA−1 首现）+ 主循环（tasks LA..ntask−1 首现）= 同一序列。槽位按序 0,1,0,1 交替，
  两侧严格同序同数 ⇒ F_PANEL set/wait 1:1，无漏组（LA 窗口内的短组由 prologue 扫描覆盖——这是
  prologue 必须扫描而不只看 task LA 的原因）。
- 提前量：LA=32 × C 每 task ~0.4-0.5us ≈ 13-16us ≥ 单组 gather ~5-8us（含 KV 行读 ~3us/AIV 半区 +
  MTE3 2-4us），组切换对 C 零暴露（pslot 双缓冲：新组 L1 装载与旧组残余 gemm 重叠，装载 144KB@46GB/s
  ≈ 3.1us 隐藏于 ≥13us 提前量内）。

### 6.2 尾块保护分析（#22 教训强制项）

现 kernel #22：gather 段被 guard 跳过时其内部保护对随之缺失 → rs_ub 竞态。panel kernel 同类风险
逐一排查：

| 风险点 | 保护机制 |
|---|---|
| lookahead gather 被 `k+LA ≥ ntask` 跳过 → 内部 mte3→mte2 保护对缺失 | 被跳过的 gather 不写 panel_stage_ub ⇒ 无需保护；**post 段 rs_ub 保护对无条件存在**（不依赖 lookahead 是否执行——#22"尾体无条件保护"原则直接沿用） |
| prologue/主循环 gather 连续复用 panel_stage_ub | 相邻 mte3→mte2 对（#15v2/#22 论证的唯一安全形态；gather 稀疏（每组一次），wait ≈ 免费） |
| ws_panel 槽 s 被组 g+2 重写 vs C 读组 g | 显式 `F_PDONE[s]` 握手（C 的 L1 装载 MTE2 完成后 set；V 重写前 wait）——不依赖时序推断 |
| panel_l1 槽 s 被组 g+2 装载 vs 组 g gemm 读 | C 同核程序序（组 g 全部 gemm 先于组 g+2 装载）+ gemm_v0 尾部内建 M→MTE1→MTE2 排空（现 kernel 跨 task q_l1/kv_l1 重写同款保护，common.h L1306/L1373） |
| ws_acc 槽 k%D 被 C(task k) 重写 vs V(task k−D) 读 | 显式 `F_WSF[k%D]` 握手（gqa_sink FLAG_WS*_FREE 同款；V prologue init-set ×2 + C 尾部 destroy-wait ×2 平衡计数） |
| 两 AIV 分支发散 | 全部 guard 条件（k、ntask、组键）vid 无关 ⇒ 两 AIV 恒同分支（mode-2 quorum 前提，现 kernel 同论证） |
| 悬挂（HISA 教训） | 全部 set/wait 位于 `ntask>0` / 组存在 的有效程序段；两侧序列一致性（§6.1）保证计数封闭；各 flag init/destroy 配平表见 §7.2 |

---

## 7. 同步协议

### 7.1 flag 预算（A2 FFTS flagId 4bit、8 个合法 id——现 kernel L134-135 注释）

| id | 名称 | 方向 | set 管道 | 语义 |
|---|---|---|---|---|
| 0,1 | F_PANEL[2] | V→C | MTE3（quorum：两 AIV 均写毕 ws_panel 才放行 C——mode-2） | 组 g 的 panel 已落 ws_panel[槽] |
| 2,3 | F_PDONE[2] | C→V | MTE2（C 侧 L1 装载完成） | ws_panel[槽] 已被消费，可重写 |
| 4,5 | F1_ACC[2] | C→V | FIX（L0C→GM 落地，现 kernel 同款） | ws_acc[槽] score 就绪 |
| 6,7 | F_WSF[2] | V→C | MTE2（V 读毕 ws_acc，gqa_sink 同款） | ws_acc[槽] 可重写（free） |

DEPTH=2（ws_acc 双槽），共 8 id 恰满合法预算。

### 7.2 计数配平表（每核，ntask>0 时）

| flag | set 总数 | wait 总数 | init/destroy 配平 |
|---|---|---|---|
| F_PANEL[s] | V: 该槽组数 | C: 该槽组数 | 严格 1:1，无 init |
| F_PDONE[s] | C: 该槽组数 | V: max(0, 该槽组数−1) | V 尾部 destroy-wait ×1/槽 |
| F1_ACC[s] | C: 该槽 task 数 | V: 该槽 task 数 | 严格 1:1（循环界 ntask 同构于现 kernel） |
| F_WSF[s] | V: 2(init) + 该槽 task 数 | C: 该槽 task 数 + 尾部 destroy ×1 | gqa_sink "pretend free + destroy" 同款 |

ntask==0：两侧整体不进入（F_PANEL/F_PDONE/F1 零 set 零 wait；F_WSF 不 init）⇒ 无悬挂。
核内（同 pipe 对间）沿用现 kernel 全套相邻对纪律（v→mte2 / mte2→v / mte2→mte3 / mte3→v / s→v）。

### 7.3 与编程模式匹配

Expert 手动同步：核内 11 类 set/wait_flag 相邻对（沿用现 kernel 逐点复用）、核间 8 个 cross-flag
（上表）。V 侧全部 cross-flag set/wait 位于有效块程序段（组存在/k<ntask guard 内）——HISA 头注释
教训的纪律性延续。

---

## 8. 分发策略（方案 A/B 对比与推荐）

### 8.1 两案对比

| 维度 | 方案 A：调用方声明（panel_mode="shared"） | 方案 B：自动检测 + 双发射 |
|---|---|---|
| 契约来源 | 调用方保证 + kernel 内逐任务廉价校验兜底（违约 → ErrFlag，不沉默） | on-device 检测 kernel（读 Indices 1MB + Offsets，跨行逐 (seq,g) 比对 → GM 标志）+ 特化/通用两 kernel 齐射、各自读标志自门控（非己模式整体早退） |
| host 同步 | 零 | 零（检测在 device） |
| graph 捕获 | ✓（metric-2 graph-replay 协议两案均可捕获） | ✓ |
| 通用路径改动 | **零** | **必须给通用 kernel 加入口守卫（动已终态 #19-p2+#22 代码）** |
| 编译/launch 成本 | 单 kernel | 双 kernel 编译 + 每次 launch 双发射（8.4us×2）+ 检测 kernel（估 10-20us：1MB 读 + varlen 分段比对逻辑本身即一个小型 varlen kernel） |
| 检测正确性责任 | 调用方（声明）+ kernel 校验（兜底） | 框架。**误判 = 沉默错误结果**（roofline §5.2 风险原文），检测 kernel 自身的 varlen 边界/竞态 bug 面与主 kernel 同级 |
| 交付速度 | 快（最小改动面） | 慢（检测 kernel + 通用路径守卫 + 双份回归） |

### 8.2 推荐：**方案 A**（v1），B 列为后续演进

理由：(1) 最小风险——通用路径零修改，终态代码不回炉；(2) 220us 判定与 metric-2 测量协议
（graph-replay 纯 kernel）不受影响；(3) B 的核心代价是"检测错误的代价 = 沉默错误结果"（roofline
原文）+ 动已终态代码，收益仅"调用方透明"，在 DSA sink-token 类调用方明知索引模式的现实场景下
收益有限；(4) A 的安全兜底完整：host 门控（G1-G5 元数据可判定）miss → 自动回退通用路径（调用方
无感知的性能回退），运行时逐任务校验违约 → ErrFlag 显式报错。

### 8.3 host 分发逻辑（伪代码）

```python
def sparse_mla_topk_reducesum(..., panel_mode="generic"):   # 默认 generic：行为与现版本逐字节一致
    token_indices = prepare_token_indices(offsets)           # 既有（offsets 已 .cpu().tolist()）
    if panel_mode == "shared":
        seq_lens = diff(offsets_list); B = len(seq_lens 非零段)
        if G1..G5 全过 and 域检查全过:                        # 纯 Python 元数据运算
            err_flag = torch.zeros(1, int32).npu()            # 新建张量 CPU→H2D（数据准备，允许）
            out = panel_kernel(Q, KV, Indices, Lse, Offsets, TokenIndices,
                               ReduceSum, ws_panel, ws_acc, ErrFlag)
            if err_flag.item() != 0:                          # 调用后检查（非计时路径）
                raise RuntimeError("panel contract violated ...")
            return out
        # 门控 miss → 落入通用路径（静默回退，性能非正确性差异）
    return generic_kernel(...)                                # 现路径，零修改
```

[HOST-METADATA-AUDIT]
```text
operation: offsets → seq_lens/B 门控计算
input_stride -> output_stride: n/a（纯 Python int 列表，源自 prepare_token_indices 既有 .cpu().tolist()）
shares_storage / same_data_ptr: n/a（不触碰任何 NPU tensor）
aclnn_or_physical_copy: false
result: allow

operation: err_flag = torch.zeros(1, dtype=int32).npu()
aclnn_or_physical_copy: false（新建张量 + 一次 H2D，与 prepare_token_indices 同模式——数据准备类）
result: allow

operation: err_flag.item()（调用后读取）
aclnn_or_physical_copy: false（只读；位于计时路径外）
result: allow
```

---

## 9. 验证方案（Golden + L0 门槛测试计划）

> L0 计划由 `tilelang-op-test-design` 场景 A 生成；L1（功能含不规则 shape）/L2（异常）/Boundary
> 扩展由 Stage 2 在 L0 通过后按场景 B 生成，本阶段不产出。

### 9.1 Golden 函数草案（PyTorch 参考实现）

**主参考（数值权威）**：逐字复用 `test_sparse_mla_topk_reducesum.py::ref_mla_topk_softmax`
（fp32 softmax + varlen 逐 batch 循环 + causal mask + gather + renormalize + lse 生成——GPU 主仓
golden 的忠实移植，本仓已验证）。

**等价性参考（panel 特化专属）**：同一输入下 `panel 路径输出 vs 通用路径输出`。理论依据：两路径
gemm 同形（[16,576]×[576,128]）同 K 推进序（kL0Size=128、K=576）同 fp32 累加 ⇒ logits 逐位一致；
V 侧 mask/exp/reduce 链逐指令同构 ⇒ **ReduceSum 期望逐位相等**（torch.equal；若实现期发现偏差，
降级判据为 §9.3 fp32 混合容差并要求归因说明——属实现偏差而非设计许可）。

### 9.2 L0 门槛测试计划（规则 shape / dtype / 判据）

L0 用例总表（全部 golden 家族数据构造 = test 文件同源：seed=42 randn bf16、offsets varlen、
lse 由 fp32 参考生成）：

| # | 用例 | 配置 | 路径 | 判据 |
|---|---|---|---|---|
| L0-P1 | golden 经特化路径 | B varlen [0,1023,2048] / S=2048 / H=16 / D=512 / tail=64 / topk=128 / arange 重复索引 / panel_mode="shared" | panel | vs 纯 torch ref：err_ratio ≤ 1e-2 且 abs_err ≤ 1e-3 且 NaN=0（golden 判据原封）；AND vs 通用路径输出 torch.equal |
| L0-P2 | 通用路径回归 | 同 L0-P1 输入，panel_mode 缺省 | generic | 现 golden 判据不变（证明通用路径零变化；两 block_I 档 32/128 照跑） |
| L0-P3 | 契约违约检测 | L0-P1 输入但随机 1 行索引改动 1 个元素，panel_mode="shared" | panel | ErrFlag 置位、host 抛 RuntimeError（不沉默）；同输入缺省声明 → 走通用路径结果正确（回退通道） |
| L0-P4a | 空序列 | offsets=[0,0,1023,2048]（batch0 空），其余同 golden | panel | 判据同 L0-P1（空 seq 零 task、无组装载） |
| L0-P4b | ΣS=1 | S=1 / topk=32 / 单 task 单核 | panel+generic 双跑 | 判据同 L0-P1（ntask=0 核不进循环、无悬挂） |
| L0-P4c | 共享向量含 -1 | P = arange(96) ⊕ [-1]×32（尾填充），s<96 行触发 causal 截断 | panel | 判据同 L0-P1（clamp+mask 语义） |
| L0-P4d | 越界索引值 | P 含 ΣS_kv 越界值（clamp 保护 + mask 置 0） | panel | 判据同 L0-P1 |
| L0-P4e | 短 seq 密集超预算 | B=32 × S=64（B×G=32 > 8 → G4 miss） | host 回退 generic | 门控断言走了通用路径 + 结果正确 |
| L0-P5 | **竞态压测族（#22 强制）** | 小 ΣS=64 × 非默认 D（D=128/tail=64、D=256/tail=64）× topk ∈ {32,64} × **≥3 独立 run** | panel | 每 run 判据同 L0-P1；**3 run 间输出 torch.equal（bitwise 稳定）**——直接针对组边界×lookahead×槽复用三重交互的竞态面 |
| L0-P6 | host 门控单元 | topk=256（G2/G3 miss）、B×G>8（G4）、ΣS/B<16（G5）构造输入 | host 分发 | 断言各 miss 分支均分发到 generic（返回路径标记或行为验证） |

计时（粗测，沿用测试文件 `time_wall/time_event`，精确 profiling 归 Stage 3）：L0-P1 附带 wall 计时
打印，供 220us 判定的早期信号（非阻塞）。

### 9.3 精度标准表（按 dtype 混合容差；输出 dtype 全集 = float32 单值——本算子 dtype 固定组合，无 dtype 扫描维度）

| 数据 | dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|---|---|---|---|---|---|
| ReduceSum（唯一输出） | float32 | 1e-4 | 1e-2 | **1e-3** | 0.99 |

辅助全局判据（继承 golden，与上表联合生效）：`err_ratio = RMS(err)/RMS(ref) ≤ 1e-2`；
NaN 计数 = 0。整型输入（Indices/Offsets/TokenIndices）无输出精度行（不参与数值容差）。
panel-vs-generic 等价性判据：torch.equal（逐位，§9.1 论证）。

---

## 10. 性能预算分解（设计期预判）

### 10.1 C 链（AIC，per wall us，20 核并行口径，率取 roofline 校准值）

| 分量 | 估算 | 说明 |
|---|---|---|
| gemm | 28.6 | 4.53 GFLOP @ 158.4 TFLOPS（msprof 校准，±10%） |
| Q 读 | ~11.8（可重叠） | 37.7MB @ 20×160GB/s clean L2；MTE2 与 M 管线重叠，暴露 ≈ 0-4 |
| panel L1 装载 | ~3.1/组 ×2 组，隐藏于 ≥13us 提前量 | 144KB @ 46GB/s，双缓冲重叠 |
| ws_acc 写（FIX） | ~18 | 16.8MB 往返保留（A2 无 L0C→UB fixpipe——API 级缺失，§3.1）；与 gemm 部分重叠后暴露 10-14 |
| **C 合计（含暴露）** | **~45-55** | |

### 10.2 V 链（AIV×2）

| 分量 | 估算 | 说明 |
|---|---|---|
| exp/归约 | 39.2 | 3.93M elems @ 40×2.51 elems/ns（生产 msprof 校准） |
| ws_acc 读 | ~18（MTE2，与 V 计算流水重叠后暴露 5-8） | 16.8MB |
| panel gather | ~6/核（2 组 × ~3/AIV 半区） | 64 行×1152B MTE2 + 72KB MTE3，稀疏事件 |
| 逐任务校验 | ~4-8 | 256B/AIV/task MTE2 + [64] 比较 + reduce（§9 L0-P1 判据含） |
| 输出 | ≈0 | 1MB |
| **V 合计（含暴露）** | **~50-60** | |

### 10.3 系统预估

```
wall ≈ max(C 45-55, V 50-60) + coupling(F1/WSF 每 task 握手 + 流水填充/排空 5-10) + launch 8.4
     ≈ 60-80us
```

- vs theoA 47.6us（无 ws_acc 理想）：差距 = ws_acc 34MB 往返暴露 + 校验/ gather 杂项——已计入。
- vs 220us 判定线：**余量 2.7-3.7×**；vs 现 364.7us：**4.5-6× 提升**。
- 不确定性：±10%（cube 率校准）+ coupling 无独立测量（roofline §8 声明 28-105us 历史带；本设计
  每 task 显式握手 ×102/核，类比 gqa_sink cross_interval 模式，取 5-10us 保守带内低段）。

---

## 11. 风险点与降级预案

| # | 风险 | 概率/影响 | 降级预案 |
|---|---|---|---|
| 1 | **C-scope GM 标量读在本 kernel 上下文编译失败**（gqa_sink 先例存在但 dsa 系未用过；AIC 标量 load 的 codegen 路径未在本文件验证） | 低/高（阻塞） | 实现期首个 spike（半天）：最小 C-scope TokenIndices 标量读冒烟。失败则整路径回退通用（host 门控探测失败 = 静默 generic，零正确性风险，性能目标落空并如实上报）；备选架构（V 经 ws 携带组号 + C 经 GM 小向量读）同样依赖 C 侧标量读，不单独成案 |
| 2 | **flag 预算恰满 8 id**（DEPTH=2 + panel 双槽 + PDONE 双槽）；若框架对 cross-flag id 分配有额外保留 | 低/中 | 预案 1：F_PDONE 合并为单 id（组数奇偶配对论证放宽——牺牲小并发）；预案 2：DEPTH=1 + F1/WSF 各 1 id（coupling +2-4us） |
| 3 | **L1 规划冲突**（panel_l1 306KB + q_l1；若编译器 L1 地址自动规划与 gemm_v0 内部 staging 冲突） | 低/中 | 显式 `T.annotate_address` 固定 panel_l1/q_l1 基址（gqa_sink L202-217 先例：显式地址注解规避共享冲突）；仍冲突 → panel 单缓冲（−144KB，组切换暴露 ~3us/组，golden +6us 可接受） |
| 4 | **fixpipe/FIX 率低于预期**（ws_acc 写 18us 假设不成立） | 中/中 | ws_acc 双槽已给 FIX/M 重叠留结构空间；若 msprof 显示 FIX 串行化，改 L0C 双缓冲（[2,16,topk]=16KB，L0C 占用 12.5% ✓）+ gemm/FIX 交替 |
| 5 | **coupling 残留高于 10us**（每 task 4 组 cross-flag 握手 ×102） | 中/中 | F1/WSF 按 2-task 批量握手（gqa_sink cross_interval=2 同款），握手次数减半； flag 语义不变 |
| 6 | **L0B 恰满风险（topk=128, kL0=128 → 32KB 实为 50%）**；topk>128 扩展域 | — | v1 域已限 topk ≤ 128；扩展时 kL0Size=64 + panel 单缓冲 + BufferRegion 行切片多 gemm（§5） |
| 7 | **短 seq 密集输入误入 panel 路径**（摊销崩溃） | 低/低 | G4/G5 host 门控拦截（L0-P4e/L0-P6 用例锁定）；门控公式保守带（B×G ≤ 8、摊销 ≥ 16） |
| 8 | **运行时契约违约未被调用方发现**（ErrFlag 检查被跳过） | 低/高（正确性） | host 入口强制 `err_flag.item()` 检查（§8.3，非可选）；L0-P3 锁定 |

技术约束检测清单（逐项）：三维 Kernel ✗ 不涉及（一维 persistent grid，现 kernel 同）；threads ✗
不涉及（C:V=1:2 内建）；动态循环边界 ✗ 不涉及（ntask 为循环界、组检测为分支守卫——静态界+条件
判断合规模式）；L0C 溢出 ✗（8KB/128KB）；GEMM 非整除 ✗（M/N/K 分形全满足，K=576 余数步实证）；
T.copy 列 strided ✗（§4.2 核查）；host 侧 buffer 改动 ✗（§8.3 审计）。

---

## 11.5 同类实现引用（examples/ 具体参考路径与引用点）

| 参考文件 | 引用点 |
|---|---|
| `examples/dsa_sparse_finetune/sparse_mla_topk_reducesum.py` | **主参考（架构真相源）**：persistent 轮询映射与 ntask 推导（L241-243）、gemm_v0 单段 K=576 用法与 kL0Size 实证（L266-269）、cross-flag 槽位/mode-2 quorum 与 DEPTH 纪律（L132-148）、V 侧 gather/相邻对保护全套（L299-574）、#22 尾体竞态注释（L577-585）、GM 标量读 clamp 安全模式（L429-439）、host 入口与 _BUILD_CACHE（L666-765） |
| `examples/attention_sink/example_gqa_sink_fwd_varlen/example_gqa_sink_fwd_varlen.py` | **C-scope GM 标量读先例**（L268-270 `q_seqlens[bz]`，本设计 §6.1 组检测的直接依据）；6-flag init/destroy 配平模式（L98-105, L384-392, L613-622——本设计 F_WSF/F_PDONE 配平同款）；L1 512KB 容量注释（L201）；连续分块 task_range 映射（L85-88，§1.3 备选映射） |
| `examples/sparse_flash_attention/example_sparse_flash_attn_dynamic_shape.py` | 现 kernel docstring 声明的同步结构同构参照（T.Scope("C"/"V") + 手动 cross-flag Expert 范式，L148 起）；panel 变体整体结构继承该同构 |
| `.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-kernel-memory.md` | 容量表（§2，Ascend910B3.ini 出处）、UB↔L1 不可直拷（§1/§3）、T.copy 切片限制（§3）、V 核半分模式（§4） |
| `tilelang/language/ascend.py` | `gemm_v0` BufferRegion 支持与 kL0Size 语义（L418-463）、`set_cross_flag` mode=2 quorum（L191-210） |

---

## 12. 交付清单

| 产物 | 状态 | 说明 |
|---|---|---|
| `examples/dsa_sparse_finetune/DESIGN_panel_mode.md` | ✅ 本文件 | 变体设计（契约/架构/内存/调度/同步/分发/回退/测试/预算/风险） |
| `examples/dsa_sparse_finetune/proto.yaml` | ✅ 同步产出 | 接口规格：dtype 固定组合全集 + attr（含新 panel_mode）——覆盖门禁 `coverage_check.py --proto` 派生 D-DTYPE-*/D-PARAM-* |
| kernel 变体实现 | ⬜ Stage 2 | `tl_sparse_mla_topk_reducesum_panel` + host 分发（同文件新增，通用路径零修改） |
| 测试扩展 | ⬜ Stage 2 | L0-P1..P6 落地（本文件 §9.2）；L0 通过后场景 B 扩展 L1/L2/Boundary |
| 性能验证 | ⬜ Stage 2/3 | graph-replay 主口径复测（golden_metric2_bench.py 同协议），220us 判定 |

[DISPATCH-COVERAGE]
```text
supported_domain: 现 kernel 全域（dtype 固定组合；dim/tail_dim 幂等；topk%BI==0 且 BI 偶≥32；
  head_kv ∈ {16..64 幂等} ∪ {%64==0}；padded_H≠head_kv ⇒ kv_group==1；ΣS/ΣS_kv/batch 运行时任意）
generic_fallback: tl_sparse_mla_topk_reducesum（现 kernel，零修改）
specializations:
  - predicate: panel_mode=="shared"（调用方声明，数据内容性质）
               AND host 门控 G1(L1 容量)/G2(L0B)/G3(L0C)/G4(B×G≤8)/G5(摊销≥16)（纯 metadata：shape+offsets）
               AND 域检查沿用（编译期 assert 同集）
    fallback_on_miss: host 门控/域 miss → generic（静默性能回退，L0-P4e/P6 锁定）；
                      运行时逐任务校验违约 → ErrFlag + host RuntimeError（显式报错，L0-P3 锁定）
    equivalence_evidence: gemm 同形同 K 序 ⇒ logits 逐位一致（§9.1）+ V 侧指令链同构；
                      L0-P1 panel-vs-generic torch.equal 实证
    unsupported_inputs: 无（panel 谓词 ⊂ 现域，不收缩支持域）
  result: pass
```
