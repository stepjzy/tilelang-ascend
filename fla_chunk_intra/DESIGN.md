# fla_chunk_intra 算子设计文档（KDA intra-chunk，fwd 块逆生成 + bwd intra 梯度）

> GPU→Ascend 迁移设计（tilelang-op-orchestrator Stage 1 / tilelang-op-design）。
> 语义权威：`ref_gpu/fla_chunk_intra.py`（Triton/FLA，只读）；代码级起点：`ref_gpu/fla_chunk_intra_tilelang_gpu_fwd.py`、`ref_gpu/fla_chunk_intra_tilelang_gpu_bwd.py`（上游 TileLang CUDA，只读）；测试输入构造语义：`ref_gpu/fla_chunk_intra_cumsum.py`（只读）；计时口径：`ref_gpu/test_utils_kda.py`（只读）。接口/公式/门禁权威：`proto_fla_chunk_intra.yaml`（任务书，只读）。
> 目标硬件：**Ascend 910B3**（20 AI Cores / L2 192MB / HBM 64GB；片上容量按 A2 级：L1 512KB、UB 192KB、L0A/L0B 64KB、L0C 128KB per core；HBM 实测 r+w 带宽 **1259–1273 GB/s**（D2D 大拷贝，取 1263 GB/s）——同族 fla_wy_fast Stage 3 实测标定值）。
> 本文档区分【已确认事实】（源码/示例/同族交付实测验证）、【推断】（标注依据）、【需 Stage 2 验证】三类信息。
> 同族已交付先例（★ 标注引用）：`../tilelang-dev-journal/fla_wy_fast/`（KDA WY 表示，CV 融合 Developer 模式，含工具链缺陷清单与实证 API 链）、`../tilelang-dev-journal/fla_chunk_o/`（KDA 输出 O，CV 融合）、`../tilelang-dev-journal/fla_chunk_intra_token_parallel/`（KDA 对角块生产者，同族 Aqk/Akk 数据结构与 gk 衰减语义）。

---

## 1. 概述

### 1.1 算子名称

`fla_chunk_intra`（含 fwd / bwd 两个 host API，上游名 `chunk_kda_fwd_inter_solve_fused` / `chunk_kda_bwd_intra`）

### 1.2 功能描述

KDA (Kimi Delta Attention) intra-chunk 计算：

- **fwd**：对每个 (chunk, bh)，计算 4×4 sub-chunk 网格的 off-diagonal Aqk/Akk 块（带 per-channel 衰减的 GEMM，Aqk 写出）＋ 4 个对角块的单位下三角前代求逆（消费上游 token_parallel 产出的 `Akk_diag`）＋ 4×4 块双对角递推合并出 64×64 块下三角逆，写出 `Akk`（= 合并逆）。
- **bwd**：由 `dAqk/dAkk` 梯度计算 `dq2/dk2/db2/dg2`（前序块 GEMM + 对角 masked 项 + 后序转置块 GEMM + 对角转置 masked 项）。host 侧后处理（`db2.sum(0)+db`、reverse chunk-local cumsum）不属 kernel、不计入性能计时（上游测试边界，proto 声明）。

本算子为 **CV 融合**（Cube GEMM + Vector exp/掩码/cast/reduce）。

**【已确认事实】Aqk 对角块写出/比较口径（任务点名核实项）**：核查 `ref_gpu/fla_chunk_intra_tilelang_gpu_fwd.py`（L50-65 prepare_input：`Aqk = torch.empty(...)` 未初始化；L522-548 run_test）：Triton 参考实现只写 off-diagonal 块（fla_chunk_intra.py L213-236）；tilelang 版 Aqk 由框架 `out_idx` 新分配、同样只写 off-diagonal 块；`compare_tensors("Aqk", Aqk_ref, Aqk_tilelang)` 做**全量张量比较**——Aqk 对角 4 块与块上三角 6 块为垃圾值 vs 垃圾值，实际通过依赖 CUDA 新分配零页（非确定性口径）。Akk 同理：上游只写 10 个下三角+对角块（L329-349），块上三角 6 块未写出（合并逆数学上该处为 0）。
**本设计口径（确定性化）**：NPU kernel 通过"零填充 staging + 单次整块写出"将 Aqk 对角 4 块 + 块上三角 6 块、Akk 块上三角 6 块显式写 **0**（合并逆的块上三角数学值即 0；Aqk 未写区域与上游零页实际行为一致）；golden 同口径写 0；比对为全量张量、确定性。该决策只补定义未写出区域，不改动任何上游写出块的语义。

**【已确认事实】任务书公式 (F) 记号勘误**：proto formula 的 (F) 写 `dg2 += (dk2_preβ − dkt) ⊙ k + dg`，但 Triton 语义权威（fla_chunk_intra.py L481-482 `b_db = tl.sum(b_dk2 * b_k)` 先于 `b_dk2 *= b_b`，L556 使用 post-β 的 b_dk2）与上游 tilelang bwd 样例（fla_chunk_intra_tilelang_gpu_bwd.py L246-254、L378-384）**两处独立一致**地表明：(F) 使用 **post-β** 的 dk2。本设计与 golden 均按代码语义（post-β）实现；db_partial 仍用 pre-β dk2（两处源码一致）。非 [DESIGN_ERROR]（代码无歧义，判定为任务书笔误）。

### 1.3 数学公式（逐 kernel 逐块，proto formula 展开）

记 chunk CS=BT=64、sub-chunk SCS=BC=16、NC=4、scale=1.0（kernel 内常量乘，proto 声明）；`g = gk` 为 chunk 内累积 log2 衰减（cumsum 输出，fp32）；`2^x = exp2(x)`；块内数学与上游一致按 fp32 累加语义。

**[fwd]**（每 program 处理一个 (chunk, bh)）：

(1) off-diagonal 块（s2 ∈ {1,2,3}，K 维为收缩维，锚点 `g_n(s2) = g[s2·16, :]` 子块首行）：

```
qg_{s2}[r,k]  = q[s2·16+r, k] · 2^(g[s2·16+r,k] − g_n(s2)[k])     # 行衰减
kg_{s2}[r,k]  = k[s2·16+r, k] · 2^(g[s2·16+r,k] − g_n(s2)[k])     # 同衰减（Akk 左操作数）
rhs_{s2}[j,k] = k[j, k] · 2^(g_n(s2)[k] − g[j,k])    (j < 16·s2)  # 列衰减
Aqk[s2,s1]  = scale · Σ_k qg_{s2} · rhs_{s2}ᵀ        （写出 bf16，仅 off-diag）
Akkβ'[s2,s1]= −β[s2行] · Σ_k kg_{s2} · rhs_{s2}ᵀ      （workspace，负号折叠，见 (3)）
```

锚点因式分解 `2^(g_i−g_j) = 2^(g_i−g_n)·2^(g_n−g_j)` 与上游逐元素直接式**代数恒等**；数值域安全：g 单调不增（cumsum(logsigmoid)≤0）⟹ 两个指数均 ≤0（测试 randn 域有界 ±8），无上溢；`2^{−g_j}` 分解（会溢出）被否决，见 §10.4-C。

(2) 对角块前代求逆（逐行 i = 2..15 顺序构造，4 块独立）：`Ai_ss = (I + strict_tril(A_s))⁻¹`，A_s = Akk_diag 对应 [16,16] 块；N_s = −strict_tril(A_s)：

```
row_i = N_s[i, :] + Σ_{j<i} N_s[i,j] · Ai_s[j, :]    （替换 Ai_s 第 i 行；行 0/1 = N_s 行 0/1）
```

(3) 合并块逆（块矩阵形式）：记 `M[s,s] = I + strict_tril(A_s)`、`M[s2,s1] = Akkβ[s2,s1] = −Akkβ'[s2,s1]`（正号 beta 缩放），则 `Ai = M⁻¹` 为块下三角（含对角），且：

```
Ai = (I+G)(I+G²)·D⁻¹,   G = D⁻¹·Akkβ',   D⁻¹ = block_diag(Ai_00..Ai_33)
```

展开与上游逐块公式逐项等价（已代数验证：Ai10 = −(Ai11·Akkβ10)·Ai00 = G[1,0]·Ai00 ✓；Ai20 = −Ai22·(Akkβ20·Ai00 + Akkβ21·Ai10) ✓；Ai21/Ai30/Ai31/Ai32 同理 ✓）。**负号折叠**：kernel 以 `Akkβ' = −Akkβ`（β 缩放时乘 −β）写入 workspace，使 `G = D⁻¹·Akkβ'` 为单次 GEMM、链上零取负向量操作（每个 Akkβ 块在公式中恰出现一次、单项取负可分配 ✓）。恒等式 `(I−G)⁻¹ = I+G+G²+G³ = (I+G)(I+G²)`（G 块严格下三角 ⟹ G⁴=0）。

(4) 写出：Aqk（off-diag × scale，bf16，其余区域 0，§1.2 口径）；Akk = Ai（bf16，块上三角为计算精确 0，无需零填充）。

**[bwd]**（本设计每 program 处理一个 (K-half, chunk, bh)，**一次覆盖全部 4 个 sub-chunk**，§5.2）：统一记号 `MdA = dAqk ⊙ Mlo`、`MdK = dAkk ⊙ Mlo`，`Mlo[a,b] = (a ≥ b)`（含对角下三角指示，[64,64] 常量）；锚点 `gn_s = g[s·16, :]`（子块 s 首行）、`gn'_s = g[(s+1)·16−1, :]`（子块 s 末行）：

```
Kdec_s[j,k]  = k[j,k] · 2^(gn_s[k] − g[j,k])       # (A)+(B) 统一右操作数
Qdec_s[j,k]  = q[j,k] · 2^(gn'_s[k] − g[j,k])      # (D)+(E) 统一右操作数
KBdec_s[j,k] = (β[j]·k[j,k]) · 2^(gn'_s[k] − g[j,k])
D_s[r,k]     = 2^(g[s·16+r,k] − gn_s[k])           # (A)+(B) 行后置缩放
V_s[r,k]     = 2^(gn'_s[k] − g[s·16+r,k])          # (D)+(E) 行后置缩放
```

(A)+(B) 合并为 masked GEMM（s = 0..3，行块 s；`KMAX = 16·max(s+1,2)` = 32/32/48/64）：

```
dq2_pre_s = D_s ⊙ ( MdA[s·16:+16, 0:KMAX] @ Kdec_s[0:KMAX] )     # k≥32 分形满足
dk2preβ_s = 同式换 MdK
```

列范围 `[0:16(s+1)]` 中：prev 块列（< s·16）无需掩码（r 块号 > j 块号自动通过 Mlo）；对角块列由 Mlo 掩码 (r ≥ j)——即上游 (B) 的 `where(i ≥ j)`；s=0 时 k=16 借 MdA 块 (0,1) 的 Mlo 精确零列自然补零到 k=32（分形 K≥32）。**逐列顺序依赖消除依据（任务点名难点）**：Σ_j (i≥j)·dA[i,j]·k_j·2^{g_i−g_j} = 2^{g_i−gn}·Σ_j (i≥j)·dA[i,j]·(k_j·2^{gn−g_j})——掩码只依赖 (i,j)、不依赖迭代状态，"顺序累加"实为独立项求和，可整体 GEMM 化（恒等式精确；锚点域安全同上）。

(D)+(E) 合并为转置 masked GEMM：

```
dkt_s = V_s ⊙ ( MdA[:, s·16:+16]ᵀ @ Qdec_s + MdK[:, s·16:+16]ᵀ @ KBdec_s )   # 2 gemm 累加, k=64
```

A 操作数取 MdA 的**列切片** [64(j), 16(i)]，`transpose_A=True` 后掩码语义 (j ≥ i) 恰为同一 Mlo（块号 < s 的 j 行被 Mlo 自动清零 ✓）。(E) 的逐列依赖消除与 (B) 对称：`2^{g_j−g_i} = 2^{g_j−gn'}·2^{gn'−g_i}`，掩码 (i ≤ j) 只依赖 (i,j)。

尾段（s 逐块，V 侧；(F) 按 §1.2 勘误为 post-β）：

```
db_s    = rowsum_k(dk2preβ_s ⊙ k[s·16:+16, :])                     # pre-β（上游 L481）
dk2β_s  = dk2preβ_s ⊙ β[s·16:+16 行广播]
dq2_s   = D_s⊙(A+B 结果) + dq[s·16:+16, :]                          # (C)
dg2_s   = q_s ⊙ (A+B 结果) + (dk2β_s − dkt_s) ⊙ k_s + dg[s·16:+16, :]
dk2_s   = dk2β_s + dk[s·16:+16, :] + dkt_s
```

输出：dq2/dk2/dg2（[B,S,H,DK]）、db2（[NK,B,S,H] partial）。**host 后处理（非 kernel、不计时）**：`db_out = db2.sum(0) + db`（out-of-place 返回新张量，避免改写输入 db buffer；语义值与上游 in-place `add_` 一致）；`dg_out = reverse chunk-local cumsum(dg2)`（`view(B,NT,64,H,DK) → flip(2) → cumsum(2) → flip(2)`，与 golden 逐位同构）。

### 1.4 算法描述（kernel 拆分，6 个 JIT kernel）

**⚠️ 工具链强制的 kernel 拆分（★fla_wy_fast 先例）**：上游 2 kernel 结构在昇腾 Developer 模式不可编译——(a) fwd 合并逆是 6 级串行 GEMM 链，每级间需 frag→UB→L1 的 V 段中转，形成 V→C→V→C→V 多跳直线交替，触发 sync-point 计数失配编译错（★final_summary §8-4，z+da 合并复现）；(b) serial 循环内 V→C 交换 mis-sync（★模块 docstring-1）；(c) 每 kernel 仅支持单个 cross-core Pipelined 体（★§8-2）。**实证安全包络：每 kernel ≤1 个直线 C 段（z/da 形）或 Pipelined 体 + 相邻 C epilogue（fla_chunk_o 形）**。故 fwd 拆 5、bwd 1，跨 kernel 中转走 GM workspace（★"cross-kernel accumulators are handed over through GM" 同款）：

| kernel | grid | 结构 | 职责 |
|---|---|---|---|
| `fwd_k_inter` | NT·B·H | 直线 V→C→V（1 C 段；BK=DK 无循环） | off-diag Aqk/Akkβ' GEMM；写 Aqk（bf16 终态）+ Akkβ' ws（fp32 紧凑 6 块） |
| `fwd_k_g` | NT·B·H | 直线 V→C→V | 4 块前代求逆 → AiDiag；G = AiDiag·Akkβ'；写 G ws（6 块）+ AiDiag ws（4 对角块） |
| `fwd_k_g2` | NT·B·H | 直线 V→C→V | G² = G·G；写 G² ws（4 块） |
| `fwd_k_y` | NT·B·H | 直线 V→C→V | Y = (I+G²)·AiDiag；写 Y ws（整块 [64,64]） |
| `fwd_k_ai` | NT·B·H | 直线 V→C→V | Ai = (I+G)·Y；写 Akk 输出（bf16 单次整块） |
| `bwd_k_intra` | NK·NT·B·H | Pipelined(2) 体 + 尾段（k1 形） | §1.3 (A)~(F) 全部；16 逻辑 GEMM（×3 hi/lo = 48 gemm_v0） |

### 1.5 数据流图（fwd 主链）

```
GM[q,k:bf16] ─T.copy(同dtype)→ UB[bf16] ─T.copy(cast)→ UB[fp32] ─┐
GM[gk:fp32] ─T.copy(同dtype)→ UB[fp32] ─────────────────────────┤→ UB: qg/kg'/rhs (fp32 锚点衰减)
GM[beta] flat-view span → tile.gather 链 → UB[β fp32] ───────────┘       │ bf16 hi/lo 分裂 → L1
                                                       L0C[Aqk_s2 / Akkβ'_s2 frags] ← gemm_v0 ×3 hi/lo 累加
Aqk:    L0C → UB(fp32) → ×scale → [64,64] bf16 零填充 staging → GM（单次整块）
Akkβ':  L0C → UB(fp32) → 3 个 [16,16/32/48] ws 切片写
GM[Akkβ' ws] → staging(零填充+6块) → L1 ┐
GM[Akk_diag] → UB → 前代求逆(纯V,14步×4块) → AiDiag → L1 ┤→ gemm → G ws → G·G → G² ws
                                                        └→ (I+G²)·AiDiag → Y ws → (I+G)·Y → Akk(bf16) GM
```

bwd 数据流同构（全 fp32，fp16 hi/lo 分裂，见 §4.4）。

### 1.6 范围与分派覆盖审计

【已确认事实】范围裁剪（proto 声明）：定长序列、scale=1.0（kernel 内常量乘保留）、chunk_size=64、sub_chunk_size=16；varlen（cu_seqlens/chunk_indices）与 scale≠1 透传不在范围。DK 域由 exact-2 Pipelined / 分形 / UB 约束收紧。

```text
[DISPATCH-COVERAGE]
supported_domain: S%64==0; DK ∈ {64, 96, 128}（bwd exact-2 BK=DK//2 ∈ {32,48,64} 满足分形 K≥32 + UB 预算；
  fwd BK=DK 整块 64/96/128，★fla_wy_fast _pick_block_d 同款语义）; B,H ≥ 1 任意;
  chunk_size=64, sub_chunk_size=16 固定; fwd dtype=(bf16 q/k/beta + fp32 gk/Akk_diag 输入,
  bf16 Aqk/Akk 输出); bwd dtype=(全 fp32 输入, fp32 dq2/dk2/db2/dg2 输出); scale=1.0 常量乘
generic_fallback: fwd_k_*×5 / bwd_k_intra×1 单一通用路径（无 dtype/shape 特化分支；BK 由
  _pick_bk(DK) 机械推导，属性能参数非分派谓词）
specializations: 无
unsupported_inputs: varlen、S%64≠0、chunk_size≠64、DK∉{64,96,128}（DK=32: bwd BK=16 违反 L0A/L0B
  K≥32 分形——★fla_chunk_o documented limitation 同款）、fwd q/k fp32 输入变体（proto dtype 表
  含 fp32 但上游 tilelang fwd 测试样例与 benchmark 均为 bf16 主路径；fp32 变体属 L1 扩展，
  本期不实现、host 断言拒绝）——均与 supported_domain 不冲突（proto 显式排除或 benchmark 主路径声明）
result: pass
```

---

## 2. 编程模式选型

### 2.1 模式结论

**Developer 模式**（用户明确指定，2026-11-02）——`alloc_shared` / `alloc_fragment` + 自动同步 + `threads=1`，无显式 `T.Scope`、无手动 barrier（bwd db2 scatter 处的 `T.set_flag/wait_flag("MTE3","MTE2",0)` 为 ★fla_wy_fast 已交付验证的 MTE3→MTE2 显式链，属计算源 scatter 缺陷规避而非模式切换，见 §7.2）。

### 2.2 选型理由

- 算子是典型 CV 融合（Cube GEMM + Vector exp/掩码/cast/reduce），无跨核自定义同步需求；上游 TileLang CUDA 版本身即 Developer 风格 API（alloc_shared/fragment + T.gemm/T.copy），迁移映射最直接。
- 同族 ★fla_wy_fast / fla_chunk_o 已在 910B3 以 Developer 模式（threads=1）完整交付，全部实证 API 链（含工具链缺陷规避）可直接复用。
- threads=2 有 ★final_summary §7-8 gather 链 vid 切分越界先例；本算子 bwd 含 tile.gather（beta 载入），主选保守值 threads=1；threads=2 列为 Stage 3 调优候选。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（编译器映射 L1/UB）+ `T.alloc_fragment`（映射 L0C） |
| GEMM | `T.gemm_v0(A, B, C, transpose_A/B=, init=)`：A/B 为 L1(shared)（支持 BufferRegion），C 为 fragment(L0C) |
| 计算方式 | `T.tile.mul/sub/add/exp/fill/broadcast/gather/arith_progression`（buffer 级全 tile 指令）+ `T.reduce_sum`（限 UB） |
| 作用域 | 编译器自动 C/V 分离（AUTO_CV_COMBINE） |
| 同步方式 | 自动同步（AUTO_SYNC + AUTO_CV_SYNC），无手动 barrier/cross_flag |

---

## 3. API 映射设计

### 3.1 公式拆解（fwd）

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| F1 | `qg/kg'/rhs = src ⊙ 2^(±(g−锚点))` | UB fp32 域：tile.sub + tile.mul(LN2) + tile.exp + tile.mul（exp2 无后端，★§3.5.2-3） |
| F2 | `Aqk[s2,s1] / Akkβ'[s2,s1]` | gemm_v0 ×3 hi/lo（bf16 分裂），transpose_B=True，m16 k=DK n=16s2 |
| F3 | `Aqk ×scale → bf16 GM`（零填充 staging 单次写） | L0C→UB→cast→GM 三步链 |
| F4 | `Akkβ' → fp32 GM ws`（3 切片写） | 同上（无 cast） |
| F5 | 前代求逆（§1.3(2)） | 纯 V：行切片 copy + tile.broadcast + tile.mul + reduce_sum(dim=0) + tile.add + 行写 |
| F6 | `G / G² / Y / Ai`（4 kernel 各 1 gemm） | gemm_v0 ×3 hi/lo，[64,64]@[64,64] |
| F7 | `Akk → bf16 GM` 单次整块 | 块上三角计算精确 0，无需零填充 |

### 3.2 公式拆解（bwd）

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| B1 | `MdA = dAqk ⊙ Mlo`、`MdK = dAkk ⊙ Mlo` | 常量 Mlo + tile.mul（fp32 域） |
| B2 | `Kdec/Qdec/KBdec/D/V`（§1.3） | UB fp32：sub(锚点广播)+mul(LN2)+exp+mul；β 经 gather 链 |
| B3 | `(A)+(B)`：dq2_pre_s / dk2preβ_s | gemm m16 k∈{32,32,48,64} n=BK，×3 fp16 hi/lo，行切片 A 操作数 |
| B4 | `(D)+(E)`：dkt_s | gemm MdA[:, s·16:+16]ᵀ@Qdec_s（transpose_A=True, k=64）+ MdK 同款累加 |
| B5 | 尾段（db/dq2/dg2/dk2） | tile.mul + reduce_sum(dim=1) + tile.add；db2 经 scratch+flag 链 scatter |
| B6 | host 后处理 | db2.sum(0)+db（out-of-place）、reverse chunk cumsum（不计时） |

### 3.3 TileLang API 映射（含出处，全部本项目实证）

| 步骤 | TileLang API（昇腾） | 参数 | 出处（已验证） |
|------|---------------------|------|---------------|
| Kernel | `T.Kernel(GRID, threads=1, is_npu=True) as (cid)` | fwd：bs=cid%NT, bbh=cid//NT；bwd：i_k=cid%NK, i_t=(cid//NK)%NT, i_bh=cid//(NK·NT)（i_k 最内 → 同 (t,bh) 两 K-half 相邻，L2 命中） | ★impl_final.py:130；gdn_chunk_o.py:35 |
| GEMM | `T.gemm_v0(A, B, C, transpose_A=, transpose_B=, init=)` | A/B→L1 shared（**支持 BufferRegion**），C→fragment fp32；init=True 清零 / False 累加（hi/lo 3 连击） | `tilelang/language/ascend.py:418-427`（签名实证）；★impl_final.py:181-182, 395-403；gdn_wy_fast.py:83 |
| 内存分配 | `T.alloc_shared(shape, dtype)` / `T.alloc_fragment(shape, dtype)` | shared→L1/UB、fragment→L0C | ★impl_final.py:137-154 |
| 逐元素 | `T.tile.mul(dst, src0, src1/标量)`、`T.tile.sub`、`T.tile.add`、`T.tile.exp(dst, src)` | fp32 域全 tile 指令 | ascend_tile.py:1152(mul)/1277(exp)；★impl_final.py:262-273 |
| 清零/单位 | `T.tile.fill(buf, 0.0)`；I 矩阵 = 4 个 [16,16] 对角块 `tile.fill(region, 1.0)` | staging 零填充；免 Mid 常量 | ascend_tile.py:352-376 |
| 广播 | `T.tile.broadcast(dst, src, axis=0/1)` | 行向量 [1,BK]→[64,BK]（axis=0 锚点）/ 列向量 [·,1]→[·,BK]（axis=1 β 行） | ascend_tile.py:2507；★impl_final.py:172（axis=1）+ wkg（axis=0，★docstring-6） |
| 归约 | `T.reduce_sum(buf_ub, out_ub, dim=0/1, clear=True)` | **限 UB**；2D 支持 dim 0/1/-1/-2（前代求逆 dim=0 / bwd db dim=1）；循环携带累加 = clear=True + tile.add（clear=False in-loop 为工具链 bug ★docstring-6） | reduce_ascend.py:456-494；★impl_final.py:771-772 |
| β 载入 | `T.copy(BetaFlat[bb, s0·H:(s0+64)·H], beta_tile)` + `T.tile.arith_progression(off, bh·4, H·4)` + `T.reinterpretcast(off_u32, off_i32, "uint32_t")` + `T.tile.gather(bt, beta_tile, off_u32, 0)` | [B,S,H] sub-32B strided DMA 规避（fp32 变体；fwd bf16 用 ×2 步长、48 元素） | ★impl_final.py:156-160（bf16）/356-358（fp32） |
| hi/lo 分裂 | `T.copy(x_f32→x_hi)` → `T.copy(x_hi→x_hi_f32)` → `T.tile.sub(x_lo_f32, x_f32, x_hi_f32)` → `T.copy(x_lo_f32→x_lo)` → L1 | fwd：bf16 hi/lo 3-gemm（两操作数均非精确 bf16）；bwd：fp16 hi/lo 3-gemm（fp32 门禁） | ★impl_final.py:175-180（bf16）/380-392（fp16）逐字模式 |
| db2 scatter | `T.copy(db64, db2flat[cid, 0:64])` → `T.set_flag("MTE3","MTE2",0)` → `T.wait_flag("MTE3","MTE2",0)` → `T.copy(db2flat_v[cid,0:64,0:1], db2d)` → `T.copy(db2d, db2[i_k,bb,s0:s0+64,bh])` | 计算源 [64] scatter 到 strided GM 的已验证链（scratch 暂存 + flag + [64,1] 视图重载） | ★impl_final.py:774-782（verified R3 chain） |
| 搬运/cast | `T.copy(GM, shared 同dtype)` + `T.copy(ub_bf16, ub_fp32)` 两步；`T.copy(frag, ub_fp32)` L0C→UB 直连 | **禁止 GM↔UB 跨 dtype 直拷**（★§8-6）；L0C→GM 跨 dtype 直存损坏（★§8-7）走三步链 | ★impl_final.py:157-160, 183-188 |
| 掩码 | 常量 `Mlo[64,64]`（tril(ones,0)）/ `Mneg[16,16]`（−tril(ones,−1)）GM 输入 → UB → `T.tile.mul` | T.Parallel 内 if_then_else 不可用（SIMD 无分支） | ★impl_final.py:913-914；gdn_chunk_o.py:123 |
| 循环 | bwd `for i_kk in T.Pipelined(DK // BK, num_stages=2)`（**恰好 2 迭代**） | extent=1 崩溃 / extent≥3 回卷损坏（★§8-2）；体 = V-prep + C gemm + V 尾段（k1 形） | ★impl_final.py:162, 251 |
| 前代求逆行写 | `T.copy(a_new, Ai_s[i, 0:16])`（符号行切片写） | 【需 Stage 2 验证】R4（两级回退见 §10.1） | 符号切片读先例 ★token_parallel §3.3（beta_f[i, 0:BH_V]） |

### 3.4 计算伪代码（关键主体，结构示意）

```python
LN2 = 0.6931471805599453   # 2^x = exp(x*ln2)，exp2 无昇腾后端（★§3.5.2-5）

# ---- fwd_k_inter（BK=DK 整块，直线 V→C→V）----
@tilelang.jit(out_idx=[...], pass_configs=PASS_CONFIGS)
def fla_chunk_intra_fwd_k_inter(B, S, H, DK, chunk_size=64, scale=1.0):
    BS, NT, BK = 64, S // 64, DK
    @T.prim_func
    def main(Q, K, GK, BetaFlat, Aqk, Akkbp):        # dtype 见 §4.1/4.2
        with T.Kernel(NT * B * H, threads=1, is_npu=True) as (cid):
            bs, bbh = cid % NT, cid // NT; bb, bh = bbh // H, bbh % H; s0 = bs * BS
            # buffer：k_half[64,BK]bf16 / k_f,g_f[64,BK]fp32 / q_half[48,BK]bf16 / q_f[48,BK]fp32
            #         rhs_f[48,BK] / lhs_f[16,BK]（复用）/ gn[1,BK] / 锚点 2D / β 链 / hi-lo 中转
            #         L1：qg_hi/lo, kg'_hi/lo [16,BK]×3s2, rhs_hi/lo [16s2,BK]×3；L0C：6 frags [16,16/32/48]
            T.copy(K[bb, s0:s0+64, bh, :], k_half); T.copy(k_half, k_f)
            T.copy(GK[bb, s0:s0+64, bh, :], g_f)
            T.copy(Q[bb, s0+16:s0+64, bh, :], q_half); T.copy(q_half, q_f)
            # beta gather 链（bf16 变体，偏移 = (s0+16)*H + bh 起 48 元素、步长 H）→ bet
            for s2 in range(1, 4):                     # 追踪期静态展开（range 展开为 3 份直线代码）
                r0 = s2 * 16
                T.copy(g_f[r0, 0:BK], gn)                              # 锚点 g_n(s2)
                T.tile.sub(e, g_f[r0:r0+16, :], gn)                     # g − gn（列广播语义，见 R7）
                T.tile.mul(e, e, LN2); T.tile.exp(e, e)                 # 2^(g−gn)
                T.tile.mul(qg_f, q_f[r0-16:r0+32, :], e)                # qg_{s2}
                T.tile.mul(kg_f, k_f[r0:r0+16, :], e)
                T.tile.mul(bcol, bet[s2 局部切片], …)                   # kg' = kg ⊙ (−β) 行广播
                # rhs_{s2} = K[0:16s2] ⊙ exp((gn − g)·ln2)：同链对 g_f[0:16s2] 反向
                # hi/lo bf16 分裂 ×(qg, kg', rhs) → L1
                T.gemm_v0(qg_hi, rhs_hi, Aqk_s2, transpose_B=True, init=True)
                T.gemm_v0(qg_hi, rhs_lo, Aqk_s2, transpose_B=True, init=False)
                T.gemm_v0(qg_lo, rhs_hi, Aqk_s2, transpose_B=True, init=False)
                T.gemm_v0(kgp_hi, rhs_hi, Akkb_s2, transpose_B=True, init=True)   # Akkβ'（−β 已折叠）
                ... ×2 init=False
            # epilogue V：Aqk_s2 → UB(fp32) → ×scale → [64,64] bf16 零填充 staging
            #   （tile.fill 0 + 3 个 [16,16s2] 放置）→ 单次 GM 写
            #   Akkb_s2 → UB(fp32) → 3 个 [16,16/32/48] ws 切片写
```

```python
# ---- fwd_k_g（前代求逆 + G；直线 V→C→V；k_g2/k_y/k_ai 同构略）----
            T.copy(Akk_diag[bb, s0:s0+64, bh, :], ad_f)                 # [64,16] fp32
            T.copy(Mneg[0:16, 0:16], mneg)                              # 常量
            for s in range(4):                                          # 静态展开
                r0 = s * 16
                T.copy(ad_f[r0:r0+16, :], Ad_s)
                T.tile.mul(Ai_s, Ad_s, mneg)                            # N_s = −strict_tril(A_s)
                for i in T.serial(2, 16):                               # 纯 V 串行（无 C 交换，安全）
                    T.copy(Ai_s[i, 0:16], a_row)                         # = N_s[i,:]（已含掩码零）
                    T.copy(a_row, acol[:, 0]); T.tile.broadcast(ab, acol)
                    T.tile.mul(tmp, ab, Ai_s)                           # a_row[j]·Ai_s[j,c]
                    T.reduce_sum(tmp, red, dim=0, clear=True)           # Σ_j → [16]
                    T.tile.add(a_new, a_row, red)
                    T.copy(a_new, Ai_s[i, 0:16])                        # 行写（R4，回退见 §10.1）
                T.copy(Ai_s, AiDiag[r0:r0+16, r0:r0+16])                # 块对角放置
            # AiDiag staging 零填充 + 4 对角块 1 填充（tile.fill(region,1.0)×4）
            # Akkβ' ws 6 切片 → staging(零填充+放置) → hi/lo bf16 ×2 份（A/B 分开，R3 规避）
            T.gemm_v0(aid_hi, akbp_hi, G_frag, init=True)                # G = AiDiag·Akkβ'
            ... ×2 init=False
            # epilogue：G_frag → UB(fp32) → 6 个 [16,16] ws 切片写；AiDiag → 4 对角 ws 切片写
```

```python
# ---- bwd_k_intra（Pipelined(DK//BK=2)；体 = V-prep + C(48 gemm) + V 尾段）----
@tilelang.jit(out_idx=[...], pass_configs=PASS_CONFIGS)
def fla_chunk_intra_bwd_k_intra(B, S, H, DK, chunk_size=64):
    BS, NT, BK = 64, S // 64, DK // 2; NK = DK // BK                     # 恒 2
    @T.prim_func
    def main(Q, K, G, BetaFlat, Mlo, dAqk, dAkk, dq, dk, dg,
             dq2, dk2, db2, dg2, db2flat, db2flat_v):
        with T.Kernel(NK * NT * B * H, threads=1, is_npu=True) as (cid):
            i_k = cid % NK; i_t = (cid // NK) % NT; i_bh = cid // (NK * NT)
            # prologue：Mlo 载入；beta gather（fp32 变体）→ bet
            for i_kk in T.Pipelined(DK // BK, num_stages=2):
                # V-prep：K/G/Q [64,BK] fp32 直载；dAqk/dAkk [64,64] fp32 载入（载入双缓冲）
                #   MdA/MdK = dAqk/dAkk ⊙ Mlo（幂等重做）；锚点 gn_s/gn'_s 8 行切片
                #   Kdec_s/Qdec_s/KBdec_s ×4（sub+broadcast+mul+exp+mul）
                #   MdA/MdK/Kdec/Qdec/KBdec hi/lo fp16 分裂 → L1
                for s in range(4):                                      # 静态展开
                    KMAX = 16 * max(s + 1, 2)                           # 32/32/48/64
                    # dq2_pre_s / dk2preβ_s（×3 hi/lo，init=(i_kk==0) 跨 half 累加）：
                    T.gemm_v0(MdA_l1[s*16:s*16+16, 0:KMAX], Kdec_s_l1[0:KMAX],
                              dq2_s_frag[s], init=(i_kk == 0))
                    ... ×2 hi/lo init=False；同款 MdK → dk2pre_s_frag[s]
                    # dkt_s（×3 hi/lo ×2 组，第 2 组 init=False 累加）：
                    T.gemm_v0(MdA_l1[:, s*16:s*16+16], Qdec_s_l1,
                              dkt_s_frag[s], transpose_A=True, init=True)
                    T.gemm_v0(MdK_l1[:, s*16:s*16+16], KBdec_s_l1,
                              dkt_s_frag[s], transpose_A=True, init=False)
                    ... ×2 hi/lo（各逻辑 gemm 3 连击）
                # V 尾段（体内，按 half 输出切片写——half 间输出列不相交，无循环携带累加）：
                #   D_s/V_s 计算（g 行块 ± 锚点）；per s：dq2_s 片 → +dq 片 → 写
                #   dk2preβ_s → db_s=rowsum(⊙k 片)（db_acc 跨 half 用 clear=False 陷阱规避：
                #     见 §6.2）→ ×β 行 → dg2_s / dk2_s 组合 → 写
            # db2 scatter：db_acc [64] → db2flat[cid] → set/wait_flag → db2flat_v 重载 → db2 GM 切片写
```

> 伪代码为结构示意（buffer 声明/部分链路省略），Stage 2 按 §3.3 表逐 API 落地；`for s in range(...)` 为追踪期静态展开（★token_parallel §3.3 range+静态 guard 同款），不产生运行时循环。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（强制检测）

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **是**（上游 fwd 2 维 / bwd 3 维 `(NK·NC, NT, B·H)`） | 全部一维化：fwd `NT·B·H`；bwd `NK·NT·B·H`（i_k 最内 → 同 (t,bh) 两 K-half 相邻，L2 命中）；上游 NC 维（i_i）被"每 program 覆盖 4 sub-chunk"消解（§5.2） |
| threads 仅 1/2 | **是**（上游 32/128） | `threads=1`（★交付同款） |
| 动态循环边界 | 否 | S/DK 均 JIT 编译期常量；bwd `DK//BK` 恒 2（exact-2 构造）；前代求逆 `T.serial(2,16)` 静态界；上游 `min(BC, T−i_ti)` 类动态界被 S%64==0 域约束消除 |
| Pipelined extent 约束 ⚠️ | **是**（关键） | 唯一安全 extent=2（★§8-2）：bwd `_pick_bk(DK)=DK//2`（64→32 / 96→48 / 128→64）；fwd k_inter 无循环（BK=DK 直线）。DK=32 无合法 exact-2 block → 不支持（documented limitation） |
| L0C 溢出（128KB） | 已核算 | fwd k_inter 6 frags ≈ 6KB；k_g/k_g2/k_y/k_ai 各 1×[64,64]fp32=16KB；bwd 12 frags [16,BK=64]fp32=48KB —— 全部 ≤ 128KB ✓（§4.5） |
| T.copy 列方向 strided 切片 | 已规避 | 全部 GM 切片为"标量前缀 + 行区间 × 最内维连续 block"；UB/L1 内 [64,16] 列切片（行 64B ≥ 32B，行间 gap 合法）；db2 scatter 走 §3.3 已验证链 |
| GEMM 分形限制 | 已核算 | bf16/fp16：M≥16, K≥32(L0A/L0B), N≥16（fla_chunk_o §3.5.1 口径）。fwd m16 k=DK n∈{16,32,48} ✓；solve [64,64,64] ✓；bwd m16 k∈{32,32,48,64} / k=64, n=BK∈{32,48,64} ✓——k=16 场景经自然零列补齐到 32（§1.3） |
| GEMM 非整除 | 低风险 | S%64==0（域约束）；DK 被 BK 整除（fwd BK=DK；bwd exact-2）；域外 host `_check_shape` 断言拒绝（L2 负向覆盖） |
| **fp32 gemm_v0 损坏** ⚠️（★§8-1：fp32 A/B 产出 1e37 级交替垃圾） | **是**（全部 GEMM 操作数为 fp32 计算值） | 全部 GEMM 走 hi/lo 分裂多 gemm 累加：fwd bf16 hi/lo 3-gemm（两操作数均非精确 bf16，2-gemm 不足，§10.4-E）~2^-17 ≪ bf16 门禁 2^-6；bwd fp16 hi/lo 3-gemm ~2^-22 ≪ fp32 门禁 2^-10（★交付验证模式；fp16 域限制见 §10.3） |
| **GM↔UB cast-on-copy 缺陷** ⚠️（★§8-6） | 已规避 | 跨 dtype 只在 UB↔UB 两步（fwd q/k/beta 载入链；输出三步链） |
| **L0C(fp32)→GM(bf16) 直存损坏** ⚠️（★§8-7） | 已规避 | 输出三步链 frag→UB(fp32)→cast→GM |
| **serial 循环内 V→C 交换 mis-sync**（★docstring-1） | 已规避 | 前代求逆 T.serial 为**纯 V**；V→C 交换全部在 Pipelined 体内或直线单 C 段前 |
| **prologue V→C 交换错版本化**（★§8-5） | 已规避 | bwd MdA/MdK 的 hi/lo 分裂在循环体内幂等重复（每 half 重做，~10 ops） |
| host 侧 Buffer 约束（ascend-constraints §5） | 已审计 | §9.6：零物理化（BetaFlat 零拷贝 view；db 后处理 out-of-place） |

#### 3.5.2 上游 CUDA 专属机制的昇腾适配（逐项）

| # | CUDA 机制（上游） | 昇腾适配结论 | 依据 |
|---|------------------|-------------|------|
| 1 | fwd 2 维 / bwd 3 维 Kernel | 一维化 + cid 分解（NC 维消解） | ascend-constraints §1；★impl_final |
| 2 | threads=32/128 + autotune | threads=1；**L0 不启用 autotune**（fwd BK=DK 由 UB 预算唯一化、bwd BK=DK//2 由 exact-2 唯一化、num_stages=2，搜索空间全消解）；Stage 3 仅剩 threads∈{1,2} 与 §10.4 优化项 | ★§8-2；fla_chunk_o §3.5.2-2 |
| 3 | `T.exp2` | `T.tile.mul(LN2)` + `T.tile.exp`；fp32 下相对差 ~1e-7 ≪ 门禁 | ★impl_final.py:262-263 |
| 4 | `T.gemm(shared, shared, fragment, transpose, clear_accum)` | `T.gemm_v0(..., init=)` + fp32 操作数 hi/lo ×3 | ascend.py:418；★impl_final |
| 5 | `T.if_then_else` 掩码/行选择 | 常量 mask（Mlo/Mneg）+ `T.tile.mul`；**m_tc 行掩码被 S%64==0 域约束消除**；行选择用符号行切片 T.copy（R4） | ★impl_final.py:913-914；api-compute §3.1 |
| 6 | `T.reduce_sum(frag→frag)` | frag→UB 后 UB reduce；循环携带 = clear=True + tile.add | reduce_ascend.py:456-494 |
| 7 | `T.clear(fragment)` | gemm `init=` 参数化 | gdn_chunk_o.py:67 |
| 8 | `T.Pipelined(loops)` 动态边界（bwd j<i_i / NC_actual） | **结构级消除**：(A)/(B) 与 (D)/(E) 逐列 j 循环被 masked-GEMM 重表述消除（§1.3，掩码不依赖迭代状态）；s 循环静态展开 | 本设计 §1.3 |
| 9 | 逐列标量 GM 载入（p_kj/p_gkj/p_bj 行指针步进） | 整块 [64,BK] 载入 + 锚点行切片（消除 16 次小行载入/段） | ★impl_final 整块模式 |
| 10 | `tl.debug_barrier()` | 删除（AUTO_SYNC 兜底；CUDA warp 同步专属） | — |

#### 3.5.3 参考实现差异说明（GPU → Ascend 汇总）

| 差异项 | 参考实现（CUDA/上游） | 本项目（Ascend 910B3） | 转换方案 |
|--------|---------------------|----------------------|---------|
| Kernel 数 | 2 | **6**（fwd 5 + bwd 1；工具链 C/V 多跳限制强制，★z+da 先例） | §1.4 / §10.4-B |
| Kernel 维度 | 2D / 3D | 一维 + cid 分解 | §3.5.2-1 |
| threads | 32/128 | 1 | §3.5.2-2 |
| GEMM 精度 | fp32 累加 + tf32 操作数（Triton SOLVE_TRIL_DOT_PRECISION） | fp32 累加 + **bf16/fp16 hi/lo 分裂操作数**（fp32 gemm 损坏规避） | §3.5.1；§9.3 |
| bwd 逐列循环 | 16 次顺序 j 迭代（B/E 两处循环携带） | masked GEMM | §1.3（任务点名难点） |
| fwd 前代求解 | 逐行 i 顺序（寄存器内标量行） | 纯 V 逐行向量化 | §1.3(2)（任务点名难点） |
| exp2 | tl.exp2 | tile.exp(x·ln2) | §3.5.2-3 |
| 输出未写区域 | 垃圾值（torch.empty + 全量比较偶然通过） | 显式写 0（零填充 staging） | §1.2 |
| 衰减因式 | fwd 锚点式；bwd (A)/(D) 锚点式、(B)/(E) 直接式 | 全部统一锚点式（(B)/(E) 重述为锚点 + 后置行缩放） | §1.3 |

#### 3.5.4 本项目同类实现参考（强制步骤 0 结果）

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `../tilelang-dev-journal/fla_wy_fast/impl_final.py`（★交付源） | **最高（同族同沙箱）** | hi/lo 分裂链、threads=1、一维化、Pipelined extent-2 体、Msk 常量掩码、beta gather 链、dbeta scatter flag 链、输出三步链、缺陷规避 1-10 全集 |
| `../tilelang-dev-journal/fla_chunk_o/DESIGN.md` + `examples/linear_attention_and_rnn/gdn/gdn_chunk_o.py` | 高 | tril(ones,0) 常量、gemm init=(i==0)、epilogue 相邻 C 段、extent-2 论证 |
| `../tilelang-dev-journal/fla_chunk_intra_token_parallel/DESIGN.md` | 高（**同族 Akk_diag 生产者**） | gk 衰减语义/域安全论证、静态展开 + guard、L0 计划结构、Mneg 类常量 |
| `examples/developer_mode/flash_attn_bshd_developer.py` | 中 | Developer CV 骨架（cast 链/UB reduce/pass_configs） |
| `../tilelang-dev-journal/fla_wy_fast/perf_tuning/fla_wy_fast_final_summary.md` | 标定 | 910B3 实测带宽 1263GB/s、单 kernel 逻辑带宽上限 81%、JIT launch ~1ms、缺陷清单 §8 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量（布局 bshd 连续，kernel 直读 GM）

| 参数名 | Shape | dtype（fwd） | dtype（bwd） | 说明 |
|--------|-------|-------------|--------------|------|
| q / k | [B, S, H, DK] | bfloat16 | float32 | |
| gk | [B, S, H, DK] | **float32** | float32 | chunk 内累积 log2 衰减（fwd 测试构造需 cumsum；bwd 测试 randn 不 cumsum——上游口径） |
| beta | [B, S, H] | bfloat16 | float32 | kernel 内经 `BetaFlat` 零拷贝 flat view (B, S·H) 消费（§3.3 gather 链） |
| Akk_diag | [B, S, H, 16] | **float32** | —（fwd 专用） | 对角 sub-chunk 内积（token_parallel 产物；上游测试 torch.ones） |
| dAqk / dAkk | [B, S, H, 64] | — | float32 | bwd 梯度输入 |
| dq / dk | [B, S, H, DK] | — | float32 | 初始梯度（增量累加） |
| db | [B, S, H] | — | float32 | 初始梯度（host 后处理合并） |
| dg | [B, S, H, DK] | — | float32 | 初始梯度 |
| Mlo | [64, 64] | float32 | float32 | **设计新增常量**：`tril(ones, diagonal=0)`；host 一次性构造 + per-(device) 缓存（★_MSK_CACHE 模式），16KB，L2 常驻 |
| Mneg | [16, 16] | float32 | —（fwd 专用） | **设计新增常量**：`−tril(ones, −1)`（strict 下三角 −1）；1KB |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| Aqk | [B, S, H, 64] | bfloat16 | off-diag 块 × scale；对角 4 块 + 块上三角 = 0（§1.2 口径） |
| Akk | [B, S, H, 64] | bfloat16 | 4×4 块合并逆（块上三角计算精确 0） |
| dq2 / dk2 | [B, S, H, DK] | float32 | bwd |
| db2 | [NK, B, S, H] | float32 | bwd partial（host 后 sum） |
| dg2 | [B, S, H, DK] | float32 | bwd（pre reverse-cumsum） |
| （ws）Akkβ' / AiDiag / G / G² / Y | [B, S, H, 64] ×5 | float32 | **fwd 跨 kernel workspace**（host `torch.empty` 分配，kernel 间 GM 中转，★fla_wy_fast 先例；Akkβ' 紧凑 6 块 / AiDiag 4 对角块 / G 6 块 / G² 4 块 / Y 整块） |
| （ws）db2flat / db2flat_v | [NK·NT·B·H, 64] | float32 | bwd db2 scatter 暂存（★impl_final.py:428 同款，含 [·,64,1] 视图别名） |

### 4.3 中间缓冲区（每 program，代表配置 DK=128：fwd BK=128 / bwd BK=64）

**fwd_k_inter**：k_half [64,128]bf16 16KB + k_f/g_f [64,128]fp32 ×2 64KB + q_half [48,128]bf16 12KB + q_f [48,128]fp32 24KB + rhs_f [48,128]fp32 24KB（槽位复用） + lhs/qg/kg' [16,128]fp32 复用 8KB + 锚点/β 链 ~2KB + hi/lo bf16 中转 [48,128]×2 24KB；L1：qg/kg' hi/lo 6×[16,128]bf16 48KB + rhs hi/lo (16+32+48)·128·2B·2 48KB ≈ 96KB；L0C 6 frags [16,16/32/48]fp32 ≈ 6KB；epilogue：staging [64,64]bf16 8KB + fp32 中转 16KB。

**fwd_k_g**：ad_f [64,16]fp32 4KB + 4×Ai_s [16,16]fp32 4KB + 求逆临时（a_row/acol/ab/tmp/red）~3KB + AiDiag staging [64,64]fp32 16KB + Akkβ' staging [64,64]fp32 16KB；L1：AiDiag/Akkβ' hi/lo bf16 各 2 份（A/B 分开，R3 规避）32KB；L0C G_frag [64,64]fp32 16KB。**k_g2/k_y/k_ai** 同量级。

**bwd_k_intra**（Pipelined 双缓冲敏感）：prologue 持久 MdA/MdK fp32 [64,64]×2 32KB；体内：K/G/Q [64,64]fp32 ×3 48KB（载入双缓冲 ~96KB 峰值）+ dAqk/dAkk [64,64]fp32 32KB（双缓冲）+ Kdec/Qdec/KBdec fp32 staging 复用 16KB + 锚点 8×[64] + hi/lo fp16 中转 8KB；L1：MdA/MdK hi/lo [64,64]fp16 ×4 32KB + Kdec/Qdec/KBdec hi/lo 12×[64,64]fp16 96KB ≈ 128KB；L0C 12 frags [16,64]fp32 48KB；尾段 D_s/V_s + frag→UB + dq/dk/dg 载入片 ≈ 30KB。

### 4.4 内存搬运路径

```
fwd（bf16 in / fp32 gate）:
GM[q,k:bf16] ─T.copy(同dtype)→ UB[bf16] ─T.copy(cast)→ UB[fp32] ─tile 链(锚点衰减)→ UB[fp32]
  ─T.copy(cast)→ UB[bf16 hi/lo] ─T.copy→ L1 ┐
GM[gk:fp32] ─T.copy(同dtype)→ UB[fp32] ──────┤→ L0C[frag fp32]（gemm_v0 ×3 hi/lo 累加）
GM[beta] flat-span → tile.gather → UB[β] ────┘
L0C → UB(fp32) → (×scale/直接) → staging(bf16 零填充/整块) → GM（Aqk / Akkβ' ws）
GM[Akk_diag] → UB → 前代求逆(纯V) → AiDiag → staging → L1 → gemm → G ws → G·G → G² ws
  → (I+G²)·AiDiag → Y ws → (I+G)·Y → Akk(bf16) GM（三步输出链）

bwd（全 fp32）:
GM[q,k,g,dAqk,dAkk,dq,dk,dg:fp32] ─T.copy(同dtype)→ UB[fp32] ─tile 链→ UB[fp32]
  ─T.copy(cast)→ UB[fp16 hi/lo] ─T.copy→ L1 ┐
GM[Mlo] → UB ─tile.mul→ MdA/MdK ─hi/lo→ L1 ─┤→ L0C[12 frags]（48 gemm_v0）
GM[beta] flat-span → tile.gather → UB[β] ───┘
L0C → UB(fp32) → tile 尾段 → GM（dq2/dk2/dg2 切片）+ db2flat → flag → db2 切片
```

> 层级合法性：无 GM→L0 直搬；Cube 输入经 L1（gemm_v0 内部 L1→L0A/L0B）；Vector 计算与 reduce 全在 UB；L0C→UB 用 T.copy 直连；跨 dtype 转换全部 UB↔UB 片内（★缺陷 8-6/8-7 规避）。

### 4.5 片上内存预算表（910B3：L1=512KB、UB=192KB、L0A/L0B=64KB、L0C=128KB per core；bench 配置）

| kernel | L0C | L1 | UB 峰值（估） | 结论 |
|--------|-----|-----|--------------|------|
| fwd_k_inter | 6KB（5%） | ~96KB（19%） | ~150KB（**78%**；k_f/g_f/q_f 持久 88KB + staging/中转，依赖 MEMORY_PLANNING lifetime 复用） | ✓ 紧张，R5 |
| fwd_k_g | 16KB（13%） | ~32KB（6%） | ~50KB（26%） | ✓ |
| fwd_k_g2 | 16KB | ~32KB | ~50KB | ✓ |
| fwd_k_y | 16KB | ~48KB | ~66KB（34%） | ✓ |
| fwd_k_ai | 16KB | ~48KB | ~50KB | ✓ |
| bwd_k_intra | 48KB（38%） | ~128KB（25%） | ~170KB（**89%**；K/G/Q/dA 双缓冲 128KB + MdA/MdK 持久 32KB + staging，MEMORY_PLANNING 复用后估算） | ✓ 紧张，R6 |

**降级预案**：fwd_k_inter UB 溢出 → 改 Pipelined(DK//64=2) 形（体减半 ~90KB，★k1 先例形；代价：体翻倍 + 双缓冲）或 DK 域收紧 {64,96}；bwd UB 溢出 → 按输出拆 2 kernel（dq2+db2 / dk2+dg2，重读 dAqk/K/G ~+80MB，§10.4-F）。

### 4.6 动态轴定义

无运行时动态轴。B/S/H/DK 均为 JIT 编译期常量（随 shape 重编译）；NT/NK/grid 编译期展开。

### 4.7 JIT 配置

```python
PASS_CONFIGS = {                                              # ★impl_final.py:79-84 同款四件套
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 C/V 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,       # 自动核间同步
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,          # 自动核内同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,    # UB/L1 lifetime 规划（紧张预算前提）
}
@tilelang.jit(out_idx=[...], pass_configs=PASS_CONFIGS)
```

> ws 张量（Akkβ'/AiDiag/G/G²/Y/db2flat）以普通 GM tensor 入参（host `torch.empty` 分配），不用 workspace_idx 机制；out_idx 用**绝对索引**（★docstring-7：自动注入的 GM 参数会平移负索引）。

---

## 5. Tiling 策略

### 5.1 计算类型

**混合（CV 融合）**：Cube（fwd k_inter 18 gemm_v0/program + solve 链 4×3；bwd 48/program）+ Vector（exp、掩码、cast、reduce、前代求逆）。

### 5.2 Block 划分

```python
BS = 64            # = chunk_size（语义约束）
BK_fwd = DK        # 整块（直线 1 C 段；64/96/128）
BK_bwd = DK // 2   # exact-2 Pipelined 强制（32/48/64）；NK = 2
NC = 4             # 每 program 覆盖全部 4 个 sub-chunk → 上游 grid 的 NC 维（i_i）消解
grid_fwd = NT * B * H          # bench: 128×1×64 = 8192
grid_bwd = NK * NT * B * H     # bench: 2×128×1×8 = 2048（i_k 最内 → L2）
```

**选型理由**：① bwd 把上游 4 个 sub-chunk program 合并为 1（masked-GEMM 重表述使 (B)/(E) 逐列依赖消失 → 无需按 i_i 切 program），K/G/Q/dA 每元素恰读 1 次（上游结构 ~4× 重读），grid 8192→2048（program 内 GEMM 批量更大）；② BK_bwd=DK//2 是 extent=2 工具链约束下的唯一解；③ fwd BK=DK 消除 K 循环 → 单 C 段直线（z/da 安全形），UB 单缓冲 78%。

### 5.3 约束分析

- 对齐：bf16 尾轴 DK·2B ≥ 128B；fp32 DK·4B / 行 64·4B=256B ≥ 32B ✓；[16,16] 切片行 64B ✓；L0A/L0B 512B 分形由 gemm_v0 内部处理（kL0Size=128 默认）
- L0C：max 48KB ≤ 128KB ✓；GEMM 非整除：域内无（§3.5.1）；列向 strided：无（§3.5.1）

### 5.4 注意事项（尾块）

**声明域内无尾块**：S%64==0（proto 范围）、DK 整块/exact-2 构造。域外输入由 host `_check_shape` 断言拒绝（★`_check_shape` + `_pick_block_d` 同款）——L2 负向覆盖，不设计 host padding + crop（禁止）。上游 `min(BC, T−i_ti)` 尾块逻辑被域约束消除（恒满块）。

### 5.5 数据搬运性能可行性（REORDER-COST）

```text
[REORDER-COST]
path: fwd 主链（5 kernel 合计，bench 1×8192×64×128，bf16 in/fp32 gate）
gm_passes: 唯一 IO 673MB（读 q 128 + k 128 + gk 256 + Akk_diag 32 + beta 1；写 Aqk 64 + Akk 64）
  + ws 中转 624MB（Akkβ' 96 + G 144 + G² 64 + Y 256 + AiDiag 64）+ beta span 逻辑读 ~64MB（L2 命中）
  ≈ 1.36GB 逻辑流量
dma_transactions: ~40 次/program·5 kernel（整块 [64,·] 载入 + 切片写 + gather span），8192 programs ≈ 16 万次
average_dma_bytes: ~8.5KB/次（[64,64]/[64,128] 块为主）
gm_scalar_accesses: 0（标量/行写均在 UB；GM 无标量访问；db2 scatter 走已验证链）
address_div_mod_per_element: 0（cid 分解每 program 3-4 次；s/s2 追踪期展开）
active_cores / serial_tasks_per_core: 20 / ~410 programs/core
largest_case_timeout_gate: perf_fwd_bench 单 case 300s（§9.4，含 5 kernel JIT 编译余量）
result: pass

path: bwd 主路径（单 kernel，bench 1×8192×8×128，全 fp32）
gm_passes: 唯一 IO 322MB（读 q/k/g 各 32 + dAqk/dAkk 各 16 + dq/dk/dg 各 32 + beta 1 ≈ 225；写 96.5）
  + beta span ~4MB + db2flat 暂存 ~1MB ≈ 327MB 逻辑流量（K/G/Q/dA 的上游 4× 跨子块重读已被
  program 合并 + i_k 相邻 L2 消除）
dma_transactions: ~25 次/program（K/G/Q/dA/dq/dk/dg 载入 + Mlo + gather span + 4 输出 + scatter 链），2048 programs ≈ 5 万次
average_dma_bytes: ~6.5KB/次
gm_scalar_accesses: 0
active_cores / serial_tasks_per_core: 20 / ~102
largest_case_timeout_gate: perf_bwd_bench 单 case 180s（§9.4）
result: pass
```

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| (chunk, bh[, k-half]) | block 级并行 | `T.Kernel(GRID)` | 每 program 独立，无依赖 |
| fwd s2 ∈ {1,2,3} / solve 阶段 | 追踪期静态展开 | Python `range` | 编译期常量展开（★token_parallel range 同款），无运行时循环 |
| 前代求逆 i ∈ [2,16) | **纯 V 串行**（循环携带） | `T.serial(2, 16)` | 行 i 依赖行 <i 的 Ai——真顺序依赖；体纯 V（无 C 交换），serial 安全（★mis-sync 仅限 V→C 交换） |
| bwd K-half | 迭代+预取 | `T.Pipelined(DK//BK=2, num_stages=2)` | 恰好 2 迭代（extent 约束）；体 = V-prep + C(48 gemm) + V 尾段（★k1 形） |
| bwd s ∈ {0..3} | 追踪期静态展开 | Python `range(4)` | (A)(B)(D)(E) 重表述后 s 间无依赖（输出行块不相交），可展开；gemm 操作数切片为常量界 |

### 6.2 循环携带依赖说明（任务点名难点汇总）

- **fwd 前代求逆**：14 步真顺序（§6.1）；每步为 [16] 行向量 + [16,16] tile 操作，向量化映射见 §3.4；不可 GEMM 化/倍增化的论证见 §10.4-D。
- **bwd (B)/(E) 逐列循环**：**已消除**——重表述为 masked GEMM（§1.3），掩码只依赖 (i,j) 不依赖迭代状态，"顺序累加"实为独立项求和。数值恒等 + 锚点域安全。
- **bwd Pipelined 体内的跨 half 状态**：(a) gemm frag 累加用 `init=(i_kk==0)`（C 侧语义，无 V 参与）；(b) db 跨 half 累加（4 个 s 的 rowsum 在两个 half 各得部分和）：**用 clear=True + tile.add 手动累加模式**（★docstring-6：clear=False 的 in-loop merge 为工具链 bug）——db_acc [64] prologue 清零（tile.fill），体内每 half `reduce_sum(..., clear=True)` 后 `tile.add(db_acc, db_acc, part)`；(c) 其余输出（dq2/dk2/dg2）按 half 写**不相交的 K 列片**（`dq2[bb, s0:s0+64, bh, i_kk·BK:+BK]`），无跨迭代携带。
- **db2 scatter 的 flag 链**在 Pipelined 体**外**的 kernel 尾部（★impl_final.py:774-782 位置一致：循环后、kernel 末尾）。

### 6.3 流水线优化

bwd Pipelined num_stages=2（软件预取 K/G/Q/dA 的下一 half）；核内流水与核间流水互斥由 AUTO_CV_COMBINE 自动编排（★fla_wy_fast §6.3 同款）。fwd 各 kernel 直线无循环——MTE 与 Vector/Cube 的重叠交给多 program 并发与 Stage 3 优化（§10.4-G）。

### 6.4 尾块处理

无尾块（§5.4）。

---

## 7. 同步策略

### 7.1 同步模式

**自动同步**（Developer 标准）：AUTO_SYNC 处理核内 MTE/Vector 依赖，AUTO_CV_SYNC 处理 Cube↔Vector 核间依赖。源码无 `T.barrier_all`；唯一显式原语为 bwd db2 scatter 的 `T.set_flag/wait_flag("MTE3","MTE2",0)`（★impl_final.py:779-780 逐字同款——计算源 scatter 的 MTE3→MTE2 可见性链，Developer 模式下已交付验证）。

### 7.2 同步点说明

| 依赖 | 说明 |
|------|------|
| tile ops / gemm 操作数 ← GM 载入（T.copy） | AUTO_SYNC |
| gemm ← L1 操作数就绪 / L0C 结果 → UB 中转 | AUTO_SYNC（C 核内）/ AUTO_CV_SYNC（跨核） |
| Pipelined 体内 V-prep → C gemm → V 尾段 | AUTO_CV_SYNC（★k1 形已验证） |
| db2flat 写 → flag → 重载 → db2 scatter | **显式 set_flag/wait_flag**（§7.1） |
| 跨 kernel（fwd 5 链） | GM 中转，host 顺序调用天然同步 |

### 7.3 pass_configs 配置

见 §4.7（四件套全开）。

---

## 8. CV 融合设计

### 8.1 融合算子判定

**是**（Cube GEMM + Vector exp/掩码/cast/reduce/前代求逆）。

### 8.2 CV 交互设计（Developer 模式）

- `T.Kernel(GRID, threads=1, is_npu=True) as (cid)`：单轴 + threads=1，**无 workspace_idx、无 vid 偏移索引**（vid 切分被 threads=1 消除；★fla_wy_fast 最终交付同款）
- Cube↔Vector 片上直连：`T.copy(frag, shared_ub)` / `T.copy(shared_ub, l1_shared)`，中转与同步交给四件套 pass
- **每 kernel ≤1 个直线 C 段**（§1.4 安全包络）：fwd 的合并逆链因此拆 4 kernel、ws 走 GM；这是本算子相对上游最大的结构性偏差，依据为 ★final_summary §8-4（z+da 多跳编译错）与 §8-2（单 Pipelined 体）
- fp32 gemm 损坏规避：全部 GEMM 操作数 hi/lo 分裂（§3.5.1），分裂与重装在 V 侧、累加在 L0C（init 链）

### 8.3/8.4 Cube 与 Vector 核计算流程

见 §3.4：Cube 侧只做 gemm_v0（hi/lo 3 连击）；Vector 侧做 exp/掩码/cast/前代求逆/尾段；两侧经 alloc_shared(L1/UB) 直连。

### 8.5 pass_configs

见 §4.7。

### 8.6 注意事项

- 核内 Pipelined 与核间流水互斥（pass 自动决策）
- fwd_k_g2 的 G·G：同一 L1 buffer 作 A/B 双角色未经实证（★缺陷 #3 为跨 gemm 组角色互换复用损坏）——**预置 A/B 两份 hi/lo 拷贝**规避（§10.1 R3）
- bwd MdA/MdK hi/lo 与 Kdec/Qdec/KBdec hi/lo 共 16 组 L1 操作数、48 gemm 调用共处一个 C 段：数量无先例（★最大 3 gemm/段）但结构同类（无 V 间隔）——R1 风险 + 两级回退（§10.1）

---

## 9. 验证方案

### 9.1 Golden 函数（PyTorch 重写，fp64 内部域，供 Stage 2 直接落地）

> NPU 环境 Triton 不可运行（ref_gpu 仅语义对照）。golden 按 precision-standard.md §三"单标杆高精度"口径在 **fp64** 内部计算（消除极端门控下任何中间溢出/下溢隐患），输出 cast 到声明 dtype 后比对；kernel 侧 fp32 累加 + exp(x·ln2) + hi/lo 分裂的实现差异全部计入混合容差验收。输入构造对齐上游 prepare_input 语义（fwd: gk = logsigmoid(randn) + chunk 内 cumsum 的 PyTorch 等价、Akk_diag = ones；bwd: 全 randn、g 不 cumsum）。数据准备/golden 全在 CPU 完成后 H2D。

```python
import torch

def _chunk_cumsum(g, chunk=64):                       # chunk 内局部 cumsum（fwd 输入构造 + 无）
    B, S, H, D = g.shape
    return g.view(B, S // chunk, chunk, H, D).cumsum(2).view(B, S, H, D)

def fla_chunk_intra_fwd_ref(q, k, gk, beta, Akk_diag, chunk_size=64,
                            sub_chunk_size=16, scale=1.0):
    """fwd golden。q/k [B,S,H,DK] bf16, gk fp32(cumsum 后), beta [B,S,H] bf16,
    Akk_diag [B,S,H,16] fp32 -> (Aqk, Akk) bf16。未写出区域 = 0（§1.2 口径）。"""
    B, S, H, DK = q.shape
    CS, BC = chunk_size, sub_chunk_size
    assert S % CS == 0 and CS % BC == 0
    qf, kf, bf, gf, adf = (t.double() for t in (q, k, beta, gk, Akk_diag))
    Aqk = torch.zeros(B, S, H, CS, dtype=torch.float64)
    Akk_out = torch.zeros(B, S, H, CS, dtype=torch.float64)
    for c in range(S // CS):
        s0 = c * CS
        g_c, k_c, q_c = gf[:, s0:s0+CS], kf[:, s0:s0+CS], qf[:, s0:s0+CS]
        # (1) off-diag Aqk：直接式 2^(g_r − g_j)（fp64 域安全；r 块号 > j 块号）
        for s2 in range(1, 4):
            r0 = s2 * BC
            for s1 in range(0, s2):
                j0 = s1 * BC
                e = torch.exp2(g_c[:, r0:r0+BC].unsqueeze(2)          # [B,BC,BC,H,DK] (r,j,h,k)
                               - g_c[:, j0:j0+BC].unsqueeze(1))
                val = torch.einsum("brhd,bjhd,brjhd->brjh", q_c[:, r0:r0+BC],
                                   k_c[:, j0:j0+BC], e) * scale
                Aqk[:, s0+r0:s0+r0+BC, :, j0:j0+BC] = val.permute(0, 1, 3, 2)
        # (2) M 组装（fp64）：M[s,s] = I + strict_tril(A_s)；M[s2,s1] = Akkβ[s2,s1]（正号）
        #     A_s[b, r, c, h] = Akk_diag[b, s0+s·BC+r, h, c]
        M = torch.zeros(B, CS, CS, H, dtype=torch.float64)
        Iblk = torch.eye(BC, dtype=torch.float64)                      # [BC, BC]
        Str = torch.tril(torch.ones(BC, BC, dtype=torch.float64), -1)  # strict 下三角指示
        for s in range(4):
            r0 = s * BC
            A_s = adf[:, s0+r0:s0+r0+BC, :, :].permute(0, 1, 3, 2)      # [B, BC(r), BC(c), H]
            M[:, r0:r0+BC, r0:r0+BC, :] = (A_s * Str[None, :, :, None]
                                           + Iblk[None, :, :, None])
            for s2 in range(s + 1, 4):
                r2 = s2 * BC
                e = torch.exp2(g_c[:, r2:r2+BC].unsqueeze(2)             # [B,BC,BC,H,DK]
                               - g_c[:, r0:r0+BC].unsqueeze(1))
                akk = torch.einsum("brhd,bjhd,brjhd->brjh", k_c[:, r2:r2+BC],
                                   k_c[:, r0:r0+BC], e)                 # Akk 原始块
                M[:, r2:r2+BC, r0:r0+BC, :] = (akk                     # × β[s2 行]（正号）
                    * bf[:, s0+r2:s0+r2+BC].unsqueeze(-1)).permute(0, 1, 3, 2)
        # (3) Ai = M⁻¹（单位下三角；与逐块前代求解 + 块递推代数等价，§1.3(2)(3)）
        Mi = M.permute(0, 3, 1, 2)                                      # [B, H, CS, CS]
        eye = torch.eye(CS, dtype=torch.float64).expand(B, H, CS, CS).contiguous()
        Ai = torch.linalg.solve_triangular(Mi, eye, upper=False, unitriangular=True)
        Akk_out[:, s0:s0+CS, :, :] = Ai.permute(0, 2, 3, 1)
    return Aqk.to(q.dtype), Akk_out.to(q.dtype)
```

> 注：`M[s,s] = I + strict_tril(A_s)`，其中 `A_s[b,r,c,h] = Akk_diag[b, s0+s·BC+r, h, c]`；`M[s2,s1] = Akkβ[s2,s1]` 与 kernel §1.3(1) 的 Akkβ 同式（正号；负号折叠 Akkβ' = −Akkβ 仅存在于 kernel 侧 ws）。`solve_triangular(unitriangular=True)` 与"逐块前代求解 + 6 块递推"代数等价（§1.3(2)(3) 已验证）。**交叉校验建议（Stage 2 可选）**：极小 shape（S=64）追加"逐块前代求解 + 6 块递推"显式版本与 solve_triangular 版本对拍，锁定块结构与符号方向。

```python
def fla_chunk_intra_bwd_ref(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg,
                            chunk_size=64, sub_chunk_size=16):
    """bwd golden。全 fp32 输入 -> (dq2, dk2, dg2, db_pre)（fp64 域）；返回 kernel 输出
    同构 4 元组 + host 后处理同构（db2.sum(0)+db、reverse cumsum 由测试侧对两路
    分别施加同构序列后比对，见 §9.2）。"""
    B, S, H, DK = q.shape
    CS, BC = chunk_size, sub_chunk_size
    qf, kf, bf, gf = (t.double() for t in (q, k, beta, g))
    dAf, dKf, dqf, dkf, dbf, dgf = (t.double() for t in (dAqk, dAkk, dq, dk, db, dg))
    dq2 = torch.zeros(B, S, H, DK, dtype=torch.float64)
    dk2 = torch.zeros(B, S, H, DK, dtype=torch.float64)
    dg2 = torch.zeros(B, S, H, DK, dtype=torch.float64)
    dbp = torch.zeros(B, S, H, dtype=torch.float64)          # = db2.sum(0) 等价（NK 无关）
    for c in range(S // CS):
        s0 = c * CS
        g_c, k_c, q_c, b_c = gf[:, s0:s0+CS], kf[:, s0:s0+CS], qf[:, s0:s0+CS], bf[:, s0:s0+CS]
        dA_c, dK_c = dAf[:, s0:s0+CS], dKf[:, s0:s0+CS]
        ar = torch.arange(CS)
        for i in range(CS):            # 统一掩码直接式（§1.3 锚点恒等式的直接式，fp64 域安全）
            m_lo = (ar <= i).double()                     # j ≤ i：(A)+(B) 统一掩码
            m_up = (ar >= i).double()                     # j ≥ i：(D)+(E) 统一掩码
            eij = torch.exp2(g_c[:, i].unsqueeze(1) - g_c)  # 2^(g_i−g_j) [B,CS,H,DK]
            eji = torch.exp2(g_c - g_c[:, i].unsqueeze(1))  # 2^(g_j−g_i) [B,CS,H,DK]
            # (A)+(B)：dq2[i] / dk2pre[i]（pre-β）
            dq2[:, s0+i] = torch.einsum("bjh,bjhk->bhk", dA_c[:, i] * m_lo, k_c * eij)
            dk2pre = torch.einsum("bjh,bjhk->bhk", dK_c[:, i] * m_lo, k_c * eij)
            # (D)+(E)：dkt[i]（转置侧；β_j 乘在 dK 项内）
            dkt = (torch.einsum("bjh,bjhk->bhk", dK_c[:, :, :, i] * m_up * b_c, k_c * eji)
                   + torch.einsum("bjh,bjhk->bhk", dA_c[:, :, :, i] * m_up, q_c * eji))
            # 尾段（(F) post-β，§1.2 勘误；dg2 的 q⊙dq2 用 +dq 之前的值，上游 L488-489 顺序）
            dbp[:, s0+i] = (dk2pre * k_c[:, i]).sum(-1)                    # pre-β（上游 L481）
            dk2b = dk2pre * b_c[:, i].unsqueeze(-1)
            dg2[:, s0+i] = (q_c[:, i] * dq2[:, s0+i]
                            + (dk2b - dkt) * k_c[:, i] + dgf[:, s0+i])
            dq2[:, s0+i] = dq2[:, s0+i] + dqf[:, s0+i]                     # (C)
            dk2[:, s0+i] = dk2b + dkf[:, s0+i] + dkt
    return dq2, dk2, dg2, dbp
```

> **掩码统一性依据**：(A) 的 j ∈ 前序块 + (B) 的 j ∈ 对角块且 j ≤ i，并集恰为 j ≤ i；(D)(E) 同理并集恰为 j ≥ i——golden 掩码（m_lo/m_up）与 kernel 的 Mlo 逐位一致。`dbp` 为 `db2.sum(0)` 的等价（NK 无关）——测试侧对 kernel 输出先施加同构后处理 `db2.sum(0)+db` 后与 `dbp+db` 比对；dg 同理由测试侧对两路分别施加同构 reverse cumsum 后比对（proto"逐位同构"口径）。
>
> **交叉校验建议（Stage 2 可选）**：极小 shape（S=64）以 Python `min(...)` 显式复现上游 Triton 逐列循环（B)/(E) 与本统一式对拍，锁定掩码方向与 (F) post-β 顺序。

### 9.2 L0 门槛测试计划（tilelang-op-test-design 场景 A 生成，仅 L0）

> Stage 1 只产出 L0（规则 shape：S%64==0、DK ∈ {64,128}、block 整除）。L1（功能，含不规则 shape 的确定性生成——尾块/质数对本算子为不支持域，由 L2 负向覆盖替代）/L2（异常）/Boundary（特殊值，含极端衰减 gk、fp16 hi/lo 域边界）由 Stage 2 在 L0 通过后按场景 B 扩展并跑 `scripts/coverage_check.py`。

**算子类别判定（场景 A Phase 2）**：计算类型 = 混合（CV 融合：Cube gemm_v0 + Vector exp/掩码/cast/reduce/前代求逆）；复杂度 = Fusion（GEMM + 衰减 + 求逆 + masked GEMM 多算子组合）；数学特征 = matmul + exp2 衰减 + 三角求逆；综合类别 = **Fusion（KDA 家族）**。测试聚焦：CV 交互正确性、5-kernel ws 链正确性、hi/lo 分裂精度、掩码方向、未写区域 0 口径、db2 scatter 链、host 后处理边界。

| 用例名 | 级别 | Shape (B,S,H,DK) | dtype | 覆盖标签 | 说明 |
|--------|------|------------------|-------|----------|------|
| l0_fwd_min | L0 | (1, 64, 2, 128) | q/k/beta bf16 + gk/Akk_diag fp32 → Aqk/Akk bf16 | D-DTYPE-bf16, D-DTYPE-fp32, D-SHAPE-ALIGNED | 最小冒烟：1 chunk/2 head（grid=2）；覆盖 6 个 off-diag 块、前代求逆（Akk_diag=ones）、(I+G)(I+G²) 链 5 kernel、Aqk 未写区域 0 口径、beta gather、bf16 hi/lo |
| l0_fwd_repr | L0 | (2, 256, 4, 128) | 同上 | D-DTYPE-bf16, D-DTYPE-fp32, D-SHAPE-ALIGNED | 多 chunk/batch/head（grid=32）；chunk 间独立、跨 chunk 边界 |
| l0_fwd_dk64 | L0 | (1, 128, 4, 64) | 同上 | D-SHAPE-ALIGNED, D-DTYPE-bf16 | DK=64 域变体（fwd BK=64 整块路径；n∈{16,32,48} gemm 形状） |
| l0_fwd_bench_repr | L0 | (1, 1024, 64, 128) | 同上 | D-DTYPE-bf16, D-SHAPE-ALIGNED | bench 结构代表（H=64，grid=1024）：大 H 的 beta gather span、多 program 并发下的 ws 链 |
| l0_bwd_min | L0 | (1, 64, 2, 128) | 全 fp32 | D-DTYPE-fp32, D-SHAPE-ALIGNED | 最小冒烟：覆盖 s=0（k=32 零列补齐路径）/s=3（无后序块路径）两端 + 对角/转置对角 masked gemm + db2 scatter flag 链 + host 后处理 |
| l0_bwd_repr | L0 | (2, 256, 4, 128) | 全 fp32 | D-DTYPE-fp32, D-SHAPE-ALIGNED | 多 chunk + NK=2 双 K-half（Pipelined extent-2、跨 half frag 累加 init 链、L2 相邻共享） |
| l0_bwd_dk64 | L0 | (1, 128, 4, 64) | 全 fp32 | D-DTYPE-fp32, D-SHAPE-ALIGNED | DK=64 → BK=32（exact-2 变体；n=32 gemm 形状） |

**golden 对照法**：每用例构造输入（上游对齐：fwd 用 `prepare_input_fwd`——q/k/beta randn bf16、gk = logsigmoid(randn fp32) 后 `_chunk_cumsum`、Akk_diag = torch.ones fp32；bwd 用 `prepare_input_bwd`——全 randn fp32、g 不 cumsum）→ kernel 侧跑 wrapper 得 (Aqk, Akk) / (dq2, dk2, db2, dg2)；bwd 侧对 kernel 输出与 golden 输出**分别施加同构 host 后处理**（`db2.sum(0)+db`、`_chunk_cumsum(dg2, reverse=True)` 的 flip-cumsum-flip 等价）后比对 db/dg。对每个输出独立调用 `check_precision(out, ref, dtype)`（precision-standard.md §4.1 整段复制不改动）：fwd 判 Aqk/Akk（bf16 档），bwd 判 dq2/dk2/dg2（fp32 档）+ 后处理后的 db/dg（fp32 档）。全部通过打 `[PRECISION_PASS]`，任一失败 `[PRECISION_FAIL]`（L0 阻塞，计入退出码）。

**测试文件结构**：`test_fla_chunk_intra.py`（tilelang-op-test-design §9.1 模板）：`get_precision`/`check_precision` 原样复制 + 两个 golden（§9.1）+ 两个 prepare_input + `_run_precision` + L0 七用例 + L1/L2/Boundary 桩（Stage 2 场景 B 填充）+ `--level` main 分发器（`tilelang.disable_cache()`、seed=0）。

### 9.3 精度标准（混合容差，precision-standard.md 口径）

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|-------|------|------|---------------------|------------------------|
| bfloat16（fwd 输出 Aqk/Akk；输入 q/k/beta） | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |
| float32（bwd 输出 dq2/dk2/db2/dg2；输入 gk/g/dg/Akk_diag） | 2⁻¹⁶ (1.53e-5) | 2⁻¹⁰ (9.77e-4) | 1e-2 | 0.99 |

> 与 proto_fla_chunk_intra.yaml 声明的门禁逐字一致；dtype 全集 = {bfloat16, float32} 与 proto `inputs[].dtype` 一致。hi/lo 分裂误差论证：fwd bf16 hi/lo 3-gemm ~2⁻¹⁷ 相对 ≪ rtol 2⁻⁶；bwd fp16 hi/lo 3-gemm ~2⁻²² ≪ rtol 2⁻¹⁰（★fla_wy_fast 交付验证同款；fp32 直接操作数因 gemm 损坏不可用，单精度 bf16 操作数经估算 ~6% 元素超逐元素容差被否决，§10.4-E）。特殊值（NaN/Inf）：输入域为 randn 有限值；极端衰减 gk（sub-chunk 内累计 < −24 触发 fp16 hi/lo 下溢为 0）与 dg2 后处理溢出场景由 Stage 2 Boundary 扩展并按 mask 结构比对。

### 9.4 性能可行性哨兵（强制，不可跳过）

| 用例名 | Shape (B,S,H,DK) | dtype/属性 | 覆盖路径 | 单 case timeout | 选择理由 |
|--------|------------------|------------|----------|-----------------|----------|
| perf_fwd_bench | (1, 8192, 64, 128) | bf16 in/fp32 gate，CS=64，SCS=16，scale=1.0 | fwd 主链 5 kernel（grid 8192/核 410，最大任务数 + ws 中转吞吐） | 300s（5 kernel JIT 编译余量） | benchmark 验收 shape |
| perf_bwd_bench | (1, 8192, 8, 128) | 全 fp32 | bwd 单 kernel（grid 2048，48 gemm/program，fp16 hi/lo Cube 最大压力） | 180s | benchmark 验收 shape |

计时口径：NPU 事件计时（`torch.npu.Event(enable_timing=True)` 等价实现 ref_gpu/test_utils_kda.py do_bench），warmup=20、rep=10、取均值，仅设备侧 kernel 时间；wrapper 内**不加 sync**（★fla_wy_fast 口径教训：内部 sync 把 ~1ms/launch 的 host JIT 分发延迟暴露进设备侧计时）；fwd 计 5 kernel 顺序调用总时间、bwd 计单 kernel（host 后处理不计时，proto 声明）。

### 9.5 性能预算分析（910B3，含 540us 目标可行性结论）

**IO 明细（物理唯一字节；bwd program 合并已消除上游 ~4× 跨子块重读）**

| 项 | fwd bench (1,8192,64,128) | bwd bench (1,8192,8,128) |
|----|--------------------------|--------------------------|
| 读 | q 128 + k 128 + gk 256 + Akk_diag 32 + beta 1 = **545 MiB** | q/k/g 各 32 + dAqk/dAkk 各 16 + dq/dk/dg 各 32 + beta 1 ≈ **225 MiB** |
| 写 | Aqk 64 + Akk 64 = **128 MiB** | dq2/dk2/dg2 各 32 + db2 0.5 ≈ **96.5 MiB** |
| 唯一 IO 合计 | **673 MiB ≈ 706 MB** | **322 MiB ≈ 338 MB** |
| ws 中转（工具链强制拆分） | Akkβ' 96 + G 144 + G² 64 + Y 256 + AiDiag 64 ≈ **624 MiB** | db2flat ~1 MiB |
| 逻辑流量合计 | **~1.36 GB**（+ beta span ~64MB L2 命中） | **~0.33 GB** |

**GEMM/Vector 计算量（逻辑，×3 hi/lo 后为 fp16 Cube 实际量）**

| 项 | fwd | bwd |
|----|-----|-----|
| Cube 逻辑 MAC | (393K off-diag + 4×262K solve) × 8192 = **11.8 GMAC** | 876K × 2048 = **1.79 GMAC** |
| ×3 hi/lo 后 | 35.4 GMAC = 70.8 GFLOP fp16 | 5.4 GMAC = 10.7 GFLOP fp16 |
| Vector 主项 | 前代求逆 ~390 tile ops/program（k_g 一次） + 衰减链 ~100 ops | 衰减/掩码/尾段 ~450 tile ops/program |

**Roofline 对照（910B3 实测标定）**

| 项 | 数值 | 依据 |
|----|------|------|
| HBM 实测上限（D2D r+w） | **1259-1273 GB/s**（取 1263） | 【已确认事实】★fla_wy_fast perf_tuning 实测 |
| 唯一 IO 带宽下限（完美单体融合、100% 利用率） | (706+338)MB ÷ 1263GB/s ≈ **826 us** | 物理下限（无 ws 版） |
| 逻辑流量带宽下限（本设计 6-kernel 结构，100%） | (1.36+0.33)GB ÷ 1263GB/s ≈ **1.34 ms** | 工具链拆分代价 |
| @81%（★同族单 kernel 实测最高逻辑带宽占比） | ≈ **1.65 ms** | 【推断】 |
| fwd Cube 时间 | 70.8 GFLOP @ 50-100 TFLOPS（m16 小 GEMM 有效吞吐【推断】）≈ **0.7-1.4 ms** | 与 IO 重叠后取 max |
| bwd Cube 时间 | 10.7 GFLOP ≈ **0.11-0.21 ms** | |
| 前代求逆 Vector | ~0.16-0.4 ms（390 ops/program × 410 serial） | 【推断】 |
| **现实预期合计** | **≈ 2-4 ms**（6 kernel × 每 program 槽 ~2-5us 延迟主导，★fla_wy_fast 交付实测同量级外推） | 【推断】，perf 哨兵实测后收敛 |

**⚠️ [DESIGN_ERROR] 级发现：540us 目标在 910B3 物理不可达**

1. **带宽侧**：目标隐含 (706+338)MB ÷ 540us ≈ **1.93 TB/s = 实测 r+w 上限的 1.53×**——即便按"完美单体融合 + 唯一 IO + 100% 带宽"的物理下限（826us）也超目标 53%，**不存在 kernel 层优化能弥合**（唯一 IO 已是该算子下限：gk fp32 256MB 为接口契约、每张量单读单写、bwd 跨子块重读已结构性消除）。
2. **工具链侧**：当前 C/V 多跳限制强制 fwd 拆 5 kernel（ws +624MiB），逻辑流量下限升至 1.34ms（100%）/1.65ms（81%）——为物理下限的 1.6-2.0×，差距构成与 ★fla_wy_fast 交付（实测 5.95× 其修正下限）同源（kernel 拆分重复读 + fp32 gemm 损坏 hi/lo ×3 + 单 Pipelined 体限制）。
3. **计算侧**：fwd m16 小 GEMM 有效吞吐未实测，70.8 GFLOP fp16 即便 @100 TFLOPS 也需 0.7ms——与 IO 同量级，fwd 预期计算/IO 双重受限。

**替代路线（需编排层/用户决策，不静默简化）**：
1. 验收口径改为**带宽/延迟效率**：如 fwd+bwd 合计 ≤ 2.5ms（对应 ~65% 逻辑带宽效率 + Cube 余量），或 perf 哨兵首次实测标定后重定；
2. 确认目标硬件为更高带宽平台（910C / CloudMatrix 级）：910B3 仅做功能/精度验收，性能按带宽比例折算报告（★fla_wy_fast 同款处理）；
3. 维持 540us 字面目标 → 本设计无法承诺：需上游工具链修复（fp32 gemm、多跳 C/V sync-point、单 Pipelined 体限制）后方可能逼近 826us 物理下限，属上游依赖而非算子层工作。

### 9.6 Host 侧 Buffer 操作审计

```text
[HOST-METADATA-AUDIT]
operation: 输入构造（CPU randn/logsigmoid/cumsum 后 H2D，数据准备）+ BetaFlat = beta.view(B, S*H)
  （零拷贝 metadata view，contiguous 输入上 view 共享 storage ✓）+ ws 张量 torch.empty 分配
  （新 buffer，非输入替代）+ 6 次 kernel 顺序调用 + bwd host 后处理（db2.sum(0)+db 为 out-of-place
  新张量、dg2 reverse cumsum 作用于输出张量——proto 显式声明的 wrapper 语义、不计时）+ check_precision
  （.cpu() 只读校验）
input_stride -> output_stride: N/A（新建张量与只读校验，无输入变换）
shares_storage / same_data_ptr: BetaFlat 与 beta 共享（metadata-only view，证明成立 ✓）；其余 N/A
aclnn_or_physical_copy: 输入路径 false；bwd host 后处理含 sum/cumsum（NPU 上为 aclnn 类调用，
  proto 显式授权的 kernel 外边界、上游测试同款、不计入性能计时——cann-bench 裁剪风险已知并记录）
result: allow（后处理边界为 proto 声明的算子契约组成部分；输入/输出 kernel 路径零物理化）
```

---

## 10. 风险点与注意事项

### 10.1 风险清单（高/中/低分级）

| 级别 | 风险 | 影响 | 缓解/验证 |
|------|------|------|-----------|
| **高** | R1 bwd 单 C 段 48 gemm_v0 调用（★先例最大 3/段） | 编译/正确性 | l0_bwd_min 首验；回退-1：按输出拆 2 kernel（dq2+db2 / dk2+dg2，+80MB 重读，§10.4-F）；回退-2：按 (A+B)/(D+E) 拆 2 kernel |
| **高** | R2 fwd_k_inter 单 C 段 18 gemm + rhs_s2 同角色（B 位）双 gemm 复用 | 编译/正确性（★缺陷 #3 为角色互换复用，同角色未证实） | l0_fwd_min 首验；回退：Aqk / Akkβ' 拆 2 kernel（+80MB）或 rhs 双份 L1 拷贝 |
| **高** | R10 性能目标 540us 物理不可达（需 1.53× 实测带宽；逻辑结构下限 1.34-1.65ms） | 验收口径 | §9.5 完整论证 + 替代路线，**需编排层/用户决策** |
| 中 | R3 G·G 同一 L1 buffer 双角色（k_g2） | 正确性 | 已内置规避：A/B 各持一份 hi/lo 拷贝（+16KB L1） |
| 中 | R4 前代求逆行写 `T.copy(a_new, Ai_s[i, 0:16])` 符号行切片写 dst 无先例 | 编译/正确性 | l0_fwd_min 首验；回退-1：E 常量行指示 masked-merge（4 tile ops 替代行写）；回退-2：`for j in range(16)` 标量赋值（慢 ~16×，确定正确，★token_parallel R8 模式） |
| 中 | R5 fwd_k_inter UB 78%（DK=128） | 编译溢出 | MEMORY_PLANNING lifetime 复用；降级：Pipelined(DK//64=2) 形（体减半）或 DK 域收紧 |
| 中 | R6 bwd UB 89%（双缓冲 128KB + 持久 32KB） | 编译溢出 | 同上；降级：拆 2 kernel（§10.4-F）或 MdA/MdK hi/lo 移入循环体每次重算（省持久 32KB） |
| 中 | R7 锚点行向量 [1,BK]/[64,1] 直接参与 tile.sub/mul 的广播语义（★先例均为显式 broadcast 后二元 op） | 编译 | 全部按"显式 T.tile.broadcast → 二元 tile op"链书写（§3.4）；L0 首验 |
| 中 | R9 fp16 hi/lo 域限制（\|x\|>65504 / <2⁻²⁴ 失精） | Boundary 精度 | bwd 操作数锚点域安全（§1.3：指数 ≤0 或子块内有界）；极端衰减用例 Stage 2 Boundary 重审（★fla_wy_fast KNOWN LIMITATION 同款） |
| 低 | R8 exp(x·ln2) vs exp2 精度 | 精度 | fp32 相对差 ~1e-7 ≪ 门禁（★同款论证）；golden torch.exp2 对拍覆盖 |
| 低 | R11 ws 紧凑切片写（[16,16/32/48] 行 64-192B）DMA 效率 | 性能微小 | 行 ≥64B ≥32B 合法；REORDER-COST 已计 |

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| fp32 操作数直接进 gemm_v0 | 照抄上游 fp32 GEMM | 1e37 级垃圾（★缺陷 #1） | hi/lo 分裂 3-gemm（本设计已内置） |
| 2 kernel 单体重构上游 | 忽略 C/V 多跳限制 | sync-point 编译错 | 6-kernel 拆分 + GM ws（本设计已内置） |
| `T.if_then_else` 写进 T.Parallel | 照抄上游掩码 | 编译错误 | 常量 Mlo/Mneg + tile.mul（已内置） |
| 逐列 j 循环直接移植 bwd (B)/(E) | 照抄上游 | 循环携带依赖 + 动态边界 + 16 次小载入 | masked GEMM 重表述（已内置） |
| GM↔UB 跨 dtype 直拷 / L0C→GM 直存 | 图省事 | codegen 非法模板 / fixpipe 损坏 | 两步 cast 链 / 三步输出链（已内置） |
| threads=128 / 2D-3D Kernel | 照抄上游 | 编译报错 | threads=1 + 一维化（已内置） |
| 无锚点因式分解（2^{-g_j}） | "简化"衰减 | 极端门控 inf×0=NaN | 锚点式（已内置，§10.4-C） |
| host 侧 gk 预处理（·ln2/2^g 预计算） | 减 kernel 工作量 | aclnn 违规 + IO 翻倍 | 禁止（§10.4-I） |
| golden 用 fp32 域 + 无锚点直接式 | golden 简化 | 极端门控下 golden 自身 inf | fp64 内部域（本设计 golden 规格） |

### 10.3 特殊场景处理

- **非整除 S / DK 域外**：不支持域（host `_check_shape` 断言拒绝，L2 负向覆盖；不设计 host padding+crop）。
- **极小 shape**（S=64）：grid=B·H，负载少但正确；l0_*_min 覆盖。
- **极端衰减 gk**（sub-chunk 内累计 < −24）：fwd bf16 hi/lo 无域风险（bf16 指数域=fp32）；bwd fp16 hi/lo 下溢为 0（真值 ~0，atol 吸收）；`2^{g_n−g_j}` 上溢仅在非单调 g（bwd randn 域 ±8 有界）——Boundary 用例（Stage 2）覆盖并按 mask 结构比对。
- **NaN/Inf 输入**：上游契约 randn 有限值；Boundary 由 Stage 2 扩展。
- **db2 scatter 位置**：flag 链必须在 Pipelined 体外、kernel 尾部（★impl_final.py:774-782 位置同款）。

### 10.4 被否方案（设计决策记录）

| # | 方案 | 否决理由 |
|---|------|---------|
| A | 忠实上游 2-kernel 单体（合并逆 6 级 GEMM 链单 kernel） | V→C→V→C→V 多跳直线交替触发 sync-point 计数失配编译错（★final_summary §8-4，z+da 复现）；serial 循环 V→C 交换 mis-sync（★docstring-1）。改为 6-kernel 拆分 + GM ws（§1.4） |
| B | 合并逆按上游 6 级块递推细拆（7 kernel：inter+s1..s6，每级 [16,16] 小 gemm） | ws 交接块数更多（~10 块/级）+ gemm 更碎（m16n16k16-48）；(I+G)(I+G²)·D⁻¹ 形以 4 个 [64,64] gemm 等价覆盖（代数验证 §1.3(3)），kernel 数与 ws 均更优。块递推细拆保留为 Stage 3 选项（省 ~0.5M MAC/program 的上三角浪费） |
| C | 无锚点因式分解（Aqk = (q⊙2^g)@(k⊙2^{-g})ᵀ 批量 GEMM） | `2^{-g_j}` 在 chunk 内累计衰减 > 128 时上溢 inf×0=NaN（fp32 上限 2^127；fwd 测试 gk cumsum 域即可达 −200）；锚点式两指数恒 ≤0 域安全且与上游逐元素式代数恒等。★token_parallel 被否方案 C 同款论证 |
| D | 前代求逆倍增 GEMM 化（(I+N)(I+N²)(I+N⁴)(I+N⁸)） | 每 [16,16] 块需 6 次 [16,16,16] gemm（4 块 24 次）且仍为串行 gemm 链（不解决 kernel 拆分），相对逐行递推（~390 个 [16,16] 级 tile 操作，纯 V 无 C 依赖）计算量与调用数均劣。**逐行映射（本设计）**：每步 5-7 个 tile 操作（行切片 copy→broadcast→mul→reduce(dim=0)→add→行写），T.serial 纯 V 体安全（mis-sync 仅限 V→C 交换），掩码零成本（N_s init 行切片自带） |
| E | fwd GEMM 单精度 bf16 操作数（免 hi/lo，3× 加速） | 精度估算 FAIL：bf16 操作数舍入 ~2⁻⁹ → Aqk 绝对误差 ~0.026（√128·2⁻⁹ 量级），|golden| ≲ 1.5 的元素（~6-10%）超出逐元素容差 ≫ 1% 预算，且经合并逆 3-4 级链式放大；bf16 hi/lo 2-gemm 也不足（两操作数均非精确 bf16，★fla_wy_fast fwd 的 2-gemm 依赖 A 侧精确 bf16）。3-gemm hi/lo 误差 ~2⁻¹⁷ ≪ bf16 门禁 |
| F | bwd 按输出拆 2 kernel（dq2+db2 / dk2+dg2）作为主设计 | 更稳（每 kernel ~24 gemm）但 dAqk/K/G 重读 ~+80MB（+25% bwd IO）；主设计单 kernel（masked-GEMM 重表述后 4 子块合一，IO 唯一），拆分保留为 R1/R6 回退 |
| G | fwd_k_inter 用 Pipelined(DK//64=2) 作主设计 | 体翻倍（12 组操作数 prep）+ 双缓冲 UB 压力（★k2 满 UB 教训）；直线形单 C 段更简且 UB 78%。Pipelined 形保留为 R5 降级 |
| H | Expert 模式（显式 L1/UB/L0 + 手动 Scope/同步） | 用户指定 Developer；CV 交织无自定义跨核同步需求（唯一显式 flag 为 scatter 缺陷规避，Developer 下已交付验证）；Expert 的 GM workspace 中转反而加重 IO |
| I | host 侧预计算 2^g / g·ln2 / 掩码矩阵 | 违反 host Buffer 约束（ascend-constraints §5，aclnn）；gk 256MB 读+写翻倍；掩码为 KB 级常量已由 GM 常量输入覆盖 |

---

## 11. 交付清单

### 11.1 目录结构（沙箱约定：共享目录 + 算子前缀隔离）

```
examples_experiment/dsa_sparse_finetune/
├── DESIGN_fla_chunk_intra.md          # 本设计文档（含 L0 门槛测试计划 §9.2）
├── proto_fla_chunk_intra.yaml         # 算子接口规格（任务书载体，proto.yaml 角色）
├── ref_gpu/                           # 冻结参考（只读）
├── fla_chunk_intra.py                 # ⬜ 待实现（Stage 2）：6 JIT kernel + 2 host 包装
└── test_fla_chunk_intra.py            # ⬜ 待实现（Stage 2）：golden（§9.1）+ L0（§9.2）+ main 分发器
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN_fla_chunk_intra.md` | ✅ 已完成 | 本文档 |
| `proto_fla_chunk_intra.yaml` | ✅ 已完成（任务书载体，沿用不改动） | proto.yaml 角色：`operator.inputs[].dtype` 全集 = {bfloat16, float32}（与 §9.3 精度表两行一致）、attrs 覆盖 chunk_size/sub_chunk_size/scale、schema/benchmark/test_alignment 完整——经本次设计逐项核对与 §1.3/§4/§9.3 结论一致（§1.2 记录 (F) 记号勘误，按代码语义实现），满足覆盖门禁 `coverage_check.py --proto` 的派生要求，不重复产出 |
| `fla_chunk_intra.py` | ⬜ 待实现 | `fla_chunk_intra_fwd_k_inter/_k_g/_k_g2/_k_y/_k_ai` + `fla_chunk_intra_bwd_k_intra`（Developer 模式，§4.7 pass_configs）+ host 包装 `fla_chunk_intra_fwd_inter_solve_fused` / `fla_chunk_intra_bwd_intra`（上游命名；ws 分配 + BetaFlat view + bwd 后处理 out-of-place） |
| `test_fla_chunk_intra.py` | ⬜ 待实现 | `prepare_input_fwd/bwd` + golden 两函数（§9.1 逐行落地，fp64 域）+ `check_precision`（precision-standard.md §4.1 原样复制）+ L0 七用例（§9.2，带覆盖标签）+ L1/L2/Boundary 桩 + `--level` main 分发器 |

### 11.3 命名规范

- kernel 文件 `fla_chunk_intra.py`、测试 `test_fla_chunk_intra.py`（沙箱目录内算子前缀，符合任务硬性约束）
- host 包装函数名对齐上游：`fla_chunk_intra_fwd_inter_solve_fused`（上游 `chunk_kda_fwd_inter_solve_fused` 的沙箱前缀版）/ `fla_chunk_intra_bwd_intra`；JIT kernel 名 `fla_chunk_intra_fwd_k_*` / `fla_chunk_intra_bwd_k_intra`

### 11.4 实现顺序

1. ✅ 设计文档 + L0 门槛测试计划（本文档 §9.2，tilelang-op-test-design 场景 A）
2. ⬜ `fla_chunk_intra.py`：先 bwd 单 kernel（R1/R7 首验：48-gemm 段 + 广播链 + scatter 链），后 fwd 链（k_inter R2 → k_g R4 → k_g2/k_y/k_ai）
3. ⬜ `test_fla_chunk_intra.py`：golden + L0 七用例 + main
4. ⬜ L0 通过（精度收敛；check_precision 全绿）
5. ⬜ 扩展分层套件 L1/L2/Boundary（tilelang-op-test-design 场景 B + coverage_check 门禁）
6. ⬜ perf 哨兵实测（R10 标定：修正 §9.5 推断区间，触发用户口径决策）
