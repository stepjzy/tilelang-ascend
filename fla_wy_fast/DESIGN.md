# fla_wy_fast 算子设计文档（KDA chunked WY 表示，fwd + bwd 双 kernel）

> GPU→Ascend 迁移设计（tilelang-op-migration Phase 2 / Stage 1）。
> 语义权威：`ref_gpu/fla_wy_fast.py`（Triton/FLA，只读）；代码级起点：`ref_gpu/fla_wy_fast_tilelang_fwd.py`、`ref_gpu/fla_wy_fast_tilelang_bwd.py`（上游 TileLang CUDA，只读）。
> 目标硬件：**Ascend 910B3**（本机实测：40 AI Core / 20 Cube Core / L2 192MB / HBM 64GB；片上容量按 A2 级：L1 512KB、UB 192KB、L0A/L0B 64KB、L0C 128KB per core）。
> 本文档区分【已确认事实】（源码/示例/环境验证）、【推断】（标注依据）、【需 Stage 2 验证】三类信息。

---

## 1. 概述

### 1.1 算子名称

`fla_wy_fast`（含 2 个独立 kernel：`fla_wy_fast_fwd` / `fla_wy_fast_bwd`，上游名 `recompute_w_u_fwd` / `prepare_wy_repr_bwd`）

### 1.2 功能描述

KDA (Kimi Delta Attention) chunked WY 表示计算。fwd 从已求逆的 chunk 矩阵 A_c 重构 w/u/kg 三个后续 chunk 计算的输入表示；bwd 计算其反向梯度（dA/dk2/dv/dbeta/dg2）。逐 chunk（BT=64）逐 head 独立计算，chunk 间无数据依赖。

### 1.3 数学公式

记 chunk 长度 BT=64，A_c = A[b, t:t+BT, h, :] 为该 chunk 的 64×64 矩阵（上游 `chunk_delta` 已算好的 (I − tril(diag(β)·A_raw, −1))⁻¹，**本算子只消费不生产**）；g = gk 为逐 token 逐通道累积 log2 衰减；n 为 chunk 内末行索引；strict_tril(M) = 仅保留下三角（i>j，对角与上三角置 0）；⊙ 逐元素乘。

**fwd（逐 chunk 逐 head）**：

$$u_{[t:t+BT,h]} = A_c \cdot (\beta \odot v) \quad [BT, DV]$$
$$w_{[t:t+BT,h]} = A_c \cdot (\beta \odot k \odot 2^g) \quad [BT, DK]$$
$$kg_{[t:t+BT,h]} = k \odot 2^{(g[n,:] - g)} \quad [BT, DK] \;(\text{use\_kg=True})$$

**bwd（记 K_bg = β⊙k⊙2^g，V_b = β⊙v，S = strict\_tril(dw @ K_bgᵀ + du @ V_bᵀ)）**：

$$dA = -\text{strict\_tril}(A_c^T \cdot S \cdot A_c^T) \quad [BT, BT]$$
$$dk2 = (A_c^T \cdot dw) \odot \beta \odot 2^g + dk \quad [BT, DK]$$
$$dv = (A_c^T \cdot du) \odot \beta \quad [BT, DV]$$
$$dbeta = \text{rowsum}((A_c^T dw) \odot k \odot 2^g) + \text{rowsum}((A_c^T du) \odot v) \quad [BT]$$
$$dg2 = K_{bg} \odot (A_c^T \cdot dw) + dg \quad [BT, DK]$$

### 1.4 算法描述（计算步骤分解）

**fwd**（每 block 处理一个 (chunk, b, h)）：
1. 载入 beta[64]、A_c[64,64]、按 block_DV 分块载入 v → UB 内 fp32 域计算 V_b = β⊙v → cast bf16 → L1 → `gemm(A_c, V_b)` → u 写 GM。
2. 按 block_DK 分块载入 k、gk（fp32）→ UB 内 fp32 域计算 2^g = exp(g·ln2)、K_bg = β⊙k⊙2^g → cast bf16 → L1 → `gemm(A_c, K_bg)` → w 写 GM。
3. 同一循环内载入 gk 末行 g[n,:] → KG = k ⊙ exp((g_n−g)·ln2) → cast bf16 → kg 写 GM。

**bwd**（每 block 处理一个 (chunk, b, h)）：
1. 预清零 dA 累加器（跨 i_k/i_v 循环累加）。
2. i_k 循环（DK 分块）：载入 k/gk/dw/dk/dg → UB 内计算 2^g、K_bg → `gemm(dw, K_bg, transpose_B=True, 累加)` → dA；`gemm(A, dw, transpose_A=True)` → X = A_cᵀ@dw → UB 内：dk2 = X⊙β⊙2^g + dk 写 GM；dbeta 部分和 reduce_sum(X⊙k⊙2^g, dim=1, clear=False)；dg2 = K_bg⊙X + dg 写 GM。
3. i_v 循环（DV 分块）：载入 v/du → V_b = β⊙v → `gemm(du, V_b, transpose_B=True, 累加)` → dA；`gemm(A, du, transpose_A=True)` → Y = A_cᵀ@du → dv = Y⊙β 写 GM；dbeta 部分和 reduce_sum(Y⊙v, dim=1, clear=False)。
4. dbeta 写 GM；dA 后处理：L0C→UB → ⊙tril_mask → L1 → `gemm(S, A, transpose_B=True)` → tmp1 = S@A_cᵀ → UB→L1 → `gemm(A, tmp1, transpose_A=True)` → tmp2 = A_cᵀ@S@A_cᵀ → UB → ⊙mask → ⊙(−1) → dA 写 GM。

> **Host 侧 Buffer 操作约束**（ascend-constraints.md §5）：本算子 host 路径只含 (a) 输入张量构造（数据准备）、(b) 下三角常量 mask 的构造（一次性 [64,64] 常量输入，见 §3.5.2）、(c) kernel 调用、(d) 精度验证。无 `.contiguous()`/pad/cat/`reshape` 物理化/aclnn 调用。布局全 bshd 连续，kernel 内直读 GM。

### 1.5 数据流图

```
fwd:
GM[k,v:bf16] ─T.copy→ UB(fp32 域) ─⊙beta─┐
GM[gk:fp32]  ─T.copy→ UB ─exp(·ln2)→ 2^g ┤→ K_bg(fp32) ─cast→ L1(bf16) ┐
GM[beta:bf16]─T.copy→ UB(fp32) ──────────┘                              ├─gemm_v0(A_c,·)→ L0C(fp32)
GM[A:bf16]   ─T.copy→ L1(bf16, A_c) ────────────────────────────────────┘        │
                                                                                  T.copy
GM[v] → UB → V_b=β⊙v → L1(bf16) ──gemm_v0(A_c,·)→ L0C → UB(fp32) → cast bf16 ─T.copy→ GM[u,w]
GM[k,gk] → UB → KG=k⊙exp((g_n−g)ln2) → cast bf16 ──────────────────────T.copy→ GM[kg]

bwd: GM[k,v,beta,gk,A,dw,du,dk,dg] ─T.copy→ UB(fp32 计算域)/L1(gemm 输入)
  i_k 循环: dw@K_bgᵀ ─累加→ L0C[dA]；Aᵀ@dw → L0C[X] → UB → dk2/dg2 写 GM、reduce_sum → UB[dbeta]
  i_v 循环: du@V_bᵀ ─累加→ L0C[dA]；Aᵀ@du → L0C[Y] → UB → dv 写 GM、reduce_sum → UB[dbeta]
  dA 后处理: L0C[dA] → UB ─⊙mask→ L1 → gemm(S,A,ᵀB) → L0C → UB → L1 → gemm(A,·,ᵀA) → L0C → UB ─⊙mask⊙(−1)→ GM[dA]
```

### 1.6 范围与分派覆盖审计

【已确认事实】范围裁剪（proto_fla_wy_fast.yaml 声明）：定长序列（S % 64 == 0）、use_qg=False、use_kg=True、DK/DV 为 16 的倍数（benchmark 配置 128）。varlen（cu_seqlens/chunk_indices）与 use_qg 分支不在本次范围。

```text
[DISPATCH-COVERAGE]
supported_domain: S%64==0; DK,DV ∈ 16 的倍数（实测 64/128）; fwd dtype=(bf16 in/out + fp32 gate), bwd dtype=全 fp32; use_qg=False, use_kg=True, chunk_size=64
generic_fallback: fla_wy_fast_fwd / fla_wy_fast_bwd 各一条通用路径（无 dtype/shape 特化分支；block_DK/block_DV/num_stages 为性能参数，不构成分派谓词）
specializations: 无
unsupported_inputs: varlen、use_qg=True、S%64≠0、DK/DV 非 16 倍数 —— 与 supported_domain 不冲突（proto 已显式排除）
result: pass
```

---

## 2. 编程模式选型

### 2.1 模式结论

**Developer 模式**（用户明确指定，2026-10-21）——`alloc_shared` / `alloc_fragment` + 自动同步 + `threads=2` 消 workspace/vid。

### 2.2 选型理由

算子是典型 CV 融合（Cube GEMM + Vector exp/逐元素乘/rowsum 掩码），计算交织密集但无跨核自定义同步需求；上游 TileLang CUDA 版（`fla_wy_fast_tilelang_fwd.py`）本身即 Developer 风格 API（alloc_shared/fragment + T.gemm/T.copy），迁移映射最直接；带宽敏感型算子应避免 Expert 显式 GM workspace 中转（见 §10.4 被否方案 B）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（编译器映射 L1/UB）+ `T.alloc_fragment`（映射 L0C） |
| GEMM | `T.gemm_v0(A, B, C, transpose_A/B, init)`：A/B 为 shared(L1)，C 为 fragment(L0C) |
| 计算方式 | `T.Parallel` + 符号 API（+,-,*,T.exp）；`T.tile.mul/exp/fill`；`T.reduce_sum`（UB） |
| 作用域 | 编译器自动 C/V 分离（AUTO_CV_COMBINE） |
| 同步方式 | 自动同步（AUTO_SYNC + AUTO_CV_SYNC），无手动 barrier/cross_flag |

---

## 3. API 映射设计

### 3.1 公式拆解（fwd）

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| F1 | `2^g = exp(g·ln2)` | gk fp32 → UB，逐元素（exp2 无昇腾后端支持，见 §3.5.2-5） |
| F2 | `V_b = β ⊙ v` | UB fp32 域，行广播 β |
| F3 | `u = A_c @ V_b` | gemm_v0，M=64, K=64, N=block_DV |
| F4 | `K_bg = β ⊙ k ⊙ 2^g` | UB fp32 域 |
| F5 | `w = A_c @ K_bg` | gemm_v0，N=block_DK |
| F6 | `kg = k ⊙ 2^(g[n,:]−g)` | UB fp32 域，g_n 为 chunk 末行切片 [block_DK] 广播 |

### 3.2 公式拆解（bwd，i_k 循环体）

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| B1 | `2^g, K_bg = β⊙k⊙2^g` | 同 F1/F4 |
| B2 | `dA += dw @ K_bgᵀ` | gemm_v0(dw, K_bg, transpose_B=True, init=False)，跨 i_k/i_v 累加 |
| B3 | `X = A_cᵀ @ dw` | gemm_v0(A, dw, transpose_A=True, init=True) |
| B4 | `dk2 = X⊙β⊙2^g + dk` | L0C[X]→UB 后逐元素（fp32） |
| B5 | `dbeta += rowsum(X⊙k⊙2^g, dim=1)` | reduce_sum(UB, dim=1, clear=False) |
| B6 | `dg2 = K_bg⊙X + dg` | L0C[X]→UB 后逐元素 |
| B7 | `dA += du @ V_bᵀ`；`Y = A_cᵀ@du`；`dv = Y⊙β`；`dbeta += rowsum(Y⊙v)` | i_v 循环体 |
| B8 | `tmp1 = S@A_cᵀ` | S=dA⊙mask 后回 L1；gemm_v0(S, A, transpose_B=True, init=True) |
| B9 | `tmp2 = A_cᵀ@tmp1` | gemm_v0(A, tmp1, transpose_A=True, init=True) |
| B10 | `dA = −(tmp2 ⊙ tril_mask)` | UB 内 T.tile.mul 两次（mask、−1.0） |

### 3.3 TileLang API 映射（含出处）

| 步骤 | TileLang API（昇腾） | 参数 | 出处（已验证） |
|------|---------------------|------|---------------|
| Kernel | `T.Kernel(NT*B*H, threads=2, is_npu=True) as (cid)` | 一维 block；cid 分解 bs=cid%NT, bbh=cid//NT | `examples/developer_mode/flash_attn_bshd_developer.py:47`；threads 限制见 ascend-constraints.md §1 |
| GEMM | `T.gemm_v0(A, B, C, transpose_A=, transpose_B=, init=)` | A/B→shared(L1)，C→fragment(L0C) | `tilelang/language/ascend.py:418`；`examples/linear_attention_and_rnn/gdn/gdn_wy_fast.py:83`；bf16 组合见 `examples/sparse_flash_attention/bench_sfa/sparse_flash_attn_pa_developer.py` |
| 内存分配 | `T.alloc_shared(shape, dtype)` / `T.alloc_fragment(shape, dtype)` | shared→L1/UB、fragment→L0C | flash_attn_bshd_developer.py:52-59 |
| 元素运算 | `T.Parallel` 循环内 `+, -, *, T.exp(x)`；1D→2D 行广播 | 索引须简单变量 | api-compute.md §3；`examples/linear_attention_and_rnn/gdn/gdn_chunk_o.py:90` |
| 逐元素 buffer 级 | `T.tile.mul(dst, src0, src1/scalar)`、`T.tile.fill(buf, 0.0)` | — | api-compute.md §4.1；flash_attn_bshd_developer.py:73-75 |
| 归约 | `T.reduce_sum(buf_ub, out_ub, dim=1, clear=False)` | **限 UB**；clear=False 为 merge 累加 | `tilelang/language/reduce_ascend.py:456-494`（"Must be in UB"）；flash_attn_bshd_developer.py:100 |
| 清零 | `T.clear(fragment)`（L0C）/ `T.tile.fill(ub, 0.0)` | — | `tilelang/language/ascend_tile.py:379` |
| 搬运/隐式 cast | `T.copy(GM, shared)` / `T.copy(fragment, shared)` / 跨 dtype T.copy（fp32↔bf16 自动 cast） | L0C→UB 直拷合法 | flash_attn_bshd_developer.py:82,105-106,121-122 |
| 循环 | `T.Pipelined(T.ceildiv(D, block_D), num_stages=2)`（静态边界） | 预取阶段数 | api-schedule-sync.md §2；flash_attn_bshd_developer.py:78 |
| 下三角掩码 | 常量 `Msk[64,64]` GM 输入 → UB → `T.tile.mul` | host 一次性构造 | `examples/linear_attention_and_rnn/gdn/gdn_chunk_scaled_dot_kkt.py:59,76`（同款 strict-tril 语义） |

### 3.4 计算伪代码（fwd 主体）

```python
LN2 = 0.6931471805599453
with T.Kernel(NT * B * H, threads=2, is_npu=True) as (cid):
    bs, bbh = cid % NT, cid // NT
    bb, bh = bbh // H, bbh % H
    # buffer 声明（详见 §4.3）
    A_l1   = T.alloc_shared([64, 64], "bfloat16")          # L1
    Bt_ub_half = T.alloc_shared([64], "bfloat16")          # beta 载入中转
    Bt_ub  = T.alloc_shared([64], "float32")               # beta (fp32 域)
    T.copy(Beta[bb, bs*64:(bs+1)*64, bh], Bt_ub_half)      # GM bf16 → UB
    T.copy(Bt_ub_half, Bt_ub)                              # 跨 dtype T.copy 隐式 cast → fp32
    T.copy(A[bb, bs*64:(bs+1)*64, bh, :], A_l1)
    for i_v in T.Pipelined(T.ceildiv(DV, block_DV), num_stages=2):
        V_ub = ...  # T.copy 载入 + cast fp32
        for i, j in T.Parallel(64, block_DV):              # V_b = β⊙v（fp32 域）
            Vb_ub[i, j] = V_ub[i, j] * Bt_ub[i]
        T.copy(Vb_ub, Vb_half); T.copy(Vb_half, Vb_l1)     # cast bf16 → L1
        T.gemm_v0(A_l1, Vb_l1, U_frag, init=True)          # u = A_c @ V_b
        T.copy(U_frag, U_ub); T.copy(U_ub, U_half); T.copy(U_half, U[...])
    for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=2):
        K_ub, G_ub = ...                                   # 载入 k(bf16→fp32)、gk(fp32)
        for i, j in T.Parallel(64, block_DK):
            Gexp_ub[i, j] = T.exp(G_ub[i, j] * LN2)        # 2^g 替代 exp2
            Kbg_ub[i, j] = K_ub[i, j] * Bt_ub[i] * Gexp_ub[i, j]
        T.copy(Kbg_ub, Kbg_half); T.copy(Kbg_half, Kbg_l1)
        T.gemm_v0(A_l1, Kbg_l1, W_frag, init=True)         # w = A_c @ K_bg
        T.copy(W_frag, W_ub); T.copy(W_ub, W_half); T.copy(W_half, W[...])
        T.copy(Gk[bb, (bs+1)*64 - 1, bh, ik*block_DK:(ik+1)*block_DK], Gn_ub)  # 末行切片
        for i, j in T.Parallel(64, block_DK):              # kg = k⊙2^(gn−g)
            KG_ub[i, j] = K_ub[i, j] * T.exp((Gn_ub[j] - G_ub[i, j]) * LN2)
        T.copy(KG_ub, KG_half); T.copy(KG_half, KG[...])
```

bwd 伪代码见 §6.2（结构同 §1.4 步骤 2-4）。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（强制检测）

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **是**（上游 `T.Kernel(ceildiv(S,block_S), B*H)` 二维） | 一维化：`T.Kernel(NT*B*H)` + `bs=cid%NT; bbh=cid//NT` 索引分解（gdn_wy_fast.py:34-37 同款） |
| threads 仅 1/2 | **是**（上游 threads=128） | `threads=2`（Developer CV 融合标准配置，消 workspace/vid） |
| 动态循环边界 | 否 | S 定长、D 维循环边界为编译期常量（`T.ceildiv(DK, block_DK)`） |
| Pipelined 动态边界 | 否 | 同上，全部静态 |
| L0C 溢出（128KB） | 已核算 | fwd 峰值 32KB、bwd 峰值 64KB（§4.5 预算表） |
| T.copy 列方向 strided 切片 | 已规避 | 所有 GM 切片均为"行区间 × 完整尾维 block"（`X[bb, s0:s1, bh, d0:d1]` 中 d0:d1 为尾维且连续），无 4D 列向 strided 搬运；Gk 末行切片 `Gk[bb, row, bh, d0:d1]` 是尾维连续 1D 切片 ✓ |
| GEMM 分形限制 | 已核算 | bf16：M,K,N≥16；fp32：M≥16,K≥8,N≥16。全部 GEMM M=64, K=64, N∈{64,128} ✓ |
| GEMM 非整除 | 低风险 | S%64==0（范围约束）；DK/DV 声明为 16 倍数，主选 block_D=64 时 DK=DV∈{64,128} 整除。非整除 D（如 DK=96）留给 Stage 2 用 `T.ceildiv`+valid extent 或裁剪（见 §10.3） |

#### 3.5.2 CUDA 专属机制的昇腾适配（任务点名 8 项）

| # | CUDA 机制（上游） | 昇腾适配结论 | 依据 |
|---|------------------|-------------|------|
| 1 | `T.disable_warp_group_reg_alloc()` | **直接删除**。warp group 是 CUDA SM 专属概念，昇腾无对应物、无此 API | tilelang/language 中无该符号；ascend-constraints §1 "GPU 专用 API" |
| 2 | `@autotune` 的 threads/num_stages 搜索空间 | threads 昇腾只支持 1/2 且 Developer CV 融合固定 2 → 搜索空间删除 threads 维；num_stages 在昇腾为 `T.Pipelined` 预取阶段数（语义保留、限静态边界）；block_DK/block_DV 保留。**L0 阶段不启用 autotune（固定 block_D=64, num_stages=2）**，Stage 3 调优时按 (block_D, num_stages) 枚举 | ascend-constraints §1；api-schedule-sync §2；本仓 `tilelang/autotuner/` 存在、`examples/autotune/` 有先例 |
| 3 | `TL_DISABLE_WARP_SPECIALIZED` pass_config | **不配置**。warp specialization 是 CUDA 后端 pass，昇腾 pipeline 无此 pass | pass_configs 键空间为 TL_ASCEND_*（flash_attn_bshd_developer.py:13-18） |
| 4 | `TL_ENABLE_FAST_MATH` pass_config | **不配置**。CUDA fast math 影响_intrin 精度选择；昇腾 AscendC 指令精度固定（`AscendC::Exp`），无对应开关 | 同上；src/op 中 ascend_exp 映射唯一 |
| 5 | `T.exp2(x)` | **昇腾后端不可用 → 替代 `T.exp(x * LN2)`**。前端 `tilelang/language/tir/op.py:1837` 有定义，但 src/op 仅有 `ascend_exp → AscendC::Exp`（无 exp2 指令映射、无 ln2 重写 pass、examples 无先例）。精度影响：fp32 下 `exp(g·ln2)` 与 `exp2(g)` 相对误差 ~1e-7（ln2 常数舍入 + exp 实现），远小于精度门禁（rtol 2⁻⁶/2⁻¹⁰）| 【已确认事实】源码检索；【推断】误差量级；golden 用 `torch.exp2` 精确对拍可验证 |
| 6 | `T.gemm(shared, shared, fragment, transpose_A/B=, clear_accum=)` | **改用 `T.gemm_v0(A, B, C, transpose_A=, transpose_B=, init=)`**。通用 `T.gemm` 是 CUDA `tl.gemm` intrinsic（tilelang/language/gemm.py:119）；gemm_v0 支持同款 transpose 语义，参数名 `clear_accum`→`init`，A/B 须 L1(shared)、C 须 L0C(fragment) | ascend.py:418-463；gdn_wy_fast.py:83（transpose_B）、上游 bwd 的 4 种转置组合逐一映射见 §3.2 |
| 7 | `T.reduce_sum(fragment→fragment, dim=1, clear=False)` | **不可用（fast-path 限 UB）→ 三步替代**：`T.copy(X_frag, X_ub)`（L0C→UB）→ 元素运算在 UB → `T.reduce_sum(tmp_ub, dbeta_ub, dim=1, clear=False)`。clear=False 的 merge 语义昇腾保留（`new_out = old_out + reduced`） | reduce_ascend.py:471 "Must be in UB"、:478-479 clear 语义；flash_attn_bshd_developer.py:100 UB reduce 先例 |
| 8 | `T.clear(fragment)` / `T.if_then_else(i>j, x, 0)` 掩码 | `T.clear` **可用**（ascend_tile.py:379）；`T.if_then_else` **在 T.Parallel 内不可用**（SIMD 无分支，api-compute §3.1 明确编译错误）→ strict_tril 掩码改 **host 常量 `Msk[64,64]`（值 = tril(ones, -1)）GM 输入 + `T.tile.mul`**，与上游数学语义逐位一致；dbeta 的初始清零用 `T.tile.fill(dbeta_ub, 0.0)` | api-compute.md §3.1 不支持表；gdn_chunk_scaled_dot_kkt.py:59+76（同款 strict 下三角掩码先例） |

#### 3.5.3 参考实现差异说明（GPU → Ascend 汇总）

| 差异项 | 参考实现（CUDA/上游 TileLang） | 本项目（Ascend 910B3） | 转换方案 |
|--------|------------------------------|----------------------|---------|
| Kernel 维度 | 二维 `(ceildiv(S,BS), B*H)` | 一维 `NT*B*H` + 索引分解 | §3.5.1-1 |
| threads | 128（SIMT warp） | 2（C/V 双核） | §3.5.2-2 |
| GEMM API | `T.gemm` + `clear_accum` | `T.gemm_v0` + `init` | §3.5.2-6 |
| exp2 | `T.exp2` | `T.exp(x·ln2)` | §3.5.2-5 |
| fragment reduce | `T.reduce_sum(frag→frag)` | L0C→UB 拷贝后 UB reduce | §3.5.2-7 |
| strict_tril 掩码 | `T.if_then_else` 逐元素 | 常量 mask 输入 + `T.tile.mul` | §3.5.2-8 |
| autotune/pass_configs | `TL_ENABLE_FAST_MATH` / `TL_DISABLE_WARP_SPECIALIZED` | 昇腾四件套 pass_configs（§7.3） | §3.5.2-3/4 |
| 共享内存语义 | shared = SMEM 单层 | shared 分裂 L1/UB 两层，编译器规划 | Developer MEMORY_PLANNING |

#### 3.5.4 本项目同类实现参考（强制步骤 0 结果）

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/linear_attention_and_rnn/gdn/gdn_wy_fast.py` | **高度相似**（同族 WY 表示：A_c @ (β⊙·) 结构、chunk×head grid、gemm_v0 transpose、cv 协作） | 一维 cid 分解、L1/L0C buffer 划分、golden chunk 循环写法（Expert 风格，本设计转 Developer） |
| `examples/linear_attention_and_rnn/gdn/gdn_chunk_scaled_dot_kkt.py` | 高（strict 下三角掩码 + gemm_v0 transpose_B + T.Parallel 内 T.exp） | Msk 常量输入掩码模式（bwd dA 掩码直接复用） |
| `examples/developer_mode/flash_attn_bshd_developer.py` | 高（Developer 模式 CV 融合标准骨架） | threads=2、alloc_shared/fragment、L0C→UB 直拷、跨 dtype T.copy cast、Pipelined+num_stages、UB reduce_sum、四件套 pass_configs |
| `examples/sparse_flash_attention/bench_sfa/sparse_flash_attn_pa_developer.py` | 中（bf16 + gemm_v0 组合实证） | bf16 GEMM 输入路径可行性 |
| `examples/linear_attention_and_rnn/gdn/gdn_chunk_o.py` | 中（gating 逐元素 + chunk 循环） | T.Parallel 内 T.exp 逐元素先例 |

> 注：gdn 系列的 g 为 per-token 标量 [B,H,L]；本算子 gk 为 **per-token per-channel [B,S,H,DK]**（KDA 特性），元素运算量更大但结构同族。gdn 系列为 Expert 风格，本设计按用户要求用 Developer 风格重写（以 flash_attn_bshd_developer.py 为骨架）。

---

## 4. 数据规格与内存规划

### 4.1 输入张量（fwd / bwd 通用，布局全部 bshd 连续）

| 参数名 | Shape | dtype（fwd） | dtype（bwd） | 说明 |
|--------|-------|-------------|--------------|------|
| k | [B, S, H, DK] | bfloat16 | float32 | key |
| v | [B, S, H, DV] | bfloat16 | float32 | value |
| beta | [B, S, H] | bfloat16 | float32 | 门控系数 |
| gk | [B, S, H, DK] | **float32** | float32 | 逐 token 逐通道累积 log2 衰减 |
| A | [B, S, H, 64] | bfloat16 | float32 | chunk 内 (I−tril(βA,−1))⁻¹ |
| Msk | [64, 64] | float32 | float32 | **设计新增**：strict 下三角常量（host 一次性构造，tril(ones,−1)），仅 bwd 使用 |
| dw / du | [B,S,H,DK] / [B,S,H,DV] | —（无） | float32 | 上游梯度 |
| dk / dg | [B, S, H, DK] | —（无） | float32 | 入参部分梯度（累加项） |

### 4.2 输出张量

| 参数名 | Shape | dtype（fwd） | dtype（bwd） |
|--------|-------|-------------|--------------|
| w / kg | [B, S, H, DK] | bfloat16 | — |
| u | [B, S, H, DV] | bfloat16 | — |
| dA | [B, S, H, 64] | — | float32 |
| dk2 / dg2 | [B, S, H, DK] | — | float32 |
| dv | [B, S, H, DV] | — | float32 |
| dbeta | [B, S, H] | — | float32 |

### 4.3 中间缓冲区（fwd，block_S=64, block_DK=block_DV=64 主选）

| Buffer | Shape | dtype | 层级 | 用途 |
|--------|-------|-------|------|------|
| A_l1 | [64, 64] | bf16 | L1(shared) | gemm 左输入 A_c |
| Vb_l1 / Kbg_l1 | [64, 64] | bf16 | L1(shared) | gemm 右输入 |
| Bt_ub_half | [64] | bf16 | UB(shared) | beta 载入中转（GM bf16 → UB） |
| Bt_ub | [64] | fp32 | UB(shared) | beta（fp32 域，T.copy 隐式 cast） |
| K_ub / V_ub | [64, 64] | fp32 | UB(shared) | k/v 载入 + 元素计算域 |
| G_ub / Gexp_ub | [64, 64] | fp32 | UB(shared) | gk / 2^g（Gexp 可 in-place 复用 G） |
| Gn_ub | [64] | fp32 | UB(shared) | gk 末行切片 |
| Kbg_ub / Vb_ub / KG_ub | [64, 64] | fp32 | UB(shared) | fp32 中间结果 |
| *_half | [64, 64] | bf16 | UB(shared) | cast 中转（fp32→bf16） |
| U_frag / W_frag | [64, 64] | fp32 | L0C(fragment) | gemm 累加器 |
| U_ub / W_ub | [64, 64] | fp32 | UB(shared) | L0C→UB 输出中转（可复用） |

### 4.4 内存搬运路径

```
fwd: GM[bf16 k,v,beta,A] --T.copy--> UB[bfp32 域] --元素运算--> UB[fp32 结果]
     --T.copy cast--> UB[bf16 half] --T.copy--> L1[bf16 gemm 输入]
     GM[fp32 gk] --T.copy--> UB[fp32]
     L1[A_c, K_bg/V_b] --gemm_v0--> L0C[fp32] --T.copy--> UB[fp32] --T.copy cast--> UB[bf16] --T.copy--> GM[bf16 w,u,kg]
bwd: 同构（全 fp32，无 cast；fragment 结果经 UB 中转做元素运算/reduce 后回 L1 或写 GM）
```

> 层级合法性：无 GM→L0 直搬；所有 Cube 输入经 L1（gemm_v0 内部处理 L1→L0A/L0B）；Vector 计算与 reduce 全在 UB；L0C→UB 用 T.copy 直连（Developer 片上直连，flash_attn_bshd_developer.py:82 先例）。

### 4.5 片上内存预算表（910B3：L1=512KB、UB=192KB、L0A/L0B=64KB、L0C=128KB per core）

**fwd（bf16 in/out + fp32 gate；block_S=64, block_DK=block_DV=64）**

| 层级 | 构成 | 大小 | 限额 | 结论 |
|------|------|------|------|------|
| L0C | U_frag 16KB + W_frag 16KB（顺序复用可再省） | **32KB** | 128KB | ✓ 25% |
| L1 | A 8KB + K 8KB + V 8KB + Kbg_l1 8KB + Vb_l1 8KB | **40KB** | 512KB | ✓ 8% |
| UB 峰值 | Bt 0.25 + G/Gexp 16 + K_ub 16 + V_ub 16 + Gn 0.25 + Kbg/Vb/ KG fp32 16×2 + half 8 + 输出 fp32 中转 16 | **~105KB** | 192KB | ✓ 55%（Gexp 复用 G、half/中转按 lifetime 复用后更低，MEMORY_PLANNING 自动处理） |

**fwd 备选 block_DK=block_DV=128**（单遍 D 维，减少循环次数）：L0C = 64KB ✓；L1 = 72KB ✓；UB 峰值 ≈ 192KB **临界**（G 32KB + K_ub 32KB + fp32 中转 32KB + KG 32KB + …）→ 列为 Stage 3 调优候选，需显式 buffer 复用；**L0/Stage 2 主选 64**。

**bwd（全 fp32；block_S=64, block_DK=block_DV=64）**

| 层级 | 构成 | 大小 | 限额 | 结论 |
|------|------|------|------|------|
| L0C | dA_frag 16 + dA_tmp1 16 + dA_tmp2 16 + dkbg_frag 16（dvb_frag 与 dkbg 顺序复用同槽） | **64KB** | 128KB | ✓ 50% |
| L1 | A 16 + dw 16 + du 16 + K_bg 16 + V_b 16 + dA_l1(掩码后回写) 16 | **96KB** | 512KB | ✓ 19% |
| UB 峰值 | Bt 0.25 + GK/Gexp 16 + K_ub 16 + V_ub 16 + X_ub(L0C 中转) 16 + dk_old 16 + dg_old 16 + reduce_tmp 16 + dbeta 0.25 + dA_ub 16 + msk_ub 16 + 输出中转 16 | **~161KB** | 192KB | ✓ 84%（**紧张**：依赖 MEMORY_PLANNING lifetime 复用；若溢出，dk_old/dg_old 可与输出中转分时复用，或 block_D 降 32） |

**bwd 备选 block_DK=block_DV=128**：L0C 80KB ✓、L1 160KB ✓、UB ~300KB ✗ **超限** → 否决（Stage 3 若要上 128 须拆 i_k/i_v 循环为两次 GM 往返，代价超过收益）。

### 4.6 动态轴定义

无运行时动态轴（S/DK/DV 均为 JIT 编译期常量，随 shape 重编译）。NT = S//64、grid = NT×B×H 编译期展开。

### 4.7 JIT 配置

```python
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 C/V 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # 自动核间同步
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动核内同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # UB/L1 lifetime 规划（bwd 紧张预算的关键）
}
@tilelang.jit(out_idx=[...], pass_configs=PASS_CONFIGS)       # 无 workspace_idx（Developer 消 workspace）
```

---

## 5. Tiling 策略

### 5.1 计算类型

**混合（CV 融合）**：Cube（gemm_v0，5~6 处/block）+ Vector（exp、逐元素乘、广播、reduce_sum、掩码乘）。

### 5.2 Block 划分

```python
block_S = 64    # 固定 = chunk_size（语义约束，assert chunk_size == block_S）
block_DK = 64   # 主选；候选 128（fwd 调优）、32（bwd UB 溢出降级）
block_DV = 64   # 同上
block_num = (S // 64) * B * H    # fwd bench: 128×64=8192；bwd bench: 512×8=4096；40 核负载均衡 ✓
```

**block_D=64 选型理由**：① L0C 分形 N≥16 ✓、bf16 K≥16 ✓、fp32 K≥8 ✓；② fwd/bwd UB 预算均安全（§4.5），128 在 bwd UB 超限；③ DK=DV=128 时循环仅 2 次迭代，Pipelined num_stages=2 恰好覆盖；④ 与上游 autotune 最优域（32/64）一致。

### 5.3 约束分析

- 对齐：bf16 尾轴 64×2B=128B ≥ 32B（UB/L1 对齐）✓；fp32 64×4B=256B ✓；L0A/L0B 512B 分形按 gemm_v0 内部处理 ✓
- L0C：fwd 32KB / bwd 64KB ≤ 128KB ✓
- GEMM 非整除：S%64==0（范围约束）；DK/DV=64 或 128 被 block_D=64 整除 ✓

### 5.4 注意事项（尾块）

S 维无尾块（范围约束 S%64==0）。D 维：本次声明域 DK/DV ∈ 16 的倍数；**block_D=64 下 DK=96 之类的非整除 D 不在 L0**，Stage 2 扩展 L1 时用 `T.ceildiv(DK, block_DK)` + 输入/输出两侧 valid extent 切片（`T.copy(K[..., d0:d0+valid], ...)`，参考 `examples/grouped_gemm/example_grouped_gemm_fwd.py:107-108` 模式），禁止 host padding。

### 5.5 数据搬运性能可行性（REORDER-COST）

本算子无数据重排（bshd 连续直读、无 permute/gather），全部搬运为规则 tile 块 DMA：

```text
[REORDER-COST]
path: fwd 主路径 (block_D=64)
gm_passes: 每张量 1 读 + 1 写（唯一 IO 961 MiB @bench shape）
dma_transactions: ~13 次/block（beta 1 + A 1 + V 2 + K 2 + G 2 + 写 U/W/KG 各 ~2），8192 blocks ≈ 10.5 万次
average_dma_bytes: ~10KB/次（block 级）
gm_scalar_accesses: 0（无标量 GM 访问；所有元素运算在 UB）
address_div_mod_per_element: 0（cid 分解每 block 3 次）
active_cores / serial_tasks_per_core: 40 / ~205 blocks/core
largest_case_timeout_gate: fwd bench shape 单 case 60s（§9.4）
result: pass

path: bwd 主路径 (block_D=64)
gm_passes: 每张量 1 读 + 1 写（唯一 IO 1410 MiB @bench shape）
dma_transactions: ~18 次/block（9 张量读 + 5 写 + Msk 1[192KB L2 常驻，HBM 仅 1 次]），4096 blocks ≈ 7.4 万次
average_dma_bytes: ~19KB/次
gm_scalar_accesses: 0
active_cores / serial_tasks_per_core: 40 / ~102 blocks/core
largest_case_timeout_gate: bwd bench shape 单 case 120s（§9.4）
result: pass
```

---

## 6. 循环与调度结构

### 6.1 循环结构总结

**fwd**

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| (chunk, b, h) | block 级并行 | `T.Kernel(NT*B*H)` | 每 block 独立一个 chunk-head，无依赖 |
| DV 维 | 迭代+预取 | `T.Pipelined(DV//64, num_stages=2)` | 掩盖 GM→UB 搬运延迟 |
| DK 维 | 迭代+预取 | `T.Pipelined(DK//64, num_stages=2)` | 同上；kg/w 同循环（共享 K/G 载入） |
| 元素级 | 向量化 | `T.Parallel(64, 64)` | SIMD 元素运算（2D 并行上限） |

**bwd**

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| (chunk, b, h) | block 级并行 | `T.Kernel(NT*B*H)` | 同上 |
| DK 维 | 迭代+预取 | `T.Pipelined(DK//64, num_stages=2)` | dA 累加器跨迭代保持（init=False + 预清零） |
| DV 维 | 迭代+预取 | `T.Pipelined(DV//64, num_stages=2)` | 接续累加 dA |
| 元素级 | 向量化 | `T.Parallel(64, 64)` | SIMD |

### 6.2 bwd 循环伪代码（关键差异部分）

```python
with T.Kernel(NT * B * H, threads=2, is_npu=True) as (cid):
    # ... buffer 声明（§4.5 bwd 表）...
    T.clear(dA_frag)                       # 跨 i_k/i_v 累加器清零（替代上游 T.clear，昇腾可用）
    T.tile.fill(dbeta_ub, 0.0)             # dbeta 累加器（UB）
    T.copy(Msk[:, :], msk_ub)              # 常量掩码（L2 常驻，仅首次 HBM 读）
    T.copy(A[...], A_l1); T.copy(Beta[...], Bt_ub)
    for i_k in T.Pipelined(DK // 64, num_stages=2):
        # 载入 K/GK/dw/dk/dg → UB；Gexp = T.exp(G_ub*LN2)；Kbg = K*beta*Gexp
        # Kbg → L1；dw → L1
        T.gemm_v0(dw_l1, Kbg_l1, dA_frag, transpose_B=True, init=False)   # dA += dw@K_bgᵀ
        T.gemm_v0(A_l1, dw_l1, X_frag, transpose_A=True, init=True)       # X = A_cᵀ@dw
        T.copy(X_frag, X_ub)
        # dk2 = X⊙β⊙2^g + dk_old → 写 GM；dg2 = K_bg⊙X + dg_old → 写 GM（T.Parallel fp32）
        # reduce_tmp = X⊙k⊙2^g；T.reduce_sum(reduce_tmp, dbeta_ub, dim=1, clear=False)
    for i_v in T.Pipelined(DV // 64, num_stages=2):
        # 载入 V/du；Vb = v⊙β → L1；du → L1
        T.gemm_v0(du_l1, Vb_l1, dA_frag, transpose_B=True, init=False)    # dA += du@V_bᵀ
        T.gemm_v0(A_l1, du_l1, Y_frag, transpose_A=True, init=True)       # Y = A_cᵀ@du
        T.copy(Y_frag, Y_ub)
        # dv = Y⊙β → 写 GM；reduce_tmp2 = Y⊙v；T.reduce_sum(..., dbeta_ub, dim=1, clear=False)
    T.copy(dbeta_ub, dbeta[...])
    # dA 后处理（§1.4 步骤 4）：
    T.copy(dA_frag, dA_ub); T.tile.mul(dA_ub, dA_ub, msk_ub)   # strict_tril 掩码
    T.copy(dA_ub, dA_l1)
    T.gemm_v0(dA_l1, A_l1, tmp1_frag, transpose_B=True, init=True)  # tmp1 = S@A_cᵀ
    T.copy(tmp1_frag, dA_ub); T.copy(dA_ub, dA_l1)
    T.gemm_v0(A_l1, dA_l1, tmp2_frag, transpose_A=True, init=True)  # tmp2 = A_cᵀ@S@A_cᵀ
    T.copy(tmp2_frag, dA_ub)
    T.tile.mul(dA_ub, dA_ub, msk_ub); T.tile.mul(dA_ub, dA_ub, -1.0)
    T.copy(dA_ub, dA[...])
```

### 6.3 流水线优化

`T.Pipelined(num_stages=2)`（核内预取）：D 维循环仅 2 次迭代（DK=DV=128, block=64），num_stages=2 恰好双缓冲。**核间流水线（Cube/Vector 交织）与核内流水线不能同时开启**——由 AUTO_CV_COMBINE 自动编排，源码层不显式干预。核间流水线收益分析留给 Stage 3（带宽主导型算子，流水线主要保证搬运不被计算阻塞，非吞吐倍增器）。

### 6.4 尾块处理

见 §5.4（S 维无尾块；D 维非整除留给 L1 扩展用 valid extent）。

---

## 7. 同步策略

### 7.1 同步模式

**自动同步**（Developer 模式标准）：AUTO_SYNC 处理核内 DMA/计算依赖，AUTO_CV_SYNC 处理 Cube↔Vector 核间依赖。源码无任何 `T.barrier_all` / `T.set_flag` / `T.set_cross_flag`。

### 7.2 同步点说明（编译器自动插入，设计仅声明数据依赖）

| 依赖 | 说明 |
|------|------|
| UB 元素运算 ← GM 载入 | AUTO_SYNC |
| gemm_v0 ← L1 输入就绪 | AUTO_SYNC（C 核内 barrier） |
| UB 元素运算 ← L0C 结果（T.copy(X_frag, X_ub)） | AUTO_CV_SYNC（跨核） |
| L1 gemm 输入 ← UB cast 结果 | AUTO_CV_SYNC |
| GM 写 ← UB 结果 | AUTO_SYNC |

### 7.3 pass_configs 配置

见 §4.7（四件套全开）。

---

## 8. CV 融合设计

### 8.1 融合算子判定

**是**（Cube GEMM + Vector exp/逐元素/reduce/掩码）。

### 8.2 CV 交互设计（Developer 模式，按 design-template §8.2）

- `T.Kernel(block_num, threads=2, is_npu=True) as (cid)`：单轴 + threads=2，**无 workspace_idx、无 workspace_* 参数、无 vid 偏移**
- Cube↔Vector 片上直连：`T.copy(fragment, shared_ub)` / `T.copy(shared_ub, l1_shared)`，中转与同步交给四个 pass
- 模板：tilelang-programming-model-guide mode-examples.md §6；已验证参考 `examples/developer_mode/sparse_flash_attn_developer_vid_reduce.py`

### 8.3/8.4 Cube 与 Vector 核计算流程

见 §3.4（fwd 伪代码）与 §6.2（bwd 伪代码）：Cube 侧只做 gemm_v0；Vector 侧做 exp/乘法/reduce/掩码；两侧通过 alloc_shared 直连。

### 8.5 pass_configs

见 §4.7。

### 8.6 注意事项

- 核内 Pipelined 与核间流水线互斥（由 pass 自动决策）
- threads=2 是消 vid 前提，消 vid 是消 workspace 前提（本设计全链成立）
- bwd UB 预算 84% 依赖 MEMORY_PLANNING 的 lifetime 复用；若 Stage 2 编译报 UB 溢出，首选 block_D 降 32，其次手动合并 dk_old/dg_old 与输出中转的 buffer

---

## 9. 验证方案

### 9.1 Golden 函数（PyTorch chunk 循环实现，供 Stage 2 直接落地）

> 布局 bshd 与 kernel 一致；**golden 用 `torch.exp2` 精确语义**，kernel 侧 `exp(g·ln2)` 的实现差异计入混合容差验收。fp32 计算域，输出 cast 回原 dtype。NPU 环境 Triton 不可用（ref_gpu/fla_wy_fast.py 仅作语义对照）。

```python
import torch


def fla_wy_fast_fwd_ref(k, v, beta, gk, A, use_kg=True):
    """fwd golden。k[B,S,H,DK]/v[B,S,H,DV]/beta[B,S,H] bf16, gk fp32, A[B,S,H,C] bf16
    -> (w, u, kg) 与输入同 dtype（bf16）。C = A.shape[-1] = 64。"""
    B, S, H, DK = k.shape
    DV = v.shape[-1]
    C = A.shape[-1]
    assert S % C == 0, "S must be divisible by chunk size"
    NT = S // C
    k_f, v_f = k.float(), v.float()
    b_f, g_f, A_f = beta.float(), gk.float(), A.float()
    w = torch.empty(B, S, H, DK, dtype=k.dtype)
    u = torch.empty(B, S, H, DV, dtype=v.dtype)
    kg = torch.empty(B, S, H, DK, dtype=k.dtype) if use_kg else None
    for i in range(NT):
        sl = slice(i * C, (i + 1) * C)
        A_c = A_f[:, sl, :, :]                                  # [B, C, H, C]
        b_c = b_f[:, sl, :]                                     # [B, C, H]
        k_c, v_c, g_c = k_f[:, sl], v_f[:, sl], g_f[:, sl]
        # u = A_c @ (beta ⊙ v)
        V_b = v_c * b_c.unsqueeze(-1)
        u[:, sl] = torch.einsum("bmhj,bjhd->bmhd", A_c, V_b).to(v.dtype)
        # w = A_c @ (beta ⊙ k ⊙ 2^g)
        pow2 = torch.exp2(g_c)                                  # [B, C, H, DK]
        K_bg = k_c * b_c.unsqueeze(-1) * pow2
        w[:, sl] = torch.einsum("bmhj,bjhd->bmhd", A_c, K_bg).to(k.dtype)
        if use_kg:
            g_n = g_f[:, (i + 1) * C - 1, :, :]                 # chunk 末行 [B, H, DK]
            kg[:, sl] = (k_c * torch.exp2(g_n.unsqueeze(1) - g_c)).to(k.dtype)
    return w, u, kg


def fla_wy_fast_bwd_ref(k, v, beta, gk, A, dw, du, dk, dg):
    """bwd golden。全 fp32 -> (dA, dk2, dv, dbeta, dg2) 全 fp32。"""
    B, S, H, DK = k.shape
    DV = v.shape[-1]
    C = A.shape[-1]
    assert S % C == 0
    NT = S // C
    k_f, v_f = k.float(), v.float()
    b_f, g_f, A_f = beta.float(), gk.float(), A.float()
    dw_f, du_f, dk_f, dg_f = dw.float(), du.float(), dk.float(), dg.float()
    dA = torch.empty(B, S, H, C)
    dk2 = torch.empty(B, S, H, DK)
    dv = torch.empty(B, S, H, DV)
    dbeta = torch.empty(B, S, H)
    dg2 = torch.empty(B, S, H, DK)
    for i in range(NT):
        sl = slice(i * C, (i + 1) * C)
        A_c = A_f[:, sl]                                        # [B, C, H, C]
        b_c = b_f[:, sl]                                        # [B, C, H]
        k_c, v_c, g_c = k_f[:, sl], v_f[:, sl], g_f[:, sl]
        dw_c, du_c = dw_f[:, sl], du_f[:, sl]
        dk_c, dg_c = dk_f[:, sl], dg_f[:, sl]
        pow2 = torch.exp2(g_c)
        K_bg = k_c * b_c.unsqueeze(-1) * pow2
        V_b = v_c * b_c.unsqueeze(-1)
        # S = strict_tril(dw @ K_bg^T + du @ V_b^T)
        S = (torch.einsum("bmhd,bjhd->bmhj", dw_c, K_bg)
             + torch.einsum("bmhd,bjhd->bmhj", du_c, V_b))
        S = torch.tril(S, -1)
        # X = A_c^T @ dw ; Y = A_c^T @ du
        X = torch.einsum("bjhi,bjhd->bihd", A_c, dw_c)
        Y = torch.einsum("bjhi,bjhd->bihd", A_c, du_c)
        dk2[:, sl] = X * b_c.unsqueeze(-1) * pow2 + dk_c
        dv[:, sl] = Y * b_c.unsqueeze(-1)
        dbeta[:, sl] = (X * k_c * pow2).sum(-1) + (Y * v_c).sum(-1)
        dg2[:, sl] = K_bg * X + dg_c
        # dA = -strict_tril(A_c^T @ S @ A_c^T)
        T1 = torch.einsum("bjhi,bmhj->bihj", A_c, S)            # A_c^T @ S
        T2 = torch.einsum("bihj,bnjh->binh", T1, A_c)           # @ A_c^T
        dA[:, sl] = -torch.tril(T2, -1)
    return dA, dk2, dv, dbeta, dg2
```

输入数据构造（对齐上游 `prepare_input`）：`torch.randn` 全量（含 beta/A 也用 randn——上游即如此，A 非真实逆矩阵，本算子只消费不生产，randn 对拍成立）；gk 用 `torch.randn`（fp32）。

### 9.2 L0 门槛测试计划（tilelang-op-test-design 场景 A，仅 L0）

> Stage 1 只产出 L0（规则 shape，S%64==0 且 DK/DV 被 block_D=64 整除）。L1（功能，含不规则 shape）/L2（异常）/Boundary（特殊值）由 Stage 2 在 L0 通过后按场景 B 扩展。

| 用例名 | 级别 | Shape (B,S,H,DK,DV) | dtype | 说明 |
|--------|------|---------------------|-------|------|
| l0_fwd_min | L0 | (1, 128, 2, 64, 64) | in/out bf16 + gate fp32 | 最小冒烟（NT=2, BH=2, 4 blocks），覆盖 fwd 全数据流（u/w/kg 三输出、exp 替代、cast 链） |
| l0_fwd_repr | L0 | (2, 256, 4, 128, 128) | in/out bf16 + gate fp32 | benchmark 结构代表：DK=DV=128 → D 维 Pipelined 2 次迭代，bf16 gemm 路径 |
| l0_bwd_min | L0 | (1, 128, 2, 64, 64) | 全 fp32 | 最小冒烟，覆盖 bwd 全 5 输出 + 4 种 gemm 转置组合 + tril 掩码 + reduce_sum 累加 |
| l0_bwd_repr | L0 | (2, 256, 4, 128, 128) | 全 fp32 | benchmark 结构代表，fp32 gemm + 双循环累加 |

**golden 对照法**：每用例对每个输出张量独立调用 `check_precision(out, ref, dtype)`（precision-standard.md §4.1 的混合容差判定，整段复制不改动）；fwd 判 w/u/kg（bf16 档），bwd 判 dA/dk2/dv/dbeta/dg2（fp32 档）。全部通过打 `[PRECISION_PASS]`，任一失败 `[PRECISION_FAIL]`。

### 9.3 精度标准（混合容差，precision-standard.md 口径）

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|-------|------|------|---------------------|------------------------|
| bfloat16（fwd 输出 w/u/kg） | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |
| float32（bwd 输出 dA/dk2/dv/dbeta/dg2） | 2⁻¹⁶ (1.53e-5) | 2⁻¹⁰ (9.77e-4) | 1e-2 | 0.99 |

> 与 proto_fla_wy_fast.yaml 声明的门禁逐字一致。fwd 的 fp32 gate（gk）为输入中间量，不直接对外，无独立精度行。inputs dtype 全集 = {bfloat16, float32}，与上表两行一致。特殊值（NaN/Inf）位置契约：本算子输入域为 randn 有限值（上游对拍口径），Boundary 特殊值用例由 Stage 2 场景 B 扩展（输入 exp2(g) 理论上可能上溢——g>128 时 2^g=inf，Boundary 用例需含稀疏 inf 并按 mask 结构比对）。

### 9.4 性能可行性哨兵（强制，不可跳过）

| 用例名 | Shape (B,S,H,DK,DV) | dtype/属性 | 覆盖路径 | 单 case timeout | 选择理由 |
|--------|---------------------|------------|----------|-----------------|----------|
| perf_fwd_bench | (1, 8192, 64, 128, 128) | bf16/fp32 gate | fwd 主路径（8192 blocks，最大任务数） | 60s | benchmark 验收 shape；最大 DMA 吞吐压力 |
| perf_bwd_bench | (1, 32768, 8, 128, 128) | 全 fp32 | bwd 主路径（4096 blocks，fp32 Cube 最大压力） | 120s | benchmark 验收 shape；fp32 GEMM + UB 紧张预算联合压力 |

计时口径：NPU 事件计时，warmup=20、rep=10、取均值（对齐 `ref_gpu/test_utils_kda.py do_bench`；torch_npu 侧用 `torch.npu.Event(enable_timing=True)` 等价实现），仅设备侧 kernel 时间。测试数据、随机数与 golden 均在 CPU 完成后 H2D，运行段无 aclnn 依赖。

### 9.5 性能预算分析（910B3，含 1370us 目标可行性结论）

**IO 明细（物理唯一字节，每张量单读单写）**

| 项 | fwd bench (1,8192,64,128,128) | bwd bench (1,32768,8,128,128) |
|----|------------------------------|-------------------------------|
| 读 | k 128 + v 128 + beta 1 + gk 256 + A 64 = **577 MiB** | k/v/gk/dw/du/dk/dg 各 128 + beta 1 + A 64 = **961 MiB** |
| 写 | w 128 + u 128 + kg 128 = **384 MiB** | dA 64 + dk2 128 + dv 128 + dbeta 1 + dg2 128 = **449 MiB** |
| 合计 | **961 MiB ≈ 1.008 GB** | **1410 MiB ≈ 1.478 GB** |
| GEMM FLOPs | 17.2 GFLOP（bf16，8192 chunk-head × 2.1M） | 25.8 GFLOP（fp32，4096 chunk-head × 6.3M） |

**Roofline 对照（910B3）**

| 项 | 数值 | 依据 |
|----|------|------|
| HBM 标称带宽 | ~392 GB/s | 【推断】910B3 公开资料；**Stage 2 需 STREAM 实测校准** |
| 有效带宽（80%） | ~313 GB/s | 【推断】经验值 |
| fwd 带宽下限 | 2571 us（标称）/ 3214 us（80%） | 1.008GB ÷ 带宽 |
| bwd 带宽下限 | 3771 us（标称）/ 4713 us（80%） | 1.478GB ÷ 带宽 |
| fwd GEMM 时间 | ~110 us（@157 TFLOPS bf16 折算） | 计算非瓶颈 |
| bwd GEMM 时间 | ~330 us（fp32=bf16/4）~ 1310 us（fp32=bf16/16，保守） | 【推断】fp32 Cube 吞吐未实测；即便最保守也 < 带宽下限 |
| **合计理论下限** | **~6340 us（标称 100%）/ ~7930 us（80% 有效）** | 带宽主导 |

**⚠️ [DESIGN_ERROR] 级发现：性能目标与 910B3 硬件能力矛盾**

- 目标 `fwd + bwd < 1370 us` 隐含有效带宽 = 2.486GB / 1370us ≈ **1.82 TB/s**
- 910B3 标称带宽 392GB/s → 目标是硬件能力的 **4.6 倍**，物理不可达
- 设计已处理论下限：每张量单读单写（无重排/重复搬运空间）、A 与 gk 片上复用（gk 的 2^g 在 UB 算一次复用 2 处）、Msk 常量 L2 常驻（16KB 共享，HBM 仅读一次）、fwd 的 kg 与 w 共享 K/G 载入。**唯一 IO 已是该算子的 IO 下限，不存在 kernel 层优化能弥合 4.6 倍差距**

**替代路线（需用户决策，不静默简化）**：
1. 验收口径改为**带宽效率**：如 fwd+bwd 合计 ≤ 8.4ms（对应 ~75% 标称带宽效率，含 bwd fp32 GEMM 余量）——建议口径；
2. 确认目标硬件为更高带宽平台（910C / CloudMatrix，HBM 1.6TB/s+ 级）：910B3 环境仅做功能/精度验收，性能数字按带宽比例折算报告；
3. 维持 1370us 字面目标 → 本设计无法承诺，需回退需求。

### 9.6 Host 侧 Buffer 操作审计

```text
[HOST-METADATA-AUDIT]
operation: 输入构造 torch.randn(...).npu()（数据准备）+ Msk 常量构造 torch.tril(torch.ones(64,64),-1).npu()（一次性常量输入）
input_stride -> output_stride: N/A（新建张量，非变换）
shares_storage / same_data_ptr: N/A
aclnn_or_physical_copy: false（randn/tril/ones 为数据生成，非对输入的重排；kernel 调用前后无 .contiguous()/pad/cat/reshape 物理化/.to()）
result: allow
```

---

## 10. 风险点与注意事项

### 10.1 风险清单（高/中/低分级）

| 级别 | 风险 | 影响 | 缓解/验证 |
|------|------|------|-----------|
| **高** | R1 性能目标 1370us 在 910B3 物理不可达（需 1.82TB/s = 4.6× 标称） | 验收口径 | §9.5 替代路线，**需用户决策** |
| **高** | R2 `T.exp2` 无昇腾后端（源码无指令映射/无重写 pass/examples 无先例） | 编译或运行失败 | 设计已替代为 `T.exp(x·LN2)`；若 `T.Parallel` 内符号 `T.exp` 异常，回退 `T.tile.exp` + `T.tile.mul`（buffer 级）；L0 用例先行验证 |
| **高** | R3 bwd UB 峰值 ~161KB/192KB（84%） | 编译溢出 | MEMORY_PLANNING lifetime 复用；降级预案：block_D=32（UB 减半）或合并 dk_old/dg_old 中转槽 |
| **高** | R4 跨 dtype `T.copy` 隐式 cast（bf16↔fp32）round 行为 | 精度 | examples 先例充分（flash_attn_bshd_developer.py:105-106）；L0 对拍；异常时改 `T.tile.cast(dst, src, "CAST_RINT"/"CAST_NONE", count)` 显式转换 |
| 中 | R5 fp32 Cube 吞吐未实测（25.8 GFLOP；若 <19.6 TFLOPS 计算时间 >1.3ms） | bwd 性能 | perf_bwd_bench 哨兵实测；必要时 bwd 拆 dA 后处理为独立轻量 kernel 减 L0C 压力（回退方案，不默认） |
| 中 | R6 `T.reduce_sum(UB[64,64]→UB[64], dim=1, clear=False)` 的 Reduce2D 语义/性能 | bwd dbeta 正确性 | l0_bwd_min 首先验证（clear=False merge 语义源码已确认 reduce_ascend.py:478） |
| 中 | R7 dbeta 数据流由上游 fragment 累加改为 UB 累加 | 精度 | fp32 下数学等价；L0 对拍 dbeta 输出 |
| 中 | R8 `gemm_v0(transpose_A=True)` 先例较 `transpose_B=True` 少 | bwd X/Y/tmp2 计算 | l0_bwd_min 覆盖全部 4 种转置组合；源码 ascend.py:440 声明支持 |
| 低 | R9 `T.Pipelined(num_stages=2)` 在 2 次迭代循环的边界行为 | 性能微小 | 无正确性风险（AUTO_SYNC 兜底） |
| 低 | R10 Msk 常量 L2 命中假设（16KB 共享 × 4096 blocks） | bwd 性能微小 | L2 192MB 必命中；REORDER-COST 已计 |
| 低 | R11 `T.Kernel` grid 上限（fwd 8192 blocks） | 无 | gdn_wy_fast.py 同量级先例（B*H*chunk_num） |

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| `T.reduce_sum` 传 fragment | 直接照抄上游 bwd L226 | 前端报错/未定义行为 | 先 `T.copy(frag, ub)` 再 reduce（本设计已内置） |
| `T.if_then_else` 写进 `T.Parallel` | 照抄上游 dA 掩码 | 编译错误（undefined Variable v_thread） | 常量 Msk + `T.tile.mul`（本设计已内置） |
| `T.gemm` 通用版当昇腾 API 用 | 照抄上游 | `tl.gemm` intrinsic 昇腾不支持 | 全部替换 `T.gemm_v0` |
| threads=128 | 照抄上游 | 编译报错 | threads=2 |
| bf16 元素表达式直接混合 fp32 scalar/buffer | 图省事跳过 fp32 域 | 类型提升行为未定义 | 全 fp32 域计算 + T.copy cast（本设计已内置） |
| host 侧 `gk.pow(2)`/`exp2` 预处理 | 优化幻觉 | aclnn 调用违规 + IO 翻倍 | 禁止（被否方案 E） |

### 10.3 特殊场景处理

- **D 维非整除**（DK/DV 非 64 倍数但 16 倍数，如 96）：不在 L0；Stage 2 L1 扩展用 `T.ceildiv` + 输入/输出 valid extent 切片（§5.4）。非 16 倍数 D 违反分形限制，属不支持域（proto 已声明）。
- **极小 shape**（S=64, NT=1）：grid=B*H，负载少但正确；L0 不含（用 S=128 保证 NT≥2 触发 Pipelined 多迭代路径）。
- **g 大值上溢**（2^g=inf）：Boundary 范畴（Stage 2 扩展，输入混合稀疏 inf，按 precision-standard §3.1 mask 结构比对）。

### 10.4 被否方案（设计决策记录）

| # | 方案 | 否决理由 |
|---|------|---------|
| A | 合并 fwd/bwd 为单 kernel | 上游为两个独立 host API，训练中 fwd/bwd 调用时机与输入集合不同（bwd 多 dw/du/dk/dg 四入参）；合并使 fwd 场景空跑 bwd 分支（+60% 无效 IO，带宽敏感算子不可接受）或引入运行时分支；benchmark 口径也是分别计时。**维持双 kernel。** |
| B | Expert 模式 + 显式 GM workspace 中转（gdn_wy_fast.py 模式） | 每次跨核中转经 GM 读写，唯一 IO 近乎翻倍（带宽受限下致命）；手动 set/wait_cross_flag 编排复杂且用户已指定 Developer；gdn 系列的 VEC_NUM=2 手工切分正是 threads=2 自动化所消除的 |
| C | bwd block_DK=block_DV=128 | UB 预算 ~300KB 超限（§4.5）；拆循环引入二次 GM 往返得不偿失 |
| D | dA 掩码用 kernel 内 compare+select 现算 | 需构造 2D 行/列索引 broadcast buffer，多 3 条向量指令且可读性差；常量 Msk（16KB，L2 常驻）+ `T.tile.mul` 零边际成本，gdn_chunk_scaled_dot_kkt.py 已验证同款 strict-tril 语义 |
| E | host 侧预计算 2^g（对 gk 做 aclnn exp） | 违反 host Buffer 约束（ascend-constraints §5：`.to()`/函数计算触发 aclnn）；且 gk 256MB 读+写翻倍；kernel 内 UB exp 一次、复用两处（K_bg 与 KG）才是最优 |

---

## 11. 交付清单

### 11.1 目录结构

```
examples_experiment/dsa_sparse_finetune/     # 沙箱目录（本算子与 indexer_bwd 共存，互不触碰）
├── DESIGN_fla_wy_fast.md        # 本设计文档（含 L0 门槛测试计划）
├── proto_fla_wy_fast.yaml       # 算子接口规格（承担 proto.yaml 角色，见 §11.2）
├── ref_gpu/                     # 冻结参考（只读）
├── fla_wy_fast.py               # ⬜ 待实现（Stage 2）：双 kernel（@tilelang.jit）
└── test_fla_wy_fast.py          # ⬜ 待实现（Stage 2）：golden（§9.1）+ L0（§9.2）+ main 分发器
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN_fla_wy_fast.md` | ✅ 已完成 | 本文档 |
| `proto_fla_wy_fast.yaml` | ✅ 已完成（任务书载体） | proto.yaml 角色：`operator.inputs[].dtype` 全集 = {bfloat16, float32}（与 §9.3 精度表一致）、attrs 覆盖 chunk_size/use_qg/use_kg、schema 完整——经核对已满足覆盖门禁 `coverage_check.py --proto` 的派生要求，不重复产出 |
| `fla_wy_fast.py` | ⬜ 待实现 | `fla_wy_fast_fwd_kernel` / `fla_wy_fast_bwd_kernel`（Developer 模式，§4.7 pass_configs） |
| `test_fla_wy_fast.py` | ⬜ 待实现 | golden 两函数（§9.1 逐行落地）+ `check_precision`（precision-standard.md §4.1 原样复制）+ L0 四用例（§9.2）+ `--level` main 分发器 |

### 11.3 命名规范

- kernel 文件 `fla_wy_fast.py`、测试 `test_fla_wy_fast.py`（沙箱目录内 fla_wy_fast 前缀，符合任务硬性约束）
- host 包装函数名对齐上游：`fla_wy_fast_fwd` / `fla_wy_fast_bwd`

### 11.4 实现顺序

1. ✅ 设计文档 + L0 门槛测试计划（本文档 §9.2）
2. ⬜ `fla_wy_fast.py`：先 fwd（bf16 + fp32 gate 链路），后 bwd（4 种 gemm 转置 + 掩码 + reduce）
3. ⬜ `test_fla_wy_fast.py`：golden + L0 四用例 + main
4. ⬜ L0 通过（精度收敛；`check_precision` 全绿）
5. ⬜ 扩展分层套件 L1/L2/Boundary（tilelang-op-test-design 场景 B）
6. ⬜ 性能哨兵实测（perf_fwd_bench / perf_bwd_bench；含 STREAM 带宽校准与 §9.5 结论复核）

### 11.5 算子 proto.yaml（覆盖门禁用）

见 §11.2：本算子的 proto 角色由 `proto_fla_wy_fast.yaml` 承担（其 `operator.inputs[].dtype` 全集与本文档 §9.3 精度表 dtype 行一致：bfloat16 / float32；`operator.attrs[].name` 覆盖 chunk_size、use_qg、use_kg 全部关键属性）。

---

## 附：本文档信息类别声明

- 【已确认事实】：§3.3 全部 API 出处（源码行号/examples 路径）、§3.5.2 八项适配结论、§4.5 硬件容量（910B3 实测 40 AI Core/20 Cube/64GB HBM）、§9.3 精度门禁（precision-standard.md 口径）
- 【推断】：§9.5 HBM 带宽 392GB/s 与 Cube 吞吐折算（公开资料，Stage 2 实测校准）、R5 数值
- 【需 Stage 2 验证】：R2 exp 替代精度、R4 跨 dtype cast round 模式、R6 Reduce2D 语义、R8 transpose_A 实测、§9.5 带宽 STREAM 实测

