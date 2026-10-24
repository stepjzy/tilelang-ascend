# fla_chunk_o 算子设计文档（KDA chunked 前向输出 O，单 kernel）

> GPU→Ascend 迁移设计（Stage 1 / tilelang-op-design）。
> 语义权威：`ref_gpu/fla_chunk_o.py`（Triton/FLA `chunk_gla_fwd_kernel_o` / wrapper `chunk_gla_fwd_o_gk`，只读）；代码级起点：`ref_gpu/fla_chunk_o_tilelang_fwd.py`（上游 TileLang CUDA `tilelang_chunk_fwd_o`，只读）；计时口径：`ref_gpu/fla_chunk_o_tilelang_fwd.py` 内嵌 `do_bench`（warmup=10, rep=10, 均值，仅设备侧）。
> 目标硬件：**Ascend 910B3**（20 AI Cores / L2 192MB / HBM 64GB；片上容量按 A2 级：L1 512KB、UB 192KB、L0A/L0B 64KB、L0C 128KB per core；HBM 实测 r+w 带宽 **1259–1273 GB/s**，D2D 大拷贝，取 1263 GB/s——同族 fla_wy_fast Stage 3 实测校准值，修正其 DESIGN 未校准的 392GB/s 假设）。
> 本文档区分【已确认事实】（源码/examples/同族实测验证）、【推断】（标注依据）、【需 Stage 2 验证】三类信息。
> 同族先例：`../tilelang-dev-journal/fla_wy_fast/`（KDA chunked WY 表示，已交付，含工具链缺陷清单与实证 API 链路——本文多处以 ★ 标注引用）。

---

## 1. 概述

### 1.1 算子名称

`fla_chunk_o`（单 kernel：KDA chunked 前向输出 O，上游 Triton 名 `chunk_gla_fwd_kernel_o` / host 名 `chunk_gla_fwd_o_gk`）

### 1.2 功能描述

KDA (Kimi Delta Attention) chunked 注意力的输出侧算子：给定逐 token 累积 log2 衰减 g、chunk 内 attention 矩阵 A、chunk 级隐状态 h，逐 chunk（BT=64）逐 head 独立计算输出 o = inter-chunk 贡献（qg @ h）+ intra-chunk 贡献（tril(A) @ v）。chunk 间 / head 间无数据依赖，纯并行。A 与 h 由上游算子（chunk_intra / chunk_delta_h_fwd）生产，**本算子只消费不生产**。

### 1.3 数学公式

记 chunk 长度 BT=64（S 可整除），c = t // BT 为 chunk 序号，scale = 1/sqrt(DK)，g 为逐 token 逐通道累积 log2 衰减（fp32），A 为 chunk 内 attention 矩阵 [BT, BT]，h 为 chunk 级隐状态 [DK, DV]，tril_incl_diag(M) = 保留下三角含对角（i ≥ j），上三角（i < j）置 0；⊙ 逐元素乘：

$$qg_{[t:t+BT,h]} = (q_{[t:t+BT,h]} \cdot scale) \odot 2^{g_{[t:t+BT,h]}} \quad [BT, DK]$$
$$o_{[t:t+BT,h]} = \underbrace{qg \cdot h_{[c,h]}}_{inter\text{-}chunk,\ \sum_{DK}\ 分块累加} + \underbrace{tril\_incl\_diag(A_{[t:t+BT,h]}) \cdot v_{[t:t+BT,h]}}_{intra\text{-}chunk} \quad [BT, DV]$$

2^g 即 use_exp2=True 语义：`2^g = exp2(g) = exp(g · ln2)`（★昇腾无 exp2 后端，见 §3.5.2-5）。

### 1.4 算法描述（计算步骤分解，每 block 处理一个 (chunk, b, h)）

1. **prologue（V 侧，无 C 依赖）**：载入 A 块 [64,64] bf16 → cast fp32 → 载入下三角含对角常量掩码 Msk（fp32，host 一次性构造）→ `A_mask = A_f32 ⊙ Msk` → cast bf16 备用（上三角置 0）。
2. **i_k 循环（DK 分块，恰好 2 次迭代，T.Pipelined num_stages=2）**：
   - V 侧：载入 q 切片 [64, block_DK]（bf16 → cast fp32）与 g 切片 [64, block_DK]（fp32）→ UB fp32 域计算 `qg = (q ⊙ scale) ⊙ exp(g·ln2)` → cast bf16；
   - V→C 交换：qg(bf16) → L1；A_mask(bf16) → L1（幂等重复）；
   - C 侧：载入 h 切片 [block_DK, DV]（GM→L1 直拷）→ `gemm_v0(qg_l1, h_l1, o_frag, init=(i_k==0))` 累加 inter-chunk 项。
3. **epilogue（C 侧 gemm → V 侧输出，C→V 单跳）**：载入 v 块 [64, DV]（GM→L1）→ `gemm_v0(A_mask_l1, v_l1, o_frag, init=False)` 追加 intra-chunk 项 → o_frag(L0C fp32) → UB(fp32) → cast bf16 → GM o。

> **Host 侧 Buffer 操作约束**（ascend-constraints.md §5）：本算子 host 路径只含 (a) 输入张量构造（数据准备）、(b) Msk 常量构造（一次性 [64,64] 常量输入，per-(device,chunk) 缓存，★fla_wy_fast `_MSK_CACHE` 同款）、(c) kernel 调用、(d) 精度验证。无 `.contiguous()`/`view`/pad/cat/`reshape` 物理化/aclnn 调用（bshd 布局直读，连 view 都不需要）。

### 1.5 数据流图

```
GM[q:bf16]  ─T.copy→ UB[q_half:bf16] ─T.copy cast→ UB[q_ub:fp32] ─┐
GM[g:fp32]  ─T.copy→ UB[g_ub:fp32] ─tile.mul(LN2)→ tile.exp ──────┤→ UB[qg:fp32] = (q⊙scale)⊙exp(g·ln2)
                                                                    └─T.copy cast→ UB[qg_half:bf16] ─T.copy→ L1[qg_l1] ┐
GM[h:bf16]  ─T.copy(5D 切片)→ L1[h_l1:bf16] ───────────────────────────────────────────────────────────────┤→ gemm_v0 → L0C[o_frag:fp32]（i_k 累加）
GM[A:bf16]  ─T.copy→ UB[A_half] ─cast→ UB[A_f32] ⊙ Msk(fp32) ─cast→ UB[A_mask:bf16] ─T.copy→ L1[A_l1] ┐  │
GM[v:bf16]  ─T.copy→ L1[v_l1:bf16] ────────────────────────────────────────────────────────────────────┤→ gemm_v0（epilogue 追加）
L0C[o_frag] ─T.copy→ UB[o_ub:fp32] ─T.copy cast→ UB[o_half:bf16] ─T.copy→ GM[o:bf16]
```

### 1.6 范围与分派覆盖审计

【已确认事实】范围裁剪（proto_fla_chunk_o.yaml 声明 + 用户确认）：定长序列（S % 64 == 0）、use_exp2=True、chunk_size=64；varlen（cu_seqlens/chunk_indices）与 use_exp2=False 分支不在范围。DK/DV 支持域由工具链约束收紧（§3.5.1）。

```text
[DISPATCH-COVERAGE]
supported_domain: S%64==0; DK ∈ {64, 96, 128}（exact-2 Pipelined + 分形 K≥32 联合约束，见 §3.5.1）;
  DV ∈ 16 的倍数且 16 ≤ DV ≤ 128（单块 block_DV=DV，分形 N≥16）; B,H ≥ 1 任意; chunk_size=64 固定;
  dtype 组合固定 = in/out bf16 + gate fp32 + 累加 fp32; use_exp2=True
generic_fallback: fla_chunk_o 单一通用 kernel（无 dtype/shape 特化分支；block_DK/block_DV 由 shape
  机械推导 _pick_block_dk(DK) / block_dv=DV，属性能参数非分派谓词）
specializations: 无
unsupported_inputs: varlen、use_exp2=False、S%64≠0、chunk_size≠64 —— proto 已显式排除;
  DK=32（exact-2 → block_DK=16 违反 L0A/L0B K≥32 分形；单迭代 extent-1 Pipelined 崩溃——工具链
  documented limitation，见 §3.5.1）、DK∉{64,96,128} 的其他 16 倍数（无合法 exact-2 block，同
  fla_wy_fast _pick_block_d 的 documented limitation ★）、DV>128（bv 分块需嵌套 Pipelined，无先例，
  本期不做）、DV 非 16 倍数（分形）——与 supported_domain 不冲突（DESIGN 显式声明 + L2 负向可测）
result: pass
```

---

## 2. 编程模式选型

### 2.1 模式结论

**Developer 模式**（用户明确指定，2026-10-22）——`alloc_shared` / `alloc_fragment` + 自动同步 + `threads=1`。

### 2.2 选型理由

算子是典型 CV 融合（Cube GEMM ×2 + Vector exp/逐元素乘/掩码/cast），无跨核自定义同步需求；上游 TileLang CUDA 版本身就是 Developer 风格 API（alloc_shared/fragment + T.gemm/T.copy），迁移映射最直接；同族 fla_wy_fast（★）已在 910B3 上以 Developer 模式（threads=1）完整交付，其全部实证 API 链路（含工具链缺陷规避）可直接复用。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（编译器映射 L1/UB）+ `T.alloc_fragment`（映射 L0C） |
| GEMM | `T.gemm_v0(A, B, C, init=)`（本算子无 transpose 需求）：A/B 为 shared(L1) bf16，C 为 fragment(L0C) fp32 |
| 计算方式 | `T.tile.mul / T.tile.exp`（buffer 级全 tile 指令，★反模式：T.Parallel 逐元素串行循环 -26%~-71%/kernel） |
| 作用域 | 编译器自动 C/V 分离（AUTO_CV_COMBINE） |
| 同步方式 | 自动同步（AUTO_SYNC + AUTO_CV_SYNC），无手动 barrier/cross_flag |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| F1 | `2^g = exp(g · ln2)` | g fp32 → UB，逐元素（exp2 无昇腾后端，★§3.5.2-5） |
| F2 | `qg = (q ⊙ scale) ⊙ 2^g` | UB fp32 域（q 先 bf16→fp32 cast）；scale 为标量 |
| F3 | `o_inter = qg @ h`，按 block_DK 分块累加 | gemm_v0，M=64, K=block_DK, N=DV，init=(i_k==0) |
| F4 | `A_mask = A ⊙ tril_incl_diag(I)` | fp32 域 tile.mul × 常量 Msk（含对角，diagonal=0） |
| F5 | `o_intra = A_mask @ v` | gemm_v0，M=64, K=64, N=DV，init=False 追加累加 |
| F6 | `o = bf16(o_inter + o_intra)` | L0C fp32 累加器 → UB → cast bf16 → GM |

### 3.2 TileLang API 映射（含出处）

| 步骤 | TileLang API（昇腾） | 参数 | 出处（已验证） |
|------|---------------------|------|---------------|
| Kernel | `T.Kernel(NT*B*H, threads=1, is_npu=True) as (cid)` | 一维 block；cid 分解 bs=cid%NT, bbh=cid//NT, bb=bbh//H, bh=bbh%H | ★fla_wy_fast impl_final.py:130（threads=1 同款）；`examples/linear_attention_and_rnn/gdn/gdn_chunk_o.py:35`（chunk×head 一维化先例） |
| GEMM | `T.gemm_v0(A, B, C, init=)` | A/B→L1 shared bf16，C→fragment fp32；init=(i_k==0) 首块清零 / False 累加 | `tilelang/language/ascend.py:418`（签名实证）；★impl_final.py:181-182（init=True/False 对）；`gdn_chunk_o.py:67`（init=(i==0) 先例）；bf16 组合 ★impl_final fwd_u/wkg |
| 内存分配 | `T.alloc_shared(shape, dtype)` / `T.alloc_fragment(shape, dtype)` | shared→L1/UB、fragment→L0C | ★impl_final.py:137-154 |
| 逐元素（buffer 级） | `T.tile.mul(dst, src0, src1)`，src1 可为标量（LN2/scale/-1.0）；`T.tile.exp(dst, src)` | fp32 域全 tile 指令 | `tilelang/language/ascend_tile.py:1152 (mul) / 1277 (exp)`；★impl_final.py:262-263, 273（mul 链 + exp）；`examples/linear_attention_and_rnn/opt_gdn/opt_gdn_chunk_o.py:100,115`（fp32 tile.mul 掩码先例） |
| 掩码 | 常量 `Msk[64,64]` GM 输入（fp32）→ UB → `T.tile.mul(A_f32, A_f32, msk_ub)` | host 一次性构造 `tril(ones, 0)` 含对角 + per-(device,chunk) 缓存 | ★impl_final.py:913-914（da kernel 同款掩码乘）；`gdn_chunk_o.py:123`（tril(ones, diagonal=0) **含对角**同款常量）；Msk 缓存 ★impl_final.py:931-945 |
| 搬运/隐式 cast | `T.copy(GM, shared 同dtype)` + `T.copy(ub_bf16, ub_fp32)` 两步（**禁止 GM↔UB 跨 dtype 直拷**，cast-on-copy codegen 缺陷 ★）；`T.copy(frag, ub_fp32)` L0C→UB 直连 | 跨 dtype 只在 UB↔UB 内做 | ★impl_final.py:157-160, 168, 183-185, 269-274（q 载入 cast 链 / 输出三步链全部实证）；缺陷记录 ★final_summary §8-6/8-7 |
| 循环 | `T.Pipelined(DK // block_DK, num_stages=2)`（静态边界，**恰好 2 次迭代**） | 预取阶段数=2 | ★impl_final.py:162, 251（extent-2 Pipelined 全部 kernel 同款）；extent 约束见 §3.5.1-3 |
| 清零 | gemm init 参数（本算子无独立 T.clear 需求：o_frag 首个 gemm init=True 即清零） | — | ★impl_final.py:181 |

### 3.3 计算伪代码（kernel 主体）

```python
LN2 = 0.6931471805599453   # 2^g = exp(g*ln2)，exp2 无昇腾后端

@tilelang.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)   # 四件套见 §4.7
def fla_chunk_o_kernel(B, S, H, DK, DV, chunk_size=64, scale=1/(DK**0.5),
                       block_DK=None, block_DV=None):
    BS = chunk_size; NT = S // BS
    bDK = block_DK or _pick_block_dk(DK)   # DK//2，强制 extent=2：64→32 / 96→48 / 128→64
    bDV = block_DV or DV                   # 单块（DV ≤ 128），无 bv 分块 → IO 唯一
    GRID = NT * B * H

    @T.prim_func
    def main(
        Q: T.Tensor((B, S, H, DK), "bfloat16"),        # [B,S,H,DK]
        V: T.Tensor((B, S, H, DV), "bfloat16"),
        G: T.Tensor((B, S, H, DK), "float32"),         # 逐 token 累积 log2 衰减
        A: T.Tensor((B, S, H, BS), "bfloat16"),        # chunk 内 attention
        Hstate: T.Tensor((B, NT, H, DK, DV), "bfloat16"),  # chunk 级隐状态
        Msk: T.Tensor((BS, BS), "float32"),            # 设计新增常量：tril(ones, 0) 含对角
        O: T.Tensor((B, S, H, DV), "bfloat16"),
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % NT; bbh = cid // NT
            bb = bbh // H; bh = bbh % H
            s0 = bs * BS

            # ---- buffer 声明（层级由编译器规划，命名指示预期归属，§4.3）----
            A_l1   = T.alloc_shared([BS, BS], "bfloat16")      # gemm 左输入（掩码后）
            V_l1   = T.alloc_shared([BS, bDV], "bfloat16")     # gemm 右输入
            H_l1   = T.alloc_shared([bDK, bDV], "bfloat16")    # gemm 右输入（i_k 切片）
            qg_l1  = T.alloc_shared([BS, bDK], "bfloat16")     # gemm 左输入（i_k 切片）
            a_half = T.alloc_shared([BS, BS], "bfloat16")      # A 载入中转
            a_f32  = T.alloc_shared([BS, BS], "float32")       # A fp32 域
            msk_ub = T.alloc_shared([BS, BS], "float32")       # 掩码常量
            a_mask = T.alloc_shared([BS, BS], "bfloat16")      # 掩码结果 bf16
            q_half = T.alloc_shared([BS, bDK], "bfloat16")     # q 载入中转
            q_ub   = T.alloc_shared([BS, bDK], "float32")      # q fp32 域
            g_ub   = T.alloc_shared([BS, bDK], "float32")      # g
            gL_ub  = T.alloc_shared([BS, bDK], "float32")      # g*ln2
            gexp_ub= T.alloc_shared([BS, bDK], "float32")      # 2^g
            qg_ub  = T.alloc_shared([BS, bDK], "float32")      # qg fp32
            qg_half= T.alloc_shared([BS, bDK], "bfloat16")     # qg cast 中转
            o_frag = T.alloc_fragment([BS, bDV], "float32")    # L0C 累加器
            o_ub   = T.alloc_shared([BS, bDV], "float32")      # L0C→UB 中转
            o_half = T.alloc_shared([BS, bDV], "bfloat16")     # 输出 cast 中转

            # ---- prologue（V 侧，无 C 依赖；GM→UB DMA + V 内计算）----
            T.copy(A[bb, s0:s0+BS, bh, :], a_half)             # GM bf16 → UB bf16（同 dtype）
            T.copy(a_half, a_f32)                               # UB 内 cast → fp32
            T.copy(Msk[0:BS, 0:BS], msk_ub)
            T.tile.mul(a_f32, a_f32, msk_ub)                    # tril 含对角掩码（fp32 域）
            T.copy(a_f32, a_mask)                               # cast → bf16

            # ---- i_k 循环（inter-chunk，恰好 2 次迭代）----
            for i_k in T.Pipelined(DK // bDK, num_stages=2):
                T.copy(Q[bb, s0:s0+BS, bh, i_k*bDK:(i_k+1)*bDK], q_half)
                T.copy(q_half, q_ub)                            # bf16 → fp32
                T.copy(G[bb, s0:s0+BS, bh, i_k*bDK:(i_k+1)*bDK], g_ub)   # fp32 直拷
                T.tile.mul(gL_ub, g_ub, LN2)                    # g·ln2
                T.tile.exp(gexp_ub, gL_ub)                      # 2^g = exp(g·ln2)
                T.tile.mul(q_ub, q_ub, scale)                   # q ⊙ scale
                T.tile.mul(qg_ub, q_ub, gexp_ub)                # qg = q⊙scale⊙2^g（fp32 域）
                T.copy(qg_ub, qg_half)                          # cast → bf16
                T.copy(qg_half, qg_l1)                          # V→C 交换（循环内，安全模式★）
                T.copy(a_mask, A_l1)                            # 幂等重复（规避 prologue V→C 错版本化★）
                T.copy(Hstate[bb, bs, bh, i_k*bDK:(i_k+1)*bDK, 0:bDV], H_l1)  # 5D 尾维连续切片
                T.gemm_v0(qg_l1, H_l1, o_frag, init=(i_k == 0)) # o += qg @ h

            # ---- epilogue（C gemm → V 输出，C→V 单跳）----
            T.copy(V[bb, s0:s0+BS, bh, 0:bDV], V_l1)            # GM → L1 直拷（无 V 计算）
            T.gemm_v0(A_l1, V_l1, o_frag, init=False)           # o += A_mask @ v
            T.copy(o_frag, o_ub)                                # L0C → UB fp32
            T.copy(o_ub, o_half)                                # cast → bf16
            T.copy(o_half, O[bb, s0:s0+BS, bh, 0:bDV])          # → GM
    return main
```

> 伪代码中 buffer 归属（L1 vs UB）由 AUTO_CV_COMBINE + MEMORY_PLANNING 自动规划；gemm 输入（`*_l1`）必须落在 L1，元素计算 buffer（fp32 域）落在 UB——★impl_final 同款命名约定。

### 3.4 API 可行性确认

全部 API 均有本项目实证出处（§3.2 表逐行标注源码行号 / examples 路径 / ★同族交付代码），**无凭记忆 API**。本算子操作集是 ★fla_wy_fast impl_final 已验证操作的子集（仅新增两类组合，均已核对先例）：
1. **5D GM 切片 T.copy**（`Hstate[bb, bs, bh, dk0:dk1, dv0:dv1]`）：`gdn_chunk_o.py:66` 的 `T.copy(S[bz,by,bx,i*BK,0], s_l1)` 同款（[B,H,chunk,DK,DV] 5D 张量行区间×尾维 block 切片，最内维 DV 连续，行间 gap 由 DataCopyPad 支持，非列方向 strided——ascend-constraints §1 合规）；
2. **Pipelined 循环 + 后置单 gemm epilogue**：无逐字先例（★impl_final 各 kernel 的 epilogue 均为纯 V 侧），但结构上收敛为循环后 **C→V 单跳**（gemm 在 C、输出链在 V），与 ★da kernel 的 C(gemm)→V(mul)→GM 段同构、且比 z+da 合并失败案例（V→C→V→C→V 五跳，final_summary §8-4）简单两级——列为【需 Stage 2 验证】R1，含两级回退（§10.4）。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（强制检测）

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **是**（上游 `T.Kernel(ceildiv(DV,block_DV), ceildiv(S,block_S), B*H, threads=128)` 三维） | 一维化：`T.Kernel(NT*B*H)` + `bs=cid%NT; bbh=cid//NT`（gdn_chunk_o.py:35-38 / ★impl_final 同款）。bv 维消解：block_DV=DV 单块（§10.4-A 论证 IO 收益） |
| threads 仅 1/2 | **是**（上游 threads=128） | `threads=1`（★fla_wy_fast 最终交付同款；threads=2 的 vid 切分在 ★final_summary §7-8 有 gather 越界先例，本算子无 gather 但主选保守值，threads=2 列为 Stage 3 调优候选） |
| 动态循环边界 | 否 | S/DK/DV 均为 JIT 编译期常量；`DK // block_DK` 静态 |
| Pipelined 动态边界 / **extent 约束** ⚠️ | **是**（关键） | ★工具链实证：单 cross-core Pipelined 体 extent=1 崩溃 codegen、extent≥3 双缓冲回卷损坏、**唯一安全 extent=2**（final_summary §8-2；D=96/block32/3 迭代实测 2/3 切片损坏）。本设计 `_pick_block_dk(DK)=DK//2` 强制 i_k 循环恰好 2 次迭代：DK=64→block 32、DK=96→block 48、DK=128→block 64。**DK=32 无合法 exact-2 block（16 违反 L0A/L0B K≥32 分形）→ 不支持（documented limitation，L2 负向测试）**；其他非 {64,96,128} 的 16 倍数同理拒绝（★fla_wy_fast `_pick_block_d` 同款语义） |
| L0C 溢出（128KB） | 已核算 | o_frag [64, DV≤128] fp32 ≤ 32KB = 25% 限额 ✓（§4.5） |
| T.copy 列方向 strided 切片 | 已规避 | 全部 GM 切片为"标量前缀 + 行区间 × 最内维连续 block"：q/g [bb, s0:s0+64, bh, d0:d1]（4D，最内维连续）、A [bb, s0:s0+64, bh, :]、v/o [bb, s0:s0+64, bh, 0:DV]、h [bb, bs, bh, dk0:dk1, dv0:dv1]（5D，最内维 DV 连续，行间 gap 合法）——与 ★impl_final / gdn_chunk_o 全部切片模式同款 |
| GEMM 分形限制 | 已核算 | bf16：M≥16, K≥32(L0A/L0B), N≥16。inter gemm M=64, K=block_DK∈{32,48,64} ✓, N=DV≥16 ✓；intra gemm M=64, K=64, N=DV ✓ |
| GEMM 非整除 | 低风险 | S%64==0（范围约束）；DK 被 block_DK=DK//2 整除（exact-2 构造保证）；DV=block_DV 单块无尾块。**声明域内无非整除路径**；域外 shape（S%64≠0 等）由 L2 负向测试拒绝 |
| **fp32 gemm_v0 损坏** ⚠️（★final_summary §8-1：fp32 A/B 产出 1e37 级交替垃圾） | 已规避 | 本算子全部 gemm 输入天然 bf16（qg cast 后、h、A_mask、v）——**无需 hi/lo 模拟**（★fla_wy_fast bwd 被迫 fp16×3 模拟的场景在本算子不存在） |
| **GM↔UB cast-on-copy 缺陷** ⚠️（★§8-6：bf16↔fp32 跨 dtype GM↔UB 直拷 codegen 模板错配） | 已规避 | 全部跨 dtype 转换在 **UB↔UB 内**两步完成：GM 同 dtype → UB → `T.copy(ub_bf16, ub_fp32)` / `T.copy(ub_f32, ub_bf16)`（★impl_final q 载入链 / 输出链实证） |
| **L0C(fp32)→GM(bf16) 直存损坏** ⚠️（★§8-7：fixpipe 转换损坏） | 已规避 | 输出三步链：`o_frag → o_ub(fp32) → o_half(bf16) → GM`（★impl_final.py:183-188 实证） |
| **serial 循环内 V→C 交换 mis-sync**（★模块 docstring-1） | 已规避 | i_k 循环用 T.Pipelined（不用 T.serial）；V→C 交换（qg_l1/A_l1 拷贝）全部在 Pipelined 体内 |
| **prologue V→C 交换错版本化**（★§8-5：stage 偏移读一次性写入） | 已规避 | prologue 只有 GM→UB DMA + V 内计算（A cast/掩码），**无 V→C 交换**；A_mask→A_l1 交换在循环体内（幂等重复执行，每迭代写同值） |
| host 侧 Buffer 约束（ascend-constraints §5） | 已审计 | §9.6 [HOST-METADATA-AUDIT]：零 view/零 reshape/零 aclnn（bshd 直读） |

#### 3.5.2 CUDA 专属机制的昇腾适配（对照上游 fla_chunk_o_tilelang_fwd.py）

| # | CUDA 机制（上游） | 昇腾适配结论 | 依据 |
|---|------------------|-------------|------|
| 1 | `T.Kernel(ceildiv(DV,block_DV), ceildiv(S,block_S), B*H, threads=128)` 三维 + threads=128 | 一维化 `T.Kernel(NT*B*H, threads=1)`；bv 维以 block_DV=DV 单块消解 | ascend-constraints §1；gdn_chunk_o.py:35；★impl_final.py:130 |
| 2 | `@autotune`（block_DK×block_DV×threads×num_stages 搜索） | **L0 阶段不启用 autotune**：block_DK=DK//2 由 exact-2 约束唯一确定、block_DV=DV 由 IO 唯一化确定、threads=1、num_stages=2——四个参数全部失去搜索空间；Stage 3 调优仅剩 threads∈{1,2} 微调 | ★final_summary §8-2 extent 约束；§10.4-A IO 论证 |
| 3 | `T.exp2(GK_shared[i,j])`（T.Parallel 逐元素内） | **替代**：`T.tile.mul(gL_ub, g_ub, LN2)` + `T.tile.exp(gexp_ub, gL_ub)`（buffer 级全 tile 指令；exp2 无后端映射★；T.Parallel 逐元素是性能反模式★-26%~-71%）。精度影响：fp32 下 exp(g·ln2) 与 exp2(g) 相对差 ~1e-7 << 门禁 2⁻⁶/2⁻¹⁰ | ★impl_final.py:262-263 同款链路 |
| 4 | `T.gemm(GQ_shared, HIDDEN_shared, O_fragment)`（+循环外第二个 T.gemm） | `T.gemm_v0(qg_l1, H_l1, o_frag, init=)` / `T.gemm_v0(A_l1, V_l1, o_frag, init=False)`；**epilogue gemm 为无先例组合**（R1，两级回退） | ascend.py:418；gdn_chunk_o.py:67 init=(i==0) |
| 5 | `T.if_then_else(i_s1 < i_s2, 0, A_shared[i_s1, i_s2])`（T.Parallel 内逐元素掩码） | **T.Parallel 内不可用**（SIMD 无分支，api-compute §3.1 编译错误）→ 常量 `Msk[64,64]`（tril(ones,**0**) 含对角——本算子与 fla_wy_fast 的 strict(-1) 不同，gdn_chunk_o.py:123 为同款 diagonal=0 先例）GM 输入 + `T.tile.mul`（fp32 域） | ★impl_final.py:913-914 da kernel 同款 |
| 6 | `Q_shared[i_s,i_k] = Q_shared[i_s,i_k] * scale`（bf16 buffer 内混合 fp32 标量） | 全 fp32 域计算：q 载入后 cast fp32 → `T.tile.mul(q_ub, q_ub, scale)`（标量 mul 实证）→ qg cast bf16 供 gemm。bf16 域混合标量类型提升行为未定义（★fla_wy_fast §10.2 常见错误） | ★impl_final.py:273 标量 tile.mul（-1.0）先例 |
| 7 | `T.copy(O_fragment, O_shared)`（fragment fp32 → shared bf16 跨 dtype 直拷） | 三步链 `o_frag → o_ub(fp32) → o_half(bf16) → GM`（L0C 跨 dtype 直连未验证且直存损坏★§8-7） | ★impl_final.py:183-188 |
| 8 | `T.clear(O_fragment)` + T.gemm 累加 | `init=(i_k==0)` 参数化（首块覆盖、后续累加；epilogue gemm init=False），无独立 clear | gdn_chunk_o.py:67；★impl_final.py:181-182, 400-403 |

#### 3.5.3 参考实现差异说明（GPU → Ascend 汇总）

| 差异项 | 参考实现（CUDA/上游 TileLang） | 本项目（Ascend 910B3） | 转换方案 |
|--------|------------------------------|----------------------|---------|
| Kernel 维度 | 三维 (bv, bs, bbh) | 一维 NT*B*H（bv 消解） | §3.5.1-1 |
| threads | 128（SIMT warp） | 1（C/V 自动协作） | §3.5.2-1 |
| GEMM API | `T.gemm`（通用 intrinsic） | `T.gemm_v0` + init 参数 | §3.5.2-4 |
| exp2 | `T.exp2`（T.Parallel 内） | `T.tile.exp(x·ln2)`（buffer 级） | §3.5.2-3 |
| tril 掩码 | `T.if_then_else` 逐元素 | 常量 Msk + `T.tile.mul`（fp32 域） | §3.5.2-5 |
| 跨 dtype cast | 元素表达式内隐式 | UB 内两步 copy 链（GM↔UB cast-on-copy 缺陷规避） | §3.5.1 |
| L0C→GM | fragment→shared 直拷 | 三步中转链 | §3.5.2-7 |
| autotune | 4 维搜索空间 | 全参数被约束唯一确定，不启用 | §3.5.2-2 |
| shared 语义 | SMEM 单层 | L1/UB 两层（编译器规划） | Developer MEMORY_PLANNING |
| 元素循环 | T.Parallel(i,j) 逐元素 | T.tile.* 全 tile 指令 | ★perf 教训 |

#### 3.5.4 本项目同类实现参考（强制步骤 0 结果）

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples_experiment/dsa_sparse_finetune/fla_wy_fast.py` 的交付源 `../tilelang-dev-journal/fla_wy_fast/impl_final.py`（★） | **最高（同族同沙箱）** | 全部实证 API 链（cast 链/输出链/gemm init/掩码乘/Pipelined extent-2）、工具链缺陷规避清单（模块 docstring 1-9）、Msk 缓存模式、threads=1 |
| `examples/linear_attention_and_rnn/gdn/gdn_chunk_o.py` | 高（**同功能 GLA chunk O**：chunk×head 一维化 grid、5D 隐状态 S 切片载入、tril(ones,0) 含对角 Msk 常量、gemm_v0 init=(i==0)） | 语义骨架先例（Expert 风格 + workspace，本设计按 Developer 重写；其 g 为 per-token 标量，本算子 per-channel [B,S,H,DK]） |
| `examples/linear_attention_and_rnn/opt_gdn/opt_gdn_chunk_o.py` | 高（优化版 chunk O：fp32 域 tile.mul 掩码、chunk 循环结构） | fp32 tile.mul 掩码先例（line 100,115） |
| `examples/developer_mode/flash_attn_bshd_developer.py` | 中（Developer 模式 CV 融合标准骨架） | threads/alloc/直连/pass_configs 基础范式（fla_wy_fast DESIGN §3.5.4 已核对） |
| `../tilelang-dev-journal/fla_wy_fast/DESIGN.md` + `perf_tuning/fla_wy_fast_final_summary.md` | 高（同族设计方法论 + **实测数据**） | 910B3 实测带宽 1263GB/s、20 AI Cores、单 kernel 逻辑带宽上限 81%、JIT launch ~1ms、工具链缺陷清单 §8-1~10 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量（全部 bshd 连续，kernel 直读 GM）

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| q | [B, S, H, DK] | bfloat16 | query |
| v | [B, S, H, DV] | bfloat16 | value |
| g | [B, S, H, DK] | **float32** | 逐 token 逐通道累积 log2 衰减（use_exp2=True 语义：2^g） |
| A | [B, S, H, 64] | bfloat16 | chunk 内 attention 矩阵（上游 chunk_intra 产出，只消费） |
| h | [B, S//64, H, DK, DV] | bfloat16 | chunk 级隐状态（上游 chunk_delta_h_fwd 产出，只消费） |
| Msk | [64, 64] | float32 | **设计新增常量**：`tril(ones, diagonal=0)` 含对角（i≥j 保 1，i<j 置 0），host 一次性构造 + per-(device,chunk) 缓存（★_MSK_CACHE 模式），16KB，L2 常驻 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| o | [B, S, H, DV] | bfloat16 | o = qg@h + tril(A)@v（fp32 累加，输出 cast bf16） |

### 4.3 中间缓冲区（block_S=64, block_DK=DK//2, block_DV=DV；bench 配置 block_DK=64/block_DV=128 为例）

| Buffer | Shape | dtype | 层级 | 用途 |
|--------|-------|-------|------|------|
| A_l1 | [64, 64] | bf16 | L1(shared) | intra gemm 左输入（掩码后 A） |
| V_l1 | [64, DV] | bf16 | L1(shared) | intra gemm 右输入 |
| H_l1 | [block_DK, DV] | bf16 | L1(shared) | inter gemm 右输入（i_k 切片，Pipelined 双缓冲） |
| qg_l1 | [64, block_DK] | bf16 | L1(shared) | inter gemm 左输入（i_k 切片，双缓冲） |
| a_half / a_f32 / msk_ub / a_mask | [64, 64] | bf16/fp32/fp32/bf16 | UB(shared) | A 载入中转 / fp32 域 / 掩码常量 / 掩码结果 bf16 |
| q_half / q_ub | [64, block_DK] | bf16/fp32 | UB(shared) | q 载入中转 / fp32 计算域（q_half 双缓冲） |
| g_ub / gL_ub / gexp_ub | [64, block_DK] | fp32 | UB(shared) | g / g·ln2 / 2^g（g_ub 双缓冲；gL/gexp 可复用槽位） |
| qg_ub / qg_half | [64, block_DK] | fp32/bf16 | UB(shared) | qg fp32 结果 / cast 中转 |
| o_frag | [64, DV] | fp32 | **L0C(fragment)** | 双 gemm 累加器 |
| o_ub / o_half | [64, DV] | fp32/bf16 | UB(shared) | L0C→UB 中转 / 输出 cast |

### 4.4 内存搬运路径

```
GM[q:bf16]  --T.copy(同dtype)--> UB[q_half:bf16] --T.copy(cast)--> UB[q_ub:fp32] --tile.mul(scale)--> UB
GM[g:fp32]  --T.copy(同dtype)--> UB[g_ub:fp32] --tile.mul(LN2)+tile.exp--> UB[gexp:fp32]
UB --tile.mul--> UB[qg:fp32] --T.copy(cast)--> UB[qg_half:bf16] --T.copy--> L1[qg_l1:bf16] ─┐
GM[h:bf16]  --T.copy(5D切片)--> L1[H_l1:bf16] ──────────────────────────────────────────────┤→ L0C[o_frag:fp32]
GM[A:bf16]  --T.copy--> UB[a_half] --cast--> UB[a_f32] --tile.mul(Msk)--> UB --cast--> UB[a_mask:bf16]
UB[a_mask]  --T.copy(循环内幂等)--> L1[A_l1:bf16] ──────────────────────────────────────────┤     (init=(i_k==0), i_k×2 累加)
GM[v:bf16]  --T.copy--> L1[V_l1:bf16] ──────────────────────────────────────────────────────┘     (init=False 追加)
L0C[o_frag] --T.copy--> UB[o_ub:fp32] --T.copy(cast)--> UB[o_half:bf16] --T.copy--> GM[o:bf16]
```

> 层级合法性：无 GM→L0 直搬；Cube 输入经 L1（gemm_v0 内部 L1→L0A/L0B）；Vector 计算全在 UB（fp32 域）；L0C→UB 用 T.copy 直连；跨 dtype 转换全部 UB↔UB 片内完成（cast-on-copy 缺陷规避★）。GM→L1 直拷仅用于无元素计算的 bf16 gemm 输入（v/h，★impl_final A_l1 先例）。

### 4.5 片上内存预算表（910B3：L1=512KB、UB=192KB、L0A/L0B=64KB、L0C=128KB per core；bench 配置 block_DK=64/block_DV=128）

| 层级 | 构成 | 大小 | 限额 | 结论 |
|------|------|------|------|------|
| L0C | o_frag [64,128] fp32 | **32KB** | 128KB | ✓ 25% |
| L1 | A_l1 8 + V_l1 16 + H_l1 16×2(双缓冲) + qg_l1 8×2(双缓冲) | **~72KB** | 512KB | ✓ 14% |
| UB 峰值 | A/msk 链常驻 24（a_half 8+a_f32 16 → 掩码后 a_f32 释放剩 a_mask 8+msk 16）+ i_k 阶段（q_half 8×2 + g_ub 16×2 + q_ub 16 + gL 16 + gexp 16 + qg 16 + qg_half 8）≈ 88 + epilogue 阶段（o_ub 32 + o_half 16 = 48，与 i_k 阶段分时） | **~112KB** | 192KB | ✓ 58%（余量 80KB；无 CSE/hoist 类生命周期延长优化——★k2 kernel 满 UB 零容忍教训） |

**备选 block_DV=64**（DV=128 时 bv=2 分块，调优候选）：L0C 16KB、UB 更低，但 q/g/A 重复读 +384MiB（§9.5），仅当 UB 溢出时降级（本设计预算不触发）。

### 4.6 动态轴定义

无运行时动态轴。B/S/H/DK/DV 均为 JIT 编译期常量（Python 参数传入，随 shape 重编译）——与上游 tilelang_chunk_fwd_o 一致；NT=S//64、grid=NT×B×H 编译期展开。验收基准为静态 shape（proto 声明）。

### 4.7 JIT 配置

```python
PASS_CONFIGS = {                                              # ★impl_final.py:79-84 同款四件套
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 C/V 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # 自动核间同步
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动核内同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # UB/L1 lifetime 规划
}
@tilelang.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)        # 无 workspace_idx（Developer 无 workspace）
```

---

## 5. Tiling 策略

### 5.1 计算类型

**混合（CV 融合）**：Cube（gemm_v0 ×3 次/block：inter×2 迭代 + intra×1）+ Vector（exp、标量乘、逐元素乘、掩码乘、cast 链）。

### 5.2 Block 划分

```python
block_S  = 64          # 固定 = chunk_size（语义约束，assert chunk_size == block_S）
block_DK = DK // 2     # exact-2 Pipelined 约束（★final_summary §8-2）：64→32 / 96→48 / 128→64
block_DV = DV          # 单块（DV ≤ 128），消除 bv 分块 → q/g/A 零重复读（§10.4-A）
grid    = NT * B * H   # bench: 128 × 1 × 64 = 8192 blocks；20 AI Cores → 410 blocks/core 负载均衡 ✓
```

**选型理由**：① block_DV=DV 使每张量每元素恰好读/写 1 次（IO 唯一化 960MiB，带宽敏感算子的第一优先级）；② block_DK=DK//2 是 extent=2 工具链约束下的唯一解（extent-1 崩溃 / extent≥3 回卷损坏★），且 block_DK∈{32,48,64} 均满足分形 K≥32 与 UB/L1 预算；③ 上游 GPU autotune 的 block_DK/block_DV 搜索空间在昇腾被约束全消解（§3.5.2-2），无调优自由度损失。

### 5.3 约束分析

- 对齐：bf16 尾轴 block_DK∈{32,48,64}×2B = 64/96/128B、DV×2B ≥ 32B；fp32 g 128/192/256B——全部 ≥ 32B（UB/L1 对齐）✓；A/o [64,64]/[64,DV] bf16 128/256B ✓；L0A/L0B 512B 分形由 gemm_v0 内部处理（M=64, K≥32, N≥16 ✓）
- L0C：32KB ≤ 128KB ✓
- GEMM 非整除：S%64==0（范围）、DK exact-2 构造整除、DV 单块——声明域内无非整除路径（§3.5.1）

### 5.4 注意事项（尾块）

**声明域内无尾块**：S%64==0（proto 范围约束）、DK ∈ {64,96,128} 被 block_DK=DK//2 精确二分、DV=block_DV 单块。域外输入（S%64≠0、DK∉{64,96,128}、DV>128 或非 16 倍数、chunk_size≠64）由 host wrapper 的 `_check_shape` 显式 assert 拒绝（★fla_wy_fast `_check_shape` + `_pick_block_d` raise 同款）——L2 负向测试覆盖，**不设计 host padding + crop**（禁止，ascend-constraints §5）。

### 5.5 数据搬运性能可行性（REORDER-COST）

本算子无数据重排（bshd 连续直读、无 permute/gather），全部搬运为规则 tile 块 DMA：

```text
[REORDER-COST]
path: 单 kernel 主路径 (block_DK=64, block_DV=128, bench shape 1×8192×64×128×128)
gm_passes: 每张量 1 读 + o 1 写（唯一 IO 960 MiB：q 128 + v 128 + g 256 + A 64 + h 256 读，o 128 写）
dma_transactions: ~10 次/block（A 1 + Msk 1[L2 常驻，HBM 仅首次] + q 2 + g 2 + h 2 + v 1 + o 写 1），8192 blocks ≈ 8.2 万次
average_dma_bytes: 960MiB / 8.2万 ≈ 12.2KB/次（block 级大块 DMA）
gm_scalar_accesses: 0（无标量 GM 访问；所有元素运算在 UB）
address_div_mod_per_element: 0（cid 分解每 block 4 次整数运算）
active_cores / serial_tasks_per_core: 20 / 410
largest_case_timeout_gate: bench shape 单 case 60s（§9.4 perf_bench）
result: pass
```

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| (chunk, b, h) | block 级并行 | `T.Kernel(NT*B*H)` | 每 block 独立一个 chunk-head，无依赖 |
| DK 维 | 迭代+预取 | `T.Pipelined(DK//block_DK, num_stages=2)` | **恰好 2 次迭代**（extent 约束★）；num_stages=2 双缓冲掩盖 GM→UB/L1 搬运延迟（★impl_final 全 kernel 同款） |
| 元素级 | 向量化 | `T.tile.*` buffer 级全 tile 指令 | ★逐元素 T.Parallel 串行循环为性能反模式（-26%~-71%/kernel），一律 tile 指令 |

### 6.2 循环伪代码

见 §3.3（kernel 主体伪代码，prologue → i_k Pipelined ×2 → epilogue 三段）。

### 6.3 流水线优化

`T.Pipelined(num_stages=2)` 核内预取：i_k 循环 2 次迭代（DK=128 时）恰为双缓冲覆盖。核间流水线（Cube/Vector 交织）由 AUTO_CV_COMBINE 自动编排（与核内 Pipelined 互斥，源码层不显式干预）。调优空间：threads∈{1,2}（Stage 3，★threads=2 有 gather 越界前科，本算子无 gather 链，候选但非主选）。

### 6.4 尾块处理

见 §5.4（声明域内无尾块；域外 shape 显式拒绝）。

---

## 7. 同步策略

### 7.1 同步模式

**自动同步**（Developer 模式标准）：AUTO_SYNC 处理核内 DMA/计算依赖，AUTO_CV_SYNC 处理 Cube↔Vector 核间依赖。源码无任何 `T.barrier_all` / `T.set_flag` / `T.set_cross_flag`。

### 7.2 同步点说明（编译器自动插入，设计仅声明数据依赖）

| 依赖 | 说明 |
|------|------|
| UB 元素运算 ← GM 载入（q/g/A/Msk） | AUTO_SYNC |
| gemm_v0 ← L1 输入就绪（qg_l1/H_l1/A_l1/V_l1） | AUTO_SYNC（C 核内） |
| UB 元素运算 ← L0C 结果（T.copy(o_frag, o_ub)） | AUTO_CV_SYNC（跨核） |
| L1 gemm 输入 ← UB cast 结果（qg_half→qg_l1、a_mask→A_l1） | AUTO_CV_SYNC（V→C 交换，全部位于 Pipelined 体内——安全模式★） |
| GM 写 ← UB 结果（o_half→O） | AUTO_SYNC |

### 7.3 pass_configs 配置

见 §4.7（四件套全开）。

---

## 8. CV 融合设计

### 8.1 融合算子判定

**是**（Cube GEMM ×3 + Vector exp/逐元素/掩码/cast）。

### 8.2 CV 交互设计（Developer 模式，默认消除 workspace/vid）

- `T.Kernel(NT*B*H, threads=1, is_npu=True) as (cid)`：单轴 + threads=1（★fla_wy_fast 最终交付同款）
- 装饰器无 `workspace_idx`，签名无 `workspace_*` 参数、无 vid 偏移（threads=1 单返回值 cid）
- Cube↔Vector 片上直连：`T.copy(fragment, shared_ub)` / `T.copy(ub, l1_shared)`，中转与同步交给四个 pass
- **无 GM workspace 中转**（带宽敏感算子，★被否方案 B 教训：workspace 中转 IO 近乎翻倍）

### 8.3 / 8.4 Cube 与 Vector 核计算流程

见 §3.3 伪代码：V 侧 = qg 链（cast/mul/exp）+ A 掩码链 + 输出 cast 链；C 侧 = gemm_v0 ×3（inter×2 + intra×1）+ h/v GM→L1 DMA。两侧经 alloc_shared/alloc_fragment 直连。

### 8.5 pass_configs

见 §4.7。

### 8.6 注意事项

- 核内 Pipelined 与核间流水线互斥（pass 自动决策）
- **C/V 跳数控制**（本设计关键安全约束）：完整序列 = prologue(V) → [V→C]×2 迭代（循环内）→ epilogue C(gemm) → C→V(输出)。循环后仅 **C→V 单跳**——刻意将 A 掩码链留在 prologue V 侧、A_l1 交换留在循环内（幂等重复），规避"后置多跳 C/V 交替 sync-point 失配"（★final_summary §8-4 z+da 合并失败先例）与"prologue V→C 错版本化"（★§8-5）
- **多 gemm 组共享 L1 操作数**（★§8-3 角色互换复用损坏 L0 staging）：本设计 qg_l1/H_l1/A_l1/V_l1 四个 gemm 输入各自独立 buffer、无角色互换复用 ✓

---

## 9. 验证方案

### 9.1 Golden 函数（PyTorch chunk 循环实现，供 Stage 2 直接落地）

> 布局 bshd 与 kernel 一致；golden 用 **fp32 计算域**（`torch.exp2` 精确语义），kernel 侧 `exp(g·ln2)` 替代与 qg 的 bf16 cast 差异计入混合容差验收（误差预算见 §9.3 注）。NPU 环境 Triton 不可用（ref_gpu/fla_chunk_o.py 仅作语义对照，不跑 golden）。

```python
import torch


def fla_chunk_o_ref(q, v, g, A, h, scale=None, chunk_size=64):
    """fla_chunk_o golden。q[B,S,H,DK]/v[B,S,H,DV]/A[B,S,H,C]/h[B,NT,H,DK,DV] bf16,
    g[B,S,H,DK] fp32 -> o[B,S,H,DV] bf16。C = chunk_size = A.shape[-1]。fp32 计算域。"""
    B, S, H, DK = q.shape
    DV = v.shape[-1]
    C = A.shape[-1]
    assert C == chunk_size and S % C == 0, "S must be divisible by chunk_size"
    if scale is None:
        scale = 1.0 / DK ** 0.5
    NT = S // C
    qf, vf, Af, hf = q.float(), v.float(), A.float(), h.float()
    M = torch.tril(torch.ones(C, C, dtype=torch.float32), 0)   # 含对角（i >= j）
    o = torch.empty(B, S, H, DV, dtype=v.dtype)
    for i in range(NT):
        sl = slice(i * C, (i + 1) * C)
        qc, gc = qf[:, sl], g[:, sl]                            # [B,C,H,DK] fp32
        Ac, vc, hc = Af[:, sl], vf[:, sl], hf[:, i]             # [B,C,H,C] / [B,C,H,DV] / [B,H,DK,DV]
        qg = qc * scale * torch.exp2(gc)                        # [B,C,H,DK]
        o_inter = torch.einsum("bchk,bhkd->bchd", qg, hc)       # inter-chunk
        o_intra = torch.einsum("bcij,bcjd->bcid", Ac * M, vc)   # intra-chunk（tril 含对角）
        o[:, sl] = (o_inter + o_intra).to(v.dtype)
    return o
```

输入数据构造（对齐上游 `prepare_input`，ref_gpu/fla_chunk_o_tilelang_fwd.py:26-44）：`torch.randn` 全量（q/A/v/h bf16，g fp32）——上游即如此（A 非真实 attention 分布、h 非真实隐状态，本算子只消费不生产，randn 对拍成立）；`scale = 1.0 / DK**0.5`。

### 9.2 L0 门槛测试计划（tilelang-op-test-design 场景 A，仅 L0）

> Stage 1 只产出 L0（规则 shape：S%64==0、DK∈{64,96,128}、DV 16 倍数——全部被 block 精确整除）。L1（功能，含域外/边界 shape）/ L2（异常拒绝）/ Boundary（特殊值）由 Stage 2 在 L0 通过后按场景 B 扩展（含 DK=32 拒绝、g 极值上溢等）。
> 算子类别判定（场景 A Phase 2）：计算类型 = 混合（CV 融合：Cube GEMM + Vector exp/掩码/cast）；复杂度 = Fusion（2 类 GEMM + 逐元素前处理 + 掩码）；数学特征 = matmul + element-wise gating；综合类别 = **Fusion（gated chunked attention 输出）**→ 测试策略聚焦 CV 协作正确性 + chunk 语义 + dtype 链（bf16 in/out + fp32 gate 固定组合，无 dtype 变体空间）。

| 用例名 | 级别 | Shape (B,S,H,DK,DV) | dtype | 说明 |
|--------|------|---------------------|-------|------|
| l0_min | L0 | (1, 128, 2, 64, 64) | in/out bf16 + gate fp32 | 最小冒烟：NT=2、BH=2、block_DK=32（**K=32 边界分形** + 2 迭代 Pipelined）、block_DV=64、grid=4 blocks。覆盖全链路：qg cast/exp2 替代/scale 链、inter 累加（init=(i_k==0)）、tril(0) 掩码、intra 追加（init=False）、L0C 三步输出链、5D h 切片 |
| l0_repr | L0 | (2, 256, 4, 128, 128) | in/out bf16 + gate fp32 | benchmark 结构代表：block_DK=64（2 迭代）、block_DV=128（单块满宽）、B>1 触发 bb/bh 分解、grid=32 blocks。bf16 gemm 全量程路径 |

**golden 对照法**：每用例对输出 o 独立调用 `check_precision(o, ref, "bfloat16")`（precision-standard.md §4.1 混合容差判定，Stage 2 原样复制不改动）；通过打 `[PRECISION_PASS]`，失败 `[PRECISION_FAIL]`（阻塞，计入退出码）。

### 9.3 精度标准（混合容差，precision-standard.md 口径）

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|-------|------|------|---------------------|------------------------|
| bfloat16（输出 o，唯一判定对象） | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |
| float32（gate 输入 g；数值按 precision-standard.md fp32 档登记） | 2⁻¹⁶ (1.53e-5) | 2⁻¹⁰ (9.77e-4) | 1e-2 | 0.99 |

> float32 行说明：g 为 fp32 **输入中间量**（逐元素参与 qg 计算），本算子**无 fp32 输出**，该 dtype 无独立输出判定对象（其误差折算进 o 的 bf16 行验收）；此行数值按 precision-standard.md fp32 档登记，仅为满足"§9.3 dtype 行与 proto `inputs[].dtype` 全集 {bfloat16, float32} 一致"的覆盖门禁一致性约束（★fla_wy_fast §9.3 同款处理）。
>
> 误差预算【推断】：kernel 相对 golden 的主要差异 = qg 的 bf16 cast（单 round，相对 ~2⁻⁹；与语义权威 Triton 的 `b_qg.to(bf16)` 一致）+ exp2→exp(g·ln2)（~1e-7）+ bf16 gemm 输入 cast。o 量级 ~1.7（randn、scale=1/√128、DK=128 累加），逐元素容差 ≈ atol + rtol·|o| ≈ 0.027，典型误差 ~0.003（2⁻⁹ 相对）——余量 ~9×，matched_ratio 预期远超 0.99。若 L0 实测不足，回退 qg hi/lo 双 gemm（★fla_wy_fast fwd_u 同款 2-gemm 模式，相对误差 2⁻¹⁶）。
>
> 特殊值（NaN/Inf）位置契约：输入域为 randn 有限值（上游对拍口径）；g 极值（g>128 时 2^g=inf）属 Boundary 范畴，由 Stage 2 场景 B 扩展（"有限值 + 稀疏特殊值"混合输入 + mask 结构比对，precision-standard §3.1）。

### 9.4 性能可行性哨兵（强制，不可跳过）

| 用例名 | Shape (B,S,H,DK,DV) | dtype/属性 | 覆盖路径 | 单 case timeout | 选择理由 |
|--------|---------------------|------------|----------|-----------------|----------|
| perf_bench | (1, 8192, 64, 128, 128) | bf16 in/out + fp32 gate | 单 kernel 主路径（8192 blocks = 最大任务数、最大 DMA 吞吐压力、block_DK=64/block_DV=128 满配） | 60s | benchmark 验收 shape（proto 性能目标口径） |

计时口径：对齐上游 `ref_gpu/fla_chunk_o_tilelang_fwd.py` 内嵌 `do_bench`——**warmup=10、rep=10、取均值、仅设备侧 kernel 时间**（NPU 事件 `torch.npu.Event(enable_timing=True)` 等价实现），不含张量构造/H2D-D2H/JIT/框架开销。wrapper 内**无** `torch.npu.synchronize()`（★fla_wy_fast 优化项 #10：内部 sync 会把 host JIT 分发延迟暴露进设备侧计时，与口径相悖）。测试数据、随机数与 golden 均在 CPU 完成后 H2D，运行段无 aclnn 依赖。

### 9.5 性能预算分析（910B3，含 1320us 目标可行性结论）

**IO 明细（物理唯一字节，每张量单读单写，block_DV=DV 零重复读）**

| 项 | bench shape (1, 8192, 64, 128, 128) |
|----|-------------------------------------|
| 读 | q 128 + v 128 + g 256(fp32) + A 64 + h 256 = **832 MiB** |
| 写 | o = **128 MiB** |
| 合计 | **960 MiB ≈ 1.007 GB**（该数据布局下的 IO 下限，无 kernel 层压缩空间） |
| GEMM FLOPs | 25.8 GFLOP bf16（inter 17.2 + intra 8.6，8192 chunk-head × 3.15M） |
| Vector ops | exp 67M 次 + 逐元素 ~0.3 GFLOP（tile 指令）——均非瓶颈 |

**Roofline 对照（910B3，实测口径）**

| 项 | 数值 | 依据 |
|----|------|------|
| HBM 实测带宽（r+w） | **1263 GB/s**（1259–1273 稳定区间取中） | 【已确认事实】★fla_wy_fast D2D 大拷贝实测（修正其 DESIGN 未校准的 392GB/s 标称假设，3.2× 偏差） |
| 带宽理论下限 | 1.007 GB ÷ 1263 GB/s = **797 us** | 唯一 IO ÷ 实测上限 |
| Cube 计算时间 | ~164 us @157 TFLOPS bf16 折算 | 【推断】非瓶颈（< 带宽下限的 21%） |
| **1320us 目标隐含带宽** | 1.007 GB ÷ 1320 us = **762 GB/s = 实测上限的 60.3%** | 有效性需求 |

**结论：1320us 目标物理可达（797us 硬下限 = 目标的 60%），但处于临界区，需 Stage 2/3 实测确认**：

1. 与 fla_wy_fast 的"物理不可达"结论（1370us 需 1.44× 实测上限）**本质不同**——本算子单 kernel、无 workspace 中转、无 kernel 拆分重复读、无 fp32 gemm 模拟开销，结构上更接近 ★实测中逻辑带宽最高的 k1/k2 单 kernel（**81% 实测上限**）。60.3% 的有效性需求在 k1/k2 实证带宽利用率区间之内。
2. **JIT launch ~1ms 风险**（★final_summary §8-8：host 分发 ~1ms/launch）：do_bench rep=10 连续 launch 下，kernel ≥1ms 时分发完全隐藏；本算子预期 kernel 时间 1.0–1.3ms（60–80% 带宽效率）恰在隐藏边界附近。若实测事件时间被 launch 垫底在 ~1ms 附近，仍 < 1320us ✓；若带宽效率 <60%，目标失败——perf_bench 哨兵实测判定，Stage 3 调优手段：threads∈{1,2}、block_DV=64+L2 复用（§10.4-A 变体）、num_stages 微调。
3. 设计期已内建的 IO 优化：block_DV=DV 消除 bv 重复读（vs 上游 GPU 版 block_DV=32 的 4× 重复读 q/g/A，+1.35GB IO）、单 kernel、Msk L2 常驻（16KB，HBM 仅首次）、A 掩码链片上复用（无重复 GM 读）。

### 9.6 Host 侧 Buffer 操作审计

```text
[HOST-METADATA-AUDIT]
operation: 输入构造 torch.randn(...).npu()（数据准备，CPU 生成后 H2D）
  + Msk 常量构造 torch.tril(torch.ones(64,64), 0).npu()（一次性常量输入，per-(device,chunk) 缓存）
  + kernel 调用（直传 5 张量 + Msk）+ 精度验证（CPU 侧对拍）
input_stride -> output_stride: N/A（新建张量，非变换；kernel 直读 bshd 连续布局，无任何 view/reshape/permute）
shares_storage / same_data_ptr: N/A（无变换操作）
aclnn_or_physical_copy: false（randn/tril/ones 为 CPU 数据生成 + 一次性 H2D；kernel 调用前后无
  .contiguous()/pad/cat/reshape 物量化/.to(dtype)/.clone()）
result: allow
```

---

## 10. 风险点与注意事项

### 10.1 风险清单（高/中/低分级）

| 级别 | 风险 | 影响 | 缓解/验证 |
|------|------|------|-----------|
| **高** | R1 Pipelined 循环 + 后置单 gemm epilogue（C→V 单跳）无逐字先例（★impl_final 各 kernel epilogue 均为纯 V 侧；z+da 五跳合并失败先例） | 编译错（sync-point 失配）或 L0 损坏 | 结构已收敛至最保守形态（§8.6 跳数分析：da kernel C→V 段同构）；l0_min 首验；**回退 1**：A 掩码与 A_l1 交换移回 epilogue（V→C→V 三跳，da kernel 全同构）；**回退 2**：拆双 kernel（inter GM partial + intra 合并输出，+256MiB IO ~200us，最后手段，需向 orchestrator 报 [DESIGN_ERROR] 决策） |
| **高** | R2 性能临界：1320us 需 60.3% 实测带宽效率 + JIT launch ~1ms 隐藏边界 | 性能验收失败 | §9.5 论证（k1/k2 实证 81% 上限支持可行性）；perf_bench 哨兵实测；Stage 3 调优空间（threads/block_DV 变体/num_stages）；如实测确认不可达，按 Stage 3 流程上报带宽效率口径替代方案（★fla_wy_fast §4 同款处理） |
| 中 | R3 `T.tile.mul` fp32 域掩码 + bf16 cast 链的 round 模式（CAST_RINT vs CAST_NONE） | 精度 | ★impl_final 同款链路实证（fwd 全链 L0/L1 PASS）；l0_min 对拍；异常时改 `T.tile.cast` 显式 round 模式 |
| 中 | R4 epilogue 的 GM→L1 直拷（v_l1）位于 Pipelined 循环之后（C 侧 DMA 无先例记录） | 编译/数据错位 | ★循环不变 GM→L1 载入在循环内重复执行的先例（impl_final fwd_u A_l1 line 163）可作回退（v_l1 载入移入循环内，+128MiB 重复读 ~100us）；l0_min 首验 |
| 中 | R5 5D h 切片 `Hstate[bb,bs,bh,dk0:dk1,dv0:dv1]` 的 DataCopyPad 行间 gap 行为 | 数据错位（静默） | gdn_chunk_o.py:66 同款 5D 切片先例（Expert 模式）；l0_min 用 h 逐元素校验（h 是 o_inter 的直接乘子，错位必然暴露）；如失败改 4D 化 host view（h.view 不物理化——[B,NT,H,DK,DV] 连续张量 view 为 [B*NT*H, DK, DV] 零拷贝，kernel 签名同步调整） |
| 低 | R6 exp2→exp(g·ln2) 精度（~1e-7 相对） | 精度微小 | 远小于门禁（2⁻⁶/2⁻¹⁰）；golden 用 torch.exp2 对拍验证 |
| 低 | R7 DK=96 路径 block_DK=48（非 2 幂）的 tile 指令/gemm 支持 | L1 用例失败 | ★fla_wy_fast d96 用例已实证 block=48 路径（L1 4/4 PASS）；不在 L0（Stage 2 扩展） |
| 低 | R8 grid 上限 8192 blocks | 无 | ★fla_wy_fast bench 同量级（8192）实证 |
| 低 | R9 threads=1 的 C/V 核利用率 | 性能 | ★fla_wy_fast 全部 kernel threads=1 交付实证；threads=2 为 Stage 3 候选（无 gather 链，风险低于先例） |

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| GM↔UB 跨 dtype 直拷（照抄"cast-on-copy"直觉） | `T.copy(Q[...], q_ub_fp32)` | codegen 非法模板调用（★§8-6） | 两步链：GM 同 dtype → UB → UB 内 cast copy（本设计已内置） |
| L0C→GM bf16 直存 | `T.copy(o_frag, O[...])` | fixpipe 转换损坏（★§8-7） | 三步链 frag→ub(fp32)→ub(bf16)→GM（已内置） |
| Pipelined extent≠2 | block_DK=DK（1 迭代）或 DK/32（≥3 迭代） | extent-1 崩溃 / 第 3 迭代回卷损坏（★§8-2） | `_pick_block_dk(DK)=DK//2` 强制 exact-2（已内置） |
| `T.if_then_else` 写进 T.Parallel/T.tile | 照抄上游 A 掩码 | 编译错误 | 常量 Msk + T.tile.mul（已内置） |
| `T.gemm` 通用版 / fp32 gemm_v0 | 照抄上游 / h cast fp32 | intrinsic 不支持 / 1e37 垃圾（★§8-1） | 全部 bf16 gemm_v0（已内置） |
| threads=128 / 三维 Kernel | 照抄上游 | 编译报错 | 一维 + threads=1（已内置） |
| host 侧预计算 2^g / 掩码 A / `.to()` 预处理 | 优化幻觉 | aclnn 违规 + IO 翻倍 | 禁止（被否方案 E）；kernel 内 UB 一次计算 |
| 逐元素 T.Parallel 串行循环 | 照抄上游 `for i_s,i_k in T.Parallel` | -26%~-71%/kernel（★perf 教训） | 全 T.tile.* buffer 级指令（已内置） |

### 10.3 特殊场景处理

- **域外 shape**（S%64≠0、DK∉{64,96,128}、DV>128/非 16 倍数、chunk_size≠64）：host `_check_shape` assert 拒绝（L2 负向测试；documented limitation 清单见 §1.6）
- **极小 shape**（S=64, NT=1）：grid=B*H，正确但 L0 不含（用 S=128 保证 NT≥2）；DK=64 用例同时覆盖 block_DK=32 边界分形
- **g 极值上溢**（g 大正值 → 2^g=inf）：Boundary 范畴（Stage 2 场景 B：稀疏 inf 混合输入 + mask 结构比对）
- **B/H 任意**：cid 分解纯整数运算，无约束（bbh=B*H 编译期展开）

### 10.4 被否方案（设计决策记录）

| # | 方案 | 否决理由 |
|---|------|---------|
| A | block_DV=64 + bv 维 grid 分块（对齐上游 GPU 版 autotune 域） | bv=2 时 q/g/A 各重复读 2×（+384MiB IO ≈ +304us @1263GB/s），L2 掩盖不可靠（block 调度顺序无跨核保证）；block_DV=DV 单块 IO 唯一（960MiB）且 L0C/L1/UB 预算全部安全。仅当 UB 溢出时降级启用（本设计预算不触发） |
| B | 照抄上游 GPU 配置（block_DK=32/block_DV=32/threads=128/num_stages=1/三维 Kernel） | 三维 Kernel 不支持 + threads 违规 + i_k extent=4（≥3 双缓冲回卷损坏★）多重工具链违规；GPU smem 约束下的 autotune 解在昇腾无意义 |
| C | epilogue 内做 A 掩码（循环后 V→C→V 三跳自然结构） | 后置多跳 C/V 交替 sync-point 失配前科（★§8-4）；主选将掩码留在 prologue V 侧 + A_l1 交换在循环内（幂等重复，代价 8KB×2 次 UB→L1 片上拷贝 <1%），循环后收敛为 C→V 单跳。C 方案降为 R1 回退 1 |
| D | 拆双 kernel（inter：qg@h → GM partial fp32；intra：掩码 + A@v + add + 输出） | +256MiB GM 中转 IO（~200us）+ 2× JIT launch + 违背 proto"单 kernel"口径；仅在 R1 两级回退均失败时作为最后手段（需上报决策） |
| E | host 侧预计算 2^g（对 g 做 aclnn exp）或预掩码 A | 违反 host Buffer 约束（ascend-constraints §5：`.to()`/函数计算触发 aclnn，评测环境可能裁剪）；且 g 256MiB 读+写翻倍；kernel 内 UB exp 一次即最优 |
| F | qg hi/lo 双 gemm 提精度（★fla_wy_fast fwd 同款） | 过度设计：本算子门限 bf16 rtol 2⁻⁶，单 cast 误差 2⁻⁹ 余量 9×（§9.3 预算），且语义权威 Triton 即单 cast（`b_qg.to(bf16)` 后 dot）；hi/lo 使 gemm 次数 ×2。降级为 L0 精度不足时的回退手段（§9.3 注） |
| G | Expert 模式 + 显式 workspace 中转（gdn_chunk_o.py 模式） | 每次跨核中转经 GM 读写，唯一 IO 近乎翻倍（带宽受限下致命）；用户已指定 Developer；★fla_wy_fast 被否方案 B 同款结论 |

---

## 11. 交付清单

### 11.1 目录结构（沙箱，{op} 前缀命名规范——对齐 example_indexer_bwd.py 先例）

```
examples_experiment/dsa_sparse_finetune/     # 沙箱目录（与 indexer_bwd / fla_chunk_intra 等共存，互不触碰）
├── DESIGN_fla_chunk_o.md        # 本设计文档（含 L0 门槛测试计划）
├── proto_fla_chunk_o.yaml       # 算子接口规格（任务书载体，已存在，承担 proto.yaml 角色，见 §11.2）
├── ref_gpu/                     # 冻结参考（只读）
├── example_fla_chunk_o.py       # ⬜ 待实现（Stage 2）：单 kernel（@tilelang.jit）+ host wrapper
└── test_example_fla_chunk_o.py  # ⬜ 待实现（Stage 2）：golden（§9.1）+ L0（§9.2）+ perf 哨兵 + main 分发器
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN_fla_chunk_o.md` | ✅ 已完成 | 本文档 |
| `proto_fla_chunk_o.yaml` | ✅ 已存在（任务书载体，**不修改**） | proto.yaml 角色：`operator.inputs[].dtype` 全集 = {bfloat16, float32}（与 §9.3 精度表 dtype 行一致）、`attrs[].name` 覆盖 chunk_size、scale 全部关键属性、schema 完整——经核对已满足覆盖门禁 `coverage_check.py --proto` 的派生要求，不重复产出（★fla_wy_fast §11.2 同款处理）。设计新增的 Msk 常量为 kernel 内部常量输入（非算子语义输入），不纳入 proto 语义 inputs |
| `example_fla_chunk_o.py` | ⬜ 待实现 | `fla_chunk_o_kernel`（Developer 模式，§3.3 伪代码落地 + §4.7 pass_configs）+ host wrapper `fla_chunk_o(q, v, g, A, h, scale, chunk_size)`（Msk 缓存 + `_check_shape` + `_pick_block_dk`） |
| `test_example_fla_chunk_o.py` | ⬜ 待实现 | golden `fla_chunk_o_ref`（§9.1 逐行落地）+ `check_precision`（precision-standard.md §4.1 原样复制）+ L0 两用例（§9.2）+ perf_bench 哨兵（§9.4，NPU 事件 warmup=10/rep=10）+ `--level` main 分发器 |

### 11.3 命名规范

- kernel 文件 `example_fla_chunk_o.py`、测试 `test_example_fla_chunk_o.py`（沙箱 `example_` 前缀，对齐 `example_indexer_bwd.py` 先例）
- host API 名对齐上游语义：`fla_chunk_o`（对应上游 `chunk_gla_fwd_o_gk`）；kernel 内部名 `fla_chunk_o_kernel`

### 11.4 实现顺序

1. ✅ 设计文档 + L0 门槛测试计划（本文档 §9.2）
2. ⬜ `example_fla_chunk_o.py`：单 kernel（bf16 gemm + fp32 gate 链 + 掩码 + epilogue）
3. ⬜ `test_example_fla_chunk_o.py`：golden + L0 两用例 + perf 哨兵 + main
4. ⬜ L0 通过（精度收敛；`check_precision` 全绿；R1/R4/R5 首验结论回填）
5. ⬜ 扩展分层套件 L1/L2/Boundary（tilelang-op-test-design 场景 B：含 DK=32/S%64≠0 拒绝、DK=96 路径、g 极值 Boundary）
6. ⬜ perf_bench 实测（§9.5 结论复核：60.3% 带宽效率需求 + launch 隐藏验证；不达标进入 Stage 3）

### 11.5 算子 proto.yaml（覆盖门禁用）

见 §11.2：本算子的 proto 角色由 `proto_fla_chunk_o.yaml` 承担（任务书冻结，不修改）。其 `operator.inputs[].dtype` 全集 {bfloat16, float32} 与本文档 §9.3 精度表 dtype 行一致；`operator.attrs[].name` = {chunk_size, scale} 覆盖全部关键属性；与设计无冲突（设计新增 Msk 常量为 kernel 内部输入，不影响 proto 语义接口）。

---

## 附：本文档信息类别声明

- 【已确认事实】：§3.2 全部 API 出处（源码行号/examples 路径/★同族交付代码）、§3.5.1/3.5.2 全部工具链约束与适配结论（★fla_wy_fast impl_final.py + final_summary.md 实测记录）、§4.5 硬件容量（910B3：20 AI Cores/L2 192MB/L1 512KB/UB 192KB/L0C 128KB）、§9.3 精度门禁（precision-standard.md 口径 + 用户确认）、§9.5 实测 HBM 带宽 1263GB/s（★D2D 实测）、gdn_chunk_o.py 同名先例（tril(0) 掩码/5D 切片/init=(i==0)）
- 【推断】：§9.5 Cube 157 TFLOPS 折算与 60.3% 带宽效率可达性（结构类比 k1/k2 的 81% 实证）、§9.3 误差预算（2⁻⁹ 余量 9×）、JIT launch 隐藏边界行为
- 【需 Stage 2 验证】：R1（epilogue C→V 单跳组合）、R3（cast round 模式）、R4（循环后 GM→L1 直拷）、R5（5D h 切片 DataCopyPad）、R9（threads=1 核利用率）、§9.5 性能实测复核
