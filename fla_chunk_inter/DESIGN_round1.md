# fla_chunk_inter 算子设计文档（KDA chunked 反传 inter-chunk 梯度，单 kernel，4 输出）

> GPU→Ascend 迁移设计（tilelang-op-orchestrator Stage 1 / tilelang-op-design）。
> 语义权威：`ref_gpu/fla_chunk_inter.py`（Triton/FLA `chunk_kda_bwd_kernel_inter` / wrapper `chunk_kda_bwd_dqkwg`，只读）；代码级起点：`ref_gpu/fla_chunk_inter_tilelang_gpu.py`（上游 TileLang CUDA `chunk_bwd_dqkwg`，只读）；计时口径：`ref_gpu/test_utils_kda.py`（do_bench：warmup=20 / rep=10 / mean / 仅设备侧）。接口/公式/门禁权威：`proto_fla_chunk_inter.yaml`（任务书，只读）。
> 目标硬件：**Ascend 910B3**（20 AI Cube + 40 AI Vector Cores / L2 192MB / HBM 64GB；片上容量按 A2 级：L1 512KB、**UB 硬上限 196352B**（ascend_memory_planning.cc:35，fla_chunk_o 4-chunk 批处理实证封死）、L0A/L0B 64KB、L0C 128KB per core；HBM 实测 r+w 带宽 **1259–1273 GB/s**，取 1263 GB/s——同族 fla_wy_fast Stage 3 D2D 实测标定值）。
> 本文档区分【已确认事实】（源码/示例/同族交付实测验证）、【推断】（标注依据）、【需 Stage 2 验证】三类信息。
> 同族已交付先例（★ 标注引用）：`../tilelang-dev-journal/fla_wy_fast/`（KDA WY 表示，Developer 模式 6 kernel 交付，含工具链缺陷清单与实证 API 链）、`../tilelang-dev-journal/fla_chunk_o/`（KDA 输出 O，fixed-core + L2 中继架构交付，含每任务同步协议地板核算）；同沙箱兄弟设计：`DESIGN_fla_chunk_intra.md`（KDA intra bwd，同为"全 fp32 + fp16 hi/lo 拆分 + Pipelined(2) 体 + V 尾段"结构）。

---

## 1. 概述

### 1.1 算子名称

`fla_chunk_inter`（单 kernel：KDA chunked 反传 inter-chunk 梯度，上游 Triton 名 `chunk_kda_bwd_kernel_inter` / host 名 `chunk_kda_bwd_dqkwg`，上游 TileLang CUDA 名 `chunk_bwd_dqkwg`）

### 1.2 功能描述

KDA (Kimi Delta Attention) chunked 反传的 inter-chunk 梯度算子：给定逐 token 累积 log2 衰减 g、chunk 级隐状态 h 及其梯度 dh、上游梯度 dv/do，逐 chunk（BT=64）逐 head 逐 DK-block 计算 4 个输出梯度 dq/dk/dw/dg。3 条 GEMM 主干（do@h^T、v@dh^T、-(dv@h^T)）沿 DV 维分块 fp32 累加；门控缩放（2^g / 2^(gn−g)）、dgk 末行注入、dg 组装为 Vector 侧逐元素/归约计算。chunk 间 / head 间 / DK-block 间无数据依赖，纯并行；DV 维（GEMM 收缩维）必须在任务内循环累加。

本算子为 **CV 融合**（Cube GEMM ×3 组 + Vector exp/逐元素乘/reduce）。

### 1.3 数学公式

记 chunk 长度 BT=64（S 可整除，见 §1.6），c = t // BT，last(c) = min(S,(c+1)·BT)−1（**域内恒为 chunk 内局部行 63**），scale=1.0（标量乘保留），g 为逐 token 累积 log2 衰减（fp32），h 为 chunk 级隐状态 [DK,DV]，dh 为其梯度；⊙ 逐元素乘，2^x = exp2(x)：

```
dq_raw[t,k] = Σ_v do[t,v] · h[c,k,v]            # GEMM: do @ h^T   [BT, DK]
dk_raw[t,k] = Σ_v v[t,v]  · dh[c,k,v]           # GEMM: v  @ dh^T  [BT, DK]
dw[t,k]     = -Σ_v dv[t,v] · h[c,k,v]           # GEMM 取负: -(dv @ h^T)
dgk[k]      = (Σ_v h[c,k,v] · dh[c,k,v]) · 2^g[last(c),k]
dq[t,k]     = dq_raw[t,k] · scale · 2^g[t,k]
dk[t,k]     = dk_raw[t,k] · 2^(g[last(c),k] − g[t,k])
dg[t,k]     = q[t,k]·dq[t,k] − k[t,k]·dk[t,k] + (t == last(c)) · (dgk[k] + Σ_{t'} dk[t',k]·k[t',k])
```

所有 GEMM 沿 DV 维分块累加（fp32 累加语义）。注意 dgk 的最终值 = exp2(gn)·Σ_v h·dh 与 Σ_t dk_scaled·k **先加后注入**（dk 为门控后的最终值），与 Triton 参考的语句顺序一致（ref_gpu/fla_chunk_inter.py L140-153）。

### 1.4 算法描述（每任务处理一个 (dk-block, chunk, b, h)，DK-block 沿 grid 维）

1. **prologue（V 侧）**：`dgkn` 累加器清零（fill 0.0）。
2. **i_v 循环（Pipelined，恰好 2 次迭代，bDV = DV/2）**：
   - V 侧：5 路 GM→UB fp32 载入（h/dh [bDK,bDV]、v/do/dv [BT,bDV]）；
   - dgkn 部分和：`tile.mul`（h⊙dh，in-place 复用 dh_ub 槽位，见 §3.3 顺序说明）→ `reduce_sum(dim=1, clear=True)` → `tile.add` 累加（规避 in-loop clear=False 工具链 bug）；
   - **fp16 hi/lo 拆分**（fp32 gemm_v0 工具链损坏的强制替代，★fla_wy_fast §8-1）：5 个操作数逐个 `cast→sub→cast` 拆为 hi/lo 对，经 UB→L1 中继供 gemm（dw 组使用专用 h2 副本，规避多 gemm 组共享 L1 操作数，§8.6）；dv 在拆分前取负（−1 标量乘），将 dw 的负号**精确折叠**进 GEMM 操作数（(−dv) = −hi − lo，逐位恒等）；
   - C 侧：9 个 `gemm_v0(transpose_B=True)`（3 组 × hi@hi + hi@lo + lo@hi，L0C 内 fp32 累加，首迭代 init=(i_v==0)）。
3. **epilogue（C→V 单向 + V 侧尾段）**：
   - C 侧：`dw_frag → GM` 直存（已含负号，k1 同款 f32 frag→f32 GM）；`dq_frag/dk_frag → UB`（C→V，2 路同向）；
   - V 侧：载入 g [BT,bDK]、gn（chunk 末行切片 [bDK]）、q/k [BT,bDK]、LastRow 常量 [BT]；门控链：dq ⊙= scale·exp(g·ln2)，dk ⊙= exp((gn−g)·ln2)，dgkn ⊙= exp(gn·ln2)；`dkk = dk⊙k` → `reduce_sum(dim=0)` → `pp = dgkn + Σ_t dkk`；`dg = q⊙dq − dkk + (LastRow⊙pp 广播)`；
   - 4 路 GM 写（dq/dk/dg 由 V 侧，dw 已由 C 侧直存）。

> **⚠️ Host 侧 Buffer 操作约束**（ascend-constraints.md §5）：host 路径只含 (a) 输入张量构造（CPU randn + H2D）、(b) LastRow 常量构造（一次性 [64] 常量输入，per-device 缓存）、(c) kernel 调用（13 张量直传，bshd/5D 布局直读，**零 view/零 reshape/零 permute**）、(d) CPU 侧精度验证。无 `.contiguous()`/pad/cat/`reshape` 物理化/aclnn 调用。审计见 §9.6。

### 1.5 数据流图

```
GM[h/dh:fp32] --T.copy(5D切片)--> UB[fp32] --(先 raw h 用于 h⊙dh；dh 拆分后 in-place 乘)--> reduce(dim=1) --> dgkn 累加
UB[h/dh/v/do/dvn(取负):fp32] --cast→sub→cast 拆分--> UB[hi/lo:fp16] --UB→L1 中继--> L1[fp16]
GM[v/do/dv:fp32] --T.copy(4D切片)--> UB[fp32]（并入上行拆分；含 dw 组专用 h2 副本）
L1[do_hi/lo] × L1[h_hi/lo]  --gemm_v0(ᵀB)×3 累加--> L0C[dq_frag:fp32]（i_v ×2 累加）
L1[v_hi/lo]  × L1[dh_hi/lo] --gemm_v0(ᵀB)×3 累加--> L0C[dk_frag:fp32]
L1[dvn_hi/lo]× L1[h2_hi/lo] --gemm_v0(ᵀB)×3 累加--> L0C[dw_frag:fp32] --T.copy--> GM[dw:fp32]（C 侧直存，负号已折叠）
L0C[dq/dk_frag] --T.copy--> UB[fp32] --门控链(exp·ln2/scale/broadcast)--> ┐
GM[g/gn/q/k/LastRow:fp32] --T.copy--> UB[fp32] --exp/乘/reduce(dim=0)/broadcast--> ├→ UB[dg/dq/dk:fp32] --T.copy--> GM
```

### 1.6 范围与分派覆盖审计

【已确认事实】范围裁剪（proto 声明 + 用户确认 2026-11-02）：定长序列（非 varlen）、scale=1.0（标量乘保留）、chunk_size=64、全 fp32。varlen（cu_seqlens/chunk_indices）分支不在范围；Triton wrapper 的 w 参数未被 kernel 使用（dw 由 −(dv@h^T) 推导），本算子不收 w。

**【已确认事实】S % 64 ≠ 0 属参考实现未定义域（任务点名核实项：m_t 掩码与 last(c) 行为）**：
- 数学层：h/dh 的 chunk 维 = S//64（floor）。S%64≠0 时最后一个部分 chunk **不存在对应的 h/dh 状态**，inter-chunk 项无定义；
- 语义权威层：Triton kernel 的 `i_tg = i_b*NT + i_t`（NT = cdiv(T,BT)）对末 chunk 越过 h 张量末尾（h 仅 T//BT 个 chunk），`boundary_check` 只作用于 (V,K) 虚拟形状、不约束已被指针算术移出张量的基址 → **OOB 读**；
- 代码级起点层：上游 TileLang CUDA 版 `BS = S // block_S` 与 grid `ceildiv(S, block_S)` 同样失配，且搬运完全无掩码 → 静默 OOB。

**本设计口径**：支持域收紧为 S % 64 == 0（与同族 fla_chunk_o / fla_wy_fast / fla_chunk_intra 一致）；host wrapper `_check_shape` 显式 assert 拒绝 S%64≠0（确定性报错替代参考实现的静默 OOB）。域内 m_t 掩码恒为全真（o_t < T 对所有完整 chunk 成立，**无需任何行掩码**）、last(c) 恒为局部行 63（LastRow 常量精确表达）。Stage 2 场景 B 将 S%64≠0 归入 **L2 负向测试**（正确拒绝）。

```text
[DISPATCH-COVERAGE]
supported_domain: S%64==0（chunked 数据模型数学要求，§1.6 三层证据）; DK ∈ 32 的倍数（bDK=32 grid 整分）;
  DV ∈ 16 的倍数且 DV≥64（bDV=DV/2 须 ≥32 满足 L0A/L0B K≥32 分形 + 8 倍数满足 32B 对齐）;
  B,H ≥ 1 任意; chunk_size=64 固定; scale 任意 float（标量乘）; dtype 固定 = 全 fp32（输入/输出/累加）
generic_fallback: fla_chunk_inter 单一通用 kernel（无 dtype/shape 特化分支；bDK/bDV 由 shape 机械推导，
  属性能参数非分派谓词）
specializations: 无
unsupported_inputs: varlen、S%64≠0（数学未定义域，wrapper assert 拒绝，L2 负向）、chunk_size≠64、
  DK 非 32 倍数、DV < 64 或非 16 倍数（bDV 分形/对齐违反）、非 fp32 dtype 变体（proto 全集仅 fp32）
  —— 均与 supported_domain 不冲突（proto 显式声明或 wrapper assert）
result: pass
```

---

## 2. 编程模式选型

### 2.1 模式结论

**Developer 模式**（用户明确指定，2026-11-02）——`alloc_shared` / `alloc_fragment` + 自动同步 + `threads=1`。

### 2.2 选型理由

算子是典型 CV 融合（3 组 GEMM + Vector exp/逐元素/双维 reduce），无跨核自定义同步需求（无全局归约、无任务间依赖）；上游 TileLang CUDA 版本身即 Developer 风格 API（alloc_shared/fragment + T.gemm/T.copy），迁移映射最直接；同族 fla_wy_fast / fla_chunk_o 均以 Developer 模式在 910B3 完整交付，其全部实证 API 链（含工具链缺陷规避清单）可直接复用。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（编译器映射 L1/UB）+ `T.alloc_fragment`（映射 L0C） |
| GEMM | `T.gemm_v0(A, B, C, transpose_B=True, init=)`：A/B 为 shared(L1) **fp16 hi/lo**，C 为 fragment(L0C) fp32（fp32 A/B 工具链损坏，★fla_wy_fast §8-1） |
| 计算方式 | `T.tile.mul / sub / add / exp / fill / broadcast`（buffer 级全 tile 指令；★T.Parallel 逐元素/逐行循环为反模式，-26%~-71%/kernel） |
| 作用域 | 编译器自动 C/V 分离（AUTO_CV_COMBINE） |
| 同步方式 | 自动同步（AUTO_SYNC + AUTO_CV_SYNC），无手动 barrier/cross_flag |
| threads | **1**（偏离模板默认 threads=2 的理由：★fla_wy_fast threads=2 的 gather 链 vid 切分越界先例 + 其最终交付全 kernel threads=1；本算子无 gather 链，threads=2（fla_chunk_o 交付形态，可唤醒 40 AIV）列为 Stage 3 调优候选） |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| F1 | `x = hi + lo`（5 操作数 do/v/dv/h/dh） | fp16 hi/lo 拆分：hi=f16(x)，lo=f16(x−f32(hi))；误差 ~2⁻²² 相对 |
| F2 | `dv' = −dv` | dw 负号折叠（拆分前一次性标量乘；(−x) 的 hi/lo = −(x 的 hi/lo)，逐位恒等） |
| F3 | `dgkn += Σ_v h⊙dh`（逐 i_v 累加） | UB fp32 域 mul + reduce(dim=1) + add |
| F4 | `dq_raw += do @ h^T`（i_v 累加） | gemm_v0 ×3（hi@hi / hi@lo / lo@hi） |
| F5 | `dk_raw += v @ dh^T`（i_v 累加） | 同上 |
| F6 | `dw_frag += (−dv) @ h^T`（i_v 累加） | 同上（负号已在 F2 折叠） |
| F7 | `dq = dq_raw · scale · 2^g` | UB：exp(g·ln2) 链 + 标量乘（exp2 无昇腾后端，§3.5.2-2） |
| F8 | `dk = dk_raw · 2^(gn−g)`；`dgkn ⊙= 2^gn` | UB：gn 行广播(axis=0) + sub + exp 链 |
| F9 | `pp = dgkn + Σ_t dk⊙k` | UB：mul + reduce(dim=0, clear=True) + add |
| F10 | `dg = q⊙dq − k⊙dk + m_last ⊙ pp` | UB：LastRow 常量广播(axis=1) + mul + sub + add |

### 3.2 TileLang API 映射（含出处）

| 步骤 | TileLang API（昇腾） | 参数 | 出处（已验证） |
|------|---------------------|------|---------------|
| Kernel | `T.Kernel(GRID, threads=1, is_npu=True) as (cid)` | 一维；cid 分解 bk=cid%NK（最快）→ bsh=cid//NK → bs=bsh%NT, bbh=bsh//NT, bb=bbh//H, bh=bbh%H | ★fla_wy_fast impl_final.py:130（threads=1 同款）；`examples/linear_attention_and_rnn/gdn/gdn_chunk_o.py:35`（chunk×head 一维化先例） |
| GEMM | `T.gemm_v0(A, B, C, transpose_B=True, init=)` | A/B→L1 fp16、C→L0C fp32；init=(i_v==0) 首迭代覆盖 / False 累加 | `tilelang/language/ascend.py:418`（签名实证）；★impl_final.py:395-403（hi/lo 三连 gemm + init 对，硬件 PASS） |
| 内存分配 | `T.alloc_shared(shape, dtype)` / `T.alloc_fragment(shape, dtype)` | shared→L1/UB、fragment→L0C | ★impl_final.py:330-353 |
| 拆分链 | `T.copy(f32, f16)` → `T.copy(f16, f32)` → `T.tile.sub(lo_f32, x, hi_f32)` → `T.copy(lo_f32, lo_f16)` | 跨 dtype 只在 UB↔UB 内两步完成（GM↔UB cast-on-copy 缺陷规避★§8-6） | ★impl_final.py:380-392（k1 拆分链逐行同款） |
| 逐元素 | `T.tile.mul(dst, src0, src1/标量)`（含 in-place dst==src0）、`T.tile.sub/add`、`T.tile.exp(dst, src)`、`T.tile.fill(buf, 0.0)` | fp32 域全 tile 指令 | `tilelang/language/ascend_tile.py:1113/1130/1152/1277/352`；★impl_final.py:369-374, 380-392 |
| 广播 | `T.tile.broadcast(dst, src, axis=0/1)` | axis=0：[N]→[M,N] 行复制；axis=1：[M]→[M,N] 行标量 | `ascend_tile.py:2507`（签名实证）；★impl_final.py:372（axis=1）；`examples/mha_sink_fwd_bhsd/mha_sink_fwd_bhsd.py`、`examples/pos_embedding/rope.py` |
| 归约 | `T.reduce_sum(buf_ub, out_ub, dim=1/dim=0, clear=True)` | **限 UB**；in-loop 用 clear=True + tile.add 累加（clear=False in-loop 为工具链 bug） | `tilelang/language/reduce_ascend.py:456`（签名 + UB 约束）；dim=0：`examples/HISA/paged_block_sparse_mqa_attn_expert.py:343`、`examples/tile_kernels/mhc/head_compute_mix_kernel.py:147`；in-loop 累加修复模式：★debug_log Attempt 1（flash_attn sumexp 模式，硬件 PASS） |
| 搬运 | `T.copy(GM 切片, ub 同 dtype)` / `T.copy(ub, l1)` / `T.copy(frag, ub)` / `T.copy(frag, GM 切片)` | f32 frag→f32 GM 直存合法（同 dtype，无 fixpipe 转换） | ★impl_final.py:361-364（GM→UB f32）、384-385（UB→L1）、405（frag→GM 直存）；5D 切片：`gdn_chunk_o.py:66` + ★fla_chunk_o impl（5D h 切片交付实证） |
| 循环 | `T.Pipelined(DV // bDV, num_stages=2)` | **恰好 2 次迭代**（extent-1 崩溃 / extent≥3 回卷损坏，★§8-2） | ★impl_final.py:360（extent-2 同款）；约束见 §3.5.1 |
| 常量掩码 | `LastRow[64]` fp32 GM 输入 → UB → `T.tile.broadcast(axis=1)` → `T.tile.mul` | host 一次性构造（行 63 = 1，其余 0），per-device 缓存 | ★fla_wy_fast Msk 常量模式（_MSK_CACHE 同款）；tril 常量掩码先例 `gdn_chunk_scaled_dot_kkt.py:59,76` |

### 3.3 计算伪代码（kernel 主体，bench 配置 bDK=32 / bDV=64 为例）

```python
LN2 = 0.6931471805599453   # 2^x = exp(x·ln2)：exp2 无昇腾后端（src/op/ascend.cc 仅 ascend_exp）

@tilelang.jit(out_idx=[9, 10, 11, 12], pass_configs=PASS_CONFIGS)   # 绝对索引（★out_idx 负索引错位坑）
def fla_chunk_inter(B, S, H, DK, DV, scale=1.0, chunk_size=64, block_DK=32):
    # wrapper 侧 assert：S % chunk_size == 0、DK % block_DK == 0、DV % 16 == 0 且 DV >= 64、chunk_size == 64
    BS = chunk_size; NT = S // BS; NK = DK // block_DK
    bDV = DV // 2                       # i_v 循环 extent 恒为 2（工具链唯一安全值）
    GRID = NK * NT * B * H

    @T.prim_func
    def main(
        Q:  T.Tensor((B, S, H, DK), "float32"),   K:  T.Tensor((B, S, H, DK), "float32"),
        V:  T.Tensor((B, S, H, DV), "float32"),   G:  T.Tensor((B, S, H, DK), "float32"),
        h:  T.Tensor((B, NT, H, DK, DV), "float32"),
        dv: T.Tensor((B, S, H, DV), "float32"),   DO: T.Tensor((B, S, H, DV), "float32"),
        Dh: T.Tensor((B, NT, H, DK, DV), "float32"),
        LastRow: T.Tensor((BS,), "float32"),      # 设计新增常量：m_last 行指示（域内恒为行 63）
        dq: T.Tensor((B, S, H, DK), "float32"),   dk: T.Tensor((B, S, H, DK), "float32"),
        dw: T.Tensor((B, S, H, DK), "float32"),   dg: T.Tensor((B, S, H, DK), "float32"),
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            # bk 最快变化：同一 (chunk,head) 的 NK 个 dk-block 任务拿相邻 cid → 同一波次调度
            # → v/do/dv 的 NK 重读由 L2 吸收（§5.2 / §9.5）
            bk = cid % NK; bsh = cid // NK
            bs = bsh % NT; bbh = bsh // NT
            bb, bh = bbh // H, bbh % H
            s0 = bs * BS; dk0 = bk * block_DK

            # ---- fragment（L0C，fp32，跨 i_v 累加）----
            dq_frag = T.alloc_fragment((BS, block_DK), "float32")
            dk_frag = T.alloc_fragment((BS, block_DK), "float32")
            dw_frag = T.alloc_fragment((BS, block_DK), "float32")
            # ---- UB：fp32 载入域 ----
            h_ub  = T.alloc_shared((block_DK, bDV), "float32")
            dh_ub = T.alloc_shared((block_DK, bDV), "float32")   # dh 拆分后 in-place 复用为 h⊙dh（§4.5 省槽）
            v_ub  = T.alloc_shared((BS, bDV), "float32")
            do_ub = T.alloc_shared((BS, bDV), "float32")
            dvn_ub = T.alloc_shared((BS, bDV), "float32")
            # ---- UB：拆分暂存（共享，逐操作数交错复用 lifetime）----
            t_hi = T.alloc_shared((BS, bDV), "float32");        t_lo = T.alloc_shared((BS, bDV), "float32")
            th_hi = T.alloc_shared((block_DK, bDV), "float32"); th_lo = T.alloc_shared((block_DK, bDV), "float32")
            # ---- UB：f16 hi/lo（中继前暂存）----
            do_hi = T.alloc_shared((BS, bDV), "float16"); do_lo = T.alloc_shared((BS, bDV), "float16")
            v_hi  = T.alloc_shared((BS, bDV), "float16"); v_lo  = T.alloc_shared((BS, bDV), "float16")
            dvn_hi = T.alloc_shared((BS, bDV), "float16"); dvn_lo = T.alloc_shared((BS, bDV), "float16")
            h_hi  = T.alloc_shared((block_DK, bDV), "float16"); h_lo  = T.alloc_shared((block_DK, bDV), "float16")
            dh_hi = T.alloc_shared((block_DK, bDV), "float16"); dh_lo = T.alloc_shared((block_DK, bDV), "float16")
            # ---- L1：gemm 操作数（f16）----
            do_hi_l1 = T.alloc_shared((BS, bDV), "float16"); do_lo_l1 = T.alloc_shared((BS, bDV), "float16")
            v_hi_l1  = T.alloc_shared((BS, bDV), "float16"); v_lo_l1  = T.alloc_shared((BS, bDV), "float16")
            dvn_hi_l1 = T.alloc_shared((BS, bDV), "float16"); dvn_lo_l1 = T.alloc_shared((BS, bDV), "float16")
            h_hi_l1  = T.alloc_shared((block_DK, bDV), "float16"); h_lo_l1  = T.alloc_shared((block_DK, bDV), "float16")
            h2_hi_l1 = T.alloc_shared((block_DK, bDV), "float16"); h2_lo_l1 = T.alloc_shared((block_DK, bDV), "float16")  # dw 组专用 h 副本（§8.6）
            dh_hi_l1 = T.alloc_shared((block_DK, bDV), "float16"); dh_lo_l1 = T.alloc_shared((block_DK, bDV), "float16")
            # ---- UB：dgkn 累加器 ----
            dgkn_ub = T.alloc_shared((block_DK,), "float32"); dgkn_pt = T.alloc_shared((block_DK,), "float32")

            T.tile.fill(dgkn_ub, 0.0)

            for i_v in T.Pipelined(DV // bDV, num_stages=2):        # 恰好 2 次迭代
                dv0 = i_v * bDV
                T.copy(h[bb, bs, bh, dk0:dk0+block_DK, dv0:dv0+bDV], h_ub)     # 5D 尾维连续切片
                T.copy(Dh[bb, bs, bh, dk0:dk0+block_DK, dv0:dv0+bDV], dh_ub)
                # -- dh 拆分（先于 in-place 乘：raw dh 先被消费）→ 中继 --
                T.copy(dh_ub, dh_hi); T.copy(dh_hi, th_hi)
                T.tile.sub(th_lo, dh_ub, th_hi); T.copy(th_lo, dh_lo)
                T.copy(dh_hi, dh_hi_l1); T.copy(dh_lo, dh_lo_l1)
                # -- dgkn 部分和（dh_ub in-place 复用为 h⊙dh；mul 交换律取 dst==src0 已验证形态）--
                T.tile.mul(dh_ub, dh_ub, h_ub)
                T.reduce_sum(dh_ub, dgkn_pt, dim=1, clear=True)
                T.tile.add(dgkn_ub, dgkn_ub, dgkn_pt)
                # -- h 拆分 → 中继（dq 组 + dw 组专用副本 h2）--
                T.copy(h_ub, h_hi); T.copy(h_hi, th_hi)
                T.tile.sub(th_lo, h_ub, th_hi); T.copy(th_lo, h_lo)
                T.copy(h_hi, h_hi_l1);  T.copy(h_lo, h_lo_l1)
                T.copy(h_hi, h2_hi_l1); T.copy(h_lo, h2_lo_l1)
                # -- v：载入 → 拆分 → 中继 --
                T.copy(V[bb, s0:s0+BS, bh, dv0:dv0+bDV], v_ub)
                T.copy(v_ub, v_hi); T.copy(v_hi, t_hi)
                T.tile.sub(t_lo, v_ub, t_hi); T.copy(t_lo, v_lo)
                T.copy(v_hi, v_hi_l1); T.copy(v_lo, v_lo_l1)
                # -- do：同 --
                T.copy(DO[bb, s0:s0+BS, bh, dv0:dv0+bDV], do_ub)
                T.copy(do_ub, do_hi); T.copy(do_hi, t_hi)
                T.tile.sub(t_lo, do_ub, t_hi); T.copy(t_lo, do_lo)
                T.copy(do_hi, do_hi_l1); T.copy(do_lo, do_lo_l1)
                # -- dv：载入 → 取负（dw 符号折叠）→ 拆分 → 中继 --
                T.copy(dv[bb, s0:s0+BS, bh, dv0:dv0+bDV], dvn_ub)
                T.tile.mul(dvn_ub, dvn_ub, -1.0)
                T.copy(dvn_ub, dvn_hi); T.copy(dvn_hi, t_hi)
                T.tile.sub(t_lo, dvn_ub, t_hi); T.copy(t_lo, dvn_lo)
                T.copy(dvn_hi, dvn_hi_l1); T.copy(dvn_lo, dvn_lo_l1)
                # -- 9 gemms（3 组 × hi/lo 三连，L0C fp32 累加）--
                T.gemm_v0(do_hi_l1, h_hi_l1, dq_frag, transpose_B=True, init=(i_v == 0))
                T.gemm_v0(do_hi_l1, h_lo_l1, dq_frag, transpose_B=True, init=False)
                T.gemm_v0(do_lo_l1, h_hi_l1, dq_frag, transpose_B=True, init=False)
                T.gemm_v0(v_hi_l1, dh_hi_l1, dk_frag, transpose_B=True, init=(i_v == 0))
                T.gemm_v0(v_hi_l1, dh_lo_l1, dk_frag, transpose_B=True, init=False)
                T.gemm_v0(v_lo_l1, dh_hi_l1, dk_frag, transpose_B=True, init=False)
                T.gemm_v0(dvn_hi_l1, h2_hi_l1, dw_frag, transpose_B=True, init=(i_v == 0))
                T.gemm_v0(dvn_hi_l1, h2_lo_l1, dw_frag, transpose_B=True, init=False)
                T.gemm_v0(dvn_lo_l1, h2_hi_l1, dw_frag, transpose_B=True, init=False)

            # ---- epilogue：C 侧直存 dw（负号已折叠）+ C→V 双路同向 ----
            T.copy(dw_frag, dw[bb, s0:s0+BS, bh, dk0:dk0+block_DK])
            T.copy(dq_frag, dq_ub)      # UB [BS, block_DK] fp32
            T.copy(dk_frag, dk_ub)
            # ---- V 侧尾段 ----
            T.copy(G[bb, s0:s0+BS, bh, dk0:dk0+block_DK], g_ub)
            T.copy(G[bb, s0+BS-1, bh, dk0:dk0+block_DK], gn_ub)     # chunk 末行 1D 连续切片（域内恒局部行 63）
            T.copy(LastRow[0:BS], ml_ub)
            T.copy(Q[bb, s0:s0+BS, bh, dk0:dk0+block_DK], q_ub)
            T.copy(K[bb, s0:s0+BS, bh, dk0:dk0+block_DK], k_ub)
            # dq = dq_raw · scale · 2^g
            T.tile.mul(gexp_ub, g_ub, LN2); T.tile.exp(gexp_ub, gexp_ub)
            T.tile.mul(dq_ub, dq_ub, gexp_ub); T.tile.mul(dq_ub, dq_ub, scale)
            # dgkn ⊙= 2^gn；dk ⊙= 2^(gn−g)
            T.tile.mul(gnexp_ub, gn_ub, LN2); T.tile.exp(gnexp_ub, gnexp_ub)
            T.tile.mul(dgkn_ub, dgkn_ub, gnexp_ub)
            T.tile.broadcast(gn2d_ub, gn_ub, axis=0)
            T.tile.sub(gnd_ub, gn2d_ub, g_ub); T.tile.mul(gnd_ub, gnd_ub, LN2); T.tile.exp(gnd_ub, gnd_ub)
            T.tile.mul(dk_ub, dk_ub, gnd_ub)
            # pp = dgkn + Σ_t dk⊙k（dkk 同时复用为 dg 的 −k⊙dk 项）
            T.tile.mul(dkk_ub, dk_ub, k_ub)
            T.reduce_sum(dkk_ub, pp_ub, dim=0, clear=True)
            T.tile.add(pp_ub, pp_ub, dgkn_ub)
            # dg = q⊙dq − k⊙dk + m_last⊙pp
            T.tile.broadcast(pp2d_ub, pp_ub, axis=0)
            T.tile.broadcast(ml2d_ub, ml_ub, axis=1)
            T.tile.mul(mp2d_ub, pp2d_ub, ml2d_ub)
            T.tile.mul(dg_ub, q_ub, dq_ub)
            T.tile.sub(dg_ub, dg_ub, dkk_ub)
            T.tile.add(dg_ub, dg_ub, mp2d_ub)
            # ---- 3 路 GM 写（dw 已直存）----
            T.copy(dq_ub, dq[bb, s0:s0+BS, bh, dk0:dk0+block_DK])
            T.copy(dk_ub, dk[bb, s0:s0+BS, bh, dk0:dk0+block_DK])
            T.copy(dg_ub, dg[bb, s0:s0+BS, bh, dk0:dk0+block_DK])
    return main
```

> 伪代码中 epilogue 侧 UB buffer（g_ub/gexp_ub/gn_ub/gnexp_ub/gn2d_ub/gnd_ub/q_ub/k_ub/dq_ub/dk_ub/dkk_ub/dg_ub/pp_ub/pp2d_ub/ml_ub/ml2d_ub/mp2d_ub，均 [BS,block_DK] 或 [block_DK]/[BS] fp32）与循环侧 buffer 生命周期不重叠，由 MEMORY_PLANNING 复用同一物理空间（§4.5）。buffer 归属（L1 vs UB）由 AUTO_CV_COMBINE + MEMORY_PLANNING 规划；gemm 输入（`*_l1`）必须落 L1，元素计算/载入 buffer 落 UB——★impl_final 命名约定同款。

### 3.4 API 可行性确认

全部 API 均有本项目实证出处（§3.2 表逐行标注源码行号 / examples 路径 / ★同族交付代码），**无凭记忆 API**。本算子操作集是 ★fla_wy_fast impl_final 已验证操作的组合，新增/扩展的 4 类组合及依据：

1. **单 Pipelined 体内 3 gemm 组（9 gemm_v0、3 个 L0C 累加器）**：k1 实证了 1 组 3 gemm 进 1 累加器（★impl_final.py:394-403）；本设计 3 组各用**独立不共享** L1 操作数（含 dw 组专用 h2 副本），无角色互换复用（★§8-3 损坏模式的触发条件是"同一 buffer 在不同 gemm 中担任左/右不同角色"）；同族 DESIGN_fla_chunk_intra 规划单 Pipelined 体 48 gemm_v0（16 组）为家族设计包络。列为【需 Stage 2 验证】R1，含 3-kernel 拆分回退（§10.4-E）。
2. **循环后 C 侧混合出口（dw frag→GM 直存 + dq/dk frag→UB 双路 C→V）**：k1 实证 frag→GM（★:405）、z/da kernel 实证 gemm 后 frag→UB→tile 链→GM（★da kernel）；本设计为两者并列（无 V→C→V 交替，规避 ★§8-4 sync-point 失配触发形态）。列为【需 Stage 2 验证】R1'，回退：dw 改走 frag→UB（+8KB UB + 2 op）。
3. **dim=0 / dim=1 双向 UB reduce**：dim=1 in-loop（clear=True + tile.add）= ★debug_log Attempt 1 修复模式；dim=0 直线段（epilogue）= HISA/mhc examples 先例。
4. **in-place `T.tile.mul(dh_ub, dh_ub, h_ub)` 复用槽位**：in-place dst==src0 形态实证（★impl_final.py:374 `T.tile.mul(kbg_f32, kbg_f32, gexp_ub)`）；用于消除独立 hdh buffer（§4.5 UB 预算），顺序约束（raw dh 先被拆分消费）已在伪代码中固化。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（强制检测）

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **是**（上游 `T.Kernel(ceildiv(DK,block_DK), ceildiv(S,block_S), B*H, threads=128)` 三维） | 一维化：`T.Kernel(NK·NT·B·H)` + **bk 最快变化**的 cid 分解（§5.2 L2 局部性）；gdn_chunk_o.py:35 / ★impl_final 同款 |
| threads 仅 1/2 | **是**（上游 threads=128） | `threads=1`（★fla_wy_fast 最终交付同款；threads=2 列 Stage 3 候选，见 §2.3） |
| 动态循环边界 | 否 | B/S/H/DK/DV 均为 JIT 编译期常量；`DV // bDV` 静态 |
| Pipelined 动态边界 / **extent 约束** ⚠️ | **是**（关键） | ★工具链实证：单 cross-core Pipelined 体 extent=1 崩溃、extent≥3 双缓冲回卷损坏、**唯一安全 extent=2**（final_summary §8-2；D=96/block32/3 迭代实测 2/3 切片损坏）。本设计 `bDV = DV // 2` 强制 i_v 循环恰好 2 次迭代 → **DV 必须为偶数**；叠加 bDV≥32 分形（K=bDV）与 32B 对齐（bDV×4B）→ 支持域 DV ∈ 16 的倍数且 ≥64（§1.6） |
| **每 kernel 单 cross-core Pipelined 体** ⚠️ | 已遵守 | 唯一循环 = i_v Pipelined 体；epilogue 为直线段（无第二循环体） |
| L0C 溢出（128KB） | 已核算 | 3 frags [64, 32] fp32 = **24KB = 19%** 限额 ✓（§4.5） |
| T.copy 列方向 strided 切片 | 已规避 | 全部 GM 切片为"标量前缀 + 行区间 × 最内维连续 block"：q/k/g [bb, s0:s0+64, bh, dk0:dk0+32]（4D 最内维连续）、v/do/dv [bb, s0:s0+64, bh, dv0:dv0+64]、h/dh [bb, bs, bh, dk0:dk0+32, dv0:dv0+64]（5D 最内维 DV 连续，行间 gap 合法）、gn 末行 [bb, row, bh, dk0:dk0+32]（1D 连续 ✓★"gk 末行切片"同款） |
| GEMM 分形限制 | 已核算 | fp16：M≥16, K≥32(L0A/L0B), N≥16（decision-tree §3）。gemm 形状 M=64, K=bDV∈{32,48,64,...} ✓, N=bDK=32 ✓ |
| GEMM 非整除 | 已规避 | 声明域内：S%64==0、DK%32==0（grid 整分）、DV%(DV/2)==0 恒成立（bDV=DV/2 构造）。域外 shape 由 wrapper assert 拒绝（L2 负向） |
| **fp32 gemm_v0 损坏** ⚠️（★final_summary §8-1：fp32 A/B 产出 1e37 级交替垃圾） | **是（本算子全部 GEMM 操作数天然 fp32）** | **fp16 hi/lo 3-gemm 拆分**（★fla_wy_fast bwd 同款，CPU mimic ratio=1.0000 + 硬件 L0/L1 PASS）：x = hi + lo，C = hi@hi + hi@lo + lo@hi（L0C fp32 累加），误差 ~2⁻²² 相对。已知限制：\|操作数\| > 65504（fp16 范围）精度退化——本算子 gemm 操作数为**原始** do/v/dv/h/dh（门控 2^g 在 gemm 之后的 V 侧 fp32 域施加，**不进入 gemm 操作数**，与 fla_wy_fast 的 kbg=β·k·2^g 结构性不同），randn/训练域 \|x\| 远小于 65504，仅对抗性极端输入退化（Boundary 文档化） |
| **GM↔UB cast-on-copy 缺陷** ⚠️（★§8-6：bf16↔fp32 模板错配） | 已规避 | 全部跨 dtype 转换在 **UB↔UB 内**两步完成（GM f32 → UB f32 → UB f16）；本算子无 bf16 |
| **L0C(fp32)→GM 直存** ⚠️（★§8-7：fixpipe 跨 dtype 转换损坏） | 已规避 | 本算子输出全 fp32：dw_frag→GM 为**同 dtype** 直存（★k1 :405 实证）；dq/dk 经 frag→UB→GM（同 dtype） |
| **serial 循环内 V→C 交换 mis-sync**（★模块 docstring-1） | 已规避 | 无 serial 循环；全部 V→C 交换（12 路 UB→L1/迭代）在 Pipelined 体内 |
| **多 gemm 组共享 L1 操作数（角色互换）损坏 L0 staging**（★§8-3） | 已规避（防御性） | 3 组操作数完全独立：do/v/dvn/dh 各一组专用，h 在 dq 组与 dw 组间**同角色**（均为 transpose_B 右操作数）共享 UB 源但 L1 落地用专用副本 h2_hi/lo_l1 → 零共享 L1 buffer、零角色互换（R1 缓解，§3.4-1） |
| **后置多跳 C/V 交替 sync-point 失配**（★§8-4） | 已规避 | 循环后 C 侧出口全部同向（1×C→GM + 2×C→V），无 V→C→V 交替；V 侧尾段纯直线（da kernel 同构） |
| **prologue V→C 交换错版本化**（★§8-5） | 已规避 | prologue 仅 V 侧 fill（无 V→C 交换）；全部交换在循环体内 |
| **in-loop reduce_sum(clear=False) 累加错误**（★debug_log Attempt 1） | 已规避 | dgkn 累加用 clear=True 归约到 part + tile.add（flash_attn sumexp 修复模式） |
| **out_idx 负索引错位**（★fla_chunk_o debug_log） | 已规避 | `out_idx=[9,10,11,12]` 绝对索引（LastRow 之后、13 参数签名） |
| **UB 硬上限 196352B**（★fla_chunk_o 4-chunk 实证） | 已核算 | 循环侧声明 184.25KB = 188,672B ≤ 196,352B（余量 7.5KB）；epilogue 侧 ~96.5KB 分时复用（§4.5） |
| host 侧 Buffer 约束（ascend-constraints §5） | 已审计 | §9.6 [HOST-METADATA-AUDIT]：零 view / 零 reshape / 零 aclnn（bshd + 5D 直读） |

#### 3.5.2 CUDA 专属机制的昇腾适配（对照上游 fla_chunk_inter_tilelang_gpu.py）

| # | CUDA 机制（上游） | 昇腾适配结论 | 依据 |
|---|------------------|-------------|------|
| 1 | `T.Kernel(ceildiv(DK,block_DK), ceildiv(S,block_S), B*H, threads=128)` 三维 + threads=128 | 一维化 `T.Kernel(NK·NT·B·H, threads=1)`，**bk 最快** cid 分解（L2 局部性，§5.2） | ascend-constraints §1；gdn_chunk_o.py:35；★impl_final.py:130 |
| 2 | `T.exp2(x)`（T.Parallel 逐元素内） | **替代**：`T.tile.mul(gL, g, LN2)` + `T.tile.exp(gL, gL)`（buffer 级；src/op/ascend.cc 仅 ascend_exp 映射，无 exp2）。精度：fp32 下 exp(x·ln2) vs exp2(x) 相对差 ~1e-7 << 门禁 2⁻¹⁰ | ★impl_final.py:369-370 同款链路（源码复核 ascend.cc:1085） |
| 3 | `T.gemm(A_shared, B_shared, C_fragment, transpose_B=True, clear_accum=False)`（fp32） | `T.gemm_v0(A_l1, B_l1, C_frag, transpose_B=True, init=)`，操作数 **fp16 hi/lo 拆分 ×3 gemm**（fp32 gemm_v0 工具链损坏）；clear_accum→init 参数名 | ascend.py:418；★impl_final.py:395-403（hi/lo 三连 + init 对） |
| 4 | `T.clear(dq_fragment)` 等预清零 | `init=(i_v == 0)` 首迭代参数化（首块覆盖 / 后续累加），无独立 clear | gdn_chunk_o.py:67；★impl_final.py:181-182, 400-403 |
| 5 | `T.reduce_sum(dgkn_shared, dgkn_fragment, dim=1, clear=True)`（shared→fragment 跨层） | 昇腾 reduce **限 UB**：`T.reduce_sum(ub, ub_out, dim=1, clear=True)` + `T.tile.add` 跨迭代累加（fragment 不作 reduce 输出） | reduce_ascend.py:456-494 "Must be in UB"；★debug_log Attempt 1 修复模式 |
| 6 | `T.if_then_else(chunk_last_idx == bs*block_S + i_s4, pp[i_k4], 0.0)`（T.Parallel 内 m_last 行选择） | **T.Parallel 内不可用**（SIMD 无分支）→ `LastRow[64]` fp32 常量输入（域内恒行 63=1）+ `T.tile.broadcast(axis=1)` + `T.tile.mul` | ★impl_final da kernel 掩码同构；gdn_chunk_scaled_dot_kkt.py 常量掩码先例 |
| 7 | `@autotune`（block_DK×block_DV×threads×num_stages 4 维搜索） | **L0 阶段不启用**：bDK=32（UB/分形约束主选）、bDV=DV//2（extent-2 唯一解）、threads=1、num_stages=2——四参数约束确定；Stage 3 枚举 bDK∈{32,64} / threads∈{1,2} | ★final_summary §8-2；§5.2 |
| 8 | Triton `tl.where(m_t[:,None], exp2(...), 0)` 行掩码 | 域内（S%64==0）m_t 恒全真 → **无需掩码**；域外为数学未定义域（§1.6 三层证据），wrapper assert 拒绝 | ref_gpu/fla_chunk_inter.py L90-91, L143 语义分析 |
| 9 | `tl.store(p_dw, -b_dw)`（store 取负） | dv 载入后一次性 `T.tile.mul(dvn_ub, dvn_ub, -1.0)`，负号折叠进 hi/lo 拆分（(−x) 的 hi/lo = −(x 的 hi/lo)，f16(−x)=−f16(x) 逐位恒等）→ dw_frag 直存即为负值 | §3.4；数学恒等 |
| 10 | `out_idx=[-4,-3,-2,-1]` 负索引 | **绝对索引 [9,10,11,12]**（LastRow 追加在输入区后；自动注入 workspace 参数会使负索引错位） | ★fla_wy_fast docstring #7；★fla_chunk_o debug_log out_idx 坑 |

#### 3.5.3 参考实现差异说明（GPU → Ascend 汇总）

| 差异项 | 参考实现（CUDA/上游 TileLang） | 本项目（Ascend 910B3） | 转换方案 |
|--------|------------------------------|----------------------|---------|
| Kernel 维度 | 三维 (bk, bs, bbh) | 一维 NK·NT·B·H（bk 最快） | §3.5.2-1 |
| threads | 128（SIMT warp） | 1（C/V 自动协作） | §3.5.2-1 |
| GEMM API/精度 | `T.gemm` fp32 直算 | `T.gemm_v0` fp16 hi/lo ×3 | §3.5.2-3 |
| GEMM 物理次数 | 3/迭代 | 9/迭代（hi/lo 代价，工具链强制） | §9.5 性能预算 |
| exp2 | `T.exp2` / `tl.exp2` | `T.tile.exp(x·ln2)` | §3.5.2-2 |
| fragment reduce | shared→fragment reduce_sum | UB→UB reduce + tile.add | §3.5.2-5 |
| m_last 行选择 | `T.if_then_else` 逐元素 | LastRow 常量 + broadcast + tile.mul | §3.5.2-6 |
| m_t 行掩码 | boundary_check / tl.where | 域内恒真无需掩码；域外 assert 拒绝 | §3.5.2-8 |
| dw 取负 | store 时取负 | dv 拆分前取负折叠 | §3.5.2-9 |
| V↔C 数据交换 | SMEM 单层直连 | **GM workspace 自动注入中继**（无 UB↔L1 直连路径，★fla_wy_fast final_summary §4） | §9.5 逻辑流量核算 + Stage 3 fixed-core 演进路径 |
| autotune | 4 维搜索 | 全参数约束确定，Stage 3 枚举 | §3.5.2-7 |
| 元素循环 | T.Parallel(i,j) 逐元素 | T.tile.* 全 tile 指令 + broadcast | ★perf 教训（-26%~-71%） |

#### 3.5.4 本项目同类实现参考（强制步骤 0 结果）

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `../tilelang-dev-journal/fla_wy_fast/impl_final.py`（★，硬件交付实证） | **最高（同族 bwd 全 fp32）** | k1 kernel L308-407：Pipelined(2) + GM→UB f32 载入 + fp16 hi/lo 拆分链（cast→sub→cast）+ UB→L1 中继 + 3 gemm init 对 + frag→GM 直存；PASS_CONFIGS L79；out_idx 绝对索引（docstring #7）；模块 docstring 工具链约束 1-9 |
| `../tilelang-dev-journal/fla_wy_fast/perf_tuning/{fla_wy_fast_tuning,fla_wy_fast_final_summary}.md` | 最高（实测校准） | HBM 1263GB/s 实测、k1 逐 kernel 1.313ms/4096 块/逻辑 1.35GB→1030GB/s、V↔C GM 中继实证、工具链缺陷清单 §8-1~10 |
| `../tilelang-dev-journal/fla_chunk_o/`（★，硬件交付实证） | 高（同族 CV 中继型） | fixed-core + L2 常驻中继架构（R4 配方，-25%）、每任务同步协议地板核算（final_summary §5：~1.4-1.8us/任务纯同步开销、AIC scalar 53%）、UB 硬上限 196352B、out_idx 负索引坑、5D h 切片交付实证 |
| `examples/developer_mode/flash_attn_bshd_developer.py` | 高（Developer CV 融合标准骨架） | threads/alloc/片上直连/pass_configs 四件套基础范式 |
| `examples/linear_attention_and_rnn/gdn/gdn_chunk_o.py` | 高（GLA chunk O 同构） | chunk×head 一维化 grid、5D 隐状态 S 切片载入（:66）、gemm_v0 init=(i==0)（:67） |
| `examples/HISA/paged_block_sparse_mqa_attn_expert.py` / `examples/tile_kernels/mhc/head_compute_mix_kernel.py` | 中（reduce dim=0 先例） | `T.reduce_sum(ub, out, dim=0, clear=True)`（:343 / :147） |
| `examples/pos_embedding/rope.py` / `examples/mha_sink_fwd_bhsd/mha_sink_fwd_bhsd.py` | 中（broadcast 先例） | `T.tile.broadcast` axis=0/1 全 tile 广播链 |
| `examples_experiment/dsa_sparse_finetune/DESIGN_fla_chunk_intra.md` | 高（同沙箱兄弟设计） | 同为"全 fp32 + hi/lo 拆分 + Pipelined(2) 体 + V 尾段（k1 形）"结构规划（其 bwd_k_intra 单 Pipelined 体 48 gemm_v0 多组设计），S%64==0 域声明一致 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量（全部 bshd / 5D 连续，kernel 直读 GM，全 fp32）

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| q | [B, S, H, DK] | float32 | query |
| k | [B, S, H, DK] | float32 | key |
| v | [B, S, H, DV] | float32 | value |
| g | [B, S, H, DK] | float32 | 逐 token 累积 log2 衰减（2^g 语义） |
| h | [B, S//64, H, DK, DV] | float32 | chunk 级隐状态（上游 chunk_delta_h_fwd 产出，只消费） |
| dv | [B, S, H, DV] | float32 | v 的上游梯度 |
| do | [B, S, H, DV] | float32 | o 的上游梯度 |
| dh | [B, S//64, H, DK, DV] | float32 | h 的上游梯度 |
| LastRow | [64] | float32 | **设计新增常量**：m_last 行指示（行 63 = 1.0，其余 0.0；域内 last(c) 恒为局部行 63，§1.6）。host 一次性构造 + per-device 缓存（★_MSK_CACHE 模式），256B，L2 常驻 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| dq | [B, S, H, DK] | float32 | = dq_raw·scale·2^g |
| dk | [B, S, H, DK] | float32 | = dk_raw·2^(gn−g) |
| dw | [B, S, H, DK] | float32 | = −(dv@h^T)（负号折叠，C 侧直存） |
| dg | [B, S, H, DK] | float32 | = q⊙dq − k⊙dk + m_last⊙pp |

### 4.3 中间缓冲区（bench 配置 block_DK=32 / bDV=64；层级由编译器规划，命名指示预期归属）

**循环侧（Pipelined 体内，声明集合；lifetime 由 MEMORY_PLANNING 交错复用）**

| Buffer | Shape | dtype | 层级 | 用途 |
|--------|-------|-------|------|------|
| v_ub / do_ub / dvn_ub | [64, 64] | fp32 | UB | v/do/dv 载入域（dvn 为取负后 dv） |
| h_ub / dh_ub | [32, 64] | fp32 | UB | h/dh 载入域（dh_ub 拆分后 in-place 复用为 h⊙dh） |
| t_hi / t_lo | [64, 64] | fp32 | UB | v/do/dv 拆分暂存（共享，逐操作数复用） |
| th_hi / th_lo | [32, 64] | fp32 | UB | h/dh 拆分暂存（共享） |
| do_hi/lo, v_hi/lo, dvn_hi/lo | [64, 64] ×6 | fp16 | UB | 左操作数 hi/lo（中继前暂存） |
| h_hi/lo, dh_hi/lo | [32, 64] ×4 | fp16 | UB | 右操作数 hi/lo |
| do_hi/lo_l1, v_hi/lo_l1, dvn_hi/lo_l1 | [64, 64] ×6 | fp16 | **L1** | gemm 左操作数 |
| h_hi/lo_l1, h2_hi/lo_l1, dh_hi/lo_l1 | [32, 64] ×6 | fp16 | **L1** | gemm 右操作数（h2 为 dw 组专用副本） |
| dq_frag / dk_frag / dw_frag | [64, 32] ×3 | fp32 | **L0C(fragment)** | 3 组 gemm 累加器（i_v ×2 累加） |
| dgkn_ub / dgkn_pt | [32] ×2 | fp32 | UB | dgk 累加器 / 归约部分和 |

**尾段侧（epilogue，与循环侧分时复用同一 UB 空间）**

| Buffer | Shape | dtype | 层级 | 用途 |
|--------|-------|-------|------|------|
| g_ub / gexp_ub | [64, 32] | fp32 | UB | g / 2^g = exp(g·ln2) |
| gn_ub / gnexp_ub | [32] | fp32 | UB | chunk 末行 g / 2^gn |
| gn2d_ub / gnd_ub | [64, 32] | fp32 | UB | gn 广播 / 2^(gn−g) |
| q_ub / k_ub | [64, 32] | fp32 | UB | q / k |
| dq_ub / dk_ub | [64, 32] | fp32 | UB | frag→UB 中转（C→V） |
| dkk_ub | [64, 32] | fp32 | UB | dk⊙k（reduce 输入 + dg 负项复用） |
| dg_ub | [64, 32] | fp32 | UB | dg 组装 |
| pp_ub | [32] | fp32 | UB | dgk 终值 |
| pp2d_ub / ml2d_ub / mp2d_ub | [64, 32] | fp32 | UB | pp 广播 / LastRow 广播 / m_last⊙pp |
| ml_ub | [64] | fp32 | UB | LastRow 常量载入 |

### 4.4 内存搬运路径

```
GM[h/dh:fp32]  --T.copy(5D 切片)--> UB[fp32] --cast/sub/cast 拆分--> UB[hi/lo:fp16] --UB→L1 中继--> L1[fp16] --┐
GM[v/do/dv:fp32] --T.copy(4D 切片)--> UB[fp32] --(dv 先 ×-1)--> 同上拆分/中继 ----------------------------------┤→ gemm_v0(ᵀB) → L0C[fp32 ×3 frags]
UB[h_ub⊙dh_ub] --reduce(dim=1)--> UB[dgkn_pt] --tile.add--> UB[dgkn]（i_v 累加）                                 │
L0C[dw_frag] --T.copy(同 dtype)--> GM[dw:fp32]（C 侧直存）                                                        │
L0C[dq/dk_frag] --T.copy--> UB[fp32] --门控链(exp·ln2/broadcast/mul)---------------------------------------------┘
GM[g/gn/q/k/LastRow:fp32] --T.copy--> UB[fp32] --exp/乘/reduce(dim=0)/broadcast--> UB[dg/dq/dk:fp32] --T.copy--> GM
```

> 层级合法性：无 GM→L0 直搬；Cube 输入经 L1（gemm_v0 内部 L1→L0A/L0B）；Vector 计算全在 UB（fp32 域）；L0C→UB 用 T.copy 直连；跨 dtype 转换全部 UB↔UB 片内两步完成（cast-on-copy 缺陷规避★§8-6）。V↔C 交换（UB→L1 的 12 路 hi/lo 中继/迭代）经工具链自动注入的 GM workspace（无片上直连路径，★fla_wy_fast final_summary §4）——逻辑流量代价见 §9.5。

### 4.5 片上内存预算表（910B3：L1=512KB、UB 硬上限 **196352B**、L0A/L0B=64KB、L0C=128KB per core；bench 配置 bDK=32/bDV=64）

| 层级 | 构成 | 大小 | 限额 | 结论 |
|------|------|------|------|------|
| L0C | 3 frags [64,32] fp32 | **24KB** | 128KB | ✓ 19% |
| L1 | 12 gemm 操作数 f16（6×[64,64]=48KB + 6×[32,64]=24KB），Pipelined 双缓冲 ×2 | **144KB** | 512KB | ✓ 28% |
| UB（循环侧声明） | v/do/dvn f32 48 + h/dh f32 16 + 拆分暂存 t 32 + th 16 + 左操作数 hi/lo f16 48 + 右操作数 hi/lo f16 16（h_hi/lo、dh_hi/lo；h2 副本不占 UB、直落 L1）+ dgkn 0.25 | **184.25KB = 188,672B** | 196,352B | ✓ 余量 7.5KB（声明口径；k1 先例声明 ~195KB 编译通过，靠 lifetime 复用；独立 hdh buffer 已由 dh_ub in-place 复用消除） |
| UB（尾段侧声明） | g/gexp/gn2d/gnd/q/k/dq/dk/dkk/dg/pp2d/ml2d/mp2d 各 8KB + 标量 ~0.5KB | **~96.5KB** | — | ✓ 与循环侧分时复用（生命周期不重叠，MEMORY_PLANNING 复用循环侧空间） |

> UB 削峰措施（已内建）：① hdh 乘积 in-place 复用 dh_ub 槽位（拆分先行，§3.3 顺序）；② 拆分暂存 t/th 对全操作数共享 + 载入→拆分→中继逐操作数交错（伪代码顺序固化）；③ epilogue 全部 buffer 与循环侧分时。若 Stage 2 编译报 UB 溢出：回退 1 = 去掉 h2 专用 L1 副本（h2 改由 h_hi/h_lo_l1 共享，R1 风险上移，UB 不变）；回退 2 = 探索 t/th 对 region 级复用；极端回退 = bDV 不变、v/do/dv 三操作数两两分时（编译器不支持再议）。

### 4.6 动态轴定义

无运行时动态轴。B/S/H/DK/DV 均为 JIT 编译期常量（Python 参数传入，随 shape 重编译）——与上游 chunk_bwd_dqkwg 一致；NT=S//64、NK=DK//32、grid=NK·NT·B·H 编译期展开。验收基准为静态 shape（proto 声明）。

### 4.7 JIT 配置

```python
PASS_CONFIGS = {                                              # ★fla_wy_fast impl_final.py:79 同款四件套
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 C/V 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # 自动核间同步
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动核内同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # UB/L1 lifetime 规划（§4.5 削峰的关键）
}
@tilelang.jit(out_idx=[9, 10, 11, 12], pass_configs=PASS_CONFIGS)   # 绝对索引；无 workspace_idx（Developer 无显式 workspace）
```

---

## 5. Tiling 策略

### 5.1 计算类型

**混合（CV 融合）**：Cube（gemm_v0 ×9/迭代 ×2 迭代 = 18/任务）+ Vector（拆分链 20 op/迭代、exp 链、双维 reduce、broadcast、门控乘加）。

### 5.2 Block 划分

```python
block_S  = 64         # 固定 = chunk_size（语义约束，assert chunk_size == block_S）
block_DK = 32         # DK 的 grid 级分块（主选；Stage 3 A/B 候选 64，见下）
bDV      = DV // 2    # i_v 循环分块：强制 extent=2（工具链唯一安全值，§3.5.1）
NK   = DK // block_DK # DK 分块数（grid 维，非循环）
NT   = S // 64
GRID = NK * NT * B * H        # bench: 4 × 128 × 64 = 32768 任务；20 AI Cores → 1638 任务/核
# cid 映射（bk 最快变化）：
bk = cid % NK; bsh = cid // NK; bs = bsh % NT; bbh = bsh // NT; bb = bbh // H; bh = bbh % H
```

**选型理由**：
1. **bDV = DV//2（extent=2）是工具链唯一解**（extent-1 崩溃 / extent≥3 回卷损坏★§8-2），由此 DV 支持域 = 16 的倍数且 ≥64（bDV≥32 满足 L0A/L0B K≥32 分形 + bDV×4B 32B 对齐）；
2. **bDK=32 主选**：匹配上游 bench 配置（block_DK=32）；UB 循环侧声明 184.25KB 有 7.5KB 余量（bDK=64 需 ~208KB 超 196352B 硬上限，razor-thin，见 §10.4-B）；gemm N=32 满足分形；L0C 24KB；
3. **bk 最快变化的 cid 映射**：同一 (chunk, head) 的 NK=4 个 dk-block 任务获得相邻 cid → 同一调度波次（910B3 任务按序派发到 20 核，~2us/任务量级下 4 兄弟任务启动间距 <10us）→ v/do/dv（无 DK 轴、被 4 任务重复读）的 3× 额外读全部 **L2 命中**（每 chunk-head 唯一工作集 ~320KB，波次级时间局部性，L2 192MB 富余）→ HBM 流量保持唯一 IO（§9.5）；
4. **上游 autotune 搜索空间被约束全消解**（§3.5.2-7）：Stage 3 仅剩 bDK∈{32,64}（任务数 32768↔16384、v/do/dv 重读 4×↔2×、UB 184↔208KB 临界）与 threads∈{1,2} 两维枚举。

### 5.3 约束分析

- **分形**：gemm M=64 ✓(≥16)、K=bDV∈{32,48,64} ✓(≥32)、N=bDK=32 ✓(≥16)；L0A/L0B 512B 分形由 gemm_v0 内部处理（kL0Size 默认 128 ≥ K）
- **对齐（32B）**：q/k/g 尾块 32×4=128B ✓；v/do/dv 尾块 bDV×4 ∈ {128,192,256B} ✓（bDV 为 8 倍数由 DV∈16 倍数保证）；h/dh 行 stride DV×4（32B 对齐 ⇔ DV∈8 倍数 ✓）、尾块 bDV×4 ✓；gn 末行切片 128B ✓；LastRow 256B ✓
- **L0C**：24KB ≤ 128KB ✓；**UB**：循环侧 188,672B ≤ 196,352B ✓（余量 7.5KB；k1 声明先例 ~195KB 编译通过）
- **GEMM 非整除**：声明域内无（S%64==0、DK%32==0、bDV=DV/2 构造整除）；域外 wrapper assert 拒绝（L2 负向，§1.6）

### 5.4 注意事项（尾块）

**声明域内无尾块**：S%64==0（chunked 数据模型数学要求，§1.6 三层证据）、DK 被 bDK=32 整分、DV 被 bDV=DV/2 精确二分。域外输入（S%64≠0、DK 非 32 倍数、DV<64 或非 16 倍数、chunk_size≠64、非 fp32）由 host wrapper `_check_shape` 显式 assert 拒绝（★fla_wy_fast `_check_shape` 同款）——Stage 2 场景 B 归入 L2 负向测试，**不设计 host padding + crop**（禁止，ascend-constraints §5）。

**S%64≠0 的 m_t / last(c) 行为（任务点名项）**：域外为参考实现未定义域（h/dh 无末部分 chunk 状态；Triton i_tg 越界 OOB、上游 TileLang 无掩码静默 OOB，§1.6）→ 本算子确定性 assert 拒绝。域内 m_t 恒全真（无行掩码）、last(c) 恒为局部行 63（LastRow 常量）。

### 5.5 数据搬运性能可行性（REORDER-COST）

本算子无数据重排（bshd/5D 连续直读、无 permute/gather），全部搬运为规则 tile 块 DMA：

```text
[REORDER-COST]
path: 单 kernel 主路径 (bDK=32, bDV=64, bench shape 1×8192×64×128×128)
gm_passes: 8 输入各 1 读（唯一 2560 MiB；v/do/dv 另有 NK-1=3 次 L2 命中重读 +2.25GB L2 流量）+ 4 输出各 1 写（1024 MiB）
  + V→C 中继（12 路 hi/lo 操作数/迭代 × 2 迭代，一写一读 ~9.4GB，经工具链自动注入 GM workspace，短写-读距离）
  + C→V（dq/dk frag ~1GB）
dma_transactions: ~70 次/任务（循环 2×(5 GM 载入 + 24 中继 w/r) + 尾段 4 载入 + 4 中继 + 4 写）× 32768 任务 ≈ 229 万次
average_dma_bytes: ~17GB 逻辑 / 229 万 ≈ 7.4KB/次（block 级大块 DMA；最短为 gn 末行 128B × 32768 = 4MB 总量，
  家族已验证同款 1D 末行切片）
gm_scalar_accesses: 0（无标量 GM 访问；所有元素运算在 UB）
address_div_mod_per_element: 0（cid 分解每任务 5 次整数运算）
active_cores / serial_tasks_per_core: 20 / 1638
largest_case_timeout_gate: bench shape 单 case 120s（§9.4 perf_bench，JIT 编译 32768 任务 grid + 18 gemm/任务较家族先例更重）
result: pass（无标量/strided GM 主路径；DMA 均为 ≥128B 对齐 block 块；交易笔数随任务数放大属结构固有，性能口径见 §9.5）
```

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| (dk-block, chunk, b, h) | block 级并行 | `T.Kernel(NK·NT·B·H)`（bk 最快） | 4 轴无依赖纯并行；bk 最快保 L2 局部性（§5.2-3） |
| DV 维（GEMM 收缩维） | 迭代+预取 | `T.Pipelined(DV // bDV, num_stages=2)` | **恰好 2 次迭代**（extent 约束★§8-2）；num_stages=2 双缓冲掩盖 GM→UB/中继延迟（★impl_final 全 kernel 同款）；3 组累加器跨迭代保持（init=(i_v==0)） |
| 元素级 | 向量化 | `T.tile.*` 全 tile 指令 + `T.tile.broadcast` | ★逐元素 T.Parallel / 逐行 serial 循环均为反模式（-26%~-71%/kernel） |

### 6.2 循环伪代码

见 §3.3（prologue fill → i_v Pipelined ×2 → epilogue 三段）。

### 6.3 流水线优化

`T.Pipelined(num_stages=2)` 核内预取：i_v 循环 2 次迭代恰为双缓冲覆盖（迭代 i+1 的 5 路 GM 载入与迭代 i 的 9 gemm 重叠）。核间流水线（Cube/Vector 交织）由 AUTO_CV_COMBINE 自动编排（与核内 Pipelined 互斥，源码层不显式干预）。调优空间（Stage 3）：bDK∈{32,64}、threads∈{1,2}、fixed-core + L2 常驻中继架构（§9.5 阶梯）。

### 6.4 尾块处理

见 §5.4（声明域内无尾块；域外 shape 显式拒绝）。

---

## 7. 同步策略

### 7.1 同步模式

**自动同步**（Developer 模式标准）：AUTO_SYNC 处理核内 DMA/计算依赖，AUTO_CV_SYNC 处理 Cube↔Vector 核间依赖（含 V→C 中继 / C→V 直连的自动 flag 配对）。源码无任何 `T.barrier_all` / `T.set_flag` / `T.set_cross_flag`。

### 7.2 同步点说明（编译器自动插入，设计仅声明数据依赖）

| 依赖 | 说明 |
|------|------|
| UB 拆分/乘法 ← GM 载入（h/dh/v/do/dv/g/gn/q/k/LastRow） | AUTO_SYNC |
| gemm_v0 ← L1 操作数就绪（12 路 hi/lo） | AUTO_SYNC（C 核内）+ AUTO_CV_SYNC（V→C 中继写读配对） |
| UB 尾段运算 ← L0C 结果（dq/dk_frag→UB） | AUTO_CV_SYNC（C→V 跨核） |
| GM 写 ← UB 结果（dq/dk/dg）/ L0C 结果（dw 直存） | AUTO_SYNC |
| reduce_sum ← 同核 UB 输入 | AUTO_SYNC |

### 7.3 pass_configs 配置

见 §4.7（四件套全开）。

---

## 8. CV 融合设计

### 8.1 融合算子判定

**是**（3 组 Cube GEMM（hi/lo ×9/迭代）+ Vector exp/逐元素/双维 reduce/broadcast）。

### 8.2 CV 交互设计（Developer 模式，默认消除 workspace/vid）

- `T.Kernel(GRID, threads=1, is_npu=True) as (cid)`：单轴 + threads=1（★fla_wy_fast 最终交付同款；与模板默认 threads=2 的偏离理由见 §2.3——threads=2 的 gather vid 切分越界先例 + 本算子 Stage 2 以正确性优先，threads=2 列 Stage 3 候选）
- 装饰器无 `workspace_idx`，签名无 `workspace_*` 参数、无 vid 偏移（threads=1 单返回值 cid）
- Cube↔Vector 数据交换：`T.copy(ub, l1_shared)`（V→C，12 路/迭代，经工具链自动注入的 GM 中继）+ `T.copy(fragment, ub)`（C→V，2 路）+ `T.copy(fragment, GM 切片)`（C→GM，1 路），中转与同步交给四个 pass
- **无显式 GM workspace**（无跨 kernel partial；自动注入中继的代价核算见 §9.5 逻辑流量）

### 8.3 Cube 核计算流程

见 §3.3：C 侧 = 9 gemm_v0/迭代（3 组 hi/lo 三连，init=(i_v==0) 首覆盖后累加）+ dw_frag→GM 直存 + dq/dk_frag→UB。L1 操作数由 V 侧中继供给。

### 8.4 Vector 核计算流程

见 §3.3：V 侧 = 5 路 GM→UB 载入 + dgkn 部分和（mul/reduce/add）+ 5 操作数 hi/lo 拆分（cast→sub→cast 链）+ 12 路 UB→L1 中继 + 尾段门控链（exp·ln2、broadcast、双维 reduce、LastRow 掩码、dg 组装）+ 3 路 GM 写。

### 8.5 pass_configs

见 §4.7。

### 8.6 注意事项

- 核内 Pipelined 与核间流水线互斥（pass 自动决策）
- **单 cross-core Pipelined 体**（★§8-2）：本 kernel 唯一循环体 = i_v Pipelined；尾段无循环
- **C/V 跳数控制**（关键安全约束）：完整序列 = prologue(V) → [V→C 中继 ×12/迭代]×2 迭代（循环内，安全模式★）→ epilogue C 出口（1×C→GM + 2×C→V **同向并列**，无 V→C→V 交替——规避 ★§8-4 sync-point 计数失配）→ V 尾段纯直线 → GM 写。z+da 合并失败先例（V→C→V→C→V 五跳）的触发形态不存在
- **多 gemm 组 L1 操作数隔离**（★§8-3 角色互换复用损坏 L0 staging 的防御）：3 组操作数完全独立，h 在 dq/dw 两组间用**专用 L1 副本**（h_hi/h_lo_l1 与 h2_hi/h2_lo_l1，UB 同源双拷贝，+2 中继/迭代），零共享、零角色互换
- **V↔C 交换无片上直连路径**（工具链现状，★fla_wy_fast final_summary §4）：全部经自动注入 GM workspace 中继——性能代价与 Stage 3 fixed-core 演进路径见 §9.5

---

## 9. 验证方案

### 9.1 Golden 函数（PyTorch chunk 循环实现，供 Stage 2 直接落地）

> 布局 bshd/5D 与 kernel 一致；golden 用 **fp32 计算域 + torch.exp2 精确语义**（kernel 侧 exp(g·ln2) 替代差异 ~1e-7 相对，计入混合容差验收）。NPU 环境 Triton 不可用（ref_gpu/fla_chunk_inter.py 仅作语义对照，不跑 golden——用户要求 PyTorch 重写，2026-11-02）。
> **⚠️ golden 交叉验证强制项（家族前科）**：★fla_wy_fast debug_log 记录过 golden 自身 einsum 轴序静默错（kernel 对、golden 错）。Stage 2 落地本 golden 后，必须先对一个 chunk 用**显式 per-(b,h) matmul 循环**对照 einsum 三条（dq_raw/dk_raw/dw_blk）逐位验证后再启用（验证代码不进交付）。

```python
import torch


def fla_chunk_inter_ref(q, k, v, g, h, dv, do, dh, scale=1.0, chunk_size=64):
    """fla_chunk_inter golden。全 fp32：q/k/g [B,S,H,DK]，v/dv/do [B,S,H,DV]，
    h/dh [B,NT,H,DK,DV]（NT=S//chunk_size）-> (dq, dk, dw, dg) 全 [B,S,H,DK]。"""
    B, S, H, DK = q.shape
    DV = v.shape[-1]
    C = chunk_size
    assert S % C == 0 and h.shape[1] == S // C, "S must be divisible by chunk_size"
    NT = S // C
    qf, kf, vf, gf = q.float(), k.float(), v.float(), g.float()
    do_f, dv_f, h_f, dh_f = do.float(), dv.float(), h.float(), dh.float()
    dq = torch.empty(B, S, H, DK, dtype=torch.float32)
    dk = torch.empty_like(dq)
    dw = torch.empty_like(dq)
    dg = torch.empty_like(dq)
    for i in range(NT):
        sl = slice(i * C, (i + 1) * C)
        qc, kc, gc = qf[:, sl], kf[:, sl], gf[:, sl]           # [B,C,H,DK]
        vc, doc, dvc = vf[:, sl], do_f[:, sl], dv_f[:, sl]     # [B,C,H,DV]
        hi, dhi = h_f[:, i], dh_f[:, i]                         # [B,H,DK,DV]
        gn = gf[:, (i + 1) * C - 1]                             # [B,H,DK]（域内恒为 chunk 末行）
        # 三条 GEMM（fp32；einsum 轴序须按 §9.1 警告做显式 matmul 交叉验证）
        dq_raw = torch.einsum("bthv,bhkv->bthk", doc, hi)       # do @ h^T
        dk_raw = torch.einsum("bthv,bhkv->bthk", vc, dhi)       # v  @ dh^T
        dw_blk = -torch.einsum("bthv,bhkv->bthk", dvc, hi)      # -(dv @ h^T)
        # 门控
        dq_blk = dq_raw * scale * torch.exp2(gc)                # [B,C,H,DK]
        dk_blk = dk_raw * torch.exp2(gn.unsqueeze(1) - gc)      # gn 沿 chunk 行广播
        # dgk 与 dg
        dgk = (hi * dhi).sum(-1) * torch.exp2(gn)               # [B,H,DK]
        pp = dgk + (dk_blk * kc).sum(1)                         # Σ_t dk·k（门控后 dk）
        dg_blk = qc * dq_blk - kc * dk_blk                      # [B,C,H,DK]
        dg_blk[:, -1] += pp                                     # m_last 行注入（域内恒末行）
        dq[:, sl], dk[:, sl], dw[:, sl], dg[:, sl] = dq_blk, dk_blk, dw_blk, dg_blk
    return dq, dk, dw, dg
```

输入数据构造（对齐上游 `prepare_input`，ref_gpu/fla_chunk_inter_tilelang_gpu.py:31-52）：`torch.randn` 全量 fp32（q/k/v/g/h/dv/do/dh，g 为 randn——上游即如此，h 非真实隐状态，本算子只消费不生产，randn 对拍成立）；`scale=1.0`。输出对拍在 **CPU** 完成（D2H 后调 check_precision，测试数据/golden 均在 CPU 构造后 H2D，运行段无 aclnn 依赖）。

### 9.2 L0 门槛测试计划（tilelang-op-test-design 场景 A，仅 L0）

> Stage 1 只产出 L0（规则 shape：S%64==0、DK%32==0、DV∈16 倍数且 ≥64——全部被 block 精确整除）。L1（功能，含域内其他 shape 组合如 DK=96/DV=96）/ L2（异常拒绝，含 S%64≠0、DK 非 32 倍、DV<64、chunk_size≠64、非 fp32）/ Boundary（特殊值，含 g 极值 2^g 溢出、gemm 操作数 >65504 的 hi/lo 范围限制）由 Stage 2 在 L0 通过后按场景 B 扩展。
> 算子类别判定（场景 A Phase 2，依据本 DESIGN §1.3/§1.4/§2/§4/§5/§9.3）：计算类型 = **混合（CV 融合：3 组 fp16 hi/lo GEMM + Vector exp2 替代链/逐元素/双维 reduce）**；复杂度 = **Fusion**（3 类 GEMM + 门控 + 行注入 + 4 输出）；数学特征 = matmul + element-wise gating + reduction；综合类别 = **Fusion（gated chunked attention 反传 inter-chunk 梯度）**→ 测试策略聚焦：CV 协作正确性（V→C 中继 / C→V 尾段）、chunk 语义（末行注入 / 跨迭代累加 / gn 末行）、hi/lo 拆分精度、4 输出独立判定；dtype 无变体空间（全 fp32 固定组合）。

| 用例名 | 级别 | Shape (B,S,H,DK,DV) | dtype | 说明 |
|--------|------|---------------------|-------|------|
| l0_min | L0 | (1, 128, 2, 64, 64) | 全 fp32 | 最小冒烟：NT=2（双 chunk 末行注入 ×2）、BH=2、NK=2（DK=64）、bDV=32（**K=32 分形边界** gemm）、grid=8 任务。覆盖全链路：5 路载入、hi/lo 拆分链、9 gemm/迭代（init=(i_v==0) 跨迭代累加）、dgkn clear=True+add 累加、gn 末行 1D 切片、exp·ln2 门控链、LastRow 掩码、dw 取负折叠直存、frag→UB C→V、4 输出 |
| l0_repr | L0 | (2, 256, 4, 128, 128) | 全 fp32 | benchmark 结构代表：bDK=32（NK=4，bk 分解 + L2 局部性映射）、bDV=64（K=64 gemm）、B>1 触发 bb/bh 分解、grid=128 任务。全量程 fp32 + hi/lo 精度路径 |

**golden 对照法**：每用例对 4 个输出独立调用 `check_precision(out, ref, "float32")`（precision-standard.md §4.1 混合容差判定，Stage 2 原样复制不改动）；全部通过打 `[PRECISION_PASS]`，任一失败 `[PRECISION_FAIL]`（阻塞，计入退出码）。

### 9.3 精度标准（混合容差，precision-standard.md 口径）

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|-------|------|------|---------------------|------------------------|
| float32（dq/dk/dw/dg 全部输出；输入全集亦仅 fp32） | 2⁻¹⁶ (1.53e-5) | 2⁻¹⁰ (9.77e-4) | 1e-2 | 0.99 |

> 与 proto_fla_chunk_inter.yaml 声明的门禁逐字一致（inputs dtype 全集 = {float32}，与上行一致，无第二个 dtype 行）。特殊值（NaN/Inf）位置契约：输入域为 randn 有限值（上游对拍口径）；g 极值（g>128 时 2^g=inf，出现在 V 侧 fp32 门控域，kernel 与 golden 同为 inf → 结构比对一致）与 gemm 操作数 >65504（fp16 hi/lo 范围限制）属 Boundary 范畴，Stage 2 场景 B 扩展（"有限值 + 稀疏特殊值"混合输入 + mask 结构比对，precision-standard §3.1）。
>
> 误差预算【推断】：kernel 相对 golden 的差异 = fp16 hi/lo 拆分（~2⁻²² 相对/操作数对，DV=128 累加后绝对误差 ~3e-6）+ exp2→exp(g·ln2)（~1e-7 相对）+ gemm L0C fp32 累加（精确）。输出量级 ~11（randn、DV=128 累加），逐元素容差 ≈ atol + rtol·|out| ≈ 0.011，典型误差 ~3e-6——余量 ~3000×，matched_ratio 预期 ≈ 1.0。对照被否的 hi-only 单 cast 方案（误差 ~5.5e-3，小 \|out\| 元素 atol 主导区失败，估算 ratio ~90-93% < 0.99，且家族两例实证 0.988 / 超差 FAIL，§10.4-C）。

### 9.4 性能可行性哨兵（强制，不可跳过）

| 用例名 | Shape (B,S,H,DK,DV) | dtype/属性 | 覆盖路径 | 单 case timeout | 选择理由 |
|--------|---------------------|------------|----------|-----------------|----------|
| perf_bench | (1, 8192, 64, 128, 128) | 全 fp32, scale=1.0, chunk=64 | 单 kernel 主路径（32768 任务 = 最大任务数、最大 DMA 吞吐压力、bDK=32/bDV=64 满配、L2 局部性映射满载） | 120s | benchmark 验收 shape（proto 性能目标口径） |

计时口径：对齐上游 `ref_gpu/test_utils_kda.py do_bench`——**warmup=20、rep=10、取均值、仅设备侧 kernel 时间**（NPU 事件 `torch.npu.Event(enable_timing=True)` 等价实现），不含张量构造/H2D-D2H/JIT/框架开销。wrapper 内**无** `torch.npu.synchronize()`（★fla_wy_fast 优化项 #10：内部 sync 把 host JIT 分发延迟暴露进设备侧计时，与口径相悖）。测试数据、随机数与 golden 均在 CPU 完成后 H2D，运行段无 aclnn 依赖。

### 9.5 性能预算分析（910B3，含 3400us 目标可行性结论）

**IO 明细（物理唯一字节，每张量单读单写；bench shape 1×8192×64×128×128）**

| 项 | 字节 |
|----|------|
| 读 | q/k/g 各 256 + v/do/dv 各 256 + h/dh 各 512 = **2560 MiB** |
| 写 | dq/dk/dw/dg 各 256 = **1024 MiB** |
| 合计（唯一 HBM） | **3584 MiB ≈ 3.758 GB**（该数据布局下的 HBM 下限，无 kernel 层压缩空间） |
| GEMM FLOPs | 51.5 GFLOP 逻辑 fp32 × 3（hi/lo）= **154.6 GFLOP 物理 fp16**（@157 TFLOPS 折算 ~985us，与 IO 重叠，非独立下限） |
| Vector | ~67M exp + 逐元素 ~0.5 GFLOP（tile 指令）——非瓶颈 |

**逻辑流量分解（本工具链 V↔C 无片上直连，全部经 GM workspace 中继，★fla_wy_fast final_summary §4 实证）**

| 项 | 字节 | 说明 |
|----|------|------|
| 唯一 HBM | 3.76 GB | 下限 |
| V→C 中继（一写一读） | ~9.4 GB | 12 路 hi/lo 操作数/迭代 × 2 迭代 × 32768 任务（fp16 hi+lo 字节数 = fp32 原字节数，hi/lo 不增加中继量，§10.4-D） |
| v/do/dv NK=4 重读 | +2.25 GB | bk 最快 cid → L2 命中（§5.2-3），不计 HBM |
| C→V（dq/dk frag） | ~1.0 GB | 短距离写-读，L2 命中为主 |
| **逻辑总量** | **≈ 16-17 GB** | 唯一 + 中继 + 重读 |

**三模型对照（设计期不可判定真值，perf_bench 哨兵实测）**

| 模型 | 估算 | 依据 |
|------|------|------|
| ① HBM 唯一 IO 下限 | 3.758 GB ÷ 1263 GB/s = **2976 us** | 实测带宽上限（★fla_wy_fast D2D） |
| ② 家族逻辑吞吐标定 | ~17 GB ÷ 1030 GB/s（★k1 实测逻辑带宽）≈ **16 ms 量级（悲观界）** | k1 为同构 kernel（Pipelined + hi/lo + 中继）；但 k1 的 4096 块中继槽位 ~128MB 可 L2 常驻，本算子 32768 块 × ~144KB ≈ 4.5GB 槽位足迹超 L2 192MB——短生命周期写-读距离的 L2 命中率是关键不确定项 |
| ③ 理想中继全吸收 | HBM 3.76 GB 独立饱和 → **3.0-3.7 ms（乐观界）** | 需同时满足：87.5% HBM 效率 + 中继/重读全 L2 吸收 + 每任务同步开销全隐藏 |

**结论：3400 us 目标【高风险 / at-risk】，需 perf_bench 实测判定**：

1. 最乐观模型（③）的 2976-3700 us 恰好覆盖目标区间，但要求 **87.5% HBM 带宽效率——已超家族实证最佳 81%**（★k1/k2 逻辑口径；★fla_chunk_o 实际 47%、★fla_wy_fast bwd 16.9%）；且需中继全部 L2 吸收、32768 任务的每任务同步开销（★fla_chunk_o 实证 ~1.4-1.8us/任务纯同步 + 本算子 ~125 op/任务，k1 标定 ~0.09-0.15us/op）全部隐藏——三重理想条件同时成立的概率低；
2. 与 ★fla_wy_fast 的"物理不可达"（需 1.44× 实测上限）本质不同：本算子 HBM 下限 2976us **< 3400us 目标，物理可达、工程临界**——与 ★fla_chunk_o（需求 60.3% vs 实证 81%，最终以 107.5% 交付）同类但更紧；
3. **Stage 3 调优阶梯**（按预期收益排序）：① **fixed-core grid + 显式 L2 常驻中继 workspace**（★fla_chunk_o R4 配方：`T.Kernel(20, threads=2)` + 核内任务循环 + 20 槽中继 ~2.9MB 常驻 L2，替代 ~4.5GB 自动注入槽位足迹，实证 -25% @8192 任务规模，本算子 32768 任务预期收益更大；含手动 cross-core flag 破锁步——★其 final_summary §2 优化项 3/4/5/6 完整配方）；② bDK=64 A/B（任务数减半 16384、v/do/dv 重读 4×→2× 即 −1.1GB 逻辑流量、UB ~192.5KB 临界需削峰验证）；③ threads=2（唤醒 40 AIV，★fla_chunk_o 交付形态）；④ op 数削减（拆分链 cast 融合探索）；
4. 若 Stage 3 后仍未达标：按家族政策**如实披露**（★fla_chunk_o 1.417ms vs 1320us 超 7.5% 交付先例）——报告带宽效率口径（唯一 IO ÷ 时间 ÷ 1263GB/s）+ 地板核算，不做任何破坏精度/正确性的达标动作。

### 9.6 Host 侧 Buffer 操作审计

```text
[HOST-METADATA-AUDIT]
operation: 输入构造 torch.randn(...).npu()（CPU 生成后 H2D，数据准备）
  + LastRow 常量构造（CPU zeros + [63]=1.0 → npu()，一次性常量输入，per-device 缓存）
  + kernel 调用（13 张量直传：8 输入 + LastRow + 4 输出 out_idx 框架分配）
  + 精度验证（D2H 后 CPU 侧对拍 check_precision）
input_stride -> output_stride: N/A（新建张量，非变换；kernel 直读 bshd/5D 连续布局，无任何 view/reshape/permute）
shares_storage / same_data_ptr: N/A（无变换操作）
aclnn_or_physical_copy: false（randn/常量为 CPU 数据生成 + 一次性 H2D；kernel 调用前后无
  .contiguous()/pad/cat/reshape 物理化/.to(dtype)/.clone()；wrapper 内无 torch.npu.synchronize()）
result: allow
```

---

## 10. 风险点与注意事项

### 10.1 风险清单（高/中/低分级）

| 级别 | 风险 | 影响 | 缓解/验证 |
|------|------|------|-----------|
| **高** | R1 单 Pipelined 体 3 gemm 组（9 gemm_v0/迭代、3 L0C 累加器）无逐字硬件先例（k1 实证 1 组 3 gemm 进 1 累加器；家族失败案例为"角色互换共享 L1 操作数"，本设计零共享） | 编译错 / L0 损坏（§8-3 类） | 专用 h2 L1 副本隔离（+2 中继/迭代，廉价保险）；l0_min 首验；**回退 1**：去 h2 副本（同角色共享 h）；**回退 2**：dw 组拆独立 kernel（§10.4-E）；同族 DESIGN_fla_chunk_intra 多组规划为家族包络 |
| **高** | R1' epilogue C 侧混合出口（dw frag→GM + dq/dk frag→UB 同向并列）无逐字先例（k1 = 仅 C→GM；z/da = 仅 C→V） | 编译错（sync-point）或数据错位 | 无 V→C→V 交替（★§8-4 触发形态不存在）；l0_min 首验（dw 是 h⊙dv 的直接函数，错位必然暴露）；**回退**：dw 改走 frag→UB（+8KB UB 尾段 + 2 op） |
| **高** | R2 性能目标 3400us 临界/高风险（最乐观模型已需 87.5% HBM 效率 > 实证最佳 81%；逻辑流量 ~17GB 中中继占 9.4GB） | 性能验收失败 | §9.5 三模型论证 + Stage 3 阶梯（fixed-core + L2 常驻中继为最大杠杆）+ perf_bench 哨兵实测 + 家族如实披露政策 |
| 中 | R3 UB 循环侧声明 184.25KB（96% 硬上限 196352B） | 编译溢出 | in-place hdh 复用 + 交错拆分顺序已内建（余量 7.5KB；k1 先例 ~195KB 编译通过）；回退阶梯见 §4.5 |
| 中 | R4 grid 32768 任务（家族最大 8192 的 4×）+ 每任务 ~125 op（k1 标定 ~0.09-0.15us/op 同步开销） | 任务派发/同步开销放大 | threads=1 无显式 workspace；bk 最快保 L2；perf_bench 实测；回退 bDK=64（16384）；Stage 3 fixed-core |
| 中 | R5 golden einsum 轴序（家族前科：golden 静默错、kernel 正确） | 假 FAIL / 假 PASS | §9.1 强制交叉验证（显式 per-(b,h) matmul 对照单 chunk 逐位）后才启用 golden |
| 低 | R6 exp(x·ln2) vs exp2 精度（~1e-7 相对） | 精度微小 | 远小于门禁 2⁻¹⁰；golden 用 torch.exp2 对拍验证 |
| 低 | R7 S%64≠0 语义 | 域外误用 | 已论证为参考实现未定义域（§1.6 三层证据），wrapper assert 确定性拒绝 |
| 低 | R8 reduce_sum dim=0/dim=1（fp32） | 数值/形状 | examples 先例充分（HISA/mhc dim=0；★v2 dim=1）；in-loop clear=True+add 为家族验证修复模式 |

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| fp32 gemm_v0 直用 | 照抄上游 fp32 T.gemm | 1e37 级垃圾（工具链损坏） | fp16 hi/lo 3-gemm 拆分（本设计已内建） |
| `T.gemm` 通用版当昇腾 API | 照抄上游 | intrinsic 不支持 | 全部 `T.gemm_v0` |
| `T.exp2` 直用 | 照抄 Triton 语义 | 无后端映射 | `T.tile.exp(x·LN2)` |
| `T.if_then_else` 写进 T.Parallel | 照抄上游 m_last 行选择 | 编译错误（SIMD 无分支） | LastRow 常量 + broadcast + tile.mul |
| threads=128 / 三维 Kernel | 照抄上游 | 编译报错 | threads=1 + 一维化（bk 最快） |
| out_idx 负索引 | 照抄上游 [-4..-1] | 输出错位 | 绝对索引 [9,10,11,12] |
| in-loop reduce_sum(clear=False) | 累加 dgkn | 数值偏移（工具链 bug） | clear=True + tile.add |
| `T.reduce_sum` 传 fragment | 照抄上游 shared→fragment | 前端报错 | UB→UB reduce |
| 跨 dtype GM↔UB 直拷 | fp32 GM 直接 cast 载入 | codegen 模板错配 | GM 同 dtype → UB → UB 内两步 cast |
| host 侧 g 预处理（exp2/aclnn） | 优化幻觉 | aclnn 违规 + IO 翻倍 | 禁止；门控全在 kernel 内 fp32 域 |
| wrapper 内部 `torch.npu.synchronize()` | 计时"保险" | 分发延迟暴露进设备侧计时 | 移除（上游 do_bench 口径） |

### 10.3 特殊场景处理

- **S%64≠0（m_t / last(c)）**：域外，参考实现未定义（§1.6）→ wrapper assert 拒绝，L2 负向；域内无掩码、last(c) 恒局部行 63；
- **极小 shape**（S=64, NT=1）：grid=NK·B·H，NT=1 时 m_last 注入只在唯一 chunk 的末行——合法；L0 不含（用 S=128 保证 NT≥2，触发多 chunk 的末行注入差异）；
- **非 2 幂 D**：DK=96（NK=3）✓ 域内；DV=96（bDV=48，K=48 分形 ✓ 对齐 192B ✓）✓ 域内——均留 L1 扩展（Stage 2 场景 B）；
- **g 极值上溢**（g>128 → 2^g=inf）：V 侧 fp32 门控域，kernel 与 golden 同为 inf → Boundary 结构比对（Stage 2）；gemm 操作数不受影响（g 不进 gemm 操作数——本算子与 fla_wy_fast 的结构性差异，R2 精度面更安全）；
- **操作数 >65504**（fp16 hi/lo 范围）：Boundary 文档化（★fla_wy_fast 同款已知限制）；训练域/上游 randn 对拍域不触发。

### 10.4 被否方案（设计决策记录）

| # | 方案 | 否决理由 |
|---|------|---------|
| A | DV 维上 grid（任务 = dv-block，i_k 循环内累加） | 输出沿 DV 不因子分解（dq_raw = Σ_DV 收缩）→ 跨任务部分和 + GM partial 二次归约（+1GiB IO + 额外 kernel）+ g/q/k 重复读。结构性劣化，否决 |
| B | bDK=64/128 主选（任务=更大 DK 块） | bDK=128：UB 声明 ~352KB 远超 196352B 硬上限（h/dh fp32+f16 暂存随 bDK 线性放大）→ 死刑；bDK=64：声明 ~208KB 超 16KB → 不作主选，列为 Stage 3 A/B 候选（§9.5 阶梯②，任务数减半 + 重读减半的收益明确，UB 需削峰验证） |
| C | hi-only 单 f16 cast（1× gemm，9→3 gemm/迭代） | 误差预算：f16 单次相对误差 ~2⁻¹¹/操作数，DV=128 累加后误差 std ~5.5e-3，小 \|out\| 元素（atol 主导区，预算 →1.53e-5）系统性失败，估算 matched_ratio ~90-93% < 0.99；家族两例实证（★fla_chunk_o ratio 0.988 FAIL、★fla_wy_fast bwd 拆分前超差）。精度否决——hi/lo 3-gemm 强制 |
| D | host 预拆分 f16 hi/lo 双 kernel（先 Vector cast 写 GM，再 Cube kernel 直读 L1） | +4.1GB IO（2.05 读 + 2.05 写）且不减少 gemm 侧中继（f16 hi+lo 字节 = fp32 原字节）→ 纯增开销，否决；本设计 hi/lo 中继量与 fp32 直中继量相等（字节不增） |
| E | 3-kernel 拆分（dq/dk/dw 组分 kernel + dg 尾段 kernel） | dg 依赖 dq 与 dk 的双门控结果 → 需 GM partial 中转（dq_raw/dk_raw 写读 +1GiB）+ 3× JIT launch；仅作 R1 失败回退（正确性优先时可用），不作主路径 |
| F | host 侧预计算 2^g / 常量折叠 scale·2^g | aclnn 违规（ascend-constraints §5）+ g 256MiB 读翻倍；scale·2^g ≠ 2^(g+log2 scale) 精确性（scale 非 2 幂时）——kernel 内 fp32 域计算，否决 |

---

## 11. 交付清单

### 11.1 目录结构（沙箱布局，tilelang-op-migration 共享目录）

```
examples_experiment/dsa_sparse_finetune/     # 共享沙箱（与 fla_chunk_intra / indexer_bwd 等共存，互不触碰）
├── DESIGN_fla_chunk_inter.md     # 本设计文档（含 L0 门槛测试计划）✅
├── proto_fla_chunk_inter.yaml    # 算子接口规格（任务书，Orchestrator 预写；本设计产出后一致性核对 ✅ 无需改动）
├── ref_gpu/                      # 冻结参考（只读）：fla_chunk_inter.py / fla_chunk_inter_tilelang_gpu.py / test_utils_kda.py
├── fla_chunk_inter.py            # ⬜ 待实现（Stage 2）：单 kernel（@tilelang.jit，§3.3 结构）
└── test_fla_chunk_inter.py       # ⬜ 待实现（Stage 2）：golden（§9.1）+ L0（§9.2）+ perf_bench（§9.4）+ --level main
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN_fla_chunk_inter.md` | ✅ 已完成 | 本文档（11 章节 + L0 门槛测试计划 §9.2） |
| `proto_fla_chunk_inter.yaml` | ✅ 已核对一致 | 承担 proto.yaml 角色：`operator.inputs[].dtype` 全集 = {float32}（与 §9.3 精度表唯一 fp32 行一致）、attrs 覆盖 chunk_size/scale、schema 完整（13 参数含 LastRow 说明见 §3.3 注）、性能门禁 <3400us@bench shape——经逐项核对满足覆盖门禁 `coverage_check.py --proto` 派生要求，**无需改动**（本次全 fp32，与任务书预期一致） |
| `fla_chunk_inter.py` | ⬜ 待实现 | `fla_chunk_inter`（JIT 单 kernel，Developer 模式，§4.7 pass_configs，out_idx=[9,10,11,12]）+ `_check_shape` assert + LastRow 常量构造/缓存 |
| `test_fla_chunk_inter.py` | ⬜ 待实现 | golden（§9.1 逐行落地 + 强制交叉验证）+ `check_precision`（precision-standard.md §4.1 原样复制）+ L0 两用例（§9.2）+ perf_bench 哨兵（§9.4）+ `--level` main 分发器 |

### 11.3 命名规范

- kernel 文件 `fla_chunk_inter.py`、测试 `test_fla_chunk_inter.py`（沙箱目录内算子前缀，符合迁移任务硬性约束）
- JIT kernel 函数名 `fla_chunk_inter`（`from fla_chunk_inter import fla_chunk_inter`）；host 语义名 `fla_chunk_inter_bwd`（proto schema 口径）作为测试侧别名

### 11.4 实现顺序

1. ✅ 设计文档 + L0 门槛测试计划 + proto 一致性核对（本文档）
2. ⬜ `fla_chunk_inter.py`：kernel（§3.3 伪代码逐段落地；先 L0 编译过，R1/R1' 首验 l0_min）
3. ⬜ `test_fla_chunk_inter.py`：golden（含 §9.1 交叉验证）+ L0 + main
4. ⬜ L0 门槛测试通过（精度收敛；check_precision 4 输出全绿）
5. ⬜ perf_bench 哨兵（§9.5 三模型实测判定；未达标触发 Stage 3 阶梯）
6. ⬜ 扩展分层套件 L1/L2/Boundary（tilelang-op-test-design 场景 B；含 S%64≠0 负向、DK=96/DV=96、g 极值、>65504 操作数）
7. ⬜ 全量套件运行（L0/L1 须通过；L2/Boundary 失败仅记录不阻塞）+ 覆盖门禁 `coverage_check.py --proto proto_fla_chunk_inter.yaml`

### 11.5 算子 proto.yaml 一致性说明（覆盖门禁用）

> proto.yaml 角色由 `proto_fla_chunk_inter.yaml`（Orchestrator 预写任务书）承担。本设计完成后逐项核对：**dtype 全集**（§9.3 精度表唯一 float32 行 = proto `inputs[].dtype` 全部 [float32] ✓）；**attrs**（chunk_size/scale = proto `attrs[].name` ✓，均影响计算路径：chunk_size 定 BT、scale 进 dq 门控）；**schema**（13 参数 = 8 输入 + LastRow 设计新增常量（proto 未列，属设计新增 kernel 输入而非算子接口参数，wrapper 内部构造）+ 4 输出 ✓）。核对结论：**proto 无需改动**（与任务书"本次全 fp32，预期无需改动"一致）。

---

## 12. Stage 3 R2 结构级调整增补（2026-11-02，Developer 模式续挖）

> R2 用户检查点决策：不做 Expert 重写，继续 Developer 模式内挖掘。本节同步 R2 对实现结构的
> 变更（细节与逐轮 A/B 数据见 `perf_tuning/fla_chunk_inter_perf_log.md` R2 段与
> `perf_tuning/fla_chunk_inter_final_summary.md`）。性能：19556（Stage 2）→ 12028（R1）→
> **6767 us do_bench / 6534 us msprof（R2 终态）**；目标 3400us 未达（根因与封堵清单见 §12.4）。

### 12.1 Tiling 与任务分解变更（覆盖 §5.2 的 GRID 定义）

- **任务 = (B, chunk, head)**（GRID = NT·B·H，bench 8192），NK = DK/block_DK 个 dk-block 在任务内
  **trace-time 全展开**（r2_iter1 "NK-merge"）：v/do/dv 无 DK 轴，合并消除了原 bk 兄弟任务对同一
  [BS,DV] 拼块的重复装载（bench 下 -786MB V-mte2）与左操作数拆分链/中继的重复生产；C 侧 L1 的
  A 操作数跨 bk gemm 组复用。V-mte2 字节 480→384.5KB/(chunk,head) 对，实测引擎带宽
  47→53（mte2）/67→87（mte3）GB/s（合并后 DMA 突发更大）。
- block_DK 形状自适应不变（64 if DK%64==0 and DV>=128 else 32，reduce_2d [64,64] 损坏 guard 保留）。

### 12.2 中继 workspace 变更（补充 §4.3/§7）

- 右操作数中继（h/dh hi-lo，workspace_3/4/7/8）与结果中继（dq_relay/dk_relay）改 **扁平
  (task, bk) 槽位**：形状 (CORE_EFF, NSLOT·NK, ...)、索引 [cid, slot·NK+bk, ...]——bk 间区域
  不相交，消除同槽覆写 FIFO 耦合；左操作数中继（workspace_1/2/5/6/9/10）保持 (CORE_EFF, NSLOT,
  BS, DV) 单槽（每任务单产）。
- dgkn 跨排水携带环改 **[NSLOT·NK, block_DK]**（行 slot·NK+bk）——保持 2-D buffer + 满宽行切片
  （reduce_2d 输出实证干净形态）。
- workspace 足迹（bench）：~23.6MB，L2 192MB 常驻（未变）。

### 12.3 同步策略变更（覆盖 §7.2 的 dq/dk 结果中继部分）

- **1-late flag 配对 + head-set 预充**（r2_iter2）：drain(j) 实读 task j-1 的 dq/dk 中继，wait
  配对 C(j-1) 的 set（wait 40+(j+1)%2 ↔ set 40+(j-1)%2）；j=0 由 C 侧循环前一次性 set flag 41
  预充（计数语义平衡：head-set ↔ drain(0)）。R1 的 0-late 同迭代 flag 在 V busy 下降后暴露为
  ~3.1ms 串行等待（V 等 C 整条消费链），1-late 配对后 V 侧 flag 等待实测归零。drain lag 保持 1。
- AUTO_CV_SYNC 按名配对 / 手动 flag 逃逸 FIFO 的机制均未变。

### 12.4 R2 封堵结论（供后续设计引用，详见 final_summary §3.2）

threads=2（40 AIV 唯一入口）源码级闭合：ascend_vid_reduction.cc 将全部 UB buffer 第 0 维减半并在
GM 访问 (gm_dims−ub_dims) 轴加 vid 偏移——produce 侧 [bDK,DV] 沿 DK 切半且中继 slot 轴错位、
drain 侧 1-D [bDK] 沿 DK vs 2-D [BS,bDK] 沿 S 轴不一致；grid=40 受 KERNEL_TYPE_MIX_AIC_1_1
launch 形态限制（1 AIC+1 AIV/block，最多 20 并发组）；pass_configs 无流水深度类旋钮；
produce/drain 语句重排家族三战役六例实证高风险（planner 别名 WAR 崩溃或屏障劣化）。

### 12.5 门控链算术等价变形（r2_iter3）

ln2 预折：g/gn 原位 ×ln2 后 gnd = gn·ln2 − g·ln2（原 (gn−g)·ln2 链省一个 [BS,bDK] mul）；
ml2d 广播提升至 drain 头（bk 不变量）。舍入差异 ~2e-7 相对（≪ rtol 9.77e-4），±200 边界
溢出/下溢路径逐位一致；valrange_l dg 8.301e-3 与 R1 逐位相同。

---

## 13. Stage 3 R3 结构级结论增补（2026-11-03，Expert 授权 + 3400us 原指标续挖）

> R3 用户指令：按原指标 <3400us 继续调优；Expert 路线（显式 alloc_L1/ub/L0A/L0B/L0C + 手动
> T.Scope + 手动 flag）授权使用；若仍未达标，最终报告必须逐项列出限制（描述 + 证据 + 突破条件 +
> 地板值）。R3 终态：**kernel 与 R2 终态逐位一致（md5 db1d9e8a，无净结构变更）**；两实现候选失败
> 回滚，五个冲刺方向全部证据闭合。性能 6767us（R2 档案）/ 6639-6720us（R3 会话带）vs 目标 3400us。
> 完整限制清单见 `perf_tuning/fla_chunk_inter_final_summary.md` §3；逐轮数据见
> `perf_tuning/fla_chunk_inter_perf_log.md` R3 段。

### 13.1 threads=2（40 AIV 唯一入口）——实验闭合，覆盖 §12.4 源码结论

微形状 kernel 实验（`perf_tuning/fla_chunk_inter_vid2_micro.py`，[64,64] bench 精确算子形态，
threads=1 对照组全 PASS、threads=2 实验组）：

| 子链 | t2 结果 | 机理（ascend_vid_reduction.cc） |
|---|---|---|
| 纯 2-D 行切分算子（dq/dk 路径） | PASS | dim-0 减半 + 3D 视图 S 轴 vid 偏移，一致 |
| [BS]→[BS,bDK] broadcast（m_last，"形状一致"） | **FAIL**（末行） | tl.ascend_broadcast 不在 tile_ops 折半集（L563-586），显式 shape 参数不折半 → dst 越界 |
| [bDK] 1-D 链 + broadcast(axis=0)（gn） | **FAIL**（ratio 0.25） | 1-D 沿 DK vs 2-D 沿 BS 轴失配 + broadcast 不感知 |
| dim=0 reduce（pp 的 Σ_t dk·k） | **FAIL**（末行） | per-vid 部分和，无跨 vid 合并（Analyze 仅定 workspace 除数） |
| produce 侧 [bDK,DV] DK 切分 + dim=1 reduce | PASS | produce 形态可行——但 threads 为 kernel 级参数 |

**结论**：本算子 drain 数学（逐列不变量 + 跨行归约 + 行掩码广播）在 vid 语义下结构性不可表达。
突破需上游三项修复（broadcast vid 感知 / 跨 vid reduce 合并 / per-buffer 切分轴标注）；届时
V-mte2 地板 2867→~1434us/20AIV、V 链地板 ~3.5ms 级（仍需 C 链同步压缩方可达标）。

### 13.2 Expert V 侧三引擎手动流水——假设证伪 + 地板修正

vchain 微实验（`perf_tuning/fla_chunk_inter_vchain_micro.py`，drain 形态链 ×400 任务）：
AUTO_SYNC 逐 op 屏障 1.66us/任务 vs 手动突发协议（T.Scope("V") + 3 对 flag + head-set 预充，
精度双 PASS）1.61us/任务——**仅 -3.1%**。假设"逐 op 屏障串行化发射流"证伪：小工作集链在
AUTO_SYNC 下已达 mte2 引擎界。真因：**UB 196KB 容量强制 produce 逐操作数链式**（224KB 装载集
无法突发）+ produce/drain 交叉别名屏障。Expert 同步重写的可回收空间仅在 drain 内部重叠
（~3.5us/任务）与跨相位重叠（produce ~130KB + drain ~64KB ≈ 194KB 贴 196KB 上限，razor-thin），
修正后 Expert-V 地板 ~9-12us/任务 ≈ 3.7-5.0ms。

### 13.3 R3 实现候选失败记录（工具链新证据）

- **T.tile.mul_add_dst 禁用**（r3_iter2）：两处 mul+add 融合致 l0 dg 0.9852/0.9857 确定性偏差
  （~1% 量级，末行注入路径；分离二分确认非规划器布局扰动）。根因：CANN 9.2.0
  `kernel_operator_vec_binary_intf.h:249-250` MulAddDst 带 `__ASC_USE_RESERVED_UBUF__(3510)`
  标注（使用保留 UBUF 暂存的受限指令），与规划器管理的 UB 共存冲突。
- **drain in-place 链（broadcast→sub→exp 原位）崩溃**（r3_iter3）：ACL 507014 设备挂起——
  sync-insert 对 in-place 序列处理缺陷（R1 iter6a 家族签名再现）。同时清理了一个 R2 世代残留
  探针进程（/tmp/opencode/probe/r2_hang_locate.py 占卡）。

### 13.4 对 §12.4 封堵清单的增量

- launch 形态（方向④）：codegen_ascend.cc:858-860 仅存在 MIX_AIC_1_1 / MIX_AIC_1_2 两种
  KERNEL_TASK_TYPE（cv_ratio 决定）；无 AIV-only 或稀疏 C 配比形态——grid=40 恒为两波串行 20。
- bDK=128（方向⑤）：重核 UB 预算——do/v/dvn fp32 96KB + h/dh fp32 128KB = 224KB > 196352B
  （未计 hi/lo 暂存），死刑复核成立。
- C 链压缩（方向③）：C busy 4481us/任务 10.9us = mte2 中继读 1208 + mte1 staging 925 + scalar
  地址计算 983 + cube 703 + fixpipe 662——全部由 V↔C GM 中继架构强制（无片上直连路径），
  DSL 层无旋钮；波次化（HISA 式手动 C 流水）可压向 max-engine ~2.9us/任务但属全量 Expert
  重写范畴（L1 2 深槽 448KB/512KB razor + 手动 flag 协议，家族 3/4 失败区）。

### 13.5 R3 终局判定

3400us 低于本工具链/硬件代际上**一切可实现配置**的复合地板（最乐观全 Expert 手写包 both-V-and-C
理论 ~2.9-3.4ms 需 razor-edge UB/L1 装配 + 手动协议全部成立；现实 Expert-V 地板 3.7-5.0ms）。
绝对物理界：HBM 唯一 IO 2975us@100%/3673us@81%。逐项限制清单（含证据文件:行 + 突破条件 + 各路径
地板值）见 final_summary §3——上游修复项（threads=2 vid 语义、planner 别名感知同步、跨 dtype
DMA、C 中继架构）与架构物理上限分别列示。

## 14. 交付修复：scratch-free 向量算子形式化（2026-11-03，D-2）

### 14.1 背景与根因修正

交付期（D-1）发现 kernel 对 memory planner 注入 scratch（reduce_sum /
tile.broadcast 共享 tmp_ub）的 UB 打包次序敏感：打包翻转（scratch 低于 dst）
即确定性 aivector Illegal instruction（vec error 0x700000042）。D-2 修正了根因
链中的一个关键混杂变量：**沙箱与 worktree 的"同字节文件"对比实际跑的是两个
不同工具链构建**（沙箱 source set_env.sh 用 repo 源码版；worktree 裸跑解析到
pip 安装的 0.1.4 旧版）。pip 旧版对该 kernel 族基础结构（与 scratch 无关）即
崩溃（bisect 实证），且存在独立缺陷：**无锚的 loop-外 set_cross_flag 被静默
丢弃**（→ 首轮 drain wait 死锁）。planner 指针序非确定性（D-1 已录）依然成立，
本节修复以"结构性消除全部注入 scratch op"为纲，使布局翻转无害化。

### 14.2 五个 scratch op 位点的替换形式（全部实证）

| 位点 | 旧形式（注入 scratch） | 新形式（0 scratch） | 实证 |
|------|----------------------|--------------------|------|
| produce dgkn（dim=1 行和） | reduce_sum(dh_ub, ring_row, dim=1) | 原位乘积 → 1 次硬件 transpose 至 [DV,bDK] → Var-min 整行 chunk-add（DV%64 尾块 trace-time 字面区域）→ 6 层字面量折半树 → 行拷贝入 ring | P7：8 组 (bDK,DV) 全 PASS |
| drain pp（dim=0 列和） | reduce_sum(dkk_ub, pp_ub, dim=0) | 6 层字面量折半树（dkk 先被 dg=q*dq−k*dk 消费再折叠） | P1/P6r PASS |
| ml2d broadcast + mp2d 掩码乘 | broadcast(axis=1) + mul | **删除**：末行单行加 dg[BS-1] += pp；LastRow 输入与 5 个 UB buffer 删除，签名 25→24 | P6r PASS；有限 pp 逐位等价，溢出时更贴 golden（bd_g_gn_overflow dg WARN→PASS） |
| gn2d broadcast | broadcast(gn2d, gn, axis=0) + sub | **T.unroll 行语句链** gnd[t,:]=gn[:]−g[t,:]（64 逐行 Sub）+ exp/mul 不变 | P4 逐位精确 0.000e+00 |
| pp2d broadcast | broadcast(pp2d, pp, axis=0) + mul | **删除**（并入末行单行加） | P6r PASS |

关键工程约束（实证）：
- 折半树必须折叠**外层维**（行）：第二操作数基址 = 行起点（bDK 对齐），
  无 32B 偏移约束；沿连续维折叠 h<8 层必失配（0x700000042）。
- TIR `while` 与 Python 列表下标在 trace-time 不可用（while 常量条件折叠成
  死循环）；所有 shape 相关调度用**字面量语句**（BS=64 域常量）或 **TIR range
  整行区域 + 常量 extent**（chunk-add），DV%64 尾块用 trace-time `if`。
- fp32 行 reduce 在向量选择层无条件路由到 reduce_2d_v2（必带 scratch），
  narrow/WholeReduce 路径仅非 fp32/非行方向可达——故 dim=1 和唯有改折叠。
- pip 旧版丢弃无锚 loop-外 set_cross_flag：prologue 以 live 填充锚
  `T.tile.fill(dgkn_ring, 0.0)` 固定 head-set（j=0 垃圾 drain 读填充值→非死码）。

### 14.3 验收与性能

- 精度：沙箱全量 40/40（L0+L1）、L2 8/8 拒绝、Boundary 4 PASS + 1 结构性 WARN；
  L0 数字与旧版同量级（dgkn 树折叠求和序使 l0_repr dg 7.3e-4→1.2e-3，帽 1e-2，
  ratio 全 1.0000）；valrange_l dg 8.3e-3→7.8e-3（余量略改善）。
- 交付件：worktree（feat/fla-chunk-inter-pr）`python test_example_fla_chunk_inter.py`
  → "Kernel Output Match!" exit 0；不同长度路径复跑通过（布局仍随路径翻转，
  但已无放置敏感 op）。生成 C（bench 与 l0_min 两 shape）grep
  tmp_ub/Broadcast/reduce_2d/reduce_sum/sharedTmp/scratch 均 0 次。
- 性能：bench (1,8192,64,128,128) do_bench 3 次均值 **7510.3 us**（基线 6767，
  **+11.0%**）。归因：produce dgkn 链 +539us（transpose 为主）、gn gate +565us
  （64 逐行 Sub 调用开销）、drain fold +170us；移除旧 op 约 −560us。
  收敛候选：① ones-gemm 对角化（dgkn 移 C 侧，V −539us；dgkn 精度降为 hi/lo
  2^-22 类，valrange_l dg 7.8e-3 贴近 1e-2 帽需实测）② 上游 fused 列广播乘
  原语（消除 64 次行 Sub）。
- 上游反馈项：planner 指针序确定性（D-1）；pip 旧版 head-set 丢弃；行广播缺
  fused 原语；TIR while 死循环。

## 15. Stage 3 R4 结构级增补（2026-11-03，收敛战役：scratch-free 形态下的性能回收）

### 15.1 三项采纳（7510.3 → 6868.3 us，-8.55%）

| # | 优化 | 结构 | 实测（5 次独立 do_bench 均值口径） | 精度 |
|---|------|------|----------------------------------|------|
| R4-2 | dgkn ones-gemm 对角化移 C 侧 | diag(h@dh^T) 由 C 侧 3 个 hi/lo 小 gemm（复用 L1 已有 h_hi/lo、dh_hi/lo，L0C 第 4 frag 64KB<128KB）产出 [bDK,bDK] fp32 → fixpipe 入 dgm_relay 段（与 dq/dk 同 40+j%2 手动 flag 组）→ drain 一次 T.tile.gather 常量 byte 偏移对角提取（prologue arith_progression+reinterpretcast 构建，AscendC gather NoWorkspace 零注入）；produce 侧 mul+transpose+chunk-adds+树+行拷贝整链与 32KB prodT_ub 删除 | 7503.7 → 7167.1（-4.49%） | L0+L1 40 PASS；valrange_l dg 7.812e-3 与基态持平（hi/lo 类误差贡献不显著，实证） |
| R4-3 | gn 门控向量化（bDK=64 路径） | 64 逐行 Sub → transpose(g_ub→gT，复用已死 gexp_ub) + row_expand_sub_experiment（brcb 逐行标量块展开 + 单次 masked Sub）+ mul(-1) 保号（exp 参数逐位一致）+ exp + transpose 回；bDK=32 行宽 128B 不满足 256B 行窗 → trace-time 分支保留 legacy 行链 | 7167.1 → 7012.3（-2.16%；实测噪声 ≤0.5%、区间不交叠 ~14σ，按"无硬性数字门禁，按证据推进"条款采纳） | l0_repr dg 1.221e-3 与改前逐位一致（bitwise 等价实证） |
| R4-4 | drain 三中继读融合 | dq/dk/dgm 三个 GM 中继 → 一个 ddr_relay（行段 [0,BS)\|[BS,2BS)\|[2BS,2BS+bDK)），drain 3×16KB 读 → 1×48KB 单读；消费侧全行区域视图（flat-contiguous）；签名 25→23（out_idx=[19..22]） | 7012.3 → 6868.3（-2.05%；区间不交叠 ~10σ） | L0 误差与改前逐位一致 |

终态 vs 目标：**6868.3 vs 6767（R2 档案）= +1.50%，进入 ±3% 收敛带**（R4 收敛条件①达成）。

### 15.2 结构地板归因（剩余 +101us 的构成，全部实证封堵）

- dgm 中继回读：16KB/(task,bk) ≈ 291us mte2 时间（53GB/s 引擎界）。**封堵**：
  gather 仅读 UB（C 侧无 gather）、fixpipe 仅连续写（无法抽对角）、gemm 无法紧凑
  产出对角列（置换分析：diag 列映射恒等，任何 B 行置换均破坏对应关系）。
- 门控 2 次转置 ≈ 2×1.6us×102.4/(task,bk) ≈ 328us vs 旧 broadcast 形态。
  **封堵**：broadcast 触发 scratch 注入（D-1/D-2 红线）；转置+brcb+单 Sub 是
  无 scratch 行广播的实测最小形态（反模式清单"正交轴串行化"官方解法）。
- drain 树 6 语句 ≈ 170us vs 旧 reduce_sum。**封堵**：wholereduce 每 repeat 仅
  8-block lane 求和、不跨 repeat 累加（64 行折叠不可能，codegen 语义实证）；
  sum_experiment 为 last-dim 归约（轴不符）；先转置再归约成本反超树本身。
- mte2 字节地板：416KB/task（h/dh 128 + do/v/dv 96 + G/Q/K 64 + 中继 48 + 尾项）
  53GB/s 引擎界——全部为接口/结构必需字节；fp16 中继 hi/lo 对与 fp32 同字节。
- 结论：**Developer + scratch-free 双约束下的形态地板 ≈ 6868us**；6767 为
  允许 scratch 注入 op（reduce_sum/broadcast）时的形态。R3 会话对同代 R2 形态
  实测 6639-6720（会话间方差 ±1.2%），本会话锚点法消除该混杂。

### 15.3 工程要点（新增实证）

- gather 的 offset 构建范式：int32 buffer + arith_progression(first=0,
  diff=(bDK+1)*4) + T.reinterpretcast(uint32)（cann-bench/gather 生产同款）；
  偏移 buffer 在 prologue 一次构建，兼作 head-set 的 live 锚（替代旧 dgkn_ring
  零填充锚——无锚 loop 外 set_cross_flag 被 pip 旧版丢弃的教训仍适用）。
- row_expand_*_experiment 家族约束：dst/src0 需 256B 行宽（f32 64 列）、行数
  8..248 且 %8、物理行 stride 匹配 ≤255 block、src1 标量线性 [R]、tmp 显式
  [rows×8]（tmp 路径有 UT 覆盖；FA expert_v2/v3 为无 tmp packed 生产路径）。
- 中继读融合的发射收益超预期（-144us vs 标量缓解估计 20-50us）：3 次独立
  GM 读 → 1 次的 mte2 发射/屏障开销削减在 mte2 带宽界场景仍有效。
- 精度纪律：R4-2/R4-3/R4-4 的 L0 误差数字与各自改前逐位一致（除 R4-2 的
  dgkn hi/lo 类迁移，valrange_l dg 持平 7.812e-3）。
