# fla_chunk_delta 算子设计文档（KDA gated delta rule chunk 状态递推，fwd + bwd 双 kernel）

> GPU→Ascend 迁移设计（tilelang-op-migration / Stage 1，tilelang-op-design skill 产出）。
> 语义权威（只读，禁止修改）：`ref_gpu/fla_chunk_delta.py`（Triton/FLA，tile-ai/tilelang@4805777d `examples/kda/FLA_KDA/fla_chunk_delta.py`，579 行）+ `ref_gpu/fla_utils.py`（exp/exp2 语义）。
> 任务书：`proto_fla_chunk_delta.yaml`（orchestrator 已按冻结源码逐行提炼）。
> 目标硬件：**Ascend 910B3（DAV_2201）**——20 Cube + 40 Vector（AIC:AIV=1:2）、L1 512KB、L0A/L0B 各 64KB、L0C 128KB、UB 192KB、L2 192MB、HBM 标称 1.6TB/s【本机同族实测上限 1263GB/s，见 §9.5】。参数出处：`.agents/skills/npu-arch/references/npu-hardware-params.md` §2.3/§3（DAV_2201 行）与 `npu-arch-guide.md` §3（910B2 Memory 带宽 1.6TB/s，910B3 为同族 20 核降规 SKU）。
> 本文档区分【已确认事实】（源码/示例/实测验证）、【推断】（标注依据）、【需 Stage 2 验证】三类信息。

---

## 1. 概述

### 1.1 算子名称

`fla_chunk_delta`（含 2 个独立 kernel：`fla_chunk_delta_fwd` / `fla_chunk_delta_bwd`，上游名 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` / `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`）

### 1.2 功能描述

KDA (Kimi Delta Attention) gated delta rule 的 chunk 状态递推。fwd 对每个 (b, h) 维护 fp32 状态 H ∈ R^{K×V}，跨 chunk（BT=64）正向串行递推：每 chunk 先存状态快照 h[t]，再计算 v_new = u − w@H，施加 g（逐 token 标量）与 gk（逐 token per-channel）双门控衰减后更新 H += k^T@v_new。bwd 为反向对偶：维护状态梯度 dH 反向串行递推，产出 dh（每 chunk 状态梯度快照）与 dv2（完整 value 梯度）。**跨 chunk 串行依赖 + 状态常驻 + 每 chunk 多组小 GEMM 与 Vector 门控交织**是本算子的结构特征。

### 1.3 数学公式

记 chunk 长度 BT=64（支持域 S%64==0），NT = S/64；每 (b, h) 维护状态 H_t ∈ R^{K×V}（fp32，K≤256 按 64 行 slab 分块）；n_t = chunk t 末 token 下标；⊙ 逐元素乘；exp 按 USE_EXP2 取 exp/exp2（g 输入在对应底数域，本设计统一以 exp(x) 表述，exp2 走 exp(x·ln2) 等价替代，见 §3.5.2-5）。

**fwd（t = 0..NT−1 正向递推）**：

$$H_0 = h_0 \;(\text{USE\_INITIAL\_STATE}) \text{ 或 } 0$$
$$h[b,t,h,:,:] = H_t \qquad \text{（每 chunk 先存状态，进入 chunk t 时）}$$
$$v\_new[t] = u[t] - w[t] @ H_t \qquad [BT, DV]\;(\text{SAVE\_NEW\_VALUE 时写出})$$
$$v\_new[t] \;\odot=\; \text{mask}_t \odot \exp(g[n_t] - g[t]) \qquad \text{USE\_G，逐 token 标量门控}$$
$$H_t \;\odot=\; \exp(g[n_t]) \qquad \text{USE\_G，整状态衰减}$$
$$H_t[k,:] \;\odot=\; \exp(gk[n_t, k]) \qquad \text{USE\_GK，per-channel 衰减}$$
$$v\_new[t] = v\_new[t].\text{to}(k.\text{dtype}) \qquad \text{GEMM 前降精度}$$
$$H_{t+1} = H_t + k[t]^T @ v\_new[t] \qquad [K, BT]@[BT, DV]$$
$$ht = H_{NT} \qquad \text{STORE\_FINAL\_STATE，fp32}$$

**bwd（t = NT−1..0 反向递推）**：

$$dH_{NT} = dht \;(\text{USE\_FINAL\_STATE\_GRADIENT}) \text{ 或 } 0$$
$$dh[b,t,h,:,:] = dH_t \qquad \text{（每 chunk 先存状态梯度）}$$
$$dv\_part[t] = (k[t] @ dH_t) \odot \text{mask}_t \odot \exp(g[n_t] - g[t]) \qquad \text{USE\_G}$$
$$dv2[t] = dv[t] + dv\_part[t] \qquad [BT, DV] \text{ 写出}$$
$$dH_t \;\odot=\; \exp(g[n_t]) \qquad \text{USE\_G}$$
$$\tilde{q}[t] = q[t]^T \odot \exp(g[t])_{[\text{broadcast},:]} \qquad \text{USE\_G（}q^T\text{ 逐列衰减）}$$
$$dH_t[k,:] \;\odot=\; \exp(gk[n_t, k]) \qquad \text{USE\_GK}$$
$$dH_{t-1} = dH_t + (\tilde{q}[t] @ do[t]) \cdot scale - w[t]^T @ dv2[t]$$
$$dh0 = dH_{-1} \qquad \text{USE\_INITIAL\_STATE，fp32}$$

注：mask_t = (t·BT + arange(BT)) < T（末 chunk 越界行置 0）；本算子支持域 S%64==0 → mask_t 恒真；gk 越界通道以 0 填充（exp(0)=1 不衰减）；fwd/bwd 状态存取均发生在 chunk 处理前。

### 1.4 算法描述（计算步骤分解）

**fwd**（每任务 = 一条 (b, h, i_v) 递推链，共 B·H·NV 条，NV = ceil(DV/BV)；链内 NT 个 chunk 串行）：

1. （USE_INITIAL_STATE）h0 [K, BV] fp32 分 slab 载入 UB，初始化 H。
2. chunk t 循环（正向串行，NT 次）：
   a. **存 h[t]**：H（UB fp32）分 slab cast fp16 → 写 GM（h 数组 5D）。
   b. **v_new 前半**：H 分 slab cast fp16 → relay → C 侧读入 L1；w[t] slab GM→L1；`gemm_v0(w, H16)`（K 维 slab 累加）→ L0C b_v → relay → V 侧读回 UB fp32；v_new = u[t] − b_v（UB fp32 域，u 经 GM→UB 两步 cast 链）。
   c.（SAVE_NEW_VALUE）v_new cast fp16 → 写 GM。
   d. **门控（USE_G）**：g 经 gather 链读入（[B,S,H] strided 规整化）；exp(g[n_t]−g[t]) 行广播乘 v_new（mask_t 恒真，直乘）；exp(g[n_t]) 标量乘 H。
   e. **门控（USE_GK）**：gk 末行 [K] 连续读入；exp 后行广播乘 H（分 slab）。
   f. v_new cast fp16 → relay → C 侧读入 L1；k[t] slab GM→L1；`gemm_v0(k, v_new16, transpose_A=True)`（K 维 slab 输出拼接）→ L0C kH → relay → V 侧分 slab 读回 UB；H += kH（T.tile.add）。
3. （STORE_FINAL_STATE）ht：H cast fp32 直写 GM（已是 fp32，免 cast）。

**bwd**（对偶，链内 NT 个 chunk 反向串行）：

1. （USE_FINAL_STATE_GRADIENT）dht [K, BV] fp32 载入初始化 dH。
2. chunk t 循环（NT−1..0）：
   a. **存 dh[t]**：dH 分 slab cast bf16 → 写 GM。
   b. **dv2**：dH 分 slab cast bf16 → relay → C 读 L1；k[t] GM→L1；`gemm_v0(k, dH16)` slab 累加 → L0C dv_part → relay → V 读回；dv_part ⊙= exp(g[n_t]−g[t]) 行门控；dv2 = dv[t] + dv_part → 写 GM（bf16）。
   c. **dH 更新**：q[t] 门控（q ⊙ exp(g[t]) 行门控，等价于上游转置域列门控）→ cast bf16 → relay → C 读 L1；do GM→L1；`gemm_v0(q̃, do, transpose_A=True)` → L0C qdo；w[t] GM→L1、dv2（b 步 V 侧结果）cast bf16 → relay → C 读 L1；`gemm_v0(w, dv2_16, transpose_A=True)` → L0C wdv2；qdo/wdv2 合包 relay → V 读回；dH = dH⊙exp(g[n_t]) + qdo·scale − wdv2（UB fp32 域；gk per-channel 门控分 slab 施加）。
3. （USE_INITIAL_STATE）dh0：dH fp32 直写 GM。

> **Host 侧 Buffer 操作约束**（ascend-constraints.md §5）：本算子 host 路径只含 (a) 输入张量构造（CPU randn/cumsum → 单次 H2D）、(b) 输出张量分配（`new_empty`，数据准备范畴）、(c) bshd→3D 的零拷贝 metadata view（见 §4.4）、(d) relay workspace 的 `empty` 分配、(e) kernel 调用、(f) 精度验证。无 `.contiguous()`/pad/cat/`reshape` 物理化/`.to()`/aclnn 调用。

### 1.5 数据流图

```
fwd（每链，chunk t；┄┄ 为跨核 relay（L2 常驻 GM workspace），── 为核内搬运/计算）：
GM[h_out 5D] ←──cast fp16── H_ub(fp32, V 侧常驻 [K,BV])
GM[w] ──MTE1→ L1[w16] ─┐
H_ub ──cast fp16─┄┄relay┄┄→ L1[H16] ─┴─gemm_v0(w,H16)→ L0C[b_v] ─┄┄relay┄┄→ UB[b_v]
GM[u] ──MTE2→ UB[u16→u_f32] ──sub── v_new = u − b_v ──cast fp16── GM[v_new]（SAVE_NEW_VALUE）
GM[g] ─gather链→ UB[g] ──exp/broadcast── v_new ⊙= exp(g_last−g)；H ⊟= exp(g_last)
GM[gk 末行] ──MTE2→ UB[gk_last] ──exp/broadcast── H ⊟= exp(gk_last)（分 slab）
v_new ──cast fp16─┄┄relay┄┄→ L1[v16]；GM[k] ─MTE1→ L1[k16] ─gemm_v0(k,v16,ᵀA)→ L0C[kH] ─┄┄relay┄┄→ UB[kH] ──add── H

bwd 同构（见 §1.4-2）：3 组 gemm（k@dH、q̃@do、wᵀ@dv2），dH 驻 UB，门控链同款。
```

### 1.6 范围与分派覆盖审计

【已确认事实】范围裁剪（proto_fla_chunk_delta.yaml + Phase 1 用户裁定）：定长序列（IS_VARLEN 不实现，S%64==0）、USE_G=True + USE_GK=True（g/gk 必传，性能验收口径）、USE_INITIAL_STATE / STORE_FINAL_STATE（fwd）/ USE_FINAL_STATE_GRADIENT（bwd）/ SAVE_NEW_VALUE / USE_EXP2 以 constexpr 分支全支持并测试覆盖；benchmark 取 wrapper 默认（无 h0/ht、save_new_value=True、use_exp2=False）。

```text
[DISPATCH-COVERAGE]
supported_domain: S%64==0 定长; DK, DV ∈ 16 的倍数且 DK ≤ 256（slab=64 分块）、DV ≥ 16;
  fwd dtype = k/w/u fp16 + g/gk fp32 + h/v_new fp16 输出（+ ht/h0 fp32 可选）;
  bwd dtype = q/k/w/do/dv bf16 + g/gk fp32 + dh/dv2 bf16 输出（+ dht/dh0 fp32 可选）;
  USE_G=USE_GK=True 恒开; chunk_size=64; use_exp2 ∈ {False, True}
generic_fallback: fla_chunk_delta_fwd / fla_chunk_delta_bwd 各一条通用路径
  （constexpr 分支为编译期特化，不构成运行时分派谓词；无 dtype/shape 快路径分派）
specializations: 无
unsupported_inputs: varlen（cu_seqlens）、S%64≠0、USE_G/GK=False（g/gk 缺省）、DK>256
  —— 与 supported_domain 不冲突（proto 已显式排除）
result: pass
```

---

## 2. 编程模式选型

### 2.1 模式结论

**Developer 模式**（用户明确指定 2026-11-11，禁止更改）——`alloc_shared` / `alloc_fragment` + 四件套 auto pass + `threads=2`。本算子属 Developer 复杂场景（跨 chunk 串行递推 + 每 chunk 4 次跨核交换），按 design-template §8.2 的"复杂场景回退"分支使用**显式命名 relay workspace**（`workspace_idx`，chunk_o 已验证先例），不用 AUTO_CV_COMBINE 自动注入的 per-block workspace（实测 ~768MB/launch，chunk_o 注释记录）。

### 2.2 选型理由

1. 用户硬性指定 Developer；家族先例（沙箱 `fla_wy_fast.py`、`example_fla_chunk_o.py`）均为 Developer API（alloc_shared/fragment + gemm_v0 + T.tile.*）。
2. 算子是典型 CV 融合（Cube GEMM + Vector exp 门控/逐元素/cast），计算交织密集但**无跨核自定义同步拓扑需求**（relay 交换由 AUTO_CV_SYNC 对 `workspace_*` 命名 buffer 自动插 flag，chunk_o 实证）。
3. 带宽敏感 + 延迟敏感（串行链），应避免 Expert 显式 GM 大 workspace 中转的通用方案（fla_wy_fast 被否方案 B 的教训）；但串行递推场景下小容量 L2 常驻 relay 是 chunk_o 实测最优结构（见 §10.4 被否方案 C 的对比论证）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（编译器映射 L1/UB）+ `T.alloc_fragment`（映射 L0C）+ 显式 relay workspace（GM 地址空间，L2 常驻） |
| GEMM | `T.gemm_v0(A, B, C, transpose_A/B, init)`：A/B 为 shared(L1)，C 为 fragment(L0C)；**全部操作数 fp16/bf16（fp32 A/B 在本工具链产生交错垃圾数据——chunk_o/fla_wy_fast 双重实测）** |
| 计算方式 | buffer 级 `T.tile.*`（mul/add/sub/exp/broadcast/select）；**禁止 element T.Parallel 循环（实测 -26%~-71% 反模式）** |
| 作用域 | 编译器自动 C/V 分离（AUTO_CV_COMBINE），无 T.Scope |
| 同步方式 | 自动同步（AUTO_SYNC + AUTO_CV_SYNC 对 relay 自动插 flag）；Stage 3 可选手动 throttle flag（chunk_o 模式，见 §6.3） |

---

## 3. API 映射设计

### 3.1 公式拆解（fwd）

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| F1 | `h[t] = H.to(fp16)` | 状态快照存 GM（分 slab cast） |
| F2 | `b_v = w[t] @ H.to(fp16)` | gemm_v0，M=BT=64, K=DK(=128 合并单 gemm), N=BV |
| F3 | `v_new = u[t] − b_v` | UB fp32 域 sub |
| F4 | `v_new ⊙= exp(g[n_t] − g[t])`（行门控） | gather 链读 g + exp + broadcast(axis=1) + mul |
| F5 | `H ⊟= exp(g[n_t])`（标量乘） | 标量 mul（USE_EXP2 时 exp((·)·ln2)） |
| F6 | `H[k,:] ⊟= exp(gk[n_t, k])`（行门控，分 slab） | gk 末行连续读 + exp + broadcast(axis=1) + mul |
| F7 | `kH = k[t]^T @ v_new.to(fp16)` | gemm_v0(k, v16, **transpose_A=True**)，M=DK, K=BT, N=BV（slab 输出拼接） |
| F8 | `H += kH` | T.tile.add（分 slab） |

### 3.2 公式拆解（bwd）

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| B1 | `dh[t] = dH.to(bf16)` | 状态梯度快照（分 slab cast） |
| B2 | `dv_part = k[t] @ dH.to(bf16)` | gemm_v0，M=BT, K=DK, N=BV（slab 累加） |
| B3 | `dv_part ⊙= exp(g[n_t] − g[t])`；`dv2 = dv + dv_part` | 行门控 + add → 写 GM bf16 |
| B4 | `q̃ = (q ⊙ exp(g[t])).to(bf16)` | q 非转置域行门控（≡ 上游转置域列门控 q^T⊙exp(g)[None,:]）+ cast |
| B5 | `qdo = q̃^T @ do`；`wdv2 = w^T @ dv2` | gemm_v0(·, transpose_A=True) ×2，M=DK, K=BT, N=BV |
| B6 | `dH = dH⊙exp(g[n_t]) + qdo·scale − wdv2` | UB fp32 域（gk per-channel 门控分 slab 施加） |

### 3.3 TileLang API 映射（含出处，全部已在本仓验证）

| 步骤 | TileLang API（昇腾） | 参数 | 出处（已验证） |
|------|---------------------|------|---------------|
| Kernel（性能主选） | `T.Kernel(CORE_EFF=20, threads=2, is_npu=True) as (cid)` | fixed-core + 核内 task loop（striped：`tid = cid + k*CORE_EFF`，任务总数 B·H·NV） | `examples_experiment/dsa_sparse_finetune/example_fla_chunk_o.py:285`（CORE_EFF/threads=2）；`examples/fused_sigmoid_gating_delta_rule/fused_sigmoid_gating_delta_rule_varlen.py:66`（NUM_CORES + max_work_per_block 同款） |
| Kernel（回退/L0 冒烟对照） | `T.Kernel(B*H*NV, threads=2, is_npu=True) as (cid)` | 每 block 一条链 + `T.Pipelined(NT, num_stages=2)` | `examples/developer_mode/flash_attn_bshd_developer.py:47`；fla_wy_fast.py:130（Pipelined 内片上直连） |
| GEMM | `T.gemm_v0(A, B, C, transpose_A=, transpose_B=, init=)` | A/B→shared(L1) 半精度，C→fragment(L0C) fp32 | `tilelang/language/ascend.py:418`；`examples/linear_attention_and_rnn/gdn/gdn_chunk_h.py:73`（**transpose_A=True 先例**）；fla_wy_fast.py:181 |
| 内存分配 | `T.alloc_shared(shape, dtype)` / `T.alloc_fragment(shape, dtype)` | shared→L1/UB、fragment→L0C | flash_attn_bshd_developer.py:52-59；fla_wy_fast.py:137-152 |
| 元素运算（buffer 级） | `T.tile.mul/add/sub(dst, src0, src1/scalar)`、`T.tile.exp(dst, src)` | 禁 element T.Parallel | api-compute.md §4.1；fla_wy_fast.py:172-176（实测反模式记录于 chunk_o 注释 #7） |
| 行/列广播 | `T.tile.broadcast(dst2d, src1d, axis=1/0)` | 行广播 axis=1（token 门控）、列广播 axis=0 | fla_wy_fast.py:172（axis=1）、:292（axis=0） |
| [B,S,H] strided 读 | flat span `T.copy` + `T.tile.arith_progression` + `T.reinterpretcast` + `T.tile.gather` | sub-32B strided DMA 不可靠的规整化（g [B,S,H] fp32 同 beta 款） | fla_wy_fast.py:156-159（beta [B,S,H] 同款实测） |
| 跨 dtype cast | UB 内两步链：`T.copy(fp32_ub, half_ub)` / `T.copy(half_ub, fp32_ub)` | **GM↔UB cast-on-copy codegen 损坏（fla_wy_fast #8 / chunk_o #3 双重实测），禁止跨层级直 cast** | fla_wy_fast.py:160/168/183-184 |
| relay 交换 | 显式命名 `workspace_*` GM tensor + `T.copy` | AUTO_CV_SYNC 按名匹配自动插跨核 flag | example_fla_chunk_o.py:279-282 + 注释（"AUTO_CV_SYNC matches buffers named workspace_*"） |
| 循环 | `T.serial(NT)`（task loop 内 chunk 递推）；`T.Pipelined(extent≥2, num_stages=2)`（回退路径） | Pipelined body 需 ≥2 迭代（extent-1 崩溃）；**plain serial 内的片上 V→C 直连交换会 mis-sync（fla_wy_fast #1），故 serial 主选必须走 GM relay** | fla_wy_fast.py:162；gdn_chunk_h.py:63（serial + GM workspace 中转同款） |
| 清零/初始化 | `T.tile.fill(ub, 0.0)`；h0 载入覆盖 | — | fla_wy_fast.py（dbeta 清零同款） |
| L0C 读取 | `T.copy(fragment, shared_ub)` | L0C→UB 直拷 | flash_attn_bshd_developer.py:82 |

### 3.4 计算伪代码（fwd 主结构，性能主选路径）

```python
LN2 = 0.6931471805599453  # exp2 无昇腾后端，exp(x*LN2) 等价替代
CORE_EFF, NSLOT = 20, 1   # L0 朴素单槽 lockstep；Stage 3 → 9 + 手动 throttle

@tilelang.jit(out_idx=[...], workspace_idx=[...], pass_configs=PASS_CONFIGS)
def fla_chunk_delta_fwd(B, S, H, DK, DV, chunk_size=64, BV=64):
    BS, NT, NV = chunk_size, S // chunk_size, T.ceildiv(DV, BV)
    NCHAIN = B * H * NV
    NSLAB = T.ceildiv(DK, 64)
    ntask = T.ceildiv(NCHAIN, CORE_EFF)          # 每核任务数（编译期）

    @T.prim_func
    def main(
        K: T.Tensor((B, S, H, DK), "float16"), W: T.Tensor((B, S, H, DK), "float16"),
        U: T.Tensor((B, S, H, DV), "float16"),
        G: T.Tensor((B, S, H), "float32"), Gk: T.Tensor((B, S, H, DK), "float32"),
        H0: T.Tensor((B, H, DK, DV), "float32"),           # USE_INITIAL_STATE 时有效
        Hout: T.Tensor((B, NT, H, DK, DV), "float16"),
        Vnew: T.Tensor((B, S, H, DV), "float16"),
        Ht: T.Tensor((B, H, DK, DV), "float32"),
        GFlat: T.Tensor((B, S * H), "float32"),            # g 的 3D flat view（host metadata 传入）
        workspace_1: T.Tensor((CORE_EFF, NSLOT, DK, BV), "float16"),  # V→C: H16 (fwd)
        workspace_2: T.Tensor((CORE_EFF, NSLOT, BS, BV), "float32"),  # C→V: b_v
        workspace_3: T.Tensor((CORE_EFF, NSLOT, BS, BV), "float16"),  # V→C: v_new16
        workspace_4: T.Tensor((CORE_EFF, NSLOT, DK, BV), "float32"),  # C→V: kH
    ):
        with T.Kernel(CORE_EFF, threads=2, is_npu=True) as (cid):
            # ---- V 侧 buffer（alloc_shared，AUTO_CV_COMBINE 自动分核）----
            h_ub    = T.alloc_shared([DK, BV], "float32")     # 状态 H 常驻（fp32）
            h16_ub  = T.alloc_shared([64, BV], "float16")     # slab cast 中转
            bv_ub   = T.alloc_shared([BS, BV], "float32")     # b_v 读回
            kh_ub   = T.alloc_shared([64, BV], "float32")     # kH slab 读回
            u16_ub  = T.alloc_shared([BS, BV], "float16"); u_ub = T.alloc_shared([BS, BV], "float32")
            vn_ub   = T.alloc_shared([BS, BV], "float32"); vn16_ub = T.alloc_shared([BS, BV], "float16")
            g_span  = T.alloc_shared([BS * H], "float32")     # g gather 链（flat span）
            off32   = T.alloc_shared([BS], "int32"); offu32 = T.alloc_shared([BS], "uint32")
            g_ub    = T.alloc_shared([BS], "float32"); gexp_ub = T.alloc_shared([BS], "float32")
            ge2d    = T.alloc_shared([BS, BV], "float32")     # 行门控 broadcast
            gkl_ub  = T.alloc_shared([64], "float32"); gkl2d = T.alloc_shared([64, BV], "float32")
            # ---- C 侧 buffer ----
            w_l1    = T.alloc_shared([BS, 64], "float16")     # w slab（GM→L1 直读）
            k_l1    = T.alloc_shared([BS, 64], "float16")     # k slab
            h16_l1  = T.alloc_shared([64, BV], "float16")     # relay 读入
            v16_l1  = T.alloc_shared([BS, BV], "float16")
            bv_frag = T.alloc_fragment([BS, BV], "float32")   # L0C
            kh_frag = T.alloc_fragment([64, BV], "float32")   # L0C（slab 槽）

            for k_task in T.serial(ntask):
                tid = cid + k_task * CORE_EFF                # striped 任务索引
                # tid → (i_v, i_nh) 分解；S%64==0 支持域，无尾块分支
                # ... (USE_INITIAL_STATE: h0 载入 h_ub；否则 T.tile.fill(h_ub, 0.0))
                for i_t in T.serial(NT):                      # 跨 chunk 串行递推
                    # (a) 存 h[t]：分 slab cast fp16 写 GM（5D 切片，行 gap=2*DV≥32B 合法）
                    # (b) V: h_ub slab → h16_ub（两步 cast 链）→ workspace_1[cid, 0] ┄┄
                    #     C: workspace_1 → h16_l1；w_l1 GM→L1；
                    #        for s in serial(NSLAB): gemm_v0(w_l1[s], h16_l1, bv_frag, init=(s==0))
                    #     C: bv_frag → workspace_2 ┄┄ → V: bv_ub
                    # (c) V: v_new = u − b_v（u16→u 两步链）；SAVE_NEW_VALUE: vn16 写 GM
                    # (d) V: g_span span 读 + gather 链 → g_ub；
                    #        gexp = exp(g_last − g)（use_exp2: ×LN2）；
                    #        broadcast(ge2d, gexp, axis=1)；vn ⊟= ge2d；
                    #        h_ub ⊟= exp(g_last)（标量，全 [DK,BV]）
                    # (e) V: gk 末行 [64] slab 读 → exp → broadcast(axis=1) → h_ub slab ⊟=
                    # (f) V: vn → vn16 → workspace_3 ┄┄ → C: v16_l1；k_l1 GM→L1；
                    #        for s in serial(NSLAB): gemm_v0(k_l1, v16_l1, kh_frag, transpose_A=True, init=True)
                    #        逐 slab: kh_frag → workspace_4 ┄┄ → V: kh_ub；h_ub slab ⊟= kh_ub
                # (STORE_FINAL_STATE) Ht fp32 直写
```

bwd 伪代码同构（§3.2 B1-B6；3 组 gemm，qdo/wdv2 合包单 relay slot），完整结构见 §6.2。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（强制检测）

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **是**（上游 `grid=(V/BV, B*H)` 二维） | 性能主选：一维 fixed-core `T.Kernel(20)` + 核内 task loop，`tid = cid + k*CORE_EFF` striped 分解（chunk_o:285 / fused_sigmoid_gating_delta_rule:66 双先例）；回退路径：`T.Kernel(NCHAIN)` + `i_v = cid % NV; i_nh = cid // NV`（fla_wy_fast.py:131 同款） |
| threads 仅 1/2 | **是**（上游 num_warps 2/4） | `threads=2`（1 AIC + 2 AIV，910B3 全核占用：20×2=40 AIV ✓；chunk_o:67 实测论证） |
| 动态循环边界 | **否**（上游 `do_not_specialize=["T"]` 的 T 为运行时标量） | S 作为 JIT 编译期常量（shape 参数）→ NT=S//64、NSLAB=DK//64、NV=DV//BV 全编译期；每 shape 重编译（家族惯例，fla_wy_fast `_grid_dims` 同款） |
| Pipelined 动态边界 | **否** | 全静态；回退路径 `T.Pipelined(NT, num_stages=2)` 的 NT≥2 由 L0 用例保证（S≥128） |
| L0C 溢出（128KB） | 已核算 | fwd 峰值 48KB、bwd 峰值 48KB（§4.5 预算表，slab 槽复用后） |
| T.copy 列方向 strided 切片 | 已规避 | (1) g [B,S,H] 的 per-token stride-H 读 → flat span + gather 链（fla_wy_fast #4 同款）；(2) gk 末行为尾维连续 1D 切片 ✓；(3) h/dh 写出为"行区间 × 尾维 block"（行 gap = 2·DV ≥ 32B，DataCopyPad 行间 gap 合法）；(4) vid 偏移安全：Q/K/W/U/G 等 GM 访问经 3D row views `[B, S, H*D]`（chunk_o #8：4D bshd 切片的 vid 偏移会落到 H 索引造成静默错位），V/Hstate/relay 由 C 侧消费或经 [CORE,NSLOT,...] 显式槽定址 |
| GEMM 分形限制 | 已核算 | fp16/bf16：M≥16, K≥16, N≥16。全部 gemm：[64, 128]@[128, 64]（M=64,K=128,N=64）、[128, 64]@[64, 64]（M=128,K=64,N=64）✓（api-compute.md §分形表） |
| GEMM 非整除 | 低风险 | S%64==0（支持域断言）；DK 按 64 行 slab（DK≤256，尾 slab 不足 64 行时 gemm 维度仍为 64——**上游以 mask/边界检查处理，本设计支持域要求 DK 为 64 的倍数（benchmark 128）**，非 64 倍数的 DK 留 Stage 2 L1 扩展（§10.3）；DV 按 BV=64 整除（benchmark 128 → NV=2） |
| C/V combine 单 Pipelined body | **是**（关键） | 性能主选的 chunk 递推循环用 **T.serial + GM relay**（gdn_chunk_h:63 同款结构 + AUTO_CV_SYNC 自动 flag），不依赖 Pipelined 片上直连；回退路径（Pipelined 片上直连）仅限每-block-一链的简单 grid（fla_wy_fast 已验证形态，无循环携带交换） |

#### 3.5.2 Triton/CUDA 专属机制的昇腾适配（8 项）

| # | 上游机制 | 昇腾适配结论 | 依据 |
|---|---------|-------------|------|
| 1 | 二维 grid `(cdiv(V,BV), B*H)` + program_id | 一维化：fixed-core(20) + task loop（主选）/ `T.Kernel(NCHAIN)` + 索引分解（回退） | §3.5.1-1 |
| 2 | `@triton.autotune`（BV∈{32,64}×warps×stages） | L0 固定 BV=64、NSLOT=1（lockstep）；threads 昇腾仅 1/2 → 搜索空间删 threads 维；Stage 3 按 (BV, NSLOT, 批处理深度) 枚举 | ascend-constraints §1；chunk_o 调优路径先例 |
| 3 | `tl.load(make_block_ptr)` + boundary_check | S%64==0 支持域 → 边界分支退化为直拷；gk 末行掩码（o_k < K）在 DK 为 64 倍数域恒真 | §3.5.1 尾块行 |
| 4 | `do_not_specialize=["T"]`（T 运行时标量） | S 提升为 JIT 编译期常量，循环边界全静态 | §3.5.1 动态边界行 |
| 5 | `exp2`（fla_utils `tldevice.exp2`） | **`T.exp2` 无昇腾后端 → `T.tile.exp(dst, src·LN2)` 等价替代**（USE_EXP2 分支：g 输入由调用方给 log2 域，kernel 内 exp(g·ln2) = 2^g 精确成立；fp32 相对误差 ~1e-7，远小于 fp16/bf16 档容差） | fla_wy_fast #2（LN2 常数实测）；src/op 仅 ascend_exp→AscendC::Exp 映射 |
| 6 | `tl.dot(b_w, b_h.to(b_w.dtype))`（状态 cast 参与点积） | `T.gemm_v0`，操作数两步 cast 链进 L1；**fp32 A/B 禁用**（工具链垃圾输出），H16 单 cast 精度风险与 hi/lo 预案见 §9.1/R4 | chunk_o #2 / fla_wy_fast #2 双实测 |
| 7 | `tl.where(m_t, exp(...), 0)` 掩码 | S%64==0 域 m_t 恒真 → 直接乘；（若 Stage 2 扩展非整除域）用 bit-packed 常量 + `T.tile.select`（VSEL），禁乘 0/1 掩码（Inf·0=NaN 污染，chunk_o #6） | chunk_o:87-93 |
| 8 | 寄存器驻留状态（b_h1..b_h4 fp32 block） | 状态驻 **V 侧 UB**（`alloc_shared` [DK, BV] fp32）：V 侧门控乘 exp 必须在 UB；UB→L0C 无通路（fragment 仅作 gemm 输出）；gdn_chunk_h.py:52（s_ub）/fused_sigmoid_gating_delta_rule.py:84（h_vec）双先例 | §4.4 路径图 |

#### 3.5.3 参考实现差异说明（GPU → Ascend 汇总）

| 差异项 | 参考实现（Triton/CUDA） | 本项目（Ascend 910B3） | 转换方案 |
|--------|------------------------|----------------------|---------|
| 状态驻留 | 寄存器块（b_h1..b_h4 fp32，per-CTA） | V 侧 UB fp32 [DK, BV]（per-链） | §3.5.2-8 |
| grid | (V/BV, B*H) 每 CTA 一条链，warp 内并行 | fixed-core(20) + task loop（每核多链交织） | §3.5.1-1（**结构性差异：GPU 靠 CTA 海量并行掩盖串行链延迟；Ascend 20 核必须靠核内多链交织 + relay 流水填满 C/V 队列**） |
| 状态操作数交换 | 寄存器直接参与 tl.dot | cast → relay workspace（L2 常驻）→ L1 → gemm | §1.5 数据流 |
| exp2 | tldevice.exp2 | T.tile.exp(x·LN2) | §3.5.2-5 |
| GEMM | tl.dot（任意精度组合） | gemm_v0（半精度 A/B + fp32 L0C） | §3.5.2-6 |
| threads/warps | num_warps 2/4 | threads=2 | §3.5.1-2 |
| mask 边界 | boundary_check + tl.where | 支持域 S%64==0 免 mask；扩展用 bit-packed select | §3.5.2-7 |

#### 3.5.4 本项目同类实现参考（强制步骤 0 结果）

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples_experiment/dsa_sparse_finetune/example_fla_chunk_o.py` | **最高（同族 KDA、同 bench 形状、同 Developer+relay 模式）** | fixed-core task loop、显式 relay workspace（AUTO_CV_SYNC 按名插 flag）、threads=2 vid 语义、3D views 防 vid 错位、fp32 gemm 禁用、cast 两步链、T.tile 反模式记录、L0C ping-pong、throttle flag ring（Stage 3 全套） |
| `examples/linear_attention_and_rnn/gdn/gdn_chunk_h.py` | **最高（数学同构：w@H / k^T@v_new / g 门控 / chunk 串行递推）** | 状态驻 UB、serial chunk 循环 + GM workspace 中转 + cross flag、transpose_A=True 先例、golden chunk 循环写法 |
| `examples_experiment/dsa_sparse_finetune/fla_wy_fast.py` | 高（同沙箱同族，Developer 纯片上直连变体） | gather 链（[B,S,H] strided）、broadcast 行/列门控、hi/lo 精度补偿、Pipelined 片上直连（本设计回退路径）、exp·LN2 |
| `examples/fused_sigmoid_gating_delta_rule/fused_sigmoid_gating_delta_rule_varlen.py` | 高（gated delta rule 家族、状态递推） | NUM_CORES + max_work_per_block task 分配、H 驻 UB fp32、输出 ping-pong |
| `examples/developer_mode/flash_attn_bshd_developer.py` | 中（Developer CV 融合标准骨架） | threads=2、alloc_shared/fragment、L0C→UB 直拷、跨 dtype T.copy、Pipelined、pass_configs 四件套 |
| `examples/linear_attention_and_rnn/gdn/gdn_wy_fast.py` | 中（WY 表示同族） | gemm_v0 transpose 组合、一维 cid 分解 |

> 注：gdn 系列为 Expert 风格（alloc_L1/ub/L0C + T.Scope + 手动 flag）；本设计按用户裁定用 Developer API 重写，以 chunk_o（Developer + 显式 relay）+ fla_wy_fast（Developer 片上直连）为骨架。gdn_chunk_h 的**数学结构与数据流**是状态递推的直接参照。

---

## 4. 数据规格与内存规划

### 4.1 输入张量（布局全部 bshd 连续；host 侧仅做零拷贝 3D view）

| 参数名 | Shape | dtype（fwd） | dtype（bwd） | 说明 |
|--------|-------|-------------|--------------|------|
| k | [B, S, H, DK] | float16 | bfloat16 | key（bwd 的 k@dH 操作数） |
| w | [B, S, H, DK] | float16 | bfloat16 | WY 表示权重（上游已算好，只消费） |
| u | [B, S, H, DV] | float16 | —（无） | value（fwd；上游形参名 v） |
| g | [B, S, H] | float32 | float32 | 逐 token 标量 log 衰减（chunk 内累积） |
| gk | [B, S, H, DK] | float32 | float32 | 逐 token per-channel log 衰减（kernel 仅读各 chunk 末行） |
| h0 | [B, H, DK, DV] | float32（可选） | — | fwd 初始状态 |
| q | [B, S, H, DK] | — | bfloat16 | bwd 查询 |
| do | [B, S, H, DV] | — | bfloat16 | 上游梯度 |
| dv | [B, S, H, DV] | — | bfloat16 | 入参部分梯度 |
| dht | [B, H, DK, DV] | — | float32（可选） | bwd 终态梯度 |

### 4.2 输出张量

| 参数名 | Shape | dtype（fwd） | dtype（bwd） |
|--------|-------|-------------|--------------|
| h / dh | [B, NT, H, DK, DV] | float16 | bfloat16 |
| v_new / dv2 | [B, S, H, DV] | float16 | bfloat16 |
| ht / dh0 | [B, H, DK, DV] | float32（可选） | float32（可选） |

### 4.3 中间缓冲区（fwd，BV=64，性能主选路径；bwd 同构见 §6.2）

**V 侧（alloc_shared → UB）**：

| Buffer | Shape | dtype | 用途 |
|--------|-------|-------|------|
| h_ub | [DK, BV] | fp32 | **状态 H 常驻**（128×64 = 32KB） |
| h16_ub | [64, BV] | fp16 | H slab cast 中转（写 relay 前） |
| bv_ub | [64, BV] | fp32 | b_v relay 读回 |
| kh_ub | [64, BV] | fp32 | kH slab relay 读回 |
| u16_ub / u_ub | [64, BV] | fp16/fp32 | u 载入 + cast 域（时序复用同槽） |
| vn_ub / vn16_ub | [64, BV] | fp32/fp16 | v_new 计算域 / cast 输出 |
| g_span | [BS·H] | fp32 | g gather 链 flat span（L2 命中，见 §5.5） |
| off32 / offu32 | [BS] | int32/uint32 | gather 偏移链 |
| g_ub / gexp_ub | [BS] | fp32 | g / 门控系数 |
| ge2d | [BS, BV] | fp32 | 行门控 broadcast 域 |
| gkl_ub / gkl2d | [64] / [64, BV] | fp32 | gk 末行 slab / 行门控域 |

**C 侧（alloc_shared → L1 / alloc_fragment → L0C）**：

| Buffer | Shape | dtype | 层级 | 用途 |
|--------|-------|-------|------|------|
| w_l1 / k_l1 | [BS, 64] | fp16 | L1 | gemm 左操作数 slab（GM→L1 直读） |
| h16_l1 | [64, BV] | fp16 | L1 | relay 读入的 H slab |
| v16_l1 | [BS, BV] | fp16 | L1 | v_new16 relay 读入 |
| bv_frag | [BS, BV] | fp32 | L0C | w@H 累加器（16KB） |
| kh_frag | [64, BV] | fp32 | L0C | k^T@v_new slab 输出（16KB，与 bv_frag 顺序复用 32KB 大槽） |

**relay workspace（GM 地址空间，L2 常驻，wrapper `empty` 分配）**：

| workspace | Shape（L0: NSLOT=1） | dtype | 用途 |
|-----------|---------------------|-------|------|
| workspace_1 | [20, NSLOT, DK, BV] | fp16 | V→C：H16 slab（16KB/slot） |
| workspace_2 | [20, NSLOT, BS, BV] | fp32 | C→V：b_v（16KB/slot） |
| workspace_3 | [20, NSLOT, BS, BV] | fp16 | V→C：v_new16（8KB/slot） |
| workspace_4 | [20, NSLOT, DK, BV] | fp32 | C→V：kH slab（32KB/slot） |

fwd relay 合计 72KB/slot × 20 核 × NSLOT；NSLOT=1 → 1.44MB，NSLOT=9（Stage 3）→ 13MB，L2 192MB 常驻 ✓。bwd：workspace_1(dH16 16KB) + workspace_2(dv_part 16KB) + workspace_3(q̃16 8KB) + workspace_4(qdo+wdv2 合包 32KB) = 72KB/slot 同量级。

### 4.4 内存搬运路径

```
fwd 每 chunk（链内；C=Cube 核，V=Vector 核；┄┄ relay=GM 地址空间 L2 常驻）：
GM[w] ──MTE1(C)──→ L1[w_l1 fp16]
V: h_ub(fp32) ──两步cast──→ h16_ub(fp16) ┄┄relay_1┄┄→ L1[h16_l1]（C 侧读）
C: gemm_v0(w_l1, h16_l1) → L0C[bv_frag] ┄┄relay_2┄┄→ V: bv_ub(fp32)
GM[u] ──MTE2(V)──→ UB[u16→u_f32]；v_new = u − b_v（UB fp32）──cast──→ GM[v_new]（SAVE_NEW_VALUE）
GM[g]（flat span，L2 命中）──gather链──→ UB[g] ──exp/broadcast──→ v_new ⊟= 门控；h_ub ⊟= exp(g_last)
GM[gk 末行] ──MTE2──→ UB[gkl] ──exp/broadcast──→ h_ub slab ⊟= 门控
V: v_new ──cast──→ vn16_ub ┄┄relay_3┄┄→ L1[v16_l1]
GM[k] ──MTE1(C)──→ L1[k_l1]；C: gemm_v0(k_l1, v16_l1, ᵀA) → L0C[kh_frag] ┄┄relay_4┄┄→ V: kh_ub ──add──→ h_ub
V: h_ub slab ──cast──→ GM[h_out]（每 chunk 快照）
```

> 层级合法性：无 GM→L0 直搬；Cube 输入经 L1（gemm_v0 内部 L1→L0A/L0B）；Vector 计算全在 UB；C→V / V→C 交换经显式 relay（AUTO_CV_SYNC 跨核 flag）；跨 dtype 转换全部 UB 内两步链（GM↔UB cast-on-copy 禁用）。

### 4.5 片上内存预算表（910B3：L1=512KB、UB=192KB、L0A/L0B=64KB、L0C=128KB per core）

**fwd（fp16 域，BV=64，DK=128，NSLAB=2）**

| 层级 | 构成 | 大小 | 限额 | 结论 |
|------|------|------|------|------|
| L0C | bv_frag 16KB + kh_frag 16KB（顺序复用，峰值 32KB；并列双槽上限 48KB） | **48KB** | 128KB | ✓ 38% |
| L1 | w_l1 8KB×2slab + k_l1 8KB×2 + h16_l1 16KB + v16_l1 8KB + relay 读入双缓冲 | **~80KB** | 512KB | ✓ 16% |
| UB 峰值（V 侧） | h_ub 32（常驻）+ h16 8 + bv 16 + kh 16 + u/vn 域 16（复用槽）+ g 链 16.5 + ge2d 16 + gkl2d 16 + gather span 16 | **~137KB**（lifetime 复用后；无复用粗算 ~169KB） | 192KB | ✓ 71%（**紧张**，依赖 MEMORY_PLANNING lifetime 分析；降级预案 BV=32 → ~90KB） |

**bwd（bf16 域，BV=64，DK=128）**

| 层级 | 构成 | 大小 | 限额 | 结论 |
|------|------|------|------|------|
| L0C | dv_part 16 + qdo 16 + wdv2 16（qdo/wdv2 并行双槽 32KB，dv_part 顺序前置 16KB → 峰值 48KB） | **48KB** | 128KB | ✓ 38% |
| L1 | k/q/w_l1 各 8×2 + dH16/do16/dv2_16/q̃16 relay 读入 ~16×4 | **~112KB** | 512KB | ✓ 22% |
| UB 峰值（V 侧） | dH 32（常驻）+ dv_part 16 + do/dv 域 16（复用）+ q̃ 域 16 + 门控 ge2d 16 + gkl2d 16（slab）+ g 链 16.5 + qdo/wdv2 读回 32 | **~150KB** | 192KB | ✓ 78%（**紧张**，同上依赖 lifetime 复用 + BV=32 降级预案） |

### 4.6 动态轴定义

无运行时动态轴（S/DK/DV/H/B 均为 JIT 编译期常量，随 shape 重编译）。NT=S//64、NSLAB=DK//64、NV=DV//BV、NCHAIN=B·H·NV 编译期展开。

### 4.7 JIT 配置

```python
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 C/V 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # 跨核 flag（relay 按名匹配）
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 核内同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # UB/L1 lifetime 规划（§4.5 紧张预算关键）
}
@tilelang.jit(out_idx=[...], workspace_idx=[...], pass_configs=PASS_CONFIGS)
```

> workspace_idx 为 Developer 复杂场景回退写法（design-template §8.2），指向 §4.3 relay 表的签名位置（绝对索引，fla_wy_fast #7：auto 注入参数会平移负索引）。

---

## 5. Tiling 策略

### 5.1 计算类型

**混合（CV 融合）**：Cube（gemm_v0，fwd 2 组/bwd 3 组 per chunk）+ Vector（exp 门控、broadcast、cast、add/sub）。

### 5.2 Block 划分

```python
BS   = 64    # chunk 长度（语义约束，= chunk_size，assert chunk_size == 64）
BV   = 64    # V 维块（主选）；候选 128（Stage 3：gemm N 翻倍 + 链数减半，UB 常驻 64KB 需核验）
NSLAB = DK // 64   # K 维 64 行 slab（DK=128 → 2；支持域 DK ≤ 256）
NV   = DV // BV    # benchmark 128 → 2
NCHAIN = B * H * NV   # benchmark: 1*64*2 = 128 条递推链
CORE_EFF = 20        # 910B3 Cube 核数；每核 ceil(128/20)=7 条链（striped）
```

**BV=64 / gemm 合并选型理由**：① 分形 N≥16 ✓；② UB 常驻 H 32KB + 峰值 ~137KB 可行（BV=128 时 H 64KB + 峰值 >190KB 超限，仅 Stage 3 以显式复用重核）；③ **gemm 合并**——w@H 的 2 个 slab 合并为单 gemm（M=64, K=128, N=64，L0A [64,128] fp16 16KB + L0B [128,64] 16KB ≤ 64KB ✓），k^T@v_new 输出 [128, 64] 单 gemm（M=128, K=64），每 chunk gemm 数 fwd=2 / bwd=3（vs 上游 slab 展开 fwd=4 / bwd=6），**GEMM 调用次数减半是串行链延迟的直接的削减**；④ DK=64（NSLAB=1）时退化为 [64,64,64] 单 gemm，DK=256 时 4 slab 两两合并 ×2 组——同一 slab 循环结构覆盖。

### 5.3 约束分析

- 对齐：fp16 尾轴 64×2B=128B ≥ 32B ✓；h 写出行 gap = 2·DV = 256B ≥ 32B（MTE3 行间 gap 合法）✓
- L0C：峰值 48KB ≤ 128KB ✓（§4.5）
- GEMM 非整除：S%64==0（支持域）；DK=64 倍数、DV=BV 倍数（L0 支持域，benchmark 128 ✓）

### 5.4 注意事项（尾块）

S 维无尾块（支持域 S%64==0，wrapper `assert S % chunk_size == 0`）。D 维：DK 为 64 倍数（L0 域）；非 64 倍数的 DK（如 96 = 64+32 尾 slab）与非 BV 倍数的 DV 不在 L0，Stage 2 扩展 L1 时用 `T.ceildiv` + 输入/输出两侧 valid extent 切片（参考 ascend-constraints.md §2 非整除行与 `examples/grouped_gemm/example_grouped_gemm_fwd.py:107-108` 模式），禁止 host padding。非 16 倍数 D 违反分形限制，属不支持域（proto 已声明）。

### 5.5 数据搬运性能可行性（REORDER-COST）

本算子无数据重排（bshd 连续直读 + 3D 零拷贝 view），全部搬运为规则 tile 块 DMA；唯二非平凡访问：

```text
[REORDER-COST]
path: fwd 主路径 (BV=64, relay NSLOT=1, bench B=1,S=8192,H=64,DK=DV=128)
gm_passes: 每张量 1 读 + 1 写（HBM 唯一 IO 815MB @bench；relay 走 L2 不占 HBM——
  20核×72KB×NSLOT ≤ 1.44MB L2 常驻，chunk_o 同款实证）
g_gather: [B,S,H] fp32 per-token stride-H 读 → flat span [64·H=16KB/chunk] + gather 抽 64；
  同 chunk 同 b 的 H 个 head 读重叠 span → L2 命中，HBM 实际 ~2.1MB（§9.5 已计入）
gk_末行: 每 (链, chunk) 一次 512B 连续读 = 8MB @bench（L2 部分命中）
dma_transactions: ~20 次/chunk/链（w/k/u GM 读 3 + h 写 2 slab + v_new 1 + relay 4×2 端 +
  g/gk 3）× 128 chunk × 128 链 ≈ 33 万次，平均 ~7KB/次
gm_scalar_accesses: 0（无标量 GM 访问；所有元素运算在 UB，buffer 级 T.tile.*）
address_div_mod_per_element: 0（task/chunk 级索引分解，每链每 chunk ~5 次）
active_cores / serial_tasks_per_core: 20 cube + 40 vector / 7 链·128 chunk 递推
largest_case_timeout_gate: fwd bench shape 单 case 60s（§9.4）
result: pass

path: bwd 主路径 (同形状 bf16)
gm_passes: 唯一 IO 1082MB @bench（q/k/w/do/dv 读 5×134MB + dh 268MB + dv2 134MB + g/gk ~10MB）
dma_transactions: ~26 次/chunk/链 × 128 × 128 ≈ 43 万次，平均 ~8KB/次
gm_scalar_accesses: 0
address_div_mod_per_element: 0
active_cores / serial_tasks_per_core: 20 + 40 / 7×128
largest_case_timeout_gate: bwd bench shape 单 case 120s（§9.4）
result: pass
```

---

## 6. 循环与调度结构

### 6.1 循环结构总结

**fwd / bwd 共用**（性能主选路径）：

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 递推链 (b, h, i_v) | **核内任务循环**（fixed-core + striped） | `T.serial(ntask)`，`tid = cid + k_task*CORE_EFF` | 128 链 / 20 核 → 每核 7 链交织，C/V 队列跨任务流动（chunk_o 实测：单 block 摊薄 dispatch/TPipe-init，MTE2/MTE3/V 队列不排空）；**结构性必需：GPU 靠 CTA 海量并行，Ascend 20 核必须核内交织** |
| chunk 递推 | **串行循环**（循环携带依赖 H） | `T.serial(NT)` | 跨 chunk 真依赖不可并行；**交换走 GM relay**（serial 内片上直连会 mis-sync——fla_wy_fast #1 实测） |
| K-slab | 串行子循环 | `T.serial(NSLAB)`（DK=128 时并入单 gemm，循环不展开） | gemm 合并（§5.2） |
| 元素级 | 向量化 | buffer 级 `T.tile.*` | element T.Parallel 实测 -26%~-71% 反模式（chunk_o #7） |

**回退路径（L0 冒烟对照，fla_wy_fast 形态）**：`T.Kernel(NCHAIN)` 每 block 一链 + `T.Pipelined(NT, num_stages=2)` 主循环 + 片上直连（UB↔L1 同名 shared buffer）。仅用于正确性对照与 relay 故障隔离（见 §10.1 R2 预案），非性能路径。

### 6.2 bwd 循环伪代码（关键差异部分）

```python
with T.Kernel(CORE_EFF, threads=2, is_npu=True) as (cid):
    # ... V/C 侧 buffer 声明（§4.5 bwd 表；dH_ub [DK, BV] fp32 常驻）...
    for k_task in T.serial(ntask):
        tid = cid + k_task * CORE_EFF
        # (USE_FINAL_STATE_GRADIENT: dht 载入 dH_ub；否则 fill 0)
        for i_t in T.serial(NT - 1, -1, -1):          # 反向递推
            # (a) 存 dh[t]：dH slab cast bf16 → GM
            # (b) V: dH16 slab ┄relay_1┄→ C: gemm_v0(k_l1, dH16_l1, dv_frag)（slab 累加）
            #     C: dv_frag ┄relay_2┄→ V: dv_part ⊟= exp(g_last−g) 行门控；
            #     dv2 = dv + dv_part → cast bf16 → GM
            # (c) V: q̃ = q ⊙ exp(g[t])（行门控，载入域）→ cast bf16 ┄relay_3┄→ C
            #     C: gemm_v0(q̃_l1, do_l1, qdo_frag, ᵀA)；gemm_v0(w_l1, dv2_16_l1, wdv2_frag, ᵀA)
            #     qdo/wdv2 合包 ┄relay_4┄→ V
            # (d) V: dH = dH⊙exp(g_last) + qdo·scale − wdv2（gk per-channel 门控分 slab 施加）
        # (USE_INITIAL_STATE) dh0: dH fp32 直写 GM
```

### 6.3 流水线优化（分级演进，Stage 2 → Stage 3）

| 阶段 | 结构 | 预期 | 依据 |
|------|------|------|------|
| L0（Stage 2 起步） | NSLOT=1 严格 lockstep（每 relay 双向握手，auto flag） | 正确性优先，无领先/落后竞态 | auto flag 语义 |
| Stage 3-a | NSLOT=9 slot ring + V 领先 2 task（粗节流） | relay flag 预置，等待坍缩为协议开销 | chunk_o:136-153（ring 安全条件 NSLOT > 2L+1） |
| Stage 3-b | 3-chunk 批处理 unroll + L0C ping-pong（o_frag_a/b）+ 手动 throttle flag（`T.set_cross_flag("FIX", id)` / `T.wait_cross_flag(id, "V")`，**user id ≥ 48**，auto pass 不追踪用户 id，碰撞静默损坏握手——chunk_o:82-86 实测） | 每 flag wait ~470ns × 每 chunk 4 次 → 批内摊薄；C/V 深度交织 | chunk_o 全套（3.285→1.417ms 实测路径） |

核内 Pipelined 与核间流水互斥（AUTO_CV_COMBINE 自动决策）；本算子主路径无核内 Pipelined（串行递推），核间交织由 relay ring + task loop 天然提供。

### 6.4 尾块处理

见 §5.4（S 维支持域整除；D 维非 64 倍数留给 L1 扩展 valid extent 方案）。

---

## 7. 同步策略

### 7.1 同步模式

**自动同步**（Developer 模式标准）：AUTO_SYNC 处理核内 MTE/计算依赖，AUTO_CV_SYNC 对 `workspace_*` 命名的 relay buffer 读写自动插跨核 flag（chunk_o 实证："Their V->C handoffs keep the auto-inserted cross-core flags"）。源码无 `T.barrier_all` / `T.set_flag`（Stage 3-b 引入的 throttle flag 除外，属用户显式跨核节流，非数据依赖同步）。

### 7.2 同步点说明（编译器自动插入，设计仅声明数据依赖）

| 依赖 | 说明 |
|------|------|
| UB 元素运算 ← GM 载入（MTE2） | AUTO_SYNC |
| gemm_v0 ← L1 输入就绪（MTE1 / relay 读入） | AUTO_SYNC（C 核内 barrier） |
| C 读 workspace_i ← V 写 workspace_i | AUTO_CV_SYNC（跨核 flag，按名匹配） |
| V 读 workspace_i ← C 写 workspace_i | AUTO_CV_SYNC |
| GM 写 ← UB 结果（MTE3） | AUTO_SYNC |
| relay slot 复用（NSLOT>1 时的迭代间 WAR） | Stage 3-b 手动 throttle flag 保证（chunk_o ring 论证）；NSLOT=1 lockstep 天然安全 |

### 7.3 pass_configs 配置

见 §4.7（四件套全开）。

---

## 8. CV 融合设计

### 8.1 融合算子判定

**是**（Cube GEMM + Vector exp 门控/broadcast/cast/add）。

### 8.2 CV 交互设计（Developer 复杂场景回退分支，按 design-template §8.2）

- `T.Kernel(CORE_EFF=20, threads=2, is_npu=True) as (cid)`：单轴 + threads=2，**无 vid 显式偏移**（vid-reduction 自动对半切 V 侧 binary ops；`T.tile.select` 类需 half-region 的例外本算子 L0 域不涉及——掩码恒真，无 select）
- **复杂场景回退**：显式 relay workspace 表（§4.3）+ `workspace_idx`（绝对索引）——依据：chunk_o 同款（"Developer mode, threads=2" + workspace_idx=[6,7,8,9]）；规避 AUTO_CV_COMBINE 自动注入 per-block workspace（~768MB/launch，chunk_o 注释实测记录）
- Cube↔Vector 交换全部经 4 个 relay（fwd：H16/b_v/v_new16/kH；bwd：dH16/dv_part/q̃16/qdo+wdv2 合包）

| workspace | Shape（NSLOT=1） | dtype | 用途 |
|-----------|-----------------|-------|------|
| workspace_1 | [20, 1, DK, BV] | 半精度 | V→C：H16/dH16/q̃16（fwd/bwd 各自 kernel 内不复用） |
| workspace_2 | [20, 1, BS, BV] | fp32 | C→V：b_v / dv_part |
| workspace_3 | [20, 1, BS, BV] | 半精度 | V→C：v_new16（fwd） |
| workspace_4 | [20, 1, DK, BV] | fp32 | C→V：kH / qdo+wdv2 合包 |

`workspace_idx`：按 fwd/bwd 各自 kernel 签名的 workspace 参数绝对位置确定（Stage 2 落地时对齐签名；fla_wy_fast #7：负索引会被 auto 注入参数平移）。

### 8.3 / 8.4 Cube 与 Vector 核计算流程

Cube 侧：GM→L1 直读 gemm 操作数（w/k/q 的 3D views，**不 vid 分裂**，chunk_o #8）；relay 读入（H16/v_new16/…）；gemm_v0（slab 合并/累加）；L0C 结果 → relay 写出。Vector 侧：GM→UB 载入（u/do/dv/g/gk）；gather 链；门控 exp/broadcast/mul；cast 两步链；h/dh/v_new/dv2 GM 写出（3D views）；状态 H/dH 常驻维护。两侧经 relay 解耦，由 task loop 各自推进 + auto flag 握手。

### 8.5 pass_configs

见 §4.7。

### 8.6 注意事项

- threads=2 是消 vid 前提、消 vid 是消（自动注入）workspace 前提；本设计显式 workspace 属"复杂场景回退"合法形态（模板 §8.2 分支），与 threads=2 共存（chunk_o 实证）
- **user 手动 flag id（Stage 3-b）必须 ≥ 48**：auto pass 分配 0..40 区间且不追踪用户 id，碰撞静默损坏（chunk_o:82-86 实测）
- `T.set_cross_flag("FIX", ...)` 经 pipe 参数归 Cube 核、`T.wait_cross_flag(..., "V")` 归 Vector 核（ResourceForPipe 仅接受 "V"/"M"/"MTE1"/"FIX"）
- UB 峰值 fwd 71% / bwd 78% 依赖 MEMORY_PLANNING lifetime 复用；编译报溢出时首选 BV=32，其次手动合并中转槽

---

## 9. 验证方案

### 9.1 Golden 函数（PyTorch fp64 重写 + 上游 cast 语义对齐，供 Stage 2 直接落地）

> **精度结构关键决策**：本算子状态 H 为递推中间量，上游 Triton 每 chunk 有 2 个 cast 点（H→半精度参与 w@H、v_new→半精度参与 k^T@v_new），128 chunk 累积后单 cast 的绝对误差量级 ~|w@H|·2⁻¹¹ ≈ 5e-2（|w@H|~O(10²) 时），**远超 fp16 档 atol+rtol·|v|（~2e-3）**——纯 fp64 golden 与任何忠实复刻上游 cast 的 kernel 对拍都必然大面积超差。故 golden 采用**fp64 主域 + 逐点模拟上游 cast**（同 cast 点同精度），kernel 侧实现差异（exp·ln2、求和顺序、slab 合并）计入混合容差。此口径与家族先例一致（fla_wy_fast golden 用 torch.exp2 精确语义、实现差入容差）。若 Stage 2 仍见超差，启用 hi/lo 拆分增强（chunk_o 实测模式，误差 → 2⁻²²，代价 gemm 数翻倍，见 R4）。

```python
import torch

LN2 = 0.6931471805599453


def _exp(g, use_exp2):
    """门控指数：use_exp2 时 g 为 log2 域（2^g），否则 ln 域（e^g）。"""
    return torch.exp2(g) if use_exp2 else torch.exp(g)


def fla_chunk_delta_fwd_ref(k, w, u, g, gk, initial_state=None, output_final_state=False,
                            chunk_size=64, save_new_value=True, use_exp2=False):
    """fwd golden。k/w [B,S,H,DK] fp16, u [B,S,H,DV] fp16, g [B,S,H] fp32,
    gk [B,S,H,DK] fp32（均为 chunk 内累积 log/log2 衰减域）。
    返回 (h [B,NT,H,DK,DV] fp16, v_new fp16|None, ht fp32|None)。"""
    B, S, H, DK = k.shape
    DV = u.shape[-1]
    C = chunk_size
    assert S % C == 0
    NT = S // C
    kd = k.dtype
    kf, wf, uf = k.double(), w.double(), u.double()
    gf, gkf = g.double(), gk.double()
    Hd = torch.zeros(B, H, DK, DV, dtype=torch.float64)
    if initial_state is not None:
        Hd += initial_state.double()
    h_out = torch.empty(B, NT, H, DK, DV, dtype=kd)
    v_new = torch.empty(B, S, H, DV, dtype=kd) if save_new_value else None
    for i in range(NT):
        sl = slice(i * C, (i + 1) * C)
        n_t = (i + 1) * C - 1
        h_out[:, i] = Hd.to(kd)                          # 快照（cast 模拟点 1：h 存储）
        # v_new = u − w @ H16（cast 模拟点 2：H 半精度参与 gemm）
        H16 = Hd.to(kd).double()
        b_v = torch.einsum("bthd,bhdv->bthv", wf[:, sl], H16)
        vn = uf[:, sl] - b_v
        if save_new_value:
            v_new[:, sl] = vn.to(kd)
        # USE_G 门控（S%64==0 域 mask 恒真）
        g_last = gf[:, n_t, :]                            # [B, H]
        gate = _exp(g_last.unsqueeze(1) - gf[:, sl], use_exp2)   # [B, C, H]
        vn = vn * gate.unsqueeze(-1)
        Hd = Hd * _exp(g_last, use_exp2).unsqueeze(-1).unsqueeze(-1)
        # USE_GK per-channel 门控
        gk_last = gkf[:, n_t, :, :]                       # [B, H, DK]
        Hd = Hd * _exp(gk_last, use_exp2).unsqueeze(-1)
        # H += k^T @ v_new16（cast 模拟点 3：v_new 半精度参与 gemm）
        vn16 = vn.to(kd).double()
        Hd = Hd + torch.einsum("bthd,bthv->bhdv", kf[:, sl], vn16)
    ht = Hd.float() if output_final_state else None         # STORE_FINAL_STATE
    return h_out, v_new, ht


def fla_chunk_delta_bwd_ref(q, k, w, do, dv, g, gk, h0=None, dht=None,
                            scale=None, chunk_size=64, use_exp2=False):
    """bwd golden。q/k/w [B,S,H,DK] bf16, do/dv [B,S,H,DV] bf16, g/gk fp32。
    返回 (dh bf16, dh0 fp32|None, dv2 bf16)。scale 默认 DK**-0.5。"""
    B, S, H, DK = q.shape
    DV = do.shape[-1]
    C = chunk_size
    assert S % C == 0
    NT = S // C
    if scale is None:
        scale = DK ** -0.5
    qd = q.double()
    kd = k.double()
    wd = w.double()
    dod, dvd = do.double(), dv.double()
    gf, gkf = g.double(), gk.double()
    dH = torch.zeros(B, H, DK, DV, dtype=torch.float64)
    if dht is not None:
        dH += dht.double()
    dh_out = torch.empty(B, NT, H, DK, DV, dtype=q.dtype)
    dv2 = torch.empty(B, S, H, DV, dtype=dv.dtype)
    for i in range(NT - 1, -1, -1):
        sl = slice(i * C, (i + 1) * C)
        n_t = (i + 1) * C - 1
        dh_out[:, i] = dH.to(q.dtype)                     # 快照
        g_last = gf[:, n_t, :]
        gate = _exp(g_last.unsqueeze(1) - gf[:, sl], use_exp2)
        # dv_part = (k @ dH16) ⊙ gate（cast 模拟点：dH 半精度参与 gemm）
        dH16 = dH.to(q.dtype).double()
        dv_part = torch.einsum("bthd,bhdv->bthv", kd[:, sl], dH16)
        dv_part = dv_part * gate.unsqueeze(-1)
        dv2[:, sl] = (dvd[:, sl] + dv_part).to(dv.dtype)
        # dH 更新
        dH = dH * _exp(g_last, use_exp2).unsqueeze(-1).unsqueeze(-1)
        qt = (qd[:, sl] * _exp(gf[:, sl], use_exp2).unsqueeze(-1)).to(q.dtype).double()
        qdo = torch.einsum("bthd,bthv->bhdv", qt, dod[:, sl])
        wdv2 = torch.einsum("bthd,bthv->bhdv", wd[:, sl], dv2[:, sl].double())
        dH = dH + qdo * scale - wdv2
        gk_last = gkf[:, n_t, :, :]
        dH = dH * _exp(gk_last, use_exp2).unsqueeze(-1)
    dh0 = dH.float() if h0 is not None else None            # USE_INITIAL_STATE：仅 h0 给定时输出
    return dh_out, dh0, dv2
```

> 注：`ht` 由 `output_final_state` 控制、`dh0` 由 `h0 is not None` 控制（对齐上游 wrapper：`final_state = ... if output_final_state else None` / `dh0 = torch.empty_like(h0, ...) if h0 is not None else None`）。**输入构造**（对齐 KDA 家族惯例 + 递推数值安全）：k/w/u(/q/do/dv) `torch.randn` CPU → cast 目标 dtype；**g/gk 必须构造为 chunk 内累积衰减**（`raw = F.logsigmoid(torch.randn(...))` → reshape [.., NT, C] → `cumsum(-1)` → 还原；gk 同款 per-channel）——保证 exp(g_last−g)∈(0,1]、跨 128 chunk 状态连乘有界、fp16 输出域不溢出（randn 直用会使 H 指数漂移 e^{√t}，t=128 时溢出 fp16——gdn_chunk_h.py:194-196 家族构造先例）。

### 9.2 L0 门槛测试计划（tilelang-op-test-design 场景 A，仅 L0）

> 判断结论：计算类型=**混合（CV 融合）**（gemm_v0 + exp 门控/broadcast/cast）；复杂度=**Fusion**（跨 chunk 串行递推 + 双 kernel）；数学特征=GEMM+gating（线性注意力/RNN chunk 递推家族）；综合类别=**Fusion**。Stage 1 只产出 L0（规则 shape：S%64==0、DK/DV 为 64 倍数、block 整除）。L1（功能，含非 64 倍数 D/尾块）/L2（异常）/Boundary（特殊值）由 Stage 2 在 L0 通过后按场景 B 扩展。

| 用例名 | 级别 | Shape (B,S,H,DK,DV) | dtype / 分支 | 说明 |
|--------|------|---------------------|--------------|------|
| l0_fwd_min | L0 | (1, 128, 2, 64, 64) | fwd fp16 域；g+gk 开；默认分支（无 h0/ht、save_new_value=True、use_exp2=False） | 最小冒烟：NT=2（≥2 迭代触发 task/递推循环多轮）、NCHAIN=2、NSLAB=1；覆盖 fwd 全数据流（h 快照、v_new、双门控、k^T@v_new、状态递推跨 chunk 传递） |
| l0_fwd_repr | L0 | (2, 256, 4, 128, 128) | 同上 | benchmark 结构代表：NSLAB=2（slab 合并 gemm 路径）、NV=2（BV 切分）、NCHAIN=16、多 batch/head |
| l0_fwd_opts | L0 | (1, 256, 2, 128, 128) | fwd fp16 域 + initial_state + output_final_state + save_new_value=False + **use_exp2=True** | constexpr 分支全覆盖：h0 载入、ht 直写、SAVE_NEW_VALUE 关闭（v_new=None）、exp2 域（g 用 log2 域构造） |
| l0_bwd_min | L0 | (1, 128, 2, 64, 64) | bwd bf16 域；g+gk 开；默认（无 dht、h0=None → dh0 不输出、use_exp2=False） | 最小冒烟：反向递推、3 组 gemm、dv2 输出（对齐上游：dh0 仅 h0 给定时产出） |
| l0_bwd_repr | L0 | (2, 256, 4, 128, 128) | 同上 | benchmark 结构代表：NSLAB=2、NV=2、scale=DK**-0.5 |
| l0_bwd_opts | L0 | (1, 256, 2, 128, 128) | bwd bf16 域 + dht + **h0**（→ dh0 输出）+ **use_exp2=True** + scale=0.37 | constexpr 分支：USE_FINAL_STATE_GRADIENT 载入、USE_INITIAL_STATE（dh0 fp32 输出）、exp2 域、非默认 scale |

**golden 对照法**：每用例对每个实际产出输出张量独立调用 `check_precision(out, ref, dtype)`（precision-standard.md §4.1 混合容差判定，整段复制不改动）；fwd 判 h/v_new（fp16 档）+ ht（fp32 档，仅 l0_fwd_opts），bwd 判 dh/dv2（bf16 档）+ dh0（fp32 档，仅 l0_bwd_opts）。全部通过打 `[PRECISION_PASS]`，任一失败 `[PRECISION_FAIL]`。输入构造统一走 §9.1 末段的家族惯例（g/gk = chunk 内 cumsum(logsigmoid(randn))；use_exp2 用例构造 log2 域：`g2 = g_ln / LN2`）。

### 9.3 精度标准（混合容差，precision-standard.md 口径）

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|-------|------|------|---------------------|------------------------|
| float16（fwd 输出 h / v_new） | 2⁻¹⁴ (6.10e-5) | 2⁻⁹ (1.95e-3) | 1e-1 | 0.99 |
| bfloat16（bwd 输出 dh / dv2） | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |
| float32（ht / dh0） | 2⁻¹⁶ (1.53e-5) | 2⁻¹⁰ (9.77e-4) | 1e-2 | 0.99 |

> 与 proto_fla_chunk_delta.yaml 声明的门禁逐字一致（fwd fp16 档 / bwd bf16 档 / fp32 档）。inputs dtype 全集 = {float16, bfloat16, float32}，与上表三行一致（g/gk/h0/dht 输入 fp32 为中间量，其精度由输出档间接约束，无独立精度行）。特殊值（NaN/Inf）位置契约：本算子输入域为 randn + 衰减门控有限值（上游对拍口径），Boundary 特殊值用例由 Stage 2 场景 B 扩展（g 大负值 → exp 下溢 0 属合法行为，含稀疏 inf 的混合输入按 mask 结构比对）。

### 9.4 性能可行性哨兵（强制执行，不可跳过）

| 用例名 | Shape (B,S,H,DK,DV) | dtype/属性 | 覆盖路径 | 单 case timeout | 选择理由 |
|--------|---------------------|------------|----------|-----------------|----------|
| perf_fwd_bench | (1, 8192, 64, 128, 128) | fwd fp16（k/w/u fp16、g/gk fp32）、g+gk 开、wrapper 默认（无 h0/ht、save_new_value=True、use_exp2=False） | fwd 主路径：NCHAIN=128 链、每核 7 链 × NT=128 chunk 递推、2 gemm/chunk | 60s | **benchmark 验收 shape**；最大任务数 + HBM 最大吞吐压力（815MB） |
| perf_bwd_bench | (1, 8192, 64, 128, 128) | bwd bf16（q/k/w/do/dv bf16、g/gk fp32）、g+gk 开、scale=DK**-0.5、use_exp2=False | bwd 主路径：反向递推、3 gemm/chunk | 120s | **benchmark 验收 shape**；bwd 3 组 gemm + 门控联合压力（1082MB） |

计时口径：do_bench（对齐 `ref_gpu/test_utils_kda.py`：warmup=20、rep=10、取均值，仅设备侧 kernel 执行时间，不含张量构造/H2D-D2H/JIT/框架开销）；torch_npu 侧 `torch.npu.Event(enable_timing=True)` 等价实现（chunk_o/家族先例口径）。验收判据：**fwd + bwd 两 kernel 纯 Kernel 时间合计 < 3400us**（精度全部通过为前置条件）。测试数据、随机数与 golden 均在 CPU 完成后单次 H2D，运行段无 aclnn 依赖。

### 9.5 性能预算分析（910B3，含 3400us 目标可行性结论）

**IO 明细（HBM 物理唯一字节，每张量单读单写；bench shape B=1,S=8192,H=64,DK=DV=128,cs=64）**

| 项 | fwd（fp16 域） | bwd（bf16 域） |
|----|----------------|----------------|
| 读 | k/w/u 各 134.2MB + g 2.1MB（span 读经 L2 命中收敛，见 §5.5）+ gk 末行稀疏 8MB = **547MB** | q/k/w/do/dv 各 134.2MB + g 2.1 + gk 8 = **682MB** |
| 写 | h 268.4MB + v_new 134.2MB = **402.6MB** | dh 268.4 + dv2 134.2 = **402.6MB** |
| 合计 | **~950MB**（orchestrator 口径 804MB 未计 g/gk 稀疏读，+10MB 级修正） | **~1085MB**（口径 1072MB 同上修正） |
| GEMM FLOPs | 34.4 GFLOP（128 链 × 128 chunk × 2.1 MFLOP） | 51.5 GFLOP（× 3.15 MFLOP） |

**Roofline 对照（910B3 实测锚点）**

| 项 | 数值 | 依据 |
|----|------|------|
| HBM 实测带宽上限 | **1263 GB/s** | 【已确认事实】同机 fla_wy_fast Stage 3 画像实测（debug_log_fla_wy_fast.md:10）；npu-arch 标称 1.6TB/s（910B 族） |
| 带宽下限 | fwd 950MB/1263GB/s = **752us**；bwd 1085MB = **859us**；合计 **~1.61ms**（100% 实测带宽） | 带宽口径 |
| 目标隐含带宽 | 2.04GB / 3400us = **600GB/s** = 实测上限的 **47.5%** | ✓ 带宽余量充足（vs fla_wy_fast 当年需 4.6× 超带宽的物理不可达） |
| GEMM 纯算 | 85.9 GFLOP / 294T（20 核 fp16 稠密折算，4096MAC×20×1.8GHz×2）= **292us** | 计算非带宽瓶颈，但见下行 |
| **串行链延迟口径（真瓶颈）** | 每 chunk 关键路径 = 2-3 个小 gemm + 4 次 relay 交换 + V 门控链。**外推锚点：chunk_o 同形状量级实测 1.417ms 含 ~1229 gemm/核 → ~1.15us/gemm（端到端折算）**；本算子 fwd 1638 gemm/核（128 链/20 核 × 128 × 2）、bwd 2458 gemm/核 → 合计 4.7ms（悲观上界）；纯 gemm 背靠背（~0.4us）→ 1.64ms（乐观下界） | 【推断】chunk_o proxy + 理论延迟；**目标 3.4ms 落在 [1.64, 4.71]ms 区间内** |

**结论：物理可达（非 fla_wy_fast 式倍数级矛盾），但处于边缘区间，达标依赖三条已验证的结构手段**：

1. **gemm 合并**（本设计 §5.2，已内置）：每 chunk gemm 数 fwd 4→2 / bwd 6→3（vs 上游 slab 展开）；
2. **核内多链交织**（本设计 §6.1 性能主选，已内置）：fixed-core task loop 让 7 链的 C 工作/V 工作/搬运互相填充——chunk_o 实证同机制将同族算子从 3.285ms 压到 1.417ms；
3. **Stage 3 relay ring + 3-chunk 批处理 + throttle flag**（§6.3 演进路径）：flag 等待坍缩 + L0C ping-pong，chunk_o 实测 -57% 路径。

**量化判据**：达标 3400us 需有效吞吐 ~0.83us/gemm（fwd+bwd 合计 4096 gemm/核），介于乐观 0.4 与 chunk_o 端到端折算 1.15 之间——**Stage 2 末（NSLOT=1 lockstep）预计落在 4-6ms 区间不达标属预期，Stage 3 按上述 3 手段优化后收敛至 2.5-3.4ms**。若 Stage 3 穷尽后仍 >3400us，按迁移 Phase 1 裁定中断上报（附 msprof 画像证据），不静默简化。

### 9.6 Host 侧 Buffer 操作审计

```text
[HOST-METADATA-AUDIT]
operation: 输入构造（CPU randn/logsigmoid/cumsum → .npu() 单次 H2D，数据准备）+
  输出分配（k.new_empty：数据准备）+ relay workspace 分配（torch.empty：数据准备）+
  bshd→3D 零拷贝 view（reshape [B, S, H*D] / [B, S*H]，contiguous 张量的纯 metadata view）+
  kernel 调用 + check_precision 验证（D2H 后 CPU 比较）
input_stride -> output_stride: 3D view 仅改 shape/stride 元数据（contiguous 输入的兼容 reshape，
  零拷贝成立——chunk_o #8 同款先例）；不触碰数据指针
shares_storage / same_data_ptr: true（view 与原张量共享 storage）
aclnn_or_physical_copy: false（无 .contiguous()/pad/cat/.to()/clone/F.* 计算；
  randn/logsigmoid/cumsum/empty 均为 CPU 侧数据生成，kernel 前后无 NPU 张量内容改写）
result: allow
```

---

## 10. 风险点与注意事项

### 10.1 风险清单（高/中/低分级）

| 级别 | 风险 | 影响 | 缓解/验证 |
|------|------|------|-----------|
| **高** | R1 性能目标 3400us 处于边缘区间（§9.5：带宽口径 47.5% 余量 ✓，但串行链延迟口径 [1.64, 4.71]ms 跨越目标） | 验收 | §6.3 三级演进（gemm 合并已内置 → task loop 交织已内置 → Stage 3 ring/批处理/throttle）；Stage 3 穷尽仍超 → 按裁定中断上报（附画像）；**不构成 Stage 1 物理不可达中断** |
| **高** | R2 串行递推循环（T.serial + 循环携带 H）与 AUTO_CV_SYNC/auto flag 组合的正确性——gdn_chunk_h 为手动 flag 先例、chunk_o 为 auto flag + relay 先例，但**本算子的 4 交换/chunk × NT 迭代密度高于两者** | 正确性/死锁 | L0 从 NSLOT=1 严格 lockstep 起步（每 relay 双向握手，无领先竞态）；l0_fwd_min 首先验证；回退链：①回退路径（每 block 一链 + Pipelined 片上直连，fla_wy_fast 已验证形态）做正确性对照 → ②Expert 手动 flag（gdn_chunk_h 直译，需 orchestrator 批准模式豁免） |
| **高** | R3 UB 峰值 fwd 71% / bwd 78%（192KB），依赖 MEMORY_PLANNING lifetime 复用 | 编译溢出 | 降级预案 BV=32（[·,BV] buffer 减半 → ~90KB）；其次手动合并 u/vn/bv 中转槽（时序错开）；h16/kh 按 slab 单槽 |
| **高** | R4 状态 cast 精度：H16 单 cast 绝对误差 ~\|w@H\|·2⁻¹¹，v_new 抵消结构（u − w@H）下相对误差放大 | L0 精度 | golden 已对齐上游 cast 语义（§9.1，cast 模拟点 1-3）；若仍有超差（exp·ln2 差异/求和顺序）：**hi/lo 拆分**（chunk_o 实测：H16 = hi + lo 两次 gemm 累加，误差 2⁻²²；代价 fwd gemm 4/chunk，先精度后性能） |
| 中 | R5 relay 经 GM 地址空间的 L2 常驻假设（20 核 × 72KB × NSLOT） | 性能 | NSLOT=1 → 1.44MB、NSLOT=9 → 13MB ≪ L2 192MB（chunk_o 同款实证）；Stage 3 用 msprof 复核 L2 命中率 |
| 中 | R6 task loop 尾核不均（128 链 / 20 核 → 7.2/核，18 核处理 7 链、2 核 6 链？striped 下 128=6×20+8 → 8 核 7 链、12 核 6 链） | 性能微小 | 负载差 ≤1 链（~7%）；Stage 3 可按 NCHAIN 精调 CORE_EFF（如 16 核 × 8 链） |
| 中 | R7 `T.tile.gather` 链在 threads=2 vid 分裂下的行为（fla_wy_fast #4 记录 gather 链在 threads=2 下断过） | 正确性 | g 的 gather 放 C 侧消费前完成（fla_wy_fast fwd_u 的 gather 在循环外 threads=1 形态验证）；Stage 2 若 threads=2 下异常：g 改为 V 侧直接 strided span 读 + UB 内 gather（或每 chunk 的 g 预排由 wrapper 3D view 规避——仅 metadata，合规） |
| 中 | R8 gemm_v0 transpose_A=True 与 [BS,64] slab 输入组合在本算子的 M=128 形态（k^T@v_new 输出 [DK, BV]） | 正确性 | gdn_chunk_h:73 transpose_A 先例（M=DK 形态一致）；l0_fwd_min 首验；异常时改 Vector 侧 T.tile.transpose（fp16 硬件路径）+ 无转置 gemm |
| 低 | R9 NT=128 串行循环的编译期体积（T.serial 不展开，无风险） | 无 | serial 运行时循环 |
| 低 | R10 h 写出行 gap 模式的 MTE3 效率（gap=256B/行） | 性能微小 | DataCopyPad 行间 gap 原生支持；chunk_o 的 v relay 注释实证行 gap 读 ~48GB/s（写侧同量级），非瓶颈 |

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| gemm_v0 传 fp32 A/B | 照抄上游 `tl.dot(b_w, b_h.to(...))` 时保留 fp32 | 交错垃圾数据（chunk_o/fla_wy_fast 双实测） | 全部操作数半精度；fp32 结果只在 L0C/UB |
| GM↔UB 跨 dtype 直 cast | 省略两步链 | codegen 模板失配损坏 | UB 内两步链（本设计 §4.4 已内置） |
| element T.Parallel 元素循环 | 照抄 gdn_chunk_h 的 `for i,j in T.Parallel(...)` | -26%~-71%/kernel | buffer 级 T.tile.*（本设计已内置） |
| serial 循环内片上 V→C 直连 | 跳过 relay 直接 UB↔L1 | mis-sync（fla_wy_fast #1） | 主路径全 relay；片上直连仅限回退路径 Pipelined body |
| threads=4 / num_warps 照抄 | 上游 autotune 配置 | 编译错误 | threads=2 |
| host 侧对 g 做 exp/cumsum 预处理 | 图省事 | aclnn 违规 + IO 翻倍 | kernel 内 gather 链 + exp（被否方案 E） |
| 手动 flag id < 48 | Stage 3 加 throttle | 与 auto flag 碰撞静默损坏 | user id ≥ 48（chunk_o:82-86） |
| golden 纯 fp64 不模拟 cast | 直接对拍 | 大面积误判超差（§9.1 分析） | golden 带 cast 模拟点 |

### 10.3 特殊场景处理

- **DK 非 64 倍数**（如 96 = 64+32）：不在 L0；Stage 2 L1 扩展用 `T.ceildiv(DK, 64)` slab 循环 + 尾 slab valid extent 切片（gemm 维度仍 64，输入/输出两侧 `T.copy(A[..., d0:d0+valid])` 裁剪；参考 grouped_gemm 模式）。非 16 倍数 D 违反分形，不支持域。
- **极小 shape**（S=64, NT=1）：递推 1 轮，正确；L0 不含（S=128 保证 NT≥2）。
- **S%64≠0**：不支持域（wrapper assert 拒绝，L2 负向用例覆盖）。
- **g 极端值**（exp 上溢/下溢）：g 为 log 衰减域（构造保证 ≤0 累积），exp(g_last−g)∈(0,1]；下溢 0 为合法衰减极限；Boundary 由 Stage 2 扩展（稀疏 inf + mask 结构比对）。
- **USE_EXP2 域混用**：g 的底数域由调用方保证（log2 vs ln），kernel 不做域转换；L0 的 l0_fwd_opts/l0_bwd_opts 用 log2 域构造覆盖。

### 10.4 被否方案（设计决策记录）

| # | 方案 | 否决理由 |
|---|------|---------|
| A | 状态 H 驻 L0C fragment（寄存器域对应物） | ① V 侧门控 H⊙exp 必须在 UB，L0C 无 Vector 通路；② UB→L0C 无通路（fragment 仅 gemm 输出语义，T.copy 方向仅 L0C→UB）；③ 每 chunk H 需 L0C↔UB 往返 2 次 + cast，无净收益。状态驻 UB 是 gdn_chunk_h/fused_sigmoid_gating_delta_rule 双先例的既定结构 |
| B | 每 block 一链 + Pipelined 片上直连（fla_wy_fast 纯 Developer 形态）作性能主选 | ① serial→Pipelined 后**循环携带依赖（H）的跨迭代 V→C 交换版本管理无先例**（fla_wy_fast 无 carry 场景）；② 无核内多链交织——每核 6-7 个 block 串行 dispatch，队列排空开销 + 串行链延迟无法互相填充，§9.5 延迟口径估算 4-8ms 不达标。降级为 L0 冒烟对照路径（回退） |
| C | Expert 手动 flag 全套（gdn_chunk_h 直译） | 用户指定 Developer；手动 set/wait_cross_flag 编排复杂度高且 chunk_o 已证明 Developer API + 显式 relay + auto flag 可达同等结构；T.Scope/alloc_L1 显式层级与 Developer 裁定冲突 |
| D | 按 chunk 段拆多 kernel（fla_wy_fast 拆分策略） | 跨 chunk 真依赖（H_{t+1} = f(H_t)）无法拆——段间必须经 GM 传递 H（IO 增加 + launch 开销 ~13us×段数），且段内仍是串行递推 + CV 交换，mis-sync/交织问题不随拆分消失 |
| E | host 侧预计算 exp(g)/gk 末行抽取 | 违反 host Buffer 约束（aclnn 调用 + 数据物化）；且 gk 全量 268MB 读+写翻倍；kernel 内 gather 链 + 末行稀疏读（8MB）才是最优 |
| F | BV=128（gemm N 翻倍 + 链数减半）作 L0 主选 | UB 常驻 H [128,128] fp32 = 64KB，叠加中转峰值 >190KB 超限；列为 Stage 3 调优候选（需显式 buffer 复用重核） |
| G | 上游 mask（m_t 越界行置 0）逐 chunk 现算 | 支持域 S%64==0 下恒真，死代码；扩展域时用 bit-packed select（chunk_o #6：乘 0/1 掩码无法清非有限值，Inf·0=NaN） |

---

## 11. 交付清单

### 11.1 目录结构

```
examples_experiment/dsa_sparse_finetune/     # 共享沙箱（多算子混居，一律带算子前缀）
├── DESIGN_fla_chunk_delta.md     # 本设计文档（含 L0 门槛测试计划）
├── proto_fla_chunk_delta.yaml    # 算子接口规格（refine 而非重建，见 §11.2）
├── ref_gpu/                      # 冻结参考（只读，禁止修改）
├── fla_chunk_delta.py            # ⬜ 待实现（Stage 2）：双 kernel（@tilelang.jit）
└── test_fla_chunk_delta.py       # ⬜ 待实现（Stage 2）：golden（§9.1）+ L0（§9.2）+ main 分发器
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN_fla_chunk_delta.md` | ✅ 已完成 | 本文档 |
| `proto_fla_chunk_delta.yaml` | ✅ 已 refine | proto.yaml 角色：`operator.inputs[].dtype` 全集 = {float16, bfloat16, float32}（与 §9.3 精度表一致）、attrs 补齐 save_new_value / output_final_state / use_initial_state（constexpr 分支全覆盖的派生维度），满足覆盖门禁 `coverage_check.py --proto` 派生要求 |
| `fla_chunk_delta.py` | ⬜ 待实现 | `fla_chunk_delta_fwd` / `fla_chunk_delta_bwd`（Developer 模式 + relay workspace，§4.7 pass_configs） |
| `test_fla_chunk_delta.py` | ⬜ 待实现 | golden 两函数（§9.1 逐行落地，含 cast 模拟点）+ `check_precision`（precision-standard.md §4.1 原样复制）+ L0 六用例（§9.2）+ `--level` main 分发器 |

### 11.3 命名规范

- kernel 文件 `fla_chunk_delta.py`、测试 `test_fla_chunk_delta.py`（沙箱目录内算子前缀，任务硬性约束）
- host 包装函数名对齐上游：`chunk_gated_delta_rule_fwd_h` → `fla_chunk_delta_fwd`、`chunk_gated_delta_rule_bwd_dhu` → `fla_chunk_delta_bwd`（签名对齐 proto schema）

### 11.4 实现顺序

1. ✅ 设计文档 + L0 门槛测试计划（本文档 §9.2）+ proto refine
2. ⬜ `fla_chunk_delta.py`：先 fwd（fp16 域 + relay + 双门控链路），后 bwd（3 组 gemm + 反向递推）
3. ⬜ `test_fla_chunk_delta.py`：golden（含 cast 模拟）+ L0 六用例 + main
4. ⬜ L0 通过（精度收敛；`check_precision` 全绿）
5. ⬜ 扩展分层套件 L1/L2/Boundary（tilelang-op-test-design 场景 B）
6. ⬜ 性能验收（perf_fwd_bench / perf_bwd_bench；§9.5 量化判据 + §6.3 Stage 3 演进；不达标按裁定中断上报）

### 11.5 算子 proto.yaml（覆盖门禁用）

见 §11.2：proto 角色由 `proto_fla_chunk_delta.yaml` 承担（refine 后 `operator.inputs[].dtype` 全集与本文档 §9.3 精度表 dtype 行一致：float16 / bfloat16 / float32；`operator.attrs[].name` 覆盖 chunk_size、use_exp2、scale、save_new_value、output_final_state、use_initial_state 全部关键属性）。

---

## 附：本文档信息类别声明

- 【已确认事实】：§3.3 全部 API 出处（源码行号/examples 路径）、§3.5.2 八项适配结论、§4.5 硬件容量（npu-arch INI 真源）、HBM 实测上限 1263GB/s（fla_wy_fast Stage 3 画像）、chunk_o 1.417ms / 1.15us·gemm⁻¹ / 470ns·flag⁻¹ 实测锚点、§9.3 精度门禁（precision-standard.md 口径）
- 【推断】：§9.5 串行链延迟区间 [1.64, 4.71]ms 与 0.83us/gemm 达标判据（chunk_o proxy 外推，Stage 2/3 实测校准）、R5-R6 数值、UB 峰值 lifetime 复用后估计
- 【需 Stage 2 验证】：R2（serial + auto flag + 高密度交换）、R3（UB 峰值实测）、R4（cast 对拍收敛 / hi/lo 触发）、R7（gather 链 threads=2）、R8（transpose_A M=128 形态）
