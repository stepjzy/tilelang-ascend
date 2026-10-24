# fla_chunk_intra_token_parallel 算子设计文档

> GPU→Ascend 迁移设计（tilelang-op-orchestrator Stage 1 / tilelang-op-design）。
> 语义权威：`ref_gpu/fla_chunk_intra_token_parallel.py`（Triton/FLA，只读）；代码级起点：`ref_gpu/fla_chunk_intra_token_parallel_tilelang_gpu.py`（上游 TileLang CUDA，只读）；测试输入构造语义：`ref_gpu/fla_chunk_intra_cumsum.py`（只读）；计时口径：`ref_gpu/test_utils_kda.py`（只读）。
> 目标硬件：**Ascend 910B3**（本机实测 20 AI Cores / L2 192MB / HBM 64GB；实测 D2D 带宽 1259-1273 GB/s (r+w)，标定来源 ../tilelang-dev-journal/fla_wy_fast/perf_tuning/；片上容量按 A2 级：L1 512KB、UB 192KB、L0A/L0B 64KB、L0C 128KB per core）。
> 本文档区分【已确认事实】（源码/示例/journal 实测验证）、【推断】（标注依据）、【需 Stage 2 验证】三类信息。

---

## 1. 概述

### 1.1 算子名称

`fla_chunk_intra_token_parallel`（KDA intra-chunk token-parallel：per-token 衰减内积 Aqk/Akk 生成）

### 1.2 功能描述

KDA (Kimi Delta Attention) intra-chunk 前向子步骤：对每个 token i，计算其与所在 sub_chunk（SCS=16）内自身及前序 token j 的逐通道衰减内积，写出两个注意力中间矩阵：Aqk（chunk 内 64 列行，供后续 chunk_o 消费）与 Akk（subchunk 内 16 列行，对角块）。上游为 token-parallel（每 token 一个 block）设计；本设计在保持逐 (i,j) 语义不变的前提下改为 subchunk-parallel 分块（见 §5 与 §10.4-A），原因：上游结构在 NPU 上 k_j/g_j 平均重读 8.5 倍（GM 流量 ~3.4GB，带宽不可行）。

### 1.3 数学公式

记 chunk_size CS=64（S 可整除）、sub_chunk_size SCS=16。对 batch b、全局 token 序号 i ∈ [0,S)、head h：

- chunk 序号 c = i // CS；subchunk 序号 s = (i % CS) // SCS；subchunk 首坐标 i_ts = c·CS + s·SCS

对 j ∈ [i_ts, min(i+1, i_ts+SCS))（subchunk 内自身及前序 token）：

$$kgj[h,k] = k[b,j,h,k] \cdot 2^{(g[b,i,h,k] - g[b,j,h,k])}$$
$$Aqk[b,i,h,(j \bmod CS)] = scale \cdot \sum_k q[b,i,h,k] \cdot kgj[h,k] \quad (scale=1.0)$$
$$Akk[b,i,h,(j-i_{ts})] = \begin{cases} \sum_k (\beta[b,i,h] \cdot k[b,i,h,k]) \cdot kgj[h,k], & j < i \\ 0, & j = i \end{cases}$$

- Aqk 行内未被 j 覆盖的位置置 0（上游 tilelang 版 T.clear 后全行写出；Triton 版仅写覆盖位——**golden 以清零语义为准**，proto 同口径）。
- g = gk 为 chunk 内累积 log2 衰减（cumsum 输出，g_i−g_j ≤ 0 对 j ≤ i，即 2^(g_i−g_j) ∈ (0,1] 恒无上溢）。
- 2^x 的昇腾实现为 exp(x·ln2)（§3.5.2-4）。

### 1.4 算法描述

每 block 处理一个 (batch b, subchunk, head-group BH=8)：

1. **载入**（每 vid 负责其中 BH_V=4 个 head）：按 token 逐行 `T.copy` 载入该 subchunk 的 q/k（bf16，[BH_V, DK] 连续行）与 gk（fp32 直载，无 cast）；beta 以 [SCS, BH_V] 小块载入。
2. **预处理**（UB fp32 域）：`T.tile.cast` 整块 q/k → fp32；g2 = gk ⊙ ln2（in-place，为 exp 等价 exp2 做底数换算）。
3. **逐对计算**（i 遍历 subchunk 内 16 个 token，j ∈ [0, i] 静态循环 + guard）：
   - e = exp(g2_i − g2_j)（[BH_V, DK] tile：sub → exp）
   - kgj = k_j ⊙ e；Aqk 值 = rowsum(q_i ⊙ kgj)；j<i 时 Akk 值 = rowsum((β_i·k_i) ⊙ kgj)
   - 结果以 UB 标量赋值写入 fp32 staging 行
4. **写回**：staging 整块 `T.tile.cast` → bf16，单次 2D 行间 gap `T.copy` 写 GM（Aqk [16, BH_V·CS]、Akk [16, BH_V·SCS]）。

> **Host 侧 Buffer 操作约束**（ascend-constraints.md §5）：本算子 host 路径只含 (a) 输入张量构造（数据准备）、(b) 形状/dtype 断言与 JIT 参数派生（纯 Python 整数运算）、(c) kernel 调用、(d) 精度验证。无 `.contiguous()`/pad/cat/`reshape` 物理化/aclnn 调用；布局全 bshd 连续，kernel 内直读 GM。

### 1.5 数据流图

```
GM[q,k:bf16] ─T.copy(同dtype)→ UB[bf16 rows] ─T.tile.cast→ UB[fp32 rows] ─┐
GM[gk:fp32]  ─T.copy(同dtype)→ UB[fp32 rows] ─T.tile.mul(·ln2)→ UB[g2]    ├─ 逐对 (i,j):
GM[beta:bf16]─T.copy(同dtype)→ UB[bf16] ─T.copy(UB↔UB cast)→ UB[fp32] ────┘
    e=exp(g2_i−g2_j) → kgj=k_j⊙e → pq=q_i⊙kgj ─reduce_sum→ Aqk 值
                                   → pk=kb_i⊙kgj ─reduce_sum→ Akk 值 (j<i)
    → UB 标量赋值 → UB[fp32 staging] ─T.tile.cast(CAST_RINT)→ UB[bf16 staging] ─T.copy→ GM[Aqk/Akk:bf16]
```

### 1.6 范围与分派覆盖审计

【已确认事实】范围裁剪（proto_fla_chunk_intra_token_parallel.yaml 声明）：定长序列（S % 64 == 0）、CS=64、SCS=16、scale=1.0 固定（kernel 不透传，对齐上游 tilelang 测试样例）；varlen（cu_seqlens 二分查找）不在本次范围。

```text
[DISPATCH-COVERAGE]
supported_domain: S%64==0; H%BH==0 (BH=8 主选, BH=4 降级，均编译期参数); DK%16==0 且 DK≥32（实测 128）; CS=64, SCS=16; dtype=(bf16 q/k/beta + fp32 gk 输入, bf16 输出); 定长序列; scale=1.0
generic_fallback: fla_chunk_intra_token_parallel 单一通用路径（BH 为性能参数非分派谓词，无 dtype/shape 特化分支）
specializations: 无
unsupported_inputs: varlen(cu_seqlens)、S%64≠0、H%BH≠0（BH 自适应到 4/2 后仍不整除）、DK%16≠0、scale≠1.0、fp16/fp32 输入变体 —— 与 supported_domain 不冲突（proto 已显式排除）
result: pass
```

---

## 2. 编程模式选型

### 2.1 模式结论

**Developer 模式**（用户明确指定，2026-10-22）——`alloc_shared` + 自动同步 + 默认 `(cid, vid)` 双线程，无显式 `T.Scope`、无手动同步、无 workspace。

### 2.2 选型理由

- 算子是**纯 Vector**（逐元素 exp/乘 + rowsum 归约，无 matmul）→ 无 Cube 单元、无 CV 核间编排需求，Expert 手动同步/显式层级无收益。
- 上游 TileLang CUDA 版本身即 Developer 风格（alloc_shared/fragment + T.copy/T.reduce_sum），迁移映射最直接。
- 本仓 KDA 家族先例 `examples_experiment/cumsum_kda/example_cumsum_kda.py` 为同款纯向量 Developer 实现（alloc_shared + tile.fill/reduce_sum + (cid,vid) VEC_NUM=2 头切分 + AUTO_SYNC），模式已验证。
- 纯向量无跨核数据依赖（各 vid 写 disjoint head 切片），避免 journal 记录的 CV 工具链缺陷域（多 gemm 组 L0 损坏 / cross-core sync 计数失配等均不触及）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（编译器映射 UB；reduce_sum 限 UB 已满足） |
| 计算方式 | `T.tile.sub/mul/exp/cast/fill/broadcast` + `T.reduce_sum(dim=-1, clear=True)` + 循环内 UB 标量赋值 |
| 作用域 | 无显式 T.Scope（纯向量，编译器统一放 V） |
| 同步方式 | 自动同步（AUTO_SYNC），无任何手动 barrier/flag |
| Kernel 轴 | `T.Kernel(grid, is_npu=True) as (cid, vid)`，vid 切分 head（VEC_NUM=2） |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `g2 = gk · ln2` | 底数换算：exp2(x) = exp(x·ln2)（T.exp2 昇腾无后端，§3.5.2-4） |
| 2 | `e_ij = exp(g2_i − g2_j)` | 逐对逐通道衰减因子，[BH_V, DK] tile |
| 3 | `kgj = k_j ⊙ e_ij` | 衰减 key |
| 4 | `Aqk_ij = rowsum(q_i ⊙ kgj)` | DK 归约（含 j==i 对角位） |
| 5 | `kb_i = β_i ⊙ k_i`；`Akk_ij = rowsum(kb_i ⊙ kgj)` | 仅 j < i |
| 6 | staging 清零 + 结果散写；整块 cast bf16 写 GM | Aqk 未覆盖位 0 / Akk 对角位 0 |

### 3.2 TileLang API 映射（含出处）

| 步骤 | TileLang API（昇腾） | 参数 | 出处（已验证） |
|------|---------------------|------|---------------|
| Kernel | `T.Kernel(B * NSC * HG, is_npu=True) as (cid, vid)` | 一维 grid；NSC=S//SCS、HG=H//BH；cid 分解 sc/hg/b（sc 主序） | cumsum_kda.py:120-124（(cid,vid)+VEC_NUM=2 同款）；一维化依据 ascend-constraints §1 |
| 内存分配 | `T.alloc_shared(shape, dtype)` | 全部缓冲（Developer 映射 UB） | cumsum_kda.py:126；gelu_mul_developer.py:26 |
| 载入 | `T.copy(GM 行切片, ub_region)` | 每 token 一次 [BH_V, DK] 连续行拷（q/k/gk）+ beta [SCS, BH_V] | cumsum_kda.py:131（同款行切片）；flash_attn_bshd_developer.py:76 |
| UB↔UB cast | `T.copy(ub_fp32, ub_bf16)` / `T.tile.cast(dst, src, mode, count)` | bf16→fp32 输入；fp32→bf16 输出用 CAST_RINT（count 16 对齐） | flash_attn_bshd_developer.py:105-106；api-compute.md §T.tile.cast |
| 底数换算 | `T.tile.mul(g2, gk, LN2)` | in-place 标量乘 | gelu_mul_developer.py:41 |
| 逐对衰减 | `T.tile.sub(e, g2_i_region, g2_j_region)` + `T.tile.exp(e, e)` | region 切片作操作数 | gdn_chunk_scaled_dot_kkt.py:84-85（同款 exp(g_i−g_j)）；fa_opt/expert_v3/kernel.py:335（tile op 切片操作数先例） |
| 逐元素乘 | `T.tile.mul(dst, src0, src1)` | kgj/pq/pk/kb | gelu_mul_developer.py:37-53 |
| 行广播 | `T.copy(beta_f32[i,:], beta_col[:,0])` + `T.tile.broadcast(beta_bc, beta_col)` | [BH_V,1]→[BH_V,DK] | gdn_chunk_scaled_dot_kkt.py:77,82（同款 g_r_ub 列拷+广播链） |
| 归约 | `T.reduce_sum(pq, sum_aqk, dim=-1, clear=True)` | 限 UB（满足）；clear=True 覆写语义 | reduce_ascend.py:456-494（签名）；flash_attn_bshd_developer.py:100（dim=-1）；HISA block_sparse_mqa_attn_expert.py:526（clear=True） |
| 清零 | `T.tile.fill(region, 0.0)` | staging 行清零（替代上游 T.clear，语义相同） | ascend_tile.py:352-376（BufferRegion 版 fill）；cumsum_kda.py:130 |
| 结果散写 | `for h in range(BH_V): stage[i, col(h)] = sum_aqk[h]` | UB 标量赋值（避免 T.Parallel 标量索引——journal #9 bug 规避） | cumsum_kda.py:133-142（循环内标量赋值同款） |
| 写回 | `T.tile.cast(bf16_stage, fp32_stage, "CAST_RINT", n)` + `T.copy(bf16_stage, GM)` | 整块 cast + 单次 2D 行间 gap 拷 | api-compute.md §T.tile.cast（high→low 用 CAST_RINT） |

### 3.3 计算伪代码

```python
LN2 = 0.6931471805599453
BH_V = BH // 2                       # 每 vid 的 head 数（BH=8 → BH_V=4）
NSC, HG = S // SCS, H // BH

@T.prim_func
def kernel(Q, K, GK, Beta, Aqk, Akk):            # dtype 见 §4.1/4.2
    with T.Kernel(B * NSC * HG, is_npu=True) as (cid, vid):
        sc  = cid % NSC                            # subchunk 主序（同 subchunk 各 head-group 相邻，
        hg  = (cid // NSC) % HG                    #  wave 内 GM 访问集中于 786KB 连续区）
        bb  = cid // (NSC * HG)
        c, i_s = sc // (CS // SCS), sc % (CS // SCS)
        i_ts = c * CS + i_s * SCS                  # subchunk 首坐标
        h0v  = hg * BH + vid * BH_V                # 本 vid 的 head 起点

        q_h   = T.alloc_shared([SCS * BH_V, DK], "bfloat16")   # 载入中转（lifetime 短）
        k_h   = T.alloc_shared([SCS * BH_V, DK], "bfloat16")
        q_f   = T.alloc_shared([SCS * BH_V, DK], "float32")    # 计算域（持久）
        k_f   = T.alloc_shared([SCS * BH_V, DK], "float32")
        g2    = T.alloc_shared([SCS * BH_V, DK], "float32")    # gk·ln2
        beta_h = T.alloc_shared([SCS, BH_V], "bfloat16")
        beta_f = T.alloc_shared([SCS, BH_V], "float32")
        e/kgj/pq/pk/kb = T.alloc_shared([BH_V, DK], "float32") # 逐对临时 ×5
        beta_col = T.alloc_shared([BH_V, 1], "float32")
        beta_bc  = T.alloc_shared([BH_V, DK], "float32")
        sum_aqk/sum_akk = T.alloc_shared([BH_V], "float32")
        aqk_s = T.alloc_shared([SCS, BH_V * CS], "float32")    # staging（(h,col) 平铺，与 GM 行布局一致）
        akk_s = T.alloc_shared([SCS, BH_V * SCS], "float32")
        aqk_h = T.alloc_shared([SCS, BH_V * CS], "bfloat16")
        akk_h = T.alloc_shared([SCS, BH_V * SCS], "bfloat16")

        # 1. 载入：每 token 一行（源/目的均为 BH_V·DK 连续段；gk fp32 同 dtype 直载无 cast）
        for t in T.serial(SCS):
            T.copy(Q[bb, i_ts + t, h0v, 0],  q_h[t * BH_V, 0])
            T.copy(K[bb, i_ts + t, h0v, 0],  k_h[t * BH_V, 0])
            T.copy(GK[bb, i_ts + t, h0v, 0], g2[t * BH_V, 0])
        T.copy(Beta[bb, i_ts, h0v], beta_h)

        # 2. 预处理：整块 cast fp32 + 底数换算
        T.tile.cast(q_f, q_h, "CAST_NONE", SCS * BH_V * DK)
        T.tile.cast(k_f, k_h, "CAST_NONE", SCS * BH_V * DK)
        T.copy(beta_h, beta_f)                     # UB↔UB 跨 dtype cast（1024-105 先例）
        T.tile.mul(g2, g2, LN2)                    # g2 = gk · ln2（in-place）

        # 3. 逐对计算（静态边界 + guard；i/j 均为编译期常量界循环变量）
        for i in T.serial(SCS):
            T.tile.fill(aqk_s[i, 0 : BH_V * CS], 0.0)     # 行清零（未覆盖位 0）
            T.tile.fill(akk_s[i, 0 : BH_V * SCS], 0.0)
            if i > 0:
                T.copy(beta_f[i, 0 : BH_V], beta_col[:, 0])
                T.tile.broadcast(beta_bc, beta_col)        # [BH_V,1] → [BH_V,DK]
                T.tile.mul(kb, k_f[i * BH_V : (i + 1) * BH_V, :], beta_bc)
            for j in T.serial(SCS):
                if j <= i:
                    T.tile.sub(e,   g2[i * BH_V : (i + 1) * BH_V, :],
                                    g2[j * BH_V : (j + 1) * BH_V, :])
                    T.tile.exp(e, e)                       # e = 2^(g_i−g_j)
                    T.tile.mul(kgj, k_f[j * BH_V : (j + 1) * BH_V, :], e)
                    T.tile.mul(pq,  q_f[i * BH_V : (i + 1) * BH_V, :], kgj)
                    T.reduce_sum(pq, sum_aqk, dim=-1, clear=True)
                    for h in range(BH_V):
                        aqk_s[i, h * CS + i_s * SCS + j] = sum_aqk[h]
                    if j < i:                              # Akk 对角位（j==i）保持 0
                        T.tile.mul(pk, kb, kgj)
                        T.reduce_sum(pk, sum_akk, dim=-1, clear=True)
                        for h in range(BH_V):
                            akk_s[i, h * SCS + j] = sum_akk[h]

        # 4. 写回：整块 cast + 单次 2D 行间 gap 拷（每 token 行 BH_V·CS 连续，行距 H·CS）
        T.tile.cast(aqk_h, aqk_s, "CAST_RINT", SCS * BH_V * CS)
        T.tile.cast(akk_h, akk_s, "CAST_RINT", SCS * BH_V * SCS)
        T.copy(aqk_h, Aqk[bb, i_ts, h0v, 0])
        T.copy(akk_h, Akk[bb, i_ts, h0v, 0])
```

> Aqk 列号推导：j 全局 = i_ts + j_local，j % CS = (c·CS + i_s·SCS + j_local) % CS = i_s·SCS + j_local（i_s<4、j_local<16 恒 <CS）✓；staging 行内 (h, col) 平铺与 GM [b, s, h0v:h0v+BH_V, :] 行布局逐元素一致（flat 偏移同构）。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| T.tile.sub / mul / exp / fill / broadcast / cast | gdn_chunk_scaled_dot_kkt.py、gelu_mul_developer.py、api-compute.md | 已验证 |
| T.reduce_sum(dim=-1, clear=True) | reduce_ascend.py:456-494 签名；dim=-1（flash_attn_bshd_developer.py:100）与 clear=True（HISA:526）各有先例 | 【需 Stage 2 验证】组合用法（两者合并）L0 首验 |
| T.copy 同 dtype GM↔UB、UB↔UB 跨 dtype cast、region 切片 | flash_attn_bshd_developer.py:76,105-106；cumsum_kda.py:131 | 已验证（GM↔UB 直跨 dtype 被工具链缺陷禁止，见 §3.5.2-6，本设计规避） |
| T.tile.* 以 buffer region 切片为操作数 | fa_opt/expert_v3/kernel.py:335 | 已验证 |
| 循环变量 guard（if j<=i / if i>0） | cumsum_kda.py:133-136（range+if 同款） | 已验证 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（强制检测）

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **是**（上游 `T.Kernel(B*S, ceildiv(H,block_H))` 二维） | 一维化：`T.Kernel(B*NSC*HG)` + cid 分解（cumsum_kda.py:120-124 / gdn_wy_fast.py:34-37 同款） |
| threads 参数限制（仅 1/2） | **是**（上游 threads=128/256 autotune） | 不显式指定 threads，默认 `(cid, vid)` 双线程 + vid 切分 head（cumsum_kda 同款）；threads 搜索维度删除 |
| 动态循环边界不支持 | **已消除**（结构级） | 上游 `T.Pipelined(loops)` 的 loops=bs+1−i_ts 为 per-block 动态 → subchunk 分块后 i/j 循环边界全为编译期常量 + guard（`for j in T.serial(SCS): if j <= i`，cumsum_kda 静态界+guard 先例），不触碰动态边界限制；block 索引派生动态界的等价先例见 moe_topk_gate.py:46,70（本设计不依赖） |
| 流水线不支持动态边界 | 不涉及 | 无 T.Pipelined（逐对工作集小、~950 tile ops/block，静态双缓冲收益有限；Stage 3 可评估对载入循环做静态 T.Pipelined） |
| L0C 溢出（128KB） | 不涉及 | 纯 Vector，无 L0C/fragment 分配 |
| T.copy 列方向 strided 切片 | 已规避 | 所有 GM 切片的最后非 1 extent 维（DK / (h,col) 平铺维）均为 buffer 最内维且连续；多级 gap 通过 (h,d)/(h,col) 平铺降为 2D 行间 gap（DataCopyPad 支持域） |
| GEMM 非整除 / 分形限制 | 不涉及 | 无 GEMM（Stage 3 GEMM 卸载研究选项需重审，见 §10.4-D） |

#### 3.5.2 上游 CUDA 专属机制的昇腾适配（逐项）

| # | 上游机制（CUDA） | 昇腾适配结论 | 依据 |
|---|------------------|-------------|------|
| 1 | `T.Kernel(B*S, ceildiv(H, block_H))` 二维 | 一维化 `B*NSC*HG` + cid 分解（§3.5.1） | ascend-constraints §1；cumsum_kda/gdn 先例 |
| 2 | `threads=128/256` + autotune 搜索 | 默认双线程 `(cid, vid)`；autotune 不进 L0（固定 BH=8），Stage 3 枚举 (BH, 计算形式) | ascend-constraints §1；fla_wy_fast journal §3.5.2-2 同款决策 |
| 3 | `T.Pipelined(loops, num_stages)` 动态边界 | **静态化消除**（subchunk 分块后边界为常量；guard 模式） | ascend-constraints §1"流水线不支持动态边界"；cumsum_kda 静态界先例 |
| 4 | `T.exp2(x)` | **昇腾后端不可用 → `T.tile.exp(x·LN2)`**。前端 tilelang/language/tir/op.py:1837 有定义，但 src/op 无 exp2 指令映射（仅 ascend_exp→AscendC::Exp）、examples 无先例；fp32 下 exp(g·ln2) 与 exp2(g) 相对误差 ~1e-7 量级，远小于 bf16 门禁（rtol 2⁻⁶） | 【已确认事实】源码检索 + fla_wy_fast journal §3.5.2-5 同款论证；【推断】误差量级（golden 用 torch.exp2 对拍覆盖） |
| 5 | `T.clear(Sum_Aqk_shared)` | `T.tile.fill(region, 0.0)`（ascend_tile.py:352 BufferRegion 版，语义相同） | ascend_tile.py:352-376；cumsum_kda.py:130 |
| 6 | `T.copy(Aqk_shared_fp32, Aqk_bf16_GM)` 输出直存 / 跨 dtype GM 直拷 | **规避 GM↔UB 跨 dtype T.copy**（工具链 codegen 缺陷：copy_gm_to_ub<bfloat16_t>(LocalTensor<float>) 非法模板，journal 工具链缺陷 #6）→ 输出走 fp32 staging → `T.tile.cast(CAST_RINT)` → 同 dtype `T.copy` 写 GM | 【已确认事实】journal debug 记录；flash_attn_bshd_developer.py:105-106 两段式先例 |
| 7 | `T.copy(Sum_fragment, Sum_shared[:, j % CS])` 列向小片写 | 改 UB 标量赋值散写 staging（4 elem 列写在昇腾低效且跨 dtype；journal #9 的 T.Parallel 标量索引 bug 以 range 循环规避） | cumsum_kda.py:133-142；journal 工具链缺陷 #9 |
| 8 | `T.disable_warp_group_reg_alloc()` | **直接删除**（CUDA SM 专属概念，昇腾无对应物） | fla_wy_fast journal §3.5.2-1 同款 |
| 9 | `@autotune(block_H, threads, num_stages)` | L0 固定 (BH=8)；Stage 3 按 (BH ∈ {4,8}, 直接式/因式分解, 载入聚合方式) 枚举 | 本仓 tilelang/autotuner 存在、examples/autotune 有先例；性能口径见 §9.5 |

#### 3.5.3 参考实现差异说明（GPU → Ascend 汇总）

| 差异项 | 参考实现（CUDA/上游） | 本项目（Ascend 910B3） | 转换方案 |
|--------|---------------------|----------------------|---------|
| 并行粒度 | 每 block 1 token × block_H heads（grid 8192×64 级） | 每 block 1 subchunk × 8 heads（grid B×512×8） | §5 / §10.4-A（消除 8.5× k_j/g_j 重读） |
| Kernel 维度 | 二维 | 一维 + cid 分解 | §3.5.2-1 |
| threads | 128/256 | 默认 2 (cid, vid) | §3.5.2-2 |
| j 循环 | `T.Pipelined(loops)` per-block 动态 | 静态 SCS + guard | §3.5.2-3 |
| exp2 | `T.exp2` | `T.tile.exp(x·ln2)` | §3.5.2-4 |
| T.clear | buffer 级 | `T.tile.fill(region)` | §3.5.2-5 |
| 输出写 | reduce 后逐列 fragment→shared 拷 | fp32 staging 散写 + 整块 cast + 单次 2D 拷 | §3.5.2-6/7 |
| autotune/pass_configs | TL_ENABLE_FAST_MATH | 昇昇 pass_configs 三件套（§4.7） | §3.5.2-9 |

#### 3.5.4 本项目同类实现参考（强制步骤 0 结果）

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples_experiment/cumsum_kda/example_cumsum_kda.py` | **高度相似**（KDA 家族已交付：同 pipeline 的 gk 生产者；Developer 纯向量、(cid,vid)+VEC_NUM=2、alloc_shared+tile.fill/reduce_sum、chunk 索引算术、三件套 pass_configs） | Kernel 骨架与 pass_configs 直接沿用 |
| `examples/linear_attention_and_rnn/gdn/gdn_chunk_scaled_dot_kkt.py` | 高（同族 exp(g_i−g_j) 衰减内积、strict-tril 语义、tile.exp/broadcast/列拷链） | 衰减因子计算链（sub→exp）与 beta 广播模式 |
| `examples/developer_mode/flash_attn_bshd_developer.py` | 高（Developer 骨架：跨 dtype UB↔UB cast、reduce_sum dim=-1、tile 切片操作数） | cast 两段式、reduce_sum 用法 |
| `examples/tile_kernels/moe/moe_topk_gate.py` | 中（block 索引派生 per-block 动态迭代数 `T.serial` 先例） | 动态界证据（本设计最终不依赖） |
| `examples/developer_mode/gelu_mul_developer.py` | 中（纯向量 Developer tile 链、in-place 标量乘） | tile 链组织 |
| `../tilelang-dev-journal/fla_wy_fast/`（DESIGN.md + perf_tuning/） | 标定参考 | 910B3 实测带宽/向量吞吐标定、exp2 替代论证、文档结构 |

> 注：gdn 系列的 g 为 per-token 标量；本算子 gk 为 per-token per-channel [B,S,H,DK]（KDA 特性），元素运算量更大但结构同族。gdn 为 Expert 风格，本设计按用户指定用 Developer 风格（以 cumsum_kda/flash_attn_bshd_developer 为骨架）。

---

## 4. 数据规格与内存规划

### 4.1 输入张量（布局 bshd 连续，kernel 直读 GM）

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| q | [B, S, H, DK] | bfloat16 | query |
| k | [B, S, H, DK] | bfloat16 | key |
| gk | [B, S, H, DK] | float32 | chunk 内累积 log2 衰减（cumsum 输出；g_i−g_j ≤ 0 对 j ≤ i） |
| beta | [B, S, H] | bfloat16 | 门控系数 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| Aqk | [B, S, H, CS=64] | bfloat16 | chunk 内 A_qk 行（未覆盖位置 0） |
| Akk | [B, S, H, SCS=16] | bfloat16 | 对角 subchunk 的 A_kk 行（对角位 0） |

### 4.3 中间缓冲区（每 vid，BH=8 → BH_V=4，SCS=16，DK=128；全部 UB）

| Buffer | Shape | dtype | 用途 |
|--------|-------|-------|------|
| q_h / k_h | [64, 128] | bfloat16 | 载入中转（lifetime 止于整块 cast） |
| q_f / k_f / g2 | [64, 128] | float32 | 计算域持久缓冲（行 t·BH_V+h = token t、head h） |
| beta_h / beta_f | [16, 4] | bf16 / fp32 | beta 载入与 fp32 域 |
| beta_col / beta_bc | [4,1] / [4,128] | float32 | 行广播链 |
| e / kgj / pq / pk / kb | [4, 128] | float32 | 逐对临时 ×5 |
| sum_aqk / sum_akk | [4] | float32 | reduce 输出 |
| aqk_s / aqk_h | [16, 256] / [16, 256] | fp32 / bf16 | Aqk staging（(h,col) 平铺）+ cast |
| akk_s / akk_h | [16, 64] / [16, 64] | fp32 / bf16 | Akk staging + cast |

### 4.4 内存搬运路径

```
GM[q,k:bf16] ─T.copy→ UB[q_h,k_h:bf16] ─T.tile.cast→ UB[q_f,k_f:fp32]
GM[gk:fp32]  ─T.copy(同dtype)→ UB[g2:fp32] ─T.tile.mul(·ln2)→ UB[g2]
GM[beta:bf16]─T.copy→ UB[beta_h] ─T.copy(UB↔UB cast)→ UB[beta_f:fp32]
UB[fp32 计算域] ─逐对 tile ops + reduce_sum→ UB[sum_*] ─标量散写→ UB[staging:fp32]
UB[staging:fp32] ─T.tile.cast(CAST_RINT)→ UB[staging:bf16] ─T.copy(2D 行间gap)→ GM[Aqk,Akk:bf16]
```

> 层级合法性：无 GM→L0 直搬（无 L0）；全部 Vector 计算与 reduce 在 UB；无跨级访问。

### 4.5 UB 内存预算（192KB 上限，A2/A3 级）

| 阶段 | 构成 | 大小 |
|------|------|------|
| 载入+cast 峰值 | q_h 16 + k_h 16 + g2 32 + beta 0.3 + staging 30 | ~94KB |
| 计算域稳态 | q_f 32 + k_f 32 + g2 32 + 临时 10 + staging 30 + beta 链 2.3 | **~138KB（72%）** |

结论：138KB / 192KB ✓（依赖 MEMORY_PLANNING 对 q_h/k_h 的 lifetime 复用——载入中转与计算域稳态不重叠）。**降级预案**：BH=4（BH_V=2）时持久缓冲减半至 ~72KB；再溢出则 DK 分块循环（DK%16 域内）。

### 4.6 动态轴定义

无运行时动态轴。B/S/H/DK/CS/SCS/BH 均为 JIT 编译期常量（对齐上游 @tilelang.jit 写法，随 shape 重编译）；NSC/HG/grid 编译期展开。

### 4.7 JIT 配置

```python
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # cumsum_kda 同款（纯向量下无副作用）
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,    # UB lifetime 规划（§4.5 预算前提）
}

@tilelang.jit(out_idx=[-2, -1], pass_configs=PASS_CONFIGS)
def fla_chunk_intra_token_parallel_kernel(B, S, H, DK, CS=64, SCS=16, BH=8,
                                          input_dtype="bfloat16",
                                          gate_dtype="float32",
                                          output_dtype="bfloat16"): ...
# host 包装：fla_chunk_intra_token_parallel(q, k, gk, beta, chunk_size=64, sub_chunk_size=16)
#   断言 S%CS==0、CS%SCS==0、H%BH==0、DK%16==0、dtype 契约 → 返回 (Aqk, Akk)
```

---

## 5. Tiling 策略

### 5.1 计算类型

**纯 Vector**（逐元素 exp/乘 + rowsum 归约；无 matmul → 无分形/L0 约束）。

### 5.2 Block 划分

```python
SCS = 16                # 每 block 的 token 数 = 整个 subchunk（语义常量）
BH  = 8                 # 每 block 的 head 数（BH_V=4/vid）；候选 {4, 8}，16 时 UB 超限（§4.5 量级外推）
NSC = S // SCS          # subchunk 数（bench: 512）
HG  = H // BH           # head-group 数（bench: 8）
grid = B * NSC * HG     # bench: 1×512×8 = 4096 blocks（×2 vid = 8192 向量任务）
# cid 分解（sc 主序）：sc = cid % NSC; hg = (cid//NSC) % HG; bb = cid//(NSC*HG)
# i_ts = (sc//(CS//SCS))*CS + (sc%(CS//SCS))*SCS；h0v = hg*BH + vid*BH_V
```

**选型理由**：① 每 block 覆盖完整 subchunk → k_j/g_j 片上复用，每 GM 字节恰好读 1 次（上游 1-token/block 结构平均重读 8.5×，GM 流量 ~3.4GB 带宽不可行，§10.4-A）；② j 循环边界静态化（§3.5.2-3）；③ BH_V=4 使逐对 tile 为 [4,128]=512 elem（有效向量宽度）；④ sc 主序使同 subchunk 的 8 个 head-group block 相邻调度，wave 内 GM 访问集中于 16 token × 64 head × (2+2+4)B ≈ 786KB 连续区。

### 5.3 约束分析

- 对齐：bf16 行拷 BH_V·DK=512 elem=1KB（32B 对齐 ✓）；fp32 行拷 2KB ✓；gk fp32 直载无 cast ✓；cast count（8192/4096/1024）均 16 对齐 ✓
- UB 容量：138KB / 192KB ✓（§4.5）
- L0/L0C：不涉及（纯 Vector）
- 尾块：无（S%64==0 ⟹ 每个 subchunk 恒满 16 token；H%BH==0、DK 为整块缓冲无切分）

### 5.4 注意事项

- **无尾块路径**（S%64==0 / H%BH==0 为支持域硬约束，违反者 host 断言拒绝——L2 负向用例覆盖，不属尾块处理）；未来扩展非整除 S 时须在 GM 输入/输出两侧引入 valid extent/BufferRegion（禁止 host padding+crop），属 L1 范畴。
- beta 载入为 [16, 4] 行间 gap 小拷（行 8B < 32B burst）——数据量 128 elem 可忽略；Stage 3 可评估与 q 行拷聚合。

### 5.5 数据搬运性能可行性（REORDER-COST）

本算子无数据重排（bshd 连续直读，无 permute/gather），全部搬运为规则 tile 块 DMA：

```text
[REORDER-COST]
path: 主路径 (BH=8, SCS=16, DK=128)
gm_passes: 每张量 1 读 + 1 写（唯一 IO 593 MiB @bench shape，§9.5 明细）
dma_transactions: ~51 次/block（3 张量 ×16 token 行拷 + beta 1 + 输出 2），4096 blocks ≈ 21 万次
average_dma_bytes: ~1.4KB/次（1KB bf16 / 2KB fp32 行拷为主）
gm_scalar_accesses: 0（标量散写均在 UB；GM 无标量访问）
address_div_mod_per_element: 0（cid 分解每 block 4 次；i/j/h 循环索引编译期展开）
active_cores / serial_tasks_per_core: ~40 向量核（20 AI Core × 2 vid）/ ~205 vid 任务/核
largest_case_timeout_gate: perf_bench 单 case 180s（§9.4）
result: pass
```

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| (b, subchunk, head-group) | block 级并行 | `T.Kernel(B*NSC*HG)` | 每 block 独立一个 subchunk×head-group，无依赖 |
| vid（head 切分） | block 内 2 线程 | `(cid, vid)` 默认 | 每 vid 处理 BH_V=4 head，数据 disjoint |
| token i ∈ [0,SCS) | 静态串行 | `T.serial(SCS)` | 编译期常量界 16 |
| 对偶 j ∈ [0,SCS) | 静态串行 + guard | `T.serial(SCS)` + `if j <= i` | 规避动态边界（§3.5.2-3）；空迭代仅 1 条分支 |
| head 散写 h | 小串行 | `range(BH_V)` | 4 elem 标量赋值（规避 T.Parallel 标量索引 bug，journal #9） |
| 元素级 | 向量化 | tile ops [BH_V, DK] | SIMD |

### 6.2 循环伪代码

见 §3.3（i/j 双层静态循环 + guard 为核心结构）。

### 6.3 流水线优化

不使用 `T.Pipelined`（上游动态边界不可迁移；静态化后载入循环仅 16 次迭代、每次 1-2KB，双缓冲收益有限且引入 journal 工具链缺陷 #2 风险域）。Stage 3 可评估：载入循环静态 `T.Pipelined(num_stages=2)` 与逐对循环的软件流水（当前 ~950 tile ops/block 的指令下发与 MTE 搬运重叠）。

### 6.4 尾块处理

无尾块（§5.4）。

---

## 7. 同步策略

### 7.1 同步模式

**自动同步**（Developer 标准）：AUTO_SYNC 处理核内 MTE/Vector 依赖。源码无任何 `T.barrier_all` / `T.set_flag` / `T.set_cross_flag`。

### 7.2 同步点说明（编译器自动插入，设计仅声明数据依赖）

| 依赖 | 说明 |
|------|------|
| tile ops / 标量散写 ← GM 载入（T.copy） | AUTO_SYNC |
| 整块 cast ← 载入中转写满 | AUTO_SYNC |
| reduce_sum ← pq/pk 就绪；散写 ← reduce 输出 | AUTO_SYNC |
| staging cast ← 散写完成（整块 lifetime） | AUTO_SYNC |
| GM 写 ← bf16 staging 就绪 | AUTO_SYNC |
| vid 间 | 无同步需求（head 切片 disjoint，无共享 buffer 写冲突） |

### 7.3 pass_configs 配置

见 §4.7（AUTO_CV_COMBINE + AUTO_SYNC + MEMORY_PLANNING 三件套，cumsum_kda 同款）。

---

## 8. 融合算子设计

### 8.1 融合算子判定

**否**（纯 Vector，无 Cube/GEMM 单元；无 CV 交互、无 workspace）。本节按模板声明：Developer 模式、无 `workspace_idx`、无 vid 偏移索引（vid 仅作 head 切分）、无片上跨核数据流。Stage 3 若启用 GEMM 卸载研究选项（§10.4-D）需重开本节设计。

---

## 9. 验证方案

### 9.1 Golden 函数（PyTorch 重写，供 Stage 2 直接落地）

> NPU 环境 Triton 不可运行（ref_gpu 仅语义对照）。golden 用 `torch.exp2` 精确语义，kernel 侧 exp(g·ln2) 的实现差异计入混合容差验收。按 subchunk 向量化（512 次迭代 @bench shape，CPU 可承受）；输入构造对齐上游 prepare_input 语义（logsigmoid + chunk 内 cumsum，见 fla_chunk_intra_cumsum.py）。数据准备/随机数/golden 全在 CPU 完成后 H2D，运行段无 aclnn 依赖。

```python
import torch


def prepare_input_kda(B, S, H, DK, chunk_size=64, input_dtype=torch.bfloat16,
                      gate_dtype=torch.float32):
    """CPU 构造：q/k/beta randn bf16；gk = logsigmoid(randn fp32) 后 chunk 内 cumsum。"""
    q = torch.randn(B, S, H, DK, dtype=input_dtype)
    k = torch.randn(B, S, H, DK, dtype=input_dtype)
    beta = torch.randn(B, S, H, dtype=input_dtype)
    gk = torch.nn.functional.logsigmoid(torch.randn(B, S, H, DK, dtype=gate_dtype))
    gk = gk.view(B, S // chunk_size, chunk_size, H, DK).cumsum(dim=2).view(B, S, H, DK)
    return q, k, gk, beta


def fla_chunk_intra_token_parallel_ref(q, k, gk, beta, chunk_size=64,
                                       sub_chunk_size=16, scale=1.0):
    """golden。q/k [B,S,H,DK] bf16, gk [B,S,H,DK] fp32, beta [B,S,H] bf16
    -> (Aqk [B,S,H,CS] bf16, Akk [B,S,H,SCS] bf16)。清零语义（未覆盖位 0，对角位 0）。"""
    B, S, H, DK = q.shape
    CS, SCS = chunk_size, sub_chunk_size
    assert S % CS == 0 and CS % SCS == 0
    qf, kf, bf, gf = q.float(), k.float(), beta.float(), gk.float()
    Aqk = torch.zeros(B, S, H, CS, dtype=torch.bfloat16)
    Akk = torch.zeros(B, S, H, SCS, dtype=torch.bfloat16)
    for c in range(S // CS):
        for s in range(CS // SCS):
            i_ts = c * CS + s * SCS
            sl = slice(i_ts, i_ts + SCS)
            g_sub, k_sub = gf[:, sl], kf[:, sl]                     # [B,SCS,H,DK]
            e = torch.exp2(g_sub.unsqueeze(2) - g_sub.unsqueeze(1)) # [B,SCS(i),SCS(j),H,DK]
            kgj = k_sub.unsqueeze(2) * e
            aqk = torch.einsum("bihd,bijhd->bijh", qf[:, sl], kgj) * scale
            akk = torch.einsum("bihd,bijhd->bijh",
                               bf[:, sl].unsqueeze(-1) * k_sub, kgj)
            keep = torch.tril(torch.ones(SCS, SCS, dtype=torch.bool))          # j <= i
            keep_strict = torch.tril(torch.ones(SCS, SCS, dtype=torch.bool), -1)  # j < i
            aqk = aqk.masked_fill(~keep, 0.0)
            akk = akk.masked_fill(~keep_strict, 0.0)
            Aqk[:, sl, :, s * SCS : s * SCS + SCS] = aqk.permute(0, 1, 3, 2).to(torch.bfloat16)
            Akk[:, sl, :, :] = akk.permute(0, 1, 3, 2).to(torch.bfloat16)
    return Aqk, Akk
```

交叉校验建议（Stage 2 可选）：极小 shape（S=64）追加三重显式循环逐 (i,j) 版本与本向量化版本对拍，锁定散布列号/掩码方向。

### 9.2 L0 门槛测试计划（tilelang-op-test-design 场景 A 生成，仅 L0）

> Stage 1 只产出 L0（规则 shape：S%64==0、H%BH==0、DK=128，block 整除）。L1（功能，含不规则 shape）/L2（异常）/Boundary（特殊值，含极端衰减 gk）由 Stage 2 在 L0 通过后按场景 B 扩展。

| 用例名 | 级别 | Shape (B,S,H,DK) | dtype | 覆盖标签 | 说明 |
|--------|------|------------------|-------|----------|------|
| l0_min | L0 | (1, 64, 8, 128) | q/k/beta bf16 + gk fp32 → 输出 bf16 | D-DTYPE-bf16, D-DTYPE-fp32, D-SHAPE-ALIGNED | 最小冒烟：1 chunk / 4 subchunk / 1 head-group（grid=4）；覆盖 i=0 首位、j==i 对角（Akk=0）、Aqk 列偏移 i_s∈{0,1,2,3} |
| l0_typical | L0 | (2, 256, 16, 128) | 同上 | D-DTYPE-bf16, D-DTYPE-fp32, D-SHAPE-ALIGNED | 多 batch / 4 chunk / 2 head-group（grid=64）；覆盖 chunk 边界 i_ts 重置、多 head-group 切分 |
| l0_bench_repr | L0 | (1, 1024, 64, 128) | 同上 | D-DTYPE-bf16, D-DTYPE-fp32, D-SHAPE-ALIGNED | benchmark 结构代表（H 全宽 64、16 chunk、grid=512）：exp 替代、cast 链、staging 散写在真实规模下验证 |

**golden 对照法**：每用例对 Aqk、Akk 分别调用 `check_precision(out, ref, "bfloat16")`（precision-standard.md §4.1 混合容差判定，整段复制不改动）；两输出全部通过打 `[PRECISION_PASS]`，任一失败 `[PRECISION_FAIL]`。数据构造用 §9.1 `prepare_input_kda`（CPU 生成后 H2D）。

### 9.3 精度标准（混合容差，precision-standard.md 口径）

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|-------|------|------|---------------------|------------------------|
| bfloat16（输出 Aqk/Akk；输入 q/k/beta） | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |
| float32（输入 gk 通道 dtype） | 2⁻¹⁶ (1.53e-5) | 2⁻¹⁰ (9.77e-4) | 1e-2 | 0.99 |

> 与 proto_fla_chunk_intra_token_parallel.yaml 声明的门禁逐字一致。gk 为 fp32 输入通道（不直接对外），输出域仅 bf16；float32 行满足 proto dtype 全集一致性（{bfloat16, float32} 与 inputs[].dtype 一致），若未来开 fp32 输出变体直接套用。特殊值（NaN/Inf）：本算子输入域为 randn 有限值（g_i−g_j ≤ 0 契约下 2^(g_i−g_j) ∈ (0,1] 恒无上溢；下溢到 0 为良性），Boundary 极端衰减用例由 Stage 2 场景 B 扩展并按 mask 结构比对。

### 9.4 性能可行性哨兵（强制，不可跳过）

| 用例名 | Shape (B,S,H,DK) | dtype/属性 | 覆盖路径 | 单 case timeout | 选择理由 |
|--------|------------------|------------|----------|-----------------|----------|
| perf_bench | (1, 8192, 64, 128) | bf16/fp32 gate, CS=64, SCS=16, scale=1.0 | 主路径（4096 blocks / 8192 vid 任务，最大任务数 + 最大 DMA 吞吐压力） | 180s | benchmark 验收 shape（含 JIT 编译余量） |

计时口径：NPU 事件计时，warmup=20、rep=10、取均值（对齐 ref_gpu/test_utils_kda.py do_bench；torch_npu 侧用 `torch.npu.Event(enable_timing=True)` 等价实现），仅设备侧 kernel 时间，wrapper 内不加 sync（fla_wy_fast journal 口径教训：内部 sync 会把 ~1ms/launch 的 host JIT 分发延迟暴露进设备侧计时）。

### 9.5 性能预算分析（910B3，含 370us 目标可行性结论）

**IO 明细（物理唯一字节，每张量单读单写；subchunk 分块已消除重读）**

| 项 | bench (1,8192,64,128) |
|----|----------------------|
| 读 | q 128 + k 128 + gk(fp32) 256 + beta 1 = **513 MiB** |
| 写 | Aqk 64 + Akk 16 = **80 MiB** |
| 合计 | **593 MiB ≈ 621 MB** |

**Roofline 对照（910B3，实测标定）**

| 项 | 数值 | 依据 |
|----|------|------|
| HBM 实测上限（D2D 大拷贝 r+w） | **1259-1273 GB/s** | 【已确认事实】../tilelang-dev-journal/fla_wy_fast/perf_tuning/（fla_wy_fast_final_summary.md §3） |
| 带宽下限 @ 实测上限 | 621MB ÷ 1263GB/s ≈ **492 us** | 100% 利用率 |
| 带宽下限 @ 81%（journal k1/k2 单 kernel 实际达到的逻辑带宽占比） | ≈ **608 us** | 【推断】同类大流量 kernel 实测占比 |
| 370us 隐含带宽 | 621MB ÷ 370us ≈ **1.68 TB/s = 实测上限的 1.33×** | **物理不可达** |

**向量计算预算（低算术强度：直接式 ≈ 6.4 elem-op/唯一字节）**

| 项 | 直接式（本设计） | 因式分解式（Stage 3 选项，§10.4-C） |
|----|----------------|-------------------------------------|
| (i,j,h) 对数 | 512 subchunk × 64 head × 136 对 = **4.456M** | 同左 |
| 每对 elem-ops | 896（sub 128 + exp 128 + mul 384 + rowsum 256） | 512 + 摊销 setup ≈ 560 |
| 总量 | **3.99G elem-ops**（含 571M exp） | **~2.8G**（含 134M exp） |
| 910B3 向量 fp32 吞吐 | 【推断，未实测】~2-5 T elem-op/s（20 AI Core × 2 vid × 64 lane 量级估算） | 同左 |
| 计算时间估计 | **0.8-2.0 ms** | **0.6-1.4 ms** |

**⚠️ [DESIGN_ERROR] 级发现：370us 目标与 910B3 硬件能力矛盾**

1. **带宽侧**：目标隐含 1.68 TB/s = 实测 D2D 上限（1263 GB/s）的 **1.33×**，纯带宽下限 492us（100% 利用率）/ 608us（81% 同类实测占比）均超目标——**不存在 kernel 层优化能弥合**（唯一 IO 已是该算子下限：subchunk 分块后每 GM 字节恰读/写 1 次，gk fp32 268MB 为接口契约不可压缩，Aqk/Akk 清零语义要求全行写出）。
2. **计算侧**：纯向量直接式估算 0.8-2.0ms（吞吐未实测，区间宽），即便因式分解式（-30% ops）也大概率高于 608us 带宽下限 → 算子实际预期**计算受限**，现实预期 ~0.8-2ms【推断，perf_bench 哨兵实测后收敛】。
3. **理论极限路径**（仅供编排层参考，非本设计承诺）：因式分解 + Cube GEMM 卸载（Aqk/Akk 行堆叠为单 gemm 组 [32,DK]@[DK,16]）理论 ~0.2-0.35ms（向量 setup ~0.54G ops + Cube 4.3 GFLOP bf16 小 GEMM 效率 10-25% 假设），逼近但无余量，且带 3 项工具链风险（journal #2/#3 缺陷、因式分解溢出域变化，§10.4-D）。

**替代路线（需编排层/用户决策，不静默简化）**：
1. 验收口径改为**带宽效率**：如 ≤ 750us（对应 ~66% 实测上限利用率）或按 perf_bench 实测首次标定后重定；
2. 确认目标硬件为更高带宽平台（910C / CloudMatrix 级 HBM）：910B3 仅做功能/精度验收，性能按带宽比例折算报告（fla_wy_fast journal 同款处理）；
3. 维持 370us 字面目标 → 本设计无法承诺，需回退需求或立项 GEMM 卸载研究路径（Stage 3，高风险）。

### 9.6 Host 侧 Buffer 操作审计

```text
[HOST-METADATA-AUDIT]
operation: 输入构造 torch.randn(...)/logsigmoid/cumsum（CPU 数据准备）+ 形状/dtype 断言（纯 Python 整数运算）+ kernel 调用 + check_precision 验证
input_stride -> output_stride: N/A（新建张量与只读校验，无变换）
shares_storage / same_data_ptr: N/A
aclnn_or_physical_copy: false（数据生成在 CPU；kernel 调用前后无 .contiguous()/pad/cat/reshape 物理化/.to()；输出由 tilelang out_idx 分配）
result: allow
```

---

## 10. 风险点与注意事项

### 10.1 风险清单（高/中/低分级）

| 级别 | 风险 | 影响 | 缓解/验证 |
|------|------|------|-----------|
| **高** | R1 性能目标 370us 物理不可达（需 1.33× 实测带宽上限；计算侧预期更紧 0.8-2ms） | 验收口径 | §9.5 完整论证 + 替代路线，**需编排层/用户决策** |
| **高** | R2 向量算力为实际瓶颈（3.99G elem-ops / 593MiB ≈ 6.4 ops/byte 低算术强度；910B3 向量吞吐未实测） | 性能 | perf_bench 哨兵首测标定；Stage 3 杠杆：因式分解式（-30% ops）、GEMM 卸载（研究性）、BH/流水调优 |
| 中 | R3 `T.reduce_sum(dim=-1, clear=True)` 组合用法（两参数各有先例、合并无先例） | 编译/正确性 | l0_min 首验；异常时拆为 fill(0)+clear=False 或两步 |
| 中 | R4 UB 稳态 138KB/192KB（72%）依赖 MEMORY_PLANNING lifetime 复用（journal #10：满 UB kernel 对 lifetime 延长零容忍） | 编译溢出 | 降级 BH=4（~72KB）；q_h/k_h 与计算域不重叠已显式设计 |
| 中 | R5 逐对循环 ~950 tile ops/block 的小 tile（512 elem）指令下发开销 | 性能 | journal 教训：tile 化已规避行串行（-9.21ms 级收益已锁定）；Stage 3 评估 (i+1) 行批量 tile 化与因式分解 |
| 低 | R6 1-2KB 小行拷 DMA 效率（51 次/block） | 性能微小 | REORDER-COST 已计（21 万次 × 1.4KB，连续 burst 友好）；Stage 3 可验证 3D T.copy 聚合载入 |
| 低 | R7 exp(x·ln2) vs exp2 精度 | 精度 | fp32 相对误差 ~1e-7 量级 ≪ bf16 门禁（journal 同款论证）；golden torch.exp2 对拍覆盖 |
| 低 | R8 `for h in range(BH_V)` UB 标量赋值散写 | 正确性 | cumsum_kda 同款先例；刻意不用 T.Parallel 标量索引（journal #9 bug 规避） |

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| 照抄 `T.exp2` | 直接移植上游 | 昇腾无 lowering | `T.tile.exp(x·LN2)`（本设计已内置） |
| GM↔UB 跨 dtype T.copy 直载直存 | 图省事跳过 cast 链 | codegen 非法模板（journal #6） | 同 dtype 载入 + tile.cast / UB↔UB 两段式（本设计已内置） |
| `T.Pipelined(loops)` 动态边界 | 直接移植上游 | 昇腾不支持 | subchunk 分块静态化 + guard（本设计已内置） |
| threads=128 / 二维 T.Kernel | 直接移植上游 | 编译报错 | 默认 (cid,vid) + 一维化（本设计已内置） |
| T.Parallel 内标量索引散写 | 加速 staging 写 | journal #9 已知 bug | range 小串行标量赋值（本设计已内置） |
| host 侧 gk 预处理（·ln2 / exp2 预计算） | 减少 kernel 工作量 | aclnn 违规 + IO 翻倍 | 禁止（被否方案 F，§10.4） |

### 10.3 特殊场景处理

- **非整除 S / H**：不支持域（host 断言拒绝，L2 负向覆盖）；未来扩展用 GM 两侧 valid extent/BufferRegion，禁止 host padding+crop。
- **极小 shape**（S=64）：grid=B×4×HG，负载少但正确；l0_min 覆盖。
- **极端衰减 gk**（subchunk 内累计 < -128）：直接式 exp(g_i−g_j) 恒 ∈ (0,1] 无上溢、深度下溢到 0 为良性（golden 同域）；因式分解式在该域会产生 inf×0=NaN（§10.4-C）——Boundary 用例（Stage 2）若启用因式分解式须重审。
- **NaN/Inf 输入**：上游契约 randn 有限值；Boundary 由 Stage 2 扩展（mask 结构比对）。

### 10.4 被否方案（设计决策记录）

| # | 方案 | 否决理由 |
|---|------|---------|
| A | 忠实 token-parallel（每 block 1 token × block_H heads，对齐上游） | k_j/g_j 平均重读 8.5×（每 (j,h) 对被 subchunk 内 ~8.5 个 token block 各读一次）→ GM 流量 ~3.4GB，@1263GB/s 需 ~2.7ms 仅搬运；且 per-block 动态循环界触发昇腾限制。subchunk 分块语义逐 (i,j) 等价（同一对 (i,j) 的计算式不变），GM 流量降为唯一 IO 593MiB |
| B | chunk 级 block（64 token × head-group） | UB 超限：64 token × BH_V × DK fp32 持久缓冲 ≥ 256KB > 192KB |
| C | 因式分解主路径（e_i·e'_j 锚定 subchunk 首，Aqk_ij = Q'_i⊙K'_j） | 溢出域变化：2^(g_ts−g_j) 在 subchunk 内累计衰减 >128 时上溢 inf×0=NaN（golden 直接式恒良定义）；节省仅 ~30% ops，不足以改变 R1 结论 → 降级为 Stage 3 计算优化选项（附带 Boundary 重审义务） |
| D | Cube GEMM 卸载主路径（因式分解 + Aqk/Akk 行堆叠 [32,DK]@[DK,16] gemm_v0） | 3 项工具链风险：journal #3 多 gemm 组共享 L1 操作数损坏 L0 staging、#2 单 cross-core Pipelined 体限制、小 GEMM（16×16×128）效率未实测；且继承 C 的溢出域变化 → 研究性选项（§9.5-3），不进主设计 |
| E | Expert 模式（gdn 系列风格） | 用户指定 Developer；纯向量无 CV 编排需求，显式同步无收益 |
| F | host 侧预计算 2^g / g·ln2 | 违反 host Buffer 约束（aclnn）；gk 256MB 读+写翻倍，带宽不可行 |

---

## 11. 交付清单

### 11.1 目录结构（沙箱约定：共享目录 + 算子前缀隔离）

```
examples_experiment/dsa_sparse_finetune/
├── DESIGN_fla_chunk_intra_token_parallel.md   # 本设计文档（含 L0 门槛测试计划）
├── proto_fla_chunk_intra_token_parallel.yaml  # 算子接口规格（proto.yaml 角色）
├── ref_gpu/                                   # 冻结参考（只读）
├── fla_chunk_intra_token_parallel.py          # ⬜ 待实现（Stage 2）：@tilelang.jit kernel + host 包装
└── test_fla_chunk_intra_token_parallel.py     # ⬜ 待实现（Stage 2）：golden（§9.1）+ L0（§9.2）+ main 分发器
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN_fla_chunk_intra_token_parallel.md` | ✅ 已完成 | 本文档 |
| `proto_fla_chunk_intra_token_parallel.yaml` | ✅ 已完成（任务书载体） | proto.yaml 角色：`operator.inputs[].dtype` 全集 = {bfloat16, float32}（与 §9.3 精度表一致）、attrs 覆盖 chunk_size/sub_chunk_size/scale、schema/benchmark 完整——经本次设计逐项核对与 §1.3/§4/§9.3 结论一致，沿用不改动 |
| `fla_chunk_intra_token_parallel.py` | ⬜ 待实现 | kernel（§3.3 伪代码落地，Developer 模式，§4.7 pass_configs）+ host 包装 `fla_chunk_intra_token_parallel(q,k,gk,beta,chunk_size,sub_chunk_size)` |
| `test_fla_chunk_intra_token_parallel.py` | ⬜ 待实现 | `prepare_input_kda` + golden（§9.1 逐行落地）+ `check_precision`（precision-standard.md §4.1 原样复制）+ L0 三用例（§9.2，带覆盖标签）+ `--level` main 分发器（模板对齐 tilelang-op-test-design §7） |

### 11.3 命名规范

- kernel 文件 `fla_chunk_intra_token_parallel.py`、测试 `test_fla_chunk_intra_token_parallel.py`（沙箱目录内算子前缀，符合任务硬性约束）
- host 包装函数名对齐上游：`fla_chunk_intra_token_parallel`；JIT kernel 名 `fla_chunk_intra_token_parallel_kernel`

### 11.4 实现顺序

1. ✅ 设计文档 + L0 门槛测试计划（本文档 §9.2，tilelang-op-test-design 场景 A）
2. ⬜ `fla_chunk_intra_token_parallel.py`：载入/cast 链 → 逐对循环（l0_min 先行验证 R3/R8）→ staging 写回
3. ⬜ `test_fla_chunk_intra_token_parallel.py`：golden + L0 三用例 + main
4. ⬜ L0 通过（精度收敛；check_precision 全绿）
5. ⬜ 扩展分层套件 L1/L2/Boundary（tilelang-op-test-design 场景 B）
6. ⬜ perf_bench 哨兵实测（R1/R2 标定：向量吞吐与有效带宽占比，修正 §9.5 推断区间）
