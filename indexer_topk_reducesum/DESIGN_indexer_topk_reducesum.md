# indexer_topk_reducesum 算子设计文档（rev1，first_design）

> 迁移来源：`examples_experiment/dsa_sparse_finetune/ref_gpu/indexer_topk_reducesum.py`（GPU 版主仓参考，只读，语义权威 = `ref_index_score` golden）
> 目标平台：Ascend 910B3（20 AI Core，L0C 128KB / UB 192KB / L1 512KB，硬件参数实测出处：姊妹算子 `examples_experiment/dsa_sparse_finetune/DESIGN.md`（indexer_bwd rev2）§9.4 校准锚点，直接复用并注明）
> 编程模式：**Developer**（用户显式指定，禁止更改；性能不达标时的演进留到 Stage 3）
> 性能目标：kernel 延迟 < 1428us @ B=1, S=2048, H=64, D=128, topk=64, bf16（Stage 3 考核）

---

## 1. 概述

### 1.1 算子名称

indexer_topk_reducesum（DSA sparse finetune lightning indexer 前向 topk+reducesum，varlen 因果）

### 1.2 功能描述

DSA（DeepSeek Sparse Attention）稀疏微调中 lightning indexer 的前向算子：在变长（varlen）序列上，对每个 query token 计算其对全部因果前缀 key 的 indexer score（per-head GEMM → relu → 加权头归约 → sm_scale → 因果 mask），随后取 top-k 个位置的索引与 softmax 概率，输出 `TopkIndices [ΣS, topk] int32` 与 `ReduceSum（topk_score）[ΣS, topk] fp32`。

### 1.3 数学公式（golden = `ref_index_score`，ref_gpu/indexer_topk_reducesum.py:208-229）

每序列（段长 s，段内 query 位置 s1、key 位置 s2，sm_scale = D^-0.5）：

$$\text{logits}[s_1, h, s_2] = \mathrm{einsum}(q, k)\,[s_1, h, s_2] \quad \text{（bf16 round，golden einsum 输出即 bf16）}$$

$$\text{score}[s_1, s_2] = \Big(\sum_{h} \mathrm{relu}(\text{logits}[s_1, h, s_2]) \times \text{weights}[s_1, h] \Big)_{fp32} \times \text{sm\_scale}$$

$$\text{score}[s_1, s_2] = -\infty \quad \text{若 } s_2 > s_1 \text{（因果 mask）}$$

$$\text{topk\_logits}[s_1, :],\ \text{topk\_indices}[s_1, :] = \mathrm{topk}(\text{score}[s_1, :],\ k=\text{topk}) \quad \text{（降序）}$$

$$\text{topk\_score}[s_1, :] = \mathrm{softmax}(\text{topk\_logits}[s_1, :])_{fp32}$$

**数值路径关键事实**（设计决策依据，精读 golden 与 GPU kernel 得出）：

| 环节 | golden（语义权威） | GPU kernel（性能参考） | 本设计（Ascend） |
|------|-------------------|----------------------|-----------------|
| per-head logits | fp32 累加 → **round bf16**（einsum bf16 输出） | fp32 累加，不 round | fp32 累加（L0C）→ **中转 cast bf16**（复刻 golden 的 round 点，§3.2） |
| relu | bf16 上 relu | fp32 上 relu | 中转 copy `enable_relu`（等价 relu(bf16)） |
| ×weights | bf16×bf16 → **bf16 round** → fp32 sum | fp32 乘（bf16 w 提升） | fp32 乘（bf16 w cast fp32，**无乘积 round，比 golden 少一次舍入**） |
| ×sm_scale | sum 后乘（fp32） | **预乘进 q（bf16 round）** | sum 后乘（fp32，与 golden 一致，不预乘 q） |
| 头累加 | fp32（sum dim=-2, dtype=fp32） | fp32（L0C→UB fp32→reduce） | fp32（reduce_sum fp32） |
| softmax | fp32 | fp32 | fp32 |
| topk 平局 | torch.topk 任意序（-inf 尾部不确定） | bitonic 序 | 硬件 Sort 序（-inf 尾部按 pad 位置；输出侧 idx>i_t → -1 修正，与 GPU kernel L180-183 同款） |

→ **本设计精度基线 = 复刻 golden 数值路径**（bf16 logits round + fp32 头累加 + sm_scale 后乘 + fp32 softmax），与 golden 的残差仅剩 golden 独有的 bf16 乘积 round（相对误差 ≤ 2⁻⁸，被 §9.3 bf16/fp32 混合容差轻松覆盖）；比 GPU kernel（全程 fp32 logits、sm_scale 预乘 q）**更贴近 golden**。

### 1.4 算法描述（双路径架构）

计算类型：**Cube+Vector 混合（CV 融合）**；复杂度：**Fusion**（GEMM + relu + 加权归约 + 因果 mask + topk + softmax）。

```text
indexer_topk_reducesum_interface(q, weights, k, topk, offsets, dtype=BF16, dispatch="auto")
  │
  ├─ 数据准备（host，仅新建张量 + 一次 H2D，§10.5 审计）：
  │    token_indices [ΣS, 2]  = prepare_token_indices(offsets)（保持接口行为，GPU 参考同款）
  │    group_table   [num_groups, 3] = (bos, t0_local, len)（QG token/组，段内连续；indexer_bwd
  │                                    BlockIndices 同款先例，examples_experiment/.../DESIGN.md §4.1）
  │    输出分配：TopkIndices torch.zeros int32、ReduceSum torch.zeros fp32
  │
  ├─ 分派检测（host，纯 metadata + 1 次 D2H 标量读回，indexer_bwd host 门控先例）：
  │    max_seg_len = torch.diff(offsets).max().item()
  │    max_seg_len ≤ 4096 → fast（CAP = next_pow2(max_seg_len) ∈ {512,1024,2048,4096} 编译缓存档）
  │    否则 → generic（流式分段 topk，任意段长兜底）
  │
  ├─ fast kernel（tl_indexer_topk_reducesum_fast，persistent GRID=20，本设计主路径）
  │    单遍融合：任务 = QG token 组 → (m,h) 展平 GEMM → bf16 中转+relu → 加权头归约
  │    → 因果 mask → score_row UB [QG, CAP] 常驻 → 每 token 一次硬件 T.tile.topk
  │    → softmax → idx 修正 → 写出
  │
  └─ generic kernel（tl_indexer_topk_reducesum_generic，persistent GRID=20）
       同 GEMM 数据流；score 分段 [QG, CH=1024]，每段硬件 topk → merge_sort 池归并
       （段长 > 4096 时 topk src 缓冲超硬件 repeatTimes≤255 约束，必须分段）
```

**分派正确性论证**：检测为精确布尔判定（max_seg_len 与 4096 比较，非启发式）；fast 在 `max_seg_len ≤ CAP ≤ 4096` 契约下对任意该域输入正确（CAP 编译期常量 ≥ 段长 → score_row 无截断；topk src buffer ≤ 4096 ≤ 8160 硬件上限）；契约外（段长 > 4096）全部由 generic 覆盖（任意段长，分块 + 归并树数学上等价于全行 topk）。`dispatch` 参数（auto/fast/generic）供测试注入（l0_dispatch_equiv 用例）。

### 1.5 数据流图（fast 路径，每任务 = QG 个同段连续 token）

```
GM[IndexQ]  --T.copy 连续行段 [QG*H, D]（整段 if / 逐 token 守卫 else）--> L1(Q_l1)      # bf16，128KB@H=64
GM[Weights] --T.copy--> UB(w_ub)                                        # bf16→fp32
GM[IndexK]  --T.copy 守卫 [BN, D]（尾块 if/else）--> L1(K_l1)           # bf16，8KB

per K 块 j（编译期上界 ceildiv(CAP, BN)，运行时守卫 k_st ≤ t_max）:
  L1(Q_l1) × L1(K_l1)^T --T.gemm_v0(transpose_B=True, init=True)--> L0C(logits [QG*H, BN] fp32, 64KB)
  L0C --T.copy auto-cast bf16--> GM(ws_logits[cid]) --T.copy auto-cast fp32 + enable_relu--> UB(logits_ub)
    # L0C→UB 平台强制 GM 往返（indexer_bwd R8 已验证机制），ws 显式参数化
  per token u（T.serial(QG)，[H, BN] 小 tile 段）:
    UB: logits_ub[u*H:(u+1)*H, :] × broadcast(w_u) --T.tile.mul--> --T.reduce_sum(dim=0)--> score_u [BN]
    UB: score_u × sm_scale --T.tile.mul--> ; 因果 mask（arith_progression + compare + select → -inf）
       --> score_row[u, k_st : k_st+BN]                                  # UB [QG, CAP] fp32 常驻，任务头 fill(-inf)

per token u（K 块循环后，守卫 t0+u < len）:
  UB[score_row[u]] --T.tile.topk(dst, src, K=topk, actual_num=max(i_u+1, topk))--> topk_dst [2*topk]（val,idx 交织）
  topk_dst --T.tile.gather_mask("P0101"/"P1010")--> val_ub [topk] / idx_f_ub [topk]（fp32）
  val_ub: T.reduce_max → T.tile.exp(val - max) → T.reduce_sum → T.tile.div --> softmax_ub [topk] fp32
  idx_f_ub: T.tile.compare(idx ≤ i_u) + select(-1) → T.tile.cast("CAST_ROUND") → idx_ub int32
  UB --T.copy--> GM[TopkIndices[bos+t0+u, :]] / GM[ReduceSum[bos+t0+u, :]]
```

generic 差异（仅 topk 消费侧）：score 段 [QG, CH=1024]（K 块循环嵌在 CH 大段循环内，段边界做段 topk）；每段 `T.tile.topk` 取段前 topk（含 -inf pad）→ `T.tile.merge_sort(池, 段)`（2 路降序归并，稳定）→ 取前 topk 对回池；末段归并完成后池即全行 topk 降序，尾段处理与 fast 完全一致。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Developer（用户显式指定，不得更改）

### 2.2 选型理由

1. CV 融合（GEMM + 向量后处理 + 硬件 topk）且**不需要手动跨核流水**——与 `examples/topk_selector/example_topk_selector.py`（Developer 4 True pass_configs + 动态 grid + `T.tile.topk` 全链路）和 `examples/tile_kernels/moe/moe_topk_gate.py`（Developer + persistent 常量 grid 128 + topk/gather_mask/cast + `pad_value=-inf`）两个直接先例同构。
2. `T.tile.topk/sort/merge_sort/gather_mask/cast` 均为 `T.tile.*` 向量原语（ascend_tile.py:434/487/599/671/2076），在 Developer 模式（无 T.Scope）下由 AUTO_CV_COMBINE 自动分离 C/V 指令——lightning_indexer（Expert）与 topk_selector（Developer）双模式均有先例，Developer 无能力缺口。
3. persistent 常量 grid + 4 True pass_configs 与 indexer_bwd fast（已验证 Developer persistent 先例，DESIGN.md §2.2/§6.1）一致。
4. L0C→UB 中转走 indexer_bwd 已验证的显式 ws 参数 + AUTO_CV_SYNC 按 buffer 名配对 flag 机制（README 偏差 #1/#2，indexer_bwd rev1 最小复现 kernel 验证），无需 Expert 手动 flag。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（编译器映射 L1/UB）+ `T.alloc_fragment`（L0C）；不使用显式 alloc_L1/ub/L0A/L0B/L0C |
| 计算方式 | Cube：`T.gemm_v0`；Vector：`T.tile.mul/broadcast/compare/select/exp/div/reduce_sum/reduce_max/gather_mask/cast/topk/merge_sort/fill/arith_progression`；MTE：`T.copy`（auto-cast / enable_relu / pad_value） |
| 作用域 | 无 `T.Scope`，编译器自动分离 C/V 指令（AUTO_CV_COMBINE） |
| 同步方式 | 全自动（AUTO_SYNC + AUTO_CV_SYNC），零手动 flag；CV 中转 workspace 靠 buffer 名含 "workspace" 由 pass 自动配对插 flag（indexer_bwd rev1 已验证机制） |
| threads | `threads=1`（fast/generic 一致；gqa_sink_bwd:317 / indexer_bwd 先例） |

---

## 3. API 映射设计

### 3.1 公式拆解（fast 路径，每任务 QG=8 token、每 K 块 BN=32）

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `Q_flat = IndexQ[bos+t0 : bos+t0+QG, :, :]` | [QG, H, D] 连续行段，内存布局即 [QG*H, D] 展平（零拷贝重解释，indexer_bwd fast 的 [seq_len, heads*dim] 先例） |
| 2 | `logits[(u,h), n] = Q_flat[u*H+h, :] · K[n, :]` | GEMM：[QG*H, D] × [D, BN]（transpose_B=True），fp32 累加进 L0C——**M=QG*H=512, N=BN=32, K=D=128** |
| 3 | `logits_bf16 = cast_bf16(logits)` | 中转段 1：L0C→GM auto-cast bf16（**复刻 golden 的 bf16 logits round**，§1.3） |
| 4 | `logits_f32 = relu(cast_fp32(logits_bf16))` | 中转段 2：GM(bf16)→UB(fp32) auto-cast + `enable_relu` |
| 5 | `w_u = cast_fp32(Weights[bos+t0+u, :])` | [H] fp32（cast 随 copy） |
| 6 | `score_u[n] = (Σ_h logits_f32[u*H:(u+1)*H, n] × w_u[h]) × sm_scale` | `T.tile.mul`（broadcast w_u 到 [H, BN]）→ `T.reduce_sum(dim=0)` → `T.tile.mul(sm_scale)` |
| 7 | `score_row[u, k_st+i] = score_u[i] if k_st+i ≤ i_u else -inf` | arith_progression(col, k_st, 1, BN) + fill(t_abs) + compare("LE") + select(score, mask, -inf) |
| 8 | `(val[0..topk), idx[0..topk)) = TopK(score_row[u, :CAP], K=topk, actual_num=max(i_u+1, topk))` | **硬件 `T.tile.topk`**（降序，输出交织 (val, idx) 对；idx = score_row 内 0-based 位置 = 段内 s2 位置 ✓ 与 golden 语义一致） |
| 9 | `val ← gather_mask("P0101")`、`idx_f ← gather_mask("P1010")` | 从交织对抽取值/索引（lightning_indexer:118 同款） |
| 10 | `softmax_ub = exp(val - max(val)) / Σexp` | reduce_max + exp + reduce_sum + div，全 fp32 |
| 11 | `idx_out[i] = -1 if idx_f[i] > i_u else idx_f[i]` | compare + select(-1)（GPU kernel L180-183 同款：-inf 尾部位置索引置 -1；golden 在这些位置输出任意 -inf 平局索引，测试用 score>0 mask 域比对，§9.2） |
| 12 | `TopkIndices[bos+t0+u, :] = cast_int32(idx_out)`；`ReduceSum[bos+t0+u, :] = softmax_ub` | 行级 T.copy 写出 |

### 3.2 TileLang API 映射（GPU → Ascend；全部附出处）

| 步骤 | GPU 版 API（ref_gpu 行号） | Ascend API | 证据（skill 文档 / examples / 源码） |
|------|---------------------------|-----------|------|
| Kernel 声明 | `T.Kernel(seq_len, threads=128)`（L83） | `T.Kernel(GRID=20, threads=1, is_npu=True) as (cid)`（persistent；GRID 编译期常量，host 查 cube_core_num） | gqa_sink_bwd:317；sparse_mla:262/1302；moe_topk_gate:38（Developer persistent grid=128 先例）；indexer_bwd DESIGN §3.2 |
| 任务轮询 | 每 token 一个 block（隐式） | `ntask = T.max(0, T.ceildiv(total - cid, GRID))` + `T.serial(ntask)` + `pid = k*GRID + cid`（striped 轮询） | sparse_mla:261/309（generic/panel 双变体实测）；indexer_bwd DESIGN R10 |
| 动态 shape | `T.dynamic("seq_len"/"batch_plus_one")`（L35-36） | `T.symbolic("seq_len")` + `T.symbolic("batch_plus_one")` + `T.symbolic("num_groups")` | sparse_mla:104-106；lightning_indexer_dynamic:18-20；indexer_bwd DESIGN §3.2 |
| 任务元数据 GM 标量读 | `TokenIndices[bx,0/1]`、`Offsets[i_b]`（L84-85） | `GroupTable[bx, 0..2]`（bos, t0, len；每任务 3 次标量读） | sparse_mla:289-291；indexer_bwd DESIGN §3.2（TokenIndices/BlockIndices 实测） |
| Q 装载 | `T.copy(IndexQ[bos+i_t, :, :], index_q_shared)`（L96，逐 token [H, D]） | `T.copy(IndexQ[bos+t0, 0:QG*H*D], Q_l1)`（连续行段 [QG*H, D] 整段）；尾组 if/else 逐 token 守卫 | GM→L1 直连整段：gqa_sink:390；if/else 双路：indexer_bwd DESIGN §5.3（unsorted:42 动态 if 先例） |
| Weights 装载 | `T.copy(Weights[bos+i_t, :], weights_frag)`（L100） | `T.copy(Weights[bos+t0, 0:QG*H], w_ub)` + cast fp32（尾组逐行守卫） | T.copy auto-cast：indexer_bwd DESIGN §3.2（dk cast kernel） |
| K 块装载 | `T.Parallel` 逐元素 if_then_else（L112-113） | `T.copy(IndexK[bos+k_st : bos+k_ed, :], K_l1)` 整段 if / 逐行守卫 else（尾块） | unsorted:42；indexer_bwd DESIGN §5.3 |
| GEMM | `T.gemm(index_k_shared, index_q_shared, logits, transpose_A=False, transpose_B=True, clear_accum=True)`（L117-124） | `T.gemm_v0(Q_l1, K_l1, logits_l0c, transpose_B=True, init=True)`（A=Q_flat [QG*H, D] 不转置、B=K [BN, D] 转置读；bf16 操作数 + fp32 C） | ascend.py:418-463（bf16 L434、fp32 累加 L439）；gqa_sink:400；lightning_indexer:50（同款 transpose_B） |
| **中转 1（L0C→GM cast bf16）** | —（GPU shared memory 直接读） | `T.copy(logits_l0c, ws_logits[cid, :])`（fp32→bf16 auto-cast） | gqa_sink:478（L0C→GM fp32→fp16 auto-cast 直写先例）；indexer_bwd DESIGN §3.2（dq 直写行） |
| **中转 2（GM→UB + relu）** | `T.gemm` 后 fragment 直读 | `T.copy(ws_logits[cid, :], logits_ub, enable_relu=True)`（bf16→fp32 auto-cast + relu；cast 与 relu 同 copy 待验证，fallback 见 R7） | copy_op.py:260（enable_relu 参数）；enable_relu 先例：lightning_indexer:52-58（L0C→GM）、indexer_bwd DESIGN §3.2（GM→UB 段 2） |
| 权重广播乘 | `logits[i,j] = T.max(logits[i,j], 0) * weights_frag[j]`（L128，标量循环） | `T.tile.broadcast(w_2d, w_1d, axis=1)` + `T.tile.mul(logits_ub[u*H:(u+1)*H, :], logits_seg, w_2d)`（向量化 [H, BN] 段） | broadcast：gqa_sink:416（ascend_tile.py:2451）；mul BufferRegion：ascend_tile.py:1089-1146（签名支持 BufferRegion） |
| 头归约 | `T.reduce_sum(logits, logits_sum, dim=1)`（L132，[block_K, heads]→[block_K]） | `T.reduce_sum(logits_seg [H, BN], score_u [BN], dim=0, clear=True)`（dim=0 行归约 [H 行→1 行]） | reduce_ascend.py:452-500（dim 0/1/-1/-2 + clear）；indexer_bwd DESIGN §3.2（dw 列归约 dim=1 / generic dim=0 先例） |
| sm_scale 乘 | 预乘 q（L103-104，bf16 round） | `T.tile.mul(score_u, score_u, sm_scale)`（**sum 后乘，与 golden 一致**，§1.3） | T.tile.mul scalar 右操作数：ascend_tile.py:1130-1146 |
| 因果 mask | `if k_st + i > i_t: logits_sum[i] = -inf`（L141-143，标量） | `T.tile.arith_progression(col_1d, k_st, 1, BN)` + `T.tile.fill(tval, i_u)` + `T.tile.compare(causal, col_1d, tval, "LE")` + `T.tile.select(score_u, causal, score_u, -inf, "VSEL_TENSOR_SCALAR_MODE")` | arith_progression：ascend_tile.py:398、gqa_sink:393；compare：gqa_sink:419；select 标量 else：ascend_tile.py:879-898、indexer_bwd DESIGN §3.2 |
| score 行累积 | N=2*topk 环形缓冲 + bitonic_sort 维持（L50-71/141-152） | **`score_row[u, k_st:k_st+BN] = score_u`（UB 直写区域）+ 任务尾一次性硬件 topk**（环形缓冲+bitonic 整体废弃） | T.tile.topk：ascend_tile.py:599-668；Developer 全链路先例：topk_selector:48-50、moe_topk_gate:66-69 |
| **topk** | bitonic 排序网络（28 pass × N=128，L56-71） | `T.tile.topk(topk_dst, score_row[u], K=topk, actual_num=T.max(i_u+1, topk))`（降序，(val, idx) 交织输出；actual_num 运行时标量） | ascend_tile.py:599（"actual_num can be symbolic"）；symbolic actual_num 先例：lightning_indexer_dynamic:119（S2 symbolic）；fp32 index 精确性：test_topk.py:145-146（fp32 index 与 ref 精确比对） |
| val/idx 抽取 | 环形缓冲前 topk 直读（L158/185） | `T.tile.gather_mask(val_ub, topk_dst, "P0101")`（偶位 val）/ `T.tile.gather_mask(idx_f_ub, topk_dst, "P1010")`（奇位 idx） | ascend_tile.py:671-729（7 种固定 pattern）；lightning_indexer:118（P1010 抽 idx）；moe_topk_gate:68 |
| softmax | reduce_max → exp → reduce_sum → div（L154-174，fragment） | `T.reduce_max(val_ub, max_ub, dim=-1)` + `T.tile.exp(e_ub, val_ub - max)` + `T.reduce_sum(e_ub, sum_ub, dim=-1)` + `T.tile.div(softmax_ub, e_ub, sum_ub[0])` | reduce_max：reduce_ascend.py:362；exp：ascend_tile.py:1255（sparse_mla:641 T.tile.exp 先例）；div：ascend_tile.py:1147 |
| idx 修正 | `if topk_index_shared[i] > i_t: = -1`（L180-183） | `T.tile.compare(valid, idx_f_ub, tval, "LE")` + `T.tile.select(idx_f_ub, valid, idx_f_ub, -1.0, "VSEL_TENSOR_SCALAR_MODE")` | gqa_sink:417-422；ascend_tile.py:879-898 |
| idx cast int | —（shared int32） | `T.tile.cast(idx_ub, idx_f_ub, "CAST_ROUND", topk)`（fp32 → int32；-1.0 → -1） | ascend_tile.py:2076；lightning_indexer:120；moe_topk_gate:69 |
| 输出写出 | `T.copy(topk_index_shared[:topk], TopkIndices[...])`（L185-186） | `T.copy(idx_ub, TopkIndices[bos+t0+u, :])` + `T.copy(softmax_ub, ReduceSum[bos+t0+u, :])` | 行级写出：indexer_bwd DESIGN §3.2；host torch.zeros 输出：unsorted:247 |
| 【generic 归并】 | bitonic（同上） | 段 topk 后 `T.tile.merge_sort(merged, pool, seg)`（2 路降序稳定归并）+ `T.copy(merged[0:2*topk], pool)`（取前 topk 对） | ascend_tile.py:487-596（2-4 路，稳定，fp32 only ✓）；examples/sort/example_merge_sort.py；merge_sort dst/src 均为交织对格式 |
| fill(-inf) | `T.fill(topk_value_shared, float("-inf"))`（L92） | `T.tile.fill(score_row, -inf)`（任务头全量 fill，QG×CAP） | ascend_tile.py:352；moe_topk_gate:56（copy pad_value=-inf 同语义先例） |
| 流水线 | `T.Pipelined(num_blocks, num_stages=0)`（L107） | 不使用（num_stages=0 本无流水；边界含运行时标量，ascend-constraints §1） | ascend-constraints §1 |
| 【generic 段 topk】 | — | `T.tile.topk(seg_dst, score_seg[u], topk, T.max(seg_valid, topk))` | 同 topk 行 |
| tmp（topk/sort/reduce 显式备用） | — | 默认 auto 注入（InjectTmpBuffer）；fallback 显式 `tmp=` | explicit_tmp 测试：reduce/sort/merge_sort/topk/select 均支持 tmp=（test_tilelang_ascend_language_explicit_tmp.py:89-107）；auto 注入 + MEMORY_PLANNING 共享：lightning_indexer:11-13 文件头注释 |

### 3.3 计算伪代码（fast kernel）

```python
GRID  = 20                  # persistent 编译期常量（host 查 cube_core_num，sparse_mla:1501）
QG, BN = 8, 32              # token/组、K 块宽（CAP=4096 档 QG=4，§5.2）
CAP   = 2048                # score_row 宽（编译期档位 512/1024/2048/4096；≤4096 = topk src 硬件 8160 上限内）
NBLK  = T.ceildiv(CAP, BN)  # K 块循环编译期常量上界（CAP 由 host 分派保证 ≥ 段长）
seq_len, batch_plus_one = T.symbolic("seq_len"), T.symbolic("batch_plus_one")
num_groups = T.symbolic("num_groups")

@tl.jit(pass_configs=4_True_Developer)   # §4.7
def tl_indexer_topk_reducesum_fast(
    IndexQ,                    # [seq_len, heads*dim] bf16（二维声明，连续行段零拷贝，indexer_bwd 先例）
    Weights,                   # [seq_len, heads] bf16
    IndexK,                    # [seq_len, dim] bf16
    TopkIndices, ReduceSum,    # [seq_len, topk] int32 / fp32（host torch.zeros）
    Offsets,                   # [batch_plus_one] int32
    TokenIndices,              # [seq_len, 2] int32（接口行为保持；本 kernel 不消费，generic 同）
    GroupTable,                # [num_groups, 3] int32 (bos, t0, len)（host 预计算）
    ws_logits,                 # [GRID, QG*heads, BN] bf16（CV 中转，显式参数，名含 workspace）
    heads, dim, topk, sm_scale, CAP,
):
    with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
        total = num_groups
        ntask = T.max(0, T.ceildiv(total - cid, GRID))            # sparse_mla:309 形态
        # ---- 缓冲分配（§4.3；Developer 编译器映射层级）----
        Q_l1      = T.alloc_shared([QG*heads, dim], bf16)          # L1（H=64: 128KB）
        K_l1      = T.alloc_shared([BN, dim], bf16)                # L1（8KB）
        logits_l0c = T.alloc_fragment([QG*heads, BN], fp32)        # L0C（64KB）
        logits_ub = T.alloc_shared([QG*heads, BN], fp32)           # UB（64KB）
        score_row = T.alloc_shared([QG, CAP], fp32)                # UB（64KB@CAP=2048）
        w_ub      = T.alloc_shared([QG*heads], fp32)               # UB（2KB）
        col_1d/w_1d/w_2d/score_u/tval/causal 等 [BN]/[heads, BN] 小 tile  # UB（~10KB）
        topk_dst  = T.alloc_shared([2*topk], fp32)                 # UB（512B；交织 (val,idx)）
        val_ub/idx_f_ub/softmax_ub/e_ub [topk] fp32；max_ub/sum_ub [1]    # UB（~1.5KB）
        idx_ub    = T.alloc_shared([topk], int32)

        for k in T.serial(ntask):
            bx = k * GRID + cid
            bos, t0, ln = GroupTable[bx, 0], GroupTable[bx, 1], GroupTable[bx, 2]
            t_max = T.min(t0 + QG, ln) - 1                          # 组内最大有效 token 位置
            # ---- 装载（尾组 if/else 双路守卫）----
            if t0 + QG <= ln:
                T.copy(IndexQ[bos+t0, 0 : QG*heads*dim], Q_l1)      # 连续行段 GM→L1
                T.copy(Weights[bos+t0, 0 : QG*heads], w_ub_src)     # bf16
            else:
                for u in T.serial(QG):
                    if t0 + u < ln:
                        T.copy(IndexQ[bos+t0+u, 0 : heads*dim], Q_l1[u*heads : (u+1)*heads, :])
                        T.copy(Weights[bos+t0+u, 0:heads], w_ub_src[u*heads : (u+1)*heads])
            T.copy(w_ub_src, w_ub)                                   # bf16→fp32 cast 随 copy
            T.fill(score_row, -inf)                                  # 任务头全量（topk pad 语义，§5.4）
            # ---- K 块循环（编译期上界 + 运行时守卫；空转 = 单 if 判断）----
            for j in T.serial(NBLK):
                k_st = j * BN
                if k_st <= t_max:
                    k_ed = T.min(k_st + BN, ln)
                    if k_st + BN <= ln:                              # K 尾块守卫
                        T.copy(IndexK[bos+k_st : bos+k_st+BN, :], K_l1)
                    else:
                        for i in T.serial(BN):
                            if k_st + i < ln:
                                T.copy(IndexK[bos+k_st+i, 0:dim], K_l1[i, :])
                    # GEMM: [QG*H, D] × [D, BN] → L0C fp32
                    T.gemm_v0(Q_l1, K_l1, logits_l0c, transpose_B=True, init=True)
                    # CV 中转两段（平台强制 L0C→UB GM 往返，显式 ws）
                    T.copy(logits_l0c, ws_logits[cid, :])            # fp32→bf16（golden round 点）
                    T.copy(ws_logits[cid, :], logits_ub, enable_relu=True)  # bf16→fp32 + relu
                    # per-token 向量段 [heads, BN]
                    for u in T.serial(QG):
                        t_abs = t0 + u
                        T.copy(w_ub[u*heads : (u+1)*heads], w_1d)
                        T.tile.broadcast(w_2d, w_1d, axis=1)         # [H, BN]
                        T.tile.mul(logits_seg, logits_ub[u*heads : (u+1)*heads, :], w_2d)
                        T.reduce_sum(logits_seg, score_u, dim=0, clear=True)   # [BN]
                        T.tile.mul(score_u, score_u, sm_scale)       # sum 后乘（golden 序）
                        T.tile.arith_progression(col_1d, k_st, 1, BN)
                        T.tile.fill(tval, t_abs)
                        T.tile.compare(causal, col_1d, tval, "LE")   # k_st+i <= i_u
                        T.tile.select(score_u, causal, score_u, -inf, "VSEL_TENSOR_SCALAR_MODE")
                        T.copy(score_u, score_row[u, k_st : k_st+BN])
            # ---- per-token topk + softmax + 写出 ----
            for u in T.serial(QG):
                t_abs = t0 + u
                if t_abs < ln:
                    i_u = t_abs                                       # 段内位置 = 因果上界
                    n_eff = T.max(i_u + 1, topk)                      # K ≤ actual_num 保护（i_u+1<topk 短 token）
                    T.tile.topk(topk_dst, score_row[u, :], topk, n_eff)
                    T.tile.gather_mask(val_ub,  topk_dst, "P0101")    # 偶位 = val
                    T.tile.gather_mask(idx_f_ub, topk_dst, "P1010")   # 奇位 = idx
                    # softmax（fp32；-inf 位置 exp=0，与 golden 一致）
                    T.reduce_max(val_ub, max_ub, dim=-1)
                    T.tile.sub(e_ub, val_ub, max_ub[0])
                    T.tile.exp(e_ub, e_ub)
                    T.reduce_sum(e_ub, sum_ub, dim=-1)
                    T.tile.div(softmax_ub, e_ub, sum_ub[0])
                    # idx 修正（-inf 尾部/越界 → -1，GPU kernel L180-183 同款）
                    T.tile.compare(valid, idx_f_ub, tval, "LE")
                    T.tile.select(idx_f_ub, valid, idx_f_ub, -1.0, "VSEL_TENSOR_SCALAR_MODE")
                    T.tile.cast(idx_ub, idx_f_ub, "CAST_ROUND", topk)
                    T.copy(idx_ub, TopkIndices[bos+t_abs, :])
                    T.copy(softmax_ub, ReduceSum[bos+t_abs, :])
```

generic kernel 伪代码差异（topk 消费侧）：

```python
CH = 1024                                    # score 分段宽（静态 UB [QG, CH]）
# K 块循环嵌在 seg 大段循环内：for sg in T.serial(NSEG=ceildiv(CAP_max, CH))（上界按 8160 硬件上限
#   换算的编译期常量段数档；段守卫 sg*CH <= i_u_max_group）
#   每 CH 段：与 fast 相同的 K 块产出 → score_seg[u, 0:CH] → 段 topk：
#       seg_valid = T.max(T.min(CH, i_u+1 - sg*CH), topk)   # 末段短/短 token 保护
#       T.tile.topk(seg_dst, score_seg[u, :], topk, seg_valid)
#   池归并（池初始 -inf + idx=-1 的 topk 对）：
#       T.tile.merge_sort(merged, pool, seg_dst)             # 2 路降序稳定归并（fp32 only ✓）
#       T.copy(merged[0 : 2*topk], pool)                     # 取前 topk 对回池
# 末段后池 = 全行 topk 降序 → 与 fast 完全相同的 softmax/修正/写出尾段
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| `T.tile.topk`（fp32 src / symbolic actual_num / (val,idx) 交织输出 / K=1..actual_num 边界 / src in-place pad -inf） | ascend_tile.py:599-668；test_tilelang_ascend_language_topk.py（ascendc：actual_num 51 非对齐、K=1/64、4096 大小、in-place 验证） | 仓测试已验证；symbolic actual_num 有 lightning_indexer_dynamic:119 先例 |
| `T.tile.merge_sort`（2-4 路 / 降序稳定 / 交织对格式 / fp32 only / 1D slice 或整行 slice） | ascend_tile.py:487-596 | examples/sort/example_merge_sort.py + 仓测试（test_merge_sort.py） |
| `T.tile.gather_mask`（P0101/P1010 固定 pattern） | ascend_tile.py:671-729 | lightning_indexer:118、moe_topk_gate:68（P1010）、explicit_tmp 测试（P0101） |
| `T.tile.cast("CAST_ROUND", count)` | ascend_tile.py:2076 | lightning_indexer:120、moe_topk_gate:69 |
| `T.gemm_v0`（bf16 操作数 / fp32 C / transpose_B / init） | ascend.py:418-463 | gqa_sink:400、lightning_indexer:50、indexer_bwd 实测 |
| `T.copy`（GM→L1 整段 / auto-cast / enable_relu / 动态 if 内 / L0C→GM） | copy_op.py:83/260 | gqa_sink:390/478、lightning_indexer:52-58、indexer_bwd 实测；**cast+relu 同 copy 待验证（R7，有 fallback）** |
| `T.reduce_sum(buffer|region, out, dim=0/-1, clear=)` / `T.reduce_max` | reduce_ascend.py:362/452 | gqa_sink:386、indexer_bwd 实测（dim=0/1） |
| `T.tile.broadcast / compare / select(标量) / mul / sub / div / exp / fill / arith_progression` | ascend_tile.py:2451/1979/770/1130/1106/1147/1255/352/398 | gqa_sink:393-423、sparse_mla:626-643、indexer_bwd 实测 |
| `T.Kernel` 常量 grid persistent + `T.max/T.ceildiv` + `T.serial` 运行时边界 | sparse_mla:262/309 | sparse_mla generic/panel 双变体实测 |
| `T.symbolic` 三动态维 | tilelang/language/__init__.py:89 | sparse_mla:104-106、lightning_indexer_dynamic:18-20 |
| GM 标量读（GroupTable[bx, 0..2]） | TIR BufferLoad | sparse_mla:289-291、indexer_bwd BlockIndices 实测 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（ascend-constraints §1/§2 强制检测）

| 约束 | 是否涉及 | 处理方案 |
|------|---------|----------|
| 不支持三维 Kernel | No | fast/generic 均一维 persistent grid（常量 20） |
| threads 参数限制（仅 1/2；GPU 参考为 128） | **Yes** | `threads=1`（两 kernel 一致；gqa_sink/indexer_bwd 先例）。GPU 的 128 threads 用于 block 内 bitonic 并行——本设计以向量化 tile 指令 + 硬件 Sort 替代，无线程语义需求 |
| 动态循环边界（GPU `T.Pipelined(num_blocks)`，num_blocks 依赖 i_t 每 token 不同） | **Yes（核心改造点）** | ① K 块循环上界 = `T.ceildiv(CAP, BN)` **编译期常量**（CAP 为构建参数，host 分派保证 ≥ 段长）+ 运行时守卫 `if k_st <= t_max`（t_max 来自 GroupTable GM 标量读，if 条件内 GM 标量先例：indexer_bwd generic `if pos <= i_t`）；② topk/softmax 段循环 `T.serial(QG)` 静态；③ persistent 外层 `T.serial(ntask)` 运行时边界（ntask 依赖 symbolic 维，sparse_mla:309 已验证形态——不依赖 tensor 数据值） |
| 流水线不支持动态边界 | Yes | 不使用 T.Pipelined（GPU 参考 num_stages=0 本无流水）；引擎级重叠由 AUTO_CV_COMBINE 自动排布 |
| L0C 容量上限 128KB | **Yes** | fast：logits [QG*H=512, BN=32] fp32 = **64KB < 128KB**（CAP=4096 档 QG=4 → [256,32] = 32KB）；generic 同。单 fragment 无共存（§4.5 保守口径） |
| T.copy 列方向 strided 切片 | No | 所有 GM 切片最内维连续（Q/K/Weights/输出行段）；logits_ub 的 [u*H:(u+1)*H, :] 是 **UB region**（buffer 内切片，非 GM），向量算子签名支持 BufferRegion（ascend_tile.py:1089-1146） |
| GPU 专用 API | **Yes** | `T.gemm`→`T.gemm_v0`；`T.sync_threads`（bitonic 内 28 处）→ 整体废弃（单核向量化 + 硬件 Sort 无跨线程同步）；`T.alloc_shared` int32 环形缓冲 → UB score_row；`T.dynamic`→`T.symbolic`。差异全表见 §3.5.2 |
| Host 侧 Buffer 操作 | Yes | §10.5 审计：仅新建辅助张量（CPU 构造 + H2D）+ torch.zeros 输出 + 1 次标量 D2H；无 permute/reshape 物理化、无 aclnn |

#### 3.5.2 参考实现差异说明（相对 ref_gpu，允许语义保真的重新架构）

| 差异项 | GPU 版（ref_gpu 行号） | 本设计（Ascend） | 语义保真论证 |
|--------|----------------------|-----------------|-------------|
| **topk 维持结构** | N=2*topk 环形缓冲 + bitonic 排序网络（L45-71：log²(128)=28 pass × 每 pass T.sync_threads；每 topk 新元素触发一次，L149-152） | **UB score_row [QG, CAP] 全行累积 + 任务尾单次硬件 `T.tile.topk`**（generic：CH 分段 topk + merge_sort 稳定归并） | 硬件 Sort 为降序全排序取前 K，数学上等价于"增量维持 top-k"（集合与次序均唯一，非平局时）；平局与 -inf 尾部语义见 §9.2 比对策略（score>0 mask 域 + idx>i_t→-1 修正，与 GPU kernel L180-183 同款） |
| Kernel 拓扑 | 单 kernel，grid=seq_len，每 token 一个 block + threads=128 | 双 kernel（fast/generic）+ persistent GRID=20 + host 分派（indexer_bwd 双路径先例） | 任务粒度 QG=8 token/组， striped 轮询均衡因果三角负载（§6.1） |
| GEMM 形态 | [block_K=32, D] × [D, H] per token per 块（M=32 小矩阵，65K 次） | **(m,h) 展平**：[QG*H=512, D] × [D, BN=32] per 组 per 块（8320 次，M=512 大矩阵，固定开销摊薄 16×） | 数学恒等：logits[(u,h), n] = Q[u,h,:]·K[n,:]；GEMM 累加顺序变化仅在 fp32 内（结合律，误差 ≤ 加法重排，被容差覆盖） |
| logits 精度 | fp32 直读（不 round） | **中转 cast bf16**（复刻 golden einsum 的 bf16 round） | §1.3 数值基线：与 golden 逐 round 点对齐（比 GPU 版更贴 golden） |
| sm_scale | 预乘 q（bf16 round，L103-104） | sum 后乘（fp32，golden 序） | §1.3：消除 GPU 版的额外 bf16 round，与 golden 完全一致 |
| 因果 mask | 标量 if 逐元素（L142-143） | arith_progression + compare + select 向量化 | 逐位等价（gqa_sink:416-422 同款链） |
| 动态 K 块数 | `num_blocks = T.ceildiv(i_t+1, 32)`（L86，tensor 值循环边界） | 编译期上界 NBLK=CAP/BN + `if k_st <= t_max` 守卫 | 空转块仅 1 次 if 判断（§6.1 量化：≤4-8us）；守卫内不发射 GEMM/拷贝，结果与精确边界一致 |
| 填充语义 | 环形缓冲初始化 -1/-inf（L91-92） | score_row fill(-inf) + topk actual_num=max(i_u+1, topk) + idx>i_u→-1 | i_u+1 < topk 时尾部输出 -inf val + idx=-1；golden 在这些位置输出任意 -inf 平局索引但 score=0 → mask 域比对不受影响（§9.2） |

#### 3.5.3 本项目同类实现参考（同类引用，门禁项）

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/lightning_indexer/example_lightning_indexer.py` + `_dynamic_shape.py` | **最高（同族算子）** | DSA indexer 前向 Ascend 版：`T.tile.topk`（L116/119，dynamic 版 actual_num=S2 symbolic）+ `gather_mask("P1010")`（L118）+ `cast("CAST_ROUND")`（L120）+ score_accum 全行累积后单次 topk（L67-116 同款"accumulate then topk once"策略）+ GEMM→relu→w 乘→G 维 reduce→topk 完整数据流 + MEMORY_PLANNING 让 auto-injected sort/topk tmp 共享 UB（L11-13）。差异：该示例为 Expert 模式、非 varlen、无因果 mask、QK_RES 全量物化（[B,N2,S1,G,S2] fp32，本设计改为融合中转避免 1GB 物化） |
| `examples/topk_selector/example_topk_selector.py` | **高度相似** | **Developer 模式**（4 True pass_configs）+ 动态 grid + `T.tile.topk`/`gather_mask`/`cast` 全链路（L48-50）+ score_accum [N] 静态 UB——本设计 fast 路径的直系模板 |
| `examples/tile_kernels/moe/moe_topk_gate.py` | 高 | **Developer persistent**（grid=128 常量 + my_iters 轮询 + ping-pong，L38-50）+ `pad_value=-inf` copy（L56）+ topk/gather_mask/cast（L66-69）+ `T.min` clamp 寻址 |
| `examples/dsa_sparse_finetune/sparse_mla_topk_reducesum.py` | 高（同族姊妹算子） | persistent GRID=20 + `ntask=T.max(0,T.ceildiv(...))` striped 轮询（L261-262）+ symbolic 三动态维（L111-113）+ GM 标量读（L289-291）+ mask 算术链 clamp/compare（L626-632）；其 `T.tile.exp`（L641）先例 |
| `examples/attention_sink/example_gqa_sink_bwd_bhsd/example_gqa_sink_bwd_bhsd.py` | 高 | Developer CV 融合全链路：gemm_v0 + arith_progression/broadcast/compare/bitwise_and/select 向量化 mask（L393-423）+ GM→L1 直拷（L390）+ L0C→GM 直写 auto-cast（L478）+ reduce_sum dim=-1（L386） |
| `examples_experiment/dsa_sparse_finetune/DESIGN.md`（indexer_bwd rev2） | **方法学蓝本** | 910B3 实测硬件锚点（§9.4：L2 ~1TB/s 中转口径、gemm ~500-1500cy、短 DMA 100-200cy、聚合带宽 510GB/s）；Developer 4 True + threads=1；L0C→UB 显式 ws 中转机制（rev1 已验证）；host 分派/GroupTable（BlockIndices）先例；双路径架构范式 |
| `examples/sort/example_merge_sort.py` | 中 | merge_sort 2-4 路归并用法（generic 路径池归并） |
| `examples/lightning_indexer/...`（testing） | 支撑 | `testing/python/language/test_tilelang_ascend_language_topk.py`：fp32 index 精确性断言（L145-146）、repeatTimes≤255（buffer ≤8160）硬件上限注释（L390-394）、actual_num 非对齐 pad 行为 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量（接口签名保持不变）

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| q（IndexQ） | [ΣS, H, D] | bfloat16 | fast kernel 内按 [ΣS, H*D] 二维零拷贝解释（连续行段，indexer_bwd 先例） |
| weights（Weights） | [ΣS, H] | bfloat16 | |
| k（IndexK） | [ΣS, D] | bfloat16 | |
| offsets | [B+1] | int32 | varlen cu_seqlens；段长 ≥ topk（golden assert，host 复刻） |
| topk（attr） | — | int | 2 的幂（主仓 kernel assert L31）；topk ≤ 段长 |
| TokenIndices（host 派生） | [ΣS, 2] | int32 | prepare_token_indices 生成（接口行为保持，GPU 参考同款） |
| GroupTable（host 派生，新增） | [num_groups, 3] | int32 | (bos, t0_local, len)；num_groups = Σ_b ceildiv(len_b, QG)；per 组一行，fast/generic 共用（indexer_bwd BlockIndices 同款） |

### 4.2 输出张量（host `torch.zeros` 预分配，每 token 行独占写，无原子）

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| TopkIndices | [ΣS, topk] | int32 | 值域 [-1, 段长)；-1 = -inf 尾部位置（GPU kernel 同款语义） |
| ReduceSum（topk_score） | [ΣS, topk] | float32 | softmax(topk_logits) fp32；-inf 位置 = 0 |

### 4.3 中间缓冲区（fast kernel，QG=8, H=64, D=128, BN=32, topk=64, CAP=2048）

| Buffer 名 | Shape | dtype | 层级 | 大小 |
|-----------|-------|-------|------|------|
| Q_l1 | [512, 128] | bf16 | L1（alloc_shared） | 128KB |
| K_l1 | [32, 128] | bf16 | L1 | 8KB |
| logits_l0c | [512, 32] | fp32 | L0C（alloc_fragment） | **64KB** |
| logits_ub | [512, 32] | fp32 | UB | 64KB |
| score_row | [8, 2048] | fp32 | UB | 64KB |
| w_ub（bf16 源 + fp32） | [512] | fp32 | UB | 2KB（+1KB bf16 源） |
| per-u 工作集（w_1d/w_2d [64,32]/logits_seg/score_u/col_1d/tval/causal） | [64,32] 级 | fp32 | UB | ~10KB |
| topk_dst | [128] | fp32 | UB | 512B |
| val_ub/idx_f_ub/e_ub/softmax_ub | [64] | fp32 | UB | 1KB |
| max_ub/sum_ub | [1] | fp32 | UB | 8B |
| idx_ub | [64] | int32 | UB | 256B |
| sort/topk auto tmp | 注入 | — | UB | 编译器分配（MEMORY_PLANNING 共享；显式 fallback 见 R1） |

generic kernel（CH=1024, QG=8）：score_seg [8, 1024] fp32 32KB + logits_ub 64KB + 池 pool [2*topk 对 = 256 fp32 元素] 1KB + merge dst [4*topk] 2KB + 段 topk dst [2*topk] 1KB + per-u 同 fast ≈ **112KB**。

### 4.4 内存搬运路径（无跨级违规）

```
GM[IndexQ]  --T.copy 连续行段 [QG*H, D]--> L1(Q_l1)                    # GM→L1 直连（gqa_sink:390）
GM[Weights] --T.copy--> UB(w_src bf16) --T.copy cast--> UB(w_ub fp32)
GM[IndexK]  --T.copy 守卫--> L1(K_l1)
L1(Q_l1) × L1(K_l1)^T --T.gemm_v0(内部 L1→L0A/L0B)--> L0C(logits fp32)
L0C --T.copy auto-cast bf16--> GM(ws_logits) --T.copy auto-cast fp32 + enable_relu--> UB(logits_ub)
    # 平台强制 L0C→UB GM 往返（indexer_bwd R8），显式参数化规避 auto workspace 动态 grid 限制
UB: 向量链（mul/reduce/mul/mask）--> UB(score_row)                      # 全程 UB，不落 GM
UB(score_row[u]) --T.tile.topk（Vector 硬件指令）--> UB(topk_dst)
UB --gather_mask/cast/softmax--> UB(val/idx/softmax)
UB(idx_ub / softmax_ub) --T.copy--> GM[TopkIndices / ReduceSum 行]
```

无 GM→L0 直搬；Cube 操作数全经 L1；Vector 数据在 UB；唯一 GM 往返 = 平台强制的 L0C→UB 中转（已显式参数化并计入预算 §9.4）。✓ 符合硬件内存层级。

### 4.5 内存预算汇总（保守口径：无 liveness 复用假设）

| 层级 | fast（CAP=2048/QG=8） | fast（CAP=4096/QG=4） | generic（CH=1024/QG=8） | 容量 |
|------|----------------------|----------------------|------------------------|------|
| L0C | **64KB**（[512,32] fp32） | 32KB（[256,32]） | 64KB | 128KB ✓ |
| UB | **~145KB**（64 score_row + 64 logits + 10 per-u + ~5 杂项 + tmp） | ~110KB（64+32+…） | ~112KB | 192KB ✓（余量 ≥40KB） |
| L1 | 136KB（Q 128 + K 8） | 72KB（Q 64 + K 8） | 136KB | 512KB ✓ |

**CAP-QG 联动规则**（编译期常量参数对，_BUILD_CACHE 缓存键组成部分）：CAP ≤ 2048 → QG=8；CAP=4096 → QG=4（score_row [4,4096]=64KB + logits [256,32]=32KB，避免 UB 192KB 恰满）。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| seq_len（ΣS） | `T.symbolic("seq_len")` | topk ≤ ΣS，典型 ≤ 64K |
| batch_plus_one | `T.symbolic("batch_plus_one")` | 2 ~ 1025 |
| num_groups | `T.symbolic("num_groups")` | Σ_b ceildiv(len_b, QG)，GroupTable 维与 persistent 轮询上界 |

grid 为编译期常量 20（host 查 cube_core_num）；无动态 grid（规避 auto workspace 限制）。

### 4.7 JIT 配置

```python
@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # C/V 自动分核合并
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # CV 交叉自动同步（ws_logits 配对 flag）
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # MTE/Vector 自动同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # UB 规划 + auto-injected sort/topk tmp 共享
})                                                            # 4 True 与 topk_selector:24-27 / indexer_bwd §4.7 一致
# 无 out_idx / workspace_idx：输出与 ws_logits 全显式参数（indexer_bwd 方案，规避 auto workspace 动态 grid 限制）
# 编译缓存键：(mode=fast/generic, H, D, topk, QG, CAP/CH)——host 按 max_seg_len 选 CAP 档，_BUILD_CACHE 复用
```

---

## 5. Tiling 策略

### 5.1 计算类型

混合（CV 融合）：Cube（gemm_v0 × NBLK 次/任务）+ Vector（relu 随 copy、w 广播乘、头归约、mask、硬件 topk、softmax）+ MTE（Q/K/w 装载 + CV 中转 + 输出写）。

### 5.2 Block 划分

```python
QG  = 8      # token/任务（CAP=4096 档降为 4）——GEMM M 维 = QG*H，摊薄 GEMM 固定开销
BN  = 32     # K 块宽（GEMM N 维）——L0C [QG*H, BN] 容量与 GEMM 规模的平衡点
GRID = 20    # persistent 编译期常量（host 查 cube_core_num，910B3=20；sparse_mla:1501 先例）
CAP = 512/1024/2048/4096   # score_row 档（= next_pow2(max_seg_len)，host 分派；≤4096 受 topk src 硬件上限）
CH  = 1024   # generic 分段宽
```

**QG=8 / BN=32 选取论证**：

- **L0C 硬约束**：logits [QG*H, BN] fp32 ≤ 128KB → QG*H*BN ≤ 32768。QG=8,H=64,BN=32 → 512×32=16K 元素 = 64KB（50%，余量充足）✓；QG=8,BN=64 → 128KB 恰满（拒绝，indexer_bwd §5.2 同款"不赌恰满"原则）；QG=16,BN=32 同样恰满，拒绝。
- **GEMM 次数与固定开销**：S=2048 单段因果总块数 = Σ_g ceil((g+1)·QG/BN)（g 为组号）= **8320 次** [512,32,128] GEMM（4.19 MFLOP/次，总 34.8 GFLOP = 数学下限 ✓）；对比 per-token 方案（GPU 同构）65K 次 [32,64,128] 小 GEMM——**固定开销摊薄 ~8×**（锚点 A3：gemm ~500-1500cy 中固定开销为主）。
- **(m,h) 展平的关键收益**：Q [ΣS, H, D] 的连续行段天然即 [QG*H, D] 展平矩阵（内存零重排），M 维从 32（per-token）升到 512，且 **H 循环消失**（64 次小 GEMM → 1 次大 GEMM）。
- **UB 约束**：score_row [QG, CAP] 与 logits_ub [QG*H, BN] 联动（§4.5 表）；CAP=4096 档 QG 降 4。
- **Stage 3 升级路径**：QG=16/BN=16（[1024,16] L0C 64KB，GEMM 数再 ÷2 但 N=16 效率待测）；中转 bf16 已默认（§9.4）。

### 5.3 约束分析（含 GEMM 非整除处理，门禁项）

- **GEMM 16 对齐**：M = QG*H（QG=8 且 H%8==0 → M%64==0 ✓）；N = BN = 32 ✓；K = D = 128 ✓（D 为 2 的幂主仓惯例；构建期 assert D%16==0）。H=16 → M=128 ✓；H=48 → M=384（384%16=0 ✓）。
- **段长 % QG ≠ 0（varlen 尾组）**：组内有效 token m_valid = len - t0 < QG。处理（indexer_bwd §5.3 双路守卫先例）：① Q/w 装载整段 if / 逐 token 守卫 else；② topk/写出循环 `if t_abs < ln` 整体跳过无效 token；③ 无效 token 的 score_row 槽位不被消费。
- **段长 % BN ≠ 0（K 尾块）**：k_ed = min(k_st+BN, ln)。处理：整段 if / 逐行守卫 else（`if k_st+i < ln: T.copy(IndexK[...])`，unsorted:42 先例）；块内 n ≥ len 的 score 位置由**因果 mask 天然覆盖**（n ≥ len > i_u ⇒ n > i_u ⇒ select -inf）——不需额外尾块 mask ✓。
- **i_u+1 < topk（短 token，主仓"段长 ≥ topk"assert 下仍存在的行内边界）**：topk `actual_num = T.max(i_u+1, topk)` 保护（硬件 K ≤ actual_num）；score_row 全量 fill(-inf) ⇒ 尾部 val=-inf（softmax→0，与 golden 一致）+ idx 为 pad 位置（> i_u）→ select 置 -1（GPU kernel 同款）。
- **max_seg_len 与 CAP 档**：host 保证 CAP = next_pow2(max_seg_len) ≥ 段长；CAP ≤ 4096（topk src buffer 硬件上限 8160 元素 = repeatTimes≤255，test_topk.py:390-394 注释），4096 档已留 2× 余量；段长 > 4096 → generic（CH=1024 分段 + merge_sort）。
- **topk % 1**：topk 与 BN 无耦合（topk 是选择数不是块宽）；topk ∈ 2 的幂（主仓 assert 复刻于构建期）。
- **K 块循环空转**：上界 NBLK = CAP/BN（编译期），守卫 k_st ≤ t_max。S=2048/CAP=2048 时空转率 = 1 - 8320/(256×64) = 1 - 8320/16384 ≈ 49%（因果三角固有）；空转代价 = 每空块 ~10cy 标量判断 × 8064 块 / 20 核 ≈ **4-8us**，可忽略。

### 5.4 注意事项

- score_row 每任务头 `T.fill(-inf)` 全量（QG×CAP）：既是 topk 的 pad 前提（硬件 fp32 src 会 in-place pad [actual_num, aligned)，test_topk.py:178-218），也是跨任务复用的脏数据防护；fill 为向量指令（64KB ≈ 数百拍），代价可忽略。
- `T.tile.topk` 的 fp32 src **会被 in-place 修改**（pad -inf）——score_row 槽位复用安全（下轮任务头重新 fill）✓。
- topk 输出 idx 为 **score_row 内 0-based 位置** = 段内 s2 位置 ✓ 与 golden topk_indices（段内索引）直接可比，无需坐标换算。
- ws_logits 按 cid 分槽（[GRID, ...]），persistent 同核程序序保证写→读→写次序（indexer_bwd §7.2 同款）。
- GroupTable 行号溢出：num_groups 为 symbolic 维度，persistent ntask 上界精确（striped），无越界读。

### 5.5 数据搬运性能可行性

```text
[REORDER-COST]
path: indexer_topk_reducesum fast（S=2048 单段，QG=8/BN=32/GRID=20）
gm_passes: 读 IndexQ 32MB（每 q 行恰好 1 次）+ IndexK 64MB 名义（L2 驻留 0.5MB，HBM 实际 ~1-2MB）
           + Weights 0.26MB + GroupTable 3KB + CV 中转读 266MB（L2 局部）
           写 CV 中转 266MB（L2 局部）+ 输出 1.05MB
dma_transactions: bulk 装载/写回 ~3 条/K 块 × 8320 + per-u 行写 ~2×8×8320 ≈ 25K 条（对每条 [64,32]/[32] 级
                  向量宽度，非短行标量 DMA；对比 indexer_bwd rev1 per-token 短行 ~27 万条）
gm_scalar_accesses: GroupTable 3 次/任务 × 256 任务 = 768 次（vs 数据量 34.8 GFLOP，占比可忽略）
address_div_mod_per_element: 0（行号加法预计算；pid 分解为纯整数乘加）
active_cores / serial_tasks_per_core: 20 persistent / ceil(256/20)=13 任务峰值（striped 因果负载：
           首任务最重 ~64 块、末任务 ~2 块；striped 步长 20 使重任务分散到不同核，最坏核 ≈ Σ ~13 任务 × 平均 32 块）
largest_case_timeout_gate: l0_main（S=2048）120s（§9.4 性能哨兵）
result: pass（量化预算 §9.4：550-800us < 1428us）
```

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 任务维（token 组） | **persistent 常量 grid** | `T.Kernel(20, threads=1, is_npu=True) as (cid)` + `T.serial(ntask)`（运行时边界）+ `pid = k*20+cid` | 因果三角负载（token i 处理 i+1 行 K）→ 块间 2× 负载差；striped 轮询把重任务分散到 20 核（sparse_mla:309 已验证形态；对比动态 grid=seq_len 的"每 block 一任务"会因 2048 任务/20 核的调度量化损失更难均衡） |
| K 块维 | 编译期常量上界 + 运行时守卫 | `T.serial(NBLK=CAP/BN)` + `if k_st <= t_max` | GPU 版 num_blocks 依赖 tensor 值（i_t）不可用（ascend-constraints §1）；编译期上界（CAP 由 host 分派保证 ≥ 段长）+ 守卫为已验证形态 |
| 组内 token（向量段） | 编译期常量 | `T.serial(QG=8)` | 静态边界；[H, BN] 小 tile 段复用同槽 buffer |
| topk/softmax 段 | 编译期常量 | `T.serial(QG)` + `if t_abs < ln` | 尾组无效 token 跳过 |
| generic CH 大段 | 编译期常量上界 + 守卫 | `T.serial(NSEG)` + `if sg*CH <= t_max` | 段数上界按 8160/CH 编译期档；每段尾 topk+归并 |
| 元素级 | 向量化 | `T.tile.*` 整 tile 指令 + 硬件 topk/merge_sort | 无逐元素 T.Parallel 标量循环（GPU 版 L58-70 的 T.Parallel(N) bitonic 整体替换） |

### 6.2 循环伪代码

见 §3.3（fast）与 §3.3 末尾（generic 差异段）。

### 6.3 流水线优化

L0 不启用 T.Pipelined（GPU 参考num_stages=0 本无流水；K 块循环含运行时守卫不可流水）。引擎级重叠由 AUTO_CV_COMBINE/AUTO_CV_SYNC 自动排布：MTE2（下一 K 块装载）∥ Cube（GEMM）∥ Vector（上一块 w-mul/reduce/mask）∥ MTE3（中转写/输出写）。**Stage 3 优化项**：ws_logits 双槽轮转（中转与计算更深流水，moe_topk_gate ping-pong 先例）、QG=16/BN=16、topk 分段 AIV 并行。

### 6.4 尾块处理

varlen 尾组（段长 % QG ≠ 0）与 K 尾块（段长 % BN ≠ 0）：§5.3 双路守卫 + 因果 mask 天然覆盖；短 token（i_u+1 < topk）：§5.3 actual_num 保护 + idx=-1 修正。无 shape 尾块于 H/D/topk 维（整维进 tile 或构建期 assert 拒绝）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：全自动同步（Developer）——零手动 `T.set_flag/wait_flag/barrier_all/cross_flag`。

### 7.2 自动同步点（pass 注入；设计层列依赖关系）

| 依赖 | 保证方 |
|------|--------|
| Q_l1/K_l1/w_ub 装载完 → gemm_v0 / tile.mul | AUTO_SYNC（MTE2→Cube / MTE2→V 序） |
| logits_l0c 定界 → ws_logits 写 → logits_ub 读（CV 中转两段） | **AUTO_CV_SYNC 按 buffer 名含 "workspace"（ws_logits）配对插 CrossCoreSetFlag/WaitFlag**（indexer_bwd rev1 已用最小复现 kernel 验证 4 组往返 flag 生成、按 cid 分槽；本设计仅 1 组往返，机制更简单） |
| logits_ub 就绪 → 向量链（mul/reduce/mask） | AUTO_SYNC（MTE2→V） |
| score_row 写 → topk 读（同核程序序） | AUTO_SYNC（V→V 序 + 硬件 topk 前的 pipe 序） |
| topk src in-place pad → gather_mask 读 | AUTO_SYNC（V 内序） |
| sort/topk auto tmp 的注入与生命周期 | InjectTmpBuffer pass + MEMORY_PLANNING（lightning_indexer:11-13 注释先例；显式 tmp= fallback 见 R1） |
| 输出 copy → kernel 返回 | AUTO_SYNC（V→MTE3） |
| fast/generic kernel 全部完成 → host 返回 | runtime stream 同步 |

### 7.3 pass_configs 配置

见 §4.7（4 True；与 topk_selector:24-27、indexer_bwd §4.7 逐键一致）。

---

## 8. 融合算子设计

### 8.1 融合算子判定

**判定结果**：是（单 kernel 融合 GEMM + relu + 加权头归约 + 因果 mask + 硬件 topk + softmax + 索引修正；无第二 kernel）。

### 8.2 CV 交互设计（Developer 模式）

- **无 `workspace_idx` / 无 `out_idx`**：输出与 ws_logits 全显式参数（规避 auto workspace 动态 grid 限制——indexer_bwd rev1 实测教训，本设计沿用）。
- Cube↔Vector 交换仅 1 处：logits（L0C→UB，经 ws_logits bf16 中转，平台强制 GM 往返显式参数化）。流量 266MB/调用（bf16）已入 §9.4 预算。
- 对照 design-template §8.2：Developer 默认"片上直连消除 workspace"适用于 L0C→UB 存在直连的场景；本平台 L0C→UB 被编译器改写 GM 往返（indexer_bwd R8 实测），故按"Developer 复杂场景回退"处理——**显式 ws 参数 + threads=1**，与 indexer_bwd fast（Developer + 显式 ws）完全同型，该形态已在本目录 indexer_bwd 上实测通过。

### 8.3 Cube 核计算流程

```python
T.gemm_v0(Q_l1, K_l1, logits_l0c, transpose_B=True, init=True)   # [QG*H, D]×[D, BN] → fp32 L0C
```

### 8.4 Vector 核计算流程

见 §3.3 伪代码"中转段 2 + per-token 向量段 + topk/softmax 段"（w 广播乘 → reduce_sum(dim=0) → ×sm_scale → 因果 select → score_row；topk → gather_mask → softmax → 修正 → cast → 写出）。

### 8.5 pass_configs 配置

见 §4.7。

### 8.6 注意事项

- ws_logits dtype **bf16**（中转 cast）是数值基线的一部分（复刻 golden 的 logits bf16 round，§1.3），不可改回 fp32 而不同步更新 §9.3 容差论证。
- L0C→GM cast（fp32→bf16）与 GM→UB cast+relu（bf16→fp32）两条 copy 的 auto-cast 方向均有先例（gqa_sink:478 / indexer_bwd dk cast），但 **cast 与 enable_relu 同一条 copy 的组合待验证（R7）**；fallback：中转段 2 拆为 cast copy + `T.tile.relu`（ascend_tile.py:1376）两步，UB 预算余量足够（+64KB 临时或原地）。
- topk/merge_sort 的 tmp 默认 auto 注入；若 persistent + symbolic 维组合下 InjectTmpBuffer 失败（无直接先例，理论风险），显式传 `tmp=`（ascend_tile.py:599 签名支持，explicit_tmp 测试覆盖）。
- generic 的 merge_sort 仅支持 fp32（ascend_tile.py:525）——池/段 topk 输出本就是 fp32 ✓。

---

## 9. 验证方案

### 9.1 Golden 函数（逐字节复刻主仓 `ref_index_score`，ref_gpu/indexer_topk_reducesum.py:208-229，NPU 上运行）

```python
def ref_index_score(Q, Weights, K, topk, offsets):
    """主仓 ref_gpu/indexer_topk_reducesum.py:208-229 逐字节复刻（golden 语义权威）。"""
    all_topk_indices, all_topk_score = [], []
    for i in range(offsets.shape[0] - 1):
        assert (offsets[i + 1] - offsets[i]).item() >= topk      # 段长 >= topk（主仓 assert 语义）
        q = Q[offsets[i] : offsets[i + 1]]; weights = Weights[offsets[i] : offsets[i + 1]]
        k = K[offsets[i] : offsets[i + 1]]
        softmax_scale = q.shape[-1] ** -0.5
        s = q.shape[0]
        mask = (torch.arange(s)[:, None] >= torch.arange(s)[None, :]).to(q.device)
        logits = einsum(q, k, "s1 h k, s2 k -> s1 h s2")          # bf16 输出（round 点）
        logits = F.relu(logits)
        logits = (logits * weights.unsqueeze(-1)).sum(dim=-2, dtype=torch.float32) * softmax_scale
        logits = torch.where(mask, logits, float("-inf"))
        topk_logits, topk_indices = torch.topk(logits, k=topk, dim=-1)
        topk_score = F.softmax(topk_logits, dim=-1, dtype=torch.float32)
        all_topk_indices.append(topk_indices); all_topk_score.append(topk_score)
    return torch.cat(all_topk_indices, dim=0), torch.cat(all_topk_score, dim=0)
```

**输入构造**（复刻主仓 `test_kernel:232-244`）：`torch.manual_seed(42)`；q/w/k = `torch.randn(...)` bf16；offsets int32。L0 全部用例沿用该确定性构造（shape/offsets 按用例表变化）。

### 9.2 L0 门槛测试计划（tilelang-op-test-design 场景 A 生成，仅 L0）

**测试策略判定**（场景 A Phase 2）：计算类型 = 混合（CV 融合）；复杂度 = Fusion；数学特征 = matmul + relu + 加权归约 + mask + **topk（排序选择）** + softmax；综合类别 = **Fusion（前向 indexer，双路径分派变体）**。测试重点 = 功能验证 + 精度收敛（L0）+ **topk 索引/分数双输出比对策略** + 分派正确性；dtype_count=1（bf16 固定精度路径，主仓语义）、shape_count=8、dispatch_count=2（fast/generic）。

**L0 用例表**（全部 bf16 输入 / D=128；golden = §9.1）：

| 用例名 | 级别 | Shape（B, ΣS, H, D, topk） | offsets | 预期分派 | 说明 |
|--------|------|---------------------------|---------|----------|------|
| l0_main | L0 | (1, 2048, 64, 128, 64) | [0, 2048] | fast（CAP=2048, QG=8） | **主仓测试输入原样复刻**（seed 42 randn bf16）= 性能哨兵（§9.4）；覆盖 2048 行全量 topk/softmax、K 块 64 次满循环、组尾 2048%8=0 无尾组 |
| l0_varlen | L0 | (1, 2048, 64, 128, 64) | [0, 1023, 2047+1] → [0, 1023, 2048] | fast | varlen 2 段非对称（1023 + 1025）：段长%QG≠0 尾组守卫 + 段长%BN≠0 K 尾块 + 跨段 token 的因果域隔离（段边界 token 的 topk 不得越段）+ 段内 i_t<topk 短 token（段首 63 个） |
| l0_multibatch | L0 | (3, 1536, 64, 128, 64) | [0, 512, 1024, 1536] | fast | B=3 多段等长（512 = CAP=512 档边界）；GroupTable 多段展开 + persistent 跨段轮询 |
| l0_small | L0 | (1, 128, 64, 128, 64) | [0, 128] | fast（CAP=512） | 小规模冒烟：前 63 个 token 全部 i_t+1 < topk（actual_num 保护 + -inf 尾部 + idx=-1 修正的密集路径）+ 4 任务 persistent 轮询 |
| l0_equal_topk | L0 | (1, 64, 64, 128, 64) | [0, 64] | fast（CAP=512） | **段长 == topk 边界**（主仓 assert 下界）：t=0 行仅 1 个有效位置、t=63 行恰 topk 个；-inf 尾部占比最大行（t=0：63 个 -inf） |
| l0_h16 | L0 | (1, 512, 16, 128, 64) | [0, 512] | fast | 不同 head 数（M=QG*H=128 对齐路径；w 归约链 [16, 32] 段） |
| l0_topk32 | L0 | (1, 512, 64, 128, 32) | [0, 512] | fast | 不同 topk（2 的幂；topk_dst/val/idx/softmax 缓冲全链随 topk 变化；K ≤ actual_num 保护在 topk=32 下的短 token 域更小） |
| l0_generic | L0 | (1, 5120, 64, 128, 64) | [0, 5120] | generic（CH=1024, 5 段归并） | **段长 > 4096 → generic 流式路径**：分段 topk + merge_sort 池归并 4 次 + 末段短（5120 = 5×1024，无短末段；短末段由 l0_generic_tail 补充——见下行） |
| l0_generic_tail | L0 | (1, 4608, 64, 128, 64) | [0, 4608] | generic | generic 末段短块（4608 = 4×1024 + 512：末段 actual_num=512 < CH 与 < topk 保护路径） |
| l0_dispatch_equiv | L0 | (1, 2048, 64, 128, 64) | [0, 2048] | force fast + generic 双路 | **分派正确性/等价性**：同输入 dispatch="fast" 与 "generic" 分别执行，两路输出独立 vs golden 判定（§9.3）+ 两路之间 TopkIndices（mask 域）与 ReduceSum 逐元素一致（panel P1 等价性先例）+ host 检测函数断言（max_seg_len=2048 → fast） |

**比对策略（topk 双输出，位置敏感设计）**：

1. **ReduceSum（topk_score, fp32）**：全位置（含 -inf 尾部 → 0）混合容差比对（§9.3 fp32 行）。-inf 位置双方 softmax 均输出精确 0，abs_err=0 不拉高误差。
2. **TopkIndices（int32）**：**mask 域比对**——`mask = (ref_topk_score > 0)`（主仓测试 L257 同款口径），mask 域内逐元素精确匹配，matched_ratio ≥ 0.99；非 mask 域（score=0 的 -inf 尾部）**不比对索引**（golden torch.topk 对 -inf 平局返回任意索引，无确定真值；本设计与 GPU kernel 均输出 -1）。
3. **平局概率论证**（mask 域内 ratio 0.99 而非 1.0 的依据）：randn 连续输入下，两个不同 s2 位置的 fp32 score 严格相等的概率——score 由 H=64 个独立 per-head 内积（每个 ~N(0, D) 级连续分布）加权求和而成，fp32 离散但值域 ~2²⁴ 桶，随机采样碰撞概率逐对 < 10⁻⁶；score=0 的退化平局需全部 64 head 内积 ≤ 0（概率 2⁻⁶⁴ ≈ 0）。topk 边界附近数值微差导致的**相邻位次交换**（非平局但 score 接近）：两位置 score 差 < 容差内时交换仍可能使两输出在各自位置通过 ReduceSum 混合容差（score 相近），TopkIndices mask 域出现成对 mismatch——0.99 ratio 门限吸收此类稀疏事件；诊断输出交集比（主仓 L259-263 同款 set 交集比）。
4. 每用例两输出分别判定，任一 FAIL 该用例 FAIL；`[PRECISION_PASS/FAIL]` 打印 matched_ratio/max_abs/交集比三指标。

### 9.3 精度标准（混合容差；dtype 全集 = 本算子固定 dtype 组合）

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|-------|------|------|---------------------|------------------------|
| bfloat16 | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |
| float32 | 2⁻¹⁶ (1.53e-5) | 2⁻¹⁰ (9.77e-4) | 1e-2 | 0.99 |
| int32 | 0 | 0 | 0 | 1.0（精确匹配；**TopkIndices 在 §9.2 mask 域内执行，域内 ratio 门限 0.99**） |

> bfloat16 = q/w/k 输入 dtype（参与计算无直接输出比对，保持 proto dtype 全集一致性）；float32 = **ReduceSum 输出**比对标准（全位置混合容差）；int32 = **TopkIndices 输出**（mask 域精确匹配）。
> **容差校准说明**（经验校准依据，非占位）：本设计与 golden 的数值路径逐 round 点对齐（§1.3 表：唯一差异 = golden 的 bf16 乘积 round，相对误差 ≤ 2⁻⁸ = 3.9e-3），残差远小于 fp32 行的 rtol 9.77e-4？——注意 2⁻⁸ = 3.9e-3 **大于** fp32 rtol 9.77e-4！论证：golden 的 bf16 乘积 round 发生在 bf16 值域粒度上（logits*weights 的积 round 到 bf16，误差 ≤ 2⁻⁸·|积|），但 **sum(dim=-2, dtype=fp32) 在 PyTorch 中先对 bf16 乘积做 fp32 提升**——bf16 乘积的 round 误差经 H=64 头累加后相对不变（≤ 2⁻⁸ 相对，头间独立随机化后 ~2⁻⁸/√64 = 4.9e-4 量级），落在 rtol 9.77e-4 内 ✓；max_abs_error_limit 1e-2 吸收 score 绝对值 ~O(1) 下的尾部（H=64 头、sm_scale=0.088、单 head 积 ~O(10)，score 典型 ~O(1-10)——1e-2 上限对应 0.1-1% 相对，与 ratio 0.99 门限配套）。**L0 首跑后若 max_abs 超限，优先排查中转 cast 语义（R7）而非放宽容差**；topk 位次交换导致的 ReduceSum 局部大误差由 ratio 0.99 吸收（交换对 score 接近，误差 < 位次差）。GPU 版（fp32 logits、无乘积 round）与本设计的差异同量级——两者均在 golden 容差内，但本设计更贴 golden（round 点一致）。

### 9.4 性能预算（实测校准锚点 + 分解；验收：< 1428us @ l0_main 配置）

**校准锚点**（全部实测，出处：indexer_bwd DESIGN.md §9.4，910B3）：

| 锚点 | 数值 | 来源 |
|------|------|------|
| A1 聚合有效带宽 | ~510GB/s（读 273 ∥ 写 238 并发）；L2 中转 ~1TB/s 口径 | panel 实测（msprof 交叉） |
| A2 gemm_v0 单次开销 | ~500-1500cy（小 GEMM 固定开销为主） | indexer_bwd §9.4 A3 |
| A3 短 DMA 开销 | ~100-200cy/条 | indexer_bwd §9.4 A4 |
| A4 Vector 吞吐 | ~64 fp32 elem/cy/核（256B/拍） | 910B3 向量宽度推算（保守） |
| A5 硬件 Sort/TopK 吞吐 | **待实测**（按 sort32×repeatTimes + merge 树保守估 ~2-8K cy/千元素） | 无实测锚点 → 风险项 R3 |

**fast 路径量化预算分解**（S=2048 单段，QG=8/BN=32/GRID=20；8320 GEMM / 256 任务 / 每核 13 任务峰值）：

| 项目 | 每核估算 | 依据 |
|------|---------|------|
| Cube（GEMM） | 8320×~800cy / 20 = 333K cy ≈ **167-333us** @1-2GHz | 8320 次 [512,32,128]（4.19 MFLOP/次 = 34.8 GFLOP 总量 ✓ 数学下限）；锚点 A2 |
| CV 中转（bf16） | 266MB 写 + 266MB 读 = 532MB 总线，L2 局部（ws 窗口 32KB×20 核 = 640KB 驻留 L2）@ ~1TB/s ≈ **530us**；HBM 溢出部分按 273/238GB/s 更慢（R4 风险） | 锚点 A1；indexer_bwd §9.4 L2 中转单列口径 |
| Vector（w 乘/归约/mask） | 8320 块 × 512×32 元素 × ~2 op / 20 核 / 64 elem/cy = 213K cy ≈ **107-213us** | 锚点 A4 |
| Vector（硬件 topk + softmax） | 2048 行 × [~1024 平均 actual_num] sort ≈ 2048×2-4K cy / 20 = 205-410K cy ≈ **102-410us** | 锚点 A5（区间宽，待实测） |
| HBM 必需流量 | Q 32MB + K ~2MB（L2 吸收后）+ 输出 1MB + 表 3KB ≈ 35MB @ 273GB/s ≈ **128us**（读通道，与其他引擎并行） | 锚点 A1 |
| **关键路径（四引擎重叠 + striped 尾部）** | **≈ max(GEMM 333, 中转 530, Vector 320, topk 410) + 尾部 ≈ 550-800us** | panel 同款重叠模型 |
| **合计 vs 目标** | **550-800us < 1428us（余量 44-61%）** | 非临界；两大不确定项（R3/R4）单项最坏 ~×1.5 仍 < 1428us |

**generic 路径不考核性能**（正确性兜底；段 topk 次数 ~5× 于 fast + merge 开销，估算 1.5-3ms 量级）。

**性能哨兵用例**：perf_l0_main（=l0_main 配置，fast 自动命中），单 case timeout 120s；Stage 2 记录基线，Stage 3 考核 < 1428us。

**Stage 3 优化路径**（按优先级，不回退架构）：① ws_logits 双槽流水（中转与 GEMM 深度重叠，moe ping-pong 先例）；② QG=16/BN=16（GEMM 数 ÷2，L0C 64KB 不变，需实测 N=16 cube 效率）；③ topk 段内 AIV 并行（threads=2 消 vid，topk 拆 2 半行归并）；④ 中转 fp32→bf16 已默认（本轮选定）；⑤ K 装载 L2 预取提示（若 R4 HBM 溢出显著）。

---

## 10. 风险点与注意事项

### 10.1 技术约束检测结论（每项附证据）

| # | 风险 | 结论 | 证据 / 缓解 |
|---|------|------|------------|
| R1 | **T.tile.topk 的 auto tmp 注入**（persistent + symbolic 维组合） | ⚠️ 低风险 | auto 注入 + MEMORY_PLANNING 有三先例（lightning_indexer:11-13 文件头注释 + topk_selector + moe_topk_gate，后两者动态 grid）；**fallback**：显式 `tmp=`（ascend_tile.py:599 签名，explicit_tmp 测试覆盖 topk/sort/merge_sort/reduce/select 全族）。L0 首跑验证 |
| R2 | L0C→UB 平台强制 GM 往返 | ✅ 已验证机制复用 | indexer_bwd rev1（显式 ws 参数 + workspace 命名 + AUTO_CV_SYNC 配对 flag，最小复现 kernel 验证）；本设计中转仅 1 组往返（indexer_bwd 4 组） |
| R3 | **硬件 Sort/TopK 吞吐无实测锚点**（预算区间 102-410us 跨 4×） | ⚠️ 中风险 | test_topk.py 有 4096 元素用例（功能）无计时；预算按保守上界仍达标（§9.4）；L0 期用 l0_small/l0_main 计时标定，超预期走 Stage 3 路径 ③ |
| R4 | CV 中转 532MB 的 L2 吸收率 | ⚠️ 中风险 | ws 窗口 640KB 驻留（L2 64MB 的 1%），写读紧邻（同任务程序序）；indexer_bwd fast 的 ws_logits 同款模式实测可行（16.6MB 规模）；若 HBM 溢出 → Stage 3 路径 ①⑤ |
| R5 | GEMM M/N/K 16 对齐 | ✅ | M=QG*H（QG=8,H%8==0 → %64==0）、N=32、K=128；§5.3 验算含 H=16/48 |
| R6 | L0C/L1/UB 容量 | ✅ | §4.5 保守口径：L0C 64KB/128KB、UB ~145KB/192KB、L1 136KB/512KB；CAP=4096 档 QG=4 联动规避恰满 |
| R7 | **cast 与 enable_relu 同一条 copy**（bf16→fp32 auto-cast + relu 组合） | ⚠️ 低风险 | 各单要素有先例（gqa_sink:478 cast；lightning_indexer:54-58 relu；indexer_bwd cast copy），组合无直接先例；**fallback**：中转段 2 拆 cast copy + `T.tile.relu`（ascend_tile.py:1376），UB 余量足够 |
| R8 | topk actual_num 运行时标量 / K ≤ actual_num | ✅ | symbolic actual_num 先例（lightning_indexer_dynamic:119）；K ≤ actual_num 用 `T.max(i_u+1, topk)` 保护（§5.3）；actual_num ≥ 1 恒成立（i_u ≥ 0） |
| R9 | i_u+1 < topk 边界（-inf/-1 填充） | ✅ | fill(-inf) + actual_num 保护 + idx>i_u→-1 修正（GPU kernel L180-183 同款）+ 测试 mask 域比对（§9.2）；l0_small/l0_equal_topk 密集覆盖 |
| R10 | 动态循环边界（GPU num_blocks 依赖 i_t） | ✅ | 编译期上界 CAP/BN + 运行时守卫（§3.5.1）；persistent ntask 依赖 symbolic 维（sparse_mla:309 已验证形态）；空转 4-8us 可忽略 |
| R11 | varlen 尾组/K 尾块 | ✅ | if/else 双路守卫（indexer_bwd §5.3 / unsorted:42 先例）+ 因果 mask 天然覆盖 K 尾块溢出（§5.3） |
| R12 | fp32 topk index 精确性 | ✅ | fp32 可精确表示 ≤ 2²⁴ 整数；CAP ≤ 4096 ≪ 2²⁴；test_topk.py:145-146 fp32 index 精确断言（fp16 才有 >2048 精度问题，本设计不用 fp16 index） |
| R13 | merge_sort 仅 fp32 / slice 对齐限制 | ✅ | 池/段缓冲均 fp32；slice 用整 buffer 或整行（ascend_tile.py:529-532 限制规避：merged[0:2*topk] 是 1D 连续 slice ✓） |
| R14 | host 分派检测 | ✅ | max_seg_len = diff(offsets).max().item()：设备侧原语 + 1 次标量 D2H（indexer_bwd host 门控 _panel_gate_ok 先例）；纯 metadata 判定不读改数据 |

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| score_row 未 fill(-inf) 复用脏数据 | 任务头漏 fill | topk 混入上任务残留值 → 索引越界/分数错 | 每任务头 `T.tile.fill(score_row, -inf)` 全量（§5.4） |
| topk actual_num < K（短 token） | i_u+1 < topk 直传 | 硬件行为未定义/异常 | `n_eff = T.max(i_u+1, topk)`（§5.3） |
| 段长 > CAP 走 fast | host 分派错误/直接调 fast | score_row 截断 → 错误 topk | host max_seg_len 精确检测 + 构建期 assert；dispatch 参数供测试注入负向验证（Stage 2 L2 扩展） |
| ws_logits 未按 cid 分槽 | persistent 多任务同核并发窗口 | 中转数据竞态 | ws [GRID, ...] 按 cid 分槽 + 同核程序序（indexer_bwd §7.2 同款） |
| GEMM 操作数误放 UB | alloc 后直接 gemm_v0 | 运行时错误 | Q_l1/K_l1 用 alloc_shared（编译器映射 L1） |
| cast 与 relu 组合 copy 编译失败 | R7 fallback 未启用 | 编译期报错 | 拆两步：cast copy + tile.relu |
| golden 的 topk_indices 非 mask 域比对 | 测试把 -inf 尾部也精确比对 | 假 FAIL（golden -inf 平局任意序） | §9.2 比对策略 2：mask = ref_score > 0 域内比对 |
| sm_scale 预乘 q（照抄 GPU） | 迁移惯性 | 引入额外 bf16 round，偏离 golden 数值基线 | sum 后乘（§1.3，golden 序） |

### 10.3 特殊场景处理

- **全 -inf 行**：不存在（i_u ≥ 0 ⇒ score_row[u, 0] 有限 ⇒ topk val[0] 有限 ⇒ softmax max 有限）✓。
- **NaN 传播**：输入 randn 无 NaN（主仓域）；若输入含 NaN，golden/本设计同步传播（topk 排序中 NaN 行为平台差异——**不在 L0 域**，Boundary 由 Stage 2 扩展并按位置敏感比较）。
- **极小段（len == topk）**：l0_equal_topk 覆盖（assert 下界）。
- **空 batch（offsets 相邻相等）**：段长 0 < topk 违反 assert → host 拒绝（与 golden assert 同款语义）。

### 10.4 支持域与分派审计

```text
[DISPATCH-COVERAGE]
supported_domain: dtype 固定组合（q/w/k bf16 + 输出 int32/fp32 + offsets/token_indices/group_table int32）；
  seq_len/batch 运行时任意（varlen，段长 ≥ topk，golden assert 同款）；heads % 8 == 0 且 ≤ 64（主仓语义，
  fast 的 M=QG*H 16 对齐在 QG=8 下自动满足）；dim % 16 == 0（2 的幂主仓惯例）；topk 为 2 的幂（主仓 assert）
  且 ≤ min(段长, 4096 fast 域无额外约束)；sm_scale 任意 float（缺省 dim**-0.5）
dispatch: host 精确检测 max_seg_len = diff(offsets).max() ——
  max_seg_len ≤ 4096 → fast（CAP = next_pow2 clamp [512, 4096] 编译档；CAP≤2048→QG=8，CAP=4096→QG=4）
  max_seg_len > 4096 → generic（CH=1024 流式分段 + merge_sort 池归并，任意段长）
  dispatch 参数三值（auto/fast/generic，缺省 auto）供测试注入与调用方显式声明
generic_fallback: generic kernel（全 supported_domain 覆盖，含段长 > 4096 与任意 varlen 拓扑）
specializations: fast（CAP 4 档 × QG 2 档 = 4 个有效编译缓存键：(512,8)/(1024,8)/(2048,8)/(4096,4)）
  等价性依据：① score_row/CAP ≥ 段长 → 无截断；② topk 硬件上限 8160 ≥ CAP=4096（test_topk.py:390）；
  ③ generic 分段归并与全行 topk 数学等价（merge_sort 稳定降序，-inf 平局沉尾）
unsupported_inputs: 段长 < topk（语义非法，golden assert 同款拒绝）；fp16/fp32 主输入 dtype（主仓固定 bf16）；
  heads % 8 != 0 或 > 64（主仓 kernel assert L33 同款）；topk 非 2 的幂（主仓 assert L31）
result: pass（两路全覆盖 + 精确检测，无枚举遗漏）
```

### 10.5 Host 侧审计

```text
[HOST-METADATA-AUDIT]
operation: prepare_token_indices(offsets)（ref_gpu/index.py:80-83 同款）+ prepare_group_table(offsets, QG)
input_stride -> output_stride: 新建张量（CPU int32 构造）→ .npu() 一次 H2D（两表合并一次传输）
shares_storage / same_data_ptr: false（新建输入，数据准备类）
aclnn_or_physical_copy: false（CPU 构造 + 单次 H2D；sparse_mla:1278-1296 / indexer_bwd §10.5 先例）
result: allow

operation: torch.zeros(TopkIndices int32 / ReduceSum fp32) → kernel 显式参数
shares_storage / same_data_ptr: false（新建输出，每 token 行独占写无原子）
aclnn_or_physical_copy: false（unsorted:247 / gqa_sink:531-535 先例）
result: allow

operation: max_seg_len 检测 torch.diff(offsets).max().item()（分派判定）
shares_storage / same_data_ptr: true（只读比较）
aclnn_or_physical_copy: false（设备侧原语 + 1 次标量 D2H；indexer_bwd host 门控先例）
result: allow（.item() 同步在纯 kernel 计时口径外）

operation: kernel 调用后直接返回 (topk_indices, topk_score)
aclnn_or_physical_copy: false（无 .to()/.contiguous()/cat/pad——输出 shape 与原始一致）
result: allow
```

---

## 11. 交付清单

### 11.1 目录结构（目录共享，全部带算子名前缀，不覆盖 indexer_bwd 工件）

```
examples_experiment/dsa_sparse_finetune/
├── DESIGN_indexer_topk_reducesum.md   # 本设计文档（新增）
├── proto_indexer_topk_reducesum.yaml  # 接口规格（新增）
├── indexer_topk_reducesum.py          # 【Stage 2 待实现】fast/generic 双 kernel + host 分派 interface
├── test_indexer_topk_reducesum.py     # 【Stage 2 待实现】golden + L0 10 用例 + perf 基准 + main
├── DESIGN.md / proto_indexer_bwd.yaml / indexer_bwd.py / test_indexer_bwd.py   # indexer_bwd 工件（只读）
└── ref_gpu/                            # 主仓参考（只读）
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| DESIGN_indexer_topk_reducesum.md（本文件） | 已完成 | 双路径架构 + 硬件 topk 方案 + 实测锚点预算 + L0 计划（§9.2） |
| proto_indexer_topk_reducesum.yaml | 已完成 | 接口规格（dtype 全集 = §9.3 表；attrs = §1/§4/§10.4 派生） |
| indexer_topk_reducesum.py | 待实现（Stage 2） | 双 kernel builder + indexer_topk_reducesum_interface（含检测/分派/GroupTable 构造） |
| test_indexer_topk_reducesum.py | 待实现（Stage 2） | golden（§9.1）+ L0 10 用例（§9.2，含 mask 域比对策略）+ perf 基准 + main（--level 分发） |

### 11.3 命名规范

- kernel builder：`tl_indexer_topk_reducesum_fast` / `tl_indexer_topk_reducesum_generic`（对齐主仓 `tl_indexer_topk_reducesum_impl` 的 tl_ 前缀）
- host 入口：`indexer_topk_reducesum_interface(q, weights, k, topk, offsets, dtype=BF16, dispatch="auto")`（签名向后兼容主仓 L191-205，新增尾参带默认值）
- 检测 helper：`_detect_dispatch(offsets) -> str`（供测试断言）

### 11.4 实现顺序

1. ✅ 设计文档 + proto + L0 门槛测试计划（本文件）
2. ⬜ fast kernel（先 l0_small 冒烟 → l0_main；R1/R7 两项待验证 API 组合优先打通）
3. ⬜ generic kernel（l0_generic / l0_generic_tail）
4. ⬜ host 分派/GroupTable + l0_dispatch_equiv 用例
5. ⬜ L0 门槛测试全通过（10 用例 × 2 输出，混合容差 + mask 域比对）
6. ⬜ 性能基线记录（perf_l0_main；Stage 3 考核 < 1428us，不达标按 §9.4 Stage 3 路径，不回退设计）
7. ⬜ 扩展分层套件（L1/L2/Boundary，tilelang-op-test-design 场景 B）
8. ⬜ 全量套件 + 覆盖门禁（coverage_check.py --proto proto_indexer_topk_reducesum.yaml）
