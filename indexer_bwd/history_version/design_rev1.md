# indexer_bwd 算子设计文档

> 迁移来源：`examples_experiment/dsa_sparse_finetune/ref_gpu/indexer_bwd.py`（GPU 版主仓参考，只读）
> 目标平台：Ascend A2/A3（910B3，20 Cube Core，L0C 128KB / UB 192KB / L1 512KB）
> 编程模式：**Developer**（用户指定）

## 1. 概述

### 1.1 算子名称

indexer_bwd（DSA sparse finetune lightning indexer 反向，varlen KL 散度损失反传）

### 1.2 功能描述

DSA（DeepSeek Sparse Attention）稀疏微调中 lightning indexer 的反向算子：给定前向保存的 topk 选中位置 `TopkIndices`、注意力分数 `AttnScore`（teacher）与 indexer 输出分布 `IndexScore`（student，经 log_softmax），在变长（varlen）序列上计算 4 组梯度：`dIndexQ`、`dWeights`、`dIndexK`，其中 `dIndexK` 通过 atomic scatter-add 跨 token 累加。

### 1.3 数学公式

前向语义（每 token t，sm_scale = D**-0.5）：

$$\text{logits}[t,h,s] = \mathrm{relu}(q[t,h,:] \cdot k[s,:] \times \text{sm\_scale})$$

反向（遍历 topk 索引 pos，causal mask 为 `-1 < pos <= t`；`1{...}` 为指示函数）：

$$d\_logits[t,h,pos] = (\text{IndexScore}[t,pos] - \text{AttnScore}[t,pos]) \times 1\{\text{logits}>0\} \times \text{Weights}[t,h]$$

$$d\text{Weights}[t,h] \mathrel{+}= \sum_{pos} (\text{IndexScore} - \text{AttnScore})[t,pos] \times \text{logits}[t,h,pos]$$

$$d\text{IndexQ}[t,h,:] \mathrel{+}= \sum_{pos} d\_logits[t,h,pos] \times \text{IndexK}[pos,:] \times \text{sm\_scale}$$

$$d\text{IndexK}[pos,:] \mathrel{+}= \sum_{h} d\_logits[t,h,pos] \times \text{IndexQ}[t,h,:] \times \text{sm\_scale} \quad (\text{atomic scatter-add})$$

与主仓 GPU 版一致的关键精度路径：IndexQ 预乘 sm_scale（供 logits GEMM 与 dk GEMM 使用，bf16 存储）；dq GEMM 用未缩放 Q 的等价形式（GPU 版对 dq 结果在 fp32 fragment 上补乘 sm_scale，本设计保持同一等价形式：dq GEMM 复用预乘后的 Q，最后不补乘——**两者数学等价**，见 §3.1 注）；d_logits 从 fp32 两次 cast 到 bf16 分别作为 dq/dk GEMM 操作数；dk 以 bf16 部分和逐块 atomic 累加。

> 注：GPU 版 L82-83 将 `index_q_shared *= sm_scale`，随后 L161-162 对 `d_index_q_frag *= sm_scale` 再补乘。仔细核对 GPU 版：logits GEMM（L104）与 dk GEMM（L147）均使用预乘 Q；dq GEMM（L135）同样使用预乘 Q，但 L161-162 又对 dq 结果补乘 sm_scale——即 GPU 版 dq 实际乘了两次 sm_scale 中的**一次**（预乘）并在末尾**再乘一次**。这与上节数学公式（dq 含一次 sm_scale、dk 含一次 sm_scale）不符？
> **澄清**：重新推导。前向 logits = relu(q·k × sm_scale)，故 ∂loss/∂q[t,h,:] = Σ_pos d_logits × k[pos,:] × sm_scale（一次），∂loss/∂k[pos,:] = Σ_h d_logits × q[t,h,:] × sm_scale（一次）。GPU 版把 sm_scale 预乘进 Q_shared：logits GEMM 输出 = (q×sm_scale)·k ✓ 一次；dk GEMM = d_logits × (q×sm_scale) ✓ 一次；dq GEMM = d_logits^T × k **不含** sm_scale，故 L161-162 补乘一次 ✓。**本设计完全复刻该结构**：预乘 Q 用于 logits/dk 两个 GEMM，dq GEMM 后在 fp32 上补乘 sm_scale。

### 1.4 算法描述

计算类型：**Cube+Vector 混合（CV 融合）**——每 token 3 个小 GEMM（logits / dq / dk）+ 向量后处理（relu、差值、mask、broadcast 乘、reduce）+ DMA gather/scatter-atomic。复杂度：Fusion（多算子组合）。

Host 预处理（数据准备，允许类）：`prepare_token_indices(offsets)` 在 CPU 构造 `TokenIndices [ΣS, 2] int32` 后一次 H2D（先例：`examples/dsa_sparse_finetune/sparse_mla_topk_reducesum.py:1278-1296`，语义与主仓 `ref_gpu/index.py::prepare_token_indices` 逐字节一致）。

> **⚠️ Host 侧 Buffer 操作约束**（ascend-constraints.md §5）：host 侧只允许 ① `prepare_token_indices` 的 CPU 构造 + H2D（新建输入张量，数据准备）；② `torch.zeros` 分配 3 个输出张量（新建 buffer，数据准备；先例 unsorted_segment_sum.py:247、gqa_sink_bwd:531-535）；③ kernel 调用与验证。**无** reshape/contiguous/cat/to()/clone 等 aclnn 触发操作。

### 1.5 数据流图

```
GM[IndexQ)  --T.copy--> UB(q_bf16) --auto-cast--> UB(q_fp32) --×sm_scale--> UB --auto-cast--> L1(q_scaled_l1)
GM[Weights] --T.copy--> UB(w_fp32)
GM[IndexK]  --逐行条件 T.copy（无效行保持 0）--> UB(gather_ub) --整块 T.copy--> L1(k_gathered_l1)
GM[TopkIndices] --T.copy--> UB(indices_ub) --标量读--> pos
GM[AttnScore/IndexScore] --T.copy--> UB(attn_ub / idx_score_ub)

L1(k_gathered_l1) × L1(q_scaled_l1) --T.gemm_v0(transpose_B)--> L0C(logits_l0c)
L0C(logits_l0c) --T.copy(enable_relu=True)--> UB(logits_ub)                    # relu 在搬运中完成
UB: d_logits = (IndexScore-AttnScore) ⊗ relu'(logits>0) ⊗ Weights              # 向量化 broadcast+select
UB(d_weights_i) = (IndexScore-AttnScore) ⊗ logits_ub --T.reduce_sum(dim=0,clear=False)--> UB(dw_frag)
UB(d_logits) --auto-cast--> L1(dlog_l1 / dlog_t_l1)
L1(dlog_t1)^T × L1(k_gathered_l1) --T.gemm_v0(transpose_A, init=False)--> L0C(dq_l0c)  # 跨 topk 块累加
L1(dlog_l1) × L1(q_scaled_l1) --T.gemm_v0(init=True)--> L0C(dk_l0c)

L0C(dq_l0c) --T.copy--> UB(dq_fp32) --×sm_scale--> UB --T.copy(auto-cast)--> GM[dIndexQ]
UB(dw_frag) --T.copy(auto-cast)--> GM[dWeights]
L0C(dk_l0c) --T.copy(auto-cast)--> UB(dk_bf16) --逐行 T.tile.atomic_add--> GM[dIndexK]   # host torch.zeros 预清零
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer（用户明确指定，禁止更改）

### 2.2 选型理由

算子为 CV 融合（3 GEMM + 向量后处理 + DMA atomic scatter），结构与 `examples/attention_sink/example_gqa_sink_bwd_bhsd/example_gqa_sink_bwd_bhsd.py` 的 Kernel 2（flashattn_bwd，Developer 模式单 kernel bwd：Phase0 Vector + KV-loop 5 GEMM + atomic dK/dV）高度同构。该先例已验证：Developer 模式下 `T.alloc_shared/alloc_fragment` + `T.gemm_v0` + `T.tile.atomic_add` + 4 个自动 pass_configs 可覆盖全部所需能力，无需手动 `T.Scope`/flag。自动同步由 `TL_ASCEND_AUTO_CV_SYNC` / `TL_ASCEND_AUTO_SYNC` 注入，降低 Stage 2 实现风险。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（编译器映射 L1/UB）+ `T.alloc_fragment`（L0C）；不使用 `alloc_L1/ub/L0A/L0B/L0C` 显式层级 |
| 计算方式 | Cube：`T.gemm_v0`；Vector：`T.tile.sub/mul/broadcast/compare/select`、`T.reduce_sum` |
| 作用域 | 无 `T.Scope`，编译器自动分离 C/V 指令（AUTO_CV_COMBINE） |
| 同步方式 | 全自动（AUTO_SYNC + AUTO_CV_SYNC），零手动 flag |
| threads | `threads=1`（无 vid；gqa_sink_bwd:317 先例；`threads=2` 留作 Stage 3 调优手段） |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `q_scaled = IndexQ[t] × sm_scale` | 每 token [H,D]，fp32 乘后 cast bf16 入 L1 |
| 2 | `K_gathered[i,:] = IndexK[bos+pos_i, :]`（无效 pos → 0） | 逐行 gather，[block_I, D] |
| 3 | `logits = relu(K_gathered × q_scaled^T)` | GEMM1：[block_I,D]×[D,H] → [block_I,H] fp32 |
| 4 | `diff = IndexScore - AttnScore` | [block_I] fp32 |
| 5 | `d_logits = diff ⊗ 1{logits>0} ⊗ Weights` | [block_I,H] fp32，两次 cast bf16 入 L1 |
| 6 | `dWeights[t] += Σ_i d_weights_i`，`d_weights_i = diff ⊗ logits` | reduce_sum(dim=0, clear=False) 跨 topk 块累加 |
| 7 | `dIndexQ[t] += d_logits^T × K_gathered`（后补 ×sm_scale） | GEMM2：[H,block_I]×[block_I,D] → [H,D] |
| 8 | `dk_block = d_logits × q_scaled` | GEMM3：[block_I,H]×[H,D] → [block_I,D] |
| 9 | `dIndexK[bos+pos_i,:] += dk_block[i,:]`（有效行） | 逐行 DMA atomic，bf16 |
| 10 | 尾部：`dIndexQ ×= sm_scale`，写回 3 个输出 | dq 补乘（步骤 7 注） |

### 3.2 TileLang API 映射（GPU → Ascend，全部经源码/示例核实）

| 步骤 | GPU 版 API（ref_gpu 行号） | Ascend API | 证据 |
|------|---------------------------|-----------|------|
| Kernel 声明 | `T.Kernel(seq_len, threads=128)`（L66） | `T.Kernel(seq_len, threads=1, is_npu=True) as (cid)` | gqa_sink_bwd:317；threads 限 1/2（ascend-constraints §1） |
| 动态 shape | `T.dynamic("seq_len, batch_plus_one")`（L41） | `T.symbolic("seq_len")` + `T.symbolic("batch_plus_one")`（两条独立声明） | sparse_mla:104-106、lightning_indexer_dynamic_shape:18-20；grep 无 `def dynamic` |
| 输出声明 | `T.empty(...)`（L57-59） | kernel 参数全显式传入（host `torch.zeros` 预分配，dIndexK 需零初始化） | unsorted_segment_sum:247-248、gqa_sink_bwd:531-535 |
| 共享内存 | `T.alloc_shared`（L71-72） | `T.alloc_shared`（Developer，编译器映射 L1/UB） | gqa_sink_bwd:329-362 |
| fragment | `T.alloc_fragment`（L74-75） | `T.alloc_fragment`（映射 L0C） | gqa_sink_bwd:339-343 |
| Q 预乘 sm_scale | `T.Parallel` 逐元素（L82-83） | `T.copy` auto-cast 到 fp32 UB + `T.tile.mul(q_fp32, q_fp32, sm_scale)` + auto-cast 入 L1 | gqa_sink_bwd:383-384（auto-cast 先例）、447（标量 mul 先例） |
| 动态行 gather | `T.if_then_else(cond, IndexK[bos+pos,j], 0)`（L95） | `T.tile.fill(gather_ub, 0)` + `if (pos>-1)&(pos<=i_t): T.copy(IndexK[bos+pos,:], gather_ub[i,:])` | 动态 if 内 T.copy：unsorted:42、sparse_mla:383；fill：sparse_mla:84 |
| GEMM | `T.gemm(A,B,C,transpose_A/B,clear_accum)`（L104/135/147） | `T.gemm_v0(A,B,C,transpose_A/B,init)`（A/B 在 L1，C 在 L0C；init=False 累加） | ascend.py:418-463（bf16 支持 L434、fp32 累加 L439）；gqa_sink_bwd:400/434/469 |
| relu | `T.max(logits, 0)`（L113） | `T.copy(logits_l0c, logits_ub, enable_relu=True)`（L0C→UB 搬运中完成） | copy_op.py:260；lightning_indexer:54-60 |
| reduce | `T.reduce_sum(d_w_i, d_w_frag, dim=0, clear=False)`（L119） | `T.reduce_sum(buffer, out, dim=0, clear=False)`（同签名） | reduce_ascend.py:452-500（clear 参数 L474-475） |
| d_logits 构造 | `T.Parallel` 逐元素 + `T.alloc_var`（L125-131） | `T.tile.sub` + `T.tile.broadcast(axis=1/0)` + `T.tile.compare("GT")` + `T.tile.select(...,0.0,"VSEL_TENSOR_SCALAR_MODE")` + `T.tile.mul` | gqa_sink_bwd:404-423（broadcast/compare/select 全先例） |
| d_logits cast | `T.copy(fp32_frag, bf16_shared)`（L134/145） | `T.copy(dlog_ub, dlog_l1)`（fp32→bf16 auto-cast + UB→L1 直连） | gqa_sink_bwd:383/427（auto-cast、UB→L1 direct 先例） |
| atomic scatter | `T.atomic_add(dIndexK[bos+pos,j], scalar)`（L159） | `T.tile.atomic_add(dIndexK[bos+pos, 0], dk_bf16_ub[i,:])`（GM region ← local tile，dtype 匹配 bf16） | ascend_tile.py:313-349；unsorted:50、gqa_sink_bwd:437/472、sparse_mla:1104 |
| 流水线 | `T.Pipelined(num_blocks, num_stages=0)`（L85） | `T.serial(NI)`（num_stages=0 本无流水；NI 编译期常量） | ascend-constraints §1（Pipelined 需静态边界） |
| 标量 GM 读 | `TokenIndices[bx,0]`、`Offsets[i_b]`（L67-68） | 同语法（GM 标量读，量少：2 次/token） | sparse_mla:289-291、425-427 |

### 3.3 计算伪代码

```python
seq_len, batch_plus_one = T.symbolic("seq_len"), T.symbolic("batch_plus_one")
NI = topk // block_I  # 编译期常量（assert topk % block_I == 0）

with T.Kernel(seq_len, threads=1, is_npu=True) as (cid):
    i_b, i_t = TokenIndices[cid, 0], TokenIndices[cid, 1]      # GM 标量读 ×2
    bos = Offsets[i_b]

    # ---- 分配（Developer：shared→L1/UB，fragment→L0C）----
    q_scaled_l1  = T.alloc_shared([H, D], bf16)                # L1
    k_l1         = T.alloc_shared([block_I, D], bf16)          # L1
    dlog_l1      = T.alloc_shared([block_I, H], bf16)          # L1（dk GEMM 的 A）
    dlog_t_l1    = T.alloc_shared([block_I, H], bf16)          # L1（dq GEMM 的 A，transpose_A=True）
    logits_l0c   = T.alloc_fragment([block_I, H], fp32)        # L0C
    dq_l0c       = T.alloc_fragment([H, D], fp32)              # L0C（跨 NI 累加）
    dk_l0c       = T.alloc_fragment([block_I, D], fp32)        # L0C
    gather_ub    = T.alloc_shared([block_I, D], bf16)          # UB（K gather staging）
    indices_ub   = T.alloc_shared([block_I], int32)            # UB
    attn_ub, is_ub = T.alloc_shared([block_I], fp32) ...       # UB
    logits_ub, dw_ub, dlog_ub, mask_ub, diff2d, w2d = ... [block_I, H] fp32
    dw_frag      = T.alloc_shared([H], fp32)                   # UB
    dq_ub        = T.alloc_shared([H, D], fp32)                # UB
    dk_ub        = T.alloc_shared([block_I, D], bf16)          # UB

    # ---- per-token 常量装载 ----
    T.copy(IndexQ[bos+i_t, :, :], q_ub_bf16)                   # [H,D] 4KB
    T.copy(q_ub_bf16, q_ub_fp32)                               # auto-cast
    T.tile.mul(q_ub_fp32, q_ub_fp32, sm_scale)                 # fp32 乘（≥ 主仓 bf16 乘精度）
    T.copy(q_ub_fp32, q_scaled_l1)                             # auto-cast bf16 入 L1
    T.copy(Weights[bos+i_t, :], w_ub)                          # [H]
    T.fill(dw_frag, 0.0)

    for it in T.serial(NI):
        st = it * block_I
        T.copy(TopkIndices[bos+i_t, st:st+block_I], indices_ub)
        T.copy(AttnScore[bos+i_t, st:st+block_I], attn_ub)
        T.copy(IndexScore[bos+i_t, st:st+block_I], is_ub)
        T.tile.sub(diff_ub, is_ub, attn_ub)                    # [block_I] fp32

        # ---- gather（无效行保持 0，等价 GPU if_then_else(...,0)）----
        T.tile.fill(gather_ub, 0)
        for i in T.serial(block_I):
            pos = indices_ub[i]                                # UB 标量读
            if (pos > -1) and (pos <= i_t):
                T.copy(IndexK[bos+pos, :], gather_ub[i, :])    # 256B/行
        T.copy(gather_ub, k_l1)                                # 整块 UB→L1

        # ---- GEMM1: logits = relu(K × Q_scaled^T) ----
        T.gemm_v0(k_l1, q_scaled_l1, logits_l0c, transpose_B=True, init=True)
        T.copy(logits_l0c, logits_ub, enable_relu=True)        # relu 随搬运完成

        # ---- Vector: diff2d / w2d / d_logits / d_weights_i ----
        T.tile.broadcast(diff2d, diff_ub, axis=1)              # [block_I,H]
        T.tile.broadcast(w2d, w_ub, axis=0)                    # [block_I,H]
        T.tile.mul(dw_ub, diff2d, logits_ub)                   # d_weights_i
        T.reduce_sum(dw_ub, dw_frag, dim=0, clear=False)       # 跨 NI 累加
        T.tile.mul(dlog_ub, diff2d, w2d)                       # diff ⊗ Weights
        T.tile.compare(mask_ub, logits_ub, zero_ub, "GT")      # 1{logits>0}
        T.tile.select(dlog_ub, mask_ub, dlog_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")

        # ---- GEMM2: dq += d_logits^T × K ----
        T.copy(dlog_ub, dlog_t_l1)                             # fp32→bf16 auto-cast
        T.gemm_v0(dlog_t_l1, k_l1, dq_l0c, transpose_A=True, init=(it == 0))

        # ---- GEMM3: dk = d_logits × Q_scaled ----
        T.copy(dlog_ub, dlog_l1)
        T.gemm_v0(dlog_l1, q_scaled_l1, dk_l0c, init=True)

        # ---- atomic scatter dk ----
        T.copy(dk_l0c, dk_ub)                                  # fp32→bf16 auto-cast
        for i in T.serial(block_I):
            pos = indices_ub[i]
            if (pos > -1) and (pos <= i_t):
                T.tile.atomic_add(dIndexK[bos+pos, 0], dk_ub[i, :])   # 256B DMA atomic

    # ---- 尾部写回 ----
    T.copy(dq_l0c, dq_ub)                                      # L0C→UB fp32
    T.tile.mul(dq_ub, dq_ub, sm_scale)                         # dq 补乘（§1.3 注）
    T.copy(dq_ub, dIndexQ[bos+i_t, :, :])                     # auto-cast bf16
    T.copy(dw_frag, dWeights[bos+i_t, :])                      # auto-cast bf16
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| `T.gemm_v0`（bf16 操作数 / fp32 累加 / transpose_A/B / init） | tilelang/language/ascend.py:418-519 | 本仓示例已验证（gqa_sink_bwd:400/434/469、sparse_mla:270） |
| `T.tile.atomic_add`（dst=GM / src=local / bf16） | tilelang/language/ascend_tile.py:313-349 | 本仓示例已验证（unsorted_segment_sum:50、gqa_sink_bwd:437/472） |
| `T.copy`（auto-cast / enable_relu / 动态 if 内） | tilelang/language/copy_op.py:260 | gqa_sink_bwd:381-384、lightning_indexer:54-60、unsorted:42 |
| `T.reduce_sum(buffer, out, dim, clear=False)` | tilelang/language/reduce_ascend.py:452-500 | gqa_sink_bwd:386（clear 默认路径） |
| `T.tile.sub/mul/broadcast/compare/select/fill` | ascend_tile.py（sub L1106、max L1173 等，DESIGN_panel_mode.md:799 已核实索引） | gqa_sink_bwd:404-423、sparse_mla:84 |
| `T.symbolic` | tilelang/language/__init__.py:89 | sparse_mla:104-106 |
| GM 标量读（`TokenIndices[cid,0]`） | TIR BufferLoad 标量 | sparse_mla:289-291、425-427 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | GPU 版本就是一维 `T.Kernel(seq_len)`，直接对应 `T.Kernel(seq_len, is_npu=True)` |
| threads 参数限制（仅 1 或 2） | **Yes**（GPU threads=128） | `threads=1`（gqa_sink_bwd:317 先例）；`threads=2` 列入 Stage 3 调优 |
| 动态循环边界不支持 | No | `NI = topk // block_I` 为编译期 Python 常量；`if pos <= i_t` 是分支判断非循环边界 |
| 流水线不支持动态边界 | No | 不使用 T.Pipelined（GPU 版 num_stages=0 本无流水）；用 T.serial |
| L0C 容量上限 128KB | No（已验算） | logits[64,16]+dq[16,128]+dk[64,128] fp32 = 4+8+32 = 44KB < 128KB |
| T.copy 列方向 strided 切片 | No | 所有 GM 切片均为最内维（D=128 连续整行 / 整行段），无 4D+ 列 strided |
| GPU 专用 API | **Yes**（见 3.5.2） | 全部替换为 Ascend 等价 API（3.2 表） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU，ref_gpu/indexer_bwd.py） | 本项目（Ascend） | 转换方案 |
|--------|----------------------------------------|-----------------|----------|
| Kernel 签名 | `T.Kernel(seq_len, threads=128)`（L66） | `T.Kernel(seq_len, threads=1, is_npu=True) as (cid)` | NPU threads 限制；block↔core 由 runtime 分发（2048 block / 20 core） |
| 动态 shape | `T.dynamic("seq_len, batch_plus_one")`（L41） | 2× `T.symbolic` | Ascend 无 T.dynamic 逗号语法 |
| 输出分配 | `T.empty`（L57-59，依赖 atomic 前 zero 假设） | host `torch.zeros` 显式预清零 + 参数传入 | Ascend `T.tile.atomic_add` 语义要求 dst 预初始化；先例 unsorted:247 |
| GEMM | `T.gemm(..., clear_accum=)` | `T.gemm_v0(..., init=)` | init=True 清零 / init=False 累加 ↔ clear_accum |
| atomic | `T.atomic_add(GM[idx], scalar)`（标量，fp32→bf16 隐式） | `T.tile.atomic_add(GM 行 region, UB 行 tile)`（src/dst dtype 必须一致=bf16） | 每 block 部分和先 auto-cast bf16 再整行 atomic；与 GPU 版"fp32 部分和落 bf16 后累加"精度路径一致 |
| gather | `T.if_then_else` 逐元素（L95） | `T.tile.fill(0)` + 动态 `if` 内逐行 `T.copy` | 无效行保持 0，语义等价 |
| relu | `T.max(x, 0)` 逐元素（L113） | `T.copy(..., enable_relu=True)`（L0C→UB） | lightning_indexer:54-60 先例 |
| d_relu 指示 | `if logits>0: 1.0 else 0.0`（L127-130） | `T.tile.compare("GT")` + `T.tile.select(..., 0.0)` | gqa_sink_bwd:419-423 先例 |
| 逐元素循环 | `T.Parallel(block_I, dim)` 等 | `T.tile.*` 向量化指令（broadcast/mul/sub） | Vector 引擎整 tile 运算 |
| `T.alloc_var` 标量 | `d_relu = T.alloc_var(accum_dtype)`（L126） | 消除——mask/select 向量化 | 无逐元素标量循环 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| examples/attention_sink/example_gqa_sink_bwd_bhsd/example_gqa_sink_bwd_bhsd.py | **高度相似**（Developer bwd + gemm_v0 ×3 + T.tile.atomic_add 到 GM + auto-cast + 4 True pass_configs + threads=1） | Kernel 2 flashattn_bwd 整体结构、atomic 输出 host zeros 预分配（L531-535） |
| examples/unsorted_segment_sum/unsorted_segment_sum.py | 高（atomic_add scatter 累加 + 动态 if 内 T.copy + host zeros 输出） | atomic_add_kernel（L36-52）、_atomic_impl（L232-248） |
| examples/dsa_sparse_finetune/sparse_mla_topk_reducesum.py | 高（同族 fwd：varlen + TokenIndices/Offsets + gather 行号向量化 + 动态标量 if + bf16） | prepare_token_indices（L1278-1296）、rows 向量化（L328-335）、`if s < nblk` 动态分支（L383） |
| examples/lightning_indexer/example_lightning_indexer_dynamic_shape.py | 中（indexer 前向同族：T.symbolic 动态 shape + enable_relu + gemm_v0 transpose_B） | L18-33（symbolic + Kernel）、L54-60（enable_relu） |
| examples/deepseek_v4/lightning_indexer.py | 中（同族 fwd 静态版） | gemm_v0 用法 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| IndexQ | [seq_len, heads, dim] | bfloat16 | ΣS=S=2048, H=16, D=128 |
| Weights | [seq_len, heads] | bfloat16 | |
| IndexK | [seq_len, dim] | bfloat16 | 同一 K 空间被 varlen 各 batch 引用（bos 偏移） |
| AttnScore | [seq_len, topk] | float32 | teacher 概率 |
| IndexScore | [seq_len, topk] | float32 | student log_softmax 概率 |
| TopkIndices | [seq_len, topk] | int32 | 值域 [-1, seq_len)，-1 表示无效 |
| Offsets | [batch_plus_one] | int32 | varlen cu_seqlens |
| TokenIndices | [seq_len, 2] | int32 | host 预计算 (i_b, i_t)，CPU 构造 + H2D |

### 4.2 输出张量（host `torch.zeros` 预分配后传入 kernel，无 out_idx）

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| dIndexQ | [seq_len, heads, dim] | bfloat16 | 每 token 行独占写（bos+i_t 唯一） |
| dWeights | [seq_len, heads] | bfloat16 | 每 token 行独占写 |
| dIndexK | [seq_len, dim] | bfloat16 | **atomic scatter-add，必须零初始化** |

### 4.3 中间缓冲区（block_I=64, H=16, D=128, topk=64）

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| q_scaled_l1 | [16, 128] | bf16 | L1（alloc_shared） | Q × sm_scale，GEMM1/3 操作数 |
| k_l1 | [64, 128] | bf16 | L1 | gather 后的 K 块，GEMM1/2 操作数 |
| dlog_t_l1 / dlog_l1 | [64, 16] | bf16 | L1 | d_logits cast，GEMM2/3 操作数 |
| logits_l0c | [64, 16] | fp32 | L0C（alloc_fragment） | GEMM1 累加器 |
| dq_l0c | [16, 128] | fp32 | L0C | GEMM2 累加器（init=(it==0)，跨 NI 累加） |
| dk_l0c | [64, 128] | fp32 | L0C | GEMM3 累加器 |
| gather_ub | [64, 128] | bf16 | UB | K gather staging（无效行 0） |
| indices_ub / attn_ub / is_ub / diff_ub | [64] | int32 / fp32 | UB | 索引与分数 |
| logits_ub / dw_ub / dlog_ub / mask_ub / diff2d / w2d | [64, 16] | fp32 | UB | 向量后处理 |
| q_ub_fp32 / dq_ub | [16, 128] | fp32 | UB | sm_scale 乘法中转 |
| dk_ub | [64, 128] | bf16 | UB | atomic 源（dtype 与 dIndexK 匹配） |
| w_ub / dw_frag / zero_ub | [16] / [64] | fp32 | UB | 广播源 / dWeights 累加器 / 比较零点 |

### 4.4 内存搬运路径（GM ↔ L1/UB ↔ L0A/L0B → L0C，无跨级违规）

```
GM[IndexQ]     --T.copy--> UB(bf16) --auto-cast--> UB(fp32) --T.tile.mul(sm_scale)--> UB --auto-cast--> L1(q_scaled_l1)
GM[Weights]    --T.copy--> UB(w_ub, auto-cast fp32)
GM[IndexK]     --(逐行条件 T.copy, 256B/行)--> UB(gather_ub bf16) --T.copy--> L1(k_l1)
GM[TopkIndices/AttnScore/IndexScore] --T.copy--> UB(indices/attn/is_ub)
L1(q_scaled_l1) ┐
L1(k_l1)        ┘--gemm_v0(内部 L1→L0A/L0B)--> L0C(logits_l0c) --T.copy(enable_relu)--> UB(logits_ub)
UB(dlog_ub) --auto-cast--> L1(dlog_l1 / dlog_t_l1)
L1(dlog_t_l1)+L1(k_l1) --gemm_v0--> L0C(dq_l0c) --T.copy--> UB(dq_ub) --×sm_scale--> GM[dIndexQ](auto-cast)
L1(dlog_l1)+L1(q_scaled_l1) --gemm_v0--> L0C(dk_l0c) --T.copy(auto-cast)--> UB(dk_ub bf16) --T.tile.atomic_add--> GM[dIndexK]
UB(dw_frag) --T.copy(auto-cast)--> GM[dWeights]
```

无 GM→L0 直搬；所有 Cube 操作数经 L1（gemm_v0 内部管理 L1→L0A/L0B）；所有 Vector 数据在 UB。✓ 符合硬件内存层级。

### 4.5 UB 内存预算（block_I=64）

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| gather_ub | [64,128] | bf16 | 16384 |
| dk_ub | [64,128] | bf16 | 16384 |
| logits_ub / dw_ub / dlog_ub / mask_ub / diff2d / w2d | [64,16] ×6 | fp32 | 6×4096 = 24576 |
| q_ub_fp32 / dq_ub | [16,128] ×2 | fp32 | 2×8192 = 16384 |
| indices/attn/is/diff/zero_ub 等 1D | [64] ×~5 | fp32/int32 | ~1280 |
| w_ub / dw_frag | [16] ×2 | fp32 | 128 |
| **总计（峰值，含编译器 double-buffer 余量 ×1.5）** | | | **~82KB / 192KB（A2/A3 UB 容量） ✓** |

L1 峰值：q_scaled 4KB + k 16KB + dlog ×2 2KB ≈ 22KB ≪ 512KB ✓；L0C：44KB < 128KB ✓。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| seq_len（ΣS） | `T.symbolic("seq_len")` | topk ≤ ΣS（主仓 assert 每 batch 段长 ≥ topk），典型 ≤ 64K |
| batch_plus_one | `T.symbolic("batch_plus_one")` | 2 ~ 1025 |

grid = seq_len（动态，`T.Kernel(seq_len, ...)`，先例 lightning_indexer:33 `T.Kernel(B * N2)` B 为 symbolic）。

### 4.7 JIT 配置

```python
@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # C/V 自动分核合并
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # CV 交叉自动同步
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # MTE/Vector 自动同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # UB/L1 内存规划
})
# 无 out_idx / workspace_idx：3 个输出由 host torch.zeros 预分配显式传入
#（dIndexK 的 atomic 语义要求零初始化；先例 unsorted_segment_sum、gqa_sink_bwd）
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 混合（CV 融合）——Cube（3 GEMM/token）+ Vector（broadcast/compare/select/reduce）+ MTE（gather/DMA atomic）。

### 5.2 Block 划分

```python
block_I = 64    # topk 方向 tile = min(topk, 64)；主仓默认 32，本设计取 64：
                #   NI = topk // block_I = 1（topk=64），单迭代消除 2 次循环的
                #   gather/gemm/atomic 序列开销；GEMM M 维 32→64 提升 Cube 效率
block_H = H     # heads 整维进 tile（H=16 满足 16 对齐）
block_D = D     # dim=128 整维（bf16 尾轴 128 ≥ 16 对齐 ✓）
grid = seq_len  # 每 block 1 token（与主仓一致；varlen 行号由 TokenIndices 解码）
```

约束（kernel 入口 assert，继承主仓语义 + Ascend 对齐）：
- `topk % block_I == 0`（主仓 L39 同款；保证无 topk 尾块）
- `heads % 16 == 0 and heads <= 64`（Ascend Cube N/K 16 对齐——主仓为 %8；**差异显式声明**：H=8 不在本算子支持域，见 §10.3）
- `dim % 16 == 0`（K/N 对齐；D=128 ✓）
- `block_I % 16 == 0`（M 对齐）
- 每 batch 段长 ≥ topk（主仓 ref L195 同款语义）

### 5.3 约束分析

- **对齐约束**: GEMM1 [block_I=64, H=16] K=128；GEMM2 [H=16, D=128] K=block_I=64；GEMM3 [block_I=64, D=128] K=H=16。M/N/K 全部 16 的倍数 ✓（gemm_v0 n_actual 注释 ascend.py:456-459 "16-column fractal granularity" 佐证 16 列分形粒度）。
- **UB 容量**: ~82KB（含余量）< 192KB ✓
- **L0C 容量**: 44KB < 128KB ✓（A2/A3）
- **GEMM 非整除**: 不涉及——seq_len 维每 block 恰 1 token（无 M 尾块）；topk % block_I == 0（无 K 尾块）；H/D 整维进 tile。**唯一动态边界是因果 mask `pos <= i_t` 与 `pos == -1`，以 fill-0 + 分支处理，不产生尾块。**

### 5.4 注意事项

- 无效 pos（-1 或 > i_t）的 K 行保持 gather_ub 预填的 0：logits=0 → d_weights 贡献 0、d_relu=0 → d_logits=0 → dq/dk 贡献 0，且 atomic 跳过该行。与主仓 `T.if_then_else(..., 0)`（L95）逐位等价。
- dIndexK 目标行 `bos+pos` 在有效 pos 下必在 [bos, bos+段长) 内（pos ≤ i_t < 段长），无越界写。
- L0 用例（topk_indices=arange 平铺）下 token t < 64 时存在大量无效 pos——正确性由上述 0 填充保证，性能由分支跳过自然获益。

### 5.5 数据搬运性能可行性

```text
[REORDER-COST]
path: indexer_bwd 主路径（per-token gather + atomic scatter）
gm_passes: 读 IndexQ 1 + IndexK gather topk/D·2B·S + Scores 3×topk·4B·S；写 dq/dw 1 + dk atomic topk·D·2B·S
dma_transactions: gather 131072 条 MTE2（S×topk 行 × 256B，平均 256B/条）+ atomic 131072 条 MTE3（同规模）
                + 每 token 大块 DMA ~7 条（Q/Weights/Indices/Scores×2/dq/dw）≈ 14336 条
gm_scalar_accesses: 2 次/token（TokenIndices）+ block_I 次/token UB 标量读 pos（非 GM）≈ 4096 次 GM 标量
address_div_mod_per_element: 0（行号加法预计算，无 div/mod）
active_cores / serial_tasks_per_core: 20 AIC / ~102 token（grid=2048）
largest_case_timeout_gate: l0_default（S=2048）120s
result: pass（临界——依赖 MTE 与 Cube 流水重叠 + L2 命中，详见 §9.4 性能预算与 §10.1 风险 R5）
```

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| token 维 | block 级并行 | `T.Kernel(seq_len, threads=1, is_npu=True)` | 每 block 1 token，与主仓一致；2048 block 由 runtime 分发到 20 核 |
| topk 维 | 编译期常量迭代 | `T.serial(NI)`（NI=1 @ topk=64, block_I=64） | 不使用 T.Pipelined（主仓 num_stages=0 本无流水；Ascend Pipelined 要求静态边界，此处虽静态但无收益） |
| block_I 内 gather/atomic 行循环 | 编译期常量迭代 | `T.serial(block_I)` + 动态 `if` | 逐行条件搬运/scatter；动态 if 内 T.copy/T.tile.atomic_add 有先例（unsorted:42/50） |
| 元素级 | 向量化 | `T.tile.*` 整 tile 指令 | 无逐元素 T.Parallel 标量循环 |

### 6.2 循环伪代码

见 §3.3（完整结构：外层 Kernel(grid) → T.serial(NI) → T.serial(block_I) gather → gemm → vector → T.serial(block_I) atomic）。

### 6.3 流水线优化

L0 不启用 T.Pipelined。潜在优化（Stage 3）：NI ≥ 2 时对 gather（MTE2）与 GEMM（Cube）做双缓冲流水；AUTO_CV_SYNC 已自动处理同核 C/V/MTE 依赖排队，MTE 与 Cube 天然异引擎可重叠。

### 6.4 尾块处理

无 shape 尾块（§5.3）。动态有效域（causal mask / pos=-1）通过 fill-0 + `if` 分支处理，等价主仓语义。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 全自动同步（Developer）——零手动 `T.set_flag/wait_flag/barrier_all/cross_flag`。

### 7.2 自动同步点（由 pass 注入，设计层列出依赖关系）

| 依赖 | 保证方 |
|------|--------|
| gather_ub 写完 → 整块 UB→L1 copy | AUTO_SYNC（MTE2→MTE2 序） |
| k_l1/q_scaled_l1 就绪 → gemm_v0 | AUTO_CV_SYNC（MTE1→Cube flag 序，gemm_v0 内建） |
| logits_l0c 定界 → L0C→UB relu copy | AUTO_CV_SYNC（FIX→V flag 序） |
| dlog_ub 就绪 → UB→L1 cast copy | AUTO_SYNC |
| dq_l0c 跨 NI 累加次序 | 程序序（同核串行）+ init=False 语义 |
| atomic_add 与后续 token 对同 GM 行的竞争 | **硬件 DMA 原子性**（tl.ascend_atomic_add 底层原子语义） |
| host 读回（torch.npu.synchronize） | runtime stream 同步 |

### 7.3 pass_configs 配置

见 §4.7（4 True，gqa_sink_bwd:62-67 逐键一致）。

---

## 8. 融合算子设计

### 8.1 融合算子判定

**判定结果**: 是（GEMM + element-wise 后处理 + scatter-atomic 单 kernel 融合）

### 8.2 CV 交互设计（Developer 模式）

- `T.Kernel(seq_len, threads=1, is_npu=True) as (cid)`（单轴，无 vid）
- 装饰器无 `workspace_idx`，签名无 workspace 参数——**不产出 workspace 规格**
- Cube↔Vector 片上直连：`T.copy(logits_l0c, logits_ub)`（L0C→UB direct）、`T.copy(dlog_ub, dlog_l1)`（UB→L1 direct，gqa_sink_bwd:427/462 "on-chip direct" 先例）
- 中转/同步交给 4 个自动 pass

### 8.3 Cube 核计算流程

```python
T.gemm_v0(k_l1, q_scaled_l1, logits_l0c, transpose_B=True, init=True)   # GEMM1 logits
T.gemm_v0(dlog_t_l1, k_l1, dq_l0c, transpose_A=True, init=(it == 0))    # GEMM2 dq（跨 NI 累加）
T.gemm_v0(dlog_l1, q_scaled_l1, dk_l0c, init=True)                      # GEMM3 dk
```

### 8.4 Vector 核计算流程

```python
T.copy(logits_l0c, logits_ub, enable_relu=True)          # L0C→UB + relu
T.tile.sub / broadcast / mul / compare / select          # d_logits、d_weights_i
T.reduce_sum(dw_ub, dw_frag, dim=0, clear=False)         # dWeights 跨 NI 累加
T.tile.mul(dq_ub, dq_ub, sm_scale)                       # dq 尾部缩放
T.copy(dq_ub, dIndexQ[...]); T.copy(dw_frag, dWeights[...])   # 写回（auto-cast bf16）
T.tile.atomic_add(dIndexK[bos+pos, 0], dk_ub[i, :])      # 逐行 scatter-atomic
```

### 8.5 pass_configs 配置

见 §4.7。

### 8.6 注意事项

- Developer 默认消除 workspace/vid：`threads=1`（gqa_sink_bwd 同款；`threads=2` 是消 vid 的备选形态，列入 Stage 3 调优——AIV0 专职 gather/atomic、AIV1 专职 vector 的手工分工需 Expert 回退，非 L0 范围）
- dlog_t_l1 与 dlog_l1 分配两个独立 buffer（2KB×2）：dq GEMM 以 transpose_A=True 读、dk GEMM 以不转置读，同一 L1 buffer 两种 fractal 读法无先例，双 buffer 规避风险
- atomic_add 的 src 必须与 dst dtype 一致（bf16）——dk 部分和先 auto-cast 再 atomic

---

## 9. 验证方案

### 9.1 Golden 函数（复刻主仓 ref_indexer_bwd，逐字节一致）

```python
def ref_indexer_bwd(Q, Weights, K, TopkIndices, AttnScore, offsets):
    """主仓 ref_gpu/indexer_bwd.py:185-215 逐字节复刻（autograd 参考实现）。
    在 NPU 上运行（与主仓在 GPU 上运行 golden 等价；torch_npu 支持 bf16 autograd/
    einsum/kl_div，主仓 test_kernel 同款流程 cuda→npu）。"""
    Q.requires_grad_(True)
    Weights.requires_grad_(True)
    K.requires_grad_(True)
    softmax_scale = Q.shape[-1] ** -0.5
    all_loss, all_log_topk_prob = [], []
    for i in range(offsets.shape[0] - 1):
        assert (offsets[i + 1] - offsets[i]).item() >= TopkIndices.shape[-1]
        q, weights, k = Q[offsets[i]:offsets[i+1]], Weights[offsets[i]:offsets[i+1]], K[offsets[i]:offsets[i+1]]
        topk_indices, attn_score = TopkIndices[offsets[i]:offsets[i+1]], AttnScore[offsets[i]:offsets[i+1]]
        s = q.shape[0]
        mask = (torch.arange(s)[:, None] >= torch.arange(s)[None, :]).to(q.device)
        logits = einsum(q, k, "s1 h k, s2 k -> s1 h s2") * softmax_scale
        logits = F.relu(logits)
        score = (logits * weights.unsqueeze(-1)).sum(dim=-2, dtype=torch.float32)
        score = torch.where(mask, score, float("-inf"))
        topk_value = torch.gather(score, dim=-1, index=topk_indices.to(torch.int64))
        log_topk_prob = F.log_softmax(topk_value, dim=-1, dtype=torch.float32)
        loss = F.kl_div(log_topk_prob.clip(-100, 0), attn_score.log().clip(-100, 0),
                        log_target=True, reduction="sum")
        all_loss.append(loss); all_log_topk_prob.append(log_topk_prob)
    loss = torch.stack(all_loss).sum()
    loss.backward()
    log_topk_prob = torch.cat(all_log_topk_prob, dim=0)
    return log_topk_prob.exp(), Q.grad, Weights.grad, K.grad
```

**输入构造（复刻主仓 test_kernel:218-244）**：`torch.manual_seed(42)`；`q/w/k = torch.randn(...).bfloat16()`（NPU）；`offsets = torch.tensor([0, 1023, S], int32)`；`attn_score` = 对每段 `mask = arange(s)[:,None] >= arange(topk)[None,:]` 的 `softmax(ones masked)`（fp32）；`topk_indices = repeat(arange(topk, int32), "k -> s k")`。golden 的 `index_score` 取 `ref_indexer_bwd` 返回的第一个值（`log_topk_prob.exp()`）——与主仓测试一致。

### 9.2 L0 门槛测试计划（tilelang-op-test-design 场景 A 生成，仅 L0）

**测试策略判定**（场景 A Phase 2）：计算类型 = 混合（CV 融合）；复杂度 = Fusion；数学特征 = matmul + relu + broadcast 乘 + reduce + atomic scatter；综合类别 = **Fusion（反向梯度算子）**。测试重点 = 功能验证 + 精度收敛（L0），`dtype_count=1`（bf16 固定精度路径）、`shape_count=4`（varlen 拓扑 × topk 分块）、`block_count=2`（block_I ∈ {64, 32}）。

| 用例名 | 级别 | Shape（B, S, H, D, topk, block_I） | offsets | dtype | 说明 |
|--------|------|-----------------------------------|---------|-------|------|
| l0_default | L0 | (1, 2048, 16, 128, 64, 64) | [0, 1023, 2048] | bf16 | **主仓默认配置**：varlen 2 段（1023+1025 非对称段长）+ token<64 的 mask 边界 + atomic 跨 token 冲突路径全覆盖 |
| l0_single_seq | L0 | (1, 1024, 16, 128, 64, 64) | [0, 1024] | bf16 | 单 batch 无 varlen 切分，全 pos 有效 |
| l0_small | L0 | (1, 128, 16, 128, 64, 64) | [0, 128] | bf16 | 小规模冒烟：S<256，token<64 占比高（大量无效 pos 的 fill-0 路径） |
| l0_blockI32 | L0 | (1, 512, 16, 128, 64, 32) | [0, 512] | bf16 | block_I=32 → NI=2：dq_l0c 跨块累加（init=(it==0)）+ dWeights clear=False 路径 |

- 全部用例输入构造复刻 §9.1（seed 42 固定）；比对 3 个输出（dq/dw/dk）vs golden（`ref_indexer_bwd` 返回的 Q.grad / Weights.grad / K.grad）。
- 比对方式：混合容差（§9.3），三输出分别判定，任一失败该用例 FAIL。
- **dk 是 atomic 累加输出，bf16 部分和多次累加的舍入与主仓 GPU 版精度路径一致**（主仓同为 fp32 部分和 → bf16 GM atomic），容差内收敛。
- L1（不规则 shape）/ L2（异常）/ Boundary（特殊值）由 Stage 2 在 L0 通过后按场景 B 扩展，本节不生成。

### 9.3 精度标准（混合容差，按 §4 声明 dtype 全集）

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|-------|------|------|---------------------|------------------------|
| bfloat16 | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |
| float32 | 2⁻¹⁶ (1.53e-5) | 2⁻¹⁰ (9.77e-4) | 1e-2 | 0.99 |
| int32 | 0 | 0 | 0 | 1.0（精确匹配） |

> bfloat16 为输出（dIndexQ/dWeights/dIndexK）比对标准；float32 为 AttnScore/IndexScore 输入 dtype（参与计算，无直接输出比对，列入保持 proto 一致性）；int32 为索引输入（TokenIndices/TopkIndices/Offsets）。判定：逐元素 `|actual-golden| ≤ atol + rtol·|golden|` 且整体 `matched_ratio ≥ 0.99` **且** `max_abs_error ≤ max_abs_error_limit`。本算子无 NaN/Inf 输入路径（Scores 为概率值域 [0,1]），L0 不构造特殊值用例。

### 9.4 性能可行性哨兵 + 性能目标（验收指标 2）

**性能目标**：B=1, S=2048, H=16, D=128, topk=64, bf16，**纯 kernel 平均耗时 < 80us**（所有测试样例 kernel 时间总和平摊；排除 H2D/D2D/同步）。

**预算分解（910B3：20 AIC，~2.6GHz，grid=2048 → ~102 token/核）**：

| 项目 | 估算 | 依据 |
|------|------|------|
| Cube：6144 gemm（3/token） | ~1500-4500 cycles/token（3 gemm × 500-1500 cycles，含 flag 序与 L1→L0 搬运） | 小 GEMM 固定开销为主（sparse_mla iter3 经验：gemm 调用含内建 flag 开销） |
| MTE2：K gather 64 行×256B + 每 token ~7 条大块 DMA | ~1300-2000 cycles/token（逐行短 DMA 排队） | 131072 条/全 grid |
| MTE3：dk atomic 64 行×256B | ~1900-3200 cycles/token | 131072 条 DMA atomic |
| Vector：diff/broadcast/select/reduce | ~300-500 cycles/token | [64,16] tile 整拍指令 |
| **串行上界** | ~5000-10000 cycles/token → 102 token → **197-393us** | **超标** |
| **重叠下界**（MTE/Cube/Vector 异引擎流水，AUTO_CV_SYNC 自动排布） | 临界路径 ~max(Cube 3000, MTE 5000) ≈ 3000-5000 cycles/token → **118-197us** | 仍需优化 |
| **达标路径**（3 条依赖） | ① L2 命中：L0 case 的 K gather（64 行热区 16KB）与 dIndexK atomic（64 行热区）经 L2 合并，MTE 有效流量从 64MB 降至 ~18MB；② MTE3 atomic 与下一 token 的 gemm/MTE2 流水重叠；③ gemm_v0 flag 开销实测 ≤800 cycles | 估算 < 80us 可达但**临界** |

| 用例名 | Shape | dtype/属性 | 覆盖路径 | 单 case timeout | 选择理由 |
|--------|-------|------------|----------|-----------------|----------|
| perf_l0_default（=l0_default） | (1, 2048, 16, 128, 64) | bf16, block_I=64 | 主路径全量（gather+3GEMM+atomic） | 120s | 最大任务数 + atomic 冲突峰值 |

**性能风险缓解（Stage 3 调优路径，按优先级）**：
1. `block_I=32` 对比（gemm 更小但 gather/atomic 序列翻倍——双向实测）
2. `threads=2`：AIV 分工（vid0 gather/atomic，vid1 vector）
3. **共享索引快路径**（sparse_mla panel_mode 先例，attr 声明式分派）：调用方声明 topk_indices 跨 token 一致时，T_BLOCK 个 token 共享一份 K gather，3 个 GEMM 升维合并（logits [topk, T×H]、dq [T×H, D]、dk [topk, D]，gemm 数 /T_BLOCK），L0 case（arange 平铺）恰好命中
4. 连续 pos 检测：rows 与 arith_progression 逐位比较全真时整块 atomic（L0 case pos 连续 0..63 → 2 条 [32,128] region atomic 替代 64 条行 atomic）

---

## 10. 风险点与注意事项（6 大迁移风险调研结论）

### 10.1 技术约束检测结论（每项附证据）

| # | 风险 | 结论 | 证据 |
|---|------|------|------|
| R1 | **GM atomic_add 支持性** | ✅ 支持，采用逐行 DMA atomic。`T.tile.atomic_add(dst, src)`：dst 必须 GM scope、src 必须 local（UB/L0C/L1）、dtype 须一致、支持 bf16（tilelang/language/ascend_tile.py:313-349，L215 错误信息佐证 V1 语义）。先例：unsorted_segment_sum.py:50（`atomic_add(Output[seg, col], row_ub)` 逐行 scatter）、gqa_sink_bwd:437/472（L0C→GM region atomic）、sparse_mla:1104。**GPU 标量版 `T.atomic_add`（customize.py:15，call_extern "AtomicAdd"）为 CUDA extern，Ascend 不可用**。清零方案：host `torch.zeros` 预分配（unsorted:247、gqa_sink:534-535 先例），无第二 kernel、无 host 后处理，80us 纯 kernel 指标不含额外 kernel | ascend_tile.py:313-349 + 3 示例 |
| R2 | **动态索引 GM 访问**（`IndexK[bos+pos, j]`） | ✅ 可行，采用 fill-0 + 动态 `if` 内逐行 `T.copy`（GM→UB staging → 整块 UB→L1）。逐元素 `T.if_then_else` GM 标量读模式弃用（GM 标量访问性能差）。动态 if 内 T.copy 先例：unsorted:42（`if row < N`）、sparse_mla:383（`if s < nblk`）。行号从 indices_ub 标量读（UB 标量读先例 sparse_mla:334）。T.copy 列 strided 限制（ascend-constraints §1）不涉及——每行切片为最内维 D=128 连续 | unsorted:42-50、sparse_mla:328-335 |
| R3 | **小 GEMM 形状约束** | ✅ [64,128]×[128,16] 等 3 个 GEMM 的 M/N/K 均 16 倍数（H=16 作 N 维/K 维满足 16 列分形粒度，ascend.py:456-459）。`T.gemm_v0` 支持 bf16 操作数 + fp32 L0C 累加 + transpose_A/B + init 累加语义（ascend.py:418-463；操作数须 L1、C 须 L0C——Developer 模式 alloc_shared/fragment 由编译器映射，gqa_sink:329-343 先例）。L0C 峰值 44KB < 128KB。GPU `T.gemm(clear_accum=)` → `T.gemm_v0(init=)` 一一对应。**注意 `T.gemm`（npu_gemm，customize.py:115）是另一条 PTO 路径，不使用** | ascend.py:418-519、gqa_sink:400/434/469 |
| R4 | **动态 shape** | ✅ `T.symbolic("seq_len")` + `T.symbolic("batch_plus_one")`（GPU 的 `T.dynamic("a, b")` 逗号语法在 Ascend 前端不存在——grep `def dynamic` 无结果）。动态 grid `T.Kernel(seq_len, ...)` 先例：lightning_indexer:33（`T.Kernel(B * N2)`，B 为 symbolic）、sparse_mla:200 | sparse_mla:104-106、lightning_indexer:18-33 |
| R5 | **T.Kernel 签名** | ✅ `T.Kernel(grid, threads=1, is_npu=True) as (cid)`——`is_npu=True` 必需；threads 仅支持 1/2（GPU threads=128 移除）；`(cid, vid)` 双 vid 形态仅 threads=2 时使用。先例：gqa_sink:317（threads=1）、unsorted:36（`(cid, vid)`） | gqa_sink:317、ascend-constraints §1 |
| R6 | **bf16 精度路径** | ✅ 完整复刻主仓精度链：bf16 输入 → fp32 累加（gemm_v0 L0C fp32）→ bf16 输出；d_logits 两次 fp32→bf16 cast（对应主仓 L134/145 的 cast1/cast2，分别作 dq/dk GEMM 操作数）；dk 以 bf16 部分和逐块 atomic（对应主仓 L159 fp32 frag → bf16 GM 的隐式转换路径）；唯一增强：sm_scale 乘法在 fp32 UB 完成（主仓在 bf16 shared 乘，本设计精度 ≥ 主仓，golden 为 fp32 autograd 无损参考）。混合容差按 §9.3 bf16 行 | ascend.py:434/439、gqa_sink:383-384 |

### 10.2 性能风险（R7，最重大风险）

per-token 结构的 **MTE3 atomic 吞吐**（131072 条 256B）与 **gemm_v0 固定开销**（6144 次）是 80us 目标的临界瓶颈（§9.4 预算分解：重叠下界 118-197us，依赖 L2 命中与流水重叠才可 < 80us）。缓解：§9.4 的 4 条 Stage 3 调优路径（其中共享索引快路径可使 gemm 数 ÷T_BLOCK、atomic 不变；连续 pos 整块 atomic 可使 MTE3 条数 ÷32）。**若 L0 精度通过但性能不达标，按 tilelang-perf-optimization 流程走 Stage 3，不回退设计**（设计语义正确性不受性能调优影响）。

### 10.3 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| atomic 到未清零 buffer | 输出用 T.empty/torch.empty 分配 | dk 结果含垃圾初值 | host torch.zeros 三输出（§4.2） |
| atomic src/dst dtype 不匹配 | fp32 L0C 直接 atomic 到 bf16 GM | ascend_tile.py L215 ValueError | 先 `T.copy(dk_l0c, dk_ub)` auto-cast bf16 再 atomic |
| gemm 操作数误放 UB | alloc_ub 后直接 gemm_v0 | 运行时错误 | 操作数走 alloc_shared（编译器映射 L1） |
| heads=8 输入 | H%16≠0 | Cube N/K 低于分形粒度 | **不支持域显式声明**（§5.2 assert），L2 用例覆盖拒绝路径 |
| offsets 段长 < topk | 非法 varlen 配置 | golden assert 失败 | kernel 入口同步 assert（主仓 L195 语义） |
| 动态 if 内 BufferRegion 越界 | pos 未 clamp 即 gather | 越界读 | 有效 pos 判定（pos>-1 & pos≤i_t）先于 copy；无效行不访问 GM |

### 10.4 支持域与分派审计

```text
[DISPATCH-COVERAGE]
supported_domain: dtype 组合固定（Q/W/K bf16 + Scores fp32 + 索引 int32，无 dtype 扫描）；
  seq_len/batch 任意（varlen，段长 ≥ topk）；heads ∈ {16,32,48,64}（%16==0，≤64）；
  dim % 16 == 0；topk % block_I == 0 且 topk ≤ block_I×NI_max；sm_scale 任意 float
generic_fallback: none（单通用路径覆盖全部 supported_domain）
specializations: 无（性能特化路径为 Stage 3 可选扩展，attr 声明式分派、miss 自动回落本路径）
unsupported_inputs: heads=8（%16≠0，Cube 分形粒度物理约束——与主仓 %8 的差异显式声明）；
  段长 < topk 的 varlen（语义非法，主仓同款 assert）；fp16/fp32 输入 dtype（精度路径固定 bf16）
result: pass（单路径全覆盖，无枚举分支）
```

### 10.5 Host 侧审计

```text
[HOST-METADATA-AUDIT]
operation: prepare_token_indices(offsets) → TokenIndices
input_stride -> output_stride: 新建张量（torch.stack/cat 在 CPU int32 上构造）→ .npu() 一次 H2D
shares_storage / same_data_ptr: false（新建输入，属数据准备非输入改写）
aclnn_or_physical_copy: false（CPU 构造 + 单次 H2D；sparse_mla:1278-1296 同款先例）
result: allow

operation: torch.zeros(dIndexQ/dWeights/dIndexK) → kernel 显式参数
input_stride -> output_stride: 新建输出 buffer（零初始化，atomic 语义要求）
shares_storage / same_data_ptr: false（新建输出）
aclnn_or_physical_copy: false（分配 + 清零；unsorted:247 / gqa_sink:531-535 先例）
result: allow

operation: kernel 调用后直接返回 (dq, dw, dk)
aclnn_or_physical_copy: false（无 .to()/.contiguous()/切片 reshape——输出 shape 与原始一致）
result: allow
```

---

## 11. 交付清单

### 11.1 目录结构

```
examples_experiment/dsa_sparse_finetune/
├── DESIGN.md                     # 本设计文档（indexer_bwd）
├── proto_indexer_bwd.yaml        # indexer_bwd 接口规格（⚠ 目录已有 sparse_mla 的 proto.yaml，
│                                 #   为不覆盖兄弟算子工件而独立命名；coverage_check --proto 指向本文件）
├── indexer_bwd.py                # 纯 kernel（@tilelang.jit，可 import，无 golden/测试）【待实现】
├── test_indexer_bwd.py           # golden + L0 用例 + main（from indexer_bwd import）【待实现】
└── ref_gpu/indexer_bwd.py        # 主仓参考（只读）
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| DESIGN.md（本文件） | 已完成 | 设计文档（含 L0 门槛测试计划 §9.2 + 性能目标 §9.4） |
| proto_indexer_bwd.yaml | 已完成 | 接口规格（dtype 全集 = §9.3 表；attrs = §4/§1 派生） |
| indexer_bwd.py | 待实现 | 纯 kernel（@tilelang.jit + indexer_bwd_interface host 包装） |
| test_indexer_bwd.py | 待实现 | golden（§9.1 复刻）+ L0 4 用例（§9.2）+ 性能基准（§9.4）+ main（--level 分发） |

### 11.3 命名规范

- kernel 文件：`indexer_bwd.py`；测试文件：`test_indexer_bwd.py`（顶部 `from indexer_bwd import indexer_bwd_interface`）
- kernel 函数名：`tl_indexer_bwd_impl`（对齐主仓命名）；host 入口：`indexer_bwd_interface(q, weights, k, attn_score, index_score, topk_indices, offsets)`

### 11.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ proto_indexer_bwd.yaml + L0 门槛测试计划（§9.2）
2. ⬜ kernel 实现（indexer_bwd.py，纯 @tilelang.jit + Developer 4 True pass_configs）
3. ⬜ 测试文件（test_indexer_bwd.py）：golden + L0 4 用例 + 性能基准 + main
4. ⬜ L0 门槛测试通过（精度收敛：3 输出 × 4 用例全过混合容差）
5. ⬜ 性能验收（纯 kernel 平均 < 80us @ 主仓默认配置；不达标进 Stage 3）
6. ⬜ 扩展分层套件（L1/L2/Boundary，tilelang-op-test-design 场景 B）
7. ⬜ 全量套件 + 覆盖门禁（coverage_check.py --proto proto_indexer_bwd.yaml）
