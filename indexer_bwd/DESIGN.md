# indexer_bwd 算子设计文档（rev2）

> 迁移来源：`examples_experiment/dsa_sparse_finetune/ref_gpu/indexer_bwd.py`（GPU 版主仓参考，只读，语义权威）
> 目标平台：Ascend A2/A3（910B3，20 Cube Core，L0C 128KB / UB 192KB / L1 512KB）
> 编程模式：**Developer**（用户指定，rev1 → rev2 不变）
> **rev2（设计回退修订，revision_index=1）**：修复 rev1 两大设计错误——① dk 精度路径与 0.99 容差门限互斥（bf16 GM atomic 累加实测 ratio 0.9405-0.9611 全 FAIL）；② 性能预算未计入平台强制偏差导致的 GM workspace 流量（实测 1522.6us vs 目标 80us，19 倍差距）。修复 = **双路径架构（fast/generic + host 精确分派）+ dk 全链 fp32 累加 + 实测锚点校准的性能预算**。修订详情见 §12。

## 1. 概述

### 1.1 算子名称

indexer_bwd（DSA sparse finetune lightning indexer 反向，varlen KL 散度损失反传）

### 1.2 功能描述

DSA（DeepSeek Sparse Attention）稀疏微调中 lightning indexer 的反向算子：给定前向保存的 topk 选中位置 `TopkIndices`、注意力分数 `AttnScore`（teacher）与 indexer 输出分布 `IndexScore`（student，经 log_softmax），在变长（varlen）序列上计算 3 组梯度：`dIndexQ`、`dWeights`、`dIndexK`，其中 `dIndexK` 通过 atomic scatter-add 跨 token 累加。

**rev2 双路径架构**（§1.4）：语义对所有输入正确（generic 全覆盖）；性能目标（< 80us）只考核主仓默认配置（topk_indices = arange(topk) 全 token 平铺），该配置命中 fast 路径 layout=0。

### 1.3 数学公式

前向语义（每 token t，sm_scale = D**-0.5）：

$$\text{logits}[t,h,s] = \mathrm{relu}(q[t,h,:] \cdot k[s,:] \times \text{sm\_scale})$$

反向（遍历 topk 索引 pos，causal 有效域为 `-1 < pos <= t`；`1{...}` 为指示函数）：

$$d\_logits[t,h,pos] = (\text{IndexScore}[t,pos] - \text{AttnScore}[t,pos]) \times 1\{\text{logits}>0\} \times \text{Weights}[t,h]$$

$$d\text{Weights}[t,h] \mathrel{+}= \sum_{pos} (\text{IndexScore} - \text{AttnScore})[t,pos] \times \text{logits}[t,h,pos]$$

$$d\text{IndexQ}[t,h,:] \mathrel{+}= \sum_{pos} d\_logits[t,h,pos] \times \text{IndexK}[pos,:] \times \text{sm\_scale}$$

$$d\text{IndexK}[pos,:] \mathrel{+}= \sum_{h} d\_logits[t,h,pos] \times \text{IndexQ}[t,h,:] \times \text{sm\_scale} \quad (\text{atomic scatter-add})$$

**sm_scale 折叠（rev2 沿用 rev1 已验证等价形式并进一步简化）**：因 sm_scale > 0 与 relu 可交换，`relu(qk×s) = s×relu(qk)`、`1{qk×s>0} = 1{qk>0}`。rev2 将 sm_scale **一次性折叠进 diff**：`diffm = (IndexScore - AttnScore) × sm_scale`（[T_BLOCK, topk] fp32，一次 `T.tile.mul`）。此后 logits GEMM 用未缩放 Q（L0C 输出 relu(qk)）、d_weights = Σ diffm×logits、d_logits = diffm ⊗ 1{logits>0} ⊗ Weights、dq/dk GEMM 直接用 d_logits——三条梯度经 diffm 统一获得 sm_scale，数学等价（rev1 实现偏差 #3/#4 已实测验证同款等价，且比主仓少 2 个 bf16 舍入点，精度 ≥ 主仓路径；golden 为 fp32 autograd 无损参考）。

### 1.4 算法描述与双路径架构（rev2 核心变更）

计算类型：**Cube+Vector 混合（CV 融合）**。复杂度：Fusion。per-token 数学为 3 个小 GEMM（logits / dIndexQ / dIndexK）+ 向量后处理（relu、差值、causal mask、broadcast 乘、reduce）+ gather / atomic scatter。

**Host 侧分派架构**（参照姊妹算子 `examples/dsa_sparse_finetune/sparse_mla_topk_reducesum.py` 的 panel_mode host 分发机制，`DESIGN_panel_mode.md` §8）：

```text
indexer_bwd_interface(q, weights, k, attn_score, index_score, topk_indices, offsets,
                      dispatch="auto")
  │
  ├─ 数据准备（host，数据准备类操作，ascend-constraints §5）：
  │    token_indices [ΣS, 2]（generic 用，prepare_token_indices，rev1 已验证）
  │    block_indices [num_blocks, 3]（fast 用：(bos, t0_local, seq_len)，per 段 ceildiv(len, T_BLOCK) 行）
  │    输出分配：dIndexQ/dWeights/dIndexK torch.zeros + ws_dk [ΣS, D] fp32 torch.zeros（dk fp32 累加工作区）
  │
  ├─ 共享性检测（host，设备侧比较 + 一次 D2H bool 读回；panel host 门控 G1-G5 同类先例）：
  │    is_arange = (topk_indices == arange(topk, int32)).all()      # 全 token 平铺 arange → layout=0
  │    is_shared = (topk_indices == topk_indices[:1]).all()         # 全行相等（非 arange）→ layout=1
  │    二者皆非 → generic
  │
  ├─ fast kernel（tl_indexer_bwd_fast，persistent grid=20，layout 为编译期常量参数 → 2 个编译缓存键）
  │    layout=0（arange 平铺，主仓默认/perf 配置）：K 整块 GM→L1 直拷 + causal 纯位置向量化 mask
  │             + dk 整块 region atomic；中转流量与 gemm 数均最小
  │    layout=1（任意共享索引）：逐行 gather（有效性分支）+ indices compare 向量化 mask + 逐行 atomic
  │
  ├─ generic kernel（tl_indexer_bwd_generic，grid=seq_len，rev1 已验证结构 + dk fp32 化）
  │
  └─ dk cast kernel（tl_dk_cast，persistent grid=20，两路径共用）：ws_dk fp32 → dIndexK bf16
```

**分派正确性论证**：检测为**精确布尔判定**（非启发式）——`is_arange`/`is_shared` 恒真恒假，无误判路径；fast kernel 在各自 layout 契约下对**任意**该 layout 输入正确（layout=0 契约 = 全 token 索引恰为 arange(0..topk) 平铺；layout=1 契约 = 全部行严格相等且值域 [0, 段长)）；契约外输入全部由 generic 覆盖（rev1 结构，L0 已验证 dq ratio=1.0000 / dw 全过，dk 修复后全过）。语义对所有输入正确；性能只考核 layout=0 命中的主仓默认配置。`dispatch` 参数四值（auto/fast_arange/fast_shared/generic）供测试注入与调用方显式声明（panel_mode 先例）。

### 1.5 数据流图（fast 路径 layout=0，每 block = T_BLOCK=32 个同段连续 token）

```
GM[IndexQ]   --T.copy 连续行段 [TS*H, D]（GM→L1 直连，per sub-tile）--> L1(Q_l1)
GM[IndexK]   --T.copy 整块 [topk, D]（GM→L1 直连，per block 一次）--> L1(K_l1)
GM[AttnScore/IndexScore] --T.copy--> UB(attn_ub/is_ub [T_BLOCK, topk] fp32)
GM[TopkIndices]（仅 layout=1 读）--T.copy--> UB(indices_ub [topk] int32)

UB: diffm = (is - attn) × sm_scale                       # [T_BLOCK, topk] fp32，sm_scale 唯一折叠点

per sub-tile（TSB = T_BLOCK/TS = 8 次）:
  L1(Q_l1) × L1(K_l1)^T --T.gemm_v0(transpose_B, init)--> L0C(logits [TS*H, topk] fp32)
  L0C(logits) --T.copy(enable_relu)--> ws_logits(GM 中转, 显式参数) --T.copy--> UB(logits_ub [TS*H, topk])
  UB per-t 向量段（T.serial(TS)，[H, topk] 小 tile，gqa_sink 先例 API）:
      causal = compare(col_pos, t0+t, "LE") & valid_t              # arith_progression + fill + bitwise_and
      dw_t = reduce_sum(diffm_t ⊗ select(logits_t, causal, 0.0), dim=1)   # logits 用 relu 后值 ✓
      dlog_t = select(diffm_t ⊗ w_t, causal & 1{logits_t>0}, 0.0)         # d_relu = 1{logits>0} ✓
  UB(dlog_ub [TS*H, topk] fp32) --T.copy auto-cast--> ws_dlog(GM 中转) --T.copy--> L1(dlog_a_l1/dlog_b_l1, bf16 ×2)
  L1(dlog_a_l1) × L1(K_l1)     --T.gemm_v0(init=True)------------------> L0C(dq [TS*H, D] fp32)
  L1(dlog_b_l1)^T × L1(Q_l1)   --T.gemm_v0(transpose_A, init=(ts==0))--> L0C(dk [topk, D] fp32，跨 sub-tile 累加)
  L0C(dq) --T.copy(auto-cast)--> GM[dIndexQ 行段]          # L0C→GM 直写，不经 UB（gqa_sink:478 先例）
  UB(dw_acc [T_BLOCK, H]) --T.copy(auto-cast)--> GM[dWeights 行段]（block 尾部一次）
  L0C(dk) --T.tile.atomic_add(ws_dk[bos:bos+topk, :], dk_l0c)-->   # 整块 fp32 region atomic，L0C 直 src（gqa_sink:437）

cast kernel: GM[ws_dk fp32] --T.copy auto-cast--> GM[dIndexK bf16]（主 kernel 全部完成后）
```

fast 路径 layout=1 差异（仅 3 处）：① K gather 改 `T.tile.fill(0)` + `T.serial(topk)` 逐行动态 `if (pos>-1)&(pos<=t_max)&(pos<len): T.copy(IndexK[bos+pos,:], gather_ub[i,:])`，UB→L1 经 GM 中转（bf16 UB→L1 直连 static_assert half-only，README 偏差 #1）；② causal mask 改 `compare(broadcast(indices_1d), fill(t0+t), "LE")`（索引值 vs token 位置，gqa_sink:416-419 同款结构）；③ dk 改 `T.copy(dk_l0c, dk_ub fp32)` + `T.serial(topk)` 逐行 `if (pos>-1)&(pos<len): T.tile.atomic_add(ws_dk[bos+pos, 0], dk_ub[i,:])`（512B fp32/行）。

generic 路径数据流 = rev1 §1.5 原样（已验证结构）+ 两处 dk 修改：dk_l0c → UB **fp32**（不 cast bf16）→ 逐行 `T.tile.atomic_add(ws_dk[bos+pos, 0], dk_fp32_ub[i,:])`；dIndexK 输出改由 cast kernel 统一产出。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer（用户明确指定，rev1 → rev2 不变，禁止更改）

### 2.2 选型理由

fast/generic/cast 三个 kernel 均为 CV 融合（GEMM + 向量后处理 + DMA atomic），与 `examples/attention_sink/example_gqa_sink_bwd_bhsd/example_gqa_sink_bwd_bhsd.py` Kernel 2（Developer 模式单 kernel bwd：KV-loop 5 GEMM + atomic dK/dV）高度同构。该先例已验证 Developer 模式下 `T.alloc_shared/alloc_fragment` + `T.gemm_v0` + `T.tile.atomic_add`（含 **L0C 直 src 与整块 region**）+ `T.copy`（**GM→L1 直连 / L0C→GM 直写 auto-cast**）+ 4 个自动 pass_configs 覆盖全部所需能力。persistent grid（`T.Kernel(20)` 编译期常量）有 `examples/dsa_sparse_finetune/sparse_mla_topk_reducesum.py:1302`（generic/panel 两变体）先例，同为 Developer 模式。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（编译器映射 L1/UB）+ `T.alloc_fragment`（L0C）；不使用显式 `alloc_L1/ub/L0A/L0B/L0C` |
| 计算方式 | Cube：`T.gemm_v0`；Vector：`T.tile.sub/mul/fill/broadcast/compare/select/bitwise_and/arith_progression`、`T.reduce_sum`；MTE：`T.copy` / `T.tile.atomic_add` |
| 作用域 | 无 `T.Scope`，编译器自动分离 C/V 指令（AUTO_CV_COMBINE） |
| 同步方式 | 全自动（AUTO_SYNC + AUTO_CV_SYNC），零手动 flag；CV 中转 workspace 靠 buffer 名含 "workspace" 由 pass 自动配对插 flag（rev1 已验证机制，§7.2） |
| threads | `threads=1`（fast/generic/cast 三 kernel 一致；gqa_sink_bwd:317 先例） |

---

## 3. API 映射设计

### 3.1 公式拆解

**fast 路径（layout=0，M = TS*H = 64，三个 GEMM 升维合并，每 sub-tile 一组）**：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 0 | `diffm = (IndexScore - AttnScore) × sm_scale` | [T_BLOCK, topk] fp32，整 block 一次（sm_scale 唯一折叠点，§1.3） |
| 1 | `K_panel = IndexK[bos : bos+topk, :]` | 整块 GM→L1 直拷（arange 契约下 64 行连续），per block 一次 |
| 2 | `logits = relu(Q_flat × K_panel^T)` | GEMM1：[TS*H, D]×[D, topk] → [TS*H, topk] fp32（Q_flat = Q 子块展平 [TS*H, D]） |
| 3 | `causal[t, j] = (j <= t0+t) & (t0+t < len)` | [H, topk]/行，arith_progression + fill + compare + bitwise_and 向量化（gqa_sink:416-422 同款） |
| 4 | `dWeights[t] = Σ_j diffm[t,j] × masked_logits` | masked_logits = select(logits, causal, 0.0)；reduce_sum(dim=1) |
| 5 | `d_logits = diffm ⊗ 1{logits>0} ⊗ Weights ⊗ causal` | select + compare + bitwise_and（d_relu = 1{logits>0}，与 causal 合并门控） |
| 6 | `dq = d_logits × K_panel` | GEMM2：[TS*H, topk]×[topk, D] → [TS*H, D] fp32，L0C→GM 直写 auto-cast |
| 7 | `dk_panel = d_logits^T × Q_flat` | GEMM3：[topk, TS*H]×[TS*H, D] → [topk, D] fp32，**跨 sub-tile init=False 片上累加** |
| 8 | `ws_dk[bos+j, :] += dk_panel[j, :]` | **整块 [topk, D] fp32 region atomic**（per block 一次；跨 block 累加次数 S/T_BLOCK） |

**generic 路径**：rev1 §3.1 十步表原样继承（block_I=64 逐 token 三 GEMM + 向量后处理 + 逐行 gather/atomic），仅步骤 8/9 的 dk 出口改 fp32：`dk_l0c → UB(fp32) → T.tile.atomic_add(ws_dk[bos+pos, 0], dk_fp32_ub[i,:])`，尾部统一 cast kernel。

### 3.2 TileLang API 映射（GPU → Ascend；rev1 已核实证据链直接继承，rev2 新增行标注【新】）

| 步骤 | GPU 版 API（ref_gpu 行号） | Ascend API | 证据 |
|------|---------------------------|-----------|------|
| Kernel 声明（generic） | `T.Kernel(seq_len, threads=128)`（L66） | `T.Kernel(seq_len, threads=1, is_npu=True) as (cid)` | gqa_sink_bwd:317；rev1 已验证 |
| 【新】Kernel 声明（fast/cast persistent） | — | `T.Kernel(grid_size, threads=1, is_npu=True) as (cid)`，grid_size=20 编译期常量；任务轮询 `ntask = T.max(0, T.ceildiv(total - cid, grid_size))`、`pid = k*grid_size + cid` | sparse_mla:262/309/1302（generic/panel 双变体 persistent 先例） |
| 动态 shape | `T.dynamic("seq_len, batch_plus_one")`（L41） | `T.symbolic("seq_len")` + `T.symbolic("batch_plus_one")` + `T.symbolic("num_blocks")`（fast 增第三动态维） | sparse_mla:104-106/142-144；rev1 已验证 |
| 输出声明 | `T.empty(...)`（L57-59） | kernel 参数全显式传入（host `torch.zeros` 预分配） | unsorted:247、gqa_sink:531-535；rev1 已验证 |
| 共享内存 / fragment | `T.alloc_shared` / `T.alloc_fragment`（L71-75） | 同名 API（Developer 映射 L1/UB 与 L0C） | gqa_sink:329-362；rev1 已验证 |
| 【新】Q/K 整块装载 | `T.copy(IndexQ[bos+i_t,:,:], …)`（L77，逐 token） | `T.copy(IndexQ[bos+t0+ts*TS, 0:TS*H*D], Q_l1)`（GM→L1 直连，[TS*H, D] 连续行段）；`T.copy(IndexK[bos:bos+topk,:], K_l1)` | GM→L1 直拷：gqa_sink:390-391（q_l1/do_l1）；fast kernel 的 IndexQ/dIndexQ 参数按 [seq_len, heads*dim] 二维声明实现连续行段切片（同一连续内存的零拷贝布局重解释；fallback：host 侧 `q.view(S, H*D)` 零拷贝元数据视图传入） |
| 【新】等差位置向量 | `torch.arange`（mask 构造） | `T.tile.arith_progression(col_1d, 0, 1, topk)`（起点/步长可为运行时标量） | gqa_sink:393/416 |
| 【新】causal mask 向量化 | `mask = arange(s)[:,None] >= arange(s)[None,:]`（golden 域） | `T.tile.broadcast(col_2d, col_1d, axis=0)` + `T.tile.fill(tval_2d, t0+t)` + `T.tile.compare(causal, col_2d, tval_2d, "LE")` + `T.tile.bitwise_and(causal, causal, valid)` | gqa_sink:417-422（broadcast/compare/bitwise_and 全先例）；layout=1 改 compare(broadcast(indices_1d), tval, "LE") |
| GEMM | `T.gemm(A,B,C,transpose_A/B,clear_accum)`（L104/135/147） | `T.gemm_v0(A,B,C,transpose_A/B,init,kL0Size=128)`（A/B 在 L1，C 在 L0C；init=False 累加 ↔ clear_accum=False；bf16 操作数 + fp32 C） | ascend.py:418-463（bf16 L434、fp32 累加 L439）；gqa_sink:400/434/469；rev1 已验证 |
| relu | `T.max(logits, 0)`（L113） | `T.copy(logits_l0c, logits_ub, enable_relu=True)`（随中转 copy 完成） | copy_op.py:260；lightning_indexer:54-60；rev1 已验证 |
| d_relu 指示 | `if logits>0: 1.0`（L127-130） | `T.tile.compare(drelu, logits_t, zero_2d, "GT")` + `T.tile.bitwise_and(causal, causal, drelu)` | gqa_sink:419-422；rev1 已验证（select 路径保留于 generic） |
| d_logits 构造 | `T.Parallel` + `T.alloc_var`（L125-131） | `T.tile.sub/mul` + `T.tile.broadcast(axis=0/1)` + `T.tile.compare` + `T.tile.select(..., 0.0, "VSEL_TENSOR_SCALAR_MODE")` | gqa_sink:404-423；rev1 已验证 |
| 【新】dw 列归约 | `T.reduce_sum(d_w_i, d_w_frag, dim=0, clear=False)`（L119，行归约跨块累加） | fast：`T.reduce_sum(dw_2d, dw_t, dim=1)`（[H, topk] → [H]，per-t 列归约）；generic：dim=0, clear=False 原样 | gqa_sink:386（dim=-1 列归约先例）；reduce_ascend.py:452-500 |
| d_logits cast（UB→L1） | `T.copy(dlog_ub, dlog_l1)`（L134/145） | `T.copy(dlog_ub fp32, ws_dlog GM) + T.copy(ws_dlog, dlog_l1 bf16)`（**两段显式经 GM 中转**；bf16 UB→L1 直连 static_assert half-only，README 偏差 #1） | rev1 实现已验证（workspace 命名 + AUTO_CV_SYNC 配对插 flag；common.h copy_ub_to_l1 static_assert） |
| 【新】L0C→UB（logits） | — | `T.copy(logits_l0c, ws_logits GM) + T.copy(ws_logits, logits_ub)`（L0C→UB 被编译器改写 GM 往返，显式参数化规避 auto workspace 动态 grid 限制） | rev1 实现已验证（ascend_workspace_reduction.cc 改写 + cython_wrapper.pyx 动态 shape 限制，README 偏差 #2） |
| 【新】L0C→GM 直写（dq） | `T.copy(d_index_q_frag, dIndexQ[...])`（L164，经 UB） | `T.copy(dq_l0c, dIndexQ[bos+t0+…, :])`（fp32→bf16 auto-cast 随程，不经 UB） | gqa_sink:478（"write dQ from L0C -> GM (fp32 -> fp16 auto-cast)"） |
| 【新】atomic（fp32 + L0C src + 整块） | `T.atomic_add(dIndexK[bos+pos,j], scalar)`（L159，标量） | fast layout=0：`T.tile.atomic_add(ws_dk[bos:bos+topk, :], dk_l0c)`（**fp32 整块 region，L0C 直 src**）；layout=1/generic：`T.tile.atomic_add(ws_dk[bos+pos, 0], dk_fp32_ub[i,:])`（逐行 512B fp32） | **fp32 支持：ascend_tile.py:324（"Supports float16, float32, int16, int32, bfloat16"）**；L0C src + 连续 region：gqa_sink:437/472；逐行：unsorted:50；rev1 已验证调用形态 |
| 【新】cast kernel | — | `T.copy(ws_dk[chunk, :], ub_fp32) + T.copy(ub_fp32, dIndexK[chunk, :])`（fp32→bf16 auto-cast） | auto-cast 随 copy 先例：rev1/gqa_sink 全链 |
| 流水线 | `T.Pipelined(num_blocks, num_stages=0)`（L85） | `T.serial(NI/TSB/…)`（num_stages=0 本无流水；边界含运行时值不用 Pipelined） | ascend-constraints §1；rev1 已验证 |
| 标量 GM 读 | `TokenIndices[bx,0]`、`Offsets[i_b]`（L67-68） | 同语法（fast：BlockIndices 3 次/block；generic：2 次/token） | sparse_mla:289-291；rev1 已验证 |
| 动态行 gather（layout=1/generic） | `T.if_then_else(cond, IndexK[…], 0)`（L95） | `T.tile.fill(0)` + 动态 `if (pos>-1)&(pos<=t_max): T.copy(IndexK[bos+pos,:], gather_ub[i,:])` | unsorted:42、sparse_mla:383；rev1 已验证 |

### 3.2.1 fast kernel 伪代码（layout=0；layout=1 差异见 §1.5 括注）

```python
GRID = 20                                  # 编译期常量（host 查 cube_core_num，sparse_mla:1501）
TS, T_BLOCK, TSB = 4, 32, 8                # sub-tile token 数 / block token 数 / sub-tile 迭代数
seq_len, batch_plus_one = T.symbolic("seq_len"), T.symbolic("batch_plus_one")
num_blocks = T.symbolic("num_blocks")      # BlockIndices 维 = Σ ceildiv(len_i, T_BLOCK)

@tl.jit(pass_configs=4_True_Developer)     # §4.7
def tl_indexer_bwd_fast(
    IndexQ,                                # [seq_len, heads*dim] bf16（布局重解释，§3.2 Q 装载行）
    Weights, AttnScore, IndexScore,        # [seq_len, heads] bf16 / [seq_len, topk] fp32 ×2
    IndexK, TopkIndices,                   # [seq_len, dim] bf16 / [seq_len, topk] int32
    BlockIndices,                          # [num_blocks, 3] int32 (bos, t0_local, seq_len)
    ws_logits, ws_dlog_a, ws_dlog_b,       # CV 中转 GM（显式参数，名含 workspace，§4.2）
    ws_dk,                                 # [seq_len, dim] fp32（host torch.zeros）
    dIndexQ, dWeights,                     # 输出（dIndexK 由 cast kernel 产出）
    heads, dim, topk, sm_scale, layout,    # layout: 编译期常量 0/1
):
    with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
        total = num_blocks
        ntask = T.max(0, T.ceildiv(total - cid, GRID))          # sparse_mla:309 同款
        # ---- L1 / L0C / UB 分配（§4.3；Developer 编译器映射） ----
        K_l1      = T.alloc_shared([topk, dim], bf16)           # L1
        Q_l1      = T.alloc_shared([TS*heads, dim], bf16)       # L1
        dlog_a_l1 = T.alloc_shared([TS*heads, topk], bf16)      # L1（GEMM2 A，不转置读）
        dlog_b_l1 = T.alloc_shared([TS*heads, topk], bf16)      # L1（GEMM3 A，transpose_A 读；双 buffer
                                                                #  规避同 buffer 两种 fractal 读法，rev1 §8.6）
        logits_l0c = T.alloc_fragment([TS*heads, topk], fp32)   # L0C 16KB
        dq_l0c     = T.alloc_fragment([TS*heads, dim], fp32)    # L0C 32KB
        dk_l0c     = T.alloc_fragment([topk, dim], fp32)        # L0C 32KB（跨 sub-tile 累加）
        logits_ub, dlog_ub = T.alloc_shared([TS*heads, topk], fp32) ×2   # UB
        attn_ub, is_ub, diffm_ub = T.alloc_shared([T_BLOCK, topk], fp32) ×3
        dw_acc_ub  = T.alloc_shared([T_BLOCK, heads], fp32)     # UB
        col_1d     = T.alloc_shared([topk], fp32)               # UB（j 位置向量，块外常驻）
        # per-t 小 tile buffer（T.serial 体内复用同槽）：logits_t/diff_2d/w_2d/col_2d/tval_2d/
        # causal/drelu/logits_m/dwq/dlog_t/zero_2d/valid_2d 等 [heads, topk] fp32 ×~12
        T.tile.arith_progression(col_1d, 0, 1, topk)

        for k in T.serial(ntask):
            bx = k * GRID + cid
            bos, t0, ln = BlockIndices[bx, 0], BlockIndices[bx, 1], BlockIndices[bx, 2]
            # ---- 装载（layout=0：整块直拷；layout=1：fill0 + 逐行 if copy → UB → GM 中转 → L1）----
            T.copy(IndexK[bos : bos+topk, :], K_l1)                     # GM→L1 直连
            if t0 + T_BLOCK <= ln:                                       # 尾块动态守卫（§5.3）
                T.copy(AttnScore[bos+t0 : bos+t0+T_BLOCK, :], attn_ub)
                T.copy(IndexScore[bos+t0 : bos+t0+T_BLOCK, :], is_ub)
            else:
                for u in T.serial(T_BLOCK):
                    if t0 + u < ln:
                        T.copy(AttnScore[bos+t0+u, 0:topk], attn_ub[u, 0:topk])
                        T.copy(IndexScore[bos+t0+u, 0:topk], is_ub[u, 0:topk])
            T.tile.sub(diffm_ub, is_ub, attn_ub)
            T.tile.mul(diffm_ub, diffm_ub, sm_scale)                    # sm_scale 唯一折叠点
            T.fill(dw_acc_ub, 0.0)
            for ts in T.serial(TSB):                                    # 8 个 sub-tile
                t_cur = t0 + ts * TS
                if t_cur + TS <= ln:                                    # 整块（常态路径）
                    T.copy(IndexQ[bos+t_cur, 0 : TS*heads*dim], Q_l1)   # 连续行段 GM→L1
                else:                                                   # varlen 尾块逐 token 守卫
                    for u in T.serial(TS):
                        if t_cur + u < ln:
                            T.copy(IndexQ[bos+t_cur+u, 0 : heads*dim],
                                   Q_l1[u*heads : (u+1)*heads, 0:dim])
                # ---- GEMM1: logits = relu(Q_flat × K^T)（Q 未缩放，sm_scale 在 diffm）----
                T.gemm_v0(Q_l1, K_l1, logits_l0c, transpose_B=True, init=True)
                T.copy(logits_l0c, ws_logits[cid, ts % 2, :, :])        # L0C→GM 中转段 1（双槽轮转）
                T.copy(ws_logits[cid, ts % 2, :, :], logits_ub, enable_relu=True)  # 段 2 + relu
                # ---- Vector per-t：causal mask / dw / dlog ----
                for u in T.serial(TS):
                    t_abs = t_cur + u                                   # 段内绝对位置
                    T.copy(logits_ub[u*heads : (u+1)*heads, :], logits_t)
                    T.copy(diffm_ub[u, 0:topk], diff_t)
                    T.tile.broadcast(diff_2d, diff_t, axis=0)           # [H, topk]
                    if t_abs < ln:
                        T.copy(Weights[bos+t_abs, 0:heads], w_t)        # [H]
                    T.tile.broadcast(w_2d, w_t, axis=1)                 # [H, topk]
                    T.tile.broadcast(col_2d, col_1d, axis=0)            # j 向量
                    T.tile.fill(tval_2d, t_abs)                         # 标量 fill
                    T.tile.compare(causal, col_2d, tval_2d, "LE")       # j <= t
                    T.tile.fill(valid_2d, 1.0)                          # t_abs < ln 时（else 0.0）
                    T.tile.bitwise_and(causal, causal, valid_2d)
                    T.tile.select(logits_m, causal, logits_t, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                    T.tile.mul(dwq_2d, diff_2d, logits_m)               # dw 被积式（relu 后 logits ✓）
                    T.reduce_sum(dwq_2d, dw_t, dim=1)                   # [H]
                    T.copy(dw_t, dw_acc_ub[u, 0:heads])
                    T.tile.compare(drelu, logits_t, zero_2d, "GT")      # d_relu = 1{logits>0}
                    T.tile.bitwise_and(causal, causal, drelu)
                    T.tile.mul(dlog_t, diff_2d, w_2d)
                    T.tile.select(dlog_t, causal, dlog_t, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                    T.copy(dlog_t, dlog_ub[u*heads : (u+1)*heads, :])   # 累积进 sub-tile 大 tile
                # ---- dlog UB→L1（bf16，经 GM 中转两段；双 buffer）----
                T.copy(dlog_ub, ws_dlog_a[cid, ts % 2, :, :])           # auto-cast fp32→bf16
                T.copy(ws_dlog_a[cid, ts % 2, :, :], dlog_a_l1)
                T.copy(dlog_ub, ws_dlog_b[cid, ts % 2, :, :])
                T.copy(ws_dlog_b[cid, ts % 2, :, :], dlog_b_l1)
                # ---- GEMM2: dq = dlog × K（值已含 sm_scale；L0C→GM 直写）----
                T.gemm_v0(dlog_a_l1, K_l1, dq_l0c, init=True)
                if t_cur + TS <= ln:
                    T.copy(dq_l0c, dIndexQ[bos+t_cur, 0 : TS*heads*dim])   # L0C→GM auto-cast
                else:
                    for u in T.serial(TS):
                        if t_cur + u < ln:
                            T.copy(dq_l0c[u*heads:(u+1)*heads, 0:dim],
                                   dIndexQ[bos+t_cur+u, 0:heads*dim])
                # ---- GEMM3: dk += dlog^T × Q_flat（跨 sub-tile 片上 fp32 累加）----
                T.gemm_v0(dlog_b_l1, Q_l1, dk_l0c, transpose_A=True, init=(ts == 0))
            # ---- block 尾部：dw 写出 + dk fp32 atomic ----
            if t0 + T_BLOCK <= ln:
                T.copy(dw_acc_ub, dWeights[bos+t0 : bos+t0+T_BLOCK, 0:heads])
            else:
                for u in T.serial(T_BLOCK):
                    if t0 + u < ln:
                        T.copy(dw_acc_ub[u, 0:heads], dWeights[bos+t0+u, 0:heads])
            T.tile.atomic_add(ws_dk[bos : bos+topk, :], dk_l0c)     # layout=0：整块 region（gqa_sink:437）
            #   layout=1：T.copy(dk_l0c, dk_ub_fp32) + T.serial(topk) 逐行
            #             if (pos > -1) and (pos < ln): T.tile.atomic_add(ws_dk[bos+pos, 0], dk_ub_fp32[i, :])
```

### 3.3 generic kernel 伪代码

rev1 §3.3 伪代码原样继承（`indexer_bwd.py` 现实现即其落地，dq ratio=1.0000 / dw 全过已实测），仅两处修改：

```python
# 修改 1：dk 中间缓冲 fp32 化（原 bf16）
dk_ub = T.alloc_shared([block_I, D], FP32)             # atomic src 与 ws_dk fp32 一致
# 修改 2：atomic 目标改 fp32 工作区（原 dIndexK bf16）
for i in T.serial(block_I):
    pos = indices_ub[i]
    if (pos > -1) and (pos <= i_t):
        T.tile.atomic_add(ws_dk[bos + pos, 0], dk_ub[i, :])   # 512B fp32/行（ascend_tile.py:324）
# 尾部：dq/dw 写回不变；dIndexK 由 cast kernel 统一产出（host 顺序：generic kernel → cast kernel）
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| `T.gemm_v0`（bf16 操作数 / fp32 累加 / transpose_A/B / init / kL0Size） | tilelang/language/ascend.py:418-463 | 本仓示例已验证（gqa_sink:400/434/469/474-475） |
| `T.tile.atomic_add`（**fp32** dst GM / src local（UB 或 **L0C 直**）/ dtype 一致 / 整块连续 region） | tilelang/language/ascend_tile.py:313-349（dtype 列表 L324） | gqa_sink:437/472（L0C src + region）、unsorted:50（逐行）；rev1（逐行 bf16 形态实测） |
| `T.copy`（auto-cast / enable_relu / GM→L1 直连 / L0C→GM 直写 / 动态 if 内 / UB region 行块） | copy_op.py:260；gqa_sink:390/391/478 | rev1 实测 + gqa_sink 全先例 |
| `T.tile.arith_progression / broadcast / compare / select / bitwise_and / fill / sub / mul` | ascend_tile.py（fill L352）；gqa_sink:393/404-423 | gqa_sink 全先例（rev1 §3.4 已核索引） |
| `T.reduce_sum(buffer, out, dim, clear)`（dim=0 行归约 / dim=1 列归约） | reduce_ascend.py:452-500 | gqa_sink:386（dim=-1）；rev1（dim=0, clear=False 实测） |
| `T.Kernel` 常量 grid（persistent）+ `T.ceildiv` + `T.max` | sparse_mla:262/309/1302 | sparse_mla generic/panel 双变体实测（graph-replay 143.3us） |
| `T.symbolic`（含 num_blocks 第三动态维） | tilelang/language/__init__.py:89；sparse_mla:104-106/142-144 | rev1 动态 grid 多形状复用同一编译产物已实测 |
| GM 标量读（`BlockIndices[bx, 0..2]`） | TIR BufferLoad 标量 | sparse_mla:289-291；rev1 TokenIndices 实测 |
| UB→UB copy / UB region 行块读写 | gqa_sink:426/428/463；rev1 gather_ub[i,:] | rev1 + gqa_sink 先例 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（rev2 更新）

| 约束 | 是否涉及 | 处理方案 |
|------|---------|---------|
| 不支持三维 Kernel | No | fast/generic/cast 均一维 grid（常量 20 / seq_len / 常量 20） |
| threads 参数限制（仅 1/2） | Yes（GPU 128） | `threads=1`（三 kernel 一致） |
| 动态循环边界 | **Yes（fast）** | persistent 外层 `T.serial(ntask)` 运行时边界——**sparse_mla:309 已验证形态**（ntask = T.max(0, T.ceildiv(...))，generic/panel 双变体实测）；sub-tile/per-t/topk 循环边界全为编译期常量（TSB=8/TS=4/topk）。generic 的 `if pos <= i_t` 为分支非循环边界（rev1 已验证） |
| 流水线不支持动态边界 | No | 不使用 T.Pipelined（num_stages=0 本无流水） |
| L0C 容量上限 128KB | **Yes（fast）** | 已验算：logits [64,64] + dq [64,128] + dk [64,128] fp32 = 16+32+32 = **80KB < 128KB**（余量 48KB，TS 选取论证 §5.2；无 liveness 复用假设的保守口径，与 rev1/panel G3 门一致）；generic 44KB（rev1 已验算） |
| **UB→L1 bf16 直连 static_assert（half-only）** | **Yes（fast dlog / layout=1 gather）** | dlog 经 `ws_dlog_*` GM 两段中转（rev1 实现已验证机制：显式参数 + buffer 名含 workspace + AUTO_CV_SYNC 自动配对 flag）；layout=1 K gather 同法中转（~2MB/调用，非考核路径） |
| **L0C→UB 被编译器改写 GM 往返** | **Yes（fast logits relu）** | `ws_logits` 显式参数两段中转（rev1 实现已验证；流量 16.6MB/调用已入 §9.4 预算） |
| auto workspace 不支持动态 grid | Yes | 全部中转 buffer 显式参数化（rev1 方案原样；fast grid 虽为常量 20 但 seq_len/num_blocks 仍 symbolic，维持显式参数规避） |
| T.copy 列方向 strided 切片 | No | fast 的 Q/dq 用 [seq_len, heads*dim] 二维声明取**连续行段**（§3.2）；所有 GM 切片最内维连续 |
| GPU 专用 API | Yes | 全部替换（rev1 §3.5.2 差异表继承 + §3.5.2 本表新增行） |

#### 3.5.2 参考实现差异说明（相对 ref_gpu）

| 差异项 | GPU 版 | 本设计（Ascend） | 说明 |
|--------|--------|-----------------|------|
| Kernel 拓扑 | 单 kernel grid=seq_len per-token | **三 kernel**：fast（persistent 20）+ generic（grid=seq_len）+ cast（persistent 20），host 分派 | rev2 架构变更；等价性由 l0_dispatch_equiv 用例验证（§9.2） |
| Q/dq 访问布局 | [s, h, d] 逐 token | fast kernel 按 [s, h*d] 二维连续行段访问（同一内存零拷贝重解释） | 连续行段整块 DMA；generic 不变 |
| sm_scale | 预乘 Q（bf16）+ dq 尾部补乘 | 折叠进 diffm（fp32，一次 mul） | 数学等价（§1.3），少 2 个 bf16 舍入点（rev1 实测验证） |
| dk 累加 dtype | bf16 GM atomic | **fp32 GM ws 累加 + 尾部单次 cast** | **rev2 核心修复**：主仓 GPU 同为 bf16 atomic 但主仓测试从不 gate dk（README 根因结论）；fp32 仿真 ratio 0.9972 |
| dk 跨 block 累加次数 | S×topk（每 token 每 topk 行） | fast：S/T_BLOCK（每 block 整块一次）；generic：S×topk（fp32 下精度与次数解耦） | fast 片上 fp32 预累加 + 整块 atomic |

#### 3.5.3 本项目同类实现参考（同类引用，门禁项）

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/attention_sink/example_gqa_sink_bwd_bhsd/example_gqa_sink_bwd_bhsd.py` | **高度相似** | Kernel 2 flashattn_bwd：5×gemm_v0 + `T.tile.atomic_add(dV region, l0c)` L0C 直 src 整块 atomic（L437/472）+ arith_progression/broadcast/compare/bitwise_and/select 向量化 mask（L393-423）+ L0C→GM 直写 auto-cast（L478）+ GM→L1 直拷（L390-391）+ reduce_sum dim=-1（L386）+ host zeros 输出（L531-535） |
| `examples/dsa_sparse_finetune/sparse_mla_topk_reducesum.py` | **高度相似**（同目录姊妹算子） | persistent `T.Kernel(grid_size=20)` + `ntask=T.max(0,T.ceildiv(total-cid,grid))` striped 轮询（L262/309/1302）；symbolic 多动态维（L104-106/142-144）；host panel_mode 分发 + `_panel_gate_ok` 门控 + `_BUILD_CACHE`（L1625-1761，`DESIGN_panel_mode.md` §8 分派机制蓝本）；GM workspace 显式参数 + workspace_idx（L259-262/733） |
| `examples/unsorted_segment_sum/unsorted_segment_sum.py` | 高 | 逐行 atomic scatter（L50）+ 动态 if 内 T.copy（L42）+ host zeros（L247） |
| `examples/lightning_indexer/example_lightning_indexer_dynamic_shape.py` | 中 | T.symbolic 动态 shape + enable_relu（L18-33/54-60） |
| `examples_experiment/dsa_sparse_finetune/indexer_bwd.py`（rev1 实现，本仓） | **直接复用** | generic 路径主体（dq 1.0000 / dw 全过实测）；CV 中转 workspace 显式参数机制（4 组往返 flag 生成已用最小复现 kernel 验证）；动态 grid 多形状复用编译产物 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量（接口不变，rev1 §4.1 基础上新增 BlockIndices）

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| IndexQ | [seq_len, heads, dim] | bfloat16 | ΣS=2048, H=16, D=128（fast kernel 内按 [seq_len, heads*dim] 二维解释，零拷贝） |
| Weights | [seq_len, heads] | bfloat16 | |
| IndexK | [seq_len, dim] | bfloat16 | 同一 K 空间被 varlen 各 batch 引用（bos 偏移） |
| AttnScore | [seq_len, topk] | float32 | teacher 概率 |
| IndexScore | [seq_len, topk] | float32 | student log_softmax 概率 |
| TopkIndices | [seq_len, topk] | int32 | 值域 [-1, seq_len)（-1 仅 generic 路径处理；fast 契约下无 -1） |
| Offsets | [batch_plus_one] | int32 | varlen cu_seqlens |
| TokenIndices | [seq_len, 2] | int32 | host 预计算 (i_b, i_t)，generic 用（rev1 已验证） |
| BlockIndices | [num_blocks, 3] | int32 | **rev2 新增**：host 预计算 (bos, t0_local, seq_len)，per 段 ceildiv(len_i, T_BLOCK) 行，fast 用 |

### 4.2 输出张量与工作区（host `torch.zeros` 预分配后传入 kernel，无 out_idx）

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| dIndexQ | [seq_len, heads, dim] | bfloat16 | 每 token 行独占写 |
| dWeights | [seq_len, heads] | bfloat16 | 每 token 行独占写 |
| dIndexK | [seq_len, dim] | bfloat16 | **由 cast kernel 从 ws_dk 产出**（fast/generic 共用） |
| ws_dk | [seq_len, dim] | **float32** | **rev2 新增**：dk fp32 累加工作区（host torch.zeros；cast kernel 输入；S=2048 时 1MB） |
| ws_logits | [20, 2, TS*H, topk] | float32 | CV 中转（fast，双槽轮转）：20×2×16KB = 640KB |
| ws_dlog_a / ws_dlog_b | [20, 2, TS*H, topk] | bfloat16 | CV 中转（fast，双槽）：2×320KB |

### 4.3 中间缓冲区（fast kernel，TS=4, T_BLOCK=32, H=16, D=128, topk=64）

| Buffer 名 | Shape | dtype | 层级 | 大小 |
|-----------|-------|-------|------|------|
| K_l1 | [64, 128] | bf16 | L1（alloc_shared） | 16KB |
| Q_l1 | [64, 128] | bf16 | L1 | 16KB |
| dlog_a_l1 / dlog_b_l1 | [64, 64] | bf16 | L1 | 8KB ×2 |
| logits_l0c | [64, 64] | fp32 | L0C（alloc_fragment） | 16KB |
| dq_l0c | [64, 128] | fp32 | L0C | 32KB |
| dk_l0c | [64, 128] | fp32 | L0C | 32KB（跨 sub-tile init=False 累加） |
| logits_ub / dlog_ub | [64, 64] | fp32 | UB | 16KB ×2 |
| attn_ub / is_ub / diffm_ub | [32, 64] | fp32 | UB | 8KB ×3 |
| dw_acc_ub | [32, 16] | fp32 | UB | 2KB |
| per-t 小 tile（logits_t/diff_2d/w_2d/col_2d/tval_2d/causal/drelu/logits_m/dwq/dlog_t/zero_2d/valid_2d） | [16, 64] | fp32 | UB | 4KB ×12 |
| col_1d / w_t / diff_t / dw_t 等 1D 杂项 | [64]/[16] | fp32 | UB | ~1KB |
| （layout=1 增）indices_ub [64] int32、gather_ub [64,128] bf16、dk_ub [64,128] fp32 | | UB | +48.3KB |

generic kernel 缓冲表 = rev1 §4.3 原样 + dk_ub 改 fp32（[64,128] fp32 32KB，原 bf16 16KB）。

### 4.4 内存搬运路径（fast layout=0；无跨级违规）

```
GM[IndexQ]  --T.copy 连续行段--> L1(Q_l1)                        # GM→L1 直连（gqa_sink:390 先例）
GM[IndexK]  --T.copy 整块------> L1(K_l1)                        # GM→L1 直连
GM[AttnScore/IndexScore/Weights/TopkIndices] --T.copy--> UB
L1(Q_l1)+L1(K_l1) --gemm_v0(内部 L1→L0A/L0B)--> L0C(logits)
  --T.copy×2 经 ws_logits(GM)--> UB(logits_ub, enable_relu)       # 平台强制中转，显式参数化
UB(dlog_ub) --T.copy×2 经 ws_dlog(GM, auto-cast bf16)--> L1(dlog_a_l1/dlog_b_l1)
L1(dlog_a_l1)+L1(K_l1)   --gemm_v0--> L0C(dq) --T.copy(auto-cast)--> GM[dIndexQ]   # L0C→GM 直写
L1(dlog_b_l1)^T+L1(Q_l1) --gemm_v0--> L0C(dk) --T.tile.atomic_add--> GM[ws_dk]     # L0C→GM fp32 整块 atomic
UB(dw_acc) --T.copy(auto-cast)--> GM[dWeights]
cast kernel: GM[ws_dk] --T.copy--> UB(fp32) --T.copy(auto-cast)--> GM[dIndexK]
```

无 GM→L0 直搬；Cube 操作数全经 L1；Vector 数据在 UB；L0C↔UB / UB→L1(bf16) 的平台强制 GM 中转全部显式参数化并计入预算（§9.4）。✓ 符合硬件内存层级。

### 4.5 内存预算汇总

| 层级 | fast layout=0 | fast layout=1 | generic（rev1 +16KB） | 容量 |
|------|--------------|---------------|-----------------------|------|
| L0C | **80KB**（16+32+32） | 80KB | 44KB | 128KB ✓ |
| UB | ~105KB | ~153KB | ~98KB | 192KB ✓ |
| L1 | 48KB | 48KB | ~22KB | 512KB ✓ |

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| seq_len（ΣS） | `T.symbolic("seq_len")` | topk ≤ ΣS（每段长 ≥ topk，主仓 assert），典型 ≤ 64K |
| batch_plus_one | `T.symbolic("batch_plus_one")` | 2 ~ 1025 |
| num_blocks | `T.symbolic("num_blocks")`（fast） | Σ ceildiv(len_i, T_BLOCK)，作 BlockIndices 维与 persistent 轮询上界 |

fast/cast 的 grid 为编译期常量 20；动态 grid 仅 generic（rev1 已验证多形状复用编译产物）。

### 4.7 JIT 配置

```python
@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # C/V 自动分核合并
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # CV 交叉自动同步
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # MTE/Vector 自动同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # UB/L1 内存规划
})
# 三个 kernel（fast/generic/cast）同配置；无 out_idx，输出与 ws 全显式参数
# fast 的 layout 以编译期常量参数传入 → (layout=0, layout=1) 两个编译缓存键（panel _BUILD_CACHE 先例）
```

---

## 5. Tiling 策略

### 5.1 计算类型

混合（CV 融合）：fast 路径 Cube（3 GEMM/sub-tile × 8 sub-tile/block）+ Vector（per-t mask/dw/dlog）+ MTE（bulk 装载 + CV 中转 + fp32 atomic）；generic 为 rev1 per-token 结构。

### 5.2 Block 划分

```python
# fast kernel：
T_BLOCK = 32     # token/block（同段连续；K gather 与 dk atomic 的共享粒度）
TS      = 4      # sub-tile token 数（GEMM 升维合并粒度：M = TS*H = 64）
TSB     = 8      # sub-tile 迭代数 = T_BLOCK // TS
GRID    = 20     # persistent 编译期常量（host 查 cube_core_num，sparse_mla:1501）
# generic kernel（rev1 原样）：
block_I = 64     # topk 方向 tile；grid = seq_len（per-token）
```

**T_BLOCK=32 / TS=4 选取论证**：

- **L0C 硬约束**：三 fragment 同时驻留（无 liveness 复用假设的保守模型，与 rev1 预算/panel G3 门同口径）需 `TS×H×(topk+D)×4 + topk×D×4 ≤ 128KB`。TS=8 → 128KB **恰满**（零余量，编译器 staging 风险，rev2 拒绝赌博式恰满——最后一次重试不容编译期失败）；TS=6 → 104KB（81%）；**TS=4 → 80KB（62.5%，余量 48KB）✓ 选定**。
- **gemm 数**：TS=4 → 24 gemm/block = 0.75 gemm/token（rev1 per-token 3 gemm → **÷4**）；gemm 规模 [64,64,128]/[64,128,64]（单 gemm ~1.05 MFLOP，固定开销摊薄 4×）。
- **DMA 短行**：K gather 64 行/block（rev1 64 行/token → **÷32**）；dk atomic 1 条整块/block（layout=0，rev1 64 行/token → **÷2048**）。
- **dk 跨 block 累加次数**：S/T_BLOCK = 64 次/热行（rev1 S=2048 次 → **÷32**）；fp32 下精度与累加次数解耦（§9.3）。
- **TS 升级路径（Stage 3）**：TS=8（12 gemm/block，L0C 128KB 恰满需编译验证）与 TS=6/T_BLOCK=30（15 gemm/block，104KB）列为调优选项，Cube 段成瓶颈时按序试提。

### 5.3 约束分析（含 GEMM 非整除处理，门禁项）

- **16 对齐**：GEMM1 M=TS*H=64, N=topk=64, K=D=128；GEMM2 M=64, N=128, K=topk=64；GEMM3 M=topk=64, N=128, K=TS*H=64——M/N/K 全 16 倍数 ✓（ascend.py 16 列分形粒度）。generic 三 GEMM rev1 已验算（[64,16,128]/[16,128,64]/[64,128,16]）。
- **varlen 段长 % T_BLOCK ≠ 0（GEMM 非整除主场景）**：尾块 token 数 < T_BLOCK。处理：① Q 装载与 dq/dw 写出按"整块 if / 逐 token 守卫 else"双路（动态 if 内 T.copy，unsorted:42 先例）；② per-t `valid = (t_abs < ln)` 并入 causal mask（fill + bitwise_and），无效 token 的 dlog=0 → dq/dw/dk 贡献全零；③ K 整块装载 `IndexK[bos:bos+topk]` 依赖段长 ≥ topk（主仓 assert 语义，kernel 入口同步 assert）——**无越界**；④ dk 整块 atomic 目标 `ws_dk[bos:bos+topk]` 恒在段内 ✓。
- **token < topk 的 causal 非整除**：`compare(col_pos, t_abs, "LE")` 向量化掩蔽（l0_small / l0_equal_topk 用例覆盖，§9.2）。
- **topk % block_I == 0**（generic，主仓 L39 同款）；**fast 支持域 topk ∈ {16, 32, 64}**（dk_l0c [topk, D] fp32：topk=128 → L0C 总量恰满 128KB 超保守预算，topk=128 走 generic；§10.4 显式声明）。
- **heads % 16 == 0 且 ≤ 64**（Ascend Cube 分形粒度；与主仓 %8 的差异显式声明，H=8 不在支持域）。

### 5.4 注意事项

- layout=1 的 K gather 有效性上界用块内最大 token 位置 `t_max = min(t0+T_BLOCK, ln)-1`（pos ≤ t_max 才搬入），per-token 精确有效性仍由 causal mask 向量化完成（与 rev1 §5.4 逐位等价：无效行 K=0 → logits=0 → drelu=0 → 贡献全零）。
- persistent 单核串行处理多 block：ws_logits/ws_dlog 双槽轮转 + 同核程序序保证复用安全（写→读→写）；AUTO_CV_SYNC 按 workspace 名配对插 flag（rev1 已验证）。
- atomic 到 ws_dk 的行 `bos+pos` 在有效 pos 下必在段内（pos ≤ t_max < ln），无越界写（rev1 §5.4 论证继承）。

### 5.5 数据搬运性能可行性

```text
[REORDER-COST]
path: indexer_bwd rev2 fast 路径（layout=0，主仓默认配置 S=2048）
gm_passes: 读 IndexQ 8.4MB + IndexK 1MB + Scores 2MB + CV 中转读 16.6MB（L2 局部）
           写 dIndexQ 8.4MB + dWeights 64KB + ws_dk atomic 2MB + CV 中转写 16.6MB（L2 局部）
dma_transactions: bulk 装载/写回 ~7 条/block × 65 + 中转 4 条/sub-tile × 520 + atomic 1 条/block（layout=0）
                ≈ 455 + 2080 + 65 ≈ 2600 条（rev1：~277000 条，短行 DMA ÷100+）
gm_scalar_accesses: BlockIndices 3 次/block（195 次）+ Weights 行 32 次/block（rev1 为 8192 次 GM 标量）
address_div_mod_per_element: 0（行号加法预计算，无 div/mod）
active_cores / serial_tasks_per_core: 20 AIC persistent / ceil(65/20)=4 block 峰值（striped 均衡）
largest_case_timeout_gate: l0_default（S=2048）120s
result: pass（量化预算见 §9.4：估算 45-62us < 80us，全部锚点为实测数据，非临界）
```

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| fast 任务维 | **persistent 常量 grid** | `T.Kernel(20, threads=1, is_npu=True) as (cid)` + `T.serial(ntask)`（运行时边界）+ `pid = k*20+cid` | 消除动态 grid 风险与 wave 量化（65 任务 striped 均衡 20 核）；`T.serial` 运行时边界 sparse_mla:309 已验证 |
| fast block 内 sub-tile | 编译期常量迭代 | `T.serial(TSB=8)` | 静态边界；dk_l0c 跨 sub-tile init=(ts==0) 累加 |
| fast sub-tile 内 token | 编译期常量迭代 | `T.serial(TS=4)` | per-t 向量段 [H, topk] 小 tile（§3.2.1） |
| fast layout=1 gather/atomic 行 | 编译期常量迭代 | `T.serial(topk)` + 动态 `if` | 逐行条件搬运/scatter（unsorted:42/50 先例） |
| generic token 维 | 动态 grid | `T.Kernel(seq_len)` | rev1 已验证 |
| generic topk/行循环 | 编译期常量 | `T.serial(NI)` / `T.serial(block_I)` + 动态 `if` | rev1 已验证 |
| 元素级 | 向量化 | `T.tile.*` 整 tile 指令 | 无逐元素 T.Parallel 标量循环 |

### 6.2 循环伪代码

fast 见 §3.2.1（Kernel(20) → T.serial(ntask) → 装载 → T.serial(TSB) → GEMM1 → T.serial(TS) 向量段 → GEMM2/3 → block 尾部 atomic）；generic 见 §3.3（rev1 §3.3 结构）。

### 6.3 流水线优化

L0 不启用 T.Pipelined（与 rev1 一致，主仓 num_stages=0 本无流水）。fast 路径的引擎级重叠由 AUTO_CV_COMBINE/AUTO_CV_SYNC 自动排布：MTE2（下一 block 装载）∥ Cube（GEMM）∥ Vector（mask/dlog）∥ MTE3（中转写/atomic）。Stage 3 优化项：ws 双槽改多槽（中转与计算更深流水）、TS 提级（gemm 数再 ÷2）、causal mask 常量提升。

### 6.4 尾块处理

varlen 尾块（段长 % T_BLOCK ≠ 0）：§5.3 双路守卫（整块 if / 逐 token else + valid 并入 causal mask）。无 shape 尾块于 topk/H/D 维（整维进 tile 或 assert 拒绝）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 全自动同步（Developer）——零手动 `T.set_flag/wait_flag/barrier_all/cross_flag`（rev1 → rev2 不变）。

### 7.2 自动同步点（pass 注入；设计层列依赖关系）

| 依赖 | 保证方 |
|------|--------|
| Q/K/Scores 装载完 → gemm_v0 / tile.sub | AUTO_SYNC（MTE2→Cube / MTE2→V 序） |
| logits_l0c 定界 → ws_logits 写 → logits_ub 读（CV 中转两段） | **AUTO_CV_SYNC 按 buffer 名含 "workspace" 配对 GM 写/读自动插 CrossCoreSetFlag/WaitFlag**（rev1 已用最小复现 kernel 验证 4 组往返 flag 生成、按 cid 分槽、动态 grid 复用） |
| dlog_ub 就绪 → ws_dlog 写 → dlog_l1 读（bf16 中转两段） | 同上 |
| dlog_l1 就绪 → GEMM2/3 | AUTO_CV_SYNC（MTE1→Cube，gemm_v0 内建） |
| dk_l0c 跨 sub-tile 累加次序 | 程序序（同核串行）+ init=False |
| ws_dk 的跨 block atomic 竞争 | 硬件 DMA 原子性（tl.ascend_atomic_add） |
| fast/generic kernel 全部完成 → cast kernel | host 同 stream 顺序提交（程序序） |
| persistent ws 双槽复用（写→读→写） | 同核程序序 + 双槽轮转（ts % 2） |
| host 检测 .item() 读回 | runtime stream 同步（纯 kernel 计时口径排除，§9.4） |

### 7.3 pass_configs 配置

见 §4.7（4 True，gqa_sink_bwd:62-67 逐键一致）。

---

## 8. 融合算子设计

### 8.1 融合算子判定

**判定结果**: 是（fast：3 GEMM + elementwise 后处理 + fp32 整块 atomic 单 kernel 融合 + 尾部 cast 小 kernel；generic：rev1 单 kernel 融合 + cast）。

### 8.2 CV 交互设计（Developer 模式）

- 三 kernel 均无 `workspace_idx`（中转 ws 全显式参数传入，规避 auto workspace 动态 grid 限制——rev1 实测方案）。
- Cube↔Vector 交换：logits（L0C→UB，经 ws_logits）与 dlog（UB→L1，经 ws_dlog_a/b）——两处平台强制 GM 中转，流量与 flag 机制均已入账（§5.5 / §7.2 / §9.4）。
- dq 走 L0C→GM 直写（不经 UB，gqa_sink:478）；dk 走 L0C→GM fp32 atomic（不经 UB，gqa_sink:437）——**rev2 相比 rev1 减少 2 类中转**。

### 8.3 Cube 核计算流程（fast）

```python
T.gemm_v0(Q_l1, K_l1, logits_l0c, transpose_B=True, init=True)          # GEMM1 logits（relu 随中转 copy）
T.gemm_v0(dlog_a_l1, K_l1, dq_l0c, init=True)                           # GEMM2 dq（sm_scale 已折叠进 dlog）
T.gemm_v0(dlog_b_l1, Q_l1, dk_l0c, transpose_A=True, init=(ts == 0))    # GEMM3 dk（跨 sub-tile fp32 片上累加）
```

### 8.4 Vector 核计算流程（fast，per-t [H, topk] tile）

```python
T.copy(logits_ub[u*H:(u+1)*H, :], logits_t)                  # UB region 行块
T.tile.broadcast / fill / compare / bitwise_and               # causal & valid mask（layout=1 含 indices 广播）
T.tile.select(logits_m, causal, logits_t, 0.0, "VSEL_TENSOR_SCALAR_MODE")
T.tile.mul(dwq_2d, diff_2d, logits_m); T.reduce_sum(dwq_2d, dw_t, dim=1)   # dWeights（relu 后 logits ✓）
T.tile.compare(drelu, logits_t, zero_2d, "GT"); T.tile.bitwise_and(causal, causal, drelu)
T.tile.mul(dlog_t, diff_2d, w_2d); T.tile.select(dlog_t, causal, dlog_t, 0.0, ...)  # d_logits（d_relu=1{logits>0} ✓）
```

### 8.5 pass_configs 配置

见 §4.7。

### 8.6 注意事项

- dlog_a_l1 / dlog_b_l1 双 buffer：GEMM2 不转置读、GEMM3 transpose_A 读，同一 L1 buffer 两种 fractal 读法无先例，双 buffer 规避（rev1 §8.6 结论继承）。
- atomic src 与 ws_dk dtype 必须一致（fp32/fp32）——fast layout=0 用 L0C 直 src（fp32 ✓）、layout=1/generic 用 UB fp32 中转；**不再出现 fp32→bf16 cast 后 atomic 的舍入累加**（rev1 错误①根因，§12）。
- cast kernel 必须在 fast/generic kernel 之后由 host 同 stream 顺序提交（dk 未定界前不得 cast）。

---

## 9. 验证方案

### 9.1 Golden 函数（复刻主仓 ref_indexer_bwd，逐字节一致——rev1 §9.1 原样）

```python
def ref_indexer_bwd(Q, Weights, K, TopkIndices, AttnScore, offsets):
    """主仓 ref_gpu/indexer_bwd.py:185-215 逐字节复刻（autograd 参考实现），NPU 上运行。"""
    Q.requires_grad_(True); Weights.requires_grad_(True); K.requires_grad_(True)
    softmax_scale = Q.shape[-1] ** -0.5
    all_loss, all_log_topk_prob = [], []
    for i in range(offsets.shape[0] - 1):
        assert (offsets[i + 1] - offsets[i]).item() >= TopkIndices.shape[-1]
        q = Q[offsets[i]:offsets[i+1]]; weights = Weights[offsets[i]:offsets[i+1]]
        k = K[offsets[i]:offsets[i+1]]
        topk_indices = TopkIndices[offsets[i]:offsets[i+1]]
        attn_score = AttnScore[offsets[i]:offsets[i+1]]
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
    loss = torch.stack(all_loss).sum(); loss.backward()
    return torch.cat(all_log_topk_prob, dim=0).exp(), Q.grad, Weights.grad, K.grad
```

**输入构造**：① arange 系（l0_default / l0_single_seq / l0_small / l0_equal_topk）：复刻主仓 `test_kernel:218-244`（`torch.manual_seed(42)`；q/w/k = randn bf16；attn_score = softmax(masked ones) fp32；topk_indices = repeat(arange(topk), "k -> s k")）；golden 的 index_score 取 ref 返回的第一个值（与主仓一致）。② l0_shared_shifted：`topk_indices = repeat(arange(5, 5+topk), "k -> s k")`。③ l0_generic_random：per 段 `torch.randint(0, s, (s, topk))` 固定 seed（值域 [0, s)，含 pos>t 因果无效与重复索引；golden 的 gather/autograd 通用良定义；**-1 不在 L0 构造**——golden 的 torch.gather 不接受负索引，-1 语义边界列 L2/Boundary 由 Stage 2 扩展）。

### 9.2 L0 门槛测试计划（tilelang-op-test-design 场景 A 生成，仅 L0）

**测试策略判定**（场景 A Phase 2）：计算类型 = 混合（CV 融合）；复杂度 = Fusion；数学特征 = matmul + relu + broadcast 乘 + 向量化 mask + 列归约 + fp32 atomic scatter + **host 分派（双路径架构）**；综合类别 = **Fusion（反向梯度算子，双路径分派变体）**。测试重点 = 功能验证 + 精度收敛（L0）+ **分派正确性**；dtype_count=1（bf16 固定精度路径）、shape_count=6、block_count=2（fast T_BLOCK=32/TS=4 + generic block_I=64）、**dispatch_count=3（layout=0 / layout=1 / generic）**。

| 用例名 | 级别 | Shape（B, S, H, D, topk） | offsets | topk_indices 构造 | 预期分派 | 说明 |
|--------|------|---------------------------|---------|-------------------|----------|------|
| l0_default | L0 | (1, 2048, 16, 128, 64) | [0, 1023, 2048] | arange 平铺（主仓默认） | fast layout=0 | **主仓默认配置 = perf 用例**：varlen 2 段非对称（1023+1025）+ token<topk mask 边界 + 跨 block fp32 累加（64 次/热行）峰值 |
| l0_single_seq | L0 | (1, 1024, 16, 128, 64) | [0, 1024] | arange 平铺 | fast layout=0 | 单段无 varlen 切分；1024%32=0 恰整除对照（无尾块守卫路径） |
| l0_small | L0 | (1, 128, 16, 128, 64) | [0, 128] | arange 平铺 | fast layout=0 | 小规模冒烟：token<topk 占比高（大量 causal mask 置零路径）+ 4 block persistent 轮询 |
| l0_equal_topk | L0 | (1, 64, 16, 128, 64) | [0, 64] | arange 平铺 | fast layout=0 | **段长 == topk 边界**（assert 下界；t<63 全 mask、仅末 token 全有效；尾块 = 整段） |
| l0_shared_shifted | L0 | (1, 512, 16, 128, 64) | [0, 512] | arange(5, 69) 平铺 | fast layout=1 | **共享非 arange**：逐行 gather/atomic + indices-compare mask + pos>t 大量无效 + 非零起点 |
| l0_generic_random | L0 | (1, 512, 16, 128, 64) | [0, 512] | randint(0, 512) per-token | generic | **随机 per-token 索引**：rev1 已验证结构 + dk fp32 ws 修复验证；含重复索引与 pos>t |
| l0_dispatch_equiv | L0 | (1, 2048, 16, 128, 64) | [0, 1023, 2048] | arange 平铺（=l0_default） | force 三路 | **分派正确性/等价性**：同输入分别 force fast-0 / fast-1 / generic（dispatch 参数注入），三路结果独立 vs golden 判定（panel P1 panel-vs-generic 等价性先例）+ host 检测函数断言（arange→fast-0、shifted→fast-1、random→generic） |

- 全部用例比对 3 输出（dq/dw/dk）vs golden（ref 的 Q.grad / Weights.grad / K.grad），混合容差按 §9.3，三输出分别判定，任一失败该用例 FAIL。
- **dk 精度断言依据**：全链 fp32 累加（L0C fp32 片上 → ws_dk fp32 GM atomic → cast 单次舍入）→ 受控仿真 ratio 0.9972（README：同贡献 fp32 顺序累加通过、bf16 同路径 0.9610 复现失败）；fast layout=0 每 block 片上 fp32 预累加后仅 64 次跨 block fp32 原子累加，累加噪声低于仿真模型（仿真的逐 token bf16 部分和路径舍入与本设计 dlog 舍入点一致，GM 累加段全 fp32）。
- L1（不规则 shape）/ L2（异常：-1 索引、段长<topk、H=8 拒绝路径）/ Boundary（特殊值）由 Stage 2 在 L0 通过后按场景 B 扩展，本节不生成。

### 9.3 精度标准（混合容差，与 rev1 §9.3 一致——门限不变，修复的是路径）

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|-------|------|------|---------------------|------------------------|
| bfloat16 | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |
| float32 | 2⁻¹⁶ (1.53e-5) | 2⁻¹⁰ (9.77e-4) | 1e-2 | 0.99 |
| int32 | 0 | 0 | 0 | 1.0（精确匹配） |

> bfloat16 为 3 输出（dIndexQ/dWeights/dIndexK）比对标准；float32 为 Scores 输入 dtype（参与计算无直接输出比对，保持 proto 一致性；ws_dk 为内部工作区不参与比对）；int32 为索引输入。判定：逐元素 `|actual-golden| ≤ atol + rtol·|golden|` 且 `matched_ratio ≥ 0.99` 且 `max_abs_error ≤ max_abs_error_limit`。**rev1 失败教训**：rev1 dk（bf16 GM atomic，ratio 0.9405-0.9611）证明该门限下 bf16 多次累加路径不可行；rev2 dk 全链 fp32 后唯一 bf16 舍入点 = cast kernel 最终一次（与 dq/dw 输出同级），仿真 ratio 0.9972 对 0.99 门限余量 0.0072。

### 9.4 性能预算（实测校准；验收指标 2：纯 kernel 平均 < 80us @ 主仓默认配置）

**校准锚点（全部实测，无乐观假设）**：

| 锚点 | 数值 | 来源 |
|------|------|------|
| A1 rev1 per-token 通路实测 | 1522.6us（S=2048；GM workspace ~280MB/调用 + 131072 MTE2 短行 + 131072 MTE3 atomic + 6144 gemm） | README §性能实测（torch.npu 事件计时 20 轮均值，排除 H2D/同步/host 分配） |
| A2 姊妹算子 panel 路径实测 | 143.3us @ 73MB GM 流量 / 4.53 GFLOP → 聚合有效带宽 ~510GB/s（读 ~273GB/s ∥ 写 ~238GB/s 并发），MTE1（L1→L0 装载）串行为其瓶颈（52% AIC busy） | perf_tuning/panel_metric2_bench.md（graph-replay 2 组均值 + msprof 交叉 147.1us） |
| A3 gemm_v0 单次开销 | ~500-1500 cycles（含 flag 序与 L1→L0 装载；小 GEMM 固定开销为主） | rev1 §9.4 估算 + sparse_mla iter3 经验 |
| A4 短 DMA 开销 | ~100-200 cycles/条（行级 256-512B） | rev1 §9.4 MTE 估算口径 |

**fast 路径（layout=0）量化预算分解**（S=2048, H=16, D=128, topk=64 → 65 block，persistent 20 核，每核 3-4 block）：

| 项目 | 每 block（32 token） | 聚合 / 每核 | 依据 |
|------|---------------------|-------------|------|
| Cube | 24 gemm × ~500-800cy ≈ 12-19K cy ≈ 4.6-7.3us | 每核 3.25 block → **15-24us** | gemm 数/token 3→0.75（÷4）；锚点 A3 |
| Vector | per-t ~250cy × 32 ≈ 8K cy ≈ 3.1us | 每核 **~10us**（与 Cube 同核分时，部分与 MTE 重叠） | [16,64] tile 整拍指令 ×~12 op/t；锚点 A3/A4 类比 |
| MTE2 读 | bulk 176KB（Q 128 + K 16 + Scores 32）+ 中转读 256KB（ws_logits/dlog，**L2 局部**） | HBM 读 ~11.4MB + L2 读 16.6MB | @A2 读带宽 273GB/s → HBM 42us；L2 ~1TB/s → 17us（并行） |
| MTE3 写 | bulk 129KB（dq 128 + dw 1）+ atomic 32KB + 中转写 256KB（L2 局部） | HBM 写 ~10.5MB + L2 写 16.6MB | @A2 写带宽 238GB/s → HBM 44us；L2 17us（并行） |
| **引擎流水关键路径** | 四引擎（MTE2/Cube/Vector/MTE3）重叠，读/写通道并发 | **≈ max(HBM 读 42, HBM 写 44, Cube+Vector 25-34) + 3.25→4 wave 尾部 ≈ 45-55us** | panel 同款重叠模型；panel 的 MTE1 串行瓶颈在本设计被 ÷4 的 gemm 数与块内常驻 K_l1/Q_l1 规避 |
| cast kernel | 1.5MB（读 1MB fp32 + 写 0.5MB bf16）bulk，persistent 20 核 | **~2-3us** | 带宽口径同 A2 |
| 分派检测 | 设备侧比较 [2048,64] int32 + 1 次 D2H bool | **~4us**（host 同步在纯 kernel 计时口径外；保守计入） | 0.5MB 读 @ 273GB/s ≈ 2us |
| **合计估算** | | **45-62us < 80us（余量 22-44%）** | **非临界**（rev1 教训：不接受"依赖 L2 命中才可达"类临界断言——本预算将 L2 中转流量单列，HBM 流量按 A2 实测带宽核算，HBM 下限流量 22MB 为输入输出数学必需） |

**与 rev1 的结构性对比**（为何 19 倍差距被消除）：gemm 数 6144→1560（÷4，规模 ↑4×）；短行 DMA ~262144→~2600（÷100）；GM 总流量 ~280MB→~55MB（÷5，其中 HBM ~22MB / L2 中转 ~33MB）；atomic 条数 131072→65（layout=0 整块，÷2016）；workspace 中转从"per-token 4 组"降为"per-sub-tile 4 组"且 dq/dk 直写免中转。

**若需进一步压缩（Stage 3 调优路径，按优先级）**：① TS=8（gemm 12/block，L0C 128KB 恰满需编译验证）；② dlog 单 L1 buffer（中转流量 -25%，需验证同 buffer 双 fractal 读法）；③ causal mask 常量提升（col_1d/tval_2d 块级预构造）；④ ws 多槽流水（中转与计算更深重叠）。

**性能哨兵用例**：perf_l0_default（=l0_default 配置，fast layout=0 自动命中），单 case timeout 120s。

---

## 10. 风险点与注意事项

### 10.1 技术约束检测结论（每项附证据，rev2 更新）

| # | 风险 | 结论 | 证据 |
|---|------|------|------|
| R1 | **GM atomic_add 支持性（含 fp32）** | ✅ **fp32 明确支持**（rev1 仅核 bf16）：`T.tile.atomic_add` dst 须 GM、src 须 local（UB/L0C/L1）、dtype 须一致，docstring dtype 列表含 **float32**；且支持 **L0C 直 src** 与**连续 region 整块**。rev2 的 dk fp32 ws 方案硬依赖成立 | ascend_tile.py:313-349（dtype L324）；gqa_sink:437/472；unsorted:50 |
| R2 | 动态索引 GM 访问 | ✅ layout=0 免 gather（整块连续）；layout=1/generic 用 fill-0 + 动态 `if` 内逐行 T.copy（rev1 已验证） | unsorted:42、sparse_mla:383；rev1 实测 |
| R3 | 小 GEMM 形状约束 + **L0C 容量** | ✅ fast 三 GEMM M/N/K 全 16 倍数；**L0C 80KB < 128KB**（TS=4 保守选取，TS=8 恰满被拒，§5.2）；generic 44KB | ascend.py:418-463；gqa_sink:329-343；panel G3 门（131072B 预算）同口径 |
| R4 | 动态 shape / 动态 grid | ✅ 三 symbolic（seq_len/batch_plus_one/num_blocks）；**fast/cast 用常量 grid=20 规避动态 grid**（persistent 先例）；generic 动态 grid rev1 已实测多形状复用 | sparse_mla:104-106/262/1302；rev1 README |
| R5 | T.Kernel 签名 | ✅ `T.Kernel(grid, threads=1, is_npu=True) as (cid)`；persistent 常量 grid 与动态 grid 两形态均有先例 | gqa_sink:317；sparse_mla:262 |
| R6 | **bf16 精度路径（rev1 错误①）** | ✅ **已修复**：dk 全链 fp32（L0C fp32 片上累加 → ws_dk fp32 GM atomic → cast kernel 单次舍入）；受控仿真 ratio 0.9972（bf16 同路径 0.9610 复现失败、fp32 0.9972 通过）；dq/dw 路径不变（1.0000 / 0.99+ 实测）。**dk 累加链上不再存在任何 bf16 舍入环节** | README §dk 失败根因（受控仿真）；ascend_tile.py:324 |
| R7 | **UB→L1 bf16 直连 static_assert（half-only）** | ✅ dlog/gather 经显式 GM 中转两段 copy + workspace 命名配对 flag（rev1 实现已验证 4 组往返 CrossCoreSetFlag/WaitFlag 生成、按 cid 分槽）；auto workspace 动态 grid 限制由显式参数规避 | common.h copy_ub_to_l1 static_assert；ascend_combinecv.cc FetchWorkspaceName；cython_wrapper.pyx；README 偏差 #1/#2 |
| R8 | **L0C→UB 编译器改写 GM 往返** | ✅ logits relu 中转显式参数化（ws_logits 640KB）；流量 16.6MB/调用已入 §9.4 预算（L2 局部） | ascend_workspace_reduction.cc；README 偏差 #2 |
| R9 | **GEMM 非整除 / varlen 尾块**（门禁项） | ✅ §5.3：整块 if / 逐 token 守卫 else 双路 + valid 并入 causal mask + K/atomic 无越界论证；l0_equal_topk/l0_small 用例覆盖 | unsorted:42 动态 if 先例；主仓 assert 语义 |
| R10 | **persistent 运行时任务边界** | ✅ `T.serial(ntask)` 运行时边界（ntask=T.max(0,T.ceildiv(...))）为 sparse_mla generic/panel 双变体已实测形态（rev1 设计将"动态循环边界不支持"误扩到此处——该约束指 Pipelined/复杂边界，serial+运行时上界有直接先例） | sparse_mla:309/994-1266（panel graph-replay 143.3us 实测） |

### 10.2 性能风险（R11）

fast 路径预算 45-62us（余量 22-44%，§9.4），主要不确定性：① L2 对 CV 中转流量的实际吸收率（按 ~1TB/s 估算，panel ws_acc 同款模式佐证）；② 20 核 striped 轮询尾部不均（65 任务 → 3-4 block/核，已按 4 峰值计入）；③ gemm_v0 固定开销区间较宽（500-800cy 取中）。若 L0 精度通过但性能不达标：§9.4 四条 Stage 3 路径（TS 提级 / dlog 单 buffer / mask 常量提升 / ws 多槽），**不回退设计**（语义正确性不受调优影响）。generic 路径性能不考核（正确性兜底路径，1522.6us 量级维持）。

### 10.3 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| atomic 到未清零 buffer | ws_dk/输出用 torch.empty | dk 结果含垃圾初值 | host torch.zeros 全部输出与 ws_dk |
| atomic src/dst dtype 不匹配 | L0C fp32 直接 atomic 到 bf16 GM | ascend_tile.py L215 ValueError | **rev2：dst 恒为 ws_dk fp32，src 恒 fp32（L0C 直或 UB 中转）** |
| gemm 操作数误放 UB | alloc 后直接 gemm_v0 | 运行时错误 | 操作数走 alloc_shared（编译器映射 L1） |
| UB→L1 bf16 单段直连 | `T.copy(ub_bf16, l1_bf16)` | 编译期 static_assert 失败 | 两段经 ws_dlog GM 中转（§3.2） |
| fast 契约外输入走 fast | host 检测误判 | 沉默错误结果 | 检测为精确布尔判定（非启发式）；l0_dispatch_equiv 用例覆盖 |
| 段长 < topk / H=8 / topk=128 走 fast | 非法配置 | assert 失败或 L0C 溢出 | kernel 入口同步 assert + host 分派域检查（§10.4） |
| cast kernel 先于主 kernel 提交 | host 乱序/多 stream | dk 部分丢失 | 同 stream 顺序提交（§7.2） |

### 10.4 支持域与分派审计

```text
[DISPATCH-COVERAGE]
supported_domain: dtype 组合固定（Q/W/K bf16 + Scores fp32 + 索引 int32）；seq_len/batch 任意
  （varlen，段长 ≥ topk）；heads ∈ {16,32,48,64}（%16==0，≤64）；dim % 16 == 0；topk ∈ {16,32,64}
  （2 的幂，主仓语义；topk=128 走 generic——fast 的 dk_l0c [topk,D] fp32 会顶满 L0C 128KB 保守预算）；
  sm_scale 任意 float
dispatch（rev2）：host 精确检测（非启发式）→ fast(layout=0) | fast(layout=1) | generic
  layout=0 契约：topk_indices 全 token == arange(0..topk-1) 平铺
  layout=1 契约：topk_indices 全部行严格相等且值域 [0, 段长)（无 -1）
  契约外全部输入 → generic（rev1 结构，任意 per-token 索引含 -1 语义）
  dispatch attr 四值（auto/fast_arange/fast_shared/generic，缺省 auto）供测试注入与调用方显式声明
generic_fallback: generic kernel（全 supported_domain 覆盖，含 -1 索引）
specializations: fast 双 layout（编译期常量参数 → 2 编译缓存键，panel _BUILD_CACHE 先例）
unsupported_inputs: heads=8（%16≠0，Cube 分形粒度物理约束，与主仓 %8 差异显式声明）；
  段长 < topk 的 varlen（语义非法，主仓同款 assert）；fp16/fp32 主输入 dtype（精度路径固定 bf16）
result: pass（三路全覆盖 + 精确检测，无枚举遗漏）
```

### 10.5 Host 侧审计

```text
[HOST-METADATA-AUDIT]
operation: prepare_token_indices(offsets) → TokenIndices；prepare_block_indices(offsets, T_BLOCK) → BlockIndices
input_stride -> output_stride: 新建张量（CPU int32 构造）→ .npu() 一次 H2D（两表可合并一次传输）
shares_storage / same_data_ptr: false（新建输入，数据准备类）
aclnn_or_physical_copy: false（CPU 构造 + 单次 H2D；sparse_mla:1278-1296 同款先例）
result: allow

operation: torch.zeros(dIndexQ/dWeights/dIndexK/ws_dk/ws_logits/ws_dlog_*) → kernel 显式参数
input_stride -> output_stride: 新建输出/工作区 buffer（atomic 语义要求零初始化）
shares_storage / same_data_ptr: false（新建输出）
aclnn_or_physical_copy: false（分配 + 清零；unsorted:247 / gqa_sink:531-535 先例）
result: allow

operation: 共享性检测 (topk_indices == arange).all() / (topk_indices == topk_indices[:1]).all() + .item()
input_stride -> output_stride: 只读比较（不改输入）
shares_storage / same_data_ptr: true（只读）——无写入
aclnn_or_physical_copy: false（设备侧比较原语 + 1 次标量 D2H；panel host 门控 _panel_gate_ok + ErrFlag .item() 同类先例）
result: allow（分派逻辑必需；.item() 同步在纯 kernel 计时口径外，§9.4）

operation: kernel 调用后直接返回 (dq, dw, dk)
aclnn_or_physical_copy: false（无 .to()/.contiguous()/cat——输出 shape 与原始一致）
result: allow
```

---

## 11. 交付清单

### 11.1 目录结构

```
examples_experiment/dsa_sparse_finetune/
├── DESIGN.md                     # 本设计文档（indexer_bwd rev2，覆盖 rev1）
├── proto_indexer_bwd.yaml        # indexer_bwd 接口规格（rev2 覆盖写；目录另有 sparse_mla 的 proto.yaml 兄弟工件，不覆盖）
├── indexer_bwd.py                # 【rev2 待改写】fast/generic/cast 三 kernel + host 分派 interface
├── test_indexer_bwd.py           # 【rev2 待改写】golden + L0 7 用例（§9.2）+ 性能基准 + main
├── README.md                     # 实验记录（rev1 失败证据，保留）
└── ref_gpu/indexer_bwd.py        # 主仓参考（只读）
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| DESIGN.md（本文件，rev2） | 已完成 | 双路径架构 + dk fp32 方案 + 实测校准预算 + L0 计划（§9.2） |
| proto_indexer_bwd.yaml | 已完成（rev2 覆盖） | 接口规格（dtype 全集 = §9.3 表；attrs = §4/§1 派生 + dispatch 四值） |
| indexer_bwd.py | 待改写（rev2） | 三 kernel + indexer_bwd_interface（含检测/分派/BlockIndices 构造） |
| test_indexer_bwd.py | 待改写（rev2） | golden（§9.1）+ L0 7 用例（§9.2）+ perf 基准（§9.4）+ main（--level 分发） |

### 11.3 命名规范

- kernel 函数：`tl_indexer_bwd_fast`（layout 0/1 两编译键）/ `tl_indexer_bwd_generic` / `tl_dk_cast`（对齐主仓 tl_ 前缀命名）
- host 入口：`indexer_bwd_interface(q, weights, k, attn_score, index_score, topk_indices, offsets, dispatch="auto")`（签名向后兼容 rev1，新增尾参带默认值）
- 检测 helper：`_detect_indices_layout(topk_indices, topk) -> int`（供测试断言）

### 11.4 实现顺序

1. ✅ 设计文档（DESIGN.md rev2）+ proto_indexer_bwd.yaml + L0 门槛测试计划（§9.2）
2. ⬜ generic kernel dk fp32 化改造（最小改动验证 dk 修复：4 个 rev1 L0 用例 dk 转 PASS）
3. ⬜ cast kernel + host 分派/检测/BlockIndices（l0_dispatch_equiv 用例先行）
4. ⬜ fast kernel（layout=0 → layout=1 顺序；先 l0_small 冒烟再 l0_default）
5. ⬜ L0 门槛测试全通过（7 用例 × 3 输出，混合容差）
6. ⬜ 性能验收（纯 kernel 平均 < 80us @ 主仓默认配置；不达标按 §9.4 Stage 3 路径，不回退设计）
7. ⬜ 扩展分层套件（L1/L2/Boundary，tilelang-op-test-design 场景 B）
8. ⬜ 全量套件 + 覆盖门禁（coverage_check.py --proto proto_indexer_bwd.yaml）

---

## 12. rev2 修订说明（相对 rev1 的关键调整，revision 专属章节）

### 12.1 两大设计错误与修复对照

| 维度 | rev1（已否决） | rev2（本设计） | 为何不会再犯同一错误 |
|------|---------------|---------------|---------------------|
| **错误① dk 精度路径与容差门限互斥** | dk 路径：fp32 L0C 部分和 → cast bf16 → GM bf16 逐块 atomic（复刻主仓 GPU）；§9.2 断言"容差内收敛"。**实验否定**：主仓测试（arange 平铺）下每热行 ~1000-2000 次 bf16 原子累加 → 4 用例 dk 全 FAIL（0.9405-0.9611）；受控仿真同贡献 bf16 顺序累加 0.9610 复现失败、fp32 累加 0.9972 通过 | **dk 全链 fp32**（fast 与 generic 统一）：L0C fp32 片上累加 → `T.tile.atomic_add` fp32 到 ws_dk GM 工作区（ascend_tile.py:324 明确支持 fp32）→ 尾部 cast kernel 单次 fp32→bf16 舍入。fast layout=0 更进一步：跨 block 累加次数 S×topk → S/T_BLOCK=64，且整块 region atomic（gqa_sink:437 先例） | ① 精度路径上**不存在任何 bf16 多次累加环节**（唯一舍入点 = 最终 cast 一次，与 dq/dw 输出同级）；② 修复方案有**受控仿真先行验证**（0.9972 vs 门限 0.99，余量 0.0072）+ 源码 dtype 列表直接证据；③ 根因教训写入流程：主仓 GPU "同路径"不再作为精度充分条件（主仓测试从不 gate dk——上游未验证过的断言不可继承，必须本平台受控仿真） |
| **错误② 性能预算未计入平台强制偏差（19 倍差距）** | per-token 通路（grid=seq_len，逐 token 3 小 GEMM + 逐行 gather/atomic）；§9.4 预算"重叠下界 118-197us、依赖 L2 命中与流水重叠 < 80us 临界可达"。**实验否定**：UB→L1 bf16 / L0C→UB 平台强制 GM 中转（static_assert / 编译器改写）→ workspace ~280MB/调用 → 实测 1522.6us；且 auto workspace 不支持动态 grid，中转全靠显式参数 | **双路径架构**：fast（persistent grid=20 + T_BLOCK=32 共享 K gather + 3 GEMM 升维合并 [TS*H=64 行] + causal 向量化 mask + L0C→GM 直写/整块 fp32 atomic）+ generic（rev1 结构兜底）+ cast；**预算全部实测锚点**（A1 rev1 1522.6us 分解 / A2 panel 143.3us@73MB 带宽口径 / A3 gemm / A4 短 DMA），CV 中转流量 33MB 显式入账（L2 局部单列），HBM 下限 22MB 按实测带宽核算 → 45-62us（余量 22-44%，非临界） | ① 预算模型**逐项标注锚点来源**，不存在无锚点的乐观项；② 平台强制偏差（static_assert / 编译器改写 / auto workspace 限制）作为**已知约束进入 §3.5.1 检查表**（R7/R8），中转流量是预算的一等公民而非事后发现；③ 结构上 per-token 通路被整体替换（gemm ÷4、短 DMA ÷100、atomic ÷2016、GM 流量 ÷5），不存在"临界达标"的依赖链；④ 性能不达标时的 Stage 3 路径（§9.4）为参数级调优（TS/槽深），不动架构 |

### 12.2 rev1 保留部分（已验证资产复用清单）

| 资产 | 复用方式 |
|------|---------|
| generic kernel 主体（`indexer_bwd.py`：grid=seq_len、block_I=64 三 GEMM、向量后处理、逐行 gather/atomic） | 原样保留为 generic 路径，仅 dk 出口 fp32 化（§3.3 两处修改）；dq ratio=1.0000 / dw 全过的结构不动 |
| CV 中转 workspace 显式参数机制（buffer 名含 workspace + AUTO_CV_SYNC 配对 flag + 按 cid 分槽 + 动态 grid 复用） | fast kernel 的 ws_logits/ws_dlog 直接继承该机制（最小复现 kernel 已验证） |
| API 证据链（rev1 §3.2 全表） | 直接继承（§3.2 未标【新】各行），rev2 新增 10 行全部附新证据（gqa_sink/sparse_mla/ascend_tile.py 源码行号） |
| golden 与混合容差（§9.1/§9.3） | 原样（golden=ref_indexer_bwd 逐字节复刻；bf16 atol 9.77e-4/rtol 1.56e-2/max_abs 1e0/ratio 0.99 维持不变） |
| host 预处理先例（prepare_token_indices、torch.zeros 输出） | 原样 + 扩展 prepare_block_indices（同款 CPU 构造 + 单次 H2D） |

### 12.3 与历史备份的差异声明

rev1 备份（`history_version/design_rev1.md`，若 orchestrator 未归档则以本节记录为准）核心选择：单路径 per-token / grid=seq_len / dk bf16 GM atomic / block_I=64。rev2 关键选择：**双路径 + host 精确分派 / fast persistent grid=20 + T_BLOCK=32/TS=4 / dk 全链 fp32 + cast kernel / 三 kernel 拓扑**——编程模式（Developer）、pass_configs（4 True）、golden、混合容差、接口签名（向后兼容）与 rev1 一致，其余架构层面全部差异化，无与历史备份关键选择重合的路径。
