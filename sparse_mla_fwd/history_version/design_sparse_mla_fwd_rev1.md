# sparse_mla_fwd 算子设计文档

> 迁移来源：`examples_experiment/dsa_sparse_finetune/ref_gpu/sparse_mla_fwd.py`（GPU 版主仓参考，只读，语义权威；逐字节校验与 tile-ai/tilelang main 一致）
> 目标平台：Ascend 910B3（20 Cube Core + 40 Vector Core；L1 512KB / UB 192KB / L0C 128KB / L0A 64KB / L0B 64KB 每核，见 api-kernel-memory.md §2 平台表）
> 编程模式：**Developer**（用户指定；若性能调优后仍不达标再升级混合——先例 indexer_bwd 的 fast/generic host 分派升级路径）
> 双后端约束（用户硬性要求）：**ascendc（target="ascendc"）与 PTO（target="pto"）双后端均须通过编译 + 运行 + 精度验证**
> 性能目标：latency < 24288 us（B=1, S=4096, H=128, HKV=1, DQK=576, DV=512, topk=1024, bf16；理论算力 ~1.167 TFLOP → 有效算力 ≥ 48 TFLOPS ≈ Cube 峰值（~320 TFLOPS 口径）的 ~15%）

---

## 1. 概述

### 1.1 算子名称

sparse_mla_fwd（DSA 稀疏 MLA 前向，varlen packed 布局）

### 1.2 功能描述

对 varlen packed 序列的每个 packed 行 `s_row`（属于 batch `b`、段内位置 `s_i`）、每个 kv_group `g`，按 `Indices[s_row, g, j]`（j ∈ [0, topk)，值域含 -1 与越界填充）gather KV 行，计算 scaled dot-product score（K 主段 512 + 吸收段 tail 64，合并 576 维），做 causal + 无效槽 mask，流式在线 softmax（m_i / sumexp / alpha 更新），`O = P @ V`（V = KV 行的前 512 列），同时输出 logsumexp。输出 `Output[ΣS, H, DV]` 与 `Lse[ΣS, H]`。

### 1.3 数学公式

对每个 packed 行 s_row（batch b、行内位置 s_i ≡ s_row − bos）、每个 kv_group g、每个 head h：

$$\text{score}[h,j] = \big(Q[s_{row},h,:512]\cdot KV_{main}[idx_j,:512] + Q[s_{row},h,512:576]\cdot KV_{tail}[idx_j,512:576]\big)\times sm\_scale$$

$$idx_j = Indices[s_{row},g,j],\quad mask[j] = 1\{idx_j \le s_i\} \wedge 1\{idx_j \ne -1\}$$

在线 softmax（块 t = 0..NI-1，BI = topk/NI）：

$$m_i^{(t)} = \max\big(m_i^{(t-1)},\ \max_{j \in t} \widetilde{score}[h,j]\big),\quad \alpha = \exp\big(m_i^{(t-1)} - m_i^{(t)}\big)$$

$$p_j = \exp\big(\widetilde{score}[h,j] - m_i^{(t)}\big)\ (\text{masked 项为 } 0),\quad sumexp^{(t)} = sumexp^{(t-1)}\times\alpha + \textstyle\sum_{j\in t} p_j$$

$$O = \Big(\sum_t \alpha\text{-rescaled } P^{(t)} \times V\Big) \Big/ sumexp^{(NI-1)},\quad V = KV[idx_j, :512]$$

$$Lse[s_{row},h] = \log\big(sumexp^{(NI-1)}\big) + m_i^{(NI-1)}$$

其中 $\widetilde{score} = score \times sm\_scale$（本设计在 V 侧乘 scale 后再取 max，与主仓「先 max 未缩放值、减法处补乘」代数等价，见 §3.2 步骤 5）；$sm\_scale$ 缺省 $= (dim+tail\_dim)^{-0.5} = 576^{-0.5}$。

### 1.4 算法描述（迁移语义注意点，硬性记录）

计算类型：**Cube + Vector 混合（CV 融合）**——GEMM（QK^T / PV）+ 向量链（mask / online softmax / rescale）+ 随机行 gather（GM→UB→GM workspace 中转）。复杂度：Fusion（flash-attention 型流式结构）。

| # | 主仓语义要点 | 本设计落地 |
|---|-------------|-----------|
| 1 | **重复索引不去重**：主仓 kernel 语义 = 重复索引重复计贡献；ref golden 的 scatter mask = 去重。两者仅在无重复输入上一致（主仓测试 randperm 无重复）。**迁移以主仓 kernel 语义为准**（本设计的 exp 后乘 mask 与 gemm 天然保留重复贡献） | 不做任何去重/共享检测，纯通用路径（先例 reducesum「纯通用路径」原则） |
| 2 | **causal 判据用行内位置 s_i**（`TokenIndices[s_row, 1]`），不是 packed 全局行号；行寻址恒等式 `s_row ≡ bos + s_i`（packed 布局保证，reducesum §3.3） | `mask = 1{idx ≤ s_i} · 1{idx ≥ 0}`，s_i 由 TokenIndices GM 标量读 |
| 3 | **越界 / -1 索引**：clamp 到合法行保护寻址 + mask 置 0 贡献（reducesum §5.4） | `rows = clamp(idx + bos, 0, ΣS_kv − 1)`（仅保护寻址）；mask 由 1{idx ≤ s_i}∧1{idx ≥ 0} 自然为 0 |
| 4 | **首行自注意**：t=0 时索引 [0] 有效（ref `mask[:,:,:,0]=True` 与 kernel `randperm(1)=[0]` 一致） | mask 链不特殊化 t=0，数学自然覆盖（idx=0, s_i=0 → 1{0≤0}·1{0≥0}=1 ✓） |
| 5 | **禁止针对 golden 固定数值/索引规律优化**：Indices 为任意合法分布（逐行独立/乱序/重复/-1/越界） | 单一通用 kernel 路径，无任何索引内容假设 |
| 6 | 主仓 `assert topk % block_I == 0`、`dim/tail_dim` 为 2 的幂、`is_causal == True`、`head_kv % 64 == 0`（>64 时） | 编译期 assert 原样继承（§3.5.1 / §11.4） |

Host 侧 Buffer 操作约束（ascend-constraints §5）：host 路径仅含 TokenIndices 预计算（CPU 构造 → 一次 H2D，reducesum `prepare_token_indices` 同款）、workspace / 输出 `torch.zeros` 分配、kernel 调用与验证；无任何 aclnn 触发点（§10.4 审计表）。

### 1.5 数据流图（单 kernel，persistent grid，每 task = (packed row, kv_group)）

```
GM[Q]        --T.copy 连续行段 [Hseg, 576]（GM→L1 直连，每 task 一次）--> L1(q_l1)
GM[Indices]  --T.copy [BI]（GM→UB）--> UB(indices_ub)
GM[TokenIndices/Offsets] --GM 标量读--> b_i / s_i / bos（V 流）
GM[KV]       --T.copy 逐行 [576]（GM→UB，AIV-vid 各 HALF 行）--> UB(kv_ub) --T.copy--> GM[workspace_kv[cid]]
                                                        ↑ bf16 UB→L1 直连被 static_assert 禁止（half-only），必须 GM 中转

per topk 块 t（NI 次）:
  [V] mask 链（fp32 min/max 算术，clamp≡min(max)）: mask/invalid [BI]
  [V] rows = clamp(idx + bos, 0, ΣS_kv-1) → UB 标量读逐行 gather → workspace_kv[cid, vid*HALF.., :]
  [C] T.copy(workspace_kv[cid, 0:BI, :], kv_l1 [BI,576])
  [C] for r in REPLICATE_H:  # head 分片，M=64
        T.gemm_v0(q_l1[r*64:(r+1)*64,:], kv_l1, acc_s_l0c [64,BI], transpose_B=True, init=True)  # K=576 合并单 GEMM
        T.copy(acc_s_l0c → workspace_s[cid, r*64:(r+1)*64, :])       # L0C→GM（fp32）
  [V]（AIV-vid 持有 head 分片 vid 的 softmax 状态）读 workspace_s[cid, vid*64:.., :] → acc_s_ub [64,BI] fp32
      ×sm_scale → mask 应用（x·mask − invalid·BIGC）→ m_i/alpha/exp/sumexp 在线更新 → p
      p →bf16 auto-cast→ workspace_p[cid, vid*64:.., :]
  [C] for r, for d_seg in {0:256, 256:512}:  # PV，N=512 分 2 段（512B 对齐）
        T.copy(workspace_p[cid, r 段] → p_l1 [64,BI])
        T.gemm_v0(p_l1, kv_l1[0:BI, d_seg], acc_o_l0c [64,256], init=True)
        T.copy(acc_o_l0c → workspace_o[cid, r*64:.., d_seg])          # L0C→GM（fp32）
  [V] 读 workspace_o[cid, vid 段, d_seg]（分段 [64,128]，auto-cast fp32）→ acc_o[h,:]×alpha + Δo 累加（UB fp32 跨块持续）

[V 尾部] acc_o /= sumexp（逐行 div）；Lse = ln(sumexp) + m_i
         → GM[Output[s_row, H0+vid*64 : H0+vid*64+64, :]]（bf16 auto-cast）
         → GM[Lse[s_row, H0+vid*64 : H0+vid*64+64]]
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer（用户明确指定，禁止更改）

### 2.2 选型理由

本算子为 CV 融合（GEMM + 向量 softmax + gather），与两个已验证 Developer 先例同构：

- `examples/sparse_flash_attention/example_sparse_flash_attn_gqa_pto_developer.py`（**pto 后端**，4-True AUTO pass_configs + 显式 ws 参数 + `(cid, vid)` + `T.gemm_v0` + `T.tile.*` 全链）
- `examples_experiment/dsa_sparse_finetune/indexer_bwd.py`（**ascendc 后端**，L275-280 同款 4-True pass_configs + 显式 ws 参数（命名含 "workspace" 触发 `TL_ASCEND_AUTO_CV_SYNC` 自动握手，docstring L138-141 已验证机制）+ persistent grid + varlen GM 标量读）

两先例共同覆盖了本算子全部所需能力（persistent grid、workspace 命名自动 CV 同步、GM 标量读、gemm_v0、T.tile 算术链、reduce_max/sum、auto-cast copy），且各是双后端之一的直接形态先例 → Developer 模式零新增机制风险。Expert 手动同步（sparse_flash_mla_kernel.py 形态）留作 Stage 3 性能不达标时的升级路径（该先例为 Expert + 固定 shape 契约，非本任务基线）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | 显式层级 API：`T.alloc_L1`（Cube 操作数）/ `T.alloc_L0C`（累加器）/ `T.alloc_ub`（Vector 链）——gqa pto / indexer_bwd 同款 Developer 形态（无 `T.Scope`） |
| 计算方式 | Cube：`T.gemm_v0`；Vector：`T.tile.fill/add/sub/mul/div/max/min/exp/ln/axpy` + `T.reduce_max/reduce_sum`；MTE：`T.copy` |
| 作用域 | 无 `T.Scope`，AUTO_CV_COMBINE 自动分离 C/V 指令 |
| 同步方式 | 全自动（4 个 AUTO pass，零手动 flag）；CV 中转靠 buffer 名含 "workspace" 由 pass 自动配对插 flag（indexer_bwd 已验证机制） |
| threads | 不显式指定；`(cid, vid)` 启动形式 → MIX_AIC_1_2（1 AIC + 2 AIV / block，indexer_bwd rev4 先例） |

---

## 3. API 映射设计

### 3.1 公式拆解（per task = (s_row, g)，Hseg = REPLICATE_H × 64 个 head 一体处理）

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 0 | `TokenIndices[s_row, 0/1]`、`Offsets[b_i]` | varlen 标量解码（GM 标量读） |
| 1 | `Q_seg = Q[s_row, H0:H0+Hseg, :]` | 连续行段 GM→L1 直拷（每 task 一次，[Hseg, 576]） |
| 2 | `mask/invalid = f(idx, s_i)` | mask = 1{idx≤s_i}·1{idx≥0}；invalid = max(1{idx>s_i}, 1{idx<0})（fp32 算术链） |
| 3 | `rows = clamp(idx + bos, 0, ΣS_kv−1)` | gather 寻址保护（仅保护，贡献由 mask 置 0） |
| 4 | `KV_blk = KV[rows, g, :]` | 逐行 gather GM→UB→GM workspace_kv（bf16 UB→L1 直连被平台禁止） |
| 5 | `S = QK^T`（K=576 合并） | gemm_v0，M=64/head 分片，transpose_B=True，init=True；fp32 L0C |
| 6 | `S̃ = S×sm_scale`；`S̃_m = S̃·mask − invalid·BIGC` | V 侧缩放 + mask（等价主仓 gemm 前预填 -inf：BIGC=1e30 有限大负偏移，exp 下溢为 0，语义推导见 §3.4 注 A） |
| 7 | `m_i = max(m_i, rowmax(S̃_m))`；`α = exp(m_prev − m_i)`；`p = exp(S̃_m − m_i)` | 在线 softmax 状态更新（m_i 缩放域，见 §3.4 注 B） |
| 8 | `sumexp = sumexp·α + Σ_j p`；`acc_o = acc_o·α` | sumexp/acc_o 跨块持续（UB fp32） |
| 9 | `P = bf16(p)` → PV | p 写 workspace_p（bf16，主仓 S_shared 同为 bf16 粒度）；`O_blk = P × V`（N=512 分 2×256 段） |
| 10 | `acc_o += O_blk` | V 侧 fp32 累加（读 workspace_o） |
| 11 | `O = acc_o / sumexp`；`Lse = ln(sumexp) + m_i` | 尾部归一化 + logsumexp |

### 3.2 TileLang API 映射（GPU → Ascend，逐行附双后端证据）

| 步骤 | GPU 版（ref_gpu 行号） | Ascend API | 双后端证据 |
|------|------------------------|-----------|-----------|
| Kernel 声明 | `T.Kernel(seq_len*RH, kv_group, threads=128)`（L86，二维 + threads=128） | `T.Kernel(grid_size=20, is_npu=True) as (cid, vid)` persistent + striped 任务轮询 `ntask = T.max(0, T.ceildiv(total−cid, grid))` | pto: gqa_pto L82；ascendc: indexer_bwd L358 / reducesum L218/261 |
| 动态 shape | `T.dynamic("batch_plus_one"/"seq_len")`（L43-44） | `T.symbolic("seq_len"/"seq_len_kv"/"batch_plus_one")` | pto: gqa_pto L59-62；ascendc: reducesum L111-113 |
| 输出声明 | `T.empty(...)`（L83-84） | kernel 参数全显式传入（host `torch.zeros` 预分配，含 Lse） | pto: gqa_pto 显式 Output 参数；ascendc: reducesum L214 |
| GM 标量读（varlen） | `TokenIndices[bx,0]`、`Offsets[b]`（L110-111） | 同语法（TIR BufferLoad 标量） | ascendc: reducesum L307-309 / indexer_bwd L798；pto: codegen_ascend_pto.cc L802-812（BufferLoad→GetValue 已实现） |
| Q 装载 | `T.copy(Q[bos+s_i, H0:H1, :D], Q_shared)`（L119） | `T.copy(Q[s_row, H0:H0+Hseg, :], q_l1)`（连续行段 GM→L1 直连） | pto: gqa_pto L127；ascendc: gqa_sink fwd L160 |
| KV gather | `KV_shared[bi,d] = KV[bos+idx, g, d]`（L127，T.Parallel 标量寻址） | 两段：`T.copy(KV[kv_row, g_i, :], kv_ub[576])`（GM→UB）+ `T.copy(kv_ub, workspace_kv[cid, row, :])`（UB→GM）；逐行 `for j in range(HALF)`，kv_row 为 rows_ub UB 标量读 | pto: gqa_pto L147-148（同款两段 gather）；ascendc: reducesum L350-353 |
| mask 构造 | `mask[bi] = (idx<=max_kv)&(idx!=-1)`（L124，T.Parallel） | fp32 算术链（**不依赖 compare/select/clamp/broadcast**）：`m1 = min(max((s_i+1)−idx, 0), 1)`（neg=mul(idx,−1) + add(T.float32(s_i+1)) + min/max）；`m2 = min(max(idx+1, 0), 1)`；`mask = m1·m2`；`invalid = max(min(max(idx−s_i,0),1), min(max(−idx,0),1))`；idx 经 `T.copy` int32→fp32 auto-cast | mul/add/sub/min/max：ascend_tile.py L1089-1190（sub 标量仅右操作数，故 (s_i+1)−idx 用 mul(−1)+add）；pto.py L396-457 + gqa_pto L156-186 全链先例 |
| 寻址 clamp | —（主仓靠 mask 丢弃） | `rows = min(max(idx_f + T.float32(bos), 0), ΣS_kv−1)`（fp32 域 min/max 标量）→ `T.copy` auto-cast 回 int32 rows_ub | 同上（reducesum L346-347 clamp 同义，min/max 替代规避 pto 风险） |
| QK GEMM | 两个 `T.gemm`（L133/140，主 512 + tail 64 累加） | **合并单 GEMM**：`T.gemm_v0(q_l1[r 段], kv_l1 [BI,576], acc_s_l0c, transpose_B=True, init=True)`（K=576，zN fractal K%16==0） | 双实现：ascend.py:418 / pto.py:227；K=576 合并：reducesum L288（实测验证，替代两段省一次调用）；先例 gqa_pto L134 |
| L0C→V 中转 | `T.copy(acc_s, S_shared)` 共享内存直连（L159，GPU 无 CV 分离） | `T.copy(acc_s_l0c, workspace_s[cid, r 段, :])` + V 侧 `T.copy(workspace_s[cid, vid 段, :], acc_s_ub_)`（fp32） | pto: gqa_pto L136/154；ascendc: indexer_bwd ws 机制（bf16 场景 GM 中转为平台强制，§8.2） |
| softmax 缩放 | `acc_s[h,bi]·sm_scale − m_i[h]·sm_scale`（L152） | `T.tile.axpy(acc_s_ub, acc_s_ub_, sm_scale)`（dst += s·src，gqa_sink fwd L176 同款）或 mul 标量 | ascend_tile.py:1441 / pto.py:551 |
| mask 应用 | gemm 前预填 `if_then_else(mask, 0, −inf)`（L132） | gemm 后 V 侧：`x_p = x·mask − invalid·BIGC`（mul + mul + sub，BIGC=1e30；§3.4 注 A 等价性推导） | mul/sub 双后端（见上）；ascendc 备选 `T.tile.select(…, −T.infinity, "VSEL_TENSOR_SCALAR_MODE")`（gqa_sink L188-194 先例，pto 未验证 → 非主路径） |
| m_i 更新 | `T.reduce_max(acc_s, m_i, clear=False)`（L148） | `T.reduce_max(acc_s_ub, m_blk, dim=-1)` + `T.tile.max(m_i, m_i, m_blk)`（clear=True 覆盖再 max 合并，gqa_pto L213/216 同款） | reduce.py:42（统一前端）；pto: gqa_pto L213；ascendc: gqa_sink L196 |
| alpha / p | `T.exp((m_prev−m_i)·s)`、`T.exp(acc_s·s − m_i·s)`（L150/152） | `T.tile.sub(m_prev, m_prev, m_i)` + `T.tile.exp`；p = `exp(S̃_m − m_i)`（**m_i 缩放域**，无需再乘 s） | pto: gqa_pto L164-171；ascendc: gqa_sink L198-202 |
| 逐行行内减 m_i | `T.Parallel(H, BI)` 2D（L151） | `for h in range(v_block): T.tile.sub(acc_s_ub[h,:], acc_s_ub[h,:], m_i[h])`（行切片 + BufferLoad 标量，规避 broadcast 依赖） | pto: gqa_pto L169（逐行同款）；broadcast 本身双后端可用（ascend_tile L2451 文档明示 PTO 路径），留作等价优化 |
| sumexp 更新 | `T.reduce_sum(acc_s, sumexp_i)` + Parallel（L153-155） | `T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)` + `T.tile.mul(sumexp, sumexp, alpha)` + `T.tile.add` | pto: gqa_pto L173-177 |
| acc_o rescale | `acc_o[h,d] *= alpha[h]`（L156-157） | `for h: T.tile.mul(acc_o[h,:], acc_o[h,:], alpha[h])` | mul + 逐行（同上） |
| p cast + 回传 | `T.copy(acc_s, S_shared)` fp32→bf16（L159） | `T.copy(acc_s_ub, p_bf16_ub)`（auto-cast fp32→bf16）+ `T.copy(p_bf16_ub, workspace_p[cid, vid 段, :])` | pto: gqa_pto L179-181 |
| PV GEMM | `T.gemm(S_shared, KV_shared, acc_o)`（L160，N=512 整段） | N=512 分 2×256 段：`T.gemm_v0(p_l1, kv_l1[0:BI, d_seg], acc_o_l0c [64,256], init=True)`（L1 列切片，段起始 0/512B 均满足 B 512B 对齐，T.gemm_v0.md §2.4 约束 6） | gemm BufferRegion：T.gemm_v0.md §2.1（"或其切片 BufferRegion"）；行切片 A：同文档 |
| acc_o 累加 | fragment 片上累加（GPU） | `T.copy(workspace_o[cid, vid 段, d_seg], o_seg_ub)`（bf16/fp32 auto-cast）+ `T.tile.add(acc_o, acc_o, o_seg_ub)` | pto: gqa_pto L183-187 |
| 归一化 / Lse | `acc_o /= sumexp`；`log(sumexp)+m_i·s`（L164-166） | 逐行 `T.tile.div(acc_o[h,:], acc_o[h,:], sumexp[h])`；`T.tile.ln(sumexp)` + `T.tile.add(sumexp, sumexp, m_i)` | pto: gqa_pto L189-190；ln: ascend_tile:1326 / pto.py:503 / gqa_sink L233 |
| 输出写回 | `T.copy(acc_o, Output[bos+s_i, H0:H1, :])`（L168-169） | `T.copy(acc_o_bf16, Output[s_row, H0+vid·64 : …, :])`（fp32→bf16 auto-cast）+ `T.copy(lse_ub, Lse[s_row, …])` | pto: gqa_pto L192-193 |
| 流水线 | `T.Pipelined(NI, num_stages=2)`（L122） | `T.serial(NI)`（NI 编译期常量；动态 task 外层 `T.serial(ntask)` 运行时边界——reducesum L265 已验证形态；AUTO_CV_SYNC 的自动 flag 配对替代手动流水） | ascend-constraints §1（动态边界禁 Pipelined）；reducesum L261 |
| threads | `threads=128`（L86） | 不指定（(cid,vid) 启动 → MIX_AIC_1_2） | indexer_bwd rev4 L358-367 |

### 3.3 计算伪代码

```python
GRID = 20                          # 编译期常量（host 查 cube_core_num，910B3=20；reducesum L727）
BI = 128                           # topk 方向分块（topk % BI == 0）
NI = topk // BI                    # 块数（编译期常量）
HALF = BI // 2                     # 每 AIV gather 行数
Hseg = REPLICATE_H * 64            # 每 task head 数（H=128/HKV=1 → 128；RH=1 时 = padded_H）
v_block = Hseg // 2                # 每 AIV softmax 行数（= RH×64/2 = 64 = 恰一个 head 分片）
DV_SEG = 256                       # PV 的 D 分段（512B 对齐）
BIGC = 1e30                        # mask 无效项负偏移（有限大，见 §3.4 注 A）

@tilelang.jit(pass_configs=4_True_Developer, target=target)   # §4.7；target 参数化 "ascendc"/"pto"
def tl_sparse_mla_fwd(heads, dim, tail_dim, topk, kv_group=1, sm_scale=None,
                      block_I=128, grid_size=20):
    # ---- 编译期 assert（§3.5.1 支持域，主仓 L34-36/71 原样继承 + Ascend 侧新增）----
    seq_len, seq_len_kv, batch_plus_one = T.symbolic(...), T.symbolic(...), T.symbolic(...)

    @T.prim_func
    def main(Q, KV, Indices, Offsets, TokenIndices,          # 输入（§4.1）
             Output, Lse,                                    # 输出（host zeros 预分配）
             workspace_kv, workspace_s, workspace_p, workspace_o):   # CV 中转（§4.3，显式参数）
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):
            # ---- C 侧（L1/L0C）----
            q_l1    = T.alloc_L1([Hseg, 576], dtype)         # 147KB
            kv_l1   = T.alloc_L1([BI, 576], dtype)           # 147KB
            p_l1    = T.alloc_L1([64, BI], dtype)            # 16KB
            acc_s_l0c = T.alloc_L0C([64, BI], accum_dtype)   # 32KB
            acc_o_l0c = T.alloc_L0C([64, DV_SEG], accum_dtype)  # 64KB
            # ---- V 侧（UB，每 AIV v_block=64 行 = head 分片 vid）----
            acc_o     = T.alloc_ub([v_block, 512], accum_dtype)   # 128KB 跨块持续
            acc_s_ub  = T.alloc_ub([v_block, BI], accum_dtype)    # 32KB（p 产出后复用为 o_seg 读入）
            p_bf16    = T.alloc_ub([v_block, BI], dtype)          # 16KB
            m_i, m_prev, sumexp, sumexp_i, alpha = T.alloc_ub([v_block], fp32) ×5
            indices_ub = T.alloc_ub([BI], int32); rows_f = T.alloc_ub([BI], fp32)
            idx_f, m1, m2, mask, invalid, t_ub = T.alloc_ub([BI], fp32) ×6   # mask 链
            kv_ub = T.alloc_ub([576], dtype)                       # gather 单行 staging

            total = seq_len * kv_group
            ntask = T.max(0, T.ceildiv(total - cid, grid_size))
            # V 侧状态 init（每 task 重置）
            for k in T.serial(ntask):
                pid = k * grid_size + cid
                s_row = pid % seq_len; g_i = pid // seq_len     # 行连续优先（同 g 行相邻）
                b_i, s_i = TokenIndices[s_row, 0], TokenIndices[s_row, 1]
                bos = Offsets[b_i]
                H0 = g_i * padded_H                              # Q 段起点（Hseg 连续 head）
                # [C] Q 装载（每 task 一次，GM→L1 直连）
                T.copy(Q[s_row, H0 : H0 + Hseg, :], q_l1)
                # [V] 状态 init
                T.tile.fill(acc_o, 0.0); T.tile.fill(sumexp, 0.0); T.tile.fill(m_i, -(2.0**30))
                for i in T.serial(NI):
                    # [V] indices + mask 链 + rows（fp32 算术，§3.2 行 6/7）
                    T.copy(Indices[s_row, g_i, i*BI : (i+1)*BI], indices_ub)
                    T.copy(indices_ub, idx_f)                     # int32→fp32 auto-cast
                    ...  # m1/m2/mask/invalid/rows_f = min/max/add/mul 链（§3.2）
                    T.copy(rows_f, rows_int)                      # fp32→int32（整数值精确表示）
                    # [V] gather：AIV-vid 搬 vid*HALF 起的 HALF 行
                    for j in range(HALF):
                        kv_row = rows_int[vid * HALF + j]
                        T.copy(KV[kv_row, g_i, :], kv_ub)
                        T.copy(kv_ub, workspace_kv[cid, vid*HALF + j, :])
                    # [C] KV 块进 L1 + QK（head 分片 r 串行，M=64）
                    T.copy(workspace_kv[cid, 0:BI, :], kv_l1)
                    for r in range(REPLICATE_H):
                        T.gemm_v0(q_l1[r*64:(r+1)*64, :], kv_l1, acc_s_l0c,
                                  transpose_B=True, init=True)    # K=576 合并
                        T.copy(acc_s_l0c, workspace_s[cid, r*64:(r+1)*64, :])
                    # [V] AIV-vid 持分片 vid 的在线 softmax（§3.2 步骤 6-8）
                    T.copy(workspace_s[cid, vid*64:(vid+1)*64, :], acc_s_ub_)
                    T.tile.mul(acc_s_ub_, acc_s_ub_, sm_scale)
                    ...  # mask 应用 → m_i/alpha/p/sumexp/acc_o rescale（逐行循环）
                    T.copy(acc_s_ub_, p_bf16)                     # fp32→bf16
                    T.copy(p_bf16, workspace_p[cid, vid*64:(vid+1)*64, :])
                    # [C] PV（head 分片 × D 分段）
                    for r in range(REPLICATE_H):
                        T.copy(workspace_p[cid, r*64:(r+1)*64, :], p_l1)
                        for d in range(2):
                            T.gemm_v0(p_l1, kv_l1[0:BI, d*256:(d+1)*256], acc_o_l0c, init=True)
                            T.copy(acc_o_l0c, workspace_o[cid, r*64:(r+1)*64, d*256:(d+1)*256])
                    # [V] acc_o 累加（4×[64,128] 段，auto-cast 读入复用 acc_s_ub 槽）
                    for d in range(4):
                        T.copy(workspace_o[cid, vid*64:(vid+1)*64, d*128:(d+1)*128], o_seg)
                        T.tile.add(acc_o[:, d*128:(d+1)*128], acc_o[:, d*128:(d+1)*128], o_seg)
                # [V 尾部] 归一化 + Lse + 写回
                for h in range(v_block):
                    T.tile.div(acc_o[h,:], acc_o[h,:], sumexp[h])
                T.tile.ln(sumexp, sumexp); T.tile.add(sumexp, sumexp, m_i)
                T.copy(acc_o, acc_o_bf16)                          # fp32→bf16
                T.copy(acc_o_bf16, Output[s_row, H0+vid*64 : H0+vid*64+64, :])
                T.copy(sumexp, Lse[s_row, H0+vid*64 : H0+vid*64+64])
```

> 伪代码中 `acc_s_ub_`/`o_seg`/`rows_int` 等为 §4.3 中间缓冲别名；`...` 展开段即 §3.2 对应行的逐条 tile 指令。实现以 §3.2 表为准。

### 3.4 API 可行性确认

**注 A（mask 等价性推导，替代主仓 gemm 前预填 −inf）**：主仓 `acc_s = mask ? qk : −inf` 后 gemm 累加（−inf + 有限 = −inf），max/exp 语义为 IEEE −inf。本设计 gemm 输出真实值（clamp 行合法，值有限），V 侧以 `x_p = x·mask − invalid·BIGC`（BIGC = 1e30 有限大）构造等效负大值：
- 有效项（mask=1, invalid=0）：x_p = x，与主仓一致；
- 无效项（mask=0, invalid=1）：x_p = x − 1e30 ≈ −1e30（x 有界）。
- m_i 更新取 x_p 的行 max：全无效块 rowmax = −1e30 < m_i 初值 −2^30 → m_i 不更新（与主仓 −inf 不更新等价）；
- p = exp(x_p − m_i)：无效项 ≈ exp(−1e30 − m_i) 双精度下溢为 **0**（fp32 exp 输入 < −87.3 即下溢；−1e30 量级远超），与主仓 exp(−inf) = 0 等价；
- 全无效块且 m_i 仍为初值 −2^30 时：p = exp(−1e30 + 2^30×sm_scale) = exp(−1e30) = 0 ✓（无主仓「有限替代值 p=exp(0)=1」的退化，无 NaN：BIGC 有限，x − 1e30 不产生 ±inf，0×inf 不出现）。
- **约束**：该等价要求 |x| ≪ 1e30 且 (m_i − x_p) 不越过 fp32 exp 上溢界（x_p ≤ m_i 保证 ≤ 0，恒不上溢）✓。

**注 B（m_i 缩放域）**：主仓 m_i 为未缩放 score 的 max（L148），p = exp(acc_s·s − m_i·s)。本设计沿用 gqa/gqa_sink 先例：先 axpy(mul) 乘 sm_scale 再 reduce_max，m_i 即缩放域 max，p = exp(S̃ − m_i)、Lse = ln(sumexp) + m_i。代数等价（乘法分配律），少 2 次 ×s 指令；浮点差异 ≤ 1 ulp 量级，远小于 bf16 容差。

**注 C（PV 的 N 分段与对齐）**：kv_l1 [BI, 576] 上取列切片 [0:BI, 0:256] / [0:BI, 256:512] 作 gemm_v0 的 B：段起始字节偏移 0 / 512，均满足 B 起始 512B 对齐（T.gemm_v0.md §2.4 约束 5/6）；N=256、M=64 均满足 16 分形。**若列切片路径在任一后端实测失败**（文档支持但无切片先例，列为本设计首要验证项 R1），回退方案：workspace_kv 分立双段布局（ws_main [grid, BI, 512] + ws_tail [grid, BI, 64]，即 reducesum DESIGN 预案的"SFA 双 ws 分立"），C 侧整块拷入 kv_main_l1 [BI,512] / kv_tail_l1 [BI,64]，QK 拆 K=512（init=True）+ K=64（init=False）两段累加（与主仓 GPU 两段 T.gemm 同构）。

**注 D（V 侧 N=512 段宽）**：acc_o 的 UB 段读按 4×128 列切（workspace_o 读 [64,128]），与 acc_s_ub 复用槽位对齐；gemm 写侧仍按 2×256 段（L0C 64KB 预算），段边界 128 整除 ✓。

API 清单及来源（全部有双后端证据，**无一项凭记忆**）：

| API | ascendc 证据 | pto 证据 | 验证状态 |
|-----|-------------|----------|---------|
| `T.gemm_v0`（bf16 A/B + fp32 C / transpose_B / init / BufferRegion 切片） | ascend.py:418；reducesum L288（K=576）；gqa_sink L169/215 | pto.py:227；gqa_pto L134/140 | 先例已验证（切片形态为 R1 首验项） |
| `T.copy`（GM→L1 直连 / GM↔UB / UB→GM / L0C→GM / auto-cast / 4D 内 3D 切片最内维连续） | copy_op.py:257；gqa_sink L160/170；reducesum L284 | gqa_pto L127-154 | 先例已验证 |
| `T.tile.fill / add / sub / mul / div / max / min`（标量右操作数） | ascend_tile.py:352/1089-1190（sub 标量仅右侧 → mul(−1)+add 规避） | pto.py:307/396-457；gqa_pto 全链 | 先例已验证 |
| `T.tile.exp / ln / axpy` | ascend_tile.py:1255/1326/1441；gqa_sink L176/202/233 | pto.py:499/503/551 | 先例已验证 |
| `T.reduce_max / reduce_sum`（fp32 2D dim=-1，整 buffer 非切片输入） | reduce.py:42/79；gqa_sink L196/203 | pto.py:573/577；gqa_pto L213/227 | 先例已验证（fp32 无后端限制；fp16 列归约仅 pto 的限制不涉及） |
| `T.tile.broadcast`（如启用优化） | ascend_tile.py:2451 | 同文档"PTO broadcast … omits the operand" | 可用；主路径以逐行循环规避依赖 |
| `T.Kernel(grid, is_npu=True) as (cid, vid)` + MIX_AIC_1_2 | indexer_bwd L358-367 | gqa_pto L82 | 先例已验证 |
| `T.symbolic` 动态维 | reducesum L111-113 | gqa_pto L59-62 | 先例已验证 |
| GM 标量读（BufferLoad） | reducesum L307-309 | codegen_ascend_pto.cc L802-812 | **pto 无端到端先例（R2 风险项），codegen 已实现** |
| workspace 命名 + AUTO_CV_SYNC 自动配对 | indexer_bwd L138-141（已验证机制） | gqa_pto（4-True + 显式 ws 参数形态） | 先例已验证 |
| **规避项**：`T.tile.compare/select`（pto.py 无定义）、`T.tile.clamp`（用 min(max) 等价替代）、`arith_progression/wholereduce/block_reduce`（pto 不支持） | — | — | 主路径零依赖 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 是否涉及 | 处理方案 |
|------|---------|----------|
| 不支持三维 Kernel | **Yes**（GPU 源 L86 二维 `T.Kernel(seq_len*RH, kv_group)`） | 一维 persistent `T.Kernel(20)` + striped 任务轮询（`ntask = T.max(0, T.ceildiv(total−cid, grid))`，reducesum L261 已验证）；task = (s_row, g) 二维语义折叠进 pid 分解 |
| threads 参数限制（仅 1/2） | **Yes**（GPU 源 threads=128） | 不指定 threads，`(cid, vid)` 启动 → MIX_AIC_1_2（indexer_bwd rev4） |
| 动态循环边界不支持 | **Yes**（外层 task 数运行时） | persistent `T.serial(ntask)` 运行时边界——reducesum L265 / indexer_bwd L426 **已验证形态**；块循环 NI、head 分片循环 RH、D 段循环全为编译期常量 |
| 流水线不支持动态边界 | Yes（GPU 源 T.Pipelined） | 不使用 T.Pipelined；`T.serial(NI)` + AUTO_CV_SYNC 自动 flag 配对（indexer_bwd 同构） |
| L0C 容量上限 128KB | **Yes** | 验算：acc_s [64,128] fp32 32KB + acc_o [64,256] fp32 64KB = **96KB < 128KB**（余 32KB；§5.3 完整表；BI=256 档会触 128KB 贴线 → 不入基线） |
| T.copy 列方向 strided 切片 | **No** | 所有 GM 切片：Q 行段（行连续）、Indices 行内连续段、KV 整行、ws 行段——最内维均连续；PV 的 L1 列切片走 gemm_v0（非 T.copy，注 C）；workspace_o 读 [cid, 行段, 列段] 为 3D buffer 最内维列段连续（行间 gap 合法模式） |
| bf16 UB→L1 直连 static_assert（half-only） | **Yes** | 全部 CV 中转走 GM workspace（gqa/indexer_bwd 模式，§8.2）；fp16 cast 直连（gqa_sink 消 ws 模式）因 bf16→fp16 溢出语义风险不入主路径，列 Stage 3 探索项 |
| bf16 标量算术不允许（CCE 限制） | **Yes** | 全标量链 fp32（mask 链经 int32→fp32 auto-cast；sm_scale/alpha/m_i 均 fp32）；bf16 仅作搬运/auto-cast dtype |
| GPU 专用 API | Yes | T.gemm→T.gemm_v0、T.alloc_shared/fragment→显式 alloc_L1/L0C/ub（§3.2 全表）、T.Pipelined→T.serial、T.infinity 预填→算术 mask（注 A）、T.dynamic→T.symbolic |

#### 3.5.2 参考实现差异说明（相对 ref_gpu/sparse_mla_fwd.py）

| 差异项 | GPU 版 | 本设计（Ascend） | 说明 |
|--------|--------|-----------------|------|
| Kernel 拓扑 | 二维 grid (seq_len×RH, kv_group)，8192 blocks | persistent grid=20 + striped 任务轮询，task=(s_row,g) 且 **RH 个 head 分片在 task 内串行共享 gather**（pair 模式） | 语义等价（head 分片只是循环维度重排）；gather 中转流量减半（§9.5） |
| task 粒度 | 每 (行, 分片) 独立 block，各自 gather | 每 (行, kv_group) 一 task，块内 r 循环共享同一 workspace_kv 块 | Indices[s_row,g,:] 对全部分片相同 → gather 天然可共享 |
| mask 时机 | gemm 前预填 −inf（L132） | gemm 后 V 侧算术 mask（注 A 等价） | gemm_v0 init=True 会清掉预填值，平台强制后置 |
| KV gather | T.Parallel 标量寻址直读 GM（L127） | 两段 T.copy 经 GM workspace（bf16 UB→L1 禁直连） | 平台强制；§9.5 流量已入预算 |
| QK 两段 GEMM | 主 512 + tail 64 累加（L133/140） | K=576 合并单 gemm（reducesum 验证） | 省一次 gemm 固定开销；fp32 累加顺序不变（K 顺序推进） |
| acc_o 累加 | fragment fp32 片上（L0C 等价） | L0C 单块输出 → GM ws → UB fp32 跨块累加（gqa 模式） | L0C 无行缩放原语，alpha rescale 必须在 V 侧 → 每块中转 |
| Lse 尾部 | `T.log`（L166） | `T.tile.ln`（gqa_sink L233 同款） | 同义 API |
| threads=128 | SIMT | (cid,vid) MIX_AIC_1_2 | NPU 限制 |

#### 3.5.3 本项目同类实现参考（同类引用，门禁项）

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/sparse_flash_attention/example_sparse_flash_attn_gqa_pto_developer.py` | **最高（pto 基线骨架）** | target="pto" + 4-True pass_configs（L11-16）+ (cid,vid)（L82）+ 显式 ws 参数（L77-80）+ gemm_v0 双向（L134/140）+ T.tile.* softmax 全链（L150-190）+ workspace 读写段切（L132-154）|
| `examples/sparse_flash_attention/example_sparse_flash_attn_gqa.py` | **最高（ascendc 姊妹，结构对照）** | 同构计算流的 ascendc 侧（ws1/ws2/ws3/ws4 四中转布局、V 侧 gather 行分摊、acc_o UB 累加）；其 T.Scope+手动 flag 为 Expert 形态，本设计以 indexer_bwd 的 AUTO 形态替代 |
| `examples_experiment/dsa_sparse_finetune/indexer_bwd.py` | **最高（ascendc 侧 Developer + varlen 先例）** | 4-True pass_configs（L275-280）+ persistent grid + varlen TokenIndices GM 标量读（L798）+ workspace 命名触发 AUTO_CV_SYNC（L138-141）+ _CachedLauncher host 开销优化（L170）+ bf16 组合支持域 |
| `examples/dsa_sparse_finetune/sparse_mla_topk_reducesum.py` | **高度相似（同系列 varlen 算子）** | 行寻址恒等式 s_row≡bos+s_i（L269）；mask 算术链 clamp≡min(max) + sub 标量左操作数规避（L626-631）；DDT=576 合并布局单 GEMM（L288）；支持域编译期 assert（L98-106）；persistent grid=20（L218）；ws_kv/ws_acc 显式参数（L215-216） |
| `examples/attention_sink/example_gqa_sink_bwd_bhsd/example_gqa_sink_bwd_bhsd.py` | 高（fwd 形态最近亲） | flashattn_fwd（L76-237）：在线 softmax + Lse 输出 + workspace 三中转 + axpy 缩放（L176）+ ln（L233）；select(−inf) mask 先例（L188-194，ascendc） |
| `examples/deepseek_v4/sparse_flash_mla/sparse_flash_mla_kernel.py` | 中（性能手段参考，非基线） | D=512 分 2×256 段（D2=D//2）、L1 KV ring 多槽、L0A/L0B 双缓冲——Stage 3 优化菜单；其为 Expert + 固定 shape 契约 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| Q | [seq_len(ΣS), heads(H), dim+tail_dim(576)] | bfloat16 | 动态维 ΣS；H=128（perf shape） |
| KV | [seq_len_kv(ΣS_kv), kv_group(G), 576] | bfloat16 | 同一 KV 空间被 varlen 各段引用（bos 偏移）；G=HKV=1 |
| Indices | [ΣS, G, topk] | int32 | 值域 {−1} ∪ [0, ΣS_kv)（越界填充任意），**任意分布，禁止假设规律** |
| Offsets | [batch_plus_one] | int32 | varlen cu_seqlens |
| TokenIndices | [ΣS, 2] | int32 | host 预计算 (batch_id, 行内位置)，CPU 构造 + 一次 H2D（reducesum prepare_token_indices 同款） |

### 4.2 输出张量（host `torch.zeros` 预分配后显式传入，无 out_idx）

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| Output | [ΣS, H, 512] | bfloat16 | 每 (s_row, head) 行独占写，无竞态 |
| Lse | [ΣS, H] | float32 | 每 (s_row, head) 独占写 |

### 4.3 中间缓冲区（perf shape：H=128/HKV=1 → REPLICATE_H=2, Hseg=128, v_block=64, BI=128）

| Buffer 名 | Shape | dtype | 存储层级 | 大小 | 用途 |
|-----------|-------|-------|----------|------|------|
| q_l1 | [128, 576] | bf16 | L1 | 147.5KB | Q 连续行段（两 head 分片拼接，GM→L1 直连） |
| kv_l1 | [128, 576] | bf16 | L1 | 147.5KB | gather 块（ws_kv→L1 整块） |
| p_l1 | [64, 128] | bf16 | L1 | 16KB | p 分片（PV 的 A） |
| acc_s_l0c | [64, 128] | fp32 | L0C | 32KB | QK 累加器（head 分片串行复用） |
| acc_o_l0c | [64, 256] | fp32 | L0C | 64KB | PV 段累加器（D 段串行复用） |
| acc_o | [64, 512] | fp32 | UB | **128KB** | 跨块 acc_o（AIV-vid 持有分片 vid） |
| acc_s_ub | [64, 128] | fp32 | UB | 32KB | score/p 工作槽（p 产出后复用为 o_seg 读入） |
| p_bf16 | [64, 128] | bf16 | UB | 16KB | p 写出前 cast 槽 |
| m_i / m_prev / sumexp / sumexp_i / alpha | [64] | fp32 | UB | 0.25KB×5 | softmax 状态 |
| indices_ub / rows_int | [128] | int32 | UB | 0.5KB×2 | 索引 / gather 行号 |
| idx_f / m1 / m2 / mask / invalid / t_ub / rows_f | [128] | fp32 | UB | 0.5KB×7 | mask 算术链 |
| kv_ub | [576] | bf16 | UB | 1.2KB | gather 单行 staging |
| misc（acc_o_bf16 等 cast 槽） | [64,128] | bf16 | UB | ~16KB | 输出 cast |

### 4.4 workspace（GM，显式 kernel 参数，命名含 "workspace"，host `torch.zeros` 分配，双后端统一传参约定）

| workspace | Shape | dtype | 大小 | 用途 |
|-----------|-------|-------|------|------|
| workspace_kv | [grid=20, BI=128, 576] | bf16 | 2.95MB | V→C gather 中转（**同 task 的 RH 个 head 分片共享**——pair 模式核心收益） |
| workspace_s | [20, Hseg=128, 128] | fp32 | 1.31MB | C→V score 中转 |
| workspace_p | [20, 128, 128] | bf16 | 0.66MB | V→C p 中转 |
| workspace_o | [20, 128, 512] | fp32 | 5.24MB | C→V PV 段中转（bf16 化为 Stage 3 优化档，§9.5） |

### 4.5 内存搬运路径（完整层级链）

```
GM[Q] ──T.copy──> L1(q_l1)                                  # 直连（连续行段）
GM[Indices] ──T.copy──> UB(indices_ub) ──auto-cast──> UB(idx_f)
GM[TokenIndices/Offsets] ──GM 标量读──> 标量 b_i/s_i/bos
GM[KV] ──T.copy──> UB(kv_ub) ──T.copy──> GM[workspace_kv] ──T.copy──> L1(kv_l1)
L1(q_l1) × L1(kv_l1) ──T.gemm_v0(transpose_B)──> L0C(acc_s_l0c) ──T.copy──> GM[workspace_s] ──T.copy──> UB(acc_s_ub)
UB 向量链（mask/softmax/p）──auto-cast──> GM[workspace_p] ──T.copy──> L1(p_l1)
L1(p_l1) × L1(kv_l1 列段) ──T.gemm_v0──> L0C(acc_o_l0c) ──T.copy──> GM[workspace_o] ──T.copy──> UB(o_seg) ──T.tile.add──> UB(acc_o)
UB(acc_o) ──auto-cast──> GM[Output]；UB(lse) ──T.copy──> GM[Lse]
```

跨级合法性：无 GM↔L0 直搬；UB↔L1 无直搬（经 GM workspace，bf16 平台强制）；全部 T.copy 为文档支持路径（api-kernel-memory.md §3 路径表）。

### 4.6 UB / L1 / L0C 内存预算（910B3：UB 192KB / L1 512KB / L0C 128KB 每核）

| 层级 | 主要项 | 合计 | 上限 | 占用 | 结论 |
|------|--------|------|------|------|------|
| L0C | acc_s 32KB + acc_o 64KB | **96KB** | 128KB | 75% | ✓（余 32KB） |
| L1 | q_l1 147.5 + kv_l1 147.5 + p_l1 16 | **311KB** | 512KB | 61% | ✓（余 201KB，可容 ws 双槽流水升级） |
| UB | acc_o 128 + acc_s_ub(复用 o_seg) 32 + p_bf16 16 + misc ~13 | **~189KB** | 192KB | **98%** | ⚠️ 贴线（§10.2 R4 缓解：mask 链 7×0.5KB 中 t_ub/rows_f 等槽复用、acc_o_bf16 复用 p_bf16 死槽、MEMORY_PLANNING liveness 复用；实测超限则 ws_o 读入改 8×[64,64] 段） |

### 4.7 动态轴定义与 JIT 配置

| 动态轴 | 声明 | 运行时范围 |
|--------|------|-----------|
| seq_len（ΣS） | T.symbolic | ≥ 1（task 数 = ΣS×G，persistent 轮询消化） |
| seq_len_kv（ΣS_kv） | T.symbolic | ≥ 1（gather clamp 上界） |
| batch_plus_one | T.symbolic | ≥ 2 |

```python
_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # 核内自动同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # 内存规划（UB 贴线时 liveness 复用关键）
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # 自动 C/V 指令分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,     # workspace 命名自动配对插 flag
}
@tilelang.jit(pass_configs=_pass_configs, target=target)     # target: "ascendc" | "pto"（builder 参数）
```

一次编译覆盖任意 ΣS/ΣS_kv/batch（symbolic 动态维，reducesum/gqa 同款）；H/DQK/DV/topk/G 为编译期常量（builder 参数 → `_BUILD_CACHE` 编译缓存键，reducesum L809 先例）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 混合（Cube GEMM + Vector softmax 链 + DMA gather）

### 5.2 Block 划分

```python
BI = 128        # topk 方向分块：NI = topk/BI = 8（1024/128）
                #   依据：① L0C 96KB<128KB（BI=256 → acc_s [64,256] 64KB + acc_o 64KB = 128KB 贴线，弃）
                #          ② L1 311KB<512KB（BI=256 → kv_l1 295KB，q+p 合计 474KB 贴线，弃）
                #          ③ ws_o 中转频率 ∝ NI（BI 越大越省，§9.5 敏感性）
Hseg = 128      # 每 task head 数 = REPLICATE_H×64（H=128/HKV=1 → RH=2 分片共享 gather）
                #   依据：Indices[s_row,g,:] 对全部分片相同 → gather/中转减半（pair 模式，§1.5）
M_gemm = 64     # QK/PV 的 M（head 分片粒度，主仓 H_per_block=64 语义 1:1）
DV_SEG = 256    # PV 的 N 分段（kv_l1 列切片 512B 对齐约束，注 C；L0C [64,256]=64KB）
grid_size = 20  # persistent（= cube 核数，host 查询 get_cube_core_num，reducesum L727）
```

### 5.3 约束分析

- **分形对齐**：M=64 ≥ 16 ✓；N（QK 的 BI=128 / PV 的 256、512）≥ 16 ✓；K（QK 576 = 36×16 / PV 128）% 16 == 0 ✓（T.gemm_v0.md §2.3.4 bf16 分形 16×16）
- **L0C**：96KB < 128KB ✓（§4.6）
- **L1**：311KB < 512KB ✓
- **UB**：~189KB / 192KB 贴线 ⚠️（§4.6 缓解清单）
- **512B 对齐**：gemm B 操作数列切片段起始 0/512B ✓（注 C）；q_l1/p_l1 行切片起始 64×576×2B、64×128×2B 均 512B 倍数 ✓；L0C C 起始 256 元素对齐（64×128 / 64×256 均为 256 倍数）✓
- **GEMM 非整除**：**无**——M/N/K 全静态且被 16 整除；topk % BI == 0 为编译期 assert（主仓 L37 原样）；task 粒度 = 行（无 M 方向尾块）；输出行段 [64, 512] 静态完整（head 维全量覆盖，无 N 方向尾块）。**本算子无非整除尾块场景**（与 grouped_gemm 类尾块竞态无涉）

### 5.4 注意事项

- **varlen 段长与 topk 无耦合**：语义只依赖 Indices 值域与 s_i（主仓测试构造保证段长 ≥ topk，但 kernel 不依赖该假设——无效槽由 mask 置 0）
- **gather 行随机性**：rows 已 clamp 到 [0, ΣS_kv−1]，任何索引值均不越界读；-1 → clamp 至行 0（贡献被 mask 置 0，reducesum §5.4 同款）
- **重复索引**：不去重、不检测——重复行被多次 gather、多次计贡献（主仓 kernel 语义，§1.4 #1）

### 5.5 数据搬运性能可行性（[REORDER-COST]，gather 主路径）

```text
[REORDER-COST]
path: generic gather（唯一路径，无分派）
gm_passes: KV 读 4.83GB（4096 task × topk×576×2B，RH 分片共享后单份）+ ws_kv 写读 9.66GB
           + ws_s 4.3GB + ws_p 2.1GB + ws_o 16.8GB（fp32 基线；bf16 档 8.4GB）+ Q/Out/Lse ~1.2GB
dma_transactions: ~行级 4.2M 条 gather T.copy（GM→UB 1152B/条 + UB→GM 1152B/条）
                 + 块级 ~328K 条（ws 整块/行段）≈ 平均 1.1KB/条（gather 段）~100KB/条（块段）
gm_scalar_accesses: 每 task 3 次 GM 标量读（TokenIndices×2 + Offsets×1）= 12K 次（无关数据量）
address_div_mod_per_element: 0（行号经 rows_ub 向量化预计算，每行 1 次 UB 标量读，reducesum #16 同款）
active_cores / serial_tasks_per_core: 20 AIC + 40 AIV / 205 task/核（4096 task striped）
largest_case_timeout_gate: l0_perf_shape（B=1,S=4096,topk=1024）单 case 600s（§9.4）
result: pass（预算 §9.5：V 侧 21.5GB/s/AIV < 26GB/s 实测、C 侧 34.4GB/s/AIC < 50-60GB/s 实测；
        gather 为行级 DMA 但每行 1152B 连续（≥ 32B 聚合阈值），无逐元素 strided GM 访问）
```

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| task（外层） | 运行时边界串行 | `T.serial(ntask)`（ntask = T.max(0, T.ceildiv(total−cid, grid))） | persistent + striped；动态边界已验证形态（reducesum L261/265） |
| topk 块 | 编译期常量串行 | `T.serial(NI)` | 流式在线 softmax 顺序依赖（m_i/alpha 链），不可并行化 |
| head 分片 | 编译期常量串行 | `for r in range(REPLICATE_H)`（Python 展开） | 共享同一 kv_l1/ws_kv 块（pair 模式） |
| D 段（PV） | 编译期常量串行 | `for d in range(2)`（gemm 侧）/ `range(4)`（V 读侧） | L0C 64KB 预算 + 512B 对齐 |
| gather 行 | 编译期常量串行 | `for j in range(HALF)`（每 AIV 64 行） | 随机行寻址（UB 标量读 + 两段 copy，gqa L190 同款） |
| 逐 head 行（softmax 内） | 编译期常量串行 | `for h in range(v_block)` | 行内减 m_i / rescale / div（BufferLoad 标量，gqa_pto L169 同款） |
| 元素级 | 向量化 | T.tile.* 全链 | SIMD（无 T.Parallel 标量表达式） |

### 6.2 循环伪代码

见 §3.3。

### 6.3 流水线优化

不使用 `T.Pipelined`（task 边界运行时值）。C/V 重叠由 `TL_ASCEND_AUTO_CV_COMBINE` 自动指令分核 + `TL_ASCEND_AUTO_CV_SYNC` 的 workspace 读写配对 flag 提供（indexer_bwd 同构验证：C 侧 gemm 与 V 侧 softmax 经 ws 乒乓自然重叠）。**进一步流水（Stage 3 菜单）**：workspace_kv 双槽（DEPTH=2 跨块预取，reducesum L144 DEPTH 机制）、gather 批量 staging 多槽（reducesum #15v2 [SIDES, HALF, DDT]）、q_l1 双缓冲。

### 6.4 尾块处理

**无尾块**（§5.3）：topk % BI 编译期 assert；head/D 维静态全量；task 粒度 = 行。persistent 轮询的末核任务不均（4096 % 20 = 16 → 最多 206 vs 204 task/核，偏差 1%）已入 §9.5 预算。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 全自动（Developer，4-True pass_configs，零手动 flag / 零 T.Scope）

### 7.2 同步点说明（由 pass 自动插入，语义点位记录）

| 点位 | 机制 | 说明 |
|------|------|------|
| V gather 写 ws_kv → C 读 ws_kv | AUTO_CV_SYNC（命名配对） | indexer_bwd L138-141 已验证：pass 按 buffer 名含 "workspace" 配对 GM 写/读自动插 cross flag |
| C 写 ws_s → V 读；V 写 ws_p → C 读；C 写 ws_o → V 读 | 同上 | 同一批 ws 四路乒乓 |
| kernel 内核内依赖（UB 读写序） | AUTO_SYNC | gqa_pto 全链先例 |
| 内存复用 liveness | MEMORY_PLANNING | UB 贴线时关键（§4.6） |

### 7.3 pass_configs 配置

见 §4.7（4-True；与 gqa_pto L11-16、indexer_bwd L275-280 逐项一致）。

---

## 8. CV 融合设计

### 8.1 融合算子判定

**判定结果**: 是（GEMM + 在线 softmax 向量链 + rescale 后处理，flash-attention 型）

### 8.2 CV 交互设计（Developer 复杂场景回退：GM workspace 中转）

> 模板 §8.2 默认指引（threads=2 片上直连消除 workspace）适用于 fp16 场景（gqa_sink bwd 先例：L0C→UB / UB→L1 direct，"All 6 GM workspace buffers eliminated"）。**本算子为 bf16：`copy_ub_to_l1` 模板 static_assert half-only（src/tl_templates/ascend/common.h:358-361），UB→L1 片上直连不可用** → GM workspace 中转是 bf16 下的平台强制路径（indexer_bwd DESIGN 偏差 #1 同结论），按回退写法设计。

| workspace | Shape | dtype | 用途 |
|-----------|-------|-------|------|
| workspace_kv | [20, 128, 576] | bf16 | V gather → C QK/PV 操作数（RH 分片共享） |
| workspace_s | [20, 128, 128] | fp32 | C QK → V softmax |
| workspace_p | [20, 128, 128] | bf16 | V p → C PV（A） |
| workspace_o | [20, 128, 512] | fp32 | C PV 段 → V acc_o 累加 |

**workspace_idx**: 不使用（显式参数 + host `torch.zeros` 分配 + 统一双后端调用约定；gqa_pto 与 indexer_bwd fast kernel 共同先例。若 ascendc 侧 AUTO_CV_SYNC 配对需要 workspace_idx 标注，Stage 2 补 `workspace_idx=[7,8,9,10]`，语义不变——记录为 R5 验证项）

### 8.3 Cube 核计算流程

```python
T.copy(Q[s_row, H0:H0+Hseg, :], q_l1)                    # GM→L1 直连（每 task）
for i in T.serial(NI):
    T.copy(workspace_kv[cid, 0:BI, :], kv_l1)            # ws→L1 整块
    for r in range(REPLICATE_H):
        T.gemm_v0(q_l1[r*64:(r+1)*64, :], kv_l1, acc_s_l0c, transpose_B=True, init=True)
        T.copy(acc_s_l0c, workspace_s[cid, r*64:(r+1)*64, :])
    for r in range(REPLICATE_H):
        T.copy(workspace_p[cid, r*64:(r+1)*64, :], p_l1)
        for d in range(2):
            T.gemm_v0(p_l1, kv_l1[0:BI, d*256:(d+1)*256], acc_o_l0c, init=True)
            T.copy(acc_o_l0c, workspace_o[cid, r*64:(r+1)*64, d*256:(d+1)*256])
```

### 8.4 Vector 核计算流程（AIV-vid 持 head 分片 vid 的全部 softmax 状态）

```python
# init / 块循环内：indices+mask 链 → rows → gather(HALF 行) → ws_kv
#   读 ws_s[vid 段] → ×sm_scale → mask（注 A）→ m_i/alpha/p/sumexp → acc_o×alpha
#   p → bf16 → ws_p；读 ws_o[vid 段, 4×128 列] → acc_o +=
# 尾部：acc_o/sumexp → Output；ln(sumexp)+m_i → Lse   （§3.3 全展开）
```

### 8.5 pass_configs 配置

§4.7（4-True）。

### 8.6 注意事项

- workspace 四路乒乓的 flag 计数由 AUTO_CV_SYNC 自动管理（命名配对，indexer_bwd 验证）；**HISA 教训（防悬挂）由 pass 保障**——若 Stage 2 发现配对遗漏（挂死），检查 buffer 命名前缀是否含 "workspace" 子串
- vid 分工恒等式：AIV-vid 负责 head 分片 vid（v_block = RH×64/2，RH=2 时恰一分片/AIV）+ gather 半区 vid×HALF；RH=1（head_kv ≤ 64）时 v_block = padded_H/2，两 AIV 各半分片——通用公式 v_block = Hseg//2，无需分支
- 单 kernel（无 cast kernel / 无 host 中间同步）：Output/Lse 一次写出

---

## 9. 验证方案

### 9.1 Golden 函数

主仓 `ref_sparse_mla_fwd_interface`（ref_gpu/sparse_mla_fwd.py L214-257）原样移植 NPU（device 参数化）：

```python
def ref_sparse_mla_fwd_interface(Q, KV, Indices, offsets, sm_scale=None, is_casual=True):
    """主仓逐行参考：fp32 域 scatter-mask softmax + einsum PV；输出 bf16。
    device: 输入张量所在设备（NPU 或 CPU）；内部 torch.arange device 随输入。"""
    Q = Q.float(); KV = KV.float()
    all_o = []
    for i in range(offsets.shape[0] - 1):
        q, kv = Q[None, offsets[i]:offsets[i+1]], KV[None, offsets[i]:offsets[i+1]]
        indices = Indices[None, offsets[i]:offsets[i+1]].clone().transpose(1, 2)
        # …（主仓 L223-255 逐行原样：scatter mask + causal + einsum score + softmax + PV）
    return torch.cat(all_o, dim=0).to(torch.bfloat16)
```

**语义对齐依据**（§1.4）：golden 的 scatter mask 为去重语义，主仓 kernel 为重复语义，两者在无重复输入上一致；L0 测试输入 = 主仓 randperm 构造（无重复）→ golden 有效。**L0 不构造含重复索引的输入**（重复语义验证属 L1，Stage 2 场景 B 以「kernel 语义 = 重复计贡献」为基准另行设计 golden 对照）。

### 9.2 L0 门槛测试计划（由 `tilelang-op-test-design` 场景 A 生成）

> **算子类别判断**（场景 A Phase 2）：计算类型 = **混合（CV 融合）**（gemm_v0 QK/PV + mask/在线 softmax 向量链 + 随机行 gather）；复杂度 = **Fusion**（GEMM + softmax + GEMM 流式组合）；数学特征 = matmul + exp/max/sum + matmul；综合类别 = **Fusion（融合算子）**。测试策略：CV 交互结果正确性优先、双后端各验、规则 shape 冒烟。
>
> 本节仅 **L0**（门槛、规则 shape、topk % block_I == 0 block 整除）；L1（功能，含不规则 shape / 重复索引语义 / -1 与越界语义边界 / 多段非整除）/ L2（异常输入）/ Boundary（特殊值）由 Stage 2 在 L0 通过后按 **场景 B** 扩展，此处不手工枚举。

**L0 用例表**（4 用例，≤50 门槛约束内；输入构造与主仓 `test_sparse_mla_fwd` L274-286 **逐行一致**——硬性忠实性约束）：

| 用例名 | 级别 | Shape（B/S/H/HKV/DQK/DV/topk） | offsets | dtype | target | 说明 |
|--------|------|-------------------------------|---------|-------|--------|------|
| l0_main_ascendc | L0 | 1/4096/128/1/576/512/1024 | [0, 2047, 4096] | bf16 | ascendc | 主仓 `__main__` 配置复刻 = 用户 perf shape（§9.4 哨兵同 shape）；block_I=128 |
| l0_main_pto | L0 | 同上 | 同上 | bf16 | pto | 同用例 pto 后端（双后端硬性要求） |
| l0_small_ascendc | L0 | 1/2050/128/1/576/512/1024 | [0, 1024, 2050] | bf16 | ascendc | 小尺寸快速回归（段长 1024/1026，满足主仓 `assert seq_len >= topk` 且段长非 topk 整除） |
| l0_small_pto | L0 | 同上 | 同上 | bf16 | pto | 同上 pto |

> 小 shape 说明：段长约束（主仓 L282 `assert seq_len >= topk`，topk=1024）要求每段 ≥ 1024 → S=2050、offsets=[0, 1024, 2050]（段 1024/1026，均 ≥ 1024 且非整除）；主配置 S=4096 → 段 2047/2049 天然满足。

**输入构造代码草案**（与主仓逐行一致，`S`/`offsets` 参数化）：

```python
def make_l0_inputs(S=4096, H=128, HKV=1, DQK=576, topk=1024, dtype=torch.bfloat16, device="npu"):
    torch.random.manual_seed(0)                                    # 主仓 L274（硬性）
    q = torch.randn((S, H, DQK), dtype=dtype, device=device)
    kv = torch.randn((S, HKV, DQK), dtype=dtype, device=device)
    offsets = torch.tensor([0, S // 2 - 1, S], dtype=torch.int32, device=device)  # 主仓 L277
    indices = torch.full((S, HKV, topk), S, dtype=torch.int32, device=device)     # 越界填充 = S
    for i in range(offsets.shape[0] - 1):
        seq_len = (offsets[i + 1] - offsets[i]).item()
        assert seq_len >= topk                                     # 主仓 L282
        for t in range(seq_len):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]             # 主仓 L285（无重复 → golden 去重语义一致）
                indices[offsets[i] + t, h, : len(i_i)] = i_i
    return q, kv, indices, offsets
```

**golden 判据（主仓口径 + 增补）**：
1. `assert_tensors_similar(tl_out, ref_out, eps=1e-2, name="out")`——主仓 L293 判据（余弦相似度 diff < 1e-2，ref_gpu/utils.py 原样移植）；
2. **增补（本设计加严，主仓不 check）**：Output 同时过 §9.3 bf16 混合容差（双门禁，两道都过）；Lse 过 §9.3 fp32 混合容差。
3. Golden = 主仓 `ref_sparse_mla_fwd_interface`（§9.1）在**与被测 kernel 相同 device** 上运行；CPU 回退开关（环境裁剪 aclnn 时）。

### 9.3 精度标准（混合容差，按 dtype）

> 逐元素 `|actual−golden| ≤ atol + rtol·|golden|`，整体判定 `matched_ratio ≥ required_matched_ratio` **且** `max_abs_error ≤ max_abs_error_limit`。主仓判据（相似度 eps=1e-2）作为 out 的**并列门禁**（两道都过）。本算子 NaN/Inf：输入为 randn 有限值、score/m 均有界（注 A 推导），**不支持特殊值输入**（无 NaN/Inf 位置契约需求）；若 L1/Boundary 阶段引入特殊值，按 precision-standard 的位置敏感比较扩展（Stage 2）。

| dtype（张量） | atol | rtol | max_abs_error_limit | required_matched_ratio |
|-------|------|------|---------------------|------------------------|
| bfloat16（Output / Q / KV） | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |
| float32（Lse） | 2⁻¹⁶ (1.53e-5) | 2⁻¹⁰ (9.77e-4) | 1e-2 | 0.99 |
| int32（Indices/Offsets/TokenIndices，仅寻址输入，无数值输出） | 0 | 0 | 0 | 1.0（精确匹配） |

### 9.4 性能可行性哨兵（强制执行，不可跳过）

| 用例名 | Shape | dtype/属性 | 覆盖路径 | 单 case timeout | 选择理由 |
|--------|-------|------------|----------|-----------------|----------|
| perf_shape_latency | 1/4096/128/1/576/512/1024 | bf16 | 唯一通用路径（persistent grid 满载 4096 task） | 600 s（含双后端编译 ~2-5 min + 运行） | 最大 DMA/GEMM 任务量（§9.5 预算锚点）；目标 < 24288 us |

**测量方法**（host 开销控制，硬性）：
1. **纯 kernel 计时**：NPU 事件（`torch.npu.Event(enable_timing)`）或 msprof device Task Duration（sparse_flash_mla bench_mark 口径：op_summary 首次 launch 剔除、余下取平均）——排除 host launch 开销；
2. **接口级 do_bench 等价口径**：`tilelang.profiler.do_bench(fn, rep=100, warmup=250)`（主仓 L301-305 同款）——**fn 内 prepare_token_indices 必须经 `tensor_cache` 缓存**（ref_gpu/index.py L49 装饰器原样保留），使 do_bench 稳态循环只测 kernel + workspace 传参，不含 CPU 构造；
3. 调用开销优化：host 入口按 `_BUILD_CACHE`（编译缓存）+ `_CachedLauncher`（参数 marshal 缓存，~13 us/launch，indexer_bwd L170）封装——**Stage 2 实现可选项**，性能测量以纯 kernel 口径为准；
4. 双后端各自测量，报告 ascendc / pto 两行。

测试数据、随机数、golden 物理重排（ref 的 clone/transpose/scatter）均在 golden 侧完成（属验证路径，允许 torch op）；**算子主路径**（§10.4 审计表）零 aclnn。

### 9.5 性能预算（910B3，实测锚点口径）

> 锚点：A2/910B3 实测 HBM 读 273GB/s、写 238GB/s；L2 中转 ~1TB/s 估算（indexer_bwd DESIGN.md §9.4 同款口径）；AIC ws→L1 读 ~50-60GB/s/核、AIV 行读 ~26GB/s/核（reducesum msprof 实测）；Cube bf16 峰值 ~320 TFLOPS（20 核聚合，任务口径"48 TFLOPS ≈ 15%"反推 + sparse_flash_mla csa 实测有效 91 TFLOPS 佐证可达性）。**预算遵循 rev1 教训：不依赖「L2 命中才可达」的临界断言——以下按单核 MTE 实测口径核算（L2/HBM 聚合带宽仅作交叉验证）。**

**任务量**（perf shape）：task = ΣS×G = 4096；每 task FLOP = 2×128 head×(576+512)×1024 = 285 MFLOP（QK 151 + PV 134）；总 1.167 TFLOP；205 task/核（striped，末核 206）。

**GEMM 时间**：每 task 48 次 gemm_v0（2 QK [64,576]×[576,128] + 4 PV [64,128]×[128,256] 每块 ×8 块）；纯算 3.65ms/核 @ 峰值 → 占 24288us 预算的 15%；与 MTE 流水重叠后**非独立瓶颈**。

**MTE 时间（主预算，fp32 workspace_o 基线）**：

| 侧 | 每 task 流量（MB） | 每核 205 task 总量 | 需求带宽 | 实测能力 | 占用 | 时间下限 |
|----|--------------------|--------------------|----------|----------|------|----------|
| C 侧（AIC） | ws_kv 读 1.18 + ws_s 写 0.5 + ws_p 读 0.25 + ws_o 写 2.1 + Q 读 0.15 = **4.18** | 857MB | **35.3GB/s** | 50-60GB/s | 59-71% | **14.3-17.1ms** |
| V 侧（每 AIV） | gather 读 0.59 + ws_kv 写 0.59 + ws_s 读 0.25 + ws_p 写 0.125 + ws_o 读 1.0 = **2.56** | 525MB | **21.6GB/s** | 26GB/s | 83% ⚠️ | **20.2ms** |

**预算结论**：V 侧 gather + 中转为关键路径，纯带宽下限 ~20.2ms；叠加 gemm 重叠（C/V 流水）、launch/尾部（<1%）、workspace 冷启动（~10MB HBM 填充一次性 ~40us）后估算 **~21-23ms < 24288us 目标，余量 5-13%**（贴线）。**优化档（Stage 3 菜单，按序启用）**：
1. workspace_o bf16 化（L0 精度验证后）：V 侧 2.06MB/task → 16.9GB/s（65%）→ 下限 ~15.8ms，**余量 ~35%**；C 侧同步降至 ~26GB/s；
2. workspace_kv 双槽预取 + gather 批量 staging（reducesum DEPTH/SIDES 机制）：提升 MTE 发射效率（50-70% → 80%+）；
3. ws_kv 直连探索档（**仅 ascendc fast path，PTO 走基线路径**）：gather 后 bf16→fp16 kernel 内 cast + fp16 UB→L1 直连（gqa_sink 消 ws 模式），省 ws_kv 9.66GB 中转；bf16→fp16 溢出（|x|>65504）语义风险 → 仅作可选 fast path，须以支持域谓词（host 侧 KV 值域预检）+ generic 回退成对出现（§11.4）。
4. M=128 合并 QK/PV（gemm 调用减半，L0C 贴线 128KB 需实测）。

**聚合带宽交叉验证**（L2/HBM 视角，非达成依据）：总 GM 流量 ≈ 33.5GB（fp32 ws_o）/24288us = 1.38TB/s。workspace+KV+Q 工作集 ~20MB 常驻 L2（910B3 L2 容量以平台配置为准）→ 中转流量主要由 L2 吸收（~1TB/s 估算口径下需双后端实测确认吸收率；**若 L2 吸收不足，上述单核口径预算仍然成立**——单核 MTE 实测锚点本身已含存储层级延迟）。

### 9.6 双后端验证方案（硬性）

| 项 | ascendc | pto |
|----|---------|-----|
| 编译 | `tl_sparse_mla_fwd(..., target="ascendc")`（默认） | `tl_sparse_mla_fwd(..., target="pto")` |
| 运行 | 同一 host 入口 `sparse_mla_fwd_interface(q, kv, indices, offsets, target=...)`，内部 `_BUILD_CACHE` 按 (shape 常量, target) 双键缓存 | 同左 |
| 精度 | §9.2 L0 用例 × §9.3 判据 | 同左 |
| 差异记录 | workspace 显式传参双后端一致；**pto 首验项**：GM 标量读（R2）、gemm_v0 L1 列切片（R1）、reduce fp32 dim=-1（gqa_pto 已有先例）、auto-cast bf16↔fp32（gqa_pto 已有先例） | — |

测试文件参数化：`@pytest.mark.parametrize("target", ["ascendc", "pto"])`（Stage 2 落地）。

---

## 10. 风险点与注意事项

### 10.1 已知约束（技术约束检测结论汇总）

| 检测项 | 结论 | 处理 |
|--------|------|------|
| 三维 Kernel | **命中**（GPU 源二维 grid） | persistent grid + pid 分解（§3.5.1） |
| threads > 2 | **命中**（GPU 源 128） | (cid,vid) MIX_AIC_1_2 |
| 动态循环边界 | **命中**（task 数运行时） | `T.serial(ntask)` 已验证形态 |
| GEMM 非整除 | **不命中**（全静态 tile，§5.3） | — |
| L0C 溢出 | 不命中（96KB/128KB；BI=256 档贴线已排除基线） | — |
| UB 溢出 | **贴线命中**（189KB/192KB） | §4.6 缓解清单 + R4 |
| T.copy 列 strided | 不命中（全部切片最内维连续） | — |
| bf16 UB→L1 直连 | **命中**（平台禁止） | GM workspace 中转（§8.2） |
| bf16 标量算术 | **命中** | 全标量链 fp32 |
| Host 侧 buffer 物理化 | 不命中（§10.4 审计） | — |

### 10.2 风险清单（含 PTO 可用性风险点）

| # | 风险 | 概率 | 影响 | 缓解 |
|---|------|------|------|------|
| R1 | **gemm_v0 的 L1 列切片**（kv_l1[0:BI, 0:256]）文档支持但无切片先例，任一后端失败 | 中 | PV 路径编译失败/数据错位 | 回退：ws_kv 分立双段布局 + QK 拆 K=512/K=64 两段（注 C，主仓 GPU 同构）；**Stage 2 首个验证项** |
| R2 | **PTO GM 标量读**（TokenIndices/Offsets）：codegen_ascend_pto.cc L802-812 已实现但无端到端 varlen 先例 | 中 | pto 编译失败/取值错误 | 最小复现 kernel 先验（5 行：读标量→printf）；失败则 pto 走 host 预展开 BlockIndices [num_tasks, 3] 表（indexer_bwd fast 同款，编译期元数据路径） |
| R3 | **AUTO_CV_SYNC 四路 ws 配对**在 (cid,vid) + persistent + 串行 r/d 子循环下的 flag 计数正确性 | 低 | 挂死/数据竞态 | indexer_bwd 同构验证（ws 更多数量的成对配对）；挂死时核对命名前缀；必要时分 kernel 化（r 循环外提） |
| R4 | **UB 贴线 189KB**：MEMORY_PLANNING 复用不达预期 → 编译期 UB 超限 | 中 | 编译失败 | 槽位复用清单（§4.6）；ws_o 读段改 8×[64,64]；p_bf16/acc_o_bf16 共槽；最后手段 v_block 相关 buffer 下放 ws（GM 中转小张量） |
| R5 | workspace 显式参数（无 workspace_idx）在 ascendc 的 AUTO_CV_SYNC 配对 | 低 | ascendc 挂死 | indexer_bwd fast kernel 同款（无 workspace_idx + 命名触发）已验证；若失效补 `workspace_idx=[7,8,9,10]`（gqa ascendc 版先例） |
| R6 | **性能贴线**（基线余量 5-13%） | 中 | 不达 24288us | §9.5 优化档 1（ws_o bf16，+22pp 余量）为第一响应；档 2/3/4 递进 |
| R7 | ws_s 传 V 的 fp32→(V 读) 链路上，mask 算术链 BIGC=1e30 在极端 score（|score| 接近 fp32 上界）下失效 | 低 | 极端输入精度 | 支持域约束随主仓（bf16 输入 score 有界 < 1e30×sm_scale 恒成立）；L1/Boundary 阶段验证 |
| R8 | pto 的 reduce fp32 dim=-1 在 [64,128] UB tile 的行为差异（docs T.reduce_sum.md 约束 8：BufferRegion 切片仅 ascendc） | 低 | pto 编译失败 | 主路径 reduce 输入恒为**整 buffer**（非切片）；acc_s_ub 整块归约 ✓ 已规避 |
| R9 | do_bench 口径下 host 构造（token_indices CPU→H2D）混入计时 | 中 | 测量偏差 | tensor_cache 装饰器缓存（§9.4）；纯 kernel 口径双保险 |

### 10.3 特殊场景处理

- **head_kv ≤ 64 且非 64 倍数**（如 H=64/HKV=4 → head_kv=16）：REPLICATE_H=1、Hseg=padded_H=16、v_block=8；GEMM M=16 满足分形下限 ✓；mask/gather 链不变——支持域内（编译期 assert，§11.4）
- **kv_group > 1**：task = (s_row, g) pid 分解通用；Indices/KV 的 g 维寻址已参数化
- **段长 < topk**：语义不依赖（无效槽 mask 置 0）；主仓测试构造 assert 段长 ≥ topk（L0 遵守）
- **B=1 单段**：offsets=[0, S] 退化 ✓

### 10.4 Host 侧 Buffer 操作审计（[HOST-METADATA-AUDIT]）

| operation | input→output | shares_storage | aclnn/物理拷贝 | 结果 |
|-----------|--------------|----------------|----------------|------|
| `prepare_token_indices(offsets)`（CPU 构造 + `.to(device)` 一次 H2D） | 新建张量 | 新分配（非改写用户输入） | 否（H2D 数据准备，reducesum L740-760 同款） | allow |
| `torch.zeros(...)` 分配 Output/Lse/workspace | 新建张量 | 新分配 | 否（分配 + 零填充属数据准备） | allow |
| kernel 调用（显式传参） | — | — | 否 | allow |
| golden 验证（ref/相似度） | 只读 | — | 验证路径允许 | allow |

无 `.contiguous()`/`reshape(非view)`/`torch.cat`/`F.pad`/`.to(dtype)`/`.clone()` 于算子主路径；无输出侧切片+reshape。

---

## 11. 交付清单

### 11.1 目录结构（本目录多算子共存，工件独立命名）

```
examples_experiment/dsa_sparse_finetune/
├── sparse_mla_fwd.py            # 纯 kernel（@tilelang.jit，target 参数化，可 import）【待实现】
├── test_sparse_mla_fwd.py       # kernel import + golden + L0 + 双后端参数化 + main【待实现】
├── DESIGN_sparse_mla_fwd.md     # 本设计文档（已交付；非 DESIGN.md——那是 indexer_bwd 的）
├── proto_sparse_mla_fwd.yaml    # 接口规格（已交付，§11.5）
└── ref_gpu/                     # 主仓参考（只读，不修改）
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN_sparse_mla_fwd.md` | ✅ 已完成 | 本文档 |
| `proto_sparse_mla_fwd.yaml` | ✅ 已完成 | dtype 全集（§9.3）/ attrs（§1/§4/§5.2）机械派生 |
| `sparse_mla_fwd.py` | ⬜ 待实现 | `tl_sparse_mla_fwd`（builder）+ `sparse_mla_fwd_interface`（host 入口：token_indices 预计算 + _BUILD_CACHE + target 分派）+ `prepare_token_indices` |
| `test_sparse_mla_fwd.py` | ⬜ 待实现 | `ref_sparse_mla_fwd_interface`（golden，device 参数化）+ L0 用例 ×{ascendc, pto} + perf 哨兵 + main |

### 11.3 命名规范

kernel 文件 `sparse_mla_fwd.py`；测试 `test_sparse_mla_fwd.py`（顶部 `from sparse_mla_fwd import sparse_mla_fwd_interface`）；公开符号 `tl_sparse_mla_fwd` / `sparse_mla_fwd_interface` / `prepare_token_indices`（reducesum 命名先例）。

### 11.4 [DISPATCH-COVERAGE]

```text
[DISPATCH-COVERAGE]
supported_domain:
  dtype 固定组合：Q/KV=bfloat16、Indices/Offsets/TokenIndices=int32、Output=bfloat16、Lse=float32（无 dtype 扫描维度）
  shape：DQK=dim+tail_dim（dim/tail_dim 各为 2 的幂，编译期 assert）；DV=dim；
        topk % block_I == 0 且 block_I ∈ {64, 128}（默认 128）；
        heads % kv_group == 0 且 head_kv = heads//kv_group 满足
        (16 ≤ head_kv ≤ 64 且为 2 的幂) 或 (head_kv % 64 == 0)（主仓 L60-72 原样继承）；
        ΣS / ΣS_kv / batch 运行时任意（T.symbolic）；Indices 值域 {−1} ∪ 任意 int32（越界 clamp + mask）
generic_fallback: tl_sparse_mla_fwd（单一通用 kernel，无分派）
specializations: 无（Stage 3 若引入 ws_kv fp16 直连 fast path，须按 §9.5 档 3 的
                支持域谓词 + generic 回退成对补充本表）
unsupported_inputs: topk % block_I != 0；dim/tail_dim 非 2 的幂；is_causal=False
                   （主仓 assert 原样继承，编译期拒绝，与 supported_domain 不冲突）
result: pass
```

### 11.5 算子 proto（proto_sparse_mla_fwd.yaml 摘要）

dtype 全集 = §9.3 精度表 dtype 行（固定组合无扫描维度）；attrs = heads/dim/tail_dim/topk/kv_group/sm_scale/block_I（§1/§4/§5.2 派生）。完整 YAML 见 `proto_sparse_mla_fwd.yaml`（已随本文档交付）。

### 11.6 实现顺序

1. ✅ 设计文档 + proto.yaml + L0 门槛测试计划（本文档 §9.2）
2. ⬜ R1/R2 最小复现验证（gemm_v0 列切片 / pto GM 标量读）——Stage 2 首日项
3. ⬜ kernel 实现（`sparse_mla_fwd.py`，ascendc 先行 → pto 复验）
4. ⬜ 测试文件（L0 × 双后端 + perf 哨兵）
5. ⬜ L0 门槛通过（精度收敛：相似度 eps=1e-2 + 混合容差双门禁）
6. ⬜ 性能验证（§9.4 口径 < 24288us；不达标按 §9.5 优化档递进，不回退语义）
7. ⬜ 扩展分层套件（L1/L2/Boundary，tilelang-op-test-design 场景 B）
