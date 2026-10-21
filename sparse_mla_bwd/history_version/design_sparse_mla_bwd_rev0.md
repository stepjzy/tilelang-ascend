# sparse_mla_bwd 算子设计文档

> 迁移来源：`examples_experiment/dsa_sparse_finetune/ref_gpu/sparse_mla_bwd.py`（GPU 版主仓参考，343 行，md5 fc4e8e5c9fd9d5a36e0c3a79c10b9a85，只读，语义权威）；配套依赖 `ref_gpu/sparse_mla_fwd.py`（325 行，只读）、`ref_gpu/utils.py`、`ref_gpu/index.py`
> 目标平台：Ascend 910B3（20 Cube Core + 40 Vector Core；每核 L1 512KB / UB 192KB / L0C 128KB / L0A 64KB / L0B 64KB；全芯片 L2 192MiB——锚点 `$ASCEND_HOME_PATH/aarch64-linux/data/platform_config/Ascend910B3.ini`：`l1_size=524288 / ub_size=196608 / l0_c_size=131072 / l2_size=201326592 / cube_core_cnt=20 / vector_core_cnt=40`，与 tilelang-api-best-practices api-kernel-memory.md §2 平台表一致）
> 编程模式：**Developer**（用户指定，不可更改；pass_configs 4-True AUTO 形态，indexer_bwd L275-280 先例）
> 双后端约束（用户硬性要求）：**ascendc（target="ascendc"，默认）与 PTO（target="pto"，显式）双后端均须通过编译 + 运行 + 精度验证**；性能以较优后端考核，另一后端如实报告不设门槛
> 性能目标：全接口（preprocess + bwd + postprocess 串行三段，与主仓 `do_bench(fn)` 口径一致，`delta=None` 每次调用含 preprocess）在 B=1, S=2048, H=64, HKV=1, DQKV=576(D=512, D_tail=64), DV=512, topk=512, bf16 下 **< 7680 us**；GEMM 总量 ~370 GFLOP（主仓 per_token_flop 公式 × S）
> 外部依赖：`sparse_mla_fwd` 由并行会话交付（DESIGN_sparse_mla_fwd.md 已落盘，只读协调）；本算子禁止自行实现 / 修改 fwd（依赖策略见 §11）

---

**文档生成状态（Stage 1 分块交付）**：本文档按任务书分 4 块生成——块 A（§1-§4，本次交付）；块 B（§5-§7）；块 C（§8-§11）；块 D（§12-§14 + proto）。占位章节内容由后续块填充，后块不得推翻前块决策。

## 1. 算子语义与接口契约

### 1.1 算子名称

sparse_mla_bwd（DSA 稀疏 MLA 反传，varlen packed 布局，因果 causal-only）

### 1.2 功能描述

对 varlen packed 序列的每个 packed 行 `s_row`（属于 batch `b`、段内位置 `s_i ≡ s_row − bos`）、每个 kv_group `g`：用前向保存的 topk 索引 `Indices[s_row, g, j]` 与 `Lse` **重计算**注意力概率 P（免二次 softmax），结合上游梯度 `dO` 与前向输出 `O` 计算 `q` / `kv` 的梯度。计算分三段（主仓 host 串行编排，计时口径一致）：

- **preprocess**：`Delta[t,h] = Σ_d O[t,h,d]·dO[t,h,d]`（fp32，纯带宽归约）
- **bwd**：逐 token、逐 topk 索引块（BS=32）重计算 P/dP，7 类 GEMM 产出 dq 逐块累加、dkv 逐块 fp32 原子散射
- **postprocess**：`dKV` fp32 → bf16 单次舍入

### 1.3 数学公式

前向语义（引用，由 `sparse_mla_fwd` 提供，本算子不实现）：

$$\text{score}[t,h,j] = \big(q[t,h,:512]\cdot kv[idx_j,:512] + q[t,h,512{:}576]\cdot kv[idx_j,512{:}576]\big)\times sm\_scale$$

$$P = \text{softmax}(\text{causal+topk mask 后的 score}),\quad o = P\,@\,v,\ \ v = kv[idx_j,:512],\quad lse = \text{logsumexp}$$

反传（`sm_scale` 缺省 `= 576^{-0.5}`；`mask[j] = 1\{idx_j \le s_i\} \wedge 1\{idx_j \ne -1\}`，因果判据用**段内位置 s_i**）：

$$\Delta[t,h] = \sum_{d<512} O[t,h,d]\cdot dO[t,h,d] \qquad (\text{preprocess, fp32})$$

$$P[t,h,j] = \exp\big(\text{score}_{raw}[t,h,j]\times sm\_scale - lse[t,h]\big)\ \ (\text{masked}\to 0,\ \text{bwd 重计算})$$

$$dP[t,h,j] = P[t,h,j]\cdot\Big(\sum_{d<512} dO[t,h,d]\cdot v[idx_j,d] - \Delta[t,h]\Big)\times sm\_scale$$

$$dq[t,h,:512] \mathrel{+}= \sum_j dP[t,h,j]\cdot v[idx_j,:512],\qquad dq[t,h,512{:}576] \mathrel{+}= \sum_j dP[t,h,j]\cdot kv[idx_j,512{:}576]$$

$$dKV[idx_j,:512] \mathrel{+}= dP[t,:,j]^{T}\! \cdot q[t,:,:512] + P[t,:,j]^{T}\!\cdot dO[t,:,:512],\qquad dKV[idx_j,512{:}576] \mathrel{+}= dP[t,:,j]^{T}\!\cdot q[t,:,512{:}576]$$

（dKV 两式均为 fp32 原子散射累加，跨 token 聚合；`kv[...,:512]` 兼作 v 与 dq 的 D 段操作数，`kv[...,512:576]` 为 score/dq 的 tail 段操作数——与主仓源码逐行对应，见 §2.2）

$$dkv_{out} = \text{bf16}\big(dKV_{fp32}\big)\ \ (\text{postprocess, 单次舍入})$$

### 1.4 接口契约（与主仓一字不差）

```python
def sparse_mla_bwd(q, kv, o, do, indices, lse, offsets,
                   sm_scale=None, is_casual=True, return_kernel=False, delta=None) -> (dq, dkv)
```

> 主仓参数名即为 `is_casual`（拼写如此，不得"修正"）；`return_kernel` 主仓接受但未使用，原样保留。

| 参数 | shape | dtype | 语义 / 约束 |
|------|-------|-------|------------|
| q | [S, H, 576] | bf16 | 查询（D=512 + D_tail=64）；contiguous |
| kv | [S, HKV, 576] | bf16 | 压缩 KV；前 512 维兼作 v，后 64 维为 tail；S_kv == S；contiguous |
| o | [S, H, 512] | bf16 | 前向输出（供 Delta 计算；`delta` 显式传入时不使用） |
| do | [S, H, 512] | bf16 | 上游梯度 |
| indices | [S, HKV, topk] | int32 | topk 索引；值域含 -1 与越界填充（如 S）——OOB 防护见 §4.6/§7；contiguous |
| lse | [S, H] | fp32 | 前向 logsumexp；contiguous |
| offsets | [B+1] | int32 | varlen 段边界（packed 布局） |
| sm_scale | 标量 | — | 缺省 `(D+D_tail)^{-0.5} = 576^{-0.5}` |
| is_casual | — | — | 恒 True（kernel 内 assert，非因果不支持） |
| delta | [S, H] | fp32 | 预计算 Delta 复用入口；None 则内部跑 preprocess |
| **返回** dq | [S, H, 576] | bf16 | |
| **返回** dkv | [S, HKV, 576] | bf16 | |

继承的断言（host / kernel 入口，原样保留）：`is_causal == True`；`topk % block_size == 0`（block_size 为内部索引块 BS）；`D = 512` 为 host 硬编码、`D_tail = 576 − D`；dtype 组合固定（q/kv/o/do bf16、indices int32、lse/delta fp32）；各输入 contiguity。

### 1.5 golden 判据与主仓测试口径

- **golden**：PyTorch autograd——`ref_sparse_mla_bwd_interface`（ref_gpu/sparse_mla_bwd.py:281-290）：q/kv clone + requires_grad，`o = ref_sparse_mla_fwd_interface(q, kv, indices, offsets)`（内部 fp32、输出 bf16；越界索引 clamp 到 sk 后 scatter 到第 sk+1 列再裁剪 → 贡献为 0；首行自注意强制有效），`o.backward(do)`，返回 `(q.grad, kv.grad)`。q.grad 覆盖全 576 维（score 双段），kv.grad 覆盖全 576 维（v 段经 o=P@v、tail 段经 score）。
- **判据**：`assert_tensors_similar(eps=1e-4)`（ref_gpu/utils.py:48-73）——全局归一化点积相似度 `sim = 2⟨x,y⟩ / (‖x‖²+‖y‖²)`（double 精度），要求 `0 ≤ 1 − sim ≤ 1e-4`；dq / dkv 分别判；全零张量特判通过。
- **主仓测试口径**（test_sparse_mla_bwd:293-343，移植须保持）：B=1, S=2048, H=64, HKV=1, DQKV=576, DV=512, topk=512, bf16；`randn` 输入与逐 token `randperm(max(1,t))[:topk]` 索引**无固定 seed**，indices 尾部填充 S；`offsets=[0,S]` 单段且 `seq_len ≥ topk` 断言；经 `sparse_mla_fwd_interface` 取 `(o, lse)`（fwd 依赖策略 §11）。移植可加固定 seed 但须注明与主仓的偏差（L0 计划 §12 细化）。
- **性能口径**：`do_bench(fn, rep=100, warmup=250)`，`fn` 调 `sparse_mla_bwd(..., delta=None)` → **每次调用含 preprocess**，串行三段计时；目标 < 7680 us（§13 分解）。

## 2. 主仓源码结构分析

### 2.1 三段结构与 host 编排（ref_gpu/sparse_mla_bwd.py）

| 段 | 位置 | grid / 形态 | 计算内容 |
|----|------|------------|---------|
| preprocess | L10-45 | `T.Kernel(H, ceildiv(S,32))`，block_ND=32，`T.Pipelined(D/32, num_stages=5)` | O/dO [32,32] fragment 逐块累乘 + `T.reduce_sum` → Delta[S,H] fp32；纯带宽（读 268MB、写 0.5MB） |
| bwd | L77-251 | `T.Kernel(S, kv_group)` **逐 token grid**，threads=128，num_stages=0；pass_configs 关 TMA / WARP_SPECIALIZED | 核心计算，见 §2.2 |
| postprocess | L48-74 | `T.Kernel(ceildiv(S_kv,64), kv_group)`，threads=128 | dKV fp32 → bf16 纯拷贝 cast（单次舍入） |

host `sparse_mla_bwd`（L254-278）：contiguity 断言 → `prepare_token_indices(offsets)`（CPU 构造 + tensor_cache 单条缓存，ref_gpu/index.py:80-82）→ `delta is None` 则 preprocess → bwd → postprocess，三段串行。`ref_sparse_mla_bwd_interface`（L281-290）为 golden（§1.5）。

### 2.2 bwd 逐 token GEMM 清单（每 token × NS=16 索引块，BS=32，padded_H=64）

| # | 调用（源码行） | 形状 M×N×K | 累加语义 |
|---|---------------|-----------|---------|
| 1 | `T.gemm(Q_shared, KV_shared, acc_p, transpose_B=True, FullCol)` L196 | 64×32×512 | acc_p 跨 #1/#2 两段累加 |
| 2 | `T.gemm(Q_tail_shared, KV_tail_shared, acc_p, transpose_B=True)` L200 | 64×64 尾段 ×32 | 并入 #1（score 双段） |
| 3 | `T.gemm(dO_shared, KV_shared, acc_dp, transpose_B=True, clear_accum=True)` L207 | 64×32×512 | 每块 clear 单次 |
| 4 | `T.gemm(dP, KV_shared, acc_dq)` L213 | 64×512×32 | **acc_dq 跨 16 块累加** |
| 5 | `T.gemm(dP, KV_tail_shared, acc_dq_tail)` L214 | 64×64×32 | acc_dq_tail 跨 16 块累加 |
| 6 | `T.gemm(dP, Q_shared, acc_dkv, transpose_A=True, clear_accum=True)` L216 | 32×512×64 | **pair 第 1 GEMM**（clear） |
| 6' | `T.gemm(P, dO_shared, acc_dkv, transpose_A=True)` L217 | 32×512×64 | **pair 第 2 GEMM**（同 acc 累加） |
| 7 | `T.gemm(dP, Q_tail_shared, acc_dkv_tail, transpose_A=True, clear)` L220 | 32×64×64 | 每块 clear 单次 |

向量链（每块）：mask `idx<=s_i & idx!=-1`（L186）→ acc_p 预填 −inf（L190）→ `exp(acc_p·sm_scale − lse)`（L203，P 重计算）→ fp32→bf16 cast（P/dP_shared_cast L205/212）→ dP 链 `P·(acc_dp−Delta)·sm_scale`（L210）→ split_store=2 搬运 + per-element `T.atomic_add` 散射 dKV（L222-242）。

形状特征：M=64(H) 固定、**N=32(BS) 极小**（GEMM 效率核心难点）、K∈{512,64,576(可合并)}；GEMM #4/#5 的 K=32、#6/#6'/#7 的 M=32 同为小维。FLOP：主仓 `per_token_flop = 2·H·topk·(2·DV+3·DQKV)` ≈ 180.4 MFLOP/token（与逐类精确和一致），×S=2048 → **~370 GFLOP**。

### 2.3 OOB 索引事实（Orchestrator 已核实，设计必须覆盖）

- **测试生成**：indices 初始全填 S(=2048)，每 token 前 min(t, topk) 个槽写 `randperm(max(1,t))` 值 → t<topk 时尾部槽保持 S；接口契约另允许 **-1**（mask 判据含 `!= -1`）。
- **gather 无防护**（L194/199）：`KV[bos + Indices[...]]` 直接寻址，idx=S 越界（合法行 `[bos, bos+S_kv)`）；GPU 加载垃圾值，被掩码项贡献恰为 0（acc_p 预填 −inf → exp=0 → P=0、dP=0×垃圾）**侥幸通过**——若垃圾值含 NaN 则 0×NaN=NaN 会失败。
- **scatter 无防护**（L232-241）：`T.atomic_add(dKV[bos + Indices[...]], ...)` 无条件散射，idx=S 为 **OOB 写**。
- NPU 实现必须双端防护且数学结果与 GPU 语义一致（决策 §4.6，机制 §7）。

### 2.4 动态维度与 host 分派

- `T.dynamic("S, B_plus_one")`（L115）：S（packed 总行数）与 B+1 运行时确定；其余 H/D/D_tail/topk/kv_group 为编译期常量参数。
- TokenIndices [S,2] 由 host `prepare_token_indices` 预计算（position_ids / sequence_ids，一次 H2D）；kernel 内 GM 标量读 `b_i, s_i = TokenIndices[...]`（L170），`bos, eos = Offsets[b_i], Offsets[b_i+1]`。
- varlen 语义：行寻址恒等式 `s_row = bos + s_i`；causal 判据用**段内位置** `max_kv_i = s_i`（L173），非 packed 全局行号。
- host 硬编码：`D = 512`（L264）、`D_tail = 576 − D`；`kv_group = kv.shape[1]`；`topk = indices.shape[-1]`。

### 2.5 GPU 形态（不迁移）与语义锚点（必须迁移）

**GPU 形态**：逐 token grid（2048 block × threads=128）、`GemmWarpPolicy.FullCol`、num_stages=0（Pipelined 无实际流水）、split_store=2 + `T.view(KV_shared)` 复用（warp 布局产物）、per-element 标量 atomic、`alloc_shared/alloc_fragment`（GPU 语义）。**Ascend 侧架构自由度完全开放**（persistent / 多 token 合并 / C-V 分工 / MIX_AIC，以达标为准）。

**语义锚点**：§1.3 全部公式与 mask 语义（−inf 预填 + exp）；fp32 dKV 中间量 + postprocess 单次舍入；逐 token 的 Q/dO 单次装载、dq 跨 16 块 fp32 累加；重复索引不去重（重复计贡献，与 fwd DESIGN §1.4 第 1 条一致）；OOB 行为等价性（§7 论证）。

## 3. 技术约束检测结论

### 3.1 三维 Kernel

主仓 `T.Kernel(S, kv_group)` 为 2D grid + threads（GPU 形态）。Ascend 侧改 **persistent 1D 编译期常量 grid**（`T.Kernel(grid_size=20, is_npu=True)`），任务维折叠进核内 striped 轮询循环——彻底规避高维 grid 限制。先例：`sparse_mla_topk_reducesum.py` L218/261、主仓 `sparse_mla_topk_reducesum` 的 persistent 双变体（graph-replay 143.3us 实测）。**✓ 通过**。

### 3.2 threads 映射

主仓 `threads=128` 为 GPU warp 布局参数，Ascend 无对应语义。改用 `(cid, vid)` 启动形态 → codegen 默认 **MIX_AIC_1_2**（1 Cube + 2 Vector / block；`indexer_bwd.py` L358-366 注释已验证：AIV 见 `cid = GetBlockIdx()/2`、`vid = GetSubBlockIdx()∈{0,1}`）。grid=20 → **20 AIC + 40 AIV 全部用满**（与 910B3 核数一致）。AIV-only kernel（preprocess/postprocess/cast）用 `threads=1` 的 `(cid)` 形态（gqa_sink Kernel 2 L317 先例）或 (cid,vid) 形态（§11 定）。**✓ 通过**。

### 3.3 动态循环边界

`S / B_plus_one` 为运行时动态维。处理：外层 persistent 轮询 `ntask = T.max(0, T.ceildiv(total − cid, grid))` + `T.serial(ntask)` **运行时边界**（reducesum L265 已验证形态）；内层 NS（=topk/BS）、BS、D 段循环全为编译期常量。约束：动态边界禁 `T.Pipelined`（ascend-constraints §1）→ 主仓 `T.Pipelined(NS, num_stages=0)` 本就无实际流水，Ascend 侧用 `T.serial(NS)` + AUTO_CV_SYNC 自动 flag 配对替代（§9）。varlen 段内 token 边界不进循环结构（mask 链吸收，§10）。**✓ 通过**。

### 3.4 L0C 容量（查证：**128KB/核**，任务书预估 256KB 不成立）

锚点：`Ascend910B3.ini` `l0_c_size=131072`（= 128KB；另 l0_a/l0_b 各 64KB、l1 512KB、ub 192KB），与 tilelang-api-best-practices api-kernel-memory.md §2 平台表一致。

需求核算（考核 shape，fp32 累加器全驻留）：acc_p / acc_dp [64, BS=32] 各 8KB、acc_dq [64, 576] **144KB**、acc_dkv [32, 512] 64KB、acc_dkv_tail [32, 64] 8KB → 峰值同时驻留 >192KB ≫ 128KB，**✗ 不可全驻**。

结论：**dq 累积移出 L0C**——Vector 侧 UB fp32 跨块累积（fwd DESIGN acc_o 同款先例），head 分片到 2 AIV 后每 AIV [32, 576] fp32 = 72KB < UB 192KB；dkv / dkv_tail 每块产出后即时散射，L0C 逐块复用；acc_p / acc_dp 8KB 级分时复用。BN 加宽（N=32→64/128，§5）后相应重核。详细分配 §8。

### 3.5 GEMM 非整除（分形 16 对齐逐项过检）

| 维 | 值 | %16 | 说明 |
|----|----|-----|------|
| M | 64（padded_H） | ✓ | H_kv=64；`padded_H = max(next_pow2(H_kv),16)` 兜底 H_kv<16 |
| N | BS=32 | ✓ | `topk % BS == 0` 断言继承（512%32=0） |
| K | 512 / 64 / 576 | ✓ | 576 = 36×16（QK 双段可合并单 GEMM，fwd 先例） |
| D_tail | 64 | ✓ | |

varlen 段长任意**不产生块尾非整除**：gather 行数恒为整块 BS，越界/无效行由 mask 归零（§7）；段尾 token 数任意由任务=token 粒度天然吸收（无 token 打包截断）。残余风险：kv_group>1 通用路径的 head padding（考核 shape kv_group=1 无 pad，§5/§14 记录）。**✓ 通过**。

### 3.6 平台强制偏差（已知约束，rev2 教训，设计必须内建）

1. **bf16 UB→L1 直连被 static_assert 禁止**（half-only）→ C↔V 中转必须走 GM workspace（fwd DESIGN §1.5 注、indexer_bwd README 偏差 #1）。
2. **L0C→UB 可能被编译器改写为 GM 往返**（ascend_workspace_reduction.cc，indexer_bwd README 偏差 #2）→ 中转流量按 GM 口径显式入性能预算（§13）。
3. **auto workspace 不支持动态 grid** → 全部 workspace 显式参数化 + buffer 命名含 `"workspace"` 触发 `TL_ASCEND_AUTO_CV_SYNC` 自动握手（indexer_bwd docstring L138-141 已验证机制）。
4. **`T.tile.atomic_add` 约束**（tilelang/language/ascend_tile.py:313-330 已查证）：dst 必须 GM（scope global）、src 须 local（UB/L0C/L1）、dtype 与 dst 一致、**支持 fp32**（float16/float32/int16/int32/bfloat16）。

### 3.7 检测结论汇总

| 约束项 | 结论 | 设计响应 |
|--------|------|---------|
| 三维 Kernel | ✓ | persistent 1D grid=20（§4.2） |
| threads 映射 | ✓ | (cid,vid) → MIX_AIC_1_2，20 AIC + 40 AIV（§4.2） |
| 动态边界 | ✓ | T.serial(ntask) 运行时外层 + 编译期常量内层（§10） |
| L0C 容量 | **✗ 全驻 → 分层处置** | dq 累积移 UB fp32；dkv 即时散射；L0C 逐块复用（§4.3/§8） |
| GEMM 非整除 | ✓ | 全维 16 对齐；varlen 由 mask 吸收（§5/§7） |
| 平台强制偏差 | ✓ 已内建 | GM 中转 + 显式 ws 参数 + fp32 atomic（§4.5/§8/§13） |

## 4. 总体架构决策

> 本节只写决策与理由；机制细节（分块参数 / 流量估算 / clamp 与掩码实现）分别在 §5-§7（块 B）。

### 4.1 三段 kernel 架构（保持主仓结构）

**k1 preprocess（AIV 纯带宽）→ k2 bwd（CV 融合核心）→ k3 postprocess（AIV cast）**，host 串行编排。理由：① 与主仓接口 / `do_bench` 串行三段计时口径逐字一致（性能考核可比）；② preprocess/postprocess 为纯带宽 / 纯 cast，与 bwd 的 CV 资源特征不同，独立 kernel 各自最优（§11）；③ postprocess 兼作 dKV fp32→bf16 的单次舍入点，语义锚点（§1.3）。三 kernel 均 Developer 模式、persistent。

### 4.2 bwd：persistent grid 与任务映射（grid → 20 AIC / 40 AIV）

- **启动形态**：`T.Kernel(grid_size=20, is_npu=True) as (cid, vid)` → MIX_AIC_1_2（1 AIC + 2 AIV/block）→ **20 AIC + 40 AIV 全用满**（§3.2）。
- **任务粒度**：`(token, kv_group)`（考核 shape kv_group=1 → 2048 任务），striped 轮询 `ntask = T.max(0, T.ceildiv(total − cid, grid))`（reducesum L261 先例）。每任务 Q/dO 装载一次、16 索引块循环、dq 累积一次写出。
- **拒绝逐 token grid=S（主仓 GPU 形态）**：2048 个小 block 的启动 / 调度开销占比高；MIX_AIC 混核比例无法按 block 粒度分配；散射随 block 数碎片化。persistent 20 块为 reducesum / indexer_bwd 双先例验证形态，且核内程序序天然提供装载 / 散射与计算的重叠窗口。
- **多 token 合并（TS 捆绑）判定**：indices 逐 token 独立随机 → KV gather 不可跨 token 共享，QK GEMM 无法直接升 M——**不采用跨 token 捆绑**；小 N 效率改由 **BN 加宽**（索引块合并，N=32→64/128）解决，可行性论证与参数表 §5。

### 4.3 C/V 分工（每 block：1 AIC 计算 + 2 AIV 向量链）

- **Cube**：§2.2 全部 7 类 GEMM。QK 双段合并 **K=576 单 GEMM**（fwd DESIGN §3.2 先例，zN fractal K%16==0）；dkv **pair-GEMM**（dPᵀ@Q `init=True` + Pᵀ@dO `init=False` 同 L0C 累加，gqa_sink:434-436/469-471 先例）；dq/dq_tail 每块产出贡献块。
- **Vector（2 AIV 按 head 半片分工，各 32 head，fwd DESIGN AIV-vid 先例）**：mask 链（向量化 compare/select，gqa_sink L393-423 先例）；`P = exp(s·sm_scale − lse)` 与 dP 链；**dq 累积驻 UB fp32**（每 AIV [32,576]=72KB < 192KB，§3.4 结论）；dkv 逐行条件原子散射；lse/Delta GM 标量读。
- **C↔V 中转**：L0C→GM ws→UB（acc_p/acc_dp/dq 贡献块，AIV 按 head 半片读）与 UB→GM ws→L1（P/dP bf16、gather 的 KV——bf16 UB→L1 直连禁止，§3.6-1）；buffer 命名含 `"workspace"` 触发 AUTO_CV_SYNC 自动握手。Developer 模式下 C/V 指令分离与同步全交 4 个 AUTO pass，无 `T.Scope` / 手动 flag（边界说明 §9）。

### 4.4 双后端兼容策略（ascendc + PTO 单实现）

**单一 kernel 源码，API 限定双先例交集**：ascendc 侧 indexer_bwd（persistent / (cid,vid) / `T.gemm_v0` / `T.tile.*` / `T.tile.atomic_add` fp32 / GM 标量读 / 4-True pass_configs）与 pto 侧 gqa_pto（同能力集的 PTO 验证）共同覆盖本算子全部所需能力，零新增机制风险。已知后端差异（L0C 累积 dtype、workspace 行为、reduce/broadcast 路径——docs/TileLang-Ascend Programming Guide）在 §5/§8 逐 API 标注双后端可用性；host 侧 `_CachedLauncher` 按 `(target, 编译键)` 双份缓存（indexer_bwd 先例）。**ascendc 为默认主考核路径，pto 显式 `target="pto"` 同源验证、如实报告**。

### 4.5 dKV 散射策略选型结论（细节与流量估算 §6）

**选定方案 1：fp32 GM 工作区 + 逐行条件 `T.tile.atomic_add` + postprocess 单次 cast**。理由：

1. **与主仓语义逐字一致**（主仓 dKV 本就是 fp32 GM `T.empty` + atomic_add + postprocess cast）——精度路径零偏离，fp32 中间量 + 单次舍入为语义锚点。
2. **fp32 原子为精度硬要求**：每 dKV 行期望 ~topk=512 次累加；indexer_bwd rev2 教训——bf16 GM atomic 实测 ratio 0.9405-0.9611 全 FAIL（仅 64 次累加已然失败），fp32 后 0.9972 通过。`T.tile.atomic_add` fp32 GM dst 已查证（ascend_tile.py:313-330）。
3. **L2 驻留**：ws_dkv [2048,1,576] fp32 = 4.5MB，加 KV 2.25MB / indices 4MB 工作集 ~10.75MB ≪ L2 192MiB（ini 锚点）→ ~2.4GB 原子流量由 L2 吸收，HBM 仅见 4.5MB 终态。
4. **逐行条件（非整块 region）**：索引随机无 dst 连续性，region atomic（gqa_sink:437 需连续行）不适用；先例 unsorted_segment_sum:50 / indexer_bwd layout=1 逐行形态。
5. **备选**：方案 2（分核私有 ws 20×4.5MB + 无原子散写 + 归约 cast）散写字节同量级且 +90MB workspace / 归约 pass / 更大 L2 压力——列为 **fallback**（若 §6/§13 实测原子吞吐不足）；方案 3（行中心重构）**拒绝**：q/dO 重读 ~74GB 或需物化 P/dP（2×268MB）且计算退化为 GEMV，Cube 效率崩塌。

### 4.6 OOB 防护结论（机制与等价性论证 §7）

- **gather 侧**：`rows = clamp(idx, 0, S_kv−1)` 保护寻址 + mask（`idx ≤ s_i ∧ idx ≠ −1`）归零贡献——载入恒为合法行**有限值**，被掩码项贡献恒 0（0×有限=0），杜绝 GPU 形态的 0×NaN 传播路径；数学结果与 GPU 语义等价。
- **scatter 侧**：掩码行**条件跳过**（per-row 动态 if，unsorted:42 先例），禁止 OOB 写；"零填充 + clamp 到固定行"方案否决（考核 shape ~131k 掩码槽 × 2304B 原子集中单行 → 热点）。

### 4.7 决策汇总表

| 决策点 | 结论 | 依据 / 先例 |
|--------|------|------------|
| 三段架构 | preprocess / bwd / postprocess 三 kernel + host 串行 | 主仓计时口径（§4.1） |
| bwd grid | persistent grid=20，(cid,vid) MIX_AIC_1_2，20 AIC + 40 AIV | reducesum L218/261、indexer_bwd L358（§4.2） |
| 任务粒度 | (token, kv_group) striped 轮询；拒绝逐 token grid 与跨 token TS 捆绑 | §4.2 |
| 小 N GEMM | BN 加宽（N=32→64/128），非 TS 捆绑 | §4.2，参数表 §5 |
| C/V 分工 | Cube 7 类 GEMM（QK 合并 K=576、dkv pair-GEMM）；Vector mask/exp/dP/dq 累积/散射；head 半片分 2 AIV | gqa_sink、fwd DESIGN（§4.3） |
| dq 累积 | UB fp32 跨块累积（L0C 128KB 放不下，§3.4） | fwd acc_o 先例 |
| dKV 散射 | 方案 1：fp32 GM ws + 逐行条件 atomic + cast；方案 2 fallback；方案 3 拒绝 | §4.5 |
| OOB 防护 | gather clamp+mask 归零；scatter 条件跳过 | §4.6 |
| 双后端 | 单实现，API 限双先例交集，_CachedLauncher 按 target 缓存 | §4.4 |

## 5. Tiling 与 GEMM 策略

### 5.1 GEMM 形态重构：主仓 7 发 → Ascend 6 类

主仓 §2.2 的 7 类 GEMM 在 Ascend 侧归并为 6 类（§4.3 已决策）：QK 双段合并为单发 K=576（#1+#2）；dkv pair-GEMM 归一为一个 acc 槽双发（#6+#6'，gqa_sink L434-436 `init=True`/`init=False` 同款）。**形状设计原则：M/N/K 全部静态编译期常量且 %16==0（§3.5），topk 方向分块 BN 与 D 方向分段 DSEG 为仅有的两个可调参数**。

| 类 | 形状 M×N×K（BN=128, DSEG=2 档） | L0C acc（fp32） | 发射/块 | 对应主仓 |
|----|--------------------------------|-----------------|---------|---------|
| G1 score | 64×BN×576 | [64,BN] 64KB | 1 | #1+#2（合并双段） |
| G2 dO·v | 64×BN×512 | [64,BN]（复用 G1 槽，分时） | 1 | #3 |
| G3 dq | 64×(512/DSEG)×BN ×DSEG 段 | [64,256] 64KB | 2 | #4 |
| G4 dq_tail | 64×64×BN | [64,64] 16KB | 1 | #5 |
| G5 dkv pair | BN×(512/DSEG)×64 ×2发×DSEG 段 | [128,256] 128KB（贴线，单发独占段） | 4 | #6+#6' |
| G6 dkv_tail | BN×64×64 | [128,64] 32KB | 1 | #7 |

- **K=576 单发依据**：576=36×16 分形全对齐；Q [64,576] 驻 L1（72KB），L0A/L0B 按 m16n16k16 分形流式装载（Developer 模式 codegen 管），无需全操作数驻 L0A——fwd DESIGN §5.2 同款先例。
- **G1→G2 分时复用**：G1 产出先 `T.copy` 出 ws_s（AIV 消费）后 G2 覆写同一 fragment——生命周期天然分离（gqa_sink l0c_s/l0c_dp 分槽因 head_dim 小无压力；本算子 L0C 紧张，能复用则复用，§8 细化）。
- **发射数/task**：BN=128 → (1+1+2+1+4+1)×NI=4 块 = **40 发**；BN=64 → 10×8 = 80 发；BN=32（DSEG=1，[32,512]=64KB 可单发）→ 7×16 = 112 发。

### 5.2 小 N 难点与 BN 加宽（索引块合并）

主仓 BS=32 → N=32 仅 2 个 n 分形列：MMA 阵列利用率受限于单 GEMM 分形数（M/16 × N/16 = 4×2=8），且每发的 GEMM 指令序列短、发射开销占比高。**BN 加宽 = 合并 NI_adj 个索引块为一发**（topk=512 → BN=128 时 NI=4）：

1. **Q 复用机理**：Q [64,576] 的 L0A 分形装载在同发内服务全部 N 列——Q 的 L0A→MMA 供给流量相对 BN=32 降 4 倍（N 列从 2 → 8 个分形共享同一 A 分形流）。
2. **发射摊薄**：GEMM 发射 112 → 40 发/task；每次发射的 L0C 分配 / fragment 管理 / AUTO_CV_SYNC 握手开销同步摊薄。
3. **AIV 向量化长度**：mask/exp/dP 链的 tile 行长 [64,BN]——BN=128 时单条 `T.tile.*` 指令覆盖 8192 元素（vs BN=32 的 2048），向量核发射效率提升。
4. **散射不受益也不受损**：dKV 散射行次 = 有效 idx 数（917,760，§6.1），与 BN 无关；BN 只改单次散射循环长度与 mask 评估粒度。

**三档对比**（L0C 按"最大单发独占段"口径，L1/UB 按单缓冲，§8 复核）：

| 参数 | BN=32 | BN=64 | **BN=128（主选）** | BN=256 |
|------|-------|-------|--------------------|--------|
| NI（topk/BN） | 16 | 8 | 4 | 2 |
| GEMM 发射/task | 112 | 80 | **40** | 20 |
| L0C 峰值 | G5 [32,512] 64KB | G5 [64,256] 64KB | **G5 [128,256] 128KB 贴线** | G5 [256,128] 128KB（DSEG=4） |
| L1 占用 | q 72+do 64+kv 36+pd 8 =180KB | 72+64+72+16=224KB | **72+64+144+32=312KB** | 72+64+288+64=488KB ✗ 贴线弃 |
| UB（每 AIV 半片） | ~100KB | ~120KB | **~140KB ⚠** | ~170KB ✗ |
| ws_s/ws_p 单块中转 | ×16 次 | ×8 次 | **×4 次** | ×2 次 |

主选 **BN=128**（GEMM 效率与内存占用的平衡点，L0C/L1 双贴线但均单发独占段内可行，fwd DESIGN §5.2 BI=128 同档先例）；**BN=64 为 Stage 2 回退档**（若 L0C fragment 分配冲突或 UB 超 192KB，全部约束宽裕、发射数翻倍仍可接受）；BN=256 因 L1 488KB 贴线 + UB 超限弃；BN=32 仅作对照不选。

### 5.3 dq / dkv 的 N 分段（DSEG=2）

acc_dq [64,512] fp32 = 128KB 单发即贴满 L0C（无并发余量）、acc_dkv [BN,512] 在 BN=128 时 256KB 直接超限（§3.4）——**N 维（D 输出列）按 DSEG=2 分段**，与 fwd DESIGN §5.2 `DV_SEG=256` 同款：

- G3 拆 2 发：`[64,256]×BN`，acc [64,256] 64KB；
- G5 pair 拆 2 段 ×2 发：`[BN,256]×64`，acc [BN,256]（BN=128 → 128KB 贴线，pair 双发共用同 acc，第二发 `init=False` 累加——gqa_sink L434-436/L469-471 形态）；
- **512B 对齐核验**（fwd 注 C 约束：B 操作数列切片段起始 512B 对齐）：G3 的 B = kv_l1 [BN,576] 列段 `d*256:(d+1)*256` 起始 512B ✓；G5 的 B = q_l1 [64,576] 列段起始 512B ✓；G4/G6 的 64 列尾段起始 512×2B=1024B ✓；L0C C 起始 256 元素对齐（64×256、128×256 均 256 倍数）✓。
- **D 分段多轮方案（任务书 §3.3 备选）不采用**：即 dq 累积留 L0C、D 分 2 轮各 [64,256] 驻留——需 KV gather 块跨轮驻留 L1（16 块全驻 1.15MB > 512KB ✗）或重 gather，且 P/dP 需重算或物化；§3.4 已决策 dq 累积移 UB（块 A），不推翻。

### 5.4 跨 token TS 捆绑不可行论证（§4.2 决策展开）

indexer_bwd 的 TS 捆绑（M=TS×H 升维）前提是**同 token 的 T 子块共享同一组 KV 行**（规则块结构）。本算子该前提不成立：

1. **gather 无共享**：indices 逐 token 独立 `randperm`——token t₁ 的 512 行与 t₂ 的 512 行交集期望仅 ~topk²/S ≈ 128 行，GEMM 的 B 操作数（gather 后 KV）逐 token 不同 → QK/PV 类 GEMM 的 M 维合并后 B 无法共享，须退化为逐 token 分别发射，捆绑零收益；
2. **Q 侧 M 扩展收益微小**：唯一可合并量是 Q 装载摊薄（0.203MB/task → 捆 TS=2 省 0.1MB），但代价是 mask / lse / Delta / dKV 散射全部引入 token 间交错（mask_2d 变 [TS×64, BN]、散射行源跨 token 混排），复杂度收益比完全不成立；
3. **小 N 难点已由 BN 加宽独立解决**（§5.2），无需捆绑路径。

故小 GEMM 效率问题**唯一解法 = BN 加宽**（N 维 32→128），与 §4.2/§4.7 决策一致。

### 5.5 分块参数表与 per-task 中转流量参数

**BN=128 主档全参数**：

```python
BN    = 128   # topk 方向索引块合并（NI = topk/BN = 4）；BS=32 语义由 BN 吸收
DSEG  = 2     # dq/dkv 的 D=512 输出列分段（256 列/段，512B 对齐）
Hseg  = 64    # M 维（padded_H，kv_group=1 考核 shape 无 head 分片）
v_block = 32  # 每 AIV 的 head 半片（2 AIV × 32 = 64，fwd DESIGN §8.6 vid 分工恒等式）
grid_size = 20  # persistent（§4.2）
```

**per-task GM 中转流量**（BN=128、fp32 ws_dq 基线，供 §6/§13 消耗；ws 槽形态 [20, …] per-core 覆写，fwd §8.2 同款）：

| 流 | shape | dtype | 写侧 | 读侧 | 字节/task |
|----|-------|-------|-------|------|-----------|
| workspace_kv | [BN,576]×NI | bf16 | AIV(gather 半区) | C(整块×NI) | 0.5625MB（总量与 BN 无关） |
| workspace_s | [64,BN]×NI | fp32 | C | AIV(半片) | 0.125MB |
| workspace_p / workspace_dp | [64,BN]×NI | bf16 | AIV(半片) | C | 0.125MB |
| workspace_dq | [64,576]×NI | fp32 | C | AIV(半片读累加) | 0.5625MB（bf16 档减半，§6.4） |
| workspace_dkv（贡献槽） | [BN,576]×NI | fp32 | C(L0C 直写) | AIV(分摊散射) | 1.125MB |
| ws_dkv（终态原子区） | [S,1,576] 全局 | fp32 | AIV 原子（有效行 0.985MB/task） | postprocess | 4.5MB 全局 |
| Q/dO GM→L1 | [64,576]+[64,512] | bf16 | — | C 直读 | 0.203MB |
| dq 终写 | [64,576] | bf16 | AIV(半片) | — | 0.141MB |

## 6. dKV 散射策略专题

### 6.1 散射规模推导（考核 shape，因果语义决定有效行次）

dKV 散射的数学规模由因果 mask 决定——token s_i 的有效 j 槽数 = min(s_i+1, topk)：

```text
有效行次 = Σ_{t=0}^{S-1} min(t+1, topk) = Σ_{t=0}^{511}(t+1) + Σ_{t=512}^{2047} 512
         = 131,328 + 786,432 = 917,760 行次（全槽 2048×512 = 1,048,576 的 87.5%）
掩码槽   = 1,048,576 − 917,760 = 130,816（~131k，§4.6 热点论证同源）
原子字节 = 917,760 × 576 × 4B = 2.12 GB（全槽口径 2.42 GB，任务书 ~2.4GB 锚点一致）
每 task  = 917,760 / 2048 ≈ 448 有效行 × 2304B ≈ 0.985 MB
```

每行散射粒度 = [576] fp32 = 2304B（G5 的 512 段 + G6 的 64 段，两段分别 atomic 或 UB 拼接后单发，Stage 2 二选一）。**dst 行号（idx 值）随机无连续性**——这是与 gqa_sink L437（连续行 region atomic）和 indexer_bwd L681（同段连续 topk 行 region atomic）的本质差异，region 形态均不适用，必须**逐行条件 atomic**。

### 6.2 工作集与 L2 容量（锚点）

- **L2 容量**：192MiB（`Ascend910B3.ini` `l2_size=201326592`，文档头锚点；任务书预估"请查证"已闭环）。
- **散射相关常驻工作集**：ws_dkv 终态 4.5MB + KV 原始 2.25MB + indices 4MB ≈ **10.75MB ≪ 192MiB**——散射热数据（终态累积区 + gather 源）全量驻 L2，2.12GB 原子流量**理论上由 L2 吸收、HBM 仅见终态 4.5MB**。
- **谨慎口径（fwd rev1 教训，不依赖临界断言）**：L2 吸收率受 hash 冲突 / 其它 ws 流量干扰，预算按**单核 MTE 实测口径**核算（AIV 行读 ~26GB/s/核、AIC ws→L1 ~50-60GB/s/核、L2 中转 ~1TB/s 估算，fwd DESIGN §9.5 锚点），L2 吸收仅作交叉验证（§6.4/§13）。

### 6.3 三方案流量估算（bwd 段全序列口径，L2/HBM 分列）

**方案 1（选定）：fp32 GM 终态区 + 逐行条件 `T.tile.atomic_add` + postprocess 单次 cast**

```text
路径：G5/G6 acc_dkv [BN,576] fp32 --L0C 整块直写--> workspace_dkv 贡献槽（[20,BN,576] per-core 覆写）
      --AIV 按 mask 逐行读 UB + T.tile.atomic_add--> ws_dkv[bs*+idx, 0:576] 终态 fp32
      --postprocess--> dkv bf16
先例：indexer_bwd L688-697 完整同构（L0C→ws_dkf 贡献槽→if 条件→逐行 atomic，
      两 AIV 分摊行 gi = vid*(topk//2)+i）；T.tile.atomic_add fp32 GM dst 已查证（§3.6-4）
```

| 流量项 | 字节（全序列） | 落点 |
|--------|---------------|------|
| 贡献槽写（L0C 直出，C 侧） | 2048 task × 1.125MB = 2.31GB | L2（槽 20×128×576×4=5.9MB 常驻） |
| 贡献槽读（散射源，AIV 侧） | 2.31GB | L2 |
| 终态原子写（有效行） | 2.12GB | L2（终态 4.5MB 常驻） |
| postprocess 读终态 + 写 bf16 | 4.5MB + 2.25MB | HBM |
| **L2 合计** | **~6.7GB** | @bwd ~7.3ms → ~0.92TB/s，贴 L2 ~1TB/s 估算口径 ⚠️ 交叉验证项 |
| **HBM 合计** | **~6.75MB** | 可忽略 |

**方案 2（fallback）：分核私有 ws + 无原子散写 + 归约 cast**

```text
路径：每核私有 ws_dkv_private[cid] [S,1,576] fp32（20×4.5MB=90MB GM）——普通 store 累加（每核
      独占区无竞争，但跨 task 仍需读-改-写：task 内 16 块贡献对同行的多次累加可留 L0C/UB，
      每 task 结束一次性 store 私有区）→ 归约 pass：读 20 份求和 → cast bf16
```

| 流量项 | 字节 | 落点 |
|--------|------|------|
| 私有区写（task 末 store） | ~2.31GB（有效贡献，非原子） | L2 |
| 私有区写前读（同行二次累加，核内 task 序列对同行命中率 ~1/102） | ~2.2GB 读 | L2 |
| 归约 pass 读 90MB + 写 4.5MB + cast 2.25MB | ~97MB | L2/HBM |
| **L2 合计** | **~4.7GB** | 低于方案 1（无原子序列化） |
| **GM workspace** | **90MB**（vs 方案 1 的 5.9MB） | 挤占 L2（90/192MiB=46%）+ host 分配 |

**方案 3（拒绝）：行中心重构（按 dKV 行重组循环）**

每 dKV 行 r 的贡献需收集全部引用 (t, j)：期望引用数 = S×topk/S = topk=512 次/行 → 每行需 512 个 token 的 q [576]、dO [512]、P/dP [64]——P/dP 未保存：(a) 重算 = 重做全部 score GEMM + softmax 链（计算量翻倍）；(b) 物化 P/dP 到 GM = [S,H,topk] bf16 ×2 = 537MB 附加流量；(c) q/dO 按行 gather 重读 = 512 行 × (1152+1024)B × 2048 dst 行 ≈ 74GB。三径皆不可行，**拒绝**（§4.5 决策，数据在此闭合）。

### 6.4 选型论证与带宽核算（方案 1 落地条件）

**方案 1 基线（AIV 逐行散射，两 AIV 分摊块内行）**——散射侧 V 侧带宽核算（锚点：AIV ~26GB/s 实测，fwd §9.5）：

| 侧 | per-task 流量（MB） | 每核 102.4 task 总量 | @bwd 7.3ms 需求 | 实测能力 | 结论 |
|----|---------------------|---------------------|-----------------|----------|------|
| C 侧（AIC） | ws_kv 读 0.5625 + ws_s 写 0.125 + ws_p/dp 读 0.125 + ws_dq 写 0.5625 + **ws_dkv 贡献写 1.125** + Q/dO 0.203 = 2.70 | 276MB | **37.8GB/s** | 50-60GB/s | ✓ 余量 25-37% |
| V 侧（每 AIV） | gather 读 0.28125 + ws_kv 写 0.28125 + ws_s 读 0.0625 + ws_p/dp 写 0.0625 + ws_dq 读 0.28125 + **ws_dkv 贡献读 0.5625 + 原子写 0.49** + dq 终写 0.07 ≈ 2.02 | 207MB | **28.4GB/s** | 26GB/s | ⚠️ 超 9%，贴线 |

**优化菜单（按序启用，Stage 2/3 验证）**：

1. **workspace_dq bf16 档**：贡献块 [64,576] cast bf16 中转（AIV 读半片 fp32 累加，16 块累加舍入 16 次——主仓 dP 本就 bf16 进 GEMM、acc 链 fp32，语义偏差仅贡献块出 L0C 时一次舍入；精度验证过则启用）→ V 侧 −0.14MB/task → **26.4GB/s 贴线达标**；C 侧 −0.28MB → 33.9GB/s。
2. **L0C 直 src 逐行 atomic 档**（indexer_bwd L681 已验证 L0C direct src，但其形态为连续 region；逐行 L0C 切片 src 待 Stage 2 DSL 验证）：免 ws_dkv 贡献槽读——V 侧 −0.56MB → **20.5GB/s ✓✓**；C 侧贡献写改原子直写（~同量），mask 行向量经 ws_mask [20,BN] 传递 AIV→C。若 DSL 不支持则放弃，基线不受影响。
3. **散射/累积 AIV 负载再平衡**：vid0 偏重散射、vid1 偏重 dq 累积的动态分工（两链带宽 0.99 vs 0.35MB/task 不均）——实现期按实测微调。

**方案 1 vs 方案 2 权衡**（§4.5 决策的量化闭合）：方案 2 免原子序列化（L2 流量 4.7 vs 6.7GB）但 +90MB workspace（L2 占用 46%，挤压 ws_kv/ws_dkv 终态等热工作集）+ 归约 pass 额外 kernel + task 内块间累积需驻留复核；方案 1 工作集 5.9MB、零额外 pass、语义与主仓逐字一致（fp32 原子 + 单次舍入锚点）。原子吞吐风险由优化菜单 2/3 与 fallback 判据兜底。

### 6.5 fallback 判据（方案 1 → 方案 2 切换）

- **精度触发**：方案 1 出现原子丢失/错序导致的 dkv 精度 FAIL（双后端任一）且排除实现 bug → 查 `T.tile.atomic_add` fp32 dst 语义（ascend_tile.py:313-330 已查证支持，若运行时行为异常上报平台问题）→ 切方案 2。
- **性能触发**：bwd 段实测散射相关耗时（msprof task duration 中 atomic 段）> 2.5ms（占 bwd 预算 1/3）或 V 侧带宽 > 30GB/s 无法通过优化菜单压回 → 切方案 2（预期 L2 流量 −30%，代价 90MB ws + 归约 pass ~100us 级）。
- **切换成本**：方案 2 复用方案 1 的全部 GEMM/向量链结构，仅散射出口与 postprocess 改为归约+cast 合并 kernel——接口签名不变，风险局部化。

## 7. OOB 防护方案

### 7.1 OOB 源头分类（值域穷举）

| 源头 | 产生条件 | 示例值 | 出现规模（考核 shape） |
|------|---------|--------|----------------------|
| 尾部填充 | t < topk 时 `randperm(max(1,t))[:topk]` 不足 topk 个，槽保持初始填充 S | S=2048 | 130,816 槽（§6.1） |
| 显式 -1 | 接口契约 mask 判据含 `idx != -1`（fwd/topk 阶段语义残留） | −1 | 契约允许，测试未必构造 |
| 任意越界 | 防御性覆盖：任何 int32 值均须不崩溃 | 任意 | 按"任意合法输入须正确"硬约束（§9 任务书） |

防护双端：gather 读侧（§7.2）+ scatter 写侧（§7.3）；**两端的 mask 判据一律用原始 idx 评估，clamp 仅用于寻址**——这是本方案的正确性枢纽。

### 7.2 gather 侧防护：clamp 寻址 + mask 归零（时序）

AIV 侧每索引块（BN 行）的向量链时序（gqa_sink L393-423 / fwd §8.4 同构）：

```text
① idx_ub  ← T.copy(Indices[s_row, g, i0 : i0+BN])              # 原始 idx 进 UB
② mask_2d ← (idx_ub <= s_i) & (idx_ub != -1)                   # T.tile.compare ×2 + bitwise_and
                                                                 #   ——用【原始 idx】，clamp 前评估
③ rows_ub ← clamp(idx_ub, 0, seg_len-1)                        # T.tile.min/max 向量化；
                                                                 #   seg_len = eos - bos（段内上界）
④ gather:  kv_ws[j, :] ← T.copy(KV[bos + rows_ub[j], :])       # 行号 ∈ [bos, eos-1] 恒合法
⑤ P/dP 链: P = select(mask_2d, exp(s·sm_scale − lse), 0)        # T.tile.select（§7.4）
```

- **③ 的上界是 seg_len−1（段内）而非 S−1（全局）**：主仓 gather 寻址为 `KV[bos + idx]`，段内 clamp 与寻址基匹配；填充值 S > 任意 s_i ≤ seg_len−1 → mask ② 恒 false，与 clamp 后载入哪一行无关。多段 varlen 时填充值语义不变（idx > s_i 即掩码）。
- **−1 防护**：clamp 前 mask 已判 `!= -1`（②）；clamp 后 −1→0 载入行 0 合法有限值，贡献被 ⑤ 归零。fwd DESIGN §5.4 同款（"-1 → clamp 至行 0，贡献被 mask 置 0"）。
- **重复索引不去重**：合法重复 idx 各自 gather、各自计贡献（§2.5 语义锚点），clamp/mask 不影响重复值（重复值经 ②③ 后仍为合法行、mask true）。
- gather 行宽 576×2B=1152B 连续（≥32B 聚合阈值），行级 `T.copy` GM→UB（fwd §5.5 [REORDER-COST] 同款口径）。

### 7.3 scatter 侧防护：掩码行条件跳过（禁止 OOB 写）

AIV 侧散射循环（indexer_bwd L688-697 完整同构，两 AIV 分摊 BN 行）：

```text
for j in T.serial(BN):                    # 编译期常量（§3.3）
    if mask_row[j]:                       # ①原始 idx 的 mask（§7.2 ②同源），
                                          #   per-row 动态 if——unsorted:42 / indexer_bwd L695 先例
        T.copy(workspace_dkv[cid, j, :], row_ub)          # 贡献行 [576] fp32 → UB
        T.tile.atomic_add(ws_dkv[bos + idx_ub[j], 0], row_ub)   # ②寻址行号也用【原始 idx】
```

- **掩码行零写入**（连 0 贡献都不写）：省 130,816 行 × 2304B ≈ 0.3GB 无效原子流量（§6.1 全槽与有效口径之差），且**根绝 OOB 写**——主仓 L232-241 的无条件散射（idx=S 越界写，GPU 侥幸不崩）在 NPU 上为必须消除的未定义行为。
- **② 用原始 idx 而非 clamp 后 rows**：mask true 的行 idx ∈ [0, s_i] ⊂ [0, seg_len−1] 天然合法，clamp 在 scatter 侧是冗余的（写侧合法性由 mask 保证，读侧合法性由 clamp 保证——双保险各司其职）。
- **"零填充 + clamp 到固定行"方案否决复述**（§4.6）：~131k 掩码槽的零贡献集中打到单行 → 同地址原子热点（L2 行锁序列化），且引入无意义的 0.3GB 原子流量。
- **pair 贡献原子性**：G5 pair 双发在 L0C 内完成求和后整行散射（dPᵀ@q + Pᵀ@dO 一次 atomic），掩码行两贡献分量皆 0、整行跳过——不会出现"半贡献"散射。

### 7.4 0×NaN / inf 防护机理：select 语义 vs GPU 乘法掩码

GPU 主仓的掩码路径是"**乘 0**"：P = exp(…)×mask → 若 gather 垃圾值使 score=±inf/NaN，exp 产出 inf/NaN，0×inf=**NaN 污染**（主仓 §2.3"侥幸通过"成立的隐含前提是垃圾值为有限值）。本设计改用 **select 语义**（gqa_sink L423/458 `T.tile.select(dst, mask, src, 0.0, "VSEL_TENSOR_SCALAR_MODE")` 同款）：

| 链路 | 掩码位置 | 垃圾值传播分析 |
|------|---------|---------------|
| P 链 | `P = select(mask, exp(s·scale−lse), 0)` | masked 位置 exp 结果（可能 inf/NaN）**被选择丢弃**而非乘 0——无 0×inf 路径 |
| dP 链 | `dP = select(mask, P·(acc_dp−Δ)·scale, 0)`（P 已 select 归零，本层冗余防御） | P=0（有限）× acc_dp 垃圾（**有限**，§7.2 clamp 保证载入合法行真实数据）= 0 ✓ |
| GEMM | dP/P 的掩码列/行为精确 0 | 0 × 有限操作数 = 0，贡献恒 0 ✓ |

关键不变式：**clamp 保证一切进入 GEMM 的操作数为合法内存的有限值**（输入 kv 本身含 NaN 属输入域问题，GPU 版同样不定义）+ **select 保证掩码位置不产生 0×inf**——双机制下 NaN/inf 传播路径被完全切断，强于 GPU 语义。

### 7.5 与 GPU 语义等价性论证

**命题**：对任意有限值输入，NPU 实现的输出 (dq, dkv) 与 GPU 理想语义（OOB 项贡献为 0、无越界写观测）逐元素相等（数值精度差异除外）。

论证（对全部贡献通道穷举）：

1. **P/dP 掩码项**：GPU——OOB j 的 P=0（−inf 预填→exp=0）、dP=0（P=0 × 有限垃圾）；NPU——select 直接置 0。两侧 OOB 项系数恒等 = 0。
2. **dq 通道**：dq[t] = Σ_j dP[t,j]·KV 行——OOB j 项 dP=0，两侧贡献 0；合法 j 项操作数相同（clamp 不改合法行号）、GEMM 同构 → 贡献相等。
3. **dkv 通道**：dKV[r] = Σ_{(t,j): idx=r} (dPᵀ@q + Pᵀ@dO)[j]——OOB j 项 P=dP=0 双分量皆 0（GPU 写 0 到 OOB 地址**不可观测**；NPU 跳过写）；合法 (t,j) 项两侧同构。
4. **重复索引**：不去重语义两侧一致（各计贡献，§7.2）。
5. **差异面**：GPU 对 OOB 地址的写行为未定义（主仓侥幸不崩）；NPU 零写。可观测输出（dq/dkv 合法域）无差异。

### 7.6 与 varlen / 动态维的交互

- mask 上界 s_i 与 clamp 上界 seg_len−1 均为**运行时标量**（TokenIndices/Offsets GM 标量读，§2.4）——`T.tile.compare` 与 `T.tile.min/max` 的操作数为 UB 广播标量，无动态循环边界（§3.3 约束不触碰）；
- 多段（B>1）时每 task 的 bos/eos/seg_len 独立取值，⑤/③ 公式逐 task 求值，段间无状态残留（persistent 轮询下 task 独立性由 §9 流水边界保证）；
- **首行自注意**（fwd 强制 j=0 对 s_i=0 有效）：合法路径不受 clamp/mask 影响（idx=0 ∈ [0, s_i]）；单 token 段（seg_len=1）时 topk 全掩码除 j=0，退化正确。

测试锚点：OOB 专项用例（t<topk 强制尾部填充 + 全 -1 段 + 极端值 idx）与 varlen 用例在 §12 L0 计划（块 D）落表，本节机制为其设计依据。

## 8. L0C/UB/L1 内存规划

> 参数承接 §5.5：BN=128、DSEG=2、Hseg(=padded_H)=64、v_block=32、grid=20；平台每核 L0C 128KB / L1 512KB / UB 192KB（文档头锚点）。§5.2 表的 L0C/L1/UB 三行占用在 本节逐层展开闭合。

### 8.1 L0C（128KB）：GEMM 累加器分时复用（alloc_fragment，fp32）

每索引块 i（NI=4 循环内）的 GEMM 程序序：G1 → 拷出 ws_s → G2（覆写同槽）→ G3×2 段 → G4 → G5 pair×(2 段×2 发) → G6。槽位清单：

| 槽 | shape | 容量 | 服务 GEMM | 生命周期（块 i 内） |
|----|-------|------|-----------|--------------------|
| acc_s | [64, BN=128] | 32KB | G1 / G2 分时共槽 | G1 出→`T.copy` 到 workspace_s→死；G2 覆写→拷出（dO·v 结果）→死 |
| acc_dq | [64, 256] | 64KB | G3（每 D 段一发） | 段出→`T.copy` 到 workspace_dq 列段→死 |
| acc_dq_tail | [64, 64] | 16KB | G4 | 出→拷到 workspace_dq 列段 [512:576]→死 |
| acc_dkv | [BN=128, 256] | **128KB** | G5 pair（2 段双发同槽累加，`init=True/False`） | 段出→L0C 直写 workspace_dkv→死；**独占段** |
| acc_dkv_tail | [BN=128, 64] | 32KB | G6 | 出→直写 workspace_dkv 尾段列→死 |

**生命周期峰值分析**（TVM 作用域 + MEMORY_PLANNING liveness 自动分时的设计依据）：

```text
阶段①  G1+G2           ：acc_s 32KB                          → 32KB
阶段②  G3+G4           ：acc_dq 64 + acc_dq_tail 16          → 80KB（acc_s 已死可复用）
阶段③  G5 pair         ：acc_dkv 128KB（① ② 槽全死）        → 128KB 贴线独占段
阶段④  G6              ：acc_dkv_tail 32KB（acc_dkv 已死）   → 32KB
峰值 = 128KB / 128KB 上限 = 100%（阶段③ 独占）
```

- **贴线成立条件**：G5 独占段内无其他 L0C 活槽——G1/G2 槽在阶段③前已死、G3/G4 槽已死，程序序保证；fragment 分配依赖 liveness 分析（4-True MEMORY_PLANNING）。
- **冲突回退**：任一后端 fragment 分配失败 → **BN=64 档**（§5.2 回退档：G5 [64,256]=64KB，峰值降至 80KB/63%，发射数 40→80 仍可接受）。不推翻 §5.2 决策，仅引用其回退路径。
- L0C C 起始 256 元素对齐（§5.3 注 C 约束）：64×128 / 64×256 / 128×256 / 128×64 均 256 倍数 ✓。

### 8.2 L1（512KB）：Cube GEMM 操作数（alloc_shared→L1）

| buffer | shape | dtype | 容量 | 用途（对应 §5.1 GEMM） |
|--------|-------|-------|------|------------------------|
| q_l1 | [64, 576] | bf16 | 72KB | G1 的 A / G5 的 B 列切片 / G6 的 B（每 task 装载一次，GM→L1 直连） |
| do_l1 | [64, 512] | bf16 | 64KB | G2 的 A / G5'（Pᵀ@dO）的 B 列切片 |
| kv_l1 | [BN=128, 576] | bf16 | 144KB | G1/G2 的 B（ᵀ）/ G3/G4 的 B 列切片（每索引块整块 ws_kv→L1） |
| dp_l1 | [64, 128] | bf16 | 16KB | G3/G4 的 A / G5 的 Aᵀ（normal + transposed 双用，indexer_bwd L371 dlog_l1 先例） |
| p_l1 | [64, 128] | bf16 | 16KB | G5' 的 Aᵀ |
| **合计** | | | **312KB（61%）** | 余 200KB（双槽流水升级余量，§9.3） |

与 §5.2 表 BN=128 行 "72+64+144+32=312KB" 逐项一致（32 = dp_l1 16 + p_l1 16）。512B 对齐：各 buffer 行宽 576×2B=1152B、512×2B=1024B、128×2B=256B——kv_l1/q_l1/do_l1 行起点均 512B 倍数 ✓；列切片段起始对齐已在 §5.3 核验。

### 8.3 UB（192KB / 每 AIV，v_block=32 head 半片）

| buffer | shape | dtype | 容量 | 用途 |
|--------|-------|-------|------|------|
| acc_dq_ub | [32, 576] | fp32 | 72KB | 跨 NI=4 块 dq 累积（§3.4 决策：dq 累积移出 L0C；task 尾 cast bf16 终写 dq） |
| s_ub | [32, 128] | fp32 | 16KB | workspace_s 半片读入 → P 链（`exp(s·sm_scale − lse)`） |
| dv_ub | [32, 128] | fp32 | 16KB | G2 输出半片读入（dP 链：`P·(dO·v − Δ)·sm_scale`） |
| p_bf16_ub | [32, 128] | bf16 | 8KB | P cast 写出槽（→ workspace_p） |
| dp_bf16_ub | [32, 128] | bf16 | 8KB | dP cast 写出槽（→ workspace_dp） |
| idx_ub / rows_ub / mask_ub | [128] | int32 / fp32 | ~1.5KB | OOB 防护链（§7.2 ①②③；mask 用原始 idx，一维 [BN] 对半片 head 广播） |
| lse_ub / delta_ub | [32] | fp32 | 0.25KB×2 | GM 标量读半片（每 task 头部一次，§10.3） |
| row_ub | [576] | fp32 | 2.25KB | 散射贡献行 staging（§7.3：workspace_dkv 行 → UB → atomic） |
| kv_row_ub | [576] | bf16 | 1.125KB | gather 单行 staging（GM→UB→ws_kv，fwd kv_ub 同款） |
| misc（cast / 标量链预留） | — | — | ~16KB | sm_scale / 段标量暂存 |
| **合计** | | | **~140KB（73%）** | 余 ~52KB |

与 §5.2 表 BN=128 行 "~140KB ⚠" 一致；73% 占用（fwd acc_o 98% 贴线的显著改善——dq 累积半片 [32,576]=72KB vs fwd [64,512]=128KB，head 半片分摊红利）。**缓解预案**（若 MEMORY_PLANNING 复用后仍超 192KB）：s_ub 在 P 链完成后即为死槽，复用为 dv_ub 读入（生命周期天然分离，首版内建）；row_ub 与 misc 共槽；最后手段 v_block 相关小槽下放 GM ws。BN=64 档 UB 需求同步降（s/dv/p/dp 槽减半，§5.2 表）。

### 8.4 workspace 清单（GM 显式 kernel 参数，命名含 "workspace"，host `torch.zeros` 分配）

| workspace | shape | dtype | 大小 | 方向（写→读） | 先例 |
|-----------|-------|-------|------|---------------|------|
| workspace_kv | [20, 128, 576] | bf16 | 2.95MB | V gather 半区 → C | fwd §4.4 同款 |
| workspace_s | [20, 64, 128] | fp32 | 0.66MB | C G1 → V 半片 | 同上 |
| workspace_p | [20, 64, 128] | bf16 | 0.33MB | V 半片 → C G5' | 同上 |
| workspace_dp | [20, 64, 128] | bf16 | 0.33MB | V 半片 → C G3/G4/G5 | 同上 |
| workspace_dq | [20, 64, 576] | fp32 | 2.95MB | C G3/G4 → V 半片累加 | bf16 化为优化档（§6.4 菜单 1） |
| workspace_dkv（贡献槽） | [20, 128, 576] | fp32 | 5.9MB | C L0C 直写 → V 两 AIV 分摊散射 | indexer_bwd ws_dkf 同构（L688-697） |
| workspace_dkv_final（终态区） | [S, kv_group, 576] | fp32 | 4.5MB | V `T.tile.atomic_add` → postprocess 读 | 主仓 dKV fp32 `T.empty` 语义锚点 |
| **GM 合计** | | | **~18MB** | | 远小于 L2 192MiB（§6.2 工作集论证） |

- per-core 槽 `[20, …]` 用 cid 索引、核内 task 序列覆写（fwd §8.2 同款）；终态区为全局唯一区（原子目标）。
- workspace_idx 不使用（显式参数 + 命名触发 AUTO_CV_SYNC，indexer_bwd fast 先例；若 ascendc 配对失效补 `workspace_idx`，fwd R5 同款记录）。
- workspace_dkv_final 同 kernel 内仅 AIV 写、无核内读（核间由 postprocess 消费，launch 顺序保证）→ 不产生核内 flag 配对，命名含 workspace 安全。

### 8.5 内存搬运路径（完整层级链）

```text
GM[Q/dO] ──T.copy──> L1(q_l1/do_l1)                                # 直连（连续行段，每 task 一次）
GM[Indices] ──T.copy──> UB(idx_ub) ──compare/clamp──> UB(mask_ub/rows_ub)
GM[KV] ──T.copy──> UB(kv_row_ub) ──T.copy──> GM[workspace_kv] ──T.copy──> L1(kv_l1)
L0C(acc_s) ──T.copy──> GM[workspace_s] ──T.copy──> UB(s_ub)         # G2 结果同路出 ws（dv 半片）
UB(p_bf16_ub / dp_bf16_ub) ──T.copy──> GM[workspace_p/dp] ──T.copy──> L1(p_l1/dp_l1)
L0C(acc_dq 段) ──T.copy──> GM[workspace_dq] ──T.copy──> UB ──T.tile.add──> UB(acc_dq_ub)
L0C(acc_dkv) ──L0C 直写──> GM[workspace_dkv] ──T.copy──> UB(row_ub) ──T.tile.atomic_add──> GM[workspace_dkv_final]
UB(acc_dq_ub) ──cast──> GM[dq bf16]                                  # task 尾一次
GM[Lse/Delta/TokenIndices/Offsets] ──GM 标量读──> 标量 s_i/bos/lse/Δ（§10.3）
```

跨级合法性：无 GM↔L0 直搬（L0C→GM / GM→L0C 直写为文档支持路径，indexer_bwd ws_dkf 先例）；bf16 UB→L1 不直连（经 GM workspace，§3.6-1 平台强制）；全部 `T.copy` 路径在 api-kernel-memory.md §3 路径表内（fwd §4.5 同款审计口径）。

### 8.6 双后端差异标注（§4.4 策略的逐项落地）

| 项 | ascendc | pto | 处理 |
|----|---------|-----|------|
| L0C 累加 dtype（fp32） | ✓ gemm_v0 accum_dtype | 累积 dtype 行为差异（Programming Guide） | 双后端统一显式 `accum_dtype=fp32` 声明；pto 首验项（块 D 风险表） |
| workspace 显式参数 | ✓ indexer_bwd 先例 | ✓ gqa_pto 先例 | 统一传参约定（§8.4） |
| L0C→GM 直写（acc_dkv） | ✓ indexer_bwd L688 先例 | 待 Stage 2 验证 | 失败则 pto 走 L0C→UB→GM 中转（+0.56MB/task，V 侧带宽预算 §6.4 重核） |
| `T.tile.atomic_add` fp32 GM dst | ✓ ascend_tile.py:313-330 | gqa_pto 待验 | 失败则 pto 侧切方案 2（§6.5） |
| reduce fp32 dim=-1（preprocess） | ✓ | gqa_pto 先例（输入恒整 buffer，fwd R8 规避口径） | §11.1 归约输入为整 buffer |

## 9. 同步与流水

### 9.1 Developer 模式同步边界声明

4-True pass_configs（fwd §4.7 / indexer_bwd L275-280 同款）：`TL_ASCEND_AUTO_SYNC`（核内自动同步）+ `TL_ASCEND_AUTO_CV_COMBINE`（C/V 指令分离）+ `TL_ASCEND_AUTO_CV_SYNC`（workspace 命名配对自动插 flag）+ `TL_ASCEND_MEMORY_PLANNING`（UB/L0C liveness 复用，§8 贴线关键）。**零 `T.Scope` / 零手动 `T.barrier_all/set_flag/wait_flag`**——Developer 模式硬边界（§4.3）；本算子复杂度（NI 循环内 6 类 ws 乒乓）已由 indexer_bwd（更多 ws 数量成对配对）验证该边界可行，配对遗漏风险记入块 D 风险表。

### 9.2 CV 软件流水时序（单 task、NI=4 块循环）

程序序数据依赖（C 侧 / V 侧各为一条指令流，AUTO_CV_COMBINE 分离到 AIC / 2 AIV）：

```text
C: 装载 q_l1/do_l1 ──> 块 i: ws_kv→kv_l1 ─> G1 ─> ws_s 写 ─> G2 ─> (dv 出 ws)
                     ─> [等 ws_p/dp i] ─> G3/G4 ─> ws_dq 写 ─> G5 pair ─> G6 ─> ws_dkv 写
V: 块 i: idx/mask/gather ─> ws_kv 写 ─> [等 ws_s i] ─> P/dP 链 ─> ws_p/dp 写
   ─> [等 ws_dq i] ─> acc_dq_ub += ─> [等 ws_dkv i] ─> 逐行条件散射
```

- **稳态重叠窗口**：V 块 i 的向量链与 C 块 i+1 的 GEMM 并行（数据经 ws 传递，AUTO_CV_SYNC 的 flag 保证写读序）；首块 V 需等 C 的 G1（流水建立开销 ~1 块，占 NI=4 的 25%——BN 加宽后块数少，建立开销占比可接受，§13 计入）。
- **C 侧最长无依赖段**：G1（依赖 ws_kv i，V 侧 gather）之后 G2→G3/G4 前需 ws_p/dp i（V 的 P/dP 链）——C 与 V 互为生产者/消费者，双向握手 5 路 ws（§9.4 表）。
- **主仓对照**：主仓 num_stages=0 无软件流水（GPU 靠 warp 并行硬件形态，§2.5）；Ascend 以 CV 分核 + ws flag 流水等效替代，语义不变机制不同。

### 9.3 双缓冲与流水加深（可选项，Stage 2/3 按实测启用）

| 项 | 方案 | 收益 | 代价 | 结论 |
|----|------|------|------|------|
| kv_l1 双槽 | L1 余 200KB → kv_l1×2（合计 456KB / 89%） | C 侧块间 ws_kv 装载与 GEMM 重叠 | L1 贴线 | 可选（首版单槽） |
| ws_kv 乒乓 | workspace_kv 双套 [2,20,…]（+2.95MB GM） | V gather 块 i+1 与 C 读块 i 重叠 | flag 计数 ×2 | 可选（indexer_bwd 双槽先例） |
| UB 死槽复用 | s_ub（P 链后死）复用为 dv_ub 读入 | UB 峰值 −16KB | 无 | **首版内建**（§8.3 预案） |

基线 = 单槽 + 程序序流水（AUTO pass 自动完成）；加深项均为局部参数化改动，不改 kernel 结构——与 §6.4 优化菜单同属 Stage 2/3 实测驱动的递进项。

### 9.4 同步点清单（语义点位；flag 由 AUTO_CV_SYNC 自动插入）

| # | workspace | 写者 | 读者 | 粒度 |
|---|-----------|------|------|------|
| 1 | workspace_kv | V（两 AIV 各半区行） | C（整块→L1） | 每索引块 |
| 2 | workspace_s（含 G2 的 dv 同路） | C | V（head 半片） | 每索引块 |
| 3 | workspace_p / workspace_dp | V（head 半片） | C（→L1） | 每索引块 |
| 4 | workspace_dq | C（列段） | V（head 半片，读累加） | 每索引块 |
| 5 | workspace_dkv | C（L0C 直写） | V（两 AIV 分摊行散射） | 每索引块 |
| 6 | dq 终写（acc_dq_ub → GM） | V（半片） | —（核外） | 每 task，程序序 |

V→C 方向两 AIV 均写（半区/半片拼接；MIX_AIC_1_2 的 1_2 flag 握手，indexer_bwd L127 "both AIVs write every cross-core workspace" 同款）。

### 9.5 挂死风险与排查（HISA 教训承接）

- 症状 kernel 挂死（flag 不配对）→ 核对本表 #1-#5 的 buffer 命名前缀含 "workspace" 子串、写读方全覆盖；fwd §8.6 同款排查路径。
- pto 后端 AUTO_CV_SYNC 行为以 gqa_pto 先例为准；pto 侧配对异常单独记录（不阻塞 ascendc 主路径，§4.4 双后端策略）。
- 跨 kernel（preprocess→bwd→postprocess）：host launch 顺序天然串行（§11.3），无核间同步需求。

## 10. 动态 shape 与 varlen 方案

### 10.1 符号维与编译期常量划分

| 维 | 声明 | 说明 |
|----|------|------|
| S（packed 行数） | `T.symbolic`（主仓 `T.dynamic("S")` L115 的 Ascend 对应，indexer_bwd L330 `num_blocks = T.symbolic` 先例） | 运行时任意 ≥1，task 总数 = S×kv_group |
| B_plus_one | `T.symbolic` | ≥2（offsets 长度） |
| S_kv | 不单列符号：host `assert S == S_kv`（主仓 L262 原样）→ kernel/postprocess 复用 S | bwd 与 postprocess 的行空间一致 |
| H / D=512 / D_tail / topk / kv_group / BN / DSEG / v_block / grid_size | 编译期常量（builder 参数） | 进 `_BUILD_CACHE` 编译键（reducesum L809 先例），一次编译覆盖任意 S/B |

主仓 kernel 内 `H = H_kv; padded_H = max(next_power_of_2(H_kv), 16)`（L130-131）原样继承：考核 shape H=64 / kv_group=1 → H_kv=64 → padded_H=64 即 GEMM 的 M 维（§3.5 表）。

### 10.2 host 分派与 TokenIndices

- `prepare_token_indices(offsets)` 原样复用（ref_gpu/index.py L80-82 `tensor_cache` 版本）：CPU 构造 [S,2] (batch_id, 段内位置) + 一次 H2D——**与 fwd 共用同一函数与缓存**（同 offsets 命中同一 `tensor_cache` 条目，fwd/bwd 串联调用零重复构造，do_bench 稳态口径关键，§1.5）。
- host 入口程序序（主仓 L254-278 原样）：contiguity 断言 → `D = 512` 硬编码 / `D_tail = 576−D` / `kv_group`、`topk` 从 shape 提取 → token_indices → `delta is None` 则 preprocess → bwd → postprocess。
- `_CachedLauncher`（参数 marshal 缓存 ~13us/launch，indexer_bwd L170）+ `_BUILD_CACHE` 按 `(H, DQKV, topk, kv_group, target)` 键缓存——ascendc / pto 双份（§4.4）。

### 10.3 varlen 段语义进 kernel（§3.3 决策展开）

每 task 头部一次 GM 标量读（fwd §4.5 同款路径）：

```text
tid → s_row = tid % S, g = tid // S        # 行优先分解：相邻 task 同段聚集，L2 局部性有利
b_i, s_i = TokenIndices[s_row, 0], TokenIndices[s_row, 1]
bos, eos = Offsets[b_i], Offsets[b_i + 1]  # seg_len = eos − bos
lse[s_row, v0*32 : (v0+1)*32] / Delta 同址半片读（fp32 标量行）
```

- **段边界不进循环结构**（§3.3）：mask 上界 `s_i` 与 clamp 上界 `seg_len−1` 均为运行时标量，经 `T.tile.compare` / `T.tile.min/max` 的 UB 广播标量参与向量链（§7.6），无动态循环边界。
- persistent 轮询：`ntask = T.max(0, T.ceildiv(total − cid, grid))` + `T.serial(ntask)` 运行时外层（reducesum L261-265 已验证形态）；NS/BS/D 段循环全为编译期常量。
- task 独立性：每 task 的 bos/eos/seg_len 独立取值，段间无状态残留（persistent 下的正确性由 §9.4 同步点表每块粒度配对保证——acc_dq_ub 每 task 头部清零）。

### 10.4 PTO GM 标量读风险与 fallback

fwd R2 同款：pto 的 GM 标量读（codegen_ascend_pto.cc L802-812）已实现但无端到端 varlen 先例。**fallback**：host 预展开 RowMeta 表 [S, 4]（b_i, s_i, bos, eos）替代 TokenIndices+Offsets 两次读——kernel 单表读，语义不变；Stage 2 以 5 行最小复现 kernel（读标量→`T.printf`）先验，失败即切表。记入块 D 风险表。

### 10.5 kv_group > 1 通用路径（考核 shape 外的支持域）

task = (s_row, g) 分解与 KV/Indices 的 g 维寻址已参数化（§10.3）；`H_kv = H // kv_group < 16` 时 `padded_H = 16` 兜底（GEMM M=16 分形下限 ✓，主仓 L131 原样）。**padding head 行语义**（H_kv < padded_H 时尾行装载与贡献归零处理）继承主仓 L149 起的装载实现，Stage 2 逐行核对——考核 shape kv_group=1 无 pad（§3.5 残余风险记录，L1 扩展验证项，不展开设计）。

## 11. preprocess / postprocess 方案与 fwd 外部依赖策略

### 11.1 preprocess：AIV 纯带宽 kernel（k1）

主仓形态（§2.1）：`T.Kernel(H, ceildiv(S,32))` fragment [32,32] 乘加 + `T.reduce_sum` → Delta [S,H] fp32；读 O+dO 约 268MB、写 0.5MB，纯带宽（§3.2 AIV-only 任务）。

Ascend 方案（**平铺行块**，替代主仓 (h, 行块) 2D grid 的 strided 访问）：

- **形态**：`T.Kernel(grid=40, threads=1, is_npu=True) as (cid)`——AIV-only，40 AIV 全用（gqa_sink Kernel 2 L317 `threads=1` 先例，§3.2）。
- **任务划分**：O/dO [S,H,512] contiguous → 视作平铺 [S·H, 512]，任务 = 64 平铺行块（每行即一个 (s_row, h) 对，输出 Delta[s,h] 行独立，无跨任务归约）；total = ceildiv(S·H, 64) = 2048（考核 shape），40 AIV striped 轮询（§10.3 同款）。
- **核内**：o_ub/do_ub [64, 512] bf16 各 64KB（UB 128KB/67%）→ `T.tile.mul` fp32 升精度乘 → `T.reduce_sum` dim=-1（输入恒整 buffer，pto R8 规避口径，§8.6）→ delta [64] fp32 写 GM（连续段）。
- **带宽**：268MB 读 + 0.5MB 写；40 AIV × ~26GB/s 行读 = 1.04TB/s 聚合需求 → 单核口径下限 ~260us；HBM 聚合上限交叉验证与预算占位见 §13（块 D）。全部访问为连续大块（64×1KB/张量），无 strided（优于主仓 h 列跨步形态）。

### 11.2 postprocess：AIV cast kernel（k3）

主仓 L48-74：`T.Kernel(ceildiv(S_kv,64), kv_group)` [64, 576] fp32→bf16 `T.copy` cast（单次舍入语义锚点 §1.3）。

Ascend 方案：AIV-only persistent grid=40 同形态（§11.1 启动形态）；任务 = (64 行块, g)；读 workspace_dkv_final [64,576] fp32（2304B/行连续）→ UB → `T.copy` auto-cast bf16（fwd §4.5 auto-cast 先例）→ 写 dkv [64,576] bf16。流量 4.5+2.25MB → 40 AIV 下 ~10-30us 量级（§13 计入）。**cast 单次性**：fp32 终态一次转 bf16（主仓语义锚点；方案 1 散射全程 fp32 原子，无中间舍入，§4.5）。

### 11.3 三段串行与计时口径

host 程序序：`delta is None` → k1 → k2 → k3 三次 launch 串行（主仓 L273-276 原样）；kernel 间数据依赖（Delta→bwd、ws_dkv_final→postprocess）由 launch 顺序保证，无额外同步。`do_bench` 口径（rep=100/warmup=250，`delta=None` 每次含 preprocess）与主仓逐字一致（§1.5）；`delta` 显式传入时跳过 k1（接口契约，bwd 计时不含 preprocess 的调试口径）。

### 11.4 fwd 外部依赖策略（并行会话交付物消费）

- **消费契约**（主仓 L310-312 口径，不得更改）：test_sparse_mla_bwd.py `from sparse_mla_fwd import sparse_mla_fwd_interface`；`tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, offsets)`——与 fwd DESIGN §11.3 公开符号一致 ✓。**禁止**本任务实现/修改 sparse_mla_fwd.py（任务书 §4 硬约束）。
- **fwd 状态**：DESIGN_sparse_mla_fwd.md 已落盘；sparse_mla_fwd.py ⬜ 待实现（fwd §11.2）——存在时间窗风险。
- **过渡 fallback**（文档化，仅测试文件内，最终验收前移除）：

```python
try:
    from sparse_mla_fwd import sparse_mla_fwd_interface
except ImportError:
    sparse_mla_fwd_interface = _torch_fwd_fallback   # 精确 PyTorch：o = ref fwd（bf16）；lse = fp32 全量 logsumexp
```

  `_torch_fwd_fallback(q, kv, indices, offsets) -> (o, lse)`：o 取 fwd DESIGN §9.1 的 `ref_sparse_mla_fwd_interface`（device 参数化，bf16 输出）；lse 按同 mask 语义（causal+topk，越界列置 −inf）fp32 全量 `logsumexp`——两者均为 golden 路径的精确数学，**与 golden autograd 内部同源**（golden L288 亦用 ref fwd），被测 bwd 与 golden 消费相同 (o, lse) → 判据不受 fallback 影响。
- **切换判据**：fwd sparse_mla_fwd.py 落盘且其 L0 双后端通过（fwd §11.6 步骤 5）→ 移除 fallback（断言 import 成功）→ bwd L0 复跑双后端通过 = 最终验收。
- **lse 数值差风险**：fwd kernel 在线 softmax 与 PyTorch 全量 logsumexp 的归约序差 δ → bwd 的 P 乘性偏 exp(δ)；fwd L0 已含 Lse fp32 容差门禁（fwd §9.3）→ δ 有界，由 bwd 判据（eps=1e-4 相似度 + §12 混合容差）覆盖。记入块 D 风险表。
- **性能验收不依赖 fwd**：7680us 口径的 fn 只调 `sparse_mla_bwd`（o/lse 为预计算输入，§1.5）——性能验证可先于 fwd 就绪执行。

## 12. 精度容差与验证方案（含 L0 门槛测试计划）

### 12.1 双判据体系（主仓全局相似度为主 + 按 dtype 混合容差为辅，均阻塞）

- **主判据**（与主仓逐字一致，§1.5）：`assert_tensors_similar(eps=1e-4)`——全局归一化点积相似度 `sim = 2⟨x,y⟩/(‖x‖²+‖y‖²)`（double 精度），要求 `0 ≤ 1−sim ≤ 1e-4`；dq / dkv 分别判；全零张量特判通过（utils.py:48-73）。全局能量口径对离群点鲁棒，是主仓语义验收的权威判据。
- **辅判据**（tilelang-op-test-design 精度标准，逐元素定位）：混合容差 `|a−g| ≤ atol + rtol·|g|`，双门限 `matched_ratio ≥ required` 且 `max_abs_error ≤ limit`——主判据 FAIL 时定位误差分布（dkv 散射 / dq 累积 / 特定 head / 长尾行），双后端差异定位同理。
- 两判据均计入 L0 阻塞（任一 FAIL → `[PRECISION_FAIL]`，退出码 1）；辅判据单独 PASS 不豁免主判据。

### 12.2 按 dtype 精度标准表（混合容差，precision-standard.md §二口径）

| 量 | dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|----|-------|------|------|---------------------|------------------------|
| dq（输出） | bfloat16 | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |
| dkv（输出） | bfloat16 | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |

- **dtype 全集说明**（与 proto_sparse_mla_bwd.yaml 一致）：本算子为**固定 dtype 组合**（§1.4 断言继承），无 dtype 扫描维度——输入 q/kv/o/do `bfloat16`、indices/offsets `int32`、lse/delta `float32`；输出 dq/dkv `bfloat16`。整型无输出量（int32 索引仅作输入，精确匹配条目 0/0/0/1.0 不适用，记 N/A）；lse/delta fp32 为上游产物（fwd / preprocess），其自身精度由 fwd DESIGN §9.3 Lse 门禁与本算子 preprocess 链覆盖，不在输出判据内。
- **阈值合理性论证**：dkv 终值 = 期望 ~512 次 fp32 原子累加（§6.1 有效行次）+ postprocess 单次 bf16 舍入；dq = NI=4 块 UB fp32 累积 + 单次 cast。与 golden（autograd 内部 fp32、grad 随输入 bf16）的差异由 bf16 终端舍入（2⁻⁸ 相对量化）+ 跨核归约序差主导 → rtol 2⁻⁶（bf16 标准档）覆盖；atol 2⁻¹⁰ 保护 dkv 长尾近零行（有效贡献次数少的行，§6.1 早期 token 行）。indexer_bwd rev2 实测佐证：同为 fp32 原子 + 单次 cast 路径，ratio 0.9972 通过 bf16 判据。

### 12.3 L0 门槛测试计划（tilelang-op-test-design 场景 A，仅 L0）

**算子类别判定**（场景 A Phase 2，判断依据 §1.3/§2.2）：计算类型 = **混合（CV 融合）**——6 类 GEMM@Cube + mask/exp/dP 链@Vector + fp32 atomic scatter@AIV；复杂度 = **Fusion**（三段 host 编排 × NI 块循环 × 6 类 GEMM）；数学特征 = matmul + exp + scatter → **综合类别 Fusion**。测试聚焦：CV 交互正确性（§9.4 五路 ws 握手）、OOB 双端防护（§7）、varlen 段语义（§10.3）、fp32 原子散射精度（§4.5）。L1（不规则 shape / 短段 / 边界 t）/ L2 / Boundary 由 Stage 2 场景 B 扩展，本节不生成。

**shape 规则性口径**（"block 整除"的本算子映射）：topk % BN(=128) == 0（断言继承，§1.4）→ L0 全部用例 topk ∈ {128, 512}；S / 段长为运行时动态维（token 粒度任务无块尾，§3.5），L0 覆盖 S ∈ {128, 256, 640, 645, 2048}；H=64 / HKV=1 / DQKV=576 / DV=512 固定（host 硬编码 + 考核 shape）。

**L0 用例表**（9 条；生成流程 = 主仓 test 口径（ref_gpu/sparse_mla_bwd.py L293-320）原样 + seed 固定）：

| ID | 类别 | 配置 (B,S,HKV,topk) | 构造要点 | 覆盖目标 |
|----|------|--------------------|---------|---------|
| L0-1 | 主仓口径 | 1, 2048, 1, 512 | randn + 逐 token `randperm(max(1,t))[:topk]`、尾部填 S、offsets=[0,S]；seed=0 | 全链路考核 shape（t<511 尾部填充自然出现 130,816 槽，§6.1 同源） |
| L0-2 | 快速冒烟 | 1, 128, 1, 128 | 同上；seed=0 | 小规模 CI 冒烟（S=topk 边界） |
| L0-3 | pto 后端 | 同 L0-1，target="pto" | 同 L0-1 | 双后端（§4.4；ascendc 即 L0-1 本体） |
| L0-4 | delta 传入 | 同 L0-2，delta 显式 | delta = golden preprocess 结果 (O·dO).sum(-1) fp32 | 接口分支（跳过 k1，§11.3） |
| L0-5 | OOB 强化 | 1, 256, 1, 128 | 高填充率构造（t<127 每 token 尾部 ~50% 填充槽） | gather clamp + scatter 条件跳过（§7.2/§7.3） |
| L0-6 | -1 混入 | 1, 256, 1, 128 | 有效槽后、填充前中段插 -1 | mask `!= -1` 判据 + golden 裁剪列语义（§7.2） |
| L0-7 | 重复索引 | 1, 256, 1, 128 | 前 64 有效值复制两份填满 128 槽（重复计贡献） | 不去重语义锚点（§2.5/§7.2） |
| L0-8 | varlen 多段 | 3, 640, 1, 128 | offsets=[0,128,512,640]（段长 128/384/128） | 段边界 + 段内 mask 语义（§10.3） |
| L0-9 | varlen 尾块 | 2, 645, 1, 128 | offsets=[0,129,645]（段长 129/516，均非 BN 倍数） | 段长非整除 + t=128 段内边界行 |

判据：全部用例双判据（§12.1），dq / dkv 分别判；L0 全绿（`[PRECISION_PASS]` × 9）= Stage 2 解锁 L1/L2/Boundary 扩展的门槛。L0-3 pto 侧同等阻塞（双后端硬性要求，§4.4）。

### 12.4 golden 草案（PyTorch 参考实现，Stage 2 落地为测试文件内函数）

```python
def ref_sparse_mla_bwd_interface(q, kv, o, do, indices, lse, offsets, sm_scale=None, is_casual=True):
    # 主仓 ref_gpu/sparse_mla_bwd.py L281-290 原样移植（仅 device 参数化，去 "cuda" 硬编码）：
    #   q, kv = q.detach().clone(), kv.detach().clone(); q.requires_grad_(True); kv.requires_grad_(True)
    #   o_ = ref_sparse_mla_fwd_interface(q, kv, indices, offsets, sm_scale, is_casual)
    #   o_.backward(do); return q.grad, kv.grad        # grad dtype 随输入 bf16
    # ref_sparse_mla_fwd_interface（ref_gpu/sparse_mla_fwd.py L214-257 移植）OOB 语义锚点：
    #   indices[indices > sk] = sk → scatter 到 sk+1 宽 mask 末列 → 裁剪 → 上越界贡献 0
    #   -1：scatter 负索引落末列 → 裁剪 → 贡献 0（与 §7 NPU 掩码语义等价，§7.5 论证）
    #   causal：arange(t) >= arange(sk)；首行 j=0 强制有效（L239-243）——L0-6/-1 与 L0-8 段首行依赖此语义
```

- **fwd 消费契约**：`from sparse_mla_fwd import sparse_mla_fwd_interface` 取 `(o, lse)`（§11.4）；fwd 未落盘期间用 `_torch_fwd_fallback`（精确 PyTorch，与 golden 内部同源 → 判据不受 fallback 影响，§11.4 论证）；最终验收前切换并断言 import 成功。
- golden 无 kernel 特化（纯 PyTorch，任意合法输入正确），符合任务书 §9"禁止针对 golden 固定数值特化"约束。

### 12.5 seed 与主仓偏差声明

- 主仓测试**无固定 seed**（randn / randperm 每次随机，§1.5）→ L0 全部用例 `torch.manual_seed(0)`（skill 模板同款）。**偏差声明**：生成流程逐行一致（randn / randperm / 尾部填充逻辑不变），仅固定随机序列——换取可复现性，无语义偏差。
- **失败复测协议**：L0 FAIL 先以 seed∈{1,2} 复跑排除环境噪声（多会话共享 NPU，任务书 §9）；仍 FAIL → 双判据定位（§12.1）→ `T.dump_tensor` 分段 dump（preprocess Delta / P/dP / ws_dkv 贡献 / 终态）定位首错段。

## 13. 性能预算分解

### 13.1 7680us 预算总表（全接口串行三段，do_bench delta=None 口径）

| 段 | 预算 | 构成依据 |
|----|------|---------|
| k1 preprocess | ≤ 300 us | 268.4MB 读 + 0.5MB 写（O/dO [2048,64,512] bf16 ×2 + Delta fp32）÷ 40 AIV × 26GB/s = **258us 带宽下限**（§11.1）+ launch/余量 |
| k2 bwd | ≤ 7250 us | 算力/带宽分解见 §13.2/§13.3（§6.4 按 7.3ms 核算的 C/V 需求为保守上界，本预算不推翻其结论，收紧至 7.25ms 需求 +0.7%） |
| k3 postprocess | ≤ 80 us | 6.75MB ÷ 1.04TB/s = **6.5us 下限**，launch 开销主导（§11.2） |
| host 编排 + 3×launch | ~ 50 us | `_CachedLauncher` marshal ~13us/launch（§10.2，indexer_bwd L170 锚点）+ 段间 host 程序序 |
| **合计** | **≤ 7680 us** | 与主仓 `do_bench(fn, rep=100, warmup=250)` 口径逐字一致（§1.5） |

### 13.2 算力下限（Cube 侧）

- **GEMM 总量**：`per_token_flop = 2·H·topk·(2·DV+3·DQKV)` = 180.4 MFLOP/token（主仓公式，§2.2 逐类精确和一致）× S=2048 → **369.4 GFLOP**（与任务书 §5 ~370 GFLOP 锚点一致）。
- **有效算力需求**：369.4 GFLOP ÷ 7.25ms = **51.0 TFLOPS** = bf16 峰值 ~320 TFLOPS（20 核聚合，fwd §9.5 锚点）的 **~16%**。
- **锚点校准**：同族稀疏注意力 fwd 实测有效算力 **91 TFLOPS**（sparse_flash_mla csa，fwd DESIGN §9.5 引）→ **1.8× 覆盖**；fwd 预算口径 48 TFLOPS ≈ 峰值 15%（S=4096 大 shape）与本算子 16% 同量级。
- **折减项（诚实计入）**：本算子 GEMM 形态 M=64（fwd 锚点为大 M 形态），M=64 → 4 个 m 分形；BN=128 已最大化 N 分形列（§5.2），DSEG=2 消化 K/N 输出列段（§5.3）。若 M=64 形态实测效率折半（~45 TFLOPS），仍贴预算线——残余风险记 §14 R11，调参空间（BN/DSEG 档位，§5.2/§5.3）+ Stage 3 perf-tuning 兜底。
- **纯算时间视角**：369.4 GFLOP @ 峰值 320 TFLOPS = **1.15ms**——GEMM 效率 ≥16% 即可；CV 软件流水（§9.2）下 Cube 计算与向量链重叠，GEMM 窗口占用 1.15/7.25 = 16%，重叠余量充足。

### 13.3 带宽下限（三口径分层，达成依据 = 单核 MTE 实测）

| 口径 | 需求 | 能力/锚点 | 余量 | 定位 |
|------|------|----------|------|------|
| **HBM（L2 完美）** | ~575MB 全接口（preprocess 268 读 + bwd 首读 q 151/do 134 共享、kv 2.25/idx 4/lse·delta 1 + dq 终写 151 + dkv/postprocess 7）÷ 7.68ms = **75GB/s** | 实测 273 读 / 238 写 GB/s | **3.2-3.6×** | ✓ 非瓶颈（与任务书 §5 "~570MB" 锚点一致） |
| **HBM（保守全量）** | ~723MB（O/dO 不共享 L2、全按 HBM 计）→ **94GB/s** | 同上 | **2.5-2.9×** | ✓ 非瓶颈 |
| **L2 中转（bwd）** | ~6.7GB（§6.3 方案 1：贡献槽写读 2.31×2 + 终态原子 2.12）÷ 7.25ms = **0.92TB/s** | ~1TB/s（估算口径） | 贴线 ⚠️ | 交叉验证项，**非达成依据**（rev1 教训：单核 MTE 口径为准，fwd §9.5） |
| **C 侧 MTE（每 AIC）** | 276MB/核 ÷ 7.25ms = **38.1GB/s**（§6.4 @7.3ms 37.8 的 +0.7% 收紧） | 实测 50-60 GB/s/核 | **24-37%** | ✓ |
| **V 侧 MTE（每 AIV）** | 207MB/AIV ÷ 7.25ms = **28.6GB/s** | 实测 ~26 GB/s/核 | **−10% ⚠️ 贴线** | 唯一贴线项：菜单 1（ws_dq bf16）→ 26.4 贴线 / 菜单 2（L0C 直 src）→ 20.5 ✓（§6.4） |

- preprocess/postprocess 带宽（AIV-only）：258us / 6.5us 下限已入 §13.1 预算（40 AIV 聚合 1.04TB/s 口径）。
- **达成判定路径**：V 侧 28.6GB/s 超 AIV 行读实测 10% → Stage 2 首版按基线实现 + msprof 实测散射段，超 §6.5 性能触发线（bwd 散射 >2.5ms 或 V 侧 >30GB/s 压不回）则依序启用菜单 1 → 菜单 2 → 方案 2 fallback（§6.4/§6.5 判据链，不重复展开）。

### 13.4 实测锚点校准表（全部数值来源，禁止无锚点估算）

| 锚点 | 数值 | 来源 |
|------|------|------|
| Cube bf16 峰值（20 核聚合） | ~320 TFLOPS | fwd DESIGN §9.5（任务口径 48 TFLOPS≈15% 反推 + sparse_flash_mla csa 实测 91 TFLOPS 佐证） |
| 同族实测有效算力 | 91 TFLOPS | sparse_flash_mla csa（fwd §9.5 引） |
| HBM 读 / 写 | 273 / 238 GB/s | A2/910B3 实测（fwd §9.5，indexer_bwd DESIGN §9.4 同款） |
| AIC ws→L1 读 | 50-60 GB/s/核 | reducesum msprof 实测 |
| AIV 行读 | ~26 GB/s/核 | 同上 |
| L2 中转 | ~1TB/s（估算） | indexer_bwd DESIGN §9.4 |
| launch marshal | ~13 us/launch | indexer_bwd L170 `_CachedLauncher` |
| GEMM 工作量 | 369.4 GFLOP（180.4 MFLOP/token） | 主仓 per_token_flop 公式 × S（§2.2） |
| dKV 原子流量 | 2.12GB 有效 / 2.42GB 全槽 | §6.1 因果有效行次推导（任务书 ~2.4GB 锚点一致） |

### 13.5 置信度与对账协议

| 项 | 需求 | 锚点能力 | 余量 | 置信度 |
|----|------|---------|------|--------|
| HBM 带宽 | 75-94 GB/s | 273/238 GB/s | 2.5-3.6× | **高** |
| Cube 算力 | 51.0 TFLOPS @7.25ms | 同族实测 91 | 1.8×（M=64 形态折减后仍贴线） | **中高** |
| C 侧 MTE | 38.1 GB/s/核 | 50-60 | 24-37% | **高** |
| V 侧 MTE | 28.6 GB/s/AIV | 26 | −10%（菜单 1/2 落地后 26.4 贴线 / 20.5 宽裕） | **中** |
| L2 吸收 | 0.92 TB/s | ~1TB/s 估算 | 贴线（非达成依据） | **中** |
| **总体 7680us** | — | — | — | **中高** |

**结论**：达标置信度**中高**。唯一贴线项为 V 侧散射带宽（−10%），由优化菜单（§6.4）+ 方案 2 fallback（§6.5）双层兜底；HBM / Cube / C 侧均双位数余量。**对账协议**：Stage 2 首次实测即 msprof 分段计时（k1/k2/k3 + bwd 内 GEMM 段/散射段），与 §13.1 预算表逐行对账，偏差 >15% 的段触发对应风险项（§14）排查。

## 14. 风险与 fallback

> 前置声明：§3 技术约束检测（L0C 128KB / 三维 Kernel / 动态边界 / 非整除 / 平台偏差）已逐项过检并在 §4-§8 内建处置；本表只列**残余/实施期风险**。概率/影响为设计期预估，Stage 2 实测对账（§13.5）后修正。

| ID | 风险 | 概率/影响 | 触发信号 | fallback / 处置 |
|----|------|----------|---------|----------------|
| R1 | V 侧散射带宽贴线（28.6 vs 26GB/s，−10%，§13.3 唯一贴线项） | 中 / 高 | msprof 散射段 >2.5ms 或 V 侧 >30GB/s | 菜单 1 ws_dq bf16（精度先验 §6.4）→ 菜单 2 L0C 直 src → 方案 2（§6.5 判据链） |
| R2 | L0C fragment 分配失败（G5 acc_dkv 128KB = 100% 贴线，§8.1 阶段③） | 中 / 中 | 编译期分配报错 / 双后端任一失败 | BN=64 回退档（G5 [64,256]=64KB、峰值 80KB/63%，发射 40→80，§5.2） |
| R3 | UB 超 192KB（140KB/73% 估算 + MEMORY_PLANNING 复用不及预期） | 低 / 中 | 编译期 UB 分配报错 | §8.3 缓解预案：s_ub 死槽复用 dv_ub（首版内建）→ row_ub/misc 共槽 → v_block 小槽下放 GM ws → BN=64 档 |
| R4 | fwd 并行会话时间窗（sparse_mla_fwd.py 未落盘，§11.4） | 中 / 低 | ImportError | `_torch_fwd_fallback`（精确 PyTorch，与 golden 同源，判据不受影响）；性能验收不依赖 fwd（o/lse 预计算输入）；fwd L0 通过后切换 |
| R5 | pto 后端逐项差异（L0C 直写 / atomic fp32 / reduce，§8.6 三项待验） | 中 / 中 | pto 编译或精度 FAIL | L0C→UB→GM 中转（+0.56MB/task，§6.4 重核）→ pto 侧切方案 2；ascendc 主路径不阻塞（§4.4 如实报告口径） |
| R6 | pto GM 标量读无 varlen 端到端先例（§10.4，fwd R2 同款） | 低 / 中 | pto 核挂死/读值异常 | host 预展开 RowMeta 表 [S,4]（b_i/s_i/bos/eos 单表读）；Stage 2 以 5 行最小复现 kernel 先验 |
| R7 | AUTO_CV_SYNC 配对遗漏挂死（6 路同步点，§9.4；indexer_bwd HISA 教训） | 低 / 高 | kernel 挂死无输出 | 核对 §9.4 表 #1-#5 命名前缀 "workspace" + 写读方全覆盖；补 `workspace_idx`（fwd R5 先例）；Developer 模式不引入手动 flag |
| R8 | lse 数值差传播（fwd 在线 softmax vs fallback 全量 logsumexp 归约序差 δ → P 乘性偏 exp(δ)，§11.4） | 低 / 中 | L0-1/3 dq/dkv 相似度贴 1e-4 下沿 | fwd L0 Lse fp32 门禁（fwd §9.3）有界 δ；本算子判据双门限（§12.1）；最终验收切 fwd kernel 路径后复跑 |
| R9 | kv_group>1 通用路径 padding head 语义（§10.5，考核 shape 外支持域） | 低 / 低 | L1 扩展用例 FAIL | 主仓 L149 装载语义逐行核对；本算子 L0 不覆盖（kv_group=1），Stage 2 L1 扩展项 |
| R10 | 主仓无 seed 用例波动 + 多会话共享环境噪声（§12.5） | 低 / 低 | L0 偶发 FAIL | seed∈{1,2} 复跑协议；仍 FAIL 走双判据定位 + 分段 dump |
| R11 | M=64 小 GEMM Cube 效率低于 91 TFLOPS 锚点形态（§13.2 折减项） | 中 / 中 | msprof GEMM 段有效算力 <51 TFLOPS | BN/DSEG 档位调参（§5.2/§5.3）→ Stage 3 perf-tuning（pass_configs / 双缓冲 §9.3）；预算线 16% 峰值需求本身宽裕 |

**风险总评**：无单点致命风险（R7 影响高但概率低且有先例排查清单）；R1/R2/R11 为性能类主风险，均已有量化判据与档位回退；设计期已知的平台强制偏差（§3.6）全部内建于架构（GM 中转 / 显式 ws / fp32 原子），不构成 fallback 触发项。**前块决策不推翻承诺**：块 D 未发现块 A-C 决策错误——块 A-C 的 L0C 128KB、BN=128 贴线、方案 1 散射选型在本块量化核算（§13）下成立，贴线项 R1/R2/R11 均在既有回退框架内闭环。

---

**文档完成声明**：本文档按任务书 4 块分块交付完毕（块 A §1-§4 / 块 B §5-§7 / 块 C §8-§11 / 块 D §12-§14），配套接口规格 `proto_sparse_mla_bwd.yaml`（dtype 全集与 §12.2 精度表一致、attrs 与 §1.4 接口契约一致）。L1/L2/Boundary 测试扩展、kernel 实现与实测对账归 Stage 2（tilelang-op-develop），性能调优归 Stage 3。
