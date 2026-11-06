# fla_chunk_inter 算子设计文档（KDA chunked 反传 inter-chunk 梯度，单 kernel，4 输出）

> 第二轮全量重迁移 Stage 1 设计（tilelang-op-orchestrator / tilelang-op-design，2026-11-05）。
> 语义权威：`ref_gpu/fla_chunk_inter.py`（Triton/FLA `chunk_kda_bwd_kernel_inter` / wrapper `chunk_kda_bwd_dqkwg`，只读）；代码级起点：`ref_gpu/fla_chunk_inter_tilelang_gpu.py`（上游 TileLang CUDA `chunk_bwd_dqkwg`，只读）；计时口径：`ref_gpu/test_utils_kda.py`（do_bench：warmup=20 / rep=10 / mean / 仅设备侧）。接口/公式/门禁权威：`proto_fla_chunk_inter.yaml`（任务书，只读）。
> 目标硬件：**Ascend 910B3**（20 AI Cube + 40 AI Vector Core / L2 192MB / HBM 64GB；L1 512KB、**UB 硬上限 196352B**（ascend_memory_planning.cc，上轮实证）、L0A/L0B 64KB、L0C 128KB per core；HBM 实测 r+w 带宽 1263 GB/s，同族 fla_wy_fast D2D 标定）。
> 执行环境（本轮）：分设架构——CPU（本机）为唯一真源（编辑/文档/git），NPU 为远程执行器（DevEnvC_DCbyi_npu，1×910B3），一切 kernel 运行经 cannlab CLI；Stage 1 纯设计在 CPU 侧完成。
> **本轮定位（与首轮首版设计的结构性差异）**：本算子为第二次迁移，上一轮（2026-11-02~03，合设环境）完成设计→实现→四场调优战役→交付，产物归档于 `../tilelang-dev-journal/fla_chunk_inter/`（DESIGN 15 章 / debug_log / perf_tuning 四轮日志）。上轮实证结论作为**领域知识**注入本设计（如同族先例档案），本设计基于其**重新推导成文**。与首轮"从零推导"的本质差异有二：① **scratch-free 红线首版强制**——禁用 `T.reduce_sum` / `T.tile.broadcast` 及一切触发 memory-planner 注入 scratch 的 op（上轮 D-1/D-2 实证：scratch 的 UB 打包次序随编译上下文翻转，坏打包 = 设备 Illegal instruction 交付级崩溃；R-EV 分设复验中基线轮在新环境首例复现该布局彩票），全部归约/广播采用上轮验证过的无 scratch 替代形式；② **首版即纳入上轮终态结构**——fixed-core(20) + 显式 L2 常驻中继、NK-merge 任务重构造、1-late flag 配对、block_DK 形状自适应、fp16 hi/lo 3-gemm 拆分、dw 负号折叠、dgkn C 侧对角化——使 Stage 2 起点即 ~6.8ms 量级（首轮首版 19.6ms）而非从原始形态重新爬坡。
> 本文档区分【已确认事实】（源码/示例/同族交付实测验证）、【上轮实证】（journal 归档的 fla_chunk_inter 实测结论，标注证据出处）、【推断】（标注依据）、【需 Stage 2 验证】四类信息。

---

## 1. 概述

### 1.1 算子名称

`fla_chunk_inter`（单 kernel：KDA chunked 反传 inter-chunk 梯度，上游 Triton 名 `chunk_kda_bwd_kernel_inter` / host 名 `chunk_kda_bwd_dqkwg`，上游 TileLang CUDA 名 `chunk_bwd_dqkwg`）

### 1.2 功能描述

KDA (Kimi Delta Attention) chunked 反传的 inter-chunk 梯度算子：给定逐 token 累积 log2 衰减 g、chunk 级隐状态 h 及其梯度 dh、上游梯度 dv/do，逐 chunk（BT=64）逐 head 计算 4 个输出梯度 dq/dk/dw/dg。3 条 GEMM 主干（do@h^T、v@dh^T、-(dv@h^T)）加 1 条 dgkn 对角化 GEMM（diag(h@dh^T)，上轮 R4 实证的最优形式）沿 DV 维 fp32 累加；门控缩放（2^g / 2^(gn−g)）、dg 末行注入、dg 组装为 Vector 侧逐元素/归约计算。chunk 间 / head 间无数据依赖，纯并行；DK 维以 dk-block 在任务内展开（NK-merge，上轮 R2 实证结构）；同一核上 produce（装载/拆分/中继）与 drain（结果回读/门控/GM 写）按 1-late 跨任务流水交织。

本算子为 **CV 融合**（Cube GEMM ×4 组 + Vector exp/逐元素乘/归约/gather）。

### 1.3 数学公式

记 chunk 长度 BT=64（S 可整除，见 §1.6），c = t // BT，last(c) = min(S,(c+1)·BT)−1（**域内恒为 chunk 内局部行 63**），scale=1.0（标量乘保留），g 为逐 token 累积 log2 衰减（fp32），h 为 chunk 级隐状态 [DK,DV]，dh 为其梯度；⊙ 逐元素乘，2^x = exp2(x)：

```
dq_raw[t,k] = Σ_v do[t,v] · h[c,k,v]            # GEMM: do @ h^T   [BT, DK]
dk_raw[t,k] = Σ_v v[t,v]  · dh[c,k,v]           # GEMM: v  @ dh^T  [BT, DK]
dw[t,k]     = -Σ_v dv[t,v] · h[c,k,v]           # GEMM 取负: -(dv @ h^T)
dgk[k]      = (Σ_v h[c,k,v] · dh[c,k,v]) · 2^g[last(c),k]   # = diag(h @ dh^T) · 2^gn（C 侧对角化）
dq[t,k]     = dq_raw[t,k] · scale · 2^g[t,k]
dk[t,k]     = dk_raw[t,k] · 2^(g[last(c),k] − g[t,k])
dg[t,k]     = q[t,k]·dq[t,k] − k[t,k]·dk[t,k] + (t == last(c)) · (dgk[k] + Σ_{t'} dk[t',k]·k[t',k])
```

所有 GEMM 沿 DV 维 fp32 累加语义。dgk 最终值 = exp2(gn)·Σ_v h·dh 与 Σ_t dk_scaled·k **先加后注入**（dk 为门控后终值），与 Triton 参考语句顺序一致（ref_gpu/fla_chunk_inter.py L140-153）。

### 1.4 算法描述（fixed-core 单 kernel，任务 = (b, chunk, head)，produce/drain 1-late 流水）

**任务粒度与网格**（上轮 R1 fixed-core + R2 NK-merge 实证结构）：

- 任务 = (b, chunk, head)，`GRID = NT · B · H`（bench：128×1×64 = 8192）；DK 的 NK 个 dk-block 在任务内 **trace-time 全展开**（NK 为 Python int，逐 bk 展开语句）——v/do/dv 无 DK 轴，合并装载消除 NK 兄弟任务对同一 [BS,DV] 拼块的重复读与左操作数拆分链重复生产（上轮实证 V-mte2 480→384.5KB/(chunk,head) 对、mte2 引擎带宽 47→53GB/s）。
- `T.Kernel(CORE_EFF, threads=1)`，`CORE_EFF = min(20, GRID)` fixed-core；核内**单一 `T.serial(L)` 任务循环**，striped 分解：核 cid 的第 t 轮任务 = `cid + t·CORE_EFF`；`L = GRID // CORE_EFF + 2`（+2 轮 slack 覆盖 1-late drain 尾轮与 Select-redirect 幂等重做）。越界 tid 用 `T.if_then_else(tid < GRID, tid, cid)` 重定向到自身 cid 任务（幂等重做，保每 on-chip buffer 单一连续 liveness 区间——上轮 R2 planner 脆弱性教训）。**不采用 drain-first**（drain 等待的 flag 由同迭代 produce 尚未置位，死锁）。
- 显式中继 workspace（11 个，`workspace_idx`，§8.2）：V 产 C 消的操作数中继 + C 产 V 消的结果中继，槽位环 NSLOT=4（cube lag 上界 1 + 余量），全部 L2 常驻（bench 足迹 ~25MB ≪ 192MB）。

**每迭代的 produce（V+C 协作）**：

1. V 侧逐 bk 装载 h/dh [bDK,DV] fp32 → hi/lo 拆分（cast→sub→cast 两步）→ 右中继（扁平 (task,bk) 槽位）；
2. V 侧装载 do/v/dv [BS,DV] 各一次（负号折叠：dv 装载后 ×(-1)）→ hi/lo 拆分 → 左中继（单槽/任务）；
3. C 侧回读 6 路左中继入 L1（每任务一次，A 操作数跨 bk 复用）→ 逐 bk 回读 4 路右中继 → **12 个 `T.gemm_v0(transpose_B=True)`**：dq/dk/dw 三组 × (hi@hi + hi@lo + lo@hi) + dgkn 对角化组（`dgm_frag = h @ dh^T` 的 [bDK,bDK] fp32，复用 L1 已有 h/dh 操作数，零新增中继）；
4. C 侧 dw_frag → GM 直存（负号已折叠）；dq/dk/dgm_frag → **融合 drain 中继**（每 (task,bk) 槽内行段 [0,BS)=dq | [BS,2BS)=dk | [2BS,2BS+bDK)=dgm）；
5. C 侧手动 `T.set_cross_flag("FIX", 40 + j%2)`（交替 ID，防单 FIFO ID 锁步）。

**每迭代的 drain（V 侧，读 task j-1 的中继）**：

1. `T.wait_cross_flag(40 + (j+1)%2, "V")`（1-late 配对：与 set 40+(j-1)%2 成对，上轮 R2 实证 V 侧等待归零）；
2. 逐 bk：**单次融合回读** ddr_relay [2BS+bDK, bDK] fp32（三中继读合一，上轮 R4 实证 -2.05%）+ 装载 G/Q/K [BS,bDK] 与 gn 末行 [bDK]；
3. ln2 预折（g/gn 原位 ×ln2）→ dq ⊙= scale·exp(g·ln2)；`T.tile.gather` 常量 byte 偏移从 dgm 行段提取对角 dgkn（prologue 预构建偏移）→ dgkn ⊙= exp(gn·ln2)；
4. gn 门控（无 scratch 形式）：bDK=64 路径 = transpose + `row_expand_sub_experiment`（brcb 单 Sub）+ mul(-1) 保号 + exp + transpose 回；bDK=32 路径 = `T.unroll(BS)` 64 逐行 Sub + exp；
5. dk ⊙= exp(gn·ln2 − g·ln2)；`dkk = dk⊙k` → **6 层 T.Parallel 字面量折半树**归约 Σ_t dkk（无 scratch）→ `pp = dkk[0,:] + dgkn`；
6. `dg = q⊙dq − dkk`；**末行单行加** `dg[BS-1,:] += pp`（取代 LastRow 常量 + broadcast 掩码链——上轮 D-2 实证形式，有限 pp 逐位等价、溢出时更贴 golden）；
7. 3 路 GM 写（dq/dk/dg；dw 已由 C 侧直存）。

> **⚠️ Host 侧 Buffer 操作约束**（ascend-constraints.md §5）：host 路径只含 (a) 输入张量构造（CPU randn + H2D）、(b) kernel 调用（8 输入 + 4 输出直传，bshd/5D 布局直读，**零 view/零 reshape/零 permute**；中继 workspace 由框架按 `workspace_idx` 自动分配，host 不触碰）、(c) CPU 侧精度验证。无 `.contiguous()`/pad/cat/`reshape` 物理化/aclnn 调用/`torch.npu.synchronize()`。审计见 §9.6。

### 1.5 数据流图

```
GM[h/dh:fp32] --T.copy(5D切片)--> UB[fp32] --cast→sub→cast 拆分--> UB[hi/lo:fp16] --写--> WS_R 中继(GM,L2常驻)
GM[do/v/dv:fp32] --T.copy(4D切片)--> UB[fp32] --(dv 先 ×-1)--> 同上拆分 --写--> WS_L 中继(GM,L2常驻)
WS_L/WS_R --C 侧回读--> L1[fp16] --gemm_v0(ᵀB)×12--> L0C[fp32 ×4 frags: dq/dk/dw/dgm]
L0C[dw_frag] --T.copy(同 dtype)--> GM[dw]（C 侧直存，负号已折叠）
L0C[dq/dk/dgm_frag] --T.copy--> ddr_relay(GM 行段中继, 手动 flag 40+j%2)
ddr_relay --V 侧单次回读(1-late wait)--> UB[ddr_ub:fp32] --gather 对角提取--> dgkn
GM[G/Q/K/gn:fp32] --T.copy--> UB[fp32] --exp(·ln2)/mul/折半树/末行加--> UB[dq/dk/dg] --T.copy--> GM
```

> V↔C 全部交换经 GM 中继 workspace（工具链无片上直连路径，上轮实证/同族 fla_wy_fast final_summary §4）；中继槽位环 L2 常驻使交换流量不落 HBM（上轮 R1 fixed-core 配方核心收益）。

### 1.6 范围与分派覆盖审计

【已确认事实】范围裁剪（proto 声明 + 用户确认 2026-11-02 承继）：定长序列（非 varlen）、scale=1.0（标量乘保留，trace-time `if scale != 1.0` 折叠跳过恒等乘）、chunk_size=64、全 fp32。varlen（cu_seqlens/chunk_indices）分支不在范围；Triton wrapper 的 w 参数未被 kernel 使用（dw 由 −(dv@h^T) 推导），本算子不收 w。

**【已确认事实】S % 64 ≠ 0 属参考实现未定义域**（上轮已做三层证据核实，结论承继有效）：

- 数学层：h/dh 的 chunk 维 = S//64（floor）。S%64≠0 时末部分 chunk 不存在对应 h/dh 状态，inter-chunk 项无定义；
- 语义权威层：Triton kernel 的 `i_tg = i_b*NT + i_t` 对末 chunk 越过 h 张量末尾，`boundary_check` 只作用于 (V,K) 虚拟形状 → OOB 读；
- 代码级起点层：上游 TileLang CUDA 版 `BS = S // block_S` 与 grid `ceildiv(S, block_S)` 失配且搬运无掩码 → 静默 OOB。

**本设计口径**：支持域收紧为 S % 64 == 0（与同族 fla_chunk_o / fla_wy_fast / fla_chunk_intra 一致）；host wrapper `_check_shape` 显式 assert 拒绝（确定性报错替代静默 OOB）。域内 m_t 掩码恒全真（**无需行掩码**）、last(c) 恒为局部行 63（末行单行加精确表达）。Stage 2 场景 B 将 S%64≠0 归入 L2 负向测试（正确拒绝）。

```text
[DISPATCH-COVERAGE]
supported_domain: S%64==0（chunked 数据模型数学要求，§1.6 三层证据）; DK ∈ 32 的倍数（bDK=32 整分）;
  DV ∈ 16 的倍数且 DV≥64（gemm K=DV ≥64 满足 K≥32 分形 + 32B 对齐）; B,H ≥ 1 任意; chunk_size=64 固定;
  scale 任意 float（标量乘，trace-time 恒等折叠）; dtype 固定 = 全 fp32（输入/输出/累加）
generic_fallback: fla_chunk_inter 单一通用 kernel（无 dtype/shape 特化分支；block_DK=64/32 由
  "DK%64==0 且 DV>=128" 纯 metadata 谓词自适应选择，属性能参数非分派谓词——谓词未命中即回落
  bDK=32 通用路径，两路径均覆盖全部声明域输入）
specializations: 无（block_DK 为同一 kernel 内 trace-time 语句分支，非 kernel 特化）
unsupported_inputs: varlen、S%64≠0（数学未定义域，wrapper assert 拒绝，L2 负向）、chunk_size≠64、
  DK 非 32 倍数、DV < 64 或非 16 倍数（分形/对齐违反）、非 fp32 dtype 变体（proto 全集仅 fp32）
  —— 均与 supported_domain 不冲突（proto 显式声明或 wrapper assert）
result: pass
```

---

## 2. 编程模式选型

### 2.1 模式结论

**Developer 模式**（用户明确指定，2026-11-02 承继）——`alloc_shared` / `alloc_fragment` 声明式分配 + 四件套 pass_configs 自动同步。因 V↔C 交换需 L2 常驻中继协议，采用 **Developer 复杂场景回退写法**（design-template §8.2 许可形态）：显式 `workspace_idx` 中继 + 结果中继链手动 `set_cross_flag` / `wait_cross_flag` 配对（1-late 协议），核内 DMA/计算依赖仍全自动同步。**不使用** Expert 的 `T.Scope("C"/"V")` / 显式 `alloc_L1/ub/L0A/L0B/L0C`。

### 2.2 选型理由

1. 算子是典型 CV 融合（4 组 GEMM + Vector exp/逐元素/归约），核内无跨核自定义同步需求；上游 TileLang CUDA 版本身即 Developer 风格 API，迁移映射最直接；
2. 上轮全战役（R1-R4）在 Developer 模式内完成交付，其终态结构（fixed-core + 显式中继 + 手动 flag 结果链配对 + scratch-free 向量形式）为**已验证可行且已贴形态地板**的形态；上轮 R3 实证的 Expert 路线（V 侧手动三引擎流水）yield 仅 -3.1%（假设证伪），Expert 全量重写在家族内 4 试 3 败、且不改变可达域（final_summary §3.2）——**无换模式的收益证据**；
3. threads=1：threads=2 的 vid 语义对本算子 drain 数学结构性破坏（上轮 R3 微实验 5 子链闭合 + ascend_vid_reduction.cc 源码双证，见 §3.5.1）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（编译器映射 L1/UB）+ `T.alloc_fragment`（映射 L0C）；V↔C 中继 = 显式 GM workspace（`workspace_idx` 框架托管分配，L2 常驻） |
| GEMM | `T.gemm_v0(A, B, C, transpose_B=True, init=True/False)`：A/B 为 L1 **fp16 hi/lo**，C 为 L0C fp32（fp32 A/B 工具链损坏，上轮实证） |
| 计算方式 | `T.tile.mul/sub/add/exp/fill/transpose/gather/arith_progression/row_expand_sub_experiment` + `T.Parallel` 折半树 + `T.unroll` 行链（buffer 级全 tile 指令；**禁用清单见 §3.5.1 红线**） |
| 作用域 | 编译器自动 C/V 分离（AUTO_CV_COMBINE）；produce/drain 同核内直线交织（fixed-core 单循环体） |
| 同步方式 | 自动同步（AUTO_SYNC + AUTO_CV_SYNC：核内依赖 + 操作数中继写读按名配对）+ **手动 cross-flag**（结果中继链 1-late 配对，§7.2）——Developer 复杂场景回退形态 |
| threads | **1**（threads=2 结构性闭合，§3.5.1；grid 恒为 20 核内任务循环，无 launch 形态问题） |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| F1 | `x = hi + lo`（5 操作数 do/v/dv/h/dh） | fp16 hi/lo 拆分：hi=f16(x)，lo=f16(x−f32(hi))；误差 ~2⁻²² 相对/操作数对（fp32 gemm_v0 损坏的强制替代） |
| F2 | `dv' = −dv` | dw 负号折叠（拆分前一次性标量乘；(−x) 的 hi/lo = −(x 的 hi/lo)，逐位恒等） |
| F3 | `dgm = h @ dh^T`；`dgk = diag(dgm) · 2^gn` | **C 侧对角化**：3 个 hi/lo 小 gemm 产 [bDK,bDK] fp32 → 中继 → drain 一次 gather 对角提取（上轮 R4 实证，替代 V 侧 mul+reduce 整链） |
| F4 | `dq_raw = do @ h^T` | gemm_v0 ×3（hi@hi / hi@lo / lo@hi，L0C fp32 累加） |
| F5 | `dk_raw = v @ dh^T` | 同上 |
| F6 | `dw = −(dv @ h^T)` | 同上（负号已在 F2 折叠；C 侧直存） |
| F7 | `dq = dq_raw · scale · 2^g` | UB：ln2 预折 + exp 链 + 标量乘（scale≠1.0 才乘，trace-time 折叠） |
| F8 | `dk = dk_raw · 2^(gn−g)` | UB：ln2 预折后指数 = gn·ln2 − g·ln2（单次 Sub）；行展开无 scratch 形式（§3.5.1 红线） |
| F9 | `pp = dgk + Σ_t dk⊙k` | UB：mul + **6 层折半树**（无 scratch）+ add |
| F10 | `dg = q⊙dq − k⊙dk + m_last·pp` | UB：mul/sub + **末行单行加** `dg[BS-1,:] += pp`（无 broadcast/掩码乘） |

### 3.2 TileLang API 映射（含出处；★ = 上轮 journal 实证，† = 本轮源码复核）

| 步骤 | TileLang API（昇腾） | 参数 | 出处 |
|------|---------------------|------|------|
| Kernel | `T.Kernel(CORE_EFF, threads=1, is_npu=True) as (cid)` | CORE_EFF = min(20, GRID)；fixed-core | †T.Kernel 标准用法；★上轮 impl_final.py:257（fixed-core 交付形态，R1-R4 实证） |
| 任务循环 | `for j in T.serial(L)` + `T.if_then_else(tid < GRID, tid, cid)` Select-redirect | L = GRID//CORE_EFF + 2；striped：tid = cid + k·CORE_EFF | ★上轮 impl_final.py:336-340（幂等重做保单一 liveness，R2 planner 教训） |
| GEMM | `T.gemm_v0(A, B, C, transpose_B=True, init=True/False)` | A/B→L1 fp16、C→L0C fp32；init=True 组首覆盖 / False 组内累加 | †ascend.py:418（本轮签名复核：`gemm_v0(A, B, C, transpose_A=False, transpose_B=False, init=False, kL0Size=128, n_actual=None)`）；★上轮 12 gemm/任务硬件 PASS |
| 内存分配 | `T.alloc_shared(shape, dtype)` / `T.alloc_fragment(shape, dtype)` | shared→L1/UB、fragment→L0C | ★上轮 impl_final.py:259-319 |
| 拆分链 | `T.copy(f32, f16)` → `T.copy(f16, f32)` → `T.tile.sub(lo_f32, x, hi_f32)` → `T.copy(lo_f32, lo_f16)` | 跨 dtype 只在 UB↔UB 内两步（GM↔UB cast-on-copy 缺陷规避） | ★上轮 impl_final.py:358-363（同族 k1 拆分链同款） |
| 逐元素 | `T.tile.mul(dst, src0, src1/标量)`（含 in-place dst==src0）、`T.tile.sub/add`、`T.tile.exp(dst, src)` | fp32 域全 tile 指令 | †ascend_tile.py:1152/1130/1113/1277；★上轮全链实证 |
| 对角提取 | `T.tile.gather(dst, src, src_offset, src_base_addr)` + `T.tile.arith_progression(buf, first, diff, count)` + `T.reinterpretcast(dst, src, "uint32_t")` | offset = i·(bDK+1)·4 byte，prologue 一次构建；gather 无 tmp → 不注入 planner scratch（NoWorkspace） | †ascend_tile.py:1848/398 + ascend.py:618（本轮签名复核）；★上轮 R4-2 实证（cann-bench gather 生产同款） |
| 行展开门控 | `T.tile.transpose(dst, src)` + `T.tile.row_expand_sub_experiment(dst, dst, src1_vec, tmp)` | bDK=64 路径（行宽 256B 满足 256B 行窗）；exp 参数逐位一致需 mul(-1) 保号 | †ascend_tile.py:1800/2983（本轮签名复核）；★上轮 R4-3 实证（bitwise 等价，-2.16%） |
| 行链门控 | `for t in T.unroll(BS): T.tile.sub(gnd[t, 0:bDK], gn_ub, g_ub[t, 0:bDK])` | bDK=32 路径（行宽 128B < 256B 行窗） | ★上轮 D-2 实证（P4 逐位精确） |
| 列归约 | 6 层 `for t, kk in T.Parallel(32, bDK): dkk[t,kk] += dkk[t+32,kk]`（32→16→8→4→2→1 字面量折半树） | 折叠**外层维**（行）：第二操作数基址 = 行起点（bDK 对齐）；沿连续维折叠 h<8 层必失配 | ★上轮 D-2 实证（P1/P6r PASS；折半树工程约束） |
| 末行注入 | `T.tile.add(dg_ub[BS-1, 0:bDK], dg_ub[BS-1, 0:bDK], pp_row)` | 单行加，取代 LastRow+broadcast+掩码乘 | ★上轮 D-2 实证（P6r PASS；有限 pp 逐位等价） |
| 手动 flag | `T.set_cross_flag("FIX", 40 + j % 2)` / `T.wait_cross_flag(40 + (j + 1) % 2, "V")` | 1-late 配对 + 交替 ID + head-set 预充 | †ascend.py:191/213（本轮签名复核：`set_cross_flag(pipe, flag, mode=2)` / `wait_cross_flag(flag, pipe="")`）；★上轮 R2 iter2 实证（V 侧等待归零） |
| 搬运 | `T.copy(GM 切片, ub)` / `T.copy(ub, ws[cid, slot, ...])` / `T.copy(ws[cid, slot, ...], l1_shared)` / `T.copy(frag, GM 切片)` / `T.copy(frag, ws)` | f32 frag→f32 GM 直存合法（同 dtype）；5D/4D 切片最内维连续 | ★上轮 impl_final.py:355-356/362-363/405-410/437/440-442 |
| 循环 | `T.serial(L)`（任务）+ `for bk in range(NK)`（trace-time 展开）+ `T.unroll(BS)` / `T.Parallel(...)`（drain 内） | DV 轴**无循环**（单遍全宽，上轮 R1 iter1 实证） | ★上轮终态结构 |
| 常量 | LN2 = 0.6931471805599453（模块级 Python float） | 2^x = exp(x·ln2)：exp2 无昇腾后端 | †src/op/ascend.cc 仅 ascend_exp 映射（上轮源码复核承继）；★上轮实证 |

### 3.3 计算伪代码（kernel 主体，bench 配置 bDK=64 / DV=128 为主路径）

```python
LN2 = 0.6931471805599453   # 2^x = exp(x·ln2)
CORE_NUM = 20              # 910B3 AI Cube 核数（fixed-core 上限）
NSLOT = 4                  # 中继槽位环深度（cube lag 上界 1 + 余量）

@tilelang.jit(out_idx=[19, 20, 21, 22], workspace_idx=[8..18], pass_configs=PASS_CONFIGS)
def fla_chunk_inter(B, S, H, DK, DV, scale=1.0, chunk_size=64, block_DK=None):
    # wrapper 侧：_check_shape assert 域（§1.6）+ block_DK 自适应：
    #   block_DK = 64 if (DK % 64 == 0 and DV >= 128) else 32   # 形状自适应性能参数
    BS = chunk_size; NT = S // BS; NK = DK // block_DK
    GRID = NT * B * H                      # 任务 = (b, chunk, head)
    CORE_EFF = min(CORE_NUM, GRID)         # fixed-core（小 grid 时 clamp）
    L = GRID // CORE_EFF + 2               # +2 slack：1-late drain 尾轮 + redirect redo
    WS_L = (CORE_EFF, NSLOT, BS, DV)                 # f16 左操作数中继（单槽/任务）
    WS_R = (CORE_EFF, NSLOT * NK, block_DK, DV)      # f16 右操作数中继（扁平 (task,bk) 槽）
    WS_F = (CORE_EFF, NSLOT * NK, 2*BS + block_DK, block_DK)  # f32 融合 drain 中继

    @T.prim_func
    def main(Q, K, V, G, h, dv, DO, Dh,                      # 0-7 输入
             workspace_1..10, ddr_relay,                      # 8-18 中继（§8.2 表）
             dq, dk, dw, dg):                                 # 19-22 输出
        with T.Kernel(CORE_EFF, threads=1, is_npu=True) as (cid):
            # ---- L0C（fp32，跨迭代复用）----
            dq_frag / dk_frag / dw_frag = T.alloc_fragment((BS, block_DK), "float32")   # ×3
            dgm_frag = T.alloc_fragment((block_DK, block_DK), "float32")   # dgkn 对角化
            # ---- UB produce 侧 ----
            h_ub / dh_ub = T.alloc_shared((block_DK, DV), "float32")                    # ×2
            v_ub / do_ub / dvn_ub = T.alloc_shared((BS, DV), "float32")                 # ×3
            t_hi / t_lo = T.alloc_shared((BS, DV), "float32")              # 左拆分暂存（共享）
            th_hi / th_lo = T.alloc_shared((block_DK, DV), "float32")      # 右拆分暂存（共享）
            do_hi/lo, v_hi/lo, dvn_hi/lo = T.alloc_shared((BS, DV), "float16")           # ×6
            h_hi/lo, dh_hi/lo = T.alloc_shared((block_DK, DV), "float16")                # ×4
            # ---- L1（gemm 操作数 f16）----
            do_hi/lo_l1, v_hi/lo_l1, dvn_hi/lo_l1 = T.alloc_shared((BS, DV), "float16")     # ×6
            h_hi/lo_l1, dh_hi/lo_l1 = T.alloc_shared((block_DK, DV), "float16")             # ×4
            # ---- UB drain 侧 ----
            ddr_ub = T.alloc_shared((2*BS + block_DK, block_DK), "float32")   # 融合回读域
            dgkn_ub / gn_ub / gnexp_ub = T.alloc_shared((block_DK,), "float32")   # ×3
            dgk_off_i32 = T.alloc_shared((block_DK,), "int32")
            dgk_off_u32 = T.alloc_shared((block_DK,), "uint32")
            g_ub / gexp_ub / gnd_ub / q_ub / k_ub / dkk_ub / dg_ub = T.alloc_shared((BS, block_DK), "float32")  # ×7
            gexp_tmp = T.alloc_shared((block_DK, 8), "float32")   # brcb tmp（仅 bDK=64 路径）

            # ---- prologue：对角偏移（live 锚，兼保 head-set 挂在非空 prologue）----
            T.tile.arith_progression(dgk_off_i32, 0, (block_DK + 1) * 4, block_DK)
            T.reinterpretcast(dgk_off_u32, dgk_off_i32, "uint32_t")
            T.set_cross_flag("FIX", 41)      # head-set：预充 drain(0) 的 1-late wait

            # ============ 单一任务循环（produce j + drain j-1 交织）============
            for j in T.serial(L):
                tid_p = cid + j * CORE_EFF
                pt = T.if_then_else(tid_p < GRID, tid_p, cid)    # Select-redirect 幂等重做
                slot = j % NSLOT
                # ================= produce(pt) =================
                bh = pt % H; bt = pt // H; bs = bt % NT; bb = bt // NT; s0 = bs * BS
                for bk in range(NK):                             # trace-time 展开
                    dk0 = bk * block_DK
                    T.copy(h[bb, bs, bh, dk0:dk0+block_DK, 0:DV], h_ub)     # 5D 尾维连续
                    T.copy(Dh[bb, bs, bh, dk0:dk0+block_DK, 0:DV], dh_ub)
                    # dh 拆分 → 右中继（扁平 (task,bk) 槽：bk 间区域不相交，无同槽覆写耦合）
                    # ...cast→sub→cast 拆分链（dh_ub → dh_hi/dh_lo，§3.2 四语句）...
                    T.copy(dh_hi, workspace_7[cid, slot*NK+bk, 0:block_DK, 0:DV])
                    T.copy(dh_lo, workspace_8[cid, slot*NK+bk, 0:block_DK, 0:DV])
                    # h 拆分 → 右中继（dq 组与 dw 组共享同一中继——同 UB 源单 L1 目的，§8.6）
                    # ...同上（h_ub → h_hi/h_lo → workspace_3/4）...
                # do/v/dvn 装载+拆分+左中继（每任务一次，NK 个 bk 共享）
                T.copy(DO[bb, s0:s0+BS, bh, 0:DV], do_ub)    # ...拆分... → workspace_1/2[cid, slot]
                T.copy(V[bb, s0:s0+BS, bh, 0:DV], v_ub)      # ...拆分... → workspace_5/6[cid, slot]
                T.copy(dv[bb, s0:s0+BS, bh, 0:DV], dvn_ub)
                T.tile.mul(dvn_ub, dvn_ub, -1.0)             # dw 负号折叠
                # ...拆分... → workspace_9/10[cid, slot]
                # C 侧：左中继回读入 L1（一次，A 操作数跨 bk 复用）
                T.copy(workspace_1[cid, slot, 0:BS, 0:DV], do_hi_l1)   # ×6 路（2/5/6/9/10 同构）
                for bk in range(NK):
                    dk0 = bk * block_DK
                    T.copy(workspace_3[cid, slot*NK+bk, ...], h_hi_l1)   # 首用前回读，MTE2 与 gemm 重叠
                    T.gemm_v0(do_hi_l1, h_hi_l1, dq_frag, transpose_B=True, init=True)
                    T.copy(workspace_4[cid, slot*NK+bk, ...], h_lo_l1)
                    T.gemm_v0(do_hi_l1, h_lo_l1, dq_frag, transpose_B=True, init=False)
                    T.gemm_v0(do_lo_l1, h_hi_l1, dq_frag, transpose_B=True, init=False)
                    # ...（dk 组 ×3：v_hi/lo × dh_hi/dh_lo；dw 组 ×3：dvn_hi/lo × h_hi/h_lo）...
                    # dgkn 对角化组：dgm = h @ dh^T（复用 L1 已有操作数，零新增中继）
                    T.gemm_v0(h_hi_l1, dh_hi_l1, dgm_frag, transpose_B=True, init=True)
                    T.gemm_v0(h_hi_l1, dh_lo_l1, dgm_frag, transpose_B=True, init=False)
                    T.gemm_v0(h_lo_l1, dh_hi_l1, dgm_frag, transpose_B=True, init=False)
                    T.copy(dw_frag, dw[bb, s0:s0+BS, bh, dk0:dk0+block_DK])   # C 侧直存
                    T.copy(dq_frag, ddr_relay[cid, slot*NK+bk, 0:BS, 0:block_DK])
                    T.copy(dk_frag, ddr_relay[cid, slot*NK+bk, BS:2*BS, 0:block_DK])
                    T.copy(dgm_frag, ddr_relay[cid, slot*NK+bk, 2*BS:2*BS+block_DK, 0:block_DK])
                T.set_cross_flag("FIX", 40 + j % 2)          # 交替 ID（防单 FIFO 锁步）

                # ================= drain(j-1) =================
                tid_d = cid + (j - 1) * CORE_EFF
                dt = T.if_then_else(tid_d < GRID, T.if_then_else(tid_d >= 0, tid_d, cid), cid)  # 嵌套 Select（j=0 恒负 → redirect）
                dslot = (j + NSLOT - 1) % NSLOT
                bh = dt % H; bt = dt // H; bs = bt % NT; bb = bt // NT; s0 = bs * BS
                T.wait_cross_flag(40 + (j + 1) % 2, "V")     # 1-late：等 set 40+(j-1)%2
                for bk in range(NK):
                    dk0 = bk * block_DK
                    # 单次融合回读（dq|dk|dgm 三行段一读）
                    T.copy(ddr_relay[cid, dslot*NK+bk, 0:2*BS+block_DK, 0:block_DK], ddr_ub)
                    T.copy(G[bb, s0:s0+BS, bh, dk0:dk0+block_DK], g_ub)
                    T.copy(G[bb, s0+BS-1, bh, dk0:dk0+block_DK], gn_ub)      # chunk 末行 1D 切片
                    T.copy(Q[bb, s0:s0+BS, bh, dk0:dk0+block_DK], q_ub)
                    T.copy(K[bb, s0:s0+BS, bh, dk0:dk0+block_DK], k_ub)
                    # ln2 预折（省一个 [BS,bDK] mul；舍入差 ~2e-7 ≪ rtol）
                    T.tile.mul(g_ub, g_ub, LN2); T.tile.mul(gn_ub, gn_ub, LN2)
                    # dq = dq_raw · scale · 2^g（dq 在 ddr_ub 行段 [0,BS) 原位门控）
                    T.tile.exp(gexp_ub, g_ub)
                    T.tile.mul(ddr_ub[0:BS, :], ddr_ub[0:BS, :], gexp_ub)
                    if scale != 1.0:                          # trace-time 恒等折叠
                        T.tile.mul(ddr_ub[0:BS, :], ddr_ub[0:BS, :], scale)
                    # dgk = diag(dgm) · 2^gn（gather 无 scratch 对角提取）
                    T.tile.gather(dgkn_ub, ddr_ub[2*BS:2*BS+block_DK, :], dgk_off_u32, 0)
                    T.tile.exp(gnexp_ub, gn_ub)
                    T.tile.mul(dgkn_ub, dgkn_ub, gnexp_ub)
                    # dk = dk_raw · 2^(gn−g)：行展开门控（无 scratch，红线）
                    if block_DK == 64:                        # trace-time：256B 行窗
                        T.tile.transpose(gexp_ub, g_ub)       # gT（gexp_ub 此前已死，复用槽位）
                        T.tile.row_expand_sub_experiment(gexp_ub, gexp_ub, gn_ub, gexp_tmp)
                        T.tile.mul(gexp_ub, gexp_ub, -1.0)    # 保号：exp 参数逐位一致
                        T.tile.exp(gexp_ub, gexp_ub)
                        T.tile.transpose(gnd_ub, gexp_ub)
                    else:                                     # bDK=32：行链
                        for t in T.unroll(BS):
                            T.tile.sub(gnd_ub[t, 0:block_DK], gn_ub, g_ub[t, 0:block_DK])
                        T.tile.exp(gnd_ub, gnd_ub)
                    T.tile.mul(ddr_ub[BS:2*BS, :], ddr_ub[BS:2*BS, :], gnd_ub)
                    # dg = q⊙dq − k⊙dk；dkk = dk⊙k 复用为负项（先消费再折叠）
                    T.tile.mul(dkk_ub, ddr_ub[BS:2*BS, :], k_ub)
                    T.tile.mul(dg_ub, q_ub, ddr_ub[0:BS, :])
                    T.tile.sub(dg_ub, dg_ub, dkk_ub)
                    # pp = Σ_t dkk：6 层字面量折半树（外层维折叠，无 scratch）
                    for t, kk in T.Parallel(32, block_DK): dkk_ub[t,kk] += dkk_ub[t+32,kk]
                    for t, kk in T.Parallel(16, block_DK): dkk_ub[t,kk] += dkk_ub[t+16,kk]
                    for t, kk in T.Parallel(8, block_DK):  dkk_ub[t,kk] += dkk_ub[t+8,kk]
                    for t, kk in T.Parallel(4, block_DK):  dkk_ub[t,kk] += dkk_ub[t+4,kk]
                    for t, kk in T.Parallel(2, block_DK):  dkk_ub[t,kk] += dkk_ub[t+2,kk]
                    for kk in T.Parallel(block_DK):         dkk_ub[0,kk] += dkk_ub[1,kk]
                    # pp += dgk；末行单行注入（无 broadcast/掩码乘）
                    T.tile.add(dkk_ub[0, 0:block_DK], dkk_ub[0, 0:block_DK], dgkn_ub[0:block_DK])
                    T.tile.add(dg_ub[BS-1, 0:block_DK], dg_ub[BS-1, 0:block_DK], dkk_ub[0, 0:block_DK])
                    # 3 路 GM 写（dw 已 C 侧直存）
                    T.copy(ddr_ub[0:BS, :], dq[bb, s0:s0+BS, bh, dk0:dk0+block_DK])
                    T.copy(ddr_ub[BS:2*BS, :], dk[bb, s0:s0+BS, bh, dk0:dk0+block_DK])
                    T.copy(dg_ub, dg[bb, s0:s0+BS, bh, dk0:dk0+block_DK])
    return main
```

> 层级归属（L1 vs UB）由 AUTO_CV_COMBINE + MEMORY_PLANNING 规划；gemm 输入（`*_l1`）落 L1，元素计算/载入 buffer 落 UB。produce 侧 fp32 声明集 224KB > UB 196352B，靠**逐操作数"载入→拆分→中继"链式序 + lifetime 交错复用**压活动峰值（上轮实证形态，§4.5）。伪代码中 `...拆分...` 展开为 §3.2 拆分链四语句，`×N` 为 N 个同形 alloc。

### 3.4 API 可行性确认

全部 API 均有出处（§3.2 表逐行标注 †本轮源码行号复核 / ★上轮交付实证 / examples 先例），**无凭记忆 API**。关键组合的可行性依据：

1. **fixed-core + 单 T.serial 任务循环 + Select-redirect**：★上轮 R1-R4 全战役实证（同构 fla_chunk_o R4 配方家族验证）；redirect 重做幂等性——produce/drain 对同 tid 重复执行写出相同值（GM 写幂等 + 中继槽位幂等覆盖），无跨迭代累加语义残留（L0C frags 每任务 init=True 组首重置）。
2. **单循环体内 4 gemm 组（12 gemm_v0、4 L0C 累加器）**：★上轮终态逐字实证（含 dgm 对角化组）；多组操作数经**中继回读隔离**（每路独立 workspace 槽），不存在首版设计担心的"L1 操作数角色互换共享"形态——dw 组与 dq 组共享 h 中继（同 transpose_B 右角色，§8.6 约束允许）。
3. **gather 对角提取 + arith_progression + reinterpretcast**：†签名复核（ascend_tile.py:1848/398、ascend.py:618）；★上轮 R4-2 硬件 PASS（valrange_l dg 与基态持平）；gather 无 tmp 参数路径不注入 planner scratch（NoWorkspace，D-2 红线兼容）。
4. **手动 flag 与 AUTO_CV_SYNC 共存**：操作数中继（workspace_1..10）由 AUTO_CV_SYNC 按名自动配对；结果中继（ddr_relay）命名脱离 `workspace_*` 前缀**逃逸自动 FIFO 配对**，改手动 1-late 协议——★上轮实证共存形态。
5. **scratch-free 替代形式全家**（折半树/行链/单行加/转置门控/gather 对角）：★上轮 D-2 五位点替换 + R4 三项采纳全部硬件 PASS，生成 C grep 零 scratch 命中（R4.4 终检）。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（强制检测）

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **是**（上游 `T.Kernel(ceildiv(DK,block_DK), ceildiv(S,block_S), B*H, threads=128)` 三维） | 一维化两段式：任务空间一维化 GRID=NT·B·H + fixed-core `T.Kernel(CORE_EFF)` 核内 striped 循环（★上轮终态形态）；`examples/linear_attention_and_rnn/gdn/gdn_chunk_o.py:35` 一维化先例 |
| threads 参数限制（仅 1 或 2） | **是**（上游 threads=128） | `threads=1`；**threads=2 结构性闭合**（★上轮 R3 微实验 5 子链：drain 的 1-D [bDK] 逐列不变量沿 DK 切半 vs 2-D [BS,bDK] 沿 BS 切半失配、dim-0 reduce 无跨 vid 合并、broadcast 不在 tile_ops 折半集——ascend_vid_reduction.cc L184-204/L475-487/L563-586 源码双证；突破需上游三项修复，上轮实证） |
| 动态循环边界不支持 | 否 | B/S/H/DK/DV 均为 JIT 编译期常量；L、NK、NSLOT 编译期展开 |
| 流水线不支持动态边界 | 否（无 T.Pipelined） | DV 轴单遍全宽直线（★上轮 R1 iter1 实证替代 Pipelined 分块；首版 i_v Pipelined(2) 结构已废弃） |
| **L0C 溢出（128KB）** | 已核算 | 4 frags：dq/dk/dw [64,64] fp32 16KB×3 + dgm [64,64] 16KB = **64KB = 50%** 限额 ✓（§4.5） |
| T.copy 列方向 strided 切片 | 已规避 | 全部 GM 切片"标量前缀 + 行区间 × 最内维连续 block"：q/k/g [bb, s0:s0+64, bh, dk0:dk0+bDK]（4D 最内维连续）、v/do/dv [bb, s0:s0+64, bh, 0:DV]、h/dh [bb, bs, bh, dk0:dk0+bDK, 0:DV]（5D 最内维 DV 连续）、gn 末行 [bb, row, bh, dk0:dk0+bDK]（1D 连续） |
| GEMM 分形限制 | 已核算 | fp16：M≥16, K≥32(L0A/L0B), N≥16。dq/dk/dw：M=64, K=DV≥64 ✓, N=bDK∈{32,64} ✓；dgm：M=bDK≥32 ✓, K=DV ✓, N=bDK ✓；kL0Size=128 默认覆盖 K=DV≤128 域内满配 |
| GEMM 非整除 | 已规避 | 声明域内：S%64==0、DK%32==0（bDK 整分）、DV 全宽无分块。域外 shape 由 wrapper assert 拒绝（L2 负向） |
| **fp32 gemm_v0 损坏**（★上轮实证：fp32 A/B 产出 1e37 级交替垃圾） | **是（全部 GEMM 操作数天然 fp32）** | **fp16 hi/lo 3-gemm 拆分**（上轮全战役形态）：x = hi + lo，C = hi@hi + hi@lo + lo@hi（L0C fp32 累加），误差 ~2⁻²² 相对/操作数对。已知限制：操作数绝对值 > 65504（fp16 范围）精度退化——本算子 gemm 操作数为原始 do/v/dv/h/dh（门控 2^g 在 gemm 后 V 侧施加，不进 gemm 操作数），randn/训练域远小于 65504，仅对抗性极端输入退化（Boundary 文档化，§10.3） |
| **scratch 注入红线（本轮新增硬约束）** ⚠️ | **是（强制首版贯彻）** | **禁用 `T.reduce_sum`（reduce_ascend.py；fp32 行归约无条件路由 reduce_2d_v2 带 scratch）与 `T.tile.broadcast`（ascend_tile.py:2507）及任何注入 planner scratch 的 op**——★上轮 D-1/D-2 实证：scratch 的 UB 打包次序按 VarNode 指针键排序（ascend_memory_planning.cc L1018 起），随编译上下文（文件路径/模块文本/堆分配）翻转，坏打包 = aivector Illegal instruction（vec error 0x700000042）交付级崩溃；R-EV 分设复验中基线轮（含 reduce/broadcast 形态）在新环境首例复现。替代形式：① 行向量→矩阵展开 = 转置 + row_expand_sub_experiment（bDK=64）或 T.unroll 行 Sub 链（bDK=32）；② 归约 = 字面量折叠树（逐层折半 T.Parallel add）；③ Σ_v h·dh = C 侧 gemm 对角化（hi/lo 三 gemm）+ 中继 + gather 对角提取；④ m_last 行注入 = 末行单行加。**Stage 2 终检：生成 C grep tmp_ub/Broadcast/reduce_2d/reduce_sum/sharedTmp/scratch 全零命中**（上轮 R4.4 终检形式） |
| **`T.tile.mul_add_dst` 受限**（★上轮 R3 实证） | 已规避 | CANN 9.2.0 `kernel_operator_vec_binary_intf.h:249-250` `__ASC_USE_RESERVED_UBUF__(3510)` 标注；实测 ~1% 确定性偏差（l0 dg 0.9852/0.9857）。mul+add 分离写法 |
| **in-place broadcast→sub→exp 链崩溃**（★上轮 R3 实证） | 已规避 | ACL 507014 设备挂起（sync-insert 对 in-place 序列缺陷）。门控链保持已验证形态（out-of-place 或实证安全的 in-place mul） |
| **同 UB 源双 L1 目的失配**（★上轮实证） | 已规避 | workspace pass 按源 buffer 名解析 V→C 中继写——dw 组不建独立 h 中继副本，**共享 dq 组 h 中继**（同 transpose_B 右角色，单 L1 目的） |
| **TIR `while` / Python 列表下标不可用**（★上轮 D-2 实证） | 已规避 | `while` 常量条件折叠成死循环挂核；shape 相关调度全部用字面量语句（BS=64 域常量折半树）或 TIR range 整行区域 + 常量 extent 或 trace-time `if` / `for bk in range(NK)` |
| **无锚 loop 外 set_cross_flag 丢弃**（★上轮 D-2 实证） | 已规避 | pip 旧版构建静默丢弃无锚 loop 外 flag → drain(0) 死锁；head-set 锚在 prologue **live** 代码（offset 准备链之后） |
| **GM↔UB cast-on-copy 缺陷**（★上轮实证） | 已规避 | 跨 dtype 转换全在 UB↔UB 内两步（GM f32 → UB f32 → UB f16）；本算子无 bf16 |
| **L0C(fp32)→GM 直存**（★上轮实证） | 已规避 | dw 输出同 dtype（f32→f32）直存合法；dq/dk/dgm 走中继（frag→GM f32 同 dtype） |
| **out_idx/workspace_idx 负索引错位**（★上轮实证） | 已规避 | 绝对索引 `out_idx=[19,20,21,22]`、`workspace_idx=[8..18]`（23 参数签名） |
| **UB 硬上限 196352B**（★上轮实证） | 已核算 | produce 装载集 fp32 域 224KB > 196KB → **逐操作数轮次链式装载是结构必需而非优化**（★final_summary 限制 2：单 AIV UB 物理上限，Expert 同步重写不可解锁）；声明集 448KB 靠 MEMORY_PLANNING lifetime 交错复用（单一 liveness 区间纪律，§4.5） |
| host 侧 Buffer 约束（ascend-constraints §5） | 已审计 | §9.6 [HOST-METADATA-AUDIT]：零 view/零 reshape/零 aclnn/零 synchronize |

#### 3.5.2 CUDA 专属机制的昇腾适配（对照上游 fla_chunk_inter_tilelang_gpu.py）

| # | CUDA 机制（上游） | 昇腾适配结论 | 依据 |
|---|------------------|-------------|------|
| 1 | `T.Kernel(ceildiv(DK,block_DK), ceildiv(S,block_S), B*H, threads=128)` 三维 + threads=128 | 一维化两段式：GRID=NT·B·H 任务空间 + fixed-core `T.Kernel(min(20,GRID), threads=1)` 核内 striped 循环 | §3.5.1 行 1-2；★上轮终态 |
| 2 | `T.exp2(x)`（T.Parallel 逐元素内） | **替代**：`T.tile.mul(x, x, LN2)` + `T.tile.exp`（buffer 级；src/op/ascend.cc 仅 ascend_exp 映射）。精度：fp32 下相对差 ~1e-7 ≪ rtol 2⁻¹⁰ | ★上轮实证链路 |
| 3 | `T.gemm(A_shared, B_shared, C_fragment, transpose_B=True, clear_accum=False)`（fp32） | `T.gemm_v0(A_l1, B_l1, C_frag, transpose_B=True, init=)`，操作数 **fp16 hi/lo 拆分 ×3 gemm**；clear_accum→init | †ascend.py:418；★上轮 |
| 4 | `T.clear(dq_fragment)` 预清零 | `init=True` 组首参数化（首 bk 覆盖 / 组内累加） | ★上轮 |
| 5 | `T.reduce_sum(dgkn_shared, dgkn_fragment, dim=1)`（shared→fragment 跨层） | **双重替代**：① 归约限 UB 且禁用（红线）→ ② dgkn 整项移 C 侧 gemm 对角化（diag(h@dh^T)）+ gather 提取 | ★上轮 R4-2 |
| 6 | `T.if_then_else(chunk_last_idx == ..., pp[i_k4], 0.0)`（m_last 行选择） | **末行单行加** `dg[BS-1,:] += pp`（无 LastRow 常量、无 broadcast、无掩码乘） | ★上轮 D-2（P6r PASS，溢出时更贴 golden） |
| 7 | `@autotune`（block_DK×block_DV×threads×num_stages） | **不启用**：block_DK 形状自适应（64/32 二值）、block_DV 废弃（DV 全宽）、threads=1、无 num_stages——全参数约束确定 | ★上轮终态；§5.2 |
| 8 | Triton `tl.where(m_t[:,None], ...)` 行掩码 | 域内 m_t 恒全真 → 无掩码；域外 assert 拒绝（§1.6） | ref_gpu 语义分析（上轮承继） |
| 9 | `tl.store(p_dw, -b_dw)` | dv 装载后 ×(-1) 折叠进 hi/lo 拆分（f16(−x)=−f16(x) 逐位恒等）→ dw 直存即负 | ★上轮 |
| 10 | `out_idx=[-4,-3,-2,-1]` 负索引 | **绝对索引**（workspace 参数注入使负索引错位） | ★上轮 |
| 11 | SIMT block 内天然同步/共享 SMEM | C/V 两类核 + GM 中继架构（无片上直连）→ 显式 workspace + 1-late 手动 flag 协议 | ★上轮；fla_wy_fast final_summary §4 |

#### 3.5.3 本项目同类实现参考（强制步骤 0 结果）

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `../tilelang-dev-journal/fla_chunk_inter/impl_final.py`（★上轮终态交付实证，本轮最直接结构参考） | **最高（同一算子上轮终态）** | fixed-core + NK-merge + 1-late flag + scratch-free 全结构；12 gemm/任务；gather 对角化；形状自适应门控；23 参数签名 |
| `../tilelang-dev-journal/fla_chunk_inter/DESIGN.md` + `debug_log.md` + `perf_tuning/fla_chunk_inter_final_summary.md`（★上轮证据链） | 最高（领域知识） | §8 工具链缺陷清单、D-1/D-2 五位点替换、R2-R4 结构演进与 A/B 数据、六项限制清单（§3.1）与地板核算 |
| `examples/linear_attention_and_rnn/gdn/gdn_chunk_o.py` | 高（GLA chunk O 同构） | chunk×head 一维化 grid、5D 隐状态切片载入（:66）、gemm_v0 init=(i==0)（:67） |
| `examples/developer_mode/flash_attn_bshd_developer.py` | 高（Developer CV 融合标准骨架） | threads/alloc/pass_configs 四件套基础范式 |
| `examples/HISA/paged_block_sparse_mqa_attn_expert.py` | 中（Expert 手动 flag / C 波次化先例） | 手动 cross-flag 协议形态（Stage 3 若探 C 波次化的参考） |
| `examples/tile_kernels/mhc/head_compute_mix_kernel.py` | 中 | 混合核计算形态 |
| `../tilelang-dev-journal/fla_chunk_o/`（★同族交付） | 高（fixed-core + L2 中继配方源） | fixed-core grid + 核内任务循环 + L2 常驻中继（R4 配方 -25%）、每任务同步协议地板核算 |
| `../tilelang-dev-journal/fla_wy_fast/`（★同族交付） | 高（bwd 全 fp32 + hi/lo 拆分链源） | k1 kernel 拆分链/中继/frag→GM 直存、HBM 1263GB/s 标定、工具链缺陷 §8-1~10 |
| `examples_experiment/dsa_sparse_finetune/DESIGN_fla_chunk_intra.md`（同沙箱兄弟设计） | 高 | 同为"全 fp32 + hi/lo 拆分 + V 尾段"家族结构、S%64==0 域声明一致 |

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

> 与首轮首版设计相比**无 LastRow 常量输入**（末行单行加取代，上轮 D-2 后签名 25→23；本轮首版即 23 参数）。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| dq | [B, S, H, DK] | float32 | = dq_raw·scale·2^g |
| dk | [B, S, H, DK] | float32 | = dk_raw·2^(gn−g) |
| dw | [B, S, H, DK] | float32 | = −(dv@h^T)（负号折叠，C 侧直存） |
| dg | [B, S, H, DK] | float32 | = q⊙dq − k⊙dk + m_last·pp |

### 4.3 中间缓冲区（bench 配置 block_DK=64 / DV=128；层级由编译器规划，命名指示预期归属）

**produce 侧（fp32 载入/拆分域 + f16 暂存 + L1 操作数）**

| Buffer | Shape | dtype | 层级 | 用途 |
|--------|-------|-------|------|------|
| h_ub / dh_ub | [64, 128] | fp32 | UB | h/dh 载入域 |
| v_ub / do_ub / dvn_ub | [64, 128] | fp32 | UB | v/do/dv 载入域（dvn 为取负后 dv） |
| t_hi / t_lo | [64, 128] | fp32 | UB | 左操作数拆分暂存（共享，逐操作数复用） |
| th_hi / th_lo | [64, 128] | fp32 | UB | 右操作数拆分暂存（共享） |
| do_hi/lo, v_hi/lo, dvn_hi/lo | [64, 128] ×6 | fp16 | UB | 左操作数 hi/lo（中继前暂存，dead once relayed） |
| h_hi/lo, dh_hi/lo | [64, 128] ×4 | fp16 | UB | 右操作数 hi/lo |
| do_hi/lo_l1, v_hi/lo_l1, dvn_hi/lo_l1 | [64, 128] ×6 | fp16 | **L1** | gemm 左操作数（跨 bk 复用） |
| h_hi/lo_l1, dh_hi/lo_l1 | [64, 128] ×4 | fp16 | **L1** | gemm 右操作数（逐 bk 回读） |
| dq_frag / dk_frag / dw_frag | [64, 64] ×3 | fp32 | **L0C** | dq/dk/dw gemm 累加器 |
| dgm_frag | [64, 64] | fp32 | **L0C** | dgkn 对角化矩阵 h@dh^T |

**drain 侧（与 produce 侧分时复用 UB 空间）**

| Buffer | Shape | dtype | 层级 | 用途 |
|--------|-------|-------|------|------|
| ddr_ub | [192, 64] | fp32 | UB | 融合回读域（行段 dq\|dk\|dgm） |
| dgkn_ub / gn_ub / gnexp_ub | [64] ×3 | fp32 | UB | dgk 终值 / 末行 g / 2^gn |
| dgk_off_i32 / dgk_off_u32 | [64] ×2 | int32/uint32 | UB | gather 字节偏移（prologue 构建） |
| g_ub / gexp_ub / gnd_ub | [64, 64] ×3 | fp32 | UB | g（ln2 预折）/ 2^g（dq 门控 + 转置复用）/ 2^(gn−g) |
| q_ub / k_ub / dkk_ub / dg_ub | [64, 64] ×4 | fp32 | UB | q / k / dk⊙k（负项 + 树归约复用）/ dg 组装 |
| gexp_tmp | [64, 8] | fp32 | UB | row_expand brcb tmp（仅 bDK=64 路径） |

**中继 workspace（GM，框架托管，L2 常驻）——详表见 §8.2**

### 4.4 内存搬运路径

```
GM[h/dh:fp32] --T.copy(5D 切片)--> UB[fp32] --cast/sub/cast 拆分--> UB[hi/lo:fp16] --T.copy--> WS_R 中继(GM/L2)
GM[v/do/dv:fp32] --T.copy(4D 切片)--> UB[fp32] --(dv ×-1)--> 同上拆分 --T.copy--> WS_L 中继(GM/L2)
WS_L --T.copy(C 侧)--> L1[fp16] ─┐
WS_R --T.copy(C 侧)--> L1[fp16] ─┴→ gemm_v0(ᵀB)×12 → L0C[fp32 ×4 frags]
L0C[dw_frag] --T.copy(同 dtype)--> GM[dw]（C 侧直存，负号已折叠）
L0C[dq/dk/dgm_frag] --T.copy--> ddr_relay(GM/L2，手动 flag 40+j%2)
ddr_relay --T.copy(V 侧，1-late wait)--> UB[ddr_ub] --gather--> dgkn ┐
GM[G/Q/K/gn:fp32] --T.copy--> UB[fp32] --exp(·ln2)/mul/折半树/末行加 ┴→ UB[dq/dk/dg] --T.copy--> GM
```

> 层级合法性：无 GM→L0 直搬；Cube 输入经 L1（gemm_v0 内部 L1→L0A/L0B）；Vector 计算全在 UB（fp32 域）；L0C→GM 中继为同 dtype T.copy；跨 dtype 转换全部 UB↔UB 两步。V↔C 交换经 GM 中继（无片上直连，工具链现状）——中继槽位环 L2 常驻（§5.2）使交换流量不落 HBM。

### 4.5 片上内存预算表（910B3：L1=512KB、UB 硬上限 **196352B**、L0A/L0B=64KB、L0C=128KB per core；bench 配置 bDK=64/DV=128）

| 层级 | 构成 | 大小 | 限额 | 结论 |
|------|------|------|------|------|
| L0C | 4 frags [64,64] fp32（dq/dk/dw + dgm） | **64KB** | 128KB | ✓ 50% |
| L1 | 10 gemm 操作数 f16 [64,128]（6 左跨 bk 复用 + 4 右逐 bk） | **160KB** | 512KB | ✓ 31% |
| UB（produce 侧声明） | fp32 域 224KB（h/dh 64 + v/do/dvn 96 + t 64）+ th 64KB + f16 左暂存 96KB + f16 右暂存 64KB = **448KB 声明** | 声明 448KB / **活动峰值 < 196KB** | 196,352B | ✓ 靠 lifetime 复用（逐操作数"载入→拆分→中继"链式序 + 单一 liveness 区间；★上轮终态实证形态：produce 装载集 224KB fp32 > 196KB → 逐操作数轮次链式为**结构必需**，final_summary 限制 2 物理上限论证） |
| UB（drain 侧声明） | ddr 48 + g/gexp/gnd 48 + q/k/dkk/dg 64 + tmp 2 + 标量 ~1 = **~163KB 声明** | 同上 | — | ✓ 与 produce 侧分时（1-late 使 drain(j-1) 与 produce(j) 语义重叠但 planner 对不相交 buffer 集交错排布，★上轮实证可行；【需 Stage 2 验证】编译复现） |
| L2（中继槽位环） | WS_L ×6（1.25MB each）+ WS_R ×4（2.5MB each）+ WS_F（7.5MB） | **~25MB** | 192MB | ✓ 13%，常驻 |

> UB 削峰纪律（Stage 2 实现约束，非可选优化）：① 装载→拆分→中继逐操作数链式序（伪代码顺序固化）；② 拆分暂存 t/th 对全操作数共享；③ f16 暂存 dead once relayed；④ 循环内 alloc + 单一 liveness 区间（不跨循环搬迁）；⑤ 禁语句重排（planner 别名脆弱性，上轮八例实证）。若编译报 UB 溢出回退序：bDK 64→32（声明约减半）→ th 域分时 → 中继写前移。

### 4.6 动态轴定义

无运行时动态轴。B/S/H/DK/DV 均为 JIT 编译期常量（Python 参数传入，随 shape 重编译）——与上游一致；NT/NK/CORE_EFF/L 编译期展开。验收基准为静态 shape（proto 声明）。

### 4.7 JIT 配置

```python
PASS_CONFIGS = {                                              # 四件套全开（★上轮终态同款）
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 C/V 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # 操作数中继（workspace_*）按名自动配对
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动核内同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # UB/L1 lifetime 规划（§4.5 削峰关键）
}
@tilelang.jit(out_idx=[19, 20, 21, 22], workspace_idx=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
              pass_configs=PASS_CONFIGS)
# 绝对索引；workspace 显式回退形态（Developer 复杂场景，§8.2）
```

---

## 5. Tiling 策略

### 5.1 计算类型

**混合（CV 融合）**：Cube（gemm_v0 ×12/任务：4 组 hi/lo 三连）+ Vector（拆分链 ~20 op/任务、exp 门控链、折半树归约、gather 对角提取、行展开门控）。

### 5.2 Block 划分

```python
block_S  = 64                        # 固定 = chunk_size（语义约束，assert chunk_size == block_S）
block_DK = 64 if (DK % 64 == 0 and DV >= 128) else 32   # 形状自适应（★上轮 R1 iter5 引入 + R4 终态保留）
NK   = DK // block_DK                # dk-block 数（任务内 trace-time 展开，非 grid 维）
NT   = S // 64
GRID = NT * B * H                    # 任务 = (b, chunk, head)（NK-merge，★上轮 R2 iter1）
CORE_EFF = min(20, GRID)             # fixed-core（910B3 20 AIC）
L    = GRID // CORE_EFF + 2          # 核内任务循环长度（+2 slack：1-late 尾轮 + redirect redo）
# 核 cid 第 t 轮任务 = cid + t*CORE_EFF（striped）；越界 Select-redirect 重做 cid
```

**选型理由（逐项依据）**：

1. **任务 = (b, chunk, head)（NK-merge）**：v/do/dv 无 DK 轴，合并后每任务装载一次（而非 NK 次）、左操作数拆分链/中继生产一次、C 侧 L1 A 操作数跨 bk 复用——★上轮 R2 iter1 实证：V-mte2 480→384.5KB/(chunk,head) 对、mte2 引擎带宽 47→53GB/s（DMA 突发更大）、mte3 67→87GB/s；
2. **fixed-core(20) + 核内 striped 循环**：替代 GRID=8192 的直接 grid——消除逐任务派发开销，且显式中继槽位环（(CORE_EFF, NSLOT·NK, ...) ≈ 25MB）**L2 常驻**，替代 ~4.5GB 的自动注入槽位足迹（超 L2 192MB）——★上轮 R1 fixed-core 配方（fla_chunk_o R4 同源，-25% @8192 任务规模实证）；
3. **block_DK 形状自适应（64 主选 / 32 fallback）**：bDK=64 减半任务内 bk 数与右中继流量、激活向量化门控路径（256B 行窗）；谓词 "DK%64==0 且 DV>=128" 为纯 metadata 判定，未命中回落 bDK=32 通用路径（DISPATCH-COVERAGE §1.6）——guard 的历史缘由是 [64,64] reduce_2d 损坏（★上轮 R1 iter5 实证），本轮结构已无 reduce_2d，但 guard 保留为已验证干净形状域的保守边界（[64, ≥128] 与 [32, *] 为实证干净域）；
4. **DV 轴单遍全宽**：无 i_v 分块循环（★上轮 R1 iter1 实证：DV 分块被 full-width 单遍替代，每操作数一次 [*, DV] 装载、一次 hi/lo 中继、一次 full-K gemm）；由此支持域 DV ∈ 16 倍数且 ≥64（gemm K=DV ≥ 64 ≥ 32 分形 ✓、DV×4B ≥ 256B 32B 对齐 ✓、kL0Size=128 覆盖 K≤128 域内满配）；
5. **上游 autotune 空间全消解**：block_DK 二值自适应、block_DV 废弃、threads=1、无 num_stages——无搜索空间（Stage 3 枚举空间见 §9.5）。

### 5.3 约束分析

- **分形**：dq/dk/dw gemm M=64 ✓(≥16)、K=DV∈[64,128] ✓(≥32)、N=bDK∈{32,64} ✓(≥16)；dgm gemm M=bDK≥32 ✓、K=DV ✓、N=bDK ✓；
- **对齐（32B）**：v/do/dv 尾块 DV×4 ∈ {256,512}B ✓；h/dh 行 stride DV×4 ✓、尾块 ✓；q/k/g 尾块 bDK×4 ∈ {128,256}B ✓；gn 末行 [bDK] ✓；workspace 尾块同构 ✓；
- **L0C**：64KB ≤ 128KB ✓；**UB**：活动峰值 < 196,352B（声明 448KB 靠 lifetime 复用，§4.5）✓；**L1**：160KB ≤ 512KB ✓；
- **GEMM 非整除**：声明域内无（S%64==0、DK%32==0、DV 全宽）；域外 wrapper assert 拒绝（L2 负向）。

### 5.4 注意事项（尾块）

**声明域内无尾块**：S%64==0（chunked 数据模型数学要求，§1.6 三层证据）、DK 被 bDK 整分（自适应谓词保证）、DV 全宽。域外输入（S%64≠0、DK 非 32 倍数、DV<64 或非 16 倍数、chunk_size≠64、非 fp32）由 host wrapper `_check_shape` 显式 assert 拒绝——Stage 2 场景 B 归入 L2 负向测试，**不设计 host padding + crop**（禁止，ascend-constraints §5）。

**S%64≠0 的 m_t / last(c) 行为**：域外为参考实现未定义域（§1.6）→ 确定性 assert 拒绝。域内 m_t 恒全真（无行掩码）、last(c) 恒为局部行 63（末行单行加精确表达）。

### 5.5 数据搬运性能可行性（REORDER-COST）

本算子无数据重排（bshd/5D 连续直读、无 permute/scatter 语义；gather 仅作 UB 内对角提取读，非 GM 重排），全部搬运为规则 tile 块 DMA：

```text
[REORDER-COST]
path: 单 kernel 主路径 (bDK=64, DV=128, bench shape 1×8192×64×128×128, GRID=8192 任务)
gm_passes: 8 输入各 1 读（唯一 2560 MiB）+ 4 输出各 1 写（1024 MiB）
  + V→C 操作数中继一写一读（f16 hi+lo 字节 = fp32 原字节：左 6 路 16KB + 右 NK×4 路 16KB ≈ 224KB/任务对）
  + C→V 结果中继一写一读（ddr 48KB×NK ≈ 96KB/任务对）
  —— 中继全部 L2 常驻（槽位环 ~25MB ≪ 192MB），不落 HBM
dma_transactions: ~60 次/任务（produce：5 类装载 + 20 中继写 + 10 中继读 + gemm staging；drain：4 装载
  + 1 融合回读 + 3 写）× 8192 任务 ≈ 49 万次（fixed-core 消除任务级派发放大，★上轮 R1 收益）
average_dma_bytes: 逻辑 ~9 GB（唯一 3.76 + 中继 ~5.2）/ 49 万 ≈ 18KB/次（block 级大块 DMA）
gm_scalar_accesses: 0（无标量 GM 访问；所有元素运算在 UB）
address_div_mod_per_element: 0（cid/tid 分解每任务 ~10 次整数运算）
active_cores / serial_tasks_per_core: 20 / ~410（8192/20）
largest_case_timeout_gate: bench shape 单 case 120s（§9.4 perf_bench）
result: pass（无标量/strided GM 主路径；DMA 均 ≥128B 对齐 block 块；中继 L2 吸收不占 HBM 预算）
```

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 任务维（(b, chunk, head) × striped 轮次） | 核内串行任务循环 | `T.serial(L)` | fixed-core 20 核 × L 轮覆盖 GRID 任务（+2 slack：1-late 尾轮 + redirect redo）；无 T.Pipelined（核间流水由 1-late flag 协议在语义层实现，§6.3） |
| DK 维（dk-block） | trace-time 全展开 | `for bk in range(NK)`（Python int） | NK 编译期常量；展开保每 bk 语句序确定（planner 友好）；NK=1 时逐语句退化为单 bk 形态（L1 覆盖不损失） |
| DV 维 | **无循环**（单遍全宽） | — | ★上轮 R1 iter1 实证形态（full-width 装载/中继/gemm） |
| 列归约（pp） | 字面量折半树 | `T.Parallel(32,bDK)` ×6 层（32→16→8→4→2→1） | 无 scratch 红线替代（reduce_sum 禁用）；外层维折叠无 32B 偏移约束 |
| 行展开门控（bDK=32 路径） | 行链 | `T.unroll(BS)` ×64 行 Sub | 无 scratch 红线替代（broadcast 禁用）；行宽 128B < 256B 不满足 row_expand 行窗 |
| 元素级 | 向量化 | `T.tile.*` 全 tile 指令 | ★逐元素 T.Parallel / 逐行 serial 为反模式（-26%~-71%/kernel，家族 perf 教训）——**例外**：折半树的 T.Parallel 逐元素加是归约语义本身（每层一次 masked Add 指令） |

### 6.2 循环伪代码

见 §3.3（prologue 锚 → 单一 T.serial(L) 循环 [produce(j) + drain(j-1) 交织]）。

### 6.3 流水线优化

**核间流水线 = 1-late flag 协议（语义层实现，非 T.Pipelined）**：

- drain(j) 消费 produce(j-1) 的结果中继（ddr_relay），wait 40+(j+1)%2 ↔ set 40+(j-1)%2 配对——★上轮 R2 iter2 实证：0-late 同迭代 flag 使 V 侧暴露 ~3.1ms 串行等待（V 等 C 整条消费链），1-late 后实测归零；
- 槽位安全：drain(j) 的槽 (j+NSLOT-1)%NSLOT 直到 produce(j+3) 才被覆写（NSLOT=4 环深度 > lag 1 + 余量）；
- 交替 flag ID + 每迭代 set/wait 保持 pending < 10（★上轮 R1 lag-10 死锁证据的容量纪律）；
- prologue head-set（flag 41）预充 drain(0) 的 wait（计数语义平衡：一个额外 set 平衡 drain(0)），锚在 live 的 offset 准备代码后（无锚 loop 外 set_cross_flag 被 pip 旧版丢弃，★上轮 D-2 实证）；
- **操作数中继**（workspace_1..10）不走手动 flag——AUTO_CV_SYNC 按名自动配对（同迭代 0-late：C 侧 produce(j) 内回读，V 侧同迭代写，自动 flag 保证序）。

**核内**：无 Pipelined（DV 单遍直线）；produce 的逐操作数链式装载天然与 C 侧 gemm 重叠（中继读发在首用前）。

### 6.4 尾块处理

见 §5.4（声明域内无尾块；域外 shape 显式拒绝）。

---

## 7. 同步策略

### 7.1 同步模式

**混合**：自动同步（四件套）为主体 + **结果中继链手动 cross-flag**（1-late 配对）——Developer 复杂场景回退形态（design-template §8.2 许可；★上轮终态交付形态）。源码仅 3 处手动 flag 语句（set ×2 / wait ×1），无 `T.barrier_all` / `T.Scope`。

### 7.2 同步点说明

| 依赖 | 机制 | 说明 |
|------|------|------|
| UB 拆分/门控 ← GM 装载 | AUTO_SYNC | 核内 DMA→计算 |
| C 侧 L1 操作数 ← V 侧中继写（workspace_1..10） | **AUTO_CV_SYNC 按名配对** | 操作数中继写读自动 flag（同迭代） |
| C 侧 gemm ← L1 操作数就绪 | AUTO_SYNC | C 核内 |
| **V 侧 drain(j) ← C 侧 produce(j-1) 结果中继（ddr_relay）** | **手动 `set_cross_flag("FIX", 40+j%2)` / `wait_cross_flag(40+(j+1)%2, "V")`** | 1-late 配对 + 交替 ID；ddr_relay 命名脱离 workspace_* 前缀逃逸自动 FIFO 配对（防单 ID 锁步） |
| drain(0) 的 wait | prologue `set_cross_flag("FIX", 41)` head-set | 计数平衡；锚在 live offset 准备后 |
| GM 写 ← UB/L0C 结果 | AUTO_SYNC | 核内 |

### 7.3 pass_configs 配置

见 §4.7（四件套全开：AUTO_CV_COMBINE / AUTO_CV_SYNC / AUTO_SYNC / MEMORY_PLANNING）。

---

## 8. CV 融合设计

### 8.1 融合算子判定

**是**（4 组 Cube GEMM（hi/lo ×12/任务）+ Vector exp/逐元素/归约/gather/行展开）。

### 8.2 CV 交互设计（Developer 复杂场景回退：显式 workspace + workspace_idx）

> 判定：本算子 V↔C 交换量大（每任务 ~320KB 中继字节）且需 L2 常驻槽位环 + 1-late 流水——超出 Developer 默认片上直连形态的表达能力，按 design-template §8.2 "Developer 复杂场景回退"设计显式 workspace。**无 vid**（threads=1）；签名 23 tensor（0-7 输入 / 8-18 中继 / 19-22 输出）。

| workspace | Shape | dtype | 用途 |
|-----------|-------|-------|------|
| workspace_1 / 2（do_hi / do_lo） | (CORE_EFF, NSLOT, BS, DV) | fp16 | do hi/lo 左操作数中继（dq 组；单槽/任务） |
| workspace_3 / 4（h_hi / h_lo） | (CORE_EFF, NSLOT·NK, bDK, DV) | fp16 | h hi/lo 右操作数中继（**dq 与 dw 组共享**——同 UB 源单 L1 目的，§8.6；扁平 (task,bk) 槽） |
| workspace_5 / 6（v_hi / v_lo） | (CORE_EFF, NSLOT, BS, DV) | fp16 | v hi/lo 左操作数中继（dk 组） |
| workspace_7 / 8（dh_hi / dh_lo） | (CORE_EFF, NSLOT·NK, bDK, DV) | fp16 | dh hi/lo 右操作数中继（dk 组；扁平 (task,bk) 槽） |
| workspace_9 / 10（dvn_hi / dvn_lo） | (CORE_EFF, NSLOT, BS, DV) | fp16 | 取负 dv hi/lo 左操作数中继（dw 组） |
| ddr_relay（融合 drain 中继） | (CORE_EFF, NSLOT·NK, 2BS+bDK, bDK) | fp32 | 每 (task,bk) 槽行段 [0,BS)=dq \| [BS,2BS)=dk \| [2BS,2BS+bDK)=dgm；**手动 flag**（命名逃逸自动配对） |

**`workspace_idx`**: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]`（绝对索引）；**`out_idx`**: `[19, 20, 21, 22]`。

**槽位纪律**（★上轮 R2 实证）：右中继/ddr 用扁平 (task, bk) 槽（`[cid, slot·NK+bk, ...]`）——bk 间 GM 区域不相交，消除同槽覆写 FIFO 耦合；左中继单槽/任务（每任务单产单消）；NSLOT=4 环深度（cube lag 上界 1 + 余量）。

### 8.3 Cube 核计算流程

见 §3.3 produce 段 C 侧：6 路左中继回读入 L1（每任务一次，A 操作数跨 bk 复用）→ 逐 bk：4 路右中继回读（首用前发射，MTE2 与 gemm 重叠）→ 12 gemm_v0（4 组 hi/lo 三连，init=True 组首）→ dw_frag→GM 直存 + dq/dk/dgm_frag→ddr_relay → set_cross_flag。

### 8.4 Vector 核计算流程

见 §3.3：produce 段 V 侧（5 类装载 + 拆分 + 20 路中继写）与 drain 段（融合回读 + 门控链 + 折半树 + 末行加 + 3 路 GM 写），1-late 交织于同一 T.serial 循环体。

### 8.5 pass_configs 配置

见 §4.7。

### 8.6 注意事项

- **V↔C 无片上直连**（工具链现状，★fla_wy_fast final_summary §4）：全部交换经 GM 中继；中继 L2 常驻（fixed-core 配方核心）；
- **同 UB 源单 L1 目的**（workspace pass 按源 buffer 名解析 V→C 中继写，★上轮实证）：同一 f16 暂存不得中继到两个 L1 目的——dw 组**共享** dq 组的 h 中继（同 transpose_B 右角色），不建 h2 副本（首版设计的 h2 专用副本在中继架构下既无必要也不可行）；
- **ddr_relay 命名纪律**：脱离 `workspace_*` 前缀 → 逃逸 AUTO_CV_SYNC 自动 FIFO 配对 → 手动 1-late flag（否则自动同迭代配对复现 0-late 锁步）；
- **C/V 跳数**：每迭代 produce（V→C ×20 中继写 + C 消费）→ C 出口（1×C→GM + 3×C→中继）→ drain（V）——交替模式固定，无 V→C→V 五跳形态（★上轮 §8-4 sync-point 失配触发形态不存在）；
- **核内 Pipelined 与核间流水互斥**（pass 自动决策）：本设计无 T.Pipelined，核间流水由 1-late 协议承载。

---

## 9. 验证方案

### 9.1 Golden 函数（PyTorch chunk 循环实现，供 Stage 2 直接落地）

> 布局 bshd/5D 与 kernel 一致；golden 用 **fp32 计算域 + torch.exp2 精确语义**（kernel 侧 exp(g·ln2) 替代差异 ~1e-7 相对，计入混合容差验收）。NPU 环境 Triton 不可用（ref_gpu/fla_chunk_inter.py 仅作语义对照，不跑 golden——用户要求 PyTorch 重写，2026-11-02 承继）。
> **⚠️ golden 交叉验证强制项（家族前科）**：★fla_wy_fast debug_log 记录过 golden 自身 einsum 轴序静默错（kernel 对、golden 错）。Stage 2 落地本 golden 后，必须先对一个 chunk 用**显式 per-(b,h) matmul 循环**对照 einsum 三条（dq_raw/dk_raw/dw_blk）+ dgk 对角项逐位验证后再启用（验证代码不进交付）。

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

> Stage 1 只产出 L0（规则 shape：S%64==0、DK%32==0、DV∈16 倍数且 ≥64——全部被 block 精确整除）。L1（功能，含域内其他 shape 组合如 DK=96/DV=96、NK=1 退化路径）/ L2（异常拒绝，含 S%64≠0、DK 非 32 倍、DV<64、chunk_size≠64、非 fp32）/ Boundary（特殊值，含 g 极值 2^g 溢出、gemm 操作数 >65504 的 hi/lo 范围限制）由 Stage 2 在 L0 通过后按场景 B 扩展。
> 算子类别判定（场景 A Phase 2，依据本 DESIGN §1.3/§1.4/§2/§4/§5/§9.3）：计算类型 = **混合（CV 融合：4 组 fp16 hi/lo GEMM + Vector exp2 替代链/逐元素/折半树归约/gather 对角提取/行展开门控）**；复杂度 = **Fusion**（4 类 GEMM + 门控 + 行注入 + 4 输出）；数学特征 = matmul + element-wise gating + reduction；综合类别 = **Fusion（gated chunked attention 反传 inter-chunk 梯度）**→ 测试策略聚焦：CV 协作正确性（V→C 操作数中继 / C→V 结果中继 + 1-late flag 协议）、fixed-core striped 分解与 Select-redirect 幂等性、chunk 语义（末行注入 / gn 末行切片）、NK-merge 跨 bk 展开与 gather 对角提取、hi/lo 拆分精度、block_DK 自适应双路径、4 输出独立判定；dtype 无变体空间（全 fp32 固定组合）。

| 用例名 | 级别 | Shape (B,S,H,DK,DV) | dtype | 说明 |
|--------|------|---------------------|-------|------|
| l0_min | L0 | (1, 128, 2, 64, 64) | 全 fp32 | 最小冒烟：NT=2（双 chunk 末行注入 ×2）、BH=2、DK=64/DV=64 → **block_DK 自适应 fallback 路径**（bDK=32：T.unroll 行链门控 + [32,*] 干净域）、GRID=4 任务 < CORE_EFF=20（clamp 生效 + L=BASE+2=3 轮 + redirect-redo 覆盖）。覆盖全链路：5 类装载、hi/lo 拆分链、12 gemm/任务（init 组首/累加）、NK=2 跨 bk 展开与扁平槽位、dgm 对角化 + gather 提取、gn 末行 1D 切片、ln2 预折门控链、折半树归约、末行单行加、dw 取负折叠直存、融合回读 + 1-late flag + head-set、4 输出 |
| l0_repr | L0 | (2, 256, 4, 128, 128) | 全 fp32 | benchmark 结构代表：**block_DK=64 主路径**（DK%64==0 且 DV≥128：transpose+row_expand 向量化门控 + [64,128] 满配 UB + 256B 行窗）、B>1 触发 bb/bh 分解、GRID=2×4×4=32 任务 > 20 核（striped 多轮 L=1+2=3：跨轮槽位环 NSLOT 环回 + 1-late 跨轮配对覆盖）。全量程 fp32 + hi/lo 精度路径 + 12 gemm×NK=2 |

**golden 对照法**：每用例对 4 个输出独立调用 `check_precision(out, ref, "float32")`（precision-standard.md §4.1 混合容差判定，Stage 2 原样复制不改动）；全部通过打 `[PRECISION_PASS]`，任一失败 `[PRECISION_FAIL]`（阻塞，计入退出码）。

### 9.3 精度标准（混合容差，precision-standard.md 口径）

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|-------|------|------|---------------------|------------------------|
| float32（dq/dk/dw/dg 全部输出；输入全集亦仅 fp32） | 2⁻¹⁶ (1.53e-5) | 2⁻¹⁰ (9.77e-4) | 1e-2 | 0.99 |

> 与 proto_fla_chunk_inter.yaml 声明的门禁逐字一致（inputs dtype 全集 = {float32}，与上行一致，无第二个 dtype 行）。特殊值（NaN/Inf）位置契约：输入域为 randn 有限值（上游对拍口径）；g 极值（g>128 时 2^g=inf，出现在 V 侧 fp32 门控域，kernel 与 golden 同为 inf → 结构比对一致）与 gemm 操作数 >65504（fp16 hi/lo 范围限制）属 Boundary 范畴，Stage 2 场景 B 扩展（"有限值 + 稀疏特殊值"混合输入 + mask 结构比对，precision-standard §3.1）。
>
> 误差预算【推断】：kernel 相对 golden 的差异 = fp16 hi/lo 拆分（~2⁻²² 相对/操作数对，DV=128 累加后绝对误差 ~3e-6）+ exp2→exp(g·ln2)（~1e-7 相对）+ ln2 预折舍入（~2e-7 相对）+ gemm L0C fp32 累加（精确）+ dgkn 对角化（hi/lo 类误差，同其他 gemm 输出，★上轮 R4 实证 valrange_l dg 7.812e-3 贴 1e-2 帽仍有余量）。输出量级 ~11（randn、DV=128 累加），逐元素容差 ≈ atol + rtol·|out| ≈ 0.011，典型误差 ~3e-6——余量 ~3000×，matched_ratio 预期 ≈ 1.0。对照被否的 hi-only 单 cast 方案（误差 ~5.5e-3，小 |out| 元素 atol 主导区失败，估算 ratio ~90-93% < 0.99，家族两例实证 FAIL，§10.4-C）。

### 9.4 性能可行性哨兵（强制，不可跳过）

| 用例名 | Shape (B,S,H,DK,DV) | dtype/属性 | 覆盖路径 | 单 case timeout | 选择理由 |
|--------|---------------------|------------|----------|-----------------|----------|
| perf_bench | (1, 8192, 64, 128, 128) | 全 fp32, scale=1.0, chunk=64 | 单 kernel 主路径（GRID=8192 任务 = 最大 striped 轮次深度 ~410/核、最大 DMA 吞吐压力、bDK=64/DV=128 满配 UB、12 gemm×NK=2/任务满载、中继槽位环满环回） | 120s | benchmark 验收 shape（proto 性能目标口径） |

计时口径：对齐上游 `ref_gpu/test_utils_kda.py do_bench`——**warmup=20、rep=10、取均值、仅设备侧 kernel 时间**（NPU 事件计时等价实现），不含张量构造/H2D-D2H/JIT/框架开销。wrapper 内**无** `torch.npu.synchronize()`（家族优化项：内部 sync 把 host JIT 分发延迟暴露进设备侧计时，与口径相悖）。测试数据、随机数与 golden 均在 CPU 完成后 H2D，运行段无 aclnn 依赖。本轮全部性能运行经 cannlab CLI 远程驱动 NPU 执行（分设架构，Stage 2/3 落实）。

### 9.5 性能预算分析（910B3，含 3400us 目标可行性结论）

**IO 明细（物理唯一字节，每张量单读单写；bench shape 1×8192×64×128×128）**

| 项 | 字节 |
|----|------|
| 读 | q/k/g 各 256 + v/do/dv 各 256 + h/dh 各 512 = **2560 MiB** |
| 写 | dq/dk/dw/dg 各 256 = **1024 MiB** |
| 合计（唯一 HBM） | **3584 MiB ≈ 3.758 GB**（该数据布局下的 HBM 下限，无 kernel 层压缩空间） |
| GEMM FLOPs | 51.5 GFLOP 逻辑 fp32 × 3（hi/lo）= **154.6 GFLOP 物理 fp16**（@157 TFLOPS 折算 ~985us，与 IO 重叠，非独立下限） |
| Vector | ~67M exp + 逐元素 ~0.5 GFLOP（tile 指令）——非瓶颈 |

**逻辑流量分解（V↔C 全经 GM 中继、L2 常驻吸收）**

| 项 | 字节 | 说明 |
|----|------|------|
| 唯一 HBM | 3.76 GB | 下限 |
| V→C 操作数中继 | ~1.8 GB（单向；写+读逻辑流量 ×2 ≈ 3.6 GB） | 20 路 f16 hi/lo/任务（左 6×16KB + 右 NK×4×16KB = 224KB/任务单向）× 8192 任务；f16 hi+lo 字节 = fp32 原字节（hi/lo 不增中继量） |
| C→V 结果中继 | ~0.8 GB（单向；写+读逻辑流量 ×2 ≈ 1.5 GB） | ddr 48KB×NK/任务单向（R4 融合后单读） |
| 中继合计 | ~2.6 GB 单向 / ~5.2 GB 往返 | **L2 常驻槽位环吸收（~25MB ≪ 192MB），不落 HBM**（fixed-core 配方核心，★上轮实证） |
| 逻辑总量 | **~9 GB**（唯一 3.76 + 中继往返 5.2） | 对照首版结构 ~17GB：NK-merge 消除 v/do/dv 重读 + 任务数 32768→8192 |

**三模型对照（设计期估算，perf_bench 哨兵实测判定）**

| 模型 | 估算 | 依据 |
|------|------|------|
| ① HBM 唯一 IO 下限 | 3.758 GB ÷ 1263 GB/s = **2976 us** | 实测带宽上限（★fla_wy_fast D2D 标定） |
| ② 上轮终态参照值 | **6868.3 us**（do_bench 5 次均值）/ 6673.2 us（msprof Task Duration）| ★上轮 R4 终态（本设计结构等价物）；R-EV 分设环境复测 6884.7 us（+0.24% 噪声内——设备侧计时跨环境可复现） |
| ③ Developer + scratch-free 双约束形态地板 | **≈ 6868 us（上轮实测、待本轮复验）** | ★上轮 R4 §15.2 结构地板归因（dgm 中继回读 ~291us + 门控 2 次转置 ~328us + drain 树 ~170us + mte2 字节地板 416KB/task 全部实证封堵）——本设计结构等价物的预期终态量级 |

**结论：3400 us 目标【高风险 / at-risk】，需 perf_bench 实测判定**：

1. **目标低于双约束形态地板**：3400us < 上轮实证的 Developer + scratch-free 形态地板 ~6868us（差距 2.02×）。上轮六项限制清单（★final_summary §3.1，证据链完整）表明该地板之下每条通路均被 ≥2 项独立上游修复或 razor-edge 手工装配条件卡死：① threads=2（40 AIV 唯一入口）的 vid 语义对本算子 drain 数学结构性破坏（微实验 5 子链 + 源码双证，需上游三项修复）；② UB 196KB 容量物理上限强制 produce 逐操作数链式（Expert 重写不可解锁）；③ C 消费链 10.9us/任务由 V↔C GM 中继架构强制（无片上直连，需上游立项级修复或全量 Expert 波次化——家族 4 试 3 败区）；④ launch 形态仅 MIX_AIC_1_1/1_2；⑤ 目标隐含 88% HBM 唯一 IO 效率 > 家族实证最佳 81%（绝对物理界 2975us@100% / 3673us@81%）；⑥ 指令/规划器缺陷族（mul_add_dst 受限、in-place 链崩溃、planner 别名脆弱）；
2. **本轮结构性有利项**：首版即终态结构 → Stage 2 新基线预期 ~6.8-6.9ms 量级（对照上轮首版 19.6ms，省去四场战役的爬坡）；地板数字为上轮合设环境实测，**待本轮分设环境复验**（R-EV 已证设备侧计时跨环境可复现，预期一致）；
3. **Stage 3 调优方向**（本轮在新基线上重跑，按预期收益排序）：① 新环境复测锚点（cannlab 口径对齐 do_bench 20/10/mean 设备侧）；② 上轮 R4 已封堵方向的**增量复核**（planner 指针序若上游已修复可回收 ~500us 级——scratch 注入 op 本轮红线禁用，该项仅在上游修复后由 Stage 3 评估解禁）；③ C 波次化（HISA 式 Expert 手动 flag，`examples/HISA/` 先例）为唯一理论可达路径（2.9-3.4ms razor），家族政策下如实评估不强行；
4. 若 Stage 3 后仍未达标：按家族政策**如实披露**（★fla_chunk_o 超 7.5% 交付先例）——报告带宽效率口径（唯一 IO ÷ 时间 ÷ 1263GB/s）+ 地板核算 + 限制清单，不做任何破坏精度/正确性的达标动作。

### 9.6 Host 侧 Buffer 操作审计

```text
[HOST-METADATA-AUDIT]
operation: 输入构造 torch.randn(...).npu()（CPU 生成后 H2D，数据准备）
  + kernel 调用（12 张量直传：8 输入 + 4 输出 out_idx 框架分配；中继 workspace 由框架按
    workspace_idx 自动分配，host 零触碰）
  + 精度验证（D2H 后 CPU 侧对拍 check_precision）
input_stride -> output_stride: N/A（新建张量，非变换；kernel 直读 bshd/5D 连续布局，
  无任何 view/reshape/permute）
shares_storage / same_data_ptr: N/A（无变换操作）
aclnn_or_physical_copy: false（randn 为 CPU 数据生成 + 一次性 H2D；kernel 调用前后无
  .contiguous()/pad/cat/reshape 物理化/.to(dtype)/.clone()；wrapper 内无 torch.npu.synchronize()）
result: allow
```

---

## 10. 风险点与注意事项

### 10.1 风险清单（高/中/低分级）

| 级别 | 风险 | 影响 | 缓解/验证 |
|------|------|------|-----------|
| **高** | R1 scratch-free 红线的首版贯彻（任何 reduce_sum/broadcast 形态引入 = 交付级崩溃彩票，与代码正确性无关，D-1/D-2 + R-EV 三重实证） | 移植期崩溃难定位；交付后环境迁移即崩（R-EV 首例复现） | 首版强制（§3.5.1 红线表）；**Stage 2 终检门禁：生成 C grep tmp_ub/Broadcast/reduce_2d/reduce_sum/sharedTmp/scratch 全零命中**（上轮 R4.4 终检形式，两 shape 类均查） |
| **高** | R2 性能目标 3400us at-risk（目标低于 Developer + scratch-free 双约束形态地板 ~6868us，2.02×；六项限制清单每条通路需 ≥2 项联合解锁） | 性能验收失败 | §9.5 三模型论证 + 首版即终态结构（新基线 ~6.8ms 起步）+ perf_bench 哨兵实测 + Stage 3 重跑 + 家族如实披露政策 |
| 中 | R3 UB 声明 448KB vs 硬上限 196352B（活动峰值靠 lifetime 复用；produce 装载集 224KB fp32 > 196KB 物理上限） | 编译溢出 / 布局翻转 | 逐操作数链式序 + 单一 liveness 区间纪律（§4.5 削峰五条，上轮实证形态）；回退序：bDK 64→32 → th 分时 → 中继写前移；Stage 2 编译验证 |
| 中 | R4 memory-planner 别名脆弱性（上轮三战役八例实证：合法语句重排即崩溃或屏障劣化） | 修订期崩溃 / 性能劣化 | 禁语句重排纪律；循环内 alloc；保持伪代码固化的语句序；A/B 对照基线锚点法 |
| 中 | R5 手动 flag 协议正确性（head-set 锚 / 计数平衡 / lag 容量 / 槽位环深度） | 死锁（drain(0)）/ 覆写竞态 | 上轮实证配方原样内建（1-late 配对 + NSLOT=4 + 交替 ID + pending<10 + live 锚）；l0_min 首验（GRID=4 < 20 边界）+ l0_repr 覆盖跨轮环回 |
| 中 | R6 golden einsum 轴序（家族前科：golden 静默错、kernel 正确 → 假 FAIL/假 PASS） | 精度判定失效 | §9.1 强制交叉验证（显式 per-(b,h) matmul 对照单 chunk 逐位）后才启用 golden |
| 低 | R7 exp(x·ln2) vs exp2 + ln2 预折（合计 ~3e-7 相对） | 精度微小 | ≪ rtol 9.77e-4 四个量级；golden 用 torch.exp2 对拍验证 |
| 低 | R8 S%64≠0 语义 | 域外误用 | 已论证为参考实现未定义域（§1.6 三层证据），wrapper assert 确定性拒绝 |
| 低 | R9 gemm 操作数 >65504（fp16 hi/lo 范围） | 极端输入精度退化 | randn/训练域不触发；Boundary 文档化（Stage 2） |
| 低 | R10 分设环境执行链（cannlab CLI 远程构建/运行；30 分钟平台空闲自动关机教训——长构建需交互心跳保活） | Stage 2/3 执行中断 | ★R-EV 实证教训注入 Stage 2/3 操作手册（240s 轻量 exec 心跳）；Stage 1 不涉及 |

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| `T.reduce_sum` / `T.tile.broadcast` 任何形态 | 照抄上游 reduce/掩码语义 | planner 注入 scratch → UB 打包彩票 → Illegal instruction（D-1/D-2 红线） | 折半树 / 行链 / 末行单行加 / C 侧对角化 + gather（§3.5.1 红线全表） |
| fp32 gemm_v0 直用 | 照抄上游 fp32 T.gemm | 1e37 级交替垃圾（工具链损坏） | fp16 hi/lo 3-gemm 拆分（已内建） |
| `T.exp2` 直用 | 照抄 Triton 语义 | 无后端映射 | `T.tile.mul(x, x, LN2)` + `T.tile.exp` |
| threads=2 | 调优尝试唤醒 40 AIV | drain 数学结构性损坏（5 子链微实验闭合） | 保持 threads=1；该通路需上游 vid 三项修复 |
| `T.tile.mul_add_dst` 融合 | mul+add 算子融合 | ~1% 确定性偏差（CANN 受限指令） | mul/add 分离写法 |
| in-place broadcast→sub→exp 链 | UB 削峰尝试 | ACL 507014 设备挂起 | 保持已验证形态（out-of-place 或实证安全 in-place mul） |
| out_idx / workspace_idx 负索引 | 照抄上游 [-4..-1] | 参数错位（workspace 注入使负索引错位） | 绝对索引 [19..22] / [8..18] |
| 无锚 loop 外 set_cross_flag | prologue 空置 head-set | pip 旧版静默丢弃 → drain(0) 死锁 | head-set 锚在 live offset 准备代码后 |
| TIR `while` / Python 列表下标 | shape 相关循环 | 常量条件折叠死循环挂核 | 字面量语句 / TIR range + 常量 extent / trace-time if |
| 同一 UB 源中继到两个 L1 目的 | dw 组建独立 h 中继 | workspace pass 按源名解析失配 | dw 组共享 dq 组 h 中继（同右角色） |
| 语句重排（planner 别名） | 后期优化微调顺序 | 崩溃或屏障劣化（八例实证） | 禁重排；A/B 锚点法验证任何变更 |
| host 侧 aclnn（pad/cat/reshape 物理化/.to/.clone/synchronize） | wrapper "保险" 代码 | 评测环境 aclnn 裁剪即失败 / 计时口径污染 | 全部 kernel 内完成；wrapper 仅 assert + 直传 |
| wrapper 内 `torch.npu.synchronize()` | 计时"保险" | host 分发延迟暴露进设备侧计时 | 移除（do_bench 口径） |

### 10.3 特殊场景处理

- **S%64≠0（m_t / last(c)）**：域外，参考实现未定义（§1.6）→ wrapper assert 拒绝，L2 负向；域内无掩码、last(c) 恒局部行 63；
- **极小 shape**（S=64, NT=1；或 GRID < 20 触发 CORE_EFF clamp）：GRID=4 时 CORE_EFF=4、L=3——redirect-redo 与 flag 计数平衡仍成立（l0_min 即此形态）；NT=1 时末行注入只在唯一 chunk 末行——合法；L0 不含 S=64（用 S=128 保证 NT≥2，触发多 chunk 末行注入差异）；
- **NK=1 退化**（DK=bDK）：`for bk in range(1)` trace-time 展开为单 bk 直线形态（与上轮 R1 终态逐语句等价）——L1 套件覆盖（Stage 2 场景 B）；
- **非 2 幂 D**：DK=96（bDK=32 路径 NK=3）✓ 域内；DV=96（gemm K=96，分形/对齐 ✓）✓ 域内——均留 L1 扩展（Stage 2 场景 B）；
- **g 极值上溢**（g>128 → 2^g=inf）：V 侧 fp32 门控域，kernel 与 golden 同为 inf → Boundary 结构比对（Stage 2）；gemm 操作数不受影响（g 不进 gemm 操作数——与 fla_wy_fast 的结构性差异，精度面更安全）；ln2 预折 ±200 边界溢出/下溢路径逐位一致（★上轮 R2 iter3 实证）；
- **操作数 >65504**（fp16 hi/lo 范围）：Boundary 文档化（家族同款已知限制）；训练域/上游 randn 对拍域不触发；
- **pp 溢出**（2^gn·Σ h·dh + Σ dk·k 超 fp32）：末行单行加形态在溢出时**更贴 golden**（无非末行 0·inf=NaN——golden 行索引加法同构，★上轮 D-2 实证 bd_g_gn_overflow WARN→PASS）。

### 10.4 被否方案（设计决策记录）

| # | 方案 | 否决理由 |
|---|------|---------|
| A | reduce_sum/broadcast 向量形式（首轮首版 §3.3 形态：dim=0/dim=1 UB reduce + LastRow 常量 + broadcast 掩码乘 + Pipelined(2) i_v 分块 + GRID=NK·NT·B·H bk-最快） | **D-1/D-2 布局彩票**（scratch UB 打包随编译上下文翻转 → Illegal instruction 交付级崩溃；R-EV 新环境首例复现）——本轮红线禁用，全部替换为 scratch-free 形式；Pipelined/GRID 形态亦被上轮 R1-R2 结构演进取代 |
| B | DV 维上 grid（任务 = dv-block，i_k 循环内累加） | 输出沿 DV 不因子分解（dq_raw = Σ_DV 收缩）→ 跨任务部分和 + GM partial 二次归约（+1GiB IO + 额外 kernel）+ g/q/k 重复读。结构性劣化，否决 |
| C | hi-only 单 f16 cast（1× gemm，12→4 gemm/任务） | 误差预算：f16 单次相对误差 ~2⁻¹¹/操作数，DV=128 累加后误差 std ~5.5e-3，小 \|out\| 元素（atol 主导区）系统性失败，估算 matched_ratio ~90-93% < 0.99；家族两例实证（fla_chunk_o 0.988 FAIL、fla_wy_fast 超差）。精度否决——hi/lo 3-gemm 强制 |
| D | host 预拆分 f16 hi/lo 双 kernel（先 Vector cast 写 GM，再 Cube kernel 直读 L1） | +4.1GB IO 且不减少 gemm 侧中继（f16 hi+lo 字节 = fp32 原字节）→ 纯增开销，否决 |
| E | 3-kernel 拆分（dq/dk/dw 组分 kernel + dg 尾段 kernel） | dg 依赖 dq 与 dk 的双门控结果 → 需 GM partial 中转 + 3× JIT launch；仅作正确性回退（不作主路径） |
| F | host 侧预计算 2^g / 常量折叠 scale·2^g | aclnn 违规（ascend-constraints §5）+ g 256MiB 读翻倍；scale·2^g ≠ 2^(g+log2 scale) 精确性——kernel 内 fp32 域计算，否决 |
| G | threads=2（唤醒 40 AIV） | vid 语义对 drain 数学结构性破坏（★上轮 R3 微实验 5 子链闭合 + ascend_vid_reduction.cc 源码双证；launch 形态仅 MIX_AIC_1_1/1_2，grid=40 恒两波串行）——需上游三项修复方可解禁 |
| H | mul_add_dst 融合 / drain in-place 链 / wholereduce 形态（上轮 R3/R4 全部候选回滚项） | 逐一实证否决（~1% 偏差 / ACL 507014 挂起 / 不跨 repeat 累加）——本轮不重试 |
| I | Expert 全量重写（V 侧手动三引擎流水 / C 波次化） | vchain 微实验 yield -3.1%（假设证伪）；UB 196KB 物理上限；波次化 2.9-3.4ms 为 razor 理论、家族 4 试 3 败——不作首版路径，列 Stage 3 唯一理论可达方向 |

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
| `proto_fla_chunk_inter.yaml` | ✅ 已核对一致 | 承担 proto.yaml 角色：`operator.inputs[].dtype` 全集 = {float32}（与 §9.3 精度表唯一 fp32 行一致）、attrs 覆盖 chunk_size/scale、schema 完整（8 输入 + scale/chunk_size → 4 输出；23 tensor 内部签名的 workspace 为框架托管机制不进算子接口）、性能门禁 <3400us@bench shape——经逐项核对满足覆盖门禁 `coverage_check.py --proto` 派生要求，**无需改动**（本次全 fp32，与任务书预期一致） |
| `fla_chunk_inter.py` | ⬜ 待实现 | `fla_chunk_inter`（JIT 单 kernel，Developer 复杂场景回退形态，§4.7 pass_configs，out_idx=[19..22]/workspace_idx=[8..18]）+ `_check_shape` assert + block_DK 形状自适应 + 红线 grep 终检 |
| `test_fla_chunk_inter.py` | ⬜ 待实现 | golden（§9.1 逐行落地 + 强制交叉验证）+ `check_precision`（precision-standard.md §4.1 原样复制）+ L0 两用例（§9.2）+ perf_bench 哨兵（§9.4）+ `--level` main 分发器 |

### 11.3 命名规范

- kernel 文件 `fla_chunk_inter.py`、测试 `test_fla_chunk_inter.py`（沙箱目录内算子前缀，符合迁移任务硬性约束）
- JIT kernel 函数名 `fla_chunk_inter`（`from fla_chunk_inter import fla_chunk_inter`）；host 语义名 `fla_chunk_inter_bwd`（proto schema 口径）作为测试侧别名

### 11.4 实现顺序

1. ✅ 设计文档 + L0 门槛测试计划 + proto 一致性核对（本文档）
2. ⬜ `fla_chunk_inter.py`：kernel（§3.3 伪代码逐段落地；先 L0 编译过；**红线 grep 终检零命中后再提交精度**）
3. ⬜ `test_fla_chunk_inter.py`：golden（含 §9.1 交叉验证）+ L0 + main
4. ⬜ L0 门槛测试通过（精度收敛；check_precision 4 输出全绿；经 cannlab 远程 NPU 执行）
5. ⬜ perf_bench 哨兵（§9.5 三模型实测判定；新基线锚点 = 本轮 Stage 3 起点；未达标触发 Stage 3 战役）
6. ⬜ 扩展分层套件 L1/L2/Boundary（tilelang-op-test-design 场景 B；含 S%64≠0 负向、DK=96/DV=96、NK=1 退化、g 极值、>65504 操作数）
7. ⬜ 全量套件运行（L0/L1 须通过；L2/Boundary 失败仅记录不阻塞）+ 覆盖门禁 `coverage_check.py --proto proto_fla_chunk_inter.yaml`

### 11.5 算子 proto.yaml 一致性说明（覆盖门禁用）

> proto.yaml 角色由 `proto_fla_chunk_inter.yaml`（Orchestrator 预写任务书）承担。本设计完成后逐项核对：**dtype 全集**（§9.3 精度表唯一 float32 行 = proto `inputs[].dtype` 全部 [float32] ✓）；**attrs**（chunk_size/scale = proto `attrs[].name` ✓，均影响计算路径：chunk_size 定 BT + assert 域、scale 进 dq 门控 trace-time 折叠）；**schema**（8 输入 + scale/chunk_size → 4 输出 ✓；kernel 内部 23 tensor 签名的 11 个中继 workspace 为 `workspace_idx` 框架托管机制，非算子接口参数——与首轮 LastRow 常量不同，本轮无任何设计新增 kernel 输入）。核对结论：**proto 无需改动**（与任务书"本次全 fp32，预期无需改动"一致）。
