# sparse_mla_bwd 算子设计文档（rev3 · Developer pair-loop 重构版 + Expert fallback）

> 迁移来源：`examples_experiment/dsa_sparse_finetune/ref_gpu/sparse_mla_bwd.py`（GPU 版主仓参考，343 行，md5 fc4e8e5c9fd9d5a36e0c3a79c10b9a85，只读，语义权威）；配套依赖 `ref_gpu/sparse_mla_fwd.py`（325 行，只读）、`ref_gpu/utils.py`、`ref_gpu/index.py`
> 目标平台：Ascend 910B3（20 Cube Core + 40 Vector Core；每核 L1 512KB / UB 192KB / L0C 128KB / L0A 64KB / L0B 64KB；全芯片 L2 192MiB——锚点 `$ASCEND_HOME_PATH/aarch64-linux/data/platform_config/Ascend910B3.ini`：`l1_size=524288 / ub_size=196608 / l0_c_size=131072 / l2_size=201326592 / cube_core_cnt=20 / vector_core_cnt=40`）
> 编程模式：**Developer 形态 pair-loop 主路径 + Expert fallback（rev3 修正版，2026-10-15）**——k2 bwd 主循环以 indexer_bwd 同款 pair-loop 生产/消费错位结构重写（零 Expert API：alloc_shared/alloc_fragment + workspace 命名自动握手 + 4-True pass_configs，AUTO 栈对该结构编译出 C 领先软件流水）；Expert 手动 flag 仅为锁步复现时的升级路径（升级判据：复测 msprof AIC 等 flag 占比 > 30%，§4.2）；k3 postprocess 亦 Developer 形态（k1 已折叠入 bwd）
> 双后端约束（用户硬性要求）：**ascendc（target="ascendc"，默认）与 PTO（target="pto"，显式）双后端均须通过编译 + 运行 + 精度验证**；性能以较优后端考核，另一后端如实报告不设门槛
> 性能目标：全接口（delta=None 两段发射口径，amend-1）在 B=1, S=2048, H=64, HKV=1, DQKV=576(D=512, D_tail=64), DV=512, topk=512, bf16 下 **< 7680 us**；GEMM 总量 ~370 GFLOP
> 外部依赖：`sparse_mla_fwd` 由并行会话交付（DESIGN_sparse_mla_fwd.md 已落盘，只读协调）；本算子禁止自行实现 / 修改 fwd（依赖策略见 §11）

> **版本修订（amend-1，2026-10-14）**：Δ 恒等式 + 值域门限校准——`delta=None` 时 kernel 内以 Δ=Σ_j P·dP_raw（fp32 先于 cast 量）替代主仓 Δ=Σ_d o·do 路径；host 发射段数 3→2（k1 不再发射，详见 §1.3 恒等式小节与 §11.3）。
> **版本修订（amend-2，2026-10-14）**：§12.2/§12.3 值域门限收紧——域 B 门限 sim ≥ 0.97 → **sim ≥ 0.999**；valrange-m（σ=10）升入域 B 阻塞判据；valrange-l（σ=100）维持域 C WARN。rev3 全文继承此口径（§12）。
> **版本修订（rev2，2026-10-14）**：Developer AUTO 栈全量实现（历史全文 `history_version/design_sparse_mla_bwd_rev2.md`，952 行）。Stage 3 iter-0/1/2 实测 54.9ms → 32.6ms → 28.5ms 后触顶，iter-2 以 [DESIGN_ERROR] 回退（Orchestrator 已裁决）。
> **版本修订（rev3，2026-10-15，本文；§2 方向修正版）**：设计回退重做（revision_index=2）。回退证据与 rev3 架构方向见文首 §0 与 §4：① AUTO_CV_SYNC 锁步结构固化（AIC 73% 等 flag）＋ ② AIV 读量 ~147MB @ 12.2GB/s 饱和 → 带宽下限 ~12ms > 7680us ＋ ③ rev2 结构内 6 条绕行全部证伪（§3.7）。rev3 响应（任务书 §2 修正后口径）：**bwd 主循环重构为 indexer_bwd 同款 pair-loop 生产/消费错位形态（Developer 形态 + 4-True pass_configs，Expert 手动 flag 降为 fallback）**——rev2 死因重定性为"两遍立即消费的循环结构"而非 AUTO 栈本身（indexer_bwd 零 Expert API 70.9us 实证）；dKV 散射按 §6 双后端源码级查证选型（随机行寻址能力二选一，C 侧 L0C→GM 写仅用于可连续化腿——贡献槽）；transit 腿融合压缩（AIV 总读量目标 ≤~75MB）。**接口签名 / amend-1 Δ 恒等式 / amend-2 分层精度门限 / 全部测试资产（test_sparse_mla_bwd.py 用例与 golden）原样保留**，rev3 实现须直接通过现有套件。

---

**文档生成状态（Stage 1 rev3 分块交付）**：本文档按 rev3 任务书分 4 块生成——块 A（文首 + §1-§4）已交付；块 B（§5-§7）已交付（§6 散射查证裁决 + §4.4/§4.5 就地修正）；块 C（§8-§11：§8 内存规划 pair-loop 双缓冲预算/stash 槽布局/§6.5 工作集勘误 + §9 同步与流水 flag 记账/时序图/Expert fallback + §10 动态 shape/varlen + §11 k3 残余优化/fwd 状态更新）已交付；**块 D（§12-§14 + proto 校验，本次交付，rev3 全部 4 块完成）：§12 精度容差（amend-1/amend-2 分层门限与 L0 9 用例计划原样继承 + §12.6 rev3 专项：pair-loop 槽复用正确性协议、bf16 槽档门控复验、fallback 精度等价）+ §13 性能预算 rev3（逐腿读量表 62-70MB 口径/L5 fp32 兜底 88-96MB、算力/带宽下限、pair-loop 流水预期、7680us 可行性账面临界/置信度中、rev2 28541us 对照）+ §14 风险与 fallback（§14.1 rev2 六条证伪防重蹈表、R13-R17 新增 + R1-R12 重估、§14.3 fallback 决策树、目标不可达诚实出口）；proto_sparse_mla_bwd.yaml 校验通过（接口未变，无需重写）**。后块不得推翻前块决策（既有例外：任务书 §2 明示的散射选型开放问题已由块 B §6 裁决；块 C 的修正为数字勘误与 API 名查证订正——§6.5 工作集 5.9→23.0MB、§4.2 "AUTO_CV_COMPILE" 名称笔误订正（PassConfigKey 无此项，以 4-True + indexer_bwd L275-279 为准）——决策语义均不变；块 D 的修正同性质：§5.5 "297MB" 读+写合计粗口径勘误为 §13.1 管道分列口径——决策语义均不变）。

## 0. rev3 回退证据与重设计总纲（回退说明，门禁必需章节）

- **回退判定**：rev2（Developer AUTO 栈）在 Stage 3 iter-2 候选判定 [DESIGN_ERROR]，Orchestrator 裁决成立。完整证据链：`perf_tuning/sparse_mla_bwd/optimization_log.md`、`perf_tuning/sparse_mla_bwd/baseline_iter2.json`、`perf_tuning/sparse_mla_bwd/iter1_diagnostics.md`。
- **为何 rev2 不可修补（逐条证据，rev3 设计必须逐一回应，见 §4.1）**：
  1. **锁步等待 73%**：AIC 等 flag 20.4ms（其中 19.8ms 等 V 侧 kv stash 生产）、AIV 10.7ms/38%——AUTO_CV_SYNC 按序配对 + flag 对数 ≤16 + C/V 语句数相等约束把 C/V 锁死在同频往返（fwd 会话 iter-4 RC1-RC5 同源结论）。
  2. **读量带宽下限超目标**：rev2 两遍 transit 数据流 AIV 读量 ~147MB（ws_dkv 读回 60 + KV 30 + ws_dq 30 + s/dp 13 + pst 13 + idx ~1）@ 实测 12.2GB/s 已饱和 → 零开销下限 ~12ms > 7680us。**读量必须架构级削减，非调度级**。
  3. **AUTO 栈 6 条绕行全部证伪**（复现签名见 §3.7 表）：[64,576]/[32,576]/[64,512] staging 死于 UB 规划器 "available: 0"（线性扫描规划器不跨 symbolic 任务循环回收槽位）；C 侧按段拆分死于 L0C 累加器复用；按行半拆分死于 vid 数据就绪不对称（破坏顺序配对）；iter-1 3:3 设想追溯证伪。
  4. **平台约束（已实证，rev3 必须内建）**：跨核 flag 对数 ≤16；单条无内循环大 2D 写可触发编译器 SEGV（640 宽 stash NI=4 两次复现）；3D-of-4D copy 静默截断（双后端 smoke 实证，dkv sim 0.999996/frac 36% 签名）；gemm_v0 双操作数同 dtype 约束；L0C 128KB/核。
- **rev3 关键调整（相对 rev2，防重蹈覆辙声明；2026-10-15 §2 方向修正版）**：循环结构两遍立即消费 → **pair-loop 生产/消费错位**（Developer 形态保留——AUTO_CV_SYNC/AUTO_CV_COMPILE 对 pair-loop 结构编译出 C 领先软件流水，indexer_bwd 零 Expert API 70.9us 实证；Expert 手动 flag 降为 fallback，升级判据与边界 §4.2）；dKV 散射按 §6 双后端寻址查证选型（V 侧逐行保留 + bf16 贡献槽档，C 侧 L0C→GM 写仅用于贡献槽整 tile 写——可连续化腿）；两遍 transit → **融合单遍数据流**（§4.5 逐腿读量核算表）。为何不会再犯同一错误：锁步死因经源码级重查证定性为**两遍立即消费的循环结构**（produce 与 consume 同块立即配对 → C/V 同频往返）而非 AUTO 栈本身——indexer_bwd 在同一 AUTO 栈上以 pair-loop 错位结构取得零锁步深度流水，证明证伪清单（§3.7，全部在 rev2 结构内尝试）未覆盖"循环结构换轨"这一轴；读量死因以腿融合 + bf16 档削减——两者均不在 rev2 已证伪的 6 条绕行空间内（§3.7 逐一对照）。
- **rev2 已验证机制的继承处置**（rev3 组件化，不推翻）：ws_kv 按块暂存、gather 全量+微体 staging、P/dP_raw V-only GM stash（零 flag transit）、k3 行宽 32、mask/bias 向量化——逐项处置表见 §4.8；scatter 载荷瘦身在 scatter 移交 C 侧后**退役并记录**。

## 1. 算子语义与接口契约

> 本章继承 rev2 §1 全文（语义与接口在 rev3 中零变化——回退只改架构不改契约），仅按 rev3 口径压缩转写；amend-1 Δ 恒等式与 amend-2 门限为不可推翻锚点。

### 1.1 算子名称

sparse_mla_bwd（DSA 稀疏 MLA 反传，varlen packed 布局，因果 causal-only）

### 1.2 功能描述

对 varlen packed 序列的每个 packed 行 `s_row`（属于 batch `b`、段内位置 `s_i ≡ s_row − bos`）、每个 kv_group `g`：用前向保存的 topk 索引 `Indices[s_row, g, j]` 与 `Lse` **重计算**注意力概率 P（免二次 softmax），结合上游梯度 `dO` 与前向输出 `O` 计算 `q` / `kv` 的梯度。计算分段（主仓 host 串行编排；rev3 口径下 delta=None 为两段发射，见 1.3 amend-1）：

- **preprocess（rev3 已折叠入 bwd 主 kernel 的 Δ 归约路径）**：`Delta[t,h] = Σ_d O[t,h,d]·dO[t,h,d]`（fp32）；delta=None 时由恒等式替换为 kernel 内 Δ=Σ_j P·dP_raw 归约（1.3），k1 kernel 保留为对照诊断 fallback 不删（§11.1）
- **bwd**：逐 token、逐 topk 索引块重计算 P/dP，7 类 GEMM 产出 dq 逐块累加、dkv 逐块 fp32 原子散射（rev3：散射改 C 侧 L0C 直写，§4.4——语义仍为 fp32 累加链 + 单次舍入，锚点不变）
- **postprocess**：`dKV` fp32 → bf16 单次舍入

### 1.3 数学公式

前向语义（引用，由 `sparse_mla_fwd` 提供，本算子不实现）：

$$\text{score}[t,h,j] = \big(q[t,h,:512]\cdot kv[idx_j,:512] + q[t,h,512{:}576]\cdot kv[idx_j,512{:}576]\big)\times sm\_scale$$

$$P = \text{softmax}(\text{causal+topk mask 后的 score}),\quad o = P\,@\,v,\ \ v = kv[idx_j,:512],\quad lse = \text{logsumexp}$$

反传（`sm_scale` 缺省 `= 576^{-0.5}`；`mask[j] = 1\{idx_j \le s_i\} \wedge 1\{idx_j \ne -1\}`，因果判据用**段内位置 s_i**）：

$$\Delta[t,h] = \sum_{d<512} O[t,h,d]\cdot dO[t,h,d] \qquad (\text{preprocess, fp32；delta=None 时的实际计算路径见本节末 amend-1 恒等式替换})$$

$$P[t,h,j] = \exp\big(\text{score}_{raw}[t,h,j]\times sm\_scale - lse[t,h]\big)\ \ (\text{masked}\to 0,\ \text{bwd 重计算})$$

$$dP[t,h,j] = P[t,h,j]\cdot\Big(\sum_{d<512} dO[t,h,d]\cdot v[idx_j,d] - \Delta[t,h]\Big)\times sm\_scale$$

$$dq[t,h,:512] \mathrel{+}= \sum_j dP[t,h,j]\cdot v[idx_j,:512],\qquad dq[t,h,512{:}576] \mathrel{+}= \sum_j dP[t,h,j]\cdot kv[idx_j,512{:}576]$$

$$dKV[idx_j,:512] \mathrel{+}= dP[t,:,j]^{T}\! \cdot q[t,:,:512] + P[t,:,j]^{T}\!\cdot dO[t,:,:512],\qquad dKV[idx_j,512{:}576] \mathrel{+}= dP[t,:,j]^{T}\cdot q[t,:,512{:}576]$$

（dKV 两式均为 fp32 累加链，跨 token 聚合；**rev3 §6 查证修正后的精度路径：块内 pair 求和在 L0C fp32 在片完成 → bf16 贡献槽单次舍入（§6.3 精度门控，fp32 兜底档保留）→ V 侧 fp32 GM 原子跨 token 累积 → k3 单次 cast**；`kv[...,:512]` 兼作 v 与 dq 的 D 段操作数，`kv[...,512:576]` 为 score/dq 的 tail 段操作数——与主仓源码逐行对应，见 §2.2）

$$dkv_{out} = \text{bf16}\big(dKV_{fp32}\big)\ \ (\text{postprocess, 单次舍入})$$

**Δ 计算恒等式替换（amend-1，2026-10-14；rev3 原样继承为锚点，依据 Stage 2 attempt1 replica 阶梯证据）**：当 `delta is None` 时，kernel 内部以数学恒等的变形

$$\Delta[t,h] = \sum_{j<\text{topk}} P[t,h,j]\cdot dP_{raw}[t,h,j],\qquad dP_{raw}[t,h,j]=\sum_{d<512} dO[t,h,d]\cdot v[idx_j,d]$$

替代主仓的 Δ = Σ_d o·do 路径（P 为 kernel 内 **fp32 重算值**；dP_raw 即 dP 公式括号内的 dO·v^T 项；P/dP_raw 均取**先于 bf16 cast / hi-lo 拆分的 fp32 中间量**参与归约——精度收益必要条件）。要点（完整论证见 rev2 §1.3，此处存档口径）：

1. **数学等价性**：由前向 o = P@v 交换求和序得 Σ_j P·dP_raw ≡ Σ_d o·do（精确恒等，非近似）；恒等前提 o 为同源 fwd 输出，接口契约下恒成立；**用户显式传入 delta 时直接使用传入值，行为与主仓完全一致**（不覆盖、不重算）。
2. **数值动机**：主仓路径含 bf16-o 双重舍入，是 sm_scale=0.1 用例 dq 0.9713 / dkv 0.9631 FAIL 的主导误差；恒等式路径实测 **ratio → 1.0000**（replicaP 证据）。
3. **偏差披露**：delta=None 时与主仓 kernel 逐位颗粒不再一致（主仓消费 GM bf16-o，本实现消费 kernel 内 fp32 P/dP_raw）；主仓自身在该区域对其 golden 即有同量级误差且从未测试覆盖。o 参数在 delta=None 时不消费（契约内行为等价；契约外不一致 o 输入的处置记 §14 R12，rev3 沿用编号）。
4. **结构含义（rev3 强化）**：dP 消费前 Δ 必须完成该 token 全部 NI 块累加 ⇒ **两遍结构在 rev2 中的"遍 1 仅 score/exp/dP_raw/Δ 部分和、遍 2 完整链"两遍数据流正是 147MB 读量的来源**——rev3 的 transit 融合（§4.5）在 pair-loop 深度流水内以"pair p 生产 / pair p−1 消费"的软件流水方式实现同等两遍语义，**不再需要两遍独立数据流**（这是读量削减的语义可行性根据）。
5. **host 编排（amend-1 口径，rev3 不变）**：delta=None → **k2（含 Δ 归约）→ k3 两次发射**；delta 显式传入路径同样不发射 k1。do_bench 计时口径两段，7680us 目标不变（§13.1 重估继承）。

### 1.4 接口契约（与主仓一字不差，rev3 零变化）

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
| indices | [S, HKV, topk] | int32 | topk 索引；值域含 -1 与越界填充（如 S）——OOB 防护见 §4.7/§7；contiguous |
| lse | [S, H] | fp32 | 前向 logsumexp；contiguous |
| offsets | [B+1] | int32 | varlen 段边界（packed 布局） |
| sm_scale | 标量 | — | 缺省 `(D+D_tail)^{-0.5} = 576^{-0.5}` |
| is_casual | — | — | 恒 True（kernel 内 assert，非因果不支持） |
| delta | [S, H] | fp32 | 预计算 Delta 复用入口；None 则内部走恒等式归约（amend-1） |
| **返回** dq | [S, H, 576] | bf16 | |
| **返回** dkv | [S, HKV, 576] | bf16 | |

继承的断言（host / kernel 入口，原样保留）：`is_causal == True`；`topk % block_size == 0`（block_size 为内部索引块 BS）；`D = 512` 为 host 硬编码、`D_tail = 576 − D`；dtype 组合固定（q/kv/o/do bf16、indices int32、lse/delta fp32）；各输入 contiguity。

### 1.5 golden 判据与主仓测试口径（amend-2 收紧后口径，rev3 原样继承）

- **golden**：PyTorch autograd——`ref_sparse_mla_bwd_interface`（ref_gpu/sparse_mla_bwd.py:281-290）：q/kv clone + requires_grad，`o = ref_sparse_mla_fwd_interface(q, kv, indices, offsets)`（内部 fp32、输出 bf16；越界索引 clamp 到 sk 后 scatter 到第 sk+1 列再裁剪 → 贡献为 0；首行自注意强制有效），`o.backward(do)`，返回 `(q.grad, kv.grad)`。q.grad 覆盖全 576 维（score 双段），kv.grad 覆盖全 576 维（v 段经 o=P@v、tail 段经 score）。
- **判据**：`assert_tensors_similar(eps=1e-4)`（ref_gpu/utils.py:48-73）——全局归一化点积相似度 `sim = 2⟨x,y⟩ / (‖x‖²+‖y‖²)`（double 精度），要求 `0 ≤ 1 − sim ≤ 1e-4`；dq / dkv 分别判；全零张量特判通过。amend-2 分层门限（域 A/B/C 划分与数值）在 §12.2/§12.3 细化，rev3 实现必须直接通过现有 `test_sparse_mla_bwd.py` 全部用例（测试资产零重写，任务书 §3 硬性要求）。
- **主仓测试口径**（test_sparse_mla_bwd:293-343，移植保持）：B=1, S=2048, H=64, HKV=1, DQKV=576, DV=512, topk=512, bf16；`randn` 输入与逐 token `randperm(max(1,t))[:topk]` 索引**无固定 seed**，indices 尾部填充 S；`offsets=[0,S]` 单段且 `seq_len ≥ topk` 断言；经 `sparse_mla_fwd_interface` 取 `(o, lse)`（fwd 依赖策略 §11）。
- **性能口径**：`do_bench(fn, rep=100, warmup=250)`，`fn` 调 `sparse_mla_bwd(..., delta=None)` → amend-1 两段发射计时（k2+k3）；目标 < 7680 us（§13 分解）。

## 2. 主仓源码结构分析

> 本章继承 rev2 §2 全文（主仓源码与 rev3 无关，分析结论零变化），按 rev3 压缩转写；行号锚点不变。

### 2.1 三段结构与 host 编排（ref_gpu/sparse_mla_bwd.py）

| 段 | 位置 | grid / 形态 | 计算内容 |
|----|------|------------|---------|
| preprocess | L10-45 | `T.Kernel(H, ceildiv(S,32))`，block_ND=32，`T.Pipelined(D/32, num_stages=5)` | O/dO [32,32] fragment 逐块累乘 + `T.reduce_sum` → Delta[S,H] fp32；纯带宽（读 268MB、写 0.5MB） |
| bwd | L77-251 | `T.Kernel(S, kv_group)` **逐 token grid**，threads=128，num_stages=0；pass_configs 关 TMA / WARP_SPECIALIZED | 核心计算，见 §2.2 |
| postprocess | L48-74 | `T.Kernel(ceildiv(S_kv,64), kv_group)`，threads=128 | dKV fp32 → bf16 纯拷贝 cast（单次舍入） |

host `sparse_mla_bwd`（L254-278）：contiguity 断言 → `prepare_token_indices(offsets)`（CPU 构造 + tensor_cache 单条缓存，ref_gpu/index.py:80-82）→ `delta is None` 则 preprocess → bwd → postprocess 三段串行（rev3 host 侧改两段发射，amend-1 口径，§11.3）。`ref_sparse_mla_bwd_interface`（L281-290）为 golden（§1.5）。

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

形状特征：M=64(H) 固定、**N=32(BS) 极小**（GEMM 效率核心难点）、K∈{512,64,576(可合并)}；GEMM #4/#5 的 K=32、#6/#6'/#7 的 M=32 同为小维。FLOP：主仓 `per_token_flop = 2·H·topk·(2·DV+3·DQKV)` ≈ 180.4 MFLOP/token，×S=2048 → **~370 GFLOP**。

### 2.3 OOB 索引事实（Orchestrator 已核实，设计必须覆盖；rev3 散射路径变更后机制重述于 §7）

- **测试生成**：indices 初始全填 S(=2048)，每 token 前 min(t, topk) 个槽写 `randperm(max(1,t))` 值 → t<topk 时尾部槽保持 S；接口契约另允许 **-1**（mask 判据含 `!= -1`）。
- **gather 无防护**（L194/199）：`KV[bos + Indices[...]]` 直接寻址，idx=S 越界（合法行 `[bos, bos+S_kv)`）；GPU 加载垃圾值，被掩码项贡献恰为 0（acc_p 预填 −inf → exp=0 → P=0、dP=0×垃圾）**侥幸通过**——若垃圾值含 NaN 则 0×NaN=NaN 会失败。
- **scatter 无防护**（L232-241）：`T.atomic_add(dKV[bos + Indices[...]], ...)` 无条件散射，idx=S 为 **OOB 写**。
- NPU 实现必须双端防护且数学结果与 GPU 语义一致（决策 §4.7，机制 §7；rev3 scatter 移交 C 侧后防护语义不变、执行位置变更）。

### 2.4 动态维度与 host 分派

- `T.dynamic("S, B_plus_one")`（L115）：S（packed 总行数）与 B+1 运行时确定；其余 H/D/D_tail/topk/kv_group 为编译期常量参数。
- TokenIndices [S,2] 由 host `prepare_token_indices` 预计算（position_ids / sequence_ids，一次 H2D）；kernel 内 GM 标量读 `b_i, s_i = TokenIndices[...]`（L170），`bos, eos = Offsets[b_i], Offsets[b_i+1]`。
- varlen 语义：行寻址恒等式 `s_row = bos + s_i`；causal 判据用**段内位置** `max_kv_i = s_i`（L173），非 packed 全局行号。
- host 硬编码：`D = 512`（L264）、`D_tail = 576 − D`；`kv_group = kv.shape[1]`；`topk = indices.shape[-1]`。

### 2.5 GPU 形态（不迁移）与语义锚点（必须迁移）

**GPU 形态**：逐 token grid（2048 block × threads=128）、`GemmWarpPolicy.FullCol`、num_stages=0（Pipelined 无实际流水）、split_store=2 + `T.view(KV_shared)` 复用（warp 布局产物）、per-element 标量 atomic、`alloc_shared/alloc_fragment`（GPU 语义）。**Ascend 侧架构自由度完全开放**（rev3 取 Developer 形态 pair-loop 主路径 + Expert fallback，§4）。

**语义锚点（rev3 全部保留）**：§1.3 全部公式与 mask 语义（−inf 预填 + exp）；fp32 dKV 中间量 + postprocess 单次舍入；逐 token 的 Q/dO 单次装载、dq 跨 16 块 fp32 累加；重复索引不去重（重复计贡献，与 fwd DESIGN §1.4 第 1 条一致）；OOB 行为等价性（§7 论证）；amend-1 Δ 恒等式与两段发射；amend-2 分层精度门限。

## 3. 技术约束检测结论（rev3 更新）

> 本章在 rev2 §3 基础上更新：3.1-3.5 平台事实不变（压缩继承）；**3.6/3.7 为 rev3 新增/重写**——AUTO 栈证伪清单与混合模式约束是 rev3 设计的输入边界，§4 全部架构决策不得越界。

### 3.1 三维 Kernel

主仓 `T.Kernel(S, kv_group)` 为 2D grid + threads（GPU 形态）。Ascend 侧 **persistent 1D 编译期常量 grid**（`T.Kernel(grid_size=20, is_npu=True)`），任务维折叠进核内 striped 轮询循环。先例：`sparse_mla_topk_reducesum.py` L218/261（persistent 双变体，graph-replay 143.3us 实测）、`indexer_bwd.py`（persistent + pair-loop，rev3 最直接先例）。**✓ 通过**。

### 3.2 threads 映射

主仓 `threads=128` 为 GPU warp 布局参数，Ascend 无对应语义。改用 `(cid, vid)` 启动形态 → codegen 默认 **MIX_AIC_1_2**（1 Cube + 2 Vector / block；`indexer_bwd.py` L124-131 注释已验证：AIV 见 `cid = GetBlockIdx()/2`、`vid = GetSubBlockIdx()∈{0,1}`；**1_2 flag 握手要求两个 AIV 子块都 set 每个 AIC 等待的 flag**——单 AIV 缺席 transit 会确定性死锁；主路径由 AUTO 生成序保证（indexer_bwd MIX_1_2 运行实证），Expert fallback 手动设计必须遵守（§9.7））。grid=20 → **20 AIC + 40 AIV 全部用满**。AIV-only kernel（k3 postprocess）用 `threads=1` 的 `(cid)` 形态或 (cid,vid) 形态（§11 定）。**✓ 通过**。

### 3.3 动态循环边界

`S / B_plus_one` 为运行时动态维。处理：外层 persistent 轮询 `ntask = T.max(0, T.ceildiv(total − cid, grid))` + `T.serial(ntask)` **运行时边界**（reducesum L265 已验证形态）；内层 NI、BS、D 段循环全为编译期常量。约束：动态边界禁 `T.Pipelined` → rev2 用 `T.serial(NS)` + AUTO_CV_SYNC；**rev3 主路径仍 T.serial + AUTO_CV_SYNC**（软件流水以 pair-loop 手写错位结构表达，AUTO 栈对该结构编译出 C 领先流水——indexer_bwd L512 `for p in T.serial(PAIRS+1)` 同款实证，§4.3/§9）；symbolic 任务循环引发的 UB 规划器不回收槽位证伪事实（3.6-6）**由数据流重构规避**——rev3 新数据流不再需要证伪清单 #1-3 的 [64,576]/[32,576]/[64,512] 级跨段大 staging（ws_dkv 改逐行微读、dq 累积驻持续槽，§4.5/§5），UB 预算按"持续驻留、不跨循环回收"口径重排（§8）；Expert fallback 路径则以显式 alloc_ub 手动放置绕开规划器（§8.7）。varlen 段内 token 边界不进循环结构（mask 链吸收，§10）。**✓ 通过（机制不变：AUTO + pair-loop 结构重构；Expert 手动为 fallback）**。

### 3.4 L0C 容量（查证：**128KB/核**，任务书预估 256KB 不成立）

锚点：`Ascend910B3.ini` `l0_c_size=131072`（= 128KB；另 l0_a/l0_b 各 64KB、l1 512KB、ub 192KB）。rev2 核算结论不变：acc_dq [64,576] 144KB 等全驻方案超限 → **dq 累积移出 L0C 驻 UB fp32**、dkv 每块产出后即时散射、acc_p/acc_dp 分时复用。**rev3 增量结论**：dKV 散射改 C 侧 L0C 直写后，L0C 内 dkv 累加器（G5 pair [BN,256]×DSEG 段）在片累积跨块复用（indexer_bwd note 11 同段连续块累积形态），L0C 预算表与 GEMM 编排在 §5/§8 重核——**128KB 上限仍是硬边界**。**✓ 分层处置**。

### 3.5 GEMM 非整除（分形 16 对齐逐项过检，继承不变）

| 维 | 值 | %16 | 说明 |
|----|----|-----|------|
| M | 64（padded_H） | ✓ | H_kv=64；`padded_H = max(next_pow2(H_kv),16)` 兜底 |
| N | BN=128（rev3 主档，索引块合并） | ✓ | `topk % BS == 0` 断言继承（512%32=0；BN 为 BS 的合并倍数） |
| K | 512 / 64 / 576 | ✓ | 576 = 36×16（QK 双段可合并单 GEMM） |
| D_tail | 64 | ✓ | |

varlen 段长任意**不产生块尾非整除**：gather 行数恒为整块，越界/无效行由 mask 归零（§7）；段尾 token 数任意由任务=token 粒度天然吸收。残余风险：kv_group>1 通用路径的 head padding（考核 shape kv_group=1 无 pad，§5/§14 记录）。**✓ 通过**。

### 3.6 平台强制偏差与混合模式约束（rev3 重写：Expert 路径新增约束逐条查证）

**平台强制偏差（rev2 已实证，rev3 全部内建）**：

1. **bf16 UB→L1 直连被 static_assert 禁止**（half-only）→ C↔V 中转必须走 GM workspace。
2. **L0C→UB 可能被编译器改写为 GM 往返**（ascend_workspace_reduction.cc）→ 中转流量按 GM 口径显式入性能预算（§13）。
3. **auto workspace 不支持动态 grid** → 全部 workspace 显式参数化；rev2 靠 buffer 命名含 `"workspace"` 触发 `TL_ASCEND_AUTO_CV_SYNC` 自动握手，**rev3 主路径全量沿用该机制**（indexer_bwd L138-141 先例：跨核 transit 槽全部以 `workspace*` 命名，combinecv FetchWorkspaceName 按名配对自动插 flag；**V-only 单侧 stash 以非 workspace 命名避开配对实现零 flag transit——iter-2 已验证机制**）；Expert fallback 升级时才改手动 `T.set_flag/T.wait_flag`（§9.7），pto 侧可用性查证同节落实。
4. **`T.tile.atomic_add` 约束**（tilelang/language/ascend_tile.py:313-330 已查证）：dst 必须 GM（scope global）、src 须 local（UB/L0C/L1）、dtype 与 dst 一致、支持 fp32（float16/float32/int16/int32/bfloat16）——**rev3 实际使用的是其 UB src 逐行形态（V 侧散射，§6.2——L0C direct src 形态经 §6 查证仅适用连续 region，随机行不可用）；indexer_bwd dk 链两形态（layout=0 region 原子 / layout=1 UB 逐行）均为已验证先例**。

**rev3 Stage 3 实测新增平台约束（全部有复现签名，rev3 设计必须绕开而非重试）**：

5. **跨核 flag 对数 ≤16**（硬件限制）：rev2 16 对预算耗尽后 3:3 形态需第 17 对被阻（baseline_iter2.json）；rev3 主路径（AUTO）下 flag 对数 = workspace 槽族配对数——**设计以槽族数控制对数**（rev3 槽族 ≤6 → 对数 ≤12，§8/§9 记账）；Expert fallback 手动 flag 设计**必须逐对记账**（§9.7 预算表）。
6. **UB 线性扫描规划器不跨 symbolic 任务循环回收槽位**：liveness 不相交的 pass-1 槽（kv_stage 73728B）无法被 pass-2 消费者复用，报 `Memory allocation failed ... new memory available: 0`（[32,576] staging 复现）；rev3 主路径**由数据流重构规避**——新数据流不再需要跨段回收的大 staging（ws_dkv 逐行微读、dq 累积驻持续槽，证伪形态 #1-3 消失，§4.5/§5），UB 按"持续驻留、不跨循环回收"口径预算（§8）；Expert fallback 路径则以显式 `T.alloc_ub` 手动放置 + 生命周期由 flag 序独占推导（indexer_bwd note 2 同款论证）绕开规划器（§8.7）。
7. **单条无内循环大 2D 写可触发编译器 SEGV**（640 宽 stash NI=4 两次复现，"Vec loop times (=1) not enough to catch up cube loop times (=4)" 族；NI=1 通过）→ 大平面写出必须带内循环分段或按 pair 粒度拆分（§5/§9）。
8. **3D-of-4D region T.copy 静默截断**（双后端 smoke 实证：dkv sim 0.999996 / frac 36%，lo 通道截断特征签名）→ hi/lo 3D 合并写禁用，平面数 ≤2 的 2D copy 或逐平面写。
9. **gemm_v0 双操作数同 dtype 约束** → bf16 操作数 GEMM 不可 fp32 混搭（ws_p/ws_dp 的 hi/lo 拆分与 cast 档位受此约束，§5/§12）。

**混合模式（Expert API）约束（rev3 新增，逐 API 查证：programming-model-guide / api-best-practices / indexer_bwd 实现）**：

10. `T.alloc_L1/alloc_ub/alloc_L0A/alloc_L0B/alloc_L0C`（Expert 显式分配）、`T.Scope("C")/T.Scope("V")`（C/V 语句域切换）、`T.set_flag/T.wait_flag`（跨核握手）、`T.barrier_all`（核内同步）——签名与语义以 `tilelang/language/ascend.py` / `ascend_tile.py` 与 api-best-practices 速查表为准，实现阶段引用前逐条复查（任务书硬性约束）；indexer_bwd.py 为已验证用法先例。
11. **AIV 侧 set_flag 插在源管道（MTE2/MTE3）指令流，标记被 wait_flag 消费前该管道停转**（fwd 会话 R-AIV-FLAG 规则：(a) 悬空/预置 set = 永久停转；(b) 延迟消费 = 停转窗口；(c) 只有相邻 [set;wait] 全排空对安全高效）——rev3 手动 flag 的发射位置必须遵守（§9）。
12. **MIX_AIC_1_2 的 flag 握手要求两个 AIV 子块都 set 每个 AIC 等待的 flag**（indexer_bwd note 10，缺一确定性死锁）；两 AIV 的 transit 分工必须保证每个跨核 flag 都有双方 AIV 的贡献（§9 分工表）。

### 3.7 AUTO 栈绕行证伪清单（rev2 Stage 3 iter-0~2 全记录；rev3 任何设计不得复用这些路径）

| # | 绕行路径 | 证伪签名 | 证据 |
|---|---------|---------|------|
| 1 | [64,576] fp32 ws_dkv 批量读回 staging | UB 规划器 `available: 0`（147456B > 可用 196352B 内的剩余） | baseline_iter2.json `ws_dkv_batched_readback_full_576` |
| 2 | [32,576] fp32 3:3 staging（liveness 不相交槽） | 双重阻塞：规划器不跨 symbolic 任务循环回收 kv_stage 槽（同 "available: 0"）+ 3 点 V 侧形态需第 17 对 flag | baseline_iter2.json `ws_dkv_batched_readback_3_3` |
| 3 | seg-only [64,512] staging | 同 #1 规划器 | baseline_iter2.json `attempted bypasses` |
| 4 | C 侧按段拆分（dkv 段间分担） | L0C 累加器复用冲突（G5 pair acc 跨段不可共存） | baseline_iter2.json `per-segment C-side split` |
| 5 | 按行半拆分（AIV 各半） | vid 数据就绪不对称破坏 AUTO_CV_SYNC 顺序配对 | baseline_iter2.json `row-half C-side split` |
| 6 | iter-1 3:3 设想（C 领先 3 块 / V 滞后 3 块） | 追溯证伪：约束 #5/#6（flag ≤16、语句数相等）下不可表达 | iter1_diagnostics.md + fwd iter-4 RC1-RC5 |

补充证伪（iter-0/1 遗留，平台约束固化）：single-batch [64,576] 整 2D 写编译器 SEGV；ws_p/dpb hi+lo 3D 合并静默截断；fp32 P/dP 直传（gemm_v0 同 dtype）。**结论（2026-10-15 §2 方向修正版）：上述 6 条证伪的共同根因是 rev2 两遍立即消费的循环结构——在此结构内，staging 需求（大槽跨 symbolic 循环）、flag 需求（3 点形态）、配对约束（vid 对称 / 语句数相等）互相锁死，AUTO 栈确无绕行空间，"结构内封闭"成立。但封闭是结构局部的，不是 AUTO 栈全局的：indexer_bwd 以 pair-loop 生产/消费错位结构在同一 AUTO 栈上取得零锁步深度流水（70.9us，零 Expert API，docstring 自述 lockstep 92% → 重构后消除）——证伪清单未覆盖"循环结构换轨"这一轴。rev3 主路径沿该轴重构（§4.2/§4.3）；Expert 手动 flag 保留为 pair-loop 形态仍锁步时的 fallback（升级判据 §4.2），其约束（显式放置 / flag 逐对记账）见 §8.7/§9.7。**

### 3.8 检测结论汇总

| 约束项 | 结论 | rev3 设计响应 |
|--------|------|---------|
| 三维 Kernel | ✓ | persistent 1D grid=20（§4.2） |
| threads 映射 | ✓ | (cid,vid) → MIX_AIC_1_2，20 AIC + 40 AIV；双 AIV flag 贡献由 AUTO 生成序保证（indexer_bwd MIX_1_2 运行实证），fallback 手动遵守（§9.7） |
| 动态边界 | ✓（结构重构） | T.serial(ntask) 运行时外层 + 编译期常量内层；pair-loop 手写错位结构替代 T.Pipelined（AUTO 编译软件流水，§9） |
| L0C 容量 | ✓ 分层处置 | dq 累积驻 V 侧 UB fp32（半片 [32,576]）；dkv 块内 pair L0C 在片累积 + bf16 贡献槽（§5/§6/§8） |
| GEMM 非整除 | ✓ | 全维 16 对齐；varlen 由 mask 吸收（§5/§7） |
| 平台偏差 1-4 | ✓ 已内建 | GM 中转 / 显式 ws / fp32 atomic（UB src 逐行形态，§6） |
| flag ≤16 | ✓ 槽族控制 | 主路径（AUTO）workspace 槽族 ≤6 → 对数 ≤12；fallback 逐对记账（§9.7） |
| UB 规划器不回收 | ✓ 数据流规避 | 新数据流消除跨段大 staging（证伪形态 #1-3 消失）+ UB 持续驻留预算（§8）；fallback 显式 alloc_ub（§8.7） |
| 大 2D 写 SEGV / 3D 截断 / 同 dtype | ✓ 绕开 | pair 粒度分段写 / 平面 ≤2 / bf16 单档（§5/§9/§12） |
| AIV flag 管道停转 / MIX_1_2 双 AIV set | ✓ 内建 | AUTO 生成序遵守相邻 [set;wait] 排空对（indexer_bwd 实证）；fallback 手动设计遵守（§9.7） |

## 4. 总体架构决策（rev3 修正版：Developer pair-loop 主路径 + Expert fallback + C/V 深度流水 + transit 融合 + 散射 §6 查证选型）

> 本节只写决策与理由；机制细节（GEMM 编排 / L0C 预算 / flag 时序 / UB 放置）分别在 §5-§9（块 B/C）。**本节是 rev3 相对 rev2 的架构变更核心**，每个决策均标注回应的回退证据条目（文首 §0 之 ①②③④）与依据的先例。§4.4/§4.5 中 dKV 散射的 C 侧直写表述为块 A 乐观预置，**已由块 B §6 双后端源码级查证修正**（随机行散射 C 侧直写不可行）——修正以"§6 查证修正"标注就地保留，以完整记录决策演化。

### 4.1 回退证据的架构级回应总览（决策动机链）

| 回退证据 | 根因定性 | rev3 架构回应 | 落点 |
|---------|---------|--------------|------|
| ① 锁步等待 73%（AIC 等 flag 20.4ms / AIV 10.7ms） | 两遍立即消费的循环结构把 C/V 锁死同频往返——**循环结构性质**（源码级重查证：非 AUTO 栈本身，§3.7 结论） | pair-loop 结构重构的深度流水（Developer 形态，AUTO 栈对错位结构编译出 C 领先流水；Expert 手动 flag 预置为 fallback，升级判据 §4.2） | §4.3、§9 |
| ② 读量 ~147MB @ 12.2GB/s 饱和（下限 ~12ms > 7680us） | 两遍 transit 数据流的 V 侧读回体量——**数据流性质** | transit 腿融合压缩（目标 AIV 总读量 ≤~75MB）：ws_dkv 贡献槽 bf16 档（~53→~26MB，§6.3 精度门控）+ L2/L3 合槽 + L1 批量化 | §4.5、§6 |
| ③ rev2 结构内 6 条绕行证伪 | UB 规划器不回收 / L0C 复用 / vid 不对称 / 3:3 不可表达——**rev2 结构内的工具链能力边界** | 循环结构换轨（两遍立即消费 → pair-loop 错位，证伪清单未覆盖的轴，indexer_bwd 实证）+ 数据流重构消除大 staging 需求（证伪形态 #1-3 消失）；Expert 手动放置保留为 fallback（§8.7） | §3.6/3.7、§8、§9 |
| ④ 平台约束（flag≤16 / SEGV / 截断 / 同 dtype / L0C 128KB） | 硬边界 | 全部内建为设计规则：槽族数控 flag 对数（≤12）、pair 粒度分段写、平面 ≤2、bf16 单档、L0C 分层 | §3.6、§5、§8、§9 |

关键判断：证据 ①③ 经源码级重查证定性为 rev2 **循环结构**的结构性死局（结构内封闭，§3.7 结论），证据 ② 是"无论调度多优、体量本身超限"的物理性约束（读量必须削减）——**rev3 两者必须同时做**（tuner 预估 Expert/混合形态 ~19-21ms 仍 > 7680us，正是读量未削减时的模式上界——模式升级单独不解决问题，佐证"结构 + 数据流"双轴响应的必要性）。

### 4.2 rev3 编程模式决策（2026-10-15 §2 方向修正）：Developer 形态 pair-loop 主路径 + Expert fallback

**决策**：k2 bwd 主 kernel 走 **Developer 形态**——零 Expert API（`T.alloc_shared`（编译器按语句域映射 L1/UB）+ `T.alloc_fragment` 逻辑累加器（编译器管 L0C）+ workspace 命名自动 CV 握手 + 4-True pass_configs），主循环重构为 indexer_bwd 同款 **pair-loop 生产/消费错位结构**（§4.3）；k3 postprocess 维持 Developer 形态（AIV-only 纯 cast，无跨核 transit）；k1 已被 amend-1 折叠入 k2 的 Δ 归约（不再发射）。**Expert 手动 flag 仅为 fallback**：pair-loop 形态在 AUTO 编译下复测仍锁步时按判据升级。

**修正依据（为何从块 A 预置的"混合主路径"改为"Developer pair-loop 主路径"；indexer_bwd 源码逐条查证，任务书 §2 修正指令）**：

1. **indexer_bwd 是零 Expert API 的 Developer 形态**（docstring L1 自述 "Developer mode"；全文件无 `T.alloc_L1/alloc_ub/L0A/L0B/L0C`、无 `T.Scope`、无 `T.set_flag/T.wait_flag`——本节块 B 已全文核对），其 70.9us 深度流水（重构前 lockstep 92% → 重构后消除）完全由源码**结构**驱动：L512 `for p in T.serial(PAIRS + 1)` + `if p < PAIRS: C produce(p) / V consume(p)` + `if p >= 1: C consume(p-1)` 的 if-guard 错位形态——**AUTO_CV_SYNC/AUTO_CV_COMPILE 对该结构编译出软件流水**，其 docstring L44-60 描述的 flag 序是**编译器生成行为**而非手写。
2. **配套机制同为 Developer 形态**：gemm_v0 逻辑累加器（`T.alloc_fragment` L374-377：logits_l0c/dq_l0c/dk_l0c——编译器管 L0C 物理槽）；buffer 命名含 "workspace" 触发自动 CV 握手（L138-141 + src/transform/ascend_combinecv.cc FetchWorkspaceName）；dk 链 L0C→GM 直写原子 `T.tile.atomic_add`（§6）。
3. **rev2 死因重定性**：rev2 的两遍立即消费循环（produce 块 p 完成后同迭代立即 consume 块 p）使 AUTO_CV_SYNC 的按序配对退化为同频往返锁死；**死因是循环结构，不是 AUTO 栈**——证伪清单 6 条（§3.7）全部在 rev2 结构内尝试，未覆盖"循环结构换轨"轴。

**Expert fallback（升级判据、边界与预置）**：

- **升级判据（量化，唯一触发条件）**：pair-loop 形态实现完成后复测 msprof，**AIC 等 flag 时间占比 > 30%**（rev2 锁步签名同口径对照：73% 为病态，健康流水预期 <15%）。
- **升级边界（混合模式形态）**：仅 k2 内 C↔V 跨核 transit 段改 Expert——显式 `T.alloc_ub/alloc_L0*` 手动放置 + 手动 `T.set_flag/T.wait_flag`（≤16 逐对记账）+ pass_configs 关闭 `TL_ASCEND_AUTO_CV_SYNC`/`TL_ASCEND_AUTO_CV_COMPILE`（其余 pass 维持；完整配置表与 flag 预算表 §9.7）；k3、V-only 单侧 stash 段（iter-2 零 flag 机制）维持 Developer 形态。
- **预置工作**：§8.7 记 fallback 手动放置预案；§9.7 记 flag 预算与 pass_configs 关闭项；升级不改变接口签名与数据流（仅同步机制替换），Stage 2 结构按"两条路径可切换"组织 transit 段边界。

**合规性**：用户原 Developer 选项 + "Stage 3 视瓶颈引入专家级手段"预留 + 2026-10-14 全权自主授权——主路径保持 Developer 形态（与用户原指定完全一致），Expert 仅按量化判据升级（授权链内，且判据写入设计可审计）。

**双后端影响**：主路径（Developer + AUTO 栈）双后端可用性由 rev2 全量验证背书（ascendc + pto 双通过至 28.5ms，同一 API 集）；pto 侧 Expert 形态（手动 flag/Scope/L0 分配）可用性查证仅 fallback 需要（docs/TileLang-Ascend Programming Guide 差异章节，§9.7 落实）——若 pto 不可用，fallback 在 pto 后端降级为主路径形态（正确性同源、性能如实报告，符合双后端约束原文）。

### 4.3 C/V 深度流水：pair-loop 软件流水（indexer_bwd rev3 先例移植）

**结构**：任务循环（token × kv_group，persistent striped 轮询）之上叠 **pair-loop**——每 iteration p：C 侧**生产** pair p 的 score/dP_raw GEMM 结果并写 transit stash，随后**消费** pair p−1 的 P/dP（V 侧已产出的 bf16）完成 dq/dkv GEMM；V 侧 lag 一个 pair 消费 transit stash 完成 mask/exp/dP 链并写回。**C 领先 V 一个 pair，深度流水取代锁步**。

```
pair-loop 时序（DEP=1，双缓冲 p%2）：
  iteration p:  C: produce(p) → consume(p−1)        V: consume(p) → produce(p)
  前置 drain:   C: produce(0)                       —
  收尾 drain:   C: consume(N−1)                     V: consume(N−1)
```

**与 rev2 3:3 设想的区别（为何 3:3 证伪而 pair-loop 可行）**：3:3 需要 C 领先 3 块 → transit stash 需 3 组槽位 + flag 对数翻倍（16 对预算放不下，证据 ③#6/#2）；pair-loop 深度 DEP=1 只需 **2 组槽位（p%2 双缓冲）**，且 AUTO 模式下 flag 对数由 workspace 槽族数决定（rev3 槽族 ≤6 → ≤12 对 ≤16，§8/§9 记账；fallback 手动路径逐对记账 §9.7）。流水深度的收益不来自"领先更多"而来自"生产/消费错位重叠"——AIC 生产 pair p 的 GEMM 与 AIV 处理 pair p−1 的向量链同时进行，消掉同频往返的相互等待（回应证据 ①）。

**槽位安全论证（结构层面，indexer_bwd L44-60 同款推理的 rev3 移植；完整版记 §9.4）**：双缓冲下 C 生产 p+1 前必先读 V 的 dlog(p−1)（⇒ V 已读完 stash(p−2)，槽位 p%2 可覆写）；V 改写 dlog(p−2) 前必先读 C 的 logits(p)（⇒ C 已读完 dlog(p−2)）。全部 WAR/RAW 冲突由 pair 序的读后写传递性**在源码结构层面**独占推导；AUTO_CV_SYNC 按 workspace 配对为每个 transit 槽族生成对应的 set/wait flag 序（indexer_bwd 运行实证该生成序正确承载结构安全），无需额外同步（约束 3.6-11 的相邻 [set;wait] 排空对由生成器保证）。Expert fallback 升级时此论证转为手写 flag 序的显式表达（§9.7）。

**amend-1 两遍语义的 pair-loop 表达**：Δ 归约要求 dP 完整链消费前，该 token 全部 NI 块的 dP_raw 部分和已累加完毕（Δ 定格）。rev3 将其表达为 **token 粒度的两段 pair-loop 嵌套**：外层 token，内层 NI 个 pair——前半程（pass-1 语义）V 侧仅做 score/exp/dP_raw 计算与 Δ 前缀累加（produce-only），NI 块完成后 Δ 定格；后半程（pass-2 语义）以定格 Δ 发起 dP 链与 dq/dkv 消费 GEMM。**语义硬约束：Δ 定格先于任何 dP 消费 GEMM 发射**（§9.5 时序图逐 pair 申明）。与 rev2 的区别：rev2 的两遍是**两遍独立数据流**（KV 重 gather + ws_dkv 读回，147MB 主因）；rev3 两段嵌在**同一遍 pair-loop 流水内**——V 侧 gather 单遍（AIV gather 腿 1×，无第二遍重 gather）；C 侧 G3/G4 消费段自 workspace_kv bulk 重读 kv 列段（**C 引擎口径 2×：L1 512KB 容不下 NI=4 块全驻 576KB，produce/consume 各读一遍——§5.4/§5.5 记账，§13 预算**），**V 侧不产生第二遍 gather 读量**（回应证据 ②的 V 侧读量根据）。

### 4.4 dKV 散射：C 侧 L0C→GM 直写预期与 §6 查证修正（块 A 预置 → 块 B 裁决）

> **【§6 查证修正，2026-10-15 块 B】本节为块 A 乐观预置的决策记录，已被 §6 双后端源码级查证部分推翻**：`T.tile.atomic_add` 的 L0C→GM 直写在双后端均为**单基址连续 region** 形态（ascendc `atomic_add_l0c_to_gm` = SetAtomicAdd + DataCopyPad 连续块；pto `atomic_add_l0c_to_gm_dynamic` 支持运行时基址但传输仍为单连续 region），**随机行散射不可表达**（逐行调用需 L0C 行切片 src——indexer_bwd note 4 实证"beyond the first fractal tile 不可靠"）。§6 裁决：**C 侧 L0C→GM 写仅用于可连续化腿——ws_dq/ws_dkv 贡献槽整 tile 写（per-core 连续槽区、非原子；indexer_bwd workspace_dkf staging L691 同款）；dkv 随机行散射保留 V 侧 UB 逐行（iter-1 优化形态）+ bf16 贡献槽档（读回 ~53→~26MB，§6.3 精度门控）；dq 终写为 V 侧 UB→GM（§5.3 选定 V 侧累积）**。下表保留原预期供决策演化追溯，右列以 ~~删除线~~ + 修正值标注。

**决策**：dkv 的 GEMM 产出（G5 pair / G6 tail）~~不再经"L0C → GM ws_dkv 贡献槽 → AIV 读回 UB → 逐行 T.tile.atomic_add 散射"三跳链，改为 C 侧直写~~ **修正：三跳链保留但贡献槽降 bf16 档**——L0C fp32 在片累积（块内 pair 求和，accumulator 为 alloc_fragment 逻辑累加器）→ 贡献槽写出（~~fp32~~ **bf16，单次舍入**）→ AIV 按 mask 逐行读回 + `T.tile.atomic_add` fp32（UB src，cast 后发射）→ ws_dk GM 终态原子区 → k3 cast。~~indexer_bwd rev2 同款先例~~ **修正先例：indexer_bwd layout=1 L691-697（随机行：C 整片 L0C→ws staging + V 侧逐行 UB atomic）——最强先例对随机行明确选择 V 侧逐行；layout=0 的 C 侧 region 原子仅适用连续行（arange-tiled），本算子 randperm 索引不满足**。

**收益核算（回应证据 ②；修正后口径）**：

| 项 | rev2（V 侧散射三跳链） | rev3（§6 修正后） |
|----|----------------------|-----------------|
| AIV 读回 | ws_dkv 贡献槽读回 **~53MB**（fp32，917,760 有效行 × 2304B ÷ 40 AIV） | **~26.4MB**（bf16 档，§6.3 门控；fp32 兜底档 ~53MB 性能披露） |
| AIV 逐行发射链 | 26k 有效行/AIV × (3×PIPE_ALL + UB scalar GetValue + 1 atomic) 串行发射 | **保留逐行发射**（iter-1 优化形态：idx_ub 源 pos + 段微读合并）——**由 pair-loop 跨 token 错位吸收**：V 散射(t) 与 C 生产(t+1) 重叠（§5.4/§9.5） |
| V 侧 UB 占用 | staging 槽（证伪清单 #1-3 的 "available: 0" 元凶） | 逐行微读 [576] 行缓冲（~2.5KB）——**大 staging 形态消失**（证伪形态 #1-3 不复现的结构根据） |
| C 侧新增 | — | ~~ws_dk 直写原子 ~2.4GB~~ 修正：贡献槽写出 2.31GB→**1.15GB**（bf16 半量，C bulk 出口）；终态原子写 2.12GB fp32 不变（L2 吸收，§6.5） |
| OOB 防护 | V 侧逐行条件跳过 | **V 侧逐行条件跳过原样保留**（§4.7 修正：掩码消费位置不变，无需传 C） |

**关键澄清——"C 侧按段拆分"证伪不适用（修正后仍成立）**：证据 ③#4 的证伪对象是 rev2 语境下"L0C 累加器复用冲突"（AUTO 栈内 dkv 段拆分与 G5 pair acc 抢 L0C）。rev3 的 GEMM 编排中 dkv 累加器为块内 pair 短生命周期逻辑累加器（每块 2 发 ×DSEG 后即 flush 贡献槽），按段（DSEG）拆分是 GEMM 编排的自然段（§5.1），不引入额外 acc 槽——**与证伪签名不同源**。L0C 时分排布在 §5.1/§8 重核，128KB 上限仍是硬边界。

**L2 吸收论证（修正后口径）**：ws_dk [2048,1,576] fp32 = 4.5MB + KV 2.25MB + indices 4MB + 贡献槽 20 核 × 2 token × 4 块 × [128,576] bf16 ≈ 5.9MB 工作集 ~17MB ≪ L2 192MiB → 原子与贡献槽流量由 L2 吸收，HBM 仅见终态写（rev2 §4.5-3 论证框架继承，实测锚点：indexer_bwd region atomic 已验证该量级吞吐；**原子溢价实测仅 2.4%**——iter-1 差分 var_noatomic 1367us/56.4ms，真瓶颈是逐行串行发射而非原子本身，§6.5）。**V 侧读回为何不经 12.2GB/s 瓶颈**：12.2GB/s 是 rev2 微事务读的饱和值；rev3 贡献槽读回为 bf16 行读（2304B→1152B/行，批量行读形态），且 pair-loop 重叠下不占关键路径——逐腿带宽口径统一入 §4.5 核算表与 §13。

### 4.5 transit 融合数据流与逐腿读量核算表（AIV 总读量 ≤~75MB 目标）

**rev2 数据流（两遍，AIV 视角，147MB）→ rev3（融合单遍）逐腿对照**：

| 腿 | rev2 内容 | rev2 读量/AIV | rev3 融合后 | rev3 读量/AIV |
|----|----------|--------------|------------|--------------|
| L1 KV gather | V 侧逐行读 GM KV（1152B 散行） | 30MB @ ~7.6-8GB/s | **AIV 仍执行 gather，但写 stash 后 C 侧整块读**；行读批量化（batch32 整 2D 形态，fwd probe2 8.6GB/s rw / 3.6× 加速锚点） | ~15MB（批量 2D 后有效带宽 ×3.6，时间口径折算读量同幅降；字节量 30MB 不变但**不再占关键路径**） |
| L2 ws_s/dp 读回 + pst 重读 | V 读 C 产出的 score/dP_raw fp32 中转（pass-1）+ pass-2 重读 | 26MB | **Transit 槽 token 奇偶双缓冲持久化 + V-only fp32 pst stash（iter-2 机制）双读结构**：C 产出写 ws_s/ws_dp（fp32，amend-1 锚点——Δ 归约的 P/dP_raw fp32 量在 V 链内先于 cast 取出），V pass-1 消费算 P/Δ 前缀、pass-2 经 pst 重读算 dP 链（workspace 槽单次配对读 + pst 零 flag 二读——iter-2 已验证形态；两 AIV 行分工半片）；bf16 grain 为门控候选（见下行注） | **~12.8MB fp32 基准**（半片口径；bf16 档 ~6.4MB，amend-1 P 重算精度风险——仅域 B 实测门控后可议，§12） |
| L3 pst（P/dP stash） | V 写 P/dP bf16 供 C 消费 + 读回 | 13MB | 并入 L2 同一 stash 槽族（腿融合） | 0（已并入 L2 记账） |
| L4 ws_dq 读回 | V 读 C 产出的 dq 贡献块 fp32 做跨块累加 | 30MB | **dq 累积保留 V 侧 UB fp32 半片 [32,576]（rev2 §3.4 决策继承，§5.3 数据选型重证）**：C 侧 L0C 全驻 [64,576]=144KB>128KB 不可行、dseg-outer C 侧 kv 3× 读>GEMM 下限（§5.3 否决）；贡献档 bf16 7.2MB / fp32 14.4MB 半片口径（§12 精度门控） | **7.2-14.4MB** |
| L5 ws_dkv 读回 | V 读贡献槽做逐行散射 | 60MB | **bf16 贡献槽档**（§6 查证修正：随机行散射 C 侧直写不可行——双后端单调用均为单基址连续 region + L0C 行切片不可靠；V 侧逐行保留 + 槽 dtype fp32→bf16 精度门控） | **~26.4MB**（917,760 有效行×1152B÷40 AIV；fp32 兜底档 ~53MB，§6.3） |
| L6 idx/lse/delta 标量族 | 逐块 GM 标量读 | ~1MB | 继承（微量化，rev2 已优化形态） | ~1MB |
| **合计** | | **~147MB @ 12.2GB/s ≈ 12ms** | | **AIV 关键路径读量（fp32 L4 基准档，vid 行分工半片口径）~62-70MB**（L1 时间口径 15 + L2 12.8 + L4 7.2-14.4 + L5 ~26.4 + L6 1；L2/L4/L5 均按两 AIV 行分工半片折算——fwd §8.6 恒等式；L5 = 917,760 有效行 × 1152B bf16 ÷ 40 AIV）**≤75MB 目标达成**；L4 bf16 门控档再省 ~7MB；L5 fp32 兜底档（+26MB → ~88-96MB）超目标、性能如实披露（§6.3/§13）。诚实修正记录：块 A 预置的 22.5-37.5MB 低估 L2（fp32 双腿）与 L5（§6 修正后恢复），本行为块 B 复核口径；最终 7680us 可行性以 §13（块 D）逐腿带宽-时间口径为准（多数腿为批量 2D 连续读，适用 ~20GB/s 而非 12.2GB/s 微事务口径） |

**腿融合机制**：① L2/L3 合并为单 stash 槽族（score→exp→dP 在同一 UB 槽位就地变换，GM 中转只保留 C↔V 必要的两向——bf16 UB→L1 禁直连的强制 GM 路径，约束 3.6-1）；② L1 的 gather 改"全量+微体 staging"（rev2 已验证）+ 批量 2D 写（约束 3.6-7 的 SEGV 边界内，NI 粒度分段循环）；③ L4 档位决策入 §12 精度门控（bf16 transit 评估：Δ 恒等式后 dkv 域 A ratio 1.0000，plain-bf16 或单 bf16 通道回退是合法候选，须过域 B sim≥0.999 收紧门限）。

**批量 2D MTE 锚点汇总（"12.2GB/s 是微事务饱和值"的实证）**：fwd probe2 batch32（2×[32,576] 整 2D 写）**8.6GB/s rw、3.6× 于 frag 微写**；fwd 实测 V mte2 20.5GB/s / C mte2 bulk 15-19.3GB/s / C fixpipe 49.9GB/s；rev2 bwd iter-2 AIV 微事务 12.2GB/s——**同一硬件，事务形态决定有效带宽 3-5×差距**，rev3 全部 transit 腿按批量 2D 设计（每腿事务形态与预期带宽入 §13.1 逐腿表）。

### 4.6 双后端兼容策略（ascendc + PTO，继承 rev2 框架 + rev3 查证项）

**单一 kernel 源码，API 限定已验证交集**：主路径（Developer 形态）所需 API 集——persistent `T.Kernel` / (cid,vid) 启动 / `T.gemm_v0` 逻辑累加器 / `T.alloc_shared`/`alloc_fragment` / `T.tile.*` 向量链 / `T.tile.atomic_add`（V 侧 UB src 逐行 + C 侧整 tile 直写两形态，§6.2 双后端结论表）/ workspace 命名自动握手 / GM 标量读——**ascendc 侧由 indexer_bwd（同形态全量）与 rev2 本算子实现双先例覆盖，pto 侧由 rev2 全量验证背书**（双后端跑通至 28.5ms）+ unsorted_segment_sum 双形态参考。**Expert fallback 所需查证项（仅升级时需要）：pto 侧手动 set_flag/wait_flag、T.Scope、L0 显式分配的可用性**（docs/TileLang-Ascend Programming Guide 差异章节，块 C §9.7 落实）。若 fallback 在 pto 不可用：pto 后端维持主路径形态（正确性同源、性能如实报告——符合双后端约束原文）；host 侧 `_CachedLauncher` 按 `(target, 编译键)` 双份缓存（indexer_bwd 先例）不变。

### 4.7 OOB 防护结论（语义继承 rev2；§6 查证修正后执行位置全部不变）

- **gather 侧（不变）**：`rows = clamp(idx, 0, seg_len−1)` 保护寻址 + mask（`idx ≤ s_i ∧ idx ≠ −1`）归零贡献——载入恒为合法行有限值，杜绝 0×NaN 传播；数学结果与 GPU 语义等价。
- **scatter 侧（§6 查证修正：散射保留 V 侧，掩码消费位置不变）**：dkv 逐行散射的掩码条件跳过机制 **rev2 原样保留**（V 侧自身持有 mask/idx UB 副本，逐行判 mask_row 后条件 atomic——机制 §7）。块 A 预置的"掩码向量传 C（机制 A/B）"随散射移交 C 侧的预期一并作废——**C 侧新增写面仅 ws_dq/ws_dkv 贡献槽整 tile 写**（写往 per-core 槽区，构造上界内、无 idx 参与寻址，无 OOB 面）；dq 终写在 V 侧（rev2 位置不变，行号 = token 行恒合法）；k3 postprocess 只读 ws_dkv 终态区（界内构造），亦无 OOB 面。

### 4.8 rev2 已验证组件继承处置表 + 决策汇总

**rev2 组件处置（任务书 §2 保留清单逐项）**：

| rev2 组件 | rev2 证据 | rev3 处置 |
|-----------|----------|----------|
| ws_kv 按块暂存（per-block stash） | iter-0 验证（pass-2 C 侧 bulk 重读形态） | **保留**：gather 腿的 C 侧批量读载体（§4.5 L1） |
| gather 全量 + 微体 staging | iter-1 验证 | **保留 + 批量化升级**：行读改 batch32 整 2D 形态（§4.5 L1） |
| scatter 载荷瘦身 | iter-1/2 记录 | **保留**（§6 查证修正：散射保留 V 侧，iter-1 优化形态——idx_ub 源 pos + 段微读合并——直接复用；任务书"若移 C 侧则退役"的条件不成立，本行即处置声明） |
| P/dP_raw V-only GM stash（零 flag transit） | iter-1 验证 | **保留**：pass-2 零 flag 二读机制原样（workspace 槽单次配对读 + pst 二读的双腿结构，§4.5 L2 / §5.5） |
| k3 行宽 32 | rev2 实现验证 | **保留**：k3 维持 Developer 形态（§4.2） |
| mask/bias 向量化 | iter-2 验证（vec 1.79→0.99ms） | **保留**：V 侧向量链主体（§9） |

**rev3 决策汇总表**：

| 决策点 | 结论 | 依据 / 先例 |
|--------|------|------------|
| 编程模式 | **Developer 形态 pair-loop 主路径**（k2/k3 零 Expert API）+ Expert fallback（判据：AIC 等 flag 占比 >30%，§4.2） | indexer_bwd 零 Expert API 70.9us 先例 + rev2 死因重定性（§3.7 结论）+ rev2 双后端 AUTO 栈验证背书 |
| bwd grid | persistent grid=20，(cid,vid) MIX_AIC_1_2，20 AIC + 40 AIV | reducesum / indexer_bwd（§3.1/3.2） |
| C/V 流水 | **pair-loop 深度流水**（DEP=1 token 错位，p%2 双缓冲，槽位安全由 pair 序读后写传递性在结构层保证，AUTO 生成 flag 序） | indexer_bwd L512-611（§4.3，时序 §9.5） |
| 两遍语义 | token 粒度两段 pair-loop 嵌套，单遍数据流闭合；Δ 屏障由跨 token 错位吸收（§5.4） | amend-1 锚点（§4.3） |
| dKV 散射 | **V 侧逐行 UB 原子 + bf16 贡献槽档**（§6 查证修正：随机行 C 侧直写不可行）；C 侧 L0C→GM 写仅用于贡献槽整 tile 写（per-core 连续槽区，非原子）；dq 终写 V 侧 UB→GM | §6 双后端寻址查证（unsorted_segment_sum L50 / pto dynamic / indexer_bwd layout=1 先例） |
| Δ-因式分解 | **否决**——两遍 stash 胜出（读量/发射/精度三轴数据对比，§6.4） | §6.4 对比表 |
| transit 数据流 | **腿融合单遍**：L2/L3 合槽、L1 批量化、L4 贡献档门控、L5 bf16 档 | §4.5 核算表 |
| AIV 读量目标 | ~62-70MB（fp32 L4 基准，vid 半片口径）≤75MB 达成；L5 fp32 兜底档 ~88-96MB 披露；最终预算 §13 | §4.5（vs rev2 147MB） |
| dq 累积 | V 侧 UB fp32 半片 [32,576]，贡献档 fp32 14.4MB 基准 / bf16 7.2MB 门控（§12）；C 侧全驻/dseg-outer 均数据否决 | §5.3 选型账 |
| OOB 防护 | gather clamp+mask（继承）；scatter V 侧掩码条件跳过（rev2 机制原样）；dq 终写无 OOB 面 | §4.7/§7 |
| 双后端 | 主路径单实现（rev2 AUTO 栈双后端验证背书）；pto Expert 查证仅 fallback 需要（§9.7） | §4.6 |
| 测试资产 | test_sparse_mla_bwd.py 全量直接复用，零重写 | 任务书 §3 |

## 5. Tiling 与 GEMM 策略（rev3）

> **继承声明**：rev2 §5.1-§5.5 的形态学结论零变化继承——GEMM 类归并（主仓 7 发 → Ascend 6 类，QK 双段合并 K=576 单发、dkv pair-GEMM 归一 acc 双发）、BN=128 主档 / BN=64 回退档、DSEG=2 分段与 512B 对齐核验、跨 token TS 捆绑不可行论证（gather 无共享 / M 扩展收益微小 / 小 N 已由 BN 加宽独立解决）、全维 %16 对齐。本节只写 **rev3 增量**：(a) Developer 形态下 L0C 以 alloc_fragment 逻辑累加器表达（编译器管物理槽，§5.1）；(b) dq 累积路径数据选型（§5.3）；(c) pair-loop GEMM 发射编排与 Δ 屏障吸收（§5.4）；(d) per-task 流量参数表更新（§5.5）。rev2 原文见 `history_version/design_sparse_mla_bwd_rev2.md` §5。

### 5.1 GEMM 类与 L0C 逻辑累加器（Developer 形态，时分排布）

六类 GEMM 形状**逐项继承 rev2 §5.1 表**（BN=128 / DSEG=2 档：G1 score 64×128×576、G2 dO·v 64×128×512、G3 dq 64×256×128×2 段、G4 dq_tail 64×64×128、G5 dkv pair 128×256×64×2发×2段、G6 dkv_tail 128×64×64）。rev3 增量：

1. **累加器表达**：Developer 形态下全部 acc 为 `T.alloc_fragment` 逻辑累加器（indexer_bwd L374-377 `logits_l0c/dq_l0c/dk_l0c` 同款），编译器管理 L0C 物理槽分配与 m16n16k16 分形装载——**不写 `T.alloc_L0C`**（Expert API，仅 fallback §8.7）。
2. **生命周期重定义（与 rev2 的关键差异）**：rev2 的 G5 acc 跨块长驻复用；rev3 pair-loop 编排下全部 acc 为**块内短生命周期**——G1/G2 分时共槽（G1 产出 transit 写出后 G2 覆写，rev2 §5.1 分时复用继承）；G3/G4 每块 `init=True` 单发（dq 跨块累积移 V 侧 UB，§5.3——**L0C 不跨块驻留 dq acc**，这是 L0C 可行性的关键）；G5 pair 双发共 acc（`init=True`/`init=False`，gqa_sink L434-436 形态）× DSEG 段顺序复用；G6 单发。**L0C 峰值 = G5 [128,256] fp32 = 128KB 贴线，单发独占段**（G1-G4/G6 均在 G5 前后时隙让出；rev2 §5.1 同结论，块内短生命周期使并发冲突面更小）。
3. **发射数/task**：produce 段（G1+G2）×NI=4 = 8 发 + consume 段（G3×2+G4+G5×4+G6）×NI=4 = 32 发 → **40 发/task**（与 rev2 BN=128 档相同，编排重排不改总数）。

### 5.2 分块参数档位（继承 rev2 §5.2 三档对比结论）

主选 **BN=128**（NI=4，L0C G5 128KB 贴线单发独占、L1 312KB、UB ~140KB 预算复核 §8）；**BN=64 回退档**（NI=8，G5 [64,256] 64KB 宽裕、发射 80 发/task）；BN=256 弃（L1 488KB 贴线）；BN=32 对照不选。参数表：

```python
BN    = 128   # topk 方向索引块合并（NI = topk//BN = 4）；BS=32 语义由 BN 吸收
DSEG  = 2     # dq/dkv 的 D=512 输出列分段（256 列/段，512B 对齐核验继承 rev2 §5.3）
Hseg  = 64    # M 维（padded_H）
v_block = 32  # 每 AIV 的 head 半片（2 AIV × 32 = 64，fwd §8.6 vid 分工恒等式）
grid_size = 20  # persistent（§3.1/§4.2）
PARITY = 2    # pair-loop 槽族双缓冲（token 奇偶，§5.4）
```

varlen 段长任意不产生块尾非整除（gather 行数恒整块，越界/无效行由 mask 归零 §7）；kv_group>1 通用路径 head padding 残余风险记 §14（考核 shape kv_group=1 无 pad）。

### 5.3 dq 累积路径选型（数据账，rev3 重证 rev2 §3.4 决策）

dq 跨 NI=4 块累加（acc [64,576] fp32 = **144KB > L0C 128KB**，全驻不可行）。三候选数据账：

| 候选 | 机制 | 数据账 | 结论 |
|------|------|--------|------|
| (a) C 侧 L0C 全驻 | DSEG 拆槽跨块 K 链（init=False） | [64,256]×2+[64,64] = 144KB > 128KB 物理不可行；且与 G5 128KB 独占段冲突 | **否决（容量）** |
| (b) C 侧 dseg-outer 重扫 | dseg 外层 × blk 内层，G3 每段跨块 K 链（L0C 只驻 64KB/段） | kv ws 重读 ×2：1152MB×2 / (20 AIC × ~17GB/s) ≈ **6.8ms > GEMM 下限 ~3-5ms** → C 侧 memory wall 取代 GEMM 成为瓶颈（单次读 3.4ms 可被 GEMM 覆盖，2 次不可） | **否决（带宽账）** |
| (c) V 侧 UB fp32 半片累积 | C 每块 G3/G4 init=True 单发 → ws_dq 贡献块 [64,576] → V 读半片 [32,576] 加入驻留 acc → task 末终写 bf16 | 贡献档读回 bf16 7.2MB / fp32 14.4MB per AIV（半片口径，§4.5 L4）；V 发射每块 2 tile-op ×4；UB 驻留 [32,576] fp32 = 72KB（§8 预算内）；L0C acc 块内短命（§5.1-2） | **选定** |

选定 (c)：rev2 §3.4 决策（"dq 累积移出 L0C 驻 UB fp32"）在 rev3 数据流下重证成立——(a)(b) 的否决账 above 为新增证据。贡献档 dtype：**fp32 14.4MB 为基准档**（与 rev2 精度路径一致，AIV 总读量 ~62-70MB ≤75MB）；**bf16 7.2MB 为优化档**（每块一次舍入 ×4 次，§12 域 B sim≥0.999 门控，通过则启用）。

### 5.4 pair-loop GEMM 编排（核心增量：发射流、循环骨架、Δ 屏障吸收）

**发射流（steady state，C 侧视角；pair 单元 = (token, block)，跨 token 错位 DEP = 1 token = NI 块）**：

```text
C:  P(t,0) P(t,1) P(t,2) P(t,3) │ C(t,0) P(t+1,0) C(t,1) P(t+1,1) C(t,2) P(t+1,2) C(t,3) P(t+1,3) │ C(t+1,0) ...
V:  E(t,0) E(t,1) E(t,2) E(t,3) [Δ(t) 定格] D(t,0) D(t,1) D(t,2) D(t,3) → 散射/终写(t) ‖ E(t+1,·) 交错
    P=produce（C：G1+G2；V：gather+mask+exp+P stash+Δ 前缀）  E=V 消费 produce 结果
    C=consume（C：G3-G6；V：dP 链+散射+dq 累积+终写）        D=V 的 consume 侧向量链
```

- **循环骨架（indexer_bwd L512-611 if-guard 形态移植）**：persistent 任务循环内，以 token 奇偶 `tslot = t % 2` 选槽族，块流扁平化为 `for p in T.serial(NI·2 + 1)` 型两段 guard 循环——`if p < NI: produce(t, p)`（C 侧 G1/G2 + transit 写；V 侧 gather/exp 链）；`if p >= 1: consume(t, p−1)`（V 侧 dP 链 + C 侧 G3-G6）+ 跨 token 边界的 produce(t+1, ·) 交错（steady state 每 C 消费一块生产下一 token 一块）。**全部为 T.serial + if guard（编译期常量界），无 T.Pipelined**（§3.3 约束遵守）。
- **Δ 屏障吸收（amend-1 语义的流水表达）**：Δ(t) 定格需 token t 全部 NI 块 produce 完成 ⇒ V 的 dP 链与 C 的 consume(t) 整段被屏障挡住——**pair-loop 的回应**：C 在等 Δ(t)/dP(t) 期间不空转，立即推进 produce(t+1,·)（gather(t+1) 由 V 在 E(t) 段后段隙执行）；V 的 D(t)（dP 链+散射）与 C 的 P(t+1) 重叠。屏障代价从"整段空等"（rev2 锁步形态）降为"consume 段延后一个 token 深度"——DEP=1 token 的 stash 槽位（PARITY=2）承载该深度。
- **槽族双缓冲**：kv ws、s/dp stash、P/dP transit、ws_dkv 贡献槽、ws_dq 贡献块 5 个跨核槽族 × token 奇偶双缓冲；V-only pst（P/dP_raw fp32 单侧 stash，iter-2 机制，非 workspace 命名零 flag）不参与配对。AUTO 模式 flag 对数 = 跨核槽族配对数 ≤ 12 ≤16（§3.6-5；§9 记账）。
- **kv 双读的编排表达（C 引擎 2× bulk 读，§4.3/§5.5 记账）**：G1/G2（produce 段）与 G3/G4（consume 段）均以 kv 为 B 操作数，L1 容不下 NI=4 块全驻（576KB > 512KB−Q/dO）→ kv 块按 (token, block) 从 workspace_kv GM 槽 bulk 读入 L1 两次（produce 一次 + consume 一次，每次 [128,576] bf16 144KB 驻 L1 至该块该段 GEMM 完）。C 侧 bulk 总读 ~297MB/AIC @ ~17GB/s 与 GEMM 重叠（§13 预算）；备选的 dq 侧 Δ-因式分解（消 consume 段 kv 读）已按精度/发射数据否决（§6.4）。
- **大 2D 写约束内建（§3.6-7）**：全部 transit 写按块粒度（[64,128]/[128,576] 平面 ≤2、NI 内循环分段）；dq 终写 [64,576] bf16 为单平面整写（640 宽 SEGV 签名规避：576 宽 + 块内循环形态，Stage 2 若复现 SEGV 则按 DSEG 拆段——记 §14 风险）。

### 5.5 per-task GM 中转流量参数表（rev3 口径更新）

**BN=128 主档，bf16 贡献槽档（fp32 档括注）；ws 槽形态 [20, PARITY, …] per-core 槽族（§8）**：

| 流 | shape（每槽） | dtype | 写侧 | 读侧 | 字节/task |
|----|--------------|-------|-------|------|-----------|
| workspace_kv | [NI·BN,576] | bf16 | AIV(gather 批量 2D) | C(整块 bulk，**2×：produce G1/G2 + consume G3/G4——L1 512KB 容不下 NI=4 块全驻 576KB**) | 0.5625MB（V 写）/ C 读 1.125MB |
| workspace_s / workspace_dp | [64,BN]×NI | **fp32**（amend-1 锚点：P/dP_raw 以先于 cast 的 fp32 量参与 Δ 归约与 dP 中心化——bf16 grain 档仅 §12 域 B 门控后可议） | C | AIV(半片行分工，pass-1 单次配对读) | 0.125MB/AIV |
| pst_p / pst_dp（V-only，非 ws 命名） | [64,BN]×NI | fp32 | AIV | AIV(pass-2 经 pst 重读——workspace 槽单次配对读 + 零 flag 二读，iter-2 机制) | 0.125MB/AIV |
| workspace_pd（dP+P V→C transit） | [64,BN]×NI×2 | bf16 | AIV(半片) | C | 0.125MB |
| workspace_dq（贡献块） | [64,576]×NI | fp32（bf16 档减半） | C | AIV(半片读累加) | 0.5625MB（0.28MB） |
| workspace_dkv（贡献槽） | [BN,576]×NI | **bf16**（fp32 兜底档 ×2） | C(块内 pair flush) | AIV(逐行微读) | 0.5625MB |
| ws_dkv（终态原子区） | [S,1,576] 全局 | fp32 | AIV 原子（有效行 0.985MB/task） | k3 | 4.5MB 全局 |
| Q/dO GM→L1 | [64,576]+[64,512] | bf16 | — | C 直读 | 0.203MB |
| dq 终写 | [64,576] | bf16 | AIV(半片，UB acc 终写) | — | 0.141MB |

与 rev2 §5.5 表的差异：workspace_dkv 贡献槽 fp32→bf16（§6.3）；workspace_dq 明确 fp32 基准/bf16 优化档（§5.3）；workspace_s/dp 维持 fp32 grain 且显式化双腿结构（transit 配对读 + pst 零 flag 二读，§4.5 L2 修正口径）；workspace_kv 显式化 C 侧 2× bulk 读（§4.3 修正）。**C 侧 bulk 总读/task ≈ 0.203(Q/dO) + 1.125(kv) + 0.125(pd) = 1.45MB**（≈297MB/AIC 全序列，@C mte2 ~17GB/s 与 GEMM 重叠——§13 记账）；全序列 AIV 读量核算入 §13.1 逐腿表（fp32 基准档 ~85-99MB / 全 bf16 门控档 ~72MB，§4.5 合计行）。

## 6. dKV 散射策略专题（rev3：随机行散射寻址能力查证与选型）

> **查证任务（任务书 §2 开放问题，块 B 必须解决）**：`T.tile.atomic_add` 的 L0C→GM 直写是连续 region 单基址（DataCopyPad 形态）——本算子散射目标为**随机行**（逐 token randperm）。必须查证行间址形态（unsorted_segment_sum.py L50 / pto 侧 `atomic_add_l0c_to_gm_dynamic`）能否承载 C 侧 L0C 源随机行散射，二选一：(a) 行间址承载；(b) C 侧直写仅用于可连续化腿 + 随机行散射保留 V 侧 UB 逐行（或分桶散射，需评估代价）。**结论：选 (b)**——(a) 不可行（双后端单调用均为单基址连续 region，逐行需 L0C 行切片 src 且实证不可靠），分桶散射否决（§6.2）；C 侧 L0C→GM 写仅用于贡献槽整 tile 写（可连续化腿）；dkv 随机行散射保留 V 侧逐行 + bf16 贡献槽档（§6.3）；两遍 stash vs Δ-因式分解以数据选型：**两遍 stash 胜出**（§6.4）。双后端可用性结论见 §6.2 末表。

### 6.1 双后端寻址能力源码级查证（2026-10-15 块 B 实查）

**ascendc 后端**：

1. **模板形态**（`src/tl_templates/ascend/common.h` L295-306）：`atomic_add_l0c_to_gm(dstTensor, srcTensor, realDstN, realTailM, realTailN)` = `AscendC::SetAtomicAdd<T2>()` + `copy_l0c_to_gm`（DataCopyPad 家族）——**dst 为单基址 GlobalTensor，传输一整块连续 [rows, cols] region**。同文件 L283 `atomic_add_ub_to_gm` 同构（UB src 变体）。
2. **op 层基址计算**（`src/op/ascend.cc` L866-890，tl.ascend_atomic_add V1 lowering）：`dst_offset = dst_buffer.OffsetOf(dst_scalar_indices).back()`，`dst_scalar_indices` 允许**运行时标量表达式**——即前端 `T.tile.atomic_add(Output[seg_ub[0], col], src)` 的行号可为运行时值（UB 标量/GM 标量）。约束同查（L778-812）：dst 必须 global、src 须 UB 或 L0C(wmma.accumulator)、**src/dst dtype 一致**、src 内层（列）维须编译期常量。
3. **实证先例**：`examples/unsorted_segment_sum/unsorted_segment_sum.py` L50 `T.tile.atomic_add(Output[seg_ub[0], d_blk*block_D], row_ub)`——**UB src + 运行时行基址**逐行散射，全 dtype 测试通过（ascendc 默认 target）；indexer_bwd L697 `T.tile.atomic_add(ws_dk[bos + pos, 0], dk_row_ub)`（pos = TopkIndices GM 标量运行时读）同形态。

**pto 后端**：

4. **模板形态**（`src/tl_templates/pto/common.h` L1487-1521）：`atomic_add_l0c_to_gm_dynamic(handle, shape, stride, l0c_shape_addr, l0c_offset, valid_row, valid_col)`——`handle` 为 `__gm__ T1*` 裸指针，shape 第 3/4 维为 DYNAMIC（运行时 valid_row/valid_col），内部 `pto::TSTORE<..., AtomicType::AtomicAdd>` 写出。
5. **codegen 调用点**（`src/target/codegen_ascend_pto.cc` GMCopyCall L1283-1390）：**统一恒发 dynamic 变体**（"Always use dynamic version for unified codegen"，L1352-1354），GM 基址 = `copy_base_addr_map_.at(id) + PrintExpr(gm_info.offset)`——**offset 为运行时表达式，支持动态基址**；L1841-1846 确认 `atomic_add_l0c_to_gm`/`atomic_add_ub_to_gm` 均走此路径。`atomic_add_ub_to_gm_dynamic`（L1452）同构（UB src）。
6. **共同限制（决定性）**：**单次调用 = 单（可运行时偏移的）基址 + 单连续 region**。L0C 多行 tile 的行间随机散射不可表达为少量调用——每行一次调用则 src 需 **L0C 行切片**，而 indexer_bwd note 4 实证（rev2 L0 调试证据，docstring L69-71）："**per-token L0C row-slice copies are unreliable beyond the first fractal tile**"（分形布局 zN band 不容许行切片，note 2 L57-60 同源："every L0C->GM write is a whole-tile copy"）。M=1 退化 GEMM 亦不可行（分形退化、发射爆炸）。

**最强先例的对照印证**：indexer_bwd 对两种索引形态分别选型——layout=0（arange-tiled，dst 行连续）用 **C 侧 L0C direct src 整 tile region 原子**（L681 `T.tile.atomic_add(ws_dk[bos:bos+topk, 0:dim], dk_l0c)`，dk sim 0.9977 验证）；layout=1（随机行）用 **C 整片 L0C→GM ws staging + V 侧逐行 UB src 原子**（L691-697）——**仓内最强精调先例对随机行明确回避 C 侧直写**，与本查证结论独立互证。

### 6.2 二选一选型与分桶评估

- **(a) C 侧 L0C 直写随机行散射：不可行**。依据链：§6.1-6（单基址 region + L0C 行切片不可靠 + M=1 退化）——**双后端一致**（ascendc DataCopyPad 形态 / pto TSTORE 形态物理传输均为连续 region）。"行间址"能力**存在但只到基址级**（运行时行偏移 ✓，行间离散目标 ✗）——unsorted_segment_sum L50 的行间址正是"UB src + 每行一调用"形态，其 src 在 **UB** 而非 L0C。
- **(b) 选定**：**C 侧 L0C→GM 写仅用于可连续化腿——ws_dq / ws_dkv 贡献槽整 tile 写**（per-core 连续槽区、非原子普通 copy；indexer_bwd L691 `T.copy(dk_l0c, workspace_dkf[cid, 0:topk, 0:dim])` 同款 staging 形态）；**dkv 随机行散射保留 V 侧 UB 逐行**（iter-1 优化后形态直接复用：idx_ub 源 pos + 段微读合并，iter-1 实测 −42.2% 组成部分）；dq 终写为 V 侧 UB→GM（§5.3 选定 V 侧累积，rev2 位置不变）。
- **分桶散射评估（否决，数据）**：① token 内 idx 排序：512 个 [0,2048) 均匀随机值的期望连续 run 数 ≈ 512 − 511·(512/2048) ≈ **384 runs**（仅 −25% 调用数），且每 run 仍需 L0C 行切片 src（§6.1-6 不可靠）——**死于此**；② 跨 token 按 KV 行分桶 = rev2 方案 3 行中心重构（P/dP 重算计算量翻倍 / 物化 537MB / q·dO 行 gather 74GB，rev2 §6.3 数据闭合）——**维持否决**；③ per-core 私有区 + 归约（rev2 方案 2）：原子溢价实测仅 2.4%（iter-1 var_noatomic 差分 1367us/56.4ms）+ 90MB ws 挤占 L2 46%——**维持否决**（rev2 §6.4 权衡继承）。

**双后端可用性结论（选定方案）**：

| 后端 | V 侧逐行 UB 原子（dkv 散射） | C 侧整 tile L0C→GM 写（贡献槽 staging） | 依据 |
|------|------------------------------|------------------------------|------|
| ascendc | ✓ 可用（已验证：unsorted_segment_sum 全 dtype 测试 / indexer_bwd layout=1） | ✓ 可用（indexer_bwd L691 workspace_dkf staging + L681 region 原子同族 copy_l0c_to_gm 路径） | §6.1-2/3 |
| pto | ✓ 可用（unified codegen 恒走 `atomic_add_ub_to_gm_dynamic`，运行时基址 + TSTORE AtomicAdd） | ✓ 可用（`copy_l0c_to_gm_dynamic` 同 GMCopyCall 路径，L1288 分发表内） | §6.1-4/5 |

### 6.3 贡献槽 bf16 档与精度门控（L5 读量 ~53→~26MB）

**机制**：G5 pair/G6 的块内 fp32 求和在 L0C 完成后，贡献槽写出降为 **bf16**（[128,576] bf16 = 144KB/槽，C bulk 出口半量）；V 侧逐行读回 bf16 行（1152B/行）→ UB 内 cast fp32 → `T.tile.atomic_add` fp32（dst=ws_dkv 终态区 fp32，**原子累积链全程 fp32**——src/dst 一致约束 §3.6-4 由 cast 满足）→ k3 单次 cast。

**精度账**：每 (token, row) 贡献在出 L0C 时**单次 bf16 舍入**（2^-8 相对），跨 token ~448 次贡献的 GM 原子累积保持 fp32；随机符号下累积相对误差 ~2^-8/√448 ≈ 0.02% 量级——**预期可过域 B sim≥0.999，但必须实测门控**（amend-2 收紧门限，§12.3）：域 B（含 σ=10 valrange）dkv sim ≥ 0.999 阻塞判据。**fp32 兜底档**（零额外舍入、与 rev2 精度路径逐位一致）保留：L5=~53MB、AIV 总读量 ~88-96MB 超 75MB 目标——仅当 bf16 档门控 FAIL 时启用，性能超标如实披露（§13/§14）。

**发射账（iter-1 优化形态复用）**：每有效行 = 1 次 [576] bf16 行读（段微读合并后批粒度）+ 1 次 cast + 1 次原子 + 谓词判断；掩码行（~131k/1.05M 槽）零读零发射（§7.3）。全序列有效行 917,760 / 40 AIV ≈ 22,944 行/AIV——**由 pair-loop 跨 token 错位吸收**（V 散射(t) 与 C produce(t+1) 重叠，§5.4），残余成本入 §13.1 逐腿表。

### 6.4 Δ-因式分解尾声 vs 两遍 stash（读量/发射对比选型，数据闭合）

**候选定义**：Ⅰ 两遍 stash（选定，§4.3/§4.5 结构）——produce 段 V 算 P/Δ 前缀，Δ 定格后 consume 段 V 算 dP = P⊙(dP_raw−Δ)·sm_scale（**中心化减法在 V 侧 fp32 UB 内完成，单次 bf16 舍入作用于小量**）→ dP/P transit → C 消费 GEMM。Ⅱ Δ-因式分解尾声——dq = sm_scale·(A−Δ⊙B)（A=Σ(P⊙dP_raw)·k、B=ΣP·k 单遍内累积，Δ 仅尾声需要）；dkv 的 Δ 项 Σ_h P[h,j]·Δ[h]·q[h,:] 不易分解 → 拆 termA（(P⊙dP_raw)^T·q，单遍内）+ termB（P^T·(Δ⊙q)，尾声）。

**per-token 数据账（BN=128/NI=4/DSEG=2，全序列 2048 token 折算 per-AIV）**：

| 轴 | Ⅰ 两遍 stash | Ⅱ Δ-因式分解 + P 物化尾声 | 判定 |
|----|-------------|---------------------------|------|
| V 侧读量 | s/dp+pst 双读 12.8MB（fp32 半片）+ dkv 贡献槽 26.4MB（bf16）+ dq 贡献 7.2-14.4MB ≈ **46-53MB** | s/dp 6.4MB（无 pst/dq 腿——dq A/B 在 C 侧 L0C 累积、dP 不存在）+ **dkv termA+termB 双轮贡献槽 ~52.9MB**（termA 逐块 flush + termB 尾声 flush，均随机行散射源）+ P 尾声重读（C bulk，不入 AIV）≈ **59MB** | **Ⅰ 胜**（Ⅱ V 读量 +6~+13MB 且双轮散射） |
| GEMM 发射 | 40 发/task（produce 8 + consume 32） | ~64 发/task（in-pass 52：G1/G2 + termA 3 + P^T·dO 2 + dqA 3 + dqB 3 每块；尾声 12：termB [512,576] M-tile×DSEG）——**+60%** | **Ⅰ 胜** |
| 散射发射 | 1 轮（917,760 有效行） | 2 轮（termA + termB 各 917,760 行；或减法串行化 = 读量再翻倍） | **Ⅰ 胜** |
| 精度 | 中心化减法 fp32 UB 内 → 小量单次 bf16 舍入（rev2 验证路径） | **A−Δ⊙B 形态灾难性抵消**：dP=P⊙(dP_raw−Δ) 为中心化小量 ⇒ A ≈ Δ⊙B 近等幅，两者各自 bf16 舍入（2^-8·\|A\| 量级误差）后相减，而 \|dq\| ≪ \|A\| → 相对误差 O(2^-8·\|A\|/\|dq\|) 可达 O(1)；走 fp32 transit 则读量再翻倍 | **Ⅰ 决定性胜出** |
| Δ 屏障 | 挡 consume 段 32 发——**pair-loop 跨 token 错位吸收**（C 等 Δ 期间 produce(t+1)，§5.4） | 仅挡尾声 12 发（Ⅱ 唯一优势） | Ⅱ 优势被 Ⅰ 的流水结构吸收，不构成交换理由 |

**结论：Ⅰ 两遍 stash 全面胜出（读量 / 发射 / 精度三轴 + 屏障优势被 pair-loop 吸收）——Ⅱ 否决，数据闭合非拍脑袋**。Ⅱ 的唯一理论收益（Δ 屏障缩小）在 pair-loop 错位结构下不再稀缺：屏障等待转化为 produce(t+1) 的填充工作（§5.4 时序）。

### 6.5 工作集、L2 吸收与吞吐论证（继承 rev2 §6.2/6.4 框架，rev3 数字）

- **L2 容量**：192MiB（Ascend910B3.ini `l2_size=201326592`）。
- **散射相关常驻工作集（rev3 bf16 档）**：ws_dkv 终态 4.5MB + KV 原始 2.25MB + indices 4MB + 贡献槽 20 核×2 奇偶×4 块×[128,576] bf16 ≈ 5.9MB + ws_dq/pst/transit 槽族 ~3MB ≈ **~20MB ≪ 192MiB**——贡献槽读、终态原子写全量驻 L2，HBM 仅见终态 4.5MB。
- **谨慎口径**：L2 吸收率受 hash 冲突干扰，预算按单核 MTE 实测口径核算（AIV 行读 ~26GB/s/核、C mte2 bulk 15-19.3GB/s/核），L2 吸收仅作交叉验证——逐腿入 §13.1。
- **原子吞吐锚点**：iter-1 var_noatomic 差分——**原子溢价仅 2.4%**（1367us/56.4ms），真瓶颈是逐行串行发射（PIPE_ALL + GM 标量读 + 微读 + 握手）；rev3 的响应不是替换原子（方案 2 已否决）而是 **pair-loop 重叠吸收发射链**（§5.4）+ bf16 半量读（§6.3）。

### 6.6 fallback 判据（散射腿）

- **精度触发**：bf16 贡献槽档在域 B（含 σ=10）dkv sim < 0.999 且排除实现 bug → 切 **fp32 兜底档**（L5=~53MB，AIV 总读量 ~88-96MB 超 75MB 目标，性能如实披露，§13.3 重估）——精度路径与 rev2 逐位一致，门禁必过。
- **性能触发**：msprof 散射段（逐行发射链）占 bwd 段 > 1/3 且 pair-loop 重叠后仍为关键路径 → 首选检核 §5.4 错位深度（DEP 是否退化为 0——即 produce/consume 是否又被编译器按序锁死，对照 §4.2 升级判据走 Expert fallback）；散射机制本身无 AUTO 栈内替代（逐行微读模式已冻结为唯一可行形态，§3.7 补充证伪；分桶/私有区/行中心三路均数据否决，§6.2）。
- **切换成本**：fp32 档仅改贡献槽 dtype 参数（发射结构不变）；Expert fallback 仅改同步机制（§4.2 边界）——接口签名与测试资产均不动。

## 7. OOB 防护（rev3：散射保留 V 侧，rev2 机制继承 + 执行位置声明）

> 继承声明：rev2 §7 的源头分类（尾部填充 idx=S / 显式 -1 / 任意 int32 防御性覆盖）与"gather clamp 寻址 + mask 归零、scatter 掩码行条件跳过、**mask 判据一律用原始 idx 评估、clamp 仅用于寻址**"的正确性枢纽零变化继承。本节按 §6 选定方案（散射保留 V 侧、dq 移 C 侧终写）重述执行位置并补 rev3 增量。

### 7.1 防护面清单（rev3 位置）

| 防护面 | 位置 | 机制 | OOB 风险 |
|--------|------|------|---------|
| KV gather 读 | V 侧（不变） | clamp 寻址 + mask 归零（§7.2） | 有（idx=S/-1/任意）→ 已防护 |
| dkv scatter 写 | **V 侧（§6 选定，位置不变）** | 掩码行条件跳过（§7.3） | 有 → 已防护 |
| ws_dq/ws_dkv 贡献槽写 | **C 侧（rev3 新增面）** | 整 tile 写往 per-core 槽区 [20, PARITY, …]（构造上界内，无 idx 参与寻址） | **无**（不需掩码） |
| dq 终写 | V 侧（rev2 位置不变） | 行号 = token 行 s_row 恒合法，写侧独占 | 无 |
| k3 postprocess | AIV-only（不变） | 只读 ws_dkv 终态区 [S,1,576]（界内构造）+ 行宽 32 写 dkv bf16 | 无 |

### 7.2 gather 侧防护：clamp 寻址 + mask 归零（继承 rev2 §7.2 全文口径）

AIV 侧每索引块（BN 行）向量链时序（gqa_sink L393-423 / fwd §8.4 同构）：① idx_ub ← `T.copy(Indices[s_row, g, i0:i0+BN])`；② mask_2d ← `(idx_ub <= s_i) & (idx_ub != -1)`（**原始 idx，clamp 前评估**，`T.tile.compare` ×2 + `bitwise_and`——mask/bias 向量化继承）；③ rows_ub ← `clamp(idx_ub, 0, seg_len−1)`（段内上界，`T.tile.min/max`）；④ gather：`kv_ws[j,:] ← T.copy(KV[bos + rows_ub[j], :])`（行号 ∈ [bos, eos−1) 恒合法，1152B 连续行读、批量 2D 形态 §4.5 L1）；⑤ P/dP 链：`P = select(mask_2d, exp(s·sm_scale − lse), 0)`。

要点继承：③ 上界为段内 seg_len−1（填充值 S > 任意 s_i → mask 恒 false，与 clamp 后载入行无关，多段 varlen 语义不变）；−1 在 ② 已判（clamp 后载入行 0 合法有限值、贡献归零）；重复索引不去重（各自 gather 各自计贡献，§2.5 语义锚点）；**NaN 无法经 gather 传播**（载入恒有限 + 掩码项在 P/dP 层归零——rev2 "0×NaN 根绝"论证原样）。

### 7.3 scatter 侧防护：掩码行条件跳过（继承 rev2 §7.3，V 侧逐行）

AIV 侧散射循环（indexer_bwd L692-697 / unsorted_segment_sum L42-50 同构先例，两 AIV 分摊 BN 行）：

```text
for j in T.serial(BN):                    # 编译期常量（§3.3）
    if mask_row[j]:                       # 原始 idx 的 mask（§7.2 ② 同源），per-row 动态 if
        T.copy(workspace_dkv[tslot, blk, j, :], row_ub)   # bf16 行 [576] → UB（段微读合并形态）
        T.tile.atomic_add(ws_dkv[bos + idx_ub[j], 0], row_ub_fp32)  # cast 后 fp32 原子；寻址用原始 idx
```

- **掩码行零写入**（连 0 贡献都不写）：省 ~131k 行 × 1152B ≈ 0.15GB bf16 档无效读+原子（§6.3 发射账），且**根绝 OOB 写**——主仓 L232-241 无条件散射（idx=S 越界写）在 NPU 上为必须消除的未定义行为，rev2 已消除、rev3 原样。
- **寻址用原始 idx 而非 clamp 后 rows**：mask true 的行 idx ∈ [0, s_i] ⊂ [0, seg_len−1] 天然合法（写侧合法性由 mask 保证、读侧由 clamp 保证——双保险各司其职，rev2 枢纽继承）。
- **"零填充 + clamp 到固定行"否决复述**：~131k 掩码槽零贡献集中单行 → 同地址原子热点（L2 行锁序列化）+ 无意义流量——维持否决。
- **pair 贡献原子性**：G5 pair 双发在 L0C 内完成求和后整行散射（dPᵀ@q + Pᵀ@dO 一次原子），掩码行两贡献分量皆 0、整行跳过——不会出现"半贡献"散射。
- **termB 类双轮散射不存在**：§6.4 已否决 Δ-因式分解，dkv 每有效行恰一轮散射（单原子读-改-写序列，与主仓逐元素原子语义一致性论证继承 rev2 §7.3）。

### 7.4 OOB 等价性结论（与 GPU 语义）

数学结果与主仓 GPU 语义等价的论证链继承 rev2：gather 侧载入行与掩码归零使被掩码槽贡献恰为 0（GPU 侥幸通过的 0×垃圾路径在 NPU 上为确定性 0）；scatter 侧掩码行零写入 vs GPU 的越界写（不可观察副作用）——输出 dq/dkv 逐元素等价；测试资产（test_sparse_mla_bwd.py 含 OOB 构造用例）零重写直接复用（任务书 §3）。

## 8. 内存规划（rev3：pair-loop 双缓冲 UB/L1/L0C 预算 + stash 槽布局 + 跨任务循环生命周期）

> 参数承接 §5.2/§5.5：BN=128、NI=4、DSEG=2、Hseg=64、v_block=32、PARITY=2、grid=20；平台每核 L1 512KB / UB 192KB / L0C 128KB（文档头锚点）。rev3 相对 rev2 §8 的结构性变化：① 全部 GM 中转槽族按 **[20 核, PARITY=2 奇偶, NI=4 块, …] 驻留形态**显式布局（pair-loop 错位深度的槽位承载，§8.4）；② L1 新增 q/do 的 **token 奇偶双槽**（交错段两 token 操作数并存，§8.2）；③ UB 按 E/D 段错位并存口径重排（§8.3）+ 跨任务循环生命周期声明（§8.6，UB 规划器约束 3.6-6 的规避口径）。**L5 fp32 兜底档**（贡献槽 fp32，§6.3）只改 workspace_dkv/dq 的 dtype 字节翻倍，槽结构与本节布局不变——各表括注兜底档数字。

### 8.1 L0C（128KB/核）：逻辑累加器分时复用（alloc_fragment，块内短生命周期）

**Developer 形态：不写 `T.alloc_L0C`**——全部累加器为 `T.alloc_fragment` 逻辑形态（indexer_bwd L374-377 先例），L0C 物理槽由编译器（MEMORY_PLANNING liveness）按 §5.1-2 的块内短生命周期自动分时。槽位清单（每索引块 i，NI=4 循环内）：

| 槽 | shape | 容量 | 服务 GEMM | 生命周期（块内） |
|----|-------|------|-----------|--------------------|
| acc_s | [64, 128] | 32KB | G1 / G2 分时共槽 | G1 出→写 workspace_s→死；G2 覆写→写 workspace_dp→死 |
| acc_dq | [64, 256] | 64KB | G3（每 DSEG 段一发） | 段出→写 workspace_dq 列段→死（**不跨块驻留**，§5.1-2） |
| acc_dq_tail | [64, 64] | 16KB | G4 | 出→写 workspace_dq [512:576] 列段→死 |
| acc_dkv | [128, 256] | **128KB** | G5 pair（双发同槽累加 init=True/False）× DSEG 段顺序复用 | 段出→整 tile 写 workspace_dkv 贡献槽→死；**单发独占段** |
| acc_dkv_tail | [128, 64] | 32KB | G6 | 出→写贡献槽尾段列→死 |

**峰值分析（程序序分时，同 rev2 §8.1 骨架但块内闭合）**：

```text
阶段①  G1+G2（produce） ：acc_s 32KB                                  → 32KB
阶段②  G3+G4（consume） ：acc_dq 64 + acc_dq_tail 16                  → 80KB（acc_s 已死）
阶段③  G5 pair（consume）：acc_dkv 128KB（①②槽全死）                 → 128KB 贴线独占
阶段④  G6（consume）     ：acc_dkv_tail 32KB（acc_dkv 已死）          → 32KB
峰值 = 128KB / 128KB 上限 = 100%（阶段③独占；块内短生命周期使跨块并发冲突面为零）
```

- **贴线成立条件**：G5 独占段内无其他活槽——produce 段槽（阶段①）在 consume 段（②③④）前已死，程序序保证；pair-loop 交错流中 P(t+1,i)（阶段①）与 C(t,i)（阶段②③④）交替推进，**两者恰好错开 L0C 峰值段**（① 32KB 与 ③ 128KB 不同时隙）——若 MEMORY_PLANNING 按并列分配（P 的 acc_s 与 C 的 acc_dkv 同峰 160KB 超限），Stage 2 以块内程序序断言（produce 的 workspace 写出语句在 consume GEMM 语句前）显式化 liveness 边界；仍失败则 BN=64 回退档（G5 [64,256]=64KB，峰值 96KB/75%，§5.2）。
- **L0C 起始 256 元素对齐**（§5.3 注 C 约束继承）：64×128 / 64×256 / 128×256 / 128×64 均为 256 倍数 ✓。

### 8.2 L1（512KB/核）：Cube GEMM 操作数——q/do token 奇偶双槽 + kv 单槽双读（rev3 结构变化）

pair-loop 稳态交错流（§5.4：C(t,i) 与 P(t+1,i) 交替）下，token t 的 q/dO（C(t,·) 的 G5/G5'/G6 操作数）与 token t+1 的 q/dO（P(t+1,·) 的 G1/G2 操作数）**同时活跃** → q_l1/do_l1 必须 token 奇偶双槽；kv_l1 为块粒度双读（produce/consume 各装一次，每次 GEMM 完即死）单槽交替装载。rev2 §8.2 的每 task 单次装载形态不满足交错并存，此为 rev3 L1 布局的关键差异：

| buffer | shape | dtype | 槽数 | 容量 | 用途（§5.1 GEMM） |
|--------|-------|-------|------|------|-------------------|
| q_l1 | [64, 576] | bf16 | **2（token 奇偶）** | 144KB | G1 的 A（t+1）/ G5、G6 的 B 列切片（t）——token 粒度装载，交错段两 token 并存 |
| do_l1 | [64, 512] | bf16 | **2（token 奇偶）** | 128KB | G2 的 A（t+1）/ G5' 的 B 列切片（t）——同上 |
| kv_l1 | [128, 576] | bf16 | 1（块粒度交替） | 144KB | G1/G2 的 Bᵀ（produce 读）/ G3/G4 的 B 列切片（consume 读）——**同块两次装载**（§5.4 kv 双读：每次驻 L1 至该块该段 GEMM 完） |
| dp_l1 | [64, 128] | bf16 | 1 | 16KB | G3/G4 的 A / G5 的 Aᵀ（consume 段块粒度覆写；produce 段不使用 → 单槽安全） |
| p_l1 | [64, 128] | bf16 | 1 | 16KB | G5' 的 Aᵀ（同上） |
| **合计** | | | | **448KB（87.5%）** | 余 64KB |

- **kv_l1 单槽装载序（交错流）**：C(t,0) 装 kv(t,0)→G3-G6 → P(t+1,0) 装 kv(t+1,0)→G1/G2 → C(t,1) 装 kv(t,1)→…——装载 8 次/task（NI=4 × produce/consume），144KB/次，MTE2 与 Cube GEMM 异单元重叠（C 侧 bulk 总读 297MB/AIC 已 §5.5 记账，@~17GB/s 与 GEMM 重叠，§13 预算）；装载间隙空泡风险记 §14。**kv_l1 双槽（produce/consume 各一）需 +144KB = 592KB 超 512KB 物理不可行**——单槽双读是 L1 容量强制的编排（不是可选优化）。
- **512B 对齐**：各 buffer 行宽 576×2B=1152B / 512×2B=1024B / 128×2B=256B——行起点均 512B 倍数 ✓；DSEG 列切片段起始对齐 §5.3 已核验。
- **BN=64 回退档**：kv_l1 [64,576] 72KB、q/do 双槽不变 → 合计 376KB（73%）宽裕。

### 8.3 UB（192KB/每 AIV，v_block=32 head 半片）：E/D 段错位并存预算（rev3 重排）

V 侧稳态 = D(t,·)（consume 向量链）与 E(t+1,·)（produce 向量链）交错（§5.4）→ 两段槽位**并存**预算（rev2 的"s_ub P 链后死槽复用为 dv_ub"缓解在错位结构下**不成立**——s_ub(t+1)（E 段 P 链）与 p_ub(t)（D 段 pst 重读）同时活跃，这是 pair-loop 的 UB 代价，必须显式记账）：

| 组 | buffer | shape | dtype | 容量 | 生命周期 |
|----|--------|-------|-------|------|---------|
| 持续驻留 | acc_dq_ub | [32, 576] | fp32 | 72KB | 跨 NI 块 dq 累积（§5.3 选定 (c)：task 头清零、每块 +=、task 尾 cast bf16 终写） |
| 持续驻留 | idx_ub / rows_ub / mask_ub | [128] | int32/fp32 | ~1.5KB | 每块刷新（§7.2 ①②③链） |
| 持续驻留 | lse_ub / delta_ub / delta_acc | [32]×3 | fp32 | ~0.75KB | GM 标量读 + Δ 前缀累加器（E 段累加、Δ 定格后冻结，amend-1） |
| E 段（produce V） | kv_gather_staging | [32, 576] | bf16 | 36KB | batch32 整 2D gather staging（§4.5 L1 批量化：128 行块 = 4 批循环，每批写出 workspace_kv 后死——段内时序在 ws_s 读之前，liveness 与 score 槽错开） |
| E 段 | s_ub | [32, 128] | fp32 | 16KB | workspace_s 半片配对读 → P 链（exp(s·sm_scale − lse)）→ 写 pst_p / p_bf16 后死 |
| E 段 | dpraw_ub | [32, 128] | fp32 | 16KB | workspace_dp 半片配对读 → 写 pst_dp → Δ 前缀累加（P⊙dP_raw，fp32 先于 cast，amend-1 锚点）后死 |
| E 段 | p_bf16_ub | [32, 128] | bf16 | 8KB | P cast 写出槽（→ workspace_pd P 半） |
| D 段（consume V） | p_ub | [32, 128] | fp32 | 16KB | pst_p 零 flag 二读（iter-2 机制）→ dP 链 |
| D 段 | dpraw_ub2 | [32, 128] | fp32 | 16KB | pst_dp 二读 → dP = P⊙(dP_raw−Δ)·sm_scale（中心化 fp32 UB 内，§6.4 Ⅰ 精度路径） |
| D 段 | dp_bf16_ub | [32, 128] | bf16 | 8KB | dP cast 写出槽（→ workspace_pd dP 半） |
| D 段 | dq_seg_ub | [32, 144] | fp32 | 18KB | workspace_dq 贡献块半片**列段切片读**（[32,576] 半片按 4 列段 ×144 循环流加 acc_dq_ub——避免全片 [32,576] fp32 72KB 驻留；bf16 档减半 9KB） |
| D 段 | row_ub | [576]×2 | bf16+fp32 | ~3.4KB | 散射贡献行微读（bf16 行读入 + cast fp32 → T.tile.atomic_add src，§7.3） |
| 预留 | misc（cast 链 / 段标量 / sm_scale） | — | — | ~4KB | 紧凑口径 |
| **稳态峰值（分时口径）** | | | | **~184KB（95.8%）** | 持续 ~78.25 + E 段峰值 max(36, 16+16+8)=40 + D 段 61.4 ≈ 184KB |
| 保守口径（全并列） | | | | ~220KB **超限** | 削减链见下 |

**UB 贴线削减链（若 MEMORY_PLANNING 按并列分配报 `available: 0`——证伪签名 3.6-6——按序降级）**：① gather staging 降 batch16 [16,576] 18KB（保守口径 −18KB，2D 写出批数 ×2 仍在 3.6-7 SEGV 边界内）；② E 段 score 槽（s_ub/dpraw_ub/p_bf16_ub 40KB）与 gather staging 槽（36KB）显式段内分时——Stage 2 以程序序断言（staging 写出语句置于 ws_s 读语句前）压缩 liveness 区间；③ workspace_dq 贡献档切 bf16（dq_seg_ub 9KB + §5.3 优化档，需 §12 域 B 门控）；④ BN=64 档（s/dp/p/dp_seg 槽减半，UB 降 ~30KB，发射 80 发/task）。**设计基线 = 分时口径 184KB + 削减链预案**（首版按分时口径组织程序序，编译期以 UB 规划器实际分配验证——rev2 实现 28.5ms 路径的规划器行为为参照系）。

### 8.4 workspace / stash 槽布局（[20, PARITY, NI, …] 驻留形态 + 命名握手清单 + GM 总量勘误）

**槽族全表**（per-core [20] × token 奇偶 [PARITY=2] × 块 [NI=4] 驻留——pair-loop 错位深度 DEP=1 token 的槽位承载，§4.3；NI 维驻留的必要性：C(t,i) flush 与 V D(t,i) 读之间隔 0-8 块时间，第 i+1 块写不可覆写第 i 块未读槽）：

| 槽族 | 单槽 shape | dtype | 单槽字节 | 总量（20×2×4） | 方向（写→读） | flag 配对 |
|------|-----------|-------|---------|---------------|---------------|----------|
| workspace_kv | [128, 576] | bf16 | 144KB | **23.0MB** | V gather → C produce G1/G2 + consume G3/G4（2 读） | ✓（命名握手） |
| workspace_s | [64, 128] | fp32 | 32KB | 5.2MB | C G1 → V E 段半片配对读 | ✓ |
| workspace_dp | [64, 128] | fp32 | 32KB | 5.2MB | C G2 → V E 段半片配对读 | ✓ |
| pst_p / pst_dp（V-only） | [64, 128]×2 族 | fp32 | 32KB/族 | 10.5MB | V E 写 → V D 零 flag 二读（iter-2 机制） | **✗（非 workspace 命名，程序序单侧）** |
| workspace_pd（P 半 + dP 半） | [64, 128]×2 半 | bf16 | 16KB/半 | 5.2MB | V（E 段写 P 半 / D 段写 dP 半）→ C consume G3-G6 | ✓ |
| workspace_dq | [64, 576] | fp32（bf16 档 72KB） | 144KB | 23.6MB（bf16 11.8MB） | C G3/G4 → V D 段列段切片读累加 | ✓ |
| workspace_dkv（贡献槽） | [128, 576] | **bf16**（fp32 兜底 288KB） | 144KB | **23.0MB**（兜底 46MB） | C G5/G6 整 tile 写 → V D 逐行微读 | ✓ |
| workspace_dkv_final（终态原子区） | [S, 1, 576] 全局 | fp32 | — | 4.5MB | V T.tile.atomic_add → k3（launch 顺序） | ✗（核内无读 → 命名含 workspace 亦安全，rev2 §8.4 口径） |
| **GM 合计** | | | | **~95MB**（fp32 dq 档；bf16 dq 档 ~83MB；L5+L4 双 fp32 兜底 ~130MB） | | |

- **【§6.5 勘误（块 C 修正）】**：块 B §6.5 "贡献槽 ≈5.9MB" 算术有误（[128,576] bf16 = 144KB 非 36KB，且 NI=4 驻留未乘）——正确值 23.0MB，散射相关常驻工作集修正为 **~61MB**（贡献槽 23 + ws_dkv 终态 4.5 + KV 2.25 + indices 4 + kv 槽 23 + 小槽族 ~4）；全量工作集 ~95-130MB ≪ L2 192MiB（49-68%）——**L2 吸收结论不变但余量收窄**，§6.5 谨慎口径（单核 MTE 实测核算为准、L2 吸收仅交叉验证）在 rev3 更重要，§13 沿用。决策不受影响（吸收论证仅要求工作集 < L2，非贴线依赖）。
- **命名握手清单（AUTO_CV_SYNC 配对触发，§3.6-3）**：`workspace_kv` / `workspace_s` / `workspace_dp` / `workspace_pd` / `workspace_dq` / `workspace_dkv` 六族命名含 "workspace" 前缀 → FetchWorkspaceName 按名配对自动插 flag（indexer_bwd L138-141 先例）；`pst_p` / `pst_dp` **刻意不用 workspace 前缀**（V-only 单侧，避开配对实现零 flag 二读——iter-2 已验证机制，§5.5）；`workspace_dkv_final` 核内无读不产生配对（安全）。**flag 对数 = 6 配对族 × ≤2（方向或双 AIV 记账）= ≤12 对 ≤16**（§3.6-5 口径，与 §4.3/§3.8 一致；逐族记账表 §9.3）。
- **PARITY 内 NI 顺序覆写安全**（槽位安全 §4.3 论证的重申）：kv 槽(t,i) 覆写者 E(t+2,i) 晚于第二读者 C(t,i)（C 领先 V 1 token ⇒ C(t,i) ≈ V D(t−1,i) 时刻 < V E(t+2,i)）；贡献槽(t,i) 覆写者 C(t+2,i) 晚于读者 V D(t,i)（D(t,i) < D(t+2,·) < C(t+2,i)）——全部由 pair 序读后写传递性在结构层独占推导，AUTO 生成 flag 序承载（indexer_bwd L505-511 注释同款推理）。
- host 侧分配：`torch.zeros` 显式参数化（3.6-3 auto workspace 不支持动态 grid——S 运行时维在终态区 [S,1,576]，per-core 槽区尺寸编译期常量）；`_CachedLauncher` marshal 缓存（indexer_bwd L170 先例）。

### 8.5 内存搬运路径（完整层级链，rev3 口径）

```text
GM[Q/dO] ──T.copy──> L1(q_l1/do_l1, token 奇偶双槽)                    # 直连，每 token 一次（交错段两槽交替）
GM[Indices] ──T.copy──> UB(idx_ub) ──compare/clamp──> UB(mask_ub/rows_ub)
GM[KV] ──T.copy──> UB(kv_gather_staging, batch32) ──T.copy──> GM[workspace_kv] ──T.copy──> L1(kv_l1, 单槽双读)
L0C(acc_s/acc_dq 段) ──T.copy──> GM[workspace_s/dp/dq]（produce/consume 段各自 flush）
GM[workspace_s/dp] ──T.copy──> UB(s_ub/dpraw_ub) ──P 链/Δ 前缀──> GM[pst_p/pst_dp]（V-only）+ UB(p_bf16_ub) ──> GM[workspace_pd P 半]
GM[pst_p/pst_dp] ──T.copy──> UB(p_ub/dpraw_ub2) ──dP 链──> UB(dp_bf16_ub) ──> GM[workspace_pd dP 半]
GM[workspace_pd] ──T.copy──> L1(p_l1/dp_l1)
GM[workspace_dq] ──T.copy──> UB(dq_seg_ub, 列段切片) ──T.tile.add──> UB(acc_dq_ub)
L0C(acc_dkv) ──整 tile T.copy──> GM[workspace_dkv 贡献槽 bf16] ──行微读──> UB(row_ub) ──cast──> T.tile.atomic_add ──> GM[workspace_dkv_final fp32]
UB(acc_dq_ub) ──cast──> GM[dq bf16]（task 尾一次，[64,576] 半片 ×2 AIV 拼合）
GM[workspace_dkv_final] ──k3──> UB ──cast──> GM[dkv bf16]（launch 顺序串接）
GM[Lse/Delta/TokenIndices/Offsets] ──GM 标量读──> 标量 s_i/bos/lse/Δ（§10.3）
```

跨级合法性审计（AGENTS.md 原则 3）：无 GM↔L0 直搬（L0C→GM 为 T.copy 文档路径，indexer_bwd L691 staging 先例）；bf16 UB→L1 不直连（一律经 GM workspace，3.6-1）；workspace_pd/s/dp 的 fp32 C→V 中转经 GM（L0C→UB 无直接路径时同样 GM 中转，3.6-2 的 GM 口径已入 §13 预算）。

### 8.6 跨任务循环生命周期（UB 规划器约束 3.6-6 的规避口径声明）

**设计口径：全部 UB 槽"持续驻留、不跨 symbolic 任务循环回收"**——上表三组槽（持续驻留 / E 段 / D 段）的 liveness 均覆盖块循环全程（每块活跃），不依赖"pass-1 槽死后被 pass-2 复用"的跨循环回收（rev2 证伪形态 #1-3 的触发条件，3.6-6）。规划器仅需处理**段内时序分时**（gather staging 36KB 与 score 槽 40KB 的段内先后——语句级 liveness，标准能力）与 E/D 并存组的并列分配。验证协议：Stage 2 编译期检查 UB 分配日志（`available: 0` 签名 = 触发削减链 §8.3）；BN=64 档为结构性回退（全槽减半，不依赖分时）。**Expert fallback 路径**：显式 `T.alloc_ub` 手动放置 + 生命周期由 flag 序独占推导（indexer_bwd note 2 同款论证），规划器完全旁路（§8.7）。

### 8.7 Expert fallback 手动放置预案（仅 §4.2 升级判据触发时启用；API 逐条查证 2026-10-15）

**API 查证（tilelang/language 源码实查，任务书硬性约束）**：`T.alloc_L1(shape, dtype)` = `T.alloc_buffer(scope="shared.l1")`（allocate.py L140）；`T.alloc_ub` = scope "shared.ub"（L156）；`T.alloc_L0A/L0B/L0C` = scope "wmma.matrix_a / matrix_b / accumulator"（L144-152）；`T.set_flag(src, dst, eventId)` / `T.wait_flag(src, dst, eventId)`——src/dst 为管道枚举 `_pipe = Literal["fix","mte1","mte2","mte3","m","v","s"]`（ascend.py L9/L233/L251）；`T.barrier_all()`（ascend.py L275）。**全部为 ascendc 侧实现，pto 侧可用性见 §9.7 查证结论**。

**手动放置映射**（槽布局不变，仅分配方式替换——§4.2 升级边界"仅同步机制替换，不改数据流"的落地）：

| 主路径（Developer） | fallback（Expert 显式） | 放置 |
|---------------------|------------------------|------|
| `T.alloc_shared`（编译器映射 L1/UB） | `T.alloc_L1` ×（q_l1×2/do_l1×2/kv_l1/dp_l1/p_l1）+ `T.alloc_ub` ×（§8.3 全表槽） | 布局与 §8.2/§8.3 逐槽一致（手动放置只是显式化，尺寸不变） |
| `T.alloc_fragment` 逻辑累加器 | `T.alloc_L0C` ×（acc_s/acc_dq/acc_dq_tail/acc_dkv/acc_dkv_tail） | §8.1 分时表转为手动时分（flag 序保证独占段边界） |
| workspace 命名自动握手 | 手动 `T.set_flag/wait_flag` 逐对（§9.7 预算表） | 6 槽族 ≤12 对逐对记账 |
| 4-True pass_configs | 关闭 `TL_ASCEND_AUTO_CV_SYNC` 单项（§9.7 查证修正：PassConfigKey 无 "AUTO_CV_COMPILE"——§4.2 该名系笔误，实际 4-True 集 = CV_COMBINE/CV_SYNC/AUTO_SYNC/MEMORY_PLANNING，indexer_bwd L275-279 为准） | 其余三项保留（CV 指令分离 / 核内自动同步 / liveness 复用仍有效） |

**升级不改动的部分**（Stage 2 结构按"两条路径可切换"组织的边界）：接口签名、槽布局（§8.4）、GEMM 编排（§5.4）、OOB 防护（§7）、k3 形态（§11.2）——切换点收敛在 k2 的 transit 段（alloc 方式 + flag 机制），以条件编译或双函数分支承载。



## 9. 同步与流水（rev3：pair-loop 软件流水 + AUTO_CV_SYNC 生成序预期 + Expert fallback flag 预算）

> 主路径 Developer 形态：**零 `T.Scope` / 零手动 `T.set_flag/T.wait_flag/T.barrier_all`**——全部跨核同步由 AUTO_CV_SYNC 按槽族命名自动生成的 flag 序承载（§4.2 决策）。本节写：该生成的**编译行为预期**（9.1-9.3，作为 Stage 2/3 的验证基线）、pair-loop **时序图与槽位安全完整论证**（9.4-9.5，§4.3 引用落点）、语义同步点清单（9.6）、Expert fallback 的 flag 预算表与升级判据（9.7）。

### 9.1 Developer 主路径同步边界（4-True pass_configs）

`TL_ASCEND_AUTO_CV_COMBINE`（C/V 指令分离到 AIC/2 AIV）+ `TL_ASCEND_AUTO_CV_SYNC`（workspace 命名配对自动插 flag）+ `TL_ASCEND_AUTO_SYNC`（核内管道自动同步）+ `TL_ASCEND_MEMORY_PLANNING`（UB/L0C liveness 复用，§8 贴线关键）——与 indexer_bwd L275-279 逐项一致（rev3 最直接先例，70.9us 实证该 4-True 集承载 pair-loop 结构）。**查证修正（2026-10-15 块 C）**：PassConfigKey 无 `TL_ASCEND_AUTO_CV_COMPILE` 项（tilelang ir/pass 配置源码核对 + indexer_bwd 实配为准）——§4.2/文首提及的 "AUTO_CV_COMPILE" 为名称笔误，其对 pair-loop 结构编译出软件流水的实际机制载体即 4-True 中的 AUTO_CV_SYNC + AUTO_CV_COMBINE；决策语义不变（AUTO 栈对 pair-loop 结构生成 C 领先流水），本节起统一以 4-True 口径表述。

### 9.2 AUTO_CV_SYNC 编译行为预期（Stage 2/3 验证基线）

对 pair-loop 结构（§5.4 if-guard 错位形态），AUTO_CV_SYNC 的生成序预期（indexer_bwd docstring L44-60 描述的同款生成行为，rev3 槽族映射）：

1. **按名配对**：六族 workspace 槽（§8.4 清单）的 GM 写/读语句按 "workspace" 名串配对，每配对生成 set（写侧管道）→ wait（读侧管道）flag 对；P 半/dP 半同槽两写点生成两 set（同 flag 不同时隙，由写侧程序序串行化）。
2. **相邻 [set;wait] 排空对**（3.6-11 约束遵守）：AIV 侧 set 插在源管道（MTE2/MTE3/V）指令流，生成器保证 set 与被消费 wait 相邻排空——无悬空/预置 set（管道停转风险 (a)(b) 规避）。
3. **MIX_AIC_1_2 双 AIV set**（3.6-12）：V→C 方向槽族（kv、pd）的两 AIV 子块均对每 flag 贡献 set（indexer_bwd note 10 "both AIVs write every cross-core workspace"）——两 AIV 半片/半区分工（§8.3/§8.4）保证覆盖，无单 AIV 缺席死锁。
4. **软件流水**：对 `for p in T.serial(NI·2+1)` + if-guard 错位结构，生成器按 pair 粒度发射 flag 序——C 的 produce(p) 写完成 flag 先行，V 的 consume(p) wait 后置，C 的 consume(p−1) 与 V 的 produce(p) 重叠——**C 领先 V 一个 pair 的深度流水是结构 + 生成序的复合行为**（indexer_bwd 运行实证：重构前 lockstep 92% → pair-loop 重构后消除）。
5. **验证基线**：Stage 2 冒烟（单 token）+ Stage 3 msprof 对照——AIC 等 flag 占比预期 <15%（健康流水）；>30% 触发 Expert fallback 升级判据（§9.7）。

### 9.3 槽族命名握手清单与 flag 对数记账（≤12 对 ≤16，3.6-5）

| # | 槽族（命名） | 方向 | 写→读 | 预期 flag 对数（保守 ×2） | 粒度 |
|---|-------------|------|-------|--------------------------|------|
| 1 | workspace_kv | V→C | V gather（两 AIV 半区行）→ C produce + consume 2 读 | 2 | 每块 |
| 2 | workspace_s | C→V | C G1 → V E 段半片 | 2 | 每块 |
| 3 | workspace_dp | C→V | C G2 → V E 段半片 | 2 | 每块 |
| 4 | workspace_pd（P 半 + dP 半） | V→C | V E/D 段 → C consume | 2 | 每块 |
| 5 | workspace_dq | C→V | C G3/G4 → V D 段列段读累加 | 2 | 每块 |
| 6 | workspace_dkv（贡献槽） | C→V | C G5/G6 flush → V D 逐行微读 | 2 | 每块 |
| — | pst_p / pst_dp | V→V（单侧） | 程序序（零 flag，iter-2 机制） | 0 | 每块 |
| — | workspace_dkv_final | V→k3（launch 顺序） | 无核内配对 | 0 | 每 task |
| — | dq 终写 | V→核外 | 无核内读 | 0 | 每 task |
| | **合计** | | | **≤12 对 ≤16** ✓ | |

记账口径：每族按 ≤2 对保守计（方向 × 双 AIV 记账上限），实际生成序可能合并（同 flag 复用时隙）——Stage 2 以编译产物 flag 计数实测校准（预期 6-10 对），>16 即设计违规回查槽族命名。

### 9.4 pair-loop produce/consume 时序图（完整版；§4.3 论证落点）

```text
── token t ──────────────────────────── token t+1 ──────────────►
C:  P(t,0) P(t,1) P(t,2) P(t,3) │ C(t,0) P(t+1,0) C(t,1) P(t+1,1) C(t,2) P(t+1,2) C(t,3) P(t+1,3) │ C(t+1,0) ...
V:  E(t,0) E(t,1) E(t,2) E(t,3) [Δ(t) 定格] D(t,0)‖E(t+1,0) D(t,1)‖E(t+1,1) ... D(t,3) →散射/终写(t)
     ▲ 前置 drain（首 token）：C 仅 produce，V 仅 E 段        ▲ 收尾 drain（末 token）：C 仅 consume
符号：P=produce 段（C：kv 装载→G1→ws_s 写→G2→ws_dp 写；V：idx/mask→gather→ws_kv 写）
      E=V 消费 produce（ws_s/dp 配对读→P 链→Δ 前缀→pst 写→pd P 半写）
      C=consume 段（C：kv 重装→pd 读→G3/G4→ws_dq 写→G5 pair→G6→贡献槽 flush）
      D=V 消费段（pst 二读→dP 链→pd dP 半写→ws_dq 读累加→贡献槽逐行散射）
```

**Δ 屏障的流水表达（amend-1 语义硬约束：Δ 定格先于任何 dP 消费 GEMM 发射，§4.3）**：V 的 D(t,0) 启动条件 = E(t,0..3) 全完成（Δ(t) 定格）；C 的 C(t,0)（consume GEMM）启动条件 = pd 槽(t,0) dP 半就绪（V D(t,0) 写）⇒ 传递保证 Δ(t) 已定格。**C 在等 Δ(t) 期间不空转**：稳态下 C(t,0) 前的时隙由 P(t+1,0) 填充（交错序），屏障代价 = consume 段延后一个 token 深度（DEP=1），无整段空等（rev2 锁步形态的消除机制）。

**槽位安全完整论证（WAR/RAW 由 pair 序读后写传递性独占推导；indexer_bwd L505-511 同款）**：

- kv 槽（parity p= t%2）：写者 E(t,i)；读者 P(t,i)（produce 读）与 C(t,i)（consume 重读）；下任写者 E(t+2,i)。E(t+2,i) 前置为 V 序 D(t+1,·)（§5.4 V 流交错），而 C(t,i) 完成时刻 ≈ V D(t−1,i)（C 领先 1 token）< D(t+1,·) < E(t+2,i) ⇒ 第二次读完成先于覆写 ✓（RAW/WAR 均传递性闭合）。
- 贡献槽（parity p）：写者 C(t,i) flush；读者 D(t,i)；下任写者 C(t+2,i)。C(t+2,i) 前置为 pd 槽(t+2,i) 就绪（V D(t+2,i) 写出）> V D(t,i)（V 序）⇒ 读先于覆写 ✓。
- s/dp/pd/dq 槽：单写单读 + parity 双缓冲，同 parity 下任写者隔 2 token，传递性同上闭合。
- **全部冲突在源码结构层推导完毕，无需额外同步**；AUTO_CV_SYNC 生成序按槽族承载上述读写序（indexer_bwd 运行实证该复合行为正确）。Expert fallback 升级时本论证转为手写 flag 序的显式表达（§9.7 表逐对对应）。

### 9.5 双缓冲方案（PARITY=2 槽位生命周期）

六族跨核槽 × token 奇偶双槽（§8.4 形态）+ V-only pst 双槽 + L1 q/do 双槽 + UB E/D 段并存组——**双缓冲深度 = DEP = 1 token**，与 §4.3 结构一致（3:3 设想的 3 组槽位 + flag 翻倍被证伪 ③#6/#2；DEP=1 的 2 组在 flag ≤16 与 GM ~95MB 内闭合）。生命周期表：

| 槽族 | 写者到达 | 读者完成 | 槽可覆写窗口 |
|------|---------|---------|-------------|
| parity(t%2) 全族 | token t 各段 | token t 各段（C 领先 V 1 token ⇒ 写读交叠 1 token） | 读者完成后至 E/C(t+2,·) 前任一时刻 |
| pst（同 parity） | V E(t,·) | V D(t,·) | V 序内（程序序保证） |
| L1 q/do 双槽 | 每 token 装载 | 该 token 末 GEMM | 下一同奇偶 token 装载前 |

**加深余量（Stage 3 可选项，非首版）**：PARITY=3（DEP=2）可再吸 Δ 屏障尾延——代价：GM 槽族 ×1.5（~143MB 逼近 L2 75%）、flag 对数不变（槽族数不变）；仅在 msprof 显示 Δ 屏障残余等待 >10% 时评估。首版锁定 PARITY=2。

### 9.6 同步点清单（语义点位；flag 由 AUTO_CV_SYNC 生成，9.3 表为其记账）

| # | 点位 | 写者→读者 | 语义 |
|---|------|----------|------|
| 1 | ws_kv 就绪 | V gather → C kv_l1 装载 | kv 块可 GEMM（produce/consume 两读点各一次配对） |
| 2 | ws_s/dp 就绪 | C G1/G2 → V E 段配对读 | score/dP_raw 半片可入 P/Δ 链 |
| 3 | pd P 半就绪 | V E 段 → C consume | G3-G6 的 P 操作数 |
| 4 | pd dP 半就绪 | V D 段 → C consume | **Δ 定格的下游传递点**（amend-1 屏障的结构表达） |
| 5 | ws_dq 就绪 | C G3/G4 → V D 段 | 贡献块列段可累加 acc_dq_ub |
| 6 | 贡献槽就绪 | C G5/G6 flush → V D 逐行 | dkv 行可散射 |
| 7 | Δ 定格 | V 内（E→D 段界） | 程序序（V 单侧），非 flag 点 |
| 8 | dq 终写 / k3 读终态 | V→核外 / launch 顺序 | 非 flag 点 |

挂死排查承接（rev2 §9.5 HISA 教训）：症状挂死 → 核对 9.3 表槽族命名前缀（workspace 六族 / pst 两族非前缀）、写读方全覆盖、双 AIV 半区分工完整；fwd §8.6 同款排查路径。pto 后端 AUTO_CV_SYNC 行为以 rev2 本算子 pto 全量验证背书（28.5ms 双后端通过）；pto 配对异常单独记录不阻塞 ascendc 主路径（§4.6）。

### 9.7 Expert fallback：升级判据、flag 预算逐对记账、pass_configs 与 pto 可用性

**升级判据（§4.2 唯一触发条件，量化重申）**：pair-loop 形态实现完成后复测 msprof，**AIC 等 flag 时间占比 > 30%**（rev2 锁步 73% 为病态参照、健康 <15% 预期、30% 为升级阈值）。诊断分流先行：排除 DEP 退化（produce/consume 被编译器按序锁死——对照 §6.6 性能触发判据）与槽族命名漏配（§9.3 记账）后再升级。

**flag 预算表（手动形态逐对记账，≤16 硬边界；eventId 0-15 分配）**：

| eventId | 槽族/点位 | 方向 | set 侧 | wait 侧 | 备注 |
|---------|----------|------|--------|---------|------|
| 0-1 | workspace_kv | V→C | AIV×2（MTE3，半区行写完） | AIC（MTE2，装载前） | 双 AIV 均 set（3.6-12）；produce/consume 两读点复用同对（时隙串行） |
| 2-3 | workspace_s | C→V | AIC（MTE3/fix，ws 写完） | AIV×2（MTE2，配对读前） | C→V 方向 wait 双 AIV |
| 4-5 | workspace_dp | C→V | 同上 | 同上 | |
| 6-7 | workspace_pd | V→C | AIV×2（P 半 E 段 / dP 半 D 段两 set） | AIC（consume GEMM 前） | 两半两时隙同 eventId 复用或拆 6/7 |
| 8-9 | workspace_dq | C→V | AIC | AIV×2（列段读前） | |
| 10-11 | workspace_dkv | C→V | AIC（flush 完） | AIV×2（行微读前） | |
| 12-15 | 预留（L1 装载完成同步 / 段界 barrier 类） | | | | 手动形态新增点位于此扩展 |
| | **合计 ≤12 对（0-11）+ 4 预留** | | | | **≤16 ✓** |

发射位置遵守 3.6-11（相邻 [set;wait] 排空对，AIV set 插源管道 MTE3/V 指令流）；MIX_1_2 双 AIV set 全覆盖（上表 set 侧标注）；核内段界以 `T.barrier_all()`（ascend.py L275，查证）处理跨管道序。

**pass_configs 切换**：关闭 `TL_ASCEND_AUTO_CV_SYNC` 单项（手动 flag 接管配对职能）；`TL_ASCEND_AUTO_CV_COMBINE` / `TL_ASCEND_AUTO_SYNC` / `TL_ASCEND_MEMORY_PLANNING` 三项保留（C/V 指令分离与核内同步仍由 pass 承载；L0C/UB 的 alloc_fragment/alloc_shared 可换 §8.7 显式形态，pass 不冲突）。查证修正：PassConfigKey 无 AUTO_CV_COMPILE（§9.1）——§4.2 "关闭 AUTO_CV_SYNC/AUTO_CV_COMPILE" 的实际落地 = 关闭 AUTO_CV_SYNC 单项。

**pto 侧可用性（§4.6 遗留查证项）**：Expert API（set_flag/wait_flag/barrier_all/alloc_L0*）定义于 tilelang/language（ascend.py/allocate.py，后端无关前端），但 pto codegen 对手动 flag 的端到端先例**仓内无**（rev2 pto 验证走 AUTO 栈；docs/TileLang-Ascend Programming Guide 差异章节未列 pto 手动 flag 支持）——**结论：fallback 若触发，ascendc 后端先行（手动 flag 全量），pto 后端维持主路径形态**（Developer pair-loop + AUTO 栈，正确性同源、性能如实报告，符合双后端约束原文"性能以较优后端考核，另一后端如实报告"）；pto 手动 flag 作为 Stage 3 二线实验项（5 行最小复现 kernel 先验，同 §10.4 协议）。

## 10. 动态 shape 与 varlen 边界（rev3：继承 rev2 §10 + persistent 轮询/T.serial 边界 + Expert fallback 下的 varlen 交互）

> 继承声明：10.1-10.5 继承 rev2 §10 全部结论（S/B_plus_one 符号维划分、TokenIndices host 预计算、段语义进 kernel 不进循环结构、pto GM 标量读 fallback、kv_group>1 支持域），本节按 rev3 pair-loop 结构写增量与交互。

### 10.1 符号维与编译期常量划分（继承 rev2 §10.1）

| 维 | 声明 | 说明 |
|----|------|------|
| S（packed 行数） | `T.symbolic`（indexer_bwd L330 `num_blocks = T.symbolic` 先例） | 运行时任意 ≥1，task 总数 = S×kv_group |
| B_plus_one | `T.symbolic` | ≥2（offsets 长度） |
| S_kv | 不单列符号：host `assert S == S_kv`（主仓 L262 原样）→ bwd/k3 复用 S | 行空间一致 |
| H / D=512 / D_tail / topk / kv_group / BN=128 / NI=4 / DSEG / v_block / PARITY / grid_size | 编译期常量（builder 参数） | 进 `_BUILD_CACHE` 编译键，一次编译覆盖任意 S/B |

### 10.2 host 分派与 TokenIndices（继承 + 两段发射口径）

`prepare_token_indices(offsets)` 原样复用（ref_gpu/index.py L80-82 `tensor_cache` 版本）——与 fwd 共用同一缓存条目（do_bench 稳态零重复构造，§1.5）；host 程序序：contiguity 断言 → `D=512` 硬编码 / `kv_group`、`topk` 提取 → token_indices → **两段发射 k2→k3**（amend-1，delta=None；delta 显式传入同样不发射 k1，§11.3）→ 返回 (dq, dkv)。`_CachedLauncher` + `_BUILD_CACHE` 按 `(H, DQKV, topk, kv_group, target)` 键双份缓存（ascendc/pto，indexer_bwd L170 先例）。

### 10.3 varlen 段语义进 kernel：persistent 轮询 + T.serial 边界（rev3 交互重述）

```text
ntask = T.max(0, T.ceildiv(total − cid, grid))     # persistent striped 轮询（reducesum L261-265 已验证形态）
for task in T.serial(ntask):                        # 运行时边界（动态界禁 T.Pipelined，§3.3）
    tid = cid + task * grid                          # striped：相邻核任务错开 → L2 行聚集
    s_row = tid % S; g = tid // S                    # 行优先分解（rev2 §10.3 同款）
    b_i, s_i = TokenIndices[s_row, 0], TokenIndices[s_row, 1]    # GM 标量读
    bos, eos = Offsets[b_i], Offsets[b_i+1]; seg_len = eos − bos
    for p in T.serial(NI·2 + 1):                     # pair-loop 内层（编译期常量界，§5.4 if-guard）
        if p < NI:   produce(t, p)                   #   C: G1/G2 + transit 写；V: gather/P 链/Δ 前缀
        if p >= 1:   consume(t, p−1)                 #   C: G3-G6；V: dP 链/散射/累加
    # 跨 token 交错的 produce(t+1,·) 在 task 序列的相邻 task 间自然形成（persistent 流，§9.4 时序）
```

- **段边界不进循环结构**：mask 上界 `s_i` / clamp 上界 `seg_len−1` 均运行时标量，经 `T.tile.compare/min/max` UB 广播参与向量链（§7.2/§7.6 口径继承）；动态循环边界仅外层 `T.serial(ntask)` 一处，pair-loop 内层全为编译期常量（§3.3 ✓）。
- **task 独立性**：每 task 的 bos/eos/seg_len 独立取值，段间无状态残留——acc_dq_ub 每 task 头部清零；Δ(t) 的 task 内闭合（E 段 NI 块前缀累加 → 定格 → D 段消费，不跨 task）。
- **varlen 与 pair-loop 交互**：token 粒度 task = pair-loop 的自然单元（DEP=1 token 恰为 task 间深度，§5.4）；段首/段尾 token 的 produce/consume drain 由 if-guard 天然吸收（前置 drain = 每 task 的 p<NI 段，收尾 drain = p≥NI 段——**per-task 闭合**，不跨 task 残留 flag 时隙）。

### 10.4 PTO GM 标量读风险与 fallback（继承 rev2 §10.4）

pto 的 GM 标量读（codegen_ascend_pto.cc L802-812）已实现但无端到端 varlen 先例；rev2 已在 pto 后端全量验证本算子（28.5ms）——**rev2 验证已覆盖该风险**（TokenIndices/Offsets 标量读在 pto 通过），rev3 数据流不变此路径，风险降级为"实现回归项"。fallback 预案保留：host 预展开 RowMeta 表 [S,4]（b_i, s_i, bos, eos）单表读。

### 10.5 kv_group > 1 通用路径（继承 rev2 §10.5）

task=(s_row, g) 分解与 KV/Indices 的 g 维寻址已参数化；`H_kv < 16` 时 `padded_H=16` 兜底；padding head 行贡献归零处理 Stage 2 逐行核对——考核 shape kv_group=1 无 pad（§3.5/§14 残余风险记录，不展开设计）。

### 10.6 Expert fallback 下的 varlen 边界（rev3 增量：手动 flag 与动态 ntask 的交互）

手动 flag 序在 `T.serial(ntask)` 运行时边界内的配对安全性论证：**flag 序模板按 task 内语句固定**（每 task 的 produce/consume 段 flag 序完全同构——eventId 复用同对，set/wait 在 task 内相邻排空，3.6-11 相邻对约束在 task 边界处自然满足：task 尾 flag 全排空后进入下一 task，无跨 task 悬空 set）；token 数（ntask）运行时变化不改变每 task 的 flag 序形状 ⇒ **配对安全性与 ntask 无关**。两处 varlen 特有核对点：① 段尾 token 的 mask 全 false 块（t < topk 尾部填充）——produce 段仍完整执行（GEMM 形状不变，贡献由 mask 归零），flag 序不缺拍（AUTO 生成序与手写序均按语句而非数据发射）；② 挂死排查时优先核对段尾 task（mask 全 false 的 flag 时隙完整性——若手写序以数据条件短路了 set，即触发 3.6-12 双 AIV 缺席死锁，禁止数据条件短路 flag 发射）。

## 11. preprocess / postprocess 与 fwd 依赖（rev3：k1 折叠处置 + k3 行宽 32 残余优化 + fwd 已交付状态更新）

### 11.1 preprocess（k1）：amend-1 折叠处置与 delta 显式路径

**k1 不再发射**（amend-1，2026-10-14）：`delta=None` 时 Δ=Σ_j P·dP_raw 归约折叠入 k2 的 V 侧 E 段（fp32 先于 cast 量，§5.4/§9.4 Δ 前缀累加器）——两段发射口径（§11.3）。**k1 kernel 代码保留为对照诊断 fallback 不删**（§1.2 口径）：诊断模式（delta 显式传入）下可独立发射 k1 与恒等式路径对照，定位 Δ 相关精度问题；AIV-only 平铺行块形态（rev2 §11.1：`T.Kernel(grid=40, threads=1)`、[64,512] o/do_ub、T.tile.mul 升精度 + reduce_sum、~260us 单核口径）原样保留。**delta 显式传入路径**：host 校验 shape/dtype 后直接传 k2（kernel 内 Δ 源切换为 GM 读，E 段跳过前缀累加——`delta_ub` 槽复用为传入值缓存），行为与主仓完全一致（amend-1 第 1 条）。

### 11.2 postprocess（k3）：AIV cast kernel + k3 残余优化

**形态（rev2 实现验证，§4.8 保留项）**：AIV-only persistent `T.Kernel(grid=40, threads=1)` striped 轮询；任务 = [32, 576] 行块（**行宽 32**，rev2 实测验证档——rev2 设计稿的 [64,576] 在实现期收敛为行宽 32 的小行块，2D 连续读写形态稳定）；读 workspace_dkv_final [32,576] fp32（2304B/行连续）→ UB → `T.copy` auto-cast bf16（fwd §4.5 auto-cast 先例）→ 写 dkv。cast 单次性语义锚点（fp32 终态一次转 bf16，§1.3）。

**k3 残余优化评估（任务书块 C 项；数据口径）**：k3 流量 4.5MB 读 + 2.25MB 写，40 AIV × ~26GB/s 行读聚合下 ~10-30us 量级（rev2 §11.2 口径），占 7680us 预算 <0.4%——**优化空间与优先级均低**，残余选项记录如下：

| 选项 | 机制 | 预期收益 | 结论 |
|------|------|---------|------|
| 行块批量 2D 化 | [32,576] 行块 ×4 聚为 [128,576] 单任务 2D 读写 | 事务数 ÷4，~5-10us | 可选（Stage 3 顺手项，非目标项） |
| fold 入 k2 尾声 | bwd 末核就地进行终态 cast | 省一次 launch ~13us + k3 全部 | **否决**：终态区 [S,1,576] 为跨核全局原子区，cast 需全部核的原子完成——k2 内无全核完成检测原语（barrier_all 为核内），强行 fold 需 host 中转等价于两段；launch 顺序语义（§9.6 #8）即为正确机制 |
| k2/k3 launch 重叠 | k3 grid 与 k2 尾部并行发射 | ~10us | **否决**：数据依赖（终态区写完才可读），流内串行 launch 已是次优解外唯一安全形态 |

**k3 的真实约束是语义而非性能**：行宽 32 + 2D 平面 ≤2（3.6-7/3.6-8 边界内）+ 只读界内构造区（§7.1 无 OOB 面）——rev2 实现已验证，rev3 零改动继承。

### 11.3 两段发射与计时口径（amend-1，继承）

host 程序序：`delta is None` → **k2（含 Δ 归约）→ k3** 两次 launch 串行；kernel 间数据依赖（k2 的 workspace_dkv_final 终态 → k3 读）由 launch 顺序保证，无额外同步。`do_bench(fn, rep=100, warmup=250)`、`fn` 调 `sparse_mla_bwd(..., delta=None)` → 两段计时，7680us 目标不变（§1.5/§13）。k1 不在发射路径（§11.1）。

### 11.4 fwd 外部依赖策略（fwd 已交付状态更新）

- **状态更新（rev3，2026-10-15）**：`sparse_mla_fwd.py` 与 `test_sparse_mla_fwd.py` **已落盘**（本目录文件清单核实）——rev2 §11.4 的"fwd 待实现时间窗风险"解除；消费契约不变：test_sparse_mla_bwd.py `from sparse_mla_fwd import sparse_mla_fwd_interface`，`tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, offsets)`。
- **禁止实现/修改 fwd**（任务书 §4 硬约束）；fwd 的 L0/性能状态以 fwd 会话交付记录为准，本算子只消费接口。
- **过渡 fallback 保留为防御性路径**（rev2 §11.4 的 ImportError 分支 + `_torch_fwd_fallback`：o = ref fwd bf16、lse = fp32 全量 logsumexp，与 golden autograd 内部同源）：fwd 文件在位即不触发；若 fwd 会话回退重做导致接口瞬断，bwd 测试仍可独立运行（判据不受影响，rev2 论证继承）。最终验收前确认 fwd import 直连（移除 fallback 的判据：fwd L0 双后端通过——fwd 会话口径）。
- **lse 数值差风险**（rev2 §11.4 末条继承）：fwd kernel 在线 softmax 与全量 logsumexp 的归约序差 δ → bwd 的 P 乘性偏 exp(δ)，由 fwd Lse fp32 门限（fwd DESIGN §9.3）+ bwd 判据（eps=1e-4 + §12 混合容差）双层覆盖；性能验收不依赖 fwd（7680us 口径 fn 只调 bwd，o/lse 预计算输入，§1.5）。

## 12. 精度容差与验证方案（rev3：amend-1/amend-2 分层门限与 L0 9 用例计划原样继承 + rev3 专项精度风险项）

> **继承声明（任务书 §3 硬性要求）**：§12.1-§12.5 与全部测试资产（test_sparse_mla_bwd.py 用例与 golden）**零重写原样继承** rev2 §12（含 amend-1 Δ 恒等式精度路径、amend-2 收紧门限与 Stage 2 场景 B 已扩展的 L1 用例）——rev3 改架构不改精度契约，实现必须直接通过现有套件。§12.6 为 rev3 新增专项（pair-loop 槽复用正确性、bf16 槽档散射门控复验）。proto_sparse_mla_bwd.yaml 接口未变，块 D 校验通过（dtype 全集与 §12.2 一致、attrs 覆盖 §1.4 全部关键属性，无需重写）。

### 12.1 双判据体系（继承 rev2 §12.1，一字不动）

- **主判据**（与主仓逐字一致，§1.5）：`assert_tensors_similar(eps=1e-4)`——全局归一化点积相似度 `sim = 2⟨x,y⟩/(‖x‖²+‖y‖²)`（double 精度），要求 `0 ≤ 1−sim ≤ 1e-4`；dq / dkv 分别判；全零张量特判通过（ref_gpu/utils.py:48-73）。全局能量口径对离群点鲁棒，是主仓语义验收的权威判据。
- **辅判据**（tilelang-op-test-design 精度标准，逐元素定位）：混合容差 `|a−g| ≤ atol + rtol·|g|`，双门限 `matched_ratio ≥ required` 且 `max_abs_error ≤ limit`——主判据 FAIL 时定位误差分布（dkv 散射 / dq 累积 / 特定 head / 长尾行），双后端差异定位同理。
- 两判据均计入 L0 阻塞（任一 FAIL → `[PRECISION_FAIL]`，退出码 1）；辅判据单独 PASS 不豁免主判据。

### 12.2 按 dtype 精度标准表 + 值域轴分层门限（amend-1/amend-2 口径，原样继承）

| 量 | dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|----|-------|------|------|---------------------|------------------------|
| dq（输出） | bfloat16 | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |
| dkv（输出） | bfloat16 | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |

- **dtype 全集说明**：本算子为固定 dtype 组合（§1.4 断言继承），无 dtype 扫描维度——输入 q/kv/o/do `bfloat16`、indices/offsets `int32`、lse/delta `float32`；输出 dq/dkv `bfloat16`。整型无输出量（N/A）；lse/delta fp32 为上游产物，由 fwd Lse 门禁与本算子链覆盖，不在输出判据内。
- **阈值合理性（rev3 路径复核，§6.3 精度账）**：dkv 终值 = L0C fp32 块内 pair 求和 → bf16 贡献槽单次舍入 → V 侧 fp32 GM 原子跨 token 累积（~448 次）→ k3 单次 cast；dq = NI=4 块 UB fp32 累积 + 单次 cast。与 golden 差异由 bf16 终端舍入 + 归约序差主导 → rtol 2⁻⁶ 覆盖；atol 2⁻¹⁰ 保护 dkv 长尾近零行。indexer_bwd 同路径实测 ratio 0.9972 佐证。

**值域轴分层门限（amend-1，2026-10-14；amend-2 收紧，2026-10-14；rev3 原样继承为锚点）**：

| 输入域 | 覆盖用例 | 门限（amend-2） | 依据（debug_log attempt-1 / attempt-2 实测） |
|--------|---------|----------------|-------------------------------------|
| **域 A 主仓口径域**：σ=1（randn 默认）、默认 sm_scale | L0 全部 9 条 + L1 形状轴 11 条 | **维持 §12.1 sim ≥ 1−1e-4 双判据 + 本节 dtype 表，一字不动** | L0 9/9 双后端 PASS（dkv ratio 0.9969-0.9998）；L1 形状轴 11/11 PASS |
| **域 B 值域扩展域**：σ≤10 且任意 sm_scale | L1-smscale（sm_scale=0.1）、L1-valrange-s（σ=0.5）、L1-valrange-asym（σ=3, shift=6）、L1-valrange-m（σ=10） | 主判据校准为 **sim ≥ 0.999**（dq/dkv 分别判）；逐元素混合容差维持**非阻塞参考**（数值照常输出，未达标注 WARN 供定位） | attempt-2 实测（双后端一致）：域内最差 valrange-asym dq sim **0.999907**（门限余量 ~9e-4）；valrange-m sim **0.999996**；smscale 0.999996 / valrange-s 0.999997 同步佐证；大 shift 子域逐元素门限对完美实现不可满足（asym dq ratio 0.3011 实测）⇒ ratio 维持非阻塞 |
| **域 C 高集中度抵消区**：σ≥100 | L1-valrange-l（σ=100） | **整例降 WARN 级参考用例**：双判据数值如实输出、判定记 WARN，不阻塞 `[PRECISION_PASS]` | golden 自身失效域：attempt-2 实测 dq sim **0.4579** / dkv 0.4662，且 fwd lse sanity FAIL → torch_fallback；误差由 golden bf16-P 内部颗粒 + fwd 侧大 σ 域精度边界主导，sim/ratio 判据在该域对任何实现不可满足；σ∈(10,100) 未锚定，后续增用例先按域 B 判据初判、FAIL 即回 Stage 1 校准 |

- 分层不改变 L0 阻塞语义（L0 全部属域 A）；域 C 用例仍执行并如实输出数值（覆盖门禁照常计入），仅判定记 WARN。
- **禁止无证据调门限**：上表每个校准值均锚定 debug_log 具体实验数字；后续重跑若实测优于门限依据，门限只允许收紧、不允许再放宽——收紧后需全量重跑确认。**rev3 增补适用条款**：§12.6(2) 的 bf16 槽档切换属"实现路径变更"而非门限校准——门限数值不动，门控对象是新路径在新门限下的复验。

### 12.3 L0 门槛测试计划（tilelang-op-test-design 场景 A，仅 L0；9 用例原样继承）

**算子类别判定**（场景 A Phase 2，依据 §1.3/§2.2）：计算类型 = 混合（CV 融合）——6 类 GEMM@Cube + mask/exp/dP 链@Vector + fp32 atomic scatter@AIV；复杂度 = Fusion（两段 host 编排 × pair-loop NI 块循环 × 6 类 GEMM）；数学特征 = matmul + exp + scatter → **综合类别 Fusion**。测试聚焦：CV 交互正确性（§9.3 六族槽命名握手 + §9.4 槽位安全链）、**pair-loop 槽复用正确性（§12.6(1)）**、OOB 双端防护（§7）、varlen 段语义（§10.3）、fp32 原子散射精度（§6）。L1（不规则 shape / 短段 / 边界 t）/ L2 / Boundary 不在本节（Stage 2 场景 B 已按 amend-2 扩展值域轴，rev3 直接复用）。

**shape 规则性口径**（"block 整除"的本算子映射）：topk % BN(=128) == 0（断言继承，§1.4）→ L0 全部用例 topk ∈ {128, 512}；S / 段长为运行时动态维（token 粒度任务无块尾，§3.5），L0 覆盖 S ∈ {128, 256, 640, 645, 2048}；H=64 / HKV=1 / DQKV=576 / DV=512 固定（host 硬编码 + 考核 shape）。

**L0 用例表**（9 条；生成流程 = 主仓 test 口径（ref_gpu/sparse_mla_bwd.py L293-320）原样 + seed 固定）：

| ID | 类别 | 配置 (B,S,HKV,topk) | 构造要点 | 覆盖目标 |
|----|------|--------------------|---------|---------|
| L0-1 | 主仓口径 | 1, 2048, 1, 512 | randn + 逐 token `randperm(max(1,t))[:topk]`、尾部填 S、offsets=[0,S]；seed=0 | 全链路考核 shape（t<511 尾部填充自然出现 130,816 槽） |
| L0-2 | 快速冒烟 | 1, 128, 1, 128 | 同上；seed=0 | 小规模 CI 冒烟（S=topk 边界） |
| L0-3 | pto 后端 | 同 L0-1，target="pto" | 同 L0-1 | 双后端（§4.6；ascendc 即 L0-1 本体） |
| L0-4 | delta 传入 | 同 L0-2，delta 显式 | delta = golden preprocess 结果 (O·dO).sum(-1) fp32 | 接口分支（Δ 源切换，§11.1） |
| L0-5 | OOB 强化 | 1, 256, 1, 128 | 高填充率构造（t<127 每 token 尾部 ~50% 填充槽） | gather clamp + scatter 条件跳过（§7.2/§7.3） |
| L0-6 | -1 混入 | 1, 256, 1, 128 | 有效槽后、填充前中段插 -1 | mask `!= -1` 判据 + golden 裁剪列语义（§7.2） |
| L0-7 | 重复索引 | 1, 256, 1, 128 | 前 64 有效值复制两份填满 128 槽（重复计贡献） | 不去重语义锚点（§2.5/§7.2） |
| L0-8 | varlen 多段 | 3, 640, 1, 128 | offsets=[0,128,512,640]（段长 128/384/128） | 段边界 + 段内 mask 语义（§10.3） |
| L0-9 | varlen 尾块 | 2, 645, 1, 128 | offsets=[0,129,645]（段长 129/516，均非 BN 倍数） | 段长非整除 + t=128 段内边界行 |

判据：全部用例双判据（§12.1），dq / dkv 分别判；L0 全绿（`[PRECISION_PASS]` × 9）= Stage 2 解锁后续扩展的门槛。L0-3 pto 侧同等阻塞（双后端硬性要求，§4.6）。**rev3 复跑口径**：L0 全部用例在 rev3 实现（pair-loop 主路径、默认档 L4 fp32 + L5 bf16）上直接复跑，测试代码零改动；L0-1/2/5/8/9 同时充当 §12.6(1) 槽复用正确性的多 task 奇偶覆盖载体（S≥128 ⇒ 每 AIV ≥6 task，两奇偶槽族均多次命中）。

**值域轴门限适用（amend-2 口径继承）**：L0 用例全部属域 A，§12.1/§12.2 判据一字不动。Stage 2 场景 B 已扩展的 L1 值域/缩放轴用例（smscale / valrange-*）按 §12.2 分层门限执行：smscale / valrange-s / valrange-asym / valrange-m 四例均按**域 B 阻塞判据 sim ≥ 0.999** 判，valrange-l（σ=100）维持 WARN 级参考用例。

### 12.4 golden 草案（PyTorch 参考实现，继承；Stage 2 测试文件已落地直接复用）

```python
def ref_sparse_mla_bwd_interface(q, kv, o, do, indices, lse, offsets, sm_scale=None, is_casual=True):
    # 主仓 ref_gpu/sparse_mla_bwd.py L281-290 原样移植（仅 device 参数化，去 "cuda" 硬编码）：
    #   q, kv = q.detach().clone(), kv.detach().clone(); q.requires_grad_(True); kv.requires_grad_(True)
    #   o_ = ref_sparse_mla_fwd_interface(q, kv, indices, offsets, sm_scale, is_casual)
    #   o_.backward(do); return q.grad, kv.grad        # grad dtype 随输入 bf16
    # ref_sparse_mla_fwd_interface（ref_gpu/sparse_mla_fwd.py L214-257 移植）OOB 语义锚点：
    #   indices[indices > sk] = sk → scatter 到 sk+1 宽 mask 末列 → 裁剪 → 上越界贡献 0
    #   -1：scatter 负索引落末列 → 裁剪 → 贡献 0（与 §7 NPU 掩码语义等价，§7.4 论证）
    #   causal：arange(t) >= arange(sk)；首行 j=0 强制有效（L239-243）——L0-6/-1 与 L0-8 段首行依赖此语义
```

- **fwd 消费契约**：`from sparse_mla_fwd import sparse_mla_fwd_interface` 取 `(o, lse)`（§11.4 状态：fwd 已落盘，import 直连）；`_torch_fwd_fallback` 保留为防御性路径（fwd 接口瞬断时判据不受影响，与 golden 内部同源）。
- golden 无 kernel 特化（纯 PyTorch，任意合法输入正确），符合"禁止针对 golden 固定数值特化"约束。

### 12.5 seed 与主仓偏差声明（继承）

- 主仓测试无固定 seed（randn / randperm 每次随机，§1.5）→ L0 全部用例 `torch.manual_seed(0)`。偏差声明：生成流程逐行一致，仅固定随机序列——换取可复现性，无语义偏差。
- **失败复测协议**：L0 FAIL 先以 seed∈{1,2} 复跑排除环境噪声（多会话共享 NPU）；仍 FAIL → 双判据定位（§12.1）→ `T.dump_tensor` 分段 dump（Δ 前缀 / P/dP / 贡献槽 / 终态区）定位首错段；**若 FAIL 模式呈"偶发/按槽位奇偶聚类"，优先按 §12.6(1) 槽复用竞态排查（rev3 新增首选分流）**。

### 12.6 rev3 专项精度风险项（新增：pair-loop 槽复用正确性 + bf16 槽档散射门控复验）

**(1) pair-loop 双缓冲的槽复用正确性（§8.4/§9.4/§9.5 的精度面）**

- **风险机理**：pair-loop 的全部正确性依赖"槽位安全 = WAR/RAW 由 pair 序读后写传递性在结构层独占推导"（§9.4）。若 AUTO_CV_SYNC 生成序与结构推导不符（生成器缺陷或槽族命名漏配），故障面是**陈旧读 / 部分覆写**——表现为**精度错误而非挂死**（挂死是 flag 缺失的另一种表现，§9.6 排查路径；本项专治"不挂死但算错"）。canonical 症状：dkv 局部行错（贡献槽覆写过早：C(t+2,i) 覆写未读槽）、dq 整 token 错（ws_dq 同理）、错误随 task 奇偶呈 2-周期聚类（PARITY=2 槽位）。
- **验证协议（Stage 2 强制项）**：① L0 seed∈{0,1,2} 三跑——同 seed 输出逐位一致（确定性验证）+ 跨 seed 全绿；② 单 token debug shape（B=1, S=1, topk=128）跑通（drain 路径：前置/收尾 drain 各一次，if-guard 两分支全覆盖）；③ 复现时 dump 点按"槽族 tslot 读写序"取证：workspace_kv/贡献槽的 (t mod 2) 槽在覆写前是否已读毕——对照 §9.4 传递性链逐环定位；④ 修复优先序：核对 §9.3 命名清单（六族 workspace 前缀 / pst 两族刻意非前缀）→ 程序序断言（produce 写出语句位置）→ 仍复现则按 §4.2 判据链升级 Expert fallback（手动 flag 显式化读写屏障，§9.7 表逐对对应）。
- **边界声明**：本项不改变 §12.1-§12.5 门限——是"实现正确性验证协议"而非新判据；L0 全绿即本项通过的第一层证据（多 task 奇偶覆盖见 §12.3 复跑口径）。

**(2) bf16 槽档散射的精度门控复验要求（L5 贡献槽 bf16 §6.3 + L4 dq 优化档 §5.3）**

- **默认档与门控对象**：rev3 默认实现 = **L4 fp32（dq 贡献块，与 rev2 精度路径一致）+ L5 bf16（dkv 贡献槽，新增出 L0C 单次 bf16 舍入）**；L4 bf16 为进一步优化档（再增一重舍入）。L5 bf16 的精度账（§6.3）：每 (token, row) 贡献单次 2⁻⁸ 相对舍入，跨 token ~448 次 fp32 GM 原子累积 → 累积相对误差 ~0.02% 量级，**预期过门限但必须实测复验**。
- **门控复验协议（隔离变量，Stage 2 实施顺序约束）**：① 首版以"L5 fp32 兜底档 + L4 fp32"跑 L0 9 例（精度路径与 rev2 逐位一致，门禁必过——建立正确性基线）；② 切 L5 bf16 档全量复跑 L0 9 例 + 域 B 四例（smscale / valrange-s / valrange-asym / valrange-m）——**dkv sim ≥ 0.999 阻塞判据（amend-2 收紧口径）为准，dq 同判**；③ L4 bf16 优化档仅在 ② 全绿后单独评估（再跑域 B 四例 + 域 A 抽样）。任一 FAIL → 按档位回退（L5 → fp32 兜底 / L4 → fp32 基准），性能影响如实披露（§6.6/§13）。
- **判读细则**：域 B FAIL 但域 A 全绿 → 与 §6.3 精度账预期不符（域 B σ=10 放大效应），先排除实现 bug（dump 贡献槽写出值 vs L0C 原值差 = 舍入量）再定档；域 A 即 FAIL → 实现错误（非档位问题），走 §12.5 复测协议。**门限数值全程不动（§12.2 禁止条款）**——门控是路径复验，不是门限校准。

**(3) Expert fallback 升级的精度等价性（§4.2/§9.7 的精度面）**

升级（手动 flag 替换自动配对）不改变数值路径（同数据流、同算子序、同 dtype 档位）→ 精度预期与主路径**逐位一致**。验证：升级后 L0 全量重跑 + 与主路径输出 diff（预期逐位一致；不一致即 flag 序错误造成陈旧读——按 §12.6(1) ③ 取证）。pto 后端维持主路径形态（§9.7），其精度判据独立执行（L0-3）。

## 13. 性能预算（rev3：新数据流逐腿读量表 + 算力/带宽下限 + pair-loop 流水预期 + 7680us 可行性 + rev2 实测 28541us 对照）

### 13.1 AIV 逐腿读量表（62-70MB 口径；L5 fp32 兜底档 88-96MB）与 C 侧流量表

**AIV 侧逐腿读量（全序列 S=2048，per-AIV 口径；腿定义继承 §4.5，本表补事务形态 / 适用带宽锚点 / 时间估算三列——"62-70MB"为 §4.5 合计行的块 D 复核口径，腿内数字原样继承）**：

| 腿 | 内容 | 读量/AIV | 事务形态 | 适用带宽锚点（实测） | 时间估算 |
|----|------|---------|---------|---------------------|---------|
| L1 | KV gather 行读 | 字节量 30MB（时间口径折算 ~15MB） | 1152B 散行读 + batch32 整 2D 写出 | 散行 ~12.2（rev2 微事务）/ 批量 2D ×3.6（fwd probe2）；fwd V mte2 20.5 | 1.5-2.5ms |
| L2 | ws_s/dp 配对读 + pst 零 flag 二读（fp32 双腿） | 12.8MB | [32,128] 半片批量行 | 20.5（fwd V mte2 批量） | ~0.65ms |
| L4 | ws_dq 贡献块读累加（半片列段切片） | 14.4MB fp32 基准 / 7.2MB bf16 门控档 | [32,144] 列段 2D | 20.5 | 0.70 / 0.35ms |
| L5 | ws_dkv 贡献槽逐行微读（散射源） | 26.4MB bf16 档 / 53MB fp32 兜底 | 1152B 逐行（段微读合并）——**无法批量 2D**（随机行，§6.2） | 12.2（微事务饱和值） | 2.17 / 4.34ms |
| L6 | idx/lse/delta 标量族 | ~1MB | GM 标量 | 低效口径 | ~0.1ms |
| **合计（MTE2 管道）** | | **62-70MB**（fp32 L4 基准）/ 55-63MB（双 bf16 门控档） | | | **5.0-6.5ms**（基准档）|
| 合计（L5 fp32 兜底档） | | **88-96MB** | | | **7.2-8.7ms ⚠️ MTE2 单管道超 7.565ms 预算**——兜底档大概率超 7680us，如实披露（§6.6/§14） |

**AIV 侧 MTE3（写管道，与 MTE2 异管道可重叠）**：原子终态写 52.9MB（917,760 有效行 × 2304B fp32 ÷ 40——**物理下限量，与档位无关**）+ ws_kv 写 28.8MB + pst/pd/dq 终写 ~17MB ≈ **~99MB @ 15.8-18GB/s（iter-1/2 实测写口径）≈ 5.5-6.3ms**——与 MTE2 行内流水重叠后 V 链见 §13.2。

**C 侧流量（per-AIC，全序列 102.4 task/核）**：

| 管道 | 流量构成 | 字节/AIC | 锚点 | 时间 |
|------|---------|---------|------|------|
| C MTE2（读） | kv 2× bulk 重读 115.2MB（1.125MB/task，L1 512KB 容量强制：topk×576×2B=576KB > L1，§8.2）+ Q/dO 13.6-20.8 + ws_pd 12.8 | **141.6-148.9MB** | 15-19.3GB/s（fwd C mte2，strided 256B 窄行口径）/ 50-60GB/s（reducesum AIC ws→L1 bulk 连续口径） | 2.4-9.9ms（口径跨度即主对账项，§13.7） |
| C fixpipe（写） | ws_s/dp 0.25 + ws_dq 0.5625(fp32 档) + 贡献槽 0.5625(bf16 档) = 1.375MB/task | ~141MB | 49.9GB/s（fwd 实测） | ~2.8ms ✓ |
| Cube（GEMM） | 40 发/task × 102.4 | 369.4 GFLOP | 91 TFLOPS（同族实测） | ~4.1ms（§13.3） |

- **【块 D 口径勘误】**：§5.5 尾行 "C 侧 bulk 总读 ≈1.45MB/task（≈297MB/AIC）" 中 297 为**读+写合计粗口径**（读 148.9 + fixpipe 写 ~143 ≈ 292）——本表按管道分列修正：MTE2 读 141.6-148.9MB / fixpipe 写 ~141MB，两管道独立对账（MTE2 与 Cube/fixpipe 异管道重叠，不构成叠加约束）；§5.4 引用的 "~297MB @ ~17GB/s" 相应理解为合计流量粗算，**决策语义不变**。
- **工作集与 L2**：全量工作集 ~95MB（fp32 dq 档，§8.4 勘误口径）< L2 192MiB（49%）——C 侧全部读源（ws_kv/pd/Q/dO 首读后驻 L2）与 AIV 贡献槽读均为 L2 驻留连续 bulk，倾向 reducesum 50-60GB/s 口径；HBM 仅见首读 ~292MB + 终态写。fwd 15-19.3 锚点为 strided 窄行形态（176KB/块 @ ~15GB/s，fwd iter-5 msprof），**非 rev3 事务形态**——两锚点差 3-4× 即 §13.7 首要对账项。

### 13.2 7680us 预算分解与账面推演

| 段 | 预算 | 构成依据 |
|----|------|---------|
| k2 bwd（含 Δ 归约，pair-loop 主路径） | ≤ 7565 us | 账面推演见下 |
| k3 postprocess | ≤ 80 us | 6.75MB ÷ 1.04TB/s = 6.5us 下限，launch 开销主导（§11.2，继承） |
| host 编排 + 2×launch | ~ 35 us | 2×`_CachedLauncher` marshal ~13us/launch（indexer_bwd L170 锚点）+ 段间程序序 |
| **合计** | **≤ 7680 us** | do_bench(fn, rep=100, warmup=250)、delta=None 两段发射口径（§1.5/§11.3） |

**k2 账面推演（V/C 双链取 max + drain，fwd iter-5 "串行和折叠为 max(V,C)" 同法）**：

- **V 链（per-AIV）**：MTE2 5.0-6.5ms 与 MTE3 5.5-6.3ms 行内流水重叠（散射行循环的读-cast-原子三管道软件流水，iter-1 载荷瘦身后发射链无跨核 flag）+ vec ~1.0ms（iter-2 实测 0.99 同量）+ scalar 谓词段 → **V 链 ≈ 6.5-7.5ms**。
- **C 链（per-AIC）**：max(GEMM 4.1ms, MTE2 2.4-9.9ms, fixpipe 2.8ms) → **4.1-9.9ms**（区间完全由 C MTE2 带宽口径决定）。
- **k2 ≈ max(V 链, C 链) + drain（首末 token 各一段，~几十 us）**：
  - 乐观带（C MTE2 按 L2 驻留 bulk 口径 50-60GB/s 兑现 + V 行内流水兑现）：**~6.5-7.5ms ✓ 预算内贴线达标**
  - 保守带（C MTE2 按 fwd strided 口径 15-19.3GB/s）：**~7.4-9.9ms ✗ 超预算 0-31%**
  - **判定：临界**——达标条件为两项首次实测对账（§13.7 判定门），不可由设计期单方面宣布达成。

### 13.3 算力下限（Cube 侧）

- **GEMM 总量**：per_token_flop = 2·H·topk·(2·DV+3·DQKV) = 180.4 MFLOP/token × S=2048 → **369.4 GFLOP**（§2.2 主仓公式；40 发/task §5.1-3，**无两遍重算**——pst stash 机制（iter-2 形态继承）删除 pass-2 G1/G2 重算，rev2 amend-1 时代的 +145.9 GFLOP 增量不存在）。
- **有效算力需求**：369.4 GFLOP ÷ 7.565ms = **48.8 TFLOPS** = bf16 峰值 ~320 TFLOPS（20 核聚合）的 **~15%**。
- **锚点校准**：同族稀疏注意力 fwd 实测有效算力 91 TFLOPS（sparse_flash_mla csa）→ **1.9× 覆盖**；fwd 预算口径 48 TFLOPS ≈ 峰值 15% 与本算子同量级。
- **折减项（诚实计入，R11 继承）**：本算子 GEMM 形态 M=64（4 个 m 分形）、N=32 类极小维由 BN=128 合并解决（§5.2）；若 M=64 形态实测效率折半（~45 TFLOPS），仍贴预算线——调参空间（BN/DSEG 档位）+ Stage 3 perf-tuning 兜底。
- **纯算时间视角**：369.4 GFLOP @ 320 TFLOPS = 1.15ms；GEMM 窗口占用 1.15/7.565 = 15%，pair-loop 重叠余量充足——**算力不是 rev3 的主约束**（主约束为带宽与流水，§13.4/§13.5）。

### 13.4 带宽下限（逐腿带宽-时间口径；达成依据 = 单核 MTE 实测锚点）

| 口径 | 需求 @7.565ms | 锚点（实测） | 余量 | 定位 |
|------|---------------|-------------|------|------|
| HBM（全接口） | ~545MB（首读 q 151+do 134+kv 2.25+idx 4+lse 0.5 + ws 首次填充 ~95 + 输出 dq 151+dkv 2.25+终态 4.5）→ **~71GB/s** | 273 读 / 238 写 GB/s | **3.3-3.8×** | ✓ 非瓶颈 |
| AIV MTE2（每 AIV） | 62-70MB → **8.2-9.3GB/s** | 12.2（微事务）/ 20.5（批量行） | 1.3-2.5× | ✓（L5 微事务腿单列 2.17ms 已入 V 链） |
| AIV MTE3（每 AIV） | ~99MB → **13.1GB/s** | 15.8-18GB/s（iter-1/2 bwd 实测写口径） | **18-37% ⚠️ 贴线** | 与 MTE2 重叠是 V 链兑现前提（§13.2） |
| C MTE2（每 AIC） | 141.6-148.9MB → **18.7-19.6GB/s** | **15-19.3（fwd strided）/ 50-60（reducesum bulk L2 驻留）** | 0.8×-3×（**口径依赖**） | ⚠️ **主对账项**（§13.7 判定门；rev3 读源为 L2 驻留连续 2D bulk，事务形态倾向后一口径） |
| C fixpipe（每 AIC） | ~141MB → 18.6GB/s | 49.9（fwd 实测） | 2.7× | ✓ |
| L2 中转 | 工作集 ~95-130MB 常驻（§8.4）+ 原子/贡献槽流量 | 192MiB 容量 | 容量面 49-68% | 容量达标；吞吐面按单核 MTE 实测口径核算（rev1 教训：L2 吞吐估算不作达成依据），交叉验证项 |

- **L5 fp32 兜底档重估**（§6.6 触发后）：AIV MTE2 88-96MB → 11.6-12.7GB/s ≈ 微事务饱和值 12.2——**单管道贴死**；叠加 MTE3 5.5-6.3ms 重叠需求 → 全接口预计 **~9-10ms 超预算 ~20-30%**，如实披露不设遮掩（§14 R14）。

### 13.5 pair-loop 流水预期（锁步消除的量化目标）

- **主路径预期锚点（AUTO 栈 + if-guard 错位结构，indexer_bwd 实证）**：indexer_bwd 重构前 lockstep 92% → pair-loop 重构后**消除**（70.9us 达成，零 Expert API，§4.2 查证）→ rev3 预期 **AIC 等 flag 73% → <15%（健康带）**、AIV 等 flag 38% → <15%；**>30% 触发 Expert fallback 升级判据**（§4.2/§9.7，唯一量化触发条件）。
- **健康 pair-loop 的量纲上界（fwd iter-5/6 Expert ring 参照，DEPTH=2/LEAD=1——与 rev3 PARITY=2/DEP=1 同深度）**：fwd 实测 **C/V 重叠率 ≈100%、双侧 cross-flag 等待 ≈0**（C 自缚 34.2us/块 / V 自缚 34.2us/块，串行和折叠为 max(V,C)）——rev3 账面推演（§13.2）即按此量纲（max 折叠）组织。
- **AUTO 栈结构性赌注的诚实披露**：fwd 会话 iter-4 RC1-RC5 探针证明 **3+ 槽轮转 ring 在 AUTO 栈被五层机制封死**（flag 轮转 RC1 / 静态配对 RC2 / UB 别名 RC3 / 规划器 bug RC4 / Mod 区域索引 aicore exception RC5）——fwd 的 ring 最终以 Expert 形态达成。rev3 主路径**不在该封死面内**的根据：采用 indexer_bwd 的**静态 if-guard 2 槽错位形态**（无跨迭代 flag id 轮转、无 3+ 槽 Mod 环、UB 槽持续驻留无别名多区间——§8.6 口径），该形态在 AUTO 栈有 indexer_bwd 运行实证；但"本算子的槽族规模（6 族 vs indexer_bwd 3 族）下生成序仍正确"**未经本算子验证**——此即 §4.2 升级判据存在的理由，§14 R15 记风险。
- **Δ 屏障与 drain 残余**：Δ 定格屏障代价 = consume 段延后 1 token 深度（DEP=1，§9.4）；首/末 token drain 各 ~1 段（per-task 闭合，§10.3）——两项均入 §13.2 的 max 推演余量，不单列预算。

### 13.6 与 rev2 实测 28541us 的对照（削减来源逐项映射）

**rev2 iter-2 实测分解（ascendc，全接口 28541us；perf_log.md Iteration 2 / baseline_iter2.json）→ rev3 账面**：

| 项 | rev2 iter-2 实测 | rev3 账面（§13.1/§13.2） | 削减机制 |
|----|------------------|--------------------------|---------|
| AIC 等 flag | 20.4ms / 73%（其中 19.8ms 等 V 侧 kv stash 生产） | <15% 预期（健康带） | pair-loop 生产/消费错位（indexer_bwd 锚点，§13.5） |
| AIV 等 flag | 10.7ms / 38% | <15% 预期 | 同上（V 散射(t) 与 C produce(t+1) 重叠） |
| AIV MTE2 | 10.8ms @ 12.2GB/s（读量 ~147MB：ws_dkv 60 + KV 30 + ws_dq 30 + s/dp 13 + pst 13 + idx 1） | 5.0-6.5ms（62-70MB @ 批量口径） | 读量减半（两遍→融合单遍 + L5 bf16 档 60→26.4）+ 批量 2D 事务化 |
| AIV MTE3 | 5.6ms（gather 微写 30 + 原子散射 ~53 + 小腿） | 5.5-6.3ms（~99MB，**原子 52.9MB 物理下限不变**） | 写字节量同量——收益在**行内三管道流水化**（发射链与 MTE2 重叠）而非字节削减 |
| vec | 0.99ms | ~1.0ms | 不变（iter-2 向量化形态继承） |
| C 侧 | busy ~7.6ms（含于 73% 等待的间隙内：GEMM 发射 + MTE 发射 + flag 开销交织） | 4.1-9.9ms（max 折叠后独立对账） | 同流量去锁步——C MTE2 口径为最大不确定项（§13.4） |
| k2 合计 | ~28.0ms | 6.5-9.9ms | — |
| **全接口** | **28541us** | **~6.6-10.0ms** | **−65% ~ −77%（乐观带）/ −65%（保守带）** |

- **不可再减的物理下限（诚实声明）**：① 原子终态写 52.9MB/AIV（2.12GB 全序列，fp32 终态区 + k3 单次 cast 语义锚点，§1.3——档位/结构均不改变该量）；② C 侧 kv 2× bulk 读 115.2MB/AIC（L1 512KB < 576KB 全驻强制，§8.2——任何 BN 档位下 topk 行全驻恒 576KB，无档位缓解）；③ L5 逐行微读 26.4MB（随机行散射寻址查证冻结，§6.2）。三项合计构成 rev3 数据流的下界——7680us 的达成空间全部来自锁步消除与流水重叠，**无进一步的"免费"削减项**。
- **若保守带兑现（超预算）**：缺口来源排序 = C MTE2（最大）> V MTE3 贴线 > GEMM M=64 效率；处置链见 §14.3——C MTE2 无结构性替代（物理约束），届时按"实测缺口 + 物理下限论证"如实上报 Orchestrator 重新裁决目标或架构（§14 R16）。

### 13.7 实测锚点校准表 + 置信度与对账协议

**锚点表（全部数值来源，禁止无锚点估算）**：

| 锚点 | 数值 | 来源 |
|------|------|------|
| Cube bf16 峰值（20 核聚合） | ~320 TFLOPS | fwd DESIGN §9.5（48 TFLOPS ≈15% 反推 + sparse_flash_mla csa 实测 91 佐证） |
| 同族实测有效算力 | 91 TFLOPS | sparse_flash_mla csa |
| HBM 读 / 写 | 273 / 238 GB/s | A2/910B3 实测（fwd §9.5，indexer_bwd DESIGN §9.4 同款） |
| AIC ws→L1 bulk 连续读 | 50-60 GB/s/核 | reducesum msprof |
| AIC mte2 strided 窄行读 | 15-19.3 GB/s/核 | fwd iter-5 msprof（176KB/块 strided 256B） |
| C fixpipe 写 | 49.9 GB/s | fwd 实测 |
| AIV mte2 批量行读 | 20.5 GB/s | fwd iter-5 msprof |
| AIV 微事务读写饱和 | 12.2 GB/s（读）/ 15.8-18 GB/s（写） | rev2 bwd iter-1/2 msprof |
| 批量 2D vs 微写加速比 | ×3.6（8.6GB/s rw） | fwd probe2 batch32 |
| pair-loop 锁步消除 | 92% → 消除（70.9us） | indexer_bwd（AUTO 栈实证） |
| Expert ring 重叠率 | ≈100%（cross-flag 等待 ≈0） | fwd iter-5/6 fast（DEPTH=2/LEAD=1） |
| launch marshal | ~13 us/launch | indexer_bwd L170 |
| GEMM 工作量 | 369.4 GFLOP | 主仓 per_token_flop × S（§2.2） |
| dKV 原子流量 | 2.12GB 有效 | §6.1 因果有效行次（917,760 行 × 2304B） |
| rev2 全接口实测 | 28541us（iter-2） | perf_tuning/sparse_mla_bwd/perf_log.md |

**置信度表**：

| 项 | 需求 | 锚点能力 | 余量 | 置信度 |
|----|------|---------|------|--------|
| HBM 带宽 | ~71GB/s | 273/238 GB/s | 3.3-3.8× | **高** |
| Cube 算力 | 48.8 TFLOPS | 91（同族） | 1.9×（M=64 折减后贴线） | **中高** |
| AIV MTE2 | 8.2-9.3GB/s | 12.2-20.5 | 1.3-2.5× | **中高** |
| AIV MTE3 | 13.1GB/s | 15.8-18 | 18-37% | **中（贴线）** |
| C MTE2 | 18.7-19.6GB/s | 15-19.3 / 50-60（口径依赖） | 0.8×-3× | **中低（主对账项）** |
| C fixpipe | 18.6GB/s | 49.9 | 2.7× | **高** |
| pair-loop 流水 | AIC 等 flag <15% | indexer_bwd 实证 / fwd ring 量纲 | 结构先例在、本算子未验 | **中** |
| L5 fp32 兜底档 | 11.6-12.7GB/s（MTE2 单管道） | 12.2（微事务饱和） | ≈0× | **低（预计超预算，披露项）** |
| **总体 7680us（基准档）** | — | — | — | **中** |

- **总体置信度：中**（较 rev2 设计期的"中高"诚实降一档）：乐观带贴线达标（6.5-7.5ms）、保守带超 0-31%——差距集中在两项首次实测对账（C MTE2 带宽口径、pair-loop 编译行为），均有量化判定门与 fallback（§14），不存在"无判据的乐观"。
- **对账协议（Stage 2 强制项）**：首次实测即 msprof 分段——k2/k3 分段 + AIC/AIV 等 flag 占比 + MTE2/MTE3/fixpipe 分管道有效带宽——与 §13.1/§13.2 逐行对账，偏差 >15% 的段触发 §14 对应风险项排查。
## 14. 风险与 fallback（rev3：rev2 证伪清单入表防重蹈 + Expert fallback 升级路径 + 散射/槽档兜底）

> 前置声明：§3 技术约束检测（含 §3.7 证伪清单、§3.8 汇总）已逐项过检并在 §4-§9 内建处置；本表列**残余/实施期风险**。概率/影响为设计期预估，Stage 2 实测对账（§13.7）后修正。编号延续 rev2 R1-R12（任务书 §3.3 沿用），rev3 新增自 R13 起。

### 14.1 rev2 六条证伪清单入风险表（防重蹈覆辙——§3.7 的风险面表述，Stage 2/3 任何优化尝试先对照本表）

| # | 证伪路径（rev2 结构内） | 复现签名 | rev3 结构性规避（为何不复发） | 残余暴露面（何场景需再查） |
|---|------------------------|---------|------------------------------|---------------------------|
| 1 | [64,576] fp32 ws_dkv 批量读回 staging | UB 规划器 `available: 0` | 大 staging 形态消失（逐行微读 [576] 行缓冲 ~2.5KB，§8.3）；UB 持续驻留口径（§8.6） | 无——Stage 2 禁止重新引入任何 >10KB 级跨段 staging |
| 2 | [32,576] fp32 3:3 staging + 第 17 对 flag | 双重阻塞（规划器 + flag >16） | 3 点形态不存在（pair-loop DEP=1 → 2 槽）；flag 对数 = 槽族数 ≤12（§9.3 记账） | 同步语句重构后须复核 ≤16（§9.3 口径） |
| 3 | seg-only [64,512] staging | 同 #1 | 同 #1 | 同 #1 |
| 4 | C 侧按段拆分（dkv 段间分担） | L0C 累加器复用冲突 | DSEG 为 GEMM 编排自然段、acc 块内短生命周期（§5.1-2），与证伪签名不同源 | 若 Stage 3 重排 GEMM 段序，须复核 §8.1 分时表峰值 |
| 5 | 按行半拆分（AIV 各半） | vid 数据就绪不对称破坏顺序配对 | 不做按行半拆分（两 AIV 为半片/半区分工，§8.3/§8.4 全对称） | 任何"两 AIV 分工不等价"的优化需先证对称性 |
| 6 | iter-1 3:3 设想（C 领先 3 块） | 追溯证伪（flag ≤16 + 语句数相等下不可表达） | pair-loop DEP=1（领先 1 token），槽位/flag 均闭合（§4.3） | PARITY 加深（§9.5 可选项）须重算 flag 与 GM 槽族 |
| 补 | 单条大 2D 写 SEGV / 3D-of-4D 截断 / fp32 直传 | 640 宽 NI=4 SEGV；dkv sim 0.999996/frac 36%；gemm_v0 同 dtype | pair 粒度分段写 + 平面 ≤2 + bf16 单档（§5.4/§8.5）内建 | dq 终写 [64,576] 单平面整写为边界形态——SEGV 复现即按 DSEG 拆段（§5.4 记录） |

### 14.2 风险表（rev3 新增 R13-R17 + rev2 R1-R12 重估）

**rev3 新增**：

| ID | 风险 | 概率/影响 | 触发信号 | fallback / 处置 |
|----|------|----------|---------|----------------|
| R13 | **pair-loop 双缓冲槽复用竞态**（AUTO 生成序与 §9.4 结构推导不符 → 陈旧读/部分覆写——精度错误而非挂死，§12.6(1)） | 中 / 高 | L0 偶发 FAIL 或错误按 task 奇偶 2-周期聚类；同 seed 不逐位一致 | §12.6(1) 验证协议（seed 三跑 + 单 token debug + tslot 读写序 dump）→ 核对 §9.3 命名清单 → 程序序断言 → **Expert fallback 手动 flag 显式化读写屏障**（§9.7 表逐对对应） |
| R14 | **bf16 槽档域 B 门控 FAIL**（L5 贡献槽 bf16 或 L4 优化档在 σ≤10 域 sim < 0.999，§12.6(2)） | 中 / 中 | 域 B 四例 dkv/dq sim < 0.999 且排除实现 bug（dump 贡献槽舍入量复核） | 档位回退：L5 → fp32 兜底（精度路径与 rev2 逐位一致必过；AIV 读量 88-96MB、全接口预计 ~9-10ms 超预算 ~20-30%，**如实披露**，§13.4）；L4 → fp32 基准（+7MB） |
| R15 | **AUTO 栈对 if-guard pair-loop 不生成预期软件流水**（fwd RC1-RC5 同族封死机制的延伸风险——本算子 6 槽族规模未经验证，§13.5） | 中 / 高 | 复测 msprof **AIC 等 flag > 30%**（升级判据，§4.2/§9.7）；诊断分流先排除 DEP 退化与命名漏配 | **Expert fallback 升级**：k2 transit 段改手动 set_flag/wait_flag（≤12 对逐对记账 §9.7 表）+ 关闭 AUTO_CV_SYNC 单项；ascendc 先行、pto 维持主路径（正确性同源、性能如实报告）；数据流/接口/测试资产不动（§8.7 切换点收敛） |
| R16 | **C 侧 MTE2 bulk 读带宽贴线**（141.6-148.9MB → 需 18.7-19.6GB/s；fwd strided 锚点 15-19.3 不足、reducesum bulk 锚点 50-60 宽裕——口径依赖，§13.4） | 中 / 高 | 首次 msprof C 侧 mte2 有效带宽 **<25GB/s**（判定门 §13.7） | **无结构性替代**（kv 2× 读为 L1 容量物理强制，§13.6 下界声明）——按"实测缺口 + 物理下限论证"上报 Orchestrator 重新裁决目标/架构；Stage 3 可评估项仅剩 pd 槽布局优化（12.8MB 边际）与 GEMM/MTE 重叠度微调 |
| R17 | **V 侧 MTE3 原子散射 + ws_kv 写贴线**（~99MB → 13.1GB/s vs 15.8-18 锚点，余量 18-37%；兑现前提 = 散射行循环三管道流水化，§13.2） | 中 / 中 | msprof MTE3 段 **>6.5ms** 或散射段占 bwd >1/3 且重叠后仍关键路径（§6.6） | 先核行内流水是否退化（读-cast-原子是否被编译器串行化——对照 iter-1 载荷瘦身形态）；散射机制本身冻结（§6.2 查证 + §3.7 补充证伪，无 AUTO 栈内替代）；残余走 §6.6 性能触发链 |

**rev2 R1-R12 重估（压缩表；处置已由 rev3 架构内建或继承）**：

| ID | rev2 风险 | rev3 重估 |
|----|-----------|----------|
| R1 | V 侧散射带宽贴线（28.6 vs 26GB/s） | **已架构级回应**：读量 147→62-70MB（§4.5）；残余贴线移至 R17（MTE3 写侧）与 R16（C 侧）——风险未消失但换了主战场，判据重锚 |
| R2 | L0C fragment 分配失败（G5 128KB 贴线） | 继承：BN=64 回退档（G5 [64,256]=64KB，§8.1）；块内短生命周期使冲突面更小（§5.1-2） |
| R3 | UB 超 192KB | 继承：rev3 分时口径 184KB/95.8% + 四级削减链（§8.3）；BN=64 档结构性回退（−30KB） |
| R4 | fwd 并行会话依赖不稳定 | **已解除**：fwd 已落盘且交付（§11.4）；`SPARSE_MLA_BWD_FWD` 环境开关与 `_torch_fwd_fallback` 保留为防御路径 |
| R5 | pto 后端逐项差异 | 降级：rev2 全量双后端验证背书（28.5ms 双通过）；rev3 主路径同一 API 集；fallback 的 pto 手动 flag 不可用性单独记 §9.7（pto 维持主路径，不阻塞） |
| R6 | pto GM 标量读无 varlen 先例 | 降级：rev2 已在 pto 全量验证覆盖（§10.4）；RowMeta 预展开预案保留 |
| R7 | AUTO_CV_SYNC 配对遗漏挂死 | 继承（形态更新）：六族槽命名清单 §9.3 + 挂死排查路径 §9.6；rev3 增 R13（不挂死但算错的对应面） |
| R8 | lse 数值差传播 | 继承：fwd Lse fp32 门禁 + 本算据双门限覆盖（§11.4）；性能验收不依赖 fwd |
| R9 | kv_group>1 padding head 语义 | 继承：考核 shape kv_group=1 无 pad；Stage 2 L1 扩展项 |
| R10 | 无 seed 用例波动 + 环境噪声 | 继承：seed∈{1,2} 复跑协议（§12.5）；偶发 FAIL 分流新增 R13 首查 |
| R11 | M=64 小 GEMM Cube 效率 | 继承（中高）：需求降至 48.8 TFLOPS（无两遍重算，§13.3）、余量 1.9×；GEMM 段 msprof 对账为强制项 |
| R12 | Δ 恒等式语义边界（契约外 o 输入） | 继承：披露不变（§1.3 amend-1 第 3 条）；k1 对照诊断路径保留（§11.1） |

### 14.3 fallback 升级路径汇总（决策树；判据全部量化，无模糊触发）

```text
精度轴（§12.6(2)）：
  L5 bf16 档域 B 门控 FAIL（排除实现 bug）
    → L5 切 fp32 兜底档（发射结构不变，仅 dtype 参数；AIV 读量 88-96MB、
      预计 ~9-10ms 超预算——性能缺口如实披露，不遮掩）
    → L4 bf16 优化档仅在 L5 bf16 全绿后单独评估；任一 FAIL 同法回退 fp32 基准
  L0 偶发 FAIL / 奇偶聚类（精度面）
    → §12.6(1) 槽复用竞态协议（seed 三跑 → tslot dump → 命名/程序序核对）
    → 仍复现：并入性能轴的 Expert fallback（手动 flag 显式化读写屏障，§9.7）

性能轴（§4.2 唯一升级判据 + §13.7 判定门）：
  复测 msprof AIC 等 flag > 30%（先分流：排除 DEP 退化 / 槽族命名漏配 / R13 竞态）
    → Expert fallback：k2 transit 段手动 set_flag/wait_flag（≤12 对 §9.7 预算表）
      + 关闭 TL_ASCEND_AUTO_CV_SYNC 单项（其余 3 项保留）
      + 显式 alloc_ub/L0* 手动放置（§8.7 映射表）
      —— ascendc 先行；pto 维持主路径（§9.7 查证结论）；数据流/接口/测试资产零变化
  C 侧 mte2 < 25GB/s（R16）
    → 无结构性替代（L1 容量物理强制）→ 实测缺口 + 物理下限论证上报 Orchestrator 裁决
  AIV MTE3 > 6.5ms（R17）→ 行内流水核验 → §6.6 性能触发链

散射轴（§6.2 查证冻结 + §6.6）：
  随机行散射 C 侧直写：双后端源码级不可行（单基址连续 region + L0C 行切片不可靠）
    —— 兜底即选定方案本身（V 侧逐行 UB 原子 + bf16 贡献槽），无进一步回退空间；
    分桶/私有区/行中心三路均数据否决（§6.2）；精度 FAIL 走精度轴 L5 fp32 档
```

- **升级不改动的边界（重申 §8.7）**：接口签名、槽布局（§8.4）、GEMM 编排（§5.4）、OOB 防护（§7）、k3 形态（§11.2）、测试资产（§12）——全部 fallback 切换点收敛在 k2 transit 段（alloc 方式 + flag 机制 + 槽档 dtype 参数），Stage 2 按双路径可切换组织。
- **目标不可达时的诚实出口**：保守带全兑现（R16+R17 叠加）且 Expert fallback 亦不达 7680us → 按 §13.6 物理下限论证（原子 52.9MB + kv 2× 115.2MB + L5 微读 26.4MB 三下界）上报 Orchestrator 重新裁决目标值或架构方向——**禁止以放宽精度门限（§12.2 禁止条款）或削减测试覆盖换取性能数字**。

### 14.4 风险总评与文档完成声明

**风险总评**：无单点致命风险，但 rev3 的达标置信度为**中**（§13.7）——与 rev2 设计期"中高"的差异是**诚实降档**：主贴线项从 rev2 的单项（V 侧读带宽）变为双项（C 侧 MTE2 口径依赖 + pair-loop 编译行为未本算子验证），且两者均有量化判定门（<25GB/s / >30%）与既定 fallback（R16 上报 / R15 Expert 升级），不存在无判据乐观。精度面风险（R13/R14）有协议化验证路径（§12.6），门限数值全程不动。rev2 六条证伪清单已入 §14.1 防重蹈表——Stage 2/3 任何优化尝试先对照该表与 §3.7 签名库。

**前块决策不推翻承诺（块 D 核验）**：块 D 未发现块 A-C 决策错误；本块两处数字勘误（§5.5 "297MB" 读+写合计粗口径 → §13.1 管道分列修正；§4.5/§5.5 逐腿读量的任务数折算口径统一按 §13.1 表为准）均为口径澄清，决策语义不变。块 A-C 的关键决策（Developer pair-loop 主路径 + Expert fallback、§6 散射 V 侧保留 + bf16 槽档、§8 双缓冲预算、§9 flag ≤12 记账）在块 D 量化核算（§13）下成立，贴线项 R14/R16/R17 均在既有回退框架内闭环。

---

**文档完成声明（Stage 1 rev3，块 D 收尾，2026-10-15）**：本文档按 rev3 任务书 4 块分块交付完毕——块 A（文首 + §0-§4）/ 块 B（§5-§7，含 §6 散射查证裁决与 §4.4/§4.5 就地修正）/ 块 C（§8-§11，含 §6.5 勘误与 §9.1 API 名订正）/ 块 D（§12-§14 + proto 校验，本次）。配套接口规格 `proto_sparse_mla_bwd.yaml` 接口未变、块 D 校验通过（dtype 全集与 §12.2 精度表一致、attrs 覆盖 §1.4 全部关键属性，无需重写）。测试资产（test_sparse_mla_bwd.py 全部用例与 golden）零重写直接复用（任务书 §3）。L1/L2/Boundary 扩展复用 Stage 2 场景 B 已有产物；kernel 实现与 §13.7 对账协议落地归 Stage 2（tilelang-op-develop），性能调优与 fallback 判据复测归 Stage 3（tilelang-perf-optimization）。
