# sparse_mla_topk_reducesum 通用路径重构设计（Panel/Auto 移除 + P1 AIC GEMM 流水重构）

> **文档定位**：本设计替代 `DESIGN_panel_mode.md` 成为算子的现行设计文档（panel 设计随 git 基线
> `0f11b3cc` 保留为历史参考，不再维护）。本版包含两条主线：
>
> 1. **移除**：删除 panel/auto/shared 三模式分发全链路（shared-panel 路径、共享索引检测 kernel、
>    双 guarded 推测发射、panel_mode 参数），回归**纯通用接口**——支持任意合法 topk_indices，
>    不依赖索引共享 / 固定分布 / Golden 数据内容假设。
> 2. **P1**：AIC GEMM 流水重构——gemm_v0 尾部 MTE1_MTE2 同步改 per-slot event（框架 opt-in
>    参数）+ kv_l1/q_l1 双缓冲，打通"当前 tile 计算 ∥ 下一 tile 搬运"。
>
> 语义（数学定义、支持域、精度标准）与移除前**逐位一致**——本重构是纯性能/结构重构，不改任何
> 计算语义。性能基线：**generic random BI128 = 441.7µs**（perf_tuning/generic_pipeline_analysis.md
> 锚点；本会话配对 432.8µs，跨会话漂移 ±2-3%）。

---

## 1. 概述

### 1.1 算子名称

`sparse_mla_topk_reducesum`（DSA 稀疏注意力 topk 概率质量 head 归约）

### 1.2 功能描述

varlen packed 布局下，对每个 token 行 s、每个 kv_group g、每个 topk 选中位置
idx = Indices[s, g, j]，计算 score = exp((Q·KV_main + Q_tail·KV_tail)·sm_scale − Lse)
（causal mask 内置：idx ≤ s 且 idx ≥ 0，否则 score=0），沿 head 维求和得
ReduceSum[s, g, r, j]。接口层归一化（sum(REPLICATE_H)/div）属交付边界之外。

**任意索引通用性（本版核心承诺）**：topk_indices 只需满足值域约束 [-1, ΣS_kv)（-1/越界经
clamp+mask 语义安全丢弃）；逐行独立、部分共享、全同、非连续、乱序等**任意分布均正确且走同一
条代码路径**——不存在任何索引内容假设分支。随机逐行独立 TopK 索引为**主要性能评估对象**。

### 1.3 数学公式

$$
\text{ReduceSum}[s,g,r,j] = \sum_{h \in \text{slice}(g,r)} \exp\big((Q[s,h,:]\cdot K[\text{idx}] + Q_t[s,h,:]\cdot K_t[\text{idx}])\cdot sm\_scale - Lse[s,h]\big)\cdot \mathbb{1}[\text{idx} \le s_i \wedge \text{idx} \ge 0]
$$

其中 idx = Indices[s_row, g, j]（seq 内相对位置，行寻址经 bos 平移），无效位贡献为 0。

### 1.4 算法描述（数据流不变，仅 C 侧流水重构）

- **V 流（AIV×2，40 核）**：行 gather KV → ws_kv GM 中转（DEPTH=4 槽）→ set F0；post 段
  等 F1 → 载 ws_acc → mask/exp/reduce → 写 ReduceSum。**P1 不触碰 V 侧任何语句**。
- **C 流（AIC，20 核）**：等 F0 → ws_kv slot → L1 → gemm_v0（K=DDT 单段，fp32 L0C 累加）
  → L0C → ws_acc → set F1。**P1 重构点**：L1 双缓冲 + per-slot 释放事件，使
  MTE2(n+1) 搬运与 MTE1(n) 消费重叠。

### 1.5 数据流图

```
GM[Q] ──MTE2──> L1 q_l1[p] ──┐
GM[ws_kv slot] ──MTE2──> L1 kv_l1[p] ──┤── gemm_v0（内部 MTE1 L1→L0A/B 乒乓 + Cube mma）
                              │              └─ opt-in: 尾部 per-slot MTE1_MTE2 释放事件 EV[p]
L0C acc ──FIX──> GM[ws_acc slot] ──set F1──> V 侧
（V 侧链路不变：GM[KV] 行 gather → UB side → MTE3 bulk → GM[ws_kv] → set F0）
```

### 1.6 重构范围与路线

| 项 | 内容 | 依据 |
|---|---|---|
| 移除 | 5-builder 族谱收缩为 1 个 generic builder；panel/auto/shared 分发、detect kernel、_BUILD_CACHE 扩展键全删 | 用户指令 #1/#2；git 基线 0f11b3cc 保存历史 |
| P1（本版实现） | gemm_v0 尾部同步 per-slot 化（框架 opt-in）+ kv_l1/q_l1 双缓冲 | generic_pipeline_analysis.md §2/§5 P1 行 |
| P2（后续） | V 侧 SIDE_PAIR=False @SIDES=2（1 行，spike 已验证安全） | 同上 §4 决定性差分实验 |
| P2b（后续） | V 侧 LEAD=3（gather 发射点前移越过 F1 标量等待） | 同上 §4 结论修正 |
| 不做 | R2 手工 gemm 流水 / R5 AIC 直采 / 反转 Q / L0C→UB fixpipe（四路已判死，§10.1） | 同上 §5 |

**性能路线（诚实预期框架）**：当前 441.7 → P1 单独 ≈ 437-441（**0-2%，V 侧 4.26µs/块自封顶，
共门控**；P1 价值 = C 侧使能项）→ P1+P2+P2b ≈ 375-395（−11~−15%，乐观 ~350）→ 220µs 目标
**在当前 ws 往返架构下不可达**（C 侧 ws 读引擎时间 332µs > 220µs）。**严禁把 ~350-370µs 的
"完美流水极限"当硬件理论下界**——它是当前架构（ws 302MB 往返）的估算极限，架构级突破另论。

> **⚠️ Host 侧 Buffer 操作约束**（ascend-constraints.md §5）：host 侧只允许经证明不物理化的
> metadata view、数据准备（新建张量 + 一次 H2D）与 kernel 调用；本算子 host 路径见 §8.6 审计。

---

## 2. 编程模式选型

### 2.1 模式结论

**Expert 模式**（用户指定，历史版本已定，本版不变；P1 属 Expert 框架内的手动同步扩展）

### 2.2 选型理由

- CV 双流融合（gemm_v0 Cube + mask/exp/reduce Vector）+ GM workspace 中转 + 手动 cross-flag
  协议——Developer 模式的自动 CV 分离/同步 pass 无法表达"V 预取 LEAD 步 + DEPTH 槽 ping-pong +
  扁平块流跨 task 旋转流水"这类显式流水结构（与 examples/sparse_flash_attention/
  example_sparse_flash_attn_dynamic_shape.py 同构的既定选择）。
- P1 的 per-slot in-core 事件属纯手动同步范畴（mha_sink_fwd_bhsd / example_gemm_intrinsic
  先例，见 §3.5.3），进一步排除 Developer 模式。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | T.alloc_L1 / T.alloc_L0C / T.alloc_ub 显式层级（P1：q_l1/kv_l1 升为双槽） |
| 计算方式 | T.Scope("C") gemm_v0 + T.Scope("V") tile 算术链 |
| 作用域 | 显式 T.Scope("C")/T.Scope("V") |
| 同步方式 | 手动：cross-flag（F0/F1，不变）+ in-core set_flag/wait_flag（P1 新增 per-parity L1 token）+ gemm_v0 内建链（P1 opt-in 改尾部） |
| pass_configs | 不设置（全手动，不变） |

---

## 3. API 映射设计

### 3.1 公式拆解（计算步骤不变，沿用）

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | logits = (Q·KV_main + Q_tail·KV_tail)·sm_scale | 单段 gemm K=DDT=576（合并布局，C 侧每块一次 gemm_v0） |
| 2 | mask = 1{idx ≤ s_i}·1{idx ≥ 0} | clamp 算术链（V 侧） |
| 3 | p = exp(logits − Lse)·mask | V 侧 tile 链 |
| 4 | ReduceSum = Σ_h p | reduce_sum dim=0（V 侧） |

### 3.2 TileLang API 映射（P1 增量）

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | gemm | `T.gemm_v0(q_l1[p], kv_l1[p], acc_s_l0c, transpose_B=True, init=True, l1_release_event=L1EV_BASE+p)` | **新 kwarg**（框架 opt-in，§3.3） | Expert |
| 1a | L1 槽写所有权获取 | `T.wait_flag("mte1", "mte2", L1EV_BASE+p)` | MTE2 等 MTE1 释放同奇偶槽 | Expert |
| 1b | L1 槽生命周期开/闭 | `T.set_flag("mte1", "mte2", L1EV_BASE+{0,1})`（init）/ `T.wait_flag(...)`（destroy） | pretend-free 惯例 | Expert |
| 2-4 | mask/exp/reduce | T.tile.add/clamp/mul/exp、T.reduce_sum、T.copy（不变） | — | Expert |

其余 API（T.copy / T.set_cross_flag / T.wait_cross_flag / T.alloc_* / T.tile.*）与现行逐条一致。

### 3.3 框架侧改动规格（gemm_v0 opt-in 参数——本设计唯一框架改动）

**现状（源码实读）**：`tilelang/language/ascend.py::gemm_v0`（L418-531）把模板参数烘焙进
args[0] 字符串 `gemm_v0<{T1},{T2},{M},{N},{K},{ta},{tb},{kL0Size}>`；`GemmOpCodegen`
（codegen_ascend.cc L2714-2735）以 `op_name = "tl::ascend::" + args[0]` 直通输出，不解析字段；
pass 侧（operation_config.h / pipeline_planning.cc L131-141 / ascend_sync_insert*）按**注册名**
`tl.ascend_gemm_v0` 索引，与模板参数无关。设备模板 `src/tl_templates/ascend/common.h`
L1185-1371 的尾部为**队列粒度**排空链：

```cpp
// 现尾部（common.h L1364-1370）——P1 攻击点
WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT);      // MTE1 队列：等 mma(s0/s1) 排空
WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT + 1);
SetFlag<HardEvent::MTE1_MTE2>(L0AB_EVENT);    // MTE1 排空(含 mma) → 通知 MTE2
WaitFlag<HardEvent::MTE1_MTE2>(L0AB_EVENT);   // MTE2 队列：**无条件**门控其后所有 MTE2 op
SetFlag<HardEvent::M_FIX>(L0AB_EVENT);
WaitFlag<HardEvent::M_FIX>(L0AB_EVENT);
```

尾部 WaitFlag 排进 MTE2 队列后，**下一块的所有 MTE2 搬运被硬编码等待本块 MTE1 全队列
（含 mma）排空**——跨块 MTE2(n+1)↔MTE1(n) 重叠被模板锁死（generic_pipeline_analysis §2
的源码级结论；"kv_l1 双缓冲单独无效"的根因）。

**改动 1：前端**（`tilelang/language/ascend.py`）

```python
def gemm_v0(A, B, C, transpose_A=False, transpose_B=False, init=False,
            kL0Size=128, n_actual=None, l1_release_event=None):
    # l1_release_event: int ∈ [0,7]。opt-in：本调用尾部以
    # SetFlag<MTE1_MTE2>(l1_release_event) 在 **MTE1 拷贝排空点**（mma 排空等待之前）
    # 发布 per-slot L1 释放事件，并**不再发射**旧尾部 SetFlag/WaitFlag<MTE1_MTE2>(0)
    # 队列门控。None → 模板字符串逐字节不变（所有既有调用者零影响）。
    ...
    tmpl = f"gemm_v0<{_dtype(A)}, {_dtype(C)}, {M}, {N}, {K}, {ta}, {tb}, {kL0Size}"
    if l1_release_event is not None:
        assert isinstance(l1_release_event, int) and 0 <= l1_release_event <= 7
        tmpl += f", {l1_release_event}"
    return T.call_intrin("handle", tir.op.Op.get("tl.ascend_gemm_v0"),
                         tmpl + ">", Aptr, Bptr, Cptr, init, n_actual)
```

**改动 2：设备模板**（`src/tl_templates/ascend/common.h`，gemm_v0 模板追加带默认值的模板参）

```cpp
template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          bool transpose_A = false, bool transpose_B = false,
          uint32_t kL0Size = 128, int l1ReleaseEv = -1>
CATLASS_DEVICE void gemm_v0(...) {
  ...  // 头部与主循环逐指令不变
  // ---- 尾部（P1 分支）----
  if constexpr (l1ReleaseEv >= 0) {
    // per-slot 释放：置于 mma 排空等待**之前**——本调用全部 L1→L0 拷贝完成即发布
    // （L1 奇偶槽可写），消费方为调用方 wait_flag("mte1","mte2",l1ReleaseEv)
    SetFlag<HardEvent::MTE1_MTE2>(l1ReleaseEv);
  }
  WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT);      // L0A/B 槽复用协议（不动）
  WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT + 1);
  if constexpr (l1ReleaseEv < 0) {
    SetFlag<HardEvent::MTE1_MTE2>(L0AB_EVENT);   // 旧路径：全队列排空释放
    WaitFlag<HardEvent::MTE1_MTE2>(L0AB_EVENT);  // 旧路径：无条件 MTE2 门控
  }
  SetFlag<HardEvent::M_FIX>(L0AB_EVENT);         // FIX 释放（不动）
  WaitFlag<HardEvent::M_FIX>(L0AB_EVENT);
}
```

**爆炸半径分析与决策（原地修改 vs opt-in vs 算子侧绕过）**：

| 方案 | 评估 | 决策 |
|---|---|---|
| 算子侧绕过 | **不可行（证明）**：尾部 `WaitFlag<MTE1_MTE2>(0)` 是模板**无条件**排入 MTE2 队列的指令，任何 DSL 级发射顺序重排都无法绕过队列内指令（FIFO 语义）；gemm 调用后再 set per-slot 事件也会排在 M_MTE1 等待之后（退化为 mma 粒度）；分析文档 §2 已实证"buffer 数量不改变 flag 语义" | 排除 |
| 原地修改（改默认行为） | examples/+testing 共 **80 个文件**使用 gemm_v0（dsa_sparse 除外）——全部变更 codegen，不可测爆炸半径，违反最小必要改动 | 排除 |
| **opt-in 参数（选定）** | 事件 id 随模板字符串传递：**不改** op 注册（仍 6 输入）、**不改** codegen_ascend.cc（字符串直通）、**不改** operation_config.h / pipeline_planning / ascend_sync_insert*（按注册名索引）、**不改** codegen_ascend_pto.cc（PTO 模板独立于 tl_templates/ascend/common.h，且 pto.py 已废弃不会传入新 kwarg）。C++ 侧 `int l1ReleaseEv = -1` 默认值使 8 参实例化逐指令等价旧模板；未传 kwarg 的 Python 调用模板字符串逐字节不变 | **采纳** |

框架回归门禁：既有 gemm_v0 用户字节级不变由首日 spike S-P1c 锁定（§10.3）。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| T.gemm_v0（含新 kwarg） | ascend.py L418-531 实读 + common.h L1185-1371 实读 | kwarg 待实现（S-P1a 冒烟） |
| T.wait_flag("mte1","mte2",id) / T.set_flag 同对 | ascend.py L233-269（_pipe 含 mte1/mte2）；**C-scope 生产先例**：example_gemm_intrinsic.py L30/L69/L76/L90、mha_sink_fwd_bhsd.py（SIG_K_L1 等四对 MTE1_MTE2 token） | 已验证（先例在库） |
| per-pair 事件 id 命名空间 | tilelang-api-best-practices/references/api-schedule-sync.md L196-200（"event ID 属于有向 (src,dst) pair"）+ 本算子 V 侧 mte3→v / mte3→mte2 同 id=0 共存的生产证据 | 已确认 |
| T.set_cross_flag / wait_cross_flag（mode=2） | ascend.py L191-230 | 不变 |
| T.tile.* / T.reduce_sum / T.copy | 现行实现逐条沿用 | 不变 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 是否涉及 | 处理方案 |
|------|---------|----------|
| 三维 Kernel | No | 一维 persistent grid（不变） |
| threads 参数限制 | No | Expert T.Scope 双流，无 threads 参数（不变） |
| 动态循环边界 | Yes（既有形态） | C 侧 `for k in T.serial(ntask)`、ntask 为 T.ceildiv 符号表达式——ntask 先例（既有）；P1 新增的 init/destroy 与 per-block wait_flag 均**不在**循环界表达式中（纯语句序列 + 常量 id），不触发 CanonicalSimplifier 风险（对照：guarded 版曾因循环界含 GM load 表达式致编译 13s→600s+，本设计零新增循环界表达式） |
| 流水线动态边界 | No | 不使用 T.Pipelined（手动流水） |
| GPU 专用 API | No | — |
| L0C 溢出 | No | H_per_block×BI×4 ≤ 64×128×4 = 32KB ≪ 128KB（A2/A3）（不变） |
| GEMM 非整除 | Yes（K 尾） | K=DDT（如 576），kL0Size=128 → kL0split=5、kL0Tail=64；DDT 为两个 2 的幂之和 ⇒ 恒为 16 的倍数，L1 zN 布局 16 行分形对齐满足（common.h L1220-1228 的 int8 K-tail 约束不触发，bf16 ELE_NUM_PER_C0=16）。M/N 为编译期常量（Hpb∈{16..64} 2 的幂；BI≥32 偶数），MMA 分形粒度满足 |
| T.copy 列方向 strided 切片 | No | ws_kv/ws_acc 均最内维连续切片（不变） |
| Host 侧 aclnn/物理化 | No | 见 §8.6 [HOST-METADATA-AUDIT] |

#### 3.5.2 参考实现差异说明

外部参考为 GPU 原版（ref_mla_topk_softmax 迁移而来）；差异集与历史版本一致（行寻址恒等式、
clamp+mask 语义、合并 DDT 布局、Expert 双流），本版零新增差异。P1 不引入任何语义差异。

#### 3.5.3 本项目同类实现参考（必列）

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| examples/dsa_sparse_finetune/sparse_mla_topk_reducesum.py（现行 1803 行） | 本体 | generic builder = 保留基座；panel/detect/guarded = 移除对象 |
| examples/gemm/example_gemm_intrinsic.py L28-103 | 高（同步协议） | **C-scope DSL 级 per-slot MTE1_MTE2 token 全生命周期先例**：init_flag 预置（L29-34）→ `wait_flag("mte1","mte2",(k+1)%S1)` 获取槽所有权（L76）→ `set_flag("mte1","mte2",k%S1)` L1 消费完毕释放（L90）→ clear_flag 销毁（L37-42）——与 P1 协议同构 |
| examples/mha_sink_fwd_bhsd/mha_sink_fwd_bhsd.py | 高（CV 融合） | C-scope 多 buffer L1 ownership token（SIG_Q/K/P/V_L1 独立 id）+ L0AB/L0C 双槽 token + init/destroy——CV 融合算子手动 per-slot 同步的生产先例 |
| examples/sparse_flash_attention/example_sparse_flash_attn_dynamic_shape.py | 高（结构） | Expert 双流 + varlen + cross-flag 同构先例（现行文件头引用） |
| src/tl_templates/ascend/common.h L1185-1371 | 框架 | gemm_v0 模板实读（P1 改动点） |
| .agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-schedule-sync.md §4 | 文档 | set_flag/wait_flag "物理缓冲 ownership"模式 + per-pair id 命名空间——P1 token 协议的直接文档依据 |
| examples/dsa_sparse_finetune/perf_tuning/generic_pipeline_analysis.md | 依据 | P1 收益模型 / 共门控结构 / 消融协议基础 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量（不变）

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| q | [ΣS, heads, DDT] | bfloat16 | DDT = dim_v + tail_dim |
| kv | [ΣS_kv, kv_group, DDT] | bfloat16 | |
| topk_indices | [ΣS, kv_group, topk] | int32 | 值域 [-1, ΣS_kv)，**任意分布** |
| lse | [ΣS, heads] | float32 | |
| offsets | [batch+1] | int32 | cu_seqlens |
| token_indices | [ΣS, 2] | int32 | host 预构建派生输入（prepare_token_indices） |

### 4.2 输出张量（不变）

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| reducesum | [ΣS, kv_group, REPLICATE_H, topk] | float32 | 框架经 out_idx 分配 |

### 4.3 中间缓冲区（P1 变更仅 C 侧 L1 两项）

| Buffer 名 | Shape | dtype | 存储层级 | 用途 | 变更 |
|-----------|-------|-------|----------|------|------|
| q_l1 | **[2, H_per_block, DDT]** | bf16 | L1 | Q 双槽（P1，含 P8 Q 双缓冲） | 单槽→双槽 |
| kv_l1 | **[2, BI, DDT]** | bf16 | L1 | ws_kv 块双槽 | 单槽→双槽 |
| acc_s_l0c | [H_per_block, BI] | fp32 | L0C | 单块累加器（FIX 排空由 gemm 头部 FIX_M 保护，无需双槽） | 不变 |
| ws_kv | [grid, DEPTH·BI, DDT] | bf16 | GM | V→C 中转（DEPTH=4 槽） | 不变 |
| ws_acc | [grid, DEPTH·Hpb, BI] | fp32 | GM | C→V 中转 | 不变 |
| V 侧全部 UB（indices_ub/kv_gather_ub/rows_ub/...） | — | — | UB | | **不变** |

### 4.4 内存搬运路径（P1 后 C 侧）

```
GM[Q row] ──MTE2──> L1 q_l1[p]      ┐ 奇偶槽 p = n%2；写前 wait_flag(EV[p])（MTE1 消费完毕）
GM[ws_kv slot] ──MTE2──> L1 kv_l1[p] ┘ MTE2(n+1) 填 (n+1)%2 槽 ∥ MTE1(n) 读 n%2 槽（P1 核心）
L1 q_l1[p], kv_l1[p] ──MTE1(gemm_v0 内部)──> L0A/L0B（乒乓，M_MTE1 协议不变）
L0A×L0B ──Cube mma──> L0C acc ──FIX──> GM[ws_acc slot] ──set F1──> V
```

### 4.5 内存预算（golden 配置：H=16/D=512/tail=64/topk=128/BI=128/DDT=576）

**L1（AIC 512KiB，工作预算 511KiB）**：

| Buffer | 现行（单槽） | P1（双槽） |
|---|---|---|
| q_l1 | 16×576×2 = 18,432B | 2×18,432 = 36,864B |
| kv_l1 | 128×576×2 = 147,456B | 2×147,456 = 294,912B |
| 合计（+4KiB 余量） | 165,888B（162KiB） | **335,872B（328KiB）≤ 523,264B ✓**（余量 187KiB） |

域内最坏（Hpb=64，含 REPLICATE_H>1 场景）：2×73,728 + 2×147,456 + 4,096 = **446,464B ✓**。
**编译期回退**：`KVDB = 2 if (2·BI·DDT·2 + 2·Hpb·DDT·2 + 4096 ≤ 523264) else 1`——BI=256
（topk=256）等超预算配置自动落 KVDB=1（旧语义，gemm_v0 不传 kwarg），正确性不变、性能不
劣于现状；L0-G4/G5 各含 ≥1 回退路径用例。分析文档 §5 P1 行的"L1 303KB<511KB ✓"即本预算
（294,912+18,432+对齐 ≈ 306KiB 口径一致）。

**L0A/L0B（gemm_v0 内部，64KiB×2，乒乓预算 32KiB/槽）**：A 槽 M×kL0Size×2 = 16×128×2 = 4KiB ✓；
B 槽 nTile×kL0Size×2 = 128×128×2 = 32KiB（恰满，现状即如此，P1 不改 kL0Size——kL0Size=576
曾实测 +2.9% 劣化，维持 128）。**L0C**：16×128×4 = 8KiB ✓。**UB（V 侧）**：不变
（kv_gather_ub 2×64×576×2 = 147,456B 等，总预算 170KiB 内，现行已验证）。

### 4.6 动态轴定义（不变）

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| seq_len (ΣS) / seq_len_kv (ΣS_kv) / batch+1 | T.symbolic | 任意 ≥ 0（一次编译覆盖） |

### 4.7 JIT 配置（不变 + P1 常量）

```python
@tilelang.jit(out_idx=[6], workspace_idx=[7, 8])   # 移除 Violations 形参后的 9 参签名
def tl_sparse_mla_topk_reducesum(heads, dim, tail_dim, topk, kv_group=1,
                                 sm_scale=None, block_I=32, grid_size=20): ...
# builder 内新增编译期常量：
KVDB = 2 if (2*BI*DDT*2 + 2*Hpb*DDT*2 + 4096 <= 523264) else 1
L1EV_BASE = 2   # MTE1_MTE2 pair id 2,3（pair 命名空间私有；id 0=gemm_v0 旧尾部/回退路径，
                # id 1 保留；与 gemm 内部 L0AB_EVENT∈{0,1} 在共享源管道 MTE1 上显式错开——
                # 防御 "(源管道,eventId) 共享" 解读下的任何别名可能）
```

---

## 5. Tiling 策略

### 5.1 计算类型

**混合（CV 融合）**：Cube gemm_v0（logits）+ Vector mask/exp/reduce——判定不变。

### 5.2 Block 划分（不变）

- topk 方向：BI = block_I（topk % BI == 0），NI = topk // BI；golden 配置 BI=128/NI=1。
- head 方向：H_per_block = padded_H（≤64）或 64（REPLICATE_H>1 分片）。
- K 方向：单 gemm K=DDT 整宽（合并布局，kL0Size=128 内部分 5 步）。
- 任务映射：persistent 轮询 ntask = #{k: k·grid+cid < total}，扁平块流 n = k·NI+i。

### 5.3 约束分析

- 对齐/容量：见 §4.5，全部满足（含 KVDB 回退）。
- ws 槽 DEPTH=4 与 L1 双槽 KVDB=2 的**配对关系**：二者独立（ws 槽管 V→C GM 中转的
  F0/F1 握手轮转；L1 槽管 C 核内 MTE2→MTE1 消费轮转）；P1 只动后者。

### 5.4 注意事项（尾块/边界）

- ΣS 不整除 grid：ntask 按核差异 + nblk=ntask·NI——C 循环界即 ntask（防悬挂不变式不变）。
- **P1 新增边界**：L1 token 生命周期在 nblk∈{0,1,2,3} 短流下的配平（§7.3 全 cases 验证）；
  KVDB=1 回退配置（BI=256 等）走旧语义（L0-G4 显式覆盖）。
- 非整除历史结论沿用：本算子无 M/N/K 运行时非整除（维度全编译期），varlen 由 offsets
  运行时分段 + clamp+mask 保护（不变）。

### 5.5 数据搬运性能可行性

沿用 generic_pipeline_analysis.md 实测口径（P1 不改任何 GM 访问形态/字节数——搬运成本
不变，仅改变**等待结构**使既有 MTE2 引擎占空比提升）：C 侧 ws 读 147KB/块@45.4GB/s/AIC
（贴 E1 dirty 天花板）；V 侧行 gather 44.4ns/行、bulk 61.8GB/s。无新增 GM pass/DMA。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 任务轴 | persistent 轮询 | `for k in T.serial(ntask)`（C/V 同界） | F0/F1 计数恒 nblk 1:1（不变式不变） |
| topk 块轴 | 编译期常量界 | `for i in T.serial(NI)` / V 侧 prologue `for s in range(LEAD)` | 不变 |
| 引擎级流水 | **手动**（非 T.Pipelined） | per-parity token（P1 新增） | Expert 模式显式控制；T.Pipelined 不支持本结构 |

### 6.2 P1 C 侧重构伪代码（核心交付）

```python
with T.Scope("C"):
    for k in T.serial(ntask):
        pid = k * grid_size + cid
        ...  # pid 算术（不变）
        for i in T.serial(NI):
            n = k * NI + i                       # 扁平块号（V 侧同域，不变）
            slot = n % DEPTH                     # ws 槽（F0/F1 轮转，不变）
            p = n % KVDB                         # L1 奇偶槽（P1 新增；KVDB=1 时恒 0）
            T.wait_cross_flag(F0_READY_BASE + slot)        # 等 V gather 完毕（不变）
            if KVDB == 2:
                T.wait_flag("mte1", "mte2", L1EV_BASE + p) # 获取 q_l1[p]/kv_l1[p] 写所有权
            if NI == 1 or i < 2:                 # Q 装载：每 task 首两块（NI=1 时每块）
                T.copy(Q[s_row_i, H0:H1, :], q_l1[p])      #   填本奇偶槽的 Q（跨 task 重写
                                                          #   由同一 token 保护；i≥2 复用
                                                          #   本 task 已填的同奇偶 Q）
            T.copy(ws_kv[cid, slot*BI:(slot+1)*BI, :], kv_l1[p])   # MTE2 填槽 p
            T.gemm_v0(q_l1[p], kv_l1[p], acc_s_l0c, transpose_B=True, init=True,
                      l1_release_event=(L1EV_BASE + p) if KVDB == 2 else None)
            #   ↑ KVDB=2：尾部在 MTE1 拷贝排空点发布 EV[p]（mma 之前）；不再发射旧 MTE2 门控
            #     KVDB=1：模板字符串不含新参 → 逐字节旧语义（回退路径）
            T.copy(acc_s_l0c, ws_acc[cid, slot*Hpb:(slot+1)*Hpb, :])   # FIX（不变）
            T.set_cross_flag("FIX", F1_ACC_BASE + slot)                # （不变）

# C scope 入口/出口（P1，KVDB=2 时；对照 example_gemm_intrinsic init_flag/clear_flag）：
#   入口： T.set_flag("mte1", "mte2", L1EV_BASE + 0)
#          T.set_flag("mte1", "mte2", L1EV_BASE + 1)     # pretend-free ×2（MTE1 空队列即时 fire，
#                                                        #   latch 语义消费——先例在库）
#   出口： T.wait_flag("mte1", "mte2", L1EV_BASE + 0)    # destroy：消费各奇偶最后一次释放
#          T.wait_flag("mte1", "mte2", L1EV_BASE + 1)    #   （ntask==0 时恰消费 init 的 fire，
#                                                        #    空核洁净退出，无悬挂）
```

V 侧（preamble/prologue/主循环/尾体）**逐条不变**。

### 6.3 流水线优化（P1 时序模型）

```
现行（每块 4.25µs，串行链）：release(n)@mma(n)排空 → MTE2 fill(n+1) 3.25 → MTE1(n+1) 0.71 → mma(n+1) → ...
P1（目标 3.55-3.7µs/块）：   MTE2 fill(n+1) ∥ MTE1(n) 0.71 + mma(n) 0.28（奇偶槽隔离 +
                             EV[p] 拷贝排空点释放）→ MTE2 引擎接近满占空比（3.25µs/块节拍）
```

### 6.4 尾块处理

nblk<2 的核/流：init/destroy 配平闭合（§7.3 cases nblk=0/1/2/3 全验证）；ntask==0 空核
C 循环零次 + destroy 消费 init fire——无半更新、无悬挂（§7.5 洁净性，对照历史 §7.5 论证）。

---

## 7. 同步策略

### 7.1 Cross-flag 协议（不变式沿用 + 两条 V 侧安全推导的 P1 下重锚定）

F0 = 0..3（V→C，set@MTE3，mode-2 quorum）、F1 = 4..7（C→V，set@FIX）——**8 id 恰满 A2 FFTS
合法预算，P1 零新增 cross-flag**。扁平块流上 set/wait 严格 1:1、全部位于有效块程序段的
不变式不变。V 侧两条曾引用"gemm_v0 尾部内建 MTE1→MTE2 排空"的推导，在 P1（opt-in 路径
无旧尾部）下**改锚**为经 FIX/M_FIX 链成立：

- **bulk(gn) 重写 ws_kv[slot] 安全**（原注释 L511-518）：需 F1(m) ⇒ C 对同 slot 上一读者的
  GM→L1 读（MTE2）已排空。P1 下：F1(m) 由 FIX(m) 发出，FIX(m) 排队在 gemm(m) 之后且经
  `SetFlag<M_FIX>` 门控（mma(m) 排空才放行 FIX）；mma(m) 经 M_MTE1/MTE1_M 协议必然晚于
  MTE1(m) 拷贝 ⇒ **F1(m) ⇒ mma(m) done ⇒ MTE1(m) 拷贝 done ⇒ C 侧 ws_kv slot 读毕**——
  推导链不经过被移除的旧尾部门控，P1 下依然成立。✓
- **C 重写 ws_acc[slot] 安全**（原注释 L519-523）：F0(n) ⇒ V 的 bulk(n) 前置 mte2→mte3
  marker 捕获 post(n−DEPTH) 的 acc 读 ⇒ 已排空——纯 V 侧链，P1 不触碰。✓

### 7.2 P1 per-parity L1 token 协议（新增，in-core MTE1_MTE2 pair，id 2/3）

**协议**（对照 api-schedule-sync.md §4 ownership 模式 + example_gemm_intrinsic 先例）：

| 阶段 | 操作 | 语义 |
|---|---|---|
| init（C scope 入口） | `set_flag("mte1","mte2",2)` / `set_flag("mte1","mte2",3)` | pretend-free：两奇偶槽初始可写（MTE1 空队列，即时 fire，latch 消费） |
| acquire（每块 n，fill 前） | `wait_flag("mte1","mte2",2+(n%2))` | MTE2 获得槽 p 写所有权——等**2 块前** gemm(n−2) 的 MTE1 消费排空事件 |
| release（gemm_v0 尾部，opt-in） | `SetFlag<MTE1_MTE2>(2+p)`（模板内，位于 mma 排空等待**之前**） | 本 gemm 全部 L1→L0 拷贝完成 ⇒ q_l1[p]+kv_l1[p] 可写（MTE1 队列内发布，fire 于拷贝排空点） |
| destroy（C scope 出口） | `wait_flag("mte1","mte2",2)` / `wait_flag("mte1","mte2",3)` | 消费各奇偶最后一次释放，token 生命周期闭合（graph replay 下事件状态确定性清零） |

**时序安全论证（fire/arm 裕量）**：稳态下 EV[p]（gemm(n) 发布）fire 于
fill(n) 落地 + MTE1(n) 拷贝 ≈ T+4.0µs（T=fill(n) 起点锚）；消费方 wait 位于 MTE2 队列中
fill(n+1)（另一奇偶，3.25µs）之后 ⇒ MTE2 执行到该 wait ≈ T+6.5µs——**arm 先于 fire 2.5µs
裕量**（执行时机 arming 模型下与 latch 语义无关地安全）。冷启动由 init pretend-free 覆盖
（fire 先于任何 arm——依赖 latch，**生产先例**：example_gemm_intrinsic/mha_sink 的
init_flag 即此形态且在库运行；历史"错位 marker 挂死"事故的根因更可能是 (源管道,id) 槽位
复用冲突，见 §7.6）。首日 spike S-P1b 在真机对协议原形做最终裁决。

### 7.3 事件生产/消费 1:1 配平表（每核、每奇偶 p，B_p = #{n<nblk : n%2==p}）

| 事件 | 生产（fire） | 消费（wait） | 计数 |
|---|---|---|---|
| EV[p]（MTE1_MTE2 id 2+p） | init ×1 + 每奇偶 p 的 gemm 尾部 ×B_p | 每奇偶 p 块的 acquire ×B_p + destroy ×1 | **1+B_p = B_p+1 ✓ 恒配平** |

边界 cases 逐项验证（配平 = sets − waits）：nblk=0（空核）：1+0 vs 0+1 ✓；nblk=1：
EV0: 1+1 vs 1+1 ✓，EV1: 1+0 vs 0+1 ✓（destroy 消费 init）✓；nblk=2：各 1+1 vs 1+1 ✓；
nblk=3：EV0: 1+2 vs 2+1 ✓，EV1: 1+1 vs 1+1 ✓；nblk≥4 稳态 ✓。**无 init 守卫、无 destroy
守卫、无 `if n≥2` 运行时条件**——比逐块条件守卫更简（编译器零新增循环界/条件负载）。

Cross-flag F0/F1 配平：不变（1:1 于扁平块流，历史 §7.2 表沿用）。

### 7.4 Buffer 生命周期状态机与写读保护配平表

每奇偶槽 p 状态环：`FREE --acquire(EV[p])--> FILLING --fill 入队--> FILLED --gemm 头部
MTE2_MTE1(0) 排空--> READING --gemm 尾部 release(EV[p])--> FREE`（周期 2 块）。

| 写 | 上一周期读 | 保护机制 | 结论 |
|---|---|---|---|
| kv_l1[p] @块 n | gemm(n−2) MTE1 拷贝 | acquire wait_flag(EV[p])（release 于拷贝排空点） | ✓ |
| q_l1[p] @task τ 块 i<2 | task τ−1 同奇偶块的 gemm MTE1 | 同一 acquire（release 覆盖 q+kv 双 buffer——单 gemm 单 release 语义） | ✓ |
| L0A/B 乒乓槽 @gemm(n+1) 拷贝 | mma(n) 同槽 | M_MTE1 既有协议（**未改动**，含跨调用头预置臂） | ✓ |
| acc_s_l0c @mma(n+1) | FIX(n) L0C→GM | FIX_M 既有头部协议（未改动） | ✓ |
| ws_kv[slot] @V bulk(n+DEPTH) | C 块 n 的 MTE2 GM→L1 | F1 链（§7.1 重锚推导） | ✓ |
| ws_acc[slot] @C FIX(n) | V post(n−DEPTH) MTE2 | F0 链（纯 V 侧，未触碰） | ✓ |
| V 侧全部 UB 复用 | — | 现行相邻对纪律（未触碰） | ✓ |

**覆盖未消费数据**：MTE2 fill(n+1) 写奇偶 (n+1)%2，与 MTE1(n) 正在读的 n%2 物理隔离；
Q 复用（i≥2 不重填）在 task 内无重写点；跨 task 由 acquire 保护。无覆盖窗口。

### 7.5 死锁分析（等待 DAG，无环）

```
scalar ──F0 wait──> V 的 MTE3 bulk（跨核）
MTE2 ──EV[p] wait──> MTE1（拷贝排空）          ← P1 新增边（核内）
MTE1 ──MTE2_MTE1 wait──> MTE2 head-set（fill 落地）
MTE1 ──M_MTE1 wait──> M（mma 排空，既有）
M ──MTE1_M wait──> MTE1（拷贝落地，既有）；M ──FIX_M wait──> FIX（既有）
FIX ──M_FIX wait──> M（既有）；V 侧链（未触碰）
```

新增边只落在 C 核内 MTE2→MTE1 方向，与既有边构成无环 DAG；唯一跨核依赖仍是 F0/F1（拓扑
不变）。标量推进由 F0 节流（DEPTH=4 槽，C 领先 V ≤ DEPTH−LEAD 块——上限结构与现行一致），
不存在 EV 等待环（EV[p] 的 fire 只依赖本核 MTE1 推进，MTE1 推进只依赖 head-set/F0 已放行的
fill）。**#22 尾体纪律继承**：全部 set/wait 位于 scope 入口/出口或每块语句段内，计数由
init/B_p 结构性配平（§7.3），无 phantom set/wait。

### 7.6 语义开放问题与证据权衡（首日 spike 裁决）

| 问题 | 支持证据 | 反向信号 | 裁决途径 |
|---|---|---|---|
| in-core 事件 fire-before-arm 是否 latch | example_gemm_intrinsic init_flag 即时 fire 被 3 块外的 wait 消费（在库运行）；gemm_v0 自身 M_MTE1 头预置臂跨调用 fire-then-arm 生产运行 | 历史错位 marker 实验 printf 实证挂死（但其同时存在 (源管道,id) 槽位复用，根因未隔离） | S-P1b：P1 协议原形最小 kernel + S∈{1,2,3,64,2048} × replay ×N + printf/dump 插桩 |
| in-core 事件 id 命名空间 | api-schedule-sync.md L196-200（per-pair）；本算子 V 侧 mte3→v/mte3→mte2 同 id=0 共存 | 历史实验注释"(源管道,eventId) 共享"字面读法 | L1EV_BASE=2 显式避开 MTE1 源上 gemm 内部 {0,1}——**两种读法下均无冲突**（防御性消解，spike 顺带确认） |

**降级阶梯**（S-P1b 证伪 latch 时）：① P1-lite——保持单槽 kv_l1，仅用 opt-in 事件把释放点
从 mma 排空提前到拷贝排空（set@gemm(n) 与 wait@fill(n+1) 发射距离 1 块，稳态 arm 裕量 =
MTE1 拷贝时长 0.71µs，与文档化 ownership 模式同级安全）；预期 C 链 4.25→~3.95µs（约半收益）。
② 放弃框架改动，先行落地 P2/P2b（算子本地、spike 已验证），C 侧优化重议。两级降级均不动
V 侧、不动语义。

---

## 8. CV 融合设计与移除设计

### 8.1 融合算子判定

**是**（Cube gemm_v0 + Vector mask/exp/reduce，Expert 显式 workspace 中转——混合模式回退
写法，与 Developer 默认消除 workspace 的路径无关，历史版本既定）。

### 8.2 workspace 规格（不变）

| workspace | Shape | dtype | 用途 |
|-----------|-------|-------|------|
| ws_kv | [grid, DEPTH·BI, DDT] | bf16 | V→C gather 中转（每核 DEPTH=4 槽） |
| ws_acc | [grid, DEPTH·Hpb, BI] | fp32 | C→V score 中转（DEPTH=4 槽） |

`workspace_idx=[7, 8]`（9 参签名移除 Violations 后；与现行 unguarded strip 后一致）。

### 8.3/8.4 Cube/Vector 核计算流程

见 §6.2（C 侧 P1 伪代码）与现行 V 侧（逐条不变）。

### 8.5 pass_configs 配置

不设置（全手动，不变）。

### 8.6 移除设计——精确范围清单（交付核心 A）

**kernel 侧（sparse_mla_topk_reducesum.py，1803 行 → 约 850 行）**：

| 对象 | 行号（现行） | 处置 |
|---|---|---|
| `_reducesum_panel_body`（含全部 panel 协议/组调度/契约校验链） | L849-1286 | **删除** |
| `tl_sparse_mla_topk_reducesum_panel` / `_panel_guarded` | L1289-1366 | **删除** |
| `tl_sparse_mla_topk_reducesum_detect`（检测 kernel） | L1392-1489 | **删除** |
| `tl_sparse_mla_topk_reducesum_guarded` + `_guard_run`/`_guard_v_entry` + `guarded` 形参 | L770-808, L226-247 | **删除**（generic builder 回到 rev1 单形态） |
| `_strip_unused_param` | L74-98 | **删除**（Violations 形参随 guarded 机制一并消失） |
| `_panel_gate_ok` / `_PANEL_V1_MAX_TOPK` / `_token_indices_from_list` / `_replicate_h_of` | L1557-1612 | **删除**（后者 rs 预分配专用，generic 经 out_idx 由框架分配） |
| `panel_mode` 形参与三模式分发（auto 三连发/shared 门控/静默回退） | host L1615-1803 | **删除**；host 收缩为：shape 推导 → `_BUILD_CACHE` 查建 → `prepare_token_indices` → 调用（rev1 形态） |
| `_BUILD_CACHE` 键 | 现行 5 族 | **收缩为 1 族**：`(heads, dim_v, tail_dim, topk, kv_group, sm_scale, block_I, grid_size)` |
| `_reducesum_generic_body` | L104-728 | **保留 + P1 改造**（§6.2）：移除 Violations 形参/guard 闭包；签名 9 参（Q..ReduceSum, ws_kv, ws_acc）；`out_idx=[6], workspace_idx=[7,8]` 与现行 strip 后逐值一致 |
| 模块 docstring / `__all__` | L1-71 | 同步收缩（移除 panel/detect/guarded 导出） |

**TIR 等价性论证（移除 ≠ 改动）**：现行 unguarded 变体的 traced 语句与 rev1 逐条一致为
既有 spike S1 实证（文件头 L43-47）；guarded=False 时 `_guard_run` 返回 Python True
（`if True:` 编译期直通，零 TIR 语句）、`_guard_v_entry` 原样返回 `ntask > 0`——删除该
机制后的 traced body 与现行 unguarded 路径**逐语句一致**，即与 git 0f11b3cc 内的 rev1 等
价 generic 路径 TIR 等价。验证：移除 commit 上对 golden 配置 dump prim_func（`tvm.script`
/ ascii）与基线 diff 为空 + L0-G 全绿。

**host 侧移除后完整路径 [HOST-METADATA-AUDIT]**：

| operation | 性质 | 判定 |
|---|---|---|
| `prepare_token_indices`（offsets.cpu().tolist → CPU torch.full/arange 构造 → 一次 H2D） | 数据准备类（新建张量，不改用户 buffer） | allow（不变） |
| `_BUILD_CACHE` 查建 / kernel 调用 | 调度 | allow |
| ~~auto/shared 的 offsets D2H 门控、violations/err_flag torch.zeros、rs torch.empty 预分配~~ | — | **随移除消失**（host 反而更净） |

**测试侧（test_sparse_mla_topk_reducesum.py 重写）**：

| 现行用例 | 处置 |
|---|---|
| L0-P1（panel golden）/ P3（违约检测）/ P4b panel 直调段 / P4e（G4 miss）/ P6（门控单元）/ P7a-g（detect/auto/guarded 全族） | **删除** |
| L0-P2（generic golden 回归） | 改写为 **L0-G2**（golden arange 回归，GPU 原版参数逐字） |
| L0-P4a/b/c/d（空首段/单 task/-1 尾/OOB） | 改写为 L0-G3/G4 成员（generic 单路径） |
| L0-P5（竞态压测） | **保留并加强**为 L0-G5（P1 动了 C 侧同步结构，竞态面强制重压 + 短流/回退扩展） |
| `auto_route_probe`/`panel_builder_probe`/`_cache_key_kind`/cache-key 断言族 | 删除 |
| L1 层 `run_shared`/`run_auto` 通道与 `panel_mode=` 全部传参 | 删除；L1 主索引构造从 arange 升级为**随机逐行独立**（主）+ 分布族 |
| COVERAGE_MANIFEST `D-PARAM-panel_mode: 2` | **删除该维度**（attr 已不存在）；其余维度保留 |
| Boundary 层 `panel_mode="generic"` 传参 | 去参（签名变更同步） |
| L2 层（dtype/topk/head_kv 拒绝） | 保留（全 generic，无 panel 专属项） |

**移除后基线锚定（perf-neutral 验证标准）**：移除 commit 上 random BI128（S=2048/seed=123）
graph-replay 配对 A/B（vs 基线 0f11b3cc 同配置 generic 构建或同会话先后构建）：期望
**≈441.7µs（跨会话 ±3% 带内；同会话配对 |Δ|≤0.5%）**——移除是零性能变化的纯收缩。

---

## 9. 验证方案

### 9.1 Golden 函数（PyTorch 参考实现——不变，沿用现行文件移植版）

```python
def ref_mla_topk_softmax(Q, K, TopkIndices, offsets):        # 基础（arange 族）
def ref_mla_topk_softmax_masked(Q, K, TopkIndices, offsets, sm_scale=None)   # mask 族
def ref_mla_topk_softmax_grouped(Q, K, TopkIndices, offsets, kv_group, ...)  # GQA 族
# 纯 torch：fp32 softmax、per-batch varlen 循环、causal mask、gather、renormalize；
# 返回 (lse, ref_attn_score)。与 GPU 主仓 ref_mla_topk_softmax 逐字一致（现行已移植验证）。
```

随机主用例的索引构造（CPU 完成后一次 H2D，aclnn 审计 allow）：

```python
def make_random_causal_indices(S, offsets_list, topk, seed):
    """逐行独立随机 causal TopK：每行从 [0, s_i] 内无放回采样 topk 个（s_i+1 < topk 的行
    采样补 -1 尾填充——天然覆盖 -1 边界）；行间零共享、零连续性假设。seed=123 与
    generic_pipeline_analysis random_BI128 锚点同源。"""
```

### 9.2 L0 门槛测试计划（tilelang-op-test-design 场景 A 产出；仅 L0，L1/L2/Boundary 由 Stage 2 场景 B 扩展）

> 判定输入：设计文档 §1.3/§1.4/§4/§5。判断结论：计算类型 = **混合（CV 融合）**，复杂度 =
> **Fusion**，数学特征 = matmul+exp+mask+reduce，综合类别 = **Fusion**。dtype 为固定组合
> （无 dtype 扫描维度），测试策略以 **shape/topk/分段/索引分布** 扫描 + 竞态压测为主。
> 测试重点/数量/特殊场景决策由任务书给定（随机索引为主、Golden 回归保留、竞态强制），
> 交互环节按"已提供跳过"规则省略。

| 用例名 | 级别 | Shape / 配置（全部规则 shape：topk%BI==0） | dtype | 索引分布 | 判据 |
|--------|------|---------------------------------------------|-------|----------|------|
| **L0-G1 随机独立索引主用例** | L0 | S=2048/H=16/D=512/tail=64/topk=128/**BI=128**/offsets=[0,1023,2048] | 固定组合 | **逐行独立随机 causal（seed=123，同 perf 锚点）** | golden 判据 + §9.3 混合容差（正确性主锚点；同配置即性能主评估对象，perf 通道见 §9.5） |
| L0-G2 Golden 样例回归 | L0 | 同 shape，seed=42（GPU 原版逐字），BI∈{32,128} 双跑 | 固定组合 | arange(128) 全同共享（旧 golden 构造**仅作回归**，禁止任何 kernel 侧内容特化） | golden 判据 + 混合容差 |
| L0-G3 索引分布族 | L0 | S=2048 同 shape | 固定组合 | ①全同（共享向量）②部分重合（行簇共享前缀 + 独立尾部）③非连续乱序（stride/permute 构造）④边界值（-1 尾行 / OOB 值 / idx==s_i / idx==0 / 全 -1 行） | 各 1 例，golden 判据 + 混合容差 |
| L0-G4 shape/topk/分段扫描（规则子集） | L0 | S∈{64,512,2048} × topk∈{32,64,128} × BI∈{32,64,128} 代表组合；offsets 单段/[0,1023,2048]/四段/**[0,0,1023,2048] 空首段**；kv_group∈{1,2}（heads∈{16,32}）；含 **topk=256/BI=256（KVDB=1 回退路径）≥1 例** | 固定组合 | 随机 causal | 每例 golden 判据 + 混合容差 |
| L0-G5 竞态压测族（P1 强制） | L0 | ①ΣS=64 × D∈{128,256} × topk∈{32,64} × **3 独立运行**；②短流 nblk∈{1,2,3}（S=1/21/41 类，单/双/三块核流——token 生命周期边界）；③BI=256 回退路径 ×3 | 固定组合 | 随机 causal | 3-run **两两 torch.equal**（逐位稳定）+ 每 run vs ref |
| L0-G6 移除锚定 + P1 消融 | L0-perf | random BI128 S=2048 seed=123 | 固定组合 | 随机 causal | §9.5 协议（性能，非精度门） |

覆盖维度（供 Stage 2 场景 B 扩展时的基线）：D-DTYPE-{bf16,fp32,int32}（固定组合域内；
域外 dtype 维持 L2 拒绝）、D-SHAPE-{ALIGNED,TAIL-1,TAIL-MID,PRIME,EDGE}、D-VALRANGE-{S,M,L,ASYM}、
D-PARAM-{dim_v,kv_group,sm_scale,block_I,grid_size}（panel_mode 已删）、D-SPECIAL-*、D-EXC-*。

### 9.3 精度标准（混合容差；按 dtype；与历史逐字一致——语义未变）

| 数据 | dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|---|---|---|---|---|---|
| ReduceSum（唯一输出） | float32 | 1e-4 | 1e-2 | **1e-3** | 0.99 |
| Q/KV 输入（无 bf16 输出；经 fp32 L0C 累加，比较恒作用于 fp32 输出） | bfloat16 | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |
| Indices/Offsets/TokenIndices 输入（无整型输出；若比较任何整型量按精确匹配） | int32 | 0 | 0 | 0 | 1.0 |

辅助全局判据（继承 golden 任务书，与上表联合生效）：err_ratio = RMS(err)/RMS(ref) ≤ 1e-2、
NaN 计数 = 0。特殊值（±Inf/NaN）比较的位置敏感契约：先严格比较 isnan/±Inf mask 全一致，
再仅对双有限值位置套用混合容差（Boundary 层沿用现行实现，禁止全 NaN/Inf 唯一用例）。

### 9.4 性能可行性哨兵（强制，不可跳过）

| 用例名 | Shape | dtype/属性 | 覆盖路径 | 单 case timeout | 选择理由 |
|--------|-------|------------|----------|-----------------|----------|
| perf_sentinel_random_bi128 | S=2048/H=16/D=512/t=64/topk=128/BI=128 随机 causal | 固定组合 | KVDB=2 主路径（最大 DMA/块 + 双槽 L1 + 全 token 链） | 300s（含编译） | 主评估对象；P1 消融 A 面 |
| perf_sentinel_fallback_bi256 | topk=256/BI=256/S=512 | 固定组合 | KVDB=1 回退路径（L1 最大单槽） | 300s | 回退路径性能不劣于现状的守门 |

### 9.5 消融协议（P1 验证——AIC+AIV 双侧每块耗时 + 端到端三项全报）

**测量方法（一致口径）**：graph-replay 配对 A/B 交错 2 组（会话内 σ≤0.12% 判有效；跨会话
漂移 ±2-3% 只作参考）+ msprof op 深度档位（`--aic-metrics=PipeUtilization,Memory,L2Cache,…
`；MemoryL0/UB/Occupancy 真机空表时如实降级——现行已知限制）+ 节拍模型重构（无
TimelineDetail，同 generic_pipeline_analysis §0 方法论）。

| 组 | 配置 | 报告项（**缺一即判协议违规**） |
|---|---|---|
| A：移除后基线 | random BI128（§9.4 sentinel 1） | 端到端 wall（≈441.7µs 锚定）；AIC/AIV 每核 wall + **每块节拍**（≈4.25/4.26µs）；AIC MTE2/MTE1/Cube/FIX 占用、scalar_mte1_stall；AIV MTE2/MTE3/vec 占用、F1 wait、引擎和/墙钟 |
| B：P1 后 | 同配置同 seed | 同上全部 + **Δ 逐项**；验收门：AIC MTE2 占空比 76%→≥90%、scalar_mte1_stall 91%→显著下降、AIC 每块 ≤3.8µs；**AIV 每块与端到端必须同时报告**（禁止只报 AIC 局部提速） |

**诚实预期与判定框架**：P1 单独端到端 ≈ 441.7→**437-441µs（0-2%）**——V 侧 4.26µs/块自封
顶（共门控，generic_pipeline_analysis §3/§5 模型）；本阶段 P1 验收 = **C 侧使能项达标**
（上表 B 验收门），端到端移动不作为 P1 单独的验收条件。若 P1 后端到端意外移动 >5% → 共门控
模型被证伪 → 触发联合瓶颈分析（见下）。**联合瓶颈分析框架（P1 单独不足时的后续决策依据，
禁止盲目叠加 P2/P2b）**：以 msprof 复测 AIV 侧——V 引擎和/墙钟 ≈1.0（MTE3+vec 交替零重叠）
确认为 P2b 方向（LEAD=3 前移 gather 发射点）+ P2（SIDE_PAIR=False，spike 已验证）的组合
候选；两侧引擎占空比数据先行，逐项归因后再立项。

### 9.6 全量回归范围（移除 + P1 交付前）

1. L0-G1..G5 全绿（阻塞）；L2/Boundary（Stage 2 场景 B 重生成后，非阻塞）。
2. **竞态压测全族**（L0-G5——P1 变更 C 侧同步结构的强制补偿测试）。
3. 框架回归：S-P1c（gemm_v0 既有用户字节级不变）+ examples/gemm 代表用例运行。
4. 移除锚定（§8.6 末条）+ P1 消融（§9.5）双 perf 报告落 perf_tuning/。
5. graph replay 下事件状态复用验证（S-P1b 含 ×N replay——destroy 配平闭合的直接证据）。

---

## 10. 与失败方案 R2 的区别论证（强制章节）+ 风险与预案

### 10.1 R2 失败复盘与本设计的逐条区别

**R2 = 手工 gemm 流水**（npu_gemm/T.mma + unit_flag 自管同步）：单核微基准 −44%，生产
+12.7% 回退。两条根因：① chunk 化 ws 填充引发**生产者邻近脏 burst 惩罚 +130µs**（V 侧
ws_kv 写形态被碎片化，破坏单条 73.7KB@61.8GB/s bulk 写，且与 C 侧读流争 L2 dirty 服务）；
② **结构三元互斥**（三把锁互相嵌套等待）。

| 维度 | R2（失败） | P1（本设计） | 为何不触发 R2 根因 |
|---|---|---|---|
| V 侧 ws 填充形态 | **改为 chunk 化多次写** | **零改动**（V 侧全部语句逐条不变；bulk 仍单条 73.7KB） | 根因 ① 的载体（写形态）不存在——C 侧读流面对的 L2 dirty 模式与现行完全一致 |
| GEMM 实现 | 手写 npu_gemm/T.mma 全流水 | **保留 gemm_v0 模板**（同一 mma、同一 L0A/B 乒乓、同一 M_MTE1 槽协议、同一分形拷贝）；只把尾部 MTE1_MTE2 的释放粒度从"全队列"改为"per-parity"、opt-in 地去掉冗余 MTE2 自门控 | 计算内核零替换——R2 的手工流水排布风险（L0A/B 乒乓行为变化）不引入；唯一残留类比风险（MTE1 发射时序变化影响乒乓）由 msprof MTE1/Cube 占用对比 + 竞态压测覆盖（§9.5/§9.6） |
| 同步结构 | 新增 unit_flag 互锁 → 三元互斥 | **净减少**一处无条件门控（旧尾部 WaitFlag）；新增的 per-parity token 是 1:1 配平的 ownership 传递（§7.3），等待 DAG 无环（§7.5） | 根因 ② 的结构（互斥环）不出现——P1 是"松绑"而非"加锁" |
| GM 访问形态 | 变化（chunk 写） | **零变化**（同 copy、同序、同字节） | 中转税总量（random 下 C 侧 332µs 读引擎时间）不变——P1 只提流水效率，不碰总量 |
| 收益模型来源 | 单核微基准 | 生产 msprof 引擎占空比 + 节拍模型（generic_pipeline_analysis §1/§3） | R2 的"微基准不迁移"陷阱由消融协议直接在生产配置上验证规避（§9.5 A/B 同配置同 seed） |

**四路已判死方向不重开**（R2 手工流水 / R5 AIC 直采 ND→zN 100ns/行 / 反转 Q（Q 重载墙 +
202ns 散射原子）/ L0C→UB fixpipe（A2 无 API））——本设计的全部收益来源限定在"调度级流水
效率"（generic_pipeline_analysis §5 的 P1 行），与四路正交。

### 10.2 风险表（top 风险 + 全量）

| # | 风险 | 等级 | 缓解 |
|---|---|---|---|
| R1 | in-core 事件 latch 语义证伪（S-P1b 挂死） | 高 | §7.6 证据权衡（两生产先例）+ S-P1b 真机裁决 + 降级阶梯（P1-lite 半收益 / 转 P2+P2b 先行） |
| R2' | per-slot 释放改变 MTE1 发射时序 → L0A/B 乒乓回归（R2 教训的对应面，分析文档 §7.4 明示） | 中 | msprof MTE1/Cube 占用前后对比 + L0-G5 竞态压测 + 精度全绿三重门 |
| R3 | KVDB=1 回退路径（BI=256 等超 L1 预算配置）未被测到 | 中 | L0-G4 显式 ≥1 例 + §9.4 sentinel 2 + 编译期 assert 记录回退原因 |
| R4 | 框架 opt-in 泄漏影响既有 80 个 gemm_v0 用户 | 低 | 模板字符串不变 + C++ 默认参双保险；S-P1c 字节 diff + examples/gemm 运行回归 |
| R5 | msprof 档位限制（MemoryL0/UB/Occupancy 空表、无 TimelineDetail） | 低（已知） | 节拍模型重构法（同 generic_pipeline_analysis §0），如实降级标注 |
| R6 | 编译时间回归（新语句引发 CanonicalSimplifier 敏感） | 低 | P1 零新增循环界表达式（§3.5.1 对照 guarded 教训）；CI 监控编译时长 |
| R7 | graph replay 跨调用事件状态残留 | 低 | destroy 配平闭合（§7.3）+ S-P1b replay ×N 验证 |
| R8 | 移除遗漏（死代码/测试引用残留） | 低 | `__all__`/import 全仓 grep + L0 重写后全绿 + mypy 级自查（ruff F401/F811） |

### 10.3 首日 spike 清单（框架/协议风险前置消解）

| ID | 内容 | 通过标准 | 半天预估 |
|---|---|---|---|
| S-P1a | IR 可表达性：`T.gemm_v0(..., l1_release_event=2)` 模板字符串渲染 + 编译通过（前端→codegen→C++ 实例化全链）；含 C-scope `T.wait_flag("mte1","mte2",id)` lowering 正确（AIC MTE2 队列） | 最小 gemm 例编译运行且结果正确 | 0.5d |
| S-P1b | token 协议原形：双槽 L1 + init/destroy + per-parity acquire/release 的最小 kernel，S∈{1,2,3,64,2048}（短流边界）× replay ×N + printf/dump 插桩 | 全部运行正确、无挂死、replay 稳定 → latch 裁决；挂死 → 降级阶梯 | 1d |
| S-P1c | 框架回归：既有 gemm_v0 调用（examples/gemm/example_gemm.py）生成 C++ 与改动前 diff | 字节级一致（或语义等价的空 diff） | 0.5d |
| S-P1d | 性能冒烟：golden 配置 random BI128 + P1 → msprof AIC 每块节拍 | 信号确认（C MTE2 占空比明显上升 / mte1_stall 下降）后再进入全量消融 | 0.5d |

### 10.4 回退预案

1. **P1 正确性问题**：`T.dump_tensor` + `T.printf` 分段定位（L1 双槽内容 / token 时序）；
   单点降级 = builder 内 KVDB 置 1（编译期常量开关，kernel 回旧语义，框架改动可独立保留）。
2. **P1 性能退化**（消融 B 组劣于 A）：msprof 对比归因（MTE1 乒乓回归嫌疑优先）→ 10.4.1
   降级或回退；禁止未归因即叠加其他优化。
3. **完全回退**：移除与 P1 分独立 commit（先移除[锚定 L0-G6-A] 后框架 opt-in 再 P1），
   `git revert` 到移除后基线即可（历史全程在 git 0f11b3cc + 新 commit 链上可溯）。
4. **框架改动回退**：opt-in 性质保证 revert ascend.py/common.h 两文件即恢复，无下游残留。

---

## 11. 交付清单

### 11.1 目录结构（不变）

```
examples/dsa_sparse_finetune/
├── sparse_mla_topk_reducesum.py      # 移除 + P1 重构（Stage 2）
├── test_sparse_mla_topk_reducesum.py # L0-G 家族重写（Stage 2；L1/L2/Boundary 场景 B 扩展）
├── proto.yaml                        # 本版更新（panel_mode 移除）
├── DESIGN_generic_p1.md              # 本文档（现行设计）
├── DESIGN_panel_mode.md              # 历史保留（git 0f11b3cc 同源，不再维护）
├── history_version/                  # 历史实现备份（不变）
└── perf_tuning/                      # 消融报告追加（Stage 2/3）
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN_generic_p1.md` | ✅ 本文档 | 移除 + P1 设计 |
| `proto.yaml` | ✅ 本版更新 | panel_mode attr 删除；dtype 全集 = §9.3（bfloat16/float32/int32 固定组合）；attr = dim_v/kv_group/sm_scale/block_I/grid_size |
| `tilelang/language/ascend.py` | ⬜ Stage 2 | gemm_v0 l1_release_event opt-in（S-P1a） |
| `src/tl_templates/ascend/common.h` | ⬜ Stage 2 | gemm_v0 模板尾部分支（S-P1a/S-P1c） |
| `sparse_mla_topk_reducesum.py` | ⬜ Stage 2 | §8.6 移除清单 + §6.2 P1 改造 |
| `test_sparse_mla_topk_reducesum.py` | ⬜ Stage 2 | §9.2 L0-G 家族 + §9.5 消融 harness |
| perf_tuning/p1_ablation_report.md | ⬜ Stage 2/3 | §9.5 A/B 双组全项报告 |

### 11.3 实现顺序（commit 切分保证可独立回退）

1. ✅ 设计文档 + proto.yaml（本文件）
2. ⬜ **commit-1 移除**：§8.6 kernel/host/测试清单 + TIR 等价 diff + L0-G1..G5（P1 前）全绿
   + 基线锚定 441.7µs
3. ⬜ **commit-2 框架 opt-in**：ascend.py + common.h + S-P1a/S-P1c（spike 转正式）
4. ⬜ **commit-3 P1 算子**：§6.2 改造 + S-P1b 转正式 + L0 全绿 + §9.5 消融报告
5. ⬜ Stage 2 场景 B：L1/L2/Boundary 扩展 + 覆盖门禁 `coverage_check.py --proto`
6. ⬜ 后续（另行立项）：P2（1 行）/ P2b（LEAD=3）——依据 §9.5 联合瓶颈分析框架决策

### 11.4 [DISPATCH-COVERAGE] 审计

```text
supported_domain: dtype 固定组合（Q/KV=bf16, Lse/ReduceSum=fp32, Indices/Offsets/TokenIndices=int32）；
  dim/tail_dim 为 2 的幂；topk % block_I == 0；block_I 偶数且 >= 32；head_kv ∈ {16..64 的 2 的幂} ∪ {64 的倍数}；
  padded_H != head_kv 时 kv_group == 1；ΣS/ΣS_kv/batch 运行时任意；topk_indices 值域 [-1, ΣS_kv) 任意分布
generic_fallback: 唯一 generic kernel 即全域路径（无分发）
specializations:
  - predicate: 无（KVDB=1/2 为编译期 L1 容量分支，两分支语义等价、同一 kernel 形态，非数据分发；
      回退分支由 L0-G4 sentinel 显式覆盖）
    fallback_on_miss: n/a（容量分支双臂均 tested）
    equivalence_evidence: KVDB=1 臂 = 移除后基线逐字节（gemm_v0 不传 kwarg）
unsupported_inputs: 域外 dtype/shape（L2 拒绝，D-EXC-* 维持）
result: pass
```

### 11.5 算子 proto.yaml（覆盖门禁用，本版产出）

已同步更新至 `examples/dsa_sparse_finetune/proto.yaml`：`attrs` 移除 `panel_mode`（保留
dim_v/kv_group/sm_scale/block_I/grid_size 五项）；`inputs[].dtype` 与 §9.3 精度表 dtype 全集
一致（固定组合，无 dtype 扫描维度）；schema 签名移除 panel_mode 形参。
