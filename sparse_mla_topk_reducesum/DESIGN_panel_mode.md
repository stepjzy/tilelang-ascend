# DESIGN_panel_mode — sparse_mla_topk_reducesum 共享索引面板（panel-mode）快速路径变体设计

日期：2026-10-10 ｜ 模式：design_revision（**rev2**——分发策略修订：方案 A 调用方声明 → 方案 B 自动检测为缺省）
基线：kernel 终态 #19-p2+#22 + panel 变体已实现并验收（`sparse_mla_topk_reducesum.py`，1351 行，**只读参考**）
rev1 备份：`history_version/design_panel_mode_rev1_planA.md`（方案 A 版设计，535 行；panel 143.3us PASS 220us 线的验收依据）
直接依据：`perf_tuning/golden_roofline.md` §5.2（方案蓝图）+ `perf_tuning/panel_metric2_bench.md`（panel 143.3us / generic BI128 378.9 / BI32 471.4，graph-replay 主口径）
语义声明：**语义零变化，纯分发机制重构**。数学公式与现 kernel 完全一致——
`ReduceSum[s,g,r,j] = Σ_h exp((Q[s,h]·KV_main + Q_tail·KV_tail)*sm_scale - Lse[s,h]) * causal_mask(idx<=s_i 且 idx!=-1)`
任意输入、任意 `panel_mode` 取值下输出正确性不变（§8 三模式语义表）；panel-vs-generic 逐位相等（torch.equal，rev1 L0-P1 实证）继续成立。

---

## 0. 概述

### 0.1 问题与机会（rev1 结论，不变）

golden_metric2 实测（graph-replay 主口径）：BI=128 = **364.7us（vs 220us 判定线 FAIL +65.8%）**。roofline 归因：
G1 的 **90%（328us）= C 侧 ws_kv 302MB GM 往返**——通用索引架构下每个 (task, topk 块) 都要把
gather 好的 KV 行经 `ws_kv` GM 中转给 C，该成本与数据形态无关、架构强制。

golden 数据（arange 全行重复索引）满足一个更强的索引模式契约：**对每个 (sequence, g)，所有行共享
完全相同的索引向量 P**。此时每个 (seq, g) 只有一个 128 行的共享 KV 面板（panel），把它一次性载入
L1 常驻，C 流对每个 task 做一次小 dense GEMM `Q[16,576] × panel[128,576]^T`，即可**完全跳过
ws_kv 302MB 往返**（面板 staging 仅 ~11.8MB 全核总量，51× 缩减）。

**实测终值（panel_metric2_bench.md，2026-10-09）**：panel 路径 graph-replay 2 组均值 **143.3us
（141.8-144.3，漂移 0.3%）→ 220us 判定 PASS（余量 1.53×）**；msprof 交叉 147.1us（比值 1.027 ✓）。

### 0.2 交付边界与不变量（rev2 修订）

- 同文件（`sparse_mla_topk_reducesum.py`）新增：**检测 kernel builder**（`tl_sparse_mla_topk_reducesum_detect`）、
  **guarded 变体 builders**（generic/panel 各一，见 §3.5 族谱）、host 分发扩展；**unguarded 通用路径与
  unguarded panel 路径的 builder 公开名/签名/装饰器零修改**（源码以共享 body 函数重组，guarded=False 的
  traced 语句序列与现版逐条一致 ⇒ TIR 等价，P2/P1 回归锁定，§8.2）。
- 输出接口不变：`ReduceSum [ΣS, kv_group, REPLICATE_H, topk] float32`；接口层归一化仍在交付边界外。
- **正确性硬约束（本 rev 核心验收线）**：任意输入自动选路结果必须正确——panel 合格 → panel（与 generic
  torch.equal 已实证）；不合格 → generic。**检测错误 = 沉默错误结果**，该风险由三层机制锁死（§8.2 裁决、
  §9.2 L0-P7 测试族、§11 R9 分层防御论证）。
- `panel_mode` 缺省改为 **`"auto"`**（自动检测 + 推测三连发）；`"generic"` / `"shared"` 保留为显式覆盖
  （向后兼容：语义结果恒等，性能差异见 §10.4）。

### 0.3 rev2 修订范围（相对 rev1）

| 项 | rev1（方案 A） | rev2（方案 B 落地为 auto 缺省） |
|---|---|---|
| 契约来源 | 调用方声明 `panel_mode="shared"` + kernel 内逐任务校验兜底 | 设备侧检测 kernel（§1.5/§3.4）+ 守卫式双发射（§8.2）+ 逐任务校验链保留为纵深防御 |
| 缺省行为 | `"generic"`（与既往版本逐字节一致） | `"auto"`（元数据门控 miss 时仍为零开销单发，§8.3） |
| 通用路径改动 | 零 | 新增 guarded 变体（unguarded 零修改；守卫 = ntask 钳位一行/scope，§8.2） |
| host 同步 | 零 | 零新增（off_list 复用使 D2H 次数与旧默认路径持平，§8.3） |
| 性能（panel 合格输入） | 143.3us | ≈ 155-157us（+detect+guarded-generic 空载，§10.4，PASS 余量 ~1.4×） |

rev1 已实证内容（panel 架构、L1 内存规划、8-flag 协议、组调度、等价性、L0-P1..P6）**全部保留不推翻**；
rev2 为纯增量修订（§3.4/§3.5、§4 增补、§7.4/§7.5、§8 重写、§9.2 增补 L0-P7、§10.4、§11 增补）。

---

## 1. 模式契约（特化触发条件，精确定义）

### 1.1 契约定义（不变）

对每个 (sequence b, kv_group g)：所有行 s ∈ seq(b) 的索引向量逐元素相等（含顺序）：

```
∀ s, s' ∈ seq(b):  Indices[s, g, 0..topk) == Indices[s', g, 0..topk)   （记公共向量为 P[b,g]）
```

panel 定义：`panel[b,g] = KV[bos_b + P[b,g][j], g, :] for j in 0..topk`（topk × DDT 行集，
P 含 -1/越界值时行号 clamp 到 [0, ΣS_kv-1] 仅保护寻址，贡献由 causal mask 置 0——与现 kernel
§5.4 语义一致）。causal mask（`P[j] <= s_i && P[j] != -1`）**逐 task 逐列**计算，与共享与否
无关——契约只约束索引描述符，不改变逐行 mask 语义。

### 1.2 panel 预算门控（容量实算，不变，全部经出处核实）

| 存储单元 | 容量 | 出处 |
|---|---|---|
| L1 | 524,288 B | `tilelang-api-best-practices/references/api-kernel-memory.md` §2（出处 `$ASCEND_HOME_PATH/.../Ascend910B3.ini`）；`examples/attention_sink/.../example_gqa_sink_fwd_varlen.py` L201 注释交叉验证 |
| L0A / L0B | 65,536 B 各 | 同上 |
| L0C | 131,072 B | 同上 |
| UB | 196,608 B | 同上（现 kernel L162 注释 `170*1024` 预算同源） |

**每核 L1 panel 足迹（双缓冲 2 槽 + q_l1）**：

```
footprint = 2 × topk × DDT × 2B  (panel_l1 × 2)  +  H_per_block × DDT × 2B  (q_l1)  +  对齐余量
golden (topk=128, DDT=576, H=16): 2×128×576×2 + 16×576×2 = 294,912 + 18,432 = 313,344 B ≈ 306KB ≤ 512KB ✓（60%）
```

通式门控（host 侧纯元数据计算）：

```
G1(容量):  2×topk×DDT×2 + H_per_block×DDT×2 + 4KB 余量 ≤ 524,288
G2(L0B):   topk × kL0Size × 2 ≤ 65,536     # kL0Size=128 → topk ≤ 256（gemm_v0 K 分块 staging，
                                            #   tilelang/language/ascend.py L444-454 kL0Size 语义）
G3(L0C):   H_per_block × topk × 4 ≤ 131,072  # 16×topk×4 → topk ≤ 2048
```

golden 配置三点全过（G2: 128×128×2=32KB ✓，G3: 8KB ✓）。v1 域取保守值 **topk ≤ 128**（golden 域；
topk ∈ (128,256] 需 kL0Size=64 + 单缓冲 panel 变体，列为后续扩展，§11）；topk > 128 → 回退通用路径。

### 1.3 任务→核映射分析与摊销门控（不变）

沿用现 kernel 的 **persistent 轮询（striped）映射**：`pid = k×grid_size + cid`，`ntask = ceildiv(total−cid, grid)`。
沿每核任务路径，(g, b) 键**字典序单调**（pid 升 → g 不减；g 内 s_row 不减 → b 不减，packed 布局保证），
故 panel 按序装载、无乱序回跳。

摊销门控（host 侧可算，offsets 在 host 分发中本就 `.cpu().tolist()`，零额外同步）：

```
G4(组数):  B × kv_group ≤ 8                     # 轮询映射下每核 panel 装载数上界；
                                            #   golden B×G=2 ✓。每组装载成本 ~5-8us（§10），8 组 ≈ 40-64us 仍
                                            #   ≪ 通用路径 365us，安全垫充足
G5(摊销):  ΣS × REPLICATE_H × G / (B × G) = ΣS×RH/B ≥ 16
                                            # 每 panel 平均任务数；panel staging 流量 ≤ 通用 ws 流量 1/16
                                            #   golden: 2048/2 = 1024 ✓✓
```

**rev2 附注**：G4（B×G ≤ 8）同时是检测 kernel 批序言循环的实际上界（§3.4）——门控通过的输入 batch ≤ 8，
检测 kernel 的 `for b in T.serial(batch)` 每 lane 至多 8 次迭代、Offsets 标量读 ≤ 18 次（L2 热）。

### 1.4 域检查（不变）

dtype 固定组合（Q/KV=bfloat16、Lse/ReduceSum=float32、Indices/Offsets/TokenIndices=int32）、
dim/tail_dim 幂等、topk 偶且 ≥32、head_kv 域（16≤h≤64 幂等 或 %64==0）、padded_H≠head_kv 时
kv_group==1——全部沿用现 kernel 编译期 assert；**任一不满足 → 回退通用路径**（通用路径域 ⊇ panel 域）。

### 1.5 设备侧检测：契约的等价判定（rev2 新增）

检测 kernel 判定的谓词与 §1.1 契约**逻辑等价**，采用**相邻行链式比较**形式：

```
violations = #{ r ∈ [1, ΣS) : bid(r) == bid(r-1) ∧ ∃g: Indices[r, g, :] ≠ Indices[r-1, g, :] }
契约成立 ⟺ violations == 0
```

**等价性论证**（严格）：packed 布局下 seq(b) 的行区间为 [bos_b, bos_b+len_b)（行连续）。契约"seq 内
所有行逐元素相等（传递性）" ⟺ "seq 内每个相邻行对 (r-1, r) 相等"（链式相等 ⟺ 全等：⇐ 由传递性沿链
归纳到任意对；⇒ 平凡）。`bid(r) == bid(r-1)`（相邻行同 seq）将比较严格限制在 seq 内部，跨 seq 边界
的行对（r = bos_b，其前驱属上一 seq）不参与判定——**无跨 seq 伪违约**。

**相邻行比较 vs 首行代表比较（裁决）**：采用相邻行比较。理由：
(a) 数学形式最弱假设——无需"每 seq 读首行向量"的代表行管理与按 seq 分组缓存，免 L2 热假设；
(b) 分段边界处理退化为一行前驱读取（每 (seq, lane) 子区间 [a, e) 自带行窗 [a-1, e-1]，a-1 ≥ bos 恒成立，
本 lane 自行加载前驱行，无跨 lane 协调——与任务书"跨段比较 row_start vs row_start-1"的顾虑等价消解，
代价仅为相邻 lane 间 ≤ 1 行的重复只读，无害）；
(c) 判定链为纯整数比较（int32 sub → |d| → max 归约），无浮点参与语义、无跨核同步——极简性是
"检测 bug = 沉默错误"风险的最好缓解（§11 R9）。

---

## 2. 编程模式选型

**Expert（沿用现 kernel 同模式，全部 5 个 builder 一致）**：显式 `T.alloc_L1 / T.alloc_L0C / T.alloc_ub` +
`T.Scope("C")/T.Scope("V")` 双流 + 手动 `T.set_flag/wait_flag`（核内）+ `T.set_cross_flag/wait_cross_flag`
（核间，mode=2 quorum：`tilelang/language/ascend.py` L191-210）。pass_configs 不设置。

rev2 增补：**检测 kernel 为 V-only Expert**（`T.alloc_ub` + 手动核内相邻对 + `T.tile.atomic_add`，
无 cross-flag）——V-only 一维 grid 在本仓 16 个示例中广泛先例（`examples/moe_token_permute/
moe_token_unpermute.py` L137/L288/L398、`examples/linear_attention_and_rnn/gdn/gdn_chunk_cumsum.py`
L24-32 的 `cid*2+vid` lane 映射同款）。

理由（沿用 rev1）：(a) 现 kernel 即 Expert 且已终态，变体必须同构以复用其全部同步纪律与实证教训
（HISA 悬挂、#15v2 相邻对、#22 尾体竞态）；(b) panel 常驻 L1 + 跨组调度需要显式 L1 生命周期控制；
(c) 语义/结构零漂移降低评审与回归成本；(d) 检测 kernel 的校验链与 panel kernel 逐任务校验链
（L1088-1104）逐指令同构，复用同一套已实证的 cast→sub→abs→[barrier]→reduce→标量分支→atomic 纪律。

---

## 3. 特化 kernel 架构

### 3.1 数据流总图（panel 主 kernel，不变）

```
                    ┌────────────────────── AIC（C 流, Scope("C")）──────────────────────┐
                    │ per (组变更): wait F_PANEL[pslot] → copy ws_panel[cid,pslot] →       │
                    │              panel_l1[pslot] (MTE2 144KB) → set F_PDONE[pslot]      │
                    │ per task k:   b(k) GM 标量读(组检测) → wait F_WSF[k%D] →            │
                    │              copy Q[s_row,H0:H1] → q_l1 → gemm_v0(q_l1,             │
                    │              panel_l1[pslot], acc_l0c, transpose_B=True, init=True) │
                    │              → copy acc_l0c → ws_acc[cid,k%D] (FIX) → set F1[k%D]   │
                    └───────▲───────────────────────────────────────┬────────────────────┘
                     F_PANEL/F_PDONE                                     F1（score 就绪）
                    ┌───────┴───────────────────────────────────────▼────────────────────┐
                    │ AIV×2（V 流, Scope("V")）                                          │
                    │ prologue: init-set F_WSF×2；扫 task 0..LA-1 组键（LA=32 次 GM 标量 │
                    │   读）→ 按序 gather 新组: P=Indices[s_row,g,:] → rows=clamp(P+bos) │
                    │   → 分块行读 KV → panel_stage_ub → ws_panel[cid,slot] (MTE3) →     │
                    │   set F_PANEL[slot]（复用槽前 wait F_PDONE[slot]）                  │
                    │ per task k: lookahead 组检测(b(k+LA) 1 次 GM 标量读,新组→gather) →   │
                    │   wait F1[k%D] → read ws_acc → acc_ub → mask 链(缓存 P + s_i) →    │
                    │   exp → reduce_sum → store ReduceSum → set F_WSF[k%D]              │
                    │   + 逐任务契约校验: Indices[s_row,g,半区] vs 缓存 P → 违约          │
                    │     atomic_add(ErrFlag, 1)（不沉默出错——rev2 保留为纵深防御，§8.2）│
                    └────────────────────────────────────────────────────────────────────┘
```

- **panel 载入（GM→L1，每组每核一次）**：V 侧 gather（AIV MTE2 行读为已证最快路径，26.1GB/s×40
  生产校准；AIC 直采已判死——R5 spike r_aic=100ns/行）→ `ws_panel`(GM) → C 整块拷入 L1。UB↔L1 不能
  直拷（api-kernel-memory.md §1/§3，硬件架构），GM 中转为强制路径，但流量仅 2×panel_bytes×组数×核数。
  golden 全核总量 ≈ 11.8MB（写 5.9 + 读 5.9）vs 消除的 302MB。
- **C 流（AIC）**：Q 读 37.7MB（L2 热，每 task 18KB，与 gemm 的 M/MTE1 重叠）；单 gemm
  `[16,576]×[576,topk]`（K 序与现 kernel 逐块 gemm 完全一致 → **logits 逐位一致**，§9 等价性依据）；
  ws_acc GM 往返**保留**（A2 无 L0C→UB fixpipe，API 级缺失——roofline §5.1 m_tile 判死结论），
  16.8MB 写 + 16.8MB 读计入现实地板。
- **V 流（AIV）**：mask 链与现 kernel 同构（m1/m2 clamp 算术链），P 按 (b,g) 组缓存（m2 = 1{P≠−1}
  组内只算一次，m1 = 1{P[j]≤s_i} 每 task 用新 s_i 重算）；exp/head 归约/输出同现 kernel。

### 3.2 与现 kernel 的关系（rev2 修订）

同文件 builder 族谱扩展为 **5 个**（§3.5）：unguarded generic（现，零修改）、unguarded panel（现，零修改）、
guarded generic（新）、guarded panel（新）、detect（新）。host 入口 `sparse_mla_topk_reducesum(...,
panel_mode="auto")` 三模式分发（§8.3）；`_BUILD_CACHE` 键扩展 guarded/detect 维度（§8.3）。
测试 `test_sparse_mla_topk_reducesum.py` 扩展 L0-P7 家族（§9.2），golden 判据不变。

### 3.3 循环与调度结构（panel/generic 主循环，不变；guarded 变体见 §8.2）

- **T.Kernel(grid_size, is_npu=True) 一维 persistent grid**（现 kernel 同；三维 Kernel 本项目不支持）。
- C/V 双 scope 各一个 **`T.serial(ntask)`** 每 task 主循环（运行时界 `ntask = max(0, ceildiv(total−cid, grid))`
  ——静态上限 + 运行时界的合规模式；**不用 T.Pipelined**：其循环界必须静态，ntask 动态不满足，
  ascend-constraints §1 强制检测项）。
- V prologue 为**编译期静态展开** `for i in range(LA)` + 每步 `if i < ntask` 运行时守卫（§6.1；
  现 kernel prologue `for s in range(LEAD)` + guard 同模式）。
- 行级循环 `for j in range(topk/2)`（gather 行读）静态展开，MTE2 描述符流水发射（现 kernel
  `for j in range(HALF)` 同款）；**无 T.Parallel**（Expert 模式手动 scope，元素级并行由 V pipe 内建）。

### 3.4 检测 kernel 架构（rev2 新增，~3-5us 预算 @golden）

**职责边界（极简性承诺）**：仅判定 §1.5 谓词——读 Indices（只读）+ Offsets（只读）→ 写 Violations
（atomic 计数）。不读 Q/KV/Lse，不做任何浮点语义计算，无 cross-flag，无 workspace。输入合法性
（offsets 单调、Indices 值域）沿用两主 kernel 的既有信任契约，不重复校验。

```
T.Kernel(grid_size, is_npu=True) as (cid, vid):        # grid=20，lane = cid*2+vid ∈ [0,40)
  with T.Scope("V"):                                    # V-only（moe_token_unpermute 同款先例）
    vmax_acc_ub = fill(0.0)                             # [1] fp32 全局违约幅值累积器
    one_i32_ub   = fill(1)                              # [1] int32 atomic 加数（只写一次）
    for b in T.serial(batch):                           # 动态界（≤ G4 后 ≤ 8；T.serial(ntask) 同款合规）
        bos = Offsets[b]; eos = Offsets[b+1]            # 每 lane 每 seq 2 次 GM 标量读（L2 热）
        ncmp = eos - bos - 1                            # 本 seq 相邻行比较数（len-1）
        if ncmp > 0:                                    # 空 seq(-1)/单行 seq(0) 自然跳过
            a = bos + 1 + (lane  * ncmp) // 40          # 本 lane 比较子区间 [a, e)（整除分区：
            e = bos + 1 + ((lane+1) * ncmp) // 40       #   覆盖 [bos+1, eos) 恰好一次、两两不交）
            cnt = e - a
            for c in T.serial(ceildiv(cnt, C)):         # 大 ΣS 分块（C 见 §5；golden 单块）
                lo = a + c*C; n = min(C, e - lo)
                T.set_flag("v","mte2"); T.wait_flag("v","mte2")     # 前块 V 读者排空（#15v2 相邻对）
                for i in T.serial(n + 1):               # 预载行窗 [lo-1, lo+n)（每行 1 条 MTE2，
                    T.copy(Indices[lo-1+i, :, :],       #   行流水发射，generic gather 同款）
                           pre_ub[i, :, :])
                T.set_flag("mte2","v"); T.wait_flag("mte2","v")     # 预载落地（相邻对）
                for i in T.serial(n):                   # 比较链：panel 逐任务校验链同构
                    T.tile.sub(d_i32, pre_ub[i+1, :, :], pre_ub[i, :, :])   # int32 邻差（动态行偏移
                                                                            #   静态 [G,topk] extent——
                                                                            #   indices_ub[islot] 同款先例）
                    T.copy(d_i32, d_f32)                # int32→fp32 Cast（chk_ub→chk_f_ub 同款）
                    T.tile.abs(d_f32, d_f32)
                    T.pipe_barrier("V")                 # cast 链与 reduce 向量状态隔离（panel spike 实证）
                    T.reduce_max(d_f32, vmax_g, dim=1, clear=True, tmp=chk_tmp)   # [G,topk]→[G] 行最大
                    T.reduce_max(vmax_g, vmax_ub, dim=0, clear=True)             # [G]→[1]（两步均
                                                                               #   reduce_ascend.py L378-380
                                                                               #   2D/1D 合法 dim；S3 备选
                                                                               #   1D flat GST 单次归约）
                    T.tile.max(vmax_acc_ub, vmax_acc_ub, vmax_ub)  # 累积（全已证 op 形态；
                                                                    #   优化备选 reduce_max clear=False）
    T.set_flag("v","s"); T.wait_flag("v","s")           # 标量读落地（vmax_ub[0] 同款 panel 先例）
    if vmax_acc_ub[0] > 0.5:
        T.set_flag("v","mte3"); T.wait_flag("v","mte3") # atomic 源落地保护（panel L1102-1104 同款）
        T.tile.atomic_add(Violations, one_i32_ub)
```

要点：
- **计数语义**：Violations = 发现违约的 lane 数（每 lane 至多 1 次 atomic，≤ 40 次）。布尔判定只看
  `>0`：过计无害（同一违约被多 lane 重复发现不可能——分区不交；多违约累计 ≥ 真值），**漏计为零**
  （分区覆盖恰好一次 + 链式等价 §1.5）。host 侧不做精确计数断言依赖（测试断言 >0/==0，§9.2）。
- **无悬挂**：kernel 无 cross-flag；核内相邻对全部无条件成对出现（首块的 v→mte2 对捕获空集，
  wait ≈ 免费——#22"无条件保护"纪律）。
- **空 seq / 单行 seq / ΣS=0**：ncmp ≤ 0 → 跳过；ΣS=0 时全部 seq 跳过、violations 保持 0（该输入
  上游 G5 门控必 miss，检测不会被发射——kernel 行为仍防御性正确）。
- **G>1**：行单位为 [G, topk] 整行（flat GST = G×topk 连续），一次比较覆盖全部 g——契约按 (seq,g)
  独立成立的语义由"逐 g 同时比较"精确覆盖（§1.5 谓词的 ∃g）。

### 3.5 builder 族谱与共享 body 构造（rev2 新增）

| builder（`@tilelang.jit` 公开名） | 装饰参数 | prim_func 参数序（粗体=新增） | 用途 |
|---|---|---|---|
| `tl_sparse_mla_topk_reducesum`（**现，零修改**） | out_idx=[6], ws_idx=[7,8] | Q KV Indices Lse Offsets TokenIndices ReduceSum ws_kv ws_acc | unguarded 通用（generic 显式 / auto·shared 门控 miss） |
| `tl_sparse_mla_topk_reducesum_guarded`（新） | out_idx=[], ws_idx=[8,9] | Q KV Indices Lse Offsets TokenIndices **Violations** ReduceSum ws_kv ws_acc | guarded 通用（auto 推测三连发） |
| `tl_sparse_mla_topk_reducesum_panel`（**现，零修改**） | out_idx=[7], ws_idx=[8,9] | Q KV Indices Lse Offsets TokenIndices ErrFlag ReduceSum ws_panel ws_acc | unguarded panel（shared 显式，rev1 行为） |
| `tl_sparse_mla_topk_reducesum_panel_guarded`（新） | out_idx=[], ws_idx=[9,10] | Q KV Indices Lse Offsets TokenIndices **Violations** ErrFlag ReduceSum ws_panel ws_acc | guarded panel（auto 推测三连发） |
| `tl_sparse_mla_topk_reducesum_detect`（新） | out_idx=[], ws_idx=[] | Indices Offsets **Violations** | 契约检测（auto） |

**共享 body 构造**：generic 与 panel 各自抽出单一 body 构造函数（Python 普通函数，持 `guarded: bool`
编译期布尔参——trace 时 Python 分支，`SIDE_PAIR`/`OFFSET_N>1` 同款既有模式），两个薄 `@tilelang.jit`
包装分别以 `guarded=False/True` 调用。**guarded=False 的 traced 语句序列与现版逐条一致**（守卫行在
`if guarded:` 内不被 trace；名字重绑定为纯 Python 别名，不产生 TIR 语句）⇒ TIR 等价 ⇒ 编译产物等价
（编译确定性前提下）——"现行为二进制保全"的实现形态。guarded=True 的 body 差异 = 每 scope 一行
ntask 钳位（§8.2）+ Violations 形参。

**ReduceSum 在 guarded 变体中为普通输入**（host 预分配共享输出，两 kernel 互补守卫下恰一写入）：
`out_idx=[]` 机制经源码核实——`tilelang/jit/adapter/cython/cython_wrapper.pyx` L102-141 对非
result_idx/workspace_idx 参数按位置由调用方传入（仅校验 contiguous/dtype/device），kernel 写普通输入
张量有本仓先例（panel kernel `atomic_add(ErrFlag)`——ErrFlag 即普通输入，L803/L1104）；空 result_idx
时 wrapper 返回空表（L200-203），host 以自分配张量为返回值。**spike 项 S1**：out_idx=[] 的最小冒烟
（风险低，机制已核实；失败则回退非对称形态——panel_guarded 保留 out_idx、generic_guarded 收
panel 的输出张量为输入，机制相同仅管线挪位）。

---

## 4. 数据规格与内存规划（Expert 显式层级）

### 4.1 输入/输出（主 kernel 与现版逐字段一致；rev2 增补粗体）

| 张量 | shape | dtype | 说明 |
|---|---|---|---|
| Q | [ΣS, heads, DDT] | bfloat16 | DDT = dim + tail_dim（golden 576） |
| KV | [ΣS_kv, kv_group, DDT] | bfloat16 | |
| Indices | [ΣS, G, topk] | int32 | panel 契约对象；值域 [-1, ΣS_kv)；**检测 kernel 的唯一数据输入** |
| Lse | [ΣS, heads] | float32 | |
| Offsets | [batch+1] | int32 | cu_seqlens；**检测 kernel 的分区输入** |
| TokenIndices | [ΣS, 2] | int32 | host 预构建（`prepare_token_indices`，既有） |
| **Violations（新，auto 内部）** | [1] | int32 | 检测违约计数（host 每调用 `torch.zeros` 新建；detect 写、guards 读——**调用方不可见，不入 proto**） |
| **ReduceSum（guarded 变体形态）** | [ΣS, G, RH, topk] | float32 | host 预分配共享输出（unguarded 变体仍为 out_idx 框架分配，不变） |
| ErrFlag | [1] | int32 | panel 逐任务校验违约计数（host 预置 0，调用后检查——rev2 在 auto 模式保留，§8.2） |
| ws_panel（panel ws） | [grid, 2, topk, DDT] | bfloat16 | panel GM 中转双缓冲；golden ≈ 5.9MB |
| ws_acc（panel ws） | [grid, 2, H_per_block, topk] | float32 | DEPTH=2 槽；golden ≈ 320KB |
| ws_kv / ws_acc（generic ws） | [grid, DEPTH·BI, DDT] / [grid, DEPTH·HpB, BI] | bf16 / fp32 | 通用路径既有，不变 |

（通用路径 ws_kv 不再被 panel 路径使用；各 builder workspace_idx 独立。）

### 4.2 片上存储实算（panel 主 kernel，golden 配置，per core / per AIV——不变）

| 缓冲 | 层级 | shape | 字节 | 容量占比 |
|---|---|---|---|---|
| panel_l1 ×2 槽 | L1 | [2, topk, DDT] bf16 | 294,912 | 56.2% of 512KB |
| q_l1 | L1 | [16, 576] bf16 | 18,432 | 3.5% |
| **L1 合计** | | | **313,344 ≈ 306KB** | **59.7%** ✓ |
| acc_s_l0c | L0C | [16, topk] fp32 | 8,192 | 6.3% of 128KB ✓ |
| L0B staging（gemm_v0 内部，kL0Size=128） | L0B | [topk, 128] bf16 | 32,768 | 50% of 64KB ✓（与现 kernel BI=128 同形，实证） |
| panel_stage_ub | UB | [topk/2, DDT] bf16/半区 | 73,728 | 37.5% of 192KB |
| acc_ub/lse2d/mask2d/杂项 | UB | — | ≈ 15,000 | ≈ 7.6% |
| **UB 合计** | | | **≈ 89KB** | **≈ 45%** ✓ |

对齐核查（decision-tree §3）：DDT=576 与行 1152B 均 32B 倍数（UB/L1 32B 对齐 ✓）；L0A/B 512B、
L0C 64B 由 gemm_v0 内部分形管理（M=16≥16、K=576 整除 16 满足 zN fractal——现 kernel L201-203 同论证）。
T.copy 切片核查：全部切片的列维均为 buffer 最内维（ws_panel/ws_acc/ReduceSum 的 topk/行维切片），
无 4D 列 strided 场景（api-kernel-memory.md §3 限制不触发）。

### 4.3 检测 kernel 片上存储（rev2 新增，per AIV lane）

| 缓冲 | 层级 | shape | 字节（GST=128） | 说明 |
|---|---|---|---|---|
| pre_ub | UB | [C+1, G, topk] int32 | (C+1)×GST×4 ≤ 64KB+行 | 行窗预载（C = min(256, 16384//GST)，§5） |
| d_i32 / d_f32 | UB | [G, topk] | 512B / 512B | 邻差 int32 / fp32（reduce 限 fp32/fp16——reduce_ascend.py L373 文档） |
| vmax_g / vmax_ub / vmax_acc_ub | UB | [G] / [1] / [1] fp32 | ≤ 256B | 行最大中间量 / 当行幅值 / 累积幅值（两步归约 §3.4） |
| one_i32_ub / chk_tmp_ub | UB | [1] int32 / [256] uint8 | 4B / 256B | atomic 加数 / reduce 显式 tmp（panel L834-836 同款） |
| **UB 合计** | | | **≤ ~68KB** | **≈ 35% of 192KB** ✓ |

GM 流量（golden）：Indices 1MB 只读（每 lane ≈ 26KB + 相邻 lane ≤ 1 行重复读）+ Offsets ≤ 18×40
标量读（L2 热）+ Violations ≤ 40 次 4B atomic——**vs panel 路径 55.8MB：流量比 1/56**，是 §10.4
缩放论证（检测恒 ≈ panel wall 的 2-4%）的根据。

### 4.4 [REORDER-COST] 审计（panel 路径，不变 + detect 附注）

```text
path: panel-gather（KV 散射行 → ws_panel 连续 → L1）
gm_passes: panel 数据 GM 读(KV 行)+GM 写(ws_panel)+GM 读(→L1) = 3 × panel_bytes × 组数 × 核数
            = golden 3×144KB×2×20 ≈ 17.3MB…（实际读 5.9/写 5.9/读 5.9，C 侧读后常驻，共 11.8MB 净流量 + KV 源行读 288KB L2 热）
            vs 通用路径 ws_kv 往返 604MB（写 302+读 302）—— 51× 缩减
dma_transactions: 每 (组,核): 128 条 MTE2 行读(1152B) + 2 条 MTE3 bulk(72KB×半区) + 1 条 C 侧 MTE2(144KB)
                   golden 全核 ≈ 40 组装 × 131 ≈ 5.2K 条（vs 通用 26.2K×2 行描述符 + 824 块拷贝）
gm_scalar_accesses: C 侧 1/task + V 侧 2/task + V prologue 33/核 ≈ 3×102+33 ≈ 340/核（≈ numel 级？否：
                    ΣS=2048，占比 17%，且全部为流水化标量读，先例 gqa_sink 2/task 生产验证）
address_div_mod_per_element: pid 算术 per task（同现 kernel），无逐元素 div/mod
active_cores / serial_tasks_per_core: 20 核 / 102 task（golden）
largest_case_timeout_gate: golden S=2048/topk=128，单 case 预算 < 220us（实测 143.3us，§10.4）
result: pass

附注（detect，rev2）: gm_passes = Indices 1MB 单读（无重排，顺序行读）；dma = ΣS/40 行 MTE2/lane +
            ≤40 atomic；gm_scalar = ≤18×40 Offsets 读（L2 热）+ lane 分区纯算术；result: pass
```

---

## 5. Tiling 策略

### 5.1 主 kernel（不变）

| 维度 | 块 | 依据 |
|---|---|---|
| gemm M | H_per_block=16（padded_H/RH 逻辑沿用） | 现 kernel 同；L0A fractal M≥16 |
| gemm N | **topk 整宽单 gemm**（topk ≤ 128 v1 域） | L0B staging topk×kL0×2 ≤ 64KB；golden 128×128×2=32KB；与现 kernel BI=128 gemm 完全同形（已验证配置） |
| gemm K | DDT=576，kL0Size=128（gemm_v0 内部 4.5 步推进，余数步内部处理——现 kernel 实证） | K 序一致 ⇒ logits 逐位一致（等价性证据） |
| topk > 128 | v1 不支持（host 门控回退）；扩展方案 = panel_l1 行切片 BufferRegion 多 gemm（gemm_v0 接受 BufferRegion，ascend.py L432-437） | §11 风险表 |
| V 侧 topk 半分 | 每 AIV 处理 topk/2 列（vid 半区，同现 kernel HALF 机制） | ws_acc/acc_ub/mask 全部按半区切 |
| Q 行 | 每 task 整行 [H, DDT] 连续切片，无 gather | 同现 kernel |
| 非整除 | 沿用现 kernel 域约束（topk%BI==0 等编译期 assert），panel 路径无新增尾块维度（topk 即 gemm N，无块尾） | — |

（GEMM 非整除检测：M=16/N=topk/K=576 均满足分形与对齐；576 % 128 = 64 ≠ 0 的 K 余数步由 gemm_v0
内部处理，现 kernel L266-269 同配置生产验证，非新增风险。）

### 5.2 检测 kernel（rev2 新增）

| 维度 | 块 | 依据 |
|---|---|---|
| 行窗 chunk C | `C = min(256, 16384 // GST)`（GST = G×topk 编译期常量） | pre_ub (C+1)×GST×4 ≤ ~64KB（§4.3 UB 预算）；golden GST=128 → C=128，每 (seq,lane) 子区间 ~26 比较 → 单块 |
| 比较粒度 | 整行 [G, topk]（flat GST 连续）一次比较 | 动态行偏移 + 静态 extent 切片（`indices_ub[islot]` 同款先例）；比较次数 = ΣS−B_nonempty |
| lane 分区 | (seq, lane) 二维整除分区（§3.4 公式） | 覆盖恰好一次、两两不交、行窗全在 seq 内（无跨 seq 比较） |
| 对齐 | GST×4 % 32 == 0（编译期 assert） | T.tile 操作数 32B 对齐（ascend_tile.py sub 注释 L1101）；门控上游 topk%32==0 保证恒成立，assert 为防御镜像 |

---

## 6. panel 调度协议（C/V 组序列一致性——不变；guards 正交性注记）

### 6.1 组序列与单调性（不变）

- C 侧检测：每 task 1 次 GM 标量读 `b(k) = TokenIndices[s_row(k), 0]`（g(k) 纯 pid 算术），与
  `prev_key` 比较；变更 ⇒ `pslot ^= 1; wait F_PANEL[pslot]; copy ws_panel[cid,pslot] → panel_l1[pslot];
  set F_PDONE[pslot]`。**C-scope GM 标量读先例**：`examples/attention_sink/example_gqa_sink_fwd_varlen/
  example_gqa_sink_fwd_varlen.py` L268-270 + panel kernel spike 实证（debug_log Attempt 1：一次通过）。
- V 侧供给：prologue 静态展开扫描 task i=0..LA-1（LA=32）的组键，按首现顺序 gather 各组（组 g →
  槽 g%2，槽复用前 `wait F_PDONE[槽]`）；主循环每 task 检查 `b(k+LA)`（`k+LA < ntask` 守卫 + clamp
  寻址），新组 ⇒ gather。**prologue 每步必须带 `if i < ntask` 运行时守卫**（幻影组 → F_PANEL set/wait
  错位 → 死锁——rev1 红字教训保留）。
- **一致性论证**：C 的 wait 序列 = tasks 0..ntask−1 的组首现序列；V 的 set 序列 = 同一序列；槽位按序
  0,1,0,1 交替，两侧严格同序同数 ⇒ F_PANEL set/wait 1:1，无漏组。
- 提前量：LA=32 × C 每 task ~0.4-0.5us ≈ 13-16us ≥ 单组 gather ~5-8us，组切换对 C 零暴露。

### 6.2 尾块保护分析（#22 教训强制项——不变，风险表保留）

| 风险点 | 保护机制 |
|---|---|
| lookahead gather 被 `k+LA ≥ ntask` 跳过 → 内部 mte3→mte2 保护对缺失 | 被跳过的 gather 不写 panel_stage_ub ⇒ 无需保护；**post 段 rs_ub 保护对无条件存在**（#22"尾体无条件保护"原则） |
| prologue/主循环 gather 连续复用 panel_stage_ub | 相邻 mte3→mte2 对（#15v2/#22 论证的唯一安全形态） |
| ws_panel 槽 s 被组 g+2 重写 vs C 读组 g | 显式 `F_PDONE[s]` 握手 |
| panel_l1 槽 s 被组 g+2 装载 vs 组 g gemm 读 | C 同核程序序 + gemm_v0 尾部内建 M→MTE1→MTE2 排空 |
| ws_acc 槽 k%D 被 C(task k) 重写 vs V(task k−D) 读 | 显式 `F_WSF[k%D]` 握手（V prologue init-set ×2 + C 尾部 destroy-wait ×2 平衡计数） |
| 两 AIV 分支发散 | 全部 guard 条件（k、ntask、组键）vid 无关 ⇒ 两 AIV 恒同分支（mode-2 quorum 前提） |
| 悬挂（HISA 教训） | 全部 set/wait 位于有效程序段；两侧序列一致性保证计数封闭；配平表 §7.2 |
| **guard 早退与组调度交互（rev2 注记）** | 守卫 = ntask 钳位到 0（§8.2）——早退时 C 循环零次、V `if ntask>0` 整体不进入，与"空闲核"路径逐字节同构；**无任何组调度状态被部分更新**（pslot/gcnt 等循环携带变量均在 ntask>0 的有效段内才触碰） |

---

## 7. 同步协议

### 7.1 flag 预算（panel 主 kernel，不变：A2 FFTS flagId 4bit、8 个合法 id）

| id | 名称 | 方向 | set 管道 | 语义 |
|---|---|---|---|---|
| 0,1 | F_PANEL[2] | V→C | MTE3（quorum：两 AIV 均写毕 ws_panel 才放行 C——mode-2） | 组 g 的 panel 已落 ws_panel[槽] |
| 2,3 | F_PDONE[2] | C→V | MTE2（C 侧 L1 装载完成） | ws_panel[槽] 已被消费，可重写 |
| 4,5 | F1_ACC[2] | C→V | FIX（L0C→GM 落地，现 kernel 同款） | ws_acc[槽] score 就绪 |
| 6,7 | F_WSF[2] | V→C | MTE2（V 读毕 ws_acc，gqa_sink 同款） | ws_acc[槽] 可重写（free） |

DEPTH=2（ws_acc 双槽），共 8 id 恰满合法预算。**guarded 变体不新增任何 flag id**（守卫仅钳位 ntask）。

### 7.2 计数配平表（每核，ntask>0 时——不变）

| flag | set 总数 | wait 总数 | init/destroy 配平 |
|---|---|---|---|
| F_PANEL[s] | V: 该槽组数 | C: 该槽组数 | 严格 1:1，无 init |
| F_PDONE[s] | C: 该槽组数 | V: max(0, 该槽组数−1) | V 尾部 destroy-wait ×1/槽 |
| F1_ACC[s] | C: 该槽 task 数 | V: 该槽 task 数 | 严格 1:1（循环界 ntask 同构于现 kernel） |
| F_WSF[s] | V: 2(init) + 该槽 task 数 | C: 该槽 task 数 + 尾部 destroy ×1 | gqa_sink "pretend free + destroy" 同款 |

ntask==0：两侧整体不进入（F_PANEL/F_PDONE/F1 零 set 零 wait；F_WSF 不 init）⇒ 无悬挂。
核内（同 pipe 对间）沿用现 kernel 全套相邻对纪律（v→mte2 / mte2→v / mte2→mte3 / mte3→v / s→v）。

### 7.3 与编程模式匹配（不变）

Expert 手动同步：核内 11 类 set/wait_flag 相邻对、核间 8 个 cross-flag（上表）。V 侧全部 cross-flag
set/wait 位于有效块程序段——HISA 头注释教训的纪律性延续。

### 7.4 检测 kernel 同步纪律（rev2 新增）

- **无 cross-flag、无 barrier_all**：40 lane 互相独立（分区不交），唯一共享写 = Violations atomic
  （`T.tile.atomic_add`：dst 限 GM、src 限 local、dtype 含 int32——ascend_tile.py L313-349 签名核实；
  panel ErrFlag 生产先例）。
- 核内相邻对（全部无条件成对，#22 纪律）：每 chunk `v→mte2`（前块 V 读者排空）→ 预载行 MTE2 →
  `mte2→v`（预载落地）→ 比较链；尾部 `v→s`（vmax 标量读）与 `v→mte3`（atomic 源落地）各一对
  （panel L1097-1104 同款）。
- **嵌套动态 T.serial**（batch → chunk → 行，3 层动态界）：TIR for 循环逐层合法（T.serial(ntask)
  先例泛化），**spike 项 S2**（检测 kernel 冒烟即覆盖；失败则 chunk 层并入行层——单层动态界，
  代价为每行多一次 mte2→v 判定，预算内可吸收）。

### 7.5 守卫早退的同步洁净性论证（rev2 新增——任务书强制项）

**守卫位置**：每 scope 首条可执行语句（C：`for k in T.serial(ntask)` 循环界之前；V：`if ntask > 0:`
之前）——即**全部 cross-flag 协议启动之前**（F_WSF init / prologue / 组调度均在 ntask>0 有效段内）。

**论证**：守卫的实现是 `ntask = if_then_else(Violations[0] ⊙ 0, ntask_raw, 0)`（⊙ 按变体取 > 或 ==，
§8.2）——早退路径 = **ntask==0 的空闲核路径，逐字节同构**。该路径的无悬挂性有双重证据：
(1) 结构证明——§7.2 配平表"ntask==0：两侧整体不进入（零 set 零 wait；F_WSF 不 init）⇒ 无悬挂"；
(2) 实证——L0-P4b（ΣS=1 单 task）在 panel+generic 双路径上运行过 19/20 空闲核（debug_log Attempt 1
precision_pass）。两 scope 读同一 Violations[0]（detect 完成后为确定值，流序 §8.4）⇒ 全核全 scope
分支一致，无发散 flag 状态。**结论：guarded 早退不产生任何悬挂或半更新状态——这是"守卫动已终态代码"
零回归风险的论证核心。**

---

## 8. 分发策略（rev2 重写：auto 缺省 + 推测三连发）

### 8.0 rev1 结论与 rev2 演进关系

rev1 §8 在方案 A（调用方声明）与方案 B（自动检测 + 双发射）间裁决 A，核心理由是 B 的两个代价：
(i) "检测错误的代价 = 沉默错误结果"（roofline §5.2 风险原文）；(ii) 必须给通用 kernel 加入口守卫
（动已终态代码）。rev2 的用户需求是调用方零声明（透明分发），据此把 B 的两个代价分别化解后落地为
`panel_mode="auto"` 缺省：

- (i) 的化解：**三层防御**——① 检测谓词极简（§1.5 纯整数链式比较，无浮点无跨核同步）+ L0-P7
  专项测试族锁死（含分段边界/多违约/varlen 边界/竞态）；② panel 逐任务校验链**保留**（detect 漏报
  ⟹ panel 实际逐行校验必捕获 ⟹ ErrFlag → host RuntimeError——把最坏失败模式从"沉默错误"升格为
  "响亮报错"；注意 detect **误报**方向天然安全：violations>0 → generic 全量执行 → 结果恒正确，仅
  性能回退）；③ 误报/漏报的不对称性使唯一危险路径 = "detect 漏报 ∧ panel 校验链同坏"——两个独立
  bug 的合取，测试族分别锁死每一层（§11 R9）。
- (ii) 的化解：**方案 f 的 ntask 钳位守卫**（§8.2）——unguarded 变体 traced 语句与现版逐条一致，
  现行为二进制保全；守卫复用已实证的 ntask==0 无悬挂路径（§7.5）。

方案 A（`panel_mode="shared"`）**原样保留**为显式覆盖——无检测、无守卫、ErrFlag 检查不变 = rev1
已验收行为。

### 8.1 三模式语义总表

| `panel_mode` | 语义 | 发射序列 | 检测 | panel 校验链 | 行为相对 rev1 |
|---|---|---|---|---|---|
| **`"auto"`（新缺省）** | 设备侧自动判定：元数据门控（G1-G5+域，纯 offsets/shape）通过 → detect + panel(guarded) + generic(guarded) 推测三连发，violations 定路（==0 → panel 执行/generic 整核早退；>0 → panel 早退/generic 执行）；门控 miss → unguarded generic 单发 | miss: 1×generic；pass: zeros+detect+panel_g+generic_g | ✓（§3.4） | 保留（纵深防御 + host ErrFlag 检查） | 新增（调用方零声明） |
| `"generic"`（显式） | 恒通用路径 | 1× unguarded generic | ✗ | n/a | **零变化**（rev1 缺省路径原样） |
| `"shared"`（显式） | 调用方保证契约 + host 门控 + ErrFlag 兜底 | miss: 1×generic；pass: 1× unguarded panel | ✗ | 保留（ErrFlag → RuntimeError） | **零变化**（rev1 验收行为原样） |

**输出正确性不变式**：三模式对同一输入的输出两两 torch.equal（auto 的两分支各自等价于 generic 或
shared 的对应分支；L0-P7f 锁定）。

### 8.2 generic 入口守卫机制裁决（唯一动已终态代码之处）

**候选对比与裁决：方案 f（采纳）**。

| 维度 | 方案 f：guarded 编译期变体（采纳） | 方案 e：恒发射 detect + 恒读标志 |
|---|---|---|
| 默认/门控 miss 路径开销 | **零**（unguarded 单发，路径与 rev1 逐条一致） | 每调用 +15-25us 全路径税（detect 发射 ~3-5 + 额外 launch ~8.4 + 守卫读）或至少 +1-2us（伪造标志变体仍需每调用 dummy 张量） |
| 现行为保全 | unguarded 变体 traced 语句逐条一致（真二进制保全） | 通用 kernel 签名/行为必变（Violations 恒在参数表中） |
| 维护面 | 双变体（缓解：共享 body 单源，§3.5；guarded delta = 2 行/scope） | 单变体（但被 auto 语义绑架，默认路径永久付费） |
| 被 f 支配？ | — | 是——e 的唯一性质"单变体"由共享 body 构造等效获得，而 e 的税无法消除 |

**方案 f 的实现裁决（三处关键选择）**：

1. **双薄包装 + 共享 body（而非单 builder 加 guarded kwarg）**：单 builder 的 prim_func 签名固定，
   Violations 恒在参数表 ⇒ unguarded 调用需传 dummy（+每调用分配或模块级缓存 hack）且**既有 builder
   直调点（bench 脚本、测试 L597 探针）全部破坏接口**——与"零开销零变化"矛盾。双薄包装使 unguarded
   的公开名/签名/装饰器**零修改**（§3.5 族谱），guarded 布尔参留在 body 构造函数（trace 时分支，
   `SIDE_PAIR` 既有模式）。
2. **守卫 = ntask 钳位**（`ntask = if_then_else(Violations[0] ⊙ 0, ntask_raw, 0)`，generic_g 取
   `Violations[0] > 0`（违约才需要 generic），panel_g 取 `Violations[0] == 0`（合格才执行 panel）——
   **互补条件 ⇒ 恰一执行**）：每 scope 一行、复用 ntask==0 已证无悬挂路径（§7.5）、零新增同步结构。
   替代形态（如整核 if 包裹）被否决：分支体巨大、TVMScript IfFrame 作用域约束（rev1 实现教训：
   "分支体自包含"）增加重构面。
3. **ReduceSum 在 guarded 变体中为 host 预分配共享输入（out_idx=[]）**：三连发两 kernel 互补守卫下
   恰一写入同一 rs；host 无需 device 同步即可返回 rs（若各自 out_idx 框架分配，host 无法在不读
   violations 的情况下知道哪个输出有效——那需要 D2H 同步，破坏捕获与零同步）。机制核实见 §3.5
   （spike 项 S1 + 非对称回退形态）。

**panel 入口守卫**：同 ntask 钳位（§7.5 论证共用）。**panel 逐任务契约校验链处置：保留（裁决）**——
理由：(a) **零边际成本**——143.3us 实测基线已含该链（panel_metric2_bench 在实现版 kernel 上测量，
链为 kernel 体内固有序列），移除属纯优化且牺牲安全性；(b) 纵深防御价值真实且完备——detect 漏报时
panel 仍逐行比对实际 Indices vs 缓存 P（与 detect 同一谓词的逐行形式），必置 ErrFlag → host
RuntimeError（auto 模式保留 err_flag.item() 尾部检查）；(c) `"shared"` 显式模式行为不变（无 detect，
ErrFlag 检查保留 = rev1 原样）。

### 8.3 host 分发逻辑（重写）

```python
def sparse_mla_topk_reducesum(..., panel_mode="auto"):    # 缺省改为 auto
    seq_len, heads, ddt = q.shape; ...; grid_size = ...
    off_list = None
    if panel_mode == "generic":
        return _generic_unguarded(q, kv, ..., prepare_token_indices(offsets))   # rev1 路径逐字节一致
    if panel_mode in ("auto", "shared"):
        off_list = offsets.detach().cpu().tolist()          # 既有 D2H（1 次；两模式共用）
        gate_ok, reason = _panel_gate_ok(heads, dim_v, tail_dim, topk, kv_group, seq_len, off_list)
        if panel_mode == "shared":
            if gate_ok:
                return _panel_unguarded_with_errcheck(...)  # rev1 §8.3 原样（ErrFlag 强制检查）
            # miss → 落入通用（off_list 复用构建 token_indices——host 微优化，语义零变化）
        elif panel_mode == "auto" and gate_ok:
            # ---- 推测三连发（零 host 同步；全部为 device 异步 op，graph 可捕获，§8.4）----
            token_indices = _token_indices_from_list(off_list, offsets.device)      # 既有 helper
            violations = torch.zeros(1, dtype=torch.int32, device=q.device)         # err_flag 同款先例
            err_flag   = torch.zeros(1, dtype=torch.int32, device=q.device)
            rs = torch.empty((seq_len, kv_group, REPLICATE_H, topk), dtype=torch.float32, device=q.device)
            detect_f  = _build_cached("detect",  (topk, kv_group, grid_size))
            panel_g   = _build_cached("panel_guarded",  (heads, dim_v, tail_dim, topk, kv_group, sm_scale, grid_size))
            generic_g = _build_cached("generic_guarded", (heads, dim_v, tail_dim, topk, kv_group, sm_scale, block_I, grid_size))
            detect_f(topk_indices, offsets, violations)          # 同流首发：流序保证先于守卫读取完成（§8.4）
            panel_g(q, kv, topk_indices, lse, offsets, token_indices, violations, err_flag, rs)
            generic_g(q, kv, topk_indices, lse, offsets, token_indices, violations, rs)
            if int(err_flag.item()) != 0:                        # 尾部防御检查（eager；捕获口径见 §8.4）
                raise RuntimeError("panel contract violated despite detect pass (detect bug?) ...")
            return rs
        # auto gate-miss → 落入通用（off_list 复用——D2H 总次数与旧默认路径持平）
    # 显式 generic / shared·auto 门控 miss 的公共出口；auto·shared 分支已取 off_list 时复用之
    #（省一次 D2H，语义零变化），否则走 prepare_token_indices（rev1 原样）
    token_indices = (_token_indices_from_list(off_list, offsets.device) if off_list is not None
                     else prepare_token_indices(offsets))
    return _generic_unguarded(q, kv, topk_indices, lse, offsets, token_indices)
```

**`_BUILD_CACHE` 键扩展**（沿用 rev1 panel 键的元组标记模式，L1306）：
`generic: (heads, dim_v, tail_dim, topk, kv_group, sm_scale, block_I, grid_size)`（不变）；
`panel: (..., "panel", grid_size)`（不变）；新增 `generic_guarded: (..., block_I, grid_size, "guarded")`、
`panel_guarded: (..., "panel", grid_size, "guarded")`、`detect: (topk, kv_group, "detect", grid_size)`。

**D2H 计数对账（零新增同步论证）**：rev1 默认（generic）路径 = 1 次 D2H（prepare_token_indices 内
`.cpu().tolist()`）；rev2 auto 全分支 = 1 次 D2H（off_list，且 token_indices 由 off_list 复用构建——
`_token_indices_from_list` 既有 helper，rev1 panel 路径同款）。auto 推测段（gate 通过后）为纯 device
异步 op 序列（2×zeros + 1×empty + 3 发射）+ 尾部 1 次 `.item()`（eager 防御检查，rev1 shared 路径
同款先例、位于计时/捕获路径外）——**hot path 零新增 host 同步**。

**graph 捕获口径（metric-2 协议兼容性，§10.4 引用）**：捕获单元 = `[violations 置零（fill op，必须
在捕获内——replay 重置，否则跨 replay 累积翻转选路，禁止移出）, detect, panel_g, generic_g]`，
token_indices/rs/err_flag 预构建（捕获外），off_list 门控在捕获外（eager 一次性）。eager e2e 口径
（~1.5ms host-bound）不适用 auto 三连发计时——与 rev1 结论一致（Phase 1-4 eager 口径失效）。

[HOST-METADATA-AUDIT]（rev2 增补条目；rev1 条目沿用）
```text
operation: off_list 复用构建 token_indices（_token_indices_from_list）
input_stride -> output_stride: n/a（纯 Python int 列表 → 新建张量，rev1 先例原样）
aclnn_or_physical_copy: false（数据准备类，一次 H2D）
result: allow

operation: violations/err_flag = torch.zeros(1, int32, device)；rs = torch.empty(shape, fp32, device)
aclnn_or_physical_copy: false（新建张量 + device fill/空分配；err_flag 为 rev1 既有同款；
                        rs 为输出容器分配（数据准备类），由 kernel 写入）
result: allow

operation: err_flag.item()（auto 尾部防御检查）
aclnn_or_physical_copy: false（只读；位于计时/捕获路径外，rev1 shared 同款）
result: allow
```

### 8.4 流序与捕获论证（任务书强制项）

**TileLang 默认流串行语义（源码实证）**：kernel 发射路径 = `tilelang/jit/adapter/cython/
cython_wrapper.pyx` L86-90——`stream == -1`（缺省）时取 `torch.npu.current_stream().npu_stream`，
经 L194 传入 C++ launcher。三个 builder 的缓存 func 在同一 host 线程按 detect → panel_g → generic_g
顺序调用 ⇒ 三次发射进入**同一当前流** ⇒ ACL 流内按入队序串行执行（aclrt 流语义，与 CUDA 同构；
torch_npu graph 捕获时 `current_stream()` 即捕获流，捕获保持发射序）。⇒ **detect 的 GM 写
（Violations atomic）在其后任意 kernel 的守卫读之前全局可见**。本仓实证交叉：golden_e2e_graph.py
将 tilelang kernel + torch sum/div 捕获进同一 graph 并按序依赖前序输出（panel_metric2_bench §0 协议），
跨 kernel 同流顺序已被生产测量协议使用。

**守卫读取的一致性**：panel_g 与 generic_g 各自的 C/V scope 独立读同一 Violations[0]——流序保证
该值为 detect 终值（确定）⇒ 全核全 scope 判定一致 ⇒ 互补条件恰一执行（§8.2）⇒ rs 单写者、无写写
竞态；早退方零 flag op（§7.5）。

**violations 张量生命周期裁决：每调用新建（采纳）；复用禁止**。理由：(a) `torch.zeros(1, device)`
为 err_flag 既有同款（rev1 §8.3 审计 allow，L0-P3 测试锁定）；(b) 复用在 graph 捕获下错误——replay
不重置则跨 replay 累积翻转选路；重置需在捕获内加 fill op（等价于新建的语义但多一条 host 持有约束：
张量必须与 graph 同生命周期，双 graph 共享即互 corrupt）；(c) eager 下复用同样需要每调用 `.zero_()`
（同一 fill op 数，无节省）——复用无收益纯风险。**不变式（写入文档与测试）：violations 必须每调用
新建/重置；捕获单元必须包含置零 op。**

---

## 9. 验证方案（Golden + L0 门槛测试计划）

> L0 计划由 `tilelang-op-test-design` 场景 A 生成（rev1 生成 P1-P6，rev2 增补 P7 家族——本节）；
> L1（功能含不规则 shape）/L2（异常）/Boundary 扩展由 Stage 2 在 L0 通过后按场景 B 生成，本阶段不产出。

### 9.1 Golden 函数草案（PyTorch 参考实现——不变）

**主参考（数值权威）**：逐字复用 `test_sparse_mla_topk_reducesum.py::ref_mla_topk_softmax`
（fp32 softmax + varlen 逐 batch 循环 + causal mask + gather + renormalize + lse 生成——GPU 主仓
golden 的忠实移植，本仓已验证）。

**等价性参考（panel 特化专属）**：同一输入下 `panel 路径输出 vs 通用路径输出`。理论依据：两路径
gemm 同形（[16,576]×[576,128]）同 K 推进序（kL0Size=128、K=576）同 fp32 累加 ⇒ logits 逐位一致；
V 侧 mask/exp/reduce 链逐指令同构 ⇒ **ReduceSum 期望逐位相等**（torch.equal；rev1 L0-P1 已实证，
rev2 L0-P7b/P7f 复验于 auto 路径）。

**rev2 增补（检测单元 golden）**：violations 期望值由 host 侧 Python 参考谓词直接计算
（`viol_ref = sum(1 for r in range(1, ΣS) if bid[r]==bid[r-1] and Indices[r]!=Indices[r-1])`——
逐行 numpy/torch 比较，int32 精确断言：`violations == 0 ⟺ 契约成立`；计数语义 = lane 数故仅对
0/单违约用例断言精确值，多违约用例断言 `>0` 与 `< 谓词真值+40`）。

### 9.2 L0 门槛测试计划（规则 shape / dtype / 判据）

L0 用例总表（rev1 P1-P6 沿用 + **P2 修订注记** + **P7 家族新增**；数据构造与 test 文件同源：
seed=42 randn bf16、offsets varlen、lse 由 fp32 参考生成）：

| # | 用例 | 配置 | 路径 | 判据 |
|---|---|---|---|---|
| L0-P1 | golden 经特化路径 | B varlen [0,1023,2048] / S=2048 / H=16 / D=512 / tail=64 / topk=128 / arange 重复索引 / panel_mode="shared" | panel(unguarded) | vs 纯 torch ref：err_ratio ≤ 1e-2 且 abs_err ≤ 1e-3 且 NaN=0；AND vs 通用路径输出 torch.equal |
| L0-P2 | 通用路径回归（**修订：显式 panel_mode="generic"**） | 同 L0-P1 输入，**panel_mode="generic" 显式**（缺省已改为 auto——回归对象 = unguarded 通用路径，必须显式钉住） | generic(unguarded) | 现 golden 判据不变（两 block_I 档 32/128 照跑；证明 unguarded 变体二进制保全） |
| L0-P3 | 契约违约检测（shared） | L0-P1 输入但随机 1 行索引改动 1 个元素，panel_mode="shared" | panel(unguarded) | ErrFlag 置位、host 抛 RuntimeError；同输入显式 generic 声明结果正确 |
| L0-P4a-e | varlen/边界/门控（不变） | 空 seq / ΣS=1 / P 含 -1 / 越界值 / B=32×S=64 超预算 | panel / host 回退 | 判据同 rev1（P4e 断言走了通用路径） |
| L0-P5 | 竞态压测族（#22 强制，不变） | 小 ΣS=64 × 非默认 D × topk ∈ {32,64} × ≥3 独立 run | panel(unguarded) | 每 run 判据同 L0-P1；3 run 间 torch.equal |
| L0-P6 | host 门控单元（不变） | topk=256 / B×G>8 / ΣS/B<16 构造输入 | host 分发 | 断言各 miss 分支均分发到 generic |
| **L0-P7a** | **检测 kernel 单元——分段边界与违约定位** | 直调 detect builder；golden 家族 offsets [0,1023,2048]；子用例：(1) 契约成立 → violations==0；(2) 违约置首行（r=bos+1）；(3) 违约置末行（r=eos-1）；(4) 违约置 lane 子区间边界行 ±1（按 §3.4 分区公式 `bos+1+lane·ncmp//40` 构造边界对齐违约）；(5) 多违约累计（2 个 seq 各 1 行）→ violations>0 且 ≤ 2·40；(6) 空 seq [0,0,1023,2048] + 契约成立 → 0；(7) 单行 seq（len=1 段）→ 该段零比较；(8) ΣS=1 → 0；(9) kv_group=2：g=0 违约 / g=1 成立 → >0（∃g 语义） | detect | **violations int32 精确断言**（0/单违约精确；多违约半精确区间），host Python 谓词参考（§9.1 增补）；每子用例独立 zeros |
| **L0-P7b** | **auto 端到端——panel 合格输入（不声明模式）** | L0-P1 输入，缺省调用（panel_mode 不传） | auto → panel(guarded) 执行 | vs torch ref golden 判据同 L0-P1；AND vs 显式 generic 输出 torch.equal；AND 路由探针（detect/panel_guarded/generic_guarded builder 计数各 ≥1，unguarded generic == 0，`_BUILD_CACHE` 三键在场）；附带 wall 计时信号（< generic 显著差，非阻塞） |
| **L0-P7c** | **auto 端到端——随机索引（元数据过、契约破坏）** | randint 索引（同 bench_graph_smtk 构造）+ golden 元数据（G1-G5 过），缺省调用 | auto → generic(guarded) 执行 | vs torch ref golden 判据（generic 正确性）；路由探针三连发齐发；同输入直调 detect → violations>0 |
| **L0-P7d** | **auto 元数据不合格（不声明）——零开销零变化验证** | topk=256（G2/G3 miss），缺省调用 | unguarded generic 单发 | 结果正确（L0-P4e 判据）；**探针断言：detect/panel_guarded/generic_guarded builder 零调用、`_BUILD_CACHE` 对应键缺席**（check_no_panel_route 模式扩展） |
| **L0-P7e** | **guarded-generic 变体回归 + 竞态复跑** | 直调 generic_guarded，violations 预置 [1]（强制走执行路径），复用 L0-P5 竞态族配置（ΣS=64 × D∈{128,256}/tail=64 × topk∈{32,64}）× ≥3 run；另一组 violations 预置 [0]（强制早退） | generic(guarded) | 执行组：golden 判据 + 3 run 间 torch.equal（守卫不引入竞态）；早退组：rs 预填 sentinel 值 → 调用后 rs 逐位等于 sentinel（早退不写）+ 调用正常返回无悬挂 |
| **L0-P7f** | **三模式语义验证** | 同一 golden 输入 × {auto, generic, shared} 三次调用 | 三模式 | 三输出两两 torch.equal；探针各自路由正确（generic → 仅 unguarded generic；shared → 仅 unguarded panel 无 detect；auto → 三连发） |
| **L0-P7g** | **检测 kernel 竞态压测（#22 纪律——新代码强制项）** | 大 ΣS 多 chunk/lane：ΣS=16384（每 lane ~410 比较 → 多 chunk）× topk=128 × {arange 契约成立, 中段随机违约} × ≥3 独立 run | detect | 每组 3 run 间 violations 计数 bitwise 一致（int32 相等）；契约成立版 ==0、违约版 >0；同时覆盖 pre_ub 跨 chunk 复用的相邻对保护面 |

计时（粗测，沿用测试文件 `time_wall/time_event`，精确 profiling 归 Stage 3）：L0-P7b 附带 wall 计时
打印，供 155-157us 预算带的早期信号（非阻塞）。

**L0-P7 家族执行要点**：全部阻塞判据（`[PRECISION_PASS/FAIL]`，计入退出码）；路由探针沿用
`panel_builder_probe` monkeypatch 模式（test 文件 L295-309）扩展为 detect/generic_guarded/
panel_guarded 三计数器 + `_BUILD_CACHE` 键检查（`check_no_panel_route` 泛化为 `check_auto_route`）；
单用例超时预算 ≤ 60s（检测单元 <10s；P7g 压测 ~6×1s 量级）。

### 9.3 精度标准表（不变；按 dtype 混合容差；输出 dtype 全集 = float32 单值——本算子 dtype 固定组合，无 dtype 扫描维度）

| 数据 | dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|---|---|---|---|---|---|
| ReduceSum（唯一输出） | float32 | 1e-4 | 1e-2 | **1e-3** | 0.99 |

辅助全局判据（继承 golden，与上表联合生效）：`err_ratio = RMS(err)/RMS(ref) ≤ 1e-2`；
NaN 计数 = 0。整型输入（Indices/Offsets/TokenIndices）无输出精度行（不参与数值容差）。
panel-vs-generic 等价性判据：torch.equal（逐位，§9.1 论证）。**violations（检测输出，int32）：
精确匹配判据（0/0/0/1.0）——整型输出按 tilelang-op-test-design 精度体系整型行**。

---

## 10. 性能预算分解

### 10.1-10.3 panel/generic 主 kernel 预算（rev1 保留；实测终值替代预判）

rev1 设计期预估 60-80us；**实测 143.3us（panel_metric2_bench.md，graph-replay 主口径）**，三大
归因（MTE1 串行 73us / per-task cross-flag 42us / Q 读率 38.8≠160GB/s）已在该报告 §3 定量闭环，
此处不重复。vs 220us 判定线：**PASS 余量 1.53×**；vs generic BI128 378.9us：**2.64×**。

### 10.4 auto 路径预算修订（rev2 新增）

| 输入形态（auto 缺省调用） | 预算构成 | 预估 wall | vs 220us | 相对显式模式 |
|---|---|---|---|---|
| **panel 合格（golden arange）** | panel 143.3 + detect ~3-5 + generic_guarded 空载 ~8.4 | **≈ 155-157us** | **PASS（余量 ~1.4×）** | vs shared 143.3：+12-14us（透明性代价）；余量侵蚀 1.53×→1.40×，已文档化 |
| 随机索引（元数据 G1-G5 过） | generic 全量 378.9(BI128)/471.4(BI32) + detect ~3-5 + panel_guarded 空载 ~8.4 | **≈ 391 / 483us** | FAIL（与显式 generic 同判——判定线本就未达，非 rev2 退化） | vs 显式 generic：+~12us（+3.1%/+2.5%） |
| 元数据不合格（G1-G5/域 miss） | unguarded generic 单发，路径与 rev1 逐条一致 | **378.9 / 471.4us（零变化）** | FAIL（不变） | **零开销零变化**（P7d 探针锁定） |
| guarded generic 正常执行（违约输入） | 守卫读 2×GM 标量/核 + ntask 钳位（无同步结构） | +~1-2us vs unguarded | — | 计入上行"随机索引"行 |

**detect 预算校准（golden ΣS=2048）**：载荷 Indices 1MB/40 lane ≈ 26KB MTE2 行读 ~1us + 比较链
51 行/lane × 5 op × ~10ns（panel 逐任务校验链校准：6 op × 102 task ≈ 4-8us ⇒ ~10ns/op 流水化）
≈ 2.6us + 批序言标量（≤8 seq × 2 读，L2 热）<0.5us + 尾部 atomic/分支 ~0.1us ⇒ **≈ 3-5us**。
**缩放论证**：detect 流量 = Indices 单读（1MB/2K 行）vs panel 路径 55.8MB/2K 行（流量比 1/56）
⇒ 任意 ΣS 下 detect ≈ panel wall 的 **2-4%**（两者均 ΣS 线性；ΣS=16K 时 detect ~24-40us vs
panel ~1.1ms，占比不变）——不构成大 ΣS 的预算风险。

**generic_guarded 空载 ~8.4us 的口径说明**：取 rev1 §10.3 的 launch 8.4us 口径（eager 单发射成本）
作为 graph-replay 下空 kernel device 成本的保守上界；graph 捕获消除 host 侧发射开销后实际空载
预计 2-5us（待 panel_metric2 协议复测）。若实测取上界，panel 合格输入总量 ≈ 155-157us 仍 PASS
（余量 ≥1.40×）；若空载为 2-5us，余量恢复至 ~1.44-1.48×。

**graph-replay 三连发捕获口径（metric-2 测量协议修订）**：捕获单元 = [violations 置零（捕获内，
§8.4 不变式）, detect, panel_g, generic_g]，batch=25 replay 事件计时，iters=20，2 组独立测量——
协议与 rev1 单 kernel 捕获同构，仅捕获内容从 1 kernel 扩为 4 op 序列；220us 判定对象 = 该捕获
单元 wall（auto 缺省的真实成本口径）。token_indices/rs 预构建不计时（rev1 同口径）。

---

## 11. 风险点与降级预案

| # | 风险 | 概率/影响 | 降级预案 |
|---|---|---|---|
| 1 | C-scope GM 标量读（rev1 风险，已闭环） | 已实证 ✓ | spike 一次通过（debug_log Attempt 1）；guarded generic 的 C-scope Violations 读复用同一先例 |
| 2 | flag 预算恰满 8 id（rev1 风险，已闭环） | 已实证 ✓ | rev1 实现版 8 id 生产运行（143.3us 验收）；guarded 变体零新增 id |
| 3 | L1 规划冲突（rev1 风险，已闭环） | 已实证 ✓ | 实现版未触发（306KB/512KB 规划成立）；预案保留（显式地址注解/单缓冲） |
| 4 | fixpipe/FIX 率（rev1 风险，保留） | 中/中 | msprof 实测 FIX 20us（设计预估 18 ✓ 已闭环为实测带内）；L0C 双缓冲预案保留 |
| 5 | coupling 残留（rev1 风险，实测 42us 已归因） | 已归因 | per-task F_PANEL/F_PDONE 42us 为 panel_metric2 §3.2 实测构成；2-task 批量握手预案保留（Stage 3 选项） |
| 6 | L0B 恰满 / topk>128 扩展域（rev1 保留） | — | v1 域已限 topk ≤ 128 |
| 7 | 短 seq 密集误入 panel（rev1 保留） | 低/低 | G4/G5 host 门控拦截（P4e/P6/P7d 锁定） |
| 8 | ErrFlag 检查被跳过（rev1 保留） | 低/高 | auto/shared 双模式 host 强制 `.item()` 检查（§8.3）；graph 捕获口径接受 detect 正确性为前提（与 rev1 shared 捕获口径一致——捕获本就绕过 host 检查） |
| **9** | **检测正确性责任转移（rev2 核心新风险）：detect 漏报 ⟹ 沉默错误结果** | 低/高 | **三层防御**：① 谓词极简（§1.5 纯整数链式比较、无浮点、无跨核同步、单 atomic——可审计面约 30 行 kernel 体）+ L0-P7a/g 专项族（分段边界/违约定位/多违约/varlen 边界/竞态/bitwise 稳定）；② panel 逐任务校验链保留（detect 漏报 ⟹ 该链逐行比对必捕获 ⟹ ErrFlag → RuntimeError——危险路径需"detect 漏报 ∧ 校验链同坏"两个独立 bug 合取）；③ detect 误报方向天然安全（violations>0 → generic 全量 → 恒正确，仅性能回退——不对称性使唯一危险方向被 ② 覆盖）。残余风险：两 bug 合取，测试族分别锁死每层 |
| **10** | **guarded/unguarded 双变体维护面**（generic×2 + panel×2 + detect = 5 builders） | 中/中 | 共享 body 单源构造（§3.5：guarded delta = 2 行/scope + Violations 形参；unguarded traced 语句逐条一致）；回归锁定：P2（unguarded generic 双 block_I）+ P1/P3-P6（unguarded panel）+ P7e/f（guarded 双变体）+ P7d（auto-miss 零变化探针）——任何一侧漂移即红 |
| **11** | **violations 生命周期误用**（复用/置零移出捕获 → replay 累积翻转选路） | 低/高 | 文档不变式（§8.4：每调用新建；置零 op 必须在捕获内）+ P7b 连续多次调用（累积翻转会使第 2 次调用走 generic——输出仍正确但路由探针计数变化被断言捕获）+ 代码注释红字 |
| **12** | **新机制 spike 项**：S1 out_idx=[] 装饰形态（§3.5）；S2 嵌套动态 T.serial 3 层（§7.4）；S3 detect 比较 ReduceTwo 组合（T.tile.max(1D) → reduce_max(2D, dim) 两步 2D 归约——mhc_post 等 V-only 示例交叉） | 低/中 | S1 回退：非对称形态（panel_guarded 保留 out_idx、generic_guarded 收其输出为输入——机制同、管线挪位）；S2 回退：chunk 层并入行层（单层动态界）；S3 回退：行链改 1D flat GST（pre_ub [C+1, GST] 2D + 行切片 1D——indices_ub[islot] 同款）。三个 spike 均为实现期首日冒烟项（半天内），任一失败走回退不阻塞架构 |

技术约束检测清单（rev2 增补项）：三维 Kernel ✗ 不涉及（detect 同为一维 grid，V-only 16 例先例）；
threads ✗ 不涉及（detect V-only 无 C:V 配比；主 kernels 1:2 内建）；动态循环边界 ✗ 合规（batch/
chunk/行三层 T.serial 动态界 = ntask 先例泛化，S2 spike；非 T.Pipelined）；L0C 溢出 ✗（主 kernel
8KB/128KB；detect 无 L0C）；GEMM 非整除 ✗（主 kernel 同 rev1；detect 无 GEMM）；T.copy 列 strided
✗（detect 行切片 = 运行行偏移 + 静态内维，Q[s_row, H0:H1, :] 同款；violations [1] 标量 atomic 无
切片）；host 侧 buffer 改动 ✗（§8.3 审计：zeros/empty/item 三类均为数据准备/只读，allow）；
`tilelang/language/pto.py` 废弃 ✗ 未使用；新 API 凭记忆引入 ✗（atomic_add/reduce_max/max/sub/
abs/clamp/fill/arith_progression 签名均经 ascend_tile.py / reduce_ascend.py 源码核实，§11.5）。

---

## 11.5 同类实现引用（examples/ 具体参考路径与引用点——rev2 增补粗体）

| 参考文件 | 引用点 |
|---|---|
| `examples/dsa_sparse_finetune/sparse_mla_topk_reducesum.py` | **主参考（架构真相源）**：persistent 轮询映射与 ntask 推导（L241-243，**守卫钳位的复用对象**）、gemm_v0 单段 K=576 用法（L266-269）、cross-flag 槽位/mode-2 quorum 与 DEPTH 纪律（L132-148）、V 侧 gather/相邻对保护全套（L299-574）、#22 尾体竞态注释（L577-585）、GM 标量读 clamp 安全模式（L429-439）、host 入口与 _BUILD_CACHE 元组标记键（L666-765、L1306）；**rev2 直接复用**：panel 逐任务校验链（L1088-1104——detect 比较链与 v→s/v→mte3/atomic 纪律的逐指令同构原型）、ErrFlag 普通输入写先例（L803/L1104）、`torch.zeros(1, int32, device)` 数据准备先例（L1321）、`_token_indices_from_list` off_list 复用（L1175-1193）、trace 期 Python bool 分支（L513 SIDE_PAIR、L637 OFFSET_N>1） |
| `examples/attention_sink/example_gqa_sink_fwd_varlen/example_gqa_sink_fwd_varlen.py` | C-scope GM 标量读先例（L268-270，panel spike 实证）；6-flag init/destroy 配平模式（L98-105, L384-392, L613-622）；L1 512KB 容量注释（L201）；连续分块 task_range 映射（L85-88） |
| `examples/sparse_flash_attention/example_sparse_flash_attn_dynamic_shape.py` | 同步结构同构参照（T.Scope("C"/"V") + 手动 cross-flag Expert 范式，L148 起） |
| **`examples/moe_token_permute/moe_token_unpermute.py`** | **V-only kernel + T.Kernel(actual_cores, is_npu=True) as (cid, vid) 先例（L137/L288/L398——detect kernel 的 grid/lane 结构直接参照）** |
| **`examples/linear_attention_and_rnn/gdn/gdn_chunk_cumsum.py`** | **`cid*2+vid` lane 映射先例（L24-26 by = ...*2 + vid——detect 的 40 lane 划分同款）；V-only + 动态派生索引算术** |
| `.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-kernel-memory.md` | 容量表（§2，Ascend910B3.ini 出处）、UB↔L1 不可直拷（§1/§3）、T.copy 切片限制（§3）、V 核半分模式（§4） |
| `tilelang/language/ascend.py` | `gemm_v0` BufferRegion 支持与 kL0Size 语义（L418-463）、`set_cross_flag` mode=2 quorum（L191-210） |
| **`tilelang/language/ascend_tile.py`** | **rev2 API 签名核实**：`atomic_add`（L313-349：dst 限 GM/src 限 local/dtype 含 int32）、`sub`（L1106-1127：int32 支持、in-place 别名）、`max`（L1173-1204：二元逐元素）、`abs`（L1336）、`clamp`（L2377-2423：count: PrimExpr 动态计数）、`fill`（L352-376）、`arith_progression`（L398-429：int32 行号序列备选）、binary_op 区域推导（L1030-1086：动态偏移静态 extent 先例） |
| **`tilelang/language/reduce_ascend.py`** | **`reduce_max`（L362-404：fp16/fp32 限定——detect 链 cast 到 fp32 的根据；clear 语义；tmp 显式 scratch）** |
| **`tilelang/jit/adapter/cython/cython_wrapper.pyx`** | **流序与输出机制源码实证（L86-90 current_stream 发射；L102-141 非 result_idx 参数由调用方传入、kernel 可写；L200-203 空 result_idx 返回空表——§3.5/§8.4 论证依据）** |

---

## 12. 交付清单与 changelog

| 产物 | 状态 | 说明 |
|---|---|---|
| `examples/dsa_sparse_finetune/DESIGN_panel_mode.md` | ✅ 本文件（rev2） | 分发策略修订：auto 缺省 + detect + guarded 双变体（契约/架构/内存/调度/同步/分发/回退/测试/预算/风险） |
| `examples/dsa_sparse_finetune/history_version/design_panel_mode_rev1_planA.md` | ✅ 已备份 | rev1 原文（方案 A 验收依据，143.3us PASS 证据链的出处） |
| `examples/dsa_sparse_finetune/proto.yaml` | ✅ rev2 同步修订 | panel_mode 枚举扩为 ["auto", "generic", "shared"]、缺省 auto（覆盖门禁 D-PARAM-panel_mode 三值派生源） |
| kernel 变体实现 | ⬜ Stage 2 | detect + generic_guarded + panel_guarded builders + host auto 分发（共享 body 构造；unguarded 零修改）+ spike S1-S3 首日冒烟 |
| 测试扩展 | ⬜ Stage 2 | L0-P7a..g 落地（本文件 §9.2）+ P2 显式 generic 修订；L0 通过后场景 B 扩展 L1/L2/Boundary |
| 性能验证 | ⬜ Stage 2/3 | graph-replay 三连发捕获口径复测（§10.4）：panel 合格 ≈155-157us 判定、随机索引 ~391/483、auto-miss 零变化 |

### 12.1 changelog

- **rev2（2026-10-10，design_revision）**：分发策略修订——`panel_mode` 缺省 generic → auto（设备侧检测
  + 推测三连发）。新增：§1.5 检测等价判定、§3.4 检测 kernel、§3.5 builder 族谱与共享 body 构造、
  §4.3 检测内存、§4.1 Violations/共享 rs 规格、§5.2 检测 tiling、§7.4 检测同步纪律、§7.5 守卫洁净性
  论证、§8 全章重写（三模式/方案 f 裁决/host 分发/流序与捕获/violations 生命周期）、§9.2 L0-P7 家族
  （tilelang-op-test-design 场景 A）+ P2 修订注记、§10.4 auto 预算（155-157us，余量 ~1.4×）、§11
  R9-R12 与 spike 项 S1-S3、§11.5 引用增补、proto.yaml 三值枚举。**不变**：panel/generic 主 kernel
  架构（§3.1/§3.3/§5.1/§6/§7.1-7.3）、精度标准（§9.3）、已实证结论（143.3us PASS / torch.equal /
  8-id 协议）。rev1 全文备份于 history_version/。
- **rev1（2026-10-09，first_design）**：方案 A（panel_mode="shared" 调用方声明）设计 + L0-P1..P6；
  Stage 2 实现验收（debug_log Attempt 1 precision_pass；panel 143.3us PASS 220us 线，余量 1.53×）。

### 12.2 相对上一版的关键调整（revision 模式强制项）

1. **分发契约**：调用方声明（shared）→ 设备侧自动判定（auto 缺省）；shared/generic 保留为显式覆盖，
   行为与 rev1 验收版逐字节一致（P1/P2/P7f 锁定）。
2. **守卫机制 = 方案 f（ntask 钳位）**：否决方案 e（全路径税 15-25us 被支配）；否决"单 builder 加
   kwarg"（签名破坏 + dummy 开销）——采用共享 body + 双薄包装，unguarded traced 语句逐条一致。
3. **检测 kernel = 相邻行链式比较 + (seq, lane) 整除分区**：否决首行代表比较（代表行管理复杂、L2 假设）；
   否决 host plan 张量（新增 H2D + 捕获面）——kernel 直读 Offsets（≤ G4 后 8 seq）。计数语义为
   "发现违约的 lane 数"，布尔判定 >0（过计无害/漏计为零，§3.4）。
4. **为何不会重蹈 rev1 已知风险**：rev1 风险 #1-#3（C-scope 标量读/flag 预算/L1 冲突）已在实现期
   spike + 143.3us 验收中闭环，rev2 的守卫读（C-scope Violations 标量）与 8-id 协议（零新增）直接
   继承该实证；rev1 对方案 B 的两项否决理由（沉默错误/动终态代码）分别以三层防御（§8.0/§11 R9）
   与 ntask 钳位二进制保全（§8.2/§7.5）化解——即 rev2 恰好逐条处理了 rev1 否决 B 的原因，而非绕过。
5. **性能代价显式化**：panel 合格输入 143.3 → ≈155-157us（余量 1.53×→~1.4×）——透明分价的价签
   写入 §10.4 并作为 metric-2 判定口径；随机索引元数据过输入 +~12us；其余形态零变化。

[DISPATCH-COVERAGE]
```text
supported_domain: 现 kernel 全域（dtype 固定组合；dim/tail_dim 幂等；topk%BI==0 且 BI 偶≥32；
  head_kv ∈ {16..64 幂等} ∪ {%64==0}；padded_H≠head_kv ⇒ kv_group==1；ΣS/ΣS_kv/batch 运行时任意）
generic_fallback: tl_sparse_mla_topk_reducesum / tl_sparse_mla_topk_reducesum_guarded
  （unguarded：generic 显式与一切门控 miss；guarded：auto 推测三连发——两者 traced body 除守卫外逐条一致）
specializations:
  - predicate: panel_mode ∈ {"auto"(缺省), "shared"}（显式声明或自动检测，数据内容性质）
                AND host 门控 G1(L1 容量)/G2(L0B)/G3(L0C)/G4(B×G≤8)/G5(摊销≥16)（纯 metadata：shape+offsets）
                AND 域检查沿用（编译期 assert 同集）
                AND (auto 模式额外) detect kernel 判定 violations == 0（§1.5 链式等价判定，设备侧）
    fallback_on_miss: host 门控/域 miss → unguarded generic 单发（零开销，P7d 探针锁定）；
                      auto 检测违约（violations>0）→ panel_guarded 整核早退 + generic_guarded 全量执行
                      （结果恒正确，性能回退 ~+12us，P7c 锁定）；
                      shared 运行时逐任务校验违约 → ErrFlag + host RuntimeError（显式报错，P3 锁定）；
                      auto 检测漏报（detect bug）→ panel 逐任务校验链兜底 → ErrFlag + RuntimeError（响亮报错，§11 R9 ②层）
    equivalence_evidence: gemm 同形同 K 序 ⇒ logits 逐位一致（§9.1）+ V 侧指令链同构；
                      L0-P1 panel-vs-generic torch.equal 实证（rev1）；L0-P7b/f auto 路径复验
    unsupported_inputs: 无（panel 谓词 ⊂ 现域，不收缩支持域；auto 对任意输入结果正确）
  result: pass
```
