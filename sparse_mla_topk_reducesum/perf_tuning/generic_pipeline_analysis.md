# generic 路径深度流水剖析（random 索引 · BI128 主战场）

**mode=perf_analysis（只测只析）｜2026-10-10｜msprof pipe 级重构 + 可优化点判定 + 天花板**

任务：用户否定 panel 特化方向后，聚焦 **generic 路径在任意（随机）索引下的流水级瓶颈**——即"中转税总量已封死（四路判死）的前提下，提高中转税的流水效率"还剩多少空间。用 msprof（与 MindStudio 流水图同引擎）的 pipe 占用/停顿/flag 等待数据重构流水图。

---

## 0. 协议与数据源

| 数据组 | 配置 | Task Duration | 用途 |
|---|---|---|---|
| **random_BI128** | golden shape, random causal 索引 seed=123 | **440.8µs** | 主剖析对象（≈ G3 锚点 441.7，本会话配对基线 432.8） |
| arange_BI128 | 同 shape, arange 索引 | 380.9µs | 对照（L2 局部性差异隔离） |
| random_BI32 | BI=32 | 513.4µs | 副剖析（粒度敏感性 + 自然实验：BI32 无 SIDE_PAIR） |
| **spike SIDE_PAIR=False** | random_BI128, /tmp 副本仅改 1 行 | 431.2µs | 决定性差分实验（见 §4） |

- msprof op 深度档位：`--aic-metrics=PipeUtilization,Memory,L2Cache,MemoryL0,Occupancy,MemoryDetail` → 产出 PipeUtilization / Memory / L2Cache / ResourceConflictRatio / ArithmeticUtilization CSV；**MemoryL0 / MemoryUB / Occupancy / MemoryDetail 为空表（真机不支持的档位，如实降级）**
- **TimelineDetail 档位仅仿真产品有效（msprof help 明示），真机无时间线 trace** → 本文"流水图"由 ①每核每引擎 busy/stall 聚合 ②每块节拍（cadence）模型 ③spike 差分 三路重构，非直接 trace（证据等级：msprof 实测 + 模型重构）
- PipeUtilization.csv 粒度：**60 行 = 20 AIC（cube0）+ 40 AIV（vector*）每核一行**（首证于本次，此前会话按跨核均值解读）
- 计时口径：graph-replay 配对 A/B（σ≤0.12% 会话内）；msprof Task Duration 交叉验证；跨会话漂移 ±2-3%

**剖析对象编译期结构（BI=128/D=512/topk=128，源码 L150-216 推导）**：

| 常量 | 值 | 含义 |
|---|---|---|
| NI | 1 | 每 task 1 块（BI=topk） |
| DEPTH | 4 | ws slot 数；**flag 预算已用满**（F0=id0-3, F1=id4-7，A2 FFTS 8 id 合法上限） |
| LEAD | 2 | V 前瞻 2 块（OFFSET_N=2） |
| SIDES | **2**（降级） | 3×73.7KB=221KB > `_kv_gather_budget=170KB`（UB 192KB）→ **SIDE_PAIR=True** |
| IDXSLOTS | 4 | idx 跨 body 预取槽 |

---

## 1. 流水图数据版（核心交付）

### 1.1 C 流（AIC，20 核，102.4 块/核，每块节拍 4.25µs）

每块程序链：`waitF0(slot)` → `MTE2 ws_kv→kv_l1`（147KB+Q 18.4KB）→ `gemm_v0`（内部 MTE1 L1→L0 乒乓 + Cube + 尾部 MTE1→MTE2 排空 flag）→ `FIX L0C→ws_acc` → `setF1(slot)`。

| C 侧引擎/事件 | 均值µs | 占墙钟 | 每块 | 证据等级 |
|---|---|---|---|---|
| aic_time（墙钟） | 435.1 | 100% | 4.25 | msprof |
| **MTE2（ws+Q → L1）** | **332.5** | **76%** | **3.25** | msprof |
| MTE1（L1→L0 fractal） | 73.2 | 17% | 0.71 | msprof |
| Cube（mma） | 28.6 | 7% | 0.28 | msprof |
| FIX（L0C→GM） | 33.7 | 8% | 0.33 | msprof |
| **scalar_mte1_stall** | **394.4** | **91%** | 3.86 | msprof（见 §2 解读） |
| wait F0（等 V，id0-3） | 18.6 | 4% | 0.18 | msprof |
| gemm 链暴露（MTE1+Cube+FIX−重叠） | ~100 | ~23% | ~0.98 | 差分推导 |

**C 侧速率锚点**：ws 读 147KB/3.25µs = **45.4GB/s/AIC**（≈ E1 dirty-burst 天花板 46GB/s，**贴死**）；arange 对照 52.7GB/s（V 侧随机行流消失 → L2 slice 争用缓解，差 +16% 证明 random 下 C 读速被 V 的随机 gather 争用拖低）。MTE2 引擎空闲 24%（103µs/核）= gemm 链暴露 + flag 间隙。

### 1.2 V 流（AIV，40 核，102.4 块/核，每块节拍 4.26µs）

每 body 程序链：`[SIDE_PAIR: 等MTE3全排空]` → `MTE2 行 gather×64`（填 side）→ `idx 预取` → `mte2→mte3 marker` → `bulk MTE3 → ws_kv slot` → `setF0(mode-2)` → `post: waitF1(slot)`（**scalar 阻塞**）→ `MTE2 acc 载入` → `vec mask/exp/reduce` → `MTE3 store`。

| V 侧引擎/事件 | 均值µs | 占墙钟 | 每块 | 证据等级 |
|---|---|---|---|---|
| aiv_time（墙钟） | 435.5 | 100% | 4.26 | msprof |
| **MTE2（行 gather + acc + idx）** | **291.2** | **67%** | **2.84** | msprof |
| MTE3（bulk 73.7KB + store） | 103.9 | 24% | 1.01 | msprof |
| Vector（mask/exp/reduce） | 41.7 | 10% | 0.41 | msprof |
| **引擎和/墙钟** | **443.0/435.5 = 1.004** | — | — | **MTE2/MTE3/vec 近零重叠（严格交替）** |
| wait F1（等 C，id4-7） | 176.8 | 41%（标量） | 1.86 | msprof |
| aiv_scalar_time | 363.4 | 83% | 3.55 | msprof |

**V 侧速率锚点**：行 gather 64 行/2.84µs = **44.4ns/行**（E1 纯 gather 27-31ns/行 → ~13-17ns/行是标量 GetValue+发射链，非引擎极限）；bulk MTE3 61.8GB/s；L2 行为：**AIV 读命中 99.6%**（KV 源 2.25MB 全 L2 热）、写命中 95.8%。

### 1.3 关键修正：F1 wait 的真实身份（"耦合 105µs"落地）

- **F0 wait（C 等 V）仅 18.6µs**（且集中在 id0=kernel 启动段；稳态 id1-3≈0）→ **C 几乎从不饿**
- **F1 wait 176.8µs 不是 V 侧空转**：引擎和=墙钟证明 V 引擎零空闲——F1 等待期间 MTE3 正在排空刚发射的 bulk（1.86µs ≈ bulk 排空 1.0-1.6µs）+ 前瞻 gather 排空，**标量等待被引擎排空吸收**
- **真正的间隙结构：两侧各自内部串行**（C：gemm_v0 flag 链；V：SIDE_PAIR 全队列排空 + F1 标量门控发射流），两条串行链**恰好均衡在 ~4.25µs/块** → 系统墙钟 = 102.4 × 4.25。旧归因"耦合放大 105µs"修正为：**~100µs（C 的 gemm 链暴露）+ ~105-145µs（V 的 MTE3+vec 暴露，被共门控掩盖）的"双侧内部串行暴露"，非跨核 flag 互锁**（flag 本身近似免费）

### 1.4 每核分布（尾块/guard 空转排查）

| 指标 | min | med | max | 极差 |
|---|---|---|---|---|
| AIC 墙钟（20 核） | 431.7 | 434.8 | 438.3 | **±0.8%** |
| AIV 墙钟（40 核） | 432.0 | 435.3 | 438.9 | ±0.8% |
| AIV MTE2 | 273.7 | 292.8 | 300.7 | ±5%（L2 slice 运气差异） |
| AIC MTE2 | 328.2 | 333.2 | 335.4 | ±1% |

**判定：尾块不均 / guard 空转均 ≤1%，非可攻击点**（本剖析为 unguarded 直调，无 guard 空转项）。

### 1.5 对照组与 BI32 自然实验

| 指标 | random BI128 | arange BI128 | random BI32 |
|---|---|---|---|
| 墙钟µs | 435.1 | 375.5 | 503.3 |
| C MTE2 | 332.5 (76%) | 285.6 (76%) | 265.9 (53%) |
| C MTE1/Cube/FIX | 73.2/28.6/33.7 | 73.2/28.6/27.5 | 105.1/56.2/68.3 |
| **C F0 wait** | **18.6 (4%)** | 4.8 (1%) | **229.7 (46%)** |
| V MTE2 | 291.2 (67%) | 238.7 (64%) | 373.2 (74%) |
| V MTE3/vec | 103.9/41.7 | 95.4/41.7 | 156.7/123.1 |
| **V F1 wait** | **176.8** | 175.5 | **9.3 (2%)** |
| **V 引擎和/墙钟** | **1.004（串行）** | 1.000（串行） | **1.30（重叠！）** |
| V scalar | 83% | 84% | **91%** |
| L2 AIC/AIV 读命中 | 91.3%/99.6% | —（未采） | —（未采） |

- **BI32 = V 侧软件流水的天然实验**：SIDES=4≥LEAD+1 → SIDE_PAIR=False → 引擎重叠率 1.30；但 V 变成 **scalar 发射瓶颈（91%）**（16 行/段的 GetValue/flag/每 task 标量重载摊不平），C 链 1.23µs/块 × 409.6 块同样 ~504µs → **BI32 双侧共门控（C 链 + V 标量），比 BI128 差 15%**
- **arange vs random 的 -60µs 分解**：C ws 读速 52.7→45.4GB/s（V 随机行流 L2 争用，-47µs）+ V gather 36.4→44.4ns/行（L2 行复用消失，-52µs）——**random 税 = 两侧 MTE2 各 ~+16% 引擎时间**，属中转税总量范畴（调度不可恢复）

---

## 2. C 侧串行化的源码级根因（common.h 实读）

`src/tl_templates/ascend/common.h` gemm_v0 模板（L1306-1373 区域）：

```
头： SetFlag<MTE2_MTE1>(ev); WaitFlag<MTE2_MTE1>(ev)   ← MTE2 全队列排空后 MTE1 才启动
循环： per k-tile 乒乓 [WaitFlag<M_MTE1> → L1→L0A/B 拷贝 → SetFlag<MTE1_M> → mma]
尾： WaitFlag<M_MTE1>×2 → SetFlag<MTE1_MTE2>(ev); WaitFlag<MTE1_MTE2>(ev)  ← MTE1 全队列排空后下一 MTE2 才入队
```

- 头/尾均为**队列粒度排空 marker（单一 event id）**→ 跨块的 MTE2(n+1)↔MTE1(n) 重叠被模板硬编码锁死；唯一重叠窗口 = MTE1 排空后的 Cube 尾+FIX（实测 ~0.6µs/块）
- **scalar_mte1_stall=394.7µs（91%）的解读**：SetFlag/WaitFlag 指令入队到各 pipe、scalar 不停——scalar 冲在前面把指令全部发射，MTE1 发射队列被 flag 链堵满 → scalar 停在"队列满"上。**这是引擎 flag 链串行的指示器，不是独立成本**（scalar 无别的事可做）
- 推论：**kv_l1 双缓冲单独无效**——尾部 MTE1_MTE2 等待按队列（而非按 slot）阻塞下一 MTE2，buffer 数量不改变 flag 语义

---

## 3. 双侧串行链共门控模型（汇总）

```
系统节拍 = 102.4 块 × max(C 链, V 链) ≈ 102.4 × 4.25µs = 435µs

C 链（4.25µs/块）：[MTE2 3.25 (贴 L2 dirty 46GB/s 天花板)] → [MTE1 0.71 → Cube 0.28] → [FIX 0.33 ∥ 下一 MTE2 头 0.6]
   暴露：gemm 链 ~0.98µs/块（gemm_v0 队列粒度 flag，源码级锁死）
V 链（4.26µs/块）：[SIDE_PAIR 等 MTE3 排空] → [MTE2 行 2.84] → [marker] → [MTE3 bulk 1.01] → [F1 标量等待 1.86∥bulk 排空] → [acc 0.26 + vec 0.41 + store 0.1]
   暴露：MTE3+vec ~1.4µs/块（SIDE_PAIR 全队列等待 + F1 标量门控发射流）
   引擎极限：MTE2 行服务 2.0µs/块（31ns/行）+ acc/idx 0.36 → ~3.2-3.4µs/块可达
```

**单侧修复 ≈ 零收益**（另一侧仍在 4.25µs/块把墙钟锁住）——spike 已实证 V 侧单修无效（§4）；C 侧单修按对称性同样无效（V 引擎和 4.43 > 3.7）。**必须双侧同时修**。

---

## 4. 决定性差分实验：SIDE_PAIR 移除 spike（/tmp 副本，仅改 1 行，examples/ 未动）

**假设**：SIDE_PAIR 的 mte3 全队列排空等待是 V 侧 MTE2/MTE3 串行主因；移除后（自然排空论证：side 复用周期 2 body ≈ 8.5µs ≫ bulk 排空 ~1.6µs，余量 5×）V 引擎重叠率应升、wall 若 C 绑死则不变。

| 指标 | adopted | spike SIDE_PAIR=False | Δ |
|---|---|---|---|
| 精度（4 例含 2 尾形） | — | **4/4 torch.equal=True 逐字节** | **自然排空安全性实证 ✓** |
| graph-replay 配对（2 组交错） | 432.8µs | 433.6µs | **+0.2%（噪声内，wall 不变）** |
| msprof 墙钟/核 | 435.1 | 425.5 | -2.2%（跨进程漂移带内） |
| V MTE2 | 297.8 | 286.6 | **-11.2µs（发射流平滑）** |
| C MTE2 | 333.6 | 324.5 | **-9.1µs（V 流平滑 → L2 争用缓解）** |
| V F1 wait | 176.8 | 183.1 | +6.3µs（V 前移 → 多等 C）✓ |
| **V 引擎和/墙钟** | 1.04 | **1.04（未升！）** | **假设部分证伪** |

**结论（对假设的修正）**：SIDE_PAIR 移除安全且平滑了发射流（两侧 MTE2 各 -3%），但 **V 侧串行的第二把锁是 `post(n)` 中段的 `wait_cross_flag(F1)`——跨核 flag 只能 scalar 阻塞等待（无法入队到 MTE2 pipe），下一 body 的 gather 发射被锁在 C 节拍上**，引擎重叠压回 1.04。V 侧要真正到达引擎极限需 **SIDE_PAIR=False + LEAD=3**（把下一块 gather 的发射点移到 F1 等待之前）；LEAD=3/SIDES=2 的 side 复用余量同为 2 body 周期（spike 已验证的安全性形态）。**wall 不变 = C 侧绑死的直接实证**。

---

## 5. 可优化点评估（证据 → 预期收益 → 910B3 判定）

> 已封死四路不重提：R2 手工流水（生产 +12.7% 回退）、R5 AIC 直采（ND→zN 100ns/行）、反转 Q（Q 重载墙 + 202ns 散射原子）、L0C→UB fixpipe（A2 无 API）。以下为**调度级（流水效率）新评估**。

| # | 优化点 | 证据 | 预期收益 | 判定 |
|---|---|---|---|---|
| **P1** | **C 侧 MTE2↔gemm 链重叠**：gemm_v0 尾部 MTE1_MTE2 改 per-slot event（2 个 id）+ kv_l1 双缓冲 | common.h 源码（队列粒度 flag）+ mte1_stall 394µs + MTE2 空闲 24% | C 链 4.25→**3.55-3.7µs/块**（MTE2 引擎贴 45.4GB/s 满占空比）；**单修 ≈0（V 绑死 4.43）**，与 P2/P2b 组合后系统 -8~-14% | **可攻击，但为框架级改动**（common.h 模板 + 算子 kv_l1[2]，L1 303KB<511KB ✓）；非算子本地 |
| **P2** | **V 侧 SIDE_PAIR=False @SIDES=2** | spike 实证：精度 4/4 exact、发射流平滑（双侧 MTE2 -3%）、安全性余量 5× | 单修 ~0（C 绑死）；**P1 的必要前提**，1 行改动，风险已验证 | **可攻击（算子本地，已 spike 验证）** |
| **P2b** | **V 侧 LEAD=3**（gather 发射点前移越过 F1 标量等待） | F1 标量门控实证（spike overlap 1.04 未升）；引擎极限 3.2-3.4µs/块 vs 现 4.26 | V 链 4.26→**3.3-3.5µs/块**；单修 ~0（C 绑死）；组合后系统 -8~-14%。代价：标量窗 2→4 组 task 标量（+2 次 GM 标量读/body） | **可攻击（算子本地，中等工作量，需实现验证）** |
| **P1+P2+P2b** | **双侧同修** | §3 共门控模型 + spike | **441→~375-395µs（-11~-15%）**；乐观（C 读速回 52.7GB/s）~350 | **唯一有实质收益的组合，总改动 = 框架 1 处 + 算子 2 处** |
| P3 | DEPTH 4→6+ | flag id 预算：F0 0-3 + F1 4-7 = 8 id 已用满（A2 FFTS 8 合法） | 0（且 F0 wait 18µs / F1 被排空吸收——无气泡可填） | **架构封死（flag 位宽）** |
| P4 | 握手批量化（F1/WSF per 2-task） | F0 wait 0.18µs/块；F1 与 bulk 排空重叠 | <0.1µs/块（标量节省，全被引擎吸收） | **无收益（无可批量的暴露等待）** |
| P5 | MTE 描述符/burst 整形 | bulk 写已单条 73.7KB@61.8GB/s；C 读已单条 147KB@45.4（dirty 天花板）；V 行 44.4ns 中 ~13-17ns 是标量发射非描述符形状；910B3 无 gather-DMA（NpuArch 2201） | ≈0（形状已最优）；标量发射优化被 P2b 包含 | **架构封死（无 gather-DMA）+ 已最优** |
| P6 | 任务→core 重排 | 每核墙钟极差 ±0.8%；ws 每核私有；KV 全 L2 热 | ≤1%（尾块） | **无收益** |
| P7 | V/C 负载再平衡 / BI 粒度 | 跨侧无可迁移工作（gather 只能 V、ws 读只能 C——内存层级强制）；BI128 最优（BI32 双侧共门控差 15%；BI>topk 不可能） | 0 | **架构封死（内存层级 + 形状边界）** |
| P8 | Q 载入双缓冲 | Q 18.4KB/块 ∈ C MTE2 队列，被同一尾部 flag 链阻塞 | 随 P1 免费获得（~0.3µs/块） | **并入 P1** |

**新角度（bubble 位置揭示）**：本剖析发现的全部可回收 bubble 都在**各侧内部**（C 的 gemm 链 ~100µs、V 的 MTE3+vec ~105-145µs），**跨核 flag 间隙本身已近似免费**——这与"加深 DEPTH/批量握手"类优化方向的直觉相反，是本次最重要的结构性结论。

---

## 6. generic 路径 random 索引现实天花板

| 层级 | 值 | 依据 |
|---|---|---|
| 当前 | **441.7µs**（锚点）/ 432.8（本会话配对） | 实测 |
| P1+P2+P2b 全修后 | **~375-395µs**（-11~-15%） | 共门控模型 + spike（模型估算） |
| 完美流水极限（双侧引擎满占空比） | **~350-370µs** | C MTE2 引擎时间 332-334µs（302MB ws @ 45.4GB/s/AIC × 20 = 332µs，贴 E1 dirty 天花板）+ 启动/同步 ε |
| **220µs 目标** | **不可达（差 1.6-1.8×）** | ws 302MB 往返的 C 侧读引擎时间 332µs > 220µs；消除往返 = 架构级四路已判死；C 读速即使回到 arange 级 52.7GB/s 也仅 ~286µs |

**两层次诚实框架**：
- **中转税总量（架构级）**：random 索引下 C 侧 ws 读引擎时间 332µs 是调度不可触碰的地板（45.4GB/s/AIC 已贴 L2 dirty 服务率；arange 能到 52.7 但那是数据形态红利=panel 方向，用户已排除）
- **流水效率（调度级，本文主战场）**：当前 441 vs 完美 ~350-370 → **可回收空间 ~70-90µs（-16~-20%），但要求框架级 gemm_v0 flag 改造（P1）与算子级两处改动（P2/P2b）同时落地**；单侧修复经实证/对称性分析均为 ~0

---

## 7. 不确定性声明

1. **无直接时间线 trace**（TimelineDetail 仅仿真产品）——流水图由聚合指标+节拍模型+spike 差分重构；每块内各引擎的精确起止时刻为模型推断（误差 ~±0.1µs/块）
2. spike 单会话、精度 4 例（含 S=64/200 尾形）——自然排空安全性为**经验支持**（余量 5×）而非形式化证明；上生产前建议跑 `--level all` + 覆盖检查
3. C 侧 MTE2 45.4GB/s 与 E1 46GB/s 天花板的一致性存在 ±2% 拟合松散；arange 52.7GB/s 说明该"天花板"对 L2 争用状态敏感（非恒定硬件常数）
4. P1 预期收益为源码结构 + 引擎占空比模型推导，**未实测**（需框架改动，超出本模式权限）；R2 教训（微基准收益不迁移）在此的对应风险：per-slot flag 改造后 MTE1 的 L0A/L0B 乒乓行为需回归验证
5. 跨会话漂移 ±2-3%（本会话 adopted 432.8 vs 锚点 441.7）；所有配对结论以同会话交错 A/B 为准
6. BI32/arange 组未采 L2Cache 档位（时间预算）；arange 的 L2 命中推断基于 KV 工作集 2.25MB ≪ L2 的结构论证
