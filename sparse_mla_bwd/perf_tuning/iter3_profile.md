# sparse_mla_bwd iter-3 画像（rev3 交付态，仅画像轮，无代码改动）

## 0. 口径与样本

- 对象：`sparse_mla_bwd.py` rev3（flattened produce/consume STAGGERED pair-loop）。
  md5 `625c3ba1…`（与画像采集时 `/tmp/opencode/b3i3/stock.py` 逐位一致，解析前已核对）。
- 全接口口径（stage3 任务书 §1）：`sparse_mla_bwd(q,kv,o,do,indices,lse,offsets,delta=None)`，
  B=1,S=2048,H=64,HKV=1,DQKV=576,DV=512,topk=512,bf16，seed=0 主仓口径输入。
  do_bench(warmup=250,rep=100) mean = **28762.2us**（本会话 11:12 复测；debug_log 交付基线
  28965us，漂移 −0.7%，同 rev3 态）。
- msprof 形态（沿用 iter-2）：op-profiling 采集 k2（`_run_bwd` 直接发射，考核 shape），
  数据 `/tmp/opencode/b3i3/msprof_op/OPPROF_20261016102712_*/main_kernel/{0,1}/`
  （0/1 = 同一 k2 的两次采样，全指标一致，差 0.8%）。k2 task **28435.3us / 28206.8us**
  = 全接口的 98.9-99.0%；k3+双发射 ≈ 330us。本轮解析脚本 `parse_prof.py` + 字节核算。
- 结构参数：grid=20 MIX_AIC_1_2（20 AIC cid + 40 AIV vid），每 cid 迭代数
  `kb ≈ 412-416`（`(⌈2048/20⌉)×NI + NI`，NI=4），每 AIV 处理 v_block=32/64 头行。

## 1. 时间构成表（k2，per-core 均值，采样 0；采样 1 括注）

### AIC（busy 27734.8us）

| 分项 | us | % busy | 说明 |
|---|---:|---:|---|
| cube（GEMM 纯算） | 1997.0 | 7% | G1-G6 全部发射仅 2ms——算力大量闲置 |
| MTE1（ws→L1 读） | 1883.7 | 7% | 158.1MB @ **84GB/s**（ws 足迹 ~47MB 全 L2 命中）；**1040B/insn** × 155,651 insn（≈377/迭代，Nd2Nz 碎片化） |
| MTE2 | 2019.1 | 7% | 4,610 insn |
| fixpipe（L0C→GM） | 2845.6 | 10% | 229.4MB @ **80.5GB/s**（ws_s/dp 数据+dummy 64KB + ws_dq 144KB + ws_dkv 288KB /迭代） |
| scalar | 4839.2 | 17% | |
| scalar mte1_stall | 5001.4 | 18% | 微事务 MTE1 供给不足 |
| **跨核 flag 等待（重叠）** | **20444.8** | **74%** | 见下明细 |

AIC 等待明细（id→腿映射由生成源码 `kernel_src.c` 的 CrossCoreSet/WaitFlag 邻域实证）：

| id | 腿（V→C） | us | % busy |
|---|---|---:|---:|
| **id8** | **V 的 ws_kv stash 生产（gather 批量写后置位）** | **18430.2**（采样1: 18834） | **66%** |
| id10 | V 的下一 token gather staging 写（ws_kv 256/64） | 1089.6 | 4% |
| id4 | V 的 ws_dpb(hi) 生产 | 739.4 | 3% |
| id11/9/5 | V 的 ws_p(lo)/ws_kv mini/ws_dpb(lo) | 49-85 | <1% |

### AIV（busy 27766.3us，临界路径，占用 ~100%）

| 分项 | us | % busy | 说明 |
|---|---:|---:|---|
| MTE2（GM→UB 读） | 12148.3 | 44% | 137.7MB @ **11.3GB/s**；**1826B/insn** × 77,198 insn（≈187/迭代） |
| MTE3（UB→GM 写） | 5776.9 | 21% | 98.6MB @ **17.1GB/s**；3612B/insn × 27,950；mte3_wait_ratio 24.5%（供给断续——排空特征） |
| scalar | 3788.0 | 14% | scatter 64 行/迭代 j-loop（每行 2×GetValue+if）+ 双段地址算术 |
| vec | 1306.4 | 5% | P 链 ×2（produce+consume 重算）+ delta + mask |
| **跨核 flag 等待（重叠）** | **8490.1** | **31%** | 见下明细 |

AIV 等待明细：

| id | 腿（C→V） | us | % busy |
|---|---|---:|---:|
| id1 | C 的 ws_dkv tail 生产（散射行读前置） | 3221.1 | 12% |
| id14 | C 的 ws_s 生产相（**0 距离同 token 依赖**，§5） | 2255.2 | 8% |
| id6 | C 的 ws_dq 生产 | 2211.3 | 8% |
| id2 / id7 | ws_dp / ws_dq tail | 584.5 / 218.0 | 2% / 1% |
| id15 / id3 | consume 相 ws_s / ws_dp | **≈0（<1us）** | — |

**id15/id3 归零是 stagger 对消费链生效的直接证据**；V 的剩余等待集中在 dkv(3.2ms)、
s 生产相(2.3ms)、dq(2.2ms)。

## 2. MTE 各腿带宽（per AIV；字节量由源码核算，与实测总量闭合 ±6%）

读腿（137.7MB / 12.15ms，聚合 11.3GB/s）：

| 腿 | MB | 占比 | 事务粒度 | 估时 | 估带宽 |
|---|---:|---:|---|---:|---:|
| **ws_dkv 读回**（fp32 槽，2KB+256B/行 × 64 行/迭代） | 59.5 | 43% | 微读 | **~6.1ms** | ~9.7GB/s |
| **KV gather**（1152B 散行 × 64 行/迭代） | 29.7 | 22% | 微读 | ~3.0ms | ~10GB/s |
| ws_dq（bulk [128,32]×2/迭代） | 29.7 | 21% | 整 2D | ~1.65ms | ~18GB/s |
| **ws_s/ws_dp 重读**（produce+consume 两腿，[128,32] bulk） | 26.4 | 19% | 整 2D | ~1.5ms | ~18GB/s |
| Q(P1a 探针)/lse/idx | 0.7 | 1% | — | ~0.1ms | — |

写腿（98.6MB / 5.78ms，聚合 17.1GB/s）：

| 腿 | MB | 占比 |
|---|---:|---:|
| ws_dkv 原子散射（2304B/行 → fp32 终端区） | 60.8 | 62% |
| ws_kv stash（gather 批量 2D 写） | 29.7 | 30% |
| ws_p/dpb hi+lo | 6.6 | 7% |
| dQ 写回（token 边界） | 3.8 | 4% |

C 侧：MTE1 158.1MB @84GB/s（L2 命中，ws 往返不走 DRAM）；fixpipe 229.4MB @80.5GB/s。
注：bulk/微事务分拆带宽为 iter-2 锚点（12.2GB/s 微事务饱和、fwd bulk 18-20GB/s）+
总量闭合反推的估计值，msprof 无 per-leg 计数；聚合值与字节量为实测。

## 3. 排空探针（P1a/P1b/P2/P3）定量

- 事件数：每 AIV ≈ 927 次（P1a+P1b 各 412、P2 412、P3 103），即**每迭代 3-4 次强制排空**
  （P1a: MTE3→MTE2；P1b/P2/P3: MTE3→V）。
- **直接成本 <0.1ms（可忽略）**：P1a 0.47MB GM 读 + 927×24KB UB↔UB cast ≈ 22MB UB 流量，
  占 vec 元素量 ~4.5%。
- **串行化成本（主项）实测下界 1.35ms**：与 iter-2 做同字节构成对照——两版 AIV 读腿的
  bulk（56.1MB）/微事务（~89MB）分拆**完全一致**（iter-2: s/dp 13.2+dq 29.7+pst 13.2；
  rev3: s/dp 26.4+dq 29.7），MTE2 却从 10.8ms→12.15ms（有效带宽 13.6→11.3GB/s，−17%）。
  归因：P1a 每迭代打断 gather(MTE2)/stash(MTE3) 的跨迭代重叠。
- 结构上界 ~4-5ms（MTE3 5.78ms 若全部藏入 MTE2 重叠）；mte3_wait_ratio 24.5% 佐证
  MTE3 供给断续。**中心估计 2-4ms（k2 的 7-14%）**。

## 4. P 重算链成本（consume 相，替代 rev2 pst 档）

| 项 | 成本 | 依据 |
|---|---:|---|
| vec 增量 | +0.32ms | 1.31ms vs iter-2 pst 档 0.99ms；~25 tile ops [32,128] × 412 迭代 |
| ws_s/ws_dp consume 重读腿 | ~1.16ms | 13.2MB @11.3GB/s 聚合（bulk 口径 ~0.73ms） |
| C 侧配对 dummy 写 | ~0.26ms | 32KB×2/迭代 fixpipe（非临界路径，C 空闲 66%） |
| **合计** | **~1.5-1.8ms（k2 的 5-6%）** | 远小于其替换的 rev2 pass-2 GEMM 重算发射负债（+39.5% 预估） |

## 5. 与 iter-2 画像对照

| 指标 | iter-2（两遍锁步） | iter-3/rev3（交错 pair-loop） | Δ |
|---|---|---|---|
| k2 task | 28549us | 28435 / 28207us | ≈0（−0.2%） |
| AIC flag 等 | 20.4ms/73%（id7/id9 19.8ms kv stash） | 20.4ms/74%（id8 18.4ms kv stash） | 结构等价：V 的 gather+stash 仍是流水线头 |
| AIV flag 等 | 10.7ms/38% | 8.5ms/31% | **−2.2ms**：消费链 s/dp 等待归零（stagger 生效） |
| AIV MTE2 | 10.8ms@12.2GB/s（~147MB） | 12.15ms@11.3（137.7MB） | **+1.35ms**：探针串行化（同字节构成，§3） |
| AIV MTE3 | 5.6ms | 5.78ms | +0.2ms |
| AIV vec | 0.99ms | 1.31ms | +0.32ms：P 重算链回归 |
| AIV scalar | 2.7ms | 3.79ms | +1.1ms：pair-loop 每迭代双段地址算术+发射 |

**结论**：rev3 把 V 的等待 −2.2ms 换成了 V 的忙时 +2.8ms（探针串行化 +1.35、P 重算
+0.32、scalar +1.1）——净持平。pair-loop 的重叠收益被每迭代排空与重算吃掉；结构性
主项（V gather+stash 为头、C 66% 空闲）与 iter-2 相同。

**流水线距离发现**：id14（2.26ms）是 V produce(t+1) 的 P 链对 C s(t+1) 的 **0 距离同
token 依赖**——当前 stagger 距离 1 token 只藏住消费链，藏不住生产链；需 2-token 错位
（PARITY=2 槽已具备，flag 配对序待证）。

## 6. 优化候选 top5（按预期收益×可行性排序）

1. **排空探针削减**（P2/P3 合并或 P3 条件化、P1a 弱化为跨迭代一次）：中心 2-4ms
   （实测下界 1.35ms）。依据 §3 同字节构成对照。风险：探针是 rev3 fix#4 的正确性机制
   （跨分支管道序），弱化需重证 RAW 序；A/B 成本低（变体+smoke 即可定量）。
2. **ws_dkv 贡献槽 bf16 档复评**（env `SPARSE_MLA_BWD_DKV_SLOT=bf16` 已留）：读回
   59.5→~30MB ≈ **−3.0ms** MTE2（微事务速率不变口径）；散射 MTE3 与 C fixpipe 同步减半
   （C 非临界）。风险：R14 精度门（域 A ratio 0.965-0.983 < 0.99，2026-10-15 FAIL 记录），
   需域 A 复测；批量 staging 仍 UB 封死（196352B 顶满，iter-2 P1 追溯证伪）。
3. **Expert/混合模式手动 flag + stagger 加深至 2 token**：id14 2.26ms 归零 + V 剩余等待
   （8.5ms 池）结构化压缩 + 消除每迭代 AUTO 配对 dummy；fwd iter-5 先例 −16.3%。依据：
   id14 为 0 距离依赖（§5），AUTO 栈无解（RC1-RC5）。架构级改动，按 agent 约束需本轮
   先穷尽 AUTO 内候选——排 3 位。
4. **KV gather 腿事务放大**（64 行/迭代散读 → 32 行批量 gather_mask / 行组 2D）：~3.0ms
   腿预期 −1~−1.5ms。依据：fwd probe2 批量 2D ×3.6 带宽锚点；KV 全量 2.36MB L2 驻留，
   瓶颈是事务数（187 MTE2 insn/迭代）非带宽。
5. **pst 档复活**（P/dP_raw GM stash 替代重算，workspace 化解决跨分支同步）：省重读
   1.16ms + vec 0.32ms，付 GM transit 写 ~0.8ms + 2 flag 对（16/16 已满需重排）；净
   −0.5~−0.7ms。优先级最低。
   （备选：scatter j-loop 标量削减——per-row 条件与 P 链 m1_ub 同源可复用，−0.3~−0.8ms。）

**目标差距判断（供 Orchestrator 裁决，非本轮结论）**：28762us vs 目标 7680us 差 3.7×。
V 临界路径管道功（MTE2 12.15+MTE3 5.78+scalar 3.79+vec 1.31 ≈ 23ms，理想重叠下限
~19ms）与读量下限（237MB @ fwd bulk 18-20GB/s ≈ 12-13ms）**在当前数据流下均 >12ms**——
即使 top1-2 全部兑现（−5~7ms → ~21-23ms）仍距目标 3×。iter-2 的 [DESIGN_ERROR] 证据
（transit 融合/C 侧驻留/精度门控 bf16 transit 等数据流级削减为必选项）在本 rev3 画像下
成立并强化。

## 7. 测量有效性记录（如实）

- 画像数据 2026-10-16 10:26-10:27 采集于 /tmp/opencode/b3i3/（msprof op-profiling，k2
  driver 形态同 iter-2：`_run_bwd` 3 次调用、考核 shape、seed=0）；本轮仅解析+源码核算，
  解析前 md5 核对模块与当前 rev3 逐位一致，无需重跑、无代码改动。
- 两个 k2 采样（main_kernel/0、/1）差 0.8%，全指标一致（id8 18.43/18.83ms、MTE2
  12.15/11.97ms、vec 1.31/1.31ms）。
- 字节核算闭合：V 读预期 146.1MB vs 实测 137.7（−6% = OOB 行条件跳过）；V 写 100.9 vs
  98.6；C fixpipe 231 vs 229.4；全接口 28762.2 与 k2 28435.3 闭合（k3+发射 330us）。
- 已知局限：per-leg 带宽为锚点反推估计（§2 注）；iter-2 对照基线为两遍结构，其 L2 命中
  模式与 rev3 略有差异，1.35ms 串行化归因含 ±0.3ms 不确定度。
