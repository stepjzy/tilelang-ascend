# P3-v2 实验记录：C1 元素 gather 选值 + C4 组内 pos 表整合（Stage B）— 2026-10-12

> **route**: P3-v2 = P3 存档（`history_version/sparse_mla_impl_p3_falsified.py`）+ Stage A spike 胜者
> C1（`T.reinterpretcast` 位视图 + `T.tile.gather` 单指令选值）+ C4（组内 pos 表一次构造、组内复用）
> **baseline**: git `22bc68a8`（P2 交付态，**保护不动**——交付版本不受本实验影响，未 commit）
> **改动范围**: 仅选值机制（P3 等值比较矩阵 + k 归一链 → pos-gather + 匹配守卫）与组切换处 pos 构造；
> prep / panel staging / 8-flag 协议 / 转置 gemm [P,H] / host 双发射分发 / 大 P_seq 回退全部 P3 原样
> **verdict**: **[PRECISION_PASS]**（正确性全绿）+ 官方 BI128 = **729.0µs**（Δ 算术预测 729 精确命中；
> 高于 Stage-A 模型带 660-690 上缘 +5.7%，在 ±10% 规则内，不触发回退）

## 1. 官方 BI128 实测（graph-replay 三连发捕获，d9 协议逐字：batch=25/iters=10/warmup=5，官方 arange seed=42 逐字，同会话 4 轮配对）

| 路径 | 本会话实测（中位/轮） | 参照 |
|---|---|---|
| P2 基线（direct-kernel 口径） | **378.1µs**（436.4/378.1/375.4/377.9） | 历史带 350-381 ✓ 带内 |
| P3 存档（三连发） | **899.0µs**（899.2/898.1/899.0/898.2） | P3 会话 894（893.5-896.4）**复现 +0.6%** ✓ |
| **P3-v2（三连发）** | **729.0µs**（726.5/728.9/729.0/736.5） | **vs P3 配对 −170.0µs（−18.9%）**；vs P2 基线 1.93× |

- **vs 660-690 预期带**: +5.7%（690 上缘之上），**< 10% 规则线 → 不回退**。
- **Δ 算术精确命中**: Stage A 预测选值链收益 = (1.87 − 0.23)µs/task × 102.4 task/核 = **168µs**；
  实测 P3 → P3-v2 = 899.0 − 729.0 = **170.0µs**（误差 1.2%）；panel 腿分解口径 163.3µs = **1.60µs/task**
  （vs 预测 1.64，误差 2.5%）——**C1+C4 的选值收益与 Stage A 微基准预测逐点吻合**。
- 带差归因（如实）: Stage-A 模型带（660-690）假设 C 侧地板 ≈ 我的 MB-4 bench 结构（~5.9µs/task）绑定；
  真实 P3 8-flag 跨核协议 + 每 task GM 标量读（TokenIndices×2 + Offsets）使真实地板更重
  （见 §3 四腿分解），V 侧仍为临界路径。**剩余到 220 的差距 = 每 task 装载/flag 地板**
  （Stage A 结论在真机完整 kernel 上再次成立）——任务合批粒度阶段的靶子，非选值机制问题。

## 2. 四腿分解（P3-v2，同会话 graph-replay，4 轮中位）

| 腿 | 实测 | 说明 |
|---|---|---|
| prep（资格判定） | **27.9µs** | 与 P3 会话 29.3 同级（band 内） |
| panel 主体（C+V 全量，直调 panel_guarded） | **685.1µs** | P3 同口径 ≈ 848.4µs（899.0−27.9−22.7）→ **−163.3µs = C1+C4 净收益** |
| generic 早退（launch+守卫开销） | **22.7µs** | 与 P3 会话 25.3 同级 |
| **全量三连发** | **729.0µs** | 三腿和 735.7 ≈ 729（launch glue ≈ 0/轻微重叠） |
| **V 选值链隔离**（mb_v2chain，P3-v2 逐字链 22 V 指令） | **0.497µs/task** | vs 预期 0.19-0.23：差值 = 匹配守卫 6 条指令（+0.27µs）；vs P3 链 1.87µs → **3.7×** |

per-task 换算：panel 主体 685.1/102.4 = **6.69µs/task**（P3：8.29µs/task）。

## 3. 正确性（全绿）

- **L0-D 家族 16/16 PASS**（D 用例从 P3 会话 `/tmp/opencode/p3/quick_d.py`+`smoke_d1.py` 逐字恢复，
  落入当前 test 文件为 `L0-D` 门控用例，修掉原脚本 D7b 参数序 bug）：
  D1 官方 golden（viol=0，panel 路径，err_ratio **1.648e-03** 与 P3 逐字同级）/
  D2 随机非共享（viol=40 → generic，1.877e-03）/
  D3 重复索引共享（raw ref matched=1.000000，max_abs 1.431e-06；P3: 1.9e-06 同级）/
  D4 乱序集合共享（viol=40 → generic）/
  D5 -1/越界共享（panel，1.789e-07）+ D5b 全 -1 行（行级精确零）/
  D6 G=2 大 P_seq（viol=40 → generic）/
  D8 offsets 单段与 [0,0,1023,2048] 空首段（panel，masked ref）/
  D8b RH=2 H=128（panel，err_ratio **4.872e-04** vs P3 4.9e-04 同级）/
  **D7a 新选值路径 3-run 位级稳定** / D7a run0 ref 全过 / **D7b 混合路径交替 ×3 全过**（竞态压测 ✓）
- **全量 `--level all`**: 86 × [PRECISION_PASS] + 8 × [BOUNDARY_PASS] + **0 FAIL / 0 WARN**，
  "Test Passed!"（P2 套件 70 PASS + L0-D 16 PASS）
- **覆盖门禁**: `coverage_check.py --proto proto.yaml` → **23 PASS / 0 MISS / 0 N/A → PASS**
- **L0-G6 锚**（generic 直调，路径未动）: BI128 = 372.5µs（+2.2% vs 364.7 锚，带内）；BI32 = 478.3µs（+0.6%，带内）

## 4. 流量核算（用户 req 四.5 口径延续）

C1+C4 **零 GM 流量增量**（新增 buffer 全部 `T.alloc_ub`：row0_half_f/pos_f/pos_i/pos_off/rows_at_pos/
eqv [HALF]×6 ≈ 1.5KB；GM 张量集与 P3 逐项相同：Q/KV/Indices/Lse/Offsets/TokenIndices/Violations/
ReduceSum/ws_panel/ws_acc）。官方 BI128 各腿（P3 账本逐项不变）：V 行读 ~5.9 + V bulk 写 ~5.9 +
C ws 读回 ~5.9 + Q 37.7 + ws_acc 写读 ~49 + 输出等 ~13 ≈ **~118MB 维持** ✓（选值替换纯 UB 内部）。

## 5. 实现记录（差异清单 vs P3 存档）

| 位置 | P3 | P3-v2 |
|---|---|---|
| post (c) 选值 | 等值比较矩阵（2×broadcast+sub+abs+clamp+sub+mul 于 [topk,HALF]）+ 2×reduce_sum + k clamp + div（~15 V 指令） | **1×vgather**（rs=rs_all[pos]）+ 守卫 gather（rows_f[pos]）+ 5 条 [HALF] 算术 + 1 mul（8 V 指令） |
| 组切换 ×2（prologue + 主循环前瞻） | — | **C4 pos 构造**：row0 半区 copy + 2×broadcast + sub + clamp + reduce_sum（[topk,HALF] 计数矩阵，复用原 eq 槽位）+ f32→i32 cast + ×4 + 位视图（摊销 ~0.02-0.04µs/task，Stage A 实测 6.71µs/组） |
| UB | k_ub | 移除 k_ub；+6 个 [HALF] 工作位（+1.5KB，断言 13→18×HALF×4） |

**语义等价论证**（正确性的核心，测试 D1-D8b 逐项验证）：
- 匹配 ⇒ rs_all[pos]（单槽）≡ P3 num/k（k 个同值副本的平均）——重复值多副本 rs_all 同值，逐位一致；
- 不匹配（idx ∉ 表，仅非共享域理论分支；day-1 门下恒匹配）⇒ 守卫置 0 ≡ P3 k=0→0/1=0，逐位一致；
- pos ∈ [0, topk−1] 恒界内（row0[j] 必 ∈ panel ⇒ 严格大于计数 ≤ topk−1）——gather 无越界。

**整合期踩坑（记录给下一阶段）**：
1. `T.reinterpretcast` 别名声明的作用域规则——组切换 if 块内声明对 post 的 gather 不可见
   （解析到未初始化 dummy 槽 → 垃圾偏移 → aicore 越界）；与 alloc 同层则 C++ 重定义。
   **解法：零成本别名声明放主循环体首语句**（循环体块作用域合法遮蔽 + 消费者同体可见 + 每迭代零成本）。
2. P3 存档 quick_d.py 的 D7b 参数序 bug（ref 误占 dim_v 位）——移植时修复。

## 6. 已知继承限制（P3 原样保留，如实标注；三类限制延续 Stage A 口径）

| # | 限制 | 归类 | 状态 |
|---|---|---|---|
| 1 | rows_f/pos 单缓冲在前瞻组切换（LA=32 窗口）覆写——段间 row0 不同的多段输入在边界 LA 窗口内选值用邻组表（P3 等值矩阵同款窗口；官方口径与全部测试族为全局统一 idx，窗口恒空） | 算法设计限制（P3 继承） | 保留（"其余原样"纪律）；下一阶段随粒度重构一并处理（pos 表按 pslot 双缓冲） |
| 2 | panel kernel 正确域实际为 {组内 idx 值集 ⊆ row0 值集}（P3 实现口径；day-1 门为其子集）——设计文档 {P_seq ≤ topk} 口径需并集 panel（tier-2） | 算法设计限制（P3 继承） | 守卫使 P3-v2 与 P3 逐位同域（不劣化） |
| 3 | 每 task 装载/flag 流水地板（V 6.69 / C 侧同量级 µs/task 串行口径）——220 的真绑定项 | 算法设计限制（Stage A 裁决延续） | 本阶段不处理（粒度阶段靶子，四腿分解即其基座数据） |

## 7. 状态与回退

- **P2 交付基线不动**：git `22bc68a8` 无 commit；交付版本 = P2。
- P3-v2 留工作树作为**粒度阶段基座**（未 commit）；回退路径：
  `git checkout 22bc68a8 -- examples/dsa_sparse_finetune/{sparse_mla_topk_reducesum.py,test_sparse_mla_topk_reducesum.py}`
  或 `cp history_version/*_pre_p3v2.py` 恢复。
- 备份：`history_version/sparse_mla_impl_pre_p3v2.py`（P2 kernel 825 行）+
  `history_version/test_sparse_pre_p3v2.py`（P2 test 1295 行）。
- **框架零改动**：`git diff src/ tilelang/` = 空 ✓（Stage A 的 17 行 codegen patch 未使用——纯算子侧 T.reinterpretcast 路径已足够）。
- 复现：`/tmp/opencode/mb/p3v2_perf.py`（三方+四腿）、`/tmp/opencode/mb/mb_v2chain.py`（选值链隔离）、
  `python examples/dsa_sparse_finetune/test_sparse_mla_topk_reducesum.py --level all`（全量正确性）。
