# indexer_topk_reducesum — PTO fast path 性能调优最终报告（Stage 3 PTO round）

日期：2026-10-16 ｜ 目标：PTO fast 纯 kernel 从基线 6554.1us 收敛结构因子，向同结构下界 ~2700-3000us 推进（l0_main：B=1,S=2048,H=64,D=128,topk=64,bf16,offsets=[0,2048]，torch.npu Event，5 warmup + 20 iters，清态复测）

## 结果

| 指标 | 基线 | 终态 | 改善 |
|------|------|------|------|
| **PTO fast 纯 kernel** | 6554.1（任务书）/ 6587.4（本轮复测） | **1925.5 us**（清态 4 次独立进程复测：1925.5 / 1926.9 / 1930.4 / 1932.7，簇宽 0.4%） | **-70.7%** |
| vs 目标区间 2700-3000 | — | 低于下限 29% | 中止条件①达成并大幅超越 |
| PTO generic（非门控） | 7370.8 | 7132.5 | kernel 未动（波动） |
| PTO full interface | — | 2364.0 | — |
| AscendC fast（冻结带 1142-1165） | 1157.8 | 1146.8-1156.9（带内） | **零回退** |

精度：`--level all --backend both` 132 PRECISION_PASS / 0 FAIL / 0 WARN（exit 0）；
覆盖门禁 15 PASS / 0 MISS / 0 N/A；ruff check + format 全过；golden/容差/比对逻辑零改动。

## 迭代时间线

| 轮 | 改动 | before → after | 精度 | 结论 |
|----|------|----------------|------|------|
| 0 | 基线（phase-4b 终态） | 6587.4 | 全过 | — |
| 1a-1e | 探索（generic 方向融合链 6037.9；转置+融合 2274.6 但 TROWSUM 越界；算术 mask 等） | — | 多个失败 | 库缺陷定位（见下） |
| **1** | **转置方向 + half 宽行融合链 + transpose-reduce + tail 标量 mask** | 6587.4 → 3876.1（-41.2%） | PTO L0 10/10 + ascendc L0/带内 | **采纳** |
| **2** | **背压 ping-pong 统一（produce-then-wait）** | 3876.1 → 3123.8（-19.4%） | 双后端 L0 + ascendc 1155.4 带内 | **采纳** |
| 3x | j-loop 紧凑化（删 if 并入运行时上界） | 3120.9（不变） | 崩（0.06） | **否决**：auto CV sync pass 拥有 C/V 调度权，Python 层循环结构是 pass 输入契约 |
| **3** | **TADD 折叠树替代 TCOLSUM + CAP=4096 host 路由 generic** | 3123.8 → 1925.5（-38.3%） | 双后端 L0 + 全量 132 PASS + ascendc 1146.8 带内 | **采纳** |

## 瓶颈拆解（差分画像证据）

基线 6587.4 构成：per-token 向量链 4698.4（71%）+ topk 尾链 380.6 + 骨架 1508.4
（norelay+ 变体因 CV 配对契约破坏挂死，未细分——骨架含 GEMM/transit/relay/背压/PIPE_ALL）。

终态 1925.5 构成（同法差分）：tail ~600 + TTRANS ~580 + TADD 树 ~250 +
融合宽行链 ~11（宽行 tile 下几乎免费）+ relay/GEMM/transit/背压/骨架 ~480。

## 结构因子与库缺陷定论（本轮核心发现）

1. **"后端因子 2.4×"被打破**：generic-vs-generic 2.4× 是【旧结构】的同构对比；
   结构正确的 PTO fast（1925.5）已低于 AscendC fast × 2.4（2779）——结构因子
   完全收敛后，PTO 与 AscendC 的 fast 比值 = 1.66×（1925.5/1156.9），其中仍含
   TTRANS 绕行成本（~580us，PTO 库 TROWSUM 缺陷的代价）与 PTO 编译器差异。
2. **两个 PTO 库缺陷实锤**（最小单元测试可复现）：
   - tensor-tensor compare+select 在所有 tile 形状错误（4×32…2×128 全测）；
     仅 compare_scalar 路径正确——历史 phase-4 的"转置方向竞态"主因之一；
   - TROWSUM（dim=-1）对 validRow>1 的 dst 越界写 ~4KB（vcadd dstRptStride=Cols），
     隔离复现设备 UB-out-of-bounds，kernel 内表现为静默邻居污染。
   - 附带：-1e30 penalty 与 -inf 在 topk 强制补选路径有 mask 域内可观察差异
     （基线 -inf→-1e30 对照 0.9389）；[1,64]/[1,128] compare 的 repeat 边界位错误。
3. **当年 phase-4 归因修正**：转置方向本身无竞态——失败是上述库缺陷的组合
   （per-token 链恰好避开它们）。docstring note 15(b) 已改写。
4. **方法论教训**：idx 比对必须用 mask-domain 口径（golden 的 -inf 尾部输出原始
   列号，kernel 输出 -1；naive 全域比对在基线上也只 0.9846——本轮前 6 个变体中
   2 个正确实现被误判，浪费 ~6 轮）。

## 撞墙说明

不适用（远超目标）。剩余可优化项（未做，收益递减）：TTRANS 合并为 [32,256] 单次
（UB +16KB 需重排）；tail mask 的 forced-pad-only 化需先修 [1,N] compare 边界位。

## 平台约束记录

- PTO fast@CAP=4096 UB 不可行（auto topk tmp 24B/elem = 96KB + 链 + TTRANS 内部
  tmp 越界）——host 路由 generic（正确性等价，非 perf 门控）。
- auto CV sync pass 拥有 C/V 语句调度权：j-loop 的 if 结构不可在 kernel 层重构。
- planner 不感知 TTRANS 内部 tmp 的空间需求（布局满时越界崩溃）。

## 备份与工件

- history_version/indexer_topk_reducesum_pto_perf_iter{0,1,2,3}.py（iter0=改前基线）
- backend_verify/generated_fast_l0main_pto_perf_final.cc（终态生成代码）
- perf_tuning/perf_log.md（PTO 分节完整迭代记录）
- /tmp/opencode/t3pto/（本轮全部探针：单元测试 8 组、变体 30+、差分画像、dump）
- 终态 = 当前 indexer_topk_reducesum.py（iter-3 采纳版 + docstring 修订）
