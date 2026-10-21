# indexer_topk_reducesum Stage 3 性能调优 — 最终报告

日期：2026-10-14 ｜ 目标：fast 路径纯 kernel 平均 < 1428 us（l0_main：B=1,S=2048,H=64,D=128,topk=64,bf16,offsets=[0,2048]，torch.npu 事件 5 warmup + 20 iters，排除 H2D/同步/host 分配/host 检测）

## 结果

| 指标 | 基线（任务书 / 同口径复测） | 最终（3 次复测） | 改善 |
|------|------|------|------|
| fast 纯 kernel | 2022.1 / 2002.9 us | **1142.7-1161.5 us**（1142.8 / 1142.7 / 1145.3 / 1161.5） | **-43.5%**（vs 2022.1） |
| msprof fast Task Duration | 1942.2 us | 1129.9 us | -41.8% |
| generic 纯 kernel（不考核） | 3103.7 us | 3104-3455 us（波动） | 持平（kernel 未动） |
| full interface | 4776.2 us | 1594-1617 us | -66.4%（host 开销 ~430us） |

精度：每轮迭代 L0 复验 + 终态 `--level all` 全 PASS（L0 24 + L1 42 + L2 13 正确拒绝 + Boundary 4）；覆盖门禁 24 PASS / 0 MISS / 0 N/A；s≤0 回退分支补测（sm_scale=-0.5/0.0）max_abs ~6e-8、mask 域索引 1.0000。

## 迭代摘要

| 轮 | 改动 | fast us | 精度 | 采纳 |
|----|------|---------|------|------|
| 0 | 基线（Stage 2 交付） | 2002.9 | 全过 | — |
| 1 | V 链融合（half-tile 大 op）+ straddle-only mask + scale 移位 [CAP] 行乘 + score_2d 融合 store | 1788.9 (-10.7%) | L0 全过 | yes |
| 2 | 转置 GEMM [BN, QG*H] 宽 burst 中转 + post-reduce 融合 [QG2,BN] mask + 整 tile 稠密融合 + QG 自适应回退 | 1153.7 (-35.5%) | --level all 全过 + coverage 24/24 | yes |
| 3 | host 低开销化：单同步 + numpy 构表 + 内容键表缓存 + _CachedLauncher（次要目标） | 1151-1164（kernel 不变） | --level all 全过 + s≤0 补测 | yes |

## 最终瓶颈画像（msprof，per-core）

- AIV（关键路径）：wall 1038us = vec 832 + mte2 72（85.5GB/s，宽 burst 后）+ scalar 258
- AIC：wall 1024us；自身工作 cube 153 + mte1 246 + mte2 126 + fix 122（四 pipe 重叠），ack 等待 354（仍有富余但不临界）
- 基线 → 终态的分项变化：AIV vec 1015→832（融合+删 op）；AIV mte2 529→72（64B→512B burst）；AIC fix 226→122（64B→1KB burst）
- 剩余优化空间（未做）：barrier 税（每 IfThenElse 2×PIPE_ALL + 同 pipe RAW/WAW PipeBarrier，估算 ~30-40% vec 相邻时间）、AIC 2-ahead + relay ping-pong 软件流水（藏 mte2/ack）、topk 段审计

## 关键根因与教训（详见 perf_log.md）

1. 基线主瓶颈是 AIV 的指令粒度与 barrier 数（每 token 每 block 13 条向量指令 + ~12 条 PipeBarrier_V），不是预算假设的中转带宽单项。
2. 中转 64B burst 使 mte2 有效带宽仅 12.6GB/s（转置后 512B/1KB burst → 85.5GB/s / fix 55→~110GB/s）。
3. 转置布局下跨行区域的 `T.tile.select` 会被 lowering 扁平化（选中错误内存区）——post-reduce 全 buffer mask 是正确形式；跨行 2D copy_ub_to_ub 退化为逐行小拷贝循环——融合链必须保持稠密 flat 形态。
4. TVM script round-trip：prim_func 体内定义的形状常量会变符号变量，形状常量必须放 builder 层。
5. prim_func 体内 `half_rows` 类 Python 常量、共享机器上的外部 bench 进程（测量污染 ~3.5×）、torch.zeros 冷启动 8.7ms 伪影——均已定位并规避。

## 备份清单

- history_version/indexer_topk_reducesum_perf_iter1.py（= Stage 2 交付版，iter-1 改前）
- history_version/indexer_topk_reducesum_perf_iter2.py（= iter-1 采纳版，iter-2 改前）
- history_version/indexer_topk_reducesum_perf_iter3.py（= iter-2 采纳版，iter-3 改前）
- 终态 = 当前 indexer_topk_reducesum.py（iter-3 采纳版）
