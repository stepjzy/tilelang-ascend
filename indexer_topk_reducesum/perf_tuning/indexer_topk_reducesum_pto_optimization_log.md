# indexer_topk_reducesum PTO fast-path perf optimization log (Stage 3 PTO round, 2026-10-16)

## Part A 优化点清单（基于差分画像 + 生成代码静态分析）

画像证据（/tmp/opencode/t3pto/ablation_profile2.py，20 iters 差分）：
- baseline 6587.4us = topk 尾链 380.6 + **per-token 向量链 4698.4 (71%)** + 骨架 1508.4
- 生成代码：PTO u-loop 每 token 13 op + ~12 PipeBarrier_V + 1 组冗余
  set/wait(MTE2,V,ID2)（relay 依赖在 u-loop 内被重复同步 ×4/block）
- ascendc 同结构位（融合 [32,256] 大 tile 链）vec 仅 ~832us

[#1] V 侧融合大 tile 向量链（参考：optimization-guide.md 向量链融合 /
     performance-antipatterns.md 小粒度向量 op 反模式；ascendc perf iter-1
     同构先例）：[适用] — 差分画像 71% 主导；保持 generic 方向 transit
     （PTO 稳健形）不变，仅把 per-token [H,BN] 小链换成 [QG2H,BN] 行 flat
     融合链（cast/mul/round 一次 + 4 次 [H,BN] 行块 reduce + straddle-only
     post-reduce [QG2,BN] 张量 mask）。当年融合链失败绑定于转置方向 transit
     竞态，与 generic 方向正交。
[#2] post-reduce [QG2,BN] 张量 causal mask + straddle-only（ascendc perf
     iter-2 形态）：[适用] — 依赖 #1；PTO 隔离复现已 MATCH（phase 4 D.3），
     失败症状是转置竞态连带；packed-uint8 dst 契约已满足（note 15a）。
[#3] 背压 produce-then-wait ping-pong 恢复（优化指南 CV 流水/重叠章）：
     [适用性待验证] — PTO 序列化是当年保守化的一部分；竞态归因于 transit
     方向而非 flag 协议本身（phase 4b 机制定论）。iter-2 实验对象。
[#4] 转置方向 transit 宽 burst 重新归因（尾行过期复现 + 分块 flag/行组宽读
     候选解法）：[暂缓] — 差分显示 transit 读腿在 novec 残差 1508us 内，
     需先吃 #1 的 4698us；#1/#3 落地后再评估残差构成。
[#5] 装载精化 / API 形态选优：[暂缓] — 同 #4，残差内小项。

## [ORDER-PLAN]
1. [#1]+[#2]（同一结构重写，iter-1）— 前置依赖: [无] — 理由: 71% 主导项
2. [#3]（iter-2）— 前置依赖: [#1 ✅] — 理由: 消除 4698us 后骨架串行气泡
   占比上升，ping-pong 重叠收益才可测（避免归因混淆）
3. [#4]（iter-3+）— 前置依赖: [#1,#3] — 理由: 布局依赖（transit 方向决定
   relay 形态），残差画像先行

## 迭代记录

### Iteration 1 — V 侧融合链恢复（generic 方向不变）
[IMPL-#1] 关键约束：relay 读保持 [QG2H,BN] 行连续（PTO 稳健形）；融合 op
稠密 flat（列跨行区域会退化逐行循环，perf iter-2 教训）；reduce 用行块
[H,BN] 切片（行连续 2D region，pto_tile.hpp GetSliceInfo 支持）；buffer
声明互斥分支（对侧后端 DCE）；_fast_gate_ok 模型已含融合套件字节（原 PTO
小链 < 模型，改后 = 模型，无需改 gate）。

## 终态汇总（[RESULT] 汇总）

- [RESULT-#1] 转置方向 + 宽行融合链：6587.4 → 3876.1（-41.2%）| 精度 pass | 采纳
- [RESULT-#2] ping-pong 背压统一：3876.1 → 3123.8（-19.4%）| 精度 pass | 采纳
- [RESULT-#3] TADD 折叠树 + CAP=4096 路由：3123.8 → 1925.5（-38.3%）| 精度 pass | 采纳
- [RESULT-#4] j-loop 紧凑化：3120.9（不变）| 崩 | 否决（pass 调度权契约）
- [RESULT-#5] [P2] 域 tail mask：| 0.85 | 否决（compare 边界位缺陷）
- 累计：**-70.7%（6587.4 → 1925.5）**，低于任务目标区间下限 29%

## 反模式对照检查（performance-antipatterns.md）
- 小粒度向量 op（per-token 链）：已消除（宽行融合，差分 4698→11us 级）
- 窄行 tile（<256B 内层行宽）：已消除（[QG2H,BN] 128B → [BN,HALFC] 512B）
- 全序列化 CV：已消除（ping-pong）
- 保守化结构（generic 方向/逐 token/序列化）：全部解除（库缺陷绕行替代）
- 遗留保留项：TTRANS 绕行（TROWSUM 库缺陷的唯一安全替代，成本 ~580us）
