# indexer_topk_reducesum 迁移与调优学习档案

## 一句话：这个算子教会我什么
双路径架构（fast 特化 + generic 通用兜底）的设计取舍，以及同一算子在
PTO / AscendC 双后端下各自独立的调优方法论。

## 阅读顺序（约 2.5h）
1. `ref_gpu.py`（30min）→ 迁移源头：golden = `ref_index_score`
   （ref_gpu/indexer_topk_reducesum.py:208-229）
2. `DESIGN_indexer_topk_reducesum.md`（1h）→ §1.4 双路径架构（fast/generic）、
   §1.5 数据流图（fast 路径，每任务 QG=8 token、每 K 块 BN=32）、
   §2 编程模式选型、§3.2 API 映射（全部附出处）——模板级的设计文档范例
3. `perf_tuning/indexer_topk_reducesum_final_summary.md`（30min）→
   fast 纯 kernel 2022.1 → 1142.7us（**-43.5%**），msprof 交叉验证
   （1942.2 → 1129.9us），full interface 4776 → 1594us（host 开销剖析）
4. `perf_tuning/indexer_topk_reducesum_pto_*.md`（30min）→ PTO 后端独立的
   基线/优化日志/最终报告——双后端调优口径差异
5. `history_version/`（30min）→ impl_s2_attempt1 v0→v3 四版演进、
   phase4_v1/v2、pto_perf_iter0-3

## 三大关键决策
1. **fast + generic 双路径**：fast 吃特化 shape，generic 兜全shape 域
2. **PTO 后端单独走一轮完整调优**（而非复用 AscendC 结论）
3. generic 路径**不纳入性能考核**（3103.7us 持平，kernel 未动）——
   明确"优化什么、不优化什么"

## 交付物
- 分支 feat/indexer-topk-reducesum-pr（origin，已 push）
- **PR 尚未创建** → PR_DESCRIPTION.md 为占位待补

## 遗留疑问 / 未走通的路
- generic 路径性能未优化（是否有必要？）
- PTO 与 AscendC 的优化手段差异总结（两份 final_summary 对照）
