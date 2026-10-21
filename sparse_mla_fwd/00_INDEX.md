# sparse_mla_fwd 迁移与调优学习档案

## 一句话：这个算子教会我什么
"支持域 gate + 域外 portable 分派"的工程化设计，以及用**实测证据审查**
（evidence audit）代替估算来驱动 DESIGN_ERROR 重定标的调优方法论。

## 阅读顺序（约 2.5h）
1. `ref_gpu.py`（30min）→ 迁移源头：sparse_mla 前向（a3 两遍法）
2. `DESIGN_sparse_mla_fwd.md`（1h）→ §2 模式选型、§3 API 映射、
   §4 数据规格与内存规划、§5 Tiling；**支持域 NI∈[1,6]** 的 gate 定义
   （H=128 ∧ G=1 ∧ (dim,tail)=(512,64) ∧ topk%256==0）与 §11.4 域外
   portable 分派树——特化与通用性的边界设计范例
3. `perf_tuning/sparse_mla_fwd_evidence_audit.md`（40min）→ 核心方法论：
   fast v2 55258us（后修正为干净基线 46667us，机器态双峰 Δ18% 见 §3.0）
   的逐项访存量推导（三栏分类）+ 实测剖析，为 DESIGN_ERROR 重定标供证
4. `perf_tuning/sparse_mla_fwd_perf_log.md` + `baseline_iter1-6.json`（30min）
   → 基线演进与数据口径
5. `history_version/`（30min）→ rev1 → rev2 的 **chunk A→D 渐进开发**痕迹
   （chunkA/B/C/D/D_final 五个中间件）→ best_rev2b 收敛版——大重构如何
   分片推进的实证

## 三大关键决策
1. **fast v2 特化路径 + portable 兜底**，gate 条件显式写进 DESIGN
2. **两遍法布局**：BI=256 / NI=4 / M=64×2 / ws_kv DEPTH=NI+1=5 环
3. **实测证据优先**：性能结论必须同会话 A/B + 交错轮测中位，
   机器态波动（Δ18%）显式记录而非忽略

## 交付物
- 分支 feat/sparse-mla-fwd-pr（origin，已 push）
- **PR 尚未创建** → PR_DESCRIPTION.md 为占位待补

## 遗留疑问 / 未走通的路
- 46667us 干净基线之后的优化空间（evidence_audit 的推导结论）
- 机器态双峰问题（同型机器性能两态）的规避策略
