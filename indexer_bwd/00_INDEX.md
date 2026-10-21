# indexer_bwd 迁移与调优学习档案

## 一句话：这个算子教会我什么
GM bf16 atomic 累加的精度陷阱及"受控仿真定位根因"的完整方法论；以及一个
1229.9us → 67.9us（-94.4%）的极致调优案例。

## 阅读顺序（约 2.5h）
1. `ref_gpu.py`（30min）→ 迁移源头：DSA lightning indexer 反传，每 token
   3 个小 GEMM（logits / dIndexQ / dIndexK）+ 向量后处理（relu/差值/mask/
   broadcast/reduce）+ 逐行 gather / DMA atomic scatter
2. `DESIGN.md` §2-3（40min）→ Developer 模式选型理由 + GPU→Ascend API 映射
3. `debug_log.md`（40min）→ 核心坑：dk matched_ratio 0.94 < 0.99 门限，
   失配元素恰为 atomic 热行；受控仿真：同顺序 bf16 累加复现失败 0.9610、
   fp32 累加通过 0.9972 ⇒ 根因唯一指向 GM bf16 atomic 累加路径
4. `perf_tuning/final_summary.md`（30min）→ -94.4% 每一步怎么来的
   （fast+cast 纯 kernel 1229.9 → 67.9us，目标 <80us 达成；msprof 交叉验证）
5. `history_version/`（30min）→ design_rev1→rev2 演进；perf_iter0→final
   六轮迭代；impl_s2_rev1_attempt1 的实现尝试
6. `README.md` → 实现结构说明（kernel + interface + 分层测试组织）

## 三大关键决策
1. **fp32 部分和替代 bf16 GM atomic**（精度根因的直接对策，debug_log Attempt 1）
2. **AscendC + PTO 双后端支持**（backend_verify 的对比结论已沉淀在 DESIGN/README）
3. **fast layout=0 + cast 路径**作为性能主路径（final_summary 口径基础）

## 交付物
- 分支 feat/indexer-bwd-pr（origin）
- **PR #1887 已合并 upstream**（2026-10-20，正文见 PR_DESCRIPTION.md）

## 遗留疑问 / 未走通的路
- 见 debug_log 各 Attempt 的 boundary_warnings 与 coverage 记录
