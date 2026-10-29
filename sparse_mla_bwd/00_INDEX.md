# sparse_mla_bwd 迁移与调优学习档案

## 一句话：这个算子教会我什么
"精度校准问题定位到设计层而非实现层"的判别方法，以及 Developer pair-loop
主路径 + Expert fallback 的混合架构如何在大改动（rev3 重构）后干净落地。

## 阅读顺序（约 2.5h）
1. `ref_gpu.py`（30min）→ 迁移源头：sparse_mla 反传语义
2. `DESIGN_sparse_mla_bwd.md`（1h）→ **先读 §0 rev3 回退证据与重设计总纲**
   （门禁必需章节——为什么 rev2 被推翻）；§4 总体架构（Developer pair-loop
   主路径 + Expert fallback + C/V 深度流水 + transit 融合）；§6 dKV 散射
   策略专题（随机行散射寻址能力查证与选型）——专题查证式设计的范例
3. `debug_log_sparse_mla_bwd.md`（40min）→ 核心坑：L1 值域轴 4/15 失败，
    根因**定位到 DESIGN §12 校准层而非实现层**（Δ-from-bf16-o 接口颗粒 +
    高集中度抵消区逐元素门限不可满足）——"设计错还是实现错"的判别范式；
    末条 Review fix（2026-10-29）：identity 缓存键的原地更新坑 + **评测热路径
    host 同步拆除 do_bench 流水掩盖（+3.6%）** 的实测权衡
4. `perf_tuning/`（40min）→ optimization_log / perf_log / iter1_diagnostics /
   iter3_profile / baseline_iter0-4（含 3b 系列）
5. `history_version/`（30min）→ design rev0→rev2→rev3 三代设计演进；
   impl_s2 attempt1/attempt2/rev3_attempt1；perf_iter0→4
   （含 3b_v0-v2 与 4_v1_stagger 交错变体）

## 三大关键决策
1. **rev3 回退重来**：rev2 整体推翻，回退证据前置为 DESIGN §0
2. **Developer pair-loop 主路径 + Expert fallback** 双层架构
3. **dKV 散射选型靠能力查证**（查证随机行散射寻址，而非假设可行）

## 交付物
- 分支 feat/sparse-mla-bwd-pr（origin，已 push）
- [PR #1928](https://github.com/tile-ai/tilelang-ascend/pull/1928)（open，2026-10-29 review
  意见已修复：`_tensor_cache` 身份键 → identity + mutation version，附内容键 +3.6% 计时
  代价的 A/B 实测权衡，见 debug_log 末条 Review fix）

## 遗留疑问 / 未走通的路
- Δ-from-bf16-o 接口颗粒问题的最终口径（校准层修订后的门禁定义）
- stagger 变体（perf_iter4_v1_stagger）的取舍原因
- `tensor.data` 裸写不触发 version counter——`_tensor_cache` 的文档化残留洞（torch
  autograd 同样不追踪，若 review 坚持全覆盖需换内容键并接受 +3.6% 计时影响）
