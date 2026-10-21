# sparse_mla_topk_reducesum 迁移与调优学习档案

## 一句话：这个算子教会我什么
三条设计路线的完整证伪史（panel/auto 特化 → 通用基线 → 数据流重构），
以及"性能证伪即回退"的调优纪律——最有复盘价值的一份档案。

## 阅读顺序（约 3h）
1. `proto.yaml`（20min）→ 接口规格 rev3：panel/auto 分发已移除，
   block_I ∈ [32,128] 偶数（L0B 乒乓槽 32KiB@kL0Size=128/bf16 硬上限，
   见 common.h L1263-1280）
2. **三份 DESIGN 按时间线对比读**（1h）→
   `DESIGN_panel_mode.md`（初版三模式分发）→ `DESIGN_generic_p1.md`
   （移除特化建立通用基线，对应 git 0f11b3cc/4e153c7f）→
   `DESIGN_dataflow.md`（P3 数据流重构）
3. `debug_log.md`（40min）→ P1/P2/P3 各阶段的失败与回退记录
4. `perf_tuning/`（40min）→ P2 交付（SIDE_PAIR 保护对移除 + g_l 缺陷修复）、
   P3 数据流重构证伪、Gather+GEMM 组合证伪（Stage A 微基准 + Stage B C1+C4）
5. `history_version/`（20min）→ **交付基线 = `sparse_mla_impl_pre_p3.py`**
   （P2 版本，825 行）；终态 `sparse_mla_topk_reducesum.py` 是 Gather+GEMM
   证伪实验的存档（1622 行），**不要当作交付版本学习**
6. `PR_DESCRIPTION.md（#1884 线上正文）` → 线上 PR 正文：Expert 模式
   （T.alloc_L1/L0C/ub 层级、T.Scope("C") AIC + T.Scope("V") 2×AIV 双流、
   set/wait_cross_flag mode-2 quorum、persistent 1-D grid=20）

## 三大关键决策
1. **panel/auto/shared 三模式特化 → 移除，走通用基线**（P1 结论）
2. **P2 交付**：SIDE_PAIR 保护对移除 + g_l 缺陷修复（最终交付内容）
3. **P3 与 Gather+GEMM 双双证伪并回退**——"机制成立/性能证伪"如何用
   微基准证明并干净回退

## 交付物
- 分支 feat/sparse-mla-topk-reducesum-pr（origin）
- **PR #1884 open**（线上正文已存 PR_DESCRIPTION.md（#1884 线上正文）；
  本地稿 PR_DESCRIPTION_draft.md / official_template.md 留档对照）
- 注意：-pr 分支最终版与沙箱版有差异（Ruff 格式化 + 提交前清理），
  对照学习时以 -pr 分支为交付真源

## 遗留疑问 / 未走通的路
- generic 路径未优化（3103.7us 波动，kernel 未动）
- 框架实验改动（common.h / ascend.py 的 gemm_v0 opt-in）未进 PR，
  已随开发分支删除——是否值得复活？
