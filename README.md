# TileLang 算子迁移学习档案（dev-journal）

每个算子一目录，**从 00_INDEX.md 进入**（阅读顺序 / 关键决策 / 遗留疑问）。
本分支为 orphan 纯文档分支，不含源码历史，永不提 PR。

| 算子 | 状态 | PR | 一句话收获 |
|---|---|---|---|
| indexer_bwd | ✅ 已合并 upstream | [#1887](https://github.com/tile-ai/tilelang-ascend/pull/1887) | GM bf16 atomic 累加精度陷阱与受控仿真定位法；1229→68us（-94.4%） |
| sparse_mla_topk_reducesum | 🔄 PR open | [#1884](https://github.com/tile-ai/tilelang-ascend/pull/1884) | 三条设计路线证伪史；panel 特化→通用基线的取舍 |
| indexer_topk_reducesum | ⏳ 分支已 push，待提 PR | — | fast+generic 双路径；PTO/AscendC 双后端独立调优 |
| sparse_mla_fwd | ⏳ 分支已 push，待提 PR | — | 支持域 gate + portable 分派；实测证据审查驱动重定标 |
| sparse_mla_bwd | ⏳ 分支已 push，待提 PR | — | 精度问题定位到设计层的判别法；rev3 回退重来的门禁纪律 |

## 维护规则

- 归档时机：算子交付（-pr 分支 push）后、主仓清理前（见 tilelang-op-migration skill Phase 4）。
- PR 状态变化时更新本表 + 对应 00_INDEX.md 的"交付物"章节。
- 主仓沙箱 `ref_gpu/` 为共享语义参考，本档案内各算子 `ref_gpu.py` 为其冻结副本。
