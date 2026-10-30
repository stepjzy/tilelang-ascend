# TileLang 算子迁移学习档案（dev-journal）

每个算子一目录，**从 00_INDEX.md 进入**（阅读顺序 / 关键决策 / 遗留疑问）。
本分支为 orphan 纯文档分支，不含源码历史，永不提 PR。

| 算子 | 状态 | PR | 一句话收获 |
|---|---|---|---|
| indexer_bwd | ✅ 已合并 upstream | [#1887](https://github.com/tile-ai/tilelang-ascend/pull/1887) | GM bf16 atomic 累加精度陷阱与受控仿真定位法；1229→68us（-94.4%） |
| sparse_mla_topk_reducesum | 🔄 PR open | [#1884](https://github.com/tile-ai/tilelang-ascend/pull/1884) | 三条设计路线证伪史；panel 特化→通用基线的取舍 |
| indexer_topk_reducesum | ⏳ 分支已 push，待提 PR | — | fast+generic 双路径；PTO/AscendC 双后端独立调优 |
| sparse_mla_fwd | ⏳ 分支已 push，待提 PR | — | 支持域 gate + portable 分派；实测证据审查驱动重定标；投影必须 A/B 实证（dtype 包笔误撤回）；138.6→46.7ms（-66.3%，目标 24.3ms 经实测判定平台不可达） |
| sparse_mla_bwd | ⏳ 分支已 push，待提 PR | — | 精度问题定位到设计层的判别法；rev3 回退重来的门禁纪律 |
| fla_wy_fast | 📦 已归档交付 | — | fp32 gemm 损坏下的 fp16 hi/lo 拆分模拟；实测带宽校准推翻标称假设；23.15→11.72ms（-49.4%） |
| fla_chunk_intra_token_parallel | ⏳ 分支已 push，待提 PR | — | msprof 流水图驱动瓶颈重归因（空闲 Cube 卸载 -49%）；四条直觉瓶颈实测证伪史；分离核 relay 地板论证；19150→1739us（11.0×，目标 370us 经论证物理不可达） |
| fla_chunk_o | 🔄 PR open | [#1953](https://github.com/tile-ai/tilelang-ascend/pull/1953) | CV 中继型算子的同步协议地板画像法（msprof 消融 + host 分发地板核算）；Developer 模式边界实证；3.285→1.417ms（-57%，目标 1320us 为工具链级地板所封） |

## 维护规则

- 归档时机：算子交付（-pr 分支 push）后、主仓清理前（见 tilelang-op-migration skill Phase 4）。
- PR 的提交/合并与状态跟踪由用户自行处理，journal 归档内容定格在归档时刻，不做后续状态更新。
- 主仓沙箱 `ref_gpu/` 为共享语义参考，本档案内各算子 `ref_gpu.py` 为其冻结副本。
