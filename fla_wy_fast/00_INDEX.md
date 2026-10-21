# fla_wy_fast 迁移与调优学习档案

## 一句话：这个算子教会我什么

在"性能目标物理不可达"的前提下如何干活：用**实测带宽校准**推翻设计期未校准的
标称假设（392GB/s → 实测 1263GB/s），把优化空间压到工具链边界（23.151→11.719ms，
-49.4%），最后按"目标/估计/实测"三分类如实披露 8.55× 缺口——而不是编数据或做
破坏正确性的达标动作。

## 阅读顺序（约 4 h）

1. `ref_gpu.py`（30min）→ 迁移源头：KDA chunked WY 表示（Triton fwd+bwd 双 kernel，
   上游 FLA 语义权威；配套 `ref_gpu_tilelang_fwd/bwd.py` 为上游 TileLang CUDA 版起点）
2. `DESIGN.md` §3/§4/§9（50min）→ GEMM 密集算子 Developer 模式设计：API 映射（gemm_v0/
   exp(x·ln2)/Msk 常量）、910B 片上内存预算表、带宽预算方法；§9.5 是 [DESIGN_ERROR]
   级目标冲突（1370us vs 硬件带宽）的完整论证
3. `debug_log.md`（50min）→ 核心坑：fp32 gemm_v0 工具链损坏、beta 子 32B DMA 不可靠、
   C/V 融合结构五连坑、**golden 自身 einsum 轴序 bug**（kernel 是对的、参考实现错了，
   dA ratio 恰为结构零占比 0.5078 佐证）
4. `perf_tuning/fla_wy_fast_final_summary.md`（40min）→ 基线→最终怎么来的：
   broadcast 化 -9.21ms + Msk 缓存 -0.97ms + sync 口径对齐 -1.37ms；实测 D2D 1263GB/s
   如何修正 DESIGN 的 392GB/s 假设
5. `perf_tuning/fla_wy_fast_tuning.md` + `fla_wy_fast_impl_iter1_before.py`（30min）→
   被否决的 4 项（A cast hoist 中性 / scatter 移位负收益 / k2 CSE UB 溢出 / kg div 持平）
   差在哪；`optimization_log.md` 为逐项 RESULT 原始记录

（本算子无独立 history_version/：唯一方案转折点是 Stage 3 调优前后，快照即
`perf_tuning/fla_wy_fast_impl_iter1_before.py`。）

## 三大关键决策

1. **性能目标冲突的处理路径**：论证 1370us 物理不可达（先按标称 392GB/s 得 4.6×、
   实测校准后修正为需 1.44× 实测上限）后，与用户确认按"尽力优化 + 如实披露"推进，
   不做任何破坏精度/正确性的达标动作。
2. **fp32 gemm_v0 损坏 → fp16 hi/lo 3-gemm 模拟 fp32**：隔离用例证实工具链损坏后，
   拆分方案先经 CPU mimic 按精度门验证（ratio=1.0000）再落地，保住 fp32 门禁
   （atol 2⁻¹⁶/rtol 2⁻¹⁰）而非放宽标准；已知限制（|x|>65504）进边界用例文档化。
3. **行循环 broadcast 化**：`T.tile.broadcast`(axis=0/1) + 全量 tile 向量指令替换
   6 个 kernel 的 64 行串行循环——单 kernel -26%~-71%、合计 -9.21ms，是本轮主收益；
   根因是 Vector 核瘦指令下发开销（行循环 lower 成 64 条单行 Muls）。

## 交付物

- 分支 `feat/fla-wy-fast-pr`（origin，commit 4666a5ef，2 文件：
  `examples_experiment/dsa_sparse_finetune/example_fla_wy_fast.py` +
  `test_example_fla_wy_fast.py`）
- PR：待提（PR_DESCRIPTION.md 已按归档材料起草）

## 遗留疑问 / 未走通的路

- 1370us 目标需领导重定：当前 8.55× 缺口，实测带宽下限 ~1.97ms（目标仍差 1.44×）。
- 工具链 10 项缺陷清单（final_summary §8）待上游修复，其中 5 项直接阻塞本算子
  进一步收益（fp32 gemm、单 Pipelined 体、多 gemm 组 L0 staging、sync-point 计数、
  cast-on-copy 模板错配）。
- bwd g=40 极端门控的 fp16 拆分范围限制：根治依赖工具链 fp32 gemm 修复。
