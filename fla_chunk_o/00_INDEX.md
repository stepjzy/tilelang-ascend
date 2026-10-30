# fla_chunk_o 迁移与调优学习档案

## 一句话：这个算子教会我什么

CV 中继型（Vector 预处理→Cube GEMM）算子在 Ascend 910B3 上的性能墙不是带宽也不是
算力，而是**每任务同步协议地板**——本文档展示了用 msprof 消融图谱把 1.4ms 的墙逐层
归因到 relay 往返 / AUTO_SYNC 指令流 / flag 容量的完整方法，以及 do_bench 口径下
host 分发延迟的隐藏/暴露机制与地板核算。

## 阅读顺序（约 3 h）

1. ref_gpu.py（20min）→ 迁移源头：KDA chunked 前向输出 O（Triton 语义权威，
   仅 chunk_gla_fwd_kernel_o 在范围）；ref_gpu_tilelang_fwd.py 是上游 TileLang
   CUDA 版迁移起点
2. DESIGN.md（40min）→ 单 kernel 一维化/fixed-core/中继架构的完整设计与被否方案
3. debug_log.md（30min）→ 核心坑：out_idx 负索引、单 cast 消零超差（hi/lo 双 gemm
   的由来）、vrange 标定、R3 flag ID 冲突
4. perf_tuning/fla_chunk_o_final_summary.md（40min）→ 基线 3.285ms → 1.417ms
   （-57%）五轮调优全景 + 地板核算表（R4 章节）；fla_chunk_o_perf_log.md 是逐轮数据
5. history_version/（20min）→ 26 个方案转折点快照：fixed-core → 手动 flag →
   v 中继 → 3-chunk 批处理 → _FastLaunch 分发绕过的演进链

## 三大关键决策

1. **Fixed-core grid + 显式 L2 常驻中继 workspace**（替代每 launch ~768MiB 自动注入）
   ——把 8192 个逻辑任务的每 block 派发开销摊薄到 20 个常驻核，中继字节 L2 命中；
   这是 -50% 的最大来源。
2. **hi/lo 双 gemm 保精度**：qg 单次 bf16 cast 在消零元素上 ratio=0.988 < 0.99
   （bf16 混合容差门禁），hi/lo 拆分用 ~2^-18 相对误差换回 1.0000——精度政策直接
   决定了中继字节数（32KB/任务），是性能-精度权衡的教科书案例。
3. **不切 Expert、按 1.417ms 交付**：R5 实证 Developer 模式内假设空间穷尽（编译器
   旋钮无效/单 flag 粗化 +36%/名字级 WAW 伪屏障），唯一破壁路径是全手动同步重写
   （理论 -300~600us）——用户权衡 4 试 3 败的正确性风险后选择不授权。地板核算
   （msprof 设备侧 1391us > 1320us 目标）证明交付值已是当前工具链下的最优。

## 交付物

- 分支 feat/fla-chunk-o-pr（origin，基于 upstream/ascendc_pto@43b0b697）：
  examples_experiment/dsa_sparse_finetune/{example_fla_chunk_o.py,
  test_example_fla_chunk_o.py}
- PR: [#1953](https://github.com/tile-ai/tilelang-ascend/pull/1953)（open，
  评审轮 1 已修复：乘法 mask → VSEL Select 位打包掩码 + where 语义 golden +
  非有限结构门限 + 上三角 Inf 回归用例；详见 debug_log PR 修复节）
- 性能：官方均值 1.417ms（含 _FastLaunch；公共路径 1.492ms）vs 目标 1320us，
  超 7.5%（设备侧同步协议地板，工具链级，详见 final_summary R4/R5 地板核算表）

## 遗留疑问 / 未走通的路

- 全手动同步重写（TL_ASCEND_AUTO_SYNC=False）：理论 -300~600us 可越目标，
  ~700 行重写 + 4 试 3 败的正确性风险，未获授权
- relay 合并 4→1：被 L1 zN 分形布局阻断（region 逻辑偏移 ≠ 物理偏移），
  需上游提供 zN 对齐 region 操作数
- flag 计数器容量 <10 pending：限制更深的节流流水（lag 10 死锁实证）
- UB 硬上限 196352B（ascend_memory_planning.cc:35）：4-chunk 批处理编译期
  段错误实证封死
- host 分发 ~1ms（Arith Analyzer 每调用重建 0.6ms + Cython 逐参校验 0.4ms）：
  已用 _FastLaunch 绕过至 163us；根因属框架层，值得上游修复
