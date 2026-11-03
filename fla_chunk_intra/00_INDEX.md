# fla_chunk_intra 迁移与调优学习档案

## 一句话：这个算子教会我什么

CANN 跨核 flag「16 槽硬截断」的实锤定位法（40 个 flag ID 全碰撞 → 以 kernel 边界作 V→C 同步的 2-kernel 结构性重构）；逐列循环携带依赖的 masked-GEMM 重表述（掩码只依赖 (i,j)，顺序循环全部可进 Cube）；带宽 roofline 下限核算驱动的验收目标重定（540us 物理不可达论证 → "先实测再定目标" → 实测 11.44ms 定验收 ≈11.5ms）。

## 阅读顺序（约 3.5 h）

1. ref_gpu.py + ref_gpu_tilelang_gpu_fwd.py / _bwd.py（30min）→ 迁移源头：fwd 融合 inter+solve_tril（off-diag GEMM + 对角前代 + 4×4 块合并逆）+ bwd 逐列循环梯度（上游 tilelang 测试样例是代码级起点）
2. DESIGN.md（60min）→ masked-GEMM 重表述消循环携带依赖、C/V 多跳限制下的 kernel 拆分与内存预算（UB/L1/L0C 表）、§9.5 性能预算（唯一 IO 1044MB ÷ 1263GB/s = 826us 物理下限）与被否方案
3. debug_log.md（40min）→ 核心坑：G4 UB 窄 region 作 UB→GM 源的打包错位（打包整块中转修复）；G6/G7 flag 16 槽截断 + zN 布局约束（推翻 v3 单 kernel slab 方案）；G11 H=1 的 32B 槽散布；G13 0×NaN 穿越；attempt-1 数学 bug（D_s/V_s 后置缩放从未应用——flag 修好也算不对）
4. perf_tuning/fla_chunk_intra_final_summary.md（40min）→ 基线 19774 → 11443us（-42.1%）怎么来的：锚点消去/中点锚点超级 GEMM、预分裂 ws 直读链、k_g 拆分、fixed-core 任务循环；R2 地板核算（fwd 合成地板 9.8ms 贴线 1.04×、bwd 传输地板）
5. history_version/（30min）→ 被否决路线差在哪：单 kernel bwd（flag 截断 + zN 约束双重推翻）、ILP 交织（+2.5%）、threads=2（+47%，两方向双重证伪）、k_y frag 协议（-1.1% 子阈值）

## 三大关键决策

1. **bwd 拆 2 kernel（k_stage / k_gemm），以 kernel 边界作 V→C 同步**——flag 16 槽实锤（单 kernel 需 40 flag，16-39 全截断碰撞）后，放弃在单 kernel 内表达 V→C 握手（Pipelined 体内 T.Scope 被展平），把同步成本移到 host 顺序 launch。
2. **逐列循环携带依赖 → masked GEMM 重表述**——上游 (B)/(E) 部分的顺序 j 循环重写为只依赖 (i,j) 的掩码 GEMM，消除循环携带依赖，全部进 Cube；掩码常量（Mlo/Mneg）host 缓存。
3. **性能验收"先实测再定目标"**——设计期 roofline 论证 540us 物理不可达（唯一 IO 下限 826us = 1.53× 实测带宽）当场中断交用户；用户决策先实测；R1 后 12.03ms 二次中断；用户决策 R2 再调后交付 → 终态 11.44ms、验收值 ≈11.5ms，PR 如实披露三类数字（目标/下限估算/实测）。

## 交付物

- 分支 `feat/fla-chunk-intra-pr`（origin @18973eb9，基于 upstream/ascendc_pto@557b11b8）/ PR: 待用户网页创建（compare 入口 https://github.com/tile-ai/tilelang-ascend/compare/ascendc_pto...stepjzy:feat/fla-chunk-intra-pr）
- 交付文件：`examples_experiment/dsa_sparse_finetune/example_fla_chunk_intra.py`（1428 行）+ `test_example_fla_chunk_intra.py`（215 行）；交付测试 4 用例全绿（含 bench shape）
- PR 文稿：本目录 `PR_DESCRIPTION.md`（dev-journal 权威正本）

## 遗留疑问 / 未走通的路

- k_inter drain mask 乘（估算 -0.25ms）被 planner UB 预算（192KB 贴线）+ 语句重排确定性挂死阻断——planner 别名/屏障鲁棒性改进后可重试。
- 逼近 826us 唯一 IO 物理下限需上游工具链：单 kernel 多 C 段（消 6× GM 流量放大）、fp32 gemm_v0 修复（消 hi/lo 3×）、V 批量行解算原语（消前代 2.5ms V 语义地板）。
- 设备 ffts 锁存故障：planner 挂死 kernel 级联设备级错误，跨进程 ~40 分钟自愈，容器内无复位权限——纯 torch/简单 kernel 探针可先行判定设备状态，避免误判好代码挂死。
- bwd H=1 的 [64,1] 散写自补偿依赖 H≥2 的 GM 行距条件——通用化需 issue-#1682 类修复（32B 槽散布约定）。
