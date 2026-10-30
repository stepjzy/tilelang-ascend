# fla_chunk_intra_token_parallel 迁移与调优学习档案

## 一句话：这个算子教会我什么
低算术强度 per-token 内积算子的正确出路不是把 Vector 管线优化到极致，而是用 msprof 流水图找到**全程空闲的 Cube 核**把点积整体卸载过去——以及 910B3 分离核架构下 C/V 协作 GM 往返（relay）是绕不开的性能地板。

## 阅读顺序（约 3 h）
1. ref_gpu.py（20min）→ 迁移源头：token-parallel KDA intra-chunk，kgj=k_j·2^(g_i-g_j) 衰减内积
2. ref_gpu_tilelang_gpu.py（20min）→ 上游 TileLang CUDA 版：per-token block + 动态循环 + autotune——注意它与 NPU 后端的全部不兼容点（二维 grid、threads、动态循环边界）
3. DESIGN.md（40min）→ subchunk×head-group 分块决策、技术约束逐项闭环（§9.5 的 370us 物理不可达论证值得先读）
4. debug_log.md（15min）→ beta 窄行 DMA 32B slot 错位（Aqk 完美/Akk 全错的症状隔离法）
5. perf_tuning/perf_log.md（60min）→ **核心**：13 轮迭代全记录——四条被实测证伪的"直觉瓶颈"（barrier、flag 条数、reduce 重组、手动同步）与四个真杠杆（热循环重构/因式分解/GEMM 卸载/C·V 流水）
6. perf_tuning/final_summary.md（15min）→ 基线 19150→1739us（11.0×）全景与三层收敛论证
7. history_version/（20min）→ impl_iter*_before 轨迹：iter7（GEMM 卸载）与 iter8（固定核流水）是两大结构转折点；impl_iter13_candidate 是 relay 重构的"差一点"（Ak 确定性损坏的 pass 耦合证据）

## 三大关键决策
1. **GEMM 卸载（iter7，-49.2%）**：msprof 证伪"带宽瓶颈"假象（AIV 97.98% 满载、DMA 满效率、20 Cube 空转、29% 时间是比 Mul 慢 5.3× 的 Broadcast 指令）→ 点积走 Cube，bf16 操作数被 CPU 模拟精度门限否决（0.9848<0.99），fp32 直传。
2. **固定核 C/V 软件流水（iter8，-39.5%）**：每任务 V→ws→C→ws→V 全串行 → 固定核网格 + 2 任务延迟 drain + NSLOT=2 双缓冲；结果 relay 的手动 FIX flag 用交替事件 ID 配对跨迭代同步（auto FIFO 会同迭代配对死锁）；越界任务 branch-free 重定向到本核首任务（每个 IfThenElse 值 2 个 PIPE_ALL barrier）。
3. **不在假瓶颈上浪费轮次**：iter2（手动同步 -2.3%）、iter4（reduce 重组三路径否决）、iter10（flag 条数 -27% = wall 0%）都是"技术成功、性能证伪"——先差分剖析/差分探针定位真墙（向量链→V 侧负载→DMA 地板），再押注重组，每轮模型预测必须实测裁决。

## 交付物
- 分支 feat/fla-chunk-intra-token-parallel-pr（origin，PR **#1952** open）：f29d3f9f 原始交付 → ecb85906 review 修复（2026-10-30 committer ChaoyangJi 三条意见全采纳：ln2 先减后乘 / UB 容量门 / GGRP 整除，详见 PR_DESCRIPTION.md 修订记录）
- PR: https://github.com/tile-ai/tilelang-ascend/pull/1952
- 性能：19150.1 → 1739.2 us（-90.9%，11.0×）@ B=1,S=8192,H=64,DK=128 bf16；review 修复后稳态 ~1698us（+1.5%，深衰减正确性代价）；精度 26 用例全过（ratio 全 1.0000）

## 遗留疑问 / 未走通的路
- relay 结构重构（iter13 路径③）：DMA 与计算重叠量化成立（mte2 2.79us/任务暴露），但双缓冲实施后 Akk 在 t=1 槽位 100% 确定性损坏（Aqk 位精确）——同步 pass 与循环直线化的耦合，需 pass 级修复（region 粒度依赖分析），DSL 层无解。
- 370us 验收目标在 910B3 + 该接口契约（gk fp32）下物理不可达：IO 下限 593MiB → 纯带宽下限 492us。若未来 A5 融合核（C/V 共享本地内存）落地，relay 地板可整体消除。
- 编译器工具链问题清单（30+ 项，见 perf_tuning/optimization_log.md）：stride-0 广播乘原语缺失、N=1 gm→ub 拷贝损坏、packed [2,*] L0C 双槽 zN 分形腐蚀、AUTO_CV_SYNC 按字面 "workspace" 命名识别、ExtendKillIndex 强制 drain+produce 全程共存等——多数已在本算子中绕开并沉淀了绕法。
