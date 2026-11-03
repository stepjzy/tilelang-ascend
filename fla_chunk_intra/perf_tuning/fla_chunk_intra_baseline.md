# fla_chunk_intra Stage 3 起点基线（R1 iter0）

日期：2026-11-03（Stage 3 R1 开始时点）
快照：`history_version/fla_chunk_intra_perf_r1_iter0_before.py`（md5 d54201649875e50aca0a3f17dfd8a7ce）

## 精度状态（起点）

attempt-2 收敛态：L0 7/7 + L1 9/9 + L2 12/12 正确拒绝 + Boundary 5 PASS / **2 WARN**
（bnd_bwd_decay dk2/dg2 ratio=0.5/max_abs=nan——旧逐 s 锚点使 Kdec 类操作数指数达 +18.24，
fp16 hi/lo 溢出为 inf，与掩码零相乘得 NaN；DESIGN §10.3 R9 已知域限制）。

## 性能基线（do_bench：NPU 事件，warmup=20 / rep=10 / mean，仅设备侧）

fwd shape (1,8192,64,128) bf16 in / fp32 gate；bwd shape (1,8192,8,128) fp32。

| 方向 | kernel | 基线 us | 备注 |
|---|---|---|---|
| fwd | k_inter | 5164.1 | 逐 s2 锚定操作数（9 组）18 gemm_v0，~110 V op，84KB relay/块 |
| fwd | k_g | 5672.3 | 4×14 行递归串行前代 + akbp staging/split/relay，AIC 占用 13.4us/块中 8.6 等 aid relay |
| fwd | k_g2 | 1576.2 | G@G，A/B 从 fp32 ws 重 split + 4 relay |
| fwd | k_y | 2404.8 | (I+G2)@AiDiag，同上 |
| fwd | k_ai | 1852.6 | (I+G)@Y，同上 |
| fwd | **chain（5 kernel）** | **16451.4** | sum-of-parts 16670.0 |
| bwd | k_stage | 1195.1 | 纯 V 逐 tile staging 210KB/cid（OpscrM 54.5KB + Opscr 77.8KB/半）|
| bwd | k_gemm | 2088.2 | 48 gemm_v0/半 + 132KB staging 读/半，mte1_stall 26.6us/块 |
| bwd | **chain（2 kernel）** | **3290.5** | sum-of-parts 3283.3 |
| **合计** | | **19741.9** | attempt-2 档案 19774.3（差 0.2% 内复现）|

参考地板（DESIGN §9.5）：唯一 IO 1044MB ÷ 1263GB/s = **826us** 物理下限（完美单体融合，
当前工具链 C/V 多跳限制下不可达）；6-kernel 逻辑结构下限 1.34ms（100% BW）/ 1.65ms（81%）。
proto 540us 目标物理不可达（需 1.93TB/s = 实测 r+w 上限 1.53×）。

## 起点画像要点（msprof_intra_*，per-block 均值）

- fwd 为 transfer+sync 混合：每 kernel V→C→V 块级锁步（auto-relay flag 等待占 AIC 占用大头），
  auto workspace 中继流量 ~2.4GB r+w（DESIGN 模型外），V 侧 scalar 发射流（k_inter 8.6us/块）
  与串行前代（k_g vec 7.7us/块）为 V 墙。
- bwd 为 transfer：staging GM 中转 ~678MB（写 210KB + 读 228KB + Fragscr 96×2 per cid）。

## 本轮任务口径

用户决策 = 先实测再定目标：尽可能逼近下限，产出可信实测数据与剩余差距归因，供重定目标。
每采纳一项优化：bench 配置 + L0 全量双门禁；终态 --level all + 官方口径 bench ×3。
