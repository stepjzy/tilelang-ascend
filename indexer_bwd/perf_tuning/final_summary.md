# indexer_bwd Stage 3 性能调优 — 最终报告

日期：2026-10-11 ｜ 目标：fast 路径纯 kernel 平均 < 80 us（l0_default：B=1,S=2048,H=16,D=128,topk=64,bf16，fast layout=0 + cast，torch.npu 事件 5 warmup + 20 iters）

## 结果

| 指标 | 基线 | 最终 | 改善 |
|------|------|------|------|
| fast+cast 纯 kernel 平均 | 1229.9 us（任务书 1173.9，同会话复测） | **67.9-69.3 us**（连续 3 次官方口径：69.3 / 67.9 / 68.9） | **-94.4%** |
| msprof fast-only Task Duration | 223.7 us | 65.5 us | -70.7% |
| 精度 | L0 7 + L1 17 全 PASS | 全量 PASS（每轮迭代后复验：--level all = L0 7 + L1 17 + L2 8 + boundary 4，dispatch 三路等价） | 无退化（ratio 与基线逐项一致） |

## 迭代摘要

| 轮 | 改动 | 实测 | 累计 |
|----|------|------|------|
| 0 | 基线（rev2 双路径） | 1229.9 | — |
| 1 | _CachedLauncher 缓存式低开销 launch（绕过 cython 重编组 + TVM Analyzer ~550us/次；raw stream 查询 0.9us；lib.call 12.9us） | 212.2 | -82.8% |
| 2 | 设备侧重构：C/V 跨 sub-tile 软件流水（produce 提前一拍）、flat-2D Q/dq 视图（整块 DMA + L0C→GM 直写）、bf16 logits 中转（relu+cast 折入 FIX）、big-tile [TS_H, topk] 向量链（单 compare 因果掩码） | 111.6 | -90.9% |
| 3 | op 削减：s 折入 w、row_shift 块级预构造、w_big 直接 slice broadcast、dw flat 累积、dlog 单 buffer（gqa_sink ds_l1 先例） | 96.2 | -92.2% |
| 4 | MIX_AIC_1_2（(cid, vid) launch，40 AIV 全用）：AIV-vid 按行半分（每次中转双 AIV 各写一半 — 1_2 flag 语义所需） | 86.9 | -92.9% |
| 5 | cast kernel (cid,vid)+cast_rows=64；连续任务分配 + K 段内复用（65→~22 次 K 装载） | 83.4 | -93.2% |
| 6 | pair-GEMM（token 序，M=128 合并：24→12 gemm/block）+ dk 段内累积（65→~22 atomic） | 68-72 | -94.4% |

## 关键根因与教训（详见 perf_log.md 各轮）

1. 基线 1229.9us 中 ~1230us 为 host launch 开销（设备仅 223.7us）— 官方事件计时对 back-to-back launch 敏感。
2. 原实现的 CV 严格 lockstep：AIC 每 block 54.7us 中 50.5us 在等 cross-core flag（msprof wait_id0）。
3. MIX_AIC_1_2 的 cross-core flag：AIC 的 wait 需要两个 AIV sub-block 都 set（奇偶分裂单 AIV set → 确定性死锁 507014；双 AIV 各写一半 = sparse_flash 已验证形态）。
4. L0C 为 N-major 16 列 band 布局、L1 gemm 操作数为 zN 分形 band：**行切片不可单指针表达** — 行重排方案不可行，token 序 + 整块拷贝是唯一正确形式（两次失败实验确证：dq ratio 0.065 / FIXP OOB）。
5. 恰满 128KB L0C 本身可用（此前崩溃源于错误 chunk 偏移，非容量）。

## 最终架构（fast layout=0）

- persistent grid=20，MIX_AIC_1_2（1 AIC + 2 AIV/block），连续任务分配
- pair 级软件流水：C produce(pair p) 提前一拍；12 gemm/block（M=128/128/64）
- AIV-vid 拥有 sub-tile 2p+vid 的完整 [TS_H, topk] 链（~20 tile ops/sub-tile）
- 中转：ws_logits bf16（relu+grain-round-1 折入 FIX）、ws_dlog a/b 按 pair 奇偶轮转、ws_dq 仅尾 pair
- dq 整块 L0C→GM 直写；dk L0C 段内累积 + 段尾整块 fp32 region atomic；尾块 score 零填充（validity 折叠）
- host：_CachedLauncher（flat-2D/1D 零拷贝视图 + 缓存 ctypes 参数 + raw stream）

## 剩余观察

- AIC 16.2 / AIV 15.9 us/block（cube 2.0 + mte1 1.9 + mte2 4.2 + fix 2.6 + scalar 4.7 / vec 6.9 + mte 3.5 + scalar 5.4）— 两侧均衡，进一步压缩需 TS 参数变更（测试固定 TS=4/T_BLOCK=32，不可动）或框架级 gemm_v0 flag 链改造。
- 偶发 82-86us 首跑离群（设备空闲后状态；进程内连续复测稳定 70-72）。
