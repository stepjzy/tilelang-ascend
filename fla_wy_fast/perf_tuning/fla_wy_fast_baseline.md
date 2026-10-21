# fla_wy_fast 未调优性能基线（Stage 2 交付，910B3）

> 记录时间：2026-10-22。精度全量通过（L0/L1 [PRECISION_PASS]，L2 正确拒绝，
> Boundary 5/6 PASS + 1 预期内 WARN）后的**未调优**基线。调优属 Stage 3。

## 计时口径

对齐上游 `ref_gpu/test_utils_kda.py do_bench`：NPU 事件计时
（`torch.npu.Event(enable_timing=True)`），warmup 20、rep 10、取平均，仅设备侧
kernel 执行时间（host 包装内含 `torch.npu.synchronize()`，每 rep 完整落地）。
脚本：`bench_fla_wy_fast.py`（`elapsed_time` 单位 ms）。

## 基线数据（Ascend 910B3，64GB，CANN 9.2.0，tilelang 0.1.4+1d752539）

| 方向 | benchmark shape | dtype | 时间 |
|---|---|---|---|
| fwd | B=1, S=8192, H=64, DK=DV=128, chunk=64 | bf16 in/out + fp32 gate | **8.8 ms** |
| bwd | B=1, S=32768, H=8, DK=DV=128, chunk=64 | 全 fp32 | **14.3 ms** |
| 合计 | — | — | **23.1 ms** |

## 对照分析

| 项 | 数值 | 说明 |
|---|---|---|
| DESIGN §9.5 理论带宽下限 | fwd 2.57ms + bwd 3.77ms ≈ **6.34ms** | 唯一 IO ÷ 392GB/s 标称 |
| 本基线 | 23.1ms = 下限的 **3.65×** | 未调优架构的流量放大（见下） |
| 领导目标 1370us | 23.1ms 为其 **16.9×** | DESIGN 已论证 910B3 物理不可达（需 1.82TB/s = 4.6× 标称带宽） |

### 与理论下限差距的构成（Stage 3 优化空间）

1. **kernel 拆分的输入重复读**（工具链约束，见 debug_log Attempt 2）：
   fwd 拆 2 kernel（A/beta 双份读）；bwd 拆 6 kernel（k/g/dw/v/du 多份读 +
   dA_part/S/Z 中转往返）。估算 +1.5~2× 流量。
2. **C/V 交换经 GM workspace 中转**（工具链无片上直连路径可用）：每个 gemm 操作数
   的 UB→L1 搬运写+读 GM。fwd bf16 拆分 2 份操作数 + bwd fp16 拆分 3×gemm
   操作数 → gemm 输入流量约 ×2~×3。
3. **fp16 拆分 gemm 3× 计算量**（bwd）：77.4 GFLOP fp16，但仍远低于带宽瓶颈。
4. 有效带宽估算：fwd ≈ 3.5GB/8.8ms ≈ 400GB/s、bwd ≈ 5GB/14.3ms ≈ 350GB/s——
   **两方向均已逼近 HBM 饱和**，说明当前瓶颈是流量放大而非调度/计算。

### Stage 3 优化方向（按预期收益排序）

1. 合并 bwd 6 kernel → 2~3（依赖工具链修复多 gemm 组 L0 staging / sync-point
   计数），消除重复输入读与 dA_part/S/Z 中转（预计 -30%~-40%）。
2. bf16/fp16 拆分的 L1 操作数直连（工具链 UB→L1 片上路径修复后），消除 gemm
   输入 GM 往返（预计 -20%~-30%）。
3. fwd 双 kernel 合并（单 Pipelined 约束解除后）。
4. beta 列载入改 32B 对齐头组 tile + 组内 gather（-6% fwd 流量）。
5. block_D / num_stages 枚举调优（当前固定 block=D/2、stages=2）。

> 注意：以上 1/2/3 均依赖 tilelang-ascend 工具链修复（fp32 gemm、多 gemm 组共享
> 操作数、sync-point 计数、Pipelined extent=3 回卷、行广播 scalar 索引——
> 详见 debug_log_fla_wy_fast.md），非算子层可独立完成。
