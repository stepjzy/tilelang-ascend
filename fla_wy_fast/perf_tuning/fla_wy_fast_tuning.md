# fla_wy_fast Stage 3 性能调优报告（fla_wy_fast_tuning.md）

> 环境：Ascend 910B3（64GB，20 AI Cores）/ CANN 9.2.0 / tilelang 0.1.4+1d752539
> （ascendc_pto 基线）。日期：2026-10-22。
> 计时口径：NPU 事件计时、warmup=20、rep=10、均值、仅设备侧（upstream
> `ref_gpu/test_utils_kda.py do_bench` 等价）。
> 脚本：`bench_fla_wy_fast.py`（端到端）、`bench_per_kernel.py`（逐 kernel）。

## 1. 瓶颈定位（数据支撑）

### 1.1 逐 kernel 分解（调优前，bench shape）

| kernel | 时间 | us/块 | 逻辑 GM 流量* | 唯一 HBM 流量 |
|---|---|---|---|---|
| **HBM 参照（D2D 拷贝 512MiB r+w）** | 0.848 ms | — | 1 GiB | **1263-1266 GB/s** |
| fwd-u（u = A@(β·v)） | 3.789 ms | 9.2 | ~1.4 GB | ~321 MiB |
| fwd-wkg（w/kg） | 4.238 ms | 10.3 | ~1.7 GB | ~705 MiB |
| bwd-k1（dA_part） | 2.311 ms | 11.3 | ~1.35 GB | ~577 MiB |
| bwd-k2（X→dk2/dg2/dbeta） | 2.588 ms | 12.6 | ~1.8 GB | ~1.09 GB |
| bwd-v1（S） | 2.248 ms | 11.0 | ~1.0 GB | ~385 MiB |
| bwd-v2（Y→dv/dbeta） | 4.332 ms | 21.1 | ~1.0 GB | ~450 MiB |
| bwd-z（Z=S@Aᵀ） | 0.808 ms | 3.9 | ~260 MiB | 192 MiB |
| bwd-da（dA） | 0.803 ms | 3.9 | ~260 MiB | 192 MiB |

\* 逻辑流量 = 唯一 IO + kernel 拆分重复读 + C/V combine 的 GM workspace 中继
（每个 gemm 操作数 UB→GM→L1 一写一读、L0C→GM→UB 一写一读）。

### 1.2 三个关键判定

**判定 1：Stage 2 的"HBM 已饱和（350-400GB/s）"结论不成立。**
大容量 D2D 拷贝实测 **1263-1266 GB/s (r+w)**，为 DESIGN §9.5 假设标称
（392GB/s，标注【推断】未校准）的 **3.2 倍**。即便把全部 workspace 中继算作
HBM 流量，fwd-u 理想耗时 ~1.1ms，实测 3.789ms（3.4× 差距）；按唯一流量算
仅 ~0.27ms（14×）。→ 瓶颈不在 HBM 带宽。

**判定 2：瓶颈是 Vector 核的串行行循环指令下发开销。**
- 无串行行循环的 z/da 单发 kernel：3.9us/块；
- 串行行循环最多的 v2（dv 64 行 + A 链 ×2 slice）：21.1us/块。
- 生成代码证据（`get_kernel_source()`）：`T.serial(64)` 行循环 **与** in-place
  2D `T.Parallel` 行广播（fwd-u 原实现）都 lower 成 64 次标量循环——
  `bt.GetValue(i)` + 每行一条 `Muls`（repeat=1 瘦指令，每次仅 64 元素/256B）。
  fwd-wkg 的 kg 循环最重：64 行 × 4 ops × 2 slice = **512 瘦指令/块**。
  对应反模式：`performance-antipatterns.md` "Vector Core 内逐元素/逐行 for
  loop 计算"。

**判定 3：结构性放大（kernel 拆分 + workspace 中继）是第二层瓶颈。**
bwd 6 kernel 使 k/g/dw/v/du 各读 2 次、A 读 4 次（逻辑流量 ~6.5GB vs 唯一
1.478GB）；C/V 中继经 GM workspace。修复被工具链 bug 阻塞（单 cross-core
Pipelined 体限制、多 gemm 组 L0 staging 损坏、sync-point 计数——见
debug_log Attempt 2），Stage 3 维持 8 kernel 结构。

### 1.3 host 侧定位（cProfile + 空队列探针）

- tilelang JIT 每次 launch 的 Python 分发 ~1ms（cython_wrapper.forward 每次
  **重建 TVM arith Analyzer** + 查询 current_stream + packed_func marshaling）。
- Stage 2 wrapper 内部 `torch.npu.synchronize()` 使每 rep 首次分发延迟暴露进
  设备侧计时（上游 do_bench 与上游算子 wrapper 均无内部 sync）。
- bwd wrapper 每次调用重建 Msk（CPU tril + H2D pageable 拷贝）在计时区内。

## 2. 优化项与消融记录（每项单独开关、单独计时）

| # | 优化项 | kernel | 前 → 后（ms） | Δ | 采纳 |
|---|---|---|---|---|---|
| 1a | bt 行广播：`broadcast(axis=1)`+全量 mul 替换 64 行循环 | fwd-u | 3.789 → 2.023 | **-47%** | ✅ |
| 1b | 同上（kbg） | fwd-wkg | 4.238 → 3.118 | -26% | ✅ |
| 2 | kg 列广播链：`broadcast(axis=0)`+sub/exp/mul 全 2D | fwd-wkg | 3.118 → 2.971 | -4.7% | ✅ |
| 1c | kbg 广播 | bwd-k1 | 2.311 → 1.313 | -43% | ✅ |
| 1d | vb 广播 | bwd-v1 | 2.248 → 1.229 | -45% | ✅ |
| 1e | kbg+dk2 广播 | bwd-k2 | 2.588 → 1.797 | -31% | ✅ |
| 1f | dv 广播 | bwd-v2 | 4.332 → 1.258 | **-71%** | ✅ |
| 3 | A cast 链提 prologue（relay 留循环内） | k2/v2 | v2 1.258→1.278；k2 DK=128 UB 溢出 | 中性/阻塞 | ❌ 回滚 |
| 5 | dbeta scatter 从 v2 移 da（去 barrier） | v2+da | 2.061 → 2.099 | +1.8% | ❌ 回滚 |
| 6 | k2 公共子表达式 Xg=X·2^g | bwd-k2 | 编译失败（UB 溢出 +16KB） | — | ❌ 回滚 |
| 7 | GM↔UB bf16↔fp32 直载直存（去 staging） | fwd-u/wkg | 编译失败（工具链） | — | ❌ 回滚 |
| 8 | kg div 变体（1D exp + div） | fwd-wkg | 2.971 → 2.987 | 持平 | ❌ 回滚 |
| 9 | Msk 常量缓存（host） | bwd 端到端 | 8.635 → 7.663 | -11% | ✅ |
| 10 | wrapper 内部 sync 移除（上游口径对齐） | 端到端 | 13.084 → 11.719 | -10% | ✅ |

消融方法论备注：
- #3 的中性结果说明 **Pipelined 软件流水已自动把循环不变量 cast 链与 gemm 重叠**
  （手动 hoist 冗余），同时暴露 k2 在 UB 容量边缘（任何生命周期延长都会溢出）。
- #5 说明 v2 尾部 MTE3→MTE2 barrier 的成本被流水线吸收（~0），而 da 无循环可
  隐藏新增的 load+scatter → 结构迁移需按 kernel 流水形态评估，不能只数操作。
- #8 说明 Vector 核上 **div 与 exp 同价**（都是超越函数级吞吐），1D exp 换 2D
  div 无收益。
- 基线同口径复测方法：`perf_tuning/fla_wy_fast_impl_iter1_before.py` 复制后仅
  删两处 `torch.npu.synchronize()`（保留其原 per-call msk），同一 do_bench 测得
  基线上游口径 fwd 8.013 / bwd 13.622 / 合计 21.636 ms。

## 3. 调优后逐 kernel 状态

| kernel | 基线 ms | 最终 ms | Δ | us/块 | 逻辑带宽 (GB/s) |
|---|---|---|---|---|---|
| fwd-u | 3.789 | 1.870 | -51% | 4.6 | ~750 |
| fwd-wkg | 4.238 | 2.819 | -33% | 6.9 | ~600 |
| bwd-k1 | 2.311 | 1.321 | -43% | 6.5 | ~1020 |
| bwd-k2 | 2.588 | 1.802 | -30% | 8.8 | ~1015 |
| bwd-v1 | 2.248 | 1.224 | -46% | 6.0 | ~820 |
| bwd-v2 | 4.332 | 1.257 | -71% | 6.1 | ~765 |
| bwd-z | 0.808 | 0.825 | +2%（噪声） | 4.0 | ~315 |
| bwd-da | 0.803 | 0.807 | 噪声 | 3.9 | ~320 |
| **设备合计** | 16.12 | **11.93** | **-26%** | — | — |

（逻辑带宽 = 逻辑 GM 流量 ÷ 时间，对照 D2D 实测上限 1263GB/s：k1/k2 达
81%，z/da 仅 25%——后者为单发延迟受限，无循环可流水。）

## 4. 端到端结果

| 口径 | 基线 | 最终 | 改善 |
|---|---|---|---|
| Stage 2 口径（wrapper 内含 sync，与基线记录同口径） | 23.151 ms | 13.084 ms（fwd 5.440 + bwd 7.643） | **-43.5%** |
| 上游对齐口径（wrapper 无 sync，upstream do_bench 语义） | 21.636 ms | **11.719 ms（fwd 4.783 + bwd 6.935）** | **-45.8%** |

## 5. 剩余瓶颈与未走通的路

1. **kernel 合并**（bwd 6→2-3、fwd 2→1、z+da 合一）：全部被工具链 bug 阻塞
   ——单 cross-core Pipelined 体限制、多 gemm 组共享操作数 L0 staging 损坏、
   后置多跳 C/V 交替 sync-point 计数失配（z+da 合并即复现该编译错）。
2. **C/V workspace 中继消除**（UB↔L1 片上直连）：工具链无可用路径；这是
   gemm 输入输出流量 ~2-3× 放大的根因。
3. **GM↔UB cast-on-copy（bf16↔fp32）**：codegen 生成
   `copy_gm_to_ub<bfloat16_t>(LocalTensor<float>...)` 非法模板调用（探针复现），
   fwd 的 bf16 staging buffer + 显式 cast 无法去除。
4. **JIT launch host 开销 ~1ms/次**：`cython_wrapper.forward` 每次 call 重建
   TVM `Analyzer()` + 查询 `torch.npu.current_stream()`；8 kernel 串行场景，
  上游口径下被队列深度部分掩盖。
5. **UB 容量**：k2 处于 memory-planning 边缘，CSE / A-hoist 均因生命周期延长
   16KB 溢出被拒。
6. **z/da 单发 kernel ~4us/块**：结构（2 加载 + 12 拆分链 ops + 3 gemm + 输出）
   无循环可流水，合并路径被 1 阻塞。
