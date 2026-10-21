# fla_wy_fast Stage 3 最终交付报告（fla_wy_fast_final_summary.md）

> 设备：**Ascend 910B3**（64GB，20 AI Cores）｜CANN 9.2.0｜tilelang 0.1.4+1d752539
> 日期：2026-10-22｜计时口径：upstream do_bench 等价（NPU 事件、warmup=20、
> rep=10、均值、仅设备侧）

## 1. 基线 → 最终对比

| 方向 | 基线（未调优） | 最终 | 改善 |
|---|---|---|---|
| fwd（1,8192,64,128,128 bf16+fp32 gate） | 8.745 ms | **4.783 ms** | **-45.3%** |
| bwd（1,32768,8,128,128 全 fp32） | 14.406 ms | **6.935 ms** | **-51.9%** |
| **合计** | **23.151 ms** | **11.719 ms** | **-49.4%** |

> 上表为**上游对齐口径**（wrapper 内无 sync；基线同口径复测 fwd 8.013 /
> bwd 13.622 / 合计 21.636 ms，改善 -45.8%）。
> 与 Stage 2 基线记录**完全同口径**（wrapper 内含 sync）的最终数字：
> **fwd 5.440 / bwd 7.643 / 合计 13.084 ms（-43.5%）**。
> 口径差异说明：Stage 2 wrapper 内部的 `torch.npu.synchronize()` 为自发添加，
> 上游 do_bench 与上游算子 wrapper 均无内部 sync；内部 sync 会把 host 侧 JIT
> 分发延迟（~1ms/launch）暴露进设备侧计时，与"仅设备侧 kernel 时间"口径相悖，
> 已移除（优化项 #10，消融数据见 tuning 文档 §2）。

## 2. 每项优化的贡献分解（消融，逐项单独计时）

| 优化项 | 作用域 | 贡献 |
|---|---|---|
| **T.tile.broadcast + 全量 tile 指令替换 64 行串行循环**（6 kernel：fwd-u/-wkg、k1/k2/v1/v2；fwd-wkg 另含 kg 列广播链 axis=0） | 设备侧 | **-9.21 ms**（逐 kernel：u -1.92、wkg -1.42、k1 -0.99、k2 -0.79、v1 -1.02、v2 -3.08） |
| Msk 常量 host 缓存（去除计时区内 CPU tril + H2D） | bwd 端到端 | **-0.97 ms** |
| wrapper 内部 sync 移除（上游口径对齐） | 端到端 | **-1.37 ms**（13.084→11.719） |
| 回滚项（4）：A cast 链 hoist（中性）、dbeta scatter 移 da（+0.04ms）、k2 CSE（UB 溢出）、kg div 变体（持平） | — | 0（按规则回滚） |

根因链条：23.151 →（broadcast 系列）13.06 →（msk 缓存）→ 13.08* →（sync 移除）11.719
（*msk 缓存与 broadcast 在同批次内先后实施，端到端中间值 8.635→7.663 为 bwd 单独消融）

## 3. 有效带宽利用率（对照实测上限）

**实测 HBM 参照（D2D 大拷贝，256MiB-1GiB 稳定）**：**1259-1273 GB/s (r+w)**。
> 重要修正：DESIGN §9.5 假设的 392GB/s 标称为未校准【推断】，与实测相差 3.2×。
> Stage 2"两方向 350-400GB/s 已近 HBM 饱和"的结论不成立——该数字把 workspace
> 中继流量全部计入 HBM，且对照了错误的带宽上限。

| 口径 | fwd | bwd | 合计 |
|---|---|---|---|
| 唯一 IO 带宽（物理必需字节 ÷ 时间） | 211 GB/s（16.7%） | 213 GB/s（16.9%） | 212 GB/s |
| 逻辑流量带宽（含拆分重读 + workspace 中继） | 648 GB/s（51%） | 816 GB/s（65%） | — |

（% 为实测 1263GB/s 上限的占比；k1/k2 单 kernel 逻辑带宽已达 ~1020GB/s = 81%，
接近饱和；z/da 单发 kernel 仅 ~25%，为延迟受限。）

## 4. 与 1370us 目标的差距（如实）

| 项 | 数值 |
|---|---|
| 最终合计（上游口径） | 11.719 ms = 目标的 **8.55×** |
| 最终合计（Stage 2 口径） | 13.084 ms = 目标的 **9.55×** |
| 理论下限修正（唯一 IO 2.486GB ÷ **实测** 1263GB/s） | **~1.97 ms**（DESIGN 原 6.34ms 基于 392GB/s，需修正） |
| 1370us 隐含带宽 | 1.82 TB/s = 实测上限的 **1.44×** |

**结论（修正后仍成立，倍数修正）**：1370us 目标在 910B3 上**物理不可达**——
所需带宽为实测 r+w 上限的 1.44×（DESIGN 的"4.6×"基于未校准标称带宽，予以修正，
但方向性结论不变）。当前 11.719ms 为实测带宽下限（1.97ms）的 5.95×，差距
构成：kernel 拆分重复读（工具链单 Pipelined 体/多 gemm 组 bug）+ C/V workspace
中继（无 UB↔L1 片上直连路径）+ 单发 kernel 延迟（z/da 合并被 sync-point bug
阻塞）+ host JIT 分发（~1ms/launch，工具链框架层）。

## 5. 精度复验（最终门禁）

`python test_fla_wy_fast.py --level all` → **Test Passed!（exit 0）**
- L0 4/4、L1 4/4 全部 `[PRECISION_PASS]`（含 bench shape 与 d96 非 64 倍 D）
- L2 7/7 全部正确拒绝（含 D=48 无合法 block 的工具链限制用例）
- Boundary 5/6 PASS + 1 预期内 WARN（bnd_bwd_gk_extreme：fp16 拆分范围限制，
  与基线行为逐位一致，非本次调优引入）
- 每项优化实施后均先过 L0 再采纳；全量复验在批次末与最终交付前各执行一次

## 6. 交付文件清单（绝对路径）

| 文件 | 内容 |
|---|---|
| `/mnt/workspace/gitCode/skybasejzy/tilelang-ascend/examples_experiment/dsa_sparse_finetune/fla_wy_fast.py` | 调优后 kernel（接口不变，test 兼容） |
| `.../perf_tuning/fla_wy_fast_tuning.md` | 瓶颈定位数据 + 逐项优化消融记录 |
| `.../perf_tuning/fla_wy_fast_final_summary.md` | 本报告 |
| `.../perf_tuning/optimization_log.md` | skill 规范的优化点清单/ORDER-PLAN/逐项 RESULT |
| `.../perf_tuning/perf_log.md` | 跨轮结构化日志 |
| `.../perf_tuning/baseline_iter1.json` | 基线记录（含口径与假设修正说明） |
| `.../perf_tuning/fla_wy_fast_impl_iter1_before.py` | 调优前实现备份（回滚基线） |
| `.../perf_tuning/bench_per_kernel.py` | 逐 kernel 计时脚本（含 HBM 参照） |
| `.../debug_log_fla_wy_fast.md` | 追加 Stage 3 踩坑（见 §8） |

## 7. 未走通的路（算子层已尝试并回滚/阻塞）

1. **k2/v2 A cast 链 prologue hoist**：中性（Pipelined 软件流水已重叠不变量）；
   k2 侧 UB 溢出（DK=128）。
2. **dbeta scatter 移 da（去 MTE barrier）**：净 +0.04ms（barrier 被流水吸收，
   da 无循环隐藏新增开销）。
3. **k2 公共子表达式 Xg=X·2^g**：r_tmpk 生命周期延长 +16KB → UB 溢出。
4. **GM↔UB bf16↔fp32 cast-on-copy 直载直存**：工具链 codegen 模板错配（见 §8-6）。
5. **kg div 变体**：div 与 exp 在 Vector 核同价，无收益。
6. **z+da 合并**：复现 sync-point 编译错（debug_log #6，工具链阻塞）。
7. **bwd 6→2-3 kernel / fwd 2→1 合并**：单 Pipelined 体 + 多 gemm 组 L0 损坏
   （工具链阻塞，Stage 2 已论证）。
8. **threads=2**：gather 链 vid 切分越界（debug_log Attempt 1，工具链阻塞）。

## 8. 工具链级缺陷清单（供上游修复参考，不在本算子 PR 范围内）

1. **fp32 gemm_v0 损坏**（Stage 2 已记录）：隔离复现产出 1e37 级交替垃圾；
   本算子被迫以 fp16 hi/lo 3-gemm 模拟（bwd GEMM 次数 ×3）。
2. **单 cross-core Pipelined 体/kernel + extent-1 崩溃 + extent≥3 双缓冲回卷
   损坏**：强制 block=D/2、恰好 2 迭代，阻塞 kernel 合并与更细流水。
3. **多 gemm 组共享 L1 操作数（角色互换复用）损坏 L0 staging**：阻塞 bwd
   k1+k2 / v1+v2 合并（输入重复读的主因）。
4. **后置多跳 C/V 交替（V→C→V→C→V）sync-point 计数失配**（编译错）：阻塞
   z+da 合并（`Mismatch in sync points: cube has 1, vec has 0/2`）。
5. **prologue V→C 交换被 pipeline pass 错误版本化**（stage 偏移读一次性写入的
   workspace）：阻塞循环不变操作数 hoist。
6. **GM↔UB cast-on-copy（bf16↔fp32）codegen 缺陷**：生成
   `copy_gm_to_ub<bfloat16_t>(LocalTensor<float>, ...)` 非法模板调用（探针
   `perf_tuning` 内可复现；cast_on_copy 测试矩阵缺 bf16↔fp32 用例）。
7. **L0C(fp32)→GM(bf16) 直存 fixpipe 转换损坏**（Stage 2 已记录）。
8. **JIT launch 路径每次调用重建 TVM arith Analyzer + 查询 stream**（
   `tilelang/jit/adapter/cython/cython_wrapper.pyx: forward()`）：~1ms Python/
   launch；Analyzer 应在 adapter 构造时缓存。
9. **行广播 out-of-place 2D T.Parallel scalar 索引 bug**（GetValue(i·dtype_size)，
   Stage 2 已记录）：本算子以 `T.tile.broadcast` 规避（反而更优）。
10. **memory-planning 对 k2 类满 UB kernel 的生命周期延长零容忍**：CSE/hoist
    类优化在 DK=128 一律溢出拒绝（16KB 余量都没有）。
