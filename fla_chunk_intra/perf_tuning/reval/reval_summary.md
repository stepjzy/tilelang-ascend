# fla_chunk_intra 新环境重验证（2026-11-05，revalidation）

分设架构（CPU 唯一真源编辑 / NPU `DevEnvC_DCbyi_npu` 纯执行）下按新版
tilelang-op-migration skill 重走验证流程。代码零改动（md5 与交付基线一致），
全部数据为新环境实测。

## 0. 环境与基线对账

| 项 | 值 |
|---|---|
| NPU 环境 | `DevEnvC_DCbyi_npu`（ASCEND 1×910B3，22 vCPU，76GiB；2026-11-04 建） |
| CPU/NPU 基线 | 双端 `git rev-parse HEAD` = 463c234c（ascendc_pto，一致） |
| 代码版本 | `fla_chunk_intra.py` md5 ef253da1（= journal impl_final.py = -pr 分支 AST 等价版） |
| 环境差异信号 | NPU 侧日志：`CANN and HDK(driver) versions require processing for 32 padding size`（CANN/HDK 版本配对与旧机不同） |
| 验证方式 | cannlab exec/job 远程执行，产物当轮 cp 取回（reval_precision.log / reval_bench.log / reval_prof/ / reval_gap.log） |

## 1. 精度（新环境全量重跑）

`--level all`：**59 PRECISION_PASS + 31 BOUNDARY_PASS + 0 FAIL + 0 WARN = 90 项全过**，
exit 0（与旧环境套件构成逐项一致）。日志：`reval_precision.log`。

## 2. 性能（do_bench 官方口径：NPU 事件，warmup=20 rep=10 mean，仅设备侧窗口）

| 项 | 新环境 4 次 | 新环境代表值 | 旧环境（R2 终态） | Δ |
|---|---|---|---|---|
| fwd chain (1,8192,64,128) bf16 | 10358.5/10506.8/10373.8/10477.7 | **10429.2 us** | 10159.3 | **+2.6%** |
| bwd chain (1,8192,8,128) fp32 | 1755.7/1680.0/1620.5/1648.1 | **1676.1 us** | 1283.5 | **+30.6%** |
| 合计 | 11994-12187 | **12105.3 us** | 11442.8 | +5.8% |

日志：`reval_bench.log`。

## 3. OPPROF 逐 kernel 读数（msprof op，单发 Task Duration + per-block pipe 均值）

| kernel | dur(us) | blk | 关键读数（per-block/子核均值 us） |
|---|---|---|---|
| inter | 3041.7 | 20 | V 累计 3013.6：vec 1783(59%) + scalar 1675(55%) + mte3 631 + mte2 478；V vec_stall 2004（行递归 RAW）；C cube 仅 190(6%)，C mte1_stall 435 → **V 主导（每 core V bound）** |
| subst | 2822.3 | 8192 | 纯 V：t=12.5，vec 8.3(67%)，vec_stall 8.4 → **前代行递归 V 语义地板**（与旧环境同构） |
| g / g2 / y / ai | 1170/1056/1188/1117 | 8192 | V t≈2.7-3.2（wait_id0≈1.5 主导）+ C t≈1.8-1.9 → **C→V frag relay 依赖等待**（与旧环境同构） |
| stage | 221.2 | 1024 | V t=7.8（vec 4.2 + mte2 2.5 + mte3 2.6）→ V 串行链 + GM 写（同构） |
| gemm | 571.8 | 1024 | V t=18.0（mte2 9.6 主导 + vec 5.3 + mte3 4.2）+ **wait_id9=5.4（FIX flag 等 C 侧）**；C t=10.4（mte2 5.5 + scalar 3.9），C mte1_stall 8.0 → **V mte2 传输墙 + flag 握手**（同构） |

fwd 六 kernel 单发合计 10395 ≈ bench 链 10429（**fwd 设备侧无间隙，device-bound**）。
bwd 两 kernel 单发合计 **793us ≪ bench 链 1676us —— 链式与单发差 ~0.88ms**。

## 4. host 分发地板核算（reval_gap_probe，NPU 环境 host = 22 vCPU ARM）

| 探针 | 值 | 含义 |
|---|---|---|
| A1 stage_alone（do_bench 单核） | 755.4 us | = host 提交 ~534 + 设备 221 |
| A2 gemm_alone（do_bench 单核） | 954.0 us | = host 提交 ~382 + 设备 572 |
| B chain（do_bench） | 1755.7 us | 与 bench 带一致 |
| C 链内事件切分 | stage 窗 734.8 / gemm+gap 窗 1211.6 | 间隙在 kernel 提交等待内 |
| **D host_launch_per_call（wall，无 sync）** | **1450.2 us / 链调用** | **新环境 host 分发地板**（2 次 launch + wrapper + 5 次 alloc） |
| E workspace_alloc_set | 70.8 us | 缓存分配器开销可忽略（非 alloc 问题） |

**结论：bwd +30.6% 全部来自 host 分发地板逐 rep 暴露**——新环境 host 每链调用需
~1.45ms（每次 kernel launch ~0.7ms：JIT 适配层 Cython forward + TVM Arith 重建 +
新 CANN/HDK 运行时 32B padding 处理），超过 bwd 设备工作量 793us，do_bench 事件窗
口逐 rep 包含设备等 host 的空隙（pipelining 失效，skill 地板核算条款预言的 regime）。
fwd 设备 10.4ms ≫ host ~4.2ms，保持 device-bound，仅 +2.6%（噪声量级）。
**设备侧 kernel 本身无退化**（bwd 单发 793us，含与旧环境同构的 pipe 签名）。

## 5. 剩余差距分类（按 skill 路线穷尽判据）

- **kernel 层（本仓可做）**：无新增项——bwd 均值已由 host 地板主导，继续压 kernel
  （设备 793us）不改善 do_bench 均值；fwd 与旧环境同构（贴合成地板 1.04×）。
- **框架/工具链层（需配合）**：①host 分发 ~0.7ms/launch（新环境 22vCPU ARM + 新
  CANN 运行时）——需 AclGraph / CUDA-Graph 类捕获重放、launch 批量化或 JIT 适配层
  减负，使 host 速率 < 设备工作量后 pipelining 恢复；②既有项不变（单 C 段拆分、
  fp32 gemm、V per-op 发射）。
- 跨环境对比标注：旧环境（本地直连 910B3 主机，x86 host）11.44ms vs 新环境
  （分设 22vCPU ARM host）12.11ms；do_bench 口径两者一致（事件窗口均含 host 间隙
  的暴露量不同）。

## 6. 双口径画像产物（本目录）

- `reval_prof/opprof_{inter,subst,g,g2,y,ai,stage,gemm}/OPPROF_*/`——agent CLI 读数
  口径（CSV 逐 pipe + dump/），本轮全量留存；
- `reval_prof/prof_app/PROF_000001_*/`——应用级 msprof（bench 链完整 Timeline，
  MindStudio Insight 图形化口径，可直接观察 bwd 两 kernel 间的 host 间隙）。
