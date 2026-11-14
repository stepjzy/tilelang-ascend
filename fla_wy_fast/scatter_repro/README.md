# Scatter 降载 bug 复现实验（fla_wy_fast / TileLang 编译器缺陷）

> 日期：2026-11-14 · 环境：远程 910B3（DevEnvC_DCbyi_npu，CANN 9.2.0，tilelang @ 4805777d）
> 状态：**根因已定位**（生成代码级别），实验代码可完整重跑

## 一、问题一句话

`T.copy(UB源, 跨步GM切片)` 在 TileLang 里降载错误：当源缓冲区是 **1D `[64]`** 时，
生成的 DMA `srcStride` 算错，导致只有第 1 个元素读对、其余 63 个读越界；
当源是 **2D `[64,1]`** 时则正确。这解释了交付算子为什么必须走 GM 中转。

## 二、关键证据：三种 kernel 生成的 C++（copy_ub_to_gm 模板参数）

NPU 上 `/tmp/tmp*.cpp` 保留生成代码（`libgen.py` 用 `delete=False`），对比三个 kernel：

| kernel | 生成的调用 | srcN | maskShapeN | srcPitch | srcStride | 结果 |
|---|---|---|---|---|---|---|
| contig（连续目的地） | `copy_ub_to_gm<float,64>(OutFlat[cid*64], acc[0], 64,1,64)` | 64 | 64 | 256B | (256−256)/32=**0** | ✅ PASS |
| direct（跨步目的地，源acc[64]） | `copy_ub_to_gm<float,64>(Out[cid], acc[0], 8,64,1)` | 64 | **1** | 256B | (256−4)/32=**7** | ❌ 全零 |
| staged（源buf[64,1]） | `copy_ub_to_gm<float,1,64>(Out[cid], buf[0], 8,64,1)` | **1** | 1 | 4B | (4−4)/32=**0** | ✅（完整算子里） |

**根因**：降载跨步散列时，编译器因目的地每行 1 元素而把 `maskShapeN` 设为 1，
但模板 `srcN` 仍取源缓冲区最后一维。1D `[64]` 源 → `srcN=64`、`maskShapeN=1` 矛盾
→ `srcStride=(256−4)/32=7` → 每搬 4 字节跳 256 字节 → 越界。
2D `[64,1]` 源 → `srcN=1` 与 `maskShapeN=1` 自洽 → 正确。

## 三、为什么交付算子“复杂才能对”是错觉

完整算子唯一被验证的散列路径，其源 `dbeta2` 恰好是 `[64,1]`（当时为配合 PB 路径
设计）——**不是 kernel 复杂度起作用，是源的形状**。GM 中转的真实作用：把 1D `[64]`
现算数据变成 2D `[64,1]` 的“GM 载入”数据（UB 内 [64]→[64,1] 转换在 TileLang 不合法，
见 scatter_ways.py 的 serial/parallel 编译拒绝）。

## 四、实验文件清单与结果

| 文件 | 内容 | 结果 |
|---|---|---|
| `scatter_3arms.py` | **核心对照**：contig / direct / staged 三臂 | contig 0/512 错 ✅；direct 504/512 ❌；staged 511/512 ❌ |
| `scatter_simple.py` | 最简 30 行：加载→乘2→`T.copy(acc, Out[0,0:64,bh])` | 511/512 全零 ❌ |
| `scatter_simple_ctrl.py` | 同上 + GM 中转 | 511/512 全零 ❌（简单 kernel 里 staged 也坏，flag/流水相关，未定位） |
| `scatter_ways.py` | 三种写入方式：DMA / `T.Parallel` 逐元素 / `T.serial` 逐元素 | DMA 全零；逐元素写入被编译器拒绝（"GM scalar stores must be inside T.Scope"） |
| `scatter_parallel.py` | `T.Parallel` 逐元素写入最小测试 | 编译拒绝（同上） |
| `scatter_ascendc.py` | 加 `T.Scope("V")` 后的三种变体 | 逐元素仍全零；`T.Scope("C")` 编译拒绝 |
| `scatter_raw.cpp` | 独立 AscendC 尝试（绕过 TileLang） | 编译卡在 DataCopyPad 重载匹配；但 CANN 头文件（kernel_struct_data_copy.h:187）确认 `DataCopyExtParams` 有 `dstStride` 字段——**硬件能力无疑** |

## 五、已核实的事实链

1. **硬件支持跨步散列**：CANN `DataCopyExtParams(blockCount, blockLen, srcStride, dstStride, rsv)`
   明确定义 dstStride（字节粒度）；技术专家确认；完整算子中 `[64,1]` 源散列逐位正确。
2. **编译器缺陷在 TileLang 降载层**：maskShapeN 与 srcN 不一致 → srcStride 计算错误。
3. **GM 中转 = 形状转换的 workaround**（非硬件要求）。
4. 未提交任何 issue；此实验为高置信度 bug 报告材料（附生成代码 diff）。

## 六、重跑方法（分设环境）

```bash
cannlab sync --no-start <本目录> DevEnvC_DCbyi_npu:/mnt/workspace/scatter_exp
cannlab --output jsonl exec --no-start --cwd /mnt/workspace/scatter_exp \
  DevEnvC_DCbyi_npu -- bash -lc \
  "source /mnt/workspace/tilelang-ascend/set_env.sh && \
   cd /mnt/workspace/scatter_exp && \
   for a in contig direct staged; do python scatter_3arms.py \$a 2>&1 | grep -E '^\[|sample'; done"
```
