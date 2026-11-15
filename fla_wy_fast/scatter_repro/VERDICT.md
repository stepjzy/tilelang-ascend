# scatter 降载 bug 核验三问——最终定稿

> 目的：对 scatter 相关两起故障（① direct 散列确定性错误 ② 简单 kernel 视图读回非确定性全零）
> 做**源码级**核验，不凭模板注释推断。证据文件在本目录 `cann_src/` 下。
> 参考：`../README.md`（三臂实验与机理初析）、`generated/README_part2.md`（视图读回二分）。

## 核验三问与最终答案

### 问①：`copy_gm_to_ub<float,1,64>` 如何写 UB——32B 槽布局还是紧凑字节？

**答案：没有"自动槽布局"，UB 侧就是连续字节流；唯一 32B 约束是起点对齐。**

证据链（CANN 源码，`cann_src/c220_datacopy_440_640.txt` = dav_c220/kernel_operator_data_copy_impl.h:440-640）：

1. `DataCopyPadGm2UBImpl`（Ext 版，~L502）：开头
   `ASCENDC_DEBUG_ASSERT(TransUBAddr<VECIN>(dst) % ONE_BLK_SIZE == 0)` ——
   dst UB 起点必须 32B 对齐；按 dtype 分派 `copy_gm_to_ubuf_align_b32(...)`，
   对齐搬运单元 = 32B（`align_b32`）。
2. TileLang `copy_gm_to_ub`（`src/tl_templates/ascend/common.h` L404-408）把
   `maskShapeM/maskShapeN/realSrcN/dstN` 换算为
   `DataCopyExtParams(maskShapeM, EleNumToBytes(maskShapeN), EleNumToBytes(realSrcN-maskShapeN), EleNumToBytes(dstN-maskShapeN)/32, 0)`。
   对 `<float,1,64>`：blockCount=1、blockLen=256B、srcStride=0、dstStride=0 →
   **单 burst 连续写 256B**，就是 64 个 fp32 紧凑排列。
3. 结论：UB 布局由 dstStride 参数控制（0=紧凑连续），不存在 MTE 自动把行拆进
   32B 槽的行为。"槽"只是起点对齐 + 源侧块粒度步进两个约束的副作用。

### 问②：`T.tile.add` 是否紧凑写 256B？（历史结论，本轮复核无推翻）

**答案：是。** `generated/gen_n3.cpp`：acc 取 `GetWithOffset(acc[0], 64)` 连续 256B，
Add 指令连续覆盖。V 算输出是紧凑字节流，与问①的 MTE 写入语义一致——
即 UB 中"紧凑 256B"是唯一正确假设，任何按"64 行 × 32B 槽"的读法都是错的。

### 问③：MTE3（UB→GM DataCopyPad）能否表达 4B 源步进？

**答案：不能。源侧 stride 的最小非零粒度是 32B 块；4B 步进在硬件上不可表达。**

证据链：

1. **CANN 溢出检查层**（`cann_src/stride_unit.txt` = kernel_check_data_copy_util.h:342 起）：
   `GetStrideUnit` 注释 "get unit of dst src stride, unit is byte"；
   UB→GM 的 srcDstId 不在 `srcStrideUnitMap`（kernel_utils_mode_cpu.h:204-208，
   只有 L1L0C/L0CUB 等条目）→ 落默认 `DEFAULT_C0_SIZE`。
   而 `DEFAULT_C0_SIZE = 32B`（CANN softmax_grad_nd_impl.h:115
   `elementNumPerBlk = DEFAULT_C0_SIZE / B32_BYTE_SIZE` = 8 fp32 = 32B）。
   → **UB→GM srcStride 的换算单位 = 32 字节/单位**。
2. **TileLang 侧实现**（common.h L443）：
   `srcStride = (srcPitch - blockLen) / kBlockBytes`（kBlockBytes=32）——
   同样按 32B 块折算。模板注释（L211-215）："per burst the UB source advances
   ceil(blockLen / 32) + srcStride whole 32B blocks"。
3. **dav_c220 `DataCopyPadUB2GMImpl`**（Ext 版，L605）：srcStride（32-bit）无换算
   直传 `copy_ubuf_to_gm_align_b32` intrinsic（bisheng 内建，闭源）。
4. 推论：源侧每 burst 后前进 `ceil(blockLen/32)+srcStride` 个 32B 块。
   表达"每 4B 取一个元素"的唯一形态是 **blockLen=4B + srcStride=0 + blockCount=N**
   （紧凑连续流读）。真正的 4B 步进（stride=4B）不可能。

## 两起故障的最终机理

### 故障 A（direct 散列，确定性坏数据）

调用：`copy_ub_to_gm<float,64>(Out[cid], acc[0], 8, 64, 1)`（`generated/gen_direct.cpp:37`）。

TileLang 模板把 UB 源 buffer 的"行距"取为 `srcPitch = EleNumToBytes(srcN) = 256B`
（srcN=64 照抄 buffer 末维），而视图语义要求的行距是 `EleNumToBytes(maskShapeN) = 4B`。
于是 `srcStride = (256-4)/32 = 7`（32B 块）——硬件按"每行跳 7+1=8 个 32B 块 = 256B"
读源，第 2 burst 起即越过 256B 的 buffer 尾部，读 UB 未初始化区 → 确定性垃圾。

正确形态（staged 证明可行，`gen_staged.cpp:50`）：`copy_ub_to_gm<float,1,64>(..., 8, 64, 1)`
→ srcPitch=4B → srcStride=0 → 64×4B 紧凑流读。

**根因**：TileLang `AscendCopy::Lower`（src/op/ascend.cc）在生成 ub→gm 模板实参时，
对"源为 1D 紧凑 buffer 但被散列成 (M,1) 视图"的场景，srcN 与 maskShapeN 来自两个
不同来源（buffer 末维 vs 目标视图），模板未做一致性约束 → 步进算错。修复方向：
模板内做 srcN 语义归一（当 maskShapeM>1 且 maskShapeN=1 时行距取 maskShapeN），
或 Lower 时显式校验并改用紧凑流形式。

### 故障 B（简单 kernel 视图读回，非确定性全零/多数）

二分结论（`generated/README_part2.md`）：`copy_gm_to_ub<float,1,64>` 把紧凑 256B
重新读进 UB 后，下游 V 算按视图读回即错（同码三次 509/508/510 各不同）——
**非确定性**，属同步/流水竞态。交付算子（`gen_v2_real.cpp`）因 C/V 结构自动含
`SetFlag/WaitFlag<V_MTE3>` + `SetFlag/WaitFlag<MTE2_MTE3>` 链而稳定。

问①结论（256B 紧凑写入）与本故障兼容：写入本身紧凑正确，错在"读回时序"——
简单 kernel 缺 MTE3→V 的 flag 链，V 算在 MTE3 未完成时读 UB。

## 与硬件约定的最终对账

| 项 | 事实 | 出处 |
|---|---|---|
| UB 起点 32B 对齐 | 强制（断言） | dav_c220 L442/L505 |
| GM→UB blockLen 单位 | 字节（EleNumToBytes） | common.h L406 |
| UB→GM srcStride 单位 | 32B 块（默认 DEFAULT_C0_SIZE） | kernel_check_data_copy_util.h L342；common.h L443 |
| UB→GM dstStride 单位 | 字节 | common.h L444（dstPitch-blockLen 不除 32） |
| 4B 源步进可表达性 | 否（最小非零步进 32B） | 问③ |
| V 算输出布局 | 紧凑字节流 | gen_n3.cpp |

## 证据文件

- `cann_src/c220_datacopy_440_640.txt`：DataCopySlice/Pad Gm2UBImpl、Pad Ub2GMImpl
  （dav_c220 架构头 440-640 行，含 32B 对齐断言与 intrinsic 分派）
- `cann_src/intf_ub2gm_ext.txt`：DataCopyPad UB→GM Ext 重载
  （kernel_operator_data_copy_intf_impl.h 1620-1700）
- `cann_src/stride_unit.txt`：GetStrideUnit / stride 单位函数与默认值
- `generated/gen_direct.cpp` / `gen_staged.cpp` / `gen_n3.cpp` / `gen_v2_real.cpp`：
  四份生成代码对照
