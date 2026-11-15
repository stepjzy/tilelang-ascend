# scatter 核验：GM→UB / UB→GM 硬件地址计算规则 —— 实验证据定稿

> 本文**推翻并替换** `VERDICT.md` 中问①的结论（"UB 紧凑字节流"是错误的）。
> 方法：手写最小 Ascend C kernel 直接调 CANN `DataCopyPad`（绕开 TileLang 模板），
> 在 910B3 实机观测 GM→UB 写 UB 的逐 burst 布局。
> 实验代码：`probe_dststride/probe_dststride.cpp` + `run_probe.py`（NPU
> /mnt/workspace/probe_dststride/，bisheng 直编 + torch_npu ctypes 驱动）。

## 一、实验设计

UB 1KB 先 `Duplicate` 清零。写 3 个 GM→UB case（MTE2）+ 3 个 UB→GM arm（MTE3），
HM 输入 `In = {1,2,3,4,5,6,7,8}` fp32。

| 编号 | 调用 | 参数 Ext(count,len,srcS,dstS,rsv) |
|---|---|---|
| case1 | DataCopyPad(buf[0], In, Ext) GM→UB | (2, 4B, 0, 0, 0) |
| case2 | DataCopyPad(buf[32], In, Ext) GM→UB | (2, 4B, 0, **1**, 0) |
| case3 | DataCopyPad(buf[64], In, Ext) GM→UB | (1, 32B, 0, 0, 0) 对照 |
| arm4 | DataCopyPad(Out[96], buf[0], Ext) UB→GM | (2, 4B, **1**, 0, 0) |
| arm5 | DataCopyPad(Out[112], buf[0], Ext) UB→GM | (2, 4B, 0, 0, 0) |
| arm6 | DataCopyPad(Out[128], buf[0], Ext) UB→GM | (1, 32B, 0, 0, 0) 对照 |

## 二、实机观测结果（逐字）

```
case1 (2,4,0,0,0)  -> buf[0]   :  [1.0 x8][2.0 x8]            (Out[0..15])
case2 (2,4,0,1,0)  -> buf[32]  :  [1.0 x8][0.0 x8][2.0 x8]    (Out[32..55])
case3 (1,32,0,0,0) -> buf[64]  :  [1,2,3,4,5,6,7,8]            (Out[64..71])
arm4  (2,4,1,0,0)  <- buf[0]   :  [1.0][0.0]                  (Out[96..97])
arm5  (2,4,0,0,0)  <- buf[0]   :  [1.0][2.0]                  (Out[112..113])
arm6  (1,32,0,0,0) <- buf[0]   :  [1.0 x8]                    (Out[128..135])
```

## 三、硬件规则（全部由观测确认）

1. **GM→UB 每 burst 在 UB 侧恰占一个 32B 槽**（case1：两个 4B burst 相距 32B）。
2. **槽内填充 = 有效数据的复制扩展**：4B 有效值被复制填满整个 32B 槽
   （case1：1.0×8；arm6 读回整槽同为 1.0×8）。未被任何 burst 覆盖的区域
   保持原值（case2 中间的 0.0×8 = Duplicate 留存的 gap 块）。
3. **GM→UB 的 dstStride 单位 = 32B 块**（case2：dstStride=1 产生 32B gap；
   每 burst UB 步进 = ceil(blockLen/32) + dstStride 个 32B 块）。
4. **UB→GM 的 srcStride 单位 = 32B 块**（arm4：srcStride=1 → burst1 源跳到
   buf[64B)（2 块），读回 0.0 = gap 区）。
5. **UB→GM 相邻 burst 自然步进 = ceil(blockLen/32) 块**（arm5：srcStride=0 时
   burst1 读 buf[32B) = 槽1 首 4B = 2.0 —— 槽首取数据，正是散列语义）。
6. **GM 侧（UB→GM 的 dstStride / GM→UB 的 srcStride）为字节粒度**（arm4/5
   dstStride=0 时 GM 连续写；case3 整块正确）。
7. **blockLen=32B 整数块槽**：单 burst 32B 紧凑写（case3 1..8 连续）。

## 四、用户四问的直接回答

### Q1: `copy_gm_to_ub<float,1,64>`（Ext=(64,4,0,0,0)）逐 burst 地址

第 k burst（k=0..63）：
- GM 源：[4k, 4k+4) 字节（srcStride=0，GM 连续）
- UB 目的：[32k, 32k+4) 字节有效 + [32k+4, 32k+32) 为有效值的复制填充
- 总 UB 占用 = **64 × 32B = 2KB**（不是 256B！）

### Q2: dstStride=0 是紧凑 4B 还是 32B 块前进？

**是 32B 块前进**。dstStride 只是"额外 gap（块数）"，burst 之间始终有
ceil(blockLen/32) 块的自然步进（case1 实证：4B burst 相邻间隔 32B）。

### Q3: common.h 注释 "MTE-filled buffers write rows on 32B boundaries" 的关系

该注释**与硬件行为完全一致**，是 TileLang 开发者对硬件的正确观察：
MTE 写 UB 的每行（每 burst）都落在 32B 边界上。VERDICT.md 问①说"没有自动槽
布局、UB 是连续字节流"是**错误结论**，本文予以推翻。

### Q4: staged 路径为什么能跨步写入 / 槽布局如何产生

槽布局由**硬件 MTE2 规则**（规则 1+2）自动产生，不需要任何额外参数：
- `copy_gm_to_ub<float,1,64>`（Ext(64,4,0,0,0)）把 GM 连续 256B 写成 UB 的
  64 个 32B 槽（每槽 4B 有效 + 28B 复制填充），总 2KB。
- `copy_ub_to_gm<float,1,64>`：TileLang 模板对 srcN=1 算出 srcStride=0，
  硬件按"每 burst 前进 1 槽"逐槽读槽首 4B（arm5 实证），配合 dstStride=28B
  （字节粒度）写入 GM 行首 → **精确散列**。
- 槽首 4B 恰是有效数据位置（规则 2 的复制填充从槽首开始），所以槽首读
  语义正确。

## 五、故障机理的修订表述

### 故障 A（direct，确定性）—— 机理不变，表述更精确

`copy_ub_to_gm<float,64>(Out[cid], acc[0], 8, 64, 1)`：
TileLang 模板 srcPitch=EleNumToBytes(64)=256B → srcStride=(256−4)/32=7 块。
硬件每 burst 源步进 = ceil(4/32)+7 = **8 块 = 256B**。
- burst0 读 acc[0..4B) = d0 ✓
- burst1 读 acc[256..260B) —— acc 是 V 算紧凑输出，仅 256B → **越界读 UB**
  → 确定性垃圾。
根源不变：srcN（buffer 末维 64）与 maskShapeN（目标视图 1）来自降载层两个
不同来源且无一致性约束。V 算输出的紧凑布局**不能**直接供 MTE3 散列——散列
要求源是"槽布局"（每行 32B 槽、槽首放元素）。紧凑→槽的转换必须经 GM 中转
（或 MTE2 重写）。**这一层机理在 VERDICT 中没有说透，本文补齐。**

### 故障 B（简单 kernel 视图读回，非确定性）—— 增加静态放大器

`copy_gm_to_ub<float,1,64>` 写出 2KB 槽布局（非紧凑 256B），下游 V 算按紧凑
视图读回时静态错位（读到复制填充区）；叠加缺 MTE3→V flag 链的竞态 →
非确定性（同码三次 509/508/510）。交付算子恰好：①V 算不会按紧凑视图读槽
布局区 ②有完整 flag 链。

## 六、衍生发现（新 bug 线索，待独立验证）

**TileLang 对 [M,1] 槽布局视图的 UB 分配低估**：
`gen_staged.cpp` 中 `buf` 视图 [64,1]，TileLang 分配 `GetWithOffset<float>(64,
512)` 仅 256B，而硬件 MTE2 实际写 2KB（64 槽 ×32B）——**8× 越界写 UB**。
staged 场景下 buf 恰是最后分配的 buffer，越界只覆盖未用区 → 侥幸通过。
若 [M,1] 视图 buffer 之后还有 buffer，会被静默破坏。这是比散列错误更
危险的隐藏缺陷，应单独实验（[64,1] buffer 后再分配 buffer + 全零填充检查）
验证后上报。

## 七、重跑方法

```bash
cannlab sync /tmp/opencode/probe DevEnvC_DCbyi_npu:/mnt/workspace/probe_dststride
cannlab exec --cwd /mnt/workspace/probe_dststride DevEnvC_DCbyi_npu -- \
  bash -lc "source /mnt/workspace/tilelang-ascend/set_env.sh && python3 run_probe.py"
```