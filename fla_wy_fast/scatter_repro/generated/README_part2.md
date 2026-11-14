# Staged 链失败定位（第二部分）：为什么简单 kernel 里 staged 也全零

> 日期：2026-11-15 · 910B3 / CANN 9.2.0 / tilelang @ 4805777d
> 状态：定位到"缺同步"这一层；精确机理待一步验证实验确认

## 一、问题

交付算子（fla_wy_fast bwd v2）里 staged 链逐位正确；但把同样的四步链
（写仓库→flag→视图读回→散列）放进简单 kernel 却 511/512 全零。为什么？

## 二、二分定位实验（scatter_bisect.py / scatter_bisect2.py）

| 臂 | 结构 | 结果 |
|---|---|---|
| s1 / n1 | 写仓库 + **同参数名**读回 `Scratch[cid,0:64]`（[64] 连续） | ✅ 0/512 错 |
| s2 / n2 | 写仓库 + **视图**读回 `ScratchV[cid,0:64,0:1]`（[64,1]），无 flag | ❌ 511/512 全零 |
| s3 / n3 | 同上 + 手动 flag | ❌ 511/512 全零 |

结论：坏的不是散列、不是 flag，而是**经 [GRID,64,1] 视图读 [64,1] 的搬运
在简单 kernel 里读到全零**。且手动 flag 在此无效（n2=n3）。

## 三、与交付算子生成代码的对比（gen_v2_real.cpp vs gen_n3.cpp）

交付算子 v2 尾部（能工作）多出两组**编译器自动插入**的同步：

```cpp
// 写仓库之前：
AscendC::SetFlag<HardEvent::V_MTE3>(7);   // V 算完 → 才允许 MTE3 写
AscendC::WaitFlag<HardEvent::V_MTE3>(7);
copy_ub_to_gm<float,64>(dbeta_flat[cid*64], acc[0], ...);   // ① 写仓库
// 手动 flag（我们加的，两者都有）：
SetFlag<MTE3_MTE2>(0); WaitFlag<MTE3_MTE2>(0);
copy_gm_to_ub<float,1,64>(dbeta2[0], dbeta_flat_v[cid*64], ...);  // ③ 视图读回
// 读回之后：
AscendC::SetFlag<HardEvent::MTE2_MTE3>(0);  // 读完 → 才允许 MTE3 写
AscendC::WaitFlag<HardEvent::MTE2_MTE3>(0);
copy_ub_to_gm<float,1,64>(dbeta[...], dbeta2[0], 8, 64, 1);      // ④ 散列
```

简单 kernel（坏）只有手动 MTE3_MTE2，**没有 V_MTE3 也没有 MTE2_MTE3**。
另有 CrossCoreWaitFlag 等跨核握手仅存在于交付算子（其 Pipelined 结构触发）。

## 四、最可能的机理（待一步验证）

手动 MTE3_MTE2 只保证"写仓库 vs 读回"的顺序；但**保证不了"V 核计算完成 vs
MTE3 写仓库"的顺序**。简单 kernel 里 MTE3 可能在 V 核还没算完 acc 时就把
旧值（初始零）搬进仓库 → 读回自然是零 → 全零输出。
这解释 n2/n3 结果相同（flag 管的是写↔读，坏在更前面的算↔写）。

**验证实验（待做）**：在简单 kernel 的写仓库前手动补
`T.set_flag("V","MTE3",7) + T.wait_flag("V","MTE3",7)`，若翻转 PASS 即证实。

## 五、与此前根因（第一部分）的关系

- 第一部分（scatter_repro/README.md）：**散列步的降载 bug**（srcN vs
  maskShapeN 矛盾 → srcStride=7）——direct 臂的失败原因，已钉死；
- 本部分：**staged 链在简单 kernel 里的额外失败**——同步缺失，与散列 bug
  无关，是第二层问题。交付算子的正确性不受影响（其 C/V 结构自动带全同步链）。

## 六、归档文件

| 文件 | 内容 |
|---|---|
| `gen_s1.cpp / gen_s2.cpp` | 二分定位 s1（PASS）/ s2（FAIL）的生成代码 |
| `gen_n3.cpp` | 视图读回 + flag 仍 FAIL 的生成代码 |
| `gen_v2_real.cpp` | **交付算子 v2 真实生成代码**（同步链对比基准） |
| `scatter_bisect.py / scatter_bisect2.py` | 定位实验脚本（随附在上级目录） |
