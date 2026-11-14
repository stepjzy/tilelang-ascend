# Generated（Lowering）代码学习材料：scatter 三臂

> 来源：2026-11-14 实测（910B3 / CANN 9.2.0 / tilelang @ 4805777d）。三份文件分别是
> `scatter_3arms.py` 三个臂编译后留在 NPU `/tmp` 的 C++ 代码（TileLang 生成的
> "翻译结果"），已按臂命名取回存档。

## 一、这三个文件是什么

TileLang 编译分两步：Python DSL → **C++ 源码（本目录三份文件）** → bisheng 编译成 .so。
本目录三份 `.cpp` 就是中间那步的产物——**看它们就知道编译器把你的 `T.copy` 翻译成了什么**。

| 文件 | 对应实验臂 | 写入方式 | 实测结果 |
|---|---|---|---|
| `gen_contig.cpp` | contig | 目的地连续 | ✅ 0/512 错 |
| `gen_direct.cpp` | direct | 目的地跨步，源 `[64]` | ❌ 504/512 错 |
| `gen_staged.cpp` | staged | GM 中转后跨步，源 `[64,1]` | ❌ 511/512 错（简单 kernel） |

## 二、核心行对比（一眼看出问题）

三个 kernel 唯一的实质差异，就在各自的搬运调用：

```cpp
// contig（✅ PASS）—— 连续写
tl::ascend::copy_ub_to_gm<float,  64>(OutFlat[(cid * 64)], acc[0],     64, 1,  64);
//                                  ↑模板 srcN=64                       ↑realdstN ↑shapeM ↑shapeN

// direct（❌ FAIL）—— 跨步写，源还是 [64]
tl::ascend::copy_ub_to_gm<float,  64>(Out[cid],             acc[0],     8, 64, 1);
//                                  ↑模板 srcN 还是 64!      ↑realdstN=8（跨步）→ shapeN 被压成 1

// staged（完整算子里 ✅）—— 跨步写，但源换成了 [64,1]
tl::ascend::copy_ub_to_gm<float, 1, 64>(Out[cid],             buf[0],     8, 64, 1);
//                                  ↑模板 srcN=1              ↑源形状[64,1]，自洽
```

具体的 staging 四步（`gen_staged.cpp` 完整链路）：

```cpp
// ① 现算 acc[64] 连续写进 GM 仓库（这步和 contig 一模一样，是正确的）
tl::ascend::copy_ub_to_gm<float, 64>(Scratch[(cid * 64)], acc[0], 64, 1, 64);
// ②③ 同步 + 从仓库的 [64,1] 视图读回 buf（GM→UB，源形状发生了"洗牌"）
AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
tl::ascend::copy_gm_to_ub<float, 1, 64>(buf[0], ScratchV[(cid * 64)], 1, 64, 1, 0.000000e+00f);
// ④ 从 buf[64,1] 跨步散列（模板 srcN=1，正确！）
tl::ascend::copy_ub_to_gm<float, 1, 64>(Out[cid], buf[0], 8, 64, 1);
```

## 三、逐参数拆解（学习重点）

`copy_ub_to_gm<数据类型, srcN, srcM>(目的地, 源, realdstN, maskShapeM, maskShapeN)`

| 参数 | 含义 | contig | direct | staged(第④步) |
|---|---|---|---|---|
| 模板 `srcN` | 源缓冲区**最后一维** | 64（acc[64]） | 64（acc[64]） | **1**（buf[64,1]） |
| `realdstN` | 目的地行宽（元素数） | 64 | **8**（H，跨步！） | 8 |
| `maskShapeM` | 分几行搬 | 1 | 64 | 64 |
| `maskShapeN` | 每行搬几个元素 | 64 | **1** | 1 |

**关键观察**：目的地跨步（realdstN=8）时，编译器把搬运拆成"64 行 × 每行 1 元素"
（maskShapeM=64, maskShapeN=1）。此时模板里 srcN 本该也变成 1（源要按"每行 1 元素"
读取），但 **direct 臂的模板 srcN 还是抄了源缓冲区最后一维 64** ——矛盾出现了。

## 四、错误的数学推导（根因）

模板内部公式（`src/tl_templates/ascend/common.h` `copy_ub_to_gm`）：

```
srcPitch  = srcN 元素 × 4B      ← 源"行宽"（用模板 srcN 算！）
blockLen  = maskShapeN × 4B     ← 每行实际搬的字节
srcStride = (srcPitch − blockLen) / 32   ← 每搬一行后，源址跳多少"32B 块"
```

代入三臂：

| 臂 | srcPitch | blockLen | srcStride | 后果 |
|---|---|---|---|---|
| contig | 64×4=256B | 64×4=256B | (256−256)/32=**0** | ✅ 源址连续步进，正确 |
| direct | 64×4=256B | 1×4=4B | (256−4)/32=**7** | ❌ 每搬 4 字节，源址往后跳 **256 字节** → 只读对第 1 个元素，其余 63 个越界读垃圾 |
| staged ④ | 1×4=4B | 1×4=4B | (4−4)/32=**0** | ✅ 自洽，正确 |

direct 的错误形象化：**你以为在连续剥 64 粒玉米，编译器却在每剥 1 粒后把玉米棒
转 360°——只有第 1 粒是正常的，后面全是空转。**

## 五、结论链

1. **硬件没问题**：staged 第④步和完整算子里的散列都逐位正确，`DataCopyPad` 的
   `dstStride` 参数（字节粒度）就是为此设计的；
2. **编译器 bug 在降载层**：跨步目的地触发"64×1 拆行"时，模板 `srcN` 没有同步
   改为 1，仍取源缓冲区最后一维 → `srcStride` 算错；
3. **GM 中转的真实作用**：把 1D `[64]` 现算数据洗成 2D `[64,1]` 的"GM 载入"数据——
   因为 UB 内 [64]→[64,1] 的形状转换在 TileLang 里不合法，只能绕道 GM；
4. **"复杂 kernel 才能对"是错觉**：与 kernel 复杂度无关，只与散列源的形状
   `[64]` vs `[64,1]` 有关。

## 六、自己复现抓取（分设环境）

```bash
# 1. 清空旧的生成代码（干净对比）
cannlab exec --no-start --cwd /mnt/workspace DevEnvC_DCbyi_npu \
  -- bash -lc "rm -f /tmp/tmp*.cpp /tmp/tmp*.so"

# 2. 跑一臂，立刻抓最新生成的 cpp
for a in contig direct staged; do
  echo "===== $a ====="
  cannlab exec --no-start --cwd /mnt/workspace/scatter_exp DevEnvC_DCbyi_npu \
    -- bash -lc "source /mnt/workspace/tilelang-ascend/set_env.sh && \
                 cd /mnt/workspace/scatter_exp && python scatter_3arms.py $a" 2>&1 \
    | grep -o '"[a-z]*\] wrong[^"]*'
  cannlab exec --no-start --cwd /mnt/workspace DevEnvC_DCbyi_npu \
    -- bash -lc "ls -t /tmp/tmp*.cpp | head -1 | xargs grep copy_ub_to_gm"
done
```