# Scatter 实验三臂对照（bug A 验证）

## 目的
验证 TileLang 编译器对"现算源 + 跨步散列"是否生成错误地址（bug A），
以及 GM 中转是否是当前工具链上的必要绕路。

## 三臂说明

| 臂 | 文件 | 散列路径 | 预期结果 |
|---|---|---|---|
| control | `control/fla_wy_fast.py` | acc → GM 仓库 → flag → 读回 → 散列 | ✅ 全 PASS（交付版） |
| direct_A | `direct_A/fla_wy_fast.py` | acc → 直接散列（无 GM 中转，无 flag） | ❌ bwd 的 dbeta 输出全错 |
| direct_B | `direct_B/fla_wy_fast.py` | acc → UB 内 [64,1] 缓冲一跳 → 散列 | ❌ bwd 编译不过 |

每个目录都包含独立的 `fla_wy_fast.py` + `test_fla_wy_fast.py`，
互相不依赖，可以独立运行。

## 与交付版的 diff

只改了 v2 kernel 尾部散列链（5 行），其余代码完全一致：

```diff
# control（交付版原样）
  T.copy(acc, dbeta_flat[cid, 0:BS])
  T.set_flag("MTE3", "MTE2", 0)
  T.wait_flag("MTE3", "MTE2", 0)
  T.copy(dbeta_flat_v[cid, 0:BS, 0:1], dbeta2)
  T.copy(dbeta2, dbeta[bb, s0 : s0 + BS, bh])

# direct_A（去掉中转，直接散列）
- T.copy(acc, dbeta_flat[cid, 0:BS])
- T.set_flag("MTE3", "MTE2", 0)
- T.wait_flag("MTE3", "MTE2", 0)
- T.copy(dbeta_flat_v[cid, 0:BS, 0:1], dbeta2)
- T.copy(dbeta2, dbeta[bb, s0 : s0 + BS, bh])
+ T.copy(acc, dbeta[bb, s0 : s0 + BS, bh])

# direct_B（UB 内 [64,1] 一跳，无 GM 往返）
- T.copy(acc, dbeta_flat[cid, 0:BS])
- T.set_flag("MTE3", "MTE2", 0)
- T.wait_flag("MTE3", "MTE2", 0)
- T.copy(dbeta_flat_v[cid, 0:BS, 0:1], dbeta2)
- T.copy(dbeta2, dbeta[bb, s0 : s0 + BS, bh])
+ T.copy(acc, dbeta2)          # [64] → [64,1]，UB 内形状转换
+ T.copy(dbeta2, dbeta[bb, s0 : s0 + BS, bh])
```

## 实测结果（2026-11-14，910B3 / CANN 9.2.0 / tilelang @ 4805777d）

| 臂 | fwd | bwd | 退出码 |
|---|---|---|---|
| control | 全 PASS | 全 PASS | 0 |
| direct_A | 全 PASS | dbeta ratio=0.0156 ❌（其余 dA/dk2/dv/dg2 全 PASS） | 1 |
| direct_B | 全 PASS | 编译失败（DiagnosticError） | 1 |

关键观察：
- direct_A 的失败**精准隔离在 dbeta 一个输出**上——同一 kernel 的 dA/dk2/dv/dg2 
  四个输出全部正确，说明不是算子逻辑错，是散列地址错；
- direct_A 的 dbeta max_abs = 442~1231（数量级远超正常梯度范围），说明数据
  写进了完全错误的 batch/head 位置；
- direct_B 连编译都过不了——编译器要求散列源必须"真的来自 GM 加载"，
  UB 内形状转换不满足条件。

## 如何自己运行（分设环境）

```bash
# 1. 启动 NPU 环境
cannlab env start DevEnvC_DCbyi_npu
cannlab env wait --status RUNNING --timeout 10m DevEnvC_DCbyi_npu

# 2. 同步实验目录到 NPU
cannlab sync --no-start <本目录> DevEnvC_DCbyi_npu:/mnt/workspace/scatter_exp

# 3. 逐臂运行（在 NPU 上）
cannlab --output jsonl exec --no-start \
  --cwd /mnt/workspace/scatter_exp/control \
  DevEnvC_DCbyi_npu -- bash -lc \
  "source /mnt/workspace/tilelang-ascend/set_env.sh && cd /mnt/workspace/scatter_exp/control && python test_fla_wy_fast.py --level all"

# direct_A 同理，换目录名即可

# 4. 观察重点：bwd 用例的 dbeta 行
#    control 应看到  [PRECISION_PASS] l0_bwd_min :: ... dbeta:ratio=1.0000 ...
#    direct_A 应看到 [PRECISION_FAIL] l0_bwd_min :: ... dbeta:ratio=0.0156 ...
#    对比即可看到"同一个 kernel、同一份输入，散列源不同 → dbeta 对/错"

# 5. 用完后关机
cannlab env stop DevEnvC_DCbyi_npu
```
