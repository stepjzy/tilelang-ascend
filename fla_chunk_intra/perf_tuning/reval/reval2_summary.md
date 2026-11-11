# fla_chunk_intra 基线更新重验证（2026-11-07，reval2：#1948 host 分发杠杆试验）

**目的**：验证上游 #1948（Cython wrapper 懒构造 TVM Analyzer，静态 shape 全调用开销
442-467→100-102us）能否消除 reval 发现的新环境 host 分发地板（1450us/链调用 >
bwd 设备工作量 793us，逐 rep 暴露）。

**变更**：NPU 环境仓库基线 463c234c → **4805777d**（fast-forward，13 提交，含 #1948；
不含子模块改动）+ install_ascend.sh 重建（exit 0，TILELANG_OK）。**算子代码零改动**
（md5 与交付基线一致：ef253da1）。PR 分支已同步至同基线（551b85cd，交付文件未动，
ruff 过，已 push origin）。

## 结果（官方 do_bench 口径，4 次运行带）

| run | fwd (us) | bwd (us) | 合计 (us) |
|---|---|---|---|
| 1 | 10219.7 | 1222.1 | 11441.9 |
| 2 | 10335.4 | 1007.8 | 11343.2 |
| 3 | 10235.0 | 1004.3 | 11239.2 |
| 4 | 10359.2 | 988.6 | 11347.8 |
| **代表值** | **10287.3** | **1055.7** | **11343.0** |

- 对照旧基线（463c234c，无 #1948）同环境：合计 12105.3 → **11343.0（-6.3%）**；
  bwd 链 1676.1 → 1055.7（**-37%**）
- 对照旧环境（x86 host）do_bench 合计 11442.8：**11343.0（-0.9%，持平）**——两环境
  结论一致，跨环境可比性恢复
- L0 精度套件：**Test Passed!**（`&&` 链继续 = exit 0；reval1 已全量 90/90）

## gap 探针复测（host 地板归因验证）

| 探针 | 旧基线 | 新基线 | 含义 |
|---|---|---|---|
| A1 stage_alone（do_bench 单核） | 755.4 | **417.4 us** | host 提交 ~200us + 设备 221 |
| A2 gemm_alone（do_bench 单核） | 954.0 | **653.8 us** | host 提交 ~80us + 设备 572 |
| B chain（do_bench） | 1755.7 | **971.2 us** | 与 bench 带一致 |
| **D host_launch_per_call（wall）** | **1450.2** | **700.0 us** | **host 分发地板 -51.7%**（#1948 兑现） |
| E workspace_alloc_set | 70.8 | 74.4 us | 分配器依旧可忽略 |

**结论**：host 速率（700us）< bwd 设备工作量（793us）→ **pipelining 恢复**，bwd
do_bench 链回到设备主导（~1000-1220us ≈ 设备 793us + 残余小间隙）；fwd 本就
device-bound（-1.4% 噪声）。**新环境 do_bench 与旧环境持平；主口径（纯 kernel
11188us，msprof Task Duration）不变——#1948 只影响 host 侧，不影响设备侧 kernel
时间**（本次未重采 msprof，设备侧无相关基线变更且代码零改动）。

## 纯 kernel 主口径不变说明

验收判定依据维持 reval1 的 msprof Task Duration：fwd 10395.1 + bwd 793.0 =
11188.1us（agreed 验收带 ≈11.5ms 内）。本次基线变更只作用于 Cython wrapper
（host 侧）；13 提交中无 planner/CombineCV 相关变更（drain mask 阻塞项仍维持
阻塞性结论），无设备侧 kernel 语义变更。

## 产物

`reval2_workblock.log`（本文件同目录：L0 + bench×4 + gap 探针完整输出）。
