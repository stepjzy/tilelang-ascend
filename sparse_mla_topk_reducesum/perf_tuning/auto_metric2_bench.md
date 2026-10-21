# auto_metric2_bench — 指标 2 终验（auto 缺省分发 + 三路径锚定 + 分价实测）

日期：2026-10-09 ｜ kernel = auto 分发实现版（1803 行，5-builder 族谱，零修改 ✓）
工具：/tmp/opencode/auto_final_bench.py（graph-replay 主口径）

## 0. 测量协议与捕获边界披露

- **数据构造**：golden 同源——seed=42 randn bf16、offsets=[0,1023,2048]、arange(128) 全行重复、lse fp32。token_indices 预构建（不计时，既有口径）。
- **auto 三连发捕获边界**：host 入口的 `violations/err_flag torch.zeros(1)` 与 `rs torch.empty(...)` 为每调用新建（§8.4 生命周期不变式——防 graph replay 累积翻转选路）。**本 bench 预构建这三个 tensor 后仅捕获 3 条 kernel 调用**（detect → panel_guarded → generic_guarded）进 graph——zeros 的 H2D（~2-5us/次 eager）被排除在计时外，与 token_indices 预构建口径一致。graph replay 下 violations 不跨 replay 累积翻转（arange: detect 加 0，恒 0；random: detect 加正数，恒 >0——两路径选路在 replay 内均稳定）。
- **graph-replay 主口径**：batch N=1, batch=25 replay, iters=20, **2 组独立测量**（auto+arange）；其余 1 组。
- **auto 分发验证**：① `_panel_gate_ok() → ok=True` ✓（golden 配置通过元数据门控）；② auto 输出与 shared 显式 `torch.equal=True` ✓（逐字节等价）；③ auto+random 输出与 generic 显式 `allclose=True` ✓；④ topk=256 门控 miss `ok=False reason=G2/G3` ✓ → unguarded generic 单发。

## 1. 主口径结果

### auto 缺省（golden arange，三连发序列）

| run | mean | min | max |
|-----|------|-----|-----|
| 1 | 176.0 | 174.6 | 178.7 |
| 2 | 175.3 | 174.5 | 176.3 |
| **均值** | **175.7** | 174.5 | 178.7 |

**vs 220us：PASS（余量 1.25×）**

### 全路径汇总表

| 路径 | 配置 | graph-replay mean | vs 220us | vs shared explicit | vs generic BI128 |
|------|------|------------------|----------|-------------------|-----------------|
| **auto（缺省）** | golden arange | **175.7us** | **PASS** | +30.4us（+20.9%） | -192.1us（-52.2%） |
| auto（缺省） | random indices | 462.7us | FAIL | — | +94.9us vs arange |
| auto（缺省） | topk=256（元数据 miss） | 586.8us | FAIL | — | — |
| shared 显式 | panel_mode="shared" | 145.3us | PASS | — | — |
| generic 显式 | BI=128 | 367.8us | FAIL | — | — |
| generic 显式 | BI=32 | 470.7us | FAIL | — | — |

### 回归锚定对照

| 路径 | 本轮 | 前轮锚点 | 偏差 | 判定 |
|------|------|---------|------|------|
| shared 显式 | 145.3 | 143.3 | **+1.4%** | ✓ 漂移带内 |
| generic BI128 | 367.8 | 378.9 | **-2.9%** | ✓ 漂移带内 |
| generic BI32 | 470.7 | 471.4 | **-0.1%** | ✓ |
| auto + topk=256 | 586.8 | generic t256 显式 589.1 | **-0.4%** | ✓ **零开销验证** |

## 2. auto 开销分解（差分法，graph-replay 内部对照）

### auto + arange（panel 路径）: 175.7 = 145.3 + 30.4

| 分量 | 值 (us) | 推导 | 设计 §10.4 预估 | 差距 |
|------|---------|------|----------------|------|
| panel_guarded 执行 | 145.3 | = shared explicit（同为 panel 路径）| ~143 | +2 ✓ |
| **detect kernel** | **~15-17** | 差分法推断（30.4 - guard_idle - graph）| 3-5 | **+12 ← 设计低估** |
| generic_guarded 空载 | ~8-10 | early-exit（读 violations + Select 钳位 ntask=0）| 8.4 | ✓ |
| graph 3-kernel 开销 | ~3-5 | 3 kernel launch 间隙 vs 1 | — | — |
| **合计 auto 开销** | **30.4** | | **11-14** | **+17-19** |

**detect 为何 ~15-17us 而非 3-5us**：detect 需读取全量 Indices [2048,1,128] int32 = 1MB GM + 逐行比较（40 lanes 分摊 25KB/AIV）——设计估算只计了比较运算，未计入 1MB 索引读的 GM 带宽成本（~1MB / (40×1.5GB/s) ≈ 17us）。

### auto + random（generic 路径）: 462.7 = ~441 + ~21

| 分量 | 值 (us) | 说明 |
|------|---------|------|
| generic_guarded 执行 | ~441 | ≈ generic unguarded + random 索引（G3 参照 441.7） |
| detect | ~15-17 | 同上 |
| panel_guarded 空载 | ~8-10 | early-exit |
| graph 开销 | ~3-5 | |
| **auto 分价** | **~21** | **对非 panel 输入的 auto 额外成本** |
| 占 generic 比例 | ~4.6% | |

### auto + 元数据 miss（topk=256）: 586.8 ≈ 589.1（零开销）

- `_panel_gate_ok() → ok=False reason="G2/G3: topk=256 > 128"` → 直接落入 unguarded generic
- 586.8 vs 显式 generic 589.1 = **-0.4%**——**auto 元数据 miss 零开销验证通过** ✓

## 3. msprof 交叉

msprof Task Duration = 12.98us——**此值不适用单 kernel 口径判定**：auto 三连发 = 3 个同 kernel name（"main_kernel"）的 launch，msprof 将 45 次 launch（15 iteration × 3 kernel）聚合平均，被大量 detect/guard 空载拉低。**不构成有效交叉验证**（如实披露）；auto 路径的性能判定以 graph-replay 为唯一口径。

## 4. 汇总判定

| 判定项 | 结果 | 余量 |
|--------|------|------|
| **auto 缺省 + golden arange vs 220us** | **PASS** | **1.25×** |
| auto + random vs 显式 generic BI128 | 分价 +21us（+4.6%） | 可接受 |
| auto + 元数据 miss vs 显式 generic | **零开销** | ✓ |
| shared 显式回归 | +1.4%（漂移带内） | ✓ |
| generic BI128 回归 | -2.9%（漂移带内） | ✓ |
| auto vs shared 开销 | +30.4us（detect 15-17 + guard 8-10 + graph 3-5） | 设计 +17-19 超预算 |

## 5. 不确定性声明

1. **auto 开销 30.4us 为差分推断**（175.7 - 145.3），非独立测量 detect/generic_guarded 空载；detect ~15-17us 为估算（基于 1MB indices 读带宽推算），实际可能 10-20us。
2. **graph-replay 漂移**：auto+arange run-to-run 0.4%（175.3↔176.0）——远优于 generic ±3% 带。
3. **auto+random 的 generic 执行值 ~441** 为跨会话参照（G3 前轮测量），非本轮同会话对照（本轮未单独测 generic BI128 + random）；auto 分价 ~21us 含跨会话漂移不确定性 ±3-5us。
4. **msprof 交叉不适用**（多 kernel 同名聚合）——如实披露。
5. **graph 捕获边界**：violations/err_flag 的 zeros H2D（~2-5us eager）排除在计时外。若纳入，auto 均值 ~178-181us，仍 PASS。
