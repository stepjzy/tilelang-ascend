# fla_wy_fast Phase 2R 调优日志 — r1: beta gather 链常量化

> 轮次：Phase 2R 第 1 轮（画像驱动单变量优化）
> 日期：2026-11-09 · 环境：远程 910B3（DevEnvC_DCbyi_npu，基线 463c234c）经 cannlab
> 改前快照：`history_version/fla_wy_fast_impl_prof_r1_before.py`（禁改）
> 上游画像：`perf_tuning/fla_wy_fast_prof_analysis.md`（baseline 轮 msprof 双口径）

## 1. 现象与画像证据（改前）

msprof OPPROF（baseline 轮，8 核逐采）判定：**AIV scalar pipe 是全部 8 个 kernel 的
最终瓶颈**。可复现口径（本文件统一使用）：

```
AIV scalar share = mean(aiv_scalar_time) / mean(aiv_time)   # PipeUtilization.csv 计时行
```

改前数值（bench 形状：fwd 1×8192×64×128×128 bf16 / bwd 1×32768×8×128×128 fp32）：

| 核 | TaskDur(us) | aiv_time/block(us) | aiv_scalar_time/block(us) | scalar share |
|---|---|---|---|---|
| u   | 1934 | 4.45 | 2.64 | 59% |
| wkg | ~2827* | 6.34 | 4.09 | 64% |
| k1  | ~1315* | 4.72 | 4.70 | 99% |
| k2  | ~1714* | 9.45 | 5.56 | 59% |

（* 时间线均值；scalar share 与 fla_wy_fast_prof_analysis.md §2 一致）

根因：每个 block 的 prologue 执行 beta gather 链（6 个消费 beta 的 kernel 同款：
u/wkg/k1/k2/v1/v2；z/da 不含 beta）：

```python
T.copy(BetaFlat[bb, s0*H : (s0+BS)*H], beta_tile)   # 整 chunk 全 H 列 span 载入
T.tile.arith_progression(off_i32, bh*dsz, H*dsz)     # 纯 scalar 构造字节偏移
T.reinterpretcast(off_u32, off_i32, "uint32_t")      # scalar
T.tile.gather(bt, beta_tile, off_u32, 0)             # scalar 重操作
```

- span 载入每 block 搬 BS×H 元素只用 BS 个（fwd H=64：64× 放大，8KB/block × 8192
  block ≈ 64MB/轮 fwd 的无效流量）；
- arith_progression + reinterpretcast + gather 三步全是 scalar 指令，逐 block 重复
  8192（fwd）/4096（bwd）次。

## 2. 改动方案（单变量：只动 beta 供给路径）

- **Host**：新增 `_pack_beta(beta, B, S, H, chunk_size)`，把 [B,S,H] beta 预排成
  `[B, S//64, H, 64]` 连续布局（`view → permute(0,1,3,2) → contiguous()`，一次
  ~1-2MB 的设备侧小转置，dtype 跟随 beta：fwd bf16 / bwd fp32）。对调用方透明，
  fwd/bwd 接口不变。
- **Kernel**（6 个 beta 消费核统一）：签名 `BetaFlat: (B, S*H)` →
  `BetaPacked: (B, nt, H, BS)`（参数位次不变，out_idx 不变）；prologue 的
  4 步 gather 链替换为一次连续 span 拷贝 `T.copy(BetaPacked[bb, bs, bh, :], bt)`；
  fwd 因 GM→UB bf16→fp32 cast-on-copy 损坏（docstring note 8）保留 bf16 staging
  （bt_bf）+ 显式 UB cast；bwd fp32 直载 bt。删除 beta_tile/off_i32/off_u32 三个
  alloc（fwd 每 block 省 8KB+256B shared，bwd 16KB+256B）。
- **4D 切片 [bb, bs, bh, :] 可靠性**：同文件既有在用先例（wkg 核
  `T.copy(G[bb, s0+BS-1, bh, slice], gn_ub)`，4D 张量 3 标量 + 切片 → 1D tile，
  交付版已验证 green）；且 64 元素 = 128B(bf16)/256B(fp32) ≥ 32B 粒度。直路先行，
  失败再退 2D 视图 [B*nt*H, 64] 按行载入。
- 对照反模式清单：本改动即 "离散 Gather → 预排连续块 + 普通 span 拷贝" 模式
  （optimization-guide.md §离散数据 Gather；antipatterns "逐行/逐元素 scalar 操作"）。
- 布局数学：BetaPacked[b,t,h,r] = beta[b, t*64+r, h] = 旧 gather 的
  beta_tile[r*H+h]，逐元素等价。

**改动量**：`fla_wy_fast.py` diff 118 行（6 kernel 签名/alloc/链 + 2 wrapper +
docstring note 4 与模式声明）；ruff 全过。

**[SELF-CHECK]** 本轮只涉及 gather 常量化，未动任何其他逻辑（gemm/同步/scatter
链全部原样）。

## 3. 验证流程与结果

（执行中，边做边填）

### 3.1 精度门禁

- 命令：`python test_fla_wy_fast.py --level l0` → `--level all`
- 远端沙箱：`/mnt/workspace/tilelang-ascend/examples_experiment/dsa_r1_gather/`
- Job：fla-r1-gather-gate（exit 0，643s）

**全绿**。逐用例（ratio / max_abs）：

| 用例 | 结果 | 关键数字 |
|---|---|---|
| l0_fwd_min (1,128,2,64,64) | PASS | w/u/kg ratio=1.0000 |
| l0_fwd_repr (2,256,4,128,128) | PASS | ratio=1.0000（kg max_abs=0） |
| l0_bwd_min | PASS | dA ratio=0.9999, 其余 1.0000 |
| l0_bwd_repr | PASS | 全 1.0000 |
| l1_fwd_bench (1,8192,64,128,128) | PASS | ratio=1.0000, w max_abs=5e-1（bf16 门内） |
| l1_fwd_d96 / l1_bwd_d96 | PASS | ratio≥0.9999 |
| l1_bwd_bench (1,32768,8,128,128) | PASS | dA/dbeta ratio=0.9999, 其余 1.0000 |
| l2 负向 7 例 | 全部正确拒绝 | [BOUNDARY_PASS] ×7 |
| bnd_beta_zero / gk_large ×2 / fwd_gk_extreme | PASS | ratio=1.0000 |
| bnd_bwd_gk_extreme | WARN（非阻塞） | 文档化 fp16 分裂范围外限制（g=+40），与改前行为一致 |

直路 `T.copy(BetaPacked[bb, bs, bh, :], bt)`（4D 张量 3 标量+全切片 → 1D tile）
在全部形状（H=2/4/8/64，bf16/fp32）下数值正确，无需 2D 视图回退。

### 3.2 A/B 计时（do_bench 口径：NPU 事件、warmup 20、rep 10、均值、仅设备侧）

- 同一 job（fla-r1-gather-ab，exit 0，230s）内先 BEFORE 后 AFTER，同机同会话公平对照。
- 形状与 baseline 轮完全一致（fwd 1×8192×64×128×128 bf16 / bwd 1×32768×8×128×128 fp32，seed 42）。

| 版本 | fwd (ms) | bwd (ms) | 合计 (ms) |
|---|---|---|---|
| 改前（快照，同机复测） | 4.773 | 6.884 | 11.657 |
| 改前（baseline 轮记录） | 4.783 | 6.935 | 11.719 |
| **改后（packed beta）** | **4.685** | **6.778** | **11.463** |
| Δ vs 同机复测 | **−1.84%** | **−1.54%** | **−1.66%** |

判读：
- 同机复测与 baseline 记录吻合（<0.7%），测量口径可信、噪声 ~0.2-0.5% 量级。
- fwd/bwd 两方向一致为正，净收益真实但**低于 3% 噪声阈值**。
- 净收益中含 host repack 开销的抵消（每次 wrapper 调用多一个 ~1-2MB 设备侧转置
  kernel，估 ~20-50us/次，计入 do_bench 窗口）——kernel 侧真实收益略高于净值。
- **画像假设被 A/B 证伪**：若 gather 链真占 aiv_scalar_time（u: 2.64us/block）的
  主导份额，u 的 TaskDur 应显著下降；实测 fwd 仅 −1.8%。说明 gather 链只是 scalar
  pipe 占用的一部分，且 scalar pipe 高占用与关键路径的关系需要复测画像定量确认
  （可能存在 pipe 间重叠）。待 §3.3 数据。

### 3.3 复测画像（msprof op，u/wkg/k1/k2 四代表核 + 应用级时间线）

产物：`perf_tuning/fla_wy_fast_prof/iter_r1_gather/{msprof_out, profiling/}`（已取回 CPU；
r1 实现采集于远端 dsa_r1_gather，md5 b5974d89 与 CPU 一致；app 级 PROF 71MB）。

| 核 | 指标 | baseline | r1_gather | Δ |
|---|---|---|---|---|
| u | TaskDur(us) | 1934 | 1889 | **−2.4%** |
| | AIV/blk(us) | 4.45 | 4.23 | −4.9% |
| | AIC/blk(us) | 3.79 | 3.61 | −4.7% |
| | AIV scalar share | 60.8% | 58.0% | −2.8pp |
| wkg | TaskDur(us) | 2838 | 2813 | −0.9% |
| | AIV/blk(us) | 6.34 | 6.65 | +4.9% |
| | AIC/blk(us) | 5.22 | 5.69 | +9.0% |
| | AIV scalar share | 62.7% | 71.9% | +9.2pp |
| k1 | TaskDur(us) | 1300 | 1265 | **−2.7%** |
| | AIV/blk(us) | 4.72 | 4.66 | −1.3% |
| | AIC/blk(us) | 6.01 | 5.72 | −4.8% |
| | AIV scalar share | 99.2% | 95.0% | −4.2pp |
| k2 | TaskDur(us) | 1678 | 1657 | −1.3% |
| | AIV/blk(us) | 9.45 | 8.70 | −7.9% |
| | AIC/blk(us) | 7.08 | 6.87 | −3.0% |
| | AIV scalar share | 58.6% | 64.7% | +6.1pp |

（wkg 的 AIV/AIC per-block 上升但 TaskDur 微降：有效通道数 15.1→16.6，调度重叠
变化，量级在噪声边缘；wkg 因 fwd 保 bf16 staging+cast 路径，4D 切片载入的地址
scalar 略增，见 §4 归因。）

## 4. 结论

### 4.1 判定：不采纳（低于 3% 噪声安全门槛）

- 端到端 **−1.66%**（11.657→11.463ms，两方向一致为正、超实测噪声 3~8 倍 → 真实但小）；
  TaskDur 复核 −0.9~−2.7%。按既有采纳门槛（3%）判**不独立采纳**，交付物不动；
  改动作为已验证候选存档 `history_version/fla_wy_fast_impl_prof_r1_gather.py`
  （精度全绿 + ruff 过，上游修复 cast-on-copy 后 fwd 侧收益会变大，可复启）。
- 沙箱已回滚到交付版基线（= history_version/fla_wy_fast_impl_prof_r1_before.py）。

### 4.2 本轮真正产出：画像假设证伪 + 剩余 scalar 归因（比 −1.66% 更值钱）

1. **证伪**："gather 链主导 AIV scalar"不成立——消除 gather 链后 scalar share 仍
   58~95%，u 的 AIV/blk 仅降 0.22us（4.45→4.23）。gather 链只是 scalar 的一小片。
2. **归因（画像 + 消融交叉）**：剩余 per-block scalar 是**随 chunk 工作量增长的
   部分**——fp16/bf16 hi/lo 拆分链（每 gemm 操作数 ~6 次 copy + sub，工具链 fp32
   gemm 损坏所迫）、~15 次 T.copy 的 4D 切片地址计算与 DMA 描述符、staging 中转
   （cast-on-copy 损坏所迫）、flag/循环控制。**即剩余 scalar ≈ 工具链强制的模拟开销**。
3. **连锁修正**：原候选 2（多 chunk/block 摊薄）的预期收益被 1 大幅削弱——可摊薄的
   固定 prologue scalar（gather 链 + A 载入设置）实测只有 ~0.2-0.5us/block，
   摊薄上限 <5%，不值得冒 debug_log 记录的"serial 循环包 Pipelined 体"工具链风险。
   **候选 2 取消。**

### 4.3 穷尽判定更新（对照 migration skill 判据）

- 流水图 stall 归因：剩余 scalar（58~95%）经 r1 消融定位为**已证伪工具链缺口的
  强制开销**（fp32 gemm 损坏 → 拆分链；cast-on-copy 损坏 → staging；已提 issue ①③）；
- 结构性重排：broadcast 化（已采纳）/ hoist·CSE·scatter（已试无收益）/ kernel
  合并与多 chunk（工具链锁死或预期 <5%）——本轮后无剩余候选；
- 地板核算：唯一 IO 1.97ms 带宽地板 + host 分发稳态隐藏（实测）；
- **结论：kernel 层路线穷尽成立**（全部剩余瓶颈归因到上游 issue ①③⑥ 的工具链
  缺口）。当前 11.66ms（同机复测口径）= 目标 1370us 的 8.5×；物理不可达结论不变。
