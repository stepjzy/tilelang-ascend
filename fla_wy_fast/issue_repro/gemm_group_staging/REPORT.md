# 多 gemm 组 L0 staging 问题 —— 复现尝试报告（2026-11-16）

> 背景：交付总结 §8-3 记录"多 gemm 组共享 L1 操作数（角色互换复用）损坏 L0 staging：
> 阻塞 bwd k1+k2 / v1+v2 合并（输入重复读的主因）"。本报告还原问题定义、给出
> 复现脚本与最新仓（ce411463）复验结果。

## 一、问题定义（基于 gemm_v0 模板源码 + 交付记录）

**"L0 staging"指什么**：`gemm_v0`（`src/tl_templates/ascend/common.h` ~1399 行）内部
实现 L0A/L0B 的 **ping-pong 双缓冲流水**（源码注释原文）：

> "Pipelined main loop. Prime/drain the L0A/L0B ping-pong buffers ONCE and let the
> ping-pong run continuously across the WHOLE (N-tile, K-tile) sequence. Each tile's
> L1->L0 load goes into the free buffer while the previous tile's mma runs."

每次 `gemm_v0` 调用固定用事件 `L0AB_EVENT = 0`（和 1）做 prime/drain 握手，操作数
按角色从 L1 搬运进 L0A（A 操作数）或 L0B（B 操作数）。

**"多 gemm 组共享 L1（角色互换）"指什么**：fla_wy_fast bwd 的数学里，
- k1 组：`dA += dw @ K_bg^T`（transpose_B）—— **dw 是 A 操作数，进 L0A**
- k2 组：`X = A_c^T @ dw`（transpose_A）—— **dw 是 B 操作数，进 L0B**

合并 k1+k2 到单 kernel 时，同一个 `dw` L1 buffer 先被 L1→L0A staging、后被 L1→L0B
staging——**角色互换复用**。当时的记录（baseline.md/final_summary.md §8-3）称此
组合"损坏 L0 staging"，阻塞了 bwd kernel 合并（合并可消除输入重复读，是
-30%~-40% 性能项）。

## 二、交付算子中遇到的位置

bwd 被拆 6 kernel（k1/k2/v1/v2/gk/gz）。k1（dA 部分和 + dk2/dg2/dbeta）内部已有
4 个 gemm_v0/迭代（dw_hi/lo × kbg_hi/lo），但 **X=A^T@dw 被拆到 k2 kernel**，
原因正是"合并时 dw 角色互换损坏 L0 staging"（Stage 2 论证，详见
`perf_tuning/fla_wy_fast_final_summary.md` §8-3 与 `fla_wy_fast_baseline.md`）。

## 三、复现脚本（本目录，4 个层级逐步逼近 bwd 实际形态）

| 脚本 | 形态 | 与 bwd 的差距 |
|---|---|---|
| `gemm_group_repro.py` | 2 gemm 背靠背共享 dw_l1（角色互换） | 最小 |
| `gemm_group_repro2.py` | 循环外 X gemm + Pipelined 循环内 dA 累加 | 接近 |
| `gemm_group_repro3.py` | hi/lo 4 gemm 组 + 角色互换 + Pipelined | 很接近 |
| `gemm_group_repro4.py` | 同上，M=N=128/DK=4096 放大到 ms 级（性能观测） | 性能放大版 |

## 四、最新仓（ce411463）复验结果

| 层级 | 正确性 | 性能 |
|---|---|---|
| repro1（2 gemm） | ✅ PASS（max_err=0.000） | merged 0.198ms vs split 合计 0.387ms |
| repro2（X 循环外 + dA 循环内） | ✅ PASS（0.000） | merged 0.224ms vs split 0.386ms |
| repro3（hi/lo 4 gemm） | ✅ PASS（0.000） | merged 0.216ms |
| repro4（性能放大 M=128/DK=4096） | ✅ PASS（≤0.001） | **merged 0.223ms vs split 合计 0.423ms（合并快 1.9×）** |

**结论：四个层级均无法复现"L0 staging 损坏"。当前工具链下，多 gemm 组共享 L1
（角色互换）的合并形态正确且更快。**

## 五、无法复现的可能解释（如实）

1. **时间差**：Stage 2 论证在旧 commit（约 `1d752539`，10 月中旬）上进行；本次复验
   在 `ce411463`（11-13 最新）。中间 1000+ 提交中 gemm_v0 模板未变，但
   schedule / auto-sync / CV 划分 pass 可能有间接修复。
2. **触发条件未还原**：当时场景含完整 V 算链（exp/mul/broadcast/sub cast 链 ~12 条
   ops/迭代）、beta gather、dbeta reduce、更多 UB/L1 压力——C/V 交替更密集。
3. **当时"损坏"的表现可能不是数值/性能可测的**：可能实为挂死或编译错
   （sync-point 失配，§8-4 同族），与 #1926 交织；这类表现在本最小形态不触发。

## 六、下一步建议（供决策）

- **A（成本低，价值高）**：在旧 commit（1d752539）跑 repro3/repro4，确认"损坏"
  是否曾在旧版可复现。若可复现 → 上游已在区间内修复 → 无需报 issue，记录即可；
  若也不复现 → 说明触发条件比"角色互换"更复杂，回到 §5-2 还原完整形态。
- **B**：直接以"合并更优"为结论，在 bwd 交付代码上重新尝试 k1+k2 合并
  （在新工具链下），若合并成功且正确 → §8-3 问题已过期，性能账重估。

## 七、参考资料

- 官方文档：TileLang Ascend gemm 的公开文档仅 autoapi 骨架页
  （https://tilelang.ai/autoapi/tilelang/ascend/op/gemm/index.html），
  **L0 staging 行为的最权威来源是模板源码注释**（common.h gemm_v0 的
  "Pipelined main loop" 段）。
- 相关上游条目：
  - #1237 [open] gemm_v0 需要 ping-pong 流水操作（L0 流水的 feature 请求）
  - #110 [open] double buffer + 自动同步 → 生成 PipeBarrier<PIPE_ALL>（同步过重）
  - #1907 [open] Preserve Block annotations and ND offset for UB→L1 staging
  - #1981 [open] Correct copy_pipe_to_l1 auto-sync pipeline annotation
  - #1908 [open] AscendLowerParallelToVector drops Block annotations →
    silent wrong results for flat-L1 K-slice
  - 我们自己的 #1926（多跳 C/V 挂死，与本问题相邻但不同）

## 八、权威来源核验（2026-11-16 追加）：gemm_v0 是否支持"同一 L1 buffer 先 L0A 后 L0B"

按用户要求，逐级查官方文档源码对"角色互换"的契约声明：

1. **TileLang 官方 API docstring**（`tilelang/language/ascend.py:418` `gemm_v0`，
   对应文档站 https://tilelang.ai/autoapi/tilelang/language/index.html）：
   > "A and B must be in L1 (``T.alloc_L1``). gemm_v0 internally handles
   > L1→L0A/L0B movement."
   
   **无任何"A/B 不可复用同一 L1 buffer / 不可角色互换"约束。**

2. **项目官方编程指南**（`docs/TileLang-Ascend Programming Guide.md`）：
   - L596：shared→L1/UB、fragment→L0A/L0B/L0C，编译器按上下文自动分析和识别；
   - L629-630：L1→L0A（左矩阵）与 L1→L0B（右矩阵）是**两条独立搬运路径**，
     无排他性声明。

3. **gemm_v0 模板实现**（`src/tl_templates/ascend/common.h`）：
   每次调用自包含（开头 prime `SetFlag<MTE2_MTE1>/FIX_M/M_MTE1`、
   结尾 drain `WaitFlag<M_MTE1>`），循环内对 caller 传入的 L1 地址
   执行 `copy_l1_to_l0a / copy_l1_to_l0b`——**无任何地址角色状态机**。

4. **copy_l1_to_l0a / copy_l1_to_l0b 实现**（common.h L181/L232，底层为
   catlass `TileCopyTla` + `AscendC::LoadData/LoadDataWithTranspose`）：
   **纯函数式搬运**——入参只是 src/dst 地址+几何，无"该地址曾用作何种角色"
   的检查，代码路径对第一次/第二次/角色互换使用完全相同。

5. **官方测试覆盖**（`testing/python/language/test_tilelang_ascend_language_l1_to_l0.py`，
   8 组 L1→L0A/L0B 正确性场景）：**没有任何"同一 buffer 先后充当 A/B"用例**。
   未覆盖 ≠ 禁止，但说明官方从未回归这个组合。

6. **CANN 侧**：公共 `DataCopy` API 无 L1→L0A/L0B overload（该搬运是 Cube 专用，
   只在 catlass 模板层）；CANN 9.2.0 安装包内无文档目录可查。

**结论**：所有官方来源都把 L1→L0A/L0B 描述为对地址无角色约束的纯搬运——
**gemm_v0 的 API 契约支持"同一 L1 buffer 先入 L0A 再入 L0B"**；不存在 gemm_v0
单点 API 层面的"角色互换 Bug"。若该组合出现问题，缺陷必然位于更高层 pass
（memory planning / C/V 划分 / sync insertion / Pipelined 版本化），而非
gemm_v0 本身——这与 §8-3 上下文（多 gemm 组 + 完整 V 算链 + UB 压力）以及
相邻 issue（#1926 多跳 C/V 挂死、#110 double-buffer auto-sync、#1908
Block 注解丢失 → L1 K-slice 静默错）的层次一致。即：**"多 gemm 组 L0 staging
损坏"最可能的机理不是 gemm_v0 不支持，而是高层编译 pass 在 buffer 复用场景
的缺陷（且本最小复现形态未触发该 pass 缺陷的激活条件）。**
