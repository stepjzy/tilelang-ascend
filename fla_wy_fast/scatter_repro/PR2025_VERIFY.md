# PR #2025 验证记录（2026-11-20）

> PR: https://github.com/tile-ai/tilelang-ascend/pull/2025
> "[Fix] Correct Ascend UB column stores and DMA sizing (#2017)"，by ArmandAlbert
> #2017 = 我们提交的 direct 散列 issue（三臂实验 + 三层根因）。

## 一、拉取

worktree 位于 `/mnt/workspace/gitCode/skybasejzy/tilelang-pr2025-test`
（PR head = `aae51016dac7f024b3cc6184df103e035a67dc36`，base = ce411463，1 commit）。

> 坑：`git fetch upstream pull/2025/head` 后 FETCH_HEAD 有两条记录
> （ascendc_pto 顺带更新 + PR head），`git worktree add ... FETCH_HEAD`
> 会取第一条（ce411463）——必须显式指定 PR head SHA。

## 二、静态分析（9 文件 / +628 行）

| 文件 | 改动 | 与 VERDICT2 的对应 |
|---|---|---|
| src/op/ascend.cc | 新增 `compact_column_copy` 检测（ub2gm + 1D 紧凑源 + 2D 单列目的）→ 分配临时 UB 槽 buffer → `sync_all → copy_ub_column_pack → sync_all → copy_ub_to_gm → sync_all` | **第一/二层根因：紧凑源槽化**（片上完成，优于我们交付算子的 GM 中转） |
| src/tl_templates/ascend/common.h | 新增 `copy_ub_column_pack`（Duplicate 逐元素填 32B 槽首；注释明言避开 scalar SetValue+bisheng 死锁） | 槽化原语 |
| src/transform/ascend_memory_planning.cc | buffer 大小在循环绑定后计算 + `NarrowCopyBounds` 细化 Select 尾块 | **第三层根因：UB 分配低估** |
| src/transform/lower_tile_op.cc | 生成 scratch buffer 名预占用户 buffer 名 | 配套 |
| +2 个测试文件 | 21 项回归（含 issue2017 精确复现用例） | — |

**静态结论：修复方向与我们的三层根因逐条一致，代码逻辑正确。**
lowering 检测条件覆盖我们的 direct 场景（1D [64] 现算源 → Out[0,:,bh] 单列）。

## 三、动态验证：被 NPU 环境问题阻断（重要）

本环境（DevEnvC_DCbyi_npu）torch_npu 已升级为 **2.10.0.post7**，之前已发现
它破坏 copy 基线。本轮进一步坐实**非确定性竞态**：

| 脚本 | 代码版本 | 结果 |
|---|---|---|
| scatter contig | PR 2025 (aae51016) | 511/512 错 |
| scatter contig | ce411463（无 PR） | **104/512 错** |
| scatter contig | ce411463（重跑 1） | **512/512 错** |
| scatter contig | ce411463（重跑 2） | **512/512 错** |

同一份代码三次运行结果漂移（104→512→512）= 运行时竞态，与本环境
torch_npu 2.10 相关（11-16 已发现 contig 基线 511 错；当时回退 4805777d 同样坏）。

**PR 自带测试 `test_tilelang_ascend_language_strided_copy.py`：13/13 FAIL**
（含 aicore exception）。PR 作者 CI（CANN 9.1.0-beta.1）自述 2936 项测试 0 失败。
由于本环境连 contiguous flat copy 都非确定性错误，**13/13 FAIL 不能归咎于
PR 本身，也无法确认修复效果**。

## 四、结论与后续

1. **静态层**：PR 2025 修复正确、方向与我们 VERDICT2 一致，可以去 PR 页面做
   代码 review 级确认/approve。
2. **动态层**：本环境（torch_npu 2.10）无法给出干净判读——连基线都竞态。
3. **后续**：
   - 向环境管理员报告 torch_npu 2.10 的 copy 非确定性（现在有 3 次漂移铁证）；
   - 环境恢复（2.7 系列或干净 CANN 9.1.0）后：
     a) 在 PR 代码跑 scatter_3arms 三臂 → 期望 contig 0 错 + direct 0 错；
     b) 跑 `test_tilelang_ascend_language_strided_copy.py` 全量；
     c) 在 PR 代码跑交付算子 fla_wy_fast 全量测试（L0/L1/L2/Boundary），
        确认修复对交付算子无回归（尤其 bwd dbeta 散列路径）。
   - 若三臂 direct 归零，可在 PR 页面评论确认修复有效（附复现数据）。