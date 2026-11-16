# 上游 rebase 复验记录（2026-11-16）

> 目的：把本地主仓 rebase 到上游 tile-ai/tilelang-ascend 最新，并重跑 scatter
> 三臂实验，验证上游 PR 是否已解决我们的问题。
> 结论：**代码层面上游未修复 direct scatter；实验层面 NPU 环境被升级（torch_npu
> 2.7.1→2.10.0）导致基线（contig）也坏，无法在本环境做有效对比。**

## 一、仓更新

| 位置 | 更新前 | 更新后 | 说明 |
|---|---|---|---|
| 本地主仓 ascendc_pto | 4805777d（11-06） | **ce411463**（11-13，上游最新） | 快进 rebase，无本地独有 commit |
| NPU 执行侧 repo | 4805777d | ce411463 | 同步后跑实验，实验后切回 ce411463 |
| fork origin（stepjzy） | — | — | 未动（fork 快照停在 09-10，比本地旧） |

本地工作区 untracked 开发产物全部保留，主仓零 commit 纪律未破坏。

## 二、上游对 scatter 问题的覆盖情况（代码审计）

4805777d..ce411463 之间与 copy 相关的代码变化：
- `src/op/ascend.cc`：**无变化**
- `src/tl_templates/ascend/common.h` 的 copy_ub_to_gm/copy_gm_to_ub：**无变化**
  （仅 catlass v2.2.0 升级引起的 include/API 拼写修正，copy 逻辑不变）
- 区间内只有 8 个提交，全部是算子 example + PTO 模板，与 ascendc copy 无关

GitHub 相关条目状态（2026-11-16）：
- #1682（我们问题的官方同族版）→ 已由 **#1795 合并修复**（flat 折叠 + 非32B倍数行距 trap），
  但其 PR 正文明言"已知边界"：**紧凑亚 32B 行距 + 带行距目的（我们的 direct scatter）
  未修复**——本地 4805777d 已含 #1795，direct 臂在旧环境下实测 504/512 错即铁证。
- **#1913**（sub-8B strided GM↔UB lower 修复，fix #929）：**open 未合并**
- **#1940**（物理 UB 行距保留）：**open 未合并**
- **#1756/#1767**（sliced width 当物理行距）：**open 未合并**
- #1325（T.copy 加 stride 的 feature 请求）：open，0 评论

**代码层面结论不变：上游最新仍没有修复 direct scatter（紧凑源+跨步目的）。**

## 三、三臂重跑结果（NPU，最新 ce411463）

| 臂 | 旧环境（11-14，torch_npu 2.7.1） | 新环境（11-16，torch_npu 2.10.0） |
|---|---|---|
| contig | **0/512 错（PASS）** | **511/512 错（FAIL）** |
| direct | 504/512 错 | 511/512 错 |
| staged | 511/512 错 | 511/512 错 |

## 四、关键发现：NPU 环境被升级，基线被破坏

1. **环境变化实锤**：
   - torch_npu：2.7.1.post4 → **2.10.0.post7**（torch 2.7.1+cpu → 2.10.0+cpu）
   - npu-smi（driver）：25.5.0
   - 新出现 warning："NPUCachingAllocator: The current CANN and HDK(driver)
     versions require processing for 32 padding size"
2. **归因实验**：把 NPU 切回 4805777d（与旧环境验证时同 commit）重跑 contig，
   结果同样 511/512 错 → **回归来自环境升级，不是 4805777d..ce411463 的代码**。
3. **形态**：contig/staged 输出全零（仅期望值为 0 的 [0,0] 碰巧对），direct 输出
   垃圾值——与故障 B（MTE3 写可见性/同步竞态）形态一致。疑似 torch_npu 2.10 的
   acl runtime 行为变化导致 MTE3 写 GM 的完成语义变化。
4. **影响**：本环境（torch_npu 2.10）上无法对任何 copy 实验做有效判定——连
   contig 基线都坏。若要严谨验证"上游修复后 direct 是否变好"，需等环境恢复
   2.7.1 或用其他干净环境。
5. 这是我们交给用户决策的新线索：**torch_npu 2.10 + 简单 kernel 的 MTE3 flat 写
   全零**，值得单独验证（用带 flag 链的完整算子对照）。

## 五、生成代码归档

本次（ce411463）三臂生成的 copy 调用与 4805777d 完全一致（copy 代码未变）：

| 文件 | 关键调用 |
|---|---|
| generated/gen_contig_latest.cpp | `copy_ub_to_gm<float,64>(OutFlat[cid*64], acc[0], 64,1,64)` |
| generated/gen_direct_latest.cpp | `copy_ub_to_gm<float,64>(Out[cid], acc[0], 8,64,1)` |
| generated/gen_staged_latest.cpp | `copy_ub_to_gm<float,1,64>(Out[cid], buf[0], 8,64,1)` |

即：**lower 层产物未受上游更新影响，问题依旧**。

## 六、结论与后续建议

1. 我们的 scatter issue 仍值得提交：上游 #1913/#1940/#1756 均未合并，且我们的
   探针证据（槽内复制填充、stride 单位）与 UB 分配低估（8× 越界）是上游无人
   提及的增量。
2. 后续实验需注意 NPU 环境版本（torch_npu 2.10 下 copy 基线坏，先别用它做
   精度判定；可用交付算子完整 kernel 复验 2.10 环境是否只影响简单 kernel）。
3. 建议动作：在新环境下用**交付算子**（含完整 flag 链）跑一次精度，判定 2.10
   环境是否影响复杂 kernel；若也坏，向环境管理员报告。
