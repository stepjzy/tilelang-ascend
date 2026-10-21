# P3 数据流重构证伪报告（sequence-level dedup panel residency）— 2026-10-12

> **route**: P3（DESIGN_dataflow.md §5 选定方案：去重驻留 panel + 转置 gemm + 选值 + 双发射回退）
> **baseline**: git `22bc68a8`（P2 交付态，官方 BI128 = 363.1µs）
> **verdict**: **路线证伪（§6.2 预注册条件 1 触发）→ 已按用户规则回退基线**。
> 正确性全绿（L0-D 快速族 D1-D8b + D7 竞态全过，含路径断言）——证伪纯因性能；
> P3 完整实现归档 `history_version/sparse_mla_impl_p3_falsified.py`（1560 行）。

## 1. 官方口径终值（graph-replay，同会话 3 轮，official arange seed=42 逐字）

| 项 | 实测 | vs 基线 363.1µs | vs 预期带 175-200 | vs 证伪线 220 |
|---|---|---|---|---|
| **P3 端到端（三连发捕获：prep+panel_g+generic_g）** | **894µs**（893.5/894.6/896.4） | **+146%** | 4.5× 带外 | **4.1× 超线** |

## 2. 分解归因（§6.2 协议：prep / launch / panel 主体 / 选列）

| 分量 | 实测 | 攻击线（§6.2） | 判定 |
|---|---|---|---|
| prep kernel（detect+幅值门） | **29.3µs** | >35 才攻 | 带内（无需攻） |
| 双发射空跑 launch（generic 早退） | **25.3µs** | （8-12 估计的 2 倍余） | 次要 |
| panel 主体（V3 地板变体：C 侧+握手+装载） | **291µs** | >165 攻 panel | **超线 +126µs** |
| 选列+reduce-first V 计算（full−floor） | **~565µs** | >40 攻选列 | **超线 14×** |

**根因链（V3 变体二分 + 生成码实读 + issue-latency 模型）**：

1. **通用化选值的每 task V 代价是 issue-latency 界**：P3 的无契约选值（等值比较矩阵
   [topk,HALF] + k 计数归一 + reduce-first 链）≈ **36 条串行向量指令/task/AIV**，
   实测 ~5.5µs/task（≈153ns/issue 依赖链延迟）× 102.4 task/核 ≈ 565µs。Plan-A 的
   契约直映射 V post 仅 ~15 条小指令（[HALF] 级）≈1.4µs/task 且深藏于 C 节拍之下。
   **设计的 10-20µs 选列预算对任何逐 task O(topk×HALF) 算术选值链低约 30×**。
2. **gather 硬件墙封锁单指令选值路径**（spike-2 裁决）：910B3（dav_c220）上
   `Gatherb` 仅支持 uint16/uint32（fp32 静默无效，实测全零）；元素粒度
   `AscendC::Gather`（支持 float）的 offset 硬性 `LocalTensor<uint32_t>`，而
   设备侧计算偏移无法抵达 uint32（int32→uint32 vconv cast 无实现，CAST_NONE/
   CAST_RINT 实测均编译失败；uint32 tile 算术仅 fill）。⇒ 选值只能走算术矩阵
   （根因 1）或 GM 标量读（设计 R1 备选链形态，64 GetValue/task ≈ 1.6µs/task，
   同样超预算且更弱）。
3. **panel 地板 291µs vs Plan-A 143.3µs 锚（+148µs）**：嫌疑主因 = 转置 gemm 的
   L0A 满预算 tile（M=128：[128×128]×2B = 32KB 恰满 kNumSteps>1 时的乒乓槽预算，
   mte1↔mma 重叠余量归零；Plan-A 的 M=16 L0A tile 4KB 自由乒乓）。未单独实证
   （证伪已由 1+2 决定，地板归因列开放）。

**单轮定向迭代不可达性算术**：即使同时做到（a）compare+select 融合 + rows2d 组内
hoist（选列 ~565→~350，域通用合法）；（b）非转置 gemm 修复地板（291→~150，未实证）；
（c）idx 变更检测的组内 eq 缓存（域通用：按 task 实测 idx 比较invalidate，共享输入
恒命中）——三者合计最优 ≈ 350-450µs，仍 2× 超线。达到 220 需要 V post ≤ ~1.4µs/task
（≤10 条指令），其必要条件是**单指令硬件 gather 选值**（框架 codegen 泛化 int32
offset + 组内 pos 表缓存）——两项框架级改动超出"一次定向迭代"边界，且即便达成
端到端仍估 235-305µs（>220）。**结论：在本硬件 API 面与设计机制约束下，175-200µs
带不可达，220 证伪线不可达。**

## 3. spike 记录（§8 首日清单）

| spike | 结果 |
|---|---|
| spike-1 转置 gemm | **PASS**：rel=3.19e-07 vs torch；与非转置方向 torch.equal=True（K 累加序位等价实证，§5.3 论证成立） |
| spike-2 gatherb 参数化 | **FAIL on 910B3**：dav_c220 GatherbImpl SupportType<uint16,uint32>（fp32 静默无效，实测含仓内测试原配置全零）；元素 Gather 需 uint32 计算偏移（vconv cast 缺失）⇒ 改用 R1 备选链的 UB 内精化形态（reduce-first + 等值比较矩阵 + k 计数归一） |
| spike-3 prep 计时 | 29.3µs（带内） |
| spike-4 端到端正确性 | L0-D 快速族全绿（见 §4） |
| spike-5 三连发+graph | 三连发 kernel 边界可捕获 ✓；**host 分发的 prepare_token_indices D2H 不可捕获**（capture 内 aclrtMemcpy 报错）——D9 计时改用三连发捕获口径（与 363.1 基线的 direct-kernel 口径一致） |

## 4. 正确性验证（证伪前完成——实现本身正确且通用）

L0-D 快速族（/tmp/opencode/p3/quick_d.py，路径断言 = prep Violations 值）：

| 用例 | 路径 | 结果 |
|---|---|---|
| D1 官方 golden arange | panel（viol=0） | PASS（err_ratio 1.648e-03，与 generic 历史值同级） |
| D2 同 shape 随机非共享 | generic（viol=40） | PASS |
| D3 重复索引共享（k 归一语义） | panel | PASS（raw ref matched=1.0，max_abs 1.9e-06） |
| D4 乱序集合共享（逐行独立排列） | generic（viol=40） | PASS |
| D5 -1/越界共享 + 全 -1 行 | panel | PASS（行级精确零） |
| D6 G=2 + full-range（大 P_seq） | generic | PASS |
| D8 offsets {单段/[0,0,1023,2048] 空首段} × shared | panel | PASS（空首段用 masked ref——朴素 ref 的 amax 零尺寸限制） |
| D8b RH=2（H=128，4 h-chunk） | panel | PASS（err_ratio 4.9e-04） |
| D7a 官方 shape ×3 run | panel | 3-run 位级稳定 + ref 全过 |
| D7b 混合路径交替（shared/random）×3 | 双路径 | 位级稳定 + 全过 |

实现期修复的缺陷（过程记录）：① clamp count 必须为输出张量总尺寸（2D [topk,HALF]
传 HALF 只处理首 64 元素——rs 链错误根因）；② h-chunk 装载/处理必须单循环交错（分离
双循环导致 accT_ub 被末 chunk 覆写——H≥32 全错根因）；③ UB 32B 对齐（移位拷贝
[1:topk] 4B 未对齐 → aicore 异常——改为 k 计数归一消除移位）。

## 5. 流量核算（用户 req 四.5——机制账本 + 实测核对口径）

| 流量腿 | 基线（generic） | P3 panel 路径（机制） | 说明 |
|---|---|---|---|
| V 行读（gather 源） | 302MB | **~5.9MB**（20核×2组×128行×1152B staging 读） | 每核每组一次 |
| V bulk 写 ws | 302MB | **~5.9MB**（ws_panel 写） | 同上 |
| C ws 读回 | 302MB | **~5.9MB**（ws_panel→L1） | 同上 |
| Q 读 | 37.7MB | 37.7MB | 不变 |
| ws_acc 写+读 | 33MB | ~49MB（FIX 8KB + 双 AIV 全片读/task） | 全片读代价 |
| 输出+idx/lse 等 | ~20MB | ~13MB | |
| **合计** | **~995MB** | **~118MB（8.4×↓）** | **流量消除机制成立**——但带宽不再是瓶颈（995MB@2.96TB/s≈333µs 地板被打破后，V 侧指令延迟成为新瓶颈 565µs） |

**结论**：流量账本的方向正确（9× 消除达成），但联合瓶颈分析（joint_bottleneck
§1.3"改进空间只在更平滑的仲裁"）的推论在此延伸——**消除带宽瓶颈后暴露的是 V 侧
per-task 指令延迟瓶颈**，而通用（无契约）选值机制恰恰是该瓶颈的最大加项。Plan-A
的 143µs 锚依赖契约直映射把选值降为 O(HALF)；去除契约（用户红线）后该成本不可
回避。**带宽 ∝ 流量 与 延迟 ∝ 指令数 在本算子的 V 侧不可同时最优化——除非硬件
提供 fp32 gather（当前 API 面封锁）。**

## 6. 备选链建议（供 orchestrator 重裁决）

1. **框架级（若可立项）**：codegen 泛化 `AscendC::Gather` 的 int32 offset（tl::ascend
   包装 + ReinterpretCast，~10 行，uint32 既有用户零影响）+ 组内 pos 表缓存（idx
   变更检测 invalidate，域通用）+ 非转置 gemm——三项合一估 235-305µs，仍不保证 <220，
   且含 2 项框架改动；立项前需先单独 spike gather 的真机时序收益。
2. **接受 363.1µs 基线**（当前最优已验证状态；220 目标在本硬件 API 面下经本轮证据
   应评估为不可达——除非上述框架级方向成立）。
3. **不推荐**：任何形式的共享契约特化（用户已禁；Plan-A 143µs 的本质即契约红利）。

## 7. 回退执行记录

- `sparse_mla_topk_reducesum.py`：git checkout 还原至 22bc68a8（825 行，P2 交付态）
  ✓；test 文件全程未改（基线 1295 行）✓；common.h 的 spike 期 gatherb 模板试验
  已在中途还原 ✓（工作树 = 基线，仅 tvm 子模块标记）
- P3 实现存档：`history_version/sparse_mla_impl_p3_falsified.py`（1560 行，
  含 prep/panel/guarded-generic/三连发 host 分发全链，可复现）
- 修改前备份（基线态）：`history_version/sparse_mla_impl_pre_p3.py` /
  `test_sparse_pre_p3.py`
