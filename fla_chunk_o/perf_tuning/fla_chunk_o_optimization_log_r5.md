# fla_chunk_o R5 优化记录（同步协议架构轮）

> R5 输入状态：R4 终态（3-chunk + _FastLaunch），官方均值 1.417ms（本会话
> 复测 1.421ms），精度全过，msprof 设备侧单 launch 1391.1us（R4）/ 1434.2us
> （R5 会话复测）。目标 < 1320us（超 ~7.5%）。用户决策"继续第五轮调优"，
> 主攻方向：同步协议架构本身。**模式红线：Developer 模式内深化（手动
> flag、workspace、pass_configs、compile_flags）允许；整体切换 Expert
> 模式禁止直接执行，需要时上报请示。**本轮预算 10 轮，噪声阈值 3%。

## Part A 优化点清单（主线 1：pass_configs / 编译旗标全量清点）

### A.0 同步旋钮全量清点（源码级，tilelang @ HEAD 43b0b697）

**TileLang TVM 层（tilelang/transform/pass_config.py PassConfigKey 全 28 键，
17 个 tl.* + 11 个 tir.*；影响同步的加粗）**：

| 键 | 默认 | 当前 | 影响同步？ | 说明 |
|----|------|------|-----------|------|
| **`TL_ASCEND_AUTO_SYNC`** | False | True | **是（核心）** | AscendSyncInsert pass：跟踪全部管线（M/V/MTE1/2/3/FIX/S），按 buffer 访问历史插显式 `PipeBarrier`/`SetFlag`/`WaitFlag` 源码级调用（ascend_sync_insert.cc, 1829 行）。当前 kernel 生成源码含 417 行同步指令 |
| **`TL_ASCEND_AUTO_SYNC_VS`** | False（pto 目标 True） | 未设 | **是（补充）** | AscendSyncInsertVS：**只**跟踪 V/S/MTE2/MTE3，是主 pass 的**补充**（transform/__init__.py L384 docstring；engine/phase.py L125-126 两 pass 串行执行）——开它只会**加**同步（V→V、S↔Others），方向与我们相反；但其配对效应（cce_auto_sync=False + O3，pass_config.py L219-221）可用 compile_flags 直接获得 |
| **`TL_ASCEND_AUTO_CV_SYNC`** | False | True | 是（跨核） | CombineCV pass 对 `workspace_*` 名字 buffer 的 V→C relay 自动跨核 flag（ascend_combinecv.cc）。o 路径已手动化；relay 路径仍自动 = 主线 3 对象 |
| `TL_ASCEND_AUTO_CV_COMBINE` | True | True | 间接 | C/V 划分本身（禁用即 Expert 路径，红线内不动） |
| `TL_ASCEND_MEMORY_PLANNING` | False | True | 间接 | UB/L1 生命周期规划（R4 已证 196352B 硬上限在此） |
| `TL_ASCEND_VECTOR_MASK_REUSE` | True | (默认True) | 间接 | A2/A3 vector mask 状态复用，已是理想值 |
| `TL_ASCEND_TAIL_MASK` | False | 未设 | 否 | 尾块 valid-region 改写；本 kernel Select-free 无尾块向量域 |
| `TL_ASCEND_PTO_USE_PIPE_IN_CV_COPY` | True | 未设 | 否 | PTO target 专属 |
| `TL_ASCEND_EXCEPTION_DUMP` | False | 未设 | 否 | 诊断 |
| 其余 tl.*/tir.* 键 | — | — | 否 | Simplify/CSE/vectorize/index bitwidth 等常规编译选项，无同步语义 |

**Bisheng 编译器层（libgen.py resolve_compile_flags + bisheng --help 实测）**：

| 旗标 | 默认 | 说明 |
|------|------|------|
| **`--cce-auto-sync=on/off`** | **on（ascendc）** | 编译器对**内存类操作自动插入同步**（bisheng --help: "automatic synchronization of memory based operations"）。tilelang 仅在派生选项关闭时才发 `=off`（libgen.py L102-105）——**当前 kernel 从未关过，即"TVM pass 显式同步 + 编译器每内存操作再插一层"双层结构** |
| **`--cce-remove-auto-sync=true/false`** | 未知（推测 false） | 编译器级冗余 set_flag/wait_flag 消除 pass（bisheng --help: "Enable cce remove sync: set_flag and wait_flag"） |
| `--cce-auto-sync-log=<file>` | — | 诊断：记录自动同步插入决策 |
| `-O0..3` | O2（#1346 VS 配对用 O3） | 编译优化级别 |
| env `TL_CCE_AUTO_SYNC` / `TL_CCE_OPT_LEVEL` | — | 进程级回退（#1386 后 kernel 级优先）——**不用**，改用 kernel 级 compile_flags |

**清点结论**：当前四件套（AUTO_CV_COMBINE/AUTO_CV_SYNC/AUTO_SYNC/MEMORY_PLANNING）之外，
唯一从未动过的同步旋钮是**编译器层 `--cce-auto-sync`**（默认 on，每内存操作插同步指令流——
正是 R3 定性"每 op 同步指令流地板"中 TVM 显式同步之外的那部分）与
`--cce-remove-auto-sync`（消除冗余 flag 对）。`@tilelang.jit(compile_flags=...)`
kernel 级传参受支持且进缓存键（jit/__init__.py L76-90 docstring）。

### A.1 候选优化点

- [#R5-1] [`--cce-auto-sync=off` 编译器自动同步层关闭]（主线 1；参考: 本文件
  A.0 清点 + libgen.py L81-110 + #1346 配对先例——VS pass 即设计为与
  cce-auto-sync=off 配对，证明该组合为受支持的合法配置）：**适用（待实测）** —
  TVM 层显式同步（417 站点）保持不变承担正确性；编译器层按内存操作粒度
  的自动同步指令流移除。风险：TVM 同步若有覆盖缺口（历史上靠编译器层兜底）
  → 竞态 → 精度失败；护栏：全量精度 + 双 stream 交错。
- [#R5-2] [`--cce-remove-auto-sync=true` 编译器冗余 flag 消除]（主线 1）：
  **适用性存疑（待实测）** — 与 #R5-1 独立/叠加均试；若 #R5-1 已关自动同步，
  本键作用于显式 flag 的死代码消除。
- [#R5-3] [`-O3` 优化级别]（主线 1 伴生）：**低成本顺带** — #1346 配对即
  cce-auto-sync=off + O3；单独 A/B 分离归因。
- [#R5-4] [同步站点数削减 12 → ≤8]（主线 2；R3 记录"FIX 写 12 站点"每站点
  ~0.25us 同步指令流）：**适用** — 大融合 copy、循环不变量外推、peel 边界
  flag 预置扩展。依赖 #R5-1 结果（若编译器层关闭后站点成本变化需重估）。
- [#R5-5] [relay 路径手动 flag 协议]（主线 3；R1-⑥ o 路径手动 flag 省 9%
  先例）：**适用（高风险高收益）** — hi/lo/a/v 四 relay 的 AUTO_CV_SYNC
  自动配对替换为手动跨核 flag（ID 空间 ≥65 可用，计数器容量 <10 pending
  约束）。正确性风险（竞态/死锁）每步 A/B + 精度 + 双 stream。
- [#R5-6] [_FastLaunch stream 查询缓存]（主线 4，次要）：37.6us/调用可缓存
  （护栏 stream 变化检测 + 回退）；预期 -3~4us 低于噪声门，仅顺手时做。
- [禁止重试清单核对] threads=2 裸切 / PTO target / L1 同名 ping-pong /
  纯自动 flag 2-late drain（死锁）/ 深度节流 lag 6/10 / hi/lo 行打包 /
  K-concat 单 gemm / bf16 域掩码 / gate 链融合 / 去 hi/lo 双 gemm /
  block_DV=64 / O 写回迁 V / C 侧全批装载上移 / Select 地址链 /
  v in-body vs 预取 / 4-chunk 批处理（UB 硬上限 196352B）— **全部保持禁止**
  （R1-R4 证伪，本轮无新前提）。

## [ORDER-PLAN] 实施顺序：
1. [#R5-1 cce-auto-sync=off] — 前置依赖: [无] — 理由: 唯一从未动过的同步
   层，纯编译旗标零源码改动，一次 A/B 定方向；成功则 #R5-4/#R5-5 的站点
   成本模型全部重估，必须先行。
2. [#R5-2 cce-remove-auto-sync] — 前置依赖: [#R5-1 结论] — 理由: 作用对象
   与 #R5-1 同层，先定 #R5-1 再定叠加方式。
3. [#R5-3 -O3] — 前置依赖: [#R5-1/2 结论] — 理由: #1346 配对项，分离归因。
4. [#R5-4 站点削减] — 前置依赖: [#R5-1..3 ✅/❌] — 理由: 编译器层结论决定
   每站点成本，避免优化已消失的开销。
5. [#R5-5 手动 relay flag] — 前置依赖: [#R5-4] — 理由: 主线 3 最大剩余
   杠杆，但改动面大，放在低成本项之后。
6. [#R5-6 stream 缓存] — 前置依赖: [无] — 理由: 顺手项，噪声门下仅记录。

## 实施记录

### 基线（本轮起点）
- 官方 bench：**1.421 ms**（mean of 10；per-rep 1.381-1.580，稳态 ~1.40）
- msprof 设备侧单 launch：**1434.2us**（msprof_output_r5_base，Block Dim 20 /
  Mix 40，与 R4 的 1391.1us 同量级，profiling 开销差异内）
- 精度：perf 级 reference check ratio=1.0000（全量 --level all 于候选采纳
  时执行）
- 快照：history_version/fla_chunk_o_perf_r5_start.py（md5 =
  58c43db00e485cf4ceb1b7e85c5a037f，与 r4_final 一致）

[ORDER-CHECK] 准备实施: [#R5-1 cce-auto-sync=off] | 前置依赖: [无] | 结论: ✅
[RESULT-R5-1] 编译旗标 A/B（compile_flags=["--cce-auto-sync=off"]，kernel 源零改动）:
  1.426ms vs 基线 1.421ms（+0.35%，噪声内）。旗标到达性实证：编译失败信息中
  可见命令行 `-O2 --cce-auto-sync=off --cce-auto-sync-log=...`（log 旗标本身
  触发 bisheng ERR99999，弃用）。结论：**编译器层 auto-sync 对本 kernel 无
  实际插入**——417 个同步站点全部来自 TVM AscendSyncInsert pass 的显式调用。
[ORDER-CHECK] 准备实施: [#R5-2 cce-remove-auto-sync] | 前置依赖: [#R5-1 ✅] | 结论: ✅
[RESULT-R5-2] compile_flags=["--cce-auto-sync=off","--cce-remove-auto-sync=true"]:
  1.426ms，噪声内无效果（编译器不消除显式 flag）。
[ORDER-CHECK] 准备实施: [#R5-3 -O3] | 前置依赖: [#R5-2 ✅] | 结论: ✅
[RESULT-R5-3] compile_flags=["--cce-auto-sync=off","-O3"]: 1.435ms（+1.0%，噪声
  内无改善）。**主线 1 关闭：编译器层三旋钮全部无效，compile_flags 已回滚**
  （恢复 R4 终态 md5 58c43db0...）。同步地板 100% 在 TVM pass 显式站点。

### 同步站点结构分析（主线 2/3 的输入，R4 终态生成码逐条清点）

生成码 417 同步行分布（AIC 段 prologue 45 + 主循环体 130；AIV 段主循环体 84 等）：
- **AIV 每迭代 48 个 PipeBarrier<PIPE_V>**：V 管线每个元素 op（cast/Mul/Exp/Sub）
  一次（pass 逻辑 ascend_sync_insert.cc L1599-1607：同管线 W→R/WAW → 每依赖对
  一 barrier；HasDataDependency L1485 **按 buffer 名字判共享内存，无切片分析**）
- **AIC 每迭代 12 个 CrossCoreWaitFlag**（relay flag 检查）+ AIV 每迭代 12 个
  CrossCoreSetFlag + 若干 WAW PipeBarrier<MTE3>
- R4 msprof 量化：AIC relay 等待 ~733us = 12 检查 × 133 批 × ~459ns（与 R2 的
  470ns/次吻合，lag-2 下为**预置检查开销**非阻塞）；AIV throttle 等待 1114.6us
  → **Cube 是 pacer**（V 侧节流等待为主）→ C 侧开销 ~1:1 进墙时间
- ResourceConflictRatio（block0）：C cube_wait 52.9%/mte1_wait 50.2%；V vec_wait
  19.2%/mte2 39.1%/mte3 36.4%——全线握手占满

[ORDER-CHECK] 准备实施: [#R5-4 → 升级为 Iter2"relay 协议批量化"] | 前置依赖:
  [#R5-1/2/3 已关闭] | 结论: ✅（主线 3 提前——量化证据显示 C 侧 relay 检查
  开销 733us 是最大可寻址项，且 C 是 pacer）
[KERNEL-REUSE-PRECHECK]
candidate: relay 手动 flag 协议（主线 3，本文件 A.1 #R5-5）
expanded_domain: 无（同 shape 集合、同调用方式；kernel 内部协议重构）
compatibility_dimensions: 调用域不变；workspace_idx 按索引（cython_wrapper.pyx
  L103 `if i in self.workspace_idx`）不依赖参数名；_FastLaunch 键含 shape/设备/
  流不含参数名
unknown_or_false: 无（AUTO_CV_SYNC 触发条件已源码实证：ascend_combinecv.cc
  L201 `buf_name.find("workspace")` 纯名字匹配；ResourceForPipe("MTE1")=kCube
  L647 实证；环安全由 THROTTLE lag-2 独立保证——R2-9 设计，relay flag 仅承载
  "数据就绪"信号）
plan: repair-kernel-first（改 kernel 内部协议，调用域零变化）
[IMPL-R5-Iter2] 方案：① workspace_1..4 改名 relay_hi/lo/a/v（逃逸
  AUTO_CV_SYNC 名字匹配 → 12 组自动跨核 flag 对消失）；② 单一手动
  RELAY_READY flag（ID 49）：V 侧每 batch 体末尾（最后一个 lo relay copy 后）
  T.set_cross_flag("MTE3", 49)（入 MTE3 队，按序在全部 12 个 relay copy 后
  触发）；C 侧每 batch 体 relay_v hoisted loads 前 T.wait_cross_flag(49,
  "MTE1")（分类到 Cube，MTE1 队上排在 relay loads 前）。稳态 lag-2 下 flag
  预置（V 领先 ~2 batch），单检查替代 12 检查。预期 C 侧省 11×459ns×133 ≈
  673us 检查开销（部分在关键路径）+ V 侧省 11 sets。护栏：全量精度 + 双
  stream + 多次重复运行（竞态间歇性）。

[ORDER-CHECK] 准备实施: [#R5-Iter2 relay 协议批量化（单 batch flag）] | 前置依赖:
  [站点分析 ✅] | 结论: ✅（已实施并实测）
[RESULT-R5-Iter2] relay 改名（逃逸 AUTO_CV_SYNC 名字匹配，12 组自动跨核
  flag 对消失）+ 单一 RELAY_READY flag（ID 49，V 侧每 batch 体末尾
  set_cross_flag("MTE3")，C 侧体首 wait_cross_flag(..., "MTE1")；set 需
  T.Scope("V") 包裹——ResourceForPipe 只认 V/M/MTE1/FIX，MTE3 单独分类
  kExplicit 被 C/V splitter 留在段外，编译期 AscendResourceScopeVerify 报错，
  实证修复）。**官方 A/B：1.421 → 1.934ms（+36% 劣化）**，per-rep 1.82-2.08。
  msprof 设备侧反而 **1346.8us（比基线 1434 快 87us）**——劣化全在设备侧行为：
  AIC cube_wait 98.1%（基线 52.9%）。逐 launch 事件计时：1.36-1.96 剧烈波动
  （基线 1.38-1.58 紧致）。机制：**单 flag 把 C 的 12 次 relay 装载绑定到 V 的
  MTE3 全批排空（最后一个 lo relay copy 退休）之后——批内 chunk 级 C/V 交错
  （C 的 chunk-c 消费重叠 V 的 chunk-c+1 生产）被摧毁，退化为批粒度锁步**；
  逐 relay 渐进触发（每个 set 紧随各自 copy 入 MTE3 队）是承重结构。快发射
  launch（flag 恰好预置）仅 −20us：12 次检查的临界路径份额 ~3%（其余与 MTE2
  装载重叠），远小于锁步损失。**弃用回滚（md5 恢复 r5_start）**。关键数据：
  R4 msprof 的"AIC relay 等待 733us"≈ 12×459ns×133 的预置检查流，但墙临界
  份额极小——flag 检查开销理论（预计 −673us）被实测否定。

[ORDER-CHECK] 准备实施: [#R5-Iter3 hi/lo K=256 融合 + 9 站点手动 flag] |
  前置依赖: [Iter2 教训 ✅] | 结论: ❌（分析性放弃，未写 body）
[ANALYSIS-R5-Iter3] 方案：qg_hi/qg_lo 融合为 relay_hilo [.., BS, 2DK]
  （K-concat），C 侧一次装载 + 单个 K=256 gemm（o_inter = [hi|lo]@[H;H] 分块
  恒等式），H 以两次静态行切片装载堆叠进 [2DK, bDV] L1。收益预算（R3 消融
  定标）：−3 relay 站点（6→3，~33us）−3 gemm issue −3 M-barrier（gemm 全消融
  仅 −30us → 此项 ~−10us）；代价：H 双读 +96KB/batch（Hstate 装载全消融
  −13us → **+13us**）且**同 buffer 名 WAW 使 AUTO_SYNC 在两次 H 切片装载间插
  PipeBarrier<MTE2>（HasDataDependency 按名判依赖无切片分析，L1485 实证）
  ——C 标量流每 chunk 停顿一个 copy 延迟（~0.7us），3 chunk/batch ≈ +250us
  量级，且停顿位于 relay 装载排队之前/之后均入 gemm 关键路径**。净收益
  ≈ −250us（直接变体）；V 侧预堆叠 relay 变体（+3 hh 站点抵消 hi/lo −3 站点）
  净 ≈ −10~−20us，高风险低回报。zN region 操作数路径（单 L1 装载多 gemm 切片）
  为 R2 证伪禁区。**弃用**（预算内 2 轮分析性放弃——源码级证据充分，A/B 无法
  产生正收益假设）。

[ORDER-CHECK] 准备实施: [#R5-6 _FastLaunch stream 查询缓存] | 前置依赖:
  [无] | 结论: ✅（主线 4 收尾）

[ORDER-CHECK] 准备实施: [#R5-4 收尾——终态确认] | 前置依赖: [主线 1-4 全部
  结论化] | 结论: ✅
[RESULT-R5-4] _FastLaunch stream 快路径（私有 _npu_getCurrentRawStream +
  导入期/首用交叉校验 + 永久回退；中途修复 wrapper 内 global 赋值
  UnboundLocalError）：host 分发 163→~127us，官方 bench 1.430/1.417
  （稳态 ~1.403-1.405，噪声内）——**host 完全隐藏于设备时间后，官方口径
  无感**（与任务书预期一致）。按"无变化回滚"规则回滚；实现与护栏已验证
  可行（history_version/fla_chunk_o_perf_r5_iter4_stream.py 备查）。

## R5 结论
- 采纳 0 项；主线 1-4 全部穷尽：主线 1（编译器三旋钮无效）、主线 2
  （站点削减：zN region 禁区 + H 堆叠 WAW + 站点数下限 4/任务）、主线 3
  （手动 flag 协议：单 flag +36%，检查开销临界份额 ~3%，渐进触发承重）、
  主线 4（stream 快路径无墙收益）。
- 终态 = R4 终态（md5 58c43db00e485cf4ceb1b7e85c5a037f 校验一致），
  官方 1.417-1.421ms，精度 --level all 全量 PASS（r5_final_precision.log，
  exit=0）。
- 连续 4 轮无净提升 → 触发中止条件"主线 1-4 穷尽且连续 2 轮无净提升"。
- **[EXPERT_MODE_REQUIRED]**：Developer 模式内杠杆穷尽；剩余 ~100us 全部
  为 AUTO_SYNC 协议架构地板（每 op 同步指令流 + relay 站点结构 + 边界
  配对，~1.2ms 的 93%）；唯一继续路径 = TL_ASCEND_AUTO_SYNC=False 全手动
  同步重写（Expert 级，模式红线需用户授权），预估设备侧 -300~-600us
  （不确定性大，见 final_summary R5 章节论证）。已尝试优化方案数 ≥ 6
  （编译器旋钮×3、单 flag、融合分析、stream 快路径），满足 [DESIGN_ERROR]
  前置条件——本判定为架构级而非实现级缺陷：地板由 pass 的名字级依赖分析
  与每 op 屏障插入策略结构性决定。
