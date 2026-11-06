# fla_chunk_inter Stage 3 性能调优日志（第二轮，分设环境 DevEnvC_DCbyi_npu）

> 第二轮战役（2026-11-05 起）。计时口径：do_bench（warmup=20/rep=10/mean/仅设备侧 NPU 事件）；
> msprof op 单 launch（--warm-up=10 --launch-count=1，OpBasicInfo.csv Task Duration）交叉验证。
> 基线锚点：Stage 2 交付 6810.9us（复测 6833.4，±0.2%）；目标 <3400us。
> 上轮（第一轮，合设环境）领域知识：R4 终态 6868.3us；六项限制清单；已证伪路线不盲试。

## Iteration 0 — 2026-11-05T11:15+08:00（基线画像轮，双口径采集）
- bottleneck_type: transfer（V 侧串行链）+ other（UB 196KB 容量约束）
- optimization: 无实现变更；OPPROF（CSV 口径）+ 应用级 msprof（Insight 口径）基线采集
- baseline_time: 6810.9 us（Stage 2 档案 do_bench）/ 6762.2 us（本轮 OPPROF 单 launch）
- candidate_time: n/a
- improvement: n/a
- precision: pass（Stage 2 全量 44/44 承继）
- adopted: n/a（画像轮）
- 上轮地板主张复测（全部成立）：
  - 满频 1800/1800；20 核均衡 6235~6759us（尾核 striped 效应）
  - V-mte2 active_bw 52.1GB/s（上轮界 47-54）；mte2 字节地板 416.3KB/task（精确吻合）
  - V 三引擎 mte2 3133 + vec 2275 + mte3 1473 ≈ 6881 ≈ wall 6744（UB 串行链签名；
    aiv_scalar_vector_stall 5822us）
  - C busy 5216us（12.7us/task，77%）；C 等 V 中继 scalar_wait_id0 2159us
  - HBM 唯一 IO 3.758GB/6762.2us = 556.4GB/s = 44.1%
- 源码复核（本轮 clone@463c234c）：MIX 形态仍仅 1_1/1_2（限制 4 成立）；vid 折半集仍不含
  tl.ascend_broadcast（限制 1 成立，threads=2 仍封闭）；T.tile.sub 同 dtype 约束（split 4-op 封堵）
- 产物：fla_chunk_inter_r2_opprof_iter0/（OPPROF CSV）、fla_chunk_inter_r2_prof_baseline.tar.gz
  （应用级 PROF）、fla_chunk_inter_r2_baseline_iter0.json、fla_chunk_inter_r2_probe.py（本轮探针）
- 探针事故记录：首版探针 sys.path 只加了 perf_tuning/ 自身目录 → ModuleNotFoundError；
  修复为加 op 根目录后成功（OPPROF 自检通过：OpBasicInfo.csv 存在）

## Iteration 1 — 2026-11-05T11:35+08:00（do/v/dv 装载批处理，采纳）
- bottleneck_type: transfer（V-mte2 装载流被 split 插入的屏障切碎）
- optimization: 左操作数（do/v/dv）3 个 GM 装载连发（mte2 突发）+ split 链集中后置
  （上轮 iter4 家族精确复刻，独立环境复测）
- baseline_time: 6864.7 us（本会话 A/B A 侧 3 组 mean：6852.6/6867.3/6874.3）
- candidate_time: 6699.8 us（A/B B 侧 3 组 mean：6695.7/6691.6/6712.2）
- improvement: **-2.40%（A/B 配对，3/3 方向为正）**
- precision: pass（--level l0 即时 + --level all 全量：Test Passed! exit 0；Boundary WARN 集与
  Stage 2 基线逐值一致——bnd_g_extreme dk/dg + bnd_operand_over65504 ×4，DESIGN §10.3 预期）
- adopted: **yes**（证据条款采纳：幅度 <3% 但跨环境 8/8 方向一致【上轮合设 5/5 -2.55% +
  本轮分设 3/3 -2.40%】，本轮 A/B 区间完全不交叠 [6852.6,6874.3] vs [6691.6,6712.2]
  （间隔 140.4us，组内散布 ≤0.32%，差距 ~7σ+）——对齐上轮 R4-3/R4-4 的证据采纳先例）
- 采纳态锚点（官方口径 3 次独立）：6730.3 / 6738.5 / 6741.4 us（mean **6736.7**）
  vs Stage 2 档案 6810.9 = -1.09%；OPPROF 单 launch 6502.7us（-3.84% vs 6762.2）
- 归因（iter1 OPPROF，block1 稳态）：
  - V 侧 aiv_mte2_stall 775.7 → **9.7us**（装载流连续化，barrier 等待消失）
  - V 侧 mte2 active_bw 52.1 → 57.0GB/s；aiv_scalar_vector_stall 5822 → 5181us
  - C 侧 wait_id5/id9 -292/-327us；两侧 wall 同步 -307/-308us
- rollback_reason: n/a
- next_hint: iter1 后 C 侧最大等待 = scalar_wait_id0 2230us（C 等 do_hi relay——do split 在
  produce 右链之后）。iter2 = do-only 前移（do split/relay 移到右链前；v/dv 保持右链后。
  上轮 iter5 证伪的是全左链前移；do-only 最小重排未试过，v/dv 的 C 消费在 dq 组后有到达余量）

## Iteration 2 — 2026-11-05T12:10+08:00（do-only 前移，回滚=不采纳）
- bottleneck_type: transfer（假设：C 等 do relay 的 scalar_wait_id0 为可回收墙时）→ 实测证伪：
  C 侧等待为**吸收型 slack**（见归因）
- optimization: do 装载+split+relay 前移到 produce 右链之前（v/dv 装载批处理保持右链后；
  bDK=64 时左 split 复用 th_hi/th_lo scratch 控 UB 预算——三段 liveness 单块复用）
- 候选过程事故（上会话，本会话承继验证）：第 1 版独立 t_* scratch → TVMError
  "Memory allocation failed for: th_hi required: 32768, new memory available: 15616"（t_*/th_*
  liveness 交叠爆 196KB UB，实锤限制 2 razor 边界）；第 2 版语句级 if/else 选 scratch →
  TVM Script parser 块级作用域 "Undefined variable: lt_hi"；第 3 版三元表达式（现行候选 92a3a78c）
  编译通过
- baseline_time: A(iter1 采纳态) 本会话 3 组 mean：6643.2us（6665.0/6629.4/6635.3）
- candidate_time: B(do-first) 本会话 3 组 mean：6928.0us（6924.3/6930.9/6928.8）
- improvement: **+4.29%（劣化）**；跨会话合并 6/6 轮方向一致【本会话 iter2d：+3.89/+4.55/+4.42；
  上会话 iter2c：+4.03/+4.60/+4.54】，区间 [6622.0,6665.0] vs [6903.7,6930.9] 完全不交叠
- precision: 未触发采纳门禁（性能劣化先判回滚；候选仅存在于 perf_tuning/ 沙箱，
  主实现 fla_chunk_inter.py 全程未改动 = iter1 采纳态 738a0980）
- adopted: **no**（6/6 轮 +3.9~+4.6%，远超 3% 噪声阈与任何证据条款）
- rollback_reason: 性能劣化。归因（画像 + 结构双证）：
  1. do split 前移破坏 iter1 建立的左三装载 mte2 突发（do/v/dv 连发）——do 单发后
     h/dh 装载与 split vec 交错，mte2 装载流重新碎片化（iter1 修复项回退）；
  2. 前提本身不成立：iter1 OPPROF 20 核复核 aic_wall 均值 6452.1 vs aiv_wall 6448.9
     （逐核差 +2~3us 锁步），C busy（cube 937+mte1 1233+mte2 1440+fixpipe 979+scalar 1257）
     ≈5.85ms < V 循环节拍 15.7us/任务——C 的 id0 等待 2212us 为吸收型 slack，非墙时；
     V 串行链（mte2 2876+vec 2237+mte3 1448=6561≈wall 6449）才是墙
- 家族闭合结论：produce 重排家族就此全部证伪——全左链前移 +1.9%（R1 iter5）、do-only 前移
  +3.9~+4.6%（本轮 ×6）、部分 [do+v] 突发+dv 前移 +4.3%（R1 iter6b）、drain 装载前移精度崩溃
  （R1 iter6a planner 别名）；批处理装载+split 集中后置（iter1 形态）为该家族实测最优
- next_hint: C 消费序重排（右中继先读 + dgm 组先算）为重排家族最后未试变体——但稳态分析
  预测亚噪声（C 完成时点被最晚左中继 dvn 到达 + 尾部 gemm/fixpipe 锚定，与当前序等价）；
  iter3 实证之。注意 ascend_combinecv.cc L378-391 同步点计数相等硬约束：C 侧每 workspace
  读语句数必须 = V 侧写语句数（LOG(FATAL)），双读中继方案编译期即封堵 → 3-D L1 单读形态

## R2 反模式清单复核（iter2 后，2026-11-05T12:10+08:00）

对照 `.agents/skills/tilelang-perf-optimization/references/performance-antipatterns.md` 逐项核查
（R1 终态核查项承继，kernel 结构同族 + iter1 批处理装载）：

| 反模式项 | 判定 |
|---|---|
| launch core 数（关注项 A/B） | ✓ fixed-core(20) + 核内任务循环（R1 iter2 采纳配方承继） |
| V 内逐元素/逐行 for loop | ✓ 无（全 tile 级；bDK=32 行链为 128B 行窗物理约束保留，非本 shape 路径） |
| 冗余全局同步 | ✓ 手动 flag 1-late slot 10 + head-set 预充；AUTO_CV_SYNC 10 对按名配对 |
| mul+add 未融合 | ⚠️ 有意保留分离形态（R3 iter2 实证 mul_add_dst ~1% 确定性偏差，CANN
  kernel_operator_vec_binary_intf.h:249-250 受限标注——skill 推荐写法与实测冲突，精度优先） |
| tile size 过小 | ✓ [64,64]/[64,128] 为 UB 界内最大（iter2 事故反向实锤：+32KB 即爆） |
| 混合算子未开 CV overlap | ✓ AUTO_CV_COMBINE=True（MIX_AIC_1_1） |
| 纯 AIV memory bound 未做流水/双 buffer | ⚠️ 有意保留串行形态：UB 196KB 容量强制逐操作数链式
  （限制 2）；iter1 批处理装载已回收装载流连续化收益（-2.4%），iter2 复测前移家族全部劣化 |
| 多行 tile 循环内逐块归约 | ✓ 无（每任务单发 6 层折半树） |
| 正交轴串行化 | ✓ 无标量扫描（bDK=64 转置+row_expand 路径） |
| AUTO_CV_COMBINE 误分核 | ✓ 有真实 C 侧（24 gemm/任务） |
| Vector mask reuse fallback | ✓ 未使用 |

## Iteration 3 — 2026-11-05T12:35+08:00（C 消费序重排：右中继先读 + dgm 组先算，回滚=不采纳）
- bottleneck_type: transfer（假设：C 早段 id0 空转为可回收墙时）→ 实测证伪：C 完成时点
  被最晚左中继 dvn 到达 + 尾部 gemm/fixpipe 锚定，早段空转为吸收型 slack
- optimization: C 侧两遍消费重排——pass1 右中继（produce 最早落）先读 + 右操作数专属 dgm
  组先算并先写 ddr dgm 行（[NK,bDK,DV] L1 slab 免重读，保同步点计数 1:1）；左中继后读；
  pass2 dq/dk/dw 组 + dq/dk 行写入。produce 侧完全不动（iter1 批处理装载保持）
- 设计合规预检（本轮新增方法论）：ascend_combinecv.cc L378-391 同步点计数相等硬约束
  （C 侧每 workspace 读语句数 = V 侧写语句数，违反即 LOG(FATAL)）→ 双读中继方案编译期
  封堵；gemm_v0 文档确认 BufferRegion 操作数（区域切片 2-D 过 leading-dim 断言）；
  flag id 按名字典序分配（ws_1→0…ws_9→9），语句序重排不扰动 flag 结构；L1 预算
  224KB ≪ 512KB
- baseline_time: A(iter1 采纳态) 3 组 mean：6633.7us（6643.7/6628.2/6629.2）
- candidate_time: B(iter3_c_first) 3 组 mean：6680.1us（6693.6/6678.8/6667.9）
- improvement: +0.70%（劣化，3/3 方向一致；区间 [6628.2,6643.7] vs [6667.9,6693.6] 不交叠）
- precision: SPOT PASS（A/B 脚本内嵌 4 输出对拍 bench shape，B 与 A 全 allclose——重排家族
  R1 iter6a 精度崩溃前科下本轮新增的廉价自证；未触发 --level 门禁因性能先判回滚）
- adopted: no（无提升即回滚；候选仅存 perf_tuning/，主实现全程未动 = 738a0980）
- rollback_reason: 性能劣化 +0.6~0.8%。归因：dgm 行 fixpipe 前移 + pass2 尾部
  （dvn 到达后仍需右操作数 gemm + dq/dk 行写入）比原序更长——C 完成时点反而推后
  ~0.45us/任务，与 id10 等待（V 等 C 的 1-late flag）同向恶化
- 家族闭合结论：C 消费序重排证伪。结合 iter2：**重排家族（produce 序 + C 消费序）双向
  全部证伪**——稳态分析成立：C busy ≈5.85ms < V 节拍，C 空转为吸收型 slack；
  kernel 墙 = 双墙锁步（aic 6452.1 / aiv 6448.9 均值，逐核差 +2~3us）
- next_hint: 最后一个未试重排变体 = 左 split 逆序（dvn→v→do）+ C 消费同逆序（dw→dk→dq）：
  使 C 尾部锚点从"最晚中继 dvn + dw/dq 尾链"改为"最晚中继 do + dq 短尾"，理论上限
  ~1.1%（id10 消除后墙仍被 C busy 6434 锚定）。预期亚噪声；iter4 实证之，
  若 <3% 无提升即第 3 连击收官

## Iteration 4 — 2026-11-05T18:28+08:00（左 split 逆序 + C 消费同逆序，回滚=不采纳；3 连击收官）
- bottleneck_type: sync（假设：C 完成时点被"最晚到达中继 + 其后剩余 C 工作"锚定，双向逆序
  可缩短 C 尾部 → aiv_wait_id10 部分回收）→ 实测证伪：稳态锁步使重排不可见
- optimization: 左 split 链逆序（dvn→v→do；三装载突发保持 iter1 形态不动，仅移动三个
  split 块的顺序）+ C 消费同逆序（左中继 L1 读序 dvn→v→do；gemm 组序 dw→dk→dq、
  dgm 末位；dw 直存随 dw 组前移；ddr 行带布局不变）。同步点计数逐 workspace 保持 1:1
  （每 bk 仍 ws_3/4/7/8 各 1 读；左中继 6 读不变）；flag id 按名分配不受语句序扰动
- 候选：perf_tuning/fla_chunk_inter_r2_iter4_candidate.py（md5 e604ec0e，上会话 12:26
  起草、本会话核验结构后原样执行）；改前快照 history_version/fla_chunk_inter_perf_r2_iter4_before.py
  （= iter1 采纳态 738a0980）
- baseline_time: A(iter1 采纳态) 本会话 3 组 mean：6711.0us（6710.9/6723.2/6698.8）
- candidate_time: B(iter4_reversed) 本会话 3 组 mean：6710.8us（6707.3/6703.2/6721.9）
- improvement: **-0.00%（亚噪声；3 轮方向 -0.05/-0.30/+0.34 不一致，区间
  [6698.8,6723.2] 与 [6703.2,6721.9] 完全交叠）**
- precision: SPOT PASS（A/B 脚本内嵌 4 输出对拍 bench shape，B 与 A 全 allclose，
  rtol=1e-4/atol=1e-3；重排家族精度崩溃前科下的标准自证；未触发 --level 门禁因
  性能先判回滚，主实现全程未动 = 738a0980）
- adopted: **no**（<3% 噪声阈；无提升即回滚，候选仅存 perf_tuning/ 沙箱）
- rollback_reason: 性能无变化。归因（稳态分析精确命中）：iter1 形态下 C 早段空转
  （aic_wait_id0 ~2.2-2.4ms）与 V 尾部等待（aiv_wait_id10 ~0.2-0.6ms）均为耦合
  锁步的吸收型 slack——C 侧任何组序/到达序重排都被对侧墙吸收，墙由两侧 busy
  联合锚定（aic 6452.1 / aiv 6448.9，逐核差 +2~4us）；理论上限 ~1.1% 中实际可
  回收部分 ≈ 0。A 侧跨会话漂移（6634→6643→6711，~1%）远大于本候选效应，
  会话内配对 A/B 判定有效
- **3 连击成立（iter2 +4.29% / iter3 +0.70% / iter4 ±0.00%），R2 战役收官**
- 家族终局闭合：重排家族 6 变体全谱证伪——①全左链前移 +1.9%（R1 iter5）、
  ②do-only 前移 +3.9~4.6%（R2 iter2 ×6 轮）、③[do+v] 突发+dv 前移 +4.3%（R1 iter6b）、
  ④drain 装载前移精度崩溃（R1 iter6a）、⑤C 消费序右中继+dgm 先 +0.70%（R2 iter3）、
  ⑥双向逆序 -0.00%（本轮）。**批处理装载+split 集中后置（iter1 形态）为全家族实测
  最优且已采纳；语序空间耗尽**
- 运行记录：job 01a1ab40（request-id prof-fla_chunk_inter-r2-iter4，58s exit 0）；
  日志存档 perf_tuning/fla_chunk_inter_r2_ab_iter4_run.log
- next_hint: 无（收官）。终态 = iter1 采纳态 738a0980；终态双口径与地板核算见下方终态节

---

# 终态节 — 2026-11-05T18:39+08:00（R2 战役收官）

## 终态判定

**终态 kernel = iter1 采纳态（md5 738a0980）**。iter2/3/4 全部回滚，连续无提升 = 3（iter2 +4.29% /
iter3 +0.70% / iter4 ±0.00%），3 连击中止条件成立，R2 战役收官。主实现全程未被非采纳候选触碰；
改前快照链完整（history_version/fla_chunk_inter_perf_r2_iter{1,2,4}_before.py）。

## 终态双口径采集（收官 job 01a1ab46，request-id prof-fla_chunk_inter-r2-final，289s exit 0）

| 口径 | 数值 | 说明 |
|---|---|---|
| do_bench 官方口径 ×3 独立进程 | **6734.4 / 6749.8 / 6741.3 us（mean 6741.8）** | warmup=20/rep=10/mean/仅设备侧 NPU 事件；与 iter1 采纳锚点 6736.7 复现吻合（+0.08%） |
| OPPROF 单 launch（CSV 口径） | **6502.7 us**（Task Duration） | **复用 iter1 轮采集**（终态=iter1 采纳态且无新采纳，kernel md5 未变；原产物 fla_chunk_inter_r2_opprof_iter1/ 已归档） |
| 应用级 PROF（Insight 口径） | 稳态 11 次 launch 6601.9~6709.3 us，**mean 6657.2** | **本轮新采**：`env -u ASCEND_RT_VISIBLE_DEVICES msprof --output=... --application='python fla_chunk_inter_r2_probe.py' --task-time=on --aic-mode=task-based --aic-metrics=PipeUtilization`；12 launch（首launch 6811.7 含采集预热，弃用）；自检 op_summary 13 行非空；归档 **fla_chunk_inter_r2_prof_final.tar.gz**（79531B，PROF_000001_20261105183906822_00028753QPEANCCO） |

三口径基线→终态方向一致性核验：do_bench 6810.9→6741.8（-1.03%）；OPPROF 6762.2→6502.7（-3.84%）；
应用级 PROF 稳态 6775.8→6657.2（-1.75%）——iter1 采纳的装载批处理收益在三口径下全部为正。

## 终态精度门禁（--level all，收官 job STEP1）

**Test Passed!（exit 0）**：L0 8/8 + L1 36/36 全 [PRECISION_PASS]（ratio 全 1.000000，最低
0.999939 = l1_valrange_l dk，>0.99 帽）；L2 13/13 正确拒绝；Boundary 17 输出 PASS + 6 WARN
（bnd_g_extreme dk/dg + bnd_operand_over65504 ×4，与 Stage 2 基线逐值一致，DESIGN §10.3 预期）。
4 输出全量 max_abs 上界 3.906e-03 ≪ 1e-2 帽。

## 地板核算表（新环境终版数字；iter1 采纳态 OPPROF，20 核均值，满频 1800/1800）

**A. 逐 pipe busy（us/核；每核 409.6 任务均值；wall = aic 6452.1 / aiv 6448.9）**

| 侧 | pipe | busy | 占本侧 wall | 备注 |
|---|---|---|---|---|
| AIC (C) | cube | 937.1 | 14.5% | 24 gemm/task fp16 hi/lo 三连 |
| AIC (C) | mte1 | 1233.0 | 19.1% | L0A/L0B staging |
| AIC (C) | mte2 | 1258.9 | 19.5% | 中继 GM→L1 读 |
| AIC (C) | fixpipe | 879.0 | 13.6% | dw 直存 + ddr 中继写 |
| AIC (C) | scalar | 1199.1 | 18.6% | 地址计算等 |
| **AIC 合计** | | **5507.0** | **85.3%** | 13.44 us/task；block0=5845.9（尾核 striped 效应上沿） |
| AIV (V) | mte2 | 2880.4 | 44.7% | 装载流（iter1 后连续化，mte2 stall 23.6us≈0） |
| AIV (V) | vec | 2237.5 | 34.7% | split/copy/exp/mul UB 链 |
| AIV (V) | mte3 | 1452.8 | 22.5% | 中继写 + dq/dk/dg 写出 |
| **AIV 串行链合计** | | **6570.7** | **101.9%** | 16.04 us/task；≈wall（差 1.9% 为引擎级交叠/量化）；V scalar 4182.8 中 vec_stall 5338.1 为主（UB 串行链签名） |

**B. mte2 引擎界与字节地板**
- V-mte2 字节量：**416.3KB/task**（装载 do/v/dv 96KB + h/dh 右操作数 128KB + drain 重读 ddr 96KB +
  drain 重读 q/k/g 96.5KB，全部结构必需）× 8192 task / 20 核 = 170.6MB/核
- 实测 active_bw **56.7GB/s**（iter0 53.4 → iter1 突发装载后 56.7；R1 引擎界实证 47-54GB/s，
  现已贴界上沿）→ mte2 时间由"字节量 × 引擎界"联合锚定 ≈ 2.88ms，**单引擎即占墙 44.7%**
- 即使 vec/mte3 与 mte2 完全重叠（UB 容量不允许），墙仍 ≥ mte2 ~2.9ms

**C. 双墙锁步证据（重排家族全谱证伪的结构性解释）**
- aic_wall 6452.1 / aiv_wall 6448.9（20 核均值）；**逐核 aic−aiv 差全为正：+2.15~+4.34us**
  （排序后 20 对逐一同步移动）
- C busy 5507（85.3%）+ 吸收型等待（aic_wait_id0 2355.4 + id5 242.2 + id9 229.4 + id1/2/6 ~107）
  填满 aic_wall；V 串行链 6571 填满 aiv_wall + 尾部 aiv_wait_id10 409.1
- 实证闭环：iter2（do 前移）、iter3（C 消费序）、iter4（双向逆序）全部无效——两侧墙由耦合锁步
  联合锚定，单侧语句序重排不可移动墙（吸收型 slack 定理）

**D. HBM 唯一 IO 效率**
- 算法唯一 IO **3.758GB**（输入 2.684GB + 输出 1.074GB）/ 6502.7us = **577.7GB/s = 45.8%**
  （vs 1261GB/s 参照，R1/R2 既定口径；iter0 基线 556.4GB/s = 44.1%）
- 引擎级实测总流量 8.547GB（V 读 3.265 + V 写 2.522 + C 读 1.756 + C 写 1.004，含 fp16 操作数
  中继往返与 fp32 ddr 中继）；L2 总命中率 ~97%，pin 级流量低于引擎级
- 纯带宽地板：3.758GB / 1261GB/s = **2980us** ≈ 3400 目标量级——目标隐含"唯一 IO 单遍流 +
  全引擎完美流水 + 中继消除"

**E. vs 3400us 差距分解**
- do_bench 终态 6741.8 vs 3400 = **超 3337.9us = 1.98×**（OPPROF 口径 6502.7 → 1.91×）
- 结构分解（OPPROF 口径，超额 3102.7us）：
  1. V-mte2 2880.4us——字节地板 416.3KB/task 结构必需 + 引擎界 56.7GB/s 已贴 R1 界上沿（不可再快）
  2. V-vec 2237.5us——split/copy/exp/mul 链，UB 196KB 容量强制串行（限制 2，R1 限制清单，架构常数）
  3. V-mte3 1452.8us——中继写 + 算法写出（fp32 gemm_v0 损坏 → fp16 hi/lo 3-gemm 放大 C 侧，
     限制 3；中继字节已最小化 NK-merge + fp16）
  4. C busy 5507us 与 V 锁步——即使 V 侧完美流水（墙→max(2880,2238,1453)≈2880），C 墙 5.5ms
     仍在，需 C 波次化/片上直连（R1 限制 3/4，工具链立项级）
- R1 六项限制清单逐项复验（iter0 源码复核 + 本轮重排家族全谱证伪）：**全部成立**——
  ①threads=2 vid 折半集不含 tl.ascend_broadcast（clone@463c234c 复核）②UB 196KB 容量
  ③C 消费链 GM 中继架构（无片上直连）④MIX 形态仅 1_1/1_2（codegen_ascend.cc:858-860）
  ⑤目标隐含 88% HBM 唯一 IO 效率 > 家族实证最佳 81% ⑥工具链指令/规划器缺陷族
  （mul_add_dst 受限标注、in-place 链崩溃、reduce_2d [64,64] 损坏、planner 别名）

## 战役判定

**目标 <3400us 未达成**：终态 6741.8us（do_bench 官方口径 3 次独立 mean）/ 6502.7us（OPPROF 单
launch）/ 6657.2us（应用级 PROF 稳态均值），**超目标 1.91~1.98×**。根因 = R1 六项限制清单复验
全部成立（工具链/架构限制，非实现层问题）；R2 增量空间（语序重排）已全谱探索并闭合。

## R2 已试清单（iter0-4）

| 轮 | 内容 | A/B 结果（同会话 3+3 do_bench） | 判定 |
|---|---|---|---|
| iter0 | 基线双口径画像（无实现变更） | 6810.9 do_bench 档案 / 6762.2 OPPROF；上轮地板主张全部复测成立 | 画像轮 |
| iter1 | do/v/dv 装载批处理 + split 链集中后置 | 6864.7→6699.8（**-2.40%**，3/3 方向为正，区间不交叠） | **采纳**（证据条款：跨环境 8/8 方向一致） |
| iter2 | do 装载+split+relay 前移（右链前） | 6643.2→6928.0（**+4.29%**，6/6 轮劣化） | 回滚 |
| iter3 | C 消费序重排（右中继先读 + dgm 组先算） | 6633.7→6680.1（**+0.70%**，3/3 劣化） | 回滚 |
| iter4 | 左 split 逆序 + C 消费同逆序 | 6711.0→6710.8（**-0.00%** 亚噪声，方向不一致） | 回滚；**3 连击收官** |

R2 净效果：6810.9 → 6741.8（do_bench 口径 **-1.02%**）；累计采纳 1 项（iter1 装载批处理）。

## R2 产出文件清单（perf_tuning/，fla_chunk_inter_r2_ 前缀）

| 文件 | 内容 |
|---|---|
| fla_chunk_inter_r2_perf_log.md | 本日志（iter0-4 + 终态节） |
| fla_chunk_inter_r2_baseline_iter0.json | iter0 基线画像档案 |
| fla_chunk_inter_r2_probe.py | R2 探针（12 launch，OPPROF/应用级共用） |
| fla_chunk_inter_r2_ab_iter{1,2,3,4}.py | 各轮 A/B 脚本（iter3/4 含内嵌 SPOT 对拍） |
| fla_chunk_inter_r2_iter{1,2,3,4}_candidate.py | 各轮候选（iter1=采纳态副本，其余为沙箱证伪候选） |
| fla_chunk_inter_r2_ab_iter4_run.log | iter4 A/B job 运行记录 |
| fla_chunk_inter_r2_final_run.log | 收官 job 运行记录（精度门禁 + do_bench ×3 + PROF 采集全记录） |
| fla_chunk_inter_r2_opprof_iter{0,1}/ | OPPROF CSV 口径画像（iter0=基线，iter1=终态，**终态复用**） |
| fla_chunk_inter_r2_prof_baseline.tar.gz | 应用级 PROF 基线轮（iter0） |
| **fla_chunk_inter_r2_prof_final.tar.gz** | 应用级 PROF 终态轮（本轮新采） |
| ../history_version/fla_chunk_inter_perf_r2_iter{1,2,4}_before.py | 改前快照链 |
| ../fla_chunk_inter.py | 终态主实现（md5 738a0980 = iter1 采纳态） |
