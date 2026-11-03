# fla_chunk_inter R2 optimization log (skill Part A + ORDER-PLAN, Developer-mode-only round)

R2 约束（用户检查点 2026-11-02）：仅 Developer 模式继续；禁止 Expert 原语（显式
alloc_L1/ub/L0A/L0B/L0C + T.Scope + 手动 set_flag/wait_flag 均不允许——注：T.set_cross_flag /
wait_cross_flag 为 R1 已采纳的 Developer 模式内 API，非 Expert 专属，予以保留）。

前置证据（R1 final_summary §4 地板核算 + msprof_output_final）：
- AIV mte2 4032us@47GB/s/核（3.93GB 装载，V 侧墙）
- AIV vec 3042us（拆分链 24 op + 门控链 20 op/任务）
- AIV mte3 2455us（10 中继写 + 3 输出写）
- AIC busy ~5.9ms，wait_id0 8.8ms（76% 空闲等 V）
- v/do/dv 装载在 bk 兄弟任务间 2× 冗余（240KB/任务中 96KB 无 DK 轴，NK=2 任务各读一遍）

## Part A 优化点清单（R2 候选）

[#R2-1] NK-merge 任务重构造（v/do/dv 单次装载 + 左中继单产 + L1 A 操作数跨 bk 复用）
（参考: performance-antipatterns.md L63-127 launch core 数/任务形态 + L272-283 tile size/任务数）：
**适用** — GRID 16384→8192 任务（每任务 2× 粒度），消除 v/do/dv 的跨 bk 冗余读
（786MB V-mte2 流量）、左操作数拆分链/中继写（各 -50%/任务对）、C 侧左中继读（-50%/任务对）。
全 Developer 模式（alloc_shared/fragment + 显式 workspace 中继 + AUTO_CV_SYNC 按名配对）。
预算核算：UB 缓冲集合不变（+dgkn_ring 1KB）；L1 10×16KB 不变；L0C 3×16KB 不变；
workspace 足迹 ~23.6MB < L2 192MB。NK=1 时语句级退化为 R1 形态（回退安全性）。

[#R2-2] threads=2（40 AIV 唤醒）重审：**不适用（源码级封堵）** — R2 通读
src/transform/ascend_vid_reduction.cc（L172-260 Analyze/ModifyExtents/ModifyBufferLoadIndices）：
threads=2 将全部 UB buffer 第 0 维减半并在 GM 访问的 (gm_dims-ub_dims) 轴加 vid 偏移。
produce 侧 h/dh [bDK,DV] 沿 DK 轴切半且中继写落到 ws 第 (NSLOT/2) 槽位错位，Cube gemm
需要的整 [bDK,DV] B 操作数无法由单 vid 提供；drain 侧 1-D [bDK] 门控缓冲（gn/gnexp/pp）
沿 DK 切半与 2-D [BS,bDK] 沿 S 切半轴不一致（broadcast src/dst 尺寸失配）。无 Developer
模式表述可绕过（该 pass 无 per-buffer 免切选项，buffers_skip_vid_reduction_ 仅限 GM 索引
含 UB load 的 gather 形态）。与 R1 iter3 实证崩溃（ratio 0.05-0.54）互为印证。**方向闭合**。

[#R2-3] grid=40（CORE_EFF 20→40 唤醒 20 空闲 AIV）：**不适用（launch 形态封堵）** —
生成代码 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)（codegen_ascend.cc L858）：
每 block = 1 AIC + 1 AIV 绑定组，910B3 20 AIC → 最多 20 并发组；grid=40 为两波串行
（每波仍 20 AIV），无 AIV 增益且加 2× 流水填充/排空。40 AIV 唯一入口是 MIX_AIC_1_2
（= threads=2，#R2-2 已封堵）。**方向闭合**。

[#R2-4] bDK=128：**不适用** — DESIGN 10.4-B：UB 声明 ~352KB 远超 196352B 硬上限（h/dh
fp32+f16 暂存随 bDK 线性放大），死刑（家族 R4 4-chunk 段错误实证同类）。

[#R2-5] PASS_CONFIGS 编译旋钮补测：**候选（低优先级，#R2-1 后视新画像再定）** — R1 "旋钮无效"
结论移植自 fla_chunk_o 档案；本算子未逐一实测。方向 5。实施前先列 PassConfigKey 全集。

[#R2-6] dq/dk 结果中继读消除（epilogue 直消费）：**不适用（代数封堵）** — dq/dk 门控
2^g[t,k] 双下标元素级（g 为 [BS,bDK]），无法折进 gemm 操作数；exp 无 Cube 侧后端 →
门控必在 V 侧读回结果。fp16 结果中继（省 262MB）精度死（|dq_raw|~O(30) → fp16 ulp
1.6e-2 >> atol 1.53e-5）。

[#R2-7] 2^gn 折进 dh 操作数（消除 drain 1-D 缓冲的候选路径）：**暂缓** — dh'=dh*2^gn 动态
范围风险（gn∈[-20,0] 时 dh' 下探 fp16 次正规区，hi/lo 相对精度崩坏）＋该路径仅服务
threads=2（已闭合）。

[ORDER-PLAN] 实施顺序：
1. [#R2-1] NK-merge — 前置依赖: [无] — 理由: 唯一同时削 V-mte2/V-vec/V-mte3/C-mte2 的
   结构杠杆（-20% 量级建模），全部其他方向被封堵或依赖其新画像
2. [#R2-5] pass_configs 补测 — 前置依赖: [#R2-1] — 理由: 旋钮效果依赖采纳后的流水形态；
   先 msprof 新画像定位新墙再决定是否值得测

---

## 结果记录（[RESULT] 汇总，R2 战役终态）

[RESULT-#R2-1] NK-merge 任务重构造 | 精度: pass（L0 一次过、逐位一致；--level all 51 PASS） | 性能: 11968.4→9752.0us（同会话 A/B） | 对比: **-18.5%** → 采纳
[RESULT-#R2-flag] 1-late flag 配对 + head-set 预充 | 精度: pass（无死锁） | 性能: 9796.7→6919.4us | 对比: **-29.4%** → 采纳
[RESULT-#R2-3] drain 门控链削减（ml2d 提升 + ln2 预折） | 精度: pass（dk/dg 舍入漂移在预算内，valrange 逐位一致） | 性能: 6981.5→6727.6us | 对比: **-3.6%** → 采纳
[RESULT-#R2-4] do/v/dv 装载批处理 | 精度: pass（逐位） | 性能: 合并 5 配对 -2.55%（5/5 方向为正） | 对比: < 3% 噪声阈值 → **回滚**（规则优先）
[RESULT-#R2-5] 左链前移+批处理组合 | 精度: pass（逐位） | 性能: +1.9%/+1.1%（两会话） | 对比: 劣化 → **回滚**

R2 终态：**6777.0/6755.8/6767.4 us（do_bench 3 次）/ 6534.3 us（msprof 单_launch）**，
vs R1 终态 12028 = **-43.8%**；目标 3400us 未达（1.99×）。
新地板核算与 R2 封堵清单见 fla_chunk_inter_final_summary.md §3。
