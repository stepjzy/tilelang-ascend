
## Phase 2R r1 — 2026-11-09（画像驱动调优：gather 常量化，未采纳）

- mode: perf_tuning（Phase 2R 轮次 1，分设环境远程执行）
- classification: 优化低于门槛（不采纳），沙箱已回滚，候选存档

### 画像轮（baseline msprof 双口径）
- 8 核逐采 OPPROF + 应用级时间线（120 launch 位置映射，与 do_bench 逐核吻合 <3%）。
- **核心发现：AIV scalar pipe 占 59~99% 是全部 8 核最终瓶颈**；带宽仅 13~27%
  （k1 345GB/s / u 166GB/s vs 实测上限 1263GB/s）——**证伪 Stage 3 的"带宽近饱和"**。
- Stage 3 broadcast 化（-9.21ms）有效的根因由此解释：削的正是 scalar 指令条数。
- 产物：perf_tuning/fla_wy_fast_prof/baseline/（app 75MB + opprof 8 核 + summaries）。
- 注意：8 个内核编译后同名 main_kernel，msprof op 必须单核入口模式采集。

### r1 消融（beta gather 链 → host 预排 [B,nt,H,64] + 单次 span 拷贝）
- 精度：--level all 全绿（含 L2 7/7 拒绝、Boundary 行为与基线一致）。
- A/B：11.657 → 11.463ms（**−1.66%**，<3% 门槛 → 不采纳）；TaskDur 复核 −0.9~−2.7%。
- **证伪**：消除 gather 链后 scalar share 仍 58~95%（u AIV/blk 仅 −0.22us）——
  gather 链非 scalar 主体。
- **归因收口**：剩余 scalar = 随 chunk 增长的拆分链/staging/描述符开销
  （工具链 fp32 gemm 与 cast-on-copy 损坏所迫，issue ①③）。
- **连锁取消**候选"多 chunk/block 摊薄"：固定 prologue scalar 实测仅 ~0.2-0.5us/block，
  摊薄上限 <5%，不值工具链风险。
- **穷尽判定成立**：全部剩余瓶颈归因到上游 issue ①③⑥；11.66ms = 目标 8.5×，物理不可达。
- changes: perf_tuning/fla_wy_fast_prof_r1_tuning.md（完整轮次记录）、
  history_version/fla_wy_fast_impl_prof_r1_gather.py（候选存档）、
  perf_tuning/fla_wy_fast_prof/{baseline,iter_r1_gather}/（双轮画像产物）

## Issue 复检轮 — 2026-11-12（#1924 被 reviewer 证伪后的全量自查）

- 背景：#1924（reduce clear=False）被 wangzhao-11a 复核发现最小复现的**参考值笔误**
  （(M,) vs (M,N) 广播陷阱，M=N=64 使其静默通过）。据此对已提交的 6+1 条 issue 全量复检。
- 方法：v2 复现全部加形状断言 + 正确参考值重跑；可疑项加对照组隔离（v4/v5）。
- 结论：
  - #1921 fp32 gemm：✅ 复现（err=31.24，shape 断言下）——保留
  - #1922 extents：✅ ext1 崩溃 + ext3 slice2 全零均复现——保留
  - #1923 cast-on-copy：✅ C++ 编译错复现（无数值成分）——保留
  - #1924 reduce：❌ 修正参考值后 serial/pipelined 全 MATCH（本机 910B3 独立证实
    reviewer 结论）——**撤回**
  - #1925 alias：⚠️ **声明有效但原帖复现有缺陷**（[1,M,1] 视图读混杂子 32B DMA 坑，
    且 flags 对照组也失败=x[0] 复制填充）。v5 干净对照（同 storage 双参数、直接行读）：
    无 flag 0/10 全读旧值，有 flag 10/10 精确——竞争证实，**需评论更正复现**
  - #1926 multihop：✅ 挂死复现（本轮 150s 未完成 + 原始会话 exit=124 干净捕获）——保留
  - #1927 JIT：已修（上游 #1948）——n/a
- 教训入库：① 复现脚本的一切 out/ref 比较前必须 assert shape 相等（广播陷阱）；
  ② 涉及已知不可靠路径（子 32B 行 DMA）的复现不具归因效力，对照组必须能独立通过。
- 产物：journal fla_wy_fast/issue_repro/ 下的 fla_wy_fast_repro_v2_recheck.py / _v4_alias_arms.py / _v5_alias_definitive.py（2026-11-13 归档）


## 标记值实验（重建版）— 2026-11-13（bug A 复现尝试 + 实验方法示范）

- 背景：用户询问开发期"标记值实验"的代码写法。原探针为一次性调试代码未归档（结论在
  Attempt 1 条目中），本轮按原方法论重建独立版，兼作伞形草案第 3 条（散列地址重写，未提交）
  的最小复现尝试。
- 方法：marker = cid*64 + i（float32 ≤2^24 整数精确无损）；期望由位置正推、观察由数值
  反解，落点-来源对照；一切比较前断言形状。
- 三臂结果（远程 910B3 / CANN 9.2.0 / repo 463c234c）：
  - ARM1 现算源[64]：1023/1024 错（读到 0.0 / 0.5=half 缓冲内容 / 杂讯——源都取错）
  - ARM2 载入源[64]（对照）：**1023/1024 也错**
  - ARM3 载入源[64,1]（对照）：960/1024 错（仅 1 个 block 的 64 位置正确）
- 结论：
  1. **bug A 最小复现未达成**：对照组同样失败 → 无法归因到"现算源"特性；跨步散列
     （→[B,S,H] 中维切片）在最小形态下两种源都静默损坏（疑与子 32B 跨步问题同族）。
     印证伞形"无法最小化"判断；bug A 唯一有效证据仍是开发期真实算子内的实验。
  2. 方法示范有效——包括当场抓出实验者自己的 out_idx 索引错误（返回形状断言暴露），
     meta 证明"值对不上位置即有错"检查的威力。
  3. 方法论闭环：对照组必须独立通过才具归因效力（#1924/#1925 教训本轮自用）。
- 产物：journal fla_wy_fast/issue_repro/ 下 marker_scatter_exp/_v2 与结果 md。
