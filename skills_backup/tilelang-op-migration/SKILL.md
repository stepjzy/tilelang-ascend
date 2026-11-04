---
name: tilelang-op-migration
description: >-
  TileLang-Ascend 算子迁移全流程工作流（Phase 0-7）：从"用户提出要迁移某个算子"开始，
  经主仓开发（永不 commit）、交付（新建 PR 分支与 worktree、只放实现+测试两个文件、
  push 到 origin）、归档（dev-journal 分支按学习价值收纳 DESIGN/debug_log/perf_tuning、
  生成 00_INDEX.md 学习导航、归档 PR 阐述）、PR 文稿交付（用户网页自行提 PR，agent
  永不代提、不做 GitHub 认证）、清理主仓沙箱；调优完成后若性能未达标，立即中断
  并告知用户、待其决策后再继续；归档+清理+文稿交付全部完成即工作结束，回复含
  性能介绍与达标判定的最终交付报告；PR 的提交/合并/分支清理及 journal 的后续
  状态维护均由用户自行处理。
  定义三区架构：主仓开发沙箱（ascendc_pto 基线）/ PR worktree（feat/{op}-pr）/
  归档 worktree（dev-journal orphan 分支）。分设环境（常驻 CPU + 按需 NPU）下，
  CPU 侧为唯一真源（编辑/文档/git），一切 kernel 验证经 cannlab CLI 远程驱动
  NPU 环境执行（exec/job/sync/cp），NPU 只作执行器、永不承载真源。当用户提到：
  迁移算子、把 GPU 算子迁到
  NPU/昇腾、/migrate、算子开发完成要交付/上库/提 PR、新建 PR 分支或 worktree、归档
  学习材料/过程文档/调优记录、清理主仓/沙箱、整理算子目录、询问"过程文档放哪"、
  "PR 里能放什么文件"、"学习材料怎么保留"，或任何算子从需求到 PR 全程流转的操作时，
  必须使用本 skill。单独执行某一段（仅归档/仅清理/仅准备 PR 文稿）也应触发本 skill
  对应 Phase。开发过程的技术执行（设计/实现/测试/调优）由 tilelang-op-orchestrator 及
  其下属 skill 负责，本 skill 负责全程编排、交付、归档与清理。
---

# TileLang 算子迁移工作流（op-migration）

## 三区架构（一切规则的根基）

```
┌─ 主仓 tilelang-ascend ──────────────────────────────────────────┐
│ 分支：ascendc_pto（与 upstream 同步），【永不 commit】           │
│ 职责：开发 / 测试 / 性能调优的沙箱，产物全部未跟踪                │
│ 沙箱：examples_experiment/dsa_sparse_finetune/                   │
│   ├── {op}.py + test_{op}.py          ← 开发文件（交付后清理）    │
│   ├── proto_{op}.yaml                 ← 任务书（归档后清理）      │
│   ├── DESIGN_{op}.md                  ← 设计文档（含被否 rev）    │
│   ├── debug_log_{op}.md               ← 结构化踩坑记录            │
│   ├── perf_tuning/{op}_*              ← 调优结论 + 数据           │
│   ├── history_version/{op}_*          ← 关键迭代快照              │
│   └── ref_gpu/                        ← 共享 GPU 参考（永不清理）  │
├─ PR worktree ../tilelang-{op}-pr ───────────────────────────────┤
│ 分支：feat/{op}-pr（从 upstream/ascendc_pto 新建）               │
│ 职责：【只放 ≤2 个交付文件】（算子实现 + 算子测试），在此 commit+push│
├─ 归档 worktree ../tilelang-dev-journal ─────────────────────────┤
│ 分支：dev-journal（orphan，不含源码历史），push 到 origin 备份    │
│ 职责：学习档案。每算子一目录，00_INDEX.md 是唯一入口              │
└─────────────────────────────────────────────────────────────────┘
```

为什么这样分：主仓 commit 会让后续算子的 PR 带上脏历史（交叉污染）；过程文档进
PR 分支会违反"PR 只收 2 个文件"的上库约束；过程文档留在主仓未跟踪则无版本无备份，
清理时极易误删（sparse_mla_topk 的 35 份文档曾因此差点永久丢失，仅靠 dangling
提交 962b00f1 抢回）。

## 分设环境执行模型（CPU 编辑 / NPU 执行）

分设架构（常驻 CPU + 按需 NPU，2026-11 起）下，三区架构的 git/文档/交付流程
零改动——但**一切"跑 kernel"的动作**（编译、精度验证、性能测试、msprof 画像）
不再本地直跑，统一经 cannlab CLI 远程驱动 NPU 环境：

```
┌─ CPU 环境（常驻，唯一真源）────────────────────────────────┐
│ 主仓沙箱编辑 / 过程文档 / git 三区流转 / PR 文稿               │
└──────────┬──────────────────────────────────▲───────────────┘
      sync 推送（代码/候选版本）        cp 取回（日志/性能/画像产物）
           │                              │
┌─ NPU 执行区（按需启停，纯执行器）──────▼──────────────────────┐
│ 环境名：Phase 1 记录（默认 DevEnvC_DCbyi_npu，1×910B3）        │
│ exec/job 跑测试与画像 / install_ascend.sh 构建产物驻留复用      │
└────────────────────────────────────────────────────────────────┘
```

### 数据流与版本纪律

1. **单向流**：CPU 编辑 → sync → NPU 执行 → cp 取回产物 → CPU 归档进沙箱
   `perf_tuning/`。**NPU 侧禁止修改代码**——两端存储独立，任何 NPU 侧改动都会
   漂移成无主版本。
2. **任务块化**（对抗 NPU 空闲 30 分钟自动关机）：一次开机 = 一个工作块——
   sync 全部待测候选 → 批量 exec/job 跑完 → cp 取回 → 申请关机。分析与改码
   在关机状态下做，下一轮候选备齐再开机。禁止"改一行开一次机"的碎片节奏。
3. **产物即取**：NPU 上的测试日志、性能 json、msprof 产物必须**当轮取回 CPU**
   （停机保留不保证、环境回收必丢）；NPU 侧只允许驻留 build/ 等可再生重物。
4. **基线对账**：每次开机 sync 前记录 CPU 主仓基线 commit（`git rev-parse HEAD`）
   写入当日 debug_log；NPU 无 git 元数据（sync 装载方案）时这是唯一版本凭据。

### cannlab 命令映射（原本地动作 → 分设动作）

| 原（单机时代） | 分设后 |
|---|---|
| `source set_env.sh && python test_{op}.py`（短验证） | `cannlab --output jsonl exec --cwd <NPU项目目录> $NPU_ENV -- bash -lc "source set_env.sh && python test_{op}.py"` |
| 全套测试 / 多轮编译（长任务） | `cannlab --output json job start --cwd <NPU项目目录> $NPU_ENV -- bash -lc "..."`（保存任务 ID，断线不丢；`job wait`/`job logs` 跟踪） |
| 本地 msprof 画像 | 远程 job 跑 msprof → `cannlab cp -r` 取回 prof 数据 → **CPU 侧分析** |
| 本地改文件 | **禁止**——回 CPU 改，重新 sync |
| 取回结果 | `cannlab --output jsonl cp -r $NPU_ENV:<结果目录> <CPU目录>` |

`$NPU_ENV` = Phase 1 记录的 NPU 环境名（环境 ID/devstage_id/唯一名称均可）。
输出模式：一次性结果用 `--output json`，实时流用 `--output jsonl`（全局参数必须
放在子命令之前）。

### 启停与查询纪律（自动启停，2026-11 用户授权放宽）

- **自动 start**：需要执行验证时直接拉起（`cannlab env start $NPU_ENV` →
  `env wait` 至 RUNNING），不逐次询问用户；NPU 中途被平台空闲回收/自动关机时，
  下一次需要自动重新拉起，全程无感知；
- **自动 stop（带产物自检闸门）**：仅当满足以下两条后才允许自动关机——
  ① 本工作块全部验证执行完毕（本轮迁移任务不再有排队的验证需求）；
  ② **产物取回自检通过**（对照红线 11 清单：测试日志/性能 json/msprof 产物
  均已 cp 回 CPU 沙箱）——自检不通过禁止 stop；
- **分析间隙不主动关机**：调优会话内的 CPU 侧分析/改码期间不发起 stop，
  平台 30 分钟空闲自动释放；下次验证自动再拉起；
- **仍需用户确认的例外**：`cannlab env recover`（关机重启会中断环境内运行中的
  任务）与 NPU 处于 EXCEPTION 状态时的任何处置；
- 纯查询用 `cannlab env list` / `env show`（只读不启机）；`exec`/`env inspect`
  类命令做纯查询时必须加 `--no-start`，禁止"看一眼"意外开机；
- 长时间验证一律 job（不占本地连接），**禁止交互式 SSH 长占 NPU**；
- NPU 已 RUNNING 时不得重复 start。

### 调优类 skill 的分设约束注入（下游兜底）

`tilelang-op-perf-optimization` 与 `tilelang-op-profiling` 的 SKILL.md 仍为单机
时代写法（msprof/采集命令本地直跑）。**经 orchestrator / perf-tuner 调用它们
时，其 Step 内一切 msprof / 采集 / kernel 运行命令一律按本节 cannlab 命令映射
执行**（远程 exec/job + prof 产物 cp 取回 CPU 分析），忽略其文内的本地直跑
命令字样；两个 skill 本体的全面分设化改造为独立待办。

## 命名规范（强制统一，禁止自创变体）

| 对象 | 格式 | 正例 | 反例（历史遗留，勿再犯） |
|---|---|---|---|
| PR worktree 目录 | `tilelang-{op}-pr`（op = 完整算子名，连字符） | tilelang-sparse-mla-fwd-pr | tilelang-ascend-pr-sparse-mla-fwd、tilelang-sparse-mla-pr（缩写算子名） |
| PR 分支 | `feat/{op}-pr` | feat/sparse-mla-fwd-pr | feat/sparse-mla-fwd（缺 -pr 后缀） |
| 归档区 | worktree `tilelang-dev-journal` / 分支 `dev-journal` | — | — |
| 沙箱文件前缀 | `{op}_`（下划线） | sparse_mla_fwd.py、DESIGN_sparse_mla_fwd.md | — |

worktree 目录名与分支名只是标签，不影响 Git 功能，但不统一会持续制造认知负担。
**发现不一致时先统一再继续流程**。改名操作参考：
- 目录改名：`git worktree move <old> <new>`；若因含 admin 子模块被拒，手动
  `mv` + 更新 `.git/worktrees/<name>/gitdir` + `git worktree repair` +
  sed 修正各子模块 config 的 `core.worktree` 相对路径。
- 分支改名：`git branch -m <old> <new>` + `git push -u origin <new>` +
  `git push origin --delete <old>`（在开 PR 之前做，成本最低）。


```bash
cd /mnt/workspace/gitCode/skybasejzy/tilelang-ascend
git status -sb                        # 必须在 ascendc_pto 且与 upstream 同步
# dev-journal 不存在时才创建（orphan 分支 = 纯文档分支，不带源码历史）：
git worktree add --orphan -b dev-journal ../tilelang-dev-journal
# journal 根写 .gitignore，防本地产物（.vscode/.env/缓存）混入归档分支：
printf '.vscode/\n*.env\n__pycache__/\n' > ../tilelang-dev-journal/.gitignore
# 注：PR 文稿不进主仓（在 journal worktree 内撰写，见 Phase 4/5）
```

若 `--orphan` 不被支持（git < 2.42），改用空树提交法：
`TREE=$(git mktree </dev/null); C=$(git commit-tree $TREE -m "init dev-journal"); git branch dev-journal $C && git worktree add ../tilelang-dev-journal dev-journal`

## Phase 1 — 需求澄清（禁止自行假设）

参数不全时逐项询问用户，凑齐才进入下一 Phase：

| 参数 | 说明 |
|---|---|
| 算子名称与迁移来源 | `ref_gpu/{op}.py` 是否已有参考实现；没有则先要来源 |
| dtype / shape 范围 | 精度基准与测试覆盖范围 |
| 精度门禁 | matched_ratio 等阈值（如 ratio ≥ 0.99）；分 dtype 容差标准见下方"口径标准" |
| 性能目标 | 基线口径与目标值（如纯 kernel < XXXX us）；计时口径见下方"口径标准" |
| 编程模式 | Developer / Expert / 混合（用户明确指定） |
| 阶段范围 | 全流程（至归档+清理）/ 仅归档 / 仅清理 / 仅准备 PR 文稿 |
| NPU 执行环境 | 分设架构下的 NPU 环境名（`cannlab env list` 查询实际值，默认 `DevEnvC_DCbyi_npu`）；用户未提供时向其确认 |

### 口径标准（精度容差与纯 Kernel 计时的权威依据）

- **精度容差**：分 dtype（fp32 / bf16 / fp16 / fp8 等）的 atol / rtol /
  matched_ratio / max_abs_error_limit 取值以
  [.agents/skills/tilelang-op-test-design/references/precision-standard.md](../tilelang-op-test-design/references/precision-standard.md)
  为权威标准（混合容差 + 双门限，阈值只按 dtype 划分、与算子类别无关）。
  Phase 2 测试设计（`tilelang-op-test-design` skill）据此制定分层容差，PR 精度
  报告的阈值同源；禁止凭感觉或照抄单一默认值定容差。
- **纯 Kernel 时间**：迁移后算子的性能测量按**主仓原 GPU 测试样例**执行——
  benchmark 的 shape / dtype 配置取自主仓该算子示例的测试入口（如 `main()`），
  计时方法对齐其 `do_bench` 口径：预热若干次后多次重复运行、**取平均值**，
  只计设备侧 kernel 执行时间（不含张量构造、H2D/D2H 搬运、JIT 编译、框架调度
  开销）。Phase 2 调优与 PR 性能报告均沿用此口径，不得另立标准。

## Phase 2 — 主仓开发（沙箱约定）

- 技术执行调度 `@tilelang-op-orchestrator`（设计→实现→测试→调优三阶段状态机），
  本 skill 不亲自写 kernel；精度容差与纯 Kernel 计时统一按 Phase 1"口径标准"执行。
  **分设环境下所有验证执行走"分设环境执行模型"**（远程 exec/job，见上文）；
  沙箱文件组织与过程文档纪律不变。开发中的错误处理遵循 AGENTS.md 原则 7
  （调试链：`tilelang-debug-helper` / `tilelang-error-fixer`），禁止一遇错误
  就全部重写。

### 环境就绪流程（每次进入验证前）

1. `cannlab whoami` 确认登录——未登录则引导用户执行 `cannlab login --manual`
   （授权链接/回调地址等敏感串不经手 agent）；
2. `cannlab --output json env list` 确认 NPU 环境存在，记录名称/ID；
3. 需要执行验证时**自动拉起**（`env start` + `env wait --status RUNNING
   --timeout 10m $NPU_ENV`），不逐次询问（见"启停与查询纪律"）；中途被平台
   自动关机的，下次需要时重新拉起。

### NPU 环境首次初始化（每环境一次性）

NPU 环境为空机，需装载项目并构建（约 30~60 分钟，用 job 后台执行）：

1. **装载自适应**（探测 NPU 出网后二选一）：
   `cannlab --output json exec --no-start $NPU_ENV -- bash -lc "curl -m 5 -sI https://github.com >/dev/null && echo OK"`
   - **出网通 → 方案 A**：NPU 上 `git clone --recurse-submodules <fork URL>` +
     `git checkout <CPU 侧记录的基线 commit>`；此后沙箱未跟踪文件仍走 sync 推送；
   - **不通 → 方案 B**：`cannlab sync --exclude .git --exclude build --exclude __pycache__ <CPU主仓> $NPU_ENV:/mnt/workspace/tilelang-ascend`
     （全树镜像，tracked + 未跟踪一次到位；基线凭据靠 CPU 侧 commit 记录）。
2. **构建**：`cannlab --output json job start --cwd /mnt/workspace/tilelang-ascend $NPU_ENV -- bash install_ascend.sh`，
   `job wait` 至完成；
3. **探针复用制**：此后每次开机先跑 10 秒探针——
   `cannlab exec $NPU_ENV -- bash -lc "source set_env.sh && python -c 'import tilelang'"`——
   通过则 build 容好直接用；失败（环境被平台回收重建）则重做本节。

- 文件命名按算子前缀隔离（见架构图），多算子可并行共存，无需 stash。
- 过程文档**边开发边写**，不要事后补：
  - `DESIGN_{op}.md`：公式拆解、GPU→Ascend API 映射（附出处）、模式选型、被否方案
  - `debug_log_{op}.md`：每次失败按 Attempt 记录（分类、根因、受控验证）。
    **禁止多算子共写单一 `debug_log.md`**（历史反模式：sparse_mla_fwd 的完整踩坑史
    曾因此未随首轮归档丢失，仅靠 indexer_bwd 会话归档的共享副本才找回，2026-10-22
    拆分回迁）——共享目录下也必须按 `{op}` 前缀各建各的
   - `perf_tuning/{op}_*`：基线、消融、final_summary（含量化结论与口径）。
     **msprof 采集的完整 OPPROF 目录按轮次留存**（含 `dump/` 子目录——MindStudio
     Insight 图形化复盘的依赖；实测单轮 ~2MB，全轮留存无体积压力），禁止只留
     summary 丢弃原始目录——终态归档进 journal 供后续用 MindStudio 学习
   - `history_version/{op}_*`：只在"方案转折点"存快照（回退前存档）
- **红线**：主仓零 commit；`ref_gpu/` 只读。
- **性能达标检查点（调优一结束即判，未达标立即中断）**：调优（orchestrator
  Stage 3）完成后**立即**对照 Phase 1 性能目标判定——
  - 达标 → 直接进入 Phase 3 交付；
  - **未达标 → 立即中断流程并告知用户（不等归档/清理/交付）**：当前值 vs 目标值、
    差距倍数、根因（硬件物理不可达论证 / 工具链限制清单）、已尝试无收益的优化清单；
    **等待用户决策（五选一）后再继续**：继续交付 / 继续调优 / 调整目标 / 终止 /
    **先看当前算子的流水图再决策**——选末项时，agent 将最新（或关键）轮次的
    OPPROF 完整目录路径移交用户并附 MindStudio Insight 打开指引，同时自行输出
    该轮的图形化解读（各 pipe 占用率读数、stall 分解、瓶颈归因），用户看完
    图后回到四选一；用户表态前不得进入 Phase 3。设计期（Stage 1）发现目标物理
    不可达时同理，当场中断询问。

### 流水图驱动的调优深挖（检查点未达标且用户选择"继续调优"时）

（分设环境：msprof 经 cannlab job 远程执行，prof 产物 cp 取回后在 CPU 侧分析；
同一工作块内的多候选应一次开机批量画像，避免反复启停。画像采集与指标判定的
技术执行点名 `tilelang-op-profiling`（msprof op 采集/归档摘要/性能标准判定），
优化点识别与逐项实施点名 `tilelang-perf-optimization`；两者的本地直跑命令一律
按"调优类 skill 的分设约束注入"节执行。）

下一轮调优**必须以流水图（msprof 管线画像）证据驱动**，禁止无画像盲试；同时
"kernel 层路线穷尽"的结论必须经地板核算才能宣布：

1. **画像先行**：对当前实现跑 msprof，逐 pipe 读数——Cube/MTE1/MTE2/MTE3/fixpipe/
   scalar 占用率、scalar 各 stall 源时长（MTE 队列停顿、跨核 flag wait_id 分布）、
   AIV 侧等待与节流空闲——把瓶颈归因到具体管线与等待源；**无画像不立优化项**。
   关注三类信号：关键路径 pipe 占用率（墙在哪）、scalar stall 分解（等什么）、
   对侧核空闲率（谁在陪跑）。每轮采纳后复测画像，验证归因是否被修正。
2. **host 分发地板核算**：do_bench 口径虽只计设备侧，但**首 rep 会暴露 JIT 适配层
   host 分发延迟**（实测 ~1ms/次量级：Cython forward + 每次调用重建 TVM Arith
   Analyzer 等框架开销），对官方均值的贡献 ≈ 分发延迟 / rep 数；且 kernel 稳态
   逼近分发延迟以下时，该延迟将从"首 rep 暴露一次"变为"逐 rep 暴露"（pipelining
   失效），继续压 kernel 不再改善均值。判定路线穷尽前必须同时核算：稳态 kernel
   时间、host 分发地板、两者对官方均值的合成预估（mean ≈ max(kernel, 分发) +
   分发/rep），   区分"kernel 仍可优化"与"需框架/工具链配合"两类剩余差距。分设环境下 host
   规格（vCPU 数/主频）随 NPU 环境配置而定，分发地板测值须标注环境规格，
   跨环境对比的数字需注明来源环境。
3. **路线穷尽判据**：流水图无可寻址 stall/bubble（或剩余 stall 全部归因到已证伪的
   工具链缺口）+ 结构性重排（更深任务批处理/节流深度/peeled prologue）已试 +
   地板核算后仍超目标 → 才可宣布 kernel 层穷尽；host 分发地板与工具链缺口作为
   上游反馈项写入 final_summary，回到检查点请示。

### 调优续行（交付后回流，Phase 2R）

用户在检查点选择"继续调优"而交付（Phase 3-6）已完成时的回流程序：

1. **恢复沙箱**：从 `journal/{op}/` 逐项 cp 回 `examples_experiment/<共享目录>/`——
   `impl_final.py → {op}.py`（恢复原名）、`test_{op}.py`、`DESIGN.md → DESIGN_{op}.md`、
   `proto.yaml → proto_{op}.yaml`、`history_version/{op}_*`、`perf_tuning/{op}_*`；
    恢复后核对 md5/行数与冻结条款（DESIGN 冻结版本保护条款若存在则先验证），
    并跑一次最小精度套件确认环境健康（主仓框架可能已前移；分设环境下经
    cannlab 远程执行，见"分设环境执行模型"，含开机探针确认 build 容好）。
2. **优化执行**：遵循 Phase 2 全部纪律（改前快照 / 同会话 A/B / 精度门禁 /
   未达标当场中断）。零采纳时交付物不动。
3. **回流收尾**：按 Phase 4 重新同步 journal（刷新变更文件、撤回性结论在原结论
   文档顶部加更正横幅而非删改原文、更新 00_INDEX 与 journal README 收获行）→
   Phase 5 刷新 journal 内 `{op}/PR_DESCRIPTION.md`（交付内容变更时才需）→ 若
   交付内容变更，按用户既有指令在 -pr 分支 `amend + --force-with-lease` →
   Phase 6 再次清理沙箱（注意清理清单含 `.pkl_memoize_py3/` 等本轮新生成的缓存目录）。

## Phase 3 — 交付（PR worktree）

```bash
cd /mnt/workspace/gitCode/skybasejzy/tilelang-ascend
git fetch upstream
git worktree add ../tilelang-{op}-pr -b feat/{op}-pr upstream/ascendc_pto
```

- 交付单元 **≤2 个文件**：算子实现 + 算子测试。是否合并为单文件
  `example_{op}.py` 由 `tilelang-example-merge` skill 决定（目标仓库规范优先）。
- 把交付文件 cp 进 worktree，`git add` **只加这两个文件**。
- 提交信息风格对齐现有 -pr 分支：`[Example] Add {op} operator and test` /
  `[Feature] Implement {op} operator`。
- **交付前格式检查（强制，push 之前）**：在 -pr worktree 内跑
  `ruff format --check <交付文件>` 与 `ruff check <交付文件>`，不过则先
  `ruff format` 修复再提交（CI 的 format-check 挂过两次的真实教训：
  #1952/#1953 均因格式失败返工；规范细则见 `tilelang-review` skill）。
- `git push -u origin feat/{op}-pr`。
- 基线已前移时可先在 worktree 内 `git merge upstream/ascendc_pto` 再 push。

## Phase 4 — 归档（dev-journal 学习档案）

目标结构（journal worktree 内）：

```
dev-journal/
├── README.md                     ← 总索引：每算子一行（状态/PR链接/一句话收获）
└── {op}/
    ├── 00_INDEX.md               ← 学习导航（必写，见模板）
    ├── DESIGN.md                 ← 最终设计（多路线则保留全部 DESIGN_*.md）
    ├── debug_log.md
    ├── impl_final.py             ← 开发终态快照（注意：若终态是证伪实验，
    │                               取"交付基线"版本并在 INDEX 中说明）
    ├── ref_gpu.py                ← 从 ref_gpu/{op}.py 冻结副本
    ├── proto.yaml
    ├── history_version/          ← 仅关键节点 + 被否方案设计 rev
    ├── perf_tuning/              ← 全部 .md 结论 + 被引用的 .json
    └── PR_DESCRIPTION.md         ← PR 阐述留档（来源见下）
```

### 归档命名规范（journal 内统一，2026-10-22 补——此前三算子三套惯例）

journal `{op}/` 内一律用**模板通用名**（目录本身已按算子隔离，无需前缀），从沙箱
cp 时按下表改名；`history_version/` 与 `perf_tuning/` 内保持沙箱 `{op}_` 前缀原名
（零重命名冻结）：

| journal 文件 | 沙箱来源 | 说明 |
|---|---|---|
| `DESIGN.md` | `DESIGN_{op}.md` | 多路线则保留全部 `DESIGN_*.md` |
| `debug_log.md` | `debug_log_{op}.md` | 见 Phase 2 反模式警告 |
| `impl_final.py` | `{op}.py` | 开发终态快照（≠ -pr 分支的清理版，后者是 AST 等价精简） |
| `test_{op}.py` | `test_{op}.py` | 原名保留（模板此前未列，补） |
| `ref_gpu.py` | `ref_gpu/{op}.py` | 冻结副本 |
| `proto.yaml` | `proto_{op}.yaml` | |
| `history_version/{op}_*` | 同名 | 含被否 rev 设计与关键迭代快照 |
| `perf_tuning/{op}_*` | 同名 | |
| `PR_DESCRIPTION.md` | 归档时按模板起草 | 未提 PR 时必须产出可用初稿 |

> 历史遗留不一致（如 `sparse_mla_bwd/` 内仍为 `{op}_` 前缀名）：发现时按本表统一，
> 改名用 `git mv`，同目录 00_INDEX/交叉引用同步更新。

### 归档取舍表（按学习价值分级）

| 级别 | 内容 | 处置 |
|---|---|---|
| 必留 | DESIGN（含被否 rev）、debug_log、perf_tuning/*.md、ref_gpu 源、proto、impl 终态 | 全部归档 |
| 必留 | **perf_tuning 内 msprof 原始 OPPROF_* 目录（含 dump/）**——供 MindStudio Insight 打开复盘 Timeline；沙箱清理后 journal 是唯一存续处 | 按轮次全量归档；护栏：单算子 profiling 总量 >100MB 时整目录压缩 tar.gz（单包须 <100MB，GitHub 单文件硬限），仍超则按"基线轮/重大采纳轮/终态轮"保留关键轮并在 INDEX 注明 |
| 选留 | perf_tuning/*.json（仅被 md 引用者）、history_version/*.py（仅方案转折点） | 引用关系不明时宁多勿少 |
| 不留 | .orchestrator_state*、.stage_briefs/、backend_verify/、__pycache__、缓存 | 过程已沉淀在 md，原始态无学习价值 |

### PR_DESCRIPTION.md（归档时必须完成，禁止空占位壳）

**撰写与留存位置（强制）**：PR 文稿在 **journal worktree**
（`tilelang-dev-journal/{op}/PR_DESCRIPTION.md`，dev-journal 分支）内撰写，
随归档 commit + push 到 origin——这是唯一权威正本（本地 + 远端双备份）。
**主仓沙箱不撰写、不复制、不留存任何 PR 文稿**（沙箱不保留算子的任何
开发产物；历史 `pr_descriptions/` 目录做法废止，遗留副本由归属算子的
清理流程删除）。

按交付状态分两种处理：
1. **已提 PR**：拉取线上正文归档为 `PR_DESCRIPTION.md`（`gh pr view <branch>`
   或 GitHub API `pulls?head=<user>:<branch>`）；本地草稿以 `_draft` 等后缀留档。
2. **未提 PR**：**必须基于本目录归档材料（DESIGN / debug_log / perf_tuning /
   test）按下方模板起草可用初稿**——归档材料就是起草依据，禁止只留
   "TODO 待补" 的空文件。

**起草铁律（用户明确要求，违反即返工）**：
- 模板是通用的，技术要点和表格必须按该算子的**真实实现**调整；
- **没有实测数据的字段或章节直接删除，不得编造**；
- **严格区分性能目标、估计值、实测基线**三类数字，并标注口径与来源；
- 数据一律取自归档材料，不凭记忆填写；
- 差距未达标时如实披露（参考 sparse_mla_bwd 的 3.6× 缺口披露写法）。

**文风规范（PR 正文面向仓库维护者，不是内部工作日志）**：
- 只描述算子本身与实测结果；优化策略写成"**方法 + 效果**"，不写过程叙事
  （谁在哪轮做了什么、按什么规则回滚/采纳）；
- **禁止内部流程词汇**：iter-N、采纳 / 回滚 / 止损、DESIGN §x 等文档章节引用、
  debug_log / perf_log / final_summary 等内部文件名、覆盖门禁、任务书、
  Stage N、Orchestrator、[DESIGN_ERROR]、探针、哨兵等 skill / 编排器术语；
- 性能只呈现 **基线 → 最终** 与关键手段，版本迭代轨迹一律不出现；
- **不出现"草稿""待修改""待补"等状态字眼**——文件内容即最终 PR 正文，
  直接以"为 Ascend NPU 新增..."开头，无标题、无状态标注；
- 测量方法学（预热/重复/统计口径、Kernel-only vs E2E）用平实语言说明。

**模板全文**（`<...>` 为占位，按真实数据填写）：

```markdown
为 Ascend NPU 新增 `<算子名称>`（`<算子功能简介>`）算子。

- **Kernel 架构**：说明实际的 `@tilelang.jit` 配置、`out_idx`、Developer/Expert
  模式、`pass_configs` 及 Kernel 组织方式。
- **计算结构**：说明主要计算路径、编译期分支、核心数学公式，以及不同计算
  分支的适用条件。
- **数据流与内存管理**：说明 GEMM、Gather、Reduce 等关键操作，以及 GM、UB、
  L1、L0C、Workspace 等实际数据流与缓存组织。
- **关键优化策略**：说明 Tile 切分、Pipeline、Double Buffer、AIV/AIC 调度、
  数据复用等实际采用的优化方法及其作用。
- **数值精度处理**：说明 dtype、数值计算路径、精度转换及数值稳定性设计。
- **硬件与实现约束**：说明必要的 Shape 限制、片上内存预算、Assert 条件或
  同步要求。
- **测试验证**：简要说明实际测试覆盖范围、测试分层及精度验证结果。

## 精度验证 ✅

**测试配置**：`<实际测试 Shape、dtype>`

| **测试层级/类别** | **用例数** | **检查项数** | **测试结果** | **精度指标** | **最大绝对误差** |
| --- | --- | --- | --- | --- | --- |
| `<实际测试类别>` | `<数量>` | `<数量>` | `<结果>` | `<真实指标>` | `<实测值>` |
| **总计** | `<总数>` | `<总数>` | `<最终结果>` | — | — |

- **精度阈值**：`<实际 atol、rtol、max_abs_limit 等阈值>`
- **通过条件**：`<实际精度判断条件>`
- **参考实现**：`<PyTorch Golden 或其他实际参考实现>`

## 性能测试

**测试平台**：`<Ascend NPU 型号>`
**测试配置**：`<实际性能测试 Shape、dtype、输入配置>`

| **性能指标** | **测试结果** |
| --- | --- |
| `<msprof Task Duration / Graph Replay / NPU Event 等实际计时方式>` | **`<实测延迟> μs`** |
| 最小 / 最大延迟 | `<最小值 / 最大值> μs` |
| Block Dim / Mix Block Dim | `<实际配置>` |
| Op Type | `<实际类型>` |

> `<预热次数、重复次数、统计方式、计时范围；Kernel-only 与 E2E 差异
> 应明确区分。>`

### ⚠️ 性能差距说明

| **平台/方案** | **运行延迟** | **性能差距** |
| --- | --- | --- |
| Ascend NPU（本次 PR） | **`<实测延迟> μs`** | — |
| `<性能目标 / 参考实现>` | `<目标或参考值> μs` | `<实际差距及倍数>` |

`<差距、主要瓶颈及是否达标的如实说明；区分目标/估计/实测。>`
```

（注：模板中的"内存带宽（Block 0）"表仅在持有 msprof 分方向实测数据时保留，
否则整节省略；"运行频率"等同理。）

### 00_INDEX.md 模板（交付记忆最新鲜时写，交用户审阅）

```markdown
# {op} 迁移与调优学习档案
## 一句话：这个算子教会我什么
<如：GM bf16 atomic 累加精度陷阱及完整证伪方法>
## 阅读顺序（约 X h）
1. ref_gpu.py（30min）→ 迁移源头：<核心语义一句话>
2. DESIGN.md §<n>（40min）→ <学什么>
3. debug_log.md（40min）→ 核心坑：<坑列表>
4. perf_tuning/final_summary（30min）→ 基线→最终 <改善幅度> 怎么来的
5. history_version/（30min）→ 被否决的 <路线> 差在哪
6. perf_tuning/**/OPPROF_*（用 MindStudio Insight 打开）→ 图形化复盘各轮
   Timeline/pipe 占用，对照 md 结论验证瓶颈归因（NPU 环境同代 CANN 版本）
## 三大关键决策
1. <决策 + 为什么> 2. ... 3. ...
## 交付物
- 分支 feat/{op}-pr（origin） / PR: <链接或状态>
## 遗留疑问 / 未走通的路
```

## Phase 5 — PR 文稿交付（用户网页自行提 PR）

- **PR 由用户在网页自行创建，agent 永不代提、不做 GitHub 认证（无例外）**。
- agent 职责（Phase 4 归档完成后立即执行，不等 PR 创建）：
  1. 移交文稿路径 = **journal worktree 内的 `{op}/PR_DESCRIPTION.md`**
     （dev-journal 分支已 commit + push 的权威正本），用户直接从该路径取文
     粘贴；**禁止在主仓沙箱复制或留存 PR 文稿副本**（沙箱不保留算子的任何
     开发产物；旧 `pr_descriptions/` 复制做法废止）；
  2. 向用户移交三要素：分支名 `feat/{op}-pr`（origin）、compare 入口
     `https://github.com/tile-ai/tilelang-ascend/compare/ascendc_pto...<fork>:feat/{op}-pr`、
     文稿路径；PR 标题对齐既有风格（如 `[Example] Add {op} operator and test`）。

## Phase 6 — 清理（主仓回归干净）

**时机**：Phase 4 归档 + 交付分支 push 后即可执行，**不等 PR 创建/合并**（删前验证
只依赖远端分支与 journal 归档；PR 提交是用户侧异步动作）。

**删前验证（红线，缺一不可）**：
1. `git ls-remote origin feat/{op}-pr` 确认交付分支已在远端；
2. journal/{op}/ 文件清点通过（对照归档清单逐项 ls）；
3. 沙箱交付文件与 -pr 分支内容一致或更旧（`git show feat/{op}-pr:<path> | diff -`）。

**清理清单**：该算子的 `*.py` / `test_*.py` / `proto_*.yaml` / `DESIGN_*.md` /
`debug_log_*.md` / `perf_tuning/{op}_*` / `history_version/{op}_*` /
`.orchestrator_state*` / `.stage_briefs/` / `backend_verify/` / `extra-info/`（CANN
异常 dump）/ `__pycache__` / `.pkl_memoize_py3`。

**永不清理**：`ref_gpu/`（共享语义权威）、`.agents/skill-journal/`（skill 反馈数据源）、
**其他在开发算子的全部文件**（尤其未交付算子 = 唯一副本）、基线 tracked 文件
（如 `examples_experiment/dsa_sparse_finetune/example_indexer_bwd.py`）。
历史遗留的 `pr_descriptions/PR_DESCRIPTION_{op}.md` 副本（旧流程产物）在本算子
清理时一并删除；其他算子的遗留副本归其归属流程处理，不得代删。

## Phase 7 — 工作完成点与最终交付回复（agent 的最后一步）

- **agent 工作完成点 = 归档（Phase 4）+ 清理（Phase 6）+ PR 文稿交付（Phase 5）
  全部完成**。此后 PR 的提交/合并、-pr 分支与 worktree 的删除、journal 状态维护
  均由用户自行处理，agent 无剩余待办。
- **完成时必须向用户回复最终交付报告，四项内容缺一不可**：
  1. **交付物清单**：交付分支与 commit、PR 文稿路径（journal worktree 内
     `{op}/PR_DESCRIPTION.md`，dev-journal 分支权威正本）、
     journal 归档路径、compare 入口；
  2. **迁移算子的性能介绍**：基线 → 最终（分方向 + 合计）、计时口径（预热/
     重复/均值、仅设备侧）、主要优化手段及各自贡献、与性能目标的对比；
  3. **性能达标显式判定**：达标则明确写"达标"；未达标则如实复述判定（当前值 vs
     目标值、差距倍数、根因、已尝试优化清单）——未达标场景在 Phase 2 检查点已
     中断告知过用户并获其决策，此处为最终报告的正式记录，禁止弱化或省略；
  4. **精度验收结果摘要**（各层用例数与结论）。
- 用户明确要求同步基线时才执行：主仓 `git pull --ff-only`；若合并改动了交付
  路径，核对基线文件即最终版。

## 状态查询（用户问"现在什么局面"时）

`git worktree list` + `git branch -vv` + `git ls-remote origin`，输出三列表格：
开发中（沙箱有文件、无 -pr 分支）/ 待 PR（-pr 分支在 origin、PR 未开或未合并）/
已合并（upstream 基线含该算子）。

分设环境下另附一行 NPU 执行环境状态：`cannlab env show $NPU_ENV`（只读，
不启机），注明运行/停止及上次取回产物的位置。

## 红线（违反即返工）

1. 主仓**永不 commit**——否则后续算子 PR 带脏历史。
2. PR 分支**只放 ≤2 个交付文件**——过程文档进 PR 必被拒。
3. **归档先于删除**——任何 rm 前必须完成删前验证三步。
4. **不凭记忆判断**，用 git/diff/ls-remote 验证后再下结论。
5. 未交付算子文件 = 唯一副本，**碰都不许碰**（除非用户明示）。
6. 参数缺失先问，禁止自行假设。
7. 任何 worktree 内提交一律**明确路径 `git add <files>`，禁止裸 `git add -A`**
   （2026-10-21 事故：add -A 把 VSCode 生成的 `.vscode/.env` 环境变量转储连
   token 一起推上公开 fork，靠 amend + force-push 才清除）。提交前先
   `git status` 核对暂存内容。
8. **调优后性能未达标必须当场中断告知用户并等待决策**——中断点是 Phase 2 出口
   （调优刚结束），不是 Phase 7；不得静默走完归档/清理/交付后才说。
9. NPU 启停为**自动管理**（2026-11 用户授权）：需要时自动 start、掉线自动重拉、
   本工作块执行完且**产物取回自检通过**后自动 stop（自检不通过禁止 stop）；
   纯查询必须 `--no-start` 或只读命令，禁止意外开机。
   **例外**：`env recover` 与 EXCEPTION 状态处置仍需用户确认。
10. **NPU 侧禁止修改代码**（只执行）——一切编辑在 CPU 真源完成，改后重新 sync。
11. **NPU 上的产出必须当轮取回 CPU**（测试日志/性能数据/msprof 产物）；
    NPU 侧只允许驻留 build/ 等可再生重物。
12. **禁止交互式 SSH 长占 NPU**——验证一律 exec/job，长任务用 job 后台跑
    （断线不丢，本地连接即断即走）。

## 与其他 skill 的分工

| skill | 职责边界 |
|---|---|
| 本 skill | 全程编排、三区流转、归档、清理、PR 文稿交付 |
| tilelang-op-orchestrator | Phase 2 内部的设计→实现→调优技术执行（Stage 状态机） |
| tilelang-op-design | 写算子的设计层：design.md 生成（模式选型/API 映射/内存层级/Tiling/约束检测），经 analyst（Stage 1）调用 |
| tilelang-op-develop | 写算子的实现层：从 design.md 提取字段 + examples 参考生成代码，经 developer（Stage 2）调用 |
| tilelang-op-test-design | 写算子的测试层：L0 门槛计划（Stage 1 场景 A）与 L1/L2/Boundary 扩展（Stage 2 场景 B）；口径标准见其 precision-standard.md |
| tilelang-op-profiling | 调优深挖的画像技术执行：msprof op 采集、数据归档与统计摘要、性能标准判定（经 orchestrator/perf-tuner 调用；分设约束由本 skill 注入） |
| tilelang-perf-optimization | 调优深挖的优化技术执行：优化点识别（optimization_log）、逐项实施、效果验证（经 orchestrator/perf-tuner 调用；分设约束由本 skill 注入） |
| tilelang-debug-helper / tilelang-error-fixer | 开发中的错误诊断（GDB/IR dump/AOT），按 AGENTS.md 原则 7 由 Stage 2 按需调用 |
| tilelang-review | Phase 3 交付前的格式检查规范（ruff format/check） |
| tilelang-example-merge | Phase 3 的交付文件合并规范 |
| tilelang-github-operations | 不参与本流程——PR 由用户网页自行提交，agent 永不代提 |
| cannlab-cli | 分设环境的远程执行底层（命令规范、结构化输出与安全边界）；本 skill Phase 2 的验证执行依赖它，命令细节按需读其参考文件 |
