
## Attempt R-EV: 分设环境迁移重验证（2026-11-05，新 tilelang-op-migration skill 流程）

**分类**：环境迁移验证（CPU/NPU 分设架构，cannlab CLI 远程执行）
**背景**：环境由单机（CPU+NPU 同机）变更为分设（CPU 常驻编辑 + NPU 远程执行
DevEnvC_DCbyi_npu，1×910B3）。按新版 migration skill 重新执行迁移流程
（Phase 2R 回流：journal 恢复沙箱 → 远程重验证 → 归档刷新）。
**基线对账**：CPU 主仓 HEAD = 463c234c（NPU 侧 clone + checkout 同 commit，
文件 md5 同步校验通过：kernel 703d5914 / test 7a9ef7ca）。
**执行链（全部经 cannlab，任务块化单次开机完成）**：
1. NPU 首次初始化：git clone 方案 A（出网通）；首次构建被平台 30 分钟空闲
   自动关机中断（**教训：后台 job 不算平台"活动"，长构建需交互心跳保活**——
   加 240s 轻量 exec 心跳后增量续跑成功）；探针 PROBE_OK 0.1..4（463c234c 构建）
2. 精度重验证：--level l0（8/8 ratio=1.0000，误差与原环境逐位一致）→
   --level all（Test Passed! 0 FAIL；L2 8/8 拒绝；Boundary 4 PASS + 1 文档化 WARN）
3. 性能重验证：--level perf do_bench（warmup20/rep10/mean/仅设备侧）=
   **6884.7 us**（原环境 6868.3us，+0.24% 噪声内——设备侧计时跨环境可复现）
4. 双口径画像（终态）：msprof op（OPPROF_20261105004500，2.2MB 已归档）+
   应用级 msprof（PROF_000001_20261105004610，tar.gz 79K 已归档）；
   Task Duration 6673.2us、满频 1800/1800、BlockDim 20、mte2 54GB/s 引擎界、
   20 核均衡——与原环境 R4 画像结构一致
**关键发现**：**基线轮（attempt1 初版 kernel）在新环境上下文崩溃**——
msprof op 警告 no matched kernel、应用级 PROF 无任何 AI Core kernel 行。
attempt1 含 reduce/broadcast scratch 注入 op，其 UB 打包随上下文（路径/堆）
翻转（D-1 已证），新环境上下文恰落坏打包 → Illegal instruction。**这是 D-1
布局彩票现象在独立环境的首例复现，反向佐证 D-2 scratch-free 结构修复的
环境鲁棒性（终态 kernel 新环境全项通过）**。基线轮画像按实情记为不可采集，
基线性能数字沿用原环境实测（19556.3us，do_bench 设备侧口径）。
**结论**：重验证全绿，交付物零改动，按新 skill 补齐 OPPROF/PROF 必留归档。
