# sparse_mla_fwd debug log

结构化踩坑记录（tilelang-op-migration Phase 2 规范：每次失败按 Attempt 记录 — 分类 / 根因 / 受控验证）。

> **重建说明（2026-10-22）**：前次开发期 debug_log 未随 journal 归档而丢失；iter 1-7 与 rev2 的完整
> 踩坑史留档于 `perf_tuning/sparse_mla_fwd_perf_log.md`（含 RC1-RC5 / P1-P7 / R-AIV-FLAG /
> R-UB-LAYOUT / R-DIV 等规则）与 `DESIGN_sparse_mla_fwd.md`。本文件自调优 Round 2 起按规范维护。

## [perf-R2] 2026-10-22 稳健优化包 A1a/A1b/A3a（三项全否，零采纳）

**Attempt 1 — A1a 任务书原案「V 侧 UB cast bf16→fp16」编译失败**
- 分类：硬件限制（新规则 **R-CAST-BF16FP16**）
- 根因：dav_c220 `CastIntrinsicsImpl` 无 bf16→fp16 重载（候选解析落到 int4b_t*，no matching function）
- 受控验证：/tmp/opencode/r2/probe_casts.py（编译错误全文在 logs/probe1.log）；重设计探针
  probe_casts2.py 证三步链 fp32→bf16→fp32→fp16 位保持 bit-exact（P2b），bf16→fp32 / L0C→GM bf16
  cast / fp16 同型 staging 均可用（P3/P4+P5/P6）
- 处置：A1a 重构为 Q/KV host 预转（gather 直读 fp16 副本，V 侧零 cast）+ P 三步位保持仿真

**Attempt 2 — A1a 变体 +19.7% 劣化（证伪审查 §4.1 流量模型）**
- 分类：文档前提错误（算术）+ 硬件（新规则 **R-FP16-NOBW**）
- 根因：① fp16 与 bf16 同为 2 字节，ws_kv/Q fp16 化不省任何 GM 字节（审查 A1a 行 -7.28GB 系 ÷2 笔误，
  与其自身「ws_p 流量不变」表述矛盾）；② 同字节量下 fp16 拷贝模板效率更低（msprof：C mte2 +27% /
  V mte3 +28% / V mte2 +15%，cube/mte1 持平）——零节省 × 负效率 = 纯亏
- 受控验证：同进程交错 A/B（v0 46554 vs a1a 55743，3 轮 std<1%）；msprof_a1a 管道表 vs 审查 §3.1
  同快态 v0；精度侧 a1a 仅 57/268M out 1-ULP + 1 lse 1-ULP（p3 累加序签名），谓词边界 65280/65536
  分派正确（diag_a1a.py）
- 处置：否决回滚；审查 §4.2 C1/C2 行作废（详见 perf_log Round 2 §3）

**Attempt 3 — A1b（-0.5%）/ A3a（快 -2.6% 慢 -1.2%）/ c23 叠加（-1.5%）均不过 3% 采纳阈**
- 分类：收益量化（非错误）
- 根因：A1b 的 ws_o 中转减半只作用于 C fixpipe 0.5ms 量级；A3a 的 LEAD=2 只兑现 F0 等待 5.9ms 的
  ~20%（V 侧 F3 尾部等待不受 gather 前瞻影响）
- 受控验证：A3a bit-exact（ndiff=0，纯调度）；三窗口交错计时全记录（ab_r2/ab_c23/ab_final.log）
- 处置：按「<3% = 无变化 → 回滚」规则全否；A3a 形态留档待拆分重构复用（F0 须 mod-3，parity-2 在
  LEAD=2 下有 set 竞态——任务书「奇偶制与深度无关」表述不成立，推导见 perf_log Round 2 §7）

**工程教训 — 后台进程被工具超时连带杀死**
- 分类：流程
- 根因：bash 工具 120s 超时按进程组清理，`nohup ... &` 的子进程一并被杀（首次 ab_r2 中途消亡，
  无 traceback——勿误判为 NPU/编译崩溃）
- 处置：长跑一律 `setsid nohup ... < /dev/null & disown` 完全脱离进程组
