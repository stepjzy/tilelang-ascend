# sparse_mla_bwd 开发日志（debug_log）

## Attempt 1 — 2026-10-14（first_impl）
- mode: first_impl
- classification: precision_fail（L1 值域轴 4/15 用例；根因定位于 DESIGN §12 校准层，非实现层——见下）
- fail_category: design_precision_calibration（Δ-from-bf16-o 接口颗粒 + 高集中度抵消区逐元素门限不可满足；详见 error_summary）
- test_level: all（L0+L1+L2+Boundary，双后端）
- coverage: L0:9（全 PASS ×双后端） L1:15（11 PASS / 4 FAIL） L2:9（9 正确拒绝） Boundary:4（1 PASS + 3 WARN）
  覆盖门禁：24 PASS / 0 MISS / 0 N/A（Fusion 全维度，proto_sparse_mla_bwd.yaml 派生）
- boundary_warnings: |
    boundary inf_q dkv [BOUNDARY_WARN]（ratio 0，golden 自身 inf/NaN 散射，结构比对不匹配；dq PASS）
    boundary nan_kv dkv [BOUNDARY_WARN]（同上）
    boundary dbound_large_scale（scale 2e4）[BOUNDARY_WARN]（Δ 颗粒放大 + 极值区，非阻塞）
    L2 全部 [BOUNDARY_PASS]（9/9 正确拒绝）
- changes: |
    新建 sparse_mla_bwd.py（三段 kernel：preprocess/bwd/postprocess + host 串行；927→当前版本）与
    test_sparse_mla_bwd.py（L0 9 用例 + 场景 B 扩展 L1 15 / L2 9 / Boundary 4 + 覆盖标注）。
    过程中的 3 轮修改（每轮备份/证据如下）：
    1. [v0 → fp16 尝试 → 回退] L0 首跑 dkv 混合容差 ratio 0.9625-0.9825 < 0.99（sim 门限全过 0.999996+，
       双后端逐位一致）。replica 阶梯定位：kernel ≈ replicaB（bf16 P/dP，ratio 0.9624 复现）；replicaA
       （fp32 P/dP）ratio 0.9969 PASS ⇒ 根因 = 主仓继承的 P/dP bf16 GEMM 操作数颗粒 vs fp32-autograd
       golden。fp16 操作数探测：gemm_v0 fp16 A/B + fp32 L0C 双后端 PASS，但 GM(bf16)→L1(fp16) cast
       copy 双后端均不支持（copy_gm_to_l1 无匹配重载），且 fp16 P/dP 与共享 q/do 操作数 dtype 混用
       级联（G3-G6 全部 GEMM 强制同 dtype）⇒ 弃。
    2. [hi+lo 双 bf16 通道] V 侧 P/dP 拆 hi/lo 两路 bf16（合计 ~16 bit 尾数），G5/G6 双 GEMM 重组
       （+9 gemm/块）；G3/G4（dq 路径）沿用 hi 通道（原 bf16 dP，ratio 0.998+ 达标）。首版 p_hl/dp_hl
       单槽复用引入 MTE3-读 vs V-写同块竞态（hi 通道整槽污染，dq rel~1.0 全错——MTE3 GM 写未完成
       即被 V pipe 覆盖）⇒ 改独立 p_hi/p_lo/dp_hi/dp_lo 四槽（+16KB UB，总 ~168KB）修复。
       L0 复跑：9/9 全 PASS（dkv ratio 0.9969-0.9998）。
    3. [场景 B 扩展] L1/L2/Boundary + 覆盖门禁（checker 24 PASS/0 MISS 一次通过）。
- error_summary: |
    全量 --level all（双后端逐位一致）：
    L0: 9/9 PASS（L0-1/3 主考核 S=2048 topk=512 dkv ratio 0.9998；OOB/-1/repeat×2/varlen 全过；
        fwd kernel 路径全程健康，无 fallback）。
    L1: 11/15 PASS；4 FAIL（均为值域/缩放轴）：
      - L1-smscale (sm_scale=0.1): dq 0.9713 / dkv 0.9631
      - L1-valrange-m (σ=10): dq 0.8002 / dkv 0.6970
      - L1-valrange-l (σ=100): dq 0.2696 / dkv 0.6854（fwd lse sanity 亦 FAIL → fallback；
        主仓 sim 判据 1−sim=1.5e-4 亦超 1e-4）
      - L1-valrange-asym (σ=3, shift=6): dq sim 0.978（主仓判据亦 FAIL）/ ratio 0.2046
    L1 其余 11 例（shape 对齐/尾块/质数/退化、varlen 混合段、kv_group=2、H=32、topk=256、
    return_kernel、valrange-s σ=0.5）全部 PASS。
    根因阶梯（/tmp/opencode/diag_l1.py，replica 对照）：
      a) kernel ≈ replicaB（bf16→hi/lo 颗粒复刻）逐 scale 一致 ⇒ kernel 忠实，非实现缺陷。
      b) smscale 例：replicaA'（fp32 P/dP + golden lse/Δ）仍 0.98 FAIL；replicaP（Δ=Σ_j P·dOv，
         即 golden 内部 fp32-o Δ 的恒等变形）ratio=1.0000 ⇒ 唯一主导误差 = Δ-from-bf16-o
         接口颗粒（kernel Δ = Σ o_bf16·do vs golden autograd Δ = Σ o_fp32·do，主仓接口固有，
         任何 §1.3 忠实实现不可消除；误差 ~√512·ulp(|o|)·|do|，随 sm_scale 与 σ² 放大）。
      c) valrange-m/asym 例：replicaP2（golden lse + 一致 Δ 的完美 fp32 slot 基准）dq 仍
         0.9497/0.3059 FAIL ⇒ 高 softmax 集中区（σ≥10 / 大 shift）dq 为大数抵消量，
         golden 与任何完美实现之差即超逐元素容差——§12.2 门限在该输入域不可满足。
    数学模型：δdq[h,d] ≈ ε_Δ[h]·sm_scale·o[h,d]，ε_Δ=√512·ulp_bf16(|o|)·σ_do——解释
    默认包络（σ=1, sm_scale=576^-0.5）下 L0 全过、值域轴随 (σ², sm_scale) 失效的全部观测。
- design_error_reason: none（架构/tiling/内存/同步均按 DESIGN 实现并验证；未触发 [DESIGN_ERROR]，
  但 §12 校准层缺口需 Stage 1 决策，见 next_hint）
- rollback: no
- backup_path: history_version/sparse_mla_bwd_impl_s2_attempt1_v0.py（v0：plain-bf16 P/dP 首版）
- instrumentation_cleaned: yes（全部诊断为 /tmp/opencode 独立脚本，未入 kernel/test）
- next_hint: |
    Stage 1 需在以下三径中决策（实现层无法自行闭合）：
    (a) §1.3 Δ 公式精化：Δ = Σ_j P·dOv（数学上 ≡ Σ o_fp32·do，golden 内部 Δ 的恒等式；bwd 已重算
        P/dOv，V 侧每块一次 reduce 即得；实测 smscale 类 ratio 0.97→1.0000）——但属 §4.1 三段架构
        与接口契约（o 仅供 preprocess）的决策变更；
    (b) §12.2 值域轴门限校准：σ≥10 / 大 shift / sm_scale≥0.1 的用例改按主仓 sim 判据或尺度感知容差
        （replicaP2 证据：该域逐元素门限对任何实现不可满足）；
    (c) L1 值域轴包络收敛至可满足域（σ≲2、默认 sm_scale）并在 COVERAGE_NA/文档声明边界。
    推荐组合：(a)+(b)——(a) 修复 sm_scale 轴（ratio 1.0 证据），(b) 覆盖高集中度不可满足域。
    另：性能未计时（Stage 3）；hi/lo 使 GEMM 发射 23→32/块（+39%），Stage 3 需复核 §13 预算
    （若 Stage 1 选择 (a)，Δ 精度路径可再评估是否回退 plain-bf16 + 门限放宽的折衷）。
- 证据文件: |
    /tmp/opencode/probe_bwd_l0c.py（L0C liveness 合并探针，双后端 PASS）
    /tmp/opencode/probe_fp16_gemm.py（fp16 gemm 探针，PASS）
    /tmp/opencode/probe_fp16_all.py（GM→L1 cast 探针，双后端 FAIL——fp16 路径弃用依据）
    /tmp/opencode/smoke_bwd.py（kernel-vs-replica 冒烟，8 配置 × 双后端）
    /tmp/opencode/diag_dkv.py（dkv 颗粒阶梯定位）
    /tmp/opencode/diag_l1.py（值域轴根因阶梯：replicaA/A'/B/P/P2）
    /tmp/opencode/l0_run3.log（L0 9/9 全过日志）；/tmp/opencode/all_run1.log（全量日志）

## Attempt 2 — 2026-10-14（precision_fix，amend-1 实施）
- mode: precision_fix（attempt_index=2，design_revision_count=1-amend1；上次调度被中断未落盘，本次从头执行）
- classification: precision_pass（L0 9/9 + L1 15/15 阻塞全过 + 覆盖门禁 24 PASS + 全量双后端 exit 0）
- fail_category: none
- test_level: all（L0→L1→L2/Boundary 分层跑 + 覆盖门禁 + --level all 全量；SPARSE_MLA_BWD_FWD 默认 kernel 模式）
- coverage: L0:9（全 PASS ×双后端） L1:15（15 阻塞全过 = 域A 11 + 域B 3；域C 2 例按 §12.2 amend-1 记 WARN 非阻塞）
  L2:9（9 正确拒绝） Boundary:4（1 PASS + 3 WARN）
  覆盖门禁：24 PASS / 0 MISS / 0 N/A（与 attempt-1 一致）
- boundary_warnings: |
    L1-valrange-m（σ=10，域C）：dq ratio 0.9479 / dkv 0.9976 未达逐元素门限 → [BOUNDARY_WARN]
    （sim 大幅改善：dq 0.8002→0.999996；golden bf16-P 内部颗粒主导，per replicaP2 证据不可满足）
    L1-valrange-l（σ=100，域C）：dq sim 0.4579 / dkv 0.4662 → [BOUNDARY_WARN]；fwd lse sanity 亦 FAIL
    → torch_fallback（与 attempt-1 同行为，golden 自身失效域）
    L1-valrange-asym（域B）：dq ratio 0.3011 → ratio-ref WARN 非阻塞（sim 0.999907 主判据过；
    大 shift 子域逐元素门限对完美实现不可满足，replicaP2 实证）
    boundary inf_q / nan_kv / dbound_large_scale（scale 2e4）：[BOUNDARY_WARN]（golden 侧 inf/NaN
    散射 + 极值区，与 attempt-1 完全一致，非阻塞）
    L2 全部 [BOUNDARY_PASS]（9/9 正确拒绝）
- changes: |
    1. sparse_mla_bwd.py — amend-1 §1.3 Δ 恒等式落地：
       a) builder 新增编译期参数 delta_internal（默认 False）；
       b) delta_internal=True 时 per-task 两遍结构：pass 1 重算 idx/mask/gather + G1(score) + G2(dO·v)
          + P 链（exp 后 s_ub=P、dp_ub=dP_raw，均为 hi/lo 拆分前 fp32 中间量），行归约
          T.reduce_sum + T.tile.add 跨 NI 块累加 delta_acc；pass 2 完整链逐字不变（Δ 经
          T.copy(delta_acc, delta_ub) 注入，broadcast/dP 语句保持原样）；
       c) 外部 delta 路径 IR 逐字节不变（trace-time 分支），L0-4 数值与 attempt-1 逐位一致（锚点验证）；
       d) host 编排 §13.1：delta=None 不再发射 k1（两段 k2→k3）；_run_bwd 以 lse 作 Delta 占位参数
          （internal 模式从不读取）；k1/_run_preprocess 保留（对照诊断路径）；build cache key 增加
          delta_internal。
       实施过程 2 个结构问题已修复：① edit 工具模糊缩进匹配曾致 lse_ub/delta_ub 脱离 with 块
       （ParserError: Undefined variable: cid）→ 恢复 16 空格缩进；② TVM script parser 的 if 分支
       作用域不泄漏赋值绑定（条件 alloc delta_acc 报 Undefined variable）→ 改为无条件 alloc
       （外部模式多 128B 未用 UB 槽，无害）。
    2. test_sparse_mla_bwd.py — §12.2 amend-1 分层门禁 + R4 环境开关：
       a) run_gate_case 新增 gate 参数：A（域A，双判据一字不动）/ B（sim≥0.97 阻塞 + 混合容差非阻塞
          参考，未达标注 WARN）/ C（整例 WARN 级参考，不阻塞）；L1 5 个值域/缩放轴用例打 gate 标签
          （smscale/valrange-s/asym→B，valrange-m/l→C），L0 与 L1 形状轴 11 例保持域 A；
       b) SPARSE_MLA_BWD_FWD=kernel|torch 环境开关（默认 kernel；torch 强制 _torch_fwd_fallback 全程
          隔离 fwd 会话不稳定，R4；本次全量在 kernel 模式下通过——最终验收口径满足）。
- error_summary: none（全量 exit 0，0 个 [PRECISION_FAIL]）
- 实测 vs 门限依据（§12.2 amend-1 校准值全部达成或超越）: |
    L1-smscale(0.1)：attempt-1 dq 0.9713/dkv 0.9631 FAIL → sim 0.999996 / dkv ratio 1.0000
    （1−sim=4e-6 已达域 A 判据，replicaP "ratio 1.0000 实证" 完全兑现）
    L1-valrange-s(σ=0.5)：ratio 1.0000（照常 PASS）
    L1-valrange-asym(σ=3,shift=6)：sim 0.978→0.999907（门限 0.97 留 ~0.03 余量；亦达 1e-4 判据）
    L1-valrange-m(σ=10)：sim 0.8002→0.999996（ratio 0.9479 域C WARN，如设计预期）
    L1-valrange-l(σ=100)：golden 失效域维持 WARN（fwd sanity FAIL→fallback 同 attempt-1）
    L0 全部 9 例：dkv ratio 0.9969-0.9998 → 0.9997-1.0000（主考核 L0-1 双 1.0000）
    按 §12.2 "只允许收紧" 条款：域 B 两例（smscale/asym）实测已达域 A 1e-4 判据，门限收紧依据
    已备——属 Stage 1 决策，本 attempt 不擅自改设计门限，仅记录。
- design_error_reason: none
- rollback: no
- backup_path: history_version/sparse_mla_bwd_impl_s2_attempt2_v0.py（修改前 = attempt-1 交付版）
- instrumentation_cleaned: yes（本次零插桩——修复一次收敛，无 T.printf/dump_tensor 临时代码；
    attempt-1 的 /tmp/opencode 诊断脚本未动）
- next_hint: |
    1. 性能（Stage 3 强制项，§13.1 amend-1）：两遍重算使 GEMM 发射 +~39.5%（叠加 hi/lo 后 ~87 TFLOPS
       口径，R11 中高风险）→ msprof 分段对账 + §6.4 菜单 1/2 评估；§13.1 记录的缓解分支——恒等式
       落地后评估 plain-bf16 P/dP 回退（域 A dkv ratio 实测 ≥0.99 才可省 hi/lo 通道；本次域 A
       实测 ratio 0.9997-1.0000 为该评估提供了正向初步证据，但 replicaB→hi/lo 阶梯表明需专项复测，
       禁止默认成立）。
    2. fwd 会话稳定后最终验收须 kernel 模式全量复跑（R4；本次已是 kernel 模式，若 fwd 后续迭代则需
       复跑一次以最新 fwd 为准）。
    3. valrange-l（σ=100）golden 自身失效 + fwd kernel 该域 sanity FAIL——fwd 侧 valrange 大 σ 域
       精度为 fwd 会话的已知边界（其 DESIGN 未覆盖），如需闭合归 fwd 任务。
- 证据文件: |
    /tmp/opencode/bwd_l0_att2d.log（L0 9/9 全过，双后端）
    /tmp/opencode/bwd_l1_att2.log（L1 15/15 + 域分层结果）
    /tmp/opencode/bwd_l2b_att2.log（L2 9/9 + Boundary）
    /tmp/opencode/bwd_all_att2.log（全量 --level all：L0 9/9 + L1 15/15 + Test Passed! + exit 0）

## Attempt 2b — 2026-10-14（test hardening，amend-2 门限收紧落地确认）
- mode: test_hardening（attempt_index=2b，收尾微任务；DESIGN amend-2 已由 Stage 1 落盘，本任务仅同步测试门限并全量确认）
- classification: precision_pass（L0 9/9 + L1 15/15 阻塞全过 + 覆盖门禁 24 PASS + 全量双后端 exit 0）
- fail_category: none
- test_level: all（--level all 单次全量，默认 kernel 模式 SPARSE_MLA_BWD_FWD 未设）
- coverage: L0:9（全 PASS ×双后端 16 legs） L1:15（15 阻塞全过 = 域A 10 + 域B 4[amend-2 含新升入的
  valrange-m]；域C 1 例 valrange-l 记 WARN 非阻塞） L2:9（9 正确拒绝） Boundary:4（1 PASS + 3 WARN）
  覆盖门禁：24 PASS / 0 MISS / 0 N/A（与 attempt-1/2 一致）
- boundary_warnings: |
    L1-valrange-m（σ=10，amend-2 自域C 升入域B 阻塞判据）：dq sim 0.999996 / dkv sim 1.000000
    ≥ 0.999 → [PRECISION_PASS]（与 amend-2 依据的 attempt-2 实测逐位一致）；dq ratio 0.9479 /
    dkv ratio 0.9976 未达逐元素门限 → ratio-ref WARN 非阻塞（按 amend-2 条款 ratio 维持非阻塞参考）
    L1-valrange-asym（域B）：sim 0.999907 ≥ 0.999（新门限留 ~9e-4 余量，按设计预期）→ PASS；
    dq ratio 0.3011 → ratio-ref WARN 非阻塞（大 shift 子域，replicaP2 证据）
    L1-valrange-l（σ=100，域C）：dq sim 0.457857 / dkv 0.466167 → [BOUNDARY_WARN]；fwd lse sanity
    FAIL → torch_fallback（与 attempt-2 逐位一致，golden 自身失效域维持不动）
    boundary inf_q / nan_kv（dkv ratio 0，golden 侧 inf/NaN 散射）/ dbound_large_scale（scale 2e4）：
    [BOUNDARY_WARN]（与 attempt-1/2 完全一致，非阻塞）
    L2 全部 [BOUNDARY_PASS]（9/9 正确拒绝）
- changes: |
    仅 test_sparse_mla_bwd.py（门限常量与 tier 归属，用例参数零改动；kernel sparse_mla_bwd.py 零改动）：
    1. run_gate_case gate "B" 分支阻塞门限 sim ≥ 0.97 → sim ≥ 0.999（§12.2 amend-2 收紧）；
    2. L1-valrange-m gate "C" → "B"（σ=10 升入域 B 阻塞判据）；valrange-l 维持 "C"；
    3. 同步 docstring/注释/输出 note 的分层说明（amend-1 → amend-2：域B σ≤10 含 valrange-m、
       域C 仅 σ≥100 valrange-l）——纯文档同步，无逻辑变更。
- error_summary: none（全量 exit 0，0 个 [PRECISION_FAIL]；44 legs [PRECISION_PASS]，值域轴实测与
    attempt-2 逐位一致：smscale 0.999996 / valrange-s 0.999997 / valrange-m 0.999996 /
    asym 0.999907 / valrange-l 0.4579-0.4662（域C WARN），复跑确定性确认）
- design_error_reason: none
- rollback: no
- backup_path: n/a（仅测试门限收紧，kernel 无修改，无需备份；history_version/ 既有备份不受影响）
- instrumentation_cleaned: n/a（零插桩）
- next_hint: |
    1. amend-2 收紧后全量确认完毕（§12.2 "收紧后需全量重跑确认" 条款闭环），Stage 2 收尾完成；
       后续任何重跑若域B 实测劣化 < 0.999 即 [PRECISION_FAIL]，不得回调门限（只紧不松条款）。
    2. Stage 3 性能调优（§13.1）现在有更强的精度回归防护网（域B 0.999 阻塞）——hi/lo plain-bf16
       回退评估若致 valrange-m/asym sim 跌破 0.999 需保留 hi/lo 通道。
    3. fwd 会话若再迭代，最终验收需 kernel 模式复跑一次（R4，本次已是 kernel 模式）。
- 证据文件: |
    /tmp/opencode/bwd_all_att2b.log（全量 --level all：L0 9/9 + L1 15/15 + L2 9/9 + Boundary
    1 PASS+3 WARN + Test Passed! + EXIT_CODE=0；覆盖门禁 24 PASS/0 MISS 同轮通过）

## Attempt rev3-1 (first_impl, 2026-10-15)
- mode: first_impl (rev3 周期, design_revision_count=2)
- classification: precision_pass
- fail_category: none
- test_level: all（L0 9/9 → L1 15/15 → L2 9/9 拒绝 → Boundary 4 记录 → 覆盖门禁 24/24 PASS）
- coverage: L0:9 L1:15 L2:9 Boundary:4（--level all 双后端全过，Test Passed! exit 0）
- boundary_warnings: boundary inf_q / nan_kv / dbound_large_scale → [BOUNDARY_WARN]（极值域非阻塞）；L1-valrange-l 域C WARN（amend-2 条款，与 rev2 attempt-2 实测逐位一致：dq ratio 0.1657 与记录 0.1657 同源）
- changes: 按 DESIGN rev3 全量重写 sparse_mla_bwd.py 的 k2 kernel（备份 sparse_mla_bwd_impl_s2_rev3_attempt1_v0.py = rev2 终态）
  1. pair-loop 生产/消费错位主循环（§4.3/§5.4/§9.4）：扁平化 `for kb in T.serial(NB+NI)`，迭代 kb 产生 (kb//NI, kb%NI) 块、消费 (kb//NI−1, kb%NI) 块；全部跨核槽 [grid,2,NI,…] token 奇偶双缓冲；q/do L1 奇偶双槽（parity*segment 扁平化 leading dim）；Developer 形态（alloc_shared/alloc_fragment + workspace 命名握手 + 4-True pass_configs，零 Expert API）
  2. Δ 恒等式 pair-loop 表达（amend-1）：produce 段 V 侧 fp32 前缀累加 → token 边界冻结 delta_cons → 消费段 dP 链使用（显式 delta 路径 GM 直读）
  3. GEMM 按 §5.1 DSEG=2（DS=256）：G1 3 发/G2 2/G3 2/G4 1/G5 2段×2发/G6 2；L0C 峰值 = acc_dkv [128,256] 128KB 贴线（alloc_fragment liveness 规划成功，DS=256 编译通过）
  4. 调试期发现并修复 4 个平台级竞态/别名问题（详见 error_summary）：
     a. pst_p MTE3 写与 hi/lo split 的 V 管 sub(s_ub) 竞态（split 残差改落 lse_2d，s_ub/dp_ub 在 GM 读后永不覆写）
     b. UB 规划器别名 kv_stage↔acc_dq_all（gather 以 N(0,1) kv 值覆写累加器；dq 累加器清零移到循环体顶部使 liveness 覆盖 produce 段）
     c. 跨分支同核管道依赖失同步（sync pass 在 if 边界清空 access history；PipeBarrier<PIPE_ALL> 实测不足）：修复 = pst 机制整体退役（P/dP_raw fp32 GM stash → 消费段 ws_s/ws_dp 重读 + P 链重算，位相同，全部走跨核 flag）+ 段尾 MTE2/V 排空探针（P1a: Q行→kv_stage 触发 MTE3→MTE2；P1b/P2/P3: cast_bf16→lse_2d 触发 MTE3→V）
     d. UB 总量压到 191.9KB < 196,352B 消除全部规划器别名（batch16 gather staging §8.3-①、P/dP lo 通道经 cast_bf16 暂存 UB→UB cast、无 lo 独立槽）
  5. flag 对数 16（=硬件上限 ids 0-15）：kv 4 + s 2 + dp 2 + p 2 + dpb 2 + dq 2 + dkv 2（V 双读 ws_s/ws_dp 由 C 侧 scratch 哑写配平；哑写置于 C 消费段首避免 dpb 依赖环死锁）
  6. L5 贡献槽档门控（§12.6(2)）：①fp32 基线 L0 9/9 ✓ → ②bf16 档全量复跑：dkv ratio 0.965-0.983 < 0.99 FAIL（sim 0.999997 过）→ 按 §14.3 R14 决策树回退 fp32 为默认（SPARSE_MLA_BWD_DKV_SLOT=bf16 可再启，Stage 3 复评）
- 基线计时（考核 shape B=1,S=2048,H=64,HKV=1,topk=512,bf16,delta=None 两段发射，N=100 median）：ascendc 28965us（rev2 iter-2 28541us，+1.5%）/ pto 30056us；考核 shape 精度 dq sim 0.999997 / dkv sim 1.000000。结构达成但 7680us 目标未达（预期内——pair-loop 流水收益需 Stage 3 msprof 对账：探针排空的每迭代管道串行化 + P 重算向量链为已知代价项）
- error_summary: 修复过程四级故障链（每级有 dump 证据）：(a) pst_p 槽内容=split 残差（got[0,:4] 与 residual 逐位吻合）→ split 残差改 lse_2d；(b) dq 误差 G≈N(0,1) 恒定范数~190 + tail 任务全对 → 编译产物 GetWithOffset 显示 kv_stage/acc_dq_all 同偏移 4224 → 清零移顶部；(c) 残余时序性散布误差（跨 run 变化，1018/115/383 bad pairs）→ sync pass if 分支 history 清空 + lse_2d↔kv_stage / dqt_bf16↔acc_dq_all 别名（编译产物确认）→ pst 退役 + 探针 + 无别名预算；(d) UB→GM/GM→UB 无 dtype cast（编译错误）→ lo 通道经 cast_bf16 UB→UB cast
- design_error_reason: none（rev3 设计结构全部可实现；L5 bf16 档门控 FAIL 为设计预设的档位回退路径，非设计错误）
- rollback: no
- instrumentation_cleaned: yes（探针为最终代码的正式排空机制，非临时插桩；调试用 /tmp 脚本未入仓）
- next_hint: Stage 3（1）msprof 对账 §13.7：AIC/AIV 等 flag 占比（>30% 触发 Expert fallback §4.2）+ 逐腿带宽（C MTE2 口径为最大不确定项）；（2）探针排空的流水代价评估——P1a/P1b/P2/P3 每迭代强制 V+MTE3 排空，可能抵消 pair-loop 重叠，需 msprof 定量后考虑合并/弱化（如 P2/P3 合并、probe 条件化）；（3）P 重算向量链（~20 tile ops/块）回归 pst 档的评估需先解决跨分支同步（workspace 化 + flag 预算重排）；（4）L5 bf16 档可在 Stage 3 以域 A 门限复评（当前 ratio 0.965-0.983）
