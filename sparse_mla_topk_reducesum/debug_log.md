# debug_log — sparse_mla_topk_reducesum panel-mode 变体

## Attempt 1 — 2026-10-10T02:30:00+08:00
- mode: first_impl
- classification: precision_pass
- fail_category: none
- test_level: all（L0 先行收敛后扩展全量）
- coverage: L0 41 条标记（P1/P2×2/P3/P4a-e/P5×4配置×3run/P6a-c）；L1 13 用例 18 条标记（5 用例 generic+panel 双跑）；L2 3 用例；Boundary 4 用例
- boundary_warnings: none（sparse_inf/sparse_nan/all_zero/bf16_max_sparse 全部 [BOUNDARY_PASS]，结构掩码一致 + matched 1.0）
- changes:
  - `sparse_mla_topk_reducesum.py`（+587 行，仅 1 行既有 docstring 改写）：新增 `tl_sparse_mla_topk_reducesum_panel` builder（Expert 模式，8 cross-flag 协议 F_PANEL/F_PDONE/F1_ACC/F_WSF 恰满 A2 预算；C 侧 GM 标量读组检测 + panel_l1 双槽常驻 + 单 gemm [HpB,DDT]×[DDT,topk]；V 侧 prologue 静态展开 LA=32（带 `if i < ntask` 守卫）+ lookahead k+LA 组供给 + 逐任务契约校验 cast→sub→abs→pipe_barrier→reduce_max[1]→v→s 对→UB 标量 if→atomic_add(ErrFlag)）；host 新增 `panel_mode="shared"` 分发（G1-G5 门控纯元数据 + ErrFlag 强制 .item() 检查 → RuntimeError）+ `_panel_gate_ok`/`_token_indices_from_list` helper + _BUILD_CACHE panel 键
  - `test_sparse_mla_topk_reducesum.py`（重写为分层结构）：L0-P1..P6（现有 golden 保留为 P2）+ L1 13 用例（确定性 shape 扫描 + 值域 + 非默认 attr）+ L2 3 负向 + Boundary 4 特殊值 + COVERAGE_CATEGORY/MANIFEST/NA + --level 分发
- error_summary: 开发期 4 轮问题全部定位并修复——(1) spike 期 reduce_max→[1] 垃圾值：实为 spike 自身初始化缺 mte2→v 同步对（reduce 从未损坏）；(2) L0-P4a 参考 amax 在空 batch 崩（NPU 零尺寸归约）→ masked ref 跳过空段；(3) L1 grouped ref 的 lse `.t()` 转置 bug（[16,128] 与 [64,32] flat 同元素数被静默接受，kernel 无辜，g=2 输出与手工计算精确一致）；(4) L1 valrange M/L/ASYM NaN：kernel（原版 CUDA 同构）exp-先于-mask 的 fp32 溢出（inf×0=NaN，|logit×scale−lse|>88）→ 值域校准至实证安全包络 ±6；(5) dim_v_256 abs_err 1.16e-3：masked/grouped ref 的 bf16 einsum 噪声 → 升级 fp32 einsum 后 kernel 真实误差 ~1e-7
- design_error_reason: none（设计风险 #1 C-scope GM 标量读 spike 一次通过：max_diff=5.7e-6 纯 fp32 累加序噪声）
- rollback: no
- backup_path: examples/dsa_sparse_finetune/history_version/sparse_mla_impl_s2_attempt1.py
- instrumentation_cleaned: n/a（first_impl；spike 临时脚本均在 /tmp/opencode，未入仓）
- next_hint: Stage 3 性能调优——graph-replay 主口径已测 panel 146.7-147.2us（vs generic BI=128 384.9us，2.62×，220us 判定 PASS 余量 1.5×）；可进一步压缩 C/V 每 task F1/F_WSF 握手（设计 §11 风险 #5 预案：2-task 批量握手）与 146→60-90us 设计带差距归因（msprof）
- 关键结论:
  - **panel-vs-generic 逐位相等（torch.equal=True）**：P1（BI=128 与 BI=32 双配置）、P4a/b/c/d、L1 task_tail_1/tail_mid/prime/dim_v_256/kv_group_2 全部实证——设计 §9.1 等价性论证成立
  - **契约违约检测精确**：P3 单行单元素违约 → ErrFlag=1 → RuntimeError；同输入 generic 声明结果正确
  - **host 门控路由全部正确**：P4e/P6a-c（G4/G5/G2域 miss）经 builder 探针 + panel-key 缓存缺席双证据证明分发到 generic
  - 通用路径零改动：diff vs 备份 = 1 行 docstring 改写 + 587 行新增，无既有可执行行变更；P2 golden 双 block_I 回归通过

## Attempt 1 (rev2) — 2026-10-10T18:40:00+08:00
- mode: first_impl（design_revision rev2：auto 缺省分发 + detect + guarded 双变体）
- classification: precision_pass
- fail_category: none
- test_level: all（L0 先行收敛 92 标记全过后扩展全量）
- coverage: L0 17 用例 92 条标记（P1/P2×2/P3/P4a-e/P5×4配置×3run/P6a-c/P7a×9子用例/P7b-e/P7f/P7g×2）；L1 14 用例（rev1 13 + panel_mode_sweep 三模式全跑 + 两两 torch.equal）；L2 3 用例；Boundary 4 用例；覆盖门禁 24 PASS / 0 MISS / 0 N/A
- boundary_warnings: none（L2 3 用例 + Boundary 4 用例全部 [BOUNDARY_PASS]）
- changes:
  - `sparse_mla_topk_reducesum.py`（853→1797 行）：① 共享 body 构造重组——generic/panel 各抽 `_reducesum_*_body(..., guarded)` 持编译期布尔，4 个薄 `@tilelang.jit` 包装（unguarded 公开名/签名/装饰器零修改 + `_strip_unused_param` trace 后摘除未引用的 Violations 形参；guarded 新增 out_idx=[] / Violations 形参）；② detect kernel（`tl_sparse_mla_topk_reducesum_detect`，V-only Expert：40 lane (seq,lane) 整除分区 + 相邻行链式比较 sub(int32)→cast→abs→pipe_barrier→两步 reduce_max→tile.max 累积→条件 atomic_add，三层嵌套动态 T.serial，§3.4 逐条落地）；③ host 分发重写（§8.3：panel_mode 缺省改 "auto"；元数据门控 miss → unguarded generic 单发零开销；通过 → violations 每调用新建 zeros → detect → panel_guarded → generic_guarded 同流三连发 + err_flag 尾部防御检查；"shared"/"generic" 显式行为零变化；off_list 复用 D2H 持平）；④ `_BUILD_CACHE` 键扩展 detect/generic_guarded/panel_guarded 三维度
  - `test_sparse_mla_topk_reducesum.py`（1119→1330 行）：L0-P7a..g 全家族 + P2 修订（显式 panel_mode="generic" 钉住 unguarded 回归）+ 全部既有 no-mode 调用点显式钉 generic（rev1 语义保持）+ auto_route_probe（builder 编译与缓存分发双计数）/check_route/expected_violations（host 侧 (seq,lane) 分区精确 lane 计数参考谓词）+ L1 panel_mode_sweep（三值维度入矩阵）
- error_summary: 开发期 4 轮问题全部定位并闭环——(1) 设计 §8.2 的"ntask 重绑定为纯 Python 别名"在本仓 TVMScript 解析器下不可实现（with 帧内重绑定自动改名 ntask_1、原引用不更新，spike S1 实证）；(2) 改用 Select 钳位后 guarded generic 编译 13s→600s+（CanonicalSimplifier 对含 GM load 的循环界联合语义爆炸，gdb 栈实证；min 算术钳位同样爆炸）；(3) 最终落地 = 入口 TIR-if 守卫（`_guard_run`/`_guard_v_entry` trace 期闭包：unguarded 返回 Python True 编译期直通零 TIR、guarded 返回 TIR 比较→IfThenElse 入口守卫），循环界与 rev1 全同 ⇒ guarded 编译 13.3s 持平 unguarded 13.1s；(4) L0-P7c 首跑 NaN=4224——未 clamp 的 randint 索引产生全因果无效行（renorm 0/0=NaN 在 ref 与 kernel 同现，数据构造问题非 kernel 缺陷）→ 改用 bench_graph_smtk 同款 randint→clamp(max=s_col) 构造后 err_ratio=1.4e-07 全过；(5) 一次 edit 缩进事故（task 尾 else 块少 4 空格）被 structural_equal 校验当场捕获并修复（unguarded 保全证明链的价值实证）
- design_error_reason: none（§11 R12 三 spike 全过：S1 out_idx=[]/守卫/strip、S2 三层动态嵌套 serial+分区计数精确、S3 归约组合链；实现期两处对设计"字面机制"的等价偏航均已论证并文档化——守卫形态 Select 钳位→入口 if（编译器限制，语义等价：互补条件恰一执行/分支不进入零 flag op/全核全 scope 分支一致，§7.5 洁净性论证保持）；§8.2 的"ReduceSum 普通输入 out_idx=[]"按设计原样落地）
- rollback: no
- backup_path: examples/dsa_sparse_finetune/history_version/sparse_mla_impl_s2r2_attempt1.py（修改前备份；unguarded 二进制保全证明 = 对该备份的 structural_equal(map_free_vars=True) + traced script 逐字节一致，generic 与 panel 双双通过）
- instrumentation_cleaned: n/a（first_impl；S1-S3 spike 与验证脚本均在 /tmp/opencode，未入仓；kernel 无 printf/dump_tensor 残留）
- next_hint: Stage 3 性能验证——graph-replay 三连发捕获口径（§10.4：捕获单元 = [violations 置零, detect, panel_g, generic_g]，batch=25 replay/iters=20/2 组）：panel 合格输入预期 ≈155-157us（vs 220us 判定线余量 ~1.4×）；eager host-API 粗信号已采（auto 3.08ms wall / 2.99ms event vs shared 2.13/2.10——host-bound，eager 口径不适用判定，§8.4/§10.4）；随机索引元数据过输入 ~391/483us 预期；auto-miss 零变化（P7d 探针锁定）
- 关键结论:
  - **unguarded 二进制保全（TIR 级证明）**：generic/panel 两 unguarded 变体 traced PrimFunc 与 rev1 备份 structural_equal + script 逐字节一致（守卫经 trace 期纯 Python 闭包直通，零 TIR 语句）；L0-P2 双 block_I golden 回归通过（行为级证明）
  - **guarded 变体行为等价**：P7e 四配置 guarded-exec vs unguarded-generic torch.equal=True + 3-run bitwise 稳定 + 早退 sentinel 保持
  - **三模式语义等价**：P7f/L1 sweep——{auto, generic, shared} 输出两两 torch.equal；路由探针各自精确（generic→仅 unguarded generic；shared→仅 unguarded panel 无 detect；auto→三连发 detect/panel_guarded/generic_guarded 各 1、unguarded generic 0）
  - **auto 门控 miss 零开销零变化**：P7d——detect/guarded builder 零调用 + 三键缺席，unguarded generic 单发
  - **detect 正确性（§9.1 精确断言体系）**：P7a 9 子用例 violations 与 host 分区谓词精确 lane 计数全部相等（分段边界±1/首末行/多违约累计/空 seq/单行 seq/ΣS=1/G=2 ∃g）；P7g ΣS=16384 多 chunk ×3 run bitwise 一致（契约 [0,0,0]、违约 [36,36,36] 精确）

## Attempt 1 (generic-route removal) — 2026-10-11T00:45:00+08:00
- mode: first_impl（DESIGN_generic_p1.md §1.6/§8.6 移除 + 通用基线；P1 明确禁止、未实施）
- classification: precision_pass
- fail_category: none
- test_level: all（L0 先行收敛 51 标记全过后覆盖门禁 + 全量）
- coverage: L0 51 条标记（G1 主用例/G2 双 BI golden/G3 a-e 分布族/G4 10 例 shape 扫描/G5 4+3+1 配置×3 run/G6 锚定）；L1 13 用例（随机全幅索引为主构造）；L2 3 用例全拒；Boundary 4 用例全过；覆盖门禁 23 PASS / 0 MISS / 0 N/A（proto rev3，panel_mode 维度已除）
- boundary_warnings: none（L2 dtype/topk/head_kv 全部正确拒绝 [BOUNDARY_PASS]；Boundary inf/nan/zero/bf16max 结构掩码一致 matched=1.0）
- changes:
  - `sparse_mla_topk_reducesum.py`（1803 → 800 行）：按 §8.6 清单删除 `_reducesum_panel_body`/panel/panel_guarded/detect/guarded 4 builder + `_strip_unused_param` + `_guard_run`/`_guard_v_entry` 闭包 + `_panel_gate_ok`/`_PANEL_V1_MAX_TOPK`/`_token_indices_from_list`/`_replicate_h_of` + host 三模式分发与 panel_mode 形参；`_reducesum_generic_body` 摘除 Violations 形参与守卫（签名 8 参 + 9 参 prim_func，out_idx=[6]/workspace_idx=[7,8] 与基线 strip 后逐值一致）；`_BUILD_CACHE` 收缩单键；host 回 rev1 形态（shape 推导 → 查建 → prepare_token_indices → 调用）；模块 docstring/__all__/imports 同步收缩（tvm imports 随 strip 机制一并移除）
  - `test_sparse_mla_topk_reducesum.py`（1656 → 1204 行）：删 L0-P1/P3/P4b-e/P6/P7a-g 与 auto_route_probe/panel_builder_probe/cache-key 断言族；新建 L0-G1..G6（§9.2）+ `make_random_causal_indices`（anchor 族 randint(0,topk)+clamp 与全幅 randint(0,S)+clamp 双模式）+ `ref_reducesum_raw`（§1.3 原式参考，服务全 -1 行用例）；L1 通道删 run_shared/run_auto/panel_mode、索引构造升级随机全幅为主；Boundary 去 panel_mode；COVERAGE_MANIFEST 删 D-PARAM-panel_mode
  - 备份：history_version/sparse_mla_impl_p1route_pre_removal.py + test_sparse_mla_p1route_pre_removal.py（修改前）
- error_summary: 开发期 3 轮问题闭环——(1) G4 传参名不匹配（BI/offsets vs 形参 block_I/offsets_list）两轮 TypeError 修正；(2) G6 sanity 的 KV 寻址漏加 bos 偏移（首版 max_abs=127 全错——einsum 索引用 in-seq 相对位置直接索引全局 KV；补 bos 后 rel=1.1e-06）；(3) G6 首测 1472.7µs 假警报——全幅 randint(0,S) 索引构造与 441.7µs 锚点的 randint(0,TOPK) 集中构造不同源，A/B/C 三变量分解实验（A 锚点构造 426.9µs 带内 / B 2-seg 481.3µs / C 全幅 1378.3µs）证实为分布效应非移除回归；G1/G6 改锚点族构造后 (i) 锚点口径 434.7µs（−1.6% 带内）
- design_error_reason: none（移除清单 §8.6 逐项准确，无实施阻塞）——但发现两处设计文档缺陷已留证待 Stage-1 修订（见 issues 级发现，不阻塞本次移除交付）：
  ① §9.2 L0-G4/§9.4 sentinel-2/§4.5 的 topk=256/BI=256 用例不可编译：gemm_v0 transpose-B 路径 nTile=N=BI，K>kL0Size 时 L0B 乒乓槽预算 32KB（src/tl_templates/ascend/common.h L1263-1278 static_assert `nTile*kL0Size*sizeof(T1) <= kL0Budget`）⇒ kL0Size=128/bf16 下 BI≤128 硬上限；§4.5 KVDB 回退分析只查了 L1 容量漏查 L0B。测试侧以 topk=256/BI=128 替换覆盖 topk 扫描值（文件内 NOTE 留证）；P1 attempt 前必须修订（KVDB=1 可编译触达形态存在，如 H=64/D=512/tail=512/topk=128/BI=64 → 2·64·1024·2+2·64·1024·2+4096=528384>523264）
  ② §9.1 make_random_causal_indices docstring 描述"每行从 [0,s_i] 内无放回采样"（全幅构造，实测 ~1378µs）与其自称"同源"的 441.7µs 锚点构造（randint(0,TOPK) 集中，bench_graph_smtk/prof_generic 实读）不一致；本实现按 §9.2 G1"同 perf 锚点"裁定取锚点族（G1/G6），全幅构造保留给 G4/G5/L1 作覆盖
- rollback: no
- backup_path: examples/dsa_sparse_finetune/history_version/sparse_mla_impl_p1route_pre_removal.py（kernel）+ test_sparse_mla_p1route_pre_removal.py（test）
- instrumentation_cleaned: n/a（first_impl；调试脚本均在 /tmp/opencode 未入仓；kernel 无 printf/dump_tensor）
- next_hint: P1 attempt 前置条件——Stage-1 修订 DESIGN_generic_p1：§9.2/§9.4/§4.5 的 BI=256 用例改为可编译 KVDB=1 触达形态（H=64/D=512/tail=512/BI=64 候选已验证预算算式）或明示删除该路径覆盖；§9.1 构造描述与锚点口径二选一对齐。移除锚定已锁定：anchor 构造 434.7µs（−1.6%）/ G1 2-seg 构造 496.4µs（分布效应）；P1 消融 A 组基线可直接取用本会话口径
- 关键结论:
  - **TIR 等价（移除 ≠ 改动，结构级证明）**：移除后 generic builder traced PrimFunc 与 git 基线 0f11b3cc unguarded 变体（= rev1 通用路径 strip 后形态）在 golden_BI128/default_BI32/grouped_BI64 三配置下 structural_equal(map_free_vars=True) + tvm.script 逐字节一致（9 参，/tmp/opencode/verify_tir_equiv.py 可复跑）
  - **L0-G 全绿**：G1 主用例 err_ratio 1.95e-07；G2 GPU golden 双 BI 回归通过（abs_err 5.7e-04 历史口径复现）；G3 分布族 a-e 全过（全同/部分重合/非连续乱序/边界值/全 -1 行 raw 参考 max_abs 1.4e-06 + 行级精确零）；G4 10 例 shape/分段扫描（含空首段/kv_group=2/非默认 D/grid/topk=256）；G5 竞态族 4+3+1 配置×3 run 两两 torch.equal 逐位稳定（含 S=1/21/41 短流与 19 空核洁净退出）
  - **移除 perf-neutral（锚定粗测）**：anchor 构造（offsets=[0,2048], randint(0,128)+clamp, seed=123）graph-replay 434.7µs vs 441.7µs 锚点 = −1.6%（±5% 带内）；同构造基线口径横档 426.7-441.7µs；G1 2-seg 构造 496.4µs 为第二段 KV 窗口分布效应（基线同受，TIR 字节等价 ⇒ 非回归）
  - **全幅随机索引量级发现（供 P1/P2 决策参考）**：randint(0,S) 全幅索引（每行 128 个散布行 gather）实测 ~1378µs = 锚点构造 3.2×——441.7µs 锚点与其全部流水分析（P1/P2/P2b 收益模型、"220µs 不可达"结论）均基于 randint(0,TOPK) 集中构造的 KV 行复用红利；若真实业务索引为全幅分布，中转税与结论量级需重估（分析文档 §0 的 "random" 实为集中构造，建议 Stage-1/3 复核口径）

## Attempt 2 (P1 实施) — 2026-10-11T03:10:00+08:00
- mode: first_impl（DESIGN_generic_p1.md rev2 §6.2 P1 AIC GEMM 流水重构；attempt 1 = 移除已 PRECISION_PASS）
- classification: precision_pass
- fail_category: none
- test_level: all（L0 先行收敛 64 标记全绿 → 全量 all 69 PASS/0 FAIL + 覆盖门禁 23/23）
- coverage: L0 64 条标记（G1 鲁棒性/G2 官方 arange 双 BI/G3 a-e/G4 11 例含 kvdb1_trigger/G5 4+3+1+1 配置×3 run/G6 官方双档锚定）；L1 13 用例；L2 4 用例（+rev2 block_I_gt_128）；Boundary 4 用例；覆盖门禁 23 PASS / 0 MISS / 0 N/A
- boundary_warnings: none（L2 dtype/topk/head_kv/block_I>128 全部正确拒绝 [BOUNDARY_PASS]；Boundary inf/nan/zero/bf16max 结构掩码一致 matched=1.0）
- changes:
  - 框架（爆炸半径 2 文件，commit 由 orchestrator 管）：`tilelang/language/ascend.py`（gemm_v0 新增 opt-in kwarg l1_release_event ∈[0,7]，None 时模板字符串逐字节不变 + docstring）+ `src/tl_templates/ascend/common.h`（gemm_v0 模板追加 `int l1ReleaseEv = -1` 默认参；opt-in 尾部在 M_MTE1 排空等待之前发布 SetFlag<MTE1_MTE2>(l1ReleaseEv) 并省略旧 SetFlag/WaitFlag<MTE1_MTE2>(0) 对；默认 -1 分支指令序列与旧模板逐条一致）。不改 op 注册/codegen pipeline/pipeline_planning/pto（GemmOpCodegen 字符串直通实证）
  - `sparse_mla_topk_reducesum.py`（800→约 870 行）：C 侧 §6.2 重构——KVDB 编译期容量开关（双槽预算公式，523,264B 阈）+ L1EV_BASE=2（MTE1_MTE2 pair id 2/3，与 gemm 内部 L0AB_EVENT{0,1} 错开）+ q_l1/kv_l1 双槽（[2,Hpb,DDT]/[2,BI,DDT]，KVDB=1 时保持基线 2D 单槽形态，形状 builder 层定型规避 TVMScript 分支定义作用域限制）+ init set_flag×2/每块 acquire wait_flag(2+p)/gemm_v0(l1_release_event=2+p 奇偶拆分双静态模板调用——模板参须编译期而 p 为运行时，S-P1b 验证形态)/destroy wait_flag×2 + Q 装载块内化（NI==1 or i<2，acquire 保护跨 task 重写）+ builder 前置可读 assert block_I≤128（R10）；V 侧逐条零改动
  - `test_sparse_mla_topk_reducesum.py`（rev2 连带更新）：G4/G5 补 kvdb1_trigger（H=64/D=512/tail=512/BI=64，KVDB=1 回退臂，§4.5 全预算核验形态）；G1 降格标注 robustness/G2 升格标注 official（构造同一性=口径锁定）；G6 重写为 official arange 双档（基线 364.7/475.3、P1 预期带 330-345、R9 证伪带自动判定输出）；L2 补 block_I_gt_128 拒绝用例（D-EXC-SHAPE 2→3）
  - `perf_tuning/p1_ablation_report.md`（新增）：§9.5 Stage 2 粗测锚定 + R9 证伪带判定 + 归因二分实验全数据
- error_summary: 开发期 3 轮问题闭环——(1) TVMScript 解析器不接受 if 分支内定义的 alloc 变量跨分支使用（q_l1 双臂定义报 Undefined variable）→ 形状 builder 层定型 + 无条件 alloc；(2) 诊断变体 p1_gate 首版误留 acquire 而无 release → 真机挂死（aicore timeout）→ 修正为无 acquire（旧门控超集保护）——该事故反向实证协议配平必要性（wait 无对应 set 必挂死）；(3) G6 首会话 425.0µs 假回退信号——跨会话热态漂移实测 ±5%（同一二进制 360.5-425.0µs）超出文档化带 → 全部判定改用同会话配对 A/B 口径
- design_error_reason: none——但 **R9 证伪带触发（实测确认，见关键结论）**：P1 BI128 全部绝对锚点 >345µs 且同会话配对相对基线 **+4.0~+6.0% 净回退**（非零收益）；按任务书与 §9.5 证伪逻辑此为如实标注项（正确性是 PASS 判据），非 attempt 失败、非 DESIGN_ERROR（R9 为设计预注册风险，处置路径=Stage 3 AIV/AIC 联合瓶颈分析）
- rollback: no
- backup_path: examples/dsa_sparse_finetune/history_version/sparse_mla_impl_pre_p1.py（kernel）+ test_sparse_pre_p1.py（test）
- instrumentation_cleaned: yes（kernel/test 零 printf/dump_tensor 残留——grep 实证；spike/诊断脚本均在 /tmp/opencode/p1/ 未入仓）
- next_hint: Stage 3 AIV/AIC 联合瓶颈分析（R9 处置路径）——① msprof 引擎占空比（本环境 msprof 导出工具缺失，仅原始 sample 数据，需补工具链或换环境）；② V 引擎和/墙钟比值 → P2b（LEAD=3）+P2（SIDE_PAIR=False）候选立项；③ C 侧 MTE2 突发×V 带宽耦合机制纳入归因（p1_ablation_report §3 新发现：52.7GB/s 为旧门控节奏下的条件有效带宽）；④ 若判定 C 侧优化暂不可收：§10.4.1 一行降级开关（KVDB 置 1）恢复旧语义而保留框架 opt-in
- 关键结论:
  - **首日 spike 全过（S-P1a/b/c/d）**：S-P1a 9 参模板渲染+编译+正确（rel 3.3e-07）；S-P1b **R1 latch 语义真机裁决成立**（协议原形 S∈{1,2,3,64,2048} 全正确 + 3-run 位稳 + graph replay ×3×25 事件状态闭合 + 空核洁净退出 + 最坏 MTE2 全速 pacing）——历史"错位 marker 挂死"根因确认为 (源管道,id) 槽位复用冲突而非 latch 缺失（L1EV_BASE=2 显式错开后无冲突）；S-P1c 既有用户生成 C++ 字节级一致（transpose-B + plain 双路径）+ example_gemm/example_gemm_persistent 运行 "Kernel Output Match!"；S-P1d 性能冒烟即 R9 判定输入（见 p1_ablation_report）
  - **P1 正确性完备**：输出与基线 torch.equal 逐位相等；L0 双会话全绿（含 G5 竞态族 9 配置×3 run 两两位稳、短流 S=1/21/41、kvdb1_trigger 双臂）；KVDB=1 回退臂与移除后基线 TIR structural_equal(map_free_vars=True)（/tmp/opencode/p1/verify_kvdb1_tir.py 可复跑）
  - **R9 成立（显著标注）**：P1 BI128 锚点 360.5-425.0µs（4 会话，全 >345µs 预期带）；同会话配对 base 370.2-375.7 → P1 385.1-398.2µs（**+14.9~+22.5µs / +4.0~+6.0%**）；BI32 档 +0.4% 中性。归因二分：p1_gate（全结构组件+旧门控）=375.0µs ≈ base（**R2' 乒乓回归排除、结构零成本**）；p1_lite =397.7µs ≈ p1（**回归源=解除 MTE2 自门控本身**）；C 侧隔离微基准 optin −2.0%（协议按设计工作）⇒ **负收益仅在 V 耦合 regime 出现**：解门控后 C MTE2 突发读与 V 行读/bulk 写争共享 L2/DRAM（52.7GB/s 为旧门控节奏下的条件有效带宽，非可独占引擎下限）。p1_hoistq（Q 前移）410.3µs 劣化——Q 位置假设证伪，现行 §6.2 忠实序保留
  - **框架 opt-in 双保险验证**：默认路径模板字符串逐字节不变（S-P1c）+ C++ 默认参 -1 分支指令等价 + 两个真实用户示例运行通过——80 个既有 gemm_v0 用户零影响
  - **§9.4 sentinel 补充（收尾轮）**：sentinel-3（kvdb1 回退性能守门）以**二进制同一性判定 PASS**——KVDB=1 配置下 P1 builder 与基线生成内核源码逐字节一致（14282B）+ TIR structural_equal；运行时配对 +4.3% 经同码双实例对照实验（+0.8% 均值 / round-0 +2.6% 后收敛）归因为实例/地址级测量伪影；输出 torch.equal。sentinel-1/2 即 G6 双档（见上）。终验（最终交付态：G6 输入序修正 + reformat 后）--level all：69 PASS / 0 FAIL / 8 BOUNDARY_PASS / 0 WARN，exit 0；G6 锚定 BI128=364.8µs（+0.0% vs 364.7）/ BI32=476.1µs（+0.2% vs 475.3），R9 证伪带维持触发（全档 >345µs）

## Attempt 3 (P1 rollback) — 2026-10-11T04:20:00+08:00
- mode: rollback_impl（P1 性能证伪后的外科回退；attempt 2 = P1 正确性全绿但 R9 成立）
- classification: precision_pass（回退后全量全绿）
- fail_category: none
- test_level: all（一次全量 + G6 锚定复跑一轮）
- coverage: L0 69 条标记全 [PRECISION_PASS]（G1 鲁棒性/G2 官方 arange 双 BI/G3 a-e/G4 11 例含 h64_tail512_bi64/G5 4+3+1+1 配置×3run/G6 官方双档锚定）；L1 13 用例；L2 4 用例；Boundary 4 用例；覆盖门禁（proto rev3）23 PASS / 0 MISS / 0 N/A
- boundary_warnings: none（L2 dtype/topk/head_kv/block_I>128 全部正确拒绝 [BOUNDARY_PASS]——block_I_gt_128 在回退态由框架 common.h static_assert 以 RuntimeError 拒绝；Boundary inf/nan/zero/bf16max 结构掩码一致 matched=1.0）
- changes:
  - 回退（git checkout 4e153c7f，三文件 diff-空实证 0 行）：`sparse_mla_topk_reducesum.py`（回到移除态单 builder，无 KVDB/L1EV/token 代码）+ `tilelang/language/ascend.py`（gemm_v0 移除 l1_release_event opt-in kwarg）+ `src/tl_templates/ascend/common.h`（gemm_v0 模板移除 l1ReleaseEv 尾部分支）；P1 实验态完整存档于 git 00a93f81（含消融报告），需要时 cherry-pick
  - 测试外科处理（以 4e153c7f 版为底）：**保留** rev2 口径改进——G1"鲁棒性用例"标注、G2"官方口径锚定"标注、G6 官方 arange 双档（BI128=364.7µs/BI32=475.3µs 基线锚 + ±5% 带报告）、L2 block_I_gt_128 域外拒绝用例（D-EXC-SHAPE 2→3）；**移除** P1 专属——KVDB 回退臂语义标注（kvdb1_trigger 改名 h64_tail512_bi64，降格为普通大 DDT shape 覆盖用例，配置 H=64/D=512/tail=512/BI=64 在旧 kernel 域内合法保留）、G5(4) 同配置改标 largest-DDT race surface、G6 P1 预期带 330-345µs/R9 证伪带自动判定逻辑、模块头/NOTE 的 P1 引用（改记回退事实）
  - 备份：`history_version/sparse_mla_impl_p1_pre_rollback.py`（P1 终态 kernel，回退前快照）
- error_summary: none（回退后无任何编译/运行/精度错误）
- 官方口径锚定复测（graph-replay，seed=42 逐字，双档）：
  - 首轮：BI128=338.6µs（-7.1% vs 364.7，超 ±5% 带偏快侧）/ BI32=477.1µs（+0.4%，带内）
  - 复跑：BI128=359.6µs（**-1.4%，带内**）/ BI32=476.2µs（**+0.2%，带内**）
  - 判读：kernel 与 4e153c7f 逐字节一致（diff-空证明）⇒ 首轮 -7.1% 为会话级设备热态漂移（历史同二进制跨会话 360.5-425.0µs 实测带，attempt 2 记录），非回退漂移；复跑双档带内确认无漂移
- design_error_reason: none
- rollback: yes（P1 实验改动回退至 4e153c7f 基线；本 attempt 即回退执行）
- backup_path: examples/dsa_sparse_finetune/history_version/sparse_mla_impl_p1_pre_rollback.py
- instrumentation_cleaned: n/a（无调试插桩；回退本身即清理）
- next_hint: 交付态 = 4e153c7f 基线 kernel + rev2 口径测试。后续性能路线（Stage 3 可选）：① R9 处置=AIV/AIC 联合瓶颈分析（msprof 引擎占空比，本环境工具链缺失）；② P2b（LEAD=3）/P2（SIDE_PAIR=False）V 侧候选立项——P1 教训：C 侧 MTE2 解门控与 V 带宽耦合，V 侧减负或更优收益方向；③ 框架 opt-in 能力需要时从 00a93f81 cherry-pick（gemm_v0 l1_release_event + 模板尾部分支，S-P1c 已验证默认路径字节级不变）

## Attempt 3 (P2+P2b 组合实施) — 2026-10-11T15:05:00+08:00
- mode: first_impl（P2 + P2b 四点消融 + 全量回归；基线 25c0da76 = P1 回退态）
- classification: precision_pass
- fail_category: none
- test_level: all（4 臂快速正确性 → 消融配对 → 交付态全量 all + 第 2 轮 l0 重压）
- coverage: L0 57 标记（G1-G6，含新增 grouped_short_gl_window）×2 轮；L1 13；L2 4；Boundary 4；覆盖门禁 23 PASS / 0 MISS / 0 N/A
- boundary_warnings: none（L2 4 用例正确拒绝；Boundary 4 用例结构掩码一致 matched=1.0）
- changes:
  - `sparse_mla_topk_reducesum.py`（+93/−34）：**P2 交付**——SIDE_PAIR 变量与 gather 段相邻 mte3→mte2 保护对物理移除（仅 SIDES=LEAD 的 UB 降级档曾存在，如 BI=128@DDT=576 / BI=64@DDT=1024），改依自然排空窗口（稳态余量 ≥4.7×：bulk(gn-SIDES) 发射与行读发射间隔 2 个含 waitF1 的 body ≥6.6µs + MTE2 前压 2.8µs vs bulk 排空 ≤2.0µs；MTE3 占用 ≤45% 无积压）；SIDES/SIDE_PAIR/主循环/kv_gather_ub 四处注释重写为两档安全性推导；**g_l 缺陷修复**（idx(LEAD) 预发射组号 b_s_l→pid_l，一行 + 论证注释）；模块 docstring 更新（P2 交付 + P2b 回退记录）
  - `test_sparse_mla_topk_reducesum.py`（+5）：G4 新增 grouped_short_gl_window（S=48/H=32/G=2/topk=64/BI=64——g_l 缺陷 manifest 窗口回归锁定）
  - `perf_tuning/p2_p2b_ablation.md`（新增）：四点消融全数据 + 实例偏置方法论发现 + 裁决根因
  - 备份：history_version/sparse_mla_impl_pre_p2p2b.py（修改前）
- error_summary: 开发期 2 项关键发现闭环——(1) **基线既有 g_l 缺陷**（P2b 强制正确性分析产物）：L404 组号误用行内号 b_s_l（恒 0），kv_group>1 + NI=1 + 短流窗口（task(LEAD//NI) 落 g≠0 组）取错组 Indices——S=48/H=32/G=2 实证复现（max_abs=1.45e-2，g=1 行 1-11 精确命中预测；base/p2 臂 FAIL、修复臂 PASS），既有全部用例不触发（task(LEAD) 恒在 g=0 或 NI≥2 路径）故历史全绿；(2) **实例双峰地址偏置**：同码 jit 实例可差 +4%（p2#1 持续 355-370 vs p2#2 恒 347），单实例配对分辨率地板 ±0.8-4%（BI32 对照臂四臂 TIR 全同实测 ±0.75% 假差异）——改用每臂 5 实例交错构建 × 2 sweep 的分布级裁决
- design_error_reason: none（P2b 依赖正确性分析完成：#18 不变量重推导——(i) 标量窗 (t+LEAD)//NI=3 ∈ {0, OFFSET_N=3} ✓；(ii) prologue side 首用互异 assert 在 LEAD=3>SIDES=2 触发 → 分析结论 = 需 s≥SIDES 段一次性相邻 mte3→mte2 guard（architectural，已实现于消融臂）；(iii) IDXSLOTS=LEAD+2=5 参数化充分 ✓；槽复用安全 = LEAD 不改变 side 复用距离（恒 SIDES=2 块）与发射间隔（恒 2 个 waitF1 body）→ 余量不变；ws/F0 配对与 LEAD 无关（计数=块数）✓；#19 hoist 窗需 (LEAD+1)·grid 参数化 ✓——P2b 技术上可行且正确，**因性能回退被用户规则裁掉，非依赖不可行**）
- rollback: partial（P2b 按用户规则回退：+3.23% 三轮一致负收益；P2 保留交付）
- backup_path: examples/dsa_sparse_finetune/history_version/sparse_mla_impl_pre_p2p2b.py
- instrumentation_cleaned: n/a（first_impl；消融/诊断脚本均在 /tmp/opencode/p2/ 未入仓；kernel 无 printf/dump_tensor）
- next_hint: P2 后官方口径 ~345-356µs（配对 −1.5% + 带内锚定），距联合 bound ~333µs 带宽地板余 ~4-7%；后续方向按 joint_bottleneck_analysis §2 裁决表：P2b 谱系（发射点前移）已实证负收益不再攻；剩余开放角度为流量消除（超算子层权限）与 L2 服务率混合比扫描（Stage 3 开放项）
- 关键结论:
  - **四点消融表（BI128 同会话配对 3 轮）**：base 348.8 / P2 348.3（−0.15%，单实例噪声带内）/ P2b 360.1（**+3.23% 三轮全负**）/ P2+P2b 350.0（+0.33%）；BI32 对照（四臂 TIR 全同）±0.75% = 实例偏置地板
  - **P2 终裁（每臂 5 实例交错 × 2 sweep）**：base 355.7（spread 29.6µs 双峰）/ P2 350.2（spread 7.8µs 无离群）= **−1.56%**，与 Stage 3 spike −1.53% 跨会话复现一致 → 交付；实例方差 29.6→7.8µs 收窄为附带结构收益（msprof 机制证据：V overlap 1.031→1.102、C ws 读率 55.3→59.8GB/s）
  - **P2b 根因**：LEAD=3 使 V 行读更早脱离 flag 节流 = 更少仲裁（趋向 K3 自由跑端 2.57 vs 2.96TB/s），与联合分析"改进空间只在更平滑的仲裁"结论自洽；P2+P2b（+0.33%）劣于 P2 单独亦印证（P2 平滑化部分抵消 P2b 去仲裁化）
  - **正确性**：4 臂 × 7 用例族全绿（除 G 用例 base/p2 臂的既有 g_l 缺陷——独立于 P2/P2b，已修复并锁定）；交付态全量 70 PASS/0 FAIL + 第 2 轮 l0 57/57（G5 竞态族 9 配置 ×3 run 两轮位稳、短流 S=1/21/41 洁净、NI≥2 档 E/F 用例位稳）+ 覆盖门禁 23/23
  - **交付态 G6 官方锚定**：BI128 = 363.1/355.2µs（两轮，−0.4%/−2.6% vs 364.7 记录锚）；BI32 = 474.7µs（−0.1%）

## Attempt 4 (P3 数据流重构) — 2026-10-12T14:30:00+08:00
- mode: first_impl（DESIGN_dataflow.md §5 P3 序列级去重驻留；基线 22bc68a8）
- classification: design_error（§6.2 预注册证伪条件 1 触发：官方 e2e 894µs ≫ 220 证伪线——正确性全绿，证伪纯因性能；已按用户规则回退基线）
- fail_category: design_perf_falsified（选值机制 issue-latency 界 + gather 硬件墙 + panel 地板回退三重归因，见 perf_tuning/p3_dataflow_report.md §2）
- test_level: L0-D 快速族（D1-D8b + D7 竞态 ×3，/tmp/opencode/p3/quick_d.py）全绿后证伪回退——全量 --level all 未跑（证伪后按规则回退，基线全量状态 = 22bc68a8 已验证态）
- coverage: L0-D 11 用例族（D1 官方/D2 回退/D3 重复/D4 乱序/D5 -1·越界·全-1/D6 G=2 大 P_seq/D7a·b 竞态与混合路径/D8 分组分片含空首段与 RH=2）——全部 PASS 含路径断言（Violations 值）
- boundary_warnings: none
- changes（全部已回退，存档于 history_version/sparse_mla_impl_p3_falsified.py 1560 行）:
  - prep kernel（V-only 40 lane：链式相邻行共享检测 + |idx|>2^24 幅值门 → Violations atomic）
  - panel_guarded kernel（组调度/8-flag/staging 复用 0f11b3cc + 转置 gemm M=topk（spike-1 位等价实证）+ V 侧 sort/gather_mask 组内 panel 构造 + reduce-first（mask 从 h 求和提出）+ 等值比较矩阵选值 + k 计数归一（重复列精确）+ h-chunk 交错装载）
  - generic_guarded（P2 基线 + 入口守卫 Violations>0，0f11b3cc §8.2 机制）+ _strip_unused_param 公开形复原
  - host 三连发分发（静态门控 _panel_domain_ok → prep+panel_g+generic_g / 单 generic；API/proto 零变化）
  - 修复前备份 + common.h gatherb 模板试验（中途已还原）
- error_summary: 实现期 3 轮缺陷闭环——① clamp count=输出总尺寸（2D 传 HALF 只处理首 64 元素）；② h-chunk 装载/处理单循环交错（分离双循环致 accT_ub 末 chunk 覆写，H≥32 全错）；③ UB 32B 对齐（移位拷贝未对齐致 aicore 异常 → 改 k 计数归一消除移位）。spike-2 裁决：910B3 gatherb 仅 uint16/uint32（fp32 静默无效实测）+ 元素 Gather 需 uint32 计算偏移（int32→uint32 vconv 无实现，CAST_NONE/RINT 实测编译失败）⇒ 选值改 R1 备选链 UB 内精化形态（等值矩阵）。性能证伪链：官方 e2e 894µs（3 轮 893.5/894.6/896.4）；分解：prep 29.3 / launch 25.3 / panel 857（V3 地板二分：C+握手+装载 291 + V 计算 ~565）；根因 = 通用选值 36 条串行向量指令/task ≈ 5.5µs（153ns/issue 依赖链延迟）×102.4 task —— 设计 10-20µs 选列预算低 30×；地板 291 vs Plan-A 143（+148，嫌疑 = 转置 gemm L0A 满预算 tile 乒乓余量归零，未单独实证列开放）；迭代不可达算术：三项合法优化全做最优 350-450 仍 2× 超线，达 220 需 fp32 硬件 gather（API 面封锁）
- design_error_reason: DESIGN_dataflow.md §6.2 证伪条件 1（官方 >220µs）触发 + §5.4 选值机制在 910B3 API 面上不可达设计预算（§8 R2 备选链同样超预算）；备选链建议见 p3_dataflow_report.md §6（框架 Gather int32-offset 泛化 + pos 表缓存 + 非转置 gemm 组合 / 或接受 363.1 基线）
- rollback: yes（kernel git checkout 至 22bc68a8；test 未改；common.h 试验已还原；工作树 = 基线态）
- backup_path: history_version/sparse_mla_impl_pre_p3.py（基线备份）+ sparse_mla_impl_p3_falsified.py（P3 全实现存档）
- instrumentation_cleaned: yes（P3 实现已整体回退出仓；诊断脚本均在 /tmp/opencode/p3/）
- next_hint: 220µs 目标在本硬件 API 面下应重新评估可达性（本轮证据：无契约选值的 V 侧指令延迟界 + gather 双墙）；若继续攻：先单独 spike 框架 Gather int32-offset 泛化的真机时序（10 行 codegen 改动 + uint32 用户零影响），再决定是否立项组合方向；流量消除机制本身已验证成立（995→118MB）
- 关键结论:
  - **正确性/通用性达成**：P3 全链（prep 资格判定 + panel 去重驻留 + 双发射回退）对 D1-D8b 全绿——panel kernel 域 {P_seq ≤ topk} 内重复/乱序/-1/越界/空段/G=2/RH=2 全部正确，k 计数归一对重复索引精确，路径断言（Violations）全过
  - **流量账本验证成立**：995MB → ~118MB（8.4×）——但消除带宽瓶颈后暴露 V 侧 per-task 指令延迟瓶颈（565µs），通用选值是最大加项
  - **性能证伪（决定性）**：官方 e2e 894µs = 基线 +146%；分解归因三重（选列 14× 超线 / 地板 +126µs / gather 硬件墙）；单轮定向迭代不可达性已算术论证
  - **回退完成**：工作树 = 22bc68a8 基线（P2 交付态 825 行），P3 实现存档可复现

## Attempt S-A1 — 2026-10-12T21:30+08:00
- mode: spike_bench（新流程 Stage A：五候选可行性验证 + 微基准；不整合 kernel）
- classification: spike_done（[SPIKE_DONE]，无 [PRECISION_*] 语义）
- fail_category: none
- test_level: n/a（微基准：MB-0/1/3 每 task 选值链族 + MB-4 gemm 地板族 + C2 可行性探针；全部真机 ≥3 轮中位，MB-4 同会话交错配对）
- coverage: MB-0a/b/c（P3 链 1.87µs·选值隔离·双流交错-0.8%）+ MB-1a/b/c（gather 链 0.23µs·隔离·串行依赖）+ MB-2（C2a 位保真 OK / C2b fp32→u32 值转换编译死）+ MB-3（c3_full select-first 0.19µs）+ C4（sort 5.57µs/组·count 6.71µs/组→摊销 0.11-0.13µs/task）+ MB-4（T/N gemm 方向 -4.1%·nogemm=qonly→gemm 零增量·qonly=Q 装载 5.4µs/task）；正确性：p3_full 306/306 task、g_full/c3_full/grp_* 全对、T vs N 6/6 + golden rel 4e-7
- boundary_warnings: none
- changes: 交付 perf_tuning/gather_microbench.md（五候选证据+数据表+依赖链分析+排序+三类限制表+组合算术）；框架 spike diff（codegen_ascend.cc GatherCodegen +16/-1，int32 offset 自动 ReinterpretCast）已验证已回退（patch /tmp/opencode/mb/c1_spike_gather_int32_offset.patch）；P2 基线 kernel/test 零改动（git 状态核实）；build/ 从回退源码重建
- error_summary: ① P3 根因归因证伪——153ns/instr 是 5.5/36 的循环论证，实测链仅 1.87µs（62ns/instr），其余 ~3.6µs 为 loads/flag 地板（empty=5.94µs/task）；② C5 证伪——转置/非转置 gemm -4.1%（配对 5/6 胜），nogemm≈full、qgemm≈qonly → gemm 两方向均完全隐藏，panel 地板真凶=每 task 18KB Q 装载串行（5.4µs/task 未流水）+FIX/握手；③ V pipe 独立链无重叠（ilv2）→ 减指令数是唯一杠杆，≤16-20 指令链藏入地板阴影；④ P3 漏掉的现成路径：T.reinterpretcast（零指令位视图）+ T.tile.gather 即可 int32 偏移选 fp32 值（cann-bench gather.py 20/20 复跑为证）；⑤ 新 API 限制：f32 视图别名不可作 T.copy 存储源（实测垃圾值），别名仅可被 V 算子消费
- design_error_reason: none（无不可逾越墙：#1/2/5/7 API 限制全部有已实测的桥，#3/4/6 硬件限制全部有规避形态，#8/9/10 属算法设计选择）
- rollback: no（无 kernel/test 改动；框架 diff 单独回退并留 patch）
- backup_path: n/a（/tmp/opencode/mb/ 全套脚本+日志+patch）
- instrumentation_cleaned: yes（探针仅存 /tmp；仓库内零残留）
- next_hint: 胜者整合建议 C1+C4（保持转置 gemm [P,H] 以兼容后续 C3；C5 可顺手取 -4% 但与 C3 互斥）；预期官方 BI128 ≈660-690µs（P3 的 1.3×，仍 3× 于 220）——若目标是 220，下一阶段必须攻每 task 装载/握手粒度（4-task 合批估算 e2e≈300-350µs，再压 prep 29.3+launch 25.3 才触 220 带），该数据已在本 spike §6 给出；bench 结构坑：紧邻 GM→UB 装载与消费者有时序竞态，须用 P2 同款槽位环预取（本 spike 全部 bench 已采用，306/306 稳定）

## Attempt B1 — 2026-10-12T23:55+08:00
- mode: first_impl（Stage B：Stage A 胜者 C1+C4 整合进 P3 存档 → P3-v2）
- classification: precision_pass（[PRECISION_PASS]）
- fail_category: none
- test_level: all（L0-G1..G5 + L0-D + L0-G6 + L1 + L2 + boundary 全量）
- coverage: L0 86×[PRECISION_PASS] + 8×[BOUNDARY_PASS] + 0 FAIL/WARN（P2 套件 70 + L0-D 16）；覆盖门禁 23 PASS/0 MISS/0 N/A
- boundary_warnings: none
- changes: kernel = P3 存档 + C1（reinterpretcast 位视图 + T.tile.gather 单指令选值 + 匹配守卫）+ C4（组切换处 pos 计数矩阵构造，复用 eq 槽位）；test +148 行（L0-D 家族门控用例，自 P3 会话 quick_d/smoke_d1 恢复，修 D7b 参数序 bug）；交付基线 22bc68a8 无 commit；框架零改动（git diff src/ tilelang/ 空）
- error_summary: 整合期两坑闭环——① reinterpretcast 别名作用域（组切换 if 内声明对 post 不可见→dummy 槽垃圾偏移→aicore 越界；与 alloc 同层→C++ 重定义；解法=零成本声明放主循环体首语句）；② 移植 D7b 参数序 bug（两轮全量重跑）。性能：官方 BI128 = 729.0µs（4 轮 726.5-736.5）——Δ 算术精确命中（P3 899.0 − 170.0 = 预测 168µs 选值收益，误差 1.2%；panel 腿 163.3µs = 1.60µs/task vs 预测 1.64，误差 2.5%）；高于 Stage-A 模型带 690 上缘 +5.7%（<10% 规则线，不回退）；带差归因 = 真实 8-flag 协议 + 每 task GM 标量读的地板比 Stage-A bench 结构更重（V 侧仍临界路径 6.69µs/task）。P3 复现 899.0（vs 其会话 894，+0.6%）；P2 基线同会话 378.1（带内）
- design_error_reason: none（reinterpretcast 形态在完整 kernel 上下文成立——作用域坑已解；C1 整合可行性 ✓）
- rollback: no（正确性全绿 + 729.0 未超带 10%；P3-v2 留工作树作粒度阶段基座，回退路径备份齐全）
- backup_path: history_version/sparse_mla_impl_pre_p3v2.py + test_sparse_pre_p3v2.py（修改前 P2 态备份）
- instrumentation_cleaned: yes（诊断脚本均在 /tmp/opencode/mb/，仓库零残留）
- next_hint: 粒度阶段（任务合批）基座数据已就绪——四腿分解：prep 27.9 / panel 685.1（6.69µs/task，V 临界）/ generic-ee 22.7 / 全量 729.0；选值已免费（0.50µs/task 含守卫）；下一杠杆 = 每 task 装载/flag 摊销（4-task 合批估算 e2e ≈ 300-350µs，Stage A §6 算术）+ rows_f/pos 按 pslot 双缓冲（继承限制 #1 一并处理）；220 需再压 prep 27.9 + launch 22.7
- 关键结论:
  - **C1+C4 整合成功且收益与 Stage A 微基准逐点吻合**（选值 −1.60µs/task，预测误差 2.5%）—— spike→整合的预测链路可信
  - **P3 根因证伪的最终确认**：P3 899 → P3-v2 729 后，选值不再是瓶颈；剩余 509µs（vs 220）全部在每 task 装载/flag 地板（V 6.69µs/task）
  - **正确性域与 P3 逐位一致**（守卫设计）：匹配=单槽精确（重复值等价退化），不匹配=0（k=0 等价）；D1-D8b 值与 P3 报告逐字同级
