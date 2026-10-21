# PR Description — indexer_bwd, official checklist version

> Fills `.github/pull_request_template.md` of tilelang-ascend. Use together with
> `PR_DESCRIPTION_indexer_bwd.md` (the technical body). Content in English per
> the unified-template rule; the fixed template headers remain as-is.

---

<!--
感谢贡献！请只勾选本 PR 已实际完成的项目；不适用或跳过的项目请写明原因和跟踪 Issue。
可在文末“其他补充”区域自由增加项目、方案、测试结果或风险说明。
-->

## 变更摘要

Add `indexer_bwd` — the DSA sparse-finetune lightning indexer backward kernel for Ascend NPU — under `examples_experiment/dsa_sparse_finetune/`, together with its PyTorch-autograd-golden test. One commit, two new files, 1294 insertions; no framework, interface, or existing-operator changes. The kernel is dual-backend: the default target compiles through the AscendC pipeline and explicit `target="pto"` is fully verified with identical numerics. Architecture, precision matrix, and performance details in the PR body.

## 变更类型

- [x] `[Example]` 示例/Benchmark

## 1. 算子合入标准

- [x] 新增算子在默认场景下通过：未显式指定 `target="ascendc"`，PTO 后端已完成编译、运行和精度验证；如不适用，已说明原因。
  → Default scenario (no explicit `target=`) fully verified — compile, run, precision — with the actually-selected backend captured from the compile invocation: `LibraryGenerator("auto", "A2")` → bisheng `-xasc --npu-arch=dav-2201` (AscendC pipeline). PTO is verified explicitly: `target="pto"` → bisheng `-xcce`, full matrix PASS (see below). Note the template's "default = PTO" wording does not match current code behavior (`tilelang/jit/adapter/libgen.py` groups `auto` with `ascendc`); both pipelines are reported with captured evidence.

- [x] 适用时 AscendC 与 PTO 双后端均已通过；不支持的 dtype/shape/后端行为已显式说明，没有静默跳过。
  → Both backends verified from one source: AscendC (`-xasc`) and PTO (`-xcce`) each pass the full verification matrix (L0/L1 7 cases + L2 4 cases; blocking 25/25 PASS) with identical matched_ratio / max_abs on every case. No shape silently skipped: the fast-path domain gate falls back to the generic kernel, and out-of-domain inputs raise `ValueError`. Known pto-isa limitations were worked around at the operator level (uint8 row-packed masks `[R, C//8]`, whole-pair GM→L1 loads + host-staged edge panels instead of per-row GM→L1 copies, `vid == 0` guard on the generic atomic scatter).

- [x] 未影响现有算子用例、公共接口和已有后端行为。
  → `git diff upstream/ascendc_pto..HEAD` = 2 added files only (`example_indexer_bwd.py` 1178 lines, `test_example_indexer_bwd.py` 116 lines), 1294 insertions.

- [ ] `examples/` 下的算子有性能要求和代表性 shape；历史遗留且暂未达标的实现已迁移到 `examples_experiment/`，或有维护者批准的豁免。
  → N/A: new operator placed under `examples_experiment/` following the recent precedent of #1730 (deepseek_nsa); not an `examples/` operator.

- [ ] 有 AscendC 参考实现时，TileLang 性能达到 AscendC 的 50% 以上；如有 GPU 基线，候选耗时不超过其 2 倍。
  → N/A: no public AscendC reference implementation of the lightning-indexer backward exists; no GPU baseline measured for this operator.

- [x] 无 AscendC 参考实现时，已检查性能反模式清单，并记录保留项及原因。
  → Retained items with reasons are listed in the performance-gap disclosure of the PR body (per-row gathers on non-arange paths, per-row atomics on layout=1/generic, grain-match double casts, tail-only row loops).

- [x] 优先使用 Developer 原语；使用 Expert/Mixed、手写 intrinsic 或特殊实现时，已说明原因和验证结果。
  → Pure Developer mode: `alloc_shared`/`alloc_fragment`, 4-True pass_configs, no manual `T.Scope`/flag/barrier, no handwritten intrinsics. The `(cid, vid)` two-axis form and the `vid == 0` guard on the generic kernel's atomic scatter are DSL-level constructs, not Expert-mode synchronization primitives.

- [ ] PR 已附精度/性能测试报告，以及典型 shape 的 simulator 结果截图；报告包含环境、shape、baseline、候选结果、单位和复现命令。
  → Precision report attached (official dual-gate matrix, environment, shapes, units, repro command) and performance report attached (event-based per-kernel timings on both backends with method disclosure). Simulator screenshots not included: the camodel simulator is installed but was not run for this large persistent kernel; precedent #1730 merged without them. msprof Task Duration could not be collected in the dev environment (CSV export defect in the local CANN 9.2.0 toolchain) — event-based numbers with the same-method AscendC↔PTO ratio (1.054×) are reported instead.

- [x] 新增单算子的全量测试用例耗时不超过 2 分钟；若超过，已通过 pytest marker 仅保留核心用例在 PR 阶段执行，其余用例纳入每日/定时全量，并记录耗时和标签。
  → The committed suite is a single core case; cold-cache CI sequence for the directory ≈ 60 s (JIT compile of two kernels + test), well under 2 minutes. No marker split needed.

性能摘要/报告链接：see the Performance section of the PR body.

## 2. 框架及接口代码合入标准

- [ ] 框架或接口修改已在双后端/默认 target 下验证，不影响现有算子用例；能力边界已明确。
  → N/A: no framework or interface code modified (two new example files only).

- [ ] 已新增或更新**代表性 ST/UT**，覆盖默认 target、关键边界和适用后端。
  → The committed test covers the default-target representative case (varlen 1023+1025 → fast/layout=0) against the autograd golden, enforces the official bf16 dual-gate (`check_precision`, `AssertionError` on violation, `Test Passed!` marker + exit 0 on pass — matching the CI runner's criterion), plus the module `__main__` shape/finiteness smoke. Extended L1/L2/Boundary coverage (14 cases / 34 checks; all 25 blocking checks PASS on both backends) was run at PR time with a temporary harness against the same committed kernel and golden, and is reported in the PR body; it can be folded into the committed test file on maintainer request.

- [x] 算子及对应测试用例已在本地多轮重复验证，未发现概率性失败；如出现偶发问题，已定位修复或在“其他补充”中说明。
  → Multi-run verification on both backends with byte-identical outputs (default case repeated across sessions; generic path verified deterministic across repeated runs).

- [x] PR 阶段只保留核心冒烟/L0 与关键回归用例；核心回归不得用 marker 隐藏。
  → Single core case in the committed test; no markers used, nothing hidden.

- [ ] 非核心扩展组合统一标记 `low_priority`，由每日/全量任务使用 `-m "not ci_skip"` 执行；`ci_skip` 仅用于已有 Issue 的已知不稳定或硬件限制。
  → N/A: no non-core extension cases exist to mark.

- [ ] 已避免无意义泛化、重复编译和重复执行；新增/迁移测试已登记 `ci/operator_test_manifest.yaml`，legacy runner 与 pytest 不重复。
  → Not registered in the manifest, following the deepseek_nsa (#1730) precedent: with no manifest entry, both files are collected and executed by the legacy script runner in the incremental CI path for this directory (`examples_experiment/dsa_sparse_finetune/` → incremental dir run), with no double execution.

- [ ] 如未新增或迁移算子测试，已注明 manifest 不适用及原因。
  → A test was added (registration rationale above), so this item does not apply.

- [ ] 已更新对应文档，明确 API、后端、dtype、shape 或硬件能力边界；无文档变更时已说明原因。
  → No docs change: the example is self-documenting (module and `indexer_bwd_interface` docstrings cover API, domain constraints, dispatch modes, and backend support); matches the #1730 precedent of no doc updates for example additions.

## 3. PR 提交规范

- [x] Commit message 使用英文，并包含类型标签：`[Bugfix]`、`[Feature]`、`[Refactor]`、`[CI]`、`[Skills]`、`[Testing]` 或 `[Example]`。
  → Single commit `[Example] Add indexer_bwd operator and test` (`93cb1e47`), English, correct type tag, exactly one commit ahead of `upstream/ascendc_pto` (`7a245e26`).

- [x] 已检查 `git diff`/`git status`，PR 仅包含必要修改；CI 和 reviewer 意见已处理。
  → Branch is 1 commit ahead of the latest `upstream/ascendc_pto`; working tree clean; ruff `check` + `format --check` pass with the repo config; no review comments exist yet at submission time.

## 其他补充（可灵活增加）

- [x] 自定义检查：backend-path verification — captured the actual compile commands for all three configurations (default → `-xasc`, explicit ascendc → `-xasc`, explicit pto → `-xcce`), so no backend claim rests on inference.

### 方案/风险/测试补充

- Verification environment: Ascend 910B3 (platform A2), CANN 9.2.0, torch_npu; local verification ran against the installed tilelang 0.1.4 (cann920 build).
- Residual risk: local verification used the installed tilelang wheel; CI builds from the PR branch source. The kernel uses the same primitive generation as recently merged example PRs — risk assessed low.
- PTO-specific operator-level workarounds for pto-isa limitations (documented in the PR body): uint8 row-packed predicates, whole-pair GM→L1 loads / host-staged edge panels (per-row GM→L1 copies mis-execute under PTO), and the `vid == 0` single-execution guard on the generic kernel's atomic scatter.
- Repro: `cd examples_experiment/dsa_sparse_finetune && python test_example_indexer_bwd.py` (precision, default case); `python example_indexer_bwd.py` (smoke).

## 提交前确认

- [ ] 所有适用复选项均已完成；未完成、跳过或豁免项均已说明原因和跟踪项。
  → Remaining open item: simulator screenshots (see §1 item 8 note); all other unchecked boxes above are N/A with stated reasons.
