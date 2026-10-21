# PR Description — sparse_mla_topk_reducesum（官方模板兼容版）

> 官方 `.github/pull_request_template.md` 检查单 + 技术内容（英文，置于"其他补充"）。
> 基于分支 `feat/sparse-mla-topk-reducesum-pr` @ `533b2f7c` 逐项核实；未满足项如实未勾选并说明。

---

## 变更摘要

Add the `sparse_mla_topk_reducesum` kernel (DSA sparse attention top-k probability-mass head reduction) for Ascend NPU under `examples_experiment/dsa/`, migrated from the tile-ai/tilelang GPU implementation (`examples/dsa_sparse_finetune/sparse_mla_topk_reducesum.py`). The test is kept 1:1 aligned with the upstream `test_kernel` (Ascend device adaptations only, no semantic change). No related issue.

## 变更类型

- [x] `[Feature]` 新功能/新算子

## 1. 算子合入标准

- [x] 新增算子在默认场景下通过：PR 测试经默认路径（未显式指定 `target="ascendc"`）在真实 910B3 上完成编译、运行和精度验证（PASS，err_ratio 1.65e-03 / max_abs 5.73e-04）。
- [ ] 适用时 AscendC 与 PTO 双后端均已通过：**不适用**——本算子为 Expert 模式 Ascend kernel（手动 C/V 双流 + cross-flag 同步），设计上单后端；无静默跳过项（dtype/shape 支持域以编译期 assert 显式声明）。
- [x] 未影响现有算子用例、公共接口和已有后端行为：仅新增 2 个文件（kernel + test），框架与公共代码零改动（`git diff upstream/ascendc_pto...HEAD` 仅此 2 文件）。
- [x] `examples/` 下的算子有性能要求和代表性 shape；未达标实现已迁移到 `examples_experiment/`：本 PR 按此规则置于 `examples_experiment/dsa/`（性能尚未达标，见下方披露）。
- [ ] 有 GPU 基线时候选耗时不超过其 2 倍：**未达标，如实披露**——358.3 µs vs H20 GPU 参考（项目侧估计值，非本次实测）~110 µs ≈ **3.3×**（> 2×）。无 AscendC 参考实现。根因与已验证的优化边界见"其他补充"Performance gap disclosure（GM workspace 往返 ~995 MB/call 顶到实测共享 L2 聚合带宽 ~2.96 TB/s 的 ~333 µs 地板；kernel 已贴地板运行）。
- [x] 无 AscendC 参考实现时，已检查性能反模式清单：已按 profiling 驱动逐项核查（workspace 往返地板量化、同步仲裁实测、布局审计、gather 硬件支持面核查），保留项及原因记录于 gap disclosure。
- [x] 优先使用 Developer 原语；使用 Expert 时已说明原因和验证结果：**Expert 模式理由**——AIV 索引 gather → AIC GEMM 的跨核数据流需要手动内存层级（L1/L0C/UB）与跨核 flag 同步（mode-2 quorum），Developer 自动化模式无法表达该 C/V 握手；已在真实硬件验证，含 3-run 位级竞态压测（开发工作区）。
- [x] PR 已附精度/性能测试报告：见"其他补充"（真实硬件测量；simulator 截图不适用——提供真机实测结果替代）。
- [x] 新增单算子的全量测试用例耗时不超过 2 分钟：单用例，单 JIT 配置（编译 ~13 s + 运行亚秒级）。

性能摘要：官方配置（B=1, S=2048, H=16, D=512+64, topk=128, bf16）纯 kernel 延迟 **358.3 µs**（graph-replay，block_I=128，3 实例 × 2 sweep，min/max 344.3/373.7 µs）；默认 block_I=32 约 476 µs。

## 2. 框架及接口代码合入标准

- 不适用：本 PR 无框架/接口改动（仅新增 example 文件）。
- [ ] 新增/迁移测试已登记 `ci/operator_test_manifest.yaml`：**未完成，提交前建议补齐**——examples_experiment 的 kernel↔test 配对按仓内约定需登记（现有 27 条同款条目）；本 PR 的 `examples_experiment/dsa/sparse_mla_topk_reducesum.py: test_sparse_mla_topk_reducesum.py` 一行尚未加入（需 amend，本阶段未执行）。

## 3. PR 提交规范

- [ ] Commit message 使用英文并包含类型标签：**部分满足，提交前建议 amend**——当前 message 为 `Implement sparse_mla_topk_reducesum operator`（英文 ✓，缺 `[Feature]` 类型标签）。
- [x] 已检查 `git diff` / `git status`：仅包含 2 个必要文件。

## 其他补充

（以下为完整英文技术描述，直接可用的 PR 正文）

---

Add sparse_mla_topk_reducesum (DSA sparse attention top-k probability-mass head reduction) kernel for Ascend NPU.

- **Kernel architecture**: Single `@tilelang.jit(out_idx=[6], workspace_idx=[7, 8])` kernel in Expert mode — explicit `T.alloc_L1` / `T.alloc_L0C` / `T.alloc_ub` hierarchy, `T.Scope("C")` (AIC) + `T.Scope("V")` (2× AIV per block) dual streams, manual `T.set_cross_flag` / `T.wait_cross_flag` (mode-2 quorum), no pass_configs. Persistent 1-D grid (`grid_size` defaults to the device `cube_core_num`, 20 on 910B3) with striped task mapping.

- **Compute structure**: For each packed token row `s`, group `g`, and selected position `idx = Indices[s, g, j]`:
  `ReduceSum[s, g, r, j] = Σ_h exp((Q[s, h] · KV[bos + idx]) · sm_scale − Lse[s, h]) · mask`,
  with `mask = 1{idx ≤ s_i} · 1{idx ≠ −1}` built in. REPLICATE_H merge and renormalization stay with the caller, mirroring the upstream interface split. All shape extents (`seq_len` / `seq_len_kv` / `batch + 1`) are `T.symbolic`, so one compile serves arbitrary batch and varlen segmentation at runtime (cu_seqlens `Offsets` + `TokenIndices`).

- **Dataflow and memory**: AIV gathers the indexed KV rows into UB → `ws_kv[grid, slot, BI, D + D_tail]` (GM) → AIC reads them back into L1 ping-pong blocks → one fused `gemm_v0` over `K = D + D_tail` → L0C (fp32) → FIX → `ws_acc` (GM) → AIV applies the causal mask, `exp`, and the head reduction → `ReduceSum`. The GM workspace hop is architecturally forced: the chip provides no UB↔L1 direct copy, so cross-core data must transit GM/L2.

- **Optimization strategy**: (1) merged `D + D_tail` workspace layout enables the single fused GEMM — replacing the two-GEMM main+tail split of the reference and halving the staging DMA count; (2) DEPTH = 4 workspace slots overlap the V-side gather with the C-side GEMM; (3) LEAD-window lookahead stages indices and KV rows ahead of the compute stream (IDXSLOTS ring, side-slot reuse guarded by natural-drain margins); (4) the tail-body `rs_ub` guard pair is unconditionally present at the iteration head, keeping short streams (empty sequence / ΣS = 1) race-free.

- **Numerical precision**: Fixed dtype combination — Q/KV bfloat16, Lse/ReduceSum float32, indices int32. GEMM accumulates fp32 in L0C; mask / exp / head-reduce run in fp32 on the vector pipe; default `sm_scale = (dim + tail_dim)^-0.5`.

- **Hardware constraints** (compile-time asserts): `dim` / `tail_dim` power-of-2; `topk % block_I == 0`; `block_I` even and ≥ 32 (AIV half-split), values > 128 are rejected by the gemm_v0 transpose-B L0B slot budget; `head_kv ∈ {16..64 power-of-2} ∪ {multiples of 64}`; `padded_H ≠ head_kv ⇒ kv_group == 1`; LEAD/SIDES lookahead invariants (scalar-window coverage, prologue side-collision). Invalid indices (−1 / out-of-range) are address-clamped for memory safety and zeroed by the causal mask, matching the reference semantics for arbitrary legal `topk_indices`.

- **Addressing**: The row identity `s_row = bos + s_i` (packed varlen layout) keeps all Q / Lse / Indices / ReduceSum row addressing as pure pid arithmetic; GM scalar reads are limited to `Offsets` / `TokenIndices` where unavoidable.

- **Single generic path**: No index-sharing specialization and no dispatch modes — correct for arbitrary legal `topk_indices` (identical, partially overlapping, independent random, non-contiguous, boundary values).

- **Host entry**: `sparse_mla_topk_reducesum(q, kv, topk_indices, lse, offsets, dim_v, kv_group, sm_scale, block_I, grid_size)`; `block_I = 128` is typically the fastest configuration when `topk % 128 == 0` (default stays 32 for the general domain); per-config build cache.

### Precision ✅

Test shape: B=1, S=2048, H=16, D=512, tail_D=64, topk=128, bf16, seed=42, `offsets = [0, 1023, 2048]` (varlen 1023 + 1025), arange-repeated `topk_indices` — verbatim the upstream tile-ai/tilelang `test_kernel`, checked against a PyTorch fp32 reference (`ref_mla_topk_softmax`).

| Level / Test | Cases | Checks | Result | Precision Metric | max_abs |
|---|---|---|---|---|---|
| Upstream-aligned golden (`test_kernel`) | 1 | 1 | ✅ PASS | err_ratio 1.65e-03 | 5.73e-04 |
| **Total** | **1** | **1** | **✅ 1/1 PASS** | — | — |

Thresholds: abs_err ≤ 1e-3, err_ratio ≤ 1e-2 (error metrics inlined from the upstream `examples/utils.py`)

Pass criteria: dual-gate AND — `abs_err ≤ 1e-3` **AND** `err_ratio ≤ 1e-2`

Ascend-only adaptations (documented in-file, no semantic change): `.npu()` device placement · einops-free `repeat` · einsum axis renames (`"ahd,bd->ahb"`) · test-side REPLICATE_H merge + renormalization (upstream interface contains it; this kernel's boundary stops at ReduceSum).

Beyond the PR scope, the kernel was additionally validated in the development workspace with extended suites — a 70-case layered regression covering random/independent, duplicate, disordered, −1 and out-of-range index distributions, varlen shapes including empty segments, `kv_group = 2`, `REPLICATE_H = 2`, and 3-run bitwise race stress — all PASS.

### Performance

Test shape: B=1, S=2048, H=16, D=512 + 64, topk=128, bf16 (official config); `block_I = 128` (tuned) primary, `block_I = 32` (default call) reference.

| Metric | Value |
|---|---|
| Pure-kernel latency, block_I=128 (graph-replay, mean of 3 instances × 2 sweeps) | **358.3 µs** |
| Latency (min / max), per-instance | 344.3 / 373.7 µs |
| Pure-kernel latency, block_I=32 (default) | ~476 µs (band 466–477) |
| Grid | 20 (persistent) |
| Op Type | mix (Cube + Vector) |

Timing note: primary numbers are graph-replay event timings (N-call graph capture, warmed, batch replay) with no Python launch overhead; msprof Task Duration historically reads ~1.03–1.09× the graph number on this kernel family; a plain Python end-to-end call adds JIT wrapper allocation and launch overhead (~2 ms) and is not a performance conclusion.

### ⚠️ Performance gap disclosure

| Platform | Latency | Gap |
|---|---|---|
| Ascend 910B3 (this PR, Expert mode, block_I=128) | 358.3 µs | — |
| H20 GPU reference (project-side estimate, not measured in this work) | ~110 µs | ~3.3× slower |
| Project performance target | 220 µs | 1.63× over |

Root cause (measured): the AIV/AIC split — indexed gather on the vector cores, GEMM on the cube cores, no UB↔L1 direct copy — forces a GM workspace transit. At this shape the kernel moves ~995 MB per call (`ws_kv` 302 MB written + 302 MB read back, `ws_acc` ≈ 34 MB, Q 37.7 MB, misc). Against the measured shared-L2 aggregate service bandwidth (~2.96 TB/s) this sets a ~333 µs bandwidth floor, and the kernel runs at that floor. On the GPU, the same indexed gather lives inside the load path (SMEM direct feed, no intermediate transit). A data-flow restructuring that removes the transit (deduplicated on-chip panel residency, 8.4× traffic reduction) was prototyped and measured generic-correct but time-regressive on this API surface, and is intentionally excluded from this PR (generic-only, no data-content specialization).

## 提交前确认

- 适用复选项已完成；未满足/待补齐项共 2 处，均已说明：① `ci/operator_test_manifest.yaml` 登记待补（一行，需 amend）；② Commit message 类型标签待补（`[Feature]`，需 amend）。性能 2× 标准未达标已如实披露并置于 examples_experiment/。
