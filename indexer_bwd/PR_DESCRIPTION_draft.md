# PR Description — indexer_bwd (final, commit 93cb1e47)

> Copy everything below the line into the GitHub PR Description box.

---

Add `indexer_bwd` (DSA sparse-finetune lightning indexer backward) kernel for Ascend NPU.

Computes the gradients (dIndexQ, dWeights, dIndexK) of the lightning-indexer KL finetune loss with respect to IndexQ / Weights / IndexK under varlen (cu_seqlens) packing. Verified against a PyTorch autograd golden, with identical numerics on both the AscendC and PTO backends.

- **Kernel architecture**: three `@tilelang.jit` kernels in Developer mode (4-True `pass_configs`: `TL_ASCEND_AUTO_CV_COMBINE` + `TL_ASCEND_AUTO_CV_SYNC` + `TL_ASCEND_AUTO_SYNC` + `TL_ASCEND_MEMORY_PLANNING`), no `out_idx` (outputs are explicit tensor params): a persistent fast kernel `_tl_indexer_bwd_fast_impl(heads, dim, topk, sm_scale, T_BLOCK=32, TS=4, grid_size=20, layout)` with two compile-time index-layout contracts (layout=0 arange / layout=1 shared rows), a per-token generic kernel `tl_indexer_bwd_generic(heads, dim, topk, sm_scale, block_I=64)` as the full-semantic fallback (incl. `-1` indices), and a trailing `dk` cast kernel `_tl_dk_cast_impl(dim, grid_size=20, cast_rows=64)` (fp32 workspace → bf16). `indexer_bwd_interface` dispatches via exact host-side layout detection (`_detect_indices_layout`) plus the `_fast_gate_ok` domain gate; forced dispatch modes (`fast_arange` / `fast_shared` / `generic`) validate their index contract and raise `ValueError` on violation. Host launches go through a `_CachedLauncher` that marshals the ctypes argument vector once per distinct argument tuple and replays the compiled library call directly.
- **Compute structure**: clip-exact diff chain — `pt = 1{index_score > e⁻¹⁰⁰}` (exact proxy of the golden's `log_softmax > −100` clip; `e⁻¹⁰⁰ = 3.7200759760208356e-44`), `A′ = Σⱼ attn·pt`, `diff = index_score·A′ − attn·pt`; then `dlog = diff · (w·s) · (causal ∧ dReLU)`, `dIndexQ = dlog @ K` and `dIndexK += dlogᵀ @ Q` (two `gemm_v0` per sub-tile pair), and `dWeights = Σ_topk diff · L` with `L = fp32(bf16(bf16(relu(qk)) · s))`.
- **Dataflow and memory**: Cube GEMM operands live in L1 (the `K_l1` panel plus double-buffered `Q_l1[2, PH, dim]` pair slots), fp32 accumulators in L0C (`logits_l0c`, `dq_l0c`, `dk_l0c`), and the score/diff vector chains in UB. Cross-core transits go through GM workspaces (`workspace_logits`, alternating `workspace_dlog_a/b` pair slots, `workspace_dq`, `workspace_k` for layout=1) because direct bf16 UB→L1 transit is unsupported; `dIndexK` accumulates in fp32 in L0C across a core's contiguous same-segment blocks and atomically flushes to the GM fp32 `ws_dk` once per segment run (arange path, whole-tile), after which the cast kernel produces the final bf16 output. Compare / select / bitwise_and masks are uint8 row-packed buffers (`[R, C//8]`) per the official predicate contract. Tensor-edge tail pairs read a host-staged, zero-padded Q panel (`ws_q_blk`).
- **Optimization strategy**: persistent grid sized to the runtime-queried cube-core count (20 on 910B3) with contiguous block assignment, so the K panel reloads only at segment boundaries; pair-GEMM software pipeline — pair p's logits are produced one iteration ahead of consuming pair p−1's dlog (24 → 12 `gemm_v0` calls per block) through alternating dlog workspace slots; `TS` sub-tile bundling (GEMM M = TS·heads) to satisfy the Cube fractal minimum; MIX_AIC_1_2 `(cid, vid)` form where each AIV owns sub-tiles by parity so every cross-core flag handshake stays 1:1; ReLU folded into L0C→GM copies (`enable_relu`); whole-pair GM→L1 loads for tail pairs whose window stays inside the tensor (rows past the segment end contribute zero via `diff == 0`).
- **Numerical precision**: bf16 GEMMs with fp32 accumulation; the dWeights integrand is computed at the golden's bf16 rounding grain (`L = fp32(bf16(bf16(relu(qk)) · s))`, a deliberate double-cast hop) so the kernel tracks autograd numerics; invalid tokens are zero-filled so their dq/dw/dk contributions vanish, and diff is masked through the causal mask before multiplies (NaN·0 hazard).
- **Backend compatibility**: dual-backend by construction from one source. The default target compiles through the AscendC pipeline (captured: `LibraryGenerator("auto", "A2")` → bisheng `-xasc --npu-arch=dav-2201`), and explicit `target="pto"` is fully verified. The PTO-safe forms are contract-clean for both backends: masks follow the uint8 `[R, C//8]` row-packed predicate contract (pto-isa `TAnd` / `vcmp_dispatch` accept only 1/2-byte predicate types), the produce path avoids per-row GM→L1 copies (whole-pair loads plus the host-staged `ws_q_blk` edge panel), and the generic kernel guards its non-idempotent atomic scatter with `vid == 0` (PTO runs the vector chain on both AIVs of the block). Both backends produce identical matched_ratio / max_abs on every verification case.
- **Hardware constraints**: host-side validation mirrors the kernel's compile-time asserts — topk power-of-two (fast path restricted to {16, 32, 64}), heads % 16 == 0 and ≤ 64 (Cube fractal), dim % 16 == 0, every varlen segment ≥ topk, dispatch mode validity; the fast path additionally gates on an L0C 128 KB + UB 192 KB budget mirror (`_fast_gate_ok`) and falls back to the generic kernel on gate miss.
- **Tests**: `test_example_indexer_bwd.py` (116 lines) runs the default representative case (varlen 1023+1025, arange indices → fast/layout=0) against the PyTorch autograd golden and enforces the official bf16 dual-gate (`check_precision`: raises `AssertionError` on violation, prints `Test Passed!` only on pass — the CI runner's pass marker); the kernel module carries a `__main__` shape/finiteness smoke. A PR-time verification matrix (below) additionally covers fast/layout=1, the generic path, varlen tails, short segments, and negative/boundary inputs — on both backends.

## Precision ✅

**Test shape**: B=2 varlen (1023 + 1025), S=2048, H=16, D=128, topk=64, bf16 (committed default case); matrix variants cover H=32/D=64/topk=32, D=64/topk=16, shared-row and per-row-random index layouts, and segment lengths from exactly-topk to non-T_BLOCK-aligned tails. Verified on Ascend 910B3 (A2), CANN 9.2.0, on both the AscendC and PTO backends with identical results.

| Level / Test | Cases | Checks | Result | Precision Metric | max_abs |
|---|---|---|---|---|---|
| L0 threshold (default, fast/layout=0) | 1 | 3 | ✅ PASS | matched_ratio 0.997650 ~ 0.999995 | 7.812e-03 ~ 3.125e-02 |
| L1 functional (layout=0 variants / layout=1 shared / generic) | 6 | 18 | ✅ PASS | matched_ratio 0.995422 ~ 1.000000 | 3.906e-03 ~ 6.250e-02 |
| L2 negative (domain rejection) | 4 | 4 | ✅ PASS — all rejected with `ValueError` before compute | — | — |
| Boundary (non-blocking) | 3 | 9 | 7 PASS + 2 WARN | — | — |
| **Total (blocking)** | **11** | **25** | ✅ **25/25 PASS** | — | — |

- **Thresholds**: bf16 — atol = 2⁻¹⁰ (9.77e-4), rtol = 2⁻⁶ (1.56e-2), max_abs_limit = 1e0, required_ratio = 0.99 (repo precision standard).
- **Pass criteria**: dual-gate AND — per-element `|actual − golden| ≤ atol + rtol·|golden|`; a case passes iff `matched_ratio ≥ 0.99` **AND** `max_abs_error ≤ max_abs_limit`; INF/NAN positions compared structurally.
- L1 cases: varlen tails not aligned to T_BLOCK (segments 1000/1048), H=32/D=64/topk=32, shared-row layout (fast/layout=1), per-row random indices (generic), segments exactly topk long, D=64/topk=16.
- L2 rejections: non-power-of-two topk (48), heads not %16 (24), segment shorter than topk, invalid dispatch mode — each raises `ValueError`.
- Boundary WARN detail: with sparse NaN injected into IndexQ, dWeights still matches the golden structurally, while dIndexQ/dIndexK differ structurally — the kernel's NaN guard masks NaN-token contributions (finite outputs) whereas the autograd golden propagates NaN. Documented design divergence, non-blocking. Zeroed and uniform teacher rows PASS fully.
- Backend evidence: explicit `target="ascendc"` → bisheng `-xasc`; explicit `target="pto"` → bisheng `-xcce` (both captured from the actual compile invocations); the default target selects the AscendC pipeline. Both backends yield identical numbers on every case above.
- Repro: `cd examples_experiment/dsa_sparse_finetune && python test_example_indexer_bwd.py` (default case, prints abs err / err ratio per output; exits 0 with `Test Passed!` only when the dual-gate holds). The L1/L2/Boundary matrix was run at PR time with an equivalent harness against the same committed kernel and golden, using the official dual-gate criteria.

## Performance

**Test shape**: B=2 varlen (1023 + 1025), S=2048, H=16, D=128, topk=64, bf16, fast/layout=0 (default dispatch).

| Metric | Value |
|---|---|
| Task Duration — main fast kernel (torch.npu.Event, median of 10) | AscendC: **131.8 us** / PTO: **139.0 us** (PTO/AscendC = 1.054×) |
| Latency (min / max) | AscendC: 112.5 / 182.7 us · PTO: 121.2 / 226.4 us |
| Block Dim | 20 (persistent grid = runtime-queried cube-core count, Ascend 910B3) |
| Mix Block Dim | MIX_AIC_1_2 `(cid, vid)` — 1 Cube + 2 Vector cores per block |
| Op Type | mix (Cube + Vector) |
| Freq | 1800 MHz (AICore, msprof capture) |

> **Timing method**: per-kernel `torch.npu.Event` device timing (median of 10 after 3 warmups). `msprof` Task Duration could not be collected in the dev environment (CSV export defect in the local CANN 9.2.0 toolchain), so the numbers above are event-based and include launch overhead — treat them as upper bounds; the same-method AscendC↔PTO ratio (1.054×) is the reliable signal. E2E `indexer_bwd_interface` calls measure 1644.6 us (median) on this shape and are host-side dominated (block-index rebuild, workspace allocation); the tensor-edge staging added for PTO-safe tail pairs costs ~272 us versus the pre-PTO revision (1372.8 us) and only triggers when the last segment is not T_BLOCK-aligned (zero cost for aligned inputs); the device-side compute path is unchanged.

### ⚠️ Performance gap disclosure

| Platform | Latency | Gap |
|---|---|---|
| Ascend 910B3, AscendC pipeline (this PR) | 131.8 us (event, median) | — |
| Ascend 910B3, PTO pipeline (this PR) | 139.0 us (event, median) | 1.054× vs AscendC, same source |
| AscendC reference | N/A — no public AscendC implementation of the lightning-indexer backward exists | — |
| GPU baseline | N/A — no baseline measured for this operator | — |

No external baseline exists for this operator (the in-repo forward counterpart, `examples/lightning_indexer`, is also a TileLang implementation), so no cross-platform gap can be quantified. Known retained performance items, checked against the repo antipattern list and kept with reasons:

1. Data-dependent per-row K gathers on the shared (layout=1) and generic paths — inherent to arbitrary indices; the arange fast path avoids them via contiguous panel loads.
2. Per-row fp32 atomics for dIndexK on layout=1/generic; the arange path flushes whole L0C tiles once per segment run instead.
3. Deliberate bf16 grain-match double casts in the dWeights chain — numerical-exactness tradeoff against the autograd golden (see Numerical precision).
4. Per-row copy loops only on varlen tail blocks (guarded, tail-only).
