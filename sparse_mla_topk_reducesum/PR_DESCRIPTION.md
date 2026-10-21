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

## Precision ✅

Test shape: B=1, S=2048, H=16, D=512, tail_D=64, topk=128, bf16, seed=42, `offsets = [0, 1023, 2048]` (varlen 1023 + 1025), arange-repeated `topk_indices` — verbatim the upstream tile-ai/tilelang `test_kernel`, checked against a PyTorch fp32 reference (`ref_mla_topk_softmax`).

| Level / Test | Cases | Checks | Result | Precision Metric | max_abs |
|---|---|---|---|---|---|
| Upstream-aligned golden (`test_kernel`) | 1 | 1 | ✅ PASS | err_ratio 1.65e-03 | 5.73e-04 |
| **Total** | **1** | **1** | **✅ 1/1 PASS** | — | — |

Thresholds: abs_err ≤ 1e-3, err_ratio ≤ 1e-2 (error metrics `get_abs_err` / `get_err_ratio` inlined from the upstream `examples/utils.py`)

Pass criteria: dual-gate AND — `abs_err ≤ 1e-3` **AND** `err_ratio ≤ 1e-2`

Ascend-only adaptations (documented in-file, no semantic change): `.npu()` device placement · einops-free `repeat` via `unsqueeze(0).repeat(S, 1)` · einsum axis renames (`"ahd,bd->ahb"`) · the REPLICATE_H merge + renormalization applied on the test side (the upstream interface contains it; this kernel's boundary stops at ReduceSum).

Note: beyond the PR scope, the kernel was additionally validated in the development workspace with extended suites — a 70-case layered regression covering random/independent, duplicate, disordered, −1 and out-of-range index distributions, varlen shapes including empty segments, `kv_group = 2`, `REPLICATE_H = 2`, and 3-run bitwise race stress — all PASS.

## Performance

Test shape: B=1, S=2048, H=16, D=512 + 64, topk=128, bf16 (official config); `block_I = 128` (tuned) primary, `block_I = 32` (default call) reference.

| Metric | Value |
|---|---|
| Pure-kernel latency, block_I=128 (graph-replay, mean of 3 instances × 2 sweeps) | **358.3 µs** |
| Latency (min / max), per-instance | 344.3 / 373.7 µs |
| Pure-kernel latency, block_I=32 (default) | ~476 µs (band 466–477) |
| Grid | 20 (persistent) |
| Op Type | mix (Cube + Vector) |

Timing note: the primary numbers are graph-replay event timings (N-call graph capture, warmed, batch replay) and contain no Python launch overhead. On this kernel family, msprof Task Duration historically reads ~1.03–1.09× the graph-replay number. A plain Python end-to-end call additionally includes JIT wrapper allocation and launch overhead (~2 ms) and is not a performance conclusion.

## ⚠️ Performance gap disclosure

| Platform | Latency | Gap |
|---|---|---|
| Ascend 910B3 (this PR, Expert mode, block_I=128) | 358.3 µs | — |
| H20 GPU reference (project-side estimate, not measured in this work) | ~110 µs | ~3.3× slower |
| Project performance target | 220 µs | 1.63× over |

Root cause (measured): the AIV/AIC split — indexed gather on the vector cores, GEMM on the cube cores, no UB↔L1 direct copy — forces a GM workspace transit. At this shape the kernel moves ~995 MB per call (`ws_kv` 302 MB written + 302 MB read back, `ws_acc` ≈ 34 MB, Q 37.7 MB, misc). Against the measured shared-L2 aggregate service bandwidth (~2.96 TB/s) this sets a ~333 µs bandwidth floor, and the kernel runs at that floor. On the GPU, the same indexed gather lives inside the load path (SMEM direct feed, no intermediate transit). A data-flow restructuring that removes the transit (deduplicated on-chip panel residency, 8.4× traffic reduction) was prototyped and measured generic-correct but time-regressive on this API surface, and is intentionally excluded from this PR (generic-only, no data-content specialization).
