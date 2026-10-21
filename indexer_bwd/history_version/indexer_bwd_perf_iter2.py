"""indexer_bwd — DSA sparse finetune lightning indexer backward (Ascend NPU, Developer mode).

Computes the gradients of the varlen KL-divergence loss w.r.t. the lightning
indexer inputs (semantics replicate examples_experiment/dsa_sparse_finetune/
ref_gpu/indexer_bwd.py):

    logits[t, h, p]  = relu(q[t, h, :] . k[p, :] * sm_scale)
    d_logits[t, h, p] = (IndexScore[t, p] - AttnScore[t, p])
                        * 1{logits > 0} * Weights[t, h]
    dWeights[t, h]   += sum_p (IndexScore - AttnScore)[t, p] * logits[t, h, p]
    dIndexQ[t, h, :] += sum_p d_logits[t, h, p] * IndexK[p, :] * sm_scale
    dIndexK[p, :]    += sum_h d_logits[t, h, p] * IndexQ[t, h, :] * sm_scale

rev2 dual-path architecture (DESIGN.md rev2 §1.4), host dispatch with exact
(non-heuristic) index-layout detection:

    fast  layout=0  arange-tiled indices (main-repo default / perf config):
                    persistent T.Kernel(grid), T_BLOCK=32 tokens per block,
                    TS=4 sub-tile GEMM bundling (M = TS*H rows), whole-panel
                    K load, vectorized causal mask, whole-tile fp32 region
                    atomic into ws_dk.
    fast  layout=1  shared non-arange indices: per-row guarded gather via a
                    GM transit, index-value compare mask, per-row atomic.
    generic         rev1 per-token structure (grid = seq_len), full semantic
                    coverage (arbitrary per-token indices incl. -1).
    cast kernel     ws_dk fp32 -> dIndexK bf16 (single rounding), shared by
                    all paths, launched after the main kernel on the same
                    stream.

dk full-fp32 accumulation chain (rev2 core precision fix, DESIGN §12.1):
L0C fp32 on-chip accumulate -> T.tile.atomic_add fp32 (L0C direct src) into
the ws_dk fp32 GM workspace -> trailing cast kernel performs the single
fp32->bf16 rounding. No bf16 rounding ever sits on the accumulation path.

Implementation notes (deviations from DESIGN.md pseudocode, each forced by a
platform constraint, a strictly-simpler validated form, or the DESIGN's own
acceptance criteria; see debug_log.md rev2 attempt records for evidence):

  1. Q/dq block staging uses per-token [H, D] copies (4 per sub-tile) instead
     of one [TS, H*D] -> [TS*H, D] contiguous-row copy: T.copy requires
     structurally-equal src/dst shapes (tilelang/language/copy_op.py
     `ir.assert_structural_equal(src.shape, dst.shape)`), so a layout
     reinterpretation copy is not expressible. Bytes moved are identical; the
     per-token [H, D] GM<->L1/L0C copies are the rev1-validated API forms.
  2. relu rides the L0C->GM transit leg (`enable_relu=True` on the write into
     workspace_logits, rev1-validated form) instead of the GM->UB read leg —
     mathematically identical placement.
  3. dk atomic: layout=0 keeps the L0C whole-tile region atomic
     (`ws_dk[bos:bos+topk, :] <- dk_l0c`, gqa_sink:437 precedent, validated
     0.9977 in rev2 L0). layout=1/generic per-row atomics route through an
     fp32 UB buffer instead (whole-fragment L0C -> GM ws -> UB, then
     `T.tile.atomic_add(ws_dk[bos+pos, 0], dk_ub[i, :])`): L0C *partial-region*
     accesses (row-sliced copy src / row-sliced atomic src) are unreliable —
     only the first 16x16 fractal tile lands and the access races the
     producing GEMM (rev2 L0 evidence: dq u!=0 tokens fully bad, dk hot rows
     bad beyond tile 0, run-twice diffs up to 119). The rev1 UB form is the
     structure-validated path.
  4. dq writeback routes through a bf16 GM workspace: whole-fragment
     `T.copy(dq_l0c, workspace_dq[...])` (fp32->bf16 auto-cast, the
     rev1/generic-validated whole-fragment L0C->GM form) -> GM->UB -> per-token
     UB row-block writes. The direct per-token L0C row-slice copy is broken
     (see note 3).
  5. causal validity folds into the token position: t_eff = (t < len) ? t : -1
     (fill+compare), so invalid tokens mask every column without a separate
     valid buffer + bitwise_and.
  6. Tail-block Q loads clamp the row index to len-1 (always a valid row) and
     the score/weights UB buffers are zero-filled at kernel start, so garbage
     from never-written L1/UB rows can never propagate through the GEMMs
     (0 * NaN hazards); invalid-token contributions are masked to exactly 0.
  7. GRADIENT EXACTNESS (clip-exact A' form): the DESIGN §1.3 closed form
     `d_logits = (IndexScore - AttnScore) * ...` equals the §9.1 autograd
     golden ONLY when (a) every valid slot passes the golden's
     `log_topk_prob.clip(-100, 0)` (i.e. IndexScore > exp(-100)) and (b) the
     clip-passing attention mass A' = sum_{k: pt_k} AttnScore[k] is 1 — both
     hold for small-scale arange-tiled inputs (the main-repo test config; the
     main-repo GPU kernel itself uses this closed form and is never tested
     beyond that regime). The exact golden gradient is
     `d_v[j] = IndexScore[j] * A' - AttnScore[j] * pt[j]` with
     `pt[j] = (IndexScore[j] > exp(-100))` (the clip kills deep slots'
     gradients entirely, including their -attn term), so all three paths
     compute this clip-exact form: pt from a threshold compare on IndexScore
     (is = exp(log_softmax) is monotone, so pt is recoverable from the
     operator input; the compare operand is the exact exp(-100) fp32
     denormal constant), A' reduced from the pt-masked
     AttnScore (the generic path pre-accumulates A' across its NI topk
     blocks — the log_softmax normalizer spans the full topk), then
     `diff = (is * A' - attn * pt) * sm_scale`. Under the small-scale arange contract pt == valid and
     A' == 1, degenerating exactly to the §1.3 form (validated: clip-exact
     fp32 replica matches the golden on arange/shifted/random at scales
     0.5/1/5/100/65504 and topk=128, while the closed form fails at 0.08-0.38
     on shifted/random and 0.004-0.38 at scale >= 3). Residual ~1-2% mismatch
     exists only in a narrow transition zone (log_softmax straddling -100,
     e.g. scale ~2.5-4 randn inputs) where the golden's own bf16 backward
     roundings dominate; L1 value-range cases sit outside that zone. NaN
     IndexScore rows (all-invalid tokens, golden log_softmax artifact) are
     sanitized by masking diff through the validity select before any
     multiply (NaN * 0 = NaN hazard).
  8. dWeights uses the golden's bf16 logit grain: the §9.1 golden keeps the
     einsum logits in bf16 (bf16(bf16(qk) * sm_scale) after the scale mul),
     while the kernels carry fp32 GEMM logits; a fp32-grain dw drifts ~2 bf16
     ulp vs the golden and fails the 0.99 gate on small-S cases (l0_equal_topk
     0.9883, 12/1024 elements). The dw integrand therefore re-rounds the
     logits through bf16 twice (fp32 -> bf16 -> fp32 -> *sm_scale -> bf16 ->
     fp32), aligning both sides to ~1 ulp; dq/dk are unaffected (they only
     consume the logit SIGN via the relu indicator).

Staging buffers whose names contain "workspace" intentionally: the
TL_ASCEND_AUTO_CV_SYNC pass pairs GM writes/reads by that name substring and
inserts the C/V handshake flags automatically (src/transform/
ascend_combinecv.cc FetchWorkspaceName; rev1-validated mechanism).
"""

import ctypes

import torch

import tilelang
from tilelang import language as T

try:  # fast raw-stream query (0.9 us vs 33 us for torch.npu.current_stream)
    import torch_npu  # noqa: F401

    _npu_raw_stream = torch_npu._C._npu_getCurrentRawStream
except Exception:  # pragma: no cover - non-NPU import context
    _npu_raw_stream = None

__all__ = [
    "tl_indexer_bwd_fast",
    "tl_indexer_bwd_generic",
    "tl_dk_cast",
    "indexer_bwd_interface",
    "indexer_bwd",
    "prepare_token_indices",
    "prepare_block_indices",
    "_detect_indices_layout",
]


class _CachedLauncher:
    """Low-overhead callable around a compiled tilelang kernel.

    The stock cython launch path re-marshals every argument on each call —
    constructing a fresh TVM ``Analyzer`` alone costs ~0.55 ms, which made
    back-to-back kernel loops launch-bound (~0.75 ms/launch vs ~0.22 ms of
    device work). This wrapper marshals the low-level ctypes argument vector
    once per distinct argument tuple and replays ``adapter.lib.call`` directly
    (~13 us/launch, the bare ACL submission cost).

    Safety model: the single cache slot holds strong references to the exact
    argument objects of the last marshaled call, keyed on their ``id()`` tuple.
    While the references are alive those ids cannot be recycled, so a key hit
    guarantees the cached data pointers / dynamic-shape values are still the
    right ones for these very tensors. Any miss (fresh tensors, changed
    arg count, explicit stream) falls back to the full validating path.
    """

    __slots__ = ("_kernel", "_adapter", "_lib", "_params_len", "_key", "_args", "_args_refs", "_dev")

    def __init__(self, kernel):
        self._kernel = kernel
        adapter = kernel.adapter
        self._adapter = adapter
        self._lib = adapter.lib
        # fast path only for plain all-tensor signatures (no auto-allocated
        # result/workspace buffers — our kernels pass everything explicitly)
        ok = not getattr(adapter, "result_idx", []) and not getattr(adapter, "workspace_idx", [])
        ok = ok and not getattr(adapter, "auto_gm_idx", None)
        self._params_len = len(adapter.params) if ok else -1
        self._key = None
        self._args = None
        self._dev = None

    def __getattr__(self, name):  # delegate get_kernel_source()/etc.
        return getattr(self._kernel, name)

    def _marshal(self, args):
        ad = self._adapter
        tensors = list(args)
        c_args = []
        for t in tensors:
            if not isinstance(t, torch.Tensor):
                return None
            if not t.is_contiguous():
                raise ValueError("kernel tensors must be contiguous")
            c_args.append(ctypes.c_void_p(t.data_ptr()))
        for _, (bidx, sidx) in ad.dynamic_symbolic_map.items():
            c_args.append(ctypes.c_int64(int(tensors[bidx].shape[sidx])))
        c_args.append(None)  # stream slot (filled per call)
        return c_args

    def __call__(self, *args, stream=-1):
        plen = self._params_len
        if plen == len(args) and (stream == -1 or isinstance(stream, int)):
            key = tuple(map(id, args))
            if key == self._key:
                dev = self._dev
                if stream == -1:
                    stream = _npu_raw_stream(dev) if _npu_raw_stream is not None else torch.npu.current_stream().npu_stream
                c_args = self._args
                c_args[-1] = ctypes.c_void_p(stream)
                self._lib.call(*c_args)
                return
            c_args = self._marshal(args)
            if c_args is not None:
                dev = args[0].get_device()
                if stream == -1:
                    stream = _npu_raw_stream(dev) if _npu_raw_stream is not None else torch.npu.current_stream().npu_stream
                c_args[-1] = ctypes.c_void_p(stream)
                # commit slot (strong refs via self._args->args closure)
                self._key = key
                self._args = c_args
                self._args_refs = args
                self._dev = dev
                self._lib.call(*c_args)
                return
        return self._kernel(*args, stream=stream) if stream != -1 else self._kernel(*args)

_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# ============================================================================
# Fast kernel (persistent grid, T_BLOCK=32 / TS=4, layout 0/1 compile keys)
# ============================================================================


@tilelang.jit(pass_configs=_pass_configs)
def _tl_indexer_bwd_fast_impl(
    heads,
    dim,
    topk,
    sm_scale=None,
    T_BLOCK=32,
    TS=4,
    grid_size=20,
    layout=0,
):
    """Build the persistent fast-path kernel (Developer mode).

    Args:
        heads: number of heads H (% 16 == 0 and <= 64, Cube fractal).
        dim: head dim D (% 16 == 0).
        topk: selected positions (fast-path domain {16, 32, 64}; 128 -> generic).
        sm_scale: softmax scale, defaults to dim ** -0.5.
        T_BLOCK: tokens per block (shared K gather / dk atomic granularity).
        TS: sub-tile token count (GEMM bundling, M = TS * heads).
        grid_size: persistent grid (compile-time constant; 910B3 = 20).
        layout: 0 = arange-tiled contract, 1 = shared-row contract.
    """
    assert layout in (0, 1), f"layout must be 0 or 1, got {layout}"
    assert topk in (16, 32, 64), f"fast path supports topk in (16, 32, 64), got {topk}"
    assert heads % 16 == 0 and heads <= 64, "heads must be % 16 == 0 and <= 64 (Cube fractal)"
    assert dim % 16 == 0, "dim must be a multiple of 16 (Cube fractal)"
    assert T_BLOCK % TS == 0, "T_BLOCK must be a multiple of TS"
    TS_H = TS * heads
    assert TS_H % 16 == 0, "TS*heads must be a multiple of 16 (GEMM M)"
    l0c_bytes = TS_H * topk * 4 + TS_H * dim * 4 + topk * dim * 4
    assert l0c_bytes <= 128 * 1024, f"L0C budget exceeded: {l0c_bytes} > 131072"
    if sm_scale is None:
        sm_scale = dim**-0.5
    TSB = T_BLOCK // TS

    seq_len = T.symbolic("seq_len")  # total packed length (sum of segment lens)
    num_blocks = T.symbolic("num_blocks")  # BlockIndices rows (persistent task count)

    dtype = "bfloat16"
    accum_dtype = "float32"
    indices_dtype = "int32"

    @T.prim_func
    def main(
        IndexQ: T.Tensor([seq_len, heads, dim], dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], dtype),  # type: ignore
        IndexK: T.Tensor([seq_len, dim], dtype),  # type: ignore
        AttnScore: T.Tensor([seq_len, topk], accum_dtype),  # type: ignore
        IndexScore: T.Tensor([seq_len, topk], accum_dtype),  # type: ignore
        TopkIndices: T.Tensor([seq_len, topk], indices_dtype),  # type: ignore  # layout=1 only
        BlockIndices: T.Tensor([num_blocks, 3], indices_dtype),  # type: ignore
        workspace_logits: T.Tensor([grid_size, 2, TS_H, topk], accum_dtype),  # type: ignore
        workspace_dlog_a: T.Tensor([grid_size, 2, TS_H, topk], dtype),  # type: ignore
        workspace_dlog_b: T.Tensor([grid_size, 2, TS_H, topk], dtype),  # type: ignore
        workspace_dq: T.Tensor([grid_size, 2, TS_H, dim], dtype),  # type: ignore
        workspace_k: T.Tensor([grid_size, topk, dim], dtype),  # type: ignore  # layout=1 only
        workspace_dkf: T.Tensor([grid_size, topk, dim], accum_dtype),  # type: ignore  # layout=1 only
        ws_dk: T.Tensor([seq_len, dim], accum_dtype),  # type: ignore
        dIndexQ: T.Tensor([seq_len, heads, dim], dtype),  # type: ignore
        dWeights: T.Tensor([seq_len, heads], dtype),  # type: ignore
    ):
        with T.Kernel(grid_size, threads=1, is_npu=True) as (cid):
            # ---- L1: Cube GEMM operands (compiler maps alloc_shared -> L1) ----
            K_l1 = T.alloc_shared([topk, dim], dtype)
            Q_l1 = T.alloc_shared([TS_H, dim], dtype)
            dlog_a_l1 = T.alloc_shared([TS_H, topk], dtype)
            dlog_b_l1 = T.alloc_shared([TS_H, topk], dtype)

            # ---- L0C: GEMM accumulators (alloc_fragment) ----
            logits_l0c = T.alloc_fragment([TS_H, topk], accum_dtype)
            dq_l0c = T.alloc_fragment([TS_H, dim], accum_dtype)
            dk_l0c = T.alloc_fragment([topk, dim], accum_dtype)

            # ---- UB: Vector-side buffers (alloc_shared -> UB) ----
            # Note: GM<->UB copies require identical dtypes (common.h
            # copy_gm_to_ub/copy_ub_to_gm templates), so bf16 inputs/outputs
            # stage through same-dtype UB buffers and cast on UB<->UB hops.
            attn_ub = T.alloc_shared([T_BLOCK, topk], accum_dtype)
            is_ub = T.alloc_shared([T_BLOCK, topk], accum_dtype)
            diffm_ub = T.alloc_shared([T_BLOCK, topk], accum_dtype)
            pt_b = T.alloc_shared([T_BLOCK, topk], accum_dtype)
            attnm = T.alloc_shared([T_BLOCK, topk], accum_dtype)
            a_col = T.alloc_shared([T_BLOCK], accum_dtype)
            w_block_bf16_ub = T.alloc_shared([T_BLOCK, heads], dtype)
            w_block_ub = T.alloc_shared([T_BLOCK, heads], accum_dtype)
            dw_acc_ub = T.alloc_shared([T_BLOCK, heads], accum_dtype)
            dw_bf16_ub = T.alloc_shared([T_BLOCK, heads], dtype)
            logits_ub = T.alloc_shared([TS_H, topk], accum_dtype)
            dlog_ub = T.alloc_shared([TS_H, topk], accum_dtype)
            dlog_bf16_ub = T.alloc_shared([TS_H, topk], dtype)
            dq_bf16_ub = T.alloc_shared([TS_H, dim], dtype)
            dk_ub = T.alloc_shared([topk, dim], accum_dtype)  # layout=1 only
            col_1d = T.alloc_shared([topk], accum_dtype)
            logits_t = T.alloc_shared([heads, topk], accum_dtype)
            diff_2d = T.alloc_shared([heads, topk], accum_dtype)
            diff_t = T.alloc_shared([topk], accum_dtype)
            w_2d = T.alloc_shared([heads, topk], accum_dtype)
            col_2d = T.alloc_shared([heads, topk], accum_dtype)
            tval_2d = T.alloc_shared([heads, topk], accum_dtype)
            causal = T.alloc_shared([heads, topk], accum_dtype)
            drelu = T.alloc_shared([heads, topk], accum_dtype)
            logits_m = T.alloc_shared([heads, topk], accum_dtype)
            dwq_2d = T.alloc_shared([heads, topk], accum_dtype)
            dlog_t = T.alloc_shared([heads, topk], accum_dtype)
            zero_2d = T.alloc_shared([heads, topk], accum_dtype)
            logits_bf16_t = T.alloc_shared([heads, topk], dtype)  # dw-grain cast hop
            w_t = T.alloc_shared([heads], accum_dtype)
            dw_t = T.alloc_shared([heads], accum_dtype)
            # layout=1-only buffers (allocated unconditionally: TVM script
            # scoping does not propagate allocations out of compile-time if
            # branches; unused ones are simply never touched)
            indices_i32_ub = T.alloc_shared([topk], indices_dtype)
            indices_f32_ub = T.alloc_shared([topk], accum_dtype)
            gather_ub = T.alloc_shared([topk, dim], dtype)

            # ---- loop-invariant setup (once per kernel) ----
            T.tile.fill(attn_ub, 0.0)
            T.tile.fill(is_ub, 0.0)
            T.tile.fill(w_block_ub, 0.0)
            T.tile.fill(zero_2d, 0.0)
            if layout == 0:
                T.tile.arith_progression(col_1d, 0, 1, topk)

            # ---- persistent task loop (runtime bound, sparse_mla:309 form) ----
            ntask = T.max(0, T.ceildiv(num_blocks - cid, grid_size))
            for k in T.serial(ntask):
                bx = k * grid_size + cid
                bos = BlockIndices[bx, 0]
                t0 = BlockIndices[bx, 1]
                ln = BlockIndices[bx, 2]
                t_max = T.min(t0 + T_BLOCK, ln) - 1

                # ---- K panel load (C side) ----
                if layout == 0:
                    # arange contract: the topk rows are contiguous in-segment
                    T.copy(IndexK[bos : bos + topk, 0:dim], K_l1)
                else:
                    # shared contract: all rows identical; gather with guards,
                    # then UB -> GM -> L1 transit (bf16 UB->L1 is fp16-only)
                    T.copy(TopkIndices[bos, 0:topk], indices_i32_ub)
                    T.copy(indices_i32_ub, indices_f32_ub)
                    T.tile.fill(gather_ub, 0.0)
                    for i in T.serial(topk):
                        pos = indices_i32_ub[i]
                        if (pos > -1) & (pos <= t_max):
                            T.copy(IndexK[bos + pos, 0:dim], gather_ub[i, 0:dim])
                    T.copy(gather_ub, workspace_k[cid, 0:topk, 0:dim])
                    T.copy(workspace_k[cid, 0:topk, 0:dim], K_l1)
                    T.tile.broadcast(col_2d, indices_f32_ub, axis=0)

                # ---- scores / weights block load (V side, tail-guarded) ----
                if t0 + T_BLOCK <= ln:
                    T.copy(AttnScore[bos + t0 : bos + t0 + T_BLOCK, 0:topk], attn_ub)
                    T.copy(IndexScore[bos + t0 : bos + t0 + T_BLOCK, 0:topk], is_ub)
                    T.copy(Weights[bos + t0 : bos + t0 + T_BLOCK, 0:heads], w_block_bf16_ub)
                else:
                    for u in T.serial(T_BLOCK):
                        if t0 + u < ln:
                            T.copy(AttnScore[bos + t0 + u, 0:topk], attn_ub[u, 0:topk])
                            T.copy(IndexScore[bos + t0 + u, 0:topk], is_ub[u, 0:topk])
                            T.copy(Weights[bos + t0 + u, 0:heads], w_block_bf16_ub[u, 0:heads])
                T.copy(w_block_bf16_ub, w_block_ub)  # bf16 -> fp32 auto-cast

                # ---- clip-exact diffm (block level, V side; docstring note 7):
                #      pt = (is > fp32-normal floor) [golden clip passthrough
                #      proxy]; A' = sum_k attn_k * pt_k; diffm = (is * A' -
                #      attn * pt) * sm_scale. Computed as whole-block tile ops
                #      on the MTE-loaded score buffers — per-token row copies
                #      out of MTE-written UB buffers read stale data under
                #      AUTO_CV_COMBINE (rev2 L1 evidence), so the per-token
                #      segment only row-copies this V-written diffm tile
                #      (the rev1/v3-validated form). ----
                T.tile.compare(pt_b, is_ub, T.float32(3.7200759760208356e-44), "GT")  # packed pt
                T.tile.select(attnm, pt_b, attn_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")  # attn * pt
                T.reduce_sum(attnm, a_col, dim=1)  # [T_BLOCK] = A' per token
                T.tile.broadcast(attn_ub, a_col, axis=1)  # A'_2d (attn_ub dead -> reuse)
                T.tile.mul(diffm_ub, is_ub, attn_ub)  # is * A'
                T.tile.sub(diffm_ub, diffm_ub, attnm)  # is*A' - attn*pt (UNSCALED:
                # sm_scale is applied once AFTER the dw integrand, which pairs
                # the unscaled diff with the grain-L that already carries s)

                # ---- sub-tile loop: 3 bundled GEMMs + vector post-processing ----
                for ts in T.serial(TSB):
                    t_cur = t0 + ts * TS

                    # Q load (C side): per-token [H, D] rows, clamped row index
                    # keeps every loaded row valid (NaN-safe, note 5)
                    for u in T.serial(TS):
                        row = bos + T.min(t_cur + u, ln - 1)
                        T.copy(IndexQ[row, 0:heads, 0:dim], Q_l1[(u * heads) : ((u + 1) * heads), 0:dim])

                    # GEMM1: logits = Q_flat x K^T (C side; Q unscaled, scale in diffm)
                    T.gemm_v0(Q_l1, K_l1, logits_l0c, transpose_B=True, init=True)

                    # CV transit: L0C -> GM workspace (relu on this leg) -> UB
                    T.copy(logits_l0c, workspace_logits[cid, ts % 2, 0:TS_H, 0:topk], enable_relu=True)
                    T.copy(workspace_logits[cid, ts % 2, 0:TS_H, 0:topk], logits_ub)

                    # per-token vector segment (V side, [H, topk] tiles)
                    for u in T.serial(TS):
                        t_abs = t_cur + u
                        # causal = (pos <= t) & (t < len): an out-of-range token
                        # becomes t_eff = -1 which masks every column (note 5)
                        t_eff = T.if_then_else(t_abs < ln, t_abs, -1)
                        T.copy(logits_ub[(u * heads) : ((u + 1) * heads), 0:topk], logits_t)
                        T.copy(diffm_ub[(ts * TS + u), 0:topk], diff_t)
                        T.tile.broadcast(diff_2d, diff_t, axis=0)
                        if layout == 0:
                            T.tile.broadcast(col_2d, col_1d, axis=0)
                        T.tile.fill(tval_2d, T.float32(t_eff))
                        T.tile.compare(causal, col_2d, tval_2d, "LE")
                        # NaN/invalid guard (note 7): mask diff through causal
                        # BEFORE any multiply (NaN * 0 = NaN hazard)
                        T.tile.select(diff_2d, causal, diff_2d, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                        T.copy(w_block_ub[(ts * TS + u), 0:heads], w_t)
                        T.tile.broadcast(w_2d, w_t, axis=1)
                        # dWeights integrand: diff * logits at the golden's
                        # bf16 grain (docstring note 8): L = bf16(bf16(relu(qk))
                        # * sm_scale) — the §9.1 golden keeps einsum logits in
                        # bf16, so aligning the grain keeps dw within ~1 ulp;
                        # fp32-grain logits drift ~2 ulp on small-S cases.
                        T.copy(logits_t, logits_bf16_t)  # fp32 -> bf16
                        T.copy(logits_bf16_t, logits_m)  # bf16 -> fp32
                        T.tile.mul(logits_m, logits_m, sm_scale)
                        T.copy(logits_m, logits_bf16_t)  # -> bf16
                        T.copy(logits_bf16_t, logits_m)  # -> fp32 (dw-grain L)
                        T.tile.select(logits_m, causal, logits_m, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                        T.tile.mul(dwq_2d, diff_2d, logits_m)
                        T.reduce_sum(dwq_2d, dw_t, dim=1)
                        T.copy(dw_t, dw_acc_ub[(ts * TS + u), 0:heads])
                        # fold sm_scale into diff for the dlog/dq/dk leg
                        T.tile.mul(diff_2d, diff_2d, sm_scale)
                        # d_logits = diff * w * 1{logits > 0} * causal
                        T.tile.compare(drelu, logits_t, zero_2d, "GT")
                        T.tile.bitwise_and(causal, causal, drelu)
                        T.tile.mul(dlog_t, diff_2d, w_2d)
                        T.tile.select(dlog_t, causal, dlog_t, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                        T.copy(dlog_t, dlog_ub[(u * heads) : ((u + 1) * heads), 0:topk])

                    # CV transit: dlog UB(fp32) -> bf16 UB -> GM workspace -> L1
                    T.copy(dlog_ub, dlog_bf16_ub)
                    T.copy(dlog_bf16_ub, workspace_dlog_a[cid, ts % 2, 0:TS_H, 0:topk])
                    T.copy(workspace_dlog_a[cid, ts % 2, 0:TS_H, 0:topk], dlog_a_l1)
                    T.copy(dlog_bf16_ub, workspace_dlog_b[cid, ts % 2, 0:TS_H, 0:topk])
                    T.copy(workspace_dlog_b[cid, ts % 2, 0:TS_H, 0:topk], dlog_b_l1)

                    # GEMM2: dq = dlog x K (values already carry sm_scale).
                    # Whole-fragment L0C -> GM (auto-cast bf16) -> UB, then
                    # per-token UB row-block writes: per-token L0C row-slice
                    # copies are unreliable beyond the first fractal tile
                    # (docstring note 3/4).
                    T.gemm_v0(dlog_a_l1, K_l1, dq_l0c, init=True)
                    T.copy(dq_l0c, workspace_dq[cid, ts % 2, 0:TS_H, 0:dim])
                    T.copy(workspace_dq[cid, ts % 2, 0:TS_H, 0:dim], dq_bf16_ub)
                    for u in T.serial(TS):
                        if t_cur + u < ln:
                            T.copy(
                                dq_bf16_ub[(u * heads) : ((u + 1) * heads), 0:dim],
                                dIndexQ[bos + t_cur + u, 0:heads, 0:dim],
                            )

                    # GEMM3: dk += dlog^T x Q_flat (fp32 on-chip, across sub-tiles)
                    T.gemm_v0(dlog_b_l1, Q_l1, dk_l0c, transpose_A=True, init=(ts == 0))

                # ---- block tail: dWeights write + dk fp32 atomic (C side) ----
                T.copy(dw_acc_ub, dw_bf16_ub)  # fp32 -> bf16 auto-cast
                if t0 + T_BLOCK <= ln:
                    T.copy(dw_bf16_ub, dWeights[bos + t0 : bos + t0 + T_BLOCK, 0:heads])
                else:
                    for u in T.serial(T_BLOCK):
                        if t0 + u < ln:
                            T.copy(dw_bf16_ub[u, 0:heads], dWeights[bos + t0 + u, 0:heads])
                if layout == 0:
                    # whole-tile fp32 region atomic (gqa_sink:437 form)
                    T.tile.atomic_add(ws_dk[bos : bos + topk, 0:dim], dk_l0c)
                else:
                    # per-row atomic from an fp32 UB buffer (rev1-validated
                    # form): whole-fragment L0C -> GM ws -> UB, then per-row
                    # scatter. pos is re-read from TopkIndices GM: the atomic
                    # rides the Cube/FIX pipe, which cannot address UB buffers.
                    T.copy(dk_l0c, workspace_dkf[cid, 0:topk, 0:dim])
                    T.copy(workspace_dkf[cid, 0:topk, 0:dim], dk_ub)
                    for i in T.serial(topk):
                        pos = TopkIndices[bos, i]
                        if (pos > -1) & (pos <= t_max):
                            T.tile.atomic_add(ws_dk[bos + pos, 0], dk_ub[i, :])

    return main


def tl_indexer_bwd_fast(
    heads,
    dim,
    topk,
    sm_scale=None,
    T_BLOCK=32,
    TS=4,
    grid_size=20,
    layout=0,
):
    """Build the fast-path kernel and wrap it in the cached low-overhead launcher.

    Same signature/semantics as the underlying jit builder (see
    ``_tl_indexer_bwd_fast_impl``); the returned callable additionally caches
    the marshaled launch arguments so steady-state calls (same tensors) submit
    in ~13 us instead of ~750 us (see ``_CachedLauncher``).
    """
    kernel = _tl_indexer_bwd_fast_impl(
        heads=heads,
        dim=dim,
        topk=topk,
        sm_scale=sm_scale,
        T_BLOCK=T_BLOCK,
        TS=TS,
        grid_size=grid_size,
        layout=layout,
    )
    return _CachedLauncher(kernel) if hasattr(kernel, "adapter") else kernel


# ============================================================================
# Generic kernel (rev1 per-token structure + dk fp32 workspace exit)
# ============================================================================


@tilelang.jit(pass_configs=_pass_configs)
def tl_indexer_bwd_generic(heads, dim, topk, sm_scale=None, block_I=64):
    """Build the generic per-token kernel (Developer mode, grid = seq_len).

    rev1 structure (dq/dw precision validated) with the rev2 fixes: dk exits
    through the fp32 ws_dk workspace via the rev1-validated UB transit
    (whole-fragment L0C -> GM ws -> fp32 UB -> per-row fp32 atomic; no bf16
    rounding on the accumulation path), dIndexK bf16 is produced by the
    trailing cast kernel, and the diff carries the A_t correction (docstring
    note 7) so non-arange index patterns match the autograd golden exactly.

    Args:
        heads: number of heads H (% 16 == 0 and <= 64).
        dim: head dim D (% 16 == 0).
        topk: selected positions (power of two, % block_I == 0).
        sm_scale: softmax scale, defaults to dim ** -0.5.
        block_I: topk-direction tile size (multiple of 16).
    """
    assert topk == tilelang.math.next_power_of_2(topk), "topk must be a power of 2"
    assert topk % block_I == 0, "topk must be divisible by block_I"
    assert block_I % 16 == 0, "block_I must be a multiple of 16 (Cube fractal)"
    assert heads % 16 == 0 and heads <= 64, "heads must be % 16 == 0 and <= 64 (Cube fractal)"
    assert dim % 16 == 0, "dim must be a multiple of 16 (Cube fractal)"
    if sm_scale is None:
        sm_scale = dim**-0.5

    NI = topk // block_I

    seq_len = T.symbolic("seq_len")
    batch_plus_one = T.symbolic("batch_plus_one")

    dtype = "bfloat16"
    accum_dtype = "float32"
    indices_dtype = "int32"

    @T.prim_func
    def main(
        IndexQ: T.Tensor([seq_len, heads, dim], dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], dtype),  # type: ignore
        IndexK: T.Tensor([seq_len, dim], dtype),  # type: ignore
        AttnScore: T.Tensor([seq_len, topk], accum_dtype),  # type: ignore
        IndexScore: T.Tensor([seq_len, topk], accum_dtype),  # type: ignore
        TopkIndices: T.Tensor([seq_len, topk], indices_dtype),  # type: ignore
        Offsets: T.Tensor([batch_plus_one], indices_dtype),  # type: ignore
        TokenIndices: T.Tensor([seq_len, 2], indices_dtype),  # type: ignore
        dIndexQ: T.Tensor([seq_len, heads, dim], dtype),  # type: ignore
        dWeights: T.Tensor([seq_len, heads], dtype),  # type: ignore
        ws_dk: T.Tensor([seq_len, dim], accum_dtype),  # type: ignore
        workspace_k: T.Tensor([seq_len, topk, dim], dtype),  # type: ignore
        workspace_dlog: T.Tensor([seq_len, topk, heads], dtype),  # type: ignore
        workspace_logits: T.Tensor([seq_len, topk, heads], accum_dtype),  # type: ignore
        workspace_dk: T.Tensor([seq_len, topk, dim], accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=1, is_npu=True) as (cid):
            i_b, i_t = TokenIndices[cid, 0], TokenIndices[cid, 1]
            bos = Offsets[i_b]

            # ---- L1: Cube GEMM operands ----
            q_l1 = T.alloc_shared([heads, dim], dtype)
            k_l1 = T.alloc_shared([block_I, dim], dtype)
            dlog_l1 = T.alloc_shared([block_I, heads], dtype)

            # ---- L0C: GEMM accumulators ----
            logits_l0c = T.alloc_fragment([block_I, heads], accum_dtype)
            dq_l0c = T.alloc_fragment([heads, dim], accum_dtype)
            dk_l0c = T.alloc_fragment([block_I, dim], accum_dtype)

            # ---- UB: Vector-side buffers ----
            indices_ub = T.alloc_shared([block_I], indices_dtype)
            attn_ub = T.alloc_shared([block_I], accum_dtype)
            is_ub = T.alloc_shared([block_I], accum_dtype)
            gather_ub = T.alloc_shared([block_I, dim], dtype)
            logits_ub = T.alloc_shared([block_I, heads], accum_dtype)
            dw_ub = T.alloc_shared([block_I, heads], accum_dtype)
            dlog_ub = T.alloc_shared([block_I, heads], accum_dtype)
            mask_ub = T.alloc_shared([block_I, heads], accum_dtype)
            diff2d = T.alloc_shared([block_I, heads], accum_dtype)
            w2d = T.alloc_shared([block_I, heads], accum_dtype)
            zero_ub = T.alloc_shared([block_I, heads], accum_dtype)
            logits_bf16_g = T.alloc_shared([block_I, heads], dtype)  # dw-grain cast hop
            logits_dw = T.alloc_shared([block_I, heads], accum_dtype)  # dw-grain L
            # clip-exact gradient slots (docstring note 7)
            is2d = T.alloc_shared([block_I, heads], accum_dtype)
            pt_g = T.alloc_shared([block_I, heads], accum_dtype)
            attn2d = T.alloc_shared([block_I, heads], accum_dtype)
            a2d = T.alloc_shared([block_I, heads], accum_dtype)
            a_row_g = T.alloc_shared([heads], accum_dtype)
            a_acc = T.alloc_shared([heads], accum_dtype)
            w_ub_bf16 = T.alloc_shared([heads], dtype)
            w_ub = T.alloc_shared([heads], accum_dtype)
            dw_frag = T.alloc_shared([heads], accum_dtype)
            dw_bf16 = T.alloc_shared([heads], dtype)
            dlog_bf16_ub = T.alloc_shared([block_I, heads], dtype)
            dk_ub = T.alloc_shared([block_I, dim], accum_dtype)

            # ---- per-token prologue ----
            T.copy(IndexQ[bos + i_t, 0:heads, 0:dim], q_l1)
            T.copy(Weights[bos + i_t, 0:heads], w_ub_bf16)
            T.copy(w_ub_bf16, w_ub)  # bf16 -> fp32 auto-cast
            T.tile.broadcast(w2d, w_ub, axis=0)  # [H] -> [block_I, H]
            T.tile.fill(zero_ub, 0.0)
            T.tile.fill(dw_frag, 0.0)
            T.tile.fill(a_acc, 0.0)

            # ---- pre-pass: A' = sum over ALL clip-passing slots across the
            #      NI blocks (the log_softmax normalizer spans the full topk;
            #      per-block partial sums would under-count, docstring note 7).
            #      pt = (is > fp32-normal floor) as a proxy for the golden's
            #      log_softmax > -100 clip (monotone in is). ----
            for it0 in T.serial(NI):
                st0 = it0 * block_I
                T.copy(TopkIndices[bos + i_t, st0 : st0 + block_I], indices_ub)
                T.copy(AttnScore[bos + i_t, st0 : st0 + block_I], attn_ub)
                T.copy(IndexScore[bos + i_t, st0 : st0 + block_I], is_ub)
                T.tile.broadcast(is2d, is_ub, axis=1)  # [block_I, H]
                T.tile.compare(pt_g, is2d, T.float32(3.7200759760208356e-44), "GT")  # packed pt
                T.tile.broadcast(attn2d, attn_ub, axis=1)
                T.tile.select(attn2d, pt_g, attn2d, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.reduce_sum(attn2d, a_row_g, dim=0)  # [H] partial
                T.tile.add(a_acc, a_acc, a_row_g)  # accumulate across blocks

            for it in T.serial(NI):
                st = it * block_I

                # ---- V: load indices + scores ----
                T.copy(TopkIndices[bos + i_t, st : st + block_I], indices_ub)
                T.copy(AttnScore[bos + i_t, st : st + block_I], attn_ub)
                T.copy(IndexScore[bos + i_t, st : st + block_I], is_ub)

                # ---- V: gather K rows; invalid pos keeps the fill-0 row ----
                T.tile.fill(gather_ub, 0.0)
                for i in T.serial(block_I):
                    pos = indices_ub[i]
                    if (pos > -1) & (pos <= i_t):
                        T.copy(IndexK[bos + pos, 0:dim], gather_ub[i, 0:dim])
                T.copy(gather_ub, workspace_k[cid, st : st + block_I, 0:dim])

                # ---- C: workspace -> L1, GEMM1 logits = K x Q^T ----
                T.copy(workspace_k[cid, st : st + block_I, 0:dim], k_l1)
                T.gemm_v0(k_l1, q_l1, logits_l0c, transpose_B=True, init=True)

                # ---- C: L0C -> GM workspace (relu folded into the copy) ----
                T.copy(logits_l0c, workspace_logits[cid, st : st + block_I, 0:heads], enable_relu=True)

                # ---- V: logits -> UB (relu'd, UNSCALED — sm_scale rides the
                #      dw-grain L below and the dlog tail) ----
                T.copy(workspace_logits[cid, st : st + block_I, 0:heads], logits_ub)

                # ---- V: clip-exact diff (docstring note 7):
                #      pt = (is > fp32-normal floor) [golden clip passthrough
                #      proxy]; A' = sum_k attn_k * pt_k (pre-pass, spans all
                #      NI blocks); diff = is * A' - attn * pt ----
                T.tile.broadcast(is2d, is_ub, axis=1)
                T.tile.compare(pt_g, is2d, T.float32(3.7200759760208356e-44), "GT")
                T.tile.broadcast(attn2d, attn_ub, axis=1)
                T.tile.select(attn2d, pt_g, attn2d, 0.0, "VSEL_TENSOR_SCALAR_MODE")  # attn * pt
                T.tile.broadcast(a2d, a_acc, axis=0)  # A'_2d (token-total)
                T.tile.mul(is2d, is2d, a2d)  # is * A'
                T.tile.sub(diff2d, is2d, attn2d)  # is*A' - attn*pt

                # ---- V: dWeights accumulation (across NI blocks) ----
                T.tile.compare(mask_ub, logits_ub, zero_ub, "GT")  # 1{qk>0}
                # NaN guard: mask diff through the relu indicator BEFORE the
                # multiply (NaN * 0 = NaN hazard on all-invalid tokens)
                T.tile.select(diff2d, mask_ub, diff2d, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                # dw integrand logits at the golden's bf16 grain (note 8):
                # L = bf16(bf16(relu(qk)) * sm_scale)
                T.copy(logits_ub, logits_bf16_g)  # fp32 -> bf16
                T.copy(logits_bf16_g, logits_dw)  # bf16 -> fp32
                T.tile.mul(logits_dw, logits_dw, sm_scale)
                T.copy(logits_dw, logits_bf16_g)  # -> bf16
                T.copy(logits_bf16_g, logits_dw)  # -> fp32 (dw-grain L)
                T.tile.mul(dw_ub, diff2d, logits_dw)
                T.reduce_sum(dw_ub, dw_frag, dim=0, clear=False)

                # ---- V: d_logits = diff x 1{logits>0} x Weights, fold sm_scale ----
                T.tile.mul(dlog_ub, diff2d, w2d)
                T.tile.select(dlog_ub, mask_ub, dlog_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.tile.mul(dlog_ub, dlog_ub, sm_scale)
                T.copy(dlog_ub, dlog_bf16_ub)  # fp32 -> bf16 auto-cast
                T.copy(dlog_bf16_ub, workspace_dlog[cid, st : st + block_I, 0:heads])

                # ---- C: workspace -> L1, GEMM2 dq / GEMM3 dk ----
                T.copy(workspace_dlog[cid, st : st + block_I, 0:heads], dlog_l1)
                T.gemm_v0(dlog_l1, k_l1, dq_l0c, transpose_A=True, init=(it == 0))
                T.gemm_v0(dlog_l1, q_l1, dk_l0c, init=True)

                # ---- V: dk per-row fp32 atomic into ws_dk (rev1-validated
                # structure: whole-fragment L0C -> GM ws -> UB, per-row UB
                # atomic; per-row L0C direct src is unreliable — docstring
                # note 3). pos is re-read from TopkIndices GM (not the UB
                # copy): the atomic rides the Cube/FIX pipe, which cannot
                # address UB buffers. ----
                T.copy(dk_l0c, workspace_dk[cid, st : st + block_I, 0:dim])
                T.copy(workspace_dk[cid, st : st + block_I, 0:dim], dk_ub)
                for i in T.serial(block_I):
                    pos = TopkIndices[bos + i_t, st + i]
                    if (pos > -1) & (pos <= i_t):
                        T.tile.atomic_add(ws_dk[bos + pos, 0], dk_ub[i, :])

            # ---- tail writeback ----
            T.copy(dq_l0c, dIndexQ[bos + i_t, 0:heads, 0:dim])  # L0C fp32 -> GM bf16
            T.copy(dw_frag, dw_bf16)  # fp32 -> bf16 auto-cast
            T.copy(dw_bf16, dWeights[bos + i_t, 0:heads])

    return main


# ============================================================================
# dk cast kernel (ws_dk fp32 -> dIndexK bf16, persistent grid)
# ============================================================================


@tilelang.jit(pass_configs=_pass_configs)
def _tl_dk_cast_impl(dim, grid_size=20, cast_rows=16):
    """Build the trailing cast kernel: ws_dk [seq_len, dim] fp32 -> dIndexK bf16.

    Runs after the main kernel on the same stream (host program order); rows
    are processed in cast_rows chunks striped over the persistent grid.
    """

    seq_len = T.symbolic("seq_len")

    @T.prim_func
    def main(
        ws_dk: T.Tensor([seq_len, dim], "float32"),  # type: ignore
        dIndexK: T.Tensor([seq_len, dim], "bfloat16"),  # type: ignore
    ):
        with T.Kernel(grid_size, threads=1, is_npu=True) as (cid):
            ub = T.alloc_shared([cast_rows, dim], "float32")
            ub_bf16 = T.alloc_shared([cast_rows, dim], "bfloat16")
            total = T.ceildiv(seq_len, cast_rows)
            ntask = T.max(0, T.ceildiv(total - cid, grid_size))
            for t in T.serial(ntask):
                c = t * grid_size + cid
                row0 = c * cast_rows
                if row0 + cast_rows <= seq_len:
                    T.copy(ws_dk[row0 : row0 + cast_rows, 0:dim], ub)
                    T.copy(ub, ub_bf16)  # fp32 -> bf16 auto-cast
                    T.copy(ub_bf16, dIndexK[row0 : row0 + cast_rows, 0:dim])
                else:
                    for u in T.serial(cast_rows):
                        if row0 + u < seq_len:
                            T.copy(ws_dk[row0 + u, 0:dim], ub[u, 0:dim])
                            T.copy(ub[u, 0:dim], ub_bf16[u, 0:dim])
                            T.copy(ub_bf16[u, 0:dim], dIndexK[row0 + u, 0:dim])

    return main


def tl_dk_cast(dim, grid_size=20, cast_rows=16):
    """Build the cast kernel wrapped in the cached low-overhead launcher.

    Same signature/semantics as ``_tl_dk_cast_impl`` (see its docstring); the
    returned callable uses the cached launch path (see ``_CachedLauncher``).
    """
    kernel = _tl_dk_cast_impl(dim=dim, grid_size=grid_size, cast_rows=cast_rows)
    return _CachedLauncher(kernel) if hasattr(kernel, "adapter") else kernel


# ============================================================================
# Host side: data preparation + detection + dispatch entry
# ============================================================================


def prepare_token_indices(offsets):
    """Build TokenIndices [sum(S), 2] int32 ((batch_id, position)) from cu_seqlens.

    CPU construction followed by a single H2D copy (sparse_mla precedent).
    """
    off = offsets.detach().cpu().tolist()
    batch_ids = []
    positions = []
    for b in range(len(off) - 1):
        n = int(off[b + 1]) - int(off[b])
        batch_ids.append(torch.full((n,), b, dtype=torch.int32))
        positions.append(torch.arange(n, dtype=torch.int32))
    if batch_ids:
        ti = torch.stack([torch.cat(batch_ids), torch.cat(positions)], dim=1)
    else:
        ti = torch.zeros((0, 2), dtype=torch.int32)
    return ti.to(offsets.device)


def prepare_block_indices(offsets, t_block):
    """Build BlockIndices [num_blocks, 3] int32 ((bos, t0_local, seq_len)).

    One row per ceildiv(len_i, t_block) block of each varlen segment; CPU
    construction followed by a single H2D copy (same class as
    prepare_token_indices).
    """
    off = offsets.detach().cpu().tolist()
    rows = []
    for b in range(len(off) - 1):
        bos = int(off[b])
        ln = int(off[b + 1]) - int(off[b])
        for t0 in range(0, ln, t_block):
            rows.append((bos, t0, ln))
    if rows:
        bi = torch.tensor(rows, dtype=torch.int32)
    else:
        bi = torch.zeros((0, 3), dtype=torch.int32)
    return bi.to(offsets.device)


def _detect_indices_layout(topk_indices, topk):
    """Exact (non-heuristic) shared-layout detection (DESIGN §1.4).

    Returns:
        0 = every row equals arange(topk)   -> fast layout=0
        1 = all rows strictly equal         -> fast layout=1
        2 = otherwise                       -> generic
    """
    ar = torch.arange(topk, dtype=torch.int32, device=topk_indices.device)
    if bool((topk_indices == ar).all().item()):
        return 0
    if bool((topk_indices == topk_indices[:1]).all().item()):
        return 1
    return 2


_BUILD_CACHE: dict = {}

_FAST_T_BLOCK = 32
_FAST_TS = 4


def _get_cube_core_num(default=20):
    """Query the device cube-core count (persistent grid constant; 910B3 = 20)."""
    try:
        if torch.npu.is_available():
            n = torch.npu.get_device_properties(0).cube_core_num
            if isinstance(n, int) and n > 0:
                return n
    except Exception:  # pragma: no cover - fall back on probe failure
        pass
    return default


def _fast_gate_ok(heads, dim, topk):
    """Host mirror of the fast-path domain checks (DESIGN §10.4).

    L0C budget is the conservative no-liveness-reuse model: heads=32 @
    topk=64/dim=128 lands exactly at 128KB and is rejected (strict <).
    """
    if topk not in (16, 32, 64):
        return False
    if heads % 16 != 0 or heads > 64:
        return False
    if dim % 16 != 0:
        return False
    ts_h = _FAST_TS * heads
    if ts_h * (topk + dim) * 4 + topk * dim * 4 >= 128 * 1024:
        return False
    ub_bytes = (2 * ts_h * topk + 3 * _FAST_T_BLOCK * topk + 2 * _FAST_T_BLOCK * heads + 12 * heads * topk + ts_h * topk // 2) * 4
    return ub_bytes < 160 * 1024


def _run_fast(
    q,
    weights,
    k,
    attn_score,
    index_score,
    topk_indices,
    offsets,
    layout,
    sm_scale,
):
    """Launch the fast kernel + cast kernel for a validated layout contract."""
    seq_len, heads, dim = q.shape
    topk = topk_indices.shape[-1]
    device = q.device
    grid_size = _get_cube_core_num()
    t_block, ts = _FAST_T_BLOCK, _FAST_TS

    block_indices = prepare_block_indices(offsets, t_block)
    dq = torch.zeros(seq_len, heads, dim, dtype=q.dtype, device=device)
    dw = torch.zeros(seq_len, heads, dtype=weights.dtype, device=device)
    dk = torch.zeros(seq_len, dim, dtype=k.dtype, device=device)
    ws_dk = torch.zeros(seq_len, dim, dtype=torch.float32, device=device)
    ts_h = ts * heads
    workspace_logits = torch.empty(grid_size, 2, ts_h, topk, dtype=torch.float32, device=device)
    workspace_dlog_a = torch.empty(grid_size, 2, ts_h, topk, dtype=torch.bfloat16, device=device)
    workspace_dlog_b = torch.empty(grid_size, 2, ts_h, topk, dtype=torch.bfloat16, device=device)
    workspace_dq = torch.empty(grid_size, 2, ts_h, dim, dtype=torch.bfloat16, device=device)
    workspace_k = torch.empty(grid_size, topk, dim, dtype=k.dtype, device=device)
    workspace_dkf = torch.empty(grid_size, topk, dim, dtype=torch.float32, device=device)

    key = (heads, dim, topk, sm_scale, t_block, ts, grid_size, layout)
    func = _BUILD_CACHE.get(key)
    if func is None:
        func = tl_indexer_bwd_fast(
            heads=heads,
            dim=dim,
            topk=topk,
            sm_scale=sm_scale,
            T_BLOCK=t_block,
            TS=ts,
            grid_size=grid_size,
            layout=layout,
        )
        _BUILD_CACHE[key] = func
    ckey = (dim, grid_size)
    cast_f = _BUILD_CACHE.get(ckey)
    if cast_f is None:
        cast_f = tl_dk_cast(dim=dim, grid_size=grid_size)
        _BUILD_CACHE[ckey] = cast_f

    func(
        q,
        weights,
        k,
        attn_score,
        index_score,
        topk_indices,
        block_indices,
        workspace_logits,
        workspace_dlog_a,
        workspace_dlog_b,
        workspace_dq,
        workspace_k,
        workspace_dkf,
        ws_dk,
        dq,
        dw,
    )
    cast_f(ws_dk, dk)
    return dq, dw, dk


def _run_generic(
    q,
    weights,
    k,
    attn_score,
    index_score,
    topk_indices,
    offsets,
    sm_scale,
    block_I,
):
    """Launch the generic kernel + cast kernel (full-semantic fallback)."""
    seq_len, heads, dim = q.shape
    topk = topk_indices.shape[-1]
    device = q.device

    token_indices = prepare_token_indices(offsets)
    dq = torch.zeros(seq_len, heads, dim, dtype=q.dtype, device=device)
    dw = torch.zeros(seq_len, heads, dtype=weights.dtype, device=device)
    dk = torch.zeros(seq_len, dim, dtype=k.dtype, device=device)
    ws_dk = torch.zeros(seq_len, dim, dtype=torch.float32, device=device)
    workspace_k = torch.empty(seq_len, topk, dim, dtype=k.dtype, device=device)
    workspace_dlog = torch.empty(seq_len, topk, heads, dtype=q.dtype, device=device)
    workspace_logits = torch.empty(seq_len, topk, heads, dtype=torch.float32, device=device)
    workspace_dk = torch.empty(seq_len, topk, dim, dtype=torch.float32, device=device)

    key = (heads, dim, topk, sm_scale, block_I, "generic")
    func = _BUILD_CACHE.get(key)
    if func is None:
        func = tl_indexer_bwd_generic(
            heads=heads,
            dim=dim,
            topk=topk,
            sm_scale=sm_scale,
            block_I=block_I,
        )
        _BUILD_CACHE[key] = func
    ckey = (dim, _get_cube_core_num())
    cast_f = _BUILD_CACHE.get(ckey)
    if cast_f is None:
        cast_f = tl_dk_cast(dim=dim, grid_size=ckey[1])
        _BUILD_CACHE[ckey] = cast_f

    func(
        q,
        weights,
        k,
        attn_score,
        index_score,
        topk_indices,
        offsets,
        token_indices,
        dq,
        dw,
        ws_dk,
        workspace_k,
        workspace_dlog,
        workspace_logits,
        workspace_dk,
    )
    cast_f(ws_dk, dk)
    return dq, dw, dk


def indexer_bwd_interface(
    q,
    weights,
    k,
    attn_score,
    index_score,
    topk_indices,
    offsets,
    block_I=None,
    dispatch="auto",
    sm_scale=None,
):
    """Host entry: dispatch to fast/generic kernels, return (dq, dw, dk).

    Args:
        q: IndexQ [seq_len, heads, dim] bfloat16.
        weights: Weights [seq_len, heads] bfloat16.
        k: IndexK [seq_len, dim] bfloat16.
        attn_score: teacher probabilities [seq_len, topk] float32.
        index_score: student probabilities [seq_len, topk] float32.
        topk_indices: selected positions [seq_len, topk] int32 (values in
            [-1, seq_len); -1 is handled by the generic path only).
        offsets: cu_seqlens [batch + 1] int32; every segment length >= topk.
        block_I: generic-path topk tile (default min(topk, 64)).
        dispatch: "auto" (exact host detection), "fast_arange" (layout=0),
            "fast_shared" (layout=1), "generic". Forced paths validate their
            index contract and raise ValueError on violation.
        sm_scale: softmax scale (default dim ** -0.5, the forward semantics;
            exposed per proto_indexer_bwd.yaml schema — callers computing the
            backward of a differently-scaled forward must pass the same scale
            used by that forward and by the golden reference).

    Returns:
        (dIndexQ, dWeights, dIndexK) bfloat16 tensors.
    """
    seq_len, heads, dim = q.shape
    topk = topk_indices.shape[-1]
    if block_I is None:
        block_I = min(topk, 64)
    if sm_scale is None:
        sm_scale = dim**-0.5

    # Host-side domain validation (mirrors the kernel compile-time asserts).
    if topk != tilelang.math.next_power_of_2(topk):
        raise ValueError(f"topk must be a power of 2, got {topk}")
    if topk % block_I != 0:
        raise ValueError(f"topk ({topk}) must be divisible by block_I ({block_I})")
    if block_I % 16 != 0:
        raise ValueError(f"block_I must be a multiple of 16, got {block_I}")
    if heads % 16 != 0 or heads > 64:
        raise ValueError(f"heads must be % 16 == 0 and <= 64, got {heads}")
    if dim % 16 != 0:
        raise ValueError(f"dim must be a multiple of 16, got {dim}")

    off = offsets.detach().cpu().tolist()
    for b in range(len(off) - 1):
        seg_len = int(off[b + 1]) - int(off[b])
        if seg_len < topk:
            raise ValueError(f"varlen segment {b} has length {seg_len} < topk {topk}")

    if dispatch not in ("auto", "fast_arange", "fast_shared", "generic"):
        raise ValueError(f"dispatch must be auto/fast_arange/fast_shared/generic, got {dispatch!r}")

    if dispatch == "generic":
        return _run_generic(q, weights, k, attn_score, index_score, topk_indices, offsets, sm_scale, block_I)

    if dispatch in ("fast_arange", "fast_shared"):
        if not _fast_gate_ok(heads, dim, topk):
            raise ValueError(f"fast path out of domain (heads={heads}, dim={dim}, topk={topk}); use dispatch='generic'")
        want = 0 if dispatch == "fast_arange" else 1
        got = _detect_indices_layout(topk_indices, topk)
        if want == 0 and got != 0:
            raise ValueError("dispatch='fast_arange' requires arange-tiled topk_indices (all rows == arange(topk))")
        if want == 1 and got == 2:
            raise ValueError("dispatch='fast_shared' requires all topk_indices rows to be identical")
        return _run_fast(q, weights, k, attn_score, index_score, topk_indices, offsets, want, sm_scale)

    # dispatch == "auto": exact detection; gate miss falls back to generic
    if _fast_gate_ok(heads, dim, topk):
        layout = _detect_indices_layout(topk_indices, topk)
        if layout in (0, 1):
            return _run_fast(q, weights, k, attn_score, index_score, topk_indices, offsets, layout, sm_scale)
    return _run_generic(q, weights, k, attn_score, index_score, topk_indices, offsets, sm_scale, block_I)


# Module-level alias matching the file name (import convenience for tests).
indexer_bwd = indexer_bwd_interface
