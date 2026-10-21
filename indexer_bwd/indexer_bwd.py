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

   1. [rev3] IndexQ/dIndexQ are declared as flat [S*H, D] and Weights as flat
      [S*H] kernel parameters (zero-copy views applied by the cached launcher),
      so one whole-sub-tile Q load and one whole-tile dq writeback are
      expressible under T.copy's structural-equality rule (the [S, H, D]
      declaration would force per-token [H, D] row copies, 4x the DMA count).
   2. [rev3] The kernel is a software-pipelined PAIR loop (T_BLOCK/TS/2 + 1
      iterations): iteration p first PRODUCES the pair's logits on the Cube
      side (one [2*TS*H, dim] Q load, one M=2*TS*H GEMM1, one FIX ->
      ws_logits) and then CONSUMES the previous pair's dlog (GEMM2/3 + the
      whole-pair dq writeback), letting C run one pair ahead of V. This
      bundles 24 gemm_v0 calls per block into 12. The pre-restructure kernel
      was fully CV-lockstep (AIC spent 92% of its wall waiting on the
      cross-core dlog flag, 50.5 of 54.7 us per block, msprof). Slot safety
      with the p%2 double buffers follows from the flag order alone: C waits
      for dlog(p-1) before producing p+1 (so V finished reading
      ws_logits(p-2)), and V waits for logits(p) before rewriting
      dlog(p-2) (so C finished reading it — GEMM2's internal MTE2_M
      queue-drain orders the read before the next FIX flag). Q_l1 is
      double-buffered at pair granularity for the same reason. Token order
      is kept everywhere (no row permutation): the L0C/L1 fractal layouts
      (zN bands / N-major 16-column bands) admit no row slicing, so every
      GEMM operand and every L0C->GM write is a whole-tile copy.
   3. relu + the first dw-grain bf16 round ride the L0C->GM transit leg
      (`enable_relu=True` on the bf16 workspace_logits write): the transit
      halves (16KB -> 8KB fp32-equivalent per direction) and hands V the
      bf16(relu(qk)) grain the dw integrand needs anyway (docstring note 8);
      sign-exact for the d_relu indicator (s > 0, rounding preserves sign).
   4. dq writeback is a direct whole-tile L0C->GM copy (fp32 -> bf16
      auto-cast, gqa_sink:478 form) for whole sub-tiles; only varlen tail
      sub-tiles fall back to the ws_dq transit + per-token UB row writes
      (per-token L0C row-slice copies are unreliable beyond the first
      fractal tile — rev2 L0 evidence). dk keeps the layout=0 whole-tile
      fp32 region atomic (validated 0.9977) and the layout=1 per-row path
      reads ws_dkf one row at a time into a [dim] UB buffer.
   5. Tail-block validity folds into the score loads: tail blocks zero-fill
      attn/is/weights before the guarded per-row copies, so invalid-token
      diff rows are exactly 0 and their dq/dw/dk contributions vanish
      without any per-row validity vector in the sub-tile chain (the
      pre-restructure t_eff = -1 trick needed a per-token fill+compare).
   6. Tail-block Q loads clamp the row index to len-1 (always a valid row)
      and every loaded row is finite, so no 0 * NaN hazard can propagate
      through the GEMMs; invalid-token contributions are masked to exactly 0.
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
      logits through bf16 twice (fp32 -> bf16 [transit] -> fp32 -> *sm_scale ->
      bf16 -> fp32), aligning both sides to ~1 ulp; dq/dk are unaffected (they
      only consume the logit SIGN via the relu indicator).
   9. [rev3] The per-token vector segment is a whole-sub-tile [TS*H, topk]
      chain: the causal mask collapses to one compare
      `(t_cur*H + r >= col_scaled[j])` (row r maps to token r // H), the diff
      and weight expansions are per-token row broadcasts, and the dw reduce
      runs once per sub-tile (its [TS*H] output maps 1:1 onto the flat
      (token, head) layout of dw_acc). This cuts the vector issue count ~4x
      vs the rev2 per-token [H, topk] chain (96 -> ~20 tile ops per sub-tile).
  10. [rev4] MIX_AIC_1_2 via the (cid, vid) launch form: 1 Cube + 2 Vector
      cores per block (the 910B3's 40 AIVs were half idle before). AIV-vid
      owns sub-tile 2p+vid of every pair — contiguous rows in every transit,
      and both AIVs write every cross-core workspace (the 1_2 flag handshake
      requires BOTH AIV sub-blocks to set each flag the AIC waits on; a
      sub-tile-parity split that left one AIV out of a transit deadlocked
      deterministically). The block-level score/diffm prologue is duplicated
      on both AIVs.
  11. [rev4] dk_l0c accumulates across the core's contiguous same-segment
      blocks and flushes the fp32 region atomic once per segment run
      (65 -> ~22 atomics for the l0_default shape). Contiguous task
      assignment also lets the K panel reload only at segment boundaries
      (t0 == 0 or the chunk start).

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

    ``view_spec`` maps argument positions to zero-copy reshape modes for
    kernels whose parameters are declared in a flattened layout (e.g. the
    fast kernel's IndexQ/dIndexQ as [S*H, D] and Weights as [S*H]): raw
    caller-side tensors are viewed once per cache entry; the cached views are
    kept alive alongside the originals.

    Safety model: the single cache slot holds strong references to the exact
    argument objects of the last marshaled call, keyed on their ``id()`` tuple.
    While the references are alive those ids cannot be recycled, so a key hit
    guarantees the cached data pointers / dynamic-shape values are still the
    right ones for these very tensors. Any miss (fresh tensors, changed
    arg count, explicit stream) falls back to the full validating path.
    """

    __slots__ = (
        "_kernel", "_adapter", "_lib", "_params_len", "_key", "_args", "_args_refs", "_dev",
        "_view_spec",
    )

    def __init__(self, kernel, view_spec=None):
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
        self._view_spec = view_spec or {}

    def __getattr__(self, name):  # delegate get_kernel_source()/etc.
        return getattr(self._kernel, name)

    @staticmethod
    def _apply_view(t, mode):
        if mode == "flat2d":
            return t.reshape(-1, t.shape[-1])
        if mode == "flat1d":
            return t.reshape(-1)
        return t

    def _marshal(self, args):
        ad = self._adapter
        tensors = [
            self._apply_view(t, self._view_spec.get(i, "")) if isinstance(t, torch.Tensor) else t
            for i, t in enumerate(args)
        ]
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
                # commit slot (strong refs via _args_refs keep ids and views alive)
                self._key = key
                self._args = c_args
                self._args_refs = args
                self._dev = dev
                self._lib.call(*c_args)
                return
        # full validating fallback (also applies the view spec)
        args2 = tuple(
            self._apply_view(t, self._view_spec.get(i, "")) if isinstance(t, torch.Tensor) else t
            for i, t in enumerate(args)
        )
        return self._kernel(*args2, stream=stream) if stream != -1 else self._kernel(*args2)

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
    TS_HH = TS_H // 2  # per-AIV half-tile rows (MIX_AIC_1_2 vector split)
    assert TS_H % 16 == 0, "TS*heads must be a multiple of 16 (GEMM M)"
    assert (TS // 2) >= 1 and (TS_H // 2) % 16 == 0, "TS*heads/2 must be a multiple of 16"
    l0c_bytes = 2 * TS_H * topk * 4 + 2 * TS_H * dim * 4 + topk * dim * 4
    assert l0c_bytes <= 128 * 1024, f"L0C budget exceeded: {l0c_bytes} > 131072"
    if sm_scale is None:
        sm_scale = dim**-0.5
    TSB = T_BLOCK // TS
    assert TSB % 2 == 0, "T_BLOCK/TS (sub-tile count) must be even (pair-GEMM)"
    PAIRS = TSB // 2  # sub-tile pairs per block; one M=2*TS_H GEMM bundle each
    PH = 2 * TS_H  # pair rows (GEMM M for the bundled pair gemms)

    seq_len = T.symbolic("seq_len")  # total packed length (sum of segment lens)
    num_blocks = T.symbolic("num_blocks")  # BlockIndices rows (persistent task count)
    q_rows = T.symbolic("q_rows")  # = seq_len * heads (flat-2D IndexQ view)
    w_rows = T.symbolic("w_rows")  # = seq_len * heads (flat-1D Weights view)
    dq_rows = T.symbolic("dq_rows")  # = seq_len * heads (flat-2D dIndexQ view)

    dtype = "bfloat16"
    accum_dtype = "float32"
    indices_dtype = "int32"

    @T.prim_func
    def main(
        IndexQ: T.Tensor([q_rows, dim], dtype),  # type: ignore  # flat view [S*H, D]
        Weights: T.Tensor([w_rows], dtype),  # type: ignore  # flat view [S*H]
        IndexK: T.Tensor([seq_len, dim], dtype),  # type: ignore
        AttnScore: T.Tensor([seq_len, topk], accum_dtype),  # type: ignore
        IndexScore: T.Tensor([seq_len, topk], accum_dtype),  # type: ignore
        TopkIndices: T.Tensor([seq_len, topk], indices_dtype),  # type: ignore  # layout=1 only
        BlockIndices: T.Tensor([num_blocks, 3], indices_dtype),  # type: ignore
        workspace_logits: T.Tensor([grid_size, 2, PH, topk], dtype),  # type: ignore  # bf16, pair slots
        workspace_dlog_a: T.Tensor([grid_size, PH, topk], dtype),  # type: ignore  # bf16, even pairs
        workspace_dlog_b: T.Tensor([grid_size, PH, topk], dtype),  # type: ignore  # bf16, odd pairs
        workspace_dq: T.Tensor([grid_size, PH, dim], dtype),  # type: ignore  # tail pairs only
        workspace_k: T.Tensor([grid_size, topk, dim], dtype),  # type: ignore  # layout=1 only
        workspace_dkf: T.Tensor([grid_size, topk, dim], accum_dtype),  # type: ignore  # layout=1 only
        ws_dk: T.Tensor([seq_len, dim], accum_dtype),  # type: ignore
        dIndexQ: T.Tensor([dq_rows, dim], dtype),  # type: ignore  # flat view [S*H, D]
        dWeights: T.Tensor([w_rows], dtype),  # type: ignore  # flat view [S*H]
    ):
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):
            # MIX_AIC_1_2: 1 Cube + 2 Vector cores per block (the codegen's
            # default mix ratio for the (cid, vid) launch form; AIVs see
            # cid = GetBlockIdx()/2 so both AIV sub-blocks of a block share
            # the logical block id, vid = GetSubBlockIdx() in {0, 1}). vid
            # splits the vector chain by sub-tile parity: each sub-tile's
            # logits read / dlog write executes on exactly one AIV, so every
            # cross-core flag handshake stays 1:1 (no aggregation semantics
            # needed). The block-level score/diffm prologue is duplicated on
            # both AIVs (cheap, keeps the split trivially race-free).
            # ---- L1: Cube GEMM operands (compiler maps alloc_shared -> L1) ----
            K_l1 = T.alloc_shared([topk, dim], dtype)
            Q_l1 = T.alloc_shared([2, PH, dim], dtype)  # double-buffered (pipeline)
            dlog_l1 = T.alloc_shared([PH, topk], dtype)  # GEMM2 A-normal + GEMM3 A-transposed
            # (gqa_sink ds_l1 precedent: one L1 buffer, both read patterns)

            # ---- L0C: GEMM accumulators (alloc_fragment) ----
            logits_l0c = T.alloc_fragment([PH, topk], accum_dtype)
            dq_l0c = T.alloc_fragment([PH, dim], accum_dtype)
            dk_l0c = T.alloc_fragment([topk, dim], accum_dtype)

            # ---- UB: Vector-side buffers (alloc_shared -> UB) ----
            # block-level [T_BLOCK, topk] score / diff chain
            attn_ub = T.alloc_shared([T_BLOCK, topk], accum_dtype)
            is_ub = T.alloc_shared([T_BLOCK, topk], accum_dtype)
            pt_b = T.alloc_shared([T_BLOCK, topk], accum_dtype)  # pt -> diffm reuse
            attnm = T.alloc_shared([T_BLOCK, topk], accum_dtype)
            a_col = T.alloc_shared([T_BLOCK], accum_dtype)
            w_bf16_flat = T.alloc_shared([T_BLOCK * heads], dtype)
            w_f32_flat = T.alloc_shared([T_BLOCK * heads], accum_dtype)  # carries sm_scale
            dw_acc = T.alloc_shared([T_BLOCK * heads], accum_dtype)  # flat (token, head)
            dw_bf16 = T.alloc_shared([T_BLOCK * heads], dtype)
            # sub-tile chain tiles [TS_H, topk] (AIV-vid owns sub-tile 2p+vid)
            col_1d = T.alloc_shared([topk], accum_dtype)  # col_scaled = j*H (or idx*H)
            col_scaled_2d = T.alloc_shared([TS_H, topk], accum_dtype)
            rs_tmp = T.alloc_shared([TS_H], accum_dtype)  # sub-tile row-shift scratch
            row_shift_2d = T.alloc_shared([TS_H, topk], accum_dtype)  # -> dwq reuse
            causal = T.alloc_shared([TS_H, topk], accum_dtype)
            logits_ub = T.alloc_shared([TS_H, topk], dtype)  # transit read -> lscr reuse
            L_f32 = T.alloc_shared([TS_H, topk], accum_dtype)  # grain chain -> dlog reuse
            diff_big = T.alloc_shared([TS_H, topk], accum_dtype)
            w_big = T.alloc_shared([TS_H, topk], accum_dtype)
            dw_vec = T.alloc_shared([TS_H], accum_dtype)
            dlog_bf16_ub = T.alloc_shared([TS_H, topk], dtype)
            # tail-only dq readback row block
            dq_tok_ub = T.alloc_shared([heads, dim], dtype)
            # layout=1-only buffers (unconditional alloc: TVM scoping; never
            # touched when layout == 0)
            indices_i32_ub = T.alloc_shared([topk], indices_dtype)
            indices_f32_ub = T.alloc_shared([topk], accum_dtype)
            gather_ub = T.alloc_shared([topk, dim], dtype)
            dk_row_ub = T.alloc_shared([dim], accum_dtype)

            # ---- loop-invariant setup (once per kernel) ----
            if layout == 0:
                T.tile.arith_progression(col_1d, 0, heads, topk)  # col_scaled[j] = j*H
                T.tile.broadcast(col_scaled_2d, col_1d, axis=0)

            # ---- persistent task loop (runtime bound, sparse_mla:309 form).
            #      CONTIGUOUS assignment (core c gets a consecutive run of
            #      blocks): same 4/3 balance as striped, but consecutive
            #      blocks usually share the segment, so the K panel only
            #      reloads at segment boundaries (t0 == 0) instead of every
            #      block - the L2 read volume for K drops ~3x. ----
            base = num_blocks // grid_size
            rem = num_blocks % grid_size
            start = cid * base + T.min(cid, rem)
            ntask = base + T.if_then_else(cid < rem, 1, 0)
            for k in T.serial(ntask):
                bx = start + k
                bos = BlockIndices[bx, 0]
                t0 = BlockIndices[bx, 1]
                ln = BlockIndices[bx, 2]
                t_max = T.min(t0 + T_BLOCK, ln) - 1

                # ---- K panel load (C side) ----
                if layout == 0:
                    # arange contract: the topk rows are contiguous in-segment;
                    # reload only at chunk start or segment start
                    if (k == 0) | (t0 == 0):
                        T.copy(IndexK[bos : bos + topk, 0:dim], K_l1)
                else:
                    # shared contract: all rows identical; gather with guards,
                    # then UB -> GM -> L1 transit (bf16 UB->L1 is fp16-only).
                    # Both AIVs load the indices (each computes col_scaled in
                    # its own UB) and each gathers half the panel so the ws_k
                    # write handshake carries both sub-blocks.
                    T.copy(TopkIndices[bos, 0:topk], indices_i32_ub)
                    T.copy(indices_i32_ub, indices_f32_ub)
                    T.tile.fill(gather_ub, 0.0)
                    for i in T.serial(topk // 2):
                        gi = vid * (topk // 2) + i
                        pos = indices_i32_ub[gi]
                        if (pos > -1) & (pos <= t_max):
                            T.copy(IndexK[bos + pos, 0:dim], gather_ub[gi, 0:dim])
                    T.copy(
                        gather_ub[(vid * topk) // 2 : ((vid + 1) * topk) // 2, 0:dim],
                        workspace_k[cid, (vid * topk) // 2 : ((vid + 1) * topk) // 2, 0:dim],
                    )
                    T.copy(workspace_k[cid, 0:topk, 0:dim], K_l1)
                    # causal columns from index values: col_scaled[j] = idx[j]*H
                    T.tile.mul(col_1d, indices_f32_ub, heads)
                    T.tile.broadcast(col_scaled_2d, col_1d, axis=0)

                # ---- scores / weights block load (V side, tail-guarded) ----
                if t0 + T_BLOCK <= ln:
                    T.copy(AttnScore[bos + t0 : bos + t0 + T_BLOCK, 0:topk], attn_ub)
                    T.copy(IndexScore[bos + t0 : bos + t0 + T_BLOCK, 0:topk], is_ub)
                    T.copy(
                        Weights[(bos + t0) * heads : (bos + t0 + T_BLOCK) * heads],
                        w_bf16_flat,
                    )
                else:
                    # zero-fill first: invalid-token rows keep diff == 0, which
                    # masks their dq/dw/dk contributions without any per-row
                    # validity vector in the sub-tile chain (NaN-safe too)
                    T.tile.fill(attn_ub, 0.0)
                    T.tile.fill(is_ub, 0.0)
                    T.tile.fill(w_bf16_flat, 0.0)
                    for u in T.serial(T_BLOCK):
                        if t0 + u < ln:
                            T.copy(AttnScore[bos + t0 + u, 0:topk], attn_ub[u, 0:topk])
                            T.copy(IndexScore[bos + t0 + u, 0:topk], is_ub[u, 0:topk])
                            T.copy(Weights[(bos + t0 + u) * heads : (bos + t0 + u + 1) * heads], w_bf16_flat[u * heads : (u + 1) * heads])
                T.copy(w_bf16_flat, w_f32_flat)  # bf16 -> fp32 auto-cast
                T.tile.mul(w_f32_flat, w_f32_flat, sm_scale)  # dlog's s rides on w (dw pairs
                # the unscaled diff with the grain-L that already carries s)

                # ---- clip-exact diffm (block level, V side; docstring note 7) ----
                T.tile.compare(pt_b, is_ub, T.float32(3.7200759760208356e-44), "GT")  # packed pt
                T.tile.select(attnm, pt_b, attn_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")  # attn * pt
                T.reduce_sum(attnm, a_col, dim=1)  # [T_BLOCK] = A' per token
                T.tile.broadcast(attn_ub, a_col, axis=1)  # A'_2d (attn_ub dead -> reuse)
                T.tile.mul(pt_b, is_ub, attn_ub)  # diffm = is * A' (pt_b dead -> reuse)
                T.tile.sub(pt_b, pt_b, attnm)  # diffm = is*A' - attn*pt (UNSCALED)

                # ---- software-pipelined PAIR loop (pair-GEMM, token
                #      order): each iteration bundles two sub-tiles into one
                #      M=2*TS_H GEMM set (24 -> 12 gemm_v0 calls per block).
                #      AIV-vid owns sub-tile 2p+vid's FULL [TS_H, topk] tile
                #      (contiguous rows in every transit - the sparse-flash
                #      both-AIVs-write-disjoint-halves pattern the 1_2 flag
                #      handshake requires). Token order everywhere: the pair's
                #      8 Q rows / 8 dq rows are contiguous GM ranges, so the
                #      loads/stores are single whole-tile copies (the L0C/L1
                #      fractal layouts admit no row slicing). C produces pair
                #      p one iteration ahead of consuming pair p-1; the dlog
                #      pair slots alternate workspace_dlog_a (even pairs) / b
                #      (odd pairs), the logits pair slots use the dim-1 of
                #      workspace_logits. Slot safety mirrors the sub-tile
                #      analysis: C's wait for dlog(p-1) at iteration p
                #      transitively guarantees V finished reading
                #      ws_logits(p-2), and V's wait for f3(p) transitively
                #      guarantees C finished reading dlog(p-2). ----
                for p in T.serial(PAIRS + 1):
                    tp = t0 + p * (2 * TS)  # pair's first token
                    pslot = p % 2
                    if p < PAIRS:
                        # ---- produce(p): Q pair load + GEMM1 + logits transit (C) ----
                        if tp + 2 * TS <= ln:
                            T.copy(
                                IndexQ[(bos + tp) * heads : (bos + tp + 2 * TS) * heads, 0:dim],
                                Q_l1[pslot, 0:PH, 0:dim],
                            )
                        else:
                            # varlen tail: clamp rows to ln-1 (finite, NaN-safe)
                            for u in T.serial(2 * TS):
                                row = bos + T.min(tp + u, ln - 1)
                                T.copy(
                                    IndexQ[row * heads : (row + 1) * heads, 0:dim],
                                    Q_l1[pslot, (u * heads) : ((u + 1) * heads), 0:dim],
                                )
                        T.gemm_v0(Q_l1[pslot, 0:PH, 0:dim], K_l1, logits_l0c, transpose_B=True, init=True)
                        # L0C fp32 -> GM bf16 with relu folded in (gqa_sink:478
                        # class; the bf16 transit also performs grain-round 1)
                        T.copy(logits_l0c, workspace_logits[cid, pslot, 0:PH, 0:topk], enable_relu=True)

                    # ---- V consume(p): AIV-vid runs sub-tile 2p+vid's chain
                    #      on its full [TS_H, topk] tile ----
                    if p < PAIRS:
                        T.copy(
                            workspace_logits[cid, pslot, (vid * TS_H) : ((vid + 1) * TS_H), 0:topk],
                            logits_ub,
                        )
                        # causal[r, j] = (row_token*H + r_in_token >= col_scaled[j]):
                        # one arith/broadcast segment (the sub-tile's tokens
                        # are contiguous); validity rides on the zero-filled
                        # diff rows
                        T.tile.arith_progression(rs_tmp, (tp + vid * TS) * heads, 1, TS_H)
                        T.tile.broadcast(row_shift_2d, rs_tmp, axis=1)
                        T.tile.compare(causal, row_shift_2d, col_scaled_2d, "GE")
                        # d_relu fold (see docstring): grain-L is 0 exactly
                        # where drelu is 0, so folding drelu into causal before
                        # the dwq mul is a no-op for dw and exactly right for
                        # dlog
                        T.copy(logits_ub, L_f32)  # bf16 -> fp32
                        T.tile.compare(row_shift_2d, L_f32, T.float32(0.0), "GT")  # drelu
                        T.tile.bitwise_and(causal, causal, row_shift_2d)  # causal &= drelu
                        # dw-grain L = fp32(bf16(bf16(relu(qk)) * s)) - docstring note 8
                        T.tile.mul(L_f32, L_f32, sm_scale)
                        T.copy(L_f32, logits_ub)  # fp32 -> bf16 (logits_ub dead -> lscr)
                        T.copy(logits_ub, L_f32)  # bf16 -> fp32 (grain L final)
                        # expand diff / w to row space (TS token broadcasts)
                        for u in T.serial(TS):
                            T.tile.broadcast(
                                diff_big[(u * heads) : ((u + 1) * heads), 0:topk],
                                pt_b[(p * 2 * TS + vid * TS + u), 0:topk],
                                axis=0,
                            )
                        T.tile.broadcast(
                            w_big,
                            w_f32_flat[(p * 2 * TS + vid * TS) * heads : (p * 2 * TS + vid * TS + TS) * heads],
                            axis=1,
                        )
                        # NaN guard: mask diff through causal before multiplies
                        T.tile.select(diff_big, causal, diff_big, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                        # dWeights integrand (unscaled diff x grain-L)
                        T.tile.mul(row_shift_2d, diff_big, L_f32)  # dwq (row_shift dead)
                        T.reduce_sum(row_shift_2d, dw_vec, dim=1)
                        # dw_vec maps 1:1 onto the flat (token, head) rows
                        T.copy(
                            dw_vec,
                            dw_acc[(p * 2 * TS + vid * TS) * heads : (p * 2 * TS + vid * TS + TS) * heads],
                        )
                        # d_logits = diff * w(scaled) * (causal & drelu)
                        T.tile.mul(L_f32, diff_big, w_big)  # dlog (L_f32 dead -> reuse)
                        T.copy(L_f32, dlog_bf16_ub)  # fp32 -> bf16
                        # CV transit: dlog UB -> GM workspace -> L1; pair slots
                        # alternate a/b (both AIVs write their sub-tile's
                        # disjoint rows of the same slot)
                        if pslot == 0:
                            T.copy(
                                dlog_bf16_ub,
                                workspace_dlog_a[cid, (vid * TS_H) : ((vid + 1) * TS_H), 0:topk],
                            )
                        else:
                            T.copy(
                                dlog_bf16_ub,
                                workspace_dlog_b[cid, (vid * TS_H) : ((vid + 1) * TS_H), 0:topk],
                            )

                    if p >= 1:
                        # ---- C consume(p-1): dlog -> GEMM2/3 + dq writeback ----
                        tpp = t0 + (p - 1) * (2 * TS)
                        cpslot = (p - 1) % 2
                        if cpslot == 0:
                            T.copy(workspace_dlog_a[cid, 0:PH, 0:topk], dlog_l1)
                        else:
                            T.copy(workspace_dlog_b[cid, 0:PH, 0:topk], dlog_l1)
                        # GEMM2: dq = dlog x K (sm_scale rides on w); the whole
                        # pair is one M=PH gemm and the 8 token rows are one
                        # contiguous GM range (token order)
                        T.gemm_v0(dlog_l1, K_l1, dq_l0c, init=True)
                        if tpp + 2 * TS <= ln:
                            # whole pair: one direct L0C -> GM write
                            T.copy(
                                dq_l0c,
                                dIndexQ[(bos + tpp) * heads : (bos + tpp + 2 * TS) * heads, 0:dim],
                            )
                        else:
                            # tail pair: transit through ws_dq, V writes valid rows
                            T.copy(dq_l0c, workspace_dq[cid, 0:PH, 0:dim])
                        # GEMM3: dk += dlog^T x Q_pair (fp32 on-chip; the same
                        # dlog_l1 region is re-read in the transposed fractal
                        # layout - gqa_sink ds_l1 precedent)
                        T.gemm_v0(
                            dlog_l1,
                            Q_l1[cpslot, 0:PH, 0:dim],
                            dk_l0c,
                            transpose_A=True,
                            init=((k == 0) | (t0 == 0)) & (p == 1),
                        )

                    # ---- V tail writeback for dq: AIV-vid writes its own
                    #      sub-tile's valid tokens of the tail pair ----
                    if p >= 1:
                        tpp_v = t0 + (p - 1) * (2 * TS)
                        if tpp_v + 2 * TS > ln:
                            for u in T.serial(TS):
                                tu = vid * TS + u
                                if tpp_v + tu < ln:
                                    T.copy(
                                        workspace_dq[cid, (tu * heads) : ((tu + 1) * heads), 0:dim],
                                        dq_tok_ub,
                                    )
                                    T.copy(
                                        dq_tok_ub,
                                        dIndexQ[(bos + tpp_v + tu) * heads : (bos + tpp_v + tu + 1) * heads, 0:dim],
                                    )

                # ---- block tail: dWeights write - each AIV writes its own
                #      sub-tile rows of every pair (dw_acc's other-sub-tile
                #      rows were never produced in this AIV's UB) ----
                for p in T.serial(PAIRS):
                    tpw = t0 + p * (2 * TS)
                    if tpw + 2 * TS <= ln:
                        T.copy(
                            dw_acc[(p * 2 * TS + vid * TS) * heads : (p * 2 * TS + vid * TS + TS) * heads],
                            dw_bf16[(p * 2 * TS + vid * TS) * heads : (p * 2 * TS + vid * TS + TS) * heads],
                        )
                        T.copy(
                            dw_bf16[(p * 2 * TS + vid * TS) * heads : (p * 2 * TS + vid * TS + TS) * heads],
                            dWeights[(bos + tpw + vid * TS) * heads : (bos + tpw + vid * TS + TS) * heads],
                        )
                    else:
                        for u in T.serial(TS):
                            tu = vid * TS + u
                            if tpw + tu < ln:
                                T.copy(
                                    dw_acc[((p * 2 * TS) + tu) * heads : ((p * 2 * TS) + tu + 1) * heads],
                                    dw_bf16[((p * 2 * TS) + tu) * heads : ((p * 2 * TS) + tu + 1) * heads],
                                )
                                T.copy(
                                    dw_bf16[((p * 2 * TS) + tu) * heads : ((p * 2 * TS) + tu + 1) * heads],
                                    dWeights[(bos + tpw + tu) * heads : (bos + tpw + tu + 1) * heads],
                                )
                # ---- dk fp32 atomic (C side): dk_l0c accumulates across
                #      the core's contiguous same-segment blocks (contiguous
                #      task assignment) and flushes once per segment run -
                #      65 -> ~22 atomics, less L2 line contention ----
                if layout == 0:
                    if (k == ntask - 1) | (t0 + T_BLOCK >= ln):
                        # whole-tile fp32 region atomic (gqa_sink:437 form)
                        T.tile.atomic_add(ws_dk[bos : bos + topk, 0:dim], dk_l0c)
                else:
                    # per-row atomic via a one-row fp32 UB buffer: whole-fragment
                    # L0C -> GM ws, then row-by-row scatter split across both
                    # AIVs (each handles half the rows -> the ws_dkf read
                    # handshake carries both sub-blocks). pos re-read from
                    # TopkIndices GM (atomic rides the Cube/FIX pipe).
                    if (k == ntask - 1) | (t0 + T_BLOCK >= ln):
                        # stage + scatter only on flush blocks (the ws_dkf
                        # handshake must stay set/wait matched)
                        T.copy(dk_l0c, workspace_dkf[cid, 0:topk, 0:dim])
                        for i in T.serial(topk // 2):
                            gi = vid * (topk // 2) + i
                            pos = TopkIndices[bos, gi]
                            if (pos > -1) & (pos <= t_max):
                                T.copy(workspace_dkf[cid, gi, 0:dim], dk_row_ub)
                                T.tile.atomic_add(ws_dk[bos + pos, 0], dk_row_ub)

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
    in ~13 us instead of ~750 us (see ``_CachedLauncher``). The kernel declares
    IndexQ/dIndexQ as flat [S*H, D] and Weights as flat [S*H] — the launcher
    applies the matching zero-copy views to raw [S, H, D] / [S, H] arguments.
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
    if not hasattr(kernel, "adapter"):
        return kernel
    # param order: IndexQ, Weights, IndexK, AttnScore, IndexScore, TopkIndices,
    # BlockIndices, ws_logits, ws_dlog_a, ws_dlog_b, ws_dq, ws_k, ws_dkf,
    # ws_dk, dIndexQ, dWeights
    return _CachedLauncher(kernel, view_spec={0: "flat2d", 1: "flat1d", 14: "flat2d", 15: "flat1d"})


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
def _tl_dk_cast_impl(dim, grid_size=20, cast_rows=64):
    """Build the trailing cast kernel: ws_dk [seq_len, dim] fp32 -> dIndexK bf16.

    Runs after the main kernel on the same stream (host program order). Uses
    the MIX_AIC_1_2 (cid, vid) form: both AIV sub-blocks of a block cast
    disjoint halves of each chunk's rows.
    """

    seq_len = T.symbolic("seq_len")

    @T.prim_func
    def main(
        ws_dk: T.Tensor([seq_len, dim], "float32"),  # type: ignore
        dIndexK: T.Tensor([seq_len, dim], "bfloat16"),  # type: ignore
    ):
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):
            ub = T.alloc_shared([cast_rows // 2, dim], "float32")
            ub_bf16 = T.alloc_shared([cast_rows // 2, dim], "bfloat16")
            half = cast_rows // 2
            total = T.ceildiv(seq_len, cast_rows)
            ntask = T.max(0, T.ceildiv(total - cid, grid_size))
            for t in T.serial(ntask):
                c = t * grid_size + cid
                row0 = c * cast_rows
                if row0 + cast_rows <= seq_len:
                    # whole chunk: each AIV handles its half of the rows
                    T.copy(ws_dk[row0 + vid * half : row0 + (vid + 1) * half, 0:dim], ub)
                    T.copy(ub, ub_bf16)  # fp32 -> bf16 auto-cast
                    T.copy(
                        ub_bf16,
                        dIndexK[row0 + vid * half : row0 + (vid + 1) * half, 0:dim],
                    )
                else:
                    for u in T.serial(half):
                        ru = vid * half + u
                        if row0 + ru < seq_len:
                            T.copy(ws_dk[row0 + ru, 0:dim], ub[u, 0:dim])
                            T.copy(ub[u, 0:dim], ub_bf16[u, 0:dim])
                            T.copy(ub_bf16[u, 0:dim], dIndexK[row0 + ru, 0:dim])

    return main


def tl_dk_cast(dim, grid_size=20, cast_rows=64):
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
    The UB mirror approximates the rev3 buffer set (see tl_indexer_bwd_fast:
    6 fp32 [TS_H, topk] big tiles + 2 bf16 [TS_H, topk] + 4 fp32 [T_BLOCK,
    topk] score tiles + flat weights/dw + layout=1 gather/row buffers) with
    a 24KB planner-scratch allowance.
    """
    if topk not in (16, 32, 64):
        return False
    if heads % 16 != 0 or heads > 64:
        return False
    if dim % 16 != 0:
        return False
    ts_h = _FAST_TS * heads
    tb = _FAST_T_BLOCK
    # pair-GEMM L0C: pair tiles are M=2*TS_H (logits + dq) plus dk [topk, dim]
    if 2 * ts_h * (topk + dim) * 4 + topk * dim * 4 > 128 * 1024:
        return False
    ub_bytes = (
        6 * ts_h * topk * 4            # fp32 big tiles
        + 2 * ts_h * topk * 2          # bf16 transit/cast tiles
        + 4 * tb * topk * 4            # score chain tiles
        + tb * heads * (2 + 4 + 2 + 2)  # flat w + dw (bf16/fp32)
        + topk * dim * 2               # layout=1 gather_ub
        + dim * 4                      # layout=1 dk_row_ub
        + 4 * (ts_h + topk) * 4        # 1D vectors
        + 8 * 1024                     # planner / tmp scratch allowance
    )
    return ub_bytes < 196352


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
    workspace_logits = torch.empty(grid_size, 2, 2 * ts_h, topk, dtype=torch.bfloat16, device=device)
    workspace_dlog_a = torch.empty(grid_size, 2 * ts_h, topk, dtype=torch.bfloat16, device=device)
    workspace_dlog_b = torch.empty(grid_size, 2 * ts_h, topk, dtype=torch.bfloat16, device=device)
    workspace_dq = torch.empty(grid_size, 2 * ts_h, dim, dtype=torch.bfloat16, device=device)
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
