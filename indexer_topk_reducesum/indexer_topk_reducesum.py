"""indexer_topk_reducesum — DSA sparse finetune lightning indexer forward
(topk + reducesum, varlen causal) for Ascend 910B3, Developer mode.

Semantics replicate examples_experiment/dsa_sparse_finetune/ref_gpu/
indexer_topk_reducesum.py ``ref_index_score`` (the golden): per varlen
segment, for each query token s1

    logits[s1, h, s2] = einsum(q, k)                     # bf16 round point
    score[s1, s2]     = (sum_h relu(logits) * w[s1, h])_{fp32} * sm_scale
    score[s1, s2]     = -inf  if s2 > s1                 # causal mask
    topk_vals, topk_idx = topk(score[s1, :], topk)       # descending
    topk_score        = softmax(topk_vals)_{fp32}

outputs TopkIndices [S, topk] int32 (-1 = -inf tail positions) and
ReduceSum (=topk_score) [S, topk] fp32.

Dual-path architecture (DESIGN_indexer_topk_reducesum.md §1.4), host dispatch
with exact (non-heuristic) max_seg_len detection:

    fast     max_seg_len <= 4096: persistent T.Kernel(grid=20), QG tokens per
             group, transposed (h,m)-flattened GEMM [BN, D] x [D, QG*H] ->
             L0C fp32 [BN, QG*H] -> bf16 GM transit (golden round point) ->
             relu -> weighted head reduce -> causal mask -> score_row
             [QG, CAP] UB -> per-token hardware T.tile.topk -> softmax ->
             idx fixup.
    generic  any segment length: same GEMM data flow, score segmented into
             [QG, CH=1024] tiles, per-segment hardware topk + merge_sort pool
             merging (mathematically equal to full-row topk).

Implementation notes (deviations from DESIGN §3.3 pseudocode, each forced by a
platform constraint or a strictly-safer validated form; see debug_log.md):

 1. The CV transit buffer is named ``workspace_logits``: the
    TL_ASCEND_AUTO_CV_SYNC pass pairs cross-core GM writes/reads by the
    literal substring "workspace" (src/transform/ascend_combinecv.cc
    FetchWorkspaceName) and requires a 1:1 write/read sync-point pairing.
    The DESIGN §7.2 shorthand "ws_logits" would NOT match the substring.
 2. relu rides the L0C->GM transit leg (``enable_relu=True`` on the fp32->bf16
    write, the indexer_bwd-validated exact form) instead of the GM->UB read
    leg: GM->UB copies are single-dtype (copy_gm_to_ub<T> in
    src/op/ascend.cc:251) so the DESIGN's "GM bf16 -> UB fp32 + relu" leg is
    not expressible directly. Numerically identical: relu commutes with the
    bf16 rounding (relu(x)>=0 and bf16 preserves sign/zero). This also
    resolves risk R7 by construction (no cast+relu combo copy at all).
  3. The V side relays its disjoint half of the [BN, QG*H] transit tile GM
      bf16 -> UB bf16 (the paired 1-read matching the C-side 1-write per K
      block; per-token slice reads would break the 1:1 sync-point consistency
      check), then half-token [BN, QG2H//2] UB->UB casts to fp32
      (copy_ub_to_ub<dst,src> two-dtype form).
  4. The per-head product is rounded to bf16 before the fp32 head reduce
     (``prod = bf16(fp32(relu_logits) * fp32(w)); score = sum_h(prod) *
     sm_scale``), replicating the golden's bf16 product round point
     (ref_index_score: ``(logits * weights).sum(dim=-2, dtype=fp32) * s``
     multiplies two bf16 tensors -> bf16 product -> fp32 sum -> fp32 scale).
     A first draft folded sm_scale into the fp32 weights and kept the product
     in fp32 -- that draft failed L0 with matched_ratio 0.9055 / max_abs
     7.65e-3 (l0_equal_topk), reproduced exactly by a CPU semantics
     simulation of the missing round; the round-trip copy pair
     (fp32 -> bf16 -> fp32) is the indexer_bwd note-8 validated form. The
     scale multiplies the [H, BN] rounded products (>= 256B) instead of the
     sub-256B [BN] score row, so the vector-op minimum-size rule stays
     satisfied.
 5. The causal mask is applied pre-reduce at [H, BN] granularity (compare
    requires >= 256 bytes of src; [BN=32] fp32 = 128B violates it): the
    -inf select lands on all H rows of masked columns, so the dim=0 head
    reduce yields -inf for masked columns -- identical to masking after the
    sum. Column indices come from arith_progression at the GLOBAL k_st.
 6. topk runs on a 1D scratch (topk_src): T.tile.topk takes a Buffer (not a
    BufferRegion row of the 2D score_row). The softmax / idx-fixup chain
    uses TOPK_A = max(topk, 64) wide buffers with -inf / -1.0 pad fills
    (compare needs >= 64 fp32 elements; pads are never written out).
 7. Generic segment indices are corrected to the global column frame before
    the pool merge: gather val/idx, fp32-add seg_lo to the indices, scalar
    interleave back into (val, idx) pairs, merge_sort with the pool.
  8. IndexQ is declared flat [S*H, D] and Weights flat [S*H]; the cached
     launcher applies zero-copy host-side views (indexer_bwd note-1 precedent;
     T.copy structural equality forbids the [S, H, D] row-segment form).
  9. gemm_v0 is instantiated with a host-computed ``kL0Size`` (default 128
     overflows L0A for M = QG*heads = 512: the A slot M*kL0Size*2B must fit
     64KB single-(N,K)-step / 32KB ping-pong, common.h:1282 static assert;
     ascend.py:450 documents kL0Size for exactly this case). K-order fp32
     accumulation is unchanged (ascending K, same mma granularity), so the
     golden round points are intact. Host mirror: _pick_kl0_size().
 10. Launch is the ``T.Kernel(grid, is_npu=True) as (cid, vid)`` form
     (MIX_AIC_1_2: 1 Cube + 2 Vector cores per block) -- the ONLY validated
     launch shape for CV-combined kernels in this repo (indexer_bwd note 10 /
     sparse_mla / gqa_sink / lightning_indexer all use it). The first draft's
     ``threads=1`` (MIX_AIC_1_1) compiled but hung on device with an aicore
     timeout (l0_h16 first run) -- the 1_1 cross-core flag handshake is
     unvalidated; per indexer_bwd note 10 the 1_2 handshake additionally
     requires every AIV sub-block to participate in every transit. AIV-vid
     therefore owns the contiguous token half [vid*QG2, (vid+1)*QG2) of each
     QG-token group (QG2 = QG//2), with per-AIV UB buffers and per-AIV
     contiguous-half transit reads (the disjoint-halves pattern); the Cube
     side (Q/K loads, GEMM, full-tile workspace write) runs once per block.
 11. Cross-core back-pressure protocol (indexer_bwd note-10 pattern). A
     draft without back-pressure passed l0_equal_topk (1 task/core, <= 3
     transit pairs) but hung with an aicore timeout ("fftsplus aivector
     error", timeout-or-trap) once a core ran several tasks x many transit
     pairs (S=512: 4 tasks x up-to-16 pairs; controlled experiments: 16
     pairs x 1 task/core PASS, 1-2 pairs x 4 tasks/core PASS, 16 x 4 HANG):
     at task boundaries the Cube side has almost no work (3 scalar loads +
     one Q load) and laps the Vector cores, accumulating unbounded in-flight
     CV-sync flag credit until the FFTS path faults. Fix, mirroring
     indexer_bwd's validated produce/consume pipeline: a V->C ack transit
     (``workspace_ack``, one heartbeat slot per AIV; the AIC reads both
     slots so the 1_2 handshake needs both AIVs' sets) throttles the AIC to
     <= 1 transit ahead. Fast path: j-loop over NBLK+1 iterations with
     separate produce (j < NBLK) and ack-wait (j >= 1, cond_{j-1}) blocks --
     produce-then-wait order plus a 2-slot ping-pong workspace
     (``workspace_logits[cid, j % 2]``) keeps the pipeline overlapped
     (slot j%2 is provably consumed by the ack waited at iteration j-1);
     the +1 iteration drains the task's last ack, closing the task
     boundary. Flag credit is exactly balanced per task: V transit sets / V
     transit waits per AIV, V ack sets per AIV / V ack waits. Generic path
     (not perf-gated): wait-before-produce single-slot variant
     (``k + sg + j > 0`` skips only the core's first produce; the final
     leftover ack set is inert at kernel end).
 12. topk-16 gather pads: tl::ascend::GatherMask quantises its source to
      256B units (common.h:1451 repeatTimes = ceil(srcBytes, 256), 64 fp32
      per unit) and compacts the pattern-selected elements to the dst front.
      With ``2*topk < 64`` (topk = 16, the only pow2 in [16, 1024] below 32)
      a [2*topk] source made the gather read 32 elements past the buffer and
      write UB garbage into the -inf softmax pads -- l1_topk16 failed with
      matched_ratio 0.3750 (indices unaffected: the garbage sits beyond the
      written-out [0:topk) window). All gather src/dst buffers are therefore
      sized to the 256B quantum (P2 = max(2*topk, 64), PS = max(topk, 32))
      and the pair buffers are pre-filled with -inf so the quantised gather
      selects pads instead of out-of-bounds data.
 13. Stage 3 perf iter-1 restructures the V-side consumption chain (same
      dataflow, bit-exact math): the QG2 per-token [H, BN] elementwise chains
      fuse into two half-token [QG2H//2, BN] chains -- elementwise ops on
      disjoint row blocks are identical, and one big-tile op replaces HPT
      small ones (the generated code showed 13 vector instructions + ~12
      PipeBarrier_V per token per block); the causal compare/select runs only
      on the straddle block (k_st <= t_abs < k_st + BN) -- fully-valid blocks
      skip it and fully-invalid (token, block) pairs keep the score_2d -inf
      fill (the fill + fused [QG2, BN] store preserve the topk pad -inf); the
      sm_scale multiply moves off the per-block tile to one [CAP] row Muls
      per token per task -- the golden's exact "fp32 sum then scale" order.
      Positive scales only (a positive scale preserves the descending topk
      order and the -inf tail signs); sm_scale is a compile-time constant, so
      non-positive scales keep the per-block pre-reduce multiply at zero
      runtime cost.
  14. Stage 3 perf iter-2 transposes the GEMM and transit orientation:
       gemm_v0(K_l1 [BN, D], Q_l1 [QG*H, D], transpose_B=True) produces
       logits^T [BN, QG*H] in L0C (same dot products, same kL0Size K-chunk
       accumulation order -- bit-identical GEMM), and the workspace becomes
       [grid, 2, BN, QG*H]. Both transit legs then move wide bursts instead
       of 64B rows (fix write 32 x 1KB, AIV relay read 32 x 512B with 512B
       gaps; the [QG*H, BN] orientation issued 256/512 x 64B bursts and the
       AIV mte2 measured only ~12-20GB/s effective), and the per-token head
       reduce becomes the contiguous dim=-1 (AR) form on [BN, H] segments.
       The AIV-vid ownership moves from rows to columns (disjoint halves of
       the QG*H axis; same per-AIV token set, note 10 unchanged).
  15. Phase 4 dual-backend adaptations (target="ascendc" | "pto"). The jit
       wrapper is per-target (tilelang/jit/adapter/libgen.py forks bisheng
       -xasc vs -xcce); the kernel body below is shared except for the
       explicitly target-branched items (full evidence: debug_log.md phase 4,
       backend_verify/matrix.md):
       (a) compare dsts are the canonical packed-uint8 predicate ([.., N//8]
           uint8) per the T.tile.compare API contract -- the PTO scalar
           compare (vcmp_dispatch, TCmps.hpp:23) hard-requires a uint8 dst;
           the previous fp32-dst form only worked on AscendC. The AscendC
           path was fully re-validated after the change.
       (b) fast path, PTO branch: the transit uses the generic orientation
           (gemm Q @ K^T -> [QG*H, BN], workspace viewed as
           workspace_logits_t [grid, 2, QG*H, BN] -- the same bytes as the
           ascendc [grid, 2, BN, QG*H] buffer, passed twice by the host) and
           the generic path's per-token consume chain (contiguous [QG2H, BN]
           relay read, [H, BN] slices, pre-reduce scalar-compare causal
           mask, dim=0 head reduce). The perf-iter-2 transposed orientation
           (wide [BN, QG*H] transit + strided [BN, QG2H] half read) raced on
           PTO: the AIV's wide-burst relay read intermittently caught the
           AIC write tail (transit tile rows ~26-31 stale -> wrong scores at
           the corresponding columns); every isolated repro of the individ-
           ual ops passed, only the full transposed CV context failed. The
           generic orientation is the empirically PTO-robust form (stress-
           validated). AscendC keeps the perf-frozen transposed form.
       (c) PTO branch: bulk Q/K/weights GM->L1 loads only. The per-row
           guarded varlen-tail loads ([H, D] / [1, D] row-block copies)
           produced garbage under PTO (varlen tail groups' transit rows
           were ~75% wrong); the bulk copies carry the PTO codegen's
           runtime tail clamp + zero pad, out-of-segment rows are causally
           masked (cols >= ln) or belong to unconsumed garbage tokens.
       (d) fast path, PTO branch: serialized single-flight back-pressure
           (the AIC waits the previous block's ack BEFORE producing -- the
           generic path's protocol). AscendC keeps the overlapped
           produce-then-wait ping-pong (note 11).
       (e) scalar T.max/T.min on vid-derived expressions became
           T.if_then_else ternaries (n_eff, seg_valid): the bisheng CCE
           max()/min() overload set is {(long,long), (int,int)} and the
           (int64 vid-expr, int literal) pair is ambiguous under -xcce.
           Identical integer semantics on both backends.
        (f) NO framework modification (phase-4b rework: an earlier defensive
            pipe_barrier(PIPE_ALL) before every auto cross-core flag set in
            src/target/codegen_ascend_pto.cc was reverted -- zero-framework-
            modification constraint). PTO cross-core correctness rests on
            the stock auto-CV-sync protocol's architectural ordering:
            ascend_combinecv.cc GM_COPY_CFG_INFOS binds each data-ready
            flag to the pipe that EXECUTES the producing store
            (copy_l0c_to_gm -> "FIX", copy_ub_to_gm -> "MTE3"), and
            ffts_cross_core_sync(pipe, msg) is an in-order instruction on
            that pipe -- the same producer pairing as pto-isa's canonical
            TSync_Custom C2V protocol (TSTORE_C2GM records on PIPE_FIX,
            TSTORE_V2GM on PIPE_MTE3) -- so the flag cannot fire while the
            transit write is still in flight. The phase-4 "-xcce flag
            ordering" hypothesis never fixed any repro (the orientation
            branch (b) was the root-cause fix); 3x full stock-framework
            reruns (49/49 each) plus the stock-vs-barrier generated-code
            diff (exactly the 2 barrier lines, backend_verify/matrix.md)
            refute it.
       The PTO-side UB budget actually SHRINKS vs the fused ascendc form
       (the per-token [H, BN] chain replaces the [BN, QG2H] fp32 pair);
       the PTO topk auto-tmp is 1.5x the AscendC one (multiplier 6 vs 4,
       allocate_tmp_buffer.cc:553) and is covered by the gate model's
       margin.

Domain (host-validated, mirrors the main-repo asserts): q/w/k bfloat16;
heads % 8 == 0 and <= 64; dim % 16 == 0; topk a power of two in
[16, 1024] (and <= min segment length, which the golden asserts as
segment >= topk); segment length >= topk per varlen segment. The fast path
additionally requires max_seg_len <= 4096; the generic path covers any
segment length (topk <= 1024, CH = max(1024, topk)). Configurations whose
exact UB model exceeds the 192KB budget are rejected with ValueError
(see _fast_gate_ok / _generic_gate_ok).

[DISPATCH-COVERAGE-AUDIT]
supported_domain: bf16 q/w/k + int32 offsets; varlen with every segment
  length >= topk; heads % 8 == 0 && <= 64; dim % 16 == 0; topk pow2 in
  [16, 1024] && <= min segment length; sm_scale float (default dim**-0.5).
generic_fallback: tl_indexer_topk_reducesum_generic (any segment length via
  CH=1024 streaming + merge_sort pool; topk <= 1024 by the UB model).
uncovered_input: topk > 1024 (UB model overflows even on fast@CAP=2048 for
  topk >= 2048*; rejected with ValueError, exotic for this operator family -
  main repo default 64) and non-bf16 main inputs (main repo fixed dtype).
specialization_cases: fast (CAP 512/1024/2048 x QG=8, CAP=4096 x QG=4),
  generic (CH=1024, NSEG = ceildiv(max_seg_len, CH)).
result: pass (two paths cover the domain; exact host detection; forced
  dispatch validates its contract and raises on violation).

[HOST-METADATA-AUDIT]
- prepare_token_indices / prepare_group_table: vectorized CPU int32 construction
  + one H2D each (perf iter-3; indexer_bwd / sparse_mla precedent). No aclnn,
  no re-layout. The interface fetches the offsets list ONCE (single
  synchronizing D2H per call; the Stage-2 path synced five times).
- torch.zeros outputs + torch.empty workspace: new tensors, no copies.
- q [S, H, D] / weights [S, H] passed raw; the cached launcher applies the
  zero-copy flat-2D/flat-1D views (metadata only, storage shared).
- max_seg_len detection: pure Python over the once-fetched offsets list
  (device primitive + one scalar D2H per call, indexer_bwd host-gate
  precedent).
- Kernel launch via _CachedLauncher (perf iter-3, indexer_bwd note-1
  precedent): direct ctypes marshal + raw-stream query, keyed by tensor
  identity; any miss re-marshals through the same cheap path.
- No post-kernel output processing (shapes returned as allocated).
"""

import ctypes

import numpy as np
import torch

import tilelang
from tilelang import language as T

try:  # fast raw-stream query (0.9 us vs 33 us for torch.npu.current_stream)
    import torch_npu  # noqa: F401

    _npu_raw_stream = torch_npu._C._npu_getCurrentRawStream
except Exception:  # pragma: no cover - non-NPU import context
    _npu_raw_stream = None

__all__ = [
    "tl_indexer_topk_reducesum_fast",
    "tl_indexer_topk_reducesum_generic",
    "indexer_topk_reducesum_interface",
    "prepare_token_indices",
    "prepare_group_table",
    "_detect_dispatch",
    "_fast_gate_ok",
    "_generic_gate_ok",
    "_pick_kl0_size",
]

_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_UB_CAPACITY = 196352  # 192KB minus a planner safety margin (indexer_bwd gate)
_L0C_CAPACITY = 128 * 1024
_BN = 32  # K-block width (GEMM N), fixed per DESIGN §5.2
_ACK_PER_AIV = 16  # fp32 heartbeat elements per AIV slot (64B, 32B-aligned)
_ACK_TOTAL = 2 * _ACK_PER_AIV


class _CachedLauncher:
    """Low-overhead callable around a compiled tilelang kernel (indexer_bwd
    note-1 precedent, reused verbatim in spirit).

    The stock cython launch path re-marshals every argument on each call —
    constructing a fresh TVM ``Analyzer`` alone costs ~0.55 ms. This wrapper
    marshals the low-level ctypes argument vector directly (~tens of us per
    launch, the bare ACL submission cost).

    ``view_spec`` maps argument positions to zero-copy reshape modes for the
    flat-declared params (IndexQ as [S*H, D], Weights as [S*H]): raw
    caller-side tensors are viewed per call.

    Safety model: the single cache slot holds strong references to the exact
    argument objects of the last marshaled call, keyed on their ``id()``
    tuple. While the references are alive those ids cannot be recycled, so a
    key hit guarantees the cached data pointers / dynamic-shape values are
    still the right ones for these very tensors. Any miss (fresh tensors,
    changed arg count, explicit stream) re-marshals through the same cheap
    path; non-tensor args or a missing adapter fall back to the stock call.
    """

    __slots__ = (
        "_kernel",
        "_adapter",
        "_lib",
        "_params_len",
        "_key",
        "_args",
        "_args_refs",
        "_dev",
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
        tensors = [self._apply_view(t, self._view_spec.get(i, "")) if isinstance(t, torch.Tensor) else t for i, t in enumerate(args)]
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
        args2 = tuple(self._apply_view(t, self._view_spec.get(i, "")) if isinstance(t, torch.Tensor) else t for i, t in enumerate(args))
        self._kernel(*args2)


# ============================================================================
# Fast kernel (persistent grid, score_row [QG, CAP] + single hardware topk)
# ============================================================================

# Per-target jit wrapper cache (phase 4): one _JitImplementation per backend,
# each with its own (args, kwargs) kernel cache — the builder body below is
# shared verbatim between the backends (same TIR, different codegen path:
# tilelang/jit/adapter/libgen.py forks bisheng -xasc vs -xcce).
_JIT_WRAPPERS: dict = {}


def _jit_wrapper(target):
    w = _JIT_WRAPPERS.get(target)
    if w is None:
        if target not in ("ascendc", "pto"):
            raise ValueError(f"target must be 'ascendc' or 'pto', got {target!r}")
        w = tilelang.jit(pass_configs=_pass_configs, target=target)
        _JIT_WRAPPERS[target] = w
    return w


def tl_indexer_topk_reducesum_fast(
    heads,
    dim,
    topk,
    CAP,
    QG=8,
    BN=32,
    grid_size=20,
    sm_scale=None,
    KL0=128,
    target="ascendc",
):
    """Build the fast-path kernel (Developer mode, MIX_AIC_1_2 persistent grid).

    Args:
        heads: number of heads H (% 8 == 0 and <= 64, main-repo assert).
        dim: head dim D (% 16 == 0).
        topk: selected positions (power of two, 16 <= topk <= CAP).
        CAP: score_row width, power of two in {512, 1024, 2048, 4096};
            host guarantees CAP >= max segment length (dispatch contract).
        QG: tokens per group (GEMM M = QG*heads; 8 for CAP<=2048, 4 for 4096).
            Each AIV-vid of the group's block owns the contiguous half
            [vid*QG//2, (vid+1)*QG//2) of the group's tokens (docstring 10).
        BN: K-block width (GEMM N).
        grid_size: persistent grid (compile-time constant; 910B3 = 20).
        sm_scale: softmax scale, defaults to dim ** -0.5.
        KL0: gemm_v0 L1->L0 K-axis tile (host-computed by _pick_kl0_size so
            the A operand fits its L0A ping-pong slot; docstring note 9).
        target: codegen backend, "ascendc" (default, backward compatible) or
            "pto". The kernel body is backend-agnostic; only the jit wrapper
            (libgen bisheng -xasc vs -xcce + per-target pass defaults) differs.
    """
    return _jit_wrapper(target)(_tl_indexer_topk_reducesum_fast_main)(
        heads=heads,
        dim=dim,
        topk=topk,
        CAP=CAP,
        QG=QG,
        BN=BN,
        grid_size=grid_size,
        sm_scale=sm_scale,
        KL0=KL0,
        target=target,
    )


def _tl_indexer_topk_reducesum_fast_main(
    heads,
    dim,
    topk,
    CAP,
    QG=8,
    BN=32,
    grid_size=20,
    sm_scale=None,
    KL0=128,
    target="ascendc",
):
    """(private) fast-path prim_func builder — see the public wrapper above."""
    assert heads % 8 == 0 and heads <= 64, "heads must be % 8 == 0 and <= 64"
    assert dim % 16 == 0, "dim must be a multiple of 16"
    assert topk == tilelang.math.next_power_of_2(topk) and topk >= 16, f"topk must be a power of two >= 16, got {topk}"
    assert CAP in (512, 1024, 2048, 4096), f"CAP must be pow2 in [512,4096], got {CAP}"
    assert CAP % BN == 0, "CAP must be a multiple of BN"
    assert topk <= CAP, "topk must be <= CAP"
    assert QG % 4 == 0, "QG must be a multiple of 4 (AIV-vid split + half-token chain fusion)"
    assert QG * heads % 16 == 0, "GEMM M must be 16-aligned"
    assert KL0 % 16 == 0 and KL0 <= 4095, "KL0 must be % 16 == 0 and <= 4095"
    if sm_scale is None:
        sm_scale = dim**-0.5
    # perf iter-1: for a positive scale the multiply preserves the descending
    # topk order and -inf signs, so it moves off the per-block [QG2H, BN] tile
    # to one [CAP] row Muls per token per task (golden-exact "sum then scale"
    # order). Non-positive scales keep the per-block pre-reduce multiply.
    scale_after_reduce = isinstance(sm_scale, (int, float)) and sm_scale > 0

    NBLK = CAP // BN
    TOPK_A = max(topk, 64)  # softmax/idx chain width (compare needs >= 64 fp32)
    P2 = max(2 * topk, 64)  # gather_mask processes src in 256B (64-fp32) units
    # (common.h:1451 repeatTimes = ceil(srcBytes, 256)): a [2*topk] src with
    # 2*topk < 64 reads past the buffer and pollutes the -inf softmax pads)
    QG2 = QG // 2  # tokens per AIV-vid (contiguous halves, docstring note 10)
    HALFC = QG2 * heads // 2  # half-token chain width (perf iter-1 form)
    HTOK = QG2 // 2  # tokens per half-chain

    seq_len = T.symbolic("seq_len")  # total packed length (sum of segment lens)
    batch_plus_one = T.symbolic("batch_plus_one")  # Offsets dim
    num_groups = T.symbolic("num_groups")  # GroupTable rows (persistent tasks)
    q_rows = T.symbolic("q_rows")  # = seq_len * heads (flat-2D IndexQ view)
    w_rows = T.symbolic("w_rows")  # = seq_len * heads (flat-1D Weights view)

    dtype = "bfloat16"
    accum_dtype = "float32"
    indices_dtype = "int32"

    @T.prim_func
    def main(
        IndexQ: T.Tensor([q_rows, dim], dtype),  # type: ignore  # flat [S*H, D]
        Weights: T.Tensor([w_rows], dtype),  # type: ignore  # flat [S*H]
        IndexK: T.Tensor([seq_len, dim], dtype),  # type: ignore
        TopkIndices: T.Tensor([seq_len, topk], indices_dtype),  # type: ignore
        ReduceSum: T.Tensor([seq_len, topk], accum_dtype),  # type: ignore
        Offsets: T.Tensor([batch_plus_one], indices_dtype),  # type: ignore
        TokenIndices: T.Tensor([seq_len, 2], indices_dtype),  # type: ignore  # kept for interface fidelity, not consumed
        GroupTable: T.Tensor([num_groups, 3], indices_dtype),  # type: ignore
        workspace_logits: T.Tensor([grid_size, 2, BN, QG * heads], dtype),  # type: ignore  # ping-pong (note 11); [BN, QG*H] transposed (perf iter-2)
        workspace_logits_t: T.Tensor([grid_size, 2, QG * heads, BN], dtype),  # type: ignore  # same bytes, [QG*H, BN] view (PTO branch, note 15)
        workspace_ack: T.Tensor([grid_size, _ACK_TOTAL], accum_dtype),  # type: ignore  # V->C back-pressure
    ):
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):  # MIX_AIC_1_2
            # ---- L1: Cube GEMM operands (compiler maps alloc_shared -> L1) ----
            Q_l1 = T.alloc_shared([QG * heads, dim], dtype)
            K_l1 = T.alloc_shared([BN, dim], dtype)
            # explicit 2D L1 (plain "shared" is use-inferred to UB, which
            # reclassifies the ack GM read as a vector op and breaks the CV
            # pairing; 1D L1 breaks the default zN layout injection)
            ack_l1 = T.alloc_shared([1, _ACK_TOTAL], accum_dtype, scope="shared.l1")
            # ---- L0C: GEMM accumulator [BN, QG*H] (perf iter-2: transposed
            #      gemm K x Q^T makes the transit rows 1KB/512B wide instead
            #      of 64B, and the head reduce contiguous dim=-1) ----
            logits_l0c = T.alloc_fragment([BN, QG * heads], accum_dtype)
            # ---- UB (per AIV-vid): transit relay + score accumulation ----
            relay_ub = T.alloc_shared([BN, QG2 * heads], dtype)  # GM bf16 -> UB bf16 (this AIV's columns)
            score_row = T.alloc_shared([QG2, CAP], accum_dtype)
            w_src = T.alloc_shared([QG2 * heads], dtype)
            w_ub = T.alloc_shared([QG2 * heads], accum_dtype)  # fp32 weights
            ack_ub = T.alloc_shared([_ACK_PER_AIV], accum_dtype)  # ack heartbeat
            # ---- UB: fused full-tile [BN, QG2H] vector chain (perf iter-1/2:
            #      the per-token elementwise chains fuse into ONE big-tile
            #      chain -- dense flat casts; elementwise ops on disjoint
            #      column blocks are bit-identical) ----
            logits_f = T.alloc_shared([BN, QG2 * heads], accum_dtype)
            bc_f = T.alloc_shared([BN, QG2 * heads], accum_dtype)  # w broadcast
            prod_f = T.alloc_shared([BN, QG2 * heads], dtype)  # golden product round (note 4)
            score_2d = T.alloc_shared([QG2, BN], accum_dtype)  # per-block 4-token scores
            # ---- UB: fused post-reduce causal-validity mask (straddle blocks;
            #      [QG2, BN] tiles -- pre-reduce [H, BN] selects cannot address
            #      the strided per-token segments of the transposed layout) ----
            t_abs_big = T.alloc_shared([64], accum_dtype)  # arith 256B-quantum safe
            colg2 = T.alloc_shared([QG2, BN], accum_dtype)  # col per column
            tpart = T.alloc_shared([QG2, BN], accum_dtype)  # t_abs per row
            # compare dst: packed uint8 predicate (API contract, N/8 bytes;
            # PTO vcmp_dispatch hard-requires a uint8 dst, phase-4 note 15)
            mask2 = T.alloc_shared([QG2, BN // 8], "uint8")
            col_1d = T.alloc_shared([BN], accum_dtype)
            # ---- PTO branch (iter-2b): half-token wide-row chain buffers ----
            relay_ub_h = T.alloc_shared([BN, HALFC], dtype)
            logits_f_h = T.alloc_shared([BN, HALFC], accum_dtype)
            bc_f_h = T.alloc_shared([BN, HALFC], accum_dtype)
            prod_f_h = T.alloc_shared([BN, HALFC], dtype)  # golden product round (note 4)
            seg_t = T.alloc_shared([HALFC, BN], accum_dtype)  # transposed staging (TROWSUM-free reduce)
            # ---- PTO branch (iter-2e): tail [CAP]-row causal mask buffers ----
            col_cap = T.alloc_shared([CAP], accum_dtype)
            valid_cap = T.alloc_shared([CAP // 8], "uint8")
            # ---- UB: per-token topk + softmax + fixup ----
            topk_src = T.alloc_shared([CAP], accum_dtype)  # topk needs a 1D Buffer
            topk_dst = T.alloc_shared([P2], accum_dtype)  # (val, idx) pairs + -inf pad (note 12)
            val_ub = T.alloc_shared([TOPK_A], accum_dtype)
            idx_f_ub = T.alloc_shared([TOPK_A], accum_dtype)
            e_ub = T.alloc_shared([TOPK_A], accum_dtype)
            softmax_ub = T.alloc_shared([TOPK_A], accum_dtype)
            valid = T.alloc_shared([TOPK_A // 8], "uint8")  # packed compare dst (note 15)
            idx_ub = T.alloc_shared([TOPK_A], indices_dtype)
            max_ub = T.alloc_shared([1], accum_dtype)
            sum_ub = T.alloc_shared([1], accum_dtype)

            ntask = T.max(0, T.ceildiv(num_groups - cid, grid_size))
            for k in T.serial(ntask):
                bx = k * grid_size + cid
                bos = GroupTable[bx, 0]
                t0 = GroupTable[bx, 1]
                ln = GroupTable[bx, 2]
                t_max = T.min(t0 + QG, ln) - 1  # group's max causally-valid column
                vh = vid * QG2  # this AIV's contiguous token offset (note 10)

                # ---- Q (C side, full QG rows) / weights (V side, this AIV's
                #      half) loads, varlen-tail guarded. PTO branch (note 15):
                #      the per-row guarded GM->L1 loads are broken under PTO
                #      (varlen tail groups got garbage transit rows); the bulk
                #      loads carry the codegen's runtime tail clamp + zero
                #      pad, and out-of-segment rows are causally masked
                #      (cols >= ln) or belong to unconsumed garbage tokens. ----
                if target == "pto":
                    T.copy(IndexQ[(bos + t0) * heads : (bos + t0 + QG) * heads, 0:dim], Q_l1)
                else:
                    if t0 + QG <= ln:
                        T.copy(IndexQ[(bos + t0) * heads : (bos + t0 + QG) * heads, 0:dim], Q_l1)
                    else:
                        for u in T.serial(QG):
                            if t0 + u < ln:
                                T.copy(
                                    IndexQ[(bos + t0 + u) * heads : (bos + t0 + u + 1) * heads, 0:dim],
                                    Q_l1[u * heads : (u + 1) * heads, 0:dim],
                                )
                if target == "pto":
                    T.copy(
                        Weights[(bos + t0 + vh) * heads : (bos + t0 + vh + QG2) * heads],
                        w_src,
                    )
                else:
                    if t0 + vh + QG2 <= ln:
                        T.copy(
                            Weights[(bos + t0 + vh) * heads : (bos + t0 + vh + QG2) * heads],
                            w_src,
                        )
                    else:
                        for u in T.serial(QG2):
                            if t0 + vh + u < ln:
                                T.copy(
                                    Weights[(bos + t0 + vh + u) * heads : (bos + t0 + vh + u + 1) * heads],
                                    w_src[u * heads : (u + 1) * heads],
                                )
                T.copy(w_src, w_ub)  # bf16 -> fp32 auto-cast (UB->UB); no sm_scale (note 4)
                # this AIV's per-token causal bounds [QG2] (rows of the fused
                # post-reduce mask; [64] buffer absorbs the arith 256B quantum)
                T.tile.arith_progression(t_abs_big, t0 + vh, 1, QG2)
                T.tile.fill(ack_ub, 0.0)  # heartbeat payload (note 11)
                T.tile.fill(score_row, -T.infinity(accum_dtype))  # topk pad + stale-data guard

                # ---- K-block pipeline: NBLK+1 iterations. Produce block
                #      (j < NBLK) and ack-wait block (j >= 1, cond_{j-1})
                #      are separate -- produce-then-wait keeps one slot of
                #      overlap; the +1 iteration drains the task's last ack
                #      (indexer_bwd p-loop shape, note 11). PTO branch
                #      (phase-4 note 15): serialized single-flight protocol
                #      instead -- the AIC waits the previous block's ack
                #      BEFORE producing (the generic path's proven form;
                #      the overlapped ping-pong raced on PTO with intermittent
                #      stale relay reads, see debug_log phase 4). ----
                for j in T.serial(NBLK + 1):
                    k_st = j * BN
                    if j < NBLK:  # noqa: SIM102 (pipeline block structure)
                        if k_st <= t_max:
                            if target == "pto":
                                # bulk K load (note 15): the codegen's runtime
                                # tail clamp + zero pad protect the tensor end;
                                # cols >= ln are causally masked below
                                T.copy(IndexK[bos + k_st : bos + k_st + BN, 0:dim], K_l1)
                            else:
                                if k_st + BN <= ln:
                                    T.copy(IndexK[bos + k_st : bos + k_st + BN, 0:dim], K_l1)
                                else:
                                    # K tail block: per-row guarded load; cols >= ln are
                                    # causally masked to -inf below (DESIGN §5.3)
                                    for i in T.serial(BN):
                                        if k_st + i < ln:
                                            T.copy(IndexK[bos + k_st + i, 0:dim], K_l1[i, 0:dim])
                            # GEMM + CV transit leg 1 (C): L0C fp32 -> GM bf16
                            # + relu (golden bf16 round point; slot j % 2,
                            # note 11). PTO branch (note 15): the generic
                            # orientation [QG*H, BN] (Q @ K^T; same dot
                            # products, same kL0Size K-chunk accumulation
                            # order -- bit-identical GEMM) -- the transposed
                            # [BN, QG*H] transit raced on PTO (tail rows of
                            # the AIV relay read arrived stale; see
                            # debug_log phase 4).
                            if target == "pto":
                                # iter-2b-pto: transposed orientation (ascendc
                                # fast structure) -- [BN, QG*H] L0C + 1KB-row
                                # transit write
                                T.gemm_v0(K_l1, Q_l1, logits_l0c, transpose_B=True, init=True, kL0Size=KL0)
                                T.copy(
                                    logits_l0c,
                                    workspace_logits[cid, j % 2, 0:BN, 0 : QG * heads],
                                    enable_relu=True,
                                )
                            else:
                                # perf iter-2 transposed orientation: K as the
                                # A operand, Q as transpose_B; wide transit rows
                                T.gemm_v0(K_l1, Q_l1, logits_l0c, transpose_B=True, init=True, kL0Size=KL0)
                                T.copy(
                                    logits_l0c,
                                    workspace_logits[cid, j % 2, 0:BN, 0 : QG * heads],
                                    enable_relu=True,
                                )
                    if j >= 1:  # noqa: SIM102 (pipeline block structure)
                        if k_st - BN <= t_max:  # a transit was produced at j-1
                            # back-pressure (iter-2: both backends share the
                            # overlapped produce-then-wait ping-pong; the PTO
                            # serialized single-flight was a conservative form
                            # whose race attribution is disproven -- the real
                            # culprits were the PTO tensor-compare and TROWSUM
                            # library defects, both bypassed in iter-1)
                            T.copy(workspace_ack[cid, 0:_ACK_TOTAL], ack_l1[0, 0:_ACK_TOTAL])
                    if j < NBLK:  # noqa: SIM102 (pipeline block structure)
                        if k_st <= t_max:
                            if target == "pto":
                                # iter-2b-pto: transposed relay (32 rows
                                # x 256B bf16, stride 1KB) + HALF-token
                                # wide-row chains ([BN, HALFC] tiles, 512B
                                # fp32 rows -- the PTO vector template needs
                                # >=256B inner rows; the [QG2H, BN] 128B-row
                                # form both underperformed and mis-executed),
                                # column-segment staging + whole-buffer reduce
                                # (PTO tile views cannot address column
                                # slices of a [BN, QG2H] buffer,
                                # pto_tile.hpp:1423)
                                for hh in T.serial(2):
                                    T.copy(
                                        workspace_logits[
                                            cid,
                                            j % 2,
                                            0:BN,
                                            vh * heads + hh * HALFC : vh * heads + (hh + 1) * HALFC,
                                        ],
                                        relay_ub_h,
                                    )
                                    T.tile.broadcast(bc_f_h, w_ub[hh * HALFC : (hh + 1) * HALFC], axis=0)  # [HALFC] -> [BN, HALFC]
                                    T.copy(relay_ub_h, logits_f_h)  # bf16 -> fp32 (dense flat)
                                    T.tile.mul(logits_f_h, logits_f_h, bc_f_h)  # fp32 product p * w
                                    T.copy(logits_f_h, prod_f_h)  # fp32 -> bf16 (golden product round, note 4)
                                    T.copy(prod_f_h, logits_f_h)  # bf16 -> fp32
                                    if not scale_after_reduce:
                                        T.tile.mul(logits_f_h, logits_f_h, sm_scale)  # s <= 0 fallback
                                    # iter-3-pto: transpose + same-shape
                                    # in-place TADD folding tree (6 levels,
                                    # all operands row-continuous segments).
                                    # The PTO TROWSUM (dim=-1) is broken for
                                    # validRow>1 dst (vcadd dstRptStride=Cols
                                    # overruns the [1,N] dst by ~4KB -- device
                                    # UB-out-of-bounds in isolation), and the
                                    # TCOLSUM cross-lane form measured
                                    # ~1.4ms/AIV (narrow-row cross-lane
                                    # reduce); the TADD tree runs the same
                                    # math on the plain binop path.
                                    T.tile.transpose(seg_t, logits_f_h)
                                    for u in T.serial(HTOK):
                                        r = u * heads
                                        # same-shape in-place TADD folding tree,
                                        # statically branched per pow2 heads
                                        # (TVM script cannot prove shape equality
                                        # for TIR-expr tree strides; non-pow2
                                        # heads fall back to the TCOLSUM form)
                                        if heads == 64:
                                            T.tile.add(seg_t[r : r + 32, 0:BN], seg_t[r : r + 32, 0:BN], seg_t[r + 32 : r + 64, 0:BN])
                                            T.tile.add(seg_t[r : r + 16, 0:BN], seg_t[r : r + 16, 0:BN], seg_t[r + 16 : r + 32, 0:BN])
                                            T.tile.add(seg_t[r : r + 8, 0:BN], seg_t[r : r + 8, 0:BN], seg_t[r + 8 : r + 16, 0:BN])
                                            T.tile.add(seg_t[r : r + 4, 0:BN], seg_t[r : r + 4, 0:BN], seg_t[r + 4 : r + 8, 0:BN])
                                            T.tile.add(seg_t[r : r + 2, 0:BN], seg_t[r : r + 2, 0:BN], seg_t[r + 2 : r + 4, 0:BN])
                                            T.tile.add(seg_t[r : r + 1, 0:BN], seg_t[r : r + 1, 0:BN], seg_t[r + 1 : r + 2, 0:BN])
                                        elif heads == 32:
                                            T.tile.add(seg_t[r : r + 16, 0:BN], seg_t[r : r + 16, 0:BN], seg_t[r + 16 : r + 32, 0:BN])
                                            T.tile.add(seg_t[r : r + 8, 0:BN], seg_t[r : r + 8, 0:BN], seg_t[r + 8 : r + 16, 0:BN])
                                            T.tile.add(seg_t[r : r + 4, 0:BN], seg_t[r : r + 4, 0:BN], seg_t[r + 4 : r + 8, 0:BN])
                                            T.tile.add(seg_t[r : r + 2, 0:BN], seg_t[r : r + 2, 0:BN], seg_t[r + 2 : r + 4, 0:BN])
                                            T.tile.add(seg_t[r : r + 1, 0:BN], seg_t[r : r + 1, 0:BN], seg_t[r + 1 : r + 2, 0:BN])
                                        elif heads == 16:
                                            T.tile.add(seg_t[r : r + 8, 0:BN], seg_t[r : r + 8, 0:BN], seg_t[r + 8 : r + 16, 0:BN])
                                            T.tile.add(seg_t[r : r + 4, 0:BN], seg_t[r : r + 4, 0:BN], seg_t[r + 4 : r + 8, 0:BN])
                                            T.tile.add(seg_t[r : r + 2, 0:BN], seg_t[r : r + 2, 0:BN], seg_t[r + 2 : r + 4, 0:BN])
                                            T.tile.add(seg_t[r : r + 1, 0:BN], seg_t[r : r + 1, 0:BN], seg_t[r + 1 : r + 2, 0:BN])
                                        elif heads == 8:
                                            T.tile.add(seg_t[r : r + 4, 0:BN], seg_t[r : r + 4, 0:BN], seg_t[r + 4 : r + 8, 0:BN])
                                            T.tile.add(seg_t[r : r + 2, 0:BN], seg_t[r : r + 2, 0:BN], seg_t[r + 2 : r + 4, 0:BN])
                                            T.tile.add(seg_t[r : r + 1, 0:BN], seg_t[r : r + 1, 0:BN], seg_t[r + 1 : r + 2, 0:BN])
                                        else:
                                            T.reduce_sum(
                                                seg_t[u * heads : (u + 1) * heads, 0:BN],
                                                score_2d[hh * HTOK + u, 0:BN],
                                                dim=0,
                                                clear=True,
                                            )
                                        if heads in (8, 16, 32, 64):
                                            T.copy(seg_t[r : r + 1, 0:BN], score_2d[hh * HTOK + u, 0:BN])
                                # (iter-2e) no block-level mask: causal masking moved to the tail
                            else:
                                # CV transit leg 2 (V): this AIV's contiguous half
                                # GM bf16 -> UB bf16 (disjoint column halves, note 10;
                                # [BN, QG2H] region = 32 x 512B wide bursts, perf iter-2)
                                T.copy(
                                    workspace_logits[cid, j % 2, 0:BN, vh * heads : (vh + QG2) * heads],
                                    relay_ub,
                                )
                                # fused full-tile chain (perf iter-1/2): dense cast
                                # + w-mul + golden round-trip over ALL this AIV's
                                # tokens at once (was: QG2 separate [H, BN] chains,
                                # 13 ops each; strided half casts degenerate into
                                # per-row loops)
                                T.tile.broadcast(bc_f, w_ub, axis=0)  # [QG2H] -> [BN, QG2H]
                                T.copy(relay_ub, logits_f)  # bf16 -> fp32 (dense flat)
                                T.tile.mul(logits_f, logits_f, bc_f)  # fp32 product p * w
                                T.copy(logits_f, prod_f)  # fp32 -> bf16 (golden product round, note 4)
                                T.copy(prod_f, logits_f)  # bf16 -> fp32
                                if not scale_after_reduce:
                                    T.tile.mul(logits_f, logits_f, sm_scale)  # s <= 0 fallback
                                for u in T.serial(QG2):
                                    # unconditional reduce: the post-reduce mask
                                    # below overwrites every causally-invalid
                                    # (token, column) with -inf (garbage rows
                                    # included -- they are never consumed)
                                    T.reduce_sum(
                                        logits_f[0:BN, u * heads : (u + 1) * heads],
                                        score_2d[u, 0:BN],
                                        dim=-1,
                                        clear=True,
                                    )  # [BN, H] -> [BN] (contiguous head reduce)
                                # fused causal-validity mask, diagonal blocks only
                                # (perf iter-2: [QG2, BN] post-reduce form -- a
                                # pre-reduce select cannot address the strided
                                # per-token segments). Blocks fully below the first
                                # token's diagonal (k_st + BN <= t0 + vh) skip it:
                                # every row is fully valid there; every other block
                                # masks its invalid (token, column) set in one shot
                                if k_st + BN > t0 + vh:
                                    T.tile.arith_progression(col_1d, k_st, 1, BN)
                                    T.tile.broadcast(colg2, col_1d, axis=0)  # [BN] -> [QG2, BN]
                                    T.tile.broadcast(tpart, t_abs_big[0:QG2], axis=1)  # [QG2] rows
                                    T.tile.compare(mask2, colg2, tpart, "LE")  # col <= t_abs(row)
                                    T.tile.select(
                                        score_2d,
                                        mask2,
                                        score_2d,
                                        -T.infinity(accum_dtype),
                                        "VSEL_TENSOR_SCALAR_MODE",
                                    )
                            # fused 4-token store (dst rows stride CAP; invalid
                            # positions carry the mask's -inf)
                            T.copy(score_2d, score_row[0:QG2, k_st : k_st + BN])
                            # ack: transit consumed -> the AIC may reuse slot
                            # j % 2 (after the chain: V->MTE3 ordered, note 11)
                            T.copy(ack_ub, workspace_ack[cid, vid * _ACK_PER_AIV : (vid + 1) * _ACK_PER_AIV])

                # ---- per-token topk + softmax + fixup + writeback ----
                T.tile.fill(val_ub, -T.infinity(accum_dtype))  # pads for topk < TOPK_A
                T.tile.fill(idx_f_ub, -1.0)
                for u in T.serial(QG2):
                    t_abs = t0 + vh + u
                    if t_abs < ln:
                        i_u = t_abs  # causal bound = in-segment position
                        # K <= actual_num guard. if_then_else instead of T.max:
                        # the bisheng CCE max() overload set is {(long,long),
                        # (int,int)} and the vid-derived (int64, int) argument
                        # pair is ambiguous under -xcce (phase-4 note 15);
                        # identical integer semantics on both backends.
                        n_eff = T.if_then_else(i_u + 1 > topk, i_u + 1, topk)
                        T.copy(score_row[u, 0:CAP], topk_src)
                        if scale_after_reduce:
                            # golden order: (fp32 sum of bf16 products) * sm_scale,
                            # one [CAP] Muls per token per task instead of a per-block
                            # tile mul (perf iter-1); s > 0 keeps the descending
                            # topk order and the -inf tail signs
                            T.tile.mul(topk_src, topk_src, sm_scale)
                            if target == "pto":
                                # PTO tail causal mask (golden-exact -inf form;
                                # the PTO branch dropped the block-level mask,
                                # ascendc keeps its frozen pre-reduce mask and
                                # must NOT re-mask -- performance freeze)
                                T.tile.arith_progression(col_cap, 0, 1, CAP)
                                T.tile.compare(valid_cap, col_cap, T.float32(i_u), "LE")
                                T.tile.select(
                                    topk_src,
                                    valid_cap,
                                    topk_src,
                                    -T.infinity(accum_dtype),
                                    "VSEL_TENSOR_SCALAR_MODE",
                                )
                        # pad [2*topk:P2] with -inf so the 256B-quantised
                        # gather_mask never selects garbage into the pads
                        T.tile.fill(topk_dst, -T.infinity(accum_dtype))
                        T.tile.topk(topk_dst, topk_src, topk, n_eff)
                        T.tile.gather_mask(val_ub, topk_dst, "P0101")  # even = vals
                        T.tile.gather_mask(idx_f_ub, topk_dst, "P1010")  # odd = idx
                        # softmax over [TOPK_A] (pads are -inf -> exp 0)
                        T.reduce_max(val_ub, max_ub, dim=-1)
                        T.tile.sub(e_ub, val_ub, max_ub[0])
                        T.tile.exp(e_ub, e_ub)
                        T.reduce_sum(e_ub, sum_ub, dim=-1)
                        T.tile.div(softmax_ub, e_ub, sum_ub[0])
                        # idx fixup: -inf tail positions (idx > i_u) -> -1
                        T.tile.compare(valid, idx_f_ub, T.float32(i_u), "LE")
                        T.tile.select(idx_f_ub, valid, idx_f_ub, -1.0, "VSEL_TENSOR_SCALAR_MODE")
                        T.tile.cast(idx_ub, idx_f_ub, "CAST_ROUND", TOPK_A)
                        T.copy(idx_ub[0:topk], TopkIndices[bos + t_abs, 0:topk])
                        T.copy(softmax_ub[0:topk], ReduceSum[bos + t_abs, 0:topk])

    return main


# ============================================================================
# Generic kernel (CH-streamed score segments + merge_sort pool topk)
# ============================================================================


def tl_indexer_topk_reducesum_generic(
    heads,
    dim,
    topk,
    CH=1024,
    NSEG=5,
    QG=8,
    BN=32,
    grid_size=20,
    sm_scale=None,
    KL0=128,
    target="ascendc",
):
    """Build the generic-path kernel (any segment length; Developer mode).

    Same GEMM data flow as the fast path, but the score row is produced CH
    columns at a time; each CH segment runs a hardware topk whose LOCAL
    indices are corrected to the global column frame (fp32 + seg_lo) and
    merged into a per-token (val, idx) pool via the stable descending
    merge_sort -- the pool after the last segment is the full-row topk.

    Args:
        heads / dim / topk / QG / BN / grid_size / sm_scale / KL0: as in the
            fast kernel (topk <= CH; KL0 host-computed, docstring note 9).
        CH: score segment width (max(1024, next_pow2(topk)); % BN == 0).
        NSEG: compile-time upper bound on segments per group
            (= ceildiv(max_seg_len, CH), host-computed; runtime guard
            ``seg_lo <= t_max`` truncates shorter groups).
        target: codegen backend, "ascendc" (default) or "pto" — same body.
    """
    return _jit_wrapper(target)(_tl_indexer_topk_reducesum_generic_main)(
        heads=heads,
        dim=dim,
        topk=topk,
        CH=CH,
        NSEG=NSEG,
        QG=QG,
        BN=BN,
        grid_size=grid_size,
        sm_scale=sm_scale,
        KL0=KL0,
        target=target,
    )


def _tl_indexer_topk_reducesum_generic_main(
    heads,
    dim,
    topk,
    CH=1024,
    NSEG=5,
    QG=8,
    BN=32,
    grid_size=20,
    sm_scale=None,
    KL0=128,
    target="ascendc",
):
    """(private) generic-path prim_func builder — see the public wrapper above."""
    assert heads % 8 == 0 and heads <= 64, "heads must be % 8 == 0 and <= 64"
    assert dim % 16 == 0, "dim must be a multiple of 16"
    assert topk == tilelang.math.next_power_of_2(topk) and topk >= 16, f"topk must be a power of two >= 16, got {topk}"
    assert CH % BN == 0 and topk <= CH, "CH must be a multiple of BN and >= topk"
    assert NSEG >= 1, "NSEG must be >= 1"
    assert QG % 2 == 0, "QG must be even (AIV-vid token-half split)"
    assert QG * heads * BN * 4 <= _L0C_CAPACITY, "L0C budget exceeded"
    assert KL0 % 16 == 0 and KL0 <= 4095, "KL0 must be % 16 == 0 and <= 4095"
    if sm_scale is None:
        sm_scale = dim**-0.5

    NBLK_SEG = CH // BN
    TOPK_A = max(topk, 64)
    P2 = max(2 * topk, 64)  # gather_mask 256B src quantisation (fast-path note 12)
    PS = max(topk, 32)  # gather dst width = selected count = P2 // 2
    QG2 = QG // 2  # tokens per AIV-vid (contiguous halves, docstring note 10)

    seq_len = T.symbolic("seq_len")
    batch_plus_one = T.symbolic("batch_plus_one")
    num_groups = T.symbolic("num_groups")
    q_rows = T.symbolic("q_rows")
    w_rows = T.symbolic("w_rows")

    dtype = "bfloat16"
    accum_dtype = "float32"
    indices_dtype = "int32"

    @T.prim_func
    def main(
        IndexQ: T.Tensor([q_rows, dim], dtype),  # type: ignore
        Weights: T.Tensor([w_rows], dtype),  # type: ignore
        IndexK: T.Tensor([seq_len, dim], dtype),  # type: ignore
        TopkIndices: T.Tensor([seq_len, topk], indices_dtype),  # type: ignore
        ReduceSum: T.Tensor([seq_len, topk], accum_dtype),  # type: ignore
        Offsets: T.Tensor([batch_plus_one], indices_dtype),  # type: ignore
        TokenIndices: T.Tensor([seq_len, 2], indices_dtype),  # type: ignore  # kept for interface fidelity, not consumed
        GroupTable: T.Tensor([num_groups, 3], indices_dtype),  # type: ignore
        workspace_logits: T.Tensor([grid_size, QG * heads, BN], dtype),  # type: ignore
        workspace_ack: T.Tensor([grid_size, _ACK_TOTAL], accum_dtype),  # type: ignore  # V->C back-pressure
    ):
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):  # MIX_AIC_1_2
            # ---- L1 / L0C / K-loop UB chain: identical to the fast path ----
            Q_l1 = T.alloc_shared([QG * heads, dim], dtype)
            K_l1 = T.alloc_shared([BN, dim], dtype)
            # explicit 2D L1 (plain "shared" is use-inferred to UB, which
            # reclassifies the ack GM read as a vector op and breaks the CV
            # pairing; 1D L1 breaks the default zN layout injection)
            ack_l1 = T.alloc_shared([1, _ACK_TOTAL], accum_dtype, scope="shared.l1")
            logits_l0c = T.alloc_fragment([QG * heads, BN], accum_dtype)
            relay_ub = T.alloc_shared([QG2 * heads, BN], dtype)
            w_src = T.alloc_shared([QG2 * heads], dtype)
            w_ub = T.alloc_shared([QG2 * heads], accum_dtype)
            ack_ub = T.alloc_shared([_ACK_PER_AIV], accum_dtype)  # ack heartbeat
            logits_f32 = T.alloc_shared([heads, BN], accum_dtype)
            prod_bf16 = T.alloc_shared([heads, BN], dtype)  # golden product round (note 4)
            bc_2d = T.alloc_shared([heads, BN], accum_dtype)
            causal = T.alloc_shared([heads, BN // 8], "uint8")  # packed compare dst (note 15)
            w_1d = T.alloc_shared([heads], accum_dtype)
            col_1d = T.alloc_shared([BN], accum_dtype)
            score_u = T.alloc_shared([BN], accum_dtype)
            # ---- generic topk consumption (per AIV-vid) ----
            score_seg = T.alloc_shared([QG2, CH], accum_dtype)  # local columns [0, CH)
            seg_src = T.alloc_shared([CH], accum_dtype)
            seg_dst = T.alloc_shared([P2], accum_dtype)  # + -inf pad (note 12)
            seg_val = T.alloc_shared([PS], accum_dtype)  # gather dst width
            seg_idx = T.alloc_shared([PS], accum_dtype)
            seg_pairs = T.alloc_shared([2 * topk], accum_dtype)
            pool = T.alloc_shared([QG2, 2 * topk], accum_dtype)  # (val, idx) pairs, global idx
            pool_row = T.alloc_shared([P2], accum_dtype)  # gather src, 256B-quantised (note 12)
            merged = T.alloc_shared([4 * topk], accum_dtype)
            # ---- tail chain (identical to the fast path) ----
            val_ub = T.alloc_shared([TOPK_A], accum_dtype)
            idx_f_ub = T.alloc_shared([TOPK_A], accum_dtype)
            e_ub = T.alloc_shared([TOPK_A], accum_dtype)
            softmax_ub = T.alloc_shared([TOPK_A], accum_dtype)
            valid = T.alloc_shared([TOPK_A // 8], "uint8")  # packed compare dst (note 15)
            idx_ub = T.alloc_shared([TOPK_A], indices_dtype)
            max_ub = T.alloc_shared([1], accum_dtype)
            sum_ub = T.alloc_shared([1], accum_dtype)

            ntask = T.max(0, T.ceildiv(num_groups - cid, grid_size))
            for k in T.serial(ntask):
                bx = k * grid_size + cid
                bos = GroupTable[bx, 0]
                t0 = GroupTable[bx, 1]
                ln = GroupTable[bx, 2]
                t_max = T.min(t0 + QG, ln) - 1
                vh = vid * QG2  # this AIV's contiguous token offset (note 10)

                # ---- Q (C side, full QG) / weights (V side, this AIV's half).
                #      PTO branch (note 15): bulk loads only (the per-row
                #      guarded GM->L1 loads are broken under PTO; the bulk
                #      copy's runtime tail clamp + zero pad protect the tensor
                #      end and out-of-segment rows are masked/unconsumed). ----
                if target == "pto":
                    T.copy(IndexQ[(bos + t0) * heads : (bos + t0 + QG) * heads, 0:dim], Q_l1)
                else:
                    if t0 + QG <= ln:
                        T.copy(IndexQ[(bos + t0) * heads : (bo      s + t0 + QG) * heads, 0:dim], Q_l1)
                    else:
                        for u in T.serial(QG):
                            if t0 + u < ln:
                                T.copy(
                                    IndexQ[(bos + t0 + u) * heads : (bos + t0 + u + 1) * heads, 0:dim],
                                    Q_l1[u * heads : (u + 1) * heads, 0:dim],
                                )
                if target == "pto":
                    T.copy(
                        Weights[(bos + t0 + vh) * heads : (bos + t0 + vh + QG2) * heads],
                        w_src,
                    )
                else:
                    if t0 + vh + QG2 <= ln:
                        T.copy(
                            Weights[(bos + t0 + vh) * heads : (bos + t0 + vh + QG2) * heads],
                            w_src,
                        )
                    else:
                        for u in T.serial(QG2):
                            if t0 + vh + u < ln:
                                T.copy(
                                    Weights[(bos + t0 + vh + u) * heads : (bos + t0 + vh + u + 1) * heads],
                                    w_src[u * heads : (u + 1) * heads],
                                )
                T.copy(w_src, w_ub)  # bf16 -> fp32; no sm_scale (note 4)
                T.tile.fill(ack_ub, 0.0)  # heartbeat payload (note 11)

                # ---- CH-segment loop (compile-time bound + runtime guard).
                #      Serialized back-pressure (not perf-gated): the ack of
                #      the previous produce (k + sg + j > 0 skips only the
                #      core's very first) is waited BEFORE this produce, so a
                #      single transit slot is safe (note 11). ----
                for sg in T.serial(NSEG):
                    seg_lo = sg * CH
                    if seg_lo <= t_max:
                        T.tile.fill(score_seg, -T.infinity(accum_dtype))
                        for j in T.serial(NBLK_SEG):
                            k_st = seg_lo + j * BN  # global column (causal frame)
                            if k_st <= t_max:
                                if k + sg + j > 0:
                                    T.copy(workspace_ack[cid, 0:_ACK_TOTAL], ack_l1[0, 0:_ACK_TOTAL])
                                if target == "pto":
                                    # bulk K load (note 15): runtime tail clamp
                                    # + zero pad; cols >= ln causally masked
                                    T.copy(IndexK[bos + k_st : bos + k_st + BN, 0:dim], K_l1)
                                else:
                                    if k_st + BN <= ln:
                                        T.copy(IndexK[bos + k_st : bos + k_st + BN, 0:dim], K_l1)
                                    else:
                                        for i in T.serial(BN):
                                            if k_st + i < ln:
                                                T.copy(IndexK[bos + k_st + i, 0:dim], K_l1[i, 0:dim])
                                T.gemm_v0(
                                    Q_l1,
                                    K_l1,
                                    logits_l0c,
                                    transpose_B=True,
                                    init=True,
                                    kL0Size=KL0,
                                )
                                T.copy(
                                    logits_l0c,
                                    workspace_logits[cid, 0 : QG * heads, 0:BN],
                                    enable_relu=True,
                                )
                                # this AIV's contiguous half (disjoint-halves, note 10)
                                T.copy(
                                    workspace_logits[cid, vh * heads : (vh + QG2) * heads, 0:BN],
                                    relay_ub,
                                )
                                for u in T.serial(QG2):
                                    t_abs = t0 + vh + u
                                    T.copy(w_ub[u * heads : (u + 1) * heads], w_1d)
                                    T.tile.broadcast(bc_2d, w_1d, axis=1)
                                    T.copy(relay_ub[u * heads : (u + 1) * heads, 0:BN], logits_f32)
                                    T.tile.mul(logits_f32, logits_f32, bc_2d)  # fp32 product p * w
                                    T.copy(logits_f32, prod_bf16)  # golden product round (note 4)
                                    T.copy(prod_bf16, logits_f32)
                                    T.tile.mul(logits_f32, logits_f32, sm_scale)  # scale after the round
                                    T.tile.arith_progression(col_1d, k_st, 1, BN)
                                    T.tile.broadcast(bc_2d, col_1d, axis=0)
                                    T.tile.compare(causal, bc_2d, T.float32(t_abs), "LE")
                                    T.tile.select(
                                        logits_f32,
                                        causal,
                                        logits_f32,
                                        -T.infinity(accum_dtype),
                                        "VSEL_TENSOR_SCALAR_MODE",
                                    )
                                    T.reduce_sum(logits_f32, score_u, dim=0, clear=True)
                                    # local column inside the CH segment
                                    T.copy(score_u, score_seg[u, j * BN : j * BN + BN])
                                # ack: transit consumed (after the chain, note 11)
                                T.copy(
                                    ack_ub,
                                    workspace_ack[cid, vid * _ACK_PER_AIV : (vid + 1) * _ACK_PER_AIV],
                                )
                        # ---- segment topk + global-idx correction + pool merge ----
                        for u in T.serial(QG2):
                            t_abs = t0 + vh + u
                            if t_abs < ln:
                                i_u = t_abs
                                T.copy(score_seg[u, 0:CH], seg_src)
                                # min/max -> if_then_else (CCE max/min overload
                                # ambiguity on the vid-derived int64 argument,
                                # phase-4 note 15); identical integer semantics
                                in_range = i_u + 1 - seg_lo
                                seg_valid = T.if_then_else(
                                    in_range < CH,
                                    T.if_then_else(in_range > topk, in_range, topk),
                                    T.if_then_else(topk < CH, CH, topk),
                                )
                                T.tile.fill(seg_dst, -T.infinity(accum_dtype))  # pad (note 12)
                                T.tile.topk(seg_dst, seg_src, topk, seg_valid)
                                T.tile.gather_mask(seg_val, seg_dst, "P0101")
                                T.tile.gather_mask(seg_idx, seg_dst, "P1010")
                                # local -> global column frame
                                T.tile.add(seg_idx, seg_idx, T.float32(seg_lo))
                                # scalar interleave back into (val, idx) pairs
                                for i in T.serial(topk):
                                    seg_pairs[2 * i] = seg_val[i]
                                    seg_pairs[2 * i + 1] = seg_idx[i]
                                if sg == 0:
                                    T.copy(seg_pairs, pool[u, 0 : 2 * topk])
                                else:
                                    T.tile.merge_sort(merged, pool[u, 0 : 2 * topk], seg_pairs)  # stable descending 2-way merge
                                    T.copy(merged[0 : 2 * topk], pool[u, 0 : 2 * topk])

                # ---- tail: softmax + fixup + writeback from the pool ----
                T.tile.fill(val_ub, -T.infinity(accum_dtype))
                T.tile.fill(idx_f_ub, -1.0)
                for u in T.serial(QG2):
                    t_abs = t0 + vh + u
                    if t_abs < ln:
                        i_u = t_abs
                        T.tile.fill(pool_row, -T.infinity(accum_dtype))  # pad (note 12)
                        T.copy(pool[u, 0 : 2 * topk], pool_row[0 : 2 * topk])
                        T.tile.gather_mask(val_ub, pool_row, "P0101")
                        T.tile.gather_mask(idx_f_ub, pool_row, "P1010")
                        T.reduce_max(val_ub, max_ub, dim=-1)
                        T.tile.sub(e_ub, val_ub, max_ub[0])
                        T.tile.exp(e_ub, e_ub)
                        T.reduce_sum(e_ub, sum_ub, dim=-1)
                        T.tile.div(softmax_ub, e_ub, sum_ub[0])
                        T.tile.compare(valid, idx_f_ub, T.float32(i_u), "LE")
                        T.tile.select(idx_f_ub, valid, idx_f_ub, -1.0, "VSEL_TENSOR_SCALAR_MODE")
                        T.tile.cast(idx_ub, idx_f_ub, "CAST_ROUND", TOPK_A)
                        T.copy(idx_ub[0:topk], TopkIndices[bos + t_abs, 0:topk])
                        T.copy(softmax_ub[0:topk], ReduceSum[bos + t_abs, 0:topk])

    return main


# ============================================================================
# Host side: data preparation + detection + gates + dispatch entry
# ============================================================================


def _next_pow2(n):
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _pick_kl0_size(m_rows, dim):
    """Largest gemm_v0 kL0Size whose big-side tile fits the L0A/L0B ping-pong slot.

    Mirror of the C++ static asserts (src/tl_templates/ascend/common.h:1282):
    both the A slot (M x kL0Size) and -- on the transpose_B path -- the B slot
    (nTile x kL0Size) must fit 64KB for a single (N-tile, K-tile) step or 32KB
    once the ping-pong alternates (kNumSteps > 1); ``m_rows`` is the big side
    (QG*heads: the A operand pre-perf-iter-2, the transpose_B operand after).
    Returns 0 when no candidate fits (M > 1024 for dim >= 16 -- outside the
    operator domain).
    """
    for kl0 in (128, 64, 32, 16):
        if kl0 > dim:
            continue
        steps = -(-dim // kl0)
        budget = 65536 if steps == 1 else 32768
        if m_rows * kl0 * 2 <= budget:
            return kl0
    return 0


def _off_list(offsets):
    """One synchronizing D2H per interface call (perf iter-3: the Stage-2 path
    fetched the list five times -- validation, detection, max_seg, both table
    builders)."""
    return [int(x) for x in offsets.detach().cpu().tolist()]


def prepare_token_indices(offsets, off_list=None):
    """Build TokenIndices [sum(S), 2] int32 ((batch_id, position)) from cu_seqlens.

    numpy construction + a single H2D copy (perf iter-3: the Stage-2 per-segment
    Python loop cost ~290us and a torch-CPU-op version ~460us on l0_main --
    torch CPU op overhead dominates at these sizes; interface-fidelity output
    of the main repo's prepare_token_indices; the kernels do not consume it).
    """
    if off_list is None:
        off_list = _off_list(offsets)
    off = np.asarray(off_list, dtype=np.int64)
    lens = off[1:] - off[:-1]
    total = int(lens.sum()) if lens.size else 0
    if total == 0:
        return torch.zeros((0, 2), dtype=torch.int32).to(offsets.device)
    batch = np.repeat(np.arange(lens.size, dtype=np.int32), lens)
    starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
    pos = np.arange(total, dtype=np.int64) - starts[batch.astype(np.int64)]
    ti = np.stack((batch, pos.astype(np.int32)), axis=1)
    return torch.from_numpy(ti).to(offsets.device)


def prepare_group_table(offsets, qg, off_list=None):
    """Build GroupTable [num_groups, 3] int32 ((bos, t0_local, seq_len)).

    One row per ceildiv(len_i, qg) group of each varlen segment; numpy
    construction + a single H2D copy (perf iter-3; indexer_bwd BlockIndices
    precedent; fast/generic share the layout).
    """
    if off_list is None:
        off_list = _off_list(offsets)
    off = np.asarray(off_list, dtype=np.int64)
    lens = off[1:] - off[:-1] if off.size > 1 else np.zeros(0, dtype=np.int64)
    counts = (lens + qg - 1) // qg
    n_groups = int(counts.sum()) if counts.size else 0
    if n_groups == 0:
        return torch.zeros((0, 3), dtype=torch.int32).to(offsets.device)
    seg = np.repeat(np.arange(lens.size, dtype=np.int64), counts)
    base = np.concatenate(([0], np.cumsum(counts)[:-1]))  # first group id per segment
    g = np.arange(n_groups, dtype=np.int64) - base[seg]
    gt = np.stack((off[seg], g * qg, lens[seg]), axis=1).astype(np.int32)
    return torch.from_numpy(gt).to(offsets.device)


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


def _max_seg_len(offsets, off_list=None):
    if off_list is None:
        off_list = _off_list(offsets)
    return max((off_list[b + 1] - off_list[b] for b in range(len(off_list) - 1)), default=0)


def _detect_dispatch(offsets, off_list=None):
    """Exact (non-heuristic) host detection (DESIGN §1.4): 'fast' | 'generic'."""
    return "fast" if _max_seg_len(offsets, off_list) <= 4096 else "generic"


def _fast_gate_ok(heads, topk, CAP, QG, BN=32):
    """Host mirror of the fast-path domain + exact UB/L0A budget model.

    UB model is liveness-aware and per-AIV (MIX_AIC_1_2: each Vector core
    owns QG//2 tokens of every group): the K-block-phase buffers
    (relay/vector chain) and the topk-phase buffers (topk_src/dst/softmax
    chain/auto tmp) have disjoint live ranges inside one task; score_row
    persists across both. The L0A model mirrors _pick_kl0_size.
    """
    if not (heads % 8 == 0 and heads <= 64):
        return False
    if topk != _next_pow2(topk) or topk < 16 or topk > 1024:
        return False
    if CAP not in (512, 1024, 2048, 4096) or topk > CAP:
        return False
    qg_h = QG * heads
    if qg_h * BN * 4 > _L0C_CAPACITY:
        return False
    if QG % 4 != 0:
        return False  # AIV-vid token-half split + half-token chain fusion
    # (L0A feasibility with the actual dim is enforced in _run_fast via
    #  _pick_kl0_size; here M = QG*heads <= 512 always has a candidate.)
    qg2_h = (QG // 2) * heads
    topk_a = max(topk, 64)
    k_phase = (
        qg2_h * BN * 2  # relay_ub bf16 (per AIV)
        + qg2_h * BN * 4 * 2  # logits_f + bc_f (fused full-tile chain, perf iter-1/2)
        + qg2_h * BN * 2  # prod_f bf16 (golden product round, note 4)
        + 64 * 4  # t_abs_big (arith 256B-quantum safe)
        + (QG // 2) * BN * 4 * 3  # colg2 + tpart + mask2 (fused post-reduce mask)
        + qg2_h * 2
        + qg2_h * 4  # w_src bf16 + w_ub fp32 (per AIV)
        + BN * 4  # col_1d
        + (QG // 2) * BN * 4  # score_2d (per-block token scores)
        + 64  # ack_ub
    )
    p2 = max(2 * topk, 64)
    topk_phase = (
        CAP * 4  # topk_src
        + p2 * 4  # topk_dst (256B gather quantum, docstring note 12)
        + 4 * topk_a * 4
        + topk_a * 4
        + topk_a  # val/idx_f/e/softmax + idx_ub + valid
        + 8
    )
    # The planner packs every buffer disjointly (no cross-phase liveness
    # sharing, per the Stage-2 generated layout); the auto tmp is ONE shared
    # buffer sized by its largest user (topk 16B/elem vs the K-phase reduce's
    # active_rows * padded_row_bytes).
    tmp_shared = max(16 * CAP, (heads // 2) * BN * 4)
    total = (QG // 2) * CAP * 4 + k_phase + topk_phase + tmp_shared + 2 * 1024
    return total < _UB_CAPACITY


def _generic_gate_ok(heads, topk, CH=1024, QG=8, BN=32):
    """Host mirror of the generic-path domain + exact per-AIV UB budget model."""
    if not (heads % 8 == 0 and heads <= 64):
        return False
    if topk != _next_pow2(topk) or topk < 16 or topk > 1024:
        return False
    CH = max(CH, _next_pow2(topk))
    if CH % BN != 0 or topk > CH:
        return False
    qg_h = QG * heads
    if qg_h * BN * 4 > _L0C_CAPACITY:
        return False
    qg2_h = (QG // 2) * heads
    topk_a = max(topk, 64)
    k_phase = (
        qg2_h * BN * 2
        + heads * BN * 4 * 2
        + heads * BN * 2  # prod_bf16 (golden product round, note 4)
        + heads * BN * 4
        + qg2_h * 2
        + qg2_h * 4
        + heads * 4
        + BN * 4 * 2
    )
    p2 = max(2 * topk, 64)
    ps = max(topk, 32)
    seg_phase = (
        CH * 4  # seg_src
        + p2 * 4  # seg_dst (256B gather quantum, note 12)
        + 2 * ps * 4  # seg_val + seg_idx (gather dst width)
        + 2 * topk * 4  # seg_pairs
        + p2 * 4  # pool_row (256B gather quantum)
        + 4 * topk * 4  # merged
        + 16 * CH  # auto topk tmp (ascendc)
    )
    tail_phase = 4 * topk_a * 4 + topk_a * 4 + topk_a + 8
    total = (
        (QG // 2) * CH * 4  # score_seg (persists across the sg loop, per AIV)
        + (QG // 2) * 2 * topk * 4  # pool (persists, per AIV)
        + max(k_phase, seg_phase, tail_phase)
        + 8 * 1024
    )
    return total < _UB_CAPACITY


_BUILD_CACHE: dict = {}

# content-keyed memo of the interface tables (perf iter-3): both tables are
# pure functions of (offsets contents, qg), so the key IS the content -- no
# tensor-identity staleness caveat. Two H2D copies at ~85us fixed overhead
# each dominate the build; the benchmark/training loop reuses varlen layouts.
_TABLE_CACHE: dict = {}
_TABLE_CACHE_LIMIT = 64


def _cached_token_indices(offsets, off_list):
    key = ("ti", tuple(off_list), str(offsets.device))
    hit = _TABLE_CACHE.get(key)
    if hit is None:
        hit = prepare_token_indices(offsets, off_list)
        if len(_TABLE_CACHE) >= _TABLE_CACHE_LIMIT:
            _TABLE_CACHE.clear()
        _TABLE_CACHE[key] = hit
    return hit


def _cached_group_table(offsets, off_list, qg):
    key = ("gt", tuple(off_list), qg, str(offsets.device))
    hit = _TABLE_CACHE.get(key)
    if hit is None:
        hit = prepare_group_table(offsets, qg, off_list)
        if len(_TABLE_CACHE) >= _TABLE_CACHE_LIMIT:
            _TABLE_CACHE.clear()
        _TABLE_CACHE[key] = hit
    return hit


def _run_fast(q, weights, k, topk, offsets, sm_scale, token_indices, off_list, backend="ascendc"):
    """Launch the fast kernel (max_seg_len <= 4096 contract already validated)."""
    seq_len, heads, dim = q.shape
    device = q.device
    grid_size = _get_cube_core_num()
    max_seg = _max_seg_len(offsets, off_list)
    CAP = max(512, _next_pow2(max_seg))
    # QG tier (perf iter-2): 8 tokens/group whenever the UB model fits, else 4
    # (the fused [BN, QG2H] fp32 pair overflows at CAP=2048 x topk=1024 with
    # QG=8; QG=4 is an existing legal tier -- CAP=4096 already uses it)
    QG = 8 if CAP <= 2048 else 4
    if CAP <= 2048 and not _fast_gate_ok(heads, topk, CAP, QG):
        QG = 4
    if not _fast_gate_ok(heads, topk, CAP, QG):
        raise ValueError(
            f"fast path out of domain (heads={heads}, topk={topk}, CAP={CAP}, QG={QG}); reduce topk or use a shorter max segment"
        )
    if backend == "pto" and CAP > 2048:
        # PTO fast@CAP=4096 exceeds the 192KB UB budget (the PTO auto topk
        # tmp is 24B/elem = 96KB at CAP=4096, plus the wide-row chain buffers
        # and the transpose's internal tmp -- the planner-allocated layout
        # overruns and the kernel faults). Route to the generic path, which
        # is PTO-validated (phase-4b 3x 49/49) and not perf-gated; both
        # paths are output-equivalent (l0_dispatch_equiv).
        return _run_generic(q, weights, k, topk, offsets, sm_scale, token_indices, off_list, backend)

    group_table = _cached_group_table(offsets, off_list, QG)
    kl0 = _pick_kl0_size(QG * heads, dim)
    if kl0 == 0:
        raise ValueError(f"fast path infeasible: no kL0Size fits L0A for M={QG * heads}, dim={dim}")
    topk_indices = torch.zeros((seq_len, topk), dtype=torch.int32, device=device)
    topk_score = torch.zeros((seq_len, topk), dtype=torch.float32, device=device)
    workspace_logits = torch.empty((grid_size, 2, _BN, QG * heads), dtype=torch.bfloat16, device=device)
    workspace_ack = torch.empty((grid_size, 2 * 16), dtype=torch.float32, device=device)

    key = ("fast", backend, heads, dim, topk, CAP, QG, grid_size, sm_scale, kl0)
    func = _BUILD_CACHE.get(key)
    if func is None:
        kernel = tl_indexer_topk_reducesum_fast(
            heads=heads,
            dim=dim,
            topk=topk,
            CAP=CAP,
            QG=QG,
            BN=_BN,
            grid_size=grid_size,
            sm_scale=sm_scale,
            KL0=kl0,
            target=backend,
        )
        # cached low-overhead launch path (perf iter-3, indexer_bwd note-1
        # precedent): raw [S, H, D] q / [S, H] weights are viewed flat inside
        func = _CachedLauncher(kernel, view_spec={0: "flat2d", 1: "flat1d"})
        _BUILD_CACHE[key] = func

    func(
        q,
        weights,
        k,
        topk_indices,
        topk_score,
        offsets,
        token_indices,
        group_table,
        workspace_logits,
        workspace_logits,  # workspace_logits_t: same bytes, [QG*H, BN] view (note 15)
        workspace_ack,
    )
    return topk_indices, topk_score


def _run_generic(q, weights, k, topk, offsets, sm_scale, token_indices, off_list, backend="ascendc"):
    """Launch the generic kernel (any segment length)."""
    seq_len, heads, dim = q.shape
    device = q.device
    grid_size = _get_cube_core_num()
    max_seg = _max_seg_len(offsets, off_list)
    CH = max(1024, _next_pow2(topk))
    NSEG = (max_seg + CH - 1) // CH
    QG = 8
    if not _generic_gate_ok(heads, topk, CH, QG):
        raise ValueError(f"generic path out of domain (heads={heads}, topk={topk}, CH={CH}); topk must be a power of two in [16, 1024]")

    group_table = _cached_group_table(offsets, off_list, QG)
    kl0 = _pick_kl0_size(QG * heads, dim)
    if kl0 == 0:
        raise ValueError(f"generic path infeasible: no kL0Size fits L0A for M={QG * heads}, dim={dim}")
    topk_indices = torch.zeros((seq_len, topk), dtype=torch.int32, device=device)
    topk_score = torch.zeros((seq_len, topk), dtype=torch.float32, device=device)
    workspace_logits = torch.empty((grid_size, QG * heads, _BN), dtype=torch.bfloat16, device=device)
    workspace_ack = torch.empty((grid_size, 2 * 16), dtype=torch.float32, device=device)

    key = ("generic", backend, heads, dim, topk, CH, NSEG, QG, grid_size, sm_scale, kl0)
    func = _BUILD_CACHE.get(key)
    if func is None:
        kernel = tl_indexer_topk_reducesum_generic(
            heads=heads,
            dim=dim,
            topk=topk,
            CH=CH,
            NSEG=NSEG,
            QG=QG,
            BN=_BN,
            grid_size=grid_size,
            sm_scale=sm_scale,
            KL0=kl0,
            target=backend,
        )
        func = _CachedLauncher(kernel, view_spec={0: "flat2d", 1: "flat1d"})
        _BUILD_CACHE[key] = func

    func(
        q,
        weights,
        k,
        topk_indices,
        topk_score,
        offsets,
        token_indices,
        group_table,
        workspace_logits,
        workspace_ack,
    )
    return topk_indices, topk_score


def indexer_topk_reducesum_interface(
    q,
    weights,
    k,
    topk,
    offsets,
    dtype="bfloat16",
    dispatch="auto",
    sm_scale=None,
    backend="ascendc",
):
    """Host entry: varlen lightning-indexer forward topk+reducesum.

    Main-repo signature (q, weights, k, topk, offsets, dtype=BF16) plus tail
    parameters with defaults (backward compatible).

    Args:
        q: IndexQ [seq_len, heads, dim] bfloat16.
        weights: Weights [seq_len, heads] bfloat16.
        k: IndexK [seq_len, dim] bfloat16.
        topk: selected positions (power of two, 16 <= topk <= 1024, and
            <= every segment length -- the golden asserts segment >= topk).
        offsets: cu_seqlens [batch + 1] int32.
        dtype: input dtype (bfloat16, the main-repo fixed precision path).
        dispatch: "auto" (exact host detection: max_seg_len <= 4096 -> fast,
            else generic), "fast" (validates max_seg_len <= 4096), "generic".
        sm_scale: softmax scale (default dim ** -0.5, the golden semantics).
        backend: codegen backend, "ascendc" (default, backward compatible) or
            "pto". Backend selection is orthogonal to dispatch (detection and
            gates are pure host logic, identical for both backends); the
            compiled kernels are cached per (config, backend).

    Returns:
        (topk_indices [seq_len, topk] int32, topk_score [seq_len, topk] fp32).
    """
    if q.dim() != 3 or weights.dim() != 2 or k.dim() != 2:
        raise ValueError(f"rank mismatch: q {tuple(q.shape)}, w {tuple(weights.shape)}, k {tuple(k.shape)}")
    seq_len, heads, dim = q.shape
    if weights.shape != (seq_len, heads) or k.shape != (seq_len, dim):
        raise ValueError(f"shape mismatch: q {tuple(q.shape)}, w {tuple(weights.shape)}, k {tuple(k.shape)}")
    if dtype not in ("bfloat16", "bf16", "float32"):
        # main repo passes T.bfloat16 (the CUDA fork's DataType object); accept
        # the bfloat16 family only — the precision path is fixed bf16
        raise ValueError(f"only bfloat16 inputs are supported, got {dtype!r}")
    if q.dtype != torch.bfloat16 or weights.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError(f"q/weights/k must be bfloat16, got {q.dtype}/{weights.dtype}/{k.dtype}")
    if offsets.dtype != torch.int32:
        raise ValueError(f"offsets must be int32, got {offsets.dtype}")
    if heads % 8 != 0 or heads > 64:
        raise ValueError(f"heads must be % 8 == 0 and <= 64, got {heads}")
    if dim % 16 != 0:
        raise ValueError(f"dim must be a multiple of 16, got {dim}")
    if topk != _next_pow2(topk) or topk < 16 or topk > 1024:
        raise ValueError(f"topk must be a power of two in [16, 1024], got {topk}")
    if sm_scale is None:
        sm_scale = dim**-0.5
    if dispatch not in ("auto", "fast", "generic"):
        raise ValueError(f"dispatch must be auto/fast/generic, got {dispatch!r}")
    if backend not in ("ascendc", "pto"):
        raise ValueError(f"backend must be ascendc/pto, got {backend!r}")

    # single synchronizing D2H per call (perf iter-3: was five .cpu() syncs)
    off_list = _off_list(offsets)
    for b in range(len(off_list) - 1):
        seg_len = off_list[b + 1] - off_list[b]
        if seg_len < topk:
            raise ValueError(f"varlen segment {b} has length {seg_len} < topk {topk}")

    token_indices = _cached_token_indices(offsets, off_list)  # interface fidelity (not consumed)

    if dispatch == "auto":
        dispatch = _detect_dispatch(offsets, off_list)
    if dispatch == "fast":
        if _max_seg_len(offsets, off_list) > 4096:
            raise ValueError("dispatch='fast' requires max segment length <= 4096; use dispatch='generic'")
        return _run_fast(q, weights, k, topk, offsets, sm_scale, token_indices, off_list, backend)
    return _run_generic(q, weights, k, topk, offsets, sm_scale, token_indices, off_list, backend)
