"""sparse_mla_bwd: DSA sparse MLA backward (varlen packed, causal) for Ascend 910B3.

rev3 (Developer pair-loop restructure, DESIGN_sparse_mla_bwd.md rev3 §4-§9):
the bwd core (k2) is a flattened produce/consume STAGGERED pair-loop on the
same 4-True AUTO stack that rev2 used, expressed with zero Expert APIs
(T.alloc_shared / T.alloc_fragment + "workspace*" naming auto CV handshake):

  * global block stream: every core flattens its striped (token, kv_group)
    tasks x NI index blocks into ONE serial loop `for kb in T.serial(NB+NI)`.
    Iteration kb PRODUCES block (kb//NI, kb%NI) and CONSUMES block
    (kb//NI-1, kb%NI) — C runs one TOKEN ahead of V on the consume chain
    (indexer_bwd L512 if-guard form, DESIGN §5.4/§9.4):
      C: P(t,0..3) | C(t,0) P(t+1,0) C(t,1) P(t+1,1) ...
      V: E(t,0..3) [Delta(t) frozen] D(t,0)||E(t+1,0) D(t,1)||E(t+1,1) ...
  * all cross-core transit slots are [grid, PARITY=2, NI, ...] token-parity
    double-buffered (§8.4): the 1-token stagger distance makes every next
    writer 2 tokens away from the last reader (slot safety §9.4).
  * Delta identity (amend-1 §1.3) inside the pipeline: V reduces
    Delta[t,h] = sum_j P*dP_raw from the fp32 pre-split intermediates during
    the produce phase (per-block prefix into delta_acc), FREEZES it into the
    consume slot at the token boundary (end of iteration), and the dP chain
    of the staggered consume phase reads the frozen value — the Delta
    barrier is absorbed by the stagger instead of a second data pass.
  * P/dP transit keeps the rev2-verified hi+lo bf16 channel pairs (~16
    mantissa bits): the plain-bf16 cast is replica-attributed to fail the
    §12.2 mixed-tolerance gate (dkv ratio 0.96, debug_log attempt-1) and the
    single-bf16 downgrade is gated on a dedicated re-test (§12.6(2)-adjacent;
    debug_log attempt-2 "plain-bf16 回退评估" defers it to Stage 3).
  * dKV scatter: C flushes the G5/G6 pair sums to per-core bf16 CONTRIBUTION
    SLOTS (workspace_dkv [grid, 2, NI, BN, 576] bf16, single rounding out of
    L0C — L5 default per §6.3; env SPARSE_MLA_BWD_DKV_SLOT=fp32 selects the
    bit-exact rev2 fallback slot), V re-reads each valid row, casts fp32 and
    atomically accumulates into the fp32 terminal region (k3 rounds once).
  * dq accumulation stays V-side UB fp32 (rev2 §3.4/(c) re-proven §5.3):
    C writes per-block contributions to workspace_dq slots, V accumulates
    into acc_dq_all across the token's NI blocks and writes back at the
    token boundary.
  * OOB protection unchanged (§7): gather clamps rows for addressing only,
    mask zeroes contributions, scatter rows are conditionally skipped using
    the RAW index (no OOB writes); repeated indices not deduplicated.
  * AUTO_CV_SYNC pairing (pair = matched TIR-level workspace GM copy
    statement per side, in order; counts must be equal): kv 4 (produce
    batch+mini, consume dummies) + s 2 (produce write, consume dummy) +
    dp 2 + p 2 + dpb 2 + dq 2 + dkv 2 = 16 pairs = the hardware flag-id
    limit (ids 0..15; V reads ws_s/ws_dp twice — produce P chain + consume
    P recompute — so C emits scratch-slot dummy writes to balance).
  * inherited rev2 components (§4.8): ws_kv per-block stash + gather
    full+mini staging, scatter payload slimming (idx_ub-sourced positions),
    P/dP_raw V-only GM stash + zero-flag second read, k3 row width 32,
    mask/bias vectorization (EG-row group broadcast), host two-launch
    orchestration (delta=None skips k1 entirely, amend-1).

Semantics (migrated from ref_gpu/sparse_mla_bwd.py, authoritative):
  * P = exp(score*sm_scale - lse) (masked -> 0); dP = P*(dO.v - Delta)*sm_scale;
    dq += dP @ kv_rows (fp32 UB, cross-block); dKV[idx] += dP^T @ q + P^T @ dO
    (fp32 GM atomic, one bf16 rounding in postprocess).
  * is_casual (main-repo spelling) must be True; causal test uses in-segment
    position s_i; varlen via TokenIndices/Offsets GM scalar reads.

Support domain (host asserts): D=512 hardcoded (D_tail = dim-512);
head_kv = H // kv_group in {16, 32, 48, 64}; topk % 128 == 0; dtypes q/kv/o/do
bf16, indices/offsets int32, lse/delta fp32.
"""

import os

import tilelang
from tilelang import language as T

__all__ = [
    "sparse_mla_bwd",
    "tl_sparse_mla_bwd_preprocess",
    "tl_sparse_mla_bwd_postprocess",
    "prepare_token_indices",
    "get_cube_core_num",
]

BIGC = 1.0e30  # mask invalid-slot negative bias (finite; exp underflows to 0)

_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

# L5 contribution-slot档 (DESIGN §6.3/§12.6(2)): fp32 fallback is the DEFAULT —
# the bf16 slot (26MB readback) FAILED the §12.1 dual gate on 2026-10-15
# (domain-A dkv mixed-tolerance ratio 0.965-0.983 < 0.99 while sim passes at
# 0.999997; R14 门控回退 per §14.3 — L0C single-rounding grain is too coarse
# for the per-element tolerance). SPARSE_MLA_BWD_DKV_SLOT=bf16 re-enables it
# for Stage-3 re-evaluation; the fp32 slot's AIV readback is ~53MB (design
# §13.4: L5 fp32 兜底档 AIV 读量 88-96MB, disclosed overage).
_DKV_SLOT_BF16 = os.environ.get("SPARSE_MLA_BWD_DKV_SLOT", "fp32").strip().lower() == "bf16"


def _derived_bwd_layout(heads, kv_group, topk, block_N):
    """Compile-time layout derivation shared by builder and host asserts."""
    assert heads % kv_group == 0, f"heads={heads} must be divisible by kv_group={kv_group}"
    head_kv = heads // kv_group
    assert head_kv % 16 == 0 and 16 <= head_kv <= 64, (
        f"head_kv={head_kv} unsupported: need % 16 == 0 and 16 <= head_kv <= 64 (no head-padding path; head_kv > 64 out of scope for bwd)"
    )
    assert topk % block_N == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    v_block = head_kv // 2
    assert v_block >= 1, "head_kv >= 16 guarantees v_block >= 8"
    return head_kv, v_block


# ============================================================================
# k1 preprocess: Delta[t, h] = sum_d O . dO  (pure-AIV persistent kernel)
# Retained for the explicit-delta / diagnostic path only (amend-1 §13.1: the
# default host flow never launches it — Delta is reduced inside k2).
# ============================================================================


def _make_preprocess_builder(target):
    """One @tilelang.jit instance per target (decorator fixes target)."""

    @tilelang.jit(pass_configs=_pass_configs, target=target)
    def _preprocess_builder(dim=512, grid_size=20, rows_per_aiv=16):
        """AIV-only kernel: flat (s_row, head) rows of O/dO [rows, dim] bf16
        -> Delta [rows] fp32 (row-wise fp32 dot). Task = 2*rows_per_aiv flat
        rows, striped over grid; AIV-vid handles its half rows.

        Delta is declared with its own symbolic extent so the host can pass a
        PADDED buffer (>= o_rows + 2*rows_per_aiv): the varlen tail branch
        writes whole [rows_per_aiv] vectors whose invalid rows land in the
        pad (values never read by bwd).
        """
        o_rows = T.symbolic("o_rows")
        do_rows = T.symbolic("do_rows")
        delta_rows = T.symbolic("delta_rows")

        @T.prim_func
        def main(
            O: T.Tensor([o_rows, dim], "bfloat16"),  # type: ignore
            dO: T.Tensor([do_rows, dim], "bfloat16"),  # type: ignore
            Delta: T.Tensor([delta_rows], "float32"),  # type: ignore
        ):
            with T.Kernel(grid_size, is_npu=True) as (cid, vid):
                o_ub = T.alloc_ub([rows_per_aiv, dim], "bfloat16")
                o_f32 = T.alloc_ub([rows_per_aiv, dim], "float32")
                do_f32 = T.alloc_ub([rows_per_aiv, dim], "float32")
                delta_ub = T.alloc_ub([rows_per_aiv], "float32")

                rows_per_task = 2 * rows_per_aiv
                total_tasks = T.ceildiv(o_rows, rows_per_task)
                ntask = T.max(0, T.ceildiv(total_tasks - cid, grid_size))
                for t in T.serial(ntask):
                    c = t * grid_size + cid
                    r0 = c * rows_per_task
                    r_start = r0 + vid * rows_per_aiv
                    if r_start + rows_per_aiv <= o_rows:
                        # whole half-tile: contiguous [rows, dim] chunk
                        T.copy(O[r_start : r_start + rows_per_aiv, 0:dim], o_ub)
                        T.copy(o_ub, o_f32)  # bf16 -> fp32
                        T.copy(dO[r_start : r_start + rows_per_aiv, 0:dim], o_ub)
                        T.copy(o_ub, do_f32)
                        T.tile.mul(o_f32, o_f32, do_f32)
                        T.reduce_sum(o_f32, delta_ub, dim=-1)
                        T.copy(delta_ub, Delta[r_start : r_start + rows_per_aiv])
                    else:
                        # varlen tail: zero-fill first (invalid rows -> 0),
                        # guarded per-row loads, whole-vector store into pad
                        T.tile.fill(o_f32, 0.0)
                        T.tile.fill(do_f32, 0.0)
                        for u in range(rows_per_aiv):
                            if r_start + u < o_rows:
                                T.copy(O[r_start + u, 0:dim], o_ub[u, 0:dim])
                                T.copy(o_ub[u, 0:dim], o_f32[u, 0:dim])
                                T.copy(dO[r_start + u, 0:dim], o_ub[u, 0:dim])
                                T.copy(o_ub[u, 0:dim], do_f32[u, 0:dim])
                        T.tile.mul(o_f32, o_f32, do_f32)
                        T.reduce_sum(o_f32, delta_ub, dim=-1)
                        T.copy(delta_ub, Delta[r_start : r_start + rows_per_aiv])

        return main

    return _preprocess_builder


def tl_sparse_mla_bwd_preprocess(dim=512, grid_size=20, target="ascendc"):
    """Build the preprocess kernel for the requested backend.

    Returns: func(O_flat [rows, dim] bf16, dO_flat [rows, dim] bf16,
    Delta_padded [rows + pad] fp32) — flat views supplied by the host runner.
    """
    return _make_preprocess_builder(target)(dim=dim, grid_size=grid_size)


# ============================================================================
# k3 postprocess: dKV fp32 -> dkv bf16 (single rounding, pure-AIV)
# rev2 form inherited verbatim (§11.2: row width 32, zero rev3 changes).
# ============================================================================


def _make_postprocess_builder(target):
    @tilelang.jit(pass_configs=_pass_configs, target=target)
    def _postprocess_builder(ddt=576, grid_size=20, rows_per_aiv=32):
        """AIV-only cast kernel: flat rows [S*kv_group, ddt] fp32 -> bf16.

        Single fp32->bf16 rounding of the fp32 accumulation buffer (semantic
        anchor; indexer_bwd tl_dk_cast form).
        """
        src_rows = T.symbolic("src_rows")
        dst_rows = T.symbolic("dst_rows")

        @T.prim_func
        def main(
            dKV: T.Tensor([src_rows, ddt], "float32"),  # type: ignore
            dKV_out: T.Tensor([dst_rows, ddt], "bfloat16"),  # type: ignore
        ):
            with T.Kernel(grid_size, is_npu=True) as (cid, vid):
                src_ub = T.alloc_ub([rows_per_aiv, ddt], "float32")
                dst_ub = T.alloc_ub([rows_per_aiv, ddt], "bfloat16")

                rows_per_task = 2 * rows_per_aiv
                total_tasks = T.ceildiv(src_rows, rows_per_task)
                ntask = T.max(0, T.ceildiv(total_tasks - cid, grid_size))
                for t in T.serial(ntask):
                    c = t * grid_size + cid
                    r0 = c * rows_per_task
                    r_start = r0 + vid * rows_per_aiv
                    if r_start + rows_per_aiv <= src_rows:
                        T.copy(dKV[r_start : r_start + rows_per_aiv, 0:ddt], src_ub)
                        T.copy(src_ub, dst_ub)  # fp32 -> bf16 (single rounding)
                        T.copy(dst_ub, dKV_out[r_start : r_start + rows_per_aiv, 0:ddt])
                    else:
                        for u in range(rows_per_aiv):
                            if r_start + u < src_rows:
                                T.copy(dKV[r_start + u, 0:ddt], src_ub[u, 0:ddt])
                                T.copy(src_ub[u, 0:ddt], dst_ub[u, 0:ddt])
                                T.copy(dst_ub[u, 0:ddt], dKV_out[r_start + u, 0:ddt])

        return main

    return _postprocess_builder


def tl_sparse_mla_bwd_postprocess(ddt=576, grid_size=20, target="ascendc"):
    """Build the postprocess kernel for the requested backend.

    Returns: func(dKV_flat [rows, ddt] fp32, dkv_flat [rows, ddt] bf16).
    """
    return _make_postprocess_builder(target)(ddt=ddt, grid_size=grid_size)


# ============================================================================
# k2 bwd: persistent CV-fused core kernel (rev3 pair-loop form)
# ============================================================================


def _make_bwd_builder(target):
    @tilelang.jit(pass_configs=_pass_configs, target=target)
    def _sparse_mla_bwd_builder(
        heads,
        dim,
        tail_dim,
        topk,
        kv_group=1,
        sm_scale=None,
        block_N=128,
        grid_size=20,
        delta_internal=False,
        dkv_slot_bf16=False,
    ):
        """Build the bwd core kernel (Developer mode, persistent MIX_AIC_1_2).

        rev3 structure (DESIGN §5.4/§9.4): one flattened serial loop over the
        core's block stream; iteration kb produces block pi of token pt =
        kb//NI and consumes block pi of token pt-1. All transit slots are
        token-parity double-buffered [grid, 2, NI, ...]; q/dO L1 slots are
        parity-flattened leading dims. GEMM segmentation follows DESIGN §5.2
        (DSEG=2: D=512 as N_SEG=2 x DS=256 column segments + 64-wide tail).

        Args:
            heads: total head count H (head_kv = heads // kv_group, no pad).
            dim: QK main / V dimension D (host hardcodes 512).
            tail_dim: QK tail dimension (576 - dim).
            topk: selected positions per row (topk % block_N == 0).
            kv_group: KV group count G.
            sm_scale: defaults to (dim + tail_dim) ** -0.5.
            block_N: topk-direction tile BN.
            grid_size: persistent grid (910B3 = 20).
            delta_internal: DESIGN amend-1 §1.3 — reduce Delta = sum_j P*dP_raw
                inside the kernel from the fp32 pre-split intermediates (the
                GM Delta input is never read; host passes a placeholder).
                False consumes the GM Delta verbatim (explicit-delta path).
            dkv_slot_bf16: L5 contribution-slot档 (§6.3): True flushes the
                G5/G6 sums to bf16 slots (single rounding out of L0C, 26MB
                V-side readback); False keeps the rev2 fp32 slots.
        """
        assert dim == 512 and tail_dim == 64, f"(dim={dim}, tail_dim={tail_dim}) unsupported: fixed (512, 64) layout"
        head_kv, v_block = _derived_bwd_layout(heads, kv_group, topk, block_N)
        if sm_scale is None:
            sm_scale = (dim + tail_dim) ** (-0.5)

        M = head_kv  # GEMM M dimension (padded_H == head_kv, no pad path)
        BN = block_N
        DS = 256  # DSEG=2 column segment width (DESIGN §5.2)
        N_SEG = dim // DS
        NI = topk // BN
        HALF = BN // 2  # gather/scatter rows per AIV
        DDT = dim + tail_dim
        DCH = dim // BN  # V-side dq accumulate chunks (BN-wide)
        GB = 16  # gather staging batch rows (DESIGN §8.3 reduction chain step 1)
        EG = 8  # mask/bias broadcast row-group (fwd iter-3 vectorized form)
        NGRP = v_block // EG  # v_block in {8,16,24,32} -> NGRP in {1,2,3,4}
        PARITY = 2  # token-parity double buffering (pair-loop DEP=1 token)
        QSEG = PARITY * N_SEG  # q/do L1 parity*segment flattened leading dim
        assert dim % DS == 0 and DS % 16 == 0 and tail_dim % 16 == 0
        assert v_block % EG == 0, "v_block must be a multiple of the 8-row broadcast group"

        seq_len = T.symbolic("seq_len")
        seq_len_kv = T.symbolic("seq_len_kv")
        batch_plus_one = T.symbolic("batch_plus_one")

        dtype = "bfloat16"
        indices_dtype = "int32"
        accum_dtype = "float32"
        dkv_slot_dtype = dtype if dkv_slot_bf16 else accum_dtype

        q_shape = [seq_len, heads, DDT]
        kv_shape = [seq_len_kv, kv_group, DDT]
        do_shape = [seq_len, heads, dim]
        indices_shape = [seq_len, kv_group, topk]
        lse_shape = [seq_len, heads]
        offsets_shape = [batch_plus_one]
        token_indices_shape = [seq_len, 2]
        dkv_rows = seq_len_kv * kv_group  # flat atomic target rows

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            KV: T.Tensor(kv_shape, dtype),  # type: ignore
            dO: T.Tensor(do_shape, dtype),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
            Delta: T.Tensor(lse_shape, accum_dtype),  # type: ignore
            Offsets: T.Tensor(offsets_shape, indices_dtype),  # type: ignore
            TokenIndices: T.Tensor(token_indices_shape, indices_dtype),  # type: ignore
            dQ: T.Tensor(q_shape, dtype),  # type: ignore
            dKV: T.Tensor([dkv_rows, DDT], accum_dtype),  # type: ignore  # flat fp32 accum
            # [grid, PARITY, NI+1, BN, DDT]: NI per-block stash slots + 1
            # scratch slot absorbing the consume-side CV-pairing dummy writes
            workspace_kv: T.Tensor([grid_size, PARITY, NI + 1, BN, DDT], dtype),  # type: ignore
            # [grid, 2, NI+1, M, BN]: NI data slots + 1 scratch slot for the
            # consume-side C dummy writes (balances V's second ws_s/ws_dp read
            # of the P-recompute path; AUTO_CV_SYNC counts must match)
            workspace_s: T.Tensor([grid_size, PARITY, NI + 1, M, BN], accum_dtype),  # type: ignore
            workspace_dp: T.Tensor([grid_size, PARITY, NI + 1, M, BN], accum_dtype),  # type: ignore
            # P / dP hi+lo bf16 channel pairs (rev2-verified precision path)
            workspace_p: T.Tensor([grid_size, PARITY, NI, 2, M, BN], dtype),  # type: ignore
            workspace_dpb: T.Tensor([grid_size, PARITY, NI, 2, M, BN], dtype),  # type: ignore
            workspace_dq: T.Tensor([grid_size, PARITY, NI, M, DDT], accum_dtype),  # type: ignore
            # dKV contribution slots (L5): bf16 default / fp32 fallback
            workspace_dkv: T.Tensor([grid_size, PARITY, NI, BN, DDT], dkv_slot_dtype),  # type: ignore
        ):
            with T.Kernel(grid_size, is_npu=True) as (cid, vid):
                # ======== Cube side: L1 GEMM operands (alloc_shared -> L1) ========
                v_all = T.alloc_shared([N_SEG, BN, DS], dtype)
                tail_l1 = T.alloc_shared([BN, tail_dim], dtype)
                # q/do: token-parity double slots (§8.2: the stagger keeps
                # token t's consume GEMMs alive while t+1's produce loads),
                # parity*segment flattened into one leading dim (3D buffers,
                # leading-dim slicing only — offset != 0 inner regions are
                # broken on both backends, rev2 R1 evidence)
                q_all = T.alloc_shared([QSEG, M, DS], dtype)
                qt_l1 = T.alloc_shared([PARITY, M, tail_dim], dtype)
                do_all = T.alloc_shared([QSEG, M, DS], dtype)
                p_hi_l1 = T.alloc_shared([M, BN], dtype)
                p_lo_l1 = T.alloc_shared([M, BN], dtype)
                dp_hi_l1 = T.alloc_shared([M, BN], dtype)
                dp_lo_l1 = T.alloc_shared([M, BN], dtype)
                # ======== L0C accumulators (alloc_fragment; §8.1 time-sharing:
                # acc_sd (produce G1/G2) dies at its workspace writeout before
                # the consume GEMMs; peak = acc_dkv [BN, DS] 128KB single-slot
                # segment, program-order liveness planned) ========
                acc_sd = T.alloc_fragment([M, BN], accum_dtype)
                acc_dq = T.alloc_fragment([M, DS], accum_dtype)
                acc_dqt = T.alloc_fragment([M, tail_dim], accum_dtype)
                acc_dkv = T.alloc_fragment([BN, DS], accum_dtype)
                acc_dkvt = T.alloc_fragment([BN, tail_dim], accum_dtype)
                # ======== Vector side: UB (alloc_shared -> UB) ========
                # rev3 no-alias UB budget (fix #3, dbg evidence 2026-10-15):
                # the linear-scan UB planner aliases liveness-disjoint slots
                # (observed in compiled artifacts: kv_stage<->acc_dq_all,
                # kv_stage<->lse_2d, acc_dq_all<->dqt_bf16) while the
                # sync-insert pass tracks per-buffer names per if-branch —
                # aliased buffers therefore carry NO pipe ordering and a
                # gather can overwrite a live accumulator with raw kv rows.
                # This slot set sums to ~192KB <= the 196352B usable UB,
                # forcing a fully disjoint (alias-free) allocation:
                #  * kv_stage is batch-16 [GB, DDT] (DESIGN §8.3 reduction
                #    chain step 1); the batch loop keeps the ws_kv stash at
                #    ONE TIR copy statement + the trailing mini rewrite.
                #  * the P/dP lo channels are written directly from the fp32
                #    residual in lse_2d (T.copy auto-cast) — no lo staging
                #    slots; only the hi channels stage through p_hi_ub /
                #    dp_hi_ub (V-write -> MTE3-read, same-branch synced).
                #  * the P/dP_raw fp32 GM stash (rev2 pst mechanism) is GONE:
                #    the consume segment re-reads ws_s/ws_dp (cross-core
                #    flagged; C balanced by scratch-slot dummy writes) and
                #    recomputes the P chain bit-identically — the un-flagged
                #    same-core MTE3->MTE2 pst transit was a race source.
                #  * lse_ub (produce) / lse_ub2 (consume recompute) split so
                #    the produce's lse values never race the consume read.
                #  * produce/consume segment ends carry MTE2 drain probes
                #    (P1/P2: a GM->UB copy into a segment-MTE3-read buffer):
                #    the probe's MTE3->MTE2 hard event (inserted by the sync
                #    pass for the same-branch RAW) stalls the scalar stream
                #    until the V and MTE3 pipes drain, ordering every
                #    cross-branch reuse of s_ub/lse_2d/idx_ub etc.
                acc_dq_all = T.alloc_shared([DCH, v_block, BN], accum_dtype)
                acc_dqt_ub = T.alloc_shared([v_block, tail_dim], accum_dtype)
                s_ub = T.alloc_shared([v_block, BN], accum_dtype)  # P chain / dq chunk slot
                dp_ub = T.alloc_shared([v_block, BN], accum_dtype)
                p_hi_ub = T.alloc_shared([v_block, BN], dtype)  # P hi staging (produce)
                dp_hi_ub = T.alloc_shared([v_block, BN], dtype)  # dP hi staging (consume)
                cast_bf16 = T.alloc_shared([v_block, BN], dtype)  # dq writeback cast
                lse_2d = T.alloc_shared([v_block, BN], accum_dtype)  # broadcast / residual scratch
                tail_slot = T.alloc_shared([v_block, tail_dim], accum_dtype)
                dqt_bf16 = T.alloc_shared([v_block, tail_dim], dtype)
                lse_ub = T.alloc_shared([v_block], accum_dtype)  # produce lse
                lse_ub2 = T.alloc_shared([v_block], accum_dtype)  # consume recompute lse
                # amend-1 §1.3: per-task Delta prefix accumulator (produce
                # side) + frozen consume slot (filled at the token boundary
                # for the internal path / from GM for the explicit path)
                delta_acc = T.alloc_shared([v_block], accum_dtype)
                delta_cons = T.alloc_shared([v_block], accum_dtype)
                delta_vec = T.alloc_shared([v_block], accum_dtype)
                idx_ub = T.alloc_shared([BN], indices_dtype)
                # mask/bias row-group broadcast scratch (fwd iter-3 form)
                bcast_h = T.alloc_shared([EG, BN], accum_dtype)
                idx_f = T.alloc_shared([BN], accum_dtype)
                m1_ub = T.alloc_shared([BN], accum_dtype)
                m2_ub = T.alloc_shared([BN], accum_dtype)
                t_ub = T.alloc_shared([BN], accum_dtype)
                mask_ub = T.alloc_shared([BN], accum_dtype)
                bias_ub = T.alloc_shared([BN], accum_dtype)
                rows_int = T.alloc_shared([BN], indices_dtype)
                kv_stage = T.alloc_shared([GB, DDT], dtype)  # batch-16 gather staging
                kv_ub = T.alloc_shared([DDT], dtype)  # dummy-write source
                row_ub = T.alloc_shared([DDT], accum_dtype)  # scatter row (fp32 atomic src)
                row_bf16 = T.alloc_shared([DDT], dtype)  # bf16 slot row buffer

                # ======== flattened produce/consume pair-loop (§5.4) ========
                total = seq_len * kv_group
                ntask = T.max(0, T.ceildiv(total - cid, grid_size))
                NB = ntask * NI  # this core's produce block count
                for kb in T.serial(NB + NI):
                    pt = kb // NI  # produce task (per-core striped index)
                    pi = kb % NI  # produce block
                    pslot = pt % 2
                    # ---- dq accumulator handoff (rev3 UB-planner fix): clear
                    # for the consume task whose FIRST block lands at this
                    # iteration — placed at the TOP of the body so the
                    # accumulators' statement-level liveness starts BEFORE the
                    # gather (the linear-scan UB planner otherwise aliases
                    # acc_dq_all with kv_stage — their uses are produce-vs-
                    # consume disjoint within one iteration and the planner
                    # does not model cross-iteration liveness through the
                    # symbolic loop; the gather then overwrites the
                    # accumulator with raw kv values, dbg evidence
                    # 2026-10-15). The previous task's boundary writeback ran
                    # at the END of the previous iteration — order preserved.
                    if (kb >= NI) & (pi == 0):
                        for d in range(DCH):
                            T.tile.fill(acc_dq_all[d, :, :], 0.0)
                        T.tile.fill(acc_dqt_ub, 0.0)
                    # ================= PRODUCE block (pt, pi) =================
                    if kb < NB:
                        pid_p = pt * grid_size + cid
                        s_row_p = pid_p % seq_len
                        g_p = pid_p // seq_len
                        H0p = g_p * M
                        b_p = TokenIndices[s_row_p, 0]
                        s_i_p = TokenIndices[s_row_p, 1]
                        bos_p = Offsets[b_p]
                        qb = pslot * N_SEG  # q/do L1 slot base for token pt
                        # ---- [C] task-boundary Q/dO load (token-parity slot) ----
                        if pi == 0:
                            for s in range(N_SEG):
                                T.copy(
                                    Q[s_row_p, H0p : H0p + M, s * DS : (s + 1) * DS],
                                    q_all[qb + s, :, :],
                                )
                            T.copy(Q[s_row_p, H0p : H0p + M, dim:DDT], qt_l1[pslot, :, :])
                            for s in range(N_SEG):
                                T.copy(
                                    dO[s_row_p, H0p : H0p + M, s * DS : (s + 1) * DS],
                                    do_all[qb + s, :, :],
                                )
                        # ---- [C] kv block stash -> L1 (produce read #1) ----
                        for s in range(N_SEG):
                            T.copy(
                                workspace_kv[cid, pslot, pi, 0:BN, s * DS : (s + 1) * DS],
                                v_all[s, :, :],
                            )
                        T.copy(workspace_kv[cid, pslot, pi, 0:BN, dim:DDT], tail_l1)
                        # ---- [C] G1 score -> workspace_s (K=576 init chain) ----
                        T.gemm_v0(
                            q_all[qb, :, :],
                            v_all[0, :, :],
                            acc_sd,
                            transpose_B=True,
                            init=True,
                        )
                        for s in range(1, N_SEG):
                            T.gemm_v0(
                                q_all[qb + s, :, :],
                                v_all[s, :, :],
                                acc_sd,
                                transpose_B=True,
                                init=False,
                            )
                        T.gemm_v0(qt_l1[pslot, :, :], tail_l1, acc_sd, transpose_B=True, init=False)
                        T.copy(acc_sd, workspace_s[cid, pslot, pi, 0:M, 0:BN])
                        # ---- [C] G2 dO.v -> workspace_dp (dP_raw) ----
                        T.gemm_v0(
                            do_all[qb, :, :],
                            v_all[0, :, :],
                            acc_sd,
                            transpose_B=True,
                            init=True,
                        )
                        for s in range(1, N_SEG):
                            T.gemm_v0(
                                do_all[qb + s, :, :],
                                v_all[s, :, :],
                                acc_sd,
                                transpose_B=True,
                                init=False,
                            )
                        T.copy(acc_sd, workspace_dp[cid, pslot, pi, 0:M, 0:BN])
                        # ---- [V] task-boundary scalars ----
                        if pi == 0:
                            T.copy(
                                Lse[s_row_p, H0p + vid * v_block : H0p + (vid + 1) * v_block],
                                lse_ub,
                            )
                            if delta_internal:
                                # amend-1 §1.3: Delta reduced across this
                                # task's NI blocks (frozen at the boundary)
                                T.tile.fill(delta_acc, 0.0)
                        # ---- [V] index block + mask chain (rev2 arithmetic
                        # form: mask = 1{0 <= idx <= s_i}; rows = clamp for
                        # addressing only; negative values incl. -1 masked) ----
                        T.copy(Indices[s_row_p, g_p, pi * BN : (pi + 1) * BN], idx_ub)
                        T.copy(idx_ub, idx_f)  # int32 -> fp32 auto-cast
                        # m1 = 1{idx <= s_i}: (s_i+1) - idx, clamp01
                        T.tile.mul(m1_ub, idx_f, T.float32(-1.0))
                        T.tile.add(m1_ub, m1_ub, T.float32(s_i_p + 1))
                        T.tile.max(m1_ub, m1_ub, T.float32(0.0))
                        T.tile.min(m1_ub, m1_ub, T.float32(1.0))
                        # m2 = 1{idx >= 0}: idx + 1, clamp01
                        T.tile.add(t_ub, idx_f, T.float32(1.0))
                        T.tile.max(m2_ub, t_ub, T.float32(0.0))
                        T.tile.min(m2_ub, m2_ub, T.float32(1.0))
                        T.tile.mul(mask_ub, m1_ub, m2_ub)
                        # bias = (1 - mask) * BIGC (m1_ub dead -> reuse)
                        T.tile.mul(m1_ub, mask_ub, T.float32(-1.0))
                        T.tile.add(m1_ub, m1_ub, T.float32(1.0))
                        T.tile.mul(bias_ub, m1_ub, T.float32(BIGC))
                        # rows = clamp(bos + idx, 0, seq_len_kv - 1)
                        T.tile.add(idx_f, idx_f, T.float32(bos_p))
                        T.tile.max(idx_f, idx_f, T.float32(0.0))
                        T.tile.min(idx_f, idx_f, T.float32(seq_len_kv - 1))
                        T.copy(idx_f, rows_int)  # fp32 -> int32 (values exact)
                        # ---- [V] gather -> ws_kv stash slot (batch-16 staging,
                        # DESIGN §8.3-①: per-row GM reads land in the [GB, DDT]
                        # kv_stage, one 2D MTE3 burst per batch writes the AIV's
                        # half-block, plus a trailing idempotent 1-row rewrite
                        # pairing with the C-side tail read; the batch loop is
                        # ONE TIR copy statement so the ws_kv CV pairing count
                        # is unchanged) ----
                        for b in range(HALF // GB):
                            for j in range(GB):
                                kv_row = rows_int[vid * HALF + b * GB + j]
                                T.copy(KV[kv_row, g_p, :], kv_stage[j, 0:DDT])
                            T.copy(
                                kv_stage,
                                workspace_kv[
                                    cid,
                                    pslot,
                                    pi,
                                    vid * HALF + b * GB : vid * HALF + (b + 1) * GB,
                                    0:DDT,
                                ],
                            )
                        T.copy(
                            kv_stage[GB - 1, :],
                            workspace_kv[cid, pslot, pi, (vid + 1) * HALF - 1, 0:DDT],
                        )
                        # ---- [V] P chain: read C's score/dP_raw (fp32), exp ----
                        T.copy(
                            workspace_s[cid, pslot, pi, vid * v_block : (vid + 1) * v_block, 0:BN],
                            s_ub,
                        )
                        T.copy(
                            workspace_dp[cid, pslot, pi, vid * v_block : (vid + 1) * v_block, 0:BN],
                            dp_ub,
                        )
                        T.tile.mul(s_ub, s_ub, sm_scale)
                        # vectorized mask/bias apply (fwd iter-3 form, bit-exact)
                        T.tile.broadcast(bcast_h, mask_ub, axis=0)
                        for g in range(NGRP):
                            T.tile.mul(
                                s_ub[g * EG : (g + 1) * EG, :],
                                s_ub[g * EG : (g + 1) * EG, :],
                                bcast_h,
                            )
                        T.tile.broadcast(bcast_h, bias_ub, axis=0)  # overwrite
                        for g in range(NGRP):
                            T.tile.sub(
                                s_ub[g * EG : (g + 1) * EG, :],
                                s_ub[g * EG : (g + 1) * EG, :],
                                bcast_h,
                            )
                        T.tile.broadcast(lse_2d, lse_ub, axis=1)
                        T.tile.sub(s_ub, s_ub, lse_2d)
                        T.tile.exp(s_ub, s_ub)  # P (fp32, masked -> 0)
                        # ---- [V] Delta prefix (amend-1: fp32 pre-split values) ----
                        if delta_internal:
                            T.tile.mul(lse_2d, s_ub, dp_ub)  # P*dP_raw (fp32)
                            T.reduce_sum(lse_2d, delta_vec, dim=-1)  # row partial
                            T.tile.add(delta_acc, delta_acc, delta_vec)
                        # ---- [V] P -> hi+lo bf16 channels: hi stages through
                        # p_hi_ub, lo through cast_bf16 (both V-write ->
                        # MTE3-read, same-branch synced; UB->GM copies do not
                        # auto-cast, so the fp32 residual round-trips through
                        # a bf16 UB staging slot). s_ub is never written after
                        # this point in the segment (a V-pipe tile-op write to
                        # a UB slot an in-flight MTE3 copy reads is not
                        # ordered by the auto-sync pass, dbg 2026-10-15) ----
                        T.copy(s_ub, p_hi_ub)  # P_hi (bf16 cast)
                        T.copy(
                            p_hi_ub,
                            workspace_p[cid, pslot, pi, 0, vid * v_block : (vid + 1) * v_block, 0:BN],
                        )
                        T.copy(p_hi_ub, lse_2d)  # hi -> fp32 (exact)
                        T.tile.sub(lse_2d, s_ub, lse_2d)  # residual -> lse_2d (s_ub read-only)
                        T.copy(lse_2d, cast_bf16)  # P_lo (bf16 cast)
                        T.copy(
                            cast_bf16,
                            workspace_p[cid, pslot, pi, 1, vid * v_block : (vid + 1) * v_block, 0:BN],
                        )
                        # ---- [V] P1 drain probes (rev3 fix #4): the if-guarded
                        # structure clears the sync-insert pass's access
                        # history at every branch boundary, so cross-branch
                        # same-core pipe dependencies (this segment's MTE3
                        # reads vs the consume segment's MTE2/V writes; the
                        # stash MTE3 read vs the NEXT gather's kv_stage MTE2
                        # writes) carry no hardware ordering. Two probes at
                        # the segment end restore it, exploiting the
                        # same-branch RAW the pass DOES sync:
                        #   P1a: GM->UB write into kv_stage (an MTE3-read
                        #        buffer) -> MTE3->MTE2 hard event (pipe-
                        #        granular: every later MTE2 op waits the MTE3
                        #        drain; the next gather's kv_stage writes are
                        #        MTE2-queue-ordered behind it).
                        #   P1b: UB->UB copy cast_bf16 -> lse_2d (cast_bf16 is
                        #        an MTE3-read buffer) -> MTE3->V hard event
                        #        (the V queue stalls until MTE3 drains; every
                        #        later V op — the consume segment's tile ops
                        #        on s_ub/dp_ub/lse_2d/idx_f — queues behind).
                        # The probe targets' next uses are MTE2-ordered
                        # (kv_stage) / V-ordered (lse_2d) respectively, so
                        # the probe writes themselves cannot corrupt them. ----
                        #   P1b REMOVED (iter3b V1): the produce-tail
                        #   UB->UB probe copy (cast_bf16 -> lse_2d) only ever
                        #   lowered to a PIPE_V barrier (kernel_src.c rev3:
                        #   no MTE3_V pair is generated — a V-pipe READ of
                        #   cast_bf16 vs the MTE3 READ of cast_bf16 is
                        #   read-read, and lse_2d is only ever touched by V
                        #   ops in both segments, which are queue-ordered);
                        #   its 22MB/iter-tail UB traffic + V barrier were
                        #   pure overhead. P1a (MTE3_MTE2) and P2/P3
                        #   (MTE3_V) carry all real cross-pipe ordering.
                        T.copy(Q[s_row_p, H0p + vid * v_block, 0:DDT], kv_stage[0, 0:DDT])
                    # ================= CONSUME block (pt-1, pi) =================
                    if kb >= NI:
                        cbi = kb - NI
                        ct = cbi // NI  # consume task
                        ci = cbi % NI  # consume block (same block index)
                        cslot = ct % 2
                        pid_c = ct * grid_size + cid
                        s_row_c = pid_c % seq_len
                        g_c = pid_c // seq_len
                        H0c = g_c * M
                        b_c = TokenIndices[s_row_c, 0]
                        s_i_c = TokenIndices[s_row_c, 1]
                        bos_c = Offsets[b_c]
                        # ---- [C] CV-pairing dummy writes (ws_s/ws_dp scratch
                        # slot NI, never read by V's data path: V reads the
                        # ws_s/ws_dp data slots TWICE per iteration — produce
                        # P chain + consume P recompute — so C emits a second
                        # (dummy) write per buffer to balance the AUTO_CV_SYNC
                        # statement counts; placed at the TOP of C's consume
                        # segment so the pair-2 flag fires early and V's
                        # consume read is not pushed behind C's GEMMs, which
                        # would deadlock against the dpb dependency) ----
                        T.copy(acc_sd, workspace_s[cid, cslot, NI, 0:M, 0:BN])
                        T.copy(acc_sd, workspace_dp[cid, cslot, NI, 0:M, 0:BN])
                        # ---- [C] kv block stash -> L1 (consume read #2; §5.4
                        # kv double read: L1 cannot hold NI blocks of 576KB) ----
                        for s in range(N_SEG):
                            T.copy(
                                workspace_kv[cid, cslot, ci, 0:BN, s * DS : (s + 1) * DS],
                                v_all[s, :, :],
                            )
                        T.copy(workspace_kv[cid, cslot, ci, 0:BN, dim:DDT], tail_l1)
                        # ---- [C] P/dP hi+lo -> L1 ----
                        T.copy(workspace_p[cid, cslot, ci, 0, 0:M, 0:BN], p_hi_l1)
                        T.copy(workspace_p[cid, cslot, ci, 1, 0:M, 0:BN], p_lo_l1)
                        T.copy(workspace_dpb[cid, cslot, ci, 0, 0:M, 0:BN], dp_hi_l1)
                        T.copy(workspace_dpb[cid, cslot, ci, 1, 0:M, 0:BN], dp_lo_l1)
                        # ---- [C] G3 dq main (per DSEG) -> workspace_dq ----
                        # dq passed the bf16 gate with the plain-bf16 dP grain
                        # (hi channel); the lo channel rides the dkv path only
                        for s in range(N_SEG):
                            T.gemm_v0(dp_hi_l1, v_all[s, :, :], acc_dq, init=True)
                            T.copy(
                                acc_dq,
                                workspace_dq[cid, cslot, ci, 0:M, s * DS : (s + 1) * DS],
                            )
                        # ---- [C] G4 dq tail -> workspace_dq ----
                        T.gemm_v0(dp_hi_l1, tail_l1, acc_dqt, init=True)
                        T.copy(acc_dqt, workspace_dq[cid, cslot, ci, 0:M, dim:DDT])
                        # ---- [C] G5 dkv pair -> contribution slots (hi+lo
                        # double-GEMM recombination:
                        # dkv = (dP_hi+dP_lo)^T @ q + (P_hi+P_lo)^T @ dO) ----
                        for s in range(N_SEG):
                            T.gemm_v0(
                                dp_hi_l1,
                                q_all[cslot * N_SEG + s, :, :],
                                acc_dkv,
                                transpose_A=True,
                                init=True,
                            )
                            T.gemm_v0(
                                dp_lo_l1,
                                q_all[cslot * N_SEG + s, :, :],
                                acc_dkv,
                                transpose_A=True,
                                init=False,
                            )
                            T.gemm_v0(
                                p_hi_l1,
                                do_all[cslot * N_SEG + s, :, :],
                                acc_dkv,
                                transpose_A=True,
                                init=False,
                            )
                            T.gemm_v0(
                                p_lo_l1,
                                do_all[cslot * N_SEG + s, :, :],
                                acc_dkv,
                                transpose_A=True,
                                init=False,
                            )
                            T.copy(
                                acc_dkv,
                                workspace_dkv[cid, cslot, ci, 0:BN, s * DS : (s + 1) * DS],
                            )
                        # ---- [C] G6 dkv tail (dP only: dO has no tail) ----
                        T.gemm_v0(dp_hi_l1, qt_l1[cslot, :, :], acc_dkvt, transpose_A=True, init=True)
                        T.gemm_v0(dp_lo_l1, qt_l1[cslot, :, :], acc_dkvt, transpose_A=True, init=False)
                        T.copy(acc_dkvt, workspace_dkv[cid, cslot, ci, 0:BN, dim:DDT])
                        # ---- [V] CV-pairing dummy writes (ws_kv scratch slot NI,
                        # never read: balances C's consume-side kv reads; the
                        # flags fire after the dummies, which follow ALL of
                        # V's real stash writes in program order — the stashed
                        # data is guaranteed visible, rev2 iter-0 mechanism) ----
                        T.copy(kv_ub[0:DS], workspace_kv[cid, cslot, NI, 0, 0:DS])
                        T.copy(kv_ub[dim:DDT], workspace_kv[cid, cslot, NI, 0, dim:DDT])
                        # ---- [V] explicit-delta source (amend-1 §11.1: GM
                        # verbatim, single read per consume task) ----
                        if not delta_internal:  # noqa: SIM102 — trace-time guard; inner is a TIR runtime guard
                            if ci == 0:
                                T.copy(
                                    Delta[
                                        s_row_c,
                                        H0c + vid * v_block : H0c + (vid + 1) * v_block,
                                    ],
                                    delta_cons,
                                )
                        # ---- [V] raw index re-read: feeds BOTH the P-recompute
                        # mask chain and the scatter's per-row position test
                        # (rev2 iter-1 payload slimming) ----
                        T.copy(Indices[s_row_c, g_c, ci * BN : (ci + 1) * BN], idx_ub)
                        # ---- [V] P recompute (rev3: the rev2 pst GM stash is
                        # replaced by a re-read of the flagged ws_s/ws_dp
                        # slots + the same vector chain — bit-identical P
                        # (same ws_s bits, same idx row, same lse row, same op
                        # sequence) without any un-flagged same-core transit) ----
                        T.copy(idx_ub, idx_f)  # int32 -> fp32 auto-cast
                        T.tile.mul(m1_ub, idx_f, T.float32(-1.0))
                        T.tile.add(m1_ub, m1_ub, T.float32(s_i_c + 1))
                        T.tile.max(m1_ub, m1_ub, T.float32(0.0))
                        T.tile.min(m1_ub, m1_ub, T.float32(1.0))
                        T.tile.add(t_ub, idx_f, T.float32(1.0))
                        T.tile.max(m2_ub, t_ub, T.float32(0.0))
                        T.tile.min(m2_ub, t_ub, T.float32(1.0))
                        T.tile.mul(mask_ub, m1_ub, m2_ub)
                        T.tile.mul(m1_ub, mask_ub, T.float32(-1.0))
                        T.tile.add(m1_ub, m1_ub, T.float32(1.0))
                        T.tile.mul(bias_ub, m1_ub, T.float32(BIGC))
                        if ci == 0:
                            T.copy(
                                Lse[
                                    s_row_c,
                                    H0c + vid * v_block : H0c + (vid + 1) * v_block,
                                ],
                                lse_ub2,
                            )
                        T.copy(
                            workspace_s[cid, cslot, ci, vid * v_block : (vid + 1) * v_block, 0:BN],
                            s_ub,
                        )
                        T.copy(
                            workspace_dp[cid, cslot, ci, vid * v_block : (vid + 1) * v_block, 0:BN],
                            dp_ub,
                        )
                        T.tile.mul(s_ub, s_ub, sm_scale)
                        T.tile.broadcast(bcast_h, mask_ub, axis=0)
                        for g in range(NGRP):
                            T.tile.mul(
                                s_ub[g * EG : (g + 1) * EG, :],
                                s_ub[g * EG : (g + 1) * EG, :],
                                bcast_h,
                            )
                        T.tile.broadcast(bcast_h, bias_ub, axis=0)  # overwrite
                        for g in range(NGRP):
                            T.tile.sub(
                                s_ub[g * EG : (g + 1) * EG, :],
                                s_ub[g * EG : (g + 1) * EG, :],
                                bcast_h,
                            )
                        T.tile.broadcast(lse_2d, lse_ub2, axis=1)
                        T.tile.sub(s_ub, s_ub, lse_2d)
                        T.tile.exp(s_ub, s_ub)  # P (fp32, masked -> 0; bit-identical to produce)
                        # ---- [V] dP chain: dP = P * (dP_raw - Delta) * sm_scale
                        # (P=0 masks rows); the centralization runs in fp32 UB
                        # (§6.4 path I) ----
                        T.tile.broadcast(lse_2d, delta_cons, axis=1)
                        T.tile.sub(dp_ub, dp_ub, lse_2d)
                        T.tile.mul(dp_ub, s_ub, dp_ub)
                        T.tile.mul(dp_ub, dp_ub, sm_scale)
                        # ---- [V] dP -> hi+lo bf16 channels (hi via dp_hi_ub,
                        # lo via cast_bf16 — same form as the produce split) ----
                        T.copy(dp_ub, dp_hi_ub)  # dP_hi (bf16 cast)
                        T.copy(
                            dp_hi_ub,
                            workspace_dpb[cid, cslot, ci, 0, vid * v_block : (vid + 1) * v_block, 0:BN],
                        )
                        T.copy(dp_hi_ub, lse_2d)  # hi -> fp32 (exact)
                        T.tile.sub(lse_2d, dp_ub, lse_2d)  # residual -> lse_2d (dp_ub read-only)
                        T.copy(lse_2d, cast_bf16)  # dP_lo (bf16 cast)
                        T.copy(
                            cast_bf16,
                            workspace_dpb[cid, cslot, ci, 1, vid * v_block : (vid + 1) * v_block, 0:BN],
                        )
                        # ---- [V] dq accumulate (UB fp32, BN-wide chunks; s_ub
                        # dead after the dP chain -> chunk read slot reuse) ----
                        for s in range(N_SEG):
                            for ss in range(DS // BN):
                                T.copy(
                                    workspace_dq[
                                        cid,
                                        cslot,
                                        ci,
                                        vid * v_block : (vid + 1) * v_block,
                                        s * DS + ss * BN : s * DS + (ss + 1) * BN,
                                    ],
                                    s_ub,
                                )
                                T.tile.add(
                                    acc_dq_all[s * (DS // BN) + ss, :, :],
                                    acc_dq_all[s * (DS // BN) + ss, :, :],
                                    s_ub,
                                )
                        T.copy(
                            workspace_dq[cid, cslot, ci, vid * v_block : (vid + 1) * v_block, dim:DDT],
                            tail_slot,
                        )
                        T.tile.add(acc_dqt_ub, acc_dqt_ub, tail_slot)
                        # ---- [V] P2 drain probe (before the scatter's per-row
                        # ifs clear the access history): cast_bf16 -> lse_2d
                        # forces the MTE3->V hard event (cast_bf16 is MTE3-read
                        # by the ws_dpb lo write above), draining the MTE3 pipe
                        # before the token boundary's and next iteration's V
                        # writes to cast_bf16 / lse_2d / s_ub / dp_ub ----
                        T.copy(cast_bf16, lse_2d)
                        # ---- [V] dKV scatter (conditional per-row; pos from
                        # idx_ub, raw index — masked rows skipped entirely:
                        # no OOB write, no wasted atomics, §7.3) ----
                        for j in range(HALF):
                            jj = vid * HALF + j
                            pos = idx_ub[jj]
                            if (pos > -1) & (pos <= s_i_c):
                                if dkv_slot_bf16:
                                    # bf16 contribution slot: read row, cast
                                    # fp32, single atomic (src/dst dtype match)
                                    T.copy(workspace_dkv[cid, cslot, ci, jj, 0:dim], row_bf16[0:dim])
                                    T.copy(workspace_dkv[cid, cslot, ci, jj, dim:DDT], row_bf16[dim:DDT])
                                    T.copy(row_bf16[0:dim], row_ub[0:dim])
                                    T.copy(row_bf16[dim:DDT], row_ub[dim:DDT])
                                else:
                                    # fp32 slot (rev2 bit-exact fallback)
                                    T.copy(workspace_dkv[cid, cslot, ci, jj, 0:dim], row_ub[0:dim])
                                    T.copy(workspace_dkv[cid, cslot, ci, jj, dim:DDT], row_ub[dim:DDT])
                                T.tile.atomic_add(dKV[(bos_c + pos) * kv_group + g_c, 0], row_ub)
                    # ================= token boundary (V tail) =================
                    # Fires at the last block of every produce task. Order:
                    # (1) dq writeback of the consume task whose LAST block
                    #     was just consumed (kb-NI points at its block NI-1);
                    # (2) freeze the produce task's Delta prefix into the
                    #     consume slot (its first dP chain lands at kb+1; the
                    #     previous frozen value's last read was in THIS
                    #     iteration's consume segment — program order safe).
                    # The dq-accumulator CLEAR moved to the TOP of the loop
                    # body (UB-planner liveness fix — see the handoff block);
                    # the next task's clear fires at iteration kb+1, after
                    # this writeback.
                    if kb % NI == NI - 1:
                        if kb >= 2 * NI - 1:
                            wpid = ((kb - NI) // NI) * grid_size + cid
                            ws_row = wpid % seq_len
                            wg = wpid // seq_len
                            wH0 = wg * M
                            for d in range(DCH):
                                T.copy(acc_dq_all[d, :, :], s_ub)  # 3D -> 2D fp32
                                T.copy(s_ub, cast_bf16)  # fp32 -> bf16
                                T.copy(
                                    cast_bf16,
                                    dQ[
                                        ws_row,
                                        wH0 + vid * v_block : wH0 + (vid + 1) * v_block,
                                        d * BN : (d + 1) * BN,
                                    ],
                                )
                            T.copy(acc_dqt_ub, tail_slot)
                            T.copy(tail_slot, dqt_bf16)
                            T.copy(
                                dqt_bf16,
                                dQ[
                                    ws_row,
                                    wH0 + vid * v_block : wH0 + (vid + 1) * v_block,
                                    dim:DDT,
                                ],
                            )
                        if delta_internal:  # noqa: SIM102 — trace-time guard; inner is a TIR runtime guard
                            if kb < NB:
                                T.copy(delta_acc, delta_cons)
                        # ---- [V] P3 drain probe: cast_bf16 -> lse_2d drains
                        # the boundary's dQ MTE3 reads (cast_bf16 / dqt_bf16)
                        # before the next iteration's produce-segment V writes
                        # (the lo-channel cast into cast_bf16) ----
                        T.copy(cast_bf16, lse_2d)

        return main

    return _sparse_mla_bwd_builder


def tl_sparse_mla_bwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    block_N=128,
    grid_size=20,
    target="ascendc",
    delta_internal=False,
    dkv_slot_bf16=False,
):
    """Build the bwd core kernel for the requested backend.

        Returns: func(Q, KV, dO, Indices, Lse, Delta, Offsets, TokenIndices,
        dQ, dKV, workspace_kv, workspace_s, workspace_dp, workspace_p,
        workspace_dpb, workspace_dq, workspace_dkv) — all buffers explicit
        (host torch.zeros / torch.empty). Transit slots are [grid, PARITY=2,
        NI(+1), ...] token-parity double-buffered; workspace_kv, workspace_s
        and workspace_dp carry NI+1 block slots (slot NI = CV-pairing
        scratch); workspace_dkv is the bf16 (default) / fp32
        (dkv_slot_bf16=False) contribution-slot family; dKV is the flat fp32
        [S*kv_group, 576] atomic target.
    delta_internal=True (amend-1 §1.3): Delta is reduced in-kernel (produce
    phase prefix + boundary freeze) and the Delta tensor argument is never
    read — the host passes any same-shaped fp32 placeholder (e.g. lse).
    """
    return _make_bwd_builder(target)(
        heads=heads,
        dim=dim,
        tail_dim=tail_dim,
        topk=topk,
        kv_group=kv_group,
        sm_scale=sm_scale,
        block_N=block_N,
        grid_size=grid_size,
        delta_internal=delta_internal,
        dkv_slot_bf16=dkv_slot_bf16,
    )


# ============================================================================
# Host side: data preparation + orchestration (amend-1 §13.1 two-launch flow:
# contiguity asserts -> token_indices -> bwd (k2, with in-kernel Delta when
# delta is None) -> postprocess; k1 preprocess retained for the explicit /
# diagnostic path only).
# ============================================================================


def _tensor_cache(fn):
    """Single-entry cache keyed on input tensor identity (ref_gpu/index.py
    form): keeps prepare_token_indices' CPU build + H2D out of steady-state
    benchmark loops."""

    last_args = None
    last_result = None

    def wrapper(*args):
        nonlocal last_args, last_result
        if last_args is not None and len(args) == len(last_args) and all(a is b for a, b in zip(args, last_args)):
            return last_result
        result = fn(*args)
        last_args, last_result = args, result
        return result

    return wrapper


def get_cube_core_num(default=20):
    """Query device cube core count (grid_size compile constant; 910B3 = 20)."""
    try:
        import torch

        if torch.npu.is_available():
            n = torch.npu.get_device_properties(0).cube_core_num
            if isinstance(n, int) and n > 0:
                return n
    except Exception:  # pragma: no cover - probe failure falls back to default
        pass
    return default


@_tensor_cache
def prepare_token_indices(offsets):
    """Build TokenIndices [Sigma S, 2] int32 ((batch_id, in-segment position)).

    CPU construction -> single H2D (ref_gpu/index.py prepare_token_indices).
    """
    import torch

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


_BUILD_CACHE: dict = {}

_DELTA_PAD = 64  # fp32 rows padded onto Delta (tail-branch whole-vector writes)

_PARITY = 2  # token-parity double buffering (matches the kernel layout)


def _run_preprocess(o, do, target):
    """k1 runner: o/do [S, H, D] bf16 -> Delta [S, H] fp32 (padded buffer)."""
    import torch

    S, H, D = o.shape
    assert o.is_contiguous() and do.is_contiguous()
    assert o.dtype == do.dtype == torch.bfloat16 and D == 512
    dev = o.device
    grid = get_cube_core_num()
    key = ("preprocess", D, grid, target)
    func = _BUILD_CACHE.get(key)
    if func is None:
        func = tl_sparse_mla_bwd_preprocess(dim=D, grid_size=grid, target=target)
        _BUILD_CACHE[key] = func
    delta_padded = torch.zeros(S * H + _DELTA_PAD, dtype=torch.float32, device=dev)
    func(o.reshape(S * H, D), do.reshape(S * H, D), delta_padded)
    return delta_padded[: S * H].view(S, H)


def _run_bwd(
    q,
    kv,
    do,
    indices,
    lse,
    delta,
    offsets,
    token_indices,
    heads,
    dim,
    tail_dim,
    topk,
    kv_group,
    sm_scale,
    target,
):
    """k2 runner: returns (dQ bf16 [S, H, 576], dKV fp32 flat [S*kv_group, 576]).

    delta: [S, H] fp32 tensor (explicit path), or None for the amend-1 §1.3
    in-kernel Delta identity (produce-phase prefix + boundary freeze; the GM
    Delta input is never read and lse is passed as the placeholder)."""
    import torch

    S = q.shape[0]
    DDT = dim + tail_dim
    head_kv = heads // kv_group
    dev = q.device
    grid = get_cube_core_num()
    BN = 128
    NI = topk // BN
    delta_internal = delta is None
    key = (
        "bwd",
        heads,
        dim,
        tail_dim,
        topk,
        kv_group,
        sm_scale,
        BN,
        grid,
        target,
        delta_internal,
        _DKV_SLOT_BF16,
    )
    func = _BUILD_CACHE.get(key)
    if func is None:
        func = tl_sparse_mla_bwd(
            heads=heads,
            dim=dim,
            tail_dim=tail_dim,
            topk=topk,
            kv_group=kv_group,
            sm_scale=sm_scale,
            block_N=BN,
            grid_size=grid,
            target=target,
            delta_internal=delta_internal,
            dkv_slot_bf16=_DKV_SLOT_BF16,
        )
        _BUILD_CACHE[key] = func
    dQ = torch.zeros((S, heads, DDT), dtype=q.dtype, device=dev)
    # flat fp32 atomic target, zero-initialised (atomic accumulate from 0)
    dKV = torch.zeros((S * kv_group, DDT), dtype=torch.float32, device=dev)
    # transit workspaces: fully overwritten before every read (per-core slots)
    workspace_kv = torch.empty(
        (grid, _PARITY, NI + 1, BN, DDT), dtype=torch.bfloat16, device=dev
    )
    workspace_s = torch.empty(
        (grid, _PARITY, NI + 1, head_kv, BN), dtype=torch.float32, device=dev
    )
    workspace_dp = torch.empty(
        (grid, _PARITY, NI + 1, head_kv, BN), dtype=torch.float32, device=dev
    )
    workspace_p = torch.empty(
        (grid, _PARITY, NI, 2, head_kv, BN), dtype=torch.bfloat16, device=dev
    )
    workspace_dpb = torch.empty(
        (grid, _PARITY, NI, 2, head_kv, BN), dtype=torch.bfloat16, device=dev
    )
    workspace_dq = torch.empty(
        (grid, _PARITY, NI, head_kv, DDT), dtype=torch.float32, device=dev
    )
    workspace_dkv = torch.empty(
        (grid, _PARITY, NI, BN, DDT),
        dtype=torch.bfloat16 if _DKV_SLOT_BF16 else torch.float32,
        device=dev,
    )
    func(
        q,
        kv,
        do,
        indices,
        lse,
        lse if delta_internal else delta,  # Delta arg: placeholder (never read)
        offsets,
        token_indices,
        dQ,
        dKV,
        workspace_kv,
        workspace_s,
        workspace_dp,
        workspace_p,
        workspace_dpb,
        workspace_dq,
        workspace_dkv,
    )
    return dQ, dKV


def _run_postprocess(dkv_flat, kv_group, ddt, out_shape, dtype, device, target):
    """k3 runner: dKV fp32 flat -> dkv bf16 [S, kv_group, ddt]."""
    import torch

    rows = dkv_flat.shape[0]
    grid = get_cube_core_num()
    key = ("postprocess", ddt, grid, target)
    func = _BUILD_CACHE.get(key)
    if func is None:
        func = tl_sparse_mla_bwd_postprocess(ddt=ddt, grid_size=grid, target=target)
        _BUILD_CACHE[key] = func
    dkv = torch.empty(out_shape, dtype=dtype, device=device)
    func(dkv_flat, dkv.reshape(rows, ddt))
    return dkv


def sparse_mla_bwd(
    q,
    kv,
    o,
    do,
    indices,
    lse,
    offsets,
    sm_scale=None,
    is_casual=True,
    return_kernel=False,
    delta=None,
    target="ascendc",
):
    """Host entry (main-repo signature verbatim; ``target`` appended for the
    dual-backend requirement, ascendc default / "pto" explicit).

    Args:
        q: [S, H, 576] bfloat16 (D=512 host-hardcoded + D_tail=64).
        kv: [S, kv_group, 576] bfloat16 (first 512 dims double as v).
        o: [S, H, 512] bfloat16 forward output (not consumed when delta is
            None — amend-1 §1.3 in-kernel Delta identity; §14 R12).
        do: [S, H, 512] bfloat16 upstream gradient.
        indices: [S, kv_group, topk] int32; values may include -1 and
            out-of-range fills (clamped for addressing, masked to 0).
        lse: [S, H] float32 forward logsumexp.
        offsets: [batch+1] int32 varlen segment boundaries.
        sm_scale: defaults to (D + D_tail) ** -0.5 = 576 ** -0.5.
        is_casual: must be True (main-repo param name, causal only).
        return_kernel: accepted, unused (main-repo contract).
        delta: [S, H] float32 precomputed Delta; None reduces Delta inside
            k2 via the amend-1 §1.3 identity (k1 not launched, §13.1).
        target: "ascendc" (default) or "pto".

    Returns:
        (dq [S, H, 576] bfloat16, dkv [S, kv_group, 576] bfloat16)
    """
    import torch

    assert q.is_contiguous()
    assert kv.is_contiguous()
    assert indices.is_contiguous()
    assert lse.is_contiguous()
    assert is_casual == True, "non-casual is not supported now"  # noqa: E712
    S, H, dim_plus_tail_dim = q.shape
    S_kv, kv_group, _ = kv.shape
    assert kv.shape[-1] == dim_plus_tail_dim
    assert S == S_kv  # noqa: SIM300  (main-repo verbatim)
    # dim should be assigned
    D = 512
    D_tail = dim_plus_tail_dim - D
    topk = indices.shape[-1]
    assert indices.shape == (S, kv_group, topk)
    assert lse.shape == (S, H)
    # dtype contract (fixed combination, DESIGN §1.4)
    assert q.dtype == kv.dtype == o.dtype == do.dtype == torch.bfloat16
    assert indices.dtype == offsets.dtype == torch.int32
    assert lse.dtype == torch.float32
    if delta is None:
        assert o.is_contiguous() and o.shape == (S, H, D)
    else:
        assert delta.is_contiguous() and delta.shape == (S, H)
        assert delta.dtype == torch.float32
    # support domain (host mirror of builder asserts)
    assert H % kv_group == 0, f"heads={H} must be divisible by kv_group={kv_group}"
    head_kv = H // kv_group
    assert head_kv % 16 == 0 and 16 <= head_kv <= 64, f"head_kv={head_kv} unsupported: need % 16 == 0 and 16 <= head_kv <= 64"
    assert topk % 128 == 0, f"topk={topk} must be divisible by 128 (block_N)"
    if sm_scale is None:
        sm_scale = (D + D_tail) ** (-0.5)

    token_indices = prepare_token_indices(offsets)

    # amend-1 (§1.3 / §13.1): delta=None never launches k1 — Delta is reduced
    # inside k2 (produce-phase prefix, frozen at the token boundary) from the
    # fp32 P/dP_raw intermediates; the host emits k2 -> k3 (two launches).
    dQ, dKV = _run_bwd(
        q,
        kv,
        do,
        indices,
        lse,
        delta,
        offsets,
        token_indices,
        H,
        D,
        D_tail,
        topk,
        kv_group,
        sm_scale,
        target,
    )
    dkv = _run_postprocess(
        dKV,
        kv_group,
        D + D_tail,
        (S, kv_group, D + D_tail),
        torch.bfloat16,
        q.device,
        target,
    )
    return dQ, dkv
