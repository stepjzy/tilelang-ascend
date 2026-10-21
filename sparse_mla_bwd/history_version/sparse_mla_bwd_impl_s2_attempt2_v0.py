"""sparse_mla_bwd: DSA sparse MLA backward (varlen packed, causal) for Ascend 910B3.

Developer mode (4-True AUTO pass_configs), dual backend (target="ascendc" |
"pto"), three-stage host orchestration replicating the main repo
(ref_gpu/sparse_mla_bwd.py, semantics authoritative):

  k1 preprocess   Delta[t, h] = sum_d O[t, h, d] * dO[t, h, d]        (fp32)
  k2 bwd          per (token, kv_group) task, per topk index block:
                    P  = exp(score * sm_scale - lse)   (masked -> 0)
                    dP = P * (dO.v - Delta) * sm_scale (masked -> 0)
                    dq += dP @ kv_rows                 (UB fp32, cross-block)
                    dKV[idx] += dP^T @ q + P^T @ dO    (fp32 GM atomic scatter)
  k3 postprocess  dKV fp32 -> bf16 single rounding

Semantics (migrated from ref_gpu/sparse_mla_bwd.py):
  * repeated indices are NOT deduplicated (each slot contributes; with
    uniform multiplicity the result equals the set-based golden exactly);
  * causal test uses in-segment position s_i (TokenIndices[s_row, 1]);
  * OOB indices (>= segment end / -1 / any negative): gather rows clamped
    for addressing only, contributions zeroed by the arithmetic mask chain
    (mask = 1{0 <= idx <= s_i}, fwd-sibling form); scatter rows skipped
    (conditional per-row atomic) - no OOB writes;
  * dq accumulates in fp32 (UB) across index blocks, one bf16 rounding at
    the end; dKV accumulates in fp32 GM via T.tile.atomic_add, one bf16
    rounding in postprocess. P/dP ride the dkv GEMMs as hi+lo bf16 channel
    pairs (V-side split + double-GEMM recombination, ~16 mantissa bits):
    the main repo's plain-bf16 cast fails the DESIGN §12.2 mixed-tolerance
    gate vs the fp32-autograd golden (dkv ratio 0.96, replica-attributed
    in debug_log_sparse_mla_bwd.md); fp16 operands are uncastable on the
    GM->L1 path and mix dtypes across shared q/do operands (probe-
    rejected). The dq path keeps the plain-bf16 dP hi channel (gate
    margin 0.998+).

Implementation layout (sparse_mla_fwd F6b sibling form; every mechanism
probe- or sibling-validated, see debug_log_sparse_mla_bwd.md):
  * persistent T.Kernel(grid=20, is_npu=True) as (cid, vid) -> MIX_AIC_1_2;
    task = (s_row, g) striped; AIV-vid owns head rows [vid*v_block, ...).
  * gemm_v0 L1 operands are 3D leading-dim slices only (offset != 0
    inner-dim regions broken on both backends, fwd R1 evidence); D=512 is
    split into N_SEG=4 x DS=128 column segments + 64-wide tail, DS == BN.
  * AUTO_CV_SYNC statement-count matching per workspace buffer (in order):
    workspace_kv 2:2, workspace_s 1:1, workspace_dp 1:1, workspace_p 2:2,
    workspace_dpb 2:2, workspace_dq 5:5, workspace_dkv 5:5.
  * L0C fragments (acc_sd 32K + acc_dqt 16K + acc_dkv 64K + acc_dkvt 32K =
    144KB sum > 128KB) rely on TL_ASCEND_MEMORY_PLANNING liveness merging
    (peak live 64KB) - probe-validated on both backends.
  * dKV atomic target is a flat 2D [S*kv_group, 576] fp32 buffer (per-row
    T.tile.atomic_add with UB src, indexer_bwd layout=1 form).

Support domain (host asserts): D=512 hardcoded (D_tail = dim-512);
head_kv = H // kv_group in {16, 32, 48, 64} (% 16 == 0, <= 64 - no head
padding path); topk % 128 == 0; dtypes q/kv/o/do bf16, indices/offsets
int32, lse/delta fp32; is_casual must be True (main-repo param name).
"""

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
# ============================================================================


def _make_postprocess_builder(target):
    @tilelang.jit(pass_configs=_pass_configs, target=target)
    def _postprocess_builder(ddt=576, grid_size=20, rows_per_aiv=16):
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
# k2 bwd: persistent CV-fused core kernel
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
    ):
        """Build the bwd core kernel (Developer mode, persistent MIX_AIC_1_2).

        Args:
            heads: total head count H (head_kv = heads // kv_group, no pad).
            dim: QK main / V dimension D (host hardcodes 512).
            tail_dim: QK tail dimension (576 - dim).
            topk: selected positions per row (topk % block_N == 0).
            kv_group: KV group count G.
            sm_scale: defaults to (dim + tail_dim) ** -0.5.
            block_N: topk-direction tile BN (DS = BN, N_SEG = dim // BN).
            grid_size: persistent grid (910B3 = 20).
        """
        assert dim == 512 and tail_dim == 64, f"(dim={dim}, tail_dim={tail_dim}) unsupported: fixed (512, 64) layout"
        head_kv, v_block = _derived_bwd_layout(heads, kv_group, topk, block_N)
        if sm_scale is None:
            sm_scale = (dim + tail_dim) ** (-0.5)

        M = head_kv  # GEMM M dimension (padded_H == head_kv, no pad path)
        BN = block_N
        DS = BN  # D-segment width == BN (s_ub slot reuse, fwd DS == BI note)
        N_SEG = dim // DS
        NI = topk // BN
        HALF = BN // 2  # gather/scatter rows per AIV
        DDT = dim + tail_dim
        assert dim % DS == 0 and DS % 16 == 0 and tail_dim % 16 == 0

        seq_len = T.symbolic("seq_len")
        seq_len_kv = T.symbolic("seq_len_kv")
        batch_plus_one = T.symbolic("batch_plus_one")

        dtype = "bfloat16"
        indices_dtype = "int32"
        accum_dtype = "float32"

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
            workspace_kv: T.Tensor([grid_size, BN, DDT], dtype),  # type: ignore
            workspace_s: T.Tensor([grid_size, M, BN], accum_dtype),  # type: ignore
            workspace_dp: T.Tensor([grid_size, M, BN], accum_dtype),  # type: ignore
            workspace_p: T.Tensor([grid_size, 2, M, BN], dtype),  # type: ignore  # P hi/lo
            workspace_dpb: T.Tensor([grid_size, 2, M, BN], dtype),  # type: ignore  # dP hi/lo
            workspace_dq: T.Tensor([grid_size, M, DDT], accum_dtype),  # type: ignore
            workspace_dkv: T.Tensor([grid_size, BN, DDT], accum_dtype),  # type: ignore
        ):
            with T.Kernel(grid_size, is_npu=True) as (cid, vid):
                # ============ Cube side: L1 GEMM operands ============
                # P/dP ride the dkv GEMMs as hi+lo bf16 channel pairs
                # (V-side split, double-GEMM recombination): the plain bf16
                # cast reproduces the main-repo GPU kernel but fails the
                # DESIGN §12.2 mixed-tolerance gate vs the fp32-autograd
                # golden (dkv ratio 0.96, replica-attributed); hi+lo carries
                # ~16 mantissa bits (2^-17 grain) with plain-bf16 GEMMs
                # (fp16 operands are uncastable on the GM->L1 path and mix
                # dtypes across the shared q/do operands - probe-rejected).
                v_all = T.alloc_L1([N_SEG, BN, DS], dtype)
                tail_l1 = T.alloc_L1([BN, tail_dim], dtype)
                q_all = T.alloc_L1([N_SEG, M, DS], dtype)
                qt_l1 = T.alloc_L1([M, tail_dim], dtype)
                do_all = T.alloc_L1([N_SEG, M, DS], dtype)
                p_hi_l1 = T.alloc_L1([M, BN], dtype)
                p_lo_l1 = T.alloc_L1([M, BN], dtype)
                dp_hi_l1 = T.alloc_L1([M, BN], dtype)
                dp_lo_l1 = T.alloc_L1([M, BN], dtype)
                # ============ Cube side: L0C accumulators ============
                # sum 144KB > 128KB, peak live 64KB (liveness-planned;
                # probe-validated). acc_sd serves G1 -> G2 -> G3 (sequential).
                acc_sd = T.alloc_L0C([M, BN], accum_dtype)
                acc_dqt = T.alloc_L0C([M, tail_dim], accum_dtype)
                acc_dkv = T.alloc_L0C([BN, DS], accum_dtype)
                acc_dkvt = T.alloc_L0C([BN, tail_dim], accum_dtype)
                # ============ Vector side: UB ============
                acc_dq_all = T.alloc_ub([N_SEG, v_block, DS], accum_dtype)
                acc_dqt_ub = T.alloc_ub([v_block, tail_dim], accum_dtype)
                s_ub = T.alloc_ub([v_block, BN], accum_dtype)  # P chain / dq seg slot
                dp_ub = T.alloc_ub([v_block, BN], accum_dtype)
                p_hi = T.alloc_ub([v_block, BN], dtype)  # P hi channel
                p_lo = T.alloc_ub([v_block, BN], dtype)  # P lo channel
                dp_hi = T.alloc_ub([v_block, BN], dtype)  # dP hi channel
                dp_lo = T.alloc_ub([v_block, BN], dtype)  # dP lo channel
                cast_bf16 = T.alloc_ub([v_block, BN], dtype)  # dq writeback cast
                lse_2d = T.alloc_ub([v_block, BN], accum_dtype)  # lse/delta broadcast
                tail_slot = T.alloc_ub([v_block, tail_dim], accum_dtype)
                dqt_bf16 = T.alloc_ub([v_block, tail_dim], dtype)
                lse_ub = T.alloc_ub([v_block], accum_dtype)
                delta_ub = T.alloc_ub([v_block], accum_dtype)
                idx_ub = T.alloc_ub([BN], indices_dtype)
                idx_f = T.alloc_ub([BN], accum_dtype)
                m1_ub = T.alloc_ub([BN], accum_dtype)
                m2_ub = T.alloc_ub([BN], accum_dtype)
                t_ub = T.alloc_ub([BN], accum_dtype)
                mask_ub = T.alloc_ub([BN], accum_dtype)
                bias_ub = T.alloc_ub([BN], accum_dtype)
                rows_int = T.alloc_ub([BN], indices_dtype)
                kv_ub = T.alloc_ub([DDT], dtype)
                row_ub = T.alloc_ub([DDT], accum_dtype)

                total = seq_len * kv_group
                ntask = T.max(0, T.ceildiv(total - cid, grid_size))
                for k in T.serial(ntask):
                    pid = k * grid_size + cid
                    s_row = pid % seq_len  # row-major: same-g rows adjacent
                    g_i = pid // seq_len
                    b_i = TokenIndices[s_row, 0]
                    s_i = TokenIndices[s_row, 1]
                    bos = Offsets[b_i]
                    H0 = g_i * M  # Q head-segment base for this task

                    # ---- [C] Q / dO load, once per task (K/N seg columns) ----
                    for s in range(N_SEG):
                        T.copy(
                            Q[s_row, H0 : H0 + M, s * DS : (s + 1) * DS],
                            q_all[s, :, :],
                        )
                    T.copy(Q[s_row, H0 : H0 + M, dim:DDT], qt_l1)
                    for s in range(N_SEG):
                        T.copy(
                            dO[s_row, H0 : H0 + M, s * DS : (s + 1) * DS],
                            do_all[s, :, :],
                        )

                    # ---- [V] per-task head-half scalars + dq accumulators ----
                    T.copy(
                        Lse[s_row, H0 + vid * v_block : H0 + (vid + 1) * v_block],
                        lse_ub,
                    )
                    T.copy(
                        Delta[s_row, H0 + vid * v_block : H0 + (vid + 1) * v_block],
                        delta_ub,
                    )
                    for d in range(N_SEG):
                        T.tile.fill(acc_dq_all[d, :, :], 0.0)
                    T.tile.fill(acc_dqt_ub, 0.0)

                    for i in T.serial(NI):
                        # ================ [V] index block + mask chain ================
                        # mask = 1{0 <= idx <= s_i} (fwd-sibling arithmetic form;
                        # negative values incl. -1 are all masked); rows =
                        # clamp(bos + idx, 0, S_kv-1) absolute gather addresses.
                        T.copy(Indices[s_row, g_i, i * BN : (i + 1) * BN], idx_ub)
                        T.copy(idx_ub, idx_f)  # int32 -> fp32 auto-cast
                        # m1 = 1{idx <= s_i}: (s_i+1) - idx, clamp01
                        T.tile.mul(m1_ub, idx_f, T.float32(-1.0))
                        T.tile.add(m1_ub, m1_ub, T.float32(s_i + 1))
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
                        T.tile.add(idx_f, idx_f, T.float32(bos))
                        T.tile.max(idx_f, idx_f, T.float32(0.0))
                        T.tile.min(idx_f, idx_f, T.float32(seq_len_kv - 1))
                        T.copy(idx_f, rows_int)  # fp32 -> int32 (values exact)

                        # ================ [V] gather -> workspace_kv ================
                        # AIV-vid fetches its HALF rows; each row written as
                        # column segments + tail (2 stmts, C-side 2:2 match).
                        for j in range(HALF):
                            kv_row = rows_int[vid * HALF + j]
                            T.copy(KV[kv_row, g_i, :], kv_ub)
                            for s in range(N_SEG):
                                T.copy(
                                    kv_ub[s * DS : (s + 1) * DS],
                                    workspace_kv[
                                        cid,
                                        vid * HALF + j,
                                        s * DS : (s + 1) * DS,
                                    ],
                                )
                            T.copy(
                                kv_ub[dim:DDT],
                                workspace_kv[cid, vid * HALF + j, dim:DDT],
                            )

                        # ================ [V] P / dP chain (head half) ================
                        T.copy(
                            workspace_s[cid, vid * v_block : (vid + 1) * v_block, 0:BN],
                            s_ub,
                        )
                        T.copy(
                            workspace_dp[cid, vid * v_block : (vid + 1) * v_block, 0:BN],
                            dp_ub,
                        )
                        # P = exp((s * mask - bias) * sm_scale - lse)
                        T.tile.mul(s_ub, s_ub, sm_scale)
                        for h in range(v_block):
                            T.tile.mul(s_ub[h, :], s_ub[h, :], mask_ub)
                        for h in range(v_block):
                            T.tile.sub(s_ub[h, :], s_ub[h, :], bias_ub)
                        T.tile.broadcast(lse_2d, lse_ub, axis=1)
                        T.tile.sub(s_ub, s_ub, lse_2d)
                        T.tile.exp(s_ub, s_ub)  # P (masked -> exp underflow 0)
                        # dP = P * (dO.v - Delta) * sm_scale (P=0 masks rows)
                        T.tile.broadcast(lse_2d, delta_ub, axis=1)  # reuse slot
                        T.tile.sub(dp_ub, dp_ub, lse_2d)
                        T.tile.mul(dp_ub, s_ub, dp_ub)
                        T.tile.mul(dp_ub, dp_ub, sm_scale)
                        # P -> hi+lo bf16 channels (lse_2d dead -> hi scratch;
                        # separate hi/lo slots: an intra-block slot overwrite
                        # after the GM write races MTE3 vs the V pipe)
                        T.copy(s_ub, p_hi)  # P_hi
                        T.copy(
                            p_hi,
                            workspace_p[cid, 0, vid * v_block : (vid + 1) * v_block, 0:BN],
                        )
                        T.copy(p_hi, lse_2d)  # hi -> fp32 (exact)
                        T.tile.sub(s_ub, s_ub, lse_2d)  # residual
                        T.copy(s_ub, p_lo)  # P_lo
                        T.copy(
                            p_lo,
                            workspace_p[cid, 1, vid * v_block : (vid + 1) * v_block, 0:BN],
                        )
                        # dP -> hi+lo bf16 channels
                        T.copy(dp_ub, dp_hi)  # dP_hi
                        T.copy(
                            dp_hi,
                            workspace_dpb[cid, 0, vid * v_block : (vid + 1) * v_block, 0:BN],
                        )
                        T.copy(dp_hi, lse_2d)  # hi -> fp32 (exact)
                        T.tile.sub(dp_ub, dp_ub, lse_2d)  # residual
                        T.copy(dp_ub, dp_lo)  # dP_lo
                        T.copy(
                            dp_lo,
                            workspace_dpb[cid, 1, vid * v_block : (vid + 1) * v_block, 0:BN],
                        )

                        # ================ [V] dq accumulate (UB fp32) ================
                        # s_ub dead after the P/dP chain -> reused as the
                        # [v_block, DS] segment read slot (DS == BN).
                        for s in range(N_SEG):
                            T.copy(
                                workspace_dq[
                                    cid,
                                    vid * v_block : (vid + 1) * v_block,
                                    s * DS : (s + 1) * DS,
                                ],
                                s_ub,
                            )
                            T.tile.add(acc_dq_all[s, :, :], acc_dq_all[s, :, :], s_ub)
                        T.copy(
                            workspace_dq[
                                cid,
                                vid * v_block : (vid + 1) * v_block,
                                dim:DDT,
                            ],
                            tail_slot,
                        )
                        T.tile.add(acc_dqt_ub, acc_dqt_ub, tail_slot)

                        # ================ [V] dKV scatter (per-row cond) ================
                        # pos re-read from GM (indexer_bwd form); masked rows
                        # skipped entirely (no OOB write, no wasted atomics).
                        for j in range(HALF):
                            jj = vid * HALF + j
                            pos = Indices[s_row, g_i, i * BN + jj]
                            if (pos > -1) & (pos <= s_i):
                                for s in range(N_SEG):
                                    T.copy(
                                        workspace_dkv[cid, jj, s * DS : (s + 1) * DS],
                                        row_ub[s * DS : (s + 1) * DS],
                                    )
                                T.copy(
                                    workspace_dkv[cid, jj, dim:DDT],
                                    row_ub[dim:DDT],
                                )
                                T.tile.atomic_add(dKV[(bos + pos) * kv_group + g_i, 0], row_ub)

                        # ================ [C] workspace_kv -> L1 ================
                        for s in range(N_SEG):
                            T.copy(
                                workspace_kv[cid, 0:BN, s * DS : (s + 1) * DS],
                                v_all[s, :, :],
                            )
                        T.copy(workspace_kv[cid, 0:BN, dim:DDT], tail_l1)

                        # ================ [C] G1 score -> workspace_s ================
                        # K = 576 via 4 x DS segments + tail (init chain).
                        T.gemm_v0(
                            q_all[0, :, :],
                            v_all[0, :, :],
                            acc_sd,
                            transpose_B=True,
                            init=True,
                        )
                        for s in range(1, N_SEG):
                            T.gemm_v0(
                                q_all[s, :, :],
                                v_all[s, :, :],
                                acc_sd,
                                transpose_B=True,
                                init=False,
                            )
                        T.gemm_v0(qt_l1, tail_l1, acc_sd, transpose_B=True, init=False)
                        T.copy(acc_sd, workspace_s[cid, 0:M, 0:BN])

                        # ================ [C] G2 dO.v -> workspace_dp ================
                        T.gemm_v0(
                            do_all[0, :, :],
                            v_all[0, :, :],
                            acc_sd,
                            transpose_B=True,
                            init=True,
                        )
                        for s in range(1, N_SEG):
                            T.gemm_v0(
                                do_all[s, :, :],
                                v_all[s, :, :],
                                acc_sd,
                                transpose_B=True,
                                init=False,
                            )
                        T.copy(acc_sd, workspace_dp[cid, 0:M, 0:BN])

                        # ================ [C] consume P / dP (hi+lo) ================
                        T.copy(workspace_p[cid, 0, 0:M, 0:BN], p_hi_l1)
                        T.copy(workspace_p[cid, 1, 0:M, 0:BN], p_lo_l1)
                        T.copy(workspace_dpb[cid, 0, 0:M, 0:BN], dp_hi_l1)
                        T.copy(workspace_dpb[cid, 1, 0:M, 0:BN], dp_lo_l1)

                        # ================ [C] G3 dq main -> workspace_dq ================
                        # dq passed the bf16 gate with the plain-bf16 dP grain
                        # (hi channel); the lo channel rides the dkv path only.
                        for s in range(N_SEG):
                            T.gemm_v0(dp_hi_l1, v_all[s, :, :], acc_sd, init=True)
                            T.copy(
                                acc_sd,
                                workspace_dq[cid, 0:M, s * DS : (s + 1) * DS],
                            )

                        # ================ [C] G4 dq tail -> workspace_dq ================
                        T.gemm_v0(dp_hi_l1, tail_l1, acc_dqt, init=True)
                        T.copy(acc_dqt, workspace_dq[cid, 0:M, dim:DDT])

                        # ================ [C] G5 dkv pair -> workspace_dkv ================
                        # double-GEMM hi+lo recombination:
                        # dkv = (dP_hi + dP_lo)^T @ q + (P_hi + P_lo)^T @ dO
                        for s in range(N_SEG):
                            T.gemm_v0(
                                dp_hi_l1,
                                q_all[s, :, :],
                                acc_dkv,
                                transpose_A=True,
                                init=True,
                            )
                            T.gemm_v0(
                                dp_lo_l1,
                                q_all[s, :, :],
                                acc_dkv,
                                transpose_A=True,
                                init=False,
                            )
                            T.gemm_v0(
                                p_hi_l1,
                                do_all[s, :, :],
                                acc_dkv,
                                transpose_A=True,
                                init=False,
                            )
                            T.gemm_v0(
                                p_lo_l1,
                                do_all[s, :, :],
                                acc_dkv,
                                transpose_A=True,
                                init=False,
                            )
                            T.copy(
                                acc_dkv,
                                workspace_dkv[cid, 0:BN, s * DS : (s + 1) * DS],
                            )

                        # ================ [C] G6 dkv tail -> workspace_dkv ================
                        T.gemm_v0(dp_hi_l1, qt_l1, acc_dkvt, transpose_A=True, init=True)
                        T.gemm_v0(dp_lo_l1, qt_l1, acc_dkvt, transpose_A=True, init=False)
                        T.copy(acc_dkvt, workspace_dkv[cid, 0:BN, dim:DDT])

                    # ---- [V tail] dq writeback (single bf16 rounding) ----
                    for d in range(N_SEG):
                        T.copy(acc_dq_all[d, :, :], s_ub)  # 3D -> 2D fp32
                        T.copy(s_ub, cast_bf16)  # fp32 -> bf16
                        T.copy(
                            cast_bf16,
                            dQ[
                                s_row,
                                H0 + vid * v_block : H0 + (vid + 1) * v_block,
                                d * DS : (d + 1) * DS,
                            ],
                        )
                    T.copy(acc_dqt_ub, tail_slot)
                    T.copy(tail_slot, dqt_bf16)
                    T.copy(
                        dqt_bf16,
                        dQ[
                            s_row,
                            H0 + vid * v_block : H0 + (vid + 1) * v_block,
                            dim:DDT,
                        ],
                    )

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
):
    """Build the bwd core kernel for the requested backend.

    Returns: func(Q, KV, dO, Indices, Lse, Delta, Offsets, TokenIndices, dQ,
    dKV, workspace_kv, workspace_s, workspace_dp, workspace_p, workspace_dpb,
    workspace_dq, workspace_dkv) — all buffers explicit (host torch.zeros /
    torch.empty), dKV is the flat fp32 [S*kv_group, 576] atomic target.
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
    )


# ============================================================================
# Host side: data preparation + three-stage orchestration (main-repo order:
# contiguity asserts -> token_indices -> [delta is None] preprocess -> bwd ->
# postprocess; only torch.zeros/empty allocations and kernel calls).
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
    """k2 runner: returns (dQ bf16 [S, H, 576], dKV fp32 flat [S*kv_group, 576])."""
    import torch

    S = q.shape[0]
    DDT = dim + tail_dim
    head_kv = heads // kv_group
    dev = q.device
    grid = get_cube_core_num()
    BN = 128
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
        )
        _BUILD_CACHE[key] = func
    dQ = torch.zeros((S, heads, DDT), dtype=q.dtype, device=dev)
    # flat fp32 atomic target, zero-initialised (atomic accumulate from 0)
    dKV = torch.zeros((S * kv_group, DDT), dtype=torch.float32, device=dev)
    # transit workspaces: fully overwritten before every read (per-core slots)
    workspace_kv = torch.empty((grid, BN, DDT), dtype=torch.bfloat16, device=dev)
    workspace_s = torch.empty((grid, head_kv, BN), dtype=torch.float32, device=dev)
    workspace_dp = torch.empty((grid, head_kv, BN), dtype=torch.float32, device=dev)
    workspace_p = torch.empty((grid, 2, head_kv, BN), dtype=torch.bfloat16, device=dev)
    workspace_dpb = torch.empty((grid, 2, head_kv, BN), dtype=torch.bfloat16, device=dev)
    workspace_dq = torch.empty((grid, head_kv, DDT), dtype=torch.float32, device=dev)
    workspace_dkv = torch.empty((grid, BN, DDT), dtype=torch.float32, device=dev)
    func(
        q,
        kv,
        do,
        indices,
        lse,
        delta,
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
        o: [S, H, 512] bfloat16 forward output (unused when delta is given).
        do: [S, H, 512] bfloat16 upstream gradient.
        indices: [S, kv_group, topk] int32; values may include -1 and
            out-of-range fills (clamped for addressing, masked to 0).
        lse: [S, H] float32 forward logsumexp.
        offsets: [batch+1] int32 varlen segment boundaries.
        sm_scale: defaults to (D + D_tail) ** -0.5 = 576 ** -0.5.
        is_casual: must be True (main-repo param name, causal only).
        return_kernel: accepted, unused (main-repo contract).
        delta: [S, H] float32 precomputed Delta; None runs preprocess (k1).
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

    if delta is None:
        delta = _run_preprocess(o, do, target)
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
