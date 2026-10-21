"""sparse_mla_fwd: DSA sparse MLA forward (varlen packed layout) for Ascend 910B3.

Developer mode (4-True AUTO pass_configs), dual backend (target="ascendc" | "pto"),
persistent grid, task = (packed row s_row, kv_group g). Per task: gather topk KV
rows by Indices, causal+invalid arithmetic mask, QK with K split into dim/BI
segments + tail (F6b, see below), streaming online softmax (m_i / sumexp /
alpha), PV with N split into dim/BI segments, Output [S, H, 512] bf16 + Lse
[S, H] fp32.

Semantics (migrated from ref_gpu/sparse_mla_fwd.py, kernel semantics authoritative):
  * repeated indices are NOT deduplicated (repeat contribution kept);
  * causal test uses in-segment position s_i (TokenIndices[s_row, 1]);
  * out-of-range / -1 indices: rows clamped for addressing safety only, their
    contribution is zeroed by the mask chain (invalid -> score - 1e30);
  * first-row self-attention falls out of the mask math naturally;
  * no assumption on index distribution.

[F6b deviation from DESIGN §3.2/§3.3 — R1/R4 verification results 2026-10-14]
(1) T.gemm_v0 L1-operand BufferRegions with NON-ZERO offsets are broken on BOTH
backends (B column slice and A row slice at offset != 0 read garbage; R1/R1c
evidence in debug_log.md). (2) The AUTO_CV_SYNC pass requires EQUAL statement
counts per workspace buffer on the cube and vec sides, paired in order (R1
first-compile failure: "cube has 3, vec has 1"). (3) 3D L1/UB buffers with
leading-dim-slice regions DO work as T.copy dests, gemm_v0 operands and tile-op
targets (R4 probe, both backends). The implemented layout therefore:
  * single QK/PV granularity M = Hseg (all heads of the task in one gemm,
    gqa_pto statement granularity) -> every workspace is 1 C-statement vs
    1 V-statement (workspace_kv: segment loop + tail = 2:2);
  * workspace_kv stays a single [grid, BI, 576] buffer; the V side gathers its
    HALF rows into kv_stage [GKB, 576] UB (row-by-row GM reads with clamped
    scalar addressing) and writes them out as TWO whole-2D MTE3 bursts
    (batches of GKB rows; 2 V-statements matching the C-side s-loop + tail
    reads); the C side reads the segments into v_all [N_SEG, BI, BI] and
    tail_l1 (strided GM->L1, R4-validated 3D regions). Perf iter-2 note:
    2D column-slice T.copy SOURCES are broken on ascendc (probe t3i2), so the
    gather write uses whole-2D sources only; kv_stage relies on
    MEMORY_PLANNING liveness aliasing with the softmax-phase scratch
    (acc_s_ub / cast_bf16 are dead during the gather phase — barrier pass is
    address-based, runs after planning);
  * QK = 1 gemm per K segment (init on the first, accumulate on the rest) +
    tail gemm; PV = 1 gemm per N segment into acc_o_l0c [Hseg, BI];
  * V-side acc_o lives in acc_o_all [N_SEG, v_block, BI] 3D UB (R4-validated
     tile ops on [d, :, :] / [d, h, :] regions), chunks read through acc_s_ub;
  * [perf iter-3] the per-h softmax row-op loops are vectorized through a
     half-row broadcast scratch bcast_h [v_block/2, BI] (T.tile.broadcast
     axis=0 for the per-slot mask/bias, axis=1 with 1D-slice sources for the
     per-row m_i/alpha; 2D row-half regions on acc_s_ub and 3D middle-dim
     regions on acc_o_all as in-place tile-op dst/src — probe t3i3 validated
     all forms bit-exact on both backends; a full [v_block, BI] temp does NOT
     fit the 196KB UB next to acc_o_all + acc_s_ub, hence halves). The mask
     chain uses the clamp01(min(idx+1, s_i+1-idx)) form and the tail div loop
     stays per-h (per-task, 1/8 the frequency of the per-block loops);
  * zero offset-!=0 slicing on any L1/L0C gemm operand; zero Python-list
     indexing by TIR loop vars (all segment loops are TIR loops over 3D buffers).
L0C note: Hseg=128 x BI=128 lands on exactly 128 KB (acc_s 64KB + acc_o 64KB,
inclusive limit per indexer_bwd budget form); BI=64 sits at 64 KB.
"""

import tilelang
from tilelang import language as T

__all__ = [
    "tl_sparse_mla_fwd",
    "sparse_mla_fwd_interface",
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


def _derived_head_layout(heads, kv_group):
    """Head layout shared by builder and host (single source of truth).

    Domain (inherited from the CUDA original): head_kv = heads // kv_group with
    (16 <= head_kv <= 64 and power of 2) or head_kv % 64 == 0; padded_H !=
    head_kv only allowed for kv_group == 1.
    """
    head_kv = heads // kv_group
    assert kv_group >= 1 and heads % kv_group == 0, f"heads={heads} must be divisible by kv_group={kv_group}"
    assert (16 <= head_kv <= 64 and head_kv == tilelang.math.next_power_of_2(head_kv)) or head_kv % 64 == 0, (
        f"head_kv={head_kv} out of supported domain: need (16<=head_kv<=64 and power of 2) or head_kv % 64 == 0"
    )
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != head_kv:
        assert kv_group == 1, "H padding path only supports kv_group == 1 (inherited from the CUDA original)"
    if head_kv > 64:
        RH = head_kv // 64
        M_SLICE = 64
    else:
        RH = 1
        M_SLICE = padded_H
    Hseg = RH * M_SLICE
    v_block = Hseg // 2
    return head_kv, padded_H, RH, M_SLICE, Hseg, v_block


def _make_jit_builder(target):
    """Closure factory: one builder definition, one @tilelang.jit instance per
    target (the decorator fixes `target` at definition time — gqa_pto form)."""

    @tilelang.jit(pass_configs=_pass_configs, target=target)
    def _sparse_mla_fwd_builder(
        heads,
        dim,
        tail_dim,
        topk,
        kv_group=1,
        sm_scale=None,
        block_I=128,
        grid_size=20,
    ):
        # ---- compile-time asserts (support domain, DESIGN §3.5.1 / §11.4) ----
        assert dim == tilelang.math.next_power_of_2(dim), f"dim must be a power of 2, dim={dim}"
        assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"tail_dim must be a power of 2, tail_dim={tail_dim}"
        assert topk % block_I == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
        assert block_I % 2 == 0 and block_I >= 32, f"block_I must be even and >= 32 (AIV half-split), got {block_I}"
        # F6b: (dim, tail_dim) = (512, 64) is the only pair passing dim % BI == 0
        # (segment loop needs BI | dim), dim/tail powers of 2, DDT = 576.
        assert dim == 512 and tail_dim == 64, f"(dim={dim}, tail_dim={tail_dim}) unsupported: F6b layout requires (512, 64)"
        head_kv, padded_H, RH, M_SLICE, Hseg, v_block = _derived_head_layout(heads, kv_group)
        if sm_scale is None:
            sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

        BI = block_I
        NI = topk // BI
        HALF = BI // 2
        DDT = dim + tail_dim
        DS = BI  # KV/V segment width (QK K-seg == PV N-seg == V o-chunk)
        N_SEG = dim // DS  # 4 (BI=128) or 8 (BI=64)
        # [perf iter-2] gather staging rows per batch. HALF rows are gathered
        # in 2 explicit batches of GKB = HALF // 2; GKB is bounded by the UB
        # budget (kv_stage [GKB, DDT] bf16; relies on MEMORY_PLANNING to
        # alias it with the softmax-phase scratch, see module docstring).
        assert HALF % 2 == 0, f"block_I={block_I} must be a multiple of 4 (two-batch gather staging)"
        GKB = HALF // 2
        # [perf iter-3] broadcast group span: FIXED 8 rows (32-byte aligned
        # 1D slice sources m_i/m_prev[g*8:...] — a 4-row split crashed with a
        # VEC UB-alignment fault on kv_group=2/v_block=32, iso t3i3), variable
        # group count. UB budget: the reduce_max/reduce_sum workspace
        # (planner-allocated, live inside any buffer spanning the softmax
        # reduces) plus the broadcast scratch must stay clear of the
        # cast_bf16 window — an 8-row x BI fp32 scratch (4KB @ BI=128) fits
        # the tail past it; larger scratches did not (allocator overflow,
        # smoke t3i3). v_block is a power of two >= 8 (Hseg >= 16).
        assert v_block % 8 == 0, f"v_block={v_block} must be a multiple of 8"
        EG = 8
        NGRP = v_block // 8

        # ---- symbolic dynamic dims (one build covers any S / batch) ----
        seq_len = T.symbolic("seq_len")  # Sigma S
        seq_len_kv = T.symbolic("seq_len_kv")  # Sigma S_kv
        batch_plus_one = T.symbolic("batch_plus_one")

        dtype = "bfloat16"
        indices_dtype = "int32"
        accum_dtype = "float32"

        q_shape = [seq_len, heads, DDT]
        kv_shape = [seq_len_kv, kv_group, DDT]
        indices_shape = [seq_len, kv_group, topk]
        o_shape = [seq_len, heads, dim]
        lse_shape = [seq_len, heads]
        offsets_shape = [batch_plus_one]
        token_indices_shape = [seq_len, 2]

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            KV: T.Tensor(kv_shape, dtype),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            Offsets: T.Tensor(offsets_shape, indices_dtype),  # type: ignore
            TokenIndices: T.Tensor(token_indices_shape, indices_dtype),  # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
            Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
            workspace_kv: T.Tensor([grid_size, BI, DDT], dtype),  # type: ignore
            workspace_s: T.Tensor([grid_size, Hseg, BI], accum_dtype),  # type: ignore
            workspace_p: T.Tensor([grid_size, Hseg, BI], dtype),  # type: ignore
            workspace_o: T.Tensor([grid_size, Hseg, dim], accum_dtype),  # type: ignore
        ):
            with T.Kernel(grid_size, is_npu=True) as (cid, vid):
                # ================= Cube side: L1 / L0C =================
                # v_all[s] = ws_kv columns [s*DS, (s+1)*DS) as [BI, DS] (R4).
                # Serves QK (transpose_B, K=DS) AND PV (non-transposed, N=DS).
                v_all = T.alloc_L1([N_SEG, BI, DS], dtype)
                tail_l1 = T.alloc_L1([BI, tail_dim], dtype)
                l1_dummy = T.alloc_L1([16, 16], dtype)  # CV-pairing dummy read
                q_all = T.alloc_L1([N_SEG, Hseg, DS], dtype)
                qt_l1 = T.alloc_L1([Hseg, tail_dim], dtype)
                p_l1 = T.alloc_L1([Hseg, BI], dtype)
                acc_s_l0c = T.alloc_L0C([Hseg, BI], accum_dtype)
                acc_o_l0c = T.alloc_L0C([Hseg, DS], accum_dtype)

                # ================= Vector side: UB =================
                # AIV-vid owns head rows [vid*v_block, (vid+1)*v_block) of Hseg.
                # acc_o_all[d] = output chunk [v_block, DS]; softmax scratch in
                # acc_s_ub [v_block, BI] (also the ws_o chunk read slot, DS==BI).
                acc_o_all = T.alloc_ub([N_SEG, v_block, DS], accum_dtype)
                acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
                cast_bf16 = T.alloc_ub([v_block, BI], dtype)  # p out / o out cast
                m_i = T.alloc_ub([v_block], accum_dtype)
                m_prev = T.alloc_ub([v_block], accum_dtype)
                sumexp = T.alloc_ub([v_block], accum_dtype)
                sumexp_i = T.alloc_ub([v_block], accum_dtype)
                indices_ub = T.alloc_ub([BI], indices_dtype)
                rows_int = T.alloc_ub([BI], indices_dtype)
                idx_f = T.alloc_ub([BI], accum_dtype)
                t_ub = T.alloc_ub([BI], accum_dtype)
                m1_ub = T.alloc_ub([BI], accum_dtype)
                m2_ub = T.alloc_ub([BI], accum_dtype)
                mask_ub = T.alloc_ub([BI], accum_dtype)
                kv_stage = T.alloc_ub([GKB, DDT], dtype)  # gather staging (perf iter-2)
                # [perf iter-3] group-row broadcast scratch for the vectorized
                # softmax chain (sizing rationale at the EG constant above).
                bcast_h = T.alloc_ub([EG, BI], accum_dtype)

                total = seq_len * kv_group
                ntask = T.max(0, T.ceildiv(total - cid, grid_size))
                for k in T.serial(ntask):
                    pid = k * grid_size + cid
                    s_row = pid % seq_len  # row-major: same-g rows adjacent
                    g_i = pid // seq_len
                    b_i = TokenIndices[s_row, 0]
                    s_i = TokenIndices[s_row, 1]
                    bos = Offsets[b_i]
                    H0 = g_i * padded_H  # Q head-segment base for this task

                    # ---- [C] Q load, once per task (whole head rows, per K seg)
                    for s in range(N_SEG):
                        T.copy(
                            Q[s_row, H0 : H0 + Hseg, s * DS : (s + 1) * DS],
                            q_all[s, :, :],
                        )
                    T.copy(Q[s_row, H0 : H0 + Hseg, dim:DDT], qt_l1)

                    # ---- [V] softmax state init (per task) ----
                    for d in range(N_SEG):
                        T.tile.fill(acc_o_all[d, :, :], 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(m_i, -(2.0**30))  # avoid -inf - inf NaN

                    for i in T.serial(NI):
                        # ---- [V] indices block + mask arithmetic chain ----
                        # mask = 1{0 <= idx <= s_i}; bias = BIGC * (1 - mask)
                        # (masked score -> x*mask - bias). [perf iter-3] mask
                        # via the min form: clamp01(min(idx + 1, s_i + 1 - idx))
                        # — both window terms are >= 1 inside [0, s_i] and
                        # either is <= 0 outside, so min-then-clamp01 is exact
                        # (8 ops, was 11 in the m1*m2 product form). bias in 2
                        # ops: mask * (-BIGC) + BIGC == BIGC * (1 - mask)
                        # exactly for mask in {0, 1}. All-fp32 scalar chains
                        # (bf16 scalar arithmetic banned); sub with scalar LEFT
                        # operand replaced by mul(-1)+add.
                        T.copy(Indices[s_row, g_i, i * BI : (i + 1) * BI], indices_ub)
                        T.copy(indices_ub, idx_f)  # int32 -> fp32 auto-cast
                        # t_ub = idx + 1; m1_ub = (s_i + 1) - idx (mul+add form)
                        T.tile.add(t_ub, idx_f, T.float32(1.0))
                        T.tile.mul(m1_ub, idx_f, T.float32(-1.0))
                        T.tile.add(m1_ub, m1_ub, T.float32(s_i + 1))
                        # mask = clamp01(min(t_ub, m1_ub)); m2_ub holds the min
                        T.tile.min(m2_ub, t_ub, m1_ub)
                        T.tile.max(mask_ub, m2_ub, T.float32(0.0))
                        T.tile.min(mask_ub, mask_ub, T.float32(1.0))
                        # bias = mask * (-BIGC) + BIGC (m1_ub reused)
                        T.tile.mul(m1_ub, mask_ub, T.float32(-BIGC))
                        T.tile.add(m1_ub, m1_ub, T.float32(BIGC))
                        # rows = clamp(idx + bos, 0, seq_len_kv - 1): addressing
                        # guard only (contribution zeroed by mask). fp32 domain
                        # min/max with symbolic bound, cast back to int32.
                        T.tile.add(idx_f, idx_f, T.float32(bos))
                        T.tile.max(idx_f, idx_f, T.float32(0.0))
                        T.tile.min(idx_f, idx_f, T.float32(seq_len_kv - 1))
                        T.copy(idx_f, rows_int)  # fp32 -> int32 (values exact)

                        # ---- [V] gather: AIV-vid fetches its HALF rows into
                        #      kv_stage [GKB, DDT] and writes each batch out
                        #      as ONE whole-2D MTE3 burst (perf iter-2; was
                        #      per-row column-segment fragments, 320 micro
                        #      writes -> 2 large writes per AIV per block).
                        #      Whole-2D T.copy sources only: 2D column-slice
                        #      sources are broken on ascendc (probe t3i2
                        #      probe1b/1c). AUTO_CV_SYNC pairs statements IN
                        #      ORDER and each C statement may only depend on
                        #      its paired (or earlier) V statement: the C-side
                        #      s-loop/tail reads need ALL rows, so V emits
                        #      [batch0, batch1, mini] where the trailing
                        #      idempotent 1-row rewrite (kv_stage still holds
                        #      the last batch-1 row) pairs with the C-side
                        #      tail read — 3:3 statements. Semantic guards
                        #      unchanged: rows come from the clamped rows_int
                        #      (addressing safety), invalid slots masked to 0
                        #      by the mask chain.
                        for j in range(GKB):
                            kv_row = rows_int[vid * HALF + j]
                            T.copy(KV[kv_row, g_i, :], kv_stage[j, :])
                        T.copy(
                            kv_stage,
                            workspace_kv[cid, vid * HALF : vid * HALF + GKB, :],
                        )
                        for j in range(GKB):
                            kv_row = rows_int[vid * HALF + GKB + j]
                            T.copy(KV[kv_row, g_i, :], kv_stage[j, :])
                        T.copy(
                            kv_stage,
                            workspace_kv[cid, vid * HALF + GKB : (vid + 1) * HALF, :],
                        )
                        T.copy(
                            kv_stage[GKB - 1, :],
                            workspace_kv[cid, (vid + 1) * HALF - 1, :],
                        )

                        # ---- [C] ws_kv -> L1 segments + tail (R4 3D regions) ----
                        # stmt 1: tiny dummy read of rows [0:16] x cols [0:16]
                        # (produced by V stmt 1 / batch 0). AUTO_CV_SYNC pairs
                        # statements in order; without this dummy the s-loop
                        # read would pair with V stmt 1 (batch 0 only) and
                        # race batch 1's write (iter-2 smoke evidence:
                        # deterministic 97% corruption, both backends).
                        T.copy(workspace_kv[cid, 0:16, 0:16], l1_dummy)
                        for s in range(N_SEG):
                            T.copy(
                                workspace_kv[cid, 0:BI, s * DS : (s + 1) * DS],
                                v_all[s, :, :],
                            )
                        T.copy(workspace_kv[cid, 0:BI, dim:DDT], tail_l1)

                        # ---- [C] QK: K = DS * N_SEG + tail_dim, M = Hseg ----
                        T.gemm_v0(
                            q_all[0, :, :],
                            v_all[0, :, :],
                            acc_s_l0c,
                            transpose_B=True,
                            init=True,
                        )
                        for s in range(1, N_SEG):
                            T.gemm_v0(
                                q_all[s, :, :],
                                v_all[s, :, :],
                                acc_s_l0c,
                                transpose_B=True,
                                init=False,
                            )
                        T.gemm_v0(qt_l1, tail_l1, acc_s_l0c, transpose_B=True, init=False)
                        T.copy(acc_s_l0c, workspace_s[cid, 0:Hseg, :])

                        # ---- [C] PV: N split into N_SEG x DS, M = Hseg ----
                        T.copy(workspace_p[cid, 0:Hseg, :], p_l1)
                        for s in range(N_SEG):
                            T.gemm_v0(p_l1, v_all[s, :, :], acc_o_l0c, init=True)
                            T.copy(
                                acc_o_l0c,
                                workspace_o[cid, 0:Hseg, s * DS : (s + 1) * DS],
                            )

                        # ---- [V] online softmax on head rows [vid*v_block, ...) ----
                        T.copy(m_i, m_prev)
                        T.copy(
                            workspace_s[cid, vid * v_block : (vid + 1) * v_block, :],
                            acc_s_ub,
                        )
                        T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)  # scaled domain (note B)
                        # mask apply: x_p = x * mask - bias  (note A: -BIGC for
                        # invalid slots; exp underflows to 0, no -inf/NaN hazard;
                        # bias lives in m1_ub after the 1D chain above).
                        # [perf iter-3] vectorized: group broadcast ops
                        # replace the 2 x v_block per-row loops (was 128 row
                        # ops per block; probe t3i3 validated axis=0 broadcast
                        # + row-group region ops + in-place temp overwrite on
                        # both backends, bit-exact).
                        T.tile.broadcast(bcast_h, mask_ub, axis=0)  # [BI] -> [EG, BI]
                        for g in range(NGRP):
                            T.tile.mul(
                                acc_s_ub[g * EG : (g + 1) * EG, :],
                                acc_s_ub[g * EG : (g + 1) * EG, :],
                                bcast_h,
                            )
                        T.tile.broadcast(bcast_h, m1_ub, axis=0)  # bias (overwrite)
                        for g in range(NGRP):
                            T.tile.sub(
                                acc_s_ub[g * EG : (g + 1) * EG, :],
                                acc_s_ub[g * EG : (g + 1) * EG, :],
                                bcast_h,
                            )
                        # m_i update + alpha (gqa_pto form). [perf iter-3]
                        # reduce split into per-group calls: the planner-sized
                        # reduce workspace shrinks from 16KB ([v_block, BI]) to
                        # one group tile, keeping the softmax-phase UB footprint
                        # clear of the reduce/broadcast/cast window conflicts.
                        for g in range(NGRP):
                            T.reduce_max(
                                acc_s_ub[g * EG : (g + 1) * EG, :],
                                m_i[g * EG : (g + 1) * EG],
                                dim=-1,
                            )
                        T.tile.max(m_i, m_i, m_prev)
                        T.tile.sub(m_prev, m_prev, m_i)
                        T.tile.exp(m_prev, m_prev)  # alpha
                        # p = exp(x_p - m_i): per-row scalars broadcast along
                        # columns (axis=1, 1D slice sources), one group at a time
                        for g in range(NGRP):
                            T.tile.broadcast(bcast_h, m_i[g * EG : (g + 1) * EG], axis=1)
                            T.tile.sub(
                                acc_s_ub[g * EG : (g + 1) * EG, :],
                                acc_s_ub[g * EG : (g + 1) * EG, :],
                                bcast_h,
                            )
                        T.tile.exp(acc_s_ub, acc_s_ub)
                        # sumexp = sumexp * alpha + sum(p) (per-group)
                        for g in range(NGRP):
                            T.reduce_sum(
                                acc_s_ub[g * EG : (g + 1) * EG, :],
                                sumexp_i[g * EG : (g + 1) * EG],
                                dim=-1,
                            )
                        T.tile.mul(sumexp, sumexp, m_prev)
                        T.tile.add(sumexp, sumexp, sumexp_i)
                        # acc_o *= alpha: 3D middle-dim region muls replace the
                        # d x v_block row loops (was 256 row ops per block).
                        # Loop order note: d-outer/g-inner keeps broadcast+mul
                        # as siblings in the inner loop (probe t3i3's validated
                        # form). An earlier g-outer nesting also produced
                        # baseline-identical output — the 1274-mismatch smoke
                        # that motivated the swap was a smoke-tolerance false
                        # positive (the iter-2 backup kernel mismatches the
                        # same 1274 on that config; L0/L1 gates are the
                        # authoritative check).
                        for d in range(N_SEG):
                            for g in range(NGRP):
                                T.tile.broadcast(bcast_h, m_prev[g * EG : (g + 1) * EG], axis=1)
                                T.tile.mul(
                                    acc_o_all[d, g * EG : (g + 1) * EG, :],
                                    acc_o_all[d, g * EG : (g + 1) * EG, :],
                                    bcast_h,
                                )
                        # p -> bf16 -> ws_p
                        T.copy(acc_s_ub, cast_bf16)
                        T.copy(
                            cast_bf16,
                            workspace_p[cid, vid * v_block : (vid + 1) * v_block, :],
                        )
                        # acc_o += ws_o chunks (acc_s_ub reused as read slot)
                        for d in range(N_SEG):
                            T.copy(
                                workspace_o[
                                    cid,
                                    vid * v_block : (vid + 1) * v_block,
                                    d * DS : (d + 1) * DS,
                                ],
                                acc_s_ub,
                            )
                            T.tile.add(acc_o_all[d, :, :], acc_o_all[d, :, :], acc_s_ub)

                    # ---- [V tail] normalize + Lse + writeback ----
                    for d in range(N_SEG):
                        for h in range(v_block):
                            T.tile.div(acc_o_all[d, h, :], acc_o_all[d, h, :], sumexp[h])
                    T.tile.ln(sumexp, sumexp)
                    T.tile.add(sumexp, sumexp, m_i)  # Lse = ln(sumexp) + m_i
                    for d in range(N_SEG):
                        T.copy(acc_o_all[d, :, :], acc_s_ub)  # 3D -> 2D fp32
                        T.copy(acc_s_ub, cast_bf16)  # fp32 -> bf16
                        T.copy(
                            cast_bf16,
                            Output[
                                s_row,
                                H0 + vid * v_block : H0 + (vid + 1) * v_block,
                                d * DS : (d + 1) * DS,
                            ],
                        )
                    T.copy(
                        sumexp,
                        Lse[s_row, H0 + vid * v_block : H0 + (vid + 1) * v_block],
                    )

        return main

    return _sparse_mla_fwd_builder


_JIT_BUILDERS: dict = {}


def tl_sparse_mla_fwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    block_I=128,
    grid_size=20,
    target="ascendc",
):
    """Build the sparse_mla_fwd kernel for the requested backend.

    Args:
        heads: total head count H (head_kv = heads // kv_group, see domain asserts).
        dim: QK main / V dimension D (F6b layout: must be 512).
        tail_dim: QK tail dimension (F6b layout: must be 64).
        topk: selected positions per row (topk % block_I == 0).
        kv_group: KV group count G.
        sm_scale: defaults to (dim + tail_dim) ** -0.5.
        block_I: topk-direction tile BI in {64, 128}.
        grid_size: persistent grid (host queries cube_core_num; 910B3 = 20).
        target: "ascendc" (default) or "pto".

    Returns:
        Callable kernel: func(Q, KV, Indices, Offsets, TokenIndices, Output, Lse,
        workspace_kv, workspace_s, workspace_p, workspace_o) — all buffers
        explicit (host torch.zeros), no framework-allocated outputs.
    """
    if target not in _JIT_BUILDERS:
        _JIT_BUILDERS[target] = _make_jit_builder(target)
    return _JIT_BUILDERS[target](heads, dim, tail_dim, topk, kv_group, sm_scale, block_I, grid_size)


# ============================================================================
# Host side: data preparation + call entry (DESIGN §10.4 host audit — only
# TokenIndices precompute (CPU build -> one H2D), torch.zeros allocations and
# the kernel call; no aclnn on the operator main path).
# ============================================================================


def _tensor_cache(fn):
    """Single-entry cache keyed on input tensor identity (ref_gpu/index.py form).

    Keeps prepare_token_indices' CPU->H2D out of steady-state benchmark loops.
    """
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

    CPU construction -> single H2D (reducesum prepare_token_indices form).
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


def sparse_mla_fwd_interface(
    q,
    kv,
    indices,
    offsets,
    sm_scale=None,
    return_p_sum=False,
    d_v=512,
    block_I=128,
    num_stages=2,
    threads=128,
    target="ascendc",
):
    """Host entry (main-repo signature compatible).

    Args:
        q: [Sigma S, heads, dim+tail_dim] bfloat16 (NPU)
        kv: [Sigma S_kv, kv_group, dim+tail_dim] bfloat16 (NPU)
        indices: [Sigma S, kv_group, topk] int32 (NPU), values in {-1} U int32
            (out-of-range clamped for addressing, contribution masked to 0)
        offsets: [batch+1] int32 cu_seqlens
        sm_scale: defaults to (dim + tail_dim) ** -0.5
        return_p_sum: must be False (fwd-only file)
        d_v: V dimension (dim_plus_tail_dim - d_v = tail_dim)
        block_I: topk tile; explicit 64/128 respected, other values fall back to
            the design default 128 (Ascend tiling per DESIGN §5.2)
        num_stages / threads: kept for signature compatibility, unused on Ascend
        target: "ascendc" (default) or "pto"

    Returns:
        (Output [Sigma S, heads, d_v] bfloat16, Lse [Sigma S, heads] float32)
    """
    import torch

    assert return_p_sum == False, "This kernel file is for fwd only"  # noqa: E712  (main-repo verbatim)
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    seq_len, heads, dim_plus_tail_dim = q.shape
    seq_len_kv, kv_group, _ = kv.shape
    assert seq_len == seq_len_kv
    assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = d_v
    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    _, _, topk = indices.shape
    assert indices.shape == (seq_len, kv_group, topk)

    if block_I in (64, 128):
        bi = block_I
    else:
        bi = 128  # design default (§5.2); non-{64,128} user values not honoured

    grid_size = get_cube_core_num()
    key = (heads, dim, tail_dim, topk, kv_group, sm_scale, bi, grid_size, target)
    func = _BUILD_CACHE.get(key)
    if func is None:
        func = tl_sparse_mla_fwd(
            heads=heads,
            dim=dim,
            tail_dim=tail_dim,
            topk=topk,
            kv_group=kv_group,
            sm_scale=sm_scale,
            block_I=bi,
            grid_size=grid_size,
            target=target,
        )
        _BUILD_CACHE[key] = func

    _, _, _, _, Hseg, _ = _derived_head_layout(heads, kv_group)
    token_indices = prepare_token_indices(offsets)  # CPU build -> one H2D

    out = torch.zeros((seq_len, heads, dim), dtype=q.dtype, device=q.device)
    lse = torch.zeros((seq_len, heads), dtype=torch.float32, device=q.device)
    workspace_kv = torch.zeros((grid_size, bi, dim + tail_dim), dtype=torch.bfloat16, device=q.device)
    workspace_s = torch.zeros((grid_size, Hseg, bi), dtype=torch.float32, device=q.device)
    workspace_p = torch.zeros((grid_size, Hseg, bi), dtype=torch.bfloat16, device=q.device)
    workspace_o = torch.zeros((grid_size, Hseg, dim), dtype=torch.float32, device=q.device)
    func(q, kv, indices, offsets, token_indices, out, lse, workspace_kv, workspace_s, workspace_p, workspace_o)
    return out, lse
