"""sparse_mla_fwd: DSA sparse MLA forward (varlen packed layout) for Ascend 910B3.

Two paths behind one host interface (iter-5 hybrid upgrade, user-preapproved):

  * portable  — Developer mode (4-True AUTO pass_configs), dual backend
    (target="ascendc" | "pto"), full F6b support domain. Precision gate path
    (pto) and generic fallback for everything outside the fast domain.
  * fast      — Expert mode (no pass_configs; manual T.Scope("C")/("V") +
    manual set_flag/wait_flag + cross-flag ring protocol), ascendc ONLY,
    added in perf iter-5 to bypass the five AUTO-stack blockers RC1-RC5 that
    deadlocked ring pipelining (perf_tuning/sparse_mla_fwd_perf_log.md
    iteration 4). DEPTH=2 ring slots on ws_kv/ws_s/ws_p/ws_o, LEAD=1
    lookahead gather, 4 cross-flag families x 2 slots = 8 flags. Bit-exact
    vs the portable path (identical op order/dtype at every step; verified
    on NI in {1,2,8} x {short/neg_oob/repeat/single-segment} x 3 reps).

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
    "tl_sparse_mla_fwd_fast",
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
# Fast path (perf iter-5): Expert mode, ascendc only — ring-pipelined variant.
#
# RETIRED FROM DISPATCH (rev2 chunk D): sparse_mla_fwd_interface no longer
# routes here — the v2-lite a3 two-pass kernel (see the rev2 section below)
# supersedes it (the v1/v2 domains nearly coincide and v2 is a strict
# improvement; the chunk-D same-session A/B decides the switch). The builder
# and _fast_gate_ok are kept verbatim for rollback/bisection.
#
# Why Expert mode here: the AUTO_CV_SYNC + MemoryPlanning stack blocks ring
# pipelining on five independent mechanisms (flag budget 16, static statement
# pairing, single-interval liveness aliasing, two planner bugs, Mod region
# index faults — perf_log iteration 4, RC1-RC5, all source-verified). Manual
# cross flags + a hand-planned UB layout (no MEMORY_PLANNING, sum of buffers)
# bypass all five. Structure blueprint: sparse_mla_topk_reducesum.py (Expert
# C/V scopes + cross-flag protocol) + the t3i4 rc clamp-ring probe.
#
# Ring protocol (flat block stream n = k*NI + t, DEPTH=2 slots, LEAD=1):
#   V body(n) = [gather(n+1) if in range; post(n)]
#   C body(n) = [wait F0(n); ws_kv->L1; QK; FIX ws_s; set F1A(n);
#                wait F2(n); ws_p->L1; PV; FIX ws_o; set F1B(n)]
#   F0  ids 0,1  V->C (armed MTE3, mode-2 quorum over both AIVs): slot bulk drained
#   F1A ids 2,3  C->V (armed FIX): ws_s slot scores visible
#   F2  ids 4,5  V->C (armed MTE3): ws_p slot p-tile stored
#   F1B ids 6,7  C->V (armed FIX): ws_o slot PV chunk visible
# 8 ids <= the 16-slot hardware budget; every set/wait sits inside the valid
# block segments (ntask/nblk loop bounds) with exact 1:1 counts (reducesum
# DESIGN §7.2 anti-dangling discipline). Slot-reuse safety (DEPTH=2, LEAD=1):
#   * ws_kv slot m rewritten by bulk(m+2) in V body(m+1) — program-ordered
#     after post(m)'s F1A(m) wait, which transitively implies C consumed
#     ws_kv slot m (F1A is set after QK(m), which waited F0(m));
#   * ws_s/ws_p/ws_o slots: same argument through the F-chain + V serial
#     program order (post(n+1) starts after post(n) fully completes);
#   * kv_stage (single slot): explicit mte3->mte2 pairs between batches and
#     at gather start (the previous gather's last bulk is NOT otherwise
#     covered — gather(n+1) precedes post(n)'s F1A wait in program order).
#
# UB layout (Expert mode: every buffer gets its own address, no liveness
# aliasing; total 194,560B <= 196,352B incl. the injected tmp_ub ~2KB):
#   acc_o_all 128K + acc_s_ub 32K + kv_stage[16,576] 18K + bcast_h 4K
#   + cast_g 2K + 1D chains ~5K. GKB=16 (4 gather batches) is a budget
#   compromise: GKB=32 does not fit next to acc_o_all + acc_s_ub.
#
# p-store discipline (probe t3i5 probe_a_only): a single [8,128] cast_g slot
# requires per-group INTERLEAVED cast/store with mte3->v / v->mte3 pairs —
# batching [all casts; all stores] makes every store read the final group's
# cast (deterministic 7/8 corruption).
# ============================================================================


def _fast_gate_ok(heads, dim, tail_dim, topk, kv_group, block_I):
    """Host mirror of the fast-path compile-time domain (perf main family)."""
    return dim == 512 and tail_dim == 64 and block_I == 128 and kv_group == 1 and heads == 128 and topk % 128 == 0


@tilelang.jit(target="ascendc")  # no pass_configs -> Expert mode (manual sync)
def tl_sparse_mla_fwd_fast(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    block_I=128,
    grid_size=20,
):
    """Build the ring-pipelined fast kernel (Expert mode, ascendc only).

    Domain (asserts mirror _fast_gate_ok): (dim, tail_dim) == (512, 64),
    block_I == 128, (heads, kv_group) == (128, 1) — so Hseg=128, v_block=64
    and every g index is 0 — topk % 128 == 0 (NI >= 1). Everything else
    stays on the portable path via sparse_mla_fwd_interface dispatch.

    Returns:
        Callable kernel: func(Q, KV, Indices, Offsets, TokenIndices, Output,
        Lse, ws_kv [grid, 256, 576] bf16, ws_s [grid, 256, 128] fp32,
        ws_p [grid, 256, 128] bf16, ws_o [grid, 256, 512] fp32).
    """
    # ---- compile-time asserts (fast domain; host mirror: _fast_gate_ok) ----
    assert dim == 512 and tail_dim == 64, f"(dim, tail_dim) must be (512, 64), got ({dim}, {tail_dim})"
    assert block_I == 128, f"fast path requires block_I=128, got {block_I}"
    assert kv_group == 1 and heads == 128, f"fast path requires (heads=128, kv_group=1), got ({heads}, {kv_group})"
    assert topk % block_I == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (dim + tail_dim) ** -0.5

    # structural constants (fixed by the domain asserts above)
    HSEG, V_BLOCK, BI = 128, 64, 128
    HALF = 64
    DDT, DIM, TAIL = 576, 512, 64
    DS, N_SEG = 128, 4
    GKB = 16  # gather staging rows per batch (UB budget)
    NBATCH = HALF // GKB  # 4
    EG = 8  # softmax group rows (EG=8: VEC 32B alignment, iter-3)
    NGRP = V_BLOCK // EG  # 8
    DEPTH = 2  # ring slots
    # lookahead distance LEAD=1 (structural: body(n) gathers block n+1)
    IDXSLOTS = 2  # indices_ub ring (>= LEAD+1; rewrite covered by the
    # gather-head v->mte2 pair against the previous post's mask-chain reads)
    F0_BASE, F1A_BASE, F2_BASE, F1B_BASE = 0, 2, 4, 6  # 4 families x DEPTH
    NI = topk // BI

    dtype = "bfloat16"
    indices_dtype = "int32"
    accum_dtype = "float32"

    seq_len = T.symbolic("seq_len")  # Sigma S
    seq_len_kv = T.symbolic("seq_len_kv")  # Sigma S_kv
    batch_plus_one = T.symbolic("batch_plus_one")

    @T.prim_func
    def main(
        Q: T.Tensor([seq_len, heads, DDT], dtype),  # type: ignore
        KV: T.Tensor([seq_len_kv, kv_group, DDT], dtype),  # type: ignore
        Indices: T.Tensor([seq_len, kv_group, topk], indices_dtype),  # type: ignore
        Offsets: T.Tensor([batch_plus_one], indices_dtype),  # type: ignore
        TokenIndices: T.Tensor([seq_len, 2], indices_dtype),  # type: ignore
        Output: T.Tensor([seq_len, heads, DIM], dtype),  # type: ignore
        Lse: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
        ws_kv: T.Tensor([grid_size, DEPTH * BI, DDT], dtype),  # type: ignore
        ws_s: T.Tensor([grid_size, DEPTH * HSEG, BI], accum_dtype),  # type: ignore
        ws_p: T.Tensor([grid_size, DEPTH * HSEG, BI], dtype),  # type: ignore
        ws_o: T.Tensor([grid_size, DEPTH * HSEG, DIM], accum_dtype),  # type: ignore
    ):
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):
            # ---- Cube side: L1 / L0C (same budget as the portable path) ----
            v_all = T.alloc_L1([N_SEG, BI, DS], dtype)
            tail_l1 = T.alloc_L1([BI, TAIL], dtype)
            q_all = T.alloc_L1([N_SEG, HSEG, DS], dtype)
            qt_l1 = T.alloc_L1([HSEG, TAIL], dtype)
            p_l1 = T.alloc_L1([HSEG, BI], dtype)
            acc_s_l0c = T.alloc_L0C([HSEG, BI], accum_dtype)
            acc_o_l0c = T.alloc_L0C([HSEG, DS], accum_dtype)
            # ---- Vector side: UB (194,560B total <= 196,352B; see header) ----
            acc_o_all = T.alloc_ub([N_SEG, V_BLOCK, DS], accum_dtype)  # 128KB persistent
            acc_s_ub = T.alloc_ub([V_BLOCK, BI], accum_dtype)  # 32KB sm scratch / read slot
            kv_stage = T.alloc_ub([GKB, DDT], dtype)  # 18KB gather staging (4 batches)
            cast_g = T.alloc_ub([EG, BI], dtype)  # 2KB p/out bf16 cast slot
            bcast_h = T.alloc_ub([EG, BI], accum_dtype)  # 4KB broadcast scratch
            indices_ub = T.alloc_ub([IDXSLOTS, BI], indices_dtype)
            rows_ub = T.alloc_ub([BI], indices_dtype)
            m_i = T.alloc_ub([V_BLOCK], accum_dtype)
            m_prev = T.alloc_ub([V_BLOCK], accum_dtype)
            sumexp = T.alloc_ub([V_BLOCK], accum_dtype)
            sumexp_i = T.alloc_ub([V_BLOCK], accum_dtype)
            idx_f = T.alloc_ub([BI], accum_dtype)
            t_ub = T.alloc_ub([BI], accum_dtype)  # min-form mask chain result
            m1_ub = T.alloc_ub([BI], accum_dtype)  # window term -> bias

            total = seq_len * kv_group
            ntask = T.max(0, T.ceildiv(total - cid, grid_size))
            nblk = ntask * NI  # flat block count (C/V/flag protocol domain)

            # ================= Cube stream =================
            # No manual intra-core flags: every L1/L0C hazard is covered by
            # gemm_v0's internal head/tail flags (MTE2->MTE1 operand landing,
            # FIX->M L0C WAR, MTE1->MTE2 L1 WAR, M->FIX landing) — the
            # reducesum Expert-mode precedent (common.h L1306-1370).
            with T.Scope("C"):
                for k in T.serial(ntask):
                    pid = k * grid_size + cid
                    s_row = pid % seq_len  # g == 0 in the fast domain
                    # Q load per task (no V involvement, no flag needed)
                    for s in range(N_SEG):
                        T.copy(Q[s_row, 0:HSEG, s * DS : (s + 1) * DS], q_all[s, :, :])
                    T.copy(Q[s_row, 0:HSEG, DIM:DDT], qt_l1)
                    for t in T.serial(NI):
                        n = k * NI + t
                        slot = n % DEPTH
                        T.wait_cross_flag(F0_BASE + slot)
                        for s in range(N_SEG):
                            T.copy(
                                ws_kv[cid, slot * BI : (slot + 1) * BI, s * DS : (s + 1) * DS],
                                v_all[s, :, :],
                            )
                        T.copy(ws_kv[cid, slot * BI : (slot + 1) * BI, DIM:DDT], tail_l1)
                        # QK: K = DS * N_SEG + tail, M = HSEG
                        T.gemm_v0(q_all[0, :, :], v_all[0, :, :], acc_s_l0c, transpose_B=True, init=True)
                        for s in range(1, N_SEG):
                            T.gemm_v0(q_all[s, :, :], v_all[s, :, :], acc_s_l0c, transpose_B=True, init=False)
                        T.gemm_v0(qt_l1, tail_l1, acc_s_l0c, transpose_B=True, init=False)
                        T.copy(acc_s_l0c, ws_s[cid, slot * HSEG : (slot + 1) * HSEG, :])
                        T.set_cross_flag("FIX", F1A_BASE + slot)
                        # PV (p arrives from V's softmax of this block)
                        T.wait_cross_flag(F2_BASE + slot)
                        T.copy(ws_p[cid, slot * HSEG : (slot + 1) * HSEG, :], p_l1)
                        for s in range(N_SEG):
                            T.gemm_v0(p_l1, v_all[s, :, :], acc_o_l0c, init=True)
                            T.copy(
                                acc_o_l0c,
                                ws_o[cid, slot * HSEG : (slot + 1) * HSEG, s * DS : (s + 1) * DS],
                            )
                        T.set_cross_flag("FIX", F1B_BASE + slot)

            # ================= Vector stream =================
            with T.Scope("V"):
                if ntask > 0:
                    # ---- prologue: gather(0) (LEAD=1) with task-0 scalars ----
                    s_row0 = cid % seq_len
                    b0 = TokenIndices[s_row0, 0]
                    bos0 = Offsets[b0]
                    T.copy(Indices[s_row0, 0, 0:BI], indices_ub[0, :])
                    T.set_flag("mte2", "v", 0)
                    T.wait_flag("mte2", "v", 0)
                    # vectorized pre-clamp rows (int32 domain, reducesum form):
                    # rows = clamp(idx + bos, 0, S_kv - 1); clamp protects
                    # addressing only, contributions are masked (note below)
                    T.tile.add(rows_ub, indices_ub[0, :], bos0)
                    T.tile.clamp(rows_ub, rows_ub, 0, seq_len_kv - 1, BI)
                    T.set_flag("v", "s", 0)
                    T.wait_flag("v", "s", 0)
                    for b in range(NBATCH):
                        T.set_flag("mte3", "mte2", 0)
                        T.wait_flag("mte3", "mte2", 0)
                        for j in range(GKB):
                            kv_row = rows_ub[vid * HALF + b * GKB + j]
                            T.copy(KV[kv_row, 0, :], kv_stage[j, :])
                        T.set_flag("mte2", "mte3", 0)
                        T.wait_flag("mte2", "mte3", 0)
                        T.copy(
                            kv_stage,
                            ws_kv[cid, vid * HALF + b * GKB : vid * HALF + (b + 1) * GKB, :],
                        )
                    T.set_cross_flag("MTE3", F0_BASE + 0)

                    for k in T.serial(ntask):
                        pid = k * grid_size + cid
                        s_row = pid % seq_len
                        b_i = TokenIndices[s_row, 0]
                        s_i = TokenIndices[s_row, 1]
                        bos = Offsets[b_i]
                        # lookahead (task k+1) scalars — clamped memory-safe:
                        # s_row_n is always a legal row so the TokenIndices /
                        # Offsets reads are in-bounds; the VALUES are only
                        # consumed inside the `gn < nblk` gather guard
                        # (reducesum iter8/#18 pattern)
                        pid_n = pid + grid_size
                        s_row_n = T.min(pid_n % seq_len, seq_len - 1)
                        b_n = TokenIndices[s_row_n, 0]
                        bos_n = Offsets[b_n]
                        # softmax state init (after the previous task's
                        # epilogue in V program order — no extra sync needed)
                        for d in range(N_SEG):
                            T.tile.fill(acc_o_all[d, :, :], 0.0)
                        T.tile.fill(sumexp, 0.0)
                        T.tile.fill(m_i, -(2.0**30))  # avoid -inf - inf NaN

                        for t in T.serial(NI):
                            n = k * NI + t
                            # ---- gather(n+1): lookahead, 4x[16-row] batches ----
                            if n + 1 < nblk:
                                gn = n + 1
                                nxt = t + 1 >= NI  # gn belongs to task k+1?
                                nslot = gn % DEPTH
                                islot = gn % IDXSLOTS
                                tg = (t + 1) % NI  # gn's block index in its task
                                s_row_g = T.if_then_else(nxt, s_row_n, s_row)
                                # indices_ub rewrite protection: prior V
                                # readers = post(n-1)'s mask chain (same slot)
                                T.set_flag("v", "mte2", 0)
                                T.wait_flag("v", "mte2", 0)
                                T.copy(
                                    Indices[s_row_g, 0, tg * BI : (tg + 1) * BI],
                                    indices_ub[islot, :],
                                )
                                T.set_flag("mte2", "v", 0)
                                T.wait_flag("mte2", "v", 0)
                                # rows_ub rewrite protection: prior S-pipe
                                # GetValue readers = gather(n)'s row loop
                                T.set_flag("s", "v", 0)
                                T.wait_flag("s", "v", 0)
                                if nxt:
                                    # GM scalars must stay in direct-read form
                                    # (select-wrapped scalars break the Adds
                                    # codegen — reducesum L530 lesson)
                                    T.tile.add(rows_ub, indices_ub[islot, :], bos_n)
                                else:
                                    T.tile.add(rows_ub, indices_ub[islot, :], bos)
                                T.tile.clamp(rows_ub, rows_ub, 0, seq_len_kv - 1, BI)
                                T.set_flag("v", "s", 0)
                                T.wait_flag("v", "s", 0)
                                for b in range(NBATCH):
                                    # kv_stage WAR: drain the previous bulk
                                    # (previous batch here; the previous
                                    # gather's last bulk for b == 0)
                                    T.set_flag("mte3", "mte2", 0)
                                    T.wait_flag("mte3", "mte2", 0)
                                    for j in range(GKB):
                                        kv_row = rows_ub[vid * HALF + b * GKB + j]
                                        T.copy(KV[kv_row, 0, :], kv_stage[j, :])
                                    T.set_flag("mte2", "mte3", 0)
                                    T.wait_flag("mte2", "mte3", 0)
                                    T.copy(
                                        kv_stage,
                                        ws_kv[
                                            cid,
                                            nslot * BI + vid * HALF + b * GKB : nslot * BI + vid * HALF + (b + 1) * GKB,
                                            :,
                                        ],
                                    )
                                T.set_cross_flag("MTE3", F0_BASE + nslot)

                            # ---- post(n): scores read, online softmax, p
                            #      store, acc_o update (all forms = iter-3) ----
                            slot = n % DEPTH
                            T.wait_cross_flag(F1A_BASE + slot)
                            # acc_s_ub MTE2 write vs prior V readers (the
                            # previous post's chain / adds)
                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)
                            T.copy(
                                ws_s[
                                    cid,
                                    slot * HSEG + vid * V_BLOCK : slot * HSEG + (vid + 1) * V_BLOCK,
                                    :,
                                ],
                                acc_s_ub,
                            )
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            # mask chain: mask = clamp01(min(idx + 1,
                            # s_i + 1 - idx)); bias = BIGC * (1 - mask).
                            # Lse stays exact <=> QK + mask + softmax chain
                            # identical to the portable path (bit-exact oracle)
                            T.copy(indices_ub[n % IDXSLOTS, :], idx_f)  # int32 -> fp32
                            T.tile.add(t_ub, idx_f, T.float32(1.0))
                            T.tile.mul(m1_ub, idx_f, T.float32(-1.0))
                            T.tile.add(m1_ub, m1_ub, T.float32(s_i + 1))
                            T.tile.min(t_ub, t_ub, m1_ub)
                            T.tile.max(t_ub, t_ub, T.float32(0.0))
                            T.tile.min(t_ub, t_ub, T.float32(1.0))
                            T.tile.mul(m1_ub, t_ub, T.float32(-BIGC))
                            T.tile.add(m1_ub, m1_ub, T.float32(BIGC))
                            # online softmax (iter-3 group-vectorized form)
                            T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                            T.tile.broadcast(bcast_h, t_ub, axis=0)
                            for g in range(NGRP):
                                T.tile.mul(
                                    acc_s_ub[g * EG : (g + 1) * EG, :],
                                    acc_s_ub[g * EG : (g + 1) * EG, :],
                                    bcast_h,
                                )
                            T.tile.broadcast(bcast_h, m1_ub, axis=0)
                            for g in range(NGRP):
                                T.tile.sub(
                                    acc_s_ub[g * EG : (g + 1) * EG, :],
                                    acc_s_ub[g * EG : (g + 1) * EG, :],
                                    bcast_h,
                                )
                            T.copy(m_i, m_prev)
                            for g in range(NGRP):
                                T.reduce_max(
                                    acc_s_ub[g * EG : (g + 1) * EG, :],
                                    m_i[g * EG : (g + 1) * EG],
                                    dim=-1,
                                )
                            T.tile.max(m_i, m_i, m_prev)
                            T.tile.sub(m_prev, m_prev, m_i)
                            T.tile.exp(m_prev, m_prev)  # alpha
                            for g in range(NGRP):
                                T.tile.broadcast(bcast_h, m_i[g * EG : (g + 1) * EG], axis=1)
                                T.tile.sub(
                                    acc_s_ub[g * EG : (g + 1) * EG, :],
                                    acc_s_ub[g * EG : (g + 1) * EG, :],
                                    bcast_h,
                                )
                            T.tile.exp(acc_s_ub, acc_s_ub)
                            for g in range(NGRP):
                                T.reduce_sum(
                                    acc_s_ub[g * EG : (g + 1) * EG, :],
                                    sumexp_i[g * EG : (g + 1) * EG],
                                    dim=-1,
                                )
                            T.tile.mul(sumexp, sumexp, m_prev)
                            T.tile.add(sumexp, sumexp, sumexp_i)
                            for d in range(N_SEG):
                                for g in range(NGRP):
                                    T.tile.broadcast(bcast_h, m_prev[g * EG : (g + 1) * EG], axis=1)
                                    T.tile.mul(
                                        acc_o_all[d, g * EG : (g + 1) * EG, :],
                                        acc_o_all[d, g * EG : (g + 1) * EG, :],
                                        bcast_h,
                                    )
                            # p store: per-group INTERLEAVED cast + store (a
                            # single cast_g slot cannot batch the two loops —
                            # probe t3i5 probe_a_only: every store would read
                            # the final group's cast)
                            for g in range(NGRP):
                                T.set_flag("mte3", "v", 0)
                                T.wait_flag("mte3", "v", 0)
                                T.copy(acc_s_ub[g * EG : (g + 1) * EG, :], cast_g)
                                T.set_flag("v", "mte3", 0)
                                T.wait_flag("v", "mte3", 0)
                                T.copy(
                                    cast_g,
                                    ws_p[
                                        cid,
                                        slot * HSEG + vid * V_BLOCK + g * EG : slot * HSEG + vid * V_BLOCK + (g + 1) * EG,
                                        :,
                                    ],
                                )
                            T.set_cross_flag("MTE3", F2_BASE + slot)
                            # acc_o += ws_o chunks (acc_s_ub reused as the
                            # read slot; per-chunk landing + rewrite pairs)
                            T.wait_cross_flag(F1B_BASE + slot)
                            for d in range(N_SEG):
                                T.set_flag("v", "mte2", 0)
                                T.wait_flag("v", "mte2", 0)
                                T.copy(
                                    ws_o[
                                        cid,
                                        slot * HSEG + vid * V_BLOCK : slot * HSEG + (vid + 1) * V_BLOCK,
                                        d * DS : (d + 1) * DS,
                                    ],
                                    acc_s_ub,
                                )
                                T.set_flag("mte2", "v", 0)
                                T.wait_flag("mte2", "v", 0)
                                T.tile.add(acc_o_all[d, :, :], acc_o_all[d, :, :], acc_s_ub)

                        # ---- epilogue (task tail): normalize + Lse + Output ----
                        for d in range(N_SEG):
                            for h in range(V_BLOCK):
                                T.tile.div(acc_o_all[d, h, :], acc_o_all[d, h, :], sumexp[h])
                        T.tile.ln(sumexp, sumexp)
                        T.tile.add(sumexp, sumexp, m_i)  # Lse = ln(sumexp) + m_i
                        T.set_flag("v", "mte3", 0)
                        T.wait_flag("v", "mte3", 0)
                        T.copy(sumexp, Lse[s_row, vid * V_BLOCK : (vid + 1) * V_BLOCK])
                        for d in range(N_SEG):
                            T.copy(acc_o_all[d, :, :], acc_s_ub)  # 3D -> 2D fp32
                            for g in range(NGRP):
                                T.set_flag("mte3", "v", 0)
                                T.wait_flag("mte3", "v", 0)
                                T.copy(acc_s_ub[g * EG : (g + 1) * EG, :], cast_g)
                                T.set_flag("v", "mte3", 0)
                                T.wait_flag("v", "mte3", 0)
                                T.copy(
                                    cast_g,
                                    Output[
                                        s_row,
                                        vid * V_BLOCK + g * EG : vid * V_BLOCK + (g + 1) * EG,
                                        d * DS : (d + 1) * DS,
                                    ],
                                )

    return main


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


_FP16_PRECAST_STATE = {"src": None, "q16": None, "kv16": None, "ok": False}


def _fp16_precast(q, kv):
    """A1a host precast + value-range predicate for the v2 fast path.

    Returns (q_fp16, kv_fp16, ok); ok=False -> the caller falls back to
    the portable path (dispatch-tree pair update, DESIGN §11.4).
    bf16 -> fp16 is value-exact for |x| <= 65504 (mantissa 8 < 10);
    NaN/inf amax/amin comparisons are False, so non-finite inputs also
    fall back. Single-entry identity cache (strong refs) keeps the amax
    reductions and the casts out of steady-state benchmark loops.
    """
    import torch

    st = _FP16_PRECAST_STATE
    if st["src"] is not None and st["src"][0] is q and st["src"][1] is kv:
        return st["q16"], st["kv16"], st["ok"]
    ok = False
    q16 = kv16 = None
    if q.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16:
        if (
            q.amax().item() <= 65504.0
            and q.amin().item() >= -65504.0
            and kv.amax().item() <= 65504.0
            and kv.amin().item() >= -65504.0
        ):
            q16 = q.to(torch.float16)
            kv16 = kv.to(torch.float16)
            ok = True
    _FP16_PRECAST_STATE.update({"src": (q, kv), "q16": q16, "kv16": kv16, "ok": ok})
    return q16, kv16, ok


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
        block_I: topk tile for the portable path; explicit 64/128 respected,
            other values fall back to the design default 128 (DESIGN §5.2).
            The v2 fast domain (rev2 chunk D) ignores it: that kernel's
            BI=256 is structural — the gate judges (topk, H, G, dim, tail).
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

    # ---- fast/generic dispatch (rev2 chunk D): the v2-lite a3 two-pass kernel
    # is the ONLY fast path — fast v1 is RETIRED from dispatch (domains nearly
    # coincide, v2 a strict improvement; v1 builder kept above for rollback),
    # so out-of-v2-domain shapes (topk % 256 != 0, NI > 6, kv_group=2,
    # heads=64, pto backend, ...) go straight to the portable path.
    # The gate is STRUCTURAL (chunk-D ruling): user block_I is a portable-path
    # tiling hint only — the v2 kernel's internal BI=256 is passed here (the
    # L0/L1 test configs pass block_I=128; the gate judges (topk, H, G, dim,
    # tail)). The DESIGN §12.2② value-range predicate (amax(Q/KV) <= 65504) is
    # NOT applied: v2 is bf16 all-chain (no fp16 cast anywhere — only fp32->bf16
    # rounding on the P store, chunk-C verified), so the fp16 overflow bound is
    # meaningless for this path; dispatch is on the gate domain alone.
    use_fast = target == "ascendc" and _fast_v2_gate_ok(heads, dim, tail_dim, topk, kv_group, 256)
    if use_fast:
        # A1a fp16 KV route: precast + value-range predicate; any miss
        # (non-bf16 input, |x| > 65504, NaN/inf) falls back portable.
        q16, kv16, fp16_ok = _fp16_precast(q, kv)
        use_fast = fp16_ok
    if use_fast:
        ni = topk // 256
        fkey = ("fastv2", heads, topk, sm_scale, grid_size)
        ffunc = _BUILD_CACHE.get(fkey)
        if ffunc is None:
            ffunc = tl_sparse_mla_fwd_fast_v2(
                heads=heads,
                dim=dim,
                tail_dim=tail_dim,
                topk=topk,
                kv_group=kv_group,
                sm_scale=sm_scale,
                block_I=256,
                grid_size=grid_size,
            )
            _BUILD_CACHE[fkey] = ffunc
        token_indices = prepare_token_indices(offsets)  # CPU build -> one H2D
        out = torch.zeros((seq_len, heads, dim), dtype=q.dtype, device=q.device)
        lse = torch.zeros((seq_len, heads), dtype=torch.float32, device=q.device)
        # v2 workspaces (a3 two-pass, chunk C): ws_kv ring DEPTH=NI+LEAD slots
        # of [256, 576] fp16 (A3a LEAD=2 for NI>=2; A1a fp16 route); ws_s NI
        # slots fp32; ws_p NI slots fp16; ws_o one [HSEG, DIM] bf16 transit
        # buffer per core (A1b; single flush per task/MSlice)
        depth = ni + (2 if ni >= 2 else 1)
        workspace_kv = torch.zeros((grid_size, depth * 256, dim + tail_dim), dtype=torch.float16, device=q.device)
        workspace_s = torch.zeros((grid_size, ni * 128, 256), dtype=torch.float32, device=q.device)
        workspace_p = torch.zeros((grid_size, ni * 128, 256), dtype=torch.float16, device=q.device)
        workspace_o = torch.zeros((grid_size, 128, dim), dtype=torch.bfloat16, device=q.device)
        ffunc(q16, kv16, indices, offsets, token_indices, out, lse, workspace_kv, workspace_s, workspace_p, workspace_o)
        return out, lse

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


# ============================================================================
# rev2 fast v2-lite (a3 two-pass; DESIGN §12 evidence-driven fallback).
#
# Probe evidence (2026-10-15, /tmp/opencode/t3r2a/log.md, probes p1-p8):
#   * §12.2(c) fp16 UB->L1 direct connect is DEAD on 910B3/A2:
#     - DSL T.copy(ub, l1): AscendWorkspaceReduction forces a GM-workspace
#       roundtrip (needs_gm_workspace_ = platform != "A5",
#       ascend_workspace_reduction.cc:137) AND collapses row-offset dsts into
#       a single workspace slot (p1: 16384/16384 mismatched, source dump).
#     - raw tl::ascend::copy_ub_to_l1 via T._src_code: aicore hang in all 4
#       orientation/template combos (p4) and in the isolated no-gemm micro
#       kernel (p6; the flag-protocol CTRL kernel completes).
#   * C-side scalar-row GM->L1 gather is DEAD: 1-row copies emit inconsistent
#     template/offset params (p7/p7b dump: <bf16,256,256> template with
#     runtime m=1, dst offset g*256 non-fractal-aligned).
#   * What IS validated: K=576 whole-buffer gemm kL0Size=64 (g1, 1-ULP vs
#     bf16 ref), 3-segment QK init chain + v_all 3D B operands (fast v1),
#     3D L0C slots as gemm C operand with init=False cross-use accumulation
#     (p8), BI=256/M=64x2 tiling, a3 two-pass softmax (DESIGN §12.1 a3).
#
# Architecture (v2-lite, all proven forms):
#   KV route = rev1 (V-side row gather -> ws_kv GM ring -> C bulk GM->L1);
#   a3 two-pass per task: pass1 C does all NI QKs -> ws_s (NI slots, no
#   intra-task reuse: pass2 re-reads), V reduces m_final; pass2 V re-reads
#   ws_s + recomputes P = exp(s_masked - m_final) -> ws_p, C accumulates
#   PV in L0C acc3d[2, 64, 256] across blocks (init=False) -> ONE ws_o
#   flush per (task, M-slice) at the tail (16.8GB -> 2.1GB vs rev1);
#   M-slice = vid (HSEG 128 = 2 x 64); BI=256 (blocks 32768 -> 16384).
#   Flag budget: F0[2] (ws_kv) + F1A[NI] (ws_s) + F2[NI] (ws_p) + F3[2]
#   = 2*NI + 4 <= 16 -> NI <= 6 (asserted).
#
# A3 PASS-2 ROUTE (decided chunk C; orchestrator ruling + probe evidence):
# ws_kv ring RE-READ — v_all holds only ONE block's KV at a time, so pass-2
# PV re-reads each block's KV band from ws_kv (rev1-proven ws->L1 bulk
# read, zero new forms). DEPTH = NI + 1, NOT the planned NI: task k's NI
# slots must stay intact from the pass-1 write until the pass-2 re-read
# (F2-chain fenced, i.e. past V's WHOLE pass 2), while the LEAD=1 lookahead
# of task k+1's block 0 fires during V's pass-1 iteration NI-1 — NI live
# blocks + 1 lookahead = NI+1 blocks in flight, so DEPTH = NI is
# pigeonhole-unsafe (the ruling's "no overwrite" note is falsified for the
# block-0 slot; +1 slot = +5.8MB ws_kv @grid 20). Pass-2 adder @main shape
# (S=4096, topk=1024, NI=4): KV re-read 2*NI*128KB + P read 2*NI*64KB per
# task ≈ 6.3GB (the two N-seg passes double the P read; a single shared
# pass would need 4 x [64,256] fp32 L0C accumulators = 256KB > 128KB L0C).
# Rejected variants (kept on file per the chunk-C mandate):
#   * QK recompute: +2.1GB ws_s w + 2.1GB r + double QK gemm time ON TOP
#     of the same KV re-read — pure adder, no mechanism win.
#   * Single-pass half-width P pipeline (rev1 structure at BI=256): ~30.8GB
#     but V-UB overflows (acc_o_all 128KB + full-width s_ub 64KB + staging
#     ~ 232KB > 192KB) unless P half-streams through a 4D v_all
#     [2,2,128,256] + p_l1 [2,64,128] + 6-gemm chain — mechanism unproven.
# Traffic bottom line: a3 total lands at 31.4-34.6GB — misses the 24288us
# target (~41-47ms @ 0.74TB/s), which was premised on the dead direct path
# (14.3GB). Stage-1 re-scope still required.
#
# KNOWN DEVIATIONS vs DESIGN §12 (design-error fallout, needs Stage-1 re-
# approval): no fp16 cast / no host precast / no P-resident L1 (a2 dead);
# a3 replaces a2. All-invalid-row corner: the -(2**30) m_fin floor sits
# ABOVE -BIGC (1e30), so P = exp(-BIGC + 2**30) still underflows to 0 and
# the epilogue hits 0/0 -> NaN/-inf — the SAME corner behavior as rev1
# (the earlier "P=1 saturation" note was an arithmetic slip; fixed chunk B).
# ============================================================================


def _fast_v2_gate_ok(heads, dim, tail_dim, topk, kv_group, block_I):
    """Host mirror of the v2-lite compile-time domain."""
    ni = topk // block_I if topk % block_I == 0 else 99
    return (
        dim == 512
        and tail_dim == 64
        and block_I == 256
        and kv_group == 1
        and heads == 128
        and topk % block_I == 0
        and 1 <= ni <= 5  # flag budget: LEAD=2 (NI>=2) 2*NI+5 <= 16; NI=1 keeps 2*NI+4
    )


@tilelang.jit(target="ascendc")  # Expert mode (manual sync)
def tl_sparse_mla_fwd_fast_v2(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    block_I=256,
    grid_size=20,
):
    """Build the a3 two-pass BI=256 fast kernel (v2-lite; see section header).

    Domain (asserts mirror _fast_v2_gate_ok): (dim, tail_dim) == (512, 64),
    block_I == 256, (heads, kv_group) == (128, 1), topk % 256 == 0 and
    NI = topk // 256 in [1, 6] (cross-flag budget 2*NI + 4 <= 16).

    Returns:
        Callable kernel: func(Q, KV, Indices, Offsets, TokenIndices, Output,
        Lse, ws_kv [grid, (NI+1)*256, 576] bf16, ws_s [grid, NI*128, 256] fp32,
        ws_p [grid, NI*128, 256] bf16, ws_o [grid, 128, 512] fp32).
    """
    # ---- compile-time asserts (v2-lite domain; host mirror: _fast_v2_gate_ok) ----
    assert dim == 512 and tail_dim == 64, f"(dim, tail_dim) must be (512, 64), got ({dim}, {tail_dim})"
    assert block_I == 256, f"v2-lite path requires block_I=256, got {block_I}"
    assert kv_group == 1 and heads == 128, f"v2-lite requires (heads=128, kv_group=1), got ({heads}, {kv_group})"
    assert topk % block_I == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    NI = topk // block_I
    assert 1 <= NI <= 5, f"v2-lite flag budget requires NI=topk/256 in [1,5], got {NI}"
    if sm_scale is None:
        sm_scale = (dim + tail_dim) ** -0.5

    # structural constants (fixed by the domain asserts above)
    HSEG, V_BLOCK, BI = 128, 64, 256  # heads, per-vid head slice, block cols
    HALF = 128  # per-vid gather rows (BI / 2 vids)
    DDT, DIM, TAIL = 576, 512, 64
    DS, N_SEG = 256, 2  # QK K-seg / PV N-seg / o-chunk width (DS*2 + TAIL = DDT)
    GKB = 16  # gather staging rows per batch
    NBATCH = HALF // GKB  # 8
    EG = 8  # softmax group rows (VEC 32B alignment, iter-3)
    NGRP = V_BLOCK // EG  # 8
    LEAD = 2 if NI >= 2 else 1  # A3a: deepen the V gather lookahead to 2
    # blocks (V has 13.5ms F3 slack, audit §3.4; NI=1 keeps the proven
    # LEAD=1 form — its LEAD=2 lookahead would cross TWO task boundaries)
    DEPTH = NI + LEAD  # ws_kv ring slots: task k's NI blocks live from the
    # pass-1 write until the pass-2 re-read (F2-chain fenced) + LEAD
    # lookahead blocks of task k+1 gathered during V's pass-1 tail iters
    # (pigeonhole argument — see the section-header A3 PASS-2 ROUTE note)
    IDXSLOTS = LEAD + 1  # indices ring: post1(n) reads block n's slot
    # while the loop gathers n+LEAD (slots n..n+LEAD distinct mod
    # LEAD+1; pass2 reuses the per-block masks saved in mask_mem, chunk B)
    KL0 = 64  # gemm_v0 kL0Size (B slot 256*64*2 = 32KB == multi-step budget)
    # flag families: F0 ws_kv gather (V->C, ids n % (LEAD+1) — with
    # LEAD=2 the V gather runs 2 blocks ahead of C's QK, so the set for
    # block m fires after V's F1A(m-LEAD-1) wait = C finished QK(m-LEAD-1)
    # = C already consumed the block m-(LEAD+1) flag: period LEAD+1 keeps
    # the set/wait alternation exact (parity-2 is ONLY safe for LEAD=1;
    # chunk C argument re-derived for depth), F1A ws_s slot (C->V, NI),
    # F2 ws_p slot (V->C, NI), F3 ws_o flush (C->V, BOTH ids set after
    # the complete pass-2; V waits BOTH — its next-task gathers may only
    # fire past C's whole pass 2, ws_kv ring safety)
    F0IDS = LEAD + 1
    F0_BASE, F1A_BASE, F2_BASE, F3_BASE = 0, F0IDS, F0IDS + NI, F0IDS + 2 * NI

    dtype = "bfloat16"
    # A1a fp16 KV route: Q/KV/ws_kv/ws_p + all L1 gemm operands fp16
    # (host precast + value-range predicate amax<=65504; bf16 mantissa 8 <
    # fp16 10 -> precast value-exact in range; subnormal flush of |x|<6e-5
    # is the only deviation, negligible vs the bf16 gates). Output bf16,
    # Lse/ws_s fp32. bf16->fp16 VCast UNSUPPORTED on dav_c220 (R2 probe),
    # hence the host precast of KV as well.
    kv_dtype = "float16"
    indices_dtype = "int32"
    accum_dtype = "float32"

    seq_len = T.symbolic("seq_len")  # Sigma S
    seq_len_kv = T.symbolic("seq_len_kv")  # Sigma S_kv
    batch_plus_one = T.symbolic("batch_plus_one")

    @T.prim_func
    def main(
        Q: T.Tensor([seq_len, heads, DDT], kv_dtype),  # type: ignore
        KV: T.Tensor([seq_len_kv, kv_group, DDT], kv_dtype),  # type: ignore
        Indices: T.Tensor([seq_len, kv_group, topk], indices_dtype),  # type: ignore
        Offsets: T.Tensor([batch_plus_one], indices_dtype),  # type: ignore
        TokenIndices: T.Tensor([seq_len, 2], indices_dtype),  # type: ignore
        Output: T.Tensor([seq_len, heads, DIM], dtype),  # type: ignore
        Lse: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
        ws_kv: T.Tensor([grid_size, DEPTH * BI, DDT], kv_dtype),  # type: ignore
        ws_s: T.Tensor([grid_size, NI * HSEG, BI], accum_dtype),  # type: ignore
        ws_p: T.Tensor([grid_size, NI * HSEG, BI], kv_dtype),  # type: ignore
        ws_o: T.Tensor([grid_size, HSEG, DIM], dtype),  # type: ignore
    ):
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):
            # ---- Cube side: L1 464KB / L0C 128KB (== full) ----
            v_all = T.alloc_L1([N_SEG, BI, DS], kv_dtype)  # KV resident: 256KB
            tail_l1 = T.alloc_L1([BI, TAIL], kv_dtype)  # 32KB
            q_a = T.alloc_L1([N_SEG, V_BLOCK, DS], kv_dtype)  # ms=0 Q: 64KB
            q_b = T.alloc_L1([N_SEG, V_BLOCK, DS], kv_dtype)  # ms=1 Q: 64KB
            qt_all = T.alloc_L1([2, V_BLOCK, TAIL], kv_dtype)  # 16KB
            p_l1 = T.alloc_L1([V_BLOCK, BI], kv_dtype)  # 32KB, reused per block
            # pass1 QK -> slot 0; pass2 PV accumulates slot0/slot1 across
            # blocks (p8-validated 3D-slot + init=False form)
            acc3d = T.alloc_L0C([2, V_BLOCK, BI], accum_dtype)  # 128KB
            # ---- Vector side: UB ~103KB + NI*1KB per vid (<= 109KB @NI=6,
            #      budget 196352B) ----
            s_ub = T.alloc_ub([V_BLOCK, BI], accum_dtype)  # 64KB ws_s slot
            kv_stage = T.alloc_ub([GKB, DDT], kv_dtype)  # 18KB gather staging (fp16)
            cast_g = T.alloc_ub([EG, BI], dtype)  # 4KB cast slot
            bcast_h = T.alloc_ub([EG, BI], accum_dtype)  # 8KB broadcast
            indices_ub = T.alloc_ub([IDXSLOTS, BI], indices_dtype)  # 2KB
            rows_ub = T.alloc_ub([BI], indices_dtype)  # 1KB
            idx_f = T.alloc_ub([BI], accum_dtype)  # 1KB idx -> fp32
            t_ub = T.alloc_ub([BI], accum_dtype)  # 1KB mask chain
            m1_ub = T.alloc_ub([BI], accum_dtype)  # 1KB window term
            mask_mem = T.alloc_ub([NI, BI], accum_dtype)  # <=6KB pass-1 masks
            m_fin = T.alloc_ub([V_BLOCK], accum_dtype)  # task-global max
            sumexp = T.alloc_ub([V_BLOCK], accum_dtype)  # task denom
            sumexp_i = T.alloc_ub([V_BLOCK], accum_dtype)  # rowmax (pass1) /
            # rowsum (pass2) scratch — V-pipe overwrite order, no hazard
            # A1b ws_o bf16 transit (UB tail append): C fixpipe casts
            # fp32->bf16 on the ws_o write; V reads bf16 into o_stage and
            # casts back to fp32 — fp32 accumulator lives in L0C/UB, one
            # transit-grain rounding (iter-4 form, NOT bit-exact).
            o_stage = T.alloc_ub([V_BLOCK, DS], dtype)  # 32KB
            # A1a P-store bit-exact emulation (UB tail append — existing
            # offsets untouched, R-UB-LAYOUT): fp32 -> bf16 (cast_g) -> fp32
            # (cg32, exact) -> fp16 (cg16, exact on bf16 values) reproduces
            # the baseline's bf16 P bits in the fp16 ws_p (probe P2b
            # bit-exact; direct fp32->fp16 differs from torch rounding,
            # P2a — do NOT use).
            cg32 = T.alloc_ub([EG, BI], accum_dtype)  # 8KB
            cg16 = T.alloc_ub([EG, BI], kv_dtype)  # 4KB

            total = seq_len * kv_group
            ntask = T.max(0, T.ceildiv(total - cid, grid_size))
            nblk = ntask * NI  # flat block count (flag protocol domain)

            # ================= Cube stream =================
            # (intra-core L1/L0C hazards covered by gemm_v0 internal flags)
            with T.Scope("C"):
                for k in T.serial(ntask):
                    pid = k * grid_size + cid
                    s_row = pid % seq_len  # g == 0 in the fast domain
                    # ---- pass 1: Q load (col-seg, R4-proven GM form) ----
                    # (literal 2-slice unroll: range() loops are TIR loops,
                    # so python-level buffer selection is not available)
                    for s in range(N_SEG):
                        T.copy(
                            Q[s_row, 0:V_BLOCK, s * DS : (s + 1) * DS],
                            q_a[s, :, :],
                        )
                    T.copy(Q[s_row, 0:V_BLOCK, DIM:DDT], qt_all[0, :, :])
                    for s in range(N_SEG):
                        T.copy(
                            Q[s_row, V_BLOCK : 2 * V_BLOCK, s * DS : (s + 1) * DS],
                            q_b[s, :, :],
                        )
                    T.copy(Q[s_row, V_BLOCK : 2 * V_BLOCK, DIM:DDT], qt_all[1, :, :])
                    # ---- pass 1: all NI QKs -> ws_s ----
                    for t in T.serial(NI):
                        n = k * NI + t
                        slot = n % DEPTH
                        # F0 parity id (n % 2), NOT the slot id: ring depth
                        # NI+1 > flag count 2 — the F1A lag keeps the
                        # set/wait alternation exact (see F0_BASE note)
                        T.wait_cross_flag(F0_BASE + n % F0IDS)
                        for s in range(N_SEG):
                            T.copy(
                                ws_kv[cid, slot * BI : (slot + 1) * BI, s * DS : (s + 1) * DS],
                                v_all[s, :, :],
                            )
                        T.copy(ws_kv[cid, slot * BI : (slot + 1) * BI, DIM:DDT], tail_l1)
                        # QK per M-slice: 3-seg init chain (fast-v1 form;
                        # kL0Size=64 -> B slot 256*64*2 = 32KB == budget)
                        # ms=0 (heads 0:64)
                        T.gemm_v0(
                            q_a[0, :, :],
                            v_all[0, :, :],
                            acc3d[0, :, :],
                            transpose_B=True,
                            init=True,
                            kL0Size=KL0,
                        )
                        T.gemm_v0(
                            q_a[1, :, :],
                            v_all[1, :, :],
                            acc3d[0, :, :],
                            transpose_B=True,
                            init=False,
                            kL0Size=KL0,
                        )
                        T.gemm_v0(
                            qt_all[0, :, :],
                            tail_l1,
                            acc3d[0, :, :],
                            transpose_B=True,
                            init=False,
                            kL0Size=KL0,
                        )
                        T.copy(
                            acc3d[0, :, :],
                            ws_s[cid, t * HSEG : t * HSEG + V_BLOCK, :],
                        )
                        # ms=1 (heads 64:128)
                        T.gemm_v0(
                            q_b[0, :, :],
                            v_all[0, :, :],
                            acc3d[0, :, :],
                            transpose_B=True,
                            init=True,
                            kL0Size=KL0,
                        )
                        T.gemm_v0(
                            q_b[1, :, :],
                            v_all[1, :, :],
                            acc3d[0, :, :],
                            transpose_B=True,
                            init=False,
                            kL0Size=KL0,
                        )
                        T.gemm_v0(
                            qt_all[1, :, :],
                            tail_l1,
                            acc3d[0, :, :],
                            transpose_B=True,
                            init=False,
                            kL0Size=KL0,
                        )
                        T.copy(
                            acc3d[0, :, :],
                            ws_s[cid, t * HSEG + V_BLOCK : t * HSEG + 2 * V_BLOCK, :],
                        )
                        T.set_cross_flag("FIX", F1A_BASE + t)
                    # ---- pass 2 (a3): PV accumulation, N-seg outer ----
                    # acc3d[ms] = ms-th M-slice's seg-s partial, accumulated
                    # across t (init on t==0; p8-validated 3D-slot + init=
                    # False cross-block form). s is the OUTER loop: L0C's
                    # 128KB holds only 2 x [64,256] fp32 = both M-slices of
                    # ONE N-seg; each s pass re-reads its 256-col ws_kv band
                    # (per task 2*NI*128KB KV + 2*NI*64KB P — cheaper than
                    # the M-outer variant's full-width KV re-reads).
                    # F2 waits live in the s=0 pass ONLY (1:1 with V's
                    # pass-2 sets; the s=1 pass re-reads ws_p/ws_kv in C
                    # program order — both live until V's task k+1 pass 2,
                    # chain-fenced via F1A(task k+1) set only after this
                    # pass). t=0 hoisted: gemm init= must be a literal.
                    n0 = k * NI
                    slot0 = n0 % DEPTH
                    for s in range(N_SEG):
                        if s == 0:
                            T.wait_cross_flag(F2_BASE + 0)
                        T.copy(
                            ws_kv[cid, slot0 * BI : (slot0 + 1) * BI, s * DS : (s + 1) * DS],
                            v_all[s, :, :],
                        )
                        T.copy(ws_p[cid, 0:V_BLOCK, :], p_l1)
                        T.gemm_v0(p_l1, v_all[s, :, :], acc3d[0, :, :], init=True, kL0Size=KL0)
                        T.copy(ws_p[cid, V_BLOCK:HSEG, :], p_l1)
                        T.gemm_v0(p_l1, v_all[s, :, :], acc3d[1, :, :], init=True, kL0Size=KL0)
                        for t in T.serial(NI - 1):
                            n = k * NI + t + 1
                            slot = n % DEPTH
                            if s == 0:
                                T.wait_cross_flag(F2_BASE + t + 1)
                            T.copy(
                                ws_kv[
                                    cid,
                                    slot * BI : (slot + 1) * BI,
                                    s * DS : (s + 1) * DS,
                                ],
                                v_all[s, :, :],
                            )
                            T.copy(
                                ws_p[
                                    cid,
                                    (t + 1) * HSEG : (t + 1) * HSEG + V_BLOCK,
                                    :,
                                ],
                                p_l1,
                            )
                            T.gemm_v0(p_l1, v_all[s, :, :], acc3d[0, :, :], init=False, kL0Size=KL0)
                            T.copy(
                                ws_p[
                                    cid,
                                    (t + 1) * HSEG + V_BLOCK : (t + 1) * HSEG + HSEG,
                                    :,
                                ],
                                p_l1,
                            )
                            T.gemm_v0(p_l1, v_all[s, :, :], acc3d[1, :, :], init=False, kL0Size=KL0)
                        # seg-s band of both M-slices -> ws_o (v1 store form)
                        T.copy(acc3d[0, :, :], ws_o[cid, 0:V_BLOCK, s * DS : (s + 1) * DS])
                        T.copy(acc3d[1, :, :], ws_o[cid, V_BLOCK:HSEG, s * DS : (s + 1) * DS])
                    # both N-seg bands of both M-slices flushed -> release V
                    T.set_cross_flag("FIX", F3_BASE + 0)
                    T.set_cross_flag("FIX", F3_BASE + 1)

            # ================= Vector stream =================
            with T.Scope("V"):
                if ntask > 0:
                    # ---- prologue: gather(0) (LEAD=1 lookahead, rev1 form
                    #      retimed to BI=256: 8 x [16,576] batches/vid) ----
                    s_row0 = cid % seq_len
                    b0 = TokenIndices[s_row0, 0]
                    bos0 = Offsets[b0]
                    T.copy(Indices[s_row0, 0, 0:BI], indices_ub[0, :])
                    T.set_flag("mte2", "v", 0)
                    T.wait_flag("mte2", "v", 0)
                    # vectorized pre-clamp rows (int32 domain, reducesum
                    # form): rows = clamp(idx + bos, 0, S_kv - 1); clamp
                    # protects addressing only, contributions are masked
                    T.tile.add(rows_ub, indices_ub[0, :], bos0)
                    T.tile.clamp(rows_ub, rows_ub, 0, seq_len_kv - 1, BI)
                    T.set_flag("v", "s", 0)
                    T.wait_flag("v", "s", 0)
                    for b in range(NBATCH):
                        T.set_flag("mte3", "mte2", 0)
                        T.wait_flag("mte3", "mte2", 0)
                        for j in range(GKB):
                            kv_row = rows_ub[vid * HALF + b * GKB + j]
                            T.copy(KV[kv_row, 0, :], kv_stage[j, :])
                        T.set_flag("mte2", "mte3", 0)
                        T.wait_flag("mte2", "mte3", 0)
                        T.copy(
                            kv_stage,
                            ws_kv[cid, vid * HALF + b * GKB : vid * HALF + (b + 1) * GKB, :],
                        )
                    T.set_cross_flag("MTE3", F0_BASE + 0)
                    if LEAD == 2 and nblk > 1:
                        # A3a prologue lookahead block 1 (task 0's block 1
                        # for NI >= 2 — same s_row0/bos0 scalars; NI == 1
                        # stays on the LEAD=1 form above)
                        T.set_flag("v", "mte2", 0)
                        T.wait_flag("v", "mte2", 0)
                        T.copy(Indices[s_row0, 0, BI : 2 * BI], indices_ub[1, :])
                        T.set_flag("mte2", "v", 0)
                        T.wait_flag("mte2", "v", 0)
                        T.set_flag("s", "v", 0)
                        T.wait_flag("s", "v", 0)
                        T.tile.add(rows_ub, indices_ub[1, :], bos0)
                        T.tile.clamp(rows_ub, rows_ub, 0, seq_len_kv - 1, BI)
                        T.set_flag("v", "s", 0)
                        T.wait_flag("v", "s", 0)
                        for b in range(NBATCH):
                            T.set_flag("mte3", "mte2", 0)
                            T.wait_flag("mte3", "mte2", 0)
                            for j in range(GKB):
                                kv_row = rows_ub[vid * HALF + b * GKB + j]
                                T.copy(KV[kv_row, 0, :], kv_stage[j, :])
                            T.set_flag("mte2", "mte3", 0)
                            T.wait_flag("mte2", "mte3", 0)
                            T.copy(
                                kv_stage,
                                ws_kv[
                                    cid,
                                    BI + vid * HALF + b * GKB : BI + vid * HALF + (b + 1) * GKB,
                                    :,
                                ],
                            )
                        T.set_cross_flag("MTE3", F0_BASE + 1)

                    for k in T.serial(ntask):
                        pid = k * grid_size + cid
                        s_row = pid % seq_len
                        b_i = TokenIndices[s_row, 0]
                        s_i = TokenIndices[s_row, 1]
                        bos = Offsets[b_i]
                        # lookahead (task k+1) scalars — clamped memory-safe
                        # (reducesum iter8/#18 pattern)
                        pid_n = pid + grid_size
                        s_row_n = T.min(pid_n % seq_len, seq_len - 1)
                        b_n = TokenIndices[s_row_n, 0]
                        bos_n = Offsets[b_n]
                        # a3 state init (m_fin floor avoids -inf - inf NaN)
                        T.tile.fill(m_fin, -(2.0**30))
                        T.tile.fill(sumexp, 0.0)
                        # ---- pass 1 (a3): NI-block m_final reduction ----
                        # (each block's reduction is preceded by the
                        # gather(n+1) lookahead — rev1 LEAD form retimed to
                        # BI=256; F0 ring protocol as built in segment 1)
                        for t in T.serial(NI):
                            n = k * NI + t
                            # ---- gather(n+LEAD): lookahead, 8x[16-row] batches ----
                            if n + LEAD < nblk:
                                gn = n + LEAD
                                nxt = t + LEAD >= NI  # gn belongs to task k+1?
                                nslot = gn % DEPTH
                                islot = gn % IDXSLOTS
                                tg = (t + LEAD) % NI  # gn's block index in its task
                                s_row_g = T.if_then_else(nxt, s_row_n, s_row)
                                # indices_ub rewrite protection: prior V
                                # readers = post1(n-1)'s mask chain (same slot)
                                T.set_flag("v", "mte2", 0)
                                T.wait_flag("v", "mte2", 0)
                                T.copy(
                                    Indices[s_row_g, 0, tg * BI : (tg + 1) * BI],
                                    indices_ub[islot, :],
                                )
                                T.set_flag("mte2", "v", 0)
                                T.wait_flag("mte2", "v", 0)
                                # rows_ub rewrite protection: prior S-pipe
                                # GetValue readers = gather(n)'s row loop
                                T.set_flag("s", "v", 0)
                                T.wait_flag("s", "v", 0)
                                if nxt:
                                    # GM scalars must stay in direct-read form
                                    # (select-wrapped scalars break the Adds
                                    # codegen — reducesum L530 lesson)
                                    T.tile.add(rows_ub, indices_ub[islot, :], bos_n)
                                else:
                                    T.tile.add(rows_ub, indices_ub[islot, :], bos)
                                T.tile.clamp(rows_ub, rows_ub, 0, seq_len_kv - 1, BI)
                                T.set_flag("v", "s", 0)
                                T.wait_flag("v", "s", 0)
                                for b in range(NBATCH):
                                    # kv_stage WAR: drain the previous bulk
                                    # (previous batch here; the previous
                                    # gather's last bulk for b == 0)
                                    T.set_flag("mte3", "mte2", 0)
                                    T.wait_flag("mte3", "mte2", 0)
                                    for j in range(GKB):
                                        kv_row = rows_ub[vid * HALF + b * GKB + j]
                                        T.copy(KV[kv_row, 0, :], kv_stage[j, :])
                                    T.set_flag("mte2", "mte3", 0)
                                    T.wait_flag("mte2", "mte3", 0)
                                    T.copy(
                                        kv_stage,
                                        ws_kv[
                                            cid,
                                            nslot * BI + vid * HALF + b * GKB : nslot * BI + vid * HALF + (b + 1) * GKB,
                                            :,
                                        ],
                                    )
                                T.set_cross_flag("MTE3", F0_BASE + gn % F0IDS)  # F0 note
                            # post1(n): m_final step (a3 pass 1 computes m
                            # ONLY — rev1's online sumexp/alpha rescale is
                            # deliberately NOT done here)
                            T.wait_cross_flag(F1A_BASE + t)
                            # s_ub MTE2 write vs prior V readers (the
                            # previous post's chain; pass2's loads below)
                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)
                            T.copy(
                                ws_s[
                                    cid,
                                    t * HSEG + vid * V_BLOCK : t * HSEG + (vid + 1) * V_BLOCK,
                                    :,
                                ],
                                s_ub,
                            )
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            # mask chain (rev1 BIGC form, group broadcast):
                            # mask = clamp01(min(idx + 1, s_i + 1 - idx));
                            # bias = BIGC * (1 - mask)
                            T.copy(indices_ub[n % IDXSLOTS, :], idx_f)  # int32 -> fp32
                            T.tile.add(t_ub, idx_f, T.float32(1.0))
                            T.tile.mul(m1_ub, idx_f, T.float32(-1.0))
                            T.tile.add(m1_ub, m1_ub, T.float32(s_i + 1))
                            T.tile.min(t_ub, t_ub, m1_ub)
                            T.tile.max(t_ub, t_ub, T.float32(0.0))
                            T.tile.min(t_ub, t_ub, T.float32(1.0))
                            # save this block's mask for pass 2 (pure V->V,
                            # no flags; avoids an Indices GM re-read — both
                            # ring slots are clobbered by lookahead gathers
                            # by the time pass 2 starts)
                            T.copy(t_ub, mask_mem[t, :])
                            T.tile.mul(m1_ub, t_ub, T.float32(-BIGC))
                            T.tile.add(m1_ub, m1_ub, T.float32(BIGC))
                            # s_masked = s * scale * mask - BIGC * (1 - mask)
                            T.tile.mul(s_ub, s_ub, sm_scale)
                            T.tile.broadcast(bcast_h, t_ub, axis=0)
                            for g in range(NGRP):
                                T.tile.mul(
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    bcast_h,
                                )
                            T.tile.broadcast(bcast_h, m1_ub, axis=0)
                            for g in range(NGRP):
                                T.tile.sub(
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    bcast_h,
                                )
                            # m_fin = max(m_fin, rowmax(s_masked));
                            # sumexp_i doubles as the rowmax scratch (its
                            # pass-2 rowsum use overwrites it — no hazard)
                            for g in range(NGRP):
                                T.reduce_max(
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    sumexp_i[g * EG : (g + 1) * EG],
                                    dim=-1,
                                )
                            T.tile.max(m_fin, m_fin, sumexp_i)
                        # ---- pass 2 (a3): P = exp(s_masked - m_fin) ----
                        # CAVEAT decision (chunk B): ws_s RE-READ per block
                        # (the QK-recompute variant stays on file in the
                        # section header; segment 4 owns the matching route).
                        # No cross-flag wait needed: ws_s[t] was published by
                        # F1A+t in pass 1, and C cannot overwrite it for
                        # task k+1 before consuming ALL F2 flags of task k
                        # (chain-fenced via segment 4's F2 waits).
                        for t in T.serial(NI):
                            # s_ub MTE2 write vs prior V readers (the
                            # previous pass2 block's chain / exp)
                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)
                            T.copy(
                                ws_s[
                                    cid,
                                    t * HSEG + vid * V_BLOCK : t * HSEG + (vid + 1) * V_BLOCK,
                                    :,
                                ],
                                s_ub,
                            )
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            # restore this block's mask (saved in pass 1) +
                            # recompute the bias (2 vector ops — cheaper than
                            # a second [NI, BI] save array)
                            T.copy(mask_mem[t, :], t_ub)
                            T.tile.mul(m1_ub, t_ub, T.float32(-BIGC))
                            T.tile.add(m1_ub, m1_ub, T.float32(BIGC))
                            # s_masked = s * scale * mask - BIGC * (1 - mask)
                            T.tile.mul(s_ub, s_ub, sm_scale)
                            T.tile.broadcast(bcast_h, t_ub, axis=0)
                            for g in range(NGRP):
                                T.tile.mul(
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    bcast_h,
                                )
                            T.tile.broadcast(bcast_h, m1_ub, axis=0)
                            for g in range(NGRP):
                                T.tile.sub(
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    bcast_h,
                                )
                            # P = exp(s_masked - m_fin); masked-out entries
                            # are exp(-BIGC - m_fin) -> natural underflow 0
                            for g in range(NGRP):
                                T.tile.broadcast(bcast_h, m_fin[g * EG : (g + 1) * EG], axis=1)
                                T.tile.sub(
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    bcast_h,
                                )
                            T.tile.exp(s_ub, s_ub)
                            for g in range(NGRP):
                                T.reduce_sum(
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    sumexp_i[g * EG : (g + 1) * EG],
                                    dim=-1,
                                )
                            T.tile.add(sumexp, sumexp, sumexp_i)
                            # p store: per-group INTERLEAVED cast + store (a
                            # single cast_g slot cannot batch the two loops —
                            # probe t3i5 probe_a_only: every store would read
                            # the final group's cast)
                            for g in range(NGRP):
                                T.set_flag("mte3", "v", 0)
                                T.wait_flag("mte3", "v", 0)
                                T.copy(s_ub[g * EG : (g + 1) * EG, :], cast_g)
                                T.copy(cast_g, cg32)  # bf16 -> fp32 (exact)
                                T.copy(cg32, cg16)  # fp32 -> fp16 (exact on bf16 values)
                                T.set_flag("v", "mte3", 0)
                                T.wait_flag("v", "mte3", 0)
                                T.copy(
                                    cg16,
                                    ws_p[
                                        cid,
                                        t * HSEG + vid * V_BLOCK + g * EG : t * HSEG + vid * V_BLOCK + (g + 1) * EG,
                                        :,
                                    ],
                                )
                            T.set_cross_flag("MTE3", F2_BASE + t)
                        # ---- epilogue (task tail): ws_o read, normalize,
                        #      Output + Lse (v1 forms; a3 semantics) ----
                        # BOTH F3 ids: (a) this vid's ws_o rows span both
                        # N-seg bands, (b) ws_kv ring safety — the next
                        # task's gathers may only fire past C's COMPLETE
                        # pass 2 (both bands, both M-slices flushed).
                        T.wait_cross_flag(F3_BASE + 0)
                        T.wait_cross_flag(F3_BASE + 1)
                        for s in range(N_SEG):
                            # s_ub MTE2 write vs prior V readers (pass-2
                            # exp/store chain)
                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)
                            T.copy(
                                ws_o[
                                    cid,
                                    vid * V_BLOCK : (vid + 1) * V_BLOCK,
                                    s * DS : (s + 1) * DS,
                                ],
                                o_stage,
                            )
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            T.copy(o_stage, s_ub)  # bf16 -> fp32 (exact)
                            # per-row /sumexp (v1 tail-div form: per-task
                            # 1/8 frequency, not worth a broadcast chain)
                            for h in range(V_BLOCK):
                                T.tile.div(s_ub[h, :], s_ub[h, :], sumexp[h])
                            for g in range(NGRP):
                                T.set_flag("mte3", "v", 0)
                                T.wait_flag("mte3", "v", 0)
                                T.copy(s_ub[g * EG : (g + 1) * EG, :], cast_g)
                                T.set_flag("v", "mte3", 0)
                                T.wait_flag("v", "mte3", 0)
                                T.copy(
                                    cast_g,
                                    Output[
                                        s_row,
                                        vid * V_BLOCK + g * EG : vid * V_BLOCK + (g + 1) * EG,
                                        s * DS : (s + 1) * DS,
                                    ],
                                )
                        # Lse = ln(sumexp) + m_fin (a3 global-max form)
                        T.tile.ln(sumexp, sumexp)
                        T.tile.add(sumexp, sumexp, m_fin)
                        T.set_flag("v", "mte3", 0)
                        T.wait_flag("v", "mte3", 0)
                        T.copy(sumexp, Lse[s_row, vid * V_BLOCK : (vid + 1) * V_BLOCK])
                        # drain the Lse store (MTE3 reads sumexp) BEFORE the
                        # next task's V-pipe state fills: without this pair
                        # fill(sumexp, 0) can overwrite the UB data before the
                        # DMA reads it (v1 keeps its Lse store mid-epilogue
                        # and relies on the trailing Output-store pairs; the
                        # chunk-C probe evidence was last-task-only correct)
                        T.set_flag("mte3", "v", 0)
                        T.wait_flag("mte3", "v", 0)

    return main
