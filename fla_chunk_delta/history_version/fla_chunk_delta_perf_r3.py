"""fla_chunk_delta: KDA gated delta rule chunk state recurrence (fwd + bwd)
on Ascend 910B3.

Semantics (upstream flash-linear-attention chunk_gated_delta_rule_{fwd_h,
bwd_dhu}, fixed length, USE_G + USE_GK, chunk size 64): each (b, h, i_v)
chain maintains an fp32 state H/dH in R^{DK x DV} and recurses serially over
NT = S/64 chunks.

    fwd (t = 0..NT-1):  h[t] = H16; v_new = u - w @ H16      (ungated save)
                        v_new *= exp(g_last - g)            (token gate)
                        H = D * H + k^T @ v_new16
                        D[k] = exp(g_last + gk_last[k])      (folded decay)
    bwd (t = NT-1..0):  dh[t] = dH16; dv_part = k @ dH16
                        dv_part *= exp(g_last - g); dv2 = dv + dv_part
                        q~ = q * exp(g)
                        dH = D * dH + (q~^T @ do) * scale - w^T @ dv2

The per-chunk decay product exp(g_last)*exp(gk_last) is computed as a single
exp(g_last + gk_last) (relative diff ~1 ulp, far inside the output gates).

Developer mode, threads=2 fixed-core task loop + explicit named relay
workspaces (chunk_o structure) for the recurrences, plus a threads=1
Developer PREP kernel that materializes the per-chunk gate/decay tiles once
per launch. bshd inputs are consumed through zero-copy 3D views
[B, S, H*D] (metadata only, no host materialization).

Ascend toolchain adaptations (verified against this repo's sources and the
family precedents; see debug_log for the empirical record):

1. **Gate/decay tiles are precomputed by a threads=1 prep kernel** (the
   Stage-3 restructure; supersedes the Stage-2 in-kernel scalar extraction
   loops, which profiled at 79-84% of the AIV scalar pipe -- per-iteration
   MTE2_S/V_S/S_V flag pairs around every scalar statement, ~33 ns/iter).
   The prep kernel is threads=1 (no vid splitting), where the transpose /
   row_expand primitives are unconstrained:
   * GT[b, c, h, j] = exp(s * (g_last - g[s0+j]))        (token gate)
   * GE[b, c, h, j] = exp(s * g[s0+j])                    (bwd q~ gate)
   * GD[b, c, h, k] = exp(s * (gk_last[k] + g_last))      (state decay)
   with s = ln2 if use_exp2 else 1. Each tile is computed in the DIRECT
   difference form (no exp(a)*exp(b) factorization: the factored form
   overflows fp32 for |g| > ~88 where the direct form is exact). The main
   kernels then read a contiguous 256B/512B span per (chain, chunk) and
   gate via row_expand_{mul,sub} -- zero scalar statements in the hot loop.
   Gate/decay values are bit-comparable to the Stage-2 in-kernel form
   (same fp32 sub/mul/exp sequence).
2. vid-reduction (threads=2) halves dim 0 of every UB buffer and only
   rewrites an allow-listed op set (ascend_vid_reduction.cc IsTileOp:
   add/sub/mul/div/exp/fill/cast/...). T.tile.gather, T.tile.broadcast and
   arith_progression are NOT in the allow list and corrupt silently under
   vid splitting (fla_wy_fast #4) -- which is why the transpose lives in
   the threads=1 prep kernel. The vid pass applies the GM region offset at
   region dim (region_rank - ub_rank): the main-kernel GT/GE/GD reads use
   1D [BS]/[DK] UB targets so the offset lands on the token/channel Range
   dim (the W3/Hout pattern), never on a head index.
   Row gating in the main kernels uses
   T.tile.row_expand_mul_experiment (AscendC brcb + mul_mask), whose shapes
   flow entirely through access_ptr extents and therefore stay consistent
   after halving; fp32 operands need 256B rows ([*, 64] fp32), which holds
   for every gated tile here (DK=128 q~ is gated in two 64-column chunks).
3. gemm_v0 operands are half precision only (fp32 A/B produces garbage on
   this toolchain); every fp32<->fp16/bf16 conversion is a UB-internal
   two-step copy chain (GM<->UB cast-on-copy codegen is broken).
4. All C<->V exchanges go through explicit named workspace_* GM tensors
   (L2-resident, [CORE_EFF, NSLOT, ...]); AUTO_CV_SYNC inserts the
   cross-core flags by buffer-name match (chunk_o pattern). NSLOT=1 is a
   strict lockstep: correct by construction, Stage 3 owns the ring.
5. The serial chunk recurrence uses T.serial + GM relay (plain serial loops
   mis-sync on-chip V->C exchanges -- fla_wy_fast #1); the fixed-core task
   loop stripes chains over CORE_EFF = min(20, NCHAIN) cores with a tail
   guard `if tid < NCHAIN` (both core sides skip together, so the relay
   flag pairs stay balanced).
6. Support domain (wrapper asserts): S % 64 == 0, chunk_size == 64,
   DK in {64, 128} (row_expand row cap 248 + 64-column gate chunks),
   DV % 64 == 0. DK = 192/256 and non-64-multiple D are L1-extension
   territory (DESIGN SS5.4).
"""

import tilelang
import torch
from tilelang import language as T

# 2^g is computed as exp(g * ln2): exp2 has no Ascend backend mapping.
LN2 = 0.6931471805599453

# Ascend 910B3: 20 AI Cube cores (+ 40 AI Vector cores); threads=2 pairs
# each cube core with its 2 vector sub-blocks.
CORE_NUM = 20

# Relay slot ring per core. Stage-3 ring (was NSLOT=1 lockstep): each
# (task, chunk) iteration uses slot = (k_task*NT + i_t) % NSLOT, so chain
# B's V-side relay writes no longer WAR chain A's pending C-side reads
# (the single shared slot serialized every chain's C<->V exchange -- the
# AIC spent 98% of its wall time waiting on relay flags, msprof r2).
# Safety without a manual throttle flag: the V side stalls on its own
# chain's C round-trip every chunk, so the V lead is bounded by the
# in-flight chunks <= chains/core (7 at bench shape) -- within the
# documented 15-set cross-core flag counter capacity, and the ring needs
# NSLOT > 2L+1 = 15 for lead L=7 -> NSLOT=16 (23-38MB of L2-resident
# workspace, well under the 192MB L2).
NSLOT = 16

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # automatic C/V split
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,  # cross-core relay flags
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,  # intra-core pipe sync
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # UB/L1 lifetime planning
}


# ============================================================================
# prep kernel (threads=1): per-chunk gate/decay tiles GT/GE/GD
# ============================================================================
# Computes, per (b, chunk) task, the full [H]-wide tile set in the DIRECT
# difference form (no exp factorization -> no overflow narrowing):
#     GT[h, j] = exp(s * (g_last[h] - g[s0+j, h]))     (token gate, fwd+bwd)
#     GE[h, j] = exp(s * g[s0+j, h])                    (q~ gate, bwd only)
#     GD[h, k] = exp(s * (gk_last[h, k] + g_last[h]))   (state decay)
# The [BS, H] g block is loaded contiguously and transposed once
# (fp32 hardware transpose needs 32B-aligned dims -> both must be multiples
# of 8; BS=64 always is, H is staged padded to HP = ceil(H/8)*8 with the
# dead rows [H, HP) never written back to the H-exact GT/GE/GD tensors).
# row_expand_sub applies the per-row scalar (-s*g_last, shared by the GT
# and GD chains); row_expand needs 8..248 fp32 256B rows -> HP in [8, 248],
# i.e. H <= 248 (wrapper-asserted). threads=1 keeps every primitive
# vid-safe (transpose/broadcast/gather are unconstrained without vid
# splitting).
@tilelang.jit(out_idx=[2, 3, 4], pass_configs=PASS_CONFIGS)
def fla_chunk_delta_prep_kernel(
    B,
    S,
    H,
    DK,
    chunk_size=64,
    use_exp2=False,
    need_ge=False,
):
    """prep: materialize GT/GE/GD gate/decay tiles from (g, gk)."""
    BS = chunk_size
    NT = S // BS
    NPREP = B * NT
    CORE_EFF = min(CORE_NUM, NPREP)
    NTASK = (NPREP + CORE_EFF - 1) // CORE_EFF
    # decay halves: row_expand requires 256B fp32 rows -> [HP, 64] slabs
    NSLABD = DK // 64
    assert NSLABD in (1, 2), "DK in {64, 128} only (L0 domain)"
    # head-dim staging pad: transpose (32B-aligned dims) + row_expand
    # (8..248 rows) both need H rounded up to a multiple of 8
    HP = (H + 7) // 8 * 8
    assert 8 <= HP <= 248, "H <= 248 (prep row_expand/transpose staging cap)"
    # s = ln2 if use_exp2 else 1.0 (gate/decay exponent scale)
    s = LN2 if use_exp2 else 1.0

    @T.prim_func
    def main(
        G: T.Tensor((B, S, H), "float32"),  # type: ignore
        Gk: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        GE: T.Tensor((B, NT, H, BS), "float32") if need_ge else T.Tensor((1,), "float32"),  # type: ignore
        GT: T.Tensor((B, NT, H, BS), "float32"),  # type: ignore
        GD: T.Tensor((B, NT, H, DK), "float32"),  # type: ignore
    ):
        with T.Kernel(CORE_EFF, threads=1, is_npu=True) as (cid):
            # HP-padded staging: rows/cols [H, HP) hold dead garbage (their
            # exp results are computed but never written back -- no
            # cross-row contamination in any of these per-element ops)
            g2d = T.alloc_shared([BS, HP], "float32")  # g block staging
            gx2d = T.alloc_shared([BS, HP], "float32")  # +/-s*g domain
            gt_t = T.alloc_shared([HP, BS], "float32")  # transposed domain
            # (-s * g_last) as an [HP, 1] row vector: both row_expand_sub
            # sites consume it (GT: -s*g - (-s*g_last) = s*(g_last - g);
            # GD: s*gk - (-s*g_last) = s*(gk + g_last))
            glast2d = T.alloc_shared([HP, 1], "float32")
            ge_a = T.alloc_shared([HP, 64], "float32")  # decay slab A
            # slab B is allocated unconditionally (parser env registers
            # allocs outside Python-level `if`); for NSLABD == 1 its
            # statements are const-false dropped and the planner DCEs it
            ge_b = T.alloc_shared([HP, 64], "float32")
            tmp_row = T.alloc_shared([HP, 8], "float32")  # row_expand scratch

            for k_task in T.serial(NTASK):
                tid = cid + k_task * CORE_EFF
                if tid < NPREP:
                    bb = tid // NT
                    c = tid % NT
                    s0 = c * BS
                    n_t = s0 + BS - 1
                    # ---- load the g block (contiguous [BS, H]) + last row
                    T.copy(G[bb, s0 : s0 + BS, 0:H], g2d[0:BS, 0:H])
                    T.copy(G[bb, n_t, 0:H], glast2d[0:H, 0:1])
                    T.tile.mul(glast2d, glast2d, -s)
                    # ---- token gate GT = exp(s*(g_last - g))^T
                    # (-s*g) -> transpose -> row_expand_sub(-s*g_last)
                    T.tile.mul(gx2d, g2d, -s)
                    T.tile.transpose(gt_t, gx2d)
                    T.tile.row_expand_sub_experiment(gt_t, gt_t, glast2d, tmp_row)
                    T.tile.exp(gt_t, gt_t)
                    T.copy(gt_t[0:H, 0:BS], GT[bb, c, 0:H, 0:BS])
                    # ---- q~ gate GE = exp(s*g)^T (bwd only)
                    if need_ge:
                        T.tile.mul(gx2d, g2d, s)
                        T.tile.exp(gx2d, gx2d)
                        T.tile.transpose(gt_t, gx2d)
                        T.copy(gt_t[0:H, 0:BS], GE[bb, c, 0:H, 0:BS])
                    # ---- state decay GD = exp(s*(gk_last + g_last))
                    T.copy(Gk[bb, n_t, 0:H, 0:64], ge_a[0:H, 0:64])
                    T.tile.mul(ge_a, ge_a, s)
                    T.tile.row_expand_sub_experiment(ge_a, ge_a, glast2d, tmp_row)
                    T.tile.exp(ge_a, ge_a)
                    T.copy(ge_a[0:H, 0:64], GD[bb, c, 0:H, 0:64])
                    if NSLABD == 2:
                        T.copy(Gk[bb, n_t, 0:H, 64:DK], ge_b[0:H, 0:64])
                        T.tile.mul(ge_b, ge_b, s)
                        T.tile.row_expand_sub_experiment(ge_b, ge_b, glast2d, tmp_row)
                        T.tile.exp(ge_b, ge_b)
                        T.copy(ge_b[0:H, 0:64], GD[bb, c, 0:H, 64:DK])

    return main


# ============================================================================
# fwd kernel: chunk_gated_delta_rule_fwd_h (H state recurrence)
# ============================================================================
# out_idx/workspace_idx use ABSOLUTE positions over the declared signature
# (K3, W3, U3, GT, GD, H0, Hout, Vnew, Ht, workspace_1..4). Disabled
# constexpr branches keep their signature slot with a [1] dummy tensor so
# the indices stay stable across specializations.
@tilelang.jit(out_idx=[6, 7, 8], workspace_idx=[9, 10, 11, 12], pass_configs=PASS_CONFIGS)
def fla_chunk_delta_fwd_kernel(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size=64,
    BV=64,
    use_exp2=False,
    use_initial_state=False,
    output_final_state=False,
    save_new_value=True,
):
    """fwd: h/v_new(/ht) from (k, w, u, GT, GD[/h0]) per (b, h, i_v) chain."""
    BS = chunk_size
    NT = S // BS
    NV = DV // BV
    NCHAIN = B * H * NV
    CORE_EFF = min(CORE_NUM, NCHAIN)
    NTASK = (NCHAIN + CORE_EFF - 1) // CORE_EFF

    @T.prim_func
    def main(
        K3: T.Tensor((B, S, H * DK), "float16"),  # type: ignore
        W3: T.Tensor((B, S, H * DK), "float16"),  # type: ignore
        U3: T.Tensor((B, S, H * DV), "float16"),  # type: ignore
        # gate/decay tiles from the prep kernel: [B, NT, H, BS]/[B, NT, H, DK]
        # fp32. Read as GT[bb, i_t, bh, 0:BS] -> gate_ub [BS] (1D target:
        # the vid region offset then lands on the token Range dim, never on
        # the head index -- the W3/Hout pattern).
        GT: T.Tensor((B, NT, H, BS), "float32"),  # type: ignore
        GD: T.Tensor((B, NT, H, DK), "float32"),  # type: ignore
        H0: T.Tensor((B, H, DK, DV) if use_initial_state else (1,), "float32"),  # type: ignore
        Hout: T.Tensor((B, NT, H, DK, DV), "float16"),  # type: ignore
        # 3D view [B, S, H*DV]: with a 4D [B,S,H,DV] tensor the vid row offset
        # lands on the H index (chunk_o #8) -- the wrapper views it back.
        Vnew: T.Tensor((B, S, H * DV) if save_new_value else (1,), "float16"),  # type: ignore
        Ht: T.Tensor((B, H, DK, DV) if output_final_state else (1,), "float32"),  # type: ignore
        workspace_1: T.Tensor((CORE_EFF, NSLOT, DK, BV), "float16"),  # type: ignore
        workspace_2: T.Tensor((CORE_EFF, NSLOT, BS, BV), "float32"),  # type: ignore
        workspace_3: T.Tensor((CORE_EFF, NSLOT, BS, BV), "float16"),  # type: ignore
        workspace_4: T.Tensor((CORE_EFF, NSLOT, DK, BV), "float32"),  # type: ignore
    ):
        with T.Kernel(CORE_EFF, threads=2, is_npu=True) as (cid):
            # ---- V-side buffers (compiler-mapped UB; vid-halved by the pass)
            h_ub = T.alloc_shared([DK, BV], "float32")  # state H, fp32 resident
            h16_ub = T.alloc_shared([DK, BV], "float16")  # H cast staging
            bv_ub = T.alloc_shared([BS, BV], "float32")  # b_v relay readback
            kh_ub = T.alloc_shared([DK, BV], "float32")  # kH relay readback
            u16_ub = T.alloc_shared([BS, BV], "float16")
            u_ub = T.alloc_shared([BS, BV], "float32")
            vn_ub = T.alloc_shared([BS, BV], "float32")
            vn16_ub = T.alloc_shared([BS, BV], "float16")
            # 1D gate/decay targets: [BS]/[DK] (row_expand src1 accepts [R]
            # with tmp; 1D keeps the vid region offset on the token/channel
            # Range dim of the GT/GD reads)
            gate_ub = T.alloc_shared([BS], "float32")  # token gate (from GT)
            dsrc_ub = T.alloc_shared([DK], "float32")  # state decay (from GD)
            tmp_gate = T.alloc_shared([BS, 8], "float32")  # row_expand scratch
            tmp_decay = T.alloc_shared([DK, 8], "float32")  # row_expand scratch
            # ---- C-side buffers (L1 gemm operands / L0C accumulators)
            w_l1 = T.alloc_shared([BS, DK], "float16")
            k_l1 = T.alloc_shared([BS, DK], "float16")
            h16_l1 = T.alloc_shared([DK, BV], "float16")
            v16_l1 = T.alloc_shared([BS, BV], "float16")
            bv_frag = T.alloc_fragment([BS, BV], "float32")
            kh_frag = T.alloc_fragment([DK, BV], "float32")

            for k_task in T.serial(NTASK):
                tid = cid + k_task * CORE_EFF
                if tid < NCHAIN:
                    i_v = tid % NV
                    i_nh = tid // NV
                    bb = i_nh // H
                    bh = i_nh % H
                    v0 = i_v * BV
                    if use_initial_state:
                        T.copy(H0[bb, bh, 0:DK, v0 : v0 + BV], h_ub)
                    else:
                        T.tile.fill(h_ub, 0.0)
                    for i_t in T.serial(NT):
                        s0 = i_t * BS
                        # relay ring slot for this (task, chunk) iteration
                        sl = (k_task * NT + i_t) % NSLOT
                        # (a) snapshot h[t] (ungated) + H16 relay to Cube
                        T.copy(h_ub, h16_ub)
                        T.copy(h16_ub, Hout[bb, i_t, bh, 0:DK, v0 : v0 + BV])
                        T.copy(h16_ub, workspace_1[cid, sl, 0:DK, 0:BV])
                        # (b) Cube: b_v = w[t] @ H16
                        T.copy(W3[bb, s0 : s0 + BS, bh * DK : (bh + 1) * DK], w_l1)
                        T.copy(workspace_1[cid, sl, 0:DK, 0:BV], h16_l1)
                        T.gemm_v0(w_l1, h16_l1, bv_frag, init=True)
                        T.copy(bv_frag, workspace_2[cid, sl, 0:BS, 0:BV])
                        # (c) Vector: v_new = u - b_v; save UNGATED v_new
                        T.copy(
                            U3[bb, s0 : s0 + BS, bh * DV + v0 : bh * DV + v0 + BV],
                            u16_ub,
                        )
                        T.copy(u16_ub, u_ub)
                        T.copy(workspace_2[cid, sl, 0:BS, 0:BV], bv_ub)
                        T.tile.sub(vn_ub, u_ub, bv_ub)
                        if save_new_value:
                            T.copy(vn_ub, vn16_ub)
                            T.copy(
                                vn16_ub,
                                Vnew[bb, s0 : s0 + BS, bh * DV + v0 : bh * DV + v0 + BV],
                            )
                        # (d) Vector: token gate exp(g_last - g) on v_new
                        # (contiguous GT span from the prep kernel; no
                        # scalar extraction, no in-kernel exp)
                        T.copy(GT[bb, i_t, bh, 0:BS], gate_ub)
                        T.tile.row_expand_mul_experiment(vn_ub, vn_ub, gate_ub, tmp_gate)
                        # (e) Vector->Cube: v_new16; Cube: kH = k^T @ v_new
                        T.copy(vn_ub, vn16_ub)
                        T.copy(vn16_ub, workspace_3[cid, sl, 0:BS, 0:BV])
                        T.copy(K3[bb, s0 : s0 + BS, bh * DK : (bh + 1) * DK], k_l1)
                        T.copy(workspace_3[cid, sl, 0:BS, 0:BV], v16_l1)
                        T.gemm_v0(k_l1, v16_l1, kh_frag, transpose_A=True, init=True)
                        T.copy(kh_frag, workspace_4[cid, sl, 0:DK, 0:BV])
                        # (f) Vector: decay + state update H = D*H + kH
                        # (contiguous GD span from the prep kernel)
                        T.copy(GD[bb, i_t, bh, 0:DK], dsrc_ub)
                        T.copy(workspace_4[cid, sl, 0:DK, 0:BV], kh_ub)
                        T.tile.row_expand_mul_experiment(h_ub, h_ub, dsrc_ub, tmp_decay)
                        T.tile.add(h_ub, h_ub, kh_ub)
                    if output_final_state:
                        T.copy(h_ub, Ht[bb, bh, 0:DK, v0 : v0 + BV])

    return main


# ============================================================================
# bwd kernel: chunk_gated_delta_rule_bwd_dhu (dH state-gradient recurrence)
# ============================================================================
@tilelang.jit(out_idx=[9, 10, 11], workspace_idx=[12, 13, 14, 15, 16, 17, 18], pass_configs=PASS_CONFIGS)
def fla_chunk_delta_bwd_kernel(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size=64,
    BV=64,
    scale=None,
    use_exp2=False,
    use_final_state_gradient=False,
    use_initial_state=False,
):
    """bwd: dh/dv2(/dh0) from (q, k, w, do, dv, GT, GE, GD[/dht, h0])."""
    if scale is None:
        scale = DK**-0.5
    BS = chunk_size
    NT = S // BS
    NV = DV // BV
    NCHAIN = B * H * NV
    CORE_EFF = min(CORE_NUM, NCHAIN)
    NTASK = (NCHAIN + CORE_EFF - 1) // CORE_EFF
    # q~ gating runs on [BS, 64] fp32 column chunks (row_expand 256B rows)
    NQK = DK // 64
    assert NQK in (1, 2), "DK in {64, 128} only (L0 domain)"

    @T.prim_func
    def main(
        Q3: T.Tensor((B, S, H * DK), "bfloat16"),  # type: ignore
        K3: T.Tensor((B, S, H * DK), "bfloat16"),  # type: ignore
        W3: T.Tensor((B, S, H * DK), "bfloat16"),  # type: ignore
        Do3: T.Tensor((B, S, H * DV), "bfloat16"),  # type: ignore
        Dv3: T.Tensor((B, S, H * DV), "bfloat16"),  # type: ignore
        # gate/decay tiles from the prep kernel (see the fwd note for the
        # 1D-target vid rule)
        GT: T.Tensor((B, NT, H, BS), "float32"),  # type: ignore
        GE: T.Tensor((B, NT, H, BS), "float32"),  # type: ignore
        GD: T.Tensor((B, NT, H, DK), "float32"),  # type: ignore
        Dht: T.Tensor((B, H, DK, DV) if use_final_state_gradient else (1,), "float32"),  # type: ignore
        Dh: T.Tensor((B, NT, H, DK, DV), "bfloat16"),  # type: ignore
        Dh0: T.Tensor((B, H, DK, DV) if use_initial_state else (1,), "float32"),  # type: ignore
        # 3D view [B, S, H*DV] (see the fwd Vnew note; wrapper views it back)
        Dv2: T.Tensor((B, S, H * DV), "bfloat16"),  # type: ignore
        workspace_1: T.Tensor((CORE_EFF, NSLOT, DK, BV), "bfloat16"),  # type: ignore
        workspace_2: T.Tensor((CORE_EFF, NSLOT, BS, BV), "float32"),  # type: ignore
        workspace_3: T.Tensor((CORE_EFF, NSLOT, BS, 64), "bfloat16"),  # type: ignore
        workspace_3b: T.Tensor((CORE_EFF, NSLOT, BS, 64), "bfloat16"),  # type: ignore
        workspace_4: T.Tensor((CORE_EFF, NSLOT, BS, BV), "bfloat16"),  # type: ignore
        workspace_5: T.Tensor((CORE_EFF, NSLOT, DK, BV), "float32"),  # type: ignore
        workspace_6: T.Tensor((CORE_EFF, NSLOT, DK, BV), "float32"),  # type: ignore
    ):
        with T.Kernel(CORE_EFF, threads=2, is_npu=True) as (cid):
            # ---- V-side buffers
            dh_ub = T.alloc_shared([DK, BV], "float32")  # dH state, fp32
            dh16_ub = T.alloc_shared([DK, BV], "bfloat16")
            dvp_ub = T.alloc_shared([BS, BV], "float32")  # dv_part
            dv16_ub = T.alloc_shared([BS, BV], "bfloat16")
            dv_ub = T.alloc_shared([BS, BV], "float32")
            dv2_ub = T.alloc_shared([BS, BV], "float32")
            dv2_16_ub = T.alloc_shared([BS, BV], "bfloat16")
            q_half_a = T.alloc_shared([BS, 64], "bfloat16")
            q_ub_a = T.alloc_shared([BS, 64], "float32")
            qt16_a = T.alloc_shared([BS, 64], "bfloat16")
            # chunk-b buffers are allocated unconditionally (buffer allocs
            # inside a Python-level `if` are not registered by the parser
            # env); for NQK == 1 the b statements are const-false dropped
            # and the planner DCEs the dead buffers
            q_half_b = T.alloc_shared([BS, 64], "bfloat16")
            q_ub_b = T.alloc_shared([BS, 64], "float32")
            qt16_b = T.alloc_shared([BS, 64], "bfloat16")
            # 1D gate/decay targets (see the fwd note)
            gate_ub = T.alloc_shared([BS], "float32")  # token gate (from GT)
            gexp_ub = T.alloc_shared([BS], "float32")  # exp(g) for q~ (from GE)
            dsrc_ub = T.alloc_shared([DK], "float32")  # state decay (from GD)
            qdo_ub = T.alloc_shared([DK, BV], "float32")
            wdv2_ub = T.alloc_shared([DK, BV], "float32")
            tmp_gate = T.alloc_shared([BS, 8], "float32")
            tmp_qa = T.alloc_shared([BS, 8], "float32")
            tmp_qb = T.alloc_shared([BS, 8], "float32")
            tmp_decay = T.alloc_shared([DK, 8], "float32")
            # ---- C-side buffers
            k_l1 = T.alloc_shared([BS, DK], "bfloat16")
            w_l1 = T.alloc_shared([BS, DK], "bfloat16")
            qt16_l1 = T.alloc_shared([BS, DK], "bfloat16")
            do_l1 = T.alloc_shared([BS, BV], "bfloat16")
            dv2_16_l1 = T.alloc_shared([BS, BV], "bfloat16")
            dh16_l1 = T.alloc_shared([DK, BV], "bfloat16")
            dvp_frag = T.alloc_fragment([BS, BV], "float32")
            qdo_frag = T.alloc_fragment([DK, BV], "float32")
            wdv2_frag = T.alloc_fragment([DK, BV], "float32")

            for k_task in T.serial(NTASK):
                tid = cid + k_task * CORE_EFF
                if tid < NCHAIN:
                    i_v = tid % NV
                    i_nh = tid // NV
                    bb = i_nh // H
                    bh = i_nh % H
                    v0 = i_v * BV
                    if use_final_state_gradient:
                        T.copy(Dht[bb, bh, 0:DK, v0 : v0 + BV], dh_ub)
                    else:
                        T.tile.fill(dh_ub, 0.0)
                    for it_f in T.serial(NT):
                        i_t = NT - 1 - it_f
                        s0 = i_t * BS
                        # relay ring slot for this (task, chunk) iteration
                        # (flat index in loop-execution order)
                        sl = (k_task * NT + it_f) % NSLOT
                        # (a) snapshot dh[t] (ungated) + dH16 relay
                        T.copy(dh_ub, dh16_ub)
                        T.copy(dh16_ub, Dh[bb, i_t, bh, 0:DK, v0 : v0 + BV])
                        T.copy(dh16_ub, workspace_1[cid, sl, 0:DK, 0:BV])
                        # (b) Cube: dv_part = k[t] @ dH16
                        T.copy(K3[bb, s0 : s0 + BS, bh * DK : (bh + 1) * DK], k_l1)
                        T.copy(workspace_1[cid, sl, 0:DK, 0:BV], dh16_l1)
                        T.gemm_v0(k_l1, dh16_l1, dvp_frag, init=True)
                        T.copy(dvp_frag, workspace_2[cid, sl, 0:BS, 0:BV])
                        # (c) Vector: gate dv_part; dv2 = dv + dv_part
                        # (contiguous GT span from the prep kernel)
                        T.copy(workspace_2[cid, sl, 0:BS, 0:BV], dvp_ub)
                        T.copy(GT[bb, i_t, bh, 0:BS], gate_ub)
                        T.tile.row_expand_mul_experiment(dvp_ub, dvp_ub, gate_ub, tmp_gate)
                        T.copy(
                            Dv3[bb, s0 : s0 + BS, bh * DV + v0 : bh * DV + v0 + BV],
                            dv16_ub,
                        )
                        T.copy(dv16_ub, dv_ub)
                        T.tile.add(dv2_ub, dv_ub, dvp_ub)
                        T.copy(dv2_ub, dv2_16_ub)
                        T.copy(
                            dv2_16_ub,
                            Dv2[bb, s0 : s0 + BS, bh * DV + v0 : bh * DV + v0 + BV],
                        )
                        T.copy(dv2_16_ub, workspace_4[cid, sl, 0:BS, 0:BV])
                        # (d) Vector: q~ = q * exp(g); relay in DK chunks
                        # (contiguous GE span from the prep kernel)
                        T.copy(GE[bb, i_t, bh, 0:BS], gexp_ub)
                        T.copy(
                            Q3[
                                bb,
                                s0 : s0 + BS,
                                bh * DK : bh * DK + 64,
                            ],
                            q_half_a,
                        )
                        T.copy(q_half_a, q_ub_a)
                        if NQK == 2:
                            T.copy(
                                Q3[
                                    bb,
                                    s0 : s0 + BS,
                                    bh * DK + 64 : bh * DK + 128,
                                ],
                                q_half_b,
                            )
                            T.copy(q_half_b, q_ub_b)
                        T.tile.row_expand_mul_experiment(q_ub_a, q_ub_a, gexp_ub, tmp_qa)
                        T.copy(q_ub_a, qt16_a)
                        T.copy(qt16_a, workspace_3[cid, sl, 0:BS, 0:64])
                        if NQK == 2:  # (ws3b keeps a 1:1 site pair for NQK==1)
                            T.tile.row_expand_mul_experiment(q_ub_b, q_ub_b, gexp_ub, tmp_qb)
                            T.copy(q_ub_b, qt16_b)
                            T.copy(qt16_b, workspace_3b[cid, sl, 0:BS, 0:64])
                        # (e) Cube: qdo = q~^T @ do; wdv2 = w^T @ dv2
                        T.copy(
                            Do3[bb, s0 : s0 + BS, bh * DV + v0 : bh * DV + v0 + BV],
                            do_l1,
                        )
                        T.copy(workspace_3[cid, sl, 0:BS, 0:64], qt16_l1[0:BS, 0:64])
                        if NQK == 2:
                            T.copy(workspace_3b[cid, sl, 0:BS, 0:64], qt16_l1[0:BS, 64:DK])
                        T.gemm_v0(qt16_l1, do_l1, qdo_frag, transpose_A=True, init=True)
                        T.copy(qdo_frag, workspace_5[cid, sl, 0:DK, 0:BV])
                        T.copy(W3[bb, s0 : s0 + BS, bh * DK : (bh + 1) * DK], w_l1)
                        T.copy(workspace_4[cid, sl, 0:BS, 0:BV], dv2_16_l1)
                        T.gemm_v0(w_l1, dv2_16_l1, wdv2_frag, transpose_A=True, init=True)
                        T.copy(wdv2_frag, workspace_6[cid, sl, 0:DK, 0:BV])
                        # (f) Vector: dH = D*dH + qdo*scale - wdv2
                        # (contiguous GD span from the prep kernel)
                        T.copy(GD[bb, i_t, bh, 0:DK], dsrc_ub)
                        T.copy(workspace_5[cid, sl, 0:DK, 0:BV], qdo_ub)
                        T.copy(workspace_6[cid, sl, 0:DK, 0:BV], wdv2_ub)
                        T.tile.mul(qdo_ub, qdo_ub, scale)
                        T.tile.row_expand_mul_experiment(dh_ub, dh_ub, dsrc_ub, tmp_decay)
                        T.tile.add(dh_ub, dh_ub, qdo_ub)
                        T.tile.sub(dh_ub, dh_ub, wdv2_ub)
                    if use_initial_state:
                        T.copy(dh_ub, Dh0[bb, bh, 0:DK, v0 : v0 + BV])

    return main


# ============================================================================
# Host wrappers (upstream naming: chunk_gated_delta_rule_{fwd_h, bwd_dhu} ->
# fla_chunk_delta_{fwd, bwd}; signature aligned with the proto schema).
# Only zero-copy metadata views + kernel calls here (the JIT adapter owns
# output/workspace allocation). Each wrapper launches the threads=1 prep
# kernel (GT/GE/GD tiles) followed by the recurrence kernel; both launches
# are inside the do_bench timed region (bench accounting stays honest).
# ============================================================================
def _check_domain(B, S, H, DK, DV, chunk_size):
    assert chunk_size == 64, "chunk_size fixed at 64 (block_S semantic)"
    assert S % chunk_size == 0, "S must be divisible by chunk_size"
    assert DK in (64, 128), "DK in {64, 128} (L0 domain; 192/256 = L1 extension)"
    assert DV % 64 == 0, "DV must be a multiple of 64 (BV=64)"
    assert DK <= 256, "upstream assert K <= 256"
    # prep staging cap: transpose (32B-aligned dims) + row_expand (8..248
    # fp32 256B rows) round the head dim up to a multiple of 8
    assert H <= 248, "H <= 248 (prep gate/decay staging row cap)"


def _dummy(device, dtype):
    return torch.empty(1, dtype=dtype, device=device)


def fla_chunk_delta_fwd(
    k,
    w,
    u,
    g,
    gk,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    save_new_value=True,
    use_exp2=False,
):
    """fwd host API. k/w [B,S,H,DK] fp16, u [B,S,H,DV] fp16, g [B,S,H] fp32,
    gk [B,S,H,DK] fp32 (chunk-cumulative log decay domain) ->
    (h [B,NT,H,DK,DV] fp16, v_new fp16|None, ht fp32|None)."""
    B, S, H, DK = k.shape
    DV = u.shape[-1]
    _check_domain(B, S, H, DK, DV, chunk_size)
    device = k.device
    prep = fla_chunk_delta_prep_kernel(B, S, H, DK, chunk_size, use_exp2, False)
    _ge, gt, gd = prep(g, gk)
    kernel = fla_chunk_delta_fwd_kernel(
        B,
        S,
        H,
        DK,
        DV,
        chunk_size,
        64,
        use_exp2,
        initial_state is not None,
        output_final_state,
        save_new_value,
    )
    h, v_new, ht = kernel(
        k.view(B, S, H * DK),
        w.view(B, S, H * DK),
        u.view(B, S, H * DV),
        gt,
        gd,
        initial_state if initial_state is not None else _dummy(device, torch.float32),
    )
    # no internal synchronize: upstream wrapper parity (bench harness owns it)
    if save_new_value:
        v_new = v_new.view(B, S, H, DV)
    return h, (v_new if save_new_value else None), (ht if output_final_state else None)


def fla_chunk_delta_bwd(
    q,
    k,
    w,
    do,
    dv,
    g,
    gk,
    h0=None,
    dht=None,
    scale=None,
    chunk_size=64,
    use_exp2=False,
):
    """bwd host API. q/k/w [B,S,H,DK] bf16, do/dv [B,S,H,DV] bf16, g/gk fp32
    -> (dh [B,NT,H,DK,DV] bf16, dh0 fp32|None, dv2 bf16). dh0 only when the
    fwd was called with an initial state (h0 given), mirroring upstream."""
    B, S, H, DK = q.shape
    DV = do.shape[-1]
    _check_domain(B, S, H, DK, DV, chunk_size)
    device = q.device
    prep = fla_chunk_delta_prep_kernel(B, S, H, DK, chunk_size, use_exp2, True)
    ge, gt, gd = prep(g, gk)
    kernel = fla_chunk_delta_bwd_kernel(
        B,
        S,
        H,
        DK,
        DV,
        chunk_size,
        64,
        scale,
        use_exp2,
        dht is not None,
        h0 is not None,
    )
    dh, dh0, dv2 = kernel(
        q.view(B, S, H * DK),
        k.view(B, S, H * DK),
        w.view(B, S, H * DK),
        do.view(B, S, H * DV),
        dv.view(B, S, H * DV),
        gt,
        ge,
        gd,
        dht if dht is not None else _dummy(device, torch.float32),
    )
    dv2 = dv2.view(B, S, H, DV)
    return dh, (dh0 if h0 is not None else None), dv2


__all__ = [
    "fla_chunk_delta_fwd",
    "fla_chunk_delta_bwd",
    "fla_chunk_delta_fwd_kernel",
    "fla_chunk_delta_bwd_kernel",
    "fla_chunk_delta_prep_kernel",
    "PASS_CONFIGS",
]
