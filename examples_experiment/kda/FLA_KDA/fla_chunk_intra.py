"""fla_chunk_intra: KDA intra-chunk fwd (fused inter + solve_tril) + bwd on
Ascend 910B3 (Developer mode, threads=1).

Semantics follow the upstream tilelang KDA intra-chunk kernels
(examples/kda/FLA_KDA/fla_chunk_intra.py, Triton/FLA reference).

fwd (chunk_kda_fwd_inter_solve_fused), per (chunk, bh):
  * off-diagonal decayed GEMM blocks -> Aqk (x scale) and the sign-folded
    Akk-beta' (Akkbp): with gn = g[s2*16, :] the anchors cancel inside the
    products, so Aqk = scale * (q*2^g) @ (k*2^-g)^T and Akkbp carries the
    -beta row scale
  * 4 diagonal forward substitutions Ai_ss = (I + strict_tril(A_s))^{-1}
    (pure-V row recursion)
  * merged block inverse Ai = (I + G)(I + G^2) D^{-1}, G = AiDiag @ Akkbp
  * outputs bf16; Aqk diagonal + block-upper-tri are written as exact 0

bwd (chunk_kda_bwd_intra): the per-column sequential loops of the upstream
(B)/(E) steps are re-expressed as masked GEMMs whose masks depend only on
(i, j), never on iteration state.  Two JIT kernels: k_stage (pure Vector)
builds every GEMM operand tile into GM scratch; k_gemm (one Pipelined C/V
body over the DK halves) runs the gemms and the post-beta tail.  Host post-
processing (db2.sum(0)+db, reverse chunk-local cumsum of dg2) stays with the
caller, matching the upstream kernel/test boundary.

Implementation notes (Ascend toolchain):
  * The C/V combine allows only one straight-line Cube segment (or one
    Pipelined body) per kernel, so the merged-inverse chain is split into
    k_inter -> k_subst + k_g -> k_g2 -> k_y -> k_ai; intermediate results
    cross kernels through pre-split bf16 hi/lo workspace pairs that the
    consumer C sections read directly GM->L1 (no relays).
  * fp32 gemm_v0 operands are broken on this toolchain; every GEMM runs as
    hi/lo split operands accumulated in L0C (fwd bf16 3-gemm, bwd fp16
    3-gemm -- |x| > 65504 loses precision).
  * exp2(x) is computed as exp(x * ln2); the @T.prim_func body is TVM-AST
    parsed (no Python lists / helper-function calls inside kernels).
  * Cross-core ffts flags are limited to 16 slots (CANN GetffstMsg masks
    the id with 0xf, 11..14 system-reserved); the fwd drain uses manual
    alternating FIX flags and the bwd kernel boundary provides the V->C
    sync, keeping the whole operator inside the budget.
  * [B,S,H] beta loads use a flat-view span + arith_progression +
    reinterpretcast + tile.gather chain (sub-32B strided DMA is unreliable).
  * The db2 [64,1] scatter round-trips through 32B slots that only self-
    compensate for H >= 2; H == 1 writes db2 through a flat [B, S*H] view.
"""

import tilelang
import torch
from tilelang import language as T

# 2^g is computed as exp(g * ln2): exp2 has no Ascend backend mapping.
LN2 = 0.6931471805599453

# fixed-core grid ceiling (910B3 AI Cube cores) / relay slot ring depth
CORE_NUM = 20
NSLOT = 4

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _pick_bk(dim: int) -> int:
    """Block size forcing EXACTLY 2 Pipelined iterations (toolchain: extent 1
    crashes codegen, extent >= 3 corrupts the double-buffer wrap)."""
    blk = dim // 2
    if dim % 2 == 0 and blk in (32, 48, 64):
        return blk
    raise ValueError(
        f"DK={dim} unsupported: the Ascend Pipelined body requires exactly 2"
        " iterations, so DK must be one of {{64, 96, 128}}"
        " (documented limitation)"
    )


def _grid(B, S, H, chunk_size):
    nt = S // chunk_size
    return nt * B * H


# ============================================================================
# fwd kernel 1: off-diagonal Aqk / Akk-beta' GEMMs (BK = DK).
# anchor-free super-GEMM: the per-s2 anchors cancel inside every product
# (qg_i * rhs_j = q_i k_j 2^(g_i-g_j)), so three full-width hi/lo operand
# pairs feed 6 gemm_v0; Aqk gets the block-lower triangle, Akkbp the -beta
# row scale.  FIXED-CORE striped task loop (task = cid + t*CORE_EFF,
# past-GRID iterations idempotently recompute cid) with per-core relay
# slots and a MANUAL alternating FIX flag; the drain runs 1-late so
# C(j+1) overlaps produce(j)/drain(j).
# ============================================================================
@tilelang.jit(
    out_idx=[4],
    workspace_idx=[7, 8, 9, 10, 11, 12, 13, 14],
    pass_configs=PASS_CONFIGS,
)
def fla_chunk_intra_fwd_k_inter(B, S, H, DK, chunk_size=64, sub_chunk_size=16, scale=1.0):
    BS = chunk_size
    BC = sub_chunk_size
    NT = S // BS
    GRID = NT * B * H
    BK = DK
    # fixed-core striped decomposition
    CORE_EFF = min(CORE_NUM, GRID)
    BASE = GRID // CORE_EFF
    L = BASE + 2  # produce t + drain t-1 for t in [0, BASE], + drain tail
    WS_A = (CORE_EFF, NSLOT, BS - BC, BK)
    WS_B = (CORE_EFF, NSLOT, BS, BK)
    WS_R = (CORE_EFF, NSLOT, BS - BC, BS)

    @T.prim_func
    def main(
        Q: T.Tensor((B, S, H, DK), "bfloat16"),  # type: ignore
        K: T.Tensor((B, S, H, DK), "bfloat16"),  # type: ignore
        GK: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        BetaFlat: T.Tensor((B, S * H), "bfloat16"),  # type: ignore
        Aqk: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        AkkbpHi: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        AkkbpLo: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        workspace_1: T.Tensor(WS_A, "bfloat16"),  # type: ignore  # qt_hi
        workspace_2: T.Tensor(WS_A, "bfloat16"),  # type: ignore  # qt_lo
        workspace_3: T.Tensor(WS_A, "bfloat16"),  # type: ignore  # kta_hi
        workspace_4: T.Tensor(WS_A, "bfloat16"),  # type: ignore  # kta_lo
        workspace_5: T.Tensor(WS_B, "bfloat16"),  # type: ignore  # kt_hi
        workspace_6: T.Tensor(WS_B, "bfloat16"),  # type: ignore  # kt_lo
        aqk_relay: T.Tensor(WS_R, "float32"),  # type: ignore
        akk_relay: T.Tensor(WS_R, "float32"),  # type: ignore
    ):
        with T.Kernel(CORE_EFF, threads=1, is_npu=True) as (cid):
            # ---- fragments (L0C; per-task, reused every iteration) ----
            frag_aqk = T.alloc_fragment([BS - BC, BS], "float32")
            frag_akk = T.alloc_fragment([BS - BC, BS], "float32")
            # ---- L1 operands (per-task, reused) ----
            qt_hi_l1 = T.alloc_shared([BS - BC, BK], "bfloat16")
            qt_lo_l1 = T.alloc_shared([BS - BC, BK], "bfloat16")
            kta_hi_l1 = T.alloc_shared([BS - BC, BK], "bfloat16")
            kta_lo_l1 = T.alloc_shared([BS - BC, BK], "bfloat16")
            kt_hi_l1 = T.alloc_shared([BS, BK], "bfloat16")
            kt_lo_l1 = T.alloc_shared([BS, BK], "bfloat16")
            # ---- produce-side f32 domain + split temps ----
            k_half = T.alloc_shared([BS, BK], "bfloat16")
            k_f = T.alloc_shared([BS, BK], "float32")
            g_f = T.alloc_shared([BS, BK], "float32")
            q_half = T.alloc_shared([BS - BC, BK], "bfloat16")
            q_f = T.alloc_shared([BS - BC, BK], "float32")
            beta_tile = T.alloc_shared([(BS - BC) * H], "bfloat16")
            off_i32 = T.alloc_shared([BS - BC], "int32")
            off_u32 = T.alloc_shared([BS - BC], "uint32")
            bt_bf = T.alloc_shared([BS - BC], "bfloat16")
            e_f = T.alloc_shared([BS - BC, BK], "float32")
            a_hi_f32 = T.alloc_shared([BS - BC, BK], "float32")
            a_lo_f32 = T.alloc_shared([BS - BC, BK], "float32")
            hi_f32 = T.alloc_shared([BS, BK], "float32")
            lo_f32 = T.alloc_shared([BS, BK], "float32")
            # ---- relay-source staging (dedicated UB source per operand) ----
            qt_hi_ub = T.alloc_shared([BS - BC, BK], "bfloat16")
            qt_lo_ub = T.alloc_shared([BS - BC, BK], "bfloat16")
            kta_hi_ub = T.alloc_shared([BS - BC, BK], "bfloat16")
            kta_lo_ub = T.alloc_shared([BS - BC, BK], "bfloat16")
            kt_hi_ub = T.alloc_shared([BS, BK], "bfloat16")
            kt_lo_ub = T.alloc_shared([BS, BK], "bfloat16")
            # ---- bet ring (the only loop-carried V state; 1-late drain) ----
            bet_ring = T.alloc_shared((NSLOT, BS - BC), "float32")
            bet_v = T.alloc_shared([BS - BC], "float32")
            # ---- drain-side (epilogue) buffers ----
            aqk_f32 = T.alloc_shared([BS - BC, BS], "float32")
            aqk_bf = T.alloc_shared([BS - BC, BS], "bfloat16")
            aqk_st = T.alloc_shared([BS, BS], "bfloat16")
            akk_f32 = T.alloc_shared([BS - BC, BS], "float32")
            bcol2d = T.alloc_shared([BS - BC, BS], "float32")
            akk_hi = T.alloc_shared([BS - BC, BS], "bfloat16")
            akk_lo = T.alloc_shared([BS - BC, BS], "bfloat16")
            akk_hf = T.alloc_shared([BS - BC, BS], "float32")
            akk_lf = T.alloc_shared([BS - BC, BS], "float32")
            akk_hs = T.alloc_shared([BS - BC, BS], "bfloat16")
            akk_ls = T.alloc_shared([BS - BC, BS], "bfloat16")
            z16 = T.alloc_shared([BC, BS], "bfloat16")

            # ---- head-set primes the j=0 1-late drain wait ----
            T.tile.fill(z16, 0.0)
            T.set_cross_flag("FIX", 41)

            # ============ single uniform produce + C + drain loop ============
            for j in T.serial(L):
                tid_p = cid + j * CORE_EFF
                pt = T.if_then_else(tid_p < GRID, tid_p, cid)  # redo cid past GRID
                tid_d = cid + (j - 1) * CORE_EFF
                dt = T.if_then_else(tid_d < GRID, T.if_then_else(tid_d >= 0, tid_d, cid), cid)
                slot = j % NSLOT
                dslot = (j + NSLOT - 1) % NSLOT

                # ================= produce task pt (slot) =================
                bs = pt % NT
                bbh = pt // NT
                bb = bbh // H
                bh = bbh % H
                s0 = bs * BS
                T.copy(K[bb, s0 : s0 + BS, bh, 0:BK], k_half)
                T.copy(k_half, k_f)
                T.copy(GK[bb, s0 : s0 + BS, bh, 0:BK], g_f)
                T.copy(Q[bb, s0 + BC : s0 + BS, bh, 0:BK], q_half)
                T.copy(q_half, q_f)
                # beta gather chain (bf16 variant; rows 16..64)
                T.copy(BetaFlat[bb, (s0 + BC) * H : (s0 + BS) * H], beta_tile)
                T.tile.arith_progression(off_i32, bh * 2, H * 2)
                T.reinterpretcast(off_u32, off_i32, "uint32_t")
                T.tile.gather(bt_bf, beta_tile, off_u32, 0)
                T.copy(bt_bf, bet_ring[slot, 0 : BS - BC])
                # Q~ = q * 2^g in q_f; e_f then reused IN PLACE as ktp = k*2^g
                T.tile.mul(e_f, g_f[BC:BS, 0:BK], LN2)
                T.tile.exp(e_f, e_f)
                T.tile.mul(q_f, q_f, e_f)
                T.tile.mul(e_f, e_f, k_f[BC:BS, 0:BK])
                # e_minus in place on g_f, then K~ = k * 2^-g in k_f
                T.tile.mul(g_f, g_f, -LN2)
                T.tile.exp(g_f, g_f)
                T.tile.mul(k_f, k_f, g_f)
                # splits -> relay slots (48-row A sides, 64-row B side)
                T.copy(q_f, qt_hi_ub)
                T.copy(qt_hi_ub, a_hi_f32)
                T.tile.sub(a_lo_f32, q_f, a_hi_f32)
                T.copy(a_lo_f32, qt_lo_ub)
                T.copy(qt_hi_ub, workspace_1[cid, slot, 0 : BS - BC, 0:BK])
                T.copy(qt_lo_ub, workspace_2[cid, slot, 0 : BS - BC, 0:BK])
                T.copy(e_f, kta_hi_ub)
                T.copy(kta_hi_ub, a_hi_f32)
                T.tile.sub(a_lo_f32, e_f, a_hi_f32)
                T.copy(a_lo_f32, kta_lo_ub)
                T.copy(kta_hi_ub, workspace_3[cid, slot, 0 : BS - BC, 0:BK])
                T.copy(kta_lo_ub, workspace_4[cid, slot, 0 : BS - BC, 0:BK])
                T.copy(k_f, kt_hi_ub)
                T.copy(kt_hi_ub, hi_f32)
                T.tile.sub(lo_f32, k_f, hi_f32)
                T.copy(lo_f32, kt_lo_ub)
                T.copy(kt_hi_ub, workspace_5[cid, slot, 0:BS, 0:BK])
                T.copy(kt_lo_ub, workspace_6[cid, slot, 0:BS, 0:BK])

                # ================= C: consume slot, 6 gemm_v0 =================
                T.copy(workspace_1[cid, slot, 0 : BS - BC, 0:BK], qt_hi_l1)
                T.copy(workspace_2[cid, slot, 0 : BS - BC, 0:BK], qt_lo_l1)
                T.copy(workspace_3[cid, slot, 0 : BS - BC, 0:BK], kta_hi_l1)
                T.copy(workspace_4[cid, slot, 0 : BS - BC, 0:BK], kta_lo_l1)
                T.copy(workspace_5[cid, slot, 0:BS, 0:BK], kt_hi_l1)
                T.copy(workspace_6[cid, slot, 0:BS, 0:BK], kt_lo_l1)
                T.gemm_v0(qt_hi_l1, kt_hi_l1, frag_aqk, transpose_B=True, init=True)
                T.gemm_v0(qt_lo_l1, kt_hi_l1, frag_aqk, transpose_B=True, init=False)
                T.gemm_v0(qt_hi_l1, kt_lo_l1, frag_aqk, transpose_B=True, init=False)
                T.gemm_v0(kta_hi_l1, kt_hi_l1, frag_akk, transpose_B=True, init=True)
                T.gemm_v0(kta_lo_l1, kt_hi_l1, frag_akk, transpose_B=True, init=False)
                T.gemm_v0(kta_hi_l1, kt_lo_l1, frag_akk, transpose_B=True, init=False)
                # result relays + MANUAL alternating flag (a single auto
                # FIFO ID would lock-step the two cores)
                T.copy(frag_aqk, aqk_relay[cid, slot, 0 : BS - BC, 0:BS])
                T.copy(frag_akk, akk_relay[cid, slot, 0 : BS - BC, 0:BS])
                T.set_cross_flag("FIX", 40 + j % 2)

                # ================= drain task dt (slot dslot) =================
                bs = dt % NT
                bbh = dt // NT
                bb = bbh // H
                bh = bbh % H
                s0 = bs * BS
                # 1-late drain: waits the flag raised at iteration j-1
                T.wait_cross_flag(40 + (j + 1) % 2, "V")
                T.copy(aqk_relay[cid, dslot, 0 : BS - BC, 0:BS], aqk_f32)
                T.copy(akk_relay[cid, dslot, 0 : BS - BC, 0:BS], akk_f32)
                # Aqk: scale, cast, zero-filled staging write
                T.tile.mul(aqk_f32, aqk_f32, scale)
                T.copy(aqk_f32, aqk_bf)
                T.tile.fill(aqk_st, 0.0)
                T.copy(aqk_bf[0:BC, 0:BC], aqk_st[BC : 2 * BC, 0:BC])
                T.copy(aqk_bf[BC : 2 * BC, 0:BC], aqk_st[2 * BC : 3 * BC, 0:BC])
                T.copy(aqk_bf[BC : 2 * BC, BC : 2 * BC], aqk_st[2 * BC : 3 * BC, BC : 2 * BC])
                T.copy(aqk_bf[2 * BC : 3 * BC, 0:BC], aqk_st[3 * BC : 4 * BC, 0:BC])
                T.copy(aqk_bf[2 * BC : 3 * BC, BC : 2 * BC], aqk_st[3 * BC : 4 * BC, BC : 2 * BC])
                T.copy(aqk_bf[2 * BC : 3 * BC, 2 * BC : 3 * BC], aqk_st[3 * BC : 4 * BC, 2 * BC : 3 * BC])
                T.copy(aqk_st, Aqk[bb, s0 : s0 + BS, bh, 0:BS])
                # Akkbp = -(beta row) * frag: fp32 row scale from the bet
                # ring, then the hi/lo split + dense staging writes
                T.copy(bet_ring[dslot, 0 : BS - BC], bet_v)
                T.tile.broadcast(bcol2d, bet_v, axis=1)
                T.tile.mul(akk_f32, akk_f32, bcol2d)
                T.tile.mul(akk_f32, akk_f32, -1.0)
                T.copy(akk_f32, akk_hi)
                T.copy(akk_hi, akk_hf)
                T.tile.sub(akk_lf, akk_f32, akk_hf)
                T.copy(akk_lf, akk_lo)
                T.tile.fill(akk_hs, 0.0)
                T.tile.fill(akk_ls, 0.0)
                T.copy(akk_hi[0:BC, 0:BC], akk_hs[0:BC, 0:BC])
                T.copy(akk_hi[BC : 2 * BC, 0 : 2 * BC], akk_hs[BC : 2 * BC, 0 : 2 * BC])
                T.copy(akk_hi[2 * BC : 3 * BC, 0 : 3 * BC], akk_hs[2 * BC : 3 * BC, 0 : 3 * BC])
                T.copy(akk_lo[0:BC, 0:BC], akk_ls[0:BC, 0:BC])
                T.copy(akk_lo[BC : 2 * BC, 0 : 2 * BC], akk_ls[BC : 2 * BC, 0 : 2 * BC])
                T.copy(akk_lo[2 * BC : 3 * BC, 0 : 3 * BC], akk_ls[2 * BC : 3 * BC, 0 : 3 * BC])
                # deterministic zero rows 0..16 (0 x NaN = NaN poison guard)
                T.copy(z16, AkkbpHi[bb, s0 : s0 + BC, bh, 0:BS])
                T.copy(z16, AkkbpLo[bb, s0 : s0 + BC, bh, 0:BS])
                T.copy(akk_hs, AkkbpHi[bb, s0 + BC : s0 + BS, bh, 0:BS])
                T.copy(akk_ls, AkkbpLo[bb, s0 + BC : s0 + BS, bh, 0:BS])

    return main


# ============================================================================
# fwd kernel 2a: the 4 diagonal forward substitutions -> AidHi/Lo ws pair
# (pure Vector; the consumer GEMM kernel reads the pair directly GM->L1).
# ============================================================================
@tilelang.jit(pass_configs=PASS_CONFIGS)
def fla_chunk_intra_fwd_k_subst(B, S, H, DK, chunk_size=64, sub_chunk_size=16):
    BS = chunk_size
    BC = sub_chunk_size
    NT = S // BS
    GRID = NT * B * H

    @T.prim_func
    def main(
        Akk_diag: T.Tensor((B, S, H, BC), "float32"),  # type: ignore
        Mneg: T.Tensor((BC, BC), "float32"),  # type: ignore
        Meye: T.Tensor((BC, BC), "float32"),  # type: ignore
        AidHi: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        AidLo: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % NT
            bbh = cid // NT
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            ad_f = T.alloc_shared([BS, BC], "float32")
            mneg = T.alloc_shared([BC, BC], "float32")
            meye = T.alloc_shared([BC, BC], "float32")
            ad_s = T.alloc_shared([BC, BC], "float32")
            a_row = T.alloc_shared([BC], "float32")
            a_row2d = T.alloc_shared([BC, BC], "float32")
            tmp = T.alloc_shared([BC, BC], "float32")
            red = T.alloc_shared([BC], "float32")
            a_new = T.alloc_shared([BC], "float32")
            aid_st = T.alloc_shared([BS, BS], "float32")
            aid_hi = T.alloc_shared([BS, BS], "bfloat16")
            aid_lo = T.alloc_shared([BS, BS], "bfloat16")
            hi_f32 = T.alloc_shared([BS, BS], "float32")
            lo_f32 = T.alloc_shared([BS, BS], "float32")
            ai_s0 = T.alloc_shared([BC, BC], "float32")
            ai_s1 = T.alloc_shared([BC, BC], "float32")
            ai_s2 = T.alloc_shared([BC, BC], "float32")
            ai_s3 = T.alloc_shared([BC, BC], "float32")

            # ==== 4 forward substitutions (row-to-row RAW, must serialize) ====
            T.copy(Akk_diag[bb, s0 : s0 + BS, bh, 0:BC], ad_f)
            T.copy(Mneg, mneg)
            T.copy(Meye, meye)
            # s = 0: rows 0..15 of ad_f; N = A * Mneg; row recursion 2..15
            T.copy(ad_f[0:BC, 0:BC], ad_s)
            T.tile.mul(ai_s0, ad_s, mneg)
            for i in T.serial(2, BC):
                T.copy(ai_s0[i, 0:BC], a_row)
                T.tile.broadcast(a_row2d, a_row, axis=1)
                T.tile.mul(tmp, a_row2d, ai_s0)
                T.reduce_sum(tmp, red, dim=0, clear=True)
                T.tile.add(a_new, a_row, red)
                T.copy(a_new, ai_s0[i, 0:BC])
            # s = 1: rows 16..31
            T.copy(ad_f[BC : 2 * BC, 0:BC], ad_s)
            T.tile.mul(ai_s1, ad_s, mneg)
            for i in T.serial(2, BC):
                T.copy(ai_s1[i, 0:BC], a_row)
                T.tile.broadcast(a_row2d, a_row, axis=1)
                T.tile.mul(tmp, a_row2d, ai_s1)
                T.reduce_sum(tmp, red, dim=0, clear=True)
                T.tile.add(a_new, a_row, red)
                T.copy(a_new, ai_s1[i, 0:BC])
            # s = 2: rows 32..47
            T.copy(ad_f[2 * BC : 3 * BC, 0:BC], ad_s)
            T.tile.mul(ai_s2, ad_s, mneg)
            for i in T.serial(2, BC):
                T.copy(ai_s2[i, 0:BC], a_row)
                T.tile.broadcast(a_row2d, a_row, axis=1)
                T.tile.mul(tmp, a_row2d, ai_s2)
                T.reduce_sum(tmp, red, dim=0, clear=True)
                T.tile.add(a_new, a_row, red)
                T.copy(a_new, ai_s2[i, 0:BC])
            # s = 3: rows 48..63
            T.copy(ad_f[3 * BC : 4 * BC, 0:BC], ad_s)
            T.tile.mul(ai_s3, ad_s, mneg)
            for i in T.serial(2, BC):
                T.copy(ai_s3[i, 0:BC], a_row)
                T.tile.broadcast(a_row2d, a_row, axis=1)
                T.tile.mul(tmp, a_row2d, ai_s3)
                T.reduce_sum(tmp, red, dim=0, clear=True)
                T.tile.add(a_new, a_row, red)
                T.copy(a_new, ai_s3[i, 0:BC])

            # ==== AiDiag staging: block-diag(I + loop result), split, ws ====
            T.tile.fill(aid_st, 0.0)
            T.copy(meye, ad_s)
            T.tile.add(ad_s, ad_s, ai_s0)
            T.copy(ad_s, aid_st[0:BC, 0:BC])
            T.copy(meye, ad_s)
            T.tile.add(ad_s, ad_s, ai_s1)
            T.copy(ad_s, aid_st[BC : 2 * BC, BC : 2 * BC])
            T.copy(meye, ad_s)
            T.tile.add(ad_s, ad_s, ai_s2)
            T.copy(ad_s, aid_st[2 * BC : 3 * BC, 2 * BC : 3 * BC])
            T.copy(meye, ad_s)
            T.tile.add(ad_s, ad_s, ai_s3)
            T.copy(ad_s, aid_st[3 * BC : 4 * BC, 3 * BC : 4 * BC])
            # split + full [64,64] AidHi/Lo writes
            T.copy(aid_st, aid_hi)
            T.copy(aid_hi, hi_f32)
            T.tile.sub(lo_f32, aid_st, hi_f32)
            T.copy(lo_f32, aid_lo)
            T.copy(aid_hi, AidHi[bb, s0 : s0 + BS, bh, 0:BS])
            T.copy(aid_lo, AidLo[bb, s0 : s0 + BS, bh, 0:BS])

    return main


# ============================================================================
# fwd kernel 2b: G = AiDiag @ Akkbp (direct GM->L1 reads of the ws pairs).
# ============================================================================
@tilelang.jit(pass_configs=PASS_CONFIGS)
def fla_chunk_intra_fwd_k_g(B, S, H, DK, chunk_size=64, sub_chunk_size=16):
    BS = chunk_size
    NT = S // BS
    GRID = NT * B * H

    @T.prim_func
    def main(
        AkkbpHi: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        AkkbpLo: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        AidHi: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        AidLo: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        GHi: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        GLo: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % NT
            bbh = cid // NT
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            # A side: Aid pair; B side: Akkbp pair (direct GM->L1 reads)
            aid_hi_l1 = T.alloc_shared([BS, BS], "bfloat16")
            aid_lo_l1 = T.alloc_shared([BS, BS], "bfloat16")
            akbp_hi_l1 = T.alloc_shared([BS, BS], "bfloat16")
            akbp_lo_l1 = T.alloc_shared([BS, BS], "bfloat16")
            g_frag = T.alloc_fragment([BS, BS], "float32")
            g_ub = T.alloc_shared([BS, BS], "float32")
            # G pre-split staging (hi/lo pair)
            g_hi64 = T.alloc_shared([BS, BS], "bfloat16")
            g_lo64 = T.alloc_shared([BS, BS], "bfloat16")
            hi_f32 = T.alloc_shared([BS, BS], "float32")
            lo_f32 = T.alloc_shared([BS, BS], "float32")

            # ==== C: direct reads + G = AiDiag @ Akkbp ====
            T.copy(AidHi[bb, s0 : s0 + BS, bh, 0:BS], aid_hi_l1)
            T.copy(AidLo[bb, s0 : s0 + BS, bh, 0:BS], aid_lo_l1)
            T.copy(AkkbpHi[bb, s0 : s0 + BS, bh, 0:BS], akbp_hi_l1)
            T.copy(AkkbpLo[bb, s0 : s0 + BS, bh, 0:BS], akbp_lo_l1)
            T.gemm_v0(aid_hi_l1, akbp_hi_l1, g_frag, init=True)
            T.gemm_v0(aid_lo_l1, akbp_hi_l1, g_frag, init=False)
            T.gemm_v0(aid_hi_l1, akbp_lo_l1, g_frag, init=False)

            # ==== V: frag relay + G pre-split full write ====
            # Rows 0..16 are real zeros from the gemm (Akkbp's staged zero
            # rows -> AiDiag's I-block x 0)
            T.copy(g_frag, g_ub)
            T.copy(g_ub, g_hi64)
            T.copy(g_hi64, hi_f32)
            T.tile.sub(lo_f32, g_ub, hi_f32)
            T.copy(lo_f32, g_lo64)
            T.copy(g_hi64, GHi[bb, s0 : s0 + BS, bh, 0:BS])
            T.copy(g_lo64, GLo[bb, s0 : s0 + BS, bh, 0:BS])

    return main


# ============================================================================
# fwd kernel 3: G2 = G @ G (direct GM->L1 reads; A and B get separate L1
# buffers -- one L1 buffer carries only one zN layout).
# ============================================================================
@tilelang.jit(pass_configs=PASS_CONFIGS)
def fla_chunk_intra_fwd_k_g2(B, S, H, DK, chunk_size=64, sub_chunk_size=16):
    BS = chunk_size
    NT = S // BS
    GRID = NT * B * H

    @T.prim_func
    def main(
        GHi: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        GLo: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        G2Hi: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        G2Lo: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % NT
            bbh = cid // NT
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            # A/B sides get separate L1 buffers (same GM source, direct reads)
            ga_hi_l1 = T.alloc_shared([BS, BS], "bfloat16")
            ga_lo_l1 = T.alloc_shared([BS, BS], "bfloat16")
            gb_hi_l1 = T.alloc_shared([BS, BS], "bfloat16")
            gb_lo_l1 = T.alloc_shared([BS, BS], "bfloat16")
            g2_frag = T.alloc_fragment([BS, BS], "float32")
            g2_ub = T.alloc_shared([BS, BS], "float32")
            # pre-split output staging
            g2_hi = T.alloc_shared([BS, BS], "bfloat16")
            g2_lo = T.alloc_shared([BS, BS], "bfloat16")
            hi_f32 = T.alloc_shared([BS, BS], "float32")
            lo_f32 = T.alloc_shared([BS, BS], "float32")

            # C: direct reads + G2 = G @ G
            T.copy(GHi[bb, s0 : s0 + BS, bh, 0:BS], ga_hi_l1)
            T.copy(GLo[bb, s0 : s0 + BS, bh, 0:BS], ga_lo_l1)
            T.copy(GHi[bb, s0 : s0 + BS, bh, 0:BS], gb_hi_l1)
            T.copy(GLo[bb, s0 : s0 + BS, bh, 0:BS], gb_lo_l1)
            T.gemm_v0(ga_hi_l1, gb_hi_l1, g2_frag, init=True)
            T.gemm_v0(ga_lo_l1, gb_hi_l1, g2_frag, init=False)
            T.gemm_v0(ga_hi_l1, gb_lo_l1, g2_frag, init=False)

            # V: frag relay + G2 pre-split full write
            T.copy(g2_frag, g2_ub)
            T.copy(g2_ub, g2_hi)
            T.copy(g2_hi, hi_f32)
            T.tile.sub(lo_f32, g2_ub, hi_f32)
            T.copy(lo_f32, g2_lo)
            T.copy(g2_hi, G2Hi[bb, s0 : s0 + BS, bh, 0:BS])
            T.copy(g2_lo, G2Lo[bb, s0 : s0 + BS, bh, 0:BS])

    return main


# ============================================================================
# fwd kernel 4: Y = AiDiag + G2 @ AiDiag (the I+ folds into the V epilogue
# as an exact fp32 add; operands are direct GM->L1 ws reads).
# ============================================================================
@tilelang.jit(pass_configs=PASS_CONFIGS)
def fla_chunk_intra_fwd_k_y(B, S, H, DK, chunk_size=64, sub_chunk_size=16):
    BS = chunk_size
    NT = S // BS
    GRID = NT * B * H

    @T.prim_func
    def main(
        G2Hi: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        G2Lo: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        AidHi: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        AidLo: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        YHi: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        YLo: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % NT
            bbh = cid // NT
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            g2_hi_l1 = T.alloc_shared([BS, BS], "bfloat16")
            g2_lo_l1 = T.alloc_shared([BS, BS], "bfloat16")
            aid_hi_l1 = T.alloc_shared([BS, BS], "bfloat16")
            aid_lo_l1 = T.alloc_shared([BS, BS], "bfloat16")
            y_frag = T.alloc_fragment([BS, BS], "float32")
            y_ub = T.alloc_shared([BS, BS], "float32")
            # Aid fp32 reconstruction + Y staging
            aid_h = T.alloc_shared([BS, BS], "bfloat16")
            aid_l = T.alloc_shared([BS, BS], "bfloat16")
            aid_hf = T.alloc_shared([BS, BS], "float32")
            aid_lf = T.alloc_shared([BS, BS], "float32")
            y_f32 = T.alloc_shared([BS, BS], "float32")
            y_hi = T.alloc_shared([BS, BS], "bfloat16")
            y_lo = T.alloc_shared([BS, BS], "bfloat16")

            # C: direct reads + Y_core = G2 @ Aid
            T.copy(G2Hi[bb, s0 : s0 + BS, bh, 0:BS], g2_hi_l1)
            T.copy(G2Lo[bb, s0 : s0 + BS, bh, 0:BS], g2_lo_l1)
            T.copy(AidHi[bb, s0 : s0 + BS, bh, 0:BS], aid_hi_l1)
            T.copy(AidLo[bb, s0 : s0 + BS, bh, 0:BS], aid_lo_l1)
            T.gemm_v0(g2_hi_l1, aid_hi_l1, y_frag, init=True)
            T.gemm_v0(g2_lo_l1, aid_hi_l1, y_frag, init=False)
            T.gemm_v0(g2_hi_l1, aid_lo_l1, y_frag, init=False)

            # V: frag relay + exact fp32 +Aid, then the Y pre-split pair
            T.copy(y_frag, y_ub)
            T.copy(AidHi[bb, s0 : s0 + BS, bh, 0:BS], aid_h)
            T.copy(AidLo[bb, s0 : s0 + BS, bh, 0:BS], aid_l)
            T.copy(aid_h, aid_hf)
            T.copy(aid_l, aid_lf)
            T.tile.add(y_f32, y_ub, aid_hf)
            T.tile.add(y_f32, y_f32, aid_lf)
            T.copy(y_f32, y_hi)
            T.copy(y_hi, aid_hf)
            T.tile.sub(aid_lf, y_f32, aid_hf)
            T.copy(aid_lf, y_lo)
            T.copy(y_hi, YHi[bb, s0 : s0 + BS, bh, 0:BS])
            T.copy(y_lo, YLo[bb, s0 : s0 + BS, bh, 0:BS])

    return main


# ============================================================================
# fwd kernel 5: Ai = Y + G @ Y -> bf16 Akk output (I+ folds into the V
# epilogue as an exact fp32 add; single full bf16 write).
# ============================================================================
@tilelang.jit(out_idx=[4], pass_configs=PASS_CONFIGS)
def fla_chunk_intra_fwd_k_ai(B, S, H, DK, chunk_size=64, sub_chunk_size=16):
    BS = chunk_size
    NT = S // BS
    GRID = NT * B * H

    @T.prim_func
    def main(
        GHi: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        GLo: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        YHi: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        YLo: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        Akk: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % NT
            bbh = cid // NT
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            g_hi_l1 = T.alloc_shared([BS, BS], "bfloat16")
            g_lo_l1 = T.alloc_shared([BS, BS], "bfloat16")
            y_hi_l1 = T.alloc_shared([BS, BS], "bfloat16")
            y_lo_l1 = T.alloc_shared([BS, BS], "bfloat16")
            ai_frag = T.alloc_fragment([BS, BS], "float32")
            ai_ub = T.alloc_shared([BS, BS], "float32")
            # Y fp32 reconstruction + output transit
            y_h = T.alloc_shared([BS, BS], "bfloat16")
            y_l = T.alloc_shared([BS, BS], "bfloat16")
            y_hf = T.alloc_shared([BS, BS], "float32")
            y_lf = T.alloc_shared([BS, BS], "float32")
            ai_f32 = T.alloc_shared([BS, BS], "float32")
            ai_bf = T.alloc_shared([BS, BS], "bfloat16")

            # C: direct reads + Ai_core = G @ Y
            T.copy(GHi[bb, s0 : s0 + BS, bh, 0:BS], g_hi_l1)
            T.copy(GLo[bb, s0 : s0 + BS, bh, 0:BS], g_lo_l1)
            T.copy(YHi[bb, s0 : s0 + BS, bh, 0:BS], y_hi_l1)
            T.copy(YLo[bb, s0 : s0 + BS, bh, 0:BS], y_lo_l1)
            T.gemm_v0(g_hi_l1, y_hi_l1, ai_frag, init=True)
            T.gemm_v0(g_lo_l1, y_hi_l1, ai_frag, init=False)
            T.gemm_v0(g_hi_l1, y_lo_l1, ai_frag, init=False)

            # V: frag relay + exact fp32 +Y, cast, single full bf16 write
            T.copy(ai_frag, ai_ub)
            T.copy(YHi[bb, s0 : s0 + BS, bh, 0:BS], y_h)
            T.copy(YLo[bb, s0 : s0 + BS, bh, 0:BS], y_l)
            T.copy(y_h, y_hf)
            T.copy(y_l, y_lf)
            T.tile.add(ai_f32, ai_ub, y_hf)
            T.tile.add(ai_f32, ai_f32, y_lf)
            T.copy(ai_f32, ai_bf)
            T.copy(ai_bf, Akk[bb, s0 : s0 + BS, bh, 0:BS])

    return main


# ============================================================================
# bwd kernel 1 of 2 (pure Vector, no Cube segment, no cross-core flags):
# stage every GEMM operand tile to GM scratch; the kernel boundary
# (host-ordered launches) provides the V->C synchronization.  Chunk-mid
# anchor a = g[2*BC, :] -- the per-s anchors cancel inside the D_s / V_s
# post-scaled sums, so one shared operand per matrix role serves all 4
# sub-chunk blocks and the fp16 exponents stay within +-half-chunk decay:
#   K~  = k * 2^(a - g_j)         Q~  = q * 2^(g_j - a)
#   KB~ = beta_j k_j * 2^(g_j - a)     MdA/MdK: full [64,64] Mlo-masked
# GM layouts (fp16, per (chunk, bh) program):
#   OpscrM [GRID, 4, 64, 64]: 0=mda_hi 1=mda_lo 2=mdk_hi 3=mdk_lo
#   Opscr  [GRID, 2, 6, 64, BK] (per half): 0=k_hi 1=k_lo 2=q_hi 3=q_lo
#                                      4=kb_hi 5=kb_lo
# ============================================================================
@tilelang.jit(pass_configs=PASS_CONFIGS)
def fla_chunk_intra_bwd_k_stage(B, S, H, DK, chunk_size=64, sub_chunk_size=16):
    BS = chunk_size
    BC = sub_chunk_size
    NT = S // BS
    GRID = NT * B * H
    BK = _pick_bk(DK)

    @T.prim_func
    def main(
        Q: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        K: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        G: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        BetaFlat: T.Tensor((B, S * H), "float32"),  # type: ignore
        Mlo: T.Tensor((BS, BS), "float32"),  # type: ignore
        dAqk: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
        dAkk: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
        OpscrM: T.Tensor((GRID, 4, BS, BS), "float16"),  # type: ignore
        Opscr: T.Tensor((GRID, 2, 6, BS, BK), "float16"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % NT
            bbh = cid // NT
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            # ---- prologue buffers ----
            mlo_ub = T.alloc_shared([BS, BS], "float32")
            mda_f = T.alloc_shared([BS, BS], "float32")
            mdk_f = T.alloc_shared([BS, BS], "float32")
            beta_tile = T.alloc_shared([BS * H], "float32")
            off_i32 = T.alloc_shared([BS], "int32")
            off_u32 = T.alloc_shared([BS], "uint32")
            bet = T.alloc_shared([BS], "float32")
            # ---- Md split staging (shared pure-V temps) ----
            md_hi = T.alloc_shared([BS, BS], "float16")
            md_lo = T.alloc_shared([BS, BS], "float16")
            md_hf = T.alloc_shared([BS, BS], "float32")
            md_lf = T.alloc_shared([BS, BS], "float32")
            # ---- per-half decay staging ----
            k_ub = T.alloc_shared([BS, BK], "float32")
            g_ub = T.alloc_shared([BS, BK], "float32")
            q_ub = T.alloc_shared([BS, BK], "float32")
            a_v = T.alloc_shared([BK], "float32")
            t16 = T.alloc_shared([BS, BK], "float32")
            bcol2d = T.alloc_shared([BS, BK], "float32")
            odec_f = T.alloc_shared([BS, BK], "float32")
            sec_hi = T.alloc_shared([BS, BK], "float16")
            sec_lo = T.alloc_shared([BS, BK], "float16")
            sec_hf = T.alloc_shared([BS, BK], "float32")
            sec_lf = T.alloc_shared([BS, BK], "float32")

            # ==== prologue (pure V): MdA/MdK (Mlo-masked) + beta gather ====
            T.copy(Mlo, mlo_ub)
            T.copy(dAqk[bb, s0 : s0 + BS, bh, 0:BS], mda_f)
            T.copy(dAkk[bb, s0 : s0 + BS, bh, 0:BS], mdk_f)
            T.tile.mul(mda_f, mda_f, mlo_ub)
            T.tile.mul(mdk_f, mdk_f, mlo_ub)
            T.copy(BetaFlat[bb, s0 * H : (s0 + BS) * H], beta_tile)
            T.tile.arith_progression(off_i32, bh * 4, H * 4)
            T.reinterpretcast(off_u32, off_i32, "uint32_t")
            T.tile.gather(bet, beta_tile, off_u32, 0)

            # ==== Md tiles: mda/mdk hi/lo -> OpscrM (4 whole [64,64] writes) ====
            T.copy(mda_f, md_hi)
            T.copy(md_hi, md_hf)
            T.tile.sub(md_lf, mda_f, md_hf)
            T.copy(md_hi, OpscrM[cid, 0, 0:BS, 0:BS])
            T.copy(md_lf, md_lo)
            T.copy(md_lo, OpscrM[cid, 1, 0:BS, 0:BS])
            T.copy(mdk_f, md_hi)
            T.copy(md_hi, md_hf)
            T.tile.sub(md_lf, mdk_f, md_hf)
            T.copy(md_hi, OpscrM[cid, 2, 0:BS, 0:BS])
            T.copy(md_lf, md_lo)
            T.copy(md_lo, OpscrM[cid, 3, 0:BS, 0:BS])

            # ==== per-half decay operands (chunk-mid anchor a = g[2*BC, :]) ====
            for i_kk in T.serial(0, 2):
                ik0 = i_kk * BK
                ik1 = ik0 + BK
                T.copy(K[bb, s0 : s0 + BS, bh, ik0:ik1], k_ub)
                T.copy(G[bb, s0 : s0 + BS, bh, ik0:ik1], g_ub)
                T.copy(Q[bb, s0 : s0 + BS, bh, ik0:ik1], q_ub)
                T.copy(g_ub[2 * BC, 0:BK], a_v)
                # K~ = k * 2^(a - g)
                T.tile.broadcast(t16, a_v, axis=0)
                T.tile.sub(t16, t16, g_ub)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f, k_ub, t16)
                T.copy(odec_f, sec_hi)
                T.copy(sec_hi, sec_hf)
                T.tile.sub(sec_lf, odec_f, sec_hf)
                T.copy(sec_hi, Opscr[cid, i_kk, 0, 0:BS, 0:BK])
                T.copy(sec_lf, sec_lo)
                T.copy(sec_lo, Opscr[cid, i_kk, 1, 0:BS, 0:BK])
                # ea = 2^(g - a): the shared Q~/KB~ exponent factor
                T.tile.broadcast(t16, a_v, axis=0)
                T.tile.sub(t16, g_ub, t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                # Q~ = q * ea
                T.tile.mul(odec_f, q_ub, t16)
                T.copy(odec_f, sec_hi)
                T.copy(sec_hi, sec_hf)
                T.tile.sub(sec_lf, odec_f, sec_hf)
                T.copy(sec_hi, Opscr[cid, i_kk, 2, 0:BS, 0:BK])
                T.copy(sec_lf, sec_lo)
                T.copy(sec_lo, Opscr[cid, i_kk, 3, 0:BS, 0:BK])
                # KB~ = beta * k * ea
                T.tile.mul(odec_f, k_ub, t16)
                T.tile.broadcast(bcol2d, bet, axis=1)
                T.tile.mul(odec_f, odec_f, bcol2d)
                T.copy(odec_f, sec_hi)
                T.copy(sec_hi, sec_hf)
                T.tile.sub(sec_lf, odec_f, sec_hf)
                T.copy(sec_hi, Opscr[cid, i_kk, 4, 0:BS, 0:BK])
                T.copy(sec_lf, sec_lo)
                T.copy(sec_lo, Opscr[cid, i_kk, 5, 0:BS, 0:BK])

    return main


# ============================================================================
# bwd kernel 2 of 2: gemms + tail.  Pipelined(2, num_stages=1) over the DK
# halves; body = one C run then one V run.  Post-scales (a = g[2*BC, :]):
#   dq2_pre = MdA @ K~ ;  dk2_pre = MdK @ K~        (row gemms)
#   dkt_pre = MdA^T @ Q~ + MdK^T @ KB~              (transpose_A gemms)
#   dq2 = ea*dq2_pre + dq ; dk2b = ea*dk2_pre*beta ; dkt = eam*dkt_pre
#     (ea = 2^(g-a) row-wise, eam = 2^(a-g))
# C: GM->L1 whole-tile reads, 12 hi/lo gemm_v0, L0C->GM direct fp32 frag
#   writes (Fragscr) + set_cross_flag("FIX", 9 + i_kk%2).
# V: wait_cross_flag, frag readback, corrected tail + db_acc accumulation.
# Fragscr layout (per (cid, half), rows): 0:64=dq2, 64:128=dk2, 128:192=dkt.
# ============================================================================
@tilelang.jit(out_idx=[14, 15, 16], pass_configs=PASS_CONFIGS)
def fla_chunk_intra_bwd_k_gemm(B, S, H, DK, chunk_size=64, sub_chunk_size=16):
    BS = chunk_size
    BC = sub_chunk_size
    NT = S // BS
    GRID = NT * B * H
    BK = _pick_bk(DK)

    @T.prim_func
    def main(
        Q: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        K: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        G: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        BetaFlat: T.Tensor((B, S * H), "float32"),  # type: ignore
        dq: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        dk: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        dg: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        OpscrM: T.Tensor((GRID, 4, BS, BS), "float16"),  # type: ignore
        Opscr: T.Tensor((GRID, 2, 6, BS, BK), "float16"),  # type: ignore
        Fragscr: T.Tensor((GRID, 2, 12 * BC, BK), "float32"),  # type: ignore
        db2flat: T.Tensor((GRID, BS), "float32"),  # type: ignore
        db2flat_v: T.Tensor((GRID, BS, 1), "float32"),  # type: ignore
        db2: T.Tensor((B, S, H), "float32"),  # type: ignore
        db2f: T.Tensor((B, S * H), "float32"),  # type: ignore
        dq2: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        dk2: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        dg2: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % NT
            bbh = cid // NT
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            # ---- L1 tiles: full MdA/MdK hi/lo [64,64] ----
            mda_hi = T.alloc_shared([BS, BS], "float16")
            mda_lo = T.alloc_shared([BS, BS], "float16")
            mdk_hi = T.alloc_shared([BS, BS], "float16")
            mdk_lo = T.alloc_shared([BS, BS], "float16")
            # ---- L1 tiles: per-half decay operands [64,BK] ----
            k_hi = T.alloc_shared([BS, BK], "float16")
            k_lo = T.alloc_shared([BS, BK], "float16")
            q_hi = T.alloc_shared([BS, BK], "float16")
            q_lo = T.alloc_shared([BS, BK], "float16")
            kb_hi = T.alloc_shared([BS, BK], "float16")
            kb_lo = T.alloc_shared([BS, BK], "float16")
            # ---- fragments (L0C) ----
            dq2_fg = T.alloc_fragment([BS, BK], "float32")
            dk2_fg = T.alloc_fragment([BS, BK], "float32")
            dkt_fg = T.alloc_fragment([BS, BK], "float32")
            # ---- V-side buffers ----
            g_ub = T.alloc_shared([BS, BK], "float32")
            k_ub = T.alloc_shared([BS, BK], "float32")
            q_ub = T.alloc_shared([BS, BK], "float32")
            beta_tile = T.alloc_shared([BS * H], "float32")
            off_i32 = T.alloc_shared([BS], "int32")
            off_u32 = T.alloc_shared([BS], "uint32")
            bet = T.alloc_shared([BS], "float32")
            a_v = T.alloc_shared([BK], "float32")
            t16 = T.alloc_shared([BS, BK], "float32")
            ea_f = T.alloc_shared([BS, BK], "float32")
            f_dq2 = T.alloc_shared([BC, BK], "float32")
            f_dk2 = T.alloc_shared([BC, BK], "float32")
            f_dkt = T.alloc_shared([BC, BK], "float32")
            dk2b_ub = T.alloc_shared([BC, BK], "float32")
            dq2_ub = T.alloc_shared([BC, BK], "float32")
            dk2_ub = T.alloc_shared([BC, BK], "float32")
            dg2_ub = T.alloc_shared([BC, BK], "float32")
            t16b = T.alloc_shared([BC, BK], "float32")
            r_tmp = T.alloc_shared([BC, BK], "float32")
            r_part = T.alloc_shared([BC], "float32")
            dq_rows = T.alloc_shared([BC, BK], "float32")
            dk_rows = T.alloc_shared([BC, BK], "float32")
            dg_rows = T.alloc_shared([BC, BK], "float32")
            bcol2d = T.alloc_shared([BC, BK], "float32")
            db_acc = T.alloc_shared([BS], "float32")
            db2d = T.alloc_shared([BS, 8], "float32")

            # ---- prologue (pure V): beta gather + db_acc zero ----
            T.copy(BetaFlat[bb, s0 * H : (s0 + BS) * H], beta_tile)
            T.tile.arith_progression(off_i32, bh * 4, H * 4)
            T.reinterpretcast(off_u32, off_i32, "uint32_t")
            T.tile.gather(bet, beta_tile, off_u32, 0)
            T.tile.fill(db_acc, 0.0)

            for i_kk in T.Pipelined(DK // BK, num_stages=1):
                ik0 = i_kk * BK
                ik1 = ik0 + BK

                # ==== C: GM -> L1 whole-tile reads (4 Md + 6 decay) ====
                T.copy(OpscrM[cid, 0, 0:BS, 0:BS], mda_hi)
                T.copy(OpscrM[cid, 1, 0:BS, 0:BS], mda_lo)
                T.copy(OpscrM[cid, 2, 0:BS, 0:BS], mdk_hi)
                T.copy(OpscrM[cid, 3, 0:BS, 0:BS], mdk_lo)
                T.copy(Opscr[cid, i_kk, 0, 0:BS, 0:BK], k_hi)
                T.copy(Opscr[cid, i_kk, 1, 0:BS, 0:BK], k_lo)
                T.copy(Opscr[cid, i_kk, 2, 0:BS, 0:BK], q_hi)
                T.copy(Opscr[cid, i_kk, 3, 0:BS, 0:BK], q_lo)
                T.copy(Opscr[cid, i_kk, 4, 0:BS, 0:BK], kb_hi)
                T.copy(Opscr[cid, i_kk, 5, 0:BS, 0:BK], kb_lo)

                # ==== C: 12 gemm_v0 (4 logical x hi/lo 3-gemm) ====
                # dq2_pre = MdA @ K~
                T.gemm_v0(mda_hi, k_hi, dq2_fg, init=True)
                T.gemm_v0(mda_lo, k_hi, dq2_fg, init=False)
                T.gemm_v0(mda_hi, k_lo, dq2_fg, init=False)
                # dk2_pre = MdK @ K~
                T.gemm_v0(mdk_hi, k_hi, dk2_fg, init=True)
                T.gemm_v0(mdk_lo, k_hi, dk2_fg, init=False)
                T.gemm_v0(mdk_hi, k_lo, dk2_fg, init=False)
                # dkt_pre = MdA^T @ Q~ + MdK^T @ KB~
                T.gemm_v0(mda_hi, q_hi, dkt_fg, transpose_A=True, init=True)
                T.gemm_v0(mda_lo, q_hi, dkt_fg, transpose_A=True, init=False)
                T.gemm_v0(mda_hi, q_lo, dkt_fg, transpose_A=True, init=False)
                T.gemm_v0(mdk_hi, kb_hi, dkt_fg, transpose_A=True, init=False)
                T.gemm_v0(mdk_lo, kb_hi, dkt_fg, transpose_A=True, init=False)
                T.gemm_v0(mdk_hi, kb_lo, dkt_fg, transpose_A=True, init=False)

                # ==== C: L0C -> GM direct frag writes + fix-done flag ====
                T.copy(dq2_fg, Fragscr[cid, i_kk, 0:64, 0:BK])
                T.copy(dk2_fg, Fragscr[cid, i_kk, 64:128, 0:BK])
                T.copy(dkt_fg, Fragscr[cid, i_kk, 128:192, 0:BK])
                T.set_cross_flag("FIX", 9 + i_kk % 2)

                # ==== V: wait + loads + shared post-scales ====
                T.copy(G[bb, s0 : s0 + BS, bh, ik0:ik1], g_ub)
                T.copy(K[bb, s0 : s0 + BS, bh, ik0:ik1], k_ub)
                T.copy(Q[bb, s0 : s0 + BS, bh, ik0:ik1], q_ub)
                T.copy(g_ub[2 * BC, 0:BK], a_v)
                T.wait_cross_flag(9 + i_kk % 2, "V")
                # ea = 2^(g - a)  (dq2/dk2 row post-scale)
                T.tile.broadcast(t16, a_v, axis=0)
                T.tile.sub(ea_f, g_ub, t16)
                T.tile.mul(ea_f, ea_f, LN2)
                T.tile.exp(ea_f, ea_f)
                # eam = 2^(a - g)  (dkt row post-scale; lives in t16)
                T.tile.broadcast(t16, a_v, axis=0)
                T.tile.sub(t16, t16, g_ub)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)

                # ---- s = 0 (rows 0..15) ----
                T.copy(Fragscr[cid, i_kk, 0:16, 0:BK], f_dq2)
                T.copy(Fragscr[cid, i_kk, 64:80, 0:BK], f_dk2)
                T.copy(Fragscr[cid, i_kk, 128:144, 0:BK], f_dkt)
                T.tile.mul(f_dq2, f_dq2, ea_f[0:BC, 0:BK])
                T.tile.mul(f_dk2, f_dk2, ea_f[0:BC, 0:BK])
                T.tile.mul(f_dkt, f_dkt, t16[0:BC, 0:BK])
                T.copy(dq[bb, s0 : s0 + BC, bh, ik0:ik1], dq_rows)
                T.tile.add(dq2_ub, f_dq2, dq_rows)
                T.copy(dq2_ub, dq2[bb, s0 : s0 + BC, bh, ik0:ik1])
                T.tile.mul(r_tmp, f_dk2, k_ub[0:BC, 0:BK])
                T.reduce_sum(r_tmp, r_part, dim=1, clear=True)
                T.tile.add(db_acc[0:BC], db_acc[0:BC], r_part)
                T.tile.broadcast(bcol2d, bet[0:BC], axis=1)
                T.tile.mul(dk2b_ub, f_dk2, bcol2d)
                T.tile.sub(t16b, dk2b_ub, f_dkt)
                T.tile.mul(t16b, t16b, k_ub[0:BC, 0:BK])
                T.tile.mul(dg2_ub, q_ub[0:BC, 0:BK], f_dq2)
                T.tile.add(dg2_ub, dg2_ub, t16b)
                T.copy(dg[bb, s0 : s0 + BC, bh, ik0:ik1], dg_rows)
                T.tile.add(dg2_ub, dg2_ub, dg_rows)
                T.copy(dg2_ub, dg2[bb, s0 : s0 + BC, bh, ik0:ik1])
                T.copy(dk[bb, s0 : s0 + BC, bh, ik0:ik1], dk_rows)
                T.tile.add(dk2_ub, dk2b_ub, f_dkt)
                T.tile.add(dk2_ub, dk2_ub, dk_rows)
                T.copy(dk2_ub, dk2[bb, s0 : s0 + BC, bh, ik0:ik1])
                # ---- s = 1 (rows 16..31) ----
                T.copy(Fragscr[cid, i_kk, 16:32, 0:BK], f_dq2)
                T.copy(Fragscr[cid, i_kk, 80:96, 0:BK], f_dk2)
                T.copy(Fragscr[cid, i_kk, 144:160, 0:BK], f_dkt)
                T.tile.mul(f_dq2, f_dq2, ea_f[BC : 2 * BC, 0:BK])
                T.tile.mul(f_dk2, f_dk2, ea_f[BC : 2 * BC, 0:BK])
                T.tile.mul(f_dkt, f_dkt, t16[BC : 2 * BC, 0:BK])
                T.copy(dq[bb, s0 + BC : s0 + 2 * BC, bh, ik0:ik1], dq_rows)
                T.tile.add(dq2_ub, f_dq2, dq_rows)
                T.copy(dq2_ub, dq2[bb, s0 + BC : s0 + 2 * BC, bh, ik0:ik1])
                T.tile.mul(r_tmp, f_dk2, k_ub[BC : 2 * BC, 0:BK])
                T.reduce_sum(r_tmp, r_part, dim=1, clear=True)
                T.tile.add(db_acc[BC : 2 * BC], db_acc[BC : 2 * BC], r_part)
                T.tile.broadcast(bcol2d, bet[BC : 2 * BC], axis=1)
                T.tile.mul(dk2b_ub, f_dk2, bcol2d)
                T.tile.sub(t16b, dk2b_ub, f_dkt)
                T.tile.mul(t16b, t16b, k_ub[BC : 2 * BC, 0:BK])
                T.tile.mul(dg2_ub, q_ub[BC : 2 * BC, 0:BK], f_dq2)
                T.tile.add(dg2_ub, dg2_ub, t16b)
                T.copy(dg[bb, s0 + BC : s0 + 2 * BC, bh, ik0:ik1], dg_rows)
                T.tile.add(dg2_ub, dg2_ub, dg_rows)
                T.copy(dg2_ub, dg2[bb, s0 + BC : s0 + 2 * BC, bh, ik0:ik1])
                T.copy(dk[bb, s0 + BC : s0 + 2 * BC, bh, ik0:ik1], dk_rows)
                T.tile.add(dk2_ub, dk2b_ub, f_dkt)
                T.tile.add(dk2_ub, dk2_ub, dk_rows)
                T.copy(dk2_ub, dk2[bb, s0 + BC : s0 + 2 * BC, bh, ik0:ik1])
                # ---- s = 2 (rows 32..47) ----
                T.copy(Fragscr[cid, i_kk, 32:48, 0:BK], f_dq2)
                T.copy(Fragscr[cid, i_kk, 96:112, 0:BK], f_dk2)
                T.copy(Fragscr[cid, i_kk, 160:176, 0:BK], f_dkt)
                T.tile.mul(f_dq2, f_dq2, ea_f[2 * BC : 3 * BC, 0:BK])
                T.tile.mul(f_dk2, f_dk2, ea_f[2 * BC : 3 * BC, 0:BK])
                T.tile.mul(f_dkt, f_dkt, t16[2 * BC : 3 * BC, 0:BK])
                T.copy(dq[bb, s0 + 2 * BC : s0 + 3 * BC, bh, ik0:ik1], dq_rows)
                T.tile.add(dq2_ub, f_dq2, dq_rows)
                T.copy(dq2_ub, dq2[bb, s0 + 2 * BC : s0 + 3 * BC, bh, ik0:ik1])
                T.tile.mul(r_tmp, f_dk2, k_ub[2 * BC : 3 * BC, 0:BK])
                T.reduce_sum(r_tmp, r_part, dim=1, clear=True)
                T.tile.add(db_acc[2 * BC : 3 * BC], db_acc[2 * BC : 3 * BC], r_part)
                T.tile.broadcast(bcol2d, bet[2 * BC : 3 * BC], axis=1)
                T.tile.mul(dk2b_ub, f_dk2, bcol2d)
                T.tile.sub(t16b, dk2b_ub, f_dkt)
                T.tile.mul(t16b, t16b, k_ub[2 * BC : 3 * BC, 0:BK])
                T.tile.mul(dg2_ub, q_ub[2 * BC : 3 * BC, 0:BK], f_dq2)
                T.tile.add(dg2_ub, dg2_ub, t16b)
                T.copy(dg[bb, s0 + 2 * BC : s0 + 3 * BC, bh, ik0:ik1], dg_rows)
                T.tile.add(dg2_ub, dg2_ub, dg_rows)
                T.copy(dg2_ub, dg2[bb, s0 + 2 * BC : s0 + 3 * BC, bh, ik0:ik1])
                T.copy(dk[bb, s0 + 2 * BC : s0 + 3 * BC, bh, ik0:ik1], dk_rows)
                T.tile.add(dk2_ub, dk2b_ub, f_dkt)
                T.tile.add(dk2_ub, dk2_ub, dk_rows)
                T.copy(dk2_ub, dk2[bb, s0 + 2 * BC : s0 + 3 * BC, bh, ik0:ik1])
                # ---- s = 3 (rows 48..63) ----
                T.copy(Fragscr[cid, i_kk, 48:64, 0:BK], f_dq2)
                T.copy(Fragscr[cid, i_kk, 112:128, 0:BK], f_dk2)
                T.copy(Fragscr[cid, i_kk, 176:192, 0:BK], f_dkt)
                T.tile.mul(f_dq2, f_dq2, ea_f[3 * BC : 4 * BC, 0:BK])
                T.tile.mul(f_dk2, f_dk2, ea_f[3 * BC : 4 * BC, 0:BK])
                T.tile.mul(f_dkt, f_dkt, t16[3 * BC : 4 * BC, 0:BK])
                T.copy(dq[bb, s0 + 3 * BC : s0 + 4 * BC, bh, ik0:ik1], dq_rows)
                T.tile.add(dq2_ub, f_dq2, dq_rows)
                T.copy(dq2_ub, dq2[bb, s0 + 3 * BC : s0 + 4 * BC, bh, ik0:ik1])
                T.tile.mul(r_tmp, f_dk2, k_ub[3 * BC : 4 * BC, 0:BK])
                T.reduce_sum(r_tmp, r_part, dim=1, clear=True)
                T.tile.add(db_acc[3 * BC : 4 * BC], db_acc[3 * BC : 4 * BC], r_part)
                T.tile.broadcast(bcol2d, bet[3 * BC : 4 * BC], axis=1)
                T.tile.mul(dk2b_ub, f_dk2, bcol2d)
                T.tile.sub(t16b, dk2b_ub, f_dkt)
                T.tile.mul(t16b, t16b, k_ub[3 * BC : 4 * BC, 0:BK])
                T.tile.mul(dg2_ub, q_ub[3 * BC : 4 * BC, 0:BK], f_dq2)
                T.tile.add(dg2_ub, dg2_ub, t16b)
                T.copy(dg[bb, s0 + 3 * BC : s0 + 4 * BC, bh, ik0:ik1], dg_rows)
                T.tile.add(dg2_ub, dg2_ub, dg_rows)
                T.copy(dg2_ub, dg2[bb, s0 + 3 * BC : s0 + 4 * BC, bh, ik0:ik1])
                T.copy(dk[bb, s0 + 3 * BC : s0 + 4 * BC, bh, ik0:ik1], dk_rows)
                T.tile.add(dk2_ub, dk2b_ub, f_dkt)
                T.tile.add(dk2_ub, dk2_ub, dk_rows)
                T.copy(dk2_ub, dk2[bb, s0 + 3 * BC : s0 + 4 * BC, bh, ik0:ik1])

            # ---- epilogue: db2 scatter via GM scratch + intra-core flag ----
            # [64,1] copies round-trip through 32B slots and only self-
            # compensate while the GM row stride exceeds 4B (H >= 2); the
            # wide db2d keeps the scattered writes contained.  H == 1 uses
            # the flat 1-D view instead (no narrow rows).
            if H == 1:
                # H == 1: contiguous slice, single flat-view row write
                T.copy(db_acc, db2f[bb, s0 : s0 + BS])
            else:
                # H >= 2: the [64,1] read/write slot pair self-compensates;
                T.copy(db_acc, db2flat[cid, 0:BS])
                T.set_flag("MTE3", "MTE2", 0)
                T.wait_flag("MTE3", "MTE2", 0)
                T.copy(db2flat_v[cid, 0:BS, 0:1], db2d[0:BS, 0:1])
                T.copy(db2d[0:BS, 0:1], db2[bb, s0 : s0 + BS, bh])

    return main


# ============================================================================
# Host wrappers (upstream naming; zero-copy metadata views only)
# ============================================================================
def _check_shape(B, S, H, DK, chunk_size, sub_chunk_size):
    if S % chunk_size != 0:
        raise ValueError(f"S={S} must be divisible by chunk_size={chunk_size}")
    if chunk_size != 64 or sub_chunk_size != 16:
        raise ValueError("chunk_size/sub_chunk_size fixed at 64/16 (block semantic)")
    if DK not in (64, 96, 128):
        raise ValueError(f"DK={DK} unsupported: must be one of {{64, 96, 128}} (bwd exact-2 Pipelined block constraint)")
    if H < 1 or B < 1:
        raise ValueError("B/H must be >= 1")


_MNEG_CACHE = {}


def _get_mneg(device, bc):
    """-strict_tril(ones) constant, constructed once per (device, size)."""
    key = (str(device), bc)
    if key not in _MNEG_CACHE:
        _MNEG_CACHE[key] = (-torch.tril(torch.ones(bc, bc, dtype=torch.float32), -1)).to(device)
    return _MNEG_CACHE[key]


_MLO_CACHE = {}


def _get_mlo(device, bs):
    """tril(ones, 0) constant, constructed once per (device, size)."""
    key = (str(device), bs)
    if key not in _MLO_CACHE:
        _MLO_CACHE[key] = torch.tril(torch.ones(bs, bs, dtype=torch.float32), 0).to(device)
    return _MLO_CACHE[key]


_MEYE_CACHE = {}


def _get_meye(device, n):
    """Identity constant, constructed once per (device, size)."""
    key = (str(device), n)
    if key not in _MEYE_CACHE:
        _MEYE_CACHE[key] = torch.eye(n, dtype=torch.float32).to(device)
    return _MEYE_CACHE[key]


def fla_chunk_intra_fwd_inter_solve_fused(q, k, gk, beta, Akk_diag, chunk_size=64, sub_chunk_size=16, scale=1.0):
    """fwd host API (upstream chunk_kda_fwd_inter_solve_fused).

    q/k [B,S,H,DK] bf16, gk [B,S,H,DK] fp32, beta [B,S,H] bf16,
    Akk_diag [B,S,H,16] fp32 -> (Aqk, Akk) [B,S,H,64] bf16.
    """
    B, S, H, DK = q.shape
    _check_shape(B, S, H, DK, chunk_size, sub_chunk_size)
    if q.dtype is not torch.bfloat16 or k.dtype is not torch.bfloat16:
        raise ValueError("fwd q/k must be bfloat16 (fp32 variant out of scope)")
    if gk.dtype is not torch.float32 or Akk_diag.dtype is not torch.float32:
        raise ValueError("fwd gk/Akk_diag must be float32")
    if beta.dtype is not torch.bfloat16:
        raise ValueError("fwd beta must be bfloat16")
    for name, t in (("q", q), ("k", k), ("gk", gk), ("beta", beta), ("Akk_diag", Akk_diag)):
        if not t.is_contiguous():
            raise ValueError(f"input '{name}' must be contiguous")

    device = q.device
    beta_flat = beta.view(B, S * H)  # zero-copy metadata view
    mneg = _get_mneg(device, sub_chunk_size)

    # cross-kernel GM workspaces: pre-split bf16 hi/lo pairs
    bf16_pair = lambda: (  # noqa: E731
        torch.empty(B, S, H, chunk_size, dtype=torch.bfloat16, device=device),
        torch.empty(B, S, H, chunk_size, dtype=torch.bfloat16, device=device),
    )
    akkbp_hi, akkbp_lo = bf16_pair()
    g_hi, g_lo = bf16_pair()
    g2_hi, g2_lo = bf16_pair()
    y_hi, y_lo = bf16_pair()
    aid_hi, aid_lo = bf16_pair()

    k_inter = fla_chunk_intra_fwd_k_inter(B, S, H, DK, chunk_size, sub_chunk_size, scale)
    Aqk = k_inter(q, k, gk, beta_flat, akkbp_hi, akkbp_lo)
    k_subst = fla_chunk_intra_fwd_k_subst(B, S, H, DK, chunk_size, sub_chunk_size)
    k_subst(Akk_diag, mneg, _get_meye(device, sub_chunk_size), aid_hi, aid_lo)
    k_g = fla_chunk_intra_fwd_k_g(B, S, H, DK, chunk_size, sub_chunk_size)
    k_g(akkbp_hi, akkbp_lo, aid_hi, aid_lo, g_hi, g_lo)
    k_g2 = fla_chunk_intra_fwd_k_g2(B, S, H, DK, chunk_size, sub_chunk_size)
    k_g2(g_hi, g_lo, g2_hi, g2_lo)
    k_y = fla_chunk_intra_fwd_k_y(B, S, H, DK, chunk_size, sub_chunk_size)
    k_y(g2_hi, g2_lo, aid_hi, aid_lo, y_hi, y_lo)
    k_ai = fla_chunk_intra_fwd_k_ai(B, S, H, DK, chunk_size, sub_chunk_size)
    Akk = k_ai(g_hi, g_lo, y_hi, y_lo)
    # no internal synchronize: the do_bench harness owns synchronization
    return Aqk, Akk


def fla_chunk_intra_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, chunk_size=64, sub_chunk_size=16):
    """bwd host API (upstream chunk_kda_bwd_intra).

    All fp32 -> (dq2, dk2, db2 [1,B,S,H] partial, dg2 pre-reverse-cumsum).
    Host post-processing (db2.sum(0)+db, reverse chunk cumsum of dg2) belongs
    to the caller, matching the upstream kernel/test boundary.

    Two JIT kernels: k_stage (pure-V operand staging
    to GM scratch) then k_gemm (Cube gemms + Vector tail, FIX-flag handoff).
    """
    B, S, H, DK = k.shape
    _check_shape(B, S, H, DK, chunk_size, sub_chunk_size)
    for name, t in (
        ("q", q),
        ("k", k),
        ("g", g),
        ("beta", beta),
        ("dAqk", dAqk),
        ("dAkk", dAkk),
        ("dq", dq),
        ("dk", dk),
        ("db", db),
        ("dg", dg),
    ):
        if t.dtype is not torch.float32:
            raise ValueError(f"bwd input '{name}' must be float32")
        if not t.is_contiguous():
            raise ValueError(f"input '{name}' must be contiguous")

    device = k.device
    beta_flat = beta.view(B, S * H)  # zero-copy metadata view
    mlo = _get_mlo(device, chunk_size)
    grid = (S // chunk_size) * B * H
    BK = _pick_bk(DK)
    BC = sub_chunk_size
    BS = chunk_size
    # GM staging scratch (fp16 operands + fp32 gemm results + db2 scatter)
    opscrm = torch.empty(grid, 4, BS, BS, dtype=torch.float16, device=device)
    opscr = torch.empty(grid, 2, 6, BS, BK, dtype=torch.float16, device=device)
    fragscr = torch.empty(grid, 2, 12 * BC, BK, dtype=torch.float32, device=device)
    db2flat = torch.empty(grid, chunk_size, dtype=torch.float32, device=device)
    db2flat_v = db2flat.view(grid, chunk_size, 1)  # zero-copy alias view

    db2 = torch.empty(B, S, H, dtype=torch.float32, device=device)
    db2f = db2.view(B, S * H)  # zero-copy flat view (H=1 fast scatter path)
    k_stage = fla_chunk_intra_bwd_k_stage(B, S, H, DK, chunk_size, sub_chunk_size)
    k_stage(q, k, g, beta_flat, mlo, dAqk, dAkk, opscrm, opscr)
    k_gemm = fla_chunk_intra_bwd_k_gemm(B, S, H, DK, chunk_size, sub_chunk_size)
    dq2, dk2, dg2 = k_gemm(q, k, g, beta_flat, dq, dk, dg, opscrm, opscr, fragscr, db2flat, db2flat_v, db2, db2f)
    # NK=1 partial (both DK halves accumulate in-program); zero-copy view
    return dq2, dk2, db2.view(1, B, S, H), dg2


__all__ = [
    "fla_chunk_intra_fwd_inter_solve_fused",
    "fla_chunk_intra_bwd_intra",
    "fla_chunk_intra_fwd_k_inter",
    "fla_chunk_intra_fwd_k_g",
    "fla_chunk_intra_fwd_k_subst",
    "fla_chunk_intra_fwd_k_g2",
    "fla_chunk_intra_fwd_k_y",
    "fla_chunk_intra_fwd_k_ai",
    "fla_chunk_intra_bwd_k_stage",
    "fla_chunk_intra_bwd_k_gemm",
    "PASS_CONFIGS",
]


if __name__ == "__main__":
    # Smoke test: minimal shapes for both directions (fwd bf16 + bwd fp32);
    # checks output shapes and finiteness. Input construction follows the
    # upstream test harnesses (fwd gk = logsigmoid -> chunk-local cumsum,
    # Akk_diag = ones; bwd raw randn, g NOT cumsum'd).
    import torch.nn.functional as F

    tilelang.cache.clear_cache()
    torch.manual_seed(42)
    B, S, H, DK, CS, SCS = 1, 64, 2, 128, 64, 16

    # ---- fwd smoke (bf16 in, fp32 gate) ----
    q = torch.randn(B, S, H, DK).bfloat16()
    k = torch.randn(B, S, H, DK).bfloat16()
    beta = torch.randn(B, S, H).bfloat16()
    gk = F.logsigmoid(torch.randn(B, S, H, DK, dtype=torch.float32))
    gk = gk.view(B, S // CS, CS, H, DK).cumsum(2).view(B, S, H, DK)
    akk_diag = torch.ones(B, S, H, SCS, dtype=torch.float32)
    aqk, akk = fla_chunk_intra_fwd_inter_solve_fused(q.npu(), k.npu(), gk.npu(), beta.npu(), akk_diag.npu())
    assert aqk.shape == (B, S, H, CS) and akk.shape == (B, S, H, CS)
    assert torch.isfinite(aqk.float()).all().item(), "Aqk contains NaN/Inf"
    assert torch.isfinite(akk.float()).all().item(), "Akk contains NaN/Inf"

    # ---- bwd smoke (all fp32) ----
    ins = [
        torch.randn(B, S, H, DK, dtype=torch.float32),  # q
        torch.randn(B, S, H, DK, dtype=torch.float32),  # k
        torch.randn(B, S, H, DK, dtype=torch.float32),  # g (NOT cumsum'd)
        torch.randn(B, S, H, dtype=torch.float32),  # beta
        torch.randn(B, S, H, CS, dtype=torch.float32),  # dAqk
        torch.randn(B, S, H, CS, dtype=torch.float32),  # dAkk
        torch.randn(B, S, H, DK, dtype=torch.float32),  # dq
        torch.randn(B, S, H, DK, dtype=torch.float32),  # dk
        torch.randn(B, S, H, dtype=torch.float32),  # db
        torch.randn(B, S, H, DK, dtype=torch.float32),  # dg
    ]
    dq2, dk2, db2, dg2 = fla_chunk_intra_bwd_intra(*[t.npu() for t in ins])
    assert dq2.shape == (B, S, H, DK) and dk2.shape == (B, S, H, DK)
    assert db2.shape == (1, B, S, H) and dg2.shape == (B, S, H, DK)
    for name, out in (("dq2", dq2), ("dk2", dk2), ("db2", db2), ("dg2", dg2)):
        assert torch.isfinite(out.float()).all().item(), f"{name} contains NaN/Inf"

    print(f"[SMOKE_PASS] fla_chunk_intra fwd+bwd B={B} S={S} H={H} DK={DK} chunk={CS} sub_chunk={SCS}")
    print("Test Passed!")
