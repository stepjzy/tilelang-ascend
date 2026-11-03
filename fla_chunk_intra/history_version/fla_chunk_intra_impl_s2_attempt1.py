"""fla_chunk_intra: KDA intra-chunk fwd (fused inter + solve_tril) + bwd on
Ascend 910B3 (Developer mode, threads=1).

Semantics authority (read-only): ref_gpu/fla_chunk_intra.py (Triton/FLA);
code-level origin: ref_gpu/fla_chunk_intra_tilelang_gpu_{fwd,bwd}.py.
Design doc: DESIGN_fla_chunk_intra.md (Stage 1).

fwd  (chunk_kda_fwd_inter_solve_fused):  per (chunk, bh):
  * off-diagonal blocks (s2 in {1,2,3}, anchor gn = g[s2*16, :]):
        qg   = q_{s2} * 2^(g_{s2} - gn)          Aqk[s2, 0:16*s2] = scale * qg @ rhs^T
        kg'  = k_{s2} * 2^(g_{s2} - gn) * (-beta)   (sign-folded Akk-beta')
        rhs  = k_{0:16*s2} * 2^(gn - g_{0:16*s2})   Akkbp[s2, 0:16*s2] = kg' @ rhs^T
  * 4 diagonal forward substitutions: Ai_ss = (I + strict_tril(A_s))^{-1}
        (row recursion, pure-V T.serial; N_s = -strict_tril(A_s) via the
        Mneg constant -- the negation lives inside N, upstream L264-274)
  * merged block inverse via the sign-folded identity
        G  = AiDiag @ Akkbp          (Akkbp = -beta*Akk, so kernel G = -G_orig)
        Ai = (I + G)(I + G^2) D^{-1} (= M^{-1}, block lower triangular)
  * Aqk/Akk outputs bf16; Aqk diagonal + block-upper-tri explicitly written 0
    (deterministic contract, DESIGN Sec. 1.2); Akk block-upper-tri is exact 0.

bwd  (chunk_kda_bwd_intra):  per (chunk, bh), one Pipelined(2) loop over the
two DK halves (BK = DK//2, exact-2 pipeline extent).  Each half covers all 4
sub-chunks s (static unroll); the per-column sequential loops of the upstream
(B)/(E) steps are re-expressed as masked GEMMs (mask depends only on (i, j),
never on iteration state):
  (A)+(B)  dq2_pre_s  = D_s * ( MdA[s*16:+16, 0:KMAX] @ Kdec_s )   KMAX=32/32/48/64
           dk2pre_s   = same with MdK                     (s=0 pads K to 32 via
                                                             MdA block (0,1)'s
                                                             exact-zero columns)
  (D)+(E)  dkt_s      = V_s * ( MdA[:, s*16:+16]^T @ Qdec_s
                              + MdK[:, s*16:+16]^T @ KBdec_s )
           Kdec_s[j] = k_j * 2^(gn_s  - g_j)     gn_s  = g[s*16, :]
           Qdec_s[j] = q_j * 2^(g_j - gn'_s)     gn'_s = g[(s+1)*16-1, :]
           KBdec_s[j]= beta_j k_j * 2^(g_j - gn'_s)
           D_s[i]    = 2^(g_i - gn_s)             V_s[i]  = 2^(gn'_s - g_i)
  tail (per s): db_s = rowsum(dk2pre_s * k_s)  [pre-beta, accumulated across
           halves into db_acc]; dk2b_s = dk2pre_s * beta_s;
           dq2_s = D_s*dq2_pre_s + dq;   dk2_s = dk2b_s + dk + dkt_s;
           dg2_s = q_s*dq2_pre_s + (dk2b_s - dkt_s)*k_s + dg   [post-beta dk2,
           upstream L481-489 order]
  host post-processing (NOT part of the kernel, not timed): db = db2.sum(0)+db,
           dg = reverse chunk-local cumsum(dg2).

Ascend toolchain adaptations (all inherited from the fla_wy_fast / fla_chunk_o
verified chains, see DESIGN Sec. 3.5.1; plus two Stage-2 findings):

1. Six JIT kernels (fwd 5 + bwd 1): the C/V combine supports at most one
   straight-line C segment (or one Pipelined body) per kernel; the merged
   inverse chain is therefore split k_inter -> k_g -> k_g2 -> k_y -> k_ai with
   GM fp32 workspaces (Akkbp / AiDiag / G / G2 / Y).
2. fp32 gemm_v0 operands are broken on this toolchain: every GEMM runs with
   hi/lo split operands accumulated in L0C (fwd: bf16 hi/lo 3-gemm; bwd: fp16
   hi/lo 3-gemm, KNOWN LIMITATION |x| > 65504 loses precision).
3. All 18 fwd-inter / 48 bwd gemm_v0 calls are kept in ONE contiguous C run
   (staging first, gemms after, epilogue last) -- no V op may interleave.
4. exp2(x) as exp(x * ln2); bf16<->fp32 casts only UB-internally; L0C(fp32)
   ->GM(bf16) stores stage through UB.
5. [B,S,H] beta loads use the flat-view span + arith_progression +
   reinterpretcast + tile.gather chain (sub-32B strided DMA is unreliable).
6. The bwd db2 scatter stages through a GM scratch with an explicit
   MTE3->MTE2 flag sync (computed-source [64,1] scatters are not kept
   correct otherwise).
7. Loop-carried reduce uses clear=True + T.tile.add accumulation (the
   clear=False in-loop merge is a toolchain bug).
8. bwd grid is NT*B*H with the DK halves iterated INSIDE one program
   (Pipelined(2)); dAqk/dAkk/Mlo/beta are loaded once per (chunk, bh) in the
   prologue and the MdA/MdK hi/lo splits are re-run idempotently inside the
   loop body (prologue V->C exchanges get mis-versioned by the pipeline pass).
   db2 therefore has NK=1 partial (the rowsums of both halves accumulate
   in-program); the DESIGN's [NK,B,S,H] output contract is honored via a
   zero-copy host view.
9. [Stage-2 finding] the @T.prim_func body is parsed by the TVM AST parser,
   NOT traced as plain Python: no Python lists / helper-function calls /
   comprehensions / tuple loops inside the kernel body (they segfault the
   parser).  The s/s2 static unrolls are written out literally and the
   hi/lo split chains are inlined.
10. [Stage-2 finding] every UB->L1 (V->C) copy must use a DEDICATED UB source
   buffer and every L0C->UB (C->V) copy a DEDICATED UB destination buffer:
   ascend_workspace_reduction.cc keys the GM relay workspaces by buffer NAME
   (src_to_workspace_map_ / dst_to_workspace_map_), so reusing one temp across
   multiple cross-core copies collapses their relays and fails the CombineCV
   sync-point pairing ("Mismatch in sync points ... cube has 1, vec has N").
   Pure-V UB<->UB temps may be shared freely.
"""

import tilelang
import torch
from tilelang import language as T

# 2^g is computed as exp(g * ln2): exp2 has no Ascend backend mapping.
LN2 = 0.6931471805599453

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
# fwd kernel 1: off-diagonal Aqk / Akk-beta' GEMMs (BK = DK, straight line)
# ============================================================================
@tilelang.jit(out_idx=[4], pass_configs=PASS_CONFIGS)
def fla_chunk_intra_fwd_k_inter(B, S, H, DK, chunk_size=64, sub_chunk_size=16, scale=1.0):
    BS = chunk_size
    BC = sub_chunk_size
    NT = S // BS
    GRID = NT * B * H
    BK = DK

    @T.prim_func
    def main(
        Q: T.Tensor((B, S, H, DK), "bfloat16"),  # type: ignore
        K: T.Tensor((B, S, H, DK), "bfloat16"),  # type: ignore
        GK: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        BetaFlat: T.Tensor((B, S * H), "bfloat16"),  # type: ignore
        Aqk: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        Akkbp: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % NT
            bbh = cid // NT
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            # ---- persistent UB buffers ----
            k_half = T.alloc_shared([BS, BK], "bfloat16")
            k_f = T.alloc_shared([BS, BK], "float32")
            g_f = T.alloc_shared([BS, BK], "float32")
            beta_tile = T.alloc_shared([(BS - BC) * H], "bfloat16")
            off_i32 = T.alloc_shared([BS - BC], "int32")
            off_u32 = T.alloc_shared([BS - BC], "uint32")
            bt_bf = T.alloc_shared([BS - BC], "bfloat16")
            bet = T.alloc_shared([BS - BC], "float32")
            aqk_st = T.alloc_shared([BS, BS], "bfloat16")
            # ---- shared pure-V transients ----
            q2_half = T.alloc_shared([BC, BK], "bfloat16")
            q2_f = T.alloc_shared([BC, BK], "float32")
            gn = T.alloc_shared([BK], "float32")
            t16 = T.alloc_shared([BC, BK], "float32")
            e16 = T.alloc_shared([BC, BK], "float32")
            t16r = T.alloc_shared([BC, BK], "float32")
            bcol2d = T.alloc_shared([BC, BK], "float32")
            # ---- per-s2 fp32 operand staging (unique names) ----
            qg_f_1 = T.alloc_shared([BC, BK], "float32")
            kg_f_1 = T.alloc_shared([BC, BK], "float32")
            rhs_f_1 = T.alloc_shared([BC, BK], "float32")
            qg_f_2 = T.alloc_shared([BC, BK], "float32")
            kg_f_2 = T.alloc_shared([BC, BK], "float32")
            rhs_f_2 = T.alloc_shared([2 * BC, BK], "float32")
            qg_f_3 = T.alloc_shared([BC, BK], "float32")
            kg_f_3 = T.alloc_shared([BC, BK], "float32")
            rhs_f_3 = T.alloc_shared([3 * BC, BK], "float32")
            # ---- dedicated V->C exchange temps (hi/lo ub per operand) ----
            qg1_hi_ub = T.alloc_shared([BC, BK], "bfloat16")
            qg1_lo_ub = T.alloc_shared([BC, BK], "bfloat16")
            kgp1_hi_ub = T.alloc_shared([BC, BK], "bfloat16")
            kgp1_lo_ub = T.alloc_shared([BC, BK], "bfloat16")
            rhs1_hi_ub = T.alloc_shared([BC, BK], "bfloat16")
            rhs1_lo_ub = T.alloc_shared([BC, BK], "bfloat16")
            qg2_hi_ub = T.alloc_shared([BC, BK], "bfloat16")
            qg2_lo_ub = T.alloc_shared([BC, BK], "bfloat16")
            kgp2_hi_ub = T.alloc_shared([BC, BK], "bfloat16")
            kgp2_lo_ub = T.alloc_shared([BC, BK], "bfloat16")
            rhs2_hi_ub = T.alloc_shared([2 * BC, BK], "bfloat16")
            rhs2_lo_ub = T.alloc_shared([2 * BC, BK], "bfloat16")
            qg3_hi_ub = T.alloc_shared([BC, BK], "bfloat16")
            qg3_lo_ub = T.alloc_shared([BC, BK], "bfloat16")
            kgp3_hi_ub = T.alloc_shared([BC, BK], "bfloat16")
            kgp3_lo_ub = T.alloc_shared([BC, BK], "bfloat16")
            rhs3_hi_ub = T.alloc_shared([3 * BC, BK], "bfloat16")
            rhs3_lo_ub = T.alloc_shared([3 * BC, BK], "bfloat16")
            # ---- shared pure-V split f32 temps ----
            hi_f32 = T.alloc_shared([3 * BC, BK], "float32")
            lo_f32 = T.alloc_shared([3 * BC, BK], "float32")
            # ---- L1 operands ----
            qg1_hi_l1 = T.alloc_shared([BC, BK], "bfloat16")
            qg1_lo_l1 = T.alloc_shared([BC, BK], "bfloat16")
            kgp1_hi_l1 = T.alloc_shared([BC, BK], "bfloat16")
            kgp1_lo_l1 = T.alloc_shared([BC, BK], "bfloat16")
            rhs1_hi_l1 = T.alloc_shared([BC, BK], "bfloat16")
            rhs1_lo_l1 = T.alloc_shared([BC, BK], "bfloat16")
            qg2_hi_l1 = T.alloc_shared([BC, BK], "bfloat16")
            qg2_lo_l1 = T.alloc_shared([BC, BK], "bfloat16")
            kgp2_hi_l1 = T.alloc_shared([BC, BK], "bfloat16")
            kgp2_lo_l1 = T.alloc_shared([BC, BK], "bfloat16")
            rhs2_hi_l1 = T.alloc_shared([2 * BC, BK], "bfloat16")
            rhs2_lo_l1 = T.alloc_shared([2 * BC, BK], "bfloat16")
            qg3_hi_l1 = T.alloc_shared([BC, BK], "bfloat16")
            qg3_lo_l1 = T.alloc_shared([BC, BK], "bfloat16")
            kgp3_hi_l1 = T.alloc_shared([BC, BK], "bfloat16")
            kgp3_lo_l1 = T.alloc_shared([BC, BK], "bfloat16")
            rhs3_hi_l1 = T.alloc_shared([3 * BC, BK], "bfloat16")
            rhs3_lo_l1 = T.alloc_shared([3 * BC, BK], "bfloat16")
            # ---- fragments + epilogue transit (unique per s2) ----
            frag_aqk_1 = T.alloc_fragment([BC, BC], "float32")
            frag_akk_1 = T.alloc_fragment([BC, BC], "float32")
            frag_aqk_2 = T.alloc_fragment([BC, 2 * BC], "float32")
            frag_akk_2 = T.alloc_fragment([BC, 2 * BC], "float32")
            frag_aqk_3 = T.alloc_fragment([BC, 3 * BC], "float32")
            frag_akk_3 = T.alloc_fragment([BC, 3 * BC], "float32")
            aqk1_f32 = T.alloc_shared([BC, BC], "float32")
            aqk1_bf = T.alloc_shared([BC, BC], "bfloat16")
            akk1_f32 = T.alloc_shared([BC, BC], "float32")
            aqk2_f32 = T.alloc_shared([BC, 2 * BC], "float32")
            aqk2_bf = T.alloc_shared([BC, 2 * BC], "bfloat16")
            akk2_f32 = T.alloc_shared([BC, 2 * BC], "float32")
            aqk3_f32 = T.alloc_shared([BC, 3 * BC], "float32")
            aqk3_bf = T.alloc_shared([BC, 3 * BC], "bfloat16")
            akk3_f32 = T.alloc_shared([BC, 3 * BC], "float32")

            # ==== PHASE A (V): loads + per-s2 build & split (dedicated temps) ====
            T.copy(K[bb, s0 : s0 + BS, bh, 0:BK], k_half)
            T.copy(k_half, k_f)
            T.copy(GK[bb, s0 : s0 + BS, bh, 0:BK], g_f)
            # beta gather chain (bf16 variant; rows 16..64 = the kg' row blocks)
            T.copy(BetaFlat[bb, (s0 + BC) * H : (s0 + BS) * H], beta_tile)
            T.tile.arith_progression(off_i32, bh * 2, H * 2)
            T.reinterpretcast(off_u32, off_i32, "uint32_t")
            T.tile.gather(bt_bf, beta_tile, off_u32, 0)
            T.copy(bt_bf, bet)

            # ---- s2 = 1 (rows 16..31, R = 16) ----
            T.copy(Q[bb, s0 + BC : s0 + 2 * BC, bh, 0:BK], q2_half)
            T.copy(q2_half, q2_f)
            T.copy(g_f[BC, 0:BK], gn)
            T.tile.broadcast(t16, gn, axis=0)
            T.tile.sub(e16, g_f[BC : 2 * BC, 0:BK], t16)
            T.tile.mul(e16, e16, LN2)
            T.tile.exp(e16, e16)
            T.tile.mul(qg_f_1, q2_f, e16)
            T.tile.mul(kg_f_1, k_f[BC : 2 * BC, 0:BK], e16)
            T.tile.broadcast(bcol2d, bet[0:BC], axis=1)
            T.tile.mul(kg_f_1, kg_f_1, bcol2d)
            T.tile.mul(kg_f_1, kg_f_1, -1.0)
            # split qg1
            T.copy(qg_f_1, qg1_hi_ub)
            T.copy(qg1_hi_ub, hi_f32[0:BC, 0:BK])
            T.tile.sub(lo_f32[0:BC, 0:BK], qg_f_1, hi_f32[0:BC, 0:BK])
            T.copy(lo_f32[0:BC, 0:BK], qg1_lo_ub)
            T.copy(qg1_hi_ub, qg1_hi_l1)
            T.copy(qg1_lo_ub, qg1_lo_l1)
            # split kgp1
            T.copy(kg_f_1, kgp1_hi_ub)
            T.copy(kgp1_hi_ub, hi_f32[0:BC, 0:BK])
            T.tile.sub(lo_f32[0:BC, 0:BK], kg_f_1, hi_f32[0:BC, 0:BK])
            T.copy(lo_f32[0:BC, 0:BK], kgp1_lo_ub)
            T.copy(kgp1_hi_ub, kgp1_hi_l1)
            T.copy(kgp1_lo_ub, kgp1_lo_l1)
            # build rhs1 (chunk 0) and split
            T.tile.broadcast(t16r, gn, axis=0)
            T.tile.sub(t16r, t16r, g_f[0:BC, 0:BK])
            T.tile.mul(t16r, t16r, LN2)
            T.tile.exp(t16r, t16r)
            T.tile.mul(rhs_f_1, k_f[0:BC, 0:BK], t16r)
            T.copy(rhs_f_1, rhs1_hi_ub)
            T.copy(rhs1_hi_ub, hi_f32[0:BC, 0:BK])
            T.tile.sub(lo_f32[0:BC, 0:BK], rhs_f_1, hi_f32[0:BC, 0:BK])
            T.copy(lo_f32[0:BC, 0:BK], rhs1_lo_ub)
            T.copy(rhs1_hi_ub, rhs1_hi_l1)
            T.copy(rhs1_lo_ub, rhs1_lo_l1)

            # ---- s2 = 2 (rows 32..47, R = 32) ----
            T.copy(Q[bb, s0 + 2 * BC : s0 + 3 * BC, bh, 0:BK], q2_half)
            T.copy(q2_half, q2_f)
            T.copy(g_f[2 * BC, 0:BK], gn)
            T.tile.broadcast(t16, gn, axis=0)
            T.tile.sub(e16, g_f[2 * BC : 3 * BC, 0:BK], t16)
            T.tile.mul(e16, e16, LN2)
            T.tile.exp(e16, e16)
            T.tile.mul(qg_f_2, q2_f, e16)
            T.tile.mul(kg_f_2, k_f[2 * BC : 3 * BC, 0:BK], e16)
            T.tile.broadcast(bcol2d, bet[BC : 2 * BC], axis=1)
            T.tile.mul(kg_f_2, kg_f_2, bcol2d)
            T.tile.mul(kg_f_2, kg_f_2, -1.0)
            # split qg2
            T.copy(qg_f_2, qg2_hi_ub)
            T.copy(qg2_hi_ub, hi_f32[0:BC, 0:BK])
            T.tile.sub(lo_f32[0:BC, 0:BK], qg_f_2, hi_f32[0:BC, 0:BK])
            T.copy(lo_f32[0:BC, 0:BK], qg2_lo_ub)
            T.copy(qg2_hi_ub, qg2_hi_l1)
            T.copy(qg2_lo_ub, qg2_lo_l1)
            # split kgp2
            T.copy(kg_f_2, kgp2_hi_ub)
            T.copy(kgp2_hi_ub, hi_f32[0:BC, 0:BK])
            T.tile.sub(lo_f32[0:BC, 0:BK], kg_f_2, hi_f32[0:BC, 0:BK])
            T.copy(lo_f32[0:BC, 0:BK], kgp2_lo_ub)
            T.copy(kgp2_hi_ub, kgp2_hi_l1)
            T.copy(kgp2_lo_ub, kgp2_lo_l1)
            # build rhs2 (chunks 0,1) and split
            T.tile.broadcast(t16r, gn, axis=0)
            T.tile.sub(t16r, t16r, g_f[0:BC, 0:BK])
            T.tile.mul(t16r, t16r, LN2)
            T.tile.exp(t16r, t16r)
            T.tile.mul(rhs_f_2[0:BC, 0:BK], k_f[0:BC, 0:BK], t16r)
            T.tile.broadcast(t16r, gn, axis=0)
            T.tile.sub(t16r, t16r, g_f[BC : 2 * BC, 0:BK])
            T.tile.mul(t16r, t16r, LN2)
            T.tile.exp(t16r, t16r)
            T.tile.mul(rhs_f_2[BC : 2 * BC, 0:BK], k_f[BC : 2 * BC, 0:BK], t16r)
            T.copy(rhs_f_2, rhs2_hi_ub)
            T.copy(rhs2_hi_ub, hi_f32[0 : 2 * BC, 0:BK])
            T.tile.sub(lo_f32[0 : 2 * BC, 0:BK], rhs_f_2, hi_f32[0 : 2 * BC, 0:BK])
            T.copy(lo_f32[0 : 2 * BC, 0:BK], rhs2_lo_ub)
            T.copy(rhs2_hi_ub, rhs2_hi_l1)
            T.copy(rhs2_lo_ub, rhs2_lo_l1)

            # ---- s2 = 3 (rows 48..63, R = 48) ----
            T.copy(Q[bb, s0 + 3 * BC : s0 + 4 * BC, bh, 0:BK], q2_half)
            T.copy(q2_half, q2_f)
            T.copy(g_f[3 * BC, 0:BK], gn)
            T.tile.broadcast(t16, gn, axis=0)
            T.tile.sub(e16, g_f[3 * BC : 4 * BC, 0:BK], t16)
            T.tile.mul(e16, e16, LN2)
            T.tile.exp(e16, e16)
            T.tile.mul(qg_f_3, q2_f, e16)
            T.tile.mul(kg_f_3, k_f[3 * BC : 4 * BC, 0:BK], e16)
            T.tile.broadcast(bcol2d, bet[2 * BC : 3 * BC], axis=1)
            T.tile.mul(kg_f_3, kg_f_3, bcol2d)
            T.tile.mul(kg_f_3, kg_f_3, -1.0)
            # split qg3
            T.copy(qg_f_3, qg3_hi_ub)
            T.copy(qg3_hi_ub, hi_f32[0:BC, 0:BK])
            T.tile.sub(lo_f32[0:BC, 0:BK], qg_f_3, hi_f32[0:BC, 0:BK])
            T.copy(lo_f32[0:BC, 0:BK], qg3_lo_ub)
            T.copy(qg3_hi_ub, qg3_hi_l1)
            T.copy(qg3_lo_ub, qg3_lo_l1)
            # split kgp3
            T.copy(kg_f_3, kgp3_hi_ub)
            T.copy(kgp3_hi_ub, hi_f32[0:BC, 0:BK])
            T.tile.sub(lo_f32[0:BC, 0:BK], kg_f_3, hi_f32[0:BC, 0:BK])
            T.copy(lo_f32[0:BC, 0:BK], kgp3_lo_ub)
            T.copy(kgp3_hi_ub, kgp3_hi_l1)
            T.copy(kgp3_lo_ub, kgp3_lo_l1)
            # build rhs3 (chunks 0,1,2) and split
            T.tile.broadcast(t16r, gn, axis=0)
            T.tile.sub(t16r, t16r, g_f[0:BC, 0:BK])
            T.tile.mul(t16r, t16r, LN2)
            T.tile.exp(t16r, t16r)
            T.tile.mul(rhs_f_3[0:BC, 0:BK], k_f[0:BC, 0:BK], t16r)
            T.tile.broadcast(t16r, gn, axis=0)
            T.tile.sub(t16r, t16r, g_f[BC : 2 * BC, 0:BK])
            T.tile.mul(t16r, t16r, LN2)
            T.tile.exp(t16r, t16r)
            T.tile.mul(rhs_f_3[BC : 2 * BC, 0:BK], k_f[BC : 2 * BC, 0:BK], t16r)
            T.tile.broadcast(t16r, gn, axis=0)
            T.tile.sub(t16r, t16r, g_f[2 * BC : 3 * BC, 0:BK])
            T.tile.mul(t16r, t16r, LN2)
            T.tile.exp(t16r, t16r)
            T.tile.mul(rhs_f_3[2 * BC : 3 * BC, 0:BK], k_f[2 * BC : 3 * BC, 0:BK], t16r)
            T.copy(rhs_f_3, rhs3_hi_ub)
            T.copy(rhs3_hi_ub, hi_f32[0 : 3 * BC, 0:BK])
            T.tile.sub(lo_f32[0 : 3 * BC, 0:BK], rhs_f_3, hi_f32[0 : 3 * BC, 0:BK])
            T.copy(lo_f32[0 : 3 * BC, 0:BK], rhs3_lo_ub)
            T.copy(rhs3_hi_ub, rhs3_hi_l1)
            T.copy(rhs3_lo_ub, rhs3_lo_l1)

            # ==== PHASE B (C): 18 contiguous gemm_v0 (hi/lo 3-gemm each) ====
            T.gemm_v0(qg1_hi_l1, rhs1_hi_l1, frag_aqk_1, transpose_B=True, init=True)
            T.gemm_v0(qg1_lo_l1, rhs1_hi_l1, frag_aqk_1, transpose_B=True, init=False)
            T.gemm_v0(qg1_hi_l1, rhs1_lo_l1, frag_aqk_1, transpose_B=True, init=False)
            T.gemm_v0(kgp1_hi_l1, rhs1_hi_l1, frag_akk_1, transpose_B=True, init=True)
            T.gemm_v0(kgp1_lo_l1, rhs1_hi_l1, frag_akk_1, transpose_B=True, init=False)
            T.gemm_v0(kgp1_hi_l1, rhs1_lo_l1, frag_akk_1, transpose_B=True, init=False)
            T.gemm_v0(qg2_hi_l1, rhs2_hi_l1, frag_aqk_2, transpose_B=True, init=True)
            T.gemm_v0(qg2_lo_l1, rhs2_hi_l1, frag_aqk_2, transpose_B=True, init=False)
            T.gemm_v0(qg2_hi_l1, rhs2_lo_l1, frag_aqk_2, transpose_B=True, init=False)
            T.gemm_v0(kgp2_hi_l1, rhs2_hi_l1, frag_akk_2, transpose_B=True, init=True)
            T.gemm_v0(kgp2_lo_l1, rhs2_hi_l1, frag_akk_2, transpose_B=True, init=False)
            T.gemm_v0(kgp2_hi_l1, rhs2_lo_l1, frag_akk_2, transpose_B=True, init=False)
            T.gemm_v0(qg3_hi_l1, rhs3_hi_l1, frag_aqk_3, transpose_B=True, init=True)
            T.gemm_v0(qg3_lo_l1, rhs3_hi_l1, frag_aqk_3, transpose_B=True, init=False)
            T.gemm_v0(qg3_hi_l1, rhs3_lo_l1, frag_aqk_3, transpose_B=True, init=False)
            T.gemm_v0(kgp3_hi_l1, rhs3_hi_l1, frag_akk_3, transpose_B=True, init=True)
            T.gemm_v0(kgp3_lo_l1, rhs3_hi_l1, frag_akk_3, transpose_B=True, init=False)
            T.gemm_v0(kgp3_hi_l1, rhs3_lo_l1, frag_akk_3, transpose_B=True, init=False)

            # ==== PHASE C (V): Aqk zero-filled staging write + Akkbp ws ====
            T.tile.fill(aqk_st, 0.0)
            T.copy(frag_aqk_1, aqk1_f32)
            T.tile.mul(aqk1_f32, aqk1_f32, scale)
            T.copy(aqk1_f32, aqk1_bf)
            T.copy(aqk1_bf, aqk_st[BC : 2 * BC, 0:BC])
            T.copy(frag_akk_1, akk1_f32)
            T.copy(akk1_f32, Akkbp[bb, s0 + BC : s0 + 2 * BC, bh, 0:BC])
            T.copy(frag_aqk_2, aqk2_f32)
            T.tile.mul(aqk2_f32, aqk2_f32, scale)
            T.copy(aqk2_f32, aqk2_bf)
            T.copy(aqk2_bf, aqk_st[2 * BC : 3 * BC, 0 : 2 * BC])
            T.copy(frag_akk_2, akk2_f32)
            T.copy(akk2_f32, Akkbp[bb, s0 + 2 * BC : s0 + 3 * BC, bh, 0 : 2 * BC])
            T.copy(frag_aqk_3, aqk3_f32)
            T.tile.mul(aqk3_f32, aqk3_f32, scale)
            T.copy(aqk3_f32, aqk3_bf)
            T.copy(aqk3_bf, aqk_st[3 * BC : 4 * BC, 0 : 3 * BC])
            T.copy(frag_akk_3, akk3_f32)
            T.copy(akk3_f32, Akkbp[bb, s0 + 3 * BC : s0 + 4 * BC, bh, 0 : 3 * BC])
            T.copy(aqk_st, Aqk[bb, s0 : s0 + BS, bh, 0:BS])

    return main


# ============================================================================
# fwd kernel 2: forward substitution on the 4 diagonals + G = AiDiag @ Akkbp
# ============================================================================
@tilelang.jit(pass_configs=PASS_CONFIGS)
def fla_chunk_intra_fwd_k_g(B, S, H, DK, chunk_size=64, sub_chunk_size=16):
    BS = chunk_size
    BC = sub_chunk_size
    NT = S // BS
    GRID = NT * B * H

    @T.prim_func
    def main(
        Akkbp: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
        Akk_diag: T.Tensor((B, S, H, BC), "float32"),  # type: ignore
        Mneg: T.Tensor((BC, BC), "float32"),  # type: ignore
        Meye: T.Tensor((BC, BC), "float32"),  # type: ignore
        Aidws: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
        Gws: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
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
            akbp_st = T.alloc_shared([BS, BS], "float32")
            # dedicated exchange temps (aid / akbp separate)
            aid_hi_ub = T.alloc_shared([BS, BS], "bfloat16")
            aid_lo_ub = T.alloc_shared([BS, BS], "bfloat16")
            akbp_hi_ub = T.alloc_shared([BS, BS], "bfloat16")
            akbp_lo_ub = T.alloc_shared([BS, BS], "bfloat16")
            hi_f32 = T.alloc_shared([BS, BS], "float32")
            lo_f32 = T.alloc_shared([BS, BS], "float32")
            aid_hi_l1 = T.alloc_shared([BS, BS], "bfloat16")
            aid_lo_l1 = T.alloc_shared([BS, BS], "bfloat16")
            akbp_hi_l1 = T.alloc_shared([BS, BS], "bfloat16")
            akbp_lo_l1 = T.alloc_shared([BS, BS], "bfloat16")
            g_frag = T.alloc_fragment([BS, BS], "float32")
            g_ub = T.alloc_shared([BS, BS], "float32")
            # packed intermediates: UB->GM / GM->UB copies must never read or
            # write a 2D UB REGION whose column extent is narrower than the
            # buffer (the codegen packs region rows -> mangled data).
            # UB<->UB region copies are safe.
            g1_blk = T.alloc_shared([BC, BC], "float32")
            g2_blk = T.alloc_shared([BC, 2 * BC], "float32")
            g3_blk = T.alloc_shared([BC, 3 * BC], "float32")
            ai_s0 = T.alloc_shared([BC, BC], "float32")
            ai_s1 = T.alloc_shared([BC, BC], "float32")
            ai_s2 = T.alloc_shared([BC, BC], "float32")
            ai_s3 = T.alloc_shared([BC, BC], "float32")

            # ==== PHASE A (V): 4 forward substitutions + staging + splits ====
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
            # AiDiag staging: block-diag(I + loop result).  tile.add cannot
            # target a non-contiguous block region, so the +I runs on the
            # [BC,BC] scale (whole-buffer add) and the result is placed with
            # row-gap region COPIES.
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
            # Akkbp staging: zero fill + the 3 written slices (via packed
            # intermediates; GM->UB narrow-column region writes are unsafe)
            T.tile.fill(akbp_st, 0.0)
            T.copy(Akkbp[bb, s0 + BC : s0 + 2 * BC, bh, 0:BC], g1_blk)
            T.copy(g1_blk, akbp_st[BC : 2 * BC, 0:BC])
            T.copy(Akkbp[bb, s0 + 2 * BC : s0 + 3 * BC, bh, 0 : 2 * BC], g2_blk)
            T.copy(g2_blk, akbp_st[2 * BC : 3 * BC, 0 : 2 * BC])
            T.copy(Akkbp[bb, s0 + 3 * BC : s0 + 4 * BC, bh, 0 : 3 * BC], g3_blk)
            T.copy(g3_blk, akbp_st[3 * BC : 4 * BC, 0 : 3 * BC])
            # splits (dedicated exchange temps)
            T.copy(aid_st, aid_hi_ub)
            T.copy(aid_hi_ub, hi_f32)
            T.tile.sub(lo_f32, aid_st, hi_f32)
            T.copy(lo_f32, aid_lo_ub)
            T.copy(aid_hi_ub, aid_hi_l1)
            T.copy(aid_lo_ub, aid_lo_l1)
            T.copy(akbp_st, akbp_hi_ub)
            T.copy(akbp_hi_ub, hi_f32)
            T.tile.sub(lo_f32, akbp_st, hi_f32)
            T.copy(lo_f32, akbp_lo_ub)
            T.copy(akbp_hi_ub, akbp_hi_l1)
            T.copy(akbp_lo_ub, akbp_lo_l1)

            # ==== PHASE B (C): G = AiDiag @ Akkbp ====
            T.gemm_v0(aid_hi_l1, akbp_hi_l1, g_frag, init=True)
            T.gemm_v0(aid_lo_l1, akbp_hi_l1, g_frag, init=False)
            T.gemm_v0(aid_hi_l1, akbp_lo_l1, g_frag, init=False)

            # ==== PHASE C (V): G ws (3 slices) + AiDiag ws (4 diag slices) ====
            T.copy(g_frag, g_ub)
            T.copy(g_ub[BC : 2 * BC, 0:BC], g1_blk)
            T.copy(g1_blk, Gws[bb, s0 + BC : s0 + 2 * BC, bh, 0:BC])
            T.copy(g_ub[2 * BC : 3 * BC, 0 : 2 * BC], g2_blk)
            T.copy(g2_blk, Gws[bb, s0 + 2 * BC : s0 + 3 * BC, bh, 0 : 2 * BC])
            T.copy(g_ub[3 * BC : 4 * BC, 0 : 3 * BC], g3_blk)
            T.copy(g3_blk, Gws[bb, s0 + 3 * BC : s0 + 4 * BC, bh, 0 : 3 * BC])
            T.copy(meye, ad_s)
            T.tile.add(ad_s, ad_s, ai_s0)
            T.copy(ad_s, Aidws[bb, s0 : s0 + BC, bh, 0:BC])
            T.copy(meye, ad_s)
            T.tile.add(ad_s, ad_s, ai_s1)
            T.copy(ad_s, Aidws[bb, s0 + BC : s0 + 2 * BC, bh, BC : 2 * BC])
            T.copy(meye, ad_s)
            T.tile.add(ad_s, ad_s, ai_s2)
            T.copy(ad_s, Aidws[bb, s0 + 2 * BC : s0 + 3 * BC, bh, 2 * BC : 3 * BC])
            T.copy(meye, ad_s)
            T.tile.add(ad_s, ad_s, ai_s3)
            T.copy(ad_s, Aidws[bb, s0 + 3 * BC : s0 + 4 * BC, bh, 3 * BC : 4 * BC])

    return main


# ============================================================================
# fwd kernel 3: G2 = G @ G  (A/B staged with dedicated exchange temps)
# ============================================================================
@tilelang.jit(pass_configs=PASS_CONFIGS)
def fla_chunk_intra_fwd_k_g2(B, S, H, DK, chunk_size=64, sub_chunk_size=16):
    BS = chunk_size
    BC = sub_chunk_size
    NT = S // BS
    GRID = NT * B * H

    @T.prim_func
    def main(
        Gws: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
        G2ws: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % NT
            bbh = cid // NT
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            gst = T.alloc_shared([BS, BS], "float32")
            g1_blk = T.alloc_shared([BC, BC], "float32")
            g2_blk = T.alloc_shared([BC, 2 * BC], "float32")
            g3_blk = T.alloc_shared([BC, 3 * BC], "float32")
            g2a_blk = T.alloc_shared([BC, BC], "float32")
            g2b_blk = T.alloc_shared([BC, 2 * BC], "float32")
            a_hi_ub = T.alloc_shared([BS, BS], "bfloat16")
            a_lo_ub = T.alloc_shared([BS, BS], "bfloat16")
            b_hi_ub = T.alloc_shared([BS, BS], "bfloat16")
            b_lo_ub = T.alloc_shared([BS, BS], "bfloat16")
            hi_f32 = T.alloc_shared([BS, BS], "float32")
            lo_f32 = T.alloc_shared([BS, BS], "float32")
            a_hi_l1 = T.alloc_shared([BS, BS], "bfloat16")
            a_lo_l1 = T.alloc_shared([BS, BS], "bfloat16")
            b_hi_l1 = T.alloc_shared([BS, BS], "bfloat16")
            b_lo_l1 = T.alloc_shared([BS, BS], "bfloat16")
            g2_frag = T.alloc_fragment([BS, BS], "float32")
            g2_ub = T.alloc_shared([BS, BS], "float32")

            # PHASE A: stage G (zero fill + the 3 written slices), split twice.
            # GM->UB reads go through packed whole-buffer intermediates.
            T.tile.fill(gst, 0.0)
            T.copy(Gws[bb, s0 + BC : s0 + 2 * BC, bh, 0:BC], g1_blk)
            T.copy(g1_blk, gst[BC : 2 * BC, 0:BC])
            T.copy(Gws[bb, s0 + 2 * BC : s0 + 3 * BC, bh, 0 : 2 * BC], g2_blk)
            T.copy(g2_blk, gst[2 * BC : 3 * BC, 0 : 2 * BC])
            T.copy(Gws[bb, s0 + 3 * BC : s0 + 4 * BC, bh, 0 : 3 * BC], g3_blk)
            T.copy(g3_blk, gst[3 * BC : 4 * BC, 0 : 3 * BC])
            T.copy(gst, a_hi_ub)
            T.copy(a_hi_ub, hi_f32)
            T.tile.sub(lo_f32, gst, hi_f32)
            T.copy(lo_f32, a_lo_ub)
            T.copy(a_hi_ub, a_hi_l1)
            T.copy(a_lo_ub, a_lo_l1)
            T.copy(gst, b_hi_ub)
            T.copy(b_hi_ub, hi_f32)
            T.tile.sub(lo_f32, gst, hi_f32)
            T.copy(lo_f32, b_lo_ub)
            T.copy(b_hi_ub, b_hi_l1)
            T.copy(b_lo_ub, b_lo_l1)

            # PHASE B: G2 = G @ G
            T.gemm_v0(a_hi_l1, b_hi_l1, g2_frag, init=True)
            T.gemm_v0(a_lo_l1, b_hi_l1, g2_frag, init=False)
            T.gemm_v0(a_hi_l1, b_lo_l1, g2_frag, init=False)

            # PHASE C: G2 is strictly block-lower with distance >= 2:
            # blocks (2,0), (3,0), (3,1) -> packed intermediates then GM
            T.copy(g2_frag, g2_ub)
            T.copy(g2_ub[2 * BC : 3 * BC, 0:BC], g2a_blk)
            T.copy(g2a_blk, G2ws[bb, s0 + 2 * BC : s0 + 3 * BC, bh, 0:BC])
            T.copy(g2_ub[3 * BC : 4 * BC, 0 : 2 * BC], g2b_blk)
            T.copy(g2b_blk, G2ws[bb, s0 + 3 * BC : s0 + 4 * BC, bh, 0 : 2 * BC])

    return main


# ============================================================================
# fwd kernel 4: Y = (I + G2) @ AiDiag
# ============================================================================
@tilelang.jit(pass_configs=PASS_CONFIGS)
def fla_chunk_intra_fwd_k_y(B, S, H, DK, chunk_size=64, sub_chunk_size=16):
    BS = chunk_size
    BC = sub_chunk_size
    NT = S // BS
    GRID = NT * B * H

    @T.prim_func
    def main(
        G2ws: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
        Aidws: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
        Meye: T.Tensor((BS, BS), "float32"),  # type: ignore
        Yws: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % NT
            bbh = cid // NT
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            ig2 = T.alloc_shared([BS, BS], "float32")
            aid_st = T.alloc_shared([BS, BS], "float32")
            meye = T.alloc_shared([BS, BS], "float32")
            # packed intermediates (see k_g2 note)
            aid_blk = T.alloc_shared([BC, BC], "float32")
            g2a_blk = T.alloc_shared([BC, BC], "float32")
            g2b_blk = T.alloc_shared([BC, 2 * BC], "float32")
            ig2_hi_ub = T.alloc_shared([BS, BS], "bfloat16")
            ig2_lo_ub = T.alloc_shared([BS, BS], "bfloat16")
            aid_hi_ub = T.alloc_shared([BS, BS], "bfloat16")
            aid_lo_ub = T.alloc_shared([BS, BS], "bfloat16")
            hi_f32 = T.alloc_shared([BS, BS], "float32")
            lo_f32 = T.alloc_shared([BS, BS], "float32")
            ig2_hi_l1 = T.alloc_shared([BS, BS], "bfloat16")
            ig2_lo_l1 = T.alloc_shared([BS, BS], "bfloat16")
            aid_hi_l1 = T.alloc_shared([BS, BS], "bfloat16")
            aid_lo_l1 = T.alloc_shared([BS, BS], "bfloat16")
            y_frag = T.alloc_fragment([BS, BS], "float32")
            y_ub = T.alloc_shared([BS, BS], "float32")

            # PHASE A: I + G2 (Meye64 + the 2 G2 slices via packed blocks;
            # G2's diagonal blocks are 0) and AiDiag (4 packed block reads)
            T.copy(Meye, meye)
            T.copy(meye, ig2)
            T.copy(G2ws[bb, s0 + 2 * BC : s0 + 3 * BC, bh, 0:BC], g2a_blk)
            T.copy(g2a_blk, ig2[2 * BC : 3 * BC, 0:BC])
            T.copy(G2ws[bb, s0 + 3 * BC : s0 + 4 * BC, bh, 0 : 2 * BC], g2b_blk)
            T.copy(g2b_blk, ig2[3 * BC : 4 * BC, 0 : 2 * BC])
            T.tile.fill(aid_st, 0.0)
            T.copy(Aidws[bb, s0 : s0 + BC, bh, 0:BC], aid_blk)
            T.copy(aid_blk, aid_st[0:BC, 0:BC])
            T.copy(Aidws[bb, s0 + BC : s0 + 2 * BC, bh, BC : 2 * BC], aid_blk)
            T.copy(aid_blk, aid_st[BC : 2 * BC, BC : 2 * BC])
            T.copy(Aidws[bb, s0 + 2 * BC : s0 + 3 * BC, bh, 2 * BC : 3 * BC], aid_blk)
            T.copy(aid_blk, aid_st[2 * BC : 3 * BC, 2 * BC : 3 * BC])
            T.copy(Aidws[bb, s0 + 3 * BC : s0 + 4 * BC, bh, 3 * BC : 4 * BC], aid_blk)
            T.copy(aid_blk, aid_st[3 * BC : 4 * BC, 3 * BC : 4 * BC])
            T.copy(ig2, ig2_hi_ub)
            T.copy(ig2_hi_ub, hi_f32)
            T.tile.sub(lo_f32, ig2, hi_f32)
            T.copy(lo_f32, ig2_lo_ub)
            T.copy(ig2_hi_ub, ig2_hi_l1)
            T.copy(ig2_lo_ub, ig2_lo_l1)
            T.copy(aid_st, aid_hi_ub)
            T.copy(aid_hi_ub, hi_f32)
            T.tile.sub(lo_f32, aid_st, hi_f32)
            T.copy(lo_f32, aid_lo_ub)
            T.copy(aid_hi_ub, aid_hi_l1)
            T.copy(aid_lo_ub, aid_lo_l1)

            # PHASE B: Y = (I + G2) @ AiDiag
            T.gemm_v0(ig2_hi_l1, aid_hi_l1, y_frag, init=True)
            T.gemm_v0(ig2_lo_l1, aid_hi_l1, y_frag, init=False)
            T.gemm_v0(ig2_hi_l1, aid_lo_l1, y_frag, init=False)

            # PHASE C: full-block Y ws write
            T.copy(y_frag, y_ub)
            T.copy(y_ub, Yws[bb, s0 : s0 + BS, bh, 0:BS])

    return main


# ============================================================================
# fwd kernel 5: Ai = (I + G) @ Y  -> bf16 Akk output (single full write)
# ============================================================================
@tilelang.jit(out_idx=[3], pass_configs=PASS_CONFIGS)
def fla_chunk_intra_fwd_k_ai(B, S, H, DK, chunk_size=64, sub_chunk_size=16):
    BS = chunk_size
    BC = sub_chunk_size
    NT = S // BS
    GRID = NT * B * H

    @T.prim_func
    def main(
        Gws: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
        Yws: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
        Meye: T.Tensor((BS, BS), "float32"),  # type: ignore
        Akk: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % NT
            bbh = cid // NT
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            ig = T.alloc_shared([BS, BS], "float32")
            y_ub = T.alloc_shared([BS, BS], "float32")
            meye = T.alloc_shared([BS, BS], "float32")
            g1_blk = T.alloc_shared([BC, BC], "float32")
            g2_blk = T.alloc_shared([BC, 2 * BC], "float32")
            g3_blk = T.alloc_shared([BC, 3 * BC], "float32")
            ig_hi_ub = T.alloc_shared([BS, BS], "bfloat16")
            ig_lo_ub = T.alloc_shared([BS, BS], "bfloat16")
            y_hi_ub = T.alloc_shared([BS, BS], "bfloat16")
            y_lo_ub = T.alloc_shared([BS, BS], "bfloat16")
            hi_f32 = T.alloc_shared([BS, BS], "float32")
            lo_f32 = T.alloc_shared([BS, BS], "float32")
            ig_hi_l1 = T.alloc_shared([BS, BS], "bfloat16")
            ig_lo_l1 = T.alloc_shared([BS, BS], "bfloat16")
            y_hi_l1 = T.alloc_shared([BS, BS], "bfloat16")
            y_lo_l1 = T.alloc_shared([BS, BS], "bfloat16")
            ai_frag = T.alloc_fragment([BS, BS], "float32")
            ai_f32 = T.alloc_shared([BS, BS], "float32")
            ai_bf = T.alloc_shared([BS, BS], "bfloat16")

            # PHASE A: I + G (Meye64 + the 3 G slices via packed blocks) and
            # the full Y
            T.copy(Meye, meye)
            T.copy(meye, ig)
            T.copy(Gws[bb, s0 + BC : s0 + 2 * BC, bh, 0:BC], g1_blk)
            T.copy(g1_blk, ig[BC : 2 * BC, 0:BC])
            T.copy(Gws[bb, s0 + 2 * BC : s0 + 3 * BC, bh, 0 : 2 * BC], g2_blk)
            T.copy(g2_blk, ig[2 * BC : 3 * BC, 0 : 2 * BC])
            T.copy(Gws[bb, s0 + 3 * BC : s0 + 4 * BC, bh, 0 : 3 * BC], g3_blk)
            T.copy(g3_blk, ig[3 * BC : 4 * BC, 0 : 3 * BC])
            T.copy(Yws[bb, s0 : s0 + BS, bh, 0:BS], y_ub)
            T.copy(ig, ig_hi_ub)
            T.copy(ig_hi_ub, hi_f32)
            T.tile.sub(lo_f32, ig, hi_f32)
            T.copy(lo_f32, ig_lo_ub)
            T.copy(ig_hi_ub, ig_hi_l1)
            T.copy(ig_lo_ub, ig_lo_l1)
            T.copy(y_ub, y_hi_ub)
            T.copy(y_hi_ub, hi_f32)
            T.tile.sub(lo_f32, y_ub, hi_f32)
            T.copy(lo_f32, y_lo_ub)
            T.copy(y_hi_ub, y_hi_l1)
            T.copy(y_lo_ub, y_lo_l1)

            # PHASE B: Ai = (I + G) @ Y
            T.gemm_v0(ig_hi_l1, y_hi_l1, ai_frag, init=True)
            T.gemm_v0(ig_lo_l1, y_hi_l1, ai_frag, init=False)
            T.gemm_v0(ig_hi_l1, y_lo_l1, ai_frag, init=False)

            # PHASE C: block-upper-tri is computed-exact 0 (zero operands),
            # single full bf16 write
            T.copy(ai_frag, ai_f32)
            T.copy(ai_f32, ai_bf)
            T.copy(ai_bf, Akk[bb, s0 : s0 + BS, bh, 0:BS])

    return main


# ============================================================================
# bwd kernel: dq2 / dk2 / db2 / dg2 (one Pipelined(2) body over DK halves)
# ============================================================================
@tilelang.jit(out_idx=[12, 13, 14, 15], pass_configs=PASS_CONFIGS)
def fla_chunk_intra_bwd_k_intra(B, S, H, DK, chunk_size=64, sub_chunk_size=16):
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
        dq: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        dk: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        dg: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        db2flat: T.Tensor((GRID, BS), "float32"),  # type: ignore
        db2flat_v: T.Tensor((GRID, BS, 1), "float32"),  # type: ignore
        dq2: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        dk2: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        db2: T.Tensor((B, S, H), "float32"),  # type: ignore
        dg2: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % NT
            bbh = cid // NT
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            # ---- prologue (pure V/MTE; no V->C exchange) ----
            # MdA/MdK are computed IN PLACE in the load buffers (UB budget).
            mlo_ub = T.alloc_shared([BS, BS], "float32")
            mda_f = T.alloc_shared([BS, BS], "float32")
            mdk_f = T.alloc_shared([BS, BS], "float32")
            beta_tile = T.alloc_shared([BS * H], "float32")
            off_i32 = T.alloc_shared([BS], "int32")
            off_u32 = T.alloc_shared([BS], "uint32")
            bet = T.alloc_shared([BS], "float32")
            db_acc = T.alloc_shared([BS], "float32")
            db2d = T.alloc_shared([BS, 1], "float32")

            T.copy(Mlo, mlo_ub)
            T.copy(dAqk[bb, s0 : s0 + BS, bh, 0:BS], mda_f)
            T.copy(dAkk[bb, s0 : s0 + BS, bh, 0:BS], mdk_f)
            T.tile.mul(mda_f, mda_f, mlo_ub)
            T.tile.mul(mdk_f, mdk_f, mlo_ub)
            # beta gather chain (fp32 variant)
            T.copy(BetaFlat[bb, s0 * H : (s0 + BS) * H], beta_tile)
            T.tile.arith_progression(off_i32, bh * 4, H * 4)
            T.reinterpretcast(off_u32, off_i32, "uint32_t")
            T.tile.gather(bet, beta_tile, off_u32, 0)
            T.tile.fill(db_acc, 0.0)

            # ---- loop-body buffers (UB budget: one shared fp32 staging, a
            # [BC,BK] decay scratch, per-operand dedicated fp16 exchange
            # buffers -- the workspace relay keys on the UB source name) ----
            k_ub = T.alloc_shared([BS, BK], "float32")
            g_ub = T.alloc_shared([BS, BK], "float32")
            q_ub = T.alloc_shared([BS, BK], "float32")
            t16 = T.alloc_shared([BC, BK], "float32")
            bcol2d = T.alloc_shared([BC, BK], "float32")
            odec_f = T.alloc_shared([BS, BK], "float32")
            # dedicated exchange temps (fp16; unique per UB->L1 copy)
            mda_hi_ub = T.alloc_shared([BS, BS], "float16")
            mda_lo_ub = T.alloc_shared([BS, BS], "float16")
            mdk_hi_ub = T.alloc_shared([BS, BS], "float16")
            mdk_lo_ub = T.alloc_shared([BS, BS], "float16")
            s64_hi_f32 = T.alloc_shared([BS, BS], "float32")
            s64_lo_f32 = T.alloc_shared([BS, BS], "float32")
            sbk_hi_f32 = T.alloc_shared([BS, BK], "float32")
            sbk_lo_f32 = T.alloc_shared([BS, BK], "float32")
            # L1 gemm operands (all live across the single C run)
            mda_hi_l1 = T.alloc_shared([BS, BS], "float16")
            mda_lo_l1 = T.alloc_shared([BS, BS], "float16")
            mdk_hi_l1 = T.alloc_shared([BS, BS], "float16")
            mdk_lo_l1 = T.alloc_shared([BS, BS], "float16")
            # per-s anchors
            gn_s0 = T.alloc_shared([BK], "float32")
            gnp_s0 = T.alloc_shared([BK], "float32")
            gn_s1 = T.alloc_shared([BK], "float32")
            gnp_s1 = T.alloc_shared([BK], "float32")
            gn_s2 = T.alloc_shared([BK], "float32")
            gnp_s2 = T.alloc_shared([BK], "float32")
            gn_s3 = T.alloc_shared([BK], "float32")
            gnp_s3 = T.alloc_shared([BK], "float32")
            # per-s exchange temps (12 x hi/lo)
            kdec0_hi_ub = T.alloc_shared([BS, BK], "float16")
            kdec0_lo_ub = T.alloc_shared([BS, BK], "float16")
            qdec0_hi_ub = T.alloc_shared([BS, BK], "float16")
            qdec0_lo_ub = T.alloc_shared([BS, BK], "float16")
            kbdec0_hi_ub = T.alloc_shared([BS, BK], "float16")
            kbdec0_lo_ub = T.alloc_shared([BS, BK], "float16")
            kdec1_hi_ub = T.alloc_shared([BS, BK], "float16")
            kdec1_lo_ub = T.alloc_shared([BS, BK], "float16")
            qdec1_hi_ub = T.alloc_shared([BS, BK], "float16")
            qdec1_lo_ub = T.alloc_shared([BS, BK], "float16")
            kbdec1_hi_ub = T.alloc_shared([BS, BK], "float16")
            kbdec1_lo_ub = T.alloc_shared([BS, BK], "float16")
            kdec2_hi_ub = T.alloc_shared([BS, BK], "float16")
            kdec2_lo_ub = T.alloc_shared([BS, BK], "float16")
            qdec2_hi_ub = T.alloc_shared([BS, BK], "float16")
            qdec2_lo_ub = T.alloc_shared([BS, BK], "float16")
            kbdec2_hi_ub = T.alloc_shared([BS, BK], "float16")
            kbdec2_lo_ub = T.alloc_shared([BS, BK], "float16")
            kdec3_hi_ub = T.alloc_shared([BS, BK], "float16")
            kdec3_lo_ub = T.alloc_shared([BS, BK], "float16")
            qdec3_hi_ub = T.alloc_shared([BS, BK], "float16")
            qdec3_lo_ub = T.alloc_shared([BS, BK], "float16")
            kbdec3_hi_ub = T.alloc_shared([BS, BK], "float16")
            kbdec3_lo_ub = T.alloc_shared([BS, BK], "float16")
            # per-s L1 operands
            kdec0_hi_l1 = T.alloc_shared([BS, BK], "float16")
            kdec0_lo_l1 = T.alloc_shared([BS, BK], "float16")
            qdec0_hi_l1 = T.alloc_shared([BS, BK], "float16")
            qdec0_lo_l1 = T.alloc_shared([BS, BK], "float16")
            kbdec0_hi_l1 = T.alloc_shared([BS, BK], "float16")
            kbdec0_lo_l1 = T.alloc_shared([BS, BK], "float16")
            kdec1_hi_l1 = T.alloc_shared([BS, BK], "float16")
            kdec1_lo_l1 = T.alloc_shared([BS, BK], "float16")
            qdec1_hi_l1 = T.alloc_shared([BS, BK], "float16")
            qdec1_lo_l1 = T.alloc_shared([BS, BK], "float16")
            kbdec1_hi_l1 = T.alloc_shared([BS, BK], "float16")
            kbdec1_lo_l1 = T.alloc_shared([BS, BK], "float16")
            kdec2_hi_l1 = T.alloc_shared([BS, BK], "float16")
            kdec2_lo_l1 = T.alloc_shared([BS, BK], "float16")
            qdec2_hi_l1 = T.alloc_shared([BS, BK], "float16")
            qdec2_lo_l1 = T.alloc_shared([BS, BK], "float16")
            kbdec2_hi_l1 = T.alloc_shared([BS, BK], "float16")
            kbdec2_lo_l1 = T.alloc_shared([BS, BK], "float16")
            kdec3_hi_l1 = T.alloc_shared([BS, BK], "float16")
            kdec3_lo_l1 = T.alloc_shared([BS, BK], "float16")
            qdec3_hi_l1 = T.alloc_shared([BS, BK], "float16")
            qdec3_lo_l1 = T.alloc_shared([BS, BK], "float16")
            kbdec3_hi_l1 = T.alloc_shared([BS, BK], "float16")
            kbdec3_lo_l1 = T.alloc_shared([BS, BK], "float16")
            # fragments
            dq2_frag0 = T.alloc_fragment([BC, BK], "float32")
            dk2p_frag0 = T.alloc_fragment([BC, BK], "float32")
            dkt_frag0 = T.alloc_fragment([BC, BK], "float32")
            dq2_frag1 = T.alloc_fragment([BC, BK], "float32")
            dk2p_frag1 = T.alloc_fragment([BC, BK], "float32")
            dkt_frag1 = T.alloc_fragment([BC, BK], "float32")
            dq2_frag2 = T.alloc_fragment([BC, BK], "float32")
            dk2p_frag2 = T.alloc_fragment([BC, BK], "float32")
            dkt_frag2 = T.alloc_fragment([BC, BK], "float32")
            dq2_frag3 = T.alloc_fragment([BC, BK], "float32")
            dk2p_frag3 = T.alloc_fragment([BC, BK], "float32")
            dkt_frag3 = T.alloc_fragment([BC, BK], "float32")
            # tail: dedicated C->V dst per fragment (per s)
            dq2pre_ub0 = T.alloc_shared([BC, BK], "float32")
            dk2pre_ub0 = T.alloc_shared([BC, BK], "float32")
            dkt_ub0 = T.alloc_shared([BC, BK], "float32")
            dq2pre_ub1 = T.alloc_shared([BC, BK], "float32")
            dk2pre_ub1 = T.alloc_shared([BC, BK], "float32")
            dkt_ub1 = T.alloc_shared([BC, BK], "float32")
            dq2pre_ub2 = T.alloc_shared([BC, BK], "float32")
            dk2pre_ub2 = T.alloc_shared([BC, BK], "float32")
            dkt_ub2 = T.alloc_shared([BC, BK], "float32")
            dq2pre_ub3 = T.alloc_shared([BC, BK], "float32")
            dk2pre_ub3 = T.alloc_shared([BC, BK], "float32")
            dkt_ub3 = T.alloc_shared([BC, BK], "float32")
            # tail: shared pure-V transients
            d_s = T.alloc_shared([BC, BK], "float32")
            v_s = T.alloc_shared([BC, BK], "float32")
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

            for i_kk in T.Pipelined(DK // BK, num_stages=1):
                ik0 = i_kk * BK
                ik1 = ik0 + BK

                # ==== PHASE A (V): half loads + MdA/MdK splits + decays ====
                T.copy(K[bb, s0 : s0 + BS, bh, ik0:ik1], k_ub)
                T.copy(G[bb, s0 : s0 + BS, bh, ik0:ik1], g_ub)
                T.copy(Q[bb, s0 : s0 + BS, bh, ik0:ik1], q_ub)
                # MdA/MdK fp16 splits (in-body idempotent)
                T.copy(mda_f, mda_hi_ub)
                T.copy(mda_hi_ub, s64_hi_f32)
                T.tile.sub(s64_lo_f32, mda_f, s64_hi_f32)
                T.copy(s64_lo_f32, mda_lo_ub)
                T.copy(mda_hi_ub, mda_hi_l1)
                T.copy(mda_lo_ub, mda_lo_l1)
                T.copy(mdk_f, mdk_hi_ub)
                T.copy(mdk_hi_ub, s64_hi_f32)
                T.tile.sub(s64_lo_f32, mdk_f, s64_hi_f32)
                T.copy(s64_lo_f32, mdk_lo_ub)
                T.copy(mdk_hi_ub, mdk_hi_l1)
                T.copy(mdk_lo_ub, mdk_lo_l1)

                # ---- s = 0 (KMAX = 32: rows 0..31) ----
                T.copy(g_ub[0, 0:BK], gn_s0)
                T.copy(g_ub[BC - 1, 0:BK], gnp_s0)
                # Kdec_0 (chunks 0,1): k * 2^(gn_s0 - g)
                T.tile.broadcast(t16, gn_s0, axis=0)
                T.tile.sub(t16, t16, g_ub[0:BC, 0:BK])
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[0:BC, 0:BK], k_ub[0:BC, 0:BK], t16)
                T.tile.broadcast(t16, gn_s0, axis=0)
                T.tile.sub(t16, t16, g_ub[BC : 2 * BC, 0:BK])
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[BC : 2 * BC, 0:BK], k_ub[BC : 2 * BC, 0:BK], t16)
                T.copy(odec_f, kdec0_hi_ub)
                T.copy(kdec0_hi_ub, sbk_hi_f32)
                T.tile.sub(sbk_lo_f32, odec_f, sbk_hi_f32)
                T.copy(sbk_lo_f32, kdec0_lo_ub)
                T.copy(kdec0_hi_ub, kdec0_hi_l1)
                T.copy(kdec0_lo_ub, kdec0_lo_l1)
                # Qdec_0 (4 chunks): q * 2^(g - gnp_s0)
                T.tile.broadcast(t16, gnp_s0, axis=0)
                T.tile.sub(t16, g_ub[0:BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[0:BC, 0:BK], q_ub[0:BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s0, axis=0)
                T.tile.sub(t16, g_ub[BC : 2 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[BC : 2 * BC, 0:BK], q_ub[BC : 2 * BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s0, axis=0)
                T.tile.sub(t16, g_ub[2 * BC : 3 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[2 * BC : 3 * BC, 0:BK], q_ub[2 * BC : 3 * BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s0, axis=0)
                T.tile.sub(t16, g_ub[3 * BC : 4 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[3 * BC : 4 * BC, 0:BK], q_ub[3 * BC : 4 * BC, 0:BK], t16)
                T.copy(odec_f, qdec0_hi_ub)
                T.copy(qdec0_hi_ub, sbk_hi_f32)
                T.tile.sub(sbk_lo_f32, odec_f, sbk_hi_f32)
                T.copy(sbk_lo_f32, qdec0_lo_ub)
                T.copy(qdec0_hi_ub, qdec0_hi_l1)
                T.copy(qdec0_lo_ub, qdec0_lo_l1)
                # KBdec_0 (4 chunks): beta*k * 2^(g - gnp_s0)
                T.tile.broadcast(bcol2d, bet[0:BC], axis=1)
                T.tile.mul(odec_f[0:BC, 0:BK], k_ub[0:BC, 0:BK], bcol2d)
                T.tile.mul(odec_f[0:BC, 0:BK], odec_f[0:BC, 0:BK], t16)
                T.tile.broadcast(bcol2d, bet[BC : 2 * BC], axis=1)
                T.tile.mul(odec_f[BC : 2 * BC, 0:BK], k_ub[BC : 2 * BC, 0:BK], bcol2d)
                T.tile.mul(odec_f[BC : 2 * BC, 0:BK], odec_f[BC : 2 * BC, 0:BK], t16)
                T.tile.broadcast(bcol2d, bet[2 * BC : 3 * BC], axis=1)
                T.tile.mul(odec_f[2 * BC : 3 * BC, 0:BK], k_ub[2 * BC : 3 * BC, 0:BK], bcol2d)
                T.tile.mul(odec_f[2 * BC : 3 * BC, 0:BK], odec_f[2 * BC : 3 * BC, 0:BK], t16)
                T.tile.broadcast(bcol2d, bet[3 * BC : 4 * BC], axis=1)
                T.tile.mul(odec_f[3 * BC : 4 * BC, 0:BK], k_ub[3 * BC : 4 * BC, 0:BK], bcol2d)
                T.tile.mul(odec_f[3 * BC : 4 * BC, 0:BK], odec_f[3 * BC : 4 * BC, 0:BK], t16)
                T.copy(odec_f, kbdec0_hi_ub)
                T.copy(kbdec0_hi_ub, sbk_hi_f32)
                T.tile.sub(sbk_lo_f32, odec_f, sbk_hi_f32)
                T.copy(sbk_lo_f32, kbdec0_lo_ub)
                T.copy(kbdec0_hi_ub, kbdec0_hi_l1)
                T.copy(kbdec0_lo_ub, kbdec0_lo_l1)

                # ---- s = 1 (KMAX = 32) ----
                T.copy(g_ub[BC, 0:BK], gn_s1)
                T.copy(g_ub[2 * BC - 1, 0:BK], gnp_s1)
                T.tile.broadcast(t16, gn_s1, axis=0)
                T.tile.sub(t16, t16, g_ub[0:BC, 0:BK])
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[0:BC, 0:BK], k_ub[0:BC, 0:BK], t16)
                T.tile.broadcast(t16, gn_s1, axis=0)
                T.tile.sub(t16, t16, g_ub[BC : 2 * BC, 0:BK])
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[BC : 2 * BC, 0:BK], k_ub[BC : 2 * BC, 0:BK], t16)
                T.copy(odec_f, kdec1_hi_ub)
                T.copy(kdec1_hi_ub, sbk_hi_f32)
                T.tile.sub(sbk_lo_f32, odec_f, sbk_hi_f32)
                T.copy(sbk_lo_f32, kdec1_lo_ub)
                T.copy(kdec1_hi_ub, kdec1_hi_l1)
                T.copy(kdec1_lo_ub, kdec1_lo_l1)
                T.tile.broadcast(t16, gnp_s1, axis=0)
                T.tile.sub(t16, g_ub[0:BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[0:BC, 0:BK], q_ub[0:BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s1, axis=0)
                T.tile.sub(t16, g_ub[BC : 2 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[BC : 2 * BC, 0:BK], q_ub[BC : 2 * BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s1, axis=0)
                T.tile.sub(t16, g_ub[2 * BC : 3 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[2 * BC : 3 * BC, 0:BK], q_ub[2 * BC : 3 * BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s1, axis=0)
                T.tile.sub(t16, g_ub[3 * BC : 4 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[3 * BC : 4 * BC, 0:BK], q_ub[3 * BC : 4 * BC, 0:BK], t16)
                T.copy(odec_f, qdec1_hi_ub)
                T.copy(qdec1_hi_ub, sbk_hi_f32)
                T.tile.sub(sbk_lo_f32, odec_f, sbk_hi_f32)
                T.copy(sbk_lo_f32, qdec1_lo_ub)
                T.copy(qdec1_hi_ub, qdec1_hi_l1)
                T.copy(qdec1_lo_ub, qdec1_lo_l1)
                T.tile.broadcast(t16, gnp_s1, axis=0)
                T.tile.sub(t16, g_ub[0:BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.broadcast(bcol2d, bet[0:BC], axis=1)
                T.tile.mul(odec_f[0:BC, 0:BK], k_ub[0:BC, 0:BK], bcol2d)
                T.tile.mul(odec_f[0:BC, 0:BK], odec_f[0:BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s1, axis=0)
                T.tile.sub(t16, g_ub[BC : 2 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.broadcast(bcol2d, bet[BC : 2 * BC], axis=1)
                T.tile.mul(odec_f[BC : 2 * BC, 0:BK], k_ub[BC : 2 * BC, 0:BK], bcol2d)
                T.tile.mul(odec_f[BC : 2 * BC, 0:BK], odec_f[BC : 2 * BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s1, axis=0)
                T.tile.sub(t16, g_ub[2 * BC : 3 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.broadcast(bcol2d, bet[2 * BC : 3 * BC], axis=1)
                T.tile.mul(odec_f[2 * BC : 3 * BC, 0:BK], k_ub[2 * BC : 3 * BC, 0:BK], bcol2d)
                T.tile.mul(odec_f[2 * BC : 3 * BC, 0:BK], odec_f[2 * BC : 3 * BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s1, axis=0)
                T.tile.sub(t16, g_ub[3 * BC : 4 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.broadcast(bcol2d, bet[3 * BC : 4 * BC], axis=1)
                T.tile.mul(odec_f[3 * BC : 4 * BC, 0:BK], k_ub[3 * BC : 4 * BC, 0:BK], bcol2d)
                T.tile.mul(odec_f[3 * BC : 4 * BC, 0:BK], odec_f[3 * BC : 4 * BC, 0:BK], t16)
                T.copy(odec_f, kbdec1_hi_ub)
                T.copy(kbdec1_hi_ub, sbk_hi_f32)
                T.tile.sub(sbk_lo_f32, odec_f, sbk_hi_f32)
                T.copy(sbk_lo_f32, kbdec1_lo_ub)
                T.copy(kbdec1_hi_ub, kbdec1_hi_l1)
                T.copy(kbdec1_lo_ub, kbdec1_lo_l1)

                # ---- s = 2 (KMAX = 48: chunks 0,1,2) ----
                T.copy(g_ub[2 * BC, 0:BK], gn_s2)
                T.copy(g_ub[3 * BC - 1, 0:BK], gnp_s2)
                T.tile.broadcast(t16, gn_s2, axis=0)
                T.tile.sub(t16, t16, g_ub[0:BC, 0:BK])
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[0:BC, 0:BK], k_ub[0:BC, 0:BK], t16)
                T.tile.broadcast(t16, gn_s2, axis=0)
                T.tile.sub(t16, t16, g_ub[BC : 2 * BC, 0:BK])
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[BC : 2 * BC, 0:BK], k_ub[BC : 2 * BC, 0:BK], t16)
                T.tile.broadcast(t16, gn_s2, axis=0)
                T.tile.sub(t16, t16, g_ub[2 * BC : 3 * BC, 0:BK])
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[2 * BC : 3 * BC, 0:BK], k_ub[2 * BC : 3 * BC, 0:BK], t16)
                T.copy(odec_f, kdec2_hi_ub)
                T.copy(kdec2_hi_ub, sbk_hi_f32)
                T.tile.sub(sbk_lo_f32, odec_f, sbk_hi_f32)
                T.copy(sbk_lo_f32, kdec2_lo_ub)
                T.copy(kdec2_hi_ub, kdec2_hi_l1)
                T.copy(kdec2_lo_ub, kdec2_lo_l1)
                T.tile.broadcast(t16, gnp_s2, axis=0)
                T.tile.sub(t16, g_ub[0:BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[0:BC, 0:BK], q_ub[0:BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s2, axis=0)
                T.tile.sub(t16, g_ub[BC : 2 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[BC : 2 * BC, 0:BK], q_ub[BC : 2 * BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s2, axis=0)
                T.tile.sub(t16, g_ub[2 * BC : 3 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[2 * BC : 3 * BC, 0:BK], q_ub[2 * BC : 3 * BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s2, axis=0)
                T.tile.sub(t16, g_ub[3 * BC : 4 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[3 * BC : 4 * BC, 0:BK], q_ub[3 * BC : 4 * BC, 0:BK], t16)
                T.copy(odec_f, qdec2_hi_ub)
                T.copy(qdec2_hi_ub, sbk_hi_f32)
                T.tile.sub(sbk_lo_f32, odec_f, sbk_hi_f32)
                T.copy(sbk_lo_f32, qdec2_lo_ub)
                T.copy(qdec2_hi_ub, qdec2_hi_l1)
                T.copy(qdec2_lo_ub, qdec2_lo_l1)
                T.tile.broadcast(t16, gnp_s2, axis=0)
                T.tile.sub(t16, g_ub[0:BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.broadcast(bcol2d, bet[0:BC], axis=1)
                T.tile.mul(odec_f[0:BC, 0:BK], k_ub[0:BC, 0:BK], bcol2d)
                T.tile.mul(odec_f[0:BC, 0:BK], odec_f[0:BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s2, axis=0)
                T.tile.sub(t16, g_ub[BC : 2 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.broadcast(bcol2d, bet[BC : 2 * BC], axis=1)
                T.tile.mul(odec_f[BC : 2 * BC, 0:BK], k_ub[BC : 2 * BC, 0:BK], bcol2d)
                T.tile.mul(odec_f[BC : 2 * BC, 0:BK], odec_f[BC : 2 * BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s2, axis=0)
                T.tile.sub(t16, g_ub[2 * BC : 3 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.broadcast(bcol2d, bet[2 * BC : 3 * BC], axis=1)
                T.tile.mul(odec_f[2 * BC : 3 * BC, 0:BK], k_ub[2 * BC : 3 * BC, 0:BK], bcol2d)
                T.tile.mul(odec_f[2 * BC : 3 * BC, 0:BK], odec_f[2 * BC : 3 * BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s2, axis=0)
                T.tile.sub(t16, g_ub[3 * BC : 4 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.broadcast(bcol2d, bet[3 * BC : 4 * BC], axis=1)
                T.tile.mul(odec_f[3 * BC : 4 * BC, 0:BK], k_ub[3 * BC : 4 * BC, 0:BK], bcol2d)
                T.tile.mul(odec_f[3 * BC : 4 * BC, 0:BK], odec_f[3 * BC : 4 * BC, 0:BK], t16)
                T.copy(odec_f, kbdec2_hi_ub)
                T.copy(kbdec2_hi_ub, sbk_hi_f32)
                T.tile.sub(sbk_lo_f32, odec_f, sbk_hi_f32)
                T.copy(sbk_lo_f32, kbdec2_lo_ub)
                T.copy(kbdec2_hi_ub, kbdec2_hi_l1)
                T.copy(kbdec2_lo_ub, kbdec2_lo_l1)

                # ---- s = 3 (KMAX = 64: chunks 0..3) ----
                T.copy(g_ub[3 * BC, 0:BK], gn_s3)
                T.copy(g_ub[4 * BC - 1, 0:BK], gnp_s3)
                T.tile.broadcast(t16, gn_s3, axis=0)
                T.tile.sub(t16, t16, g_ub[0:BC, 0:BK])
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[0:BC, 0:BK], k_ub[0:BC, 0:BK], t16)
                T.tile.broadcast(t16, gn_s3, axis=0)
                T.tile.sub(t16, t16, g_ub[BC : 2 * BC, 0:BK])
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[BC : 2 * BC, 0:BK], k_ub[BC : 2 * BC, 0:BK], t16)
                T.tile.broadcast(t16, gn_s3, axis=0)
                T.tile.sub(t16, t16, g_ub[2 * BC : 3 * BC, 0:BK])
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[2 * BC : 3 * BC, 0:BK], k_ub[2 * BC : 3 * BC, 0:BK], t16)
                T.tile.broadcast(t16, gn_s3, axis=0)
                T.tile.sub(t16, t16, g_ub[3 * BC : 4 * BC, 0:BK])
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[3 * BC : 4 * BC, 0:BK], k_ub[3 * BC : 4 * BC, 0:BK], t16)
                T.copy(odec_f, kdec3_hi_ub)
                T.copy(kdec3_hi_ub, sbk_hi_f32)
                T.tile.sub(sbk_lo_f32, odec_f, sbk_hi_f32)
                T.copy(sbk_lo_f32, kdec3_lo_ub)
                T.copy(kdec3_hi_ub, kdec3_hi_l1)
                T.copy(kdec3_lo_ub, kdec3_lo_l1)
                T.tile.broadcast(t16, gnp_s3, axis=0)
                T.tile.sub(t16, g_ub[0:BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[0:BC, 0:BK], q_ub[0:BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s3, axis=0)
                T.tile.sub(t16, g_ub[BC : 2 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[BC : 2 * BC, 0:BK], q_ub[BC : 2 * BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s3, axis=0)
                T.tile.sub(t16, g_ub[2 * BC : 3 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[2 * BC : 3 * BC, 0:BK], q_ub[2 * BC : 3 * BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s3, axis=0)
                T.tile.sub(t16, g_ub[3 * BC : 4 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.mul(odec_f[3 * BC : 4 * BC, 0:BK], q_ub[3 * BC : 4 * BC, 0:BK], t16)
                T.copy(odec_f, qdec3_hi_ub)
                T.copy(qdec3_hi_ub, sbk_hi_f32)
                T.tile.sub(sbk_lo_f32, odec_f, sbk_hi_f32)
                T.copy(sbk_lo_f32, qdec3_lo_ub)
                T.copy(qdec3_hi_ub, qdec3_hi_l1)
                T.copy(qdec3_lo_ub, qdec3_lo_l1)
                T.tile.broadcast(t16, gnp_s3, axis=0)
                T.tile.sub(t16, g_ub[0:BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.broadcast(bcol2d, bet[0:BC], axis=1)
                T.tile.mul(odec_f[0:BC, 0:BK], k_ub[0:BC, 0:BK], bcol2d)
                T.tile.mul(odec_f[0:BC, 0:BK], odec_f[0:BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s3, axis=0)
                T.tile.sub(t16, g_ub[BC : 2 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.broadcast(bcol2d, bet[BC : 2 * BC], axis=1)
                T.tile.mul(odec_f[BC : 2 * BC, 0:BK], k_ub[BC : 2 * BC, 0:BK], bcol2d)
                T.tile.mul(odec_f[BC : 2 * BC, 0:BK], odec_f[BC : 2 * BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s3, axis=0)
                T.tile.sub(t16, g_ub[2 * BC : 3 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.broadcast(bcol2d, bet[2 * BC : 3 * BC], axis=1)
                T.tile.mul(odec_f[2 * BC : 3 * BC, 0:BK], k_ub[2 * BC : 3 * BC, 0:BK], bcol2d)
                T.tile.mul(odec_f[2 * BC : 3 * BC, 0:BK], odec_f[2 * BC : 3 * BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s3, axis=0)
                T.tile.sub(t16, g_ub[3 * BC : 4 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.broadcast(bcol2d, bet[3 * BC : 4 * BC], axis=1)
                T.tile.mul(odec_f[3 * BC : 4 * BC, 0:BK], k_ub[3 * BC : 4 * BC, 0:BK], bcol2d)
                T.tile.mul(odec_f[3 * BC : 4 * BC, 0:BK], odec_f[3 * BC : 4 * BC, 0:BK], t16)
                T.copy(odec_f, kbdec3_hi_ub)
                T.copy(kbdec3_hi_ub, sbk_hi_f32)
                T.tile.sub(sbk_lo_f32, odec_f, sbk_hi_f32)
                T.copy(sbk_lo_f32, kbdec3_lo_ub)
                T.copy(kbdec3_hi_ub, kbdec3_hi_l1)
                T.copy(kbdec3_lo_ub, kbdec3_lo_l1)

                # ==== PHASE B (C): 48 contiguous gemm_v0 ====
                # s = 0 (KMAX = 32; A-rows 0..15)
                T.gemm_v0(mda_hi_l1[0:BC, 0:2 * BC], kdec0_hi_l1[0:2 * BC, 0:BK],
                          dq2_frag0, init=True)
                T.gemm_v0(mda_lo_l1[0:BC, 0:2 * BC], kdec0_hi_l1[0:2 * BC, 0:BK],
                          dq2_frag0, init=False)
                T.gemm_v0(mda_hi_l1[0:BC, 0:2 * BC], kdec0_lo_l1[0:2 * BC, 0:BK],
                          dq2_frag0, init=False)
                T.gemm_v0(mdk_hi_l1[0:BC, 0:2 * BC], kdec0_hi_l1[0:2 * BC, 0:BK],
                          dk2p_frag0, init=True)
                T.gemm_v0(mdk_lo_l1[0:BC, 0:2 * BC], kdec0_hi_l1[0:2 * BC, 0:BK],
                          dk2p_frag0, init=False)
                T.gemm_v0(mdk_hi_l1[0:BC, 0:2 * BC], kdec0_lo_l1[0:2 * BC, 0:BK],
                          dk2p_frag0, init=False)
                T.gemm_v0(mda_hi_l1[0:BS, 0:BC], qdec0_hi_l1, dkt_frag0,
                          transpose_A=True, init=True)
                T.gemm_v0(mda_lo_l1[0:BS, 0:BC], qdec0_hi_l1, dkt_frag0,
                          transpose_A=True, init=False)
                T.gemm_v0(mda_hi_l1[0:BS, 0:BC], qdec0_lo_l1, dkt_frag0,
                          transpose_A=True, init=False)
                T.gemm_v0(mdk_hi_l1[0:BS, 0:BC], kbdec0_hi_l1, dkt_frag0,
                          transpose_A=True, init=False)
                T.gemm_v0(mdk_lo_l1[0:BS, 0:BC], kbdec0_hi_l1, dkt_frag0,
                          transpose_A=True, init=False)
                T.gemm_v0(mdk_hi_l1[0:BS, 0:BC], kbdec0_lo_l1, dkt_frag0,
                          transpose_A=True, init=False)
                # s = 1 (KMAX = 32; A-rows 16..31)
                T.gemm_v0(mda_hi_l1[BC : 2 * BC, 0:2 * BC],
                          kdec1_hi_l1[0:2 * BC, 0:BK], dq2_frag1, init=True)
                T.gemm_v0(mda_lo_l1[BC : 2 * BC, 0:2 * BC],
                          kdec1_hi_l1[0:2 * BC, 0:BK], dq2_frag1, init=False)
                T.gemm_v0(mda_hi_l1[BC : 2 * BC, 0:2 * BC],
                          kdec1_lo_l1[0:2 * BC, 0:BK], dq2_frag1, init=False)
                T.gemm_v0(mdk_hi_l1[BC : 2 * BC, 0:2 * BC],
                          kdec1_hi_l1[0:2 * BC, 0:BK], dk2p_frag1, init=True)
                T.gemm_v0(mdk_lo_l1[BC : 2 * BC, 0:2 * BC],
                          kdec1_hi_l1[0:2 * BC, 0:BK], dk2p_frag1, init=False)
                T.gemm_v0(mdk_hi_l1[BC : 2 * BC, 0:2 * BC],
                          kdec1_lo_l1[0:2 * BC, 0:BK], dk2p_frag1, init=False)
                T.gemm_v0(mda_hi_l1[0:BS, BC : 2 * BC], qdec1_hi_l1, dkt_frag1,
                          transpose_A=True, init=True)
                T.gemm_v0(mda_lo_l1[0:BS, BC : 2 * BC], qdec1_hi_l1, dkt_frag1,
                          transpose_A=True, init=False)
                T.gemm_v0(mda_hi_l1[0:BS, BC : 2 * BC], qdec1_lo_l1, dkt_frag1,
                          transpose_A=True, init=False)
                T.gemm_v0(mdk_hi_l1[0:BS, BC : 2 * BC], kbdec1_hi_l1, dkt_frag1,
                          transpose_A=True, init=False)
                T.gemm_v0(mdk_lo_l1[0:BS, BC : 2 * BC], kbdec1_hi_l1, dkt_frag1,
                          transpose_A=True, init=False)
                T.gemm_v0(mdk_hi_l1[0:BS, BC : 2 * BC], kbdec1_lo_l1, dkt_frag1,
                          transpose_A=True, init=False)
                # s = 2 (KMAX = 48; A-rows 32..47)
                T.gemm_v0(mda_hi_l1[2 * BC : 3 * BC, 0:3 * BC],
                          kdec2_hi_l1[0:3 * BC, 0:BK], dq2_frag2, init=True)
                T.gemm_v0(mda_lo_l1[2 * BC : 3 * BC, 0:3 * BC],
                          kdec2_hi_l1[0:3 * BC, 0:BK], dq2_frag2, init=False)
                T.gemm_v0(mda_hi_l1[2 * BC : 3 * BC, 0:3 * BC],
                          kdec2_lo_l1[0:3 * BC, 0:BK], dq2_frag2, init=False)
                T.gemm_v0(mdk_hi_l1[2 * BC : 3 * BC, 0:3 * BC],
                          kdec2_hi_l1[0:3 * BC, 0:BK], dk2p_frag2, init=True)
                T.gemm_v0(mdk_lo_l1[2 * BC : 3 * BC, 0:3 * BC],
                          kdec2_hi_l1[0:3 * BC, 0:BK], dk2p_frag2, init=False)
                T.gemm_v0(mdk_hi_l1[2 * BC : 3 * BC, 0:3 * BC],
                          kdec2_lo_l1[0:3 * BC, 0:BK], dk2p_frag2, init=False)
                T.gemm_v0(mda_hi_l1[0:BS, 2 * BC : 3 * BC], qdec2_hi_l1, dkt_frag2,
                          transpose_A=True, init=True)
                T.gemm_v0(mda_lo_l1[0:BS, 2 * BC : 3 * BC], qdec2_hi_l1, dkt_frag2,
                          transpose_A=True, init=False)
                T.gemm_v0(mda_hi_l1[0:BS, 2 * BC : 3 * BC], qdec2_lo_l1, dkt_frag2,
                          transpose_A=True, init=False)
                T.gemm_v0(mdk_hi_l1[0:BS, 2 * BC : 3 * BC], kbdec2_hi_l1, dkt_frag2,
                          transpose_A=True, init=False)
                T.gemm_v0(mdk_lo_l1[0:BS, 2 * BC : 3 * BC], kbdec2_hi_l1, dkt_frag2,
                          transpose_A=True, init=False)
                T.gemm_v0(mdk_hi_l1[0:BS, 2 * BC : 3 * BC], kbdec2_lo_l1, dkt_frag2,
                          transpose_A=True, init=False)
                # s = 3 (KMAX = 64; A-rows 48..63)
                T.gemm_v0(mda_hi_l1[3 * BC : 4 * BC, 0:BS], kdec3_hi_l1[0:BS, 0:BK],
                          dq2_frag3, init=True)
                T.gemm_v0(mda_lo_l1[3 * BC : 4 * BC, 0:BS], kdec3_hi_l1[0:BS, 0:BK],
                          dq2_frag3, init=False)
                T.gemm_v0(mda_hi_l1[3 * BC : 4 * BC, 0:BS], kdec3_lo_l1[0:BS, 0:BK],
                          dq2_frag3, init=False)
                T.gemm_v0(mdk_hi_l1[3 * BC : 4 * BC, 0:BS], kdec3_hi_l1[0:BS, 0:BK],
                          dk2p_frag3, init=True)
                T.gemm_v0(mdk_lo_l1[3 * BC : 4 * BC, 0:BS], kdec3_hi_l1[0:BS, 0:BK],
                          dk2p_frag3, init=False)
                T.gemm_v0(mdk_hi_l1[3 * BC : 4 * BC, 0:BS], kdec3_lo_l1[0:BS, 0:BK],
                          dk2p_frag3, init=False)
                T.gemm_v0(mda_hi_l1[0:BS, 3 * BC : 4 * BC], qdec3_hi_l1, dkt_frag3,
                          transpose_A=True, init=True)
                T.gemm_v0(mda_lo_l1[0:BS, 3 * BC : 4 * BC], qdec3_hi_l1, dkt_frag3,
                          transpose_A=True, init=False)
                T.gemm_v0(mda_hi_l1[0:BS, 3 * BC : 4 * BC], qdec3_lo_l1, dkt_frag3,
                          transpose_A=True, init=False)
                T.gemm_v0(mdk_hi_l1[0:BS, 3 * BC : 4 * BC], kbdec3_hi_l1, dkt_frag3,
                          transpose_A=True, init=False)
                T.gemm_v0(mdk_lo_l1[0:BS, 3 * BC : 4 * BC], kbdec3_hi_l1, dkt_frag3,
                          transpose_A=True, init=False)
                T.gemm_v0(mdk_hi_l1[0:BS, 3 * BC : 4 * BC], kbdec3_lo_l1, dkt_frag3,
                          transpose_A=True, init=False)

                # ==== PHASE C (V): tail per sub-chunk ====
                # s = 0 (rows 0..15)
                T.tile.broadcast(d_s, gn_s0, axis=0)
                T.tile.sub(d_s, g_ub[0:BC, 0:BK], d_s)
                T.tile.mul(d_s, d_s, LN2)
                T.tile.exp(d_s, d_s)
                T.tile.broadcast(v_s, gnp_s0, axis=0)
                T.tile.sub(v_s, v_s, g_ub[0:BC, 0:BK])
                T.tile.mul(v_s, v_s, LN2)
                T.tile.exp(v_s, v_s)
                T.copy(dq2_frag0, dq2pre_ub0)
                T.tile.mul(dq2_ub, dq2pre_ub0, d_s)
                T.copy(dq[bb, s0 : s0 + BC, bh, ik0:ik1], dq_rows)
                T.tile.add(dq2_ub, dq2_ub, dq_rows)
                T.copy(dq2_ub, dq2[bb, s0 : s0 + BC, bh, ik0:ik1])
                T.copy(dk2p_frag0, dk2pre_ub0)
                T.tile.mul(r_tmp, dk2pre_ub0, k_ub[0:BC, 0:BK])
                T.reduce_sum(r_tmp, r_part, dim=1, clear=True)
                T.tile.add(db_acc[0:BC], db_acc[0:BC], r_part)
                T.tile.broadcast(bcol2d, bet[0:BC], axis=1)
                T.tile.mul(dk2b_ub, dk2pre_ub0, bcol2d)
                T.copy(dkt_frag0, dkt_ub0)
                T.tile.sub(t16b, dk2b_ub, dkt_ub0)
                T.tile.mul(t16b, t16b, k_ub[0:BC, 0:BK])
                T.tile.mul(dg2_ub, q_ub[0:BC, 0:BK], dq2pre_ub0)
                T.tile.add(dg2_ub, dg2_ub, t16b)
                T.copy(dg[bb, s0 : s0 + BC, bh, ik0:ik1], dg_rows)
                T.tile.add(dg2_ub, dg2_ub, dg_rows)
                T.copy(dg2_ub, dg2[bb, s0 : s0 + BC, bh, ik0:ik1])
                T.copy(dk[bb, s0 : s0 + BC, bh, ik0:ik1], dk_rows)
                T.tile.add(dk2_ub, dk2b_ub, dkt_ub0)
                T.tile.add(dk2_ub, dk2_ub, dk_rows)
                T.copy(dk2_ub, dk2[bb, s0 : s0 + BC, bh, ik0:ik1])
                # s = 1 (rows 16..31)
                T.tile.broadcast(d_s, gn_s1, axis=0)
                T.tile.sub(d_s, g_ub[BC : 2 * BC, 0:BK], d_s)
                T.tile.mul(d_s, d_s, LN2)
                T.tile.exp(d_s, d_s)
                T.tile.broadcast(v_s, gnp_s1, axis=0)
                T.tile.sub(v_s, v_s, g_ub[BC : 2 * BC, 0:BK])
                T.tile.mul(v_s, v_s, LN2)
                T.tile.exp(v_s, v_s)
                T.copy(dq2_frag1, dq2pre_ub1)
                T.tile.mul(dq2_ub, dq2pre_ub1, d_s)
                T.copy(dq[bb, s0 + BC : s0 + 2 * BC, bh, ik0:ik1], dq_rows)
                T.tile.add(dq2_ub, dq2_ub, dq_rows)
                T.copy(dq2_ub, dq2[bb, s0 + BC : s0 + 2 * BC, bh, ik0:ik1])
                T.copy(dk2p_frag1, dk2pre_ub1)
                T.tile.mul(r_tmp, dk2pre_ub1, k_ub[BC : 2 * BC, 0:BK])
                T.reduce_sum(r_tmp, r_part, dim=1, clear=True)
                T.tile.add(db_acc[BC : 2 * BC], db_acc[BC : 2 * BC], r_part)
                T.tile.broadcast(bcol2d, bet[BC : 2 * BC], axis=1)
                T.tile.mul(dk2b_ub, dk2pre_ub1, bcol2d)
                T.copy(dkt_frag1, dkt_ub1)
                T.tile.sub(t16b, dk2b_ub, dkt_ub1)
                T.tile.mul(t16b, t16b, k_ub[BC : 2 * BC, 0:BK])
                T.tile.mul(dg2_ub, q_ub[BC : 2 * BC, 0:BK], dq2pre_ub1)
                T.tile.add(dg2_ub, dg2_ub, t16b)
                T.copy(dg[bb, s0 + BC : s0 + 2 * BC, bh, ik0:ik1], dg_rows)
                T.tile.add(dg2_ub, dg2_ub, dg_rows)
                T.copy(dg2_ub, dg2[bb, s0 + BC : s0 + 2 * BC, bh, ik0:ik1])
                T.copy(dk[bb, s0 + BC : s0 + 2 * BC, bh, ik0:ik1], dk_rows)
                T.tile.add(dk2_ub, dk2b_ub, dkt_ub1)
                T.tile.add(dk2_ub, dk2_ub, dk_rows)
                T.copy(dk2_ub, dk2[bb, s0 + BC : s0 + 2 * BC, bh, ik0:ik1])
                # s = 2 (rows 32..47)
                T.tile.broadcast(d_s, gn_s2, axis=0)
                T.tile.sub(d_s, g_ub[2 * BC : 3 * BC, 0:BK], d_s)
                T.tile.mul(d_s, d_s, LN2)
                T.tile.exp(d_s, d_s)
                T.tile.broadcast(v_s, gnp_s2, axis=0)
                T.tile.sub(v_s, v_s, g_ub[2 * BC : 3 * BC, 0:BK])
                T.tile.mul(v_s, v_s, LN2)
                T.tile.exp(v_s, v_s)
                T.copy(dq2_frag2, dq2pre_ub2)
                T.tile.mul(dq2_ub, dq2pre_ub2, d_s)
                T.copy(dq[bb, s0 + 2 * BC : s0 + 3 * BC, bh, ik0:ik1], dq_rows)
                T.tile.add(dq2_ub, dq2_ub, dq_rows)
                T.copy(dq2_ub, dq2[bb, s0 + 2 * BC : s0 + 3 * BC, bh, ik0:ik1])
                T.copy(dk2p_frag2, dk2pre_ub2)
                T.tile.mul(r_tmp, dk2pre_ub2, k_ub[2 * BC : 3 * BC, 0:BK])
                T.reduce_sum(r_tmp, r_part, dim=1, clear=True)
                T.tile.add(db_acc[2 * BC : 3 * BC], db_acc[2 * BC : 3 * BC], r_part)
                T.tile.broadcast(bcol2d, bet[2 * BC : 3 * BC], axis=1)
                T.tile.mul(dk2b_ub, dk2pre_ub2, bcol2d)
                T.copy(dkt_frag2, dkt_ub2)
                T.tile.sub(t16b, dk2b_ub, dkt_ub2)
                T.tile.mul(t16b, t16b, k_ub[2 * BC : 3 * BC, 0:BK])
                T.tile.mul(dg2_ub, q_ub[2 * BC : 3 * BC, 0:BK], dq2pre_ub2)
                T.tile.add(dg2_ub, dg2_ub, t16b)
                T.copy(dg[bb, s0 + 2 * BC : s0 + 3 * BC, bh, ik0:ik1], dg_rows)
                T.tile.add(dg2_ub, dg2_ub, dg_rows)
                T.copy(dg2_ub, dg2[bb, s0 + 2 * BC : s0 + 3 * BC, bh, ik0:ik1])
                T.copy(dk[bb, s0 + 2 * BC : s0 + 3 * BC, bh, ik0:ik1], dk_rows)
                T.tile.add(dk2_ub, dk2b_ub, dkt_ub2)
                T.tile.add(dk2_ub, dk2_ub, dk_rows)
                T.copy(dk2_ub, dk2[bb, s0 + 2 * BC : s0 + 3 * BC, bh, ik0:ik1])
                # s = 3 (rows 48..63)
                T.tile.broadcast(d_s, gn_s3, axis=0)
                T.tile.sub(d_s, g_ub[3 * BC : 4 * BC, 0:BK], d_s)
                T.tile.mul(d_s, d_s, LN2)
                T.tile.exp(d_s, d_s)
                T.tile.broadcast(v_s, gnp_s3, axis=0)
                T.tile.sub(v_s, v_s, g_ub[3 * BC : 4 * BC, 0:BK])
                T.tile.mul(v_s, v_s, LN2)
                T.tile.exp(v_s, v_s)
                T.copy(dq2_frag3, dq2pre_ub3)
                T.tile.mul(dq2_ub, dq2pre_ub3, d_s)
                T.copy(dq[bb, s0 + 3 * BC : s0 + 4 * BC, bh, ik0:ik1], dq_rows)
                T.tile.add(dq2_ub, dq2_ub, dq_rows)
                T.copy(dq2_ub, dq2[bb, s0 + 3 * BC : s0 + 4 * BC, bh, ik0:ik1])
                T.copy(dk2p_frag3, dk2pre_ub3)
                T.tile.mul(r_tmp, dk2pre_ub3, k_ub[3 * BC : 4 * BC, 0:BK])
                T.reduce_sum(r_tmp, r_part, dim=1, clear=True)
                T.tile.add(db_acc[3 * BC : 4 * BC], db_acc[3 * BC : 4 * BC], r_part)
                T.tile.broadcast(bcol2d, bet[3 * BC : 4 * BC], axis=1)
                T.tile.mul(dk2b_ub, dk2pre_ub3, bcol2d)
                T.copy(dkt_frag3, dkt_ub3)
                T.tile.sub(t16b, dk2b_ub, dkt_ub3)
                T.tile.mul(t16b, t16b, k_ub[3 * BC : 4 * BC, 0:BK])
                T.tile.mul(dg2_ub, q_ub[3 * BC : 4 * BC, 0:BK], dq2pre_ub3)
                T.tile.add(dg2_ub, dg2_ub, t16b)
                T.copy(dg[bb, s0 + 3 * BC : s0 + 4 * BC, bh, ik0:ik1], dg_rows)
                T.tile.add(dg2_ub, dg2_ub, dg_rows)
                T.copy(dg2_ub, dg2[bb, s0 + 3 * BC : s0 + 4 * BC, bh, ik0:ik1])
                T.copy(dk[bb, s0 + 3 * BC : s0 + 4 * BC, bh, ik0:ik1], dk_rows)
                T.tile.add(dk2_ub, dk2b_ub, dkt_ub3)
                T.tile.add(dk2_ub, dk2_ub, dk_rows)
                T.copy(dk2_ub, dk2[bb, s0 + 3 * BC : s0 + 4 * BC, bh, ik0:ik1])

            # ---- epilogue: db2 scatter via GM scratch + flag chain ----
            T.copy(db_acc, db2flat[cid, 0:BS])
            T.set_flag("MTE3", "MTE2", 0)
            T.wait_flag("MTE3", "MTE2", 0)
            T.copy(db2flat_v[cid, 0:BS, 0:1], db2d)
            T.copy(db2d, db2[bb, s0 : s0 + BS, bh])

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
        raise ValueError(
            f"DK={DK} unsupported: must be one of {{64, 96, 128}}"
            " (bwd exact-2 Pipelined block constraint)"
        )
    if H < 1 or B < 1:
        raise ValueError("B/H must be >= 1")


_MNEG_CACHE = {}


def _get_mneg(device, bc):
    """-strict_tril(ones) constant, constructed once per (device, size)."""
    key = (str(device), bc)
    if key not in _MNEG_CACHE:
        _MNEG_CACHE[key] = (
            -torch.tril(torch.ones(bc, bc, dtype=torch.float32), -1)
        ).to(device)
    return _MNEG_CACHE[key]


_MLO_CACHE = {}


def _get_mlo(device, bs):
    """tril(ones, 0) constant, constructed once per (device, size)."""
    key = (str(device), bs)
    if key not in _MLO_CACHE:
        _MLO_CACHE[key] = torch.tril(
            torch.ones(bs, bs, dtype=torch.float32), 0
        ).to(device)
    return _MLO_CACHE[key]


_MEYE_CACHE = {}


def _get_meye(device, n):
    """Identity constant, constructed once per (device, size)."""
    key = (str(device), n)
    if key not in _MEYE_CACHE:
        _MEYE_CACHE[key] = torch.eye(n, dtype=torch.float32).to(device)
    return _MEYE_CACHE[key]


def fla_chunk_intra_fwd_inter_solve_fused(
    q, k, gk, beta, Akk_diag, chunk_size=64, sub_chunk_size=16, scale=1.0
):
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

    # cross-kernel GM workspaces (host torch.empty; fp32)
    ws = [torch.empty(B, S, H, chunk_size, dtype=torch.float32, device=device)
          for _ in range(5)]
    akkbp, g_ws, g2_ws, y_ws, aid_ws = ws

    k_inter = fla_chunk_intra_fwd_k_inter(B, S, H, DK, chunk_size, sub_chunk_size, scale)
    Aqk = k_inter(q, k, gk, beta_flat, akkbp)
    k_g = fla_chunk_intra_fwd_k_g(B, S, H, DK, chunk_size, sub_chunk_size)
    k_g(akkbp, Akk_diag, mneg, _get_meye(device, sub_chunk_size), aid_ws, g_ws)
    k_g2 = fla_chunk_intra_fwd_k_g2(B, S, H, DK, chunk_size, sub_chunk_size)
    k_g2(g_ws, g2_ws)
    k_y = fla_chunk_intra_fwd_k_y(B, S, H, DK, chunk_size, sub_chunk_size)
    k_y(g2_ws, aid_ws, _get_meye(device, chunk_size), y_ws)
    k_ai = fla_chunk_intra_fwd_k_ai(B, S, H, DK, chunk_size, sub_chunk_size)
    Akk = k_ai(g_ws, y_ws, _get_meye(device, chunk_size))
    # no internal synchronize: the do_bench harness owns synchronization
    return Aqk, Akk


def fla_chunk_intra_bwd_intra(
    q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, chunk_size=64, sub_chunk_size=16
):
    """bwd host API (upstream chunk_kda_bwd_intra).

    All fp32 -> (dq2, dk2, db2 [1,B,S,H] partial, dg2 pre-reverse-cumsum).
    Host post-processing (db2.sum(0)+db, reverse chunk cumsum of dg2) belongs
    to the caller, matching the upstream kernel/test boundary.
    """
    B, S, H, DK = k.shape
    _check_shape(B, S, H, DK, chunk_size, sub_chunk_size)
    for name, t in (("q", q), ("k", k), ("g", g), ("beta", beta),
                    ("dAqk", dAqk), ("dAkk", dAkk), ("dq", dq), ("dk", dk),
                    ("db", db), ("dg", dg)):
        if t.dtype is not torch.float32:
            raise ValueError(f"bwd input '{name}' must be float32")
        if not t.is_contiguous():
            raise ValueError(f"input '{name}' must be contiguous")

    device = k.device
    beta_flat = beta.view(B, S * H)  # zero-copy metadata view
    mlo = _get_mlo(device, chunk_size)
    grid = (S // chunk_size) * B * H
    db2flat = torch.empty(grid, chunk_size, dtype=torch.float32, device=device)
    db2flat_v = db2flat.view(grid, chunk_size, 1)  # zero-copy alias view

    kernel = fla_chunk_intra_bwd_k_intra(B, S, H, DK, chunk_size, sub_chunk_size)
    dq2, dk2, db2, dg2 = kernel(
        q, k, g, beta_flat, mlo, dAqk, dAkk, dq, dk, dg, db2flat, db2flat_v
    )
    # NK=1 partial (both DK halves accumulate in-program); zero-copy view to
    # honor the [NK,B,S,H] output contract
    return dq2, dk2, db2.view(1, B, S, H), dg2


__all__ = [
    "fla_chunk_intra_fwd_inter_solve_fused",
    "fla_chunk_intra_bwd_intra",
    "fla_chunk_intra_fwd_k_inter",
    "fla_chunk_intra_fwd_k_g",
    "fla_chunk_intra_fwd_k_g2",
    "fla_chunk_intra_fwd_k_y",
    "fla_chunk_intra_fwd_k_ai",
    "fla_chunk_intra_bwd_k_intra",
    "PASS_CONFIGS",
]
