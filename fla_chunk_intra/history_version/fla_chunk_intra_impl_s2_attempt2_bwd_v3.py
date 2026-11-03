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

bwd  (chunk_kda_bwd_intra):  TWO JIT kernels (Stage-2 attempt-2 structure).
The per-column sequential loops of the upstream (B)/(E) steps are
re-expressed as masked GEMMs (mask depends only on (i, j), never on
iteration state):
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
  tail (per s): db_s = rowsum(D_s*dk2pre_s * k_s)  [pre-beta, accumulated across
           halves into db_acc]; dk2b_s = D_s*dk2pre_s * beta_s;
           dq2_s = D_s*dq2_pre_s + dq;   dk2_s = dk2b_s + dk + dkt_s;
           dg2_s = q_s*(D_s*dq2_pre_s) + (dk2b_s - dkt_s)*k_s + dg
           [post-beta dk2, upstream L481-489 order.  NOTE: the D_s/V_s
           post-scalings on the dk2/dkt chains are load-bearing -- attempt 1
           computed them but never applied them, which by itself made the
           whole bwd wrong even before any toolchain issue.]
  host post-processing (NOT part of the kernel, not timed): db = db2.sum(0)+db,
           dg = reverse chunk-local cumsum(dg2).

  k_stage (pure Vector, straight-line, zero cross-core flags): builds every
    GEMM operand tile in UB (MdA/MdK row tiles [16, KMAX] + col tiles
    [64, 16], Kdec/Qdec/KBdec [KMAX or 64, BK], fp16 hi/lo splits) and writes
    them to the GM staging tensors OpscrM / Opscr.  The kernel boundary
    (host-ordered launch) provides the V->C synchronization.
  k_gemm (Pipelined(2, num_stages=1) over the DK halves; body = one C run
    then one V run): the C side reads every tile from GM into a whole-tile
    L1 buffer (Nd2Nz per tile), runs the 48 hi/lo gemm_v0 chains, writes the
    12 results L0C->GM directly (Fragscr, k1 precedent) and signals
    T.set_cross_flag("FIX", 9 + i_kk % 2); the V side waits
    T.wait_cross_flag(9 + i_kk % 2, "V"), reads the frags back and runs the
    corrected tail.  Cross-core flag budget: exactly 2 ffts IDs (9, 10).

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
8. bwd grid is NT*B*H with the DK halves iterated INSIDE k_gemm's Pipelined(2)
   body; MdA/MdK and all decay operands are staged to GM by the separate
   k_stage kernel, so nothing cross-core depends on a prologue relay.  db2 has
   NK=1 partial (the rowsums of both halves accumulate in-program); the
   DESIGN's [NK,B,S,H] output contract is honored via a zero-copy host view.
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
11. [Stage-2 attempt-2 finding] CANN 9.2.0 GetffstMsg masks the cross-core
   flag id with 0xf: only 16 ffts event slots exist and 11..14 are
   system-reserved (SYNC_AIC/AIV/AIC_AIV/ALL), so at most ~11 user flags
   survive.  The attempt-1 single-kernel bwd generated 40 distinct ids (28
   UB->L1 relay sites + 12 C->V relay sites) -- every id >= 16 truncated onto
   lower ids, so the loop structure (V iter-1 writes vs C iter-0 reads)
   raced.  Manual flags exist (T.set_cross_flag / T.wait_cross_flag) but a
   user-side set on the "MTE3" pipe classifies kExplicit and T.Scope("V")
   wrappers inside a Pipelined body get flattened away by
   CrossCorePipeline's LoopAnalyzer, so a same-kernel V->C handshake is not
   expressible; the bwd therefore became TWO kernels with the kernel
   boundary as the V->C sync (k_stage / k_gemm), leaving exactly the 2
   C->V FIX flag ids 9+i_kk%2.
12. [Stage-2 attempt-2 finding] the implicit relay writes an L1 buffer as a
   zN tile shaped by the COPY (a whole-buffer copy -> the buffer's own
   shape), while gemm_v0 reads a zN tile shaped by the GEMM's template at
   the region's flat row-major offset.  The two zN layouts agree only when
   the relay copy's tile equals the gemm's operand tile (whole-buffer
   operands, or the copy region == the gemm region); in particular a
   row-offset/narrow-column region read of a whole-relayed buffer mixes
   mismatched zN column-band strides (roundUp16(tile rows) vs
   roundUp16(buffer rows)) and reads garbage.  Attempt-1's region operands
   (mda_hi_l1[BC:2*BC, ...] etc.) all violated this and would have been
   wrong even with perfect flags.  The v3 design keeps every gemm operand
   a WHOLE L1 buffer fed by an exactly-shaped GM->L1 Nd2Nz read.
13. [Stage-2 attempt-2 finding] L0C->GM direct fp32 stores (k1 precedent,
   T.copy(frag, gm_tensor)) emit zero cross-core flags, and GM tensors whose
   name lacks the "workspace" substring never get AUTO_CV_SYNC relay flags,
   so Fragscr/OpscrM/Opscr handoffs cost nothing from the 16-slot budget.
14. [Stage-2 attempt-2 finding] narrow-row GM/UB copies round-trip through
   32B slots: a [64,1]-region GM->UB read scatters its 4B bursts 32B apart
   (the MTE "narrow rows sit in 32B slots" convention, issue-#1682 class)
   and the matching [64,1] UB->GM write reads them back from the same slots
   -- the star-chain pair only self-compensates while the final GM row
   stride exceeds 4B (H >= 2).  At H == 1 (contiguous slice) the write goes
   flat and the mismatch corrupts db2 (and the scattered reads can clobber
   UB neighbors unless the dst buffer is at least 64*32B wide -- hence the
   [BS,8] db2d).  H == 1 therefore writes db2 as a single 256B row through a
   flat [B, S*H] view (host-allocated db2 + zero-copy db2f), which has no
   narrow rows at all.
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
# bwd kernel 1 of 2 (pure Vector, no Cube segment, no cross-core flags):
#   stage every GEMM operand tile to GM scratch (OpscrM / Opscr).
#   The kernel boundary (host-ordered launches) replaces the in-kernel V->C
#   synchronization that the 16-slot ffts flag space cannot express for a
#   40-relay operator (module docstring item 11).
# GM layouts (fp16; per (chunk, bh) program, cid = chunk-major):
#   OpscrM [GRID, 1280, 64]  (half-independent Md tiles):
#     row tiles (rows [s*16:+16, 0:KMAX_s]):
#       s=0: mda_hi @0   mda_lo @16  mdk_hi @32  mdk_lo @48   (KMAX=32)
#       s=1:             @64          @80          @96         @112
#       s=2:             @128         @144         @160        @176      (KMAX=48)
#       s=3:             @192         @208         @224        @240      (KMAX=64)
#     col tiles (rows [256 + s*256 + v*64 : +64, 0:16]):
#       v: 0=mda_hi 1=mda_lo 2=mdk_hi 3=mdk_lo
#   Opscr [GRID, 2, 1376, BK]  (per-half decay operands):
#     per s: kdec_hi@off  kdec_lo@off+KMAX  qdec_hi@+2K  qdec_lo@+3K
#            kbdec_hi@+4K  kbdec_lo@+5K     (K=64 rows for the q/kb tiles)
#       s=0 (KMAX=32): 0    32   64   128  192  256
#       s=1 (KMAX=32): 320  352  384  448  512  576
#       s=2 (KMAX=48): 640  688  736  800  864  928
#       s=3 (KMAX=64): 992  1056 1120 1184 1248 1312
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
        OpscrM: T.Tensor((GRID, 16 * BC + 16 * BS, BS), "float16"),  # type: ignore
        Opscr: T.Tensor((GRID, 2, 86 * BS, BK), "float16"),  # type: ignore
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
            # ---- Md row-tile staging (per KMAX shape class) ----
            row32_f = T.alloc_shared([BC, 32], "float32")
            row32_hi = T.alloc_shared([BC, 32], "float16")
            row32_lo = T.alloc_shared([BC, 32], "float16")
            row32_hf = T.alloc_shared([BC, 32], "float32")
            row32_lf = T.alloc_shared([BC, 32], "float32")
            row48_f = T.alloc_shared([BC, 48], "float32")
            row48_hi = T.alloc_shared([BC, 48], "float16")
            row48_lo = T.alloc_shared([BC, 48], "float16")
            row48_hf = T.alloc_shared([BC, 48], "float32")
            row48_lf = T.alloc_shared([BC, 48], "float32")
            row64_f = T.alloc_shared([BC, 64], "float32")
            row64_hi = T.alloc_shared([BC, 64], "float16")
            row64_lo = T.alloc_shared([BC, 64], "float16")
            row64_hf = T.alloc_shared([BC, 64], "float32")
            row64_lf = T.alloc_shared([BC, 64], "float32")
            # ---- Md col-tile staging ----
            col_f = T.alloc_shared([BS, BC], "float32")
            col_hi = T.alloc_shared([BS, BC], "float16")
            col_lo = T.alloc_shared([BS, BC], "float16")
            col_hf = T.alloc_shared([BS, BC], "float32")
            col_lf = T.alloc_shared([BS, BC], "float32")
            # ---- per-half decay staging ----
            k_ub = T.alloc_shared([BS, BK], "float32")
            g_ub = T.alloc_shared([BS, BK], "float32")
            q_ub = T.alloc_shared([BS, BK], "float32")
            t16 = T.alloc_shared([BC, BK], "float32")
            bcol2d = T.alloc_shared([BC, BK], "float32")
            odec_f = T.alloc_shared([BS, BK], "float32")
            sec_hi = T.alloc_shared([BS, BK], "float16")
            sec_lo = T.alloc_shared([BS, BK], "float16")
            sec_hf = T.alloc_shared([BS, BK], "float32")
            sec_lf = T.alloc_shared([BS, BK], "float32")
            gn_s0 = T.alloc_shared([BK], "float32")
            gnp_s0 = T.alloc_shared([BK], "float32")
            gn_s1 = T.alloc_shared([BK], "float32")
            gnp_s1 = T.alloc_shared([BK], "float32")
            gn_s2 = T.alloc_shared([BK], "float32")
            gnp_s2 = T.alloc_shared([BK], "float32")
            gn_s3 = T.alloc_shared([BK], "float32")
            gnp_s3 = T.alloc_shared([BK], "float32")

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

            # ==== Md row tiles: mda_f[s*16:+16, 0:KMAX] -> OpscrM ====
            # s = 0 (KMAX = 32): mda_hi @0, mda_lo @16, mdk_hi @32, mdk_lo @48
            T.copy(mda_f[0:BC, 0:32], row32_f)
            T.copy(row32_f, row32_hi)
            T.copy(row32_hi, row32_hf)
            T.tile.sub(row32_lf, row32_f, row32_hf)
            T.copy(row32_hi, OpscrM[cid, 0:16, 0:32])
            T.copy(row32_lf, row32_lo)
            T.copy(row32_lo, OpscrM[cid, 16:32, 0:32])
            T.copy(mdk_f[0:BC, 0:32], row32_f)
            T.copy(row32_f, row32_hi)
            T.copy(row32_hi, row32_hf)
            T.tile.sub(row32_lf, row32_f, row32_hf)
            T.copy(row32_hi, OpscrM[cid, 32:48, 0:32])
            T.copy(row32_lf, row32_lo)
            T.copy(row32_lo, OpscrM[cid, 48:64, 0:32])
            # s = 1 (KMAX = 32): @64, 80, 96, 112
            T.copy(mda_f[BC : 2 * BC, 0:32], row32_f)
            T.copy(row32_f, row32_hi)
            T.copy(row32_hi, row32_hf)
            T.tile.sub(row32_lf, row32_f, row32_hf)
            T.copy(row32_hi, OpscrM[cid, 64:80, 0:32])
            T.copy(row32_lf, row32_lo)
            T.copy(row32_lo, OpscrM[cid, 80:96, 0:32])
            T.copy(mdk_f[BC : 2 * BC, 0:32], row32_f)
            T.copy(row32_f, row32_hi)
            T.copy(row32_hi, row32_hf)
            T.tile.sub(row32_lf, row32_f, row32_hf)
            T.copy(row32_hi, OpscrM[cid, 96:112, 0:32])
            T.copy(row32_lf, row32_lo)
            T.copy(row32_lo, OpscrM[cid, 112:128, 0:32])
            # s = 2 (KMAX = 48): @128, 144, 160, 176
            T.copy(mda_f[2 * BC : 3 * BC, 0:48], row48_f)
            T.copy(row48_f, row48_hi)
            T.copy(row48_hi, row48_hf)
            T.tile.sub(row48_lf, row48_f, row48_hf)
            T.copy(row48_hi, OpscrM[cid, 128:144, 0:48])
            T.copy(row48_lf, row48_lo)
            T.copy(row48_lo, OpscrM[cid, 144:160, 0:48])
            T.copy(mdk_f[2 * BC : 3 * BC, 0:48], row48_f)
            T.copy(row48_f, row48_hi)
            T.copy(row48_hi, row48_hf)
            T.tile.sub(row48_lf, row48_f, row48_hf)
            T.copy(row48_hi, OpscrM[cid, 160:176, 0:48])
            T.copy(row48_lf, row48_lo)
            T.copy(row48_lo, OpscrM[cid, 176:192, 0:48])
            # s = 3 (KMAX = 64): @192, 208, 224, 240
            T.copy(mda_f[3 * BC : 4 * BC, 0:64], row64_f)
            T.copy(row64_f, row64_hi)
            T.copy(row64_hi, row64_hf)
            T.tile.sub(row64_lf, row64_f, row64_hf)
            T.copy(row64_hi, OpscrM[cid, 192:208, 0:64])
            T.copy(row64_lf, row64_lo)
            T.copy(row64_lo, OpscrM[cid, 208:224, 0:64])
            T.copy(mdk_f[3 * BC : 4 * BC, 0:64], row64_f)
            T.copy(row64_f, row64_hi)
            T.copy(row64_hi, row64_hf)
            T.tile.sub(row64_lf, row64_f, row64_hf)
            T.copy(row64_hi, OpscrM[cid, 224:240, 0:64])
            T.copy(row64_lf, row64_lo)
            T.copy(row64_lo, OpscrM[cid, 240:256, 0:64])

            # ==== Md col tiles: mda_f[0:64, s*16:+16] -> OpscrM @256+s*256 ====
            # s = 0: mda_hi @256, mda_lo @320, mdk_hi @384, mdk_lo @448
            T.copy(mda_f[0:BS, 0:BC], col_f)
            T.copy(col_f, col_hi)
            T.copy(col_hi, col_hf)
            T.tile.sub(col_lf, col_f, col_hf)
            T.copy(col_hi, OpscrM[cid, 256:320, 0:16])
            T.copy(col_lf, col_lo)
            T.copy(col_lo, OpscrM[cid, 320:384, 0:16])
            T.copy(mdk_f[0:BS, 0:BC], col_f)
            T.copy(col_f, col_hi)
            T.copy(col_hi, col_hf)
            T.tile.sub(col_lf, col_f, col_hf)
            T.copy(col_hi, OpscrM[cid, 384:448, 0:16])
            T.copy(col_lf, col_lo)
            T.copy(col_lo, OpscrM[cid, 448:512, 0:16])
            # s = 1: @512, 576, 640, 704
            T.copy(mda_f[0:BS, BC : 2 * BC], col_f)
            T.copy(col_f, col_hi)
            T.copy(col_hi, col_hf)
            T.tile.sub(col_lf, col_f, col_hf)
            T.copy(col_hi, OpscrM[cid, 512:576, 0:16])
            T.copy(col_lf, col_lo)
            T.copy(col_lo, OpscrM[cid, 576:640, 0:16])
            T.copy(mdk_f[0:BS, BC : 2 * BC], col_f)
            T.copy(col_f, col_hi)
            T.copy(col_hi, col_hf)
            T.tile.sub(col_lf, col_f, col_hf)
            T.copy(col_hi, OpscrM[cid, 640:704, 0:16])
            T.copy(col_lf, col_lo)
            T.copy(col_lo, OpscrM[cid, 704:768, 0:16])
            # s = 2: @768, 832, 896, 960
            T.copy(mda_f[0:BS, 2 * BC : 3 * BC], col_f)
            T.copy(col_f, col_hi)
            T.copy(col_hi, col_hf)
            T.tile.sub(col_lf, col_f, col_hf)
            T.copy(col_hi, OpscrM[cid, 768:832, 0:16])
            T.copy(col_lf, col_lo)
            T.copy(col_lo, OpscrM[cid, 832:896, 0:16])
            T.copy(mdk_f[0:BS, 2 * BC : 3 * BC], col_f)
            T.copy(col_f, col_hi)
            T.copy(col_hi, col_hf)
            T.tile.sub(col_lf, col_f, col_hf)
            T.copy(col_hi, OpscrM[cid, 896:960, 0:16])
            T.copy(col_lf, col_lo)
            T.copy(col_lo, OpscrM[cid, 960:1024, 0:16])
            # s = 3: @1024, 1088, 1152, 1216
            T.copy(mda_f[0:BS, 3 * BC : 4 * BC], col_f)
            T.copy(col_f, col_hi)
            T.copy(col_hi, col_hf)
            T.tile.sub(col_lf, col_f, col_hf)
            T.copy(col_hi, OpscrM[cid, 1024:1088, 0:16])
            T.copy(col_lf, col_lo)
            T.copy(col_lo, OpscrM[cid, 1088:1152, 0:16])
            T.copy(mdk_f[0:BS, 3 * BC : 4 * BC], col_f)
            T.copy(col_f, col_hi)
            T.copy(col_hi, col_hf)
            T.tile.sub(col_lf, col_f, col_hf)
            T.copy(col_hi, OpscrM[cid, 1152:1216, 0:16])
            T.copy(col_lf, col_lo)
            T.copy(col_lo, OpscrM[cid, 1216:1280, 0:16])

            # ==== per-half decay operands (pure-V serial loop) ====
            for i_kk in T.serial(0, 2):
                ik0 = i_kk * BK
                ik1 = ik0 + BK
                T.copy(K[bb, s0 : s0 + BS, bh, ik0:ik1], k_ub)
                T.copy(G[bb, s0 : s0 + BS, bh, ik0:ik1], g_ub)
                T.copy(Q[bb, s0 : s0 + BS, bh, ik0:ik1], q_ub)
                T.copy(g_ub[0, 0:BK], gn_s0)
                T.copy(g_ub[BC - 1, 0:BK], gnp_s0)
                T.copy(g_ub[BC, 0:BK], gn_s1)
                T.copy(g_ub[2 * BC - 1, 0:BK], gnp_s1)
                T.copy(g_ub[2 * BC, 0:BK], gn_s2)
                T.copy(g_ub[3 * BC - 1, 0:BK], gnp_s2)
                T.copy(g_ub[3 * BC, 0:BK], gn_s3)
                T.copy(g_ub[4 * BC - 1, 0:BK], gnp_s3)

                # ---- s = 0 (KMAX = 32): kdec@0/32, qdec@64/128, kbdec@192/256
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
                T.copy(odec_f, sec_hi)
                T.copy(sec_hi, sec_hf)
                T.tile.sub(sec_lf, odec_f, sec_hf)
                T.copy(sec_hi[0:32, 0:BK], Opscr[cid, i_kk, 0:32, 0:BK])
                T.copy(sec_lf, sec_lo)
                T.copy(sec_lo[0:32, 0:BK], Opscr[cid, i_kk, 32:64, 0:BK])
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
                T.copy(odec_f, sec_hi)
                T.copy(sec_hi, sec_hf)
                T.tile.sub(sec_lf, odec_f, sec_hf)
                T.copy(sec_hi, Opscr[cid, i_kk, 64:128, 0:BK])
                T.copy(sec_lf, sec_lo)
                T.copy(sec_lo, Opscr[cid, i_kk, 128:192, 0:BK])
                # KBdec_0 (4 chunks): (beta*k) * 2^(g - gnp_s0)
                T.tile.broadcast(t16, gnp_s0, axis=0)
                T.tile.sub(t16, g_ub[0:BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.broadcast(bcol2d, bet[0:BC], axis=1)
                T.tile.mul(odec_f[0:BC, 0:BK], k_ub[0:BC, 0:BK], bcol2d)
                T.tile.mul(odec_f[0:BC, 0:BK], odec_f[0:BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s0, axis=0)
                T.tile.sub(t16, g_ub[BC : 2 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.broadcast(bcol2d, bet[BC : 2 * BC], axis=1)
                T.tile.mul(odec_f[BC : 2 * BC, 0:BK], k_ub[BC : 2 * BC, 0:BK], bcol2d)
                T.tile.mul(odec_f[BC : 2 * BC, 0:BK], odec_f[BC : 2 * BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s0, axis=0)
                T.tile.sub(t16, g_ub[2 * BC : 3 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.broadcast(bcol2d, bet[2 * BC : 3 * BC], axis=1)
                T.tile.mul(odec_f[2 * BC : 3 * BC, 0:BK], k_ub[2 * BC : 3 * BC, 0:BK], bcol2d)
                T.tile.mul(odec_f[2 * BC : 3 * BC, 0:BK], odec_f[2 * BC : 3 * BC, 0:BK], t16)
                T.tile.broadcast(t16, gnp_s0, axis=0)
                T.tile.sub(t16, g_ub[3 * BC : 4 * BC, 0:BK], t16)
                T.tile.mul(t16, t16, LN2)
                T.tile.exp(t16, t16)
                T.tile.broadcast(bcol2d, bet[3 * BC : 4 * BC], axis=1)
                T.tile.mul(odec_f[3 * BC : 4 * BC, 0:BK], k_ub[3 * BC : 4 * BC, 0:BK], bcol2d)
                T.tile.mul(odec_f[3 * BC : 4 * BC, 0:BK], odec_f[3 * BC : 4 * BC, 0:BK], t16)
                T.copy(odec_f, sec_hi)
                T.copy(sec_hi, sec_hf)
                T.tile.sub(sec_lf, odec_f, sec_hf)
                T.copy(sec_hi, Opscr[cid, i_kk, 192:256, 0:BK])
                T.copy(sec_lf, sec_lo)
                T.copy(sec_lo, Opscr[cid, i_kk, 256:320, 0:BK])

                # ---- s = 1 (KMAX = 32): kdec@320/352, qdec@384/448, kbdec@512/576
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
                T.copy(odec_f, sec_hi)
                T.copy(sec_hi, sec_hf)
                T.tile.sub(sec_lf, odec_f, sec_hf)
                T.copy(sec_hi[0:32, 0:BK], Opscr[cid, i_kk, 320:352, 0:BK])
                T.copy(sec_lf, sec_lo)
                T.copy(sec_lo[0:32, 0:BK], Opscr[cid, i_kk, 352:384, 0:BK])
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
                T.copy(odec_f, sec_hi)
                T.copy(sec_hi, sec_hf)
                T.tile.sub(sec_lf, odec_f, sec_hf)
                T.copy(sec_hi, Opscr[cid, i_kk, 384:448, 0:BK])
                T.copy(sec_lf, sec_lo)
                T.copy(sec_lo, Opscr[cid, i_kk, 448:512, 0:BK])
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
                T.copy(odec_f, sec_hi)
                T.copy(sec_hi, sec_hf)
                T.tile.sub(sec_lf, odec_f, sec_hf)
                T.copy(sec_hi, Opscr[cid, i_kk, 512:576, 0:BK])
                T.copy(sec_lf, sec_lo)
                T.copy(sec_lo, Opscr[cid, i_kk, 576:640, 0:BK])

                # ---- s = 2 (KMAX = 48): kdec@640/688, qdec@736/800, kbdec@864/928
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
                T.copy(odec_f, sec_hi)
                T.copy(sec_hi, sec_hf)
                T.tile.sub(sec_lf, odec_f, sec_hf)
                T.copy(sec_hi[0:48, 0:BK], Opscr[cid, i_kk, 640:688, 0:BK])
                T.copy(sec_lf, sec_lo)
                T.copy(sec_lo[0:48, 0:BK], Opscr[cid, i_kk, 688:736, 0:BK])
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
                T.copy(odec_f, sec_hi)
                T.copy(sec_hi, sec_hf)
                T.tile.sub(sec_lf, odec_f, sec_hf)
                T.copy(sec_hi, Opscr[cid, i_kk, 736:800, 0:BK])
                T.copy(sec_lf, sec_lo)
                T.copy(sec_lo, Opscr[cid, i_kk, 800:864, 0:BK])
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
                T.copy(odec_f, sec_hi)
                T.copy(sec_hi, sec_hf)
                T.tile.sub(sec_lf, odec_f, sec_hf)
                T.copy(sec_hi, Opscr[cid, i_kk, 864:928, 0:BK])
                T.copy(sec_lf, sec_lo)
                T.copy(sec_lo, Opscr[cid, i_kk, 928:992, 0:BK])

                # ---- s = 3 (KMAX = 64): kdec@992/1056, qdec@1120/1184, kbdec@1248/1312
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
                T.copy(odec_f, sec_hi)
                T.copy(sec_hi, sec_hf)
                T.tile.sub(sec_lf, odec_f, sec_hf)
                T.copy(sec_hi[0:64, 0:BK], Opscr[cid, i_kk, 992:1056, 0:BK])
                T.copy(sec_lf, sec_lo)
                T.copy(sec_lo[0:64, 0:BK], Opscr[cid, i_kk, 1056:1120, 0:BK])
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
                T.copy(odec_f, sec_hi)
                T.copy(sec_hi, sec_hf)
                T.tile.sub(sec_lf, odec_f, sec_hf)
                T.copy(sec_hi, Opscr[cid, i_kk, 1120:1184, 0:BK])
                T.copy(sec_lf, sec_lo)
                T.copy(sec_lo, Opscr[cid, i_kk, 1184:1248, 0:BK])
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
                T.copy(odec_f, sec_hi)
                T.copy(sec_hi, sec_hf)
                T.tile.sub(sec_lf, odec_f, sec_hf)
                T.copy(sec_hi, Opscr[cid, i_kk, 1248:1312, 0:BK])
                T.copy(sec_lf, sec_lo)
                T.copy(sec_lo, Opscr[cid, i_kk, 1312:1376, 0:BK])

    return main


# ============================================================================
# bwd kernel 2 of 2: gemms + tail.  Pipelined(2, num_stages=1) over the DK
# halves; body = one C run then one V run.
#   C: GM->L1 per-tile Nd2Nz reads (OpscrM/Opscr sections, whole-tile L1
#      buffers -- the zN tile layout always matches the gemm operand tile),
#      48 gemm_v0 (16 logical x 3 hi/lo), L0C->GM direct fp32 frag writes
#      (Fragscr, k1-precedent pattern), T.set_cross_flag("FIX", 9 + i_kk%2).
#   V: T.wait_cross_flag(9 + i_kk%2, "V"), GM->UB frag reads, the corrected
#      tail (D_s / V_s post-scalings applied -- the attempt-1 tail read the
#      gemm results raw), output writes, db_acc accumulation.
# Cross-core flag budget: exactly 2 IDs (9, 10) -- no implicit relays, no
# workspace-named GM tensors, so the AUTO_CV_SYNC pass adds none.
# Fragscr layout (per (cid, half), rows): s*48 + {0: dq2, 16: dk2p, 32: dkt}.
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
        OpscrM: T.Tensor((GRID, 16 * BC + 16 * BS, BS), "float16"),  # type: ignore
        Opscr: T.Tensor((GRID, 2, 86 * BS, BK), "float16"),  # type: ignore
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

            # ---- L1 tiles: Md row tiles [16, KMAX_s] ----
            mda_rh0 = T.alloc_shared([BC, 32], "float16")
            mda_rl0 = T.alloc_shared([BC, 32], "float16")
            mdk_rh0 = T.alloc_shared([BC, 32], "float16")
            mdk_rl0 = T.alloc_shared([BC, 32], "float16")
            mda_rh1 = T.alloc_shared([BC, 32], "float16")
            mda_rl1 = T.alloc_shared([BC, 32], "float16")
            mdk_rh1 = T.alloc_shared([BC, 32], "float16")
            mdk_rl1 = T.alloc_shared([BC, 32], "float16")
            mda_rh2 = T.alloc_shared([BC, 48], "float16")
            mda_rl2 = T.alloc_shared([BC, 48], "float16")
            mdk_rh2 = T.alloc_shared([BC, 48], "float16")
            mdk_rl2 = T.alloc_shared([BC, 48], "float16")
            mda_rh3 = T.alloc_shared([BC, 64], "float16")
            mda_rl3 = T.alloc_shared([BC, 64], "float16")
            mdk_rh3 = T.alloc_shared([BC, 64], "float16")
            mdk_rl3 = T.alloc_shared([BC, 64], "float16")
            # ---- L1 tiles: Md col tiles [64, 16] (transpose_A operands) ----
            mda_ch0 = T.alloc_shared([BS, BC], "float16")
            mda_cl0 = T.alloc_shared([BS, BC], "float16")
            mdk_ch0 = T.alloc_shared([BS, BC], "float16")
            mdk_cl0 = T.alloc_shared([BS, BC], "float16")
            mda_ch1 = T.alloc_shared([BS, BC], "float16")
            mda_cl1 = T.alloc_shared([BS, BC], "float16")
            mdk_ch1 = T.alloc_shared([BS, BC], "float16")
            mdk_cl1 = T.alloc_shared([BS, BC], "float16")
            mda_ch2 = T.alloc_shared([BS, BC], "float16")
            mda_cl2 = T.alloc_shared([BS, BC], "float16")
            mdk_ch2 = T.alloc_shared([BS, BC], "float16")
            mdk_cl2 = T.alloc_shared([BS, BC], "float16")
            mda_ch3 = T.alloc_shared([BS, BC], "float16")
            mda_cl3 = T.alloc_shared([BS, BC], "float16")
            mdk_ch3 = T.alloc_shared([BS, BC], "float16")
            mdk_cl3 = T.alloc_shared([BS, BC], "float16")
            # ---- L1 tiles: kdec [KMAX_s, BK] ----
            kdec_h0 = T.alloc_shared([32, BK], "float16")
            kdec_l0 = T.alloc_shared([32, BK], "float16")
            kdec_h1 = T.alloc_shared([32, BK], "float16")
            kdec_l1 = T.alloc_shared([32, BK], "float16")
            kdec_h2 = T.alloc_shared([48, BK], "float16")
            kdec_l2 = T.alloc_shared([48, BK], "float16")
            kdec_h3 = T.alloc_shared([64, BK], "float16")
            kdec_l3 = T.alloc_shared([64, BK], "float16")
            # ---- L1 tiles: qdec / kbdec [64, BK] ----
            qdec_h0 = T.alloc_shared([BS, BK], "float16")
            qdec_l0 = T.alloc_shared([BS, BK], "float16")
            kbdec_h0 = T.alloc_shared([BS, BK], "float16")
            kbdec_l0 = T.alloc_shared([BS, BK], "float16")
            qdec_h1 = T.alloc_shared([BS, BK], "float16")
            qdec_l1 = T.alloc_shared([BS, BK], "float16")
            kbdec_h1 = T.alloc_shared([BS, BK], "float16")
            kbdec_l1 = T.alloc_shared([BS, BK], "float16")
            qdec_h2 = T.alloc_shared([BS, BK], "float16")
            qdec_l2 = T.alloc_shared([BS, BK], "float16")
            kbdec_h2 = T.alloc_shared([BS, BK], "float16")
            kbdec_l2 = T.alloc_shared([BS, BK], "float16")
            qdec_h3 = T.alloc_shared([BS, BK], "float16")
            qdec_l3 = T.alloc_shared([BS, BK], "float16")
            kbdec_h3 = T.alloc_shared([BS, BK], "float16")
            kbdec_l3 = T.alloc_shared([BS, BK], "float16")
            # ---- fragments (L0C) ----
            dq2_fg0 = T.alloc_fragment([BC, BK], "float32")
            dk2_fg0 = T.alloc_fragment([BC, BK], "float32")
            dkt_fg0 = T.alloc_fragment([BC, BK], "float32")
            dq2_fg1 = T.alloc_fragment([BC, BK], "float32")
            dk2_fg1 = T.alloc_fragment([BC, BK], "float32")
            dkt_fg1 = T.alloc_fragment([BC, BK], "float32")
            dq2_fg2 = T.alloc_fragment([BC, BK], "float32")
            dk2_fg2 = T.alloc_fragment([BC, BK], "float32")
            dkt_fg2 = T.alloc_fragment([BC, BK], "float32")
            dq2_fg3 = T.alloc_fragment([BC, BK], "float32")
            dk2_fg3 = T.alloc_fragment([BC, BK], "float32")
            dkt_fg3 = T.alloc_fragment([BC, BK], "float32")
            # ---- V-side buffers ----
            g_ub = T.alloc_shared([BS, BK], "float32")
            k_ub = T.alloc_shared([BS, BK], "float32")
            q_ub = T.alloc_shared([BS, BK], "float32")
            beta_tile = T.alloc_shared([BS * H], "float32")
            off_i32 = T.alloc_shared([BS], "int32")
            off_u32 = T.alloc_shared([BS], "uint32")
            bet = T.alloc_shared([BS], "float32")
            gn_s0 = T.alloc_shared([BK], "float32")
            gnp_s0 = T.alloc_shared([BK], "float32")
            gn_s1 = T.alloc_shared([BK], "float32")
            gnp_s1 = T.alloc_shared([BK], "float32")
            gn_s2 = T.alloc_shared([BK], "float32")
            gnp_s2 = T.alloc_shared([BK], "float32")
            gn_s3 = T.alloc_shared([BK], "float32")
            gnp_s3 = T.alloc_shared([BK], "float32")
            f_dq2 = T.alloc_shared([BC, BK], "float32")
            f_dk2 = T.alloc_shared([BC, BK], "float32")
            f_dkt = T.alloc_shared([BC, BK], "float32")
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

                # ==== C: GM -> L1 tile reads (whole-tile Nd2Nz) ====
                # Md row tiles (half-independent; re-read per half)
                T.copy(OpscrM[cid, 0:16, 0:32], mda_rh0)
                T.copy(OpscrM[cid, 16:32, 0:32], mda_rl0)
                T.copy(OpscrM[cid, 32:48, 0:32], mdk_rh0)
                T.copy(OpscrM[cid, 48:64, 0:32], mdk_rl0)
                T.copy(OpscrM[cid, 64:80, 0:32], mda_rh1)
                T.copy(OpscrM[cid, 80:96, 0:32], mda_rl1)
                T.copy(OpscrM[cid, 96:112, 0:32], mdk_rh1)
                T.copy(OpscrM[cid, 112:128, 0:32], mdk_rl1)
                T.copy(OpscrM[cid, 128:144, 0:48], mda_rh2)
                T.copy(OpscrM[cid, 144:160, 0:48], mda_rl2)
                T.copy(OpscrM[cid, 160:176, 0:48], mdk_rh2)
                T.copy(OpscrM[cid, 176:192, 0:48], mdk_rl2)
                T.copy(OpscrM[cid, 192:208, 0:64], mda_rh3)
                T.copy(OpscrM[cid, 208:224, 0:64], mda_rl3)
                T.copy(OpscrM[cid, 224:240, 0:64], mdk_rh3)
                T.copy(OpscrM[cid, 240:256, 0:64], mdk_rl3)
                # Md col tiles
                T.copy(OpscrM[cid, 256:320, 0:16], mda_ch0)
                T.copy(OpscrM[cid, 320:384, 0:16], mda_cl0)
                T.copy(OpscrM[cid, 384:448, 0:16], mdk_ch0)
                T.copy(OpscrM[cid, 448:512, 0:16], mdk_cl0)
                T.copy(OpscrM[cid, 512:576, 0:16], mda_ch1)
                T.copy(OpscrM[cid, 576:640, 0:16], mda_cl1)
                T.copy(OpscrM[cid, 640:704, 0:16], mdk_ch1)
                T.copy(OpscrM[cid, 704:768, 0:16], mdk_cl1)
                T.copy(OpscrM[cid, 768:832, 0:16], mda_ch2)
                T.copy(OpscrM[cid, 832:896, 0:16], mda_cl2)
                T.copy(OpscrM[cid, 896:960, 0:16], mdk_ch2)
                T.copy(OpscrM[cid, 960:1024, 0:16], mdk_cl2)
                T.copy(OpscrM[cid, 1024:1088, 0:16], mda_ch3)
                T.copy(OpscrM[cid, 1088:1152, 0:16], mda_cl3)
                T.copy(OpscrM[cid, 1152:1216, 0:16], mdk_ch3)
                T.copy(OpscrM[cid, 1216:1280, 0:16], mdk_cl3)
                # kdec / qdec / kbdec tiles
                T.copy(Opscr[cid, i_kk, 0:32, 0:BK], kdec_h0)
                T.copy(Opscr[cid, i_kk, 32:64, 0:BK], kdec_l0)
                T.copy(Opscr[cid, i_kk, 64:128, 0:BK], qdec_h0)
                T.copy(Opscr[cid, i_kk, 128:192, 0:BK], qdec_l0)
                T.copy(Opscr[cid, i_kk, 192:256, 0:BK], kbdec_h0)
                T.copy(Opscr[cid, i_kk, 256:320, 0:BK], kbdec_l0)
                T.copy(Opscr[cid, i_kk, 320:352, 0:BK], kdec_h1)
                T.copy(Opscr[cid, i_kk, 352:384, 0:BK], kdec_l1)
                T.copy(Opscr[cid, i_kk, 384:448, 0:BK], qdec_h1)
                T.copy(Opscr[cid, i_kk, 448:512, 0:BK], qdec_l1)
                T.copy(Opscr[cid, i_kk, 512:576, 0:BK], kbdec_h1)
                T.copy(Opscr[cid, i_kk, 576:640, 0:BK], kbdec_l1)
                T.copy(Opscr[cid, i_kk, 640:688, 0:BK], kdec_h2)
                T.copy(Opscr[cid, i_kk, 688:736, 0:BK], kdec_l2)
                T.copy(Opscr[cid, i_kk, 736:800, 0:BK], qdec_h2)
                T.copy(Opscr[cid, i_kk, 800:864, 0:BK], qdec_l2)
                T.copy(Opscr[cid, i_kk, 864:928, 0:BK], kbdec_h2)
                T.copy(Opscr[cid, i_kk, 928:992, 0:BK], kbdec_l2)
                T.copy(Opscr[cid, i_kk, 992:1056, 0:BK], kdec_h3)
                T.copy(Opscr[cid, i_kk, 1056:1120, 0:BK], kdec_l3)
                T.copy(Opscr[cid, i_kk, 1120:1184, 0:BK], qdec_h3)
                T.copy(Opscr[cid, i_kk, 1184:1248, 0:BK], qdec_l3)
                T.copy(Opscr[cid, i_kk, 1248:1312, 0:BK], kbdec_h3)
                T.copy(Opscr[cid, i_kk, 1312:1376, 0:BK], kbdec_l3)

                # ==== C: 48 gemm_v0 (hi/lo 3-gemm chains) ====
                # s = 0 (KMAX = 32)
                T.gemm_v0(mda_rh0, kdec_h0, dq2_fg0, init=True)
                T.gemm_v0(mda_rl0, kdec_h0, dq2_fg0, init=False)
                T.gemm_v0(mda_rh0, kdec_l0, dq2_fg0, init=False)
                T.gemm_v0(mdk_rh0, kdec_h0, dk2_fg0, init=True)
                T.gemm_v0(mdk_rl0, kdec_h0, dk2_fg0, init=False)
                T.gemm_v0(mdk_rh0, kdec_l0, dk2_fg0, init=False)
                T.gemm_v0(mda_ch0, qdec_h0, dkt_fg0, transpose_A=True, init=True)
                T.gemm_v0(mda_cl0, qdec_h0, dkt_fg0, transpose_A=True, init=False)
                T.gemm_v0(mda_ch0, qdec_l0, dkt_fg0, transpose_A=True, init=False)
                T.gemm_v0(mdk_ch0, kbdec_h0, dkt_fg0, transpose_A=True, init=False)
                T.gemm_v0(mdk_cl0, kbdec_h0, dkt_fg0, transpose_A=True, init=False)
                T.gemm_v0(mdk_ch0, kbdec_l0, dkt_fg0, transpose_A=True, init=False)
                # s = 1 (KMAX = 32)
                T.gemm_v0(mda_rh1, kdec_h1, dq2_fg1, init=True)
                T.gemm_v0(mda_rl1, kdec_h1, dq2_fg1, init=False)
                T.gemm_v0(mda_rh1, kdec_l1, dq2_fg1, init=False)
                T.gemm_v0(mdk_rh1, kdec_h1, dk2_fg1, init=True)
                T.gemm_v0(mdk_rl1, kdec_h1, dk2_fg1, init=False)
                T.gemm_v0(mdk_rh1, kdec_l1, dk2_fg1, init=False)
                T.gemm_v0(mda_ch1, qdec_h1, dkt_fg1, transpose_A=True, init=True)
                T.gemm_v0(mda_cl1, qdec_h1, dkt_fg1, transpose_A=True, init=False)
                T.gemm_v0(mda_ch1, qdec_l1, dkt_fg1, transpose_A=True, init=False)
                T.gemm_v0(mdk_ch1, kbdec_h1, dkt_fg1, transpose_A=True, init=False)
                T.gemm_v0(mdk_cl1, kbdec_h1, dkt_fg1, transpose_A=True, init=False)
                T.gemm_v0(mdk_ch1, kbdec_l1, dkt_fg1, transpose_A=True, init=False)
                # s = 2 (KMAX = 48)
                T.gemm_v0(mda_rh2, kdec_h2, dq2_fg2, init=True)
                T.gemm_v0(mda_rl2, kdec_h2, dq2_fg2, init=False)
                T.gemm_v0(mda_rh2, kdec_l2, dq2_fg2, init=False)
                T.gemm_v0(mdk_rh2, kdec_h2, dk2_fg2, init=True)
                T.gemm_v0(mdk_rl2, kdec_h2, dk2_fg2, init=False)
                T.gemm_v0(mdk_rh2, kdec_l2, dk2_fg2, init=False)
                T.gemm_v0(mda_ch2, qdec_h2, dkt_fg2, transpose_A=True, init=True)
                T.gemm_v0(mda_cl2, qdec_h2, dkt_fg2, transpose_A=True, init=False)
                T.gemm_v0(mda_ch2, qdec_l2, dkt_fg2, transpose_A=True, init=False)
                T.gemm_v0(mdk_ch2, kbdec_h2, dkt_fg2, transpose_A=True, init=False)
                T.gemm_v0(mdk_cl2, kbdec_h2, dkt_fg2, transpose_A=True, init=False)
                T.gemm_v0(mdk_ch2, kbdec_l2, dkt_fg2, transpose_A=True, init=False)
                # s = 3 (KMAX = 64)
                T.gemm_v0(mda_rh3, kdec_h3, dq2_fg3, init=True)
                T.gemm_v0(mda_rl3, kdec_h3, dq2_fg3, init=False)
                T.gemm_v0(mda_rh3, kdec_l3, dq2_fg3, init=False)
                T.gemm_v0(mdk_rh3, kdec_h3, dk2_fg3, init=True)
                T.gemm_v0(mdk_rl3, kdec_h3, dk2_fg3, init=False)
                T.gemm_v0(mdk_rh3, kdec_l3, dk2_fg3, init=False)
                T.gemm_v0(mda_ch3, qdec_h3, dkt_fg3, transpose_A=True, init=True)
                T.gemm_v0(mda_cl3, qdec_h3, dkt_fg3, transpose_A=True, init=False)
                T.gemm_v0(mda_ch3, qdec_l3, dkt_fg3, transpose_A=True, init=False)
                T.gemm_v0(mdk_ch3, kbdec_h3, dkt_fg3, transpose_A=True, init=False)
                T.gemm_v0(mdk_cl3, kbdec_h3, dkt_fg3, transpose_A=True, init=False)
                T.gemm_v0(mdk_ch3, kbdec_l3, dkt_fg3, transpose_A=True, init=False)

                # ==== C: L0C -> GM direct frag writes + fix-done flag ====
                T.copy(dq2_fg0, Fragscr[cid, i_kk, 0:16, 0:BK])
                T.copy(dk2_fg0, Fragscr[cid, i_kk, 16:32, 0:BK])
                T.copy(dkt_fg0, Fragscr[cid, i_kk, 32:48, 0:BK])
                T.copy(dq2_fg1, Fragscr[cid, i_kk, 48:64, 0:BK])
                T.copy(dk2_fg1, Fragscr[cid, i_kk, 64:80, 0:BK])
                T.copy(dkt_fg1, Fragscr[cid, i_kk, 80:96, 0:BK])
                T.copy(dq2_fg2, Fragscr[cid, i_kk, 96:112, 0:BK])
                T.copy(dk2_fg2, Fragscr[cid, i_kk, 112:128, 0:BK])
                T.copy(dkt_fg2, Fragscr[cid, i_kk, 128:144, 0:BK])
                T.copy(dq2_fg3, Fragscr[cid, i_kk, 144:160, 0:BK])
                T.copy(dk2_fg3, Fragscr[cid, i_kk, 160:176, 0:BK])
                T.copy(dkt_fg3, Fragscr[cid, i_kk, 176:192, 0:BK])
                T.set_cross_flag("FIX", 9 + i_kk % 2)

                # ==== V: wait + loads + corrected tail ====
                T.copy(G[bb, s0 : s0 + BS, bh, ik0:ik1], g_ub)
                T.copy(K[bb, s0 : s0 + BS, bh, ik0:ik1], k_ub)
                T.copy(Q[bb, s0 : s0 + BS, bh, ik0:ik1], q_ub)
                T.copy(g_ub[0, 0:BK], gn_s0)
                T.copy(g_ub[BC - 1, 0:BK], gnp_s0)
                T.copy(g_ub[BC, 0:BK], gn_s1)
                T.copy(g_ub[2 * BC - 1, 0:BK], gnp_s1)
                T.copy(g_ub[2 * BC, 0:BK], gn_s2)
                T.copy(g_ub[3 * BC - 1, 0:BK], gnp_s2)
                T.copy(g_ub[3 * BC, 0:BK], gn_s3)
                T.copy(g_ub[4 * BC - 1, 0:BK], gnp_s3)
                T.wait_cross_flag(9 + i_kk % 2, "V")

                # ---- s = 0 (rows 0..15) ----
                T.tile.broadcast(d_s, gn_s0, axis=0)
                T.tile.sub(d_s, g_ub[0:BC, 0:BK], d_s)
                T.tile.mul(d_s, d_s, LN2)
                T.tile.exp(d_s, d_s)
                T.tile.broadcast(v_s, gnp_s0, axis=0)
                T.tile.sub(v_s, v_s, g_ub[0:BC, 0:BK])
                T.tile.mul(v_s, v_s, LN2)
                T.tile.exp(v_s, v_s)
                T.copy(Fragscr[cid, i_kk, 0:16, 0:BK], f_dq2)
                T.copy(Fragscr[cid, i_kk, 16:32, 0:BK], f_dk2)
                T.copy(Fragscr[cid, i_kk, 32:48, 0:BK], f_dkt)
                T.tile.mul(f_dq2, f_dq2, d_s)
                T.tile.mul(f_dk2, f_dk2, d_s)
                T.tile.mul(f_dkt, f_dkt, v_s)
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
                T.tile.broadcast(d_s, gn_s1, axis=0)
                T.tile.sub(d_s, g_ub[BC : 2 * BC, 0:BK], d_s)
                T.tile.mul(d_s, d_s, LN2)
                T.tile.exp(d_s, d_s)
                T.tile.broadcast(v_s, gnp_s1, axis=0)
                T.tile.sub(v_s, v_s, g_ub[BC : 2 * BC, 0:BK])
                T.tile.mul(v_s, v_s, LN2)
                T.tile.exp(v_s, v_s)
                T.copy(Fragscr[cid, i_kk, 48:64, 0:BK], f_dq2)
                T.copy(Fragscr[cid, i_kk, 64:80, 0:BK], f_dk2)
                T.copy(Fragscr[cid, i_kk, 80:96, 0:BK], f_dkt)
                T.tile.mul(f_dq2, f_dq2, d_s)
                T.tile.mul(f_dk2, f_dk2, d_s)
                T.tile.mul(f_dkt, f_dkt, v_s)
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
                T.tile.broadcast(d_s, gn_s2, axis=0)
                T.tile.sub(d_s, g_ub[2 * BC : 3 * BC, 0:BK], d_s)
                T.tile.mul(d_s, d_s, LN2)
                T.tile.exp(d_s, d_s)
                T.tile.broadcast(v_s, gnp_s2, axis=0)
                T.tile.sub(v_s, v_s, g_ub[2 * BC : 3 * BC, 0:BK])
                T.tile.mul(v_s, v_s, LN2)
                T.tile.exp(v_s, v_s)
                T.copy(Fragscr[cid, i_kk, 96:112, 0:BK], f_dq2)
                T.copy(Fragscr[cid, i_kk, 112:128, 0:BK], f_dk2)
                T.copy(Fragscr[cid, i_kk, 128:144, 0:BK], f_dkt)
                T.tile.mul(f_dq2, f_dq2, d_s)
                T.tile.mul(f_dk2, f_dk2, d_s)
                T.tile.mul(f_dkt, f_dkt, v_s)
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
                T.tile.broadcast(d_s, gn_s3, axis=0)
                T.tile.sub(d_s, g_ub[3 * BC : 4 * BC, 0:BK], d_s)
                T.tile.mul(d_s, d_s, LN2)
                T.tile.exp(d_s, d_s)
                T.tile.broadcast(v_s, gnp_s3, axis=0)
                T.tile.sub(v_s, v_s, g_ub[3 * BC : 4 * BC, 0:BK])
                T.tile.mul(v_s, v_s, LN2)
                T.tile.exp(v_s, v_s)
                T.copy(Fragscr[cid, i_kk, 144:160, 0:BK], f_dq2)
                T.copy(Fragscr[cid, i_kk, 160:176, 0:BK], f_dk2)
                T.copy(Fragscr[cid, i_kk, 176:192, 0:BK], f_dkt)
                T.tile.mul(f_dq2, f_dq2, d_s)
                T.tile.mul(f_dk2, f_dk2, d_s)
                T.tile.mul(f_dkt, f_dkt, v_s)
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
            # db2d is [BS, 8] (2KB): the [64,1]-region GM->UB read scatters its
            # 4B bursts into 32B slots (MTE "narrow rows sit in 32B slots"
            # convention, issue-#1682 class) and the [64,1] GM write reads
            # them back from the same slots -- the pair self-compensates, but
            # only while the final GM stride exceeds 4B (H >= 2); the wide
            # buffer keeps the scattered writes inside db2d (no UB-neighbor
            # clobber). H == 1 (contiguous slice) uses a flat 1-D round trip.
            if H == 1:
                # H=1: the db2 slice is CONTIGUOUS -- write it as a single
                # 256B row through the flat view (no narrow rows at all).
                T.copy(db_acc, db2f[bb, s0 : s0 + BS])
            else:
                # H>=2 star-chain (fla_wy_fast precedent): the [64,1]-region
                # GM->UB read scatters 4B bursts into 32B slots and the write
                # reads them back from the same slots (the pair compensates);
                # the flag covers the computed-source GM visibility.
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

    # cross-kernel GM workspaces (host torch.empty; fp32)
    ws = [torch.empty(B, S, H, chunk_size, dtype=torch.float32, device=device) for _ in range(5)]
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


def fla_chunk_intra_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, chunk_size=64, sub_chunk_size=16):
    """bwd host API (upstream chunk_kda_bwd_intra).

    All fp32 -> (dq2, dk2, db2 [1,B,S,H] partial, dg2 pre-reverse-cumsum).
    Host post-processing (db2.sum(0)+db, reverse chunk cumsum of dg2) belongs
    to the caller, matching the upstream kernel/test boundary.

    Two JIT kernels (Stage-2 v3 structure): k_stage (pure-V operand staging
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
    opscrm = torch.empty(grid, 16 * BC + 16 * BS, BS, dtype=torch.float16, device=device)
    opscr = torch.empty(grid, 2, 86 * BS, BK, dtype=torch.float16, device=device)
    fragscr = torch.empty(grid, 2, 12 * BC, BK, dtype=torch.float32, device=device)
    db2flat = torch.empty(grid, chunk_size, dtype=torch.float32, device=device)
    db2flat_v = db2flat.view(grid, chunk_size, 1)  # zero-copy alias view

    db2 = torch.empty(B, S, H, dtype=torch.float32, device=device)
    db2f = db2.view(B, S * H)  # zero-copy flat view (H=1 fast scatter path)
    k_stage = fla_chunk_intra_bwd_k_stage(B, S, H, DK, chunk_size, sub_chunk_size)
    k_stage(q, k, g, beta_flat, mlo, dAqk, dAkk, opscrm, opscr)
    k_gemm = fla_chunk_intra_bwd_k_gemm(B, S, H, DK, chunk_size, sub_chunk_size)
    dq2, dk2, dg2 = k_gemm(
        q, k, g, beta_flat, dq, dk, dg, opscrm, opscr, fragscr,
        db2flat, db2flat_v, db2, db2f
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
    "fla_chunk_intra_bwd_k_stage",
    "fla_chunk_intra_bwd_k_gemm",
    "PASS_CONFIGS",
]
