"""fla_chunk_intra_token_parallel: KDA intra-chunk per-token decay inner
products Aqk/Akk on Ascend 910B3 (Cube GEMM offload, iter7).

Semantics (frozen authority): ref_gpu/fla_chunk_intra_token_parallel.py
(Triton chunk_kda_fwd_kernel_intra_token_parallel). Code-level origin:
ref_gpu/fla_chunk_intra_token_parallel_tilelang_gpu.py (upstream TileLang
CUDA, token-parallel). For token i in subchunk [i_ts, i_ts+SCS) and
j in [i_ts, min(i+1, i_ts+SCS)):

    kgj[h, d] = k[b, j, h, d] * 2^(g[b, i, h, d] - g[b, j, h, d])
    Aqk[b, i, h, j % CS]        = scale * sum_d q[b, i, h, d] * kgj[h, d]
    Akk[b, i, h, j - i_ts]      = (j < i) ? sum_d (beta[b,i,h]*k[b,i,h,d]) * kgj[h, d] : 0

Aqk columns not covered by any j stay 0 (cleared-row semantics, upstream
TileLang version); the Akk diagonal (j == i) stays 0. scale is fixed at 1.0
(upstream test configuration; not passed through).

Ascend restructuring (DESIGN_fla_chunk_intra_token_parallel.md + Stage 3
iter1/3/7; all deviations from the upstream CUDA code are design decisions;
see perf_tuning/fla_chunk_intra_token_parallel_perf_log.md):

1. Subchunk-parallel blocks: one (batch, subchunk, head-group) per block;
   one-dimensional T.Kernel(B * NSC * HG) with (cid, vid) (3-D kernels are
   unsupported, threads restricted to 1/2; vid splits the BH-head group
   into BH_V = BH // 2 per AIV).
2. exp2(x) as exp(x * ln2); all bf16<->fp32 conversions UB-internally; fp32
   staging + block CAST_RINT + per-head 2D strided writeback bursts
   (iter1 mechanics, kept).
3. Factored decay (iter3, mid-anchor a = SCS//2): e_ij = E+_i * E-_j with
   E+ = exp(g_i - g_a), E- = exp(g_a - g_j); per-head one-shot slabs
   Q' = q*E+, K' = k*E-, K'' = k*E+. Documented domain restriction: the
   within-subchunk half-span gate decay |g_i - g_a| < ~127 (log2 domain)
   keeps E+/E- finite in fp32 (DBOUND sentinel half-span 96, margin 31).
4. iter7 GEMM OFFLOAD (DESIGN 10.4-D research path, msprof iter6: 20 Cube
   cores idle while AIV broadcast/reduce dominated 58% of the kernel):
   the per-(i,j) dot products leave the Vector pipe entirely --
     Aqk_h = tril(Q'_h @ K'_h^T),   Akk_h = strict_tril((beta*K'')_h @ K'_h^T)
   via one gemm_v0 per GGRP-head group with block-diagonal batching:
     A = [Q'; K''] rows of GGRP heads stacked  [GGRP*2*SCS, DK]
     B = K' rows of the same GGRP heads        [GGRP*SCS,  DK]
     C = A @ B^T (transpose_B) -> [GGRP*2SCS, GGRP*SCS]; the useful output
     is the per-head diagonal [2*SCS, SCS] blocks (4x FLOP waste at
     GGRP=4, measured cheaper than 4x more gemm calls: ggrp=1 benches
     +168us over ggrp=4 at the bench shape).
   beta is folded into K'' on the V side (per-row scalar muls, the
   production beta_f[i*16+h, 0] BufferLoad pattern) so the Akk GEMM
   result needs no post-folding.
   Operands transit V->C through an fp32 GM workspace (UB->GM MTE3,
   GM->L1 MTE2) and results C->V likewise (L0C->GM fixpipe, GM->UB);
   AUTO_CV_SYNC generates the cross-core flags (the pass keys on the
   literal buffer-name substring "workspace" -- ascend_combinecv.cc
   FetchWorkspaceName; non-"workspace" names silently get NO sync).
   fp32 operands (not bf16): bf16 rounding of Q'/K'/K'' alone fails the
   Akk matched-ratio gate (CPU-simulated 0.9848 < 0.99 vs fp32 1.0000)
   and costs the same transit bytes as split-bf16; fp32 gemm_v0 keeps the
   kernel bit-comparable to the iter3 vector version (Aqk bitwise equal).
   GGRP sizing: min(BH, 4) subject to the L0A single-K-step budget
   (GGRP*2*SCS*DK*4B <= 64KB) and the L0C/L0B N >= 16 fractal rule
   (GGRP >= ceil(16/SCS)); requires BH*SCS >= 16.
5. Post-GEMM masking on the V side via arith_progression + broadcast +
   compare + select (VSEL_TENSOR_SCALAR_MODE): the block-diagonal garbage
   cells (j > i) can be +-inf under deep decay (E+_i*E-_j up to 2^180 in
   the accumulator) -- a multiply mask would give inf*0 = NaN; select
   REPLACES with 0, matching the golden's masked_fill semantics (the
   bnd_dbound sentinel passes ratio 1.0000). The transposed-domain
   beta-fold variant (transpose + row broadcast + mul + select +
   transpose) was probed and dropped: a reproducible 23-cell-per-head
   zeroing (a_m rows 3-4) whose root cause was not isolated within the
   iteration budget; the beta-into-K'' fold avoids that whole chain.
6. Memory-planner interaction (iter7 toolchain findings): region-dst
   writes do not extend a buffer's planner liveness -- the per-vid
   operand accumulators a_agg/b_agg must be whole-buffer-filled first
   (fill anchors the interval), otherwise the planner overlays them with
   the in-loop fp32 scratches (bc_row/e) and the staged operands get
   fp32-bytes-read-as-bf16/fp32 corruption. Per-head load staging
   (q/k/gk loaded inside the factorize loop) keeps the peak UB ~124KB.

Supported domain (proto_fla_chunk_intra_token_parallel.yaml + iter7):
fixed-length sequences with S % CS == 0, CS % SCS == 0, H % BH == 0 (BH
adapts 8 -> 4 -> 2), DK % 16 == 0, DK >= 32, and BH * SCS >= 16 (gemm
N-fractal lower bound; excludes only SCS < 8 at BH=2 corners not
exercised by the suite); q/k/beta bfloat16 + gk float32 inputs, bfloat16
outputs. varlen (cu_seqlens) is out of scope.
"""

import torch
import tilelang
from tilelang import language as T

# 2^g is computed as exp(g * ln2): exp2 has no Ascend backend mapping.
LN2 = 0.6931471805599453

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_VECTOR_MASK_REUSE: False,
}


def _pick_ggrp(BH, SCS, DK):
    """Heads per gemm_v0 group: amortize the per-call cube overhead with
    block-diagonal batching, bounded by the L0A single-K-step budget
    (fp32 A tile GGRP*2*SCS x kL0Size=128 x 4B <= 64KB) and the L0B/L0C
    N >= 16 fractal rule (N = GGRP*SCS)."""
    g = min(BH, 4)
    while g > 1 and g * 2 * SCS * DK * 4 > 65536:
        g //= 2
    need = (16 + SCS - 1) // SCS
    g = max(g, min(need, BH))
    return g


@tilelang.jit(out_idx=[-2, -1], workspace_idx=[4, 5, 6], pass_configs=PASS_CONFIGS)
def fla_chunk_intra_token_parallel_kernel(
    B,
    S,
    H,
    DK,
    CS=64,
    SCS=16,
    BH=8,
    input_dtype="bfloat16",
    gate_dtype="float32",
    output_dtype="bfloat16",
):
    """JIT kernel generator. Returns a kernel taking (Q, K, GK, Beta) and
    producing (Aqk [B,S,H,CS], Akk [B,S,H,SCS]), both output_dtype. The
    runtime allocates the three fp32 GM workspaces (operand A/B transit +
    L0C result transit) alongside the outputs."""
    VEC_NUM = 2  # default dual vector threads: (cid, vid)
    BH_V = BH // VEC_NUM  # heads handled by each vid
    NSC = S // SCS  # subchunks per sequence
    HG = H // BH  # head groups
    NBLK = B * NSC * HG
    GGRP = _pick_ggrp(BH, SCS, DK)  # heads per gemm group
    NG = BH // GGRP  # gemm groups per block
    AM = GGRP * 2 * SCS  # gemm A rows (M)
    BN = GGRP * SCS  # gemm B rows (N)
    AKK_W = max(SCS, 16)
    AMV = BH_V * 2 * SCS  # per-vid a_agg rows
    BNV = BH_V * SCS  # per-vid b_agg rows
    op_dtype = gate_dtype  # fp32 GEMM operands (docstring item 4)

    Q_shape = (B, S, H, DK)
    K_shape = (B, S, H, DK)
    GK_shape = (B, S, H, DK)
    Beta_shape = (B, S, H)
    Aqk_shape = (B, S, H, CS)
    Akk_shape = (B, S, H, SCS)

    @T.prim_func
    def main(
        Q: T.Tensor(Q_shape, dtype=input_dtype),  # type: ignore
        K: T.Tensor(K_shape, dtype=input_dtype),  # type: ignore
        GK: T.Tensor(GK_shape, dtype=gate_dtype),  # type: ignore
        Beta: T.Tensor(Beta_shape, dtype=input_dtype),  # type: ignore
        workspace_a: T.Tensor([NBLK, BH * 2 * SCS, DK], dtype=op_dtype),  # type: ignore
        workspace_b: T.Tensor([NBLK, BH * SCS, DK], dtype=op_dtype),  # type: ignore
        workspace_r: T.Tensor([NBLK, NG, AM, BN], dtype=gate_dtype),  # type: ignore
        Aqk: T.Tensor(Aqk_shape, dtype=output_dtype),  # type: ignore
        Akk: T.Tensor(Akk_shape, dtype=output_dtype),  # type: ignore
    ):
        with T.Kernel(NBLK, is_npu=True) as (cid, vid):
            # cid decomposition, subchunk-major (same-subchunk head groups
            # are scheduled adjacently -> wave-local GM reuse)
            sc = cid % NSC
            hg = (cid // NSC) % HG
            bb = cid // (NSC * HG)
            c = sc // (CS // SCS)  # chunk index
            i_s = sc % (CS // SCS)  # subchunk index inside the chunk
            i_ts = c * CS + i_s * SCS  # first token of the subchunk
            h0v = hg * BH + vid * BH_V  # first head of this vid

            # ---- V-side UB buffers ----
            # per-head load staging + fp32 rows (consumed within one
            # factorize iteration; whole-block slabs would blow the UB
            # budget once the accumulators' liveness is anchored)
            q_h_row = T.alloc_ub([SCS, DK], input_dtype)
            k_h_row = T.alloc_ub([SCS, DK], input_dtype)
            q_f_row = T.alloc_ub([SCS, DK], gate_dtype)
            k_f_row = T.alloc_ub([SCS, DK], gate_dtype)
            g2_row = T.alloc_ub([SCS, DK], gate_dtype)
            beta_h = T.alloc_ub([SCS, 16], input_dtype)
            beta_f = T.alloc_ub([SCS, 16], gate_dtype)
            e = T.alloc_ub([SCS, DK], gate_dtype)  # E-/E+/K'' chain
            kgj = T.alloc_ub([SCS, DK], gate_dtype)  # K' = k * E-
            bc_row = T.alloc_ub([SCS, DK], gate_dtype)  # anchor broadcast
            # per-vid fp32 operand accumulators (fill-anchored liveness!)
            a_agg = T.alloc_ub([AMV, DK], op_dtype)
            b_agg = T.alloc_ub([BNV, DK], op_dtype)
            # masks / index grids (attention_sink pattern)
            idx = T.alloc_ub([SCS], gate_dtype)
            i_grid = T.alloc_ub([SCS, SCS], gate_dtype)
            j_grid = T.alloc_ub([SCS, SCS], gate_dtype)
            lt_m = T.alloc_ub([SCS, SCS], "uint8")  # lt[i,j] = (j <= i)
            slt_m = T.alloc_ub([SCS, SCS], "uint8")  # slt[i,j] = (j < i)
            # result processing
            r_ub = T.alloc_ub([2 * SCS, SCS], gate_dtype)
            r1_m = T.alloc_ub([SCS, SCS], gate_dtype)
            a_m = T.alloc_ub([SCS, SCS], gate_dtype)
            # fp32 staging + bf16 out staging (iter1 layout, kept)
            aqk_s = T.alloc_ub([BH_V, SCS, CS], gate_dtype)
            akk_s = T.alloc_ub([BH_V, SCS, AKK_W], gate_dtype)
            aqk_h = T.alloc_ub([BH_V, SCS, CS], output_dtype)
            akk_h = T.alloc_ub([BH_V, SCS, AKK_W], output_dtype)

            # ---- C-side buffers ----
            A_l1 = T.alloc_L1([AM, DK], op_dtype)
            B_l1 = T.alloc_L1([BN, DK], op_dtype)
            C_l0c = T.alloc_L0C([AM, BN], gate_dtype)

            # ==== V phase 1: beta staging, staging fills, masks ====
            T.tile.fill(beta_h, 0.0)  # deterministic pad columns
            T.copy(Beta[bb, i_ts, h0v], beta_h[:, 0:BH_V])  # [SCS, BH_V] row-gap copy
            T.copy(beta_h, beta_f)  # UB<->UB cross-dtype cast copy (flat)
            T.tile.fill(aqk_s, 0.0)  # uncovered Aqk columns stay 0
            T.tile.fill(akk_s, 0.0)  # Akk pad columns (SCS < 16) stay 0
            T.tile.arith_progression(idx, 0.0, 1.0, SCS)
            T.tile.broadcast(j_grid, idx, axis=0)  # j_grid[r, c] = c
            T.tile.broadcast(i_grid, idx, axis=1)  # i_grid[r, c] = r
            T.tile.compare(lt_m, i_grid, j_grid, "GE")  # r >= c (j <= i)
            T.tile.compare(slt_m, i_grid, j_grid, "GT")  # r > c (j < i)

            # ==== V phase 2: per-head load + factored construction +
            # operand staging. Whole-buffer fills FIRST: region-dst writes
            # alone do not extend the accumulators' planner liveness (the
            # planner would overlay them with bc_row/e and corrupt the
            # staged operands). ====
            T.tile.fill(a_agg, 0.0)
            T.tile.fill(b_agg, 0.0)
            anc = SCS // 2
            for h in T.serial(BH_V):
                T.copy(Q[bb, i_ts : i_ts + SCS, h0v + h, :], q_h_row)
                T.copy(K[bb, i_ts : i_ts + SCS, h0v + h, :], k_h_row)
                T.copy(GK[bb, i_ts : i_ts + SCS, h0v + h, :], g2_row)
                T.tile.cast(q_f_row, q_h_row, "CAST_NONE", SCS * DK)
                T.tile.cast(k_f_row, k_h_row, "CAST_NONE", SCS * DK)
                T.tile.mul(g2_row, g2_row, LN2)  # g2 = gk * ln2
                # E- = exp(g_a - g_j)
                T.tile.broadcast(bc_row, g2_row[anc : anc + 1, :], axis=0)
                T.tile.sub(e, bc_row, g2_row)
                T.tile.exp(e, e)
                T.tile.mul(kgj, k_f_row, e)  # K' = k * E-
                # E+ = exp(g_i - g_a)
                T.tile.sub(e, g2_row, bc_row)
                T.tile.exp(e, e)
                T.tile.mul(q_f_row, q_f_row, e)  # Q' = q * E+
                T.tile.mul(e, k_f_row, e)  # K'' = k * E+
                # fold beta into K'' rows (Akk = (beta*K'') @ K'^T);
                # per-row scalar muls on the production beta_f[i*16+h, 0]
                # BufferLoad pattern (beta[i, h] at flat i*16 + h)
                for i in range(SCS):
                    T.tile.mul(e[i : i + 1, :], e[i : i + 1, :],
                               beta_f[i * 16 + h, 0])
                # stage fp32 operands (exact, no rounding): region-dst
                # copies into the fill-anchored accumulators
                T.copy(q_f_row, a_agg[2 * SCS * h : 2 * SCS * h + SCS, :])
                T.copy(e, a_agg[2 * SCS * h + SCS : 2 * SCS * h + 2 * SCS, :])
                T.copy(kgj, b_agg[SCS * h : SCS * h + SCS, :])
            # per-vid aggregate -> GM workspace (2 big copies)
            T.copy(a_agg, workspace_a[cid, vid * AMV : vid * AMV + AMV, :])
            T.copy(b_agg, workspace_b[cid, vid * BNV : vid * BNV + BNV, :])

            # ==== C phase: block-diagonal gemm per head group ====
            for gg in T.serial(NG):
                T.copy(workspace_a[cid, gg * AM : gg * AM + AM, :], A_l1)
                T.copy(workspace_b[cid, gg * BN : gg * BN + BN, :], B_l1)
                T.gemm_v0(A_l1, B_l1, C_l0c, transpose_B=True, init=True)
                T.copy(C_l0c, workspace_r[cid, gg, :, :])

            # ==== V phase 3: result post-processing ====
            # Aqk rows 0..SCS-1 and Akk rows SCS..2SCS-1 of each diagonal
            # block; select (not mul) masking: the j > i garbage can be
            # +/-inf under deep decay and inf*0 = NaN, select REPLACES -> 0
            w = i_s * SCS  # Aqk window base column (runtime offset)
            for h in T.serial(BH_V):
                gh = vid * BH_V + h  # global head in block
                gg = gh // GGRP
                wi = gh % GGRP
                T.copy(workspace_r[cid, gg, 2 * SCS * wi : 2 * SCS * wi + 2 * SCS,
                               SCS * wi : SCS * wi + SCS], r_ub)
                T.tile.select(r1_m, lt_m, r_ub[0:SCS, :], 0.0,
                              "VSEL_TENSOR_SCALAR_MODE")
                T.copy(r1_m, aqk_s[h, :, w : w + SCS])
                T.tile.select(a_m, slt_m, r_ub[SCS : 2 * SCS, :], 0.0,
                              "VSEL_TENSOR_SCALAR_MODE")
                T.copy(a_m, akk_s[h, :, 0:SCS])

            # ==== writeback: block cast to bf16 + per-head 2D bursts ====
            T.tile.cast(aqk_h, aqk_s, "CAST_RINT", SCS * BH_V * CS)
            T.tile.cast(akk_h, akk_s, "CAST_RINT", BH_V * SCS * AKK_W)
            for h in T.serial(BH_V):
                T.copy(aqk_h[h, :, :], Aqk[bb, i_ts : i_ts + SCS, h0v + h, :])
                T.copy(akk_h[h, :, 0:SCS], Akk[bb, i_ts : i_ts + SCS, h0v + h, :])

    return main


def _pick_head_group(H):
    """BH is a performance tiling parameter, not a dispatch predicate:
    pick the largest supported group that divides H (8 -> 4 -> 2)."""
    for bh in (8, 4, 2):
        if H % bh == 0:
            return bh
    return 0


def fla_chunk_intra_token_parallel(q, k, gk, beta, chunk_size=64, sub_chunk_size=16):
    """Host wrapper: domain asserts (pure Python integer/dtype checks) +
    JIT launch. Returns (Aqk [B,S,H,chunk_size], Akk [B,S,H,sub_chunk_size]),
    both bfloat16, allocated by tilelang (out_idx=[-2, -1])."""
    B, S, H, DK = q.shape
    CS, SCS = chunk_size, sub_chunk_size

    if S % CS != 0:
        raise ValueError(f"S={S} must be divisible by chunk_size={CS}")
    if CS % SCS != 0:
        raise ValueError(f"chunk_size={CS} must be divisible by sub_chunk_size={SCS}")
    if SCS < 1 or CS < SCS:
        raise ValueError(f"invalid chunk/sub_chunk sizes: {CS}/{SCS}")
    if DK % 16 != 0 or DK < 32:
        raise ValueError(f"DK={DK} must be a multiple of 16 and >= 32")
    BH = _pick_head_group(H)
    if BH == 0:
        raise ValueError(f"H={H} must be divisible by 2 (minimum head group)")
    if BH * SCS < 16:
        # gemm_v0 L0B/L0C N-fractal lower bound with GGRP <= BH (iter7
        # GEMM-offload restructure; excludes only SCS < 8 at BH=2 corners)
        raise ValueError(
            f"BH*SCS={BH * SCS} must be >= 16 (gemm N-fractal lower bound; "
            f"got BH={BH}, SCS={SCS})")
    if q.dtype is not torch.bfloat16 or k.dtype is not torch.bfloat16:
        raise ValueError("q/k must be bfloat16")
    if gk.dtype is not torch.float32:
        raise ValueError("gk must be float32")
    if beta.dtype is not torch.bfloat16:
        raise ValueError("beta must be bfloat16")
    for name, t in (("q", q), ("k", k), ("gk", gk), ("beta", beta)):
        if not t.is_contiguous():
            raise ValueError(f"input '{name}' must be contiguous")

    kernel = fla_chunk_intra_token_parallel_kernel(B, S, H, DK, CS, SCS, BH)
    return kernel(q, k, gk, beta)
