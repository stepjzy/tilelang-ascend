"""fla_wy_fast: KDA chunked WY representation (fwd + bwd) on Ascend 910B3.

fwd  (recompute_w_u_fwd):   u = A_c @ (beta * v), w = A_c @ (beta * k * 2^g),
                            kg = k * 2^(g[n,:] - g)
bwd  (prepare_wy_repr_bwd): dA = -strict_tril(A_c^T @ S @ A_c^T),
                            S = strict_tril(dw @ K_bg^T + du @ V_b^T),
                            dk2 = (A_c^T @ dw) * beta * 2^g + dk,
                            dv = (A_c^T @ du) * beta,
                            dbeta = rowsum((A_c^T@dw) * k * 2^g)
                                  + rowsum((A_c^T@du) * v),
                            dg2 = K_bg * (A_c^T @ dw) + dg

Developer mode, threads=1, bshd layout consumed directly (only zero-copy
metadata views on the host side).

Ascend toolchain adaptations (all empirically verified on 910B3):

1. Kernel splitting. The C/V combine supports a SINGLE cross-core Pipelined
   body per kernel, requires >= 2 loop iterations (extent-1 crashes codegen),
   and mis-syncs V->C exchanges in plain serial loops. Each direction is
   split into simple single-Pipelined kernels; cross-kernel accumulators
   (dA partial, dbeta partial, S, Z) are handed over through GM.

2. Emulated-precision GEMM. gemm_v0 with fp32 A/B is broken on this
   toolchain (produces interleaved garbage even in isolation; no fp32 test
   coverage exists in the repo). All GEMMs therefore run in reduced
   precision with split operands accumulated in L0C:
     * fwd: A stays bfloat16 (exact), the V-side product is split into
       bfloat16 hi/lo (2 gemms) -> ~2^-16 relative error, far inside the
       bf16 output gate (2^-6).
     * bwd (fp32 gate 2^-10): every operand is split into fp16 hi/lo
       (x = hi + lo, 3 gemms: hi@hi + hi@lo + lo@hi) -> ~2^-22 relative
       error. KNOWN LIMITATION: operands beyond the fp16 range
       (|x| > 65504 or |x| < 2^-24) lose precision (Boundary-class inputs).
   Both schemes are validated against CPU mimics of the mixed-tolerance
   gates.

3. V->C exchanges (UB->L1) only happen inside the Pipelined loop: a
   prologue exchange gets mis-versioned by the pipeline pass (stage-offset
   reads of a once-written workspace).

4. [B,S,H] beta loads use a flat-view span copy + arith_progression +
   reinterpretcast + tile.gather (sub-32B strided DMA is unreliable, and
   the gather chain breaks under threads=2 vid splitting).

5. The dbeta [B,S,H] scatter stages through a GM scratch with an explicit
   MTE3->MTE2 flag sync (the compiler neither tracks aliased view params
   nor keeps computed-source [64,1] scatters correct).

6. Loop-carried reduce uses clear=True + T.tile.add accumulation (the
   clear=False merge inside loops is a toolchain bug). exp only lowers on
   bare buffer loads / whole tiles. Column-broadcast operands in 2D
   T.Parallel are unsupported; row-scalar broadcasts and the fwd kg column
   broadcast use T.tile.broadcast (axis=1 / axis=0) + full-tile 2D ops
    instead of per-row serial tile loops (-26% .. -71% per kernel).

7. out_idx uses absolute indices (auto-injected GM workspace params append
   to the kernel signature and shift negative indices).

8. L0C(fp32)->GM(bf16) direct fixpipe conversion is broken; fwd stores
   stage through UB with an explicit cast copy. GM<->UB cast-on-copy for
   bf16<->fp32 is equally broken (codegen emits a mismatched
   copy_gm_to_ub<bfloat16_t> template call), so fwd loads keep a bf16
   staging buffer + explicit UB cast.

9. Host wrappers keep no internal torch.npu.synchronize (upstream wrapper
   parity -- the do_bench harness owns synchronization) and cache the
   strict-tril Msk constant once per (device, chunk) instead of
   reconstructing it (CPU tril + H2D copy) on every call.
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


def _pick_block_d(dim: int) -> int:
    """Block size forcing EXACTLY 2 Pipelined iterations.

    Toolchain constraints (see module docstring): a Pipelined body crashes
    codegen at extent 1 and corrupts the 3rd iteration's double-buffer wrap
    (verified: D=96 with block 32 / 3 iterations gives exactly 2/3 correct
    slices). The only safe extent is 2, so block = dim // 2, which requires
    dim // 2 to be a 16-multiple gemm N in [16, 64]: dim in {32, 64, 96,
    128}. Other 16-multiples (16, 48, 80, ...) have no valid exact-2 block
    and are rejected (documented limitation).
    """
    blk = dim // 2
    if dim % 2 == 0 and blk in (16, 32, 48, 64):
        return blk
    raise ValueError(
        f"DK/DV={dim} unsupported: the Ascend Pipelined body requires exactly"
        " 2 iterations, so DK/DV must be one of {{32, 64, 96, 128}}"
        " (16-multiples without a valid exact-2 block are rejected)"
        " (documented limitation)"
    )


def _grid_dims(B, S, H, chunk_size):
    nt = S // chunk_size
    return nt, nt * B * H


# ============================================================================
# fwd kernel 1: u = A_c @ (beta * v)
# ============================================================================
@tilelang.jit(out_idx=[3], pass_configs=PASS_CONFIGS)
def fla_wy_fwd_u_kernel(B, S, H, DV, chunk_size=64, block_DV=64):
    BS = chunk_size
    _, GRID = _grid_dims(B, S, H, BS)
    bDV = block_DV

    @T.prim_func
    def main(
        V: T.Tensor((B, S, H, DV), "bfloat16"),  # type: ignore
        BetaFlat: T.Tensor((B, S * H), "bfloat16"),  # type: ignore
        A: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        U: T.Tensor((B, S, H, DV), "bfloat16"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % (S // BS)
            bbh = cid // (S // BS)
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            A_l1 = T.alloc_shared([BS, BS], "bfloat16")
            beta_tile = T.alloc_shared([BS * H], "bfloat16")
            off_i32 = T.alloc_shared([BS], "int32")
            off_u32 = T.alloc_shared([BS], "uint32")
            bt_bf = T.alloc_shared([BS], "bfloat16")
            bt = T.alloc_shared([BS], "float32")
            bt2d = T.alloc_shared([BS, bDV], "float32")
            v_half = T.alloc_shared([BS, bDV], "bfloat16")
            v_ub = T.alloc_shared([BS, bDV], "float32")
            vb_hi_ub = T.alloc_shared([BS, bDV], "bfloat16")
            vb_hi_f32 = T.alloc_shared([BS, bDV], "float32")
            vb_lo_f32 = T.alloc_shared([BS, bDV], "float32")
            vb_lo_ub = T.alloc_shared([BS, bDV], "bfloat16")
            vb_hi_l1 = T.alloc_shared([BS, bDV], "bfloat16")
            vb_lo_l1 = T.alloc_shared([BS, bDV], "bfloat16")
            u_frag = T.alloc_fragment([BS, bDV], "float32")
            u_ub = T.alloc_shared([BS, bDV], "float32")
            u_half = T.alloc_shared([BS, bDV], "bfloat16")

            T.copy(BetaFlat[bb, s0 * H : (s0 + BS) * H], beta_tile)
            T.tile.arith_progression(off_i32, bh * 2, H * 2)
            T.reinterpretcast(off_u32, off_i32, "uint32_t")
            T.tile.gather(bt_bf, beta_tile, off_u32, 0)
            T.copy(bt_bf, bt)

            for i_v in T.Pipelined(DV // bDV, num_stages=2):
                T.copy(A[bb, s0 : s0 + BS, bh, :], A_l1)
                T.copy(
                    V[bb, s0 : s0 + BS, bh, i_v * bDV : (i_v + 1) * bDV],
                    v_half,
                )
                T.copy(v_half, v_ub)
                # row-scalar broadcast via Broadcast DMA + one full-tile mul
                # (64 thin per-row Muls -> 2 fat vector ops; antipattern
                # "Vector Core 内逐元素/逐行 for loop 计算")
                T.tile.broadcast(bt2d, bt, axis=1)
                T.tile.mul(v_ub, v_ub, bt2d)
                # bf16 hi/lo split of beta*v (A stays bf16: exact operand)
                T.copy(v_ub, vb_hi_ub)
                T.copy(vb_hi_ub, vb_hi_f32)
                T.tile.sub(vb_lo_f32, v_ub, vb_hi_f32)
                T.copy(vb_lo_f32, vb_lo_ub)
                T.copy(vb_hi_ub, vb_hi_l1)
                T.copy(vb_lo_ub, vb_lo_l1)
                T.gemm_v0(A_l1, vb_hi_l1, u_frag, init=True)
                T.gemm_v0(A_l1, vb_lo_l1, u_frag, init=False)
                T.copy(u_frag, u_ub)
                T.copy(u_ub, u_half)
                T.copy(
                    u_half,
                    U[bb, s0 : s0 + BS, bh, i_v * bDV : (i_v + 1) * bDV],
                )

    return main


# ============================================================================
# fwd kernel 2: w = A_c @ (beta * k * 2^g), kg = k * 2^(g_n - g)
# ============================================================================
@tilelang.jit(out_idx=[4, 5], pass_configs=PASS_CONFIGS)
def fla_wy_fwd_wkg_kernel(B, S, H, DK, chunk_size=64, block_DK=64):
    BS = chunk_size
    _, GRID = _grid_dims(B, S, H, BS)
    bDK = block_DK

    @T.prim_func
    def main(
        K: T.Tensor((B, S, H, DK), "bfloat16"),  # type: ignore
        BetaFlat: T.Tensor((B, S * H), "bfloat16"),  # type: ignore
        G: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        A: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        W: T.Tensor((B, S, H, DK), "bfloat16"),  # type: ignore
        KG: T.Tensor((B, S, H, DK), "bfloat16"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % (S // BS)
            bbh = cid // (S // BS)
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            A_l1 = T.alloc_shared([BS, BS], "bfloat16")
            beta_tile = T.alloc_shared([BS * H], "bfloat16")
            off_i32 = T.alloc_shared([BS], "int32")
            off_u32 = T.alloc_shared([BS], "uint32")
            bt_bf = T.alloc_shared([BS], "bfloat16")
            bt = T.alloc_shared([BS], "float32")
            bt2d = T.alloc_shared([BS, bDK], "float32")
            k_half = T.alloc_shared([BS, bDK], "bfloat16")
            k_ub = T.alloc_shared([BS, bDK], "float32")
            g_ub = T.alloc_shared([BS, bDK], "float32")
            gL_ub = T.alloc_shared([BS, bDK], "float32")
            gexp_ub = T.alloc_shared([BS, bDK], "float32")
            kbg_f32 = T.alloc_shared([BS, bDK], "float32")
            kbg_hi_ub = T.alloc_shared([BS, bDK], "bfloat16")
            kbg_hi_f32 = T.alloc_shared([BS, bDK], "float32")
            kbg_lo_f32 = T.alloc_shared([BS, bDK], "float32")
            kbg_lo_ub = T.alloc_shared([BS, bDK], "bfloat16")
            kbg_hi_l1 = T.alloc_shared([BS, bDK], "bfloat16")
            kbg_lo_l1 = T.alloc_shared([BS, bDK], "bfloat16")
            w_frag = T.alloc_fragment([BS, bDK], "float32")
            w_ub = T.alloc_shared([BS, bDK], "float32")
            w_half = T.alloc_shared([BS, bDK], "bfloat16")
            gn_ub = T.alloc_shared([bDK], "float32")
            gn2d_ub = T.alloc_shared([BS, bDK], "float32")
            kg_ub = T.alloc_shared([BS, bDK], "float32")
            kg_half = T.alloc_shared([BS, bDK], "bfloat16")

            T.copy(BetaFlat[bb, s0 * H : (s0 + BS) * H], beta_tile)
            T.tile.arith_progression(off_i32, bh * 2, H * 2)
            T.reinterpretcast(off_u32, off_i32, "uint32_t")
            T.tile.gather(bt_bf, beta_tile, off_u32, 0)
            T.copy(bt_bf, bt)

            for i_k in T.Pipelined(DK // bDK, num_stages=2):
                T.copy(A[bb, s0 : s0 + BS, bh, :], A_l1)
                T.copy(
                    K[bb, s0 : s0 + BS, bh, i_k * bDK : (i_k + 1) * bDK],
                    k_half,
                )
                T.copy(k_half, k_ub)
                T.copy(
                    G[bb, s0 : s0 + BS, bh, i_k * bDK : (i_k + 1) * bDK],
                    g_ub,
                )
                T.tile.mul(gL_ub, g_ub, LN2)
                T.tile.exp(gexp_ub, gL_ub)
                # row-scalar broadcast via Broadcast DMA + full-tile mul
                T.tile.broadcast(bt2d, bt, axis=1)
                T.tile.mul(kbg_f32, k_ub, bt2d)
                T.tile.mul(kbg_f32, kbg_f32, gexp_ub)
                # bf16 hi/lo split of K_bg
                T.copy(kbg_f32, kbg_hi_ub)
                T.copy(kbg_hi_ub, kbg_hi_f32)
                T.tile.sub(kbg_lo_f32, kbg_f32, kbg_hi_f32)
                T.copy(kbg_lo_f32, kbg_lo_ub)
                T.copy(kbg_hi_ub, kbg_hi_l1)
                T.copy(kbg_lo_ub, kbg_lo_l1)
                T.gemm_v0(A_l1, kbg_hi_l1, w_frag, init=True)
                T.gemm_v0(A_l1, kbg_lo_l1, w_frag, init=False)
                T.copy(w_frag, w_ub)
                T.copy(w_ub, w_half)
                T.copy(
                    w_half,
                    W[bb, s0 : s0 + BS, bh, i_k * bDK : (i_k + 1) * bDK],
                )
                # kg = k * 2^(g_n - g): column broadcast via Broadcast DMA
                # + full-tile 2D chain (64x4 serial row ops -> 6 fat ops).
                # The div variant (k*2^gn / 2^g, 1D exp) measured equal --
                # div and exp cost the same on the vector core.
                T.copy(
                    G[bb, s0 + BS - 1, bh, i_k * bDK : (i_k + 1) * bDK],
                    gn_ub,
                )
                T.tile.mul(gn_ub, gn_ub, LN2)
                T.tile.broadcast(gn2d_ub, gn_ub, axis=0)
                T.tile.sub(gn2d_ub, gn2d_ub, gL_ub)
                T.tile.exp(gn2d_ub, gn2d_ub)
                T.tile.mul(kg_ub, k_ub, gn2d_ub)
                T.copy(kg_ub, kg_half)
                T.copy(
                    kg_half,
                    KG[bb, s0 : s0 + BS, bh, i_k * bDK : (i_k + 1) * bDK],
                )

    return main


# ============================================================================
# bwd kernel 1 (i_k part): dA partial (dw @ K_bg^T), dk2, dg2, dbeta partial
# ============================================================================
@tilelang.jit(out_idx=[4], pass_configs=PASS_CONFIGS)
def fla_wy_bwd_k1_kernel(B, S, H, DK, chunk_size=64, block_DK=64):
    """k-part gemm 1: dA_part = dw @ K_bg^T (single gemm group)."""
    BS = chunk_size
    nt, GRID = _grid_dims(B, S, H, BS)
    bDK = block_DK

    @T.prim_func
    def main(
        K: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        BetaFlat: T.Tensor((B, S * H), "float32"),  # type: ignore
        G: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        dw: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        dA_part: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % nt
            bbh = cid // nt
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            beta_tile = T.alloc_shared([BS * H], "float32")
            off_i32 = T.alloc_shared([BS], "int32")
            off_u32 = T.alloc_shared([BS], "uint32")
            bt = T.alloc_shared([BS], "float32")
            bt2d = T.alloc_shared([BS, bDK], "float32")
            k_half = T.alloc_shared([BS, bDK], "bfloat16")
            k_ub = T.alloc_shared([BS, bDK], "float32")
            g_ub = T.alloc_shared([BS, bDK], "float32")
            gexp_ub = T.alloc_shared([BS, bDK], "float32")
            kbg_f32 = T.alloc_shared([BS, bDK], "float32")
            dw_f32 = T.alloc_shared([BS, bDK], "float32")
            dw_hi_f32 = T.alloc_shared([BS, bDK], "float32")
            dw_lo_f32 = T.alloc_shared([BS, bDK], "float32")
            dw_hi_ub = T.alloc_shared([BS, bDK], "float16")
            dw_lo_ub = T.alloc_shared([BS, bDK], "float16")
            kbg_hi_f32 = T.alloc_shared([BS, bDK], "float32")
            kbg_lo_f32 = T.alloc_shared([BS, bDK], "float32")
            kbg_hi_ub = T.alloc_shared([BS, bDK], "float16")
            kbg_lo_ub = T.alloc_shared([BS, bDK], "float16")
            dw_hi_l1 = T.alloc_shared([BS, bDK], "float16")
            dw_lo_l1 = T.alloc_shared([BS, bDK], "float16")
            kbg_hi_l1 = T.alloc_shared([BS, bDK], "float16")
            kbg_lo_l1 = T.alloc_shared([BS, bDK], "float16")
            dAp_frag = T.alloc_fragment([BS, BS], "float32")

            T.copy(BetaFlat[bb, s0 * H : (s0 + BS) * H], beta_tile)
            T.tile.arith_progression(off_i32, bh * 4, H * 4)
            T.reinterpretcast(off_u32, off_i32, "uint32_t")
            T.tile.gather(bt, beta_tile, off_u32, 0)

            for i_k in T.Pipelined(DK // bDK, num_stages=2):
                T.copy(
                    K[bb, s0 : s0 + BS, bh, i_k * bDK : (i_k + 1) * bDK],
                    k_ub,
                )
                T.copy(
                    G[bb, s0 : s0 + BS, bh, i_k * bDK : (i_k + 1) * bDK],
                    g_ub,
                )
                T.tile.mul(gexp_ub, g_ub, LN2)
                T.tile.exp(gexp_ub, gexp_ub)
                # row-scalar broadcast via Broadcast DMA + full-tile mul
                T.tile.broadcast(bt2d, bt, axis=1)
                T.tile.mul(kbg_f32, k_ub, bt2d)
                T.tile.mul(kbg_f32, kbg_f32, gexp_ub)

                T.copy(
                    dw[bb, s0 : s0 + BS, bh, i_k * bDK : (i_k + 1) * bDK],
                    dw_f32,
                )
                T.copy(dw_f32, dw_hi_ub)
                T.copy(dw_hi_ub, dw_hi_f32)
                T.tile.sub(dw_lo_f32, dw_f32, dw_hi_f32)
                T.copy(dw_lo_f32, dw_lo_ub)
                T.copy(dw_hi_ub, dw_hi_l1)
                T.copy(dw_lo_ub, dw_lo_l1)

                T.copy(kbg_f32, kbg_hi_ub)
                T.copy(kbg_hi_ub, kbg_hi_f32)
                T.tile.sub(kbg_lo_f32, kbg_f32, kbg_hi_f32)
                T.copy(kbg_lo_f32, kbg_lo_ub)
                T.copy(kbg_hi_ub, kbg_hi_l1)
                T.copy(kbg_lo_ub, kbg_lo_l1)

                # dA partial: dw @ K_bg^T (fp16x2, L0C accum across slices)
                T.gemm_v0(
                    dw_hi_l1,
                    kbg_hi_l1,
                    dAp_frag,
                    transpose_B=True,
                    init=(i_k == 0),
                )
                T.gemm_v0(dw_hi_l1, kbg_lo_l1, dAp_frag, transpose_B=True, init=False)
                T.gemm_v0(dw_lo_l1, kbg_hi_l1, dAp_frag, transpose_B=True, init=False)

            T.copy(dAp_frag, dA_part[bb, s0 : s0 + BS, bh, :])

    return main


@tilelang.jit(out_idx=[7, 8, 9], pass_configs=PASS_CONFIGS)
def fla_wy_bwd_k2_kernel(B, S, H, DK, chunk_size=64, block_DK=64):
    """k-part gemm 2: X = A_c^T @ dw -> dk2, dg2, dbeta partial."""
    BS = chunk_size
    nt, GRID = _grid_dims(B, S, H, BS)
    bDK = block_DK

    @T.prim_func
    def main(
        K: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        BetaFlat: T.Tensor((B, S * H), "float32"),  # type: ignore
        G: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        A: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
        dw: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        dk: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        dg: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        dk2: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        dg2: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        dbeta_flat: T.Tensor((GRID, BS), "float32"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % nt
            bbh = cid // nt
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            beta_tile = T.alloc_shared([BS * H], "float32")
            off_i32 = T.alloc_shared([BS], "int32")
            off_u32 = T.alloc_shared([BS], "uint32")
            bt = T.alloc_shared([BS], "float32")
            bt2d = T.alloc_shared([BS, bDK], "float32")
            k_half = T.alloc_shared([BS, bDK], "bfloat16")
            k_ub = T.alloc_shared([BS, bDK], "float32")
            g_ub = T.alloc_shared([BS, bDK], "float32")
            gexp_ub = T.alloc_shared([BS, bDK], "float32")
            kbg_f32 = T.alloc_shared([BS, bDK], "float32")
            a_f32 = T.alloc_shared([BS, BS], "float32")
            dw_f32 = T.alloc_shared([BS, bDK], "float32")
            x_ub = T.alloc_shared([BS, bDK], "float32")
            dk_old_ub = T.alloc_shared([BS, bDK], "float32")
            dg_old_ub = T.alloc_shared([BS, bDK], "float32")
            dk2_ub = T.alloc_shared([BS, bDK], "float32")
            dg2_ub = T.alloc_shared([BS, bDK], "float32")
            r_tmpk = T.alloc_shared([BS, bDK], "float32")
            r_part = T.alloc_shared([BS], "float32")
            acc = T.alloc_shared([BS], "float32")
            a_hi_f32 = T.alloc_shared([BS, BS], "float32")
            a_lo_f32 = T.alloc_shared([BS, BS], "float32")
            a_hi_ub = T.alloc_shared([BS, BS], "float16")
            a_lo_ub = T.alloc_shared([BS, BS], "float16")
            dw_hi_f32 = T.alloc_shared([BS, bDK], "float32")
            dw_lo_f32 = T.alloc_shared([BS, bDK], "float32")
            dw_hi_ub = T.alloc_shared([BS, bDK], "float16")
            dw_lo_ub = T.alloc_shared([BS, bDK], "float16")
            a_hi_l1 = T.alloc_shared([BS, BS], "float16")
            a_lo_l1 = T.alloc_shared([BS, BS], "float16")
            dw_hi_l1 = T.alloc_shared([BS, bDK], "float16")
            dw_lo_l1 = T.alloc_shared([BS, bDK], "float16")
            x_frag = T.alloc_fragment([BS, bDK], "float32")

            T.copy(BetaFlat[bb, s0 * H : (s0 + BS) * H], beta_tile)
            T.tile.arith_progression(off_i32, bh * 4, H * 4)
            T.reinterpretcast(off_u32, off_i32, "uint32_t")
            T.tile.gather(bt, beta_tile, off_u32, 0)
            T.tile.fill(acc, 0.0)

            for i_k in T.Pipelined(DK // bDK, num_stages=2):
                T.copy(
                    K[bb, s0 : s0 + BS, bh, i_k * bDK : (i_k + 1) * bDK],
                    k_ub,
                )
                T.copy(
                    G[bb, s0 : s0 + BS, bh, i_k * bDK : (i_k + 1) * bDK],
                    g_ub,
                )
                T.tile.mul(gexp_ub, g_ub, LN2)
                T.tile.exp(gexp_ub, gexp_ub)
                # row-scalar broadcast via Broadcast DMA + full-tile mul
                T.tile.broadcast(bt2d, bt, axis=1)
                T.tile.mul(kbg_f32, k_ub, bt2d)
                T.tile.mul(kbg_f32, kbg_f32, gexp_ub)

                # A split stays inside the loop: hoisting the cast chain to
                # the prologue extends buffer lifetimes across the loop and
                # overflows the UB budget at DK=128 (memory-planning reuse
                # loss > the ~5-op saving)
                T.copy(A[bb, s0 : s0 + BS, bh, :], a_f32)
                T.copy(a_f32, a_hi_ub)
                T.copy(a_hi_ub, a_hi_f32)
                T.tile.sub(a_lo_f32, a_f32, a_hi_f32)
                T.copy(a_lo_f32, a_lo_ub)
                T.copy(a_hi_ub, a_hi_l1)
                T.copy(a_lo_ub, a_lo_l1)

                T.copy(
                    dw[bb, s0 : s0 + BS, bh, i_k * bDK : (i_k + 1) * bDK],
                    dw_f32,
                )
                T.copy(dw_f32, dw_hi_ub)
                T.copy(dw_hi_ub, dw_hi_f32)
                T.tile.sub(dw_lo_f32, dw_f32, dw_hi_f32)
                T.copy(dw_lo_f32, dw_lo_ub)
                T.copy(dw_hi_ub, dw_hi_l1)
                T.copy(dw_lo_ub, dw_lo_l1)

                # X = A_c^T @ dw (fp16x2)
                T.gemm_v0(a_hi_l1, dw_hi_l1, x_frag, transpose_A=True, init=True)
                T.gemm_v0(a_hi_l1, dw_lo_l1, x_frag, transpose_A=True, init=False)
                T.gemm_v0(a_lo_l1, dw_hi_l1, x_frag, transpose_A=True, init=False)
                T.copy(x_frag, x_ub)

                # dk2 = X * beta * 2^g + dk
                # (CSE of Xg=X*2^g shared with the dbeta partial extends
                # r_tmpk's lifetime across dk2_ub -> UB overflow at DK=128;
                # k2 is at the memory-planning edge, keep the 2-mul form)
                T.copy(
                    dk[bb, s0 : s0 + BS, bh, i_k * bDK : (i_k + 1) * bDK],
                    dk_old_ub,
                )
                T.tile.mul(dk2_ub, x_ub, gexp_ub)
                T.tile.mul(dk2_ub, dk2_ub, bt2d)
                T.tile.add(dk2_ub, dk2_ub, dk_old_ub)
                T.copy(
                    dk2_ub,
                    dk2[bb, s0 : s0 + BS, bh, i_k * bDK : (i_k + 1) * bDK],
                )

                # dbeta partial: rowsum(X * k * 2^g)
                T.tile.mul(r_tmpk, x_ub, k_ub)
                T.tile.mul(r_tmpk, r_tmpk, gexp_ub)
                T.reduce_sum(r_tmpk, r_part, dim=1, clear=True)
                T.tile.add(acc, acc, r_part)

                # dg2 = K_bg * X + dg
                T.copy(
                    dg[bb, s0 : s0 + BS, bh, i_k * bDK : (i_k + 1) * bDK],
                    dg_old_ub,
                )
                T.tile.mul(dg2_ub, kbg_f32, x_ub)
                T.tile.add(dg2_ub, dg2_ub, dg_old_ub)
                T.copy(
                    dg2_ub,
                    dg2[bb, s0 : s0 + BS, bh, i_k * bDK : (i_k + 1) * bDK],
                )

            T.copy(acc, dbeta_flat[cid, 0:BS])

    return main


# ============================================================================
# bwd kernel 2 (i_v part + finalization): dv, dbeta, S
# ============================================================================
@tilelang.jit(out_idx=[5], pass_configs=PASS_CONFIGS)
def fla_wy_bwd_v1_kernel(B, S, H, DV, chunk_size=64, block_DV=64):
    """v-part gemm 1: S = strict_tril(dA_part + du @ V_b^T)."""
    BS = chunk_size
    nt, GRID = _grid_dims(B, S, H, BS)
    bDV = block_DV

    @T.prim_func
    def main(
        V: T.Tensor((B, S, H, DV), "float32"),  # type: ignore
        BetaFlat: T.Tensor((B, S * H), "float32"),  # type: ignore
        du: T.Tensor((B, S, H, DV), "float32"),  # type: ignore
        Msk: T.Tensor((BS, BS), "float32"),  # type: ignore
        dA_part: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
        S_out: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % nt
            bbh = cid // nt
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            beta_tile = T.alloc_shared([BS * H], "float32")
            off_i32 = T.alloc_shared([BS], "int32")
            off_u32 = T.alloc_shared([BS], "uint32")
            bt = T.alloc_shared([BS], "float32")
            bt2d = T.alloc_shared([BS, bDV], "float32")
            msk_ub = T.alloc_shared([BS, BS], "float32")
            v_ub = T.alloc_shared([BS, bDV], "float32")
            vb_f32 = T.alloc_shared([BS, bDV], "float32")
            du_f32 = T.alloc_shared([BS, bDV], "float32")
            du_hi_f32 = T.alloc_shared([BS, bDV], "float32")
            du_lo_f32 = T.alloc_shared([BS, bDV], "float32")
            du_hi_ub = T.alloc_shared([BS, bDV], "float16")
            du_lo_ub = T.alloc_shared([BS, bDV], "float16")
            vb_hi_f32 = T.alloc_shared([BS, bDV], "float32")
            vb_lo_f32 = T.alloc_shared([BS, bDV], "float32")
            vb_hi_ub = T.alloc_shared([BS, bDV], "float16")
            vb_lo_ub = T.alloc_shared([BS, bDV], "float16")
            du_hi_l1 = T.alloc_shared([BS, bDV], "float16")
            du_lo_l1 = T.alloc_shared([BS, bDV], "float16")
            vb_hi_l1 = T.alloc_shared([BS, bDV], "float16")
            vb_lo_l1 = T.alloc_shared([BS, bDV], "float16")
            dAv_frag = T.alloc_fragment([BS, BS], "float32")
            dAv_ub = T.alloc_shared([BS, BS], "float32")
            dA_sum_ub = T.alloc_shared([BS, BS], "float32")

            T.copy(BetaFlat[bb, s0 * H : (s0 + BS) * H], beta_tile)
            T.tile.arith_progression(off_i32, bh * 4, H * 4)
            T.reinterpretcast(off_u32, off_i32, "uint32_t")
            T.tile.gather(bt, beta_tile, off_u32, 0)
            T.copy(Msk[0:BS, 0:BS], msk_ub)

            for i_v in T.Pipelined(DV // bDV, num_stages=2):
                T.copy(
                    V[bb, s0 : s0 + BS, bh, i_v * bDV : (i_v + 1) * bDV],
                    v_ub,
                )
                # row-scalar broadcast via Broadcast DMA + full-tile mul
                T.tile.broadcast(bt2d, bt, axis=1)
                T.tile.mul(vb_f32, v_ub, bt2d)

                T.copy(
                    du[bb, s0 : s0 + BS, bh, i_v * bDV : (i_v + 1) * bDV],
                    du_f32,
                )
                T.copy(du_f32, du_hi_ub)
                T.copy(du_hi_ub, du_hi_f32)
                T.tile.sub(du_lo_f32, du_f32, du_hi_f32)
                T.copy(du_lo_f32, du_lo_ub)
                T.copy(du_hi_ub, du_hi_l1)
                T.copy(du_lo_ub, du_lo_l1)

                T.copy(vb_f32, vb_hi_ub)
                T.copy(vb_hi_ub, vb_hi_f32)
                T.tile.sub(vb_lo_f32, vb_f32, vb_hi_f32)
                T.copy(vb_lo_f32, vb_lo_ub)
                T.copy(vb_hi_ub, vb_hi_l1)
                T.copy(vb_lo_ub, vb_lo_l1)

                # dA partial: du @ V_b^T (fp16x2, L0C accum across slices)
                T.gemm_v0(
                    du_hi_l1,
                    vb_hi_l1,
                    dAv_frag,
                    transpose_B=True,
                    init=(i_v == 0),
                )
                T.gemm_v0(du_hi_l1, vb_lo_l1, dAv_frag, transpose_B=True, init=False)
                T.gemm_v0(du_lo_l1, vb_hi_l1, dAv_frag, transpose_B=True, init=False)

            # ---- S = strict_tril(dA_part + du @ V_b^T) ----
            T.copy(dA_part[bb, s0 : s0 + BS, bh, :], dA_sum_ub)
            T.copy(dAv_frag, dAv_ub)
            T.tile.add(dA_sum_ub, dA_sum_ub, dAv_ub)
            T.tile.mul(dA_sum_ub, dA_sum_ub, msk_ub)
            T.copy(dA_sum_ub, S_out[bb, s0 : s0 + BS, bh, :])

    return main


@tilelang.jit(out_idx=[6, 7], pass_configs=PASS_CONFIGS)
def fla_wy_bwd_v2_kernel(B, S, H, DV, chunk_size=64, block_DV=64):
    """v-part gemm 2: Y = A_c^T @ du -> dv, dbeta."""
    BS = chunk_size
    nt, GRID = _grid_dims(B, S, H, BS)
    bDV = block_DV

    @T.prim_func
    def main(
        V: T.Tensor((B, S, H, DV), "float32"),  # type: ignore
        BetaFlat: T.Tensor((B, S * H), "float32"),  # type: ignore
        A: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
        du: T.Tensor((B, S, H, DV), "float32"),  # type: ignore
        dbeta_flat: T.Tensor((GRID, BS), "float32"),  # type: ignore
        dbeta_flat_v: T.Tensor((GRID, BS, 1), "float32"),  # type: ignore
        dv: T.Tensor((B, S, H, DV), "float32"),  # type: ignore
        dbeta: T.Tensor((B, S, H), "float32"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % nt
            bbh = cid // nt
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            beta_tile = T.alloc_shared([BS * H], "float32")
            off_i32 = T.alloc_shared([BS], "int32")
            off_u32 = T.alloc_shared([BS], "uint32")
            bt = T.alloc_shared([BS], "float32")
            bt2d = T.alloc_shared([BS, bDV], "float32")
            v_half = T.alloc_shared([BS, bDV], "bfloat16")
            v_ub = T.alloc_shared([BS, bDV], "float32")
            a_f32 = T.alloc_shared([BS, BS], "float32")
            du_f32 = T.alloc_shared([BS, bDV], "float32")
            y_ub = T.alloc_shared([BS, bDV], "float32")
            dv_ub = T.alloc_shared([BS, bDV], "float32")
            r_tmpv = T.alloc_shared([BS, bDV], "float32")
            r_part = T.alloc_shared([BS], "float32")
            acc = T.alloc_shared([BS], "float32")
            a_hi_f32 = T.alloc_shared([BS, BS], "float32")
            a_lo_f32 = T.alloc_shared([BS, BS], "float32")
            a_hi_ub = T.alloc_shared([BS, BS], "float16")
            a_lo_ub = T.alloc_shared([BS, BS], "float16")
            du_hi_f32 = T.alloc_shared([BS, bDV], "float32")
            du_lo_f32 = T.alloc_shared([BS, bDV], "float32")
            du_hi_ub = T.alloc_shared([BS, bDV], "float16")
            du_lo_ub = T.alloc_shared([BS, bDV], "float16")
            a_hi_l1 = T.alloc_shared([BS, BS], "float16")
            a_lo_l1 = T.alloc_shared([BS, BS], "float16")
            du_hi_l1 = T.alloc_shared([BS, bDV], "float16")
            du_lo_l1 = T.alloc_shared([BS, bDV], "float16")
            y_frag = T.alloc_fragment([BS, bDV], "float32")
            dbeta2 = T.alloc_shared([BS, 1], "float32")

            T.copy(BetaFlat[bb, s0 * H : (s0 + BS) * H], beta_tile)
            T.tile.arith_progression(off_i32, bh * 4, H * 4)
            T.reinterpretcast(off_u32, off_i32, "uint32_t")
            T.tile.gather(bt, beta_tile, off_u32, 0)
            # resume the dbeta accumulator from the k2-kernel
            T.copy(dbeta_flat[cid, 0:BS], acc)

            for i_v in T.Pipelined(DV // bDV, num_stages=2):
                T.copy(
                    V[bb, s0 : s0 + BS, bh, i_v * bDV : (i_v + 1) * bDV],
                    v_ub,
                )
                # A split stays in the loop: the Pipelined software pipeline
                # already overlaps the loop-invariant cast chain with slice
                # gemms (manual prologue hoist measured neutral-to-worse and
                # costs UB lifetime pressure)
                T.copy(A[bb, s0 : s0 + BS, bh, :], a_f32)
                T.copy(a_f32, a_hi_ub)
                T.copy(a_hi_ub, a_hi_f32)
                T.tile.sub(a_lo_f32, a_f32, a_hi_f32)
                T.copy(a_lo_f32, a_lo_ub)
                T.copy(a_hi_ub, a_hi_l1)
                T.copy(a_lo_ub, a_lo_l1)

                T.copy(
                    du[bb, s0 : s0 + BS, bh, i_v * bDV : (i_v + 1) * bDV],
                    du_f32,
                )
                T.copy(du_f32, du_hi_ub)
                T.copy(du_hi_ub, du_hi_f32)
                T.tile.sub(du_lo_f32, du_f32, du_hi_f32)
                T.copy(du_lo_f32, du_lo_ub)
                T.copy(du_hi_ub, du_hi_l1)
                T.copy(du_lo_ub, du_lo_l1)

                # Y = A_c^T @ du (fp16x2)
                T.gemm_v0(a_hi_l1, du_hi_l1, y_frag, transpose_A=True, init=True)
                T.gemm_v0(a_hi_l1, du_lo_l1, y_frag, transpose_A=True, init=False)
                T.gemm_v0(a_lo_l1, du_hi_l1, y_frag, transpose_A=True, init=False)
                T.copy(y_frag, y_ub)

                # dv = Y * beta (row-scalar broadcast, full-tile mul)
                T.tile.broadcast(bt2d, bt, axis=1)
                T.tile.mul(dv_ub, y_ub, bt2d)
                T.copy(
                    dv_ub,
                    dv[bb, s0 : s0 + BS, bh, i_v * bDV : (i_v + 1) * bDV],
                )

                # dbeta partial: rowsum(Y * v)
                T.tile.mul(r_tmpv, y_ub, v_ub)
                T.reduce_sum(r_tmpv, r_part, dim=1, clear=True)
                T.tile.add(acc, acc, r_part)

            # ---- dbeta scatter via GM scratch staging (verified R3 chain).
            # Ablation: moving the scatter to the da kernel (barrier-free)
            # is net-negative -- the barrier overlaps the epilogue here, and
            # da has no loop to hide the extra load+scatter.
            T.copy(acc, dbeta_flat[cid, 0:BS])
            T.set_flag("MTE3", "MTE2", 0)
            T.wait_flag("MTE3", "MTE2", 0)
            T.copy(dbeta_flat_v[cid, 0:BS, 0:1], dbeta2)
            T.copy(dbeta2, dbeta[bb, s0 : s0 + BS, bh])

    return main


# ============================================================================
# bwd kernel 3: Z = S @ A_c^T (single gemm, one C/V round trip per block)
# ============================================================================
@tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS)
def fla_wy_bwd_z_kernel(B, S, H, chunk_size=64):
    BS = chunk_size
    nt, GRID = _grid_dims(B, S, H, BS)

    @T.prim_func
    def main(
        S_in: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
        A: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
        Z: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % nt
            bbh = cid // nt
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            a_f32 = T.alloc_shared([BS, BS], "float32")
            s_f32 = T.alloc_shared([BS, BS], "float32")
            a_hi_f32 = T.alloc_shared([BS, BS], "float32")
            a_lo_f32 = T.alloc_shared([BS, BS], "float32")
            a_hi_ub = T.alloc_shared([BS, BS], "float16")
            a_lo_ub = T.alloc_shared([BS, BS], "float16")
            s_hi_f32 = T.alloc_shared([BS, BS], "float32")
            s_lo_f32 = T.alloc_shared([BS, BS], "float32")
            s_hi_ub = T.alloc_shared([BS, BS], "float16")
            s_lo_ub = T.alloc_shared([BS, BS], "float16")
            a_hi_l1 = T.alloc_shared([BS, BS], "float16")
            a_lo_l1 = T.alloc_shared([BS, BS], "float16")
            s_hi_l1 = T.alloc_shared([BS, BS], "float16")
            s_lo_l1 = T.alloc_shared([BS, BS], "float16")
            z_frag = T.alloc_fragment([BS, BS], "float32")
            z_ub = T.alloc_shared([BS, BS], "float32")

            T.copy(A[bb, s0 : s0 + BS, bh, :], a_f32)
            T.copy(a_f32, a_hi_ub)
            T.copy(a_hi_ub, a_hi_f32)
            T.tile.sub(a_lo_f32, a_f32, a_hi_f32)
            T.copy(a_lo_f32, a_lo_ub)
            T.copy(a_hi_ub, a_hi_l1)
            T.copy(a_lo_ub, a_lo_l1)

            T.copy(S_in[bb, s0 : s0 + BS, bh, :], s_f32)
            T.copy(s_f32, s_hi_ub)
            T.copy(s_hi_ub, s_hi_f32)
            T.tile.sub(s_lo_f32, s_f32, s_hi_f32)
            T.copy(s_lo_f32, s_lo_ub)
            T.copy(s_hi_ub, s_hi_l1)
            T.copy(s_lo_ub, s_lo_l1)

            # Z = S @ A_c^T (fp16x2)
            T.gemm_v0(s_hi_l1, a_hi_l1, z_frag, transpose_B=True, init=True)
            T.gemm_v0(s_hi_l1, a_lo_l1, z_frag, transpose_B=True, init=False)
            T.gemm_v0(s_lo_l1, a_hi_l1, z_frag, transpose_B=True, init=False)
            T.copy(z_frag, z_ub)
            T.copy(z_ub, Z[bb, s0 : s0 + BS, bh, :])

    return main


# ============================================================================
# bwd kernel 4: dA = -strict_tril(A_c^T @ Z) (single gemm + mask)
# ============================================================================
@tilelang.jit(out_idx=[3], pass_configs=PASS_CONFIGS)
def fla_wy_bwd_da_kernel(B, S, H, chunk_size=64):
    """Single gemm + mask: dA = -strict_tril(A_c^T @ Z)."""
    BS = chunk_size
    nt, GRID = _grid_dims(B, S, H, BS)

    @T.prim_func
    def main(
        A: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
        Z: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
        Msk: T.Tensor((BS, BS), "float32"),  # type: ignore
        dA: T.Tensor((B, S, H, BS), "float32"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % nt
            bbh = cid // nt
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            a_f32 = T.alloc_shared([BS, BS], "float32")
            z_f32 = T.alloc_shared([BS, BS], "float32")
            a_hi_f32 = T.alloc_shared([BS, BS], "float32")
            a_lo_f32 = T.alloc_shared([BS, BS], "float32")
            a_hi_ub = T.alloc_shared([BS, BS], "float16")
            a_lo_ub = T.alloc_shared([BS, BS], "float16")
            z_hi_f32 = T.alloc_shared([BS, BS], "float32")
            z_lo_f32 = T.alloc_shared([BS, BS], "float32")
            z_hi_ub = T.alloc_shared([BS, BS], "float16")
            z_lo_ub = T.alloc_shared([BS, BS], "float16")
            a_hi_l1 = T.alloc_shared([BS, BS], "float16")
            a_lo_l1 = T.alloc_shared([BS, BS], "float16")
            z_hi_l1 = T.alloc_shared([BS, BS], "float16")
            z_lo_l1 = T.alloc_shared([BS, BS], "float16")
            t2_frag = T.alloc_fragment([BS, BS], "float32")
            t2_ub = T.alloc_shared([BS, BS], "float32")
            msk_ub = T.alloc_shared([BS, BS], "float32")

            T.copy(A[bb, s0 : s0 + BS, bh, :], a_f32)
            T.copy(a_f32, a_hi_ub)
            T.copy(a_hi_ub, a_hi_f32)
            T.tile.sub(a_lo_f32, a_f32, a_hi_f32)
            T.copy(a_lo_f32, a_lo_ub)
            T.copy(a_hi_ub, a_hi_l1)
            T.copy(a_lo_ub, a_lo_l1)

            T.copy(Z[bb, s0 : s0 + BS, bh, :], z_f32)
            T.copy(z_f32, z_hi_ub)
            T.copy(z_hi_ub, z_hi_f32)
            T.tile.sub(z_lo_f32, z_f32, z_hi_f32)
            T.copy(z_lo_f32, z_lo_ub)
            T.copy(z_hi_ub, z_hi_l1)
            T.copy(z_lo_ub, z_lo_l1)

            # dA = A_c^T @ Z (fp16x2), then mask + negate
            T.gemm_v0(a_hi_l1, z_hi_l1, t2_frag, transpose_A=True, init=True)
            T.gemm_v0(a_hi_l1, z_lo_l1, t2_frag, transpose_A=True, init=False)
            T.gemm_v0(a_lo_l1, z_hi_l1, t2_frag, transpose_A=True, init=False)
            T.copy(t2_frag, t2_ub)
            T.copy(Msk[0:BS, 0:BS], msk_ub)
            T.tile.mul(t2_ub, t2_ub, msk_ub)
            T.tile.mul(t2_ub, t2_ub, -1.0)
            T.copy(t2_ub, dA[bb, s0 : s0 + BS, bh, :])

    return main


# ============================================================================
# Host wrappers (upstream naming: recompute_w_u_fwd / prepare_wy_repr_bwd).
# Only zero-copy metadata views + output allocation here.
# ============================================================================
def _check_shape(B, S, H, chunk_size, A):
    assert S % chunk_size == 0, "S must be divisible by chunk_size"
    assert chunk_size == 64, "chunk_size fixed at 64 (block_S semantic)"
    assert A.shape[-1] == chunk_size, "A last dim must equal chunk_size"


_MSK_CACHE = {}


def _get_msk(device, chunk_size):
    """Strict-tril constant, constructed ONCE per (device, chunk) and cached.

    Per-call reconstruction costs a CPU tril + a (potentially synchronizing)
    H2D copy inside the timed region.
    """
    key = (str(device), chunk_size)
    if key not in _MSK_CACHE:
        _MSK_CACHE[key] = torch.tril(
            torch.ones(chunk_size, chunk_size, dtype=torch.float32), -1
        ).to(device)
    return _MSK_CACHE[key]


def fla_wy_fast_fwd(k, v, beta, gk, A, chunk_size=64, use_kg=True):
    """fwd host API. k/v/A bf16, gk fp32, beta [B,S,H] bf16 -> (w, u, kg) bf16.

    use_kg=False is outside the migration scope (proto: use_kg=True fixed);
    the parameter is kept for interface parity.
    """
    assert use_kg, "use_kg=False is not in the migration scope (proto)"
    B, S, H, DK = k.shape
    DV = v.shape[-1]
    _check_shape(B, S, H, chunk_size, A)
    beta_flat = beta.view(B, S * H)

    u_kernel = fla_wy_fwd_u_kernel(B, S, H, DV, chunk_size, _pick_block_d(DV))
    u = u_kernel(v, beta_flat, A)
    wkg_kernel = fla_wy_fwd_wkg_kernel(B, S, H, DK, chunk_size, _pick_block_d(DK))
    w, kg = wkg_kernel(k, beta_flat, gk, A)
    # no internal synchronize: matches the upstream wrapper semantics (the
    # do_bench harness owns synchronization; an internal sync would expose
    # host-side JIT dispatch latency inside the device-side measurement)
    return w, u, kg


def fla_wy_fast_bwd(k, v, beta, gk, A, dw, du, dk, dg, chunk_size=64):
    """bwd host API (all fp32) -> (dA, dk2, dv, dbeta, dg2) fp32.

    fp16-range note: gemm operands are split into fp16 hi/lo pairs; operands
    with |x| > 65504 lose accuracy (see module docstring).
    """
    B, S, H, DK = k.shape
    DV = v.shape[-1]
    _check_shape(B, S, H, chunk_size, A)
    device = k.device
    grid = (S // chunk_size) * B * H
    # strict lower-triangular constant (cached; one-shot construction)
    msk = _get_msk(device, chunk_size)
    beta_flat = beta.view(B, S * H)

    # six single-gemm-group kernels (multi-gemm-group loops corrupt the L0
    # staging on this toolchain; see module docstring)
    k1 = fla_wy_bwd_k1_kernel(B, S, H, DK, chunk_size, _pick_block_d(DK))
    dA_part = k1(k, beta_flat, gk, dw)
    k2 = fla_wy_bwd_k2_kernel(B, S, H, DK, chunk_size, _pick_block_d(DK))
    dk2, dg2, dbeta_flat = k2(k, beta_flat, gk, A, dw, dk, dg)
    v1 = fla_wy_bwd_v1_kernel(B, S, H, DV, chunk_size, _pick_block_d(DV))
    s_out = v1(v, beta_flat, du, msk, dA_part)
    v2 = fla_wy_bwd_v2_kernel(B, S, H, DV, chunk_size, _pick_block_d(DV))
    dv, dbeta = v2(
        v,
        beta_flat,
        A,
        du,
        dbeta_flat,
        dbeta_flat.view(grid, chunk_size, 1),
    )
    z_kernel = fla_wy_bwd_z_kernel(B, S, H, chunk_size)
    z = z_kernel(s_out, A)
    da_kernel = fla_wy_bwd_da_kernel(B, S, H, chunk_size)
    dA = da_kernel(A, z, msk)
    # no internal synchronize (upstream wrapper parity; see fwd wrapper)
    return dA, dk2, dv, dbeta, dg2


__all__ = [
    "fla_wy_fast_fwd",
    "fla_wy_fast_bwd",
    "fla_wy_fwd_u_kernel",
    "fla_wy_fwd_wkg_kernel",
    "fla_wy_bwd_k1_kernel",
    "fla_wy_bwd_k2_kernel",
    "fla_wy_bwd_v1_kernel",
    "fla_wy_bwd_v2_kernel",
    "fla_wy_bwd_z_kernel",
    "fla_wy_bwd_da_kernel",
    "PASS_CONFIGS",
]
