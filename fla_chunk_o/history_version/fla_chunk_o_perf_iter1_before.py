"""fla_chunk_o: KDA chunked forward output O (single kernel) on Ascend 910B3.

o = inter-chunk (qg @ h) + intra-chunk (tril_incl_diag(A) @ v), where
qg = (q * scale) * 2^g computed as exp(g * ln2) (exp2 has no Ascend
backend mapping). Inputs are consumed-only (A/h produced upstream).

Semantics: ref_gpu/fla_chunk_o.py chunk_gla_fwd_kernel_o (Triton, frozen);
code-level origin: ref_gpu/fla_chunk_o_tilelang_fwd.py (upstream TileLang
CUDA tilelang_chunk_fwd_o).

Developer mode, threads=1, bshd layout consumed directly (no host-side
views, no .contiguous(), no aclnn calls in the wrapper).

Ascend toolchain adaptations (inherited from the same-family fla_wy_fast
delivery, all empirically verified on 910B3):

1. One-dimensional T.Kernel(NT*B*H) with cid decomposed on-core
   (3-D kernels unsupported; threads restricted to 1/2 -> use 1).
2. The i_k Pipelined body is forced to EXACTLY 2 iterations
   (block_DK = DK//2): extent-1 crashes codegen and extent>=3 corrupts
   the double-buffer wrap. Combined with the bf16 gemm_v0 K>=32 fractal
   (K = block_DK) this restricts DK to {64, 96, 128}.
3. All gemm_v0 inputs are bf16 (fp32 A/B produces 1e37-class garbage on
   this toolchain). qg is split into a bf16 hi/lo pair and accumulated
   with two gemms (~2^-18 relative on o_inter): a single bf16 cast
   (~2^-9 relative, the Triton semantic authority's b_qg.to(bf16))
   measurably trips the bf16 mixed tolerance on cancellation elements
   (|o| << |o_inter|, L0 ratio 0.988 < 0.99) -- DESIGN SS9.3 fallback.
4. Every bf16<->fp32 conversion happens UB-internally via a two-step copy
   chain; GM<->UB cast-on-copy codegen is broken (mismatched template).
5. L0C(fp32) -> GM(bf16) direct fixpipe is broken: the output uses the
   three-step chain frag -> UB(fp32) -> UB(bf16) -> GM.
6. V->C exchanges (UB -> L1) only happen inside the Pipelined body (a
   prologue exchange gets mis-versioned by the pipeline pass). The
   A_mask -> A_l1 exchange is therefore repeated idempotently inside the
   loop; the prologue stays pure V-side (GM->UB DMA + V-internal math).
7. The tril mask is a constant Msk[64,64] GM input (inclusive diagonal,
   tril(ones, 0)) multiplied in the fp32 domain via T.tile.mul --
   T.if_then_else is unavailable in tile ops. Exact in bf16 (0/1 mask).
8. Element math uses buffer-level T.tile.* instructions only (element
   T.Parallel loops are the -26%..-71%/kernel antipattern).
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


def _pick_block_dk(dim: int) -> int:
    """block_DK forcing EXACTLY 2 Pipelined iterations with K >= 32.

    Toolchain constraints (see module docstring items 2/3): the Pipelined
    body crashes codegen at extent 1 and corrupts the 3rd iteration's
    double-buffer wrap, so block = DK // 2 with DK // block == 2. The
    inter-chunk gemm uses K = block_DK, which must satisfy the bf16
    gemm_v0 K >= 32 fractal. Together: DK in {64, 96, 128}. DK=32 has no
    valid exact-2 block (block 16 violates K >= 32); other 16-multiples
    (80, 112, ...) likewise -- documented limitation, rejected here.
    """
    blk = dim // 2
    if dim % 2 == 0 and blk in (32, 48, 64):
        return blk
    raise ValueError(
        f"DK={dim} unsupported: the Ascend Pipelined body requires exactly 2"
        " iterations with block_DK = DK//2 >= 32 (bf16 gemm K-fractal), so DK"
        " must be one of {64, 96, 128} (documented limitation)"
    )


def _pick_block_dv(dim: int) -> int:
    """block_DV = DV single block (IO-unique: every tensor read/written once).

    Constraints: the intra/inter gemm N = block_DV must satisfy the bf16
    gemm_v0 N >= 16 fractal; a single block keeps the o accumulator
    [64, DV] fp32 within the 128KB L0C budget -> DV <= 128. Larger DV
    would need a nested bv loop (no precedent, out of scope).
    """
    if dim % 16 == 0 and 16 <= dim <= 128:
        return dim
    raise ValueError(
        f"DV={dim} unsupported: DV must be a 16-multiple in [16, 128]"
        " (bf16 gemm N-fractal + single-block L0C budget; documented"
        " limitation)"
    )


# out_idx uses the ABSOLUTE index of O (Q,V,G,A,Hstate,Msk,O -> O = 6):
# AUTO_CV_COMBINE auto-injects GM workspace params AFTER the declared
# signature, which shifts negative indices (impl_final.py docstring item 7).
@tilelang.jit(out_idx=[6], pass_configs=PASS_CONFIGS)
def fla_chunk_o_kernel(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size=64,
    scale=None,
    block_DK=None,
    block_DV=None,
):
    """Single kernel: o = qg @ h + tril_incl_diag(A) @ v per (chunk, b, h)."""
    if scale is None:
        scale = DK**-0.5
    BS = chunk_size
    NT = S // BS
    bDK = block_DK if block_DK is not None else _pick_block_dk(DK)
    bDV = block_DV if block_DV is not None else _pick_block_dv(DV)
    GRID = NT * B * H

    @T.prim_func
    def main(
        Q: T.Tensor((B, S, H, DK), "bfloat16"),  # type: ignore
        V: T.Tensor((B, S, H, DV), "bfloat16"),  # type: ignore
        G: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        A: T.Tensor((B, S, H, BS), "bfloat16"),  # type: ignore
        Hstate: T.Tensor((B, NT, H, DK, DV), "bfloat16"),  # type: ignore
        Msk: T.Tensor((BS, BS), "float32"),  # type: ignore
        O: T.Tensor((B, S, H, DV), "bfloat16"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bs = cid % NT
            bbh = cid // NT
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            # ---- buffers (placement planned by the compiler; the *_l1
            # names mark gemm operands that must land in L1, the fp32
            # math domains land in UB) ----
            A_l1 = T.alloc_shared([BS, BS], "bfloat16")  # intra gemm LHS (masked)
            V_l1 = T.alloc_shared([BS, bDV], "bfloat16")  # intra gemm RHS
            H_l1 = T.alloc_shared([bDK, bDV], "bfloat16")  # inter gemm RHS (i_k slice)
            qg_hi_l1 = T.alloc_shared([BS, bDK], "bfloat16")  # inter gemm LHS hi (i_k slice)
            qg_lo_l1 = T.alloc_shared([BS, bDK], "bfloat16")  # inter gemm LHS lo (i_k slice)
            a_half = T.alloc_shared([BS, BS], "bfloat16")  # A load staging
            a_f32 = T.alloc_shared([BS, BS], "float32")  # A fp32 domain
            msk_ub = T.alloc_shared([BS, BS], "float32")  # tril constant
            a_mask = T.alloc_shared([BS, BS], "bfloat16")  # masked A (bf16, exact)
            q_half = T.alloc_shared([BS, bDK], "bfloat16")  # q load staging
            q_ub = T.alloc_shared([BS, bDK], "float32")  # q fp32 domain
            g_ub = T.alloc_shared([BS, bDK], "float32")  # g
            gL_ub = T.alloc_shared([BS, bDK], "float32")  # g * ln2
            gexp_ub = T.alloc_shared([BS, bDK], "float32")  # 2^g
            qg_ub = T.alloc_shared([BS, bDK], "float32")  # qg fp32
            qg_hi_ub = T.alloc_shared([BS, bDK], "bfloat16")  # qg hi (bf16)
            qg_hi_f32 = T.alloc_shared([BS, bDK], "float32")  # hi, exact fp32
            qg_lo_f32 = T.alloc_shared([BS, bDK], "float32")  # lo = qg - hi (fp32)
            qg_lo_ub = T.alloc_shared([BS, bDK], "bfloat16")  # qg lo (bf16)
            o_frag = T.alloc_fragment([BS, bDV], "float32")  # L0C accumulator
            o_ub = T.alloc_shared([BS, bDV], "float32")  # L0C -> UB staging
            o_half = T.alloc_shared([BS, bDV], "bfloat16")  # output cast staging

            # ---- prologue (V side: GM->UB DMA + V-internal math only;
            # NO V->C exchange -- prologue exchanges get mis-versioned) ----
            T.copy(A[bb, s0 : s0 + BS, bh, :], a_half)  # GM bf16 -> UB bf16
            T.copy(a_half, a_f32)  # UB-internal cast -> fp32
            T.copy(Msk[0:BS, 0:BS], msk_ub)  # constant (L2-resident)
            T.tile.mul(a_f32, a_f32, msk_ub)  # tril incl diag (fp32 domain)
            T.copy(a_f32, a_mask)  # cast -> bf16 (exact: 0/1 mask)

            # ---- i_k loop (inter-chunk, exactly 2 iterations) ----
            for i_k in T.Pipelined(DK // bDK, num_stages=2):
                T.copy(
                    Q[bb, s0 : s0 + BS, bh, i_k * bDK : (i_k + 1) * bDK],
                    q_half,
                )
                T.copy(q_half, q_ub)  # bf16 -> fp32 (UB-internal cast)
                T.copy(
                    G[bb, s0 : s0 + BS, bh, i_k * bDK : (i_k + 1) * bDK],
                    g_ub,
                )  # fp32 direct
                T.tile.mul(gL_ub, g_ub, LN2)  # g * ln2
                T.tile.exp(gexp_ub, gL_ub)  # 2^g = exp(g * ln2)
                T.tile.mul(q_ub, q_ub, scale)  # q * scale
                T.tile.mul(qg_ub, q_ub, gexp_ub)  # qg = q*scale*2^g (fp32)
                # bf16 hi/lo split of qg: a single cast leaves a ~2^-9
                # relative error on o_inter that trips the mixed tolerance
                # on cancellation elements (|o| << |o_inter|; measured L0
                # ratio 0.988 < 0.99). hi/lo double gemm -> ~2^-18 relative,
                # the DESIGN SS9.3 sanctioned fallback (impl_final fwd_u
                # pattern: 2 gemms, L0C accumulation).
                T.copy(qg_ub, qg_hi_ub)  # fp32 -> bf16 (hi)
                T.copy(qg_hi_ub, qg_hi_f32)  # bf16 -> fp32 (exact)
                T.tile.sub(qg_lo_f32, qg_ub, qg_hi_f32)  # lo = qg - hi
                T.copy(qg_lo_f32, qg_lo_ub)  # fp32 -> bf16 (lo)
                T.copy(qg_hi_ub, qg_hi_l1)  # V->C exchanges (inside the loop)
                T.copy(qg_lo_ub, qg_lo_l1)
                T.copy(a_mask, A_l1)  # idempotent repeat (inside the loop)
                T.copy(
                    Hstate[
                        bb,
                        bs,
                        bh,
                        i_k * bDK : (i_k + 1) * bDK,
                        0:bDV,
                    ],
                    H_l1,
                )  # 5D GM slice -> L1 (innermost DV contiguous)
                T.gemm_v0(qg_hi_l1, H_l1, o_frag, init=(i_k == 0))  # o += qg_hi @ h
                T.gemm_v0(qg_lo_l1, H_l1, o_frag, init=False)  # o += qg_lo @ h

            # ---- epilogue (C gemm -> V output; single C->V hop) ----
            T.copy(V[bb, s0 : s0 + BS, bh, 0:bDV], V_l1)  # GM -> L1
            T.gemm_v0(A_l1, V_l1, o_frag, init=False)  # o += A_mask @ v
            T.copy(o_frag, o_ub)  # L0C -> UB fp32
            T.copy(o_ub, o_half)  # cast -> bf16
            T.copy(o_half, O[bb, s0 : s0 + BS, bh, 0:bDV])  # -> GM

    return main


# ============================================================================
# Host wrapper (upstream naming: chunk_gla_fwd_o_gk). Only shape checks +
# Msk constant caching + kernel invocation; no data mutation on the host.
# ============================================================================
def _check_shape(B, S, H, chunk_size, A):
    assert S % chunk_size == 0, "S must be divisible by chunk_size"
    assert chunk_size == 64, "chunk_size fixed at 64 (block_S semantic)"
    assert A.shape[-1] == chunk_size, "A last dim must equal chunk_size"


_MSK_CACHE = {}


def _get_msk(device, chunk_size):
    """Inclusive-tril constant (diagonal=0), built ONCE per (device, chunk).

    Per-call reconstruction would cost a CPU tril + a potentially
    synchronizing H2D copy inside the timed region.
    """
    key = (str(device), chunk_size)
    if key not in _MSK_CACHE:
        _MSK_CACHE[key] = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.float32), 0).to(device)
    return _MSK_CACHE[key]


def fla_chunk_o(q, v, g, A, h, scale=None, chunk_size=64):
    """Host API (upstream: chunk_gla_fwd_o_gk, use_exp2=True, fixed length).

    q[B,S,H,DK] bf16, v[B,S,H,DV] bf16, g[B,S,H,DK] fp32 (cumulative log2
    decay), A[B,S,H,64] bf16 (intra-chunk attention), h[B,S//64,H,DK,DV]
    bf16 (chunk-level hidden state) -> o[B,S,H,DV] bf16.
    """
    B, S, H, DK = q.shape
    DV = v.shape[-1]
    _check_shape(B, S, H, chunk_size, A)
    if scale is None:
        scale = DK**-0.5
    bDK = _pick_block_dk(DK)
    bDV = _pick_block_dv(DV)
    msk = _get_msk(q.device, chunk_size)
    kernel = fla_chunk_o_kernel(B, S, H, DK, DV, chunk_size, scale, bDK, bDV)
    o = kernel(q, v, g, A, h, msk)
    # no internal synchronize: upstream wrapper parity (the do_bench harness
    # owns synchronization; an internal sync would expose host-side JIT
    # dispatch latency inside the device-side measurement)
    return o


__all__ = [
    "fla_chunk_o",
    "fla_chunk_o_kernel",
    "PASS_CONFIGS",
]
