"""fla_chunk_intra_token_parallel: KDA intra-chunk per-token decay inner
products Aqk/Akk on Ascend 910B3 (single kernel, pure Vector).

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

Ascend restructuring (DESIGN_fla_chunk_intra_token_parallel.md, all
deviations from the upstream CUDA code are design decisions; Stage 3 perf
tuning restructured the pairwise loop -- see perf_tuning/ for the analysis):

1. Subchunk-parallel blocks: one (batch, subchunk, head-group) per block
   instead of one token per block -- eliminates the 8.5x k/g GM re-read and
   makes every loop bound compile-time static (upstream T.Pipelined(loops)
   has a per-block dynamic extent, unsupported on Ascend).
2. One-dimensional T.Kernel(B * NSC * HG) with (cid, vid): 3-D kernels are
   unsupported and threads is restricted to 1/2 (default dual vector
   threads; vid splits the BH-head group into BH_V = BH // 2 per thread).
3. exp2(x) is computed as exp(x * ln2): T.exp2 has no Ascend backend
   mapping (g is stored in log2 domain, so exp((g_i - g_j) * ln2) equals
   2^(g_i - g_j) exactly in the fp32 working domain).
4. All bf16<->fp32 conversions happen UB-internally (T.tile.cast / UB<->UB
   copy); GM<->UB cross-dtype copies are a broken codegen path.
5. Outputs go through an fp32 staging buffer, a block CAST_RINT to bf16 and
   per-head 2D strided T.copy bursts to GM (one [SCS, CS] row-group per
   head; a single copy over a 3-active-dim GM region drops the outer dim
   in the 2D DMA model, so head must stay a point index).
6. Element math uses buffer-level T.tile.* ops + T.reduce_sum(dim=-1,
   clear=True) only. The i axis is unrolled at parse time via T.macro
   calls with literal i (16 guarded call sites, `if i < SCS` on the
   captured Python int evaluates as a parse-time bool), so all region
   extents are compile-time constants and the kernel body contains no
   runtime branch: the AUTO_SYNC pass wraps every TIR-level IfThenElse in
   PipeBarrier<PIPE_ALL> and every scalar UB store in a SetFlag/WaitFlag
   pair, which made the pair-per-(i,j) guard + scatter structure
   sync-bound (measured 19.15 ms @ bench shape, ~90% barrier overhead; see
   perf_tuning/fla_chunk_intra_token_parallel_perf_log.md).
7. Row-batched pairwise computation (DESIGN R5 lever): for token i the
   whole j range [0, i] is processed as one 2D tile -- broadcast(g_i) ->
   sub -> exp -> kgj = k_j * e -> broadcast(q_i) -> mul -> reduce_sum
   writes the Aqk window run directly into the fp32 staging region. Akk
   reuses kgj on [0, i) (diagonal excluded) and folds the scalar beta
   AFTER the reduction (beta[i,h] * sum_d k_i*kgj is mathematically
   identical to sum_d (beta*k_i)*kgj); the fold is a [BH_V, i] region mul
   by a broadcast beta row, leaving the zero diagonal untouched.
8. Head-major compute domain [BH_V, SCS, DK]: per-head [SCS, DK] slabs are
   contiguous, which enables both the row-batched 2D tiles and aggregated
   per-head loads ([SCS, DK] 2D strided copies, 12+1 loads instead of 48+1
   per block). i == 0 skips the exp path entirely (exp(0) == 1 exactly, so
   Aqk_diag = rowsum(q_0 * k_0)).
9. beta keeps the padded [SCS, 16] staging (narrow [SCS, BH_V] rows must
   land in 32B UB slots) and the row-gap GM copy of the current version.

Supported domain (proto_fla_chunk_intra_token_parallel.yaml): fixed-length
sequences with S % CS == 0, CS % SCS == 0, H % BH == 0 (BH adapts
8 -> 4 -> 2), DK % 16 == 0 and DK >= 32; q/k/beta bfloat16 + gk float32
inputs, bfloat16 outputs. varlen (cu_seqlens) is out of scope.
"""

import torch
import tilelang
from tilelang import language as T

# 2^g is computed as exp(g * ln2): exp2 has no Ascend backend mapping.
LN2 = 0.6931471805599453

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@T.macro
def _aqk_diag(h, w, q_f, k_f, pq, aqk_s):
    """i == 0 fast path: exp((g_0 - g_0) * ln2) == 1 exactly, so the
    diagonal Aqk value is rowsum(q_0 * k_0) with no decay chain."""
    T.tile.mul(pq[0:1, :], q_f[h, 0:1, :], k_f[h, 0:1, :])
    T.reduce_sum(pq[0:1, :], aqk_s[h, 0, w : w + 1], dim=-1, clear=True)


@T.macro
def _aqk_akk_rows(i, h, w, bh_v, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f, aqk_s, akk_s):
    """Row-batched pairwise step for token i (i >= 1, parse-time literal),
    head h (TIR loop var), Aqk window base w (TIR expr). All region extents
    are Python constants. The Aqk row covers j in [0, i]; the Akk row covers
    j in [0, i) (diagonal stays 0), with beta folded into the reduce input
    via a scalar Muls (beta[i, h] is a per-(i, h) scalar and factors out of
    the DK sum; the flat-index buffer read is the ascend_muls BufferLoad
    path)."""
    # Aqk row: e = exp(g_i - g_j) for j in [0, i]  (scale == 1.0)
    T.tile.broadcast(bc_row[0 : i + 1, :], g2[h, i : i + 1, :], axis=0)
    T.tile.sub(e[0 : i + 1, :], bc_row[0 : i + 1, :], g2[h, 0 : i + 1, :])
    T.tile.exp(e[0 : i + 1, :], e[0 : i + 1, :])
    T.tile.mul(kgj[0 : i + 1, :], k_f[h, 0 : i + 1, :], e[0 : i + 1, :])
    T.tile.broadcast(bc_row[0 : i + 1, :], q_f[h, i : i + 1, :], axis=0)
    T.tile.mul(pq[0 : i + 1, :], bc_row[0 : i + 1, :], kgj[0 : i + 1, :])
    T.reduce_sum(pq[0 : i + 1, :], aqk_s[h, i, w : w + i + 1], dim=-1, clear=True)
    # Akk row: j in [0, i), beta folded before the reduction (flat index
    # into the padded [SCS, 16] fp32 beta buffer: row stride is the fixed
    # pad width 16, so beta[i, h] lives at flat element i * 16 + h)
    T.tile.broadcast(bc_row[0:i, :], k_f[h, i : i + 1, :], axis=0)
    T.tile.mul(pq[0:i, :], bc_row[0:i, :], kgj[0:i, :])
    T.tile.mul(pq[0:i, :], pq[0:i, :], beta_f[i * 16 + h, 0])
    T.reduce_sum(pq[0:i, :], akk_s[h, i, 0:i], dim=-1, clear=True)


@tilelang.jit(out_idx=[-2, -1], pass_configs=PASS_CONFIGS)
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
    producing (Aqk [B,S,H,CS], Akk [B,S,H,SCS]), both output_dtype."""
    VEC_NUM = 2  # default dual vector threads: (cid, vid)
    BH_V = BH // VEC_NUM  # heads handled by each vid
    NSC = S // SCS  # subchunks per sequence
    HG = H // BH  # head groups
    # Akk staging row width: MTE3 reads advance whole 32B blocks per burst,
    # so sub-32B row pitches (SCS=8 -> 16B bf16 rows) scramble the writeback;
    # pad the rows to at least 16 bf16 (= 32B).
    AKK_W = max(SCS, 16)

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
        Aqk: T.Tensor(Aqk_shape, dtype=output_dtype),  # type: ignore
        Akk: T.Tensor(Akk_shape, dtype=output_dtype),  # type: ignore
    ):
        with T.Kernel(B * NSC * HG, is_npu=True) as (cid, vid):
            # cid decomposition, subchunk-major (same-subchunk head groups
            # are scheduled adjacently -> wave-local GM reuse)
            sc = cid % NSC
            hg = (cid // NSC) % HG
            bb = cid // (NSC * HG)
            c = sc // (CS // SCS)  # chunk index
            i_s = sc % (CS // SCS)  # subchunk index inside the chunk
            i_ts = c * CS + i_s * SCS  # first token of the subchunk
            h0v = hg * BH + vid * BH_V  # first head of this vid

            # ---- UB buffers (Developer mode: alloc_shared maps to UB) ----
            # Head-major load staging + fp32 compute domain: per-head
            # [SCS, DK] slabs contiguous (enables row-batched 2D tiles and
            # per-head aggregated strided copies). Row r of slab h is token
            # i_ts + r, head h0v + h.
            q_h = T.alloc_shared([BH_V, SCS, DK], input_dtype)
            k_h = T.alloc_shared([BH_V, SCS, DK], input_dtype)
            q_f = T.alloc_shared([BH_V, SCS, DK], gate_dtype)
            k_f = T.alloc_shared([BH_V, SCS, DK], gate_dtype)
            g2 = T.alloc_shared([BH_V, SCS, DK], gate_dtype)  # gk * ln2
            # beta staging: narrow [SCS, BH_V] rows (8B) must land in 32B UB
            # slots (MTE-filled narrow rows advance 32B per burst), so the
            # load staging rows are padded to 16 bf16 (= 32B) and the DMA
            # target is the [*, 0:BH_V] column region of that padded buffer.
            # The fp32 copy keeps the same padded [SCS, 16] layout (32B
            # rows on both sides -- an intra-UB copy with 16B rows violates
            # the 32B MTE burst alignment); beta[i, h] is then read back at
            # flat element i * 16 + h through the ascend_muls scalar path.
            beta_h = T.alloc_shared([SCS, 16], input_dtype)
            beta_f = T.alloc_shared([SCS, 16], gate_dtype)
            # per-(h, i) 2D scratch ([0:i+1] row windows of these)
            e = T.alloc_shared([SCS, DK], gate_dtype)
            kgj = T.alloc_shared([SCS, DK], gate_dtype)
            pq = T.alloc_shared([SCS, DK], gate_dtype)  # reused for the Akk row
            bc_row = T.alloc_shared([SCS, DK], gate_dtype)  # row-broadcast dst
            # fp32 staging, head-major [BH_V, SCS, CS] / [BH_V, SCS, SCS]:
            # the per-head [SCS, CS] writeback slice is then a contiguous UB
            # slab (a mid-dim point slice like [:, h, :] loses its physical
            # row stride in the copy lowering), and the per-(h, i) reduce
            # destination stays a flat-contiguous 1D run.
            aqk_s = T.alloc_shared([BH_V, SCS, CS], gate_dtype)
            akk_s = T.alloc_shared([BH_V, SCS, AKK_W], gate_dtype)
            aqk_h = T.alloc_shared([BH_V, SCS, CS], output_dtype)
            akk_h = T.alloc_shared([BH_V, SCS, AKK_W], output_dtype)

            # 1. Loads: one [SCS, DK] 2D strided row-group per (tensor, head)
            #    (head is a point index -> 2 active dims; a head RANGE would
            #    make 3 active dims, which the 2D DMA model truncates).
            for h in T.serial(BH_V):
                T.copy(Q[bb, i_ts : i_ts + SCS, h0v + h, :], q_h[h, :, :])
                T.copy(K[bb, i_ts : i_ts + SCS, h0v + h, :], k_h[h, :, :])
                T.copy(GK[bb, i_ts : i_ts + SCS, h0v + h, :], g2[h, :, :])
            T.tile.fill(beta_h, 0.0)  # deterministic pad columns (4..15)
            T.copy(Beta[bb, i_ts, h0v], beta_h[:, 0:BH_V])  # [SCS, BH_V] row-gap copy

            # 2. fp32 domain: block casts + base-2 conversion
            T.tile.cast(q_f, q_h, "CAST_NONE", BH_V * SCS * DK)
            T.tile.cast(k_f, k_h, "CAST_NONE", BH_V * SCS * DK)
            T.copy(beta_h, beta_f)  # UB<->UB cross-dtype cast copy (flat 256)
            T.tile.mul(g2, g2, LN2)  # g2 = gk * ln2 (in-place)

            # Staging zero-fill: uncovered Aqk columns / Akk diagonal stay 0.
            T.tile.fill(aqk_s, 0.0)
            T.tile.fill(akk_s, 0.0)

            # 3. Row-batched pairwise computation (parse-time i unrolling via
            #    macro call sites; `if i < SCS` on the captured Python int is
            #    a parse-time bool, so no TIR-level branch is generated).
            w = i_s * SCS  # Aqk window base column (runtime offset)
            for h in T.serial(BH_V):
                _aqk_diag(h, w, q_f, k_f, pq, aqk_s)
                if 1 < SCS:
                    _aqk_akk_rows(1, h, w, BH_V, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f, aqk_s, akk_s)
                if 2 < SCS:
                    _aqk_akk_rows(2, h, w, BH_V, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f, aqk_s, akk_s)
                if 3 < SCS:
                    _aqk_akk_rows(3, h, w, BH_V, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f, aqk_s, akk_s)
                if 4 < SCS:
                    _aqk_akk_rows(4, h, w, BH_V, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f, aqk_s, akk_s)
                if 5 < SCS:
                    _aqk_akk_rows(5, h, w, BH_V, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f, aqk_s, akk_s)
                if 6 < SCS:
                    _aqk_akk_rows(6, h, w, BH_V, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f, aqk_s, akk_s)
                if 7 < SCS:
                    _aqk_akk_rows(7, h, w, BH_V, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f, aqk_s, akk_s)
                if 8 < SCS:
                    _aqk_akk_rows(8, h, w, BH_V, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f, aqk_s, akk_s)
                if 9 < SCS:
                    _aqk_akk_rows(9, h, w, BH_V, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f, aqk_s, akk_s)
                if 10 < SCS:
                    _aqk_akk_rows(10, h, w, BH_V, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f, aqk_s, akk_s)
                if 11 < SCS:
                    _aqk_akk_rows(11, h, w, BH_V, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f, aqk_s, akk_s)
                if 12 < SCS:
                    _aqk_akk_rows(12, h, w, BH_V, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f, aqk_s, akk_s)
                if 13 < SCS:
                    _aqk_akk_rows(13, h, w, BH_V, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f, aqk_s, akk_s)
                if 14 < SCS:
                    _aqk_akk_rows(14, h, w, BH_V, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f, aqk_s, akk_s)
                if 15 < SCS:
                    _aqk_akk_rows(15, h, w, BH_V, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f, aqk_s, akk_s)

            # 4. Writeback: block cast to bf16, then one [SCS, CS] /
            #    [SCS, SCS] 2D strided burst per head (head point index).
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
