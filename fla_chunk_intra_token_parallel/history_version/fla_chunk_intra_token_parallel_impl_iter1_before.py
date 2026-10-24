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
deviations from the upstream CUDA code are design decisions):

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
5. Outputs go through an fp32 staging buffer, a block CAST_RINT to bf16
   and per-token flat T.copy bursts to GM (a single copy over a
   3-active-dim GM region drops the outer dim in the 2D DMA model, so the
   writeback is per token; each burst is one contiguous BH_V*CS run).
6. Element math uses buffer-level T.tile.* ops + T.reduce_sum(dim=-1,
   clear=True) only; scatter into staging uses plain UB scalar stores
   (T.Parallel scalar indexing is a known-buggy pattern).
7. Static i/j loops + runtime guards (if j <= i / if j < i / if i > 0):
   guarded bodies are pure vector ops (no DMA under a runtime branch).

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
            # load staging, bf16 (lifetime ends at the block cast)
            q_h = T.alloc_shared([SCS * BH_V, DK], input_dtype)
            k_h = T.alloc_shared([SCS * BH_V, DK], input_dtype)
            # fp32 compute domain; row t * BH_V + h == token i_ts + t, head h0v + h
            q_f = T.alloc_shared([SCS * BH_V, DK], gate_dtype)
            k_f = T.alloc_shared([SCS * BH_V, DK], gate_dtype)
            g2 = T.alloc_shared([SCS * BH_V, DK], gate_dtype)  # gk * ln2
            # beta staging: narrow [SCS, BH_V] rows (8B) must land in 32B UB
            # slots (MTE-filled narrow rows advance 32B per burst), so the
            # staging rows are padded to 16 bf16 (= 32B) and the DMA target is
            # the [*, 0:BH_V] column region of that padded buffer.
            beta_h = T.alloc_shared([SCS, 16], input_dtype)
            beta_f = T.alloc_shared([SCS, 16], gate_dtype)
            beta_col = T.alloc_shared([BH_V, 1], gate_dtype)
            beta_bc = T.alloc_shared([BH_V, DK], gate_dtype)
            # pairwise temporaries
            e = T.alloc_shared([BH_V, DK], gate_dtype)
            kgj = T.alloc_shared([BH_V, DK], gate_dtype)
            pq = T.alloc_shared([BH_V, DK], gate_dtype)
            pk = T.alloc_shared([BH_V, DK], gate_dtype)
            kb = T.alloc_shared([BH_V, DK], gate_dtype)
            sum_aqk = T.alloc_shared([BH_V], gate_dtype)
            sum_akk = T.alloc_shared([BH_V], gate_dtype)
            # fp32 staging (token, head, col); flat layout matches the GM
            # [b, token, h0v:h0v+BH_V, :] run element-for-element
            aqk_s = T.alloc_shared([SCS, BH_V, CS], gate_dtype)
            akk_s = T.alloc_shared([SCS, BH_V, SCS], gate_dtype)
            aqk_h = T.alloc_shared([SCS, BH_V, CS], output_dtype)
            akk_h = T.alloc_shared([SCS, BH_V, SCS], output_dtype)

            # 1. Loads: one [BH_V, DK] contiguous row-group per token
            #    (heads of a token are adjacent in the bshd layout).
            for t in T.serial(SCS):
                T.copy(Q[bb, i_ts + t, h0v, 0], q_h[t * BH_V : (t + 1) * BH_V, :])
                T.copy(K[bb, i_ts + t, h0v, 0], k_h[t * BH_V : (t + 1) * BH_V, :])
                T.copy(GK[bb, i_ts + t, h0v, 0], g2[t * BH_V : (t + 1) * BH_V, :])
            T.tile.fill(beta_h, 0.0)  # deterministic pad columns (4..15)
            T.copy(Beta[bb, i_ts, h0v], beta_h[:, 0:BH_V])  # [SCS, BH_V] row-gap copy

            # 2. fp32 domain: block casts + base-2 conversion
            T.tile.cast(q_f, q_h, "CAST_NONE", SCS * BH_V * DK)
            T.tile.cast(k_f, k_h, "CAST_NONE", SCS * BH_V * DK)
            T.copy(beta_h, beta_f)  # UB<->UB cross-dtype cast copy (flat 256)
            T.tile.mul(g2, g2, LN2)  # g2 = gk * ln2 (in-place)

            # Staging zero-fill: uncovered Aqk columns / Akk diagonal stay 0.
            T.tile.fill(aqk_s, 0.0)
            T.tile.fill(akk_s, 0.0)

            # 3. Pairwise computation: static bounds + guards. Everything
            #    under a guard is a pure vector op (no DMA in branches).
            for i in T.serial(SCS):
                if i > 0:
                    # kb = beta_i * k_i (row broadcast chain)
                    for h in range(BH_V):
                        beta_col[h, 0] = beta_f[i, h]
                    T.tile.broadcast(beta_bc, beta_col)
                    T.tile.mul(kb, k_f[i * BH_V : (i + 1) * BH_V, :], beta_bc)
                for j in T.serial(SCS):
                    if j <= i:
                        # e = 2^(g_i - g_j) = exp((g_i - g_j) * ln2)
                        T.tile.sub(
                            e,
                            g2[i * BH_V : (i + 1) * BH_V, :],
                            g2[j * BH_V : (j + 1) * BH_V, :],
                        )
                        T.tile.exp(e, e)
                        # kgj = k_j * e
                        T.tile.mul(kgj, k_f[j * BH_V : (j + 1) * BH_V, :], e)
                        # Aqk value = rowsum(q_i * kgj)   (scale == 1.0)
                        T.tile.mul(pq, q_f[i * BH_V : (i + 1) * BH_V, :], kgj)
                        T.reduce_sum(pq, sum_aqk, dim=-1, clear=True)
                        for h in range(BH_V):
                            aqk_s[i, h, i_s * SCS + j] = sum_aqk[h]
                        if j < i:  # Akk diagonal (j == i) stays 0
                            # Akk value = rowsum((beta_i * k_i) * kgj)
                            T.tile.mul(pk, kb, kgj)
                            T.reduce_sum(pk, sum_akk, dim=-1, clear=True)
                            for h in range(BH_V):
                                akk_s[i, h, j] = sum_akk[h]

            # 4. Writeback: block cast to bf16, then one flat burst per
            #    token ([BH_V, CS] / [BH_V, SCS] contiguous GM runs).
            T.tile.cast(aqk_h, aqk_s, "CAST_RINT", SCS * BH_V * CS)
            T.tile.cast(akk_h, akk_s, "CAST_RINT", SCS * BH_V * SCS)
            for t in T.serial(SCS):
                T.copy(aqk_h[t, :, :], Aqk[bb, i_ts + t, h0v, 0])
                T.copy(akk_h[t, :, :], Akk[bb, i_ts + t, h0v, 0])

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
