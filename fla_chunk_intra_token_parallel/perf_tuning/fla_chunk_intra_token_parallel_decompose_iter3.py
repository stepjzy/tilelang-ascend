"""Iter-3 decomposition probe: split the 9071us baseline into measured parts.

Differential variants over the SAME kernel structure (same blocks, buffers,
DMA pattern, pass_configs) as the adopted iter1 kernel:

  P_dma  : loads + casts/fills + writeback only  (no hot-loop compute)
  P_vec  : + full hot-loop vector chain (broadcast/sub/exp/mul/muls), NO reduce
  P_full : the actual adopted kernel (imported)  = P_vec + 124 reduce_2d/block

Deltas:
  P_dma              -> DMA + writeback + setup + scheduling floor
  P_vec - P_dma      -> hot-loop vector instruction stream (non-reduce)
  P_full - P_vec     -> reduce_2d contribution (124 calls/block)

Outputs are wrong by construction for P_dma/P_vec (staging never receives the
pairwise values); this probe measures TIME only, never precision.

Structure note: two module-level @tilelang.jit kernels (no closure factory /
no bool-kernel-arg `if` -- the TVM script parser silently returns a None
PrimFunc for those shapes; iter2 probes used the same module-level pattern).
"""
import json
import os
import sys

import tilelang
import torch
import torch_npu  # noqa: F401
from tilelang import language as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fla_chunk_intra_token_parallel import fla_chunk_intra_token_parallel  # noqa: E402

LN2 = 0.6931471805599453

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@T.macro
def _vec_diag(h, q_f, k_f, pq):
    """i == 0 chain without the reduce (mul only)."""
    T.tile.mul(pq[0:1, :], q_f[h, 0:1, :], k_f[h, 0:1, :])


@T.macro
def _vec_rows(i, h, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f):
    """Row-batched pairwise chain for token i without the two reduce calls."""
    # Aqk row: e = exp(g_i - g_j) for j in [0, i]
    T.tile.broadcast(bc_row[0 : i + 1, :], g2[h, i : i + 1, :], axis=0)
    T.tile.sub(e[0 : i + 1, :], bc_row[0 : i + 1, :], g2[h, 0 : i + 1, :])
    T.tile.exp(e[0 : i + 1, :], e[0 : i + 1, :])
    T.tile.mul(kgj[0 : i + 1, :], k_f[h, 0 : i + 1, :], e[0 : i + 1, :])
    T.tile.broadcast(bc_row[0 : i + 1, :], q_f[h, i : i + 1, :], axis=0)
    T.tile.mul(pq[0 : i + 1, :], bc_row[0 : i + 1, :], kgj[0 : i + 1, :])
    # Akk row: j in [0, i)
    T.tile.broadcast(bc_row[0:i, :], k_f[h, i : i + 1, :], axis=0)
    T.tile.mul(pq[0:i, :], bc_row[0:i, :], kgj[0:i, :])
    T.tile.mul(pq[0:i, :], pq[0:i, :], beta_f[i * 16 + h, 0])


@tilelang.jit(out_idx=[-2, -1], pass_configs=PASS_CONFIGS)
def probe_vec_kernel(
    B, S, H, DK, CS=64, SCS=16, BH=8,
    input_dtype="bfloat16", gate_dtype="float32", output_dtype="bfloat16",
):
    """P_vec: full adopted-kernel structure minus the reduce_sum calls."""
    VEC_NUM = 2
    BH_V = BH // VEC_NUM
    NSC = S // SCS
    HG = H // BH
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
            sc = cid % NSC
            hg = (cid // NSC) % HG
            bb = cid // (NSC * HG)
            c = sc // (CS // SCS)
            i_s = sc % (CS // SCS)
            i_ts = c * CS + i_s * SCS
            h0v = hg * BH + vid * BH_V

            q_h = T.alloc_shared([BH_V, SCS, DK], input_dtype)
            k_h = T.alloc_shared([BH_V, SCS, DK], input_dtype)
            q_f = T.alloc_shared([BH_V, SCS, DK], gate_dtype)
            k_f = T.alloc_shared([BH_V, SCS, DK], gate_dtype)
            g2 = T.alloc_shared([BH_V, SCS, DK], gate_dtype)
            beta_h = T.alloc_shared([SCS, 16], input_dtype)
            beta_f = T.alloc_shared([SCS, 16], gate_dtype)
            e = T.alloc_shared([SCS, DK], gate_dtype)
            kgj = T.alloc_shared([SCS, DK], gate_dtype)
            pq = T.alloc_shared([SCS, DK], gate_dtype)
            bc_row = T.alloc_shared([SCS, DK], gate_dtype)
            aqk_s = T.alloc_shared([BH_V, SCS, CS], gate_dtype)
            akk_s = T.alloc_shared([BH_V, SCS, AKK_W], gate_dtype)
            aqk_h = T.alloc_shared([BH_V, SCS, CS], output_dtype)
            akk_h = T.alloc_shared([BH_V, SCS, AKK_W], output_dtype)

            for h in T.serial(BH_V):
                T.copy(Q[bb, i_ts : i_ts + SCS, h0v + h, :], q_h[h, :, :])
                T.copy(K[bb, i_ts : i_ts + SCS, h0v + h, :], k_h[h, :, :])
                T.copy(GK[bb, i_ts : i_ts + SCS, h0v + h, :], g2[h, :, :])
            T.tile.fill(beta_h, 0.0)
            T.copy(Beta[bb, i_ts, h0v], beta_h[:, 0:BH_V])

            T.tile.cast(q_f, q_h, "CAST_NONE", BH_V * SCS * DK)
            T.tile.cast(k_f, k_h, "CAST_NONE", BH_V * SCS * DK)
            T.copy(beta_h, beta_f)
            T.tile.mul(g2, g2, LN2)

            T.tile.fill(aqk_s, 0.0)
            T.tile.fill(akk_s, 0.0)

            for h in T.serial(BH_V):
                _vec_diag(h, q_f, k_f, pq)
                if 1 < SCS:
                    _vec_rows(1, h, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f)
                if 2 < SCS:
                    _vec_rows(2, h, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f)
                if 3 < SCS:
                    _vec_rows(3, h, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f)
                if 4 < SCS:
                    _vec_rows(4, h, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f)
                if 5 < SCS:
                    _vec_rows(5, h, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f)
                if 6 < SCS:
                    _vec_rows(6, h, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f)
                if 7 < SCS:
                    _vec_rows(7, h, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f)
                if 8 < SCS:
                    _vec_rows(8, h, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f)
                if 9 < SCS:
                    _vec_rows(9, h, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f)
                if 10 < SCS:
                    _vec_rows(10, h, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f)
                if 11 < SCS:
                    _vec_rows(11, h, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f)
                if 12 < SCS:
                    _vec_rows(12, h, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f)
                if 13 < SCS:
                    _vec_rows(13, h, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f)
                if 14 < SCS:
                    _vec_rows(14, h, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f)
                if 15 < SCS:
                    _vec_rows(15, h, q_f, k_f, g2, e, kgj, pq, bc_row, beta_f)

            T.tile.cast(aqk_h, aqk_s, "CAST_RINT", SCS * BH_V * CS)
            T.tile.cast(akk_h, akk_s, "CAST_RINT", BH_V * SCS * AKK_W)
            for h in T.serial(BH_V):
                T.copy(aqk_h[h, :, :], Aqk[bb, i_ts : i_ts + SCS, h0v + h, :])
                T.copy(akk_h[h, :, 0:SCS], Akk[bb, i_ts : i_ts + SCS, h0v + h, :])

    return main


@tilelang.jit(out_idx=[-2, -1], pass_configs=PASS_CONFIGS)
def probe_dma_kernel(
    B, S, H, DK, CS=64, SCS=16, BH=8,
    input_dtype="bfloat16", gate_dtype="float32", output_dtype="bfloat16",
):
    """P_dma: adopted-kernel structure with the whole hot loop removed."""
    VEC_NUM = 2
    BH_V = BH // VEC_NUM
    NSC = S // SCS
    HG = H // BH
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
            sc = cid % NSC
            hg = (cid // NSC) % HG
            bb = cid // (NSC * HG)
            c = sc // (CS // SCS)
            i_s = sc % (CS // SCS)
            i_ts = c * CS + i_s * SCS
            h0v = hg * BH + vid * BH_V

            q_h = T.alloc_shared([BH_V, SCS, DK], input_dtype)
            k_h = T.alloc_shared([BH_V, SCS, DK], input_dtype)
            q_f = T.alloc_shared([BH_V, SCS, DK], gate_dtype)
            k_f = T.alloc_shared([BH_V, SCS, DK], gate_dtype)
            g2 = T.alloc_shared([BH_V, SCS, DK], gate_dtype)
            beta_h = T.alloc_shared([SCS, 16], input_dtype)
            beta_f = T.alloc_shared([SCS, 16], gate_dtype)
            e = T.alloc_shared([SCS, DK], gate_dtype)
            kgj = T.alloc_shared([SCS, DK], gate_dtype)
            pq = T.alloc_shared([SCS, DK], gate_dtype)
            bc_row = T.alloc_shared([SCS, DK], gate_dtype)
            aqk_s = T.alloc_shared([BH_V, SCS, CS], gate_dtype)
            akk_s = T.alloc_shared([BH_V, SCS, AKK_W], gate_dtype)
            aqk_h = T.alloc_shared([BH_V, SCS, CS], output_dtype)
            akk_h = T.alloc_shared([BH_V, SCS, AKK_W], output_dtype)

            for h in T.serial(BH_V):
                T.copy(Q[bb, i_ts : i_ts + SCS, h0v + h, :], q_h[h, :, :])
                T.copy(K[bb, i_ts : i_ts + SCS, h0v + h, :], k_h[h, :, :])
                T.copy(GK[bb, i_ts : i_ts + SCS, h0v + h, :], g2[h, :, :])
            T.tile.fill(beta_h, 0.0)
            T.copy(Beta[bb, i_ts, h0v], beta_h[:, 0:BH_V])

            T.tile.cast(q_f, q_h, "CAST_NONE", BH_V * SCS * DK)
            T.tile.cast(k_f, k_h, "CAST_NONE", BH_V * SCS * DK)
            T.copy(beta_h, beta_f)
            T.tile.mul(g2, g2, LN2)

            T.tile.fill(aqk_s, 0.0)
            T.tile.fill(akk_s, 0.0)

            T.tile.cast(aqk_h, aqk_s, "CAST_RINT", SCS * BH_V * CS)
            T.tile.cast(akk_h, akk_s, "CAST_RINT", BH_V * SCS * AKK_W)
            for h in T.serial(BH_V):
                T.copy(aqk_h[h, :, :], Aqk[bb, i_ts : i_ts + SCS, h0v + h, :])
                T.copy(akk_h[h, :, 0:SCS], Akk[bb, i_ts : i_ts + SCS, h0v + h, :])

    return main


def make_bench_inputs(seed=42):
    B, S, H, DK = 1, 8192, 64, 128
    torch.manual_seed(seed)
    q = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
    k = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
    beta = torch.randn(B, S, H, dtype=torch.bfloat16)
    gk = torch.nn.functional.logsigmoid(torch.randn(B, S, H, DK, dtype=torch.float32))
    gk = gk.view(B, S // 64, 64, H, DK).cumsum(dim=2).view(B, S, H, DK)
    return q.npu(), k.npu(), gk.npu(), beta.npu()


def bench_events(fn, args, warmup=20, rep=10):
    for _ in range(warmup):
        fn(*args)
    torch.npu.synchronize()
    start = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    end = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start[i].record()
        fn(*args)
        end[i].record()
    torch.npu.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start, end)]
    return sum(times) / len(times) * 1000.0, times  # us


def main():
    B, S, H, DK, CS, SCS, BH = 1, 8192, 64, 128, 64, 16, 8
    npu_in = make_bench_inputs()

    results = {}
    variants = [
        ("P_dma", probe_dma_kernel(B, S, H, DK, CS, SCS, BH)),
        ("P_vec", probe_vec_kernel(B, S, H, DK, CS, SCS, BH)),
        ("P_full", fla_chunk_intra_token_parallel),
    ]
    for name, fn in variants:
        fn(*npu_in)  # compile + first call
        torch.npu.synchronize()
        mean_us, times = bench_events(fn, npu_in)
        results[name] = {"mean_us": mean_us, "per_rep_us": times}
        print(f"[{name}] events mean = {mean_us:.1f} us  "
              f"(per-rep: {[f'{t:.0f}' for t in times]})", flush=True)

    dma = results["P_dma"]["mean_us"]
    vec = results["P_vec"]["mean_us"]
    full = results["P_full"]["mean_us"]
    print("\n=== decomposition (bench shape 1,8192,64,128) ===")
    print(f"  P_dma (DMA+writeback+setup+schedule floor): {dma:8.1f} us  ({dma/full*100:5.1f}%)")
    print(f"  vector chain (P_vec - P_dma):               {vec-dma:8.1f} us  ({(vec-dma)/full*100:5.1f}%)")
    print(f"  reduce_2d (P_full - P_vec):                 {full-vec:8.1f} us  ({(full-vec)/full*100:5.1f}%)")
    print(f"  P_full (baseline):                          {full:8.1f} us")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fla_chunk_intra_token_parallel_decomposition_iter3.json")
    with open(out, "w") as f:
        json.dump({"shape": [1, 8192, 64, 128], "variants": results,
                   "decomposition": {"dma_setup_us": dma, "vector_chain_us": vec - dma,
                                     "reduce_us": full - vec}}, f, indent=2)
    print(f"  json -> {out}")


if __name__ == "__main__":
    main()
