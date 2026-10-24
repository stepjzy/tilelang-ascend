"""Iter-4 probe: split-merge reduce_2d differential (settle model vs emitter).

The iter3 next_hint flagged reduce_2d (30% of 6917us) as the top lever. The
reduce2d_v2 constexpr planner (offline dump: reduce_plan2_iter4.cxx) prices the
candidate restructures:

  [M, 128] current call          : cost4 = 64 + ~13*M  (3-step plan; step0 is
                                   M per-row vcgadd -- the repeat-stride cannot
                                   walk both the 128-wide intra-row interleave
                                   and the row stride)
  [M, 64]@PR=128 column slice    : cost4 = 64 + ~5.5*M  (step0 becomes ONE
                                   batched multi-row vcgadd scan)

Splitting every window reduce into two column-slice calls (clear=True on the
first, clear=False merge on the second -- no combine op needed) halves the
per-row charge but doubles the per-call fixed cost. The model says the fixed
wins: per-head aqk 3744 vs 2939 (+27%). The emitter instruction count says the
opposite: ~95 vs ~190 instructions (-50%). Which effect dominates is a
micro-architectural question -- settle it by measurement before touching the
kernel.

Variants (identical structure otherwise -- loads, factorize, broadcast/mul
chain, writeback, pass_configs):
  V_cur  : the adopted iter3 kernel (imported)
  V_split: every per-window reduce [M, 128] -> pq[0:M, 0:64] (clear=True) +
           pq[0:M, 64:128] (clear=False). M=1 calls (aqk diag, akk i=1) keep
           the M1 path (model-equal, fewer calls).

Differential correctness: V_split vs V_cur on the bench inputs (the split only
reorders the fp32 summation: (d0..63)+(d64..127) vs the helper's even/odd
block tree -- sub-ULP for bf16 outputs).

This probe measures TIME + relative correctness only; the real precision gate
(test suite --level all) runs only if the change is ported into the kernel.
"""
import json
import os
import sys

import tilelang
import torch
import torch_npu  # noqa: F401  (registers .npu())
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
def _aqk_diag(h, w, q_f, k_f, pq, aqk_s):
    """i == 0 fast path (unchanged: M=1 keeps the M1 leaf)."""
    T.tile.mul(pq[0:1, :], q_f[h, 0:1, :], k_f[h, 0:1, :])
    T.reduce_sum(pq[0:1, :], aqk_s[h, 0, w : w + 1], dim=-1, clear=True)


@T.macro
def _factorize(h, scs, anc, q_f, k_f, g2, e, kgj, bc_row):
    """One-shot factored decay setup (identical to the adopted kernel)."""
    T.tile.broadcast(bc_row[0:scs, :], g2[h, anc : anc + 1, :], axis=0)
    T.tile.sub(e[0:scs, :], bc_row[0:scs, :], g2[h, 0:scs, :])
    T.tile.exp(e[0:scs, :], e[0:scs, :])
    T.tile.mul(kgj[0:scs, :], k_f[h, 0:scs, :], e[0:scs, :])  # K' = k * E-
    T.tile.sub(e[0:scs, :], g2[h, 0:scs, :], bc_row[0:scs, :])
    T.tile.exp(e[0:scs, :], e[0:scs, :])
    T.tile.mul(q_f[h, 0:scs, :], q_f[h, 0:scs, :], e[0:scs, :])  # Q' = q * E+
    T.tile.mul(e[0:scs, :], k_f[h, 0:scs, :], e[0:scs, :])  # K'' = k * E+


@T.macro
def _rows_split(i, h, w, qp, kp, kpp, pq, bc_row, beta_f, aqk_s, akk_s):
    """Factored row pair with the SPLIT-MERGE reduce (the probe variable).

    Aqk row [0, i]: pq[0:i+1] = Q'_i (.) K'_[0:i+1]; reduce in two 64-wide
    column slices, the second merging (clear=False) into the first's outputs.
    Akk row [0, i): pq[0:i] = beta_i * K''_i (.) K'_[0:i]; same split."""
    # Aqk row
    T.tile.broadcast(bc_row[0 : i + 1, :], qp[h, i : i + 1, :], axis=0)
    T.tile.mul(pq[0 : i + 1, :], bc_row[0 : i + 1, :], kp[0 : i + 1, :])
    T.reduce_sum(pq[0 : i + 1, 0:64], aqk_s[h, i, w : w + i + 1], dim=-1, clear=True)
    T.reduce_sum(pq[0 : i + 1, 64:128], aqk_s[h, i, w : w + i + 1], dim=-1, clear=False)
    # Akk row
    T.tile.broadcast(bc_row[0:i, :], kpp[i : i + 1, :], axis=0)
    T.tile.mul(pq[0:i, :], bc_row[0:i, :], kp[0:i, :])
    T.tile.mul(pq[0:i, :], pq[0:i, :], beta_f[i * 16 + h, 0])
    if i > 1:
        T.reduce_sum(pq[0:i, 0:64], akk_s[h, i, 0:i], dim=-1, clear=True)
        T.reduce_sum(pq[0:i, 64:128], akk_s[h, i, 0:i], dim=-1, clear=False)
    else:
        # i == 1: M=1 keeps the M1 leaf (splitting it is a pure loss).
        T.reduce_sum(pq[0:1, :], akk_s[h, i, 0:1], dim=-1, clear=True)


@tilelang.jit(out_idx=[-2, -1], pass_configs=PASS_CONFIGS)
def probe_split_kernel(
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
    """V_split: adopted iter3 structure with split-merge window reduces."""
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

            w = i_s * SCS
            for h in T.serial(BH_V):
                _aqk_diag(h, w, q_f, k_f, pq, aqk_s)
                if 1 < SCS:
                    _factorize(h, SCS, SCS // 2, q_f, k_f, g2, e, kgj, bc_row)
                    _rows_split(1, h, w, q_f, kgj, e, pq, bc_row, beta_f, aqk_s, akk_s)
                if 2 < SCS:
                    _rows_split(2, h, w, q_f, kgj, e, pq, bc_row, beta_f, aqk_s, akk_s)
                if 3 < SCS:
                    _rows_split(3, h, w, q_f, kgj, e, pq, bc_row, beta_f, aqk_s, akk_s)
                if 4 < SCS:
                    _rows_split(4, h, w, q_f, kgj, e, pq, bc_row, beta_f, aqk_s, akk_s)
                if 5 < SCS:
                    _rows_split(5, h, w, q_f, kgj, e, pq, bc_row, beta_f, aqk_s, akk_s)
                if 6 < SCS:
                    _rows_split(6, h, w, q_f, kgj, e, pq, bc_row, beta_f, aqk_s, akk_s)
                if 7 < SCS:
                    _rows_split(7, h, w, q_f, kgj, e, pq, bc_row, beta_f, aqk_s, akk_s)
                if 8 < SCS:
                    _rows_split(8, h, w, q_f, kgj, e, pq, bc_row, beta_f, aqk_s, akk_s)
                if 9 < SCS:
                    _rows_split(9, h, w, q_f, kgj, e, pq, bc_row, beta_f, aqk_s, akk_s)
                if 10 < SCS:
                    _rows_split(10, h, w, q_f, kgj, e, pq, bc_row, beta_f, aqk_s, akk_s)
                if 11 < SCS:
                    _rows_split(11, h, w, q_f, kgj, e, pq, bc_row, beta_f, aqk_s, akk_s)
                if 12 < SCS:
                    _rows_split(12, h, w, q_f, kgj, e, pq, bc_row, beta_f, aqk_s, akk_s)
                if 13 < SCS:
                    _rows_split(13, h, w, q_f, kgj, e, pq, bc_row, beta_f, aqk_s, akk_s)
                if 14 < SCS:
                    _rows_split(14, h, w, q_f, kgj, e, pq, bc_row, beta_f, aqk_s, akk_s)
                if 15 < SCS:
                    _rows_split(15, h, w, q_f, kgj, e, pq, bc_row, beta_f, aqk_s, akk_s)

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

    split_fn = probe_split_kernel(B, S, H, DK, CS, SCS, BH)

    # Differential correctness first (never bench a numerically broken variant).
    aqk_cur, akk_cur = fla_chunk_intra_token_parallel(*npu_in)
    aqk_sp, akk_sp = split_fn(*npu_in)
    torch.npu.synchronize()
    for name, a, b in (("Aqk", aqk_cur, aqk_sp), ("Akk", akk_cur, akk_sp)):
        diff = (a.float() - b.float()).abs()
        rel = diff / b.float().abs().clamp_min(1e-6)
        print(f"[correctness] {name}: max_abs={diff.max().item():.3e} "
              f"max_rel={rel.max().item():.3e} "
              f"mismatch(1e-2)={(rel > 1e-2).float().mean().item():.2e}")
    ok = bool(
        torch.allclose(aqk_sp.float(), aqk_cur.float(), rtol=1e-2, atol=1e-2)
        and torch.allclose(akk_sp.float(), akk_cur.float(), rtol=1e-2, atol=1e-2)
    )
    print(f"[correctness] split vs adopted allclose(1e-2): {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        print("probe abort: split variant numerically broken")
        return

    results = {}
    for name, fn in (("V_cur", fla_chunk_intra_token_parallel), ("V_split", split_fn)):
        fn(*npu_in)
        torch.npu.synchronize()
        mean_us, times = bench_events(fn, npu_in)
        results[name] = {"mean_us": mean_us, "per_rep_us": times}
        print(f"[{name}] events mean = {mean_us:.1f} us  "
              f"(per-rep: {[f'{t:.0f}' for t in times]})", flush=True)

    cur = results["V_cur"]["mean_us"]
    sp = results["V_split"]["mean_us"]
    print("\n=== split-merge reduce differential (bench shape 1,8192,64,128) ===")
    print(f"  V_cur   (adopted iter3)      : {cur:8.1f} us")
    print(f"  V_split (64+64 clear/merge)  : {sp:8.1f} us")
    print(f"  delta                        : {(sp - cur) / cur * 100:+.1f}%")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fla_chunk_intra_token_parallel_splitreduce_iter4.json")
    with open(out, "w") as f:
        json.dump({"shape": [1, 8192, 64, 128], "variants": results,
                   "delta_pct": (sp - cur) / cur * 100}, f, indent=2)
    print(f"  json -> {out}")


if __name__ == "__main__":
    main()
