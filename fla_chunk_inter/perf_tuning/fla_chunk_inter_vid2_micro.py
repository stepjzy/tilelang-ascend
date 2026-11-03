"""R3 decisive experiment 1: threads=2 (the only 40-AIV entry) vs the
fla_chunk_inter drain chain, on a micro shape, with pure manual tile
primitives (the EXACT op shapes of the real kernel, no math restructuring).

Source-level analysis (ascend_vid_reduction.cc, this session) predicts:
- The pass halves EVERY shared.ub buffer's dim-0 and offsets GM accesses at
  (gm_index_count - ub_dim_count) - the trailing-dims-aligned axis.
- [BS, bDK] 2-D buffers split along BS; 1-D [bDK] buffers split along bDK.
  The ONLY bridge (T.tile.broadcast) gets post-halving shapes that violate
  its own contract (src[bDK/2] vs dst[BS/2, bDK]) -> corrupted.
- dim=0 reduce (sum over the split rows) yields per-vid partials with NO
  cross-vid combine.
- The CONSISTENT subset (pure 2-D ops + [BS] 1-D broadcast axis=1 + dim=1
  reduces on [bDK, DV] DK-split operands) should stay correct.

Sub-tests (variant_code, each run at threads=1 and threads=2):
  0 K1_pure2d      2-D ops + ml [BS] broadcast axis=1              (consistent control)
  1 K2_gn_bcast    + gn [bDK] 1-D chain + broadcast(gn2d, axis=0)  (contested #1)
  2 K3_dim0_reduce + pp dim=0 reduce + broadcast(pp2d, axis=0)     (contested #2)
  3 K4_full_chain  the exact real-kernel drain structure            (both + dgkn)
  K5_dim1_reduce (separate kernel): produce-side [bDK, DV] DK-split, dim=1 reduce.

GM accesses use 3D [B, S, H*DK] row views (family-verified threads=2-safe;
4D bshd slices put the vid offset on the H axis = silent head swap).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import torch  # noqa: E402
import tilelang  # noqa: E402
from tilelang import language as T  # noqa: E402

tilelang.disable_cache()

LN2 = 0.6931471805599453
BS = 64  # chunk rows
BDK = 64  # bench block_DK (DK % 64 == 0 and DV >= 128)
DV = 128  # K5's [bDK, DV] operand (bench-typical)
S = 256  # 4 chunks
B = 1
H = 1
GRID = S // BS  # 4 tasks, one block per task

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[6, 7, 8], pass_configs=PASS_CONFIGS)
def drain_micro(variant_code, nth):
    """variant_code: 0=pure2d 1=gn 2=reduce 3=full; nth: 1 or 2 threads."""

    @T.prim_func
    def main(
        G3: T.Tensor((B, S, H * BDK), "float32"),  # type: ignore
        Q3: T.Tensor((B, S, H * BDK), "float32"),  # type: ignore
        K3: T.Tensor((B, S, H * BDK), "float32"),  # type: ignore
        DQR3: T.Tensor((2, B, S, H * BDK), "float32"),  # type: ignore  # [0]=dq_raw [1]=dk_raw
        DGKN: T.Tensor((GRID, BDK), "float32"),  # type: ignore
        LastRow: T.Tensor((BS,), "float32"),  # type: ignore
        dq3: T.Tensor((B, S, H * BDK), "float32"),  # type: ignore
        dk3: T.Tensor((B, S, H * BDK), "float32"),  # type: ignore
        dg3: T.Tensor((B, S, H * BDK), "float32"),  # type: ignore
    ):
        if nth == 1:
            with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
                g_ub = T.alloc_shared((BS, BDK), "float32")
                gexp_ub = T.alloc_shared((BS, BDK), "float32")
                gn_ub = T.alloc_shared((BDK,), "float32")
                gnexp_ub = T.alloc_shared((BDK,), "float32")
                gn2d_ub = T.alloc_shared((BS, BDK), "float32")
                gnd_ub = T.alloc_shared((BS, BDK), "float32")
                q_ub = T.alloc_shared((BS, BDK), "float32")
                k_ub = T.alloc_shared((BS, BDK), "float32")
                dq_ub = T.alloc_shared((BS, BDK), "float32")
                dk_ub = T.alloc_shared((BS, BDK), "float32")
                dkk_ub = T.alloc_shared((BS, BDK), "float32")
                dg_ub = T.alloc_shared((BS, BDK), "float32")
                dgkn_ub = T.alloc_shared((BDK,), "float32")
                pp_ub = T.alloc_shared((BDK,), "float32")
                pp2d_ub = T.alloc_shared((BS, BDK), "float32")
                ml_ub = T.alloc_shared((BS,), "float32")
                ml2d_ub = T.alloc_shared((BS, BDK), "float32")
                mp2d_ub = T.alloc_shared((BS, BDK), "float32")

                s0 = cid * BS
                T.copy(DQR3[0, 0, s0 : s0 + BS, 0:BDK], dq_ub)
                T.copy(DQR3[0, 1, s0 : s0 + BS, 0:BDK], dk_ub)
                T.copy(G3[0, s0 : s0 + BS, 0:BDK], g_ub)
                T.copy(Q3[0, s0 : s0 + BS, 0:BDK], q_ub)
                T.copy(K3[0, s0 : s0 + BS, 0:BDK], k_ub)
                T.copy(LastRow[0:BS], ml_ub)
                T.tile.mul(g_ub, g_ub, LN2)
                T.tile.exp(gexp_ub, g_ub)
                T.tile.mul(dq_ub, dq_ub, gexp_ub)
                if variant_code in (1, 3):
                    T.copy(G3[0, s0 + BS - 1, 0:BDK], gn_ub)
                    T.tile.mul(gn_ub, gn_ub, LN2)
                    T.tile.exp(gnexp_ub, gn_ub)
                    T.tile.broadcast(gn2d_ub, gn_ub, axis=0)
                    T.tile.sub(gnd_ub, gn2d_ub, g_ub)
                    T.tile.exp(gnd_ub, gnd_ub)
                    T.tile.mul(dk_ub, dk_ub, gnd_ub)
                else:
                    T.tile.exp(gnd_ub, g_ub)
                    T.tile.mul(dk_ub, dk_ub, gnd_ub)
                T.tile.mul(dkk_ub, dk_ub, k_ub)
                if variant_code in (2, 3):
                    T.copy(DGKN[cid, 0:BDK], dgkn_ub)
                    T.reduce_sum(dkk_ub, pp_ub, dim=0, clear=True)
                    if variant_code == 3:
                        T.tile.mul(dgkn_ub, dgkn_ub, gnexp_ub)
                        T.tile.add(pp_ub, pp_ub, dgkn_ub)
                    T.tile.broadcast(pp2d_ub, pp_ub, axis=0)
                    T.tile.broadcast(ml2d_ub, ml_ub, axis=1)
                    T.tile.mul(mp2d_ub, pp2d_ub, ml2d_ub)
                else:
                    T.tile.broadcast(ml2d_ub, ml_ub, axis=1)
                    T.tile.mul(mp2d_ub, q_ub, ml2d_ub)
                T.tile.mul(dg_ub, q_ub, dq_ub)
                T.tile.sub(dg_ub, dg_ub, dkk_ub)
                T.tile.add(dg_ub, dg_ub, mp2d_ub)
                T.copy(dq_ub, dq3[0, s0 : s0 + BS, 0:BDK])
                T.copy(dk_ub, dk3[0, s0 : s0 + BS, 0:BDK])
                T.copy(dg_ub, dg3[0, s0 : s0 + BS, 0:BDK])
        else:
            with T.Kernel(GRID, threads=2, is_npu=True) as (cid):
                g_ub = T.alloc_shared((BS, BDK), "float32")
                gexp_ub = T.alloc_shared((BS, BDK), "float32")
                gn_ub = T.alloc_shared((BDK,), "float32")
                gnexp_ub = T.alloc_shared((BDK,), "float32")
                gn2d_ub = T.alloc_shared((BS, BDK), "float32")
                gnd_ub = T.alloc_shared((BS, BDK), "float32")
                q_ub = T.alloc_shared((BS, BDK), "float32")
                k_ub = T.alloc_shared((BS, BDK), "float32")
                dq_ub = T.alloc_shared((BS, BDK), "float32")
                dk_ub = T.alloc_shared((BS, BDK), "float32")
                dkk_ub = T.alloc_shared((BS, BDK), "float32")
                dg_ub = T.alloc_shared((BS, BDK), "float32")
                dgkn_ub = T.alloc_shared((BDK,), "float32")
                pp_ub = T.alloc_shared((BDK,), "float32")
                pp2d_ub = T.alloc_shared((BS, BDK), "float32")
                ml_ub = T.alloc_shared((BS,), "float32")
                ml2d_ub = T.alloc_shared((BS, BDK), "float32")
                mp2d_ub = T.alloc_shared((BS, BDK), "float32")

                s0 = cid * BS
                T.copy(DQR3[0, 0, s0 : s0 + BS, 0:BDK], dq_ub)
                T.copy(DQR3[0, 1, s0 : s0 + BS, 0:BDK], dk_ub)
                T.copy(G3[0, s0 : s0 + BS, 0:BDK], g_ub)
                T.copy(Q3[0, s0 : s0 + BS, 0:BDK], q_ub)
                T.copy(K3[0, s0 : s0 + BS, 0:BDK], k_ub)
                T.copy(LastRow[0:BS], ml_ub)
                T.tile.mul(g_ub, g_ub, LN2)
                T.tile.exp(gexp_ub, g_ub)
                T.tile.mul(dq_ub, dq_ub, gexp_ub)
                if variant_code in (1, 3):
                    T.copy(G3[0, s0 + BS - 1, 0:BDK], gn_ub)
                    T.tile.mul(gn_ub, gn_ub, LN2)
                    T.tile.exp(gnexp_ub, gn_ub)
                    T.tile.broadcast(gn2d_ub, gn_ub, axis=0)
                    T.tile.sub(gnd_ub, gn2d_ub, g_ub)
                    T.tile.exp(gnd_ub, gnd_ub)
                    T.tile.mul(dk_ub, dk_ub, gnd_ub)
                else:
                    T.tile.exp(gnd_ub, g_ub)
                    T.tile.mul(dk_ub, dk_ub, gnd_ub)
                T.tile.mul(dkk_ub, dk_ub, k_ub)
                if variant_code in (2, 3):
                    T.copy(DGKN[cid, 0:BDK], dgkn_ub)
                    T.reduce_sum(dkk_ub, pp_ub, dim=0, clear=True)
                    if variant_code == 3:
                        T.tile.mul(dgkn_ub, dgkn_ub, gnexp_ub)
                        T.tile.add(pp_ub, pp_ub, dgkn_ub)
                    T.tile.broadcast(pp2d_ub, pp_ub, axis=0)
                    T.tile.broadcast(ml2d_ub, ml_ub, axis=1)
                    T.tile.mul(mp2d_ub, pp2d_ub, ml2d_ub)
                else:
                    T.tile.broadcast(ml2d_ub, ml_ub, axis=1)
                    T.tile.mul(mp2d_ub, q_ub, ml2d_ub)
                T.tile.mul(dg_ub, q_ub, dq_ub)
                T.tile.sub(dg_ub, dg_ub, dkk_ub)
                T.tile.add(dg_ub, dg_ub, mp2d_ub)
                T.copy(dq_ub, dq3[0, s0 : s0 + BS, 0:BDK])
                T.copy(dk_ub, dk3[0, s0 : s0 + BS, 0:BDK])
                T.copy(dg_ub, dg3[0, s0 : s0 + BS, 0:BDK])

    return main


@tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS)
def dim1_micro(nth):
    """K5: produce-side pattern - [bDK, DV] DK-split operand, dim=1 reduce."""

    @T.prim_func
    def main(
        X: T.Tensor((GRID, BDK, DV), "float32"),  # type: ignore
        Y: T.Tensor((GRID, BDK, DV), "float32"),  # type: ignore
        OUT: T.Tensor((GRID, BDK), "float32"),  # type: ignore
    ):
        if nth == 1:
            with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
                x_ub = T.alloc_shared((BDK, DV), "float32")
                y_ub = T.alloc_shared((BDK, DV), "float32")
                xy_ub = T.alloc_shared((BDK, DV), "float32")
                out_ub = T.alloc_shared((BDK,), "float32")
                T.copy(X[cid, 0:BDK, 0:DV], x_ub)
                T.copy(Y[cid, 0:BDK, 0:DV], y_ub)
                T.tile.mul(xy_ub, x_ub, y_ub)
                T.reduce_sum(xy_ub, out_ub, dim=1, clear=True)
                T.copy(out_ub, OUT[cid, 0:BDK])
        else:
            with T.Kernel(GRID, threads=2, is_npu=True) as (cid):
                x_ub = T.alloc_shared((BDK, DV), "float32")
                y_ub = T.alloc_shared((BDK, DV), "float32")
                xy_ub = T.alloc_shared((BDK, DV), "float32")
                out_ub = T.alloc_shared((BDK,), "float32")
                T.copy(X[cid, 0:BDK, 0:DV], x_ub)
                T.copy(Y[cid, 0:BDK, 0:DV], y_ub)
                T.tile.mul(xy_ub, x_ub, y_ub)
                T.reduce_sum(xy_ub, out_ub, dim=1, clear=True)
                T.copy(out_ub, OUT[cid, 0:BDK])

    return main


def golden(g4, q4, k4, dqr4, dkr4, dgkn, lastrow, variant_code):
    """fp64 CPU mirror of the exact per-variant math ([NT, BS, BDK] chunks)."""
    NT = GRID
    g = g4.reshape(NT, BS, BDK).double()
    q = q4.reshape(NT, BS, BDK).double()
    k = k4.reshape(NT, BS, BDK).double()
    dqr = dqr4.reshape(NT, BS, BDK).double()
    dkr = dkr4.reshape(NT, BS, BDK).double()
    dgkn = dgkn.double()
    lr = lastrow.double()

    dq = dqr * torch.exp(g * LN2)
    if variant_code in (1, 3):
        gn = g[:, -1, :]  # last row per chunk [NT, BDK]
        dk = dkr * torch.exp((gn.unsqueeze(1) - g) * LN2)
    else:
        dk = dkr * torch.exp(g * LN2)
    dkk = dk * k
    dg = q * dq - dkk
    if variant_code in (2, 3):
        pp = dkk.sum(dim=1)  # [NT, BDK]
        if variant_code == 3:
            gn = g[:, -1, :]
            pp = pp + dgkn * torch.exp(gn * LN2)
        dg = dg + lr.view(1, BS, 1) * pp.unsqueeze(1)
    else:
        dg = dg + lr.view(1, BS, 1) * q  # stand-in matches the kernel's q*ml2d
    return dq.float(), dk.float(), dg.float()


def compare(out, ref):
    """Family gate: fp32 mixed tolerance (atol 1.53e-5, rtol 9.77e-4, ratio>=0.99)."""
    o = out.float().cpu().reshape(-1)
    r = ref.float().cpu().reshape(-1)
    diff = (o - r).abs()
    tol = 1.53e-5 + 9.77e-4 * r.abs()
    ratio = (diff <= tol).float().mean().item()
    max_abs = diff.max().item()
    return ratio >= 0.99 and max_abs <= 1e-2, ratio, max_abs


def main():
    torch.manual_seed(0)
    dev = "npu"

    g4 = torch.randn(B, S, H, BDK, dtype=torch.float32)
    q4 = torch.randn(B, S, H, BDK, dtype=torch.float32)
    k4 = torch.randn(B, S, H, BDK, dtype=torch.float32)
    dqr4 = torch.randn(B, S, H, BDK, dtype=torch.float32)
    dkr4 = torch.randn(B, S, H, BDK, dtype=torch.float32)
    g = g4.view(B, S, H * BDK)
    q = q4.view(B, S, H * BDK)
    k = k4.view(B, S, H * BDK)
    dqkr = torch.stack([dqr4.view(B, S, H * BDK), dkr4.view(B, S, H * BDK)])
    dgkn = torch.randn(GRID, BDK, dtype=torch.float32)
    lastrow = torch.zeros(BS, dtype=torch.float32)
    lastrow[-1] = 1.0

    dev_args = [t.to(dev) for t in (g, q, k, dqkr, dgkn, lastrow)]

    for label, code in (
        ("K1_pure2d", 0),
        ("K2_gn_bcast", 1),
        ("K3_dim0_reduce", 2),
        ("K4_full_chain", 3),
    ):
        refs = golden(g4, q4, k4, dqr4, dkr4, dgkn, lastrow, code)
        for nth in (1, 2):
            try:
                outs = drain_micro(code, nth)(*dev_args)
                oks = [compare(o, r) for o, r in zip(outs, refs)]
                allok = all(o[0] for o in oks)
                detail = " | ".join(f"ratio={r:.4f} max_abs={m:.3e}" for _, r, m in oks)
                print(f"[VID2-MICRO] {label} threads={nth}: {'PASS' if allok else 'FAIL'} ({detail})")
            except Exception as e:  # noqa: BLE001
                print(f"[VID2-MICRO] {label} threads={nth}: EXCEPTION {type(e).__name__}: {str(e)[:200]}")

    x = torch.randn(GRID, BDK, DV, dtype=torch.float32)
    y = torch.randn(GRID, BDK, DV, dtype=torch.float32)
    ref5 = (x.double() * y.double()).sum(dim=2).float()
    for nth in (1, 2):
        try:
            ret = dim1_micro(nth)(x.to(dev), y.to(dev))
            out = ret[0] if isinstance(ret, (list, tuple)) else ret
            ok, ratio, max_abs = compare(out, ref5)
            print(f"[VID2-MICRO] K5_dim1_reduce threads={nth}: {'PASS' if ok else 'FAIL'} (ratio={ratio:.4f} max_abs={max_abs:.3e})")
        except Exception as e:  # noqa: BLE001
            print(f"[VID2-MICRO] K5_dim1_reduce threads={nth}: EXCEPTION {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
