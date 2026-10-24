"""Diagnostic probes for the fla_chunk_o per-block critical path (timing only,
precision intentionally broken). Variants:
  --probe skeleton : all GM loads + V->C relays + C->V o chain, NO vector ops,
                     NO gemms  -> DMA + relay + sync floor
  --probe vec      : skeleton + full V-side gate/mask/hi-lo compute, NO gemms
  (full kernel = fla_chunk_o.py itself, the reference)
"""
import argparse
import os
import sys

import tilelang
import torch
import torch_npu  # noqa: F401
from tilelang import language as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LN2 = 0.6931471805599453

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[6], pass_configs=PASS_CONFIGS)
def probe_kernel(B, S, H, DK, DV, chunk_size=64, scale=None, vec=False):
    BS = chunk_size
    NT = S // BS
    bDV = DV
    GRID = NT * B * H

    @T.prim_func
    def main(
        Q: T.Tensor((B, S, H * DK), "bfloat16"),  # type: ignore
        V: T.Tensor((B, S, H, DV), "bfloat16"),  # type: ignore
        G: T.Tensor((B, S, H * DK), "float32"),  # type: ignore
        A: T.Tensor((B, S, H * BS), "bfloat16"),  # type: ignore
        Hstate: T.Tensor((B, NT, H, DK, DV), "bfloat16"),  # type: ignore
        Msk: T.Tensor((BS, BS), "float32"),  # type: ignore
        O: T.Tensor((B, S, H * DV), "bfloat16"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=2, is_npu=True) as (cid):
            bs = cid % NT
            bbh = cid // NT
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS

            A_l1 = T.alloc_shared([BS, BS], "bfloat16")
            V_l1 = T.alloc_shared([BS, bDV], "bfloat16")
            H_l1 = T.alloc_shared([DK, bDV], "bfloat16")
            qg_hi_l1 = T.alloc_shared([BS, DK], "bfloat16")
            qg_lo_l1 = T.alloc_shared([BS, DK], "bfloat16")
            a_half = T.alloc_shared([BS, BS], "bfloat16")
            a_f32 = T.alloc_shared([BS, BS], "float32")
            msk_ub = T.alloc_shared([BS, BS], "float32")
            a_mask = T.alloc_shared([BS, BS], "bfloat16")
            q_half = T.alloc_shared([BS, DK], "bfloat16")
            q_ub = T.alloc_shared([BS, DK], "float32")
            g_ub = T.alloc_shared([BS, DK], "float32")
            gL_ub = T.alloc_shared([BS, DK], "float32")
            gexp_ub = T.alloc_shared([BS, DK], "float32")
            qg_ub = T.alloc_shared([BS, DK], "float32")
            qg_hi_ub = T.alloc_shared([BS, DK], "bfloat16")
            qg_hi_f32 = T.alloc_shared([BS, DK], "float32")
            qg_lo_f32 = T.alloc_shared([BS, DK], "float32")
            qg_lo_ub = T.alloc_shared([BS, DK], "bfloat16")
            o_frag = T.alloc_fragment([BS, bDV], "float32")
            o_ub = T.alloc_shared([BS, bDV], "float32")
            o_half = T.alloc_shared([BS, bDV], "bfloat16")

            # loads (skeleton part)
            T.copy(A[bb, s0 : s0 + BS, bh * BS : (bh + 1) * BS], a_half)
            T.copy(a_half, a_f32)
            T.copy(Msk[0:BS, 0:BS], msk_ub)
            if vec:
                T.tile.mul(a_f32, a_f32, msk_ub)
            T.copy(a_f32, a_mask)
            T.copy(Q[bb, s0 : s0 + BS, bh * DK : (bh + 1) * DK], q_half)
            T.copy(q_half, q_ub)
            T.copy(G[bb, s0 : s0 + BS, bh * DK : (bh + 1) * DK], g_ub)
            if vec:
                T.tile.mul(gL_ub, g_ub, LN2)
                T.tile.exp(gexp_ub, gL_ub)
                T.tile.mul(q_ub, q_ub, scale)
                T.tile.mul(qg_ub, q_ub, gexp_ub)
                T.copy(qg_ub, qg_hi_ub)
                T.copy(qg_hi_ub, qg_hi_f32)
                T.tile.sub(qg_lo_f32, qg_ub, qg_hi_f32)
                T.copy(qg_lo_f32, qg_lo_ub)
            else:
                T.copy(q_half, qg_hi_ub)
                T.copy(q_half, qg_lo_ub)
            # relays
            T.copy(qg_hi_ub, qg_hi_l1)
            T.copy(qg_lo_ub, qg_lo_l1)
            T.copy(a_mask, A_l1)
            T.copy(Hstate[bb, bs, bh, 0:DK, 0:bDV], H_l1)
            T.copy(V[bb, s0 : s0 + BS, bh, 0:bDV], V_l1)
            # o chain (o_frag never written in probes -> reads L0C garbage,
            # timing-only)
            T.copy(o_frag, o_ub)
            T.copy(o_ub, o_half)
            T.copy(o_half, O[bb, s0 : s0 + BS, bh * DV : (bh + 1) * DV])

    return main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", choices=["skeleton", "vec"], default="skeleton")
    args = parser.parse_args()
    tilelang.disable_cache()

    B, S, H, DK, DV = 1, 8192, 64, 128, 128
    gen = torch.Generator().manual_seed(42)
    q = torch.randn(B, S, H, DK, generator=gen).bfloat16().npu()
    v = torch.randn(B, S, H, DV, generator=gen).bfloat16().npu()
    g = torch.randn(B, S, H, DK, generator=gen).float().npu()
    A = torch.randn(B, S, H, 64, generator=gen).bfloat16().npu()
    h = torch.randn(B, S // 64, H, DK, DV, generator=gen).bfloat16().npu()
    msk = torch.tril(torch.ones(64, 64, dtype=torch.float32), 0).npu()

    func = probe_kernel(B, S, H, DK, DV, 64, DK**-0.5, vec=(args.probe == "vec"))
    _ = func(
        q.view(B, S, H * DK), v, g.view(B, S, H * DK), A.view(B, S, H * 64), h, msk
    )
    torch.npu.synchronize()

    rep, warmup = 10, 10
    for _ in range(warmup):
        func(
            q.view(B, S, H * DK), v, g.view(B, S, H * DK), A.view(B, S, H * 64), h, msk
        )
    torch.npu.synchronize()
    start = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    end = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start[i].record()
        func(
            q.view(B, S, H * DK), v, g.view(B, S, H * DK), A.view(B, S, H * 64), h, msk
        )
        end[i].record()
    torch.npu.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start, end)]
    mean = sum(times) / len(times)
    print(f"[probe {args.probe}] mean={mean:.3f} ms per-rep={[f'{t:.3f}' for t in times]}")


if __name__ == "__main__":
    main()
