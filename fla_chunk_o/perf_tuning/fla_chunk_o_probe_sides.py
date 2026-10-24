"""Probe the fixed-core pipelined kernel: which side is the critical path?
  --cut gemm    : AIC keeps loads + ws reads + FIX, drops the 3 gemms
  --cut vcompute: V keeps loads + relays + drain, drops mask/gate/hi-lo math
                  (relays carry raw staging buffers)
"""
import argparse
import os
import sys

import tilelang
import torch
import torch_npu  # noqa: F401
from tilelang import language as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fla_chunk_o as FK  # noqa: E402

LN2 = 0.6931471805599453


def build(cut):
    CORE_EFF = 20
    NSLOT = 2
    B, S, H, DK, DV = 1, 8192, 64, 128, 128
    BS = 64
    NT = S // BS
    GRID = NT * B * H
    TPT = (GRID + CORE_EFF - 1) // CORE_EFF

    @tilelang.jit(out_idx=[10], workspace_idx=[6, 7, 8, 9], pass_configs=FK.PASS_CONFIGS)
    def probe(B, S, H, DK, DV):
        BS, NT = 64, S // 64
        GRID = NT * B * H
        CORE_EFF = min(20, GRID)
        TPT = (GRID + CORE_EFF - 1) // CORE_EFF
        NSLOT = 2
        bDV = DV

        @T.prim_func
        def main(
            Q: T.Tensor((B, S, H * DK), "bfloat16"),  # type: ignore
            V: T.Tensor((B, S, H, DV), "bfloat16"),  # type: ignore
            G: T.Tensor((B, S, H * DK), "float32"),  # type: ignore
            A: T.Tensor((B, S, H * BS), "bfloat16"),  # type: ignore
            Hstate: T.Tensor((B, NT, H, DK, DV), "bfloat16"),  # type: ignore
            Msk: T.Tensor((BS, BS), "float32"),  # type: ignore
            workspace_1: T.Tensor((CORE_EFF, NSLOT, BS, DK), "bfloat16"),  # type: ignore
            workspace_2: T.Tensor((CORE_EFF, NSLOT, BS, DK), "bfloat16"),  # type: ignore
            workspace_3: T.Tensor((CORE_EFF, NSLOT, BS, BS), "bfloat16"),  # type: ignore
            workspace_4: T.Tensor((CORE_EFF, NSLOT, BS, bDV), "float32"),  # type: ignore
            O: T.Tensor((B, S, H * DV), "bfloat16"),  # type: ignore
        ):
            with T.Kernel(CORE_EFF, threads=2, is_npu=True) as (cid):
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

                T.copy(Msk[0:BS, 0:BS], msk_ub)
                for t in T.serial(TPT + 1):
                    tid = cid + t * CORE_EFF
                    pt = T.if_then_else(tid < GRID, tid, cid)
                    bs = pt % NT
                    bbh = pt // NT
                    bb = bbh // H
                    bh = bbh % H
                    s0 = bs * BS
                    slot = t % NSLOT
                    dt_raw = tid - CORE_EFF
                    dt = T.if_then_else(
                        t >= 1, T.if_then_else(dt_raw < GRID, dt_raw, cid), cid
                    )
                    dbs = dt % NT
                    dbbh = dt // NT
                    dbb = dbbh // H
                    dbh = dbbh % H
                    ds0 = dbs * BS
                    dslot = (t + 1) % NSLOT

                    T.copy(A[bb, s0 : s0 + BS, bh * BS : (bh + 1) * BS], a_half)
                    T.copy(Q[bb, s0 : s0 + BS, bh * DK : (bh + 1) * DK], q_half)
                    T.copy(G[bb, s0 : s0 + BS, bh * DK : (bh + 1) * DK], g_ub)

                    if cut == "vcompute":
                        T.copy(q_half, qg_hi_ub)
                        T.copy(q_half, qg_lo_ub)
                        T.copy(a_half, a_mask)
                    else:
                        T.copy(a_half, a_f32)
                        T.tile.mul(a_f32, a_f32, msk_ub)
                        T.copy(a_f32, a_mask)
                        T.copy(q_half, q_ub)
                        T.tile.mul(gL_ub, g_ub, LN2)
                        T.tile.exp(gexp_ub, gL_ub)
                        T.tile.mul(q_ub, q_ub, 0.088388)
                        T.tile.mul(qg_ub, q_ub, gexp_ub)
                        T.copy(qg_ub, qg_hi_ub)
                        T.copy(qg_hi_ub, qg_hi_f32)
                        T.tile.sub(qg_lo_f32, qg_ub, qg_hi_f32)
                        T.copy(qg_lo_f32, qg_lo_ub)

                    T.copy(qg_hi_ub, workspace_1[cid, slot, 0:BS, 0:DK])
                    T.copy(qg_lo_ub, workspace_2[cid, slot, 0:BS, 0:DK])
                    T.copy(a_mask, workspace_3[cid, slot, 0:BS, 0:BS])

                    T.copy(Hstate[bb, bs, bh, 0:DK, 0:bDV], H_l1)
                    T.copy(V[bb, s0 : s0 + BS, bh, 0:bDV], V_l1)

                    T.copy(workspace_1[cid, slot, 0:BS, 0:DK], qg_hi_l1)
                    if cut != "gemm":
                        T.gemm_v0(qg_hi_l1, H_l1, o_frag, init=True)
                    T.copy(workspace_2[cid, slot, 0:BS, 0:DK], qg_lo_l1)
                    if cut != "gemm":
                        T.gemm_v0(qg_lo_l1, H_l1, o_frag, init=False)
                    T.copy(workspace_3[cid, slot, 0:BS, 0:BS], A_l1)
                    if cut != "gemm":
                        T.gemm_v0(A_l1, V_l1, o_frag, init=False)
                    T.copy(o_frag, workspace_4[cid, slot, 0:BS, 0:bDV])

                    T.copy(workspace_4[cid, dslot, 0:BS, 0:bDV], o_ub)
                    T.copy(o_ub, o_half)
                    T.copy(o_half, O[dbb, ds0 : ds0 + BS, dbh * DV : (dbh + 1) * DV])

        return main

    return probe(B, S, H, DK, DV)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cut", choices=["gemm", "vcompute"], required=True)
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

    func = build(args.cut)
    _ = func(q.view(B, S, H * DK), v, g.view(B, S, H * DK), A.view(B, S, H * 64), h, msk)
    torch.npu.synchronize()
    rep, warmup = 10, 10
    for _ in range(warmup):
        func(q.view(B, S, H * DK), v, g.view(B, S, H * DK), A.view(B, S, H * 64), h, msk)
    torch.npu.synchronize()
    start = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    end = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start[i].record()
        func(q.view(B, S, H * DK), v, g.view(B, S, H * DK), A.view(B, S, H * 64), h, msk)
        end[i].record()
    torch.npu.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start, end)]
    print(f"[cut {args.cut}] mean={sum(times)/rep:.3f} ms per-rep={[f'{t:.3f}' for t in times]}")


if __name__ == "__main__":
    main()
