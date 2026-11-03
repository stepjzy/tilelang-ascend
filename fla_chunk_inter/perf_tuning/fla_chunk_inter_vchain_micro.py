"""R3 decisive experiment 2: the Expert-mode V-side barrier-thinning yield.

R2 floor accounting hypothesis: the V-side wall is NOT engine-bound (mte2
2867 + vec 2503 + mte3 1447 = 6817us busy vs 6534 wall - nearly serial), it
is the SCALAR ISSUE STREAM serialized by per-op name-level auto barriers
(msprof aiv_scalar_vector_stall 6278us). An Expert-mode manual-sync
formulation (TL_ASCEND_AUTO_SYNC=False + burst-level set_flag/wait_flag -
one flag pair per engine handoff instead of one barrier per op) should
collapse the chain toward max-engine.

This micro measures that yield at bench tile sizes, one drain-style task
chain per iteration (6 loads -> ~17 vec ops -> 3 stores), L tasks per block,
GRID blocks:

  variant A: AUTO_SYNC=True  (the current Developer formulation)
  variant B: AUTO_SYNC=False + manual burst protocol (inside T.Scope("V")):
             wait(MTE3->MTE2) [prev stores done] -> 6 loads -> set/wait
             (MTE2->V) -> vec burst -> set/wait (V->MTE3) -> 3 stores ->
             set(MTE3->MTE2); one head-set prefill before the loop.

Both variants run the IDENTICAL op sequence (bodies inlined - the AST
parser silently drops plain-Python helper calls inside prim_func).

Correctness gate: fp64 golden (fp32 mixed tolerance family gate).
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
BS = 64
BDK = 64
GRID = 20  # one block per AI Core (fixed-core family recipe)
L = 20  # serial tasks per block (400 tasks total)

NTASK = GRID * L
S_TASK = NTASK * BS

BASE_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
AUTO_CFG = dict(BASE_CONFIGS)
AUTO_CFG[tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC] = True
MANUAL_CFG = dict(BASE_CONFIGS)
MANUAL_CFG[tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC] = False

# manual flag IDs (burst-level; well under the <10 pending capacity)
SIG_MTE3_MTE2 = 0  # cross-iteration WAR: prev stores done -> new loads
SIG_MTE2_V = 1  # load burst done -> vec
SIG_V_MTE3 = 2  # vec burst done -> stores


@tilelang.jit(out_idx=[5, 6, 7], pass_configs=AUTO_CFG)
def vchain_auto():
    """Variant A: AUTO_SYNC=True (per-op name-level barriers)."""

    @T.prim_func
    def main(
        G3: T.Tensor((1, S_TASK, BDK), "float32"),  # type: ignore
        Q3: T.Tensor((1, S_TASK, BDK), "float32"),  # type: ignore
        K3: T.Tensor((1, S_TASK, BDK), "float32"),  # type: ignore
        DQR3: T.Tensor((1, S_TASK, BDK), "float32"),  # type: ignore
        DKR3: T.Tensor((1, S_TASK, BDK), "float32"),  # type: ignore
        dq3: T.Tensor((1, S_TASK, BDK), "float32"),  # type: ignore
        dk3: T.Tensor((1, S_TASK, BDK), "float32"),  # type: ignore
        dg3: T.Tensor((1, S_TASK, BDK), "float32"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            for j in T.serial(L):
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
                pp_ub = T.alloc_shared((BDK,), "float32")
                pp2d_ub = T.alloc_shared((BS, BDK), "float32")

                tid = cid + j * GRID
                s0 = tid * BS
                T.copy(DQR3[0, s0 : s0 + BS, 0:BDK], dq_ub)
                T.copy(DKR3[0, s0 : s0 + BS, 0:BDK], dk_ub)
                T.copy(G3[0, s0 : s0 + BS, 0:BDK], g_ub)
                T.copy(G3[0, s0 + BS - 1, 0:BDK], gn_ub)
                T.copy(Q3[0, s0 : s0 + BS, 0:BDK], q_ub)
                T.copy(K3[0, s0 : s0 + BS, 0:BDK], k_ub)
                T.tile.mul(g_ub, g_ub, LN2)
                T.tile.mul(gn_ub, gn_ub, LN2)
                T.tile.exp(gexp_ub, g_ub)
                T.tile.mul(dq_ub, dq_ub, gexp_ub)
                T.tile.exp(gnexp_ub, gn_ub)
                T.tile.broadcast(gn2d_ub, gn_ub, axis=0)
                T.tile.sub(gnd_ub, gn2d_ub, g_ub)
                T.tile.exp(gnd_ub, gnd_ub)
                T.tile.mul(dk_ub, dk_ub, gnd_ub)
                T.tile.mul(dkk_ub, dk_ub, k_ub)
                T.reduce_sum(dkk_ub, pp_ub, dim=0, clear=True)
                T.tile.broadcast(pp2d_ub, pp_ub, axis=0)
                T.tile.mul(dg_ub, q_ub, dq_ub)
                T.tile.sub(dg_ub, dg_ub, dkk_ub)
                T.tile.add(dg_ub, dg_ub, pp2d_ub)
                T.copy(dq_ub, dq3[0, s0 : s0 + BS, 0:BDK])
                T.copy(dk_ub, dk3[0, s0 : s0 + BS, 0:BDK])
                T.copy(dg_ub, dg3[0, s0 : s0 + BS, 0:BDK])

    return main


@tilelang.jit(out_idx=[5, 6, 7], pass_configs=MANUAL_CFG)
def vchain_manual():
    """Variant B: AUTO_SYNC=False + burst-level manual flag protocol."""

    @T.prim_func
    def main(
        G3: T.Tensor((1, S_TASK, BDK), "float32"),  # type: ignore
        Q3: T.Tensor((1, S_TASK, BDK), "float32"),  # type: ignore
        K3: T.Tensor((1, S_TASK, BDK), "float32"),  # type: ignore
        DQR3: T.Tensor((1, S_TASK, BDK), "float32"),  # type: ignore
        DKR3: T.Tensor((1, S_TASK, BDK), "float32"),  # type: ignore
        dq3: T.Tensor((1, S_TASK, BDK), "float32"),  # type: ignore
        dk3: T.Tensor((1, S_TASK, BDK), "float32"),  # type: ignore
        dg3: T.Tensor((1, S_TASK, BDK), "float32"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):  # noqa: SIM117
            with T.Scope("V"):
                # head-set prefill: iteration 0's WAR wait (family head-set
                # precedent - counting semantics balanced by one extra set)
                T.set_flag("MTE3", "MTE2", SIG_MTE3_MTE2)
                for j in T.serial(L):
                    # cross-iteration WAR: the previous task's mte3 stores
                    # read these UB buffers; the new load burst waits them.
                    T.wait_flag("MTE3", "MTE2", SIG_MTE3_MTE2)

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
                    pp_ub = T.alloc_shared((BDK,), "float32")
                    pp2d_ub = T.alloc_shared((BS, BDK), "float32")

                    tid = cid + j * GRID
                    s0 = tid * BS
                    T.copy(DQR3[0, s0 : s0 + BS, 0:BDK], dq_ub)
                    T.copy(DKR3[0, s0 : s0 + BS, 0:BDK], dk_ub)
                    T.copy(G3[0, s0 : s0 + BS, 0:BDK], g_ub)
                    T.copy(G3[0, s0 + BS - 1, 0:BDK], gn_ub)
                    T.copy(Q3[0, s0 : s0 + BS, 0:BDK], q_ub)
                    T.copy(K3[0, s0 : s0 + BS, 0:BDK], k_ub)
                    # load burst done -> vec
                    T.set_flag("MTE2", "V", SIG_MTE2_V)
                    T.wait_flag("MTE2", "V", SIG_MTE2_V)

                    T.tile.mul(g_ub, g_ub, LN2)
                    T.tile.mul(gn_ub, gn_ub, LN2)
                    T.tile.exp(gexp_ub, g_ub)
                    T.tile.mul(dq_ub, dq_ub, gexp_ub)
                    T.tile.exp(gnexp_ub, gn_ub)
                    T.tile.broadcast(gn2d_ub, gn_ub, axis=0)
                    T.tile.sub(gnd_ub, gn2d_ub, g_ub)
                    T.tile.exp(gnd_ub, gnd_ub)
                    T.tile.mul(dk_ub, dk_ub, gnd_ub)
                    T.tile.mul(dkk_ub, dk_ub, k_ub)
                    T.reduce_sum(dkk_ub, pp_ub, dim=0, clear=True)
                    T.tile.broadcast(pp2d_ub, pp_ub, axis=0)
                    T.tile.mul(dg_ub, q_ub, dq_ub)
                    T.tile.sub(dg_ub, dg_ub, dkk_ub)
                    T.tile.add(dg_ub, dg_ub, pp2d_ub)
                    # vec burst done -> mte3 stores
                    T.set_flag("V", "MTE3", SIG_V_MTE3)
                    T.wait_flag("V", "MTE3", SIG_V_MTE3)

                    T.copy(dq_ub, dq3[0, s0 : s0 + BS, 0:BDK])
                    T.copy(dk_ub, dk3[0, s0 : s0 + BS, 0:BDK])
                    T.copy(dg_ub, dg3[0, s0 : s0 + BS, 0:BDK])
                    # store burst done -> next iteration's load burst
                    T.set_flag("MTE3", "MTE2", SIG_MTE3_MTE2)

    return main


def golden(g, q, k, dqr, dkr):
    g_ = g.reshape(-1, BS, BDK).double()
    q_ = q.reshape(-1, BS, BDK).double()
    k_ = k.reshape(-1, BS, BDK).double()
    dqr_ = dqr.reshape(-1, BS, BDK).double()
    dkr_ = dkr.reshape(-1, BS, BDK).double()
    dq = dqr_ * torch.exp(g_ * LN2)
    gn = g_[:, -1, :]
    dk = dkr_ * torch.exp((gn.unsqueeze(1) - g_) * LN2)
    dkk = dk * k_
    dg = q_ * dq - dkk + dkk.sum(dim=1, keepdim=True)
    return dq.float().reshape(1, -1, BDK), dk.float().reshape(1, -1, BDK), dg.float().reshape(1, -1, BDK)


def bench(fn, warmup=20, rep=10):
    import torch.npu

    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        fn()
        ends[i].record()
    torch.npu.synchronize()
    ts = [starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(rep)]
    return sum(ts) / len(ts)


def run(label, factory):
    torch.manual_seed(0)
    g = torch.randn(1, S_TASK, BDK)
    q = torch.randn(1, S_TASK, BDK)
    k = torch.randn(1, S_TASK, BDK)
    dqr = torch.randn(1, S_TASK, BDK)
    dkr = torch.randn(1, S_TASK, BDK)
    refs = golden(g, q, k, dqr, dkr)
    kern = factory()
    args = [t.npu() for t in (g, q, k, dqr, dkr)]
    outs = kern(*args)
    oks = []
    for out, ref in zip(outs, refs):
        o = out.float().cpu().reshape(-1)
        r = ref.float().cpu().reshape(-1)
        diff = (o - r).abs()
        tol = 1.53e-5 + 9.77e-4 * r.abs()
        ratio = (diff <= tol).float().mean().item()
        oks.append((ratio >= 0.99 and diff.max().item() <= 1e-2, ratio, diff.max().item()))
    allok = all(o[0] for o in oks)
    detail = " | ".join(f"ratio={r:.4f} max={m:.3e}" for _, r, m in oks)
    t = bench(lambda: kern(*args))
    per_task = t / NTASK
    print(f"[VCHAIN] {label}: {'PASS' if allok else 'FAIL'} ({detail}) | wall={t:.1f}us ({per_task:.2f}us/task x {NTASK})")


if __name__ == "__main__":
    run("A_auto_sync", vchain_auto)
    run("B_manual_burst", vchain_manual)
