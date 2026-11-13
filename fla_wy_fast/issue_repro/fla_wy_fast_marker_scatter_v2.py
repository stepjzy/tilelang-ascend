"""Marker experiment v2 with CONTROL ARMS.

Arm 1 computed_src : acc = (m+m)*0.5 (computed on-chip)  -> strided store  [already FAILED 1023/1024]
Arm 2 loaded_src   : acc = T.copy(m_ub)     (GM-loaded)   -> strided store
Arm 3 loaded_src41 : acc[64,1] = copy from [GRID,64,1] view (GM-loaded)  -> strided store

If arm1 fails while arm2/3 pass -> clean minimal repro of bug A (computed-source scatter).
If all fail -> the minimal destination pattern itself is broken (no attribution).
"""

import sys

import tilelang
import tilelang.language as T
import torch

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

B, S, H, BS = 2, 128, 4, 64
NT = S // BS
GRID = NT * B * H

M = torch.arange(BS, dtype=torch.float32).repeat(GRID, 1)


def make_kernel(arm):
    if arm == 3:
        acc_shape = [BS, 1]
    else:
        acc_shape = [BS]

    @tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS)
    def kern():
        @T.prim_func
        def main(
            Min: T.Tensor((GRID, BS), "float32"),
            Min41: T.Tensor((GRID, BS, 1), "float32"),
            dbeta: T.Tensor((B, S, H), "float32"),
        ):
            with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
                bs = cid % NT
                bbh = cid // NT
                bb = bbh // H
                bh = bbh % H
                s0 = bs * BS

                m_ub = T.alloc_shared([BS], "float32")
                half = T.alloc_shared([BS], "float32")
                acc = T.alloc_shared(acc_shape, "float32")

                if arm == 1:
                    T.copy(Min[cid, 0:BS], m_ub)
                    T.tile.add(acc, m_ub, m_ub)
                    T.tile.fill(half, 0.5)
                    T.tile.mul(acc, acc, half)
                elif arm == 2:
                    T.copy(Min[cid, 0:BS], m_ub)
                    T.copy(m_ub, acc)
                else:
                    T.copy(Min41[cid, 0:BS, 0:1], acc)

                T.copy(acc, dbeta[bb, s0 : s0 + BS, bh])

        return main

    return kern()


def expected():
    t_ = torch.arange(S)
    bb_ = torch.arange(B).view(B, 1, 1)
    bh_ = torch.arange(H).view(1, 1, H)
    bs_ = (t_ // BS).view(1, S, 1)
    cid = bs_ * (B * H) + bb_ * H + bh_
    i_ = (t_ % BS).view(1, S, 1)
    return (cid * BS + i_).expand(B, S, H).contiguous().float()


if __name__ == "__main__":
    arm = int(sys.argv[1])
    tilelang.disable_cache()
    k = make_kernel(arm)
    if arm == 3:
        out = k(M.npu(), M.view(GRID, BS, 1).npu())
    else:
        out = k(M.npu(), M.view(GRID, BS, 1).npu())
    torch.npu.synchronize()
    out = out.cpu()

    exp = expected()
    bad = out != exp
    n_bad = int(bad.sum())
    print(f"ARM {arm}: positions={B * S * H}, mismatched={n_bad}")
    if n_bad:
        for row in bad.nonzero()[:4]:
            bb, t, bh = row.tolist()
            print(f"  pos (bb={bb}, t={t}, bh={bh}): got={out[bb, t, bh].item():.6g} "
                  f"expected={exp[bb, t, bh].item():.6g}")
