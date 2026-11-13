"""Marker-value experiment (reconstruction) for the dbeta scatter address bug.

marker = cid * BS + i   (cid = block id, i = in-chunk row index); fp32-exact.
Expected: dbeta[bb, bs*BS+i, bh] == marker of cid = bs*B*H + bb*H + bh.
A spot whose value decodes to a different origin proves misdirected stores.
"""

import tilelang
import tilelang.language as T
import torch

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

B, S, H, BS = 2, 128, 4, 64          # NT=2 -> GRID = NT*B*H = 16 blocks
NT = S // BS
GRID = NT * B * H

M = torch.arange(BS, dtype=torch.float32).repeat(GRID, 1)


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
def kern():
    @T.prim_func
    def main(
        Min: T.Tensor((GRID, BS), "float32"),
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
            acc = T.alloc_shared([BS], "float32")

            T.copy(Min[cid, 0:BS], m_ub)
            # make acc a COMPUTED value (not a plain loaded buffer):
            # acc = (m_ub + m_ub) * 0.5 == m_ub exactly, via runtime arithmetic
            T.tile.add(acc, m_ub, m_ub)
            T.tile.fill(half, 0.5)
            T.tile.mul(acc, acc, half)

            # crime scene: computed source + strided store, no GM staging
            T.copy(acc, dbeta[bb, s0 : s0 + BS, bh])

    return main


def expected():
    t_ = torch.arange(S)
    bb_ = torch.arange(B).view(B, 1, 1)
    bh_ = torch.arange(H).view(1, 1, H)
    bs_ = (t_ // BS).view(1, S, 1)
    cid = bs_ * (B * H) + bb_ * H + bh_
    i_ = (t_ % BS).view(1, S, 1)
    return (cid * BS + i_).expand(B, S, H).contiguous().float()


if __name__ == "__main__":
    tilelang.disable_cache()
    k = kern()
    out = k(M.npu())
    torch.npu.synchronize()
    out = out.cpu()

    exp = expected()
    bad = out != exp
    n_bad = int(bad.sum())
    print(f"GRID={GRID}, positions={B * S * H}, mismatched={n_bad}")
    if n_bad == 0:
        print("ALL MARKERS CORRECT -> computed-source strided store works in this minimal form")
        print("(bug A needs fuller operator context, or is fixed on this build)")
    else:
        print("--- misdirected samples ---")
        for row in bad.nonzero()[:8]:
            bb, t, bh = row.tolist()
            obs = out[bb, t, bh].item()
            cid_src, i_src = int(round(obs)) // BS, int(round(obs)) % BS
            # where cid_src SHOULD have written:
            bs_src = cid_src % NT
            true_bb = (cid_src // NT) // H
            true_bh = (cid_src // NT) % H
            print(f"pos (bb={bb}, t={t}, bh={bh}): got={obs} "
                  f"(origin: cid={cid_src}, i={i_src}, should be at bb={true_bb}, "
                  f"s0+{i_src}+{bs_src * BS}, bh={true_bh})  "
                  f"expected_here={exp[bb, t, bh].item()}")
