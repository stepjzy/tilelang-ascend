"""Control arm: same computation, but stage through GM before scatter.

Same kernel structure as scatter_simple.py, except the final 3 lines:
  direct: T.copy(acc, Out[bb, 0:BS, bh])
  staged: T.copy(acc, Scratch[cid,0:BS]) → flag → T.copy(ScratchV[cid,0:BS,0:1], buf) → T.copy(buf, Out[bb,0:BS,bh])

Usage: python scatter_simple_ctrl.py
"""

import tilelang
import tilelang.language as T
import torch

B, S, H, BS = 1, 64, 8, 64
GRID = H  # 8 blocks


@tilelang.jit(out_idx=[3])
def kern_staged():
    @T.prim_func
    def main(
        In: T.Tensor((GRID, BS), "float32"),
        Scratch: T.Tensor((GRID, BS), "float32"),
        ScratchV: T.Tensor((GRID, BS, 1), "float32"),
        Out: T.Tensor((B, S, H), "float32"),
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bh = cid
            bb = 0

            in_ub = T.alloc_shared([BS], "float32")
            acc = T.alloc_shared([BS], "float32")
            buf = T.alloc_shared([BS, 1], "float32")

            # step 1: load from GM
            T.copy(In[cid, 0:BS], in_ub)
            # step 2: compute
            T.tile.add(acc, in_ub, in_ub)   # acc = 2 * input

            # step 3: GM staging (same pattern as delivered operator)
            T.copy(acc, Scratch[cid, 0:BS])
            T.set_flag("MTE3", "MTE2", 0)
            T.wait_flag("MTE3", "MTE2", 0)
            T.copy(ScratchV[cid, 0:BS, 0:1], buf)
            # step 4: scatter from GM-loaded source
            T.copy(buf, Out[bb, 0:BS, bh])

    return main


if __name__ == "__main__":
    tilelang.disable_cache()

    In = torch.arange(GRID * BS, dtype=torch.float32).reshape(GRID, BS)
    scratch = torch.zeros(GRID, BS, dtype=torch.float32)

    k = kern_staged()
    result = k(In.npu(), scratch.npu(), scratch.view(GRID, BS, 1).npu())
    torch.npu.synchronize()
    result = result.cpu()

    expected = torch.zeros(B, S, H, dtype=torch.float32)
    for bh in range(H):
        for i in range(S):
            expected[0, i, bh] = 2.0 * (bh * BS + i)

    diff = (result - expected).abs()
    max_err = diff.max().item()
    n_wrong = int((diff > 0.01).sum())
    total = B * S * H

    print(f"result shape: {result.shape}")
    print(f"max_abs_err:  {max_err:.6e}")
    print(f"wrong:        {n_wrong}/{total}")

    if n_wrong == 0:
        print("PASS: GM-staged scatter works in simple kernel!")
    else:
        print("FAIL: GM-staged scatter ALSO broken in simple kernel!")
        print("\nsample (head 0, first 8 rows):")
        for i in range(8):
            print(f"  Out[0,{i},0]: got={result[0,i,0].item():7.1f}  "
                  f"expected={expected[0,i,0].item():7.1f}")
