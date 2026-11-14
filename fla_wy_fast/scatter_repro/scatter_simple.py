"""Simplest scatter test: compute 64 values in UB, T.copy to strided GM destination.

No Pipelined, no gemm, no C/V combine. Just:
  1. Load 64 values from GM
  2. Do one arithmetic op (acc = in + in)  → makes it a "computed source"
  3. T.copy(acc, Out[0, 0:64, bh])          → scatter to strided destination

If this works: computed-source scatter is fine in simple kernels.
If this fails: scatter is broken even in the simplest case.
"""

import tilelang
import tilelang.language as T
import torch

# ---- shapes ----
B, S, H, BS = 1, 64, 8, 64   # one batch, one chunk of 64 tokens, 8 heads
GRID = H                      # 8 blocks, one per head


@tilelang.jit(out_idx=[1])
def kern():
    @T.prim_func
    def main(
        In: T.Tensor((GRID, BS), "float32"),   # [8, 64] input
        Out: T.Tensor((B, S, H), "float32"),    # [1, 64, 8] scatter dest
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bh = cid    # block ID == head index
            bb = 0      # single batch

            in_ub = T.alloc_shared([BS], "float32")
            acc = T.alloc_shared([BS], "float32")

            # step 1: load from GM
            T.copy(In[cid, 0:BS], in_ub)

            # step 2: compute (arithmetic → "computed source")
            T.tile.add(acc, in_ub, in_ub)   # acc = 2 * input

            # step 3: scatter to strided destination
            #   Out[0, i, bh] for i in 0..63
            #   stride between elements = H * 4 = 32 bytes
            T.copy(acc, Out[bb, 0:BS, bh])

    return main


if __name__ == "__main__":
    tilelang.disable_cache()

    # input: block cid row i has value cid*64 + i
    In = torch.arange(GRID * BS, dtype=torch.float32).reshape(GRID, BS)

    # run
    k = kern()
    result = k(In.npu())
    torch.npu.synchronize()
    result = result.cpu()

    # expected: Out[0, i, bh] = 2*(bh*64 + i), everything else = 0
    expected = torch.zeros(B, S, H, dtype=torch.float32)
    for bh in range(H):
        for i in range(S):
            expected[0, i, bh] = 2.0 * (bh * BS + i)

    # compare
    diff = (result - expected).abs()
    max_err = diff.max().item()
    n_wrong = int((diff > 0.01).sum())
    total = B * S * H

    print(f"result shape: {result.shape}")
    print(f"max_abs_err:  {max_err:.6e}")
    print(f"wrong:        {n_wrong}/{total}")

    if n_wrong == 0:
        print("PASS: computed-source scatter works in simple kernel!")
    else:
        print("FAIL: scatter produced wrong results!")
        print("\nsample (head 0, first 8 rows):")
        for i in range(8):
            print(f"  Out[0,{i},0]: got={result[0,i,0].item():7.1f}  "
                  f"expected={expected[0,i,0].item():7.1f}")
        print("\nsample (head 3, first 4 rows):")
        for i in range(4):
            print(f"  Out[0,{i},3]: got={result[0,i,3].item():7.1f}  "
                  f"expected={expected[0,i,3].item():7.1f}")
