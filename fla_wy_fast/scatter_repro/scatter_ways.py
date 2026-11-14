"""Scatter test v2: three different ways to write computed UB data to strided GM.

Arm 1 "dmadirect" : T.copy(acc, Out[bb, 0:BS, bh])          ← DMA path (known broken)
Arm 2 "parallel"  : for i in T.Parallel(BS): Out[bb, i, bh] = acc[i]  ← scalar stores
Arm 3 "serial"    : for i in T.serial(BS):   Out[bb, i, bh] = acc[i]  ← serial stores

All three compute the same values (acc = 2*input) and write to the same
strided destination. The ONLY difference is the write mechanism.

Usage: python scatter_ways.py <arm>
"""

import sys

import tilelang
import tilelang.language as T
import torch

B, S, H, BS = 1, 64, 8, 64
GRID = H  # 8 blocks


def make_kernel(arm):
    @tilelang.jit(out_idx=[1])
    def kern():
        @T.prim_func
        def main(
            In: T.Tensor((GRID, BS), "float32"),
            Out: T.Tensor((B, S, H), "float32"),
        ):
            with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
                bh = cid
                bb = 0

                in_ub = T.alloc_shared([BS], "float32")
                acc = T.alloc_shared([BS], "float32")

                # same computation for all arms
                T.copy(In[cid, 0:BS], in_ub)
                T.tile.add(acc, in_ub, in_ub)   # acc = 2 * input

                # ---- the ONLY difference: how we write to strided GM ----
                if arm == "dmadirect":
                    # DMA path: T.copy to strided slice (known broken)
                    T.copy(acc, Out[bb, 0:BS, bh])

                elif arm == "parallel":
                    # scalar path: element-wise stores via T.Parallel
                    for i in T.Parallel(BS):
                        Out[bb, i, bh] = acc[i]

                elif arm == "serial":
                    # serial scalar stores
                    for i in T.serial(BS):
                        Out[bb, i, bh] = acc[i]

        return main

    return kern()


def run(arm):
    In = torch.arange(GRID * BS, dtype=torch.float32).reshape(GRID, BS)

    try:
        k = make_kernel(arm)
        result = k(In.npu())
        torch.npu.synchronize()
        result = result.cpu()
    except Exception as e:
        print(f"[{arm}] COMPILE/RUN ERROR: {type(e).__name__}: {str(e)[:200]}")
        return

    expected = torch.zeros(B, S, H, dtype=torch.float32)
    for bh in range(H):
        for i in range(S):
            expected[0, i, bh] = 2.0 * (bh * BS + i)

    diff = (result - expected).abs()
    max_err = diff.max().item()
    n_wrong = int((diff > 0.01).sum())
    total = B * S * H

    status = "PASS" if n_wrong == 0 else "FAIL"
    print(f"[{arm}] wrong={n_wrong}/{total}  max_err={max_err:.4e}  {status}")

    if n_wrong > 0:
        for i in range(min(4, S)):
            print(f"  Out[0,{i},0]: got={result[0,i,0].item():7.1f}  "
                  f"expected={expected[0,i,0].item():7.1f}")
    elif arm != "dmadirect":
        print(f"  → scalar-path scatter WORKS! (DMA path was broken)")


if __name__ == "__main__":
    tilelang.disable_cache()
    arms = sys.argv[1:] if len(sys.argv) > 1 else ["dmadirect", "parallel", "serial"]
    for arm in arms:
        print(f"\n===== ARM: {arm} =====")
        run(arm)
