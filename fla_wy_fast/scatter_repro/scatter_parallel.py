"""Test: can we write element-by-element to strided GM destination at all?"""

import tilelang
import tilelang.language as T
import torch

B, S, H, BS = 1, 64, 8, 64
GRID = H


@tilelang.jit(out_idx=[1])
def kern():
    @T.prim_func
    def main(
        In: T.Tensor((GRID, BS), "float32"),
        Out: T.Tensor((B, S, H), "float32"),
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bh = cid
            acc = T.alloc_shared([BS], "float32")
            T.copy(In[cid, 0:BS], acc)
            # element-by-element write to strided GM
            for i in T.Parallel(BS):
                Out[0, i, bh] = acc[i]

    return main


if __name__ == "__main__":
    tilelang.disable_cache()
    In = torch.arange(GRID * BS, dtype=torch.float32).reshape(GRID, BS)
    k = kern()
    r = k(In.npu())
    torch.npu.synchronize()
    r = r.cpu()

    expected = torch.zeros(B, S, H, dtype=torch.float32)
    for bh in range(H):
        for i in range(S):
            expected[0, i, bh] = bh * BS + i

    diff = (r - expected).abs()
    n_wrong = int((diff > 0.01).sum())
    print(f"parallel elementwise: wrong={n_wrong}/{B*S*H}")
    if n_wrong == 0:
        print("PASS")
    else:
        print(f"FAIL, sample: got={r[0,1,0].item():.1f} expected={expected[0,1,0].item():.1f}")
