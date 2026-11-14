"""Minimal scatter comparison: the user's exact question.

Three arms, all in the SAME simple kernel structure (no Pipelined, no gemm):

Arm "contig"  : T.copy(acc, OutFlat[cid, 0:BS])     ← contiguous write (control)
Arm "direct"  : T.copy(acc, Out[0, 0:BS, bh])       ← strided write, computed source
Arm "staged"  : acc → GM → flags → readback → T.copy(readback, Out[0, 0:BS, bh])

Question: does a CONTIGUOUS write work in a simple kernel?
  - If contig PASSES but direct/staged FAIL: the bug is specifically strided writes
  - If contig ALSO fails: the bug is in the output write path generally

Usage: python scatter_3arms.py <contig|direct|staged>
"""

import sys

import tilelang
import tilelang.language as T
import torch

B, S, H, BS = 1, 64, 8, 64
GRID = H  # 8 blocks, one per head


# ---------------------------------------------------------------
# Arm 1: contiguous write (control)
# ---------------------------------------------------------------
@tilelang.jit(out_idx=[1])
def kern_contig():
    @T.prim_func
    def main(
        In: T.Tensor((GRID, BS), "float32"),
        OutFlat: T.Tensor((GRID, BS), "float32"),   # contiguous destination!
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            in_ub = T.alloc_shared([BS], "float32")
            acc = T.alloc_shared([BS], "float32")
            T.copy(In[cid, 0:BS], in_ub)
            T.tile.add(acc, in_ub, in_ub)   # computed: acc = 2*input

            # contiguous write — the "obviously should work" case
            T.copy(acc, OutFlat[cid, 0:BS])

    return main


# ---------------------------------------------------------------
# Arm 2: direct strided write (computed source)
# ---------------------------------------------------------------
@tilelang.jit(out_idx=[1])
def kern_direct():
    @T.prim_func
    def main(
        In: T.Tensor((GRID, BS), "float32"),
        Out: T.Tensor((B, S, H), "float32"),
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bh = cid
            in_ub = T.alloc_shared([BS], "float32")
            acc = T.alloc_shared([BS], "float32")
            T.copy(In[cid, 0:BS], in_ub)
            T.tile.add(acc, in_ub, in_ub)

            # strided write, computed source
            T.copy(acc, Out[0, 0:BS, bh])

    return main


# ---------------------------------------------------------------
# Arm 3: staged strided write (GM roundtrip)
# ---------------------------------------------------------------
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
            in_ub = T.alloc_shared([BS], "float32")
            acc = T.alloc_shared([BS], "float32")
            buf = T.alloc_shared([BS, 1], "float32")
            T.copy(In[cid, 0:BS], in_ub)
            T.tile.add(acc, in_ub, in_ub)

            # GM roundtrip then strided write
            T.copy(acc, Scratch[cid, 0:BS])
            T.set_flag("MTE3", "MTE2", 0)
            T.wait_flag("MTE3", "MTE2", 0)
            T.copy(ScratchV[cid, 0:BS, 0:1], buf)
            T.copy(buf, Out[0, 0:BS, bh])

    return main


# ---------------------------------------------------------------
# Host
# ---------------------------------------------------------------
def verify_contig(result):
    expected = torch.zeros(GRID, BS, dtype=torch.float32)
    for cid in range(GRID):
        for i in range(BS):
            expected[cid, i] = 2.0 * (cid * BS + i)
    diff = (result - expected).abs()
    n_wrong = int((diff > 0.01).sum())
    print(f"[contig] wrong={n_wrong}/{GRID*BS}  "
          f"{'PASS' if n_wrong == 0 else 'FAIL'}")
    if n_wrong:
        print(f"  sample: Out[0,1] got={result[0,1].item():.1f} expected={expected[0,1].item():.1f}")


def verify_strided(result, label):
    expected = torch.zeros(B, S, H, dtype=torch.float32)
    for bh in range(H):
        for i in range(S):
            expected[0, i, bh] = 2.0 * (bh * BS + i)
    diff = (result - expected).abs()
    n_wrong = int((diff > 0.01).sum())
    print(f"[{label}] wrong={n_wrong}/{B*S*H}  "
          f"{'PASS' if n_wrong == 0 else 'FAIL'}")
    if n_wrong:
        print(f"  sample: Out[0,1,0] got={result[0,1,0].item():.1f} "
              f"expected={expected[0,1,0].item():.1f}")


if __name__ == "__main__":
    tilelang.disable_cache()
    arm = sys.argv[1]
    In = torch.arange(GRID * BS, dtype=torch.float32).reshape(GRID, BS)

    if arm == "contig":
        k = kern_contig()
        r = k(In.npu())
        torch.npu.synchronize()
        verify_contig(r.cpu())
    elif arm == "direct":
        k = kern_direct()
        r = k(In.npu())
        torch.npu.synchronize()
        verify_strided(r.cpu(), "direct")
    elif arm == "staged":
        k = kern_staged()
        scratch = torch.zeros(GRID, BS, dtype=torch.float32)
        r = k(In.npu(), scratch.npu(), scratch.view(GRID, BS, 1).npu())
        torch.npu.synchronize()
        verify_strided(r.cpu(), "staged")
