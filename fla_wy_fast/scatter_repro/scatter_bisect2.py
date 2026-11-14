"""More bisect: isolate what breaks in the read-back step of a SIMPLE kernel.

Hypotheses:
  H1: View [*, BS, 1] read-back itself is broken in simple kernels (sub-32B?).
      Test: read back with the SAME param shape [GRID, BS] instead of view.
  H2: The explicit flag pair breaks in simple kernels (maybe harmless upstream).
      Test: same as s2 but WITHOUT set_flag/wait_flag.
  H3: [BS] -> [BS,1] UB shape conversion is the problem.

Arms:
  n1_noview : write Scratch, (no flag), read back via Scratch (same param) -> out
  n2_flagless: write Scratch, NO flag, read back via ScratchV view -> out
  n3_view   : write Scratch, flag, read back via ScratchV view (known FAIL)
"""

import sys

import tilelang
import tilelang.language as T
import torch

B, S, H, BS = 1, 64, 8, 64
GRID = H


@tilelang.jit(out_idx=[2])
def kern_n1_noview():
    """Same-param roundtrip without any flag."""
    @T.prim_func
    def main(
        In: T.Tensor((GRID, BS), "float32"),
        Scratch: T.Tensor((GRID, BS), "float32"),
        OutFlat: T.Tensor((GRID, BS), "float32"),
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            in_ub = T.alloc_shared([BS], "float32")
            acc = T.alloc_shared([BS], "float32")
            T.copy(In[cid, 0:BS], in_ub)
            T.tile.add(acc, in_ub, in_ub)
            T.copy(acc, Scratch[cid, 0:BS])
            T.copy(Scratch[cid, 0:BS], acc)
            T.copy(acc, OutFlat[cid, 0:BS])

    return main


@tilelang.jit(out_idx=[3])
def kern_n2_flagless():
    """View read-back WITHOUT flag."""
    @T.prim_func
    def main(
        In: T.Tensor((GRID, BS), "float32"),
        Scratch: T.Tensor((GRID, BS), "float32"),
        ScratchV: T.Tensor((GRID, BS, 1), "float32"),
        OutFlat: T.Tensor((GRID, BS), "float32"),
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            in_ub = T.alloc_shared([BS], "float32")
            acc = T.alloc_shared([BS], "float32")
            buf = T.alloc_shared([BS, 1], "float32")
            T.copy(In[cid, 0:BS], in_ub)
            T.tile.add(acc, in_ub, in_ub)
            T.copy(acc, Scratch[cid, 0:BS])
            # NO flag here
            T.copy(ScratchV[cid, 0:BS, 0:1], buf)
            T.copy(buf, OutFlat[cid, 0:BS])

    return main


@tilelang.jit(out_idx=[3])
def kern_n3_view():
    """View read-back WITH flag (known FAIL)."""
    @T.prim_func
    def main(
        In: T.Tensor((GRID, BS), "float32"),
        Scratch: T.Tensor((GRID, BS), "float32"),
        ScratchV: T.Tensor((GRID, BS, 1), "float32"),
        OutFlat: T.Tensor((GRID, BS), "float32"),
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            in_ub = T.alloc_shared([BS], "float32")
            acc = T.alloc_shared([BS], "float32")
            buf = T.alloc_shared([BS, 1], "float32")
            T.copy(In[cid, 0:BS], in_ub)
            T.tile.add(acc, in_ub, in_ub)
            T.copy(acc, Scratch[cid, 0:BS])
            T.set_flag("MTE3", "MTE2", 0)
            T.wait_flag("MTE3", "MTE2", 0)
            T.copy(ScratchV[cid, 0:BS, 0:1], buf)
            T.copy(buf, OutFlat[cid, 0:BS])

    return main


def verify_flat(result, label):
    expected = torch.empty(GRID, BS, dtype=torch.float32)
    for cid in range(GRID):
        for i in range(BS):
            expected[cid, i] = 2.0 * (cid * BS + i)
    diff = (result - expected).abs()
    n_wrong = int((diff > 0.01).sum())
    print(f"[{label}] wrong={n_wrong}/{GRID * BS}  {'PASS' if n_wrong == 0 else 'FAIL'}")
    if n_wrong:
        print(f"  row0: got={result[0, :4].tolist()}")


if __name__ == "__main__":
    tilelang.disable_cache()
    arm = sys.argv[1]
    In = torch.arange(GRID * BS, dtype=torch.float32).reshape(GRID, BS)
    scratch = torch.zeros(GRID, BS, dtype=torch.float32)

    if arm == "n1_noview":
        k = kern_n1_noview()
        r = k(In.npu(), scratch.npu())
        torch.npu.synchronize()
        verify_flat(r.cpu(), "n1_noview")
    elif arm == "n2_flagless":
        k = kern_n2_flagless()
        r = k(In.npu(), scratch.npu(), scratch.view(GRID, BS, 1).npu())
        torch.npu.synchronize()
        verify_flat(r.cpu(), "n2_flagless")
    elif arm == "n3_view":
        k = kern_n3_view()
        r = k(In.npu(), scratch.npu(), scratch.view(GRID, BS, 1).npu())
        torch.npu.synchronize()
        verify_flat(r.cpu(), "n3_view")