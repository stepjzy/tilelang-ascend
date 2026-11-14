"""Bisect where the staged chain loses data.

Chain:  ① T.copy(acc, Scratch)  ② flag  ③ T.copy(ScratchV, buf)  ④ T.copy(buf, Out)

Arms:
  s1_write     : only ①, then read Scratch back contiguously -> OutFlat
                 (PASS => Scratch holds correct data; FAIL => write step ① broken)
  s2_readback  : ①+②+③, then write buf contiguously -> OutFlat
                 (PASS => readback ③ ok; FAIL => flag/readback broken)
  s3_full      : full chain (known: 511/512 zeros)
"""

import sys

import tilelang
import tilelang.language as T
import torch

B, S, H, BS = 1, 64, 8, 64
GRID = H


@tilelang.jit(out_idx=[3])
def kern_s1_write():
    """Only write acc into Scratch, then read Scratch back to OutFlat."""
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
            T.copy(In[cid, 0:BS], in_ub)
            T.tile.add(acc, in_ub, in_ub)
            # ① write to Scratch
            T.copy(acc, Scratch[cid, 0:BS])
            T.set_flag("MTE3", "MTE2", 0)
            T.wait_flag("MTE3", "MTE2", 0)
            # read Scratch back (SAME param name) -> contiguous Out
            T.copy(Scratch[cid, 0:BS], acc)
            T.copy(acc, OutFlat[cid, 0:BS])

    return main


@tilelang.jit(out_idx=[3])
def kern_s2_readback():
    """①+②+③: write Scratch, flag, read back via ScratchV view into buf,
    then write buf contiguously to OutFlat."""
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
            # ① write to Scratch
            T.copy(acc, Scratch[cid, 0:BS])
            T.set_flag("MTE3", "MTE2", 0)
            T.wait_flag("MTE3", "MTE2", 0)
            # ③ read back via view -> buf[64,1]
            T.copy(ScratchV[cid, 0:BS, 0:1], buf)
            # write buf out contiguously
            T.copy(buf, OutFlat[cid, 0:BS])

    return main


@tilelang.jit(out_idx=[3])
def kern_s3_full():
    """Full chain: ①+②+③+④ strided scatter."""
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
            T.copy(acc, Scratch[cid, 0:BS])
            T.set_flag("MTE3", "MTE2", 0)
            T.wait_flag("MTE3", "MTE2", 0)
            T.copy(ScratchV[cid, 0:BS, 0:1], buf)
            T.copy(buf, Out[0, 0:BS, bh])

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
        print(f"  sample row0: got={result[0, :4].tolist()} exp={expected[0, :4].tolist()}")


def verify_strided(result, label):
    expected = torch.zeros(B, S, H, dtype=torch.float32)
    for bh in range(H):
        for i in range(S):
            expected[0, i, bh] = 2.0 * (bh * BS + i)
    diff = (result - expected).abs()
    n_wrong = int((diff > 0.01).sum())
    print(f"[{label}] wrong={n_wrong}/{B * S * H}  {'PASS' if n_wrong == 0 else 'FAIL'}")


if __name__ == "__main__":
    tilelang.disable_cache()
    arm = sys.argv[1]
    In = torch.arange(GRID * BS, dtype=torch.float32).reshape(GRID, BS)
    scratch = torch.zeros(GRID, BS, dtype=torch.float32)

    if arm == "s1_write":
        k = kern_s1_write()
        r = k(In.npu(), scratch.npu(), scratch.view(GRID, BS, 1).npu())
        torch.npu.synchronize()
        verify_flat(r.cpu(), "s1_write")
    elif arm == "s2_readback":
        k = kern_s2_readback()
        r = k(In.npu(), scratch.npu(), scratch.view(GRID, BS, 1).npu())
        torch.npu.synchronize()
        verify_flat(r.cpu(), "s2_readback")
    elif arm == "s3_full":
        k = kern_s3_full()
        r = k(In.npu(), scratch.npu(), scratch.view(GRID, BS, 1).npu())
        torch.npu.synchronize()
        verify_strided(r.cpu(), "s3_full")