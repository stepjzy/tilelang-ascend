"""Direct AscendC-level scatter test via T.Scope("V") + element-wise writes.

This is the closest you can get to writing raw AscendC within TileLang:
- T.Scope("V") tells the compiler "execute this on the Vector core" (like Expert mode)
- Out[0, i, bh] = acc[i] is the direct equivalent of AscendC's SetValue / scalar store
- No T.copy, no DMA - just individual address-computed stores

If this works: hardware scatter is fine, bug is 100% in T.copy's DMA lowering.
If this fails: something even more fundamental is broken.

Also tries T.Scope("C") for completeness (Cube core version).
"""

import tilelang
import tilelang.language as T
import torch

B, S, H, BS = 1, 64, 8, 64
GRID = H  # 8 blocks


# ============================================================
# Arm 1: T.Scope("V") + Parallel element-wise stores
#   (Vector core, equivalent to AscendC SetValue on V-pipe)
# ============================================================
@tilelang.jit(out_idx=[1])
def kern_scope_v():
    @T.prim_func
    def main(
        In: T.Tensor((GRID, BS), "float32"),
        Out: T.Tensor((B, S, H), "float32"),
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bh = cid
            acc = T.alloc_shared([BS], "float32")
            T.copy(In[cid, 0:BS], acc)
            T.tile.add(acc, acc, acc)  # computed: acc = 2 * input

            # element-wise scatter with explicit Vector-scope
            with T.Scope("V"):
                for i in T.Parallel(BS):
                    Out[0, i, bh] = acc[i]

    return main


# ============================================================
# Arm 2: T.Scope("C") + Parallel element-wise stores
#   (Cube core version, just to check)
# ============================================================
@tilelang.jit(out_idx=[1])
def kern_scope_c():
    @T.prim_func
    def main(
        In: T.Tensor((GRID, BS), "float32"),
        Out: T.Tensor((B, S, H), "float32"),
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bh = cid
            acc = T.alloc_shared([BS], "float32")
            T.copy(In[cid, 0:BS], acc)
            T.tile.add(acc, acc, acc)

            with T.Scope("C"):
                for i in T.Parallel(BS):
                    Out[0, i, bh] = acc[i]

    return main


# ============================================================
# Arm 3: T.Scope("V") + T.copy (DMA inside explicit scope)
#   (Tests if the scope wrapper fixes the DMA path too)
# ============================================================
@tilelang.jit(out_idx=[1])
def kern_scope_v_copy():
    @T.prim_func
    def main(
        In: T.Tensor((GRID, BS), "float32"),
        Out: T.Tensor((B, S, H), "float32"),
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            bh = cid
            acc = T.alloc_shared([BS], "float32")
            T.copy(In[cid, 0:BS], acc)
            T.tile.add(acc, acc, acc)

            # DMA path but inside explicit scope
            with T.Scope("V"):
                T.copy(acc, Out[0, 0:BS, bh])

    return main


# ============================================================
# Host: run and verify
# ============================================================
def verify(result, label):
    expected = torch.zeros(B, S, H, dtype=torch.float32)
    for bh in range(H):
        for i in range(S):
            expected[0, i, bh] = 2.0 * (bh * BS + i)

    diff = (result - expected).abs()
    n_wrong = int((diff > 0.01).sum())
    status = "PASS" if n_wrong == 0 else "FAIL"
    print(f"[{label}] wrong={n_wrong}/{B*S*H}  {status}")
    if n_wrong > 0:
        print(f"  sample: Out[0,1,0] got={result[0,1,0].item():.1f} "
              f"expected={expected[0,1,0].item():.1f}")
    return n_wrong == 0


if __name__ == "__main__":
    tilelang.disable_cache()

    In = torch.arange(GRID * BS, dtype=torch.float32).reshape(GRID, BS)
    results = {}

    # Arm 1: Scope V + element-wise
    print("===== Scope V + element-wise stores =====")
    try:
        k = kern_scope_v()
        r = k(In.npu())
        torch.npu.synchronize()
        results["scope_v_ewise"] = verify(r.cpu(), "scope_v_ewise")
    except Exception as e:
        msg = str(e)
        last_line = msg.strip().split("\n")[-1][:200] if msg else str(e)[:200]
        print(f"[scope_v_ewise] ERROR: {last_line}")
        results["scope_v_ewise"] = False

    # Arm 2: Scope C + element-wise
    print("\n===== Scope C + element-wise stores =====")
    try:
        k = kern_scope_c()
        r = k(In.npu())
        torch.npu.synchronize()
        results["scope_c_ewise"] = verify(r.cpu(), "scope_c_ewise")
    except Exception as e:
        msg = str(e)
        last_line = msg.strip().split("\n")[-1][:200] if msg else str(e)[:200]
        print(f"[scope_c_ewise] ERROR: {last_line}")
        results["scope_c_ewise"] = False

    # Arm 3: Scope V + T.copy
    print("\n===== Scope V + T.copy (DMA in scope) =====")
    try:
        k = kern_scope_v_copy()
        r = k(In.npu())
        torch.npu.synchronize()
        results["scope_v_copy"] = verify(r.cpu(), "scope_v_copy")
    except Exception as e:
        msg = str(e)
        last_line = msg.strip().split("\n")[-1][:200] if msg else str(e)[:200]
        print(f"[scope_v_copy] ERROR: {last_line}")
        results["scope_v_copy"] = False

    # Summary
    print("\n===== SUMMARY =====")
    for name, ok in results.items():
        print(f"  {name}: {'✅ PASS' if ok else '❌ FAIL'}")
