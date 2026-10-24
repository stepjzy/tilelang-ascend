"""iter9 probe (candidate C: bf16 o_relay).

Checks:
1. Bit-agreement of (Aqk, Akk) vs the iter8 fp32-relay kernel (loaded from
   the impl_iter9_before.py backup) on randn / dbound / infq stress cases
   AND the L1 generalization shapes (H=2/6/12, SCS=8, DK=48, DK=32) --
   the fixpipe fp32->bf16 rounding must match the V-side CAST_RINT path
   bit-for-bit (the final output contract is bf16 either way).
2. Tolerance gate vs the CPU golden (the real precision contract).
3. Short bench at the bench shape vs the recorded iter8 baseline.
"""
import importlib.util
import os
import sys
import time

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import fla_chunk_intra_token_parallel as cand_mod  # noqa: E402
from test_fla_chunk_intra_token_parallel import (  # noqa: E402
    check_precision,
    fla_chunk_intra_token_parallel_ref,
)

_spec = importlib.util.spec_from_file_location(
    "iter8_ref", os.path.join(HERE, "fla_chunk_intra_token_parallel_impl_iter9_before.py"))
iter8_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(iter8_mod)

cand_op = cand_mod.fla_chunk_intra_token_parallel
ref_op = iter8_mod.fla_chunk_intra_token_parallel


def make_inputs(B, S, H, DK, CS=64, SCS=16, case="randn", seed=42):
    torch.manual_seed(seed)
    q = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
    k = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
    beta = torch.randn(B, S, H, dtype=torch.bfloat16)
    gk = torch.nn.functional.logsigmoid(torch.randn(B, S, H, DK, dtype=torch.float32))
    gk = gk.view(B, S // 64, 64, H, DK).cumsum(dim=2).view(B, S, H, DK)
    if case == "dbound":
        pos = (torch.arange(S) % 64).float()
        gk = (-12.0 * pos).view(1, S, 1, 1).repeat(B, 1, H, DK).contiguous()
    elif case == "infq":
        flat = q.view(-1)
        flat[12::13] = float("inf")
        flat[16::17] = float("-inf")
        q = q.contiguous()
    return q, k, gk, beta


def bit_case(name, B, S, H, DK, CS=64, SCS=16, case="randn"):
    q, k, gk, beta = make_inputs(B, S, H, DK, CS, SCS, case)
    args = [t.npu() for t in (q, k, gk, beta)]
    aqk_c, akk_c = cand_op(*args, chunk_size=CS, sub_chunk_size=SCS)
    aqk_r, akk_r = ref_op(*args, chunk_size=CS, sub_chunk_size=SCS)
    torch.npu.synchronize()
    b1 = (aqk_c.view(torch.int16) == aqk_r.view(torch.int16)).float().mean().item()
    b2 = (akk_c.view(torch.int16) == akk_r.view(torch.int16)).float().mean().item()
    aqk_g, akk_g = fla_chunk_intra_token_parallel_ref(q, k, gk, beta, CS, SCS)
    p1, r1, m1 = check_precision(aqk_c.cpu(), aqk_g, "bfloat16")
    p2, r2, m2 = check_precision(akk_c.cpu(), akk_g, "bfloat16")
    tag = "PASS" if (p1 and p2 and b1 == 1.0 and b2 == 1.0) else ("GATE_PASS" if (p1 and p2) else "FAIL")
    print(f"[{tag}] {name} case={case} shape=({B},{S},{H},{DK}) cs={CS} scs={SCS} "
          f"bit(aqk/akk)={b1:.6f}/{b2:.6f} | golden Aqk p={p1} r={r1:.4f} | Akk p={p2} r={r2:.4f}")
    return p1 and p2, b1, b2


def bench_events(fn, args, warmup=20, rep=10):
    for _ in range(warmup):
        fn(*args)
    torch.npu.synchronize()
    start = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    end = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start[i].record()
        fn(*args)
        end[i].record()
    torch.npu.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start, end)]
    return sum(times) / len(times) * 1000.0, times


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    ok = True
    if stage in ("all", "stress"):
        # stress cases at the bench-like head layout (BH=8)
        for case in ("randn", "dbound", "infq"):
            p, b1, b2 = bit_case("benchshape", 1, 256, 8, 128, case=case)
            ok &= p and b1 == 1.0 and b2 == 1.0
    if stage in ("all", "gen"):
        # generalization shapes (L1 coverage: BH=2/4 paths + SCS=8 + DK=48/32)
        for (name, shp, cs, scs) in [
            ("h6_bh2", (1, 128, 6, 128), 64, 16),
            ("h12_bh4", (1, 128, 12, 128), 64, 16),
            ("h2_min", (1, 64, 2, 32), 64, 16),
            ("dk48", (1, 64, 8, 48), 64, 16),
            ("scs8", (1, 128, 8, 128), 64, 8),
        ]:
            p, b1, b2 = bit_case(name, *shp, CS=cs, SCS=scs)
            ok &= p and b1 == 1.0 and b2 == 1.0
    if stage in ("all", "bench"):
        # short bench at the bench shape
        B, S, H, DK = 1, 8192, 64, 128
        q, k, gk, beta = make_inputs(B, S, H, DK)
        args = [t.npu() for t in (q, k, gk, beta)]
        t0 = time.time()
        cand_op(*args)
        torch.npu.synchronize()
        print(f"[bench] compile+first-call {time.time() - t0:.1f}s")
        mean_us, times = bench_events(cand_op, args)
        ref_us, _ = bench_events(ref_op, args)
        base = 2125.0
        print(f"[bench] candidate events mean = {mean_us:.1f} us (per-rep {[f'{t:.0f}' for t in times]})")
        print(f"[bench] iter8-ref (same session) = {ref_us:.1f} us; recorded iter8 baseline = {base:.1f} us")
        print(f"[bench] improvement vs recorded = {(base - mean_us) / base * 100:.2f}% | vs same-session ref = {(ref_us - mean_us) / ref_us * 100:.2f}%")
    print("[PROBE_RESULT]", "PASS" if ok else "FAIL")
