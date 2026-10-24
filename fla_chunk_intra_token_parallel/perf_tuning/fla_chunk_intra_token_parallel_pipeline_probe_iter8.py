"""iter8 probe: compile + quick correctness spot-check on 2 shapes + short
bench (subset of the full gate; full gate runs separately)."""
import os
import sys
import time

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fla_chunk_intra_token_parallel import fla_chunk_intra_token_parallel  # noqa: E402
from test_fla_chunk_intra_token_parallel import (  # noqa: E402
    check_precision,
    fla_chunk_intra_token_parallel_ref,
    prepare_input_kda,
)


def probe(name, B, S, H, DK):
    q, k, gk, beta = prepare_input_kda(B, S, H, DK)
    aqk_ref, akk_ref = fla_chunk_intra_token_parallel_ref(q, k, gk, beta)
    q_d, k_d, gk_d, beta_d = q.npu(), k.npu(), gk.npu(), beta.npu()
    t0 = time.time()
    aqk_out, akk_out = fla_chunk_intra_token_parallel(q_d, k_d, gk_d, beta_d)
    torch.npu.synchronize()
    print(f"[{name}] compile+first-call {time.time() - t0:.1f}s")
    p1, r1, m1 = check_precision(aqk_out, aqk_ref, "bfloat16")
    p2, r2, m2 = check_precision(akk_out, akk_ref, "bfloat16")
    tag = "PASS" if (p1 and p2) else "FAIL"
    print(f"[{tag}] {name} shape=({B},{S},{H},{DK}) "
          f"Aqk ratio={r1:.4f} max_abs={m1:.3e} | Akk ratio={r2:.4f} max_abs={m2:.3e}")
    # repeat-run race check: 5 more launches, values must stay identical
    for it in range(5):
        a2, k2 = fla_chunk_intra_token_parallel(q_d, k_d, gk_d, beta_d)
        torch.npu.synchronize()
        if not (torch.equal(a2, aqk_out) and torch.equal(k2, akk_out)):
            print(f"[RACE] {name} iter {it}: outputs differ between runs")
            return False
    print(f"[STABLE] {name}: 5 repeat launches bit-identical")
    return p1 and p2


if __name__ == "__main__":
    ok = True
    ok &= probe("l0_min", 1, 64, 8, 128)
    ok &= probe("l0_bench_repr", 1, 1024, 64, 128)
    print("[PROBE_RESULT]", "PASS" if ok else "FAIL")
