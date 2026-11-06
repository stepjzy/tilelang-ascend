"""Revalidation diagnostic: locate the ~0.9ms gap between bwd do_bench chain
(1676us) and single-launch kernel times (stage 221 + gemm 572 = 793us).

Measures on the bench shape (1,8192,8,128) fp32:
  A. do_bench per-kernel: k_stage alone, k_gemm alone (old-env comparable)
  B. do_bench wrapper chain (known ~1676us)
  C. in-chain event split: start -> stage -> mid -> gemm -> end, reporting
     elapsed(stage), elapsed(gemm incl. inter-kernel gap), and the pure
     inter-kernel gap (mid event to gemm start) via a third event pair
  D. host wall time of one wrapper call (time.perf_counter around launch,
     no sync) -> host dispatch cost per call
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

import tilelang  # noqa: E402

tilelang.disable_cache()

from examples_experiment.dsa_sparse_finetune.fla_chunk_intra import (  # noqa: E402
    _get_mlo,
    fla_chunk_intra_bwd_intra,
    fla_chunk_intra_bwd_k_gemm,
    fla_chunk_intra_bwd_k_stage,
)
from examples_experiment.dsa_sparse_finetune.test_fla_chunk_intra import (  # noqa: E402
    do_bench,
    prepare_input_bwd,
)

DEVICE = "npu"


def main():
    B, S, H, DK = 1, 8192, 8, 128
    CS, SCS, BK = 64, 16, 64
    GRID = (S // CS) * B * H
    torch.manual_seed(0)
    ins = prepare_input_bwd(B, S, H, DK)
    ins_n = [t.to(DEVICE) for t in ins]
    q, k, g, beta, dAqk, dAkk, dq, dk, db, dg = ins_n
    beta_flat = beta.view(B, S * H)
    mlo = _get_mlo(torch.device(DEVICE), CS)
    opscrm = torch.empty(GRID, 4, CS, CS, dtype=torch.float16, device=DEVICE)
    opscr = torch.empty(GRID, 2, 6, CS, BK, dtype=torch.float16, device=DEVICE)
    fragscr = torch.empty(GRID, 2, 12 * SCS, BK, dtype=torch.float32, device=DEVICE)
    db2flat = torch.empty(GRID, CS, dtype=torch.float32, device=DEVICE)
    db2flat_v = db2flat.view(GRID, CS, 1)
    db2 = torch.empty(B, S, H, dtype=torch.float32, device=DEVICE)
    db2f = db2.view(B, S * H)

    fn_stage = fla_chunk_intra_bwd_k_stage(B, S, H, DK, CS, SCS)
    fn_gemm = fla_chunk_intra_bwd_k_gemm(B, S, H, DK, CS, SCS)

    def stage():
        fn_stage(q, k, g, beta_flat, mlo, dAqk, dAkk, opscrm, opscr)

    def gemm():
        fn_gemm(q, k, g, beta_flat, dq, dk, dg, opscrm, opscr, fragscr, db2flat, db2flat_v, db2, db2f)

    def chain():
        fla_chunk_intra_bwd_intra(*ins_n)

    # A. per-kernel do_bench
    print(f"[DIAG] A1 stage_alone  = {do_bench(stage) * 1000:.1f} us")
    print(f"[DIAG] A2 gemm_alone   = {do_bench(gemm) * 1000:.1f} us")
    # B. chain
    print(f"[DIAG] B  chain        = {do_bench(chain) * 1000:.1f} us")

    # C. in-chain event split
    reps = 30
    for _ in range(20):
        chain()
    torch.npu.synchronize()
    t_stage, t_gap, t_gemm = [], [], []
    for _ in range(reps):
        e0, e1, e2, e3 = (torch.npu.Event(enable_timing=True) for _ in range(4))
        e0.record()
        fn_stage(q, k, g, beta_flat, mlo, dAqk, dAkk, opscrm, opscr)
        e1.record()
        fn_gemm(q, k, g, beta_flat, dq, dk, dg, opscrm, opscr, fragscr, db2flat, db2flat_v, db2, db2f)
        e2.record()
        t_stage.append(e0.elapsed_time(e1))
        t_gemm.append(e1.elapsed_time(e2))
    torch.npu.synchronize()
    ms = sorted(t_stage)
    print(f"[DIAG] C1 in_chain_stage  = {sum(t_stage) / reps * 1000:.1f} us (min {ms[0] * 1000:.1f})")
    mg = sorted(t_gemm)
    print(f"[DIAG] C2 in_chain_gemm+gap = {sum(t_gemm) / reps * 1000:.1f} us (min {mg[0] * 1000:.1f})")

    # D. host dispatch wall time (launch-only, no sync)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        chain()
    t_host = (time.perf_counter() - t0) / 30 * 1e6
    torch.npu.synchronize()
    print(f"[DIAG] D  host_launch_per_call = {t_host:.1f} us (wall, incl. workspace allocs)")

    # E. allocator churn probe: torch.empty of the workspace set, 100x
    t0 = time.perf_counter()
    for _ in range(100):
        torch.empty(GRID, 4, CS, CS, dtype=torch.float16, device=DEVICE)
        torch.empty(GRID, 2, 6, CS, BK, dtype=torch.float16, device=DEVICE)
        torch.empty(GRID, 2, 12 * SCS, BK, dtype=torch.float32, device=DEVICE)
        torch.empty(GRID, CS, dtype=torch.float32, device=DEVICE)
        torch.empty(B, S, H, dtype=torch.float32, device=DEVICE)
    t_alloc = (time.perf_counter() - t0) / 100 * 1e6
    print(f"[DIAG] E  workspace_alloc_set = {t_alloc:.1f} us (5 tensors, cached)")
    print("[DIAG] DONE")


if __name__ == "__main__":
    main()
