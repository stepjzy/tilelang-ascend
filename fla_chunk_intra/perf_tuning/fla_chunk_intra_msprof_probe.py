"""Single-launch msprof probe for fla_chunk_intra per-kernel profiling.

Usage (one kernel per run, KERNEL env selects):
  KERNEL=inter msprof op --kernel-name=main_kernel --output=<dir> python this.py
Keeps runtime short: one warmup (JIT) + one profiled launch of ONE kernel.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

import tilelang  # noqa: E402

tilelang.disable_cache()

from examples_experiment.dsa_sparse_finetune.fla_chunk_intra import (  # noqa: E402
    _get_meye,
    _get_mlo,
    _get_mneg,
    fla_chunk_intra_bwd_k_gemm,
    fla_chunk_intra_bwd_k_stage,
    fla_chunk_intra_fwd_k_ai,
    fla_chunk_intra_fwd_k_g,
    fla_chunk_intra_fwd_k_g2,
    fla_chunk_intra_fwd_k_subst,
    fla_chunk_intra_fwd_k_inter,
    fla_chunk_intra_fwd_k_y,
)
from examples_experiment.dsa_sparse_finetune.test_fla_chunk_intra import (  # noqa: E402
    prepare_input_bwd,
    prepare_input_fwd,
)

DEVICE = "npu"
KERNEL = os.environ.get("KERNEL", "inter")


def main():
    CS, SCS = 64, 16
    if KERNEL in ("inter", "g", "subst", "g2", "y", "ai"):
        B, S, H, DK = 1, 8192, 64, 128
        torch.manual_seed(42)
        q, k, gk, beta, ad = prepare_input_fwd(B, S, H, DK)
        q_n, k_n, gk_n, beta_n, ad_n = (t.to(DEVICE) for t in (q, k, gk, beta, ad))
        beta_flat = beta_n.view(B, S * H)
        mneg = _get_mneg(torch.device(DEVICE), SCS)
        meye16 = _get_meye(torch.device(DEVICE), SCS)
        pair = lambda: (  # noqa: E731
            torch.empty(B, S, H, CS, dtype=torch.bfloat16, device=DEVICE),
            torch.empty(B, S, H, CS, dtype=torch.bfloat16, device=DEVICE),
        )
        akkbp_hi, akkbp_lo = pair()
        g_hi, g_lo = pair()
        g2_hi, g2_lo = pair()
        y_hi, y_lo = pair()
        aid_hi, aid_lo = pair()
        if KERNEL == "inter":
            fn = fla_chunk_intra_fwd_k_inter(B, S, H, DK, CS, SCS, 1.0)
            call = lambda: fn(q_n, k_n, gk_n, beta_flat, akkbp_hi, akkbp_lo)  # noqa: E731
        elif KERNEL == "g":
            fn = fla_chunk_intra_fwd_k_g(B, S, H, DK, CS, SCS)
            call = lambda: fn(akkbp_hi, akkbp_lo, aid_hi, aid_lo, g_hi, g_lo)  # noqa: E731
        elif KERNEL == "subst":
            fn = fla_chunk_intra_fwd_k_subst(B, S, H, DK, CS, SCS)
            call = lambda: fn(ad_n, mneg, meye16, aid_hi, aid_lo)  # noqa: E731
        elif KERNEL == "g2":
            fn = fla_chunk_intra_fwd_k_g2(B, S, H, DK, CS, SCS)
            call = lambda: fn(g_hi, g_lo, g2_hi, g2_lo)  # noqa: E731
        elif KERNEL == "y":
            fn = fla_chunk_intra_fwd_k_y(B, S, H, DK, CS, SCS)
            call = lambda: fn(g2_hi, g2_lo, aid_hi, aid_lo, y_hi, y_lo)  # noqa: E731
        else:
            fn = fla_chunk_intra_fwd_k_ai(B, S, H, DK, CS, SCS)
            call = lambda: fn(g_hi, g_lo, y_hi, y_lo)  # noqa: E731
    else:
        B, S, H, DK = 1, 8192, 8, 128
        NT = S // CS
        GRID = NT * B * H
        BK = DK // 2
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
        if KERNEL == "stage":
            fn = fla_chunk_intra_bwd_k_stage(B, S, H, DK, CS, SCS)
            call = lambda: fn(q, k, g, beta_flat, mlo, dAqk, dAkk, opscrm, opscr)  # noqa: E731
        else:
            fn = fla_chunk_intra_bwd_k_gemm(B, S, H, DK, CS, SCS)
            call = lambda: fn(q, k, g, beta_flat, dq, dk, dg, opscrm, opscr,  # noqa: E731
                              fragscr, db2flat, db2flat_v, db2, db2f)
    for _ in range(2):  # first = JIT compile, second = profiled launch
        call()
    torch.npu.synchronize()
    print(f"[probe] {KERNEL} done")


if __name__ == "__main__":
    main()
