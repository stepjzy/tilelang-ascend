"""Same-session A/B for fla_chunk_intra bwd mid-anchor super-GEMM (r1 iter2).

Old form: history_version/fla_chunk_intra_perf_r1_iter0_before.py (snapshot)
New form: current fla_chunk_intra.py
Bench: do_bench NPU events warmup=20 rep=10 mean (device-side only), the
bwd bench shape (1,8192,8,128) fp32. Alternating A/B x2 for drift control.
"""
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
os.chdir(os.path.join(HERE, "..", "..", ".."))

import tilelang  # noqa: E402

tilelang.disable_cache()

from examples_experiment.dsa_sparse_finetune.fla_chunk_intra import (  # noqa: E402
    _get_mlo,
    fla_chunk_intra_bwd_k_gemm as k_gemm_new,
    fla_chunk_intra_bwd_k_stage as k_stage_new,
)
from examples_experiment.dsa_sparse_finetune.test_fla_chunk_intra import (  # noqa: E402
    do_bench,
    prepare_input_bwd,
)

_spec = importlib.util.spec_from_file_location(
    "intra_old",
    os.path.join(HERE, "..", "history_version", "fla_chunk_intra_perf_r1_iter0_before.py"),
)
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)
k_stage_old = _old.fla_chunk_intra_bwd_k_stage
k_gemm_old = _old.fla_chunk_intra_bwd_k_gemm

DEVICE = "npu"
B, S, H, DK = 1, 8192, 8, 128
CS, SCS = 64, 16
NT = S // CS
GRID = NT * B * H
BK = DK // 2

torch.manual_seed(0)
ins = prepare_input_bwd(B, S, H, DK)
ins_n = [t.to(DEVICE) for t in ins]
q, k, g, beta, dAqk, dAkk, dq, dk, db, dg = ins_n
beta_flat = beta.view(B, S * H)
mlo = _get_mlo(torch.device(DEVICE), CS)

# per-form staging allocations + partial outputs
def alloc_old():
    opscrm = torch.empty(GRID, 16 * SCS + 16 * CS, CS, dtype=torch.float16, device=DEVICE)
    opscr = torch.empty(GRID, 2, 86 * CS, BK, dtype=torch.float16, device=DEVICE)
    return opscrm, opscr


def alloc_new():
    opscrm = torch.empty(GRID, 4, CS, CS, dtype=torch.float16, device=DEVICE)
    opscr = torch.empty(GRID, 2, 6, CS, BK, dtype=torch.float16, device=DEVICE)
    return opscrm, opscr


fragscr = torch.empty(GRID, 2, 12 * SCS, BK, dtype=torch.float32, device=DEVICE)
db2flat = torch.empty(GRID, CS, dtype=torch.float32, device=DEVICE)
db2flat_v = db2flat.view(GRID, CS, 1)
db2 = torch.empty(B, S, H, dtype=torch.float32, device=DEVICE)
db2f = db2.view(B, S * H)

fs_old = k_stage_old(B, S, H, DK, CS, SCS)
fg_old = k_gemm_old(B, S, H, DK, CS, SCS)
fs_new = k_stage_new(B, S, H, DK, CS, SCS)
fg_new = k_gemm_new(B, S, H, DK, CS, SCS)

scrm_o, scr_o = alloc_old()
scrm_n, scr_n = alloc_new()

# warm + numeric cross-check (old vs new outputs)
fs_old(q, k, g, beta_flat, mlo, dAqk, dAkk, scrm_o, scr_o)
r_old = fg_old(q, k, g, beta_flat, dq, dk, dg, scrm_o, scr_o, fragscr,
              db2flat, db2flat_v, db2, db2f)
fs_new(q, k, g, beta_flat, mlo, dAqk, dAkk, scrm_n, scr_n)
r_new = fg_new(q, k, g, beta_flat, dq, dk, dg, scrm_n, scr_n, fragscr,
               db2flat, db2flat_v, db2, db2f)
torch.npu.synchronize()
for nm, a, b in zip(("dq2", "dk2", "dg2"), r_old, r_new):
    print(f"{nm} old-vs-new max diff: {(a - b).abs().max().item():.3e}")

for rnd in range(2):
    t_s_old = do_bench(lambda: fs_old(q, k, g, beta_flat, mlo, dAqk, dAkk, scrm_o, scr_o))
    t_g_old = do_bench(lambda: fg_old(q, k, g, beta_flat, dq, dk, dg, scrm_o, scr_o, fragscr,
                                      db2flat, db2flat_v, db2, db2f))
    t_s_new = do_bench(lambda: fs_new(q, k, g, beta_flat, mlo, dAqk, dAkk, scrm_n, scr_n))
    t_g_new = do_bench(lambda: fg_new(q, k, g, beta_flat, dq, dk, dg, scrm_n, scr_n, fragscr,
                                      db2flat, db2flat_v, db2, db2f))
    print(f"round{rnd}: k_stage old={t_s_old*1000:.1f} new={t_s_new*1000:.1f} "
          f"({(t_s_new/t_s_old-1)*100:+.2f}%) | k_gemm old={t_g_old*1000:.1f} "
          f"new={t_g_new*1000:.1f} ({(t_g_new/t_g_old-1)*100:+.2f}%) | "
          f"sum old={(t_s_old+t_g_old)*1000:.1f} new={(t_s_new+t_g_new)*1000:.1f}")
