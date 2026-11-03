"""Same-session A/B for the k_inter fixed-core task loop (R2 iter1).

Old form: history_version/fla_chunk_intra_perf_r2_iter0_before.py (R1 final:
per-block lockstep + auto per-cid relays)
New form: current fla_chunk_intra.py (fixed-core striped loop + explicit L2
relay workspaces + manual 1-late FIX flags)
Host API identical (workspace_idx auto-allocation). fwd bench shape.
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

import examples_experiment.dsa_sparse_finetune.fla_chunk_intra as new_mod  # noqa: E402
from examples_experiment.dsa_sparse_finetune.test_fla_chunk_intra import (  # noqa: E402
    do_bench,
    prepare_input_fwd,
)

_spec = importlib.util.spec_from_file_location(
    "intra_old_r2",
    os.path.join(HERE, "..", "history_version", "fla_chunk_intra_perf_r2_iter0_before.py"),
)
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)

DEVICE = "npu"
B, S, H, DK = 1, 8192, 64, 128
CS, SCS = 64, 16
torch.manual_seed(42)
q, k, gk, beta, ad = prepare_input_fwd(B, S, H, DK)
q_n, k_n, gk_n, beta_n, ad_n = (t.to(DEVICE) for t in (q, k, gk, beta, ad))
beta_flat = beta_n.view(B, S * H)
pair = lambda: (  # noqa: E731
    torch.empty(B, S, H, CS, dtype=torch.bfloat16, device=DEVICE),
    torch.empty(B, S, H, CS, dtype=torch.bfloat16, device=DEVICE),
)
akkbp_hi, akkbp_lo = pair()

f_old = old.fla_chunk_intra_fwd_k_inter(B, S, H, DK, CS, SCS, 1.0)
f_new = new_mod.fla_chunk_intra_fwd_k_inter(B, S, H, DK, CS, SCS, 1.0)

# warm + numeric cross-check
aqk_old = f_old(q_n, k_n, gk_n, beta_flat, akkbp_hi, akkbp_lo)
torch.npu.synchronize()
h_ref, l_ref = akkbp_hi.clone(), akkbp_lo.clone()
aqk_new = f_new(q_n, k_n, gk_n, beta_flat, akkbp_hi, akkbp_lo)
torch.npu.synchronize()
print(f"Aqk  old-vs-new max diff: {(aqk_old.float() - aqk_new.float()).abs().max().item():.3e}")
print(f"AkkbpHi old-vs-new: {(h_ref.float() - akkbp_hi.float()).abs().max().item():.3e}")
print(f"AkkbpLo old-vs-new: {(l_ref.float() - akkbp_lo.float()).abs().max().item():.3e}")

for rnd in range(2):
    t_o = do_bench(lambda: f_old(q_n, k_n, gk_n, beta_flat, akkbp_hi, akkbp_lo))
    t_n = do_bench(lambda: f_new(q_n, k_n, gk_n, beta_flat, akkbp_hi, akkbp_lo))
    print(f"round{rnd}: k_inter old={t_o*1000:.1f} new={t_n*1000:.1f} "
          f"({(t_n/t_o-1)*100:+.2f}%)")
