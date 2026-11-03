"""Same-session A/B for the k_g ILP-interleaved substitution (r1 iter4).

Old form: current-file-before-iter4 = the k_g of iter3 (serialized 4 loops)
         -> reconstructed here from the iter2 snapshot + iter3 edits is NOT
         possible cleanly; instead old = this file's k_g before the iter4
         edit, saved by the caller as
         history_version/fla_chunk_intra_perf_r1_iter3_before.py.
New form: current fla_chunk_intra.py (interleaved single loop, per-s temps).
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
    "intra_old4",
    os.path.join(HERE, "..", "history_version", "fla_chunk_intra_perf_r1_iter3_before.py"),
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
mneg = new_mod._get_mneg(torch.device(DEVICE), SCS)
meye16 = new_mod._get_meye(torch.device(DEVICE), SCS)
pair = lambda: (  # noqa: E731
    torch.empty(B, S, H, CS, dtype=torch.bfloat16, device=DEVICE),
    torch.empty(B, S, H, CS, dtype=torch.bfloat16, device=DEVICE),
)
akkbp_hi, akkbp_lo = pair()
g_hi, g_lo = pair()
aid_hi, aid_lo = pair()

kg_old = old.fla_chunk_intra_fwd_k_g(B, S, H, DK, CS, SCS)
kg_new = new_mod.fla_chunk_intra_fwd_k_g(B, S, H, DK, CS, SCS)

# warm + numeric cross-check (identical math, in-order fp32 -> expect 0 diff)
kg_old(akkbp_hi, akkbp_lo, ad_n, mneg, meye16, aid_hi, aid_lo, g_hi, g_lo)
g_ref = g_hi.clone()
torch.npu.synchronize()
kg_new(akkbp_hi, akkbp_lo, ad_n, mneg, meye16, aid_hi, aid_lo, g_hi, g_lo)
torch.npu.synchronize()
print(f"G old-vs-new max diff: {(g_ref - g_hi).abs().max().item():.3e}")

for rnd in range(2):
    t_old = do_bench(lambda: kg_old(akkbp_hi, akkbp_lo, ad_n, mneg, meye16,
                                    aid_hi, aid_lo, g_hi, g_lo))
    t_new = do_bench(lambda: kg_new(akkbp_hi, akkbp_lo, ad_n, mneg, meye16,
                                    aid_hi, aid_lo, g_hi, g_lo))
    print(f"round{rnd}: k_g old={t_old*1000:.1f}us new={t_new*1000:.1f}us "
          f"({(t_new/t_old-1)*100:+.2f}%)")
