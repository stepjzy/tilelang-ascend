"""Same-session A/B for fla_chunk_intra k_inter super-GEMM (r1 iter1).

Old form: history_version/fla_chunk_intra_perf_r1_iter0_before.py (snapshot)
New form: current fla_chunk_intra.py
Bench: do_bench NPU events warmup=20 rep=10 mean (device-side only), the
fwd bench shape (1,8192,64,128) bf16. Alternating A/B x2 for drift control.
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
    _get_meye,
    _get_mneg,
    fla_chunk_intra_fwd_k_g,
    fla_chunk_intra_fwd_k_g2,
    fla_chunk_intra_fwd_k_inter as k_inter_new,
    fla_chunk_intra_fwd_k_y,
    fla_chunk_intra_fwd_k_ai,
)
from examples_experiment.dsa_sparse_finetune.test_fla_chunk_intra import (  # noqa: E402
    do_bench,
    prepare_input_fwd,
)

_spec = importlib.util.spec_from_file_location(
    "intra_old",
    os.path.join(HERE, "..", "history_version", "fla_chunk_intra_perf_r1_iter0_before.py"),
)
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)
k_inter_old = _old.fla_chunk_intra_fwd_k_inter

DEVICE = "npu"
B, S, H, DK = 1, 8192, 64, 128
CS, SCS = 64, 16

torch.manual_seed(42)
q, k, gk, beta, ad = prepare_input_fwd(B, S, H, DK)
q_n, k_n, gk_n, beta_n, ad_n = (t.to(DEVICE) for t in (q, k, gk, beta, ad))
beta_flat = beta_n.view(B, S * H)
mneg = _get_mneg(torch.device(DEVICE), SCS)
meye16 = _get_meye(torch.device(DEVICE), SCS)
meye64 = _get_meye(torch.device(DEVICE), CS)
akkbp_a = torch.empty(B, S, H, CS, dtype=torch.float32, device=DEVICE)
akkbp_b = torch.empty(B, S, H, CS, dtype=torch.float32, device=DEVICE)

f_old = k_inter_old(B, S, H, DK, CS, SCS, 1.0)
f_new = k_inter_new(B, S, H, DK, CS, SCS, 1.0)
# downstream kernels (shared, for chain timing)
k_g = fla_chunk_intra_fwd_k_g(B, S, H, DK, CS, SCS)
k_g2 = fla_chunk_intra_fwd_k_g2(B, S, H, DK, CS, SCS)
k_y = fla_chunk_intra_fwd_k_y(B, S, H, DK, CS, SCS)
k_ai = fla_chunk_intra_fwd_k_ai(B, S, H, DK, CS, SCS)
g_ws = torch.empty(B, S, H, CS, dtype=torch.float32, device=DEVICE)
g2_ws = torch.empty(B, S, H, CS, dtype=torch.float32, device=DEVICE)
y_ws = torch.empty(B, S, H, CS, dtype=torch.float32, device=DEVICE)
aid_ws = torch.empty(B, S, H, CS, dtype=torch.float32, device=DEVICE)

# warm + numeric cross-check
f_old(q_n, k_n, gk_n, beta_flat, akkbp_a)
f_new(q_n, k_n, gk_n, beta_flat, akkbp_b)
torch.npu.synchronize()
print(f"Akkbp old-vs-new max diff: {(akkbp_a - akkbp_b).abs().max().item():.3e}")
k_g(akkbp_b, ad_n, mneg, meye16, aid_ws, g_ws)
k_g2(g_ws, g2_ws)
k_y(g2_ws, aid_ws, meye64, y_ws)
k_ai(g_ws, y_ws, meye64)
torch.npu.synchronize()

for rnd in range(2):
    t_old = do_bench(lambda: f_old(q_n, k_n, gk_n, beta_flat, akkbp_a))
    t_new = do_bench(lambda: f_new(q_n, k_n, gk_n, beta_flat, akkbp_b))
    print(f"round{rnd}: k_inter old={t_old*1000:.1f}us new={t_new*1000:.1f}us "
          f"({(t_new/t_old-1)*100:+.2f}%)")

# full chain with new k_inter
def chain():
    f_new(q_n, k_n, gk_n, beta_flat, akkbp_b)
    k_g(akkbp_b, ad_n, mneg, meye16, aid_ws, g_ws)
    k_g2(g_ws, g2_ws)
    k_y(g2_ws, aid_ws, meye64, y_ws)
    return k_ai(g_ws, y_ws, meye64)

t_chain = do_bench(chain)
print(f"fwd chain (new k_inter): {t_chain*1000:.1f}us")
