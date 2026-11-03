"""Same-session A/B for the fwd pre-split ws chain (r1 iter3).

Old form: history_version/fla_chunk_intra_perf_r1_iter2_before.py (#1+#2 adopted:
k_inter super-GEMM with fp32 Akkbp out + new bwd)
New form: current fla_chunk_intra.py (#3: all five ws as bf16 hi/lo pairs,
C-section direct reads, I+ folded into exact fp32 epilogue adds)
Bench: do_bench NPU events warmup=20 rep=10 mean; fwd shape (1,8192,64,128).
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
    "intra_old3",
    os.path.join(HERE, "..", "history_version", "fla_chunk_intra_perf_r1_iter2_before.py"),
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

# ---------- old form (#1+#2 state) ----------
mneg = new_mod._get_mneg(torch.device(DEVICE), SCS)
meye16 = new_mod._get_meye(torch.device(DEVICE), SCS)
akkbp_o = torch.empty(B, S, H, CS, dtype=torch.float32, device=DEVICE)
g_o = torch.empty(B, S, H, CS, dtype=torch.float32, device=DEVICE)
g2_o = torch.empty(B, S, H, CS, dtype=torch.float32, device=DEVICE)
y_o = torch.empty(B, S, H, CS, dtype=torch.float32, device=DEVICE)
aid_o = torch.empty(B, S, H, CS, dtype=torch.float32, device=DEVICE)
f_old = old.fla_chunk_intra_fwd_k_inter(B, S, H, DK, CS, SCS, 1.0)
kg_old = old.fla_chunk_intra_fwd_k_g(B, S, H, DK, CS, SCS)
kg2_old = old.fla_chunk_intra_fwd_k_g2(B, S, H, DK, CS, SCS)
ky_old = old.fla_chunk_intra_fwd_k_y(B, S, H, DK, CS, SCS)
kai_old = old.fla_chunk_intra_fwd_k_ai(B, S, H, DK, CS, SCS)


def chain_old():
    Aqk = f_old(q_n, k_n, gk_n, beta_flat, akkbp_o)
    kg_old(akkbp_o, ad_n, mneg, meye16, aid_o, g_o)
    kg2_old(g_o, g2_o)
    ky_old(g2_o, aid_o, new_mod._get_meye(torch.device(DEVICE), CS), y_o)
    return kai_old(g_o, y_o, new_mod._get_meye(torch.device(DEVICE), CS))


# ---------- new form (#3) ----------
pair = lambda: (  # noqa: E731
    torch.empty(B, S, H, CS, dtype=torch.bfloat16, device=DEVICE),
    torch.empty(B, S, H, CS, dtype=torch.bfloat16, device=DEVICE),
)
akkbp_hi, akkbp_lo = pair()
g_hi, g_lo = pair()
g2_hi, g2_lo = pair()
y_hi, y_lo = pair()
aid_hi, aid_lo = pair()
f_new = new_mod.fla_chunk_intra_fwd_k_inter(B, S, H, DK, CS, SCS, 1.0)
kg_new = new_mod.fla_chunk_intra_fwd_k_g(B, S, H, DK, CS, SCS)
kg2_new = new_mod.fla_chunk_intra_fwd_k_g2(B, S, H, DK, CS, SCS)
ky_new = new_mod.fla_chunk_intra_fwd_k_y(B, S, H, DK, CS, SCS)
kai_new = new_mod.fla_chunk_intra_fwd_k_ai(B, S, H, DK, CS, SCS)


def chain_new():
    Aqk = f_new(q_n, k_n, gk_n, beta_flat, akkbp_hi, akkbp_lo)
    kg_new(akkbp_hi, akkbp_lo, ad_n, mneg, meye16, aid_hi, aid_lo, g_hi, g_lo)
    kg2_new(g_hi, g_lo, g2_hi, g2_lo)
    ky_new(g2_hi, g2_lo, aid_hi, aid_lo, y_hi, y_lo)
    return kai_new(g_hi, g_lo, y_hi, y_lo)


# numeric cross-check
r_old = chain_old()
r_new = chain_new()
torch.npu.synchronize()
print(f"Akk old-vs-new max diff: {(r_old - r_new).abs().max().item():.3e}")
print(f"Akk new nan: {torch.isnan(r_new.float()).sum().item()}")

for rnd in range(2):
    names = ("k_inter", "k_g", "k_g2", "k_y", "k_ai")
    olds = [
        do_bench(lambda: f_old(q_n, k_n, gk_n, beta_flat, akkbp_o)),
        do_bench(lambda: kg_old(akkbp_o, ad_n, mneg, meye16, aid_o, g_o)),
        do_bench(lambda: kg2_old(g_o, g2_o)),
        do_bench(lambda: ky_old(g2_o, aid_o, new_mod._get_meye(torch.device(DEVICE), CS), y_o)),
        do_bench(lambda: kai_old(g_o, y_o, new_mod._get_meye(torch.device(DEVICE), CS))),
    ]
    news = [
        do_bench(lambda: f_new(q_n, k_n, gk_n, beta_flat, akkbp_hi, akkbp_lo)),
        do_bench(lambda: kg_new(akkbp_hi, akkbp_lo, ad_n, mneg, meye16, aid_hi, aid_lo, g_hi, g_lo)),
        do_bench(lambda: kg2_new(g_hi, g_lo, g2_hi, g2_lo)),
        do_bench(lambda: ky_new(g2_hi, g2_lo, aid_hi, aid_lo, y_hi, y_lo)),
        do_bench(lambda: kai_new(g_hi, g_lo, y_hi, y_lo)),
    ]
    for nm, o, n in zip(names, olds, news):
        print(f"round{rnd} {nm:9s}: old={o*1000:8.1f} new={n*1000:8.1f} ({(n/o-1)*100:+6.2f}%)")
    print(f"round{rnd} sum: old={sum(olds)*1000:.1f} new={sum(news)*1000:.1f} "
          f"({(sum(news)/sum(olds)-1)*100:+.2f}%)")
    to = do_bench(chain_old)
    tn = do_bench(chain_new)
    print(f"round{rnd} chain: old={to*1000:.1f} new={tn*1000:.1f} ({(tn/to-1)*100:+.2f}%)")
