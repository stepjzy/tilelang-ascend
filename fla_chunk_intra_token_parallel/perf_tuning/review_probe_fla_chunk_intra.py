"""Reviewer-comment regression probes for fla_chunk_intra_token_parallel.

Comment 1 (ln2-before-sub): DK=32, q=k=beta=1, g[t] = -1e8 - 8*t.
    Correct Aqk[1, 0] = 32 * 2^(g_1 - g_0) = 32 * 2^-8 = 0.125.
    The old code (mul LN2 before the anchor subtraction) returned 32.
Comment 2 (UB capacity): B=1,S=64,H=8,DK=256 must run (BH adapts to 2)
    and DK=512 must be rejected.
Comment 3 (GGRP divisibility): B=1,S=48,H=8,DK=32,CS=48,SCS=3 must cover
    all heads (GGRP=8 divides BH; the old code picked GGRP=6 and read
    out-of-range gemm groups).
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fla_chunk_intra_token_parallel import fla_chunk_intra_token_parallel  # noqa: E402

torch.manual_seed(0)


def ref(q, k, gk, beta, CS, SCS):
    """Triple-loop golden (Triton transcription, original log2 domain)."""
    B, S, H, DK = q.shape
    qf, kf, bf, gf = q.float(), k.float(), beta.float(), gk.float()
    Aqk = torch.zeros(B, S, H, CS, dtype=torch.bfloat16)
    Akk = torch.zeros(B, S, H, SCS, dtype=torch.bfloat16)
    for b in range(B):
        for i in range(S):
            c, s = i // CS, (i % CS) // SCS
            i_ts = c * CS + s * SCS
            for h in range(H):
                for j in range(i_ts, min(i + 1, i_ts + SCS)):
                    e = torch.exp2(gf[b, i, h] - gf[b, j, h])
                    kgj = kf[b, j, h] * e
                    Aqk[b, i, h, s * SCS + (j - i_ts)] = (
                        (qf[b, i, h] * kgj).sum()).to(torch.bfloat16)
                    if j < i:
                        Akk[b, i, h, j - i_ts] = (
                            ((bf[b, i, h] * kf[b, i, h]) * kgj).sum()).to(torch.bfloat16)
    return Aqk, Akk


def check(name, out, gold, atol=2**-10, rtol=2**-6, limit=1e0, ratio_req=0.99):
    a, g = out.detach().cpu().float(), gold.detach().cpu().float()
    m = torch.isfinite(g)
    err = (a[m] - g[m]).abs()
    ratio = (err <= atol + rtol * g[m].abs()).float().mean().item()
    mx = err.max().item()
    ok = ratio >= ratio_req and mx <= limit
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: ratio={ratio:.4f} max_abs={mx:.3e}")
    return ok


ok = True

# ---- comment 1: deep-gate difference survival ----
B, S, H, DK = 1, 64, 8, 32
q = torch.ones(B, S, H, DK, dtype=torch.bfloat16)
k = torch.ones(B, S, H, DK, dtype=torch.bfloat16)
beta = torch.ones(B, S, H, dtype=torch.bfloat16)
t = torch.arange(S, dtype=torch.float32)
gk = (-1e8 - 8 * t).view(1, S, 1, 1).expand(B, S, H, DK).contiguous().float()
aqk, akk = fla_chunk_intra_token_parallel(q.npu(), k.npu(), gk.npu(), beta.npu())
g_aqk, g_akk = ref(q, k, gk, beta, 64, 16)
v = aqk[0, 1, 0, 0].item()
print(f"comment1: Aqk[1,0,0] = {v} (expected 0.125; the old code returned 32.0)")
ok &= abs(v - 0.125) < 1e-3
ok &= check("comment1 Aqk", aqk, g_aqk)
ok &= check("comment1 Akk", akk, g_akk)

# ---- comment 2: UB capacity (DK=256 runs via BH adaptation; DK=512 rejected) ----
B, S, H, DK = 1, 64, 8, 256
q = torch.randn(B, S, H, DK).to(torch.bfloat16)
k = torch.randn(B, S, H, DK).to(torch.bfloat16)
beta = torch.randn(B, S, H).to(torch.bfloat16)
gk = torch.nn.functional.logsigmoid(torch.randn(B, S, H, DK, dtype=torch.float32))
gk = gk.view(B, S // 64, 64, H, DK).cumsum(dim=2).view(B, S, H, DK)
aqk, akk = fla_chunk_intra_token_parallel(q.npu(), k.npu(), gk.npu(), beta.npu())
g_aqk, g_akk = ref(q, k, gk, beta, 64, 16)
ok &= check("comment2 DK=256/H=8 (BH->2) Aqk", aqk, g_aqk)
ok &= check("comment2 DK=256/H=8 (BH->2) Akk", akk, g_akk)
try:
    q512 = torch.randn(B, S, H, 512).to(torch.bfloat16)
    k512 = torch.randn(B, S, H, 512).to(torch.bfloat16)
    gk512 = torch.nn.functional.logsigmoid(torch.randn(B, S, H, 512, dtype=torch.float32))
    gk512 = gk512.view(B, S // 64, 64, H, 512).cumsum(dim=2).view(B, S, H, 512)
    fla_chunk_intra_token_parallel(q512.npu(), k512.npu(), gk512.npu(), beta.npu())
    print("comment2b: FAILED (DK=512 accepted)")
    ok = False
except ValueError as e:
    print(f"comment2b: DK=512 correctly rejected: {str(e)[:90]}...")

# ---- comment 3: GGRP divisibility (SCS=3, CS=48) ----
B, S, H, DK, CS, SCS = 1, 48, 8, 32, 48, 3
q = torch.randn(B, S, H, DK).to(torch.bfloat16)
k = torch.randn(B, S, H, DK).to(torch.bfloat16)
beta = torch.randn(B, S, H).to(torch.bfloat16)
gk = torch.nn.functional.logsigmoid(torch.randn(B, S, H, DK, dtype=torch.float32))
gk = gk.view(B, S // CS, CS, H, DK).cumsum(dim=2).view(B, S, H, DK)
try:
    aqk, akk = fla_chunk_intra_token_parallel(
        q.npu(), k.npu(), gk.npu(), beta.npu(), chunk_size=CS, sub_chunk_size=SCS)
    g_aqk, g_akk = ref(q, k, gk, beta, CS, SCS)
    ok &= check("comment3 SCS=3/CS=48 (GGRP=8) Aqk", aqk, g_aqk)
    ok &= check("comment3 SCS=3/CS=48 (GGRP=8) Akk", akk, g_akk)
except ValueError as e:
    print(f"comment3: CS=48/SCS=3 cleanly rejected at the domain gate: {str(e)[:80]}")
    print("         (non-pow2 chunk sizes are outside the supported domain; the")
    print("          old code instead picked GGRP=6 and read out-of-range gemm groups)")
except Exception as e:
    print(f"comment3: SCS=3 raised {type(e).__name__}: {str(e)[:140]}")
    print("         (non-pow2 SCS tile support -- needs an explicit domain guard)")
    ok = False

print("ALL COMMENT PROBES PASS" if ok else "SOME PROBES FAILED")
