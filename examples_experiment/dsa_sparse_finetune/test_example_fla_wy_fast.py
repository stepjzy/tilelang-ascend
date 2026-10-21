"""fla_wy_fast example test: representative L0 + L1 cases with PyTorch golden.

fwd: k/v/A/beta bf16 + gk fp32 -> (w, u, kg) bf16
bwd: all fp32 -> (dA, dk2, dv, dbeta, dg2) fp32

Precision: mixed-tolerance dual gate inlined per dtype
(|actual - golden| <= atol + rtol * |golden| element-wise, plus
matched_ratio >= 0.99 and max_abs_error <= limit overall).
"""

import os
import sys

import tilelang
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_fla_wy_fast import fla_wy_fast_bwd, fla_wy_fast_fwd  # noqa: E402


# ============================================================================
# Golden reference implementations (PyTorch chunk-loop)
# ============================================================================
def fla_wy_fast_fwd_ref(k, v, beta, gk, A, use_kg=True):
    """fwd golden. k[B,S,H,DK]/v[B,S,H,DV]/beta[B,S,H] bf16, gk fp32,
    A[B,S,H,C] bf16 -> (w, u, kg) bf16. C = 64."""
    B, S, H, DK = k.shape
    DV = v.shape[-1]
    C = A.shape[-1]
    assert S % C == 0, "S must be divisible by chunk size"
    NT = S // C
    k_f, v_f = k.float(), v.float()
    b_f, g_f, A_f = beta.float(), gk.float(), A.float()
    w = torch.empty(B, S, H, DK, dtype=k.dtype)
    u = torch.empty(B, S, H, DV, dtype=v.dtype)
    kg = torch.empty(B, S, H, DK, dtype=k.dtype) if use_kg else None
    for i in range(NT):
        sl = slice(i * C, (i + 1) * C)
        A_c = A_f[:, sl, :, :]  # [B, C, H, C]
        b_c = b_f[:, sl, :]  # [B, C, H]
        k_c, v_c, g_c = k_f[:, sl], v_f[:, sl], g_f[:, sl]
        # u = A_c @ (beta ⊙ v)
        V_b = v_c * b_c.unsqueeze(-1)
        u[:, sl] = torch.einsum("bmhj,bjhd->bmhd", A_c, V_b).to(v.dtype)
        # w = A_c @ (beta ⊙ k ⊙ 2^g)
        pow2 = torch.exp2(g_c)  # [B, C, H, DK]
        K_bg = k_c * b_c.unsqueeze(-1) * pow2
        w[:, sl] = torch.einsum("bmhj,bjhd->bmhd", A_c, K_bg).to(k.dtype)
        if use_kg:
            g_n = g_f[:, (i + 1) * C - 1, :, :]  # chunk last row [B, H, DK]
            kg[:, sl] = (k_c * torch.exp2(g_n.unsqueeze(1) - g_c)).to(k.dtype)
    return w, u, kg


def fla_wy_fast_bwd_ref(k, v, beta, gk, A, dw, du, dk, dg):
    """bwd golden. all fp32 -> (dA, dk2, dv, dbeta, dg2) all fp32."""
    B, S, H, DK = k.shape
    DV = v.shape[-1]
    C = A.shape[-1]
    assert S % C == 0
    NT = S // C
    k_f, v_f = k.float(), v.float()
    b_f, g_f, A_f = beta.float(), gk.float(), A.float()
    dw_f, du_f = dw.float(), du.float()
    dk_f, dg_f = dk.float(), dg.float()
    dA = torch.empty(B, S, H, C)
    dk2 = torch.empty(B, S, H, DK)
    dv = torch.empty(B, S, H, DV)
    dbeta = torch.empty(B, S, H)
    dg2 = torch.empty(B, S, H, DK)
    for i in range(NT):
        sl = slice(i * C, (i + 1) * C)
        A_c = A_f[:, sl]  # [B, C, H, C]
        b_c = b_f[:, sl]  # [B, C, H]
        k_c, v_c, g_c = k_f[:, sl], v_f[:, sl], g_f[:, sl]
        dw_c, du_c = dw_f[:, sl], du_f[:, sl]
        dk_c, dg_c = dk_f[:, sl], dg_f[:, sl]
        pow2 = torch.exp2(g_c)
        K_bg = k_c * b_c.unsqueeze(-1) * pow2
        V_b = v_c * b_c.unsqueeze(-1)
        # S = strict_tril(dw @ K_bg^T + du @ V_b^T); tril over the chunk dims
        S = torch.einsum("bmhd,bjhd->bmhj", dw_c, K_bg) + torch.einsum("bmhd,bjhd->bmhj", du_c, V_b)
        S = torch.tril(S.permute(0, 2, 1, 3), -1).permute(0, 2, 1, 3)
        # X = A_c^T @ dw ; Y = A_c^T @ du
        X = torch.einsum("bjhi,bjhd->bihd", A_c, dw_c)
        Y = torch.einsum("bjhi,bjhd->bihd", A_c, du_c)
        dk2[:, sl] = X * b_c.unsqueeze(-1) * pow2 + dk_c
        dv[:, sl] = Y * b_c.unsqueeze(-1)
        dbeta[:, sl] = (X * k_c * pow2).sum(-1) + (Y * v_c).sum(-1)
        dg2[:, sl] = K_bg * X + dg_c
        # dA = -strict_tril(A_c^T @ S @ A_c^T)
        T1 = torch.einsum("bjhi,bjhm->bihm", A_c, S)  # A_c^T @ S
        T2 = torch.einsum("bihm,bnhm->bihn", T1, A_c)  # @ A_c^T
        dA[:, sl] = -torch.tril(T2.permute(0, 2, 1, 3), -1).permute(0, 2, 1, 3)
    return dA, dk2, dv, dbeta, dg2


# ============================================================================
# Input construction (CPU randn + single H2D)
# ============================================================================
def make_fwd_inputs(B, S, H, DK, DV, chunk, seed):
    gen = torch.Generator().manual_seed(seed)
    k = torch.randn(B, S, H, DK, generator=gen).to(torch.bfloat16)
    v = torch.randn(B, S, H, DV, generator=gen).to(torch.bfloat16)
    beta = torch.randn(B, S, H, generator=gen).to(torch.bfloat16)
    gk = torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32)
    A = torch.randn(B, S, H, chunk, generator=gen).to(torch.bfloat16)
    cpu = (k, v, beta, gk, A)
    npu = tuple(t.npu() for t in cpu)
    return npu, cpu


def make_bwd_inputs(B, S, H, DK, DV, chunk, seed):
    gen = torch.Generator().manual_seed(seed)
    k = torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32)
    v = torch.randn(B, S, H, DV, generator=gen, dtype=torch.float32)
    beta = torch.randn(B, S, H, generator=gen, dtype=torch.float32)
    gk = torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32)
    A = torch.randn(B, S, H, chunk, generator=gen, dtype=torch.float32)
    dw = torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32)
    du = torch.randn(B, S, H, DV, generator=gen, dtype=torch.float32)
    dk = torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32)
    dg = torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32)
    cpu = (k, v, beta, gk, A, dw, du, dk, dg)
    npu = tuple(t.npu() for t in cpu)
    return npu, cpu


def check_output(name, out, ref, atol, rtol, max_abs_limit, required_ratio):
    """Inlined mixed-tolerance dual gate (element-wise |a-g| <= atol+rtol*|g|,
    overall matched_ratio >= required and max_abs <= limit)."""
    y_cpu = out.detach().cpu().float()
    ref_cpu = ref.detach().cpu().float()
    m = torch.isfinite(ref_cpu)
    abs_err = (y_cpu[m] - ref_cpu[m]).abs()
    ratio = (abs_err <= (atol + rtol * ref_cpu[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    assert ratio >= required_ratio and max_abs <= max_abs_limit, (
        f"{name} precision fail: ratio={ratio:.4f} max_abs={max_abs:.3e}"
    )
    print(f"Test pass! {name}: matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    torch.manual_seed(0)

    # Representative configs: L0 minimal gates + L1 benchmark shapes
    test_configs = [
        ("l0_fwd_min", "fwd", (1, 128, 2, 64, 64), 1001, "L0"),
        ("l0_bwd_min", "bwd", (1, 128, 2, 64, 64), 1003, "L0"),
        ("l1_fwd_bench", "fwd", (1, 8192, 64, 128, 128), 2101, "L1"),
        ("l1_bwd_bench", "bwd", (1, 32768, 8, 128, 128), 2103, "L1"),
    ]

    for name, kind, shape, seed, level in test_configs:
        B, S, H, DK, DV = shape
        print(f"Testing fla_wy_fast {kind} {level} with shape={shape}")
        if kind == "fwd":
            npu_in, cpu_in = make_fwd_inputs(B, S, H, DK, DV, 64, seed)
            w, u, kg = fla_wy_fast_fwd(*npu_in)
            torch.npu.synchronize()
            print("Init successful!")
            gw, gu, gkg = fla_wy_fast_fwd_ref(*cpu_in)
            # bf16 mixed tolerance
            atol, rtol, max_abs_limit, required_ratio = 2**-10, 2**-6, 1e0, 0.99
            check_output("w", w, gw, atol, rtol, max_abs_limit, required_ratio)
            check_output("u", u, gu, atol, rtol, max_abs_limit, required_ratio)
            check_output("kg", kg, gkg, atol, rtol, max_abs_limit, required_ratio)
        else:
            npu_in, cpu_in = make_bwd_inputs(B, S, H, DK, DV, 64, seed)
            dA, dk2, dv, dbeta, dg2 = fla_wy_fast_bwd(*npu_in)
            torch.npu.synchronize()
            print("Init successful!")
            gdA, gdk2, gdv, gdbeta, gdg2 = fla_wy_fast_bwd_ref(*cpu_in)
            # fp32 mixed tolerance
            atol, rtol, max_abs_limit, required_ratio = 2**-16, 2**-10, 1e-2, 0.99
            check_output("dA", dA, gdA, atol, rtol, max_abs_limit, required_ratio)
            check_output("dk2", dk2, gdk2, atol, rtol, max_abs_limit, required_ratio)
            check_output("dv", dv, gdv, atol, rtol, max_abs_limit, required_ratio)
            check_output("dbeta", dbeta, gdbeta, atol, rtol, max_abs_limit, required_ratio)
            check_output("dg2", dg2, gdg2, atol, rtol, max_abs_limit, required_ratio)

    print("Kernel Output Match!")
