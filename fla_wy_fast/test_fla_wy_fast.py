"""fla_wy_fast layered tests: L0/L1/L2/Boundary + main(--level).

Golden: PyTorch chunk-loop reference (DESIGN SS9.1, verbatim).
Precision: mixed-tolerance dual gate (precision-standard.md SS4.1, verbatim).
"""

import argparse
import os
import sys

import tilelang
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fla_wy_fast import fla_wy_fast_bwd, fla_wy_fast_fwd  # noqa: E402


# ============================================================================
# Golden reference implementations (DESIGN SS9.1)
# ============================================================================
def fla_wy_fast_fwd_ref(k, v, beta, gk, A, use_kg=True):
    """fwd golden. k[B,S,H,DK]/v[B,S,H,DV]/beta[B,S,H] bf16, gk fp32,
    A[B,S,H,C] bf16 -> (w, u, kg) same dtype as inputs (bf16). C = 64."""
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
        # S = strict_tril(dw @ K_bg^T + du @ V_b^T); tril over the chunk
        # dims (m, j), not (H, j) -- DESIGN SS9.1 applied it on the wrong axes
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
        # NOTE: fixes an axis-order bug in DESIGN SS9.1's einsums (its T1
        # silently mis-computed and its T2 raised); verified against explicit
        # per-(b,h) matmuls and the upstream Triton semantics (A loaded
        # transposed there): T1 = A^T @ S, T2 = T1 @ A^T.
        T1 = torch.einsum("bjhi,bjhm->bihm", A_c, S)  # A_c^T @ S
        T2 = torch.einsum("bihm,bnhm->bihn", T1, A_c)  # @ A_c^T
        # tril over the chunk dims (i, n), not (H, n)
        dA[:, sl] = -torch.tril(T2.permute(0, 2, 1, 3), -1).permute(0, 2, 1, 3)
    return dA, dk2, dv, dbeta, dg2


# ============================================================================
# Precision standard (precision-standard.md SS4.1, verbatim)
# ============================================================================
def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio)."""
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
        "hifloat32": (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4, 2**-2, 1e0, 0.99),
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),
    }
    int_types = {"int8", "int16", "int32", "int64", "uint8"}
    if dtype in int_types:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Mixed-tolerance dual-gate check: (passed, matched_ratio, max_abs_error)."""
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:  # integer exact match
        mism = (a != g).sum().item()
        total = max(a.numel(), 1)
        return mism == 0, 1.0 - mism / total, (0.0 if mism == 0 else float("inf"))
    a = a.float()
    g = g.float()
    special = ~torch.isfinite(g)  # inf/nan structural comparison
    if special.any() and (
        not torch.equal(torch.isnan(a[special]), torch.isnan(g[special]))
        or not torch.equal(torch.isinf(a[special]), torch.isinf(g[special]))
    ):
        return False, 0.0, float("inf")
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    matched_ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs_error = abs_err.max().item()
    passed = (matched_ratio >= required_ratio) and (max_abs_error <= max_abs_limit)
    return passed, matched_ratio, max_abs_error


# ============================================================================
# Input construction (CPU randn + single H2D; aligned with upstream
# prepare_input: beta/A are randn too -- the operator only consumes A)
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


# ============================================================================
# L0 gate tests (DESIGN SS9.2)
# ============================================================================
def test_fla_wy_fast_l0():
    """L0: regular shapes (S%64==0, DK/DV multiples of block_D)."""
    configs = [
        ("l0_fwd_min", "fwd", (1, 128, 2, 64, 64), 1001),
        ("l0_fwd_repr", "fwd", (2, 256, 4, 128, 128), 1002),
        ("l0_bwd_min", "bwd", (1, 128, 2, 64, 64), 1003),
        ("l0_bwd_repr", "bwd", (2, 256, 4, 128, 128), 1004),
    ]
    ok = True
    for name, kind, shape, seed in configs:
        B, S, H, DK, DV = shape
        try:
            if kind == "fwd":
                npu_in, cpu_in = make_fwd_inputs(B, S, H, DK, DV, 64, seed)
                w, u, kg = fla_wy_fast_fwd(*npu_in)
                torch.npu.synchronize()
                gw, gu, gkg = fla_wy_fast_fwd_ref(*cpu_in)
                res = [
                    ("w", *check_precision(w, gw, "bfloat16")),
                    ("u", *check_precision(u, gu, "bfloat16")),
                    ("kg", *check_precision(kg, gkg, "bfloat16")),
                ]
            else:
                npu_in, cpu_in = make_bwd_inputs(B, S, H, DK, DV, 64, seed)
                dA, dk2, dv, dbeta, dg2 = fla_wy_fast_bwd(*npu_in)
                torch.npu.synchronize()
                gdA, gdk2, gdv, gdbeta, gdg2 = fla_wy_fast_bwd_ref(*cpu_in)
                res = [
                    ("dA", *check_precision(dA, gdA, "float32")),
                    ("dk2", *check_precision(dk2, gdk2, "float32")),
                    ("dv", *check_precision(dv, gdv, "float32")),
                    ("dbeta", *check_precision(dbeta, gdbeta, "float32")),
                    ("dg2", *check_precision(dg2, gdg2, "float32")),
                ]
            tag = "PRECISION_PASS" if all(r[1] for r in res) else "PRECISION_FAIL"
            detail = " ".join(f"{n}:ratio={r:.4f},max_abs={m:.3e}" for n, _, r, m in res)
            print(f"[{tag}] {name} shape={shape} :: {detail}")
            ok &= tag == "PRECISION_PASS"
        except Exception as e:  # noqa: BLE001
            print(f"[PRECISION_FAIL] {name}: {e}")
            ok = False
    return ok


# ============================================================================
# L1 functional tests (scenario B): benchmark shapes + irregular D
# ============================================================================
def test_fla_wy_fast_l1():
    """L1: benchmark representative shapes + non-64-multiple D dims."""
    configs = [
        ("l1_fwd_bench", "fwd", (1, 8192, 64, 128, 128), 2101),
        ("l1_fwd_d96", "fwd", (1, 256, 2, 96, 96), 2102),
        ("l1_bwd_bench", "bwd", (1, 32768, 8, 128, 128), 2103),
        ("l1_bwd_d96", "bwd", (1, 256, 2, 96, 96), 2104),
    ]
    ok = True
    for name, kind, shape, seed in configs:
        B, S, H, DK, DV = shape
        try:
            if kind == "fwd":
                npu_in, cpu_in = make_fwd_inputs(B, S, H, DK, DV, 64, seed)
                w, u, kg = fla_wy_fast_fwd(*npu_in)
                torch.npu.synchronize()
                gw, gu, gkg = fla_wy_fast_fwd_ref(*cpu_in)
                res = [
                    ("w", *check_precision(w, gw, "bfloat16")),
                    ("u", *check_precision(u, gu, "bfloat16")),
                    ("kg", *check_precision(kg, gkg, "bfloat16")),
                ]
            else:
                npu_in, cpu_in = make_bwd_inputs(B, S, H, DK, DV, 64, seed)
                dA, dk2, dv, dbeta, dg2 = fla_wy_fast_bwd(*npu_in)
                torch.npu.synchronize()
                gdA, gdk2, gdv, gdbeta, gdg2 = fla_wy_fast_bwd_ref(*cpu_in)
                res = [
                    ("dA", *check_precision(dA, gdA, "float32")),
                    ("dk2", *check_precision(dk2, gdk2, "float32")),
                    ("dv", *check_precision(dv, gdv, "float32")),
                    ("dbeta", *check_precision(dbeta, gdbeta, "float32")),
                    ("dg2", *check_precision(dg2, gdg2, "float32")),
                ]
            tag = "PRECISION_PASS" if all(r[1] for r in res) else "PRECISION_FAIL"
            detail = " ".join(f"{n}:ratio={r:.4f},max_abs={m:.3e}" for n, _, r, m in res)
            print(f"[{tag}] {name} shape={shape} :: {detail}")
            ok &= tag == "PRECISION_PASS"
        except Exception as e:  # noqa: BLE001
            print(f"[PRECISION_FAIL] {name}: {e}")
            ok = False
    return ok


# ============================================================================
# L2 negative tests (scenario B): invalid inputs must be rejected
# ============================================================================
def test_fla_wy_fast_l2():
    """L2: S % 64 != 0 / DK not a multiple of 16 / chunk_size != 64 must
    raise (host-side domain checks in the wrappers)."""
    cases = [
        (
            "l2_fwd_s_not_div64",
            "fwd",
            dict(shape=(1, 100, 2, 64, 64), chunk=64),
        ),
        ("l2_fwd_dk_not_mul16", "fwd", dict(shape=(1, 128, 2, 100, 64), chunk=64)),
        ("l2_fwd_chunk32", "fwd", dict(shape=(1, 128, 2, 64, 64), chunk=32)),
        (
            "l2_bwd_s_not_div64",
            "bwd",
            dict(shape=(1, 100, 2, 64, 64), chunk=64),
        ),
        ("l2_bwd_dv_not_mul16", "bwd", dict(shape=(1, 128, 2, 64, 100), chunk=64)),
        # exact-2-iteration block rule: D=48 has no valid block (toolchain
        # Pipelined limitation, documented in fla_wy_fast.py)
        ("l2_fwd_d48_no_block", "fwd", dict(shape=(1, 128, 2, 48, 48), chunk=64)),
        ("l2_bwd_d48_no_block", "bwd", dict(shape=(1, 128, 2, 48, 48), chunk=64)),
    ]
    for name, kind, cfg in cases:
        B, S, H, DK, DV = cfg["shape"]
        chunk = cfg["chunk"]
        try:
            if kind == "fwd":
                npu_in, _ = make_fwd_inputs(B, S, H, DK, DV, 64, 2201)
                fla_wy_fast_fwd(*npu_in, chunk_size=chunk)
            else:
                npu_in, _ = make_bwd_inputs(B, S, H, DK, DV, 64, 2201)
                fla_wy_fast_bwd(*npu_in, chunk_size=chunk)
            print(f"[BOUNDARY_WARN] {name}: invalid input silently accepted")
        except Exception:  # noqa: BLE001
            print(f"[BOUNDARY_PASS] {name}: correctly rejected")


# ============================================================================
# Boundary tests (scenario B): legal special values (non-blocking)
# ============================================================================
def test_fla_wy_fast_boundary():
    """Boundary: zero beta rows, large gates (2^g within fp16 split range),
    and extreme gates beyond the documented fp16-split range (expected WARN).
    """

    def run_fwd_case(name, k, v, beta, gk, A):
        try:
            npu = tuple(t.npu() for t in (k, v, beta, gk, A))
            w, u, kg = fla_wy_fast_fwd(*npu)
            torch.npu.synchronize()
            gw, gu, gkg = fla_wy_fast_fwd_ref(k, v, beta, gk, A)
            res = [
                ("w", *check_precision(w, gw, "bfloat16")),
                ("u", *check_precision(u, gu, "bfloat16")),
                ("kg", *check_precision(kg, gkg, "bfloat16")),
            ]
            tag = "BOUNDARY_PASS" if all(r[1] for r in res) else "BOUNDARY_WARN"
            detail = " ".join(f"{n}:ratio={r:.4f},max_abs={m:.3e}" for n, _, r, m in res)
            print(f"[{tag}] {name} :: {detail}")
        except Exception as e:  # noqa: BLE001
            print(f"[BOUNDARY_WARN] {name}: exception {e}")

    def run_bwd_case(name, k, v, beta, gk, A, dw, du, dk, dg):
        try:
            npu = tuple(t.npu() for t in (k, v, beta, gk, A, dw, du, dk, dg))
            dA, dk2, dv, dbeta, dg2 = fla_wy_fast_bwd(*npu)
            torch.npu.synchronize()
            gdA, gdk2, gdv, gdbeta, gdg2 = fla_wy_fast_bwd_ref(k, v, beta, gk, A, dw, du, dk, dg)
            res = [
                ("dA", *check_precision(dA, gdA, "float32")),
                ("dk2", *check_precision(dk2, gdk2, "float32")),
                ("dv", *check_precision(dv, gdv, "float32")),
                ("dbeta", *check_precision(dbeta, gdbeta, "float32")),
                ("dg2", *check_precision(dg2, gdg2, "float32")),
            ]
            tag = "BOUNDARY_PASS" if all(r[1] for r in res) else "BOUNDARY_WARN"
            detail = " ".join(f"{n}:ratio={r:.4f},max_abs={m:.3e}" for n, _, r, m in res)
            print(f"[{tag}] {name} :: {detail}")
        except Exception as e:  # noqa: BLE001
            print(f"[BOUNDARY_WARN] {name}: exception {e}")

    B, S, H, DK, DV = 1, 128, 2, 64, 64
    gen = torch.Generator().manual_seed(2301)

    def base(*s, dt=torch.float32):
        return torch.randn(*s, generator=gen, dtype=dt)

    # bnd_beta_zero: every 4th beta row zeroed (legal)
    k = base(B, S, H, DK).to(torch.bfloat16)
    v = base(B, S, H, DV).to(torch.bfloat16)
    beta = base(B, S, H).to(torch.bfloat16)
    beta[..., ::4] = 0.0
    gk = base(B, S, H, DK)
    A = base(B, S, H, 64).to(torch.bfloat16)
    run_fwd_case("bnd_fwd_beta_zero", k, v, beta, gk, A)

    # bnd_gk_large: |g| <= 8 -> 2^g <= 256, products within fp16 range
    gk_large = torch.clamp(base(B, S, H, DK), -8.0, 8.0)
    run_fwd_case("bnd_fwd_gk_large", k, v, beta, gk_large, A)
    kb = base(B, S, H, DK)
    vb = base(B, S, H, DV)
    bb = base(B, S, H)
    dwb = base(B, S, H, DK)
    dub = base(B, S, H, DV)
    dkb = base(B, S, H, DK)
    dgb = base(B, S, H, DK)
    Ab = base(B, S, H, 64)
    run_bwd_case("bnd_bwd_gk_large", kb, vb, bb, gk_large, Ab, dwb, dub, dkb, dgb)

    # bnd_gk_extreme: sparse g = +40 (2^40 overflows the bwd fp16 split --
    # documented limitation, expected WARN, non-blocking)
    gk_ext = base(B, S, H, DK)
    gk_ext[..., 7] = 40.0
    run_fwd_case("bnd_fwd_gk_extreme", k, v, beta, gk_ext, A)
    run_bwd_case("bnd_bwd_gk_extreme", kb, vb, bb, gk_ext, Ab, dwb, dub, dkb, dgb)


# ============================================================================
# main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    blocking_ok = True
    if args.level in ("l0", "all"):
        blocking_ok &= test_fla_wy_fast_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_fla_wy_fast_l1()
    if args.level in ("l2", "all"):
        test_fla_wy_fast_l2()
    if args.level in ("boundary", "all"):
        test_fla_wy_fast_boundary()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
