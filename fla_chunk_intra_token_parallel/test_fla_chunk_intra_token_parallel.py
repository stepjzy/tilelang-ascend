"""fla_chunk_intra_token_parallel layered tests: L0 + main(--level).

Golden: PyTorch per-subchunk vectorized reference (DESIGN SS9.1 draft, with
one bug fixed -- see fla_chunk_intra_token_parallel_ref docstring) plus a
triple-loop naive transcription of the frozen Triton authority
(ref_gpu/fla_chunk_intra_token_parallel.py) used to cross-validate the
vectorized golden on a tiny shape (DESIGN SS9.1 cross-check suggestion).
Inputs are built exactly like the upstream prepare_input
(ref_gpu/fla_chunk_intra_cumsum.py): gk = logsigmoid(randn fp32) followed by
a chunk-local cumsum; everything is generated on CPU and moved to the NPU
with a single H2D (no aclnn calls in the test path).
Precision: mixed-tolerance dual gate (precision-standard.md SS4.1, verbatim
check_precision; outputs are bfloat16: atol 2^-10, rtol 2^-6,
max_abs_error_limit 1e0, matched_ratio >= 0.99).

L0 cases come verbatim from DESIGN SS9.2 (l0_min / l0_typical /
l0_bench_repr). L1/L2/Boundary are filled per tilelang-op-test-design
scenario B (coverage tags + COVERAGE_MANIFEST; see the scenario B section
for the contract-fixed-domain mapping notes). The perf_bench sentinel
(DESIGN SS9.4, shape (1, 8192, 64, 128)) is intentionally NOT run here:
performance is Stage 3 scope for this migration round.
"""

import argparse
import os
import sys

import tilelang
import torch
import torch_npu  # noqa: F401  (registers .npu() on torch)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fla_chunk_intra_token_parallel import fla_chunk_intra_token_parallel  # noqa: E402


# ============================================================================
# Input construction (upstream prepare_input semantics, all on CPU)
# ============================================================================
def prepare_input_kda(B, S, H, DK, chunk_size=64, input_dtype=torch.bfloat16, gate_dtype=torch.float32):
    """CPU construction: q/k/beta randn bf16; gk = logsigmoid(randn fp32)
    followed by a chunk-local cumsum (aligns ref_gpu/fla_chunk_intra_cumsum.py)."""
    q = torch.randn(B, S, H, DK, dtype=input_dtype)
    k = torch.randn(B, S, H, DK, dtype=input_dtype)
    beta = torch.randn(B, S, H, dtype=input_dtype)
    gk = torch.nn.functional.logsigmoid(torch.randn(B, S, H, DK, dtype=gate_dtype))
    gk = gk.view(B, S // chunk_size, chunk_size, H, DK).cumsum(dim=2).view(B, S, H, DK)
    return q, k, gk, beta


# ============================================================================
# Golden reference (DESIGN SS9.1, fp32 domain, torch.exp2 exact semantics)
# ============================================================================
def fla_chunk_intra_token_parallel_ref(q, k, gk, beta, chunk_size=64, sub_chunk_size=16, scale=1.0):
    """Vectorized golden. q/k [B,S,H,DK] bf16, gk [B,S,H,DK] fp32,
    beta [B,S,H] bf16 -> (Aqk [B,S,H,CS] bf16, Akk [B,S,H,SCS] bf16).
    Cleared-row semantics: uncovered Aqk columns 0, Akk diagonal 0.

    NOTE (fix vs the DESIGN SS9.1 draft): kgj must broadcast k over the j
    axis, i.e. ``k_sub.unsqueeze(1) * e``. The draft's
    ``k_sub.unsqueeze(2) * e`` aligns k's token axis with e's i axis and
    computes q_i * k_i * 2^(g_i - g_j), contradicting the frozen Triton
    authority (kgj = k_j * exp2(g_i - g_j)). Cross-checked against the
    naive transcription below (see _golden_self_check).
    """
    B, S, H, DK = q.shape
    CS, SCS = chunk_size, sub_chunk_size
    assert S % CS == 0 and CS % SCS == 0
    qf, kf, bf, gf = q.float(), k.float(), beta.float(), gk.float()
    Aqk = torch.zeros(B, S, H, CS, dtype=torch.bfloat16)
    Akk = torch.zeros(B, S, H, SCS, dtype=torch.bfloat16)
    for c in range(S // CS):
        for s in range(CS // SCS):
            i_ts = c * CS + s * SCS
            sl = slice(i_ts, i_ts + SCS)
            g_sub, k_sub = gf[:, sl], kf[:, sl]  # [B,SCS,H,DK]
            e = torch.exp2(g_sub.unsqueeze(2) - g_sub.unsqueeze(1))  # [B,SCS(i),SCS(j),H,DK]
            kgj = k_sub.unsqueeze(1) * e  # k_j * 2^(g_i - g_j)
            aqk = torch.einsum("bihd,bijhd->bijh", qf[:, sl], kgj) * scale
            akk = torch.einsum("bihd,bijhd->bijh", bf[:, sl].unsqueeze(-1) * k_sub, kgj)
            # masks broadcast over (i, j) = dims 1, 2 of the [B,SCS(i),SCS(j),H] tiles
            keep = torch.tril(torch.ones(SCS, SCS, dtype=torch.bool))[None, :, :, None]  # j <= i
            keep_strict = torch.tril(torch.ones(SCS, SCS, dtype=torch.bool), -1)[None, :, :, None]  # j < i
            aqk = aqk.masked_fill(~keep, 0.0)
            akk = akk.masked_fill(~keep_strict, 0.0)
            Aqk[:, sl, :, s * SCS : s * SCS + SCS] = aqk.permute(0, 1, 3, 2).to(torch.bfloat16)
            Akk[:, sl, :, :] = akk.permute(0, 1, 3, 2).to(torch.bfloat16)
    return Aqk, Akk


def fla_chunk_intra_token_parallel_naive(q, k, gk, beta, chunk_size=64, sub_chunk_size=16, scale=1.0):
    """Triple-loop transcription of the frozen Triton authority
    (chunk_kda_fwd_kernel_intra_token_parallel). Slow; tiny shapes only."""
    B, S, H, DK = q.shape
    CS, SCS = chunk_size, sub_chunk_size
    qf, kf, bf, gf = q.float(), k.float(), beta.float(), gk.float()
    Aqk = torch.zeros(B, S, H, CS, dtype=torch.bfloat16)
    Akk = torch.zeros(B, S, H, SCS, dtype=torch.bfloat16)
    for b in range(B):
        for i in range(S):
            c = i // CS
            s = (i % CS) // SCS
            i_ts = c * CS + s * SCS
            for h in range(H):
                for j in range(i_ts, min(i + 1, i_ts + SCS)):
                    jl = j - i_ts
                    e = torch.exp2(gf[b, i, h] - gf[b, j, h])  # [DK]
                    kgj = kf[b, j, h] * e
                    aqk_v = (qf[b, i, h] * kgj).sum() * scale
                    akk_v = torch.tensor(0.0)
                    if j < i:
                        akk_v = ((bf[b, i, h] * kf[b, i, h]) * kgj).sum()
                    Aqk[b, i, h, s * SCS + jl] = aqk_v.to(torch.bfloat16)
                    if j < i:
                        Akk[b, i, h, jl] = akk_v.to(torch.bfloat16)
    return Aqk, Akk


def _golden_self_check():
    """Cross-validate the vectorized golden against the naive Triton
    transcription on a tiny shape (DESIGN SS9.1 suggestion; CPU only)."""
    torch.manual_seed(1234)
    q, k, gk, beta = prepare_input_kda(1, 64, 4, 32)
    va, vk = fla_chunk_intra_token_parallel_ref(q, k, gk, beta)
    na, nk = fla_chunk_intra_token_parallel_naive(q, k, gk, beta)
    ok = torch.allclose(va.float(), na.float(), rtol=1e-4, atol=1e-4) and torch.allclose(vk.float(), nk.float(), rtol=1e-4, atol=1e-4)
    return ok


# ============================================================================
# Precision standard (precision-standard.md SS4.1, verbatim)
# ============================================================================
def get_precision(dtype):
    """返回 (atol, rtol, max_abs_error_limit, required_matched_ratio)。
    浮点：混合容差；整型：精确匹配（0 误差）。"""
    fp_table = {
        # dtype       : (atol,   rtol,   max_abs_error_limit, required_matched_ratio)
        "float16": (2**-14, 2**-9, 1e-1, 0.99),  # atol 6.10e-5, rtol 1.95e-3
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),  # atol 9.77e-4, rtol 1.56e-2
        "float32": (2**-16, 2**-10, 1e-2, 0.99),  # atol 1.53e-5, rtol 9.77e-4
        "hifloat32": (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4, 2**-2, 1e0, 0.99),  # atol 0.0625, rtol 0.25
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),  # atol 0.125,  rtol 0.5
    }
    int_types = {"int8", "int16", "int32", "int64", "uint8"}
    if dtype in int_types:
        return (0.0, 0.0, 0.0, 1.0)  # 整型：精确匹配，一个元素不符即 FAIL
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """精度判定：返回 (passed, matched_ratio, max_abs_error)。
    浮点双门限：matched_ratio ≥ required 且 max_abs_error ≤ max_abs_error_limit；
    整型：逐元素精确相等。inf/nan 位置做结构比对，不计入数值容差。"""
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:  # 整型精确匹配
        mism = (a != g).sum().item()
        total = max(a.numel(), 1)
        return mism == 0, 1.0 - mism / total, (0.0 if mism == 0 else float("inf"))
    a = a.float()
    g = g.float()
    special = ~torch.isfinite(g)  # inf/nan 位置结构比对
    if special.any() and (
        not torch.equal(torch.isnan(a[special]), torch.isnan(g[special]))
        or not torch.equal(torch.isinf(a[special]), torch.isinf(g[special]))
    ):
        return False, 0.0, float("inf")
    m = torch.isfinite(g)  # golden 有限值位置全比
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()  # actual inf/nan -> 不达标
    matched_ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs_error = abs_err.max().item()
    passed = (matched_ratio >= required_ratio) and (max_abs_error <= max_abs_limit)
    return passed, matched_ratio, max_abs_error


# ============================================================================
# L0: threshold tests (DESIGN SS9.2 plan, verbatim shapes)
# ============================================================================
L0_CASES = [
    # (name, B, S, H, DK) -- DESIGN SS9.2 L0 门槛测试计划
    ("l0_min", 1, 64, 8, 128),  # 1 chunk / 4 subchunks / 1 head group (grid=4)
    ("l0_typical", 2, 256, 16, 128),  # multi batch / 4 chunks / 2 head groups (grid=64)
    ("l0_bench_repr", 1, 1024, 64, 128),  # benchmark structure (H=64, grid=512)
]


def _run_l0_case(name, B, S, H, DK):
    """Run one L0 case end to end; returns (passed, detail string)."""
    q, k, gk, beta = prepare_input_kda(B, S, H, DK)  # CPU
    aqk_ref, akk_ref = fla_chunk_intra_token_parallel_ref(q, k, gk, beta)  # CPU golden
    q_d, k_d, gk_d, beta_d = q.npu(), k.npu(), gk.npu(), beta.npu()  # single H2D each
    aqk_out, akk_out = fla_chunk_intra_token_parallel(q_d, k_d, gk_d, beta_d)
    p1, r1, m1 = check_precision(aqk_out, aqk_ref, "bfloat16")
    p2, r2, m2 = check_precision(akk_out, akk_ref, "bfloat16")
    detail = f"Aqk ratio={r1:.4f} max_abs={m1:.3e} | Akk ratio={r2:.4f} max_abs={m2:.3e}"
    return p1 and p2, detail


def test_fla_chunk_intra_token_parallel_l0():
    """L0 threshold tests: regular shapes (block-divisible), precision gate."""
    ok = True
    if not _golden_self_check():
        print("[PRECISION_FAIL] l0/golden_self_check vectorized golden != naive Triton transcription (golden bug, not a kernel failure)")
        ok = False
    else:
        print("[PRECISION_PASS] l0/golden_self_check vectorized == naive (tiny shape)")
    for name, B, S, H, DK in L0_CASES:
        try:
            passed, detail = _run_l0_case(name, B, S, H, DK)
            tag = "PRECISION_PASS" if passed else "PRECISION_FAIL"
            print(f"[{tag}] l0/{name} shape=({B},{S},{H},{DK}) {detail}")
            ok &= passed
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"[PRECISION_FAIL] l0/{name} shape=({B},{S},{H},{DK}) error: {exc}")
            ok = False
    return ok


# ============================================================================
# L1/L2/Boundary (scenario B, tiled per tilelang-op-test-design)
#
# Coverage mapping notes (contract-fixed domain, Fusion category = strictest):
# - The operator contract REQUIRES S % 64 == 0 (CS=64), CS % SCS == 0,
#   H % 2 == 0 (BH adapts 8 -> 4 -> 2), DK % 16 == 0 and DK >= 32: in-domain
#   compute has no tail blocks on S/DK by construction.
# - D-SHAPE-TAIL-* / PRIME are therefore covered BOTH by in-domain
#   non-canonical stress (H=6 -> BH=2 head-group adaptation, DK=48 non-pow2
#   width, H=2 prime head count, degenerate minimal shape) AND by the
#   contract-mandated rejection of out-of-domain S remainders (L2, tagged).
# - D-PARAM-chunk_size: chunk_size is a wrapper param; covered by an explicit
#   pass on an L1 case plus the CS%SCS rejection (l2_chunk_invalid).
#   D-PARAM-sub_chunk_size: genuinely non-default SCS=8 L1 case (generic
#   kernel path; SCS=32 would blow the UB budget, SCS must divide CS).
#   D-PARAM-scale: contractually fixed at 1.0 (proto: "kernel 不透传") -- the
#   fixed-scale semantics are validated end-to-end by every golden-vs-kernel
#   case (golden default scale=1.0); tagged on l1_aligned_repr.
# - dtype combination is fixed (bf16 q/k/beta + fp32 gk in, bf16 out):
#   D-DTYPE-bf16 / D-DTYPE-fp32 are exercised by every case (gk fp32
#   everywhere).
# ============================================================================
def _kernel_vs_golden(q, k, gk, beta, chunk_size=64, sub_chunk_size=16):
    """CPU golden + NPU kernel on the same CPU-built inputs (single H2D each)."""
    aqk_ref, akk_ref = fla_chunk_intra_token_parallel_ref(q, k, gk, beta, chunk_size=chunk_size, sub_chunk_size=sub_chunk_size)
    aqk_out, akk_out = fla_chunk_intra_token_parallel(
        q.npu(), k.npu(), gk.npu(), beta.npu(), chunk_size=chunk_size, sub_chunk_size=sub_chunk_size
    )
    return aqk_out, akk_out, aqk_ref, akk_ref


def _run_precision(level, name, B, S, H, DK, chunk_size=64, sub_chunk_size=16, xfn=None):
    """L0/L1 case runner: returns (passed, detail). Blocking on failure."""
    q, k, gk, beta = prepare_input_kda(B, S, H, DK, chunk_size)
    if xfn is not None:
        q, k, gk, beta = xfn(q, k, gk, beta)
    aqk_out, akk_out, aqk_ref, akk_ref = _kernel_vs_golden(q, k, gk, beta, chunk_size, sub_chunk_size)
    p1, r1, m1 = check_precision(aqk_out, aqk_ref, "bfloat16")
    p2, r2, m2 = check_precision(akk_out, akk_ref, "bfloat16")
    detail = f"Aqk ratio={r1:.4f} max_abs={m1:.3e} | Akk ratio={r2:.4f} max_abs={m2:.3e}"
    return p1 and p2, detail


def _make_xfn(scale=None, shift=None):
    """CPU fp32-domain transform applied to q/k/beta BEFORE the bf16 recast.
    gk is NEVER transformed (g_i - g_j <= 0 is an input contract; gate-range
    stress is covered by the D-SPECIAL-DBOUND boundary case)."""

    def xfn(q, k, gk, beta):
        def xf(t):
            v = t.float()
            if scale is not None:
                v = v * scale
            if shift is not None:
                v = v + shift
            return v.to(torch.bfloat16)

        return xf(q), xf(k), gk, xf(beta)

    return xfn


# value-range axes: kept calibrated so max|Aqk| stays well under 256
# (bf16 ulp(256)=2 > the 1e0 max_abs_error_limit hard cap)
_XFN = {
    "small": _make_xfn(scale=0.25),  # ~[-1, 1]
    "large": _make_xfn(scale=2.0),  # ~[-6, 6], products x4
    "asym": _make_xfn(scale=1.5, shift=3.0),  # ~[-1.5, 7.5]
}


def test_fla_chunk_intra_token_parallel_l1():
    """L1 functional tests: in-domain shape/param/value-range coverage.
    Blocking: [PRECISION_FAIL] on any mismatch."""
    l1_cases = [
        # (name, (B,S,H,DK), chunk, sub, xfn, tags)
        (
            "l1_aligned_repr",
            (2, 128, 8, 128),
            64,
            16,
            None,
            ["D-DTYPE-bf16", "D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-chunk_size", "D-PARAM-scale"],
        ),
        # H=6: 6 % 8 = 6 -> BH=2 head-group adaptation (non-canonical mid
        # remainder on the free H axis)
        ("l1_h6_bh2", (1, 128, 6, 128), 64, 16, None, ["D-DTYPE-bf16", "D-SHAPE-TAIL-MID"]),
        # H=12: 12 % 8 = 4 -> BH=4 adaptation path
        ("l1_h12_bh4", (1, 128, 12, 128), 64, 16, None, ["D-DTYPE-bf16"]),
        # DK=48: non-pow2 in-domain width (48 = 32 + 16 mid block)
        ("l1_dk48", (1, 64, 8, 48), 64, 16, None, ["D-DTYPE-bf16", "D-SHAPE-TAIL-MID"]),
        # degenerate minimal: B=1, single chunk S=64, H=2 (prime head count,
        # min legal heads), DK=32 (min legal width)
        ("l1_edge_min_h2", (1, 64, 2, 32), 64, 16, None, ["D-DTYPE-bf16", "D-SHAPE-PRIME", "D-SHAPE-EDGE"]),
        # value ranges (gk gate construction untouched, see _make_xfn)
        ("l1_vrange_s", (1, 64, 8, 128), 64, 16, "small", ["D-VALRANGE-S"]),
        ("l1_vrange_l", (1, 64, 8, 128), 64, 16, "large", ["D-VALRANGE-L"]),
        ("l1_vrange_asym", (1, 64, 8, 128), 64, 16, "asym", ["D-VALRANGE-ASYM"]),
        # genuinely non-default sub_chunk_size=8 (generic SCS path, UB-safe)
        ("l1_scs8", (1, 128, 8, 128), 64, 8, None, ["D-PARAM-sub_chunk_size"]),
    ]
    ok = True
    for name, shape, chunk, sub, xfn_key, _tags in l1_cases:
        B, S, H, DK = shape
        try:
            passed, detail = _run_precision("l1", name, B, S, H, DK, chunk, sub, _XFN.get(xfn_key))
            tag = "PRECISION_PASS" if passed else "PRECISION_FAIL"
            print(f"[{tag}] l1/{name} shape={shape} cs={chunk} scs={sub} {detail}")
            ok &= passed
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"[PRECISION_FAIL] l1/{name} shape={shape}: {exc}")
            ok = False
    return ok


def _l2_inputs(B, S, H, DK, q_dtype=torch.bfloat16, gk_dtype=torch.float32, beta_dtype=torch.bfloat16):
    """Tiny illegal inputs built on CPU then H2D (the wrapper must reject
    before any kernel launch; NPU residency keeps a silent accept observable
    as an actual run rather than a device-mismatch error)."""
    q = torch.randn(B, S, H, DK, dtype=q_dtype)
    k = torch.randn(B, S, H, DK, dtype=q_dtype)
    beta = torch.randn(B, S, H, dtype=beta_dtype)
    gk = torch.randn(B, S, H, DK, dtype=gk_dtype)
    return q.npu(), k.npu(), gk.npu(), beta.npu()


def _run_exception(name, fn):
    """L2 case: illegal input must be rejected. Exception = [BOUNDARY_PASS],
    silent accept = [BOUNDARY_WARN]. Non-blocking either way."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_PASS] l2/{name}: rejected ({type(e).__name__}: {e})")
        return
    print(f"[BOUNDARY_WARN] l2/{name}: illegal input silently accepted")


def test_fla_chunk_intra_token_parallel_l2():
    """L2 negative tests: out-of-domain shapes/dtypes/params must be rejected."""
    op = fla_chunk_intra_token_parallel
    # S remainders (contract: S % CS == 0) -- the S-axis tail/prime analogs
    _run_exception("l2_s_tail1 [D-SHAPE-TAIL-1]", lambda: op(*_l2_inputs(1, 65, 8, 128)))
    _run_exception("l2_s_tailmid [D-SHAPE-TAIL-MID]", lambda: op(*_l2_inputs(1, 96, 8, 128)))
    _run_exception("l2_s_prime [D-SHAPE-PRIME]", lambda: op(*_l2_inputs(1, 127, 8, 128)))
    # H not divisible by the minimum head group 2
    _run_exception("l2_h_odd", lambda: op(*_l2_inputs(1, 64, 7, 128)))
    # DK violations (16-multiple and >= 32)
    _run_exception("l2_dk_not_mul16", lambda: op(*_l2_inputs(1, 64, 8, 33)))
    _run_exception("l2_dk_lt32", lambda: op(*_l2_inputs(1, 64, 8, 16)))
    # chunk_size % sub_chunk_size != 0 (S=100 % 100 == 0 passes the first
    # assert, 100 % 16 != 0 must be rejected) -- non-default chunk_size axis
    _run_exception("l2_chunk_invalid [D-PARAM-chunk_size]", lambda: op(*_l2_inputs(1, 100, 8, 128), chunk_size=100))
    # dtype contract violations (q/k/beta bf16, gk fp32)
    _run_exception("l2_q_fp16 [D-EXC-DTYPE]", lambda: op(*_l2_inputs(1, 64, 8, 128, q_dtype=torch.float16)))
    _run_exception("l2_gk_bf16 [D-EXC-DTYPE]", lambda: op(*_l2_inputs(1, 64, 8, 128, gk_dtype=torch.bfloat16)))
    _run_exception("l2_beta_fp32 [D-EXC-DTYPE]", lambda: op(*_l2_inputs(1, 64, 8, 128, beta_dtype=torch.float32)))


def _inject_special(t, mod, value):
    """Sparse deterministic special-value injection (CPU): every mod-th flat
    element set to value. Guarantees >=1 special AND >=1 finite element."""
    flat = t.view(-1)
    flat[mod - 1 :: mod] = value
    return t


def _run_boundary(name, fn):
    """Boundary case: legal special values, kernel + golden compared with the
    standard mixed tolerance (inf/nan positions compared structurally inside
    check_precision). Precision miss -> [BOUNDARY_WARN], non-blocking."""
    try:
        aqk_out, akk_out, aqk_ref, akk_ref = fn()
        p1, r1, m1 = check_precision(aqk_out, aqk_ref, "bfloat16")
        p2, r2, m2 = check_precision(akk_out, akk_ref, "bfloat16")
        tag = "BOUNDARY_PASS" if (p1 and p2) else "BOUNDARY_WARN"
        print(f"[{tag}] {name} Aqk ratio={r1:.4f} max_abs={m1:.3e} | Akk ratio={r2:.4f} max_abs={m2:.3e}")
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_WARN] {name}: {type(e).__name__}: {e}")


def test_fla_chunk_intra_token_parallel_boundary():
    """Boundary tests: all-zero, sparse inf/nan injection, deep-decay gate
    (DBOUND). Legal inputs; precision-gated, non-blocking."""
    B, S, H, DK = 1, 64, 8, 128

    def bnd_zero():
        q = torch.zeros(B, S, H, DK, dtype=torch.bfloat16)
        k = torch.zeros(B, S, H, DK, dtype=torch.bfloat16)
        beta = torch.zeros(B, S, H, dtype=torch.bfloat16)
        gk = torch.zeros(B, S, H, DK, dtype=torch.float32)
        return _kernel_vs_golden(q, k, gk, beta)

    def bnd_inf_q():
        q, k, gk, beta = prepare_input_kda(B, S, H, DK)
        q = _inject_special(q, 13, float("inf"))
        q = _inject_special(q, 17, float("-inf"))
        return _kernel_vs_golden(q, k, gk, beta)

    def bnd_nan_beta():
        q, k, gk, beta = prepare_input_kda(B, S, H, DK)
        beta = _inject_special(beta, 29, float("nan"))
        return _kernel_vs_golden(q, k, gk, beta)

    def bnd_dbound():
        # deep per-token decay: within-subchunk g_i - g_j down to -180 ->
        # 2^(g_i-g_j) underflows to 0 on BOTH kernel and golden (benign
        # underflow, DESIGN SS10.3); near-diagonal terms stay finite
        q, k, gk, beta = prepare_input_kda(B, S, H, DK)
        pos = (torch.arange(S) % 64).float()
        gk = (-12.0 * pos).view(1, S, 1, 1).repeat(B, 1, H, DK).contiguous()
        return _kernel_vs_golden(q, k, gk, beta)

    _run_boundary("bnd_zero [D-SPECIAL-ZERO]", bnd_zero)
    _run_boundary("bnd_inf_q [D-SPECIAL-INF]", bnd_inf_q)
    _run_boundary("bnd_nan_beta [D-SPECIAL-NAN]", bnd_nan_beta)
    _run_boundary("bnd_dbound [D-SPECIAL-DBOUND]", bnd_dbound)


# ============================================================================
# Coverage manifest (scenario B; counts match the tags above)
# ============================================================================
COVERAGE_CATEGORY = "Fusion"  # proto-declared; strictest dimension set

COVERAGE_MANIFEST = {
    "D-DTYPE-bf16": 5,
    "D-DTYPE-fp32": 1,
    "D-SHAPE-ALIGNED": 1,
    "D-SHAPE-EDGE": 1,
    "D-SHAPE-TAIL-1": 1,
    "D-SHAPE-TAIL-MID": 3,
    "D-SHAPE-PRIME": 2,
    "D-VALRANGE-S": 1,
    "D-VALRANGE-M": 1,
    "D-VALRANGE-L": 1,
    "D-VALRANGE-ASYM": 1,
    "D-PARAM-chunk_size": 2,
    "D-PARAM-sub_chunk_size": 1,
    "D-PARAM-scale": 1,
    "D-SPECIAL-INF": 1,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-DBOUND": 1,
    "D-EXC-DTYPE": 3,
    "D-EXC-SHAPE": 6,
}

COVERAGE_NA = {}  # every mandatory dimension is genuinely covered


# ============================================================================
# Main: --level dispatcher
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()  # no stale compiled artifacts
    torch.manual_seed(0)

    blocking_ok = True  # only L0/L1 count toward the exit code
    if args.level in ("l0", "all"):
        blocking_ok &= test_fla_chunk_intra_token_parallel_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_fla_chunk_intra_token_parallel_l1()
    if args.level in ("l2", "all"):
        test_fla_chunk_intra_token_parallel_l2()  # non-blocking
    if args.level in ("boundary", "all"):
        test_fla_chunk_intra_token_parallel_boundary()  # non-blocking

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
