"""fla_chunk_inter layered tests: golden + L0 gate (DESIGN section 9.2) +
L1/L2/Boundary stubs (tilelang-op-test-design scenario B fills them after L0
passes) + perf sentinel (DESIGN section 9.4) + main(--level) dispatcher."""

import argparse
import os
import sys

import tilelang
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fla_chunk_inter import fla_chunk_inter, fla_chunk_inter_bwd  # noqa: E402, F401


# ========== Golden reference (DESIGN section 9.1, fp32 domain + torch.exp2) ==========
def fla_chunk_inter_ref(q, k, v, g, h, dv, do, dh, scale=1.0, chunk_size=64):
    """fla_chunk_inter golden. All fp32: q/k/g [B,S,H,DK], v/dv/do [B,S,H,DV],
    h/dh [B,NT,H,DK,DV] (NT=S//chunk_size) -> (dq, dk, dw, dg) all [B,S,H,DK]."""
    B, S, H, DK = q.shape
    C = chunk_size
    assert S % C == 0 and h.shape[1] == S // C, "S must be divisible by chunk_size"
    NT = S // C
    qf, kf, vf, gf = q.float(), k.float(), v.float(), g.float()
    do_f, dv_f, h_f, dh_f = do.float(), dv.float(), h.float(), dh.float()
    dq = torch.empty(B, S, H, DK, dtype=torch.float32)
    dk = torch.empty_like(dq)
    dw = torch.empty_like(dq)
    dg = torch.empty_like(dq)
    for i in range(NT):
        sl = slice(i * C, (i + 1) * C)
        qc, kc, gc = qf[:, sl], kf[:, sl], gf[:, sl]  # [B,C,H,DK]
        vc, doc, dvc = vf[:, sl], do_f[:, sl], dv_f[:, sl]  # [B,C,H,DV]
        hi, dhi = h_f[:, i], dh_f[:, i]  # [B,H,DK,DV]
        gn = gf[:, (i + 1) * C - 1]  # [B,H,DK] (in-domain: always chunk last row)
        # Three GEMMs (fp32; einsum axis order cross-validated against an
        # explicit per-(b,h) matmul loop - see debug log)
        dq_raw = torch.einsum("bthv,bhkv->bthk", doc, hi)  # do @ h^T
        dk_raw = torch.einsum("bthv,bhkv->bthk", vc, dhi)  # v  @ dh^T
        dw_blk = -torch.einsum("bthv,bhkv->bthk", dvc, hi)  # -(dv @ h^T)
        # gating
        dq_blk = dq_raw * scale * torch.exp2(gc)  # [B,C,H,DK]
        dk_blk = dk_raw * torch.exp2(gn.unsqueeze(1) - gc)  # gn broadcast along chunk rows
        # dgk and dg
        dgk = (hi * dhi).sum(-1) * torch.exp2(gn)  # [B,H,DK]
        pp = dgk + (dk_blk * kc).sum(1)  # sum_t dk*k (gated dk)
        dg_blk = qc * dq_blk - kc * dk_blk  # [B,C,H,DK]
        dg_blk[:, -1] += pp  # m_last row injection (in-domain: always last row)
        dq[:, sl], dk[:, sl], dw[:, sl], dg[:, sl] = dq_blk, dk_blk, dw_blk, dg_blk
    return dq, dk, dw, dg


# ========== Precision standard (precision-standard.md section 4.1, verbatim) ==========
def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio).
    Float: mixed tolerance; int: exact match."""
    fp_table = {
        # dtype       : (atol,   rtol,   max_abs_error_limit, required_matched_ratio)
        "float16": (2**-14, 2**-9, 1e-1, 0.99),  # atol 6.10e-5, rtol 1.95e-3
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),  # atol 9.77e-4, rtol 1.56e-2
        "float32": (2**-16, 2**-10, 1e-2, 0.99),  # atol 1.53e-5, rtol 9.77e-4
        "hifloat32": (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4, 2**-2, 1e0, 0.99),  # atol 0.0625, rtol 0.25
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),  # atol 0.125, rtol 0.5
    }
    int_types = {"int8", "int16", "int32", "int64", "uint8"}
    if dtype in int_types:
        return (0.0, 0.0, 0.0, 1.0)  # int: exact match, one mismatch fails
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Precision check: return (passed, matched_ratio, max_abs_error).
    Float dual gate: matched_ratio >= required AND max_abs_error <= limit;
    int: elementwise exact. inf/nan positions compared structurally."""
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:  # int exact match
        mism = (a != g).sum().item()
        total = max(a.numel(), 1)
        return mism == 0, 1.0 - mism / total, (0.0 if mism == 0 else float("inf"))
    a = a.float()
    g = g.float()
    special = ~torch.isfinite(g)  # structural compare at inf/nan positions
    if special.any():  # noqa: SIM102 - verbatim from precision-standard.md 4.1
        if not torch.equal(torch.isnan(a[special]), torch.isnan(g[special])) or not torch.equal(
            torch.isinf(a[special]), torch.isinf(g[special])
        ):
            return False, 0.0, float("inf")
    m = torch.isfinite(g)  # compare all finite-golden positions
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    matched_ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs_error = abs_err.max().item()
    passed = (matched_ratio >= required_ratio) and (max_abs_error <= max_abs_limit)
    return passed, matched_ratio, max_abs_error


# ========== Input construction (CPU randn aligned with upstream prepare_input) ==========
def make_inputs(B, S, H, DK, DV, chunk_size=64, seed=0):
    """All inputs built on CPU (deterministic), moved to NPU once each (H2D)."""
    gen = torch.Generator().manual_seed(seed)
    NT = S // chunk_size

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    q = rnd(B, S, H, DK)
    k = rnd(B, S, H, DK)
    v = rnd(B, S, H, DV)
    g = rnd(B, S, H, DK)
    h = rnd(B, NT, H, DK, DV)
    dv = rnd(B, S, H, DV)
    do = rnd(B, S, H, DV)
    dh = rnd(B, NT, H, DK, DV)
    return q, k, v, g, h, dv, do, dh


def run_case(B, S, H, DK, DV, scale=1.0, chunk_size=64, seed=0, tag="l0", transform=None):
    """CPU inputs -> H2D -> kernel -> CPU golden -> 4-output independent check.

    transform: optional CPU-side callable over the 8-tuple of input tensors
    (value-range scaling / special-value injection), applied before H2D."""
    q, k, v, g, h, dv, do, dh = make_inputs(B, S, H, DK, DV, chunk_size, seed)
    if transform is not None:
        q, k, v, g, h, dv, do, dh = transform((q, k, v, g, h, dv, do, dh))
    ref_q, ref_k, ref_w, ref_g = fla_chunk_inter_ref(q, k, v, g, h, dv, do, dh, scale, chunk_size)
    outs = fla_chunk_inter_bwd(
        q.npu(),
        k.npu(),
        v.npu(),
        g.npu(),
        h.npu(),
        dv.npu(),
        do.npu(),
        dh.npu(),
        scale=scale,
        chunk_size=chunk_size,
    )
    ok = True
    for name, out, ref in zip(("dq", "dk", "dw", "dg"), outs, (ref_q, ref_k, ref_w, ref_g)):
        passed, ratio, max_abs = check_precision(out, ref, "float32")
        marker = "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]"
        print(f"{marker} {tag} shape=({B},{S},{H},{DK},{DV}) out={name} matched_ratio={ratio:.4f} max_abs_error={max_abs:.3e}")
        ok &= passed
    return ok


# ========== L0 gate: DESIGN section 9.2 plan (l0_min + l0_repr, all fp32) ==========
def test_fla_chunk_inter_l0():
    """L0 gate: regular shapes (fully block-divisible), precision convergence."""
    cases = [
        ("l0_min", (1, 128, 2, 64, 64)),  # NT=2, BH=2, NK=2, bDV=32 (K=32 fractal edge)
        ("l0_repr", (2, 256, 4, 128, 128)),  # bench-representative: NK=4, bDV=64, B>1
    ]
    ok = True
    for name, (B, S, H, DK, DV) in cases:
        try:
            ok &= run_case(B, S, H, DK, DV, tag=f"l0:{name}")
        except Exception as e:  # noqa: BLE001 - report and keep going
            print(f"[PRECISION_FAIL] l0:{name} shape=({B},{S},{H},{DK},{DV}): {e}")
            ok = False
    return ok


# ========== Coverage annotations (machine-checkable; coverage-matrix.md) ==========
# Category: CV fusion (3 fp16 hi/lo gemm groups + exp2-aliasing gate chain + dual-axis reduce).
# NOTE on D-SHAPE-TAIL-*/PRIME: the supported domain REQUIRES S%64==0 / DK%32==0 /
# DV%16==0 (chunked data model: non-divisible shapes are the reference
# implementations' undefined domain - h/dh hold no state for a trailing partial
# chunk, Triton i_tg reads OOB, upstream TileLang silently OOBs; DESIGN section
# 1.6 three-layer evidence). The wrapper therefore rejects them deterministically
# (assert), and these shapes are covered as L2 rejection cases in L2_CASES below -
# correct deterministic rejection IS the specified boundary behavior.
COVERAGE_CATEGORY = "Fusion"

L1_CASES = [
    # (name, B, S, H, DK, DV, scale, tags) - all in-domain (block-divisible) fp32
    ("l1_aligned_min", 1, 64, 1, 32, 64, 1.0, ["D-SHAPE-ALIGNED", "D-SHAPE-EDGE", "D-DTYPE-fp32"]),
    ("l1_aligned_nonpow2", 1, 192, 3, 96, 96, 1.0, ["D-SHAPE-ALIGNED", "D-DTYPE-fp32"]),
    ("l1_aligned_dv80", 3, 128, 5, 64, 80, 1.0, ["D-SHAPE-ALIGNED"]),
    ("l1_scale", 2, 128, 2, 64, 64, 0.375, ["D-SHAPE-ALIGNED", "D-PARAM-scale"]),
    ("l1_valrange_s", 1, 128, 2, 64, 64, 1.0, ["D-VALRANGE-S"]),
    ("l1_valrange_m", 1, 128, 2, 64, 64, 1.0, ["D-VALRANGE-M"]),
    ("l1_valrange_l", 1, 128, 2, 64, 64, 1.0, ["D-VALRANGE-L"]),
    ("l1_valrange_asym", 1, 128, 2, 64, 64, 1.0, ["D-VALRANGE-ASYM"]),
]


# L1 value-range transforms (CPU-side, deterministic). They scale the 7
# linear-domain operands (q,k,v,h,dv,do,dh) and deliberately leave g (the
# log2-domain gate input) at its natural randn domain: scaling a log-domain
# input multiplies the gate exponents (2^g / 2^(gn-g) explode to ~2^+20), which
# amplifies |dk| past ~1e6 where the fp32 dual-gate ABSOLUTE cap (1e-2) is
# un-passable for ANY implementation (even golden vs golden with a different
# summation order). Gate-exponent extremes are covered by Boundary cases.
# The operand envelope is also cap-bounded: dg is a TRIPLE product (q*dq with
# dq ~ mul^2*gate), so max_abs_error grows ~mul^3 (measured: mul=4 -> 1.07e-2,
# exactly 4^3 * 1.755e-4). The ladder below stays within the cap with >=2x
# margin: S ~+/-1, M ~+/-4.5, L ~+/-9 (largest cap-compatible domain; true
# dtype-boundary behavior is the Boundary dbound case, non-blocking by design).
def _operands_scaled(ts, mul, add=0.0):
    q, k, v, g, h, dv, do, dh = ts
    return (q * mul + add, k * mul + add, v * mul + add, g, h * mul + add, dv * mul + add, do * mul + add, dh * mul + add)


_L1_TRANSFORMS = {
    "l1_valrange_s": lambda ts: _operands_scaled(ts, 0.25),
    "l1_valrange_m": lambda ts: _operands_scaled(ts, 1.5),
    "l1_valrange_l": lambda ts: _operands_scaled(ts, 3.0),
    "l1_valrange_asym": lambda ts: _operands_scaled(ts, 1.25, 1.0),
}

L2_CASES = [
    # (name, B, S, H, DK, DV, chunk_size, dtype, tags) - illegal inputs; the
    # wrapper must reject (AssertionError). Tail/prime shapes land here by domain.
    ("l2_s_tail1", 1, 65, 2, 64, 64, 64, "float32", ["D-EXC-SHAPE", "D-SHAPE-TAIL-1"]),
    ("l2_s_prime", 1, 67, 2, 64, 64, 64, "float32", ["D-EXC-SHAPE", "D-SHAPE-PRIME"]),
    ("l2_dk_tail1", 1, 128, 2, 33, 64, 64, "float32", ["D-EXC-SHAPE", "D-SHAPE-TAIL-1"]),
    ("l2_dk_tailmid", 1, 128, 2, 48, 64, 64, "float32", ["D-EXC-SHAPE", "D-SHAPE-TAIL-MID"]),
    ("l2_dv_lt64", 1, 128, 2, 64, 32, 64, "float32", ["D-EXC-SHAPE"]),
    ("l2_dv_tailmid", 1, 128, 2, 64, 72, 64, "float32", ["D-EXC-SHAPE", "D-SHAPE-TAIL-MID"]),
    ("l2_chunk_size", 1, 128, 2, 64, 64, 128, "float32", ["D-EXC-SHAPE", "D-PARAM-chunk_size"]),
    ("l2_dtype_fp16", 1, 128, 2, 64, 64, 64, "float16", ["D-EXC-DTYPE"]),
]

# Boundary transforms: legal special values, precision-compared (WARN non-blocking).
# - g_pinf: sparse literal +inf in g at NON-last rows (chunk-last rows stay
#   finite): 2^(+inf)=inf and 2^(gn-inf)=0 -> dq=+/-inf, dk=0, pp finite ->
#   all four outputs structurally consistent between kernel and golden.
# - g_gn_overflow: sparse g = +/-200 INCLUDING chunk-last rows -> 2^gn overflows
#   fp32 to inf (the DESIGN section 9.3 documented extreme). dq/dk/dw match
#   structurally (inf/0 positions); dg WARN is EXPECTED and documented: with an
#   infinite pp, the kernel's LastRow mask-multiply (T.tile.mul of pp2d by the
#   0/1 constant, matching the Triton semantic authority `m_last[:, None] *
#   b_dgk`, which also yields 0*inf=NaN at non-last rows) diverges from this
#   golden's row-indexed add (`dg_blk[:, -1] += pp`), which never touches
#   non-last rows. The two forms are bitwise identical whenever pp is finite
#   (x + 0*pp == x), i.e. across the entire realistic decay domain (g <= 0);
#   the corner only exists when 2^gn overflows.
# - zero: all-zero inputs -> all-zero outputs.
# - dbound: operands x16000 then clamped to +/-60000 (just inside the fp16 hi/lo
#   operand bound 65504; g untouched) - hi/lo stays exact but |out| ~1e10-1e12
#   exceeds the fp32 max_abs hard cap 1e-2 by construction -> expected WARN
#   (ratio still ~1.0: per-element rtol covers the magnitudes).


def _tf_g_pinf(ts):
    q, k, v, g, h, dv, do, dh = ts
    g = g.clone()
    nonlast = torch.tensor([s % 64 != 63 for s in range(g.shape[1])])
    g[:, nonlast, :, ::17] = float("inf")
    return (q, k, v, g, h, dv, do, dh)


def _tf_g_gn_overflow(ts):
    q, k, v, g, h, dv, do, dh = ts
    g = g.clone()
    g[..., ::17] = 200.0
    g[..., 5::23] = -200.0
    return (q, k, v, g, h, dv, do, dh)


def _tf_g_nan(ts):
    q, k, v, g, h, dv, do, dh = ts
    g = g.clone()
    g[..., ::13] = float("nan")
    return (q, k, v, g, h, dv, do, dh)


def _tf_zero(ts):
    return tuple(torch.zeros_like(t) for t in ts)


def _tf_dbound(ts):
    q, k, v, g, h, dv, do, dh = ts

    def f(t):
        return (t * 16000.0).clamp(-60000.0, 60000.0)

    return (f(q), f(k), f(v), g, f(h), f(dv), f(do), f(dh))


BOUNDARY_CASES = [
    # (name, B, S, H, DK, DV, transform, tags)
    ("bd_g_pinf", 1, 128, 2, 64, 64, _tf_g_pinf, ["D-SPECIAL-INF"]),
    ("bd_g_gn_overflow", 1, 128, 2, 64, 64, _tf_g_gn_overflow, ["D-SPECIAL-INF"]),
    ("bd_g_nan", 1, 128, 2, 64, 64, _tf_g_nan, ["D-SPECIAL-NAN"]),
    ("bd_zero", 1, 128, 2, 64, 64, _tf_zero, ["D-SPECIAL-ZERO"]),
    ("bd_dbound", 1, 128, 2, 64, 64, _tf_dbound, ["D-SPECIAL-DBOUND"]),
]


# ========== L1: functional (in-domain shape combos + value ranges), blocking ==========
def test_fla_chunk_inter_l1():
    """L1 functional tests: in-domain shape/scale/value-range combos (DESIGN
    section 9.2: DK=96/DV=96-style non-pow2 combos, degenerate, non-default
    scale, value ranges)."""
    ok = True
    for name, B, S, H, DK, DV, scale, _tags in L1_CASES:
        transform = _L1_TRANSFORMS.get(name)
        try:
            ok &= run_case(B, S, H, DK, DV, scale=scale, tag=f"l1:{name}", transform=transform)
        except Exception as e:  # noqa: BLE001 - report and keep going
            print(f"[PRECISION_FAIL] l1:{name} shape=({B},{S},{H},{DK},{DV}): {e}")
            ok = False
    return ok


# ========== L2: negative (illegal inputs must be rejected), non-blocking ==========
def _run_l2_case(name, B, S, H, DK, DV, chunk_size, dtype):
    """Negative: illegal shape/dtype/chunk_size must be rejected by the wrapper
    assert (deterministic rejection replaces the reference OOB behavior)."""
    try:
        NT = max(1, S // 64)
        f = getattr(torch, dtype)
        q = torch.randn(B, S, H, DK, dtype=f)
        k = torch.randn(B, S, H, DK, dtype=f)
        v = torch.randn(B, S, H, DV, dtype=f)
        g = torch.randn(B, S, H, DK, dtype=f)
        h = torch.randn(B, NT, H, DK, DV, dtype=f)
        dv = torch.randn(B, S, H, DV, dtype=f)
        do = torch.randn(B, S, H, DV, dtype=f)
        dh = torch.randn(B, NT, H, DK, DV, dtype=f)
        fla_chunk_inter_bwd(
            q.npu(),
            k.npu(),
            v.npu(),
            g.npu(),
            h.npu(),
            dv.npu(),
            do.npu(),
            dh.npu(),
            scale=1.0,
            chunk_size=chunk_size,
        )
    except AssertionError as e:
        print(
            f"[BOUNDARY_PASS] l2:{name} shape=({B},{S},{H},{DK},{DV}) chunk={chunk_size} dtype={dtype}: correctly rejected ({str(e)[:80]})"
        )
        return
    except Exception as e:  # noqa: BLE001 - any rejection counts, report type
        print(
            f"[BOUNDARY_PASS] l2:{name} shape=({B},{S},{H},{DK},{DV}) chunk={chunk_size} "
            f"dtype={dtype}: rejected ({type(e).__name__}: {str(e)[:60]})"
        )
        return
    print(f"[BOUNDARY_WARN] l2:{name} shape=({B},{S},{H},{DK},{DV}) chunk={chunk_size} dtype={dtype}: illegal input silently accepted")


def test_fla_chunk_inter_l2():
    """L2 negative tests: S%64!=0 / DK%32!=0 / DV<64 or DV%16!=0 / chunk_size!=64
    / non-fp32 dtype must all be rejected (DESIGN section 9.2, section 1.6)."""
    for name, B, S, H, DK, DV, chunk_size, dtype, _tags in L2_CASES:
        _run_l2_case(name, B, S, H, DK, DV, chunk_size, dtype)


# ========== Boundary: legal special values, precision-compared, non-blocking ==========
def test_fla_chunk_inter_boundary():
    """Boundary tests: INF/NAN/zero/dtype-bound inputs (legal), precision-compared
    with the same fp32 dual-gate standard; a miss is WARN only (documented
    limitations, see BOUNDARY_CASES comments)."""
    for name, B, S, H, DK, DV, transform, _tags in BOUNDARY_CASES:
        try:
            q, k, v, g, h, dv, do, dh = make_inputs(B, S, H, DK, DV, 64, seed=3)
            q, k, v, g, h, dv, do, dh = transform((q, k, v, g, h, dv, do, dh))
            refs = fla_chunk_inter_ref(q, k, v, g, h, dv, do, dh, 1.0, 64)
            outs = fla_chunk_inter_bwd(q.npu(), k.npu(), v.npu(), g.npu(), h.npu(), dv.npu(), do.npu(), dh.npu())
            all_ok = True
            for oname, out, ref in zip(("dq", "dk", "dw", "dg"), outs, refs):
                passed, ratio, max_abs = check_precision(out, ref, "float32")
                all_ok &= passed
                print(
                    f"  boundary:{name} out={oname} matched_ratio={ratio:.4f} max_abs_error={max_abs:.3e} ({'pass' if passed else 'WARN'})"
                )
            marker = "[BOUNDARY_PASS]" if all_ok else "[BOUNDARY_WARN]"
            print(f"{marker} boundary:{name} shape=({B},{S},{H},{DK},{DV})")
        except Exception as e:  # noqa: BLE001 - WARN and continue
            print(f"[BOUNDARY_WARN] boundary:{name} shape=({B},{S},{H},{DK},{DV}): {e}")


# ========== Perf sentinel: DESIGN section 9.4 (upstream do_bench semantics) ==========
def do_bench_npu(fn, warmup=20, rep=10):
    """Port of ref_gpu/test_utils_kda.py do_bench: warmup 20, rep 10, mean,
    device-side events only; no synchronize inside the benched function."""
    start = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    end = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    for i in range(rep):
        start[i].record()
        fn()
        end[i].record()
    torch.npu.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start, end)], dtype=torch.float)
    return times.mean().item()


def run_perf_bench():
    """DESIGN section 9.4 perf sentinel: bench shape (1,8192,64,128,128),
    kernel-time baseline for Stage 3 (proto target < 3400 us)."""
    B, S, H, DK, DV = 1, 8192, 64, 128, 128
    q, k, v, g, h, dv, do, dh = make_inputs(B, S, H, DK, DV, 64, seed=0)
    dev_inputs = [t.npu() for t in (q, k, v, g, h, dv, do, dh)]

    def call():
        fla_chunk_inter_bwd(*dev_inputs)

    call()  # untimed: JIT compile + first launch
    t_ms = do_bench_npu(call)
    print(
        f"[PERF_BENCH] shape=({B},{S},{H},{DK},{DV}) kernel_time={t_ms * 1000:.1f} us "
        f"(do_bench warmup=20/rep=10/mean/device-side; proto target < 3400 us)"
    )
    return t_ms * 1000.0


# ========== Main: --level dispatcher + exit code ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all", "perf"])
    args = parser.parse_args()

    tilelang.disable_cache()  # disable compilation cache (stale-artifact guard)
    torch.manual_seed(0)

    blocking_ok = True  # only L0/L1 block the exit code
    if args.level in ("l0", "all"):
        blocking_ok &= test_fla_chunk_inter_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_fla_chunk_inter_l1()
    if args.level in ("l2", "all"):
        test_fla_chunk_inter_l2()  # negative: correct rejection = PASS, non-blocking
    if args.level in ("boundary", "all"):
        test_fla_chunk_inter_boundary()  # special values: WARN only, non-blocking
    if args.level == "perf":
        run_perf_bench()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
