"""fla_chunk_o layered tests: L0/L1/L2/Boundary + perf sentinel + main(--level).

Golden: PyTorch chunk-loop reference (DESIGN SS9.1; the SS9.1 intra einsum
"bcij,bcjd->bcid" only binds when H == chunk_size and its Ac * M broadcast
only aligns when H == 64 -- fixed to the Triton semantics: mask over the
chunk dims (t, j), o_intra = einsum("bthj,bjhd->bthd", A * M[t, j], v)).
Precision: mixed-tolerance dual gate (precision-standard.md SS4.1, verbatim).
Perf: NPU-event bench (upstream do_bench parity: warmup=10, rep=10, mean,
device-side only; no tri-state markers -- the four-state judgement is owned
by L0/L1).

Coverage mapping notes (scenario B, Fusion category -- strictest set):
- The operator contract REQUIRES aligned inputs (S % 64 == 0, DK in
  {64,96,128} exact-2 split, DV a 16-multiple single block): in-domain
  compute has no tail blocks by construction.
- D-SHAPE-TAIL-* / PRIME are therefore covered BOTH by in-domain
  non-canonical stress (DK=96 -> block_DK=48 non-pow2 K block, DV=48/80
  non-pow2 N width, prime NT/H free axes) AND by the contract-mandated
  rejection of the canonical out-of-domain shapes (L2, tagged here).
- D-PARAM-chunk_size: chunk_size is contractually fixed at 64 (block_S
  semantic, upstream BT=64); the only non-default behavior is rejection,
  verified in L2 (l2_chunk32).
- dtype combination is fixed (bf16 in/out + fp32 gate): D-DTYPE-bf16 /
  D-DTYPE-fp32 are exercised by every case (gate g is fp32 everywhere).
"""

import argparse
import os
import sys

import tilelang
import torch
import torch_npu  # noqa: F401  (registers .npu() on torch)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fla_chunk_o import fla_chunk_o  # noqa: E402


# ============================================================================
# Golden reference (DESIGN SS9.1, fp32 domain, torch.exp2 exact semantics)
# ============================================================================
def fla_chunk_o_ref(q, v, g, A, h, scale=None, chunk_size=64):
    """fla_chunk_o golden. q[B,S,H,DK]/v[B,S,H,DV]/A[B,S,H,C]/h[B,NT,H,DK,DV]
    bf16, g[B,S,H,DK] fp32 -> o[B,S,H,DV] bf16. C = chunk_size = A.shape[-1]."""
    B, S, H, DK = q.shape
    DV = v.shape[-1]
    C = A.shape[-1]
    assert chunk_size == C and S % C == 0, "S must be divisible by chunk_size"
    if scale is None:
        scale = 1.0 / DK**0.5
    NT = S // C
    qf, vf, Af, hf = q.float(), v.float(), A.float(), h.float()
    M = torch.tril(torch.ones(C, C, dtype=torch.bool), 0)  # incl diag (t >= j)
    o = torch.empty(B, S, H, DV, dtype=v.dtype)
    for i in range(NT):
        sl = slice(i * C, (i + 1) * C)
        qc, gc = qf[:, sl], g[:, sl]  # [B,C,H,DK]
        Ac, vc, hc = Af[:, sl], vf[:, sl], hf[:, i]  # [B,C,H,C]/[B,C,H,DV]/[B,H,DK,DV]
        qg = qc * scale * torch.exp2(gc)  # [B,C,H,DK]
        o_inter = torch.einsum("bchk,bhkd->bchd", qg, hc)  # inter-chunk
        # intra-chunk (tril incl diag over the chunk dims (t, j); upstream
        # semantics m_s[t, j] = t >= j, b_o += dot(where(m_s, A, 0), v)).
        # where (not * mask): the strict upper triangle must be zeroed even
        # when A carries Inf/NaN there (Inf*0 = NaN would leak through a
        # multiplicative mask).
        Am = torch.where(M[None, :, None, :], Ac, torch.zeros((), dtype=Ac.dtype))  # [B,C,H,C]
        o_intra = torch.einsum("bthj,bjhd->bthd", Am, vc)
        o[:, sl] = (o_inter + o_intra).to(v.dtype)
    return o


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
# prepare_input: A/h are randn too -- the operator only consumes them).
# xfn: optional (q, v, g, A, h) -> transformed tuple applied in fp32 BEFORE
# the bf16 cast (value-range axes); xfn=None is bit-identical to the plain
# randn construction.
# ============================================================================
def make_inputs(B, S, H, DK, DV, chunk, seed, xfn=None):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, S, H, DK, generator=gen)
    v = torch.randn(B, S, H, DV, generator=gen)
    g = torch.randn(B, S, H, DK, generator=gen)
    A = torch.randn(B, S, H, chunk, generator=gen)
    h = torch.randn(B, S // chunk, H, DK, DV, generator=gen)
    if xfn is not None:
        q, v, g, A, h = xfn(q, v, g, A, h)
    q, v, A, h = (t.to(torch.bfloat16) for t in (q, v, A, h))
    cpu = (q, v, g, A, h)
    npu = tuple(t.npu() for t in cpu)
    return npu, cpu


def _xfn_small(q, v, g, A, h):
    """Symmetric small range [-1, 1]-ish."""
    return q * 0.25, v * 0.25, g * 0.25, A * 0.25, h * 0.25


def _xfn_large(q, v, g, A, h):
    """Symmetric large range, calibrated to the certifiable magnitude regime.

    The bf16 max_abs_error_limit=1e0 hard gate bounds certifiable outputs:
    ulp(|o|) must stay <= 1e0, i.e. |o| < 256 (a bf16 output pair rounding to
    adjacent values differs by one ulp). Empirical calibration (seed 2107):
    spread 1e4/g8 -> max|o| 2.4e18 (uncertifiable: per-element ratio passes
    but the absolute cap cannot), spread 2/g2 -> 580 (marginal), spread
    1.5/g1.5 -> max|o| 69 (2x margin under the cap, 2.4x the natural randn
    case). Beyond-regime behavior is documented by the Boundary DBOUND WARN.
    """
    return q * 1.5, v * 1.5, g * 1.5, A * 1.5, h * 1.5


def _xfn_asym(q, v, g, A, h):
    """Asymmetric ranges: q ~ [0,12], v ~ [-12,0], g ~ [-2,7]."""
    return q * 2 + 6, v * 2 - 6, g * 1.5 + 2.5, A, h


_XFN = {"small": _xfn_small, "large": _xfn_large, "asym": _xfn_asym}


def run_case(name, shape, seed, scale=None, xfn=None):
    """Run one precision case; returns (passed, detail)."""
    B, S, H, DK, DV = shape
    npu_in, cpu_in = make_inputs(B, S, H, DK, DV, 64, seed, xfn=xfn)
    o = fla_chunk_o(*npu_in, scale=scale)
    torch.npu.synchronize()
    ref = fla_chunk_o_ref(*cpu_in, scale=scale)
    passed, ratio, max_abs = check_precision(o, ref, "bfloat16")
    return passed, f"o:ratio={ratio:.4f},max_abs={max_abs:.3e}"


# ============================================================================
# L0 gate tests (DESIGN SS9.2)
# ============================================================================
def test_fla_chunk_o_l0():
    """L0: regular shapes (S%64==0, DK in {64,96,128}, DV 16-multiple)."""
    configs = [
        # min smoke: NT=2, BH=2, block_DK=32 (K=32 boundary fractal + 2-iter
        # Pipelined), block_DV=64, grid=4; covers the full chain
        ("l0_min", (1, 128, 2, 64, 64), 1001),
        # benchmark structure: block_DK=64 (2 iter), block_DV=128 (full
        # width), B>1 exercises the bb/bh decomposition, grid=32
        ("l0_repr", (2, 256, 4, 128, 128), 1002),
    ]
    ok = True
    for name, shape, seed in configs:
        try:
            passed, detail = run_case(name, shape, seed)
            tag = "PRECISION_PASS" if passed else "PRECISION_FAIL"
            print(f"[{tag}] {name} shape={shape} :: {detail}")
            ok &= passed
        except Exception as e:  # noqa: BLE001
            # compile/run errors are NOT precision failures: distinct marker
            # (no [PRECISION_FAIL] -> four-state classifies as runtime failure)
            print(f"[RUNTIME_ERROR] {name} shape={shape}: {type(e).__name__}: {e}")
            ok = False
    return ok


# ============================================================================
# L1 functional tests (scenario B): aligned / edge / non-canonical in-domain
# shapes + value ranges + non-default scale. Tags feed coverage_check.py.
# ============================================================================
L1_CASES = [
    # (name, shape (B,S,H,DK,DV), seed, scale, xfn, tags)
    # aligned representative (randn natural range ~ [-4,4] = mid symmetric;
    # gate g is fp32 in every case -> D-DTYPE-fp32)
    (
        "l1_aligned_repr",
        (2, 256, 4, 128, 128),
        2101,
        None,
        None,
        ["D-DTYPE-bf16", "D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-M"],
    ),
    # benchmark acceptance shape (blocking precision coverage)
    ("l1_bench", (1, 8192, 64, 128, 128), 2102, None, None, ["D-DTYPE-bf16", "D-SHAPE-ALIGNED"]),
    # degenerate: B=1, NT=1 (S=64 single chunk), H=1, DV=16 (min legal N)
    ("l1_edge_min", (1, 64, 1, 64, 16), 2103, None, None, ["D-SHAPE-EDGE"]),
    # in-domain non-canonical mid blocks: DK=96 -> block_DK=48 (non-pow2 K
    # split), DV=48 (non-pow2 N width) -- the in-domain analog of tail-mid
    (
        "l1_d96_dv48",
        (1, 128, 2, 96, 48),
        2104,
        None,
        None,
        ["D-DTYPE-bf16", "D-SHAPE-TAIL-MID"],
    ),
    # prime free axes: NT=3, H=3 (both prime) + DK=96/DV=80 non-pow2
    ("l1_prime_axes", (1, 192, 3, 96, 80), 2105, None, None, ["D-DTYPE-bf16", "D-SHAPE-PRIME"]),
    # value ranges (see _xfn_* above)
    ("l1_vrange_s", (1, 128, 2, 64, 64), 2106, None, "small", ["D-VALRANGE-S"]),
    (
        "l1_vrange_l",
        (1, 128, 2, 64, 64),
        2107,
        None,
        "large",
        ["D-VALRANGE-L", "D-DTYPE-fp32"],
    ),
    ("l1_vrange_asym", (1, 128, 2, 64, 64), 2108, None, "asym", ["D-VALRANGE-ASYM"]),
    # non-default scale (default is 1/sqrt(DK) = 0.125)
    ("l1_scale1", (1, 128, 2, 64, 64), 2109, 1.0, None, ["D-PARAM-scale"]),
]


def test_fla_chunk_o_l1():
    """L1: functional coverage (aligned/edge/non-canonical shapes, value
    ranges, non-default scale). Blocking: [PRECISION_FAIL] on any mismatch."""
    ok = True
    for name, shape, seed, scale, xfn, _tags in L1_CASES:
        try:
            passed, detail = run_case(name, shape, seed, scale=scale, xfn=_XFN.get(xfn))
            tag = "PRECISION_PASS" if passed else "PRECISION_FAIL"
            print(f"[{tag}] {name} shape={shape} :: {detail}")
            ok &= passed
        except Exception as e:  # noqa: BLE001
            print(f"[RUNTIME_ERROR] {name} shape={shape}: {type(e).__name__}: {e}")
            ok = False
    return ok


# ============================================================================
# L2 negative tests (scenario B): out-of-domain inputs must be rejected.
# S%64!=0 / DK not in {64,96,128} / DV invalid / chunk_size!=64 / wrong dtype
# are all contract-excluded (DESIGN SS1.6 / SS5.4) -- the defined behavior
# is rejection, verified here. Non-blocking: rejection = [BOUNDARY_PASS].
# ============================================================================
L2_CASES = [
    # (name, shape (B,S,H,DK,DV), chunk, dtype_override, gate_dtype, tags)
    # S with chunk-remainder 1 (canonical TAIL-1 shape; out-of-domain ->
    # rejection is the contract behavior; in-domain compute has no tails:
    # block_DK=DK//2 exact-2, block_DV=DV single block)
    ("l2_s_tail1", (1, 129, 2, 64, 64), 64, None, None, ["D-EXC-SHAPE", "D-SHAPE-TAIL-1"]),
    # S with mid remainder 32 = 64//2 (canonical TAIL-MID shape)
    ("l2_s_tailmid", (1, 96, 2, 64, 64), 64, None, None, ["D-EXC-SHAPE", "D-SHAPE-TAIL-MID"]),
    # S prime (127), fully non-aligned
    ("l2_s_prime", (1, 127, 2, 64, 64), 64, None, None, ["D-EXC-SHAPE", "D-SHAPE-PRIME"]),
    # DK=32: no valid exact-2 block (block 16 violates the gemm K>=32
    # fractal) -- documented limitation (DESIGN SS3.5.1)
    ("l2_dk32", (1, 128, 2, 32, 64), 64, None, None, ["D-EXC-SHAPE"]),
    # DK=80: 16-multiple without a valid exact-2 block (40 not in {32,48,64})
    ("l2_dk80", (1, 128, 2, 80, 64), 64, None, None, ["D-EXC-SHAPE"]),
    # DV=100: not a 16-multiple (gemm N fractal)
    ("l2_dv_not_mul16", (1, 128, 2, 64, 100), 64, None, None, ["D-EXC-SHAPE"]),
    # DV=144: exceeds the single-block L0C budget (documented limitation)
    ("l2_dv_gt128", (1, 128, 2, 64, 144), 64, None, None, ["D-EXC-SHAPE"]),
    # chunk_size=32: contractually fixed at 64 (block_S semantic); the only
    # non-default chunk_size behavior is rejection (D-PARAM-chunk_size)
    (
        "l2_chunk32",
        (1, 128, 2, 64, 64),
        32,
        None,
        None,
        ["D-EXC-SHAPE", "D-PARAM-chunk_size"],
    ),
    # wrong dtypes: fp16 where bf16 is expected / bf16 gate where fp32 is
    # expected (rejected by the JIT adapter's dtype validation)
    ("l2_dtype_fp16", (1, 128, 2, 64, 64), 64, torch.float16, None, ["D-EXC-DTYPE"]),
    ("l2_gate_bf16", (1, 128, 2, 64, 64), 64, None, torch.bfloat16, ["D-EXC-DTYPE"]),
]


def _run_exception(name, fn):
    """L2 single case: fn() feeds an out-of-domain input; rejection expected.
    Raise -> [BOUNDARY_PASS]; silent acceptance -> [BOUNDARY_WARN]."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_PASS] l2 {name}: correctly rejected ({type(e).__name__})")
        return
    print(f"[BOUNDARY_WARN] l2 {name}: invalid input silently accepted")


def test_fla_chunk_o_l2():
    """L2 negative tests: illegal shapes/dtypes/attrs must be rejected."""
    for name, shape, chunk, dtype_override, gate_dtype, _tags in L2_CASES:
        B, S, H, DK, DV = shape

        def fn(
            B=B,
            S=S,
            H=H,
            DK=DK,
            DV=DV,
            chunk=chunk,
            dtype_override=dtype_override,
            gate_dtype=gate_dtype,
        ):
            # CPU-side construction incl. dtype variants (dtype conversion on
            # the host, single H2D -- no NPU-side aclnn casts in data prep)
            gen = torch.Generator().manual_seed(2201)
            idt = dtype_override or torch.bfloat16
            gdt = gate_dtype or torch.float32
            q = torch.randn(B, S, H, DK, generator=gen).to(idt).npu()
            v = torch.randn(B, S, H, DV, generator=gen).to(idt).npu()
            g = torch.randn(B, S, H, DK, generator=gen).to(gdt).npu()
            A = torch.randn(B, S, H, chunk, generator=gen).to(idt).npu()
            h = torch.randn(B, S // chunk, H, DK, DV, generator=gen).to(idt).npu()
            fla_chunk_o(q, v, g, A, h, chunk_size=chunk)

        _run_exception(name, fn)


# ============================================================================
# Boundary tests (scenario B): legal special values, position-sensitive
# verification (sparse injection with fixed seed: >=1 special AND >=1 finite
# value; inf/nan positions compared structurally by check_precision).
# Non-blocking: precision miss -> [BOUNDARY_WARN].
# ============================================================================
def _bnd_base(B, S, H, DK, DV, seed):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, S, H, DK, generator=gen)
    v = torch.randn(B, S, H, DV, generator=gen)
    g = torch.randn(B, S, H, DK, generator=gen)
    A = torch.randn(B, S, H, 64, generator=gen)
    h = torch.randn(B, S // 64, H, DK, DV, generator=gen)
    return q, v, g, A, h


def _run_boundary(name, fn):
    """Boundary single case: fn() returns (out, ref); precision-checked
    against the same dtype gate as L0/L1; miss or exception -> WARN."""
    try:
        out, ref = fn()
        passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
        tag = "BOUNDARY_PASS" if passed else "BOUNDARY_WARN"
        print(f"[{tag}] boundary {name} :: ratio={ratio:.4f} max_abs={max_abs:.3e}")
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_WARN] boundary {name}: {type(e).__name__}: {e}")


def test_fla_chunk_o_boundary():
    """Boundary: all-zero, sparse +/-inf (v / g), sparse NaN (q / v), gate at
    the exp2 boundary (g = +/-120: 2^120 ~ 1.3e36, near the bf16 max)."""

    def run(name, cpu_tensors):
        q, v, g, A, h = cpu_tensors
        q, v, A, h = (t.to(torch.bfloat16) for t in (q, v, A, h))
        npu = tuple(t.npu() for t in (q, v, g, A, h))
        out = fla_chunk_o(*npu)
        torch.npu.synchronize()
        ref = fla_chunk_o_ref(q, v, g, A, h)
        return out, ref

    B, S, H, DK, DV = 1, 128, 2, 64, 64

    # all-zero inputs -> o == 0 exactly
    def zero():
        return run(
            "zero",
            (
                torch.zeros(B, S, H, DK),
                torch.zeros(B, S, H, DV),
                torch.zeros(B, S, H, DK),
                torch.zeros(B, S, H, 64),
                torch.zeros(B, S // 64, H, DK, DV),
            ),
        )

    # sparse +/-inf in v: structural inf propagation through the intra gemm
    def inf_v():
        q, v, g, A, h = _bnd_base(B, S, H, DK, DV, 2302)
        vf = v.view(-1)
        vf[::37] = float("inf")
        vf[::53] = float("-inf")
        return run("inf_v", (q, v, g, A, h))

    # sparse +inf in g: qg = q * 2^inf -> +/-inf; the hi/lo split yields
    # lo = inf - inf = NaN (kernel) while the golden einsum yields +/-inf --
    # expected [BOUNDARY_WARN], documenting the precision-fallback's inf
    # limitation (the single-cast Triton authority matches golden here)
    def inf_g():
        q, v, g, A, h = _bnd_base(B, S, H, DK, DV, 2303)
        gf = g.view(-1)
        gf[::41] = float("inf")
        return run("inf_g", (q, v, g, A, h))

    # sparse NaN in q and v: NaN propagates identically through both paths
    def nan_qv():
        q, v, g, A, h = _bnd_base(B, S, H, DK, DV, 2304)
        q.view(-1)[::43] = float("nan")
        v.view(-1)[::59] = float("nan")
        return run("nan_qv", (q, v, g, A, h))

    # gate at the exp2 boundary: g = +/-120 (2^120 ~ 1.3e36, just inside the
    # bf16/fp32 finite range; o ~ 1e36 finite, relative tolerance applies)
    def g_dbound():
        q, v, g, A, h = _bnd_base(B, S, H, DK, DV, 2305)
        gf = g.view(-1)
        gf[::43] = 120.0
        gf[::59] = -120.0
        return run("g_dbound", (q, v, g, A, h))

    # regression (PR review): Inf confined to the STRICT UPPER triangle of A
    # must be zeroed by the mask (where semantics). q=h=g=0, v=1, A=0 except
    # A[0, 1]=Inf on the first chunk -> o must be exactly 0 everywhere; a
    # multiplicative mask turns Inf into NaN (Inf*0) and pollutes row 0.
    def a_upper_inf():
        q = torch.zeros(B, S, H, DK)
        v = torch.ones(B, S, H, DV)
        g = torch.zeros(B, S, H, DK)
        A = torch.zeros(B, S, H, 64)
        h = torch.zeros(B, S // 64, H, DK, DV)
        A[0, 0, 0, 1] = float("inf")
        A[0, 64, 1, 2] = float("-inf")  # second chunk, another head
        return run("a_upper_inf", (q, v, g, A, h))

    _run_boundary("bnd_zero [D-SPECIAL-ZERO]", zero)
    _run_boundary("bnd_inf_v [D-SPECIAL-INF]", inf_v)
    _run_boundary("bnd_inf_g [D-SPECIAL-INF]", inf_g)
    _run_boundary("bnd_nan_qv [D-SPECIAL-NAN]", nan_qv)
    _run_boundary("bnd_g_dbound [D-SPECIAL-DBOUND]", g_dbound)
    _run_boundary("bnd_a_upper_inf [D-SPECIAL-INF]", a_upper_inf)


# ============================================================================
# Perf sentinel (DESIGN SS9.4): benchmark shape, NPU-event timing
# ============================================================================
def perf_bench_fla_chunk_o():
    """Perf sentinel: (1, 8192, 64, 128, 128) bf16 + fp32 gate, 60s budget.

    Timing mirrors the upstream do_bench (ref_gpu/fla_chunk_o_tilelang_fwd.py):
    warmup=10, rep=10, mean of per-rep device-side elapsed times, no internal
    synchronize in the timed region. Reports the reference-precision metrics
    informationally (no tri-state markers: four-state is owned by L0/L1).
    """
    B, S, H, DK, DV = 1, 8192, 64, 128, 128
    shape = (B, S, H, DK, DV)
    print(f"perf_bench shape={shape}: compiling (JIT) + reference ...")
    npu_in, cpu_in = make_inputs(B, S, H, DK, DV, 64, 42)
    o = fla_chunk_o(*npu_in)  # compile + warm correctness call
    torch.npu.synchronize()
    ref = fla_chunk_o_ref(*cpu_in)
    passed, ratio, max_abs = check_precision(o, ref, "bfloat16")
    print(f"perf_bench reference check: ratio={ratio:.4f} max_abs={max_abs:.3e} passed={passed}")
    warmup, rep = 10, 10
    start = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    end = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    for _ in range(warmup):
        fla_chunk_o(*npu_in)
    torch.npu.synchronize()
    for i in range(rep):
        start[i].record()
        fla_chunk_o(*npu_in)
        end[i].record()
    torch.npu.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start, end)]  # ms
    mean_ms = sum(times) / len(times)
    verdict = "OK" if mean_ms < 1.320 else "OVER TARGET"
    print(
        f"perf_bench: {mean_ms:.3f} ms (mean of {rep}, warmup {warmup};"
        f" per-rep {[f'{t:.3f}' for t in times]}) target < 1.320 ms -> {verdict}"
    )
    return mean_ms


# ============================================================================
# Coverage annotation (machine-checkable; see coverage-matrix.md). Tags on
# L1/L2 cases above + the manifest below; COVERAGE_NA is empty because the
# Fusion category exempts nothing (out-of-domain scenarios are covered via
# the L2 rejection path, documented per-case).
# ============================================================================
COVERAGE_CATEGORY = "Fusion"

COVERAGE_MANIFEST = {
    "D-DTYPE-bf16": 5,
    "D-DTYPE-fp32": 2,
    "D-SHAPE-ALIGNED": 2,
    "D-SHAPE-EDGE": 1,
    "D-SHAPE-TAIL-1": 1,
    "D-SHAPE-TAIL-MID": 2,
    "D-SHAPE-PRIME": 2,
    "D-VALRANGE-S": 1,
    "D-VALRANGE-M": 1,
    "D-VALRANGE-L": 1,
    "D-VALRANGE-ASYM": 1,
    "D-PARAM-scale": 1,
    "D-PARAM-chunk_size": 1,
    "D-SPECIAL-INF": 2,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-DBOUND": 1,
    "D-EXC-DTYPE": 2,
    "D-EXC-SHAPE": 8,
}

COVERAGE_NA = {}


# ============================================================================
# main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "perf", "all"],
    )
    args = parser.parse_args()

    tilelang.disable_cache()  # no stale compiled artifacts
    torch.manual_seed(0)

    blocking_ok = True
    if args.level in ("l0", "all"):
        blocking_ok &= test_fla_chunk_o_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_fla_chunk_o_l1()
    if args.level in ("l2", "all"):
        test_fla_chunk_o_l2()  # negative: rejection = PASS, silent accept = WARN
    if args.level in ("boundary", "all"):
        test_fla_chunk_o_boundary()  # special values: precision miss = WARN
    if args.level == "perf":
        perf_bench_fla_chunk_o()  # sentinel; own verdict line, non-blocking

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
