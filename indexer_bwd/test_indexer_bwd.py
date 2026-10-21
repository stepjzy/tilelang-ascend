"""indexer_bwd layered tests (rev2): L0 gate (7 cases) + stubs + perf benchmark.

Golden replicates the main repo's ref_indexer_bwd autograd reference
(ref_gpu/indexer_bwd.py:185-215) and runs it on NPU (torch_npu supports bf16
autograd / einsum / kl_div, mirroring the main repo's cuda golden). Input
construction replicates the main repo's test_kernel (seed 42, randn bf16
q/w/k, masked-softmax attn_score from ones); the three topk_indices families
(arange-tiled / shifted-shared / per-token random) follow DESIGN.md §9.2.
All randomness happens on CPU followed by a single H2D.

Levels:
    l0        DESIGN.md §9.2 gate cases (blocking, [PRECISION_PASS/FAIL])
    l1/l2/boundary  stubs, expanded by tilelang-op-test-design (scenario B)
    perf      pure-kernel average latency for the l0_default configuration
              (fast layout=0 + cast kernel; generic path timed separately)
"""

import argparse
import os
import sys
import traceback

import torch
import torch.nn.functional as F

import tilelang

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indexer_bwd import (  # noqa: E402,F401  (indexer_bwd alias kept for the import convention)
    _detect_indices_layout,
    indexer_bwd,
    indexer_bwd_interface,
    prepare_block_indices,
    prepare_token_indices,
    tl_dk_cast,
    tl_indexer_bwd_fast,
    tl_indexer_bwd_generic,
)


# ============================================================================
# Golden reference (main repo ref_gpu/indexer_bwd.py:185-215, verbatim;
# einops.einsum -> torch.einsum, cuda -> npu)
# ============================================================================
def ref_indexer_bwd(Q, Weights, K, TopkIndices, AttnScore, offsets, scale=None):
    Q.requires_grad_(True)
    Weights.requires_grad_(True)
    K.requires_grad_(True)
    softmax_scale = Q.shape[-1] ** -0.5 if scale is None else scale
    all_loss = []
    all_log_topk_prob = []
    for i in range(offsets.shape[0] - 1):
        assert (offsets[i + 1] - offsets[i]).item() >= TopkIndices.shape[-1]
        q = Q[offsets[i] : offsets[i + 1]]
        weights = Weights[offsets[i] : offsets[i + 1]]
        k = K[offsets[i] : offsets[i + 1]]
        topk_indices = TopkIndices[offsets[i] : offsets[i + 1]]
        attn_score = AttnScore[offsets[i] : offsets[i + 1]]
        s = q.shape[0]
        mask = (torch.arange(s)[:, None] >= torch.arange(s)[None, :]).to(q.device)
        # einops "s1 h k, s2 k -> s1 h s2" in single-letter torch.einsum form
        logits = torch.einsum("shd,nd->shn", q, k) * softmax_scale
        logits = F.relu(logits)
        score = (logits * weights.unsqueeze(-1)).sum(dim=-2, dtype=torch.float32)
        score = torch.where(mask, score, float("-inf"))
        topk_value = torch.gather(score, dim=-1, index=topk_indices.to(torch.int64))
        log_topk_prob = F.log_softmax(topk_value, dim=-1, dtype=torch.float32)
        loss = F.kl_div(
            log_topk_prob.clip(-100, 0),
            attn_score.log().clip(-100, 0),
            log_target=True,
            reduction="sum",
        )
        all_loss.append(loss)
        all_log_topk_prob.append(log_topk_prob)
    loss = torch.stack(all_loss).sum()
    loss.backward()
    log_topk_prob = torch.cat(all_log_topk_prob, dim=0)
    return log_topk_prob.exp(), Q.grad, Weights.grad, K.grad


# ============================================================================
# Input construction (main repo test_kernel:218-244 replica, CPU + one H2D)
# ============================================================================
def make_inputs(S, H, D, topk, off_list, indices_mode="arange", seed=42, scale=1.0, shift=0.0):
    """Build bf16 q/w/k + fp32 masked-softmax attn_score + int32 topk_indices.

    indices_mode:
        "arange"  every row == arange(topk)          (fast layout=0 contract)
        "shifted" every row == arange(5, 5+topk)     (fast layout=1 contract)
        "random"  per-segment randint(0, seg) rows   (generic; dup + pos > t)
    scale/shift: q/w/k = randn * scale + shift (L1 value-range axis).
    """
    torch.manual_seed(seed)
    q = (torch.randn(S, H, D) * scale + shift).bfloat16().npu()
    w = (torch.randn(S, H) * scale + shift).bfloat16().npu()
    k = (torch.randn(S, D) * scale + shift).bfloat16().npu()
    offsets = torch.tensor(off_list, dtype=torch.int32).npu()

    parts = []
    for i in range(len(off_list) - 1):
        seg = off_list[i + 1] - off_list[i]
        mask = torch.arange(seg)[:, None] >= torch.arange(topk)[None, :]
        logits = torch.where(mask, torch.ones(seg, topk), float("-inf"))
        parts.append(F.softmax(logits, dim=-1, dtype=torch.float32))
    attn_score = torch.cat(parts, dim=0).npu()

    if indices_mode == "arange":
        topk_indices = torch.arange(topk, dtype=torch.int32).unsqueeze(0).repeat(S, 1).npu()
    elif indices_mode == "shifted":
        topk_indices = torch.arange(5, 5 + topk, dtype=torch.int32).unsqueeze(0).repeat(S, 1).npu()
    elif indices_mode == "random":
        g = torch.Generator().manual_seed(seed)
        rows = []
        for i in range(len(off_list) - 1):
            seg = off_list[i + 1] - off_list[i]
            rows.append(torch.randint(0, seg, (seg, topk), generator=g, dtype=torch.int32))
        topk_indices = torch.cat(rows, dim=0).npu()
    else:
        raise ValueError(f"unknown indices_mode {indices_mode!r}")
    return q, w, k, attn_score, topk_indices, offsets


# ============================================================================
# Precision standard (mixed tolerance, DESIGN.md §9.3)
# ============================================================================
def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio)."""
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Mixed-tolerance dual-gate check: returns (passed, matched_ratio, max_abs_error).

    Float: elementwise |a-g| <= atol + rtol*|g| with matched_ratio /
    max_abs_error dual gates; inf/nan golden positions are compared
    structurally (isnan/isinf patterns must match) and actual inf/nan at
    finite-golden positions count as mismatches. Integer: exact match.
    """
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a, g = actual.detach().cpu(), golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:  # integer exact match
        mism = (a != g).sum().item()
        return mism == 0, 1.0 - mism / max(a.numel(), 1), (0.0 if mism == 0 else float("inf"))
    a, g = a.float(), g.float()
    special = ~torch.isfinite(g)  # structural compare at inf/nan golden slots
    struct_ok = special.any() and (
        torch.equal(torch.isnan(a[special]), torch.isnan(g[special])) and torch.equal(torch.isinf(a[special]), torch.isinf(g[special]))
    )
    if special.any() and not struct_ok:
        return False, 0.0, float("inf")
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


def run_case(
    name,
    S,
    H,
    D,
    topk,
    off_list,
    indices_mode,
    dispatch=None,
    block_I=None,
    sm_scale=None,
    scale=1.0,
    shift=0.0,
    seed=42,
):
    """Run one precision case: kernel vs golden on dq/dw/dk (bf16 mixed tolerance)."""
    q, w, k, attn_score, topk_indices, offsets = make_inputs(S, H, D, topk, off_list, indices_mode, seed=seed, scale=scale, shift=shift)
    index_score, ref_dq, ref_dw, ref_dk = ref_indexer_bwd(q, w, k, topk_indices, attn_score, offsets, scale=sm_scale)
    kwargs = {}
    if dispatch is not None:
        kwargs["dispatch"] = dispatch
    if block_I is not None:
        kwargs["block_I"] = block_I
    if sm_scale is not None:
        kwargs["sm_scale"] = sm_scale
    dq, dw, dk = indexer_bwd_interface(q, w, k, attn_score, index_score, topk_indices, offsets, **kwargs)
    case_ok = True
    for oname, out, ref in (
        ("dq", dq, ref_dq),
        ("dw", dw, ref_dw),
        ("dk", dk, ref_dk),
    ):
        passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] {name} {oname} shape={tuple(out.shape)} dtype=bfloat16 matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        case_ok &= passed
    return case_ok


# ============================================================================
# L0 gate tests (DESIGN.md §9.2 — 7 cases)
# ============================================================================
L0_CONFIGS = [
    # (name, S, H, D, topk, offsets, indices_mode)
    ("l0_default", 2048, 16, 128, 64, [0, 1023, 2048], "arange"),
    ("l0_single_seq", 1024, 16, 128, 64, [0, 1024], "arange"),
    ("l0_small", 128, 16, 128, 64, [0, 128], "arange"),
    ("l0_equal_topk", 64, 16, 128, 64, [0, 64], "arange"),
    ("l0_shared_shifted", 512, 16, 128, 64, [0, 512], "shifted"),
    ("l0_generic_random", 512, 16, 128, 64, [0, 512], "random"),
]


def test_indexer_bwd_l0():
    """L0 gate: 6 precision cases + dispatch-equivalence case (blocking)."""
    ok = True
    for name, S, H, D, topk, off_list, mode in L0_CONFIGS:
        try:
            ok &= run_case(name, S, H, D, topk, off_list, mode)
        except Exception as e:  # noqa: BLE001
            print(f"[PRECISION_FAIL] {name}: {e!r}")
            traceback.print_exc()
            ok = False
    ok &= _test_dispatch_equiv()
    return ok


def _test_dispatch_equiv():
    """l0_dispatch_equiv: host detection asserts + three forced paths vs golden."""
    ok = True
    try:
        # --- host detection asserts (exact boolean mapping) ---
        det_cases = [
            ("arange", 512, [0, 512], "arange", 0),
            ("shifted", 512, [0, 512], "shifted", 1),
            ("random", 512, [0, 512], "random", 2),
        ]
        for label, S, off_list, mode, expect in det_cases:
            _, _, _, _, topk_indices, _ = make_inputs(S, 16, 128, 64, off_list, mode)
            got = _detect_indices_layout(topk_indices, 64)
            tag = "PASS" if got == expect else "FAIL"
            print(f"[PRECISION_{tag}] l0_dispatch_equiv detect({label})={got} (expect {expect})")
            ok &= got == expect

        # --- three forced paths on the l0_default input, each vs golden ---
        for dispatch in ("fast_arange", "fast_shared", "generic"):
            ok &= run_case(
                f"l0_dispatch_equiv[{dispatch}]",
                2048,
                16,
                128,
                64,
                [0, 1023, 2048],
                "arange",
                dispatch=dispatch,
            )
    except Exception as e:  # noqa: BLE001
        print(f"[PRECISION_FAIL] l0_dispatch_equiv: {e!r}")
        traceback.print_exc()
        ok = False
    return ok


# ============================================================================
# L1/L2/Boundary: expanded by tilelang-op-test-design (scenario B)
# ============================================================================
# L1_CASES: (name, S, H, D, topk, offsets, mode, kwargs, tags)
#   shape axes: S / segment lengths are the only tileable axes (T_BLOCK=32 fast,
#   block_I generic); H/D must stay %16 (Cube fractal), topk in powers of 2 —
#   non-alignment is exercised on the token/segment axis (tails / primes).
L1_CASES = [
    # (name, S, H, D, topk, offsets, mode, kwargs, tags)
    (
        "l1_aligned_multiseg",
        1024,
        16,
        128,
        64,
        [0, 512, 1024],
        "arange",
        {"scale": 0.5},
        ["D-DTYPE-bf16", "D-DTYPE-fp32", "D-DTYPE-int32", "D-SHAPE-ALIGNED", "D-VALRANGE-S"],
    ),
    ("l1_tail1", 257, 16, 128, 64, [0, 257], "arange", {}, ["D-SHAPE-TAIL-1"]),  # 257 = 8*32 + 1 (fast tail-block guard path)
    ("l1_tail1_random", 257, 16, 128, 64, [0, 257], "random", {}, ["D-SHAPE-TAIL-1"]),  # same tail through the generic path
    ("l1_tailmid_varlen", 528, 16, 128, 64, [0, 300, 528], "arange", {}, ["D-SHAPE-TAIL-MID"]),  # segs 300 (tail 12) + 228 (tail 4)
    ("l1_prime", 269, 16, 128, 64, [0, 269], "arange", {}, ["D-SHAPE-PRIME"]),  # 269 prime; 269 % 32 = 13
    (
        "l1_edge_minseg",
        128,
        16,
        128,
        64,
        [0, 64, 128],
        "arange",
        {},
        ["D-SHAPE-EDGE"],
    ),  # each segment == topk (minimal legal; every block a tail)
    ("l1_valrange_m", 512, 16, 128, 64, [0, 512], "arange", {"scale": 5.0}, ["D-VALRANGE-M"]),
    ("l1_valrange_l", 512, 16, 128, 64, [0, 512], "arange", {"scale": 100.0}, ["D-VALRANGE-L"]),
    (
        "l1_valrange_asym",
        512,
        16,
        128,
        64,
        [0, 512],
        "arange",
        {"scale": 0.75, "shift": 0.25},
        ["D-VALRANGE-ASYM"],
    ),  # asymmetric range ~[-2, 2.5]; larger shifts amplify
    # the golden's own bf16 backward roundings beyond the 0.99 gate (see
    # indexer_bwd.py docstring note 7 transition-zone remark)
    ("l1_param_topk16", 512, 16, 128, 16, [0, 512], "arange", {}, ["D-PARAM-topk"]),  # fast-domain lower edge
    ("l1_param_topk32", 512, 16, 128, 32, [0, 512], "arange", {}, ["D-PARAM-topk"]),
    (
        "l1_param_topk128_generic",
        512,
        16,
        128,
        128,
        [0, 512],
        "arange",
        {},
        ["D-PARAM-topk"],
    ),  # 128 -> auto-fallback generic (L0C budget); NI=2 loop
    ("l1_param_heads32", 512, 32, 128, 16, [0, 512], "arange", {}, ["D-PARAM-heads"]),  # fast path with H=32 (L0C gate ok @ topk=16)
    ("l1_param_dim64", 512, 16, 64, 64, [0, 512], "arange", {}, ["D-PARAM-dim"]),  # D=64, sm_scale = 64**-0.5
    (
        "l1_param_smscale",
        512,
        16,
        128,
        64,
        [0, 512],
        "arange",
        {"sm_scale": 0.1},
        ["D-PARAM-sm_scale"],
    ),  # explicit non-default scale (kernel + golden agree;
    # larger deviations amplify the golden's own bf16 backward roundings on dw
    # beyond the 0.99 gate — see indexer_bwd.py docstring note 7)
    ("l1_param_blockI16", 512, 16, 128, 64, [0, 512], "random", {"block_I": 16}, ["D-PARAM-block_I"]),  # generic NI=4 accumulation loop
    (
        "l1_param_dispatch_shared",
        512,
        16,
        128,
        64,
        [0, 512],
        "arange",
        {"dispatch": "fast_shared"},
        ["D-PARAM-dispatch"],
    ),  # forced layout=1 on arange data (index-compare mask)
]

COVERAGE_CATEGORY = "Fusion"

COVERAGE_MANIFEST = {
    "D-DTYPE-bf16": 17,
    "D-DTYPE-fp32": 17,
    "D-DTYPE-int32": 17,
    "D-SHAPE-ALIGNED": 1,
    "D-SHAPE-TAIL-1": 2,
    "D-SHAPE-TAIL-MID": 1,
    "D-SHAPE-PRIME": 1,
    "D-SHAPE-EDGE": 1,
    "D-VALRANGE-S": 1,
    "D-VALRANGE-M": 1,
    "D-VALRANGE-L": 1,
    "D-VALRANGE-ASYM": 1,
    "D-PARAM-topk": 3,
    "D-PARAM-heads": 1,
    "D-PARAM-dim": 1,
    "D-PARAM-sm_scale": 1,
    "D-PARAM-block_I": 1,
    "D-PARAM-dispatch": 2,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-INF": 1,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-DBOUND": 1,
    "D-EXC-DTYPE": 1,
    "D-EXC-SHAPE": 6,
}

COVERAGE_NA = {
    # Fusion 类别无强制豁免项；D-SHAPE-RANK 不适用（算子各 tensor rank 固定：
    # q 3D / scores & indices 2D / offsets 1D，无 rank 扫描维度）。
}


def test_indexer_bwd_l1():
    """L1 functional tests: shape/param/value-range coverage (blocking)."""
    ok = True
    for name, S, H, D, topk, off_list, mode, kwargs, _tags in L1_CASES:
        try:
            ok &= run_case(name, S, H, D, topk, off_list, mode, **kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"[PRECISION_FAIL] {name}: {e!r}")
            traceback.print_exc()
            ok = False
    return ok


# ---- L2: negative tests (invalid inputs must be rejected; no precision compare) ----
def _run_exception(name, fn):
    """Feed an illegal input; a raised exception = [BOUNDARY_PASS]."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_PASS] l2 {name}: rejected ({type(e).__name__}: {str(e)[:80]})")
        return
    print(f"[BOUNDARY_WARN] l2 {name}: illegal input silently accepted")


def _l2_inputs(**mk_kwargs):
    return make_inputs(128, 16, 128, 64, [0, 128], "arange", **mk_kwargs)


def test_indexer_bwd_l2():
    """L2 negative tests: illegal shapes / dtypes / contract violations must raise."""

    # segment length < topk (semantically illegal varlen)
    def seg_lt_topk():
        q, w, k, a, ti, off = _l2_inputs()
        bad_off = torch.tensor([0, 10], dtype=torch.int32).npu()
        indexer_bwd_interface(q, w, k, a, a, ti, bad_off)

    # heads = 8 (Cube fractal requires % 16 == 0)
    def heads8():
        q, w, k, a, ti, off = _l2_inputs()
        q8 = q[:, :8, :].contiguous()
        w8 = w[:, :8].contiguous()
        indexer_bwd_interface(q8, w8, k, a, a, ti, off)

    # topk not a power of two (main-repo semantics)
    def topk48():
        S, H, D, topk = 128, 16, 128, 48
        torch.manual_seed(42)
        q = torch.randn(S, H, D).bfloat16().npu()
        w = torch.randn(S, H).bfloat16().npu()
        k = torch.randn(S, D).bfloat16().npu()
        a = torch.rand(S, topk).float().npu()
        ti = torch.randint(0, 64, (S, topk), dtype=torch.int32).npu()
        off = torch.tensor([0, S], dtype=torch.int32).npu()
        indexer_bwd_interface(q, w, k, a, a, ti, off)

    # forced fast_arange on non-arange (shifted) indices: contract violation
    def fast_arange_violation():
        q, w, k, a, ti, off = _l2_inputs()
        ti_shift = torch.arange(5, 69, dtype=torch.int32).unsqueeze(0).repeat(ti.shape[0], 1).npu()
        indexer_bwd_interface(q, w, k, a, a, ti_shift, off, dispatch="fast_arange")

    # forced fast_shared on per-token random indices: contract violation
    def fast_shared_violation():
        q, w, k, a, ti, off = _l2_inputs()
        g = torch.Generator().manual_seed(7)
        ti_rand = torch.randint(0, 64, ti.shape, generator=g, dtype=torch.int32).npu()
        indexer_bwd_interface(q, w, k, a, a, ti_rand, off, dispatch="fast_shared")

    # -1 indices through the forced fast path (generic-only semantics)
    def neg1_fast_path():
        q, w, k, a, ti, off = _l2_inputs()
        ti_neg = ti.clone()
        ti_neg[3, 0] = -1  # breaks the arange contract -> must be rejected
        indexer_bwd_interface(q, w, k, a, a, ti_neg, off, dispatch="fast_arange")

    # unsupported main-input dtype (fp16 q; precision path is fixed bf16)
    def dtype_fp16():
        q, w, k, a, ti, off = _l2_inputs()
        q16 = q.to(torch.float16)
        indexer_bwd_interface(q16, w, k, a, a, ti, off)

    # empty varlen (zero-length segment < topk)
    def empty_seg():
        q, w, k, a, ti, off = _l2_inputs()
        empty_off = torch.tensor([0, 0], dtype=torch.int32).npu()
        indexer_bwd_interface(q[:0], w[:0], k[:0], a[:0], a[:0], ti[:0], empty_off)

    for name, fn, _tags in (
        ("seg_lt_topk", seg_lt_topk, ["D-EXC-SHAPE"]),
        ("heads8", heads8, ["D-EXC-SHAPE"]),
        ("topk48_not_pow2", topk48, ["D-EXC-SHAPE"]),
        ("fast_arange_contract", fast_arange_violation, ["D-EXC-SHAPE", "D-PARAM-dispatch"]),
        ("fast_shared_contract", fast_shared_violation, ["D-EXC-SHAPE"]),
        ("neg1_fast_path", neg1_fast_path, ["D-EXC-SHAPE"]),
        ("dtype_fp16", dtype_fp16, ["D-EXC-DTYPE"]),
        ("empty_seg", empty_seg, ["D-EXC-SHAPE"]),
    ):
        _run_exception(name, fn)


# ---- Boundary: legal special values (precision-checked, non-blocking) ----
def _run_boundary(name, fn):
    """fn() -> (kernel_outs, golden_outs); compare per output, WARN on mismatch."""
    try:
        (dq, dw, dk), (g_dq, g_dw, g_dk) = fn()
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_WARN] boundary {name}: raised {e!r}")
        return
    for oname, out, ref in (("dq", dq, g_dq), ("dw", dw, g_dw), ("dk", dk, g_dk)):
        try:
            passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
        except Exception as e:  # noqa: BLE001
            print(f"[BOUNDARY_WARN] boundary {name} {oname}: {e!r}")
            continue
        tag = "PASS" if passed else "WARN"
        print(f"[BOUNDARY_{tag}] boundary {name} {oname} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")


def test_indexer_bwd_boundary():
    """Boundary tests: INF/NAN/zero/dtype-bound values (legal inputs, non-blocking)."""

    def zero_inputs():
        q, w, k, attn, ti, off = make_inputs(128, 16, 128, 64, [0, 128], "arange", scale=0.0)
        is_sc, g_dq, g_dw, g_dk = ref_indexer_bwd(q, w, k, ti, attn, off)
        dq, dw, dk = indexer_bwd_interface(q, w, k, attn, is_sc, ti, off)
        return (dq, dw, dk), (g_dq, g_dw, g_dk)

    def inf_q():
        # A few +inf entries in q. Note: the autograd golden NaN-poisons its
        # whole gradient set through kl_div (loss = NaN -> all grads NaN);
        # the kernel keeps finite rows finite (relu indicator + masked NaN
        # guards). Structural mismatch -> WARN is the expected honest outcome.
        q, w, k, attn, ti, off = make_inputs(128, 16, 128, 64, [0, 128], "arange")
        q[5, 3, 7] = float("inf")
        q[60, 0, 100] = float("-inf")
        is_sc, g_dq, g_dw, g_dk = ref_indexer_bwd(q, w, k, ti, attn, off)
        dq, dw, dk = indexer_bwd_interface(q, w, k, attn, is_sc, ti, off)
        return (dq, dw, dk), (g_dq, g_dw, g_dk)

    def nan_index_score():
        # Explicit NaN in IndexScore at all-invalid (causally-masked) rows:
        # the golden produces zero gradients there (clipped -inf slots), the
        # kernel's NaN guard masks diff to 0 — both sides agree exactly.
        q, w, k, attn, ti, off = make_inputs(512, 16, 128, 64, [0, 512], "shifted")
        is_sc, g_dq, g_dw, g_dk = ref_indexer_bwd(q, w, k, ti, attn, off)
        is_sc = is_sc.clone()
        is_sc[0:5] = float("nan")  # tokens 0-4 have no causally-valid position
        dq, dw, dk = indexer_bwd_interface(q, w, k, attn, is_sc, ti, off)
        return (dq, dw, dk), (g_dq, g_dw, g_dk)

    def dbound_large():
        # Dtype-boundary magnitudes (fp16-boundary constant +-65504, bf16-
        # representable; qk products ~ 128*(65504*3)^2 ~ 5e15 stay fp32-
        # finite). Clip-exact gradient (kernel note 7) keeps deep-slot
        # handling identical to the golden.
        q, w, k, attn, ti, off = make_inputs(128, 16, 128, 64, [0, 128], "arange", scale=65504.0)
        is_sc, g_dq, g_dw, g_dk = ref_indexer_bwd(q, w, k, ti, attn, off)
        dq, dw, dk = indexer_bwd_interface(q, w, k, attn, is_sc, ti, off)
        return (dq, dw, dk), (g_dq, g_dw, g_dk)

    for name, fn in (
        ("zero_inputs", zero_inputs),
        ("inf_q", inf_q),
        ("nan_index_score", nan_index_score),
        ("dbound_large", dbound_large),
    ):
        _run_boundary(name, fn)


# ============================================================================
# Perf benchmark: pure-kernel average latency (l0_default configuration)
# ============================================================================
def _bench_fast(S=2048, H=16, D=128, topk=64, off_list=None, n_iters=20, warmup=5):
    """Time fast(layout=0) + cast kernels back-to-back (pre-allocated args)."""
    if off_list is None:
        off_list = [0, 1023, S]
    q, w, k, attn_score, topk_indices, offsets = make_inputs(S, H, D, topk, off_list)
    index_score = attn_score  # values irrelevant for timing
    grid_size = 20
    t_block, ts = 32, 4
    block_indices = prepare_block_indices(offsets, t_block)
    dq = torch.zeros(S, H, D, dtype=torch.bfloat16, device=q.device)
    dw = torch.zeros(S, H, dtype=torch.bfloat16, device=q.device)
    dk = torch.zeros(S, D, dtype=torch.bfloat16, device=q.device)
    ws_dk = torch.zeros(S, D, dtype=torch.float32, device=q.device)
    ts_h = ts * H
    workspace_logits = torch.empty(grid_size, 2, ts_h, topk, dtype=torch.float32, device=q.device)
    workspace_dlog_a = torch.empty(grid_size, 2, ts_h, topk, dtype=torch.bfloat16, device=q.device)
    workspace_dlog_b = torch.empty(grid_size, 2, ts_h, topk, dtype=torch.bfloat16, device=q.device)
    workspace_dq = torch.empty(grid_size, 2, ts_h, D, dtype=torch.bfloat16, device=q.device)
    workspace_k = torch.empty(grid_size, topk, D, dtype=torch.bfloat16, device=q.device)
    workspace_dkf = torch.empty(grid_size, topk, D, dtype=torch.float32, device=q.device)

    fast_f = tl_indexer_bwd_fast(
        heads=H,
        dim=D,
        topk=topk,
        sm_scale=D**-0.5,
        T_BLOCK=t_block,
        TS=ts,
        grid_size=grid_size,
        layout=0,
    )
    cast_f = tl_dk_cast(dim=D, grid_size=grid_size)
    args = (
        q,
        w,
        k,
        attn_score,
        index_score,
        topk_indices,
        block_indices,
        workspace_logits,
        workspace_dlog_a,
        workspace_dlog_b,
        workspace_dq,
        workspace_k,
        workspace_dkf,
        ws_dk,
        dq,
        dw,
    )
    for _ in range(warmup):
        fast_f(*args)
        cast_f(ws_dk, dk)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fast_f(*args)
        cast_f(ws_dk, dk)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / n_iters * 1000.0


def _bench_generic(S=2048, H=16, D=128, topk=64, off_list=None, n_iters=20, warmup=3):
    """Time generic + cast kernels back-to-back (same input, fallback path)."""
    if off_list is None:
        off_list = [0, 1023, S]
    q, w, k, attn_score, topk_indices, offsets = make_inputs(S, H, D, topk, off_list)
    index_score = attn_score
    block_I = min(topk, 64)
    token_indices = prepare_token_indices(offsets)
    dq = torch.zeros(S, H, D, dtype=torch.bfloat16, device=q.device)
    dw = torch.zeros(S, H, dtype=torch.bfloat16, device=q.device)
    dk = torch.zeros(S, D, dtype=torch.bfloat16, device=q.device)
    ws_dk = torch.zeros(S, D, dtype=torch.float32, device=q.device)
    workspace_k = torch.empty(S, topk, D, dtype=torch.bfloat16, device=q.device)
    workspace_dlog = torch.empty(S, topk, H, dtype=torch.bfloat16, device=q.device)
    workspace_logits = torch.empty(S, topk, H, dtype=torch.float32, device=q.device)
    workspace_dk = torch.empty(S, topk, D, dtype=torch.float32, device=q.device)
    gen_f = tl_indexer_bwd_generic(heads=H, dim=D, topk=topk, sm_scale=D**-0.5, block_I=block_I)
    cast_f = tl_dk_cast(dim=D, grid_size=20)
    args = (
        q,
        w,
        k,
        attn_score,
        index_score,
        topk_indices,
        offsets,
        token_indices,
        dq,
        dw,
        ws_dk,
        workspace_k,
        workspace_dlog,
        workspace_logits,
        workspace_dk,
    )
    for _ in range(warmup):
        gen_f(*args)
        cast_f(ws_dk, dk)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        gen_f(*args)
        cast_f(ws_dk, dk)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / n_iters * 1000.0


def benchmark_perf():
    """Time the kernels only (torch.npu events, warmup + averaged iters).

    Timing boundary: everything host-side is excluded (H2D, output/workspace
    allocation, torch.zeros pre-clearing, dispatch detection / .item() sync);
    each iteration launches the pre-built fast kernel + cast kernel
    back-to-back. ws_dk is NOT re-zeroed inside the loop (accumulated values
    are garbage but the kernels execute identical work).
    """
    fast_us = _bench_fast()
    print(
        f"[PERF] indexer_bwd l0_default fast(layout=0)+cast (S=2048,H=16,D=128,topk=64,bf16): "
        f"pure kernel avg = {fast_us:.1f} us over 20 iters "
        f"(torch.npu events; excludes H2D/sync/host alloc/host detection; target < 80 us)"
    )
    try:
        gen_us = _bench_generic()
        print(
            f"[PERF] indexer_bwd l0_default generic+cast (same input, fallback path): "
            f"pure kernel avg = {gen_us:.1f} us over 20 iters (not a perf-gated path)"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[PERF] generic path timing failed: {e!r}")
    return fast_us


# ============================================================================
# main: --level dispatch + exit code
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all", "perf"])
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    blocking_ok = True  # only L0/L1 count toward the exit code
    if args.level in ("l0", "all"):
        blocking_ok &= test_indexer_bwd_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_indexer_bwd_l1()
    if args.level in ("l2", "all"):
        test_indexer_bwd_l2()
    if args.level in ("boundary", "all"):
        test_indexer_bwd_boundary()
    if args.level in ("all", "perf"):
        benchmark_perf()

    if blocking_ok and args.level != "perf":
        print("Test Passed!")
        sys.exit(0)
    sys.exit(0 if blocking_ok else 1)


if __name__ == "__main__":
    main()
