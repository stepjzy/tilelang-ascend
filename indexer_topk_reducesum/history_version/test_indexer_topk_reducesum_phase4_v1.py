"""indexer_topk_reducesum layered tests: L0 gate (DESIGN §9.2, 10 cases) +
stubs + perf benchmark.

Golden replicates the main repo's ref_index_score
(ref_gpu/indexer_topk_reducesum.py:208-229, byte-for-byte per DESIGN §9.1)
and runs it on NPU. Input construction replicates the main repo test_kernel
(torch.manual_seed(42), randn bf16 q/w/k, int32 offsets; CPU randn + bf16
cast + one H2D per tensor).

Comparison strategy (DESIGN §9.2, topk dual output):
  1. ReduceSum (topk_score, fp32): full-position mixed tolerance (§9.3 fp32
     row); -inf tail positions are exact 0 on both sides.
  2. TopkIndices (int32): mask-domain compare, mask = ref_topk_score > 0
     (the main repo test L257 criterion); in-mask exact match with ratio
     >= 0.99 (near-tie adjacent swaps absorbed); -inf tail (-1 on our side,
     arbitrary on the golden side) is not compared. Intersection ratio is
     printed as a diagnostic (main repo L259-263 form).

Levels:
    l0        DESIGN §9.2 gate cases (blocking, [PRECISION_PASS/FAIL])
    l1/l2/boundary  stubs, expanded by tilelang-op-test-design (scenario B)
    perf      pure-kernel average latency for the l0_main configuration
              (fast CAP=2048 path; generic timed separately for the R3
              sort/topk calibration data point)
"""

import argparse
import os
import sys
import traceback

import torch
import torch.nn.functional as F

import tilelang

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indexer_topk_reducesum import (  # noqa: E402
    _detect_dispatch,
    _get_cube_core_num,
    _pick_kl0_size,
    indexer_topk_reducesum_interface,
    prepare_group_table,
    prepare_token_indices,
    tl_indexer_topk_reducesum_fast,
    tl_indexer_topk_reducesum_generic,
)


# ============================================================================
# Golden reference (main repo ref_gpu/indexer_topk_reducesum.py:208-229,
# verbatim per DESIGN §9.1; einops.einsum -> torch.einsum, cuda -> npu)
# ============================================================================
def ref_index_score(Q, Weights, K, topk, offsets, scale=None):
    all_topk_indices = []
    all_topk_score = []
    for i in range(offsets.shape[0] - 1):
        assert (offsets[i + 1] - offsets[i]).item() >= topk
        q = Q[offsets[i] : offsets[i + 1]]
        weights = Weights[offsets[i] : offsets[i + 1]]
        k = K[offsets[i] : offsets[i + 1]]
        softmax_scale = q.shape[-1] ** -0.5 if scale is None else scale
        s = q.shape[0]
        mask = (torch.arange(s)[:, None] >= torch.arange(s)[None, :]).to(q.device)
        # einops "s1 h k, s2 k -> s1 h s2" in single-letter torch.einsum form
        logits = torch.einsum("ahk,bk->ahb", q, k)  # bf16 output (golden round point)
        logits = F.relu(logits)
        logits = (logits * weights.unsqueeze(-1)).sum(dim=-2, dtype=torch.float32) * softmax_scale
        logits = torch.where(mask, logits, float("-inf"))
        topk_logits, topk_indices = torch.topk(logits, k=topk, dim=-1)
        topk_score = F.softmax(topk_logits, dim=-1, dtype=torch.float32)
        all_topk_indices.append(topk_indices)
        all_topk_score.append(topk_score)
    topk_indices = torch.cat(all_topk_indices, dim=0)
    topk_score = torch.cat(all_topk_score, dim=0)
    return topk_indices, topk_score


# ============================================================================
# Input construction (main repo test_kernel:232-244 replica; CPU randn + bf16
# cast + one H2D per tensor — no NPU-side random or cast ops)
# ============================================================================
def make_inputs(S, H, D, off_list, seed=42):
    torch.manual_seed(seed)
    q = torch.randn((S, H, D)).bfloat16().npu()
    w = torch.randn((S, H)).bfloat16().npu()
    k = torch.randn((S, D)).bfloat16().npu()
    offsets = torch.tensor(off_list, dtype=torch.int32).npu()
    return q, w, k, offsets


# ============================================================================
# Precision standard (mixed tolerance, DESIGN §9.3)
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
    """Mixed-tolerance dual-gate check: (passed, matched_ratio, max_abs_error)."""
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a, g = actual.detach().cpu(), golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:  # integer exact match
        mism = (a != g).sum().item()
        return mism == 0, 1.0 - mism / max(a.numel(), 1), (0.0 if mism == 0 else float("inf"))
    a, g = a.float(), g.float()
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


def check_topk_indices(actual, golden_indices, golden_score):
    """Mask-domain exact compare (DESIGN §9.2): mask = ref_score > 0.

    Returns (passed, mask_ratio, intersection_ratio).
    """
    a = actual.detach().cpu()
    g = golden_indices.detach().cpu().to(torch.int32)
    m = golden_score.detach().cpu() > 0
    total = int(m.sum().item())
    if total == 0:
        return True, 1.0, 1.0
    match = int((a[m] == g[m]).sum().item())
    mask_ratio = match / total
    # intersection diagnostic (main repo test L259-263 form)
    inter = 0
    ref_total = 0
    for r in range(g.shape[0]):
        mr = m[r]
        s_ref = set(g[r][mr].tolist())
        s_trt = set(a[r][mr].tolist())
        inter += len(s_ref & s_trt)
        ref_total += len(s_ref)
    inter_ratio = inter / max(ref_total, 1)
    return (mask_ratio >= 0.99), mask_ratio, inter_ratio


# ============================================================================
# Case runner
# ============================================================================
def run_case(name, S, H, D, topk, off_list, dispatch=None, seed=42, expect_dispatch=None):
    """Run one precision case: kernel vs golden on both outputs."""
    ok = True
    try:
        q, w, k, offsets = make_inputs(S, H, D, off_list, seed)
        if expect_dispatch is not None:
            got = _detect_dispatch(offsets)
            tag = "PASS" if got == expect_dispatch else "FAIL"
            print(f"[PRECISION_{tag}] {name} detect={got} (expect {expect_dispatch})")
            ok &= got == expect_dispatch
        ref_idx, ref_score = ref_index_score(q, w, k, topk, offsets)
        kwargs = {}
        if dispatch is not None:
            kwargs["dispatch"] = dispatch
        out_idx, out_score = indexer_topk_reducesum_interface(q, w, k, topk, offsets, **kwargs)
        passed, ratio, max_abs = check_precision(out_score, ref_score, "float32")
        tag = "PASS" if passed else "FAIL"
        print(
            f"[PRECISION_{tag}] {name} ReduceSum shape={tuple(out_score.shape)} dtype=float32 "
            f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e}"
        )
        ok &= passed
        passed, mask_ratio, inter_ratio = check_topk_indices(out_idx, ref_idx, ref_score)
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] {name} TopkIndices mask_domain_ratio={mask_ratio:.4f} intersection_ratio={inter_ratio:.4f}")
        ok &= passed
    except Exception as e:  # noqa: BLE001
        print(f"[PRECISION_FAIL] {name}: {e!r}")
        traceback.print_exc()
        ok = False
    return ok


# ============================================================================
# L0 gate tests (DESIGN §9.2 — 10 cases)
# ============================================================================
# (name, S, H, D, topk, offsets, dispatch)
L0_CONFIGS = [
    ("l0_main", 2048, 64, 128, 64, [0, 2048], None),  # main-repo input, fast CAP=2048
    ("l0_varlen", 2048, 64, 128, 64, [0, 1023, 2048], None),  # 1023+1025: tails + causal isolation
    ("l0_multibatch", 1536, 64, 128, 64, [0, 512, 1024, 1536], None),  # B=3, CAP=512 boundary
    ("l0_small", 128, 64, 128, 64, [0, 128], None),  # short tokens (i_u+1 < topk) dense
    ("l0_equal_topk", 64, 64, 128, 64, [0, 64], None),  # segment == topk (assert lower bound)
    ("l0_h16", 512, 16, 128, 64, [0, 512], None),  # different head count
    ("l0_topk32", 512, 64, 128, 32, [0, 512], None),  # different topk
    ("l0_generic", 5120, 64, 128, 64, [0, 5120], None),  # seg > 4096 -> generic, 5 CH segs
    ("l0_generic_tail", 4608, 64, 128, 64, [0, 4608], None),  # generic short last seg (4x1024+512)
]


def test_indexer_topk_reducesum_l0():
    """L0 gate: 9 shape/dispatch cases + dispatch-equivalence case (blocking)."""
    ok = True
    for name, S, H, D, topk, off_list, dispatch in L0_CONFIGS:
        ok &= run_case(name, S, H, D, topk, off_list, dispatch=dispatch)
    ok &= _test_dispatch_equiv()
    return ok


def _test_dispatch_equiv():
    """l0_dispatch_equiv: host detection assert + fast/generic forced paths vs
    golden + cross-path consistency (both paths on the same l0_main input)."""
    ok = True
    try:
        S, H, D, topk, off_list = 2048, 64, 128, 64, [0, 2048]
        q, w, k, offsets = make_inputs(S, H, D, off_list)
        ref_idx, ref_score = ref_index_score(q, w, k, topk, offsets)

        # host detection assert: max_seg_len = 2048 -> fast
        got = _detect_dispatch(offsets)
        tag = "PASS" if got == "fast" else "FAIL"
        print(f"[PRECISION_{tag}] l0_dispatch_equiv detect={got} (expect fast)")
        ok &= got == "fast"

        outs = {}
        for dispatch in ("fast", "generic"):
            out_idx, out_score = indexer_topk_reducesum_interface(q, w, k, topk, offsets, dispatch=dispatch)
            outs[dispatch] = (out_idx, out_score)
            passed, ratio, max_abs = check_precision(out_score, ref_score, "float32")
            tag = "PASS" if passed else "FAIL"
            print(f"[PRECISION_{tag}] l0_dispatch_equiv[{dispatch}] ReduceSum matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
            ok &= passed
            passed, mask_ratio, inter_ratio = check_topk_indices(out_idx, ref_idx, ref_score)
            tag = "PASS" if passed else "FAIL"
            print(
                f"[PRECISION_{tag}] l0_dispatch_equiv[{dispatch}] TopkIndices "
                f"mask_domain_ratio={mask_ratio:.4f} intersection_ratio={inter_ratio:.4f}"
            )
            ok &= passed

        # cross-path consistency (P1 equivalence form)
        fi, fs = outs["fast"]
        gi, gs = outs["generic"]
        passed, ratio, max_abs = check_precision(fs, gs, "float32")
        m = ref_score > 0
        iratio = (fi[m] == gi[m]).sum().item() / max(int(m.sum().item()), 1)
        tag = "PASS" if (passed and iratio >= 0.99) else "FAIL"
        print(f"[PRECISION_{tag}] l0_dispatch_equiv cross-path score_ratio={ratio:.4f} max_abs={max_abs:.3e} idx_agreement={iratio:.4f}")
        ok &= passed and iratio >= 0.99
    except Exception as e:  # noqa: BLE001
        print(f"[PRECISION_FAIL] l0_dispatch_equiv: {e!r}")
        traceback.print_exc()
        ok = False
    return ok


# ============================================================================
# L1/L2/Boundary: expanded by tilelang-op-test-design (scenario B)
# ============================================================================
def make_inputs_scaled(S, H, D, off_list, scale=1.0, w_shift=0.0, seed=42):
    """L1 input builder with value-range control (scale on q/w/k, shift on w)."""
    torch.manual_seed(seed)
    q = (torch.randn((S, H, D)) * scale).bfloat16().npu()
    w = (torch.randn((S, H)) * scale + w_shift).bfloat16().npu()
    k = (torch.randn((S, D)) * scale).bfloat16().npu()
    offsets = torch.tensor(off_list, dtype=torch.int32).npu()
    return q, w, k, offsets


def _run_l1(name, S, H, D, topk, off_list, scale=1.0, w_shift=0.0, sm_scale=None, dispatch=None, seed=42):
    """One L1 precision case: kernel vs golden on both outputs (blocking)."""
    try:
        q, w, k, offsets = make_inputs_scaled(S, H, D, off_list, scale, w_shift, seed)
        ref_idx, ref_score = ref_index_score(q, w, k, topk, offsets, scale=sm_scale)
        kwargs = {}
        if dispatch is not None:
            kwargs["dispatch"] = dispatch
        if sm_scale is not None:
            kwargs["sm_scale"] = sm_scale
        out_idx, out_score = indexer_topk_reducesum_interface(q, w, k, topk, offsets, **kwargs)
        ok = True
        passed, ratio, max_abs = check_precision(out_score, ref_score, "float32")
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] {name} ReduceSum shape={tuple(out_score.shape)} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        ok &= passed
        passed, mask_ratio, inter_ratio = check_topk_indices(out_idx, ref_idx, ref_score)
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] {name} TopkIndices mask_domain_ratio={mask_ratio:.4f} intersection_ratio={inter_ratio:.4f}")
        ok &= passed
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"[PRECISION_FAIL] {name}: {e!r}")
        traceback.print_exc()
        return False


# (name, S, H, D, topk, offsets, scale, w_shift, sm_scale, dispatch, tags)
# shape axes: QG-token groups (8/4) and BN=32 K-blocks both key off the
# segment length; dim=112 exercises the GEMM K-axis tail (kL0Tail=16).
L1_CASES = [
    (
        "l1_aligned_b2",
        1024,
        64,
        128,
        64,
        [0, 512, 1024],
        1.0,
        0.0,
        None,
        None,
        ["D-DTYPE-bf16", "D-DTYPE-int32", "D-SHAPE-ALIGNED", "D-PARAM-block_n"],
    ),
    ("l1_tail1", 513, 64, 128, 64, [0, 513], 1.0, 0.0, None, None, ["D-SHAPE-TAIL-1"]),  # 513 % 8 == 1 (QG tail) and % 32 == 1 (K tail)
    ("l1_tail_mid", 516, 64, 128, 64, [0, 516], 1.0, 0.0, None, None, ["D-SHAPE-TAIL-MID"]),  # 516 % 8 == 4 (half group), % 32 == 4
    ("l1_prime", 521, 64, 128, 64, [0, 521], 1.0, 0.0, None, None, ["D-SHAPE-PRIME"]),  # 521 prime: fully non-aligned on both tiling axes
    (
        "l1_edge_topk128",
        128,
        64,
        128,
        128,
        [0, 128],
        1.0,
        0.0,
        None,
        None,
        ["D-SHAPE-EDGE", "D-PARAM-topk"],
    ),  # segment == topk (assert lower bound)
    (
        "l1_varlen_mixed",
        2048,
        64,
        128,
        64,
        [0, 100, 1100, 2048],
        1.0,
        0.0,
        None,
        None,
        ["D-SHAPE-TAIL-MID"],
    ),  # segs 100/1000/948: 100 % 8 == 4, 948 % 8 == 4
    ("l1_h8", 256, 8, 128, 64, [0, 256], 1.0, 0.0, None, None, ["D-PARAM-heads"]),
    ("l1_h32", 256, 32, 128, 64, [0, 256], 1.0, 0.0, None, None, ["D-PARAM-heads"]),
    ("l1_dim64", 256, 64, 64, 64, [0, 256], 1.0, 0.0, None, None, ["D-PARAM-dim"]),
    (
        "l1_dim112_ktail",
        256,
        64,
        112,
        64,
        [0, 256],
        1.0,
        0.0,
        None,
        None,
        ["D-PARAM-dim"],
    ),  # K-axis tail: 112 = 3*32 + 16 (kL0Tail=16 @ kL0Size=32)
    ("l1_topk16", 128, 64, 128, 16, [0, 128], 1.0, 0.0, None, None, ["D-PARAM-topk"]),
    ("l1_topk1024", 1024, 64, 128, 1024, [0, 1024], 1.0, 0.0, None, None, ["D-PARAM-topk"]),
    (
        "l1_qg4_cap4096",
        2560,
        64,
        128,
        64,
        [0, 2560],
        1.0,
        0.0,
        None,
        None,
        ["D-PARAM-qg"],
    ),  # CAP=4096 tier -> QG=4 (non-default token-group size)
    ("l1_dispatch_fast", 1024, 64, 128, 64, [0, 1024], 1.0, 0.0, None, "fast", ["D-PARAM-dispatch"]),  # forced fast (non-default dispatch)
    ("l1_smscale", 512, 64, 128, 64, [0, 512], 1.0, 0.0, 0.5, None, ["D-PARAM-sm_scale"]),  # custom scale; golden uses the same scale
    ("l1_vrange_s", 512, 64, 128, 64, [0, 512], 0.3, 0.0, None, None, ["D-VALRANGE-S"]),  # ~[-1, 1]
    ("l1_vrange_m", 512, 64, 128, 64, [0, 512], 3.0, 0.0, None, None, ["D-VALRANGE-M"]),  # ~[-10, 10]
    (
        "l1_vrange_l",
        512,
        64,
        128,
        64,
        [0, 512],
        1000.0,
        0.0,
        None,
        None,
        ["D-VALRANGE-L"],
    ),  # large magnitudes (scores ~1e9, softmax one-hot)
    ("l1_vrange_asym", 512, 64, 128, 64, [0, 512], 3.0, 5.0, None, None, ["D-VALRANGE-ASYM"]),  # w in ~[-5, 15] (asymmetric)
    (
        "l1_b8_smallsegs",
        512,
        64,
        128,
        64,
        [0, 64, 128, 192, 256, 320, 384, 448, 512],
        1.0,
        0.0,
        None,
        None,
        ["D-SHAPE-ALIGNED"],
    ),  # B=8 equal segments of 64
    (
        "l1_generic_varlen",
        4296,
        64,
        128,
        64,
        [0, 100, 4296],
        1.0,
        0.0,
        None,
        None,
        ["D-SHAPE-TAIL-MID"],
    ),  # generic: segs 100 and 4196 = 4*1024 + 100 (short last)
]


def test_indexer_topk_reducesum_l1():
    """L1 functional tests (varlen multi-seg / H / topk / B / length edges).

    Blocking: any [PRECISION_FAIL] fails the suite.
    """
    ok = True
    for name, S, H, D, topk, off_list, scale, w_shift, sm_scale, dispatch, _tags in L1_CASES:
        ok &= _run_l1(
            name,
            S,
            H,
            D,
            topk,
            off_list,
            scale=scale,
            w_shift=w_shift,
            sm_scale=sm_scale,
            dispatch=dispatch,
        )
    return ok


# ---- L2: negative tests (illegal inputs must be rejected; non-blocking) ----
def _l2_inputs(S, H, D, dtype_q=torch.bfloat16, dtype_w=torch.bfloat16, dtype_k=torch.bfloat16, dtype_off=torch.int32):
    q = torch.randn((S, H, D)).to(dtype_q).npu()
    w = torch.randn((S, H)).to(dtype_w).npu()
    k = torch.randn((S, D)).to(dtype_k).npu()
    offsets = torch.tensor([0, S], dtype=dtype_off).npu()
    return q, w, k, offsets


def _l2_call(t, topk=64, **kw):
    return indexer_topk_reducesum_interface(t[0], t[1], t[2], topk, t[3], **kw)


def _run_l2(name, fn, tags):
    """L2 negative case: fn() feeds illegal input, expects a rejection.

    Correctly raised exception -> [BOUNDARY_PASS]; silent acceptance ->
    [BOUNDARY_WARN]. Never blocking.
    """
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_PASS] l2 {name}: rejected ({type(e).__name__}: {e})")
        return
    print(f"[BOUNDARY_WARN] l2 {name}: illegal input silently accepted")


def _l2_cases():
    """Build the L2 case list (deferred so tensors are created lazily)."""
    return [
        ("l2_fp16_q", ["D-EXC-DTYPE"], lambda: _l2_call(_l2_inputs(128, 64, 128, 64, dtype_q=torch.float16))),
        ("l2_fp32_k", ["D-EXC-DTYPE"], lambda: _l2_call(_l2_inputs(128, 64, 128, 64, dtype_k=torch.float32))),
        ("l2_int64_offsets", ["D-EXC-DTYPE"], lambda: _l2_call(_l2_inputs(128, 64, 128, 64, dtype_off=torch.int64))),
        ("l2_heads12", ["D-EXC-SHAPE"], lambda: _l2_call(_l2_inputs(128, 12, 128, 64))),
        ("l2_heads72", ["D-EXC-SHAPE"], lambda: _l2_call(_l2_inputs(128, 72, 128, 64))),
        ("l2_dim100", ["D-EXC-SHAPE"], lambda: _l2_call(_l2_inputs(128, 64, 100, 64))),
        ("l2_topk48", ["D-EXC-SHAPE"], lambda: _l2_call(_l2_inputs(128, 64, 128, 64), topk=48)),
        ("l2_topk2048", ["D-EXC-SHAPE"], lambda: _l2_call(_l2_inputs(2048, 64, 128, 64), topk=2048)),
        (
            "l2_seg_lt_topk",
            ["D-EXC-SHAPE"],
            lambda: _l2_call(_l2_inputs(128, 64, 128, 64)[:3] + (torch.tensor([0, 32], dtype=torch.int32).npu(),)),
        ),
        (
            "l2_rank_mismatch",
            ["D-EXC-SHAPE"],
            lambda: _l2_call(
                (
                    torch.randn(128, 64 * 128).bfloat16().npu(),
                    torch.randn(128, 64).bfloat16().npu(),
                    torch.randn(128, 128).bfloat16().npu(),
                    torch.tensor([0, 128], dtype=torch.int32).npu(),
                )
            ),
        ),
        (
            "l2_shape_mismatch",
            ["D-EXC-SHAPE"],
            lambda: _l2_call(
                (
                    torch.randn(128, 64, 128).bfloat16().npu(),
                    torch.randn(128, 72).bfloat16().npu(),
                    torch.randn(128, 128).bfloat16().npu(),
                    torch.tensor([0, 128], dtype=torch.int32).npu(),
                )
            ),
        ),
        ("l2_dispatch_bogus", ["D-EXC-SHAPE"], lambda: _l2_call(_l2_inputs(128, 64, 128, 64), dispatch="bogus")),
        (
            "l2_fast_on_generic",
            ["D-EXC-SHAPE"],
            lambda: _l2_call(
                _l2_inputs(5120, 64, 128, 64)[:3] + (torch.tensor([0, 5120], dtype=torch.int32).npu(),),
                dispatch="fast",
            ),
        ),
    ]


def test_indexer_topk_reducesum_l2():
    """L2 negative tests: violations of the host-validated domain asserts."""
    for name, tags, fn in _l2_cases():
        _run_l2(name, fn, tags)


# ---- Boundary: legal special values (precision compared; non-blocking) ----
def _run_boundary(name, fn):
    """Boundary case: fn() -> (out_idx, out_score, ref_idx, ref_score).

    Compared with the same mixed-tolerance standard as L0/L1 (fp32 row for
    ReduceSum, mask-domain exact for TopkIndices); failure -> [BOUNDARY_WARN].
    """
    try:
        out_idx, out_score, ref_idx, ref_score = fn()
        passed, ratio, max_abs = check_precision(out_score, ref_score, "float32")
        p2, mask_ratio, inter_ratio = check_topk_indices(out_idx, ref_idx, ref_score)
        if passed and p2:
            print(f"[BOUNDARY_PASS] boundary {name} score_ratio={ratio:.4f} max_abs={max_abs:.3e} idx_mask_ratio={mask_ratio:.4f}")
        else:
            print(
                f"[BOUNDARY_WARN] boundary {name} score_ratio={ratio:.4f} "
                f"max_abs={max_abs:.3e} idx_mask_ratio={mask_ratio:.4f} inter={inter_ratio:.4f}"
            )
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_WARN] boundary {name}: {e!r}")


def _boundary_case(S, H, D, topk, mutate):
    """Build special-value inputs, run kernel + golden, return both outputs."""
    q, w, k, offsets = make_inputs(S, H, D, [0, S], seed=7)
    q, w, k = mutate(q.clone(), w.clone(), k.clone())
    ref_idx, ref_score = ref_index_score(q, w, k, topk, offsets)
    out_idx, out_score = indexer_topk_reducesum_interface(q, w, k, topk, offsets)
    return out_idx, out_score, ref_idx, ref_score


def test_indexer_topk_reducesum_boundary():
    """Boundary special-value tests: all-zero / inf / nan / dtype bound."""
    S, H, D, topk = 512, 64, 128, 64

    def all_zero(q, w, k):
        return torch.zeros_like(q), torch.zeros_like(w), torch.zeros_like(k)

    def inf_q(q, w, k):
        q[0, 0, :] = float("inf")
        q[1, 1, :] = float("-inf")
        return q, w, k

    def nan_k(q, w, k):
        k[0, :] = float("nan")
        k[7, 3:11] = float("nan")
        return q, w, k

    def dbound(q, w, k):
        q[0, 0, :] = torch.finfo(torch.bfloat16).max
        k[0, :] = -torch.finfo(torch.bfloat16).max
        return q, w, k

    _run_boundary("all_zero", lambda: _boundary_case(S, H, D, topk, all_zero))
    _run_boundary("inf_q", lambda: _boundary_case(S, H, D, topk, inf_q))
    _run_boundary("nan_k", lambda: _boundary_case(S, H, D, topk, nan_k))
    _run_boundary("dbound_qk", lambda: _boundary_case(S, H, D, topk, dbound))


# ---- Coverage annotations (machine-checkable; see coverage-matrix.md) ----
COVERAGE_CATEGORY = "Fusion"  # CV mixed: gemm_v0 + relu/weighted-reduce/mask + topk + softmax

COVERAGE_MANIFEST = {
    "D-DTYPE-bf16": 21,
    "D-DTYPE-int32": 21,
    "D-SHAPE-ALIGNED": 2,
    "D-SHAPE-TAIL-1": 1,
    "D-SHAPE-TAIL-MID": 3,
    "D-SHAPE-PRIME": 1,
    "D-SHAPE-EDGE": 1,
    "D-VALRANGE-S": 1,
    "D-VALRANGE-M": 1,
    "D-VALRANGE-L": 1,
    "D-VALRANGE-ASYM": 1,
    "D-PARAM-heads": 2,
    "D-PARAM-dim": 2,
    "D-PARAM-topk": 3,
    "D-PARAM-sm_scale": 1,
    "D-PARAM-dispatch": 1,
    "D-PARAM-qg": 1,
    "D-PARAM-block_n": 1,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-INF": 1,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-DBOUND": 1,
    "D-EXC-DTYPE": 3,
    "D-EXC-SHAPE": 10,
}

COVERAGE_NA = {}  # Fusion category: every applicable dimension is covered


# ============================================================================
# Perf benchmark: pure-kernel average latency (l0_main configuration)
# ============================================================================
def _bench_kernel(mode, S=2048, H=64, D=128, topk=64, off_list=None, n_iters=20, warmup=5):
    """Time ONE pre-built kernel with pre-allocated args (torch.npu events).

    Timing boundary: excludes H2D, output/workspace allocation, host dispatch
    detection (.item() sync) — everything host-side. Each iteration launches
    only the pre-built kernel.
    """
    if off_list is None:
        off_list = [0, S]
    q, w, k, offsets = make_inputs(S, H, D, off_list)
    grid = _get_cube_core_num()
    token_indices = prepare_token_indices(offsets)
    out_idx = torch.zeros((S, topk), dtype=torch.int32, device=q.device)
    out_score = torch.zeros((S, topk), dtype=torch.float32, device=q.device)
    QG = 8
    kl0 = _pick_kl0_size(QG * H, D)
    group_table = prepare_group_table(offsets, QG)
    # [grid, 2, BN, QG*H] transposed transit layout (perf iter-2; same bytes,
    # same timing boundary -- only the memory orientation matches the kernel)
    ws = torch.empty((grid, 2, 32, QG * H), dtype=torch.bfloat16, device=q.device)
    ws_ack = torch.empty((grid, 32), dtype=torch.float32, device=q.device)
    if mode == "fast":
        func = tl_indexer_topk_reducesum_fast(heads=H, dim=D, topk=topk, CAP=2048, QG=QG, grid_size=grid, KL0=kl0)
    else:
        NSEG = (max(S - 0, 1) + 1024 - 1) // 1024
        func = tl_indexer_topk_reducesum_generic(heads=H, dim=D, topk=topk, CH=1024, NSEG=NSEG, QG=QG, grid_size=grid, KL0=kl0)
    args = (
        q.reshape(S * H, D),
        w.reshape(-1),
        k,
        out_idx,
        out_score,
        offsets,
        token_indices,
        group_table,
        ws,
        ws_ack,
    )
    for _ in range(warmup):
        func(*args)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        func(*args)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / n_iters * 1000.0


def _bench_interface(S=2048, H=64, D=128, topk=64, off_list=None, n_iters=20, warmup=5):
    """Time the full host interface (dispatch detection + prep + kernel)."""
    if off_list is None:
        off_list = [0, S]
    q, w, k, offsets = make_inputs(S, H, D, off_list)
    for _ in range(warmup):
        indexer_topk_reducesum_interface(q, w, k, topk, offsets)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        indexer_topk_reducesum_interface(q, w, k, topk, offsets)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / n_iters * 1000.0


def benchmark_perf():
    """l0_main baseline (Stage 3 target: < 1428 us fast path)."""
    fast_us = _bench_kernel("fast")
    print(
        f"[PERF] indexer_topk_reducesum l0_main fast (S=2048,H=64,D=128,topk=64,bf16,CAP=2048,QG=8): "
        f"pure kernel avg = {fast_us:.1f} us over 20 iters (torch.npu events; excludes "
        f"H2D/sync/host alloc/host detection; target < 1428 us)"
    )
    try:
        gen_us = _bench_kernel("generic")
        print(
            f"[PERF] indexer_topk_reducesum l0_main generic (same input, CH=1024x2): "
            f"pure kernel avg = {gen_us:.1f} us over 20 iters (not perf-gated; R3 sort/topk "
            f"calibration point: generic adds per-segment topk + merge_sort)"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[PERF] generic path timing failed: {e!r}")
    try:
        iface_us = _bench_interface()
        print(
            f"[PERF] indexer_topk_reducesum l0_main full interface (host dispatch + prep + kernel): avg = {iface_us:.1f} us over 20 iters"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[PERF] interface timing failed: {e!r}")
    return fast_us


# ============================================================================
# main: --level dispatch + exit code
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "L0", "l1", "L1", "l2", "L2", "boundary", "all", "perf"])
    args = parser.parse_args()
    level = args.level.lower()

    tilelang.disable_cache()
    torch.manual_seed(0)

    blocking_ok = True  # only L0/L1 count toward the exit code
    if level in ("l0", "all"):
        blocking_ok &= test_indexer_topk_reducesum_l0()
    if level in ("l1", "all"):
        blocking_ok &= test_indexer_topk_reducesum_l1()
    if level in ("l2", "all"):
        test_indexer_topk_reducesum_l2()  # non-blocking
    if level == "boundary" or level == "all":
        test_indexer_topk_reducesum_boundary()  # non-blocking
    if level in ("all", "perf"):
        benchmark_perf()

    if blocking_ok and level != "perf":
        print("Test Passed!")
        sys.exit(0)
    sys.exit(0 if blocking_ok else 1)


if __name__ == "__main__":
    main()
