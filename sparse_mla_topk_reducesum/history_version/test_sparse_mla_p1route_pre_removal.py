"""Layered test for sparse_mla_topk_reducesum — generic golden + panel-mode + auto-dispatch L0 gate.

Layers:
  * L0-P1   golden config through the panel path (panel_mode="shared"): vs the
             pure-torch reference (golden criteria verbatim: err_ratio <= 1e-2 AND
             abs_err <= 1e-3 AND NaN == 0) AND bitwise-equal (torch.equal) to the
             generic-path output (DESIGN_panel_mode §9.1 equivalence evidence).
  * L0-P2   generic-path regression (rev2 revision: EXPLICIT panel_mode="generic"
             — the default is now "auto", so the unguarded-generic regression
             target must be pinned): the original GPU golden case verbatim
             (B=1 / S=2048 / H=16 / D=512 / tail_D=64 / topk=128 / seed=42, both
             block_I 32/128) — proves the unguarded path is unchanged (binary
             preservation, DESIGN_panel_mode rev2 §3.5).
  * L0-P3   contract-violation detection: one row's index differs from the shared
             panel vector -> ErrFlag set + host RuntimeError (no silent wrong
             results); the same input via explicit generic is correct.
  * L0-P4a  empty first sequence (offsets=[0,0,1023,2048]) via the panel path.
  * L0-P4b  ΣS=1 / topk=32 single task: panel kernel called directly through the
             builder (the host gate G5 would legitimately fall back to generic at
             this amortization ratio) + generic host call — both correct, bitwise
             equal; cores with ntask==0 must not hang.
  * L0-P4c  shared vector containing -1 (tail padding): clamp+mask semantics.
  * L0-P4d  out-of-range index value (clamp protects addressing, mask zeroes it).
  * L0-P4e  short-seq dense input (B=32): G4 gate miss -> silent generic fallback
             (proven by panel-key absence in _BUILD_CACHE + correct result).
  * L0-P5   race-stress family (DESIGN §6.2/#22): ΣS=64 × D∈{128,256} ×
             topk∈{32,64} × 3 independent runs — each run vs reference AND the 3
             runs pairwise torch.equal (bitwise stability across the group-change
             × lookahead × slot-reuse race surface).
  * L0-P6   host gate unit: topk=256 (G2/G3 v1-domain miss), B×G>8 (G4 miss),
             ΣS/B<16 (G5 miss) each route to generic (panel-key absence) with
             correct results.
  * L0-P7a  detect-kernel unit (rev2 §9.2): partition boundaries ±1, violation
             at first/last row, multi-violation accumulation, empty/single-row
             seq, ΣS=1, kv_group=2 ∃g semantics — violations int32 exact
             assertion vs the host lane-partition predicate.
  * L0-P7b  auto end-to-end on a panel-eligible input (mode NOT declared):
             golden criteria + torch.equal vs explicit generic + triple-launch
             routing probe (unguarded generic == 0) + wall timing signal.
  * L0-P7c  auto on random indices (metadata passes, contract broken):
             generic(guarded) executes correctly; triple-launch probe; direct
             detect finds violations > 0.
  * L0-P7d  auto metadata-gated-out (topk=256, not declared): unguarded generic
             single-shot — zero detect/guarded builder calls + cache keys absent.
  * L0-P7e  guarded-generic regression + race re-run: violations preset [1]
             (forced execute, L0-P5 family ×3 runs bitwise) / [0] (forced
             early-exit, sentinel output preserved).
  * L0-P7f  three-mode semantics: {auto, generic, shared} pairwise torch.equal
             + per-mode routing probes.
  * L0-P7g  detect race stress: ΣS=16384 multi-chunk ×3 runs, contract/violated
             — violations bitwise identical + exact predicate match.

L1/L2/Boundary layers follow the scenario-B extension (tilelang-op-test-design).

Run from repo root:
    source set_env.sh && python examples/dsa_sparse_finetune/test_sparse_mla_topk_reducesum.py --level l0
"""

import argparse
import contextlib
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F

import tilelang

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sparse_mla_topk_reducesum as _kernel_mod  # noqa: E402
from sparse_mla_topk_reducesum import (  # noqa: E402
    get_cube_core_num,
    prepare_token_indices,
    sparse_mla_topk_reducesum,
    tl_sparse_mla_topk_reducesum_detect,
    tl_sparse_mla_topk_reducesum_guarded,
    tl_sparse_mla_topk_reducesum_panel,
)

# PASS thresholds (golden-eval task spec): err_ratio <= 1e-2 AND abs_err <= 1e-3.
# Raw values are always printed; near-boundary results are flagged for adjudication.
ERR_RATIO_MAX = 1e-2
ABS_ERR_MAX = 1e-3
# Mixed-tolerance gate (DESIGN_panel_mode §9.3, ReduceSum float32 row): per-element
# |actual - ref| <= atol + rtol*|ref| with matched_ratio >= 0.99, jointly with the
# golden criteria above (max_abs_error_limit == ABS_ERR_MAX).
ATOL = 1e-4
RTOL = 1e-2
MATCHED_RATIO_MIN = 0.99

WARMUP_ITERS = 10
TIMING_ITERS = 50


# ===========================================================================
# Error metrics — ported verbatim from the tile-ai main-repo utils.py
# (GPU golden test utility; get_abs_err is exactly the max absolute difference).
# ===========================================================================
def get_abs_err(y, x):
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    return (x - y).flatten().abs().max().item()


def get_err_ratio(y, x):
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    err = (x - y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


# ===========================================================================
# Reference implementations — ported from the GPU main-repo ref_mla_topk_softmax.
# Pure torch: fp32 softmax, per-batch varlen loop + causal mask + gather +
# renormalize; also returns the fp32 lse consumed by the kernel under test.
# ===========================================================================
def ref_mla_topk_softmax(Q, K, TopkIndices, offsets):
    # q: [batch, seq_len, heads, dim]; k: [batch, seq_len, dim]
    sm_scale = Q.shape[-1] ** -0.5
    all_lse = []
    all_topk_score = []
    for i in range(offsets.shape[0] - 1):
        q = Q[offsets[i] : offsets[i + 1]]
        k = K[offsets[i] : offsets[i + 1]]
        topk_indices = TopkIndices[offsets[i] : offsets[i + 1]]
        seq_len = q.shape[0]
        causal = torch.arange(seq_len)[:, None] >= torch.arange(seq_len)[None, :]
        mask = causal.unsqueeze(-2).npu()  # .cuda() in the GPU original
        # einops.einsum multi-letter axes ("s1 h d, s2 d -> s1 h s2") are rejected
        # by torch.einsum — renamed to single letters (a=s1, b=s2), same contraction
        logits = torch.einsum("ahd,bd->ahb", q, k) * sm_scale
        logits = torch.where(mask, logits, float("-inf"))
        score = F.softmax(logits, dim=-1, dtype=torch.float32)
        score_sum = score.sum(dim=-2)
        topk_score = torch.gather(score_sum, dim=-1, index=topk_indices.to(torch.int64))
        topk_score = topk_score / topk_score.sum(dim=-1, keepdim=True)
        max_logits = logits.amax(dim=-1).to(torch.float32)
        shifted = (logits - max_logits.unsqueeze(-1).to(torch.float32)).exp().sum(dim=-1)
        lse = torch.log(shifted) + max_logits
        all_lse.append(lse)
        all_topk_score.append(topk_score)
    lse = torch.cat(all_lse, dim=0)
    topk_score = torch.cat(all_topk_score, dim=0)
    return lse, topk_score


def ref_mla_topk_softmax_masked(Q, K, TopkIndices, offsets, sm_scale=None):
    """ref_mla_topk_softmax + out-of-range index handling.

    The panel/generic kernels treat index values outside [0, seq_len) (including
    -1) as causally-masked zero contributions (the address is clamped purely for
    memory safety). The plain reference would gather out-of-bounds on such
    indices, so this variant clamps for the gather and zeroes invalid positions —
    identical to the plain ref when all indices are in range.

    sm_scale: optional explicit scale (defaults to the shape-derived value, same
    as the kernel host entry).
    """
    if sm_scale is None:
        sm_scale = Q.shape[-1] ** -0.5
    all_lse = []
    all_topk_score = []
    for i in range(offsets.shape[0] - 1):
        q = Q[offsets[i] : offsets[i + 1]]
        k = K[offsets[i] : offsets[i + 1]]
        topk_indices = TopkIndices[offsets[i] : offsets[i + 1]]
        seq_len = q.shape[0]
        if seq_len == 0:
            continue  # empty sequence: zero rows in the packed output (NPU amax
            # cannot reduce a zero-size dim, so skip instead of computing)
        causal = torch.arange(seq_len)[:, None] >= torch.arange(seq_len)[None, :]
        mask = causal.unsqueeze(-2).npu()
        # fp32 einsum: bf16-input einsum carries ~1e-3 rounding that would mask
        # the kernel's true (fp32-accumulated) accuracy; both operands cast
        # before contraction (test-side authority upgrade, plain ref untouched)
        logits = torch.einsum("ahd,bd->ahb", q.float(), k.float()) * sm_scale
        logits = torch.where(mask, logits, float("-inf"))
        score = F.softmax(logits, dim=-1, dtype=torch.float32)
        score_sum = score.sum(dim=-2)
        idx = topk_indices.to(torch.int64)
        valid = (idx >= 0) & (idx < seq_len)
        idx_safe = idx.clamp(0, max(seq_len - 1, 0))
        topk_score = torch.gather(score_sum, dim=-1, index=idx_safe) * valid.to(torch.float32)
        topk_score = topk_score / topk_score.sum(dim=-1, keepdim=True)
        max_logits = logits.amax(dim=-1).to(torch.float32)
        shifted = (logits - max_logits.unsqueeze(-1).to(torch.float32)).exp().sum(dim=-1)
        lse = torch.log(shifted) + max_logits
        all_lse.append(lse)
        all_topk_score.append(topk_score)
    lse = torch.cat(all_lse, dim=0)
    topk_score = torch.cat(all_topk_score, dim=0)
    return lse, topk_score


def ref_mla_topk_softmax_grouped(Q, K, TopkIndices, offsets, kv_group, sm_scale=None):
    """kv_group > 1 reference: group g serves heads [g*head_kv, (g+1)*head_kv)
    against the g-th kv stream (kernel semantics: H0 = g * head_kv).

    Returns (lse [ΣS, heads], topk_score [ΣS, kv_group, topk]) — the normalized
    attn_score per group, matching the kernel's per-group topk renormalization.
    Out-of-range indices are masked exactly like ref_mla_topk_softmax_masked.
    """
    if sm_scale is None:
        sm_scale = Q.shape[-1] ** -0.5
    head_kv = Q.shape[1] // kv_group
    all_lse = []
    all_topk_score = []
    for i in range(offsets.shape[0] - 1):
        q = Q[offsets[i] : offsets[i + 1]]
        k = K[offsets[i] : offsets[i + 1]]
        topk_indices = TopkIndices[offsets[i] : offsets[i + 1]]
        seq_len = q.shape[0]
        if seq_len == 0:
            continue
        causal = torch.arange(seq_len)[:, None] >= torch.arange(seq_len)[None, :]
        mask = causal.unsqueeze(-2).npu()
        batch_lse = []
        batch_score = []
        for g in range(kv_group):
            q_g = q[:, g * head_kv : (g + 1) * head_kv, :]  # [seq, head_kv, D]
            k_g = k[:, g, :]  # [seq, D]
            idx_g = topk_indices[:, g, :]  # [seq, topk]
            logits = torch.einsum("ahd,bd->ahb", q_g.float(), k_g.float()) * sm_scale
            logits = torch.where(mask, logits, float("-inf"))
            score = F.softmax(logits, dim=-1, dtype=torch.float32)
            score_sum = score.sum(dim=-2)  # [seq, seq]
            idx = idx_g.to(torch.int64)
            valid = (idx >= 0) & (idx < seq_len)
            idx_safe = idx.clamp(0, max(seq_len - 1, 0))
            topk_score = torch.gather(score_sum, dim=-1, index=idx_safe) * valid.to(torch.float32)
            topk_score = topk_score / topk_score.sum(dim=-1, keepdim=True)
            max_logits = logits.amax(dim=-1).to(torch.float32)  # [seq, head_kv]
            shifted = (logits - max_logits.unsqueeze(-1).to(torch.float32)).exp().sum(dim=-1)
            lse_g = torch.log(shifted) + max_logits  # [seq, head_kv]
            batch_lse.append(lse_g)  # [seq, head_kv]
            batch_score.append(topk_score)  # [seq, topk]
        all_lse.append(torch.cat(batch_lse, dim=1))  # [seq, heads]
        all_topk_score.append(torch.stack(batch_score, dim=1))  # [seq, G, topk]
    lse = torch.cat(all_lse, dim=0)
    topk_score = torch.cat(all_topk_score, dim=0)
    return lse, topk_score


# ===========================================================================
# Case helpers
# ===========================================================================
def make_case(S, H, D, tail_D, topk, offsets_list, seed=42, index_vec=None):
    """Build one golden-family case: CPU prep + one H2D per tensor (aclnn audit).

    index_vec: None -> shared P = arange(topk) (panel contract holds); otherwise
    a 1-D int32 tensor of length topk used as the shared P for every row.
    Returns (q4, kv4, idx4, lse, offsets) with kernel-ready shapes.
    """
    torch.manual_seed(seed)
    q = torch.randn((S, H, D + tail_D)).npu().bfloat16()
    kv = torch.randn((S, D + tail_D)).npu().bfloat16()
    offsets = torch.tensor(offsets_list, dtype=torch.int32).npu()
    if index_vec is None:
        p = torch.arange(topk, dtype=torch.int32)
    else:
        p = index_vec.to(torch.int32).clone()
    topk_indices = p.unsqueeze(0).repeat(S, 1).contiguous().npu()
    return q, kv, topk_indices, offsets


def reducesum_to_attn(reducesum):
    """Interface-level normalization (GPU sparse_mla_topk_reducesum_interface):
    ReduceSum [S, G, RH, topk] --sum over RH--> [S, G, topk] --renorm over topk-->
    attn_score, squeeze G=1 -> [S, topk]."""
    rs = reducesum.sum(dim=-2)
    return (rs / rs.sum(dim=-1, keepdim=True)).squeeze(-2)


def check_vs_ref(name, reducesum, ref_attn_score, blocking=True):
    """Golden-criteria + §9.3 mixed-tolerance comparison of a kernel ReduceSum
    output vs the reference attn_score. Prints the layered marker; returns passed."""
    attn = reducesum_to_attn(reducesum).float()
    ref = ref_attn_score.float()
    abs_err = get_abs_err(attn, ref)
    err_ratio = get_err_ratio(attn, ref)
    nan_count = int(torch.isnan(attn).sum().item())
    diff = (attn - ref).abs()
    matched_ratio = (diff <= ATOL + RTOL * ref.abs()).float().mean().item()
    ok = (err_ratio <= ERR_RATIO_MAX) and (abs_err <= ABS_ERR_MAX) and nan_count == 0 and matched_ratio >= MATCHED_RATIO_MIN
    if blocking:
        tag = "[PRECISION_PASS]" if ok else "[PRECISION_FAIL]"
        print(
            f"{tag} {name}: err_ratio={err_ratio:.3e} (<= {ERR_RATIO_MAX:g}), "
            f"abs_err={abs_err:.3e} (<= {ABS_ERR_MAX:g}), NaN={nan_count}, "
            f"matched_ratio={matched_ratio:.6f} (>= {MATCHED_RATIO_MIN:g})"
        )
    else:
        tag = "[BOUNDARY_PASS]" if ok else "[BOUNDARY_WARN]"
        print(f"{tag} {name}: err_ratio={err_ratio:.3e}, abs_err={abs_err:.3e}, NaN={nan_count}, matched_ratio={matched_ratio:.6f}")
    return ok


def check_equal(name, out_a, out_b):
    """Bitwise equivalence check (panel-vs-generic, DESIGN §9.1)."""
    eq = torch.equal(out_a, out_b)
    tag = "[PRECISION_PASS]" if eq else "[PRECISION_FAIL]"
    extra = "" if eq else f" max_diff={(out_a.float() - out_b.float()).abs().max().item():.3e}"
    print(f"{tag} {name}: torch.equal={eq}{extra}")
    return eq


def panel_cache_key(heads, dim_v, tail_dim, topk, kv_group, grid_size):
    """The _BUILD_CACHE key the host would insert for a panel-path compile."""
    return (heads, dim_v, tail_dim, topk, kv_group, None, "panel", grid_size)


def detect_cache_key(topk, kv_group, grid_size):
    """The _BUILD_CACHE key for the detect kernel (DESIGN_panel_mode §8.3)."""
    return (topk, kv_group, "detect", grid_size)


def guarded_generic_cache_key(heads, dim_v, tail_dim, topk, kv_group, block_I, grid_size):
    """The _BUILD_CACHE key for the guarded generic variant (§8.3)."""
    return (heads, dim_v, tail_dim, topk, kv_group, None, block_I, grid_size, "guarded")


def guarded_panel_cache_key(heads, dim_v, tail_dim, topk, kv_group, grid_size):
    """The _BUILD_CACHE key for the guarded panel variant (§8.3)."""
    return (heads, dim_v, tail_dim, topk, kv_group, None, "panel", grid_size, "guarded")


def _cache_key_kind(key):
    """Classify a _BUILD_CACHE tuple key into its builder-variant name."""
    if not isinstance(key, tuple):
        return None
    if len(key) == 4 and key[2] == "detect":
        return "detect"
    if len(key) >= 2 and key[-1] == "guarded":
        return "panel_guarded" if key[-3] == "panel" else "generic_guarded"
    if len(key) >= 2 and key[-2] == "panel":
        return "panel"
    if len(key) == 8:
        return "generic"
    return None


_BUILDERS = {
    "generic": "tl_sparse_mla_topk_reducesum",
    "panel": "tl_sparse_mla_topk_reducesum_panel",
    "generic_guarded": "tl_sparse_mla_topk_reducesum_guarded",
    "panel_guarded": "tl_sparse_mla_topk_reducesum_panel_guarded",
    "detect": "tl_sparse_mla_topk_reducesum_detect",
}


@contextlib.contextmanager
def auto_route_probe():
    """Count dispatches of all five builder variants (routing probe for the
    auto/generic/shared modes, DESIGN_panel_mode §9.2 P7 family).

    Counts both fresh compiles (builder invocation) and cached dispatches
    (pre-compiled JITKernel call), so a probe around a cache-hit call still
    observes the routing (P7b/P7c/P7f ordering-independence).
    """
    counts = {name: 0 for name in _BUILDERS}

    def _make_counting(func, name):
        def counting(*args, **kwargs):
            counts[name] += 1
            return func(*args, **kwargs)

        return counting

    saved_builders = {}
    for name, attr in _BUILDERS.items():
        orig = getattr(_kernel_mod, attr)
        saved_builders[name] = orig
        setattr(_kernel_mod, attr, _make_counting(orig, name))
    saved_cache_items = []
    try:
        for key, func in list(_kernel_mod._BUILD_CACHE.items()):
            kind = _cache_key_kind(key)
            if kind is None or not callable(func):
                continue
            saved_cache_items.append((key, func))
            _kernel_mod._BUILD_CACHE[key] = _make_counting(func, kind)
        yield counts
    finally:
        for name, attr in _BUILDERS.items():
            setattr(_kernel_mod, attr, saved_builders[name])
        for key, func in saved_cache_items:
            _kernel_mod._BUILD_CACHE[key] = func


@contextlib.contextmanager
def panel_builder_probe():
    """Count tl_sparse_mla_topk_reducesum_panel builder invocations (routing probe)."""
    original = _kernel_mod.tl_sparse_mla_topk_reducesum_panel
    calls = []

    def counting(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    _kernel_mod.tl_sparse_mla_topk_reducesum_panel = counting
    try:
        yield calls
    finally:
        _kernel_mod.tl_sparse_mla_topk_reducesum_panel = original


def check_no_panel_route(name, heads, dim_v, tail_dim, topk, kv_group, builder_calls):
    """Assert a gate-miss case actually routed to generic: no panel compile for
    its (unique) config key and no builder invocation."""
    grid = get_cube_core_num()
    key = panel_cache_key(heads, dim_v, tail_dim, topk, kv_group, grid)
    in_cache = key in _kernel_mod._BUILD_CACHE
    ok = (not in_cache) and (len(builder_calls) == 0)
    tag = "[PRECISION_PASS]" if ok else "[PRECISION_FAIL]"
    print(f"{tag} {name}: routed to generic (panel key in cache={in_cache}, builder calls={len(builder_calls)})")
    return ok


# ===========================================================================
# Coarse timing (precise profiling is deferred to the perf-tuner)
# ===========================================================================
def time_wall(fn, warmup=WARMUP_ITERS, iters=TIMING_ITERS):
    """Average wall-clock latency per call in ms (sync before/after each call)."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    total = 0.0
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        total += time.perf_counter() - t0
    return total / iters * 1e3


def time_event(fn, warmup=WARMUP_ITERS, iters=TIMING_ITERS):
    """Average per-call GPU-timeline span in ms via npu Events around a
    back-to-back loop (secondary cross-check; the host API syncs internally,
    so this may include host-side gaps)."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters


def time_config(q, kv, topk_indices, lse, offsets, dim_v, block_I):
    """Coarse timing for one block_I config: host-API kernel call (wall + event),
    token_indices preparation (public API, subtractable), and end-to-end with the
    interface-level normalization."""

    def call_kernel():
        return sparse_mla_topk_reducesum(
            q,
            kv,
            topk_indices,
            lse,
            offsets,
            dim_v=dim_v,
            block_I=block_I,
            panel_mode="generic",  # P2 修订：缺省已改 auto——回归对象 = unguarded 通用路径
        )

    def call_end2end():
        rs = call_kernel().sum(dim=-2)
        return rs / rs.sum(dim=-1, keepdim=True)

    t_api = time_wall(call_kernel)
    t_prep = time_wall(lambda: prepare_token_indices(offsets), warmup=2, iters=10)
    t_event = time_event(call_kernel)
    t_e2e = time_wall(call_end2end)
    return {
        "api_wall_ms": t_api,
        "prep_wall_ms": t_prep,
        "kernel_only_ms": max(t_api - t_prep, 0.0),
        "event_ms": t_event,
        "end2end_ms": t_e2e,
    }


# ===========================================================================
# Golden case — parameters verbatim from the GPU test_kernel defaults (L0-P2)
# ===========================================================================
def test_kernel(B=1, S=2048, H=16, D=512, tail_D=64, topk=128):
    """B is kept only for signature parity with the GPU original (unused there
    too); the actual batch segmentation comes from offsets = [0, 1023, 2048]."""
    torch.manual_seed(42)

    q = torch.randn((S, H, D + tail_D)).npu().bfloat16()
    kv = torch.randn((S, D + tail_D)).npu().bfloat16()
    offsets = torch.tensor([0, 1023, S], dtype=torch.int32).npu()

    # GPU original: repeat(torch.arange(topk, dtype=torch.int32).cuda(), "k -> s k", s=S)
    # einops unavailable -> identical torch tiling: result[s, k] = arange(topk)[k]
    arange_topk = torch.arange(topk, dtype=torch.int32).npu()
    topk_indices = arange_topk.unsqueeze(0).repeat(S, 1).contiguous()

    print("==== golden case (GPU test_kernel defaults) ====")
    print(f"B={B} S={S} H={H} D={D} tail_D={tail_D} topk={topk} seed=42")
    print(f"offsets={offsets.cpu().tolist()} (2 varlen segments: 1023 + 1025 tokens)")
    print(f"q: {tuple(q.shape)} {q.dtype}, kv: {tuple(kv.shape)} {kv.dtype}")
    print(f"topk_indices: {tuple(topk_indices.shape)} {topk_indices.dtype}")

    lse, ref_attn_score = ref_mla_topk_softmax(q, kv, topk_indices, offsets)
    torch.npu.synchronize()
    print(
        f"ref: lse {tuple(lse.shape)} {lse.dtype}, ref_attn_score "
        f"{tuple(ref_attn_score.shape)} {ref_attn_score.dtype}, "
        f"max={ref_attn_score.max().item():.6f}"
    )

    kv = kv.unsqueeze(-2)  # [S, 1, D+tail_D]
    topk_indices = topk_indices.unsqueeze(-2)  # [S, 1, topk]

    all_passed = True
    metrics = {}
    for block_I in (32, 128):  # 32: default; 128: tuned config (topk % 128 == 0)
        passed, m = eval_config(q, kv, topk_indices, lse, offsets, D, ref_attn_score, block_I)
        all_passed = all_passed and passed
        metrics[block_I] = m

    print("\n==== coarse timing (warmup + averaged, precise profiling deferred) ====")
    for block_I in (32, 128):
        t = time_config(q, kv, topk_indices, lse, offsets, D, block_I)
        metrics[block_I].update(t)
        print(
            f"block_I={block_I}: kernel host-API call = {t['api_wall_ms']:.3f} ms | "
            f"minus token_indices prep (~{t['prep_wall_ms']:.3f} ms) = "
            f"{t['kernel_only_ms']:.3f} ms | npu-event = {t['event_ms']:.3f} ms | "
            f"end-to-end (incl. normalize) = {t['end2end_ms']:.3f} ms "
            f"(warmup={WARMUP_ITERS}, iters={TIMING_ITERS})"
        )
    return all_passed, metrics


def eval_config(q, kv, topk_indices, lse, offsets, dim_v, ref_attn_score, block_I):
    """Run one block_I config, apply the GPU-interface normalization test-side,
    and report errors against the golden. Returns (passed, metrics dict)."""
    reducesum = sparse_mla_topk_reducesum(
        q,
        kv,
        topk_indices,
        lse,
        offsets,
        dim_v=dim_v,
        block_I=block_I,
        panel_mode="generic",  # P2 修订：显式 generic 钉住 unguarded 通用路径回归
    )
    # Interface-level normalization (same as the GPU sparse_mla_topk_reducesum_interface):
    #   reducesum [S, G, RH, topk] --sum over RH--> [S, G, topk] --renorm over topk-->
    #   attn_score, then squeeze the G=1 axis -> [S, topk]
    attn_score = reducesum_to_attn(reducesum)

    abs_err = get_abs_err(attn_score, ref_attn_score)  # == max abs diff
    err_ratio = get_err_ratio(attn_score, ref_attn_score)
    mean_abs_diff = (attn_score.float() - ref_attn_score.float()).abs().mean().item()
    nan_count = int(torch.isnan(attn_score).sum().item())
    inf_count = int(torch.isinf(attn_score).sum().item())

    print(f"\n---- block_I={block_I} ----")
    print(f"reducesum: shape={tuple(reducesum.shape)}, dtype={reducesum.dtype}")
    print(f"attn_score: shape={tuple(attn_score.shape)}, NaN={nan_count}, Inf={inf_count}")
    # Same line format as the GPU original test_kernel
    print(f"attn_score err: {abs_err:.6f} ratio: {err_ratio:.6f}")
    print(f"abs_err (max abs diff) = {abs_err:.6e}")
    print(f"err_ratio              = {err_ratio:.6e}")
    print(f"mean abs diff          = {mean_abs_diff:.6e}")

    passed = (err_ratio <= ERR_RATIO_MAX) and (abs_err <= ABS_ERR_MAX) and nan_count == 0
    if passed:
        print(
            f"[PRECISION_PASS] block_I={block_I}: err_ratio={err_ratio:.3e} <= {ERR_RATIO_MAX:g}, "
            f"abs_err={abs_err:.3e} <= {ABS_ERR_MAX:g}, NaN=0"
        )
    else:
        print(f"[PRECISION_FAIL] block_I={block_I}: criteria not met (raw values above)")
    if abs_err > 1e-4 or err_ratio > 1e-3:
        print("NOTE: error metrics are within 10x of the PASS thresholds —")
        print("raw values above are reported for orchestrator adjudication.")
    return passed, {
        "abs_err": abs_err,
        "err_ratio": err_ratio,
        "mean_abs_diff": mean_abs_diff,
        "nan_count": nan_count,
    }


# ===========================================================================
# L0 panel gate cases (DESIGN_panel_mode §9.2)
# ===========================================================================
def l0_p1_panel_golden():
    """L0-P1: golden config through the panel path + equivalence vs generic."""
    S, H, D, tail_D, topk = 2048, 16, 512, 64, 128
    q, kv, idx, offsets = make_case(S, H, D, tail_D, topk, [0, 1023, S])
    lse, ref_attn = ref_mla_topk_softmax(q, kv, idx, offsets)
    kv4 = kv.unsqueeze(-2)
    idx4 = idx.unsqueeze(-2)

    out_panel = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, panel_mode="shared")
    ok = check_vs_ref("L0-P1 panel vs torch-ref (golden criteria)", out_panel, ref_attn)

    out_g128 = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, block_I=128, panel_mode="generic")
    ok = check_equal("L0-P1 panel vs generic(block_I=128) torch.equal", out_panel, out_g128) and ok
    # secondary cross-check vs block_I=32 (report-only: the §9.1 equivalence
    # argument is per same-shape gemm; BI=32 splits topk into NI=4 blocks)
    out_g32 = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, block_I=32, panel_mode="generic")
    eq32 = torch.equal(out_panel, out_g32)
    print(f"[INFO] L0-P1 panel vs generic(block_I=32) torch.equal={eq32} (report-only)")

    # coarse timing (early signal for the 220us target; non-blocking)
    t_wall = time_wall(
        lambda: sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, panel_mode="shared"),
        warmup=5,
        iters=20,
    )
    t_event = time_event(
        lambda: sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, panel_mode="shared"),
        warmup=5,
        iters=20,
    )
    print(
        f"[INFO] L0-P1 panel host-API wall = {t_wall:.3f} ms, npu-event = {t_event:.3f} ms "
        "(incl. err_flag zeros + .item() check; precise profiling deferred to Stage 3)"
    )
    return ok


def l0_p2_generic_regression():
    """L0-P2: the original GPU golden case, both block_I configs (regression)."""
    passed, _ = test_kernel()
    tag = "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]"
    print(f"{tag} L0-P2 generic-path regression (block_I 32/128)")
    return passed


def l0_p3_contract_violation():
    """L0-P3: contract violation -> ErrFlag + RuntimeError; generic fallback correct."""
    S, H, D, tail_D, topk = 2048, 16, 512, 64, 128
    q, kv, idx, offsets = make_case(S, H, D, tail_D, topk, [0, 1023, S])
    # violate the shared contract at exactly one row/element (row 1500 is in
    # batch 1; element 7: 7 -> 8, still a valid in-range index)
    idx[1500, 7] = 8
    lse, ref_attn = ref_mla_topk_softmax(q, kv, idx, offsets)
    kv4 = kv.unsqueeze(-2)
    idx4 = idx.unsqueeze(-2)

    raised = False
    try:
        sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, panel_mode="shared")
    except RuntimeError as e:
        raised = True
        print(f"[PRECISION_PASS] L0-P3 panel contract violation raised RuntimeError: {str(e)[:120]}...")
    if not raised:
        print("[PRECISION_FAIL] L0-P3 panel contract violation NOT detected (no RuntimeError)")
        return False

    # same input, default declaration -> generic path, correct results
    out_generic = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, panel_mode="generic")
    ok = check_vs_ref("L0-P3 violated input via generic declaration", out_generic, ref_attn)
    return ok


def l0_p4a_empty_seq():
    """L0-P4a: empty first sequence (offsets=[0,0,1023,2048]) via panel path."""
    S, H, D, tail_D, topk = 2048, 16, 512, 64, 128
    q, kv, idx, offsets = make_case(S, H, D, tail_D, topk, [0, 0, 1023, S])
    lse, ref_attn = ref_mla_topk_softmax_masked(q, kv, idx, offsets)
    kv4 = kv.unsqueeze(-2)
    idx4 = idx.unsqueeze(-2)
    out_panel = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, panel_mode="shared")
    ok = check_vs_ref("L0-P4a empty-seq panel vs torch-ref", out_panel, ref_attn)
    out_g = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, block_I=128, panel_mode="generic")
    ok = check_equal("L0-P4a panel vs generic torch.equal", out_panel, out_g) and ok
    return ok


def l0_p4b_single_task():
    """L0-P4b: ΣS=1 / topk=32 single task — panel builder direct call (the host
    gate G5 legitimately falls back at this amortization ratio) + generic host."""
    S, H, D, tail_D, topk = 1, 16, 512, 64, 32
    q, kv, idx, offsets = make_case(S, H, D, tail_D, topk, [0, S])
    lse, ref_attn = ref_mla_topk_softmax_masked(q, kv, idx, offsets)
    kv4 = kv.unsqueeze(-2)
    idx4 = idx.unsqueeze(-2)
    grid = get_cube_core_num()

    pfunc = tl_sparse_mla_topk_reducesum_panel(heads=H, dim=D, tail_dim=tail_D, topk=topk, kv_group=1, grid_size=grid)
    ti = prepare_token_indices(offsets)
    err_flag = torch.zeros(1, dtype=torch.int32, device=q.device)
    out_panel = pfunc(q, kv4, idx4, lse, offsets, ti, err_flag)
    n_err = int(err_flag.item())
    if n_err != 0:
        print(f"[PRECISION_FAIL] L0-P4b direct panel run: ErrFlag={n_err} (no violation expected)")
        return False
    ok = check_vs_ref("L0-P4b single-task panel (direct builder) vs torch-ref", out_panel, ref_attn)

    # host "shared" declaration: gate G5 (amortization 1/1 < 16) must fall back
    with panel_builder_probe() as calls:
        out_host = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, panel_mode="shared")
    ok = check_vs_ref("L0-P4b single-task host(shared) [gate-miss->generic] vs torch-ref", out_host, ref_attn) and ok
    ok = check_equal("L0-P4b panel vs host-generic torch.equal", out_panel, out_host) and ok
    ok = check_no_panel_route("L0-P4b host gate G5 miss", H, D, tail_D, topk, 1, calls) and ok
    return ok


def l0_p4c_neg_one():
    """L0-P4c: shared vector with -1 tail padding (arange(96) + [-1]*32)."""
    S, H, D, tail_D, topk = 2048, 16, 512, 64, 128
    p = torch.cat([torch.arange(96, dtype=torch.int32), torch.full((32,), -1, dtype=torch.int32)])
    q, kv, idx, offsets = make_case(S, H, D, tail_D, topk, [0, 1023, S], index_vec=p)
    lse, ref_attn = ref_mla_topk_softmax_masked(q, kv, idx, offsets)
    kv4 = kv.unsqueeze(-2)
    idx4 = idx.unsqueeze(-2)
    out_panel = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, panel_mode="shared")
    ok = check_vs_ref("L0-P4c shared-vector-with--1 panel vs torch-ref", out_panel, ref_attn)
    out_g = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, block_I=128, panel_mode="generic")
    ok = check_equal("L0-P4c panel vs generic torch.equal", out_panel, out_g) and ok
    return ok


def l0_p4d_oob_index():
    """L0-P4d: out-of-range index value (clamp protects addressing, mask zeroes)."""
    S, H, D, tail_D, topk = 2048, 16, 512, 64, 128
    p = torch.arange(topk, dtype=torch.int32)
    p[100] = 5000  # far beyond ΣS_kv = 2048
    q, kv, idx, offsets = make_case(S, H, D, tail_D, topk, [0, 1023, S], index_vec=p)
    lse, ref_attn = ref_mla_topk_softmax_masked(q, kv, idx, offsets)
    kv4 = kv.unsqueeze(-2)
    idx4 = idx.unsqueeze(-2)
    out_panel = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, panel_mode="shared")
    ok = check_vs_ref("L0-P4d out-of-range-index panel vs torch-ref", out_panel, ref_attn)
    out_g = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, block_I=128, panel_mode="generic")
    ok = check_equal("L0-P4d panel vs generic torch.equal", out_panel, out_g) and ok
    return ok


def l0_p4e_dense_short_seq():
    """L0-P4e: B=32 short sequences (G4: B*G=32 > 8) -> silent generic fallback."""
    S, H, D, tail_D, topk = 512, 16, 512, 64, 64  # 32 batches x 16 rows
    offsets_list = [i * 16 for i in range(33)]
    q, kv, idx, offsets = make_case(S, H, D, tail_D, topk, offsets_list)
    lse, ref_attn = ref_mla_topk_softmax_masked(q, kv, idx, offsets)
    kv4 = kv.unsqueeze(-2)
    idx4 = idx.unsqueeze(-2)
    with panel_builder_probe() as calls:
        out = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, panel_mode="shared")
    ok = check_vs_ref("L0-P4e dense-short-seq (gate-miss->generic) vs torch-ref", out, ref_attn)
    ok = check_no_panel_route("L0-P4e G4 miss (B*G=32>8)", H, D, tail_D, topk, 1, calls) and ok
    return ok


def l0_p5_race_stress():
    """L0-P5: race-stress family — ΣS=64 × D∈{128,256} × topk∈{32,64} × 3 runs;
    each run vs reference AND pairwise torch.equal across runs (bitwise stable)."""
    all_ok = True
    for D in (128, 256):
        for topk in (32, 64):
            S, H, tail_D = 64, 16, 64
            q, kv, idx, offsets = make_case(S, H, D, tail_D, topk, [0, 32, 64])
            lse, ref_attn = ref_mla_topk_softmax_masked(q, kv, idx, offsets)
            kv4 = kv.unsqueeze(-2)
            idx4 = idx.unsqueeze(-2)
            outs = []
            for run in range(3):
                out = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, panel_mode="shared")
                torch.npu.synchronize()
                outs.append(out)
                ok = check_vs_ref(f"L0-P5 D={D} topk={topk} run{run} vs torch-ref", out, ref_attn)
                all_ok = all_ok and ok
            eq01 = torch.equal(outs[0], outs[1])
            eq02 = torch.equal(outs[0], outs[2])
            eq12 = torch.equal(outs[1], outs[2])
            stable = eq01 and eq02 and eq12
            tag = "[PRECISION_PASS]" if stable else "[PRECISION_FAIL]"
            print(f"{tag} L0-P5 D={D} topk={topk}: 3-run bitwise stability (equal={stable})")
            all_ok = all_ok and stable
    return all_ok


def l0_p6_gate_unit():
    """L0-P6: host gate unit — G2/G3 (topk>128), G4 (B*G>8), G5 (amortization<16)
    each dispatch to generic (panel-key absence probe) with correct results."""
    all_ok = True
    H, D, tail_D = 16, 512, 64

    # (a) topk=256 > 128: G2/G3 v1-domain miss
    S, topk = 512, 256
    q, kv, idx, offsets = make_case(S, H, D, tail_D, topk, [0, S])
    lse, ref_attn = ref_mla_topk_softmax_masked(q, kv, idx, offsets)
    kv4, idx4 = kv.unsqueeze(-2), idx.unsqueeze(-2)
    with panel_builder_probe() as calls:
        out = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, panel_mode="shared")
    all_ok = check_vs_ref("L0-P6a topk=256 (gate-miss->generic) vs torch-ref", out, ref_attn) and all_ok
    all_ok = check_no_panel_route("L0-P6a G2/G3 miss (topk>128)", H, D, tail_D, topk, 1, calls) and all_ok

    # (b) B=9 batches x 16 rows (G4: B*G=9>8; G5: 144/9=16 passes -> pure G4)
    S, topk = 144, 96
    offsets_list = [i * 16 for i in range(10)]
    q, kv, idx, offsets = make_case(S, H, D, tail_D, topk, offsets_list)
    lse, ref_attn = ref_mla_topk_softmax_masked(q, kv, idx, offsets)
    kv4, idx4 = kv.unsqueeze(-2), idx.unsqueeze(-2)
    with panel_builder_probe() as calls:
        out = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, panel_mode="shared")
    all_ok = check_vs_ref("L0-P6b B*G=9 (gate-miss->generic) vs torch-ref", out, ref_attn) and all_ok
    all_ok = check_no_panel_route("L0-P6b G4 miss (B*G=9>8)", H, D, tail_D, topk, 1, calls) and all_ok

    # (c) B=2 x 4 rows (G5: 8/2=4 < 16; G4: 2<=8 passes -> pure G5)
    S, topk = 8, 32
    q, kv, idx, offsets = make_case(S, H, D, tail_D, topk, [0, 4, 8])
    lse, ref_attn = ref_mla_topk_softmax_masked(q, kv, idx, offsets)
    kv4, idx4 = kv.unsqueeze(-2), idx.unsqueeze(-2)
    with panel_builder_probe() as calls:
        out = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, panel_mode="shared")
    all_ok = check_vs_ref("L0-P6c amortization<16 (gate-miss->generic) vs torch-ref", out, ref_attn) and all_ok
    all_ok = check_no_panel_route("L0-P6c G5 miss (ΣS/B=4<16)", H, D, tail_D, topk, 1, calls) and all_ok
    return all_ok


# ===========================================================================
# L0-P7 family — auto dispatch (DESIGN_panel_mode rev2 §9.2)
# ===========================================================================
def run_detect(idx, offsets, topk, kv_group=1):
    """Direct detect-kernel call with a fresh zeros Violations (§8.4 lifecycle:
    per-call fresh tensor; int32 exact assertion target)."""
    grid = get_cube_core_num()
    func = tl_sparse_mla_topk_reducesum_detect(topk=topk, kv_group=kv_group, grid_size=grid)
    viol = torch.zeros(1, dtype=torch.int32, device=idx.device)
    func(idx, offsets, viol)
    torch.npu.synchronize()
    return int(viol.item())


def expected_violations(idx_cpu, offsets_list, grid_lanes=40):
    """Host reference predicate (§9.1 rev2 addendum).

    Returns (viol_ref, lanes_firing):
      viol_ref = #{r in [1, S): bid(r)==bid(r-1) and idx[r] != idx[r-1]}
        (the chain predicate's true differing-pair count)
      lanes_firing = the exact number of lanes that fire an atomic under the
        kernel's (seq, lane) integer partition — deterministic, since each
        differing pair lies in exactly one lane's sub-interval and a lane
        fires at most once.
    """
    viol_ref = 0
    fires = [False] * grid_lanes
    for b in range(len(offsets_list) - 1):
        bos, eos = int(offsets_list[b]), int(offsets_list[b + 1])
        ncmp = eos - bos - 1
        if ncmp <= 0:
            continue
        diff_rows = [r for r in range(bos + 1, eos) if not torch.equal(idx_cpu[r], idx_cpu[r - 1])]
        viol_ref += len(diff_rows)
        for lane in range(grid_lanes):
            a = bos + 1 + (lane * ncmp) // grid_lanes
            e = bos + 1 + ((lane + 1) * ncmp) // grid_lanes
            if any(a <= r < e for r in diff_rows):
                fires[lane] = True
    return viol_ref, sum(fires)


def check_route(name, counts, expect):
    """Routing-probe assertion: expect maps variant -> 'zero' | 'ge1'."""
    ok = True
    detail = []
    for variant, want in expect.items():
        c = counts[variant]
        good = (c == 0) if want == "zero" else (c >= 1)
        ok = ok and good
        detail.append(f"{variant}={c}")
    tag = "[PRECISION_PASS]" if ok else "[PRECISION_FAIL]"
    print(f"{tag} {name}: routing {', '.join(detail)}")
    return ok


def _shared_idx(S, topk, kv_group=1):
    """Contract-holding Indices [S, G, topk] (arange shared vector)."""
    p = torch.arange(topk, dtype=torch.int32)
    return p.view(1, 1, topk).repeat(S, kv_group, 1).contiguous().npu()


def l0_p7a_detect_unit():
    """L0-P7a: detect-kernel unit — partition boundaries & violation placement
    (DESIGN_panel_mode §9.2): golden-family offsets [0,1023,2048]; violations
    int32 exact assertion against the host predicate."""
    S, topk = 2048, 128
    grid_lanes = get_cube_core_num() * 2
    golden = [0, 1023, S]
    all_ok = True

    def run_sub(sub, offsets_list, mutate=None, kv_group=1):
        nonlocal all_ok
        offsets = torch.tensor(offsets_list, dtype=torch.int32).npu()
        p = torch.arange(topk, dtype=torch.int32)
        idx = p.view(1, 1, topk).repeat(S, kv_group, 1).contiguous()
        if mutate is not None:
            mutate(idx)
        idx = idx.npu()
        got = run_detect(idx, offsets, topk, kv_group=kv_group)
        viol_ref, lanes = expected_violations(idx.cpu(), offsets_list, grid_lanes)
        exact = got == lanes
        consist = (got == 0) == (viol_ref == 0) and 0 <= got <= max(viol_ref, 0)
        ok = exact and consist
        tag = "[PRECISION_PASS]" if ok else "[PRECISION_FAIL]"
        print(
            f"{tag} L0-P7a({sub}): violations={got} (exact lanes_firing={lanes}, "
            f"pred viol_ref={viol_ref}) -> exact={exact}, 0-consistency={consist}"
        )
        all_ok = all_ok and ok

    # (1) contract holds -> 0
    run_sub("1-contract", golden)
    # (2) violation at first comparable row (r=bos+1 of seq0)
    def mut_first(idx):
        idx[1, 0, 0] = idx[1, 0, 0] + 1

    run_sub("2-first-row", golden, mut_first)
    # (3) violation at last row of seq0 (r=eos-1: exactly one affected pair)
    def mut_last(idx):
        idx[1022, 0, 0] = idx[1022, 0, 0] + 1

    run_sub("3-last-row", golden, mut_last)
    # (4) violations at lane sub-interval boundary rows ±1 (partition formula
    #     a_l = bos+1+(l*ncmp)//40; rows a_l and a_l-1 stress the cross-lane
    #     preload window and disjointness)
    def mut_boundary(idx):
        ncmp = 1022
        for lane in (10, 20, 30):
            a = 1 + (lane * ncmp) // grid_lanes
            idx[a - 1, 0, 1] = idx[a - 1, 0, 1] + 1
            idx[a, 0, 1] = idx[a, 0, 1] + 1

    run_sub("4-lane-boundary", golden, mut_boundary)
    # (5) multiple violations accumulated (2 seqs, 1 mid row each)
    def mut_multi(idx):
        idx[500, 0, 2] = idx[500, 0, 2] + 1
        idx[1500, 0, 2] = idx[1500, 0, 2] + 1

    run_sub("5-multi", golden, mut_multi)
    # (6) empty first sequence + contract holds -> 0
    run_sub("6-empty-seq", [0, 0, 1023, S])
    # (7) single-row sequence: zero comparisons for the len-1 segment — its row
    #     pairs with nothing (bid restriction), even when mutated
    def mut_single(idx):
        idx[0, 0, 3] = idx[0, 0, 3] + 1  # row 0 = the len-1 seq itself

    run_sub("7-single-row-seq", [0, 1, 1024, S], mut_single)
    # (8) ΣS=1 -> 0
    offsets1 = torch.tensor([0, 1], dtype=torch.int32).npu()
    got8 = run_detect(_shared_idx(1, topk), offsets1, topk)
    ok8 = got8 == 0
    print(f"{'[PRECISION_PASS]' if ok8 else '[PRECISION_FAIL]'} L0-P7a(8): ΣS=1 violations={got8} (expect 0)")
    all_ok = all_ok and ok8
    # (9) kv_group=2: g=0 violated (last row of seq0), g=1 holds -> >0 (∃g semantics)
    def mut_g0(idx):
        idx[1022, 0, 5] = idx[1022, 0, 5] + 1  # g=0 only

    offsets9 = torch.tensor(golden, dtype=torch.int32).npu()
    p = torch.arange(topk, dtype=torch.int32)
    idx9 = p.view(1, 1, topk).repeat(S, 2, 1).contiguous()
    mut_g0(idx9)
    idx9 = idx9.npu()
    got9 = run_detect(idx9, offsets9, topk, kv_group=2)
    viol_ref9, lanes9 = expected_violations(idx9.cpu(), golden, grid_lanes)
    ok9 = got9 == lanes9 and got9 >= 1
    print(
        f"{'[PRECISION_PASS]' if ok9 else '[PRECISION_FAIL]'} "
        f"L0-P7a(9): kv_group=2 g0-only violation violations={got9} (lanes={lanes9}, pred={viol_ref9})"
    )
    all_ok = all_ok and ok9
    return all_ok


def l0_p7b_auto_panel():
    """L0-P7b: auto end-to-end on a panel-eligible input (mode NOT declared) —
    golden criteria + bitwise equality vs explicit generic + triple-launch
    routing probe (unguarded generic == 0) + wall timing signal."""
    S, H, D, tail_D, topk = 2048, 16, 512, 64, 128
    q, kv, idx, offsets = make_case(S, H, D, tail_D, topk, [0, 1023, S])
    lse, ref_attn = ref_mla_topk_softmax(q, kv, idx, offsets)
    kv4 = kv.unsqueeze(-2)
    idx4 = idx.unsqueeze(-2)
    grid = get_cube_core_num()

    with auto_route_probe() as counts:
        out_auto = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D)  # 缺省 auto
    ok = check_vs_ref("L0-P7b auto(default) vs torch-ref (golden criteria)", out_auto, ref_attn)

    out_g = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, panel_mode="generic")
    ok = check_equal("L0-P7b auto vs explicit-generic torch.equal", out_auto, out_g) and ok

    ok = check_route(
        "L0-P7b auto triple-launch routing",
        counts,
        {"detect": "ge1", "panel_guarded": "ge1", "generic_guarded": "ge1", "generic": "zero"},
    ) and ok
    keys_ok = (
        detect_cache_key(topk, 1, grid) in _kernel_mod._BUILD_CACHE
        and guarded_panel_cache_key(H, D, tail_D, topk, 1, grid) in _kernel_mod._BUILD_CACHE
        and guarded_generic_cache_key(H, D, tail_D, topk, 1, 32, grid) in _kernel_mod._BUILD_CACHE
    )
    tag = "[PRECISION_PASS]" if keys_ok else "[PRECISION_FAIL]"
    print(f"{tag} L0-P7b auto _BUILD_CACHE keys present (detect/panel_guarded/generic_guarded)")
    ok = ok and keys_ok

    # wall timing signal (non-blocking; §10.4 budget band ~155-157us graph-replay
    # — eager host-API wall is host-bound, only a coarse early signal)
    t_wall = time_wall(
        lambda: sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D),
        warmup=5,
        iters=20,
    )
    print(f"[INFO] L0-P7b auto host-API wall = {t_wall:.3f} ms (incl. detect + triple-launch + err_flag check)")
    return ok


def l0_p7c_auto_random():
    """L0-P7c: auto end-to-end on random indices (metadata passes, contract
    broken) — generic(guarded) executes and is correct; triple-launch probe;
    direct detect on the same input finds violations."""
    S, H, D, tail_D, topk = 2048, 16, 512, 64, 128
    torch.manual_seed(123)
    q = torch.randn((S, H, D + tail_D)).npu().bfloat16()
    kv = torch.randn((S, D + tail_D)).npu().bfloat16()
    offsets_list = [0, 1023, S]
    offsets = torch.tensor(offsets_list, dtype=torch.int32).npu()
    # randint indices clamped to the causal boundary (bench_graph_smtk-style
    # `randint → clamp(max=s_col)` construction; CPU -> one H2D). The clamp
    # keeps every index causally valid — unclamped randint produces rows whose
    # topk are all > s_i, whose renormalization is 0/0 = NaN in BOTH the ref
    # and the kernel output (a data-construction artifact, not a kernel defect).
    idx = torch.randint(0, S, (S, topk), dtype=torch.int32)
    seg_lens = [b - a for a, b in zip(offsets_list, offsets_list[1:])]
    s_pos = torch.cat([torch.arange(n, dtype=torch.int32) for n in seg_lens])
    idx = idx.clamp(max=s_pos.unsqueeze(1)).npu()
    lse, ref_attn = ref_mla_topk_softmax_masked(q, kv, idx, offsets)
    kv4 = kv.unsqueeze(-2)
    idx4 = idx.unsqueeze(-2)

    with auto_route_probe() as counts:
        out_auto = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D)  # 缺省 auto
    ok = check_vs_ref("L0-P7c auto(random indices) vs torch-ref", out_auto, ref_attn)
    ok = check_route(
        "L0-P7c auto triple-launch routing (cache-hit dispatch)",
        counts,
        {"detect": "ge1", "panel_guarded": "ge1", "generic_guarded": "ge1", "generic": "zero"},
    ) and ok

    v_direct = run_detect(idx4, offsets, topk)
    ok = (v_direct > 0) and ok
    tag = "[PRECISION_PASS]" if v_direct > 0 else "[PRECISION_FAIL]"
    print(f"{tag} L0-P7c direct detect on random indices: violations={v_direct} (>0 expected)")
    return ok


def l0_p7d_auto_gate_miss():
    """L0-P7d: auto with metadata-gated-out input (topk=256 > 128, mode NOT
    declared) — unguarded generic single-shot: zero detect/guarded builder
    calls + their _BUILD_CACHE keys absent (zero-overhead zero-change probe)."""
    S, H, D, tail_D, topk = 512, 16, 512, 64, 256
    q, kv, idx, offsets = make_case(S, H, D, tail_D, topk, [0, S])
    lse, ref_attn = ref_mla_topk_softmax_masked(q, kv, idx, offsets)
    kv4 = kv.unsqueeze(-2)
    idx4 = idx.unsqueeze(-2)
    grid = get_cube_core_num()

    with auto_route_probe() as counts:
        out = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D)  # 缺省 auto
    ok = check_vs_ref("L0-P7d auto(gate-miss -> generic) vs torch-ref", out, ref_attn)
    ok = check_route(
        "L0-P7d auto gate-miss routing",
        counts,
        {"detect": "zero", "panel_guarded": "zero", "generic_guarded": "zero", "generic": "ge1"},
    ) and ok
    keys_absent = (
        detect_cache_key(topk, 1, grid) not in _kernel_mod._BUILD_CACHE
        and guarded_panel_cache_key(H, D, tail_D, topk, 1, grid) not in _kernel_mod._BUILD_CACHE
        and guarded_generic_cache_key(H, D, tail_D, topk, 1, 32, grid) not in _kernel_mod._BUILD_CACHE
    )
    tag = "[PRECISION_PASS]" if keys_absent else "[PRECISION_FAIL]"
    print(f"{tag} L0-P7d auto gate-miss: detect/guarded _BUILD_CACHE keys absent")
    ok = ok and keys_absent
    return ok


def l0_p7e_guarded_generic():
    """L0-P7e: guarded-generic variant regression + race re-run — direct builder
    calls with violations preset [1] (forced execute) reusing the L0-P5 race
    family, and [0] (forced early-exit) with a sentinel-filled output."""
    all_ok = True
    grid = get_cube_core_num()
    for D in (128, 256):
        for topk in (32, 64):
            S, H, tail_D = 64, 16, 64
            q, kv, idx, offsets = make_case(S, H, D, tail_D, topk, [0, 32, 64])
            lse, ref_attn = ref_mla_topk_softmax_masked(q, kv, idx, offsets)
            kv4, idx4 = kv.unsqueeze(-2), idx.unsqueeze(-2)
            gfunc = tl_sparse_mla_topk_reducesum_guarded(
                heads=H, dim=D, tail_dim=tail_D, topk=topk, kv_group=1, grid_size=grid
            )
            ti = prepare_token_indices(offsets)

            # execute group (violations=[1]): golden criteria + 3-run bitwise
            outs = []
            for run in range(3):
                viol = torch.ones(1, dtype=torch.int32).npu()
                rs = torch.empty((S, 1, 1, topk), dtype=torch.float32).npu()
                gfunc(q, kv4, idx4, lse, offsets, ti, viol, rs)
                torch.npu.synchronize()
                outs.append(rs)
                all_ok = check_vs_ref(
                    f"L0-P7e D={D} topk={topk} run{run} [guarded exec] vs torch-ref", rs, ref_attn
                ) and all_ok
            stable = torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])
            tag = "[PRECISION_PASS]" if stable else "[PRECISION_FAIL]"
            print(f"{tag} L0-P7e D={D} topk={topk}: 3-run bitwise stability (equal={stable})")
            all_ok = all_ok and stable

            # bitwise identity vs the unguarded generic path (guard adds no drift)
            out_ung = sparse_mla_topk_reducesum(
                q, kv4, idx4, lse, offsets, dim_v=D, block_I=32, panel_mode="generic"
            )
            all_ok = check_equal(
                f"L0-P7e D={D} topk={topk} guarded-exec vs unguarded-generic torch.equal",
                outs[0],
                out_ung,
            ) and all_ok

            # early-exit group (violations=[0]): rs pre-filled sentinel -> preserved
            sentinel = -12345.0
            rs0 = torch.full((S, 1, 1, topk), sentinel).npu()
            viol0 = torch.zeros(1, dtype=torch.int32).npu()
            gfunc(q, kv4, idx4, lse, offsets, ti, viol0, rs0)
            torch.npu.synchronize()
            kept = bool((rs0 == sentinel).all())
            tag = "[PRECISION_PASS]" if kept else "[PRECISION_FAIL]"
            print(
                f"{tag} L0-P7e D={D} topk={topk} [guarded early-exit]: rs sentinel preserved={kept}"
            )
            all_ok = all_ok and kept
    return all_ok


def l0_p7f_three_modes():
    """L0-P7f: three-mode semantics — same golden input through {auto, generic,
    shared}: pairwise torch.equal + per-mode routing probes."""
    S, H, D, tail_D, topk = 2048, 16, 512, 64, 128
    q, kv, idx, offsets = make_case(S, H, D, tail_D, topk, [0, 1023, S])
    lse, ref_attn = ref_mla_topk_softmax(q, kv, idx, offsets)
    kv4 = kv.unsqueeze(-2)
    idx4 = idx.unsqueeze(-2)

    with auto_route_probe() as c_g:
        out_generic = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, panel_mode="generic")
    ok = check_route(
        "L0-P7f explicit-generic routing",
        c_g,
        {"generic": "ge1", "panel": "zero", "detect": "zero", "panel_guarded": "zero", "generic_guarded": "zero"},
    )
    with auto_route_probe() as c_s:
        out_shared = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D, panel_mode="shared")
    ok = check_route(
        "L0-P7f shared routing (no detect)",
        c_s,
        {"panel": "ge1", "generic": "zero", "detect": "zero", "panel_guarded": "zero", "generic_guarded": "zero"},
    ) and ok
    with auto_route_probe() as c_a:
        out_auto = sparse_mla_topk_reducesum(q, kv4, idx4, lse, offsets, dim_v=D)  # 缺省 auto
    ok = check_route(
        "L0-P7f auto routing (triple launch)",
        c_a,
        {"detect": "ge1", "panel_guarded": "ge1", "generic_guarded": "ge1", "generic": "zero", "panel": "zero"},
    ) and ok

    ok = check_equal("L0-P7f auto vs generic torch.equal", out_auto, out_generic) and ok
    ok = check_equal("L0-P7f auto vs shared torch.equal", out_auto, out_shared) and ok
    ok = check_equal("L0-P7f shared vs generic torch.equal", out_shared, out_generic) and ok
    return ok


def l0_p7g_detect_race():
    """L0-P7g: detect race stress (#22 discipline for new code) — ΣS=16384
    multi-chunk/multi-lane, {contract, mid-segment random violations} × 3 runs:
    bitwise-identical violations counts + exact host-predicate match."""
    S, topk = 16384, 128
    grid_lanes = get_cube_core_num() * 2
    offsets_list = [0, 4095, 8191, 12287, S]  # 4 seqs -> per-lane multi-chunk
    offsets = torch.tensor(offsets_list, dtype=torch.int32).npu()

    # (a) contract holds ×3: all zero, bitwise identical
    idx = _shared_idx(S, topk)
    runs = [run_detect(idx, offsets, topk) for _ in range(3)]
    ok_a = runs == [0, 0, 0]
    print(
        f"{'[PRECISION_PASS]' if ok_a else '[PRECISION_FAIL]'} "
        f"L0-P7g contract x3: violations={runs} (all zero, bitwise identical)"
    )

    # (b) mid-segment random violations ×3: >0, bitwise identical, exact lanes
    torch.manual_seed(7)
    bad = (torch.randperm(S - 200)[:100] + 100).tolist()  # mid-segment rows
    idx_v = idx.cpu().clone()
    for r in bad:
        idx_v[r, 0, 0] = idx_v[r, 0, 0] + 1
    idx_v = idx_v.npu()
    viol_ref, lanes = expected_violations(idx_v.cpu(), offsets_list, grid_lanes)
    runs_v = [run_detect(idx_v, offsets, topk) for _ in range(3)]
    ok_b = (
        runs_v[0] > 0
        and runs_v[0] == runs_v[1] == runs_v[2]
        and runs_v[0] == lanes
        and lanes <= viol_ref
    )
    print(
        f"{'[PRECISION_PASS]' if ok_b else '[PRECISION_FAIL]'} "
        f"L0-P7g violated x3: violations={runs_v} (bitwise identical, exact lanes={lanes}, pred={viol_ref})"
    )
    return ok_a and ok_b


L0_CASES = [
    ("L0-P1", l0_p1_panel_golden),
    ("L0-P2", l0_p2_generic_regression),
    ("L0-P3", l0_p3_contract_violation),
    ("L0-P4a", l0_p4a_empty_seq),
    ("L0-P4b", l0_p4b_single_task),
    ("L0-P4c", l0_p4c_neg_one),
    ("L0-P4d", l0_p4d_oob_index),
    ("L0-P4e", l0_p4e_dense_short_seq),
    ("L0-P5", l0_p5_race_stress),
    ("L0-P6", l0_p6_gate_unit),
    ("L0-P7a", l0_p7a_detect_unit),
    ("L0-P7b", l0_p7b_auto_panel),
    ("L0-P7c", l0_p7c_auto_random),
    ("L0-P7d", l0_p7d_auto_gate_miss),
    ("L0-P7e", l0_p7e_guarded_generic),
    ("L0-P7f", l0_p7f_three_modes),
    ("L0-P7g", l0_p7g_detect_race),
]


def test_sparse_mla_topk_reducesum_l0():
    """Run all L0 gate cases (DESIGN_panel_mode §9.2); per-case try/except so one
    failure does not mask the rest."""
    all_ok = True
    for name, fn in L0_CASES:
        print(f"\n==== {name} ====")
        try:
            ok = fn()
        except Exception:
            traceback.print_exc()
            print(f"[PRECISION_FAIL] {name}: unexpected exception (traceback above)")
            ok = False
        all_ok = all_ok and ok
    return all_ok


# ===========================================================================
# L1/L2/Boundary layers — scenario-B extension (tilelang-op-test-design).
# Coverage tags follow references/coverage-matrix.md; the category is Fusion
# (CV 混合: gemm_v0 Cube + mask/exp/reduce Vector) per proto.yaml.
# ===========================================================================
COVERAGE_CATEGORY = "Fusion"


def make_case_v(S, H, D, tail_D, topk, offsets_list, seed=42, vrange=None, kv_group=1):
    """L1 case builder: CPU uniform-range construction -> CPU bf16 cast -> one
    H2D per tensor (aclnn audit: no NPU-side random/cast). vrange=None -> randn."""
    torch.manual_seed(seed)
    if vrange is None:
        q = torch.randn((S, H, D + tail_D)).bfloat16().npu()
        kv = torch.randn((S, kv_group, D + tail_D)).bfloat16().npu()
    else:
        lo, hi = vrange
        q = (torch.rand((S, H, D + tail_D)) * (hi - lo) + lo).bfloat16().npu()
        kv = (torch.rand((S, kv_group, D + tail_D)) * (hi - lo) + lo).bfloat16().npu()
    offsets = torch.tensor(offsets_list, dtype=torch.int32).npu()
    p = torch.arange(topk, dtype=torch.int32)
    topk_indices = p.unsqueeze(0).unsqueeze(0).repeat(S, kv_group, 1).contiguous().npu()
    return q, kv, topk_indices, offsets


def _run_l1_case(name, cfg, tags):
    """One L1 case: build inputs, run kernel(s) per cfg, compare vs reference.
    Blocking ([PRECISION_*])."""
    print(f"\n---- L1 {name} ----")
    H = cfg.get("heads", 16)
    G = cfg.get("kv_group", 1)
    q, kv, idx, offsets = make_case_v(
        cfg["S"],
        H,
        cfg["D"],
        cfg["tail_D"],
        cfg["topk"],
        cfg["offsets"],
        seed=cfg.get("seed", 43),
        vrange=cfg.get("vrange"),
        kv_group=G,
    )
    if G > 1:
        lse, ref_attn = ref_mla_topk_softmax_grouped(q, kv, idx, offsets, G, sm_scale=cfg.get("sm_scale"))
    else:
        # refs expect 2-D kv [ΣS, D] / idx [ΣS, topk]; make_case_v builds the
        # 3-D host-ready shapes [ΣS, G, *]
        lse, ref_attn = ref_mla_topk_softmax_masked(q, kv[:, 0, :], idx[:, 0, :], offsets, sm_scale=cfg.get("sm_scale"))
    host_kwargs = {
        "dim_v": cfg["D"],
        "kv_group": G,
        "sm_scale": cfg.get("sm_scale"),
    }
    ok = True
    outs = {}
    if cfg.get("run_generic", True):
        outs["generic"] = sparse_mla_topk_reducesum(
            q,
            kv,
            idx,
            lse,
            offsets,
            block_I=cfg.get("block_I", 32),
            grid_size=cfg.get("grid_size"),
            panel_mode="generic",  # rev1 语义保持（缺省已改 auto）
            **{k: v for k, v in host_kwargs.items() if v is not None or k == "sm_scale"},
        )
        ok = check_vs_ref(f"L1 {name} [generic]", outs["generic"], ref_attn) and ok
    if cfg.get("run_shared", False):
        outs["shared"] = sparse_mla_topk_reducesum(
            q,
            kv,
            idx,
            lse,
            offsets,
            block_I=cfg.get("block_I", 32),
            grid_size=cfg.get("grid_size"),
            panel_mode="shared",
            **{k: v for k, v in host_kwargs.items() if v is not None or k == "sm_scale"},
        )
        ok = check_vs_ref(f"L1 {name} [panel shared]", outs["shared"], ref_attn) and ok
    if cfg.get("run_auto", False):
        # rev2 panel_mode 三值维度（proto Enum）：缺省 auto（元数据门控 + 设备侧检测
        # + 推测三连发）与显式 generic/shared 输出两两一致（DESIGN_panel_mode §8.1）
        outs["auto"] = sparse_mla_topk_reducesum(
            q,
            kv,
            idx,
            lse,
            offsets,
            block_I=cfg.get("block_I", 32),
            grid_size=cfg.get("grid_size"),
            panel_mode="auto",
            **{k: v for k, v in host_kwargs.items() if v is not None or k == "sm_scale"},
        )
        ok = check_vs_ref(f"L1 {name} [panel auto]", outs["auto"], ref_attn) and ok
    if len(outs) == 3:
        ok = check_equal(f"L1 {name} auto vs generic torch.equal", outs["auto"], outs["generic"]) and ok
        ok = check_equal(f"L1 {name} auto vs shared torch.equal", outs["auto"], outs["shared"]) and ok
        ok = check_equal(f"L1 {name} shared vs generic torch.equal", outs["shared"], outs["generic"]) and ok
    return ok


# (name, config, tags) — D-* tags are machine-checked by coverage_check.py.
L1_CASES = [
    (
        "aligned_valrange_s",
        {"S": 2048, "D": 512, "tail_D": 64, "topk": 128, "offsets": [0, 1023, 2048], "vrange": (-1, 1)},
        ["D-SHAPE-ALIGNED", "D-VALRANGE-S", "D-DTYPE-bf16", "D-DTYPE-fp32", "D-DTYPE-int32"],
    ),
    # Value ranges are calibrated to the operator's fp32-exp envelope: the
    # kernel (faithful to the CUDA original) computes exp(logit*scale - lse)
    # BEFORE the causal mask, so causally-invalid positions overflow to inf and
    # inf*0 = NaN once |logit*scale - lse| > ~88. Empirically |values| <= 6
    # keeps NaN=0 for this data pattern; (-10,10) produces ~126 raw NaNs.
    (
        "valrange_m",
        {"S": 1024, "D": 512, "tail_D": 64, "topk": 128, "offsets": [0, 500, 1024], "vrange": (-3, 3)},
        ["D-VALRANGE-M"],
    ),
    (
        "valrange_l",  # largest empirically-safe symmetric range (lse ~ 59)
        {"S": 1024, "D": 512, "tail_D": 64, "topk": 128, "offsets": [0, 500, 1024], "vrange": (-6, 6)},
        ["D-VALRANGE-L"],
    ),
    (
        "valrange_asym",
        {"S": 1024, "D": 512, "tail_D": 64, "topk": 128, "offsets": [0, 500, 1024], "vrange": (-2, 6)},
        ["D-VALRANGE-ASYM"],
    ),
    (
        "task_tail_1",  # ΣS = 20*40+1: persistent task axis leaves a tail-1 on core 0
        {"S": 801, "D": 512, "tail_D": 64, "topk": 128, "offsets": [0, 401, 801], "run_shared": True},
        ["D-SHAPE-TAIL-1", "D-PARAM-panel_mode"],
    ),
    (
        "panel_mode_sweep",  # rev2 panel_mode 三值维度：auto(缺省)/generic/shared 全跑 + 两两一致
        {"S": 1024, "D": 512, "tail_D": 64, "topk": 128, "offsets": [0, 512, 1024],
         "run_generic": True, "run_shared": True, "run_auto": True},
        ["D-PARAM-panel_mode"],
    ),
    (
        "task_tail_mid",  # ΣS = 20*40+10: mid-size task tail
        {"S": 810, "D": 512, "tail_D": 64, "topk": 128, "offsets": [0, 400, 810], "run_shared": True},
        ["D-SHAPE-TAIL-MID"],
    ),
    (
        "prime_varlen",  # ΣS = 1021 prime, first segment 509 prime: fully non-aligned varlen
        {"S": 1021, "D": 512, "tail_D": 64, "topk": 128, "offsets": [0, 509, 1021], "run_shared": True},
        ["D-SHAPE-PRIME"],
    ),
    (
        "edge_single_task",  # ΣS=1 degenerate (generic; panel direct covered in L0-P4b)
        {"S": 1, "D": 512, "tail_D": 64, "topk": 32, "offsets": [0, 1]},
        ["D-SHAPE-EDGE"],
    ),
    (
        "dim_v_256",  # non-default dim_v (panel at D=256 cached from L0-P5 configs)
        {"S": 512, "D": 256, "tail_D": 64, "topk": 64, "offsets": [0, 512], "run_shared": True},
        ["D-PARAM-dim_v"],
    ),
    (
        "kv_group_2",  # non-default kv_group (heads=32, head_kv=16)
        {"S": 512, "heads": 32, "kv_group": 2, "D": 512, "tail_D": 64, "topk": 64, "offsets": [0, 512], "run_shared": True},
        ["D-PARAM-kv_group"],
    ),
    (
        "sm_scale_explicit",  # non-default explicit scale
        {"S": 512, "D": 512, "tail_D": 64, "topk": 128, "offsets": [0, 512], "sm_scale": 0.05},
        ["D-PARAM-sm_scale"],
    ),
    (
        "grid_size_8",  # non-default persistent grid
        {"S": 512, "D": 512, "tail_D": 64, "topk": 128, "offsets": [0, 512], "grid_size": 8},
        ["D-PARAM-grid_size"],
    ),
    (
        "block_I_64",  # non-default topk blocking (generic path)
        {"S": 2048, "D": 512, "tail_D": 64, "topk": 128, "offsets": [0, 1023, 2048], "block_I": 64},
        ["D-PARAM-block_I"],
    ),
]

# L2 negative cases (expect rejection; tags machine-checked)
L2_CASES = [
    ("dtype_q_f32", "D-EXC-DTYPE"),  # q in float32 (must be bfloat16) -> rejected
    ("topk_not_div_blockI", "D-EXC-SHAPE"),  # topk=48, 48 % 32 != 0 -> assert
    ("head_kv_out_of_domain", "D-EXC-SHAPE"),  # heads=15 -> head_kv=15 not pow2/64k -> assert
]

# Boundary special-value cases (legal inputs; structural mask + tolerance)
BOUNDARY_CASES = [
    ("sparse_inf", "D-SPECIAL-INF"),
    ("sparse_nan", "D-SPECIAL-NAN"),
    ("all_zero", "D-SPECIAL-ZERO"),
    ("bf16_max_sparse", "D-SPECIAL-DBOUND"),
]


def test_sparse_mla_topk_reducesum_l1():
    """L1 functional layer: deterministic shape sweep (aligned/tail/prime/edge) +
    value-range sweep + non-default attrs, generic and panel paths."""
    all_ok = True
    for name, cfg, _tags in L1_CASES:
        try:
            ok = _run_l1_case(name, cfg, _tags)
        except Exception:
            traceback.print_exc()
            print(f"[PRECISION_FAIL] L1 {name}: unexpected exception (traceback above)")
            ok = False
        all_ok = all_ok and ok
    return all_ok


def _l2_dtype_q_f32():
    S, H, D, tail_D, topk = 256, 16, 512, 64, 128
    q, kv, idx, offsets = make_case_v(S, H, D, tail_D, topk, [0, S])
    q = q.float()  # WRONG dtype: float32 instead of bfloat16
    lse = torch.zeros((S, H), dtype=torch.float32).npu()
    sparse_mla_topk_reducesum(q, kv, idx, lse, offsets, dim_v=D)


def _l2_topk_not_div():
    S, H, D, tail_D, topk = 256, 16, 512, 64, 48
    q, kv, idx, offsets = make_case_v(S, H, D, tail_D, topk, [0, S])
    lse = torch.zeros((S, H), dtype=torch.float32).npu()
    sparse_mla_topk_reducesum(q, kv, idx, lse, offsets, dim_v=D)


def _l2_head_kv_domain():
    S, H, D, tail_D, topk = 256, 15, 512, 64, 128  # heads=15 -> head_kv=15 invalid
    q, kv, idx, offsets = make_case_v(S, H, D, tail_D, topk, [0, S])
    lse = torch.zeros((S, H), dtype=torch.float32).npu()
    sparse_mla_topk_reducesum(q, kv, idx, lse, offsets, dim_v=D)


_L2_FNS = {
    "dtype_q_f32": _l2_dtype_q_f32,
    "topk_not_div_blockI": _l2_topk_not_div,
    "head_kv_out_of_domain": _l2_head_kv_domain,
}


def test_sparse_mla_topk_reducesum_l2():
    """L2 negative layer: invalid dtype / out-of-domain shapes must be rejected
    (correct exception = [BOUNDARY_PASS]; silent acceptance = [BOUNDARY_WARN]).
    Non-blocking."""
    for name, _tag in L2_CASES:
        fn = _L2_FNS[name]
        try:
            fn()
        except Exception as e:
            print(f"[BOUNDARY_PASS] l2 {name}: rejected as expected ({type(e).__name__})")
        else:
            print(f"[BOUNDARY_WARN] l2 {name}: invalid input silently accepted")


def _make_boundary_inputs(kind, S=2048, H=16, D=512, tail_D=64, topk=128):
    """CPU-side special-value construction (aclnn audit: CPU inject -> one H2D)."""
    torch.manual_seed(44)
    q = torch.randn((S, H, D + tail_D))
    kv = torch.randn((S, D + tail_D))
    if kind == "sparse_inf":
        q[100, 0, 0] = float("inf")
        q[1500, 3, 5] = float("-inf")
        kv[200, 7] = float("inf")
    elif kind == "sparse_nan":
        q[300, 1, 2] = float("nan")
        kv[700, 11] = float("nan")
        q[1024, 8, 4] = float("nan")
    elif kind == "all_zero":
        q = torch.zeros((S, H, D + tail_D))
        kv = torch.zeros((S, D + tail_D))
    elif kind == "bf16_max_sparse":
        big = torch.finfo(torch.bfloat16).max
        q[100, 0, 0] = big
        q[1500, 3, 5] = -big
        kv[200, 7] = big
    q = q.bfloat16().npu()
    kv = kv.bfloat16().npu()
    offsets = torch.tensor([0, 1023, S], dtype=torch.int32).npu()
    p = torch.arange(topk, dtype=torch.int32)
    idx = p.unsqueeze(0).repeat(S, 1).contiguous().npu()
    return q, kv, idx, offsets


def _check_boundary(name, reducesum, ref_attn):
    """Boundary comparison: strict NaN/+Inf/-Inf structural masks first, then
    the §9.3 mixed tolerance on jointly-finite positions. Non-blocking."""
    a = reducesum_to_attn(reducesum).float()
    r = ref_attn.float()
    if a.shape != r.shape:
        print(f"[BOUNDARY_WARN] boundary {name}: shape mismatch {tuple(a.shape)} vs {tuple(r.shape)}")
        return
    mask_ok = (
        torch.equal(torch.isnan(a), torch.isnan(r))
        and torch.equal(torch.isinf(a) & (a > 0), torch.isinf(r) & (r > 0))
        and torch.equal(torch.isinf(a) & (a < 0), torch.isinf(r) & (r < 0))
    )
    fin = torch.isfinite(a) & torch.isfinite(r)
    if fin.sum().item() > 0:
        diff = (a[fin] - r[fin]).abs()
        matched = (diff <= ATOL + RTOL * r[fin].abs()).float().mean().item()
        max_abs = diff.max().item()
    else:
        matched, max_abs = 1.0, 0.0
    ok = mask_ok and matched >= MATCHED_RATIO_MIN and max_abs <= ABS_ERR_MAX
    tag = "[BOUNDARY_PASS]" if ok else "[BOUNDARY_WARN]"
    print(f"{tag} boundary {name}: structural_mask={'ok' if mask_ok else 'MISMATCH'}, matched_ratio={matched:.6f}, max_abs={max_abs:.3e}")


def test_sparse_mla_topk_reducesum_boundary():
    """Boundary layer: legal special values (±inf / nan / all-zero / bf16 max),
    generic path, structural-mask + tolerance comparison. Non-blocking."""
    for name, _tag in BOUNDARY_CASES:
        try:
            q, kv, idx, offsets = _make_boundary_inputs(name)
            lse, ref_attn = ref_mla_topk_softmax_masked(q, kv, idx, offsets)
            out = sparse_mla_topk_reducesum(q, kv.unsqueeze(-2), idx.unsqueeze(-2), lse, offsets, dim_v=512, panel_mode="generic")
            _check_boundary(name, out, ref_attn)
        except Exception as e:
            print(f"[BOUNDARY_WARN] boundary {name}: {type(e).__name__}: {e}")


# Coverage summary (machine-checked by coverage_check.py; counts per dimension)
COVERAGE_MANIFEST = {
    "D-DTYPE-bf16": 1,
    "D-DTYPE-fp32": 1,
    "D-DTYPE-int32": 1,
    "D-SHAPE-ALIGNED": 1,
    "D-SHAPE-TAIL-1": 1,
    "D-SHAPE-TAIL-MID": 1,
    "D-SHAPE-PRIME": 1,
    "D-SHAPE-EDGE": 1,
    "D-VALRANGE-S": 1,
    "D-VALRANGE-M": 1,
    "D-VALRANGE-L": 1,
    "D-VALRANGE-ASYM": 1,
    "D-PARAM-dim_v": 1,
    "D-PARAM-kv_group": 1,
    "D-PARAM-sm_scale": 1,
    "D-PARAM-block_I": 1,
    "D-PARAM-grid_size": 1,
    # rev2 三值枚举（auto/generic/shared）：task_tail_1（shared）+ panel_mode_sweep（三模式全跑）
    "D-PARAM-panel_mode": 2,
    "D-SPECIAL-INF": 1,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-DBOUND": 1,
    "D-EXC-DTYPE": 1,
    "D-EXC-SHAPE": 2,
}
# Fusion category: no exemptible dimensions are left uncovered (all covered).
COVERAGE_NA = {}


def main():
    parser = argparse.ArgumentParser(description="sparse_mla_topk_reducesum layered tests")
    parser.add_argument(
        "--level",
        choices=["l0", "l1", "l2", "boundary", "all"],
        default="l0",
        help="test layer to run (default: l0)",
    )
    args = parser.parse_args()

    tilelang.disable_cache()  # avoid stale compile artifacts (repo convention)

    level = args.level
    ok = True
    if level in ("l0", "all"):
        ok = test_sparse_mla_topk_reducesum_l0() and ok
    if level in ("l1", "all"):
        ok = test_sparse_mla_topk_reducesum_l1() and ok
    if level in ("l2", "all"):
        test_sparse_mla_topk_reducesum_l2()
    if level in ("boundary", "all"):
        test_sparse_mla_topk_reducesum_boundary()

    if ok:
        print("\nAll requested layers passed.")
        print("Test Passed!")
        sys.exit(0)
    print("\nAt least one blocking case failed.")
    sys.exit(1)


if __name__ == "__main__":
    main()
