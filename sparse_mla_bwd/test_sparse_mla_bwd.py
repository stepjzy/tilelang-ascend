"""sparse_mla_bwd layered tests: L0 gate (DESIGN §12.3, 9 cases) + stubs.

Golden replicates the main repo's ref_sparse_mla_bwd_interface
(ref_gpu/sparse_mla_bwd.py L281-290, autograd through ref_sparse_mla_fwd_interface)
and runs it on NPU. Input construction replicates the main repo
test_sparse_mla_bwd L293-320 line by line (randn bf16 q/kv/do, per-token
randperm(max(1,t))[:topk] causal sampling with out-of-range tail fill S,
offsets, seq_len >= topk assert) with torch.manual_seed(0) per DESIGN §12.5
(deviation declaration: fixed seed, generation flow unchanged).

Comparison strategy (DESIGN §12.1 dual gate + §12.2 amend-2 value-domain
layering):
  1. main repo criterion: assert_tensors_similar(eps=1e-4) global normalized
     dot-product similarity (ref_gpu/utils.py verbatim port, fp64 on CPU),
     dq / dkv judged separately;
  2. bf16 mixed tolerance (atol 2^-10 / rtol 2^-6 / max_abs 1e0 /
     ratio >= 0.99) for element-level localization.
  Layering (amend-2): domain A (randn sigma=1, default sm_scale — all L0 +
     L1 shape axis) keeps the dual gate verbatim; domain B (sigma<=10 / any
     sm_scale — smscale, valrange-s, valrange-asym, valrange-m) blocks on
     sim >= 0.999 (amend-2 tightened from 0.97; valrange-m promoted from
     domain C) with the mixed tolerance as a non-blocking reference;
     domain C (sigma>=100 — valrange-l) is WARN-level reference, never
     blocks.

Forward (o, lse) supply (DESIGN §11.4 + amend-1 §14 R4): sparse_mla_fwd
kernel interface (parallel-session deliverable) when importable AND its
(o, lse) pass the sanity gate (o similarity eps=1e-2 vs ref; lse fp32 mixed
tolerance vs the SLOT-based reference lse — slot semantics matches the
kernel incl. repeat multiplicity); otherwise the in-test PyTorch fallback
(o = ref fwd output, lse = slot-based fp32 logsumexp) is used and a warning
is recorded. SPARSE_MLA_BWD=torch forces the fallback for the whole run
(dev-time isolation of fwd-session instability; final acceptance must run
kernel mode). The fallback is same-source as the golden's internal forward,
so the bwd judgment is unaffected by which path supplies (o, lse).

Levels:
    l0        DESIGN §12.3 gate cases (blocking, [PRECISION_PASS/FAIL])
    l1        functional (stub here; extended via scene B after L0 passes)
    l2        negative (stub; host domain asserts, non-blocking)
    boundary  special values (stub; non-blocking, [BOUNDARY_PASS/WARN])
"""

import argparse
import os
import sys
import traceback

import torch

import tilelang

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparse_mla_bwd import sparse_mla_bwd  # noqa: E402

try:
    from sparse_mla_fwd import sparse_mla_fwd_interface  # noqa: E402

    FWD_KERNEL_AVAILABLE = True
except ImportError:
    sparse_mla_fwd_interface = None  # type: ignore[assignment]
    FWD_KERNEL_AVAILABLE = False

# Forward-supply mode switch (DESIGN amend-1 §14 R4): the parallel fwd
# session's in-flight iterations can poison the shared device stream (the
# 2026-10-14 19:12 aicore crash record). SPARSE_MLA_BWD_FWD=kernel (default)
# keeps the sanity-gated fwd kernel path; =torch forces the exact PyTorch
# (o, lse) pair for development isolation (same-source as the golden's
# internal forward, so the bwd judgment is unaffected). FINAL ACCEPTANCE
# MUST RUN IN kernel MODE with the fwd session stable.
_FWD_MODE = os.environ.get("SPARSE_MLA_BWD_FWD", "kernel").strip().lower()
if _FWD_MODE not in ("kernel", "torch"):
    raise SystemExit(f"SPARSE_MLA_BWD_FWD must be 'kernel' or 'torch', got '{_FWD_MODE}'")
FWD_FORCE_TORCH = _FWD_MODE == "torch"


# ============================================================================
# Similarity criterion (main repo ref_gpu/utils.py L18-73, verbatim port)
# ============================================================================
def calculate_tensor_similarity(x, y, name="tensor"):
    """
    Calculate similarity between two tensors using a normalized dot product metric.

    Unlike torch.testing.assert_close which uses absolute/relative tolerance based on
    element-wise differences, this function computes a global similarity score:
        sim = 2 * <x, y> / (||x||^2 + ||y||^2)

    This metric is scale-invariant and measures the cosine-like similarity normalized
    by the magnitude of both tensors. It returns 1 for identical tensors and values
    closer to 0 for dissimilar ones. This is particularly useful for comparing tensors
    with varying magnitudes where relative errors matter more than absolute differences.

    Args:
        x: First tensor to compare.
        y: Second tensor to compare.
        name: Name of the tensor for logging purposes.

    Returns:
        Similarity score in range [0, 1] where 1 means identical
    """
    x, y = x.data.double(), y.data.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        print(f"\033[33mWARNING: {name} all zero\033[0m")
        return 1
    sim = 2 * (x * y).sum() / denominator
    return sim


def assert_tensors_similar(x, y, eps=1e-8, name="tensor", raise_assert=True):
    """
    Assert that two tensors are similar using a global similarity metric.

    Key differences from torch.testing.assert_close:
    - torch.testing.assert_close: Uses element-wise comparison with rtol/atol, checking
      that |x - y| <= atol + rtol * |y| for each element. It's sensitive to outliers
      and requires all elements to satisfy the tolerance.
    - assert_tensors_similar: Uses a single global similarity score (1 - sim) where sim is the
      normalized dot product. It's more robust to outliers and focuses on overall
      tensor similarity rather than element-wise precision. This is better suited for
      comparing large tensors where a few outlier elements shouldn't fail the test.

    Args:
        x: First tensor to compare.
        y: Second tensor to compare.
        eps: Maximum allowed difference (1 - similarity), default 1e-8.
        name: Name of the tensor for error messages.
        raise_assert: Whether to raise assertion error on failure.

    Returns:
        Similarity score (1 for identical tensors).
    """
    sim = calculate_tensor_similarity(x, y, name)
    diff = 1.0 - sim
    if not (0 <= diff <= eps):
        print(f"\033[31mERROR: {name} similarity check failed, diff={diff:.2e} (threshold={eps:.2e})\033[0m")
        if raise_assert:
            assert False  # noqa: B011
    return sim


# ============================================================================
# Mixed tolerance (tilelang-op-test-design precision-standard.md §二)
# ============================================================================
def mixed_tolerance_check(got, ref, atol, rtol, max_abs_limit, required_ratio):
    """Element-level dual gate: matched_ratio >= required AND max_abs <= limit.

    Returns (passed, matched_ratio, max_abs_error).
    """
    got_f = got.detach().float()
    ref_f = ref.detach().float()
    err = (got_f - ref_f).abs()
    tol = atol + rtol * ref_f.abs()
    matched = (err <= tol).float().mean().item()
    max_abs = err.max().item()
    passed = matched >= required_ratio and max_abs <= max_abs_limit
    return passed, matched, max_abs


BF16_GATE = (2**-10, 2**-6, 1e0, 0.99)  # DESIGN §12.2: dq / dkv (bf16)
FP32_GATE = (2**-16, 2**-10, 1e-2, 0.99)  # lse sanity (fwd test addendum)


# ============================================================================
# Golden: main repo ref_gpu/sparse_mla_fwd.py L214-257, verbatim port (only
# device: "cuda" -> Q.device). OOB semantics: > sk clamped to sk -> scatter to
# the sk+1-wide mask's last column -> cut by [..., :-1] -> zero contribution;
# -1 wraps to the same last column (NPU-verified) -> zero contribution.
# ============================================================================
def ref_sparse_mla_fwd_interface(Q, KV, Indices, offsets, sm_scale=None, is_casual=True):
    Q = Q.float()
    KV = KV.float()
    all_o = []
    for i in range(offsets.shape[0] - 1):
        q = Q[None, offsets[i] : offsets[i + 1]]
        kv = KV[None, offsets[i] : offsets[i + 1]]
        indices = Indices[None, offsets[i] : offsets[i + 1]].clone()

        indices = indices.transpose(1, 2)
        b, sq, h, dim_q = q.shape
        b, sk, g, _ = kv.shape

        assert kv.shape[-1] == 576, "you should assign dim otherwise"
        dim = 512
        k = kv
        v = kv[..., :dim]

        b, _, _, dim_v = v.shape
        g_index = g
        h_index = h // g
        compressed_casual_mask = torch.arange(0, sq, dtype=torch.int32, device=Q.device).view(-1, 1) >= torch.arange(
            1 - 1, sk * 1, 1, dtype=torch.int32, device=Q.device
        ).view(1, -1)

        indices[indices > sk] = sk
        mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
        mask = mask[..., :-1]
        mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
        mask[:, :, : 1 - 1, 0] = True
        mask = mask.view(b, g_index, 1, sq, sk)

        q = q.view(b, sq, g, -1, dim_q)
        score = torch.einsum("bmghd,bngd->bghmn", q, k)
        sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
        score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
        p = score.softmax(dim=-1)
        p = p.view(b, g_index, h_index, -1, sq, sk)
        p = p.view(b, g, -1, sq, sk)
        o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
        o = o.reshape(b, sq, h, dim_v)
        all_o.append(o.squeeze(0))
    o = torch.cat(all_o, dim=0)
    return o.to(torch.bfloat16)


def ref_sparse_mla_fwd_lse_slot(Q, KV, Indices, offsets, sm_scale=None):
    """SLOT-based fp32 logsumexp over the topk index slots (kernel semantics,
    repeats counted with multiplicity; invalid slots masked to -inf).

    Equals the set-based lse on repeat-free inputs; on uniform-multiplicity
    repeat inputs it is the value the forward kernel produces (ln c offset
    cancels in the P recomputation). Returns [S, H] float32.
    """
    Q = Q.float()
    KV = KV.float()
    S, H, DDT = Q.shape
    G = KV.shape[1]
    head_kv = H // G
    if sm_scale is None:
        sm_scale = DDT**-0.5
    lse = torch.zeros(S, H, dtype=torch.float32, device=Q.device)
    off = offsets.tolist()
    chunk = 256
    for i in range(len(off) - 1):
        bos, eos = int(off[i]), int(off[i + 1])
        for g in range(G):
            for t0 in range(bos, eos, chunk):
                t1 = min(t0 + chunk, eos)
                s_i = torch.arange(t0 - bos, t1 - bos, device=Q.device)  # [n]
                idx = Indices[t0:t1, g].long()  # [n, topk]
                valid = (idx >= 0) & (idx <= s_i[:, None])  # [n, topk]
                rows = (idx + bos).clamp(0, S - 1)  # [n, topk]
                k_g = KV[rows, g, :]  # [n, topk, DDT] fp32
                q_h = Q[t0:t1, g * head_kv : (g + 1) * head_kv, :]  # [n, hk, DDT]
                score = torch.einsum("nhd,njd->nhj", q_h, k_g) * sm_scale
                score = score.masked_fill(~valid[:, None, :], float("-inf"))
                lse[t0:t1, g * head_kv : (g + 1) * head_kv] = torch.logsumexp(score, dim=-1)
    return lse


def ref_sparse_mla_bwd_interface(q, kv, do, indices, offsets, sm_scale=None, is_casual=True):
    """Main repo ref_gpu/sparse_mla_bwd.py L281-290 verbatim (device follows
    inputs). Returns (q.grad, kv.grad) after o.backward(do)."""
    q = q.detach().clone()
    kv = kv.detach().clone()
    q.requires_grad = True
    kv.requires_grad = True
    o = ref_sparse_mla_fwd_interface(q, kv, indices, offsets, sm_scale, is_casual)
    o.backward(do)
    return q.grad, kv.grad


# ============================================================================
# Input construction (main repo test_sparse_mla_bwd L293-320 replica, line by
# line; seed fixed per DESIGN §12.5). variants: "natural" (main repo),
# "oob" (extra out-of-range scatter into valid slots), "neg1" (-1 inserts),
# "repeat" (uniform x2 duplication).
# ============================================================================
def make_case_inputs(S, H, HKV, topk, offsets, variant="natural", seed=0, device="npu", scale=1.0, shift=0.0):
    """Build (q, kv, do, indices, offsets) per the main-repo flow + variant.

    Main-repo flow preserved: randn bf16 q/kv/do; indices filled with the
    out-of-range value S; per-token randperm(max(1, t))[:topk] causal prefix;
    seq_len >= topk assert per segment. scale/shift (L1 value-range axis)
    apply q/kv/do = bf16(randn * scale + shift) — scale=1, shift=0 keeps the
    main-repo grain exactly.
    """
    DQKV, DV = 576, 512

    def rnd(*shape):
        if scale == 1.0 and shift == 0.0:
            return torch.randn(*shape, dtype=torch.bfloat16, device=device)
        return (torch.randn(*shape, dtype=torch.float32, device=device) * scale + shift).to(torch.bfloat16)

    torch.manual_seed(seed)
    q = rnd(S, H, DQKV).requires_grad_(True)
    kv = rnd(S, HKV, DQKV).requires_grad_(True)
    do = rnd(S, H, DV)
    offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    indices = torch.full((S, HKV, topk), S, dtype=torch.int32, device=device)
    for i in range(offsets.shape[0] - 1):
        seq_len = (offsets[i + 1] - offsets[i]).item()
        assert seq_len >= topk  # main repo L303
        for t in range(seq_len):
            for h in range(HKV):
                if variant == "repeat":
                    # uniform x2 duplication: all valid slots multiplicity 2
                    base = torch.randperm(max(1, t))[: min(max(1, t), topk // 2)]
                    row = torch.cat([base, base])
                    indices[offsets[i] + t, h, : len(row)] = row.to(torch.int32)
                else:
                    i_i = torch.randperm(max(1, t))[:topk]
                    indices[offsets[i] + t, h, : len(i_i)] = i_i.to(torch.int32)
        if variant == "oob":
            # OOB stress: replace ~25% of each token's valid slots with S
            # (>= 1 valid slot always kept)
            for t in range(seq_len):
                n_valid = min(t, topk)
                if n_valid >= 8:
                    n_replace = n_valid // 4
                    slots = torch.randperm(n_valid)[:n_replace]
                    indices[offsets[i] + t, 0, slots] = S
        elif variant == "neg1":
            # -1 inserts: kill at most half of each token's valid slots
            # (>= 1 valid always kept -> no all-masked token -> no lse=-inf)
            for t in range(seq_len):
                n_valid = min(t, topk)
                if n_valid >= 2:
                    n_neg = min(8, max(1, n_valid // 2))
                    slots = torch.randperm(topk)[:n_neg]
                    indices[offsets[i] + t, 0, slots] = -1
    return q, kv, do, indices, offsets


# ============================================================================
# Forward (o, lse) supply: fwd kernel (sanity-gated) or PyTorch fallback.
# ============================================================================
_FWD_WARNED = []


def get_fwd_o_lse(q, kv, indices, offsets, ref_o, ref_lse, case_id, sm_scale=None):
    """Return (o, lse, mode). Prefers the sparse_mla_fwd kernel when its
    outputs pass the sanity gate (o similarity 1e-2 + slot-lse fp32 gate);
    falls back to the exact PyTorch pair otherwise (recorded).
    SPARSE_MLA_BWD_FWD=torch skips the kernel attempt entirely (R4)."""
    if FWD_KERNEL_AVAILABLE and not FWD_FORCE_TORCH:
        try:
            o_k, lse_k = sparse_mla_fwd_interface(q, kv, indices, offsets, sm_scale=sm_scale)
            sim_ok = True
            try:
                assert_tensors_similar(o_k, ref_o, eps=1e-2, name=f"{case_id} fwd o sanity", raise_assert=False)
            except AssertionError:
                sim_ok = False
            lse_ok, _, _ = mixed_tolerance_check(lse_k, ref_lse, *FP32_GATE)
            if sim_ok and lse_ok:
                return o_k, lse_k, "fwd_kernel"
            msg = f"{case_id}: fwd kernel sanity FAILED (o_ok={sim_ok}, lse_ok={lse_ok}) -> fallback"
        except Exception as e:  # noqa: BLE001
            msg = f"{case_id}: fwd kernel call FAILED ({type(e).__name__}: {e}) -> fallback"
        print(f"\033[33mWARNING: {msg}\033[0m")
        _FWD_WARNED.append(msg)
    # fallback: exact PyTorch pair (same-source as the golden's internal fwd)
    o_f = ref_sparse_mla_fwd_interface(q, kv, indices, offsets, sm_scale)
    lse_f = ref_sparse_mla_fwd_lse_slot(q, kv, indices, offsets, sm_scale)
    return o_f, lse_f, "torch_forced" if FWD_FORCE_TORCH else "torch_fallback"


# ============================================================================
# L0 case runner (DESIGN §12.1 dual gate; golden computed once per input
# config and shared across backend legs / L0-1 vs L0-3)
# ============================================================================
_GOLDEN_CACHE: dict = {}


def _golden_bwd_with_o(q, kv, do, indices, offsets, sm_scale=None):
    """Autograd golden (main-repo ref_sparse_mla_bwd_interface inlined),
    capturing the golden forward output o for the sanity gate."""
    q2 = q.detach().clone()
    kv2 = kv.detach().clone()
    q2.requires_grad = True
    kv2.requires_grad = True
    o = ref_sparse_mla_fwd_interface(q2, kv2, indices, offsets, sm_scale)
    o.backward(do)
    return q2.grad, kv2.grad, o.detach()


def run_gate_case(
    level,
    case_id,
    S,
    H,
    HKV,
    topk,
    offsets,
    variant="natural",
    targets=("ascendc",),
    seed=0,
    delta_explicit=False,
    scale=1.0,
    shift=0.0,
    sm_scale=None,
    return_kernel=False,
    gate="A",
):
    """Run one gate case (L0/L1): construct inputs (seeded), golden
    (cached per input config), forward supply (sanity-gated), bwd on each
    target, layered judgment. Returns case_ok (blocking outcome).

    gate (DESIGN §12.2 amend-2 value-domain layering):
      "A" main-repo domain — dual blocking gate verbatim (sim 1-1e-4 AND
          bf16 mixed tolerance);
      "B" extended value domain (sigma<=10, any sm_scale) — blocking main
          criterion calibrated to sim >= 0.999 (dq/dkv separate; amend-2
          tightened from 0.97 and promoted valrange-m/σ=10 into this
          domain); the mixed tolerance degrades to a non-blocking
          reference (printed, WARN annotated when unmet);
      "C" high-concentration cancellation domain (sigma>=100 / large shift)
          — whole-case WARN-level reference: both criteria printed as-is,
          verdict recorded as WARN, never blocks [PRECISION_PASS]."""
    key = (S, H, HKV, topk, tuple(offsets), variant, seed, scale, shift, sm_scale)
    if key in _GOLDEN_CACHE:
        q, kv, do, indices, off_t, ref_dq, ref_dkv, ref_o = _GOLDEN_CACHE[key]
    else:
        q, kv, do, indices, off_t = make_case_inputs(
            S,
            H,
            HKV,
            topk,
            offsets,
            variant=variant,
            seed=seed,
            scale=scale,
            shift=shift,
        )
        print(
            f"[{case_id}] computing golden (S={S}, H={H}, G={HKV}, topk={topk}, {variant}, scale={scale}, shift={shift})...",
            flush=True,
        )
        ref_dq, ref_dkv, ref_o = _golden_bwd_with_o(q, kv, do, indices, off_t, sm_scale)
        _GOLDEN_CACHE[key] = (q, kv, do, indices, off_t, ref_dq, ref_dkv, ref_o)

    ref_lse = ref_sparse_mla_fwd_lse_slot(q, kv, indices, off_t, sm_scale)
    o_in, lse_in, fwd_mode = get_fwd_o_lse(q, kv, indices, off_t, ref_o, ref_lse, case_id, sm_scale)

    case_ok = True
    for tgt in targets:
        delta_arg = None
        if delta_explicit:
            # fp32 preprocess of the o actually passed (tests the bypass)
            delta_arg = (o_in.float() * do.float()).sum(-1)
        try:
            tl_dq, tl_dkv = sparse_mla_bwd(
                q,
                kv,
                o_in,
                do,
                indices,
                lse_in,
                off_t,
                sm_scale=sm_scale,
                is_casual=True,
                return_kernel=return_kernel,
                delta=delta_arg,
                target=tgt,
            )
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            print(f"[{case_id}/{tgt}] [PRECISION_FAIL] kernel exception")
            case_ok = False
            continue
        # gate 1: main-repo global similarity (fp64 on CPU), dq / dkv separate
        sim_dq = calculate_tensor_similarity(tl_dq.cpu(), ref_dq.cpu(), f"{case_id} dq")
        sim_dkv = calculate_tensor_similarity(tl_dkv.cpu(), ref_dkv.cpu(), f"{case_id} dkv")
        # gate 2: bf16 mixed tolerance
        ok_dq, ratio_dq, max_dq = mixed_tolerance_check(tl_dq, ref_dq, *BF16_GATE)
        ok_dkv, ratio_dkv, max_dkv = mixed_tolerance_check(tl_dkv, ref_dkv, *BF16_GATE)
        # layered judgment (§12.2 amend-2)
        if gate == "A":
            ok_sim = (1.0 - sim_dq) <= 1e-4 and (1.0 - sim_dkv) <= 1e-4
            ok = ok_sim and ok_dq and ok_dkv
            marker = "[PRECISION_PASS]" if ok else "[PRECISION_FAIL]"
            note = ""
        elif gate == "B":
            ok_sim = sim_dq >= 0.999 and sim_dkv >= 0.999
            ok = ok_sim
            marker = "[PRECISION_PASS]" if ok else "[PRECISION_FAIL]"
            note = "" if (ok_dq and ok_dkv) else " | ratio-ref WARN (non-blocking, §12.2 amend-2)"
        else:  # gate "C": domain-C reference case, WARN level, never blocks
            ok = True
            marker = "[BOUNDARY_WARN]"
            note = " | domain-C reference case (§12.2 amend-2, golden bf16-P grain bound)"
        case_ok &= ok
        print(
            f"[{case_id}/{tgt}] fwd={fwd_mode} gate={gate} "
            f"dq sim={sim_dq:.6f} ratio={ratio_dq:.4f} max={max_dq:.3e} | "
            f"dkv sim={sim_dkv:.6f} ratio={ratio_dkv:.4f} max={max_dkv:.3e} | "
            f"{marker}{note}",
            flush=True,
        )
    return case_ok


def run_l0_case(
    case_id,
    S,
    H,
    HKV,
    topk,
    offsets,
    variant="natural",
    targets=("ascendc",),
    seed=0,
    delta_explicit=False,
):
    """L0 wrapper (DESIGN §12.3 configs, no value-range axes)."""
    return run_gate_case(
        "l0",
        case_id,
        S,
        H,
        HKV,
        topk,
        offsets,
        variant=variant,
        targets=targets,
        seed=seed,
        delta_explicit=delta_explicit,
    )


# ============================================================================
# L0 gate (DESIGN §12.3, 9 cases). L0-3 shares L0-1's inputs+golden via the
# cache; every other case runs both backends (dual-backend hard requirement).
# ============================================================================
def test_sparse_mla_bwd_l0():
    cases = [
        # (id, S, H, HKV, topk, offsets, variant, targets, seed, delta_explicit)
        ("L0-1", 2048, 64, 1, 512, [0, 2048], "natural", ("ascendc",), 0, False),
        ("L0-2", 128, 64, 1, 128, [0, 128], "natural", ("ascendc", "pto"), 0, False),
        ("L0-3", 2048, 64, 1, 512, [0, 2048], "natural", ("pto",), 0, False),
        ("L0-4", 128, 64, 1, 128, [0, 128], "natural", ("ascendc", "pto"), 0, True),
        ("L0-5", 256, 64, 1, 128, [0, 256], "oob", ("ascendc", "pto"), 0, False),
        ("L0-6", 256, 64, 1, 128, [0, 256], "neg1", ("ascendc", "pto"), 0, False),
        ("L0-7", 256, 64, 1, 128, [0, 256], "repeat", ("ascendc", "pto"), 0, False),
        ("L0-8", 640, 64, 1, 128, [0, 128, 512, 640], "natural", ("ascendc", "pto"), 0, False),
        ("L0-9", 645, 64, 1, 128, [0, 129, 645], "natural", ("ascendc", "pto"), 0, False),
    ]
    n_pass = 0
    for case_id, S, H, HKV, topk, offsets, variant, targets, seed, d_exp in cases:
        ok = run_l0_case(
            case_id,
            S,
            H,
            HKV,
            topk,
            offsets,
            variant=variant,
            targets=targets,
            seed=seed,
            delta_explicit=d_exp,
        )
        n_pass += ok
    print(f"L0: {n_pass}/{len(cases)} cases passed")
    return n_pass == len(cases)


# ============================================================================
# L1 functional (scene B, deterministic shapes per coverage-matrix §6 mapped
# to this operator's tiling axes: S / segment lengths are runtime-dynamic at
# token granularity — "alignment" = S multiple of topk; tail/prime axis = S
# and segment lengths; M/N/K of the GEMMs are interface-fixed). Every case
# runs both backends. dtype dims: fixed combo (bf16 in / int32 idx / fp32
# lse) is exercised by every case.
# ============================================================================
L1_CASES = [
    # (case_id, S, H, HKV, topk, offsets, kwargs, tags)
    (
        "L1-aligned",
        512,
        64,
        1,
        128,
        [0, 512],
        {},
        ["D-DTYPE-bf16", "D-DTYPE-int32", "D-DTYPE-fp32", "D-SHAPE-ALIGNED"],
    ),
    (
        "L1-tail1",
        513,
        64,
        1,
        128,
        [0, 513],
        {},
        ["D-SHAPE-TAIL-1"],
    ),
    (
        "L1-tailmid",
        576,
        64,
        1,
        128,
        [0, 576],
        {},
        ["D-SHAPE-TAIL-MID"],
    ),
    (
        "L1-prime",
        509,
        64,
        1,
        128,
        [0, 509],
        {},
        ["D-SHAPE-PRIME"],
    ),
    (
        "L1-edge",
        256,
        64,
        1,
        256,
        [0, 256],
        {},
        ["D-SHAPE-EDGE", "D-PARAM-topk"],
    ),
    (
        "L1-varlen-mixed",
        1152,
        64,
        1,
        128,
        [0, 129, 642, 1152],
        {},
        ["D-SHAPE-TAIL-1", "D-SHAPE-TAIL-MID"],
    ),
    (
        "L1-kvgroup2",
        256,
        64,
        2,
        128,
        [0, 256],
        {},
        ["D-PARAM-kv_group"],
    ),
    (
        "L1-heads32",
        256,
        32,
        1,
        128,
        [0, 256],
        {},
        ["D-PARAM-heads"],
    ),
    (
        "L1-topk256",
        512,
        64,
        1,
        256,
        [0, 512],
        {},
        ["D-PARAM-topk", "D-SHAPE-ALIGNED"],
    ),
    (
        "L1-smscale",
        256,
        64,
        1,
        128,
        [0, 256],
        {"sm_scale": 0.1, "gate": "B"},
        ["D-PARAM-sm_scale"],
    ),
    (
        "L1-return-kernel",
        128,
        64,
        1,
        128,
        [0, 128],
        {"return_kernel": True},
        ["D-PARAM-return_kernel"],
    ),
    (
        "L1-valrange-s",
        256,
        64,
        1,
        128,
        [0, 256],
        {"scale": 0.5, "gate": "B"},
        ["D-VALRANGE-S"],
    ),
    (
        "L1-valrange-m",
        256,
        64,
        1,
        128,
        [0, 256],
        {"scale": 10.0, "gate": "B"},
        ["D-VALRANGE-M"],
    ),
    (
        "L1-valrange-l",
        256,
        64,
        1,
        128,
        [0, 256],
        {"scale": 100.0, "gate": "C"},
        ["D-VALRANGE-L"],
    ),
    (
        "L1-valrange-asym",
        256,
        64,
        1,
        128,
        [0, 256],
        {"scale": 3.0, "shift": 6.0, "gate": "B"},
        ["D-VALRANGE-ASYM"],
    ),
]


def test_sparse_mla_bwd_l1():
    """L1 functional tests (scene B): deterministic shape/param/value-range
    coverage, both backends, layered gate per §12.2 amend-2 (A blocking /
    B sim>=0.999 blocking / C WARN-level reference, non-blocking)."""
    n_pass = 0
    n_ref_c = 0
    for case_id, S, H, HKV, topk, offsets, kwargs, _tags in L1_CASES:
        gate = kwargs.get("gate", "A")
        n_ref_c += gate == "C"
        ok = run_gate_case(
            "l1",
            case_id,
            S,
            H,
            HKV,
            topk,
            offsets,
            targets=("ascendc", "pto"),
            **kwargs,
        )
        n_pass += ok
    if n_ref_c:
        print(
            f"L1: {n_pass}/{len(L1_CASES)} cases passed "
            f"({n_ref_c} domain-C reference case(s) recorded WARN, non-blocking per §12.2 amend-2)"
        )
    else:
        print(f"L1: {n_pass}/{len(L1_CASES)} cases passed")
    return n_pass == len(L1_CASES)


# ============================================================================
# L2 negative (non-blocking): illegal inputs must be rejected — a raised
# exception is a PASS, silent acceptance is a WARN.
# ============================================================================
def _run_exception(name, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_PASS] l2 {name}: 正确拒绝 ({type(e).__name__}: {e})")
        return
    print(f"[BOUNDARY_WARN] l2 {name}: 非法输入未被拒绝（静默接受）")


def _l2_base_inputs(S=128, H=64, HKV=1, topk=128):
    return make_case_inputs(S, H, HKV, topk, [0, S], seed=0)


def test_sparse_mla_bwd_l2():
    """L2 negative tests: unsupported dtypes / illegal shapes / domain
    violations must be rejected by the host asserts (non-blocking)."""
    q, kv, do, indices, lse_src, off = None, None, None, None, None, None

    def _mk():
        nonlocal q, kv, do, indices, lse_src, off
        q, kv, do, indices, off = _l2_base_inputs()
        # a consistent (o, lse) via the in-test fallback path
        o = ref_sparse_mla_fwd_interface(q, kv, indices, off)
        lse = ref_sparse_mla_fwd_lse_slot(q, kv, indices, off)
        return q, kv, o, do, indices, lse, off

    # D-EXC-DTYPE: q in fp32
    def _dtype_q_fp32():
        q, kv, o, do, indices, lse, off = _mk()
        sparse_mla_bwd(q.float(), kv, o, do, indices, lse, off)

    _run_exception("dtype_q_fp32", _dtype_q_fp32)

    # D-EXC-DTYPE: indices int64
    def _dtype_indices_i64():
        q, kv, o, do, indices, lse, off = _mk()
        sparse_mla_bwd(q, kv, o, do, indices.long(), lse, off)

    _run_exception("dtype_indices_int64", _dtype_indices_i64)

    # D-EXC-DTYPE: lse bf16
    def _dtype_lse_bf16():
        q, kv, o, do, indices, lse, off = _mk()
        sparse_mla_bwd(q, kv, o, do, indices, lse.to(torch.bfloat16), off)

    _run_exception("dtype_lse_bf16", _dtype_lse_bf16)

    # D-EXC-SHAPE: topk not divisible by 128
    def _shape_topk_100():
        S, H, HKV = 128, 64, 1
        q, kv, do, indices, off = make_case_inputs(S, H, HKV, 128, [0, S], seed=0)
        indices = indices[..., :100]
        o = ref_sparse_mla_fwd_interface(q, kv, indices, off)
        lse = ref_sparse_mla_fwd_lse_slot(q, kv, indices, off)
        sparse_mla_bwd(q, kv, o, do, indices, lse, off)

    _run_exception("shape_topk_100_not_div_128", _shape_topk_100)

    # D-EXC-SHAPE: kv rows mismatch
    def _shape_skv_mismatch():
        q, kv, o, do, indices, lse, off = _mk()
        sparse_mla_bwd(q, kv[:-1], o, do, indices, lse, off)

    _run_exception("shape_skv_mismatch", _shape_skv_mismatch)

    # D-EXC-SHAPE: lse shape wrong
    def _shape_lse_wrong():
        q, kv, o, do, indices, lse, off = _mk()
        sparse_mla_bwd(q, kv, o, do, indices, lse[:, :32], off)

    _run_exception("shape_lse_wrong", _shape_lse_wrong)

    # D-EXC-SHAPE: head_kv domain (H=65 -> 65 % 16 != 0)
    def _shape_headkv_domain():
        S, H, HKV, topk = 128, 65, 1, 128
        q, kv, do, indices, off = make_case_inputs(S, 64, HKV, topk, [0, S], seed=0)
        q65 = torch.randn(S, H, 576, dtype=torch.bfloat16, device=q.device)
        o65 = torch.randn(S, H, 512, dtype=torch.bfloat16, device=q.device)
        lse65 = torch.randn(S, H, dtype=torch.float32, device=q.device)
        sparse_mla_bwd(q65, kv, o65, do, indices, lse65, off)

    _run_exception("shape_headkv_65_domain", _shape_headkv_domain)

    # D-PARAM-is_casual (only True supported; False must be rejected)
    def _param_is_casual_false():
        q, kv, o, do, indices, lse, off = _mk()
        sparse_mla_bwd(q, kv, o, do, indices, lse, off, is_casual=False)

    _run_exception("param_is_casual_false", _param_is_casual_false)

    # delta with wrong dtype
    def _delta_bf16():
        q, kv, o, do, indices, lse, off = _mk()
        delta = (o.float() * do.float()).sum(-1).to(torch.bfloat16)
        sparse_mla_bwd(q, kv, o, do, indices, lse, off, delta=delta)

    _run_exception("dtype_delta_bf16", _delta_bf16)


# ============================================================================
# Boundary (non-blocking): legal special values, precision-checked with the
# same bf16 gate (structural NaN/inf handling per precision-standard.md);
# failures record [BOUNDARY_WARN]. ascendc leg only (diagnostic tier).
# ============================================================================
def check_precision_structural(actual, golden, atol, rtol, max_abs_limit, required_ratio):
    """Mixed tolerance with structural NaN/inf comparison (positions must
    match; finite positions compared numerically)."""
    a = actual.detach().float()
    g = golden.detach().float()
    special = ~torch.isfinite(g)
    if special.any():
        a_s, g_s = a[special], g[special]
        if not torch.equal(torch.isnan(a_s), torch.isnan(g_s)) or not torch.equal(torch.isinf(a_s), torch.isinf(g_s)):
            return False, 0.0, float("inf")
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    err = (a[m] - g[m]).abs()
    matched = (err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = err.max().item()
    return matched >= required_ratio and max_abs <= max_abs_limit, matched, max_abs


def _run_boundary_case(name, tags, S, H, HKV, topk, scale=None, special=None):
    """One boundary case: special-value inputs, golden, ascendc bwd,
    structural precision check (non-blocking)."""
    q, kv, do, indices, off = make_case_inputs(S, H, HKV, topk, [0, S], seed=0)
    if scale is not None:
        q = (q.float() * scale).to(torch.bfloat16)
        kv = (kv.float() * scale).to(torch.bfloat16)
        do = (do.float() * scale).to(torch.bfloat16)
    if special == "zero":
        q = torch.zeros_like(q)
        kv = torch.zeros_like(kv)
        do = torch.zeros_like(do)
    elif special == "inf_q":
        q = q.float()
        q[0, :8] = float("inf")
        q = q.to(torch.bfloat16)
    elif special == "nan_kv":
        kv = kv.float()
        kv[0, 0, :64] = float("nan")
        kv = kv.to(torch.bfloat16)
    try:
        ref_dq, ref_dkv, ref_o = _golden_bwd_with_o(q, kv, do, indices, off)
        ref_lse = ref_sparse_mla_fwd_lse_slot(q, kv, indices, off)
        o_in, lse_in, _mode = get_fwd_o_lse(q, kv, indices, off, ref_o, ref_lse, f"boundary-{name}")
        tl_dq, tl_dkv = sparse_mla_bwd(q, kv, o_in, do, indices, lse_in, off)
        ok_q, ratio_q, max_q = check_precision_structural(tl_dq, ref_dq, *BF16_GATE)
        ok_k, ratio_k, max_k = check_precision_structural(tl_dkv, ref_dkv, *BF16_GATE)
        ok = ok_q and ok_k
        print(
            f"[BOUNDARY_{'PASS' if ok else 'WARN'}] boundary {name} "
            f"dq ratio={ratio_q:.4f} max={max_q:.3e} | "
            f"dkv ratio={ratio_k:.4f} max={max_k:.3e}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_WARN] boundary {name}: {type(e).__name__}: {e}")


def test_sparse_mla_bwd_boundary():
    """Boundary tests: zero / inf / nan / large-magnitude inputs
    (non-blocking, structural + mixed tolerance)."""
    _run_boundary_case("all_zero", ["D-SPECIAL-ZERO"], 128, 64, 1, 128, special="zero")
    _run_boundary_case("inf_q", ["D-SPECIAL-INF"], 128, 64, 1, 128, special="inf_q")
    _run_boundary_case("nan_kv", ["D-SPECIAL-NAN"], 128, 64, 1, 128, special="nan_kv")
    _run_boundary_case("dbound_large_scale", ["D-SPECIAL-DBOUND"], 128, 64, 1, 128, scale=2.0e4)


# ============================================================================
# Coverage manifest (coverage_check.py; Fusion category — all dimensions
# mandatory). Tags live in L1_CASES / boundary / l2 call sites above.
# ============================================================================
COVERAGE_CATEGORY = "Fusion"
COVERAGE_MANIFEST = {
    "D-DTYPE-bf16": 15,  # every L0/L1 case (fixed bf16 inputs)
    "D-DTYPE-int32": 15,  # indices/offsets in every case
    "D-DTYPE-fp32": 15,  # lse (and explicit-delta L0-4) in every case
    "D-SHAPE-ALIGNED": 3,  # L1-aligned / L1-topk256 / L0 cases
    "D-SHAPE-TAIL-1": 2,
    "D-SHAPE-TAIL-MID": 2,
    "D-SHAPE-PRIME": 1,
    "D-SHAPE-EDGE": 1,
    "D-PARAM-heads": 1,
    "D-PARAM-kv_group": 1,
    "D-PARAM-topk": 2,
    "D-PARAM-sm_scale": 1,
    "D-PARAM-is_casual": 1,  # non-default (False) exercised via L2 rejection
    "D-PARAM-return_kernel": 1,
    "D-VALRANGE-S": 1,
    "D-VALRANGE-M": 1,
    "D-VALRANGE-L": 1,
    "D-VALRANGE-ASYM": 1,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-INF": 1,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-DBOUND": 1,
    "D-EXC-DTYPE": 4,
    "D-EXC-SHAPE": 4,
}
COVERAGE_NA = {
    # Fusion has no exemptable dimensions; interface-fixed aspects noted:
    # "D-SHAPE-RANK-*": tensors are fixed-rank by the interface contract.
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="all", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()  # fresh compile per run (no stale artifacts)

    levels = ["l0", "l1", "l2", "boundary"] if args.level == "all" else [args.level]
    # L0/L1 are blocking (exit 1 on FAIL); L2/Boundary record only.
    blockers = {"l0": test_sparse_mla_bwd_l0, "l1": test_sparse_mla_bwd_l1}
    recorders = {"l2": test_sparse_mla_bwd_l2, "boundary": test_sparse_mla_bwd_boundary}
    failed = False
    for lv in levels:
        if lv in blockers:
            if not blockers[lv]():
                failed = True
        else:
            recorders[lv]()
    if failed:
        print("Test Failed!")
        sys.exit(1)
    print("Test Passed!")


if __name__ == "__main__":
    main()
