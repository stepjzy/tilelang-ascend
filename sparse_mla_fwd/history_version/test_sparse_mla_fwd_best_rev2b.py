"""sparse_mla_fwd layered tests: L0 gate (DESIGN §9.2, 4 cases) + stubs.

Golden replicates the main repo's ref_sparse_mla_fwd_interface
(ref_gpu/sparse_mla_fwd.py L214-257, verbatim per DESIGN §9.1; only device
changed: "cuda" -> input device) and runs it on NPU. Input construction
replicates the main repo test_sparse_mla_fwd L274-286 line by line
(torch.random.manual_seed(0), offsets=[0, S//2-1, S], indices filled with the
out-of-range value S then per-token randperm prefix filling).

Comparison strategy (DESIGN §9.2 dual gate + Lse addendum):
  1. out (bf16): main-repo criterion assert_tensors_similar(eps=1e-2)
     (ref_gpu/utils.py verbatim port, run on CPU in fp64) AND bf16 mixed
     tolerance (atol 2^-10 / rtol 2^-6 / max_abs 1e0 / ratio >= 0.99).
  2. Lse (fp32, main repo never checks it): golden recomputed in-test from
     the ref fp32 score/mask — score.masked_fill(-inf).mul(sm_scale) then
     logsumexp(-1), first-row mask special case exactly as the main repo —
     checked with fp32 mixed tolerance (atol 2^-16 / rtol 2^-10 / max_abs
     1e-2 / ratio >= 0.99).

Levels:
    l0        DESIGN §9.2 gate cases (blocking, [PRECISION_PASS/FAIL])
    l1        functional: repeats / -1 & OOB / short segments / kv_group=2 /
              NI=1|2 / empty & single segment — kernel-semantics golden
              (blocking, [PRECISION_PASS/FAIL])
    l2        negative: host domain asserts + dtype check (correct rejection;
              recorded, non-blocking)
    boundary  special values: zero Q/KV, ±1e4 extremes, all-same index,
              inf/nan/±bf16_max probes (recorded, non-blocking,
              [BOUNDARY_PASS/WARN]; non-finite golden positions are skipped
              by the §9.3 mixed-tolerance gate, the global similarity metric
              is undefined there)
    perf      DESIGN §9.4 sentinel: l0_main shape (1/4096/128/1/576/512/1024,
              bf16), both backends, pure-kernel NPU-event timing (warmup 5 +
              20 iters) + interface-level loop (warmup 5 + 50 iters,
              tensor_cache active); target < 24288 us
"""

import argparse
import os
import sys
import traceback

import torch

import tilelang

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparse_mla_fwd import (  # noqa: E402
    _derived_head_layout,
    get_cube_core_num,
    prepare_token_indices,
    sparse_mla_fwd_interface,
    tl_sparse_mla_fwd,
)


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
        x: First tensor to compare
        y: Second tensor to compare
        name: Name of the tensor for logging purposes

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
        x: First tensor to compare
        y: Second tensor to compare
        eps: Maximum allowed difference (1 - similarity), default 1e-8
        name: Name of the tensor for error messages
        raise_assert: Whether to raise assertion error on failure
    """
    sim = calculate_tensor_similarity(x, y, name)
    diff = 1.0 - sim
    if not (0 <= diff <= eps):
        print(f"\033[31mERROR: {name} similarity check failed, diff={diff:.2e} (threshold={eps:.2e})\033[0m")
        if raise_assert:
            assert False  # noqa: B011
    return sim


# ============================================================================
# Golden reference (main repo ref_gpu/sparse_mla_fwd.py L214-257, verbatim
# per DESIGN §9.1; only change: "cuda" -> Q.device)
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


def ref_sparse_mla_fwd_lse(Q, KV, Indices, offsets, sm_scale=None):
    """Lse golden (DESIGN §9.2 addendum): recomputed in-test from the same fp32
    score/mask chain as ref_sparse_mla_fwd_interface —
    score.masked_fill(~mask, -inf).mul(sm_scale) then logsumexp(-1) — including
    the main repo's first-row mask special-case line (mask[:, :, :1-1, 0]).
    Returns [S, H] float32.
    """
    Q = Q.float()
    KV = KV.float()
    all_lse = []
    for i in range(offsets.shape[0] - 1):
        q = Q[None, offsets[i] : offsets[i + 1]]
        kv = KV[None, offsets[i] : offsets[i + 1]]
        indices = Indices[None, offsets[i] : offsets[i + 1]].clone().transpose(1, 2)
        b, sq, h, dim_q = q.shape
        b, sk, g, _ = kv.shape
        dim = 512  # noqa: F841  (main-repo verbatim chain; unused on the lse path)
        g_index = g
        h_index = h // g  # noqa: F841  (main-repo verbatim chain; unused on the lse path)
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
        score = torch.einsum("bmghd,bngd->bghmn", q, kv)
        sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
        score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
        lse = torch.logsumexp(score, dim=-1)  # [b, g, h_index, sq]
        all_lse.append(lse.squeeze(0).squeeze(0).transpose(0, 1))  # [sq, h_index]
    return torch.cat(all_lse, dim=0)


# ============================================================================
# Kernel-semantics golden (L1/Boundary). Template = A-block smoke golden
# (/tmp/opencode/smoke_fwd.py, verified on both backends). The main-repo
# scatter golden above DEDUPS repeated indices and cannot take negative
# indices, so it is valid for L0 inputs only (DESIGN §9.1/§1.4 #1: the kernel
# counts repeated contributions; invalid = idx<0 or idx>s_i, masked to -BIGC,
# addressing clamped). Segment-direct iteration is equivalent to the smoke
# template's prepare_token_indices mapping (sparse_mla_fwd.py L465-483).
# ============================================================================
def kernel_semantic_golden(q, kv, indices, offsets, sm_scale=None):
    """fp32 golden replicating kernel semantics: per (segment, token, group)
    gather KV rows at clamp(idx+bos, 0, S-1); valid = 0 <= idx <= s_i; invalid
    scores pushed to -1e30; m clamped to -2^30; softmax in m-shifted domain.
    Returns (out bf16 [S, H, 512], lse fp32 [S, H]).
    """
    S, H, DDT = q.shape
    G = kv.shape[1]
    dim = 512
    if sm_scale is None:
        sm_scale = DDT**-0.5
    off = offsets.long()
    head_kv = H // G
    out = torch.zeros(S, H, dim, dtype=torch.float32, device=q.device)
    lse = torch.zeros(S, H, dtype=torch.float32, device=q.device)
    for i in range(off.shape[0] - 1):
        bos, eos = int(off[i]), int(off[i + 1])
        for s in range(bos, eos):
            s_i = s - bos
            for g in range(G):
                idx = indices[s, g].long()
                mask = ((idx <= s_i) & (idx >= 0)).float()
                rows = (idx + bos).clamp(0, S - 1)
                k = kv[rows, g, :].float()
                v = k[:, :dim]
                qs = q[s, g * head_kv : (g + 1) * head_kv].float()
                score = (qs @ k.t()) * sm_scale
                score = score * mask[None, :] - (1.0 - mask[None, :]) * 1.0e30
                m = score.max(dim=-1).values
                m = torch.maximum(m, torch.full_like(m, -(2.0**30)))
                p = torch.exp(score - m[:, None])
                se = p.sum(-1)
                out[s, g * head_kv : (g + 1) * head_kv] = (p @ v) / se[:, None]
                lse[s, g * head_kv : (g + 1) * head_kv] = torch.log(se) + m
    return out.to(torch.bfloat16), lse


# ============================================================================
# Input construction (main repo test_sparse_mla_fwd L274-286 replica, line by
# line per DESIGN §9.2 hard fidelity constraint; only device: "cuda" -> "npu")
# ============================================================================
def make_l0_inputs(S=4096, H=128, HKV=1, DQK=576, topk=1024, dtype=torch.bfloat16, device="npu"):
    """L0 inputs: manual_seed(0), offsets=[0, S//2-1, S], indices filled with
    the out-of-range value S then per-token randperm prefix (no repeats ->
    the dedup-scatter golden is valid, DESIGN §9.1). S=4096 -> offsets
    [0, 2047, 4096] (l0_main, perf shape §9.4); S=2050 -> [0, 1024, 2050]
    (l0_small: segment lengths 1024/1026, both >= topk, non-topk-divisible).
    """
    torch.random.manual_seed(0)  # main repo L274 (hard)
    q = torch.randn((S, H, DQK), dtype=dtype, device=device)
    kv = torch.randn((S, HKV, DQK), dtype=dtype, device=device)
    offsets = torch.tensor([0, S // 2 - 1, S], dtype=torch.int32, device=device)  # main repo L277
    indices = torch.full((S, HKV, topk), S, dtype=torch.int32, device=device)  # out-of-range fill = S
    for i in range(offsets.shape[0] - 1):
        seq_len = (offsets[i + 1] - offsets[i]).item()
        assert seq_len >= topk  # main repo L282
        for t in range(seq_len):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]  # main repo L285
                indices[offsets[i] + t, h, : len(i_i)] = i_i
    return q, kv, indices, offsets


# ============================================================================
# L1/Boundary input construction (parameterized random; offsets free-form:
# empty/single/multi segments, segment length < topk allowed — no main-repo
# seq_len >= topk constraint here). Every row keeps >= 1 valid slot (slot 0 in
# [0, s_i]); the all-invalid row (sumexp=0 -> NaN, known boundary, debug_log
# A-block) is intentionally NOT constructed.
# ============================================================================
def make_varlen_inputs(
    S=128,
    H=128,
    HKV=1,
    DQK=576,
    topk=64,
    offsets_list=None,
    dtype=torch.bfloat16,
    device="npu",
    seed=0,
):
    """Random varlen inputs: same randperm-prefix scheme as make_l0_inputs,
    but offsets_list is a free-form parameter (may contain empty segments)."""
    if offsets_list is None:
        offsets_list = [0, S // 2 - 1, S]
    torch.random.manual_seed(seed)
    q = torch.randn((S, H, DQK), dtype=dtype, device=device)
    kv = torch.randn((S, HKV, DQK), dtype=dtype, device=device)
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=device)
    indices = torch.full((S, HKV, topk), S, dtype=torch.int32, device=device)
    for i in range(len(offsets_list) - 1):
        seq_len = offsets_list[i + 1] - offsets_list[i]
        for t in range(seq_len):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[offsets_list[i] + t, h, : len(i_i)] = i_i
    return q, kv, indices, offsets


def mutate_repeat(indices, offsets_list, topk, stride=7):
    """Duplicate slot-0's valid index into slots 1:11 and invalidate a few
    tail slots (kernel semantics: repeats keep contributions, DESIGN §1.4 #1)."""
    for i in range(len(offsets_list) - 1):
        for t in range(offsets_list[i], offsets_list[i + 1], stride):
            v = int(indices[t, 0, 0])  # valid by construction (<= s_i)
            indices[t, 0, 1:6] = v
            indices[t, 0, 6:11] = v
            indices[t, 0, topk - 4 :] = -1  # a few invalid slots


def mutate_neg_oob(indices, offsets_list, S, topk):
    """Sprinkle invalid/boundary probes: slot 7 = s_i (valid causal max),
    slot 6 = s_i+1 (invalid by 1), slots 8::8 cycle [-1, -5, eos (next-seg
    start), eos-1 (own-seg last row), S, S+10 (out of range)]."""
    for i in range(len(offsets_list) - 1):
        bos, eos = offsets_list[i], offsets_list[i + 1]
        for t in range(bos, eos):
            s_i = t - bos
            indices[t, 0, 6] = s_i + 1
            indices[t, 0, 7] = s_i
            cyc = [-1, -5, eos, eos - 1, S, S + 10]
            for j, slot in enumerate(range(8, topk, 8)):
                indices[t, 0, slot] = cyc[j % len(cyc)]


def mutate_same_index(indices, offsets_list, stride=9):
    """Rows whose slots are ALL the same valid index s_i (extreme repeat)."""
    for i in range(len(offsets_list) - 1):
        for t in range(offsets_list[i], offsets_list[i + 1], stride):
            indices[t, 0, :] = t - offsets_list[i]  # = s_i, always valid


# ============================================================================
# Precision standard (mixed tolerance, DESIGN §9.3)
# ============================================================================
def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio)."""
    fp_table = {
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        return (0.0, 0.0, 0.0, 1.0)  # integer exact match (§9.3 int32 row)
    return fp_table.get(dtype, (2**-10, 2**-6, 1e0, 0.99))


def check_precision(actual, golden, dtype):
    """Mixed-tolerance dual gate: (passed, matched_ratio, max_abs_error).

    Element-wise |actual - golden| <= atol + rtol * |golden|; overall pass
    requires matched_ratio >= required_matched_ratio AND max_abs_error <=
    max_abs_error_limit. Non-finite golden positions are skipped (§9.3).
    """
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


# ============================================================================
# L0 gate tests (DESIGN §9.2 — 4 cases: {main, small} x {ascendc, pto})
# ============================================================================
# (base, S, H, HKV, DQK, DV, topk, block_I); small first (fast), main second
# (input construction takes minutes + golden builds a ~2.1 GB score tensor).
L0_CONFIGS = [
    ("l0_small", 2050, 128, 1, 576, 512, 1024, 128),  # offsets [0,1024,2050]; segs 1024/1026
    ("l0_main", 4096, 128, 1, 576, 512, 1024, 128),  # offsets [0,2047,4096]; perf shape (§9.4)
]


def run_l0_pair(base, S, H, HKV, DQK, DV, topk, block_I):
    """One L0 config vs golden on both backends (inputs + golden shared)."""
    try:
        q, kv, indices, offsets = make_l0_inputs(S=S, H=H, HKV=HKV, DQK=DQK, topk=topk)
        ref_out = ref_sparse_mla_fwd_interface(q, kv, indices, offsets)
        ref_lse = ref_sparse_mla_fwd_lse(q, kv, indices, offsets)
        print(f"golden done for {base} (S={S}, offsets={offsets.tolist()})")
    except Exception as e:  # noqa: BLE001
        print(f"[PRECISION_FAIL] {base} golden: {e!r}")
        traceback.print_exc()
        return False
    ok = True
    for target in ("ascendc", "pto"):
        name = f"{base}_{target}"
        try:
            tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, offsets, d_v=DV, block_I=block_I, target=target)
            # gate 1: main-repo similarity criterion (L293; utils.py port, fp64 on CPU)
            tl_out_c, ref_out_c = tl_out.cpu(), ref_out.cpu()
            sim = assert_tensors_similar(tl_out_c, ref_out_c, eps=1e-2, name=f"{name}/out")
            # gate 2: bf16 mixed tolerance (§9.3)
            p_out, r_out, m_out = check_precision(tl_out, ref_out, "bfloat16")
            # gate 3: Lse fp32 mixed tolerance (§9.3, in-test golden)
            p_lse, r_lse, m_lse = check_precision(tl_lse, ref_lse, "float32")
            print(
                f"[PRECISION_{'PASS' if p_out else 'FAIL'}] {name} out dtype=bfloat16 "
                f"similarity={sim:.6f} matched_ratio={r_out:.4f} max_abs={m_out:.3e}"
            )
            print(f"[PRECISION_{'PASS' if p_lse else 'FAIL'}] {name} lse dtype=float32 matched_ratio={r_lse:.4f} max_abs={m_lse:.3e}")
            ok &= p_out and p_lse
        except Exception as e:  # noqa: BLE001
            print(f"[PRECISION_FAIL] {name}: {e!r}")
            traceback.print_exc()
            ok = False
    return ok


def test_sparse_mla_fwd_l0():
    """L0 gate: {main, small} x {ascendc, pto} (blocking)."""
    ok = True
    for base, S, H, HKV, DQK, DV, topk, block_I in L0_CONFIGS:
        ok &= run_l0_pair(base, S, H, HKV, DQK, DV, topk, block_I)
    return ok


# ============================================================================
# L1 functional tests (scenario B expansion; kernel-semantics golden; same
# dual gate as L0: similarity eps=1e-2 AND §9.3 mixed tolerance on
# out(bf16)/lse(fp32); blocking)
# ============================================================================
def run_semantic_case(name, q, kv, indices, offsets, d_v=512, block_I=128, blocking=True, sm_scale=None):
    """One config vs kernel_semantic_golden on both backends. blocking=True ->
    [PRECISION_PASS/FAIL] (counts toward exit code); False ->
    [BOUNDARY_PASS/WARN] (recorded only)."""
    good, bad = ("PRECISION_PASS", "PRECISION_FAIL") if blocking else ("BOUNDARY_PASS", "BOUNDARY_WARN")
    try:
        ref_out, ref_lse = kernel_semantic_golden(q, kv, indices, offsets, sm_scale=sm_scale)
    except Exception as e:  # noqa: BLE001
        print(f"[{bad}] {name} golden: {e!r}")
        traceback.print_exc()
        return False
    ok = True
    for target in ("ascendc", "pto"):
        tname = f"{name}_{target}"
        try:
            tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, offsets, sm_scale=sm_scale, d_v=d_v, block_I=block_I, target=target)
            sim = assert_tensors_similar(tl_out.cpu(), ref_out.cpu(), eps=1e-2, name=f"{tname}/out")
            p_out, r_out, m_out = check_precision(tl_out, ref_out, "bfloat16")
            p_lse, r_lse, m_lse = check_precision(tl_lse, ref_lse, "float32")
            print(
                f"[{good if p_out else bad}] {tname} out dtype=bfloat16 similarity={sim:.6f} matched_ratio={r_out:.4f} max_abs={m_out:.3e}"
            )
            print(f"[{good if p_lse else bad}] {tname} lse dtype=float32 matched_ratio={r_lse:.4f} max_abs={m_lse:.3e}")
            ok &= p_out and p_lse
        except Exception as e:  # noqa: BLE001
            print(f"[{bad}] {tname}: {e!r}")
            traceback.print_exc()
            ok = False
    return ok


# (name, S, H, HKV, topk, block_I, offsets_list, mutate[, sm_scale[, transform]])
# — grouped as 6 L1 cases (ni1+ni2 and empty+single are sub-configs of cases
# 5/6); the last 4 entries are the C2 coverage-gate completion cases.
L1_CONFIGS = [
    ("l1_repeat_indices", 128, 128, 1, 64, 64, [0, 64, 128], "repeat"),
    ("l1_neg_oob_indices", 200, 128, 1, 64, 64, [0, 100, 200], "neg_oob"),
    ("l1_short_segments", 1088, 128, 1, 1024, 128, [0, 543, 1088], None),
    ("l1_kvgroup2", 200, 128, 2, 128, 128, [0, 100, 200], None),
    ("l1_ni1_topk_eq_bi", 128, 128, 1, 128, 128, [0, 60, 128], None),
    ("l1_ni2_topk_2bi", 600, 128, 1, 256, 128, [0, 300, 600], None),
    ("l1_empty_segment", 128, 128, 1, 64, 64, [0, 0, 128], None),
    ("l1_single_batch", 128, 128, 1, 64, 64, [0, 128], None),
    # C2: S=257 = 2*128+1 (single segment, remainder-1 tail) and prime ->
    # D-SHAPE-TAIL-1 + D-SHAPE-PRIME
    ("l1_tail1_prime", 257, 128, 1, 128, 128, [0, 257], None, None, None),
    # C2: explicit non-default sm_scale + heads=64 (head_kv=64 in domain) ->
    # D-PARAM-sm_scale + D-PARAM-heads
    ("l1_smscale_heads64", 128, 64, 1, 64, 64, [0, 64, 128], None, 0.5, None),
    # C2: symmetric small value range (q, kv in ~[-1, 1]) -> D-VALRANGE-S
    ("l1_vrange_s", 128, 128, 1, 64, 64, [0, 64, 128], None, None, "vrange_s"),
    # C2: asymmetric value range (q in ~[-10, 20] via randn*3+5) -> D-VALRANGE-ASYM
    ("l1_vrange_asym", 128, 128, 1, 64, 64, [0, 64, 128], None, None, "vrange_asym"),
]


def _apply_l1_transform(q, kv, kind):
    """In-place value-range transforms (golden consumes the same tensors)."""
    if kind == "vrange_s":
        q.mul_(0.25)
        kv.mul_(0.25)
    elif kind == "vrange_asym":
        q.mul_(3.0).add_(5.0)
    return q, kv


def test_sparse_mla_fwd_l1():
    """L1 functional: 10 configs (6 grouped + 4 coverage completion) x both backends (blocking)."""
    ok = True
    for i, cfg in enumerate(L1_CONFIGS):
        name, S, H, HKV, topk, block_I, offs, mut = cfg[:8]
        sm_scale = cfg[8] if len(cfg) > 8 else None
        transform = cfg[9] if len(cfg) > 9 else None
        q, kv, indices, offsets = make_varlen_inputs(S=S, H=H, HKV=HKV, topk=topk, offsets_list=offs, seed=100 + i)
        if mut == "repeat":
            mutate_repeat(indices, offs, topk)
        elif mut == "neg_oob":
            mutate_neg_oob(indices, offs, S, topk)
        if transform is not None:
            q, kv = _apply_l1_transform(q, kv, transform)
        ok &= run_semantic_case(name, q, kv, indices, offsets, d_v=512, block_I=block_I, sm_scale=sm_scale)
    return ok


# ============================================================================
# L2 negative tests (host-side domain asserts + runtime dtype check; correct
# rejection = expected exception type; wrong type / silent accept = FAIL.
# Recorded, non-blocking per protocol.)
# ============================================================================
def _l2_inputs(S=32, H=128, topk=64, device="npu"):
    q = torch.randn((S, H, 576), dtype=torch.bfloat16, device=device)
    kv = torch.randn((S, 1, 576), dtype=torch.bfloat16, device=device)
    idx = torch.zeros((S, 1, topk), dtype=torch.int32, device=device)
    for t in range(S):
        i_i = torch.randperm(max(1, t))[:topk]
        idx[t, 0, : len(i_i)] = i_i
    off = torch.tensor([0, S], dtype=torch.int32, device=device)
    return q, kv, idx, off


def expect_raise(name, fn, exc_type):
    """PASS iff fn raises exc_type; wrong type / no raise -> [BOUNDARY_FAIL]."""
    try:
        fn()
    except exc_type as e:
        print(f"[BOUNDARY_PASS] {name}: raised {type(e).__name__}: {str(e)[:90]}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_FAIL] {name}: expected {exc_type.__name__}, got {type(e).__name__}: {str(e)[:90]}")
        return False
    print(f"[BOUNDARY_FAIL] {name}: no exception raised (silent accept)")
    return False


def test_sparse_mla_fwd_l2():
    """L2 negative: 4 host-side rejection cases (recorded, non-blocking)."""
    q, kv, idx, off = _l2_inputs()
    q16 = q.to(torch.float16)
    idx100 = torch.zeros((32, 1, 100), dtype=torch.int32, device="npu")
    ok = True
    ok &= expect_raise(  # fwd-only file (interface assert, verbatim main repo)
        "l2_return_p_sum",
        lambda: sparse_mla_fwd_interface(q, kv, idx, off, return_p_sum=True),
        AssertionError,
    )
    ok &= expect_raise(  # topk=100 % block_I=128 != 0 -> compile-time assert
        "l2_bad_topk",
        lambda: sparse_mla_fwd_interface(q, kv, idx100, off, block_I=128),
        AssertionError,
    )
    ok &= expect_raise(  # kernel buffers are bfloat16 -> runtime dtype mismatch
        "l2_bad_dtype",
        lambda: sparse_mla_fwd_interface(q16, kv, idx, off, block_I=64),
        ValueError,
    )
    ok &= expect_raise(  # d_v=256 -> tail_dim=320 outside (dim,tail)=(512,64)
        "l2_bad_dims",
        lambda: sparse_mla_fwd_interface(q, kv, idx, off, d_v=256, block_I=64),
        AssertionError,
    )
    return ok


# ============================================================================
# Boundary special-value tests (legal inputs; precision compared but
# non-blocking: fail -> [BOUNDARY_WARN], recorded). Known boundary NOT tested
# here: all-invalid rows (sumexp=0 -> NaN) — kernel behaves like the main repo
# kernel (debug_log A-block); exempted by design, see debug_log C1.
# ============================================================================
def _run_nonfinite_case(name, seed, mutate):
    """Special-value case whose golden contains non-finite positions.

    Comparison: §9.3 mixed tolerance ONLY (gate 2/3) — check_precision skips
    non-finite golden positions by design; the global similarity gate
    (normalized dot product) is mathematically undefined when either tensor
    holds nan/inf (denominator non-finite), so it is not applied here.
    """
    q, kv, indices, offsets = make_varlen_inputs(S=128, topk=64, offsets_list=[0, 64, 128], seed=seed)
    mutate(q, kv)
    try:
        ref_out, ref_lse = kernel_semantic_golden(q, kv, indices, offsets)
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_WARN] {name} golden: {e!r}")
        return False
    ok = True
    for target in ("ascendc", "pto"):
        tname = f"{name}_{target}"
        try:
            tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, offsets, d_v=512, block_I=64, target=target)
            p_out, r_out, m_out = check_precision(tl_out, ref_out, "bfloat16")
            p_lse, r_lse, m_lse = check_precision(tl_lse, ref_lse, "float32")
            fin = int(torch.isfinite(ref_out.float()).sum().item())
            tot = ref_out.numel()
            print(
                f"[BOUNDARY_{'PASS' if p_out and p_lse else 'WARN'}] {tname} "
                f"out matched_ratio={r_out:.4f} max_abs={m_out:.3e} lse matched_ratio={r_lse:.4f} "
                f"max_abs={m_lse:.3e} (golden finite {fin}/{tot}, non-finite skipped per §9.3)"
            )
            ok &= p_out and p_lse
        except Exception as e:  # noqa: BLE001
            print(f"[BOUNDARY_WARN] {tname}: {e!r}")
            traceback.print_exc()
            ok = False
    return ok


def test_sparse_mla_fwd_boundary():
    """Boundary: zero Q/KV, ±1e4 extremes, all-same index (non-blocking)."""
    ok = True
    # bnd_zero_qkv: all-zero Q/KV -> uniform softmax over the valid slots
    q, kv, indices, offsets = make_varlen_inputs(S=128, topk=64, offsets_list=[0, 64, 128], seed=201)
    q.zero_()
    kv.zero_()
    ok &= run_semantic_case("bnd_zero_qkv", q, kv, indices, offsets, block_I=64, blocking=False)
    # bnd_extreme_values: ±1e4-magnitude rows (bf16 safe domain; exp overflow
    # protection carried by the m-shift + BIGC mask chain)
    q, kv, indices, offsets = make_varlen_inputs(S=128, topk=64, offsets_list=[0, 64, 128], seed=202)
    q[::5] = q[::5] * 1e4
    kv[::3] = kv[::3] * 1e4
    ok &= run_semantic_case("bnd_extreme_values", q, kv, indices, offsets, block_I=64, blocking=False)
    # bnd_all_same_index: rows whose indices are all one repeated valid value
    q, kv, indices, offsets = make_varlen_inputs(S=128, topk=64, offsets_list=[0, 64, 128], seed=203)
    mutate_same_index(indices, [0, 64, 128])
    ok &= run_semantic_case("bnd_all_same_index", q, kv, indices, offsets, block_I=64, blocking=False)
    # C2 special-value probes (D-SPECIAL-INF/NAN/DBOUND): non-finite golden
    # positions are skipped by the §9.3 mixed-tolerance gate; the global
    # similarity metric (normalized dot product) is undefined on nan/inf, so
    # these cases bypass gate 1 by design and rely on gate 2/3 only.
    ok &= _run_nonfinite_case(
        "bnd_inf_q",
        seed=204,
        mutate=lambda q, kv: (
            q.__setitem__((0, 0, slice(None)), float("inf")),
            q.__setitem__((1, 1, slice(None)), float("-inf")),
        ),
    )
    ok &= _run_nonfinite_case(
        "bnd_nan_kv",
        seed=205,
        mutate=lambda q, kv: (
            kv.__setitem__((3, 0, slice(None)), float("nan")),
            kv.__setitem__((5, 0, slice(7, 17)), float("nan")),
        ),
    )
    ok &= _run_nonfinite_case(
        "bnd_dbound",
        seed=206,
        mutate=lambda q, kv: (
            q.__setitem__((2, 0, slice(None)), torch.finfo(torch.bfloat16).max),
            kv.__setitem__((6, 0, slice(None)), -torch.finfo(torch.bfloat16).max),
        ),
    )
    return ok


# ---- Coverage annotations (machine-checkable; coverage-matrix.md + proto_sparse_mla_fwd.yaml) ----
COVERAGE_CATEGORY = "Fusion"  # CV mixed: gemm_v0 QK/PV (Cube) + mask/online-softmax (Vector) + gather (MTE)

# Counts are per CONFIG (each config runs both backends ascendc/pto).
# Value/param provenance:
#   D-VALRANGE-M  — the default randn(std=1, ~±4) inputs of L0/L1/boundary configs;
#   D-PARAM-{heads,dim,tail_dim,topk,kv_group} — shape-derived param paths exercised
#       across distinct values (H 128/64, topk 64/128/256/1024, HKV 1/2, d_v 512
#       + 256-reject); the interface derives them from tensor shapes by design;
#   D-PARAM-sm_scale — explicit non-default 0.5 in l1_smscale_heads64 (default
#       (dim+tail)^-0.5 path is exercised by every other config).
COVERAGE_MANIFEST = {
    "D-DTYPE-bf16": 22,
    "D-DTYPE-int32": 22,
    "D-DTYPE-fp32": 18,
    "D-SHAPE-ALIGNED": 4,
    "D-SHAPE-TAIL-1": 2,
    "D-SHAPE-TAIL-MID": 3,
    "D-SHAPE-PRIME": 1,
    "D-SHAPE-EDGE": 2,
    "D-VALRANGE-S": 1,
    "D-VALRANGE-M": 13,
    "D-VALRANGE-L": 1,
    "D-VALRANGE-ASYM": 1,
    "D-PARAM-heads": 2,
    "D-PARAM-dim": 2,
    "D-PARAM-tail_dim": 2,
    "D-PARAM-topk": 4,
    "D-PARAM-kv_group": 2,
    "D-PARAM-sm_scale": 1,
    "D-PARAM-block_I": 2,
    "D-PARAM-target": 2,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-INF": 1,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-DBOUND": 1,
    "D-EXC-DTYPE": 1,
    "D-EXC-SHAPE": 2,
}

# Fusion category has NO exempt dimensions (coverage-matrix.md §二), so every
# applicable dimension is covered above; nothing to exempt. Note (proto): this
# operator has a FIXED dtype combination (bf16 q/kv/out, int32
# indices/offsets, fp32 lse) — there is no dtype-sweep dimension to declare.
COVERAGE_NA = {}


# ============================================================================
# Perf sentinel (DESIGN §9.4): l0_main shape, both backends, dual timing —
# pure kernel (NPU events, excludes host) + interface loop (tensor_cache on)
# ============================================================================
PERF_TARGET_US = 24288.0  # DESIGN §7/§9.4: latency < 24288 us


def _time_loop(fn, n_iters, warmup):
    """Per-iteration NPU event timing (first launch absorbed by warmup).

    Returns (mean_us, std_us, min_us, max_us) over the n_iters timed calls.
    """
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(n_iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(n_iters)]
    for i in range(n_iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.npu.synchronize()
    times = [starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(n_iters)]  # ms -> us
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return mean, std, min(times), max(times)


def _perf_pure_kernel(target, q, kv, indices, offsets, n_iters=20, warmup=5):
    """Pure kernel timing: pre-built func + pre-allocated args (indexer
    _bench_kernel form). Excludes H2D, output/workspace allocation, host
    dispatch detection and prepare_token_indices — each iteration launches
    only the kernel."""
    S, H, DDT = q.shape
    grid = get_cube_core_num()
    func = tl_sparse_mla_fwd(
        heads=H,
        dim=512,
        tail_dim=64,
        topk=1024,
        kv_group=1,
        sm_scale=None,
        block_I=128,
        grid_size=grid,
        target=target,
    )
    _, _, _, _, Hseg, _ = _derived_head_layout(H, 1)
    token_indices = prepare_token_indices(offsets)
    out = torch.zeros((S, H, 512), dtype=q.dtype, device=q.device)
    lse = torch.zeros((S, H), dtype=torch.float32, device=q.device)
    ws_kv = torch.zeros((grid, 128, DDT), dtype=torch.bfloat16, device=q.device)
    ws_s = torch.zeros((grid, Hseg, 128), dtype=torch.float32, device=q.device)
    ws_p = torch.zeros((grid, Hseg, 128), dtype=torch.bfloat16, device=q.device)
    ws_o = torch.zeros((grid, Hseg, 512), dtype=torch.float32, device=q.device)

    def launch():
        func(q, kv, indices, offsets, token_indices, out, lse, ws_kv, ws_s, ws_p, ws_o)

    return _time_loop(launch, n_iters, warmup)


def _perf_interface(target, q, kv, indices, offsets, n_iters=50, warmup=5):
    """Interface-level loop (do_bench-equivalent steady state): dispatch
    asserts + prepare_token_indices (served from _tensor_cache after the
    first call) + output/workspace allocation + kernel launch."""

    def call():
        sparse_mla_fwd_interface(q, kv, indices, offsets, d_v=512, block_I=128, target=target)

    return _time_loop(call, n_iters, warmup)


def benchmark_perf():
    """DESIGN §9.4 sentinel: B=1,S=4096,H=128,HKV=1,DQK=576,DV=512,topk=1024,
    bf16 (l0_main inputs reused); reports pure-kernel + interface numbers for
    both backends vs the 24288 us budget (recorded; tuning is Stage 3)."""
    print("[PERF] building l0_main inputs (B=1, S=4096, H=128, HKV=1, DQK=576, DV=512, topk=1024, bf16)...")
    q, kv, indices, offsets = make_l0_inputs(S=4096, H=128, HKV=1, DQK=576, topk=1024)
    results = {}
    for target in ("ascendc", "pto"):
        k_mean, k_std, k_min, k_max = _perf_pure_kernel(target, q, kv, indices, offsets)
        print(
            f"[PERF] sparse_mla_fwd {target} pure-kernel: mean {k_mean:.0f} us "
            f"(std {k_std:.0f}, min {k_min:.0f}, max {k_max:.0f}; 20 iters after 5 warmup, "
            f"torch.npu events; excludes host alloc/dispatch/H2D)"
        )
        i_mean, i_std, i_min, i_max = _perf_interface(target, q, kv, indices, offsets)
        print(
            f"[PERF] sparse_mla_fwd {target} interface: mean {i_mean:.0f} us "
            f"(std {i_std:.0f}, min {i_min:.0f}, max {i_max:.0f}; 50 iters after 5 warmup, "
            f"tensor_cache active)"
        )
        verdict = "PASS" if k_mean < PERF_TARGET_US else "OVER"
        print(
            f"[PERF] sparse_mla_fwd {target} sentinel vs {PERF_TARGET_US:.0f} us budget: "
            f"{verdict} ({k_mean / PERF_TARGET_US * 100:.1f}% of budget, pure-kernel口径)"
        )
        results[target] = (k_mean, k_std, i_mean, i_std)
    return results


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

    if level == "perf":
        benchmark_perf()  # §9.4 sentinel: recorded (both backends, dual口径); Stage 3 tunes
        sys.exit(0)

    blocking_ok = True  # only L0/L1 count toward the exit code
    if level in ("l0", "all"):
        blocking_ok &= test_sparse_mla_fwd_l0()
    if level in ("l1", "all"):
        blocking_ok &= test_sparse_mla_fwd_l1()
    if level in ("l2", "all"):
        test_sparse_mla_fwd_l2()  # negative tests: recorded, non-blocking
    if level in ("boundary", "all"):
        test_sparse_mla_fwd_boundary()  # special values: recorded, non-blocking

    if blocking_ok:
        print("Test Passed!")
    sys.exit(0 if blocking_ok else 1)


if __name__ == "__main__":
    main()
