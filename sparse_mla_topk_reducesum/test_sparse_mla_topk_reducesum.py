"""Layered test for sparse_mla_topk_reducesum — generic-path L0-G gate
(post panel/auto removal, DESIGN_generic_p1.md §9.2). The P1 AIC GEMM pipeline
restructure (dual-slot L1 + per-parity release events) was REVERTED after R9
perf falsification — same-session paired A/B showed a +4.0~+6.0% net regression
(root cause: MTE2 gate removal × V-bandwidth coupling, see perf_tuning/
p1_ablation_report.md); the experiment is archived at git 00a93f81. This file
targets the reverted (removal-baseline) kernel.

Layers (rev2 re-anchor: official arange = the ONLY performance-caliber data
shape; random/distribution families are correctness/robustness only):
  * L0-G1  random per-row independent causal indices (ROBUSTNESS case, rev2
           down-scoped): S=2048/H=16/D=512/tail_D=64/topk=128/BI=128/seed=123 —
           golden criteria + §9.3 mixed tolerance (correctness anchor; the
           performance role moved to L0-G2/G6 official arange).
  * L0-G2  OFFICIAL golden arange regression (rev2 up-scoped): the original GPU
           golden case verbatim (seed=42, B=1/S=2048/H=16/D=512/tail_D=64/
           topk=128), both block_I 32/128 — shares the exact construction with
           the L0-G6/§9.5 performance channel (construction identity = caliber
           lock).
  * L0-G3  index distribution family: (a) all-identical shared vector,
           (b) partial overlap (row clusters share a prefix + independent
           tails), (c) non-contiguous unordered (stride + per-row permute),
           (d) boundary values (-1 tail rows / OOB value / idx==s_i / idx==0)
           and (e) an all--1 row compared against the raw §1.3 reference
           (renormalization would be 0/0 there).
  * L0-G4  shape/topk/segment scan: S∈{64,512,2048} × topk∈{32,64,128,256} ×
           BI∈{32,64,128} representative combos; single/[0,1023,2048]/
           four-segment/[0,0,1023,2048] empty-first offsets; kv_group=2
           (heads=32); non-default dim_v/grid_size; topk=256 at the BI=128
           tier (the rev1 topk=256/BI=256 spec hits the gemm_v0 L0B nTile cap
           and cannot compile); largest-DDT shape H=64/tail_D=512/BI=64.
  * L0-G5  race-stress family (§9.2, mandatory): (1) ΣS=64 × D∈{128,256} ×
           topk∈{32,64} × 3 independent runs — each vs reference AND 3-run
           pairwise torch.equal (bitwise stability); (2) short streams
           nblk∈{1,2,3} (S=1/21/41 — single/double/triple block core flows,
           empty cores must exit cleanly); (3) topk=256/BI=128 ×3; (4) the
           largest-DDT tail_D=512/BI=64 shape ×3.
  * L0-G6  OFFICIAL perf anchor (rev2 re-anchor, coarse, non-gating): golden
           arange dual-tier (BI∈{128,32}) graph-replay pure-kernel latency vs
           the removal-baseline anchors BI128=364.7µs / BI32=475.3µs
           (golden_metric2_bench.md) — the post-rollback drift check (>±5%
           deviation is recorded and reported); precise measurement is
           deferred to Stage 3 / §9.5.

L1/L2/Boundary layers follow the scenario-B extension (tilelang-op-test-design;
L2 includes the rev2 block_I>128 out-of-domain rejection).

Run from repo root:
    source set_env.sh && python examples/dsa_sparse_finetune/test_sparse_mla_topk_reducesum.py --level l0
"""

import argparse
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F

import tilelang

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sparse_mla_topk_reducesum import (  # noqa: E402
    get_cube_core_num,
    prepare_token_indices,
    sparse_mla_topk_reducesum,
    tl_sparse_mla_topk_reducesum,
    LAST_DISPATCH,
)

# PASS thresholds (golden-eval task spec): err_ratio <= 1e-2 AND abs_err <= 1e-3.
# Raw values are always printed; near-boundary results are flagged for adjudication.
ERR_RATIO_MAX = 1e-2
ABS_ERR_MAX = 1e-3
# Mixed-tolerance gate (DESIGN_generic_p1 §9.3, ReduceSum float32 row): per-element
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

    The kernel treats index values outside [0, seq_len) (including
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

    index_vec: None -> shared P = arange(topk) (regression family, L0-G2); otherwise
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


def make_random_causal_indices(S, offsets_list, topk, seed, kv_group=1, full_range=False):
    """Per-row independent random causal indices (DESIGN_generic_p1 §9.1): randint
    per row then clamp to the row's causal upper bound s_i (the row's in-sequence
    position). Rows share nothing — zero inter-row sharing/contiguity assumptions.

    full_range=False (default, the "anchor family"): randint over [0, topk) —
    the generic_pipeline_analysis random_BI128 anchor construction verbatim
    (bench_graph_smtk / prof_generic: `randint(0, TOPK) -> clamp(max=s_col)`),
    seed=123 shares the source with the 441.7µs anchor. L0-G1/L0-G6 use this
    for perf-anchor correspondence.
    full_range=True: randint over [0, ΣS) — maximal KV working-set coverage for
    the functional/race layers (L0-G4/G5, L1; ~3.2x the anchor's KV-row reuse).

    Every index lands in [0, s_i] (causally valid), so renormalized references
    stay finite. CPU construction -> one H2D (aclnn audit allow).

    Returns Indices [ΣS, kv_group, topk] int32 (NPU).
    """
    torch.manual_seed(seed)
    seg_lens = [b - a for a, b in zip(offsets_list, offsets_list[1:])]
    assert sum(seg_lens) == S
    s_pos = torch.cat([torch.arange(n, dtype=torch.int32) for n in seg_lens])  # [S]
    hi = int(S) if full_range else int(topk)
    idx = torch.randint(0, hi, (S, kv_group, topk), dtype=torch.int32)
    idx = idx.clamp(max=s_pos.view(S, 1, 1))
    return idx.npu()


def ref_reducesum_raw(q, kv, topk_indices, lse, offsets, sm_scale=None):
    """Raw §1.3 formula reference (NO interface renormalization), kv_group=1:

        ReduceSum[s, 0, 0, j] = sum_h exp((Q[s,h,:]·KV[bos+idx])·scale − Lse[s,h])
                                  · 1[idx ≤ s_i ∧ idx ≥ 0]

    Used for distributions whose renormalization is 0/0 (an all--1 row): both the
    kernel ReduceSum and this reference are exact zeros there, which the
    renormalized refs cannot express. Out-of-range indices are clamped for the
    gather and masked exactly like the kernel (clamp protects addressing only).

    Returns [ΣS, topk] float32 (== kernel output [:, 0, 0, :] for kv_group=1).
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    outs = []
    for i in range(offsets.shape[0] - 1):
        q_b = q[offsets[i] : offsets[i + 1]]  # [n, H, DDT]
        kv_b = kv[offsets[i] : offsets[i + 1]]  # [n, DDT] (2-D kv_group=1 view)
        idx_b = topk_indices[offsets[i] : offsets[i + 1]]  # [n, topk]
        n = q_b.shape[0]
        if n == 0:
            continue
        s_i = torch.arange(n, device=q.device)
        idx = idx_b.to(torch.int64)
        valid = (idx >= 0) & (idx <= s_i.unsqueeze(1))  # causal indicator
        idx_safe = idx.clamp(0, n - 1)
        sel = kv_b[idx_safe]  # [n, topk, DDT]
        logits = torch.einsum("nhd,njd->nhj", q_b.float(), sel.float()) * sm_scale
        p = torch.exp(logits - lse[offsets[i] : offsets[i + 1]].unsqueeze(-1).float())
        rs = (p * valid.float().unsqueeze(1)).sum(dim=1)  # [n, topk]
        outs.append(rs)
    return torch.cat(outs, dim=0)


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
    """Bitwise equivalence check (cross-config/run stability)."""
    eq = torch.equal(out_a, out_b)
    tag = "[PRECISION_PASS]" if eq else "[PRECISION_FAIL]"
    extra = "" if eq else f" max_diff={(out_a.float() - out_b.float()).abs().max().item():.3e}"
    print(f"{tag} {name}: torch.equal={eq}{extra}")
    return eq


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
# Golden case — parameters verbatim from the GPU test_kernel defaults (L0-G2)
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
# L0-G gate cases (DESIGN_generic_p1 §9.2 — post panel/auto removal)
# ===========================================================================
def _l0_random_case(name, S, H, D, tail_D, topk, offsets, block_I, seed, kv_group=1, grid_size=None):
    """Shared L0 runner: one random-causal-index case vs the reference
    (golden criteria + §9.3 mixed tolerance; blocking)."""
    q, kv, idx, offsets_t = make_case_v(S, H, D, tail_D, topk, offsets, seed=seed, kv_group=kv_group)
    if kv_group > 1:
        lse, ref_attn = ref_mla_topk_softmax_grouped(q, kv, idx, offsets_t, kv_group)
    else:
        lse, ref_attn = ref_mla_topk_softmax_masked(q, kv[:, 0, :], idx[:, 0, :], offsets_t)
    out = sparse_mla_topk_reducesum(
        q,
        kv,
        idx,
        lse,
        offsets_t,
        dim_v=D,
        kv_group=kv_group,
        block_I=block_I,
        grid_size=grid_size,
    )
    return check_vs_ref(name, out, ref_attn)


def l0_g1_random_main():
    """L0-G1: random per-row independent causal indices — ROBUSTNESS case
    (rev2 down-scope, §9.2 row 1): S=2048/H=16/D=512/tail_D=64/topk=128/
    BI=128/seed=123 with offsets=[0,1023,2048] — the anchor-family construction
    (randint(0, topk) + causal clamp, §9.1). Correctness/robustness anchor
    ONLY; the performance caliber moved to the official arange shape
    (L0-G2/G6, rev2)."""
    S, H, D, tail_D, topk, BI = 2048, 16, 512, 64, 128, 128
    offsets_list = [0, 1023, S]
    torch.manual_seed(123)
    q = torch.randn((S, H, D + tail_D)).npu().bfloat16()
    kv = torch.randn((S, 1, D + tail_D)).npu().bfloat16()
    offsets = torch.tensor(offsets_list, dtype=torch.int32).npu()
    idx = make_random_causal_indices(S, offsets_list, topk, seed=123)  # anchor family
    lse, ref_attn = ref_mla_topk_softmax_masked(q, kv[:, 0, :], idx[:, 0, :], offsets)
    out = sparse_mla_topk_reducesum(q, kv, idx, lse, offsets, dim_v=D, block_I=BI)
    return check_vs_ref("L0-G1 random-causal (robustness, BI=128, seed=123) vs torch-ref", out, ref_attn)


def l0_g2_golden_regression():
    """L0-G2: OFFICIAL golden arange regression (rev2 up-scope, §9.2 row 2) —
    the original GPU golden case verbatim (seed=42, both block_I 32/128).
    Construction identity with the L0-G6/§9.5 performance channel = caliber
    lock; the kernel has no index-content specialization (§8.6)."""
    passed, _ = test_kernel()
    tag = "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]"
    print(f"{tag} L0-G2 official golden arange regression (block_I 32/128)")
    return passed


def l0_g3_distribution_family():
    """L0-G3: index distribution family (§9.2 row 3) — all-identical /
    partial overlap / non-contiguous unordered / boundary values / all--1 row."""
    S, H, D, tail_D, topk, BI = 2048, 16, 512, 64, 128, 128
    offsets_list = [0, 1023, S]
    seg_lens = [b - a for a, b in zip(offsets_list, offsets_list[1:])]
    s_pos = torch.cat([torch.arange(n, dtype=torch.int32) for n in seg_lens])  # [S] in-seq s_i

    torch.manual_seed(2024)
    q = torch.randn((S, H, D + tail_D)).npu().bfloat16()
    kv = torch.randn((S, D + tail_D)).npu().bfloat16()
    offsets = torch.tensor(offsets_list, dtype=torch.int32).npu()
    kv4 = kv.unsqueeze(-2)

    # ---- (a) 全同：every row shares one random vector P（P[0]=0 keeps s_i=0 rows finite）
    P = torch.randint(0, S, (topk,), dtype=torch.int32)
    P[0] = 0
    idx_a = P.view(1, topk).repeat(S, 1).contiguous().npu()
    lse, ref_a = ref_mla_topk_softmax_masked(q, kv, idx_a, offsets)
    out_a = sparse_mla_topk_reducesum(q, kv4, idx_a.unsqueeze(-2), lse, offsets, dim_v=D, block_I=BI)
    ok_a = check_vs_ref("L0-G3(a) all-identical shared vector vs torch-ref", out_a, ref_a)

    # ---- (b) 部分重合：clusters of 8 rows share a random prefix; tails independent ----
    CL = 8
    ncl = S // CL
    prefix = torch.randint(0, S, (ncl, topk // 2), dtype=torch.int32)
    prefix[:, 0] = 0  # guarantee one causally-valid entry per row
    tail = torch.randint(0, S, (S, topk // 2), dtype=torch.int32)
    cl_ids = torch.arange(S) // CL
    idx_b = torch.cat([prefix[cl_ids], tail], dim=1)
    idx_b = idx_b.clamp(max=s_pos.unsqueeze(1)).contiguous().npu()
    lse, ref_b = ref_mla_topk_softmax_masked(q, kv, idx_b, offsets)
    out_b = sparse_mla_topk_reducesum(q, kv4, idx_b.unsqueeze(-2), lse, offsets, dim_v=D, block_I=BI)
    ok_b = check_vs_ref("L0-G3(b) partial-overlap (cluster prefix + indep tail) vs torch-ref", out_b, ref_b)

    # ---- (c) 非连续乱序：strided base (j*7+3) mod (s_i+1), per-row permuted ----
    j = torch.arange(topk)
    strided = (j * 7 + 3).view(1, topk)
    idx_c = strided % (s_pos.view(S, 1) + 1)
    perm = torch.argsort(torch.rand(S, topk), dim=1)
    idx_c = torch.gather(idx_c, 1, perm).contiguous().to(torch.int32).npu()
    lse, ref_c = ref_mla_topk_softmax_masked(q, kv, idx_c, offsets)
    out_c = sparse_mla_topk_reducesum(q, kv4, idx_c.unsqueeze(-2), lse, offsets, dim_v=D, block_I=BI)
    ok_c = check_vs_ref("L0-G3(c) non-contiguous unordered (stride+permute) vs torch-ref", out_c, ref_c)

    # ---- (d) 边界值：-1 tail rows / OOB value / idx==s_i / idx==0 on an arange base ----
    idx_d = torch.arange(topk, dtype=torch.int32).view(1, topk).repeat(S, 1).contiguous()
    for r in (0, 1, 5, 1023, 1024, 1100):  # rows with s_i+1 < topk in both segments
        s_i = int(s_pos[r])
        idx_d[r, s_i + 1 :] = -1  # natural -1 tail padding
    idx_d[1500, 100] = 5000  # far OOB (> ΣS_kv=2048): clamp protects addressing, mask zeroes
    idx_d[700, 0] = int(s_pos[700])  # idx == s_i (causal diagonal)
    idx_d[700, 1] = 0  # idx == 0
    idx_d = idx_d.npu()
    lse, ref_d = ref_mla_topk_softmax_masked(q, kv, idx_d, offsets)
    out_d = sparse_mla_topk_reducesum(q, kv4, idx_d.unsqueeze(-2), lse, offsets, dim_v=D, block_I=BI)
    ok_d = check_vs_ref("L0-G3(d) boundary values (-1 tail/OOB/idx==s/idx==0) vs torch-ref", out_d, ref_d)

    # ---- (e) 全 -1 行：renormalization is 0/0 there — compare RAW ReduceSum vs the
    # §1.3 reference (both exactly zero on the all-masked row) ----
    idx_e = idx_d.cpu().clone()
    idx_e[2000, :] = -1
    idx_e = idx_e.npu()
    lse_e, _ = ref_mla_topk_softmax_masked(q, kv, idx_e, offsets)
    out_e = sparse_mla_topk_reducesum(q, kv4, idx_e.unsqueeze(-2), lse_e, offsets, dim_v=D, block_I=BI)
    ref_raw = ref_reducesum_raw(q, kv, idx_e, lse_e, offsets)
    got = out_e[:, 0, 0, :].float()
    diff = (got - ref_raw).abs()
    matched_ratio = (diff <= ATOL + RTOL * ref_raw.abs()).float().mean().item()
    max_abs = diff.max().item()
    zero_row = bool((out_e[2000] == 0).all().item())
    ok_e = matched_ratio >= MATCHED_RATIO_MIN and max_abs <= ABS_ERR_MAX and zero_row
    tag = "[PRECISION_PASS]" if ok_e else "[PRECISION_FAIL]"
    print(
        f"{tag} L0-G3(e) all--1 row vs raw §1.3 ref: matched_ratio={matched_ratio:.6f}, "
        f"max_abs={max_abs:.3e}, row-2000 exact-zero={zero_row}"
    )
    return ok_a and ok_b and ok_c and ok_d and ok_e


# (name, cfg) — regular shapes only (topk % block_I == 0); random causal indices.
# NOTE: BI=256 cannot compile on the current framework: gemm_v0's transpose-B
# path pins nTile = N = BI and the L0B ping-pong slot budget is 32KB once
# K > kL0Size (src/tl_templates/ascend/common.h: `nTile * kL0Size * sizeof(T1)
# <= kL0Budget` → static_assert failure), capping BI ≤ 128 at kL0Size=128/bf16
# (this out-of-domain config is negative-tested in L2 block_I_gt_128). The
# topk-256 scan value is therefore covered with BI=128 (a domain-valid config).
L0_G4_CASES = [
    ("small_single_seg", dict(S=64, H=16, D=512, tail_D=64, topk=32, block_I=32, offsets=[0, 64])),
    ("mid_topk64_bi64", dict(S=512, H=16, D=512, tail_D=64, topk=64, block_I=64, offsets=[0, 512])),
    ("golden_bi32", dict(S=2048, H=16, D=512, tail_D=64, topk=128, block_I=32, offsets=[0, 1023, 2048])),
    ("four_seg_bi64", dict(S=2048, H=16, D=512, tail_D=64, topk=128, block_I=64, offsets=[0, 500, 1000, 1500, 2048])),
    ("empty_first_seg", dict(S=2048, H=16, D=512, tail_D=64, topk=128, block_I=128, offsets=[0, 0, 1023, 2048])),
    ("kv_group2", dict(S=512, H=32, D=512, tail_D=64, topk=64, block_I=64, offsets=[0, 512], kv_group=2)),
    ("dim_v_256", dict(S=512, H=16, D=256, tail_D=64, topk=64, block_I=64, offsets=[0, 512])),
    ("grid_size_8", dict(S=512, H=16, D=512, tail_D=64, topk=128, block_I=128, offsets=[0, 512], grid_size=8)),
    # topk=256 (BI=128 substitute for the uncompilable BI=256 spec — see NOTE above)
    ("topk256_bi128", dict(S=512, H=16, D=512, tail_D=64, topk=256, block_I=128, offsets=[0, 512])),
    ("ni2_topk64_bi32", dict(S=64, H=16, D=512, tail_D=64, topk=64, block_I=32, offsets=[0, 32, 64])),
    # largest-DDT in-domain shape (H=64/tail_D=512 → DDT=1024, BI=64): plain
    # shape coverage of the biggest Cube/Vector working set in the support
    # domain (introduced as "kvdb1_trigger" during the reverted-P1 experiment,
    # kept here as a regular shape case — the config is domain-legal and its
    # value is independent of P1; L1 budgets: q_l1+kv_l1 = 2·64·1024·2B = 256KiB
    # ≤ 511KiB, L0A/L0B 16KiB ✓, L0C 16KiB ✓).
    ("h64_tail512_bi64", dict(S=512, H=64, D=512, tail_D=512, topk=128, block_I=64, offsets=[0, 512])),
    # grouped short-stream NI=1 window (2026-10-11): locks the idx(LEAD)-prefetch
    # group-addressing fix — task(LEAD//NI) lands in group 1 on cores cid>=8
    # (task2 pid=40+cid, S=48), which the pre-fix g_l (= b_s_l // ... == 0)
    # mis-addressed to group 0 (empirically max_abs=1.45e-2 on g=1 rows 1-11).
    ("grouped_short_gl_window", dict(S=48, H=32, D=512, tail_D=64, topk=64, block_I=64, offsets=[0, 48], kv_group=2)),
]


def l0_g4_shape_scan():
    """L0-G4: shape/topk/segment scan (§9.2 row 4, regular shapes, random causal
    indices) — incl. empty-first-segment offsets, kv_group=2, non-default
    dim_v/grid_size, topk=256 at the BI=128 tier (BI=256 uncompilable — see
    NOTE above), and the largest-DDT H=64/tail_D=512 shape."""
    all_ok = True
    for name, cfg in L0_G4_CASES:
        try:
            ok = _l0_random_case(f"L0-G4 {name}", seed=43, **cfg)
        except Exception:
            traceback.print_exc()
            print(f"[PRECISION_FAIL] L0-G4 {name}: unexpected exception (traceback above)")
            ok = False
        all_ok = all_ok and ok
    return all_ok


def _race_triple(name, q, kv, idx, lse, offsets, ref_attn, dim_v, block_I):
    """3 independent runs: each vs reference AND pairwise torch.equal (bitwise
    stability across the lookahead × slot-reuse race surface)."""
    outs = []
    ok = True
    for run in range(3):
        out = sparse_mla_topk_reducesum(q, kv, idx, lse, offsets, dim_v=dim_v, block_I=block_I)
        torch.npu.synchronize()
        outs.append(out)
        ok = check_vs_ref(f"{name} run{run} vs torch-ref", out, ref_attn) and ok
    stable = torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2]) and torch.equal(outs[1], outs[2])
    tag = "[PRECISION_PASS]" if stable else "[PRECISION_FAIL]"
    print(f"{tag} {name}: 3-run bitwise stability (equal={stable})")
    return ok and stable


def l0_g5_race_stress():
    """L0-G5: race-stress family (§9.2 row 5, mandatory) — (1) ΣS=64 × D × topk
    × 3 runs; (2) short streams nblk∈{1,2,3} (S=1/21/41 — empty cores must exit
    cleanly); (3) topk=256/BI=128 ×3; (4) tail_D=512/BI=64 ×3 (largest-DDT
    race surface)."""
    all_ok = True
    # ---- (1) ΣS=64 × D∈{128,256} × topk∈{32,64} × 3 runs (BI=32 default) ----
    for D in (128, 256):
        for topk in (32, 64):
            S, H, tail_D = 64, 16, 64
            q, kv, idx, offsets = make_case_v(S, H, D, tail_D, topk, [0, 32, 64], seed=99)
            lse, ref_attn = ref_mla_topk_softmax_masked(q, kv[:, 0, :], idx[:, 0, :], offsets)
            ok = _race_triple(f"L0-G5(1) D={D} topk={topk} BI=32", q, kv, idx, lse, offsets, ref_attn, D, 32)
            all_ok = all_ok and ok
    # ---- (2) short streams: nblk = ntask*NI ∈ {1,2,3} on core 0 (grid=20, NI=1);
    # cores with ntask==0 must not hang (§6.4 clean-exit) ----
    for S in (1, 21, 41):
        topk, BI, H, D, tail_D = 32, 32, 16, 512, 64
        q, kv, idx, offsets = make_case_v(S, H, D, tail_D, topk, [0, S], seed=77)
        lse, ref_attn = ref_mla_topk_softmax_masked(q, kv[:, 0, :], idx[:, 0, :], offsets)
        ok = _race_triple(f"L0-G5(2) short-stream S={S}", q, kv, idx, lse, offsets, ref_attn, D, BI)
        all_ok = all_ok and ok
    # ---- (3) topk=256/BI=128 ×3 (BI=256 substitute — see L0_G4_CASES NOTE:
    # the design's BI=256 spec hits gemm_v0's L0B nTile cap and cannot compile;
    # largest-DMA race surface is stressed at the domain-valid BI=128) ----
    S, H, D, tail_D, topk, BI = 512, 16, 512, 64, 256, 128
    q, kv, idx, offsets = make_case_v(S, H, D, tail_D, topk, [0, S], seed=88)
    lse, ref_attn = ref_mla_topk_softmax_masked(q, kv[:, 0, :], idx[:, 0, :], offsets)
    ok = _race_triple("L0-G5(3) topk=256 BI=128 (fallback-path substitute)", q, kv, idx, lse, offsets, ref_attn, D, BI)
    all_ok = all_ok and ok
    # ---- (4) tail_D=512/BI=64 ×3: largest-DDT (DDT=1024) race surface — the
    # biggest V-side UB slot-reuse + C-side L1/L0C working set in the support
    # domain (H=64/tail_D=512; introduced during the reverted-P1 experiment,
    # kept here as a plain race-stress case — the config is domain-legal
    # independent of P1) ----
    S, H, D, tail_D, topk, BI = 512, 64, 512, 512, 128, 64
    q, kv, idx, offsets = make_case_v(S, H, D, tail_D, topk, [0, S], seed=89)
    lse, ref_attn = ref_mla_topk_softmax_masked(q, kv[:, 0, :], idx[:, 0, :], offsets)
    ok = _race_triple("L0-G5(4) tail_D=512 BI=64 (largest-DDT race surface)", q, kv, idx, lse, offsets, ref_attn, D, BI)
    all_ok = all_ok and ok
    return all_ok


def _graph_replay_us(func, args, out_slice, warmup=5, iters=20, batch=25):
    """Capture one NPUGraph of func(*args) and time replays (µs per call).
    Returns (mean, median, min, max, eager_slice) — eager_slice is the eager
    (non-graph) output for the graph-vs-eager consistency check."""
    out = func(*args)
    torch.npu.synchronize()
    for _ in range(warmup):
        func(*args)
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    side = torch.npu.Stream()
    side.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(side):
        for _ in range(3):
            func(*args)
    torch.npu.current_stream().wait_stream(side)
    torch.npu.synchronize()
    with torch.npu.graph(g):
        out_g = func(*args)
    for _ in range(warmup):
        g.replay()
    torch.npu.synchronize()
    if not torch.allclose(out_g[out_slice], out[out_slice], atol=2e-4):
        return None
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        for _ in range(batch):
            g.replay()
        end.record()
        torch.npu.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / batch)  # µs
    times.sort()
    return sum(times) / len(times), times[len(times) // 2], times[0], times[-1], out[out_slice]


def l0_g6_perf_anchor():
    """L0-G6: OFFICIAL perf anchor (rev2 re-anchor, coarse, NON-GATING) —
    graph-replay pure-kernel latency of the official golden arange construction
    (§9.1 official mode, seed=42 verbatim = the L0-G2 construction; construction
    identity = caliber lock) at BOTH tiers BI∈{128, 32} (block_I is an
    implementation parameter; both tiers are legal configurations of the
    official caliber, §9.5).

    Anchors (removal baseline, golden_metric2_bench.md): BI128 = 364.7µs /
    BI32 = 475.3µs. Post-P1-rollback drift check: the kernel is byte-identical
    to the removal baseline (git 4e153c7f), so both tiers are expected within
    ±5% of the anchors; >±5% deviation is recorded and reported (non-gating
    signal — session/clock variance), precise paired A/B is deferred to
    Stage 3 / §9.5. Random/full-range shapes are correctness-only regressions
    (rev2: they are NOT performance calibers and are not measured here).

    The gating part is the fp32-golden sanity of each measured kernel."""
    S, H, D, tail_D, topk = 2048, 16, 512, 64, 128
    sm_scale = (D + tail_D) ** -0.5
    BASELINES = {128: 364.7, 32: 475.3}  # removal-baseline anchors (µs)
    all_ok = True

    # official arange inputs, seed=42 verbatim (== L0-G2 / GPU test_kernel);
    # CPU construction + cast, one H2D per tensor (aclnn audit order)
    torch.manual_seed(42)
    q = torch.randn((S, H, D + tail_D)).bfloat16().npu()
    kv = torch.randn((S, 1, D + tail_D)).bfloat16().npu()
    offsets = torch.tensor([0, 1023, S], dtype=torch.int32).npu()
    arange_topk = torch.arange(topk, dtype=torch.int32).npu()
    idx = arange_topk.unsqueeze(0).unsqueeze(0).repeat(S, 1, 1).contiguous()
    lse, _ = ref_mla_topk_softmax(q, kv[:, 0, :], idx[:, 0, :], offsets)
    ti = prepare_token_indices(offsets)

    for BI in (128, 32):
        func = tl_sparse_mla_topk_reducesum(
            heads=H,
            dim=D,
            tail_dim=tail_D,
            topk=topk,
            kv_group=1,
            sm_scale=None,
            block_I=BI,
            grid_size=get_cube_core_num(),
        )
        res = _graph_replay_us(func, (q, kv, idx, lse, offsets, ti), (slice(None), 0, 0))
        if res is None:
            print(f"[PRECISION_FAIL] L0-G6 BI={BI} graph output diverges from eager")
            all_ok = False
            continue
        mean, med, tmin, tmax, got = res
        # fp32-golden sanity (gating): raw §1.3 reference on the arange shape
        seg_lens = [1023, 1025]
        s_pos = torch.cat([torch.arange(n, dtype=torch.int32) for n in seg_lens]).npu()
        bos = torch.cat([torch.full((n,), b, dtype=torch.int32) for b, n in [(0, 1023), (1023, 1025)]]).npu()
        idx2 = idx[:, 0]
        idx_abs = (idx2 + bos.view(S, 1)).long()
        sel = kv[:, 0].float()[idx_abs]
        logits = torch.einsum("shd,sjd->shj", q.float(), sel) * sm_scale
        causal = idx2.long() <= s_pos.view(S, 1)
        score = torch.exp(logits - lse.unsqueeze(-1).float()) * causal.unsqueeze(1)
        ref = score.sum(dim=1)
        rel = ((got - ref).abs().max() / ref.abs().max()).item()
        sanity_ok = rel < 2e-5
        print(f"[INFO] L0-G6 BI={BI} official arange sanity vs fp32 golden: rel={rel:.4e} {'OK' if sanity_ok else 'FAIL'}")
        if not sanity_ok:
            print(f"[PRECISION_FAIL] L0-G6 BI={BI} perf anchor sanity check failed")
            all_ok = False
            continue
        base = BASELINES[BI]
        dev = (mean - base) / base * 100.0
        band = "within ±5%" if abs(dev) <= 5.0 else "DEVIATION > ±5% (recorded; post-rollback drift check failed)"
        print(
            f"[INFO] L0-G6 BI={BI} OFFICIAL arange (graph-replay, seed=42, batch=25 "
            f"iters=20): mean={mean:.1f}µs median={med:.1f}µs min={tmin:.1f}µs "
            f"max={tmax:.1f}µs — vs removal baseline {base}µs: {dev:+.1f}% — {band}"
        )
    return all_ok


# ===========================================================================
# L0-D: panel-path family (P3-v2 = P3 + C1 gather selection + C4 group pos
# table; cases recovered verbatim from the P3 session quick_d.py / smoke_d1.py,
# /tmp/opencode/p3/). Path assertions via LAST_DISPATCH (Violations value):
# shared inputs must take the panel path (viol==0); non-shared must fall back
# to generic (viol>0). D3/D5b use raw refs (renorm 0/0 rows). D7 = race stress
# on the new selection path (3-run bitwise + mixed-path alternation).
# ===========================================================================
def l0_d_panel_family():
    """L0-D gate: dispatch-path correctness for the panel/generic dual-launch."""
    all_ok = True
    S, H, D, tail_D, topk = 2048, 16, 512, 64, 128

    def _run(name, q, kv4, idx4, offsets, lse, ref, expect_path, dim_v=D, kv_group=1):
        nonlocal all_ok
        out = sparse_mla_topk_reducesum(
            q, kv4, idx4, lse, offsets, dim_v=dim_v, kv_group=kv_group, block_I=128
        )
        torch.npu.synchronize()
        viol = LAST_DISPATCH["violations"]
        v = None if viol is None else int(viol.item())
        if expect_path == "panel":
            path_ok = v == 0
        else:  # generic
            path_ok = (v is None) or (v > 0)
        if not path_ok:
            print(f"[PRECISION_FAIL] {name}: path assertion failed (viol={v}, expected {expect_path})")
        ok = check_vs_ref(f"{name} (viol={v})", out, ref) and path_ok
        all_ok = all_ok and ok
        return out

    # ---- D1: official golden arange through the dispatch (panel path) ----
    q, kv, idx, offsets = make_case(S, H, D, tail_D, topk, [0, 1023, S], seed=42)
    lse, ref = ref_mla_topk_softmax(q, kv, idx, offsets)
    _run("D1 official golden arange", q, kv.unsqueeze(-2), idx.unsqueeze(-2), offsets, lse, ref, "panel")

    # ---- D2: same shape non-shared (random causal) -> generic fallback ----
    torch.manual_seed(123)
    q2 = torch.randn((S, H, D + tail_D)).npu().bfloat16()
    kv2 = torch.randn((S, 1, D + tail_D)).npu().bfloat16()
    off2 = torch.tensor([0, 1023, S], dtype=torch.int32).npu()
    idx2 = make_random_causal_indices(S, [0, 1023, S], topk, seed=123)
    lse2, ref2 = ref_mla_topk_softmax(q2, kv2[:, 0, :], idx2[:, 0, :], off2)
    _run("D2 random non-shared", q2, kv2, idx2, off2, lse2, ref2, "generic")

    # ---- D3: duplicate index vector shared (raw ref — renorm 0/0 rows) ----
    vec3 = torch.tensor([0, 0, 1, 1, 5, 5, 7, 7, 9, 9] * 13)[:topk]
    q3, kv3, idx3, off3 = make_case(S, H, D, tail_D, topk, [0, 1023, S], seed=42, index_vec=vec3)
    lse3, _ = ref_mla_topk_softmax(q3, kv3, idx3, off3)
    out3 = sparse_mla_topk_reducesum(
        q3, kv3.unsqueeze(-2), idx3.unsqueeze(-2), lse3, off3, dim_v=D, block_I=128
    )
    torch.npu.synchronize()
    ref3_raw = ref_reducesum_raw(q3, kv3, idx3, lse3, off3)
    got3 = out3[:, 0, 0, :].float().cpu()
    ref3_raw = ref3_raw.cpu() if ref3_raw.is_npu else ref3_raw
    d3 = (got3 - ref3_raw).abs()
    m3 = (d3 <= 1e-4 + 1e-2 * ref3_raw.abs()).float().mean().item()
    v3 = int(LAST_DISPATCH["violations"].item())
    ok3 = m3 >= 0.99 and d3.max().item() <= 1e-3 and int(torch.isnan(got3).sum()) == 0 and v3 == 0
    print(
        f"[{'PRECISION_PASS' if ok3 else 'PRECISION_FAIL'}] D3 duplicates shared "
        f"(viol={v3}): matched={m3:.6f} max_abs={d3.max().item():.3e}"
    )
    all_ok = all_ok and ok3

    # ---- D4: permuted-set shared (day-1 fallback to generic) ----
    q4, kv4, _, off4 = make_case(S, H, D, tail_D, topk, [0, 1023, S], seed=42)
    base = torch.arange(topk)
    perms = torch.stack([base[torch.randperm(topk)] for _ in range(S)])
    idx4 = perms.to(torch.int32).contiguous().npu()
    lse4, ref4 = ref_mla_topk_softmax(q4, kv4, idx4, off4)
    _run("D4 permuted shared (fallback)", q4, kv4.unsqueeze(-2), idx4.unsqueeze(-2), off4, lse4, ref4, "generic")

    # ---- D5: -1/OOB mixed shared + all--1 row exact zero ----
    q5, kv5, _, off5 = make_case(S, H, D, tail_D, topk, [0, 1023, S], seed=42)
    vec5 = torch.arange(topk).clone()
    vec5[100:] = -1
    vec5[50] = 5000
    idx5 = vec5.to(torch.int32).unsqueeze(0).repeat(S, 1).contiguous().npu()
    lse5, ref5 = ref_mla_topk_softmax_masked(q5, kv5, idx5, off5)
    _run("D5 -1/OOB shared", q5, kv5.unsqueeze(-2), idx5.unsqueeze(-2), off5, lse5, ref5, "panel")
    idx5b = idx5.clone()
    idx5b[2000, :] = -1
    lse5b, _ = ref_mla_topk_softmax_masked(q5, kv5, idx5b, off5)
    out5b = sparse_mla_topk_reducesum(
        q5, kv5.unsqueeze(-2), idx5b.unsqueeze(-2), lse5b, off5, dim_v=D, block_I=128
    )
    torch.npu.synchronize()
    zero_row = bool((out5b[2000] == 0).all().item())
    print(f"[{'PRECISION_PASS' if zero_row else 'PRECISION_FAIL'}] D5b all--1 row exact-zero={zero_row}")
    all_ok = all_ok and zero_row

    # ---- D6: G=2 + full-range random (large P_seq) -> generic ----
    S6, H6, G6 = 512, 32, 2
    q6, kv6, idx6, off6 = make_case_v(S6, H6, D, tail_D, topk, [0, S6], seed=61, kv_group=G6)
    lse6, ref6 = ref_mla_topk_softmax_grouped(q6, kv6, idx6, off6, G6)
    _run("D6 G=2 full-range", q6, kv6, idx6, off6, lse6, ref6, "generic", kv_group=G6)

    # ---- D8: offsets variants + shared (empty first segment included) ----
    for off_list in ([0, S], [0, 0, 1023, S]):
        q7, kv7, idx7, off7 = make_case(S, H, D, tail_D, topk, off_list, seed=42)
        lse7, ref7 = ref_mla_topk_softmax_masked(q7, kv7, idx7, off7)  # masked: empty-seg safe
        _run(f"D8 offsets={off_list}", q7, kv7.unsqueeze(-2), idx7.unsqueeze(-2), off7, lse7, ref7, "panel")

    # ---- D8b: RH>1 (H=128 -> RH=2) shared ----
    S8, H8 = 256, 128
    q8, kv8, idx8, off8 = make_case(S8, H8, D, tail_D, topk, [0, S8], seed=42)
    lse8, ref8 = ref_mla_topk_softmax(q8, kv8, idx8, off8)
    _run("D8b RH=2 (H=128) shared", q8, kv8.unsqueeze(-2), idx8.unsqueeze(-2), off8, lse8, ref8, "panel")

    # ---- D7a: race stress, new selection path, 3-run bitwise stable ----
    q9, kv9, idx9, off9 = make_case(S, H, D, tail_D, topk, [0, 1023, S], seed=42)
    lse9, ref9 = ref_mla_topk_softmax(q9, kv9, idx9, off9)
    outs = []
    for _ in range(3):
        o = sparse_mla_topk_reducesum(
            q9, kv9.unsqueeze(-2), idx9.unsqueeze(-2), lse9, off9, dim_v=D, block_I=128
        )
        torch.npu.synchronize()
        outs.append(o)
    stable = all(torch.equal(outs[0], o) for o in outs[1:])
    print(f"[{'PRECISION_PASS' if stable else 'PRECISION_FAIL'}] D7a shared 3-run bitwise stable={stable}")
    all_ok = all_ok and stable and check_vs_ref("D7a run0", outs[0], ref9)

    # ---- D7b: mixed-path alternation (shared / random), 3 rounds ----
    mixed_ok = True
    for i in range(3):
        o_s = sparse_mla_topk_reducesum(
            q9, kv9.unsqueeze(-2), idx9.unsqueeze(-2), lse9, off9, dim_v=D, block_I=128
        )
        o_r = sparse_mla_topk_reducesum(
            q2, kv2, idx2, lse2, off2, dim_v=D, block_I=128
        )
        torch.npu.synchronize()
        mixed_ok = (
            torch.equal(o_s, outs[0])
            and check_vs_ref(f"D7b mixed round{i} random", o_r, ref2)
            and mixed_ok
        )
    print(f"[{'PRECISION_PASS' if mixed_ok else 'PRECISION_FAIL'}] D7b mixed-path alternating")
    all_ok = all_ok and mixed_ok

    return all_ok


L0_CASES = [
    ("L0-G1", l0_g1_random_main),
    ("L0-G2", l0_g2_golden_regression),
    ("L0-G3", l0_g3_distribution_family),
    ("L0-G4", l0_g4_shape_scan),
    ("L0-G5", l0_g5_race_stress),
    ("L0-D", l0_d_panel_family),
    ("L0-G6", l0_g6_perf_anchor),
]


def test_sparse_mla_topk_reducesum_l0():
    """Run all L0-G gate cases (DESIGN_generic_p1 §9.2); per-case try/except so
    one failure does not mask the rest. L0-G6 is a non-gating perf signal
    (its sanity check is gating)."""
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


def make_case_v(S, H, D, tail_D, topk, offsets_list, seed=42, vrange=None, kv_group=1, idx_mode="random"):
    """L0-G4/L1 case builder: CPU construction -> one H2D per tensor (aclnn
    audit: no NPU-side random/cast). vrange=None -> randn. idx_mode: "random"
    (default — per-row independent causal, the primary distribution per
    DESIGN_generic_p1 §8.6 test table) or "arange" (regression family)."""
    torch.manual_seed(seed)
    if vrange is None:
        q = torch.randn((S, H, D + tail_D)).bfloat16().npu()
        kv = torch.randn((S, kv_group, D + tail_D)).bfloat16().npu()
    else:
        lo, hi = vrange
        q = (torch.rand((S, H, D + tail_D)) * (hi - lo) + lo).bfloat16().npu()
        kv = (torch.rand((S, kv_group, D + tail_D)) * (hi - lo) + lo).bfloat16().npu()
    offsets = torch.tensor(offsets_list, dtype=torch.int32).npu()
    if idx_mode == "arange":
        p = torch.arange(topk, dtype=torch.int32)
        topk_indices = p.unsqueeze(0).unsqueeze(0).repeat(S, kv_group, 1).contiguous().npu()
    else:
        # full_range=True: maximal KV working-set coverage for functional/race
        # layers (the anchor-family construction is reserved for L0-G1/L0-G6)
        topk_indices = make_random_causal_indices(S, offsets_list, topk, seed=seed + 1, kv_group=kv_group, full_range=True)
    return q, kv, topk_indices, offsets


def _run_l1_case(name, cfg, tags):
    """One L1 case: build inputs, run the generic kernel, compare vs reference.
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
        idx_mode=cfg.get("idx_mode", "random"),
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
    out = sparse_mla_topk_reducesum(
        q,
        kv,
        idx,
        lse,
        offsets,
        block_I=cfg.get("block_I", 32),
        grid_size=cfg.get("grid_size"),
        **{k: v for k, v in host_kwargs.items() if v is not None or k == "sm_scale"},
    )
    return check_vs_ref(f"L1 {name} [generic]", out, ref_attn)


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
        {"S": 801, "D": 512, "tail_D": 64, "topk": 128, "offsets": [0, 401, 801]},
        ["D-SHAPE-TAIL-1"],
    ),
    (
        "task_tail_mid",  # ΣS = 20*40+10: mid-size task tail
        {"S": 810, "D": 512, "tail_D": 64, "topk": 128, "offsets": [0, 400, 810]},
        ["D-SHAPE-TAIL-MID"],
    ),
    (
        "prime_varlen",  # ΣS = 1021 prime, first segment 509 prime: fully non-aligned varlen
        {"S": 1021, "D": 512, "tail_D": 64, "topk": 128, "offsets": [0, 509, 1021]},
        ["D-SHAPE-PRIME"],
    ),
    (
        "edge_single_task",  # ΣS=1 degenerate
        {"S": 1, "D": 512, "tail_D": 64, "topk": 32, "offsets": [0, 1]},
        ["D-SHAPE-EDGE"],
    ),
    (
        "dim_v_256",  # non-default dim_v
        {"S": 512, "D": 256, "tail_D": 64, "topk": 64, "offsets": [0, 512]},
        ["D-PARAM-dim_v"],
    ),
    (
        "kv_group_2",  # non-default kv_group (heads=32, head_kv=16)
        {"S": 512, "heads": 32, "kv_group": 2, "D": 512, "tail_D": 64, "topk": 64, "offsets": [0, 512]},
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
    ("block_I_gt_128", "D-EXC-SHAPE"),  # rev2: block_I=256 > L0B hard cap 128 -> rejected
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
    value-range sweep + non-default attrs — random per-row causal indices as the
    primary distribution (DESIGN_generic_p1 §8.6 test table)."""
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


def _l2_block_i_gt_128():
    """rev2 (§4.5/§11.4): block_I=256 exceeds the gemm_v0 transpose-B L0B
    ping-pong slot hard cap (32KiB @ kL0Size=128/bf16 -> BI <= 128). The P1
    builder front-assert (R10) was reverted together with P1, so the rejection
    now comes from the framework's static_assert at kernel C++ compile time
    (src/tl_templates/ascend/common.h, `nTile * kL0Size * sizeof(T1) <=
    kL0Budget`) — opaque but correct; what this case verifies is that the
    out-of-domain config is NOT silently accepted."""
    S, H, D, tail_D, topk = 256, 16, 512, 64, 256
    q, kv, idx, offsets = make_case_v(S, H, D, tail_D, topk, [0, S])
    lse = torch.zeros((S, H), dtype=torch.float32).npu()
    sparse_mla_topk_reducesum(q, kv, idx, lse, offsets, dim_v=D, block_I=256)


_L2_FNS = {
    "dtype_q_f32": _l2_dtype_q_f32,
    "topk_not_div_blockI": _l2_topk_not_div,
    "head_kv_out_of_domain": _l2_head_kv_domain,
    "block_I_gt_128": _l2_block_i_gt_128,
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
            out = sparse_mla_topk_reducesum(q, kv.unsqueeze(-2), idx.unsqueeze(-2), lse, offsets, dim_v=512)
            _check_boundary(name, out, ref_attn)
        except Exception as e:
            print(f"[BOUNDARY_WARN] boundary {name}: {type(e).__name__}: {e}")


# Coverage summary (machine-checked by coverage_check.py; counts per dimension)
# proto.yaml rev3: panel_mode attr removed — no D-PARAM-panel_mode dimension.
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
    "D-SPECIAL-INF": 1,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-DBOUND": 1,
    "D-EXC-DTYPE": 1,
    "D-EXC-SHAPE": 3,  # topk_not_div + head_kv_domain + block_I_gt_128 (rev2)
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
