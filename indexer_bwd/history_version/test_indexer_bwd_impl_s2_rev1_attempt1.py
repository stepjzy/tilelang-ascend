"""indexer_bwd layered tests: L0 (precision convergence) + stubs + perf benchmark.

Golden replicates the main repo's ref_indexer_bwd autograd reference
(ref_gpu/indexer_bwd.py:185-215) and runs it on NPU (torch_npu supports bf16
autograd / einsum / kl_div, mirroring the main repo's cuda golden). Input
construction replicates the main repo's test_kernel (seed 42, randn bf16
q/w/k, masked-softmax attn_score from ones, arange-tiled topk_indices); all
randomness and dtype conversions happen on CPU followed by a single H2D.

Levels:
    l0        DESIGN.md §9.2 gate cases (blocking, [PRECISION_PASS/FAIL])
    l1/l2/boundary  stubs, expanded by tilelang-op-test-design (scenario B)
    perf      pure-kernel average latency for the l0_default configuration
"""

import argparse
import os
import sys
import traceback

import torch
import torch.nn.functional as F

import tilelang

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indexer_bwd import (  # noqa: E402,F401
    indexer_bwd,
    indexer_bwd_interface,
    prepare_token_indices,
    tl_indexer_bwd_impl,
)


# ============================================================================
# Golden reference (main repo ref_gpu/indexer_bwd.py:185-215, verbatim;
# einops.einsum -> torch.einsum, cuda -> npu)
# ============================================================================
def ref_indexer_bwd(Q, Weights, K, TopkIndices, AttnScore, offsets):
    Q.requires_grad_(True)
    Weights.requires_grad_(True)
    K.requires_grad_(True)
    softmax_scale = Q.shape[-1] ** -0.5
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
def make_inputs(S, H, D, topk, off_list, seed=42):
    torch.manual_seed(seed)
    q = torch.randn(S, H, D).bfloat16().npu()
    w = torch.randn(S, H).bfloat16().npu()
    k = torch.randn(S, D).bfloat16().npu()
    offsets = torch.tensor(off_list, dtype=torch.int32).npu()

    parts = []
    for i in range(len(off_list) - 1):
        seg = off_list[i + 1] - off_list[i]
        mask = torch.arange(seg)[:, None] >= torch.arange(topk)[None, :]
        logits = torch.where(mask, torch.ones(seg, topk), float("-inf"))
        parts.append(F.softmax(logits, dim=-1, dtype=torch.float32))
    attn_score = torch.cat(parts, dim=0).npu()

    topk_indices = torch.arange(topk, dtype=torch.int32).unsqueeze(0).repeat(S, 1).npu()
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
    """Mixed-tolerance dual-gate check: returns (passed, matched_ratio, max_abs_error)."""
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
# L0 gate tests (DESIGN.md §9.2)
# ============================================================================
L0_CONFIGS = [
    # (name, B, S, H, D, topk, block_I, offsets)
    ("l0_default", 1, 2048, 16, 128, 64, 64, [0, 1023, 2048]),
    ("l0_single_seq", 1, 1024, 16, 128, 64, 64, [0, 1024]),
    ("l0_small", 1, 128, 16, 128, 64, 64, [0, 128]),
    ("l0_blockI32", 1, 512, 16, 128, 64, 32, [0, 512]),
]


def test_indexer_bwd_l0():
    """L0 gate: regular shapes (block-divisible), precision convergence."""
    ok = True
    for name, _B, S, H, D, topk, block_I, off_list in L0_CONFIGS:
        try:
            q, w, k, attn_score, topk_indices, offsets = make_inputs(S, H, D, topk, off_list)
            index_score, ref_dq, ref_dw, ref_dk = ref_indexer_bwd(q, w, k, topk_indices, attn_score, offsets)
            dq, dw, dk = indexer_bwd_interface(q, w, k, attn_score, index_score, topk_indices, offsets, block_I=block_I)
            case_ok = True
            for oname, out, ref in (
                ("dq", dq, ref_dq),
                ("dw", dw, ref_dw),
                ("dk", dk, ref_dk),
            ):
                passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
                tag = "PASS" if passed else "FAIL"
                print(
                    f"[PRECISION_{tag}] {name} {oname} shape={tuple(out.shape)} "
                    f"dtype=bfloat16 matched_ratio={ratio:.4f} max_abs={max_abs:.3e}"
                )
                case_ok &= passed
            ok &= case_ok
        except Exception as e:  # noqa: BLE001
            print(f"[PRECISION_FAIL] {name}: {e!r}")
            traceback.print_exc()
            ok = False
    return ok


# ============================================================================
# L1/L2/Boundary: stubs, expanded by tilelang-op-test-design (scenario B)
# ============================================================================
def test_indexer_bwd_l1():
    """L1 functional tests (irregular/tail shapes) — stub."""
    print("[L1] not expanded yet — run tilelang-op-test-design (scenario B)")
    return True


def test_indexer_bwd_l2():
    """L2 negative tests (invalid inputs must be rejected) — stub."""
    print("[L2] not expanded yet — run tilelang-op-test-design (scenario B)")


def test_indexer_bwd_boundary():
    """Boundary tests (legal special values) — stub."""
    print("[BOUNDARY] not expanded yet — run tilelang-op-test-design (scenario B)")


# ============================================================================
# Perf benchmark: pure-kernel average latency (l0_default configuration)
# ============================================================================
def benchmark_perf():
    """Time the kernel only (torch.npu events, warmup + averaged iters).

    Excludes H2D, host output/workspace allocation and explicit syncs: all
    tensors are prepared once, then the compiled kernel object is launched
    back-to-back inside one event pair.
    """
    S, H, D, topk, block_I = 2048, 16, 128, 64, 64
    off_list = [0, 1023, S]
    q, w, k, attn_score, topk_indices, offsets = make_inputs(S, H, D, topk, off_list)
    token_indices = prepare_token_indices(offsets)
    index_score = attn_score  # values irrelevant for timing
    dq = torch.zeros(S, H, D, dtype=torch.bfloat16, device=q.device)
    dw = torch.zeros(S, H, dtype=torch.bfloat16, device=q.device)
    dk = torch.zeros(S, D, dtype=torch.bfloat16, device=q.device)
    workspace_k = torch.empty(S, topk, D, dtype=torch.bfloat16, device=q.device)
    workspace_dlog = torch.empty(S, topk, H, dtype=torch.bfloat16, device=q.device)
    workspace_logits = torch.empty(S, topk, H, dtype=torch.float32, device=q.device)
    workspace_dk = torch.empty(S, topk, D, dtype=torch.bfloat16, device=q.device)

    kernel = tl_indexer_bwd_impl(H, D, topk, D**-0.5, block_I)
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
        dk,
        workspace_k,
        workspace_dlog,
        workspace_logits,
        workspace_dk,
    )
    for _ in range(5):
        kernel(*args)
    torch.npu.synchronize()
    n_iters = 20
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        kernel(*args)
    end.record()
    torch.npu.synchronize()
    avg_us = start.elapsed_time(end) / n_iters * 1000.0
    print(
        f"[PERF] indexer_bwd l0_default (S={S}, H={H}, D={D}, topk={topk}, bf16): "
        f"pure kernel avg = {avg_us:.1f} us over {n_iters} iters "
        f"(torch.npu events; excludes H2D/sync/host alloc; target < 80 us)"
    )
    return avg_us


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
