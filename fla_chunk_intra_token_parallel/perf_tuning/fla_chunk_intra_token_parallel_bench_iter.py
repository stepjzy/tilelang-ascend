"""fla_chunk_intra_token_parallel Stage 3 iteration bench.

Timing protocol = upstream do_bench parity (ref_gpu/test_utils_kda.py):
NPU events (torch.npu.Event(enable_timing=True)), warmup=20, rep=10, mean,
device-side only (no H2D/D2H/JIT in the timed region), no internal sync.
Cross-check: loop mode (one event pair around N back-to-back launches, /N)
bounds host-dispatch pollution (tilelang Python dispatch ~0.1ms/launch).

Bench shape (DESIGN SS9.4 perf_bench sentinel): B=1, S=8192, H=64, DK=128,
CS=64, SCS=16, q/k/beta bf16 + gk fp32 -> Aqk/Akk bf16, scale=1.0.

IO floor reference (DESIGN SS9.5): 593 MiB = 621,409,280 B unique IO;
measured 910B3 D2D r+w ceiling 1259-1273 GB/s -> pure-bandwidth floor ~492us.
"""
import argparse
import json
import os
import sys

import tilelang
import torch
import torch_npu  # noqa: F401  (registers .npu())

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fla_chunk_intra_token_parallel import fla_chunk_intra_token_parallel  # noqa: E402

# unique IO bytes @ (1, 8192, 64, 128): q 128MiB + k 128MiB + gk 256MiB + beta 1MiB
# + Aqk 64MiB + Akk 16MiB
IO_BYTES = 1 * 8192 * 64 * 128 * (2 + 2 + 4) + 1 * 8192 * 64 * 2 + 1 * 8192 * 64 * (64 + 16) * 2
D2D_CEILING_GB_S = 1263.0  # measured 910B3 r+w ceiling (fla_wy_fast journal calibration)


def make_bench_inputs(seed=42):
    """Upstream prepare_input semantics (test suite construction), bench shape."""
    B, S, H, DK = 1, 8192, 64, 128
    torch.manual_seed(seed)
    q = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
    k = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
    beta = torch.randn(B, S, H, dtype=torch.bfloat16)
    gk = torch.nn.functional.logsigmoid(torch.randn(B, S, H, DK, dtype=torch.float32))
    gk = gk.view(B, S // 64, 64, H, DK).cumsum(dim=2).view(B, S, H, DK)
    return q.npu(), k.npu(), gk.npu(), beta.npu()


def bench_events(fn, args, warmup=20, rep=10):
    """Per-rep NPU event pairs (upstream do_bench parity)."""
    for _ in range(warmup):
        fn(*args)
    torch.npu.synchronize()
    start = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    end = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start[i].record()
        fn(*args)
        end[i].record()
    torch.npu.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start, end)]
    return sum(times) / len(times), times


def bench_loop(fn, args, warmup=20, iters=30):
    """One event pair around N back-to-back launches: bounds host dispatch."""
    for _ in range(warmup):
        fn(*args)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--loop-iters", type=int, default=30)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    # tilelang JIT cache stays ENABLED (first call compiles, re-runs hit cache)
    op = fla_chunk_intra_token_parallel
    npu_in = make_bench_inputs()
    op(*npu_in)  # compile + first call (untimed)
    torch.npu.synchronize()

    mean_ms, times = bench_events(op, npu_in, warmup=args.warmup, rep=args.rep)
    loop_ms = bench_loop(op, npu_in, warmup=args.warmup, iters=args.loop_iters)

    mean_us = mean_ms * 1000.0
    loop_us = loop_ms * 1000.0
    eff_bw = IO_BYTES / (mean_us * 1e-6) / 1e9  # GB/s
    util = eff_bw / D2D_CEILING_GB_S * 100.0
    floor_us = IO_BYTES / (D2D_CEILING_GB_S * 1e9) * 1e6

    print(f"[bench shape=(1,8192,64,128) cs=64 scs=16]")
    print(f"  events mean = {mean_us:.1f} us  (per-rep: {[f'{t*1000:.0f}' for t in times]} us)")
    print(f"  loop  mean = {loop_us:.1f} us  ({args.loop_iters} back-to-back launches)")
    print(f"  unique IO = {IO_BYTES} B -> effective bw {eff_bw:.0f} GB/s = {util:.1f}% of measured ceiling")
    print(f"  pure-bandwidth floor @ {D2D_CEILING_GB_S:.0f} GB/s = {floor_us:.0f} us")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "shape": [1, 8192, 64, 128],
                    "chunk_size": 64,
                    "sub_chunk_size": 16,
                    "timing_method": "torch.npu.Event(enable_timing=True), warmup=20 rep=10 mean, device-side only (upstream do_bench parity); loop mode cross-check",
                    "events_mean_us": mean_us,
                    "per_rep_us": [t * 1000 for t in times],
                    "loop_mean_us": loop_us,
                    "io_bytes": IO_BYTES,
                    "effective_bw_gbs": eff_bw,
                    "bw_utilization_pct": util,
                    "bandwidth_floor_us": floor_us,
                },
                f,
                indent=2,
            )
        print(f"  json -> {args.json_out}")


if __name__ == "__main__":
    main()
