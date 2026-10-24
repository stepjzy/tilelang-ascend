"""A/B iteration bench for fla_chunk_o Stage 3 tuning.

Timing protocol = test_fla_chunk_o.py --level perf (upstream do_bench parity):
NPU events, warmup=10, rep=10, mean, device-side only, no internal sync.
Plus a fast precision gate on the l0_repr shape (2,256,4,128,128) against the
CPU golden (same construction as the test suite) so iterations can be gated
without paying the full bench-shape CPU golden each time.
"""
import argparse
import os
import sys

import tilelang
import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fla_chunk_o import fla_chunk_o  # noqa: E402
from test_fla_chunk_o import fla_chunk_o_ref, check_precision, make_inputs  # noqa: E402


def bench(shape, warmup=10, rep=10, seed=42):
    B, S, H, DK, DV = shape
    npu_in, _cpu = make_inputs(B, S, H, DK, DV, 64, seed)
    o = fla_chunk_o(*npu_in)  # compile + warm call
    torch.npu.synchronize()
    for _ in range(warmup):
        fla_chunk_o(*npu_in)
    torch.npu.synchronize()
    start = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    end = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start[i].record()
        fla_chunk_o(*npu_in)
        end[i].record()
    torch.npu.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start, end)]
    mean = sum(times) / len(times)
    steady = sorted(times)[1:-1]
    steady_mean = sum(steady) / len(steady) if steady else mean
    return mean, times, steady_mean


def quick_precision(shape=(2, 256, 4, 128, 128), seed=1002):
    npu_in, cpu_in = make_inputs(*shape, 64, seed)
    o = fla_chunk_o(*npu_in)
    torch.npu.synchronize()
    ref = fla_chunk_o_ref(*cpu_in)
    return check_precision(o, ref, "bfloat16")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="1,8192,64,128,128")
    parser.add_argument("--rep", type=int, default=10)
    args = parser.parse_args()
    shape = tuple(int(x) for x in args.shape.split(","))
    tilelang.disable_cache()

    passed, ratio, max_abs = quick_precision()
    tag = "PASS" if passed else "FAIL"
    print(f"[quick-precision l0_repr] {tag} ratio={ratio:.4f} max_abs={max_abs:.3e}")
    if not passed:
        sys.exit(2)

    mean, times, steady = bench(shape, rep=args.rep)
    print(
        f"[bench {shape}] mean={mean:.3f} ms steady={steady:.3f} ms "
        f"per-rep={[f'{t:.3f}' for t in times]}"
    )
    us = mean * 1000
    print(f"target < 1320 us -> {'OK' if us < 1320 else 'OVER'} (current {us:.0f} us)")


if __name__ == "__main__":
    main()
