"""fla_wy_fast un-tuned performance baseline (910B3).

Timing mirrors the upstream do_bench (ref_gpu/test_utils_kda.py): NPU event
timing, warmup=20, rep=10, mean of per-rep elapsed times, device-side kernel
time only. fwd/bwd measured separately and summed.
"""

import sys

import torch

sys.path.insert(0, "/mnt/workspace/gitCode/skybasejzy/tilelang-ascend")
sys.path.insert(0, "/mnt/workspace/gitCode/skybasejzy/tilelang-ascend/examples_experiment/dsa_sparse_finetune")

import tilelang

tilelang.disable_cache()

from fla_wy_fast import fla_wy_fast_bwd, fla_wy_fast_fwd

torch.manual_seed(42)


def do_bench(fn, *args, warmup=20, rep=10):
    start = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    end = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    for _ in range(warmup):
        fn(*args)
    torch.npu.synchronize()
    for i in range(rep):
        start[i].record()
        fn(*args)
        end[i].record()
    torch.npu.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start, end)]
    return sum(times) / len(times)


def bench_fwd():
    B, S, H, DK, DV = 1, 8192, 64, 128, 128
    gen = torch.Generator().manual_seed(42)
    k = torch.randn(B, S, H, DK, generator=gen).to(torch.bfloat16).npu()
    v = torch.randn(B, S, H, DV, generator=gen).to(torch.bfloat16).npu()
    beta = torch.randn(B, S, H, generator=gen).to(torch.bfloat16).npu()
    gk = torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32).npu()
    A = torch.randn(B, S, H, 64, generator=gen).to(torch.bfloat16).npu()
    print("fwd bench (1, 8192, 64, 128, 128) bf16 + fp32 gate: compiling...")
    fla_wy_fast_fwd(k, v, beta, gk, A)  # compile
    torch.npu.synchronize()
    t = do_bench(fla_wy_fast_fwd, k, v, beta, gk, A)
    print(f"[FWD] {t:.3f} ms (mean of 10, warmup 20)")
    return t


def bench_bwd():
    B, S, H, DK, DV = 1, 32768, 8, 128, 128
    gen = torch.Generator().manual_seed(42)
    k = torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32).npu()
    v = torch.randn(B, S, H, DV, generator=gen, dtype=torch.float32).npu()
    beta = torch.randn(B, S, H, generator=gen, dtype=torch.float32).npu()
    gk = torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32).npu()
    A = torch.randn(B, S, H, 64, generator=gen, dtype=torch.float32).npu()
    dw = torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32).npu()
    du = torch.randn(B, S, H, DV, generator=gen, dtype=torch.float32).npu()
    dk = torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32).npu()
    dg = torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32).npu()
    print("bwd bench (1, 32768, 8, 128, 128) fp32: compiling...")
    fla_wy_fast_bwd(k, v, beta, gk, A, dw, du, dk, dg)  # compile
    torch.npu.synchronize()
    t = do_bench(fla_wy_fast_bwd, k, v, beta, gk, A, dw, du, dk, dg)
    print(f"[BWD] {t:.3f} ms (mean of 10, warmup 20)")
    return t


if __name__ == "__main__":
    print("device:", torch.npu.get_device_name(0))
    tf = bench_fwd()
    tb = bench_bwd()
    print(f"[SUM] fwd + bwd = {tf + tb:.3f} ms")
