"""Per-kernel timing decomposition for fla_wy_fast (910B3).

Bench each of the 8 kernels (fwd: u, wkg; bwd: k1, k2, v1, v2, z, da)
independently with the same do_bench protocol (warmup=20, rep=10, mean,
device-side only). Also measures a large D2D copy bandwidth as an HBM
ceiling reference.
"""

import sys

import torch

sys.path.insert(0, "/mnt/workspace/gitCode/skybasejzy/tilelang-ascend")
sys.path.insert(0, "/mnt/workspace/gitCode/skybasejzy/tilelang-ascend/examples_experiment/dsa_sparse_finetune")

import tilelang

tilelang.disable_cache()

from fla_wy_fast import (
    _pick_block_d,
    fla_wy_bwd_da_kernel,
    fla_wy_bwd_k1_kernel,
    fla_wy_bwd_k2_kernel,
    fla_wy_bwd_v1_kernel,
    fla_wy_bwd_v2_kernel,
    fla_wy_bwd_z_kernel,
    fla_wy_fwd_u_kernel,
    fla_wy_fwd_wkg_kernel,
)

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


def hbm_reference():
    """Large D2D copy: bytes moved = 2 * nbytes (read + write)."""
    n = 512 * 1024 * 1024  # 512 MiB per tensor
    a = torch.empty(n, dtype=torch.uint8, device="npu")
    b = torch.empty(n, dtype=torch.uint8, device="npu")
    t = do_bench(lambda: b.copy_(a), warmup=5, rep=10)
    bw = 2 * n / (t * 1e-3) / 1e9
    print(f"[HBM-REF] D2D copy 512MiB: {t:.3f} ms -> {bw:.0f} GB/s (r+w)")
    return bw


def bench_fwd_kernels():
    B, S, H, DK, DV = 1, 8192, 64, 128, 128
    gen = torch.Generator().manual_seed(42)
    k = torch.randn(B, S, H, DK, generator=gen).to(torch.bfloat16).npu()
    v = torch.randn(B, S, H, DV, generator=gen).to(torch.bfloat16).npu()
    beta = torch.randn(B, S, H, generator=gen).to(torch.bfloat16).npu()
    gk = torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32).npu()
    A = torch.randn(B, S, H, 64, generator=gen).to(torch.bfloat16).npu()
    beta_flat = beta.view(B, S * H)

    u_kernel = fla_wy_fwd_u_kernel(B, S, H, DV, 64, _pick_block_d(DV))
    t_u = do_bench(u_kernel, v, beta_flat, A)

    wkg_kernel = fla_wy_fwd_wkg_kernel(B, S, H, DK, 64, _pick_block_d(DK))
    t_wkg = do_bench(wkg_kernel, k, beta_flat, gk, A)

    print(f"[FWD-U]    {t_u:.3f} ms")
    print(f"[FWD-WKG]  {t_wkg:.3f} ms")
    print(f"[FWD-SUM]  {t_u + t_wkg:.3f} ms")
    return t_u, t_wkg


def bench_bwd_kernels():
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
    grid = (S // 64) * B * H
    msk = torch.tril(torch.ones(64, 64, dtype=torch.float32), -1).npu()
    beta_flat = beta.view(B, S * H)

    k1 = fla_wy_bwd_k1_kernel(B, S, H, DK, 64, _pick_block_d(DK))
    t_k1 = do_bench(k1, k, beta_flat, gk, dw)
    dA_part = k1(k, beta_flat, gk, dw)

    k2 = fla_wy_bwd_k2_kernel(B, S, H, DK, 64, _pick_block_d(DK))
    t_k2 = do_bench(k2, k, beta_flat, gk, A, dw, dk, dg)
    dk2, dg2, dbeta_flat = k2(k, beta_flat, gk, A, dw, dk, dg)

    v1 = fla_wy_bwd_v1_kernel(B, S, H, DV, 64, _pick_block_d(DV))
    t_v1 = do_bench(v1, v, beta_flat, du, msk, dA_part)
    s_out = v1(v, beta_flat, du, msk, dA_part)

    v2 = fla_wy_bwd_v2_kernel(B, S, H, DV, 64, _pick_block_d(DV))
    t_v2 = do_bench(
        v2, v, beta_flat, A, du, dbeta_flat, dbeta_flat.view(grid, 64, 1)
    )

    z_kernel = fla_wy_bwd_z_kernel(B, S, H, 64)
    t_z = do_bench(z_kernel, s_out, A)
    z = z_kernel(s_out, A)

    da_kernel = fla_wy_bwd_da_kernel(B, S, H, 64)
    t_da = do_bench(da_kernel, A, z, msk)

    total = t_k1 + t_k2 + t_v1 + t_v2 + t_z + t_da
    print(f"[BWD-K1]   {t_k1:.3f} ms  (dA_part = dw @ K_bg^T)")
    print(f"[BWD-K2]   {t_k2:.3f} ms  (X = A^T@dw -> dk2/dg2/dbeta_part)")
    print(f"[BWD-V1]   {t_v1:.3f} ms  (S = tril(dA_part + du @ V_b^T))")
    print(f"[BWD-V2]   {t_v2:.3f} ms  (Y = A^T@du -> dv/dbeta)")
    print(f"[BWD-Z]    {t_z:.3f} ms  (Z = S @ A^T)")
    print(f"[BWD-DA]   {t_da:.3f} ms  (dA = -tril(A^T @ Z))")
    print(f"[BWD-SUM]  {total:.3f} ms")
    return t_k1, t_k2, t_v1, t_v2, t_z, t_da


if __name__ == "__main__":
    print("device:", torch.npu.get_device_name(0))
    hbm_reference()
    bench_fwd_kernels()
    bench_bwd_kernels()
