"""fla_chunk_delta Stage-3 profiling entry (msprof op / app-level msprof).

Runs ONE kernel kind per process (fwd and bwd share the resolved kernel name
`main_kernel`, so msprof --kernel-name filtering cannot separate them inside
a single process). Launch pattern: --warmup W (default 15) then --launches N
(default 5) steady-state launches; msprof op uses --warm-up=10
--launch-count=1 and captures the 11th matched launch.

--dump-src additionally writes the generated kernel C source of both
specializations (bench shape, wrapper-default constexpr branch) to
fla_chunk_delta_{fwd,bwd}_kernel_source.c next to this script.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fla_chunk_delta import (  # noqa: E402
    fla_chunk_delta_bwd,
    fla_chunk_delta_fwd,
    fla_chunk_delta_bwd_kernel,
    fla_chunk_delta_fwd_kernel,
    fla_chunk_delta_prep_kernel,
    _pick_bv,
)

DEVICE = "npu:0"
BENCH_SHAPE = (1, 8192, 64, 128, 128)  # B, S, H, DK, DV


def make_fwd_inputs():
    B, S, H, DK, DV = BENCH_SHAPE
    gen = torch.Generator().manual_seed(9001)
    k = torch.randn(B, S, H, DK, generator=gen).to(torch.float16).to(DEVICE)
    w = torch.randn(B, S, H, DK, generator=gen).to(torch.float16).to(DEVICE)
    u = torch.randn(B, S, H, DV, generator=gen).to(torch.float16).to(DEVICE)
    g = torch.randn(B, S, H, generator=gen).to(torch.float32).to(DEVICE)
    gk = torch.randn(B, S, H, DK, generator=gen).to(torch.float32).to(DEVICE)
    return k, w, u, g, gk


def make_bwd_inputs():
    B, S, H, DK, DV = BENCH_SHAPE
    gen = torch.Generator().manual_seed(9001)
    q = torch.randn(B, S, H, DK, generator=gen).to(torch.bfloat16).to(DEVICE)
    k = torch.randn(B, S, H, DK, generator=gen).to(torch.bfloat16).to(DEVICE)
    w = torch.randn(B, S, H, DK, generator=gen).to(torch.bfloat16).to(DEVICE)
    do = torch.randn(B, S, H, DV, generator=gen).to(torch.bfloat16).to(DEVICE)
    dv = torch.randn(B, S, H, DV, generator=gen).to(torch.bfloat16).to(DEVICE)
    g = torch.randn(B, S, H, generator=gen).to(torch.float32).to(DEVICE)
    gk = torch.randn(B, S, H, DK, generator=gen).to(torch.float32).to(DEVICE)
    return q, k, w, do, dv, g, gk


def run_fwd(warmup, launches):
    k, w, u, g, gk = make_fwd_inputs()
    for _ in range(warmup):
        fla_chunk_delta_fwd(k, w, u, g, gk)
    torch.npu.synchronize()
    for _ in range(launches):
        fla_chunk_delta_fwd(k, w, u, g, gk)
    torch.npu.synchronize()
    print("[entry] fwd launches done", flush=True)


def run_fwd_main_only(warmup, launches):
    """Loop ONLY the fwd recurrence kernel with SYNTHETIC gate tensors
    (msprof op captures the first matched launch and replays it; with the
    prep in the stream the prep would always be the captured one). Values
    in (0.1, 0.9): representative gate/decay magnitudes, no denormals."""
    k, w, u, g, gk = make_fwd_inputs()
    B, S, H, DK, DV = BENCH_SHAPE
    NT = S // 64
    gt = (0.1 + 0.8 * torch.rand(B, NT, H, 64, dtype=torch.float32, device=DEVICE))
    gd = (0.1 + 0.8 * torch.rand(B, NT, H, DK, dtype=torch.float32, device=DEVICE))
    kern = fla_chunk_delta_fwd_kernel(B, S, H, DK, DV, 64, _pick_bv(DV), False, False, False, True)
    dummy = torch.empty(1, dtype=torch.float32, device=DEVICE)
    args = (
        k.view(B, S, H * DK),
        w.view(B, S, H * DK),
        u.view(B, S, H * DV),
        gt,
        gd,
        dummy,
    )
    for _ in range(warmup):
        kern(*args)
    torch.npu.synchronize()
    for _ in range(launches):
        kern(*args)
    torch.npu.synchronize()
    print("[entry] fwd main-only launches done", flush=True)


def run_bwd_main_only(warmup, launches):
    """Loop ONLY the bwd recurrence kernel with synthetic gate tensors.
    iter10: the main kernel takes the prep-materialized QT (pre-gated q~,
    scale folded) instead of raw Q + GE; synthetic QT carries ~1-magnitude
    values (no denormals)."""
    q, k, w, do, dv, g, gk = make_bwd_inputs()
    B, S, H, DK, DV = BENCH_SHAPE
    NT = S // 64
    gt = (0.1 + 0.8 * torch.rand(B, NT, H, 64, dtype=torch.float32, device=DEVICE))
    gd = (0.1 + 0.8 * torch.rand(B, NT, H, DK, dtype=torch.float32, device=DEVICE))
    qt = (0.5 + 0.4 * torch.rand(B, S, H * DK, dtype=torch.bfloat16, device=DEVICE))
    kern = fla_chunk_delta_bwd_kernel(B, S, H, DK, DV, 64, 64, None, False, False, False)
    dummy = torch.empty(1, dtype=torch.float32, device=DEVICE)
    args = (
        qt,
        k.view(B, S, H * DK),
        w.view(B, S, H * DK),
        do.view(B, S, H * DV),
        dv.view(B, S, H * DV),
        gt,
        gd,
        dummy,
    )
    for _ in range(warmup):
        kern(*args)
    torch.npu.synchronize()
    for _ in range(launches):
        kern(*args)
    torch.npu.synchronize()
    print("[entry] bwd main-only launches done", flush=True)


def run_bwd(warmup, launches):
    q, k, w, do, dv, g, gk = make_bwd_inputs()
    for _ in range(warmup):
        fla_chunk_delta_bwd(q, k, w, do, dv, g, gk)
    torch.npu.synchronize()
    for _ in range(launches):
        fla_chunk_delta_bwd(q, k, w, do, dv, g, gk)
    torch.npu.synchronize()
    print("[entry] bwd launches done", flush=True)


def dump_sources():
    here = os.path.dirname(os.path.abspath(__file__))
    B, S, H, DK, DV = BENCH_SHAPE
    try:
        kern = fla_chunk_delta_fwd_kernel(B, S, H, DK, DV, 64, _pick_bv(DV), False, False, False, True)
        src = kern.get_kernel_source()
        with open(os.path.join(here, "fla_chunk_delta_fwd_kernel_source.c"), "w") as f:
            f.write(src)
        print(f"[entry] fwd source dumped: {len(src.splitlines())} lines", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[entry] fwd source dump failed: {e!r}", flush=True)
    try:
        kern = fla_chunk_delta_bwd_kernel(B, S, H, DK, DV, 64, 64, None, False, False, False)
        src = kern.get_kernel_source()
        with open(os.path.join(here, "fla_chunk_delta_bwd_kernel_source.c"), "w") as f:
            f.write(src)
        print(f"[entry] bwd source dumped: {len(src.splitlines())} lines", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[entry] bwd source dump failed: {e!r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", choices=["fwd", "bwd", "all", "fwd-main", "bwd-main"], default="all")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--launches", type=int, default=5)
    ap.add_argument("--dump-src", action="store_true")
    args = ap.parse_args()
    if args.dump_src:
        dump_sources()
    if args.kernel == "fwd" or args.kernel == "all":
        run_fwd(args.warmup, args.launches)
    if args.kernel == "bwd" or args.kernel == "all":
        run_bwd(args.warmup, args.launches)
    if args.kernel == "fwd-main":
        run_fwd_main_only(args.warmup, args.launches)
    if args.kernel == "bwd-main":
        run_bwd_main_only(args.warmup, args.launches)


if __name__ == "__main__":
    main()
