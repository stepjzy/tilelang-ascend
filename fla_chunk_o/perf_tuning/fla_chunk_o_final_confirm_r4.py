"""R4 final confirmation: official bench x3 + fast-path dispatch breakdown."""
import os
import sys
import time

import tilelang
import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fla_chunk_o as m  # noqa: E402
from test_fla_chunk_o import make_inputs  # noqa: E402

tilelang.disable_cache()

B, S, H, DK, DV = 1, 8192, 64, 128, 128
npu_in, _ = make_inputs(B, S, H, DK, DV, 64, 42)
o = m.fla_chunk_o(*npu_in)
torch.npu.synchronize()
n_armed = sum(1 for v in m._FAST.values() if v is not None)
print(f"fast cache armed={n_armed} poisoned={sum(1 for v in m._FAST.values() if v is None)}")


def bench_once(rep=10, warmup=10):
    for _ in range(warmup):
        m.fla_chunk_o(*npu_in)
    torch.npu.synchronize()
    start = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    end = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start[i].record()
        m.fla_chunk_o(*npu_in)
        end[i].record()
    torch.npu.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start, end)]
    mean = sum(times) / len(times)
    steady = sorted(times)[1:-1]
    return mean, times, sum(steady) / len(steady)


for r in range(3):
    mean, times, steady = bench_once()
    print(f"[final-bench {r+1}] mean={mean:.3f} steady={steady:.3f} "
          f"first={times[0]:.3f} per-rep={[f'{t:.3f}' for t in times]}")

# ---- fast-path dispatch breakdown (device drained first) ----
torch.npu.synchronize()
time.sleep(0.1)
t0 = time.perf_counter_ns()
for _ in range(30):
    m.fla_chunk_o(*npu_in)
t1 = time.perf_counter_ns()
torch.npu.synchronize()
print(f"[dispatch] full fast wrapper: {(t1-t0)/30/1e3:.1f} us/call")

q, v, g, A, h = npu_in
entry = next(v for v in m._FAST.values() if v is not None)
msk = m._get_msk(q.device, 64)
torch.npu.synchronize()
t0 = time.perf_counter_ns()
for _ in range(50):
    entry.launch(q, v, g, A, h, msk)
t1 = time.perf_counter_ns()
torch.npu.synchronize()
print(f"[dispatch] entry.launch only: {(t1-t0)/50/1e3:.1f} us/call")

t0 = time.perf_counter_ns()
for _ in range(200):
    torch.npu.current_stream().npu_stream
t1 = time.perf_counter_ns()
print(f"[dispatch] stream query: {(t1-t0)/200/1e3:.1f} us/call")

t0 = time.perf_counter_ns()
for _ in range(100):
    torch.empty(entry.out_shape, dtype=entry.out_dtype, device=entry.dev)
t1 = time.perf_counter_ns()
print(f"[dispatch] output torch.empty: {(t1-t0)/100/1e3:.1f} us/call")
