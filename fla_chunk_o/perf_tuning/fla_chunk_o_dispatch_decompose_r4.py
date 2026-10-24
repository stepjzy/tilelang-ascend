"""R4 main line 1, step 1: decompose the per-call host dispatch cost.

Path under test (public): fla_chunk_o(*npu_in)
  -> JIT cache lookup (fla_chunk_o_kernel(*spec))
  -> JITKernel.__call__ / _generate_extra_args
  -> lambda_forward -> CythonKernelWrapper.forward
       - torch.npu.current_stream().npu_stream
       - tvm.arith.Analyzer()  <-- rebuilt EVERY call
       - 5x torch.empty (1 out + 4 relay workspaces)
       - shape resolve + ctypes arg build + device/dtype checks
       - lib.call (rtGetC2cCtrlAddr + <<<20>>> enqueue)

All timings are host wall-clock (perf_counter_ns), device NOT synced inside
the timed loop (kernel launch is async; back-to-back calls measure pure
host dispatch as long as no queue backpressure kicks in -- verified by
also timing with an idle device after full drain).
"""
import os
import sys
import time

import tilelang
import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fla_chunk_o import fla_chunk_o, fla_chunk_o_kernel  # noqa: E402
from test_fla_chunk_o import make_inputs  # noqa: E402

tilelang.disable_cache()

B, S, H, DK, DV = 1, 8192, 64, 128, 128
npu_in, _ = make_inputs(B, S, H, DK, DV, 64, 42)
q, v, g, A, h = npu_in

# ---- warm: compile + a few calls, then drain ----
o = fla_chunk_o(*npu_in)
torch.npu.synchronize()
for _ in range(5):
    fla_chunk_o(*npu_in)
torch.npu.synchronize()


def timeit(fn, n=30, drain=True):
    """Mean per-call host wall time (us); optionally drain after each batch."""
    total = 0
    for _ in range(3):  # micro warmup
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(n):
        fn()
    t1 = time.perf_counter_ns()
    if drain:
        torch.npu.synchronize()
    return (t1 - t0) / n / 1e3  # us


# ---- [1] full public-path dispatch (no sync inside) ----
full_us = timeit(lambda: fla_chunk_o(*npu_in))
print(f"[1] full fla_chunk_o dispatch (async, no sync): {full_us:8.1f} us/call")

# ---- [2] JIT decorator lookup only ----
kernel = fla_chunk_o_kernel(B, S, H, DK, DV, 64, None, None)
jit_us = timeit(lambda: fla_chunk_o_kernel(B, S, H, DK, DV, 64, None, None))
print(f"[2] JIT cache lookup fla_chunk_o_kernel(...):    {jit_us:8.1f} us/call")

# ---- [3] JITKernel.__call__ incl. cython forward ----
kc_us = timeit(lambda: kernel(q.view(B, S, H * DK), v.view(B, S, H * DV),
                              g.view(B, S, H * DK), A.view(B, S, H * 64), h,
                              fla_chunk_o.__globals__["_get_msk"](q.device, 64)))
print(f"[3] JITKernel __call__ (cython forward):        {kc_us:8.1f} us/call")

# ---- [4] individual ingredients ----
from tvm.arith import Analyzer  # noqa: E402

an_us = timeit(lambda: Analyzer(), n=50, drain=False)
print(f"[4a] tvm.arith.Analyzer() ctor:                 {an_us:8.1f} us/call")

stream_us = timeit(lambda: torch.npu.current_stream().npu_stream, n=200, drain=False)
print(f"[4b] torch.npu.current_stream().npu_stream:     {stream_us:8.1f} us/call")


def alloc5():
    dev = q.device
    o_ = torch.empty(B, S, H * DV, dtype=torch.bfloat16, device=dev)
    w1 = torch.empty(20, 9, 64, DK, dtype=torch.bfloat16, device=dev)
    w2 = torch.empty(20, 9, 64, DK, dtype=torch.bfloat16, device=dev)
    w3 = torch.empty(20, 9, 64, 64, dtype=torch.bfloat16, device=dev)
    w4 = torch.empty(20, 9, 64, DV, dtype=torch.bfloat16, device=dev)
    return o_, w1, w2, w3, w4


alloc_us = timeit(alloc5, n=50, drain=False)
print(f"[4c] 5x torch.empty (out + 4 ws):               {alloc_us:8.1f} us/call")

import ctypes  # noqa: E402

views = (q.view(B, S, H * DK), v.view(B, S, H * DV), g.view(B, S, H * DK),
         A.view(B, S, H * 64), h, fla_chunk_o.__globals__["_get_msk"](q.device, 64))


def build_args(ws):
    stream = torch.npu.current_stream().npu_stream
    args = [ctypes.c_void_p(t.data_ptr()) for t in views]
    args += [ctypes.c_void_p(w.data_ptr()) for w in ws[:4]]
    args.append(ctypes.c_void_p(views[0].data_ptr()))  # placeholder for out ptr
    args.append(ctypes.c_void_p(stream))
    return args


o_, w1, w2, w3, w4 = alloc5()
ctypes_us = timeit(lambda: build_args((o_, w1, w2, w3, w4)), n=200, drain=False)
print(f"[4d] ctypes arg build (12 c_void_p):            {ctypes_us:8.1f} us/call")

# ---- [5] lib.call alone (enqueue cost), pointers prebuilt ----
adapter = kernel.adapter
lib = adapter.lib
call_args = build_args((o_, w1, w2, w3, w4))
call_args[10] = ctypes.c_void_p(o_.data_ptr())  # real out ptr
libcall_us = timeit(lambda: lib.call(*call_args))
print(f"[5] lib.call enqueue (prebuilt args):           {libcall_us:8.1f} us/call")

# ---- [6] stream query + empty + ctypes + lib.call (fast-path body cost) ----


def fast_body():
    oo = torch.empty(B, S, H * DV, dtype=torch.bfloat16, device=q.device)
    args = [ctypes.c_void_p(t.data_ptr()) for t in views]
    args += [ctypes.c_void_p(w.data_ptr()) for w in (w1, w2, w3, w4)]
    args.append(ctypes.c_void_p(oo.data_ptr()))
    args.append(ctypes.c_void_p(torch.npu.current_stream().npu_stream))
    lib.call(*args)
    return oo


fast_us = timeit(fast_body)
print(f"[6] projected fast-path body (fresh out):       {fast_us:8.1f} us/call")

# ---- [7] attribution summary ----
print("\n--- attribution (public-path dispatch) ---")
print(f"  JIT lookup        : {jit_us:8.1f} us")
print(f"  cython forward    : {kc_us - jit_us:8.1f} us  (Analyzer ~{an_us:.1f},"
      f" alloc5 ~{alloc_us:.1f}, stream ~{stream_us:.1f},"
      f" rest ~{kc_us - jit_us - an_us - alloc_us - stream_us:.1f})")
print(f"  full wrapper      : {full_us:8.1f} us")
print(f"  projected fast    : {fast_us + jit_us:8.1f} us (JIT lookup + fast body)")
