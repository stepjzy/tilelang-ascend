"""R3 session baseline anchor: 3x do_bench of the R2-final state (r3_start)."""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402, F401
import tilelang  # noqa: E402

tilelang.disable_cache()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load_module(
    os.path.join(HERE, "..", "history_version", "fla_chunk_inter_perf_r3_start.py"),
    "fla_inter_r3_start",
)

from test_fla_chunk_inter import make_inputs, do_bench_npu  # noqa: E402

B, S, H, DK, DV = 1, 8192, 64, 128, 128
q, k, v, g, h, dv, do, dh = make_inputs(B, S, H, DK, DV, 64, seed=0)
dev_inputs = [t.npu() for t in (q, k, v, g, h, dv, do, dh)]

m.fla_chunk_inter_bwd(*dev_inputs)  # warm/JIT outside timing

ts = []
for _ in range(3):
    ts.append(do_bench_npu(lambda: m.fla_chunk_inter_bwd(*dev_inputs)) * 1000.0)
print("[R3-BASE] " + " / ".join(f"{t:.1f}us" for t in ts) + f"  mean={sum(ts) / len(ts):.1f}us")
