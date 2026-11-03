"""R2 same-session A/B: R1 final form (history snapshot) vs NK-merge candidate.

Both kernels JIT-compiled and benched in ONE process, same input tensors,
do_bench warmup=20/rep=10/mean/device-side (R1 protocol). Alternate A,B,B,A
order to cancel device drift.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402
import tilelang  # noqa: E402

tilelang.disable_cache()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r1 = load_module(
    os.path.join(HERE, "..", "history_version", "fla_chunk_inter_perf_r2_iter1_before.py"),
    "fla_inter_r1",
)
cand = load_module(os.path.join(HERE, "..", "fla_chunk_inter.py"), "fla_inter_cand")

from test_fla_chunk_inter import make_inputs, do_bench_npu  # noqa: E402

B, S, H, DK, DV = 1, 8192, 64, 128, 128
q, k, v, g, h, dv, do, dh = make_inputs(B, S, H, DK, DV, 64, seed=0)
dev_inputs = [t.npu() for t in (q, k, v, g, h, dv, do, dh)]


def call_r1():
    r1.fla_chunk_inter_bwd(*dev_inputs)


def call_cand():
    cand.fla_chunk_inter_bwd(*dev_inputs)


# JIT compile both (untimed)
call_r1()
call_cand()

results = {}
for name, fn in (("A_r1", call_r1), ("B_cand", call_cand), ("B_cand", call_cand), ("A_r1", call_r1)):
    t = do_bench_npu(fn)
    results.setdefault(name, []).append(t * 1000.0)

for name, ts in results.items():
    print(f"[AB] {name}: " + " / ".join(f"{t:.1f}us" for t in ts) + f"  mean={sum(ts)/len(ts):.1f}us")
