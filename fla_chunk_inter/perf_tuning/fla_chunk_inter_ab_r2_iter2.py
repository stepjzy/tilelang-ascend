"""R2 same-session A/B: iter1 adopted (NK-merge, 0-late flag) vs iter2
candidate (NK-merge + 1-late flag + head-set)."""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402
import tilelang  # noqa: E402

tilelang.disable_cache()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


a = load_module(
    os.path.join(HERE, "..", "history_version", "fla_chunk_inter_perf_r2_iter2_before.py"),
    "fla_inter_iter1",
)
b = load_module(os.path.join(HERE, "..", "fla_chunk_inter.py"), "fla_inter_iter2")

from test_fla_chunk_inter import make_inputs, do_bench_npu  # noqa: E402

B, S, H, DK, DV = 1, 8192, 64, 128, 128
q, k, v, g, h, dv, do, dh = make_inputs(B, S, H, DK, DV, 64, seed=0)
dev_inputs = [t.npu() for t in (q, k, v, g, h, dv, do, dh)]

a.fla_chunk_inter_bwd(*dev_inputs)
b.fla_chunk_inter_bwd(*dev_inputs)

results = {}
for name, fn in (
    ("A_iter1", lambda: a.fla_chunk_inter_bwd(*dev_inputs)),
    ("B_iter2", lambda: b.fla_chunk_inter_bwd(*dev_inputs)),
    ("B_iter2", lambda: b.fla_chunk_inter_bwd(*dev_inputs)),
    ("A_iter1", lambda: a.fla_chunk_inter_bwd(*dev_inputs)),
):
    t = do_bench_npu(fn)
    results.setdefault(name, []).append(t * 1000.0)

for name, ts in results.items():
    print(f"[AB] {name}: " + " / ".join(f"{t:.1f}us" for t in ts) + f"  mean={sum(ts)/len(ts):.1f}us")
