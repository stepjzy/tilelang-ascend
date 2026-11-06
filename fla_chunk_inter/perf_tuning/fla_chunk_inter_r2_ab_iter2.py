"""R2 iter2 A/B: iter1 adopted state vs iter2 candidate (do load+split+relay
hoisted ahead of the right chains; v/dv batched loads stay behind them).
Same process, same inputs (seed 42, bench shape), interleaved do_bench x3.

Usage: python perf_tuning/fla_chunk_inter_r2_ab_iter2.py
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, _ROOT)

import torch  # noqa: E402
import tilelang  # noqa: E402

tilelang.disable_cache()

from test_fla_chunk_inter import do_bench, prepare_input  # noqa: E402


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


a = load_module(os.path.join(_ROOT, "fla_chunk_inter.py"), "fla_inter_iter1_adopted")
b = load_module(
    os.path.join(_HERE, "fla_chunk_inter_r2_iter2_candidate.py"),
    "fla_inter_iter2_do_first",
)

torch.manual_seed(42)
ins = prepare_input(1, 8192, 64, 128, 128)
dev = [t.npu() for t in ins]

results = []
for rnd in range(3):
    ms_a = do_bench(lambda: a.fla_chunk_inter_bwd(*dev))
    ms_b = do_bench(lambda: b.fla_chunk_inter_bwd(*dev))
    delta = (ms_b - ms_a) / ms_a * 100
    results.append((ms_a, ms_b, delta))
    print(
        f"[AB] round{rnd} A(iter1)={ms_a * 1000:.1f}us "
        f"B(iter2_do_first)={ms_b * 1000:.1f}us delta={delta:+.2f}%"
    )

a_mean = sum(r[0] for r in results) / len(results)
b_mean = sum(r[1] for r in results) / len(results)
print(
    f"[AB-SUMMARY] A_mean={a_mean * 1000:.1f}us B_mean={b_mean * 1000:.1f}us "
    f"delta_mean={(b_mean - a_mean) / a_mean * 100:+.2f}% "
    f"rounds_positive={[f'{r[2]:+.2f}%' for r in results]}"
)
