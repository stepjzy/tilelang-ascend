#!/bin/bash
# fla_chunk_delta Stage-3 FINAL verification: full suite + bench + final dual-mode profiling
set -x
source set_env.sh
cd examples_experiment/dsa_sparse_finetune

echo "=== FULL SUITE + BENCH (final) ==="
python test_fla_chunk_delta.py --level all --bench 2>&1 | grep -E "PRECISION|BOUNDARY|PERF|Test" | tail -40

cd perf_tuning
echo "=== msprof op fwd MAIN (final) ==="
env -u ASCEND_RT_VISIBLE_DEVICES msprof op --warm-up=10 --launch-count=1 \
  --output=./fla_chunk_delta_prof_final/op_fwd_main_final \
  --kernel-name=main_kernel \
  python fla_chunk_delta_bench_entry.py --kernel fwd-main 2>&1 | grep -E "Task Duration|Block Dim" | head -4

echo "=== msprof op bwd MAIN (final) ==="
env -u ASCEND_RT_VISIBLE_DEVICES msprof op --warm-up=10 --launch-count=1 \
  --output=./fla_chunk_delta_prof_final/op_bwd_main_final \
  --kernel-name=main_kernel \
  python fla_chunk_delta_bench_entry.py --kernel bwd-main 2>&1 | grep -E "Task Duration|Block Dim" | head -4

echo "=== msprof op fwd WRAPPER (captures prep) ==="
env -u ASCEND_RT_VISIBLE_DEVICES msprof op --warm-up=10 --launch-count=1 \
  --output=./fla_chunk_delta_prof_final/op_fwd_prep \
  --kernel-name=main_kernel \
  python fla_chunk_delta_bench_entry.py --kernel fwd 2>&1 | grep -E "Task Duration|Block Dim" | head -4

echo "=== msprof op bwd WRAPPER (captures prep) ==="
env -u ASCEND_RT_VISIBLE_DEVICES msprof op --warm-up=10 --launch-count=1 \
  --output=./fla_chunk_delta_prof_final/op_bwd_prep \
  --kernel-name=main_kernel \
  python fla_chunk_delta_bench_entry.py --kernel bwd 2>&1 | grep -E "Task Duration|Block Dim" | head -4

echo "=== app-level msprof (final, Insight) ==="
env -u ASCEND_RT_VISIBLE_DEVICES msprof \
  --output=./fla_chunk_delta_prof_final/app_final \
  --application="python fla_chunk_delta_bench_entry.py --kernel all --warmup 12 --launches 3" \
  --task-time=on 2>&1 | tail -6

echo "=== source dump (final) ==="
python fla_chunk_delta_bench_entry.py --dump-src --kernel fwd --warmup 1 --launches 0 2>&1 | tail -3

echo "=== ALL DONE ==="
find ./fla_chunk_delta_prof_final -maxdepth 2 -type d 2>/dev/null
