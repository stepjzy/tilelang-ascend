#!/bin/bash
# fla_chunk_delta Stage-3 round-2 TERMINAL verification (R13, iter10-adopted state):
# full suite + bench x2 (session variance) + dual-mode msprof + source dump
set -x
source set_env.sh
cd examples_experiment/dsa_sparse_finetune

echo "=== FULL SUITE (R13) ==="
python test_fla_chunk_delta.py --level all --bench 2>&1 | grep -E "PRECISION|BOUNDARY|PERF" | tail -45

echo "=== BENCH REPEAT (R13, variance check) ==="
python test_fla_chunk_delta.py --bench 2>&1 | grep -E "PERF" | tail -3

cd perf_tuning
echo "=== msprof op fwd MAIN (R13, true BV=128) ==="
env -u ASCEND_RT_VISIBLE_DEVICES msprof op --warm-up=10 --launch-count=1 \
  --output=./fla_chunk_delta_prof_r13/op_fwd_main \
  --kernel-name=main_kernel \
  python fla_chunk_delta_bench_entry.py --kernel fwd-main 2>&1 | grep -E "Task Duration|Block Dim" | head -4

echo "=== msprof op bwd MAIN (R13) ==="
env -u ASCEND_RT_VISIBLE_DEVICES msprof op --warm-up=10 --launch-count=1 \
  --output=./fla_chunk_delta_prof_r13/op_bwd_main \
  --kernel-name=main_kernel \
  python fla_chunk_delta_bench_entry.py --kernel bwd-main 2>&1 | grep -E "Task Duration|Block Dim" | head -4

echo "=== msprof op bwd WRAPPER (captures prep) ==="
env -u ASCEND_RT_VISIBLE_DEVICES msprof op --warm-up=10 --launch-count=1 \
  --output=./fla_chunk_delta_prof_r13/op_bwd_wrapper \
  --kernel-name=main_kernel \
  python fla_chunk_delta_bench_entry.py --kernel bwd 2>&1 | grep -E "Task Duration|Block Dim" | head -4

echo "=== app-level msprof (R13, Insight) ==="
env -u ASCEND_RT_VISIBLE_DEVICES msprof \
  --output=./fla_chunk_delta_prof_r13/app_r13 \
  --application="python fla_chunk_delta_bench_entry.py --kernel all --warmup 12 --launches 3" \
  --task-time=on 2>&1 | tail -6

echo "=== source dump (R13 terminal) ==="
python fla_chunk_delta_bench_entry.py --dump-src --kernel all --warmup 0 --launches 0 2>&1 | grep -E "entry" | tail -3

echo "=== ALL DONE ==="
find ./fla_chunk_delta_prof_r13 -maxdepth 2 -type d 2>/dev/null
