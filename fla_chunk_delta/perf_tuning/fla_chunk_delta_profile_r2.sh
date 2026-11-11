#!/bin/bash
# fla_chunk_delta Stage-3 Round-2 candidate: bench (--bench sentinel) + msprof op x2 + source dump
set -x
source set_env.sh
cd examples_experiment/dsa_sparse_finetune

echo "=== step 0: bench sentinel (L0 + do_bench, same口径 as Stage-2 baseline) ==="
python test_fla_chunk_delta.py --level l0 --bench 2>&1 | grep -E "PRECISION|PERF|Test" | tail -12

cd perf_tuning
echo "=== step 1: msprof op fwd (candidate) ==="
env -u ASCEND_RT_VISIBLE_DEVICES msprof op --warm-up=10 --launch-count=1 \
  --output=./fla_chunk_delta_prof_r2/op_fwd_iter2 \
  --kernel-name=main_kernel \
  python fla_chunk_delta_bench_entry.py --kernel fwd 2>&1 | grep -E "Task Duration|Block Dim|Op Name|utilization|INFO.*save" | head -14

echo "=== step 2: msprof op bwd (candidate) ==="
env -u ASCEND_RT_VISIBLE_DEVICES msprof op --warm-up=10 --launch-count=1 \
  --output=./fla_chunk_delta_prof_r2/op_bwd_iter2 \
  --kernel-name=main_kernel \
  python fla_chunk_delta_bench_entry.py --kernel bwd 2>&1 | grep -E "Task Duration|Block Dim|Op Name|utilization|INFO.*save" | head -14

echo "=== step 3: source dump (candidate; verify scalar loops gone) ==="
python fla_chunk_delta_bench_entry.py --dump-src --kernel fwd --warmup 1 --launches 0 2>&1 | tail -3
echo "=== ALL DONE ==="
find ./fla_chunk_delta_prof_r2 -maxdepth 2 -type d 2>/dev/null
