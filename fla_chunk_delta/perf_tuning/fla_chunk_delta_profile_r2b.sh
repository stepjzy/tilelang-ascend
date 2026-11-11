#!/bin/bash
# fla_chunk_delta Stage-3 Round-2b: msprof op on the MAIN recurrence kernels (main-only entry)
set -x
source set_env.sh
cd examples_experiment/dsa_sparse_finetune/perf_tuning

echo "=== msprof op fwd MAIN kernel ==="
env -u ASCEND_RT_VISIBLE_DEVICES msprof op --warm-up=10 --launch-count=1 \
  --output=./fla_chunk_delta_prof_r2/op_fwd_main_iter2 \
  --kernel-name=main_kernel \
  python fla_chunk_delta_bench_entry.py --kernel fwd-main 2>&1 | grep -E "Task Duration|Block Dim|Op Name|Op Type|utilization lower" | head -14

echo "=== msprof op bwd MAIN kernel ==="
env -u ASCEND_RT_VISIBLE_DEVICES msprof op --warm-up=10 --launch-count=1 \
  --output=./fla_chunk_delta_prof_r2/op_bwd_main_iter2 \
  --kernel-name=main_kernel \
  python fla_chunk_delta_bench_entry.py --kernel bwd-main 2>&1 | grep -E "Task Duration|Block Dim|Op Name|Op Type|utilization lower" | head -14

echo "=== ALL DONE ==="
find ./fla_chunk_delta_prof_r2 -maxdepth 2 -type d 2>/dev/null
