#!/bin/bash
# fla_chunk_delta Stage-3 Round-3 (NSLOT=16 ring): bench + msprof op main x2
set -x
source set_env.sh
cd examples_experiment/dsa_sparse_finetune

echo "=== bench sentinel ==="
python test_fla_chunk_delta.py --level l0 --bench 2>&1 | grep -E "PERF|Test" | tail -5

cd perf_tuning
echo "=== msprof op fwd MAIN ==="
env -u ASCEND_RT_VISIBLE_DEVICES msprof op --warm-up=10 --launch-count=1 \
  --output=./fla_chunk_delta_prof_r5/op_fwd_main_iter5 \
  --kernel-name=main_kernel \
  python fla_chunk_delta_bench_entry.py --kernel fwd-main 2>&1 | grep -E "Task Duration|Block Dim" | head -4

echo "=== msprof op bwd MAIN ==="
env -u ASCEND_RT_VISIBLE_DEVICES msprof op --warm-up=10 --launch-count=1 \
  --output=./fla_chunk_delta_prof_r5/op_bwd_main_iter5 \
  --kernel-name=main_kernel \
  python fla_chunk_delta_bench_entry.py --kernel bwd-main 2>&1 | grep -E "Task Duration|Block Dim" | head -4

echo "=== ALL DONE ==="
find ./fla_chunk_delta_prof_r5 -maxdepth 2 -type d 2>/dev/null
