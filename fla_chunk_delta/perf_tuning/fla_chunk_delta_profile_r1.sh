#!/bin/bash
# fla_chunk_delta Stage-3 Round-1 baseline profiling (msprof op x2 + app-level + source dump)
# Run from repo root: bash examples_experiment/dsa_sparse_finetune/perf_tuning/fla_chunk_delta_profile_r1.sh
set -x
source set_env.sh
cd examples_experiment/dsa_sparse_finetune/perf_tuning

echo "=== step 0: source dump (compiles both kernels) ==="
python fla_chunk_delta_bench_entry.py --dump-src --kernel fwd --warmup 1 --launches 0 2>&1 | tail -5

echo "=== step 1: msprof op fwd ==="
env -u ASCEND_RT_VISIBLE_DEVICES msprof op --warm-up=10 --launch-count=1 \
  --output=./fla_chunk_delta_prof_r1/op_fwd_baseline \
  --kernel-name=main_kernel \
  python fla_chunk_delta_bench_entry.py --kernel fwd 2>&1 | tail -25

echo "=== step 2: msprof op bwd ==="
env -u ASCEND_RT_VISIBLE_DEVICES msprof op --warm-up=10 --launch-count=1 \
  --output=./fla_chunk_delta_prof_r1/op_bwd_baseline \
  --kernel-name=main_kernel \
  python fla_chunk_delta_bench_entry.py --kernel bwd 2>&1 | tail -25

echo "=== step 3: app-level msprof (fwd+bwd) ==="
env -u ASCEND_RT_VISIBLE_DEVICES msprof \
  --output=./fla_chunk_delta_prof_r1/app_baseline \
  --application="python fla_chunk_delta_bench_entry.py --kernel all --warmup 12 --launches 3" \
  --task-time=on 2>&1 | tail -12

echo "=== ALL DONE, artifact listing ==="
find ./fla_chunk_delta_prof_r1 -maxdepth 3 -type d 2>/dev/null
ls -la
