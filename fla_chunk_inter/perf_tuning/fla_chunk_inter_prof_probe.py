"""Profiling probe for fla_chunk_inter (bench shape), shared by msprof op
(OPPROF) and application msprof (PROF): 12 kernel launches in one process
(1st = JIT compile + launch, 2-11 = msprof op warm-up window, 12th+ = captured)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import tilelang

from fla_chunk_inter import fla_chunk_inter_bwd  # noqa: E402
from test_fla_chunk_inter import make_inputs  # noqa: E402

tilelang.disable_cache()

B, S, H, DK, DV = 1, 8192, 64, 128, 128
q, k, v, g, h, dv, do, dh = make_inputs(B, S, H, DK, DV, 64, seed=0)
dev_inputs = [t.npu() for t in (q, k, v, g, h, dv, do, dh)]
for _ in range(12):
    fla_chunk_inter_bwd(*dev_inputs)
torch.npu.synchronize()
print("OP_PROBE_DONE")
