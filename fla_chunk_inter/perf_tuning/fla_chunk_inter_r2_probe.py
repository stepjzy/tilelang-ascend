"""R2 profiling probe for fla_chunk_inter (bench shape), shared by msprof op
(OPPROF, CSV scope) and application msprof (PROF, Insight scope): 12 kernel
launches in one process (1st = JIT compile + launch, 2-11 = msprof op warm-up
window, 12th = captured).

Round-2 rewrite of the round-1 probe (old probe imported `make_inputs`, the
round-2 test exposes `prepare_input`)."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # op root hosts fla_chunk_inter.py
import torch  # noqa: E402
import tilelang  # noqa: E402

from fla_chunk_inter import fla_chunk_inter_bwd  # noqa: E402
from test_fla_chunk_inter import prepare_input  # noqa: E402

tilelang.disable_cache()

B, S, H, DK, DV = 1, 8192, 64, 128, 128
torch.manual_seed(42)  # align bench_inter
q, k, v, g, h, dv, do, dh = prepare_input(B, S, H, DK, DV)
dev_inputs = [t.npu() for t in (q, k, v, g, h, dv, do, dh)]
for _ in range(12):
    fla_chunk_inter_bwd(*dev_inputs)
torch.npu.synchronize()
print("OP_PROBE_DONE")
