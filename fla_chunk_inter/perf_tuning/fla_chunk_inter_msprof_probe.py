"""Single-launch msprof probe for fla_chunk_inter (bench shape, one call).

Used with: msprof op --kernel-name=main_kernel --output=<dir> python <this>.
Keep the runtime short: one warmup (JIT compile) + one profiled call.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tilelang

from fla_chunk_inter import fla_chunk_inter_bwd  # noqa: E402
from test_fla_chunk_inter import make_inputs  # noqa: E402

tilelang.disable_cache()

B, S, H, DK, DV = 1, 8192, 64, 128, 128
q, k, v, g, h, dv, do, dh = make_inputs(B, S, H, DK, DV, 64, seed=0)
dev_inputs = [t.npu() for t in (q, k, v, g, h, dv, do, dh)]
for _ in range(2):  # first = JIT compile, second = profiled launch
    fla_chunk_inter_bwd(*dev_inputs)
torch_npu_sync = __import__("torch").npu.synchronize()
