"""msprof target for fla_chunk_o (R3): run the bench shape a few times."""
import os
import sys

import tilelang
import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fla_chunk_o import fla_chunk_o  # noqa: E402
from test_fla_chunk_o import make_inputs  # noqa: E402

tilelang.disable_cache()

B, S, H, DK, DV = 1, 8192, 64, 128, 128
npu_in, _cpu = make_inputs(B, S, H, DK, DV, 64, 42)
o = fla_chunk_o(*npu_in)
torch.npu.synchronize()
for _ in range(6):
    fla_chunk_o(*npu_in)
torch.npu.synchronize()
print("done")
