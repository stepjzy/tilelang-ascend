"""msprof run target: single bench-shape invocation of fla_chunk_o (the
operator under test). Used with: msprof op --kernel-name=main_kernel
--output=./msprof_output_r2 python <this file>"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fla_chunk_o import fla_chunk_o  # noqa: E402
from test_fla_chunk_o import make_inputs  # noqa: E402

B, S, H, DK, DV = 1, 8192, 64, 128, 128
npu_in, _ = make_inputs(B, S, H, DK, DV, 64, 42)
o = fla_chunk_o(*npu_in)
torch.npu.synchronize()
for _ in range(5):
    fla_chunk_o(*npu_in)
torch.npu.synchronize()
print("done", o.shape)
