"""Dump the generated Ascend C source of the fla_chunk_o kernel (skill Step 2:
operator-type judgment + bottleneck evidence). Read-only probe."""
import os
import sys

import tilelang
import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fla_chunk_o import fla_chunk_o_kernel, _pick_block_dk, _pick_block_dv  # noqa: E402

B, S, H, DK, DV = 1, 8192, 64, 128, 128
func = fla_chunk_o_kernel(B, S, H, DK, DV, 64, DK**-0.5,
                          _pick_block_dk(DK), _pick_block_dv(DV))
src = func.get_kernel_source()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_source_iter1.c")
with open(out, "w") as f:
    f.write(src)
print(f"written {out}, {len(src)} bytes, {src.count(chr(10))} lines")
# quick type judgment
print("ASCEND_IS_AIC" in src or "IS_AIC" in src, "cube markers")
print("IS_AIV" in src, "vector markers")
