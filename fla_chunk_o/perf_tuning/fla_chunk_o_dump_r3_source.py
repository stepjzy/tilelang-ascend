"""Dump the generated Ascend C source of the current fla_chunk_o kernel (R3)."""
import os
import sys

import tilelang
import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fla_chunk_o import fla_chunk_o_kernel  # noqa: E402

tilelang.disable_cache()

B, S, H, DK, DV = 1, 8192, 64, 128, 128
func = fla_chunk_o_kernel(B, S, H, DK, DV, 64, None, None, None)
src = func.get_kernel_source()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fla_chunk_o_kernel_source_r3_start.c")
with open(out, "w") as f:
    f.write(src)
print(f"written: {out} ({len(src)} chars)")

# quick stats
import re
sets = re.findall(r"CrossCoreSetFlag<[^>]*>\((0x[0-9a-f]+|\d+)\)", src)
waits = re.findall(r"CrossCoreWaitFlag\((0x[0-9a-f]+|\d+)\)", src)
print(f"CrossCoreSetFlag sites: {len(sets)}, ids: {sorted(set(sets))}")
print(f"CrossCoreWaitFlag sites: {len(waits)}, ids: {sorted(set(waits))}")
