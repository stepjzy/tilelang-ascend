"""Dump translated Ascend C source for fla_chunk_inter (bench shape) for
Stage 3 op/sync-station counting. Read-only wrt the operator under test."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tilelang

from fla_chunk_inter import PASS_CONFIGS, fla_chunk_inter  # noqa: E402

tilelang.disable_cache()

B, S, H, DK, DV = 1, 8192, 64, 128, 128
kernel = fla_chunk_inter(B=B, S=S, H=H, DK=DK, DV=DV, scale=1.0, chunk_size=64)
src = kernel.get_kernel_source()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fla_chunk_inter_kernel_source_iter0.c")
with open(out, "w") as f:
    f.write(src)

aic = src.count("IS_ASCEND_AIC")
aiv = src.count("IS_ASCEND_AIV")
print(f"source lines: {len(src.splitlines())}")
print(f"IS_ASCEND_AIC occurrences: {aic}")
print(f"IS_ASCEND_AIV occurrences: {aiv}")
for kw in ("SetFlag", "WaitFlag", "set_flag", "wait_flag", "PipeBarrier", "DataCopy", "Mmad", "workspace"):
    print(f"{kw}: {src.count(kw)}")
