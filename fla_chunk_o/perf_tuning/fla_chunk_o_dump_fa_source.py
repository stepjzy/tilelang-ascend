"""Dump flash_attn_bshd_developer (threads=2 CV-fusion precedent) kernel source
to inspect its cross-core flag pattern with 2 AIV sub-blocks."""
import sys

sys.path.insert(0, "/mnt/workspace/gitCode/skybasejzy/tilelang-ascend/examples/developer_mode")
import flash_attn_bshd_developer as fa  # noqa: E401  (runs the example incl. assert)

src = fa.func.get_kernel_source()
out = "/mnt/workspace/gitCode/skybasejzy/tilelang-ascend/examples_experiment/dsa_sparse_finetune/perf_tuning/kernel_source_fa_threads2.c"
with open(out, "w") as f:
    f.write(src)
print("written", out)
# show the AIC-side loop and flag usage
import re
for i, line in enumerate(src.splitlines(), 1):
    if "CrossCore" in line or "IS_AIC" in line or "IS_AIV" in line or "cid" in line and "GetBlockIdx" in line:
        print(f"{i}: {line.strip()}")
