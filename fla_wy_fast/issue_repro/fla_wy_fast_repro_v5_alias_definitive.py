"""Definitive alias test (v5): same storage passed as TWO [1,M] params.

  write A[0,0:M]  ->  read B[0,0:M]   (direct 256B row copies, NO [1,M,1] view,
                                        NO sub-32B-row confound)
Arms (10 reps each):
  same_noflags : no explicit flags between write and read
  same_flags   : explicit MTE3->MTE2 flags

If noflags fails (stale) while flags passes 10/10 -> alias race cleanly proven.
If noflags passes 10/10 -> the posted issue evidence does not isolate a bug.

Usage: python repro_v5.py <arm>
"""

import sys

import tilelang
import tilelang.language as T
import torch

torch.manual_seed(0)

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def run(with_flags, reps=10):
    M = 64

    @tilelang.jit(out_idx=[3], pass_configs=PASS_CONFIGS)
    def kern(M):
        @T.prim_func
        def main(
            In: T.Tensor((M,), "float32"),
            A: T.Tensor((1, M), "float32"),
            B: T.Tensor((1, M), "float32"),
            Out: T.Tensor((M,), "float32"),
        ):
            with T.Kernel(1, threads=1, is_npu=True):
                in_ub = T.alloc_shared([M], "float32")
                acc = T.alloc_shared([M], "float32")
                v1d = T.alloc_shared([M], "float32")
                T.copy(In, in_ub)
                T.copy(in_ub, acc)
                T.copy(acc, A[0, 0:M])
                if with_flags:
                    T.set_flag("MTE3", "MTE2", 0)
                    T.wait_flag("MTE3", "MTE2", 0)
                T.copy(B[0, 0:M], v1d)
                T.copy(v1d, Out)

        return main

    tag = "same_flags" if with_flags else "same_noflags"
    fails = 0
    for r in range(reps):
        x = torch.randn(M, dtype=torch.float32).npu()
        scratch = torch.zeros(1, M, dtype=torch.float32).npu()
        # SAME storage passed as BOTH params (alias):
        out = kern(M)(x, scratch, scratch)
        torch.npu.synchronize()
        err = (out.cpu().float() - x.cpu()).abs().max().item()
        zero = bool((out.cpu().abs() < 1e-6).all())
        status = "MATCH" if err < 1e-3 else ("MISMATCH(zeros)" if zero else "MISMATCH(garbage)")
        if err >= 1e-3:
            fails += 1
        if r == 0 or err >= 1e-3:
            print(f"[{tag}] rep{r}: max_abs_err={err:.4e} {status}")
    print(f"[{tag}] SUMMARY: {reps - fails}/{reps} MATCH, fails={fails}")


if __name__ == "__main__":
    tilelang.disable_cache()
    run(sys.argv[1] == "same_flags")
