"""Diagnostic round v4 for issue #1925: separate kernels per arm (no host-if
inside prim_func), all arms route through UB.

  A noflags_view   : write Scratch -> read aliased view ScratchV (NO flags)
  B flags_view     : same + explicit MTE3->MTE2 flags   (intended positive control)
  C noflags_direct : write Scratch -> read Scratch (same param, NO flags)
  D flags_direct   : same + flags
  E roundtrip      : In->UB->Scratch write; Scratch read->UB->Out (same-param)

Usage: python repro_v4.py <arm>
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


def show(tag, out, x):
    out = out.detach().cpu().float().flatten()
    ref = x.detach().cpu().float().flatten()
    err = (out - ref).abs().max().item()
    print(f"[{tag}] out[:6]={out[:6].tolist()}")
    print(f"[{tag}] x[:6]  ={ref[:6].tolist()}  max_abs_err={err:.4e}  "
          f"{'MATCH' if err < 1e-3 else 'MISMATCH'}")


def arm_view(with_flags):
    M = 64

    @tilelang.jit(out_idx=[3], pass_configs=PASS_CONFIGS)
    def kern(M):
        @T.prim_func
        def main(
            In: T.Tensor((M,), "float32"),
            Scratch: T.Tensor((1, M), "float32"),
            ScratchV: T.Tensor((1, M, 1), "float32"),
            Out: T.Tensor((M, 1), "float32"),
        ):
            with T.Kernel(1, threads=1, is_npu=True):
                in_ub = T.alloc_shared([M], "float32")
                acc = T.alloc_shared([M], "float32")
                v2d = T.alloc_shared([M, 1], "float32")
                T.copy(In, in_ub)
                T.copy(in_ub, acc)
                T.copy(acc, Scratch[0, 0:M])
                if with_flags:
                    T.set_flag("MTE3", "MTE2", 0)
                    T.wait_flag("MTE3", "MTE2", 0)
                T.copy(ScratchV[0, 0:M, 0:1], v2d)
                T.copy(v2d, Out)

        return main

    x = torch.randn(M, dtype=torch.float32).npu()
    scratch = torch.zeros(1, M, dtype=torch.float32).npu()
    out = kern(M)(x, scratch, scratch.view(1, M, 1))
    torch.npu.synchronize()
    show(f"view_{'flags' if with_flags else 'noflags'}", out, x)


def arm_direct(with_flags):
    M = 64

    @tilelang.jit(out_idx=[3], pass_configs=PASS_CONFIGS)
    def kern(M):
        @T.prim_func
        def main(
            In: T.Tensor((M,), "float32"),
            Scratch: T.Tensor((1, M), "float32"),
            Extra: T.Tensor((1, M), "float32"),
            Out: T.Tensor((M,), "float32"),
        ):
            with T.Kernel(1, threads=1, is_npu=True):
                in_ub = T.alloc_shared([M], "float32")
                acc = T.alloc_shared([M], "float32")
                v1d = T.alloc_shared([M], "float32")
                T.copy(In, in_ub)
                T.copy(in_ub, acc)
                T.copy(acc, Scratch[0, 0:M])
                if with_flags:
                    T.set_flag("MTE3", "MTE2", 0)
                    T.wait_flag("MTE3", "MTE2", 0)
                T.copy(Scratch[0, 0:M], v1d)
                T.copy(v1d, Out)

        return main

    x = torch.randn(M, dtype=torch.float32).npu()
    scratch = torch.zeros(1, M, dtype=torch.float32).npu()
    extra = torch.zeros(1, M, dtype=torch.float32).npu()
    out = kern(M)(x, scratch, extra)
    torch.npu.synchronize()
    show(f"direct_{'flags' if with_flags else 'noflags'}", out, x)


def arm_roundtrip():
    M = 64

    @tilelang.jit(out_idx=[2])
    def kern(M):
        @T.prim_func
        def main(
            In: T.Tensor((M,), "float32"),
            Scratch: T.Tensor((1, M), "float32"),
            Out: T.Tensor((M,), "float32"),
        ):
            with T.Kernel(1, threads=1, is_npu=True):
                in_ub = T.alloc_shared([M], "float32")
                back_ub = T.alloc_shared([M], "float32")
                T.copy(In, in_ub)
                T.copy(in_ub, Scratch[0, 0:M])
                T.copy(Scratch[0, 0:M], back_ub)
                T.copy(back_ub, Out)

        return main

    x = torch.randn(M, dtype=torch.float32).npu()
    scratch = torch.zeros(1, M, dtype=torch.float32).npu()
    out = kern(M)(x, scratch)
    torch.npu.synchronize()
    show("roundtrip_sameparam", out, x)


CASES = {
    "A_noflags_view": lambda: arm_view(False),
    "B_flags_view": lambda: arm_view(True),
    "C_noflags_direct": lambda: arm_direct(False),
    "D_flags_direct": lambda: arm_direct(True),
    "E_roundtrip": arm_roundtrip,
}

if __name__ == "__main__":
    tilelang.disable_cache()
    CASES[sys.argv[1]]()
