"""Verification re-run of submitted issue reproducers (v2, 2026-11-12).

Fixes/controls over v1 (the #1924 lesson):
- Every comparison asserts out.shape == ref.shape FIRST (broadcast-trap guard).
- reduce (case 'reduce_*'): v1's reference x.sum(0) was WRONG ((M,) vs (M,N)
  broadcast artifact). v2 uses x.sum(dim=(0, 2)) and tests serial + pipelined.
- gm_alias (case 'alias_noflags' / 'alias_flags'): adds an all-zeros diagnosis
  and a WITH-flags control arm to isolate the missing-dependency cause.
- multihop: progress prints + external kill for the hang arm.

Usage: python repro_v2.py <case>
Cases: fp32_gemm, ext1, ext3, reduce_serial, reduce_pipelined,
       alias_noflags, alias_flags, multihop
"""

import sys
import time

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


def report(name, out, ref, atol=1e-3, rtol=1e-2):
    out = out.detach().cpu().float()
    ref = ref.detach().cpu().float()
    assert out.shape == ref.shape, (
        f"[{name}] SHAPE TRAP: out {tuple(out.shape)} vs ref {tuple(ref.shape)} — refusing to compare"
    )
    err = (out - ref).abs()
    ok = bool(torch.allclose(out, ref, atol=atol, rtol=rtol))
    print(f"[{name}] shape={tuple(out.shape)} max_abs_err={err.max().item():.4e}  "
          f"{'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        bad = err > atol + rtol * ref.abs()
        print(f"[{name}] mismatches={int(bad.sum())}/{bad.numel()} "
              f"first: out={out[bad][:2].tolist()} ref={ref[bad][:2].tolist()}")
    return ok


# ------------------------------------------------------------- fp32 gemm (#1921)
def case_fp32_gemm():
    M = N = K = 64

    @tilelang.jit(out_idx=[2])
    def kern():
        @T.prim_func
        def main(
            A: T.Tensor((M, K), "float32"),
            B: T.Tensor((K, N), "float32"),
            C: T.Tensor((M, N), "float32"),
        ):
            with T.Kernel(1, threads=1, is_npu=True):
                a_l1 = T.alloc_shared([M, K], "float32")
                b_l1 = T.alloc_shared([K, N], "float32")
                c_frag = T.alloc_fragment([M, N], "float32")
                c_ub = T.alloc_shared([M, N], "float32")
                T.copy(A, a_l1)
                T.copy(B, b_l1)
                T.gemm_v0(a_l1, b_l1, c_frag, init=True)
                T.copy(c_frag, c_ub)
                T.copy(c_ub, C)

        return main

    a = torch.randn(M, K, dtype=torch.float32).npu()
    b = torch.randn(K, N, dtype=torch.float32).npu()
    c = kern()(a, b)
    torch.npu.synchronize()
    report("fp32_gemm", c, a.cpu() @ b.cpu(), atol=1e-2, rtol=1e-2)


# ------------------------------------------------------------- extents (#1922)
def _ext_kern(M, D, bD):
    @tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS)
    def kern(M, D, bD):
        @T.prim_func
        def main(
            A: T.Tensor((M, M), "bfloat16"),
            B: T.Tensor((M, D), "bfloat16"),
            C: T.Tensor((M, D), "bfloat16"),
        ):
            with T.Kernel(1, threads=1, is_npu=True):
                a_l1 = T.alloc_shared([M, M], "bfloat16")
                b_l1 = T.alloc_shared([M, bD], "bfloat16")
                c_frag = T.alloc_fragment([M, bD], "float32")
                c_ub = T.alloc_shared([M, bD], "bfloat16")
                T.copy(A, a_l1)
                for i in T.Pipelined(D // bD, num_stages=2):
                    T.copy(B[0:M, i * bD : (i + 1) * bD], b_l1)
                    T.gemm_v0(a_l1, b_l1, c_frag, init=True)
                    T.copy(c_frag, c_ub)
                    T.copy(c_ub, C[0:M, i * bD : (i + 1) * bD])

        return main

    return kern(M, D, bD)


def case_ext1():
    M, D, bD = 64, 32, 32
    a = torch.randn(M, M).bfloat16().npu()
    b = torch.randn(M, D).bfloat16().npu()
    c = _ext_kern(M, D, bD)(a, b)
    torch.npu.synchronize()
    report("ext1", c, (a.cpu().float() @ b.cpu().float()).bfloat16(), atol=1e-1, rtol=1e-1)


def case_ext3():
    M, D, bD = 64, 96, 32
    a = torch.randn(M, M).bfloat16().npu()
    b = torch.randn(M, D).bfloat16().npu()
    c = _ext_kern(M, D, bD)(a, b)
    torch.npu.synchronize()
    ref = (a.cpu().float() @ b.cpu().float()).bfloat16()
    ok = report("ext3_all", c, ref, atol=1e-1, rtol=1e-1)
    if not ok:
        for s in range(D // bD):
            sl = slice(s * bD, (s + 1) * bD)
            report(f"ext3_slice{s}", c[:, sl], ref[:, sl], atol=1e-1, rtol=1e-1)


# --------------------------------------------- reduce, corrected ref (#1924 re-check)
def _reduce_kern(S_, M, N, loop_kind):
    @tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
    def kern(S_, M, N):
        @T.prim_func
        def main(
            X: T.Tensor((S_, M, N), "float32"),
            out: T.Tensor((M,), "float32"),
        ):
            with T.Kernel(1, threads=1, is_npu=True):
                x_ub = T.alloc_shared([M, N], "float32")
                acc = T.alloc_shared([M], "float32")
                T.tile.fill(acc, 0.0)
                if loop_kind == "pipelined":
                    for i in T.Pipelined(S_, num_stages=2):
                        T.copy(X[i, 0:M, 0:N], x_ub)
                        T.reduce_sum(x_ub, acc, dim=1, clear=False)
                else:
                    for i in T.serial(S_):
                        T.copy(X[i, 0:M, 0:N], x_ub)
                        T.reduce_sum(x_ub, acc, dim=1, clear=False)
                T.copy(acc, out)

        return main

    return kern(S_, M, N)


def case_reduce(loop_kind):
    S_, M, N = 2, 64, 64
    x = torch.randn(S_, M, N, dtype=torch.float32).npu()
    o = _reduce_kern(S_, M, N, loop_kind)(x)
    torch.npu.synchronize()
    # CORRECT reference: kernel sums over dim 0 (loop iterations) AND dim 2 (N)
    report(f"reduce_{loop_kind}", o, x.cpu().sum(dim=(0, 2)), atol=1e-2, rtol=1e-2)


# ------------------------------------------------------------- gm alias (#1925)
def _alias_kern(M, with_flags):
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

    return kern(M)


def case_alias(with_flags):
    M = 64
    x = torch.randn(M, dtype=torch.float32).npu()
    scratch = torch.zeros(1, M, dtype=torch.float32).npu()
    out = _alias_kern(M, with_flags)(x, scratch, scratch.view(1, M, 1))
    torch.npu.synchronize()
    tag = "alias_flags" if with_flags else "alias_noflags"
    ok = report(tag, out, x.cpu().unsqueeze(1), atol=1e-3, rtol=1e-2)
    if not with_flags and not ok:
        zeros = bool((out.detach().cpu().abs() < 1e-6).all())
        print(f"[{tag}] out is ALL ZEROS: {zeros}  "
              f"(True => read returned scratch's INITIAL content => write/read race confirmed)")


# ------------------------------------------------------------- multihop (#1926)
def case_multihop():
    M = 64

    @tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
    def kern(M):
        @T.prim_func
        def main(
            S_in: T.Tensor((M, M), "float32"),
            A: T.Tensor((M, M), "float32"),
            dA: T.Tensor((M, M), "float32"),
        ):
            with T.Kernel(1, threads=1, is_npu=True):
                s_ub = T.alloc_shared([M, M], "float32")
                a_ub = T.alloc_shared([M, M], "float32")
                s_hi_ub = T.alloc_shared([M, M], "bfloat16")
                a_hi_ub = T.alloc_shared([M, M], "bfloat16")
                s_l1 = T.alloc_shared([M, M], "bfloat16")
                a_l1 = T.alloc_shared([M, M], "bfloat16")
                z_frag = T.alloc_fragment([M, M], "float32")
                z_ub = T.alloc_shared([M, M], "float32")
                w_ub = T.alloc_shared([M, M], "float32")
                w_hi_ub = T.alloc_shared([M, M], "bfloat16")
                t2_frag = T.alloc_fragment([M, M], "float32")
                t2_ub = T.alloc_shared([M, M], "float32")
                T.copy(S_in, s_ub)
                T.copy(A, a_ub)
                T.copy(s_ub, s_hi_ub)
                T.copy(a_ub, a_hi_ub)
                T.copy(s_hi_ub, s_l1)
                T.copy(a_hi_ub, a_l1)
                T.gemm_v0(s_l1, a_l1, z_frag, transpose_B=True, init=True)
                T.copy(z_frag, z_ub)
                T.tile.mul(w_ub, z_ub, 2.0)
                T.copy(w_ub, w_hi_ub)
                T.copy(w_hi_ub, s_l1)
                T.gemm_v0(a_l1, s_l1, t2_frag, transpose_A=True, init=True)
                T.copy(t2_frag, t2_ub)
                T.tile.mul(t2_ub, t2_ub, -1.0)
                T.copy(t2_ub, dA)

        return main

    s = torch.randn(M, M, dtype=torch.float32).npu()
    a = torch.randn(M, M, dtype=torch.float32).npu()
    print("[multihop] compiling...")
    k = kern(M)
    print("[multihop] compile done, launching...")
    t0 = time.perf_counter()
    out = k(s, a)
    torch.npu.synchronize()
    print(f"[multihop] run done in {time.perf_counter() - t0:.1f}s")
    sf, af = s.cpu(), a.cpu()
    report("multihop", out, -(af.T @ ((sf @ af.T) * 2.0)), atol=1e-1, rtol=1e-1)


CASES = {
    "fp32_gemm": case_fp32_gemm,
    "ext1": case_ext1,
    "ext3": case_ext3,
    "reduce_serial": lambda: case_reduce("serial"),
    "reduce_pipelined": lambda: case_reduce("pipelined"),
    "alias_noflags": lambda: case_alias(False),
    "alias_flags": lambda: case_alias(True),
    "multihop": case_multihop,
}

if __name__ == "__main__":
    tilelang.disable_cache()
    CASES[sys.argv[1]]()
