"""多 gemm 组共享 L1 操作数（角色互换）L0 staging 损坏 —— 最小复现

对齐 fla_wy_fast bwd 被拆分的 k1/k2 数学：
  k1 kernel: dA += dw @ K^T   (transpose_B=True)   —— dw 是 A 操作数（进 L0A）
  k2 kernel: X  = A^T @ dw    (transpose_A=True)   —— dw 是 B 操作数（进 L0B）

三臂：
  single1  : 只有 gemm1（对照）
  single2  : 只有 gemm2（对照）
  combined : 同 kernel 连续两个 gemm，共享 dw_l1（角色互换复用）

验证点：
  1) 正确性：combined 两路输出 vs torch reference
  2) 时间：single1+single2 两次 launch 之和 vs combined 一次 launch
  3) 生成代码：/tmp/tmp*.cpp 中两个 gemm_v0 的 L0AB_EVENT 序列

Usage: python gemm_group_repro.py <single1|single2|combined> [--bench]
"""
import sys

import tilelang
import tilelang.language as T
import torch

M = N = K = 64
DTYPE = "bfloat16"

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS)
def kern_single1():
    @T.prim_func
    def main(
        dw: T.Tensor((M, K), DTYPE),
        kb: T.Tensor((N, K), DTYPE),
        dA: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True):
            dw_l1 = T.alloc_shared([M, K], DTYPE)
            kb_l1 = T.alloc_shared([N, K], DTYPE)
            da_frag = T.alloc_fragment([M, N], "float32")
            T.copy(dw, dw_l1)
            T.copy(kb, kb_l1)
            T.gemm_v0(dw_l1, kb_l1, da_frag, transpose_B=True, init=True)
            T.copy(da_frag, dA)

    return main


@tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS)
def kern_single2():
    @T.prim_func
    def main(
        a: T.Tensor((M, K), DTYPE),
        dw: T.Tensor((M, K), DTYPE),
        X: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True):
            a_l1 = T.alloc_shared([M, K], DTYPE)
            dw_l1 = T.alloc_shared([M, K], DTYPE)
            x_frag = T.alloc_fragment([M, N], "float32")
            T.copy(a, a_l1)
            T.copy(dw, dw_l1)
            T.gemm_v0(a_l1, dw_l1, x_frag, transpose_A=True, init=True)
            T.copy(x_frag, X)

    return main


@tilelang.jit(out_idx=[3], pass_configs=PASS_CONFIGS)
def kern_combined():
    @T.prim_func
    def main(
        a: T.Tensor((M, K), DTYPE),
        dw: T.Tensor((M, K), DTYPE),
        kb: T.Tensor((N, K), DTYPE),
        Out: T.Tensor((2, M, N), "float32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True):
            a_l1 = T.alloc_shared([M, K], DTYPE)
            dw_l1 = T.alloc_shared([M, K], DTYPE)  # 两个 gemm 共享（角色互换）
            kb_l1 = T.alloc_shared([N, K], DTYPE)
            da_frag = T.alloc_fragment([M, N], "float32")
            x_frag = T.alloc_fragment([M, N], "float32")
            T.copy(a, a_l1)
            T.copy(dw, dw_l1)
            T.copy(kb, kb_l1)
            # gemm1: dA = dw @ K^T   (dw -> L0A)
            T.gemm_v0(dw_l1, kb_l1, da_frag, transpose_B=True, init=True)
            # gemm2: X = A^T @ dw    (dw -> L0B)
            T.gemm_v0(a_l1, dw_l1, x_frag, transpose_A=True, init=True)
            T.copy(da_frag, Out[0, :, :])
            T.copy(x_frag, Out[1, :, :])

    return main


def bench(fn, *args, reps=50, warmup=10):
    for _ in range(warmup):
        fn(*args)
    torch.npu.synchronize()
    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn(*args)
    e.record()
    torch.npu.synchronize()
    return s.elapsed_time(e) / reps


def main():
    tilelang.disable_cache()
    arm = sys.argv[1]
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16).npu()
    dw = torch.randn(M, K, dtype=torch.bfloat16).npu()
    kb = torch.randn(N, K, dtype=torch.bfloat16).npu()

    # reference（CPU fp32 计算，bf16 输入）
    dw_f, kb_f, a_f = dw.cpu().float(), kb.cpu().float(), a.cpu().float()
    dA_ref = dw_f @ kb_f.T
    X_ref = a_f.T @ dw_f

    def check(name, got, ref, tol=1.0):
        err = (got.float() - ref).abs().max().item()
        ok = err < tol
        print(f"[{arm}] {name}: max_err={err:.3f} {'PASS' if ok else 'FAIL'}")
        return ok

    if arm == "single1":
        k = kern_single1()
        dA = k(dw, kb)
        torch.npu.synchronize()
        check("dA = dw@K^T", dA.cpu(), dA_ref)
        if "--bench" in sys.argv:
            print(f"[single1] time={bench(k, dw, kb):.3f} ms")
    elif arm == "single2":
        k = kern_single2()
        X = k(a, dw)
        torch.npu.synchronize()
        check("X = A^T@dw", X.cpu(), X_ref)
        if "--bench" in sys.argv:
            print(f"[single2] time={bench(k, a, dw):.3f} ms")
    elif arm == "combined":
        k = kern_combined()
        out = k(a, dw, kb)
        torch.npu.synchronize()
        c1 = check("dA = dw@K^T", out[0].cpu(), dA_ref)
        c2 = check("X = A^T@dw", out[1].cpu(), X_ref)
        if "--bench" in sys.argv:
            print(f"[combined] time={bench(k, a, dw, kb):.3f} ms")
        if not (c1 and c2):
            print("[combined] <<< 角色互换复用 L1 时结果损坏/挂死——复现 L0 staging 问题")


if __name__ == "__main__":
    main()