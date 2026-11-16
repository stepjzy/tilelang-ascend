"""多 gemm 组 L0 staging —— 复现 v3（hi/lo 四 gemm 组 + 角色互换，最接近 bwd 实际）

bwd 实际合并形态（fp32 gemm 损坏所迫的 hi/lo 模拟）：
  循环外: X_hi = A^T @ dw_hi ; X_lo = A^T @ dw_lo     （dw 是 B 操作数 → L0B）
  循环内: dA += dw_hi @ K_hi^T + dw_lo @ K_lo^T       （dw 是 A 操作数 → L0A）
  dw_hi_l1/dw_lo_l1 在组间角色互换复用。

Usage: python gemm_group_repro3.py [--bench]
"""
import sys

import tilelang
import tilelang.language as T
import torch

M = N = 64
K = 64
DK = 256  # 4 次迭代
bDK = 64

DTYPE = "float16"  # bwd 用 fp16 hi/lo

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def hi_lo(x):
    hi = x.to(torch.float16)
    lo = (x - hi.float()).to(torch.float16)
    return hi, lo


@tilelang.jit(out_idx=[3], pass_configs=PASS_CONFIGS)
def kern_merged():
    @T.prim_func
    def main(
        a: T.Tensor((M, K), DTYPE),
        dw: T.Tensor((M, DK), DTYPE),
        kb: T.Tensor((N, DK), DTYPE),
        Out: T.Tensor((2, M, N), "float32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True):
            a_l1 = T.alloc_shared([M, K], DTYPE)
            dwx_hi_l1 = T.alloc_shared([M, K], DTYPE)
            dwx_lo_l1 = T.alloc_shared([M, K], DTYPE)
            x_frag = T.alloc_fragment([M, N], "float32")
            x_ub = T.alloc_shared([M, N], "float32")
            dw_hi_l1 = T.alloc_shared([M, bDK], DTYPE)
            dw_lo_l1 = T.alloc_shared([M, bDK], DTYPE)
            kb_hi_l1 = T.alloc_shared([N, bDK], DTYPE)
            kb_lo_l1 = T.alloc_shared([N, bDK], DTYPE)
            da_frag = T.alloc_fragment([M, N], "float32")

            # ---- X = A^T @ dw[:, 0:K]，hi/lo 2 gemm（dw 前 64 列进 L0B）----
            T.copy(a, a_l1)
            T.copy(dw[:, 0:K], dwx_hi_l1)
            T.copy(dw[:, 0:K], dwx_lo_l1)  # 简化：直接同值（真实算子这里 hi/lo 拆分）
            T.gemm_v0(a_l1, dwx_hi_l1, x_frag, transpose_A=True, init=True)
            T.gemm_v0(a_l1, dwx_lo_l1, x_frag, transpose_A=True, init=False)
            T.copy(x_frag, x_ub)

            # ---- Pipelined dA 累加，每迭代 hi/lo 2 gemm（dw 进 L0A）----
            for i_k in T.Pipelined(DK // bDK, num_stages=2):
                T.copy(dw[:, i_k * bDK : (i_k + 1) * bDK], dw_hi_l1)
                T.copy(dw[:, i_k * bDK : (i_k + 1) * bDK], dw_lo_l1)
                T.copy(kb[:, i_k * bDK : (i_k + 1) * bDK], kb_hi_l1)
                T.copy(kb[:, i_k * bDK : (i_k + 1) * bDK], kb_lo_l1)
                T.gemm_v0(
                    dw_hi_l1, kb_hi_l1, da_frag, transpose_B=True,
                    init=(i_k == 0),
                )
                T.gemm_v0(
                    dw_lo_l1, kb_lo_l1, da_frag, transpose_B=True,
                    init=False,
                )
            T.copy(da_frag, Out[0, :, :])
            T.copy(x_ub, Out[1, :, :])

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
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16).npu()
    dw = torch.randn(M, DK, dtype=torch.float16).npu()
    kb = torch.randn(N, DK, dtype=torch.float16).npu()

    a_f, dw_f, kb_f = a.cpu().float(), dw.cpu().float(), kb.cpu().float()
    # golden：v3 里 X 用同值 hi/lo 双份 → X_ref = 2 * (A^T@dw[:,:K])
    X_ref = 2.0 * a_f.T @ dw_f[:, :K]
    # dA golden：hi/lo 双份同值 → dA_ref = 2 * (dw@kb^T)
    dA_ref = 2.0 * dw_f @ kb_f.T

    def check(name, got, ref, tol=2.0):
        err = (got.float() - ref).abs().max().item()
        ok = err < tol
        print(f"[merged] {name}: max_err={err:.3f} {'PASS' if ok else 'FAIL'}")
        return ok

    k = kern_merged()
    out = k(a, dw, kb)
    torch.npu.synchronize()
    c1 = check("dA = dw@K^T", out[0].cpu(), dA_ref)
    c2 = check("X = A^T@dw", out[1].cpu(), X_ref)
    if "--bench" in sys.argv:
        print(f"[merged] time={bench(k, a, dw, kb):.3f}")
    if not (c1 and c2):
        print("[merged] <<< 复现损坏")


if __name__ == "__main__":
    main()