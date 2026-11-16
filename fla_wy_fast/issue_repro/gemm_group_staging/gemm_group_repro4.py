"""多 gemm 组 L0 staging —— 复现 v4（性能放大：ms 级执行，测 staging 重叠效率）

merged：循环外 X=A^T@dw(hi/lo 2 gemm) + Pipelined 循环内 dA+=dw@K^T(hi/lo 2 gemm)
split ：X 与 dA 两个 kernel 分开发射（交付形态）

判定：若 merged 时间 ≈ split 的 X+dA 之和 → staging 正常重叠；
     若 merged >> split 之和 → L0 staging 被破坏（cube 空转/scalar 主导）复现。

Usage: python gemm_group_repro4.py [--bench]
"""
import sys

import tilelang
import tilelang.language as T
import torch

M = N = 128
K = 128        # X gemm 的 K（=dw 前 K 列）
DK = 4096      # dA 累积的总 K 深度
bDK = 128      # 循环内 K 分块
ITERS = DK // bDK

DTYPE = "float16"

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS)
def kern_x():
    @T.prim_func
    def main(
        a: T.Tensor((M, K), DTYPE),
        dw: T.Tensor((M, K), DTYPE),
        X: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True):
            a_l1 = T.alloc_shared([M, K], DTYPE)
            dw_hi_l1 = T.alloc_shared([M, K], DTYPE)
            dw_lo_l1 = T.alloc_shared([M, K], DTYPE)
            x_frag = T.alloc_fragment([M, N], "float32")
            T.copy(a, a_l1)
            T.copy(dw, dw_hi_l1)
            T.copy(dw, dw_lo_l1)
            T.gemm_v0(a_l1, dw_hi_l1, x_frag, transpose_A=True, init=True)
            T.gemm_v0(a_l1, dw_lo_l1, x_frag, transpose_A=True, init=False)
            T.copy(x_frag, X)

    return main


@tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS)
def kern_da():
    @T.prim_func
    def main(
        dw: T.Tensor((M, DK), DTYPE),
        kb: T.Tensor((N, DK), DTYPE),
        dA: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True):
            dw_hi_l1 = T.alloc_shared([M, bDK], DTYPE)
            dw_lo_l1 = T.alloc_shared([M, bDK], DTYPE)
            kb_hi_l1 = T.alloc_shared([N, bDK], DTYPE)
            kb_lo_l1 = T.alloc_shared([N, bDK], DTYPE)
            da_frag = T.alloc_fragment([M, N], "float32")
            for i_k in T.Pipelined(ITERS, num_stages=2):
                T.copy(dw[:, i_k * bDK : (i_k + 1) * bDK], dw_hi_l1)
                T.copy(dw[:, i_k * bDK : (i_k + 1) * bDK], dw_lo_l1)
                T.copy(kb[:, i_k * bDK : (i_k + 1) * bDK], kb_hi_l1)
                T.copy(kb[:, i_k * bDK : (i_k + 1) * bDK], kb_lo_l1)
                T.gemm_v0(
                    dw_hi_l1, kb_hi_l1, da_frag, transpose_B=True,
                    init=(i_k == 0),
                )
                T.gemm_v0(dw_lo_l1, kb_lo_l1, da_frag, transpose_B=True,
                          init=False)
            T.copy(da_frag, dA)

    return main


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

            T.copy(a, a_l1)
            T.copy(dw[:, 0:K], dwx_hi_l1)
            T.copy(dw[:, 0:K], dwx_lo_l1)
            T.gemm_v0(a_l1, dwx_hi_l1, x_frag, transpose_A=True, init=True)
            T.gemm_v0(a_l1, dwx_lo_l1, x_frag, transpose_A=True, init=False)
            T.copy(x_frag, x_ub)

            for i_k in T.Pipelined(ITERS, num_stages=2):
                T.copy(dw[:, i_k * bDK : (i_k + 1) * bDK], dw_hi_l1)
                T.copy(dw[:, i_k * bDK : (i_k + 1) * bDK], dw_lo_l1)
                T.copy(kb[:, i_k * bDK : (i_k + 1) * bDK], kb_hi_l1)
                T.copy(kb[:, i_k * bDK : (i_k + 1) * bDK], kb_lo_l1)
                T.gemm_v0(
                    dw_hi_l1, kb_hi_l1, da_frag, transpose_B=True,
                    init=(i_k == 0),
                )
                T.gemm_v0(dw_lo_l1, kb_lo_l1, da_frag, transpose_B=True,
                          init=False)
            T.copy(da_frag, Out[0, :, :])
            T.copy(x_ub, Out[1, :, :])

    return main


def bench(fn, *args, reps=20, warmup=5):
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
    X_ref = 2.0 * a_f.T @ dw_f[:, :K]
    dA_ref = 2.0 * dw_f @ kb_f.T

    kx, kda, km = kern_x(), kern_da(), kern_merged()
    X = kx(a, dw[:, :K].contiguous())
    dA = kda(dw, kb)
    out = km(a, dw, kb)
    torch.npu.synchronize()

    e1 = (X.cpu().float() - X_ref).abs().max().item()
    e2 = (dA.cpu().float() - dA_ref).abs().max().item()
    e3 = (out[0].cpu().float() - dA_ref).abs().max().item()
    e4 = (out[1].cpu().float() - X_ref).abs().max().item()
    print(f"[correct] split X err={e1:.3f} dA err={e2:.3f} | "
          f"merged dA err={e3:.3f} X err={e4:.3f}")

    if "--bench" in sys.argv:
        tx = bench(kx, a, dw[:, :K].contiguous())
        td = bench(kda, dw, kb)
        tm = bench(km, a, dw, kb)
        print(f"[bench] split: X={tx:.3f} ms, dA={td:.3f} ms, 合计={tx + td:.3f} ms")
        print(f"[bench] merged: {tm:.3f} ms  (split合计 / merged = {(tx + td) / tm:.2f})")


if __name__ == "__main__":
    main()