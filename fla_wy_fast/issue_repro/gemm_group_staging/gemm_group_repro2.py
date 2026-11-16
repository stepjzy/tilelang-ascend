"""多 gemm 组共享 L1（角色互换）L0 staging 损坏 —— 复现 v2（对齐 bwd 合并形态）

对齐 fla_wy_fast bwd k1+k2 合并的完整形态：
  X  = A^T @ dw        （循环外单次 gemm，dw 是 B 操作数 → L0B）
  for i_k in Pipelined(DK//bDK):
      dA += dw @ K^T   （循环内累加 gemm，dw 是 A 操作数 → L0A）
      dk2 = X ⊙ beta   （X 的 L0C 结果回 UB，V 算使用）

dw_l1 在两个 gemm 组之间角色互换复用（L0B ↔ L0A）。

Arms:
  merged : 上面整个形态单 kernel
  split  : X gemm 一个 kernel + dA 循环 gemm 一个 kernel（当前交付形态）

Usage: python gemm_group_repro2.py <merged|split> [--bench]
"""
import sys

import tilelang
import tilelang.language as T
import torch

M = N = 64
K = 64          # 每个 gemm 的 K 维（=DK 分块大小）
DK = 128        # 总 K 深度（2 次迭代）
bDK = 64

DTYPE = "bfloat16"

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS)
def kern_split_x():
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


@tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS)
def kern_split_da():
    @T.prim_func
    def main(
        dw: T.Tensor((M, DK), DTYPE),
        kb: T.Tensor((N, DK), DTYPE),
        dA: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True):
            dw_l1 = T.alloc_shared([M, bDK], DTYPE)
            kb_l1 = T.alloc_shared([N, bDK], DTYPE)
            da_frag = T.alloc_fragment([M, N], "float32")
            for i_k in T.Pipelined(DK // bDK, num_stages=2):
                T.copy(dw[:, i_k * bDK : (i_k + 1) * bDK], dw_l1)
                T.copy(kb[:, i_k * bDK : (i_k + 1) * bDK], kb_l1)
                T.gemm_v0(
                    dw_l1, kb_l1, da_frag, transpose_B=True, init=(i_k == 0)
                )
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
            dw0_l1 = T.alloc_shared([M, bDK], DTYPE)  # 循环内 dw 切片
            kb_l1 = T.alloc_shared([N, bDK], DTYPE)
            x_frag = T.alloc_fragment([M, N], "float32")
            da_frag = T.alloc_fragment([M, N], "float32")
            x_ub = T.alloc_shared([M, N], "float32")
            # ---- X = A^T @ dw[:, 0:64] （循环外，dw 前 bDK 列进 L0B）----
            dw_x_l1 = T.alloc_shared([M, K], DTYPE)
            T.copy(a, a_l1)
            T.copy(dw[:, 0:K], dw_x_l1)
            T.gemm_v0(a_l1, dw_x_l1, x_frag, transpose_A=True, init=True)
            T.copy(x_frag, x_ub)
            # ---- Pipelined dA 累加（dw 各切片进 L0A）----
            for i_k in T.Pipelined(DK // bDK, num_stages=2):
                T.copy(dw[:, i_k * bDK : (i_k + 1) * bDK], dw0_l1)
                T.copy(kb[:, i_k * bDK : (i_k + 1) * bDK], kb_l1)
                T.gemm_v0(
                    dw0_l1, kb_l1, da_frag, transpose_B=True, init=(i_k == 0)
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
    arm = sys.argv[1]
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16).npu()
    dw = torch.randn(M, DK, dtype=torch.bfloat16).npu()
    kb = torch.randn(N, DK, dtype=torch.bfloat16).npu()

    a_f, dw_f, kb_f = a.cpu().float(), dw.cpu().float(), kb.cpu().float()
    X_ref = a_f.T @ dw_f[:, :K]
    dA_ref = dw_f @ kb_f.T

    def check(name, got, ref, tol=1.0):
        err = (got.float() - ref).abs().max().item()
        ok = err < tol
        print(f"[{arm}] {name}: max_err={err:.3f} {'PASS' if ok else 'FAIL'}")
        return ok

    if arm == "split":
        kx, kda = kern_split_x(), kern_split_da()
        X = kx(a, dw[:, :K].contiguous())
        dA = kda(dw, kb)
        torch.npu.synchronize()
        c1 = check("X = A^T@dw", X.cpu(), X_ref)
        c2 = check("dA = dw@K^T", dA.cpu(), dA_ref)
        if "--bench" in sys.argv:
            print(f"[split] tX={bench(kx, a, dw[:, :K].contiguous()):.3f} "
                  f"tdA={bench(kda, dw, kb):.3f}")
    elif arm == "merged":
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