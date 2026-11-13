# EXACT text as posted in issue #1925 (main reproducer, no-flags arm)
import tilelang
import tilelang.language as T
import torch

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

M = 64

@tilelang.jit(out_idx=[3], pass_configs=PASS_CONFIGS)
def kern(M):
    @T.prim_func
    def main(
        In: T.Tensor((M,), "float32"),
        A: T.Tensor((1, M), "float32"),
        B: T.Tensor((1, M), "float32"),   # host 侧把同一个张量同时作为 A 和 B 传入
        Out: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True):
            in_ub = T.alloc_shared([M], "float32")
            acc = T.alloc_shared([M], "float32")
            v1d = T.alloc_shared([M], "float32")
            T.copy(In, in_ub)
            T.copy(in_ub, acc)
            T.copy(acc, A[0, 0:M])        # 经参数 A 写入（MTE3 队列）
            # 下方这次别名读没有被插入任何依赖
            T.copy(B[0, 0:M], v1d)        # 经参数 B 读回（MTE2 队列）——同一块存储
            T.copy(v1d, Out)

    return main

k = kern(M)
for r in range(10):
    x = torch.randn(M, dtype=torch.float32).npu()
    scratch = torch.zeros(1, M, dtype=torch.float32).npu()
    out = k(x, scratch, scratch)          # <-- 同一个张量既作为 A 也作为 B 传入
    torch.npu.synchronize()
    assert out.shape == x.shape
    err = (out.cpu() - x.cpu()).abs().max().item()
    print(f"rep{r}: max_abs_err={err:.4e}", "MATCH" if err < 1e-3 else "MISMATCH (stale)")
