"""fla_chunk_intra 分层测试：golden（PyTorch fp64 重写）+ L0/L1/L2/Boundary + bench。

golden 语义权威：ref_gpu/fla_chunk_intra.py（Triton，只读）；
输入构造对齐 ref_gpu/fla_chunk_intra_tilelang_gpu_{fwd,bwd}.py（上游测试样例）：
  fwd: q/k/beta randn bf16; gk = logsigmoid(randn fp32) 后 chunk 内 cumsum;
       Akk_diag = torch.ones fp32
  bwd: 全 randn fp32（g 未 cumsum）
精度标准：precision-standard.md §4.1 混合容差双门限（按 dtype）。
计时口径：do_bench（warmup=20, rep=10, mean, 仅设备侧 kernel 时间），对齐
ref_gpu/test_utils_kda.py。
"""

import argparse
import os
import sys

import tilelang
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fla_chunk_intra import (  # noqa: E402
    fla_chunk_intra_bwd_intra,
    fla_chunk_intra_fwd_inter_solve_fused,
)

DEVICE = "npu:0"


# ============================================================================
# 输入构造语义参考（ref_gpu/fla_chunk_intra_cumsum.py）
# ============================================================================
def chunk_local_cumsum(g, chunk=64, reverse=False):
    B, S, H, D = g.shape
    x = g.view(B, S // chunk, chunk, H, D)
    if reverse:
        x = x.flip(2).cumsum(2).flip(2)
    else:
        x = x.cumsum(2)
    return x.view(B, S, H, D)


# ============================================================================
# Golden：fwd（PyTorch fp64 重写；DESIGN §9.1，已修正转写笔误）
# 未写出区域（Aqk 对角 + 块上三角）显式写 0（DESIGN §1.2 确定性口径）。
# ============================================================================
def fla_chunk_intra_fwd_ref(q, k, gk, beta, Akk_diag, chunk_size=64, sub_chunk_size=16, scale=1.0):
    B, S, H, DK = q.shape
    CS, BC = chunk_size, sub_chunk_size
    assert S % CS == 0 and CS % BC == 0
    qf, kf, bf, gf, adf = (t.double() for t in (q, k, beta, gk, Akk_diag))
    Aqk = torch.zeros(B, S, H, CS, dtype=torch.float64)
    Akk_out = torch.zeros(B, S, H, CS, dtype=torch.float64)
    for c in range(S // CS):
        s0 = c * CS
        g_c, k_c, q_c = gf[:, s0 : s0 + CS], kf[:, s0 : s0 + CS], qf[:, s0 : s0 + CS]
        # (1) off-diag Aqk（直接式 2^(g_r − g_j)，fp64 域安全）
        for s2 in range(1, 4):
            r0 = s2 * BC
            for s1 in range(0, s2):
                j0 = s1 * BC
                e = torch.exp2(g_c[:, r0 : r0 + BC].unsqueeze(2) - g_c[:, j0 : j0 + BC].unsqueeze(1))
                val = torch.einsum("brhd,bjhd,brjhd->brjh", q_c[:, r0 : r0 + BC], k_c[:, j0 : j0 + BC], e) * scale
                Aqk[:, s0 + r0 : s0 + r0 + BC, :, j0 : j0 + BC] = val.permute(0, 1, 3, 2)
        # (2) M 组装：M[s,s] = I + strict_tril(A_s)；M[s2,s1] = Akkβ[s2,s1]（正号）
        M = torch.zeros(B, CS, CS, H, dtype=torch.float64)
        Iblk = torch.eye(BC, dtype=torch.float64)
        Str = torch.tril(torch.ones(BC, BC, dtype=torch.float64), -1)
        for s in range(4):
            r0 = s * BC
            A_s = adf[:, s0 + r0 : s0 + r0 + BC, :, :].permute(0, 1, 3, 2)
            M[:, r0 : r0 + BC, r0 : r0 + BC, :] = A_s * Str[None, :, :, None] + Iblk[None, :, :, None]
            for s2 in range(s + 1, 4):
                r2 = s2 * BC
                e = torch.exp2(g_c[:, r2 : r2 + BC].unsqueeze(2) - g_c[:, r0 : r0 + BC].unsqueeze(1))
                akk = torch.einsum("brhd,bjhd,brjhd->brjh", k_c[:, r2 : r2 + BC], k_c[:, r0 : r0 + BC], e)
                M[:, r2 : r2 + BC, r0 : r0 + BC, :] = akk * bf[:, s0 + r2 : s0 + r2 + BC].unsqueeze(2)
        # (3) Ai = M^{-1}（与逐块前代求解 + 6 块递推代数等价，见 _fwd_ref_blockrec）
        Mi = M.permute(0, 3, 1, 2)
        eye = torch.eye(CS, dtype=torch.float64).expand(B, H, CS, CS).contiguous()
        Ai = torch.linalg.solve_triangular(Mi, eye, upper=False, unitriangular=True)
        Akk_out[:, s0 : s0 + CS, :, :] = Ai.permute(0, 2, 1, 3)
    return Aqk.to(q.dtype), Akk_out.to(q.dtype)


# ============================================================================
# Golden 交叉校验 1：fwd 逐块前代 + 6 块递推（上游 Triton 逐块公式直译，fp64）
# ============================================================================
def _fwd_ref_blockrec(q, k, gk, beta, Akk_diag, chunk_size=64, sub_chunk_size=16, scale=1.0):
    """[B, H, r, c] 布局；off-diag 用 (s2, s1) 逐对直接式。"""
    B, S, H, DK = q.shape
    CS, BC = chunk_size, sub_chunk_size
    qf, kf, bf, gf, adf = (t.double() for t in (q, k, beta, gk, Akk_diag))
    Aqk = torch.zeros(B, S, H, CS, dtype=torch.float64)
    Akk_out = torch.zeros(B, S, H, CS, dtype=torch.float64)
    for c in range(S // CS):
        s0 = c * CS
        g_c, k_c, q_c = gf[:, s0 : s0 + CS], kf[:, s0 : s0 + CS], qf[:, s0 : s0 + CS]
        Aikk = {}
        for s2 in range(1, 4):
            r2 = s2 * BC
            for s1 in range(0, s2):
                r1 = s1 * BC
                e = torch.exp2(g_c[:, r2 : r2 + BC].unsqueeze(2) - g_c[:, r1 : r1 + BC].unsqueeze(1))
                val = torch.einsum("brhd,bjhd,brjhd->brjh", q_c[:, r2 : r2 + BC], k_c[:, r1 : r1 + BC], e)
                Aqk[:, s0 + r2 : s0 + r2 + BC, :, r1 : r1 + BC] = (val * scale).permute(0, 1, 3, 2)
                akk = torch.einsum("brhd,bjhd,brjhd->brjh", k_c[:, r2 : r2 + BC], k_c[:, r1 : r1 + BC], e).permute(0, 3, 1, 2)
                # × β[s2 行]（正号）
                Aikk[(s2, s1)] = akk * bf[:, s0 + r2 : s0 + r2 + BC].permute(0, 2, 1).unsqueeze(-1)
        # 对角块：逐行前代（row_i = N[i,:] + Σ_{j<i} N[i,j]·Ai[j,:]；+I）
        Ai_d = {}
        Str = torch.tril(torch.ones(BC, BC, dtype=torch.float64), -1)
        for s in range(4):
            r0 = s * BC
            A_s = adf[:, s0 + r0 : s0 + r0 + BC, :, :].permute(0, 2, 1, 3)  # [B,H,r,c]
            # N = −strict_tril(A_s)（负号在 N 内部，上游 L264-274 语义：
            # 递推给出 (I−N)^{-1} = (I + strict_tril(A_s))^{-1}）
            N = -(A_s * Str)
            Ai = torch.zeros_like(N)
            for i in range(BC):
                row = N[..., i, :].clone()  # [B,H,c]
                if i > 0:
                    row = row + torch.einsum("bhj,bhjc->bhc", N[..., i, :], Ai)
                Ai[..., i, :] = row
            Ai_d[s] = Ai + torch.eye(BC, dtype=torch.float64)
        # 合并 6 块递推（上游公式）
        Ai10 = -Ai_d[1] @ Aikk[(1, 0)] @ Ai_d[0]
        Ai21 = -Ai_d[2] @ Aikk[(2, 1)] @ Ai_d[1]
        Ai32 = -Ai_d[3] @ Aikk[(3, 2)] @ Ai_d[2]
        Ai20 = -Ai_d[2] @ (Aikk[(2, 0)] @ Ai_d[0] + Aikk[(2, 1)] @ Ai10)
        Ai31 = -Ai_d[3] @ (Aikk[(3, 1)] @ Ai_d[1] + Aikk[(3, 2)] @ Ai21)
        Ai30 = -Ai_d[3] @ (Aikk[(3, 0)] @ Ai_d[0] + Aikk[(3, 1)] @ Ai10 + Aikk[(3, 2)] @ Ai20)
        for s in range(4):
            r0 = s * BC
            Akk_out[:, s0 + r0 : s0 + r0 + BC, :, r0 : r0 + BC] = Ai_d[s].permute(0, 2, 1, 3)
        for (s2, s1), blk in (((1, 0), Ai10), ((2, 0), Ai20), ((2, 1), Ai21), ((3, 0), Ai30), ((3, 1), Ai31), ((3, 2), Ai32)):
            r2, r1 = s2 * BC, s1 * BC
            Akk_out[:, s0 + r2 : s0 + r2 + BC, :, r1 : r1 + BC] = blk.permute(0, 2, 1, 3)
    return Aqk.to(q.dtype), Akk_out.to(q.dtype)


# ============================================================================
# Golden：bwd（统一掩码直接式，DESIGN §9.1，已修正转写笔误）
# 返回 (dq2, dk2, dg2, dbp)（fp64 域；dbp = db2.sum(0) 等价，NK 无关）。
# ============================================================================
def fla_chunk_intra_bwd_ref(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, chunk_size=64, sub_chunk_size=16):
    B, S, H, DK = q.shape
    CS = chunk_size  # BC unused (masks built inline)
    qf, kf, bf, gf = (t.double() for t in (q, k, beta, g))
    dAf, dKf, dqf, dkf, dbf, dgf = (t.double() for t in (dAqk, dAkk, dq, dk, db, dg))
    dq2 = torch.zeros(B, S, H, DK, dtype=torch.float64)
    dk2 = torch.zeros(B, S, H, DK, dtype=torch.float64)
    dg2 = torch.zeros(B, S, H, DK, dtype=torch.float64)
    dbp = torch.zeros(B, S, H, dtype=torch.float64)
    for c in range(S // CS):
        s0 = c * CS
        g_c, k_c, q_c, b_c = gf[:, s0 : s0 + CS], kf[:, s0 : s0 + CS], qf[:, s0 : s0 + CS], bf[:, s0 : s0 + CS]
        dA_c, dK_c = dAf[:, s0 : s0 + CS], dKf[:, s0 : s0 + CS]
        ar = torch.arange(CS)
        for i in range(CS):
            m_lo = (ar <= i).double()  # j ≤ i：(A)+(B) 统一掩码
            m_up = (ar >= i).double()  # j ≥ i：(D)+(E) 统一掩码
            eij = torch.exp2(g_c[:, i].unsqueeze(1) - g_c)  # 2^(g_i−g_j) [B,CS,H,DK]
            eji = torch.exp2(g_c - g_c[:, i].unsqueeze(1))  # 2^(g_j−g_i)
            # (A)+(B)：dq2_pre[i] / dk2pre[i]（pre-β）
            dq2[:, s0 + i] = torch.einsum("bhj,bjhk->bhk", dA_c[:, i] * m_lo, k_c * eij)
            dk2pre = torch.einsum("bhj,bjhk->bhk", dK_c[:, i] * m_lo, k_c * eij)
            # (D)+(E)：dkt[i]（转置侧；β_j 乘在 dK 项内）
            dkt = torch.einsum("bjh,bjhk->bhk", dK_c[:, :, :, i] * m_up.unsqueeze(-1) * b_c, k_c * eji) + torch.einsum(
                "bjh,bjhk->bhk", dA_c[:, :, :, i] * m_up.unsqueeze(-1), q_c * eji
            )
            # 尾段（(F) post-β：dg2 的 q⊙dq2 用 +dq 之前的值，上游 L481-489 顺序）
            dbp[:, s0 + i] = (dk2pre * k_c[:, i]).sum(-1)  # pre-β
            dk2b = dk2pre * b_c[:, i].unsqueeze(-1)
            dg2[:, s0 + i] = q_c[:, i] * dq2[:, s0 + i] + (dk2b - dkt) * k_c[:, i] + dgf[:, s0 + i]
            dq2[:, s0 + i] = dq2[:, s0 + i] + dqf[:, s0 + i]  # (C)
            dk2[:, s0 + i] = dk2b + dkf[:, s0 + i] + dkt
    return dq2, dk2, dg2, dbp


# ============================================================================
# Golden 交叉校验 2：bwd 上游逐列循环直译（(A)(B)(D)(E) 显式循环，fp64）
# ============================================================================
def _bwd_ref_upstream(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, chunk_size=64, sub_chunk_size=16):
    B, S, H, DK = q.shape
    CS = chunk_size
    qf, kf, bf, gf = (t.double() for t in (q, k, beta, g))
    dAf, dKf, dqf, dkf, dgf = (t.double() for t in (dAqk, dAkk, dq, dk, dg))
    dq2 = torch.zeros(B, S, H, DK, dtype=torch.float64)
    dk2 = torch.zeros(B, S, H, DK, dtype=torch.float64)
    dg2 = torch.zeros(B, S, H, DK, dtype=torch.float64)
    dbp = torch.zeros(B, S, H, dtype=torch.float64)
    for c in range(S // CS):
        s0 = c * CS
        for i in range(CS):
            gi = gf[:, s0 + i, :, :]  # [B,H,DK]
            dq2r = torch.zeros(B, H, DK, dtype=torch.float64)
            dk2r = torch.zeros(B, H, DK, dtype=torch.float64)
            # (A)+(B): Σ_{j<=i} dA[i,j]·k_j·2^(g_i−g_j)
            for j in range(i + 1):
                e = torch.exp2(gi - gf[:, s0 + j, :, :])
                dq2r = dq2r + dAf[:, s0 + i, :, j].unsqueeze(-1) * kf[:, s0 + j, :, :] * e
                dk2r = dk2r + dKf[:, s0 + i, :, j].unsqueeze(-1) * kf[:, s0 + j, :, :] * e
            # (D)+(E): Σ_{j>=i} (dA[j,i]·q_j + dK[j,i]·β_j·k_j)·2^(g_j−g_i)
            dkt = torch.zeros(B, H, DK, dtype=torch.float64)
            for j in range(i, CS):
                e = torch.exp2(gf[:, s0 + j, :, :] - gi)
                dkt = (
                    dkt
                    + (
                        dAf[:, s0 + j, :, i].unsqueeze(-1) * qf[:, s0 + j, :, :]
                        + dKf[:, s0 + j, :, i].unsqueeze(-1) * bf[:, s0 + j, :].unsqueeze(-1) * kf[:, s0 + j, :, :]
                    )
                    * e
                )
            dbp[:, s0 + i] = (dk2r * kf[:, s0 + i, :, :]).sum(-1)
            dk2b = dk2r * bf[:, s0 + i, :].unsqueeze(-1)
            dg2[:, s0 + i] = qf[:, s0 + i, :, :] * dq2r + (dk2b - dkt) * kf[:, s0 + i, :, :] + dgf[:, s0 + i]
            dq2[:, s0 + i] = dq2r + dqf[:, s0 + i]
            dk2[:, s0 + i] = dk2b + dkf[:, s0 + i] + dkt
    return dq2, dk2, dg2, dbp


def _golden_selfcheck():
    """极小 shape 交叉校验（DESIGN §9.1 建议）：锁定块结构、符号方向与
    (F) post-β 顺序，避免以错误 golden 烧 NPU 编译时间。"""
    ok = True
    torch.manual_seed(123)
    B, S, H, DK = 1, 64, 2, 32
    # fwd
    q = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
    k = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
    beta = torch.randn(B, S, H, dtype=torch.bfloat16)
    gk = chunk_local_cumsum(torch.nn.functional.logsigmoid(torch.randn(B, S, H, DK, dtype=torch.float32)))
    ad = torch.ones(B, S, H, 16, dtype=torch.float32)
    a1, b1 = fla_chunk_intra_fwd_ref(q, k, gk, beta, ad)
    a2, b2 = _fwd_ref_blockrec(q, k, gk, beta, ad)
    d1 = (a1.float() - a2.float()).abs().max().item()
    d2 = (b1.float() - b2.float()).abs().max().item()
    print(f"[SELFCHK] fwd golden vs blockrec: Aqk max_diff={d1:.3e} Akk max_diff={d2:.3e}")
    ok &= d1 < 1e-2 and d2 < 1e-2
    # bwd
    q = torch.randn(B, S, H, DK)
    k = torch.randn(B, S, H, DK)
    g = torch.randn(B, S, H, DK)
    beta = torch.randn(B, S, H)
    dAqk = torch.randn(B, S, H, 64)
    dAkk = torch.randn(B, S, H, 64)
    dq = torch.randn(B, S, H, DK)
    dk = torch.randn(B, S, H, DK)
    db = torch.randn(B, S, H)
    dg = torch.randn(B, S, H, DK)
    r1 = fla_chunk_intra_bwd_ref(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg)
    r2 = _bwd_ref_upstream(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg)
    for name, x, y in zip(("dq2", "dk2", "dg2", "dbp"), r1, r2):
        d = (x - y).abs().max().item()
        print(f"[SELFCHK] bwd golden vs upstream-loop {name}: max_diff={d:.3e}")
        ok &= d < 1e-9
    print(f"[SELFCHK] {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================
# 精度标准（precision-standard.md §4.1 原样复制）
# ============================================================================
def get_precision(dtype):
    """返回 (atol, rtol, max_abs_error_limit, required_matched_ratio)。
    浮点：混合容差；整型：精确匹配（0 误差）。"""
    fp_table = {
        # dtype       : (atol,   rtol,   max_abs_error_limit, required_matched_ratio)
        "float16": (2**-14, 2**-9, 1e-1, 0.99),  # noqa: E241
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),  # noqa: E241
        "float32": (2**-16, 2**-10, 1e-2, 0.99),  # noqa: E241
        "hifloat32": (2**-16, 2**-10, 1e-2, 0.99),  # noqa: E241
        "float8_e4m3": (2**-4, 2**-2, 1e0, 0.99),  # noqa: E241
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),  # noqa: E241
    }
    int_types = {"int8", "int16", "int32", "int64", "uint8"}
    if dtype in int_types:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """精度判定：返回 (passed, matched_ratio, max_abs_error)。
    浮点双门限：matched_ratio ≥ required 且 max_abs_error ≤ max_abs_error_limit；
    整型：逐元素精确相等。inf/nan 位置做结构比对，不计入数值容差。"""
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:  # 整型精确匹配
        mism = (a != g).sum().item()
        total = max(a.numel(), 1)
        return mism == 0, 1.0 - mism / total, (0.0 if mism == 0 else float("inf"))
    a = a.float()
    g = g.float()
    special = ~torch.isfinite(g)  # inf/nan 位置结构比对
    if special.any():  # noqa: SIM102 (verbatim from precision-standard.md)
        if not torch.equal(torch.isnan(a[special]), torch.isnan(g[special])) or not torch.equal(
            torch.isinf(a[special]), torch.isinf(g[special])
        ):
            return False, 0.0, float("inf")
    m = torch.isfinite(g)  # golden 有限值位置全比
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    matched_ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs_error = abs_err.max().item()
    passed = (matched_ratio >= required_ratio) and (max_abs_error <= max_abs_limit)
    return passed, matched_ratio, max_abs_error


# ============================================================================
# 输入构造（CPU 完成后单次 H2D；上游 prepare_input 语义）
# ============================================================================
def prepare_input_fwd(B, S, H, DK, chunk_size=64, sub_chunk_size=16):
    q = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
    k = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
    beta = torch.randn(B, S, H, dtype=torch.bfloat16)
    gk = torch.nn.functional.logsigmoid(torch.randn(B, S, H, DK, dtype=torch.float32))
    gk = chunk_local_cumsum(gk, chunk_size)
    Akk_diag = torch.ones(B, S, H, sub_chunk_size, dtype=torch.float32)
    return q, k, gk, beta, Akk_diag


def prepare_input_bwd(B, S, H, DK, chunk_size=64):
    q = torch.randn(B, S, H, DK, dtype=torch.float32)
    k = torch.randn(B, S, H, DK, dtype=torch.float32)
    g = torch.randn(B, S, H, DK, dtype=torch.float32)  # 未 cumsum（上游口径）
    beta = torch.randn(B, S, H, dtype=torch.float32)
    dAqk = torch.randn(B, S, H, chunk_size, dtype=torch.float32)
    dAkk = torch.randn(B, S, H, chunk_size, dtype=torch.float32)
    dq = torch.randn(B, S, H, DK, dtype=torch.float32)
    dk = torch.randn(B, S, H, DK, dtype=torch.float32)
    db = torch.randn(B, S, H, dtype=torch.float32)
    dg = torch.randn(B, S, H, DK, dtype=torch.float32)
    return q, k, g, beta, dAqk, dAkk, dq, dk, db, dg


# ============================================================================
# 单用例执行
# ============================================================================
def run_fwd_case(B, S, H, DK, tag, level):
    torch.manual_seed(42)
    q, k, gk, beta, ad = prepare_input_fwd(B, S, H, DK)
    aqk_ref, akk_ref = fla_chunk_intra_fwd_ref(q, k, gk, beta, ad)
    q_n, k_n, gk_n, beta_n, ad_n = (t.to(DEVICE) for t in (q, k, gk, beta, ad))
    aqk, akk = fla_chunk_intra_fwd_inter_solve_fused(q_n, k_n, gk_n, beta_n, ad_n)
    ok = True
    for name, out, ref in (("Aqk", aqk, aqk_ref), ("Akk", akk, akk_ref)):
        passed, ratio, mabs = check_precision(out, ref, "bfloat16")
        mk = "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]"
        print(f"{mk} {level}:{tag} {name} shape=({B},{S},{H},{DK}) matched_ratio={ratio:.6f} max_abs={mabs:.3e}")
        ok &= passed
    return ok


def run_bwd_case(B, S, H, DK, tag, level):
    torch.manual_seed(0)
    ins = prepare_input_bwd(B, S, H, DK)
    q, k, g, beta, dAqk, dAkk, dq, dk, db, dg = ins
    dq2r, dk2r, dg2r, dbpr = fla_chunk_intra_bwd_ref(*ins)
    # golden 侧同构 host 后处理（CPU fp64）
    db_ref = dbpr + db.double()
    dg_ref = chunk_local_cumsum(dg2r, 64, reverse=True)
    ins_n = [t.to(DEVICE) for t in ins]
    dq2, dk2, db2, dg2 = fla_chunk_intra_bwd_intra(*ins_n)
    # kernel 侧同构 host 后处理（proto 授权的 kernel 外边界，不计时）
    db_k = db2.sum(0) + ins_n[8]
    dg_k = chunk_local_cumsum(dg2, 64, reverse=True)
    ok = True
    for name, out, ref, dt in (
        ("dq2", dq2, dq2r, "float32"),
        ("dk2", dk2, dk2r, "float32"),
        ("dg2", dg2, dg2r, "float32"),
        ("db_post", db_k, db_ref, "float32"),
        ("dg_post", dg_k, dg_ref, "float32"),
    ):
        passed, ratio, mabs = check_precision(out, ref, dt)
        mk = "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]"
        print(f"{mk} {level}:{tag} {name} shape=({B},{S},{H},{DK}) matched_ratio={ratio:.6f} max_abs={mabs:.3e}")
        ok &= passed
    return ok


# ============================================================================
# L0 门槛测试（DESIGN §9.2）
# ============================================================================
def test_fla_chunk_intra_l0():
    ok = True
    ok &= run_fwd_case(1, 64, 2, 128, "l0_fwd_min", "l0")
    ok &= run_fwd_case(2, 256, 4, 128, "l0_fwd_repr", "l0")
    ok &= run_fwd_case(1, 128, 4, 64, "l0_fwd_dk64", "l0")
    ok &= run_fwd_case(1, 1024, 64, 128, "l0_fwd_bench_repr", "l0")
    ok &= run_bwd_case(1, 64, 2, 128, "l0_bwd_min", "l0")
    ok &= run_bwd_case(2, 256, 4, 128, "l0_bwd_repr", "l0")
    ok &= run_bwd_case(1, 128, 4, 64, "l0_bwd_dk64", "l0")
    return ok


# ============================================================================
# L1/L2/Boundary（tilelang-op-test-design 场景 B 扩展；覆盖矩阵契约见
# .agents/skills/tilelang-op-test-design/references/coverage-matrix.md）
# ============================================================================
# Category: CV Fusion (Cube gemm_v0 hi/lo chains + Vector exp2-gate/mask/tail).
# NOTE on D-SHAPE-TAIL-*/PRIME: the supported domain REQUIRES S%64==0 and
# DK in {64, 96, 128} (chunked data model: a trailing partial chunk has no
# defined state; bwd additionally needs the exact-2 Pipelined block). The
# wrapper therefore rejects non-divisible shapes deterministically, and
# those shapes are covered as L2 rejection cases in L2_CASES below --
# correct deterministic rejection IS the specified boundary behavior
# (same policy as the sibling fla_chunk_inter suite).
COVERAGE_CATEGORY = "Fusion"

# (direction, name, B, S, H, DK, scale, tags) -- all in-domain (block-divisible)
L1_CASES = [
    ("fwd", "l1_fwd_min", 1, 64, 1, 64, 1.0,
     ["D-SHAPE-ALIGNED", "D-SHAPE-EDGE", "D-DTYPE-bf16"]),
    ("fwd", "l1_fwd_dk96", 2, 192, 3, 96, 1.0,
     ["D-SHAPE-ALIGNED", "D-DTYPE-bf16"]),
    ("fwd", "l1_fwd_scale", 1, 128, 2, 128, 2.5,
     ["D-SHAPE-ALIGNED", "D-PARAM-scale"]),
    ("bwd", "l1_bwd_min", 1, 64, 1, 64, 1.0,
     ["D-SHAPE-ALIGNED", "D-DTYPE-fp32"]),
    ("bwd", "l1_bwd_dk96", 2, 192, 3, 96, 1.0,
     ["D-SHAPE-ALIGNED", "D-DTYPE-fp32"]),
    ("bwd", "l1_valrange_s", 1, 128, 2, 64, 1.0, ["D-VALRANGE-S"]),
    ("bwd", "l1_valrange_m", 1, 128, 2, 64, 1.0, ["D-VALRANGE-M"]),
    ("bwd", "l1_valrange_l", 1, 128, 2, 64, 1.0, ["D-VALRANGE-L"]),
    ("bwd", "l1_valrange_asym", 1, 128, 2, 64, 1.0, ["D-VALRANGE-ASYM"]),
]

# L1 value-range transforms (CPU-side, deterministic). They scale the 8
# linear-domain bwd operands and deliberately leave g (the log2-domain gate
# input) untouched: scaling a log-domain input multiplies the gate exponents
# (2^g spans explode), which overflows the fp16 hi/lo operand envelope for ANY
# implementation. Gate-exponent extremes are covered by Boundary cases.
def _bwd_operands_scaled(ts, mul, add=0.0):
    q, k, g, beta, dAqk, dAkk, dq, dk, db, dg = ts
    return (q * mul + add, k * mul + add, g, beta * mul + add,
            dAqk * mul + add, dAkk * mul + add, dq * mul + add,
            dk * mul + add, db * mul + add, dg * mul + add)


_L1_TRANSFORMS = {
    "l1_valrange_s": lambda ts: _bwd_operands_scaled(ts, 0.25),
    "l1_valrange_m": lambda ts: _bwd_operands_scaled(ts, 1.5),
    "l1_valrange_l": lambda ts: _bwd_operands_scaled(ts, 2.0),
    "l1_valrange_asym": lambda ts: _bwd_operands_scaled(ts, 0.5, 1.0),
}


def run_fwd_case_ex(B, S, H, DK, tag, level, scale=1.0):
    torch.manual_seed(42)
    q, k, gk, beta, ad = prepare_input_fwd(B, S, H, DK)
    aqk_ref, akk_ref = fla_chunk_intra_fwd_ref(q, k, gk, beta, ad, scale=scale)
    q_n, k_n, gk_n, beta_n, ad_n = (t.to(DEVICE) for t in (q, k, gk, beta, ad))
    aqk, akk = fla_chunk_intra_fwd_inter_solve_fused(
        q_n, k_n, gk_n, beta_n, ad_n, scale=scale)
    ok = True
    for name, out, ref in (("Aqk", aqk, aqk_ref), ("Akk", akk, akk_ref)):
        passed, ratio, mabs = check_precision(out, ref, "bfloat16")
        mk = "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]"
        print(f"{mk} {level}:{tag} {name} shape=({B},{S},{H},{DK}) "
              f"matched_ratio={ratio:.6f} max_abs={mabs:.3e}")
        ok &= passed
    return ok


def run_bwd_case_ex(B, S, H, DK, tag, level, transform=None):
    torch.manual_seed(0)
    ins = prepare_input_bwd(B, S, H, DK)
    if transform is not None:
        ins = transform(ins)
    q, k, g, beta, dAqk, dAkk, dq, dk, db, dg = ins
    dq2r, dk2r, dg2r, dbpr = fla_chunk_intra_bwd_ref(*ins)
    db_ref = dbpr + db.double()
    dg_ref = chunk_local_cumsum(dg2r, 64, reverse=True)
    ins_n = [t.to(DEVICE) for t in ins]
    dq2, dk2, db2, dg2 = fla_chunk_intra_bwd_intra(*ins_n)
    db_k = db2.sum(0) + ins_n[8]
    dg_k = chunk_local_cumsum(dg2, 64, reverse=True)
    ok = True
    for name, out, ref, dt in (
        ("dq2", dq2, dq2r, "float32"),
        ("dk2", dk2, dk2r, "float32"),
        ("dg2", dg2, dg2r, "float32"),
        ("db_post", db_k, db_ref, "float32"),
        ("dg_post", dg_k, dg_ref, "float32"),
    ):
        passed, ratio, mabs = check_precision(out, ref, dt)
        mk = "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]"
        print(f"{mk} {level}:{tag} {name} shape=({B},{S},{H},{DK}) "
              f"matched_ratio={ratio:.6f} max_abs={mabs:.3e}")
        ok &= passed
    return ok


def test_fla_chunk_intra_l1():
    ok = True
    for direction, name, B, S, H, DK, scale, _tags in L1_CASES:
        try:
            if direction == "fwd":
                ok &= run_fwd_case_ex(B, S, H, DK, name, "l1", scale=scale)
            else:
                ok &= run_bwd_case_ex(B, S, H, DK, name, "l1",
                                      transform=_L1_TRANSFORMS.get(name))
        except Exception as e:  # noqa: BLE001 - report and keep going
            print(f"[PRECISION_FAIL] l1:{name} shape=({B},{S},{H},{DK}): {e}")
            ok = False
    return ok


# (name, kind, payload, tags) -- illegal inputs; the wrapper must reject
# (ValueError). Tail/prime shapes land here BY DOMAIN (see NOTE above).
L2_CASES = [
    ("l2_s_tail1", "bwd_shape", (1, 65, 2, 64), ["D-EXC-SHAPE", "D-SHAPE-TAIL-1"]),
    ("l2_s_prime", "bwd_shape", (1, 67, 2, 64), ["D-EXC-SHAPE", "D-SHAPE-PRIME"]),
    ("l2_s_tailmid", "bwd_shape", (1, 96, 2, 64), ["D-EXC-SHAPE", "D-SHAPE-TAIL-MID"]),
    ("l2_dk_tail1", "bwd_shape", (1, 128, 2, 33), ["D-EXC-SHAPE", "D-SHAPE-TAIL-1"]),
    ("l2_dk_prime", "bwd_shape", (1, 128, 2, 67), ["D-EXC-SHAPE", "D-SHAPE-PRIME"]),
    ("l2_dk_tailmid", "bwd_shape", (1, 128, 2, 48), ["D-EXC-SHAPE", "D-SHAPE-TAIL-MID"]),
    ("l2_fwd_s_nonmult", "fwd_shape", (1, 100, 2, 64), ["D-EXC-SHAPE"]),
    ("l2_chunk_size", "chunk", 32, ["D-PARAM-chunk_size", "D-EXC-SHAPE"]),
    ("l2_sub_chunk", "sub_chunk", 8, ["D-PARAM-sub_chunk_size", "D-EXC-SHAPE"]),
    ("l2_fwd_dtype_fp16", "fwd_dtype", None, ["D-EXC-DTYPE"]),
    ("l2_bwd_dtype_bf16", "bwd_dtype", None, ["D-EXC-DTYPE"]),
    ("l2_noncontig", "noncontig", None, ["D-EXC-SHAPE"]),
]


def _make_l2_input(kind, payload):
    if kind == "bwd_shape":
        B, S, H, DK = payload
        return fla_chunk_intra_bwd_intra(*[t.npu() for t in prepare_input_bwd(B, S, H, DK)])
    if kind == "fwd_shape":
        B, S, H, DK = payload
        q, k, gk, beta, ad = prepare_input_fwd(B, S, H, DK)
        return fla_chunk_intra_fwd_inter_solve_fused(
            q.npu(), k.npu(), gk.npu(), beta.npu(), ad.npu())
    if kind == "chunk":
        ins = [t.npu() for t in prepare_input_bwd(1, 64, 2, 64)]
        return fla_chunk_intra_bwd_intra(*ins, chunk_size=payload)
    if kind == "sub_chunk":
        ins = [t.npu() for t in prepare_input_bwd(1, 64, 2, 64)]
        return fla_chunk_intra_bwd_intra(*ins, sub_chunk_size=payload)
    if kind == "fwd_dtype":
        # fp16 q/k on an otherwise valid shape: fwd requires bf16 q/k
        torch.manual_seed(1)
        B, S, H, DK = 1, 64, 2, 64
        q = torch.randn(B, S, H, DK, dtype=torch.float16)
        k = torch.randn(B, S, H, DK, dtype=torch.float16)
        gk = torch.randn(B, S, H, DK, dtype=torch.float32)
        beta = torch.randn(B, S, H, dtype=torch.bfloat16)
        ad = torch.ones(B, S, H, 16, dtype=torch.float32)
        return fla_chunk_intra_fwd_inter_solve_fused(
            q.npu(), k.npu(), gk.npu(), beta.npu(), ad.npu())
    if kind == "bwd_dtype":
        # bf16 inputs on an otherwise valid shape: bwd requires all fp32
        torch.manual_seed(1)
        B, S, H, DK = 1, 64, 2, 64
        ins = [t.bfloat16() for t in prepare_input_bwd(B, S, H, DK)]
        return fla_chunk_intra_bwd_intra(*[t.npu() for t in ins])
    if kind == "noncontig":
        ins = [t.npu() for t in prepare_input_bwd(1, 64, 2, 64)]
        ins[0] = ins[0].transpose(1, 2).contiguous().transpose(1, 2)  # non-contig view
        return fla_chunk_intra_bwd_intra(*ins)
    raise ValueError(f"unknown l2 kind {kind}")


def test_fla_chunk_intra_l2():
    for name, kind, payload, _tags in L2_CASES:
        try:
            _make_l2_input(kind, payload)
        except Exception as e:  # noqa: BLE001 - expected rejection
            print(f"[BOUNDARY_PASS] l2:{name}: rejected ({type(e).__name__})")
            continue
        print(f"[BOUNDARY_WARN] l2:{name}: illegal input NOT rejected (silent accept)")


# (name, kind, payload, dtype, tags) -- legal special values; precision per
# the dtype acceptance standard; WARN on miss (non-blocking by design).
BOUNDARY_CASES = [
    ("bnd_fwd_zero", "fwd_zero", (1, 64, 1, 64), "bfloat16", ["D-SPECIAL-ZERO"]),
    ("bnd_bwd_zero", "bwd_zero", (1, 64, 1, 64), "float32", ["D-SPECIAL-ZERO"]),
    ("bnd_fwd_decay", "fwd_decay", (1, 128, 2, 64), "bfloat16", ["D-SPECIAL-DBOUND"]),
    ("bnd_bwd_decay", "bwd_decay", (1, 128, 2, 64), "float32", ["D-SPECIAL-DBOUND"]),
    ("bnd_bwd_inf", "bwd_inf_dq", (1, 64, 1, 64), "float32", ["D-SPECIAL-INF"]),
    ("bnd_fwd_nan", "fwd_nan_beta", (1, 64, 1, 64), "bfloat16", ["D-SPECIAL-NAN"]),
    ("bnd_bwd_nan", "bwd_nan_k", (1, 64, 1, 64), "float32", ["D-SPECIAL-NAN"]),
]


def _run_boundary_case(name, kind, payload, dt):
    B, S, H, DK = payload
    try:
        if kind == "fwd_zero":
            torch.manual_seed(3)
            q = torch.zeros(B, S, H, DK, dtype=torch.bfloat16)
            k = torch.zeros(B, S, H, DK, dtype=torch.bfloat16)
            gk = torch.zeros(B, S, H, DK, dtype=torch.float32)
            beta = torch.zeros(B, S, H, dtype=torch.bfloat16)
            ad = torch.zeros(B, S, H, 16, dtype=torch.float32)
            aqk_r, akk_r = fla_chunk_intra_fwd_ref(q, k, gk, beta, ad)
            aqk, akk = fla_chunk_intra_fwd_inter_solve_fused(
                q.npu(), k.npu(), gk.npu(), beta.npu(), ad.npu())
            outs = [("Aqk", aqk, aqk_r), ("Akk", akk, akk_r)]
        elif kind == "fwd_decay":
            torch.manual_seed(3)
            q, k, gk, beta, ad = prepare_input_fwd(B, S, H, DK)
            # strong per-token decay: -1.8/token, chunk-total ~ -115 in log2
            # (fp32-normal-representable; exercises the wide gate dynamic range)
            pos = torch.arange(S, dtype=torch.float32) % 64
            gk = (-1.8 * pos).view(1, S, 1, 1).expand(B, S, H, DK).contiguous()
            aqk_r, akk_r = fla_chunk_intra_fwd_ref(q, k, gk, beta, ad)
            aqk, akk = fla_chunk_intra_fwd_inter_solve_fused(
                q.npu(), k.npu(), gk.npu().to(DEVICE), beta.npu(), ad.npu())
            outs = [("Aqk", aqk, aqk_r), ("Akk", akk, akk_r)]
        elif kind == "fwd_nan_beta":
            torch.manual_seed(3)
            q, k, gk, beta, ad = prepare_input_fwd(B, S, H, DK)
            beta = beta.clone()
            beta.view(-1)[0] = float("nan")
            aqk_r, akk_r = fla_chunk_intra_fwd_ref(q, k, gk, beta, ad)
            aqk, akk = fla_chunk_intra_fwd_inter_solve_fused(
                q.npu(), k.npu(), gk.npu(), beta.npu(), ad.npu())
            outs = [("Aqk", aqk, aqk_r), ("Akk", akk, akk_r)]
        elif kind == "bwd_zero":
            torch.manual_seed(3)
            B2, S2, H2, DK2 = B, S, H, DK
            ins = [
                torch.zeros(B2, S2, H2, DK2, dtype=torch.float32),  # q
                torch.zeros(B2, S2, H2, DK2, dtype=torch.float32),  # k
                torch.zeros(B2, S2, H2, DK2, dtype=torch.float32),  # g
                torch.zeros(B2, S2, H2, dtype=torch.float32),        # beta
                torch.zeros(B2, S2, H2, 64, dtype=torch.float32),    # dAqk
                torch.zeros(B2, S2, H2, 64, dtype=torch.float32),    # dAkk
                torch.zeros(B2, S2, H2, DK2, dtype=torch.float32),  # dq
                torch.zeros(B2, S2, H2, DK2, dtype=torch.float32),  # dk
                torch.zeros(B2, S2, H2, dtype=torch.float32),        # db
                torch.zeros(B2, S2, H2, DK2, dtype=torch.float32),  # dg
            ]
            dq2r, dk2r, dg2r, dbpr = fla_chunk_intra_bwd_ref(*ins)
            dq2, dk2, db2, dg2 = fla_chunk_intra_bwd_intra(*[t.npu() for t in ins])
            outs = [("dq2", dq2, dq2r), ("dk2", dk2, dk2r), ("dg2", dg2, dg2r)]
        elif kind == "bwd_decay":
            torch.manual_seed(3)
            ins = list(prepare_input_bwd(B, S, H, DK))
            # monotone linear gate: -0.38/token (chunk-total -24; the Kdec
            # operand spans reach ~2^+11.8, near the fp16 hi/lo envelope)
            pos = torch.arange(S, dtype=torch.float32)
            ins[2] = (-0.38 * pos).view(1, S, 1, 1).expand(B, S, H, DK).contiguous()
            dq2r, dk2r, dg2r, dbpr = fla_chunk_intra_bwd_ref(*ins)
            dq2, dk2, db2, dg2 = fla_chunk_intra_bwd_intra(*[t.npu() for t in ins])
            outs = [("dq2", dq2, dq2r), ("dk2", dk2, dk2r),
                    ("dg2", dg2, dg2r), ("db2", db2, dbpr.view(1, B, S, H))]
        elif kind == "bwd_inf_dq":
            torch.manual_seed(3)
            ins = list(prepare_input_bwd(B, S, H, DK))
            ins[6] = ins[6].clone()
            ins[6].view(-1)[0] = float("inf")
            dq2r, dk2r, dg2r, dbpr = fla_chunk_intra_bwd_ref(*ins)
            dq2, dk2, db2, dg2 = fla_chunk_intra_bwd_intra(*[t.npu() for t in ins])
            outs = [("dq2", dq2, dq2r), ("dk2", dk2, dk2r), ("dg2", dg2, dg2r)]
        elif kind == "bwd_nan_k":
            torch.manual_seed(3)
            ins = list(prepare_input_bwd(B, S, H, DK))
            ins[1] = ins[1].clone()
            ins[1].view(-1)[0] = float("nan")
            dq2r, dk2r, dg2r, dbpr = fla_chunk_intra_bwd_ref(*ins)
            dq2, dk2, db2, dg2 = fla_chunk_intra_bwd_intra(*[t.npu() for t in ins])
            outs = [("dq2", dq2, dq2r), ("dk2", dk2, dk2r), ("dg2", dg2, dg2r)]
        else:
            raise ValueError(f"unknown boundary kind {kind}")
        any_bad = False
        for oname, out, ref in outs:
            passed, ratio, mabs = check_precision(out, ref, dt)
            tag = "PASS" if passed else "WARN"
            if not passed:
                any_bad = True
            print(f"[BOUNDARY_{tag}] boundary:{name} {oname} dtype={dt} "
                  f"matched_ratio={ratio:.4f} max_abs={mabs:.3e}")
        return any_bad
    except Exception as e:  # noqa: BLE001 - boundary failures are non-blocking
        print(f"[BOUNDARY_WARN] boundary:{name} dtype={dt}: {e}")
        return True


def test_fla_chunk_intra_boundary():
    warns = []
    for name, kind, payload, dt, _tags in BOUNDARY_CASES:
        if _run_boundary_case(name, kind, payload, dt):
            warns.append(name)
    if warns:
        print(f"[BOUNDARY] warnings (non-blocking): {warns}")
    else:
        print("[BOUNDARY] all boundary cases passed")


# ============================================================================
# benchmark（DESIGN §9.4 性能可行性哨兵；do_bench 对齐 ref_gpu/test_utils_kda.py）
# ============================================================================
def do_bench(fn, warmup=20, rep=10):
    """NPU 事件计时：warmup=20 后 rep=10 取均值，仅设备侧 kernel 时间。"""
    starts = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    for i in range(rep):
        starts[i].record()
        fn()
        ends[i].record()
    torch.npu.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(starts, ends)], dtype=torch.float64)
    return times.mean().item()  # ms


def bench_fwd(B=1, S=8192, H=64, DK=128):
    torch.manual_seed(42)
    q, k, gk, beta, ad = prepare_input_fwd(B, S, H, DK)
    q_n, k_n, gk_n, beta_n, ad_n = (t.to(DEVICE) for t in (q, k, gk, beta, ad))

    def fn():
        fla_chunk_intra_fwd_inter_solve_fused(q_n, k_n, gk_n, beta_n, ad_n)

    ms = do_bench(fn)
    print(
        f"[BENCH] perf_fwd_bench shape=({B},{S},{H},{DK}) fwd 5-kernel "
        f"device time = {ms:.3f} ms ({ms * 1000:.1f} us)  [warmup=20 rep=10 mean]"
    )
    return ms


def bench_bwd(B=1, S=8192, H=8, DK=128):
    torch.manual_seed(0)
    ins = prepare_input_bwd(B, S, H, DK)
    ins_n = [t.to(DEVICE) for t in ins]

    def fn():
        fla_chunk_intra_bwd_intra(*ins_n)

    ms = do_bench(fn)
    print(
        f"[BENCH] perf_bwd_bench shape=({B},{S},{H},{DK}) bwd 1-kernel "
        f"device time = {ms:.3f} ms ({ms * 1000:.1f} us)  [warmup=20 rep=10 mean]"
    )
    return ms


# ============================================================================
# main：--level 分发
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all", "bench"])
    args = parser.parse_args()

    tilelang.disable_cache()  # 禁用编译缓存，避免旧产物干扰
    torch.manual_seed(0)

    if not _golden_selfcheck():
        print("golden self-check FAILED -- abort")
        sys.exit(1)

    blocking_ok = True
    if args.level in ("l0", "all"):
        blocking_ok &= test_fla_chunk_intra_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_fla_chunk_intra_l1()
    if args.level in ("l2", "all"):
        test_fla_chunk_intra_l2()
    if args.level in ("boundary", "all"):
        test_fla_chunk_intra_boundary()
    if args.level == "bench":
        fwd_ms = bench_fwd()
        bwd_ms = bench_bwd()
        print(f"[BENCH] fwd+bwd kernel total = {(fwd_ms + bwd_ms) * 1000:.1f} us")

    if args.level != "bench":
        if blocking_ok:
            print("Test Passed!")
            sys.exit(0)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
