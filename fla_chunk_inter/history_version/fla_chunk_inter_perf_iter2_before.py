"""fla_chunk_inter: KDA chunked backward inter-chunk gradients (dq/dk/dw/dg).

Single fused CV kernel, Developer mode (auto sync, threads=1, 1D grid with
bk-fastest cid mapping so NK sibling dk-block tasks of one (chunk, head) land
in the same dispatch wave and the v/do/dv re-reads hit L2). Per task, one
(B, chunk, head, dk-block) tile computes:

    dq = (do @ h^T) * scale * 2^g
    dk = (v @ dh^T) * 2^(gn - g)
    dw = -(dv @ h^T)                       (sign folded into the dv operand)
    dg = q*dq - k*dk + m_last * pp,         pp = 2^gn * sum_v(h*dh) + sum_t(dk*k)

Stage 3 structure (iteration 1, family-verified unroll): the previous
T.Pipelined(DV//bDV, num_stages=2) body degenerated into a non-pipelined
sequential loop in generated code (family finding, reproduced in
perf_tuning/fla_chunk_inter_kernel_source_iter0.c), so the DV axis is now a
single straight-line pass: one full-width [*, DV] load per operand, ONE relay
per hi/lo operand (10 relays instead of 20), ONE full-K gemm per hi/lo pair
(9 gemms instead of 18), and a single-shot dgkn reduce without the in-loop
accumulation workaround. Halves the relay/gemm/load op count and ~40% of the
per-task sync stations.

Toolchain-mandated implementation notes (family-verified, see DESIGN section 3):
- fp32 gemm_v0 is broken on this toolchain, so every GEMM operand is split
  into fp16 hi/lo pairs (x = hi + lo) and each GEMM becomes hi@hi + hi@lo +
  lo@hi with fp32 L0C accumulation (relative error ~2^-22 per operand pair).
- exp2 has no Ascend backend: 2^x is computed as exp(x * ln2).
- The dw gemm group shares the h L1 operands with the dq group (h keeps the
  same transpose_B right-operand role in both groups; the toolchain workspace
  pass resolves V->C relay writes by source buffer name, so one UB source may
  only be relayed to a single L1 destination).
- Supported domain (asserted by _check_shape): S % 64 == 0, DK % 32 == 0,
  DV multiple of 16 and >= 64, chunk_size == 64, all fp32.
"""

import torch
import tilelang
from tilelang import language as T

LN2 = 0.6931471805599453  # 2^x = exp(x * ln2)

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # auto C/V split
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,  # auto cross-core sync
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,  # auto intra-core sync
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # UB/L1 lifetime planning
}

_LAST_ROW_CACHE: dict = {}


def get_last_row():
    """m_last row indicator constant [64] fp32 (row 63 = 1.0).

    In-domain last(c) is always chunk-local row 63 (S % 64 == 0), so the
    (t == last(c)) row selection is a fixed constant: built once on CPU,
    single H2D, cached per device.
    """
    dev = "npu"
    if dev not in _LAST_ROW_CACHE:
        last = torch.zeros(64, dtype=torch.float32)
        last[63] = 1.0
        _LAST_ROW_CACHE[dev] = last.to(dev)
    return _LAST_ROW_CACHE[dev]


def _check_shape(B, S, H, DK, DV, chunk_size=64):
    """Assert the supported domain (DESIGN section 1.6, deterministic reject).

    S % 64 != 0 is undefined in the reference implementations (h/dh have no
    state for the trailing partial chunk; Triton i_tg reads out of bounds,
    upstream TileLang silently OOBs) - rejected here deterministically.
    """
    assert chunk_size == 64, f"fla_chunk_inter: chunk_size must be 64, got {chunk_size}"
    assert S % chunk_size == 0, (
        f"fla_chunk_inter: S % 64 == 0 required (chunked data model; S%64!=0 is undefined in the reference implementations), got S={S}"
    )
    assert DK % 32 == 0, f"fla_chunk_inter: DK % 32 == 0 required (block_DK=32), got DK={DK}"
    assert DV % 16 == 0 and DV >= 64, (
        f"fla_chunk_inter: DV must be a multiple of 16 and >= 64 (gemm K=DV needs K>=32 fractal and 32B alignment), got DV={DV}"
    )
    assert B >= 1 and H >= 1, f"fla_chunk_inter: B/H must be >= 1, got B={B}, H={H}"


@tilelang.jit(out_idx=[9, 10, 11, 12], pass_configs=PASS_CONFIGS)
def fla_chunk_inter(B, S, H, DK, DV, scale=1.0, chunk_size=64, block_DK=32):
    """KDA chunked bwd inter-chunk gradients, single kernel.

    out_idx uses absolute indices (LastRow sits at 8; negative indices would
    misalign against auto-injected workspace arguments - family pitfall).
    """
    BS = chunk_size
    NT = S // BS
    NK = DK // block_DK
    GRID = NK * NT * B * H

    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    H_shape = (B, NT, H, DK, DV)

    @T.prim_func
    def main(
        Q: T.Tensor(K_shape, "float32"),  # type: ignore
        K: T.Tensor(K_shape, "float32"),  # type: ignore
        V: T.Tensor(V_shape, "float32"),  # type: ignore
        G: T.Tensor(K_shape, "float32"),  # type: ignore
        h: T.Tensor(H_shape, "float32"),  # type: ignore
        dv: T.Tensor(V_shape, "float32"),  # type: ignore
        DO: T.Tensor(V_shape, "float32"),  # type: ignore
        Dh: T.Tensor(H_shape, "float32"),  # type: ignore
        LastRow: T.Tensor((BS,), "float32"),  # type: ignore
        dq: T.Tensor(K_shape, "float32"),  # type: ignore
        dk: T.Tensor(K_shape, "float32"),  # type: ignore
        dw: T.Tensor(K_shape, "float32"),  # type: ignore
        dg: T.Tensor(K_shape, "float32"),  # type: ignore
    ):
        with T.Kernel(GRID, threads=1, is_npu=True) as (cid):
            # bk fastest: NK sibling tasks of one (chunk, head) get adjacent cids
            bk = cid % NK
            bsh = cid // NK
            bs = bsh % NT
            bbh = bsh // NT
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS
            dk0 = bk * block_DK

            # ---- L0C accumulators (fp32) ----
            dq_frag = T.alloc_fragment((BS, block_DK), "float32")
            dk_frag = T.alloc_fragment((BS, block_DK), "float32")
            dw_frag = T.alloc_fragment((BS, block_DK), "float32")
            # ---- UB: fp32 load domain (full-DV single pass) ----
            h_ub = T.alloc_shared((block_DK, DV), "float32")
            dh_ub = T.alloc_shared((block_DK, DV), "float32")  # reused as h*dh in place
            v_ub = T.alloc_shared((BS, DV), "float32")
            do_ub = T.alloc_shared((BS, DV), "float32")
            dvn_ub = T.alloc_shared((BS, DV), "float32")
            # ---- UB: fp32 split temps (shared across operands) ----
            t_hi = T.alloc_shared((BS, DV), "float32")
            t_lo = T.alloc_shared((BS, DV), "float32")
            th_hi = T.alloc_shared((block_DK, DV), "float32")
            th_lo = T.alloc_shared((block_DK, DV), "float32")
            # ---- UB: fp16 hi/lo staging ----
            do_hi = T.alloc_shared((BS, DV), "float16")
            do_lo = T.alloc_shared((BS, DV), "float16")
            v_hi = T.alloc_shared((BS, DV), "float16")
            v_lo = T.alloc_shared((BS, DV), "float16")
            dvn_hi = T.alloc_shared((BS, DV), "float16")
            dvn_lo = T.alloc_shared((BS, DV), "float16")
            h_hi = T.alloc_shared((block_DK, DV), "float16")
            h_lo = T.alloc_shared((block_DK, DV), "float16")
            dh_hi = T.alloc_shared((block_DK, DV), "float16")
            dh_lo = T.alloc_shared((block_DK, DV), "float16")
            # ---- L1: gemm operands (fp16) ----
            do_hi_l1 = T.alloc_shared((BS, DV), "float16")
            do_lo_l1 = T.alloc_shared((BS, DV), "float16")
            v_hi_l1 = T.alloc_shared((BS, DV), "float16")
            v_lo_l1 = T.alloc_shared((BS, DV), "float16")
            dvn_hi_l1 = T.alloc_shared((BS, DV), "float16")
            dvn_lo_l1 = T.alloc_shared((BS, DV), "float16")
            h_hi_l1 = T.alloc_shared((block_DK, DV), "float16")
            h_lo_l1 = T.alloc_shared((block_DK, DV), "float16")
            dh_hi_l1 = T.alloc_shared((block_DK, DV), "float16")
            dh_lo_l1 = T.alloc_shared((block_DK, DV), "float16")
            # ---- UB: dgkn accumulator ----
            dgkn_ub = T.alloc_shared((block_DK,), "float32")
            # ---- UB: epilogue (fp32; lifetimes do not overlap the body side) ----
            g_ub = T.alloc_shared((BS, block_DK), "float32")
            gexp_ub = T.alloc_shared((BS, block_DK), "float32")
            gn_ub = T.alloc_shared((block_DK,), "float32")
            gnexp_ub = T.alloc_shared((block_DK,), "float32")
            gn2d_ub = T.alloc_shared((BS, block_DK), "float32")
            gnd_ub = T.alloc_shared((BS, block_DK), "float32")
            q_ub = T.alloc_shared((BS, block_DK), "float32")
            k_ub = T.alloc_shared((BS, block_DK), "float32")
            dq_ub = T.alloc_shared((BS, block_DK), "float32")
            dk_ub = T.alloc_shared((BS, block_DK), "float32")
            dkk_ub = T.alloc_shared((BS, block_DK), "float32")
            dg_ub = T.alloc_shared((BS, block_DK), "float32")
            pp_ub = T.alloc_shared((block_DK,), "float32")
            pp2d_ub = T.alloc_shared((BS, block_DK), "float32")
            ml_ub = T.alloc_shared((BS,), "float32")
            ml2d_ub = T.alloc_shared((BS, block_DK), "float32")
            mp2d_ub = T.alloc_shared((BS, block_DK), "float32")

            # ---- single straight-line pass over the full DV axis ----
            T.copy(h[bb, bs, bh, dk0 : dk0 + block_DK, 0:DV], h_ub)
            T.copy(Dh[bb, bs, bh, dk0 : dk0 + block_DK, 0:DV], dh_ub)
            # dh split first (raw dh consumed by the split), then relay to L1
            T.copy(dh_ub, dh_hi)
            T.copy(dh_hi, th_hi)
            T.tile.sub(th_lo, dh_ub, th_hi)
            T.copy(th_lo, dh_lo)
            T.copy(dh_hi, dh_hi_l1)
            T.copy(dh_lo, dh_lo_l1)
            # dgkn = sum_v h*dh (single full-width reduce; dh_ub reused in
            # place as h*dh, raw dh already consumed by the split above)
            T.tile.mul(dh_ub, dh_ub, h_ub)
            T.reduce_sum(dh_ub, dgkn_ub, dim=1, clear=True)
            # h split -> relay (shared by the dq and dw gemm groups: h
            # keeps the same transpose_B right-operand role in both, and a
            # UB source may only be relayed to ONE L1 destination -
            # the workspace pass resolves the write side by source name)
            T.copy(h_ub, h_hi)
            T.copy(h_hi, th_hi)
            T.tile.sub(th_lo, h_ub, th_hi)
            T.copy(th_lo, h_lo)
            T.copy(h_hi, h_hi_l1)
            T.copy(h_lo, h_lo_l1)
            # v: load -> split -> relay
            T.copy(V[bb, s0 : s0 + BS, bh, 0:DV], v_ub)
            T.copy(v_ub, v_hi)
            T.copy(v_hi, t_hi)
            T.tile.sub(t_lo, v_ub, t_hi)
            T.copy(t_lo, v_lo)
            T.copy(v_hi, v_hi_l1)
            T.copy(v_lo, v_lo_l1)
            # do: load -> split -> relay
            T.copy(DO[bb, s0 : s0 + BS, bh, 0:DV], do_ub)
            T.copy(do_ub, do_hi)
            T.copy(do_hi, t_hi)
            T.tile.sub(t_lo, do_ub, t_hi)
            T.copy(t_lo, do_lo)
            T.copy(do_hi, do_hi_l1)
            T.copy(do_lo, do_lo_l1)
            # dv: load -> negate (dw sign folding) -> split -> relay
            T.copy(dv[bb, s0 : s0 + BS, bh, 0:DV], dvn_ub)
            T.tile.mul(dvn_ub, dvn_ub, -1.0)
            T.copy(dvn_ub, dvn_hi)
            T.copy(dvn_hi, t_hi)
            T.tile.sub(t_lo, dvn_ub, t_hi)
            T.copy(t_lo, dvn_lo)
            T.copy(dvn_hi, dvn_hi_l1)
            T.copy(dvn_lo, dvn_lo_l1)
            # 9 gemms (3 groups x hi/lo triple, L0C fp32 accumulation, full-K
            # single shot: every group head initializes its accumulator)
            T.gemm_v0(do_hi_l1, h_hi_l1, dq_frag, transpose_B=True, init=True)
            T.gemm_v0(do_hi_l1, h_lo_l1, dq_frag, transpose_B=True, init=False)
            T.gemm_v0(do_lo_l1, h_hi_l1, dq_frag, transpose_B=True, init=False)
            T.gemm_v0(v_hi_l1, dh_hi_l1, dk_frag, transpose_B=True, init=True)
            T.gemm_v0(v_hi_l1, dh_lo_l1, dk_frag, transpose_B=True, init=False)
            T.gemm_v0(v_lo_l1, dh_hi_l1, dk_frag, transpose_B=True, init=False)
            T.gemm_v0(dvn_hi_l1, h_hi_l1, dw_frag, transpose_B=True, init=True)
            T.gemm_v0(dvn_hi_l1, h_lo_l1, dw_frag, transpose_B=True, init=False)
            T.gemm_v0(dvn_lo_l1, h_hi_l1, dw_frag, transpose_B=True, init=False)

            # ---- epilogue: C-side dw direct store + dq/dk C->V ----
            T.copy(dw_frag, dw[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK])
            T.copy(dq_frag, dq_ub)
            T.copy(dk_frag, dk_ub)
            # ---- V-side tail ----
            T.copy(G[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK], g_ub)
            T.copy(G[bb, s0 + BS - 1, bh, dk0 : dk0 + block_DK], gn_ub)
            T.copy(LastRow[0:BS], ml_ub)
            T.copy(Q[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK], q_ub)
            T.copy(K[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK], k_ub)
            # dq = dq_raw * scale * 2^g
            T.tile.mul(gexp_ub, g_ub, LN2)
            T.tile.exp(gexp_ub, gexp_ub)
            T.tile.mul(dq_ub, dq_ub, gexp_ub)
            if scale != 1.0:  # trace-time fold: skip the identity multiply
                T.tile.mul(dq_ub, dq_ub, scale)
            # dgkn *= 2^gn; dk *= 2^(gn - g)
            T.tile.mul(gnexp_ub, gn_ub, LN2)
            T.tile.exp(gnexp_ub, gnexp_ub)
            T.tile.mul(dgkn_ub, dgkn_ub, gnexp_ub)
            T.tile.broadcast(gn2d_ub, gn_ub, axis=0)
            T.tile.sub(gnd_ub, gn2d_ub, g_ub)
            T.tile.mul(gnd_ub, gnd_ub, LN2)
            T.tile.exp(gnd_ub, gnd_ub)
            T.tile.mul(dk_ub, dk_ub, gnd_ub)
            # pp = dgkn + sum_t dk*k (dkk doubles as dg's -k*dk term)
            T.tile.mul(dkk_ub, dk_ub, k_ub)
            T.reduce_sum(dkk_ub, pp_ub, dim=0, clear=True)
            T.tile.add(pp_ub, pp_ub, dgkn_ub)
            # dg = q*dq - k*dk + m_last*pp
            T.tile.broadcast(pp2d_ub, pp_ub, axis=0)
            T.tile.broadcast(ml2d_ub, ml_ub, axis=1)
            T.tile.mul(mp2d_ub, pp2d_ub, ml2d_ub)
            T.tile.mul(dg_ub, q_ub, dq_ub)
            T.tile.sub(dg_ub, dg_ub, dkk_ub)
            T.tile.add(dg_ub, dg_ub, mp2d_ub)
            # ---- 3 GM stores (dw already stored C-side) ----
            T.copy(dq_ub, dq[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK])
            T.copy(dk_ub, dk[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK])
            T.copy(dg_ub, dg[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK])

    return main


def fla_chunk_inter_bwd(q, k, v, g, h, dv, do, dh, scale=1.0, chunk_size=64):
    """Host wrapper (proto schema alias): domain/dtype asserts + kernel call.

    No view/reshape/permute/contiguous on any NPU tensor: the 8 inputs are
    passed through with their native bshd / 5D layouts and LastRow is a
    per-device cached constant input.
    """
    B, S, Hn, DK = q.shape
    DV = v.shape[-1]
    _check_shape(B, S, Hn, DK, DV, chunk_size)
    for name, t in (("q", q), ("k", k), ("v", v), ("g", g), ("h", h), ("dv", dv), ("do", do), ("dh", dh)):
        assert t.dtype == torch.float32, f"fla_chunk_inter: fp32-only operator, got {name}.dtype={t.dtype}"
    kernel = fla_chunk_inter(B=B, S=S, H=Hn, DK=DK, DV=DV, scale=scale, chunk_size=chunk_size)
    return kernel(q, k, v, g, h, dv, do, dh, get_last_row())
