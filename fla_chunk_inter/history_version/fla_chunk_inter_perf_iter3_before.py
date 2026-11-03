"""fla_chunk_inter: KDA chunked backward inter-chunk gradients (dq/dk/dw/dg).

Single fused CV kernel, Developer mode. Stage 3 iteration 2 restructure
(family-verified fixed-core recipe, fla_chunk_o iter3/final form; evidence
in perf_tuning/fla_chunk_inter_perf_log.md iterations 0-2):

* FIXED-CORE grid: T.Kernel(CORE_EFF, threads=1) with ONE in-core
  T.serial task loop (antipatterns "launch core 数" item A: the previous
  32768-block launch paid per-block dispatch and, critically, sized every
  auto-injected V->C relay workspace per cid - a ~4.5 GB footprint that
  defeated L2, pushing ~9 GB of relay r+w through HBM).
* EXPLICIT relay workspaces (workspace_idx): 10 f16 operand relays
  (workspace_1..10, named so AUTO_CV_SYNC handles their cross-core flags;
  sites are symmetric inside the single loop) + 2 f32 result relays
  (dq_relay/dk_relay, named to ESCAPE the auto FIFO pairing - their
  head/tail trip counts are asymmetric by design, see below) shaped
  (CORE_EFF, NSLOT, ...) - an ~11.8 MB slot ring resident in the 192 MB L2.
* Single uniform loop, branch-free: iteration j produces task
  cid + j*CORE_EFF (Select-redirected to cid - an idempotent recompute -
  when the tid runs past GRID) and then drains task cid + (j-1)*CORE_EFF
  (redirected to cid when out of range: j=0 reads an unwritten slot and
  writes garbage to the core's own region, overwritten by the real drain
  at j=1; the last iterations drain the redo results - identical values).
  Every workspace site lives inside the ONE loop, so AUTO_CV_SYNC pairs
  them cleanly, and every buffer's FIRST USE is inside the loop, so the
  memory planner's loop-aware liveness extension does not pin them across
  the whole loop (the earlier peeled-head variant segfaulted on exactly
  that - sum of produce+drain UB ~457KB > 196352B).
* dq/dk result relay uses MANUAL cross-core flags with alternating IDs
  40 + j%2 (family recipe: a single-ID FIFO pairing would force a
  same-iteration lock-step between the two cores; the per-ID set/wait
  alternate 1:1 inside each iteration, deadlock-free, and bound the Cube
  lag to one iteration which keeps the NSLOT=4 ring safe).
* dgkn is carried across the drain deferral inside a [NSLOT, bDK] UB ring
  (dgkn_ring), not a single buffer (produce(j) would clobber drain(j-1)'s
  accumulator otherwise).
* Produce-before-drain within each iteration (drain-first deadlocks: the
  drain waits for a flag the same iteration's produce has not set yet).
* The DV axis is a single straight-line pass (iteration 1): one full-width
  [*, DV] load per operand, ONE relay per hi/lo operand, ONE full-K gemm
  per hi/lo pair (9 gemms, init=True group heads), single-shot dgkn reduce.

Per task, one (B, chunk, head, dk-block) tile computes:

    dq = (do @ h^T) * scale * 2^g
    dk = (v @ dh^T) * 2^(gn - g)
    dw = -(dv @ h^T)                       (sign folded into the dv operand)
    dg = q*dq - k*dk + m_last * pp,         pp = 2^gn * sum_v(h*dh) + sum_t(dk*k)

Toolchain-mandated implementation notes (family-verified, see DESIGN section 3):
- fp32 gemm_v0 is broken on this toolchain, so every GEMM operand is split
  into fp16 hi/lo pairs (x = hi + lo) and each GEMM becomes hi@hi + hi@lo +
  lo@hi with fp32 L0C accumulation (relative error ~2^-22 per operand pair).
- exp2 has no Ascend backend: 2^x is computed as exp(x * ln2).
- A UB source may only be relayed to a single L1 destination (the workspace
  pass resolves V->C relay writes by source buffer name), so the dw group
  shares the dq group's h relay (same transpose_B right-operand role).
- Supported domain (asserted by _check_shape): S % 64 == 0, DK % 32 == 0,
  DV multiple of 16 and >= 64, chunk_size == 64, all fp32.
"""

import torch
import tilelang
from tilelang import language as T

LN2 = 0.6931471805599453  # 2^x = exp(x * ln2)

CORE_NUM = 20  # Ascend 910B3 AI Cube cores (fixed-core grid ceiling)
NSLOT = 4  # relay slot ring depth (>=2: cube lag bound 1 + margin)

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # auto C/V split
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,  # cross-core flags on workspace_* relays
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


@tilelang.jit(
    out_idx=[21, 22, 23, 24],
    workspace_idx=[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    pass_configs=PASS_CONFIGS,
)
def fla_chunk_inter(B, S, H, DK, DV, scale=1.0, chunk_size=64, block_DK=32):
    """KDA chunked bwd inter-chunk gradients, single fixed-core kernel.

    out_idx/workspace_idx use ABSOLUTE indices over the declared 25-tensor
    signature (negative indices misalign - family pitfall). Signature order:
    0-8 inputs (8 operands + LastRow), 9-20 relay workspaces, 21-24 outputs.
    """
    BS = chunk_size
    NT = S // BS
    NK = DK // block_DK
    GRID = NK * NT * B * H
    # Fixed-core striped decomposition: CORE_EFF blocks (clamped to the task
    # count so the Select redirect target cid is always in-range), each
    # running a single T.serial loop of L iterations. Real striped tasks of
    # core cid are cid + t*CORE_EFF for t < T_valid (BASE or BASE+1); the +2
    # loop slack covers the 1-late drain and the redirect-redo iterations.
    CORE_EFF = min(CORE_NUM, GRID)
    BASE = GRID // CORE_EFF
    REM = GRID - BASE * CORE_EFF
    L = BASE + 2  # produce t + drain t-1 for t in [0, BASE], +1 drain-only tail

    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    H_shape = (B, NT, H, DK, DV)
    WS_L = (CORE_EFF, NSLOT, BS, DV)  # f16 left-operand relays [BS, DV]
    WS_R = (CORE_EFF, NSLOT, block_DK, DV)  # f16 right-operand relays [bDK, DV]
    WS_O = (CORE_EFF, NSLOT, BS, block_DK)  # f32 result relays [BS, bDK]

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
        workspace_1: T.Tensor(WS_L, "float16"),  # type: ignore  # do_hi
        workspace_2: T.Tensor(WS_L, "float16"),  # type: ignore  # do_lo
        workspace_3: T.Tensor(WS_R, "float16"),  # type: ignore  # h_hi
        workspace_4: T.Tensor(WS_R, "float16"),  # type: ignore  # h_lo
        workspace_5: T.Tensor(WS_L, "float16"),  # type: ignore  # v_hi
        workspace_6: T.Tensor(WS_L, "float16"),  # type: ignore  # v_lo
        workspace_7: T.Tensor(WS_R, "float16"),  # type: ignore  # dh_hi
        workspace_8: T.Tensor(WS_R, "float16"),  # type: ignore  # dh_lo
        workspace_9: T.Tensor(WS_L, "float16"),  # type: ignore  # dvn_hi
        workspace_10: T.Tensor(WS_L, "float16"),  # type: ignore  # dvn_lo
        dq_relay: T.Tensor(WS_O, "float32"),  # type: ignore  # dq results (manual flag)
        dk_relay: T.Tensor(WS_O, "float32"),  # type: ignore  # dk results (manual flag)
        dq: T.Tensor(K_shape, "float32"),  # type: ignore
        dk: T.Tensor(K_shape, "float32"),  # type: ignore
        dw: T.Tensor(K_shape, "float32"),  # type: ignore
        dg: T.Tensor(K_shape, "float32"),  # type: ignore
    ):
        with T.Kernel(CORE_EFF, threads=1, is_npu=True) as (cid):
            # ---- L0C accumulators (fp32; per-task, reused every iteration) ----
            dq_frag = T.alloc_fragment((BS, block_DK), "float32")
            dk_frag = T.alloc_fragment((BS, block_DK), "float32")
            dw_frag = T.alloc_fragment((BS, block_DK), "float32")
            # ---- UB: produce-side fp32 load domain (full-DV single pass) ----
            h_ub = T.alloc_shared((block_DK, DV), "float32")
            dh_ub = T.alloc_shared((block_DK, DV), "float32")  # reused as h*dh
            v_ub = T.alloc_shared((BS, DV), "float32")
            do_ub = T.alloc_shared((BS, DV), "float32")
            dvn_ub = T.alloc_shared((BS, DV), "float32")
            # ---- UB: fp32 split temps (shared across operands) ----
            t_hi = T.alloc_shared((BS, DV), "float32")
            t_lo = T.alloc_shared((BS, DV), "float32")
            th_hi = T.alloc_shared((block_DK, DV), "float32")
            th_lo = T.alloc_shared((block_DK, DV), "float32")
            # ---- UB: fp16 hi/lo staging (dead once relayed) ----
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
            # ---- L1: gemm operands (fp16; per-task, reused) ----
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
            # ---- UB: dgkn ring (produced in task j, drained in task j+1) ----
            dgkn_ring = T.alloc_shared((NSLOT, block_DK), "float32")
            # ---- UB: drain-side (epilogue) buffers ----
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

            # ---- loop-invariant hoist: LastRow constant once per core ----
            T.copy(LastRow[0:BS], ml_ub)

            # ============ single uniform produce+drain loop ============
            for j in T.serial(L):
                tid_p = cid + j * CORE_EFF
                pt = T.if_then_else(tid_p < GRID, tid_p, cid)  # redo cid when past GRID
                tid_d = cid + (j - 1) * CORE_EFF
                dt = T.if_then_else(
                    tid_d < GRID, T.if_then_else(tid_d >= 0, tid_d, cid), cid
                )
                slot = j % NSLOT
                dslot = (j + NSLOT - 1) % NSLOT

                # ================= produce task pt (slot) =================
                bk = pt % NK
                bsh = pt // NK
                bs = bsh % NT
                bbh = bsh // NT
                bb = bbh // H
                bh = bbh % H
                s0 = bs * BS
                dk0 = bk * block_DK
                # -- V: h/dh loads (issued together; MTE2 pipelines them) --
                T.copy(h[bb, bs, bh, dk0 : dk0 + block_DK, 0:DV], h_ub)
                T.copy(Dh[bb, bs, bh, dk0 : dk0 + block_DK, 0:DV], dh_ub)
                # -- V: dh split -> relay --
                T.copy(dh_ub, dh_hi)
                T.copy(dh_hi, th_hi)
                T.tile.sub(th_lo, dh_ub, th_hi)
                T.copy(th_lo, dh_lo)
                T.copy(dh_hi, workspace_7[cid, slot, 0:block_DK, 0:DV])
                T.copy(dh_lo, workspace_8[cid, slot, 0:block_DK, 0:DV])
                # -- V: h split -> relay (shared by the dq and dw gemm groups) --
                T.copy(h_ub, h_hi)
                T.copy(h_hi, th_hi)
                T.tile.sub(th_lo, h_ub, th_hi)
                T.copy(th_lo, h_lo)
                T.copy(h_hi, workspace_3[cid, slot, 0:block_DK, 0:DV])
                T.copy(h_lo, workspace_4[cid, slot, 0:block_DK, 0:DV])
                # -- V: dgkn = sum_v h*dh (dh_ub reused in place; ring slot) --
                T.tile.mul(dh_ub, dh_ub, h_ub)
                T.reduce_sum(dh_ub, dgkn_ring[slot, 0:block_DK], dim=1, clear=True)
                # -- V: do load+split -> relay (dq group left operand) --
                T.copy(DO[bb, s0 : s0 + BS, bh, 0:DV], do_ub)
                T.copy(do_ub, do_hi)
                T.copy(do_hi, t_hi)
                T.tile.sub(t_lo, do_ub, t_hi)
                T.copy(t_lo, do_lo)
                T.copy(do_hi, workspace_1[cid, slot, 0:BS, 0:DV])
                T.copy(do_lo, workspace_2[cid, slot, 0:BS, 0:DV])
                # -- V: v load+split -> relay (dk group left operand) --
                T.copy(V[bb, s0 : s0 + BS, bh, 0:DV], v_ub)
                T.copy(v_ub, v_hi)
                T.copy(v_hi, t_hi)
                T.tile.sub(t_lo, v_ub, t_hi)
                T.copy(t_lo, v_lo)
                T.copy(v_hi, workspace_5[cid, slot, 0:BS, 0:DV])
                T.copy(v_lo, workspace_6[cid, slot, 0:BS, 0:DV])
                # -- V: dv load -> negate (dw sign folding) -> split -> relay --
                T.copy(dv[bb, s0 : s0 + BS, bh, 0:DV], dvn_ub)
                T.tile.mul(dvn_ub, dvn_ub, -1.0)
                T.copy(dvn_ub, dvn_hi)
                T.copy(dvn_hi, t_hi)
                T.tile.sub(t_lo, dvn_ub, t_hi)
                T.copy(t_lo, dvn_lo)
                T.copy(dvn_hi, workspace_9[cid, slot, 0:BS, 0:DV])
                T.copy(dvn_lo, workspace_10[cid, slot, 0:BS, 0:DV])
                # -- C: consume relays (each read issued just before its
                # first use so MTE2 latency overlaps the gemms) --
                T.copy(workspace_1[cid, slot, 0:BS, 0:DV], do_hi_l1)
                T.copy(workspace_3[cid, slot, 0:block_DK, 0:DV], h_hi_l1)
                T.gemm_v0(do_hi_l1, h_hi_l1, dq_frag, transpose_B=True, init=True)
                T.copy(workspace_4[cid, slot, 0:block_DK, 0:DV], h_lo_l1)
                T.gemm_v0(do_hi_l1, h_lo_l1, dq_frag, transpose_B=True, init=False)
                T.copy(workspace_2[cid, slot, 0:BS, 0:DV], do_lo_l1)
                T.gemm_v0(do_lo_l1, h_hi_l1, dq_frag, transpose_B=True, init=False)
                T.copy(workspace_5[cid, slot, 0:BS, 0:DV], v_hi_l1)
                T.copy(workspace_7[cid, slot, 0:block_DK, 0:DV], dh_hi_l1)
                T.gemm_v0(v_hi_l1, dh_hi_l1, dk_frag, transpose_B=True, init=True)
                T.copy(workspace_8[cid, slot, 0:block_DK, 0:DV], dh_lo_l1)
                T.gemm_v0(v_hi_l1, dh_lo_l1, dk_frag, transpose_B=True, init=False)
                T.copy(workspace_6[cid, slot, 0:BS, 0:DV], v_lo_l1)
                T.gemm_v0(v_lo_l1, dh_hi_l1, dk_frag, transpose_B=True, init=False)
                T.copy(workspace_9[cid, slot, 0:BS, 0:DV], dvn_hi_l1)
                T.gemm_v0(dvn_hi_l1, h_hi_l1, dw_frag, transpose_B=True, init=True)
                T.gemm_v0(dvn_hi_l1, h_lo_l1, dw_frag, transpose_B=True, init=False)
                T.copy(workspace_10[cid, slot, 0:BS, 0:DV], dvn_lo_l1)
                T.gemm_v0(dvn_lo_l1, h_hi_l1, dw_frag, transpose_B=True, init=False)
                # -- C: dw direct GM store (fp32->fp32, sign pre-folded) --
                T.copy(dw_frag, dw[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK])
                # -- C: dq/dk frag -> result relays; MANUAL flag (alternating
                # ID; a single auto FIFO ID would lock-step the two cores) --
                T.copy(dq_frag, dq_relay[cid, slot, 0:BS, 0:block_DK])
                T.copy(dk_frag, dk_relay[cid, slot, 0:BS, 0:block_DK])
                T.set_cross_flag("FIX", 40 + j % 2)

                # ============ drain task dt (slot dslot) ============
                bk = dt % NK
                bsh = dt // NK
                bs = bsh % NT
                bbh = bsh // NT
                bb = bbh // H
                bh = bbh % H
                s0 = bs * BS
                dk0 = bk * block_DK
                T.wait_cross_flag(40 + j % 2, "V")
                T.copy(dq_relay[cid, dslot, 0:BS, 0:block_DK], dq_ub)
                T.copy(dk_relay[cid, dslot, 0:BS, 0:block_DK], dk_ub)
                T.copy(G[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK], g_ub)
                T.copy(G[bb, s0 + BS - 1, bh, dk0 : dk0 + block_DK], gn_ub)
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
                T.tile.mul(dgkn_ring[dslot, 0:block_DK], dgkn_ring[dslot, 0:block_DK], gnexp_ub)
                T.tile.broadcast(gn2d_ub, gn_ub, axis=0)
                T.tile.sub(gnd_ub, gn2d_ub, g_ub)
                T.tile.mul(gnd_ub, gnd_ub, LN2)
                T.tile.exp(gnd_ub, gnd_ub)
                T.tile.mul(dk_ub, dk_ub, gnd_ub)
                # pp = dgkn + sum_t dk*k (dkk doubles as dg's -k*dk term)
                T.tile.mul(dkk_ub, dk_ub, k_ub)
                T.reduce_sum(dkk_ub, pp_ub, dim=0, clear=True)
                T.tile.add(pp_ub, pp_ub, dgkn_ring[dslot, 0:block_DK])
                # dg = q*dq - k*dk + m_last*pp
                T.tile.broadcast(pp2d_ub, pp_ub, axis=0)
                T.tile.broadcast(ml2d_ub, ml_ub, axis=1)
                T.tile.mul(mp2d_ub, pp2d_ub, ml2d_ub)
                T.tile.mul(dg_ub, q_ub, dq_ub)
                T.tile.sub(dg_ub, dg_ub, dkk_ub)
                T.tile.add(dg_ub, dg_ub, mp2d_ub)
                T.copy(dq_ub, dq[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK])
                T.copy(dk_ub, dk[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK])
                T.copy(dg_ub, dg[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK])

    return main


def fla_chunk_inter_bwd(q, k, v, g, h, dv, do, dh, scale=1.0, chunk_size=64):
    """Host wrapper (proto schema alias): domain/dtype asserts + kernel call.

    No view/reshape/permute/contiguous on any NPU tensor: the 8 inputs are
    passed through with their native bshd / 5D layouts and LastRow is a
    per-device cached constant input. Relay workspaces are framework-managed
    (workspace_idx auto-allocation).
    """
    B, S, Hn, DK = q.shape
    DV = v.shape[-1]
    _check_shape(B, S, Hn, DK, DV, chunk_size)
    for name, t in (("q", q), ("k", k), ("v", v), ("g", g), ("h", h), ("dv", dv), ("do", do), ("dh", dh)):
        assert t.dtype == torch.float32, f"fla_chunk_inter: fp32-only operator, got {name}.dtype={t.dtype}"
    kernel = fla_chunk_inter(B=B, S=S, H=Hn, DK=DK, DV=DV, scale=scale, chunk_size=chunk_size)
    return kernel(q, k, v, g, h, dv, do, dh, get_last_row())
