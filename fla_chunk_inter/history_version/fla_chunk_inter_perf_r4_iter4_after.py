"""fla_chunk_inter: KDA chunked backward inter-chunk gradients (dq/dk/dw/dg).

Single fused CV kernel, Developer mode. Stage 3 R2 iteration 1 restructure
(NK-merge: one task per (batch, chunk, head) covering ALL DK blocks):

* TASK = (b, chunk, head). The former grid axis split tasks per dk-block
  (GRID = NK * NT * B * H); v/do/dv have no DK axis, so the NK sibling tasks
  of one (chunk, head) each re-loaded the SAME [BS, DV] tiles - at the bench
  shape half of the 96 KB/iteration v/do/dv MTE2 traffic was pure re-read,
  and the six A-operand (left) fp16 relays + split chains were re-produced
  per sibling. The merge loads v/do/dv ONCE per task, splits/relays the left
  operands ONCE, and the Cube side keeps the A operands in L1 across the
  per-bk gemm groups (B operands are relayed per bk). V-side mte2/mte3 drop
  480 -> 384 KB per (chunk, head) pair, the split chain loses the duplicated
  left-operand casts, and the C-side relay reads drop 320 -> 224 KB.
* The per-bk sequences are UNROLLED at trace time (NK is a Python int), so
  for NK == 1 the kernel degenerates statement-for-statement to the R1
  final form (identical layout/order) - the merge only changes behavior at
  NK >= 2, and the L1/L2 suite keeps covering the NK=1 path unchanged.
* Per-bk relays use FLAT slots (the drain-data relay and the four right relays
  are shaped (CORE_EFF, NSLOT * NK, ...) indexed [cid, slot * NK + bk, ...]):
  disjoint GM regions per (task, bk), so bk=1's produce cannot clobber the
  relay data bk=0's Cube consumer has not read yet, and every on-chip
  buffer keeps a single contiguous liveness interval (the R1 planner-fragility
  lessons: loop-external allocs with in-loop first use get pinned; statement
  reordering perturbs the alias plan).
* dq/dk result relays keep the R1 manual cross-core flag protocol with the
  drain-side wait retargeted (R2 iteration 2, "1-late flag"): the drain at
  iteration j reads the relays task j-1 produced, so it waits the flag the
  Cube side raised at iteration j-1 - wait ID 40 + (j+1)%2 pairs with the
  set 40 + (j-1)%2 - instead of the R1 same-iteration (0-late) flag that
  forced the Vector side to stall through the Cube's whole consume chain
  of task j (msprof after the NK-merge showed ~3.1 ms of exposed
  Vector-side wait on exactly that lock-step). A single head-set of flag
  41 before the loop primes the j=0 wait (counting-flag semantics: one
  extra set balances drain(0)'s wait; the R1 "count balance" discipline).
  Slot-safety is unchanged: the drain's slots (NSLOT ring) are not
  rewritten until produce(j+3), well past drain(j) in the Vector stream.
  Alternating IDs + per-iteration set/wait keep the flag capacity < 10
  pending (R1 lag-10 deadlock evidence).
* dgkn is produced on the Cube side (R4) as diag(h @ dh^T) in a [bDK, bDK]
  fp32 fragment relayed through the FUSED drain relay (rows [2*BS, 2*BS+bDK)
  of the flat (task, bk) slots, same manual-flag ring); the drain reads the
  fused relay ONCE (dq | dk | dgm rows in one [2*BS+bDK, bDK] copy - R4
  iter4) and ONE T.tile.gather on constant prologue-built byte offsets
  extracts the [bDK] diagonal (the delivery-fix produce-side fold chain is
  deleted).
* threads=2 (the only 40-AIV entry, MIX_AIC_1_2) stays structurally closed:
  the R2 source audit of ascend_vid_reduction.cc shows every UB buffer's
  dim-0 is halved with the vid GM offset applied at (gm_dims - ub_dims) -
  the [bDK, DV] produce operands split along DK while the Cube gemms need
  them whole, and the drain-side 1-D [bDK] gating buffers split along DK
  against the 2-D [BS, bDK] tensors splitting along S. No Developer-mode
  formulation closes this (matches the R1 iter3 empirical corruption).

Delivery fix (2026-11-03): scratch-free vector-op formulation. The memory
planner (ascend_memory_planning.cc) orders UB buffers by VarNode POINTER
key, so the injected shared scratch of T.reduce_sum / T.tile.broadcast
packed relative to dgkn_ring differently per import context (file path /
module name / heap allocation pattern) - the bad packing (scratch below
its destination) crashes deterministically with an aivector Illegal
instruction (vec error 0x700000042). Byte-identical kernel files were green
in one checkout path and crashed in another. ALL scratch-consuming ops are
therefore ELIMINATED and replaced by equivalent scratch-free forms
(verified bit-exact per replacement in the delivery-fix probes):

* produce-side dgkn (was reduce_sum dim=1 over [bDK, DV]): in-place product
  (as before) -> ONE hardware transpose to the row-per-v layout [DV, bDK]
  (fold along the OUTER dim has no 32B-offset constraint) -> whole-row
  chunk-adds folding every extra 64-row chunk into rows [0, 64) (a TIR
  range loop with Var-min whole-row regions; the DV%64 tail is a literal
  [rem, bDK] region under a trace-time guard) -> a 6-layer literal halving
  tree fold of rows [0, 64) (BS == 64 is a domain constant) -> one row
  copy into the dgkn ring row. Every layer is a single masked AscendC::Add.
  (SUPERSEDED by the R4 ones-gemm diagonalization above - kept for the
  delivery-fix history; the chain no longer exists in the code.)
* drain-side pp (was reduce_sum dim=0 over [BS, bDK]): the same 6-layer
  literal halving tree fold on dkk_ub rows (dkk = dk*k is first fully
  consumed by dg = q*dq - k*dk, then folded in place).
* the m_last mechanism (LastRow constant input + ml2d/pp2d broadcasts +
  mp2d mask-multiply) is REPLACED by a single last-row add
  dg[BS-1] += pp - bit-identical whenever pp is finite, and CLOSER to the
  golden's row-indexed add when pp overflows (no 0*inf NaN at non-last
  rows; the Triton authority's mask form kept the NaN). LastRow, four UB
  buffers and the per-core hoist copy are deleted.
* the gn gate broadcast (was broadcast(gn2d, gn, axis=0) + sub + exp + mul)
  is replaced by a T.unroll row statement chain
  gnd[t, :] = gn[:] - g[t, :] (one flag setup + per-row Sub re-reading the
  [bDK] gn row - no Broadcast helper, no scratch) followed by the unchanged
  element-wise exp and mul.

Stage 3 R4 (2026-11-03, convergence round): dgkn moves to the Cube side as a
ones-gemm diagonalization. dgkn[k] = sum_v h[k,v]*dh[k,v] = diag(h @ dh^T),
and the Cube side already holds h_hi/h_lo/dh_hi/dh_lo in L1 as the dq/dk
gemm-group operands, so three extra hi/lo gemms produce a [bDK, bDK] fp32
L0C fragment (4 fragments = 64 KB < 128 KB L0C) that fixpipes into a new
dgm_relay (same GM relay pattern, now fused with dq/dk into the drain relay,
covered by the SAME
manual 40+j%2 flag group - the drain's 1-late wait already orders it). The
drain reads the matrix back and extracts the diagonal with ONE
T.tile.gather (constant byte offsets i*(bDK+1)*4 built once in the prologue
by arith_progression(int32) + reinterpretcast(uint32) - the cann-bench
gather production pattern; gather is AscendC NoWorkspace so NO planner
scratch is injected, preserving the D-2 scratch-free discipline). This
DELETES the whole produce-side V chain (in-place product, [DV, bDK]
hardware transpose, whole-row chunk adds, 6-layer halving tree, ring row
copy - ~539 us of Vector-side attribution) and the 32 KB prodT_ub buffer;
dgkn's error class changes from an exact fp32 element chain to the same
hi/lo gemm class as every other gemm output (~2^-22 relative per operand
pair). The prologue's zero-fill anchor is replaced by the LIVE offset
preparation (it keeps the head-set pinned into a non-empty V-side prologue
- an unanchored loop-external set_cross_flag is dropped by the toolchain,
deadlocking drain(0)).

Fixed-core + relay workspace machinery is unchanged from the R1 final form
(family-verified fixed-core recipe; evidence in
perf_tuning/fla_chunk_inter_perf_log.md iterations 0-6 and the R2 log):

* FIXED-CORE grid: T.Kernel(CORE_EFF, threads=1) with ONE in-core T.serial
  task loop (Select-redirected idempotent redo past GRID; drain-first never
  fits: the drain waits a flag the same iteration's produce has not set).
* EXPLICIT relay workspaces (workspace_idx): 10 f16 operand relays, AUTO_CV_SYNC
  pairs their cross-core flags by name, all sites symmetric inside the single
  loop + 1 fused f32 drain relay (ddr_relay: dq | dk | dgkn-matrix rows per
  flat (task, bk) slot, named to ESCAPE the auto FIFO pairing) - the slot ring stays L2-resident (~27 MB at bench shape).
* The DV axis stays a single straight-line pass (R1 iteration 1): one
  full-width [*, DV] load per operand, one relay per hi/lo operand, one
  full-K gemm per hi/lo pair (12 gemms with init=True group heads per bk:
  9 output gemms + the 3-gemm dgkn diagonalization, R4).

Per task, one (B, chunk, head) tile computes over each dk-block bk:

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
- TIR `while` loops and Python-list-indexed TIR loops are not trace-time
  constructs (a `while` on a folded condition hangs the core): every
  shape-dependent schedule here is either a literal statement sequence
  (the two 6-layer folds; BS == 64 is a domain constant) or a TIR range
  loop over whole-row regions with constant extents (the produce-side
  chunk adds; the DV%64 tail is guarded by a trace-time `if`).
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
    out_idx=[19, 20, 21, 22],
    workspace_idx=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    pass_configs=PASS_CONFIGS,
)
def fla_chunk_inter(B, S, H, DK, DV, scale=1.0, chunk_size=64, block_DK=32):
    """KDA chunked bwd inter-chunk gradients, single fixed-core kernel.

    out_idx/workspace_idx use ABSOLUTE indices over the declared 23-tensor
    signature (negative indices misalign - family pitfall). Signature order:
    0-7 inputs (8 operands), 8-18 relay workspaces, 19-22 outputs.
    """
    BS = chunk_size
    NT = S // BS
    NK = DK // block_DK
    # One task per (b, chunk, head): all NK dk-blocks are merged into the
    # task (R2 iteration 1). v/do/dv have no DK axis, so the merged task
    # loads/splits/relays them ONCE instead of NK times, and the Cube side
    # reuses the L1 A operands across the per-bk gemm groups.
    GRID = NT * B * H
    # Fixed-core striped decomposition: CORE_EFF blocks (clamped to the task
    # count so the Select redirect target cid is always in-range), each
    # running a single T.serial loop of L iterations. Real striped tasks of
    # core cid are cid + t*CORE_EFF for t < T_valid (BASE or BASE+1); the +2
    # loop slack covers the 1-late drain and the redirect-redo iterations.
    CORE_EFF = min(CORE_NUM, GRID)
    BASE = GRID // CORE_EFF
    REM = GRID - BASE * CORE_EFF  # noqa: F841 - stripes' tail split (cores < REM
    # get BASE+1 tasks); consumed implicitly by the Select redirect below
    L = BASE + 2  # produce t + drain t-1 for t in [0, BASE], +1 drain-only tail

    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    H_shape = (B, NT, H, DK, DV)
    WS_L = (CORE_EFF, NSLOT, BS, DV)  # f16 left-operand relays [BS, DV]
    WS_R = (CORE_EFF, NSLOT * NK, block_DK, DV)  # f16 right-operand relays [bDK, DV], flat (task, bk) slots
    # f32 fused drain relay (R4 iter4): per (task, bk) rows [0, BS) = dq,
    # [BS, 2*BS) = dk, [2*BS, 2*BS+bDK) = the dgkn matrix - ONE V-side read
    WS_F = (CORE_EFF, NSLOT * NK, 2 * BS + block_DK, block_DK)

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
        ddr_relay: T.Tensor(WS_F, "float32"),  # type: ignore  # fused dq|dk|dgm drain relay (manual flag)
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
            # R4: dgkn = diag(h @ dh^T) accumulates here (3 hi/lo gemms);
            # 4 fragments of [BS/bDK, bDK] fp32 = 64 KB < 128 KB L0C.
            dgm_frag = T.alloc_fragment((block_DK, block_DK), "float32")
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
            # ---- UB: drain-side fused relay domain (R4 iter4): rows [0, BS)
            # = dq, [BS, 2*BS) = dk, [2*BS, 2*BS+bDK) = the dgkn matrix; ONE
            # back-read instead of three (whole-row region views keep every
            # consumer a flat-contiguous op). The diag extraction gather and
            # the prologue offset buffer are as in iter2. ----
            ddr_ub = T.alloc_shared((2 * BS + block_DK, block_DK), "float32")
            dgkn_ub = T.alloc_shared((block_DK,), "float32")
            dgk_off_i32 = T.alloc_shared((block_DK,), "int32")
            dgk_off_u32 = T.alloc_shared((block_DK,), "uint32")
            # ---- UB: drain-side (epilogue) buffers ----
            g_ub = T.alloc_shared((BS, block_DK), "float32")
            gexp_ub = T.alloc_shared((BS, block_DK), "float32")
            # R4: brcb expansion target for the vectorized gn gate (only the
            # block_DK==64 path uses it; [rows, elems_per_block] = 64*8 f32)
            gexp_tmp = T.alloc_shared((block_DK, 8), "float32")
            gn_ub = T.alloc_shared((block_DK,), "float32")
            gnexp_ub = T.alloc_shared((block_DK,), "float32")
            gnd_ub = T.alloc_shared((BS, block_DK), "float32")
            q_ub = T.alloc_shared((BS, block_DK), "float32")
            k_ub = T.alloc_shared((BS, block_DK), "float32")
            dkk_ub = T.alloc_shared((BS, block_DK), "float32")
            dg_ub = T.alloc_shared((BS, block_DK), "float32")

            # ---- prologue anchor + C-side head-set: prime the j=0 drain wait
            # (1-late flag pairing - drain(j) waits the flag raised at
            # iteration j-1; iteration 0 drains redirected garbage and its
            # wait pairs with this one-time set; counting semantics stay
            # balanced). The diagonal-offset preparation is LIVE code (every
            # drain's gather reads dgk_off_u32), and it also pins the
            # head-set into a non-empty V-side prologue: an unanchored
            # loop-external set_cross_flag is silently DROPPED by the pip
            # (site-packages) toolchain build, which deadlocks the first
            # drain wait - the delivery-fix probe evidence. ----
            T.tile.arith_progression(dgk_off_i32, 0, (block_DK + 1) * 4, block_DK)
            T.reinterpretcast(dgk_off_u32, dgk_off_i32, "uint32_t")
            T.set_cross_flag("FIX", 41)

            # ============ single uniform produce+drain loop ============
            for j in T.serial(L):
                tid_p = cid + j * CORE_EFF
                pt = T.if_then_else(tid_p < GRID, tid_p, cid)  # redo cid when past GRID
                tid_d = cid + (j - 1) * CORE_EFF
                dt = T.if_then_else(tid_d < GRID, T.if_then_else(tid_d >= 0, tid_d, cid), cid)
                slot = j % NSLOT
                dslot = (j + NSLOT - 1) % NSLOT

                # ================= produce task pt (slot) =================
                bh = pt % H
                bt = pt // H
                bs = bt % NT
                bb = bt // NT
                s0 = bs * BS
                # -- V: per-bk h/dh loads + splits + right relays + dgkn
                #    (relays use FLAT (task, bk) slots: bk=1's produce cannot
                #    clobber bk=0's relay data before the Cube consumed it) --
                for bk in range(NK):  # trace-time unrolled
                    dk0 = bk * block_DK
                    T.copy(h[bb, bs, bh, dk0 : dk0 + block_DK, 0:DV], h_ub)
                    T.copy(Dh[bb, bs, bh, dk0 : dk0 + block_DK, 0:DV], dh_ub)
                    # -- V: dh split -> relay --
                    T.copy(dh_ub, dh_hi)
                    T.copy(dh_hi, th_hi)
                    T.tile.sub(th_lo, dh_ub, th_hi)
                    T.copy(th_lo, dh_lo)
                    T.copy(dh_hi, workspace_7[cid, slot * NK + bk, 0:block_DK, 0:DV])
                    T.copy(dh_lo, workspace_8[cid, slot * NK + bk, 0:block_DK, 0:DV])
                    # -- V: h split -> relay (shared by the dq and dw gemm groups) --
                    T.copy(h_ub, h_hi)
                    T.copy(h_hi, th_hi)
                    T.tile.sub(th_lo, h_ub, th_hi)
                    T.copy(th_lo, h_lo)
                    T.copy(h_hi, workspace_3[cid, slot * NK + bk, 0:block_DK, 0:DV])
                    T.copy(h_lo, workspace_4[cid, slot * NK + bk, 0:block_DK, 0:DV])
                    # -- R4: dgkn = sum_v h*dh moved to the Cube side as
                    #    diag(h @ dh^T) (3 hi/lo gemms on the L1 operands the
                    #    dq/dk groups already consume) + a drain-side gather
                    #    diagonal extraction - the whole produce-side fold
                    #    chain (product, [DV, bDK] transpose, chunk adds,
                    #    halving tree, ring copy) is deleted. --
                # -- V: do load+split -> relay (dq group left operand; ONCE
                #    per task - shared by every bk's gemm group) --
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
                # -- C: consume the left relays ONCE (A operands held in L1
                #    across every bk's gemm group) --
                T.copy(workspace_1[cid, slot, 0:BS, 0:DV], do_hi_l1)
                T.copy(workspace_2[cid, slot, 0:BS, 0:DV], do_lo_l1)
                T.copy(workspace_5[cid, slot, 0:BS, 0:DV], v_hi_l1)
                T.copy(workspace_6[cid, slot, 0:BS, 0:DV], v_lo_l1)
                T.copy(workspace_9[cid, slot, 0:BS, 0:DV], dvn_hi_l1)
                T.copy(workspace_10[cid, slot, 0:BS, 0:DV], dvn_lo_l1)
                # -- C: per-bk right relays -> 3 gemm groups -> dw store +
                #    dq/dk result relays (each read issued just before its
                #    first use so MTE2 latency overlaps the gemms) --
                for bk in range(NK):  # trace-time unrolled
                    dk0 = bk * block_DK
                    T.copy(workspace_3[cid, slot * NK + bk, 0:block_DK, 0:DV], h_hi_l1)
                    T.gemm_v0(do_hi_l1, h_hi_l1, dq_frag, transpose_B=True, init=True)
                    T.copy(workspace_4[cid, slot * NK + bk, 0:block_DK, 0:DV], h_lo_l1)
                    T.gemm_v0(do_hi_l1, h_lo_l1, dq_frag, transpose_B=True, init=False)
                    T.gemm_v0(do_lo_l1, h_hi_l1, dq_frag, transpose_B=True, init=False)
                    T.copy(workspace_7[cid, slot * NK + bk, 0:block_DK, 0:DV], dh_hi_l1)
                    T.gemm_v0(v_hi_l1, dh_hi_l1, dk_frag, transpose_B=True, init=True)
                    T.copy(workspace_8[cid, slot * NK + bk, 0:block_DK, 0:DV], dh_lo_l1)
                    T.gemm_v0(v_hi_l1, dh_lo_l1, dk_frag, transpose_B=True, init=False)
                    T.gemm_v0(v_lo_l1, dh_hi_l1, dk_frag, transpose_B=True, init=False)
                    T.gemm_v0(dvn_hi_l1, h_hi_l1, dw_frag, transpose_B=True, init=True)
                    T.gemm_v0(dvn_hi_l1, h_lo_l1, dw_frag, transpose_B=True, init=False)
                    T.gemm_v0(dvn_lo_l1, h_hi_l1, dw_frag, transpose_B=True, init=False)
                    # -- C: R4 dgkn group - diag(h @ dh^T) via 3 hi/lo gemms
                    #    on the L1 operands the dq/dk groups already hold
                    #    (no new relays); the drain extracts the diagonal
                    #    from the relayed matrix with one gather. --
                    T.gemm_v0(h_hi_l1, dh_hi_l1, dgm_frag, transpose_B=True, init=True)
                    T.gemm_v0(h_hi_l1, dh_lo_l1, dgm_frag, transpose_B=True, init=False)
                    T.gemm_v0(h_lo_l1, dh_hi_l1, dgm_frag, transpose_B=True, init=False)
                    # -- C: dw direct GM store (fp32->fp32, sign pre-folded) --
                    T.copy(dw_frag, dw[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK])
                    # -- C: dq/dk/dgm frag -> fused drain relay (flat (task,
                    #    bk) slots; one V-side back-read covers all three) --
                    T.copy(dq_frag, ddr_relay[cid, slot * NK + bk, 0:BS, 0:block_DK])
                    T.copy(dk_frag, ddr_relay[cid, slot * NK + bk, BS : 2 * BS, 0:block_DK])
                    T.copy(dgm_frag, ddr_relay[cid, slot * NK + bk, 2 * BS : 2 * BS + block_DK, 0:block_DK])
                # -- C: MANUAL flag (alternating ID; a single auto FIFO ID
                #    would lock-step the two cores) --
                T.set_cross_flag("FIX", 40 + j % 2)

                # ============ drain task dt (slot dslot) ============
                bh = dt % H
                bt = dt // H
                bs = bt % NT
                bb = bt // NT
                s0 = bs * BS
                # 1-late flag: this drain reads task j-1's relays, so it
                # waits the flag the Cube side raised at iteration j-1
                # (pairing: wait 40+(j+1)%2 <-> set 40+(j-1)%2).
                T.wait_cross_flag(40 + (j + 1) % 2, "V")
                for bk in range(NK):  # trace-time unrolled
                    dk0 = bk * block_DK
                    # -- R4 iter4: ONE fused back-read (rows [0, BS) = dq,
                    #    [BS, 2*BS) = dk, [2*BS, +bDK) = the dgkn matrix);
                    #    same 1-late slot discipline, three reads -> one. --
                    T.copy(ddr_relay[cid, dslot * NK + bk, 0 : 2 * BS + block_DK, 0:block_DK], ddr_ub)
                    T.copy(G[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK], g_ub)
                    T.copy(G[bb, s0 + BS - 1, bh, dk0 : dk0 + block_DK], gn_ub)
                    T.copy(Q[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK], q_ub)
                    T.copy(K[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK], k_ub)
                    # ln2 pre-fold: scale g and gn in place once, so the
                    # dk exponent is a plain subtract (gn*ln2 - g*ln2) and
                    # the extra [BS, bDK] mul disappears. Rounding differs
                    # from ((gn-g)*ln2) by ~2e-7 relative - 4 orders below
                    # the rtol budget; +/-200 boundary overflow/underflow
                    # paths are unchanged (same exp arguments up to ulp).
                    T.tile.mul(g_ub, g_ub, LN2)
                    T.tile.mul(gn_ub, gn_ub, LN2)
                    # dq = dq_raw * scale * 2^g
                    T.tile.exp(gexp_ub, g_ub)
                    T.tile.mul(ddr_ub[0:BS, 0:block_DK], ddr_ub[0:BS, 0:block_DK], gexp_ub)
                    if scale != 1.0:  # trace-time fold: skip the identity multiply
                        T.tile.mul(ddr_ub[0:BS, 0:block_DK], ddr_ub[0:BS, 0:block_DK], scale)
                    # dgkn = diag of the relayed matrix (ddr_ub rows
                    # [2*BS, 2*BS+bDK)); dgkn *= 2^gn; dk *= 2^(gn - g)
                    # (R4: the gather replaces the produce-side fold - one
                    # vector gather on the prologue-built byte offsets
                    # i*(bDK+1)*4, no scratch)
                    T.tile.gather(
                        dgkn_ub,
                        ddr_ub[2 * BS : 2 * BS + block_DK, 0:block_DK],
                        dgk_off_u32,
                        0,
                    )
                    T.tile.exp(gnexp_ub, gn_ub)
                    T.tile.mul(dgkn_ub, dgkn_ub, gnexp_ub)
                    # gn gate (delivery fix was: T.unroll row statement chain
                    # - one flag setup + per-row Sub re-reading the [bDK] gn
                    # row, no Broadcast helper, no scratch). R4 vectorizes
                    # the block_DK==64 shape (performance-antipatterns.md
                    # "orthogonal-axis scalar scan" remedy): transpose g to
                    # the row-per-k layout, ONE row_expand_sub (brcb expands
                    # the [bDK] gn row into per-row scalar blocks + a single
                    # masked Sub) computes (g - gn)*ln2 in place, the negation
                    # restores the exact exp((gn-g)*ln2) argument (bitwise
                    # identical to the row-chain form), and a transpose back
                    # yields gnd. 6 statements instead of 64 row Subs.
                    # block_DK==32 rows are 128B - below row_expand's 256B
                    # row window - so that path keeps the delivery-fix row
                    # chain (L1 suite covers it).
                    if block_DK == 64:  # trace-time: 256B row window
                        T.tile.transpose(gexp_ub, g_ub)  # gT (gexp_ub dead after the dq mul)
                        T.tile.row_expand_sub_experiment(gexp_ub, gexp_ub, gn_ub, gexp_tmp)
                        T.tile.mul(gexp_ub, gexp_ub, -1.0)
                        T.tile.exp(gexp_ub, gexp_ub)
                        T.tile.transpose(gnd_ub, gexp_ub)
                    else:
                        for t in T.unroll(BS):
                            T.tile.sub(gnd_ub[t, 0:block_DK], gn_ub, g_ub[t, 0:block_DK])
                        T.tile.exp(gnd_ub, gnd_ub)
                    T.tile.mul(ddr_ub[BS : 2 * BS, 0:block_DK], ddr_ub[BS : 2 * BS, 0:block_DK], gnd_ub)
                    # dg = q*dq - k*dk (dkk = dk*k doubles as the -k*dk
                    # term and is consumed by the sub BEFORE its fold)
                    T.tile.mul(dkk_ub, ddr_ub[BS : 2 * BS, 0:block_DK], k_ub)
                    T.tile.mul(dg_ub, q_ub, ddr_ub[0:BS, 0:block_DK])
                    T.tile.sub(dg_ub, dg_ub, dkk_ub)
                    # pp = sum_t dk*k (scratch-free delivery fix): the same
                    # 6-layer halving tree on dkk_ub rows (fold along the
                    # OUTER dim - row starts stay bDK-aligned, every layer
                    # a single masked AscendC::Add).
                    for t, kk in T.Parallel(32, block_DK):
                        dkk_ub[t, kk] = dkk_ub[t, kk] + dkk_ub[t + 32, kk]
                    for t, kk in T.Parallel(16, block_DK):
                        dkk_ub[t, kk] = dkk_ub[t, kk] + dkk_ub[t + 16, kk]
                    for t, kk in T.Parallel(8, block_DK):
                        dkk_ub[t, kk] = dkk_ub[t, kk] + dkk_ub[t + 8, kk]
                    for t, kk in T.Parallel(4, block_DK):
                        dkk_ub[t, kk] = dkk_ub[t, kk] + dkk_ub[t + 4, kk]
                    for t, kk in T.Parallel(2, block_DK):
                        dkk_ub[t, kk] = dkk_ub[t, kk] + dkk_ub[t + 2, kk]
                    for kk in T.Parallel(block_DK):
                        dkk_ub[0, kk] = dkk_ub[0, kk] + dkk_ub[1, kk]
                    # pp += dgkn; dg[last row] += pp (delivery fix: the
                    # m_last broadcast/mask-multiply chain is replaced by a
                    # single last-row add - bit-identical whenever pp is
                    # finite, and NaN-free at non-last rows when it isn't).
                    T.tile.add(
                        dkk_ub[0, 0:block_DK],
                        dkk_ub[0, 0:block_DK],
                        dgkn_ub[0:block_DK],
                    )
                    T.tile.add(dg_ub[BS - 1, 0:block_DK], dg_ub[BS - 1, 0:block_DK], dkk_ub[0, 0:block_DK])
                    T.copy(ddr_ub[0:BS, 0:block_DK], dq[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK])
                    T.copy(ddr_ub[BS : 2 * BS, 0:block_DK], dk[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK])
                    T.copy(dg_ub, dg[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK])

    return main


def fla_chunk_inter_bwd(q, k, v, g, h, dv, do, dh, scale=1.0, chunk_size=64):
    """Host wrapper (proto schema alias): domain/dtype asserts + kernel call.

    No view/reshape/permute/contiguous on any NPU tensor: the 8 inputs are
    passed through with their native bshd / 5D layouts. Relay workspaces
    are framework-managed (workspace_idx auto-allocation).
    """
    B, S, Hn, DK = q.shape
    DV = v.shape[-1]
    _check_shape(B, S, Hn, DK, DV, chunk_size)
    for name, t in (("q", q), ("k", k), ("v", v), ("g", g), ("h", h), ("dv", dv), ("do", do), ("dh", dh)):
        assert t.dtype == torch.float32, f"fla_chunk_inter: fp32-only operator, got {name}.dtype={t.dtype}"
    # Stage 3 iteration 5: double the DK tile (half the task count, ~-40%
    # total Vector-side bytes) when DK is 64-divisible AND DV >= 128. The DV
    # guard keeps every supported shape on a verified-clean dgkn fold
    # terminal: [32,*] and [64,>=128] are empirically clean, the exact [64,64]
    # combination corrupts (toolchain reduce_2d_v2 Plan/scratch defect,
    # isolated by the DK/DV matrix in perf_log iteration 5). block_DK stays a
    # performance parameter, not a dispatch predicate (family _pick_block_dk
    # precedent in fla_chunk_o).
    block_DK = 64 if (DK % 64 == 0 and DV >= 128) else 32
    kernel = fla_chunk_inter(B=B, S=S, H=Hn, DK=DK, DV=DV, scale=scale, chunk_size=chunk_size, block_DK=block_DK)
    return kernel(q, k, v, g, h, dv, do, dh)
