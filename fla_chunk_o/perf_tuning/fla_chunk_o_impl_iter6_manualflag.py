"""fla_chunk_o: KDA chunked forward output O (single kernel) on Ascend 910B3.

o = inter-chunk (qg @ h) + intra-chunk (tril_incl_diag(A) @ v), where
qg = (q * scale) * 2^g computed as exp(g * ln2) (exp2 has no Ascend
backend mapping). Inputs are consumed-only (A/h produced upstream).

Semantics: ref_gpu/fla_chunk_o.py chunk_gla_fwd_kernel_o (Triton, frozen);
code-level origin: ref_gpu/fla_chunk_o_tilelang_fwd.py (upstream TileLang
CUDA tilelang_chunk_fwd_o).

Developer mode with explicit GM relay workspaces (Stage 3 restructure):
bshd layout consumed via zero-copy 3D views (no host-side materialization,
no aclnn calls in the wrapper).

Stage-3 performance structure (perf_tuning/ for the full ablation data):

* FIXED-CORE grid: T.Kernel(CORE_NUM=20 cube cores, threads=2) with an
  in-core T.serial task loop (antipatterns "launch core 数" 关注项 A).
  8192 logical (chunk, b, h) tasks are striped over 20 cores; one block
  instance per core amortizes the per-block dispatch/TPipe-init overhead
  and lets the MTE2/MTE3/V queues flow across tasks.
* MANUALLY SOFTWARE-PIPELINED task loop: the output chain of task t-1
  (o relay read -> cast -> GM store) runs at the TOP of iteration t,
  decoupled from the produce section of task t, so the Vector side never
  stalls waiting for the Cube side of the SAME task. Double-buffered
  relay slots (t % 2) make the reuse WAR-safe; the alternation
  produce(t) -> gemms(t) -> o-chain(t) -> produce(t+1) ... is transitively
  ordered by the auto-inserted cross-core flags.
* EXPLICIT relay workspaces (workspace_idx): qg_hi/qg_lo/a_mask/o are
  staged through small named GM tensors ([20, 2, ...], ~2.9 MiB total --
  permanently L2-resident) instead of the AUTO_CV_COMBINE auto-injected
  per-block workspaces (~768 MiB per launch). The V->C handoffs keep
  their auto-inserted cross-core flags (AUTO_CV_SYNC matches buffers
  named "workspace_*").
* UNROLLED full-DK inter term: q/g are loaded once per task as [64, DK]
  rows (256B/512B rows) and the hi/lo pair is applied with two K=DK
  gemms (the upstream i_k Pipelined loop never software-pipelined on
  this toolchain: it lowered to a single-buffered sequential loop with
  fully exposed per-iteration MTE2 latency).

Ascend toolchain adaptations (inherited from the same-family fla_wy_fast
delivery, all empirically verified on 910B3):

1. One-dimensional T.Kernel with cid decomposed on-core (3-D kernels
   unsupported; threads restricted to 1/2). threads=2 splits each task's
   Vector rows across the 2 AIV sub-blocks of the core pair (910B3 has
   20 cube + 40 vector cores, so threads=2 keeps all vector cores busy).
2. All gemm_v0 inputs are bf16 (fp32 A/B produces 1e37-class garbage on
   this toolchain). qg is split into a bf16 hi/lo pair and accumulated
   with two gemms (~2^-18 relative on o_inter): a single bf16 cast
   (~2^-9 relative, the Triton semantic authority's b_qg.to(bf16))
   measurably trips the bf16 mixed tolerance on cancellation elements
   (|o| << |o_inter|, L0 ratio 0.988 < 0.99) -- DESIGN SS9.3 fallback.
3. Every bf16<->fp32 conversion happens UB-internally via a two-step copy
   chain; GM<->UB cast-on-copy codegen is broken (mismatched template).
4. L0C(fp32) -> GM(bf16) direct fixpipe is broken: the output relays
   fp32 through workspace_4 and casts on the Vector side.
5. The tril mask is a constant Msk[64,64] GM input (inclusive diagonal,
   tril(ones, 0)) multiplied in the fp32 domain via T.tile.mul --
   T.if_then_else is unavailable in tile ops. Exact in bf16 (0/1 mask).
   The Msk load is hoisted out of the task loop (loop-invariant).
6. Element math uses buffer-level T.tile.* instructions only (element
   T.Parallel loops are the -26%..-71%/kernel antipattern).
7. Q/G/A/O are consumed through zero-copy 3D views [B, S, H*D] of the
   bshd tensors: the vid-reduction pass adds the sub-block row offset to
   GM dimension (gm_dims - ub_dims); with 4D bshd slices that lands on
   the H index (reads a different head), while 3D row slices put the S
   rows exactly there. V/Hstate are consumed Cube-side (GM->L1, not
   vid-split) and stay 4D/5D.
"""

import tilelang
import torch
from tilelang import language as T

# 2^g is computed as exp(g * ln2): exp2 has no Ascend backend mapping.
LN2 = 0.6931471805599453

# Ascend 910B3: 20 AI Cube cores (+ 40 AI Vector cores). The fixed-core
# grid launches one block per cube core; threads=2 pairs each with its 2
# vector sub-blocks.
CORE_NUM = 20

# Relay slot ring per core (t % NSLOT). NSLOT=4 supports the 2-task-deferred
# output drain: the Vector side drains task t-2 while the Cube side is still
# processing t-1/t, so neither core ever waits for the other's current phase
# (msprof showed both cores ~80% idle on cross-core flag waits before this).
NSLOT = 4

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # automatic C/V split
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,  # cross-core flags on the relays
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,  # intra-core pipe sync
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # UB/L1 lifetime planning
}


def _pick_block_dk(dim: int) -> int:
    """Domain gate for DK (historically the exact-2 Pipelined block split).

    The Stage-3 kernel computes the inter term with full-K gemms (no i_k
    loop), so any 16-multiple K >= 32 would be fractal-legal; the
    {64, 96, 128} domain is kept unchanged from the delivered contract
    (DESIGN SS1.6 dispatch coverage; L2 rejection tests pin it).
    """
    blk = dim // 2
    if dim % 2 == 0 and blk in (32, 48, 64):
        return blk
    raise ValueError(
        f"DK={dim} unsupported: the delivered contract restricts DK to"
        " one of {64, 96, 128} (documented limitation, see DESIGN SS1.6)"
    )


def _pick_block_dv(dim: int) -> int:
    """block_DV = DV single block (IO-unique: every tensor read/written once).

    Constraints: the intra/inter gemm N = block_DV must satisfy the bf16
    gemm_v0 N >= 16 fractal; a single block keeps the o accumulator
    [64, DV] fp32 within the 128KB L0C budget -> DV <= 128. Larger DV
    would need a nested bv loop (no precedent, out of scope).
    """
    if dim % 16 == 0 and 16 <= dim <= 128:
        return dim
    raise ValueError(
        f"DV={dim} unsupported: DV must be a 16-multiple in [16, 128]"
        " (bf16 gemm N-fractal + single-block L0C budget; documented"
        " limitation)"
    )


# out_idx/workspace_idx use ABSOLUTE indices over the declared signature
# (Q,V,G,A,Hstate,Msk,workspace_1..4,O): no UB->L1 / L0C->UB copies remain
# (all V<->C exchanges go through the explicit named workspaces), so
# AUTO_CV_COMBINE injects no extra GM params and the indices stay stable.
@tilelang.jit(out_idx=[10], workspace_idx=[6, 7, 8, 9], pass_configs=PASS_CONFIGS)
def fla_chunk_o_kernel(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size=64,
    scale=None,
    block_DK=None,
    block_DV=None,
):
    """Single kernel: o = qg @ h + tril_incl_diag(A) @ v per (chunk, b, h)."""
    if scale is None:
        scale = DK**-0.5
    BS = chunk_size
    NT = S // BS
    bDV = block_DV if block_DV is not None else _pick_block_dv(DV)
    GRID = NT * B * H  # logical (chunk, b, h) tasks
    # Fixed-core grid: CORE_EFF blocks (<= cube-core count, <= task count),
    # each running a TPT+1 iteration T.serial task loop with a STRIDED
    # task map tid = cid + t * CORE_EFF. The stride keeps the branch-free
    # Select fallbacks below always in-bounds ("own first task" == cid <
    # CORE_EFF <= GRID), which removes EVERY IfThenElse from the loop --
    # the AUTO_SYNC pass drains all pipes (PipeBarrier<PIPE_ALL>) around
    # each IfThenElse, measured as the dominant per-task overhead.
    CORE_EFF = min(CORE_NUM, GRID)
    TPT = (GRID + CORE_EFF - 1) // CORE_EFF  # iterations that produce

    @T.prim_func
    def main(
        # Q/G/A/O as zero-copy 3D views [B, S, H*D] of the bshd tensors
        # (vid-reduction splits the S rows only when they sit at GM dim
        # gm_dims - ub_dims, see module docstring item 7).
        Q: T.Tensor((B, S, H * DK), "bfloat16"),  # type: ignore
        V: T.Tensor((B, S, H, DV), "bfloat16"),  # type: ignore
        G: T.Tensor((B, S, H * DK), "float32"),  # type: ignore
        A: T.Tensor((B, S, H * BS), "bfloat16"),  # type: ignore
        Hstate: T.Tensor((B, NT, H, DK, DV), "bfloat16"),  # type: ignore
        Msk: T.Tensor((BS, BS), "float32"),  # type: ignore
        # Explicit V<->C relay workspaces (L2-resident, double-buffered).
        workspace_1: T.Tensor((CORE_EFF, NSLOT, BS, DK), "bfloat16"),  # type: ignore
        workspace_2: T.Tensor((CORE_EFF, NSLOT, BS, DK), "bfloat16"),  # type: ignore
        workspace_3: T.Tensor((CORE_EFF, NSLOT, BS, BS), "bfloat16"),  # type: ignore
        o_relay: T.Tensor((CORE_EFF, NSLOT, BS, bDV), "float32"),  # type: ignore
        O: T.Tensor((B, S, H * DV), "bfloat16"),  # type: ignore
    ):
        with T.Kernel(CORE_EFF, threads=2, is_npu=True) as (cid):
            # ---- buffers (placement planned by the compiler; the *_l1
            # names mark gemm operands that must land in L1, the fp32
            # math domains land in UB) ----
            A_l1 = T.alloc_shared([BS, BS], "bfloat16")  # intra gemm LHS (masked)
            V_l1 = T.alloc_shared([BS, bDV], "bfloat16")  # intra gemm RHS
            H_l1 = T.alloc_shared([DK, bDV], "bfloat16")  # inter gemm RHS (full K)
            qg_hi_l1 = T.alloc_shared([BS, DK], "bfloat16")  # inter gemm LHS hi
            qg_lo_l1 = T.alloc_shared([BS, DK], "bfloat16")  # inter gemm LHS lo
            a_half = T.alloc_shared([BS, BS], "bfloat16")  # A load staging
            a_f32 = T.alloc_shared([BS, BS], "float32")  # A fp32 domain
            msk_ub = T.alloc_shared([BS, BS], "float32")  # tril constant
            a_mask = T.alloc_shared([BS, BS], "bfloat16")  # masked A (bf16, exact)
            q_half = T.alloc_shared([BS, DK], "bfloat16")  # q load staging
            q_ub = T.alloc_shared([BS, DK], "float32")  # q fp32 domain
            g_ub = T.alloc_shared([BS, DK], "float32")  # g
            gL_ub = T.alloc_shared([BS, DK], "float32")  # g * ln2
            gexp_ub = T.alloc_shared([BS, DK], "float32")  # 2^g
            qg_ub = T.alloc_shared([BS, DK], "float32")  # qg fp32
            qg_hi_ub = T.alloc_shared([BS, DK], "bfloat16")  # qg hi (bf16)
            qg_hi_f32 = T.alloc_shared([BS, DK], "float32")  # hi, exact fp32
            qg_lo_f32 = T.alloc_shared([BS, DK], "float32")  # lo = qg - hi (fp32)
            qg_lo_ub = T.alloc_shared([BS, DK], "bfloat16")  # qg lo (bf16)
            o_frag = T.alloc_fragment([BS, bDV], "float32")  # L0C accumulator
            o_ub = T.alloc_shared([BS, bDV], "float32")  # relay -> UB staging
            o_half = T.alloc_shared([BS, bDV], "bfloat16")  # output cast staging

            # ---- loop-invariant hoist: the tril constant loads once per
            # core (L2-resident) instead of once per task ----
            T.copy(Msk[0:BS, 0:BS], msk_ub)

            # Prologue: the loads of task cid (iteration 0's produce
            # target, tid = cid + 0 * CORE_EFF) are issued ahead of the
            # loop so the first produce does not pay the MTE2 latency.
            pbs0 = (cid % NT) * BS
            pbbh = cid // NT
            pbb = pbbh // H
            pbh = pbbh % H
            T.copy(A[pbb, pbs0 : pbs0 + BS, pbh * BS : (pbh + 1) * BS], a_half)
            T.copy(Q[pbb, pbs0 : pbs0 + BS, pbh * DK : (pbh + 1) * DK], q_half)
            T.copy(G[pbb, pbs0 : pbs0 + BS, pbh * DK : (pbh + 1) * DK], g_ub)

            # ---- fixed-core software-pipelined task loop, 2-task-deferred
            # drain (msprof showed both cores ~80% idle on cross-core flag
            # waits: the Vector's next produce was gated by its drain,
            # which waited for the Cube's FIX of the PREVIOUS task -- a
            # full lock-step). Deferring the drain by TWO tasks (draining
            # t-2 at the top of iteration t, NSLOT=4 relay slots) lets the
            # Vector run ~2 tasks ahead of the Cube: the drain flag is
            # long set, and the Cube's next overwrite of that slot is
            # transitively ordered behind it (FIX(t+2) needs relay(t+2),
            # which is after V's iteration t). Iterations 0 and 1 are
            # PEELED with the drain last (redirected to garbage): this
            # avoids the t=0 deadlock (produce must precede the first
            # drain) and keeps every flag pair's site counts balanced
            # (2 peeled + 1 loop site per workspace per side).
            #
            # Strided task map: tid = cid + t * CORE_EFF; every task X <
            # GRID is produced exactly once. Out-of-range iterations are
            # redirected BRANCH-FREE (Select) to the core's own first
            # task (cid, always < GRID): the recomputed relay value is
            # bit-identical, the garbage drains are overwritten by the
            # real ones, and ZERO IfThenElse statements remain in the
            # loop (each costs two PipeBarrier<PIPE_ALL> drains).
            #
            # Per-iteration phase order:
            #   Vector: drain(t-2) -> mask chain -> EARLY a_mask relay
            #           (the Cube's intra gemm only needs it + GM v, so
            #           it overlaps the gate chain) -> gate chain -> hi/lo
            #           -> relays -> PREFETCH task t+1's GM loads (their
            #           MTE2 latency hides under the next drain).
            #   Cube:   h/v loads -> intra gemm -> hi/lo gemms -> FIX. ----

            # ---- peeled fill iteration 0 (produce task cid) ----
            pt = cid
            bs = pt % NT
            bbh = pt // NT
            bb = bbh // H
            bh = bbh % H
            s0 = bs * BS
            T.copy(a_half, a_f32)
            T.tile.mul(a_f32, a_f32, msk_ub)
            T.copy(a_f32, a_mask)
            T.copy(a_mask, workspace_3[cid, 0, 0:BS, 0:BS])
            T.copy(q_half, q_ub)
            T.tile.mul(gL_ub, g_ub, LN2)
            T.tile.exp(gexp_ub, gL_ub)
            T.tile.mul(q_ub, q_ub, scale)
            T.tile.mul(qg_ub, q_ub, gexp_ub)
            T.copy(qg_ub, qg_hi_ub)
            T.copy(qg_hi_ub, qg_hi_f32)
            T.tile.sub(qg_lo_f32, qg_ub, qg_hi_f32)
            T.copy(qg_lo_f32, qg_lo_ub)
            T.copy(qg_hi_ub, workspace_1[cid, 0, 0:BS, 0:DK])
            T.copy(qg_lo_ub, workspace_2[cid, 0, 0:BS, 0:DK])
            # prefetch task cid + CORE_EFF
            npt = T.if_then_else(cid + CORE_EFF < GRID, cid + CORE_EFF, cid)
            nbs = npt % NT
            nbbh = npt // NT
            nbb = nbbh // H
            nbh = nbbh % H
            ns0 = nbs * BS
            T.copy(A[nbb, ns0 : ns0 + BS, nbh * BS : (nbh + 1) * BS], a_half)
            T.copy(Q[nbb, ns0 : ns0 + BS, nbh * DK : (nbh + 1) * DK], q_half)
            T.copy(G[nbb, ns0 : ns0 + BS, nbh * DK : (nbh + 1) * DK], g_ub)
            # Cube: consume task cid
            T.copy(Hstate[bb, bs, bh, 0:DK, 0:bDV], H_l1)
            T.copy(V[bb, s0 : s0 + BS, bh, 0:bDV], V_l1)
            T.copy(workspace_3[cid, 0, 0:BS, 0:BS], A_l1)
            T.gemm_v0(A_l1, V_l1, o_frag, init=True)
            T.copy(workspace_1[cid, 0, 0:BS, 0:DK], qg_hi_l1)
            T.gemm_v0(qg_hi_l1, H_l1, o_frag, init=False)
            T.copy(workspace_2[cid, 0, 0:BS, 0:DK], qg_lo_l1)
            T.gemm_v0(qg_lo_l1, H_l1, o_frag, init=False)
            T.copy(o_frag, o_relay[cid, 0, 0:BS, 0:bDV])
            T.set_cross_flag("FIX", 3)
            # drain LAST (garbage, redirected to the core's own region;
            # overwritten by the real drain at t == 2)
            T.copy(o_relay[cid, 2 % NSLOT, 0:BS, 0:bDV], o_ub)
            T.copy(o_ub, o_half)
            T.copy(o_half, O[bb, s0 : s0 + BS, bh * DV : (bh + 1) * DV])

            # ---- peeled fill iteration 1 (produce cid + CORE_EFF) ----
            pt1 = T.if_then_else(cid + CORE_EFF < GRID, cid + CORE_EFF, cid)
            bs1 = pt1 % NT
            bbh1 = pt1 // NT
            bb1 = bbh1 // H
            bh1 = bbh1 % H
            s01 = bs1 * BS
            T.copy(a_half, a_f32)
            T.tile.mul(a_f32, a_f32, msk_ub)
            T.copy(a_f32, a_mask)
            T.copy(a_mask, workspace_3[cid, 1 % NSLOT, 0:BS, 0:BS])
            T.copy(q_half, q_ub)
            T.tile.mul(gL_ub, g_ub, LN2)
            T.tile.exp(gexp_ub, gL_ub)
            T.tile.mul(q_ub, q_ub, scale)
            T.tile.mul(qg_ub, q_ub, gexp_ub)
            T.copy(qg_ub, qg_hi_ub)
            T.copy(qg_hi_ub, qg_hi_f32)
            T.tile.sub(qg_lo_f32, qg_ub, qg_hi_f32)
            T.copy(qg_lo_f32, qg_lo_ub)
            T.copy(qg_hi_ub, workspace_1[cid, 1 % NSLOT, 0:BS, 0:DK])
            T.copy(qg_lo_ub, workspace_2[cid, 1 % NSLOT, 0:BS, 0:DK])
            npt1 = T.if_then_else(cid + 2 * CORE_EFF < GRID, cid + 2 * CORE_EFF, cid)
            nbs1 = npt1 % NT
            nbbh1 = npt1 // NT
            nbb1 = nbbh1 // H
            nbh1 = nbbh1 % H
            ns01 = nbs1 * BS
            T.copy(A[nbb1, ns01 : ns01 + BS, nbh1 * BS : (nbh1 + 1) * BS], a_half)
            T.copy(Q[nbb1, ns01 : ns01 + BS, nbh1 * DK : (nbh1 + 1) * DK], q_half)
            T.copy(G[nbb1, ns01 : ns01 + BS, nbh1 * DK : (nbh1 + 1) * DK], g_ub)
            T.copy(Hstate[bb1, bs1, bh1, 0:DK, 0:bDV], H_l1)
            T.copy(V[bb1, s01 : s01 + BS, bh1, 0:bDV], V_l1)
            T.copy(workspace_3[cid, 1 % NSLOT, 0:BS, 0:BS], A_l1)
            T.gemm_v0(A_l1, V_l1, o_frag, init=True)
            T.copy(workspace_1[cid, 1 % NSLOT, 0:BS, 0:DK], qg_hi_l1)
            T.gemm_v0(qg_hi_l1, H_l1, o_frag, init=False)
            T.copy(workspace_2[cid, 1 % NSLOT, 0:BS, 0:DK], qg_lo_l1)
            T.gemm_v0(qg_lo_l1, H_l1, o_frag, init=False)
            T.copy(o_frag, o_relay[cid, 1 % NSLOT, 0:BS, 0:bDV])
            T.set_cross_flag("FIX", 4)
            # drain LAST (garbage)
            T.copy(o_relay[cid, 3 % NSLOT, 0:BS, 0:bDV], o_ub)
            T.copy(o_ub, o_half)
            T.copy(o_half, O[bb1, s01 : s01 + BS, bh1 * DV : (bh1 + 1) * DV])

            # ---- steady-state loop: t = 2 .. TPT + 1 ----
            for tt in T.serial(TPT):
                t = tt + 2
                tid = cid + t * CORE_EFF
                pt = T.if_then_else(tid < GRID, tid, cid)
                bs = pt % NT
                bbh = pt // NT
                bb = bbh // H
                bh = bbh % H
                s0 = bs * BS
                slot = t % NSLOT
                # drain target: the core's task from two iterations ago;
                # fall back to cid when it was itself a redirect (the slot
                # then holds the cid recompute, so the drained value still
                # matches the O region written)
                dt_raw = tid - 2 * CORE_EFF
                dt = T.if_then_else(dt_raw < GRID, dt_raw, cid)
                dbs = dt % NT
                dbbh = dt // NT
                dbb = dbbh // H
                dbh = dbbh % H
                ds0 = dbs * BS
                dslot = (t + 2) % NSLOT  # == (t-2) % NSLOT
                # prefetch target: the next iteration's produce target
                ntid = tid + CORE_EFF
                npt = T.if_then_else(ntid < GRID, ntid, cid)
                nbs = npt % NT
                nbbh = npt // NT
                nbb = nbbh // H
                nbh = nbbh % H
                ns0 = nbs * BS

                # -- Vector: drain the t-2 output (its FIX is long done;
                # the manual flag pairs with iteration t-2's set -- the
                # auto flags are FIFO index-paired and would force a
                # same-iteration lock-step, see module docstring) --
                T.wait_cross_flag(3 + t % 2, "V")
                T.copy(o_relay[cid, dslot, 0:BS, 0:bDV], o_ub)
                T.copy(o_ub, o_half)  # cast -> bf16
                T.copy(o_half, O[dbb, ds0 : ds0 + BS, dbh * DV : (dbh + 1) * DV])

                # -- Vector: A tril mask (fp32 domain; exact 0/1) --
                T.copy(a_half, a_f32)
                T.tile.mul(a_f32, a_f32, msk_ub)
                T.copy(a_f32, a_mask)

                # -- Vector -> Cube: EARLY a_mask relay --
                T.copy(a_mask, workspace_3[cid, slot, 0:BS, 0:BS])

                # -- Vector: qg = (q*scale) * 2^g (fp32 domain) --
                T.copy(q_half, q_ub)
                T.tile.mul(gL_ub, g_ub, LN2)
                T.tile.exp(gexp_ub, gL_ub)
                T.tile.mul(q_ub, q_ub, scale)
                T.tile.mul(qg_ub, q_ub, gexp_ub)

                # -- Vector: bf16 hi/lo split of qg (a single cast leaves
                # a ~2^-9 relative error on o_inter that trips the mixed
                # tolerance on cancellation elements; hi/lo double gemm
                # -> ~2^-18, DESIGN SS9.3) --
                T.copy(qg_ub, qg_hi_ub)
                T.copy(qg_hi_ub, qg_hi_f32)
                T.tile.sub(qg_lo_f32, qg_ub, qg_hi_f32)
                T.copy(qg_lo_f32, qg_lo_ub)

                # -- Vector -> Cube relays (auto cross-core flags) --
                T.copy(qg_hi_ub, workspace_1[cid, slot, 0:BS, 0:DK])
                T.copy(qg_lo_ub, workspace_2[cid, slot, 0:BS, 0:DK])

                # -- Vector: PREFETCH the next task's GM loads (latency
                # hides under the next drain; staging is dead now) --
                T.copy(A[nbb, ns0 : ns0 + BS, nbh * BS : (nbh + 1) * BS], a_half)
                T.copy(Q[nbb, ns0 : ns0 + BS, nbh * DK : (nbh + 1) * DK], q_half)
                T.copy(G[nbb, ns0 : ns0 + BS, nbh * DK : (nbh + 1) * DK], g_ub)

                # -- Cube: operand loads first (no flag deps) --
                T.copy(Hstate[bb, bs, bh, 0:DK, 0:bDV], H_l1)
                T.copy(V[bb, s0 : s0 + BS, bh, 0:bDV], V_l1)

                # -- Cube: intra gemm FIRST (needs only a_mask + v, both
                # ready early); then the inter hi/lo gemms --
                T.copy(workspace_3[cid, slot, 0:BS, 0:BS], A_l1)
                T.gemm_v0(A_l1, V_l1, o_frag, init=True)
                T.copy(workspace_1[cid, slot, 0:BS, 0:DK], qg_hi_l1)
                T.gemm_v0(qg_hi_l1, H_l1, o_frag, init=False)
                T.copy(workspace_2[cid, slot, 0:BS, 0:DK], qg_lo_l1)
                T.gemm_v0(qg_lo_l1, H_l1, o_frag, init=False)

                # -- Cube -> Vector: relay the fp32 accumulator (manual
                # flag: ID 3 + t%2 alternates so the Vector's 2-task
                # deferred drain pairs with THIS iteration's FIX) --
                T.copy(o_frag, o_relay[cid, slot, 0:BS, 0:bDV])
                T.set_cross_flag("FIX", 3 + t % 2)

    return main


# ============================================================================
# Host wrapper (upstream naming: chunk_gla_fwd_o_gk). Only shape checks +
# Msk constant caching + kernel invocation; no data mutation on the host.
# ============================================================================
def _check_shape(B, S, H, chunk_size, A):
    assert S % chunk_size == 0, "S must be divisible by chunk_size"
    assert chunk_size == 64, "chunk_size fixed at 64 (block_S semantic)"
    assert A.shape[-1] == chunk_size, "A last dim must equal chunk_size"


_MSK_CACHE = {}


def _get_msk(device, chunk_size):
    """Inclusive-tril constant (diagonal=0), built ONCE per (device, chunk).

    Per-call reconstruction would cost a CPU tril + a potentially
    synchronizing H2D copy inside the timed region.
    """
    key = (str(device), chunk_size)
    if key not in _MSK_CACHE:
        _MSK_CACHE[key] = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.float32), 0).to(device)
    return _MSK_CACHE[key]


def fla_chunk_o(q, v, g, A, h, scale=None, chunk_size=64):
    """Host API (upstream: chunk_gla_fwd_o_gk, use_exp2=True, fixed length).

    q[B,S,H,DK] bf16, v[B,S,H,DV] bf16, g[B,S,H,DK] fp32 (cumulative log2
    decay), A[B,S,H,64] bf16 (intra-chunk attention), h[B,S//64,H,DK,DV]
    bf16 (chunk-level hidden state) -> o[B,S,H,DV] bf16.
    """
    B, S, H, DK = q.shape
    DV = v.shape[-1]
    _check_shape(B, S, H, chunk_size, A)
    if scale is None:
        scale = DK**-0.5
    _pick_block_dk(DK)  # domain gate (contract unchanged)
    _pick_block_dv(DV)
    msk = _get_msk(q.device, chunk_size)
    kernel = fla_chunk_o_kernel(B, S, H, DK, DV, chunk_size, scale, None, None)
    # Zero-copy 3D views [B, S, H*D] of the contiguous bshd tensors (pure
    # metadata, no materialization): puts the S row dim at GM position
    # (gm_dims - ub_dims) so the threads=2 vid-reduction splits rows, not
    # heads (see the kernel docstring item 7). v/h stay 4D/5D (Cube-side).
    o = kernel(
        q.view(B, S, H * DK),
        v,
        g.view(B, S, H * DK),
        A.view(B, S, H * chunk_size),
        h,
        msk,
    )
    # no internal synchronize: upstream wrapper parity (the do_bench harness
    # owns synchronization; an internal sync would expose host-side JIT
    # dispatch latency inside the device-side measurement)
    return o.view(B, S, H, DV)


__all__ = [
    "fla_chunk_o",
    "fla_chunk_o_kernel",
    "PASS_CONFIGS",
]
