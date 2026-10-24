"""fla_chunk_o: KDA chunked forward output O (single kernel) on Ascend 910B3.

o = inter-chunk (qg @ h) + intra-chunk (tril_incl_diag(A) @ v), where
qg = (q * scale) * 2^g computed as exp(g * ln2) (exp2 has no Ascend
backend mapping). Inputs are consumed-only (A/h produced upstream).

Semantics: ref_gpu/fla_chunk_o.py chunk_gla_fwd_kernel_o (Triton, frozen);
code-level origin: ref_gpu/fla_chunk_o_tilelang_fwd.py (upstream TileLang
CUDA tilelang_chunk_fwd_o).

Developer mode with explicit GM relay workspaces and ONE manual cross-core
flag pair (Stage 3 restructure; the manual flag is the mixed-mode element
-- see item 4b below). bshd layout consumed via zero-copy 3D views (no
host-side materialization, no aclnn calls in the wrapper).

Stage-3 performance structure (perf_tuning/ has the full ablation data;
baseline 3.285 ms -> ~1.51 ms on the bench shape, -54%; round 2 added the
v relay, L1/L0C ping-pong, in-place gate chains and the 2-batch throttle):

* FIXED-CORE grid: T.Kernel(CORE_EFF <= 20 cube cores, threads=2) with an
  in-core task loop (antipatterns "launch core 数" 关注项 A). The 8192
  logical (chunk, b, h) tasks are striped over the cores (tid = cid +
  t * CORE_EFF); one block instance per core amortizes the per-block
  dispatch/TPipe-init overhead and lets the MTE2/MTE3/V queues flow across
  tasks.
* EXPLICIT relay workspaces (workspace_idx): qg_hi/qg_lo/a_mask/v are
  staged through small named GM tensors ([CORE, 8, ...], a few MiB --
  permanently L2-resident) instead of the AUTO_CV_COMBINE auto-injected
  per-block workspaces (~768 MiB per launch). Their V->C handoffs keep the
  auto-inserted cross-core flags (AUTO_CV_SYNC matches buffers named
  "workspace_*"). The v relay (R2-6) moves the strided 64x256B v read
  onto the Vector's MTE2 (the Cube's MTE2 ran at ~16 GB/s on that strided
  pattern vs ~48 GB/s contiguous -- measured -6%); the Cube reads the
  contiguous relay instead.
* SOFTWARE-PIPELINED 2-CHUNK BATCHED task loop, Vector throttled TWO
  BATCHES behind the Cube. One pipeline step consumes the strided task
  pair (2k, 2k+1): the per-iteration sync scaffolding amortizes over two
  tasks, chunk B's h load issues right after chunk A's gemms (distinct
  L1 buffer names -> no forced M-pipe wait), and the L0C accumulator is
  ping-ponged per chunk parity (FIX-A overlaps intra-B). The output is
  written DIRECTLY by the Cube-side fixpipe (R2): copy_l0c_to_gm converts
  the fp32 L0C accumulator to bf16 and stores into the strided O rows
  (realDstN = H*DV, derived by compute_strideN in src/op/ascend.cc) -- no
  GM relay round-trip for the output (an o_relay + Vector-drain variant
  measured +11% WORSE once the Vector carried the v relay). The single
  manual flag (ID 12, one set per batch after chunk A's FIX, outside the
  auto pass's range) is a pure THROTTLE: the Vector waits it at the top
  of batch k, gating its batch-k relay writes on the Cube's FIX-A of
  batch k-2. The 2-BATCH lag (R2-9, NSLOT=8) keeps every auto relay flag
  pre-set when the Cube arrives (the 1-batch lag left the hi/lo relays
  just-in-time and the Cube burned ~0.5 ms total waiting on them). Two
  peeled single-task fills plus one peeled full batch keep every flag
  pair's site counts balanced and seed the pipeline.
* Branch-free tail handling: out-of-range iterations (the last core's
  remainder plus the final fill iterations) are redirected with
  Select expressions to the core's own first task (cid, always in-range):
  the recomputed relay value is bit-identical and the redundant O writes
  are exact recomputes. ZERO IfThenElse statements remain in the loop --
  the AUTO_SYNC pass drains all pipes (PipeBarrier<PIPE_ALL>) around EVERY
  IfThenElse, measured as the dominant per-task overhead.
* UNROLLED full-DK inter term: q/g are loaded once per task as [64, DK]
  rows and the hi/lo pair is applied with two K=DK gemms (the upstream
  i_k Pipelined loop never software-pipelined on this toolchain).
* Overlap micro-structure: batched A/Q/G/V loads; the a_mask relay is
  issued right after the mask chain (before the gate chain) so the
  Cube's intra gemm overlaps the Vector's gate math; the inter relay
  reads (hi+lo) are issued back-to-back; the next batch's A/Q/G/V loads
  are prefetched at the end of each body (latency hides under the next
  throttle wait).

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
4. The output is written directly by the Cube-side fixpipe as BF16 (the
   fp32 L0C -> bf16 GM conversion rides the FIX instruction, verified
   exact on this toolchain -- the fla_wy_fast "L0C->GM(bf16) direct
   fixpipe is broken" record does not reproduce on this path, R2).
4b. The single manual cross-core flag (ID 12, the Vector throttle)
   classifies onto the right core via its pipe argument:
   T.set_cross_flag("FIX", ...) -> Cube, T.wait_cross_flag(..., "V") ->
   Vector (ResourceForPipe accepts exactly "V"/"M"/"MTE1"/"FIX";
   anything else needs T.Scope).
5. The tril mask is a constant Msk[64,64] GM input (inclusive diagonal,
   tril(ones, 0)) multiplied in the fp32 domain via T.tile.mul --
   T.if_then_else is unavailable in tile ops. Exact in bf16 (0/1 mask).
   The Msk load is hoisted out of the task loop (loop-invariant).
6. Element math uses buffer-level T.tile.* instructions only (element
   T.Parallel loops are the -26%..-71%/kernel antipattern).
7. Q/G/A/O are consumed through zero-copy 3D views [B, S, H*D] of the
   bshd tensors: the vid-reduction pass adds the sub-block row offset to
   GM dimension (gm_dims - ub_dims); with 4D bshd slices that lands on
   the H index (reads a different head -- silent data corruption), while
   3D row slices put the S rows exactly there. V/Hstate are consumed
   Cube-side (GM->L1, not vid-split) and stay 4D/5D.
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

# Relay slot ring per core (t % NSLOT). R2-9: the Vector is throttled TWO
# BATCHES (4 tasks) behind the Cube (its batch-k relay writes are gated by
# the Cube's FIX-A of batch k-2), so every relay flag is long set when the
# Cube arrives -- the measured ~470ns x 4 relay-flag waits per task (777us
# total, the hi/lo pair alone ~490us) collapse to protocol overhead. The
# ring must then hold THREE in-flight batches (V at k, C at k-1, k-2's
# slots being retired): NSLOT=8 gives 4 batches of margin.
NSLOT = 8

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
# (Q,V,G,A,Hstate,Msk,workspace_1..3,O): no UB->L1 / L0C->UB copies remain
# (all V<->C exchanges go through the explicit named workspaces), so
# AUTO_CV_COMBINE injects no extra GM params and the indices stay stable.
# R2: the output is written DIRECTLY by the Cube-side fixpipe
# (copy_l0c_to_gm with realDstN = H*DV row stride, derived by
# compute_strideN in src/op/ascend.cc) -- no o_relay GM round-trip.
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
    TPT = (GRID + CORE_EFF - 1) // CORE_EFF  # produce slots t = 0 .. TPT+1
    # 2-chunk batching (R2-4): pipeline step k consumes tasks (2k, 2k+1)
    # of the strided map (tid = cid + t*CORE_EFF). NB batch steps cover the
    # TPT produce slots from t=2 up (the first two are the peeled fills).
    # Select-free fast path: for k <= NB_MAIN every core's produce AND
    # prefetch targets are in-range ((2k+4)*CORE_EFF <= GRID), so the body
    # needs no Select guards; the remaining tail batches keep the
    # branch-free Select redirects to the core's own first task.
    NB = (TPT + 1) // 2  # batch steps k = 1 .. NB
    # Select-free bound: batch k is guard-free when (2k+4)*CORE_EFF <= GRID.
    K_MAX = (GRID // CORE_EFF - 4) // 2
    # Batch 1 (tasks 2,3) is a PEELED batch (keeps the manual-flag counting
    # aligned with the 2-batch throttle lag); the loops cover k = 2 .. NB.
    N_MAIN = max(0, min(K_MAX, NB) - 1)  # Select-free batches k = 2 .. K_MAX
    N_TAIL = NB - max(1, min(K_MAX, NB))  # Select-guarded remainder

    @T.prim_func
    def main(
        # Q/G/A/O as zero-copy 3D views [B, S, H*D] of the bshd tensors
        # (vid-reduction splits the S rows only when they sit at GM dim
        # gm_dims - ub_dims, see module docstring item 7).
        Q: T.Tensor((B, S, H * DK), "bfloat16"),  # type: ignore
        V: T.Tensor((B, S, H * DV), "bfloat16"),  # type: ignore
        G: T.Tensor((B, S, H * DK), "float32"),  # type: ignore
        A: T.Tensor((B, S, H * BS), "bfloat16"),  # type: ignore
        Hstate: T.Tensor((B, NT, H, DK, DV), "bfloat16"),  # type: ignore
        Msk: T.Tensor((BS, BS), "float32"),  # type: ignore
        # Explicit V->C relay workspaces (L2-resident, slot ring).
        workspace_1: T.Tensor((CORE_EFF, NSLOT, BS, DK), "bfloat16"),  # type: ignore
        workspace_2: T.Tensor((CORE_EFF, NSLOT, BS, DK), "bfloat16"),  # type: ignore
        workspace_3: T.Tensor((CORE_EFF, NSLOT, BS, BS), "bfloat16"),  # type: ignore
        workspace_4: T.Tensor((CORE_EFF, NSLOT, BS, bDV), "bfloat16"),  # type: ignore
        O: T.Tensor((B, S, H * DV), "bfloat16"),  # type: ignore
    ):
        with T.Kernel(CORE_EFF, threads=2, is_npu=True) as (cid):
            # ---- buffers (placement planned by the compiler; the *_l1
            # names mark gemm operands that must land in L1, the fp32
            # math domains land in UB) ----
            # R2-5: L1 gemm operands are PING-PONGED BY NAME (chunk A/_a,
            # chunk B/_b). The AUTO_SYNC WAR analysis (HasDataDependency,
            # ascend_sync_insert.cc) is per buffer NAME: with shared names
            # every inter-chunk load->gemm pair gets a forced M_MTE2
            # serialization (measured: the MTE2 queue and the M pipe take
            # strict turns, zero overlap); with distinct names chunk B's
            # h/v loads drain under chunk A's gemm/FIX tail and the next
            # iteration's A-loads overlap B's gemms. L1 total 176KB/512KB.
            A_l1_a = T.alloc_shared([BS, BS], "bfloat16")  # intra LHS, chunk A
            A_l1_b = T.alloc_shared([BS, BS], "bfloat16")  # intra LHS, chunk B
            V_l1_a = T.alloc_shared([BS, bDV], "bfloat16")  # intra RHS, chunk A
            V_l1_b = T.alloc_shared([BS, bDV], "bfloat16")  # intra RHS, chunk B
            H_l1_a = T.alloc_shared([DK, bDV], "bfloat16")  # inter RHS, chunk A
            H_l1_b = T.alloc_shared([DK, bDV], "bfloat16")  # inter RHS, chunk B
            qg_hi_l1_a = T.alloc_shared([BS, DK], "bfloat16")  # inter LHS hi, A
            qg_hi_l1_b = T.alloc_shared([BS, DK], "bfloat16")  # inter LHS hi, B
            qg_lo_l1_a = T.alloc_shared([BS, DK], "bfloat16")  # inter LHS lo, A
            qg_lo_l1_b = T.alloc_shared([BS, DK], "bfloat16")  # inter LHS lo, B
            a_half = T.alloc_shared([BS, BS], "bfloat16")  # A load staging (chunk A)
            a_f32 = T.alloc_shared([BS, BS], "float32")  # A fp32 domain
            msk_ub = T.alloc_shared([BS, BS], "float32")  # tril constant
            a_mask = T.alloc_shared([BS, BS], "bfloat16")  # masked A (bf16, exact)
            q_half = T.alloc_shared([BS, DK], "bfloat16")  # q load staging (chunk A)
            q_ub = T.alloc_shared([BS, DK], "float32")  # q fp32 (reused: qg, lo)
            g_ub = T.alloc_shared([BS, DK], "float32")  # g (A; reused: g*ln2, 2^g)
            v_half = T.alloc_shared([BS, bDV], "bfloat16")  # v staging (chunk A, prefetched)
            v_half2 = T.alloc_shared([BS, bDV], "bfloat16")  # v staging (chunk B, prefetched)
            a_half2 = T.alloc_shared([BS, BS], "bfloat16")  # A staging (chunk B)
            q_half2 = T.alloc_shared([BS, DK], "bfloat16")  # q staging (chunk B)
            g_ub2 = T.alloc_shared([BS, DK], "float32")  # g (B; reused)
            qg_hi_ub = T.alloc_shared([BS, DK], "bfloat16")  # qg hi (bf16)
            qg_hi_f32 = T.alloc_shared([BS, DK], "float32")  # hi, exact fp32
            qg_lo_ub = T.alloc_shared([BS, DK], "bfloat16")  # qg lo (bf16)
            # R2-8: the L0C accumulator is PING-PONGED per chunk parity
            # (2 x 32KB <= 128KB L0C): chunk B's gemms no longer WAR chunk
            # A's FIX read of the accumulator, so the FIX pipe overlaps the
            # M pipe's next gemm group instead of serializing behind it
            # (the per-boundary FIX_M/M_FIX handshake pairs disappear).
            o_frag_a = T.alloc_fragment([BS, bDV], "float32")  # L0C, chunk A
            o_frag_b = T.alloc_fragment([BS, bDV], "float32")  # L0C, chunk B

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
            T.copy(V[pbb, pbs0 : pbs0 + BS, pbh * DV : (pbh + 1) * DV], v_half)

            # ---- fixed-core software-pipelined task loop (R2: output
            # fixpiped straight to O by the Cube; the Vector is throttled
            # 2 tasks behind via the manual flag pair 12/13 -- its relay
            # writes for task t are gated by the Cube's FIX of t-2, which
            # breaks the AUTO_CV_SYNC same-iteration FIFO lock-step AND
            # keeps the NSLOT=4 relay ring from ever wrapping). Iterations
            # 0 and 1 are PEELED: this keeps every flag pair's site counts
            # balanced (2 peeled + 1 loop site per workspace per side) and
            # lets the prologue loads hide the first produce's MTE2
            # latency.
            #
            # Strided task map: tid = cid + t * CORE_EFF; every task X <
            # GRID is produced exactly once. Out-of-range iterations are
            # redirected BRANCH-FREE (Select) to the core's own first
            # task (cid, always < GRID): the recomputed relay value is
            # bit-identical, the redundant O writes are exact recomputes,
            # and ZERO IfThenElse statements remain in the loop (each
            # costs two PipeBarrier<PIPE_ALL> drains).
            #
            # Per-iteration phase order:
            #   Vector: throttle-wait(FIX of t-2) -> mask chain -> EARLY
            #           a_mask relay (the Cube's intra gemm only needs it
            #           + GM v, so it overlaps the gate chain) -> gate
            #           chain -> hi/lo -> relays -> PREFETCH task t+1's GM
            #           loads (their MTE2 latency hides under the next
            #           throttle wait).
            #   Cube:   h/v loads -> intra gemm -> hi/lo gemms -> FIX
            #           (direct bf16 store into O). ----

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
            T.copy(v_half, workspace_4[cid, 0, 0:BS, 0:bDV])
            T.copy(q_half, q_ub)
            T.tile.mul(g_ub, g_ub, LN2)
            T.tile.exp(g_ub, g_ub)
            T.tile.mul(q_ub, q_ub, scale)
            T.tile.mul(q_ub, q_ub, g_ub)
            T.copy(q_ub, qg_hi_ub)
            T.copy(qg_hi_ub, qg_hi_f32)
            T.tile.sub(q_ub, q_ub, qg_hi_f32)
            T.copy(q_ub, qg_lo_ub)
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
            T.copy(V[nbb, ns0 : ns0 + BS, nbh * DV : (nbh + 1) * DV], v_half)
            # Cube: consume task cid (ping-pong slot A)
            T.copy(Hstate[bb, bs, bh, 0:DK, 0:bDV], H_l1_a)
            T.copy(workspace_4[cid, 0, 0:BS, 0:bDV], V_l1_a)
            T.copy(workspace_3[cid, 0, 0:BS, 0:BS], A_l1_a)
            T.gemm_v0(A_l1_a, V_l1_a, o_frag_a, init=True)
            T.copy(workspace_1[cid, 0, 0:BS, 0:DK], qg_hi_l1_a)
            T.copy(workspace_2[cid, 0, 0:BS, 0:DK], qg_lo_l1_a)
            T.gemm_v0(qg_hi_l1_a, H_l1_a, o_frag_a, init=False)
            T.gemm_v0(qg_lo_l1_a, H_l1_a, o_frag_a, init=False)
            # R2: fixpipe writes O DIRECTLY (strided rows, realDstN = H*DV);
            # the fp32->bf16 conversion on this instruction is verified exact
            # (R1 iter 7 -- the fla_wy_fast "direct store broken" record does
            # not reproduce on the copy_l0c_to_gm path with a proper stride).
            T.copy(o_frag_a, O[bb, s0 : s0 + BS, bh * DV : (bh + 1) * DV])
            # The manual flag now serves as the VECTOR-SIDE THROTTLE: V's
            # iteration-t relay writes are gated by C's FIX of task t-2, so V
            # stays <= 2 tasks ahead and the NSLOT=4 relay ring never wraps.
            T.set_cross_flag("FIX", 12)

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
            T.copy(v_half, workspace_4[cid, 1 % NSLOT, 0:BS, 0:bDV])
            T.copy(q_half, q_ub)
            T.tile.mul(g_ub, g_ub, LN2)
            T.tile.exp(g_ub, g_ub)
            T.tile.mul(q_ub, q_ub, scale)
            T.tile.mul(q_ub, q_ub, g_ub)
            T.copy(q_ub, qg_hi_ub)
            T.copy(qg_hi_ub, qg_hi_f32)
            T.tile.sub(q_ub, q_ub, qg_hi_f32)
            T.copy(q_ub, qg_lo_ub)
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
            T.copy(V[nbb1, ns01 : ns01 + BS, nbh1 * DV : (nbh1 + 1) * DV], v_half)
            # R2-4: the first batched iteration consumes TWO staging slots,
            # so the peel prefetches both of its tasks (2 and 3).
            npt2 = T.if_then_else(
                cid + 3 * CORE_EFF < GRID, cid + 3 * CORE_EFF, cid
            )
            nbs2 = npt2 % NT
            nbbh2 = npt2 // NT
            nbb2 = nbbh2 // H
            nbh2 = nbbh2 % H
            ns02 = nbs2 * BS
            T.copy(A[nbb2, ns02 : ns02 + BS, nbh2 * BS : (nbh2 + 1) * BS], a_half2)
            T.copy(Q[nbb2, ns02 : ns02 + BS, nbh2 * DK : (nbh2 + 1) * DK], q_half2)
            T.copy(G[nbb2, ns02 : ns02 + BS, nbh2 * DK : (nbh2 + 1) * DK], g_ub2)
            T.copy(V[nbb2, ns02 : ns02 + BS, nbh2 * DV : (nbh2 + 1) * DV], v_half2)
            T.copy(Hstate[bb1, bs1, bh1, 0:DK, 0:bDV], H_l1_b)
            T.copy(workspace_4[cid, 1 % NSLOT, 0:BS, 0:bDV], V_l1_b)
            T.copy(workspace_3[cid, 1 % NSLOT, 0:BS, 0:BS], A_l1_b)
            T.gemm_v0(A_l1_b, V_l1_b, o_frag_b, init=True)
            T.copy(workspace_1[cid, 1 % NSLOT, 0:BS, 0:DK], qg_hi_l1_b)
            T.copy(workspace_2[cid, 1 % NSLOT, 0:BS, 0:DK], qg_lo_l1_b)
            T.gemm_v0(qg_hi_l1_b, H_l1_b, o_frag_b, init=False)
            T.gemm_v0(qg_lo_l1_b, H_l1_b, o_frag_b, init=False)
            T.copy(o_frag_b, O[bb1, s01 : s01 + BS, bh1 * DV : (bh1 + 1) * DV])
            # R2-4: NO manual flag set here -- the throttle signal is one
            # set per batch (after the batch's task-A FIX), so the global
            # set(12)/wait(12) sequence alternates strictly.

            # ---- peeled batch 1 (tasks 2 and 3) ----
            # R2-9: batch 1 is peeled as a FULL batch (not throttled -- the
            # Vector is naturally behind here) so the loop's k-th wait_cross
            # flag(12) pairs the Cube's set from batch k-2 exactly: the
            # Vector runs TWO batches ahead in the steady state and every
            # relay flag is pre-set when the Cube arrives.
            pbta = cid + 2 * CORE_EFF
            pbpta = T.if_then_else(pbta < GRID, pbta, cid)
            pbabs_ = pbpta % NT
            pbabbh = pbpta // NT
            pbabb = pbabbh // H
            pbabh = pbabbh % H
            pbas0 = pbabs_ * BS
            pbtb = pbta + CORE_EFF
            pbptb = T.if_then_else(pbtb < GRID, pbtb, cid)
            pbbbs = pbptb % NT
            pbbbh = pbptb // NT
            pbbbb = pbbbh // H
            pbbbh_ = pbbbh % H
            pbbs0 = pbbbs * BS
            pbslotA = 2 % NSLOT
            pbslotB = 3 % NSLOT
            pbnpa = T.if_then_else(pbta + 2 * CORE_EFF < GRID, pbta + 2 * CORE_EFF, cid)
            pbnab = pbnpa % NT
            pbnabh = pbnpa // NT
            pnbb_ = pbnabh // H
            pnbh_ = pbnabh % H
            pbnas0 = pbnab * BS
            pbnpb = T.if_then_else(pbta + 3 * CORE_EFF < GRID, pbta + 3 * CORE_EFF, cid)
            pnbb2 = pbnpb % NT
            pnbbh2 = pbnpb // NT
            pnbbb = pnbbh2 // H
            pnbh2 = pnbbh2 % H
            pnbs0 = pnbb2 * BS

            # -- Vector: chunk A (task 2) mask + relays --
            T.copy(a_half, a_f32)
            T.tile.mul(a_f32, a_f32, msk_ub)
            T.copy(a_f32, a_mask)
            T.copy(a_mask, workspace_3[cid, pbslotA, 0:BS, 0:BS])
            T.copy(v_half, workspace_4[cid, pbslotA, 0:BS, 0:bDV])
            T.copy(q_half, q_ub)
            T.tile.mul(g_ub, g_ub, LN2)
            T.tile.exp(g_ub, g_ub)
            T.tile.mul(q_ub, q_ub, scale)
            T.tile.mul(q_ub, q_ub, g_ub)
            T.copy(q_ub, qg_hi_ub)
            T.copy(qg_hi_ub, qg_hi_f32)
            T.tile.sub(q_ub, q_ub, qg_hi_f32)
            T.copy(q_ub, qg_lo_ub)
            T.copy(qg_hi_ub, workspace_1[cid, pbslotA, 0:BS, 0:DK])
            T.copy(qg_lo_ub, workspace_2[cid, pbslotA, 0:BS, 0:DK])

            # -- Vector: chunk B (task 3) mask + relays --
            T.copy(a_half2, a_f32)
            T.tile.mul(a_f32, a_f32, msk_ub)
            T.copy(a_f32, a_mask)
            T.copy(a_mask, workspace_3[cid, pbslotB, 0:BS, 0:BS])
            T.copy(v_half2, workspace_4[cid, pbslotB, 0:BS, 0:bDV])
            T.copy(q_half2, q_ub)
            T.tile.mul(g_ub2, g_ub2, LN2)
            T.tile.exp(g_ub2, g_ub2)
            T.tile.mul(q_ub, q_ub, scale)
            T.tile.mul(q_ub, q_ub, g_ub2)
            T.copy(q_ub, qg_hi_ub)
            T.copy(qg_hi_ub, qg_hi_f32)
            T.tile.sub(q_ub, q_ub, qg_hi_f32)
            T.copy(q_ub, qg_lo_ub)
            T.copy(qg_hi_ub, workspace_1[cid, pbslotB, 0:BS, 0:DK])
            T.copy(qg_lo_ub, workspace_2[cid, pbslotB, 0:BS, 0:DK])

            # -- Vector: PREFETCH batch 2's GM loads --
            T.copy(A[pnbb_, pbnas0 : pbnas0 + BS, pnbh_ * BS : (pnbh_ + 1) * BS], a_half)
            T.copy(Q[pnbb_, pbnas0 : pbnas0 + BS, pnbh_ * DK : (pnbh_ + 1) * DK], q_half)
            T.copy(G[pnbb_, pbnas0 : pbnas0 + BS, pnbh_ * DK : (pnbh_ + 1) * DK], g_ub)
            T.copy(V[pnbb_, pbnas0 : pbnas0 + BS, pnbh_ * DV : (pnbh_ + 1) * DV], v_half)
            T.copy(A[pnbbb, pnbs0 : pnbs0 + BS, pnbh2 * BS : (pnbh2 + 1) * BS], a_half2)
            T.copy(Q[pnbbb, pnbs0 : pnbs0 + BS, pnbh2 * DK : (pnbh2 + 1) * DK], q_half2)
            T.copy(G[pnbbb, pnbs0 : pnbs0 + BS, pnbh2 * DK : (pnbh2 + 1) * DK], g_ub2)
            T.copy(V[pnbbb, pnbs0 : pnbs0 + BS, pnbh2 * DV : (pnbh2 + 1) * DV], v_half2)

            # -- Cube: consume chunk A (task 2) --
            T.copy(Hstate[pbabb, pbabs_, pbabh, 0:DK, 0:bDV], H_l1_a)
            T.copy(workspace_4[cid, pbslotA, 0:BS, 0:bDV], V_l1_a)
            T.copy(workspace_3[cid, pbslotA, 0:BS, 0:BS], A_l1_a)
            T.gemm_v0(A_l1_a, V_l1_a, o_frag_a, init=True)
            T.copy(workspace_1[cid, pbslotA, 0:BS, 0:DK], qg_hi_l1_a)
            T.copy(workspace_2[cid, pbslotA, 0:BS, 0:DK], qg_lo_l1_a)
            T.gemm_v0(qg_hi_l1_a, H_l1_a, o_frag_a, init=False)
            T.gemm_v0(qg_lo_l1_a, H_l1_a, o_frag_a, init=False)
            T.copy(o_frag_a, O[pbabb, pbas0 : pbas0 + BS, pbabh * DV : (pbabh + 1) * DV])
            T.set_cross_flag("FIX", 12)
            # -- Cube: issue chunk B's h load early, consume, FIX --
            T.copy(Hstate[pbbbb, pbbbs, pbbbh_, 0:DK, 0:bDV], H_l1_b)
            T.copy(workspace_4[cid, pbslotB, 0:BS, 0:bDV], V_l1_b)
            T.copy(workspace_3[cid, pbslotB, 0:BS, 0:BS], A_l1_b)
            T.gemm_v0(A_l1_b, V_l1_b, o_frag_b, init=True)
            T.copy(workspace_1[cid, pbslotB, 0:BS, 0:DK], qg_hi_l1_b)
            T.copy(workspace_2[cid, pbslotB, 0:BS, 0:DK], qg_lo_l1_b)
            T.gemm_v0(qg_hi_l1_b, H_l1_b, o_frag_b, init=False)
            T.gemm_v0(qg_lo_l1_b, H_l1_b, o_frag_b, init=False)
            T.copy(o_frag_b, O[pbbbb, pbbs0 : pbbs0 + BS, pbbbh_ * DV : (pbbbh_ + 1) * DV])

            # ---- steady-state loops: 2-chUNK BATCHED (R2-4).
            # One pipeline step consumes TWO strided tasks (2k, 2k+1):
            #   * the per-iteration sync scaffolding (loop control, the
            #     M_MTE2 top-of-iteration WAR pair, marker groups) amortizes
            #     over 2 tasks;
            #   * chunk B's h/v MTE2 loads issue right after chunk A's FIX
            #     and drain under A's tail + B's own flag waits, and the
            #     NEXT iteration's chunk-A loads overlap B's gemms -- the
            #     exposed per-task MTE2 latency roughly halves;
            #   * the Vector's fp32 chains run sequentially per chunk (the
            #     compute buffers are reused; only the load staging is
            #     double-buffered: a_half/q_half/g_ub + a_half2/q_half2/
            #     g_ub2, +28KB/vid, fits the UB freed by the R2-1 direct-O
            #     write).
            # Throttle: ONE manual-flag wait per batch (flag 12, no parity)
            # at the top of the Vector body; the Cube sets it once per
            # batch (after chunk A's FIX). The global set/wait sequence
            # alternates strictly (C's batch-k task-A FIX needs V's batch-k
            # relay writes, which follow V's k-th wait), and the ring
            # analysis holds: V writes slots {2k%4, (2k+1)%4} at step k
            # while C reads the complementary pair at step k-1 -- the two
            # in-flight batches exactly fill the NSLOT=4 ring.
            # The body is emitted twice (MAIN without Select guards, TAIL
            # with them) -- the prim_func parser does not accept nested
            # Python defs.
            # ---- main loop: Select-free fast path (k = 2 .. K_MAX) ----
            for kk in T.serial(N_MAIN):
                k = kk + 2
                ta = 2 * k
                pta = cid + ta * CORE_EFF
                abs_ = pta % NT
                abbh = pta // NT
                abb = abbh // H
                abh = abbh % H
                as0 = abs_ * BS
                ptb = pta + CORE_EFF
                bbs = ptb % NT
                bbhb = ptb // NT
                bbb = bbhb // H
                bbh_ = bbhb % H
                bs0 = bbs * BS
                slotA = ta % NSLOT
                slotB = (ta + 1) % NSLOT
                # prefetch targets (next batch): always in-range here
                npa = pta + 2 * CORE_EFF
                nab = npa % NT
                nabh = npa // NT
                nbb_ = nabh // H
                nbh_ = nabh % H
                nas0 = nab * BS
                npb = npa + CORE_EFF
                nbb2 = npb % NT
                nbbh2 = npb // NT
                nbbb = nbbh2 // H
                nbh2 = nbbh2 % H
                nbs0 = nbb2 * BS

                # -- Vector: throttle, 2-BATCH lag (Cube's FIX-A of batch
                # k-2): the relays for batch k are produced a full batch
                # ahead of the Cube's arrival -- every auto relay flag is
                # pre-set, the Cube's flag waits are protocol-only --
                T.wait_cross_flag(12, "V")

                # -- Vector: chunk A: mask + relays (v was prefetched) --
                T.copy(a_half, a_f32)
                T.tile.mul(a_f32, a_f32, msk_ub)
                T.copy(a_f32, a_mask)
                T.copy(a_mask, workspace_3[cid, slotA, 0:BS, 0:BS])
                T.copy(v_half, workspace_4[cid, slotA, 0:BS, 0:bDV])
                T.copy(q_half, q_ub)
                T.tile.mul(g_ub, g_ub, LN2)
                T.tile.exp(g_ub, g_ub)
                T.tile.mul(q_ub, q_ub, scale)
                T.tile.mul(q_ub, q_ub, g_ub)
                T.copy(q_ub, qg_hi_ub)
                T.copy(qg_hi_ub, qg_hi_f32)
                T.tile.sub(q_ub, q_ub, qg_hi_f32)
                T.copy(q_ub, qg_lo_ub)
                T.copy(qg_hi_ub, workspace_1[cid, slotA, 0:BS, 0:DK])
                T.copy(qg_lo_ub, workspace_2[cid, slotA, 0:BS, 0:DK])

                # -- Vector: chunk B: mask + relays (v was prefetched) --
                T.copy(a_half2, a_f32)
                T.tile.mul(a_f32, a_f32, msk_ub)
                T.copy(a_f32, a_mask)
                T.copy(a_mask, workspace_3[cid, slotB, 0:BS, 0:BS])
                T.copy(v_half2, workspace_4[cid, slotB, 0:BS, 0:bDV])
                T.copy(q_half2, q_ub)
                T.tile.mul(g_ub2, g_ub2, LN2)
                T.tile.exp(g_ub2, g_ub2)
                T.tile.mul(q_ub, q_ub, scale)
                T.tile.mul(q_ub, q_ub, g_ub2)
                T.copy(q_ub, qg_hi_ub)
                T.copy(qg_hi_ub, qg_hi_f32)
                T.tile.sub(q_ub, q_ub, qg_hi_f32)
                T.copy(q_ub, qg_lo_ub)
                T.copy(qg_hi_ub, workspace_1[cid, slotB, 0:BS, 0:DK])
                T.copy(qg_lo_ub, workspace_2[cid, slotB, 0:BS, 0:DK])

                # -- Vector: PREFETCH the next batch's GM loads --
                T.copy(A[nbb_, nas0 : nas0 + BS, nbh_ * BS : (nbh_ + 1) * BS], a_half)
                T.copy(Q[nbb_, nas0 : nas0 + BS, nbh_ * DK : (nbh_ + 1) * DK], q_half)
                T.copy(G[nbb_, nas0 : nas0 + BS, nbh_ * DK : (nbh_ + 1) * DK], g_ub)
                T.copy(V[nbb_, nas0 : nas0 + BS, nbh_ * DV : (nbh_ + 1) * DV], v_half)
                T.copy(A[nbbb, nbs0 : nbs0 + BS, nbh2 * BS : (nbh2 + 1) * BS], a_half2)
                T.copy(Q[nbbb, nbs0 : nbs0 + BS, nbh2 * DK : (nbh2 + 1) * DK], q_half2)
                T.copy(G[nbbb, nbs0 : nbs0 + BS, nbh2 * DK : (nbh2 + 1) * DK], g_ub2)
                T.copy(V[nbbb, nbs0 : nbs0 + BS, nbh2 * DV : (nbh2 + 1) * DV], v_half2)

                # -- Cube: consume chunk A (ping-pong slot _a) --
                T.copy(Hstate[abb, abs_, abh, 0:DK, 0:bDV], H_l1_a)
                T.copy(workspace_4[cid, slotA, 0:BS, 0:bDV], V_l1_a)
                T.copy(workspace_3[cid, slotA, 0:BS, 0:BS], A_l1_a)
                T.gemm_v0(A_l1_a, V_l1_a, o_frag_a, init=True)
                T.copy(workspace_1[cid, slotA, 0:BS, 0:DK], qg_hi_l1_a)
                T.copy(workspace_2[cid, slotA, 0:BS, 0:DK], qg_lo_l1_a)
                T.gemm_v0(qg_hi_l1_a, H_l1_a, o_frag_a, init=False)
                T.gemm_v0(qg_lo_l1_a, H_l1_a, o_frag_a, init=False)
                # -- Cube: ISSUE chunk B's h/v loads NOW. Distinct buffer
                # names (_b) mean NO WAR edge against chunk A's gemms, so
                # the pass emits no M-wait here: the MTE2 queue drains
                # under A's FIX tail + B's flag waits instead of adding
                # its full latency to the serial chain. --
                T.copy(Hstate[bbb, bbs, bbh_, 0:DK, 0:bDV], H_l1_b)
                T.copy(workspace_4[cid, slotB, 0:BS, 0:bDV], V_l1_b)
                # -- Cube: FIX chunk A + one throttle signal per batch --
                T.copy(o_frag_a, O[abb, as0 : as0 + BS, abh * DV : (abh + 1) * DV])
                T.set_cross_flag("FIX", 12)
                # -- Cube: consume chunk B (ping-pong slot _b); the next
                # iteration's A-loads (_a) overlap B's gemms (_b) --
                T.copy(workspace_3[cid, slotB, 0:BS, 0:BS], A_l1_b)
                T.gemm_v0(A_l1_b, V_l1_b, o_frag_b, init=True)
                T.copy(workspace_1[cid, slotB, 0:BS, 0:DK], qg_hi_l1_b)
                T.copy(workspace_2[cid, slotB, 0:BS, 0:DK], qg_lo_l1_b)
                T.gemm_v0(qg_hi_l1_b, H_l1_b, o_frag_b, init=False)
                T.gemm_v0(qg_lo_l1_b, H_l1_b, o_frag_b, init=False)
                T.copy(o_frag_b, O[bbb, bs0 : bs0 + BS, bbh_ * DV : (bbh_ + 1) * DV])

            # ---- tail loop: Select-guarded redirects ----
            for kk2 in T.serial(N_TAIL):
                k = kk2 + max(1, min(K_MAX, NB)) + 1
                ta = 2 * k
                tida = cid + ta * CORE_EFF
                pta = T.if_then_else(tida < GRID, tida, cid)
                abs_ = pta % NT
                abbh = pta // NT
                abb = abbh // H
                abh = abbh % H
                as0 = abs_ * BS
                tidb = tida + CORE_EFF
                ptb = T.if_then_else(tidb < GRID, tidb, cid)
                bbs = ptb % NT
                bbhb = ptb // NT
                bbb = bbhb // H
                bbh_ = bbhb % H
                bs0 = bbs * BS
                slotA = ta % NSLOT
                slotB = (ta + 1) % NSLOT
                # prefetch targets: Select-guarded
                npa = T.if_then_else(tida + 2 * CORE_EFF < GRID, tida + 2 * CORE_EFF, cid)
                nab = npa % NT
                nabh = npa // NT
                nbb_ = nabh // H
                nbh_ = nabh % H
                nas0 = nab * BS
                npb = T.if_then_else(tida + 3 * CORE_EFF < GRID, tida + 3 * CORE_EFF, cid)
                nbb2 = npb % NT
                nbbh2 = npb // NT
                nbbb = nbbh2 // H
                nbh2 = nbbh2 % H
                nbs0 = nbb2 * BS

                # -- Vector: throttle (Cube's previous batch task-A FIX) --
                T.wait_cross_flag(12, "V")

                # -- Vector: chunk A: mask + relays (v was prefetched) --
                T.copy(a_half, a_f32)
                T.tile.mul(a_f32, a_f32, msk_ub)
                T.copy(a_f32, a_mask)
                T.copy(a_mask, workspace_3[cid, slotA, 0:BS, 0:BS])
                T.copy(v_half, workspace_4[cid, slotA, 0:BS, 0:bDV])
                T.copy(q_half, q_ub)
                T.tile.mul(g_ub, g_ub, LN2)
                T.tile.exp(g_ub, g_ub)
                T.tile.mul(q_ub, q_ub, scale)
                T.tile.mul(q_ub, q_ub, g_ub)
                T.copy(q_ub, qg_hi_ub)
                T.copy(qg_hi_ub, qg_hi_f32)
                T.tile.sub(q_ub, q_ub, qg_hi_f32)
                T.copy(q_ub, qg_lo_ub)
                T.copy(qg_hi_ub, workspace_1[cid, slotA, 0:BS, 0:DK])
                T.copy(qg_lo_ub, workspace_2[cid, slotA, 0:BS, 0:DK])

                # -- Vector: chunk B: mask + relays (v was prefetched) --
                T.copy(a_half2, a_f32)
                T.tile.mul(a_f32, a_f32, msk_ub)
                T.copy(a_f32, a_mask)
                T.copy(a_mask, workspace_3[cid, slotB, 0:BS, 0:BS])
                T.copy(v_half2, workspace_4[cid, slotB, 0:BS, 0:bDV])
                T.copy(q_half2, q_ub)
                T.tile.mul(g_ub2, g_ub2, LN2)
                T.tile.exp(g_ub2, g_ub2)
                T.tile.mul(q_ub, q_ub, scale)
                T.tile.mul(q_ub, q_ub, g_ub2)
                T.copy(q_ub, qg_hi_ub)
                T.copy(qg_hi_ub, qg_hi_f32)
                T.tile.sub(q_ub, q_ub, qg_hi_f32)
                T.copy(q_ub, qg_lo_ub)
                T.copy(qg_hi_ub, workspace_1[cid, slotB, 0:BS, 0:DK])
                T.copy(qg_lo_ub, workspace_2[cid, slotB, 0:BS, 0:DK])

                # -- Vector: PREFETCH the next batch's GM loads --
                T.copy(A[nbb_, nas0 : nas0 + BS, nbh_ * BS : (nbh_ + 1) * BS], a_half)
                T.copy(Q[nbb_, nas0 : nas0 + BS, nbh_ * DK : (nbh_ + 1) * DK], q_half)
                T.copy(G[nbb_, nas0 : nas0 + BS, nbh_ * DK : (nbh_ + 1) * DK], g_ub)
                T.copy(V[nbb_, nas0 : nas0 + BS, nbh_ * DV : (nbh_ + 1) * DV], v_half)
                T.copy(A[nbbb, nbs0 : nbs0 + BS, nbh2 * BS : (nbh2 + 1) * BS], a_half2)
                T.copy(Q[nbbb, nbs0 : nbs0 + BS, nbh2 * DK : (nbh2 + 1) * DK], q_half2)
                T.copy(G[nbbb, nbs0 : nbs0 + BS, nbh2 * DK : (nbh2 + 1) * DK], g_ub2)
                T.copy(V[nbbb, nbs0 : nbs0 + BS, nbh2 * DV : (nbh2 + 1) * DV], v_half2)

                # -- Cube: consume chunk A (ping-pong slot _a) --
                T.copy(Hstate[abb, abs_, abh, 0:DK, 0:bDV], H_l1_a)
                T.copy(workspace_4[cid, slotA, 0:BS, 0:bDV], V_l1_a)
                T.copy(workspace_3[cid, slotA, 0:BS, 0:BS], A_l1_a)
                T.gemm_v0(A_l1_a, V_l1_a, o_frag_a, init=True)
                T.copy(workspace_1[cid, slotA, 0:BS, 0:DK], qg_hi_l1_a)
                T.copy(workspace_2[cid, slotA, 0:BS, 0:DK], qg_lo_l1_a)
                T.gemm_v0(qg_hi_l1_a, H_l1_a, o_frag_a, init=False)
                T.gemm_v0(qg_lo_l1_a, H_l1_a, o_frag_a, init=False)
                # -- Cube: ISSUE chunk B's h/v loads now (no WAR vs _a) --
                T.copy(Hstate[bbb, bbs, bbh_, 0:DK, 0:bDV], H_l1_b)
                T.copy(workspace_4[cid, slotB, 0:BS, 0:bDV], V_l1_b)
                # -- Cube: FIX chunk A + throttle signal --
                T.copy(o_frag_a, O[abb, as0 : as0 + BS, abh * DV : (abh + 1) * DV])
                T.set_cross_flag("FIX", 12)
                # -- Cube: consume chunk B (ping-pong slot _b) --
                T.copy(workspace_3[cid, slotB, 0:BS, 0:BS], A_l1_b)
                T.gemm_v0(A_l1_b, V_l1_b, o_frag_b, init=True)
                T.copy(workspace_1[cid, slotB, 0:BS, 0:DK], qg_hi_l1_b)
                T.copy(workspace_2[cid, slotB, 0:BS, 0:DK], qg_lo_l1_b)
                T.gemm_v0(qg_hi_l1_b, H_l1_b, o_frag_b, init=False)
                T.gemm_v0(qg_lo_l1_b, H_l1_b, o_frag_b, init=False)
                T.copy(o_frag_b, O[bbb, bs0 : bs0 + BS, bbh_ * DV : (bbh_ + 1) * DV])

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
        v.view(B, S, H * DV),  # 3D view: vid-split lands on S rows (item 7)
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
