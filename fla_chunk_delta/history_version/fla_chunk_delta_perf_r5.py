"""fla_chunk_delta: KDA gated delta rule chunk state recurrence (fwd + bwd)
on Ascend 910B3.

Semantics (upstream flash-linear-attention chunk_gated_delta_rule_{fwd_h,
bwd_dhu}, fixed length, USE_G + USE_GK, chunk size 64): each (b, h, i_v)
chain maintains an fp32 state H/dH in R^{DK x DV} and recurses serially over
NT = S/64 chunks.

    fwd (t = 0..NT-1):  h[t] = H16; v_new = u - w @ H16      (ungated save)
                        v_new *= exp(g_last - g)            (token gate)
                        H = D * H + k^T @ v_new16
                        D[k] = exp(g_last + gk_last[k])      (folded decay)
    bwd (t = NT-1..0):  dh[t] = dH16; dv_part = k @ dH16
                        dv_part *= exp(g_last - g); dv2 = dv + dv_part
                        q~ = q * exp(g)
                        dH = D * dH + (q~^T @ do) * scale - w^T @ dv2

The per-chunk decay product exp(g_last)*exp(gk_last) is computed as a single
exp(g_last + gk_last) (relative diff ~1 ulp, far inside the output gates).

Developer mode, threads=2 fixed-core task loop + explicit named relay
workspaces (chunk_o structure) for the recurrences, plus a threads=1
Developer PREP kernel that materializes the per-chunk gate/decay tiles once
per launch. bshd inputs are consumed through zero-copy 3D views
[B, S, H*D] (metadata only, no host materialization).

Ascend toolchain adaptations (verified against this repo's sources and the
family precedents; see debug_log for the empirical record):

1. **Gate/decay tiles are precomputed by a threads=1 prep kernel** (the
   Stage-3 restructure; supersedes the Stage-2 in-kernel scalar extraction
   loops, which profiled at 79-84% of the AIV scalar pipe -- per-iteration
   MTE2_S/V_S/S_V flag pairs around every scalar statement, ~33 ns/iter).
   The prep kernel is threads=1 (no vid splitting), where the transpose /
   row_expand primitives are unconstrained:
   * GT[b, c, h, j] = exp(s * (g_last - g[s0+j]))        (token gate)
   * GE[b, c, h, j] = exp(s * g[s0+j])                    (bwd q~ gate)
   * GD[b, c, h, k] = exp(s * (gk_last[k] + g_last))      (state decay)
   with s = ln2 if use_exp2 else 1. Each tile is computed in the DIRECT
   difference form (no exp(a)*exp(b) factorization: the factored form
   overflows fp32 for |g| > ~88 where the direct form is exact). The main
   kernels then read a contiguous 256B/512B span per (chain, chunk) and
   gate via row_expand_{mul,sub} -- zero scalar statements in the hot loop.
   Gate/decay values are bit-comparable to the Stage-2 in-kernel form
   (same fp32 sub/mul/exp sequence).
2. vid-reduction (threads=2) halves dim 0 of every UB buffer and only
   rewrites an allow-listed op set (ascend_vid_reduction.cc IsTileOp:
   add/sub/mul/div/exp/fill/cast/...). T.tile.gather, T.tile.broadcast and
   arith_progression are NOT in the allow list and corrupt silently under
   vid splitting (fla_wy_fast #4) -- which is why the transpose lives in
   the threads=1 prep kernel. The vid pass applies the GM region offset at
   region dim (region_rank - ub_rank): the main-kernel GT/GE/GD reads use
   1D [BS]/[DK] UB targets so the offset lands on the token/channel Range
   dim (the W3/Hout pattern), never on a head index.
   Row gating in the main kernels uses
   T.tile.row_expand_mul_experiment (AscendC brcb + mul_mask), whose shapes
   flow entirely through access_ptr extents and therefore stay consistent
   after halving; fp32 operands need 256B rows ([*, 64] fp32), which holds
   for every gated tile here (DK=128 q~ is gated in two 64-column chunks).
3. gemm_v0 operands are half precision only (fp32 A/B produces garbage on
   this toolchain); every fp32<->fp16/bf16 conversion is a UB-internal
   two-step copy chain (GM<->UB cast-on-copy codegen is broken).
4. All C<->V exchanges go through explicit named workspace_* GM tensors
   (L2-resident, [CORE_EFF, NSLOT, ...]); AUTO_CV_SYNC inserts the
   cross-core flags by buffer-name match (chunk_o pattern). NSLOT=1 is a
   strict lockstep: correct by construction, Stage 3 owns the ring.
5. The serial chunk recurrence uses T.serial + GM relay (plain serial loops
   mis-sync on-chip V->C exchanges -- fla_wy_fast #1); the fixed-core task
   loop stripes chains over CORE_EFF = min(20, NCHAIN) cores with a tail
   guard `if tid < NCHAIN` (both core sides skip together, so the relay
   flag pairs stay balanced).
6. Support domain (wrapper asserts): S % 64 == 0, chunk_size == 64,
   DK in {64, 128} (row_expand row cap 248 + 64-column gate chunks),
   DV % 64 == 0. DK = 192/256 and non-64-multiple D are L1-extension
   territory (DESIGN SS5.4).
"""

import tilelang
import torch
from tilelang import language as T

# 2^g is computed as exp(g * ln2): exp2 has no Ascend backend mapping.
LN2 = 0.6931471805599453

# Ascend 910B3: 20 AI Cube cores (+ 40 AI Vector cores); threads=2 pairs
# each cube core with its 2 vector sub-blocks.
CORE_NUM = 20

# Relay slot ring per core. Stage-3 batch structure: each core processes
# chains in 2-chain batches; within a batch iteration (chunk i_t) the two
# in-flight chains use slots (2*i_t + jb) % NSLOT (jb = 0/1 chain position),
# so concurrent chains NEVER share a slot (a flat (k*NT+i_t)%NSLOT map
# degenerates when NT % NSLOT == 0 -- measured neutral in iter3 for exactly
# this reason) and slot reuse happens only at temporal distance NSLOT/2
# chunks per chain, far beyond the <=2-phase C-side lag. Cross-chain WAR on
# the relays is thereby decoupled (the NSLOT=1 lockstep serialized every
# chain's C<->V exchange; the AIC spent 98% of its wall time waiting on
# relay flags, msprof r2). No manual throttle flag: the V side still stalls
# on its own chain's C round-trip every chunk, bounding the flag set/wait
# imbalance well under the documented 15-set counter capacity.
NSLOT = 16

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # automatic C/V split
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,  # cross-core relay flags
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,  # intra-core pipe sync
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # UB/L1 lifetime planning
}


# ============================================================================
# prep kernel (threads=1): per-chunk gate/decay tiles GT/GE/GD
# ============================================================================
# Computes, per (b, chunk) task, the full [H]-wide tile set in the DIRECT
# difference form (no exp factorization -> no overflow narrowing):
#     GT[h, j] = exp(s * (g_last[h] - g[s0+j, h]))     (token gate, fwd+bwd)
#     GE[h, j] = exp(s * g[s0+j, h])                    (q~ gate, bwd only)
#     GD[h, k] = exp(s * (gk_last[h, k] + g_last[h]))   (state decay)
# The [BS, H] g block is loaded contiguously and transposed once
# (fp32 hardware transpose needs 32B-aligned dims -> both must be multiples
# of 8; BS=64 always is, H is staged padded to HP = ceil(H/8)*8 with the
# dead rows [H, HP) never written back to the H-exact GT/GE/GD tensors).
# row_expand_sub applies the per-row scalar (-s*g_last, shared by the GT
# and GD chains); row_expand needs 8..248 fp32 256B rows -> HP in [8, 248],
# i.e. H <= 248 (wrapper-asserted). threads=1 keeps every primitive
# vid-safe (transpose/broadcast/gather are unconstrained without vid
# splitting).
@tilelang.jit(out_idx=[2, 3, 4], pass_configs=PASS_CONFIGS)
def fla_chunk_delta_prep_kernel(
    B,
    S,
    H,
    DK,
    chunk_size=64,
    use_exp2=False,
    need_ge=False,
):
    """prep: materialize GT/GE/GD gate/decay tiles from (g, gk)."""
    BS = chunk_size
    NT = S // BS
    NPREP = B * NT
    CORE_EFF = min(CORE_NUM, NPREP)
    NTASK = (NPREP + CORE_EFF - 1) // CORE_EFF
    # decay halves: row_expand requires 256B fp32 rows -> [HP, 64] slabs
    NSLABD = DK // 64
    assert NSLABD in (1, 2), "DK in {64, 128} only (L0 domain)"
    # head-dim staging pad: transpose (32B-aligned dims) + row_expand
    # (8..248 rows) both need H rounded up to a multiple of 8
    HP = (H + 7) // 8 * 8
    assert 8 <= HP <= 248, "H <= 248 (prep row_expand/transpose staging cap)"
    # s = ln2 if use_exp2 else 1.0 (gate/decay exponent scale)
    s = LN2 if use_exp2 else 1.0

    @T.prim_func
    def main(
        G: T.Tensor((B, S, H), "float32"),  # type: ignore
        Gk: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        GE: T.Tensor((B, NT, H, BS), "float32") if need_ge else T.Tensor((1,), "float32"),  # type: ignore
        GT: T.Tensor((B, NT, H, BS), "float32"),  # type: ignore
        GD: T.Tensor((B, NT, H, DK), "float32"),  # type: ignore
    ):
        with T.Kernel(CORE_EFF, threads=1, is_npu=True) as (cid):
            # HP-padded staging: rows/cols [H, HP) hold dead garbage (their
            # exp results are computed but never written back -- no
            # cross-row contamination in any of these per-element ops)
            g2d = T.alloc_shared([BS, HP], "float32")  # g block staging
            gx2d = T.alloc_shared([BS, HP], "float32")  # +/-s*g domain
            gt_t = T.alloc_shared([HP, BS], "float32")  # transposed domain
            # (-s * g_last) as an [HP, 1] row vector: both row_expand_sub
            # sites consume it (GT: -s*g - (-s*g_last) = s*(g_last - g);
            # GD: s*gk - (-s*g_last) = s*(gk + g_last))
            glast2d = T.alloc_shared([HP, 1], "float32")
            ge_a = T.alloc_shared([HP, 64], "float32")  # decay slab A
            # slab B is allocated unconditionally (parser env registers
            # allocs outside Python-level `if`); for NSLABD == 1 its
            # statements are const-false dropped and the planner DCEs it
            ge_b = T.alloc_shared([HP, 64], "float32")
            tmp_row = T.alloc_shared([HP, 8], "float32")  # row_expand scratch

            for k_task in T.serial(NTASK):
                tid = cid + k_task * CORE_EFF
                if tid < NPREP:
                    bb = tid // NT
                    c = tid % NT
                    s0 = c * BS
                    n_t = s0 + BS - 1
                    # ---- load the g block (contiguous [BS, H]) + last row
                    T.copy(G[bb, s0 : s0 + BS, 0:H], g2d[0:BS, 0:H])
                    T.copy(G[bb, n_t, 0:H], glast2d[0:H, 0:1])
                    T.tile.mul(glast2d, glast2d, -s)
                    # ---- token gate GT = exp(s*(g_last - g))^T
                    # (-s*g) -> transpose -> row_expand_sub(-s*g_last)
                    T.tile.mul(gx2d, g2d, -s)
                    T.tile.transpose(gt_t, gx2d)
                    T.tile.row_expand_sub_experiment(gt_t, gt_t, glast2d, tmp_row)
                    T.tile.exp(gt_t, gt_t)
                    T.copy(gt_t[0:H, 0:BS], GT[bb, c, 0:H, 0:BS])
                    # ---- q~ gate GE = exp(s*g)^T (bwd only)
                    if need_ge:
                        T.tile.mul(gx2d, g2d, s)
                        T.tile.exp(gx2d, gx2d)
                        T.tile.transpose(gt_t, gx2d)
                        T.copy(gt_t[0:H, 0:BS], GE[bb, c, 0:H, 0:BS])
                    # ---- state decay GD = exp(s*(gk_last + g_last))
                    T.copy(Gk[bb, n_t, 0:H, 0:64], ge_a[0:H, 0:64])
                    T.tile.mul(ge_a, ge_a, s)
                    T.tile.row_expand_sub_experiment(ge_a, ge_a, glast2d, tmp_row)
                    T.tile.exp(ge_a, ge_a)
                    T.copy(ge_a[0:H, 0:64], GD[bb, c, 0:H, 0:64])
                    if NSLABD == 2:
                        T.copy(Gk[bb, n_t, 0:H, 64:DK], ge_b[0:H, 0:64])
                        T.tile.mul(ge_b, ge_b, s)
                        T.tile.row_expand_sub_experiment(ge_b, ge_b, glast2d, tmp_row)
                        T.tile.exp(ge_b, ge_b)
                        T.copy(ge_b[0:H, 0:64], GD[bb, c, 0:H, 64:DK])

    return main


# ============================================================================
# fwd kernel: chunk_gated_delta_rule_fwd_h (H state recurrence)
# ============================================================================
# 2-CHAIN BATCH with source-order phase restructure (Stage-3 iter4): the
# auto C/V split preserves statement order on each side, so the source
# interleaving below yields V order {pro_A, pro_B, c_A, c_B, f_A, f_B} and
# C order {b_A, b_B, e_A, e_B}: while the V side blocks on chain A's ws2
# cross-core wait, the Cube streams gemm1 for BOTH chains (and vice versa
# on the ws4 path) -- the per-chunk C<->V ping-pong (measured: per-chunk
# wall 4.6us vs serialized work sum 4.2us, ~zero overlap in msprof r2/r3)
# becomes a two-chain pipeline. Out-of-range chain slots are Select-redirected
# to the core's own first task (bit-identical relay/output recomputation --
# keeps every relay flag pair unconditionally balanced and avoids the
# PIPE_ALL drain the AUTO_SYNC pass inserts around every IfThenElse, the
# chunk_o "dominant per-task overhead"). Batch-internal buffers are paired
# by chain name (h_ub_A/B state residency, prologue staging, L1 operands,
# L0C fragments) so inter-chain WAR never forces a same-name serialization;
# strictly sequential per-chain compute domains stay shared.
# out_idx/workspace_idx use ABSOLUTE positions over the declared signature
# (K3, W3, U3, GT, GD, H0, Hout, Vnew, Ht, workspace_1..4). Disabled
# constexpr branches keep their signature slot with a [1] dummy tensor.
@tilelang.jit(out_idx=[6, 7, 8], workspace_idx=[9, 10, 11, 12], pass_configs=PASS_CONFIGS)
def fla_chunk_delta_fwd_kernel(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size=64,
    BV=64,
    use_exp2=False,
    use_initial_state=False,
    output_final_state=False,
    save_new_value=True,
):
    """fwd: h/v_new(/ht) from (k, w, u, GT, GD[/h0]) per (b, h, i_v) chain."""
    BS = chunk_size
    NT = S // BS
    NV = DV // BV
    NCHAIN = B * H * NV
    CORE_EFF = min(CORE_NUM, NCHAIN)
    NTASK = (NCHAIN + CORE_EFF - 1) // CORE_EFF
    # 2-chain batches over the striped task map (k_task 2*kb, 2*kb+1)
    NBATCH = (NTASK + 1) // 2

    @T.prim_func
    def main(
        K3: T.Tensor((B, S, H * DK), "float16"),  # type: ignore
        W3: T.Tensor((B, S, H * DK), "float16"),  # type: ignore
        U3: T.Tensor((B, S, H * DV), "float16"),  # type: ignore
        # gate/decay tiles from the prep kernel: [B, NT, H, BS]/[B, NT, H, DK]
        # fp32. Read as GT[bb, i_t, bh, 0:BS] -> gate_ub [BS] (1D target:
        # the vid region offset then lands on the token Range dim, never on
        # the head index -- the W3/Hout pattern).
        GT: T.Tensor((B, NT, H, BS), "float32"),  # type: ignore
        GD: T.Tensor((B, NT, H, DK), "float32"),  # type: ignore
        H0: T.Tensor((B, H, DK, DV) if use_initial_state else (1,), "float32"),  # type: ignore
        Hout: T.Tensor((B, NT, H, DK, DV), "float16"),  # type: ignore
        # 3D view [B, S, H*DV]: with a 4D [B,S,H,DV] tensor the vid row offset
        # lands on the H index (chunk_o #8) -- the wrapper views it back.
        Vnew: T.Tensor((B, S, H * DV) if save_new_value else (1,), "float16"),  # type: ignore
        Ht: T.Tensor((B, H, DK, DV) if output_final_state else (1,), "float32"),  # type: ignore
        workspace_1: T.Tensor((CORE_EFF, NSLOT, DK, BV), "float16"),  # type: ignore
        workspace_2: T.Tensor((CORE_EFF, NSLOT, BS, BV), "float32"),  # type: ignore
        workspace_3: T.Tensor((CORE_EFF, NSLOT, BS, BV), "float16"),  # type: ignore
        workspace_4: T.Tensor((CORE_EFF, NSLOT, DK, BV), "float32"),  # type: ignore
    ):
        with T.Kernel(CORE_EFF, threads=2, is_npu=True) as (cid):
            # ---- V-side buffers (compiler-mapped UB; vid-halved by the pass)
            # chain-resident state + prologue staging are PAIRED per batch
            # chain (A/B); strictly sequential compute domains stay shared.
            h_ub_a = T.alloc_shared([DK, BV], "float32")  # state H, chain A
            h_ub_b = T.alloc_shared([DK, BV], "float32")  # state H, chain B
            h16_ub_a = T.alloc_shared([DK, BV], "float16")
            h16_ub_b = T.alloc_shared([DK, BV], "float16")
            u16_ub_a = T.alloc_shared([BS, BV], "float16")  # u prefetch
            u16_ub_b = T.alloc_shared([BS, BV], "float16")
            u_ub = T.alloc_shared([BS, BV], "float32")  # shared compute
            vn_ub = T.alloc_shared([BS, BV], "float32")  # shared ungated v_new
            vn_g_ub = T.alloc_shared([BS, BV], "float32")  # shared gated v_new
            vn16_ub_a = T.alloc_shared([BS, BV], "float16")  # ws3 staging (gated)
            vn16_ub_b = T.alloc_shared([BS, BV], "float16")
            vn16v_ub_a = T.alloc_shared([BS, BV], "float16")  # Vnew staging (ungated)
            vn16v_ub_b = T.alloc_shared([BS, BV], "float16")
            bv_ub = T.alloc_shared([BS, BV], "float32")  # b_v readback (shared)
            kh_ub = T.alloc_shared([DK, BV], "float32")  # kH readback (shared)
            gate_ub_a = T.alloc_shared([BS], "float32")  # token gate (from GT)
            gate_ub_b = T.alloc_shared([BS], "float32")
            dsrc_ub_a = T.alloc_shared([DK], "float32")  # decay (from GD)
            dsrc_ub_b = T.alloc_shared([DK], "float32")
            tmp_gate = T.alloc_shared([BS, 8], "float32")  # row_expand scratch
            tmp_decay = T.alloc_shared([DK, 8], "float32")
            # ---- C-side buffers (L1 gemm operands / L0C accumulators),
            # paired per chain so chain B's loads never WAR chain A's gemms
            w_l1_a = T.alloc_shared([BS, DK], "float16")
            w_l1_b = T.alloc_shared([BS, DK], "float16")
            k_l1_a = T.alloc_shared([BS, DK], "float16")
            k_l1_b = T.alloc_shared([BS, DK], "float16")
            h16_l1_a = T.alloc_shared([DK, BV], "float16")
            h16_l1_b = T.alloc_shared([DK, BV], "float16")
            v16_l1_a = T.alloc_shared([BS, BV], "float16")
            v16_l1_b = T.alloc_shared([BS, BV], "float16")
            bv_frag_a = T.alloc_fragment([BS, BV], "float32")
            bv_frag_b = T.alloc_fragment([BS, BV], "float32")
            kh_frag_a = T.alloc_fragment([DK, BV], "float32")
            kh_frag_b = T.alloc_fragment([DK, BV], "float32")

            for kb in T.serial(NBATCH):
                # Select-redirect OOB chain slots to the core's own first
                # task (tid = cid): the duplicate recomputes chain cid's
                # recurrence from its H0/zero init -- bit-identical to the
                # batch-0 run, so every output/relay write is idempotent,
                # and the relay flag pairs stay balanced WITHOUT IfThenElse.
                tida = cid + (2 * kb) * CORE_EFF
                pta = T.if_then_else(tida < NCHAIN, tida, cid)
                tidb = cid + (2 * kb + 1) * CORE_EFF
                ptb = T.if_then_else(tidb < NCHAIN, tidb, cid)
                # chain A decomposition
                iva = pta % NV
                inha = pta // NV
                bba = inha // H
                bha = inha % H
                v0a = iva * BV
                # chain B decomposition
                ivb = ptb % NV
                inhb = ptb // NV
                bbb = inhb // H
                bhb = inhb % H
                v0b = ivb * BV
                # per-batch state init
                if use_initial_state:
                    T.copy(H0[bba, bha, 0:DK, v0a : v0a + BV], h_ub_a)
                    T.copy(H0[bbb, bhb, 0:DK, v0b : v0b + BV], h_ub_b)
                else:
                    T.tile.fill(h_ub_a, 0.0)
                    T.tile.fill(h_ub_b, 0.0)
                for i_t in T.serial(NT):
                    s0a = i_t * BS
                    s0b = i_t * BS
                    # batch-internal ring slots: distinct per in-flight chain
                    sla = (2 * i_t) % NSLOT
                    slb = (2 * i_t + 1) % NSLOT
                    # ===== V prologue: issue both chains' snapshot + ws1
                    # relay + independent prefetches (MTE queues drain while
                    # the S-pipe later blocks on the ws2/ws4 waits) =====
                    T.copy(h_ub_a, h16_ub_a)
                    T.copy(h16_ub_a, workspace_1[cid, sla, 0:DK, 0:BV])
                    T.copy(h16_ub_a, Hout[bba, i_t, bha, 0:DK, v0a : v0a + BV])
                    T.copy(
                        U3[bba, s0a : s0a + BS, bha * DV + v0a : bha * DV + v0a + BV],
                        u16_ub_a,
                    )
                    T.copy(GT[bba, i_t, bha, 0:BS], gate_ub_a)
                    T.copy(GD[bba, i_t, bha, 0:DK], dsrc_ub_a)
                    T.copy(h_ub_b, h16_ub_b)
                    T.copy(h16_ub_b, workspace_1[cid, slb, 0:DK, 0:BV])
                    T.copy(h16_ub_b, Hout[bbb, i_t, bhb, 0:DK, v0b : v0b + BV])
                    T.copy(
                        U3[bbb, s0b : s0b + BS, bhb * DV + v0b : bhb * DV + v0b + BV],
                        u16_ub_b,
                    )
                    T.copy(GT[bbb, i_t, bhb, 0:BS], gate_ub_b)
                    T.copy(GD[bbb, i_t, bhb, 0:DK], dsrc_ub_b)
                    # ===== C phase b: gemm1 for BOTH chains (ws1 flags are
                    # long set; the M-pipe streams gemm1_A then gemm1_B) =====
                    T.copy(W3[bba, s0a : s0a + BS, bha * DK : (bha + 1) * DK], w_l1_a)
                    T.copy(workspace_1[cid, sla, 0:DK, 0:BV], h16_l1_a)
                    T.gemm_v0(w_l1_a, h16_l1_a, bv_frag_a, init=True)
                    T.copy(bv_frag_a, workspace_2[cid, sla, 0:BS, 0:BV])
                    T.copy(W3[bbb, s0b : s0b + BS, bhb * DK : (bhb + 1) * DK], w_l1_b)
                    T.copy(workspace_1[cid, slb, 0:DK, 0:BV], h16_l1_b)
                    T.gemm_v0(w_l1_b, h16_l1_b, bv_frag_b, init=True)
                    T.copy(bv_frag_b, workspace_2[cid, slb, 0:BS, 0:BV])
                    # ===== V phase c (chain A then B): v_new = u - b_v;
                    # token gate; UNGATED save; ws3 relay =====
                    T.copy(u16_ub_a, u_ub)
                    T.copy(workspace_2[cid, sla, 0:BS, 0:BV], bv_ub)
                    T.tile.sub(vn_ub, u_ub, bv_ub)
                    T.tile.row_expand_mul_experiment(vn_g_ub, vn_ub, gate_ub_a, tmp_gate)
                    T.copy(vn_g_ub, vn16_ub_a)
                    T.copy(vn16_ub_a, workspace_3[cid, sla, 0:BS, 0:BV])
                    if save_new_value:
                        T.copy(vn_ub, vn16v_ub_a)
                        T.copy(
                            vn16v_ub_a,
                            Vnew[bba, s0a : s0a + BS, bha * DV + v0a : bha * DV + v0a + BV],
                        )
                    T.copy(u16_ub_b, u_ub)
                    T.copy(workspace_2[cid, slb, 0:BS, 0:BV], bv_ub)
                    T.tile.sub(vn_ub, u_ub, bv_ub)
                    T.tile.row_expand_mul_experiment(vn_g_ub, vn_ub, gate_ub_b, tmp_gate)
                    T.copy(vn_g_ub, vn16_ub_b)
                    T.copy(vn16_ub_b, workspace_3[cid, slb, 0:BS, 0:BV])
                    if save_new_value:
                        T.copy(vn_ub, vn16v_ub_b)
                        T.copy(
                            vn16v_ub_b,
                            Vnew[bbb, s0b : s0b + BS, bhb * DV + v0b : bhb * DV + v0b + BV],
                        )
                    # ===== C phase e: gemm2 for BOTH chains =====
                    T.copy(K3[bba, s0a : s0a + BS, bha * DK : (bha + 1) * DK], k_l1_a)
                    T.copy(workspace_3[cid, sla, 0:BS, 0:BV], v16_l1_a)
                    T.gemm_v0(k_l1_a, v16_l1_a, kh_frag_a, transpose_A=True, init=True)
                    T.copy(kh_frag_a, workspace_4[cid, sla, 0:DK, 0:BV])
                    T.copy(K3[bbb, s0b : s0b + BS, bhb * DK : (bhb + 1) * DK], k_l1_b)
                    T.copy(workspace_3[cid, slb, 0:BS, 0:BV], v16_l1_b)
                    T.gemm_v0(k_l1_b, v16_l1_b, kh_frag_b, transpose_A=True, init=True)
                    T.copy(kh_frag_b, workspace_4[cid, slb, 0:DK, 0:BV])
                    # ===== V phase f (chain A then B): decay + state update =====
                    T.copy(workspace_4[cid, sla, 0:DK, 0:BV], kh_ub)
                    T.tile.row_expand_mul_experiment(h_ub_a, h_ub_a, dsrc_ub_a, tmp_decay)
                    T.tile.add(h_ub_a, h_ub_a, kh_ub)
                    T.copy(workspace_4[cid, slb, 0:DK, 0:BV], kh_ub)
                    T.tile.row_expand_mul_experiment(h_ub_b, h_ub_b, dsrc_ub_b, tmp_decay)
                    T.tile.add(h_ub_b, h_ub_b, kh_ub)
                if output_final_state:
                    T.copy(h_ub_a, Ht[bba, bha, 0:DK, v0a : v0a + BV])
                    T.copy(h_ub_b, Ht[bbb, bhb, 0:DK, v0b : v0b + BV])

    return main


# ============================================================================
# bwd kernel: chunk_gated_delta_rule_bwd_dhu (dH state-gradient recurrence)
# ============================================================================
# 2-chain batch + source-order phase restructure (see the fwd note). The
# chain-independent q~ = q*exp(g) gating is hoisted into the V PROLOGUE (its
# ws3 relay is thereby long-set before the Cube reaches the qdo gemm), and
# the Cube's e-phase is split e1(qdo)/e2(wdv2) per chain pair so the M-pipe
# streams {gemm1_A, gemm1_B, qdo_A, qdo_B} before the ws4-dependent wdv2
# pair -- only the true data dependency (V's c-phase dv2) blocks the Cube.
@tilelang.jit(out_idx=[9, 10, 11], workspace_idx=[12, 13, 14, 15, 16, 17, 18], pass_configs=PASS_CONFIGS)
def fla_chunk_delta_bwd_kernel(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size=64,
    BV=64,
    scale=None,
    use_exp2=False,
    use_final_state_gradient=False,
    use_initial_state=False,
):
    """bwd: dh/dv2(/dh0) from (q, k, w, do, dv, GT, GE, GD[/dht, h0])."""
    if scale is None:
        scale = DK**-0.5
    BS = chunk_size
    NT = S // BS
    NV = DV // BV
    NCHAIN = B * H * NV
    CORE_EFF = min(CORE_NUM, NCHAIN)
    NTASK = (NCHAIN + CORE_EFF - 1) // CORE_EFF
    NBATCH = (NTASK + 1) // 2
    # q~ gating runs on [BS, 64] fp32 column chunks (row_expand 256B rows)
    NQK = DK // 64
    assert NQK in (1, 2), "DK in {64, 128} only (L0 domain)"

    @T.prim_func
    def main(
        Q3: T.Tensor((B, S, H * DK), "bfloat16"),  # type: ignore
        K3: T.Tensor((B, S, H * DK), "bfloat16"),  # type: ignore
        W3: T.Tensor((B, S, H * DK), "bfloat16"),  # type: ignore
        Do3: T.Tensor((B, S, H * DV), "bfloat16"),  # type: ignore
        Dv3: T.Tensor((B, S, H * DV), "bfloat16"),  # type: ignore
        # gate/decay tiles from the prep kernel (see the fwd note for the
        # 1D-target vid rule)
        GT: T.Tensor((B, NT, H, BS), "float32"),  # type: ignore
        GE: T.Tensor((B, NT, H, BS), "float32"),  # type: ignore
        GD: T.Tensor((B, NT, H, DK), "float32"),  # type: ignore
        Dht: T.Tensor((B, H, DK, DV) if use_final_state_gradient else (1,), "float32"),  # type: ignore
        Dh: T.Tensor((B, NT, H, DK, DV), "bfloat16"),  # type: ignore
        Dh0: T.Tensor((B, H, DK, DV) if use_initial_state else (1,), "float32"),  # type: ignore
        # 3D view [B, S, H*DV] (see the fwd Vnew note; wrapper views it back)
        Dv2: T.Tensor((B, S, H * DV), "bfloat16"),  # type: ignore
        workspace_1: T.Tensor((CORE_EFF, NSLOT, DK, BV), "bfloat16"),  # type: ignore
        workspace_2: T.Tensor((CORE_EFF, NSLOT, BS, BV), "float32"),  # type: ignore
        workspace_3: T.Tensor((CORE_EFF, NSLOT, BS, 64), "bfloat16"),  # type: ignore
        workspace_3b: T.Tensor((CORE_EFF, NSLOT, BS, 64), "bfloat16"),  # type: ignore
        workspace_4: T.Tensor((CORE_EFF, NSLOT, BS, BV), "bfloat16"),  # type: ignore
        workspace_5: T.Tensor((CORE_EFF, NSLOT, DK, BV), "float32"),  # type: ignore
        workspace_6: T.Tensor((CORE_EFF, NSLOT, DK, BV), "float32"),  # type: ignore
    ):
        with T.Kernel(CORE_EFF, threads=2, is_npu=True) as (cid):
            # ---- V-side buffers (paired where the two in-flight chains
            # need independent staging; sequential compute stays shared)
            dh_ub_a = T.alloc_shared([DK, BV], "float32")  # dH state, chain A
            dh_ub_b = T.alloc_shared([DK, BV], "float32")  # dH state, chain B
            dh16_ub_a = T.alloc_shared([DK, BV], "bfloat16")
            dh16_ub_b = T.alloc_shared([DK, BV], "bfloat16")
            dv16_ub_a = T.alloc_shared([BS, BV], "bfloat16")  # dv prefetch
            dv16_ub_b = T.alloc_shared([BS, BV], "bfloat16")
            dv_ub = T.alloc_shared([BS, BV], "float32")  # shared compute
            dvp_ub = T.alloc_shared([BS, BV], "float32")  # dv_part (shared)
            dv2_ub = T.alloc_shared([BS, BV], "float32")  # shared compute
            dv2_16_ub_a = T.alloc_shared([BS, BV], "bfloat16")  # ws4 staging
            dv2_16_ub_b = T.alloc_shared([BS, BV], "bfloat16")
            q_half_a_a = T.alloc_shared([BS, 64], "bfloat16")  # q col-0 prefetch
            q_half_a_b = T.alloc_shared([BS, 64], "bfloat16")
            # chunk-b halves are allocated unconditionally (buffer allocs
            # inside a Python-level `if` are not registered by the parser
            # env); for NQK == 1 the b statements are const-false dropped
            # and the planner DCEs the dead buffers
            q_half_b_a = T.alloc_shared([BS, 64], "bfloat16")
            q_half_b_b = T.alloc_shared([BS, 64], "bfloat16")
            q_ub_a = T.alloc_shared([BS, 64], "float32")  # shared q~ domain
            q_ub_b = T.alloc_shared([BS, 64], "float32")
            qt16_a = T.alloc_shared([BS, 64], "bfloat16")  # shared staging
            qt16_b = T.alloc_shared([BS, 64], "bfloat16")
            gate_ub_a = T.alloc_shared([BS], "float32")  # token gate (from GT)
            gate_ub_b = T.alloc_shared([BS], "float32")
            gexp_ub_a = T.alloc_shared([BS], "float32")  # exp(g) (from GE)
            gexp_ub_b = T.alloc_shared([BS], "float32")
            dsrc_ub_a = T.alloc_shared([DK], "float32")  # decay (from GD)
            dsrc_ub_b = T.alloc_shared([DK], "float32")
            qdo_ub = T.alloc_shared([DK, BV], "float32")  # shared readbacks
            wdv2_ub = T.alloc_shared([DK, BV], "float32")
            tmp_gate = T.alloc_shared([BS, 8], "float32")
            tmp_qa = T.alloc_shared([BS, 8], "float32")
            tmp_qb = T.alloc_shared([BS, 8], "float32")
            tmp_decay = T.alloc_shared([DK, 8], "float32")
            # ---- C-side buffers, paired per chain
            k_l1_a = T.alloc_shared([BS, DK], "bfloat16")
            k_l1_b = T.alloc_shared([BS, DK], "bfloat16")
            w_l1_a = T.alloc_shared([BS, DK], "bfloat16")
            w_l1_b = T.alloc_shared([BS, DK], "bfloat16")
            do_l1_a = T.alloc_shared([BS, BV], "bfloat16")
            do_l1_b = T.alloc_shared([BS, BV], "bfloat16")
            dv2_16_l1_a = T.alloc_shared([BS, BV], "bfloat16")
            dv2_16_l1_b = T.alloc_shared([BS, BV], "bfloat16")
            dh16_l1_a = T.alloc_shared([DK, BV], "bfloat16")
            dh16_l1_b = T.alloc_shared([DK, BV], "bfloat16")
            qt16_l1_a = T.alloc_shared([BS, DK], "bfloat16")
            qt16_l1_b = T.alloc_shared([BS, DK], "bfloat16")
            dvp_frag_a = T.alloc_fragment([BS, BV], "float32")
            dvp_frag_b = T.alloc_fragment([BS, BV], "float32")
            qdo_frag_a = T.alloc_fragment([DK, BV], "float32")
            qdo_frag_b = T.alloc_fragment([DK, BV], "float32")
            wdv2_frag_a = T.alloc_fragment([DK, BV], "float32")
            wdv2_frag_b = T.alloc_fragment([DK, BV], "float32")

            for kb in T.serial(NBATCH):
                # Select-redirect OOB chain slots (see the fwd note)
                tida = cid + (2 * kb) * CORE_EFF
                pta = T.if_then_else(tida < NCHAIN, tida, cid)
                tidb = cid + (2 * kb + 1) * CORE_EFF
                ptb = T.if_then_else(tidb < NCHAIN, tidb, cid)
                iva = pta % NV
                inha = pta // NV
                bba = inha // H
                bha = inha % H
                v0a = iva * BV
                ivb = ptb % NV
                inhb = ptb // NV
                bbb = inhb // H
                bhb = inhb % H
                v0b = ivb * BV
                if use_final_state_gradient:
                    T.copy(Dht[bba, bha, 0:DK, v0a : v0a + BV], dh_ub_a)
                    T.copy(Dht[bbb, bhb, 0:DK, v0b : v0b + BV], dh_ub_b)
                else:
                    T.tile.fill(dh_ub_a, 0.0)
                    T.tile.fill(dh_ub_b, 0.0)
                for it_f in T.serial(NT):
                    i_t = NT - 1 - it_f
                    s0a = i_t * BS
                    s0b = i_t * BS
                    sla = (2 * it_f) % NSLOT
                    slb = (2 * it_f + 1) % NSLOT
                    # ===== V prologue chain A: snapshot + ws1 relay +
                    # prefetches + INDEPENDENT q~ gating and ws3 relay =====
                    T.copy(dh_ub_a, dh16_ub_a)
                    T.copy(dh16_ub_a, workspace_1[cid, sla, 0:DK, 0:BV])
                    T.copy(dh16_ub_a, Dh[bba, i_t, bha, 0:DK, v0a : v0a + BV])
                    T.copy(
                        Dv3[bba, s0a : s0a + BS, bha * DV + v0a : bha * DV + v0a + BV],
                        dv16_ub_a,
                    )
                    T.copy(GT[bba, i_t, bha, 0:BS], gate_ub_a)
                    T.copy(GE[bba, i_t, bha, 0:BS], gexp_ub_a)
                    T.copy(GD[bba, i_t, bha, 0:DK], dsrc_ub_a)
                    T.copy(
                        Q3[bba, s0a : s0a + BS, bha * DK : bha * DK + 64],
                        q_half_a_a,
                    )
                    if NQK == 2:
                        T.copy(
                            Q3[bba, s0a : s0a + BS, bha * DK + 64 : bha * DK + 128],
                            q_half_b_a,
                        )
                    T.copy(q_half_a_a, q_ub_a)
                    T.tile.row_expand_mul_experiment(q_ub_a, q_ub_a, gexp_ub_a, tmp_qa)
                    T.copy(q_ub_a, qt16_a)
                    T.copy(qt16_a, workspace_3[cid, sla, 0:BS, 0:64])
                    if NQK == 2:
                        T.copy(q_half_b_a, q_ub_b)
                        T.tile.row_expand_mul_experiment(q_ub_b, q_ub_b, gexp_ub_a, tmp_qb)
                        T.copy(q_ub_b, qt16_b)
                        T.copy(qt16_b, workspace_3b[cid, sla, 0:BS, 0:64])
                    # ===== V prologue chain B: same =====
                    T.copy(dh_ub_b, dh16_ub_b)
                    T.copy(dh16_ub_b, workspace_1[cid, slb, 0:DK, 0:BV])
                    T.copy(dh16_ub_b, Dh[bbb, i_t, bhb, 0:DK, v0b : v0b + BV])
                    T.copy(
                        Dv3[bbb, s0b : s0b + BS, bhb * DV + v0b : bhb * DV + v0b + BV],
                        dv16_ub_b,
                    )
                    T.copy(GT[bbb, i_t, bhb, 0:BS], gate_ub_b)
                    T.copy(GE[bbb, i_t, bhb, 0:BS], gexp_ub_b)
                    T.copy(GD[bbb, i_t, bhb, 0:DK], dsrc_ub_b)
                    T.copy(
                        Q3[bbb, s0b : s0b + BS, bhb * DK : bhb * DK + 64],
                        q_half_a_b,
                    )
                    if NQK == 2:
                        T.copy(
                            Q3[bbb, s0b : s0b + BS, bhb * DK + 64 : bhb * DK + 128],
                            q_half_b_b,
                        )
                    T.copy(q_half_a_b, q_ub_a)
                    T.tile.row_expand_mul_experiment(q_ub_a, q_ub_a, gexp_ub_b, tmp_qa)
                    T.copy(q_ub_a, qt16_a)
                    T.copy(qt16_a, workspace_3[cid, slb, 0:BS, 0:64])
                    if NQK == 2:
                        T.copy(q_half_b_b, q_ub_b)
                        T.tile.row_expand_mul_experiment(q_ub_b, q_ub_b, gexp_ub_b, tmp_qb)
                        T.copy(q_ub_b, qt16_b)
                        T.copy(qt16_b, workspace_3b[cid, slb, 0:BS, 0:64])
                    # ===== C phase b: dv_part gemm for BOTH chains =====
                    T.copy(K3[bba, s0a : s0a + BS, bha * DK : (bha + 1) * DK], k_l1_a)
                    T.copy(workspace_1[cid, sla, 0:DK, 0:BV], dh16_l1_a)
                    T.gemm_v0(k_l1_a, dh16_l1_a, dvp_frag_a, init=True)
                    T.copy(dvp_frag_a, workspace_2[cid, sla, 0:BS, 0:BV])
                    T.copy(K3[bbb, s0b : s0b + BS, bhb * DK : (bhb + 1) * DK], k_l1_b)
                    T.copy(workspace_1[cid, slb, 0:DK, 0:BV], dh16_l1_b)
                    T.gemm_v0(k_l1_b, dh16_l1_b, dvp_frag_b, init=True)
                    T.copy(dvp_frag_b, workspace_2[cid, slb, 0:BS, 0:BV])
                    # ===== V phase c (A then B): gate dv_part; dv2 = dv +
                    # dv_part; Dv2 write; ws4 relay =====
                    T.copy(workspace_2[cid, sla, 0:BS, 0:BV], dvp_ub)
                    T.tile.row_expand_mul_experiment(dvp_ub, dvp_ub, gate_ub_a, tmp_gate)
                    T.copy(dv16_ub_a, dv_ub)
                    T.tile.add(dv2_ub, dv_ub, dvp_ub)
                    T.copy(dv2_ub, dv2_16_ub_a)
                    T.copy(dv2_16_ub_a, workspace_4[cid, sla, 0:BS, 0:BV])
                    T.copy(
                        dv2_16_ub_a,
                        Dv2[bba, s0a : s0a + BS, bha * DV + v0a : bha * DV + v0a + BV],
                    )
                    T.copy(workspace_2[cid, slb, 0:BS, 0:BV], dvp_ub)
                    T.tile.row_expand_mul_experiment(dvp_ub, dvp_ub, gate_ub_b, tmp_gate)
                    T.copy(dv16_ub_b, dv_ub)
                    T.tile.add(dv2_ub, dv_ub, dvp_ub)
                    T.copy(dv2_ub, dv2_16_ub_b)
                    T.copy(dv2_16_ub_b, workspace_4[cid, slb, 0:BS, 0:BV])
                    T.copy(
                        dv2_16_ub_b,
                        Dv2[bbb, s0b : s0b + BS, bhb * DV + v0b : bhb * DV + v0b + BV],
                    )
                    # ===== C phase e1: qdo gemm for BOTH chains (ws3 flags
                    # long set by the V prologue) =====
                    T.copy(
                        Do3[bba, s0a : s0a + BS, bha * DV + v0a : bha * DV + v0a + BV],
                        do_l1_a,
                    )
                    T.copy(workspace_3[cid, sla, 0:BS, 0:64], qt16_l1_a[0:BS, 0:64])
                    if NQK == 2:
                        T.copy(workspace_3b[cid, sla, 0:BS, 0:64], qt16_l1_a[0:BS, 64:DK])
                    T.gemm_v0(qt16_l1_a, do_l1_a, qdo_frag_a, transpose_A=True, init=True)
                    T.copy(qdo_frag_a, workspace_5[cid, sla, 0:DK, 0:BV])
                    T.copy(
                        Do3[bbb, s0b : s0b + BS, bhb * DV + v0b : bhb * DV + v0b + BV],
                        do_l1_b,
                    )
                    T.copy(workspace_3[cid, slb, 0:BS, 0:64], qt16_l1_b[0:BS, 0:64])
                    if NQK == 2:
                        T.copy(workspace_3b[cid, slb, 0:BS, 0:64], qt16_l1_b[0:BS, 64:DK])
                    T.gemm_v0(qt16_l1_b, do_l1_b, qdo_frag_b, transpose_A=True, init=True)
                    T.copy(qdo_frag_b, workspace_5[cid, slb, 0:DK, 0:BV])
                    # ===== C phase e2: wdv2 gemm for BOTH chains =====
                    T.copy(W3[bba, s0a : s0a + BS, bha * DK : (bha + 1) * DK], w_l1_a)
                    T.copy(workspace_4[cid, sla, 0:BS, 0:BV], dv2_16_l1_a)
                    T.gemm_v0(w_l1_a, dv2_16_l1_a, wdv2_frag_a, transpose_A=True, init=True)
                    T.copy(wdv2_frag_a, workspace_6[cid, sla, 0:DK, 0:BV])
                    T.copy(W3[bbb, s0b : s0b + BS, bhb * DK : (bhb + 1) * DK], w_l1_b)
                    T.copy(workspace_4[cid, slb, 0:BS, 0:BV], dv2_16_l1_b)
                    T.gemm_v0(w_l1_b, dv2_16_l1_b, wdv2_frag_b, transpose_A=True, init=True)
                    T.copy(wdv2_frag_b, workspace_6[cid, slb, 0:DK, 0:BV])
                    # ===== V phase f (A then B): dH = D*dH + qdo*scale - wdv2 =====
                    T.copy(workspace_5[cid, sla, 0:DK, 0:BV], qdo_ub)
                    T.copy(workspace_6[cid, sla, 0:DK, 0:BV], wdv2_ub)
                    T.tile.mul(qdo_ub, qdo_ub, scale)
                    T.tile.row_expand_mul_experiment(dh_ub_a, dh_ub_a, dsrc_ub_a, tmp_decay)
                    T.tile.add(dh_ub_a, dh_ub_a, qdo_ub)
                    T.tile.sub(dh_ub_a, dh_ub_a, wdv2_ub)
                    T.copy(workspace_5[cid, slb, 0:DK, 0:BV], qdo_ub)
                    T.copy(workspace_6[cid, slb, 0:DK, 0:BV], wdv2_ub)
                    T.tile.mul(qdo_ub, qdo_ub, scale)
                    T.tile.row_expand_mul_experiment(dh_ub_b, dh_ub_b, dsrc_ub_b, tmp_decay)
                    T.tile.add(dh_ub_b, dh_ub_b, qdo_ub)
                    T.tile.sub(dh_ub_b, dh_ub_b, wdv2_ub)
                if use_initial_state:
                    T.copy(dh_ub_a, Dh0[bba, bha, 0:DK, v0a : v0a + BV])
                    T.copy(dh_ub_b, Dh0[bbb, bhb, 0:DK, v0b : v0b + BV])

    return main


# ============================================================================
# Host wrappers (upstream naming: chunk_gated_delta_rule_{fwd_h, bwd_dhu} ->
# fla_chunk_delta_{fwd, bwd}; signature aligned with the proto schema).
# Only zero-copy metadata views + kernel calls here (the JIT adapter owns
# output/workspace allocation). Each wrapper launches the threads=1 prep
# kernel (GT/GE/GD tiles) followed by the recurrence kernel; both launches
# are inside the do_bench timed region (bench accounting stays honest).
# ============================================================================
def _check_domain(B, S, H, DK, DV, chunk_size):
    assert chunk_size == 64, "chunk_size fixed at 64 (block_S semantic)"
    assert S % chunk_size == 0, "S must be divisible by chunk_size"
    assert DK in (64, 128), "DK in {64, 128} (L0 domain; 192/256 = L1 extension)"
    assert DV % 64 == 0, "DV must be a multiple of 64 (BV=64)"
    assert DK <= 256, "upstream assert K <= 256"
    # prep staging cap: transpose (32B-aligned dims) + row_expand (8..248
    # fp32 256B rows) round the head dim up to a multiple of 8
    assert H <= 248, "H <= 248 (prep gate/decay staging row cap)"


def _dummy(device, dtype):
    return torch.empty(1, dtype=dtype, device=device)


def fla_chunk_delta_fwd(
    k,
    w,
    u,
    g,
    gk,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    save_new_value=True,
    use_exp2=False,
):
    """fwd host API. k/w [B,S,H,DK] fp16, u [B,S,H,DV] fp16, g [B,S,H] fp32,
    gk [B,S,H,DK] fp32 (chunk-cumulative log decay domain) ->
    (h [B,NT,H,DK,DV] fp16, v_new fp16|None, ht fp32|None)."""
    B, S, H, DK = k.shape
    DV = u.shape[-1]
    _check_domain(B, S, H, DK, DV, chunk_size)
    device = k.device
    prep = fla_chunk_delta_prep_kernel(B, S, H, DK, chunk_size, use_exp2, False)
    _ge, gt, gd = prep(g, gk)
    kernel = fla_chunk_delta_fwd_kernel(
        B,
        S,
        H,
        DK,
        DV,
        chunk_size,
        64,
        use_exp2,
        initial_state is not None,
        output_final_state,
        save_new_value,
    )
    h, v_new, ht = kernel(
        k.view(B, S, H * DK),
        w.view(B, S, H * DK),
        u.view(B, S, H * DV),
        gt,
        gd,
        initial_state if initial_state is not None else _dummy(device, torch.float32),
    )
    # no internal synchronize: upstream wrapper parity (bench harness owns it)
    if save_new_value:
        v_new = v_new.view(B, S, H, DV)
    return h, (v_new if save_new_value else None), (ht if output_final_state else None)


def fla_chunk_delta_bwd(
    q,
    k,
    w,
    do,
    dv,
    g,
    gk,
    h0=None,
    dht=None,
    scale=None,
    chunk_size=64,
    use_exp2=False,
):
    """bwd host API. q/k/w [B,S,H,DK] bf16, do/dv [B,S,H,DV] bf16, g/gk fp32
    -> (dh [B,NT,H,DK,DV] bf16, dh0 fp32|None, dv2 bf16). dh0 only when the
    fwd was called with an initial state (h0 given), mirroring upstream."""
    B, S, H, DK = q.shape
    DV = do.shape[-1]
    _check_domain(B, S, H, DK, DV, chunk_size)
    device = q.device
    prep = fla_chunk_delta_prep_kernel(B, S, H, DK, chunk_size, use_exp2, True)
    ge, gt, gd = prep(g, gk)
    kernel = fla_chunk_delta_bwd_kernel(
        B,
        S,
        H,
        DK,
        DV,
        chunk_size,
        64,
        scale,
        use_exp2,
        dht is not None,
        h0 is not None,
    )
    dh, dh0, dv2 = kernel(
        q.view(B, S, H * DK),
        k.view(B, S, H * DK),
        w.view(B, S, H * DK),
        do.view(B, S, H * DV),
        dv.view(B, S, H * DV),
        gt,
        ge,
        gd,
        dht if dht is not None else _dummy(device, torch.float32),
    )
    dv2 = dv2.view(B, S, H, DV)
    return dh, (dh0 if h0 is not None else None), dv2


__all__ = [
    "fla_chunk_delta_fwd",
    "fla_chunk_delta_bwd",
    "fla_chunk_delta_fwd_kernel",
    "fla_chunk_delta_bwd_kernel",
    "fla_chunk_delta_prep_kernel",
    "PASS_CONFIGS",
]
