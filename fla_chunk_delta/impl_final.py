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
The bwd qdo scale stays in the main kernel's fp32 f-phase mul: folding it
into the bf16 q~ operand shifts the rounding position and the bwd
recurrence's feedback loop (dH -> dv2 -> wdv2 -> dH) amplifies the drift
(measured l0_bwd_opts dh0 ratio 0.9601 -- FAIL; reverted).

Developer mode, threads=2, fixed-core task loop + explicit named relay
workspaces (chunk_o structure) + a threads=1 Developer PREP kernel that
materializes the per-chunk gate/decay tiles once per launch. bshd inputs
are consumed through zero-copy 3D views [B, S, H*D] (metadata only).

Stage-3 structure (msprof-driven evolution; see perf_tuning/):
* iter2: the prep kernel removed the in-kernel scalar extraction loops
  (79-84% of the AIV scalar pipe, ~33 ns/iter of auto-sync flag pairs).
* iter4: 2-chain statement-order batching overlapped the C/V ping-pong
  (-25% wall) but the Select-redirect tail waste and per-batch statement
  issue cost bounded further gains (fwd 3-chain measured +3.6%: rejected).
* iter7 (this): single-chain task loop (simplest issue stream, no redirect
  waste, task-loop tail guard) + BV=128 when DV % 128 == 0 (halves the
  per-core chunk count -- the per-chunk statement-issue and round-trip
  costs are BV-invariant, so the whole serial overhead halves; row_expand
  256B-row tiles split into 64-column halves) + relay MTE3 ordering (the
  Cube-unblocking ws1 relay write goes FIRST in the V prologue MTE3 queue,
  ahead of the HBM Hout/Dh snapshot writes nobody waits on) + bwd q~ chain
  compaction (one Q read, one full-width ws3 relay -- ws3b dropped -- and
  the qdo/wdv2 readbacks share one UB buffer, sequenced by the f-phase
  arithmetic) + the scale folded into GE.

Ascend toolchain adaptations (verified against this repo's sources and the
family precedents; see debug_log for the empirical record):

1. **Gate/decay tiles are precomputed by a threads=1 prep kernel**
   (GT/GD and the bwd GE). Each tile is computed in the DIRECT difference
   form (no exp(a)*exp(b) factorization: the factored form overflows fp32
   for |g| > ~88 where the direct form is exact). The main kernels read a
   contiguous 256B/512B span per (chain, chunk) and gate via row_expand --
   zero scalar statements in the hot loop; gate/decay values are
   bit-comparable to the direct in-kernel form (same fp32 sub/mul/exp).
2. vid-reduction (threads=2) halves dim 0 of every UB buffer and only
   rewrites an allow-listed op set. T.tile.gather, T.tile.broadcast and
   arith_progression are NOT in the allow list (fla_wy_fast #4) -- the
   transpose lives in the threads=1 prep kernel. The vid pass applies the
   GM region offset at region dim (region_rank - ub_rank): the GT/GE/GD
   reads use 1D [BS]/[DK] UB targets so the offset lands on the
   token/channel Range dim (the W3/Hout pattern), never on a head index.
   Row gating uses row_expand_{mul,sub} (brcb + mul/sub_mask); fp32 tiles
   need 256B rows, so [*, 128] fp32 domains gate through two 64-column
   region halves (BV=128; the src1 gate vector is shared by both halves).
3. gemm_v0 operands are half precision only; every fp32<->fp16/bf16
   conversion is a UB-internal two-step copy chain (GM<->UB cast-on-copy
   codegen is broken).
4. All C<->V exchanges go through explicit named workspace_* GM tensors
   (L2-resident); AUTO_CV_SYNC inserts the cross-core flags by buffer-name
   match. NSLOT=16 ring: slot = (k_task*NT + i_t) % NSLOT separates the
   adjacent chunks' relay WAR (single chain in flight per core, so all
   other chains are temporally far).
5. The serial chunk recurrence uses T.serial + GM relay (plain serial
   loops mis-sync on-chip V->C exchanges -- fla_wy_fast #1); the
   fixed-core task loop stripes chains over CORE_EFF = min(20, NCHAIN)
   cores with a tail guard `if tid < NCHAIN` (both core sides skip
   together, so the relay flag pairs stay balanced).
6. Support domain (wrapper asserts): S % 64 == 0, chunk_size == 64,
   DK in {64, 128}, DV % 64 == 0 (BV = 128 when DV % 128 == 0, else 64),
   H <= 248 (prep row_expand/transpose staging cap). DK = 192/256 and
   non-64-multiple D are L1-extension territory (DESIGN SS5.4).
"""

import tilelang
import torch
from tilelang import language as T

# 2^g is computed as exp(g * ln2): exp2 has no Ascend backend mapping.
LN2 = 0.6931471805599453

# Ascend 910B3: 20 AI Cube cores (+ 40 AI Vector cores); threads=2 pairs
# each cube core with its 2 vector sub-blocks.
CORE_NUM = 20

# Relay slot ring per core: slot = (k_task*NT + i_t) % NSLOT separates
# adjacent chunks' relay WAR (a single chain is in flight per core, so
# cross-chain slot reuse is temporally far; 16 slots keep the adjacent
# chunk handshakes conflict-free well within the 15-set flag capacity).
NSLOT = 16

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # automatic C/V split
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,  # cross-core relay flags
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,  # intra-core pipe sync
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # UB/L1 lifetime planning
}


# ============================================================================
# prep kernel (threads=1): per-chunk gate/decay tiles GT/GE/GD (+ bwd q~)
# ============================================================================
# Computes, per (b, chunk) task, the full [H]-wide tile set in the DIRECT
# difference form (no exp factorization -> no overflow narrowing):
#     GT[h, j] = exp(s * (g_last[h] - g[s0+j, h]))     (token gate, fwd+bwd)
#     GE[h, j] = exp(s * g[s0+j, h])                    (bwd q~ gate)
#     GD[h, k] = exp(s * (gk_last[h, k] + g_last[h]))   (state decay)
# The [BS, H] g block is loaded contiguously and transposed once (fp32
# hardware transpose needs 32B-aligned dims -> both multiples of 8; BS=64
# always is, H is staged padded to HP = ceil(H/8)*8 with the dead rows
# [H, HP) never written back). row_expand_sub applies the per-row scalar
# (-s*g_last, shared by the GT and GD chains); row_expand needs 8..248
# fp32 256B rows -> HP in [8, 248], i.e. H <= 248 (wrapper-asserted).
# threads=1 keeps every primitive vid-safe.
#
# iter10 (round-2 R10): when need_qt, prep also materializes the bwd
# gated query QT[b, s0+j, h, :] = bf16(fp32(q) * exp(s*g)) per head -- the
# SAME arithmetic chain the main kernel's V prologue used (bf16 -> fp32
# cast, fp32 row gate, fp32 -> bf16 cast; the gate values are the very
# gt_t rows the GE form staged; bit-identical values), so the main kernel
# drops the whole 13-statement q~ path and its ws3/ws3b relay handshakes
# and reads QT directly on the Cube side (one [BS, DK] gm_to_l1 replaces
# two ws3 readbacks). qdo_scale is NOT folded (see the NOTE at the use
# site; the main f-phase keeps the fp32 scale mul).
@tilelang.jit(out_idx=[3, 4, 5, 6], pass_configs=PASS_CONFIGS)
def fla_chunk_delta_prep_kernel(
    B,
    S,
    H,
    DK,
    chunk_size=64,
    use_exp2=False,
    need_ge=False,
    qdo_scale=1.0,
    need_qt=False,
):
    """prep: materialize GT/GE/GD gate/decay tiles (and bwd QT) from (g, gk)."""
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
    assert 8 <= HP <= 248, "H <= 248 (prep gate/decay staging row cap)"
    # s = ln2 if use_exp2 else 1.0 (gate/decay exponent scale)
    s = LN2 if use_exp2 else 1.0

    @T.prim_func
    def main(
        G: T.Tensor((B, S, H), "float32"),  # type: ignore
        Gk: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        Q3: T.Tensor((B, S, H * DK) if need_qt else (1,), "bfloat16"),  # type: ignore
        GE: T.Tensor((B, NT, H, BS), "float32") if need_ge else T.Tensor((1,), "float32"),  # type: ignore
        GT: T.Tensor((B, NT, H, BS), "float32"),  # type: ignore
        GD: T.Tensor((B, NT, H, DK), "float32"),  # type: ignore
        QT: T.Tensor((B, S, H * DK) if need_qt else (1,), "bfloat16"),  # type: ignore
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
            # q~ staging (need_qt): per-head [BS, DK] cast/gate chain
            q16_st = T.alloc_shared([BS, DK], "bfloat16")
            q32_st = T.alloc_shared([BS, DK], "float32")
            q16g_st = T.alloc_shared([BS, DK], "bfloat16")
            gate1d = T.alloc_shared([BS], "float32")
            tmp_q = T.alloc_shared([BS, 8], "float32")  # row_expand scratch

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
                    # ---- q~ gate GE = exp(s*g)*scale (bwd only)
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
                    # ---- bwd q~ = q * exp(s*g), per head (iter10)
                    # gt_t restages the GE form; the per-head [BS, DK]
                    # cast/gate chain mirrors the main kernel's removed V
                    # prologue arithmetic EXACTLY (bit-identical values).
                    # NOTE: qdo_scale is deliberately NOT folded here --
                    # folding it into the bf16 operand shifts the rounding
                    # position and the bwd recurrence's feedback loop
                    # (dH -> dv2 -> wdv2 -> dH) amplifies the drift 4-8x:
                    # l0_bwd_opts dh0 ratio measured 0.9601 (FAIL). The
                    # scale stays in the main f-phase fp32 mul.
                    if need_qt:
                        T.tile.mul(gx2d, g2d, s)
                        T.tile.exp(gx2d, gx2d)
                        T.tile.transpose(gt_t, gx2d)
                        for h in T.serial(H):
                            T.copy(
                                Q3[bb, s0 : s0 + BS, h * DK : (h + 1) * DK],
                                q16_st[0:BS, 0:DK],
                            )
                            T.copy(q16_st[0:BS, 0:DK], q32_st[0:BS, 0:DK])
                            T.copy(gt_t[h, 0:BS], gate1d[0:BS])
                            T.tile.row_expand_mul_experiment(
                                q32_st[0:BS, 0:64],
                                q32_st[0:BS, 0:64],
                                gate1d,
                                tmp_q,
                            )
                            if NSLABD == 2:
                                T.tile.row_expand_mul_experiment(
                                    q32_st[0:BS, 64:DK],
                                    q32_st[0:BS, 64:DK],
                                    gate1d,
                                    tmp_q,
                                )
                            T.copy(q32_st[0:BS, 0:DK], q16g_st[0:BS, 0:DK])
                            T.copy(
                                q16g_st[0:BS, 0:DK],
                                QT[bb, s0 : s0 + BS, h * DK : (h + 1) * DK],
                            )

    return main


# ============================================================================
# fwd kernel: chunk_gated_delta_rule_fwd_h (H state recurrence)
# ============================================================================
# out_idx/workspace_idx use ABSOLUTE positions over the declared signature
# (K3, W3, U3, GT, GD, H0, Hout, Vnew, Ht, workspace_1..4). Disabled
# constexpr branches keep their signature slot with a [1] dummy tensor.
# MTE3 ordering: the ws1 relay write precedes the Hout snapshot write in
# the V prologue (the Cube blocks on the ws1 flag; nobody waits on Hout).
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
    # row_expand 256B-row tiles: [*, BV] fp32 domains gate through
    # NSPLIT 64-column halves (BV=64 -> 1, BV=128 -> 2); the 1D gate
    # vector is shared by all halves of a gated tile
    NSPLIT = BV // 64

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
            h_ub = T.alloc_shared([DK, BV], "float32")  # state H, fp32 resident
            h16_ub = T.alloc_shared([DK, BV], "float16")  # H cast staging
            bv_ub = T.alloc_shared([BS, BV], "float32")  # b_v relay readback
            kh_ub = T.alloc_shared([DK, BV], "float32")  # kH relay readback
            u16_ub = T.alloc_shared([BS, BV], "float16")
            u_ub = T.alloc_shared([BS, BV], "float32")
            vn_ub = T.alloc_shared([BS, BV], "float32")  # ungated v_new
            vn_g_ub = T.alloc_shared([BS, BV], "float32")  # gated v_new
            vn16_ub = T.alloc_shared([BS, BV], "float16")  # ws3 staging (gated)
            vn16v_ub = T.alloc_shared([BS, BV], "float16")  # Vnew staging
            gate_ub = T.alloc_shared([BS], "float32")  # token gate (from GT)
            dsrc_ub = T.alloc_shared([DK], "float32")  # state decay (from GD)
            tmp_gate = T.alloc_shared([BS, 8], "float32")  # row_expand scratch
            tmp_decay = T.alloc_shared([DK, 8], "float32")
            # ---- C-side buffers (L1 gemm operands / L0C accumulators)
            w_l1 = T.alloc_shared([BS, DK], "float16")
            k_l1 = T.alloc_shared([BS, DK], "float16")
            h16_l1 = T.alloc_shared([DK, BV], "float16")
            v16_l1 = T.alloc_shared([BS, BV], "float16")
            bv_frag = T.alloc_fragment([BS, BV], "float32")
            kh_frag = T.alloc_fragment([DK, BV], "float32")

            for k_task in T.serial(NTASK):
                tid = cid + k_task * CORE_EFF
                if tid < NCHAIN:
                    i_v = tid % NV
                    i_nh = tid // NV
                    bb = i_nh // H
                    bh = i_nh % H
                    v0 = i_v * BV
                    if use_initial_state:
                        T.copy(H0[bb, bh, 0:DK, v0 : v0 + BV], h_ub)
                    else:
                        T.tile.fill(h_ub, 0.0)
                    for i_t in T.serial(NT):
                        s0 = i_t * BS
                        # relay ring slot (adjacent chunks never share a slot)
                        sl = (k_task * NT + i_t) % NSLOT
                        # (a) snapshot h[t] (ungated) + H16 relay to Cube
                        # (ws1 relay FIRST in the MTE3 queue -- the Cube's
                        # gemm1 unblocks on its flag; the HBM Hout write is
                        # fire-and-forget)
                        T.copy(h_ub, h16_ub)
                        T.copy(h16_ub, workspace_1[cid, sl, 0:DK, 0:BV])
                        T.copy(h16_ub, Hout[bb, i_t, bh, 0:DK, v0 : v0 + BV])
                        T.copy(
                            U3[bb, s0 : s0 + BS, bh * DV + v0 : bh * DV + v0 + BV],
                            u16_ub,
                        )
                        T.copy(GT[bb, i_t, bh, 0:BS], gate_ub)
                        T.copy(GD[bb, i_t, bh, 0:DK], dsrc_ub)
                        # (b) Cube: b_v = w[t] @ H16
                        T.copy(W3[bb, s0 : s0 + BS, bh * DK : (bh + 1) * DK], w_l1)
                        T.copy(workspace_1[cid, sl, 0:DK, 0:BV], h16_l1)
                        T.gemm_v0(w_l1, h16_l1, bv_frag, init=True)
                        T.copy(bv_frag, workspace_2[cid, sl, 0:BS, 0:BV])
                        # (c) Vector: v_new = u - b_v; save UNGATED v_new
                        T.copy(u16_ub, u_ub)
                        T.copy(workspace_2[cid, sl, 0:BS, 0:BV], bv_ub)
                        T.tile.sub(vn_ub, u_ub, bv_ub)
                        # (d) token gate exp(g_last - g) on v_new; the ws3
                        # relay write precedes the Vnew write (the Cube's
                        # gemm2 unblocks on ws3; nobody waits on Vnew)
                        if NSPLIT == 1:
                            T.tile.row_expand_mul_experiment(vn_g_ub, vn_ub, gate_ub, tmp_gate)
                        else:
                            T.tile.row_expand_mul_experiment(vn_g_ub[0:BS, 0:64], vn_ub[0:BS, 0:64], gate_ub, tmp_gate)
                            T.tile.row_expand_mul_experiment(vn_g_ub[0:BS, 64:BV], vn_ub[0:BS, 64:BV], gate_ub, tmp_gate)
                        T.copy(vn_g_ub, vn16_ub)
                        T.copy(vn16_ub, workspace_3[cid, sl, 0:BS, 0:BV])
                        if save_new_value:
                            T.copy(vn_ub, vn16v_ub)
                            T.copy(
                                vn16v_ub,
                                Vnew[bb, s0 : s0 + BS, bh * DV + v0 : bh * DV + v0 + BV],
                            )
                        # (e) Vector->Cube: v_new16; Cube: kH = k^T @ v_new
                        T.copy(K3[bb, s0 : s0 + BS, bh * DK : (bh + 1) * DK], k_l1)
                        T.copy(workspace_3[cid, sl, 0:BS, 0:BV], v16_l1)
                        T.gemm_v0(k_l1, v16_l1, kh_frag, transpose_A=True, init=True)
                        T.copy(kh_frag, workspace_4[cid, sl, 0:DK, 0:BV])
                        # (f) Vector: decay + state update H = D*H + kH
                        T.copy(workspace_4[cid, sl, 0:DK, 0:BV], kh_ub)
                        if NSPLIT == 1:
                            T.tile.row_expand_mul_experiment(h_ub, h_ub, dsrc_ub, tmp_decay)
                        else:
                            T.tile.row_expand_mul_experiment(h_ub[0:DK, 0:64], h_ub[0:DK, 0:64], dsrc_ub, tmp_decay)
                            T.tile.row_expand_mul_experiment(h_ub[0:DK, 64:BV], h_ub[0:DK, 64:BV], dsrc_ub, tmp_decay)
                        T.tile.add(h_ub, h_ub, kh_ub)
                    if output_final_state:
                        T.copy(h_ub, Ht[bb, bh, 0:DK, v0 : v0 + BV])

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
# qdo/wdv2 relay workspace set (ws3/ws3b dropped in iter10 with the q~
# externalization; ws1 dh16 / ws2 dvp / ws4 dv2 / ws5 qdo / ws6 wdv2).
@tilelang.jit(out_idx=[8, 9, 10], workspace_idx=[11, 12, 13, 14, 15], pass_configs=PASS_CONFIGS)
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
    """bwd: dh/dv2(/dh0) from (qt~, k, w, do, dv, GT, GD[/dht, h0])."""
    if scale is None:
        scale = DK**-0.5
    BS = chunk_size
    NT = S // BS
    NV = DV // BV
    NCHAIN = B * H * NV
    CORE_EFF = min(CORE_NUM, NCHAIN)
    # iter8 (round-2 R9): universal batch decomposition.
    # NFULL = NCHAIN // (2*CORE_EFF) 2-chain batches are provably fully
    # in-bounds (max tid 2*NFULL*CORE_EFF - 1 <= NCHAIN - 1), so their
    # chain indices stay PLAIN arithmetic -- no T.if_then_else Select: the
    # select's if/else chains were inlined into EVERY buffer index of
    # EVERY statement (~100+ scalar branches per iteration on both core
    # sides; the round-1 final source shows 2-8 condval blocks before
    # each tl-op). The remainder R = NCHAIN mod 2*CORE_EFF chains run as
    # guarded single-chain tail batches (one per remaining task): cores
    # holding a valid chain process it at half statement cost; the other
    # cores skip the whole block (both core sides skip together, so the
    # relay flag pairs stay balanced) instead of the old Select-redirect
    # duplicate work.
    NB2 = 2 * CORE_EFF
    NFULL = NCHAIN // NB2
    R = NCHAIN - NFULL * NB2  # 0 .. 2*CORE_EFF-1
    # tail task ids are the contiguous range [2*NFULL, 2*NFULL + NTAIL)
    NTAIL = 1 if 0 < R <= CORE_EFF else (2 if R > CORE_EFF else 0)
    # DK domain (the q~ column-half gating moved to the prep kernel in
    # iter10; the main kernel reads QT as one [BS, DK] tile)
    NQK = DK // 64
    assert NQK in (1, 2), "DK in {64, 128} only (L0 domain)"

    @T.prim_func
    def main(
        # pre-gated q~ = q*exp(s*g) from the prep kernel (iter10):
        # replaces the raw Q3 + in-kernel q~ gating path + ws3/ws3b relays
        # (scale stays in the main f-phase fp32 mul)
        QT: T.Tensor((B, S, H * DK), "bfloat16"),  # type: ignore
        K3: T.Tensor((B, S, H * DK), "bfloat16"),  # type: ignore
        W3: T.Tensor((B, S, H * DK), "bfloat16"),  # type: ignore
        Do3: T.Tensor((B, S, H * DV), "bfloat16"),  # type: ignore
        Dv3: T.Tensor((B, S, H * DV), "bfloat16"),  # type: ignore
        # gate/decay tiles from the prep kernel (see the fwd note for the
        # 1D-target vid rule)
        GT: T.Tensor((B, NT, H, BS), "float32"),  # type: ignore
        GD: T.Tensor((B, NT, H, DK), "float32"),  # type: ignore
        Dht: T.Tensor((B, H, DK, DV) if use_final_state_gradient else (1,), "float32"),  # type: ignore
        Dh: T.Tensor((B, NT, H, DK, DV), "bfloat16"),  # type: ignore
        Dh0: T.Tensor((B, H, DK, DV) if use_initial_state else (1,), "float32"),  # type: ignore
        # 3D view [B, S, H*DV] (see the fwd Vnew note; wrapper views it back)
        Dv2: T.Tensor((B, S, H * DV), "bfloat16"),  # type: ignore
        workspace_1: T.Tensor((CORE_EFF, NSLOT, DK, BV), "bfloat16"),  # type: ignore
        workspace_2: T.Tensor((CORE_EFF, NSLOT, BS, BV), "float32"),  # type: ignore
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
            gate_ub_a = T.alloc_shared([BS], "float32")  # token gate (from GT)
            gate_ub_b = T.alloc_shared([BS], "float32")
            dsrc_ub_a = T.alloc_shared([DK], "float32")  # decay (from GD)
            dsrc_ub_b = T.alloc_shared([DK], "float32")
            # qdo/wdv2 readbacks share ONE buffer: the f-phase consumes qdo
            # (decay; add) BEFORE wdv2 is read back (the scale rides in the
            # prep-materialized QT since iter10)
            qdw_ub = T.alloc_shared([DK, BV], "float32")
            tmp_gate = T.alloc_shared([BS, 8], "float32")
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
            # one time-shared [DK, BV] L0C accumulator per chain for
            # qdo/wdv2 (the ws5 fixpipe drains before that chain's wdv2
            # gemm issues): 16+32+16+32 = 96KB L0C, vs 160KB if all four
            # fragments were live (over the 128KB budget)
            qw_frag_a = T.alloc_fragment([DK, BV], "float32")
            qw_frag_b = T.alloc_fragment([DK, BV], "float32")

            if NFULL > 0:
                for kb in T.serial(NFULL):
                    # full batches: plain in-bounds tids (see the NFULL note)
                    tida = cid + (2 * kb) * CORE_EFF
                    tidb = cid + (2 * kb + 1) * CORE_EFF
                    iva = tida % NV
                    inha = tida // NV
                    bba = inha // H
                    bha = inha % H
                    v0a = iva * BV
                    ivb = tidb % NV
                    inhb = tidb // NV
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
                        # prefetches (q~ arrived pre-gated from prep) =====
                        T.copy(dh_ub_a, dh16_ub_a)
                        T.copy(dh16_ub_a, workspace_1[cid, sla, 0:DK, 0:BV])
                        T.copy(dh16_ub_a, Dh[bba, i_t, bha, 0:DK, v0a : v0a + BV])
                        T.copy(
                            Dv3[bba, s0a : s0a + BS, bha * DV + v0a : bha * DV + v0a + BV],
                            dv16_ub_a,
                        )
                        T.copy(GT[bba, i_t, bha, 0:BS], gate_ub_a)
                        T.copy(GD[bba, i_t, bha, 0:DK], dsrc_ub_a)
                        # ===== V prologue chain B: same =====
                        T.copy(dh_ub_b, dh16_ub_b)
                        T.copy(dh16_ub_b, workspace_1[cid, slb, 0:DK, 0:BV])
                        T.copy(dh16_ub_b, Dh[bbb, i_t, bhb, 0:DK, v0b : v0b + BV])
                        T.copy(
                            Dv3[bbb, s0b : s0b + BS, bhb * DV + v0b : bhb * DV + v0b + BV],
                            dv16_ub_b,
                        )
                        T.copy(GT[bbb, i_t, bhb, 0:BS], gate_ub_b)
                        T.copy(GD[bbb, i_t, bhb, 0:DK], dsrc_ub_b)
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
                        # ===== C phase e1: qdo gemm for BOTH chains (QT read
                        # directly from GM -- no ws3 relay since iter10) =====
                        T.copy(
                            Do3[bba, s0a : s0a + BS, bha * DV + v0a : bha * DV + v0a + BV],
                            do_l1_a,
                        )
                        T.copy(QT[bba, s0a : s0a + BS, bha * DK : (bha + 1) * DK], qt16_l1_a)
                        T.gemm_v0(qt16_l1_a, do_l1_a, qw_frag_a, transpose_A=True, init=True)
                        T.copy(qw_frag_a, workspace_5[cid, sla, 0:DK, 0:BV])
                        T.copy(
                            Do3[bbb, s0b : s0b + BS, bhb * DV + v0b : bhb * DV + v0b + BV],
                            do_l1_b,
                        )
                        T.copy(QT[bbb, s0b : s0b + BS, bhb * DK : (bhb + 1) * DK], qt16_l1_b)
                        T.gemm_v0(qt16_l1_b, do_l1_b, qw_frag_b, transpose_A=True, init=True)
                        T.copy(qw_frag_b, workspace_5[cid, slb, 0:DK, 0:BV])
                        # ===== C phase e2: wdv2 gemm for BOTH chains =====
                        T.copy(W3[bba, s0a : s0a + BS, bha * DK : (bha + 1) * DK], w_l1_a)
                        T.copy(workspace_4[cid, sla, 0:BS, 0:BV], dv2_16_l1_a)
                        T.gemm_v0(w_l1_a, dv2_16_l1_a, qw_frag_a, transpose_A=True, init=True)
                        T.copy(qw_frag_a, workspace_6[cid, sla, 0:DK, 0:BV])
                        T.copy(W3[bbb, s0b : s0b + BS, bhb * DK : (bhb + 1) * DK], w_l1_b)
                        T.copy(workspace_4[cid, slb, 0:BS, 0:BV], dv2_16_l1_b)
                        T.gemm_v0(w_l1_b, dv2_16_l1_b, qw_frag_b, transpose_A=True, init=True)
                        T.copy(qw_frag_b, workspace_6[cid, slb, 0:DK, 0:BV])
                        # ===== V phase f (A then B): dH = D*dH + qdo*scale
                        # - wdv2 (scale in the fp32 mul -- see the prep NOTE) =====
                        T.copy(workspace_5[cid, sla, 0:DK, 0:BV], qdw_ub)
                        T.tile.mul(qdw_ub, qdw_ub, scale)
                        T.tile.row_expand_mul_experiment(dh_ub_a, dh_ub_a, dsrc_ub_a, tmp_decay)
                        T.tile.add(dh_ub_a, dh_ub_a, qdw_ub)
                        T.copy(workspace_6[cid, sla, 0:DK, 0:BV], qdw_ub)
                        T.tile.sub(dh_ub_a, dh_ub_a, qdw_ub)
                        T.copy(workspace_5[cid, slb, 0:DK, 0:BV], qdw_ub)
                        T.tile.mul(qdw_ub, qdw_ub, scale)
                        T.tile.row_expand_mul_experiment(dh_ub_b, dh_ub_b, dsrc_ub_b, tmp_decay)
                        T.tile.add(dh_ub_b, dh_ub_b, qdw_ub)
                        T.copy(workspace_6[cid, slb, 0:DK, 0:BV], qdw_ub)
                        T.tile.sub(dh_ub_b, dh_ub_b, qdw_ub)
                    if use_initial_state:
                        T.copy(dh_ub_a, Dh0[bba, bha, 0:DK, v0a : v0a + BV])
                        T.copy(dh_ub_b, Dh0[bbb, bhb, 0:DK, v0b : v0b + BV])
            # ===== remainder tail: guarded single-chain batches (iter8) =====
            for tt in T.serial(NTAIL):
                tidt = cid + (2 * NFULL + tt) * CORE_EFF
                if tidt < NCHAIN:
                    ivt = tidt % NV
                    inht = tidt // NV
                    bbt = inht // H
                    bht = inht % H
                    v0t = ivt * BV
                    if use_final_state_gradient:
                        T.copy(Dht[bbt, bht, 0:DK, v0t : v0t + BV], dh_ub_a)
                    else:
                        T.tile.fill(dh_ub_a, 0.0)
                    for it_f in T.serial(NT):
                        i_t = NT - 1 - it_f
                        s0t = i_t * BS
                        slt = (2 * it_f) % NSLOT
                        # V prologue (single-chain form; q~ pre-gated by prep)
                        T.copy(dh_ub_a, dh16_ub_a)
                        T.copy(dh16_ub_a, workspace_1[cid, slt, 0:DK, 0:BV])
                        T.copy(dh16_ub_a, Dh[bbt, i_t, bht, 0:DK, v0t : v0t + BV])
                        T.copy(
                            Dv3[bbt, s0t : s0t + BS, bht * DV + v0t : bht * DV + v0t + BV],
                            dv16_ub_a,
                        )
                        T.copy(GT[bbt, i_t, bht, 0:BS], gate_ub_a)
                        T.copy(GD[bbt, i_t, bht, 0:DK], dsrc_ub_a)
                        # C phase b: dv_part gemm
                        T.copy(K3[bbt, s0t : s0t + BS, bht * DK : (bht + 1) * DK], k_l1_a)
                        T.copy(workspace_1[cid, slt, 0:DK, 0:BV], dh16_l1_a)
                        T.gemm_v0(k_l1_a, dh16_l1_a, dvp_frag_a, init=True)
                        T.copy(dvp_frag_a, workspace_2[cid, slt, 0:BS, 0:BV])
                        # V phase c: gate dv_part; dv2 = dv + dv_part
                        T.copy(workspace_2[cid, slt, 0:BS, 0:BV], dvp_ub)
                        T.tile.row_expand_mul_experiment(dvp_ub, dvp_ub, gate_ub_a, tmp_gate)
                        T.copy(dv16_ub_a, dv_ub)
                        T.tile.add(dv2_ub, dv_ub, dvp_ub)
                        T.copy(dv2_ub, dv2_16_ub_a)
                        T.copy(dv2_16_ub_a, workspace_4[cid, slt, 0:BS, 0:BV])
                        T.copy(
                            dv2_16_ub_a,
                            Dv2[bbt, s0t : s0t + BS, bht * DV + v0t : bht * DV + v0t + BV],
                        )
                        # C phase e: qdo then wdv2 (time-shared accumulator)
                        T.copy(
                            Do3[bbt, s0t : s0t + BS, bht * DV + v0t : bht * DV + v0t + BV],
                            do_l1_a,
                        )
                        T.copy(QT[bbt, s0t : s0t + BS, bht * DK : (bht + 1) * DK], qt16_l1_a)
                        T.gemm_v0(qt16_l1_a, do_l1_a, qw_frag_a, transpose_A=True, init=True)
                        T.copy(qw_frag_a, workspace_5[cid, slt, 0:DK, 0:BV])
                        T.copy(W3[bbt, s0t : s0t + BS, bht * DK : (bht + 1) * DK], w_l1_a)
                        T.copy(workspace_4[cid, slt, 0:BS, 0:BV], dv2_16_l1_a)
                        T.gemm_v0(w_l1_a, dv2_16_l1_a, qw_frag_a, transpose_A=True, init=True)
                        T.copy(qw_frag_a, workspace_6[cid, slt, 0:DK, 0:BV])
                        # V phase f: dH = D*dH + qdo*scale - wdv2 (fp32 scale)
                        T.copy(workspace_5[cid, slt, 0:DK, 0:BV], qdw_ub)
                        T.tile.mul(qdw_ub, qdw_ub, scale)
                        T.tile.row_expand_mul_experiment(dh_ub_a, dh_ub_a, dsrc_ub_a, tmp_decay)
                        T.tile.add(dh_ub_a, dh_ub_a, qdw_ub)
                        T.copy(workspace_6[cid, slt, 0:DK, 0:BV], qdw_ub)
                        T.tile.sub(dh_ub_a, dh_ub_a, qdw_ub)
                    if use_initial_state:
                        T.copy(dh_ub_a, Dh0[bbt, bht, 0:DK, v0t : v0t + BV])

    return main


# ============================================================================
# Host wrappers (upstream naming: chunk_gated_delta_rule_{fwd_h, bwd_dhu} ->
# fla_chunk_delta_{fwd, bwd}; signature aligned with the proto schema).
# Only zero-copy metadata views + kernel calls here (the JIT adapter owns
# output/workspace allocation). Each wrapper launches the threads=1 prep
# kernel (GT/GE/GD tiles) followed by the recurrence kernel; both launches
# are inside the do_bench timed region (bench accounting stays honest).
# ============================================================================
def _pick_bv(DV):
    """fwd: BV = 128 when the V dim allows it (halves the per-core chunk
    count; the per-chunk statement-issue and round-trip costs are
    BV-invariant -- measured -1% at bench, kept for the L1/L2 headroom).
    bwd pins BV=64: its in-place [BS, BV] region row_expand (dst==src0,
    512B physical rows) corrupts under the mul_mask emitter's 256B-row
    windowing (l0_bwd_repr measured max_abs=nan, ratio exactly 0.25)."""
    return 128 if DV % 128 == 0 else 64


def _pick_bv_bwd(DV):
    return 64


def _check_domain(B, S, H, DK, DV, chunk_size):
    assert chunk_size == 64, "chunk_size fixed at 64 (block_S semantic)"
    assert S % chunk_size == 0, "S must be divisible by chunk_size"
    assert DK in (64, 128), "DK in {64, 128} (L0 domain; 192/256 = L1 extension)"
    assert DV % 64 == 0, "DV must be a multiple of 64 (BV=64/128)"
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
    prep = fla_chunk_delta_prep_kernel(B, S, H, DK, chunk_size, use_exp2, False, 1.0)
    # (GE dummy, GT, GD, QT dummy): fwd never materializes q~
    _ge, gt, gd, _qt = prep(g, gk, _dummy(device, torch.bfloat16))
    kernel = fla_chunk_delta_fwd_kernel(
        B,
        S,
        H,
        DK,
        DV,
        chunk_size,
        _pick_bv(DV),
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
    if scale is None:
        scale = DK**-0.5
    # qdo_scale=1.0: the scale is NOT folded into QT (see the prep NOTE --
    # the fold broke l0_bwd_opts dh0 via recurrence feedback amplification);
    # the main f-phase keeps the fp32 scale mul
    prep = fla_chunk_delta_prep_kernel(B, S, H, DK, chunk_size, use_exp2, False, 1.0, True)
    # (GE dummy, GT, GD, QT): q~ is pre-gated (unscaled) by the prep kernel
    _ge, gt, gd, qt = prep(g, gk, q.view(B, S, H * DK))
    kernel = fla_chunk_delta_bwd_kernel(
        B,
        S,
        H,
        DK,
        DV,
        chunk_size,
        _pick_bv_bwd(DV),
        scale,
        use_exp2,
        dht is not None,
        h0 is not None,
    )
    dh, dh0, dv2 = kernel(
        qt,
        k.view(B, S, H * DK),
        w.view(B, S, H * DK),
        do.view(B, S, H * DV),
        dv.view(B, S, H * DV),
        gt,
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
