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

Developer mode, threads=2, fixed-core task loop + explicit named relay
workspaces (chunk_o structure). bshd inputs are consumed through zero-copy
3D views [B, S, H*D] (metadata only, no host materialization).

Ascend toolchain adaptations (verified against this repo's sources and the
family precedents; see debug_log for the empirical record):

1. vid-reduction (threads=2) halves dim 0 of every UB buffer and only
   rewrites an allow-listed op set (ascend_vid_reduction.cc IsTileOp:
   add/sub/mul/div/exp/fill/cast/...). T.tile.gather, T.tile.broadcast and
   arith_progression are NOT in the allow list and corrupt silently under
   vid splitting (fla_wy_fast #4). This kernel therefore uses only
   vid-safe primitives:
   * the [B,S,H] g read uses a flat contiguous span copy + a T.Parallel
     UB-local scalar extraction loop (the loop var lands in the vid-reduced
     first dim, so the pass halves the loop extent; flat index i*H + bh is
     correct within each vid's half-span);
   * row gating uses T.tile.row_expand_mul_experiment (AscendC brcb +
     mul_mask), whose shapes flow entirely through access_ptr extents and
     therefore stay consistent after halving; fp32 operands need 256B rows
     ([*, 64] fp32), which holds for every gated tile here (DK=128 q~ is
     gated in two 64-column chunks);
   * g_last is staged once per (chain, chunk) into a [1] UB buffer via a
     scalar store (codegen_ascend.cc VisitStmt_(BufferStoreNode) /
     VisitExpr_(BufferLoadNode) flatten ALL indices through OffsetOf);
     the tile-op scalar path would only flatten indices_1[0] and insert a
     PIPE_ALL barrier, so the [BS] gate difference is computed inside the
     T.Parallel extraction loop as UB-local scalar arithmetic and only the
     [DK] decay add consumes the staged 1D scalar (indices_1[0] == 0 is
     exact for a [1] buffer).
2. gemm_v0 operands are half precision only (fp32 A/B produces garbage on
   this toolchain); every fp32<->fp16/bf16 conversion is a UB-internal
   two-step copy chain (GM<->UB cast-on-copy codegen is broken).
3. All C<->V exchanges go through explicit named workspace_* GM tensors
   (L2-resident, [CORE_EFF, NSLOT, ...]); AUTO_CV_SYNC inserts the
   cross-core flags by buffer-name match (chunk_o pattern). NSLOT=1 is a
   strict lockstep: correct by construction, Stage 3 owns the ring.
4. The serial chunk recurrence uses T.serial + GM relay (plain serial loops
   mis-sync on-chip V->C exchanges -- fla_wy_fast #1); the fixed-core task
   loop stripes chains over CORE_EFF = min(20, NCHAIN) cores with a tail
   guard `if tid < NCHAIN` (both core sides skip together, so the relay
   flag pairs stay balanced).
5. Support domain (wrapper asserts): S % 64 == 0, chunk_size == 64,
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

# Relay slot ring per core. NSLOT=1 is the Stage-2 lockstep (one hand-shake
# pair per relay per chunk, no lead/lag races).
NSLOT = 1

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # automatic C/V split
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,  # cross-core relay flags
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,  # intra-core pipe sync
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # UB/L1 lifetime planning
}


# ============================================================================
# fwd kernel: chunk_gated_delta_rule_fwd_h (H state recurrence)
# ============================================================================
# out_idx/workspace_idx use ABSOLUTE positions over the declared signature
# (K3, W3, U3, GFlat, Gk3, H0, Hout, Vnew, Ht, workspace_1..4). Disabled
# constexpr branches keep their signature slot with a [1] dummy tensor so
# the indices stay stable across specializations.
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
    """fwd: h/v_new(/ht) from (k, w, u, g, gk[/h0]) per (b, h, i_v) chain."""
    BS = chunk_size
    NT = S // BS
    NV = DV // BV
    NCHAIN = B * H * NV
    CORE_EFF = min(CORE_NUM, NCHAIN)
    NTASK = (NCHAIN + CORE_EFF - 1) // CORE_EFF
    # exp(gate_scale * (g - g_last)) == base^(g_last - g)
    gate_scale = -LN2 if use_exp2 else -1.0
    # exp(decay_scale * (gk_last + g_last)) == base^(gk_last + g_last)
    decay_scale = LN2 if use_exp2 else 1.0

    @T.prim_func
    def main(
        K3: T.Tensor((B, S, H * DK), "float16"),  # type: ignore
        W3: T.Tensor((B, S, H * DK), "float16"),  # type: ignore
        U3: T.Tensor((B, S, H * DV), "float16"),  # type: ignore
        GFlat: T.Tensor((B, S * H), "float32"),  # type: ignore
        Gk3: T.Tensor((B, S, H * DK), "float32"),  # type: ignore
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
            vn_ub = T.alloc_shared([BS, BV], "float32")
            vn16_ub = T.alloc_shared([BS, BV], "float16")
            g_span = T.alloc_shared([BS * H], "float32")  # flat g span
            gdiff_ub = T.alloc_shared([BS, 1], "float32")  # [N,1] 2D: the exp
            # runs through the count-based 2D emitter; plain 1D tile ops ride
            # a stale vector mask set by a different-sized chain (measured:
            # the [BS] chain's 32-lane mask silently halves the [DK] chain's
            # 64-element ops), so all 1D-domain arithmetic is folded into
            # the T.serial scalar loops below instead
            gate_ub = T.alloc_shared([BS, 1], "float32")
            glast_ub = T.alloc_ub([1], "float32")  # g_last staging
            gkl_ub = T.alloc_shared([DK], "float32")  # gk last row (scalar-read)
            dsrc_ub = T.alloc_shared([DK, 1], "float32")  # exp decay source
            tmp_gate = T.alloc_shared([BS, 8], "float32")  # row_expand scratch
            tmp_decay = T.alloc_shared([DK, 8], "float32")  # row_expand scratch
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
                        n_t = s0 + BS - 1
                        # (a) snapshot h[t] (ungated) + H16 relay to Cube
                        T.copy(h_ub, h16_ub)
                        T.copy(h16_ub, Hout[bb, i_t, bh, 0:DK, v0 : v0 + BV])
                        T.copy(h16_ub, workspace_1[cid, 0, 0:DK, 0:BV])
                        # (b) Cube: b_v = w[t] @ H16
                        T.copy(W3[bb, s0 : s0 + BS, bh * DK : (bh + 1) * DK], w_l1)
                        T.copy(workspace_1[cid, 0, 0:DK, 0:BV], h16_l1)
                        T.gemm_v0(w_l1, h16_l1, bv_frag, init=True)
                        T.copy(bv_frag, workspace_2[cid, 0, 0:BS, 0:BV])
                        # (c) Vector: v_new = u - b_v; save UNGATED v_new
                        T.copy(
                            U3[bb, s0 : s0 + BS, bh * DV + v0 : bh * DV + v0 + BV],
                            u16_ub,
                        )
                        T.copy(u16_ub, u_ub)
                        T.copy(workspace_2[cid, 0, 0:BS, 0:BV], bv_ub)
                        T.tile.sub(vn_ub, u_ub, bv_ub)
                        if save_new_value:
                            T.copy(vn_ub, vn16_ub)
                            T.copy(
                                vn16_ub,
                                Vnew[bb, s0 : s0 + BS, bh * DV + v0 : bh * DV + v0 + BV],
                            )
                        # (d) Vector: token gate exp(g_last - g) on v_new
                        # (extraction loop is T.serial: the parallel-loop
                        # vectorizer drops the i*H stride of the span load
                        # and emits a misaligned VEC read; the loop var
                        # still lands in the vid-reduced first dim so the
                        # pass halves the extent)
                        T.copy(GFlat[bb, s0 * H : (s0 + BS) * H], g_span)
                        glast_ub[0] = GFlat[bb, n_t * H + bh]
                        # the fill is an exact-contract op: it resets the
                        # vector mask/count for the [BS, 1]-sized exp below
                        # (the exp is mask-inert and would otherwise ride a
                        # stale mask set by a different-sized chain)
                        T.tile.fill(gdiff_ub, 0.0)
                        for i in T.serial(BS):
                            gdiff_ub[i, 0] = (g_span[i * H + bh] - glast_ub[0]) * gate_scale
                        T.tile.exp(gate_ub, gdiff_ub)
                        T.tile.row_expand_mul_experiment(vn_ub, vn_ub, gate_ub, tmp_gate)
                        # (e) Vector->Cube: v_new16; Cube: kH = k^T @ v_new
                        T.copy(vn_ub, vn16_ub)
                        T.copy(vn16_ub, workspace_3[cid, 0, 0:BS, 0:BV])
                        T.copy(K3[bb, s0 : s0 + BS, bh * DK : (bh + 1) * DK], k_l1)
                        T.copy(workspace_3[cid, 0, 0:BS, 0:BV], v16_l1)
                        T.gemm_v0(k_l1, v16_l1, kh_frag, transpose_A=True, init=True)
                        T.copy(kh_frag, workspace_4[cid, 0, 0:DK, 0:BV])
                        # (f) Vector: decay + state update H = D*H + kH
                        # (fill + scalar loop + VEC exp: the exp is
                        # mask-inert, so a same-sized exact-contract fill
                        # right before it pins the mask; scalar T.exp has
                        # no lowering on this target)
                        T.copy(Gk3[bb, n_t, bh * DK : (bh + 1) * DK], gkl_ub)
                        T.tile.fill(dsrc_ub, 0.0)
                        for k in T.serial(DK):
                            dsrc_ub[k, 0] = (gkl_ub[k] + glast_ub[0]) * decay_scale
                        T.tile.exp(dsrc_ub, dsrc_ub)
                        T.copy(workspace_4[cid, 0, 0:DK, 0:BV], kh_ub)
                        T.tile.row_expand_mul_experiment(h_ub, h_ub, dsrc_ub, tmp_decay)
                        T.tile.add(h_ub, h_ub, kh_ub)
                    if output_final_state:
                        T.copy(h_ub, Ht[bb, bh, 0:DK, v0 : v0 + BV])

    return main


# ============================================================================
# bwd kernel: chunk_gated_delta_rule_bwd_dhu (dH state-gradient recurrence)
# ============================================================================
@tilelang.jit(out_idx=[8, 9, 10], workspace_idx=[11, 12, 13, 14, 15, 16, 17], pass_configs=PASS_CONFIGS)
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
    """bwd: dh/dv2(/dh0) from (q, k, w, do, dv, g, gk[/dht, h0])."""
    if scale is None:
        scale = DK**-0.5
    BS = chunk_size
    NT = S // BS
    NV = DV // BV
    NCHAIN = B * H * NV
    CORE_EFF = min(CORE_NUM, NCHAIN)
    NTASK = (NCHAIN + CORE_EFF - 1) // CORE_EFF
    # q~ gating runs on [BS, 64] fp32 column chunks (row_expand 256B rows)
    NQK = DK // 64
    assert NQK in (1, 2), "DK in {64, 128} only (L0 domain)"
    gate_scale = -LN2 if use_exp2 else -1.0
    exp_scale = LN2 if use_exp2 else 1.0
    decay_scale = LN2 if use_exp2 else 1.0

    @T.prim_func
    def main(
        Q3: T.Tensor((B, S, H * DK), "bfloat16"),  # type: ignore
        K3: T.Tensor((B, S, H * DK), "bfloat16"),  # type: ignore
        W3: T.Tensor((B, S, H * DK), "bfloat16"),  # type: ignore
        Do3: T.Tensor((B, S, H * DV), "bfloat16"),  # type: ignore
        Dv3: T.Tensor((B, S, H * DV), "bfloat16"),  # type: ignore
        GFlat: T.Tensor((B, S * H), "float32"),  # type: ignore
        Gk3: T.Tensor((B, S, H * DK), "float32"),  # type: ignore
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
            # ---- V-side buffers
            dh_ub = T.alloc_shared([DK, BV], "float32")  # dH state, fp32
            dh16_ub = T.alloc_shared([DK, BV], "bfloat16")
            dvp_ub = T.alloc_shared([BS, BV], "float32")  # dv_part
            dv16_ub = T.alloc_shared([BS, BV], "bfloat16")
            dv_ub = T.alloc_shared([BS, BV], "float32")
            dv2_ub = T.alloc_shared([BS, BV], "float32")
            dv2_16_ub = T.alloc_shared([BS, BV], "bfloat16")
            q_half_a = T.alloc_shared([BS, 64], "bfloat16")
            q_ub_a = T.alloc_shared([BS, 64], "float32")
            qt16_a = T.alloc_shared([BS, 64], "bfloat16")
            # chunk-b buffers are allocated unconditionally (buffer allocs
            # inside a Python-level `if` are not registered by the parser
            # env); for NQK == 1 the b statements are const-false dropped
            # and the planner DCEs the dead buffers
            q_half_b = T.alloc_shared([BS, 64], "bfloat16")
            q_ub_b = T.alloc_shared([BS, 64], "float32")
            qt16_b = T.alloc_shared([BS, 64], "bfloat16")
            g_span = T.alloc_shared([BS * H], "float32")
            gdiff_ub = T.alloc_shared([BS, 1], "float32")
            gate_ub = T.alloc_shared([BS, 1], "float32")
            glast_ub = T.alloc_ub([1], "float32")  # g_last staging
            gpre_ub = T.alloc_shared([BS, 1], "float32")
            gexp_ub = T.alloc_shared([BS, 1], "float32")  # exp(g) for q~
            gkl_ub = T.alloc_shared([DK], "float32")
            dsrc_ub = T.alloc_shared([DK, 1], "float32")
            qdo_ub = T.alloc_shared([DK, BV], "float32")
            wdv2_ub = T.alloc_shared([DK, BV], "float32")
            tmp_gate = T.alloc_shared([BS, 8], "float32")
            tmp_qa = T.alloc_shared([BS, 8], "float32")
            tmp_qb = T.alloc_shared([BS, 8], "float32")
            tmp_decay = T.alloc_shared([DK, 8], "float32")
            # ---- C-side buffers
            k_l1 = T.alloc_shared([BS, DK], "bfloat16")
            w_l1 = T.alloc_shared([BS, DK], "bfloat16")
            qt16_l1 = T.alloc_shared([BS, DK], "bfloat16")
            do_l1 = T.alloc_shared([BS, BV], "bfloat16")
            dv2_16_l1 = T.alloc_shared([BS, BV], "bfloat16")
            dh16_l1 = T.alloc_shared([DK, BV], "bfloat16")
            dvp_frag = T.alloc_fragment([BS, BV], "float32")
            qdo_frag = T.alloc_fragment([DK, BV], "float32")
            wdv2_frag = T.alloc_fragment([DK, BV], "float32")

            for k_task in T.serial(NTASK):
                tid = cid + k_task * CORE_EFF
                if tid < NCHAIN:
                    i_v = tid % NV
                    i_nh = tid // NV
                    bb = i_nh // H
                    bh = i_nh % H
                    v0 = i_v * BV
                    if use_final_state_gradient:
                        T.copy(Dht[bb, bh, 0:DK, v0 : v0 + BV], dh_ub)
                    else:
                        T.tile.fill(dh_ub, 0.0)
                    for it_f in T.serial(NT):
                        i_t = NT - 1 - it_f
                        s0 = i_t * BS
                        n_t = s0 + BS - 1
                        # (a) snapshot dh[t] (ungated) + dH16 relay
                        T.copy(dh_ub, dh16_ub)
                        T.copy(dh16_ub, Dh[bb, i_t, bh, 0:DK, v0 : v0 + BV])
                        T.copy(dh16_ub, workspace_1[cid, 0, 0:DK, 0:BV])
                        # (b) Cube: dv_part = k[t] @ dH16
                        T.copy(K3[bb, s0 : s0 + BS, bh * DK : (bh + 1) * DK], k_l1)
                        T.copy(workspace_1[cid, 0, 0:DK, 0:BV], dh16_l1)
                        T.gemm_v0(k_l1, dh16_l1, dvp_frag, init=True)
                        T.copy(dvp_frag, workspace_2[cid, 0, 0:BS, 0:BV])
                        # (c) Vector: gate dv_part; dv2 = dv + dv_part
                        # (T.serial extraction loop -- see the fwd note)
                        T.copy(workspace_2[cid, 0, 0:BS, 0:BV], dvp_ub)
                        T.copy(GFlat[bb, s0 * H : (s0 + BS) * H], g_span)
                        glast_ub[0] = GFlat[bb, n_t * H + bh]
                        T.tile.fill(gdiff_ub, 0.0)
                        for i in T.serial(BS):
                            gdiff_ub[i, 0] = (g_span[i * H + bh] - glast_ub[0]) * gate_scale
                            gpre_ub[i, 0] = g_span[i * H + bh] * exp_scale
                        T.tile.exp(gate_ub, gdiff_ub)
                        T.tile.row_expand_mul_experiment(dvp_ub, dvp_ub, gate_ub, tmp_gate)
                        T.copy(
                            Dv3[bb, s0 : s0 + BS, bh * DV + v0 : bh * DV + v0 + BV],
                            dv16_ub,
                        )
                        T.copy(dv16_ub, dv_ub)
                        T.tile.add(dv2_ub, dv_ub, dvp_ub)
                        T.copy(dv2_ub, dv2_16_ub)
                        T.copy(
                            dv2_16_ub,
                            Dv2[bb, s0 : s0 + BS, bh * DV + v0 : bh * DV + v0 + BV],
                        )
                        T.copy(dv2_16_ub, workspace_4[cid, 0, 0:BS, 0:BV])
                        # (d) Vector: q~ = q * exp(g); relay in DK chunks
                        T.copy(
                            Q3[
                                bb,
                                s0 : s0 + BS,
                                bh * DK : bh * DK + 64,
                            ],
                            q_half_a,
                        )
                        T.copy(q_half_a, q_ub_a)
                        if NQK == 2:
                            T.copy(
                                Q3[
                                    bb,
                                    s0 : s0 + BS,
                                    bh * DK + 64 : bh * DK + 128,
                                ],
                                q_half_b,
                            )
                            T.copy(q_half_b, q_ub_b)
                        T.tile.exp(gexp_ub, gpre_ub)
                        T.tile.row_expand_mul_experiment(q_ub_a, q_ub_a, gexp_ub, tmp_qa)
                        T.copy(q_ub_a, qt16_a)
                        T.copy(qt16_a, workspace_3[cid, 0, 0:BS, 0:64])
                        if NQK == 2:  # (ws3b keeps a 1:1 site pair for NQK==1)
                            T.tile.row_expand_mul_experiment(q_ub_b, q_ub_b, gexp_ub, tmp_qb)
                            T.copy(q_ub_b, qt16_b)
                            T.copy(qt16_b, workspace_3b[cid, 0, 0:BS, 0:64])
                        # (e) Cube: qdo = q~^T @ do; wdv2 = w^T @ dv2
                        T.copy(
                            Do3[bb, s0 : s0 + BS, bh * DV + v0 : bh * DV + v0 + BV],
                            do_l1,
                        )
                        T.copy(workspace_3[cid, 0, 0:BS, 0:64], qt16_l1[0:BS, 0:64])
                        if NQK == 2:
                            T.copy(workspace_3b[cid, 0, 0:BS, 0:64], qt16_l1[0:BS, 64:DK])
                        T.gemm_v0(qt16_l1, do_l1, qdo_frag, transpose_A=True, init=True)
                        T.copy(qdo_frag, workspace_5[cid, 0, 0:DK, 0:BV])
                        T.copy(W3[bb, s0 : s0 + BS, bh * DK : (bh + 1) * DK], w_l1)
                        T.copy(workspace_4[cid, 0, 0:BS, 0:BV], dv2_16_l1)
                        T.gemm_v0(w_l1, dv2_16_l1, wdv2_frag, transpose_A=True, init=True)
                        T.copy(wdv2_frag, workspace_6[cid, 0, 0:DK, 0:BV])
                        # (f) Vector: dH = D*dH + qdo*scale - wdv2
                        T.copy(Gk3[bb, n_t, bh * DK : (bh + 1) * DK], gkl_ub)
                        T.tile.fill(dsrc_ub, 0.0)
                        for k in T.serial(DK):
                            dsrc_ub[k, 0] = (gkl_ub[k] + glast_ub[0]) * decay_scale
                        T.tile.exp(dsrc_ub, dsrc_ub)
                        T.copy(workspace_5[cid, 0, 0:DK, 0:BV], qdo_ub)
                        T.copy(workspace_6[cid, 0, 0:DK, 0:BV], wdv2_ub)
                        T.tile.mul(qdo_ub, qdo_ub, scale)
                        T.tile.row_expand_mul_experiment(dh_ub, dh_ub, dsrc_ub, tmp_decay)
                        T.tile.add(dh_ub, dh_ub, qdo_ub)
                        T.tile.sub(dh_ub, dh_ub, wdv2_ub)
                    if use_initial_state:
                        T.copy(dh_ub, Dh0[bb, bh, 0:DK, v0 : v0 + BV])

    return main


# ============================================================================
# Host wrappers (upstream naming: chunk_gated_delta_rule_{fwd_h, bwd_dhu} ->
# fla_chunk_delta_{fwd, bwd}; signature aligned with the proto schema).
# Only zero-copy metadata views + kernel calls here (the JIT adapter owns
# output/workspace allocation).
# ============================================================================
def _check_domain(B, S, H, DK, DV, chunk_size):
    assert chunk_size == 64, "chunk_size fixed at 64 (block_S semantic)"
    assert S % chunk_size == 0, "S must be divisible by chunk_size"
    assert DK in (64, 128), "DK in {64, 128} (L0 domain; 192/256 = L1 extension)"
    assert DV % 64 == 0, "DV must be a multiple of 64 (BV=64)"
    assert DK <= 256, "upstream assert K <= 256"


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
        g.view(B, S * H),
        gk.view(B, S, H * DK),
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
        g.view(B, S * H),
        gk.view(B, S, H * DK),
        dht if dht is not None else _dummy(device, torch.float32),
    )
    dv2 = dv2.view(B, S, H, DV)
    return dh, (dh0 if h0 is not None else None), dv2


__all__ = [
    "fla_chunk_delta_fwd",
    "fla_chunk_delta_bwd",
    "fla_chunk_delta_fwd_kernel",
    "fla_chunk_delta_bwd_kernel",
    "PASS_CONFIGS",
]
