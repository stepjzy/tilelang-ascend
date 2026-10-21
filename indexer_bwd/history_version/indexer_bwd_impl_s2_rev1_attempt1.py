"""indexer_bwd — DSA sparse finetune lightning indexer backward (Ascend NPU, Developer mode).

Computes the gradients of the varlen KL-divergence loss w.r.t. the lightning
indexer inputs (semantics replicate examples_experiment/dsa_sparse_finetune/
ref_gpu/indexer_bwd.py):

    logits[t, h, p]  = relu(q[t, h, :] . k[p, :] * sm_scale)
    d_logits[t, h, p] = (IndexScore[t, p] - AttnScore[t, p])
                        * 1{logits > 0} * Weights[t, h]
    dWeights[t, h]   += sum_p (IndexScore - AttnScore)[t, p] * logits[t, h, p]
    dIndexQ[t, h, :] += sum_p d_logits[t, h, p] * IndexK[p, :] * sm_scale
    dIndexK[p, :]    += sum_h d_logits[t, h, p] * IndexQ[t, h, :] * sm_scale  (atomic)

Architecture (Developer mode, threads=1, 4 auto pass_configs; one block per
token, grid = seq_len, varlen decoded via TokenIndices/Offsets):

    V: gather topk K rows (dynamic pos, fill-0 for invalid) -> GM workspace_k
    C: workspace_k -> k_l1; GEMM1 logits = K x Q^T (relu on the L0C->GM leg)
    V: logits (fp32) -> scale/mask/dW reduce -> d_logits (x sm_scale, bf16)
       -> GM workspace_dlog
    C: workspace_dlog -> dlog_l1; GEMM2 dq += dlog^T x K; GEMM3 dk = dlog x Q
    V: dk (L0C fp32 -> GM bf16) -> per-row DMA atomic add into dIndexK

Implementation notes (deviations from DESIGN.md §3.3 pseudocode, each forced
by a platform constraint found in the TileLang-Ascend sources):

 1. UB->L1 direct transfer is fp16-only (src/tl_templates/ascend/common.h
    copy_ub_to_l1 static_assert "only support half"), so the bf16 Cube
    operands (gathered K, d_logits) are staged through GM workspaces — the
    sparse_mla_topk_reducesum / sparse_flash_attention family pattern.
 2. The compiler's auto workspace allocation for rewritten L0C->UB / UB->L1
    copies does not support symbolic (dynamic) grid shapes
    (tilelang/jit/adapter/cython/cython_wrapper.pyx "Auto GM: Dynamic shape
    scenarios are not supported currently"), so all four staging buffers are
    explicit kernel parameters allocated by the host wrapper.
 3. The staging buffers are named "workspace_*" on purpose: the auto
    cross-core sync pass (TL_ASCEND_AUTO_CV_SYNC) pairs GM writes/reads by
    that name substring and inserts the C/V handshake flags automatically.
 4. sm_scale is folded after GEMM1 (fp32 UB multiply on the relu'd logits)
    and into d_logits before its bf16 cast, instead of pre-scaling Q in bf16.
    This is mathematically equivalent to the GPU original (which pre-multiplies
    q by sm_scale in bf16 and rescales dq in fp32) but with fewer bf16
    rounding points, so precision is >= the main repo's path.
"""

import torch
import tilelang
from tilelang import language as T

_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(pass_configs=_pass_configs)
def tl_indexer_bwd_impl(heads, dim, topk, sm_scale=None, block_I=64):
    """Build the indexer_bwd kernel (Developer mode).

    Args:
        heads: number of heads H (must be % 16 == 0 and <= 64).
        dim: head dim D (must be % 16 == 0).
        topk: number of selected positions (power of two, % block_I == 0).
        sm_scale: softmax scale, defaults to dim ** -0.5.
        block_I: topk-direction tile size (multiple of 16, default 64).
    """
    # Compile-time domain checks (DESIGN.md §5.2; the %16 head constraint is a
    # declared Ascend fractal-granularity difference vs the main repo's %8).
    assert topk == tilelang.math.next_power_of_2(topk), "topk must be a power of 2"
    assert topk % block_I == 0, "topk must be divisible by block_I"
    assert block_I % 16 == 0, "block_I must be a multiple of 16 (Cube fractal)"
    assert heads % 16 == 0 and heads <= 64, "heads must be % 16 == 0 and <= 64 (Cube fractal)"
    assert dim % 16 == 0, "dim must be a multiple of 16 (Cube fractal)"
    if sm_scale is None:
        sm_scale = dim**-0.5

    NI = topk // block_I  # topk blocks per token (compile-time constant)

    seq_len = T.symbolic("seq_len")  # total packed length (sum of segment lens)
    batch_plus_one = T.symbolic("batch_plus_one")  # Offsets length (batch + 1)

    dtype = "bfloat16"
    accum_dtype = "float32"
    indices_dtype = "int32"

    @T.prim_func
    def main(
        IndexQ: T.Tensor([seq_len, heads, dim], dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], dtype),  # type: ignore
        IndexK: T.Tensor([seq_len, dim], dtype),  # type: ignore
        AttnScore: T.Tensor([seq_len, topk], accum_dtype),  # type: ignore
        IndexScore: T.Tensor([seq_len, topk], accum_dtype),  # type: ignore
        TopkIndices: T.Tensor([seq_len, topk], indices_dtype),  # type: ignore
        Offsets: T.Tensor([batch_plus_one], indices_dtype),  # type: ignore
        TokenIndices: T.Tensor([seq_len, 2], indices_dtype),  # type: ignore
        dIndexQ: T.Tensor([seq_len, heads, dim], dtype),  # type: ignore
        dWeights: T.Tensor([seq_len, heads], dtype),  # type: ignore
        dIndexK: T.Tensor([seq_len, dim], dtype),  # type: ignore
        workspace_k: T.Tensor([seq_len, topk, dim], dtype),  # type: ignore
        workspace_dlog: T.Tensor([seq_len, topk, heads], dtype),  # type: ignore
        workspace_logits: T.Tensor([seq_len, topk, heads], accum_dtype),  # type: ignore
        workspace_dk: T.Tensor([seq_len, topk, dim], dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=1, is_npu=True) as (cid):
            i_b, i_t = TokenIndices[cid, 0], TokenIndices[cid, 1]
            bos = Offsets[i_b]

            # ---- L1: Cube GEMM operands (compiler maps alloc_shared -> L1) ----
            q_l1 = T.alloc_shared([heads, dim], dtype)
            k_l1 = T.alloc_shared([block_I, dim], dtype)
            dlog_l1 = T.alloc_shared([block_I, heads], dtype)

            # ---- L0C: GEMM accumulators (alloc_fragment) ----
            logits_l0c = T.alloc_fragment([block_I, heads], accum_dtype)
            dq_l0c = T.alloc_fragment([heads, dim], accum_dtype)
            dk_l0c = T.alloc_fragment([block_I, dim], accum_dtype)

            # ---- UB: Vector-side buffers (alloc_shared -> UB) ----
            indices_ub = T.alloc_shared([block_I], indices_dtype)
            attn_ub = T.alloc_shared([block_I], accum_dtype)
            is_ub = T.alloc_shared([block_I], accum_dtype)
            diff_ub = T.alloc_shared([block_I], accum_dtype)
            gather_ub = T.alloc_shared([block_I, dim], dtype)
            logits_ub = T.alloc_shared([block_I, heads], accum_dtype)
            dw_ub = T.alloc_shared([block_I, heads], accum_dtype)
            dlog_ub = T.alloc_shared([block_I, heads], accum_dtype)
            mask_ub = T.alloc_shared([block_I, heads], accum_dtype)
            diff2d = T.alloc_shared([block_I, heads], accum_dtype)
            w2d = T.alloc_shared([block_I, heads], accum_dtype)
            zero_ub = T.alloc_shared([block_I, heads], accum_dtype)
            w_ub_bf16 = T.alloc_shared([heads], dtype)
            w_ub = T.alloc_shared([heads], accum_dtype)
            dw_frag = T.alloc_shared([heads], accum_dtype)
            dw_bf16 = T.alloc_shared([heads], dtype)
            dlog_bf16_ub = T.alloc_shared([block_I, heads], dtype)
            dk_ub = T.alloc_shared([block_I, dim], dtype)

            # ---- per-token prologue ----
            T.copy(IndexQ[bos + i_t, :, :], q_l1)
            T.copy(Weights[bos + i_t, :], w_ub_bf16)
            T.copy(w_ub_bf16, w_ub)  # bf16 -> fp32 auto-cast
            T.tile.broadcast(w2d, w_ub, axis=0)  # [H] -> [block_I, H]
            T.tile.fill(zero_ub, 0.0)
            T.tile.fill(dw_frag, 0.0)

            for it in T.serial(NI):
                st = it * block_I

                # ---- V: load indices + scores, diff ----
                T.copy(TopkIndices[bos + i_t, st : st + block_I], indices_ub)
                T.copy(AttnScore[bos + i_t, st : st + block_I], attn_ub)
                T.copy(IndexScore[bos + i_t, st : st + block_I], is_ub)
                T.tile.sub(diff_ub, is_ub, attn_ub)  # IndexScore - AttnScore

                # ---- V: gather K rows; invalid pos keeps the fill-0 row ----
                T.tile.fill(gather_ub, 0.0)
                for i in T.serial(block_I):
                    pos = indices_ub[i]
                    if (pos > -1) & (pos <= i_t):
                        T.copy(IndexK[bos + pos, :], gather_ub[i, :])
                T.copy(gather_ub, workspace_k[cid, st : st + block_I, 0:dim])

                # ---- C: workspace -> L1, GEMM1 logits = K x Q^T ----
                T.copy(workspace_k[cid, st : st + block_I, 0:dim], k_l1)
                T.gemm_v0(k_l1, q_l1, logits_l0c, transpose_B=True, init=True)

                # ---- C: L0C -> GM workspace (relu folded into the copy) ----
                T.copy(logits_l0c, workspace_logits[cid, st : st + block_I, 0:heads], enable_relu=True)

                # ---- V: logits -> UB, fold sm_scale in fp32 ----
                T.copy(workspace_logits[cid, st : st + block_I, 0:heads], logits_ub)
                T.tile.mul(logits_ub, logits_ub, sm_scale)

                # ---- V: dWeights accumulation (across NI blocks) ----
                T.tile.broadcast(diff2d, diff_ub, axis=1)  # [block_I] -> [block_I, H]
                T.tile.mul(dw_ub, diff2d, logits_ub)
                T.reduce_sum(dw_ub, dw_frag, dim=0, clear=False)

                # ---- V: d_logits = diff x 1{logits>0} x Weights, fold sm_scale ----
                T.tile.mul(dlog_ub, diff2d, w2d)
                T.tile.compare(mask_ub, logits_ub, zero_ub, "GT")
                T.tile.select(dlog_ub, mask_ub, dlog_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.tile.mul(dlog_ub, dlog_ub, sm_scale)
                T.copy(dlog_ub, dlog_bf16_ub)  # fp32 -> bf16 auto-cast
                T.copy(dlog_bf16_ub, workspace_dlog[cid, st : st + block_I, 0:heads])

                # ---- C: workspace -> L1, GEMM2 dq / GEMM3 dk ----
                T.copy(workspace_dlog[cid, st : st + block_I, 0:heads], dlog_l1)
                T.gemm_v0(dlog_l1, k_l1, dq_l0c, transpose_A=True, init=(it == 0))
                T.gemm_v0(dlog_l1, q_l1, dk_l0c, init=True)

                # ---- C: dk L0C -> GM workspace (fp32 -> bf16 auto-cast) ----
                T.copy(dk_l0c, workspace_dk[cid, st : st + block_I, 0:dim])

                # ---- V: dk -> UB, per-row atomic scatter into dIndexK ----
                T.copy(workspace_dk[cid, st : st + block_I, 0:dim], dk_ub)
                for i in T.serial(block_I):
                    pos = indices_ub[i]
                    if (pos > -1) & (pos <= i_t):
                        T.tile.atomic_add(dIndexK[bos + pos, 0], dk_ub[i, :])

            # ---- tail writeback ----
            T.copy(dq_l0c, dIndexQ[bos + i_t, :, :])  # L0C fp32 -> GM bf16
            T.copy(dw_frag, dw_bf16)  # fp32 -> bf16 auto-cast
            T.copy(dw_bf16, dWeights[bos + i_t, :])

    return main


def prepare_token_indices(offsets):
    """Build TokenIndices [sum(S), 2] int32 ((batch_id, position)) from cu_seqlens.

    Semantics match ref_gpu/index.py::prepare_token_indices; CPU construction
    followed by a single H2D copy (sparse_mla_topk_reducesum.py precedent).
    """
    off = offsets.detach().cpu().tolist()
    batch_ids = []
    positions = []
    for b in range(len(off) - 1):
        n = int(off[b + 1]) - int(off[b])
        batch_ids.append(torch.full((n,), b, dtype=torch.int32))
        positions.append(torch.arange(n, dtype=torch.int32))
    if batch_ids:
        ti = torch.stack([torch.cat(batch_ids), torch.cat(positions)], dim=1)
    else:
        ti = torch.zeros((0, 2), dtype=torch.int32)
    return ti.to(offsets.device)


_kernel_cache = {}


def _get_kernel(heads, dim, topk, sm_scale, block_I):
    key = (heads, dim, topk, sm_scale, block_I)
    if key not in _kernel_cache:
        _kernel_cache[key] = tl_indexer_bwd_impl(heads, dim, topk, sm_scale, block_I)
    return _kernel_cache[key]


def indexer_bwd_interface(
    q,
    weights,
    k,
    attn_score,
    index_score,
    topk_indices,
    offsets,
    block_I=None,
):
    """Host entry: allocate outputs/workspaces, launch the kernel, return grads.

    Args:
        q: IndexQ [seq_len, heads, dim] bfloat16.
        weights: Weights [seq_len, heads] bfloat16.
        k: IndexK [seq_len, dim] bfloat16.
        attn_score: teacher probabilities [seq_len, topk] float32.
        index_score: student probabilities [seq_len, topk] float32.
        topk_indices: selected positions [seq_len, topk] int32 (values in
            [-1, seq_len), -1 = invalid).
        offsets: cu_seqlens [batch + 1] int32; every segment length >= topk.
        block_I: topk-direction tile (default min(topk, 64)).

    Returns:
        (dIndexQ, dWeights, dIndexK) bfloat16 tensors.
    """
    seq_len, heads, dim = q.shape
    topk = topk_indices.shape[-1]
    if block_I is None:
        block_I = min(topk, 64)

    # Host-side domain validation (mirrors the kernel compile-time asserts).
    if topk != tilelang.math.next_power_of_2(topk):
        raise ValueError(f"topk must be a power of 2, got {topk}")
    if topk % block_I != 0:
        raise ValueError(f"topk ({topk}) must be divisible by block_I ({block_I})")
    if block_I % 16 != 0:
        raise ValueError(f"block_I must be a multiple of 16, got {block_I}")
    if heads % 16 != 0 or heads > 64:
        raise ValueError(f"heads must be % 16 == 0 and <= 64, got {heads}")
    if dim % 16 != 0:
        raise ValueError(f"dim must be a multiple of 16, got {dim}")

    off = offsets.detach().cpu().tolist()
    for b in range(len(off) - 1):
        if int(off[b + 1]) - int(off[b]) < topk:
            raise ValueError(f"varlen segment {b} has length {int(off[b + 1]) - int(off[b])} < topk {topk}")

    token_indices = prepare_token_indices(offsets)

    # Outputs: dIndexK must be zero-initialized (atomic accumulation target).
    dq = torch.zeros(seq_len, heads, dim, dtype=q.dtype, device=q.device)
    dw = torch.zeros(seq_len, heads, dtype=weights.dtype, device=weights.device)
    dk = torch.zeros(seq_len, dim, dtype=k.dtype, device=k.device)

    # Staging workspaces: every element is written before it is read within
    # each kernel launch, so empty (uninitialized) allocation is sufficient.
    workspace_k = torch.empty(seq_len, topk, dim, dtype=k.dtype, device=k.device)
    workspace_dlog = torch.empty(seq_len, topk, heads, dtype=q.dtype, device=q.device)
    workspace_logits = torch.empty(seq_len, topk, heads, dtype=torch.float32, device=q.device)
    workspace_dk = torch.empty(seq_len, topk, dim, dtype=k.dtype, device=k.device)

    kernel = _get_kernel(heads, dim, topk, dim**-0.5, block_I)
    kernel(
        q,
        weights,
        k,
        attn_score,
        index_score,
        topk_indices,
        offsets,
        token_indices,
        dq,
        dw,
        dk,
        workspace_k,
        workspace_dlog,
        workspace_logits,
        workspace_dk,
    )
    torch.npu.synchronize()
    return dq, dw, dk


# Module-level alias matching the file name (import convenience for tests).
indexer_bwd = indexer_bwd_interface
