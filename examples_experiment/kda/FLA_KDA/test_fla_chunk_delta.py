"""fla_chunk_delta layered tests: L0 (+ L1/L2/Boundary stubs) + perf sentinel.

Golden: PyTorch fp64 chunk-loop reference with upstream cast simulation
(DESIGN SS9.1, verbatim; the state H is a recurrence intermediate whose
per-chunk half-precision cast points must be mirrored, otherwise the
128-chunk accumulated cast error (~|w@H|*2^-11) swamps the output gates).
Precision: mixed-tolerance dual gate (precision-standard.md SS4.1, verbatim).
"""

import argparse
import os
import sys

import tilelang
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fla_chunk_delta import fla_chunk_delta_bwd, fla_chunk_delta_fwd  # noqa: E402

LN2 = 0.6931471805599453
DEVICE = "npu:0"


# ============================================================================
# Golden reference implementations (DESIGN SS9.1, verbatim)
# ============================================================================
def _exp(g, use_exp2):
    """Gate exponent: use_exp2 -> g in log2 domain (2^g), else ln domain."""
    return torch.exp2(g) if use_exp2 else torch.exp(g)


def fla_chunk_delta_fwd_ref(
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
    """fwd golden. k/w [B,S,H,DK] fp16, u [B,S,H,DV] fp16, g [B,S,H] fp32,
    gk [B,S,H,DK] fp32 (chunk-cumulative log decay domain).
    Returns (h [B,NT,H,DK,DV] fp16, v_new fp16|None, ht fp32|None)."""
    B, S, H, DK = k.shape
    DV = u.shape[-1]
    C = chunk_size
    assert S % C == 0
    NT = S // C
    kd = k.dtype
    kf, wf, uf = k.double(), w.double(), u.double()
    gf, gkf = g.double(), gk.double()
    Hd = torch.zeros(B, H, DK, DV, dtype=torch.float64)
    if initial_state is not None:
        Hd += initial_state.double()
    h_out = torch.empty(B, NT, H, DK, DV, dtype=kd)
    v_new = torch.empty(B, S, H, DV, dtype=kd) if save_new_value else None
    for i in range(NT):
        sl = slice(i * C, (i + 1) * C)
        n_t = (i + 1) * C - 1
        h_out[:, i] = Hd.to(kd)  # cast point 1: h snapshot storage
        # v_new = u - w @ H16 (cast point 2: H half-precision into the gemm)
        H16 = Hd.to(kd).double()
        b_v = torch.einsum("bthd,bhdv->bthv", wf[:, sl], H16)
        vn = uf[:, sl] - b_v
        if save_new_value:
            v_new[:, sl] = vn.to(kd)
        # USE_G token gate (S%64==0 domain: mask is all-true)
        g_last = gf[:, n_t, :]  # [B, H]
        gate = _exp(g_last.unsqueeze(1) - gf[:, sl], use_exp2)  # [B, C, H]
        vn = vn * gate.unsqueeze(-1)
        Hd = Hd * _exp(g_last, use_exp2).unsqueeze(-1).unsqueeze(-1)
        # USE_GK per-channel gate
        gk_last = gkf[:, n_t, :, :]  # [B, H, DK]
        Hd = Hd * _exp(gk_last, use_exp2).unsqueeze(-1)
        # H += k^T @ v_new16 (cast point 3: v_new half-precision into the gemm)
        vn16 = vn.to(kd).double()
        Hd = Hd + torch.einsum("bthd,bthv->bhdv", kf[:, sl], vn16)
    ht = Hd.float() if output_final_state else None  # STORE_FINAL_STATE
    return h_out, v_new, ht


def fla_chunk_delta_bwd_ref(
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
    """bwd golden. q/k/w [B,S,H,DK] bf16, do/dv [B,S,H,DV] bf16, g/gk fp32.
    Returns (dh bf16, dh0 fp32|None, dv2 bf16). scale defaults to DK**-0.5."""
    B, S, H, DK = q.shape
    DV = do.shape[-1]
    C = chunk_size
    assert S % C == 0
    NT = S // C
    if scale is None:
        scale = DK**-0.5
    qd = q.double()
    kd = k.double()
    wd = w.double()
    dod, dvd = do.double(), dv.double()
    gf, gkf = g.double(), gk.double()
    dH = torch.zeros(B, H, DK, DV, dtype=torch.float64)
    if dht is not None:
        dH += dht.double()
    dh_out = torch.empty(B, NT, H, DK, DV, dtype=q.dtype)
    dv2 = torch.empty(B, S, H, DV, dtype=dv.dtype)
    for i in range(NT - 1, -1, -1):
        sl = slice(i * C, (i + 1) * C)
        n_t = (i + 1) * C - 1
        dh_out[:, i] = dH.to(q.dtype)  # snapshot
        g_last = gf[:, n_t, :]
        gate = _exp(g_last.unsqueeze(1) - gf[:, sl], use_exp2)
        # dv_part = (k @ dH16) * gate (cast point: dH half-precision into gemm)
        dH16 = dH.to(q.dtype).double()
        dv_part = torch.einsum("bthd,bhdv->bthv", kd[:, sl], dH16)
        dv_part = dv_part * gate.unsqueeze(-1)
        dv2[:, sl] = (dvd[:, sl] + dv_part).to(dv.dtype)
        # dH update -- NOTE: the gk per-channel decay applies BEFORE the
        # qdo/wdv2 update (upstream chunk_gated_delta_rule_bwd_dhu order and
        # DESIGN SS1.3/SS3.2; the SS9.1 code block placed it after the update,
        # which decays the update terms -- a transcription error in the design
        # doc, corrected here against the frozen upstream reference)
        dH = dH * _exp(g_last, use_exp2).unsqueeze(-1).unsqueeze(-1)
        dH = dH * _exp(gkf[:, n_t, :, :], use_exp2).unsqueeze(-1)
        qt = (qd[:, sl] * _exp(gf[:, sl], use_exp2).unsqueeze(-1)).to(q.dtype).double()
        qdo = torch.einsum("bthd,bthv->bhdv", qt, dod[:, sl])
        wdv2 = torch.einsum("bthd,bthv->bhdv", wd[:, sl], dv2[:, sl].double())
        dH = dH + qdo * scale - wdv2
    dh0 = dH.float() if h0 is not None else None  # USE_INITIAL_STATE
    return dh_out, dh0, dv2


# ============================================================================
# Precision standard (precision-standard.md SS4.1, verbatim)
# ============================================================================
def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio)."""
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
        "hifloat32": (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4, 2**-2, 1e0, 0.99),
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),
    }
    int_types = {"int8", "int16", "int32", "int64", "uint8"}
    if dtype in int_types:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Mixed-tolerance dual-gate check: (passed, matched_ratio, max_abs_error)."""
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:  # integer exact match
        mism = (a != g).sum().item()
        total = max(a.numel(), 1)
        return mism == 0, 1.0 - mism / total, (0.0 if mism == 0 else float("inf"))
    a = a.float()
    g = g.float()
    special = ~torch.isfinite(g)  # inf/nan structural comparison
    if special.any() and (
        not torch.equal(torch.isnan(a[special]), torch.isnan(g[special]))
        or not torch.equal(torch.isinf(a[special]), torch.isinf(g[special]))
    ):
        return False, 0.0, float("inf")
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    matched_ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs_error = abs_err.max().item()
    passed = (matched_ratio >= required_ratio) and (max_abs_error <= max_abs_limit)
    return passed, matched_ratio, max_abs_error


# ============================================================================
# Input construction (CPU randn + chunk-cumulative log decay + single H2D;
# family convention -- g/gk MUST be within-chunk cumulative decays so that
# exp(g_last - g) in (0, 1] and the state stays bounded over the recurrence)
# ============================================================================
def _chunk_cumsum_decay(shape, s_axis, chunk, gen, use_exp2):
    """Chunk-cumulative log-decay tensor of `shape` along axis `s_axis`.

    raw = logsigmoid(randn) -> split s_axis into [NT, C] in place ->
    cumsum over C -> restore. use_exp2 converts to the log2 domain
    (g2 = g_ln / ln2, so 2^g2 == e^g_ln)."""
    raw = F.logsigmoid(torch.randn(shape, generator=gen, dtype=torch.float32))
    ntt = shape[s_axis] // chunk
    new_shape = list(shape[:s_axis]) + [ntt, chunk] + list(shape[s_axis + 1 :])
    out = raw.reshape(new_shape).cumsum(dim=s_axis + 1).reshape(shape)
    if use_exp2:
        out = out / LN2
    return out


def _make_g_gk(B, S, H, DK, gen, use_exp2):
    g = _chunk_cumsum_decay((B, S, H), 1, 64, gen, use_exp2)
    gk = _chunk_cumsum_decay((B, S, H, DK), 1, 64, gen, use_exp2)
    return g, gk


def make_fwd_inputs(B, S, H, DK, DV, seed, initial_state=False, use_exp2=False):
    """CPU-side input construction (single H2D at the call site).

    k/w are L2-normalized per token (gdn_chunk_h.py family precedent): with
    plain randn w the recurrence's error feedback (v' = u - w@H, per-chunk
    gain ~ sigma(k^T)*sigma(w)) amplifies any legitimate fp32-vs-fp64
    difference (~1e-5: HW exp accuracy, accumulation order) by ~90x per
    chunk, swamping the fp16 output gates after a few chunks. The real
    operator consumes WY-constructed w (w@H ~ u by construction); the
    normalized family construction is the standard well-conditioned proxy.
    """
    gen = torch.Generator().manual_seed(seed)
    k = F.normalize(torch.randn(B, S, H, DK, generator=gen), dim=-1, p=2).to(torch.float16)
    w = F.normalize(torch.randn(B, S, H, DK, generator=gen), dim=-1, p=2).to(torch.float16)
    u = torch.randn(B, S, H, DV, generator=gen).to(torch.float16)
    g, gk = _make_g_gk(B, S, H, DK, gen, use_exp2)
    h0 = torch.randn(B, H, DK, DV, generator=gen, dtype=torch.float32) if initial_state else None
    return k, w, u, g, gk, h0


def make_bwd_inputs(B, S, H, DK, DV, seed, dht=False, h0=False, use_exp2=False):
    """CPU-side input construction (single H2D; q/k/w normalized, see fwd)."""
    gen = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, S, H, DK, generator=gen), dim=-1, p=2).to(torch.bfloat16)
    k = F.normalize(torch.randn(B, S, H, DK, generator=gen), dim=-1, p=2).to(torch.bfloat16)
    w = F.normalize(torch.randn(B, S, H, DK, generator=gen), dim=-1, p=2).to(torch.bfloat16)
    do = torch.randn(B, S, H, DV, generator=gen).to(torch.bfloat16)
    dv = torch.randn(B, S, H, DV, generator=gen).to(torch.bfloat16)
    g, gk = _make_g_gk(B, S, H, DK, gen, use_exp2)
    dht_t = torch.randn(B, H, DK, DV, generator=gen, dtype=torch.float32) if dht else None
    h0_t = torch.randn(B, H, DK, DV, generator=gen, dtype=torch.float32) if h0 else None
    return q, k, w, do, dv, g, gk, dht_t, h0_t


# ============================================================================
# L0 gate tests (DESIGN SS9.2)
# ============================================================================
def test_fla_chunk_delta_l0():
    """L0: the six DESIGN SS9.2 gate cases (regular shapes, all constexpr
    branches: default / h0+ht+no-save+exp2 / dht+h0+exp2+scale)."""
    configs = [
        # (name, kind, shape, seed, kwargs)
        ("l0_fwd_min", "fwd", (1, 128, 2, 64, 64), 1001, {}),
        ("l0_fwd_repr", "fwd", (2, 256, 4, 128, 128), 1002, {}),
        (
            "l0_fwd_opts",
            "fwd",
            (1, 256, 2, 128, 128),
            1003,
            dict(initial_state=True, output_final_state=True, save_new_value=False, use_exp2=True),
        ),
        ("l0_bwd_min", "bwd", (1, 128, 2, 64, 64), 1004, {}),
        ("l0_bwd_repr", "bwd", (2, 256, 4, 128, 128), 1005, {}),
        (
            "l0_bwd_opts",
            "bwd",
            (1, 256, 2, 128, 128),
            1006,
            dict(dht=True, h0=True, use_exp2=True, scale=0.37),
        ),
    ]
    ok = True
    for name, kind, shape, seed, kw in configs:
        B, S, H, DK, DV = shape
        try:
            if kind == "fwd":
                in_kw = {k: kw[k] for k in ("initial_state", "use_exp2") if k in kw}
                k, w, u, g, gk, h0 = make_fwd_inputs(B, S, H, DK, DV, seed, **in_kw)
                h, v_new, ht = fla_chunk_delta_fwd(
                    k.to(DEVICE),
                    w.to(DEVICE),
                    u.to(DEVICE),
                    g.to(DEVICE),
                    gk.to(DEVICE),
                    initial_state=h0.to(DEVICE) if h0 is not None else None,
                    output_final_state=kw.get("output_final_state", False),
                    save_new_value=kw.get("save_new_value", True),
                    use_exp2=kw.get("use_exp2", False),
                )
                torch.npu.synchronize()
                gh, gv_new, ght = fla_chunk_delta_fwd_ref(
                    k,
                    w,
                    u,
                    g,
                    gk,
                    h0,
                    output_final_state=kw.get("output_final_state", False),
                    save_new_value=kw.get("save_new_value", True),
                    use_exp2=kw.get("use_exp2", False),
                )
                res = [("h", *check_precision(h, gh, "float16"))]
                if v_new is not None:
                    res.append(("v_new", *check_precision(v_new, gv_new, "float16")))
                if ht is not None:
                    res.append(("ht", *check_precision(ht, ght, "float32")))
            else:
                in_kw = {k: kw[k] for k in ("dht", "h0", "use_exp2") if k in kw}
                q, k, w, do, dv, g, gk, dht, h0 = make_bwd_inputs(B, S, H, DK, DV, seed, **in_kw)
                dh, dh0, dv2 = fla_chunk_delta_bwd(
                    q.to(DEVICE),
                    k.to(DEVICE),
                    w.to(DEVICE),
                    do.to(DEVICE),
                    dv.to(DEVICE),
                    g.to(DEVICE),
                    gk.to(DEVICE),
                    h0=h0.to(DEVICE) if h0 is not None else None,
                    dht=dht.to(DEVICE) if dht is not None else None,
                    scale=kw.get("scale"),
                    use_exp2=kw.get("use_exp2", False),
                )
                torch.npu.synchronize()
                gdh, gdh0, gdv2 = fla_chunk_delta_bwd_ref(
                    q,
                    k,
                    w,
                    do,
                    dv,
                    g,
                    gk,
                    h0,
                    dht,
                    scale=kw.get("scale"),
                    use_exp2=kw.get("use_exp2", False),
                )
                res = [
                    ("dh", *check_precision(dh, gdh, "bfloat16")),
                    ("dv2", *check_precision(dv2, gdv2, "bfloat16")),
                ]
                if dh0 is not None:
                    res.append(("dh0", *check_precision(dh0, gdh0, "float32")))
            tag = "PRECISION_PASS" if all(r[1] for r in res) else "PRECISION_FAIL"
            detail = " ".join(f"{n}:ratio={r:.4f},max_abs={m:.3e}" for n, _, r, m in res)
            print(f"[{tag}] {name} shape={shape} :: {detail}")
            ok &= tag == "PRECISION_PASS"
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"[PRECISION_FAIL] {name}: {e}")
            ok = False
    return ok


# ============================================================================
# L1 / L2 / Boundary (scenario B expansion, tilelang-op-test-design)
# Category: Fusion (CV: gemm_v0 + exp gating + cast), coverage per
# coverage-matrix.md. In-domain shapes only (S%64==0, DK in {64,128},
# DV%64==0 -- wrapper-asserted); the non-aligned shape scenarios
# (D-SHAPE-TAIL-*/PRIME) are covered through their L2 rejection cases.
# ============================================================================
COVERAGE_CATEGORY = "Fusion"

L1_CASES = [
    # (name, kind, (B,S,H,DK,DV), uscale, kwargs, tags)
    (
        "l1_fwd_edge_nt1",
        "fwd",
        (1, 64, 2, 64, 64),
        1.0,
        {},
        ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-SHAPE-EDGE", "D-VALRANGE-S"],
    ),
    (
        # NCHAIN = 2*11*1 = 22 > CORE_EFF=20 -> the striped task map has a
        # guarded tail (tids 22..39 skipped): the only case exercising the
        # `if tid < NCHAIN` path (all L0 cases divide evenly)
        "l1_fwd_tasktail",
        "fwd",
        (2, 128, 11, 64, 64),
        1.0,
        {},
        ["D-DTYPE-fp16", "D-SHAPE-ALIGNED"],
    ),
    (
        "l1_fwd_valrange_m",
        "fwd",
        (1, 128, 2, 128, 64),
        3.0,
        {},
        ["D-DTYPE-fp16", "D-VALRANGE-M"],
    ),
    (
        # NT=1 keeps the error feedback single-chunk: |h| stays inside the
        # fp16 output dtype's absolute max_abs gate
        "l1_fwd_valrange_l",
        "fwd",
        (1, 64, 2, 64, 64),
        100.0,
        {},
        ["D-DTYPE-fp16", "D-VALRANGE-L"],
    ),
    (
        "l1_fwd_valrange_asym",
        "fwd",
        (1, 128, 2, 64, 64),
        1.0,
        {"ushift": 4.0},
        ["D-DTYPE-fp16", "D-VALRANGE-ASYM"],
    ),
    (
        "l1_fwd_param_exp2",
        "fwd",
        (1, 128, 2, 64, 64),
        1.0,
        {"initial_state": True, "output_final_state": True, "use_exp2": True},
        ["D-DTYPE-fp32", "D-PARAM-use_exp2", "D-PARAM-output_final_state"],
    ),
    (
        "l1_fwd_param_nosave",
        "fwd",
        (1, 128, 3, 128, 128),
        1.0,
        {"save_new_value": False, "output_final_state": True},
        ["D-PARAM-save_new_value", "D-PARAM-output_final_state"],
    ),
    (
        "l1_bwd_aligned",
        "bwd",
        (1, 128, 2, 64, 64),
        1.0,
        {},
        ["D-DTYPE-bf16", "D-SHAPE-ALIGNED"],
    ),
    (
        # NCHAIN = 23 > 20 -> guarded task tail on the bwd side
        "l1_bwd_tasktail",
        "bwd",
        (1, 128, 23, 64, 64),
        1.0,
        {},
        ["D-DTYPE-bf16"],
    ),
    (
        "l1_bwd_param_scale",
        "bwd",
        (1, 128, 2, 128, 64),
        1.0,
        {"dht": True, "h0": True, "scale": 0.37, "use_exp2": True},
        ["D-DTYPE-fp32", "D-PARAM-scale", "D-PARAM-use_exp2"],
    ),
    (
        # bf16 outputs have a 1e0 max_abs gate -> the largest testable value
        # range rides the bwd side (the fp16 h snapshot caps the fwd side)
        "l1_bwd_valrange_l",
        "bwd",
        (1, 128, 2, 64, 64),
        500.0,
        {},
        ["D-DTYPE-bf16", "D-VALRANGE-L"],
    ),
]


def _run_l1_fwd(name, B, S, H, DK, DV, seed, uscale, kwargs):
    gen = torch.Generator().manual_seed(seed)
    k = F.normalize(torch.randn(B, S, H, DK, generator=gen), dim=-1, p=2).to(torch.float16)
    w = F.normalize(torch.randn(B, S, H, DK, generator=gen), dim=-1, p=2).to(torch.float16)
    ushift = kwargs.pop("ushift", 0.0)
    u = (torch.randn(B, S, H, DV, generator=gen) * uscale + ushift).to(torch.float16)
    use_exp2 = kwargs.get("use_exp2", False)
    g, gk = _make_g_gk(B, S, H, DK, gen, use_exp2)
    h0 = torch.randn(B, H, DK, DV, generator=gen, dtype=torch.float32) if kwargs.get("initial_state") else None
    h, v_new, ht = fla_chunk_delta_fwd(
        k.to(DEVICE),
        w.to(DEVICE),
        u.to(DEVICE),
        g.to(DEVICE),
        gk.to(DEVICE),
        initial_state=h0.to(DEVICE) if h0 is not None else None,
        output_final_state=kwargs.get("output_final_state", False),
        save_new_value=kwargs.get("save_new_value", True),
        use_exp2=use_exp2,
    )
    torch.npu.synchronize()
    gh, gv_new, ght = fla_chunk_delta_fwd_ref(
        k,
        w,
        u,
        g,
        gk,
        h0,
        output_final_state=kwargs.get("output_final_state", False),
        save_new_value=kwargs.get("save_new_value", True),
        use_exp2=use_exp2,
    )
    res = [("h", *check_precision(h, gh, "float16"))]
    if v_new is not None:
        res.append(("v_new", *check_precision(v_new, gv_new, "float16")))
    if ht is not None:
        res.append(("ht", *check_precision(ht, ght, "float32")))
    return res


def _run_l1_bwd(name, B, S, H, DK, DV, seed, uscale, kwargs):
    gen = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, S, H, DK, generator=gen), dim=-1, p=2).to(torch.bfloat16)
    k = F.normalize(torch.randn(B, S, H, DK, generator=gen), dim=-1, p=2).to(torch.bfloat16)
    w = F.normalize(torch.randn(B, S, H, DK, generator=gen), dim=-1, p=2).to(torch.bfloat16)
    do = (torch.randn(B, S, H, DV, generator=gen) * uscale).to(torch.bfloat16)
    dv = (torch.randn(B, S, H, DV, generator=gen) * uscale).to(torch.bfloat16)
    use_exp2 = kwargs.get("use_exp2", False)
    g, gk = _make_g_gk(B, S, H, DK, gen, use_exp2)
    dht = torch.randn(B, H, DK, DV, generator=gen, dtype=torch.float32) if kwargs.get("dht") else None
    h0 = torch.randn(B, H, DK, DV, generator=gen, dtype=torch.float32) if kwargs.get("h0") else None
    dh, dh0, dv2 = fla_chunk_delta_bwd(
        q.to(DEVICE),
        k.to(DEVICE),
        w.to(DEVICE),
        do.to(DEVICE),
        dv.to(DEVICE),
        g.to(DEVICE),
        gk.to(DEVICE),
        h0=h0.to(DEVICE) if h0 is not None else None,
        dht=dht.to(DEVICE) if dht is not None else None,
        scale=kwargs.get("scale"),
        use_exp2=use_exp2,
    )
    torch.npu.synchronize()
    gdh, gdh0, gdv2 = fla_chunk_delta_bwd_ref(
        q,
        k,
        w,
        do,
        dv,
        g,
        gk,
        h0,
        dht,
        scale=kwargs.get("scale"),
        use_exp2=use_exp2,
    )
    res = [
        ("dh", *check_precision(dh, gdh, "bfloat16")),
        ("dv2", *check_precision(dv2, gdv2, "bfloat16")),
    ]
    if dh0 is not None:
        res.append(("dh0", *check_precision(dh0, gdh0, "float32")))
    return res


def test_fla_chunk_delta_l1():
    """L1 functional: in-domain shape/param/value-range coverage (task-tail
    guard, exp2/scale/nosave params, value ranges). Blocking."""
    ok = True
    seed = 2100
    for name, kind, shape, uscale, kwargs, _tags in L1_CASES:
        B, S, H, DK, DV = shape
        seed += 1
        try:
            if kind == "fwd":
                res = _run_l1_fwd(name, B, S, H, DK, DV, seed, uscale, dict(kwargs))
            else:
                res = _run_l1_bwd(name, B, S, H, DK, DV, seed, uscale, dict(kwargs))
            tag = "PRECISION_PASS" if all(r[1] for r in res) else "PRECISION_FAIL"
            detail = " ".join(f"{n}:ratio={r:.4f},max_abs={m:.3e}" for n, _, r, m in res)
            print(f"[{tag}] {name} shape={shape} :: {detail}")
            ok &= tag == "PRECISION_PASS"
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"[PRECISION_FAIL] {name}: {e}")
            ok = False
    return ok


# ---------- L2: negative tests (invalid inputs must be rejected) ----------
L2_CASES = [
    # Non-aligned shapes are OUTSIDE the supported domain (S%64==0, DK in
    # {64,128}, DV%64==0; wrapper-asserted, DESIGN SS1.6/SS5.4). Their
    # scenarios (tail-1 / tail-mid / prime remainders) are covered through
    # these rejection cases -- the D-SHAPE-TAIL-*/PRIME tags live here.
    ("l2_fwd_s_tail1", "fwd", (1, 129, 2, 64, 64), {}, ["D-EXC-SHAPE", "D-SHAPE-TAIL-1"]),
    ("l2_fwd_s_tailmid", "fwd", (1, 96, 2, 64, 64), {}, ["D-EXC-SHAPE", "D-SHAPE-TAIL-MID"]),
    ("l2_fwd_s_prime", "fwd", (1, 127, 2, 64, 64), {}, ["D-EXC-SHAPE", "D-SHAPE-PRIME"]),
    ("l2_fwd_dk256", "fwd", (1, 128, 2, 256, 64), {}, ["D-EXC-SHAPE"]),
    ("l2_fwd_dk96", "fwd", (1, 128, 2, 96, 64), {}, ["D-EXC-SHAPE"]),
    ("l2_fwd_dv100", "fwd", (1, 128, 2, 64, 100), {}, ["D-EXC-SHAPE"]),
    ("l2_fwd_chunk32", "fwd", (1, 128, 2, 64, 64), {"chunk_size": 32}, ["D-EXC-SHAPE", "D-PARAM-chunk_size"]),
    ("l2_bwd_s_tail1", "bwd", (1, 129, 2, 64, 64), {}, ["D-EXC-SHAPE", "D-SHAPE-TAIL-1"]),
    ("l2_bwd_dtype_int", "bwd", None, {"int_dtype": True}, ["D-EXC-DTYPE"]),
    ("l2_fwd_dtype_fp32", "fwd", None, {"fp32_dtype": True}, ["D-EXC-DTYPE"]),
]


def _l2_call(name, kind, shape, kwargs):
    """Build minimal (invalid) inputs and call the wrapper; expect rejection."""
    if shape is None:  # dtype cases: smallest legal shape, wrong dtype
        shape = (1, 128, 2, 64, 64)
    B, S, H, DK, DV = shape
    gen = torch.Generator().manual_seed(2201)
    if kind == "fwd":
        dt = torch.float32 if kwargs.get("fp32_dtype") else torch.float16
        k = torch.randn(B, S, H, DK, generator=gen).to(dt)
        w = torch.randn(B, S, H, DK, generator=gen).to(dt)
        u = torch.randn(B, S, H, DV, generator=gen).to(dt)
        g, gk = _make_g_gk(B, S, H, DK, gen, False)
        fla_chunk_delta_fwd(
            k.to(DEVICE),
            w.to(DEVICE),
            u.to(DEVICE),
            g.to(DEVICE),
            gk.to(DEVICE),
            chunk_size=kwargs.get("chunk_size", 64),
        )
    else:
        dt = torch.int32 if kwargs.get("int_dtype") else torch.bfloat16
        q = torch.randn(B, S, H, DK, generator=gen).to(torch.bfloat16)
        if kwargs.get("int_dtype"):
            q = q.to(torch.int32)
        k = torch.randn(B, S, H, DK, generator=gen).to(torch.bfloat16)
        w = torch.randn(B, S, H, DK, generator=gen).to(torch.bfloat16)
        do = torch.randn(B, S, H, DV, generator=gen).to(torch.bfloat16)
        dv = torch.randn(B, S, H, DV, generator=gen).to(torch.bfloat16)
        g, gk = _make_g_gk(B, S, H, DK, gen, False)
        fla_chunk_delta_bwd(
            q.to(DEVICE),
            k.to(DEVICE),
            w.to(DEVICE),
            do.to(DEVICE),
            dv.to(DEVICE),
            g.to(DEVICE),
            gk.to(DEVICE),
        )


def test_fla_chunk_delta_l2():
    """L2 negative: out-of-domain shapes/dtypes/chunk_size must be rejected
    (wrapper asserts / adapter dtype checks). Non-blocking."""
    for name, kind, shape, kwargs, _tags in L2_CASES:
        try:
            _l2_call(name, kind, shape, kwargs)
            print(f"[BOUNDARY_WARN] {name}: invalid input silently accepted")
        except Exception:  # noqa: BLE001
            print(f"[BOUNDARY_PASS] {name}: correctly rejected")


# ---------- Boundary: legal special values (non-blocking precision) ----------
def _bnd_inputs(B, S, H, DK, DV, seed, mode, uscale=1.0):
    gen = torch.Generator().manual_seed(seed)
    k = F.normalize(torch.randn(B, S, H, DK, generator=gen), dim=-1, p=2).to(torch.float16)
    w = F.normalize(torch.randn(B, S, H, DK, generator=gen), dim=-1, p=2).to(torch.float16)
    if mode == "zero":
        u = torch.zeros(B, S, H, DV).to(torch.float16)
        k = torch.zeros_like(k)
        w = torch.zeros_like(w)
    elif mode == "inf":
        u = (torch.randn(B, S, H, DV, generator=gen) * uscale).to(torch.float16)
        u.view(-1)[0] = float("inf")
    elif mode == "nan":
        u = (torch.randn(B, S, H, DV, generator=gen) * uscale).to(torch.float16)
        u.view(-1)[0] = float("nan")
    else:  # dbound / g-underflow / plain
        u = (torch.randn(B, S, H, DV, generator=gen) * uscale).to(torch.float16)
    g, gk = _make_g_gk(B, S, H, DK, gen, False)
    if mode == "dbound":
        # g_last near the fp32 exp underflow boundary: exp(-87) ~ 1.6e-38
        raw = torch.full((B, S, H), -87.0 / 64.0, dtype=torch.float32)
        g = raw.reshape(B, S // 64, 64, H).cumsum(2).reshape(B, S, H)
    return k, w, u, g, gk


def test_fla_chunk_delta_boundary():
    """Boundary: zero inputs / exp-underflow g / sparse inf / sparse nan /
    dtype-boundary values -- legal inputs, precision-compared, WARN-only."""
    cases = [
        ("bnd_zero_inputs", "zero", 1.0, ["D-SPECIAL-ZERO"]),
        ("bnd_g_exp_underflow", "dbound", 1.0, ["D-SPECIAL-DBOUND"]),
        ("bnd_sparse_inf_u", "inf", 1.0, ["D-SPECIAL-INF"]),
        ("bnd_sparse_nan_u", "nan", 1.0, ["D-SPECIAL-NAN"]),
        ("bnd_dbound_values", "plain", 200.0, ["D-SPECIAL-DBOUND"]),
    ]
    B, S, H, DK, DV = 1, 64, 2, 64, 64
    for name, mode, uscale, _tags in cases:
        try:
            k, w, u, g, gk = _bnd_inputs(B, S, H, DK, DV, 2301, mode, uscale)
            h, v_new, _ht = fla_chunk_delta_fwd(k.to(DEVICE), w.to(DEVICE), u.to(DEVICE), g.to(DEVICE), gk.to(DEVICE))
            torch.npu.synchronize()
            gh, gv_new, _ght = fla_chunk_delta_fwd_ref(k, w, u, g, gk)
            res = [
                ("h", *check_precision(h, gh, "float16")),
                ("v_new", *check_precision(v_new, gv_new, "float16")),
            ]
            tag = "BOUNDARY_PASS" if all(r[1] for r in res) else "BOUNDARY_WARN"
            detail = " ".join(f"{n}:ratio={r:.4f},max_abs={m:.3e}" for n, _, r, m in res)
            print(f"[{tag}] {name} :: {detail}")
        except Exception as e:  # noqa: BLE001
            print(f"[BOUNDARY_WARN] {name}: exception {e}")


COVERAGE_MANIFEST = {
    "D-DTYPE-fp16": 5,
    "D-DTYPE-bf16": 3,
    "D-DTYPE-fp32": 2,
    "D-SHAPE-ALIGNED": 3,
    "D-SHAPE-EDGE": 1,
    "D-SHAPE-TAIL-1": 2,
    "D-SHAPE-TAIL-MID": 1,
    "D-SHAPE-PRIME": 1,
    "D-VALRANGE-S": 1,
    "D-VALRANGE-M": 1,
    "D-VALRANGE-L": 2,
    "D-VALRANGE-ASYM": 1,
    "D-PARAM-chunk_size": 1,
    "D-PARAM-use_exp2": 2,
    "D-PARAM-scale": 1,
    "D-PARAM-save_new_value": 1,
    "D-PARAM-output_final_state": 2,
    "D-SPECIAL-INF": 1,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-DBOUND": 2,
    "D-EXC-DTYPE": 2,
    "D-EXC-SHAPE": 8,
}


# ============================================================================
# Perf sentinel (DESIGN SS9.4): do_bench, warmup=20 rep=10 mean, device-side
# NPU events only (no tensor construction / H2D / JIT in the timed region)
# ============================================================================
def do_bench(fn, warmup=20, rep=10):
    """Device-side timing aligned with ref_gpu/test_utils_kda.py."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        fn()
        ends[i].record()
    torch.npu.synchronize()
    times = [starts[i].elapsed_time(ends[i]) for i in range(rep)]
    return sum(times) / len(times) * 1000.0  # us


def run_perf_sentinel():
    """DESIGN SS9.4: bench shapes, wrapper defaults (no h0/ht,
    save_new_value=True, use_exp2=False, scale=DK**-0.5)."""
    B, S, H, DK, DV = 1, 8192, 64, 128, 128
    gen = torch.Generator().manual_seed(9001)
    k = torch.randn(B, S, H, DK, generator=gen).to(torch.float16).to(DEVICE)
    w = torch.randn(B, S, H, DK, generator=gen).to(torch.float16).to(DEVICE)
    u = torch.randn(B, S, H, DV, generator=gen).to(torch.float16).to(DEVICE)
    g, gk = _make_g_gk(B, S, H, DK, gen, False)
    g, gk = g.to(DEVICE), gk.to(DEVICE)

    fwd_us = do_bench(lambda: fla_chunk_delta_fwd(k, w, u, g, gk))
    print(f"[PERF] perf_fwd_bench shape={(B, S, H, DK, DV)} fwd={fwd_us:.1f} us")

    q = torch.randn(B, S, H, DK, generator=gen).to(torch.bfloat16).to(DEVICE)
    kb = torch.randn(B, S, H, DK, generator=gen).to(torch.bfloat16).to(DEVICE)
    wb = torch.randn(B, S, H, DK, generator=gen).to(torch.bfloat16).to(DEVICE)
    do = torch.randn(B, S, H, DV, generator=gen).to(torch.bfloat16).to(DEVICE)
    dv = torch.randn(B, S, H, DV, generator=gen).to(torch.bfloat16).to(DEVICE)
    gb, gkb = _make_g_gk(B, S, H, DK, gen, False)
    gb, gkb = gb.to(DEVICE), gkb.to(DEVICE)

    bwd_us = do_bench(lambda: fla_chunk_delta_bwd(q, kb, wb, do, dv, gb, gkb))
    print(f"[PERF] perf_bwd_bench shape={(B, S, H, DK, DV)} bwd={bwd_us:.1f} us")
    print(f"[PERF] fwd+bwd total {fwd_us + bwd_us:.1f} us (Stage-3 target < 3400)")
    return fwd_us, bwd_us


# ============================================================================
# main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    parser.add_argument("--bench", action="store_true", help="run the DESIGN SS9.4 perf sentinel")
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    blocking_ok = True
    if args.level in ("l0", "all"):
        blocking_ok &= test_fla_chunk_delta_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_fla_chunk_delta_l1()
    if args.level in ("l2", "all"):
        test_fla_chunk_delta_l2()
    if args.level in ("boundary", "all"):
        test_fla_chunk_delta_boundary()

    if args.bench:
        run_perf_sentinel()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
