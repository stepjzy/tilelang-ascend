"""sparse_mla_topk_reducesum — DSA 稀疏注意力 topk 概率质量 head 归约（Ascend Expert 模式）。

varlen packed 布局下，对每个 token 行 s、每个 kv_group g、每个 topk 选中位置
idx = Indices[s_row, g, j]，计算 score = exp((Q·KV_main + Q_tail·KV_tail)*sm_scale - Lse)
（causal mask 内置：idx <= s 且 idx != -1，否则 score=0），沿 head 维求和得到
ReduceSum[s, g, r, j]（未做 REPLICATE_H 归并与归一化——接口层归一化属本算子交付边界之外，
见 DESIGN_generic_p1.md §1.2）。

编程模式：Expert（用户指定，DESIGN_generic_p1 §2）
  - 显式层级：T.alloc_L1 / T.alloc_L0C / T.alloc_ub + GM workspace 中转（ws_kv / ws_acc）
  - 显式双流：T.Scope("C")（单段 gemm_v0）+ T.Scope("V")（mask/exp/reduce）
  - 手动同步：T.barrier_all() + T.set_cross_flag / T.wait_cross_flag（mode=2 quorum）
  - pass_configs 不设置（全手动，与 examples/sparse_flash_attention/
    example_sparse_flash_attn_dynamic_shape.py 同构）

Cross-flag 协议（扁平块流 n=task*NI+t 上严格 1:1 配对；V 侧全部 set/wait 位于 ntask/
`gn<nblk` 有效块程序段内、C 侧循环界即 ntask——防悬挂，HISA 头注释教训，DESIGN.md §7.2）：
  F0 = 0..DEPTH-1   V→C  set@MTE3：ws_kv[cid] slot 块 gather 完毕（mode-2：两 AIV 均写毕才放行 C）
  F1 = DEPTH..2*DEPTH-1  C→V  set@FIX ：ws_acc[cid] slot 块 score 就绪（L0C→GM 已落地）

实现要点（与 DESIGN_generic_p1.md §3/§4 一致；差异处已注明）：
  1. 行寻址恒等式：s_row = bos + s_i ≡ s_row_i（packed 布局保证，DESIGN §1.4）。
      所有 Q/Lse/Indices/ReduceSum 行索引直接用 s_row_i（纯 pid 算术），GM 标量读
      （TokenIndices/Offsets）仅保留在 V 流必需处：gather 基址 bos、causal mask 标量 s_i。
      （先例：unsorted_segment_sum seg_reduce 的 GM 标量寻址、SFA dynamic 的 pid 分解。）
  2. gather 寻址保护：KV 行号 clamp 到 [0, seq_len_kv - 1]（Indices 含 -1/越界时读合法
      内存，其贡献由 causal mask 置 0，与 CUDA 原版语义一致——原版靠 mask 丢弃，
      DESIGN §5.4「必要时 clamp 仅保护寻址」，前移落地以保障 L1/L2 内存安全）。
  3. ws_kv 采用 D+D_tail 合并布局 [grid_size, BI, DDT]（DESIGN §8.2，DMA 减半）；
      C 流列切片（最内维、行方向 gap）进两块 L1。若该路径失败，回退 SFA 双 ws 分立
      布局（DESIGN §10.5 风险项 #3/#4 预案）。
  4. mask 算术链（DESIGN §3.1 注）：m1 = clamp((s_i+1) - idx, 0, 1) = 1{idx <= s}，
      m2 = clamp(idx + 1, 0, 1) = 1{idx >= 0}，mask = m1 * m2，broadcast 到 [H, HALF]。
      `sub` 不支持 scalar 左操作数（ascend_tile.py L1121-1124），故 (s+1)-idx 用
      mul(-1) + add(scalar) 实现；s_i 为 GM 标量，经 T.float32(s_i + 1) 组合为
      PrimExpr scalar（binary_op ascend_adds 路径；先例：HISA T.float32(gm_scalar)）。

通用路径（DESIGN_generic_p1.md §1.6/§8.6，移除后基线形态）：panel/auto/shared 三模式
分发、共享索引检测 kernel、双 guarded 推测发射与 panel_mode 参数已全部移除——本模块
回归纯通用接口，支持任意合法 topk_indices（值域 [-1, ΣS_kv) 任意分布：逐行独立 /
部分共享 / 全同 / 非连续 / 乱序，单一代码路径，无索引内容假设）。历史 panel/auto
实现见 git 基线 0f11b3cc 与 DESIGN_panel_mode.md；traced PrimFunc 与基线 unguarded
变体（= rev1 通用路径）TIR 等价（§8.6 保全验证）。P1（AIC GEMM 流水重构）已实施→
性能证伪→回退（见 git 00a93f81/6db367b0/25c0da76 与 perf_tuning/p1_ablation_report.md）。

P2（SIDE_PAIR 移除，2026-10-11 本版交付）：V 侧 gather 段行读前的相邻 mte3→mte2
保护对移除（仅存在于 SIDES=LEAD 的 UB 降级档，如 BI=128@DDT=576 / BI=64@DDT=1024），
改依自然排空窗口覆盖 side 槽复用竞态（稳态余量 ≥4.7×，推导与实证见 SIDES 定义处）；
SIDES ≥ LEAD+1 档原本即无保护对，不受影响。依据：Stage 3 联合瓶颈分析 spike
（官方口径配对 -1.53% + 精度 5/5 torch.equal + msprof V overlap 1.031→1.102、
C ws 读率 55.3→59.8GB/s）+ 本轮 4 臂消融（正确性全绿 + 多实例配对 -1.56% 复现 +
实例方差 29.6→7.8µs 收窄）。同轮 P2b（NI=1 档 LEAD=3 前移）实测 +3.23% 确证回归
（更深前瞻 = 更少 flag 仲裁 = 趋向自由跑端，与联合分析"改进空间只在更平滑的仲裁"
结论自洽）→ 按用户回退规则不采纳。附带修复：idx(LEAD) 预发射组号 g_l 误用行内号
b_s_l（恒 0）的潜在缺陷（kv_group>1 + NI=1 + 短流窗口取错组，实证复现并修复，
L0-G4 grouped_short_gl_window 回归用例锁定）。

P3（序列级去重驻留，2026-10-11 本版交付，DESIGN_dataflow.md §5）：数据流重构——
每 (seq, kv_group) 构造"被引行并集 panel"（M=topk 编译期定宽，组内一次 sort+去重
构造），panel 经 ws_panel 单次 GM 中转驻留每核 L1（staging/8-flag 协议/组调度
复用 0f11b3cc Plan-A 机制），C 侧转置 gemm（panel×Qᵀ → accT [topk, H]，spike-1
实证与非转置方向逐位一致），V 侧 reduce-first（rs_all = Σ_h exp(accT·s − lse)，
mask 从 h 求和提出的数学恒等）+ 等值比较矩阵选值（整数域 eq = 1−clamp(|rows−idx|,
0,1)，组内去重后恰一匹配）——spike-2 裁决：gatherb 在 910B3 仅支持 uint16/uint32
（fp32 静默无效），元素 gather 需 uint32 计算偏移（vconv cast 不支持）⇒ 采用设计
§8 R1 备选链的 UB 内精化形态（无 gather/无 GM 标量读）。通用性：panel kernel
正确域 = {P_seq ≤ topk}（尺寸谓词，非内容契约）；day-1 资格 = prep kernel 设备侧
共享检测 + 值域门（|idx| ≤ 2^24，fp32 排序精确域）→ Violations 标量 → panel/
generic 推测双发射（互补入口守卫，0f11b3cc §8.2 已验证形态）——资格判错只损失
性能不损失正确性；API 签名/proto 零变化。ws 流量：302MB×2 双腿 → ~12MB
（每核每组一次 staging），全 kernel 995MB → ~110MB。

支持域（编译期 assert，DESIGN §11.4 [DISPATCH-COVERAGE]）：
  dtype 固定组合 Q/KV=bfloat16、Lse/ReduceSum=float32、Indices/Offsets/TokenIndices=int32；
  dim/tail_dim 为 2 的幂；topk % block_I == 0；block_I 为偶数且 >= 32；
  head_kv = heads // kv_group 满足 (16 <= head_kv <= 64 且为 2 的幂) 或 (head_kv % 64 == 0)；
  padded_H != head_kv 时仅支持 kv_group == 1（继承原版 assert）。
  ΣS / ΣS_kv / batch 运行时任意（T.symbolic 动态维）。
"""

import tilelang
from tilelang import language as T
from tvm import tir
from tvm.tir import PrimFunc
from tvm.tir.stmt_functor import post_order_visit

__all__ = [
    "tl_sparse_mla_topk_reducesum",
    "tl_sparse_mla_topk_reducesum_guarded",
    "tl_sparse_mla_topk_reducesum_panel_guarded",
    "tl_sparse_mla_topk_reducesum_prep",
    "sparse_mla_topk_reducesum",
    "prepare_token_indices",
    "LAST_DISPATCH",
]

# 测试路径断言通道（DESIGN_dataflow §7.2）：host 每次分发记录 Violations 张量引用
# 与所选路径（"panel"/"generic"）——仅测试在 eager 调用后读取（热路径零 D2H）。
LAST_DISPATCH = {"violations": None, "path": None}


def _strip_unused_param(func: PrimFunc, drop_idx: int) -> PrimFunc:
    """摘除 prim_func 中未被引用的 buffer 形参（0f11b3cc §3.5 机制复用）。

    共享 body 构造器以 guarded 形态签名 trace（Violations 在参数表中）；
    guarded=False 的 traced body 不含 Violations 引用（守卫经 trace 期纯 Python
    闭包直通，零 TIR 语句），据此在 trace 后摘除该形参即精确复原 unguarded
    签名（参数序/数量/buffer_map 与公开形态逐项一致）。
    """
    params = list(func.params)
    drop_var = params[drop_idx]
    assert drop_var in func.buffer_map, f"param {drop_idx} is not a buffer param"
    drop_buf = func.buffer_map[drop_var]

    refs = []

    def _check(expr):
        if isinstance(expr, (tir.BufferLoad, tir.BufferStore)) and expr.buffer.same_as(drop_buf):
            refs.append(expr)

    post_order_visit(func.body, _check)
    assert not refs, f"dropped param still has {len(refs)} refs in body"

    new_params = params[:drop_idx] + params[drop_idx + 1:]
    new_buffer_map = {v: b for v, b in func.buffer_map.items() if v != drop_var}
    return PrimFunc(new_params, func.body, func.ret_type, new_buffer_map, func.attrs, func.span)


# ============================================================================
# Kernel builder（DESIGN_generic_p1 §3/§4/§6/§8——移除后唯一 builder）
# ============================================================================
def _reducesum_generic_body(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group,
    sm_scale,
    block_I,
    grid_size,
    guarded,
):
    """通用路径 kernel body 构造器（DESIGN_dataflow §5.0——P3 双发射回退臂）。

    guarded=False：traced 语句与 P2 基线（22bc68a8）逐条一致（守卫经 trace 期纯
    Python 闭包直通，零 TIR 语句；Violations 形参不被引用，由薄包装
    _strip_unused_param 摘除）——公开 builder 行为不变。
    guarded=True：入口 if 守卫 Violations[0] > 0（资格判定不通过才执行 generic，
    0f11b3cc §8.2 已验证形态）；分支不进入 = C 循环零次 + V 整体不进入，零 flag
    op、零 cross-flag 触碰（§7.5 洁净性论证逐字适用——ntask==0 空闲核同理论证）。
    """
    # ---- 编译期 assert（DESIGN §4.7 / §10.1 支持域）----
    assert dim == tilelang.math.next_power_of_2(dim), f"dim must be a power of 2, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"tail_dim must be a power of 2, tail_dim={tail_dim}"
    assert topk % block_I == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    assert block_I % 2 == 0 and block_I >= 32, f"block_I must be even and >= 32 (AIV half-split), got {block_I}"
    assert kv_group >= 1 and heads % kv_group == 0, f"heads={heads} must be divisible by kv_group={kv_group}"
    head_kv = heads // kv_group
    assert (16 <= head_kv <= 64 and head_kv == tilelang.math.next_power_of_2(head_kv)) or head_kv % 64 == 0, (
        f"head_kv={head_kv} out of supported domain: need (16<=head_kv<=64 and power of 2) or head_kv % 64 == 0"
    )
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

    # ---- symbolic 动态维（一次编译覆盖任意 batch/seq_len）----
    seq_len = T.symbolic("seq_len")  # ΣS
    seq_len_kv = T.symbolic("seq_len_kv")  # ΣS_kv
    batch_plus_one = T.symbolic("batch_plus_one")  # batch+1（Offsets 维）

    dtype = "bfloat16"
    indices_dtype = "int32"
    accum_dtype = "float32"

    # ---- 派生编译期常量 ----
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != head_kv:
        assert kv_group == 1, (
            "H padding path only supports kv_group == 1 (inherited from the CUDA original: "
            "otherwise Q/Output head slices need explicit masks)"
        )
    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1
    H_per_block = padded_H if REPLICATE_H == 1 else 64

    BI = block_I
    HALF = BI // 2  # AIV 列半区（topk 方向按 2 AIV 拆分）
    NI = topk // BI  # 每 token 的 topk 块数（编译期常量）
    DDT = dim + tail_dim
    # head 分片偏移：REPLICATE_H == 1 → 0；否则每分片 64 head（DESIGN §5.2）
    R_HEAD_OFFSET = 0 if REPLICATE_H == 1 else 64

    # Cross flag 编号（DESIGN §7.2；DEPTH 槽 ping-pong：slot s → F0 id s%DEPTH、
    # F1 id DEPTH+s%DEPTH；A2 FFTS flagId 位宽 4bit=16 个（GetffstMsg flagId&0xf），
    # 8 个 id 合法。msprof 实证：DEPTH=2 时 AIC 74% 时间 CrossCoreWaitFlag(F0)、
    # AIV 39% 等 F1——握手互锁而非算力/带宽瓶颈 → 加深预取消除 interlock）
    DEPTH = 4  # ws_kv/ws_acc slot 数 = V 预取深度+1（BI=32/NI=4 时满预取）
    F0_READY_BASE = 0  # V→C：ws_kv slot gather 完毕（MTE3，mode-2 quorum）id 0..DEPTH-1
    F1_ACC_BASE = DEPTH  # C→V：ws_acc slot score 就绪（FIX）id DEPTH..2*DEPTH-1
    # 跨 task 前瞻距离（编译期常量，iter5/#14 + iter8/#18 窗口扩展）：
    #   NI ≥ 2：LEAD ≤ NI ⇒ τg ∈ {k, k+1}（两组 task 标量即可，同 #14）；
    #   NI == 1（BI==topk，每 task 单块）：LEAD=2，τg = k+2 ⇒ 前瞻窗口标量组
    #   偏移 OFFSET_N = (NI-1+LEAD)//NI（NI≥2 → 1；NI=1 → 2），pid_n = pid +
    #   OFFSET_N*grid。动机（msprof@BI128 452.7us）：LEAD=1 时 C/V 逐块锁定步进
    #   （AIC waitF0 172us + gemm_v0 尾部同步致 mte1_stall 256us），双侧引擎
    #   （AIV 行读 26.1GB/s×40、AIC ws→L1 ~50GB/s×20）各仅 ~65% busy——解耦后
    #   两侧可同时逼近引擎下限（~300us 侧 → 系统 ~330-370us）。
    LEAD = min(DEPTH - 1, NI) if NI > 1 else min(DEPTH - 1, 2)
    OFFSET_N = (NI - 1 + LEAD) // NI  # 前瞻窗口 task 偏移（编译期常量；NI≥2 ⇒ 1）
    # kv_gather_ub 缓冲深度（iter6/#15v2 + iter8/#18 UB 适配 + P2 自然排空语义）：
    # side = gn % SIDES 轮转。side 复用（行读 MTE2 写 vs 上一同 side bulk MTE3 读）的
    # 安全性分两档：
    #   SIDES ≥ LEAD+1（UB 充裕档，如 BI=32/BI=64@DDT=576）：bulk(gn-SIDES) 的排空由
    #     post(gn-SIDES) 的 waitF1(gn-SIDES) 架构性保证——body(gn-SIDES) 程序序先于
    #     body(gn-LEAD)，且 F1(m) ⇒ C 已消费 F0(m) ⇒ 两 AIV 的 bulk(m) 已排空；
    #   SIDES = LEAD（BI=128@DDT=576 / BI=64@DDT=1024 等 UB 降级档）：上述架构链失效
    #     （body(gn-SIDES) 与 gather 同 body、post 段在其后）——P2（2026-10-11，本版）
    #     移除原 SIDE_PAIR 相邻 mte3→mte2 保护对，改依自然排空窗口：
    #     ① 稳态：bulk(gn-SIDES) 发射于 body(gn-SIDES-LEAD)，行读发射于 body(gn-LEAD)，
    #        两发射点间隔恒为 2 个 body，各含一个 post 段 waitF1 标量停（C 节拍
    #        ~3.3µs）⇒ 标量侧 ≥6.6µs；MTE2 队列前方还压着上一 body 的整段行读
    #        （73.7KB@~26GB/s ≈ 2.8µs）⇒ bulk（≤73.7KB@~36-61GB/s ≈ 1.2-2.0µs 排空）
    #        的可用排空窗口 ≥ 6.6+2.8µs，余量 ≥4.7×（Stage 3 spike 实测 5×）；
    #     ② MTE3 无持续积压：bulk 发射节奏被 waitF1 节流至 ~3.3µs/条 > 1.5µs 排空
    #        ⇒ MTE3 占用 ≤~45%，瞬态积压 ≤1 条；
    #     ③ 启动窗（首个主循环 body / prologue 相邻段）：间隔降至一整段行读
    #        ~2.8µs，余量 ~1.9×（与移除前 SIDE_PAIR 对的实测等待窗同级）。
    #     实证：Stage 3 spike（joint_bottleneck_analysis §2）精度 5/5 torch.equal +
    #     本轮 4 臂消融正确性全绿（含 G5 竞态族 ×3 run 位稳）+ 官方口径多实例配对
    #     -1.56%（与 spike -1.53% 复现）；机制 = V 锯齿发射流平滑（msprof：V overlap
    #     1.031→1.102、C ws 读率 55.3→59.8GB/s），且实例间方差 29.6→7.8µs 显著收窄。
    # 【实证教训 2026-10-07，保留】曾以"错位 marker"（SetFlag 于 bulk 后、WaitFlag 于
    # 2 body 后，事件槽按 (源管道,eventId) 共享）实现同保护：SetFlag/WaitFlag 为
    # fire-before-arm 即 miss 语义（printf 插桩实证挂死）→ 严禁非相邻 set/wait 配对。
    _kv_gather_budget = 170 * 1024  # UB 192KB 减去其余 buffer/对齐余量
    SIDES = (LEAD + 1) if (LEAD + 1) * (block_I // 2) * (dim + tail_dim) * 2 <= _kv_gather_budget else 2
    # 【#18 编译期不变式】
    #  (i)  标量窗口覆盖：前瞻 gather 块 gn=k*NI+t+LEAD 的任务号偏移 (t+LEAD)//NI
    #       必须 ∈ {0, OFFSET_N}——body 仅持有 task k 与 k+OFFSET_N 两组标量；
    #  (ii) prologue side 首用互异：LEAD ≤ SIDES ⇒ side = s%SIDES = s（免保护）。
    assert all(((t + LEAD) // NI) in (0, OFFSET_N) for t in range(NI)), (
        f"#18 scalar-window invariant violated: LEAD={LEAD} NI={NI} OFFSET_N={OFFSET_N}"
    )
    assert LEAD <= SIDES, f"#18 prologue side-collision: LEAD={LEAD} > SIDES={SIDES}"
    # indices_ub 槽数（iter6/#12 idx 跨 body 预取）：块 m 的 idx 载入提前至 body(m-LEAD-1)
    # 的 gather 段发射（行读后立即发射、搭本段 mte2→mte3 drain 便车免费落地证明）；
    # 覆写点 slot m%IDXSLOTS 的最后 V 读者 Cast(m-IDXSLOTS) 位于 body(m-IDXSLOTS)，
    # IDXSLOTS ≥ LEAD+2 ⇒ 严格先于发射 body，v→mte2 相邻对排空即安全。
    IDXSLOTS = LEAD + 2

    def _guard_run(cond):
        # trace 期守卫选择（0f11b3cc §8.2 入口守卫机制复用）：guarded=False →
        # Python True（编译期分支直通，零 TIR 语句）；guarded=True → TIR 比较 →
        # IfThenElse 入口守卫。不得用 ntask Select/min 钳位形态（污染 serial 循环界
        # 引发 CanonicalSimplifier 爆炸，0f11b3cc 实证 13s→600s+）。
        if guarded:
            return cond
        return True

    def _guard_v_entry(ntask_pos, guard_cond):
        # V scope 入口条件复合：unguarded → ntask_pos 原样；guarded → And 复合。
        if guarded:
            return tir.all(ntask_pos, guard_cond)
        return ntask_pos

    q_shape = [seq_len, heads, DDT]
    kv_shape = [seq_len_kv, kv_group, DDT]
    indices_shape = [seq_len, kv_group, topk]
    lse_shape = [seq_len, heads]
    offsets_shape = [batch_plus_one]
    token_indices_shape = [seq_len, 2]
    reducesum_shape = [seq_len, kv_group, REPLICATE_H, topk]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore  # [ΣS, heads, DDT] bf16
        KV: T.Tensor(kv_shape, dtype),  # type: ignore  # [ΣS_kv, G, DDT] bf16
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore  # [ΣS, G, topk] int32
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore  # [ΣS, heads] fp32
        Offsets: T.Tensor(offsets_shape, indices_dtype),  # type: ignore  # [B+1] int32
        TokenIndices: T.Tensor(token_indices_shape, indices_dtype),  # type: ignore  # [ΣS, 2] int32
        Violations: T.Tensor([1], indices_dtype),  # type: ignore  # 守卫输入（unguarded trace 不引用，strip 复原）
        ReduceSum: T.Tensor(reducesum_shape, accum_dtype),  # type: ignore  # out
        ws_kv: T.Tensor([grid_size, DEPTH * BI, DDT], dtype),  # V→C gather 中转（每核 DEPTH 槽）
        ws_acc: T.Tensor([grid_size, DEPTH * H_per_block, BI], accum_dtype),  # C→V score 中转（DEPTH 槽）
    ):
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):
            # ---- Cube 侧：L1 / L0C（gemm_v0 内部管理 L0A/L0B）----
            # D+D_tail 合并布局（DDT=576，K 整除 16 满足 zN fractal；kL0Tail=64），
            # 单 gemm_v0 K=576 替代两段 gemm：fp32 累加顺序不变（K 顺序推进），
            # C 侧每 block 省一次 gemm 调用固定开销（内部 flag 序）与一次 L1 拷贝
            q_l1 = T.alloc_L1([H_per_block, DDT], dtype)
            kv_l1 = T.alloc_L1([BI, DDT], dtype)
            acc_s_l0c = T.alloc_L0C([H_per_block, BI], accum_dtype)

            # ---- Vector 侧：UB ----
            # indices 多 buffer（slot 在行维，Var 可寻址）：IDXSLOTS = LEAD+2 槽支撑
            # iter6/#12 跨 body 预取（发射点 body(m-LEAD-1) 晚于最后读者 Cast(m-IDXSLOTS)
            # 所在 body(m-IDXSLOTS)，覆写由发射点 v→mte2 相邻对保护）
            indices_ub = T.alloc_ub([IDXSLOTS, BI], indices_dtype)
            # 批量 gather staging 多缓冲（iter6/#15v2，guide §2.8）：[SIDES, HALF, DDT]，
            # side = gn % SIDES 轮转；行读(MTE2)与上一块 bulk 写出(MTE3)跨 body 并行。
            # side 重写保护（P2 自然排空语义，见 SIDES 定义处两档推导）：SIDES ≥ LEAD+1
            # 档由 post(gn-SIDES) 的 F1 wait 架构性保证；SIDES = LEAD 降级档由自然排空
            # 窗口覆盖（余量 ≥4.7×，Stage 3 spike + 本轮 4 臂消融实证）
            kv_gather_ub = T.alloc_ub([SIDES, HALF, DDT], dtype)
            # 向量化预 clamp 的 KV 行地址（int32）：每 gather 段一次 Adds+Clamp，
            # 行循环仅剩 1×GetValue + DataCopyPad（消除 16× 串行 clamp 标量链）
            rows_ub = T.alloc_ub([BI], indices_dtype)
            lse_ub = T.alloc_ub([H_per_block], accum_dtype)
            lse2d_ub = T.alloc_ub([H_per_block, HALF], accum_dtype)  # 每 task 广播一次
            idx_f_ub = T.alloc_ub([HALF], accum_dtype)
            t_ub = T.alloc_ub([HALF], accum_dtype)
            m1_ub = T.alloc_ub([HALF], accum_dtype)
            m2_ub = T.alloc_ub([HALF], accum_dtype)
            mask_ub = T.alloc_ub([HALF], accum_dtype)
            mask2d_ub = T.alloc_ub([H_per_block, HALF], accum_dtype)
            acc_ub = T.alloc_ub([H_per_block, HALF], accum_dtype)
            rs_ub = T.alloc_ub([HALF], accum_dtype)

            # ---- persistent grid：扁平块流 n = task*NI + t（跨 task 旋转流水，iter5/#14）----
            # 任务数按核推导 ntask = #{k: k*grid+cid < total}；C/V 两 scope 以同一扁平块序
            # 推进 ⇒ F0/F1 set/wait 计数恒为 nblk = ntask*NI 且 1:1（全部 set/wait 位于有效
            # 块程序段内——guard 由循环界 ntask 承担，天然满足防悬挂要求，DESIGN §7.2）。
            # V 流 body(k,t) = [前瞻 g(k*NI+t+LEAD)（可跨入 task k+1）; p(k*NI+t)]：
            #   下一 task 的首批 gather 在本 task 中段即发射，消除旧结构 task 边界的
            #   p(NI-1)→waitF1→下 task prologue 双侧 drain/refill 串行气泡
            #   （msprof@711.8us：AIV waitF1 181.8us/26%、AIC waitF0 454.5us/65%）。
            total_tasks = seq_len * REPLICATE_H * kv_group
            ntask = T.max(0, T.ceildiv(total_tasks - cid, grid_size))  # 本核有效任务数
            nblk = ntask * NI  # 本核扁平块总数

            with T.Scope("C"):
                # P3 双发射守卫（DESIGN_dataflow §5.0）：Violations[0] > 0（资格判定
                # 不通过）才执行 generic；== 0 时让位 panel_guarded（早退 = 循环零次，
                # 零 flag op——0f11b3cc §7.5 洁净性论证逐字适用）
                if _guard_run(Violations[0] > 0):
                    for k in T.serial(ntask):
                        pid = k * grid_size + cid
                        b_s_i = pid % (seq_len * REPLICATE_H)
                        g_i = pid // (seq_len * REPLICATE_H)
                        s_row_i = b_s_i // REPLICATE_H  # packed 全局行号（≡ bos + s_i）
                        r_i = b_s_i % REPLICATE_H  # REPLICATE 分片号
                        H0 = g_i * padded_H + r_i * R_HEAD_OFFSET
                        H1 = H0 + H_per_block
                        # Q 装载（每 token 一次；连续行切片 [H, DDT] 整行，无 gather）。
                        # 跨 task 的 q_l1/kv_l1 重写由上一 task 末 gemm_v0 尾部内建
                        # MTE1→MTE2 flag 排空保护（common.h L1306/L1373；程序相对序与
                        # task 嵌套旧结构一致）；本迭代 ws_kv GM→L1 与 MTE1 消费由
                        # gemm_v0 首部 MTE2→MTE1 flag 保护。
                        T.copy(Q[s_row_i, H0:H1, :], q_l1)
                        for i in T.serial(NI):
                            n = k * NI + i  # 扁平块号（与 V 侧 n/gn 同域）
                            slot = n % DEPTH  # 评估配置 DEPTH=NI ⇒ 与旧 i%DEPTH 逐值一致
                            T.wait_cross_flag(F0_READY_BASE + slot)
                            # ws_kv slot 行区 → L1 整块（BI×DDT 连续，单条拷贝）
                            T.copy(ws_kv[cid, slot * BI : (slot + 1) * BI, :], kv_l1)
                            # 单段 GEMM K=576（fp32 累加进 L0C；K 顺序与原两段一致）。
                            # kL0Size 保持默认 128（实测 kL0Size=576 单 k-step 反而
                            # +2.9%：AIC scalar_mte1_stall 上升，见 perf_log iter3）
                            T.gemm_v0(q_l1, kv_l1, acc_s_l0c, transpose_B=True, init=True)
                            # L0C → GM（FIX）；gemm_v0 尾部 M→FIX flag 已保证 Mmad 落地
                            T.copy(acc_s_l0c, ws_acc[cid, slot * H_per_block : (slot + 1) * H_per_block, :])
                            T.set_cross_flag("FIX", F1_ACC_BASE + slot)

            with T.Scope("V"):
                # GM 标量读仅在 V 流（C 流不消费 b_i/s_i/bos——行寻址用恒等式，
                # DESIGN §3.3 注 1）。前瞻 gather 块 gn = k*NI+t+LEAD 的所属 task
                # τg = k + (t+LEAD)//NI ∈ {k, k+OFFSET_N}（编译期 assert 保证；
                # iter8/#18：NI≥2 ⇒ OFFSET_N=1，NI=1/LEAD=2 ⇒ OFFSET_N=2），
                # body 头部两组标量（自身 + clamp 安全的 k+OFFSET_N 预载）即可覆盖，
                # 无需跨迭代 UB 环缓存。
                # P3 双发射守卫：条件复合 And(ntask>0, Violations[0]>0)（0f11b3cc 形态）。
                if _guard_v_entry(ntask > 0, Violations[0] > 0):
                    # ---- preamble：task0 自身标量 + lse2d(0)（ntask>0 ⇒ in-bounds）----
                    b_s_0 = cid % (seq_len * REPLICATE_H)
                    g_0 = cid // (seq_len * REPLICATE_H)
                    s_row_0 = b_s_0 // REPLICATE_H
                    r_0 = b_s_0 % REPLICATE_H
                    H0_0 = g_0 * padded_H + r_0 * R_HEAD_OFFSET
                    b_0 = TokenIndices[s_row_0, 0]
                    s_i_0 = TokenIndices[s_row_0, 1]
                    bos_0 = Offsets[b_0]
                    # lse2d 广播（V）读毕才可重写 lse_ub
                    T.set_flag("v", "mte2", 0)
                    T.wait_flag("v", "mte2", 0)
                    T.copy(Lse[s_row_0, H0_0 : H0_0 + H_per_block], lse_ub)
                    T.set_flag("mte2", "v", 0)
                    T.wait_flag("mte2", "v", 0)
                    T.tile.broadcast(lse2d_ub, lse_ub, axis=1)  # 每 task 一次

                    # ---- prologue：g(0..LEAD-1)（编译期展开，s 为 Python 常量）----
                    # NI ≥ 2：LEAD ≤ NI ⇒ τs = s//NI = 0，全部属 task0（标量取 preamble，
                    # 无 guard，结构同 iter5/#14）；
                    # NI == 1（iter8/#18，LEAD=2）：块 s≥1 属 task s ⇒ 独立分支：GM 标量读
                    # 全部经 clamp 合法坐标（task 不存在时为垃圾但内存安全、不被消费），
                    # bulk/F0 置于 `s < nblk` 存在性 guard 内整体跳过 ⇒ phantom 块不产生
                    # F0，维持 F0/F1 恒 nblk 1:1（协议不变式，DESIGN §7.2）。
                    # side = s % SIDES 首用互异免保护由 LEAD ≤ SIDES assert 保证。
                    # 分支体自包含（TVMScript IfFrame 变量作用域约束：分支内绑定不出帧）。
                    for s in range(LEAD):
                        tau_s = s // NI  # 块 s 所属任务号（编译期常量）
                        nslot = s % DEPTH
                        side = s % SIDES
                        islot = s % IDXSLOTS  # s < LEAD < IDXSLOTS ⇒ islot == s
                        if tau_s == 0:
                            # indices_ub[islot] 重写保护：更早 Cast(V) 读毕
                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)
                            # rows_ub 重写保护：上一 gather 段 GetValue(S) 读毕
                            T.set_flag("s", "v", 0)
                            T.wait_flag("s", "v", 0)
                            T.copy(Indices[s_row_0, g_0, (s % NI) * BI : (s % NI + 1) * BI],
                                   indices_ub[islot, :])
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            # 行地址向量化预计算（BI 元素 2 条 V 指令）：
                            # rows = clamp(idx + bos, 0, ΣS_kv-1)（mask 已保证 -1/越界
                            # 贡献为 0，clamp 仅保护寻址，DESIGN §5.4）
                            T.tile.add(rows_ub, indices_ub[islot, :], bos_0)
                            T.tile.clamp(rows_ub, rows_ub, 0, seq_len_kv - 1, BI)
                            T.set_flag("v", "s", 0)
                            T.wait_flag("v", "s", 0)
                            for j in range(HALF):  # HALF 个 MTE2 行读无同步排队流水发射
                                row_j = vid * HALF + j
                                kv_row = rows_ub[row_j]  # 每行仅 1× UB 标量读
                                T.copy(KV[kv_row, g_0, :], kv_gather_ub[side, j, :])
                            T.set_flag("mte2", "mte3", 0)
                            T.wait_flag("mte2", "mte3", 0)
                            # rs_ub 重写保护（iter6/#15v2 相邻对，位置关键）：置于行读排空后、
                            # 本块 bulk 发射前 ⇒ 仅捕获上一块 bulk(s-1)/rs store(s-2)（均已自然
                            # 排空 ⇒ wait ≈ 免费）；若放 post 头部会捕获本 body 刚发射的 bulk
                            # （MTE3 FIFO）而使 V 链串行等 bulk。reduce_sum(s…) 在本段之后
                            # 程序序执行 ⇒ rs_ub 保护成立
                            T.set_flag("mte3", "v", 0)
                            T.wait_flag("mte3", "v", 0)
                            # 一次性批量写出 ws_kv slot 行区（单条多 block MTE3）
                            T.copy(
                                kv_gather_ub[side, :, :],
                                ws_kv[cid,
                                      nslot * BI + vid * HALF :
                                      nslot * BI + (vid + 1) * HALF, :],
                            )
                            T.set_cross_flag("MTE3", F0_READY_BASE + nslot)
                        else:
                            # ---- NI==1, s ≥ 1：块 s 属 task s（可能不存在 ⇒ clamp 合法化）----
                            pid_s = tau_s * grid_size + cid
                            b_s_s = pid_s % (seq_len * REPLICATE_H)
                            s_row_s = T.min(b_s_s // REPLICATE_H, seq_len - 1)
                            g_s = T.min(pid_s // (seq_len * REPLICATE_H), kv_group - 1)
                            b_sp = TokenIndices[s_row_s, 0]
                            bos_s = Offsets[b_sp]
                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)
                            T.set_flag("s", "v", 0)
                            T.wait_flag("s", "v", 0)
                            T.copy(Indices[s_row_s, g_s, 0:BI], indices_ub[islot, :])
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            T.tile.add(rows_ub, indices_ub[islot, :], bos_s)
                            T.tile.clamp(rows_ub, rows_ub, 0, seq_len_kv - 1, BI)
                            T.set_flag("v", "s", 0)
                            T.wait_flag("v", "s", 0)
                            for j in range(HALF):
                                row_j = vid * HALF + j
                                kv_row = rows_ub[row_j]
                                T.copy(KV[kv_row, g_s, :], kv_gather_ub[side, j, :])
                            T.set_flag("mte2", "mte3", 0)
                            T.wait_flag("mte2", "mte3", 0)
                            T.set_flag("mte3", "v", 0)
                            T.wait_flag("mte3", "v", 0)
                            # 存在性 guard（tau_s>0 ⇒ NI==1 ⇒ s < nblk ⟺ task s 有效）：
                            # phantom 块行读已入 side（不被消费、后续 side 复用由 body
                            # SIDE_PAIR/架构保证覆盖），bulk/F0 跳过 ⇒ 计数仍 nblk 1:1
                            if s < nblk:
                                T.copy(
                                    kv_gather_ub[side, :, :],
                                    ws_kv[cid,
                                          nslot * BI + vid * HALF :
                                          nslot * BI + (vid + 1) * HALF, :],
                                )
                                T.set_cross_flag("MTE3", F0_READY_BASE + nslot)

                    # ---- idx(LEAD) 预发射（iter6/#12；主段第 0 个 gather 的 idx）----
                    # 块 LEAD 所属 task τ = LEAD//NI 为编译期常量（LEAD<NI ⇒ τ=0 用
                    # task0 标量；LEAD≥NI ⇒ τ=LEAD//NI（BI=64→1；BI=128/iter8/#18→2），
                    # guard 内 nblk>LEAD ⇒ ntask > LEAD//NI ⇒ pid_l 恒为有效任务行，
                    # 无需 clamp）。落地证明：相邻 mte2→v 对（每核一次，~300ns，可忽略）。
                    if LEAD < nblk:
                        if LEAD < NI:
                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)
                            T.copy(Indices[s_row_0, g_0, LEAD * BI : (LEAD + 1) * BI],
                                   indices_ub[LEAD % IDXSLOTS, :])
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                        else:
                            pid_l = (LEAD // NI) * grid_size + cid
                            b_s_l = pid_l % (seq_len * REPLICATE_H)
                            s_row_l = b_s_l // REPLICATE_H
                            # 组号取 pid_l（任务全局号）整除——历史误用 b_s_l（行内号，
                            # 恒 < seq_len*RH）使 g_l 恒 0：kv_group>1 且 task(LEAD//NI)
                            # 落在 g≠0 组时（短流 + NI=1 窗口，如 S=48/H=32/G=2/topk=64/
                            # BI=64：task2 pid=40+cid，cid≥8 时 g=1）预发射取错组的
                            # Indices（实证 max_abs=1.45e-2，g=1 行 1-11 精确命中）。
                            # 对既有全部用例（task(LEAD) 恒在 g=0 或 NI≥2 路径）两式
                            # 同值，行为不变；2026-10-11 P2b 依赖正确性分析发现并修复。
                            g_l = pid_l // (seq_len * REPLICATE_H)
                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)
                            T.copy(Indices[s_row_l, g_l, 0:BI],
                                   indices_ub[LEAD % IDXSLOTS, :])
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)

                    for k in T.serial(ntask):
                        # ---- task k 自身标量（k < ntask ⇒ 恒有效；post/选择用）----
                        pid = k * grid_size + cid
                        b_s_i = pid % (seq_len * REPLICATE_H)
                        g_i = pid // (seq_len * REPLICATE_H)
                        s_row_i = b_s_i // REPLICATE_H
                        r_i = b_s_i % REPLICATE_H
                        H0 = g_i * padded_H + r_i * R_HEAD_OFFSET
                        b_i = TokenIndices[s_row_i, 0]
                        s_i = TokenIndices[s_row_i, 1]
                        bos = Offsets[b_i]
                        # ---- task k+OFFSET_N 远端前瞻标量（iter8/#18：OFFSET_N=1 时即
                        # k+1，表达式与旧版逐字节一致；OFFSET_N=2（NI=1/LEAD=2）时为 k+2。
                        # k+OFFSET_N 可能越界 ⇒ s_row 先 clamp 到 [0, ΣS-1] 的合法行再读：
                        # b_n 恒为合法 batch id ⇒ Offsets[b_n] 恒 in-bounds；其值仅被
                        # gn<nblk guard 内的前瞻 gather 消费）----
                        pid_n = pid + OFFSET_N * grid_size
                        b_s_n = pid_n % (seq_len * REPLICATE_H)
                        g_n = pid_n // (seq_len * REPLICATE_H)
                        s_row_n = b_s_n // REPLICATE_H
                        r_n = b_s_n % REPLICATE_H
                        s_row_safe = T.min(s_row_n, seq_len - 1)
                        b_n = TokenIndices[s_row_safe, 0]
                        bos_n = Offsets[b_n]

                        # ---- 深度-DEPTH 免 free flag 的安全性推导（跨 task 延拓；
                        # iter8/#18 后 LEAD ≤ DEPTH-1 一般化：LEAD 越小间隔越宽）----
                        #   * bulk(gn) 重写 ws_kv slot gn%DEPTH：位于 body(gn-LEAD) 的
                        #     gather 段，程序序上晚于 body(gn-DEPTH) 的 p(gn-DEPTH)
                        #     （LEAD ≤ DEPTH-1 ⇒ body(gn-LEAD) ≥ body(gn-DEPTH+1)），
                        #     而 p(gn-DEPTH) 的 waitF1(gn-DEPTH) 已通过 ⇒ C 对同 slot
                        #     上一使用者 read(gn-DEPTH) 已由 gemm 首/尾部内建 flag
                        #     排空 ⇒ 写读无重叠；
                        #   * C 重写 ws_acc slot n%DEPTH（块 n，F0(n) 放行）：F0(n) 系于
                        #     bulk(n) 的 MTE3（发射于 body(n-LEAD)），其前置 mte2→mte3
                        #     set 捕获了更早程序点 body(n-DEPTH)（≤ body(n-LEAD-1)，
                        #     DEPTH>LEAD）的 post 段发出的 acc MTE2 read(n-DEPTH)
                        #     ⇒ 已排空；
                        #   * task 边界不再特殊：块流扁平连续，slot 复用周期恒为 DEPTH。
                        # 运行时 `if gn < nblk` 两 AIV 分支一致（k/t 同值），cross-flag
                        # set/wait 严格 1:1 quorum；全部位于有效块程序段（防悬挂，
                        # DESIGN §7.2 / HISA 教训）。
                        for t in T.serial(NI):
                            gn = k * NI + t + LEAD
                            if gn < nblk:
                                # ---- 前瞻 gather(gn) → slot gn%DEPTH，side gn%SIDES，islot gn%IDXSLOTS ----
                                tg = gn % NI  # gn 在所属 task 内的块号
                                # τg == k+OFFSET_N ⇔ t+LEAD ≥ NI（窗口不变式 assert 保证
                                # τg-k ∈ {0, OFFSET_N}，无中间任务 ⇒ 两标量组二选一即精确）
                                nxt = t + LEAD >= NI
                                nslot = gn % DEPTH
                                side = gn % SIDES
                                islot = gn % IDXSLOTS
                                s_row_g = T.if_then_else(nxt, s_row_n, s_row_i)
                                g_g = T.if_then_else(nxt, g_n, g_i)
                                # ---- idx(gn) 落地路径（iter6/#12 + iter8/#18 泛化 + p2/#19）----
                                # gn==LEAD：prologue 预发射；gn>LEAD：body(gn-LEAD-1) 的
                                # hoist 段预取（NI≥2 发射 guard t'+LEAD+1 < 2NI；NI==1 为
                                # 纯算术 k+3 窗恒发射，#19）。hoist 覆盖缺口 ⟺ LEAD ≥ NI 且
                                # NI > 1 且 t==0（t'=NI-1 时 guard 失败）⇒ 仅此配置需头部兜底；
                                # LEAD<NI 时 hoist 条件恒真 ⇒ 静态免兜底。
                                # 落地证明 = 发射方所在 body 的 mte2→mte3 行读 drain
                                # marker（idx 排在行读之后）⇒ hoisted 路径头部零 MTE2 等待
                                if LEAD >= NI and NI > 1:
                                    if t == 0:
                                        if gn > LEAD:
                                            T.set_flag("v", "mte2", 0)
                                            T.wait_flag("v", "mte2", 0)
                                            T.copy(
                                                Indices[s_row_g, g_g, tg * BI : (tg + 1) * BI],
                                                indices_ub[islot, :],
                                            )
                                            T.set_flag("mte2", "v", 0)
                                            T.wait_flag("mte2", "v", 0)
                                # rows_ub 重写保护：g(gn-1) 的 GetValue(S) 读毕
                                T.set_flag("s", "v", 0)
                                T.wait_flag("s", "v", 0)
                                # kv_gather_ub[side] 复用保护（P2，#15v2/iter8/#18 后）：
                                #  SIDES ≥ LEAD+1：免 flag——上一读者 bulk(gn-SIDES) 的
                                #    排空由 post(gn-SIDES) 的 waitF1(gn-SIDES) 架构性保证
                                #    （body(gn-SIDES) 程序序先于 body(gn-LEAD)，F1(m) ⇒
                                #    C 消费过 F0(m) ⇒ 两 AIV 的 bulk(m) 已排空）；
                                #  SIDES = LEAD（BI=128 类 UB 降级）：P2 移除原相邻
                                #    mte3→mte2 对，改依自然排空窗口（bulk 发射与行读
                                #    发射间隔 2 个含 waitF1 的 body ≥6.6µs + MTE2 前压
                                #    2.8µs vs bulk 排空 ≤2.0µs，余量 ≥4.7×——SIDES 定义
                                #    处完整推导 + Stage 3 spike/本轮消融实证）。
                                # 行地址向量化预计算（同 prologue；clamp 仅保护寻址）。
                                # bos 标量按 nxt 分支传入（GM 标量直读形式——select 包裹
                                # 会触发 AddsAndMulsOpCodegen 的 access_ptr 解析失败，
                                # codegen_ascend.cc L1786 GetAccessPtrDtype）：
                                if nxt:
                                    T.tile.add(rows_ub, indices_ub[islot, :], bos_n)
                                else:
                                    T.tile.add(rows_ub, indices_ub[islot, :], bos)
                                T.tile.clamp(rows_ub, rows_ub, 0, seq_len_kv - 1, BI)
                                T.set_flag("v", "s", 0)
                                T.wait_flag("v", "s", 0)
                                for j in range(HALF):
                                    row_j = vid * HALF + j
                                    kv_row = rows_ub[row_j]
                                    T.copy(KV[kv_row, g_g, :], kv_gather_ub[side, j, :])
                                # ---- idx(gn+1) 预取（iter6/#12 + iter9(p2)/#19）：行读后立即
                                # 发射，搭本段 mte2→mte3 drain marker 便车（落地免费证明）；
                                # 发射前 v→mte2 对排空最后 Cast 读者（IDXSLOTS ≥ LEAD+2 推导）。
                                # NI≥2：t+LEAD+1 ≥ 2NI 时块 gn+1 超 {k,k+1} 窗 ⇒ 跳过，由其
                                # gather 头兜底发射；
                                # NI==1（#19）：块 gn+1 = k+3 超旧窗（t+LEAD+1<2NI ≡ 3<2 恒假，
                                # iter8 曾全走兜底、每 body 暴露 ~300ns idx 落地等待），但
                                # Indices 行列寻址仅需 s_row/g = pid+3*grid 纯算术（无需
                                # bos/Offsets GM 标量读）⇒ 前瞻窗口免费扩展；k+3 < ntask 由
                                # 外层 gn+1<nblk guard 保证（g_h3 < kv_group 恒真）。
                                # 实测（Phase2 配对 A/B×3）：D=128（V 节拍）-4.2%（三对全胜
                                # ≥3%）；D=512（C 节拍，V 有空闲吸收）中性 ±0.5% ⇒ 采纳。
                                if gn + 1 < nblk:
                                    if NI == 1:
                                        pid_h3 = pid + 3 * grid_size
                                        b_s_h3 = pid_h3 % (seq_len * REPLICATE_H)
                                        s_row_h3 = b_s_h3 // REPLICATE_H
                                        g_h3 = pid_h3 // (seq_len * REPLICATE_H)
                                        T.set_flag("v", "mte2", 0)
                                        T.wait_flag("v", "mte2", 0)
                                        T.copy(Indices[s_row_h3, g_h3, 0:BI],
                                               indices_ub[(gn + 1) % IDXSLOTS, :])
                                    else:
                                        if t + LEAD + 1 < 2 * NI:
                                            T.set_flag("v", "mte2", 0)
                                            T.wait_flag("v", "mte2", 0)
                                            # 块 gn+1 ∈ task k + [t+LEAD+1 ≥ NI]（发射 guard
                                            # 保证 t+LEAD+1 < 2NI ⇒ τ ∈ {k, k+1}，两组标量在窗内）
                                            nh = t + LEAD + 1 >= NI
                                            s_row_h = T.if_then_else(nh, s_row_n, s_row_i)
                                            g_h = T.if_then_else(nh, g_n, g_i)
                                            T.copy(
                                                Indices[s_row_h,
                                                        g_h,
                                                        ((gn + 1) % NI) * BI :
                                                        ((gn + 1) % NI + 1) * BI],
                                                indices_ub[(gn + 1) % IDXSLOTS, :],
                                            )
                                T.set_flag("mte2", "mte3", 0)
                                T.wait_flag("mte2", "mte3", 0)
                                # rs_ub 重写保护（iter6/#15v2 相邻对，位置关键，同 prologue）：
                                # 行读排空后、bulk(gn) 发射前 ⇒ 仅捕获 bulk(gn-1)/rs(n-1)
                                # （均已自然排空 ⇒ wait ≈ 免费），不捕获本块 bulk(gn)
                                T.set_flag("mte3", "v", 0)
                                T.wait_flag("mte3", "v", 0)
                                T.copy(
                                    kv_gather_ub[side, :, :],
                                    ws_kv[cid,
                                          nslot * BI + vid * HALF :
                                          nslot * BI + (vid + 1) * HALF, :],
                                )
                                T.set_cross_flag("MTE3", F0_READY_BASE + nslot)

                            # ---- post(n=k*NI+t)：等 F1(slot) → acc 载入 → mask/exp/reduce → store ----
                            # 【iter-p3/#22 尾体 rs_ub 保护】gather 被跳过（gn≥nblk）时，原位于
                            # gather 段内的 mte3→v rs 保护对（#15v2）随之缺失：本 post 的
                            # reduce_sum(V 写 rs_ub) 可能与上一 post 的 rs store（MTE3 异步读）
                            # 竞态 ⇒ store 刷出混合数据（S=64/D128 间歇性 mask 区域脏值根因）。
                            # 尾体无新鲜 bulk ⇒ 在飞 MTE3 仅上一 store ⇒ wait≈免费；两 AIV
                            # 分支恒一致（k/t/nblk 同值），相邻对形态，协议计数不涉。
                            if gn >= nblk:
                                T.set_flag("mte3", "v", 0)
                                T.wait_flag("mte3", "v", 0)
                            n = k * NI + t
                            slot = n % DEPTH
                            T.wait_cross_flag(F1_ACC_BASE + slot)
                            # acc_ub 重写保护：p(n-1) 向量链(V) 读毕
                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)
                            T.copy(
                                ws_acc[cid, slot * H_per_block : (slot + 1) * H_per_block,
                                       vid * HALF : (vid + 1) * HALF],
                                acc_ub,
                            )
                            # acc 落地等待下沉至首个 acc 读取指令前（iter6/#12）：mask 链
                            # （~8 条 V 指令，仅依赖早已落地的 idx）与 acc MTE2 并行，暴露
                            # 仅剩 max(0, acc 延迟 - mask 链时长)。post 段零 MTE3 等待
                            # （#15v2）：rs_ub 保护由 gather 段行读排空后的相邻 mte3→v 对承担

                            # ---- mask 算术链（同 V pipe 顺序执行，零 barrier）----
                            T.copy(indices_ub[n % IDXSLOTS, vid * HALF : (vid + 1) * HALF],
                                   idx_f_ub)  # int32→fp32 Cast（idx 早于前一 body 已落地）
                            T.tile.add(t_ub, idx_f_ub, T.float32(1.0))  # t = idx + 1
                            T.tile.mul(m1_ub, idx_f_ub, T.float32(-1.0))  # m1 = -idx
                            T.tile.add(m1_ub, m1_ub, T.float32(s_i + 1))  # m1 = (s+1) - idx
                            T.tile.clamp(m1_ub, m1_ub, 0.0, 1.0, HALF)  # 1{idx <= s}
                            T.tile.clamp(m2_ub, t_ub, 0.0, 1.0, HALF)  # 1{idx >= 0}
                            T.tile.mul(mask_ub, m1_ub, m2_ub)  # mask ∈ {0, 1}
                            T.tile.broadcast(mask2d_ub, mask_ub, axis=0)  # [HALF] → [H, HALF]

                            # ---- score → p → reduce（DESIGN §8.4；lse 减法已向量化；
                            # lse2d_ub 于 task k-1 尾部（或 k=0 时 preamble）预载）----
                            # acc MTE2 落地证明（marker 同时覆盖本 body 行读与预取 idx）
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            T.tile.mul(acc_ub, acc_ub, sm_scale)
                            T.tile.sub(acc_ub, acc_ub, lse2d_ub)  # 2D 一次减，替代逐 h 循环
                            T.tile.exp(acc_ub, acc_ub)
                            T.tile.mul(acc_ub, acc_ub, mask2d_ub)  # masked → 0（等价 fill(-inf)+exp）
                            T.reduce_sum(acc_ub, rs_ub, dim=0, clear=True)
                            T.set_flag("v", "mte3", 0)
                            T.wait_flag("v", "mte3", 0)
                            T.copy(rs_ub, ReduceSum[s_row_i, g_i, r_i,
                                                    t * BI + vid * HALF :
                                                    t * BI + (vid + 1) * HALF])

                        # ---- task 尾：预载 task k+1 的 lse2d（guard 内 k+1 行有效；
                        # iter8/#18：OFFSET_N>1 时远窗标量组已让位 task k+OFFSET_N，
                        # lse 预载改用独立 task k+1 纯算术组（无 GM 标量读，单帧自包含
                        # ——TVMScript IfFrame 变量作用域约束）；OFFSET_N=1 时为原路径）----
                        # 时序安全：V pipe 序内 post(k,·) 的 sub 先于 broadcast(k+1)；
                        # lse_ub 重写由 v→mte2 guard 保护（上一 broadcast 读毕）。
                        if k + 1 < ntask:
                            if OFFSET_N > 1:
                                pid_1 = pid + grid_size
                                b_s_1 = pid_1 % (seq_len * REPLICATE_H)
                                g_1 = pid_1 // (seq_len * REPLICATE_H)
                                s_row_1 = b_s_1 // REPLICATE_H
                                r_1 = b_s_1 % REPLICATE_H
                                H0_1 = g_1 * padded_H + r_1 * R_HEAD_OFFSET
                                T.set_flag("v", "mte2", 0)
                                T.wait_flag("v", "mte2", 0)
                                T.copy(Lse[s_row_1, H0_1 : H0_1 + H_per_block], lse_ub)
                                T.set_flag("mte2", "v", 0)
                                T.wait_flag("mte2", "v", 0)
                                T.tile.broadcast(lse2d_ub, lse_ub, axis=1)
                            else:
                                H0_n = g_n * padded_H + r_n * R_HEAD_OFFSET
                                T.set_flag("v", "mte2", 0)
                                T.wait_flag("v", "mte2", 0)
                                T.copy(Lse[s_row_n, H0_n : H0_n + H_per_block], lse_ub)
                                T.set_flag("mte2", "v", 0)
                                T.wait_flag("mte2", "v", 0)
                                T.tile.broadcast(lse2d_ub, lse_ub, axis=1)

    return main


# ============================================================================
# prep kernel（DESIGN_dataflow §5.2——设备侧资格判定，V-only 40 lane；
# 架构 = 0f11b3cc detect 逐字复用 + 幅值门扩展）
# ============================================================================
def _reducesum_prep_body(topk, kv_group, grid_size):
    """prep kernel body：day-1 资格谓词（性能门控，非正确性防线，§5.7.2）。

    谓词 = 逐组"全行与相邻行逐元素相等"（链式比较数学精确，无漏报路径）∧
    索引值域 |idx| ≤ 2^24（fp32 排序/比较精确域——panel 路径 V 侧将 int32 索引
    cast fp32 后排序与等值比较，> 2^24 的整数值在 fp32 中失唯一性 ⇒ 舍入碰撞
    会破坏 pos 等值选择；违反时置 Violations 回退 generic，正确性无损）。
    共享 ⇒ 并集 = unique(row0)，|unique(row0)| ≤ topk ⇒ P_seq ≤ topk 恒成立
    （soundness）；误报（可 panel 的输入被判回 generic）只损失性能。
    """
    assert topk % 2 == 0 and topk >= 32, f"topk must be even and >= 32, got {topk}"
    GST = kv_group * topk  # 行整宽（flat 连续，一次比较覆盖全部 g）
    assert (GST * 4) % 32 == 0, f"row width {GST}*4B must be 32B aligned (T.tile operand)"
    C = min(256, 16384 // GST)  # 行窗 chunk（pre_ub (C+1)*GST*4 ≤ ~64KB）

    seq_len = T.symbolic("seq_len")  # ΣS
    batch_plus_one = T.symbolic("batch_plus_one")  # batch+1（Offsets 维）

    indices_dtype = "int32"
    accum_dtype = "float32"  # reduce_max 限 fp16/fp32

    @T.prim_func
    def main(
        Indices: T.Tensor([seq_len, kv_group, topk], indices_dtype),  # type: ignore  # [ΣS, G, topk] 只读
        Offsets: T.Tensor([batch_plus_one], indices_dtype),  # type: ignore  # [B+1] 分区输入
        Violations: T.Tensor([1], indices_dtype),  # type: ignore  # 资格谓词（atomic 累加）
    ):
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):
            lane = cid * 2 + vid  # 40 lane（V-only，0f11b3cc detect 同构）
            with T.Scope("V"):
                pre_ub = T.alloc_ub([C + 1, kv_group, topk], indices_dtype)
                d_i32 = T.alloc_ub([kv_group, topk], indices_dtype)  # int32 邻差
                d_f32 = T.alloc_ub([kv_group, topk], accum_dtype)  # fp32 归约链
                vmax_g = T.alloc_ub([kv_group], accum_dtype)  # 行最大中间量
                vmax_ub = T.alloc_ub([1], accum_dtype)  # 当行幅值
                diff_acc_ub = T.alloc_ub([1], accum_dtype)  # 邻差幅值累积器
                mag_acc_ub = T.alloc_ub([1], accum_dtype)  # 值域幅值累积器
                one_i32_ub = T.alloc_ub([1], indices_dtype)  # atomic 加数（只写一次）
                chk_tmp_ub = T.alloc_ub([256], "uint8")  # reduce 显式 tmp
                T.tile.fill(diff_acc_ub, 0.0)
                T.tile.fill(mag_acc_ub, 0.0)
                T.tile.fill(one_i32_ub, 1)
                for b in T.serial(batch_plus_one - 1):
                    bos = Offsets[b]
                    eos = Offsets[b + 1]
                    ncmp = eos - bos - 1  # 本 seq 相邻行比较数（len-1）
                    if ncmp > 0:  # 空 seq(-1)/单行 seq(0) 自然跳过
                        a = bos + 1 + (lane * ncmp) // 40
                        e = bos + 1 + ((lane + 1) * ncmp) // 40
                        for c in T.serial(T.ceildiv(e - a, C)):
                            lo = a + c * C
                            n = T.min(C, e - lo)
                            # 前块 V 读者排空（#15v2 相邻对；首块捕获空集）
                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)
                            for i in T.serial(n + 1):
                                T.copy(Indices[lo - 1 + i, :, :], pre_ub[i, :, :])
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            for i in T.serial(n):
                                # (1) 链式相邻相等检测（0f11b3cc detect 逐字）
                                T.tile.sub(d_i32, pre_ub[i + 1, :, :], pre_ub[i, :, :])
                                T.copy(d_i32, d_f32)  # int32→fp32 Cast
                                T.tile.abs(d_f32, d_f32)
                                T.pipe_barrier("V")
                                T.reduce_max(d_f32, vmax_g, dim=1, clear=True, tmp=chk_tmp_ub)
                                T.reduce_max(vmax_g, vmax_ub, dim=0, clear=True)
                                T.tile.max(diff_acc_ub, diff_acc_ub, vmax_ub)
                                # (2) 值域幅值门（P3 新增）：|idx| > 2^24 破坏 fp32
                                # 精确域 → violation（row i+1 覆盖 seq 内全部行——
                                # 比较对 (i,i+1) 的右侧行恰为 bos+1..eos-1 全集）
                                T.copy(pre_ub[i + 1, :, :], d_f32)  # int32→fp32 Cast
                                T.tile.abs(d_f32, d_f32)
                                T.pipe_barrier("V")
                                T.reduce_max(d_f32, vmax_g, dim=1, clear=True, tmp=chk_tmp_ub)
                                T.reduce_max(vmax_g, vmax_ub, dim=0, clear=True)
                                T.tile.max(mag_acc_ub, mag_acc_ub, vmax_ub)
                # 标量读落地（0f11b3cc detect 同款保护）
                T.set_flag("v", "s", 0)
                T.wait_flag("v", "s", 0)
                if diff_acc_ub[0] > 0.5:
                    T.set_flag("v", "mte3", 0)
                    T.wait_flag("v", "mte3", 0)
                    T.tile.atomic_add(Violations, one_i32_ub)
                if mag_acc_ub[0] > 16777216.0:  # 2^24：fp32 可精确整数上界
                    T.set_flag("v", "mte3", 0)
                    T.wait_flag("v", "mte3", 0)
                    T.tile.atomic_add(Violations, one_i32_ub)

    return main


@tilelang.jit(out_idx=[], workspace_idx=[])
def tl_sparse_mla_topk_reducesum_prep(
    topk,
    kv_group=1,
    grid_size=20,
):
    """构建 P3 prep kernel（设备侧资格判定，V-only）。

    Returns:
        可调用 kernel：func(Indices, Offsets, Violations) -> []。Violations [1]
        int32 由 host 每调用 torch.zeros 新建（graph capture 下 fill 被捕获，
        replay 重复置零——生命周期不变式），调用后 >0 即走 generic。
    """
    return _reducesum_prep_body(topk, kv_group, grid_size)


# ============================================================================
# panel kernel（DESIGN_dataflow §5.3/§5.4——P3 去重驻留路径；
# 组调度/8-flag/staging = 0f11b3cc 机制复用，选列 = reduce-first + 等值比较矩阵
# ——spike-2 裁决：gatherb 在 910B3 仅支持 uint16/uint32（fp32 静默无效），
# 元素 gather 需 uint32 计算偏移（vconv cast 不支持）⇒ 采用 R1 备选链的 UB 内
# 精化形态：rs_j = mask_j · rs_all[c*(j)]（mask 从 h 求和提出，数学恒等），
# c*(j) 经整数域等值比较矩阵选择（组内去重后恰一匹配）。
# ============================================================================
def _reducesum_panel_body(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group,
    sm_scale,
    grid_size,
    guarded,
):
    """panel 快速路径 body 构造器（DESIGN_dataflow §5，无契约通用机制）。

    正确域 = {P_seq ≤ topk}（尺寸谓词）；day-1 资格（逐行共享 + 值域 ≤ 2^24，
    prep 判定）为其 sound 子集。kernel 内 pos/等值映射按行独立计算——任意行序/
    重复/部分共享在正确域内均正确（§5.8）。guarded=True：入口守卫
    Violations[0] == 0（资格通过才执行）；早退 = C 循环零次 + 尾部 destroy 跳过、
    V 整体不进入（F_WSF 不 init、组调度零触碰）——无悬挂/无半更新（0f11b3cc
    §7.5 论证逐字适用）。
    """
    assert dim == tilelang.math.next_power_of_2(dim), f"dim must be a power of 2, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"tail_dim must be a power of 2, tail_dim={tail_dim}"
    assert topk % 32 == 0, f"panel domain: topk must be a multiple of 32 (sort 32-align), got {topk}"
    assert topk <= 128, f"panel domain: topk <= 128 (M=topk 编译期定宽 + L0A 预算), got {topk}"
    assert kv_group >= 1 and heads % kv_group == 0, f"heads={heads} must be divisible by kv_group={kv_group}"
    head_kv = heads // kv_group
    assert (16 <= head_kv <= 64 and head_kv == tilelang.math.next_power_of_2(head_kv)) or head_kv % 64 == 0, (
        f"head_kv={head_kv} out of supported domain: need (16<=head_kv<=64 and power of 2) or head_kv % 64 == 0"
    )
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

    seq_len = T.symbolic("seq_len")  # ΣS
    seq_len_kv = T.symbolic("seq_len_kv")  # ΣS_kv
    batch_plus_one = T.symbolic("batch_plus_one")  # batch+1（Offsets 维）

    dtype = "bfloat16"
    indices_dtype = "int32"
    accum_dtype = "float32"

    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != head_kv:
        assert kv_group == 1, (
            "H padding path only supports kv_group == 1 (inherited from the CUDA original: "
            "otherwise Q/Output head slices need explicit masks)"
        )
    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1
    H_per_block = padded_H if REPLICATE_H == 1 else 64
    R_HEAD_OFFSET = 0 if REPLICATE_H == 1 else 64

    HALF = topk // 2  # AIV 半区（topk 方向按 2 AIV 拆分）
    DDT = dim + tail_dim
    LA = 32  # V 侧组前瞻窗口（0f11b3cc 同值）
    H_CHUNK = 16  # exp/reduce-first 链的 h 分块宽（H_pb>16 时 UB 预算）
    N_HCHUNK = H_per_block // H_CHUNK

    # Cross flag 编号（DESIGN_dataflow §5.5 = 0f11b3cc §7.1/§7.2 配平表逐字沿用；
    # A2 FFTS flagId 4bit、8 个合法 id——恰满预算）
    F_PANEL_BASE = 0  # V→C：组 panel 已落 ws_panel[槽]（MTE3，mode-2 quorum）
    F_PDONE_BASE = 2  # C→V：ws_panel[槽] 已被消费，可重写（MTE2）
    F1_ACC_BASE = 4  # C→V：ws_acc[槽] score 就绪（FIX）
    F_WSF_BASE = 6  # V→C：ws_acc[槽] 可重写（MTE2，gqa_sink pretend-free 同款）

    # 片上容量自检（DESIGN_dataflow §5.6 门控的编译期镜像）
    assert 2 * topk * DDT * 2 + H_per_block * DDT * 2 + 4096 <= 524288, "panel L1 footprint over budget"
    assert topk * H_per_block * 4 <= 131072, "panel L0C accumulator over budget"
    _ub = (
        HALF * DDT * 2  # panel_stage
        + topk * H_CHUNK * 4  # accT chunk
        + topk * H_CHUNK * 4  # lseT chunk
        + 2 * topk * HALF * 4  # eq 工作矩阵 ×2
        + 2 * topk * 4  # sort dst + row0_f
        + 3 * topk * 4  # row0/rows_f/rows_i32
        + 13 * HALF * 4  # idx/mask/rs/k/misc 链
        + H_per_block * 4  # lse_ub
    )
    assert _ub <= 176 * 1024, f"panel V-side UB footprint over budget: {_ub}B"

    q_shape = [seq_len, heads, DDT]
    kv_shape = [seq_len_kv, kv_group, DDT]
    indices_shape = [seq_len, kv_group, topk]
    lse_shape = [seq_len, heads]
    offsets_shape = [batch_plus_one]
    token_indices_shape = [seq_len, 2]
    reducesum_shape = [seq_len, kv_group, REPLICATE_H, topk]

    def _guard_run(cond):
        if guarded:
            return cond
        return True

    def _guard_v_entry(ntask_pos, guard_cond):
        if guarded:
            return tir.all(ntask_pos, guard_cond)
        return ntask_pos

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore  # [ΣS, heads, DDT] bf16
        KV: T.Tensor(kv_shape, dtype),  # type: ignore  # [ΣS_kv, G, DDT] bf16
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore  # [ΣS, G, topk] int32
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore  # [ΣS, heads] fp32
        Offsets: T.Tensor(offsets_shape, indices_dtype),  # type: ignore  # [B+1] int32
        TokenIndices: T.Tensor(token_indices_shape, indices_dtype),  # type: ignore  # [ΣS, 2] int32
        Violations: T.Tensor([1], indices_dtype),  # type: ignore  # 守卫输入
        ReduceSum: T.Tensor(reducesum_shape, accum_dtype),  # type: ignore  # out（host 预分配）
        ws_panel: T.Tensor([grid_size, 2 * topk, DDT], dtype),  # type: ignore  # V→C panel GM 中转双缓冲
        ws_acc: T.Tensor([grid_size, 2 * topk, H_per_block], accum_dtype),  # type: ignore  # C→V score 中转双槽（[P,H] 转置布局）
    ):
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):
            # ---- Cube 侧：L1 / L0C ----
            # panel_l1 双槽 ping-pong（组切换时新组装载与旧组残余 gemm 重叠）；
            # 跨槽重写由 gemm_v0 尾部内建 M→MTE1→MTE2 排空 + C 同核程序序保证
            # （0f11b3cc 同款）。accT [topk, H]——转置 gemm 产物（spike-1 实证
            # 与非转置方向 torch.equal 逐位一致，K 累加序不变）。
            q_l1 = T.alloc_L1([H_per_block, DDT], dtype)
            panel_l1 = T.alloc_L1([2, topk, DDT], dtype)
            accT_l0c = T.alloc_L0C([topk, H_per_block], accum_dtype)

            # ---- Vector 侧：UB ----
            row0_ub = T.alloc_ub([topk], indices_dtype)  # 组首现行原始索引（int32）
            row0_f_ub = T.alloc_ub([topk], accum_dtype)  # fp32（sort src，原地 pad）
            sort_dst_ub = T.alloc_ub([2 * topk], accum_dtype)  # sort 交织 (val, idx) 输出
            rows_f_ub = T.alloc_ub([topk], accum_dtype)  # 降序值表 fp32（等值比较表，含重复）
            rows_i32_ub = T.alloc_ub([topk], indices_dtype)  # 值表 int32（staging 寻址）
            stage_rows_ub = T.alloc_ub([HALF], indices_dtype)  # 本 AIV staging 行地址
            panel_stage_ub = T.alloc_ub([HALF, DDT], dtype)  # staging（相邻 mte3→mte2 对保护）
            accT_ub = T.alloc_ub([topk, H_CHUNK], accum_dtype)  # ws_acc chunk
            lse_ub = T.alloc_ub([H_per_block], accum_dtype)  # 本 task lse（H_pb 全宽）
            lse_chunk_ub = T.alloc_ub([H_CHUNK], accum_dtype)  # lse chunk
            lseT_ub = T.alloc_ub([topk, H_CHUNK], accum_dtype)  # broadcast lse chunk
            rs_all_ub = T.alloc_ub([topk], accum_dtype)  # reduce-first h 归约结果
            idx_ub = T.alloc_ub([HALF], indices_dtype)  # 本行原始索引半区
            idx_f_ub = T.alloc_ub([HALF], accum_dtype)
            eq_a_ub = T.alloc_ub([topk, HALF], accum_dtype)  # 等值矩阵工作 A
            eq_b_ub = T.alloc_ub([topk, HALF], accum_dtype)  # 等值矩阵工作 B
            t_ub = T.alloc_ub([HALF], accum_dtype)
            m1_ub = T.alloc_ub([HALF], accum_dtype)
            m2_ub = T.alloc_ub([HALF], accum_dtype)
            mask_ub = T.alloc_ub([HALF], accum_dtype)
            rs_ub = T.alloc_ub([HALF], accum_dtype)
            k_ub = T.alloc_ub([HALF], accum_dtype)  # 等值匹配数（k 计数归一）

            total_tasks = seq_len * REPLICATE_H * kv_group
            ntask = T.max(0, T.ceildiv(total_tasks - cid, grid_size))

            # ---- 组调度跨迭代状态（0f11b3cc 同款）----
            prev_g = T.alloc_var("int32", init=-1)
            prev_b = T.alloc_var("int32", init=-1)
            pslot_c = T.alloc_var("int32", init=1)
            cur_g = T.alloc_var("int32", init=-1)
            cur_b = T.alloc_var("int32", init=-1)
            pslot_v = T.alloc_var("int32", init=1)
            gcnt = T.alloc_var("int32", init=0)

            with T.Scope("C"):
                if _guard_run(Violations[0] == 0):
                    for k in T.serial(ntask):
                        pid = k * grid_size + cid
                        b_s_i = pid % (seq_len * REPLICATE_H)
                        g_i = pid // (seq_len * REPLICATE_H)
                        s_row_i = b_s_i // REPLICATE_H
                        r_i = b_s_i % REPLICATE_H
                        H0 = g_i * padded_H + r_i * R_HEAD_OFFSET
                        H1 = H0 + H_per_block
                        # 组检测：每 task 1 次 C-scope GM 标量读（0f11b3cc 先例）
                        b_c = TokenIndices[s_row_i, 0]
                        if (g_i != prev_g) or (b_c != prev_b):
                            pslot_c = 1 - pslot_c
                            T.wait_cross_flag(F_PANEL_BASE + pslot_c)
                            T.copy(
                                ws_panel[cid, pslot_c * topk : (pslot_c + 1) * topk, :],
                                panel_l1[pslot_c, :, :],
                            )
                            T.set_cross_flag("MTE2", F_PDONE_BASE + pslot_c)
                        prev_g = g_i
                        prev_b = b_c
                        # Q 装载（每 task 一次；q_l1 跨 task 重写由 gemm_v0 尾部
                        # 内建 flag 排空保护——0f11b3cc 同款）
                        T.copy(Q[s_row_i, H0:H1, :], q_l1)
                        # 转置 gemm（P3 核心，spike-1 实证）：panel 作 A（M=topk，
                        # L0A [128,128]×2B=32KB 恰满预算）、Q 作 B（N=H_pb，
                        # transpose_B ⇒ K=DDT 由 B_shape[-1] 推导）；K 分块序
                        # （kL0Size=128）与非转置方向一致 ⇒ logits 逐位一致
                        T.gemm_v0(panel_l1[pslot_c, :, :], q_l1, accT_l0c, transpose_B=True, init=True)
                        # score 落地：L0C → GM（FIX）；写前 wait F_WSF（V 读毕上上
                        # task 的同槽）
                        slot = k % 2
                        T.wait_cross_flag(F_WSF_BASE + slot)
                        T.copy(
                            accT_l0c,
                            ws_acc[cid, slot * topk : (slot + 1) * topk, :],
                        )
                        T.set_cross_flag("FIX", F1_ACC_BASE + slot)
                    # 尾部 destroy-wait（消耗 V 的 init set 与末次 F_WSF set；
                    # ntask==0 / guarded 早退时 V 未 init，整体跳过——防悬挂）
                    if ntask > 0:
                        T.wait_cross_flag(F_WSF_BASE + 0)
                        T.wait_cross_flag(F_WSF_BASE + 1)

            with T.Scope("V"):
                if _guard_v_entry(ntask > 0, Violations[0] == 0):
                    # ---- init：F_WSF pretend-free ×2（gqa_sink 先例）----
                    T.set_cross_flag("MTE2", F_WSF_BASE + 0)
                    T.set_cross_flag("MTE2", F_WSF_BASE + 1)

                    # ---- prologue：扫描 task 0..LA-1 组键并按首现顺序 gather ----
                    # （`if i < ntask` 运行时守卫——0f11b3cc §6.1 红字：无守卫的
                    # 幻影组会使 V 多 set F_PANEL → 与 C wait 序列错位 → 死锁）
                    for i in range(LA):
                        if i < ntask:
                            pid_i = i * grid_size + cid
                            b_s_i = pid_i % (seq_len * REPLICATE_H)
                            g_i = pid_i // (seq_len * REPLICATE_H)
                            s_row_i = b_s_i // REPLICATE_H
                            b_i = TokenIndices[s_row_i, 0]
                            if (g_i != cur_g) or (b_i != cur_b):
                                pslot_v = 1 - pslot_v
                                if gcnt >= 2:
                                    T.wait_cross_flag(F_PDONE_BASE + pslot_v)
                                gcnt += 1
                                T.set_flag("v", "mte2", 0)
                                T.wait_flag("v", "mte2", 0)
                                T.copy(Indices[s_row_i, g_i, :], row0_ub)
                                T.set_flag("mte2", "v", 0)
                                T.wait_flag("mte2", "v", 0)
                                bos_x = Offsets[b_i]
                                # ---- panel 构造（组内一次，DESIGN_dataflow §5.2 +
                                # spike-2 裁决的 reduce-first/等值选择机制）----
                                # sort 降序（src 原地 pad -inf；topk%32==0 ⇒ 无 pad）；
                                # 重复值保留（死列无害——选值端 k 计数归一，见 post）
                                T.copy(row0_ub, row0_f_ub)  # int32→fp32 Cast
                                T.tile.sort(sort_dst_ub, row0_f_ub, topk)
                                T.tile.gather_mask(rows_f_ub, sort_dst_ub, "P0101")  # 偶位 = value lane
                                T.copy(rows_f_ub, rows_i32_ub)  # fp32→int32 Cast
                                # staging 行地址（本 AIV 半区；clamp 仅保护寻址）
                                T.tile.add(
                                    stage_rows_ub,
                                    rows_i32_ub[vid * HALF : (vid + 1) * HALF],
                                    bos_x,
                                )
                                T.tile.clamp(stage_rows_ub, stage_rows_ub, 0, seq_len_kv - 1, HALF)
                                # 行 gather（panel_stage_ub 复用：相邻 mte3→mte2 对）
                                T.set_flag("s", "v", 0)
                                T.wait_flag("s", "v", 0)
                                T.set_flag("mte3", "mte2", 0)
                                T.wait_flag("mte3", "mte2", 0)
                                T.set_flag("v", "s", 0)
                                T.wait_flag("v", "s", 0)
                                for j in range(HALF):
                                    kv_row = stage_rows_ub[j]
                                    T.copy(KV[kv_row, g_i, :], panel_stage_ub[j, :])
                                T.set_flag("mte2", "mte3", 0)
                                T.wait_flag("mte2", "mte3", 0)
                                T.copy(
                                    panel_stage_ub,
                                    ws_panel[
                                        cid,
                                        pslot_v * topk + vid * HALF : pslot_v * topk + (vid + 1) * HALF,
                                        :,
                                    ],
                                )
                                T.set_cross_flag("MTE3", F_PANEL_BASE + pslot_v)
                                cur_g = g_i
                                cur_b = b_i

                    # ---- 主循环：前瞻 gather(k+LA) + post(k) ----
                    for k in T.serial(ntask):
                        # 迭代头无条件 rs_ub 保护对（#22 尾体原则，0f11b3cc 同款：
                        # 不依赖 lookahead 是否执行；捕获上一 post 的 rs store（MTE3）
                        # ——set 在本 body lookahead bulk 之前发射 ⇒ 不捕获新鲜 bulk）
                        T.set_flag("mte3", "v", 0)
                        T.wait_flag("mte3", "v", 0)

                        pid = k * grid_size + cid
                        b_s_i = pid % (seq_len * REPLICATE_H)
                        g_i = pid // (seq_len * REPLICATE_H)
                        s_row_i = b_s_i // REPLICATE_H
                        r_i = b_s_i % REPLICATE_H
                        H0 = g_i * padded_H + r_i * R_HEAD_OFFSET
                        s_i = TokenIndices[s_row_i, 1]  # mask 链标量（causal 上界）
                        pid_n = pid + LA * grid_size
                        b_s_n = pid_n % (seq_len * REPLICATE_H)
                        g_n = pid_n // (seq_len * REPLICATE_H)
                        s_row_n = T.min(b_s_n // REPLICATE_H, seq_len - 1)
                        if k + LA < ntask:
                            b_n = TokenIndices[s_row_n, 0]
                            if (g_n != cur_g) or (b_n != cur_b):
                                pslot_v = 1 - pslot_v
                                if gcnt >= 2:
                                    T.wait_cross_flag(F_PDONE_BASE + pslot_v)
                                gcnt += 1
                                T.set_flag("v", "mte2", 0)
                                T.wait_flag("v", "mte2", 0)
                                T.copy(Indices[s_row_n, g_n, :], row0_ub)
                                T.set_flag("mte2", "v", 0)
                                T.wait_flag("mte2", "v", 0)
                                bos_x = Offsets[b_n]
                                T.copy(row0_ub, row0_f_ub)
                                T.tile.sort(sort_dst_ub, row0_f_ub, topk)
                                T.tile.gather_mask(rows_f_ub, sort_dst_ub, "P0101")
                                T.copy(rows_f_ub, rows_i32_ub)
                                T.tile.add(
                                    stage_rows_ub,
                                    rows_i32_ub[vid * HALF : (vid + 1) * HALF],
                                    bos_x,
                                )
                                T.tile.clamp(stage_rows_ub, stage_rows_ub, 0, seq_len_kv - 1, HALF)
                                T.set_flag("s", "v", 0)
                                T.wait_flag("s", "v", 0)
                                T.set_flag("mte3", "mte2", 0)
                                T.wait_flag("mte3", "mte2", 0)
                                T.set_flag("v", "s", 0)
                                T.wait_flag("v", "s", 0)
                                for j in range(HALF):
                                    kv_row = stage_rows_ub[j]
                                    T.copy(KV[kv_row, g_n, :], panel_stage_ub[j, :])
                                T.set_flag("mte2", "mte3", 0)
                                T.wait_flag("mte2", "mte3", 0)
                                T.copy(
                                    panel_stage_ub,
                                    ws_panel[
                                        cid,
                                        pslot_v * topk + vid * HALF : pslot_v * topk + (vid + 1) * HALF,
                                        :,
                                    ],
                                )
                                T.set_cross_flag("MTE3", F_PANEL_BASE + pslot_v)
                                cur_g = g_n
                                cur_b = b_n

                        # ---- post(k)：F1 → 全片读 → reduce-first → 等值选择 →
                        # mask → store → F_WSF 释放 ----
                        slot = k % 2
                        T.wait_cross_flag(F1_ACC_BASE + slot)
                        # accT_ub/lse_ub/idx_ub 重写保护：上一 post 的 V 读者排空
                        T.set_flag("v", "mte2", 0)
                        T.wait_flag("v", "mte2", 0)
                        # (a)+(b) 逐 chunk 装载-处理交错（单循环——装载与前一 chunk
                        # 计算天然经 MTE2/V 双队列重叠；列切片被 §3#7 封死 ⇒ 全读
                        # [topk, H_CHUNK] + UB 内选；H_pb=16 官方档单 chunk）：
                        #   rs_all[c] = Σ_h exp(accT[c,h]·s − lse[h])（reduce-first，
                        #   mask 从 h 求和提出的数学恒等；range 循环变量为 TIR 值，
                        #   clear 须为编译期 bool ⇒ fill+累加）
                        T.copy(Lse[s_row_i, H0 : H0 + H_per_block], lse_ub)
                        T.copy(Indices[s_row_i, g_i, vid * HALF : (vid + 1) * HALF], idx_ub)
                        # lse/idx 落地等待（单对覆盖两笔 MTE2 拷贝）
                        T.set_flag("mte2", "v", 0)
                        T.wait_flag("mte2", "v", 0)
                        T.tile.fill(rs_all_ub, 0.0)
                        for hc in range(N_HCHUNK):
                            T.copy(
                                ws_acc[
                                    cid,
                                    slot * topk : (slot + 1) * topk,
                                    hc * H_CHUNK : (hc + 1) * H_CHUNK,
                                ],
                                accT_ub,
                            )
                            T.copy(lse_ub[hc * H_CHUNK : (hc + 1) * H_CHUNK], lse_chunk_ub)
                            # 本 chunk 装载落地等待（相邻对；下一 chunk 装载在本
                            # chunk 计算期间经 MTE2 队列重叠推进）
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            T.tile.broadcast(lseT_ub, lse_chunk_ub, axis=0)  # [H_CHUNK]→[topk,H_CHUNK]
                            T.tile.mul(accT_ub, accT_ub, sm_scale)
                            T.tile.sub(accT_ub, accT_ub, lseT_ub)
                            T.tile.exp(accT_ub, accT_ub)
                            T.reduce_sum(accT_ub, rs_all_ub, dim=-1, clear=False)
                        # F_WSF 释放：set@MTE2 排在全部 chunk 装载之后（MTE2 FIFO
                        # 覆盖），装载落地即放行 C 重写（不等待 V 计算完成）
                        T.set_cross_flag("MTE2", F_WSF_BASE + slot)
                        T.copy(idx_ub, idx_f_ub)  # int32→fp32 Cast
                        # (c) 等值比较矩阵选值 + k 计数归一（spike-2 裁决机制）：
                        # 整数域 eq = 1 − clamp(|rows − idx|, 0, 1)；重复值的多列
                        # 同 KV 行 ⇒ rs_all 同值 ⇒ num = k·rs_all，除以匹配数 k 恰
                        # 精确复原（k=1 常见情形除法精确）；不在表（仅无效索引）⇒
                        # num=0 ∧ k=0 → clamp(k,1,·) ⇒ 0/1=0
                        T.tile.broadcast(eq_a_ub, rows_f_ub, axis=1)  # [topk]→[topk,HALF]
                        T.tile.broadcast(eq_b_ub, idx_f_ub, axis=0)  # [HALF]→[topk,HALF]
                        T.tile.sub(eq_b_ub, eq_a_ub, eq_b_ub)
                        T.tile.abs(eq_b_ub, eq_b_ub)
                        T.tile.clamp(eq_b_ub, eq_b_ub, 0.0, 1.0, topk * HALF)  # neq 指示（count=输出总尺寸，2D 必须全量）
                        T.tile.sub(eq_b_ub, eq_b_ub, 1.0)  # neq − 1
                        T.tile.mul(eq_b_ub, eq_b_ub, -1.0)  # eq = 1 − neq
                        T.reduce_sum(eq_b_ub, k_ub, dim=0, clear=True)  # k [HALF]
                        T.tile.clamp(k_ub, k_ub, 1.0, topk, HALF)  # k_safe（0→1）
                        T.tile.broadcast(eq_a_ub, rs_all_ub, axis=1)  # rs2d（rows 已消费）
                        T.tile.mul(eq_b_ub, eq_b_ub, eq_a_ub)  # eq × rs_all
                        T.reduce_sum(eq_b_ub, rs_ub, dim=0, clear=True)  # num [HALF]
                        T.tile.div(rs_ub, rs_ub, k_ub)  # rs_raw = num / k_safe
                        # (d) mask 链（原始 idx 构造，与 generic 算术逐字同款）
                        T.tile.add(t_ub, idx_f_ub, 1.0)
                        T.tile.clamp(m2_ub, t_ub, 0.0, 1.0, HALF)  # 1{idx >= 0}
                        T.tile.mul(m1_ub, idx_f_ub, -1.0)
                        T.tile.add(m1_ub, m1_ub, T.float32(s_i + 1))
                        T.tile.clamp(m1_ub, m1_ub, 0.0, 1.0, HALF)  # 1{idx <= s}
                        T.tile.mul(mask_ub, m1_ub, m2_ub)
                        T.tile.mul(rs_ub, rs_ub, mask_ub)
                        T.set_flag("v", "mte3", 0)
                        T.wait_flag("v", "mte3", 0)
                        T.copy(
                            rs_ub,
                            ReduceSum[s_row_i, g_i, r_i, vid * HALF : (vid + 1) * HALF],
                        )

                    # ---- 尾部 destroy：F_PDONE（仅实际用过的槽；消耗 C 的末次
                    # set 使计数封闭，0f11b3cc §7.2 配平表）----
                    if gcnt >= 1:
                        T.wait_cross_flag(F_PDONE_BASE + 0)
                    if gcnt >= 2:
                        T.wait_cross_flag(F_PDONE_BASE + 1)

    return main


@tilelang.jit(
    out_idx=[],  # ReduceSum 为 host 预分配共享输出（双发射互补守卫下恰一写入）
    workspace_idx=[8, 9],  # ws_panel, ws_acc（静态 [grid_size, ...] shape）
)
def tl_sparse_mla_topk_reducesum_panel_guarded(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    grid_size=20,
):
    """构建 P3 panel guarded kernel（去重驻留快速路径，DESIGN_dataflow §5）。

    与 generic_guarded 互补（守卫 Violations[0] == 0）：资格判定通过时执行
    panel 路径；不通过时整核早退让位 generic_guarded。

    Returns:
        可调用 kernel：func(Q, KV, Indices, Lse, Offsets, TokenIndices, Violations,
        ReduceSum) -> []；ReduceSum 由 host 预分配传入。
    """
    return _reducesum_panel_body(heads, dim, tail_dim, topk, kv_group, sm_scale, grid_size, guarded=True)


@tilelang.jit(
    out_idx=[6],  # ReduceSum 由框架分配并作为返回值
    workspace_idx=[7, 8],  # ws_kv, ws_acc（静态 [grid_size, ...] shape）
)
def tl_sparse_mla_topk_reducesum(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    block_I=32,
    grid_size=20,
):
    """构建 sparse_mla_topk_reducesum kernel（Expert 模式——DESIGN_generic_p1 §8.6
    移除后唯一 builder；公开名/签名/装饰器与 rev1 零差异，traced PrimFunc 与
    git 基线 0f11b3cc 的 unguarded 变体（= 历史通用路径）TIR 等价）。

    Args:
        heads: 总 head 数（head_kv = heads // kv_group）。
        dim: QK 主段维度 D（2 的幂）。
        tail_dim: QK 尾段维度 D_tail（2 的幂）。
        topk: 每行选中位置数（topk % block_I == 0）。
        kv_group: kv 分组数 G。
        sm_scale: 缺省 (dim + tail_dim) ** -0.5。
        block_I: topk 方向分块 BI（偶数且 >= 32；AIV 列半区 = BI // 2）。
        grid_size: persistent grid（编译期常量；host 查询 cube_core_num，910B3=20）。

    Returns:
        可调用 kernel：func(Q, KV, Indices, Lse, Offsets, TokenIndices)
        -> ReduceSum [ΣS, kv_group, REPLICATE_H, topk] float32。
    """
    return _strip_unused_param(
        _reducesum_generic_body(
            heads, dim, tail_dim, topk, kv_group, sm_scale, block_I, grid_size, guarded=False
        ),
        drop_idx=6,
    )


@tilelang.jit(
    out_idx=[],  # ReduceSum 为 host 预分配共享输出（普通输入；双发射互补守卫下
    # 恰一写入——wrapper 返回空表，host 以自分配张量为返回值）
    workspace_idx=[8, 9],  # ws_kv, ws_acc（静态 [grid_size, ...] shape）
)
def tl_sparse_mla_topk_reducesum_guarded(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    block_I=32,
    grid_size=20,
):
    """构建 sparse_mla_topk_reducesum guarded kernel（DESIGN_dataflow §5.0 回退臂）。

    与公开 unguarded 变体共享 body（守卫 = 入口 if Violations[0] > 0，0f11b3cc
    §8.2 已验证形态）：violations 非 0（prep 判定 panel 不可走）时全量执行，
    为 0 时整核早退让位 panel_guarded。

    Returns:
        可调用 kernel：func(Q, KV, Indices, Lse, Offsets, TokenIndices, Violations,
        ReduceSum) -> []；ReduceSum 由 host 预分配传入。
    """
    return _reducesum_generic_body(
        heads, dim, tail_dim, topk, kv_group, sm_scale, block_I, grid_size, guarded=True
    )


# ============================================================================
# Host 侧：数据准备 + 调用入口（DESIGN_generic_p1 §8.6 [HOST-METADATA-AUDIT]）
# 仅含：新建辅助张量（TokenIndices，CPU 构造 → 一次 H2D）、kernel 编译/调用。
# 不触碰用户输入 buffer；接口层归一化（sum(REPLICATE_H)/div）属交付边界外，不在此实现。
# ============================================================================
_BUILD_CACHE: dict = {}


def get_cube_core_num(default=20):
    """查询设备 cube 核数（grid_size 编译期常量来源；910B3 = 20）。"""
    try:
        import torch

        if torch.npu.is_available():
            n = torch.npu.get_device_properties(0).cube_core_num
            if isinstance(n, int) and n > 0:
                return n
    except Exception:  # pragma: no cover - 环境探测失败时回退默认值
        pass
    return default


def prepare_token_indices(offsets):
    """按 Offsets（cu_seqlens）生成 TokenIndices [ΣS, 2] int32（(batch_id, 行内位置)）。

    与 /tmp/tilelang_ref/index.py::prepare_token_indices 语义一致；CPU 构造 →
    一次 H2D，避免 host 侧 aclnn 依赖（DESIGN §10.4 备选路径）。
    """
    import torch

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


def _panel_domain_ok(heads, dim, tail_dim, topk, kv_group):
    """P3 host 静态门控（DESIGN_dataflow §5.7.4——编译期事实，无 D2H）。

    panel body 编译期 assert 的 host 镜像：miss 时仅编译/调用 generic（= P2
    基线行为逐字节，topk=256 / kvdb1 类 DDT=1024 等域外配置零影响）。
    """
    if dim != tilelang.math.next_power_of_2(dim) or tail_dim != tilelang.math.next_power_of_2(tail_dim):
        return False
    if topk % 32 != 0 or topk > 128:
        return False
    if kv_group < 1 or heads % kv_group != 0:
        return False
    head_kv = heads // kv_group
    ok = (16 <= head_kv <= 64 and head_kv == tilelang.math.next_power_of_2(head_kv)) or head_kv % 64 == 0
    if not ok:
        return False
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != head_kv and kv_group != 1:
        return False
    H_per_block = 64 if head_kv > 64 else padded_H
    DDT = dim + tail_dim
    if 2 * topk * DDT * 2 + H_per_block * DDT * 2 + 4096 > 524288:
        return False  # panel L1 超预算（kvdb1 类）
    if topk * H_per_block * 4 > 131072:
        return False  # L0C 超预算
    return True


def sparse_mla_topk_reducesum(
    q,
    kv,
    topk_indices,
    lse,
    offsets,
    dim_v=512,
    kv_group=None,
    sm_scale=None,
    block_I=32,
    grid_size=None,
):
    """Host 调用入口（API 零变化，DESIGN_dataflow §5.0——P3 同流三连发分发）。

    Args:
        q: [ΣS, heads, dim+tail_dim] bfloat16 (NPU)
        kv: [ΣS_kv, kv_group, dim+tail_dim] bfloat16 (NPU)
        topk_indices: [ΣS, kv_group, topk] int32 (NPU)，值域 [-1, ΣS_kv)，任意分布
        lse: [ΣS, heads] float32 (NPU)
        offsets: [batch+1] int32 cu_seqlens
        dim_v: 主段维度 D（tail_dim = 最后一维 - dim_v）
        kv_group: 缺省取 kv.shape[1]
        sm_scale: 缺省 (dim + tail_dim) ** -0.5
        block_I: generic 路径 topk 方向分块（panel 路径无 block_I 维）
        grid_size: 缺省查询设备 cube_core_num

    Returns:
        ReduceSum [ΣS, kv_group, REPLICATE_H, topk] float32 (NPU)

    分发（对任意合法输入正确，资格判定仅性能门控）：
        host 静态门控通过（topk%32==0 且 ≤128、L1/L0C 预算、head 域）→ 每调用
        三连发（graph 可捕获，kernel 边界 = 隐式 grid 屏障）：
            Violations = zeros(1) → prep(Indices, Offsets, Violations)
            → panel_guarded(..., Violations, rs)   [Violations==0 才执行]
            → generic_guarded(..., Violations, rs) [Violations!=0 才执行]
        门控 miss → 仅 generic（P2 基线行为）。
    """
    import torch

    seq_len, heads, ddt = q.shape
    tail_dim = ddt - dim_v
    kv_group = kv.shape[1] if kv_group is None else kv_group
    topk = topk_indices.shape[-1]
    if grid_size is None:
        grid_size = get_cube_core_num()

    key = (heads, dim_v, tail_dim, topk, kv_group, sm_scale, block_I, grid_size)
    bundle = _BUILD_CACHE.get(key)
    if bundle is None:
        if _panel_domain_ok(heads, dim_v, tail_dim, topk, kv_group):
            prep = tl_sparse_mla_topk_reducesum_prep(
                topk=topk, kv_group=kv_group, grid_size=grid_size
            )
            panel_g = tl_sparse_mla_topk_reducesum_panel_guarded(
                heads=heads, dim=dim_v, tail_dim=tail_dim, topk=topk,
                kv_group=kv_group, sm_scale=sm_scale, grid_size=grid_size,
            )
            generic_g = tl_sparse_mla_topk_reducesum_guarded(
                heads=heads, dim=dim_v, tail_dim=tail_dim, topk=topk,
                kv_group=kv_group, sm_scale=sm_scale, block_I=block_I,
                grid_size=grid_size,
            )
            bundle = ("triple", prep, panel_g, generic_g)
        else:
            func = tl_sparse_mla_topk_reducesum(
                heads=heads,
                dim=dim_v,
                tail_dim=tail_dim,
                topk=topk,
                kv_group=kv_group,
                sm_scale=sm_scale,
                block_I=block_I,
                grid_size=grid_size,
            )
            bundle = ("single", func)
        _BUILD_CACHE[key] = bundle

    token_indices = prepare_token_indices(offsets)  # 数据准备（新建张量，CPU→H2D）

    if bundle[0] == "single":
        LAST_DISPATCH["violations"] = None
        LAST_DISPATCH["path"] = "generic"  # 静态门控 miss（域外配置）
        return bundle[1](q, kv, topk_indices, lse, offsets, token_indices)

    _, prep, panel_g, generic_g = bundle
    head_kv = heads // kv_group
    RH = head_kv // 64 if head_kv > 64 else 1
    # Violations 每调用新建（0f11b3cc §8.4 生命周期不变式：graph capture 下
    # fill 被捕获、replay 重复置零）；rs 预分配为双守卫共享输出
    viol = torch.zeros(1, dtype=torch.int32, device=q.device)
    rs = torch.empty((seq_len, kv_group, RH, topk), dtype=torch.float32, device=q.device)
    prep(topk_indices, offsets, viol)
    panel_g(q, kv, topk_indices, lse, offsets, token_indices, viol, rs)
    generic_g(q, kv, topk_indices, lse, offsets, token_indices, viol, rs)
    LAST_DISPATCH["violations"] = viol
    LAST_DISPATCH["path"] = "triple"  # 实际路径由 viol 值决定（测试读 item() 断言）
    return rs
