"""sparse_mla_topk_reducesum — DSA 稀疏注意力 topk 概率质量 head 归约（Ascend Expert 模式）。

varlen packed 布局下，对每个 token 行 s、每个 kv_group g、每个 topk 选中位置
idx = Indices[s_row, g, j]，计算 score = exp((Q·KV_main + Q_tail·KV_tail)*sm_scale - Lse)
（causal mask 内置：idx <= s 且 idx != -1，否则 score=0），沿 head 维求和得到
ReduceSum[s, g, r, j]（未做 REPLICATE_H 归并与归一化——接口层归一化属本算子交付边界之外，
见 DESIGN.md §1.4/§10.4）。

编程模式：Expert（用户指定，DESIGN.md §2）
  - 显式层级：T.alloc_L1 / T.alloc_L0C / T.alloc_ub + GM workspace 中转（ws_kv / ws_acc）
  - 显式双流：T.Scope("C")（两段 gemm_v0）+ T.Scope("V")（mask/exp/reduce）
  - 手动同步：T.barrier_all() + T.set_cross_flag / T.wait_cross_flag（mode=2 quorum）
  - pass_configs 不设置（全手动，与 examples/sparse_flash_attention/
    example_sparse_flash_attn_dynamic_shape.py 同构）

Cross-flag 协议（扁平块流 n=task*NI+t 上严格 1:1 配对；V 侧全部 set/wait 位于 ntask/
`gn<nblk` 有效块程序段内、C 侧循环界即 ntask——防悬挂，HISA 头注释教训，DESIGN.md §7.2）：
  F0 = 0..DEPTH-1   V→C  set@MTE3：ws_kv[cid] slot 块 gather 完毕（mode-2：两 AIV 均写毕才放行 C）
  F1 = DEPTH..2*DEPTH-1  C→V  set@FIX ：ws_acc[cid] slot 块 score 就绪（L0C→GM 已落地）

实现要点（与 DESIGN.md §3.3 伪代码一致；差异处已注明）：
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
  4. mask 算术链（DESIGN §3.3 注）：m1 = clamp((s_i+1) - idx, 0, 1) = 1{idx <= s}，
      m2 = clamp(idx + 1, 0, 1) = 1{idx >= 0}，mask = m1 * m2，broadcast 到 [H, HALF]。
      `sub` 不支持 scalar 左操作数（ascend_tile.py L1121-1124），故 (s+1)-idx 用
      mul(-1) + add(scalar) 实现；s_i 为 GM 标量，经 T.float32(s_i + 1) 组合为
      PrimExpr scalar（binary_op ascend_adds 路径；先例：HISA T.float32(gm_scalar)）。

panel-mode 分发（DESIGN_panel_mode.md rev2，§8）：`panel_mode="auto"`（缺省）——
host 元数据门控 G1-G5 通过后由设备侧检测 kernel（§3.4 相邻行链式比较）判定索引共享
契约，panel/generic 双 guarded 变体按 violations 互补守卫（ntask 钳位，§8.2）同流
推测三连发恰一执行——任意输入结果正确；门控 miss → unguarded generic 单发（零开销
零变化）。"shared"（显式声明契约，rev1 行为）/ "generic"（恒通用路径）保留为显式覆盖。
builder 族谱（§3.5）：unguarded generic/panel（rev1 公开名/签名/装饰器零修改，
traced 语句与 rev1 逐条一致——guarded 守卫经 trace 期纯 Python 闭包选择，
unguarded 路径零 TIR 语句，spike S1 实证）+ guarded generic/panel（新，共享 body
构造）+ detect（新，V-only）。unguarded 变体的 Violations 形参经 _strip_unused_param
在 trace 后摘除（guarded=False 的 traced body 不含该引用）。

支持域（编译期 assert，DESIGN §10.1 [DISPATCH-COVERAGE]）：
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
    "tl_sparse_mla_topk_reducesum_panel",
    "tl_sparse_mla_topk_reducesum_panel_guarded",
    "tl_sparse_mla_topk_reducesum_detect",
    "sparse_mla_topk_reducesum",
    "prepare_token_indices",
]


def _strip_unused_param(func: PrimFunc, drop_idx: int) -> PrimFunc:
    """摘除 prim_func 中未被引用的 buffer 形参（DESIGN_panel_mode §3.5）。

    共享 body 构造器始终以 guarded 形态签名 trace（Violations 在参数表中）；
    guarded=False 的 traced body 不含 Violations 引用（守卫经 trace 期纯 Python
    闭包选择，零 TIR 语句），据此在 trace 后摘除该形参即精确复原 rev1 unguarded
    签名（参数序/数量/buffer_map 与 rev1 逐项一致）。
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

    new_params = params[:drop_idx] + params[drop_idx + 1 :]
    new_buffer_map = {v: b for v, b in func.buffer_map.items() if v != drop_var}
    return PrimFunc(new_params, func.body, func.ret_type, new_buffer_map, func.attrs, func.span)


# ============================================================================
# Kernel builder（DESIGN §3.3 / §4.7 / §8.3 / §8.4；rev2 §3.5 共享 body 构造）
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
    """通用路径共享 body 构造器（DESIGN_panel_mode §3.5）。

    guarded=False：traced 语句与 rev1 实现逐条一致（守卫经 trace 期纯 Python 闭包
    _guarded_ntask 直通，零 TIR 语句；Violations 形参不被引用，由薄包装
    _strip_unused_param 摘除）——unguarded 二进制保全的实现形态。
    guarded=True：ntask 绑定为 Select(Violations[0] > 0, raw, 0)（§8.2 ntask 钳位，
    违约才执行 generic），C 循环界/V `if ntask > 0` 均消费该值；早退路径 = ntask==0
    空闲核路径（§7.5 洁净性论证，L0-P4b 19/20 空闲核实证）。

    注：设计原文的"每 scope 一行 ntask 重绑定"在本仓 TVMScript 解析器下不可实现
    （with 帧内重绑定自动改名 ntask_1、原引用不更新——spike S1 实证）；此处为语义
    等价的单绑定形态：guard Select 绑定一次，两 scope 消费同一值（一致性更强）。
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
    # kv_gather_ub 缓冲深度（iter6/#15v2 + iter8/#18 UB 适配）：SIDES ≥ LEAD+1 时
    # side 复用免 flag——bulk(gn-SIDES) 排空由 post(gn-SIDES) 的 waitF1(gn-SIDES)
    # 架构性保证（F1(m) ⇒ C 已消费 F0(m-slot) ⇒ 两 AIV 的 MTE3 均已越过 bulk(m)
    # 后的 F0 marker ⇒ bulk(m) 排空），且 body(gn-SIDES) 程序序先于 body(gn-LEAD)。
    # UB 不足（BI=128/LEAD=2 需 3×73.7KB=221KB > 预算）时降级 SIDES=2 并恢复
    # "gather 段行读前相邻 mte3→mte2 对"保护（SIDE_PAIR=True）：body 周期 ≥ 块
    # bulk 排空时间（BI=128: ~4us vs 1.6us）⇒ 该相邻对恒近免费。
    # 【实证教训 2026-10-07】曾以"错位 marker"（SetFlag 于 bulk 后、WaitFlag 于
    # 2 body 后，事件槽按 (源管道,eventId) 共享）实现同保护：SetFlag/WaitFlag 为
    # fire-before-arm 即 miss 语义（printf 插桩实证：AIV 卡死于 MTE2 排空 wait /
    # prologue SetFlag，S=64 必现挂死、S=2048 侥幸通过），错位距离 ≫ DMA 排空
    # 时间时 arm 必然错过 fire → 严禁非相邻 set/wait 配对；相邻对（set 后至 wait
    # 仅 ns 级、事件 fire 在 μs 级）为该机制唯一安全形态。
    _kv_gather_budget = 170 * 1024  # UB 192KB 减去其余 buffer/对齐余量
    SIDES = (LEAD + 1) if (LEAD + 1) * (block_I // 2) * (dim + tail_dim) * 2 <= _kv_gather_budget else 2
    SIDE_PAIR = SIDES < LEAD + 1  # True ⇒ gather 段需相邻 mte3→mte2 保护对
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

    q_shape = [seq_len, heads, DDT]
    kv_shape = [seq_len_kv, kv_group, DDT]
    indices_shape = [seq_len, kv_group, topk]
    lse_shape = [seq_len, heads]
    offsets_shape = [batch_plus_one]
    token_indices_shape = [seq_len, 2]
    reducesum_shape = [seq_len, kv_group, REPLICATE_H, topk]

    def _guard_run(cond):
        # trace 期守卫选择（§8.2 入口守卫；纯 Python 闭包，traced 体内调用）：
        # guarded=False → 返回 Python True → `if True:` 为编译期分支，body 直通
        # trace（零 TIR 语句——unguarded traced 语句与 rev1 逐条一致，spike S1d
        # 实证 traced 脚本不含 Violations 引用）；guarded=True → 返回 TIR 比较
        # 表达式 → TIR IfThenElse 入口守卫（分支不进入 = 零 flag op / 零组调度
        # 状态触碰，§7.5 洁净性）。
        # 注：设计原文的 ntask Select 钳位与本节 min 算术钳位两种形态均在
        # serial 循环界中引入含 GM load 的表达式，实测引发 CanonicalSimplifier
        # 爆炸（generic 编译 13s → 600s+）；入口 if 守卫保持循环界与 rev1 全同
        # （分析器负载不变，编译时间持平），语义等价（互补条件恰一执行）。
        if guarded:
            return cond
        return True

    def _guard_v_entry(ntask_pos, guard_cond):
        # V scope 入口条件复合：unguarded → ntask_pos 原样（rev1 条件逐字一致）；
        # guarded → And(ntask_pos, guard_cond)——TIR if 复合条件，body 无需重缩进。
        # guard_cond 由调用方在 traced 上下文求值后传入（闭包不可见 prim_func 形参）。
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
            # side 重写保护（见 SIDES/SIDE_PAIR 定义处推导）：SIDES ≥ LEAD+1 ⇒ 免 flag
            # （上一读者 bulk(gn-SIDES) 排空由 post(gn-SIDES) 的 F1 wait 架构性保证）；
            # SIDE_PAIR=True（UB 降级 SIDES=LEAD）⇒ gather 段行读前相邻 mte3→mte2 对
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
            # 下一 task 的首批 gather 在本 task 中段即发射，消除旧结构 task 边界的
            # p(NI-1)→waitF1→下 task prologue 双侧 drain/refill 串行气泡
            # （msprof@711.8us：AIV waitF1 181.8us/26%、AIC waitF0 454.5us/65%）。
            # rev2 §8.2：guarded 变体的入口守卫（Violations[0] > 0 才执行——
            # violations 非 0 即契约破坏、需要 generic 全量；分支不进入 = 零
            # flag op / 零半更新，§7.5 洁净性）。detect 先行完成（同流串行，
            # §8.4）⇒ 全核全 scope 分支一致，互补条件与 panel_guarded 恰一执行。
            total_tasks = seq_len * REPLICATE_H * kv_group
            ntask = T.max(0, T.ceildiv(total_tasks - cid, grid_size))  # 本核有效任务数
            nblk = ntask * NI  # 本核扁平块总数

            with T.Scope("C"):
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
                # rev2 §8.2 V 入口守卫：guarded 时条件复合 And(ntask>0, Violations[0]>0)
                # （unguarded 时为 rev1 原条件，traced 语句逐条一致）。
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
                            g_l = b_s_l // (seq_len * REPLICATE_H)
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
                                # kv_gather_ub[side] 复用保护（#15v2 / iter8/#18 分岐）：
                                #  SIDE_PAIR=False（SIDES ≥ LEAD+1）：免 flag——上一读者
                                #    bulk(gn-SIDES) 的排空由 post(gn-SIDES) 的 waitF1(gn-SIDES)
                                #    架构性保证（body(gn-SIDES) 程序序先于 body(gn-LEAD)，
                                #    F1(m) ⇒ C 消费过 F0(m) ⇒ 两 AIV 的 bulk(m) 已排空）；
                                #  SIDE_PAIR=True（SIDES=LEAD，BI=128 UB 降级）：body(gn-SIDES)
                                #    与本 gather 同 body（post 段在其后）⇒ 架构保证失效，改用
                                #    下方相邻 mte3→mte2 对直接等 MTE3 全排空。
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
                                if SIDE_PAIR:
                                    # iter8/#18：同 side 上一读者 bulk(gn-SIDES) 排空保证
                                    # （相邻对形态——fire-before-arm 实证教训的唯一安全形；
                                    # 捕获已发射的全部 MTE3。body 周期 ≥ bulk 排空时间
                                    # （BI=128: ~4us vs ~1.6us）⇒ wait ≈ 免费）
                                    T.set_flag("mte3", "mte2", 0)
                                    T.wait_flag("mte3", "mte2", 0)
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
    """构建 sparse_mla_topk_reducesum kernel（Expert 模式，unguarded——rev1 公开
    名/签名/装饰器零修改，traced 语句与 rev1 逐条一致，§3.5 族谱）。

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
    out_idx=[],  # ReduceSum 为 host 预分配共享输出（普通输入；auto 三连发互补
    # 守卫下恰一写入，§8.2 裁决 3——wrapper 返回空表，host 以自分配张量为返回值）
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
    """构建 sparse_mla_topk_reducesum guarded kernel（Expert 模式，rev2 §3.5）。

    与 unguarded 变体共享 body（守卫 = ntask 钳位 Select(Violations[0] > 0, raw, 0)，
    §8.2）：violations 非 0（契约破坏）时全量执行，为 0 时整核早退（ntask==0 空闲核
    路径，零 flag op，§7.5）。

    Args:
        heads: 总 head 数（head_kv = heads // kv_group）。
        dim: QK 主段维度 D（2 的幂）。
        tail_dim: QK 尾段维度 D_tail（2 的幂）。
        topk: 每行选中位置数（topk % block_I == 0）。
        kv_group: kv 分组数 G。
        sm_scale: 缺省 (dim + tail_dim) ** -0.5。
        block_I: topk 方向分块 BI（偶数且 >= 32；AIV 列半区 = BI // 2）。
        grid_size: persistent grid（编译期常量；910B3 = 20）。

    Returns:
        可调用 kernel：func(Q, KV, Indices, Lse, Offsets, TokenIndices, Violations,
        ReduceSum) -> []；ReduceSum [ΣS, kv_group, REPLICATE_H, topk] float32 由
        host 预分配传入（两 guarded 变体互补守卫下恰一写入）。
    """
    return _reducesum_generic_body(
        heads, dim, tail_dim, topk, kv_group, sm_scale, block_I, grid_size, guarded=True
    )


# ============================================================================
# Panel-mode 变体（DESIGN_panel_mode.md）——共享索引面板快速路径
# ============================================================================
# 语义零变化的性能特化：当每个 (sequence, kv_group) 的所有行共享完全相同的
# topk 索引向量 P[b,g]（调用方声明 panel_mode="shared"，host 门控 G1-G5 +
# 域检查，kernel 内逐任务校验兜底）时，把每组的共享 KV 面板 [topk, DDT]
# 一次性载入 L1 常驻，C 流对每个 task 做单次 dense GEMM
# Q[HpB,DDT] × panel^T —— 完全跳过通用路径的 ws_kv GM 往返
# （golden 302MB → 全核 ~12MB，DESIGN §0.1/§4.3）。
#
# panel-vs-generic 等价性依据（DESIGN §9.1）：gemm 同形同 K 推进序
# （kL0Size=128、K=DDT 顺序累加）同 fp32 L0C 累加 ⇒ logits 逐位一致；
# V 侧 mask/exp/reduce 链与通用路径逐指令同构 ⇒ ReduceSum 期望逐位相等
# （L0-P1 torch.equal 实证）。
#
# Cross-flag 协议（8 id 恰满 A2 FFTS 4bit 合法预算，DESIGN §7.1/§7.2 配平表）：
#   F_PANEL[2]  id 0,1  V→C set@MTE3  组 g 的 panel 已落 ws_panel[槽]（mode-2 quorum）
#   F_PDONE[2]  id 2,3  C→V set@MTE2  ws_panel[槽] 已被 C 消费（L1 装载完成）
#   F1_ACC[2]   id 4,5  C→V set@FIX   ws_acc[槽] score 就绪（L0C→GM 已落地）
#   F_WSF[2]    id 6,7  V→C set@MTE2  ws_acc[槽] 可重写（gqa_sink pretend-free 同款）
# 计数配平（每核 ntask>0）：F_PANEL/F1 严格 1:1 无 init；F_PDONE V 运行期
# wait=组数-1 + 尾部 destroy×1/槽（仅实际用过的槽）；F_WSF V init×2 + 每 task
# set，C 每 task wait + 尾部 destroy×2。ntask==0 的核：V 整体不进入（不 init
# F_WSF）、C 循环零次且尾部 destroy 跳过 ⇒ 无悬挂。
#
# 组调度协议（DESIGN §6.1）：C 侧每 task 1 次 GM 标量读 b(k)=TokenIndices[s_row,0]
# 与 prev_key 比较（gqa_sink L268-270 C-scope 标量读先例，实现期 spike 已实证）；
# V 侧 prologue 静态展开扫描 task 0..LA-1 的组键（LA=32，每步必须带 `if i < ntask`
# 运行时守卫——无守卫时 clamp 幻影行会检测出任务流中不存在的组 → F_PANEL
# set/wait 序列错位 → 死锁），主循环每 task 前瞻 k+LA（守卫 k+LA < ntask）。
# 沿核任务流 (g,b) 键字典序单调（packed 布局保证）⇒ 两侧组首现序列严格一致，
# 槽位 0/1 交替同步（§6.1 一致性论证）。全部 cross-flag set/wait 的 guard 条件
# （k/ntask/组键/gcnt）与 vid 无关 ⇒ 两 AIV 恒同分支（mode-2 quorum 前提）。
#
# 尾体保护（#22 教训，DESIGN §6.2 风险表）：post 段 rs_ub 的 mte3→v 保护对
# 无条件存在于每 task 迭代头部（不依赖 lookahead gather 是否执行）；gather 段
# panel_stage_ub 复用由相邻 mte3→mte2 对保护；ws_panel/ws_acc 槽复用由
# F_PDONE/F_WSF 显式握手保护，不依赖时序推断。
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
    """panel 快速路径共享 body 构造器（DESIGN_panel_mode §3.5，同 generic 构造形态）。

    guarded=False：traced 语句与 rev1 panel 实现逐条一致（Violations 形参不被引用，
    薄包装 strip 复原签名）。guarded=True：ntask 绑定为 Select(Violations[0] == 0,
    raw, 0)（§8.2——violations==0 即检测契约成立才执行 panel；早退 = ntask==0
    空闲核路径，§7.5：ntask==0 时 C 循环零次/V 整体不进入，无任何组调度状态被
    部分更新、无 F_WSF init、无悬挂）。
    """
    # ---- 编译期 assert（域沿用通用路径，DESIGN §1.4；block_I 概念不适用：
    # gemm N = topk 整宽单 gemm，无 topk 方向分块）----
    assert dim == tilelang.math.next_power_of_2(dim), f"dim must be a power of 2, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"tail_dim must be a power of 2, tail_dim={tail_dim}"
    assert topk % 16 == 0, f"topk must be a multiple of 16 (gemm N fractal), got {topk}"
    assert topk % 2 == 0 and topk >= 32, f"topk must be even and >= 32 (AIV half-split), got {topk}"
    assert topk <= 128, f"panel v1 domain: topk <= 128 (kL0Size=128 L0B staging), got {topk}"
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

    HALF = topk // 2  # AIV 列半区（topk 方向按 2 AIV 拆分）
    DDT = dim + tail_dim
    LA = 32  # V 侧组前瞻窗口（编译期静态展开界，DESIGN §6.1；提前量 ~13-16us）

    # Cross flag 编号（DESIGN §7.1；A2 FFTS flagId 4bit、8 个合法 id——恰满预算）
    F_PANEL_BASE = 0  # V→C：组 panel 已落 ws_panel[槽]（MTE3，mode-2 quorum）
    F_PDONE_BASE = 2  # C→V：ws_panel[槽] 已被消费，可重写（MTE2）
    F1_ACC_BASE = 4  # C→V：ws_acc[槽] score 就绪（FIX）
    F_WSF_BASE = 6  # V→C：ws_acc[槽] 可重写（MTE2，gqa_sink pretend-free 同款）

    # 片上容量自检（DESIGN §1.2 门控的编译期镜像；golden 306KB/512KB L1）
    assert 2 * topk * DDT * 2 + H_per_block * DDT * 2 + 4096 <= 524288, "panel L1 footprint over budget (G1)"
    assert H_per_block * topk * 4 <= 131072, "panel L0C accumulator over budget (G3)"

    q_shape = [seq_len, heads, DDT]
    kv_shape = [seq_len_kv, kv_group, DDT]
    indices_shape = [seq_len, kv_group, topk]
    lse_shape = [seq_len, heads]
    offsets_shape = [batch_plus_one]
    token_indices_shape = [seq_len, 2]
    reducesum_shape = [seq_len, kv_group, REPLICATE_H, topk]

    def _guard_run(cond):
        # trace 期守卫选择（§8.2 入口守卫；同 generic 构造——unguarded 返回
        # Python True（编译期分支直通，零 TIR 语句）；guarded 返回 TIR 比较表达式
        # （TIR IfThenElse 入口守卫，分支不进入 = 零 flag op / 零组调度状态触碰，
        # §7.5）。ntask Select/min 钳位形态会污染 serial 循环界引发
        # CanonicalSimplifier 爆炸，见 generic 侧注释）。
        if guarded:
            return cond
        return True

    def _guard_v_entry(ntask_pos, guard_cond):
        # V scope 入口条件复合：unguarded → ntask_pos 原样（rev1 条件逐字一致）；
        # guarded → And(ntask_pos, guard_cond)（检测契约成立才执行 panel）。
        # guard_cond 由调用方在 traced 上下文求值后传入（闭包不可见 prim_func 形参）。
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
        Violations: T.Tensor([1], indices_dtype),  # type: ignore  # 守卫输入（unguarded trace 不引用，strip 复原）
        ErrFlag: T.Tensor([1], indices_dtype),  # type: ignore  # 契约违约计数（host 预置 0）
        ReduceSum: T.Tensor(reducesum_shape, accum_dtype),  # type: ignore  # out
        ws_panel: T.Tensor([grid_size, 2 * topk, DDT], dtype),  # V→C panel GM 中转双缓冲
        ws_acc: T.Tensor([grid_size, 2 * H_per_block, topk], accum_dtype),  # C→V score 中转双槽
    ):
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):
            # ---- Cube 侧：L1 / L0C（gemm_v0 内部管理 L0A/L0B）----
            # panel_l1 双槽 ping-pong（组切换时新组装载与旧组残余 gemm 重叠）；
            # 与通用路径 kv_l1 同形同保护（跨槽重写由 gemm_v0 尾部内建 M→MTE1→MTE2
            # 排空 + C 同核程序序保证，DESIGN §6.2 风险表第 4 行）。
            q_l1 = T.alloc_L1([H_per_block, DDT], dtype)
            panel_l1 = T.alloc_L1([2, topk, DDT], dtype)
            acc_s_l0c = T.alloc_L0C([H_per_block, topk], accum_dtype)

            # ---- Vector 侧：UB ----
            panel_idx_ub = T.alloc_ub([topk], indices_dtype)  # 当前组共享索引 P（int32）
            p_f_ub = T.alloc_ub([HALF], accum_dtype)  # P 半区 fp32（m1 链 + 契约校验共用）
            t_ub = T.alloc_ub([HALF], accum_dtype)  # P + 1（组内只算一次）
            m2_ub = T.alloc_ub([HALF], accum_dtype)  # 1{P >= 0}（组内只算一次）
            rows_ub = T.alloc_ub([HALF], indices_dtype)  # clamp 后的 KV 行地址（本 AIV 半区）
            panel_stage_ub = T.alloc_ub([HALF, DDT], dtype)  # gather staging（相邻 mte3→mte2 对保护）
            lse_ub = T.alloc_ub([H_per_block], accum_dtype)
            lse2d_ub = T.alloc_ub([H_per_block, HALF], accum_dtype)  # 每 task 广播一次
            chk_ub = T.alloc_ub([HALF], indices_dtype)  # 契约校验：本行实际 Indices 半区
            chk_f_ub = T.alloc_ub([HALF], accum_dtype)  # 校验链 fp32 中间量
            m1_ub = T.alloc_ub([HALF], accum_dtype)
            mask_ub = T.alloc_ub([HALF], accum_dtype)
            mask2d_ub = T.alloc_ub([H_per_block, HALF], accum_dtype)
            acc_ub = T.alloc_ub([H_per_block, HALF], accum_dtype)
            rs_ub = T.alloc_ub([HALF], accum_dtype)
            vmax_ub = T.alloc_ub([1], accum_dtype)  # 校验 reduce_max 输出
            one_i32_ub = T.alloc_ub([1], indices_dtype)  # 常量 1（违约 atomic 加数，只写一次）
            chk_tmp_ub = T.alloc_ub([256], "uint8")  # 校验 reduce 显式 tmp（AR rows=1 启发式
            #  仅给 32B 不足 DoReduceByBlk 所需 padLast*4B，显式 256B）

            # ---- persistent 轮询映射（与通用路径逐字节同构，#19-p2 继承）----
            # rev2 §8.2：guarded 变体的入口守卫（Violations[0] == 0 才执行——
            # 检测契约成立才走 panel；violations>0 时让位 generic_guarded 全量
            # 执行。分支不进入：C 循环零次 + 尾部 destroy 跳过、V 整体不进入
            # （F_WSF 不 init、组调度零触碰）——无悬挂/无半更新，§7.5 论证）。
            total_tasks = seq_len * REPLICATE_H * kv_group
            ntask = T.max(0, T.ceildiv(total_tasks - cid, grid_size))  # 本核有效任务数

            # ---- 组调度跨迭代状态（T.alloc_var 循环携带，sparse_flash_mla 先例）----
            # C 侧：prev_g/prev_b 组键检测；pslot_c 面板槽 ping-pong（首组 → 槽 0）
            prev_g = T.alloc_var("int32", init=-1)
            prev_b = T.alloc_var("int32", init=-1)
            pslot_c = T.alloc_var("int32", init=1)
            # V 侧：cur_g/cur_b 已 gather 组键；pslot_v 与 C 同步交替；gcnt 组计数
            # （gcnt >= 2 的 gather 复用槽位 → wait F_PDONE；尾部 destroy 仅对
            #  实际用过的槽：gcnt >= 1 → 槽 0、gcnt >= 2 → 槽 1）
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
                        s_row_i = b_s_i // REPLICATE_H  # packed 全局行号（≡ bos + s_i）
                        r_i = b_s_i % REPLICATE_H
                        H0 = g_i * padded_H + r_i * R_HEAD_OFFSET
                        H1 = H0 + H_per_block
                        # ---- 组检测：每 task 1 次 C-scope GM 标量读（§6.1；gqa_sink
                        # L268-270 先例，spike 实证）----
                        b_c = TokenIndices[s_row_i, 0]
                        if (g_i != prev_g) or (b_c != prev_b):
                            pslot_c = 1 - pslot_c
                            T.wait_cross_flag(F_PANEL_BASE + pslot_c)
                            # ws_panel 槽 → L1 整块（[topk, DDT] 连续，单条拷贝；运行时
                            # 槽位寻址与通用路径 ws_kv 切片同形）
                            T.copy(
                                ws_panel[cid, pslot_c * topk : (pslot_c + 1) * topk, :],
                                panel_l1[pslot_c, :, :],
                            )
                            T.set_cross_flag("MTE2", F_PDONE_BASE + pslot_c)
                        prev_g = g_i
                        prev_b = b_c
                        # ---- Q 装载 + 单 gemm（K 序与通用路径逐块 gemm 一致 ⇒ logits
                        # 逐位一致，§9.1 等价性依据；q_l1 跨 task 重写由 gemm_v0 尾部
                        # 内建 flag 排空保护，通用路径同款）----
                        T.copy(Q[s_row_i, H0:H1, :], q_l1)
                        T.gemm_v0(q_l1, panel_l1[pslot_c, :, :], acc_s_l0c, transpose_B=True, init=True)
                        # ---- score 落地：L0C → GM（FIX）；写前 wait F_WSF（V 读毕上上
                        # task 的同槽）——gqa_sink FLAG_WS*_FREE 同款 ----
                        slot = k % 2
                        T.wait_cross_flag(F_WSF_BASE + slot)
                        T.copy(
                            acc_s_l0c,
                            ws_acc[cid, slot * H_per_block : (slot + 1) * H_per_block, :],
                        )
                        T.set_cross_flag("FIX", F1_ACC_BASE + slot)
                    # 尾部 destroy-wait（消耗 V 的 init set 与末次 F_WSF set；ntask==0
                    # 时 V 未 init，必须整体跳过——防悬挂；guarded 早退分支整体不
                    # 进入，destroy 同样跳过——V 亦未 init，配平不变）
                    if ntask > 0:
                        T.wait_cross_flag(F_WSF_BASE + 0)
                        T.wait_cross_flag(F_WSF_BASE + 1)

            with T.Scope("V"):
                # rev2 §8.2 V 入口守卫：guarded 时条件复合 And(ntask>0, Violations[0]==0)
                # （unguarded 时为 rev1 原条件，traced 语句逐条一致）。
                if _guard_v_entry(ntask > 0, Violations[0] == 0):
                    # ---- init：F_WSF pretend-free ×2（gqa_sink L400 先例）+ 常量 1 ----
                    T.set_cross_flag("MTE2", F_WSF_BASE + 0)
                    T.set_cross_flag("MTE2", F_WSF_BASE + 1)
                    T.tile.fill(one_i32_ub, 1)

                    # ---- preamble：task0 标量 + lse2d(0) 预载（ntask>0 ⇒ in-bounds）----
                    b_s_0 = cid % (seq_len * REPLICATE_H)
                    g_0 = cid // (seq_len * REPLICATE_H)
                    s_row_0 = b_s_0 // REPLICATE_H
                    r_0 = b_s_0 % REPLICATE_H
                    H0_0 = g_0 * padded_H + r_0 * R_HEAD_OFFSET
                    T.set_flag("v", "mte2", 0)
                    T.wait_flag("v", "mte2", 0)
                    T.copy(Lse[s_row_0, H0_0 : H0_0 + H_per_block], lse_ub)
                    T.set_flag("mte2", "v", 0)
                    T.wait_flag("mte2", "v", 0)
                    T.tile.broadcast(lse2d_ub, lse_ub, axis=1)  # 每 task 一次

                    # ---- prologue：扫描 task 0..LA-1 组键并按首现顺序 gather ----
                    # （静态展开 + `if i < ntask` 运行时守卫——§6.1 红字：无守卫的
                    # 幻影组会使 V 多 set F_PANEL → 与 C wait 序列错位 → 死锁；
                    # cur_g/cur_b 仅可在守卫内更新）
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
                                # P 装载（整行 [topk] int32）
                                T.set_flag("v", "mte2", 0)
                                T.wait_flag("v", "mte2", 0)
                                T.copy(Indices[s_row_i, g_i, :], panel_idx_ub)
                                T.set_flag("mte2", "v", 0)
                                T.wait_flag("mte2", "v", 0)
                                bos_x = Offsets[b_i]
                                # 组内缓存链：P 半区 fp32 / m2 = 1{P>=0}
                                T.copy(panel_idx_ub[vid * HALF : (vid + 1) * HALF], p_f_ub)
                                T.tile.add(t_ub, p_f_ub, T.float32(1.0))
                                T.tile.clamp(m2_ub, t_ub, 0.0, 1.0, HALF)
                                # 行地址预 clamp（mask 已保证 -1/越界贡献为 0，clamp
                                # 仅保护寻址，通用路径 §5.4 语义一致）
                                T.tile.add(
                                    rows_ub,
                                    panel_idx_ub[vid * HALF : (vid + 1) * HALF],
                                    bos_x,
                                )
                                T.tile.clamp(rows_ub, rows_ub, 0, seq_len_kv - 1, HALF)
                                # 行 gather（panel_stage_ub 复用：相邻 mte3→mte2 对）
                                T.set_flag("s", "v", 0)
                                T.wait_flag("s", "v", 0)
                                T.set_flag("mte3", "mte2", 0)
                                T.wait_flag("mte3", "mte2", 0)
                                T.set_flag("v", "s", 0)
                                T.wait_flag("v", "s", 0)
                                for j in range(HALF):
                                    kv_row = rows_ub[j]
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
                        # 迭代头无条件 rs_ub 保护对（#22 尾体原则：不依赖 lookahead
                        # 是否执行；捕获上一 post 的 rs store 与上一 gather 的 bulk，
                        # 均已自然排空 ⇒ wait ≈ 免费）
                        T.set_flag("mte3", "v", 0)
                        T.wait_flag("mte3", "v", 0)

                        pid = k * grid_size + cid
                        b_s_i = pid % (seq_len * REPLICATE_H)
                        g_i = pid // (seq_len * REPLICATE_H)
                        s_row_i = b_s_i // REPLICATE_H
                        r_i = b_s_i % REPLICATE_H
                        s_i = TokenIndices[s_row_i, 1]  # mask 链标量（causal 上界）
                        # 前瞻 task k+LA 标量（越界时 clamp 寻址 + guard 跳过，
                        # 通用路径 pid_n clamp 同款）
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
                                T.copy(Indices[s_row_n, g_n, :], panel_idx_ub)
                                T.set_flag("mte2", "v", 0)
                                T.wait_flag("mte2", "v", 0)
                                bos_x = Offsets[b_n]
                                T.copy(panel_idx_ub[vid * HALF : (vid + 1) * HALF], p_f_ub)
                                T.tile.add(t_ub, p_f_ub, T.float32(1.0))
                                T.tile.clamp(m2_ub, t_ub, 0.0, 1.0, HALF)
                                T.tile.add(
                                    rows_ub,
                                    panel_idx_ub[vid * HALF : (vid + 1) * HALF],
                                    bos_x,
                                )
                                T.tile.clamp(rows_ub, rows_ub, 0, seq_len_kv - 1, HALF)
                                T.set_flag("s", "v", 0)
                                T.wait_flag("s", "v", 0)
                                T.set_flag("mte3", "mte2", 0)
                                T.wait_flag("mte3", "mte2", 0)
                                T.set_flag("v", "s", 0)
                                T.wait_flag("v", "s", 0)
                                for j in range(HALF):
                                    kv_row = rows_ub[j]
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

                        # ---- post(k)：F1 → acc 载入 → mask/exp/reduce → store →
                        # F_WSF + 逐任务契约校验 ----
                        slot = k % 2
                        T.wait_cross_flag(F1_ACC_BASE + slot)
                        # acc_ub/chk_ub 重写保护：上一 post 的 V 读者（score 链/校验
                        # 链）排空
                        T.set_flag("v", "mte2", 0)
                        T.wait_flag("v", "mte2", 0)
                        T.copy(
                            ws_acc[
                                cid,
                                slot * H_per_block : (slot + 1) * H_per_block,
                                vid * HALF : (vid + 1) * HALF,
                            ],
                            acc_ub,
                        )
                        # F_WSF 提前释放：set 排在 MTE2 管线该拷贝之后，拷贝落地即
                        # 放行 C 重写（槽可重写 ⇔ V 读毕，与 V 计算无依赖）
                        T.set_cross_flag("MTE2", F_WSF_BASE + slot)
                        # 契约校验源装载（MTE2，与 acc 拷贝同管线流水）
                        T.copy(
                            Indices[s_row_i, g_i, vid * HALF : (vid + 1) * HALF],
                            chk_ub,
                        )
                        # ---- mask 算术链（缓存 P + 新 s_i；与通用路径同构，区别仅
                        # m2/t 组内只算一次、idx 取自缓存 P）----
                        T.tile.mul(m1_ub, p_f_ub, T.float32(-1.0))  # m1 = -P
                        T.tile.add(m1_ub, m1_ub, T.float32(s_i + 1))  # m1 = (s+1) - P
                        T.tile.clamp(m1_ub, m1_ub, 0.0, 1.0, HALF)  # 1{P <= s}
                        T.tile.mul(mask_ub, m1_ub, m2_ub)  # mask = m1 * m2
                        T.tile.broadcast(mask2d_ub, mask_ub, axis=0)  # [HALF] -> [H, HALF]
                        # acc/chk 落地等待（单对覆盖两笔 MTE2）
                        T.set_flag("mte2", "v", 0)
                        T.wait_flag("mte2", "v", 0)
                        T.tile.mul(acc_ub, acc_ub, sm_scale)
                        T.tile.sub(acc_ub, acc_ub, lse2d_ub)
                        T.tile.exp(acc_ub, acc_ub)
                        T.tile.mul(acc_ub, acc_ub, mask2d_ub)  # masked -> 0
                        T.reduce_sum(acc_ub, rs_ub, dim=0, clear=True)
                        T.set_flag("v", "mte3", 0)
                        T.wait_flag("v", "mte3", 0)
                        T.copy(
                            rs_ub,
                            ReduceSum[s_row_i, g_i, r_i, vid * HALF : (vid + 1) * HALF],
                        )
                        # ---- 逐任务契约校验：Indices[s_row,g,半区] vs 缓存 P（§3.1，
                        # 违约 atomic_add(ErrFlag,1) 不沉默）----
                        T.copy(chk_ub, chk_f_ub)  # int32 -> fp32 Cast（V）
                        T.tile.sub(chk_f_ub, chk_f_ub, p_f_ub)
                        T.tile.abs(chk_f_ub, chk_f_ub)
                        # 防御性屏障：UB->UB cast 链与计数掩码 reduce 的向量状态隔离
                        # （实现期 spike 实证链路；屏障为 ns 级）
                        T.pipe_barrier("V")
                        T.reduce_max(chk_f_ub, vmax_ub, dim=0, clear=True, tmp=chk_tmp_ub)
                        T.set_flag("v", "s", 0)
                        T.wait_flag("v", "s", 0)
                        if vmax_ub[0] > 0.5:
                            # v->mte3 相邻对：atomic（DMA 类）读 one_i32_ub 的落地
                            # 保护（one_i32_ub 只写一次，保护为防御性）
                            T.set_flag("v", "mte3", 0)
                            T.wait_flag("v", "mte3", 0)
                            T.tile.atomic_add(ErrFlag, one_i32_ub)

                        # ---- task 尾：预载 task k+1 的 lse2d（纯算术组，无 GM 读）----
                        if k + 1 < ntask:
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

                    # ---- 尾部 destroy：F_PDONE（仅实际用过的槽；消耗 C 的末次
                    # set 使计数封闭，§7.2 配平表）----
                    if gcnt >= 1:
                        T.wait_cross_flag(F_PDONE_BASE + 0)
                    if gcnt >= 2:
                        T.wait_cross_flag(F_PDONE_BASE + 1)

    return main


@tilelang.jit(
    out_idx=[7],  # ReduceSum 由框架分配并作为返回值
    workspace_idx=[8, 9],  # ws_panel, ws_acc（静态 shape）
    # 注：ErrFlag 为普通输入张量（idx 6）——workspace_idx 张量由框架 torch.empty
    # 自动分配且不返回 host，无法承载"host 预置 0 + 调用后读回"语义（§8.3）。
)
def tl_sparse_mla_topk_reducesum_panel(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    grid_size=20,
):
    """构建 sparse_mla_topk_reducesum panel-mode kernel（Expert 模式，unguarded
    ——rev1 公开名/签名/装饰器零修改，traced 语句与 rev1 逐条一致，§3.5 族谱）。

    仅当调用方声明索引满足共享面板契约（每个 (sequence, kv_group) 内所有行
    的 Indices 向量逐元素相等，DESIGN_panel_mode §1.1）时使用；host 门控
    （G1-G5 + 域检查）miss 时由 `sparse_mla_topk_reducesum` 静默回退通用路径，
    运行时逐任务校验违约时 ErrFlag 置位 + host RuntimeError（不沉默出错）。

    Args:
        heads: 总 head 数（head_kv = heads // kv_group）。
        dim: QK 主段维度 D（2 的幂）。
        tail_dim: QK 尾段维度 D_tail（2 的幂）。
        topk: 每行选中位置数（v1 域：32 的倍数且 <= 128；gemm N = topk 整宽）。
        kv_group: kv 分组数 G。
        sm_scale: 缺省 (dim + tail_dim) ** -0.5。
        grid_size: persistent grid（编译期常量；910B3 = 20）。

    Returns:
        可调用 kernel：func(Q, KV, Indices, Lse, Offsets, TokenIndices, ErrFlag)
        -> ReduceSum [ΣS, kv_group, REPLICATE_H, topk] float32。
        ErrFlag [1] int32 由 host 预置 0 传入，调用后非零即契约违约。
    """
    return _strip_unused_param(
        _reducesum_panel_body(heads, dim, tail_dim, topk, kv_group, sm_scale, grid_size, guarded=False),
        drop_idx=6,
    )


@tilelang.jit(
    out_idx=[],  # ReduceSum 为 host 预分配共享输出（普通输入，同 generic_guarded）
    workspace_idx=[9, 10],  # ws_panel, ws_acc（静态 shape）
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
    """构建 sparse_mla_topk_reducesum panel-mode guarded kernel（rev2 §3.5）。

    与 unguarded panel 共享 body（守卫 = ntask 钳位 Select(Violations[0] == 0,
    raw, 0)，§8.2）：检测契约成立（violations==0）时执行 panel 快速路径并保留
    逐任务契约校验链（detect 漏报的纵深防御：违约必置 ErrFlag → host
    RuntimeError，§11 R9 ②层）；violations>0 时整核早退让位 generic_guarded。

    Args:
        heads: 总 head 数（head_kv = heads // kv_group）。
        dim: QK 主段维度 D（2 的幂）。
        tail_dim: QK 尾段维度 D_tail（2 的幂）。
        topk: 每行选中位置数（v1 域：32 的倍数且 <= 128；gemm N = topk 整宽）。
        kv_group: kv 分组数 G。
        sm_scale: 缺省 (dim + tail_dim) ** -0.5。
        grid_size: persistent grid（编译期常量；910B3 = 20）。

    Returns:
        可调用 kernel：func(Q, KV, Indices, Lse, Offsets, TokenIndices, Violations,
        ErrFlag, ReduceSum) -> []；ReduceSum 由 host 预分配传入，ErrFlag 预置 0、
        调用后非零即"detect 通过但逐任务校验违约"（detect bug 嫌疑，host 响亮报错）。
    """
    return _reducesum_panel_body(heads, dim, tail_dim, topk, kv_group, sm_scale, grid_size, guarded=True)


# ============================================================================
# 检测 kernel（DESIGN_panel_mode §3.4/§4.3/§5.2/§7.4，rev2 新增）
# ============================================================================
# 职责边界（极简性承诺，§11 R9 ①层）：仅判定 §1.5 链式契约谓词——读 Indices
# （只读）+ Offsets（只读）→ 写 Violations（atomic 计数）。不读 Q/KV/Lse，无浮点
# 语义计算，无 cross-flag，无 workspace。相邻行链式比较：契约"seq 内所有行逐元素
# 相等" ⟺ "seq 内每个相邻行对相等"（链式相等 ⟺ 全等，传递性归纳）；bid(r)==bid(r-1)
# 将比较严格限制在 seq 内部，跨 seq 边界行对不参与判定（无跨 seq 伪违约）。
#
# (seq, lane) 整除分区（§3.4 公式）：lane = cid*2+vid ∈ [0,40)（gdn_chunk_cumsum
# 同款先例），每 seq 的相邻行比较区间 [bos+1, eos) 按 a_l = bos+1+(l*ncmp)//40 整除
# 划分——覆盖恰好一次、两两不交 ⇒ 漏计为零；每 lane 至多 1 次 atomic（发现任一
# 违约即计 1）⇒ 过计无害（violations = 发现违约的 lane 数 ≤ 谓词真值，布尔判定
# 只看 >0）。行窗预载 [lo-1, lo+n) 自带前驱行（a-1 ≥ bos 恒成立，本 lane 自行
# 加载，无跨 lane 协调；相邻 lane ≤ 1 行重复只读，无害）。
#
# 同步纪律（§7.4）：无 cross-flag/barrier_all（40 lane 互相独立，分区不交），唯一
# 共享写 = Violations atomic（dst 限 GM/src 限 local/int32——ascend_tile.py L313-349；
# panel ErrFlag 生产先例）。核内相邻对全部无条件成对（#22 纪律）：每 chunk
# v→mte2（前块 V 读者排空）→ 预载行 MTE2 → mte2→v（预载落地）→ 比较链；尾部
# v→s（vmax 标量读）与 v→mte3（atomic 源落地）各一对（panel L1097-1104 同款）。
# 嵌套动态 T.serial（batch → chunk → 行，3 层动态界）= ntask 先例泛化（spike S2
# 实证：5 组 varlen 构造的分区计数全部精确）。
@tilelang.jit(out_idx=[], workspace_idx=[])
def tl_sparse_mla_topk_reducesum_detect(
    topk,
    kv_group=1,
    grid_size=20,
):
    """构建索引共享契约检测 kernel（V-only Expert，DESIGN_panel_mode §3.4）。

    判定 §1.5 链式谓词并累加违约 lane 计数：violations == 0 ⟺ 每个 (seq,
    kv_group) 内所有行共享同一索引向量（panel 契约成立）；>0 即契约破坏（auto
    分发据此回落 generic——误报方向天然安全，漏报由 panel 逐任务校验链兜底，
    §11 R9 三层防御）。

    Args:
        topk: 每行选中位置数（GST = kv_group*topk 需满足 GST*4 % 32 == 0 的
            T.tile 32B 对齐；门控上游 topk%32==0 保证恒成立，assert 为防御镜像）。
        kv_group: kv 分组数 G。
        grid_size: persistent grid（编译期常量；910B3 = 20；lane = cid*2+vid
            ∈ [0, 40)）。

    Returns:
        可调用 kernel：func(Indices, Offsets, Violations) -> []。Violations [1]
        int32 由 host 每调用 torch.zeros 新建（§8.4 生命周期不变式：禁止跨调用
        复用——graph replay 累积会翻转选路），调用后 >0 即契约破坏。
    """
    assert topk % 2 == 0 and topk >= 32, f"topk must be even and >= 32, got {topk}"
    GST = kv_group * topk  # 行整宽（flat 连续，一次比较覆盖全部 g——∃g 语义）
    assert (GST * 4) % 32 == 0, f"row width {GST}*4B must be 32B aligned (T.tile operand)"
    # 行窗 chunk（§5.2）：pre_ub (C+1)*GST*4 ≤ ~64KB（§4.3 UB 预算）
    C = min(256, 16384 // GST)

    seq_len = T.symbolic("seq_len")  # ΣS
    batch_plus_one = T.symbolic("batch_plus_one")  # batch+1（Offsets 维）

    indices_dtype = "int32"
    accum_dtype = "float32"  # reduce_max 限 fp16/fp32（reduce_ascend.py L373）

    @T.prim_func
    def main(
        Indices: T.Tensor([seq_len, kv_group, topk], indices_dtype),  # type: ignore  # [ΣS, G, topk] 只读
        Offsets: T.Tensor([batch_plus_one], indices_dtype),  # type: ignore  # [B+1] 分区输入
        Violations: T.Tensor([1], indices_dtype),  # type: ignore  # 违约 lane 计数（atomic）
    ):
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):
            lane = cid * 2 + vid  # 40 lane（V-only，moe_token_unpermute/gdn 同款）
            with T.Scope("V"):
                # 行窗预载（动态行偏移 + 静态 [G, topk] extent——indices_ub[islot]
                # 同款先例；含每 chunk 自带的前驱行）
                pre_ub = T.alloc_ub([C + 1, kv_group, topk], indices_dtype)
                d_i32 = T.alloc_ub([kv_group, topk], indices_dtype)  # int32 邻差
                d_f32 = T.alloc_ub([kv_group, topk], accum_dtype)  # fp32 归约链
                vmax_g = T.alloc_ub([kv_group], accum_dtype)  # 行最大中间量
                vmax_ub = T.alloc_ub([1], accum_dtype)  # 当行幅值
                vmax_acc_ub = T.alloc_ub([1], accum_dtype)  # 全局违约幅值累积器
                one_i32_ub = T.alloc_ub([1], indices_dtype)  # atomic 加数（只写一次）
                chk_tmp_ub = T.alloc_ub([256], "uint8")  # reduce 显式 tmp（panel L834-836 同款）
                T.tile.fill(vmax_acc_ub, 0.0)
                T.tile.fill(one_i32_ub, 1)
                for b in T.serial(batch_plus_one - 1):
                    bos = Offsets[b]
                    eos = Offsets[b + 1]
                    ncmp = eos - bos - 1  # 本 seq 相邻行比较数（len-1）
                    if ncmp > 0:  # 空 seq(-1)/单行 seq(0) 自然跳过
                        # 本 lane 比较子区间 [a, e)（整除分区：覆盖恰好一次、两两不交）
                        a = bos + 1 + (lane * ncmp) // 40
                        e = bos + 1 + ((lane + 1) * ncmp) // 40
                        for c in T.serial(T.ceildiv(e - a, C)):
                            lo = a + c * C
                            n = T.min(C, e - lo)
                            # 前块 V 读者排空（#15v2 相邻对；首块捕获空集，wait≈免费）
                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)
                            # 预载行窗 [lo-1, lo+n)（每行 1 条 MTE2，行流水发射）
                            for i in T.serial(n + 1):
                                T.copy(Indices[lo - 1 + i, :, :], pre_ub[i, :, :])
                            # 预载落地（相邻对）
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            # 比较链（panel 逐任务校验链 L1088-1104 逐指令同构）
                            for i in T.serial(n):
                                T.tile.sub(d_i32, pre_ub[i + 1, :, :], pre_ub[i, :, :])  # int32 邻差
                                T.copy(d_i32, d_f32)  # int32→fp32 Cast（reduce 限浮点）
                                T.tile.abs(d_f32, d_f32)
                                # cast 链与 reduce 的向量状态隔离（panel spike 实证）
                                T.pipe_barrier("V")
                                T.reduce_max(d_f32, vmax_g, dim=1, clear=True, tmp=chk_tmp_ub)  # [G,topk]→[G]
                                T.reduce_max(vmax_g, vmax_ub, dim=0, clear=True)  # [G]→[1]
                                T.tile.max(vmax_acc_ub, vmax_acc_ub, vmax_ub)  # 幅值累积
                # 标量读落地（vmax_ub[0] 同款 panel 先例）
                T.set_flag("v", "s", 0)
                T.wait_flag("v", "s", 0)
                if vmax_acc_ub[0] > 0.5:
                    # atomic（DMA 类）读 one_i32_ub 的落地保护（panel L1102-1104 同款）
                    T.set_flag("v", "mte3", 0)
                    T.wait_flag("v", "mte3", 0)
                    T.tile.atomic_add(Violations, one_i32_ub)

    return main


# ============================================================================
# Host 侧：数据准备 + 调用入口（DESIGN §10.4 [HOST-METADATA-AUDIT]）
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


def _token_indices_from_list(off_list, device):
    """prepare_token_indices 的列表版（panel 路径复用门控已取回的
    offsets .cpu().tolist()，零额外 D2H 同步，DESIGN_panel_mode §1.3）。

    构造语义与 prepare_token_indices 逐字节一致（数据准备类，CPU 构造 → 一次 H2D）。
    """
    import torch

    batch_ids = []
    positions = []
    for b in range(len(off_list) - 1):
        n = int(off_list[b + 1]) - int(off_list[b])
        batch_ids.append(torch.full((n,), b, dtype=torch.int32))
        positions.append(torch.arange(n, dtype=torch.int32))
    if batch_ids:
        ti = torch.stack([torch.cat(batch_ids), torch.cat(positions)], dim=1)
    else:
        ti = torch.zeros((0, 2), dtype=torch.int32)
    return ti.to(device)


# panel v1 域上界（DESIGN_panel_mode §1.2：topk ∈ (128,256] 需 kL0Size=64 +
# 单缓冲变体，列后续扩展；topk > 128 → host 门控回退通用路径）
_PANEL_V1_MAX_TOPK = 128


def _panel_gate_ok(heads, dim_v, tail_dim, topk, kv_group, seq_len, off_list):
    """panel 路径 host 门控（DESIGN_panel_mode §1.2 G1-G5 + §1.4 域检查）。

    纯 Python 元数据运算（off_list 已是 .cpu().tolist() 结果，零额外同步，
    [HOST-METADATA-AUDIT] allow）；不触碰任何 NPU tensor。门控 miss → 调用方
    静默回退通用路径（性能回退，非正确性差异，L0-P4e/P6 锁定）。

    Returns:
        (ok: bool, reason: str)
    """
    # ---- 域检查（沿用通用路径编译期 assert 的 host 镜像，§1.4）----
    if dim_v != tilelang.math.next_power_of_2(dim_v):
        return False, "dim not power of 2"
    if tail_dim != tilelang.math.next_power_of_2(tail_dim):
        return False, "tail_dim not power of 2"
    if topk < 32 or topk % 32 != 0:
        return False, f"topk={topk} not multiple of 32 (generic-compatible domain)"
    if topk > _PANEL_V1_MAX_TOPK:
        return False, f"G2/G3: topk={topk} > {_PANEL_V1_MAX_TOPK} (panel v1 domain)"
    if heads % kv_group != 0:
        return False, f"heads={heads} not divisible by kv_group={kv_group}"
    head_kv = heads // kv_group
    if not ((16 <= head_kv <= 64 and head_kv == tilelang.math.next_power_of_2(head_kv)) or head_kv % 64 == 0):
        return False, f"head_kv={head_kv} out of domain"
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != head_kv and kv_group != 1:
        return False, "H padding path requires kv_group == 1"
    replicate_h = 1 if head_kv <= 64 else head_kv // 64
    h_per_block = padded_H if replicate_h == 1 else 64
    ddt = dim_v + tail_dim
    # ---- G1：每核 L1 panel 足迹（双缓冲 panel + q_l1 + 4KB 余量 ≤ 512KB）----
    if 2 * topk * ddt * 2 + h_per_block * ddt * 2 + 4096 > 524288:
        return False, f"G1: L1 panel footprint {2 * topk * ddt * 2 + h_per_block * ddt * 2}B over budget"
    # ---- G3：L0C 累加器（G2 已由 topk <= 128 域检查覆盖）----
    if h_per_block * topk * 4 > 131072:
        return False, f"G3: L0C accumulator {h_per_block * topk * 4}B over budget"
    # ---- G4/G5：摊销门控（轮询映射下每核 panel 装载数上界 / 每 panel 平均任务数）----
    n_batch = sum(1 for a, b in zip(off_list, off_list[1:]) if int(b) - int(a) > 0)
    if n_batch <= 0:
        return False, "no non-empty sequence"
    if n_batch * kv_group > 8:
        return False, f"G4: B*G={n_batch * kv_group} > 8 (panel amortization collapse)"
    if seq_len * replicate_h < 16 * n_batch:
        return False, f"G5: amortization {seq_len * replicate_h}/{n_batch} < 16 tasks/panel"
    return True, "ok"


def _replicate_h_of(heads, kv_group):
    """host 侧 REPLICATE_H 推导（镜像 builder 编译期逻辑，§4.1 rs 预分配用）。"""
    head_kv = heads // kv_group
    return 1 if head_kv <= 64 else head_kv // 64


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
    panel_mode="auto",
):
    """Host 调用入口（原仓 sparse_mla_topk_reducesum_interface 同语义，交付边界止于 ReduceSum）。

    Args:
        q: [ΣS, heads, dim+tail_dim] bfloat16 (NPU)
        kv: [ΣS_kv, kv_group, dim+tail_dim] bfloat16 (NPU)
        topk_indices: [ΣS, kv_group, topk] int32 (NPU)，值域 [-1, ΣS_kv)
        lse: [ΣS, heads] float32 (NPU)
        offsets: [batch+1] int32 cu_seqlens
        dim_v: 主段维度 D（tail_dim = 最后一维 - dim_v）
        kv_group: 缺省取 kv.shape[1]
        sm_scale: 缺省 (dim + tail_dim) ** -0.5
        block_I: topk 方向分块（偶数且 >= 32；仅通用路径使用）
        grid_size: 缺省查询设备 cube_core_num
        panel_mode: 三模式分发（DESIGN_panel_mode §8.1 语义表，rev2）：
            "auto"（缺省）——调用方零声明：host 元数据门控（G1-G5 + 域检查，纯
              shape+offsets）通过后由设备侧检测 kernel 判定索引共享契约（§3.4），
              panel/generic 双 guarded 变体按 violations 互补守卫同流推测三连发
              恰一执行（§8.2/§8.4——任意输入结果正确）；门控 miss → unguarded
              generic 单发（零开销零变化，P7d 探针锁定）。panel 合格输入 ≈155-157us
              （§10.4，透明性代价 +12-14us vs shared）。
            "generic"（显式）——恒通用路径（unguarded 单发，rev1 缺省行为零变化）。
            "shared"（显式）——调用方声明契约（rev1 方案 A 行为原样：无检测，
              host 门控 miss 静默回退 generic，ErrFlag 运行时兜底 → RuntimeError）。
            其余取值沿用 rev1 容错语义：落入 generic（[DISPATCH-COVERAGE]
            unsupported_inputs: 无）。

    Returns:
        ReduceSum [ΣS, kv_group, REPLICATE_H, topk] float32 (NPU)

    性能配置披露（iter7/#17 sweep + iter8/#18，2026-10-08）:
        topk % 128 == 0 时显式传 block_I=128 为最快编译配置（NI=1 前瞻窗口 LEAD=2，
        考核配置 S=2048/topk=128 实测 graph-replay 426.7us / msprof 433.8us，
        vs block_I=32 的 520us；证据链见 perf_tuning/perf_log.md Iteration 7/8）。
        默认值保持 block_I=32：(a) 兼容任意 topk≥32 支持域（128 仅覆盖 topk%128==0）；
        (b) l2 负向用例（topk48_not_div_blockI32）依赖默认值的 assert 语义。
        block_I=64 为 topk%64==0 且 %128≠0 场景的次优配置（496us）。
    """
    seq_len, heads, ddt = q.shape
    tail_dim = ddt - dim_v
    kv_group = kv.shape[1] if kv_group is None else kv_group
    topk = topk_indices.shape[-1]
    if grid_size is None:
        grid_size = get_cube_core_num()

    # ---- 显式 generic：rev1 缺省路径逐字节一致（off_list=None → 原样
    # prepare_token_indices，§8.3）----
    off_list = None
    if panel_mode in ("auto", "shared"):
        # 既有 D2H（1 次；auto/shared 共用，§8.3 D2H 计数对账）
        off_list = offsets.detach().cpu().tolist()
        gate_ok, gate_reason = _panel_gate_ok(heads, dim_v, tail_dim, topk, kv_group, seq_len, off_list)
        if panel_mode == "shared":
            if gate_ok:
                import torch

                pkey = (heads, dim_v, tail_dim, topk, kv_group, sm_scale, "panel", grid_size)
                pfunc = _BUILD_CACHE.get(pkey)
                if pfunc is None:
                    pfunc = tl_sparse_mla_topk_reducesum_panel(
                        heads=heads,
                        dim=dim_v,
                        tail_dim=tail_dim,
                        topk=topk,
                        kv_group=kv_group,
                        sm_scale=sm_scale,
                        grid_size=grid_size,
                    )
                    _BUILD_CACHE[pkey] = pfunc
                token_indices = _token_indices_from_list(off_list, offsets.device)
                # 契约违约计数（数据准备类：新建张量 + 一次置零，§8.3 审计 allow）
                err_flag = torch.zeros(1, dtype=torch.int32, device=q.device)
                out = pfunc(q, kv, topk_indices, lse, offsets, token_indices, err_flag)
                n_err = int(err_flag.item())  # 调用后强制检查（非计时路径，§8.3）
                if n_err != 0:
                    raise RuntimeError(
                        f"panel contract violated ({n_err} task-level index mismatches): "
                        "panel_mode='shared' requires all rows within each (sequence, "
                        "kv_group) to share an identical Indices vector "
                        "(DESIGN_panel_mode §1.1). Re-run with panel_mode='generic' "
                        "for correct results."
                    )
                return out
            # 门控 miss → 静默回退通用路径（性能回退，非正确性差异；L0-P4e/P6 锁定）
        elif gate_ok:
            # ---- auto 推测三连发（§8.3 伪代码；零 host 同步——全部为同流异步 op
            # 序列，graph 可捕获，§8.4 流序论证：detect 的 Violations 写（atomic）
            # 在其后任意 kernel 守卫读之前全局可见）----
            import torch

            token_indices = _token_indices_from_list(off_list, offsets.device)  # off_list 复用
            # violations 每调用新建（§8.4 生命周期不变式：复用在 graph replay 下跨
            # 调用累积翻转选路；err_flag 同款数据准备先例）
            violations = torch.zeros(1, dtype=torch.int32, device=q.device)
            err_flag = torch.zeros(1, dtype=torch.int32, device=q.device)
            rs = torch.empty(
                (seq_len, kv_group, _replicate_h_of(heads, kv_group), topk),
                dtype=torch.float32,
                device=q.device,
            )
            # _BUILD_CACHE 键扩展（§8.3：detect/generic_guarded/panel_guarded 维度）
            dkey = (topk, kv_group, "detect", grid_size)
            detect_f = _BUILD_CACHE.get(dkey)
            if detect_f is None:
                detect_f = tl_sparse_mla_topk_reducesum_detect(
                    topk=topk,
                    kv_group=kv_group,
                    grid_size=grid_size,
                )
                _BUILD_CACHE[dkey] = detect_f
            pgkey = (heads, dim_v, tail_dim, topk, kv_group, sm_scale, "panel", grid_size, "guarded")
            panel_g = _BUILD_CACHE.get(pgkey)
            if panel_g is None:
                panel_g = tl_sparse_mla_topk_reducesum_panel_guarded(
                    heads=heads,
                    dim=dim_v,
                    tail_dim=tail_dim,
                    topk=topk,
                    kv_group=kv_group,
                    sm_scale=sm_scale,
                    grid_size=grid_size,
                )
                _BUILD_CACHE[pgkey] = panel_g
            ggkey = (heads, dim_v, tail_dim, topk, kv_group, sm_scale, block_I, grid_size, "guarded")
            generic_g = _BUILD_CACHE.get(ggkey)
            if generic_g is None:
                generic_g = tl_sparse_mla_topk_reducesum_guarded(
                    heads=heads,
                    dim=dim_v,
                    tail_dim=tail_dim,
                    topk=topk,
                    kv_group=kv_group,
                    sm_scale=sm_scale,
                    block_I=block_I,
                    grid_size=grid_size,
                )
                _BUILD_CACHE[ggkey] = generic_g
            # 同流首发：流序保证 detect 先于两 guards 的 Violations 读取完成
            detect_f(topk_indices, offsets, violations)
            panel_g(q, kv, topk_indices, lse, offsets, token_indices, violations, err_flag, rs)
            generic_g(q, kv, topk_indices, lse, offsets, token_indices, violations, rs)
            # 尾部防御检查（eager；rev1 shared 同款先例、位于计时/捕获路径外）：
            # detect 通过但 panel 逐任务校验链仍捕获违约 ⇒ detect bug 嫌疑，
            # 响亮报错不沉默（§11 R9 ②层）
            n_err = int(err_flag.item())
            if n_err != 0:
                raise RuntimeError(
                    f"panel contract violated despite detect pass ({n_err} task-level "
                    "index mismatches; possible detect bug — please report). "
                    "Re-run with panel_mode='generic' for correct results."
                )
            return rs
        # auto 门控 miss → 落入通用（off_list 复用——D2H 总次数与旧默认路径持平）

    # ---- 通用路径公共出口（显式 generic / shared·auto 门控 miss；off_list 已取
    # 回时复用之省一次 D2H，语义零变化，否则 rev1 原样 prepare_token_indices）----
    key = (heads, dim_v, tail_dim, topk, kv_group, sm_scale, block_I, grid_size)
    func = _BUILD_CACHE.get(key)
    if func is None:
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
        _BUILD_CACHE[key] = func

    token_indices = (
        _token_indices_from_list(off_list, offsets.device)
        if off_list is not None
        else prepare_token_indices(offsets)  # 数据准备（新建张量，CPU→H2D）
    )
    return func(q, kv, topk_indices, lse, offsets, token_indices)
