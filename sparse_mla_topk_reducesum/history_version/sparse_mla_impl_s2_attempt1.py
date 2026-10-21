"""sparse_mla_topk_reducesum — DSA 稀疏注意力 topk 概率质量 head 归约（Ascend Expert 模式）。

varlen packed 布局下，对每个 token 行 s、每个 kv_group g、每个 topk 选中位置
idx = Indices[s_row, g, j]，计算 score = exp((Q·KV_main + Q_tail·KV_tail) * sm_scale - Lse)
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

支持域（编译期 assert，DESIGN §10.1 [DISPATCH-COVERAGE]）：
  dtype 固定组合 Q/KV=bfloat16、Lse/ReduceSum=float32、Indices/Offsets/TokenIndices=int32；
  dim/tail_dim 为 2 的幂；topk % block_I == 0；block_I 为偶数且 >= 32；
  head_kv = heads // kv_group 满足 (16 <= head_kv <= 64 且为 2 的幂) 或 (head_kv % 64 == 0)；
  padded_H != head_kv 时仅支持 kv_group == 1（继承原版 assert）。
  ΣS / ΣS_kv / batch 运行时任意（T.symbolic 动态维）。
"""

import tilelang
from tilelang import language as T

__all__ = [
    "tl_sparse_mla_topk_reducesum",
    "sparse_mla_topk_reducesum",
    "prepare_token_indices",
]


# ============================================================================
# Kernel builder（DESIGN §3.3 / §4.7 / §8.3 / §8.4）
# ============================================================================
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
    """构建 sparse_mla_topk_reducesum kernel（Expert 模式）。

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

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore  # [ΣS, heads, DDT] bf16
        KV: T.Tensor(kv_shape, dtype),  # type: ignore  # [ΣS_kv, G, DDT] bf16
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore  # [ΣS, G, topk] int32
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore  # [ΣS, heads] fp32
        Offsets: T.Tensor(offsets_shape, indices_dtype),  # type: ignore  # [B+1] int32
        TokenIndices: T.Tensor(token_indices_shape, indices_dtype),  # type: ignore  # [ΣS, 2] int32
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
            total_tasks = seq_len * REPLICATE_H * kv_group
            ntask = T.max(0, T.ceildiv(total_tasks - cid, grid_size))  # 本核有效任务数
            nblk = ntask * NI  # 本核扁平块总数

            with T.Scope("C"):
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
                if ntask > 0:
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
        block_I: topk 方向分块（偶数且 >= 32）
        grid_size: 缺省查询设备 cube_core_num

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

    token_indices = prepare_token_indices(offsets)  # 数据准备（新建张量，CPU→H2D）
    return func(q, kv, topk_indices, lse, offsets, token_indices)
