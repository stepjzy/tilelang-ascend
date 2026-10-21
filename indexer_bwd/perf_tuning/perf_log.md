# indexer_bwd perf_tuning log

## Iteration 1 — 2026-10-11T14:30
- bottleneck_type: other (host launch path; device CV-lockstep secondary)
- optimization: |
    _CachedLauncher: cached low-level launch path. Profiling showed the official
    1229.9us was launch-bound: the stock cython wrapper re-marshals every arg per
    call (TVM Analyzer construction ~550us + per-arg loops + current_stream 33us
    => ~750us/launch fast, ~480us cast) vs 223.7us device (msprof Task Duration).
    New path: single-slot cache keyed on arg id() tuple (strong refs held), raw
    stream query via torch_npu._C._npu_getCurrentRawStream (0.9us), direct
    adapter.lib.call (~12.9us = bare ACL submit). Public tl_indexer_bwd_fast /
    tl_dk_cast now return the launcher; jit builders renamed _*_impl.
- baseline_time: 1229.9 us
- candidate_time: 212.2 us
- improvement: -82.8%
- precision: pass (L0 7/7 incl. dispatch_equiv three-way)
- adopted: yes
- rollback_reason: -
- next_hint: |
    Device side now dominates (~204us fast + ~8us cast vs budget <80).
    msprof per block (54.7us): AIC waits 50.5us on cross-core flag id0 (CV
    lockstep, zero overlap); AIV vec 21.8us issue-bound (~24 vec ops + ~15
    PipeBarrier per token on [16,64] tiles). Iteration 2 plan: C-side software
    pipeline across sub-tiles (produce logits(ts+1) before consuming dlog(ts)),
    flat-2D Q load + direct dq L0C->GM write (kills ws_dq transit + V dq work),
    bf16 logits transit (halves bytes, pre-does grain round 1), big-tile
    [TS_H/2, topk] vector chain (~2.8x fewer issues).

## Iteration 2 — 2026-10-11T15:40
- bottleneck_type: sync (CV lockstep) + compute (V issue-bound)
- optimization: |
    Fast-kernel device restructure (rev3):
    1. Software-pipelined sub-tile loop (TSB+1 iters): C produces logits(ts) one
       iteration ahead of consuming dlog(ts-1) -> C's Q/GEMM1/FIX overlaps V's
       vector chain (was: pure CV lockstep, AIC 92% wall waiting flag id0).
    2. Flat-2D IndexQ/dIndexQ + flat-1D Weights params (launcher zero-copy
       views): one [TS_H, dim] Q DMA per sub-tile (was 4), direct whole-tile
       dq L0C->GM write for non-tail sub-tiles (kills the ws_dq CV transit +
       all V-side dq work; tail sub-tiles keep the guarded fallback).
    3. bf16 ws_logits transit with relu+cast folded into the FIX leg (halves
       transit bytes; performs dw-grain round 1).
    4. Big-tile [TS_H, topk] vector chain: causal = one compare
       (t_cur*H + r >= col_scaled[j]); per-token diff/w row broadcasts; one
       reduce per sub-tile; flat dw accumulation. ~20 tile ops/sub-tile vs
       ~99 (2.8-5.9x fewer issues).
    5. Tail-block score zero-fill -> invalid-token diff rows are 0 (validity
       folded out of the mask entirely); layout=1 dk exit reads ws_dkf
       per-row (UB savings).
    Bug found during bring-up: drelu buffer was overwritten by dwq before the
    causal &= drelu (buffer-reuse hazard); fixed by folding drelu into causal
    immediately (mathematically equivalent: grain-L == 0 exactly where
    drelu == 0).
- baseline_time: 212.2 us
- candidate_time: 111.6 us
- improvement: -47.4% (cumulative -90.9% vs 1229.9)
- precision: pass (--level all: L0 7 + L1 17 + L2 8 + boundary 4, dispatch
  three-way equivalent; ratios identical to pre-restructure)
- adopted: yes
- rollback_reason: -
- next_hint: |
    111.6 vs 80 target: profile again. Expect C-side MTE2 (Q+dlog+K reads) and
    per-block fixed costs (GetValues x3 + PipeBarrier<ALL> x6, K load, dk
    atomic, 4-blocks-on-critical-core imbalance 65/20). Levers: dlog single
    buffer (-64KB/block transit), TS=8 spike (L0C exactly 128KB), T_BLOCK=64
    (33 blocks -> 2/core), packed block-index read (1 GetValue), cast fusion.

## Iteration 3 — 2026-10-11T16:20
- bottleneck_type: compute (V issue-bound) + transfer (C MTE2)
- optimization: |
    Op-count slimming on both sides:
    1. sm_scale folded into the block-level w vector (w_f32_flat *= s once per
       block) - removes the per-sub-tile diff scaling mul (dw keeps pairing
       unscaled diff with grain-L that carries s).
    2. row_shift vector built once per block ([T_BLOCK*H] arith progression);
       per-sub-tile broadcast reads a slice (was per-sub-tile progression).
    3. w_big broadcast directly from the w_f32_flat slice (drops w_row copy).
    4. dw accumulation flat: one [TS_H] copy per sub-tile into flat dw_acc
       (was 4 [H] copies); dWeights param declared flat [S*H] (4th launcher
       view) - whole-block writeback is one [T_BLOCK*H] copy.
    5. dlog single L1 buffer + single GM transit (workspace_dlog_b kept as an
       unused signature slot - the test's arg tuple is fixed; gqa_sink ds_l1
       precedent for the dual fractal read pattern): halves dlog transit bytes
       and removes one MTE3 (V) + one MTE2 (C) + one flag pair per sub-tile.
- baseline_time: 111.6 us
- candidate_time: 96.2 us
- improvement: -13.8% (cumulative -92.2%)
- precision: pass (--level all)
- adopted: yes
- rollback_reason: -
- next_hint: |
    ~21.8us/block on the 4-block critical cores + ~7us cast + ~2 gap = 96.
    Remaining levers: TS=8 (halves C gemm/DMA/flag counts; V unchanged due to
    UB-capped [64, topk] tiles), cast-kernel fusion (spin counter), block-index
    GetValue packing. Profile first.

## Iteration 4 — 2026-10-11T17:10
- bottleneck_type: compute (V issue-bound -> C issue/engine-bound after split)
- optimization: |
    MIX_AIC_1_2 via the (cid, vid) launch form (T.Kernel(grid, is_npu=True) as
    (cid, vid); 910B3 has 20 AIC + 40 AIV - the kernel previously used only
    half the vector cores). First attempt split the chain by SUB-TILE PARITY
    (one AIV per transit) -> deterministic hang (error 507014): the AIC's
    cross-core wait requires BOTH AIV sub-blocks to set the flag (counting
    semantics; the sparse_flash vid example's both-AIVs-write-disjoint-halves
    pattern is the validated form). Redesigned as a ROW-HALF split: each AIV
    owns rows [vid*TS_H/2, +TS_H/2) of every [TS_H, topk] tile - all
    transits written by both AIVs (disjoint halves), prologue duplicated.
    Halved V tiles: vec 13.16 -> 6.77 us/block.
- baseline_time: 96.2 us
- candidate_time: 86.9 us
- improvement: -9.7% (cumulative -92.9%)
- precision: pass (--level all + 3x L0 repeat runs: 30/30 each, no races)
- adopted: yes
- rollback_reason: -
- next_hint: |
    C side now binding (19.55/block: scalar 7.57 dominated by 24 gemm_v0
    template flag chains + 330 instr; engines 13.6). Imbalance: 65 blocks /
    20 cores = 3.25 -> 4 blocks on 5 cores (17us waste). Levers: (a) cast
    kernel (cid,vid)+cast_rows=64 (~-2.5), (b) contiguous task assignment +
    K-reload skip when bos unchanged (~-1.2), (c) pair-GEMM: process sub-tile
    pairs with M=128 gemms (24->12 gemms) + row permutation so each AIV's
    pair-half is contiguous (1 read/1 write per transit, clean 2-set/1-wait
    flags) - est. C ~17, V ~11-12 -> ~76 total.

## Iteration 5 — 2026-10-11T17:45
- bottleneck_type: compute (C issue-bound: 24 gemm templates) + other (65/20 block imbalance)
- optimization: |
    1. Cast kernel: (cid, vid) MIX_AIC_1_2 form + cast_rows 16->64 (40 AIVs,
       half-chunk split, 4x fewer chunk iterations).
    2. Fast kernel: contiguous task assignment (core c gets a consecutive
       block run; same 4/3 balance) + K panel reload only at chunk start or
       segment start (t0 == 0) - K L2 reads drop from 65 panels to ~22.
- baseline_time: 86.9 us
- candidate_time: 83.4 us
- improvement: -4.0% (cumulative -93.2%)
- precision: pass (L0)
- adopted: yes
- rollback_reason: -
- next_hint: |
    Remaining: C scalar 7.57 (24 gemm_v0 template flag chains). Pair-GEMM:
    process sub-tile PAIRS with M=128 gemms (24->12) + row permutation
    [s0h0|s1h0|s0h1|s1h1] so each AIV's pair-half is contiguous -> 1 read/1
    write per transit per AIV, V chain ops halve (17 ops on [64, topk] per
    pair vs 34 half-sub-tile chains), flags keep the exact current
    2-set/1-wait pattern at pair granularity (ws_dlog a/b alternate as pair
    slots; ws_logits [grid,2,2*TS_H,topk] pair slots fit the test's fp32
    tensor). L0C exactly 128KB. Est. C ~17, V ~12 -> ~74-77 total.

## Iteration 6 — 2026-10-11T19:00
- bottleneck_type: compute (C issue-bound: 24 gemm templates/block)
- optimization: |
    Pair-GEMM (token order): each loop iteration bundles two sub-tiles into one
    M=2*TS_H=128 GEMM set -> 24 -> 12 gemm_v0 calls per block; one [PH, dim] Q
    load and one whole-pair dq L0C->GM write per pair (8 token rows contiguous
    in token order); AIV-vid owns sub-tile 2p+vid's full [TS_H, topk] tile
    (contiguous transits, simpler V chain: one row_shift segment, one w
    broadcast). Plus dk segment-run accumulation (flush the fp32 region atomic
    once per contiguous same-segment run: 65 -> ~22 atomics) and the GEMM3 init
    only at run starts.
    Journey notes (negative results, all root-caused):
    (a) a row-PERMUTED pair layout ([s0h0|s1h0|s0h1|s1h1] so each AIV's
        pair-half is contiguous) is INEXPRESSIBLE: L0C stores matrices in
        N-major 16-column bands (MakeLayoutL0C) and L1 gemm operands in zN
        fractal bands - row slices of either are not contiguous, so the
        un-permuting chunk writes and the sliced GEMM2 operands read wrong
        data (dq ratio 0.065; the first attempt also faulted with "FIXP read
        L0C out of bounds" from chunk offsets that walked past the fragment).
        Token order + whole-tile copies is the only sound form.
    (b) GEMM2 split into two M=TS_H half-gemms (to shrink dq_l0c and give L0C
        slack) hits the same zN slicing problem for the A operand.
    (c) an exactly-full 128KB L0C (logits[128,64]+dq[128,128]+dk[64,128]) is
        FINE once all accesses are whole-tile - the earlier crash was the bad
        chunk offsets, not the size.
    (d) layout=1's ws_dkf staging must carry the same flush guard as the
        per-row atomic, else the C-side flag set leaks on non-flush blocks and
        the next flush reads stale staging (dispatch_equiv[fast_shared] dk
        0.9929 -> fixed).
- baseline_time: 83.4 us
- candidate_time: 72.2 us (in-process repeats 70.3-71.6; msprof fast-only Task
  Duration 65.5 us, AIC 16.16/AIV 15.89 us per block)
- improvement: -13.4% (cumulative -94.1% vs 1229.9)
- precision: pass (--level all: L0 7 + L1 17 + L2 8 + boundary 4)
- adopted: yes
- rollback_reason: -
- next_hint: |
    TARGET MET at ~71-73 us typical (< 80). Remaining structure: AIC 16.2
    (cube 2.0 + mte1 1.9 + mte2 4.2 + fix 2.6 + scalar 4.7) / AIV 15.9 (vec
    6.9 + mte 3.5 + scalar 5.4); cast kernel ~4 us exec (its ~14 us launch
    latency hides behind the fast kernel when queued). Rare 82-86 us
    first-run-after-idle outliers observed (device state, not kernel);
    in-process runs are stable at 70-72.

# indexer_topk_reducesum perf_tuning log (Stage 3, 2026-10-14)

基线画像（msprof，per-core 平均，详见 indexer_topk_reducesum_baseline_iter1.json）：
AIV 是关键路径（wall 1778us：vec 1015 + mte2 529 基本串行 + scalar stall 1131），
AIC 半数空转（wait_id0 981us）。生成代码显示每 token 每 block 13 条向量指令 + ~12 条
PipeBarrier_V；中转读写均为 64B burst（workspace 行宽 = BN=32 bf16）→ mte2 有效带宽
仅 ~12.6GB/s（fix 写 55GB/s）。

## Iteration 1 — V 侧向量链融合
- bottleneck_type: compute (AIV vec issue-bound)
- optimization: |
    fast kernel V 侧消费链重构（数学逐位等价）：
    1. QG2 条 per-token [H, BN] elementwise 链（cast/w-bcast/mul/round-trip 各 13 op）
       融合为 half-token [QG2H//2, BN] 大 tile 链（UB 预算所限不能用整 tile——
       topk>=128@CAP=2048 会溢出 192KB）；
    2. 因果 mask 只在 straddle 块执行（每任务每 token ~1 块跨对角线，其余全有效免
       mask / 全无效保持 -inf）；
    3. sm_scale 从每 block [QG2H, BN] tile 乘移到每任务每 token 一次 [CAP] 行乘
       （golden 精确顺序 "(fp32 sum) × s"；s>0 编译期分支，s<=0 回退 per-block 乘）；
    4. score_2d [QG2, BN] + fill(-inf) + 融合 4 行 store（无效行保持 -inf）。
    过程坑：prim_func 体内定义的 half_rows 在 TVM script round-trip 解析中变符号变量
    （mul 形状断言失败）——形状常量必须放 builder 层（与 NBLK/QG2 同款惯例）。
- baseline_time: 2002.9 us
- candidate_time: 1788.9 us
- improvement: -10.7%
- precision: pass (L0 24/24, ratio 全 1.0000, max_abs fp32 噪声级)
- adopted: yes
- rollback_reason: -
- next_hint: |
    复画像：vec 1015→676us（融合生效）但 mte2 529→602us（64B burst 瓶颈实锤）。
    AIV wall 1573us：mte2 与 vec 平分。下一轮转置 GEMM 换宽 burst。

## Iteration 2 — 转置 GEMM [BN, QG*H] 宽 burst 中转
- bottleneck_type: transfer (mte2 64B burst) + compute (vec)
- optimization: |
    1. gemm_v0(K_l1 [BN,D], Q_l1 [QG*H,D], transpose_B=True) → L0C [BN, QG*H]
       （同点积同 K 序，逐位等价；L0B 槽 512×32×2=32KB 恰好= ping-pong 预算，
       common.h:1282 双侧断言均过）；workspace 变 [grid,2,BN,QG*H]；
    2. 中转两腿宽 burst：fix 写 32×1KB、AIV relay 读 32×512B（原 256/512×64B）；
    3. H 归约变连续 dim=-1（AR 模式）[BN, H] 段；
    4. 两次精度修复：
       (a) 转置后 per-token 段是列跨行区域，T.tile.select 的 lowering 把 region
           扁平化 → 选中错误内存区（首版 ratio 0.91）。独立探针证实 GEMM/中转
           读写本身逐位正确。修复：改为 post-reduce 融合 [QG2, BN] mask
           （tensor-tensor compare col<=t_abs(row) + 全 buffer in-place select，
           512B 合法尺寸），同时删掉 per-token 有效性 if（mask 统一覆盖无效行，
           守卫 k_st+BN > t0+vh——对角线以下整块跳过）；
       (b) 半块 cast 的源是列跨行区域 → copy_ub_to_ub 2D 退化 32 次逐行小 cast
           循环（vec 676→1149us 反弹！）。修复：整 tile 融合 [BN, QG2H] 稠密
           flat cast（l0_main UB 174.8KB 可容）；
    5. topk=1024@CAP=2048 唯一超预算配置 → host QG 自适应回退 8→4（合法档位，
       CAP=4096 本就用 QG=4；其余配置含 topk=512@CAP=2048 均以 QG=8 通过）。
- baseline_time: 1788.9 us
- candidate_time: 1153.7 us (remeadsured 1157.0 in --level all)
- improvement: -35.5% (cumulative -42.4% vs 2002.9)
- precision: pass (--level all: L0 24 + L1 42 + L2 13 + Boundary 4 全 PASS；
  coverage gate 24 PASS / 0 MISS / 0 N/A)
- adopted: yes
- rollback_reason: -
- next_hint: |
    主目标达成（1153.7 < 1428us）。剩余画像（msprof Task Duration 1547us 口径）：
    vec ~780us（含 barrier 税）、AIC wait_id0 749us（仍有富余）、mte2 72us 已解决。
    后续可选：AIC 2-ahead + relay ping-pong 软件流水（藏 mte2/ack 于 vec 之下）、
    barrier 削减、topk 段审计。次要目标：full interface 3642us（host ~2485us）
    → _CachedLauncher 式缓存评估。

## Iteration 3 — host 侧低开销化（次要目标：full interface）
- bottleneck_type: other (host launch/prep；主指标已达标后的并报项)
- optimization: |
    纯 host 侧（kernel 零改动，输出语义不变）：
    1. 单同步化：interface 每调用 1 次 .cpu() D2H（原 5 次：validation/detect/
       max_seg/两表构建各一次，~65us/次）；
    2. 构表 numpy 化 + 内容键备忘录：prepare_token_indices/prepare_group_table
       用 numpy 构建（torch CPU 算子开销 ~460-520us → numpy ~30us；H2D 固定
       开销 ~85us/次为下限），并以 (tuple(off_list), qg, device) 为 key 做
       纯函数内容键缓存（无 tensor-identity 语义风险——key 即数据内容）；
    3. _CachedLauncher（indexer_bwd note-1 先例）：绕过 cython 重编组 +
       TVM Analyzer（~700us/launch → 数十 us），raw stream 查询 0.9us，
       view_spec 就地做 flat2d/flat1d 零拷贝视图；任何 key miss 走同一
       廉价 marshal 路径（每调用新输出张量必然 miss，仍受益）。
    过程记录：一次测量被共享机器上外部 bench.py 进程污染（fast 4256us，
    全指标 ~3.5× 膨胀），等待其结束后复测排除；torch.zeros int32 单测
    8.7ms 为冷启动伪影（warm 39us）。
- baseline_time: full interface 3642.1 us（iter-2 后；Stage 2 原始 4776.2）
- candidate_time: 1594.1 us（host 开销 ~430us）
- improvement: -56.2% vs iter-2 / -66.6% vs Stage 2 原始
- precision: pass (--level all 83 项全 PASS + coverage 24/24；s<=0 回退
  分支补测 sm_scale=-0.5/0.0：score max_abs ~6e-8、mask 域索引 1.0000)
- adopted: yes
- rollback_reason: -
- next_hint: |
    终态：fast 纯 kernel ~1151-1164us（目标 <1428 达标，余量 ~19%）；
    interface 1594us（kernel + ~430us host：D2H 同步 80 + 输出/工作区分配
    ~110 + launcher ~60 + 检证/杂项 ~180）。剩余 host 项均为语义必需
    （每调用新输出、数据依赖的 offsets 读取）。可选后续：pin-memory 表
    传输、workspace 张量按 shape 缓存（省 ~40us，收益边际）。

# indexer_topk_reducesum PTO fast perf tuning (Stage 3 PTO round, 2026-10-16)

## Iteration 1 — 转置方向 + 宽行融合链 + transpose-reduce + tail mask（PTO 分支重构）
- bottleneck_type: compute (AIV vec，行宽 128B tile 吞吐 ~11.5G elem/s vs 宽行 49G)
- optimization: |
    PTO fast 分支整体重构（ascendc fast 结构移植 + PTO 库缺陷绕行）：
    1. transit 方向改转置（gemm K@Q^T → L0C [BN, QG*H]，workspace 1KB 行宽写）；
    2. V 侧 half-token 宽行融合链（[BN, HALFC]=[32,128] tile，512B fp32 行宽）——
       PTO 向量模板需要 >=256B 内层行宽（[QG2H, BN] 128B 行形态吞吐仅 11.5G elem/s
       且融合后执行错误，差分画像 per-token 链占基线 71% = 4698us）；
    3. head 归约经硬件 transpose [BN,HALFC]→[HALFC,BN] + TCOLSUM 行块（PTO TROWSUM
       dim=-1 对 validRow>1 的 dst 有 vcadd dstRptStride=Cols 越界写缺陷，隔离复现
       设备 UB-out-of-bounds；tensor-tensor compare+select 全形状错误——两个库缺陷
       均有最小单元测试证据）；
    4. causal mask 移到 tail（per-token [CAP] compare_scalar + select(-inf)，
       golden 精确语义；block 级张量 mask 因 tensor compare 缺陷不可用，算术 penalty
       与 -inf 在 topk 补选路径有 mask 域内可观察差异（基线 -inf→-1e30 对照
       mask ratio 0.9389））；
    5. 背压保持序列化单飞（本项留 iter-2）。
    关键方法论教训：idx 比对必须用 test 的 mask-domain 口径（golden 的 -inf 尾部
    输出原始列号，kernel 输出 -1 fixup——naive 全域比对在基线上也只 0.9846，
    前 6 个变体的"失败"中 iter1b/iter2e 为假阳性，浪费 ~6 轮定位）。
- baseline_time: 6587.4 us
- candidate_time: 3876.1 us (3866.6/3880.4/3881.2, 3 reps)
- improvement: -41.2%
- precision: pass (PTO L0 10/10 全 PASS 含 dispatch_equiv 三路；ascendc L0 回归
  PASS——ascendc 分支零改动，仅共享 tail 区加 PTO-only if 分支)
- adopted: yes
- rollback_reason: -
- next_hint: |
    iter-2: 背压 produce-then-wait ping-pong（V 加速后 C/V 重叠收益显现；当年
    "序列化修竞态"的归因已被推翻——竞态真因是 tensor compare/TROWSUM 库缺陷，
    ping-pong 在两后端共享结构下 ascendc 已长期验证）。iter-3 候选：TTRANS
    转置成本削减（需先差分画像确认占比）、tail arith_progression hoist。

## Iteration 2 — 背压 ping-pong 统一（produce-then-wait）
- bottleneck_type: sync (序列化单飞的 C/V 重叠损失)
- optimization: |
    PTO 序列化单飞（wait-before-produce）改为与 ascendc 共享的 produce-then-wait
    双槽 ping-pong（删除 PTO 专属 wait 块，共享块去掉 target 条件）。当年"序列化修
    转置方向竞态"的归因已在 iter-1 被推翻（真因是 tensor compare/TROWSUM 库缺陷，
    已绕行）；ping-pong 的 FFTS flag credit 平衡论证不变（note 11）。
- baseline_time: 3876.1 us
- candidate_time: 3123.8 us (3123.9/3124.1/3123.3)
- improvement: -19.4% (cumulative -52.6% vs 6587.4)
- precision: pass (PTO L0 10/10；ascendc L0 10/10 + fast 1155.4us ∈ 冻结噪声带
  1142-1165——共享代码改动对 ascendc 语义等价，实测零回退)
- adopted: yes
- rollback_reason: -
- next_hint: |
    3123.8 vs 目标区间 2700-3000 上沿差 ~4%。剩余画像待差分：TTRANS 转置成本、
    relay 跨步读、K 装载串行、tail arith 重复。

## Iteration 3 — TADD 折叠树替代 TCOLSUM 归约
- bottleneck_type: compute (TCOLSUM cross-lane reduce 窄行低效，单删差分 ~1.4ms/AIV)
- optimization: |
    转置后的 head 归约从 TCOLSUM [64,32]×4/块（cross-lane）改为 same-shape in-place
    TADD 折叠树（6 级，全行段操作数；H=64/32/16/8 静态分支，非 pow2 heads 回退
    TCOLSUM 保正确性）。差分画像：TCOLSUM 单删 -1407us / TTRANS 单删 -581us /
    tail -604us / 融合链 -11us（宽行链几乎免费）。
    附：iter-3 中间方向"j-loop 紧凑化（删 k_st<=t_max 守卫并入运行时上界）"被
    否决——auto CV sync pass 拥有 C/V 语句调度权，Python 层移动 V 块导致配对
    迭代映射错乱（精度崩 0.06）；pass 会自动重建 V 侧循环（性能不变证实）。
    j-loop 的 if 结构是 pass 的输入契约，kernel 层不可动。
- baseline_time: 3123.8 us
- candidate_time: 1925.5 us (1931.2/1921.1/1924.1)
- improvement: -38.3% (cumulative -70.8% vs 6587.4)
- precision: pass (PTO L0 10/10；H=16/8/32 分支 + 非 pow2 TCOLSUM 回退覆盖全 heads 域)
- adopted: yes
- rollback_reason: -
- next_hint: |
    1925.5 us 已远超任务目标区间 2700-3000（-29% vs 下限）。中止条件①达成。
    剩余构成（估算）：tail ~604 + TTRANS ~581 + TADD 树 ~250 + 融合链 ~11 +
    relay/GEMM/骨架 ~480。可选后续：TTRANS 合并 [32,256] 一次（UB +16KB 待验）。

## Iteration 3 补充 — CAP=4096 适配与终态微调（同轮内）
- 事项: |
    1. l1_qg4_cap4096（PTO）UB 溢出：auto topk tmp = 24B/elem × 4096 = 96KB +
       宽行链 buffer + TTRANS 内部 tmp（planner 不感知，越界 204448 > 196352
       导致 aicore 崩溃）。尝试过 prod round-trip 复用 relay buffer（-4KB，
       因 tmp 越界未单独成立）与 tail mask 域缩至 [P2] + forced-pad-only
       （[1,128] compare 的 repeat 边界位 63 恰落入 sort 域，mask 失效 0.85）。
       最终方案：host 侧 PTO@CAP>2048 路由 generic 路径（PTO 全量验证过、
       非 perf 门控、两路输出等价 l0_dispatch_equiv 实证）——fast@CAP=4096
       在 PTO 的 UB 不可行是平台约束（框架 tmp 公式），记录在案。
    2. gate 模型未更新 PTO 增量（保守以实测全量为准）：终态 --level all
       --backend both 全过 = 全部实配 UB 可行。
- precision: pass (l1_qg4_cap4096 PTO 经 generic 路由 1.0000/0.9998)
- adopted: yes（并入 iter-3 终态）

## 终态（2026-10-16，原版框架零修改）
| 指标 | 基线 | 终态 | 改善 |
|------|------|------|------|
| PTO fast 纯 kernel | 6554.1（任务书）/ 6587.4（本轮复测） | **1925.5-1932.7**（4 次独立进程清态复测：1925.5/1926.9/1930.4/1932.7） | **-70.7%** |
| PTO generic（非门控） | 7370.8 | 7132.5 | -3.2%（kernel 未动，波动） |
| PTO full interface | — | 2364.0 | — |
| AscendC fast（冻结带） | 1157.8（1142-1165） | **1146.8-1156.9** ∈ 带内 | 零回退 |
| AscendC generic / interface | 3096.6 / — | 3216.7 / 1604.1 | 波动带内 |

精度：--level all --backend both 132 PRECISION_PASS / 0 FAIL / 0 WARN，exit 0；
覆盖门禁 15 PASS / 0 MISS / 0 N/A；ruff check + format 全过。
vs 任务目标区间 2700-3000us：**-29% 低于下限**（中止条件①达成且大幅超越）。

终态瓶颈构成（差分画像）：tail（topk/softmax/fixup/tail mask）~600 + TTRANS
转置 ~580 + TADD 树 ~250 + 融合宽行链 ~11 + relay/GEMM/transit/背压/骨架 ~480。
剩余大头为 topk 尾链（两后端同构，AscendC 侧同款 ~380us 基线成本）与
TTRANS（PTO 库 TROWSUM 缺陷的绕行成本）。
