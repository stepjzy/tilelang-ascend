# debug_log: fla_chunk_delta (Stage 2)

## Attempt 1 — 2026-11-11T14:10

- mode: first_impl
- classification: precision_pass (pending full-suite confirmation, see below)
- design: DESIGN_fla_chunk_delta.md (rev 0), Stage 1 passed gate
- test_level: l0 (gate passed r9) -> all (running)
- rollback: no
- instrumentation_cleaned: n/a

### Architecture (Developer mode, threads=2, fixed-core task loop + relay)

- CORE_EFF = min(20, NCHAIN) with striped task map (tid = cid + k_task*CORE_EFF)
  and `if tid < NCHAIN` tail guard; T.serial chunk recurrence; NSLOT=1
  lockstep; 4 (fwd) / 7 (bwd) explicit workspace_* relays; AUTO_CV_SYNC
  cross-core flags (verified in the generated source).

### DEVIATIONS from DESIGN SS3.3 (all implementation-level, verified by source
### analysis + generated-source audits; the design's API mapping was written
### for threads=1 semantics and several primitives are not vid-safe)

1. **g gather chain -> flat span + T.serial scalar extraction loop.**
   `ascend_vid_reduction.cc IsTileOp` (lines 563-586) only rewrites an
   allow-list; `T.tile.gather` / `arith_progression` / `reinterpretcast` are
   absent and corrupt silently under threads=2 (fla_wy_fast #4 record).
   Additionally the parallel-loop vectorizer (AscendLowerParallelToVector)
   DROPS the i*H stride of the span load and emits a misaligned VEC read
   (measured: AI Core Error "UB address accessed by the VEC instruction is
   not aligned", core id 4) -- T.serial lowers to scalar ops and the vid pass
   still halves its extent (loop var in the vid-reduced first dim).
2. **T.tile.broadcast row gating -> T.tile.row_expand_mul_experiment**
   (AscendC brcb + mul_mask). Shapes flow through access_ptr extents so the
   vid halving stays self-consistent; requires 256B fp32 rows, so the bwd
   q~ gating (DK=128) runs on two [BS, 64] column chunks.
3. **g_last: [1] UB staging via a scalar store** (codegen
   VisitStmt_(BufferStoreNode) flattens ALL indices through OffsetOf). The
   tile-op scalar path (EmitSelectedRawScalar -> buffer_1.access_ptr +
   indices_1[0]) only flattens the FIRST index: a multi-dim GM scalar load
   would silently misread (measured: kernel qdo == golden x8 probe confirmed
   the arithmetic, the golden itself had the bug below). The [1] buffer is
   allocated with T.alloc_ub: InferAllocScope cannot resolve a buffer that
   only participates in scalar statements (codegen error "unresolved dynamic
   storage scope").
4. **1D tile-op mask staleness**: plain 1D tile ops (repeat=1 + MASK_PLACEHOLDER)
   never repair their own vector mask. A [BS]-chain (32 lanes/vid) followed
   by a [DK]-chain (64 elements/vid, DK=128) halves the DK op silently
   (fwd_repr ratio 0.9998 / max_abs 0.5, all in the stale half). Fix: every
   tile exp is preceded by a same-sized exact-contract `T.tile.fill` (the
   fill resets the mask/count; the following scalar loop fully overwrites
   the buffer so the fill value is dead). Generated source verified:
   fill(count=64) -> SetVectorMask(0, 0xffffffffffffffffULL) -> Exp.
5. **Golden transcription bug in DESIGN SS9.1 (bwd)**: the SS9.1 code block
   applies the gk per-channel decay AFTER the qdo/wdv2 update; the frozen
   upstream (ref_gpu/fla_chunk_delta.py lines 404-412) and DESIGN SS1.3/SS3.2
   apply it BEFORE the update (the update terms are NOT gk-decayed). Fixed in
   the test golden to match the upstream order; cross-validated 0.0 max diff
   against an independent token-level fp64 recursion. The kernel's folded
   form dH = D*dH + qdo*scale - wdv2 (D = exp(g_last + gk_last)) matches the
   upstream order exactly.
6. **Vnew/Dv2 declared as 3D [B,S,H*DV] views** (wrapper .view()s back to 4D):
   with 4D [B,S,H,DV] tensors the vid row offset lands on the H index
   (chunk_o #8) -- and my initial 4D-with-flat-column-slicing was a plain
   region-extent mismatch (first-round parse error).
7. **bwd workspace count 7, not 4**: the design's SS4.3 list missed the
   dv2_16 V->C relay operand for the wdv2 gemm; and the q~ relay is split
   into workspace_3/workspace_3b (one V write site each) because AUTO_CV_SYNC
   requires per-workspace site parity ("Mismatch in sync points between cube
   and vec for workspace workspace_3: cube has 1, vec has 2").
8. **Test input conditioning (L1)**: plain randn w makes the recurrence's
   error feedback (v' = u - w@H, per-chunk gain ~ sigma(k^T)*sigma(w)) amplify
   any legitimate fp32-vs-fp64 difference (~1e-5: HW exp accuracy,
   accumulation order) by ~30-90x per chunk -- measured error growth
   0.001 -> 0.0156 -> 0.5 over 4 chunks. k/w(/q) are L2-normalized per token
   (gdn_chunk_h.py:195 family precedent); the real operator consumes
   WY-constructed w (w@H ~ u by construction). A pure-decay probe
   (u=k=w=0, mild decays) confirmed the kernel's decay chain is exact to
   1.5e-8 relative -- the residual was input conditioning, not a kernel bug.

### Domain restriction (wrapper assert, documented)

DK in {64, 128} (row_expand 248-row cap + 64-column gate chunks + the
offset!=0 vid-region trap forbidding DK>128 chunked forms). DK 192/256 and
non-64-multiple D remain L1-extension territory per DESIGN SS5.4 (out of the
L0/bench domain; the L2 negative tests pin the boundary).

### L0 gate results (round r9, all six DESIGN SS9.2 cases)

[PRECISION_PASS] l0_fwd_min  (1,128,2,64,64)   h:1.0000/1.221e-04 v_new:1.0000/9.766e-04
[PRECISION_PASS] l0_fwd_repr (2,256,4,128,128) h:1.0000/2.441e-04 v_new:1.0000/1.953e-03
[PRECISION_PASS] l0_fwd_opts (1,256,2,128,128) h:1.0000/2.441e-04 ht:1.0000/2.980e-08
[PRECISION_PASS] l0_bwd_min  (1,128,2,64,64)   dh:1.0000/0.000e+00 dv2:1.0000/0.000e+00
[PRECISION_PASS] l0_bwd_repr (2,256,4,128,128) dh:1.0000/3.906e-03 dv2:1.0000/9.766e-04
[PRECISION_PASS] l0_bwd_opts (1,256,2,128,128) dh:1.0000/6.104e-05 dv2:1.0000/0.000e+00 dh0:1.0000/2.384e-07
"Test Passed!" exit 0

### Coverage gate (coverage_check.py --proto proto_fla_chunk_delta.yaml)

23 PASS / 0 MISS / 0 N/A -> PASS (exit 0). Category Fusion; non-aligned
shape scenarios (D-SHAPE-TAIL-1/TAIL-MID/PRIME) covered through their
out-of-domain L2 rejection cases; L1 additionally covers the task-tail guard
path (NCHAIN=22/23 with CORE_EFF=20 -- every L0 case divides evenly, so the
`if tid < NCHAIN` skip path was otherwise untested).

### Empirical risk ledger (resolved)

- R-a GM scalar load (GFlat staging): WORKS (GetValue, full OffsetOf flatten).
- R-b row_expand under auto vid: WORKS (brcb+mul_mask explicit masks).
- R-c task-tail guard + relay flags: WORKS (L1 tasktail cases).
- R-d T.serial extraction loop with flat indexing: WORKS.
- R-e fragment->workspace copies: WORKS.
- R-f 5D Hout/Dh V-side writes: WORKS (vid offset on the DK dim).
- R-g serial recurrence + auto flags lockstep (DESIGN R2): WORKS, no
  deadlock across NT up to 4 (bench NT=128 pending in the full run).

### Toolchain findings worth upstreaming (skill-journal candidates)

1. AscendLowerParallelToVector drops strided (i*H+bh) loads -> misaligned VEC.
2. 1D tile ops never repair their vector mask across size-class changes;
   same-sized fills are an effective (if accidental) pin.
3. EmitSelectedRawScalar flattens only indices_1[0] for buffer-load scalars.
4. InferAllocScope leaves scalar-only buffers unresolved ("shared").
5. Scalar tir.exp has no Ascend lowering ("Unresolved call Op(tir.exp)").
6. Buffer allocs inside Python-level `if` in @T.prim_func are not registered
   by the parser env ("Undefined variable").

(changes/summary finalized after the full-suite run)

### Full-suite results (round r10, --level all --bench, exit 0, "Test Passed!")

- L0: 6/6 [PRECISION_PASS] (see the L0 gate results above)
- L1: 10/10 [PRECISION_PASS] -- incl. the task-tail guard path (NCHAIN=22/23
  over CORE_EFF=20), exp2/scale/nosave/final-state params, value ranges
  (many cases exact 0.0; worst max_abs 7.8e-03 on bf16 dh)
- L2: 10/10 [BOUNDARY_PASS] (all out-of-domain shapes/dtypes/chunk_size
  correctly rejected by the wrapper asserts / adapter dtype checks)
- Boundary: 5/5 [BOUNDARY_PASS] (zero inputs, exp-underflow g (-87),
  sparse inf / sparse nan in u, fp16 x200 value scaling -- all exact 0.0;
  inf/nan compared positionally by the check_precision structural masks)
- Coverage gate: 23 PASS / 0 MISS / 0 N/A (exit 0)

### Perf sentinel baseline (DESIGN SS9.4 bench shapes, do_bench warmup=20
### rep=10 mean, device-side NPU events, wrapper defaults)

- perf_fwd_bench (1,8192,64,128,128) fp16: fwd = 7020.5 us
- perf_bwd_bench (1,8192,64,128,128) bf16: bwd = 9160.0 us
- fwd+bwd total = 16180.5 us (Stage-3 acceptance target < 3400 us)

Notes for Stage 3: the design's lockstep expectation (4-6 ms) assumed the
VEC gather/broadcast gating paths; the mask-safe substitutions above leave
two known overheads visible in the generated source -- (a) the T.serial
scalar extraction/decay loops carry per-iteration MTE2_S/S_V flag pairs +
PIPE_ALL barriers (auto-sync conservatism around scalar statements), and
(b) each exp pays a same-sized fill. The natural Stage-3 moves are the
DESIGN SS6.3 ladder (NSLOT ring + batched chunks + throttle flags) plus
re-vectorizing the g-chains (e.g., a threads=1 gather prep kernel
transposing g to [B,H,S] once per launch, removing the per-chunk
extraction entirely) -- recorded here as next_hint, NOT applied (Stage 3
owns optimization).

### classification: precision_pass
### boundary_warnings: none
### fail_category: none
### test_level: all (+ bench)
### coverage: L0:6 L1:10 L2:10 Boundary:5

## Attempt 2 — 2026-11-11T18:40 (Stage 3 perf tuning)

- mode: stage3_perf
- classification: perf_partial (precision maintained; target NOT reached, structural means exhausted)
- design: DESIGN_fla_chunk_delta.md (SS6.3 ladder superseded by the measured evidence below)
- test_level: all (+ bench, final job exit 0, max_abs identical to the Stage-2 record)
- rollback: partial (3-chain fwd, bwd BV=128, full-width q~ staging reverted with root causes)

### Result: 16180.5 -> ~7148 us (-55.9%); target 3400 us NOT reached

Adopted (in order): threads=1 prep kernel materializing GT/GE/GD gate/decay
tiles (killed the scalar extraction loops, 79-84% of the AIV scalar pipe,
-38%); NSLOT=16 ring; 2-chain statement-order batch (V {pro,pro,c,c,f,f} /
C {b,b,e,e}, Select-redirect for OOB slots, -25%); relay MTE3 queue
ordering (ws1/ws3/ws4 ahead of the HBM snapshot writes, fwd -11%);
single-chain fwd + BV=128 when DV%128==0 (row_expand 64-column halves,
fwd -4.5%); bwd qdo/wdv2 shared readback + time-shared L0C accumulator.

Rejected (with evidence): fwd 3-chain batch (+3.6%, tail redirect waste);
bwd BV=128 (L0C 160KB > 128KB -> nan); full-width q~ staging via
column-region UB writes (nan on every DK=128 case -- column-sliced UB
region writes corrupt under vid splitting while ROW-region halves are
safe); q~-to-prep externalization (analysis: +270MB HBM + prep per-head
statement loop net ~0).

### Stage-3 toolchain findings (skill-journal candidates)

1. AUTO_SYNC inserts a full PipeBarrier<PIPE_V> between EVERY consecutive
   V-op chain node (Sub -> brcb -> mul_mask -> cast: 4 barriers) -- the
   per-statement S-pipe cost is ~0.10us in blocked structures vs the
   ~0.04-0.05us chunk_o achieves with a never-blocking stream. The
   barriers are the dominant remaining cost (AIC/AIV scalar pipes at
   64-67% of wall in the final msprof).
2. vid splitting + column-sliced UB region writes = silent corruption
   (q16_ub[0:BS, 64:DK] destinations); row-region halves of [*, 128] fp32
   tiles work for cross-buffer row_expand but in-place (dst==src0)
   512B-stride regions corrupt (bwd dvp gate, ratio exactly 0.25).
3. L0C fragments do NOT get lifetime-planned away: 3 live [*,128] fp32
   accumulators = 160KB > the 128KB L0C compiles silently and produces nan;
   time-sharing one fragment per (produced->drained) pair fixes it.
4. row_expand's physical-stride requirement: region operands of a
   [R, 2*64] fp32 buffer are accepted at API level but the mul_mask
   emitter's 256B-row windowing must be respected (fwd halves PASS only
   as dst!=src0 cross-buffer regions).
5. The vid GM-region offset rule (region_rank - ub_rank) also governs the
   row_expand 1D [BS]/[DK] gate reads: the GT/GD span reads split on the
   token/channel dim, never on the head index (the W3/Hout pattern).
6. cross-core flag counter capacity ~15 sets bounds the in-flight
   handshakes; the natural single-chain V-lead (1 chunk) stays far inside.

### Floor accounting (why 3400 is out of kernel-level reach)

fwd main ~2.4ms = 410 chunks x (16 mandatory statements x ~0.10us issue +
2 serial cross-core round trips x ~0.7us); bwd main ~4.5ms = 819 x (27 x
0.10 + 3 x 0.7). The serial-recurrence round trips are true data
dependencies (H_{t+1} consumes the Cube's gemm of iteration t) -- chunk_o's
never-blocking stream requires task independence this operator does not
have. Remaining levers are toolchain-level (fine-grained intra-core sync)
or algorithm-level, not kernel-level. Full evidence chain:
perf_tuning/fla_chunk_delta_final_summary.md + fla_chunk_delta_prof_final/.

### Attempt 3 — 2026-11-11 (Stage 3 round-2 deep-dive perf tuning)

- mode: stage3_perf (round 2, evidence-driven; user overruled round-1's exhaustion verdict)
- classification: perf_partial (round-2 -14.5%, cumulative -62.5%; target 3400 NOT reached;
  kernel-level means exhausted WITH a complete evidence chain -- see
  perf_tuning/fla_chunk_delta_final_summary.md round-2 section)
- test_level: all (+ bench x2, terminal job exit 0, max_abs identical to Stage-2 records)

### Result: 7098.1 -> 6069.9 us (round-2 -14.5%); cumulative 16180.5 -> 6068 (-62.5%)

Adopted: bwd Select-storm removal + guarded single-chain tail batches (condval if/else
inlined into EVERY buffer index of EVERY statement, 68 AIV + 24 AIC blocks/iteration;
provably-in-bounds NFULL = NCHAIN//(2*CORE_EFF) full batches with plain arithmetic
indices; remainder as `if tidt < NCHAIN` single-chain tails that idle cores skip
entirely -- bwd -13.8%); bwd q~ externalization to the prep kernel (bit-identical
arithmetic chain, -13 V-statements/chain-chunk and 2 relay handshakes, C-side single
[BS,DK] QT load -- bwd -7.5% same-session).

Rejected (with evidence): qdo_scale fold into QT (golden quantizes qt=bf16(q*e^g)
UNSCALED -- measured l0_bwd_opts dh0 ratio 0.9601 FAIL, feedback amplification
2.4e-07 -> 2.3e-03); ws5/ws6 L0C accumulate merge (algebra requires that fold);
fwd 2-chain BV=128 (aicore exception 507015: UB demand 284KB/AVID > 196KB forces the
memory planner's lifetime-overlap fallback and AscendSyncInsert MISSES the cross-pipe
WAR on overlapped different-name buffers -- h16_ub_b and vn_ub_b both at UB offset
90496; measured crash + full UB arithmetic); prologue load hoist + scratch name-splits
(-1.0% noise, rolled back, but the census proved the bwd PIPE_V count drops 25 -> 14
with NO wall change); parallel-scan restructure (compose gemm volume x3-4 on an
already latency-bound kernel -- analysis).

### Stage-3 round-2 toolchain findings (skill-journal candidates)

1. **Select/IfThenElse index expressions are inlined per statement per iteration**:
   T.if_then_else in a chain-index position generates condval if/else chains before
   EVERY tl-op in EVERY loop iteration (AIC + AIV both) -- ~90 scalar branch issues
   per iteration at the bwd shape. When a batch decomposition can be made provably
   in-bounds, use plain arithmetic indices and move the remainder into guarded tail
   batches (the guard folds to a per-core cid comparison in the generated C).
2. **Zero-extent T.serial loops break the memory planner**: a T.serial(0) loop body
   keeps its buffer references in the IR but the planner's address_map omits them ->
   CodeGenTileLangAscend "Cannot find pre-allocated address for buffer". Wrap such
   loops in a Python-const `if EXTENT > 0:` (folds away pre-planner).
3. **The memory planner's lifetime-overlap fallback is UNSAFE with AUTO_SYNC** when
   UB demand exceeds the budget: AscendSyncInsert's dependency analysis is
   name-and-physical-overlap based but MISSES cross-pipe WAR/WAW between
   different-name buffers that the planner overlapped (measured: aicore exception
   507015, fwd 2-chain h16_ub_b/vn_ub_b same-offset race). Keep UB demand under the
   196KB/AIV budget or share by NAME only (same-name WAR gets correct event pairs).
4. **Golden quantization points gate operand folds**: the test goldens pin the exact
   bf16/fp16 cast points (qt = bf16(q*e^g) unscaled; vn16 = (vn*gate).to(fp16)).
   Any optimization that moves a scale/gate across a golden cast point diverges by a
   full operand rounding, and the bwd recurrence's feedback loop (dH -> dv2 -> wdv2
   -> dH) amplifies it past the ratio gate. Folds must keep the golden's cast points
   bit-identical (the adopted q~ externalization does).
5. **PIPE_V barrier drains are ~free on this workload**: removing 11 of 25 per-iteration
   PIPE_V barriers (via per-half/per-chain scratch name-splits -- same-pipe deps are
   NAME-based in AscendSyncInsert) measured NEUTRAL. Round-1's ~0.10us/stmt
   "auto-sync cost" was an upper bound conflating issue + stalls; the remaining wall
   is cross-core round-trip latency (terminal msprof: both sides 41-51% mutual wait).
6. **threads=1 prep can externalize per-(b,chunk,head) gated operands cheaply**: the
   transposed gate staging (gt_t rows) feeds row_expand directly per head; the
   two-step UB cast chain keeps the golden's quantization point bit-identical
   (bf16 -> fp32 -> fp32 row-gate -> bf16).

---

## 交付后修正（2026-11-13）：CI smoke main 补齐

- 分类：交付缺陷（非 kernel 缺陷）
- 现象：用户指出 fla_chunk_delta.py 无 `__main__` 入口且未登记
  ci/operator_test_manifest.yaml——examples/bench_test.sh 全量收集
  examples_experiment/**/*.py 后按 "TEST PASSED!/KERNEL OUTPUT MATCH"
  判定，无 main 的算子文件被判 [FAILED]，CI 会挂。
- 修正：对齐已合入先例（example_fla_chunk_o.py：smoke main + 不登记
  manifest——登记需改 ci/operator_test_manifest.yaml，违反"PR 只放 ≤2
  交付文件"约束），补 L0 门 shape 的 smoke main（fwd fp16 + bwd bf16 全
  wrapper 调用，shape/dtype/finite 断言，[SMOKE_PASS]/Test Passed! 输出）。
- 踩坑：首版 smoke 把 fwd 域 fp16 的 k/w 直接喂给 bwd（bf16 域），Cython
  wrapper 报 "Buffer dtype mismatch for parameter K3: expected
  torch.bfloat16, got torch.float16"——两域输入必须分开构造
  （kb/wb = k/w.to(bfloat16)）。
- 验证：CI 同构执行（cd <dir> && python fla_chunk_delta.py）job
  01a1d28c：[SMOKE_PASS] + Test Passed!，exit 0；ruff format/check 双过。
- 交付分支 amend + --force-with-lease：77b06195 → e7df98af。
