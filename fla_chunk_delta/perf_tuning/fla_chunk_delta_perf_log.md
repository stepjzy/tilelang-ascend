# fla_chunk_delta perf log (Stage 3)

Baseline (Stage 2, 2026-11-11): fwd 7020.5us / bwd 9160.0us / total 16180.5us
Target: fwd+bwd < 3400us (4.76x reduction needed)
Shape: B=1,S=8192,H=64,DK=128,DV=128,cs=64; fwd fp16 / bwd bf16, g+gk on, wrapper defaults
Timing: do_bench warmup=20 rep=10 mean device-side NPU events (--bench sentinel)

Known suspects (debug_log next_hint):
1. T.serial scalar extraction loops (BS=64 / DK=128 iters) with per-iteration flag pairs + PIPE_ALL barriers
2. exp-prefilled fills (vector-mask pinning workaround)
3. NSLOT=1 lockstep relay handshakes (~470ns/flag-wait x 4-7 relays/chunk)
Suggested directions: threads=1 gather prep kernel / relay ring / 3-chunk batching / throttle flags

DESIGN roofline: bandwidth floor ~1.61ms (2.04GB @ 1263GB/s) -> target feasible;
true bottleneck = serial-chain small-GEMM latency; 0.83us/gemm needed for 3.4ms;
chunk_o verified toolkit: gemm merge (built-in) + chain interleave (built-in) +
ring + 3-chunk batch + throttle flag (chunk_o measured -57%, NOT yet applied).

Iteration log entries appended below (one per round, Attempt format).

## Iteration 1 — 2026-11-11T15:00 (baseline profiling round)
- bottleneck_type: sync (scalar-pipe serialization from T.serial extraction loops + lockstep relays)
- optimization: none (profiling only; msprof op fwd/bwd + app-level + source dump)
- baseline_time: fwd 7020.5 / bwd 9160.0 / total 16180.5 us
- candidate_time: n/a
- improvement: 0
- precision: pass (Stage 2 carry-over)
- adopted: n/a
- rollback_reason: n/a
- next_hint: msprof evidence: AIV scalar pipe 79%/84% of wall (4906/6972us) = scalar extraction loops
  (192 iters/chunk, per-iter MTE2_S/V_S/S_V flag pairs, ~33ns/iter, source-verified);
  AIC 98% waiting on V-side relay flags. Artifacts: perf_tuning/fla_chunk_delta_prof_r1/

## Iteration 2 — 2026-11-11T15:10 (prep kernel; ADOPTED)
- bottleneck_type: compute->sync transition (scalar loops removed; lockstep relays exposed)
- optimization: threads=1 prep kernel materializing GT/GE/GD gate/decay tiles
  (fp32 hw transpose [64,64], row_expand_sub direct-difference form, HP=ceil(H/8)*8 staging);
  main kernels: 1D [BS]/[DK] gate/dst targets (vid offset lands on token/channel Range dim),
  contiguous GT/GD span reads, zero scalar statements in hot loops (source-verified)
- baseline_time: fwd 7020.5 / bwd 9160.0 / total 16180.5 us
- candidate_time: fwd 4219.4 / bwd 5782.6 / total 10002.0 us (bench, same session L0 pass)
  [main-only msprof: fwd 3765.5 / bwd 5694.0; prep 17.9/22.1us]
- improvement: fwd -39.9% / bwd -36.9% / total -38.2%
- precision: pass (L0 6/6 same session; ratios 1.0000)
- adopted: yes
- rollback_reason: n/a
- next_hint: AIC now 98% wall waiting on relay flags (fwd wait_id0+id2 3314us): NSLOT=1 lockstep
  serializes cross-chain WAR. Round 3: NSLOT=16 ring, slot=(k_task*NT+i_t)%NSLOT, no manual
  throttle (natural V-lead <= in-flight chunks <= 7 chains/core < 15-set flag capacity;
  ring safety 16 > 2*7+1). Artifacts: perf_tuning/fla_chunk_delta_prof_r2/

## Iteration 3 — 2026-11-11T15:40 (NSLOT=16 ring alone; NO GAIN, rolled forward as base)
- bottleneck_type: sync (C/V statement-order serialization: V-side S-pipe blocks on ws2/ws4
  cross-core waits; cannot issue next chain's prologue during the wait)
- optimization: NSLOT=1 -> 16 relay ring, slot=(k_task*NT+i_t)%NSLOT (cross-chain WAR decoupled;
  ring safety 16 > 2*7+1 without manual throttle; L0 6/6 pass, no deadlock)
- baseline_time: fwd 4219.4 / bwd 5782.6 / total 10002.0 us (iter2 adopted base)
- candidate_time: fwd 4221.4 / bwd 5800.4 / total 10021.8 us (bench same session; main-only
  msprof fwd 3709.7 vs 3765.5 (-1.5%), bwd 5695.2 vs 5694.0 (flat))
- improvement: ~0 (within 3% noise) -> NOT ADOPTED as an independent win
- precision: pass (L0 6/6)
- adopted: yes (kept as base for the batch restructure: the ring is the enabling condition
  for cross-chain statement reordering -- without it chain B's ws1 write WARs chain A's
  pending C-side read. Recorded here per the "one variable" discipline.)
- rollback_reason: n/a (neutral perf, kept structurally)
- next_hint: C/V overlap is ~zero: per-chunk wall 4.6us vs serialized work sum 4.2us.
  The V-side S-pipe program order (per-chunk a->c->d->f with two blocking cross-core waits)
  forbids issuing chain B's prologue during chain A's waits. Round 4: 2-chain batch with
  source-order restructure (a_A,a_B,b_A,b_B,c_A,c_B,e_A,e_B,f_A,f_B) so the auto C/V split
  yields V order {a_A,a_B,c_A,c_B,f_A,f_B} and C order {b_A,b_B,e_A,e_B}; dual-resident
  state buffers (h_ub x2); operand buffers naturally paired by chain. Chunk_o precedent:
  same-name inter-iteration WAR forces M_MTE2 serialization; paired names restore overlap.

## Iteration 4 — 2026-11-11T16:20 (2-chain batch + phase reorder; ADOPTED)
- bottleneck_type: sync (per-chunk C<->V ping-pong, ~zero C/V overlap)
- optimization: 2-chain batches with source-order phase restructure: V order
  {pro_A, pro_B, c_A, c_B, f_A, f_B}, C order {b_A, b_B, e_A, e_B}; bwd hoists the
  ws2-independent q~ gating into the V prologue and splits the C e-phase into
  e1(qdo)/e2(wdv2); Select-redirect (branch-free) for OOB chain slots (keeps relay
  flags balanced, avoids the AUTO_SYNC PIPE_ALL drain around IfThenElse); batch-
  internal buffers paired per chain (state/prologue staging/L1 operands/L0C frags);
  batch-internal ring slots (2*i_t+jb)%NSLOT (the iter3 flat map degenerated:
  NT=128 % 16 == 0 made all chains share the same slot at the same chunk -- why
  iter3 measured neutral)
- baseline_time: fwd 4221.4 / bwd 5800.4 / total 10021.8 us (iter3 base)
- candidate_time: fwd 2971.1 / bwd 4554.3 / total 7525.4 us (bench; main-only msprof
  fwd 2593.4 vs 3709.7 (-30.1%), bwd 4488.3 vs 5695.2 (-21.2%))
- improvement: fwd -29.6% / bwd -21.5% / total -24.9% (cumulative from Stage-2: -53.5%)
- precision: pass (L0 6/6 same session)
- adopted: yes
- rollback_reason: n/a
- next_hint: msprof r4: both sides still ~full-wall waiting in cascade (fwd AIC
  wait_id0 1132us = ws1 flag late: the V prologue's MTE3 queue has the 16KB HBM
  Hout write AHEAD of the 16KB L2 ws1 write, delaying the Cube's start; bwd same
  with the Dh write). V/C S-pipe statement-issue overhead now significant (fwd
  48 stmts/batch, bwd 80). Next: (a) reorder prologue MTE3 (ws1 first, Hout/Dh
  last); (b) fwd 3-chain batch (amortize the 2 cross-core round-trips per batch
  over 3 chunks; UB fits ~139KB/vid) while bwd stays 2-chain (UB-capped);
  (c) bwd ws5/ws6 merge to cut one relay handshake.

## Iteration 5 — 2026-11-11T17:00 (relay MTE3 queue ordering; ADOPTED)
- bottleneck_type: sync (Cube start delayed behind HBM writes in the V prologue's MTE3 queue)
- optimization: ws1 relay write FIRST in the prologue MTE3 queue (ahead of the 16KB HBM Hout/Dh
  snapshot write nobody waits on); ws3/ws4 relays ahead of the Vnew/Dv2 writes; fwd c-phase
  gate-first with a separate ungated staging buffer (Vnew stores ungated v_new)
- baseline_time: fwd 2971.1 / bwd 4554.3 / total 7525.4 us
- candidate_time: fwd 2640.8 / bwd 4565.1 / total 7205.8 us
- improvement: fwd -11.1% / bwd flat (its waits sit elsewhere) / total -4.2%
- precision: pass (L0 6/6)
- adopted: yes
- rollback_reason: n/a
- next_hint: statement-issue + auto-sync barriers dominate both sides (generated source:
  a PipeBarrier<PIPE_V> between every V-op chain node; ~66 intra-core Set/Wait pairs per
  fwd batch iteration). chunk_o achieves ~0.04us/stmt via a never-blocking V stream.

## Iteration 6 — 2026-11-11T17:20 (fwd 3-chain batch; REJECTED, rolled back)
- optimization: fwd 2->3 chain batch (V-streaming hypothesis test)
- candidate_time: fwd 2735.2 us (vs 2640.8)
- improvement: +3.6% (WORSE -- the NTASK=7 tail batch waste (2 of 9 chain-slots duplicated)
  cancels the deeper pipeline)
- precision: pass (L0 6/6)
- adopted: no; rolled back to the 2-chain form
- rollback_reason: performance regression beyond noise
- next_hint: tail-batch waste is structural for NCHAIN % (chains*cores) != 0

## Iteration 7 — 2026-11-11T17:50 (single-chain + BV=128 fwd + compactions; PARTIALLY ADOPTED)
- optimization: restructure to single-chain task loop (r3 form; simplest issue stream, no
  redirect waste) + fwd BV=128 when DV%128==0 (per-core chunk count halves; row_expand
  splits into 64-column halves) + bwd qdo/wdv2 shared readback buffer + time-shared L0C
  accumulator + MTE3 ordering retained
- detours (recorded for the journal): (a) q~ full-width staging via column-region UB
  writes -> max_abs=nan on every DK=128 case (column-sliced UB region writes corrupt under
  vid splitting; reverted to per-half buffers); (b) bwd BV=128 -> L0C overflow 160KB > 128KB
  (dvp 32 + qdo 64 + wdv2 64; fixed by the time-shared accumulator, then pinned to BV=64
  after the region-class bugs); (c) scale folded into the prep GE tile -> reverted with
  the region cleanup (not re-isolated; the f-phase mul stays)
- candidate_time: fwd 2520.6 / bwd 4507.8 / total 7028.3 us (single-chain bwd);
  fwd 2545.3 / bwd 4552.7 / total 7098.1 us (2-chain-batch bwd + shared readback/accumulator)
- improvement: fwd -4.5% / bwd -0.3..-1.3% (both bwd forms within 1% noise of each other;
  the 2-chain-batch form kept -- equivalent, already in place)
- precision: pass (L0 6/6, max_abs identical to the Stage-2 record)
- adopted: yes
- rollback_reason: n/a
- next_hint: structural means exhausted (see the final floor accounting in
  fla_chunk_delta_final_summary.md); remaining gap is the auto-sync per-statement cost
  (~0.1us/stmt incl. PIPE_V barriers between every V-op chain node) x the ~16/27 mandatory
  statements per chain-chunk x the serial-recurrence round-trip floor

## Iteration 8 — 2026-11-11 (round-2 R8: measurement hygiene + evidence refresh; no kernel change)
- bottleneck_type: n/a (measurement)
- optimization: fixed perf_tuning/fla_chunk_delta_bench_entry.py fwd-main/dump-src BV=64
  hardcode (old-signature variant, 3318.8us, NOT comparable to the BV=128 bench sentinel);
  regenerated the fwd source dump at the true final config
  (fla_chunk_delta_fwd_kernel_source_final_bv128.c)
- baseline_time: fwd 2538.3 / bwd 4559.8 / total 7098.1 us (R8 same-session re-measure;
  matches the round-1 final records within noise)
- candidate_time: n/a
- improvement: 0
- precision: pass (L0 6/6 in the same job)
- adopted: n/a (bench_entry only; kernel untouched, sha 1bfd3d80...)
- rollback_reason: n/a
- next_hint: TRUE-final fwd(BV=128) V-side census per chunk: 13 V-ops + 11 PIPE_V barriers
  + 9 MTE stmts (~22 total); bwd per iteration: AIV 66 stmts + 68 condval if/else blocks
  (Select-redirect inlined into EVERY buffer index of EVERY statement!) + 25 PIPE_V;
  AIC 26 tl-ops + 24 condvals. NEW finding: the bwd Select-redirect's condval storm is
  ~90 branch issues/iteration on both sides, plus the tail batch makes 12 cores run
  duplicated chains (redirect to cid). Both addressable by an in-bounds-provable batch
  decomposition -> Iteration 9.

## Iteration 9 — 2026-11-11 (round-2 R9: bwd select-storm removal + guarded single-chain tail; ADOPTED)
- bottleneck_type: compute (scalar issue: Select-redirect condval storm) + other (tail-batch duplicate work)
- optimization: universal batch decomposition. NFULL = NCHAIN // (2*CORE_EFF) full
  2-chain batches are provably fully in-bounds (max tid 2*NFULL*CORE_EFF-1 <= NCHAIN-1),
  so their chain indices are PLAIN arithmetic -- zero T.if_then_else Select: the select's
  if/else chains were inlined into every buffer index of every statement (68 AIV + 24 AIC
  condval blocks per iteration in the round-1 final source). Remainder R = NCHAIN mod
  2*CORE_EFF chains run as guarded single-chain tail batches (T.serial(NTAIL) over tasks
  2*NFULL..; `if tidt < NCHAIN` folds to `if (cid < 8)` at the bench shape): cores with a
  valid 7th chain process it at half statement cost, the other 12 cores skip the whole
  block (both core sides skip together -- relay flags stay balanced) instead of the old
  Select-redirect duplicate chains. `if NFULL > 0:` Python-const guard around the full
  loop keeps the dead-loop body out of the IR for small shapes (zero-extent serial loops
  break the memory planner's address_map contract: "Cannot find pre-allocated address
  for buffer k_l1_b").
- baseline_time: fwd 2538.3 / bwd 4559.8 / total 7098.1 us
- candidate_time: fwd 2538.7 / bwd 3929.4 / total 6468.1 us
- improvement: fwd flat (untouched) / bwd -13.8% / total -8.9% (cumulative from Stage-2: -60.0%)
- precision: pass (L0 6/6, max_abs identical to the Stage-2 records: l0_bwd_repr dh 3.906e-03,
  l0_bwd_opts dh 6.104e-05 -- the restructure is numerically inert)
- adopted: yes
- rollback_reason: n/a
- next_hint: generated-source census (fla_chunk_delta_bwd_kernel_source_r9.c): condval=0,
  AIV 66 stmts + 25 PIPE_V per full-batch iteration (unchanged compute statements),
  AIC 26 tl-ops; wall per full-batch iter now ~8.5us with ~5.5 AIV + ~5.8 AIC busy
  (pre-R9) -> re-profile to attribute the -630us between condval issue and tail skip.
  Remaining bwd V-side bulk: the q~ path (13 stmts + 2 relay handshakes per chain-chunk:
  q_half reads x2, casts x2, row_expand x2, qt16 casts x2, ws3/ws3b writes, GE read)
  -> Iteration 10: externalize q~ = q*exp(s*g) (bit-identical arithmetic) into the
  threads=1 prep kernel; main reads QT [BS,DK] directly on the C side (one gm_to_l1
  replaces two ws3 readbacks; ws3/ws3b relays disappear). Prep cost: +268MB HBM + ~2.6K
  stmts/core (~0.4-0.5ms) vs main V-side cut ~26 stmts/iter.

## Iteration 10 — 2026-11-11 (round-2 R10: bwd q~ externalization to prep; ADOPTED after a precision detour)
- bottleneck_type: compute (V-side statement issue: the 13-stmt q~ path) + sync (2 relay handshakes)
- optimization: prep (threads=1) materializes QT = bf16(fp32(q) * exp(s*g)) per (b, chunk, head)
  -- the EXACT arithmetic chain the main V prologue used (same cast/gate points, gate
  values = the gt_t rows the GE form staged). Main bwd: Q3->QT input (pre-gated q~),
  GE slot + workspace_3/3b dropped, the whole q~ V path removed (-13 stmts/chain-chunk),
  C-side reads QT as ONE [BS,DK] gm_to_l1 (replacing two ws3 readbacks), 2 relay
  handshakes per chain-chunk eliminated. bench_entry bwd-main synthetic args updated.
- detour (recorded): folding qdo_scale into the QT gate measured bwd 3538.7us (-9.9%!)
  BUT l0_bwd_opts dh0 ratio=0.9601 FAIL -- the test golden quantizes
  qt = bf16(q*e^g) at the UNSCALED point (test_fla_chunk_delta.py bwd golden line
  `qt = (qd * _exp(gf)).to(q.dtype)`); folding scale moves the bf16 rounding to the
  scaled operand and the bwd recurrence feedback (dH -> dv2 -> wdv2 -> dH) amplifies
  the divergence: dh0 max_abs 2.384e-07 -> 2.284e-03, dh 3.906e-03 -> 1.562e-02.
  Reverted the fold; the scale stays in the main f-phase fp32 mul.
- baseline_time: fwd 2538.7 / bwd 3929.4 / total 6468.1 us (iter9 adopted)
- candidate_time: fwd 2649.1 / bwd 3635.1 / total 6284.2 us (same-session; fwd code
  untouched -- the fwd 2528-2649 spread across sessions is ~2.4% cross-session noise)
- improvement: bwd -7.5% / total -2.8% on this session pair; bwd alone well beyond
  the 3% noise threshold (also 3538.7 measured in the scale-fold detour session)
- precision: pass (L0 6/6, max_abs bit-identical to the Stage-2 records after the
  fold revert: l0_bwd_repr dh 3.906e-03, l0_bwd_opts dh0 2.384e-07)
- adopted: yes
- rollback_reason: n/a (the scale fold alone was rolled back)
- next_hint: L0C/ws5-ws6 merge analysis (blocked -- see Iteration 11), fwd 2-chain
  BV=128 (blocked -- see Iteration 11), issue-structure micro-opts (Iteration 12).

## Iteration 11 — 2026-11-11 (round-2 R11: ws5/ws6 L0C-accumulate merge + fwd 2-chain BV=128; BOTH REJECTED with evidence)
- bottleneck_type: sync (relay handshakes) / structure (fwd round trips + tail imbalance)
- optimization attempted (a) bwd ws5/ws6 merge: qdo gemm (init=True) then wdv2 gemm
  accumulating in the same L0C (init=False) -> ONE fixpipe + one readback; requires
  negating q~ in prep (exact) so the accumulate yields (wdv2 - qdo) and the f-phase
  subtracts. BLOCKED BY ALGEBRA + GOLDEN: dH = D*dH + scale*qdo - wdv2 -- the merge
  needs scale*qdo inside the L0C, i.e. the scale folded into the bf16 q~ operand --
  exactly the Iteration-10 fold that FAILED the golden gate (dh0 ratio 0.9601). No
  scale-free formulation exists (scale*(qdo+wdv2) needs wdv2 separately; folding
  1/scale into dv2 adds an inexact fp32 mul (scale=DK**-0.5, not a power of 2) on the
  dv2 path -- worse). REJECTED at analysis level.
- optimization attempted (b) fwd 2-chain BV=128 batch on CE=16 (NCHAIN=64: 2 full
  batches, ZERO tail; halves the per-chain-chunk round-trip count and removes the
  4-vs-3 chain imbalance). CRASHED: aicore exception 507015 on every 2-chain case
  (l0_fwd_repr onward). Root cause from the generated C (fla_chunk_delta_fwd_kernel_source_r11.c):
  the 2-chain fwd UB demand is ~284KB/AIV > the 196KB budget -> AscendMemoryPlanning
  falls back to lifetime-overlapping allocations (h16_ub_b and vn_ub_b BOTH at UB
  offset 90496; kh_ub/vn_ub_b/h16_ub_b mutually overlapping) -> AscendSyncInsert's
  dependency analysis MISSES the cross-pipe WAR on physically-overlapped DIFFERENT-name
  buffers (the Sub writing vn_ub_b runs while the prologue's MTE3 relay is still
  reading the same bytes as h16_ub_b) -> data race -> aicore exception. Sharing the
  L0C frags (96KB, both-shared) did NOT fix it (same crash) -- the fault is the UB
  overlap, not L0C. Same-name-only sharing of the c-phase compute buffers still needs
  ~203KB > 196KB: fwd 2-chain BV=128 is UB-INFEASIBLE without dropping the gate path
  (k_g externalization), and k_g externalization is golden-blocked (fwd golden
  quantizes vn16 = (vn*gate).to(fp16) at cast point 3 -- the same coupling class
  that failed Iteration 10). REJECTED.
- baseline_time: fwd 2649.1 / bwd 3635.1 / total 6284.2 us
- candidate_time: n/a (crash / not implemented)
- improvement: n/a
- precision: n/a
- adopted: no
- rollback_reason: (a) golden-quantization coupling (measured FAIL class from iter10);
  (b) aicore exception 507015 from the memory planner's overlap fallback +
  AscendSyncInsert's blind spot for cross-pipe WAR on overlapped different-name buffers
  (toolchain finding: skill-journal candidate -- see the journal section below)
- next_hint: the two big structural levers are both toolchain/golden-blocked; remaining:
  issue-structure micro-opts (iter12) and the manual AUTO_SYNC experiment (now with
  weakened expected value -- iter12 measures the barrier cost directly).

## Iteration 12 — 2026-11-11 (round-2 R12: issue-structure micro-opts; NEUTRAL, rolled back)
- bottleneck_type: compute (statement issue) -- probe round
- optimization: (i) prologue MTE2 input loads issued FIRST (fwd U3/GT/GD, bwd
  Dv3/GT/GD for both chains -- GM latency hides under the prologue instead of the
  c-phase MTE2_V wait); (ii) per-half/per-chain row_expand scratch splits (fwd
  tmp_gate_a/b + tmp_decay_a/b; bwd tmp_gate_a/b + tmp_decay_a/b + qdw_ub_a/b) --
  same-pipe deps are NAME-based in AscendSyncInsert, so separate names drop the
  WAW/WAR-forced PIPE_V barriers between the column halves / chains.
- baseline_time: fwd 2649.1 / bwd 3635.1 / total 6284.2 us (iter10 state)
- candidate_time: fwd 2627.2 / bwd 3593.3 / total 6220.5 us
- improvement: -1.0% total (within the 3% noise threshold) -> NOT ADOPTED
- precision: pass (L0 6/6, max_abs identical -- the changes are numerically inert)
- adopted: no; rolled back to the iter10 state (sha e5c6f154...)
- rollback_reason: no measurable win beyond noise
- next_hint: CRITICAL DIAGNOSTIC from the generated-source census
  (fla_chunk_delta_{fwd,bwd}_kernel_source_r12.c): the bwd PIPE_V barrier count DID
  drop 25 -> 14 per full-batch iteration (the name-split mechanism works as derived),
  yet the wall was NEUTRAL -> the PIPE_V barrier drain cost on the critical path is
  FAR BELOW the round-1 ~0.1us/stmt estimate (barriers largely overlap other stalls,
  or the drain of already-completed tiny V-ops is ~free). This measured-neutral
  barrier removal KILLS the expected value of the manual AUTO_SYNC experiment
  (transcribing the auto event pairs and dropping PIPE_V drains): its main payoff
  was precisely these barriers. fwd census: 11 PIPE_V/iter unchanged (the fwd split
  target's loop-carried WARs still force barriers). Remaining wall = statement issue
  + serial-recurrence round trips, both at their structural floors -> iter13 is the
  terminal verification round (full gate + dual msprof) for the exhaustion verdict.

## Iteration 13 — 2026-11-11 (round-2 R13: TERMINAL verification; full gate + dual msprof)
- bottleneck_type: sync (serial-recurrence cross-core round trips; both sides 41-51% mutual wait)
- optimization: none (verification of the iter10-adopted state after the iter12 rollback)
- baseline_time: fwd 2649.1 / bwd 3635.1 / total 6284.2 us (the iter10-session numbers;
  that session ran ~3% slow across the board)
- candidate_time: fwd 2551.3 / bwd 3518.6 / total 6069.9 us (repeat run: 2549.1 / 3517.6 /
  6066.7) -- round-2 entry was fwd 2538.3 / bwd 4559.8 / total 7098.1
- improvement: round-2 cumulative -14.5% (bwd -22.8%, fwd ~flat); Stage-2 cumulative -62.5%
- precision: pass (--level all --bench exit 0: L0 6/6 + L1 10/10 + L2 10/10 + Boundary 5/5;
  max_abs identical to the Stage-2 records; the L1 tasktail cases cover NTAIL=1 and NTAIL=2)
- adopted: n/a (verification)
- rollback_reason: n/a
- next_hint: TERMINAL EVIDENCE (perf_tuning/fla_chunk_delta_prof_r13/): fwd main-only
  2573.2us (true BV=128 caliber, bench_entry fixed), bwd main-only 3044.7us, bwd prep
  382.6us. PipeUtilization: fwd AIC wait_id0(ws1) 51% + wait_id2(ws3) 45%, AIV
  wait_id1(ws2) 49% + wait_id3(ws4) 47% -- BOTH sides spend half the wall waiting on
  each other (the serial H-recurrence ping-pong); fixpipe 16-19%, cube 6.6%, GM bw
  5.8-8.7%. bwd: AIC wait_id0 41% + fixpipe 30%, cube 8.4%, GM 8.5-10.7%. The walls
  are trip-latency bound, not issue bound (iter12 proved barrier removal is neutral).
  Floor accounting: fwd 512 chunks x 2 trips x ~2.4us ~= 2.5ms (measured 2.57 -- AT
  the floor); bwd 512 iters x ~2.5 trips x ~2.2us ~= 2.9ms (measured 3.04); + prep
  0.38 + wrapper ~0.15 -> ~6.0ms ~= the measured 6068. The 3400us target requires
  halving BOTH the trip count (2-chain fwd: UB-infeasible, iter11 crash evidence) AND
  the trip latency (operand folds: golden-quantization-blocked, iter10 FAIL evidence)
  -- kernel-level means exhausted; verdict and full chain in
  fla_chunk_delta_final_summary.md (round-2 section).
