// Offline dump of the reduce2d_v2 constexpr planner for the TWOPHASE candidate
// of fla_chunk_intra_token_parallel iter4 (fp32 kSum, Clear=true -> merge=false,
// AllowRepeatZero=false [non-A3 platform]).
//
// TWOPHASE per window (M = i+1 or i):
//   P1: flat view of pq as [M*16, 8]  -> partial sums [M*16] (packed [M, 16])
//   P2: [M, 16] packed (PR=16)        -> [M] outputs (contiguous staging run)
// versus the CURRENT single call [M, 128] (PR=128).
#include <cstdio>
#include <string>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#define static_assert_constexpr static_assert
#include "reduce_2d_v2.h"
#pragma GCC diagnostic pop

using namespace reduce2d_v2;

static const char *TailName(Tail t) {
  switch (t) {
  case Tail::None: return "None";
  case Tail::Full: return "Full";
  case Tail::Direct: return "Direct";
  case Tail::Compact: return "Compact";
  }
  return "?";
}
static const char *BodyName(Body b) {
  switch (b) {
  case Body::EvenEven: return "EvenEven";
  case Body::OddOdd: return "OddOdd";
  case Body::Mixed: return "Mixed";
  }
  return "?";
}
static const char *LeafName(Leaf l) {
  switch (l) {
  case Leaf::None: return "None";
  case Leaf::M1: return "M1";
  case Leaf::Vcmax: return "Vcmax";
  case Leaf::M1TwoBinaryReductions: return "M1TwoBinary";
  case Leaf::M1OneBinaryReduction: return "M1OneBinary";
  case Leaf::Columns: return "Columns";
  }
  return "?";
}

static void DumpPlan(const char *label, uint32_t m, uint32_t n, uint32_t pr) {
  uint32_t pitch = DivUp(pr, 8);
  Plan p = (m == 1) ? SolveM1(n, pitch, false) : Select(m, pr, n, pitch, false);
  Layout l = Place(m, p.legal ? p : Plan{});
  if (m == 1) l = Place(m, p);
  printf("== %s M=%u N=%u PR=%u : ", label, m, n, pr);
  if (!p.legal) {
    printf("ILLEGAL\n");
    return;
  }
  printf("count=%u cost4=%llu scratch=%u\n", p.count, (unsigned long long)p.cost4,
         l.elements);
  for (uint32_t i = 0; i < p.count; ++i) {
    const Step &s = p.step[i];
    printf("   step%u: logical=%u pitch=%u -> next=%u npitch=%u tail=%s body=%s leaf=%s\n",
           i, s.logical, s.pitch, s.nextLogical, s.nextPitch, TailName(s.tail),
           BodyName(s.body), LeafName(s.leaf));
  }
}

int main() {
  printf("### CURRENT single-call [M, 128] PR=128 (reference)\n");
  unsigned long long cur_total = 0;
  for (uint32_t m = 2; m <= 16; ++m) {
    Plan p = Select(m, 128, 128, 16, false);
    cur_total += p.cost4;
  }
  cur_total += 2 * SolveM1(128, 16, false).cost4; // two M=1 calls (aqk diag + akk i=1)
  printf("   per-head total (31 calls, M=1..16 x2 sides): %llu + M1 share\n",
         (unsigned long long)cur_total);

  printf("\n### TWOPHASE P1: [M*16, 8] PR=8 (flat group view)\n");
  for (uint32_t m : {2u, 4u, 8u, 12u, 15u, 16u}) {
    DumpPlan("P1", m * 16, 8, 8);
  }
  printf("\n### TWOPHASE P2: [M, 16] PR=16 (packed partials)\n");
  for (uint32_t m : {2u, 4u, 8u, 12u, 15u, 16u}) {
    DumpPlan("P2", m, 16, 16);
  }
  printf("\n### ALT P2-narrow: [M, 16] PR=128 (partials left in pq rows)\n");
  for (uint32_t m : {2u, 4u, 8u, 16u}) {
    DumpPlan("P2n", m, 16, 128);
  }

  printf("\n### SPLIT-MERGE: [M, n]@PR=128 column slices, clear=True + merge=True\n");
  for (uint32_t n : {32u, 48u, 64u, 96u}) {
    printf("-- n=%u\n", n);
    for (uint32_t m : {2u, 4u, 8u, 9u, 12u, 15u, 16u}) {
      Plan a = Select(m, 128, n, 16, false);
      Plan b = Select(m, 128, n, 16, true);
      printf("   M=%-3u clear=%-5llu merge=%-5llu sum=%llu\n", m,
             (unsigned long long)(a.legal ? a.cost4 : 999999),
             (unsigned long long)(b.legal ? b.cost4 : 999999),
             (unsigned long long)((a.legal && b.legal ? a.cost4 + b.cost4 : 999999)));
    }
  }

  printf("\n### per-head cost4 totals (Aqk M=1..16 + Akk M=1..15)\n");
  unsigned long long cur = 0, tp = 0;
  for (uint32_t m = 1; m <= 16; ++m) {
    Plan c = (m == 1) ? SolveM1(128, 16, false) : Select(m, 128, 128, 16, false);
    cur += c.cost4;
    if (m > 1) {
      Plan p1 = Select(m * 16, 8, 8, 1, false);
      Plan p2 = Select(m, 16, 16, 2, false);
      tp += p1.legal + p2.legal ? p1.cost4 + p2.cost4 : 999999;
    } else {
      tp += SolveM1(128, 16, false).cost4; // keep M=1 as-is
    }
    if (m == 15) { /* akk stops at 15 */ }
  }
  // akk side: M=1..15 (reuse loop above approximated); compute precisely:
  unsigned long long cur_a = 0, tp_a = 0;
  for (uint32_t m = 1; m <= 15; ++m) {
    Plan c = (m == 1) ? SolveM1(128, 16, false) : Select(m, 128, 128, 16, false);
    cur_a += c.cost4;
    if (m > 1) {
      Plan p1 = Select(m * 16, 8, 8, 1, false);
      Plan p2 = Select(m, 16, 16, 2, false);
      tp_a += p1.legal && p2.legal ? p1.cost4 + p2.cost4 : 999999;
    } else {
      tp_a += SolveM1(128, 16, false).cost4;
    }
  }
  printf("   CURRENT : aqk(M=1..16)=%llu (w/ M1) akk(M=1..15)=%llu total=%llu\n",
         (unsigned long long)(cur - SolveM1(128, 16, false).cost4),
         (unsigned long long)cur_a,
         (unsigned long long)(cur + cur_a - SolveM1(128, 16, false).cost4));
  printf("   TWOPHASE: aqk=%llu akk=%llu total=%llu (M=1 kept on M1 path)\n",
         (unsigned long long)(tp - Select(16 * 16, 8, 8, 1, false).cost4 -
                              Select(16, 16, 16, 2, false).cost4),
         (unsigned long long)tp_a, (unsigned long long)(tp + tp_a));
  return 0;
}
