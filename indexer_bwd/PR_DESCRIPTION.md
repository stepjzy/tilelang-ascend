## Summary

Add the `indexer_bwd` operator for DSA sparse fine-tuning on Ascend NPU, supporting both AscendC and PTO backends.

The operator computes gradients (`dIndexQ`, `dWeights`, `dIndexK`) for the lightning-indexer KL fine-tuning loss with variable-length packed sequences.

**Implementation:**
- **Fast kernel:** Persistent GEMM-based implementation optimized for contiguous and shared-row index layouts, with pipelined computation and reduced memory transfers.
- **Generic kernel:** Supports arbitrary index layouts, including invalid (`-1`) indices, with gradient accumulation.
- **Cast kernel:** Converts FP32 accumulated gradients to BF16.
- **Backend compatibility:** Uses PTO-compatible predicates, memory transfers, and atomic operations while preserving AscendC support.
- **Numerical precision:** BF16 computations with FP32 accumulation and rounding behavior aligned with the PyTorch autograd reference.

The implementation is provided in `example_indexer_bwd.py`, with the default correctness test in `test_example_indexer_bwd.py`.

## Precision

**Environment:** Ascend 910B3 (A2), CANN 9.2.0  
**Default configuration:** S=2048, H=16, D=128, topk=64, BF16, varlen offsets `[0, 1023, 2048]`.

Both explicit AscendC and PTO backends compiled and executed successfully, producing identical numerical metrics across the tested cases.

| Test | Cases | Checks | Result |
|---|---:|---:|---|
| L0 default (fast) | 1 | 3 | 3/3 PASS |
| L1 functional (fast/shared/generic) | 6 | 18 | 18/18 PASS |
| L2 invalid inputs | 4 | 4 | 4/4 PASS |
| Boundary (non-blocking) | 3 | 9 | 7 PASS, 2 WARN |
| **Blocking total** | **11** | **25** | **25/25 PASS** |

**Precision criteria (BF16):** `atol=2^-10`, `rtol=2^-6`, `matched_ratio >= 0.99`, and `max_abs_error <= 1.0`. Observed matched ratios range from `0.995422` to `1.000000` in L0/L1 cases.

The two boundary warnings concern NaN propagation differences from the PyTorch reference; they were non-blocking and consistent across both backends.

The committed default test follows the TileLang main repository's error-reporting approach. The expanded L0/L1/L2/Boundary matrix was separately verified against the same kernel and reference.

**Reproduction:**

```bash
cd examples_experiment/dsa_sparse_finetune
python test_example_indexer_bwd.py
```

## Performance

**Environment:** Ascend 910B3, CANN 9.2.0  
**Configuration:** S=2048, H=16, D=128, topk=64, BF16, fast kernel.

| Backend | Kernel latency (μs) | Relative latency |
|---|---:|---:|
| AscendC | 131.8 | 1.000× |
| PTO | 139.0 | 1.054× |

Measured with `torch.npu.Event` (median of 10 runs after 3 warmups), not pure device Task Duration. No independent AscendC reference or measured GPU baseline is available for comparison.

PTO-compatible tensor-edge staging introduces extra host-side overhead for unaligned variable-length segments. On the measured configuration, end-to-end interface latency increased from 1372.8 μs to 1644.6 μs (approximately 19.8%) versus the pre-PTO implementation.

## Backend Support and Known Limitations

- **AscendC:** Explicit backend compilation, execution, and correctness checks passed.
- **PTO:** Explicit `target="pto"` compilation, execution, and correctness checks passed.
- **Default target:** Currently selects the AscendC pipeline; default PTO compilation has not been verified.
- Some hardware-specific memory handling and synchronization are required for PTO compatibility.

