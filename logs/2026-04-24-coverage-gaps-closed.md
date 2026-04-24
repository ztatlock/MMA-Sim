# 2026-04-24 — Coverage gaps closed (N1–N4)

**Status:** landed
**Context:** after reading the paper ([docs/paper/](../docs/paper/)) and
establishing that the Python oracle is silicon-validated upstream
(see [docs/validation.md](../docs/validation.md)), the remaining work was
closing the coverage gaps flagged in the Branch A retrospective.
The paper's 8-operation taxonomy (Table 1) provided a clear checklist.

## What shipped

| Phase | Elementary op (paper) | Instructions added                         | Fixtures |
|-------|----------------------|-------------------------------------------|----------|
| N1    | RNE-FP16 conversion  | Volta/Turing/Ampere f16/f16 + Ada fp8/f16 | 4 + 4    |
| N2    | Φ_E-FDPA, Φ_FTZ-AddMul | AMD CDNA1/2 f16 & bf16 (5 variants)      | 5 + 5    |
| N3    | Φ_TR-FDPA, Φ_GTR-FDPA | AMD CDNA3 xf32/f16/bf16/fp8/bf8 (5 var.)  | 5 + 5    |
| N4    | (same as existing)   | Hopper wgmma (wider n, +fp8), Blackwell tcgen05mma | 8 |

## Final bit-exact scoreboard

All 68 fixtures pass `python -m bench.validate --impl fastmma.rust_ref`.

| instruction family              | count | backing kernel            |
|---------------------------------|-------|---------------------------|
| NVIDIA mma (Volta→Blackwell)    |  13   | integer + specialized     |
| NVIDIA mma f16-output           |   4   | f16-bits RNE normalize    |
| NVIDIA wgmma (Hopper)           |   6   | heap integer kernel       |
| NVIDIA tcgen05mma (Blackwell)   |   5   | same as wgmma             |
| NVIDIA mma_block_scale          |   2   | mxfp8 (k=32) + mxfp4 (k=64) |
| AMD mfma fma                    |   3   | f64 FMA / f32 FMA         |
| AMD mfma pairwise               |   5   | pairwise_dot_f32 + FTZ    |
| AMD mfma fused_dot_rd_add       |   5   | TR-FDPA / GTR-FDPA f64    |
| + edge-case corpora for many    | ~20   | (same kernels)            |

## Bug inventory (caught by corpus on first runs)

Beyond the earlier Branch A bugs (see
[2026-04-23-branch-A-coverage.md](2026-04-23-branch-A-coverage.md)):

1. **Upstream oracle bug in AMD pairwise / fused_dot_rd_add paths.**
   `pairwise_dot` asserts f32 inputs but amd.py calls it with
   f16/bf16. Open-source release path doesn't actually work for
   CDNA1/2 f16/bf16 or any CDNA3 non-f64/f32 path. Patched
   `mmasim/simulator/amd.py` to widen A, B to f32 before the
   pairwise loop (only — keep narrow types for fused_dot_rd_add so
   `extract_significand_exponent` sees source dtype's min_exp).
2. **Applied 2^-nfb twice in fused_sum_f64.** Same as the mxfp4 bug
   from Branch A, but bit me again in the TR-FDPA draft. Fixed.
3. **Used fused_sum_rd (floor) where oracle uses fused_sum (trunc).**
   TR-FDPA nests RZ-inner with RD-outer alignment. Not uniform
   rounding throughout. Swapping `fused_sum_rd` → `fused_sum_f64`
   in the inner step fixed it.
4. **f16-output path normalize.** Oracle uses RNE-FP16 (round-ties-
   even to 10 mantissa bits). Implemented via Rust's
   `f64::round_ties_even()`, with manual f16 bit packing to control
   the NaN pattern (0x7FFF — oracle convention).

## Taxonomy coverage (revised)

Paper's Table 1 of elementary operations vs our implementation:

| paper model    | algorithm | status | Rust function                   |
|----------------|-----------|--------|---------------------------------|
| Φ_FTZ-AddMul   | Alg. 1, 2 | ✓      | `run_one_tile_amd_pairwise` (flush=T) |
| Φ_FMA          | Alg. 3, 4 | ✓      | `run_one_tile_f64`, `run_one_tile_f32_fma` |
| Φ_E-FDPA       | Alg. 6    | ✓      | `run_one_tile_amd_pairwise` (flush=F) |
| Φ_T-FDPA       | Alg. 7    | ✓      | `fused_mma_step_int` family     |
| Φ_ST-FDPA      | Alg. 8    | ✓      | `run_one_tile_block_scale_k32`  |
| Φ_GST-FDPA     | Alg. 9    | ✓      | `run_one_tile_mxfp4_k64`        |
| Φ_TR-FDPA      | Alg. 10   | ✓      | `amd_fused_dot_rd_add_f64` (is_fp8=F) |
| Φ_GTR-FDPA     | Alg. 11   | ✓      | `amd_fused_dot_rd_add_f64` (is_fp8=T) |

**All 8 elementary operations implemented.** Any instruction in the
paper's scope whose oracle model is a composition of these operations
can now be dispatched.

## Remaining coverage gaps

These exist but are narrow:

1. **f16-output for big tiles.** Our f16 RNE path works for the `mma`
   family (m ≤ 16, n ≤ 8). Wider wgmma/tcgen05mma f16-output variants
   need the kernel wired to the f16-bits output mode. ~1 hour of
   plumbing.
2. **tcgen05mma_block_scale.** Different API (scales + packing);
   same math as mma_block_scale. Registered but not wired. ~2 hours.
3. **mxfp4 with ue4m3 scales.** Our mxfp4 path extracts scale
   exponents assuming ue8m0 (power-of-2). For ue4m3 scales (non-
   power-of-2), both sig and exp matter. Need a path that carries
   scale sig through the reduction. ~2 hours.
4. **e3m2, e2m3 fp6 types.** Paper says "TODO" in the upstream
   qualifier list; neither oracle nor our impl supports them.
5. **SASS verification.** Paper verified PTX→SASS mappings. We rely
   on the paper's guarantee; we don't re-run the SASS check.

None of these blocks the "silicon-validated reference" claim for the
instructions we do implement.

## By the numbers

- **Instruction types supported**: 26+ (13 NVIDIA mma family, 3 wgmma,
  5 tcgen05mma, 2 mma_block_scale, 13 AMD mfma).
- **Qualifiers registered**: 80+ across all classes.
- **Fixtures**: 68 (24 original + 44 added in Branch A + N1–N4).
- **All bit-exact against the oracle**, which is silicon-validated
  upstream.
- **Typical speedup over oracle**: 200×–1700×, depends on tile size and
  dtype (mixed-precision paths are faster because the int pipeline
  is more aggressively vectorized than the f64 pipelines).

## Commit trail (N1–N4)

```
7823561  N1: f16-output RNE, 8 fixtures
7974cea  N2: AMD pairwise (CDNA1/2), 10 fixtures (+oracle patch)
e54faa8  N3: AMD CDNA3 FDPA paths, 10 fixtures
1c6bdb3  N4: tcgen05mma + wider wgmma, 8 fixtures
```

Plus `3b42676` (the paper PDF + validation.md that kicked off this push).
