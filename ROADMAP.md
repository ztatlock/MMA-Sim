# ROADMAP — bit-accurate MMA-Sim, fast reimplementation

Living doc. Revised 2026-04-23 after Phases 0–4 landed.
Full retrospective in [logs/2026-04-23-full-retrospective.md](logs/2026-04-23-full-retrospective.md).

## Goal

A drop-in replacement for `mmasim` that produces **bit-identical** outputs
across every supported (arch, instruction, dtype) combination, fast
enough for realistic-scale simulations (millions of tiles).

## Status (2026-04-23, after Branch A)

**Perf:** ampere-f16 workhorse specialized at 0.581 μs/tile (16 180×).
Metal GPU port works bit-exact but doesn't beat CPU at tested sizes.

**Coverage:** **13 instruction types, 32 bit-exact fixtures** (random +
edge corpora). Covers NVIDIA `mma` (Volta→Blackwell), `mma_block_scale`
(Blackwell mxfp8/mxfp4), `wgmma` (Hopper f32-output), AMD `mfma`
(fma operation_type). Still missing: f16-output (RNE), AMD pairwise /
fused_dot_rd_add paths, tcgen05mma, wider wgmma n (>64).

**Validation:** corpus-based, 5 RNG seeds × ~10 samples × 13
instructions = ~200 curated validation points. Not yet validated
against real GPU silicon.

## Guiding principles (unchanged)

- **Bit-exact or it doesn't count.** All perf gated on corpus match.
- **Measure before optimizing.** No intuition-driven rewrites.
- **Keep the reference alive.** The Python impl in `mmasim/` is the
  oracle. Never modify without a clear reason; rewrites land alongside.

## Completed phases

| phase  | what                                       | per-tile (μs) | speedup  |
|--------|--------------------------------------------|---------------|----------|
| 0      | baseline + corpus + bench harness          | 9 400         | 1×       |
| 1      | vectorized NumPy reference                 | 110           | 85×      |
| 2      | Rust/PyO3 kernel (f64 arithmetic)          | 36            | 261×     |
| 2.1    | integer fixed-point core                   | 16            | 588×     |
| 2.2    | batched entry point                        | 10.58         | 889×     |
| 2.3    | aarch64 NEON SIMD                          | 9.06          | 1 037×   |
| 2.4    | rayon over batch (scalar kernel)           | 1.28          | 7 344×   |
| 2.5    | SIMD + rayon stacked                       | 1.10          | 8 545×   |
| 3.0    | hand-specialized ampere-f16                | 0.603         | 15 588×  |
| 3.1    | const-generic across all 5 supported       | 0.581         | 16 179×  |
| 4      | Metal GPU (ampere-f16 only)                | 1.00 / 0.25 * | varies  |
| M1a    | Ada fp8 f32-output (8 qualifiers)          | 0.020         | 580×     |
| M1b    | F64 kernel (Ampere + Hopper)               | 0.004         | ≥ 270×   |
| M1c    | Block-scaled mxfp8 (Blackwell k=32)        | 0.028         | 355×     |
| M1d    | Block-scaled mxfp4 (Blackwell k=64)        | 0.020         | 1 593×   |
| M2     | Edge-case corpus (NaN/Inf/zeros/extremes)  | —             | —        |
| M3     | AMD MFMA fma-subset (f64/f32 non-xf32)     | 0.005–0.007   | 800–1 400× |
| M4     | Hopper wgmma f32-output (3 qualifiers)     | 0.105–0.154   | 1 300×   |

\* Metal: 1.00 μs/tile at B=1024, 0.25 μs/tile at B=65k. Loses to CPU
at practical batch sizes; matches at extreme batches.

## Calibrated ceiling estimates (revised)

- **Per-tile on M3 Max CPU:** ~250 ns/tile floor, where both CPU-specialized
  and Metal converge. Current best: 581 ns/tile at B=1024. Another
  ~1.5–2× from 4-wide NEON pairs + better scheduling would approach
  the floor.
- **Per-tile on M3 Max GPU (i64 pipeline):** ~250 ns/tile at very large
  batches; net loss at typical batches due to dispatch overhead and
  i64 mul emulation cost.
- **Full-workload (≥10⁶ tiles):** already at ~10⁴× real-world speedup
  (3-hour oracle → 1-second CPU batched). Phase 5.x i32 rewrite could
  plausibly push to 10⁵×.
- **10⁶× per-tile on this hardware is unlikely without relaxing bit-exactness.**

## Next steps — three branches

These aren't exclusive, but they serve different goals. Priorities
below reflect the user's stated goal: *trustworthy, reliable,
bit-accurate simulation at scale*.

### Branch A — Coverage (largely COMPLETE; remainders below)

Everything tractable without new algorithm work landed in milestones
M1a–M4 (commits d8eef34..38c6d0f). What remains:

1. **f16-output variants.** Oracle uses RNE for f16 output; our
   normalize uses RZ. Needed for: wgmma f16-output, Ada fp8 f16-output
   (16 qualifiers), Blackwell f8f6f4 f16-output. ~½ day to add an
   `normalize_f16_rne` path.
2. **AMD pairwise + fused_dot_rd_add paths.** Covers CDNA1/2 f16/bf16
   (pairwise) and CDNA3 xf32/f16/bf16/fp8 (fused_dot_rd_add with RD
   rounding). ~1 day. RD rounding is the non-trivial piece.
3. **tcgen05mma.** Collective Blackwell matmul; math identical to
   wgmma per-tile (same `nv_fused_dot_add`). ~30 min plumbing.
4. **Wider wgmma n.** Tested n=64, ISA supports n up to 256. Kernel
   should already handle; register qualifiers. ~30 min.

**Estimated remaining effort:** ~2 days, no algorithmic novelty.

### Branch B — Validation against real silicon (priority: high)

Our bit-exactness is *relative to the Python oracle*, which itself has
unvalidated quirks (the zero-exp=-126 rule, Volta's nfb=23, Ada's
f32_e8m13 truncation). For "trustworthy simulation" the corpus should
be expanded with hardware-captured golden outputs.

Steps:

1. Identify an NVIDIA GPU with access (CUDA install).
2. Write a CUDA test harness that runs each MMA instruction on real
   silicon with the exact same inputs as our corpus.
3. Dump outputs and commit as `fixtures/hw/<label>.pt` alongside the
   existing fixtures.
4. Extend `bench/validate.py` with a `--vs hardware` mode.
5. Any discrepancy: investigate, patch oracle or our impl.

**Estimated effort:** 1–2 days contingent on hardware access.

### Branch C — Phase 5: i32 / f32 kernel (priority: medium)

Pushes the per-tile performance ceiling another ~10× (plausible
target: 30–60 ns/tile, ~2 × 10⁵× over oracle).

1. **i32 on CPU:** rewrite the inner loop with operand scale 2^12
   (half of NFB=24) so products fit in i32. 2× SIMD throughput on NEON
   (4-wide i32 vs 2-wide i64); similar win on AVX-512. Bit-exact
   achievable.
2. **i32 on GPU:** port same strategy to MSL. Apple GPU native i32
   throughput is much higher than i64; plausible 3–5× over our
   current Metal kernel, and possibly beats CPU.
3. Ripple effect: probably retire the 2-wide NEON kernel; 4-wide i32
   is uniformly better.

**Estimated effort:** 1 focused day (Rust) + 1 day (Metal) + bit-exact
debugging.

### Branch D — Accept it, write it up (priority: medium-low)

10⁴× per-tile at bit-exactness, on 5 of 10 fixtures, is already a
publishable numerical-reproducibility result. The paper writes
itself: oracle-gated progressive specialization, clean phase structure,
honest negative result on GPU.

This isn't mutually exclusive with A/B. Coverage and hardware
validation would make the writeup stronger.

## Updated recommendation (after Branch A lands)

**Hardware validation (B) is now the single highest-value task.**
Coverage A is largely done (13 instruction types, 32 bit-exact
fixtures). Our "bit-exact" claim is relative to the Python oracle;
the oracle's own claim to be bit-accurate against real silicon has
never been tested. Getting access to an actual NVIDIA/AMD GPU and
running our corpus on it would either:

- Confirm the oracle (and our impl) match silicon → publishable
  reference implementation, enables trustworthy simulation at scale.
- Surface oracle bugs → we fix upstream, add regression tests, the
  field gains a more accurate tool.

Either outcome is valuable. Without this step, the library's
correctness is only as good as the oracle's correctness, which is an
unmeasured assumption.

**Secondary priorities:**
- Residual coverage (A continuation): f16-output, AMD non-fma paths,
  tcgen05mma, wider wgmma. ~2 days, no research content.
- Perf work (C / i32 rewrite): diminishing returns but plausible
  ~10× on CPU, ~5× on GPU. Defer until workload demands it.

## Non-goals (for now)

- Supporting instructions/dtypes beyond what upstream implements.
- A full Python API surface redesign. Compat wrapper only.
- Full-matmul tiling semantics — this remains per-instruction.

## Open questions worth reopening later

- **Is bit-exactness actually what users need?** For many research
  uses, a fast f32 "approximate" mode (uses hardware FMA, no fused
  integer accumulator) would be fine. Could ship as a separate
  `fastmma.approx` module alongside the bit-exact kernels.
- **Are there batch sizes < 128 that are the realistic use case?**
  Our benchmark defaulted to 1024. If the actual workload simulates
  MMA-by-MMA (unbatched), the single-call path is the one that
  matters and the scoreboard looks very different.
- **Does the Python API need to be the wrapper?** A Rust-native
  driver with no Python dependency would shed another ~100–500 ns
  per call. Worth considering if Python isn't a hard requirement.
