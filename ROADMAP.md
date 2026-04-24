# ROADMAP — bit-accurate MMA-Sim, fast reimplementation

Living doc. Revised 2026-04-23 after Phases 0–4 landed.
Full retrospective in [logs/2026-04-23-full-retrospective.md](logs/2026-04-23-full-retrospective.md).

## Goal

A drop-in replacement for `mmasim` that produces **bit-identical** outputs
across every supported (arch, instruction, dtype) combination, fast
enough for realistic-scale simulations (millions of tiles).

## Status (2026-04-23)

**Perf:** 5 of 10 fixtures specialized on CPU at **~10⁴× over the
Python oracle** (ampere-f16 workhorse: 0.581 μs/tile = 16 180×).
Metal GPU port works bit-exact but doesn't beat CPU at tested sizes.

**Coverage:** 5/10 fixtures (f16/bf16/tf32 inputs, f32 output, Volta
through Ampere). Missing: f64, fp8, fp4, block-scaled, AMD, wgmma,
tcgen05mma.

**Validation:** corpus-based, seeded RNG samples vs Python oracle.
Never validated against real GPU silicon.

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

### Branch A — Coverage (priority: high)

Extend the specialized kernel set from 5/10 to 10/10 supported fixtures
and beyond. Each is plumbing, not invention, but unglamorous:

1. **f64 inputs (`ampere-f64`, `hopper-f64`)**. Oracle uses serial
   `libm::fma` with single rounding per step. Needs a Rust vectorized
   FMA intrinsic (`std::arch::aarch64::vfmaq_f64` or similar) and a
   different kernel shape (pairwise reduction, not integer alignment).
2. **fp8 inputs with Ada Lovelace's `f32_e8m13` output** (`ada-fp8-e4m3`).
   Already infrastructure-supported; just needs wiring.
3. **fp4 + block-scaled** (`blackwell-mxfp8`, `blackwell-mxfp4`).
   New `mma_block_scale` entry class, `unpack_fp4_tensor` equivalent,
   scale handling in fused_sum.
4. **AMD MFMA.** New ISA class (`mmasim.isa.amd`), RD rounding mode,
   different accumulator width (35 fractional bits for mxfp4).
5. **wgmma / tcgen05mma.** Collective matmuls for Hopper / Blackwell.
   Distinct ISA surface.

**Estimated effort:** 2–3 focused days total. None require
algorithmic invention.

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

## My recommendation (in the spirit of your feedback)

**Coverage (A) → Hardware validation (B)**, in that order, before any
more perf work:

- A fills in the drop-in-library goal you stated. Every new fixture
  that passes is more of the actual ISA covered.
- B answers the load-bearing unknown: is the oracle right? If the
  oracle has bugs, our impl inherits them silently. Checking before
  we publish or depend on this matters.
- If after A+B the perf ceiling still feels insufficient for the target
  workload, return to C.

The Phase 4 (Metal) result suggests diminishing returns on pure
perf work on this hardware. Time better spent on the correctness
foundation.

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
