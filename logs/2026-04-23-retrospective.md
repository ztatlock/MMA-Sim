# 2026-04-23 — Retrospective on phases 0 → 2.5

**Status:** written at the end of a single-day push that got us from a
Python-only oracle to a NEON+rayon Rust kernel running at ~10⁴× per
tile. Writing this before the next strategic decision.

## Scoreboard

| phase | what it added                              | ampere-f16 μs | speedup   |
|-------|--------------------------------------------|---------------|-----------|
| 0     | oracle baseline                            | 9400          | 1×        |
| 1     | vectorized NumPy reference                 | 110           | 85×       |
| 2     | Rust f64 kernel + PyO3                     | 36            | 261×      |
| 2.1   | integer fixed-point inner loop             | 16            | 588×      |
| 2.2   | batched entry, amortize PyO3               | 10.50         | 895×      |
| 2.3   | aarch64 NEON SIMD                          | 8.90          | 1056×     |
| 2.4   | rayon over batch (scalar kernel)           | 1.28          | 7344×     |
| 2.5   | rayon + SIMD                               | 1.08          | 8704×     |
| —     | 10⁶× target                                | 0.0094        | 10⁶×      |

Every step bit-exact against the Phase 0 corpus.

## What worked well

**Oracle-first / corpus-as-gate.** Phase 0's corpus (16 samples × 10
instruction types, ~500 KB) caught three real bugs on first run:
hardcoded `nfb=24` that broke Volta (nfb=23), hardcoded `min_exp=-14`
that broke tf32, and the signed-shift-vs-trunc-toward-zero distinction
in the integer kernel. Each would have been a head-scratching debugging
session without the corpus. The cost of building the corpus (~30 minutes)
paid back by the first bug caught.

**Keeping the oracle alive.** We never modified `mmasim/`. Every
rewrite landed alongside, not in place of. This meant the validator
could always cross-check. The macOS `libm.so.6` portability was
handled with a ctypes shim in `bench/_compat.py` rather than an
upstream patch — small friction, zero semantic risk.

**Small, reviewed commits.** The "no commit without approval" rule
forced me to articulate what each phase was trying to do, what the
honest numbers were, and what I was skipping. It also surfaced
strategic forks (SIMD vs rayon, for example) to the user rather than
burying them in a code push.

**Rayon was the biggest single lever.** 8–10× from ~50 lines of code.
The batched kernel's structure (stack-allocated buffers, no heap
inside the tile, independent outputs) was already ideal for
parallelism. The lesson: **set up the data-parallel shape first; the
threading library then barely has to ask for anything.**

## What didn't match expectations

**Single-core CPU projections were reliably optimistic.** The roadmap
(Phase 0 strategy section) projected:
- NumPy vectorized: 10²–10³×. Got 85×. (short by 1.5–12×)
- Rust integer single-core: 10⁴–10⁵×. Got 588×. (short by 20–200×)

The single-core Rust projection was the most embarrassing miss.
**Per-call fixed overhead** (PyO3 boundary, numpy array unpack, output
Vec allocation, GIL acquire) was ~7 µs even before any math. At the
tile sizes we care about (2048 products), the math itself is ~5 µs.
Fixed overhead was swamping the very thing I was optimizing. Only
batching (Phase 2.2) made the projections-shape numbers appear.

**Takeaway:** when the work-per-call is small, projections based on
"ops per second" are misleading. Per-call boundary cost dominates, and
it's not visible in a CPU spec sheet.

**SIMD gave 1.2× instead of the theoretical 2×.** Two reasons we now
understand:
1. The SIMD-able fraction (align+sum, k+1 ops per output) is ~30% of
   per-output work. Amdahl caps the win regardless of SIMD width.
2. Apple M-series has very wide scalar integer dispatch — 6+ ops/cycle
   with out-of-order speculation. 2-wide NEON competes against a
   scalar core that was already close to saturated. AVX-512 (8-wide)
   on a different machine would likely give 2–3×, still below naive.

## What the 10⁶× number meant

In retrospect, **"1M× faster" was a framing, not a tractable number.**
On a CPU we are going to plateau around 2–5 × 10⁴× per-tile. Further
CPU work (specialization, wider SIMD, maybe lock-free output) might
double that but won't cross 10⁵× without a substrate change.

But zoom out to **full-workload** throughput — simulating a typical
GEMM as ~10⁶ tiles — and the picture changes. The Python oracle on
that workload would take ~3 hours; our rayon+SIMD does it in ~1
second. **That is a 10⁴× real-world speedup on the workload people
actually want to run.** If the original goal was "make the simulator
usable for realistic research workloads," we're already there. If the
goal is literally 10⁶× per tile, only GPU gets there.

Worth calibrating the framing before deciding next steps.

## Honest footnote on "bit-exact"

We verified bit-exactness against the Python oracle. The oracle itself
uses `math.frexp` and libc `fma`, calls into Python's rounding, and
has at least a couple of artifacts we inherited (the `exp = -126`
zero-quirk that survives independent of dtype; Volta's `nfb = 23`
differing from Turing's 24; Ada's `f32_e8m13` truncation of the f32
output's bottom 10 mantissa bits). **Whether the oracle itself
matches real hardware bit-for-bit is an open question** — we've never
run the corpus against an actual GPU.

If the goal is a reference model that predicts what NVIDIA silicon
does, corpus expansion should include **hardware-captured outputs**,
not just oracle-generated ones. That's a different project from "fast
reimplementation of the oracle."

## What I'd do differently

- **Micro-benchmark the PyO3 boundary early.** We flew past single-call
  overhead in Phase 2 without realizing how much of the budget it
  already consumed. A 20-minute detour at the start of Phase 2 would
  have caused us to jump to batching sooner.
- **Plan Phase 0's corpus format for portability.** We chose
  `torch.save` bundles, which means Rust consumers can't trivially
  read them. When we port to GPU (if we do), the corpus will need a
  language-agnostic format. Add now, or pay the conversion cost later.
- **Keep a smaller number of PyO3 entry points.** We have six
  (`mma_f32_out`, `_batched`, `_batched_simd`, `_batched_rayon`,
  `_batched_simd_rayon`, plus the single-call). Good for comparative
  benchmarking, but a real library would want a single dispatch.
- **Build a batch-size sweep benchmark.** All our numbers are at
  B=1024. At small batches (B=16), rayon loses to single-threaded;
  SIMD wins. We don't actually know the crossover points. One small
  script would show the whole map.

## Known gaps (coverage)

| feature            | status       | effort     |
|--------------------|--------------|------------|
| f16 / bf16 / tf32  | ✓            | done       |
| fp8 (e4m3, e5m2)   | scalar only  | small      |
| f64                | oracle only  | medium (need vectorized IEEE FMA) |
| fp4 (e2m1)         | oracle only  | small (unpack + plug in) |
| block-scaled       | oracle only  | medium     |
| Ampere f32 (tf32)  | ✓            | done       |
| AMD MFMA           | oracle only  | medium     |
| wgmma, tcgen05mma  | oracle only  | medium-large |

Most of these are plumbing on top of the existing kernel. None require
algorithmic invention. Total coverage work ≈ 2–3 days of focused effort.

## Strategic forks

The path forward has three distinct branches:

1. **Coverage.** Extend `rust_ref` to every corpus fixture. Slow,
   low-risk, very high practical value if we want a drop-in
   replacement for the oracle. Current state is a research prototype
   on 5/10 fixtures.
2. **Specialization / more CPU perf.** Per-`(arch, dtype)` codegen via
   const generics; probably 2–3× more. Hits a hard ceiling around
   2–5×10⁴× per tile.
3. **GPU (Metal or CUDA).** Ports the integer kernel to a GPU.
   Credible 10⁶× path for full workloads. Substantial effort — 1–2
   weeks — but the kernel's integer nature makes it a clean target.

And one non-branch: **call it.** 10⁴× on a real workload with bit-exact
semantics is already a publishable result. Write it up, submit,
revisit later.

None of these is wrong; they serve different goals. Worth a
conversation before picking.
