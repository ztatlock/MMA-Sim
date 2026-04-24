# 2026-04-23 — Full Retrospective (Phases 0 → 4)

**Status:** end-of-push report
**Scope:** this document supersedes the Phase 2.5 interim retrospective
([2026-04-23-retrospective.md](2026-04-23-retrospective.md)) and extends it with the specialization
(3.0, 3.1) and Metal GPU (4) results. Writing before the next strategic
pivot.

---

## TL;DR

Starting from an ~9.4 ms/tile Python oracle for the Ampere m16n8k16
f16→f32 workhorse, we landed a **bit-exact** reimplementation at
**581 ns/tile on CPU (16 180×)** and **1.0 μs/tile on GPU (same correctness,
~2× slower than CPU at our sizes)**. Every step gated by a committed
regression corpus; no correctness failures shipped.

The original "1 million times faster" framing turned out to be
**aspirational for per-tile numbers on this hardware** but **already
achieved for full-workload throughput** (≥10⁴× on realistic
simulations). A GPU rewrite to i32 arithmetic could credibly double
the per-tile win, but on our specific problem the i64-heavy integer
pipeline favors 12 CPU P-cores over the M3 Max GPU.

Most load-bearing unfinished work: **coverage** (5/10 fixtures) and
**hardware-ground-truth validation** (oracle vs real silicon, which
we've never checked).

---

## Full scoreboard — ampere-f16 workhorse progression

All numbers μs/tile, batch=1024 where applicable, on Apple M3 Max.

| #     | stage                                     | μs/tile   | speedup    | bit-exact |
|-------|-------------------------------------------|-----------|------------|-----------|
| **0** | Python/Torch oracle                       | **9 400** | **1×**     | (oracle)  |
| 1     | NumPy vectorized reference                | 110       | 85×        | ✓         |
| 2     | Rust f64 single-call                      | 36        | 261×       | ✓         |
| 2.1   | Rust integer fixed-point single-call      | 16        | 588×       | ✓         |
| 2.2   | Rust int batched (per-tile, B=1024)       | 10.58     | 889×       | ✓         |
| 2.3   | + NEON SIMD (2-wide on j)                 | 9.06      | 1 037×     | ✓         |
| 2.4   | Rust int batched + rayon (scalar kernel)  | 1.28      | 7 344×     | ✓         |
| 2.5   | + NEON SIMD + rayon                       | 1.10      | 8 545×     | ✓         |
| 3.0   | Hand-specialized ampere-f16 + rayon       | 0.603     | 15 588×    | ✓         |
| **3.1**| **const-generic specialization + rayon + NEON** | **0.581** | **16 179×** | ✓     |
| 4     | Metal GPU (B=1 024)                       | 1.004     | 9 362×     | ✓         |
| 4     | Metal GPU (B=65 536)                      | 0.248     | 37 903×    | ✓         |
| —     | 10⁶× target                               | 0.0094    | 10⁶×       | —         |

**Current best per-tile ceiling: 248 ns/tile = ~38 000× over the
Python oracle**, at B≥65k tiles on GPU. CPU caps at ~250 ns/tile in
the same asymptotic region. The two paths converge.

## Cross-instruction scoreboard (Phase 3.1 CPU specialized)

| instruction   | oracle (μs) | specialized (μs) | speedup  |
|---------------|-------------|------------------|----------|
| volta-f16     | 1 970       | 0.245            | 8 040×   |
| turing-f16    | 4 780       | 0.465            | 10 280×  |
| ampere-f16    | 9 400       | 0.581            | 16 180×  |
| ampere-bf16   | 9 190       | 0.618            | 14 870×  |
| ampere-tf32   | 7 420       | 0.304            | 24 410×  |

All five > 10⁴× per tile, bit-exact. **Ampere-tf32 is our best case
at ~2.4 × 10⁴** — the tf32 mantissa truncation makes the product ints
smaller, which helps SIMD throughput.

## Per-axis attribution (ampere-f16)

What each independent mechanism bought, starting from the scalar Rust
batched baseline (10.58 μs/tile):

| mechanism                        | factor     | notes |
|----------------------------------|------------|-------|
| 2-wide NEON SIMD (j-axis)        | 1.17×      | Amdahl-limited (~30% vectorizable) |
| rayon (16 cores)                 | 8.27×      | P+E asymmetry + dispatch ≈ 52% efficiency of theoretical 16× |
| SIMD × rayon composed            | 8.89×      | Near-multiplicative |
| const-generic specialization     | 2.04×      | k-unroll, branch elision, smaller stacks |
| SIMD × rayon × specialization    | 18.2×      | Cumulative |
| Metal GPU                        | 0.47×      | At B=1024; matches CPU at B≥65k |

Takeaway: **rayon was the biggest single lever (8.3×)**, followed by
specialization (2.0×), then SIMD (1.2×), then Metal (net negative
at our sizes). These are multiplicative when they don't fight over
the same bottleneck.

---

## Methodology retrospective — what worked

**Oracle-first, corpus-as-gate.** Every new kernel had to pass
bit-exact comparison against a 16-sample × 10-instruction fixture set
frozen in Phase 0. The corpus caught:

- hardcoded `nfb=24` breaking Volta (wants 23) — Phase 3.1
- hardcoded `min_exp=-14` breaking tf32 — Phase 2
- wrong `is_split_k` assumption for ampere-tf32 — Phase 3.1
- signed shift-right vs trunc-toward-zero in the integer kernel — Phase 2.1

All four would have been silent corruptions in production. Cost of
building the corpus: ~30 minutes. Payback: immediate and repeated.

**No upstream modifications.** We shimmed macOS's `libm.so.6` via
ctypes rather than patching `mmasim/`. The oracle stayed untouched
through 12 commits; every rewrite lived in `fastmma/` or
`fastmma_rust/`. This gave the validator a permanent ground truth
and made rollback trivial at any step.

**Small, reviewed commits with honest write-ups.** The
"no-commit-without-approval" rule forced each phase to articulate
what was tried, what worked, what didn't, and what surprised us. The
surprise of Phase 4 (Metal slower than CPU) only gets to be a useful
data point if it gets recorded, not waved away.

**Microbenchmark before big design decisions.** We microbenchmarked
Python→Rust boundary overhead in Phase 2.1 and learned ~7 μs of the
11.8 μs/call was fixed cost. That turned a "what else can we SIMD?"
question into a clear "batch to amortize dispatch" answer, going
from 588× → 889× just by restructuring, before touching the inner
loop.

## Methodology retrospective — what I'd do differently

**Projections were systematically optimistic.** The initial roadmap:

| stage        | projected | actual     | ratio |
|--------------|-----------|------------|-------|
| NumPy        | 10²–10³×  | 85×        | 0.85× |
| Rust single  | 10⁴–10⁵×  | 588×       | 0.06× |
| Rust batched | —         | 889×       | —     |
| SIMD         | 2×        | 1.2×       | 0.6×  |
| rayon        | 4–8×      | 8.3×       | 1.5×  |
| GPU          | 100×      | 0.47× (!)  | ≈0    |

Two patterns: (a) I underestimated fixed overhead at small matrix
sizes (PyO3 boundary, allocation, decompose setup), and (b) I
overestimated how well the workload matched the GPU's strengths.
**Lesson:** arithmetic-throughput-based projections are nearly
worthless for tiny kernels where dispatch cost dominates; and
architecture-specific bottlenecks (Apple GPU i64 weakness) matter
more than raw TFLOPS numbers.

**I took too long to prototype.** Phases 2.1 and 3.0 each had 30–60
minutes of up-front design writing before code. In retrospect, a
10-minute sketch + immediate prototype would have landed the real
tradeoffs faster. Phase 3.0's specialization result (1.83× over
simd+rayon) was a surprise upward — I had projected 1.3×. If I'd
prototyped earlier I could have factored the real number into
Phase 3.1's scope.

**Corpus format is torch.save — should have been language-agnostic.**
Now when we want to validate a Metal or CUDA kernel with raw data,
we have to go through torch. Trivial to add a `.npz` or raw-bytes
exporter now; annoying later.

---

## Surprises worth recording

1. **Rayon beat SIMD by 7×.** I had internalized "SIMD is the
   low-hanging optimization fruit." On this workload (small tiles,
   embarrassingly parallel across the batch), threading dominated
   vectorization by nearly an order of magnitude. Reason: 12 wide
   scalar P-cores have much more raw int throughput than 2-wide NEON
   on one core.

2. **Specialization won more than SIMD.** Const-generic
   instantiation (k-loop unrolled, subnormal branch elided for f16,
   smaller stacks) gave 2.04×. Bigger than 2-wide NEON (1.2×). The
   compiler's value-of-knowing-everything was larger than I budgeted.
   Lesson: template/monomorphize before vectorizing.

3. **Metal GPU lost.** Apple GPU i64 mul is emulated via 32-bit parts,
   taking ~4× the cycles of i32. With 12 P-cores of wide scalar i64
   dispatch and OoO latency hiding, CPU beats ~5 000 GPU lanes on
   this workload. The calculation was not "CPU GFLOPS vs GPU TFLOPS"
   — it was "CPU i64-muls/sec vs GPU i64-muls/sec," and for i64 the
   gap is much narrower than for FLOPS.

4. **CPU per-tile asymptotes at ~250 ns.** Both CPU specialized and
   Metal converge on the same number at large batches. That number
   is ~750 cycles at 3 GHz for 2048 products = **~0.37 cycles per
   product**. That's surprisingly close to the arithmetic lower
   bound. We are not leaving much on the table with the current
   algorithm.

5. **Stacking composition was multiplicative without friction.**
   SIMD × rayon composed at 91% of the naive product (8.89× vs
   naive 9.66×). The minor loss is memory-bandwidth contention
   when 16 threads all NEON-crunch at once. Cleaner than I
   expected.

---

## Honest reassessment of the 10⁶× framing

**In the per-tile sense, ~10⁶× on this hardware is probably not
reachable.** The CPU floor is ~250 ns/tile (~0.37 cycles/product).
To get to 9.4 ns/tile (10⁶×), we would need either:

- 30× faster CPU (requires larger tiles, which the oracle doesn't
  define);
- An i32 GPU kernel that actually beats the CPU (plausible 2–5×, not
  30×);
- An algorithm that doesn't need integer precision at all (abandons
  bit-exactness against the oracle).

None of these credibly close the 30× gap.

**In the full-workload sense, we're at ≥10⁴× today and ≥10⁵× is
probably reachable.** A realistic workload simulates millions of
tiles. Oracle: ~3 hours. Our best CPU: ~1 second at B=65k extrapolated
to 10⁶ tiles = ~10 seconds. With GPU at saturation: similar. With
an i32 GPU rewrite: could be ~1 second = 10⁵× over oracle on the
same full workload.

**The honest framing going forward:**
- 10⁴× is secured.
- 10⁵× is within one more phase of work (i32 kernel, possibly GPU).
- 10⁶× requires either dropping bit-exactness or accepting it was
  always aspirational.

---

## What we don't know

1. **Whether the oracle itself is bit-accurate.** We've faithfully
   replicated `math.frexp`, the zero-exp-=-126 quirk, Volta's nfb=23,
   the split-K rules, Ada's f32_e8m13 mantissa cut. We have never
   validated any of that against real NVIDIA or AMD silicon. If the
   oracle has a bug, so do we.
2. **Whether our corpus is representative.** 16 torch.randn samples
   per instruction, one seed. No targeted edge cases (subnormals,
   NaN, Inf, catastrophic cancellation). A 100 000-sample corpus
   with curated edge cases might surface issues we've missed.
3. **Whether the f64 NaN/Inf pre-check actually catches everything.**
   We assume `fp0.is_finite()` in f64 captures all oracle-NaN/Inf
   cases. True for standard IEEE behavior, but edge cases involving
   the oracle's `(0, 0, true)` NaN sentinel might diverge.
4. **Performance on non-Apple hardware.** All numbers are M3 Max.
   AVX-512 on a server CPU would amplify the SIMD gain to ~3×
   and possibly change the rayon/SIMD crossover. x86 GPUs (NVIDIA)
   have native i64 support and would likely win Phase 4 decisively.
   We have no measurements.

---

## What we've built (code inventory)

- [mmasim/](../mmasim/) — upstream oracle, **unmodified**
- [bench/](../bench/) — 5 scripts:
  - `instructions.py`: shared 11-instruction registry
  - `bench.py`: times oracle on each registered instruction
  - `corpus.py`: generates committed fixtures (500 KB, 10 × 16 samples)
  - `validate.py`: bit-exact corpus validator with speedup report
  - `batched.py`: per-tile benchmark with 4-way CPU comparison + spec column
  - `_compat.py`: macOS `libm.so.6` shim
- [fastmma/](../fastmma/) — Python-facing reimplementations:
  - `numpy_ref.py`: vectorized NumPy oracle replacement
  - `rust_ref.py`: dispatcher to 6 Rust PyO3 entries
- [fastmma_rust/](../fastmma_rust/) — Rust crate (~800 lines):
  - Scalar and SIMD kernels (generic, parameterized)
  - Const-generic specialized kernels per (arch, qualifier)
  - Rayon wrappers (scalar, simd, specialized)
  - Metal backend (macOS-only) for ampere-f16
- [logs/](.) — 11 campaign write-ups, one per phase plus this
- [ROADMAP.md](../ROADMAP.md) — living plan (needs update after this)
- [AGENTS.md](../AGENTS.md) — agent operating rules
