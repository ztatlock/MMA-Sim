# 2026-04-23 — Phase 4, Metal GPU port (ampere-f16)

**Status:** landed, bit-exact, **slower than CPU at our sizes**
**Goal:** port the ampere-f16 workhorse to Metal, measure, learn.

## What shipped

- `fastmma_rust/Cargo.toml`: `metal = "0.29"` and `objc = "0.2"` under a
  `cfg(target_os = "macos")` target.
- `fastmma_rust/src/metal_ampere_f16.rs`: MSL source + Rust dispatcher.
  One threadgroup per tile, 128 threads (8 × 16), one thread per
  output. Integer pipeline mirrors the CPU specialized kernel; final
  `pack_f32` builds the f32 bit pattern directly (no f64 intermediate).
- PyO3 entry `mma_metal_ampere_f16`, Python `call_batched_metal`.
- OnceLock-cached Metal context (device + pipeline + queue) so per-call
  overhead is just buffer allocation + dispatch + wait.

## Bit-exactness

**First-run pass** against all 16 ampere-f16 corpus samples (2048 outputs
total). No mismatches. This is the result I expected to be hard and
wasn't — the integer pipeline translates cleanly to MSL, and building
the f32 output via bit construction avoids any f64→f32 rounding
ambiguity.

## Speed (M3 Max, unified memory)

Batch-size sweep:

| tiles  | CPU spec (μs) | Metal (μs) | CPU (μs/tile) | Metal (μs/tile) | Metal/CPU |
|--------|---------------|------------|---------------|-----------------|-----------|
| 128    | 147.6         | 320.2      | 1.153         | 2.502           | 2.17×     |
| 1 024  | 478.6         | 1 028.3    | 0.467         | 1.004           | 2.15×     |
| 4 096  | 1 162.0       | 1 692.9    | 0.284         | 0.413           | 1.46×     |
| 16 384 | 4 165.7       | 5 466.8    | 0.254         | 0.334           | 1.31×     |
| 65 536 | 16 337.0      | 16 284.1   | 0.249         | 0.248           | 1.00×     |

**Metal does not beat the specialized CPU kernel at any batch size
tested.** They tie at ~65k tiles; beyond that the GPU may edge ahead
marginally. The CPU kernel asymptotes at ~0.25 μs/tile (the rayon-
parallelized specialized per-tile math).

## Why the GPU didn't win

Three factors, roughly in order of impact:

1. **i64 arithmetic on Apple GPU is slow.** The MSL kernel uses `long`
   (i64) for the integer significand pipeline. Apple M-series GPUs
   don't have native 64-bit integer multiply units; i64 mul is emulated
   via 32-bit parts, taking ~4× as many cycles as i32. Our kernel does
   ~8 i64 muls per output × 128 outputs per tile = ~1k i64 muls/tile.
   That's a lot of serial-ish work per thread.
2. **Dispatch overhead is non-trivial at small batches.** Each Metal
   call does: create 4 MTLBuffers (3 input + 1 output), create command
   buffer, encode, commit, wait_until_completed. That's ~200–500 μs of
   fixed cost at the PyO3/Metal boundary. For 128 tiles × 2 μs each,
   the overhead doubles the total time.
3. **Our tiles are tiny.** 128 outputs per tile × 1024 tiles = 131k
   threads total. M3 Max has ~40 GPU cores × 128-wide SIMD = ~5k
   concurrent lanes. We ARE saturating parallelism at batch=1024, but
   each lane's work is i64-mul heavy which is the GPU's weakness.

The CPU side, meanwhile, has 12 P-cores with wide scalar integer
dispatch (6+ i64 ops/cycle) and an aggressive out-of-order window that
hides i64 mul latency across adjacent outputs. 12 × 6 = 72 effective
i64 ops/cycle, and the kernel fits entirely in L1/L2. It turns out
that's just more throughput than 5k GPU lanes running slow i64.

## What would actually make Metal win

An **i32 pipeline** instead of i64. For f16 inputs at scale 2^12 (half
of NFB=24), each int_sig fits in i16 (≤ 2^13), products in i32 (≤ 2^26),
sums in i32 (for k=8 per half, well under 2^31). This would:
- 4× faster i32 mul vs i64 mul on Apple GPU
- 2× better memory bandwidth on the int_sig arrays
- Cleaner vectorization opportunities within threads

A full i32 rewrite might be 2–5× faster than the i64 Metal kernel and
would plausibly beat the CPU at moderate batch sizes. But it's a
rewrite, not a tweak — the alignment shifts need re-thinking and the
tf32/f32-C path needs different scale handling.

**Other options in the GPU design space:**
- f32-native pipeline (accept some precision loss relative to oracle):
  fast and parallel on GPU but not bit-exact. Could be valuable as a
  "fast approximate" mode.
- Batched kernel: one threadgroup processes multiple tiles in series,
  amortizing setup. Not clear this helps since we're already running
  ~kilo-groups.
- `MTLStorageModeManaged` + explicit sync: unlikely to help on unified
  memory.

## Honest reassessment of the 10⁶× picture

The retrospective framed GPU as the ~100× leap needed to get from 10⁴×
to 10⁶×. **That framing was wrong for our specific kernel.** The
reason: I was thinking in terms of arithmetic throughput (Metal's TFLOPS
vs CPU's GFLOPS), but our kernel is NOT floating-point — it's
integer-64-multiply-limited on a GPU that wasn't designed to excel at
i64. The CPU's absolute advantage here (wide scalar i64 dispatch × 12
cores) is hard to beat with a GPU that's ~100× parallel but ~4× slower
per i64-mul lane.

Where GPU **would** give 10⁶× per tile:
- If we used an i32 pipeline (requires kernel redesign)
- If we used an f32 pipeline (requires relaxing bit-exact gate)
- If the tile were much larger so parallelism dominates

## So where's the remaining 2 orders of magnitude actually come from?

Candidly, on this hardware, possibly **nowhere without changing the
problem statement**. The CPU specialized kernel is running at ~250
ns/tile on 12 P-cores — that's ~3k cycles/tile for 2048 products. Close
to the arithmetic lower bound.

To cross 10⁶× per-tile (9.4 ns/tile) credibly would need:
- A redesigned **i32** CPU kernel (~2× faster) — feasible on CPU too
- Combined with AVX-512 on a server CPU (another ~2–4×)
- Running on a server with 64+ cores (another ~4×)

That gets us to ~3–5 × 10⁵×. Still not quite 10⁶×. The last bit needs
either (a) a really favorable GPU kernel or (b) a different algorithm
that doesn't need integer precision at all.

For **full-workload throughput** (e.g., simulating a full GEMM as
millions of tiles), the story is different: we've been measuring
per-tile cost at modest batches. Amortizing dispatch overhead across
millions of tiles, our CPU kernel is already near peak. A GPU with an
i32 kernel could likely get to 10⁶× on full workloads — worth trying if
full-GEMM throughput is the real target.

## What's next

Pragmatic options, not exclusive:

1. **Phase 4.1 — i32 Metal kernel.** Rewrite the MSL kernel with
   scale 2^12 operand decompose, 32-bit product arithmetic. Could
   credibly beat CPU specialized by 2–5× at our sizes; at large
   batches, maybe 10×. Bit-exactness should be achievable with care.
2. **Phase 5 — CPU i32 kernel.** Same idea on CPU. Possibly also
   a win there (i32 is 2× the SIMD throughput of i64 on NEON).
3. **Coverage pivot.** Accept CPU specialized at ~10⁴× as the usable
   ceiling and go extend coverage to all 10 fixtures + AMD + wgmma.
   Less heroic but arguably more useful for the actual use case.
4. **Validation pivot.** Also accept current speed, and invest in
   hardware-ground-truth validation — running the corpus against real
   NVIDIA silicon to check the oracle (and hence our impl) is
   actually bit-accurate.

My read: the i32 rewrite has a plausible 10× story and isn't crazy
expensive. If the ultimate workload is big GEMMs, it pays off. But
given your stated goal ("trustworthy, reliable, bit-accurate
simulations at scale"), **coverage + validation** is probably the
more load-bearing work. The perf we have is already good enough for
many real-world research workloads.

Worth a conversation before picking.

## Files / dependencies added

- Cargo deps: `metal = "0.29"`, `objc = "0.2"` (macOS-only target)
- `fastmma_rust/src/metal_ampere_f16.rs` (~200 lines Rust + ~150 lines
  MSL)
- `fastmma/rust_ref.py::call_batched_metal` method (raises
  NotImplementedError for other instructions)
