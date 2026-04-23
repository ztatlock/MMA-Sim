# 2026-04-23 — Phase 1, vectorized NumPy reference

**Status:** landed (partial coverage)
**Goal:** reimplement the oracle's MMA path in vectorized NumPy, match bit-for-bit
on the corpus, measure the floor for what "no more Python loops" buys us.

## What shipped

- [fastmma/numpy_ref.py](../fastmma/numpy_ref.py) — drop-in `mma(arch, qualifier)` that mirrors
  `mmasim.simulator.nv_ptx.mma.__call__` without per-element Python loops.
- [bench/validate.py](../bench/validate.py) — loads the corpus, runs a named impl, compares output
  bytes to the oracle bit-for-bit, and times impl vs oracle.

## Results

6 of 10 fixtures validated, all bit-exact (16/16 samples each). Speedups
relative to the Python oracle on the same machine (Darwin arm64, CPU-only,
median of 5 reps):

| label           | ms (impl) | ms (oracle) | speedup |
|-----------------|-----------|-------------|---------|
| volta-f16       | 0.047     | ~1.9        | 39×     |
| turing-f16      | 0.051     | ~4.8        | 95×     |
| ampere-f16      | 0.110     | ~9.4        | 84×     |
| ampere-bf16     | 0.106     | ~9.2        | 87×     |
| ampere-tf32     | 0.106     | ~7.4        | 67×     |
| ada-fp8-e4m3    | 0.135     | ~13.1       | 96×     |

## What's vectorized

Three loops collapsed into one pass:

1. **Outer (m, n)** over output tile — now vectorized via broadcasting.
2. **Inner k-reduction** in `fused_dot_add` — now a single `_fused_sum` with
   `.sum(axis=-1)`.
3. **Per-element significand/exponent extraction** — now `np.frexp` on the
   whole tensor, followed by vectorized subnormal-flush and zero-exp quirk.

The math matches the oracle exactly: each accumulation term is truncated
to `nfb` fractional bits at the shared max exponent before summation. The
sum stays in f64 (integer-valued f64, within 53-bit mantissa range given
our k ≤ 64, nfb ≤ 35), so order-of-summation doesn't affect the result.

## Bit-exactness notes

- **TF32**: replicated via the same `>> 13 << 13` mask on the int32 view.
- **NaN pattern**: oracle writes `0x7FFF_FFFF` for NaN outputs; we match.
- **Subnormal flush**: done at the *target dtype's* min exponent, not f64's.
- **Zero-exp quirk**: oracle hardcodes `exp = -126` when `sig == 0`,
  regardless of dtype. Looks artifactual but we replicate it.
- **NaN/Inf short-circuit**: oracle does an f64 GEMM pre-check and returns
  `fp64_sum` directly when any product overflows or is NaN. We do the same
  check and splice those cells back in after the fused path.

First-shot pass on every supported fixture; no iteration needed.

## What's not vectorized yet

- **f64 path** (`ampere-f64`, `hopper-f64`): the oracle uses serial
  `libm.fma(a[l], b[l], sum)` across k, which requires *single-rounded* IEEE
  FMA per step. NumPy 2.4 has no `np.fma`. Either call libm via ctypes
  per-element (defeats vectorization) or wait for Phase 2 (Rust with x86
  FMA3 / NEON FMLA intrinsics). Skipped for now.
- **Block-scaled (mxfp8, mxfp4)**: different call signature and code path
  (`mma_block_scale`), plus the mxfp4 path does `unpack_fp4_tensor`. Not yet
  implemented in `fastmma`.
- **fp4 as plain mma input**: upstream gap (see Phase 0 log).
- **wgmma / tcgen05mma**: not covered in the corpus yet.

## Reality check on the 10²–10³× claim

Roadmap projected 10²–10³× for the NumPy phase. We got **39–96×**. Why
lower:

- The matrices are *tiny*. Ampere f16 workhorse is 16×8×16 — only 2048
  products. NumPy overhead (call dispatch, frexp, concatenate, exp2) is a
  significant fraction of the work at this scale.
- `np.exp2` is a real transcendental call, not a cheap bit-shift. The
  oracle hits `math.trunc` + `2.0**n` per element in Python, which is
  *also* slow — so both sides pay for it. Ratio ends up modest.
- We cast to f64 internally and do a full f64 GEMM for the NaN/Inf pre-
  check. Could skip pre-check and detect post-hoc, but it's a correctness
  belt-and-suspenders for now.

The speedup ceiling here is probably ~200× for these tile sizes. To go
further we'd need to (a) fuse the inner ops into a C/Rust kernel, or
(b) batch many MMAs together so NumPy amortizes. Phase 2 does (a).

**Conclusion: semantic model is correct and the vectorized shape is right.
Headline win has to come from compilation + larger effective batch.**

## What's next

Phase 2: Rust integer fixed-point core. Pick one workhorse instruction
(m16n8k16 f16×f16→f32, Ampere), build it end-to-end with PyO3 bindings,
validate against this corpus. That benchmark will tell us the realistic
ceiling per core.
