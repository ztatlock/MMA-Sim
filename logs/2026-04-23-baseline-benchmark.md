# 2026-04-23 — Phase 0 baseline benchmark + smoke corpus

**Status:** landed
**Goal:** establish ms/instruction for the Python oracle and freeze a small
bit-exact regression corpus. This is the "1×" everything downstream beats.

## Environment

- Python 3.12.13, torch 2.11.0
- Darwin 25.4.0 arm64 (Apple Silicon, M-series)
- Single-threaded, CPU-only (the oracle is CPU-bound by design)

## Portability note

Upstream [arithmetic.py:6](../mmasim/simulator/arithmetic.py:6) hardcodes
`ctypes.CDLL("libm.so.6")`, which is Linux-only. Rather than modify the
oracle, [bench/_compat.py](../bench/_compat.py) installs a one-line ctypes
shim on Darwin that redirects `libm.so.6` to `libSystem.B.dylib`. Verified
that `fmaf(2,3,4) == 10.0` — IEEE FMA semantics match.

Worth proposing upstream as a trivial portability fix (pick the right lib
at import time by `platform.system()`), but out of scope for the fork.

## Baseline numbers (median of 3 reps, 1 call each)

| label             | arch          | mnk        | median ms | µs / output |
|-------------------|---------------|------------|-----------|-------------|
| volta-f16         | Volta         | 8×8×4      | 1.97      | 30.8        |
| turing-f16        | Turing        | 16×8×8     | 4.78      | 37.4        |
| ampere-f16        | Ampere        | 16×8×16    | 9.43      | 73.7        |
| ampere-bf16       | Ampere        | 16×8×16    | 9.19      | 71.8        |
| ampere-tf32       | Ampere        | 16×8×8     | 7.42      | 57.9        |
| ampere-f64        | Ampere        | 8×8×4      | 1.24      | 19.4        |
| ada-fp8-e4m3      | Ada Lovelace  | 16×8×32    | 13.10     | 102.4       |
| hopper-f64        | Hopper        | 16×8×16    | 8.26      | 64.6        |
| blackwell-mxfp8   | RTX Blackwell | 16×8×32    | 10.03     | 78.3        |
| blackwell-mxfp4   | RTX Blackwell | 16×8×64    | 34.63     | 270.6       |

Observations:

- **Per-output cost scales roughly with k**, as expected: the inner
  fused-dot-add is O(k) in Python loop iterations. Volta k=4 → ~30 µs,
  Ampere k=16 → ~70 µs, Ada k=32 → ~100 µs, Blackwell mxfp4 k=64 → ~270 µs.
- **F64 is faster per-output** than f16/bf16/fp8 because it takes the
  simpler `fma` path ([nv_ptx.py:70-71](../mmasim/simulator/nv_ptx.py:70)),
  not the full significand/exponent extraction.
- **mxfp4 is the slowest** — k=64 plus `unpack_fp4_tensor` has its own
  Python loop on top of the reduction.
- Order of magnitude: **all are in the 1–35 ms/call range**. Workhorse
  (ampere-f16) is **~9 ms/call**. Speedup target (10⁶×) implies ~9 ns/call,
  which is on the order of a few hundred CPU cycles — feasible only with
  a compiled integer-fixed-point core and likely batching.

## Corpus

Committed smoke corpus at `bench/fixtures/*.pt`, 16 samples per instruction,
seed 1000–1015. Total ~500 KB. Summary hashes:

```
volta-f16         9587336acba827f4
turing-f16        bc13e9ae2a5e748a
ampere-f16        4fbb54b42ccb3e94
ampere-bf16       4108157d80c35bc8
ampere-tf32       8f3240a787a15793
ampere-f64        9d88fe13ef1423dc
ada-fp8-e4m3      676916eaa6bef4d6
hopper-f64        71cf1c6192ef3d74
blackwell-mxfp8   a698e0720119f52c
blackwell-mxfp4   56a47db6d2bd4778
```

Any reimplementation running the same seeds through the same instruction
registry must produce the same digests. Digests may shift if we bump
the Torch version (RNG compatibility) — worth re-recording when we do.

## Gaps / follow-ups

- **Plain `e2m1` mma** is in the ISA qualifier list but `mma.__call__`
  doesn't unpack fp4 ([nv_ptx.py:66-96](../mmasim/simulator/nv_ptx.py:66));
  only `mma_block_scale` does. Probably an upstream oversight or deliberate
  (fp4 in practice is always block-scaled). Dropped from the registry.
- **AMD (MFMA)** not yet covered. Add when we need AMD coverage.
- **wgmma / tcgen05mma** (Hopper/Blackwell collective matmuls) also not
  covered yet — different ISA surface, add in a follow-up.
- Corpus seeds drawn from a single Torch RNG — we should verify RNG output
  is stable across torch versions before pinning digests as regression
  checks in CI. Or: drop to raw byte-pattern fixtures that don't depend on
  torch.randn at all.

## What's next

Phase 1: vectorized NumPy reference of `fused_sum` / `nv_fused_dot_add`.
Validate against this corpus. Expected 10²–10³× speedup with zero bit diffs.
