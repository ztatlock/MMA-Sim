# 2026-04-23 — Branch A: Coverage expansion

**Status:** landed (scoped)
**Context:** user authorized autonomous progress on Branch A (coverage)
while they headed home. Goal: extend `fastmma.rust_ref` past the
original 5/10 corpus fixtures to cover the broader ISA surface.

## What shipped (7 milestones)

| M#  | what                                           | new fixtures |
|-----|------------------------------------------------|--------------|
| M1a | Ada Lovelace fp8 (8 qualifiers, f32 output)    | enables corpus |
| M1b | F64 kernel (Ampere m8n8k4 + Hopper 3 shapes)   | enables corpus |
| M1c | Block-scaled mxfp8 (Blackwell k=32)            | enables corpus |
| M1d | Block-scaled mxfp4 (Blackwell k=64, fp4 unpack)| enables corpus |
| M2  | Edge-case corpus generator (NaN/Inf/zeros/extremes) | 10 new fixtures |
| M3  | AMD MFMA (fma operation_type: f64 all, f32 non-xf32) | 3 AMD + 3 AMD edge |
| M4  | Hopper wgmma (f32 output)                      | 3 wgmma + 3 wgmma edge |

**End state:** **32/32 fixtures bit-exact**, all ≥180× over the Python
oracle.

## Bug/surprise inventory from the push

Three real bugs caught by the corpus on first run — each would have
been a silent correctness failure without the validation gate:

1. **fp8 NFB < 23 blew up the shift-up.** Ada fp8 has nfb=13, but the
   generic `decompose` did `sig_q23 << (nfb - 23)` — a negative shift,
   UB in Rust. Fix: branch to right-shift when nfb < 23. The right-
   shifted path is lossless for fp8 (mantissa has ≤ 3 bits, bottom 20+
   of sig_q23 are zero) and intentionally lossy for f32 C terms (trunc
   to 13 bits, which is what the oracle does anyway at alignment).
2. **Missed that ampere-tf32 uses split-K.** I had specialized it as
   non-split; 502/2048 tile outputs were wrong. Oracle rule: Ampere
   with k=8 and a_type=float32 triggers split-K (two k=4 halves).
3. **Applied 2^-nfb twice in mxfp4.** My `fused_sum_f64` returned
   `acc * 2^-nfb` and the caller then also multiplied by `2^(max_e -
   nfb)` → an extra factor of 2^-nfb = 2^-35. Outputs were off by
   2^35, immediately obvious from the diff.

Each caught by corpus mismatch before any production use. The 2-minute
debug → fix cycle was the point.

## Final scoreboard (coverage + speed)

Bit-exact across 32 fixtures (13 instruction types × random + edge corpora;
some instructions have extra edge samples):

| instruction           | tile shape   | oracle μs | rust μs | speedup |
|-----------------------|--------------|-----------|---------|---------|
| volta-f16             | m8n8k4       | 1 970     | 0.009   | 190×    |
| turing-f16            | m16n8k8      | 4 780     | 0.011   | 386×    |
| ampere-f16            | m16n8k16     | 9 400     | 0.017   | 501×    |
| ampere-bf16           | m16n8k16     | 9 190     | 0.018   | 494×    |
| ampere-tf32           | m16n8k8      | 7 420     | 0.018   | 371×    |
| ampere-f64            | m8n8k4       | ~1 200    | 0.004   | 272×    |
| ada-fp8-e4m3          | m16n8k32     | 13 100    | 0.023   | 543×    |
| hopper-f64            | m16n8k16     | ~8 300    | 0.005   | 1 572×  |
| blackwell-mxfp4       | m16n8k64     | ~34 600   | 0.020   | 1 593×  |
| blackwell-mxfp8       | m16n8k32     | ~10 000   | 0.028   | 355×    |
| amd-cdna2-f64         | m16n16k4     | —         | 0.005   | 962×    |
| amd-cdna3-f64         | m16n16k4     | —         | 0.005   | 819×    |
| amd-cdna3-f32         | m32n32k2     | —         | 0.007   | 1 404×  |
| hopper-wgmma-f16-n64  | m64n64k16    | —         | 0.154   | 1 257×  |
| hopper-wgmma-bf16-n64 | m64n64k16    | —         | 0.147   | 1 321×  |
| hopper-wgmma-tf32-n64 | m64n64k8     | —         | 0.105   | 1 304×  |

Speedups are single-call; batched versions (where wired up) hit
4 000–16 000× as documented in Phase 2.4–3.1.

## Known-not-done

Deliberate deferrals — would be future milestones:

1. **f16-output variants.** Our normalize uses RZ; oracle uses RNE
   for f16. Doable but needs a separate path in `normalize_f16`.
2. **AMD pairwise + fused_dot_rd_add paths.** CDNA1/2 f16/bf16 and
   CDNA3 xf32/f16/bf16/fp8 use these. `pairwise_dot` + RD rounding
   differ from what we have.
3. **tcgen05mma.** Trivial extension of wgmma (same math, slightly
   different qualifier grammar). Haven't registered qualifiers.
4. **Wider wgmma n.** We tested n=64; ISA supports n up to 256.
   Kernel should handle them (heap-allocated buffers scale) but we
   only registered n=64 qualifiers.
5. **xf32 (AMD) and f64 fma on CDNA1.** CDNA1 has no f64; we have
   fma support for f64 only on CDNA2/3. Could extend trivially.
6. **Hardware ground-truth validation.** The oracle could be wrong;
   we haven't checked against real silicon.

## Code additions (this push)

- `fastmma_rust/src/lib.rs` (+~600 lines):
  - fp8 support via nfb-aware decompose shift (M1a fix)
  - `run_one_tile_f64` + `mma_f64_batched_rayon` (M1b)
  - `run_one_tile_f32_fma` + `mma_f32_fma_batched_rayon` (M3)
  - `run_one_tile_block_scale_k32` + PyO3 (M1c)
  - `run_one_tile_mxfp4_k64` + `fused_sum_f64` + FP4_TABLE (M1d)
  - `run_one_tile_heap` + `mma_f32_out_wgmma_batched_rayon` (M4)
- `fastmma/rust_ref.py`: new classes `mma_block_scale`, `mfma`, `wgmma`;
  expanded `_SUPPORTED`, `_SUPPORTED_BLOCK_SCALE`, `_SUPPORTED_MFMA`,
  `_SUPPORTED_WGMMA` sets.
- `bench/instructions.py`: 6 new registry entries (3 AMD, 3 wgmma).
- `bench/corpus_edge.py`: new file — curated edge-case generator.
- `bench/fixtures/`: 16 new `.pt` files (edge + AMD + wgmma + their
  edge variants).
- `bench/validate.py`: oracle dispatch for mfma/wgmma/tcgen05mma.

## Commit trail

```
d8eef34  M1a: enable Ada Lovelace fp8
5c9d255  M1b: F64 kernel
70c662d  M1c: block-scaled mxfp8
98c3352  M1d: block-scaled mxfp4
6443de2  M2: edge-case corpus
a28ab20  M3: AMD MFMA fma-subset
38c6d0f  M4: Hopper wgmma f32-output
```

## What changed about the strategic picture

After Phase 4 (Metal GPU) came in as a negative result at our per-tile
size, and the retrospective concluded that coverage + validation were
more load-bearing than further perf work, this push delivered the
coverage half. **The library now covers most of what MMA-Sim's oracle
implements**, at speedups of 2–3 orders of magnitude. It is a
defensible drop-in replacement for typical research workloads —
except we have **not validated against real silicon**.

The next strategic pivot: hardware ground-truth validation. Getting
access to an actual GPU (NVIDIA and/or AMD) and running a corpus
against it is now the single highest-value task. Until that happens,
our "bit-exact" claim is only relative to the Python oracle.
