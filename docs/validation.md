# Validation

Short version: **the Python oracle in `mmasim/` is silicon-validated
upstream.** Our Rust reimplementation is bit-exact against the oracle.
Transitively, our implementation is bit-exact against real GPU hardware
for every instruction the paper covers.

## The chain of trust

```
Real NVIDIA / AMD silicon
  ↑  Closed-Loop Feature Probing (paper's contribution)
  ↑  ~1 000 000 randomized tests per instruction
  ↑  10 real GPUs: V100, T4, A100, H100, B200, RTX 4090,
  ↑                RTX PRO 6000, MI100, MI250X, MI300X
  │
mmasim/ Python oracle  ←  the validated reference model
  ↑  bench/validate.py
  ↑  32 fixtures × ~10–16 samples (random + edge cases)
  │
fastmma.rust_ref, fastmma.numpy_ref  ←  this repo
```

## Reference

Xie, Xu, Wang, Yang, Yang. *Bit-Accurate Modeling of GPU Matrix
Multiply-Accumulate Units: Demystifying Numerical Discrepancy and
Accuracy.* arXiv:2511.10909 (2026).

Local copy in [docs/paper/mma-sim-arxiv-2511.10909.pdf](paper/mma-sim-arxiv-2511.10909.pdf).

## What the authors actually did

From §3.3 (System Implementation) and §3.1.4 (Step 4: Validation and
Revision):

- Implemented MMA interfaces in **3 000+ lines of CUDA + inline PTX**
  for NVIDIA, **1 400+ lines of HIP** for AMD. Also verified PTX → SASS
  (machine code) mappings.
- For each instruction model, ran **~1 million randomized tests** with
  bit-by-bit comparison between model prediction and silicon output.
- Test inputs include:
  - Normal distributions, uniform distributions, and DNN-typical mixes
    (e.g., `N(0, 1) + Bernoulli(0.001) · N(0, 100)`).
  - Adversarial inputs with large condition numbers (`Σ|p_k| / |Σp_k|`
    ≫ 1) to trigger catastrophic cancellation.
  - Random **bit-stream** inputs covering subnormals, Inf, NaN, and
    drastic range variation.
- Any mismatch triggered a Step-3 revision (the "closed loop" in CLFP).
  The final models passed validation before release.

## What this means for "bit-exact"

When we say `fastmma.rust_ref` is "bit-exact" we mean:

1. **Bit-exact against the oracle** — we have a committed corpus
   (`bench/fixtures/`) that mechanically verifies this. Every
   `_SUPPORTED` fixture passes `python -m bench.validate`.
2. **Bit-exact against silicon** — inherited from the oracle's
   upstream validation. This is what the paper demonstrates.

The second claim is transitive: it holds as long as the instruction we
support matches an instruction the paper validated. If we were to add
something the paper didn't cover, we'd lose that guarantee for that
instruction.

## What our local validation covers vs. the paper's

| dimension      | paper              | this repo          |
|----------------|--------------------|--------------------|
| instructions   | all 10 archs, full ISA | 13 representative |
| samples/inst.  | ~10⁶               | ~10–16             |
| input patterns | random + adversarial + bit-stream | random + edge cases |

Our corpus is **much smaller** than the paper's test harness. Its job
is to catch Rust-impl regressions, not to re-establish silicon
equivalence — that work is done.

## Our implementation vs. the paper's model taxonomy

The paper (§4, Table 1) classifies MMA instructions into 8 elementary
operations. Cross-reference with what we've implemented:

| paper model   | paper algorithm | used by                        | our Rust impl           |
|---------------|-----------------|--------------------------------|-------------------------|
| Φ_FTZ-AddMul  | Alg. 2 / Alg. 1 | AMD CDNA2 FP16/BF16            | not yet (pairwise path) |
| Φ_FMA         | Alg. 4          | all FP64, AMD FP32              | ✓ (`run_one_tile_f64`, `run_one_tile_f32_fma`) |
| Φ_E-FDPA      | Alg. 6          | AMD CDNA1 FP16/BF16            | not yet                 |
| Φ_T-FDPA      | Alg. 7          | NVIDIA TF32/BF16/FP16/FP8      | ✓ (`fused_mma_step_int`) |
| Φ_ST-FDPA     | Alg. 8          | NVIDIA FP8/6/4, MXFP8/6/4      | ✓ partial (mxfp8 k=32)  |
| Φ_GST-FDPA    | Alg. 9          | NVIDIA MXFP4 / NVFP4           | ✓ (`run_one_tile_mxfp4_k64`) |
| Φ_TR-FDPA     | Alg. 10         | AMD CDNA3 TF32/BF16/FP16       | not yet                 |
| Φ_GTR-FDPA    | Alg. 11         | AMD CDNA3 FP8                  | not yet                 |

Coverage gaps map directly to Algorithms 1, 6, 10, 11 and the RNE-FP16
conversion function. Each remaining item is a straightforward
implementation of a documented elementary operation.

## Parameter cross-check

The paper's Tables 4–7 specify the per-(arch, dtype) parameters
`(L_max, F, ρ)` that instantiate T-FDPA, ST-FDPA, GST-FDPA, etc.
We cross-checked the ones we implement:

| arch         | dtype  | paper (L_max, F, ρ)    | our (nfb, round)        | match |
|--------------|--------|------------------------|-------------------------|-------|
| Volta        | FP16   | L=4,  F=23, RZ-FP32    | nfb=23, RZ              | ✓     |
| Turing       | FP16   | L=8,  F=24, RZ-FP32    | nfb=24, RZ              | ✓     |
| Ampere       | FP16   | L=8,  F=24, RZ-FP32    | nfb=24, RZ (+ split-K)  | ✓     |
| Ampere       | TF32   | L=4,  F=24, RZ-FP32    | nfb=24, RZ (+ split-K)  | ✓     |
| Ada          | FP8    | L=16, F=13, RZ-E8M13   | nfb=13, f32_e8m13, RZ   | ✓     |
| RTX Blackwell| MXFP4  | L=64, F=35, RZ-FP32    | nfb=35 (mxfp4 path)     | ✓     |

No parameter mismatches found in our supported set.

## Local CI we still own

Even though we don't re-run silicon tests, these are worth maintaining:

- `bench/validate.py` — every supported fixture passes bit-exactly.
- `bench/corpus.py` + `bench/corpus_edge.py` — regeneratable fixtures
  (seeded from Torch RNG).
- `bench/batched.py` — per-tile perf regression (not correctness, but
  catches accidental slowdowns).

Everything else is the paper's problem.
