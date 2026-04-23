# ROADMAP — bit-accurate MMA-Sim, ~10⁶× faster

Goal: a drop-in replacement for `mmasim` that produces **bit-identical** outputs
for every supported (arch, instruction, dtype) combination, at a target of
~10⁶× throughput over the current Python/Torch impl.

Living doc — revise as campaigns land. Campaign notes go in [logs/](logs/).

## Guiding principles

- **Bit-exact or it doesn't count.** Every speedup is gated on matching the
  Python reference bit-for-bit across a regression corpus.
- **Measure before optimizing.** No intuition-driven rewrites without a
  baseline number to beat.
- **Keep the reference alive.** The Python impl is the oracle. Never delete
  it, never let it drift.

## Phases

### Phase 0 — Ground truth

- **Benchmark harness.** Time the Python impl on a representative instruction
  set (one per arch family + block-scaled + fp4). Establish ms/instruction.
- **Regression corpus.** Generate `(inputs → output_bits)` fixtures (~10k per
  instruction type) from the Python impl. This is the oracle for everything
  downstream.

### Phase 1 — Vectorized NumPy reference

- Rewrite `fused_sum` / `nv_fused_dot_add` without inner Python loops. Batched
  `frexp`, batched align-and-truncate as integer ops.
- Validate against the corpus. Expected: 10²–10³× speedup, 0 bit diffs.
- Purpose: a faster oracle, and a semantic sanity check before going native.

### Phase 2 — Integer fixed-point core in Rust

- Reformulate the kernel as pure integer arithmetic: mask+shift for
  decomposition, `i128`/`i256` fixed-point accumulator at max exponent,
  shift-round on normalize. No float ops in the hot path.
- Prototype one workhorse instruction first (m16n8k16 f16×f16→f32).
- PyO3 binding. Validate against corpus. Measure ceiling per core.
- Decision point: if single-instruction speedup is 10⁴–10⁵×, expand.

### Phase 3 — Codegen across the ISA

- Specialized kernel per `(tile_shape, A_dtype, B_dtype, C_dtype, accum_width,
  rounding_mode)`. Generated from a compact spec, not hand-written per combo.
- Full corpus coverage.

### Phase 4 — SIMD + batch-of-MMAs

- Process N independent MMAs per SIMD vector (AVX-512 / NEON).
- Multi-threading across tiles.
- This is where the final ~10²× comes from.

### Phase 5 (optional) — GPU backend

- CUDA/Triton port of the integer kernel, one MMA per warp.
- Only if the target workload is full-GEMM simulation at scale.

## Non-goals (for now)

- Supporting instructions/dtypes beyond what upstream already implements.
- Replacing the Python API surface. Wrapper compat only.
- Full-matmul tiling semantics — this is still a per-instruction simulator.

## Open questions

- What's the real use case driving the 10⁶× target — single-instruction
  throughput, or full-GEMM simulation? Changes where parallelism lives.
- Does `math.frexp` / `libm.fma` introduce platform dependence we need to
  freeze before building the corpus?
- Is the `f32_e8m13` / split-k / block-scale handling in the Python impl
  actually correct vs. real hardware? (Out of scope to fix, but worth knowing
  if the oracle itself has bugs.)
