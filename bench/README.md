# bench/

Phase 0 ground-truth tooling: benchmark the Python oracle, and freeze a
bit-exact regression corpus that any reimplementation must reproduce.

## Setup

```
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch numpy
```

## Run

```
python -m bench.bench                 # time every registered instruction
python -m bench.bench --only ampere-f16 --reps 5

python -m bench.corpus                # write fixtures/<label>.pt (16 samples)
python -m bench.corpus --samples 256  # bigger local run (don't commit)
```

Outputs of `bench.corpus` are torch.save'd bundles containing `{spec, samples}`
where each sample is `{inputs: {A,B,C[,scale_A,scale_B]}, output}`. The "summary
hash" printed alongside is a sha256 of the concatenated per-sample output
hashes — any future reimplementation run with the same seed and registry
must produce the same digest.

## Layout

- [instructions.py](instructions.py) — shared registry of representative MMAs, plus input
  generation. Edit here to expand coverage.
- [bench.py](bench.py) — timing harness.
- [corpus.py](corpus.py) — fixture generator.
- `_compat.py` — macOS libm shim. Imported transitively.
- `fixtures/` — checked-in smoke corpus (small). Bigger corpora stay local.

## Scope

- NVIDIA `mma` and `mma_block_scale` across Volta → RTX Blackwell.
- AMD MFMA not yet covered.
- Plain `e2m1` mma is listed in the ISA but the upstream sim doesn't unpack
  fp4 in `mma.__call__`; fp4 is exercised through the block-scaled variants.
