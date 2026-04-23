"""Regression corpus generator.

Produces (inputs, output) fixtures from the Python oracle for every
registered instruction. Consumers (vectorized NumPy reference, Rust
kernel) must match the output bit-for-bit.

Layout:
    bench/fixtures/<label>.pt
        {
          "spec": {"label", "arch", "qualifier", "kind"},
          "samples": [
            {"inputs": {"A": tensor, "B": tensor, ...}, "output": tensor},
            ...
          ],
        }

Usage:
    python -m bench.corpus                  # small smoke corpus (committed)
    python -m bench.corpus --samples 256    # bigger local run
    python -m bench.corpus --only ampere-f16
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import time

import torch

from . import instructions

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


def _tensor_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def _build(spec: instructions.InstSpec, n_samples: int, base_seed: int) -> dict:
    op = instructions.make(spec)
    samples = []
    for i in range(n_samples):
        inputs = instructions.gen_inputs(op, seed=base_seed + i)
        output = instructions.invoke(op, inputs)
        samples.append({"inputs": inputs, "output": output.clone()})
    return {
        "spec": {
            "label": spec.label,
            "arch": spec.arch,
            "qualifier": spec.qualifier,
            "kind": spec.kind,
        },
        "samples": samples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=16,
                    help="number of samples per instruction")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--out", type=pathlib.Path, default=FIXTURES_DIR)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    specs = instructions.REGISTRY
    if args.only:
        specs = [s for s in specs if s.label == args.only]
        if not specs:
            raise SystemExit(f"no instruction with label: {args.only}")

    print(f"{'label':22s} {'samples':>8s} {'time_s':>8s} "
          f"{'sha_head':>16s} {'out_bytes':>10s}")
    print("-" * 72)
    for spec in specs:
        t0 = time.perf_counter()
        bundle = _build(spec, n_samples=args.samples, base_seed=args.seed)
        dt = time.perf_counter() - t0

        path = args.out / f"{spec.label}.pt"
        torch.save(bundle, path)

        # summary hash: hash of hashes of all outputs
        h = hashlib.sha256()
        for s in bundle["samples"]:
            h.update(_tensor_hash(s["output"]).encode())
        digest = h.hexdigest()[:16]
        size = path.stat().st_size
        print(f"{spec.label:22s} {args.samples:8d} {dt:8.2f} "
              f"{digest:>16s} {size:10d}")


if __name__ == "__main__":
    main()
