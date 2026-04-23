"""Benchmark harness: time one call of each registered MMA instruction.

Usage:
    python -m bench.bench                 # all instructions, default reps
    python -m bench.bench --reps 5        # more reps for tighter numbers
    python -m bench.bench --only ampere-f16

Prints a table to stdout. For logging to a file, redirect.
"""
from __future__ import annotations

import argparse
import gc
import statistics
import time

from . import instructions


def _time_one(spec: instructions.InstSpec, reps: int, seed: int) -> dict:
    op = instructions.make(spec)
    inputs = instructions.gen_inputs(op, seed=seed)
    # warmup
    instructions.invoke(op, inputs)
    samples = []
    for _ in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        instructions.invoke(op, inputs)
        samples.append(time.perf_counter() - t0)
    return {
        "label": spec.label,
        "qualifier": spec.qualifier,
        "arch": spec.arch,
        "mnk": (op.m, op.n, op.k),
        "n_outputs": op.m * op.n,
        "median_ms": statistics.median(samples) * 1000,
        "min_ms": min(samples) * 1000,
        "max_ms": max(samples) * 1000,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3,
                    help="measured reps per instruction (after warmup)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", type=str, default=None,
                    help="run only this label (e.g. ampere-f16)")
    args = ap.parse_args()

    specs = instructions.REGISTRY
    if args.only:
        specs = [s for s in specs if s.label == args.only]
        if not specs:
            raise SystemExit(f"no instruction with label: {args.only}")

    print(f"{'label':22s} {'arch':15s} {'mnk':14s} "
          f"{'median_ms':>10s} {'min_ms':>10s} {'us/output':>10s}")
    print("-" * 90)
    for spec in specs:
        r = _time_one(spec, reps=args.reps, seed=args.seed)
        mnk = f"{r['mnk'][0]}x{r['mnk'][1]}x{r['mnk'][2]}"
        us_per_out = r["median_ms"] * 1000 / r["n_outputs"]
        print(f"{r['label']:22s} {r['arch']:15s} {mnk:14s} "
              f"{r['median_ms']:10.2f} {r['min_ms']:10.2f} {us_per_out:10.1f}")


if __name__ == "__main__":
    main()
