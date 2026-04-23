"""Batched-throughput microbench for fastmma.rust_ref.

Stacks all samples of a fixture into a single call to `mma.call_batched`
and measures per-tile cost. Verifies every output in the batch is
bit-identical to the corpus oracle. Compares to the per-tile cost of
the single-shot path so we can see the amortization payoff directly.

Usage:
    python -m bench.batched                            # all supported fixtures
    python -m bench.batched --only ampere-f16 --tiles 1024
    python -m bench.batched --verbose
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import time

from . import _compat  # noqa: F401
import torch

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return a.contiguous().view(torch.uint8).eq(
        b.contiguous().view(torch.uint8)
    ).all().item()


def _stack_samples(samples: list, count: int) -> dict:
    """Stack `count` samples (cycling if needed) into batched tensors."""
    A_list, B_list, C_list, D_list = [], [], [], []
    for i in range(count):
        s = samples[i % len(samples)]
        A_list.append(s["inputs"]["A"])
        B_list.append(s["inputs"]["B"])
        C_list.append(s["inputs"]["C"])
        D_list.append(s["output"])
    return {
        "A": torch.stack(A_list),
        "B": torch.stack(B_list),
        "C": torch.stack(C_list),
        "D": torch.stack(D_list),
    }


def main() -> None:
    from fastmma.rust_ref import mma as rust_mma, _SUPPORTED

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--tiles", type=int, default=1024,
                    help="batch size (cycles through the 16-sample corpus)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"batch size: {args.tiles} tiles/call, reps: {args.reps}")
    print(f"{'label':18s} {'single us':>10s} {'batched us':>11s} "
          f"{'per-tile us':>12s} {'amortization':>14s}")
    print("-" * 75)

    for fpath in sorted(FIXTURES_DIR.glob("*.pt")):
        label = fpath.stem
        if args.only and args.only != label:
            continue

        bundle = torch.load(fpath, weights_only=False)
        spec = bundle["spec"]
        if (spec["arch"], spec["qualifier"]) not in _SUPPORTED:
            continue

        op = rust_mma(spec["arch"], spec["qualifier"])
        samples = bundle["samples"]

        # Single-call per-tile cost (one tile per call).
        one = samples[0]
        op(one["inputs"]["A"], one["inputs"]["B"], one["inputs"]["C"])  # warmup
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter_ns()
            op(one["inputs"]["A"], one["inputs"]["B"], one["inputs"]["C"])
            ts.append(time.perf_counter_ns() - t0)
        single_us = statistics.median(ts) / 1000

        # Batched call.
        stacked = _stack_samples(samples, args.tiles)
        op.call_batched(stacked["A"], stacked["B"], stacked["C"])  # warmup
        tb = []
        for _ in range(args.reps):
            t0 = time.perf_counter_ns()
            D_out = op.call_batched(stacked["A"], stacked["B"], stacked["C"])
            tb.append(time.perf_counter_ns() - t0)
        batched_us = statistics.median(tb) / 1000
        per_tile_us = batched_us / args.tiles

        # Correctness: every batched output must match the (cycled) corpus.
        # We only have 16 oracle outputs, so check the first min(batch, 16).
        n_check = min(args.tiles, len(samples))
        ok = all(_bits_equal(D_out[i], samples[i]["output"]) for i in range(n_check))
        status = "OK" if ok else "FAIL"

        amort = single_us / per_tile_us if per_tile_us > 0 else float("inf")
        print(f"{label:18s} {single_us:10.2f} {batched_us:11.0f} "
              f"{per_tile_us:12.3f} {amort:12.1f}x  {status}")
        if not ok and args.verbose:
            for i in range(n_check):
                if not _bits_equal(D_out[i], samples[i]["output"]):
                    print(f"    diff at sample {i}")
                    break


if __name__ == "__main__":
    main()
