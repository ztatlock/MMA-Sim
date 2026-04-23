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
    print(f"{'label':18s} {'single us':>10s} {'per-tile scalar':>17s} "
          f"{'per-tile simd':>15s} {'simd/scalar':>12s}")
    print("-" * 80)

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

        stacked = _stack_samples(samples, args.tiles)
        n_check = min(args.tiles, len(samples))

        def _time(call):
            call(stacked["A"], stacked["B"], stacked["C"])  # warmup
            ts = []
            for _ in range(args.reps):
                t0 = time.perf_counter_ns()
                out = call(stacked["A"], stacked["B"], stacked["C"])
                ts.append(time.perf_counter_ns() - t0)
            return statistics.median(ts) / 1000, out

        batched_scalar_us, D_scalar = _time(op.call_batched)
        batched_simd_us, D_simd = _time(op.call_batched_simd)

        per_tile_scalar = batched_scalar_us / args.tiles
        per_tile_simd = batched_simd_us / args.tiles

        ok_scalar = all(_bits_equal(D_scalar[i], samples[i]["output"]) for i in range(n_check))
        ok_simd = all(_bits_equal(D_simd[i], samples[i]["output"]) for i in range(n_check))
        status = ("OK" if ok_scalar else "FAIL-scalar") if ok_scalar and ok_simd \
            else "FAIL-simd" if ok_scalar else "FAIL-both"

        simd_ratio = per_tile_scalar / per_tile_simd if per_tile_simd > 0 else float("inf")
        print(f"{label:18s} {single_us:10.2f} {per_tile_scalar:17.3f} "
              f"{per_tile_simd:15.3f} {simd_ratio:10.2f}x  {status}")
        if (not ok_scalar or not ok_simd) and args.verbose:
            for i in range(n_check):
                if not _bits_equal(D_simd[i], samples[i]["output"]):
                    print(f"    simd diff at sample {i}")
                    break


if __name__ == "__main__":
    main()
