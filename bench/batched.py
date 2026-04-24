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

    import os
    print(f"batch size: {args.tiles} tiles/call, reps: {args.reps}, "
          f"cores: {os.cpu_count()}")
    print(f"{'label':16s} {'scalar':>8s} {'simd':>8s} {'rayon':>8s} "
          f"{'s+r':>8s} {'spec':>8s} {'s+r/sc':>8s} {'spec/s+r':>9s}")
    print("-" * 82)

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

        us_sc, D_sc = _time(op.call_batched)
        us_v,  D_v  = _time(op.call_batched_simd)
        us_r,  D_r  = _time(op.call_batched_rayon)
        us_vr, D_vr = _time(op.call_batched_simd_rayon)

        pt_sc = us_sc / args.tiles
        pt_v  = us_v  / args.tiles
        pt_r  = us_r  / args.tiles
        pt_vr = us_vr / args.tiles

        # Specialized kernel is ampere-f16 only (Phase 3.0 ceiling probe).
        has_spec = False
        pt_sp = 0.0
        ok_sp = True
        try:
            us_sp, D_sp = _time(op.call_batched_specialized)
            pt_sp = us_sp / args.tiles
            ok_sp = all(_bits_equal(D_sp[i], samples[i]["output"]) for i in range(n_check))
            has_spec = True
        except NotImplementedError:
            pass

        ok_sc = all(_bits_equal(D_sc[i], samples[i]["output"]) for i in range(n_check))
        ok_v  = all(_bits_equal(D_v[i],  samples[i]["output"]) for i in range(n_check))
        ok_r  = all(_bits_equal(D_r[i],  samples[i]["output"]) for i in range(n_check))
        ok_vr = all(_bits_equal(D_vr[i], samples[i]["output"]) for i in range(n_check))
        all_ok = ok_sc and ok_v and ok_r and ok_vr and ok_sp
        status = "OK" if all_ok else \
                 "FAIL:" + "".join([""  if ok_sc else "s",
                                    ""  if ok_v  else "v",
                                    ""  if ok_r  else "r",
                                    ""  if ok_vr else "V",
                                    ""  if ok_sp else "S"])

        spec_str  = f"{pt_sp:8.3f}" if has_spec else f"{'—':>8s}"
        ratio_str = f"{pt_vr/pt_sp:8.2f}x" if has_spec else f"{'—':>9s}"
        print(f"{label:16s} {pt_sc:8.3f} {pt_v:8.3f} {pt_r:8.3f} {pt_vr:8.3f} "
              f"{spec_str} {pt_sc/pt_vr:7.2f}x {ratio_str}  {status}")
        if status != "OK" and args.verbose:
            for name, D in [("simd", D_v), ("rayon", D_r), ("simd+rayon", D_vr)]:
                for i in range(n_check):
                    if not _bits_equal(D[i], samples[i]["output"]):
                        print(f"    {name} diff at sample {i}")
                        break


if __name__ == "__main__":
    main()
