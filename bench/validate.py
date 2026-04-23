"""Corpus-based bit-exact validator.

Loads fixtures from bench/fixtures/, runs each sample through a named
implementation (default: fastmma.numpy_ref), and compares output bits
against the oracle output frozen in the fixture. Also times the impl
for a quick speedup readout.

Usage:
    python -m bench.validate
    python -m bench.validate --impl fastmma.numpy_ref
    python -m bench.validate --only ampere-f16
"""
from __future__ import annotations

import argparse
import importlib
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


def _first_diff(a: torch.Tensor, b: torch.Tensor):
    af = a.flatten()
    bf = b.flatten()
    for i in range(af.numel()):
        if af[i].ne(bf[i]).item():
            return i, af[i].item(), bf[i].item()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="fastmma.numpy_ref",
                    help="module providing mma(arch, qualifier) [and later mma_block_scale]")
    ap.add_argument("--only", default=None, help="validate only this label")
    ap.add_argument("--reps", type=int, default=3,
                    help="timed reps per instruction (after warmup)")
    ap.add_argument("--verbose", action="store_true",
                    help="print first differing element on mismatch")
    args = ap.parse_args()

    impl_mod = importlib.import_module(args.impl)

    print(f"impl: {args.impl}")
    print(f"{'label':22s} {'status':>6s}  {'samples':>9s}  "
          f"{'median_ms':>10s}  {'speedup':>8s}")
    print("-" * 70)

    n_pass = n_fail = n_skip = 0
    for fpath in sorted(FIXTURES_DIR.glob("*.pt")):
        label = fpath.stem
        if args.only and args.only != label:
            continue
        bundle = torch.load(fpath, weights_only=False)
        spec = bundle["spec"]

        # Build the impl op.
        ctor_name = {"mma": "mma", "mma_block_scale": "mma_block_scale"}[spec["kind"]]
        ctor = getattr(impl_mod, ctor_name, None)
        if ctor is None:
            print(f"{label:22s} {'SKIP':>6s}  (impl has no {ctor_name})")
            n_skip += 1
            continue
        try:
            op = ctor(spec["arch"], spec["qualifier"])
        except NotImplementedError as e:
            print(f"{label:22s} {'SKIP':>6s}  ({e})")
            n_skip += 1
            continue

        # Correctness: every sample bit-identical.
        samples = bundle["samples"]
        ok = 0
        first_bad = None
        for idx, s in enumerate(samples):
            try:
                out = op(**s["inputs"]) if spec["kind"] == "mma_block_scale" \
                      else op(s["inputs"]["A"], s["inputs"]["B"], s["inputs"]["C"])
            except Exception as e:
                first_bad = (idx, f"raised {type(e).__name__}: {e}")
                break
            if _bits_equal(s["output"], out):
                ok += 1
            else:
                first_bad = (idx, _first_diff(s["output"], out))
                break

        # Timing: use first sample, warmup + reps.
        inputs = samples[0]["inputs"]

        def _run():
            if spec["kind"] == "mma_block_scale":
                op(**inputs)
            else:
                op(inputs["A"], inputs["B"], inputs["C"])

        _run()  # warmup
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            _run()
            times.append(time.perf_counter() - t0)
        median_ms = statistics.median(times) * 1000

        # We need the oracle's timing for speedup. Run it once more here.
        from mmasim.simulator.nv_ptx import mma as _oracle_mma
        from mmasim.simulator.nv_ptx import mma_block_scale as _oracle_mbs
        oracle_ctor = _oracle_mma if spec["kind"] == "mma" else _oracle_mbs
        oracle = oracle_ctor(spec["arch"], spec["qualifier"])

        def _run_oracle():
            if spec["kind"] == "mma_block_scale":
                oracle(**inputs)
            else:
                oracle(inputs["A"], inputs["B"], inputs["C"])

        _run_oracle()
        o_times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            _run_oracle()
            o_times.append(time.perf_counter() - t0)
        o_median_ms = statistics.median(o_times) * 1000
        speedup = o_median_ms / median_ms if median_ms > 0 else float("inf")

        if first_bad is None:
            n_pass += 1
            status = "OK"
        else:
            n_fail += 1
            status = "FAIL"

        print(f"{label:22s} {status:>6s}  {ok}/{len(samples):>7} "
              f"  {median_ms:10.3f}  {speedup:7.0f}x")
        if first_bad and args.verbose:
            print(f"    first diff: sample {first_bad[0]}: {first_bad[1]}")

    print("-" * 70)
    print(f"pass={n_pass} fail={n_fail} skip={n_skip}")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
