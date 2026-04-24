"""Edge-case corpus generator.

Complements `corpus.py` (random torch.randn samples) with curated
inputs designed to exercise the oracle's special-value and boundary
paths: zero operands, NaN/Inf propagation, subnormal values, large
magnitudes, catastrophic cancellation. Oracle output is captured the
same way as the regular corpus.

Output goes alongside the regular fixtures with an `edge_` prefix,
so `bench/validate.py` and `bench/batched.py` pick them up
automatically.

Usage:
    python -m bench.corpus_edge
    python -m bench.corpus_edge --only ampere-f16
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import time

import torch

from . import instructions, _compat  # noqa: F401

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


def _tensor_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()[:16]


def _zeros_like(t: torch.Tensor) -> torch.Tensor:
    """Zero-valued tensor in t's dtype/shape. Works for fp8/fp4 too
    (uint8 packed zero-byte is a valid all-zero fp4/fp8 pattern)."""
    return torch.zeros_like(t)


def _with_nan(t: torch.Tensor, idx: tuple[int, ...]) -> torch.Tensor:
    """Return a copy of t with one element set to NaN. Only meaningful
    for float dtypes that can represent NaN (f64/f32/f16/bf16, some fp8 variants)."""
    out = t.clone()
    try:
        if out.dtype in (torch.float64, torch.float32):
            out[idx] = float("nan")
        elif out.dtype == torch.float16:
            out[idx] = torch.tensor(float("nan"), dtype=torch.float16)
        elif out.dtype == torch.bfloat16:
            out[idx] = torch.tensor(float("nan"), dtype=torch.bfloat16)
        elif out.dtype == torch.float8_e5m2:
            # e5m2 has a NaN encoding. Use a specific bit pattern.
            nan_e5m2 = torch.tensor([0x7F], dtype=torch.uint8).view(torch.float8_e5m2)
            out.view(torch.uint8)[idx] = 0x7F
        # e4m3fn: no NaN encoding; skip
        # uint8 (fp4 packed): no NaN
    except Exception:
        pass
    return out


def _with_inf(t: torch.Tensor, idx: tuple[int, ...], sign: int = 1) -> torch.Tensor:
    out = t.clone()
    try:
        if out.dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
            out[idx] = float("inf") * sign
        elif out.dtype == torch.float8_e5m2:
            # e5m2 Inf encoding: 0x7C (+inf), 0xFC (-inf)
            raw = 0x7C if sign > 0 else 0xFC
            out.view(torch.uint8)[idx] = raw
    except Exception:
        pass
    return out


def _scale_inputs(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, factor: float):
    """Scale f32-typed inputs by `factor`. For narrow dtypes, cast up
    to f32, scale, cast back — clamping happens naturally."""
    def _scale(x):
        if x.dtype in (torch.float32, torch.float64):
            return x * factor
        if x.dtype in (torch.float16, torch.bfloat16):
            return (x.to(torch.float32) * factor).to(x.dtype)
        # fp8, packed fp4: unchanged (values are format-bounded anyway)
        return x
    return _scale(A), _scale(B), _scale(C)


def _build_edge_samples(op, spec_kind: str, base_seed: int, n_random: int = 2) -> list:
    """Curate a small edge-case sample set per instruction. Oracle runs
    each to capture expected output."""
    samples = []
    m, n, k = op.m, op.n, op.k

    # 1-2) baseline random samples (as sanity / differentiation from fp8/etc)
    for r in range(n_random):
        inputs = instructions.gen_inputs(op, seed=base_seed + r)
        out = instructions.invoke(op, inputs)
        samples.append({"inputs": inputs, "output": out.clone()})

    base = instructions.gen_inputs(op, seed=base_seed + 100)

    def _record(inputs):
        try:
            out = instructions.invoke(op, inputs)
        except Exception as e:
            return None  # oracle refused the input; skip
        return {"inputs": inputs, "output": out.clone()}

    # 3) A = 0
    inp = dict(base)
    inp["A"] = _zeros_like(base["A"])
    s = _record(inp)
    if s is not None: samples.append(s)

    # 4) B = 0
    inp = dict(base)
    inp["B"] = _zeros_like(base["B"])
    s = _record(inp)
    if s is not None: samples.append(s)

    # 5) C = 0
    inp = dict(base)
    inp["C"] = _zeros_like(base["C"])
    s = _record(inp)
    if s is not None: samples.append(s)

    float_c = (torch.float64, torch.float32, torch.float16, torch.bfloat16)

    # 6) C has NaN in one element
    if base["C"].dtype in float_c:
        inp = dict(base)
        inp["C"] = _with_nan(base["C"], (0, 0))
        s = _record(inp)
        if s is not None: samples.append(s)

    # 7) C has +Inf in one element
    if base["C"].dtype in float_c:
        inp = dict(base)
        inp["C"] = _with_inf(base["C"], (0, 0), sign=+1)
        s = _record(inp)
        if s is not None: samples.append(s)

    # 8) C has -Inf in one element
    if base["C"].dtype in float_c:
        inp = dict(base)
        inp["C"] = _with_inf(base["C"], (0, 0), sign=-1)
        s = _record(inp)
        if s is not None: samples.append(s)

    # 8b) For f64 A: also test NaN/Inf propagation in A itself (f64 is
    # common for high-precision use cases and the serial-FMA kernel has
    # different NaN/Inf propagation than the integer fused-sum path).
    if base["A"].dtype == torch.float64:
        inp = dict(base)
        inp["A"] = _with_nan(base["A"], (0, 0))
        s = _record(inp)
        if s is not None: samples.append(s)
        inp = dict(base)
        inp["A"] = _with_inf(base["A"], (0, 0), sign=+1)
        s = _record(inp)
        if s is not None: samples.append(s)

    # 9) Very small inputs (× 1e-20 in f32, clamped for narrow dtypes)
    inp = dict(base)
    if "scale_A" in base:
        # block-scale: leave scales alone, shrink A/B/C
        inp["A"], inp["B"], inp["C"] = _scale_inputs(
            base["A"], base["B"], base["C"], 1e-20
        )
    else:
        inp["A"], inp["B"], inp["C"] = _scale_inputs(
            base["A"], base["B"], base["C"], 1e-20
        )
    s = _record(inp)
    if s is not None: samples.append(s)

    # 10) Very large inputs (× 1e20)
    inp = dict(base)
    inp["A"], inp["B"], inp["C"] = _scale_inputs(
        base["A"], base["B"], base["C"], 1e20
    )
    s = _record(inp)
    if s is not None: samples.append(s)

    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--seed", type=int, default=9000)
    args = ap.parse_args()

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'label':22s} {'samples':>8s} {'time_s':>8s} {'sha_head':>16s}")
    print("-" * 62)

    for spec in instructions.REGISTRY:
        if args.only and spec.label != args.only:
            continue
        t0 = time.perf_counter()
        op = instructions.make(spec)
        samples = _build_edge_samples(op, spec.kind, args.seed)
        dt = time.perf_counter() - t0

        bundle = {
            "spec": {
                "label": spec.label,
                "arch": spec.arch,
                "qualifier": spec.qualifier,
                "kind": spec.kind,
            },
            "samples": samples,
        }
        path = FIXTURES_DIR / f"edge_{spec.label}.pt"
        torch.save(bundle, path)

        h = hashlib.sha256()
        for s in samples:
            h.update(_tensor_hash(s["output"]).encode())
        digest = h.hexdigest()[:16]
        print(f"{spec.label:22s} {len(samples):8d} {dt:8.2f} {digest:>16s}")


if __name__ == "__main__":
    main()
