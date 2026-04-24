"""Shared registry of representative MMA instructions.

Used by both the benchmark harness and the corpus generator so they agree
on what "representative" means. Covers one instruction per NVIDIA arch
family plus notable formats (fp8, fp4, block-scaled).

AMD not yet included — add when we need coverage there.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import _compat  # noqa: F401  (side effect: libm shim)

import torch

from mmasim.simulator.nv_ptx import mma, mma_block_scale, wgmma, tcgen05mma
from mmasim.simulator.amd import mfma


@dataclass(frozen=True)
class InstSpec:
    label: str
    kind: str  # "mma" | "mma_block_scale"
    arch: str
    qualifier: str


REGISTRY: list[InstSpec] = [
    InstSpec("volta-f16",       "mma", "Volta",         "m8n8k4.f32.f16.f16.f32"),
    InstSpec("turing-f16",      "mma", "Turing",        "m16n8k8.f32.f16.f16.f32"),
    InstSpec("ampere-f16",      "mma", "Ampere",        "m16n8k16.f32.f16.f16.f32"),
    InstSpec("ampere-bf16",     "mma", "Ampere",        "m16n8k16.f32.bf16.bf16.f32"),
    InstSpec("ampere-tf32",     "mma", "Ampere",        "m16n8k8.f32.tf32.tf32.f32"),
    InstSpec("ampere-f64",      "mma", "Ampere",        "m8n8k4.f64.f64.f64.f64"),
    InstSpec("ada-fp8-e4m3",    "mma", "Ada Lovelace",  "m16n8k32.f32.e4m3.e4m3.f32"),
    InstSpec("hopper-f64",      "mma", "Hopper",        "m16n8k16.f64.f64.f64.f64"),
    # Note: plain e2m1 mma is in the ISA qualifier list upstream but the
    # sim's mma.__call__ doesn't unpack fp4 — fp4 coverage comes via the
    # block-scaled variants below.
    InstSpec("blackwell-mxfp8", "mma_block_scale", "RTX Blackwell",
             "m16n8k32.block32.f32.e4m3.e4m3.f32.ue8m0"),
    InstSpec("blackwell-mxfp4", "mma_block_scale", "RTX Blackwell",
             "m16n8k64.block32.f32.e2m1.e2m1.f32.ue8m0"),
    # AMD MFMA — phase M3. Scope: the `fma` operation_type paths (f64
    # everywhere, f32 non-xf32 on all CDNA). Pairwise / fused_dot_rd_add
    # paths are deferred.
    InstSpec("amd-cdna2-f64",    "mfma", "CDNA2", "f64_16x16x4f64"),
    InstSpec("amd-cdna3-f64",    "mfma", "CDNA3", "f64_16x16x4_f64"),
    InstSpec("amd-cdna3-f32",    "mfma", "CDNA3", "f32_32x32x2_f32"),
    # Hopper wgmma — phase M4. f32-output subset.
    InstSpec("hopper-wgmma-f16-n64",  "wgmma", "Hopper", "m64n64k16.f32.f16.f16"),
    InstSpec("hopper-wgmma-bf16-n64", "wgmma", "Hopper", "m64n64k16.f32.bf16.bf16"),
    InstSpec("hopper-wgmma-tf32-n64", "wgmma", "Hopper", "m64n64k8.f32.tf32.tf32"),
    # f16-output variants — phase N1 (RNE-FP16)
    InstSpec("ampere-f16-f16",   "mma", "Ampere",       "m16n8k16.f16.f16.f16.f16"),
    InstSpec("turing-f16-f16",   "mma", "Turing",       "m16n8k8.f16.f16.f16.f16"),
    InstSpec("volta-f16-f16",    "mma", "Volta",        "m8n8k4.f16.f16.f16.f16"),
    InstSpec("ada-fp8-e4m3-f16", "mma", "Ada Lovelace", "m16n8k32.f16.e4m3.e4m3.f16"),
]


def make(spec: InstSpec) -> Any:
    if spec.kind == "mma":
        return mma(spec.arch, spec.qualifier)
    if spec.kind == "mma_block_scale":
        return mma_block_scale(spec.arch, spec.qualifier)
    if spec.kind == "mfma":
        return mfma(spec.arch, spec.qualifier)
    if spec.kind == "wgmma":
        return wgmma(spec.arch, spec.qualifier)
    if spec.kind == "tcgen05mma":
        return tcgen05mma(spec.arch, spec.qualifier)
    raise ValueError(f"unknown kind: {spec.kind}")


def _rand_operand(shape: tuple[int, ...], dtype: torch.dtype, gen: torch.Generator) -> torch.Tensor:
    """Random tensor of the given dtype, via f32 cast for float types,
    direct uint8 fill for fp4-packed, small range for fp8/scales."""
    if dtype == torch.uint8:
        # fp4 packed: every 4-bit nibble is a valid encoding.
        return torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8)
    if dtype == torch.float8_e8m0fnu:
        # ue8m0 is an unsigned exponent (value = 2^(byte-127)). Keep scales
        # near 1.0 by sampling bytes in a narrow band around 127.
        raw = torch.randint(120, 135, shape, generator=gen, dtype=torch.uint8)
        return raw.view(torch.float8_e8m0fnu)
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2,
                 torch.float8_e4m3fnuz, torch.float8_e5m2fnuz):
        x = torch.randn(shape, generator=gen, dtype=torch.float32)
        return x.to(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        return torch.randn(shape, generator=gen, dtype=dtype)
    if dtype == torch.float32:
        return torch.randn(shape, generator=gen, dtype=torch.float32)
    if dtype == torch.float64:
        return torch.randn(shape, generator=gen, dtype=torch.float64)
    raise ValueError(f"unhandled dtype: {dtype}")


def gen_inputs(op: Any, seed: int) -> dict[str, torch.Tensor]:
    """Generate random, well-shaped inputs for one call to `op`.

    For `mma`: returns {A, B, C}.
    For `mma_block_scale`: returns {A, B, C, scale_A, scale_B}.
    """
    gen = torch.Generator().manual_seed(seed)
    m, n, k = op.m, op.n, op.k

    if isinstance(op, mma_block_scale):
        packing = op.packing
        block = op.block_size
        A = _rand_operand((m, k // packing), op.a_type, gen)
        B = _rand_operand((k // packing, n), op.b_type, gen)
        C = _rand_operand((m, n), op.c_type, gen)
        sA = _rand_operand((m, k // block), op.s_type, gen)
        sB = _rand_operand((k // block, n), op.s_type, gen)
        return {"A": A, "B": B, "C": C, "scale_A": sA, "scale_B": sB}

    A = _rand_operand((m, k), op.a_type, gen)
    B = _rand_operand((k, n), op.b_type, gen)
    C = _rand_operand((m, n), op.c_type, gen)
    return {"A": A, "B": B, "C": C}


def invoke(op: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(op, mma_block_scale):
        return op(inputs["A"], inputs["B"], inputs["C"],
                  inputs["scale_A"], inputs["scale_B"])
    return op(inputs["A"], inputs["B"], inputs["C"])
