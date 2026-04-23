"""Rust-backed reimplementation of mmasim's mma path.

Phase 2 scope: Ampere f16×f16→f32 workhorse (m16n8k16, nfb=24, split-K),
plus the related Turing/Volta variants that share the f16→f32 code path
at k=4 or k=8 without split-K. Any other (arch, qualifier) raises
NotImplementedError.

The heavy lifting is in the `fastmma_rust` extension module built from
fastmma_rust/. See that crate for the actual kernel.
"""
from __future__ import annotations

import numpy as np
import torch

import fastmma_rust as _rs  # compiled Rust extension


# (arch, qualifier) -> nothing (kernel reads params off the oracle object)
_SUPPORTED: set[tuple[str, str]] = {
    ("Ampere", "m16n8k16.f32.f16.f16.f32"),
    ("Turing", "m16n8k8.f32.f16.f16.f32"),
    ("Volta",  "m8n8k4.f32.f16.f16.f32"),
    ("Ampere", "m16n8k16.f32.bf16.bf16.f32"),
    ("Ampere", "m16n8k8.f32.tf32.tf32.f32"),
}


# Mirrors mmasim/simulator/arithmetic.py::dtype_min_exponent.
_MIN_EXP = {
    torch.float64:       -1022,
    torch.float32:       -126,
    torch.float16:       -14,
    torch.bfloat16:      -126,
    torch.float8_e4m3fn: -6,
    torch.float8_e5m2:   -14,
}


class mma:
    def __init__(self, arch: str, qualifier: str):
        from mmasim.simulator.nv_ptx import mma as _oracle_mma
        _ref = _oracle_mma(arch, qualifier)
        self.arch = arch
        self.qualifier = qualifier
        self.m, self.n, self.k = _ref.m, _ref.n, _ref.k
        self.a_type = _ref.a_type
        self.b_type = _ref.b_type
        self.c_type = _ref.c_type
        self.d_type = _ref.d_type
        self.is_split_k = _ref.is_split_k
        self.nfb = _ref.n_accum_fractional_bits
        self.output_type = _ref.output_type

        if (arch, qualifier) not in _SUPPORTED:
            raise NotImplementedError(
                f"rust_ref: (arch, qualifier) not yet supported: "
                f"({arch!r}, {qualifier!r})"
            )

        self._out_mant_bits = 13 if self.output_type == "f32_e8m13" else 23
        self._a_min = _MIN_EXP[self.a_type]
        self._b_min = _MIN_EXP[self.b_type]
        self._c_min = _MIN_EXP[self.c_type]

    @staticmethod
    def _to_f32_contig(t: torch.Tensor) -> np.ndarray:
        # bf16/f16 → f32 is lossless; tf32 truncation is applied in the
        # kernel wrapper below for Ampere tf32 specifically.
        a = t.detach().cpu().to(torch.float32).contiguous().numpy()
        return np.ascontiguousarray(a, dtype=np.float32)

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        A_f32 = self._to_f32_contig(A)
        B_f32 = self._to_f32_contig(B)
        C_f32 = self._to_f32_contig(C)

        if self.a_type is torch.float32:  # tf32
            # Mirror the `>> 13 << 13` trick from arithmetic.py.
            raw_a = A_f32.view(np.int32)
            raw_b = B_f32.view(np.int32)
            A_f32 = ((raw_a >> 13) << 13).view(np.float32)
            B_f32 = ((raw_b >> 13) << 13).view(np.float32)
            A_f32 = np.ascontiguousarray(A_f32)
            B_f32 = np.ascontiguousarray(B_f32)

        out = _rs.mma_f32_out(
            A_f32, B_f32, C_f32,
            self.nfb,
            self._a_min, self._b_min, self._c_min,
            self._out_mant_bits,
            self.is_split_k,
        )
        return torch.from_numpy(out)

    def call_batched(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Batched variant: A, B, C are 3-D with a leading batch dim.

        Shapes: A=(batch, m, k), B=(batch, k, n), C=(batch, m, n).
        Returns: (batch, m, n). Semantics are identical to calling `self`
        once per batch index; this just amortizes the PyO3 boundary.
        """
        assert A.shape[1:] == (self.m, self.k)
        assert B.shape[1:] == (self.k, self.n)
        assert C.shape[1:] == (self.m, self.n)
        A_f32 = A.detach().cpu().to(torch.float32).contiguous().numpy()
        B_f32 = B.detach().cpu().to(torch.float32).contiguous().numpy()
        C_f32 = C.detach().cpu().to(torch.float32).contiguous().numpy()
        A_f32 = np.ascontiguousarray(A_f32, dtype=np.float32)
        B_f32 = np.ascontiguousarray(B_f32, dtype=np.float32)
        C_f32 = np.ascontiguousarray(C_f32, dtype=np.float32)

        if self.a_type is torch.float32:  # tf32
            raw_a = A_f32.view(np.int32)
            raw_b = B_f32.view(np.int32)
            A_f32 = ((raw_a >> 13) << 13).view(np.float32)
            B_f32 = ((raw_b >> 13) << 13).view(np.float32)
            A_f32 = np.ascontiguousarray(A_f32)
            B_f32 = np.ascontiguousarray(B_f32)

        out = _rs.mma_f32_out_batched(
            A_f32, B_f32, C_f32,
            self.nfb,
            self._a_min, self._b_min, self._c_min,
            self._out_mant_bits,
            self.is_split_k,
        )
        return torch.from_numpy(out)
