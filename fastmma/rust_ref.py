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

    def _prep_batched(self, A, B, C):
        A_f32 = np.ascontiguousarray(
            A.detach().cpu().to(torch.float32).contiguous().numpy(),
            dtype=np.float32,
        )
        B_f32 = np.ascontiguousarray(
            B.detach().cpu().to(torch.float32).contiguous().numpy(),
            dtype=np.float32,
        )
        C_f32 = np.ascontiguousarray(
            C.detach().cpu().to(torch.float32).contiguous().numpy(),
            dtype=np.float32,
        )
        if self.a_type is torch.float32:  # tf32
            A_f32 = np.ascontiguousarray(
                ((A_f32.view(np.int32) >> 13) << 13).view(np.float32)
            )
            B_f32 = np.ascontiguousarray(
                ((B_f32.view(np.int32) >> 13) << 13).view(np.float32)
            )
        return A_f32, B_f32, C_f32

    def call_batched(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Scalar batched call. (B, M, K) / (B, K, N) / (B, M, N) -> (B, M, N)."""
        assert A.shape[1:] == (self.m, self.k)
        assert B.shape[1:] == (self.k, self.n)
        assert C.shape[1:] == (self.m, self.n)
        A_f32, B_f32, C_f32 = self._prep_batched(A, B, C)
        out = _rs.mma_f32_out_batched(
            A_f32, B_f32, C_f32,
            self.nfb, self._a_min, self._b_min, self._c_min,
            self._out_mant_bits, self.is_split_k,
        )
        return torch.from_numpy(out)

    def call_batched_simd(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """SIMD batched call (aarch64 NEON; scalar fallback elsewhere).

        Same semantics as call_batched — bit-exact. Vectorized 2-wide
        on the j-axis inside the kernel.
        """
        assert A.shape[1:] == (self.m, self.k)
        assert B.shape[1:] == (self.k, self.n)
        assert C.shape[1:] == (self.m, self.n)
        A_f32, B_f32, C_f32 = self._prep_batched(A, B, C)
        out = _rs.mma_f32_out_batched_simd(
            A_f32, B_f32, C_f32,
            self.nfb, self._a_min, self._b_min, self._c_min,
            self._out_mant_bits, self.is_split_k,
        )
        return torch.from_numpy(out)

    def call_batched_rayon(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Scalar kernel, rayon-parallelized across the batch dim.

        Same scalar inner kernel as call_batched (no SIMD). This is
        the pre-SIMD baseline with threading layered on top, so the
        speedup vs call_batched is attributable to parallelism alone.
        """
        assert A.shape[1:] == (self.m, self.k)
        assert B.shape[1:] == (self.k, self.n)
        assert C.shape[1:] == (self.m, self.n)
        A_f32, B_f32, C_f32 = self._prep_batched(A, B, C)
        out = _rs.mma_f32_out_batched_rayon(
            A_f32, B_f32, C_f32,
            self.nfb, self._a_min, self._b_min, self._c_min,
            self._out_mant_bits, self.is_split_k,
        )
        return torch.from_numpy(out)

    def call_batched_simd_rayon(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """NEON SIMD inner kernel, rayon over the batch. Combines 2.3+2.4."""
        assert A.shape[1:] == (self.m, self.k)
        assert B.shape[1:] == (self.k, self.n)
        assert C.shape[1:] == (self.m, self.n)
        A_f32, B_f32, C_f32 = self._prep_batched(A, B, C)
        out = _rs.mma_f32_out_batched_simd_rayon(
            A_f32, B_f32, C_f32,
            self.nfb, self._a_min, self._b_min, self._c_min,
            self._out_mant_bits, self.is_split_k,
        )
        return torch.from_numpy(out)

    # Phase 3.1: per-instruction specialized PyO3 entries. Dispatched by
    # (arch, qualifier); each kernel has its shape and parameters baked
    # in at compile time.
    _SPECIALIZED_FNS = {
        ("Ampere", "m16n8k16.f32.f16.f16.f32"):   "mma_spec_ampere_f16",
        ("Ampere", "m16n8k16.f32.bf16.bf16.f32"): "mma_spec_ampere_bf16",
        ("Ampere", "m16n8k8.f32.tf32.tf32.f32"):  "mma_spec_ampere_tf32",
        ("Turing", "m16n8k8.f32.f16.f16.f32"):    "mma_spec_turing_f16",
        ("Volta",  "m8n8k4.f32.f16.f16.f32"):     "mma_spec_volta_f16",
    }

    def call_batched_metal(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Metal GPU kernel. Currently only Ampere m16n8k16.f32.f16.f16.f32."""
        if (self.arch, self.qualifier) != ("Ampere", "m16n8k16.f32.f16.f16.f32"):
            raise NotImplementedError(
                "Metal backend only has ampere-f16 for now"
            )
        if not hasattr(_rs, "mma_metal_ampere_f16"):
            raise NotImplementedError("Metal backend not built (not macOS?)")
        assert A.shape[1:] == (16, 16)
        assert B.shape[1:] == (16, 8)
        assert C.shape[1:] == (16, 8)
        A_f32, B_f32, C_f32 = self._prep_batched(A, B, C)
        out = _rs.mma_metal_ampere_f16(A_f32, B_f32, C_f32)
        return torch.from_numpy(out)

    def call_batched_specialized(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Per-(arch, qualifier) specialized kernel. Dispatches at the
        Python level; each target has a dedicated monomorphized PyO3
        entry with all tile parameters as compile-time constants."""
        fn_name = self._SPECIALIZED_FNS.get((self.arch, self.qualifier))
        if fn_name is None:
            raise NotImplementedError(
                f"specialized kernel not registered for "
                f"({self.arch!r}, {self.qualifier!r})"
            )
        assert A.shape[1:] == (self.m, self.k)
        assert B.shape[1:] == (self.k, self.n)
        assert C.shape[1:] == (self.m, self.n)
        A_f32, B_f32, C_f32 = self._prep_batched(A, B, C)
        out = getattr(_rs, fn_name)(A_f32, B_f32, C_f32)
        return torch.from_numpy(out)
