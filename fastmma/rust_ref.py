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


# (arch, qualifier) -> supported under the generic path. The specialized
# path is a strict subset keyed by _SPECIALIZED_FNS below.
_SUPPORTED: set[tuple[str, str]] = {
    # Volta / Turing / Ampere core dtypes (specialized + generic)
    ("Ampere", "m16n8k16.f32.f16.f16.f32"),
    ("Turing", "m16n8k8.f32.f16.f16.f32"),
    ("Volta",  "m8n8k4.f32.f16.f16.f32"),
    ("Ampere", "m16n8k16.f32.bf16.bf16.f32"),
    ("Ampere", "m16n8k8.f32.tf32.tf32.f32"),
    # Ada Lovelace fp8, f32 output (nfb=13, f32_e8m13). Generic path only.
    ("Ada Lovelace", "m16n8k32.f32.e5m2.e5m2.f32"),
    ("Ada Lovelace", "m16n8k32.f32.e5m2.e4m3.f32"),
    ("Ada Lovelace", "m16n8k32.f32.e4m3.e5m2.f32"),
    ("Ada Lovelace", "m16n8k32.f32.e4m3.e4m3.f32"),
    ("Ada Lovelace", "m16n8k16.f32.e5m2.e5m2.f32"),
    ("Ada Lovelace", "m16n8k16.f32.e5m2.e4m3.f32"),
    ("Ada Lovelace", "m16n8k16.f32.e4m3.e5m2.f32"),
    ("Ada Lovelace", "m16n8k16.f32.e4m3.e4m3.f32"),
    # ── f16-output variants (RNE to 10 mantissa bits) ── Phase N1 ──
    # Volta f16/f16
    ("Volta", "m8n8k4.f16.f16.f16.f16"),
    ("Volta", "m8n8k4.f32.f16.f16.f16"),
    # Turing f16/f16
    ("Turing", "m16n8k8.f16.f16.f16.f16"),
    # Ampere f16/f16
    ("Ampere", "m16n8k16.f16.f16.f16.f16"),
    # Ada Lovelace fp8 → f16
    ("Ada Lovelace", "m16n8k32.f16.e5m2.e5m2.f16"),
    ("Ada Lovelace", "m16n8k32.f16.e5m2.e4m3.f16"),
    ("Ada Lovelace", "m16n8k32.f16.e4m3.e5m2.f16"),
    ("Ada Lovelace", "m16n8k32.f16.e4m3.e4m3.f16"),
    ("Ada Lovelace", "m16n8k16.f16.e5m2.e5m2.f16"),
    ("Ada Lovelace", "m16n8k16.f16.e5m2.e4m3.f16"),
    ("Ada Lovelace", "m16n8k16.f16.e4m3.e5m2.f16"),
    ("Ada Lovelace", "m16n8k16.f16.e4m3.e4m3.f16"),
    # F64 — uses dedicated `mma_f64_batched_rayon` Rust entry (serial FMA).
    ("Ampere", "m8n8k4.f64.f64.f64.f64"),
    ("Hopper", "m16n8k16.f64.f64.f64.f64"),
    ("Hopper", "m16n8k8.f64.f64.f64.f64"),
    ("Hopper", "m16n8k4.f64.f64.f64.f64"),
}


# Mirrors mmasim/simulator/arithmetic.py::dtype_min_exponent.
_MIN_EXP = {
    torch.float64:         -1022,
    torch.float32:         -126,
    torch.float16:         -14,
    torch.bfloat16:        -126,
    torch.float8_e4m3fn:   -6,
    torch.float8_e5m2:     -14,
    torch.float8_e4m3fnuz: -7,
    torch.float8_e5m2fnuz: -15,
}


# AMD MFMA operation_type → supported kernel mapping.
# Phase M3 added `fma` path; Phase N2 adds `pairwise` (CDNA1/2 f16/bf16).
# `fused_dot_rd_add` (CDNA3 f16/bf16/fp8, xf32) still deferred (N3).
_SUPPORTED_MFMA: set[tuple[str, str]] = {
    # CDNA2 f64 (operation_type = fma)
    ("CDNA2", "f64_16x16x4f64"),
    ("CDNA2", "f64_4x4x4f64"),
    # CDNA3 f64 (operation_type = fma)
    ("CDNA3", "f64_16x16x4_f64"),
    ("CDNA3", "f64_4x4x4_4b_f64"),
    # CDNA3 f32 non-xf32 (operation_type = fma)
    ("CDNA3", "f32_32x32x1_2b_f32"),
    ("CDNA3", "f32_16x16x1_4b_f32"),
    ("CDNA3", "f32_4x4x1_16b_f32"),
    ("CDNA3", "f32_32x32x2_f32"),
    ("CDNA3", "f32_16x16x4_f32"),
    # CDNA1/2 f32 (reused from cdna1 list)
    ("CDNA1", "f32_32x32x2f32"),
    ("CDNA1", "f32_32x32x1f32"),
    ("CDNA1", "f32_16x16x4f32"),
    ("CDNA1", "f32_16x16x1f32"),
    ("CDNA1", "f32_4x4x1f32"),
    ("CDNA2", "f32_32x32x2f32"),
    ("CDNA2", "f32_32x32x1f32"),
    ("CDNA2", "f32_16x16x4f32"),
    ("CDNA2", "f32_16x16x1f32"),
    ("CDNA2", "f32_4x4x1f32"),
    # ── Phase N2: pairwise path (CDNA1/2 f16/bf16) ──
    # CDNA1 f16 (group_size=4, no flush)
    ("CDNA1", "f32_32x32x8f16"),
    ("CDNA1", "f32_32x32x4f16"),
    ("CDNA1", "f32_16x16x16f16"),
    ("CDNA1", "f32_16x16x4f16"),
    ("CDNA1", "f32_4x4x4f16"),
    # CDNA1 bf16 (group_size=2, no flush)
    ("CDNA1", "f32_32x32x4bf16"),
    ("CDNA1", "f32_32x32x2bf16"),
    ("CDNA1", "f32_16x16x8bf16"),
    ("CDNA1", "f32_16x16x2bf16"),
    ("CDNA1", "f32_4x4x2bf16"),
    # CDNA2 f16 (group_size=4, flush)
    ("CDNA2", "f32_32x32x8f16"),
    ("CDNA2", "f32_32x32x4f16"),
    ("CDNA2", "f32_16x16x16f16"),
    ("CDNA2", "f32_16x16x4f16"),
    ("CDNA2", "f32_4x4x4f16"),
    # CDNA2 bf16_1k (group_size=4, flush)
    ("CDNA2", "f32_32x32x8bf16_1k"),
    ("CDNA2", "f32_32x32x4bf16_1k"),
    ("CDNA2", "f32_16x16x16bf16_1k"),
    ("CDNA2", "f32_16x16x4bf16_1k"),
    ("CDNA2", "f32_4x4x4bf16_1k"),
    # CDNA2 bf16 (no suffix, group_size=2, flush)
    ("CDNA2", "f32_32x32x4bf16"),
    ("CDNA2", "f32_32x32x2bf16"),
    ("CDNA2", "f32_16x16x8bf16"),
    ("CDNA2", "f32_16x16x2bf16"),
    ("CDNA2", "f32_4x4x2bf16"),
}


_SUPPORTED_BLOCK_SCALE: set[tuple[str, str]] = {
    # RTX Blackwell mxf8f6f4 (k=32, per-tile ue8m0 scale). Phase M1c.
    ("RTX Blackwell", "m16n8k32.block32.f32.e5m2.e5m2.f32.ue8m0"),
    ("RTX Blackwell", "m16n8k32.block32.f32.e5m2.e4m3.f32.ue8m0"),
    ("RTX Blackwell", "m16n8k32.block32.f32.e4m3.e5m2.f32.ue8m0"),
    ("RTX Blackwell", "m16n8k32.block32.f32.e4m3.e4m3.f32.ue8m0"),
    # RTX Blackwell mxf4nvf4 (k=64, per-block ue8m0 scale). Phase M1d.
    ("RTX Blackwell", "m16n8k64.block32.f32.e2m1.e2m1.f32.ue8m0"),
    ("RTX Blackwell", "m16n8k64.block16.f32.e2m1.e2m1.f32.ue8m0"),
    # ue4m3 scale variants not yet supported (non-power-of-2 scales).
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
        if self.d_type is torch.float64:
            return self._call_f64_single(A, B, C)
        if self.d_type is torch.float16:
            return self._call_f16_out_single(A, B, C)

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

    def _call_f16_out_single(self, A, B, C):
        """f16-output path (RNE-FP16). Routes through batched entry with B=1."""
        A_f32 = self._to_f32_contig(A)
        B_f32 = self._to_f32_contig(B)
        C_f32 = self._to_f32_contig(C)
        # tf32 masking not applicable here (inputs are f16/bf16/fp8).
        out_u16 = _rs.mma_f16_out_batched_rayon(
            A_f32[None], B_f32[None], C_f32[None],
            self.nfb,
            self._a_min, self._b_min, self._c_min,
            self.is_split_k,
        )
        return torch.from_numpy(out_u16[0]).view(torch.float16)

    def _call_f64_single(self, A, B, C):
        # Single-call f64: route through the batched entry with batch=1.
        A_f64 = np.ascontiguousarray(A.detach().cpu().numpy(), dtype=np.float64)
        B_f64 = np.ascontiguousarray(B.detach().cpu().numpy(), dtype=np.float64)
        C_f64 = np.ascontiguousarray(C.detach().cpu().numpy(), dtype=np.float64)
        out = _rs.mma_f64_batched_rayon(A_f64[None], B_f64[None], C_f64[None])
        return torch.from_numpy(out[0])

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

    # ─────────────────────────────────────────────────────────────────
    # end of `mma` class methods — see `mma_block_scale` below for the
    # block-scaled variants.
    # ─────────────────────────────────────────────────────────────────

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


_SUPPORTED_WGMMA: set[tuple[str, str]] = {
    # Hopper wgmma, f32-output subset (Phase M4). f16-output would need
    # RNE instead of our RZ — deferred.
    ("Hopper", "m64n64k16.f32.f16.f16"),
    ("Hopper", "m64n64k16.f32.bf16.bf16"),
    ("Hopper", "m64n64k8.f32.tf32.tf32"),
    ("Hopper", "m64n128k16.f32.f16.f16"),
    ("Hopper", "m64n128k16.f32.bf16.bf16"),
    ("Hopper", "m64n128k8.f32.tf32.tf32"),
    ("Hopper", "m64n256k16.f32.f16.f16"),
    ("Hopper", "m64n256k8.f32.tf32.tf32"),
    ("Hopper", "m64n8k16.f32.f16.f16"),
    ("Hopper", "m64n8k16.f32.bf16.bf16"),
    ("Hopper", "m64n8k8.f32.tf32.tf32"),
}


class wgmma:
    """Rust-backed Hopper wgmma. f32-output subset only."""

    def __init__(self, arch: str, qualifier: str):
        from mmasim.simulator.nv_ptx import wgmma as _oracle_wgmma
        _ref = _oracle_wgmma(arch, qualifier)
        self.arch = arch
        self.qualifier = qualifier
        self.m, self.n, self.k = _ref.m, _ref.n, _ref.k
        self.a_type = _ref.a_type
        self.b_type = _ref.b_type
        self.c_type = _ref.c_type
        self.d_type = _ref.d_type
        self.nfb = _ref.n_accum_fractional_bits
        self.output_type = _ref.output_type

        if (arch, qualifier) not in _SUPPORTED_WGMMA:
            raise NotImplementedError(
                f"rust_ref.wgmma: ({arch!r}, {qualifier!r}) not supported"
            )

        if self.d_type is not torch.float32:
            raise NotImplementedError(
                f"wgmma f16-output variants need RNE (deferred)"
            )

        self._out_mant_bits = 13 if self.output_type == "f32_e8m13" else 23
        self._a_min = _MIN_EXP[self.a_type]
        self._b_min = _MIN_EXP[self.b_type]
        self._c_min = _MIN_EXP[self.c_type]

    def __call__(self, A, B, C):
        assert A.shape == (self.m, self.k)
        assert B.shape == (self.k, self.n)
        assert C.shape == (self.m, self.n)
        A32 = np.ascontiguousarray(A.detach().cpu().to(torch.float32).numpy(), dtype=np.float32)
        B32 = np.ascontiguousarray(B.detach().cpu().to(torch.float32).numpy(), dtype=np.float32)
        C32 = np.ascontiguousarray(C.detach().cpu().to(torch.float32).numpy(), dtype=np.float32)
        if self.a_type is torch.float32:  # tf32
            A32 = np.ascontiguousarray(((A32.view(np.int32) >> 13) << 13).view(np.float32))
            B32 = np.ascontiguousarray(((B32.view(np.int32) >> 13) << 13).view(np.float32))
        out = _rs.mma_f32_out_wgmma_batched_rayon(
            A32[None], B32[None], C32[None],
            self.nfb, self._a_min, self._b_min, self._c_min, self._out_mant_bits,
        )
        return torch.from_numpy(out[0])


class mfma:
    """AMD MFMA reimplementation (Phase M3, narrow scope).

    Currently supports only `fma` operation_type paths — f64 everywhere,
    f32 non-xf32 on all CDNA generations. Pairwise (CDNA1/2 f16/bf16)
    and fused_dot_rd_add (CDNA3 f16/bf16/fp8 + all xf32) paths raise
    NotImplementedError.
    """

    def __init__(self, arch: str, qualifier: str):
        from mmasim.simulator.amd import mfma as _oracle_mfma
        _ref = _oracle_mfma(arch, qualifier)
        self.arch = arch
        self.qualifier = qualifier
        self.m, self.n, self.k = _ref.m, _ref.n, _ref.k
        self.a_type = _ref.a_type
        self.b_type = _ref.b_type
        self.c_type = _ref.c_type
        self.d_type = _ref.d_type
        self.operation_type = _ref.operation_type
        self.group_size = _ref.group_size
        self.flush_denormal = _ref.flush_denormal
        self.is_xf32 = _ref.is_xf32

        if (arch, qualifier) not in _SUPPORTED_MFMA:
            raise NotImplementedError(
                f"rust_ref.mfma: not yet supported: ({arch!r}, {qualifier!r}) "
                f"[operation_type={self.operation_type}]"
            )

    def __call__(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        assert A.shape == (self.m, self.k)
        assert B.shape == (self.k, self.n)
        assert C.shape == (self.m, self.n)
        if self.d_type is torch.float64:
            A64 = np.ascontiguousarray(A.detach().cpu().numpy(), dtype=np.float64)
            B64 = np.ascontiguousarray(B.detach().cpu().numpy(), dtype=np.float64)
            C64 = np.ascontiguousarray(C.detach().cpu().numpy(), dtype=np.float64)
            out = _rs.mma_f64_batched_rayon(A64[None], B64[None], C64[None])
            return torch.from_numpy(out[0])
        if self.operation_type == "pairwise":
            # Widen A, B to f32 (matches the oracle patch). Flush A, B, C
            # before widening if CDNA2.
            A_pre = A.detach().cpu()
            B_pre = B.detach().cpu()
            C_pre = C.detach().cpu()
            if self.flush_denormal:
                # Match oracle's flush_denormal: in-place zero of subnormals.
                def _flush(x):
                    import mmasim.simulator.arithmetic as _ar
                    min_e = _ar.dtype_min_exponent[x.dtype]
                    x = x.clone()
                    x[x.abs() < 2.0 ** min_e] = 0.0
                    return x
                A_pre = _flush(A_pre)
                B_pre = _flush(B_pre)
                C_pre = _flush(C_pre)
            A32 = np.ascontiguousarray(A_pre.to(torch.float32).numpy(), dtype=np.float32)
            B32 = np.ascontiguousarray(B_pre.to(torch.float32).numpy(), dtype=np.float32)
            C32 = np.ascontiguousarray(C_pre.to(torch.float32).numpy(), dtype=np.float32)
            out = _rs.mma_f32_amd_pairwise_rayon(
                A32[None], B32[None], C32[None],
                int(self.group_size), bool(self.flush_denormal),
            )
            return torch.from_numpy(out[0])
        if self.d_type is torch.float32 and self.operation_type == "fma":
            A32 = np.ascontiguousarray(A.detach().cpu().to(torch.float32).numpy(), dtype=np.float32)
            B32 = np.ascontiguousarray(B.detach().cpu().to(torch.float32).numpy(), dtype=np.float32)
            C32 = np.ascontiguousarray(C.detach().cpu().to(torch.float32).numpy(), dtype=np.float32)
            out = _rs.mma_f32_fma_batched_rayon(A32[None], B32[None], C32[None])
            return torch.from_numpy(out[0])
        raise NotImplementedError(
            f"mfma operation_type={self.operation_type} d_type={self.d_type} not implemented"
        )


class mma_block_scale:
    """Rust-backed block-scaled MMA (RTX Blackwell mxf8f6f4 / mxf4nvf4).

    Currently supports: k=32 mxf8f6f4 with ue8m0 (power-of-2) scales.
    Other variants (k=64 mxfp4, ue4m3 scales) raise NotImplementedError.
    """

    def __init__(self, arch: str, qualifier: str):
        from mmasim.simulator.nv_ptx import mma_block_scale as _oracle_mbs
        _ref = _oracle_mbs(arch, qualifier)
        self.arch = arch
        self.qualifier = qualifier
        self.m, self.n, self.k = _ref.m, _ref.n, _ref.k
        self.block_size = _ref.block_size
        self.packing = _ref.packing
        self.a_type = _ref.a_type
        self.b_type = _ref.b_type
        self.c_type = _ref.c_type
        self.d_type = _ref.d_type
        self.s_type = _ref.s_type
        self.nfb = _ref.n_accum_fractional_bits
        self.output_type = _ref.output_type

        if (arch, qualifier) not in _SUPPORTED_BLOCK_SCALE:
            raise NotImplementedError(
                f"rust_ref.mma_block_scale: not yet supported: "
                f"({arch!r}, {qualifier!r})"
            )

        self._out_mant_bits = 13 if self.output_type == "f32_e8m13" else 23
        # For mxfp4 (a_type=uint8 packed), decompose doesn't happen on A/B
        # directly so a_min / b_min are unused.
        self._a_min = _MIN_EXP.get(self.a_type, 0)
        self._b_min = _MIN_EXP.get(self.b_type, 0)
        self._c_min = _MIN_EXP[self.c_type]

    @staticmethod
    def _as_f32(t: torch.Tensor) -> np.ndarray:
        return np.ascontiguousarray(
            t.detach().cpu().to(torch.float32).contiguous().numpy(),
            dtype=np.float32,
        )

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        scale_A: torch.Tensor,
        scale_B: torch.Tensor,
    ) -> torch.Tensor:
        # Non-batched: wrap with batch=1 and route through the batched
        # kernel. We lose a small amount of overhead but it's fine for
        # the single-call (bench.validate) path.
        out = self.call_batched(
            A.unsqueeze(0), B.unsqueeze(0), C.unsqueeze(0),
            scale_A.unsqueeze(0), scale_B.unsqueeze(0),
        )
        return out[0]

    def call_batched(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        scale_A: torch.Tensor,
        scale_B: torch.Tensor,
    ) -> torch.Tensor:
        """Batched: shapes as per the oracle."""
        if self.k == 32:
            return self._call_batched_k32(A, B, C, scale_A, scale_B)
        elif self.k == 64:
            return self._call_batched_mxfp4(A, B, C, scale_A, scale_B)
        else:
            raise NotImplementedError(f"k={self.k} not supported")

    def _call_batched_k32(self, A, B, C, scale_A, scale_B):
        # A=(B, m, k), B=(B, k, n), scales are per-tile (shape (B, m, 1) / (B, 1, n)).
        assert A.shape[1:] == (self.m, self.k)
        assert B.shape[1:] == (self.k, self.n)
        assert C.shape[1:] == (self.m, self.n)
        assert scale_A.shape[1:] == (self.m, 1)
        assert scale_B.shape[1:] == (1, self.n)

        A_f32 = self._as_f32(A)
        B_f32 = self._as_f32(B)
        C_f32 = self._as_f32(C)
        sA_f32 = self._as_f32(scale_A)
        sB_f32 = self._as_f32(scale_B)

        out = _rs.mma_f32_out_block_scale_k32_rayon(
            A_f32, B_f32, C_f32, sA_f32, sB_f32,
            self.nfb,
            self._a_min, self._b_min, self._c_min,
            self._out_mant_bits,
        )
        return torch.from_numpy(out)

    def _call_batched_mxfp4(self, A, B, C, scale_A, scale_B):
        # A, B are u8-packed fp4: shapes (B, m, k/2), (B, k/2, n).
        # Scales shape: (B, m, k/block_size), (B, k/block_size, n).
        half_k = self.k // 2
        n_sc = self.k // self.block_size
        assert A.shape[1:] == (self.m, half_k)
        assert B.shape[1:] == (half_k, self.n)
        assert C.shape[1:] == (self.m, self.n)
        assert scale_A.shape[1:] == (self.m, n_sc)
        assert scale_B.shape[1:] == (n_sc, self.n)

        A_u8 = np.ascontiguousarray(A.detach().cpu().numpy(), dtype=np.uint8)
        B_u8 = np.ascontiguousarray(B.detach().cpu().numpy(), dtype=np.uint8)
        C_f32 = self._as_f32(C)
        sA_f32 = self._as_f32(scale_A)
        sB_f32 = self._as_f32(scale_B)

        out = _rs.mma_f32_out_mxfp4_k64_rayon(
            A_u8, B_u8, C_f32, sA_f32, sB_f32,
            self.nfb, int(self.block_size), self._out_mant_bits,
        )
        return torch.from_numpy(out)
