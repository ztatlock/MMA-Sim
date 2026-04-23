"""Vectorized NumPy reimplementation of mmasim's mma path.

Bit-exact against the oracle. Drops all per-element Python loops in
`nv_fused_dot_add` / `fused_sum` / `extract_significand_exponent`; does
the whole (M, N) output tile in one pass.

Coverage (phase 1):
- `mma` with f16 / bf16 / tf32 inputs, f32 output
- split-K and non-split-K paths
- TF32 input truncation
- Subnormal flush, zero-exp quirk, NaN/Inf short-circuit

Out of scope (phase 1 follow-ups):
- f64 inputs (pairwise path)
- fp8 inputs (Ada's n_fractional_bits=13 / f32_e8m13 output)
- fp4 inputs, block-scaled variants, wgmma, tcgen05mma, AMD
"""
from __future__ import annotations

import numpy as np
import torch

# Mirror mmasim/simulator/arithmetic.py:dtype_min_exponent
_DTYPE_MIN_EXP = {
    torch.float64:       -1022,
    torch.float32:       -126,
    torch.float16:       -14,
    torch.bfloat16:      -126,
    torch.float8_e4m3fn: -6,
    torch.float8_e5m2:   -14,
}


def _truncate_to_tf32(x_f32: np.ndarray) -> np.ndarray:
    """Mask off the low 13 mantissa bits of an f32 tensor. Mirrors
    mmasim/simulator/arithmetic.py:truncate_to_tf32."""
    assert x_f32.dtype == np.float32
    raw = x_f32.view(np.int32)
    truncated = (raw >> np.int32(13)) << np.int32(13)
    return truncated.view(np.float32)


def _extract_sig_exp(x_f64: np.ndarray, min_exp: int):
    """Vectorized extract_significand_exponent.

    Input: f64 ndarray (values of original operand, after any pre-cast).
    Output: (sig, exp) where sig is f64 in [1,2) or 0, exp is int64.
    Applies subnormal flush at `min_exp` and the zero-exp=-126 quirk.
    """
    sig, exp = np.frexp(x_f64)     # sig in [0.5, 1), exp int
    sig = sig * 2.0                # [1, 2)
    exp = exp - 1
    exp = exp.astype(np.int64)

    # Subnormal flush at target dtype's min exponent.
    sub = exp < min_exp
    if sub.any():
        scale = np.exp2((exp - min_exp).astype(np.float64))
        sig = np.where(sub, sig * scale, sig)
        exp = np.where(sub, np.int64(min_exp), exp)

    # Zero quirk: upstream sets exp=-126 when sig==0, regardless of dtype.
    zero = sig == 0.0
    exp = np.where(zero, np.int64(-126), exp)
    return sig, exp


def _fused_sum(sigs: np.ndarray, exps: np.ndarray, nfb: int):
    """Vectorized fused_sum over last axis. Returns (sum_val, max_e)."""
    max_e = exps.max(axis=-1, keepdims=True)
    shift = nfb + exps - max_e                       # <= nfb, int
    scaled = sigs * np.exp2(shift.astype(np.float64))
    trunced = np.trunc(scaled)                       # RZ
    sum_val = trunced.sum(axis=-1) * (2.0 ** -nfb)
    return sum_val, max_e.squeeze(-1)


def _normalize_f32(result_f64: np.ndarray, out_mantissa_bits: int = 23) -> np.ndarray:
    """RZ to `out_mantissa_bits` of f32 mantissa, cast to f32. Passes
    NaN/Inf through. Set out_mantissa_bits=13 for Ada Lovelace's
    f32_e8m13 output."""
    out = np.empty(result_f64.shape, dtype=np.float32)
    nan = np.isnan(result_f64)
    inf = np.isinf(result_f64)
    normal = ~(nan | inf)

    # NaN: oracle emits 0x7FFF_FFFF (quiet NaN pattern). Match it.
    nan_bits = np.array(0x7FFF_FFFF, dtype=np.int32).view(np.float32)
    out[nan] = nan_bits
    out[inf] = result_f64[inf].astype(np.float32)

    if normal.any():
        v = result_f64[normal]
        sig, exp = _extract_sig_exp(v, _DTYPE_MIN_EXP[torch.float32])
        scale = 2.0 ** out_mantissa_bits
        sig = np.trunc(sig * scale) / scale
        out[normal] = (sig * np.exp2(exp.astype(np.float64))).astype(np.float32)
    return out


def _fused_mma_step(
    A64: np.ndarray, B64: np.ndarray, C64: np.ndarray,
    nfb: int, a_min: int, b_min: int, c_min: int,
    out_mantissa_bits: int,
) -> np.ndarray:
    """One fused dot-add across K, vectorized over (M, N). Returns f32."""
    # f64 pre-check for NaN/Inf (matches upstream short-circuit).
    fp64_sum = C64 + A64 @ B64
    bad = np.isnan(fp64_sum) | np.isinf(fp64_sum)

    a_sig, a_exp = _extract_sig_exp(A64, a_min)               # (M, K)
    b_sig, b_exp = _extract_sig_exp(B64, b_min)               # (K, N)
    c_sig, c_exp = _extract_sig_exp(C64, c_min)               # (M, N)

    # Products (M, N, K): p[i,j,l] = A[i,l] * B[l,j]
    p_sig = a_sig[:, None, :] * b_sig.T[None, :, :]
    p_exp = a_exp[:, None, :] + b_exp.T[None, :, :]

    sig_all = np.concatenate([c_sig[..., None], p_sig], axis=-1)
    exp_all = np.concatenate([c_exp[..., None], p_exp], axis=-1)

    sum_val, max_e = _fused_sum(sig_all, exp_all, nfb)
    result_f64 = sum_val * np.exp2(max_e.astype(np.float64))

    # Reinstate NaN/Inf from the fp64 pre-check.
    if bad.any():
        result_f64 = np.where(bad, fp64_sum, result_f64)

    return _normalize_f32(result_f64, out_mantissa_bits=out_mantissa_bits)


class mma:
    """NumPy reimplementation of mmasim.simulator.nv_ptx.mma (subset)."""

    def __init__(self, arch: str, qualifier: str):
        # Reuse the ISA class for parsing + capability tracking. We don't
        # call its __call__; we only read its parsed fields.
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
        self.n_accum_fractional_bits = _ref.n_accum_fractional_bits
        self.output_type = _ref.output_type

        if self.d_type is not torch.float32:
            raise NotImplementedError(f"output dtype {self.d_type} not supported yet")
        if self.a_type is torch.float64:
            raise NotImplementedError("f64 input path not implemented yet")
        if self.a_type is torch.uint8:
            raise NotImplementedError("fp4-packed input not implemented yet")

        # Output mantissa bits: 23 for normal f32, 13 for Ada's f32_e8m13.
        self._out_mant_bits = 13 if self.output_type == "f32_e8m13" else 23

    def _to_f32_numpy(self, t: torch.Tensor) -> np.ndarray:
        # bf16 / f16 / f32 all cast to f32 losslessly (bf16/f16 are strict
        # subsets of f32). tf32 stays f32; we apply the mask below.
        return t.detach().cpu().to(torch.float32).numpy()

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        assert A.shape == (self.m, self.k)
        assert B.shape == (self.k, self.n)
        assert C.shape == (self.m, self.n)

        A_f32 = self._to_f32_numpy(A)
        B_f32 = self._to_f32_numpy(B)
        C_f32 = self._to_f32_numpy(C)

        if self.a_type is torch.float32:  # tf32
            A_f32 = _truncate_to_tf32(A_f32)
            B_f32 = _truncate_to_tf32(B_f32)

        A64 = A_f32.astype(np.float64)
        B64 = B_f32.astype(np.float64)
        C64 = C_f32.astype(np.float64)

        a_min = _DTYPE_MIN_EXP[self.a_type]
        b_min = _DTYPE_MIN_EXP[self.b_type]
        c_min = _DTYPE_MIN_EXP[self.c_type]
        nfb = self.n_accum_fractional_bits

        omb = self._out_mant_bits
        if self.is_split_k:
            half = self.k // 2
            mid = _fused_mma_step(
                A64[:, :half], B64[:half, :], C64,
                nfb, a_min, b_min, c_min, omb,
            ).astype(np.float64)
            out = _fused_mma_step(
                A64[:, half:], B64[half:, :], mid,
                nfb, a_min, b_min, c_min, omb,
            )
        else:
            out = _fused_mma_step(A64, B64, C64, nfb, a_min, b_min, c_min, omb)

        return torch.from_numpy(out)
