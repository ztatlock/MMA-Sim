//! Rust reimplementation of mmasim's mma path.
//!
//! Phase 2.1: integer fixed-point inner loop. Each input is decomposed
//! from its raw f32 bits to a signed 64-bit integer at scale 2^nfb, with
//! subnormal flush applied at the source dtype's min_exp. The fused-sum
//! alignment and accumulation is done in pure integer arithmetic. Only
//! the final per-output normalize uses f64 (for RZ rounding and f32 cast).
//!
//! Inputs still arrive as contiguous f32 arrays (f32 is a strict superset
//! of f16 / bf16 / tf32, so the cast is lossless). The wrapper passes
//! each source dtype's min_exp; that's enough to recover bit-exact
//! semantics.

use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

#[cfg(target_os = "macos")]
mod metal_ampere_f16;

const F32_MIN_EXP: i32 = -126;

/// Decompose an f32 value to (signed int_sig at scale 2^nfb, exp) with
/// subnormal flush at `min_exp`. Returns `is_special=true` for NaN/Inf;
/// caller must check the f64 pre-sum in that case.
#[inline(always)]
fn decompose(v: f32, min_exp: i32, nfb: i32) -> (i64, i32, bool) {
    let bits = v.to_bits();
    let sign = (bits >> 31) & 1 != 0;
    let biased_e = ((bits >> 23) & 0xFF) as i32;
    let mant = (bits & 0x7F_FFFF) as i64;

    if biased_e == 0xFF {
        return (0, 0, true); // NaN/Inf
    }
    if biased_e == 0 && mant == 0 {
        return (0, -126, false); // zero quirk
    }

    // Normalize to sig at scale 2^23, exp in the "frexp/*2/-1" convention.
    let (sig_q23, exp) = if biased_e == 0 {
        // f32 subnormal. Find the top mantissa bit, shift up.
        let top = 63 - (mant as u64).leading_zeros() as i32;
        (mant << (23 - top), top - 149)
    } else {
        ((1i64 << 23) + mant, biased_e - 127)
    };

    // Shift to scale 2^nfb. For Ada-fp8 we have nfb=13 < 23, so this may
    // be a right-shift (intentional precision loss to nfb bits of sig;
    // matches the oracle's `trunc(sig * 2^nfb)` path).
    let shift_up = nfb - 23;
    let mut int_sig = if shift_up >= 0 {
        sig_q23 << shift_up
    } else {
        // Right-shift on a non-negative magnitude is lossless for f8
        // inputs (mantissa has ≤ 3 bits, so the bottom 20+ bits of
        // sig_q23 are zero). For f32 C under nfb=13, this truncates sig
        // to 13 bits — which is what the oracle does at max_e-aligned
        // products anyway. Trunc-toward-zero is achieved via sign
        // applied *after* this shift.
        sig_q23 >> (-shift_up)
    };

    let mut exp = exp;
    if exp < min_exp {
        // Flush: sig *= 2^(exp - min_exp), exp = min_exp. Right shift
        // the non-negative int_sig. For the dtypes we care about, the
        // shifted-out bits are always zero (f16/bf16/tf32 -> f32 cast
        // leaves trailing zeros at least as wide as the max flush shift).
        let shift = (min_exp - exp) as u32;
        int_sig = if shift >= 64 { 0 } else { int_sig >> shift };
        exp = min_exp;
        if int_sig == 0 {
            exp = -126;
        }
    }

    let signed = if sign { -int_sig } else { int_sig };
    (signed, exp, false)
}

/// Signed trunc-to-zero right shift. Rust's signed `/` rounds toward zero;
/// shifts round toward −∞. We want the former (matches `math.trunc`).
#[inline(always)]
fn trunc_shr(x: i64, shift: i32) -> i64 {
    if shift <= 0 {
        return x;
    }
    if shift >= 63 {
        return 0;
    }
    x / (1i64 << shift)
}

/// Left-shift with guard. Used when aligning a smaller-exponent term up
/// to the shared scale; in our kernel the shifts are always small
/// (bounded by nfb), but the guard costs nothing and documents intent.
#[inline(always)]
fn guarded_shl(x: i64, shift: i32) -> i64 {
    if shift <= 0 {
        return x;
    }
    if shift >= 63 {
        return 0;
    }
    x << shift
}

/// Mirrors fastmma/numpy_ref.py::_normalize_f32 — RZ to `out_mantissa_bits`,
/// cast to f32, NaN/Inf passthrough.
#[inline(always)]
/// Normalize a computed f64 value to f16 bits with RNE-to-10-mantissa-bits.
/// Mirrors the oracle's f16-output path:
///   s, e = extract_sig_exp(result, f16_min_exp=-14)
///   s = round_ties_even(s * 2^10) / 2^10
///   return f16(s * 2^e)
/// NaN pattern: 0x7FFF (oracle convention).
fn normalize_f16_rne_bits(result_f64: f64) -> u16 {
    if result_f64.is_nan() {
        return 0x7FFFu16;
    }
    if result_f64.is_infinite() {
        return if result_f64 > 0.0 { 0x7C00 } else { 0xFC00 };
    }
    if result_f64 == 0.0 {
        return if result_f64.is_sign_negative() { 0x8000 } else { 0 };
    }

    // Extract (sig, exp) with flush at f16's min_exp = -14.
    let (mut s, e_isz) = libm::frexp(result_f64);
    let mut e: i32 = e_isz as i32;
    s *= 2.0;
    e -= 1;
    if e < -14 {
        s *= fast_pow2(e - (-14));
        e = -14;
    }
    // RNE to 10 mantissa bits: round to nearest multiple of 2^-10.
    let mut s_rne = (s * 1024.0).round_ties_even() / 1024.0;

    // If rounding pushed |s| to 2.0 exactly, re-normalize.
    if s_rne >= 2.0 {
        s_rne = 1.0;
        e += 1;
    } else if s_rne <= -2.0 {
        s_rne = -1.0;
        e += 1;
    }

    // Overflow → ±Inf.
    if e > 15 {
        return if s_rne > 0.0 { 0x7C00 } else { 0xFC00 };
    }

    let sign_bit: u16 = if s_rne < 0.0 { 0x8000 } else { 0 };
    let mag = s_rne.abs();

    // Subnormal: e == -14 and mag < 1. Encode with biased_e=0, mantissa = mag * 1024.
    // (Our flush above only sets e to -14, mag into [0, 1) in that case.)
    if mag < 1.0 {
        debug_assert_eq!(e, -14);
        let mantissa = (mag * 1024.0).round_ties_even() as u16;
        // If mantissa rounds up to 1024, that's the smallest normal (biased_e=1, mant=0).
        if mantissa >= 1024 {
            return sign_bit | (1u16 << 10);
        }
        return sign_bit | (mantissa & 0x3FF);
    }

    // Normal: biased_e = e + 15, mantissa = (mag - 1) * 1024.
    let biased_e = (e + 15) as u16;
    let mantissa = ((mag - 1.0) * 1024.0).round_ties_even() as u16;
    sign_bit | (biased_e << 10) | (mantissa & 0x3FF)
}

fn normalize_f32(result_f64: f64, out_mantissa_bits: i32) -> f32 {
    if result_f64.is_nan() {
        return f32::from_bits(0x7FFF_FFFF);
    }
    if result_f64.is_infinite() {
        return result_f64 as f32;
    }
    if result_f64 == 0.0 {
        return 0.0;
    }
    // Re-decompose the f64 result at f32's min_exp.
    let (sig_int, exp, _) = decompose_f64(result_f64, F32_MIN_EXP);
    let scale = f64::from_bits(((1023 + out_mantissa_bits) as u64) << 52); // 2^mant_bits
    let sig_f64 = sig_int as f64 / f64::from_bits(((1023 + 23) as u64) << 52); // /2^23
    let sig_trunc = (sig_f64 * scale).trunc() / scale;
    (sig_f64_to_pow2_times(sig_trunc, exp)) as f32
}

#[inline(always)]
fn decompose_f64(v: f64, min_exp: i32) -> (i64, i32, bool) {
    // Same shape as decompose(f32) but from f64 bits, producing an int_sig
    // at scale 2^23 (matches the f32 mantissa count we use for output).
    let bits = v.to_bits();
    let sign = (bits >> 63) & 1 != 0;
    let biased_e = ((bits >> 52) & 0x7FF) as i32;
    let mant = (bits & ((1u64 << 52) - 1)) as i128;

    if biased_e == 0x7FF {
        return (0, 0, true);
    }
    if biased_e == 0 && mant == 0 {
        return (0, -126, false);
    }

    let (sig_q52, exp) = if biased_e == 0 {
        let top = 127 - (mant as u128).leading_zeros() as i32;
        (mant << (52 - top), top - 1074)
    } else {
        ((1i128 << 52) + mant, biased_e - 1023)
    };

    // Shift down to scale 2^23 (drop the low 29 bits).
    let sig_q23 = (sig_q52 >> 29) as i64;
    let mut int_sig = sig_q23;
    let mut exp = exp;

    if exp < min_exp {
        let shift = (min_exp - exp) as u32;
        int_sig = if shift >= 64 { 0 } else { int_sig >> shift };
        exp = min_exp;
        if int_sig == 0 {
            exp = -126;
        }
    }

    let signed = if sign { -int_sig } else { int_sig };
    (signed, exp, false)
}

/// Reconstruct sig * 2^exp as f64. sig is a value in (-2, 2), exp is int.
#[inline(always)]
fn sig_f64_to_pow2_times(sig: f64, exp: i32) -> f64 {
    sig * fast_pow2(exp)
}

#[inline(always)]
fn fast_pow2(n: i32) -> f64 {
    // 2^n via direct biased-exponent construction. Normal range only;
    // for out-of-range the normalize path handles it via the f64 result
    // being Inf/subnormal.
    if n > 1023 {
        return f64::INFINITY;
    }
    if n < -1022 {
        // subnormal / underflow. Construct via scaling from smallest normal.
        if n < -1074 {
            return 0.0;
        }
        let shift = (-1022 - n) as u64;
        return f64::from_bits(1u64 << (52 - shift));
    }
    f64::from_bits(((n + 1023) as u64) << 52)
}

// Max m*k, k*n, m*n across all supported tile sizes. Bump if new tiles
// exceed this. Currently m=16, n=8, k ≤ 64 → max 512 for block-scaled
// paths we haven't wired up yet; 256 is enough for the phase-2.1 set.
const MAX_ELEMS: usize = 256;

/// One fused dot-add, integer inner loop, f32 output. Stack-allocated
/// decomposition buffers (sized for our tile tables).
#[inline]
#[allow(clippy::too_many_arguments)]
fn fused_mma_step_int(
    a: &[f32], // (M, K) row-major
    b: &[f32], // (K, N) row-major
    c: &[f32], // (M, N) row-major
    m: usize,
    n: usize,
    k: usize,
    nfb: i32,
    a_min_exp: i32,
    b_min_exp: i32,
    c_min_exp: i32,
    out_mantissa_bits: i32,
    out: &mut [f32],
) {
    assert!(m * k <= MAX_ELEMS, "tile exceeds MAX_ELEMS");
    assert!(k * n <= MAX_ELEMS, "tile exceeds MAX_ELEMS");

    // Pre-decompose. (sig_i64, exp_i32) at scale 2^nfb.
    let mut a_sig = [0i64; MAX_ELEMS];
    let mut a_exp = [0i32; MAX_ELEMS];
    let mut a_special = 0u64; // at most 64 A-elements with MAX_ELEMS=256? No — use bitset at byte granularity later
    let mut a_sp = [false; MAX_ELEMS];
    for i in 0..m * k {
        let (s, e, sp) = decompose(a[i], a_min_exp, nfb);
        a_sig[i] = s;
        a_exp[i] = e;
        a_sp[i] = sp;
        a_special |= sp as u64;
    }
    let mut b_sig = [0i64; MAX_ELEMS];
    let mut b_exp = [0i32; MAX_ELEMS];
    let mut b_sp = [false; MAX_ELEMS];
    let mut b_special = 0u64;
    for i in 0..k * n {
        let (s, e, sp) = decompose(b[i], b_min_exp, nfb);
        b_sig[i] = s;
        b_exp[i] = e;
        b_sp[i] = sp;
        b_special |= sp as u64;
    }
    let _ = (a_special, b_special); // suppress unused (folded into per-output check)

    for i in 0..m {
        for j in 0..n {
            let (c_sig_ij, c_exp_ij, c_special) = decompose(c[i * n + j], c_min_exp, nfb);

            // Running f64 sum for NaN/Inf detection, matches upstream.
            let mut any_special = c_special;
            let mut fp_sum = c[i * n + j] as f64;

            // Collect product (sig, exp) and track max_e.
            let mut prod_sig = [0i64; 128];
            let mut prod_exp = [0i32; 128];
            let mut max_e = c_exp_ij;

            for l in 0..k {
                let a_s = a_sig[i * k + l];
                let a_e = a_exp[i * k + l];
                let b_s = b_sig[l * n + j];
                let b_e = b_exp[l * n + j];
                any_special |= a_sp[i * k + l] | b_sp[l * n + j];

                fp_sum += (a[i * k + l] as f64) * (b[l * n + j] as f64);

                // Product int sig: i64 * i64 -> i64. For supported sizes the
                // magnitude stays well under 2^62. f16/bf16/tf32 x f16/bf16/tf32
                // at nfb ≤ 25: each sig ≤ 2^(nfb+1) ≤ 2^26, product ≤ 2^52.
                let ps = a_s * b_s;
                let pe = a_e + b_e;
                prod_sig[l] = ps;
                prod_exp[l] = pe;
                if pe > max_e {
                    max_e = pe;
                }
            }

            // Early exit for NaN/Inf.
            let result_f64 = if any_special || !fp_sum.is_finite() {
                fp_sum
            } else {
                // Align and sum. All sigs are at scale 2^nfb. Products are
                // at scale 2^(2*nfb); to align to 2^nfb at max_e:
                //   aligned = prod_sig >> (max_e - pe + nfb)  (trunc-to-zero)
                // C term is already at scale 2^nfb:
                //   aligned = c_sig >> (max_e - c_exp)
                let mut acc: i64 = trunc_shr(c_sig_ij, max_e - c_exp_ij);
                for l in 0..k {
                    let shift = max_e - prod_exp[l] + nfb;
                    acc += trunc_shr(prod_sig[l], shift);
                }
                // Recover f64 value: acc / 2^nfb * 2^max_e = acc * 2^(max_e - nfb)
                (acc as f64) * fast_pow2(max_e - nfb)
            };

            out[i * n + j] = normalize_f32(result_f64, out_mantissa_bits);
        }
    }
}

#[pyfunction]
#[pyo3(signature = (a, b, c, nfb, a_min_exp, b_min_exp, c_min_exp, out_mantissa_bits, split_k))]
fn mma_f32_out<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
    c: PyReadonlyArray2<'py, f32>,
    nfb: i32,
    a_min_exp: i32,
    b_min_exp: i32,
    c_min_exp: i32,
    out_mantissa_bits: i32,
    split_k: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let a_v = a.as_array();
    let b_v = b.as_array();
    let c_v = c.as_array();
    let (m, k) = (a_v.shape()[0], a_v.shape()[1]);
    let (_, n) = (b_v.shape()[0], b_v.shape()[1]);
    assert_eq!(k, b_v.shape()[0]);
    assert_eq!(c_v.shape(), &[m, n]);
    assert!(k + 1 <= 128, "k too large for stack-alloc prod arrays");

    // Python wrapper calls np.ascontiguousarray; as_slice() should succeed.
    // Fall back to an owning copy if it doesn't.
    let a_owned: Vec<f32>;
    let a_slice: &[f32] = match a_v.as_slice() {
        Some(s) => s,
        None => {
            a_owned = a_v.iter().copied().collect();
            &a_owned
        }
    };
    let b_owned: Vec<f32>;
    let b_slice: &[f32] = match b_v.as_slice() {
        Some(s) => s,
        None => {
            b_owned = b_v.iter().copied().collect();
            &b_owned
        }
    };
    let c_owned: Vec<f32>;
    let c_slice: &[f32] = match c_v.as_slice() {
        Some(s) => s,
        None => {
            c_owned = c_v.iter().copied().collect();
            &c_owned
        }
    };

    let mut out = vec![0.0f32; m * n];
    run_one_tile(
        a_slice, b_slice, c_slice, m, n, k,
        nfb, a_min_exp, b_min_exp, c_min_exp,
        out_mantissa_bits, split_k, &mut out,
    );

    let arr = ndarray::Array2::from_shape_vec((m, n), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

/// Run the kernel once for a given (a,b,c) slice into a per-tile output
/// slot. Internal helper used by both the single-shot and batched entry
/// points. Assumes slices are tile-sized and contiguous row-major.
#[inline]
#[allow(clippy::too_many_arguments)]
fn run_one_tile(
    a_slice: &[f32], b_slice: &[f32], c_slice: &[f32],
    m: usize, n: usize, k: usize,
    nfb: i32, a_min_exp: i32, b_min_exp: i32, c_min_exp: i32,
    out_mantissa_bits: i32, split_k: bool,
    out_slice: &mut [f32],
) {
    if split_k {
        let half = k / 2;
        let mut mid = [0f32; MAX_ELEMS];
        let mut a_half = [0f32; MAX_ELEMS];
        let mut a_half2 = [0f32; MAX_ELEMS];
        for i in 0..m {
            a_half[i * half..(i + 1) * half]
                .copy_from_slice(&a_slice[i * k..i * k + half]);
            a_half2[i * half..(i + 1) * half]
                .copy_from_slice(&a_slice[i * k + half..(i + 1) * k]);
        }

        fused_mma_step_int(
            &a_half[..m * half], &b_slice[..half * n], c_slice, m, n, half,
            nfb, a_min_exp, b_min_exp, c_min_exp, out_mantissa_bits,
            &mut mid[..m * n],
        );
        fused_mma_step_int(
            &a_half2[..m * half], &b_slice[half * n..], &mid[..m * n], m, n, half,
            nfb, a_min_exp, b_min_exp, F32_MIN_EXP, out_mantissa_bits, out_slice,
        );
    } else {
        fused_mma_step_int(
            a_slice, b_slice, c_slice, m, n, k,
            nfb, a_min_exp, b_min_exp, c_min_exp, out_mantissa_bits, out_slice,
        );
    }
}

/// Batched entry: run B independent MMAs in a single call. Amortizes the
/// per-call Python/FFI overhead across the batch. Inputs are 3-D:
///   a: (B, M, K), b: (B, K, N), c: (B, M, N). Output: (B, M, N).
#[pyfunction]
#[pyo3(signature = (a, b, c, nfb, a_min_exp, b_min_exp, c_min_exp, out_mantissa_bits, split_k))]
#[allow(clippy::too_many_arguments)]
fn mma_f32_out_batched<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f32>,
    b: PyReadonlyArray3<'py, f32>,
    c: PyReadonlyArray3<'py, f32>,
    nfb: i32,
    a_min_exp: i32,
    b_min_exp: i32,
    c_min_exp: i32,
    out_mantissa_bits: i32,
    split_k: bool,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let a_v = a.as_array();
    let b_v = b.as_array();
    let c_v = c.as_array();
    let batch = a_v.shape()[0];
    let m = a_v.shape()[1];
    let k = a_v.shape()[2];
    let n = b_v.shape()[2];
    assert_eq!(b_v.shape(), &[batch, k, n]);
    assert_eq!(c_v.shape(), &[batch, m, n]);
    assert!(m * k <= MAX_ELEMS && k * n <= MAX_ELEMS && m * n <= MAX_ELEMS);

    let a_owned: Vec<f32>;
    let a_slice: &[f32] = match a_v.as_slice() {
        Some(s) => s,
        None => { a_owned = a_v.iter().copied().collect(); &a_owned }
    };
    let b_owned: Vec<f32>;
    let b_slice: &[f32] = match b_v.as_slice() {
        Some(s) => s,
        None => { b_owned = b_v.iter().copied().collect(); &b_owned }
    };
    let c_owned: Vec<f32>;
    let c_slice: &[f32] = match c_v.as_slice() {
        Some(s) => s,
        None => { c_owned = c_v.iter().copied().collect(); &c_owned }
    };

    let mut out = vec![0.0f32; batch * m * n];
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    for bi in 0..batch {
        let a_i = &a_slice[bi * a_stride..(bi + 1) * a_stride];
        let b_i = &b_slice[bi * b_stride..(bi + 1) * b_stride];
        let c_i = &c_slice[bi * c_stride..(bi + 1) * c_stride];
        let out_i = &mut out[bi * c_stride..(bi + 1) * c_stride];
        run_one_tile(
            a_i, b_i, c_i, m, n, k,
            nfb, a_min_exp, b_min_exp, c_min_exp,
            out_mantissa_bits, split_k, out_i,
        );
    }

    let arr = ndarray::Array3::from_shape_vec((batch, m, n), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

// ─────────────────────────────────────────────────────────────────────
// SIMD variant (aarch64 NEON, 2-wide on the j-axis).
//
// Phase 2.3: vectorize the per-output align+accumulate across two
// adjacent output columns (j, j+1). Scalar bookkeeping (pre-decomposes,
// max_e scan, product collection) stays scalar; the k+1-term reduction
// becomes i64x2 add with a vectorized trunc-toward-zero right shift.
//
// On non-aarch64 builds, `mma_f32_out_batched_simd` falls back to the
// scalar kernel so the Python side can still import the function name.
// ─────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    /// Vectorized trunc-toward-zero right shift: for each lane,
    ///   result_i = sig_i / 2^shift_i  rounded toward zero.
    /// Requires 0 ≤ shift_i ≤ 63. Arithmetic right shift of an
    /// non-negative magnitude is equivalent to the desired trunc.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) unsafe fn trunc_shr_v(sig: int64x2_t, shift: int64x2_t) -> int64x2_t {
        // sign_mask: 0 for positives, all-ones for negatives.
        let sign_mask = vshrq_n_s64(sig, 63);
        // abs = (sig XOR sign_mask) - sign_mask
        let abs_sig = vsubq_s64(veorq_s64(sig, sign_mask), sign_mask);
        // vshlq with negative count = arith right shift; abs is ≥ 0 so
        // arithmetic vs logical is the same.
        let shifted = vshlq_s64(abs_sig, vnegq_s64(shift));
        // Reapply sign.
        vsubq_s64(veorq_s64(shifted, sign_mask), sign_mask)
    }

    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) unsafe fn pack2_s64(lo: i64, hi: i64) -> int64x2_t {
        let arr = [lo, hi];
        vld1q_s64(arr.as_ptr())
    }

    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) unsafe fn unpack2_s64(v: int64x2_t) -> (i64, i64) {
        (vgetq_lane_s64::<0>(v), vgetq_lane_s64::<1>(v))
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn fused_mma_step_int_neon(
    a: &[f32], b: &[f32], c: &[f32],
    m: usize, n: usize, k: usize,
    nfb: i32,
    a_min_exp: i32, b_min_exp: i32, c_min_exp: i32,
    out_mantissa_bits: i32,
    out: &mut [f32],
) {
    assert!(n % 2 == 0, "NEON path requires even n");
    assert!(m * k <= MAX_ELEMS);
    assert!(k * n <= MAX_ELEMS);

    let mut a_sig = [0i64; MAX_ELEMS];
    let mut a_exp = [0i32; MAX_ELEMS];
    let mut a_sp = [false; MAX_ELEMS];
    for i in 0..m * k {
        let (s, e, sp) = decompose(a[i], a_min_exp, nfb);
        a_sig[i] = s;
        a_exp[i] = e;
        a_sp[i] = sp;
    }
    let mut b_sig = [0i64; MAX_ELEMS];
    let mut b_exp = [0i32; MAX_ELEMS];
    let mut b_sp = [false; MAX_ELEMS];
    for i in 0..k * n {
        let (s, e, sp) = decompose(b[i], b_min_exp, nfb);
        b_sig[i] = s;
        b_exp[i] = e;
        b_sp[i] = sp;
    }

    for i in 0..m {
        let mut jp = 0;
        while jp + 1 < n {
            let (c0_sig, c0_exp, c0_sp) = decompose(c[i * n + jp], c_min_exp, nfb);
            let (c1_sig, c1_exp, c1_sp) = decompose(c[i * n + jp + 1], c_min_exp, nfb);

            let mut any_special = c0_sp | c1_sp;
            let mut fp0 = c[i * n + jp] as f64;
            let mut fp1 = c[i * n + jp + 1] as f64;

            // Stack buffers for per-output products; k ≤ 128 in practice.
            let mut ps0 = [0i64; 128];
            let mut ps1 = [0i64; 128];
            let mut pe0 = [0i32; 128];
            let mut pe1 = [0i32; 128];
            let mut max_e0 = c0_exp;
            let mut max_e1 = c1_exp;

            for l in 0..k {
                let a_s = a_sig[i * k + l];
                let a_e = a_exp[i * k + l];
                let b0_s = b_sig[l * n + jp];
                let b1_s = b_sig[l * n + jp + 1];
                let b0_e = b_exp[l * n + jp];
                let b1_e = b_exp[l * n + jp + 1];
                any_special |= a_sp[i * k + l] | b_sp[l * n + jp] | b_sp[l * n + jp + 1];

                fp0 += (a[i * k + l] as f64) * (b[l * n + jp] as f64);
                fp1 += (a[i * k + l] as f64) * (b[l * n + jp + 1] as f64);

                ps0[l] = a_s * b0_s;
                ps1[l] = a_s * b1_s;
                let pe0l = a_e + b0_e;
                let pe1l = a_e + b1_e;
                pe0[l] = pe0l;
                pe1[l] = pe1l;
                if pe0l > max_e0 { max_e0 = pe0l; }
                if pe1l > max_e1 { max_e1 = pe1l; }
            }

            let (r0, r1) = if any_special || !fp0.is_finite() || !fp1.is_finite() {
                (fp0, fp1)
            } else {
                // Start with C term aligned.
                let c_sig_v = neon::pack2_s64(c0_sig, c1_sig);
                let c_shift_v = neon::pack2_s64(
                    (max_e0 - c0_exp) as i64,
                    (max_e1 - c1_exp) as i64,
                );
                let mut acc_v = neon::trunc_shr_v(c_sig_v, c_shift_v);

                for l in 0..k {
                    let sig_v = neon::pack2_s64(ps0[l], ps1[l]);
                    let shift_v = neon::pack2_s64(
                        (max_e0 - pe0[l] + nfb) as i64,
                        (max_e1 - pe1[l] + nfb) as i64,
                    );
                    acc_v = core::arch::aarch64::vaddq_s64(acc_v, neon::trunc_shr_v(sig_v, shift_v));
                }

                let (acc0, acc1) = neon::unpack2_s64(acc_v);
                (
                    (acc0 as f64) * fast_pow2(max_e0 - nfb),
                    (acc1 as f64) * fast_pow2(max_e1 - nfb),
                )
            };

            out[i * n + jp] = normalize_f32(r0, out_mantissa_bits);
            out[i * n + jp + 1] = normalize_f32(r1, out_mantissa_bits);
            jp += 2;
        }
        // n is always even for supported tiles (max n = 8); unreachable tail.
        debug_assert_eq!(jp, n);
    }
}

/// Per-tile dispatch to the NEON kernel (or scalar on non-aarch64).
#[inline]
#[allow(clippy::too_many_arguments)]
fn run_one_tile_simd(
    a_slice: &[f32], b_slice: &[f32], c_slice: &[f32],
    m: usize, n: usize, k: usize,
    nfb: i32, a_min_exp: i32, b_min_exp: i32, c_min_exp: i32,
    out_mantissa_bits: i32, split_k: bool,
    out_slice: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        if n % 2 == 0 {
            unsafe {
                if split_k {
                    let half = k / 2;
                    let mut mid = [0f32; MAX_ELEMS];
                    let mut a_half = [0f32; MAX_ELEMS];
                    let mut a_half2 = [0f32; MAX_ELEMS];
                    for i in 0..m {
                        a_half[i * half..(i + 1) * half]
                            .copy_from_slice(&a_slice[i * k..i * k + half]);
                        a_half2[i * half..(i + 1) * half]
                            .copy_from_slice(&a_slice[i * k + half..(i + 1) * k]);
                    }
                    fused_mma_step_int_neon(
                        &a_half[..m * half], &b_slice[..half * n], c_slice,
                        m, n, half,
                        nfb, a_min_exp, b_min_exp, c_min_exp,
                        out_mantissa_bits,
                        &mut mid[..m * n],
                    );
                    fused_mma_step_int_neon(
                        &a_half2[..m * half], &b_slice[half * n..], &mid[..m * n],
                        m, n, half,
                        nfb, a_min_exp, b_min_exp, F32_MIN_EXP,
                        out_mantissa_bits,
                        out_slice,
                    );
                } else {
                    fused_mma_step_int_neon(
                        a_slice, b_slice, c_slice, m, n, k,
                        nfb, a_min_exp, b_min_exp, c_min_exp,
                        out_mantissa_bits, out_slice,
                    );
                }
            }
            return;
        }
    }
    // Fallback: scalar.
    run_one_tile(
        a_slice, b_slice, c_slice, m, n, k,
        nfb, a_min_exp, b_min_exp, c_min_exp,
        out_mantissa_bits, split_k, out_slice,
    );
}

#[pyfunction]
#[pyo3(signature = (a, b, c, nfb, a_min_exp, b_min_exp, c_min_exp, out_mantissa_bits, split_k))]
#[allow(clippy::too_many_arguments)]
fn mma_f32_out_batched_simd<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f32>,
    b: PyReadonlyArray3<'py, f32>,
    c: PyReadonlyArray3<'py, f32>,
    nfb: i32,
    a_min_exp: i32,
    b_min_exp: i32,
    c_min_exp: i32,
    out_mantissa_bits: i32,
    split_k: bool,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let a_v = a.as_array();
    let b_v = b.as_array();
    let c_v = c.as_array();
    let batch = a_v.shape()[0];
    let m = a_v.shape()[1];
    let k = a_v.shape()[2];
    let n = b_v.shape()[2];
    assert_eq!(b_v.shape(), &[batch, k, n]);
    assert_eq!(c_v.shape(), &[batch, m, n]);
    assert!(m * k <= MAX_ELEMS && k * n <= MAX_ELEMS && m * n <= MAX_ELEMS);

    let a_owned: Vec<f32>;
    let a_slice: &[f32] = match a_v.as_slice() {
        Some(s) => s,
        None => { a_owned = a_v.iter().copied().collect(); &a_owned }
    };
    let b_owned: Vec<f32>;
    let b_slice: &[f32] = match b_v.as_slice() {
        Some(s) => s,
        None => { b_owned = b_v.iter().copied().collect(); &b_owned }
    };
    let c_owned: Vec<f32>;
    let c_slice: &[f32] = match c_v.as_slice() {
        Some(s) => s,
        None => { c_owned = c_v.iter().copied().collect(); &c_owned }
    };

    let mut out = vec![0.0f32; batch * m * n];
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    for bi in 0..batch {
        let a_i = &a_slice[bi * a_stride..(bi + 1) * a_stride];
        let b_i = &b_slice[bi * b_stride..(bi + 1) * b_stride];
        let c_i = &c_slice[bi * c_stride..(bi + 1) * c_stride];
        let out_i = &mut out[bi * c_stride..(bi + 1) * c_stride];
        run_one_tile_simd(
            a_i, b_i, c_i, m, n, k,
            nfb, a_min_exp, b_min_exp, c_min_exp,
            out_mantissa_bits, split_k, out_i,
        );
    }

    let arr = ndarray::Array3::from_shape_vec((batch, m, n), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

// ─────────────────────────────────────────────────────────────────────
// Rayon variant (scalar kernel, parallel over the batch).
//
// Phase 2.4: deliberately wraps `run_one_tile` (scalar, pre-SIMD) so
// the speedup is attributable to threading alone. Phase 2.5 adds an
// analogous `_simd_rayon` entry that wraps `run_one_tile_simd`.
//
// Release the GIL during the parallel section (py.allow_threads) so
// other Python threads can progress; our inner work is pure Rust.
// ─────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────
// N1 — Generic kernel with f16 output (RNE to 10 mantissa bits).
//
// Same integer pipeline as `fused_mma_step_int`; the only difference is
// the output normalize. Returns u16 (raw f16 bits) so the Python wrapper
// can view-cast to torch.float16 without losing the NaN pattern.
// ─────────────────────────────────────────────────────────────────────

#[inline]
#[allow(clippy::too_many_arguments)]
fn run_one_tile_f16_out(
    a: &[f32], b: &[f32], c: &[f32],
    m: usize, n: usize, k: usize,
    nfb: i32,
    a_min_exp: i32, b_min_exp: i32, c_min_exp: i32,
    split_k: bool,
    out: &mut [u16],
) {
    // This is simplest implemented as "run f32-output kernel into a scratch
    // buffer, then convert each element to f16 via a final re-decomp". But
    // that's two roundings which is wrong. Instead: mirror fused_mma_step_int
    // inline, but with f16 normalize at the end (and when split-K is on,
    // the mid pass also uses f16-bit output then re-read as f16→f64).
    //
    // For split-K, the oracle's intermediate C (= mid) IS in f16 (since
    // output_type="f16" means each step produces f16, and that f16 is the
    // next step's C). That's what mmasim/simulator/nv_ptx.py:96 shows
    // (D[i][j] = sum, where sum is f16 after nv_fused_dot_add).

    if split_k {
        let half = k / 2;
        let mut mid_bits = vec![0u16; m * n];
        run_one_tile_f16_out_single(
            a, b, c, m, n, half, nfb,
            a_min_exp, b_min_exp, c_min_exp,
            true, // A is A[:, :half]
            &mut mid_bits,
        );
        // Second pass: C is mid (as f16).
        let mid_f32: Vec<f32> = mid_bits.iter()
            .map(|&b| f16_bits_to_f32(b))
            .collect();
        run_one_tile_f16_out_single(
            a, b, c, m, n, half, nfb,
            a_min_exp, b_min_exp, -14, // mid is f16 → min_exp = -14
            false, // A is A[:, half:]
            out,
        );
        // Wait — we need to pass mid_f32 as C in the second call. Refactor:
        let _ = mid_f32;
        unreachable!("see run_one_tile_f16_out_single below — split-K needs explicit wiring");
    } else {
        run_one_tile_f16_out_single(
            a, b, c, m, n, k, nfb,
            a_min_exp, b_min_exp, c_min_exp,
            false, // a_is_first_half: irrelevant when not split
            out,
        );
    }
}

#[inline]
fn f16_bits_to_f32(bits: u16) -> f32 {
    // Manual f16 → f32 conversion (exact).
    let sign = ((bits >> 15) & 1) as u32;
    let biased_e = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    let f32_bits = if biased_e == 0 && mant == 0 {
        sign << 31
    } else if biased_e == 0x1F {
        // Inf or NaN
        if mant == 0 {
            (sign << 31) | 0x7F800000
        } else {
            // Preserve NaN payload (align 10-bit mant to 23-bit f32 mant).
            (sign << 31) | 0x7F800000 | (mant << 13)
        }
    } else if biased_e == 0 {
        // Subnormal f16
        let mut e: i32 = -14;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        let new_biased = (e + 127) as u32;
        let new_mant = (m & 0x3FF) << 13;
        (sign << 31) | (new_biased << 23) | new_mant
    } else {
        // Normal f16 → normal f32
        let new_biased = (biased_e as i32 - 15 + 127) as u32;
        (sign << 31) | (new_biased << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// Single-pass f16-output kernel. `a_is_first_half` is a hint for split-K
/// slicing (caller responsibility to pass correct A/B pointers).
#[inline]
#[allow(clippy::too_many_arguments)]
fn run_one_tile_f16_out_single(
    a: &[f32], b: &[f32], c: &[f32],
    m: usize, n: usize, k: usize,
    nfb: i32,
    a_min_exp: i32, b_min_exp: i32, c_min_exp: i32,
    _a_is_first_half: bool,
    out: &mut [u16],
) {
    // Allocate on heap to avoid stack size issues for arbitrary shapes.
    let mut a_sig = vec![0i64; m * k];
    let mut a_exp = vec![0i32; m * k];
    for i in 0..m * k {
        let (s, e, _) = decompose(a[i], a_min_exp, nfb);
        a_sig[i] = s;
        a_exp[i] = e;
    }
    let mut b_sig = vec![0i64; k * n];
    let mut b_exp = vec![0i32; k * n];
    for i in 0..k * n {
        let (s, e, _) = decompose(b[i], b_min_exp, nfb);
        b_sig[i] = s;
        b_exp[i] = e;
    }

    for i in 0..m {
        for j in 0..n {
            let (c_sig_ij, c_exp_ij, _) = decompose(c[i * n + j], c_min_exp, nfb);
            let mut fp_sum = c[i * n + j] as f64;

            let mut prod_sig = [0i64; 128];
            let mut prod_exp = [0i32; 128];
            let mut max_e = c_exp_ij;
            assert!(k <= 128);

            for l in 0..k {
                let a_s = a_sig[i * k + l];
                let a_e = a_exp[i * k + l];
                let b_s = b_sig[l * n + j];
                let b_e = b_exp[l * n + j];
                fp_sum += (a[i * k + l] as f64) * (b[l * n + j] as f64);
                let ps = a_s * b_s;
                let pe = a_e + b_e;
                prod_sig[l] = ps;
                prod_exp[l] = pe;
                if pe > max_e { max_e = pe; }
            }

            let result_f64 = if !fp_sum.is_finite() {
                fp_sum
            } else {
                let mut acc: i64 = trunc_shr(c_sig_ij, max_e - c_exp_ij);
                for l in 0..k {
                    let shift = max_e - prod_exp[l] + nfb;
                    acc += trunc_shr(prod_sig[l], shift);
                }
                (acc as f64) * fast_pow2(max_e - nfb)
            };

            out[i * n + j] = normalize_f16_rne_bits(result_f64);
        }
    }
}

#[pyfunction]
#[pyo3(signature = (a, b, c, nfb, a_min_exp, b_min_exp, c_min_exp, split_k))]
#[allow(clippy::too_many_arguments)]
fn mma_f16_out_batched_rayon<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f32>,
    b: PyReadonlyArray3<'py, f32>,
    c: PyReadonlyArray3<'py, f32>,
    nfb: i32,
    a_min_exp: i32,
    b_min_exp: i32,
    c_min_exp: i32,
    split_k: bool,
) -> PyResult<Bound<'py, PyArray3<u16>>> {
    let a_v = a.as_array();
    let b_v = b.as_array();
    let c_v = c.as_array();
    let batch = a_v.shape()[0];
    let m = a_v.shape()[1];
    let k = a_v.shape()[2];
    let n = b_v.shape()[2];
    assert_eq!(b_v.shape(), &[batch, k, n]);
    assert_eq!(c_v.shape(), &[batch, m, n]);

    let a_owned: Vec<f32>;
    let a_slice: &[f32] = match a_v.as_slice() {
        Some(s) => s,
        None => { a_owned = a_v.iter().copied().collect(); &a_owned }
    };
    let b_owned: Vec<f32>;
    let b_slice: &[f32] = match b_v.as_slice() {
        Some(s) => s,
        None => { b_owned = b_v.iter().copied().collect(); &b_owned }
    };
    let c_owned: Vec<f32>;
    let c_slice: &[f32] = match c_v.as_slice() {
        Some(s) => s,
        None => { c_owned = c_v.iter().copied().collect(); &c_owned }
    };

    let mut out = vec![0u16; batch * m * n];
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    py.allow_threads(|| {
        out.par_chunks_mut(c_stride)
            .zip(a_slice.par_chunks(a_stride))
            .zip(b_slice.par_chunks(b_stride))
            .zip(c_slice.par_chunks(c_stride))
            .for_each(|(((o, a), b), c)| {
                if split_k {
                    // Two-pass: first half writes mid (f16 bits), then
                    // converts back to f32 for the second half's C.
                    let half = k / 2;
                    let mut mid = vec![0u16; m * n];

                    // Split A into two contiguous halves.
                    let mut a1 = vec![0f32; m * half];
                    let mut a2 = vec![0f32; m * half];
                    for i in 0..m {
                        a1[i * half..(i + 1) * half]
                            .copy_from_slice(&a[i * k..i * k + half]);
                        a2[i * half..(i + 1) * half]
                            .copy_from_slice(&a[i * k + half..(i + 1) * k]);
                    }

                    run_one_tile_f16_out_single(
                        &a1, &b[..half * n], c, m, n, half,
                        nfb, a_min_exp, b_min_exp, c_min_exp,
                        false, &mut mid,
                    );
                    // Convert mid (f16 bits) → f32 for second pass's C.
                    let mid_f32: Vec<f32> = mid.iter().map(|&b| f16_bits_to_f32(b)).collect();
                    run_one_tile_f16_out_single(
                        &a2, &b[half * n..], &mid_f32, m, n, half,
                        nfb, a_min_exp, b_min_exp, -14,
                        false, o,
                    );
                } else {
                    run_one_tile_f16_out_single(
                        a, b, c, m, n, k,
                        nfb, a_min_exp, b_min_exp, c_min_exp,
                        false, o,
                    );
                }
            });
    });

    let arr = ndarray::Array3::from_shape_vec((batch, m, n), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

// Silence the unreachable-macro-using placeholder.
#[allow(dead_code)]
fn _f16_placeholder_touch() {
    let _ = run_one_tile_f16_out;
}

// ─────────────────────────────────────────────────────────────────────
// M4 — Heap-allocated generic kernel for large tiles (wgmma, tcgen05mma).
//
// Wgmma tiles can be up to m=64, n=256, k=32, so m*k and k*n exceed
// MAX_ELEMS=256. This variant uses Vec<i64> for the per-tile decompose
// buffers. Vec::with_capacity + unsafe set_len is used to skip
// zero-initialization (buffers are immediately overwritten).
// Same kernel math as fused_mma_step_int.
// ─────────────────────────────────────────────────────────────────────

#[inline]
#[allow(clippy::too_many_arguments)]
fn run_one_tile_heap(
    a: &[f32], b: &[f32], c: &[f32],
    m: usize, n: usize, k: usize,
    nfb: i32,
    a_min_exp: i32, b_min_exp: i32, c_min_exp: i32,
    out_mantissa_bits: i32,
    out: &mut [f32],
) {
    let mut a_sig = vec![0i64; m * k];
    let mut a_exp = vec![0i32; m * k];
    for idx in 0..m * k {
        let (s, e, _) = decompose(a[idx], a_min_exp, nfb);
        a_sig[idx] = s;
        a_exp[idx] = e;
    }
    let mut b_sig = vec![0i64; k * n];
    let mut b_exp = vec![0i32; k * n];
    for idx in 0..k * n {
        let (s, e, _) = decompose(b[idx], b_min_exp, nfb);
        b_sig[idx] = s;
        b_exp[idx] = e;
    }

    // Per-output k-loop buffers stay on the stack; k ≤ 32 for wgmma.
    for i in 0..m {
        for j in 0..n {
            let (c_sig_ij, c_exp_ij, _) = decompose(c[i * n + j], c_min_exp, nfb);
            let mut fp_sum = c[i * n + j] as f64;

            let mut prod_sig = [0i64; 64];
            let mut prod_exp = [0i32; 64];
            let mut max_e = c_exp_ij;
            assert!(k <= 64, "wgmma k must be <= 64");

            for l in 0..k {
                let a_s = a_sig[i * k + l];
                let a_e = a_exp[i * k + l];
                let b_s = b_sig[l * n + j];
                let b_e = b_exp[l * n + j];

                fp_sum += (a[i * k + l] as f64) * (b[l * n + j] as f64);

                let ps = a_s * b_s;
                let pe = a_e + b_e;
                prod_sig[l] = ps;
                prod_exp[l] = pe;
                if pe > max_e { max_e = pe; }
            }

            let result_f64 = if !fp_sum.is_finite() {
                fp_sum
            } else {
                let mut acc: i64 = trunc_shr(c_sig_ij, max_e - c_exp_ij);
                for l in 0..k {
                    let shift = max_e - prod_exp[l] + nfb;
                    acc += trunc_shr(prod_sig[l], shift);
                }
                (acc as f64) * fast_pow2(max_e - nfb)
            };

            out[i * n + j] = normalize_f32(result_f64, out_mantissa_bits);
        }
    }
}

#[pyfunction]
#[pyo3(signature = (a, b, c, nfb, a_min_exp, b_min_exp, c_min_exp, out_mantissa_bits))]
#[allow(clippy::too_many_arguments)]
fn mma_f32_out_wgmma_batched_rayon<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f32>,
    b: PyReadonlyArray3<'py, f32>,
    c: PyReadonlyArray3<'py, f32>,
    nfb: i32,
    a_min_exp: i32,
    b_min_exp: i32,
    c_min_exp: i32,
    out_mantissa_bits: i32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let a_v = a.as_array();
    let b_v = b.as_array();
    let c_v = c.as_array();
    let batch = a_v.shape()[0];
    let m = a_v.shape()[1];
    let k = a_v.shape()[2];
    let n = b_v.shape()[2];
    assert_eq!(b_v.shape(), &[batch, k, n]);
    assert_eq!(c_v.shape(), &[batch, m, n]);

    let a_owned: Vec<f32>;
    let a_slice: &[f32] = match a_v.as_slice() {
        Some(s) => s,
        None => { a_owned = a_v.iter().copied().collect(); &a_owned }
    };
    let b_owned: Vec<f32>;
    let b_slice: &[f32] = match b_v.as_slice() {
        Some(s) => s,
        None => { b_owned = b_v.iter().copied().collect(); &b_owned }
    };
    let c_owned: Vec<f32>;
    let c_slice: &[f32] = match c_v.as_slice() {
        Some(s) => s,
        None => { c_owned = c_v.iter().copied().collect(); &c_owned }
    };

    let mut out = vec![0f32; batch * m * n];
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    py.allow_threads(|| {
        out.par_chunks_mut(c_stride)
            .zip(a_slice.par_chunks(a_stride))
            .zip(b_slice.par_chunks(b_stride))
            .zip(c_slice.par_chunks(c_stride))
            .for_each(|(((o, a), b), c)| {
                run_one_tile_heap(
                    a, b, c, m, n, k,
                    nfb, a_min_exp, b_min_exp, c_min_exp,
                    out_mantissa_bits, o,
                );
            });
    });

    let arr = ndarray::Array3::from_shape_vec((batch, m, n), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (a, b, c, nfb, a_min_exp, b_min_exp, c_min_exp, out_mantissa_bits, split_k))]
#[allow(clippy::too_many_arguments)]
fn mma_f32_out_batched_rayon<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f32>,
    b: PyReadonlyArray3<'py, f32>,
    c: PyReadonlyArray3<'py, f32>,
    nfb: i32,
    a_min_exp: i32,
    b_min_exp: i32,
    c_min_exp: i32,
    out_mantissa_bits: i32,
    split_k: bool,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let a_v = a.as_array();
    let b_v = b.as_array();
    let c_v = c.as_array();
    let batch = a_v.shape()[0];
    let m = a_v.shape()[1];
    let k = a_v.shape()[2];
    let n = b_v.shape()[2];
    assert_eq!(b_v.shape(), &[batch, k, n]);
    assert_eq!(c_v.shape(), &[batch, m, n]);
    assert!(m * k <= MAX_ELEMS && k * n <= MAX_ELEMS && m * n <= MAX_ELEMS);

    // Collect into owning Vecs if needed so rayon can borrow across threads.
    let a_owned: Vec<f32>;
    let a_slice: &[f32] = match a_v.as_slice() {
        Some(s) => s,
        None => { a_owned = a_v.iter().copied().collect(); &a_owned }
    };
    let b_owned: Vec<f32>;
    let b_slice: &[f32] = match b_v.as_slice() {
        Some(s) => s,
        None => { b_owned = b_v.iter().copied().collect(); &b_owned }
    };
    let c_owned: Vec<f32>;
    let c_slice: &[f32] = match c_v.as_slice() {
        Some(s) => s,
        None => { c_owned = c_v.iter().copied().collect(); &c_owned }
    };

    let mut out = vec![0.0f32; batch * m * n];
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    py.allow_threads(|| {
        out.par_chunks_mut(c_stride)
            .zip(a_slice.par_chunks(a_stride))
            .zip(b_slice.par_chunks(b_stride))
            .zip(c_slice.par_chunks(c_stride))
            .for_each(|(((out_i, a_i), b_i), c_i)| {
                run_one_tile(
                    a_i, b_i, c_i, m, n, k,
                    nfb, a_min_exp, b_min_exp, c_min_exp,
                    out_mantissa_bits, split_k, out_i,
                );
            });
    });

    let arr = ndarray::Array3::from_shape_vec((batch, m, n), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

// ─────────────────────────────────────────────────────────────────────
// Phase 3.0 / 3.1 — specialized kernels.
//
// Phase 3.0 hand-wrote one kernel for ampere-f16. Phase 3.1 generalizes:
// one const-generic `step_narrow<const M, N, K, NFB, A_MIN, B_MIN>` and
// one `step_f32<const M, N, K, NFB>` covering the five supported tile
// types. Every supported (arch, qualifier) gets its own monomorphization
// via a dedicated PyO3 entry.
//
// Kept as `ampere_f16` initially for compatibility with the Phase 3.0
// entry name — renamed to `specialized` in 3.1.
// ─────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
mod specialized {
    use core::arch::aarch64::*;

    use super::{fast_pow2, normalize_f32};

    /// Stack buffer cap for pre-decomposed operands. Covers up to
    /// M*K = 16*16 = 256 for our current tile set.
    pub(super) const MAX: usize = 256;

    /// Decompose an f32 that holds a narrow value (f16 or bf16 cast up).
    /// Neither source type can produce an f32 subnormal via cast, so we
    /// skip that branch. MIN_EXP is the target dtype's flush cutoff: -14
    /// for f16, -126 for bf16. NFB is the accumulator fractional-bit
    /// count (typically 23 or 24); int_sig is returned at scale 2^NFB.
    /// NaN/Inf → garbage int (fp64 check catches).
    #[inline(always)]
    fn decompose_narrow<const MIN_EXP: i32, const NFB: i32>(v: f32) -> (i64, i32) {
        let bits = v.to_bits();
        let sign = (bits >> 31) & 1 != 0;
        let biased_e = ((bits >> 23) & 0xFF) as i32;
        let mant = (bits & 0x7F_FFFF) as i64;
        if biased_e == 0 {
            return (0, -126); // zero quirk
        }
        // Scale to 2^NFB. NFB >= 23 for supported ISA/arch combos; shift up
        // by NFB-23 from the native f32 mantissa scale (2^23).
        let sig_qnfb = ((1i64 << 23) + mant) << (NFB - 23);
        let exp = biased_e - 127;
        let (int_sig, final_exp) = if exp < MIN_EXP {
            let shift = (MIN_EXP - exp) as u32;
            let s = sig_qnfb >> shift;
            if s == 0 { (0, -126) } else { (s, MIN_EXP) }
        } else {
            (sig_qnfb, exp)
        };
        let signed = if sign { -int_sig } else { int_sig };
        (signed, final_exp)
    }

    /// Decompose a general f32 (tf32 inputs and the C addend). Handles
    /// f32 subnormals via leading-zero normalization. NFB parameterizes
    /// the accumulator scale.
    #[inline(always)]
    fn decompose_f32_src<const NFB: i32>(v: f32) -> (i64, i32) {
        let bits = v.to_bits();
        let sign = (bits >> 31) & 1 != 0;
        let biased_e = ((bits >> 23) & 0xFF) as i32;
        let mant = (bits & 0x7F_FFFF) as i64;
        if biased_e == 0 && mant == 0 {
            return (0, -126);
        }
        let (sig_q23, exp) = if biased_e == 0 {
            let top = 63 - (mant as u64).leading_zeros() as i32;
            (mant << (23 - top), top - 149)
        } else {
            ((1i64 << 23) + mant, biased_e - 127)
        };
        let mut int_sig = sig_q23 << (NFB - 23);
        let mut exp = exp;
        if exp < -126 {
            let shift = (-126 - exp) as u32;
            int_sig >>= shift;
            exp = -126;
            if int_sig == 0 { exp = -126; }
        }
        let signed = if sign { -int_sig } else { int_sig };
        (signed, exp)
    }

    #[inline(always)]
    unsafe fn trunc_shr_v(sig: int64x2_t, shift: int64x2_t) -> int64x2_t {
        let sign_mask = vshrq_n_s64(sig, 63);
        let abs_sig = vsubq_s64(veorq_s64(sig, sign_mask), sign_mask);
        let shifted = vshlq_s64(abs_sig, vnegq_s64(shift));
        vsubq_s64(veorq_s64(shifted, sign_mask), sign_mask)
    }

    #[inline(always)]
    unsafe fn pack2(a: i64, b: i64) -> int64x2_t {
        let arr = [a, b];
        vld1q_s64(arr.as_ptr())
    }

    /// Per-output inner kernel. Operates on already-decomposed (sig, exp)
    /// buffers plus the raw f32 arrays for the NaN/Inf pre-check. All
    /// loop bounds are const-generic so the compiler fully unrolls k.
    #[target_feature(enable = "neon")]
    unsafe fn inner<const M: usize, const N: usize, const K: usize, const NFB: i32>(
        a_sig: &[i64; MAX], a_exp: &[i32; MAX],
        b_sig: &[i64; MAX], b_exp: &[i32; MAX],
        a: &[f32], b: &[f32], c: &[f32], out: &mut [f32],
    ) {
        for i in 0..M {
            let mut jp = 0;
            while jp + 1 < N {
                let (c0_sig, c0_exp) = decompose_f32_src::<NFB>(c[i * N + jp]);
                let (c1_sig, c1_exp) = decompose_f32_src::<NFB>(c[i * N + jp + 1]);

                let mut fp0 = c[i * N + jp] as f64;
                let mut fp1 = c[i * N + jp + 1] as f64;

                let mut ps0 = [0i64; 128];
                let mut ps1 = [0i64; 128];
                let mut pe0 = [0i32; 128];
                let mut pe1 = [0i32; 128];
                let mut max_e0 = c0_exp;
                let mut max_e1 = c1_exp;

                for l in 0..K {
                    let a_s = a_sig[i * K + l];
                    let a_e = a_exp[i * K + l];
                    let b_s0 = b_sig[l * N + jp];
                    let b_s1 = b_sig[l * N + jp + 1];
                    let b_e0 = b_exp[l * N + jp];
                    let b_e1 = b_exp[l * N + jp + 1];
                    fp0 += (a[i * K + l] as f64) * (b[l * N + jp] as f64);
                    fp1 += (a[i * K + l] as f64) * (b[l * N + jp + 1] as f64);
                    ps0[l] = a_s * b_s0;
                    ps1[l] = a_s * b_s1;
                    let e0 = a_e + b_e0;
                    let e1 = a_e + b_e1;
                    pe0[l] = e0;
                    pe1[l] = e1;
                    if e0 > max_e0 { max_e0 = e0; }
                    if e1 > max_e1 { max_e1 = e1; }
                }

                let (r0, r1) = if !fp0.is_finite() || !fp1.is_finite() {
                    (fp0, fp1)
                } else {
                    let mut acc_v = trunc_shr_v(
                        pack2(c0_sig, c1_sig),
                        pack2((max_e0 - c0_exp) as i64, (max_e1 - c1_exp) as i64),
                    );
                    for l in 0..K {
                        let sig_v = pack2(ps0[l], ps1[l]);
                        let shift_v = pack2(
                            (max_e0 - pe0[l] + NFB) as i64,
                            (max_e1 - pe1[l] + NFB) as i64,
                        );
                        acc_v = vaddq_s64(acc_v, trunc_shr_v(sig_v, shift_v));
                    }
                    let acc0 = vgetq_lane_s64::<0>(acc_v);
                    let acc1 = vgetq_lane_s64::<1>(acc_v);
                    (
                        (acc0 as f64) * fast_pow2(max_e0 - NFB),
                        (acc1 as f64) * fast_pow2(max_e1 - NFB),
                    )
                };

                out[i * N + jp] = normalize_f32(r0, 23);
                out[i * N + jp + 1] = normalize_f32(r1, 23);
                jp += 2;
            }
        }
    }

    /// Step for narrow (f16/bf16) inputs. Non-split-K path; callers do
    /// the split-K halving at the tile level.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn step_narrow<
        const M: usize, const N: usize, const K: usize, const NFB: i32,
        const A_MIN: i32, const B_MIN: i32,
    >(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
        let mut a_sig = [0i64; MAX];
        let mut a_exp = [0i32; MAX];
        for i in 0..M * K {
            let (s, e) = decompose_narrow::<A_MIN, NFB>(a[i]);
            a_sig[i] = s;
            a_exp[i] = e;
        }
        let mut b_sig = [0i64; MAX];
        let mut b_exp = [0i32; MAX];
        for i in 0..K * N {
            let (s, e) = decompose_narrow::<B_MIN, NFB>(b[i]);
            b_sig[i] = s;
            b_exp[i] = e;
        }
        inner::<M, N, K, NFB>(&a_sig, &a_exp, &b_sig, &b_exp, a, b, c, out);
    }

    /// Step for f32/tf32 inputs.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn step_f32<
        const M: usize, const N: usize, const K: usize, const NFB: i32,
    >(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
        let mut a_sig = [0i64; MAX];
        let mut a_exp = [0i32; MAX];
        for i in 0..M * K {
            let (s, e) = decompose_f32_src::<NFB>(a[i]);
            a_sig[i] = s;
            a_exp[i] = e;
        }
        let mut b_sig = [0i64; MAX];
        let mut b_exp = [0i32; MAX];
        for i in 0..K * N {
            let (s, e) = decompose_f32_src::<NFB>(b[i]);
            b_sig[i] = s;
            b_exp[i] = e;
        }
        inner::<M, N, K, NFB>(&a_sig, &a_exp, &b_sig, &b_exp, a, b, c, out);
    }
}

#[cfg(not(target_arch = "aarch64"))]
mod specialized {
    // Stub: no specialized NEON on non-aarch64. Callers fall back to the
    // generic simd-rayon path from the PyO3 entries themselves.
    pub(super) const MAX: usize = 256;
}

/// PyO3 wrapper for batched specialized calls. Takes a per-tile closure
/// that runs one MMA; internals handle shape check, zero-copy input
/// slicing, rayon-parallel dispatch, and output assembly.
fn run_batched_specialized<'py, F>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f32>,
    b: PyReadonlyArray3<'py, f32>,
    c: PyReadonlyArray3<'py, f32>,
    expected_m: usize, expected_n: usize, expected_k: usize,
    per_tile: F,
) -> PyResult<Bound<'py, PyArray3<f32>>>
where
    F: Fn(&[f32], &[f32], &[f32], &mut [f32]) + Sync,
{
    let a_v = a.as_array();
    let b_v = b.as_array();
    let c_v = c.as_array();
    let batch = a_v.shape()[0];
    assert_eq!(a_v.shape(), &[batch, expected_m, expected_k]);
    assert_eq!(b_v.shape(), &[batch, expected_k, expected_n]);
    assert_eq!(c_v.shape(), &[batch, expected_m, expected_n]);

    let a_owned: Vec<f32>;
    let a_slice: &[f32] = match a_v.as_slice() {
        Some(s) => s,
        None => { a_owned = a_v.iter().copied().collect(); &a_owned }
    };
    let b_owned: Vec<f32>;
    let b_slice: &[f32] = match b_v.as_slice() {
        Some(s) => s,
        None => { b_owned = b_v.iter().copied().collect(); &b_owned }
    };
    let c_owned: Vec<f32>;
    let c_slice: &[f32] = match c_v.as_slice() {
        Some(s) => s,
        None => { c_owned = c_v.iter().copied().collect(); &c_owned }
    };

    let a_stride = expected_m * expected_k;
    let b_stride = expected_k * expected_n;
    let c_stride = expected_m * expected_n;
    let mut out = vec![0f32; batch * c_stride];

    py.allow_threads(|| {
        out.par_chunks_mut(c_stride)
            .zip(a_slice.par_chunks(a_stride))
            .zip(b_slice.par_chunks(b_stride))
            .zip(c_slice.par_chunks(c_stride))
            .for_each(|(((o, a), b), c)| {
                per_tile(a, b, c, o);
            });
    });

    let arr = ndarray::Array3::from_shape_vec((batch, expected_m, expected_n), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

// ─── Per-instruction PyO3 entries ────────────────────────────────────
// Each binds the (M, N, K, NFB, A_MIN, B_MIN) constants, handles split-K
// explicitly (where applicable), and dispatches to the specialized kernel.
// On non-aarch64 we fall back to the scalar SIMD path via run_one_tile_simd.

#[pyfunction]
fn mma_spec_ampere_f16<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f32>,
    b: PyReadonlyArray3<'py, f32>,
    c: PyReadonlyArray3<'py, f32>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    // m16n8k16 f32.f16.f16.f32 — split-K.
    run_batched_specialized(py, a, b, c, 16, 8, 16, |a, b, c, out| {
        const M: usize = 16; const N: usize = 8; const K_HALF: usize = 8;
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let mut a1 = [0f32; specialized::MAX];
            let mut a2 = [0f32; specialized::MAX];
            for i in 0..M {
                a1[i * K_HALF..(i + 1) * K_HALF].copy_from_slice(&a[i * 16..i * 16 + 8]);
                a2[i * K_HALF..(i + 1) * K_HALF].copy_from_slice(&a[i * 16 + 8..(i + 1) * 16]);
            }
            let mut mid = [0f32; specialized::MAX];
            specialized::step_narrow::<M, N, K_HALF, 24, -14, -14>(
                &a1[..M * K_HALF], &b[..K_HALF * N], c, &mut mid[..M * N],
            );
            specialized::step_narrow::<M, N, K_HALF, 24, -14, -14>(
                &a2[..M * K_HALF], &b[K_HALF * N..], &mid[..M * N], out,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        { run_one_tile_simd(a, b, c, M, N, 16, 24, -14, -14, -126, 23, true, out); }
    })
}

#[pyfunction]
fn mma_spec_ampere_bf16<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f32>,
    b: PyReadonlyArray3<'py, f32>,
    c: PyReadonlyArray3<'py, f32>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    // m16n8k16 f32.bf16.bf16.f32 — split-K, bf16 min_exp=-126.
    run_batched_specialized(py, a, b, c, 16, 8, 16, |a, b, c, out| {
        const M: usize = 16; const N: usize = 8; const K_HALF: usize = 8;
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let mut a1 = [0f32; specialized::MAX];
            let mut a2 = [0f32; specialized::MAX];
            for i in 0..M {
                a1[i * K_HALF..(i + 1) * K_HALF].copy_from_slice(&a[i * 16..i * 16 + 8]);
                a2[i * K_HALF..(i + 1) * K_HALF].copy_from_slice(&a[i * 16 + 8..(i + 1) * 16]);
            }
            let mut mid = [0f32; specialized::MAX];
            specialized::step_narrow::<M, N, K_HALF, 24, -126, -126>(
                &a1[..M * K_HALF], &b[..K_HALF * N], c, &mut mid[..M * N],
            );
            specialized::step_narrow::<M, N, K_HALF, 24, -126, -126>(
                &a2[..M * K_HALF], &b[K_HALF * N..], &mid[..M * N], out,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        { run_one_tile_simd(a, b, c, M, N, 16, 24, -126, -126, -126, 23, true, out); }
    })
}

#[pyfunction]
fn mma_spec_ampere_tf32<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f32>,
    b: PyReadonlyArray3<'py, f32>,
    c: PyReadonlyArray3<'py, f32>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    // m16n8k8 f32.tf32.tf32.f32 — SPLIT-K on Ampere (oracle sets
    // is_split_k=True when k==8 && a_type==torch.float32). Two passes
    // of K_HALF=4. f32 source (subnormal path live).
    run_batched_specialized(py, a, b, c, 16, 8, 8, |a, b, c, out| {
        const M: usize = 16; const N: usize = 8; const K_HALF: usize = 4;
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let mut a1 = [0f32; specialized::MAX];
            let mut a2 = [0f32; specialized::MAX];
            for i in 0..M {
                a1[i * K_HALF..(i + 1) * K_HALF].copy_from_slice(&a[i * 8..i * 8 + 4]);
                a2[i * K_HALF..(i + 1) * K_HALF].copy_from_slice(&a[i * 8 + 4..(i + 1) * 8]);
            }
            let mut mid = [0f32; specialized::MAX];
            specialized::step_f32::<M, N, K_HALF, 24>(
                &a1[..M * K_HALF], &b[..K_HALF * N], c, &mut mid[..M * N],
            );
            specialized::step_f32::<M, N, K_HALF, 24>(
                &a2[..M * K_HALF], &b[K_HALF * N..], &mid[..M * N], out,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        { run_one_tile_simd(a, b, c, 16, 8, 8, 24, -126, -126, -126, 23, true, out); }
    })
}

#[pyfunction]
fn mma_spec_turing_f16<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f32>,
    b: PyReadonlyArray3<'py, f32>,
    c: PyReadonlyArray3<'py, f32>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    // m16n8k8 f32.f16.f16.f32 — no split-K.
    run_batched_specialized(py, a, b, c, 16, 8, 8, |a, b, c, out| {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            specialized::step_narrow::<16, 8, 8, 24, -14, -14>(a, b, c, out);
        }
        #[cfg(not(target_arch = "aarch64"))]
        { run_one_tile_simd(a, b, c, 16, 8, 8, 24, -14, -14, -126, 23, false, out); }
    })
}

#[pyfunction]
fn mma_spec_volta_f16<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f32>,
    b: PyReadonlyArray3<'py, f32>,
    c: PyReadonlyArray3<'py, f32>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    // m8n8k4 f32.f16.f16.f32 — no split-K, NFB=23 (Volta).
    run_batched_specialized(py, a, b, c, 8, 8, 4, |a, b, c, out| {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            specialized::step_narrow::<8, 8, 4, 23, -14, -14>(a, b, c, out);
        }
        #[cfg(not(target_arch = "aarch64"))]
        { run_one_tile_simd(a, b, c, 8, 8, 4, 23, -14, -14, -126, 23, false, out); }
    })
}

/// Phase 2.5: rayon + SIMD — parallel over the batch, NEON inner kernel.
#[pyfunction]
#[pyo3(signature = (a, b, c, nfb, a_min_exp, b_min_exp, c_min_exp, out_mantissa_bits, split_k))]
#[allow(clippy::too_many_arguments)]
fn mma_f32_out_batched_simd_rayon<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f32>,
    b: PyReadonlyArray3<'py, f32>,
    c: PyReadonlyArray3<'py, f32>,
    nfb: i32,
    a_min_exp: i32,
    b_min_exp: i32,
    c_min_exp: i32,
    out_mantissa_bits: i32,
    split_k: bool,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let a_v = a.as_array();
    let b_v = b.as_array();
    let c_v = c.as_array();
    let batch = a_v.shape()[0];
    let m = a_v.shape()[1];
    let k = a_v.shape()[2];
    let n = b_v.shape()[2];
    assert_eq!(b_v.shape(), &[batch, k, n]);
    assert_eq!(c_v.shape(), &[batch, m, n]);
    assert!(m * k <= MAX_ELEMS && k * n <= MAX_ELEMS && m * n <= MAX_ELEMS);

    let a_owned: Vec<f32>;
    let a_slice: &[f32] = match a_v.as_slice() {
        Some(s) => s,
        None => { a_owned = a_v.iter().copied().collect(); &a_owned }
    };
    let b_owned: Vec<f32>;
    let b_slice: &[f32] = match b_v.as_slice() {
        Some(s) => s,
        None => { b_owned = b_v.iter().copied().collect(); &b_owned }
    };
    let c_owned: Vec<f32>;
    let c_slice: &[f32] = match c_v.as_slice() {
        Some(s) => s,
        None => { c_owned = c_v.iter().copied().collect(); &c_owned }
    };

    let mut out = vec![0.0f32; batch * m * n];
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    py.allow_threads(|| {
        out.par_chunks_mut(c_stride)
            .zip(a_slice.par_chunks(a_stride))
            .zip(b_slice.par_chunks(b_stride))
            .zip(c_slice.par_chunks(c_stride))
            .for_each(|(((out_i, a_i), b_i), c_i)| {
                run_one_tile_simd(
                    a_i, b_i, c_i, m, n, k,
                    nfb, a_min_exp, b_min_exp, c_min_exp,
                    out_mantissa_bits, split_k, out_i,
                );
            });
    });

    let arr = ndarray::Array3::from_shape_vec((batch, m, n), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

#[pymodule]
fn fastmma_rust(_py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mma_f32_out, &m)?)?;
    m.add_function(wrap_pyfunction!(mma_f32_out_batched, &m)?)?;
    m.add_function(wrap_pyfunction!(mma_f32_out_batched_simd, &m)?)?;
    m.add_function(wrap_pyfunction!(mma_f32_out_batched_rayon, &m)?)?;
    m.add_function(wrap_pyfunction!(mma_f32_out_batched_simd_rayon, &m)?)?;
    m.add_function(wrap_pyfunction!(mma_spec_ampere_f16, &m)?)?;
    m.add_function(wrap_pyfunction!(mma_spec_ampere_bf16, &m)?)?;
    m.add_function(wrap_pyfunction!(mma_spec_ampere_tf32, &m)?)?;
    m.add_function(wrap_pyfunction!(mma_spec_turing_f16, &m)?)?;
    m.add_function(wrap_pyfunction!(mma_spec_volta_f16, &m)?)?;
    m.add_function(wrap_pyfunction!(mma_f64_batched_rayon, &m)?)?;
    m.add_function(wrap_pyfunction!(mma_f32_fma_batched_rayon, &m)?)?;
    m.add_function(wrap_pyfunction!(mma_f32_amd_pairwise_rayon, &m)?)?;
    m.add_function(wrap_pyfunction!(mma_f32_amd_fused_rd_rayon, &m)?)?;
    m.add_function(wrap_pyfunction!(mma_f32_out_wgmma_batched_rayon, &m)?)?;
    m.add_function(wrap_pyfunction!(mma_f16_out_batched_rayon, &m)?)?;
    m.add_function(wrap_pyfunction!(mma_f32_out_block_scale_k32_rayon, &m)?)?;
    m.add_function(wrap_pyfunction!(mma_f32_out_mxfp4_k64_rayon, &m)?)?;
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(mma_metal_ampere_f16, &m)?)?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// M1c — Block-scaled mxfp8 (k=32, per-tile scalar scales).
//
// Oracle [nv_ptx.py:122-131] dispatches to nv_fused_dot_add with
// scale_A[i, 0] and scale_B[0, j] applied per output. For ue8m0 scales
// (pure powers of 2) only the exponent is needed — sig is always 1.
// We pre-extract exponents row-by-row / col-by-col on the Python side
// and pass as i32 tables; the kernel adds them to each product's exp.
// ─────────────────────────────────────────────────────────────────────

/// Extract an ue8m0 scale's unbiased exponent from its f32 cast.
/// For ue8m0: v = 2^(raw - 127), so biased_e - 127 = raw - 127.
/// For general power-of-2 scales this returns the exponent directly.
#[inline(always)]
fn extract_pow2_exp(v: f32) -> i32 {
    let bits = v.to_bits();
    let biased_e = ((bits >> 23) & 0xFF) as i32;
    biased_e - 127
}

// Large stack buffer for block-scaled tiles (k can be 32 or 64).
// m*k = 16*64 = 1024; k*n = 64*8 = 512.
const BS_MAX: usize = 1024;

#[inline]
#[allow(clippy::too_many_arguments)]
fn run_one_tile_block_scale_k32(
    a: &[f32], b: &[f32], c: &[f32],
    scale_a_exp: &[i32], // length m
    scale_b_exp: &[i32], // length n
    m: usize, n: usize, k: usize,
    nfb: i32,
    a_min_exp: i32, b_min_exp: i32, c_min_exp: i32,
    out_mantissa_bits: i32,
    out: &mut [f32],
) {
    assert!(m * k <= BS_MAX);
    assert!(k * n <= BS_MAX);

    let mut a_sig = [0i64; BS_MAX];
    let mut a_exp = [0i32; BS_MAX];
    let mut a_sp = [false; BS_MAX];
    for i in 0..m * k {
        let (s, e, sp) = decompose(a[i], a_min_exp, nfb);
        a_sig[i] = s;
        a_exp[i] = e;
        a_sp[i] = sp;
    }
    let mut b_sig = [0i64; BS_MAX];
    let mut b_exp = [0i32; BS_MAX];
    let mut b_sp = [false; BS_MAX];
    for i in 0..k * n {
        let (s, e, sp) = decompose(b[i], b_min_exp, nfb);
        b_sig[i] = s;
        b_exp[i] = e;
        b_sp[i] = sp;
    }

    for i in 0..m {
        for j in 0..n {
            let (c_sig_ij, c_exp_ij, c_special) = decompose(c[i * n + j], c_min_exp, nfb);
            let mut any_special = c_special;
            let mut fp_sum = c[i * n + j] as f64;

            // Per-output scale exp offset applied to every product's exp.
            let scale_exp_sum = scale_a_exp[i] + scale_b_exp[j];
            let scale_f64 = fast_pow2(scale_exp_sum);
            // fp_sum needs the scale applied for the NaN/Inf pre-check:
            // products_f64 * scale_a * scale_b summed. Power-of-2 scale:
            // multiply each a*b by 2^(scale_exp_sum).

            let mut prod_sig = [0i64; 128];
            let mut prod_exp = [0i32; 128];
            let mut max_e = c_exp_ij;
            assert!(k <= 128);

            for l in 0..k {
                let a_s = a_sig[i * k + l];
                let a_e = a_exp[i * k + l];
                let b_s = b_sig[l * n + j];
                let b_e = b_exp[l * n + j];
                any_special |= a_sp[i * k + l] | b_sp[l * n + j];

                fp_sum += (a[i * k + l] as f64) * (b[l * n + j] as f64) * scale_f64;

                let ps = a_s * b_s;
                let pe = a_e + b_e + scale_exp_sum;
                prod_sig[l] = ps;
                prod_exp[l] = pe;
                if pe > max_e {
                    max_e = pe;
                }
            }

            let result_f64 = if any_special || !fp_sum.is_finite() {
                fp_sum
            } else {
                let mut acc: i64 = trunc_shr(c_sig_ij, max_e - c_exp_ij);
                for l in 0..k {
                    let shift = max_e - prod_exp[l] + nfb;
                    acc += trunc_shr(prod_sig[l], shift);
                }
                (acc as f64) * fast_pow2(max_e - nfb)
            };

            out[i * n + j] = normalize_f32(result_f64, out_mantissa_bits);
        }
    }
}

#[pyfunction]
#[pyo3(signature = (a, b, c, scale_a, scale_b, nfb, a_min_exp, b_min_exp, c_min_exp, out_mantissa_bits))]
#[allow(clippy::too_many_arguments)]
fn mma_f32_out_block_scale_k32_rayon<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f32>,  // (batch, m, k)
    b: PyReadonlyArray3<'py, f32>,  // (batch, k, n)
    c: PyReadonlyArray3<'py, f32>,  // (batch, m, n)
    scale_a: PyReadonlyArray3<'py, f32>, // (batch, m, 1)
    scale_b: PyReadonlyArray3<'py, f32>, // (batch, 1, n)
    nfb: i32,
    a_min_exp: i32,
    b_min_exp: i32,
    c_min_exp: i32,
    out_mantissa_bits: i32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let a_v = a.as_array();
    let b_v = b.as_array();
    let c_v = c.as_array();
    let sa_v = scale_a.as_array();
    let sb_v = scale_b.as_array();
    let batch = a_v.shape()[0];
    let m = a_v.shape()[1];
    let k = a_v.shape()[2];
    let n = b_v.shape()[2];
    assert_eq!(b_v.shape(), &[batch, k, n]);
    assert_eq!(c_v.shape(), &[batch, m, n]);
    assert_eq!(sa_v.shape(), &[batch, m, 1]);
    assert_eq!(sb_v.shape(), &[batch, 1, n]);

    // Pre-extract scale exponents into per-batch tables.
    let sa_flat: Vec<f32> = sa_v.iter().copied().collect();
    let sb_flat: Vec<f32> = sb_v.iter().copied().collect();
    let scale_a_exp: Vec<i32> = sa_flat.iter().map(|&v| extract_pow2_exp(v)).collect();
    let scale_b_exp: Vec<i32> = sb_flat.iter().map(|&v| extract_pow2_exp(v)).collect();

    let a_owned: Vec<f32>;
    let a_slice: &[f32] = match a_v.as_slice() {
        Some(s) => s,
        None => { a_owned = a_v.iter().copied().collect(); &a_owned }
    };
    let b_owned: Vec<f32>;
    let b_slice: &[f32] = match b_v.as_slice() {
        Some(s) => s,
        None => { b_owned = b_v.iter().copied().collect(); &b_owned }
    };
    let c_owned: Vec<f32>;
    let c_slice: &[f32] = match c_v.as_slice() {
        Some(s) => s,
        None => { c_owned = c_v.iter().copied().collect(); &c_owned }
    };

    let mut out = vec![0f32; batch * m * n];
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    py.allow_threads(|| {
        (0..batch).into_par_iter()
            .zip(out.par_chunks_mut(c_stride))
            .for_each(|(bi, out_i)| {
                let a_i = &a_slice[bi * a_stride..(bi + 1) * a_stride];
                let b_i = &b_slice[bi * b_stride..(bi + 1) * b_stride];
                let c_i = &c_slice[bi * c_stride..(bi + 1) * c_stride];
                let sa_i = &scale_a_exp[bi * m..(bi + 1) * m];
                let sb_i = &scale_b_exp[bi * n..(bi + 1) * n];
                run_one_tile_block_scale_k32(
                    a_i, b_i, c_i, sa_i, sb_i,
                    m, n, k,
                    nfb, a_min_exp, b_min_exp, c_min_exp,
                    out_mantissa_bits, out_i,
                );
            });
    });

    let arr = ndarray::Array3::from_shape_vec((batch, m, n), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

// ─────────────────────────────────────────────────────────────────────
// M1d — Block-scaled mxfp4 (k=64, per-block scales, fp4 packed inputs).
//
// Oracle [nv_ptx.py:132-143]: unpacks A and B from fp4 bytes, then uses
// `nv_fused_dot_add_with_block_scale` which (per block of 16 elements):
//   1. Sums a[k:k+16] * b[k:k+16] as f64 → block_sum.
//   2. Extracts scale_a[k/block_size] and scale_b[k/block_size] exps.
//   3. Appends (block_sum, scale_a_exp + scale_b_exp) as a fused_sum
//      term (with scale sig = 1 for ue8m0 — which is our corpus case).
// Plus C as an initial term. Then fused_sum, normalize f32 RZ.
//
// We stay in f64 throughout this kernel (no integer pipeline): the
// per-block sums are small, the sums fit in 53 bits, and matching the
// oracle's f64 trunc() path is the most direct route to bit-exact.
// NFB=35.
// ─────────────────────────────────────────────────────────────────────

/// FP4 e2m1 decode table (oracle arithmetic.py:37-50). Indexed by raw
/// 4-bit value: bit 3 is sign, bits 0-2 select magnitude.
const FP4_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

#[inline(always)]
fn unpack_fp4_byte(packed: u8) -> (f32, f32) {
    let lo = (packed & 0x0F) as usize;
    let hi = (packed >> 4) as usize;
    (FP4_TABLE[lo], FP4_TABLE[hi])
}

/// f64 equivalent of the oracle's extract_significand_exponent.
/// Returns sig in [1, 2) or 0, integer exp, with subnormal flush at min_exp.
#[inline(always)]
fn extract_sig_exp_f64(x: f64, min_exp: i32) -> (f64, i32) {
    if x == 0.0 { return (0.0, -126); }
    let (mut s, e_isz) = libm::frexp(x);
    let mut e: i32 = e_isz as i32;
    s *= 2.0;
    e -= 1;
    if e < min_exp {
        s *= fast_pow2(e - min_exp);
        e = min_exp;
    }
    if s == 0.0 { return (0.0, -126); }
    (s, e)
}

/// f64 fused_sum mirroring oracle arithmetic.py:115-129.
#[inline]
fn fused_sum_f64(sigs: &[f64], exps: &[i32], nfb: i32) -> (f64, i32) {
    let max_e = *exps.iter().max().unwrap();
    let mut acc = 0.0f64;
    for i in 0..sigs.len() {
        let shift = nfb + exps[i] - max_e;
        let scaled = sigs[i] * fast_pow2(shift);
        acc += scaled.trunc();
    }
    (acc * fast_pow2(-nfb), max_e)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn run_one_tile_mxfp4_k64(
    a_packed: &[u8],           // (m, k/2) = (16, 32) for k=64
    b_packed: &[u8],           // (k/2, n) = (32, 8)
    c: &[f32],                 // (m, n)
    scale_a_exp: &[i32],       // (m, k/block_size)
    scale_b_exp: &[i32],       // (k/block_size, n)
    m: usize, n: usize, k: usize,
    block_size: usize,
    nfb: i32,
    out_mantissa_bits: i32,
    out: &mut [f32],
) {
    // Unpack A, B to f32. k=64, so m*k = 1024, k*n = 512.
    let mut a_f32 = [0f32; 1024];
    let mut b_f32 = [0f32; 512];
    for i in 0..m {
        for p in 0..k / 2 {
            let (lo, hi) = unpack_fp4_byte(a_packed[i * (k / 2) + p]);
            a_f32[i * k + 2 * p]     = lo;
            a_f32[i * k + 2 * p + 1] = hi;
        }
    }
    for p in 0..k / 2 {
        for j in 0..n {
            let (lo, hi) = unpack_fp4_byte(b_packed[p * n + j]);
            // B layout: b_packed[p, j] packs (b[2p, j], b[2p+1, j])
            b_f32[2 * p * n + j]       = lo;
            b_f32[(2 * p + 1) * n + j] = hi;
        }
    }

    let scales_per_row = k / block_size;
    let step = 16usize; // oracle iterates blocks of 16 regardless of block_size

    for i in 0..m {
        for j in 0..n {
            let c_val = c[i * n + j] as f64;
            if c_val.is_nan() {
                out[i * n + j] = f32::from_bits(0x7FFF_FFFF);
                continue;
            }
            let (c_sig, c_exp) = extract_sig_exp_f64(c_val, -126);

            // Per-16-element-block term collection.
            let n_blocks = k / step;
            // n_blocks can be up to 4 for k=64.
            let mut sigs = [0.0f64; 16]; // over-provisioned
            let mut exps = [0i32; 16];
            sigs[0] = c_sig;
            exps[0] = c_exp;
            let mut n_terms = 1usize;

            let mut had_nan_inf = false;
            for b in 0..n_blocks {
                let k_start = b * step;
                let mut block_sum = 0.0f64;
                for l in 0..step {
                    let idx = k_start + l;
                    block_sum += (a_f32[i * k + idx] as f64)
                               * (b_f32[idx * n + j] as f64);
                }
                if !block_sum.is_finite() {
                    had_nan_inf = true;
                    break;
                }
                let scale_idx = (b * step) / block_size;
                let sae = scale_a_exp[i * scales_per_row + scale_idx];
                let sbe = scale_b_exp[scale_idx * n + j];
                sigs[n_terms] = block_sum;
                exps[n_terms] = sae + sbe;
                n_terms += 1;
            }

            if had_nan_inf {
                // Let fused_sum path handle via f64 semantics; simpler:
                // recompute products flat and let f64 produce Inf/NaN.
                let mut s = c_val;
                for idx in 0..k {
                    s += (a_f32[i * k + idx] as f64) * (b_f32[idx * n + j] as f64);
                }
                if s.is_nan() {
                    out[i * n + j] = f32::from_bits(0x7FFF_FFFF);
                } else {
                    out[i * n + j] = s as f32;
                }
                continue;
            }

            let (sum_val, max_e) = fused_sum_f64(
                &sigs[..n_terms], &exps[..n_terms], nfb,
            );
            // sum_val already incorporates the 2^-nfb factor; just apply 2^max_e.
            let result_f64 = sum_val * fast_pow2(max_e);
            out[i * n + j] = normalize_f32(result_f64, out_mantissa_bits);
        }
    }
}

#[pyfunction]
#[pyo3(signature = (a, b, c, scale_a, scale_b, nfb, block_size, out_mantissa_bits))]
#[allow(clippy::too_many_arguments)]
fn mma_f32_out_mxfp4_k64_rayon<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, u8>,         // (batch, m, k/2)
    b: PyReadonlyArray3<'py, u8>,         // (batch, k/2, n)
    c: PyReadonlyArray3<'py, f32>,        // (batch, m, n)
    scale_a: PyReadonlyArray3<'py, f32>,  // (batch, m, k/block_size)
    scale_b: PyReadonlyArray3<'py, f32>,  // (batch, k/block_size, n)
    nfb: i32,
    block_size: usize,
    out_mantissa_bits: i32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let a_v = a.as_array();
    let b_v = b.as_array();
    let c_v = c.as_array();
    let sa_v = scale_a.as_array();
    let sb_v = scale_b.as_array();

    let batch = a_v.shape()[0];
    let m = a_v.shape()[1];
    let k_half = a_v.shape()[2]; // k / 2
    let k = k_half * 2;
    let n = b_v.shape()[2];
    assert_eq!(b_v.shape(), &[batch, k_half, n]);
    assert_eq!(c_v.shape(), &[batch, m, n]);
    let nsc = k / block_size;
    assert_eq!(sa_v.shape(), &[batch, m, nsc]);
    assert_eq!(sb_v.shape(), &[batch, nsc, n]);

    let sa_flat: Vec<f32> = sa_v.iter().copied().collect();
    let sb_flat: Vec<f32> = sb_v.iter().copied().collect();
    let scale_a_exp: Vec<i32> = sa_flat.iter().map(|&v| extract_pow2_exp(v)).collect();
    let scale_b_exp: Vec<i32> = sb_flat.iter().map(|&v| extract_pow2_exp(v)).collect();

    let a_owned: Vec<u8>;
    let a_slice: &[u8] = match a_v.as_slice() {
        Some(s) => s,
        None => { a_owned = a_v.iter().copied().collect(); &a_owned }
    };
    let b_owned: Vec<u8>;
    let b_slice: &[u8] = match b_v.as_slice() {
        Some(s) => s,
        None => { b_owned = b_v.iter().copied().collect(); &b_owned }
    };
    let c_owned: Vec<f32>;
    let c_slice: &[f32] = match c_v.as_slice() {
        Some(s) => s,
        None => { c_owned = c_v.iter().copied().collect(); &c_owned }
    };

    let mut out = vec![0f32; batch * m * n];
    let a_stride = m * k_half;
    let b_stride = k_half * n;
    let c_stride = m * n;
    let sa_stride = m * nsc;
    let sb_stride = nsc * n;

    py.allow_threads(|| {
        (0..batch).into_par_iter()
            .zip(out.par_chunks_mut(c_stride))
            .for_each(|(bi, out_i)| {
                let a_i = &a_slice[bi * a_stride..(bi + 1) * a_stride];
                let b_i = &b_slice[bi * b_stride..(bi + 1) * b_stride];
                let c_i = &c_slice[bi * c_stride..(bi + 1) * c_stride];
                let sa_i = &scale_a_exp[bi * sa_stride..(bi + 1) * sa_stride];
                let sb_i = &scale_b_exp[bi * sb_stride..(bi + 1) * sb_stride];
                run_one_tile_mxfp4_k64(
                    a_i, b_i, c_i, sa_i, sb_i,
                    m, n, k, block_size,
                    nfb, out_mantissa_bits, out_i,
                );
            });
    });

    let arr = ndarray::Array3::from_shape_vec((batch, m, n), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

// ─────────────────────────────────────────────────────────────────────
// M1b — F64 kernel.
//
// Oracle ([nv_ptx.py:70-71]) does serial `libm.fma(a, b, sum)` across k
// per output. Rust's `f64::mul_add` compiles to the hardware FMA
// instruction on modern aarch64 / x86 — IEEE-754 fused with single
// rounding, same as libm.fma.
//
// No integer pipeline here; f64 precision is the oracle's precision.
// ─────────────────────────────────────────────────────────────────────

#[inline]
fn run_one_tile_f64(
    a: &[f64], b: &[f64], c: &[f64],
    m: usize, n: usize, k: usize,
    out: &mut [f64],
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = c[i * n + j];
            for l in 0..k {
                sum = a[i * k + l].mul_add(b[l * n + j], sum);
            }
            out[i * n + j] = sum;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// N3 — AMD CDNA3 fused_dot_rd_add (Φ_TR-FDPA, Φ_GTR-FDPA).
//
// Mirrors mmasim/simulator/arithmetic.py::amd_fused_dot_rd_add. f64
// arithmetic throughout. Key details:
//   - RD (round-down / floor) alignment, not RZ or RNE.
//   - For fp8: split even/odd products, two fused_sum halves, merge
//     at local max_e with RD at `nfb` fractional bits, then re-align
//     at global max_e with 31 fractional bits (F_2 per paper Table 7).
//   - C aligns at 24 fractional bits (F per Table 7).
//   - Overflow: if any product magnitude >= 2^128, output is ±Inf/NaN.
//   - After each group, caller rounds the f64 result to d_type (f32) via
//     Rust's `as f32` (IEEE RNE) — matches the oracle's `torch.tensor(x,
//     dtype=d_type)` cast.
// ─────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn amd_fused_dot_rd_add_f64(
    a: &[f32], b: &[f32], c: f32,
    n_fractional_bits: i32,
    is_fp8: bool,
    a_min_exp: i32, b_min_exp: i32, c_min_exp: i32,
) -> f64 {
    // Products + NaN/Inf short-circuit via f64 pre-check.
    let l = a.len();
    let mut products = [0.0f64; 64];
    assert!(l <= 64);
    let mut fp_sum: f64 = c as f64;
    for i in 0..l {
        products[i] = (a[i] as f64) * (b[i] as f64);
        fp_sum += products[i];
    }
    if !fp_sum.is_finite() {
        return fp_sum;
    }

    // Overflow partitioning and sig/exp collection.
    let thresh = fast_pow2(128);
    let mut p_inf = false;
    let mut n_inf = false;
    let mut sigs = [0.0f64; 64];
    let mut exps = [0i32; 64];
    let mut n_terms = 0usize;
    for i in 0..l {
        if products[i] >= thresh {
            p_inf = true;
        } else if products[i] <= -thresh {
            n_inf = true;
        } else {
            let (sa, ea) = extract_sig_exp_f64(a[i] as f64, a_min_exp);
            let (sb, eb) = extract_sig_exp_f64(b[i] as f64, b_min_exp);
            sigs[n_terms] = sa * sb;
            exps[n_terms] = ea + eb;
            n_terms += 1;
        }
    }
    if p_inf || n_inf {
        if p_inf && n_inf {
            return f64::NAN;
        }
        return if p_inf { f64::INFINITY } else { f64::NEG_INFINITY };
    }

    let (sc, ec) = extract_sig_exp_f64(c as f64, c_min_exp);

    // Stage 1: inner fused_sum uses RZ (matches oracle's fused_sum =
    // math.trunc). The outer alignments (below) use RD (floor). This
    // interleaving is specific to AMD CDNA3's TR-/GTR-FDPA.
    let (s_stage1, e_stage1) = if is_fp8 {
        let mut s0_sig = [0.0f64; 64];
        let mut s0_exp = [0i32; 64];
        let mut n0 = 0;
        let mut s1_sig = [0.0f64; 64];
        let mut s1_exp = [0i32; 64];
        let mut n1 = 0;
        for i in 0..n_terms {
            if i % 2 == 0 {
                s0_sig[n0] = sigs[i]; s0_exp[n0] = exps[i]; n0 += 1;
            } else {
                s1_sig[n1] = sigs[i]; s1_exp[n1] = exps[i]; n1 += 1;
            }
        }
        let (s0, e0) = fused_sum_f64(&s0_sig[..n0], &s0_exp[..n0], n_fractional_bits);
        let (s1, e1) = fused_sum_f64(&s1_sig[..n1], &s1_exp[..n1], n_fractional_bits);
        let max_e_local = e0.max(e1);
        let nfb_pow = fast_pow2(n_fractional_bits);
        // Outer alignment: RD (floor), matches oracle's math.floor.
        let s0a = (s0 * fast_pow2(n_fractional_bits + e0 - max_e_local)).floor() / nfb_pow;
        let s1a = (s1 * fast_pow2(n_fractional_bits + e1 - max_e_local)).floor() / nfb_pow;
        (s0a + s1a, max_e_local)
    } else {
        fused_sum_f64(&sigs[..n_terms], &exps[..n_terms], n_fractional_bits)
    };

    // Stage 2: re-align s at 31 bits and C at 24 bits, both at global max_e.
    let max_e = e_stage1.max(ec);
    let s_final = (s_stage1 * fast_pow2(31 + e_stage1 - max_e)).floor() * fast_pow2(-31);
    let sc_final = if is_fp8 && ec < max_e - 25 {
        0.0
    } else {
        (sc * fast_pow2(24 + ec - max_e)).floor() * fast_pow2(-24)
    };

    (s_final + sc_final) * fast_pow2(max_e)
}

/// Variant of fused_sum with RD (floor) alignment rather than RZ (trunc).
/// Matches the oracle's use of `math.floor` in amd_fused_dot_rd_add.
#[inline]
fn fused_sum_rd(sigs: &[f64], exps: &[i32], nfb: i32) -> (f64, i32) {
    let max_e = *exps.iter().max().unwrap();
    let nfb_pow = fast_pow2(nfb);
    let mut acc = 0.0f64;
    for i in 0..sigs.len() {
        let shift = nfb + exps[i] - max_e;
        let scaled = sigs[i] * fast_pow2(shift);
        acc += scaled.floor();
    }
    (acc / nfb_pow, max_e)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn run_one_tile_amd_fused_rd(
    a: &[f32], b: &[f32], c: &[f32],
    m: usize, n: usize, k: usize,
    group_size: usize,
    n_fractional_bits: i32,
    is_fp8: bool,
    a_min_exp: i32, b_min_exp: i32, c_min_exp: i32,
    out: &mut [f32],
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = c[i * n + j];
            let mut l = 0usize;
            while l < k {
                let end = (l + group_size).min(k);
                let gs = end - l;
                let mut a_buf = [0f32; 64];
                let mut b_buf = [0f32; 64];
                for ll in 0..gs {
                    a_buf[ll] = a[i * k + l + ll];
                    b_buf[ll] = b[(l + ll) * n + j];
                }
                let group_sum = amd_fused_dot_rd_add_f64(
                    &a_buf[..gs], &b_buf[..gs], sum,
                    n_fractional_bits, is_fp8,
                    a_min_exp, b_min_exp, c_min_exp,
                );
                // RNE cast from f64 to f32 (matches oracle's
                // `torch.tensor(x, dtype=self.d_type)` where d_type=f32).
                sum = group_sum as f32;
                l += group_size;
            }
            out[i * n + j] = sum;
        }
    }
}

#[pyfunction]
#[pyo3(signature = (a, b, c, group_size, n_fractional_bits, is_fp8, a_min_exp, b_min_exp, c_min_exp))]
#[allow(clippy::too_many_arguments)]
fn mma_f32_amd_fused_rd_rayon<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f32>,
    b: PyReadonlyArray3<'py, f32>,
    c: PyReadonlyArray3<'py, f32>,
    group_size: usize,
    n_fractional_bits: i32,
    is_fp8: bool,
    a_min_exp: i32,
    b_min_exp: i32,
    c_min_exp: i32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let a_v = a.as_array();
    let b_v = b.as_array();
    let c_v = c.as_array();
    let batch = a_v.shape()[0];
    let m = a_v.shape()[1];
    let k = a_v.shape()[2];
    let n = b_v.shape()[2];
    assert_eq!(b_v.shape(), &[batch, k, n]);
    assert_eq!(c_v.shape(), &[batch, m, n]);
    assert!(group_size <= 64);

    let a_owned: Vec<f32>;
    let a_slice: &[f32] = match a_v.as_slice() {
        Some(s) => s,
        None => { a_owned = a_v.iter().copied().collect(); &a_owned }
    };
    let b_owned: Vec<f32>;
    let b_slice: &[f32] = match b_v.as_slice() {
        Some(s) => s,
        None => { b_owned = b_v.iter().copied().collect(); &b_owned }
    };
    let c_owned: Vec<f32>;
    let c_slice: &[f32] = match c_v.as_slice() {
        Some(s) => s,
        None => { c_owned = c_v.iter().copied().collect(); &c_owned }
    };

    let mut out = vec![0.0f32; batch * m * n];
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    py.allow_threads(|| {
        out.par_chunks_mut(c_stride)
            .zip(a_slice.par_chunks(a_stride))
            .zip(b_slice.par_chunks(b_stride))
            .zip(c_slice.par_chunks(c_stride))
            .for_each(|(((o, a), b), c)| {
                run_one_tile_amd_fused_rd(
                    a, b, c, m, n, k, group_size,
                    n_fractional_bits, is_fp8,
                    a_min_exp, b_min_exp, c_min_exp, o,
                );
            });
    });

    let arr = ndarray::Array3::from_shape_vec((batch, m, n), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

// ─────────────────────────────────────────────────────────────────────
// N2 — AMD pairwise path (CDNA1 f16/bf16, CDNA2 f16/bf16 + FTZ).
//
// Mirrors mmasim/simulator/arithmetic.py::pairwise_dot exactly — recurse
// on halves, combine via `fmaf(l, 1.0, r)` (which is just f32 `l + r`
// with single rounding), optional subnormal-flush at each node.
//
// Per amd.py, outer k-loop sums groups sequentially into accumulator
// (f32 +, optionally flushed after each group for CDNA2).
// ─────────────────────────────────────────────────────────────────────

const F32_SMALLEST_NORMAL_BITS: u32 = 0x00800000; // 2^-126

#[inline(always)]
fn flush_sub_keep_sign_f32(x: f32) -> f32 {
    if x.abs() < f32::from_bits(F32_SMALLEST_NORMAL_BITS) {
        x * 0.0  // preserves sign (±0)
    } else {
        x
    }
}

/// Pairwise dot: recurse halves, combine via f32 single-rounded add.
/// Base case: `fmaf(a, b, 0) = a * b` with single round.
fn pairwise_dot_f32(a: &[f32], b: &[f32], flush: bool) -> f32 {
    let n = a.len();
    let mut sum = if n == 1 {
        // fmaf(a, b, 0.0) in libm — f32::mul_add gives the same result on
        // aarch64/x86 (hardware FMA with single rounding).
        a[0].mul_add(b[0], 0.0)
    } else {
        let m = n / 2;
        let l = pairwise_dot_f32(&a[..m], &b[..m], flush);
        let r = pairwise_dot_f32(&a[m..], &b[m..], flush);
        // fmaf(l, 1.0, r) = l + r with single rounding.
        l.mul_add(1.0, r)
    };
    if flush {
        sum = flush_sub_keep_sign_f32(sum);
    }
    sum
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn run_one_tile_amd_pairwise(
    a: &[f32], b: &[f32], c: &[f32],
    m: usize, n: usize, k: usize,
    group_size: usize,
    flush_denormal_mode: bool,
    out: &mut [f32],
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = c[i * n + j];
            // Caller already flushed C if flush_denormal_mode; no need here.
            let mut l = 0usize;
            while l < k {
                let end = (l + group_size).min(k);
                let gs = end - l;
                // Stack-buffer the group slices (gs ≤ 4).
                let mut a_buf = [0f32; 8];
                let mut b_buf = [0f32; 8];
                for ll in 0..gs {
                    a_buf[ll] = a[i * k + l + ll];
                    b_buf[ll] = b[(l + ll) * n + j];
                }
                let group_sum = pairwise_dot_f32(
                    &a_buf[..gs], &b_buf[..gs], flush_denormal_mode,
                );
                sum += group_sum;
                if flush_denormal_mode {
                    sum = flush_sub_keep_sign_f32(sum);
                }
                l += group_size;
            }
            out[i * n + j] = sum;
        }
    }
}

#[pyfunction]
#[pyo3(signature = (a, b, c, group_size, flush_denormal))]
fn mma_f32_amd_pairwise_rayon<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f32>,
    b: PyReadonlyArray3<'py, f32>,
    c: PyReadonlyArray3<'py, f32>,
    group_size: usize,
    flush_denormal: bool,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let a_v = a.as_array();
    let b_v = b.as_array();
    let c_v = c.as_array();
    let batch = a_v.shape()[0];
    let m = a_v.shape()[1];
    let k = a_v.shape()[2];
    let n = b_v.shape()[2];
    assert_eq!(b_v.shape(), &[batch, k, n]);
    assert_eq!(c_v.shape(), &[batch, m, n]);
    assert!(group_size <= 8);

    let a_owned: Vec<f32>;
    let a_slice: &[f32] = match a_v.as_slice() {
        Some(s) => s,
        None => { a_owned = a_v.iter().copied().collect(); &a_owned }
    };
    let b_owned: Vec<f32>;
    let b_slice: &[f32] = match b_v.as_slice() {
        Some(s) => s,
        None => { b_owned = b_v.iter().copied().collect(); &b_owned }
    };
    let c_owned: Vec<f32>;
    let c_slice: &[f32] = match c_v.as_slice() {
        Some(s) => s,
        None => { c_owned = c_v.iter().copied().collect(); &c_owned }
    };

    let mut out = vec![0.0f32; batch * m * n];
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    py.allow_threads(|| {
        out.par_chunks_mut(c_stride)
            .zip(a_slice.par_chunks(a_stride))
            .zip(b_slice.par_chunks(b_stride))
            .zip(c_slice.par_chunks(c_stride))
            .for_each(|(((o, a), b), c)| {
                run_one_tile_amd_pairwise(
                    a, b, c, m, n, k, group_size, flush_denormal, o,
                );
            });
    });

    let arr = ndarray::Array3::from_shape_vec((batch, m, n), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

/// AMD f32 fma kernel. Mirrors mmasim/simulator/amd.py `operation_type == "fma"`
/// path for f32 operands: `sum = fma(a, b, sum)` across k per output.
/// Uses `f32::mul_add` which compiles to native f32 FMA on aarch64/x86.
#[inline]
fn run_one_tile_f32_fma(
    a: &[f32], b: &[f32], c: &[f32],
    m: usize, n: usize, k: usize,
    out: &mut [f32],
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = c[i * n + j];
            for l in 0..k {
                sum = a[i * k + l].mul_add(b[l * n + j], sum);
            }
            out[i * n + j] = sum;
        }
    }
}

#[pyfunction]
#[pyo3(signature = (a, b, c))]
fn mma_f32_fma_batched_rayon<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f32>,
    b: PyReadonlyArray3<'py, f32>,
    c: PyReadonlyArray3<'py, f32>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let a_v = a.as_array();
    let b_v = b.as_array();
    let c_v = c.as_array();
    let batch = a_v.shape()[0];
    let m = a_v.shape()[1];
    let k = a_v.shape()[2];
    let n = b_v.shape()[2];
    assert_eq!(b_v.shape(), &[batch, k, n]);
    assert_eq!(c_v.shape(), &[batch, m, n]);

    let a_owned: Vec<f32>;
    let a_slice: &[f32] = match a_v.as_slice() {
        Some(s) => s,
        None => { a_owned = a_v.iter().copied().collect(); &a_owned }
    };
    let b_owned: Vec<f32>;
    let b_slice: &[f32] = match b_v.as_slice() {
        Some(s) => s,
        None => { b_owned = b_v.iter().copied().collect(); &b_owned }
    };
    let c_owned: Vec<f32>;
    let c_slice: &[f32] = match c_v.as_slice() {
        Some(s) => s,
        None => { c_owned = c_v.iter().copied().collect(); &c_owned }
    };

    let mut out = vec![0.0f32; batch * m * n];
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    py.allow_threads(|| {
        out.par_chunks_mut(c_stride)
            .zip(a_slice.par_chunks(a_stride))
            .zip(b_slice.par_chunks(b_stride))
            .zip(c_slice.par_chunks(c_stride))
            .for_each(|(((o, a), b), c)| {
                run_one_tile_f32_fma(a, b, c, m, n, k, o);
            });
    });

    let arr = ndarray::Array3::from_shape_vec((batch, m, n), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (a, b, c))]
fn mma_f64_batched_rayon<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f64>,
    b: PyReadonlyArray3<'py, f64>,
    c: PyReadonlyArray3<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let a_v = a.as_array();
    let b_v = b.as_array();
    let c_v = c.as_array();
    let batch = a_v.shape()[0];
    let m = a_v.shape()[1];
    let k = a_v.shape()[2];
    let n = b_v.shape()[2];
    assert_eq!(b_v.shape(), &[batch, k, n]);
    assert_eq!(c_v.shape(), &[batch, m, n]);

    let a_owned: Vec<f64>;
    let a_slice: &[f64] = match a_v.as_slice() {
        Some(s) => s,
        None => { a_owned = a_v.iter().copied().collect(); &a_owned }
    };
    let b_owned: Vec<f64>;
    let b_slice: &[f64] = match b_v.as_slice() {
        Some(s) => s,
        None => { b_owned = b_v.iter().copied().collect(); &b_owned }
    };
    let c_owned: Vec<f64>;
    let c_slice: &[f64] = match c_v.as_slice() {
        Some(s) => s,
        None => { c_owned = c_v.iter().copied().collect(); &c_owned }
    };

    let mut out = vec![0.0f64; batch * m * n];
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    py.allow_threads(|| {
        out.par_chunks_mut(c_stride)
            .zip(a_slice.par_chunks(a_stride))
            .zip(b_slice.par_chunks(b_stride))
            .zip(c_slice.par_chunks(c_stride))
            .for_each(|(((o, a), b), c)| {
                run_one_tile_f64(a, b, c, m, n, k, o);
            });
    });

    let arr = ndarray::Array3::from_shape_vec((batch, m, n), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

/// Phase 4 — Metal GPU backend for ampere-f16 workhorse.
#[cfg(target_os = "macos")]
#[pyfunction]
fn mma_metal_ampere_f16<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f32>,
    b: PyReadonlyArray3<'py, f32>,
    c: PyReadonlyArray3<'py, f32>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let a_v = a.as_array();
    let b_v = b.as_array();
    let c_v = c.as_array();
    let batch = a_v.shape()[0];
    assert_eq!(a_v.shape(), &[batch, 16, 16]);
    assert_eq!(b_v.shape(), &[batch, 16, 8]);
    assert_eq!(c_v.shape(), &[batch, 16, 8]);

    let a_owned: Vec<f32>;
    let a_slice: &[f32] = match a_v.as_slice() {
        Some(s) => s,
        None => { a_owned = a_v.iter().copied().collect(); &a_owned }
    };
    let b_owned: Vec<f32>;
    let b_slice: &[f32] = match b_v.as_slice() {
        Some(s) => s,
        None => { b_owned = b_v.iter().copied().collect(); &b_owned }
    };
    let c_owned: Vec<f32>;
    let c_slice: &[f32] = match c_v.as_slice() {
        Some(s) => s,
        None => { c_owned = c_v.iter().copied().collect(); &c_owned }
    };

    let out = py.allow_threads(|| {
        metal_ampere_f16::run_batched(a_slice, b_slice, c_slice, batch)
    });

    let arr = ndarray::Array3::from_shape_vec((batch, 16, 8), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

// Keep guarded_shl reachable for future block-scale paths and silence dead-code
// until it's used.
#[allow(dead_code)]
fn _unused() {
    let _ = guarded_shl(0, 0);
}
