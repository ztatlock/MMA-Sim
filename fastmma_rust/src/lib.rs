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

    // Upshift to scale 2^nfb. nfb >= 23 for all supported ISA/arch combos.
    let shift_up = nfb - 23;
    let mut int_sig = sig_q23 << shift_up;

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

#[pymodule]
fn fastmma_rust(_py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mma_f32_out, &m)?)?;
    m.add_function(wrap_pyfunction!(mma_f32_out_batched, &m)?)?;
    Ok(())
}

// Keep guarded_shl reachable for future block-scale paths and silence dead-code
// until it's used.
#[allow(dead_code)]
fn _unused() {
    let _ = guarded_shl(0, 0);
}
