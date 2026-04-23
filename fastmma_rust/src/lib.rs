//! Rust reimplementation of mmasim's mma path.
//!
//! Phase 2 scope: the Ampere f16×f16→f32 workhorse (m16n8k16, nfb=24,
//! split-K). Ported 1:1 from `fastmma/numpy_ref.py`, using f64 ops
//! instead of NumPy — so semantics are identical, but we pay no Python
//! dispatch or NumPy allocation per call.
//!
//! All three inputs arrive as contiguous f32 arrays (the Python wrapper
//! does the bf16/f16/tf32 → f32 cast before calling in; f32 is a strict
//! superset of f16/bf16 so this is lossless).

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

const F32_MIN_EXP: i32 = -126;

#[inline(always)]
fn extract_sig_exp(x: f64, min_exp: i32) -> (f64, i32) {
    // Mirrors fastmma/numpy_ref.py::_extract_sig_exp.
    if x == 0.0 {
        return (0.0, -126); // zero quirk
    }
    let (mut s, mut e) = libm::frexp(x); // s in [0.5, 1), e int
    s *= 2.0; // s in [1, 2)
    e -= 1;
    if e < min_exp {
        s *= f64::powi(2.0, e - min_exp);
        e = min_exp;
    }
    if s == 0.0 {
        return (0.0, -126);
    }
    (s, e)
}

/// One fused dot-add step over (M, N), one output per iteration.
/// Mirrors `fastmma/numpy_ref.py::_fused_mma_step` for f32-output paths.
#[inline]
fn fused_mma_step_f32_out(
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
    // Decompose all of A, B, C up front. Stored as two flat vectors each.
    let mut a_sig = vec![0.0f64; m * k];
    let mut a_exp = vec![0i32; m * k];
    for i in 0..m * k {
        let (s, e) = extract_sig_exp(a[i] as f64, a_min_exp);
        a_sig[i] = s;
        a_exp[i] = e;
    }
    let mut b_sig = vec![0.0f64; k * n];
    let mut b_exp = vec![0i32; k * n];
    for i in 0..k * n {
        let (s, e) = extract_sig_exp(b[i] as f64, b_min_exp);
        b_sig[i] = s;
        b_exp[i] = e;
    }

    // Per-output loop. Inner k-reduction is the only hot dimension.
    for i in 0..m {
        let a_row_sig = &a_sig[i * k..(i + 1) * k];
        let a_row_exp = &a_exp[i * k..(i + 1) * k];
        for j in 0..n {
            let (c_s, c_e) = extract_sig_exp(c[i * n + j] as f64, c_min_exp);

            // Collect all k+1 (sig, exp) pairs, find max exponent.
            let mut max_e = c_e;
            // Small stack buffers keep us out of the heap for the inner loop.
            // k=16 for Ampere f16 (or 8 per half for split-K), so 17 entries is safe.
            let mut sigs = [0.0f64; 128];
            let mut exps = [0i32; 128];
            let n_terms = k + 1;
            debug_assert!(n_terms <= sigs.len());

            sigs[0] = c_s;
            exps[0] = c_e;

            // f64 pre-check for NaN/Inf (matches oracle short-circuit).
            // We accumulate products as plain f64 while we're walking the k
            // axis anyway — free NaN/Inf detection at the same time.
            let mut fp_sum = c[i * n + j] as f64;

            for l in 0..k {
                let a_v = a[i * k + l] as f64;
                let b_v = b[l * n + j] as f64;
                fp_sum += a_v * b_v;

                let (a_s, a_e) = (a_row_sig[l], a_row_exp[l]);
                let (b_s, b_e) = (b_sig[l * n + j], b_exp[l * n + j]);
                let sp = a_s * b_s;
                let ep = a_e + b_e;
                sigs[l + 1] = sp;
                exps[l + 1] = ep;
                if ep > max_e {
                    max_e = ep;
                }
            }

            let result_f64 = if !fp_sum.is_finite() {
                fp_sum
            } else {
                // fused_sum body
                let scale_denom = f64::powi(2.0, nfb);
                let mut acc = 0.0f64;
                for idx in 0..n_terms {
                    let shift = (nfb + exps[idx] - max_e) as i32;
                    let scale = f64::powi(2.0, shift);
                    let rounded = libm::trunc(sigs[idx] * scale);
                    acc += rounded;
                }
                let sum_val = acc / scale_denom;
                sum_val * f64::powi(2.0, max_e)
            };

            // Output normalization: RZ to out_mantissa_bits, cast to f32.
            let out_v: f32 = if result_f64.is_nan() {
                f32::from_bits(0x7FFF_FFFF)
            } else if result_f64.is_infinite() {
                result_f64 as f32
            } else {
                let (mut s, e) = extract_sig_exp(result_f64, F32_MIN_EXP);
                let scale = f64::powi(2.0, out_mantissa_bits);
                s = libm::trunc(s * scale) / scale;
                (s * f64::powi(2.0, e)) as f32
            };
            out[i * n + j] = out_v;
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
    assert!(k + 1 <= 128, "k too large for stack buffer");

    // Make contiguous row-major copies. PyReadonlyArray2 gives us a view;
    // we want flat &[f32] for tight inner loops without stride arithmetic.
    let a_flat: Vec<f32> = a_v.iter().copied().collect();
    let b_flat: Vec<f32> = b_v.iter().copied().collect();
    let c_flat: Vec<f32> = c_v.iter().copied().collect();

    let mut out = vec![0.0f32; m * n];

    if split_k {
        let half = k / 2;
        let mut mid = vec![0.0f32; m * n];
        let a_half: Vec<f32> = (0..m)
            .flat_map(|i| a_flat[i * k..i * k + half].iter().copied())
            .collect();
        let b_half: Vec<f32> = b_flat[..half * n].to_vec();
        fused_mma_step_f32_out(
            &a_half, &b_half, &c_flat, m, n, half,
            nfb, a_min_exp, b_min_exp, c_min_exp, out_mantissa_bits, &mut mid,
        );

        let a_half2: Vec<f32> = (0..m)
            .flat_map(|i| a_flat[i * k + half..(i + 1) * k].iter().copied())
            .collect();
        let b_half2: Vec<f32> = b_flat[half * n..].to_vec();
        // Second half uses `mid` in the C slot; its min_exp is f32's.
        fused_mma_step_f32_out(
            &a_half2, &b_half2, &mid, m, n, half,
            nfb, a_min_exp, b_min_exp, F32_MIN_EXP, out_mantissa_bits, &mut out,
        );
    } else {
        fused_mma_step_f32_out(
            &a_flat, &b_flat, &c_flat, m, n, k,
            nfb, a_min_exp, b_min_exp, c_min_exp, out_mantissa_bits, &mut out,
        );
    }

    let arr = ndarray::Array2::from_shape_vec((m, n), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

#[pymodule]
fn fastmma_rust(_py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mma_f32_out, &m)?)?;
    Ok(())
}
