//! Metal (GPU) backend for the Ampere m16n8k16.f32.f16.f16.f32 workhorse.
//!
//! One threadgroup per tile, one thread per output (16 × 8 = 128 threads).
//! Kernel mirrors the CPU specialized kernel's integer pipeline; final
//! normalize is done in MSL via integer bit construction so we avoid any
//! f64→f32 rounding ambiguity.

use std::sync::OnceLock;

use metal::{
    CompileOptions, ComputePipelineDescriptor, ComputePipelineState, Device, Library,
    MTLResourceOptions, MTLSize,
};

const KERNEL_NAME: &str = "mma_ampere_f16";

const MSL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant int NFB = 24;
constant int F16_MIN_EXP = -14;
constant int F32_MIN_EXP = -126;
constant int K_HALF = 8;

// f16 (cast to f32) decompose — no f32-subnormal path needed.
static inline long2 decompose_narrow(float v) {
    uint bits = as_type<uint>(v);
    int sign = int((bits >> 31) & 1u);
    int biased_e = int((bits >> 23) & 0xFFu);
    long mant = long(bits & 0x7FFFFFu);
    if (biased_e == 0) {
        return long2(0, -126);
    }
    long sig_qnfb = ((1L << 23) + mant) << 1; // scale to 2^24
    int exp = biased_e - 127;
    long int_sig;
    int final_exp;
    if (exp < F16_MIN_EXP) {
        uint shift = uint(F16_MIN_EXP - exp);
        long s = sig_qnfb >> shift;
        if (s == 0) { int_sig = 0; final_exp = -126; }
        else        { int_sig = s; final_exp = F16_MIN_EXP; }
    } else {
        int_sig = sig_qnfb;
        final_exp = exp;
    }
    if (sign != 0) int_sig = -int_sig;
    return long2(int_sig, final_exp);
}

// General f32 decompose (for C addend and the mid-pass C).
static inline long2 decompose_f32(float v) {
    uint bits = as_type<uint>(v);
    int sign = int((bits >> 31) & 1u);
    int biased_e = int((bits >> 23) & 0xFFu);
    long mant = long(bits & 0x7FFFFFu);
    if (biased_e == 0 && mant == 0) {
        return long2(0, -126);
    }
    long sig_q23;
    int exp;
    if (biased_e == 0) {
        int top = 63 - int(clz(ulong(mant)));
        sig_q23 = mant << (23 - top);
        exp = top - 149;
    } else {
        sig_q23 = (1L << 23) + mant;
        exp = biased_e - 127;
    }
    long int_sig = sig_q23 << 1; // to 2^24
    if (exp < F32_MIN_EXP) {
        uint shift = uint(F32_MIN_EXP - exp);
        int_sig = int_sig >> shift;
        exp = F32_MIN_EXP;
        if (int_sig == 0) exp = -126;
    }
    if (sign != 0) int_sig = -int_sig;
    return long2(int_sig, exp);
}

// Trunc-toward-zero right shift of signed long.
static inline long trunc_shr(long x, int shift) {
    if (shift <= 0) return x;
    if (shift >= 63) return 0L;
    // Rust's signed `/` rounds toward zero. MSL doesn't guarantee that
    // for right-shift on negatives (it's arithmetic = floor). Do abs +
    // logical shift + reapply sign.
    long sign_mask = x >> 63;      // 0 or -1
    long abs_x = (x ^ sign_mask) - sign_mask;
    long shifted = long(ulong(abs_x) >> uint(shift));
    return (shifted ^ sign_mask) - sign_mask;
}

// Pack (sign, unbiased_exp, mantissa) into f32 bits. Handles normal,
// subnormal, overflow. All in integer arithmetic.
static inline float pack_f32(bool sign, long acc_int, int max_e) {
    // acc_int is the integer sum at scale 2^NFB at exponent max_e.
    // True value = acc_int * 2^(max_e - NFB), sign applied.
    // We want RZ-truncate to 23 mantissa bits.
    if (acc_int == 0) return 0.0f;
    ulong mag = ulong(acc_int);  // already non-negative (caller handled sign)
    int p = 63 - int(clz(mag));    // MSB position
    int unbiased = (max_e - NFB) + p;
    uint sign_bit = sign ? 0x80000000u : 0u;
    if (unbiased > 127) {
        return as_type<float>(sign_bit | 0x7F800000u); // ±Inf
    }
    if (unbiased < -149) {
        return as_type<float>(sign_bit); // ±0 (underflow)
    }
    if (unbiased < -126) {
        // subnormal: biased_e = 0, mantissa = floor(|true_value| * 2^149)
        // |true_value| = mag * 2^(max_e - NFB). So mantissa = floor(mag * 2^(max_e - NFB + 149)).
        int shift_val = (max_e - NFB) + 149;
        uint mant_sub;
        if (shift_val >= 0) {
            // left shift
            if (shift_val >= 64) return as_type<float>(sign_bit | 0x7F800000u);
            mant_sub = uint(mag << uint(shift_val));
        } else {
            // right shift (trunc toward zero for positive mag)
            uint right = uint(-shift_val);
            if (right >= 64) return as_type<float>(sign_bit);
            mant_sub = uint(mag >> right);
        }
        return as_type<float>(sign_bit | (mant_sub & 0x7FFFFFu));
    }
    // Normal case
    uint mant24;
    if (p >= 23) {
        mant24 = uint(mag >> uint(p - 23));
    } else {
        mant24 = uint(mag << uint(23 - p));
    }
    uint mant23 = mant24 & 0x7FFFFFu;
    uint biased = uint(unbiased + 127);
    return as_type<float>(sign_bit | (biased << 23) | mant23);
}

// Intermediate f32 output of the first-half pass. Same semantics as CPU
// normalize_f32: RZ to 23 mantissa bits, pass NaN/Inf through.
static inline float normalize_f32(long acc_int, int max_e, float fp_guard) {
    if (!isfinite(fp_guard)) {
        if (isnan(fp_guard)) return as_type<float>(0x7FFFFFFFu);
        return fp_guard; // ±Inf
    }
    bool sign = acc_int < 0;
    long abs_acc = sign ? -acc_int : acc_int;
    return pack_f32(sign, abs_acc, max_e);
}

// One split-K half step. Computes 128 outputs (one per thread).
static inline void step_half(
    device const float* a,   // (16 * 8)
    device const float* b,   // (8 * 8)
    device const float* c,   // (16 * 8)
    device       float* out, // (16 * 8)
    uint         ti,         // thread i (0..16)
    uint         tj          // thread j (0..8)
) {
    // Decompose C
    long2 c_se = decompose_f32(c[ti * 8 + tj]);
    long c_sig = c_se.x; int c_exp = int(c_se.y);

    float fp = c[ti * 8 + tj];

    // Collect K_HALF=8 product (sig, exp), track max_e.
    long prod_sig[8];
    int  prod_exp[8];
    int max_e = c_exp;
    for (uint l = 0; l < 8; ++l) {
        float av = a[ti * 8 + l];
        float bv = b[l * 8 + tj];
        long2 a_se = decompose_narrow(av);
        long2 b_se = decompose_narrow(bv);
        fp += av * bv;
        long ps = a_se.x * b_se.x;
        int  pe = int(a_se.y) + int(b_se.y);
        prod_sig[l] = ps;
        prod_exp[l] = pe;
        max_e = max(max_e, pe);
    }

    long acc = trunc_shr(c_sig, max_e - c_exp);
    for (uint l = 0; l < 8; ++l) {
        int shift = max_e - prod_exp[l] + NFB;
        acc += trunc_shr(prod_sig[l], shift);
    }

    out[ti * 8 + tj] = normalize_f32(acc, max_e, fp);
}

kernel void mma_ampere_f16(
    device const float* A        [[ buffer(0) ]],   // (batch, 16, 16)
    device const float* B        [[ buffer(1) ]],   // (batch, 16, 8)
    device const float* C        [[ buffer(2) ]],   // (batch, 16, 8)
    device       float* D        [[ buffer(3) ]],   // (batch, 16, 8)
    threadgroup  float* mid      [[ threadgroup(0) ]], // (16 * 8), shared per TG
    uint3 gid [[ threadgroup_position_in_grid ]],
    uint3 tid [[ thread_position_in_threadgroup ]]
) {
    // One threadgroup per tile. Threadgroup size: (8, 16, 1) = 128 threads.
    uint bi = gid.x;
    uint tj = tid.x;  // 0..8
    uint ti = tid.y;  // 0..16

    // Tile pointers
    device const float* a_tile = A + bi * 16 * 16;
    device const float* b_tile = B + bi * 16 *  8;
    device const float* c_tile = C + bi * 16 *  8;
    device       float* d_tile = D + bi * 16 *  8;

    // Stack-slice A into two halves via threadgroup memory would be ideal,
    // but for simplicity we index into A directly (non-contiguous K halves).
    // Each thread reads exactly its required A row-slice.

    // --- First half: K indices 0..8, C is the input C, output -> mid ---
    {
        long2 c_se = decompose_f32(c_tile[ti * 8 + tj]);
        long c_sig = c_se.x; int c_exp = int(c_se.y);
        float fp = c_tile[ti * 8 + tj];

        long prod_sig[8];
        int  prod_exp[8];
        int max_e = c_exp;
        for (uint l = 0; l < 8; ++l) {
            float av = a_tile[ti * 16 + l];          // A[i, l] for l < 8
            float bv = b_tile[l * 8 + tj];           // B[l, j] for l < 8
            long2 a_se = decompose_narrow(av);
            long2 b_se = decompose_narrow(bv);
            fp += av * bv;
            long ps = a_se.x * b_se.x;
            int  pe = int(a_se.y) + int(b_se.y);
            prod_sig[l] = ps;
            prod_exp[l] = pe;
            max_e = max(max_e, pe);
        }
        long acc = trunc_shr(c_sig, max_e - c_exp);
        for (uint l = 0; l < 8; ++l) {
            int shift = max_e - prod_exp[l] + NFB;
            acc += trunc_shr(prod_sig[l], shift);
        }
        mid[ti * 8 + tj] = normalize_f32(acc, max_e, fp);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Second half: K indices 8..16, C is `mid`, output -> D ---
    {
        long2 c_se = decompose_f32(mid[ti * 8 + tj]);
        long c_sig = c_se.x; int c_exp = int(c_se.y);
        float fp = mid[ti * 8 + tj];

        long prod_sig[8];
        int  prod_exp[8];
        int max_e = c_exp;
        for (uint l = 0; l < 8; ++l) {
            float av = a_tile[ti * 16 + 8 + l];      // A[i, 8+l]
            float bv = b_tile[(8 + l) * 8 + tj];     // B[8+l, j]
            long2 a_se = decompose_narrow(av);
            long2 b_se = decompose_narrow(bv);
            fp += av * bv;
            long ps = a_se.x * b_se.x;
            int  pe = int(a_se.y) + int(b_se.y);
            prod_sig[l] = ps;
            prod_exp[l] = pe;
            max_e = max(max_e, pe);
        }
        long acc = trunc_shr(c_sig, max_e - c_exp);
        for (uint l = 0; l < 8; ++l) {
            int shift = max_e - prod_exp[l] + NFB;
            acc += trunc_shr(prod_sig[l], shift);
        }
        d_tile[ti * 8 + tj] = normalize_f32(acc, max_e, fp);
    }
}
"#;

struct MetalContext {
    device: Device,
    pipeline: ComputePipelineState,
    queue: metal::CommandQueue,
}

// SAFETY: metal handles are internally Arc'd / thread-safe for the ops we do
// (buffer creation, enqueue). We only use them immutably after init.
unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

static CTX: OnceLock<MetalContext> = OnceLock::new();

fn ctx() -> &'static MetalContext {
    CTX.get_or_init(|| {
        let device = Device::system_default()
            .expect("no Metal device (Apple Silicon / macOS required)");
        let options = CompileOptions::new();
        let library: Library = device
            .new_library_with_source(MSL_SOURCE, &options)
            .unwrap_or_else(|e| panic!("MSL compile failed: {e}"));
        let kernel = library
            .get_function(KERNEL_NAME, None)
            .expect("kernel not found");
        let desc = ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&kernel));
        let pipeline = device
            .new_compute_pipeline_state(&desc)
            .expect("pipeline creation failed");
        let queue = device.new_command_queue();
        MetalContext { device, pipeline, queue }
    })
}

/// Run the Metal kernel on a batch of ampere-f16 tiles.
/// Shapes: A=(batch, 16, 16), B=(batch, 16, 8), C=(batch, 16, 8).
/// Output: (batch, 16, 8) f32.
pub fn run_batched(a: &[f32], b: &[f32], c: &[f32], batch: usize) -> Vec<f32> {
    let ctx = ctx();
    let bytes_a = std::mem::size_of_val(a);
    let bytes_b = std::mem::size_of_val(b);
    let bytes_c = std::mem::size_of_val(c);
    let n_out = batch * 16 * 8;
    let bytes_d = n_out * std::mem::size_of::<f32>();

    // Shared storage mode = zero-copy on unified memory.
    let buf_a = ctx.device.new_buffer_with_data(
        a.as_ptr() as *const _, bytes_a as u64, MTLResourceOptions::StorageModeShared);
    let buf_b = ctx.device.new_buffer_with_data(
        b.as_ptr() as *const _, bytes_b as u64, MTLResourceOptions::StorageModeShared);
    let buf_c = ctx.device.new_buffer_with_data(
        c.as_ptr() as *const _, bytes_c as u64, MTLResourceOptions::StorageModeShared);
    let buf_d = ctx.device.new_buffer(bytes_d as u64, MTLResourceOptions::StorageModeShared);

    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.pipeline);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_c), 0);
    enc.set_buffer(3, Some(&buf_d), 0);
    enc.set_threadgroup_memory_length(0, (16 * 8 * 4) as u64); // mid[16*8] f32

    // One threadgroup per tile; 8 x 16 = 128 threads per group.
    let grid = MTLSize::new(batch as u64, 1, 1);
    let tg = MTLSize::new(8, 16, 1);
    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Copy output back into a Vec<f32>.
    let ptr = buf_d.contents() as *const f32;
    let out = unsafe { std::slice::from_raw_parts(ptr, n_out) }.to_vec();
    out
}
