//! Metal Shading Language sources for the optimizer kernels.
//!
//! These mirror [`super::wgsl`] line for line in semantics. They exist because
//! `scirs2-core` 0.6.5 registers its optimizer kernels with an *empty*
//! `metal_source`, and because the WebGPU backend's runtime device detection
//! does not yet enumerate wgpu adapters — on macOS the Metal backend is the
//! path that actually reaches the GPU.
//!
//! Conventions (see [`super`]): buffers are bound to argument-table indices in
//! the order `x, y, a, b, result, output`, every scalar travels in a buffer,
//! and integer scalars are recovered with `as_type<uint>`.
//!
//! scirs2-core's Metal compiler extracts the entry point with
//! `source.find("kernel void ")` up to the next `(`, so the name and the
//! opening parenthesis must stay on one line.

/// Adam with coupled L2 weight decay. See [`super::wgsl::ADAM`] for bindings.
pub const ADAM: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void optirs_adam(
    device float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* a [[buffer(2)]],
    device float* b [[buffer(3)]],
    device const float* result [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    uint n = as_type<uint>(result[7]);
    if (idx >= n) { return; }

    float lr = result[0];
    float beta1 = result[1];
    float beta2 = result[2];
    float eps = result[3];
    float wd = result[4];
    float bc1 = result[5];
    float bc2 = result[6];

    float p = x[idx];
    float g = y[idx];
    if (wd > 0.0f) { g = g + wd * p; }

    float mi = beta1 * a[idx] + (1.0f - beta1) * g;
    float vi = beta2 * b[idx] + (1.0f - beta2) * g * g;
    a[idx] = mi;
    b[idx] = vi;

    float m_hat = mi / bc1;
    float v_hat = vi / bc2;
    x[idx] = p - lr * m_hat / (sqrt(v_hat) + eps);
}
"#;

/// AdamW with decoupled weight decay. See [`super::wgsl::ADAMW`].
pub const ADAMW: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void optirs_adamw(
    device float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* a [[buffer(2)]],
    device float* b [[buffer(3)]],
    device const float* result [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    uint n = as_type<uint>(result[7]);
    if (idx >= n) { return; }

    float lr = result[0];
    float beta1 = result[1];
    float beta2 = result[2];
    float eps = result[3];
    float wd = result[4];
    float bc1 = result[5];
    float bc2 = result[6];

    float p = x[idx];
    float g = y[idx];

    float mi = beta1 * a[idx] + (1.0f - beta1) * g;
    float vi = beta2 * b[idx] + (1.0f - beta2) * g * g;
    a[idx] = mi;
    b[idx] = vi;

    float m_hat = mi / bc1;
    float v_hat = vi / bc2;

    float decayed = p - lr * wd * p;
    x[idx] = decayed - lr * m_hat / (sqrt(v_hat) + eps);
}
"#;

/// SGD with momentum / dampening / Nesterov. See [`super::wgsl::SGD`].
pub const SGD: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void optirs_sgd(
    device float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* a [[buffer(2)]],
    device const float* b [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    uint n = as_type<uint>(b[6]);
    if (idx >= n) { return; }

    float lr = b[0];
    float momentum = b[1];
    float dampening = b[2];
    float wd = b[3];
    float nesterov = b[4];
    float first = b[5];

    float p = x[idx];
    float g = y[idx];
    if (wd > 0.0f) { g = g + wd * p; }

    if (momentum > 0.0f) {
        float buf = g;
        if (first < 0.5f) { buf = momentum * a[idx] + (1.0f - dampening) * g; }
        a[idx] = buf;
        if (nesterov > 0.5f) { g = g + momentum * buf; } else { g = buf; }
    }

    x[idx] = p - lr * g;
}
"#;

/// RMSprop with centering and momentum. See [`super::wgsl::RMSPROP`].
pub const RMSPROP: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void optirs_rmsprop(
    device float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* a [[buffer(2)]],
    device float* b [[buffer(3)]],
    device float* result [[buffer(4)]],
    device const float* output [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    uint n = as_type<uint>(output[6]);
    if (idx >= n) { return; }

    float lr = output[0];
    float alpha = output[1];
    float eps = output[2];
    float wd = output[3];
    float momentum = output[4];
    float centered = output[5];

    float p = x[idx];
    float g = y[idx];
    if (wd > 0.0f) { g = g + wd * p; }

    float sq = alpha * a[idx] + (1.0f - alpha) * g * g;
    a[idx] = sq;

    float avg = sq;
    if (centered > 0.5f) {
        float ga = alpha * b[idx] + (1.0f - alpha) * g;
        b[idx] = ga;
        avg = sq - ga * ga;
    }

    float denom = sqrt(max(avg, 0.0f)) + eps;

    if (momentum > 0.0f) {
        float buf = momentum * result[idx] + g / denom;
        result[idx] = buf;
        x[idx] = p - lr * buf;
    } else {
        x[idx] = p - lr * g / denom;
    }
}
"#;

/// Adagrad with learning-rate decay. See [`super::wgsl::ADAGRAD`].
pub const ADAGRAD: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void optirs_adagrad(
    device float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* a [[buffer(2)]],
    device const float* b [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    uint n = as_type<uint>(b[3]);
    if (idx >= n) { return; }

    float clr = b[0];
    float eps = b[1];
    float wd = b[2];

    float p = x[idx];
    float g = y[idx];
    if (wd > 0.0f) { g = g + wd * p; }

    float s = a[idx] + g * g;
    a[idx] = s;
    x[idx] = p - clr * g / (sqrt(s) + eps);
}
"#;

/// Local reduction step for multi-GPU all-reduce-mean. See
/// [`super::wgsl::ALL_REDUCE_MEAN`].
pub const ALL_REDUCE_MEAN: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void optirs_all_reduce_mean(
    device float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    uint n = as_type<uint>(y[0]);
    if (idx >= n) { return; }

    uint num_gpus = as_type<uint>(y[1]);
    x[idx] = x[idx] / float(num_gpus);
}
"#;

/// Two-phase LAMB with a threadgroup norm reduction. See [`super::wgsl::LAMB`].
pub const LAMB: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void optirs_lamb(
    device float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* a [[buffer(2)]],
    device float* b [[buffer(3)]],
    device float* result [[buffer(4)]],
    device const float* output [[buffer(5)]],
    uint idx [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint wgid [[threadgroup_position_in_grid]])
{
    threadgroup float scratch_p[256];
    threadgroup float scratch_u[256];

    uint n = as_type<uint>(output[7]);
    uint phase = as_type<uint>(output[8]);

    float lr = output[0];
    float beta1 = output[1];
    float beta2 = output[2];
    float eps = output[3];
    float wd = output[4];
    float bc1 = output[5];
    float bc2 = output[6];
    float trust = output[9];

    float sum_p = 0.0f;
    float sum_u = 0.0f;

    if (idx < n) {
        if (phase == 0u) {
            float p = x[idx];
            float g = y[idx];
            float mi = beta1 * a[idx] + (1.0f - beta1) * g;
            float vi = beta2 * b[idx] + (1.0f - beta2) * g * g;
            a[idx] = mi;
            b[idx] = vi;
            float m_hat = mi / bc1;
            float v_hat = vi / bc2;
            result[idx] = m_hat / (sqrt(v_hat) + eps) + wd * p;
        } else {
            x[idx] = x[idx] - lr * trust * result[idx];
        }
        float pv = x[idx];
        float uv = result[idx];
        sum_p = pv * pv;
        sum_u = uv * uv;
    }

    scratch_p[lid] = sum_p;
    scratch_u[lid] = sum_u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            scratch_p[lid] = scratch_p[lid] + scratch_p[lid + stride];
            scratch_u[lid] = scratch_u[lid] + scratch_u[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0u) {
        result[n + wgid * 2u] = scratch_p[0];
        result[n + wgid * 2u + 1u] = scratch_u[0];
    }
}
"#;
