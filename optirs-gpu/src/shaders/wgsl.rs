//! WGSL sources for the optimizer kernels (WebGPU backend).
//!
//! See [`super`] for the buffer-naming and scalar-packing conventions. In
//! short: bindings are named `x`, `y`, `a`, `b`, `result`, `output` in that
//! order, every scalar travels in a storage buffer, and integer scalars are
//! bit-cast through an `f32` slot.

/// Adam with coupled L2 weight decay, matching `optirs_core::optimizers::Adam`.
///
/// | binding | name | meaning |
/// |---|---|---|
/// | 0 | `x` | parameters (read-write) |
/// | 1 | `y` | gradients (read) |
/// | 2 | `a` | first moment `m` (read-write) |
/// | 3 | `b` | second moment `v` (read-write) |
/// | 4 | `result` | `[lr, beta1, beta2, eps, weight_decay, bc1, bc2, n]` |
pub const ADAM: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> a: array<f32>;
@group(0) @binding(3) var<storage, read_write> b: array<f32>;
@group(0) @binding(4) var<storage, read> result: array<f32>;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = bitcast<u32>(result[7]);
    let idx = gid.x;
    if (idx >= n) { return; }

    let lr = result[0];
    let beta1 = result[1];
    let beta2 = result[2];
    let eps = result[3];
    let wd = result[4];
    let bc1 = result[5];
    let bc2 = result[6];

    let p = x[idx];
    var g = y[idx];
    if (wd > 0.0) { g = g + wd * p; }

    let mi = beta1 * a[idx] + (1.0 - beta1) * g;
    let vi = beta2 * b[idx] + (1.0 - beta2) * g * g;
    a[idx] = mi;
    b[idx] = vi;

    let m_hat = mi / bc1;
    let v_hat = vi / bc2;
    x[idx] = p - lr * m_hat / (sqrt(v_hat) + eps);
}
"#;

/// AdamW with *decoupled* weight decay (Loshchilov & Hutter).
///
/// The decay never enters the moment estimates; it is applied straight to the
/// parameter, which is exactly what separates AdamW from Adam + L2.
///
/// Bindings are identical to [`ADAM`].
pub const ADAMW: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> a: array<f32>;
@group(0) @binding(3) var<storage, read_write> b: array<f32>;
@group(0) @binding(4) var<storage, read> result: array<f32>;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = bitcast<u32>(result[7]);
    let idx = gid.x;
    if (idx >= n) { return; }

    let lr = result[0];
    let beta1 = result[1];
    let beta2 = result[2];
    let eps = result[3];
    let wd = result[4];
    let bc1 = result[5];
    let bc2 = result[6];

    let p = x[idx];
    let g = y[idx];

    let mi = beta1 * a[idx] + (1.0 - beta1) * g;
    let vi = beta2 * b[idx] + (1.0 - beta2) * g * g;
    a[idx] = mi;
    b[idx] = vi;

    let m_hat = mi / bc1;
    let v_hat = vi / bc2;

    let decayed = p - lr * wd * p;
    x[idx] = decayed - lr * m_hat / (sqrt(v_hat) + eps);
}
"#;

/// SGD with optional momentum, dampening, Nesterov acceleration and L2 decay.
///
/// | binding | name | meaning |
/// |---|---|---|
/// | 0 | `x` | parameters (read-write) |
/// | 1 | `y` | gradients (read) |
/// | 2 | `a` | momentum buffer (read-write) |
/// | 3 | `b` | `[lr, momentum, dampening, weight_decay, nesterov, first_step, n]` |
///
/// `first_step` is `1.0` only on the very first update, so the momentum buffer
/// is seeded with the raw gradient rather than a dampened one.
pub const SGD: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> a: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = bitcast<u32>(b[6]);
    let idx = gid.x;
    if (idx >= n) { return; }

    let lr = b[0];
    let momentum = b[1];
    let dampening = b[2];
    let wd = b[3];
    let nesterov = b[4];
    let first = b[5];

    let p = x[idx];
    var g = y[idx];
    if (wd > 0.0) { g = g + wd * p; }

    if (momentum > 0.0) {
        var buf = g;
        if (first < 0.5) { buf = momentum * a[idx] + (1.0 - dampening) * g; }
        a[idx] = buf;
        if (nesterov > 0.5) { g = g + momentum * buf; } else { g = buf; }
    }

    x[idx] = p - lr * g;
}
"#;

/// RMSprop with optional centering and momentum (PyTorch semantics).
///
/// | binding | name | meaning |
/// |---|---|---|
/// | 0 | `x` | parameters (read-write) |
/// | 1 | `y` | gradients (read) |
/// | 2 | `a` | squared-gradient average (read-write) |
/// | 3 | `b` | mean-gradient average, used only when centered (read-write) |
/// | 4 | `result` | momentum buffer (read-write) |
/// | 5 | `output` | `[lr, alpha, eps, weight_decay, momentum, centered, n]` |
pub const RMSPROP: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> a: array<f32>;
@group(0) @binding(3) var<storage, read_write> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> result: array<f32>;
@group(0) @binding(5) var<storage, read> output: array<f32>;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = bitcast<u32>(output[6]);
    let idx = gid.x;
    if (idx >= n) { return; }

    let lr = output[0];
    let alpha = output[1];
    let eps = output[2];
    let wd = output[3];
    let momentum = output[4];
    let centered = output[5];

    let p = x[idx];
    var g = y[idx];
    if (wd > 0.0) { g = g + wd * p; }

    let sq = alpha * a[idx] + (1.0 - alpha) * g * g;
    a[idx] = sq;

    var avg = sq;
    if (centered > 0.5) {
        let ga = alpha * b[idx] + (1.0 - alpha) * g;
        b[idx] = ga;
        avg = sq - ga * ga;
    }

    let denom = sqrt(max(avg, 0.0)) + eps;

    if (momentum > 0.0) {
        let buf = momentum * result[idx] + g / denom;
        result[idx] = buf;
        x[idx] = p - lr * buf;
    } else {
        x[idx] = p - lr * g / denom;
    }
}
"#;

/// Adagrad with learning-rate decay.
///
/// | binding | name | meaning |
/// |---|---|---|
/// | 0 | `x` | parameters (read-write) |
/// | 1 | `y` | gradients (read) |
/// | 2 | `a` | accumulated squared gradients (read-write) |
/// | 3 | `b` | `[clr, eps, weight_decay, n]` |
///
/// The host passes the already-decayed step size `clr`; the decay denominator
/// counts *completed* steps, so the first update uses exactly `lr`.
pub const ADAGRAD: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> a: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = bitcast<u32>(b[3]);
    let idx = gid.x;
    if (idx >= n) { return; }

    let clr = b[0];
    let eps = b[1];
    let wd = b[2];

    let p = x[idx];
    var g = y[idx];
    if (wd > 0.0) { g = g + wd * p; }

    let s = a[idx] + g * g;
    a[idx] = s;
    x[idx] = p - clr * g / (sqrt(s) + eps);
}
"#;

/// LAMB, run as two dispatches of the *same* pipeline.
///
/// | binding | name | meaning |
/// |---|---|---|
/// | 0 | `x` | parameters (read-write) |
/// | 1 | `y` | gradients (read) |
/// | 2 | `a` | first moment `m` (read-write) |
/// | 3 | `b` | second moment `v` (read-write) |
/// | 4 | `result` | scratch: `update[0..n]` then `partials[n..n+2*groups]` |
/// | 5 | `output` | `[lr, beta1, beta2, eps, weight_decay, bc1, bc2, n, phase, trust]` |
///
/// * `phase == 0` advances the moments, materialises
///   `update = m_hat / (sqrt(v_hat) + eps) + weight_decay * p`, and emits
///   per-workgroup partial sums of `p^2` and `update^2`.
/// * `phase == 1` applies `p -= lr * trust * update`, with the trust ratio
///   computed on the host from the finished reduction.
///
/// One pipeline serves both phases because `GpuCompiler::compile` in
/// scirs2-core 0.6.5 registers every compiled WGSL shader under one internal
/// name per context, so two live handles would alias.
///
/// The workgroup reduction sits *outside* the `idx < n` guard so that
/// `workgroupBarrier()` is only ever reached in uniform control flow.
pub const LAMB: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> a: array<f32>;
@group(0) @binding(3) var<storage, read_write> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> result: array<f32>;
@group(0) @binding(5) var<storage, read> output: array<f32>;

var<workgroup> scratch_p: array<f32, 256>;
var<workgroup> scratch_u: array<f32, 256>;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wgid: vec3<u32>) {
    let n = bitcast<u32>(output[7]);
    let phase = bitcast<u32>(output[8]);
    let idx = gid.x;

    let lr = output[0];
    let beta1 = output[1];
    let beta2 = output[2];
    let eps = output[3];
    let wd = output[4];
    let bc1 = output[5];
    let bc2 = output[6];
    let trust = output[9];

    var sum_p = 0.0;
    var sum_u = 0.0;

    if (idx < n) {
        if (phase == 0u) {
            let p = x[idx];
            let g = y[idx];
            let mi = beta1 * a[idx] + (1.0 - beta1) * g;
            let vi = beta2 * b[idx] + (1.0 - beta2) * g * g;
            a[idx] = mi;
            b[idx] = vi;
            let m_hat = mi / bc1;
            let v_hat = vi / bc2;
            result[idx] = m_hat / (sqrt(v_hat) + eps) + wd * p;
        } else {
            x[idx] = x[idx] - lr * trust * result[idx];
        }
        let pv = x[idx];
        let uv = result[idx];
        sum_p = pv * pv;
        sum_u = uv * uv;
    }

    scratch_p[lid.x] = sum_p;
    scratch_u[lid.x] = sum_u;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (lid.x < stride) {
            scratch_p[lid.x] = scratch_p[lid.x] + scratch_p[lid.x + stride];
            scratch_u[lid.x] = scratch_u[lid.x] + scratch_u[lid.x + stride];
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride >> 1u;
    }

    if (lid.x == 0u) {
        result[n + wgid.x * 2u] = scratch_p[0];
        result[n + wgid.x * 2u + 1u] = scratch_u[0];
    }
}
"#;
