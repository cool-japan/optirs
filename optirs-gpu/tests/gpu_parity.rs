//! GPU/CPU parity for the WGSL optimizer kernels.
//!
//! These tests need a real GPU adapter. When none is available (headless CI,
//! no Vulkan/Metal/DX12 loader) they print a `SKIP:` line and return rather
//! than failing — but when an adapter *is* present they assert real numbers, so
//! a silent no-op GPU path cannot pass.
//!
//! Run with `cargo nextest run -p optirs-gpu --no-capture` to see which branch
//! was taken.

mod cpu_ref;
use cpu_ref::Adam;
use optirs_gpu::optimizers::SUPPORTED_BACKENDS;
use optirs_gpu::optimizers::{
    AdagradParams, AdamParams, GpuAdagrad, GpuAdam, GpuAdamW, GpuLamb, GpuRmsprop, GpuSgd,
    RmspropParams, SgdParams,
};
use optirs_gpu::{GpuOptimError, GpuOptimizer};
use scirs2_core::gpu::{GpuBackend, GpuContext};
use scirs2_core::ndarray::Array1;

const N: usize = 1_500; // deliberately not a multiple of the 256-wide workgroup

/// Deterministic pseudo-random test data (no RNG dependency, reproducible).
fn sample(seed: u32, len: usize, scale: f32) -> Array1<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    Array1::from_shape_fn(len, |_| {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let unit = ((state >> 8) as f32) / ((1u32 << 24) as f32); // [0, 1)
        (unit - 0.5) * 2.0 * scale
    })
}

fn max_abs_diff(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// A usable backend, or the reasons every candidate failed.
fn probe_backend() -> Result<GpuBackend, String> {
    let mut reasons = Vec::new();
    for backend in SUPPORTED_BACKENDS {
        match GpuContext::new(backend) {
            Ok(context) => {
                assert_eq!(
                    context.backend(),
                    backend,
                    "context reported the wrong backend"
                );
                return Ok(backend);
            }
            Err(e) => reasons.push(format!("{backend}: {e}")),
        }
    }
    Err(reasons.join("; "))
}

/// Report why the GPU path is unavailable, or `None` if it is available.
fn skip_reason() -> Option<String> {
    probe_backend().err()
}

/// Open a context on the first usable backend.
fn open_context() -> GpuContext {
    let backend = probe_backend().expect("a backend was probed successfully above");
    GpuContext::new(backend).expect("backend was reachable a moment ago")
}

macro_rules! gpu_or_skip {
    ($test:literal) => {
        if let Some(reason) = skip_reason() {
            eprintln!("SKIP: {} — no usable GPU backend ({reason})", $test);
            return;
        }
    };
}

/// A device buffer must actually hold what was written to it.
///
/// scirs2-core silently substitutes a host-side fallback buffer when device
/// allocation fails, and that fallback's `copy_from_host` is a no-op. Without
/// this check a fallback would surface later as a wrong *number* rather than as
/// an error, so every parity assertion below rests on this one passing.
#[test]
fn device_buffers_round_trip() {
    gpu_or_skip!("device_buffers_round_trip");
    let context = open_context();
    eprintln!(
        "GPU: running on backend {} ({})",
        context.backend(),
        context.backend_name()
    );

    let data: Vec<f32> = (0..N).map(|i| i as f32 * 0.25 - 3.0).collect();
    let buffer = context.create_buffer::<f32>(data.len());
    buffer
        .copy_from_host(&data)
        .expect("upload to a real device buffer must succeed");

    let mut back = vec![0.0f32; data.len()];
    buffer
        .copy_to_host(&mut back)
        .expect("readback from a real device buffer must succeed");

    assert_eq!(
        back, data,
        "device buffer did not round-trip; scirs2-core probably handed back a host fallback buffer"
    );
}

/// A single GPU Adam step must match `optirs_core::optimizers::Adam`.
#[test]
fn gpu_adam_matches_cpu_adam_single_step() {
    gpu_or_skip!("gpu_adam_matches_cpu_adam_single_step");

    let hyper = AdamParams {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weight_decay: 0.0,
    };

    let params0 = sample(1, N, 1.0);
    let grads = sample(2, N, 0.1);

    let mut cpu = Adam::new_with_config(
        hyper.learning_rate,
        hyper.beta1,
        hyper.beta2,
        hyper.epsilon,
        hyper.weight_decay,
    );
    let mut cpu_params = params0.clone();
    cpu.step_inplace(&mut cpu_params, &grads)
        .expect("CPU Adam step");

    let mut gpu = GpuAdam::new(hyper).expect("GPU Adam construction");
    eprintln!("GPU Adam running on backend {}", gpu.backend());
    gpu.move_to_gpu().expect("to_gpu");
    let mut gpu_params = params0.clone();
    gpu.step_gpu(&mut gpu_params, &grads)
        .expect("GPU Adam step");

    let diff = max_abs_diff(&cpu_params, &gpu_params);
    eprintln!("GPU/CPU Adam max |Δ| after 1 step: {diff:e}");
    assert!(
        diff < 1e-5,
        "GPU Adam diverged from CPU Adam: max |Δ| = {diff:e}"
    );
    // Guard against a no-op kernel: the parameters must have actually moved.
    assert!(
        max_abs_diff(&params0, &gpu_params) > 0.0,
        "GPU Adam did not modify the parameters at all"
    );
}

/// Ten consecutive steps: this only stays in tolerance if the `m`/`v` state
/// really persists in device memory and the step counter drives bias
/// correction correctly.
#[test]
fn gpu_adam_matches_cpu_adam_over_many_steps() {
    gpu_or_skip!("gpu_adam_matches_cpu_adam_over_many_steps");

    let hyper = AdamParams {
        learning_rate: 5e-3,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weight_decay: 1e-2,
    };

    let mut cpu = Adam::new_with_config(
        hyper.learning_rate,
        hyper.beta1,
        hyper.beta2,
        hyper.epsilon,
        hyper.weight_decay,
    );
    let mut gpu = GpuAdam::new(hyper).expect("GPU Adam construction");
    gpu.move_to_gpu().expect("to_gpu");

    let mut cpu_params = sample(3, N, 1.0);
    let mut gpu_params = cpu_params.clone();

    for step in 1..=10u32 {
        let grads = sample(100 + step, N, 0.2);
        cpu.step_inplace(&mut cpu_params, &grads)
            .expect("CPU Adam step");
        gpu.step_gpu(&mut gpu_params, &grads)
            .expect("GPU Adam step");
    }

    let diff = max_abs_diff(&cpu_params, &gpu_params);
    eprintln!("GPU/CPU Adam max |Δ| after 10 steps: {diff:e}");
    assert!(
        diff < 1e-5,
        "GPU Adam diverged from CPU Adam over 10 steps: max |Δ| = {diff:e}"
    );
    assert_eq!(gpu.step_count(), 10);
}

/// `move_to_cpu` / `move_to_gpu` must move the moment estimates for real: a
/// round trip in the middle of training must not change the trajectory.
#[test]
fn optimizer_state_survives_a_host_round_trip() {
    gpu_or_skip!("optimizer_state_survives_a_host_round_trip");

    let hyper = AdamParams::default();
    let mut reference = GpuAdam::new(hyper).expect("GPU Adam construction");
    let mut roundtrip = GpuAdam::new(hyper).expect("GPU Adam construction");
    reference.move_to_gpu().expect("to_gpu");
    roundtrip.move_to_gpu().expect("to_gpu");

    let mut a = sample(7, N, 1.0);
    let mut b = a.clone();

    for step in 1..=3u32 {
        let grads = sample(200 + step, N, 0.3);
        reference.step_gpu(&mut a, &grads).expect("reference step");
        roundtrip.step_gpu(&mut b, &grads).expect("roundtrip step");
        if step == 2 {
            roundtrip.move_to_cpu().expect("to_cpu");
            roundtrip.move_to_gpu().expect("to_gpu again");
        }
    }

    assert_eq!(
        max_abs_diff(&a, &b),
        0.0,
        "moving optimizer state to the host and back changed the trajectory"
    );
}

/// Stepping while the optimizer is on the CPU is an error, not a silent no-op.
#[test]
fn step_without_move_to_gpu_is_an_error() {
    gpu_or_skip!("step_without_move_to_gpu_is_an_error");
    let mut gpu = GpuAdam::new(AdamParams::default()).expect("GPU Adam construction");
    let mut params = Array1::from_elem(16, 1.0f32);
    let grads = Array1::from_elem(16, 0.1f32);
    let err = gpu
        .step_gpu(&mut params, &grads)
        .expect_err("stepping on the CPU must fail");
    assert!(matches!(err, GpuOptimError::InvalidState(_)));
}

/// Every optimizer kernel compiles and produces a finite, non-trivial update.
#[test]
fn all_optimizer_kernels_dispatch() {
    gpu_or_skip!("all_optimizer_kernels_dispatch");

    let grads = sample(11, N, 0.5);
    let start = sample(12, N, 1.0);

    macro_rules! exercise {
        ($label:literal, $opt:expr) => {{
            let mut opt = $opt;
            opt.move_to_gpu().expect(concat!($label, ": to_gpu"));
            let mut params = start.clone();
            opt.step_gpu(&mut params, &grads)
                .expect(concat!($label, ": step_gpu"));
            assert!(
                params.iter().all(|v| v.is_finite()),
                concat!($label, " produced non-finite parameters")
            );
            let moved = max_abs_diff(&start, &params);
            eprintln!(concat!("GPU ", $label, ": max |Δp| = {:e}"), moved);
            assert!(moved > 0.0, concat!($label, " did not modify parameters"));
        }};
    }

    exercise!("adam", GpuAdam::new(AdamParams::default()).expect("adam"));
    exercise!(
        "adamw",
        GpuAdamW::new(AdamParams::default()).expect("adamw")
    );
    exercise!(
        "sgd",
        GpuSgd::new(SgdParams {
            momentum: 0.9,
            ..SgdParams::default()
        })
        .expect("sgd")
    );
    exercise!(
        "rmsprop",
        GpuRmsprop::new(RmspropParams {
            centered: true,
            momentum: 0.5,
            ..RmspropParams::default()
        })
        .expect("rmsprop")
    );
    exercise!(
        "adagrad",
        GpuAdagrad::new(AdagradParams {
            lr_decay: 0.1,
            ..AdagradParams::default()
        })
        .expect("adagrad")
    );
    exercise!("lamb", GpuLamb::new(AdamParams::default()).expect("lamb"));
}

/// AdamW's decoupled decay must differ from Adam's coupled L2 decay.
///
/// This is the check that the two shaders are genuinely different code, not a
/// copy of one another.
#[test]
fn adamw_decay_is_decoupled() {
    gpu_or_skip!("adamw_decay_is_decoupled");

    let hyper = AdamParams {
        learning_rate: 1e-2,
        weight_decay: 0.1,
        ..AdamParams::default()
    };
    let start = sample(21, N, 1.0);
    let grads = sample(22, N, 0.2);

    let mut adam = GpuAdam::new(hyper).expect("adam");
    let mut adamw = GpuAdamW::new(hyper).expect("adamw");
    adam.move_to_gpu().expect("to_gpu");
    adamw.move_to_gpu().expect("to_gpu");

    let mut a = start.clone();
    let mut w = start.clone();
    adam.step_gpu(&mut a, &grads).expect("adam step");
    adamw.step_gpu(&mut w, &grads).expect("adamw step");

    let diff = max_abs_diff(&a, &w);
    eprintln!("Adam vs AdamW max |Δ| with weight_decay=0.1: {diff:e}");
    assert!(
        diff > 1e-6,
        "AdamW produced the same update as Adam, so its decay is not decoupled"
    );
}

/// LAMB's trust ratio must actually rescale the step relative to plain Adam.
#[test]
fn lamb_applies_a_trust_ratio() {
    gpu_or_skip!("lamb_applies_a_trust_ratio");

    let hyper = AdamParams {
        learning_rate: 1e-2,
        ..AdamParams::default()
    };
    let start = sample(31, N, 4.0);
    let grads = sample(32, N, 0.2);

    let mut adam = GpuAdam::new(hyper).expect("adam");
    let mut lamb = GpuLamb::new(hyper).expect("lamb");
    adam.move_to_gpu().expect("to_gpu");
    lamb.move_to_gpu().expect("to_gpu");

    let mut a = start.clone();
    let mut l = start.clone();
    adam.step_gpu(&mut a, &grads).expect("adam step");
    lamb.step_gpu(&mut l, &grads).expect("lamb step");

    let adam_step = max_abs_diff(&start, &a);
    let lamb_step = max_abs_diff(&start, &l);
    eprintln!("Adam |Δp| = {adam_step:e}, LAMB |Δp| = {lamb_step:e}");
    assert!(lamb_step > 0.0, "LAMB did not move the parameters");
    assert!(
        (lamb_step - adam_step).abs() > 1e-9,
        "LAMB step is identical to Adam's, so no trust ratio was applied"
    );
}

/// The deprecated `to_gpu`/`to_cpu` names (kept for 0.3.1-era callers after
/// the `move_to_gpu`/`move_to_cpu` rename) must still genuinely move state,
/// not silently no-op: a deprecated-API run and a current-API run started
/// from identical inputs must land on bit-for-bit identical parameters.
#[test]
#[allow(deprecated)]
fn deprecated_to_gpu_to_cpu_still_delegate_to_move_to_gpu_move_to_cpu() {
    gpu_or_skip!("deprecated_to_gpu_to_cpu_still_delegate_to_move_to_gpu_move_to_cpu");

    let hyper = AdamParams {
        learning_rate: 1e-2,
        ..AdamParams::default()
    };
    let start = sample(41, N, 1.0);
    let grads = sample(42, N, 0.2);

    let mut via_deprecated = GpuAdam::new(hyper).expect("adam (deprecated path)");
    let mut via_current = GpuAdam::new(hyper).expect("adam (current path)");

    // The deprecated names, exercised end to end: upload, step, download,
    // re-upload -- exactly the sequence a pre-rename caller would have run.
    via_deprecated.to_gpu().expect("deprecated to_gpu");
    let mut p_deprecated = start.clone();
    via_deprecated
        .step_gpu(&mut p_deprecated, &grads)
        .expect("deprecated-path step");
    via_deprecated.to_cpu().expect("deprecated to_cpu");
    via_deprecated
        .to_gpu()
        .expect("deprecated to_gpu after round trip");

    // The current names, same sequence.
    via_current.move_to_gpu().expect("current move_to_gpu");
    let mut p_current = start.clone();
    via_current
        .step_gpu(&mut p_current, &grads)
        .expect("current-path step");
    via_current.move_to_cpu().expect("current move_to_cpu");
    via_current
        .move_to_gpu()
        .expect("current move_to_gpu after round trip");

    let deprecated_vs_current = max_abs_diff(&p_deprecated, &p_current);
    assert_eq!(
        deprecated_vs_current, 0.0,
        "the deprecated to_gpu/to_cpu shims diverged from move_to_gpu/move_to_cpu \
         -- the shim must delegate, not reimplement, the state transition"
    );
    assert!(
        max_abs_diff(&start, &p_current) > 0.0,
        "neither path actually moved the parameters -- a no-op shim would pass \
         the equality check above too, so this guards against exactly that"
    );
}
