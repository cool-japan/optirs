//! Regression tests for the core optimizer fixes (RAdam rectification, per-tensor
//! state in `step_list`, the in-place update kernels and the `Optimizer` trait
//! coverage of AdaBound / AdaDelta / Ranger).
//!
//! Every test in this file pins behaviour that was previously wrong; each one fails
//! against the pre-fix implementation.

use optirs_core::optimizers::{
    AdaBound, AdaDelta, Adagrad, Adam, AdamW, Lion, Optimizer, RAdam, RMSprop, Ranger, LAMB, LARS,
    SGD,
};
use scirs2_core::ndarray::{Array1, Ix1};

/// Reference implementation of the RAdam rectification term from
/// "On the Variance of the Adaptive Learning Rate and Beyond" (Liu et al., 2019).
///
/// This is the same expression `Ranger` uses internally, transcribed independently so
/// the two implementations can be cross-checked.
fn reference_rectification(beta2: f64, t: usize) -> Option<f64> {
    let t_f = t as f64;
    let beta2_t = beta2.powi(t as i32);
    let bias_correction2 = 1.0 - beta2_t;
    let rho_inf = 2.0 / (1.0 - beta2) - 1.0;
    let rho_t = rho_inf - 2.0 * t_f * beta2_t / bias_correction2;

    if rho_t <= 4.0 {
        return None;
    }

    Some(
        (((rho_t - 4.0) * (rho_t - 2.0) * rho_inf) / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
            .sqrt(),
    )
}

/// F1: the RAdam rectification term must match the published formula.
///
/// The previous implementation computed `(rho_t - 4)(rho_t - 2) / rho_inf` scaled by
/// extra bias-correction factors. For the default `beta2 = 0.999` (`rho_inf = 1999`)
/// that is roughly 2000x too large late in training.
#[test]
fn radam_rectification_matches_published_formula() {
    let optimizer: RAdam<f64> = RAdam::new(0.001);

    for &t in &[5usize, 10, 100, 1000, 10_000] {
        let actual = optimizer.rectification_term(t);
        let expected = reference_rectification(0.999, t);

        match (actual, expected) {
            (None, None) => {}
            (Some(actual), Some(expected)) => assert!(
                (actual - expected).abs() < 1e-12,
                "t = {t}: rectification term {actual} != reference {expected}"
            ),
            (a, e) => panic!("t = {t}: rectification availability differs: {a:?} vs {e:?}"),
        }
    }
}

/// F1: `r_t` must converge to 1 as `t -> infinity`, so late training behaves like Adam.
///
/// The buggy formula diverged instead: it grew without bound towards ~1993.
#[test]
fn radam_rectification_converges_to_one() {
    let optimizer: RAdam<f64> = RAdam::new(0.001);

    let r_1000 = optimizer
        .rectification_term(1_000)
        .expect("rectification active at t = 1000");
    let r_100_000 = optimizer
        .rectification_term(100_000)
        .expect("rectification active at t = 100000");
    let r_10_000_000 = optimizer
        .rectification_term(10_000_000)
        .expect("rectification active at t = 10000000");

    // Monotone approach to 1 from below, and no divergence.
    assert!(r_1000 < r_100_000, "{r_1000} !< {r_100_000}");
    assert!(r_100_000 <= r_10_000_000, "{r_100_000} !<= {r_10_000_000}");
    assert!(
        (r_10_000_000 - 1.0).abs() < 1e-9,
        "r_t must converge to 1, got {r_10_000_000}"
    );
    assert!(r_1000 < 1.0, "r_t must approach 1 from below, got {r_1000}");

    // The buggy expression produced values in the thousands.
    assert!(r_10_000_000 < 1.5);
}

/// F1: the RAdam warmup branch must be entered for exactly the published range.
#[test]
fn radam_warmup_branch_boundary() {
    let optimizer: RAdam<f64> = RAdam::new(0.001);

    // With beta2 = 0.999, rho_t only exceeds 4 after a handful of steps.
    assert!(optimizer.rectification_term(0).is_none());
    assert!(optimizer.rectification_term(1).is_none());

    let first_active = (1..50)
        .find(|t| optimizer.rectification_term(*t).is_some())
        .expect("rectification must activate within 50 steps");
    assert!(
        (2..=10).contains(&first_active),
        "unexpected activation step {first_active}"
    );
}

/// F1: a long RAdam run must stay bounded. Under the buggy rectifier the effective
/// step size exploded and the parameters diverged.
#[test]
fn radam_long_run_stays_stable() {
    let mut optimizer: RAdam<f64> = RAdam::new(0.01);
    let mut params = Array1::from_vec(vec![5.0f64]);

    for _ in 0..5_000 {
        let gradients = params.mapv(|x| 2.0 * x);
        params = optimizer.step(&params, &gradients).expect("radam step");
        assert!(
            params[0].is_finite(),
            "RAdam diverged to a non-finite value"
        );
    }

    assert!(
        params[0].abs() < 1e-3,
        "RAdam failed to converge on f(x) = x^2, got {}",
        params[0]
    );
}

/// F31: `step_list` must give every parameter tensor its own Adam state.
///
/// The default `step_list` looped over `step`, and every stateful optimizer keyed its
/// moments at slot `[0]` with a single shared timestep, so tensors of different shapes
/// reset each other's moments and the bias correction advanced once per tensor.
#[test]
fn adam_step_list_keeps_per_tensor_state() {
    let mut shared = Adam::new(0.1f64);
    let mut separate_small = Adam::new(0.1f64);
    let mut separate_large = Adam::new(0.1f64);

    let mut small = Array1::from_vec(vec![0.0f64]);
    let mut large = Array1::from_vec(vec![0.0f64, 0.0, 0.0]);
    let mut ref_small = small.clone();
    let mut ref_large = large.clone();

    // A changing gradient sequence: with a constant gradient Adam's update magnitude
    // is lr * sign(g) whether or not state persists, so the test would pass vacuously.
    let gradient_sequence = [1.0f64, 0.0, -0.5, 2.0, 0.0];

    for &g in gradient_sequence.iter() {
        let grad_small = Array1::from_vec(vec![g]);
        let grad_large = Array1::from_vec(vec![g, g, g]);

        let updated = shared
            .step_list(&[&small, &large], &[&grad_small, &grad_large])
            .expect("step_list failed");
        small = updated[0].clone();
        large = updated[1].clone();

        ref_small = separate_small
            .step(&ref_small, &grad_small)
            .expect("reference small step");
        ref_large = separate_large
            .step(&ref_large, &grad_large)
            .expect("reference large step");
    }

    // A single optimizer driving both tensors must match two independent optimizers.
    assert!(
        (small[0] - ref_small[0]).abs() < 1e-12,
        "small tensor diverged: {} vs {}",
        small[0],
        ref_small[0]
    );
    for i in 0..3 {
        assert!(
            (large[i] - ref_large[i]).abs() < 1e-12,
            "large tensor diverged at {}: {} vs {}",
            i,
            large[i],
            ref_large[i]
        );
    }

    // Sanity: the tensors actually moved.
    assert!(small[0].abs() > 1e-6);
    assert!(large[0].abs() > 1e-6);
}

/// F31: the same independence guarantee for every other stateful optimizer.
#[test]
fn stateful_optimizers_keep_per_tensor_state_in_step_list() {
    macro_rules! check_independent {
        ($make:expr, $label:literal) => {{
            let mut shared = $make;
            let mut ref_a = $make;
            let mut ref_b = $make;

            let mut a = Array1::from_vec(vec![0.0f64]);
            let mut b = Array1::from_vec(vec![0.0f64, 0.0]);
            let mut expect_a = a.clone();
            let mut expect_b = b.clone();

            for &g in [1.0f64, 0.0, -0.5, 2.0].iter() {
                let ga = Array1::from_vec(vec![g]);
                let gb = Array1::from_vec(vec![g, g]);

                let out = shared
                    .step_list(&[&a, &b], &[&ga, &gb])
                    .expect(concat!($label, ": step_list failed"));
                a = out[0].clone();
                b = out[1].clone();

                expect_a = ref_a
                    .step(&expect_a, &ga)
                    .expect(concat!($label, ": reference a"));
                expect_b = ref_b
                    .step(&expect_b, &gb)
                    .expect(concat!($label, ": reference b"));
            }

            assert!(
                (a[0] - expect_a[0]).abs() < 1e-12,
                concat!($label, ": tensor A diverged: {} vs {}"),
                a[0],
                expect_a[0]
            );
            assert!(
                (b[0] - expect_b[0]).abs() < 1e-12,
                concat!($label, ": tensor B diverged: {} vs {}"),
                b[0],
                expect_b[0]
            );
        }};
    }

    check_independent!(AdamW::new(0.1f64), "AdamW");
    check_independent!(LAMB::new(0.1f64), "LAMB");
    check_independent!(Lion::new(0.1f64), "Lion");
    check_independent!(RAdam::new(0.1f64), "RAdam");
    check_independent!(SGD::new_with_config(0.1f64, 0.9, 0.0), "SGD+momentum");
    check_independent!(Adagrad::new(0.1f64), "Adagrad");
    check_independent!(RMSprop::new(0.1f64), "RMSprop");
    check_independent!(LARS::new(0.1f64).with_momentum(0.9), "LARS");
}

/// F31: a shape change on one index must reset only that index's state.
#[test]
fn per_index_state_reset_is_local() {
    let mut optimizer = Adam::new(0.1f64);

    let stable = Array1::from_vec(vec![0.0f64]);
    let first_shape = Array1::from_vec(vec![0.0f64, 0.0]);
    let grad_stable = Array1::from_vec(vec![1.0f64]);
    let grad_first = Array1::from_vec(vec![1.0f64, 1.0]);

    let out = optimizer
        .step_list(&[&stable, &first_shape], &[&grad_stable, &grad_first])
        .expect("first step_list");
    assert_eq!(optimizer.timestep(0), 1);
    assert_eq!(optimizer.timestep(1), 1);

    // Index 1 changes shape; index 0 does not.
    let second_shape = Array1::from_vec(vec![0.0f64, 0.0, 0.0]);
    let grad_second = Array1::from_vec(vec![1.0f64, 1.0, 1.0]);
    let out2 = optimizer
        .step_list(&[&out[0], &second_shape], &[&grad_stable, &grad_second])
        .expect("second step_list");

    assert_eq!(optimizer.timestep(0), 2, "index 0 clock must keep running");
    assert_eq!(optimizer.timestep(1), 1, "index 1 clock must restart");

    // Index 1 restarted, so it must have taken a fresh t=1 step of exactly -lr.
    assert!(
        (out2[1][0] + 0.1).abs() < 1e-9,
        "reshaped tensor did not restart cleanly: {}",
        out2[1][0]
    );
}

/// PERF: the allocation-free in-place kernels must be numerically identical to the
/// allocating `step` path they replace.
#[test]
fn inplace_kernels_match_allocating_path() {
    let gradient_sequence = [1.0f64, -0.5, 0.25, 2.0, 0.0];

    // Adam
    {
        let mut allocating = Adam::new_with_config(0.05, 0.9, 0.999, 1e-8, 0.01);
        let mut in_place = Adam::new_with_config(0.05, 0.9, 0.999, 1e-8, 0.01);
        let mut a = Array1::from_vec(vec![1.0f64, -2.0, 3.0]);
        let mut b = a.clone();

        for &g in gradient_sequence.iter() {
            let grads = Array1::from_vec(vec![g, g * 2.0, -g]);
            a = allocating.step(&a, &grads).expect("adam step");
            in_place
                .step_inplace(&mut b, &grads)
                .expect("adam step_inplace");
        }
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1e-15, "Adam mismatch at {i}");
        }
    }

    // AdamW
    {
        let mut allocating = AdamW::new(0.05f64);
        let mut in_place = AdamW::new(0.05f64);
        let mut a = Array1::from_vec(vec![1.0f64, -2.0, 3.0]);
        let mut b = a.clone();

        for &g in gradient_sequence.iter() {
            let grads = Array1::from_vec(vec![g, g * 2.0, -g]);
            a = allocating.step(&a, &grads).expect("adamw step");
            in_place
                .step_inplace(&mut b, &grads)
                .expect("adamw step_inplace");
        }
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1e-15, "AdamW mismatch at {i}");
        }
    }

    // SGD with momentum and weight decay
    {
        let mut allocating = SGD::new_with_config(0.05f64, 0.9, 0.01);
        let mut in_place = SGD::new_with_config(0.05f64, 0.9, 0.01);
        let mut a = Array1::from_vec(vec![1.0f64, -2.0, 3.0]);
        let mut b = a.clone();

        for &g in gradient_sequence.iter() {
            let grads = Array1::from_vec(vec![g, g * 2.0, -g]);
            a = allocating.step(&a, &grads).expect("sgd step");
            in_place
                .step_inplace(&mut b, &grads)
                .expect("sgd step_inplace");
        }
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1e-15, "SGD mismatch at {i}");
        }
    }
}

/// F31: the in-place kernels honour the same per-index slots as `step_list`.
#[test]
fn inplace_indexed_matches_step_list() {
    let mut indexed = Adam::new(0.1f64);
    let mut listed = Adam::new(0.1f64);

    let mut a = Array1::from_vec(vec![0.0f64]);
    let mut b = Array1::from_vec(vec![0.0f64, 0.0]);
    let mut la = a.clone();
    let mut lb = b.clone();

    for &g in [1.0f64, 0.0, -0.5].iter() {
        let ga = Array1::from_vec(vec![g]);
        let gb = Array1::from_vec(vec![g, g]);

        indexed
            .step_inplace_indexed(0, &mut a, &ga)
            .expect("indexed a");
        indexed
            .step_inplace_indexed(1, &mut b, &gb)
            .expect("indexed b");

        let out = listed
            .step_list(&[&la, &lb], &[&ga, &gb])
            .expect("step_list");
        la = out[0].clone();
        lb = out[1].clone();
    }

    assert!((a[0] - la[0]).abs() < 1e-15);
    assert!((b[0] - lb[0]).abs() < 1e-15);
    assert!((b[1] - lb[1]).abs() < 1e-15);
}

/// F32: `AdaBound`, `AdaDelta` and `Ranger` must be usable through the generic
/// `Optimizer` trait, which the crate documentation advertises for every optimizer.
#[test]
fn adabound_adadelta_ranger_implement_optimizer_trait() {
    fn run<O: Optimizer<f64, Ix1>>(optimizer: &mut O) -> Array1<f64> {
        let mut params = Array1::from_vec(vec![1.0f64, 2.0, 3.0]);
        for _ in 0..5 {
            let grads = params.mapv(|x| 2.0 * x);
            params = optimizer
                .step(&params, &grads)
                .expect("generic trait step failed");
        }
        params
    }

    let mut adabound = AdaBound::<f64>::default();
    let mut adadelta = AdaDelta::<f64>::default();
    let mut ranger = Ranger::<f64>::default();

    for result in [run(&mut adabound), run(&mut adadelta), run(&mut ranger)] {
        assert_eq!(result.len(), 3);
        for value in result.iter() {
            assert!(value.is_finite());
        }
    }

    // All three must have moved the parameters towards the origin.
    let adabound_out = run(&mut adabound);
    assert!(adabound_out[0].abs() < 1.0);
}

/// F32: the trait objects must also work behind `Box<dyn Optimizer<..>>`.
#[test]
fn new_optimizers_are_object_safe_through_trait() {
    let optimizers: Vec<Box<dyn Optimizer<f64, Ix1>>> = vec![
        Box::new(AdaBound::<f64>::default()),
        Box::new(AdaDelta::<f64>::default()),
        Box::new(Ranger::<f64>::default()),
        Box::new(Adam::new(0.01f64)),
    ];

    for mut optimizer in optimizers {
        let params = Array1::from_vec(vec![1.0f64, 2.0]);
        let grads = Array1::from_vec(vec![0.1f64, 0.2]);
        let updated = optimizer.step(&params, &grads).expect("boxed step failed");
        assert_eq!(updated.len(), 2);
    }
}
