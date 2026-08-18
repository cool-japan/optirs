//! Regression tests for the numerical defects in `optirs-learned::higher_order`.
//!
//! These live outside `src/higher_order.rs` because that file is close to the
//! 2000-line cap; every entry point exercised here is public API re-exported from
//! the crate root.
//!
//! Each test corresponds to a specific defect that used to be present:
//!
//! * F69 — second-order stencils used the raw configured step (`1e-5`) instead of
//!   the machine-precision-aware order-2 step, which puts the roundoff floor at
//!   `ε/h²`: ~2e-6 in `f64` and ~1.2e3 in `f32`. The `f32` "Hessian" was noise.
//! * F70 — the truncated-Newton conjugate-gradient solve materialized a full
//!   Hessian column-by-column inside *every* CG iteration, `O(n²)` objective
//!   evaluations per `Hv` where a directional difference needs `O(n)`.
//! * F89 — five advertised differentiation modes over one finite-difference
//!   implementation (two of the bodies were byte-identical).
//! * K-FAC indexed its per-layer activation/gradient slices by the layer index
//!   without checking their lengths, so a short slice panicked.
//! * `jacobian_efficient` dispatched to a *one-sided* column routine for tall
//!   problems and a central row-wise routine for wide ones, silently changing the
//!   accuracy of the answer with the problem shape.

use optirs_learned::higher_order::{
    HessianConfig, HigherOrderConfig, HigherOrderEngine, HvpMode, LayerInfo, LayerType,
    MixedPartialMethod,
};
use scirs2_core::ndarray::{arr1, arr2, Array1, Array2};

fn engine<T>() -> HigherOrderEngine<T>
where
    T: scirs2_core::numeric::Float
        + std::fmt::Debug
        + Default
        + Clone
        + Send
        + Sync
        + 'static
        + std::iter::Sum
        + scirs2_core::ndarray::ScalarOperand,
{
    HigherOrderEngine::with_config(HigherOrderConfig::default())
}

/// `f(x) = ½ xᵀ A x` with `A = [[4, 1], [1, 3]]`; the Hessian is exactly `A`.
fn quadratic_f32(x: &Array1<f32>) -> f32 {
    let a = arr2(&[[4.0f32, 1.0], [1.0, 3.0]]);
    0.5 * x.dot(&a.dot(x))
}

fn quadratic_f64(x: &Array1<f64>) -> f64 {
    let a = arr2(&[[4.0f64, 1.0], [1.0, 3.0]]);
    0.5 * x.dot(&a.dot(x))
}

/// F69, `f32`. The exact Hessian is `[[4, 1], [1, 3]]`. With the old `h = 1e-5`
/// the `f32` roundoff floor is `ε/h² ≈ 1.2e-7 / 1e-10 ≈ 1.2e3`, i.e. every entry
/// was dominated by noise thousands of times larger than the signal. The
/// order-2 step for `f32` is `ε^(1/4) ≈ 1.9e-2`, putting the floor at ~3e-4.
#[test]
fn f32_second_order_stencils_are_not_dominated_by_roundoff() {
    let expected = arr2(&[[4.0f32, 1.0], [1.0, 3.0]]);
    let point = arr1(&[0.25f32, -0.5]);

    // Path 1: the materialized finite-difference Hessian, reached through the
    // public HVP entry point.
    let mut eng = engine::<f32>();
    for (column, unit) in [arr1(&[1.0f32, 0.0]), arr1(&[0.0f32, 1.0])]
        .into_iter()
        .enumerate()
    {
        let hv = eng
            .hessian_vector_product_advanced(
                quadratic_f32,
                &point,
                &unit,
                Some(HvpMode::MaterializedHessian),
            )
            .expect("materialized hvp");
        for row in 0..2 {
            let got = hv[row];
            let want = expected[[row, column]];
            assert!(
                (got - want).abs() < 5e-2,
                "materialized f32 Hessian column {column} row {row}: got {got}, want {want}"
            );
        }
    }

    // Path 2: the column-wise nested finite difference.
    let mut eng = engine::<f32>();
    let hessian = eng
        .hessian_reverse_over_forward(quadratic_f32, &point, &HessianConfig::default())
        .expect("column-wise hessian");
    for row in 0..2 {
        for col in 0..2 {
            let got = hessian[[row, col]];
            let want = expected[[row, col]];
            assert!(
                (got - want).abs() < 5e-2,
                "column-wise f32 Hessian [{row},{col}]: got {got}, want {want}"
            );
        }
    }
}

/// The two Hessian routines are named for the autodiff schemes they emulate, so
/// it is worth pinning that they really are two distinct algorithms that agree.
#[test]
fn the_two_hessian_routines_agree_on_a_known_quadratic() {
    let point = arr1(&[0.3f64, -0.7]);
    let expected = arr2(&[[4.0f64, 1.0], [1.0, 3.0]]);

    let mut eng = engine::<f64>();
    let row_wise = eng
        .hessian_forward_over_reverse(quadratic_f64, &point, &HessianConfig::default())
        .expect("row-wise");
    let mut eng = engine::<f64>();
    let col_wise = eng
        .hessian_reverse_over_forward(quadratic_f64, &point, &HessianConfig::default())
        .expect("column-wise");

    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (row_wise[[i, j]] - expected[[i, j]]).abs() < 1e-4,
                "row-wise [{i},{j}] = {}",
                row_wise[[i, j]]
            );
            assert!(
                (col_wise[[i, j]] - expected[[i, j]]).abs() < 1e-4,
                "column-wise [{i},{j}] = {}",
                col_wise[[i, j]]
            );
        }
    }
}

/// K-FAC used to do `activations[i]` / `gradients[i]` for `i` in `0..layers.len()`
/// with no length check at all.
#[test]
fn kfac_rejects_mismatched_per_layer_slices_instead_of_panicking() {
    let layer = LayerInfo {
        layer_type: LayerType::Linear,
        input_size: 3,
        output_size: 2,
        weights: Array2::<f64>::zeros((2, 3)),
        bias: None,
    };
    let layers = vec![layer.clone(), layer];
    let mut eng = engine::<f64>();

    // Two layers, one activation: previously an out-of-bounds panic.
    let err = eng
        .kfac_hessian_approximation(
            &layers,
            &[Array1::from_vec(vec![1.0, 2.0, 3.0])],
            &[
                Array1::from_vec(vec![0.1, 0.2]),
                Array1::from_vec(vec![0.3, 0.4]),
            ],
        )
        .expect_err("mismatched activation count must be an error");
    let text = err.to_string();
    assert!(
        text.contains("activation") || text.contains("K-FAC"),
        "unhelpful error: {text}"
    );

    // Two layers, one gradient.
    let err = eng
        .kfac_hessian_approximation(
            &layers,
            &[
                Array1::from_vec(vec![1.0, 2.0, 3.0]),
                Array1::from_vec(vec![1.0, 2.0, 3.0]),
            ],
            &[Array1::from_vec(vec![0.1, 0.2])],
        )
        .expect_err("mismatched gradient count must be an error");
    assert!(err.to_string().contains("K-FAC"), "{}", err);

    // Empty layer list.
    assert!(eng.kfac_hessian_approximation(&[], &[], &[]).is_err());

    // Right *count*, wrong *width*: the activation must be `input_size` long and
    // the gradient `output_size` long, because K-FAC factors the block as
    // `E[a aᵀ] ⊗ E[g gᵀ]`. Before this check the factor builders squared
    // whatever length they were handed, so a swapped pair still produced a
    // plausible matrix instead of an error.
    let err = eng
        .kfac_hessian_approximation(
            &layers,
            &[
                Array1::from_vec(vec![1.0, 2.0]), // input_size is 3, not 2
                Array1::from_vec(vec![1.0, 2.0, 3.0]),
            ],
            &[
                Array1::from_vec(vec![0.1, 0.2]),
                Array1::from_vec(vec![0.3, 0.4]),
            ],
        )
        .expect_err("activation shorter than input_size must be an error");
    let text = err.to_string();
    assert!(
        text.contains("input_size") && text.contains("Linear"),
        "error should name the offending dimension and layer type: {text}"
    );

    let err = eng
        .kfac_hessian_approximation(
            &layers,
            &[
                Array1::from_vec(vec![1.0, 2.0, 3.0]),
                Array1::from_vec(vec![1.0, 2.0, 3.0]),
            ],
            &[
                Array1::from_vec(vec![0.1, 0.2]),
                Array1::from_vec(vec![0.3, 0.4, 0.5]), // output_size is 2, not 3
            ],
        )
        .expect_err("gradient longer than output_size must be an error");
    assert!(
        err.to_string().contains("output_size"),
        "error should name output_size: {err}"
    );

    // The well-formed call must still succeed.
    assert!(eng
        .kfac_hessian_approximation(
            &layers,
            &[
                Array1::from_vec(vec![1.0, 2.0, 3.0]),
                Array1::from_vec(vec![1.0, 2.0, 3.0]),
            ],
            &[
                Array1::from_vec(vec![0.1, 0.2]),
                Array1::from_vec(vec![0.3, 0.4]),
            ],
        )
        .is_ok());
}

/// `jacobian_efficient` picks the column routine when `input_dim <= output_dim`
/// and the row routine otherwise. Both must now be second-order accurate, so the
/// accuracy of the answer no longer depends on the problem shape.
#[test]
fn jacobian_accuracy_does_not_depend_on_the_problem_shape() {
    // Wide (input 3 > output 2) -> row-wise path.
    // Tall (input 2 <= output 3) -> column-wise path.
    // Same underlying nonlinear map restricted appropriately, so the two paths
    // are compared against their own exact Jacobians.
    let mut eng = engine::<f64>();

    // g: R^3 -> R^2, g = (x0² x1, sin(x2) + x0)
    let wide = |x: &Array1<f64>| -> Array1<f64> { arr1(&[x[0] * x[0] * x[1], x[2].sin() + x[0]]) };
    let p3 = arr1(&[1.3f64, -0.7, 0.4]);
    let exact_wide = arr2(&[
        [2.0 * p3[0] * p3[1], p3[0] * p3[0], 0.0],
        [1.0, 0.0, p3[2].cos()],
    ]);
    let j_wide = eng.jacobian_efficient(wide, &p3, 2).expect("wide jacobian");

    // h: R^2 -> R^3, h = (x0² x1, sin(x1), x0 + x1)
    let tall =
        |x: &Array1<f64>| -> Array1<f64> { arr1(&[x[0] * x[0] * x[1], x[1].sin(), x[0] + x[1]]) };
    let p2 = arr1(&[1.3f64, -0.7]);
    let exact_tall = arr2(&[
        [2.0 * p2[0] * p2[1], p2[0] * p2[0]],
        [0.0, p2[1].cos()],
        [1.0, 1.0],
    ]);
    let j_tall = eng.jacobian_efficient(tall, &p2, 3).expect("tall jacobian");

    let max_err = |got: &Array2<f64>, want: &Array2<f64>| -> f64 {
        got.iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    };
    let wide_err = max_err(&j_wide, &exact_wide);
    let tall_err = max_err(&j_tall, &exact_tall);

    // A one-sided step at h = 1e-5 leaves an O(h·f'') truncation error of roughly
    // 1e-5 here; the central stencil is ~1e-10. Requiring both below 1e-7 is what
    // separates "both central" from "one of them one-sided".
    assert!(wide_err < 1e-7, "row-wise jacobian error {wide_err}");
    assert!(tall_err < 1e-7, "column-wise jacobian error {tall_err}");
}

/// F70. On a 40-dimensional quadratic the truncated-Newton direction must solve
/// `H d = -g`. The old implementation needed `O(n²)` objective evaluations for
/// every `Hv`, so this exact call did ~40 × 40 × 80 = 128,000 objective
/// evaluations *per CG iteration*; it now needs ~160.
#[test]
fn truncated_newton_solves_a_known_quadratic_system() {
    const N: usize = 40;
    // Diagonally dominant tridiagonal SPD matrix: H = 4I - (offdiag 1).
    let curvature = |i: usize, j: usize| -> f64 {
        if i == j {
            4.0
        } else if i.abs_diff(j) == 1 {
            -1.0
        } else {
            0.0
        }
    };
    let hessian = Array2::from_shape_fn((N, N), |(i, j)| curvature(i, j));
    // Count objective evaluations. This is what makes the test a real regression
    // test rather than a correctness pin: the old code needed `n` gradient pairs
    // (`4n²` objective calls) for every `Hv`, the new one needs one pair (`4n`).
    // `truncated_newton_direction` requires `Send + Sync`, so the counter is an
    // atomic rather than a `Cell`.
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let objective = {
        let h = hessian.clone();
        let calls = std::sync::Arc::clone(&calls);
        move |x: &Array1<f64>| -> f64 {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            0.5 * x.dot(&h.dot(x))
        }
    };

    let point = Array1::<f64>::zeros(N);
    let gradient = Array1::from_shape_fn(N, |i| ((i % 7) as f64) - 3.0);

    let max_cg = 60usize;
    let mut eng = engine::<f64>();
    let direction = eng
        .truncated_newton_direction(objective, &point, &gradient, max_cg, 1e-10)
        .expect("truncated newton");

    // Per CG iteration the new path costs 2 gradients = 2·2N = 4N objective
    // evaluations. The old path cost N gradient pairs per `Hv`, i.e. 4N² — a
    // factor of N = 40 more. Budgeting 8N per iteration leaves generous slack
    // while still being 5× under the old cost for a *single* iteration.
    let budget = 8 * N * (max_cg + 2);
    let used = calls.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        used <= budget,
        "the CG solve used {used} objective evaluations; budget is {budget} (the O(n²)-per-Hv \
         implementation needed at least {} for one iteration alone)",
        4 * N * N
    );

    // H d should equal -g.
    let residual = hessian.dot(&direction) + &gradient;
    let err = residual.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let scale = gradient.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        err < 1e-3 * scale.max(1.0),
        "residual {err} too large for gradient scale {scale}"
    );
    // And the direction must actually be a descent direction.
    assert!(
        direction.dot(&gradient) < 0.0,
        "truncated Newton returned an ascent direction"
    );
}

/// The public availability flags must line up with the dispatch behaviour, so a
/// caller can branch on them instead of catching an error.
#[test]
fn availability_flags_match_dispatch() {
    let point = arr1(&[0.5f64, 0.5]);
    let vector = arr1(&[1.0f64, 0.0]);

    for mode in [
        HvpMode::CentralDifference,
        HvpMode::ForwardDifference,
        HvpMode::QuadraticSecant,
        HvpMode::MaterializedHessian,
        HvpMode::NestedAutodiff,
    ] {
        let mut eng = engine::<f64>();
        let outcome =
            eng.hessian_vector_product_advanced(quadratic_f64, &point, &vector, Some(mode));
        assert_eq!(
            outcome.is_ok(),
            mode.is_available(),
            "{mode:?}: is_available() disagrees with the dispatch result"
        );
        assert_eq!(
            mode.objective_evaluations(8).is_some(),
            mode.is_available(),
            "{mode:?}: cost model disagrees with availability"
        );
    }

    for method in [
        MixedPartialMethod::FiniteDifference,
        MixedPartialMethod::ForwardFiniteDifference,
        MixedPartialMethod::NestedAutodiff,
    ] {
        let mut eng = engine::<f64>();
        let outcome = eng.mixed_partial(quadratic_f64, &point, &[0, 1], &[1, 1], method);
        assert_eq!(
            outcome.is_ok(),
            method.is_available(),
            "{method:?}: is_available() disagrees with the dispatch result"
        );
    }
}

/// `HigherOrderEngine`'s profiler was constructed but never fed: nothing called
/// `record_*`, so the public `get_derivative_stats().performance_profile`
/// reported `avg_hvp_time_us = 0`, `avg_jacobian_time_us = 0` and a hard-coded
/// `sparsity_benefit = 1.2` no matter what the engine had computed.
#[test]
fn derivative_stats_report_measured_work_not_placeholders() {
    let mut eng = engine::<f64>();

    // Nothing recorded yet.
    let before = eng.get_derivative_stats();
    assert_eq!(before.performance_profile.recorded_computations, 0);
    assert_eq!(before.performance_profile.avg_hessian_time_us, 0.0);
    assert_eq!(before.performance_profile.avg_hvp_time_us, 0.0);
    assert_eq!(before.performance_profile.avg_jacobian_time_us, 0.0);
    // With no observations at all, the two ratios are reported as the neutral
    // 1.0 rather than an invented speedup.
    assert_eq!(before.performance_profile.sparsity_benefit, 1.0);
    assert_eq!(before.performance_profile.parallel_efficiency, 1.0);

    let quadratic = |x: &Array1<f64>| 0.5 * (4.0 * x[0] * x[0] + 3.0 * x[1] * x[1]);
    let point = arr1(&[0.4, -0.9]);
    let vector = arr1(&[1.0, -2.0]);

    eng.hessian_forward_over_reverse(&quadratic, &point, &HessianConfig::default())
        .expect("hessian");
    eng.hessian_vector_product_advanced(
        &quadratic,
        &point,
        &vector,
        Some(HvpMode::CentralDifference),
    )
    .expect("hvp");
    eng.jacobian_efficient(
        |x: &Array1<f64>| arr1(&[x[0] * x[0], x[1], x[0] + x[1]]),
        &point,
        3,
    )
    .expect("jacobian");

    let after = eng.get_derivative_stats();
    assert_eq!(
        after.performance_profile.recorded_computations, 3,
        "one Hessian, one HVP and one Jacobian must each be recorded"
    );
    assert_eq!(
        after.performance_profile.largest_hessian_problem, 2,
        "the recorded Hessian problem size must be the parameter count"
    );
    // Durations are wall-clock and can round to zero microseconds on a fast
    // machine, so assert they are defined and non-negative rather than positive.
    for value in [
        after.performance_profile.avg_hessian_time_us,
        after.performance_profile.avg_hvp_time_us,
        after.performance_profile.avg_jacobian_time_us,
    ] {
        assert!(value.is_finite() && value >= 0.0, "bad duration {value}");
    }
    // Memory estimate must grow once timing records exist.
    assert!(after.memory_usage_estimate > 0);
}

/// An unavailable HVP mode must stay an error and must not be counted as work
/// the engine performed.
#[test]
fn an_unavailable_hvp_mode_is_not_recorded_as_a_computation() {
    let mut eng = engine::<f64>();
    let quadratic = |x: &Array1<f64>| x[0] * x[0] + x[1] * x[1];
    let point = arr1(&[1.0, 1.0]);
    let vector = arr1(&[1.0, 0.0]);

    assert!(eng
        .hessian_vector_product_advanced(&quadratic, &point, &vector, Some(HvpMode::NestedAutodiff))
        .is_err());
    assert_eq!(
        eng.get_derivative_stats()
            .performance_profile
            .recorded_computations,
        0
    );
}
