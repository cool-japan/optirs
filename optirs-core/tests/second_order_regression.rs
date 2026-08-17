//! Regression tests for the second-order optimizers.
//!
//! Every test in this file pins a specific correctness bug that used to be present in
//! `optirs_core::second_order` / `optirs_core::optimizers::LBFGS`. Each one fails
//! against the pre-fix behaviour and documents which numerical property is protected.

use std::collections::HashMap;
use std::collections::VecDeque;

use optirs_core::error::OptimError;
use optirs_core::optimizers::{Optimizer, LBFGS};
use optirs_core::second_order::hessian_approximation::{
    initial_hessian_scaling, lbfgs_two_loop_recursion, update_lbfgs_approximation,
};
use optirs_core::second_order::kfac::{KFACConfig, LayerInfo, LayerType, KFAC};
use optirs_core::second_order::newton_cg::NewtonCG;
use optirs_core::second_order::{HessianInfo, Newton, SecondOrderOptimizer};
use scirs2_core::ndarray::{array, Array1, Array2};

// ---------------------------------------------------------------------------
// F29: the L-BFGS curvature pair must use the true s = x_k - x_{k-1}
// ---------------------------------------------------------------------------

/// The optimizer used to *reconstruct* `s` by re-running its own direction
/// computation, which is only correct when the caller applies the returned parameters
/// verbatim. Here the caller overrides the parameters between steps (standing in for
/// projection, clipping or an external schedule), so a reconstructed `s` is simply
/// wrong; only a stored `prev_params` yields the true curvature pair.
#[test]
fn f29_lbfgs_curvature_pair_survives_external_parameter_edits() {
    // f(x) = 0.5 * (x0^2 + 25*x1^2), gradient = [x0, 25*x1].
    let grad = |x: &Array1<f64>| array![x[0], 25.0 * x[1]];

    let mut optimizer: LBFGS<f64> = LBFGS::new(1.0);

    // Step 1 from x0. The returned suggestion is discarded on purpose.
    let x0 = array![1.0_f64, 1.0];
    let g0 = grad(&x0);
    let suggested = optimizer.step(&x0, &g0).expect("first step succeeds");

    // The caller moves somewhere else entirely.
    let x1 = array![0.4_f64, -0.2];
    assert!(
        (suggested[0] - x1[0]).abs() > 1e-3,
        "the test only discriminates if the caller deviates from the suggestion"
    );
    let g1 = grad(&x1);
    let _ = optimizer.step(&x1, &g1).expect("second step succeeds");

    // Exactly one curvature pair, built from the *actual* move x1 - x0.
    assert_eq!(optimizer.history_len(), 1);
    let (s, y) = optimizer
        .last_curvature_pair()
        .expect("a curvature pair is stored");

    for i in 0..2 {
        assert!(
            (s[i] - (x1[i] - x0[i])).abs() < 1e-12,
            "s[{}] = {}, expected the true parameter difference {}",
            i,
            s[i],
            x1[i] - x0[i]
        );
        assert!(
            (y[i] - (g1[i] - g0[i])).abs() < 1e-12,
            "y[{}] = {}, expected the true gradient difference {}",
            i,
            y[i],
            g1[i] - g0[i]
        );
    }

    // gamma_k = (s·y)/(y·y) must follow from that same true pair.
    let sy: f64 = (0..2).map(|i| s[i] * y[i]).sum();
    let yy: f64 = (0..2).map(|i| y[i] * y[i]).sum();
    assert!(
        (optimizer.initial_hessian_scale() - sy / yy).abs() < 1e-12,
        "the initial Hessian scaling must be gamma_k = (s·y)/(y·y)"
    );

    // And the resulting direction must be a descent direction.
    let x2 = array![0.4_f64, -0.2];
    let g2 = grad(&x2);
    let x3 = optimizer.step(&x2, &g2).expect("third step succeeds");
    let directional: f64 = (0..2).map(|i| g2[i] * (x3[i] - x2[i])).sum();
    assert!(
        directional < 0.0,
        "the L-BFGS direction must descend, got directional derivative {}",
        directional
    );
}

/// A zero-length move must not manufacture a curvature pair.
#[test]
fn f29_lbfgs_rejects_degenerate_curvature_pair() {
    let mut optimizer: LBFGS<f64> = LBFGS::new(0.5);
    let x = array![3.0_f64, -1.0];
    let g = array![0.6_f64, -0.2];

    let _ = optimizer.step(&x, &g).expect("first step succeeds");
    // Same point, same gradient: s = 0 and y = 0, so y·s = 0 and nothing may be stored.
    let _ = optimizer.step(&x, &g).expect("second step succeeds");

    assert_eq!(
        optimizer.history_len(),
        0,
        "a pair with y·s = 0 destroys positive definiteness and must be skipped"
    );
}

// ---------------------------------------------------------------------------
// F30: line search must actually run
// ---------------------------------------------------------------------------

/// `step` cannot line search (no objective access), so an oversized learning rate
/// overshoots badly. `step_with_loss` backtracks until the Armijo sufficient-decrease
/// condition holds, and therefore never returns a worse point.
#[test]
fn f30_lbfgs_armijo_line_search_tames_an_oversized_learning_rate() {
    // f(x) = x^4 -> f'(x) = 4 x^3.
    let loss = |x: &Array1<f64>| x[0].powi(4);
    let x = array![1.0_f64];
    let g = array![4.0_f64];
    let f0 = loss(&x);

    // Fixed step size with a wildly oversized learning rate: overshoots.
    let mut fixed: LBFGS<f64> = LBFGS::new(100.0);
    let after_fixed = fixed.step(&x, &g).expect("fixed step succeeds");
    assert!(
        loss(&after_fixed) > f0,
        "the fixed-step path is expected to overshoot here (got {} from {})",
        loss(&after_fixed),
        f0
    );

    // Same learning rate, but with the backtracking Armijo line search.
    let mut searched: LBFGS<f64> = LBFGS::new(100.0);
    let after_search = searched
        .step_with_loss(&x, &g, loss)
        .expect("line searched step succeeds");
    assert!(
        loss(&after_search) < f0,
        "the Armijo line search must decrease the objective: {} -> {}",
        f0,
        loss(&after_search)
    );

    // The accepted step must actually satisfy the Armijo condition
    // f(x + a d) <= f(x) + c1 * a * g^T d, with d = -g and a = (x_new - x)/d.
    let alpha = (after_search[0] - x[0]) / (-g[0]);
    assert!(
        alpha > 0.0 && alpha < 100.0,
        "the step must have been backtracked"
    );
    let gtd = -g[0] * g[0];
    assert!(loss(&after_search) <= f0 + 1e-4 * alpha * gtd + 1e-12);
}

/// The line search must never return a point with a worse objective than the start.
#[test]
fn f30_lbfgs_line_search_is_monotone() {
    let loss = |x: &Array1<f64>| (x[0] - 2.0).powi(4) + (x[1] + 1.0).powi(2);
    let grad = |x: &Array1<f64>| array![4.0 * (x[0] - 2.0).powi(3), 2.0 * (x[1] + 1.0)];

    let mut optimizer: LBFGS<f64> = LBFGS::new(4.0);
    let mut x = array![-3.0_f64, 5.0];

    for _ in 0..40 {
        let before = loss(&x);
        let g = grad(&x);
        let next = optimizer
            .step_with_loss(&x, &g, loss)
            .expect("step succeeds");
        let after = loss(&next);
        assert!(
            after <= before + 1e-12,
            "objective increased: {} -> {}",
            before,
            after
        );
        x = next;
    }

    assert!(
        loss(&x) < 1.0,
        "should have made real progress from 661.0, got {} at {:?}",
        loss(&x),
        x
    );
}

/// `step_with_loss` rejects mismatched shapes rather than panicking.
#[test]
fn f30_lbfgs_step_with_loss_validates_shapes() {
    let mut optimizer: LBFGS<f64> = LBFGS::new(1.0);
    let params = array![1.0_f64, 2.0];
    let gradients = array![1.0_f64, 2.0, 3.0];
    let loss = |x: &Array1<f64>| x.iter().map(|v| v * v).sum::<f64>();

    let err = optimizer
        .step_with_loss(&params, &gradients, loss)
        .expect_err("shape mismatch must be an error");
    assert!(matches!(err, OptimError::DimensionMismatch(_)));
}

// ---------------------------------------------------------------------------
// F40: two-loop recursion must filter y·s <= 0 and use gamma_k = (s·y)/(y·y)
// ---------------------------------------------------------------------------

/// A hand-built `QuasiNewton` history bypasses the insertion-time filter, so the
/// recursion itself must reject non-positive curvature pairs. With the single stored
/// pair rejected the result falls back to the scaled gradient.
#[test]
fn f40_two_loop_recursion_skips_non_positive_curvature_pairs() {
    let gradient = array![1.0_f64, 2.0, 3.0];

    let mut s_history: VecDeque<Array1<f64>> = VecDeque::new();
    let mut y_history: VecDeque<Array1<f64>> = VecDeque::new();
    // y·s = -0.09 < 0: including this pair yields a negative rho and an ascent step.
    s_history.push_back(array![0.1_f64, 0.1, 0.1]);
    y_history.push_back(array![-0.2_f64, -0.3, -0.4]);

    let scale = 0.5_f64;
    let result = lbfgs_two_loop_recursion(&gradient, &s_history, &y_history, scale)
        .expect("recursion succeeds");

    // The bad pair is skipped, so H is just `scale * I`.
    for i in 0..gradient.len() {
        assert!(
            (result[i] - scale * gradient[i]).abs() < 1e-12,
            "expected the fallback-scaled gradient, got {:?}",
            result
        );
    }

    // The result must stay a descent direction: g·(H g) > 0 so that `x - lr * H g`
    // decreases f.
    let g_dot_hg: f64 = gradient.iter().zip(result.iter()).map(|(a, b)| a * b).sum();
    assert!(g_dot_hg > 0.0, "H must stay positive definite");
}

/// `initial_hessian_scaling` must return `gamma_k = (s·y)/(y·y)` from the newest
/// *acceptable* pair, ignoring a poisoned trailing pair.
#[test]
fn f40_initial_hessian_scaling_ignores_poisoned_trailing_pair() {
    let mut s_history: VecDeque<Array1<f64>> = VecDeque::new();
    let mut y_history: VecDeque<Array1<f64>> = VecDeque::new();

    // Good pair: s·y = 0.09, y·y = 0.29 -> gamma = 0.09/0.29.
    s_history.push_back(array![0.1_f64, 0.1, 0.1]);
    y_history.push_back(array![0.2_f64, 0.3, 0.4]);
    // Poisoned trailing pair (y·s < 0).
    s_history.push_back(array![1.0_f64, 0.0, 0.0]);
    y_history.push_back(array![-1.0_f64, 0.0, 0.0]);

    let gamma = initial_hessian_scaling(&s_history, &y_history).expect("a usable pair exists");
    assert!(
        (gamma - 0.09 / 0.29).abs() < 1e-12,
        "gamma_k must come from the newest acceptable pair, got {}",
        gamma
    );
}

/// The initial inverse-Hessian scale must be `gamma_k`, not a hardcoded constant.
#[test]
fn f40_two_loop_recursion_uses_gamma_k_scaling() {
    let gradient = array![1.0_f64, 2.0, 3.0];
    let mut s_history: VecDeque<Array1<f64>> = VecDeque::new();
    let mut y_history: VecDeque<Array1<f64>> = VecDeque::new();
    s_history.push_back(array![0.1_f64, 0.1, 0.1]);
    y_history.push_back(array![0.2_f64, 0.3, 0.4]);

    // gamma = (s·y)/(y·y) = 0.09/0.29.
    let gamma = 0.09_f64 / 0.29_f64;
    let rho = 1.0 / 0.09_f64;
    let alpha = rho * 0.6_f64; // rho * (s·g)
    let q = array![1.0 - alpha * 0.2, 2.0 - alpha * 0.3, 3.0 - alpha * 0.4];
    let scaled = array![q[0] * gamma, q[1] * gamma, q[2] * gamma];
    let beta = rho * (0.2 * scaled[0] + 0.3 * scaled[1] + 0.4 * scaled[2]);
    let coeff = alpha - beta;
    let expected = array![
        scaled[0] + coeff * 0.1,
        scaled[1] + coeff * 0.1,
        scaled[2] + coeff * 0.1
    ];

    // The `initial_hessian_scale` argument is now only a fallback, so passing a very
    // different value must not change the answer.
    for fallback in [1.0_f64, 42.0] {
        let result = lbfgs_two_loop_recursion(&gradient, &s_history, &y_history, fallback)
            .expect("recursion succeeds");
        for i in 0..3 {
            assert!(
                (result[i] - expected[i]).abs() < 1e-12,
                "component {} = {}, expected {} (gamma_k scaling)",
                i,
                result[i],
                expected[i]
            );
        }
    }

    // Sanity: the old hardcoded H0 = I would have produced a measurably different
    // vector, so this test really discriminates the fix.
    let unscaled_beta = rho * (0.2 * q[0] + 0.3 * q[1] + 0.4 * q[2]);
    let unscaled_coeff = alpha - unscaled_beta;
    let old = q[0] + unscaled_coeff * 0.1;
    assert!((old - expected[0]).abs() > 1e-3);
}

/// Pairs failing the curvature test must never enter the history in the first place.
#[test]
fn f40_update_lbfgs_approximation_skips_bad_pairs() {
    let mut s_history: VecDeque<Array1<f64>> = VecDeque::new();
    let mut y_history: VecDeque<Array1<f64>> = VecDeque::new();

    let rejected = update_lbfgs_approximation(
        &mut s_history,
        &mut y_history,
        array![1.0_f64, 0.0],
        array![-1.0_f64, 0.0],
        10,
    );
    assert!(!rejected, "y·s = -1 must be rejected");
    assert!(s_history.is_empty() && y_history.is_empty());

    let accepted = update_lbfgs_approximation(
        &mut s_history,
        &mut y_history,
        array![1.0_f64, 0.0],
        array![2.0_f64, 0.0],
        10,
    );
    assert!(accepted, "y·s = 2 must be accepted");
    assert_eq!(s_history.len(), 1);
    assert_eq!(y_history.len(), 1);
}

// ---------------------------------------------------------------------------
// F41: diagonal Newton must not step uphill on negative curvature
// ---------------------------------------------------------------------------

/// Dividing by a *signed* negative `h_ii` flips the update sign and walks uphill.
/// Using `|h_ii|` keeps the step a descent step.
#[test]
fn f41_diagonal_newton_is_a_descent_step_under_negative_curvature() {
    let mut optimizer = Newton::new(0.1_f64);
    let params = array![0.0_f64, 0.0];
    let gradients = array![1.0_f64, -2.0];
    // Both curvatures negative: the raw Newton step is an ascent step in both slots.
    let hessian = HessianInfo::Diagonal(array![-2.0_f64, -4.0]);

    let updated = optimizer
        .step_second_order(&params, &gradients, &hessian)
        .expect("step succeeds");

    // A descent step must move opposite to the gradient in every coordinate.
    assert!(
        updated[0] < params[0],
        "coordinate 0 moved uphill: {} -> {}",
        params[0],
        updated[0]
    );
    assert!(
        updated[1] > params[1],
        "coordinate 1 moved uphill: {} -> {}",
        params[1],
        updated[1]
    );

    // Directional check: g·(x_new - x_old) must be negative.
    let directional: f64 = gradients
        .iter()
        .zip(updated.iter().zip(params.iter()))
        .map(|(g, (new, old))| g * (new - old))
        .sum();
    assert!(
        directional < 0.0,
        "the update must have a negative directional derivative, got {}",
        directional
    );
}

/// Zero curvature must not blow the step up; the configurable floor bounds it.
#[test]
fn f41_diagonal_newton_floors_vanishing_curvature() {
    let mut optimizer = Newton::new(1.0_f64)
        .with_regularization(0.0)
        .with_min_curvature(1e-3);
    assert!((optimizer.min_curvature() - 1e-3).abs() < 1e-15);

    let params = array![0.0_f64];
    let gradients = array![1.0_f64];
    let hessian = HessianInfo::Diagonal(array![0.0_f64]);

    let updated = optimizer
        .step_second_order(&params, &gradients, &hessian)
        .expect("step succeeds");

    // |step| == lr * |g| / min_curvature == 1000, and it points downhill.
    assert!((updated[0] + 1000.0).abs() < 1e-9, "got {}", updated[0]);
}

// ---------------------------------------------------------------------------
// F38: Newton-CG negative curvature handling and CG tolerance
// ---------------------------------------------------------------------------

/// With a negative-definite Hessian the old code computed `alpha = r·r / p·Hp` with
/// `p·Hp < 0`, contaminating the iterate *before* noticing the negative curvature; the
/// resulting direction pointed along `+g`. Newton-CG must instead fall back to
/// steepest descent when negative curvature is hit on the very first CG iteration.
#[test]
fn f38_newton_cg_negative_curvature_yields_a_descent_direction() {
    let mut optimizer = NewtonCG::<f64>::new(1.0, 1e-8, 50, 0.0).expect("valid configuration");

    let params = array![1.0_f64, 1.0];
    let grads = array![1.0_f64, 1.0];
    // H = -I: negative definite everywhere.
    let hvp_fn = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| -x).collect() };

    let updated = optimizer
        .step(params.view(), grads.view(), hvp_fn)
        .expect("step succeeds");

    let directional: f64 = grads
        .iter()
        .zip(updated.iter().zip(params.iter()))
        .map(|(g, (new, old))| g * (new - old))
        .sum();
    assert!(
        directional < 0.0,
        "Newton-CG walked uphill under negative curvature: {:?} -> {:?}",
        params,
        updated
    );

    // Specifically: the steepest-descent fallback x - g.
    assert!((updated[0] - 0.0).abs() < 1e-12);
    assert!((updated[1] - 0.0).abs() < 1e-12);
}

/// Negative curvature discovered *after* a few good CG iterations must leave the
/// already-accumulated (descent) iterate intact rather than corrupting it.
#[test]
fn f38_newton_cg_indefinite_hessian_still_descends() {
    let mut optimizer = NewtonCG::<f64>::new(1.0, 1e-10, 50, 0.0).expect("valid configuration");

    let params = array![1.0_f64, 1.0, 1.0];
    let grads = array![1.0_f64, 0.5, 0.25];
    // Indefinite: positive in the first two coordinates, negative in the third.
    let hvp_fn = |v: &[f64]| -> Vec<f64> { vec![2.0 * v[0], 3.0 * v[1], -4.0 * v[2]] };

    let updated = optimizer
        .step(params.view(), grads.view(), hvp_fn)
        .expect("step succeeds");

    let directional: f64 = grads
        .iter()
        .zip(updated.iter().zip(params.iter()))
        .map(|(g, (new, old))| g * (new - old))
        .sum();
    assert!(
        directional < 0.0,
        "indefinite Hessian produced an ascent step: {:?} -> {:?}",
        params,
        updated
    );
}

/// `cg_tolerance` is documented as a bound on the *relative residual norm*:
/// CG stops once `||r|| <= tol * ||r_0||`. Comparing the squared residual `r·r`
/// directly against an unsquared `tol` makes the test far too lax and silently
/// truncates the solve.
///
/// Setup: `H = diag(1, 4)`, `g = [1, 1]`. After the first CG iteration
/// `||r_1|| / ||r_0|| = 0.6` exactly, which straddles the two rules for
/// `tol = 0.5`:
///
/// * dimensionally consistent (`r·r <= tol^2 r_0·r_0`, i.e. `0.72 <= 0.5`): keep going,
///   so the second iteration produces the exact solution `[-1, -0.25]`;
/// * buggy (`r·r < tol * r_0·r_0`, i.e. `0.72 < 1.0`): stop immediately at the
///   truncated `[-0.4, -0.4]`.
#[test]
fn f38_newton_cg_tolerance_comparison_is_dimensionally_consistent() {
    let hvp_fn = |v: &[f64]| -> Vec<f64> { vec![v[0], 4.0 * v[1]] };
    let params = array![0.0_f64, 0.0];
    let grads = array![1.0_f64, 1.0];

    // ||r_1||/||r_0|| = 0.6 > 0.5, so CG must NOT stop after one iteration.
    let mut strict = NewtonCG::<f64>::new(1.0, 0.5, 100, 0.0).expect("valid configuration");
    let solved = strict
        .step(params.view(), grads.view(), hvp_fn)
        .expect("step succeeds");
    assert!(
        (solved[0] + 1.0).abs() < 1e-12 && (solved[1] + 0.25).abs() < 1e-12,
        "tol = 0.5 must not terminate at ||r||/||r_0|| = 0.6; got {:?}, expected \
         the exact solution [-1, -0.25]",
        solved
    );

    // ||r_1||/||r_0|| = 0.6 <= 0.7, so CG *is* allowed to stop after one iteration.
    let mut loose = NewtonCG::<f64>::new(1.0, 0.7, 100, 0.0).expect("valid configuration");
    let truncated = loose
        .step(params.view(), grads.view(), hvp_fn)
        .expect("step succeeds");
    assert!(
        (truncated[0] + 0.4).abs() < 1e-12 && (truncated[1] + 0.4).abs() < 1e-12,
        "tol = 0.7 should stop after the first CG iteration; got {:?}, expected \
         [-0.4, -0.4]",
        truncated
    );
}

// ---------------------------------------------------------------------------
// F39: Steihaug-Toint trust region
// ---------------------------------------------------------------------------

/// A configured trust region must actually bound the step length.
#[test]
fn f39_trust_region_truncates_the_cg_direction() {
    let delta = 0.25_f64;
    let mut optimizer = NewtonCG::<f64>::new(1.0, 1e-12, 100, 0.0)
        .and_then(|o| o.with_trust_region(delta))
        .expect("valid configuration");

    let params = array![0.0_f64, 0.0];
    // Unconstrained Newton step would be [-5, -5], norm ~7.07, way outside the region.
    let grads = array![10.0_f64, 10.0];
    let hvp_fn = |v: &[f64]| -> Vec<f64> { vec![2.0 * v[0], 2.0 * v[1]] };

    let updated = optimizer
        .step(params.view(), grads.view(), hvp_fn)
        .expect("step succeeds");

    let norm = (updated[0] * updated[0] + updated[1] * updated[1]).sqrt();
    assert!(
        norm <= delta + 1e-9,
        "step norm {} exceeded the trust region radius {}",
        norm,
        delta
    );
    assert!(
        (norm - delta).abs() < 1e-9,
        "the truncated step should land exactly on the boundary, got {}",
        norm
    );
}

/// The full Steihaug-Toint loop: `rho` drives radius adaptation and step acceptance.
#[test]
fn f39_trust_region_adapts_radius_and_rejects_bad_steps() {
    // f(x) = x^2 near 0 but with a sharp penalty far away, so a large first step is
    // predicted to help by the quadratic model yet actually makes things worse.
    let loss = |x: &Array1<f64>| {
        let v = x[0];
        v * v + 40.0 * v.powi(4)
    };
    let grads_at = |v: f64| array![2.0 * v + 160.0 * v.powi(3)];
    let hvp_fn = |v: &[f64]| -> Vec<f64> { vec![2.0 * v[0]] };

    let mut optimizer = NewtonCG::<f64>::new(1.0, 1e-12, 50, 0.0)
        .and_then(|o| o.with_trust_region(4.0))
        .expect("valid configuration");

    let params = array![2.0_f64];
    let grads = grads_at(params[0]);

    let updated = optimizer
        .step_with_loss(params.view(), grads.view(), hvp_fn, loss)
        .expect("step succeeds");

    // The quadratic model ignores the quartic term, so this step is far too optimistic:
    // it must be rejected and the radius must shrink.
    assert!(
        !optimizer.last_step_accepted(),
        "the bad step must be rejected"
    );
    assert!(
        (updated[0] - params[0]).abs() < 1e-15,
        "rejected steps must not move"
    );
    let radius = optimizer
        .trust_region_radius()
        .expect("the radius stays configured");
    assert!(
        radius < 4.0,
        "the radius must shrink after rho < 0.25, got {}",
        radius
    );

    // Continuing from the same point, the shrinking region eventually admits a step and
    // the iteration converges to the minimizer at the origin.
    let mut x = params.clone();
    for _ in 0..80 {
        let g = grads_at(x[0]);
        x = optimizer
            .step_with_loss(x.view(), g.view(), hvp_fn, loss)
            .expect("step succeeds");
    }
    assert!(
        x[0].abs() < 1e-6,
        "trust region failed to converge, got {}",
        x[0]
    );
}

/// On a well-behaved quadratic the radius should grow back towards its maximum.
#[test]
fn f39_trust_region_grows_on_good_steps() {
    let loss = |x: &Array1<f64>| x[0] * x[0] + x[1] * x[1];
    let hvp_fn = |v: &[f64]| -> Vec<f64> { vec![2.0 * v[0], 2.0 * v[1]] };

    let mut optimizer = NewtonCG::<f64>::new(1.0, 1e-12, 50, 0.0)
        .and_then(|o| o.with_trust_region(0.01))
        .and_then(|o| o.with_max_trust_region_radius(10.0))
        .expect("valid configuration");

    let mut x = array![5.0_f64, 5.0];
    let initial_radius = optimizer.trust_region_radius().expect("configured");

    // Every one of these steps is a boundary step on a perfect quadratic model, so
    // rho == 1 > 0.75 and the radius must double each time.
    for _ in 0..5 {
        let g = array![2.0 * x[0], 2.0 * x[1]];
        x = optimizer
            .step_with_loss(x.view(), g.view(), hvp_fn, loss)
            .expect("step succeeds");
        assert!(
            optimizer.last_step_accepted(),
            "a perfect model step must be accepted"
        );
    }

    let radius = optimizer.trust_region_radius().expect("configured");
    assert!(
        radius > initial_radius,
        "the radius must grow on boundary steps with rho > 0.75 ({} -> {})",
        initial_radius,
        radius
    );
    assert!(
        (radius - initial_radius * 32.0).abs() < 1e-12,
        "five doublings expected: {} -> {}",
        initial_radius,
        radius
    );

    // With the radius no longer binding, the exact Newton step reaches the minimizer.
    for _ in 0..40 {
        let g = array![2.0 * x[0], 2.0 * x[1]];
        x = optimizer
            .step_with_loss(x.view(), g.view(), hvp_fn, loss)
            .expect("step succeeds");
    }
    assert!(
        x[0].abs() < 1e-9 && x[1].abs() < 1e-9,
        "should have converged, got {:?}",
        x
    );
}

/// The trust region is opt-in and validated.
#[test]
fn f39_trust_region_configuration_is_validated() {
    let base = NewtonCG::<f64>::default();
    assert!(base.trust_region_radius().is_none(), "off by default");

    assert!(NewtonCG::<f64>::default().with_trust_region(0.0).is_err());
    assert!(NewtonCG::<f64>::default().with_trust_region(-1.0).is_err());
    assert!(NewtonCG::<f64>::default()
        .with_max_trust_region_radius(0.0)
        .is_err());
    assert!(NewtonCG::<f64>::default()
        .with_trust_region_eta(1.0)
        .is_err());
    assert!(NewtonCG::<f64>::default()
        .with_trust_region_eta(-0.1)
        .is_err());
    assert!(NewtonCG::<f64>::default()
        .with_trust_region_eta(0.1)
        .is_ok());
}

// ---------------------------------------------------------------------------
// F11: K-FAC apply_update must not panic on a shape mismatch
// ---------------------------------------------------------------------------

/// The previously panicking configuration: `in_features = 2`, `out_features = 3`, so
/// `A^{-1}` is 2x2 and `G^{-1}` is 3x3. The old `apply_update` fed the per-sample
/// gradient matrix straight into `G^{-1} · grad · A^{-1}` and unwound on the shape
/// mismatch. The weight-gradient API now produces a correctly shaped `[3, 2]` result
/// and rejects the ill-shaped input with an error.
#[test]
fn f11_kfac_apply_update_weight_produces_correct_shape_instead_of_panicking() {
    let config = KFACConfig::<f64> {
        cov_update_freq: 1,
        inv_update_freq: 1,
        auto_damping: false,
        damping: 1e-3,
        learning_rate: 1.0,
        ..Default::default()
    };
    let mut kfac = KFAC::new(config);

    let layer_info = LayerInfo {
        name: "dense".to_string(),
        input_dim: 2,
        output_dim: 3,
        layer_type: LayerType::Dense,
        has_bias: false,
    };
    kfac.register_layer(layer_info).expect("layer registers");

    // Drive one step so covariances *and* inverses get computed.
    let activations = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("shape is valid");
    let out_grads = Array2::from_shape_vec(
        (4, 3),
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    )
    .expect("shape is valid");

    let mut layer_gradients = HashMap::new();
    layer_gradients.insert("dense".to_string(), (&activations, &out_grads));
    let updates = kfac
        .step::<fn() -> f64>(layer_gradients, None)
        .expect("step succeeds");

    // Guard: the K-FAC product must really be exercised, not short-circuited by the
    // "inverses not ready yet" branch.
    let state = kfac.get_layer_state("dense").expect("layer exists");
    assert!(
        state.is_ready(),
        "the test must reach the preconditioning path"
    );

    // The step's own update already has the weight shape.
    assert_eq!(updates["dense"].dim(), (3, 2));

    // And the exact previously-panicking call now returns a [3, 2] matrix.
    let grad_w =
        Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("shape is valid");
    let preconditioned = kfac
        .apply_update_weight("dense", &grad_w)
        .expect("weight update succeeds");
    assert_eq!(preconditioned.dim(), (3, 2));
    assert!(
        preconditioned.iter().all(|v| v.is_finite()),
        "preconditioned update must be finite: {:?}",
        preconditioned
    );

    // G^-1 · grad_W · A^-1 must differ from the raw gradient (it is preconditioned).
    let differs = preconditioned
        .iter()
        .zip(grad_w.iter())
        .any(|(a, b)| (a - b).abs() > 1e-9);
    assert!(differs, "the update was not preconditioned at all");

    // The per-sample gradient layout [batch, out] is now an error, not a panic.
    let err = kfac
        .apply_update_weight("dense", &out_grads)
        .expect_err("a [4, 3] gradient must be rejected");
    assert!(matches!(err, OptimError::DimensionMismatch(_)));

    // Unknown layers report a clean error too.
    assert!(kfac.apply_update_weight("missing", &grad_w).is_err());
}

/// The K-FAC preconditioner must reproduce `G^{-1} · ∇W · A^{-1}` exactly.
#[test]
fn f11_kfac_preconditioner_matches_the_kronecker_formula() {
    let config = KFACConfig::<f64> {
        cov_update_freq: 1,
        inv_update_freq: 1,
        auto_damping: false,
        damping: 1e-2,
        learning_rate: 1.0,
        ..Default::default()
    };
    let mut kfac = KFAC::new(config);
    kfac.register_layer(LayerInfo::dense("l".to_string(), 2, 3, false))
        .expect("layer registers");

    let activations = Array2::from_shape_vec((4, 2), vec![1.0, 0.5, -0.5, 2.0, 3.0, 1.0, 0.0, 1.5])
        .expect("shape is valid");
    let out_grads = Array2::from_shape_vec(
        (4, 3),
        vec![
            0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 0.7, 0.8, 0.9, -1.0, 1.1, 1.2,
        ],
    )
    .expect("shape is valid");

    let mut layer_gradients = HashMap::new();
    layer_gradients.insert("l".to_string(), (&activations, &out_grads));
    kfac.step::<fn() -> f64>(layer_gradients, None)
        .expect("step succeeds");

    let state = kfac.get_layer_state("l").expect("layer exists");
    assert!(state.is_ready());
    let a_inv = state.a_cov_inv.clone().expect("a_inv present");
    let g_inv = state.g_cov_inv.clone().expect("g_inv present");
    assert_eq!(a_inv.dim(), (2, 2));
    assert_eq!(g_inv.dim(), (3, 3));

    let grad_w = Array2::from_shape_vec((3, 2), vec![1.0, -2.0, 3.0, 0.5, -1.5, 2.5])
        .expect("shape is valid");
    let expected = g_inv.dot(&grad_w).dot(&a_inv);

    let actual = kfac
        .apply_update_weight("l", &grad_w)
        .expect("weight update succeeds");

    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (actual[[i, j]] - expected[[i, j]]).abs() < 1e-9,
                "[{},{}] = {}, expected {}",
                i,
                j,
                actual[[i, j]],
                expected[[i, j]]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// F12: K-FAC covariance must be the uncentered second moment
// ---------------------------------------------------------------------------

/// Mean-centering zeroes the homogeneous bias column, which makes `A` structurally
/// singular for every layer with a bias. The uncentered second moment keeps the bias
/// block at `E[1 * 1] = 1` and stays invertible.
#[test]
fn f12_kfac_bias_column_is_not_annihilated_by_centering() {
    let config = KFACConfig::<f64> {
        cov_update_freq: 1,
        inv_update_freq: 1,
        auto_damping: false,
        damping: 0.0,
        stat_decay: 0.0,
        learning_rate: 1.0,
        ..Default::default()
    };
    let mut kfac = KFAC::new(config);
    kfac.register_layer(LayerInfo::dense("biased".to_string(), 2, 2, true))
        .expect("layer registers");

    // Homogenized activations are [[1,2,1],[3,4,1],[5,7,1]] -> full rank.
    let activations =
        Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0]).expect("shape is valid");
    let out_grads = Array2::from_shape_vec((3, 2), vec![1.0, 0.5, -0.5, 2.0, 0.25, -1.0])
        .expect("shape is valid");

    let mut layer_gradients = HashMap::new();
    layer_gradients.insert("biased".to_string(), (&activations, &out_grads));
    kfac.step::<fn() -> f64>(layer_gradients, None)
        .expect("step succeeds");

    let state = kfac.get_layer_state("biased").expect("layer exists");
    assert_eq!(state.a_cov.dim(), (3, 3));

    // stat_decay = 0 so a_cov is exactly the batch second moment.
    // A = (1/3) * Â^T Â with Â = [[1,2,1],[3,4,1],[5,7,1]].
    let expected = [
        [35.0 / 3.0, 49.0 / 3.0, 3.0],
        [49.0 / 3.0, 23.0, 13.0 / 3.0],
        [3.0, 13.0 / 3.0, 1.0],
    ];
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (state.a_cov[[i, j]] - expected[i][j]).abs() < 1e-12,
                "A[{},{}] = {}, expected {}",
                i,
                j,
                state.a_cov[[i, j]],
                expected[i][j]
            );
        }
    }

    // The crux: with mean centering this entry (and the whole bias row/column) is 0.
    assert!(
        (state.a_cov[[2, 2]] - 1.0).abs() < 1e-12,
        "the homogeneous bias block must be E[1*1] = 1, got {}",
        state.a_cov[[2, 2]]
    );

    // With zero damping the factor must still be invertible.
    assert!(state.is_ready(), "inverses must have been computed");
    let a_inv = state.a_cov_inv.as_ref().expect("a_inv present");
    let product = state.a_cov.dot(a_inv);
    for i in 0..3 {
        for j in 0..3 {
            let identity = if i == j { 1.0 } else { 0.0 };
            assert!(
                (product[[i, j]] - identity).abs() < 1e-8,
                "A · A^-1 is not the identity at [{},{}]: {}",
                i,
                j,
                product[[i, j]]
            );
        }
    }
}

/// A single-sample batch is a valid second-moment estimate (`E[a a^T] = a a^T`),
/// whereas the centered estimator degenerated to the identity.
#[test]
fn f12_kfac_single_sample_batch_uses_the_outer_product() {
    let config = KFACConfig::<f64> {
        cov_update_freq: 1,
        inv_update_freq: 1000,
        auto_damping: false,
        stat_decay: 0.0,
        learning_rate: 1.0,
        ..Default::default()
    };
    let mut kfac = KFAC::new(config);
    kfac.register_layer(LayerInfo::dense("single".to_string(), 3, 2, false))
        .expect("layer registers");

    let activations = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).expect("shape is valid");
    let out_grads = Array2::from_shape_vec((1, 2), vec![0.5, -0.5]).expect("shape is valid");

    let mut layer_gradients = HashMap::new();
    layer_gradients.insert("single".to_string(), (&activations, &out_grads));
    kfac.step::<fn() -> f64>(layer_gradients, None)
        .expect("step succeeds");

    let state = kfac.get_layer_state("single").expect("layer exists");
    let a = [1.0_f64, 2.0, 3.0];
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (state.a_cov[[i, j]] - a[i] * a[j]).abs() < 1e-12,
                "A[{},{}] = {}, expected {}",
                i,
                j,
                state.a_cov[[i, j]],
                a[i] * a[j]
            );
        }
    }
}

/// Covariance updates validate shapes instead of panicking inside ndarray.
#[test]
fn f12_kfac_covariance_update_rejects_mismatched_shapes() {
    let config = KFACConfig::<f64> {
        cov_update_freq: 1,
        inv_update_freq: 1000,
        auto_damping: false,
        ..Default::default()
    };
    let mut kfac = KFAC::new(config);
    kfac.register_layer(LayerInfo::dense("l".to_string(), 4, 2, false))
        .expect("layer registers");

    // 7 input columns for a layer declared with 4.
    let activations = Array2::<f64>::ones((2, 7));
    let out_grads = Array2::<f64>::ones((2, 2));
    let mut layer_gradients = HashMap::new();
    layer_gradients.insert("l".to_string(), (&activations, &out_grads));

    let err = kfac
        .step::<fn() -> f64>(layer_gradients, None)
        .expect_err("mismatched activation width must be an error");
    assert!(matches!(err, OptimError::DimensionMismatch(_)));
}
