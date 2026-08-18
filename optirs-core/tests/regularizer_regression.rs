// Regression tests for regularizer / constraint / adaptive-selection audit findings.
//
// Each test in this file pins down a behaviour that was previously wrong:
//
// * F65 - `ParameterConstraint::NuclearNorm` used entrywise L1 shrinkage instead
//   of acting on the singular values.
// * F66 - `MixUp`/`CutMix` drew the mixing coefficient from a uniform
//   distribution, silently ignoring the configured `alpha`/`beta`.
// * F89 - `Dropout` carried a dead mask cache and documented itself as acting on
//   activations while it actually masks gradients.
// * F67 - `SelectionNetwork::train` only updated the output layer, leaving the
//   hidden layer frozen at its initialization.

use optirs_core::adaptive_selection::SelectionNetwork;
use optirs_core::parameter_groups::{
    nuclear_norm_of_matrix, nuclear_norm_prox, project_onto_nuclear_norm_ball, ParameterConstraint,
};
use optirs_core::regularizers::{CutMix, Dropout, MixUp, Regularizer};
use scirs2_core::ndarray::{arr2, Array1, Array2, ArrayD};
use scirs2_core::random::rngs::SmallRng;
use scirs2_core::random::SeedableRng;

// ---------------------------------------------------------------------------
// Helpers that verify a spectrum WITHOUT calling the SVD code under test.
//
// For a 3x3 matrix R the Gram matrix G = RᵀR has eigenvalues σᵢ². The three
// coefficients of its characteristic polynomial — trace(G), the sum of the 2x2
// principal minors, and det(G) — are the elementary symmetric polynomials of
// those eigenvalues and therefore determine the multiset {σᵢ²} uniquely. Using
// them keeps the assertions independent of the power-iteration implementation.
// ---------------------------------------------------------------------------

fn gram(matrix: &Array2<f64>) -> Array2<f64> {
    let (rows, cols) = matrix.dim();
    let mut g = Array2::<f64>::zeros((cols, cols));
    for i in 0..cols {
        for j in 0..cols {
            let mut acc = 0.0;
            for k in 0..rows {
                acc += matrix[[k, i]] * matrix[[k, j]];
            }
            g[[i, j]] = acc;
        }
    }
    g
}

fn trace3(g: &Array2<f64>) -> f64 {
    g[[0, 0]] + g[[1, 1]] + g[[2, 2]]
}

/// Sum of the three 2x2 principal minors of a 3x3 matrix.
fn second_invariant3(g: &Array2<f64>) -> f64 {
    let minor = |a: usize, b: usize| g[[a, a]] * g[[b, b]] - g[[a, b]] * g[[b, a]];
    minor(0, 1) + minor(0, 2) + minor(1, 2)
}

fn det3(g: &Array2<f64>) -> f64 {
    g[[0, 0]] * (g[[1, 1]] * g[[2, 2]] - g[[1, 2]] * g[[2, 1]])
        - g[[0, 1]] * (g[[1, 0]] * g[[2, 2]] - g[[1, 2]] * g[[2, 0]])
        + g[[0, 2]] * (g[[1, 0]] * g[[2, 1]] - g[[1, 1]] * g[[2, 0]])
}

/// Assert that the singular values of a 3x3 matrix match `expected` (any order).
fn assert_spectrum_3x3(matrix: &Array2<f64>, expected: [f64; 3], tolerance: f64) {
    let squares = [
        expected[0] * expected[0],
        expected[1] * expected[1],
        expected[2] * expected[2],
    ];
    let want_trace = squares[0] + squares[1] + squares[2];
    let want_second = squares[0] * squares[1] + squares[0] * squares[2] + squares[1] * squares[2];
    let want_det = squares[0] * squares[1] * squares[2];

    let g = gram(matrix);
    assert!(
        (trace3(&g) - want_trace).abs() < tolerance,
        "Σσ² = {} but expected {want_trace}",
        trace3(&g)
    );
    assert!(
        (second_invariant3(&g) - want_second).abs() < tolerance,
        "e2(σ²) = {} but expected {want_second}",
        second_invariant3(&g)
    );
    assert!(
        (det3(&g) - want_det).abs() < tolerance,
        "det(RᵀR) = {} but expected {want_det}",
        det3(&g)
    );
}

/// Build `U · diag(spectrum) · Vᵀ` from two explicit orthonormal 3x3 bases.
fn matrix_with_spectrum(spectrum: [f64; 3]) -> Array2<f64> {
    // U: rotation about the z axis by 30 degrees.
    let (cu, su) = (
        (std::f64::consts::PI / 6.0).cos(),
        (std::f64::consts::PI / 6.0).sin(),
    );
    let u = arr2(&[[cu, -su, 0.0], [su, cu, 0.0], [0.0, 0.0, 1.0]]);
    // V: rotation about the x axis by 45 degrees.
    let (cv, sv) = (
        (std::f64::consts::PI / 4.0).cos(),
        (std::f64::consts::PI / 4.0).sin(),
    );
    let v = arr2(&[[1.0, 0.0, 0.0], [0.0, cv, -sv], [0.0, sv, cv]]);

    let mut m = Array2::<f64>::zeros((3, 3));
    for (k, &sigma) in spectrum.iter().enumerate() {
        for i in 0..3 {
            for j in 0..3 {
                m[[i, j]] += sigma * u[[i, k]] * v[[j, k]];
            }
        }
    }
    m
}

fn elementwise_soft_threshold(matrix: &Array2<f64>, threshold: f64) -> Array2<f64> {
    matrix.mapv(|x| {
        if x > threshold {
            x - threshold
        } else if x < -threshold {
            x + threshold
        } else {
            0.0
        }
    })
}

// ---------------------------------------------------------------------------
// F65 - nuclear norm acts on singular values, not on raw entries.
// ---------------------------------------------------------------------------

#[test]
fn f65_nuclear_prox_soft_thresholds_singular_values() {
    let m = matrix_with_spectrum([3.0, 2.0, 0.5]);
    assert_spectrum_3x3(&m, [3.0, 2.0, 0.5], 1e-8);

    let shrunk = nuclear_norm_prox(&m, 1.0);

    // prox_{1·‖·‖_*} shrinks {3, 2, 0.5} to {2, 1, 0}: rank drops to 2 and the
    // nuclear norm drops from 5.5 to 3.
    assert_spectrum_3x3(&shrunk, [2.0, 1.0, 0.0], 1e-6);

    let frobenius: f64 = shrunk.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        (frobenius - 5.0f64.sqrt()).abs() < 1e-6,
        "‖prox‖_F = {frobenius}, expected sqrt(5)"
    );

    let nuclear = nuclear_norm_of_matrix(&shrunk);
    assert!(
        (nuclear - 3.0).abs() < 1e-6,
        "‖prox‖_* = {nuclear}, expected 3"
    );
}

#[test]
fn f65_nuclear_prox_is_not_elementwise_l1_shrinkage() {
    let m = matrix_with_spectrum([3.0, 2.0, 0.5]);

    let nuclear = nuclear_norm_prox(&m, 1.0);
    let entrywise = elementwise_soft_threshold(&m, 1.0);

    let distance: f64 = nuclear
        .iter()
        .zip(entrywise.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        distance > 1e-2,
        "nuclear prox is indistinguishable from entrywise L1 shrinkage (L1 distance {distance})"
    );

    // The old implementation also rescaled by the *entrywise* L1 norm; make sure
    // the spectrum, not the entry magnitudes, is what got clipped.
    assert_spectrum_3x3(&nuclear, [2.0, 1.0, 0.0], 1e-6);
}

#[test]
fn f65_nuclear_norm_constraint_projects_onto_the_ball() {
    let m = matrix_with_spectrum([3.0, 2.0, 0.5]); // ‖M‖_* = 5.5
    let mut params = m.clone();

    let constraint = ParameterConstraint::NuclearNorm { maxnorm: 3.0 };
    constraint
        .apply(&mut params)
        .expect("nuclear norm constraint should accept a 2-D matrix");

    // Projection onto the L1 ball of the spectrum with radius 3 uses θ = 1,
    // leaving {2, 1, 0}.
    assert_spectrum_3x3(&params, [2.0, 1.0, 0.0], 1e-6);

    let direct = project_onto_nuclear_norm_ball(&m, 3.0);
    for (a, b) in params.iter().zip(direct.iter()) {
        assert!((a - b).abs() < 1e-9);
    }
}

#[test]
fn f65_nuclear_norm_constraint_is_a_no_op_inside_the_ball() {
    let m = matrix_with_spectrum([1.0, 0.5, 0.25]); // ‖M‖_* = 1.75
    let mut params = m.clone();

    let constraint = ParameterConstraint::NuclearNorm { maxnorm: 10.0 };
    constraint.apply(&mut params).expect("constraint failed");

    assert_eq!(params, m);
}

#[test]
fn f65_nuclear_norm_constraint_rejects_non_matrices() {
    let mut params = Array1::from_vec(vec![3.0, -4.0, 2.0]);
    let constraint = ParameterConstraint::NuclearNorm { maxnorm: 3.0 };

    match constraint.apply(&mut params) {
        Ok(()) => panic!("nuclear norm is undefined for 1-D parameters"),
        Err(err) => assert!(
            err.to_string().contains("2D arrays"),
            "unexpected error message: {err}"
        ),
    }
}

// ---------------------------------------------------------------------------
// F66 - MixUp / CutMix draw lambda from Beta(alpha, alpha).
// ---------------------------------------------------------------------------

struct LambdaStats {
    mean: f64,
    variance: f64,
    extreme_fraction: f64,
    central_fraction: f64,
}

fn lambda_stats<F: Fn(u64) -> f64>(sampler: F, n: u64) -> LambdaStats {
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut extreme = 0u64;
    let mut central = 0u64;
    for seed in 0..n {
        // Distinct, well-spread seeds so consecutive draws are independent.
        let lambda = sampler(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));
        assert!(
            (0.0..=1.0).contains(&lambda),
            "lambda {lambda} outside [0, 1]"
        );
        sum += lambda;
        sum_sq += lambda * lambda;
        if !(0.1..=0.9).contains(&lambda) {
            extreme += 1;
        }
        if (0.3..=0.7).contains(&lambda) {
            central += 1;
        }
    }
    let count = n as f64;
    let mean = sum / count;
    LambdaStats {
        mean,
        variance: sum_sq / count - mean * mean,
        extreme_fraction: extreme as f64 / count,
        central_fraction: central as f64 / count,
    }
}

#[test]
fn f66_mixup_lambda_follows_beta_alpha_alpha() {
    const N: u64 = 20_000;

    // Beta(a, a): mean 0.5, variance 1 / (4 (2a + 1)).
    let small = MixUp::<f64>::new(0.2).expect("alpha 0.2 is valid");
    let small_stats = lambda_stats(|seed| small.mixing_factor(seed), N);
    let small_expected_var = 1.0 / (4.0 * (2.0 * 0.2 + 1.0)); // ≈ 0.17857
    assert!(
        (small_stats.mean - 0.5).abs() < 0.02,
        "alpha=0.2 mean {} != 0.5",
        small_stats.mean
    );
    assert!(
        (small_stats.variance - small_expected_var).abs() < 0.02,
        "alpha=0.2 variance {} != {small_expected_var}",
        small_stats.variance
    );
    // Strongly U-shaped: most draws sit in the outer tenths.
    assert!(
        small_stats.extreme_fraction > 0.55,
        "alpha=0.2 is not U-shaped: only {:.3} of draws outside [0.1, 0.9]",
        small_stats.extreme_fraction
    );

    let large = MixUp::<f64>::new(5.0).expect("alpha 5.0 is valid");
    let large_stats = lambda_stats(|seed| large.mixing_factor(seed), N);
    let large_expected_var = 1.0 / (4.0 * (2.0 * 5.0 + 1.0)); // ≈ 0.022727
    assert!(
        (large_stats.mean - 0.5).abs() < 0.02,
        "alpha=5.0 mean {} != 0.5",
        large_stats.mean
    );
    assert!(
        (large_stats.variance - large_expected_var).abs() < 0.005,
        "alpha=5.0 variance {} != {large_expected_var}",
        large_stats.variance
    );
    // Concentrated around 0.5.
    assert!(
        large_stats.central_fraction > 0.70,
        "alpha=5.0 is not concentrated: only {:.3} of draws inside [0.3, 0.7]",
        large_stats.central_fraction
    );
    assert!(
        large_stats.extreme_fraction < 0.05,
        "alpha=5.0 produced too many extreme draws: {:.3}",
        large_stats.extreme_fraction
    );

    // The regression itself: a uniform draw would give the *same* variance
    // (1/12 ≈ 0.0833) for both alphas.
    assert!(
        small_stats.variance > 4.0 * large_stats.variance,
        "alpha is being ignored: var(0.2)={} var(5.0)={}",
        small_stats.variance,
        large_stats.variance
    );
    assert!(
        (small_stats.variance - 1.0 / 12.0).abs() > 0.02,
        "alpha=0.2 still looks uniform (variance {})",
        small_stats.variance
    );
    assert!(
        (large_stats.variance - 1.0 / 12.0).abs() > 0.02,
        "alpha=5.0 still looks uniform (variance {})",
        large_stats.variance
    );
}

#[test]
fn f66_cutmix_lambda_follows_beta_beta_beta() {
    const N: u64 = 20_000;

    let small = CutMix::<f64>::new(0.2).expect("beta 0.2 is valid");
    let small_stats = lambda_stats(|seed| small.mixing_factor(seed), N);
    assert!((small_stats.mean - 0.5).abs() < 0.02);
    assert!(small_stats.extreme_fraction > 0.55);

    let large = CutMix::<f64>::new(5.0).expect("beta 5.0 is valid");
    let large_stats = lambda_stats(|seed| large.mixing_factor(seed), N);
    assert!((large_stats.mean - 0.5).abs() < 0.02);
    assert!(large_stats.central_fraction > 0.70);

    assert!(small_stats.variance > 4.0 * large_stats.variance);
}

#[test]
fn f66_mixing_factor_is_deterministic_per_seed() {
    let mixup = MixUp::<f64>::new(0.7).expect("alpha 0.7 is valid");
    assert_eq!(mixup.mixing_factor(1234), mixup.mixing_factor(1234));
    assert_ne!(mixup.mixing_factor(1234), mixup.mixing_factor(4321));
    assert_eq!(mixup.alpha(), 0.7);

    let cutmix = CutMix::<f64>::new(1.5).expect("beta 1.5 is valid");
    assert_eq!(cutmix.mixing_factor(99), cutmix.mixing_factor(99));
    assert_eq!(cutmix.beta(), 1.5);
}

// ---------------------------------------------------------------------------
// F89 - Dropout masks gradients (as documented) and never caches its mask.
// ---------------------------------------------------------------------------

fn make_dropout(rate: f64) -> Dropout<f64> {
    let mut rng = SmallRng::from_seed([42u8; 32]);
    Dropout::new(rate, &mut rng)
}

fn dyn_ones(len: usize) -> ArrayD<f64> {
    Array1::from_elem(len, 1.0).into_dyn()
}

#[test]
fn f89_dropout_masks_gradients_and_leaves_parameters_alone() {
    let mut dropout = make_dropout(0.5);
    dropout.train();

    let params = Array1::from_elem(4096, 3.0).into_dyn();
    let params_before = params.clone();
    let mut gradients = dyn_ones(4096);

    let penalty = dropout
        .apply(&params, &mut gradients)
        .expect("dropout apply failed");

    // Documented behaviour: no loss penalty, parameters untouched, gradients
    // either zeroed or rescaled by 1/(1 - rate).
    assert_eq!(penalty, 0.0);
    assert_eq!(params, params_before);
    assert!(gradients
        .iter()
        .all(|&g| g == 0.0 || (g - 2.0).abs() < 1e-12));

    let dropped = gradients.iter().filter(|&&g| g == 0.0).count();
    let fraction = dropped as f64 / 4096.0;
    assert!(
        (fraction - 0.5).abs() < 0.05,
        "dropped fraction {fraction} far from the configured rate 0.5"
    );

    // Inverted dropout preserves the expected gradient magnitude.
    let sum: f64 = gradients.sum();
    assert!(
        (sum - 4096.0).abs() < 400.0,
        "inverted-dropout rescaling is off: sum {sum}"
    );
}

#[test]
fn f89_dropout_draws_a_fresh_mask_per_call() {
    let mut dropout = make_dropout(0.5);
    dropout.train();

    let params = dyn_ones(1024);
    let mut first = dyn_ones(1024);
    let mut second = dyn_ones(1024);

    dropout.apply(&params, &mut first).expect("apply failed");
    dropout.apply(&params, &mut second).expect("apply failed");

    assert_ne!(
        first, second,
        "dropout reused a cached mask instead of resampling"
    );
}

#[test]
fn f89_dropout_respects_eval_mode_and_rate_changes() {
    let mut dropout = make_dropout(0.9);
    dropout.eval();

    let params = dyn_ones(64);
    let original = dyn_ones(64);
    let mut gradients = original.clone();
    dropout
        .apply(&params, &mut gradients)
        .expect("apply failed");
    assert_eq!(gradients, original, "eval mode must not touch gradients");

    // Changing the rate takes effect immediately (there is no stale cache).
    dropout.train();
    dropout.set_rate(0.0);
    let mut gradients = original.clone();
    dropout
        .apply(&params, &mut gradients)
        .expect("apply failed");
    assert_eq!(gradients, original, "rate 0.0 must be the identity");
    assert_eq!(dropout.rate(), 0.0);
}

// ---------------------------------------------------------------------------
// F67 - SelectionNetwork trains its hidden layer, not just the output layer.
// ---------------------------------------------------------------------------

fn selection_dataset() -> (Vec<Array1<f64>>, Vec<usize>) {
    let features = vec![
        Array1::from_vec(vec![1.0, 0.2, -0.5, 0.8]),
        Array1::from_vec(vec![0.9, 0.1, -0.4, 0.7]),
        Array1::from_vec(vec![-0.8, 1.2, 0.3, -0.6]),
        Array1::from_vec(vec![-0.7, 1.1, 0.2, -0.5]),
        Array1::from_vec(vec![0.2, -1.0, 1.4, 0.1]),
        Array1::from_vec(vec![0.1, -0.9, 1.3, 0.2]),
        Array1::from_vec(vec![1.1, 0.3, -0.6, 0.9]),
        Array1::from_vec(vec![-0.9, 1.3, 0.1, -0.7]),
    ];
    let labels = vec![0, 0, 1, 1, 2, 2, 0, 1];
    (features, labels)
}

#[test]
fn f67_training_updates_the_hidden_layer_and_reduces_loss() {
    let mut network = SelectionNetwork::<f64>::new(4, 16, 3);
    let (features, labels) = selection_dataset();

    let hidden_before = network.input_weights().clone();
    let hidden_bias_before = network.input_bias().clone();
    let loss_before = network
        .average_loss(&features, &labels)
        .expect("loss evaluation failed");

    network
        .train(&features, &labels, 0.05, 400)
        .expect("training failed");

    let hidden_after = network.input_weights();
    let hidden_bias_after = network.input_bias();

    // The hidden layer must have moved. Before the fix these were bit-identical.
    let max_weight_delta = hidden_before
        .iter()
        .zip(hidden_after.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_weight_delta > 1e-6,
        "hidden-layer weights did not change (max delta {max_weight_delta})"
    );

    let max_bias_delta = hidden_bias_before
        .iter()
        .zip(hidden_bias_after.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_bias_delta > 1e-6,
        "hidden-layer biases did not change (max delta {max_bias_delta})"
    );

    let loss_after = network
        .average_loss(&features, &labels)
        .expect("loss evaluation failed");
    assert!(loss_after.is_finite(), "loss diverged to {loss_after}");
    assert!(
        loss_after < loss_before,
        "training loss did not decrease: {loss_before} -> {loss_after}"
    );
    assert!(
        loss_after < 0.9 * loss_before,
        "training barely moved the loss: {loss_before} -> {loss_after}"
    );
}

/// A fully specified network, so the gradient check below is deterministic.
fn deterministic_network() -> SelectionNetwork<f64> {
    let input_weights = Array2::from_shape_fn((5, 4), |(j, k)| {
        0.35 * ((j * 4 + k) as f64 * 0.7).sin() + 0.1
    });
    let input_bias = Array1::from_shape_fn(5, |j| 0.05 * (j as f64) - 0.1);
    let output_weights =
        Array2::from_shape_fn((3, 5), |(i, j)| 0.4 * ((i * 5 + j) as f64 * 1.1).cos());
    let output_bias = Array1::from_vec(vec![0.05, -0.02, 0.01]);

    SelectionNetwork::from_parameters(input_weights, input_bias, output_weights, output_bias)
        .expect("consistent topology")
}

#[test]
fn f67_hidden_layer_update_matches_the_numeric_gradient() {
    // One SGD step with a tiny learning rate must move every hidden weight by
    // -lr * dL/dw. Estimating dL/dw by central differences catches both the old
    // "hidden layer never updated" bug (gradient treated as 0) and any sign or
    // activation-derivative mistake in the backward pass.
    let sample = Array1::from_vec(vec![0.9, -0.4, 1.3, 0.6]);
    let features = vec![sample.clone()];
    let labels = vec![2usize];

    let base = deterministic_network();
    let before_weights = base.input_weights().clone();
    let before_bias = base.input_bias().clone();

    let learning_rate = 1e-4;
    let mut trained = deterministic_network();
    trained
        .train(&features, &labels, learning_rate, 1)
        .expect("training failed");

    let eps = 1e-6;
    let mut checked = 0usize;
    for j in 0..before_weights.nrows() {
        for k in 0..before_weights.ncols() {
            // Central difference of the loss w.r.t. input_weights[[j, k]].
            let mut plus = before_weights.clone();
            plus[[j, k]] += eps;
            let loss_plus = SelectionNetwork::from_parameters(
                plus,
                before_bias.clone(),
                base.output_weights().clone(),
                base.output_bias().clone(),
            )
            .expect("consistent topology")
            .average_loss(&features, &labels)
            .expect("loss failed");

            let mut minus = before_weights.clone();
            minus[[j, k]] -= eps;
            let loss_minus = SelectionNetwork::from_parameters(
                minus,
                before_bias.clone(),
                base.output_weights().clone(),
                base.output_bias().clone(),
            )
            .expect("consistent topology")
            .average_loss(&features, &labels)
            .expect("loss failed");

            let numeric_grad = (loss_plus - loss_minus) / (2.0 * eps);
            let expected_delta = -learning_rate * numeric_grad;
            let actual_delta = trained.input_weights()[[j, k]] - before_weights[[j, k]];

            assert!(
                (actual_delta - expected_delta).abs() < 1e-9,
                "hidden weight [{j},{k}]: moved by {actual_delta}, expected {expected_delta} \
                 (numeric gradient {numeric_grad})"
            );
            if numeric_grad.abs() > 1e-6 {
                checked += 1;
            }
        }
    }

    assert!(
        checked > 0,
        "the fixture produced only zero hidden gradients; the check would be vacuous"
    );
}

#[test]
fn f67_forward_still_produces_a_probability_distribution_after_training() {
    let mut network = SelectionNetwork::<f64>::new(4, 16, 3);
    let (features, labels) = selection_dataset();

    network
        .train(&features, &labels, 0.05, 200)
        .expect("training failed");

    for feature in features.iter() {
        let probabilities = network.forward(feature).expect("forward failed");
        assert_eq!(probabilities.len(), 3);
        let sum: f64 = probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "probabilities sum to {sum}");
        assert!(probabilities.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }

    assert_eq!(network.hidden_size(), 16);
    assert_eq!(network.input_size(), 4);
    assert_eq!(network.num_outputs(), 3);
}

#[test]
fn f67_training_rejects_malformed_datasets() {
    let mut network = SelectionNetwork::<f64>::new(4, 8, 3);
    let (features, labels) = selection_dataset();

    // Label out of range for 3 output units.
    let mut bad_labels = labels.clone();
    bad_labels[0] = 7;
    assert!(network.train(&features, &bad_labels, 0.05, 1).is_err());

    // Feature/label count mismatch.
    assert!(network.train(&features, &labels[..2], 0.05, 1).is_err());

    // Wrong feature width.
    let wrong_width = vec![Array1::from_vec(vec![1.0, 2.0])];
    assert!(network.train(&wrong_width, &[0], 0.05, 1).is_err());
}
