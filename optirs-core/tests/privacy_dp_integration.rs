//! End-to-end tests for the differential privacy stack.
//!
//! These tests exercise the public API the way a caller would: run DP-SGD
//! steps, watch the privacy budget grow, and check that the guarantees the
//! crate advertises actually hold. They are deliberately black-box -- every
//! previous accounting defect in this crate (a budget that exhausted on the
//! first call, an epsilon that never moved, noise that was identical in every
//! process) would have been caught by one of the assertions below.

use optirs_core::error::OptimError;
use optirs_core::optimizers::SGD;
use optirs_core::privacy::dp_sgd::DPSGDOptimizer;
use optirs_core::privacy::noise_mechanisms::{
    GaussianMechanism, LaplaceMechanism, NoiseMechanism as NoiseMechanismTrait,
    SparseVectorMechanism,
};
use optirs_core::privacy::{
    build_accountant, AccountingMethod, DifferentialPrivacyConfig, DifferentiallyPrivateOptimizer,
    NoiseMechanism, PrivacyAccountant, RenyiAccountant,
};
use scirs2_core::ndarray::{Array1, Ix1};

/// A configuration with enough budget to run a long training loop.
fn roomy_config() -> DifferentialPrivacyConfig {
    DifferentialPrivacyConfig {
        target_epsilon: 50.0,
        target_delta: 1e-5,
        noise_multiplier: 1.1,
        l2_norm_clip: 1.0,
        batch_size: 64,
        dataset_size: 50_000,
        max_steps: 10_000,
        ..Default::default()
    }
}

fn build_dp_optimizer(
    config: DifferentialPrivacyConfig,
) -> DifferentiallyPrivateOptimizer<SGD<f64>, f64, Ix1> {
    match DifferentiallyPrivateOptimizer::new(SGD::new(0.05), config) {
        Ok(optimizer) => optimizer,
        Err(err) => panic!("failed to construct the DP optimizer: {err}"),
    }
}

fn batch_of(count: usize, dim: usize, value: f64) -> Vec<Array1<f64>> {
    (0..count).map(|_| Array1::from_elem(dim, value)).collect()
}

#[test]
fn hundred_dp_steps_succeed_with_monotonically_increasing_epsilon() {
    let mut optimizer = build_dp_optimizer(roomy_config());
    let params = Array1::<f64>::zeros(16);
    let batch = batch_of(32, 16, 0.05);

    let mut previous_epsilon = 0.0;
    let mut current_params = params.clone();

    for step in 1..=100 {
        current_params = match optimizer.dp_step_per_example(&current_params, &batch) {
            Ok(updated) => updated,
            Err(err) => panic!("step {step} failed: {err}"),
        };

        let epsilon = match optimizer.consumed_epsilon() {
            Ok(value) => value,
            Err(err) => panic!("accounting failed at step {step}: {err}"),
        };
        assert!(
            epsilon.is_finite() && epsilon > 0.0,
            "step {step}: epsilon must be a positive finite number, got {epsilon}"
        );
        assert!(
            epsilon > previous_epsilon,
            "step {step}: epsilon must increase strictly ({previous_epsilon} -> {epsilon})"
        );
        previous_epsilon = epsilon;
    }

    let budget = match optimizer.get_privacy_budget() {
        Ok(budget) => budget,
        Err(err) => panic!("budget query failed: {err}"),
    };
    assert_eq!(budget.steps_taken, 100);
    assert!(budget.epsilon_remaining > 0.0);
    // Delta is a reporting parameter, never consumed.
    assert_eq!(budget.delta_consumed, 0.0);
    assert_eq!(budget.delta_remaining, 1e-5);
    assert!(current_params.iter().all(|value| value.is_finite()));
}

#[test]
fn exhaustion_triggers_at_the_step_the_accountant_predicts() {
    // Small population, large sampling rate: the budget is reached quickly
    // and the exhaustion step can be cross-checked against an independent
    // accountant instance.
    let config = DifferentialPrivacyConfig {
        target_epsilon: 6.0,
        target_delta: 1e-5,
        noise_multiplier: 1.0,
        l2_norm_clip: 1.0,
        batch_size: 64,
        dataset_size: 640,
        max_steps: 100_000,
        ..Default::default()
    };
    let mut optimizer = build_dp_optimizer(config);
    let params = Array1::<f64>::zeros(4);
    let batch = batch_of(64, 4, 0.1);

    let mut accepted = 0usize;
    loop {
        match optimizer.dp_step_per_example(&params, &batch) {
            Ok(_) => {
                accepted += 1;
                assert!(accepted < 100_000, "the budget was never enforced");
            }
            Err(OptimError::PrivacyBudgetExhausted {
                consumed_epsilon,
                target_epsilon,
            }) => {
                assert_eq!(target_epsilon, 6.0);
                assert!(
                    consumed_epsilon <= target_epsilon,
                    "the optimizer must refuse a step *before* exceeding the budget: \
                     consumed {consumed_epsilon} > target {target_epsilon}"
                );
                break;
            }
            Err(other) => panic!("unexpected error: {other}"),
        }
    }

    assert!(accepted > 0, "the very first step must be allowed");

    // Independent cross-check: composing exactly `accepted` steps must stay
    // within the budget, and one more must exceed it.
    let mut reference = RenyiAccountant::with_default_orders();
    match reference.add_subsampled_gaussian(1.0, 0.1, accepted) {
        Ok(()) => {}
        Err(err) => panic!("reference composition failed: {err}"),
    }
    let at_limit = match reference.to_epsilon_delta(1e-5) {
        Ok(conversion) => conversion.epsilon,
        Err(err) => panic!("reference conversion failed: {err}"),
    };
    assert!(
        at_limit <= 6.0,
        "the optimizer accepted {accepted} steps costing epsilon = {at_limit} > 6.0"
    );

    match reference.add_subsampled_gaussian(1.0, 0.1, 1) {
        Ok(()) => {}
        Err(err) => panic!("reference composition failed: {err}"),
    }
    let past_limit = match reference.to_epsilon_delta(1e-5) {
        Ok(conversion) => conversion.epsilon,
        Err(err) => panic!("reference conversion failed: {err}"),
    };
    assert!(
        past_limit > 6.0,
        "the optimizer stopped too early: step {} would have cost only {past_limit}",
        accepted + 1
    );
}

#[test]
fn budget_is_enforced_before_any_output_is_released() {
    let config = DifferentialPrivacyConfig {
        target_epsilon: 1.5,
        noise_multiplier: 1.0,
        batch_size: 64,
        dataset_size: 640,
        max_steps: 100_000,
        ..roomy_config()
    };
    let mut optimizer = build_dp_optimizer(config);
    let params = Array1::<f64>::zeros(4);
    let batch = batch_of(64, 4, 0.1);

    while optimizer.dp_step_per_example(&params, &batch).is_ok() {}

    let budget = match optimizer.get_privacy_budget() {
        Ok(budget) => budget,
        Err(err) => panic!("budget query failed: {err}"),
    };
    assert!(
        budget.epsilon_consumed <= 1.5 + 1e-12,
        "reported spend {} exceeded the target 1.5",
        budget.epsilon_consumed
    );
    assert_eq!(
        budget.epsilon_remaining,
        (1.5 - budget.epsilon_consumed).max(0.0)
    );

    // Repeated calls after exhaustion keep failing and never spend more.
    for _ in 0..5 {
        assert!(optimizer.dp_step_per_example(&params, &batch).is_err());
    }
    let after = match optimizer.get_privacy_budget() {
        Ok(budget) => budget,
        Err(err) => panic!("budget query failed: {err}"),
    };
    assert_eq!(after.steps_taken, budget.steps_taken);
    assert!((after.epsilon_consumed - budget.epsilon_consumed).abs() < 1e-15);
}

#[test]
fn dp_sgd_optimizer_runs_a_full_training_loop() {
    let mut optimizer = match DPSGDOptimizer::<_, f64, Ix1>::new(SGD::new(0.05), roomy_config()) {
        Ok(optimizer) => optimizer,
        Err(err) => panic!("failed to construct DP-SGD: {err}"),
    };

    let mut params = Array1::<f64>::zeros(8);
    let batch = batch_of(16, 8, 0.2);
    let mut previous = 0.0;

    for step in 1..=100 {
        params = match optimizer.dp_step_per_example(&params, &batch) {
            Ok(updated) => updated,
            Err(err) => panic!("step {step} failed: {err}"),
        };
        let epsilon = match optimizer.consumed_epsilon() {
            Ok(value) => value,
            Err(err) => panic!("accounting failed: {err}"),
        };
        assert!(epsilon > previous);
        previous = epsilon;
    }

    let details = optimizer.get_privacy_accounting_details();
    assert_eq!(details.accounting_segments.len(), 1);
    assert_eq!(details.accounting_segments[0].steps, 100);
    assert_eq!(details.privacy_consumption_history.len(), 100);
    assert!(params.iter().all(|value| value.is_finite()));
}

#[test]
fn aggregate_clipping_requires_explicit_opt_in() {
    let mut optimizer = build_dp_optimizer(roomy_config());
    let params = Array1::<f64>::zeros(4);
    let mut gradients = Array1::from_elem(4, 0.3);

    match optimizer.dp_step(&params, &mut gradients) {
        Err(OptimError::InvalidPrivacyConfig(message)) => {
            assert!(
                message.contains("per-example"),
                "the error must explain the missing guarantee: {message}"
            );
        }
        other => panic!("aggregate clipping must be opt-in, got {other:?}"),
    }

    let mut acknowledged = build_dp_optimizer(DifferentialPrivacyConfig {
        acknowledge_aggregate_clipping: true,
        ..roomy_config()
    });
    assert!(acknowledged.dp_step(&params, &mut gradients).is_ok());
}

#[test]
fn per_example_clipping_bounds_the_influence_of_one_example() {
    // One example with an enormous gradient must not move the released mean
    // by more than C / batch_size (plus the noise the caller asked for).
    let config = DifferentialPrivacyConfig {
        noise_multiplier: 0.001,
        l2_norm_clip: 1.0,
        target_epsilon: 1e9,
        batch_size: 8,
        dataset_size: 50_000,
        ..roomy_config()
    };
    let mut optimizer = build_dp_optimizer(config);
    let params = Array1::<f64>::zeros(4);

    let mut batch = batch_of(8, 4, 0.0);
    batch[0] = Array1::from_vec(vec![1.0e9, 0.0, 0.0, 0.0]);

    let updated = match optimizer.dp_step_per_example(&params, &batch) {
        Ok(updated) => updated,
        Err(err) => panic!("step failed: {err}"),
    };

    // SGD with lr = 0.05 applied to a mean gradient bounded by C / 8.
    let bound = 0.05 * (1.0 / 8.0) * 1.1;
    assert!(
        updated[0].abs() <= bound,
        "clipping failed to bound one example's influence: {} > {bound}",
        updated[0].abs()
    );
}

#[test]
fn two_optimizers_produce_different_noise() {
    // Deterministic noise (a hardcoded RNG seed) would make every released
    // gradient reproducible and the privacy guarantee void.
    let params = Array1::<f64>::zeros(128);
    let batch = batch_of(1, 128, 0.0);

    let mut first = build_dp_optimizer(roomy_config());
    let mut second = build_dp_optimizer(roomy_config());

    let a = match first.dp_step_per_example(&params, &batch) {
        Ok(value) => value,
        Err(err) => panic!("step failed: {err}"),
    };
    let b = match second.dp_step_per_example(&params, &batch) {
        Ok(value) => value,
        Err(err) => panic!("step failed: {err}"),
    };

    let differing = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| (*x - *y).abs() > 1e-12)
        .count();
    assert!(
        differing > 100,
        "two independent optimizers produced near-identical noise ({differing}/128 differ)"
    );

    // The noise is centred: with C = 1 and sigma = 1.1 the released mean of a
    // zero batch is pure noise, so it must not be systematically zero.
    let magnitude: f64 = a.iter().map(|value| value.abs()).sum::<f64>() / a.len() as f64;
    assert!(magnitude > 0.0, "no noise was added at all");
}

#[test]
fn renyi_accountant_golden_values_for_the_canonical_dp_sgd_setup() {
    // Canonical DP-SGD configuration: sigma = 1.0, q = 0.01, T = 1000,
    // delta = 1e-5.
    //
    // This is the *shape* check; the exact value is pinned by
    // `GOLDEN_EPSILON_SCHEDULE` above (2.107753075452), whose provenance is
    // documented there. No published Opacus/TF-Privacy figure exists for this
    // particular configuration -- the external anchor is the TF-Privacy
    // tutorial reproduction in
    // `privacy::renyi_accountant::tests::
    //  test_kernel_reproduces_the_published_tensorflow_privacy_reference`.
    //
    // The band below is deliberately generous but excludes both historical
    // defects: the old moments accountant in privacy/mod.rs reported 0.41
    // (a ~2x under-report, the dangerous direction) and the copy in
    // moment_accountant.rs reported 177.2 (a ~65x over-report).
    let mut accountant = RenyiAccountant::with_default_orders();
    match accountant.add_subsampled_gaussian(1.0, 0.01, 1000) {
        Ok(()) => {}
        Err(err) => panic!("composition failed: {err}"),
    }
    let conversion = match accountant.to_epsilon_delta(1e-5) {
        Ok(conversion) => conversion,
        Err(err) => panic!("conversion failed: {err}"),
    };

    assert!(
        (1.0..=6.0).contains(&conversion.epsilon),
        "canonical DP-SGD epsilon outside the expected band: {}",
        conversion.epsilon
    );
    assert_eq!(conversion.delta, 1e-5);
    assert!(conversion.best_order > 1.0);

    // Halving the noise multiplier must increase epsilon substantially.
    let mut noisier = RenyiAccountant::with_default_orders();
    match noisier.add_subsampled_gaussian(2.0, 0.01, 1000) {
        Ok(()) => {}
        Err(err) => panic!("composition failed: {err}"),
    }
    let quiet = match noisier.to_epsilon_delta(1e-5) {
        Ok(conversion) => conversion.epsilon,
        Err(err) => panic!("conversion failed: {err}"),
    };
    assert!(
        quiet < conversion.epsilon,
        "more noise must cost less privacy: {quiet} vs {}",
        conversion.epsilon
    );
}

/// Golden `(steps, epsilon, optimal Renyi order)` schedule for the canonical
/// DP-SGD configuration `sigma = 1.0`, `q = 0.01`, `delta = 1e-5`.
///
/// # Provenance
///
/// These are **not** copied from an Opacus or TensorFlow Privacy run: they pin
/// *this* crate's configuration (exact integer-order sampled-Gaussian RDP,
/// fractional orders bounded by `ceil`, the `DEFAULT_ALPHAS` grid, and the
/// Canonne-Kamath-Steinke RDP-to-DP conversion).
///
/// What makes them golden rather than self-referential is the external anchor
/// in `privacy::renyi_accountant::tests::
/// test_kernel_reproduces_the_published_tensorflow_privacy_reference`: with the
/// classic Mironov conversion that TF Privacy uses, the same kernel reproduces
/// the published tutorial value `eps = 1.18` (1.1799007 at alpha = 17) for
/// `N = 60000, B = 250, sigma = 1.3, 15 epochs`. The kernel is additionally
/// pinned against Simpson quadrature of the Renyi divergence integral and
/// against the closed-form Gaussian RDP `alpha / (2 sigma^2)` at `q -> 1`.
///
/// Composition and conversion then determine the values below exactly.
const GOLDEN_EPSILON_SCHEDULE: [(usize, f64); 4] = [
    (1, 0.956_281_055_679),
    (10, 1.064_496_195_732),
    (100, 1.224_845_779_636),
    (1000, 2.107_753_075_452),
];

/// Configuration realising `sigma = 1.0`, `q = 100/10_000 = 0.01`.
fn golden_config() -> DifferentialPrivacyConfig {
    DifferentialPrivacyConfig {
        target_epsilon: 5.0,
        target_delta: 1e-5,
        noise_multiplier: 1.0,
        l2_norm_clip: 1.0,
        batch_size: 100,
        dataset_size: 10_000,
        max_steps: 1000,
        accounting_method: AccountingMethod::RenyiDP,
        ..Default::default()
    }
}

#[test]
fn thousand_end_to_end_dp_steps_reproduce_the_golden_epsilon_schedule() {
    // The end-to-end regression for F114/F115: `dp_step` used to fail on the
    // very first call (delta_remaining = target - target = 0) or succeed
    // exactly once and then report the budget as exhausted. This test drives
    // 1000 real steps through the public per-example entry point and pins the
    // epsilon reported after 1, 10, 100 and 1000 of them.
    let mut optimizer = match DPSGDOptimizer::<_, f64, Ix1>::new(SGD::new(0.05), golden_config()) {
        Ok(optimizer) => optimizer,
        Err(err) => panic!("failed to construct DP-SGD: {err}"),
    };

    let params = Array1::<f64>::zeros(8);
    // 100 per-example gradients => batch size 100 => q = 100 / 10_000 = 0.01.
    let batch = batch_of(100, 8, 0.02);

    let mut checkpoint = 0usize;
    let mut previous_epsilon = 0.0;

    for step in 1..=1000usize {
        match optimizer.dp_step_per_example(&params, &batch) {
            Ok(updated) => assert!(
                updated.iter().all(|value| value.is_finite()),
                "step {step} produced non-finite parameters"
            ),
            Err(err) => panic!("step {step} failed: {err}"),
        }

        let epsilon = match optimizer.consumed_epsilon() {
            Ok(value) => value,
            Err(err) => panic!("accounting failed at step {step}: {err}"),
        };
        assert!(
            epsilon > previous_epsilon,
            "step {step}: epsilon must increase strictly ({previous_epsilon} -> {epsilon})"
        );
        previous_epsilon = epsilon;

        if checkpoint < GOLDEN_EPSILON_SCHEDULE.len()
            && GOLDEN_EPSILON_SCHEDULE[checkpoint].0 == step
        {
            let expected = GOLDEN_EPSILON_SCHEDULE[checkpoint].1;
            assert!(
                (epsilon - expected).abs() < 1e-9,
                "after {step} steps epsilon = {epsilon}, golden value {expected}"
            );
            checkpoint += 1;
        }
    }

    assert_eq!(
        checkpoint,
        GOLDEN_EPSILON_SCHEDULE.len(),
        "every golden checkpoint must have been reached"
    );

    // A short-circuit must not be able to masquerade as success.
    let budget = match optimizer.get_privacy_budget() {
        Ok(budget) => budget,
        Err(err) => panic!("budget query failed: {err}"),
    };
    assert_eq!(budget.steps_taken, 1000);
    assert_eq!(budget.delta_consumed, 0.0);
    assert_eq!(budget.delta_remaining, 1e-5);

    let segments = optimizer.accounting_segments();
    assert_eq!(segments.len(), 1, "constant parameters must be one segment");
    assert_eq!(segments[0].noise_multiplier, 1.0);
    assert_eq!(segments[0].sampling_probability, 0.01);
    assert_eq!(segments[0].steps, 1000);

    // max_steps = 1000: the 1001st step must be refused, not silently taken.
    match optimizer.dp_step_per_example(&params, &batch) {
        Err(OptimError::PrivacyBudgetExhausted { .. }) => {}
        other => panic!("the step cap must be enforced, got {other:?}"),
    }
}

#[test]
fn per_step_composition_matches_one_shot_composition_and_both_optimizers_agree() {
    // Driving 100 real steps must cost exactly what composing 100 steps in one
    // call costs -- an accountant that rebuilt itself per step (F128) or
    // ignored the batch size (F129) would diverge here.
    let steps = 100usize;
    let golden = GOLDEN_EPSILON_SCHEDULE[2].1;
    assert_eq!(GOLDEN_EPSILON_SCHEDULE[2].0, steps);

    let mut reference = match build_accountant(AccountingMethod::RenyiDP, 1.0, 1e-5, 100, 10_000) {
        Ok(accountant) => accountant,
        Err(err) => panic!("failed to build the reference accountant: {err}"),
    };
    match reference.compose_subsampled_gaussian(1.0, 0.01, steps) {
        Ok(()) => {}
        Err(err) => panic!("composition failed: {err}"),
    }
    let (eps_reference, delta_reference) = match reference.privacy_spent(1e-5) {
        Ok(value) => value,
        Err(err) => panic!("conversion failed: {err}"),
    };
    assert_eq!(delta_reference, 1e-5);
    assert!(
        (eps_reference - golden).abs() < 1e-9,
        "one-shot composition {eps_reference} must equal the golden value {golden}"
    );

    let params = Array1::<f64>::zeros(8);
    let batch = batch_of(100, 8, 0.02);

    let mut dp_sgd = match DPSGDOptimizer::<_, f64, Ix1>::new(SGD::new(0.05), golden_config()) {
        Ok(optimizer) => optimizer,
        Err(err) => panic!("failed to construct DP-SGD: {err}"),
    };
    let mut wrapper = build_dp_optimizer(golden_config());

    for step in 1..=steps {
        match dp_sgd.dp_step_per_example(&params, &batch) {
            Ok(_) => {}
            Err(err) => panic!("DPSGDOptimizer step {step} failed: {err}"),
        }
        match wrapper.dp_step_per_example(&params, &batch) {
            Ok(_) => {}
            Err(err) => panic!("DifferentiallyPrivateOptimizer step {step} failed: {err}"),
        }
    }

    let eps_dp_sgd = match dp_sgd.consumed_epsilon() {
        Ok(value) => value,
        Err(err) => panic!("accounting failed: {err}"),
    };
    let eps_wrapper = match wrapper.consumed_epsilon() {
        Ok(value) => value,
        Err(err) => panic!("accounting failed: {err}"),
    };

    assert!(
        (eps_dp_sgd - eps_reference).abs() < 1e-12,
        "per-step composition {eps_dp_sgd} must equal one-shot {eps_reference}"
    );
    assert!(
        (eps_wrapper - eps_reference).abs() < 1e-12,
        "the two DP optimizers must report the same spend: {eps_wrapper} vs {eps_reference}"
    );
}

#[test]
fn aggregate_dp_step_is_accounted_without_subsampling_amplification() {
    // `dp_step` clips an already aggregated gradient, so it only supports
    // batch-level adjacency. It must therefore be charged at q = 1 -- charging
    // it at q = batch/dataset would claim amplification the mechanism does not
    // have.
    let config = DifferentialPrivacyConfig {
        target_epsilon: 1e6,
        acknowledge_aggregate_clipping: true,
        ..golden_config()
    };
    let mut optimizer = match DPSGDOptimizer::<_, f64, Ix1>::new(SGD::new(0.05), config) {
        Ok(optimizer) => optimizer,
        Err(err) => panic!("failed to construct DP-SGD: {err}"),
    };

    let params = Array1::<f64>::zeros(4);
    let mut gradients = Array1::from_elem(4, 0.5);
    for step in 1..=10usize {
        match optimizer.dp_step(&params, &mut gradients, 100) {
            Ok(_) => {}
            Err(err) => panic!("aggregate step {step} failed: {err}"),
        }
    }

    let segments = optimizer.accounting_segments();
    assert_eq!(segments.len(), 1);
    assert_eq!(
        segments[0].sampling_probability, 1.0,
        "aggregate clipping must not claim subsampling amplification"
    );
    assert_eq!(segments[0].steps, 10);

    // With q = 1 the sampled Gaussian *is* the Gaussian mechanism, whose RDP
    // has the closed form alpha / (2 sigma^2) per step. For sigma = 1 and
    // T = 10 the CKS conversion over DEFAULT_ALPHAS therefore evaluates
    //
    //     min_alpha [ 10 * alpha / 2 + ln((alpha - 1)/alpha)
    //                 - (ln(1e-5) + ln(alpha)) / (alpha - 1) ]
    //
    // which is 19.0535975316... attained at alpha = 2.5. Pinning it two-sided
    // catches both a mechanism that silently claimed amplification (epsilon
    // would collapse towards the subsampled value) and one that stopped
    // accounting at all.
    let epsilon = match optimizer.consumed_epsilon() {
        Ok(value) => value,
        Err(err) => panic!("accounting failed: {err}"),
    };
    assert!(
        (epsilon - 19.053_597_531_631_39).abs() < 1e-9,
        "un-amplified accounting must match the closed-form Gaussian value, got {epsilon}"
    );
    assert!(
        epsilon > 10.0 * GOLDEN_EPSILON_SCHEDULE[1].1,
        "un-amplified accounting must cost far more than the subsampled one: {epsilon}"
    );
}

#[test]
fn accountants_agree_and_compose_additively() {
    let mut renyi = match build_accountant(AccountingMethod::RenyiDP, 1.0, 1e-5, 100, 10_000) {
        Ok(accountant) => accountant,
        Err(err) => panic!("failed to build the Renyi accountant: {err}"),
    };
    let mut moments =
        match build_accountant(AccountingMethod::MomentsAccountant, 1.0, 1e-5, 100, 10_000) {
            Ok(accountant) => accountant,
            Err(err) => panic!("failed to build the moments accountant: {err}"),
        };

    for accountant in [&mut renyi, &mut moments] {
        match accountant.compose_subsampled_gaussian(1.0, 0.01, 1000) {
            Ok(()) => {}
            Err(err) => panic!("composition failed: {err}"),
        }
    }

    let (eps_renyi, _) = match renyi.privacy_spent(1e-5) {
        Ok(value) => value,
        Err(err) => panic!("conversion failed: {err}"),
    };
    let (eps_moments, _) = match moments.privacy_spent(1e-5) {
        Ok(value) => value,
        Err(err) => panic!("conversion failed: {err}"),
    };

    // Both compose the same exact per-step bound; only the conversion and the
    // order grid differ, so they must agree closely. The historical pair
    // disagreed by a factor of 436.
    let ratio = eps_moments / eps_renyi;
    assert!(
        (0.5..=2.0).contains(&ratio),
        "the two accountants disagree: renyi = {eps_renyi}, moments = {eps_moments}"
    );

    assert_eq!(renyi.total_steps(), 1000);
    assert_eq!(moments.total_steps(), 1000);
    assert_eq!(renyi.segments().len(), 1);
}

#[test]
fn unimplemented_accounting_methods_are_refused() {
    for method in [
        AccountingMethod::AdvancedComposition,
        AccountingMethod::ZCDP,
    ] {
        assert!(
            build_accountant(method, 1.0, 1e-5, 100, 10_000).is_err(),
            "{method:?} must not be silently substituted with another analysis"
        );

        let config = DifferentialPrivacyConfig {
            accounting_method: method,
            ..roomy_config()
        };
        assert!(
            DifferentiallyPrivateOptimizer::<SGD<f64>, f64, Ix1>::new(SGD::new(0.05), config)
                .is_err()
        );
    }
}

#[test]
fn appending_segments_never_reduces_the_reported_spend() {
    let mut accountant = match build_accountant(AccountingMethod::RenyiDP, 1.0, 1e-5, 100, 10_000) {
        Ok(accountant) => accountant,
        Err(err) => panic!("failed to build the accountant: {err}"),
    };

    let mut previous = 0.0;
    // Alternate the parameters so a new segment is opened each time.
    for (index, (sigma, q)) in [(1.0, 0.01), (2.0, 0.02), (1.5, 0.005), (1.0, 0.01)]
        .into_iter()
        .enumerate()
    {
        match accountant.compose_subsampled_gaussian(sigma, q, 50) {
            Ok(()) => {}
            Err(err) => panic!("composition {index} failed: {err}"),
        }
        let (epsilon, _) = match accountant.privacy_spent(1e-5) {
            Ok(value) => value,
            Err(err) => panic!("conversion failed: {err}"),
        };
        assert!(
            epsilon > previous,
            "composing more mechanisms must never reduce epsilon ({previous} -> {epsilon})"
        );
        previous = epsilon;
    }

    assert_eq!(accountant.segments().len(), 4);
    assert_eq!(accountant.total_steps(), 200);
}

#[test]
fn gaussian_noise_scale_regression() {
    // sqrt(2 * ln(1.25 / 1e-5)) = 4.844813...; the historical implementation
    // computed ln(1 + 0.25/delta) instead of ln(1.25/delta) and under-noised
    // by ~7%.
    let sigma = match GaussianMechanism::<f64>::compute_noise_scale(1.0, 1.0, 1e-5) {
        Ok(value) => value,
        Err(err) => panic!("noise scale computation failed: {err}"),
    };
    assert!(
        (sigma - 4.8448).abs() < 1e-4,
        "expected sigma ~= 4.8448, got {sigma}"
    );

    // The classic bound is not valid above epsilon = 1 and must be refused.
    assert!(GaussianMechanism::<f64>::compute_noise_scale(1.0, 2.0, 1e-5).is_err());
}

#[test]
fn noise_mechanisms_are_seeded_independently() {
    let mut first = GaussianMechanism::<f64>::new();
    let mut second = GaussianMechanism::<f64>::new();
    let mut a = Array1::<f64>::zeros(256);
    let mut b = Array1::<f64>::zeros(256);

    match first.add_noise_1d(&mut a, 1.0, 1.0, Some(1e-5)) {
        Ok(()) => {}
        Err(err) => panic!("noise addition failed: {err}"),
    }
    match second.add_noise_1d(&mut b, 1.0, 1.0, Some(1e-5)) {
        Ok(()) => {}
        Err(err) => panic!("noise addition failed: {err}"),
    }

    let differing = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| (*x - *y).abs() > 1e-12)
        .count();
    assert!(
        differing > 200,
        "two mechanism instances produced identical noise ({differing}/256 differ)"
    );

    // Empirical standard deviation should be close to the calibrated sigma.
    let mean = a.iter().sum::<f64>() / a.len() as f64;
    let variance = a.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / a.len() as f64;
    let observed = variance.sqrt();
    assert!(
        observed > 2.0 && observed < 8.0,
        "observed noise scale {observed} is far from the calibrated 4.84"
    );
}

#[test]
fn laplace_mechanism_provides_pure_epsilon_dp() {
    let mut mechanism = LaplaceMechanism::<f64>::new();
    let mut data = Array1::<f64>::zeros(4096);
    match mechanism.add_noise_1d(&mut data, 1.0, 1.0, None) {
        Ok(()) => {}
        Err(err) => panic!("noise addition failed: {err}"),
    }

    // Lap(b = sensitivity / epsilon = 1) has variance 2 b^2 = 2, whereas a
    // Gaussian with the same nominal scale would have variance 1. The
    // historical implementation sampled a Normal and called it Laplace.
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    assert!(
        (variance - 2.0).abs() < 0.5,
        "Laplace variance should be ~2, observed {variance}"
    );

    let params = match mechanism.get_parameters() {
        Some(params) => params,
        None => panic!("a used mechanism must report its parameters"),
    };
    assert_eq!(params.mechanism_type, "Laplace");
    assert_eq!(params.scale, 1.0);
    assert!(params.delta.is_none());
}

#[test]
fn sparse_vector_technique_answers_only_above_threshold_queries() {
    let mut svt = match SparseVectorMechanism::<f64>::new(0.0, 1.0, 0.5, 0.5, Some(1.0), 2) {
        Ok(svt) => svt,
        Err(err) => panic!("construction failed: {err}"),
    };

    // Far-below-threshold queries are free and return None.
    for _ in 0..25 {
        match svt.answer_query(-1.0e6) {
            Ok(None) => {}
            other => panic!("below-threshold query must return Ok(None), got {other:?}"),
        }
    }
    assert_eq!(svt.queries_answered(), 0);

    // Two above-threshold answers, then the mechanism halts.
    for _ in 0..2 {
        match svt.answer_query(1.0e6) {
            Ok(Some(value)) => assert!(value.is_finite()),
            other => panic!("above-threshold query must return a value, got {other:?}"),
        }
    }
    assert!(svt.is_halted());
    assert!(matches!(
        svt.answer_query(1.0e6),
        Err(OptimError::PrivacyBudgetExhausted { .. })
    ));
}

#[test]
fn laplace_optimizer_path_reports_delta_zero() {
    let config = DifferentialPrivacyConfig {
        noise_mechanism: NoiseMechanism::Laplace,
        target_epsilon: 4.0,
        max_steps: 8,
        ..roomy_config()
    };
    let mut optimizer = build_dp_optimizer(config);
    let params = Array1::<f64>::zeros(2);
    let batch = batch_of(4, 2, 0.1);

    for step in 1..=8 {
        match optimizer.dp_step_per_example(&params, &batch) {
            Ok(_) => {}
            Err(err) => panic!("step {step} failed: {err}"),
        }
    }

    let budget = match optimizer.get_privacy_budget() {
        Ok(budget) => budget,
        Err(err) => panic!("budget query failed: {err}"),
    };
    assert!((budget.epsilon_consumed - 4.0).abs() < 1e-12);
    assert_eq!(
        budget.delta_remaining, 0.0,
        "the Laplace mechanism provides pure epsilon-DP (delta = 0)"
    );
    assert!(optimizer.dp_step_per_example(&params, &batch).is_err());
}

#[test]
fn unimplemented_noise_mechanisms_are_refused() {
    for mechanism in [
        NoiseMechanism::TreeAggregation,
        NoiseMechanism::ImprovedComposition,
    ] {
        let mut optimizer = build_dp_optimizer(DifferentialPrivacyConfig {
            noise_mechanism: mechanism,
            ..roomy_config()
        });
        let params = Array1::<f64>::zeros(2);
        let batch = batch_of(2, 2, 0.1);
        assert!(
            optimizer.dp_step_per_example(&params, &batch).is_err(),
            "{mechanism:?} must not silently fall back to Gaussian noise"
        );
    }
}

#[test]
fn invalid_configurations_are_rejected_at_construction() {
    let cases = [
        DifferentialPrivacyConfig {
            noise_multiplier: 0.0,
            ..roomy_config()
        },
        DifferentialPrivacyConfig {
            target_delta: 0.0,
            ..roomy_config()
        },
        DifferentialPrivacyConfig {
            target_epsilon: -1.0,
            ..roomy_config()
        },
        DifferentialPrivacyConfig {
            l2_norm_clip: 0.0,
            ..roomy_config()
        },
        DifferentialPrivacyConfig {
            batch_size: 10_000,
            dataset_size: 100,
            ..roomy_config()
        },
        DifferentialPrivacyConfig {
            max_steps: 0,
            ..roomy_config()
        },
    ];

    for (index, config) in cases.into_iter().enumerate() {
        assert!(
            DifferentiallyPrivateOptimizer::<SGD<f64>, f64, Ix1>::new(SGD::new(0.05), config)
                .is_err(),
            "invalid configuration {index} was accepted"
        );
    }
}

#[test]
fn max_steps_is_enforced_independently_of_the_budget() {
    let config = DifferentialPrivacyConfig {
        target_epsilon: 1e9,
        max_steps: 7,
        ..roomy_config()
    };
    let mut optimizer = build_dp_optimizer(config);
    let params = Array1::<f64>::zeros(2);
    let batch = batch_of(2, 2, 0.1);

    for step in 1..=7 {
        match optimizer.dp_step_per_example(&params, &batch) {
            Ok(_) => {}
            Err(err) => panic!("step {step} failed: {err}"),
        }
    }
    assert!(matches!(
        optimizer.dp_step_per_example(&params, &batch),
        Err(OptimError::PrivacyBudgetExhausted { .. })
    ));
}
