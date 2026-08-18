//! Integration tests for the LSTM optimizer's building-block layers.
//!
//! # F5 regression
//!
//! `OutputProjection`, `AttentionMechanism`, and `LayerNormalization` (all in
//! `optirs_learned::lstm`) used to have `forward` implementations that were
//! simply `Ok(input.clone())` -- an identity function that silently discarded
//! every learned weight regardless of what the caller passed in. These tests
//! exercise the crate's public API to confirm each layer now performs its
//! real, documented computation.
//!
//! # F4 regression
//!
//! `LSTMOptimizer::lstm_step` used to panic whenever the flattened parameter
//! count did not exactly equal `LearnedOptimizerConfig::output_features`
//! (default 256). It must now transparently resize its output projection and
//! produce a correctly-shaped update instead.

use scirs2_core::ndarray::{arr1, Array1, Array2};

use optirs_learned::lstm::{
    AttentionMechanism, LSTMOptimizer, LayerNormalization, OutputProjection, OutputTransform,
};
use optirs_learned::LearnedOptimizerConfig;

#[test]
fn output_projection_computes_a_real_linear_map() {
    let projection = OutputProjection::<f64>::new(4, 3, OutputTransform::Identity)
        .expect("projection should construct");

    let zero_in = Array1::<f64>::zeros(4);
    let out_zero = projection.forward(&zero_in).expect("forward on zeros");
    // W*0 + b == b; with a zero-initialized bias this must be exactly zero,
    // but the *point* of the test is that it is a real, addressable
    // quantity (bias), not an identity copy of a 4-wide input into a
    // 3-wide output (which would be a shape error under the old stub).
    assert_eq!(out_zero.len(), 3);
    assert!(out_zero.iter().all(|&v| v == 0.0));

    // A nonzero, Xavier-initialized weight matrix must make two different
    // inputs produce two different outputs (the old `Ok(input.clone())`
    // stub could not even change the output *width*, let alone respond to
    // the input's content).
    let a = arr1(&[1.0, 0.0, 0.0, 0.0]);
    let b = arr1(&[0.0, 1.0, 0.0, 0.0]);
    let out_a = projection.forward(&a).expect("forward a");
    let out_b = projection.forward(&b).expect("forward b");
    assert_eq!(out_a.len(), 3);
    assert_ne!(
        out_a, out_b,
        "a real linear projection must respond to which input feature is set"
    );

    // A mismatched input width must be a typed error, not a panic.
    let bad = Array1::<f64>::zeros(5);
    assert!(projection.forward(&bad).is_err());
}

#[test]
fn output_projection_reset_changes_output_width() {
    let mut projection = OutputProjection::<f64>::new(4, 3, OutputTransform::Identity)
        .expect("projection should construct");
    assert_eq!(projection.output_size(), 3);

    projection.reset(4, 7);
    assert_eq!(projection.output_size(), 7);
    let out = projection
        .forward(&Array1::<f64>::zeros(4))
        .expect("forward after reset");
    assert_eq!(out.len(), 7);
}

#[test]
fn attention_mechanism_uses_its_learned_projections() {
    let config = LearnedOptimizerConfig {
        hidden_size: 8,
        attention_heads: 2,
        ..LearnedOptimizerConfig::default()
    };
    let mut attention =
        AttentionMechanism::<f64>::new(&config).expect("attention should construct");

    let a = Array1::<f64>::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let b = Array1::<f64>::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    let out_a = attention.forward(&a).expect("forward a");
    let out_b = attention.forward(&b).expect("forward b");

    assert_eq!(out_a.len(), 8);
    assert_ne!(
        out_a, out_b,
        "attention output must depend on the input, not be an identity passthrough"
    );
    // A zero input under zero-initialized query/key/value projections would
    // have produced a zero output even for the old stub, so also check a
    // genuinely nonzero, non-trivial input differs from itself post-forward
    // (i.e. the projections actually transform the vector).
    assert_ne!(out_a.as_slice(), a.as_slice());

    // A mismatched input width must be a typed error, not a panic.
    let bad = Array1::<f64>::zeros(3);
    assert!(attention.forward(&bad).is_err());
}

#[test]
fn attention_mechanism_rejects_heads_that_do_not_divide_hidden_size() {
    let config = LearnedOptimizerConfig {
        hidden_size: 10,
        attention_heads: 3,
        ..LearnedOptimizerConfig::default()
    };
    assert!(AttentionMechanism::<f64>::new(&config).is_err());
}

#[test]
fn layer_normalization_standardizes_before_the_affine_transform() {
    let ln = LayerNormalization::<f64>::new(4).expect("layer norm should construct");
    let input = arr1(&[1.0, 2.0, 3.0, 4.0]);
    let out = ln.forward(&input).expect("forward");

    assert_eq!(out.len(), 4);
    // With gamma = 1 and beta = 0 (the freshly-constructed defaults), the
    // output must literally be the standardized input: zero mean, unit
    // variance. The old `Ok(input.clone())` stub would leave mean = 2.5 and
    // variance = 1.25, neither of which is the identity here.
    let mean: f64 = out.iter().sum::<f64>() / out.len() as f64;
    let var: f64 = out.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / out.len() as f64;
    assert!(mean.abs() < 1e-9, "mean should be ~0, got {mean}");
    // `forward` divides by `sqrt(variance + epsilon)` (epsilon = 1e-5) for
    // numerical stability, so the standardized variance is *slightly* below
    // 1, not exactly 1.
    assert!((var - 1.0).abs() < 1e-4, "variance should be ~1, got {var}");

    // A mismatched input width must be a typed error, not a panic.
    let bad = Array1::<f64>::zeros(5);
    assert!(ln.forward(&bad).is_err());
}

#[test]
fn lstm_step_handles_parameter_counts_other_than_output_features() {
    // Deliberately mismatched vs. the default `output_features = 256`: this
    // used to panic inside the final `flat_params - updates` subtraction.
    // `output_features` is left at its stale default (256) on purpose.
    let config = LearnedOptimizerConfig {
        hidden_size: 16,
        input_features: 16,
        num_layers: 1,
        ..LearnedOptimizerConfig::default()
    };
    let mut optimizer = LSTMOptimizer::<f64>::new(config).expect("optimizer should construct");

    let params = Array2::<f64>::from_elem((3, 5), 1.0_f64); // 15 elements, not 256
    let grads = Array2::<f64>::from_elem((3, 5), 0.1_f64);

    let updated = optimizer
        .lstm_step(&params, &grads, Some(1.0))
        .expect("lstm_step must adapt to the real parameter count instead of panicking");
    assert_eq!(updated.shape(), params.shape());

    // A second call at the *same* shape must keep working (no re-panic from
    // a stale cached projection size).
    let updated2 = optimizer
        .lstm_step(&updated, &grads, Some(0.9))
        .expect("second lstm_step at the same shape must also succeed");
    assert_eq!(updated2.shape(), params.shape());
}

/// `AttentionStats::head_diversity` used to be computed from the softmax
/// weights, which at sequence length 1 are all exactly `1.0` — so the figure was
/// the constant `1.0` for every input and every head count, and
/// `temporal_patterns` was a hard-coded `vec![0.0; 10]`. Diversity is now the
/// spread of the pre-softmax per-head logits, which do vary.
#[test]
fn attention_head_diversity_reflects_the_real_per_head_logits() {
    let config = LearnedOptimizerConfig {
        hidden_size: 16,
        attention_heads: 4,
        input_features: 16,
        num_layers: 1,
        use_attention: true,
        ..LearnedOptimizerConfig::default()
    };
    let mut optimizer = LSTMOptimizer::<f64>::new(config).expect("optimizer should construct");

    let params = Array1::<f64>::from_elem(12, 1.0);
    let grads = Array1::<f64>::from_vec((0..12).map(|i| 0.05 * (i as f64 + 1.0)).collect());
    optimizer
        .lstm_step(&params, &grads, Some(1.0))
        .expect("step should succeed");

    let stats = optimizer
        .get_metrics()
        .attention_stats
        .as_ref()
        .expect("attention statistics must be recorded when attention is enabled");

    // One recorded logit per head, not a fixed-length placeholder.
    assert_eq!(
        stats.temporal_patterns.len(),
        4,
        "temporal_patterns must hold one real logit per head, got {:?}",
        stats.temporal_patterns
    );
    assert!(
        stats.temporal_patterns.iter().all(|v| v.is_finite()),
        "logits must be finite: {:?}",
        stats.temporal_patterns
    );
    // Xavier-initialized, independent Q/K projections make an exactly-equal
    // logit across all four heads a measure-zero event, so a genuine spread is
    // observable. The old weights-based formula returned exactly 1.0 here.
    assert!(
        stats.head_diversity > 0.0,
        "head diversity must be a real spread, got {}",
        stats.head_diversity
    );
    assert_ne!(
        stats.head_diversity, 1.0,
        "head diversity of exactly 1.0 is the signature of the old \
         constant-weights formula"
    );
}

/// `adaptation_efficiency` used to be `‖Δθ‖ / ‖g‖`, which simply tracked
/// whatever learning rate the controller had picked. It is now the step-size
/// ratio against the SGD step of the same learning rate, `‖Δθ‖ / (lr · ‖g‖)`.
#[test]
fn adaptation_efficiency_is_normalized_by_the_learning_rate() {
    let config = LearnedOptimizerConfig {
        hidden_size: 16,
        input_features: 16,
        num_layers: 1,
        ..LearnedOptimizerConfig::default()
    };
    let mut optimizer = LSTMOptimizer::<f64>::new(config).expect("optimizer should construct");

    let params = Array1::<f64>::from_elem(8, 1.0);
    let grads = Array1::<f64>::from_elem(8, 0.25);
    let updated = optimizer
        .lstm_step(&params, &grads, Some(1.0))
        .expect("step should succeed");

    let lr = optimizer.current_learning_rate();
    let grad_norm = grads.iter().map(|g| g * g).sum::<f64>().sqrt();
    let update_norm = params
        .iter()
        .zip(updated.iter())
        .map(|(p, u)| (p - u) * (p - u))
        .sum::<f64>()
        .sqrt();

    let reported = optimizer.get_metrics().adaptation_efficiency;
    assert!(reported.is_finite() && reported > 0.0, "got {reported}");
    let expected = update_norm / (lr.abs() * grad_norm);
    assert!(
        (reported - expected).abs() < 1e-9,
        "adaptation_efficiency {reported} should be ‖Δθ‖/(lr·‖g‖) = {expected}"
    );

    // A zero gradient makes the ratio undefined; it must report the neutral
    // 1.0 rather than a NaN or an infinity.
    let zeros = Array1::<f64>::zeros(8);
    optimizer
        .lstm_step(&params, &zeros, Some(1.0))
        .expect("zero-gradient step should succeed");
    assert_eq!(optimizer.get_metrics().adaptation_efficiency, 1.0);
}
