//! F64 regression tests: Xavier/Glorot **uniform** initialization must hit
//! Glorot's target variance `2 / (fan_in + fan_out)`.
//!
//! The bug these pin down was a units error repeated across the crate:
//! `sqrt(2 / (fan_in + fan_out))` — the *standard deviation* Glorot prescribes for
//! a **normal** draw — was passed as the *half-width* of a **uniform** draw.
//! `Uniform[-b, b]` has variance `b² / 3`, so every affected weight matrix started
//! with exactly one third of the intended variance (a `1/sqrt(3) ≈ 0.577` factor
//! on the standard deviation). Attenuating the initial signal like that shrinks
//! both activations and back-propagated gradients layer by layer.
//!
//! The correct uniform limit is `sqrt(6 / (fan_in + fan_out))`.
//!
//! These are statistical assertions over thousands of samples, so the bands are
//! wide enough to be robust but far narrower than the `3×` variance error they
//! exist to catch.

use optirs_learned::lstm::{LSTMNetwork, OutputProjection, OutputTransform};
use optirs_learned::transformer_based_optimizer::MultiHeadAttention;
use optirs_learned::LearnedOptimizerConfig;
use scirs2_core::ndarray::Array2;

/// Sample variance of a matrix's entries.
fn variance(values: &Array2<f64>) -> f64 {
    let n = values.len() as f64;
    assert!(n > 1.0, "need at least two samples");
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n
}

/// Glorot's target variance for a layer with the given fans.
fn glorot_target(fan_in: usize, fan_out: usize) -> f64 {
    2.0 / (fan_in + fan_out) as f64
}

/// The transformer optimizer's multi-head attention projections are square
/// `model_dim × model_dim`, so both fans are `model_dim`. The old code used
/// `sqrt(2 / (model_dim + head_dim))` as a uniform limit: at 128/8 that is a
/// variance of `(2/144)/3 = 4.63e-3` against a target of `2/256 = 7.81e-3`
/// — 0.59×, from the units error (0.33×) partly masked by the wrong fan pair
/// (1.78×).
#[test]
fn transformer_attention_projections_hit_the_glorot_target_variance() {
    const MODEL_DIM: usize = 128;
    const HEADS: usize = 8;
    const HEAD_DIM: usize = MODEL_DIM / HEADS;

    // The limit helper is the single source of truth and is public, so pin it
    // directly as well: it must be sqrt(6/(fan_in+fan_out)), not sqrt(2/...).
    let limit = MultiHeadAttention::<f64>::xavier_limit(MODEL_DIM, MODEL_DIM);
    let expected_limit = (6.0 / (2.0 * MODEL_DIM as f64)).sqrt();
    assert!(
        (limit - expected_limit).abs() < 1e-12,
        "xavier_limit returned {limit}, expected {expected_limit}"
    );
    // Uniform[-b, b] variance is b²/3, and that must equal Glorot's target.
    let implied = limit * limit / 3.0;
    let target = glorot_target(MODEL_DIM, MODEL_DIM);
    assert!(
        (implied / target - 1.0).abs() < 1e-12,
        "a uniform draw with limit {limit} has variance {implied}, target is {target}"
    );

    // And the constructed weights must actually follow it.
    let mut attention =
        MultiHeadAttention::<f64>::new(HEADS, MODEL_DIM, HEAD_DIM).expect("attention construction");
    assert_eq!(
        attention.parameter_count(),
        4 * MODEL_DIM * MODEL_DIM,
        "four square projections expected"
    );

    let check = |attention: &MultiHeadAttention<f64>, stage: &str| {
        let (query, key, value, output) = attention.projection_snapshots();
        for (name, matrix) in [
            ("query", query),
            ("key", key),
            ("value", value),
            ("output", output),
        ] {
            assert_eq!(matrix.dim(), (MODEL_DIM, MODEL_DIM));
            let observed = variance(matrix);
            let ratio = observed / target;
            assert!(
                (0.9..1.1).contains(&ratio),
                "{stage} {name} projection variance {observed:.3e} is {ratio:.3}× the Glorot \
                 target {target:.3e}; the old sqrt(2/(model_dim+head_dim)) limit gives 0.59×"
            );
        }
    };

    check(&attention, "constructed");
    // `reset` re-draws and must use the same corrected limit.
    attention.reset().expect("reset");
    check(&attention, "reset");
}

/// `OutputProjection` is the LSTM optimizer's output layer. `fan_in` and
/// `fan_out` differ here, so this pins the fan pair as well as the constant.
#[test]
fn lstm_output_projection_hits_the_glorot_target_variance() {
    const IN: usize = 64;
    const OUT: usize = 192;

    let mut projection =
        OutputProjection::<f64>::new(IN, OUT, OutputTransform::Identity).expect("projection");
    let weights = projection.weight_snapshot();
    assert_eq!(weights.dim(), (OUT, IN));

    let target = glorot_target(IN, OUT);
    let observed = variance(weights);
    let ratio = observed / target;
    assert!(
        (0.9..1.1).contains(&ratio),
        "output projection variance {observed:.3e} is {ratio:.3}× the Glorot target \
         {target:.3e}; the sqrt(2/(fan_in+fan_out))-as-uniform-limit bug gives 0.33×"
    );
    // The bias must be zero-initialized, not drawn.
    assert!(projection.bias_snapshot().iter().all(|b| *b == 0.0));

    // `reset` re-draws at a new shape and must use the *new* fans.
    projection.reset(IN, 32);
    let resampled = projection.weight_snapshot();
    assert_eq!(resampled.dim(), (32, IN));
    let reset_ratio = variance(resampled) / glorot_target(IN, 32);
    assert!(
        (0.9..1.1).contains(&reset_ratio),
        "reset projection variance is {reset_ratio:.3}× the Glorot target for its new shape"
    );
}

/// The four LSTM gates are stacked along the rows of `weight_ih` / `weight_hh`,
/// so the fan-out of each gate is `hidden_size`, not `4 · hidden_size`; and the
/// two matrices have different fan-ins, so they must not share one limit — which
/// is exactly what the old code did (`sqrt(2 / (input_size + hidden_size))` for
/// both).
#[test]
fn lstm_cell_gate_matrices_use_per_matrix_fan_pairs() {
    const INPUT: usize = 40;
    const HIDDEN: usize = 96;

    let config = LearnedOptimizerConfig {
        input_features: INPUT,
        hidden_size: HIDDEN,
        num_layers: 1,
        ..Default::default()
    };
    let network = LSTMNetwork::<f64>::new(&config).expect("network");
    assert!(network.layer_count() >= 1);
    let layer = network.layer(0).expect("first layer");

    let weight_ih = layer.weight_ih_snapshot();
    let weight_hh = layer.weight_hh_snapshot();
    assert_eq!(weight_ih.dim(), (4 * HIDDEN, INPUT));
    assert_eq!(weight_hh.dim(), (4 * HIDDEN, HIDDEN));

    let ih_var = variance(weight_ih);
    let hh_var = variance(weight_hh);
    let ih_ratio = ih_var / glorot_target(INPUT, HIDDEN);
    let hh_ratio = hh_var / glorot_target(HIDDEN, HIDDEN);
    assert!(
        (0.9..1.1).contains(&ih_ratio),
        "weight_ih variance {ih_var:.3e} is {ih_ratio:.3}× its Glorot target"
    );
    assert!(
        (0.9..1.1).contains(&hh_ratio),
        "weight_hh variance {hh_var:.3e} is {hh_ratio:.3}× its Glorot target"
    );

    // Different fan sums (136 vs 192) mean different variances; a shared limit
    // would make them equal, and the narrower fan-in must give the larger one.
    assert!(
        (ih_var / hh_var - 1.0).abs() > 0.1,
        "weight_ih ({ih_var:.3e}) and weight_hh ({hh_var:.3e}) have nearly equal variance, so \
         they are sharing one limit instead of using their own fan pairs"
    );
    assert!(
        ih_var > hh_var,
        "weight_ih has fan sum {} vs weight_hh's {}, so its variance must be larger",
        INPUT + HIDDEN,
        2 * HIDDEN
    );
}

/// The LSTM attention projections are square, so all four share one correct
/// limit — but that limit still has to be the *uniform* one.
#[test]
fn lstm_attention_projections_hit_the_glorot_target_variance() {
    const HIDDEN: usize = 128;

    let config = LearnedOptimizerConfig {
        input_features: 32,
        hidden_size: HIDDEN,
        num_layers: 1,
        use_attention: true,
        attention_heads: 8,
        ..Default::default()
    };
    let network = LSTMNetwork::<f64>::new(&config).expect("network");
    let attention = network
        .attention_snapshot()
        .expect("use_attention = true must build an attention mechanism");
    let (query, key, value, output) = attention.projection_snapshots();

    let target = glorot_target(HIDDEN, HIDDEN);
    for (name, matrix) in [
        ("query", query),
        ("key", key),
        ("value", value),
        ("output", output),
    ] {
        assert_eq!(matrix.dim(), (HIDDEN, HIDDEN));
        let ratio = variance(matrix) / target;
        assert!(
            (0.9..1.1).contains(&ratio),
            "{name} projection variance is {ratio:.3}× the Glorot target {target:.3e}"
        );
    }
}
