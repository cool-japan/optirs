//! Regression tests for the adaptive transformer enhancement (findings F20-F23).
//!
//! Before this wave:
//!
//! * `OptimizationLandscapeAnalyzer::analyze` ignored both histories and
//!   returned `complexity = 0.5`, `difficulty = 0.3`, `confidence = 0.9`.
//! * `TransformerPerformancePredictor::predict_improvement` returned
//!   `0.15 / 0.92 / 0.85 / 0.05` for every input, and its network was
//!   zero-initialized.
//! * `AdaptiveConfig` was ignored by all five constructors.
//! * `enhance_optimizer` never touched the `&mut TransformerOptimizer` it was
//!   handed.
//! * `calculate_adaptive_learning_rate` returned `×1.1` for even parameter
//!   indices and `×0.9` for odd ones — index-parity noise, not adaptation.
//!
//! These tests live in `tests/` rather than in `adaptive/types.rs` because that
//! file is close to the 2000-line cap.

use optirs_learned::adaptive::{
    AdaptiveConfig, AdaptiveTransformerEnhancement, PredictionFeatures, PredictorSample,
};
use optirs_learned::transformer_based_optimizer::{
    TransformerBasedOptimizerConfig, TransformerOptimizer,
};
use scirs2_core::ndarray::Array1;

fn small_optimizer_config() -> TransformerBasedOptimizerConfig<f64> {
    TransformerBasedOptimizerConfig::<f64> {
        model_dimension: 8,
        num_transformer_layers: 2,
        num_attention_heads: 2,
        attention_head_dimension: 4,
        feedforward_dimension: 16,
        sequence_length: 8,
        dropout_rate: 0.0,
        ..Default::default()
    }
}

fn adaptive_config() -> AdaptiveConfig<f64> {
    AdaptiveConfig::<f64> {
        min_sequence_length: 4,
        max_sequence_length: 32,
        prediction_horizon: 8,
        adaptation_lr: 1e-2,
        landscape_analysis_frequency: 1,
        ..Default::default()
    }
}

fn descending_history() -> (Vec<Array1<f64>>, Vec<f64>) {
    let grads = (0..8)
        .map(|i| Array1::from_vec(vec![0.9_f64.powi(i), 0.0]))
        .collect();
    let losses = (0..8).map(|i| 0.5_f64.powi(i)).collect();
    (grads, losses)
}

fn oscillating_history() -> (Vec<Array1<f64>>, Vec<f64>) {
    let grads = (0..8)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            Array1::from_vec(vec![sign * (1.0 + i as f64), sign * 0.5])
        })
        .collect();
    let losses = (0..8).map(|i| if i % 2 == 0 { 1.0 } else { 2.5 }).collect();
    (grads, losses)
}

// ---------------------------------------------------------------------------
// F20: the landscape analysis is a real function of the history
// ---------------------------------------------------------------------------

#[test]
fn analysis_differs_between_easy_and_hard_histories() {
    let mut enhancement =
        AdaptiveTransformerEnhancement::<f64>::new(adaptive_config()).expect("construction");
    let mut params = Array1::from_vec(vec![1.0, 1.0]);
    let grad = Array1::from_vec(vec![0.5, -0.5]);

    let (easy_grads, easy_losses) = descending_history();
    let easy = enhancement
        .enhanced_optimize_step(&mut params, &grad, &easy_losses, &easy_grads)
        .expect("easy step");
    let easy_complexity = easy.landscape_analysis.complexity;
    let easy_difficulty = easy.landscape_analysis.difficulty;

    let mut enhancement2 =
        AdaptiveTransformerEnhancement::<f64>::new(adaptive_config()).expect("construction");
    let mut params2 = Array1::from_vec(vec![1.0, 1.0]);
    let (hard_grads, hard_losses) = oscillating_history();
    let hard = enhancement2
        .enhanced_optimize_step(&mut params2, &grad, &hard_losses, &hard_grads)
        .expect("hard step");

    assert!(
        hard.landscape_analysis.difficulty > easy_difficulty,
        "oscillating history should be harder: {} vs {}",
        hard.landscape_analysis.difficulty,
        easy_difficulty
    );
    assert!(
        hard.landscape_analysis.complexity > easy_complexity,
        "oscillating history should be more complex: {} vs {}",
        hard.landscape_analysis.complexity,
        easy_complexity
    );
    // The old constants.
    assert!(
        (easy_complexity - 0.5).abs() > 1e-9 || (easy_difficulty - 0.3).abs() > 1e-9,
        "analysis still returns the hardcoded 0.5 / 0.3 pair"
    );
    assert!(
        (easy.landscape_analysis.confidence - 0.9).abs() > 1e-9,
        "confidence is still the hardcoded 0.9"
    );
}

#[test]
fn an_empty_history_is_an_error_not_a_constant() {
    let mut enhancement =
        AdaptiveTransformerEnhancement::<f64>::new(adaptive_config()).expect("construction");
    let mut params = Array1::from_vec(vec![1.0, 1.0]);
    let grad = Array1::from_vec(vec![0.5, -0.5]);
    assert!(
        enhancement
            .enhanced_optimize_step(&mut params, &grad, &[], &[])
            .is_err(),
        "analysing nothing must fail rather than report neutral constants"
    );
}

// ---------------------------------------------------------------------------
// F21: the configuration configures something
// ---------------------------------------------------------------------------

#[test]
fn sequence_length_bounds_come_from_the_configuration() {
    let mut narrow = adaptive_config();
    narrow.min_sequence_length = 4;
    narrow.max_sequence_length = 8;

    let mut wide = adaptive_config();
    wide.min_sequence_length = 256;
    wide.max_sequence_length = 1024;

    let (grads, losses) = oscillating_history();
    let grad = Array1::from_vec(vec![0.5, -0.5]);

    let mut narrow_enh = AdaptiveTransformerEnhancement::<f64>::new(narrow).expect("narrow");
    let mut narrow_params = Array1::from_vec(vec![1.0, 1.0]);
    let narrow_result = narrow_enh
        .enhanced_optimize_step(&mut narrow_params, &grad, &losses, &grads)
        .expect("narrow step");

    let mut wide_enh = AdaptiveTransformerEnhancement::<f64>::new(wide).expect("wide");
    let mut wide_params = Array1::from_vec(vec![1.0, 1.0]);
    let wide_result = wide_enh
        .enhanced_optimize_step(&mut wide_params, &grad, &losses, &grads)
        .expect("wide step");

    assert!(
        narrow_result.sequence_adaptation.new_length <= 8,
        "narrow config produced length {} outside [4, 8]",
        narrow_result.sequence_adaptation.new_length
    );
    assert!(
        wide_result.sequence_adaptation.new_length >= 256,
        "wide config produced length {} below its minimum 256",
        wide_result.sequence_adaptation.new_length
    );
    assert_ne!(
        narrow_result.sequence_adaptation.new_length, wide_result.sequence_adaptation.new_length,
        "the configuration had no effect on the sequence adaptation"
    );
    // The attention span must respect the same window.
    let narrow_span = narrow_result
        .attention_optimization
        .attention_patterns
        .dim()
        .1;
    let wide_span = wide_result
        .attention_optimization
        .attention_patterns
        .dim()
        .1;
    assert!(narrow_span <= 8, "narrow attention span {narrow_span}");
    assert!(wide_span >= 256, "wide attention span {wide_span}");
}

#[test]
fn rejects_inverted_sequence_length_bounds() {
    let mut config = adaptive_config();
    config.min_sequence_length = 64;
    config.max_sequence_length = 8;
    assert!(AdaptiveTransformerEnhancement::<f64>::new(config).is_err());

    let mut zero = adaptive_config();
    zero.min_sequence_length = 0;
    assert!(AdaptiveTransformerEnhancement::<f64>::new(zero).is_err());
}

#[test]
fn adaptation_lr_scales_the_step() {
    let (grads, losses) = descending_history();
    let grad = Array1::from_vec(vec![1.0, 1.0]);

    let step_with = |lr: f64| -> f64 {
        let mut config = adaptive_config();
        config.adaptation_lr = lr;
        let mut enh = AdaptiveTransformerEnhancement::<f64>::new(config).expect("construction");
        let mut params = Array1::from_vec(vec![1.0, 1.0]);
        enh.enhanced_optimize_step(&mut params, &grad, &losses, &grads)
            .expect("step");
        (1.0 - params[0]).abs()
    };

    let small = step_with(1e-4);
    let large = step_with(1e-1);
    assert!(
        large > small * 10.0,
        "adaptation_lr does not scale the step: {small} vs {large}"
    );
}

// ---------------------------------------------------------------------------
// F23: a real per-parameter learning rate
// ---------------------------------------------------------------------------

#[test]
fn equal_gradients_get_equal_steps_regardless_of_index_parity() {
    let mut enhancement =
        AdaptiveTransformerEnhancement::<f64>::new(adaptive_config()).expect("construction");
    let (grads, losses) = descending_history();
    // Four identical gradient components. Under the old index-parity rule the
    // even indices moved 1.1/0.9 = 1.222x further than the odd ones.
    let grad = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
    let start = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
    let mut params = start.clone();

    enhancement
        .enhanced_optimize_step(&mut params, &grad, &losses, &grads)
        .expect("step");

    let deltas: Vec<f64> = params.iter().map(|p| p.abs()).collect();
    assert!(
        deltas[0] > 0.0,
        "no update was applied (delta {})",
        deltas[0]
    );
    for (i, d) in deltas.iter().enumerate() {
        assert!(
            (d - deltas[0]).abs() < 1e-12,
            "component {i} moved {d}, component 0 moved {} — index parity is back",
            deltas[0]
        );
    }
}

#[test]
fn the_step_adapts_to_per_parameter_gradient_scale() {
    let mut enhancement =
        AdaptiveTransformerEnhancement::<f64>::new(adaptive_config()).expect("construction");
    let (grads, losses) = descending_history();
    // Gradient magnitudes four orders of magnitude apart. An RMSProp-style rate
    // normalizes them, so the *steps* end up comparable. The old
    // `base_lr · scale · parity` rule made the step proportional to the
    // gradient, i.e. ~10000x apart.
    let grad = Array1::from_vec(vec![100.0, 0.01]);
    let mut params = Array1::from_vec(vec![0.0, 0.0]);

    enhancement
        .enhanced_optimize_step(&mut params, &grad, &losses, &grads)
        .expect("step");

    let big = params[0].abs();
    let small = params[1].abs();
    assert!(big > 0.0 && small > 0.0, "steps {big} / {small}");
    let ratio = big / small;
    assert!(
        ratio < 2.0,
        "step magnitudes still track the raw gradient (ratio {ratio}); \
         a per-parameter rate should normalize them"
    );

    // And the second-moment state that produces it must actually be populated.
    let moments = enhancement.gradient_second_moment();
    assert_eq!(moments.len(), 2);
    assert!(
        moments[0] > moments[1],
        "second moments do not track per-parameter gradient scale: {:?}",
        moments
    );
}

#[test]
fn a_parameter_gradient_length_mismatch_is_rejected() {
    let mut enhancement =
        AdaptiveTransformerEnhancement::<f64>::new(adaptive_config()).expect("construction");
    let (grads, losses) = descending_history();
    let grad = Array1::from_vec(vec![1.0, 1.0, 1.0]);
    let mut params = Array1::from_vec(vec![0.0, 0.0]);
    assert!(enhancement
        .enhanced_optimize_step(&mut params, &grad, &losses, &grads)
        .is_err());
}

// ---------------------------------------------------------------------------
// F22: `enhance_optimizer` acts on the optimizer it is given
// ---------------------------------------------------------------------------

#[test]
fn enhance_optimizer_actually_reconfigures_the_optimizer() {
    let mut enhancement =
        AdaptiveTransformerEnhancement::<f64>::new(adaptive_config()).expect("construction");
    let mut optimizer =
        TransformerOptimizer::<f64>::new(small_optimizer_config()).expect("optimizer");

    let before_layers = optimizer.config().num_transformer_layers;
    let before_dropout = optimizer.config().dropout_rate;

    let (grads, losses) = oscillating_history();
    let result = enhancement
        .enhance_optimizer(&mut optimizer, &grads, &losses)
        .expect("enhancement");

    assert!(
        !result.architecture_adaptation.changes.is_empty(),
        "a hard landscape should propose at least one architecture change"
    );

    let after_layers = optimizer.config().num_transformer_layers;
    let after_dropout = optimizer.config().dropout_rate;
    assert!(
        after_layers != before_layers || (after_dropout - before_dropout).abs() > 1e-9,
        "the optimizer was not modified: layers {before_layers}->{after_layers}, \
         dropout {before_dropout}->{after_dropout}"
    );
    // The optimizer must still work at the new shape.
    assert!(optimizer.parameter_count() > 0);
}

#[test]
fn architecture_proposal_depends_on_the_landscape() {
    let mut easy_enh =
        AdaptiveTransformerEnhancement::<f64>::new(adaptive_config()).expect("construction");
    let mut easy_opt =
        TransformerOptimizer::<f64>::new(small_optimizer_config()).expect("optimizer");
    let (easy_grads, easy_losses) = descending_history();
    let easy = easy_enh
        .enhance_optimizer(&mut easy_opt, &easy_grads, &easy_losses)
        .expect("easy");

    let mut hard_enh =
        AdaptiveTransformerEnhancement::<f64>::new(adaptive_config()).expect("construction");
    let mut hard_opt =
        TransformerOptimizer::<f64>::new(small_optimizer_config()).expect("optimizer");
    let (hard_grads, hard_losses) = oscillating_history();
    let hard = hard_enh
        .enhance_optimizer(&mut hard_opt, &hard_grads, &hard_losses)
        .expect("hard");

    // The old version always returned `[LayerCountChange(6)]`, improvement 0.1,
    // confidence 0.8 regardless of input.
    assert_ne!(
        easy_opt.config().dropout_rate,
        hard_opt.config().dropout_rate,
        "the proposed dropout does not depend on the landscape"
    );
    assert!(
        (easy.architecture_adaptation.confidence - 0.8).abs() > 1e-9
            || (hard.architecture_adaptation.confidence - 0.8).abs() > 1e-9,
        "adaptation confidence is still the hardcoded 0.8"
    );
}

// ---------------------------------------------------------------------------
// F20-F23: the performance predictor learns instead of fabricating
// ---------------------------------------------------------------------------

fn feature_row(complexity: f64, difficulty: f64) -> PredictionFeatures {
    PredictionFeatures {
        complexity,
        difficulty,
        landscape_confidence: 0.7,
        expected_improvement: 0.2 * (1.0 - complexity),
        adaptation_confidence: 0.6,
        change_count: 0.25,
        layer_count: 2.0 / 24.0,
        hidden_size: 8.0 / 2048.0,
        head_count: 2.0 / 32.0,
        dropout: 0.5 * difficulty,
        interaction: complexity * difficulty,
        complexity_sq: complexity * complexity,
        difficulty_sq: difficulty * difficulty,
        strategy_conservative: 0.5,
        strategy_aggressive: 0.5,
        strategy_exploratory: 0.0,
    }
}

#[test]
fn predictor_is_honest_before_training_and_learns_after() {
    let mut enhancement =
        AdaptiveTransformerEnhancement::<f64>::new(adaptive_config()).expect("construction");
    assert!(!enhancement.predictor_is_trained());

    let mut params = Array1::from_vec(vec![1.0, 1.0]);
    let grad = Array1::from_vec(vec![0.5, -0.5]);
    let (grads, losses) = oscillating_history();
    let untrained = enhancement
        .enhanced_optimize_step(&mut params, &grad, &losses, &grads)
        .expect("step");

    // Before training: no information, explicitly. Not 0.15 / 0.92 / 0.85 / 0.05.
    assert_eq!(
        untrained.performance_prediction.convergence_improvement,
        0.0
    );
    assert_eq!(untrained.performance_prediction.confidence, 0.0);
    assert_eq!(untrained.performance_prediction.uncertainty, 1.0);

    // Train on a synthetic but learnable relationship.
    let samples: Vec<PredictorSample> = (0..60)
        .map(|i| {
            let c = (i as f64 * 0.043).fract();
            let d = (i as f64 * 0.097).fract();
            let f = feature_row(c, d);
            PredictorSample {
                convergence_improvement: 0.30 - 0.25 * d,
                final_performance: 0.95 - 0.40 * d - 0.10 * c,
                features: f,
            }
        })
        .collect();
    let report = enhancement
        .train_performance_predictor(&samples)
        .expect("training");
    assert_eq!(report.samples, 60);
    assert!(
        report.convergence_rmse < 0.05,
        "training RMSE {} too large",
        report.convergence_rmse
    );
    assert!(enhancement.predictor_is_trained());

    // After training the prediction must depend on the landscape.
    let mut easy_params = Array1::from_vec(vec![1.0, 1.0]);
    let (easy_grads, easy_losses) = descending_history();
    let easy = enhancement
        .enhanced_optimize_step(&mut easy_params, &grad, &easy_losses, &easy_grads)
        .expect("easy step");
    let mut hard_params = Array1::from_vec(vec![1.0, 1.0]);
    let hard = enhancement
        .enhanced_optimize_step(&mut hard_params, &grad, &losses, &grads)
        .expect("hard step");

    assert!(
        (easy.performance_prediction.convergence_improvement
            - hard.performance_prediction.convergence_improvement)
            .abs()
            > 1e-6,
        "the trained predictor still returns a constant: {} vs {}",
        easy.performance_prediction.convergence_improvement,
        hard.performance_prediction.convergence_improvement
    );
    assert!(
        easy.performance_prediction.convergence_improvement
            > hard.performance_prediction.convergence_improvement,
        "an easier landscape should predict a larger improvement: {} vs {}",
        easy.performance_prediction.convergence_improvement,
        hard.performance_prediction.convergence_improvement
    );
    assert!(
        easy.performance_prediction.confidence > 0.0,
        "a trained predictor should report non-zero confidence"
    );
}

/// F22 follow-up: calling `enhance_optimizer` repeatedly on the *same* history
/// must converge, not ratchet. The first version proposed
/// `num_transformer_layers + 1` whenever the landscape was complex, and because
/// the adapter is seeded from the optimizer's live config each call, repeated
/// calls walked the layer count up to the search-space maximum — rebuilding (and
/// therefore re-initializing) the transformer every single time.
#[test]
fn repeated_enhancement_converges_instead_of_ratcheting() {
    let mut enhancement =
        AdaptiveTransformerEnhancement::<f64>::new(adaptive_config()).expect("construction");
    let mut optimizer =
        TransformerOptimizer::<f64>::new(small_optimizer_config()).expect("optimizer");
    let (grads, losses) = oscillating_history();

    let mut layer_counts = Vec::new();
    for _ in 0..24 {
        enhancement
            .enhance_optimizer(&mut optimizer, &grads, &losses)
            .expect("enhancement");
        layer_counts.push(optimizer.config().num_transformer_layers);
    }

    let settled = *layer_counts.last().expect("at least one call");
    assert!(
        settled <= 12,
        "layer count {settled} escaped the search-space maximum: {layer_counts:?}"
    );
    // `Gradual` approaches the landscape-derived target one layer at a time, so
    // it must converge in a bounded number of steps and then stop moving. Under
    // the old `current + 1` rule it never stopped until it hit the cap, and every
    // step rebuilt (and re-initialized) the transformer.
    let tail = &layer_counts[layer_counts.len() - 8..];
    assert!(
        tail.iter().all(|&n| n == settled),
        "the architecture never settled: {layer_counts:?}"
    );

    // Once settled, further enhancement must not discard weights.
    let before = optimizer.parameter_count();
    let probe = Array1::from_vec(vec![0.1_f64; 8]);
    let probe_matrix = scirs2_core::ndarray::Array2::from_shape_fn((3, 8), |(i, j)| {
        ((i + j) as f64 * 0.05).tanh()
    });
    let output_before = optimizer
        .forward_sequence(&probe_matrix)
        .expect("forward before");
    let _ = probe;

    enhancement
        .enhance_optimizer(&mut optimizer, &grads, &losses)
        .expect("settled enhancement");

    assert_eq!(
        optimizer.parameter_count(),
        before,
        "a settled enhancement changed the parameter count"
    );
    let output_after = optimizer
        .forward_sequence(&probe_matrix)
        .expect("forward after");
    assert_eq!(
        output_before, output_after,
        "a settled enhancement re-initialized the transformer, discarding \
         everything it had learned"
    );
}

/// `apply_architecture_config` must say *which* of the three things it did, so a
/// caller can never mistake a destructive rebuild for a weight-preserving tweak.
#[test]
fn apply_architecture_config_reports_what_it_did() {
    use optirs_learned::transformer_based_optimizer::ArchitectureUpdate;

    let base = small_optimizer_config();
    let mut optimizer = TransformerOptimizer::<f64>::new(base.clone()).expect("optimizer");

    assert_eq!(
        optimizer
            .apply_architecture_config(&base)
            .expect("no-op apply"),
        ArchitectureUpdate::Unchanged
    );

    let mut tweaked = base.clone();
    tweaked.learning_rate = base.learning_rate * 2.0;
    let update = optimizer
        .apply_architecture_config(&tweaked)
        .expect("in-place apply");
    assert_eq!(update, ArchitectureUpdate::InPlace);
    assert!(update.changed() && !update.discarded_weights());

    let mut deeper = tweaked.clone();
    deeper.num_transformer_layers += 1;
    let update = optimizer
        .apply_architecture_config(&deeper)
        .expect("structural apply");
    assert_eq!(update, ArchitectureUpdate::Rebuilt);
    assert!(update.changed() && update.discarded_weights());

    // An invalid proposal must leave the optimizer untouched.
    let mut invalid = deeper.clone();
    invalid.num_attention_heads = 3; // does not divide model_dimension 8
    let before = optimizer.config().num_attention_heads;
    assert!(optimizer.apply_architecture_config(&invalid).is_err());
    assert_eq!(optimizer.config().num_attention_heads, before);
}
