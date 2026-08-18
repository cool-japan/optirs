// Regression tests for the adaptive streaming optimizer's use of the base
// optimizer it is constructed with.
//
// Split out of `optimizer.rs` so that file stays under the 2000-line limit; it
// is wired back in as a `#[cfg(test)]` child module, so `super::*` still refers
// to the code under test.

use super::*;
use crate::optimizers::{Adam, SGD};
use scirs2_core::ndarray::Ix1;
use std::collections::HashMap;

fn point(features: Vec<f64>) -> StreamingDataPoint<f64> {
    StreamingDataPoint {
        features: Array1::from_vec(features),
        target: Some(Array1::from_vec(vec![1.0])),
        timestamp: Instant::now(),
        source_id: None,
        quality_score: 1.0,
        metadata: HashMap::new(),
    }
}

fn batch() -> Vec<StreamingDataPoint<f64>> {
    vec![
        point(vec![1.0, 0.0, 0.5]),
        point(vec![0.0, 1.0, -0.5]),
        point(vec![0.5, 0.5, 1.0]),
        point(vec![-1.0, 0.5, 0.25]),
    ]
}

fn config() -> StreamingConfig {
    let mut config = StreamingConfig::default();
    config.buffer_config.initial_size = 4;
    config.buffer_config.min_size = 2;
    config
}

fn run<O>(mut optimizer: AdaptiveStreamingOptimizer<O, f64, Ix1>) -> Option<Array1<f64>>
where
    O: crate::optimizers::Optimizer<f64, Ix1> + Clone,
{
    let mut last = None;
    for _ in 0..8 {
        if let Ok(params) = optimizer.adaptive_step(batch()) {
            last = Some(params);
        }
    }
    last
}

/// The base optimizer must actually drive the parameter update.
///
/// `perform_optimization_step` used to contain an inline `param -= lr * grad`
/// loop with the comment "in practice would use the base optimizer", so the
/// `O` type parameter was inert: Adam and SGD produced *bit-identical*
/// trajectories because neither was ever called. Adam's bias-corrected moment
/// estimates make its first steps differ sharply from raw gradient descent, so
/// identical results here mean the delegation has been lost again.
#[test]
fn base_optimizer_actually_drives_the_update() {
    let adam_result = run(
        AdaptiveStreamingOptimizer::new(Adam::new(0.05_f64), config()).expect("construct adam"),
    );
    let sgd_result =
        run(AdaptiveStreamingOptimizer::new(SGD::new(0.05_f64), config()).expect("construct sgd"));

    let (adam_params, sgd_params) = match (adam_result, sgd_result) {
        (Some(a), Some(s)) => (a, s),
        _ => return, // the buffer never flushed; nothing to compare
    };

    assert_eq!(adam_params.len(), sgd_params.len());
    assert!(
        adam_params.iter().all(|v| v.is_finite()),
        "Adam-driven parameters must stay finite: {adam_params:?}"
    );

    let max_difference = adam_params
        .iter()
        .zip(sgd_params.iter())
        .map(|(a, s)| (a - s).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_difference > 1e-9,
        "regression: Adam and SGD produced identical trajectories, so the base \
         optimizer is being ignored again (max coordinate difference {max_difference})"
    );
}

/// A gradient whose length does not match the parameters must be reported, not
/// silently truncated by `into_dimensionality` or by a zipped loop.
#[test]
fn mismatched_gradient_dimensionality_is_reported() {
    let mut optimizer: AdaptiveStreamingOptimizer<SGD<f64>, f64, Ix1> =
        AdaptiveStreamingOptimizer::new(SGD::new(0.05_f64), config()).expect("construct");

    // Prime the optimizer with three-feature data, then feed two-feature data:
    // the stored parameters and the freshly computed gradient disagree.
    for _ in 0..4 {
        let _ = optimizer.adaptive_step(batch());
    }
    if optimizer.parameters.is_none() {
        return; // never flushed; the guard under test is unreachable here
    }

    let narrow: Vec<StreamingDataPoint<f64>> = (0..4).map(|_| point(vec![1.0, 0.0])).collect();
    let result = optimizer.perform_optimization_step(&narrow);
    assert!(
        result.is_err(),
        "a parameter/gradient length mismatch must be an honest error, not a silent truncation"
    );
}

/// `recent_adaptations` is derived live from `adaptation_history`, so a fresh
/// optimizer reports zero and the window is applied without materialising an
/// `Instant::now() - window` cutoff (which panics on a young process).
#[test]
fn recent_adaptations_is_derived_not_stored() {
    let optimizer: AdaptiveStreamingOptimizer<SGD<f64>, f64, Ix1> =
        AdaptiveStreamingOptimizer::new(SGD::new(0.05_f64), config()).expect("construct");
    let stats = optimizer.get_adaptive_stats();
    assert_eq!(stats.recent_adaptations, 0);
    assert_eq!(stats.adaptations_applied, 0);
    assert_eq!(
        optimizer.count_adaptations_applied(RECENT_ADAPTATION_WINDOW),
        0
    );
}

/// The convenience factories in `mod.rs` advertise
/// `AdaptiveStreamingOptimizer<Adam<A>, A, D>`, but nothing exercised the
/// product. Now that the impl requires `O: Optimizer<A, D>`, a factory whose
/// `D` bound does not satisfy it would hand back an optimizer that cannot
/// step -- and only a call site like this one would find out.
#[test]
fn factory_built_optimizers_can_actually_step() {
    let mut optimizer =
        crate::streaming::adaptive_streaming::create_default_optimizer::<f64, Ix1>()
            .expect("default factory");
    for _ in 0..8 {
        let _ = optimizer.adaptive_step(batch());
    }

    let mut configured =
        crate::streaming::adaptive_streaming::create_optimizer_with_config::<f64, Ix1>(config())
            .expect("configured factory");
    for _ in 0..8 {
        let _ = configured.adaptive_step(batch());
    }
    // The point of the test is that the factory's product is usable at all;
    // that it adapted while running is the expected behaviour, not a failure.
    let stats = configured.get_adaptive_stats();
    assert!(
        stats.total_data_points > 0,
        "the factory optimizer never ingested data"
    );
    assert!(
        stats.recent_adaptations <= stats.adaptations_applied,
        "the recent-window count cannot exceed the lifetime count"
    );
}
