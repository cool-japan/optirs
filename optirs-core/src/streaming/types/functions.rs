//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::primitives::{FusionStrategy, LearningRateAdaptation, StreamingConfig};
use super::types_2::{StreamingDataPoint, StreamingOptimizer};
use super::types_3::{ConsensusAlgorithm, StreamFusionOptimizer};

/// Convert an `f64` literal to the generic float type `A`, falling back to
/// `fallback` when the numeric type cannot represent the literal.
///
/// This is defensive only: for the `f32`/`f64` types this crate targets the
/// conversion always succeeds. It exists so numeric-literal conversions
/// never panic, instead degrading gracefully to a caller-chosen safe value
/// (e.g. `A::one()` as an identity fallback for multiplicative factors and
/// divisors, so a failed conversion never produces a division by zero).
#[inline]
pub(super) fn to_a_or<A: Float>(value: f64, fallback: A) -> A {
    A::from(value).unwrap_or(fallback)
}

#[cfg(test)]
pub(super) mod streaming_optimizer_regression_tests {
    use super::*;
    use crate::optimizers::SGD;
    use scirs2_core::ndarray::{Array1, Ix1};

    fn make_config() -> StreamingConfig {
        StreamingConfig {
            buffer_size: 4,
            adaptive_learning_rate: false,
            async_updates: false,
            gradient_compression: false,
            predictive_streaming: false,
            stream_fusion: false,
            adaptive_resource_allocation: false,
            multi_stream_coordination: false,
            ..Default::default()
        }
    }

    // Ground-truth linear model: `target = 1.0*x0 - 2.0*x1`.
    fn make_point(x0: f64, x1: f64) -> StreamingDataPoint<f64> {
        StreamingDataPoint {
            features: Array1::from_vec(vec![x0, x1]),
            target: Some(x0 - 2.0 * x1),
            timestamp: Instant::now(),
            weight: 1.0,
            metadata: HashMap::new(),
        }
    }

    fn feed_one_batch(opt: &mut StreamingOptimizer<SGD<f64>, f64, Ix1>) -> Option<Array1<f64>> {
        let mut last = None;
        for &(x0, x1) in &[(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0)] {
            if let Some(update) = opt
                .process_sample(make_point(x0, x1))
                .expect("process_sample failed")
            {
                last = Some(update);
            }
        }
        last
    }

    /// F27: after the first buffer flush the live parameters must be a real
    /// vector of the feature dimensionality — not the empty `zeros(0)` that
    /// made every sync/async update a silent no-op.
    #[test]
    fn parameters_are_initialized_after_first_flush() {
        let mut opt =
            StreamingOptimizer::new(SGD::new(0.05_f64), make_config()).expect("construct");
        let update = feed_one_batch(&mut opt).expect("buffer should have flushed");
        assert_eq!(update.len(), 2, "updated params have wrong dimensionality");

        let params = opt.current_parameters().expect("params must exist (F27)");
        assert_eq!(
            params.len(),
            2,
            "live parameters were not initialized to the feature dim (F27 regression)"
        );
    }

    /// F28: with a real `prediction = dot(params, features)` residual, the
    /// batch loss must decrease as the optimizer repeatedly sees the same
    /// linear data. Under the previous hardcoded zero prediction the
    /// residual never reflected the model and loss could not improve.
    #[test]
    fn batch_loss_decreases_over_repeated_batches() {
        let mut opt =
            StreamingOptimizer::new(SGD::new(0.05_f64), make_config()).expect("construct");

        feed_one_batch(&mut opt).expect("first flush");
        let first_loss = opt.get_metrics().current_loss;

        for _ in 0..40 {
            feed_one_batch(&mut opt).expect("flush");
        }
        let final_loss = opt.get_metrics().current_loss;

        assert!(final_loss.is_finite(), "loss diverged to non-finite value");
        assert!(
            final_loss < first_loss,
            "streaming loss did not decrease (F28 regression): \
             first={first_loss}, final={final_loss}"
        );
    }

    /// T4: `should_force_update` must actually observe real elapsed time
    /// since the *first* sample of the current (unflushed) batch, not ~0
    /// because `batch_start` was re-stamped on every `process_sample` call.
    /// With `buffer_size` set larger than the number of samples fed and a
    /// tiny `latency_budget_ms`, a sleep between samples must be enough to
    /// force a flush well before the buffer fills.
    #[test]
    fn force_update_fires_after_latency_budget_elapses() {
        let config = StreamingConfig {
            buffer_size: 1000, // never reached by size alone
            latency_budget_ms: 20,
            adaptive_learning_rate: false,
            async_updates: false,
            gradient_compression: false,
            predictive_streaming: false,
            stream_fusion: false,
            adaptive_resource_allocation: false,
            multi_stream_coordination: false,
            ..Default::default()
        };
        let mut opt: StreamingOptimizer<SGD<f64>, f64, Ix1> =
            StreamingOptimizer::new(SGD::new(0.05_f64), config).expect("construct");

        // First sample starts the batch timer; well under half the latency
        // budget so it must not force-flush yet.
        let r1 = opt.process_sample(make_point(1.0, 0.0)).expect("sample 1");
        assert!(r1.is_none(), "flushed too early on first sample");

        // Sleep past latency_budget_ms/2 = 10ms, then feed a second sample:
        // should_force_update must now see real elapsed time and flush.
        std::thread::sleep(Duration::from_millis(15));
        let r2 = opt.process_sample(make_point(0.0, 1.0)).expect("sample 2");
        assert!(
            r2.is_some(),
            "T4 regression: should_force_update did not fire after the \
             latency budget elapsed (batch_start was likely re-stamped on \
             every process_sample call instead of only the first)"
        );
    }

    /// T5: `PerformanceBased` learning-rate adaptation reads
    /// `performance_history`, which must actually be populated by
    /// `process_buffer` from each batch's loss. Previously nothing ever
    /// pushed into it, so `adapt_performance_based` always saw `len() < 2`
    /// and silently no-op'd forever, leaving `current_lr` frozen at its
    /// initial value.
    #[test]
    fn performance_based_adaptation_changes_learning_rate() {
        let config = StreamingConfig {
            buffer_size: 4,
            adaptive_learning_rate: true,
            lr_adaptation: LearningRateAdaptation::PerformanceBased,
            async_updates: false,
            gradient_compression: false,
            predictive_streaming: false,
            stream_fusion: false,
            adaptive_resource_allocation: false,
            multi_stream_coordination: false,
            ..Default::default()
        };
        let mut opt: StreamingOptimizer<SGD<f64>, f64, Ix1> =
            StreamingOptimizer::new(SGD::new(0.05_f64), config).expect("construct");
        let initial_lr = opt.get_metrics().current_learning_rate;

        for _ in 0..10 {
            feed_one_batch(&mut opt).expect("flush");
        }

        let final_lr = opt.get_metrics().current_learning_rate;
        assert!(final_lr.is_finite(), "learning rate diverged");
        assert_ne!(
            final_lr, initial_lr,
            "T5 regression: PerformanceBased adaptation never changed the \
             learning rate (performance_history was never populated)"
        );
    }

    /// T7: the drift baseline must not be compared against the
    /// constructor's placeholder defaults (mean 0, std 1) the moment the
    /// loss window first fills — that comparison (real loss vs. mean-0/
    /// std-1) fired a spurious drift almost unconditionally whenever the
    /// batch loss itself exceeded the configured threshold (a near-certainty
    /// early in training, while parameters are still far from the target
    /// and losses are correspondingly large). This isolates exactly that
    /// transition (the first call where `loss_window.len()` reaches
    /// `drift_window_size`).
    ///
    /// The fixed feature/target values below (verified against a
    /// standalone trace of this exact SGD update rule before adding this
    /// test) are deliberately chosen so the batch loss is large enough
    /// (tens to hundreds) that the pre-fix z-score
    /// `|current_loss - 0| / (1 + 1e-8)` comfortably exceeds
    /// `drift_threshold = 3.0` — i.e. this test *does* fail against the
    /// pre-fix implementation, not just vacuously pass regardless of it.
    #[test]
    fn drift_baseline_does_not_fire_spuriously_on_bootstrap() {
        let config = StreamingConfig {
            buffer_size: 1, // one sample per batch -> fills the drift window fast
            drift_window_size: 10,
            drift_threshold: 3.0,
            adaptive_learning_rate: false,
            async_updates: false,
            gradient_compression: false,
            predictive_streaming: false,
            stream_fusion: false,
            adaptive_resource_allocation: false,
            multi_stream_coordination: false,
            ..Default::default()
        };
        // Small learning rate + a fixed, far-from-zero target (x0=5,
        // x1=-10 => target = x0 - 2*x1 = 25) keeps every one of the first
        // 10 batch losses large (over 50) without diverging, so the
        // pre-fix comparison against a `historical_std` of 1.0 is
        // guaranteed to exceed `drift_threshold = 3.0` every single call.
        let mut opt: StreamingOptimizer<SGD<f64>, f64, Ix1> =
            StreamingOptimizer::new(SGD::new(0.001_f64), config).expect("construct");

        // Feed exactly `drift_window_size` samples: this is the single call
        // where the pre-fix code compared a real (nonzero, >threshold-scale)
        // loss against the placeholder historical_mean=0/historical_std=1
        // and fired unconditionally.
        for _ in 0..10 {
            opt.process_sample(make_point(5.0, -10.0))
                .expect("process_sample");
        }

        assert_eq!(
            opt.get_metrics().drift_count,
            0,
            "T7 regression: drift falsely detected on baseline bootstrap \
             (compared real loss against placeholder mean=0/std=1 defaults)"
        );
    }

    /// Live-bug fix (found while implementing T7): with `DriftAware` LR
    /// adaptation active, `adapt_drift_aware` must apply its 1.5x boost only
    /// once per detected drift event. It is reachable from both
    /// `adapt_learning_rate` (every batch) and `check_concept_drift` (on a
    /// fresh drift), so an un-gated boost compounds `current_lr` by 1.5x on
    /// *every* call for as long as `last_drift` stays within the 60-second
    /// window — this calls `adapt_drift_aware` directly, many times in a
    /// row within that window, which is the actual failure mode: relying on
    /// `process_sample`'s drift-detection timing to organically trigger
    /// enough repeated calls is unreliable (each detected drift resets
    /// `last_drift`, which — depending on how *often* drift re-fires —
    /// could coincidentally reset the one-shot gate's window before enough
    /// compounding calls land; this test instead verifies the gate itself,
    /// deterministically, with no dependency on drift-detection cadence).
    #[test]
    fn drift_aware_boost_applies_once_per_drift_event_not_once_per_call() {
        let config = StreamingConfig {
            adaptive_learning_rate: true,
            lr_adaptation: LearningRateAdaptation::DriftAware,
            async_updates: false,
            gradient_compression: false,
            predictive_streaming: false,
            stream_fusion: false,
            adaptive_resource_allocation: false,
            multi_stream_coordination: false,
            ..Default::default()
        };
        let mut opt: StreamingOptimizer<SGD<f64>, f64, Ix1> =
            StreamingOptimizer::new(SGD::new(0.01_f64), config).expect("construct");

        let initial_lr = opt.lr_adaptation_state.current_lr;

        // Simulate a single fresh drift event (as `check_concept_drift`
        // does on detecting drift): set `last_drift` to now and clear the
        // one-shot gate.
        opt.drift_detector.last_drift = Some(Instant::now());
        opt.drift_detector.lr_boost_applied_for_current_drift = false;

        // Call `adapt_drift_aware` many times in a row, exactly as
        // `adapt_learning_rate` would on every subsequent batch processed
        // within the same 60-second post-drift window, with no new drift
        // detected in between (so `last_drift` never resets).
        for _ in 0..50 {
            opt.adapt_drift_aware().expect("adapt_drift_aware");
        }

        let final_lr = opt.lr_adaptation_state.current_lr;
        assert!(
            final_lr.is_finite(),
            "learning rate exploded to non-finite after repeated \
             adapt_drift_aware calls within one drift event: {final_lr}"
        );
        let expected_lr = initial_lr * 1.5; // exactly one boost application
        assert!(
            (final_lr - expected_lr).abs() < 1e-9,
            "regression: adapt_drift_aware applied its 1.5x boost more than \
             once for a single drift event (initial={initial_lr}, \
             expected one-shot result={expected_lr}, got={final_lr})"
        );
    }

    /// T9: `MedianFusion` must actually compute a coordinate-wise median
    /// across all contributing streams, not just echo back the first
    /// stream's step regardless of the others. A 3-stream input with one
    /// outlier stream must produce the *inlier* value in each coordinate,
    /// not the outlier and not `steps[0]` specifically.
    #[test]
    fn median_fusion_rejects_outlier_stream() {
        let config = make_config();
        let mut fusion = StreamFusionOptimizer::<f64>::new(&config).expect("construct fusion");
        fusion.fusion_strategy = FusionStrategy::MedianFusion;

        let steps = vec![
            ("outlier".to_string(), Array1::from_vec(vec![100.0, -100.0])),
            ("normal_a".to_string(), Array1::from_vec(vec![1.0, 2.0])),
            ("normal_b".to_string(), Array1::from_vec(vec![1.2, 1.8])),
        ];
        let fused = fusion
            .fuse_optimization_steps(&steps)
            .expect("fuse_optimization_steps");

        // Median of {100, 1.0, 1.2} = 1.2; median of {-100, 2.0, 1.8} = 1.8.
        assert!(
            (fused[0] - 1.2).abs() < 1e-9,
            "T9 regression: MedianFusion did not reject the outlier stream \
             (coordinate 0 = {}, expected 1.2)",
            fused[0]
        );
        assert!(
            (fused[1] - 1.8).abs() < 1e-9,
            "T9 regression: MedianFusion did not reject the outlier stream \
             (coordinate 1 = {}, expected 1.8)",
            fused[1]
        );
        // And it must not simply be `steps[0]` (the pre-fix behavior).
        assert_ne!(fused[0], 100.0);
        assert_ne!(fused[1], -100.0);
    }

    /// T9: Byzantine/PBFT consensus must use a trimmed-mean-style reducer
    /// that tolerates a minority of faulty (outlier) streams, rather than
    /// unconditionally returning `steps[0]`.
    #[test]
    fn byzantine_consensus_tolerates_minority_outliers() {
        let config = make_config();
        let mut fusion = StreamFusionOptimizer::<f64>::new(&config).expect("construct fusion");
        fusion.fusion_strategy = FusionStrategy::ConsensusBased;
        fusion.consensus_mechanism = ConsensusAlgorithm::Byzantine;

        // 4 honest streams near 1.0, one wild outlier: with f = (5-1)/3 = 1
        // trimmed from each end, the outlier must not skew the result.
        let steps = vec![
            ("s1".to_string(), Array1::from_vec(vec![1.0])),
            ("s2".to_string(), Array1::from_vec(vec![1.1])),
            ("s3".to_string(), Array1::from_vec(vec![0.9])),
            ("s4".to_string(), Array1::from_vec(vec![1.05])),
            ("outlier".to_string(), Array1::from_vec(vec![1000.0])),
        ];
        let fused = fusion
            .fuse_optimization_steps(&steps)
            .expect("fuse_optimization_steps");

        assert!(
            fused[0] < 2.0,
            "T9 regression: Byzantine consensus was skewed by a single \
             outlier stream (got {}, expected close to ~1.0)",
            fused[0]
        );
    }

    /// T9: mismatched step dimensionality across streams must produce an
    /// honest error, not an out-of-bounds panic (the coordinate-wise
    /// reducers index every stream's step at each coordinate up to the
    /// first stream's length).
    #[test]
    fn fusion_rejects_mismatched_dimensions() {
        let config = make_config();
        let mut fusion = StreamFusionOptimizer::<f64>::new(&config).expect("construct fusion");
        fusion.fusion_strategy = FusionStrategy::MedianFusion;

        let steps = vec![
            ("a".to_string(), Array1::from_vec(vec![1.0, 2.0])),
            ("b".to_string(), Array1::from_vec(vec![1.0])), // wrong length
        ];
        let result = fusion.fuse_optimization_steps(&steps);
        assert!(
            result.is_err(),
            "fusion with mismatched stream dimensionality must return Err, not panic or silently truncate"
        );
    }
}
