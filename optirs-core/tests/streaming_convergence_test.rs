//! Regression test for F27/F28 (streaming optimizer).
//!
//! Before the fix, `StreamingOptimizer::get_current_parameters` always
//! returned an empty vector (so async updates were a no-op and sync updates
//! never accumulated any state), and `compute_mini_batch_gradient` used a
//! hardcoded zero prediction (so the gradient never reflected the model's
//! actual error and the optimizer could not converge).
//!
//! This test fits streaming linear regression on synthetic data and checks
//! that the held-out loss drops substantially, which is only possible if
//! live parameters are tracked and genuinely updated across calls.

use std::collections::HashMap;
use std::time::Instant;

use scirs2_core::ndarray::Array1;

use optirs_core::optimizers::SGD;
use optirs_core::streaming::{StreamingConfig, StreamingDataPoint, StreamingOptimizer};

/// Minimal xorshift64 PRNG so the test has no external RNG dependency while
/// still producing statistically independent draws (unlike a simple affine
/// function of the step index, whose rows would sit near a lattice line).
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // xorshift64 requires a nonzero seed.
        Self(seed | 1)
    }

    /// Uniform draw in `[-0.5, 0.5)`.
    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    fn next_features(&mut self, dim: usize) -> Vec<f64> {
        (0..dim).map(|_| self.next_f64()).collect()
    }
}

/// True linear relationship used to generate synthetic streaming data:
/// target = 2*x0 - 1*x1 + 0.5*x2
fn true_target(features: &[f64]) -> f64 {
    2.0 * features[0] - features[1] + 0.5 * features[2]
}

fn mse(params: &Array1<f64>, test_set: &[(Vec<f64>, f64)]) -> f64 {
    let mut sum = 0.0;
    for (features, target) in test_set {
        let pred: f64 = features.iter().zip(params.iter()).map(|(f, w)| f * w).sum();
        let err = pred - target;
        sum += err * err;
    }
    sum / test_set.len() as f64
}

#[test]
fn streaming_linear_regression_converges() {
    let config = StreamingConfig {
        buffer_size: 8,
        async_updates: false,
        gradient_compression: false,
        // Fixed learning rate (taken from the base optimizer) keeps this test
        // deterministic and easy to reason about.
        adaptive_learning_rate: false,
        multi_stream_coordination: false,
        predictive_streaming: false,
        stream_fusion: false,
        adaptive_resource_allocation: false,
        ..Default::default()
    };

    let base_optimizer = SGD::new(0.05_f64);
    let mut streaming: StreamingOptimizer<SGD<f64>, f64, scirs2_core::ndarray::Ix1> =
        StreamingOptimizer::new(base_optimizer, config)
            .expect("failed to construct StreamingOptimizer");

    // Fixed held-out set for measuring generalization loss, drawn from an
    // independent RNG stream so it never overlaps the training stream.
    let mut test_rng = Xorshift64::new(0x00FE_EDFA_CEC0_FFEEu64);
    let test_set: Vec<(Vec<f64>, f64)> = (0..64)
        .map(|_| {
            let f = test_rng.next_features(3);
            let t = true_target(&f);
            (f, t)
        })
        .collect();

    let initial_loss = mse(&Array1::zeros(3), &test_set);
    assert!(initial_loss > 0.0, "test set should be non-trivial");

    let mut train_rng = Xorshift64::new(0x00C0_FFEE_1234_5678u64);
    let mut last_params: Option<Array1<f64>> = None;
    for _ in 0..3000 {
        let f = train_rng.next_features(3);
        let t = true_target(&f);
        let point = StreamingDataPoint {
            features: Array1::from_vec(f),
            target: Some(t),
            timestamp: Instant::now(),
            weight: 1.0,
            metadata: HashMap::new(),
        };
        if let Some(params) = streaming
            .process_sample(point)
            .expect("process_sample failed")
        {
            last_params = Some(params);
        }
    }
    if let Some(params) = streaming.flush().expect("flush failed") {
        last_params = Some(params);
    }

    let final_params = last_params.expect("streaming optimizer never produced an update");
    let final_loss = mse(&final_params, &test_set);

    assert!(
        final_loss < initial_loss * 0.1,
        "streaming linear regression failed to converge: \
         initial_loss={initial_loss}, final_loss={final_loss}, final_params={final_params:?}"
    );

    // The optimizer's own reported loss must also reflect real error, not
    // the old placeholder (which was never assigned and stayed at 0.0
    // regardless of how bad the model was).
    assert!(
        streaming.get_metrics().current_loss > 0.0,
        "current_loss was never populated with a real batch loss"
    );

    // Live parameters must be externally observable (F27), and must match
    // the last value returned from `process_sample`/`flush`.
    let observed = streaming
        .current_parameters()
        .expect("current_parameters() returned None after training");
    assert_eq!(observed, &final_params);

    // Sanity: step_count must reflect real processed batches (buffer_size=8
    // over 3000 samples => at least 300 sync updates were applied).
    assert!(
        streaming.step_count >= 300,
        "step_count={}",
        streaming.step_count
    );
}

/// Regression check for F27 in isolation: async updates must actually be
/// applied to the live parameters, not silently discarded.
#[test]
fn streaming_async_updates_change_parameters() {
    let config = StreamingConfig {
        buffer_size: 4,
        async_updates: true,
        max_staleness: 2,
        adaptive_learning_rate: false,
        gradient_compression: false,
        multi_stream_coordination: false,
        predictive_streaming: false,
        stream_fusion: false,
        adaptive_resource_allocation: false,
        ..Default::default()
    };

    let base_optimizer = SGD::new(0.1_f64);
    let mut streaming: StreamingOptimizer<SGD<f64>, f64, scirs2_core::ndarray::Ix1> =
        StreamingOptimizer::new(base_optimizer, config)
            .expect("failed to construct StreamingOptimizer");

    let mut train_rng = Xorshift64::new(0x00AB_CDEF_0123_4567u64);
    let mut saw_nonzero_update = false;
    for _ in 0..200 {
        let f = train_rng.next_features(3);
        let t = true_target(&f);
        let point = StreamingDataPoint {
            features: Array1::from_vec(f),
            target: Some(t),
            timestamp: Instant::now(),
            weight: 1.0,
            metadata: HashMap::new(),
        };
        if let Some(params) = streaming
            .process_sample(point)
            .expect("process_sample failed")
        {
            if params.iter().any(|&p| p != 0.0) {
                saw_nonzero_update = true;
            }
        }
    }

    assert!(
        saw_nonzero_update,
        "async updates never changed the live parameters away from zero"
    );
    assert!(
        streaming
            .current_parameters()
            .is_some_and(|p| p.iter().any(|&v| v != 0.0)),
        "current_parameters() does not reflect applied async updates"
    );
}
