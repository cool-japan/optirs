//! Regression tests for the LSTM optimizer's meta-training (findings F6/F7).
//!
//! What these replace:
//!
//! * `MetaLearner::step` returned `Ok(T::zero())` and trained nothing. The LSTM
//!   cell math was correct but **no code anywhere in the crate ever updated the
//!   controller's weights** — there was no backprop-through-time loop.
//! * `AdaptiveLearningRateController::compute_lr` returned its stored rate and
//!   read none of its arguments.
//! * `OptimizationStateTracker::update` had an empty body, so
//!   `get_state_analysis()` reported the all-zero struct forever.
//! * `TransferLearner::transfer_to_domain` returned an all-zero result struct.
//!
//! The load-bearing test here is
//! [`bptt_gradient_matches_finite_differences`]: it checks the analytic
//! meta-gradient of every single controller parameter against a central finite
//! difference of the very function it claims to differentiate. A hand-written
//! backward pass that is wrong in any term fails it.

use optirs_learned::lstm::{
    bptt, BpttConfig, DiagonalQuadraticTask, LSTMNetwork, LSTMOptimizer, MetaTask, MetaTrainer,
    MetaTrainingTask, TaskCharacteristics, TaskType, TrajectoryPoint,
};
use optirs_learned::LearnedOptimizerConfig;
use scirs2_core::ndarray::Array1;

/// Small controller so the finite-difference sweep stays cheap but still covers
/// every kind of weight (LSTM gates, layer norm, attention, output projection).
fn controller_config(use_attention: bool) -> LearnedOptimizerConfig {
    LearnedOptimizerConfig {
        hidden_size: 6,
        attention_heads: 2,
        num_layers: 2,
        input_features: 8,
        output_features: 2,
        use_attention,
        dropout_rate: 0.0,
        learning_rate: 1e-2,
        meta_learning_rate: 5e-2,
        ..Default::default()
    }
}

fn quadratic(curvature: &[f64], optimum: &[f64], start: &[f64]) -> DiagonalQuadraticTask<f64> {
    DiagonalQuadraticTask::new(
        Array1::from_vec(curvature.to_vec()),
        Array1::from_vec(optimum.to_vec()),
        Array1::from_vec(start.to_vec()),
    )
    .expect("quadratic task")
}

/// A family of 2-D quadratics with varying curvature, conditioning and optimum.
fn task_family(count: usize) -> Vec<DiagonalQuadraticTask<f64>> {
    (0..count)
        .map(|i| {
            let t = i as f64;
            let a0 = 0.5 + 0.35 * (t * 0.7).sin().abs();
            let a1 = 0.5 + 0.35 * (t * 1.3).cos().abs();
            let o0 = 0.8 * (t * 0.4).sin();
            let o1 = 0.8 * (t * 0.9).cos();
            quadratic(&[a0, a1], &[o0, o1], &[1.5, -1.2])
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The correctness check: analytic BPTT == finite differences
// ---------------------------------------------------------------------------

fn assert_gradient_matches_fd(use_attention: bool) {
    let config = controller_config(use_attention);
    let mut network = LSTMNetwork::<f64>::new(&config).expect("controller");
    let task = quadratic(&[0.9, 0.4], &[0.3, -0.6], &[1.1, -0.7]);

    let trainer = MetaTrainer::new(BpttConfig {
        // A short horizon keeps the sweep fast while still exercising the
        // recurrent carry (>1 step is what makes it BPTT rather than backprop).
        unroll_steps: 5,
        meta_learning_rate: 0.0,
        gradient_clip: 0.0,
    });

    // Record the rollout so the loss becomes a pure function of the controller
    // weights — see the `bptt` module docs on the one dropped term.
    let rollout = trainer
        .capture_rollout(&mut network, &task, config.input_features, 1.0)
        .expect("captured rollout");

    let (analytic_loss, gradients) =
        bptt::meta_gradient_frozen(&mut network, &task, &rollout).expect("analytic gradient");
    let analytic = gradients.to_vec();

    let mut params = bptt::flatten_parameters(&network);
    assert_eq!(
        analytic.len(),
        params.len(),
        "gradient layout must match the parameter layout"
    );
    assert!(
        params.len() > 100,
        "the test controller should have a non-trivial parameter count, got {}",
        params.len()
    );

    // Sanity: the recorded loss must be reproducible (dropout off, state reset).
    let repeated =
        bptt::rollout_loss_frozen(&mut network, &task, &rollout).expect("repeat forward");
    assert_eq!(
        analytic_loss, repeated,
        "the frozen rollout is not deterministic; the finite-difference check \
         below would be meaningless"
    );

    let h = 1e-6;
    let mut checked = 0usize;
    let mut worst = 0.0_f64;
    let mut worst_index = 0usize;
    for k in 0..params.len() {
        let original = params[k];

        params[k] = original + h;
        bptt::set_parameters(&mut network, &params).expect("perturb +");
        let plus = bptt::rollout_loss_frozen(&mut network, &task, &rollout).expect("loss +");

        params[k] = original - h;
        bptt::set_parameters(&mut network, &params).expect("perturb -");
        let minus = bptt::rollout_loss_frozen(&mut network, &task, &rollout).expect("loss -");

        params[k] = original;
        bptt::set_parameters(&mut network, &params).expect("restore");

        let numeric = (plus - minus) / (2.0 * h);
        let scale = numeric.abs().max(analytic[k].abs()).max(1e-4);
        let relative = (analytic[k] - numeric).abs() / scale;
        if relative > worst {
            worst = relative;
            worst_index = k;
        }
        checked += 1;
        assert!(
            relative < 2e-4,
            "parameter {k}: analytic {} vs finite-difference {numeric} \
             (relative error {relative})",
            analytic[k]
        );
    }

    assert_eq!(checked, params.len(), "every parameter must be checked");
    assert!(
        worst < 2e-4,
        "worst relative error {worst} at parameter {worst_index}"
    );
    // A gradient of exactly zero everywhere would pass an equality check but
    // would mean nothing is being learned.
    let magnitude = analytic.iter().fold(0.0_f64, |acc, &g| acc.max(g.abs()));
    assert!(
        magnitude > 1e-8,
        "the meta-gradient is identically zero (max |g| {magnitude})"
    );
}

#[test]
fn bptt_gradient_matches_finite_differences() {
    assert_gradient_matches_fd(false);
}

#[test]
fn bptt_gradient_matches_finite_differences_with_attention() {
    assert_gradient_matches_fd(true);
}

// ---------------------------------------------------------------------------
// Meta-training actually learns
// ---------------------------------------------------------------------------

#[test]
fn meta_training_reduces_the_meta_loss() {
    let config = controller_config(false);
    let mut network = LSTMNetwork::<f64>::new(&config).expect("controller");
    let tasks = task_family(6);
    let weights = vec![1.0_f64; tasks.len()];

    let mut trainer = MetaTrainer::new(BpttConfig {
        unroll_steps: 12,
        meta_learning_rate: 3e-2,
        gradient_clip: 1.0,
    });

    let first = trainer
        .meta_step(&mut network, &tasks, &weights)
        .expect("first meta step");
    let mut last = first;
    for _ in 0..60 {
        last = trainer
            .meta_step(&mut network, &tasks, &weights)
            .expect("meta step");
    }

    assert!(
        last < first,
        "meta-training did not reduce the meta-loss: {first} -> {last}"
    );
    assert!(
        last < 0.75 * first,
        "meta-loss barely moved: {first} -> {last} (needed at least a 25% drop)"
    );
    assert_eq!(trainer.loss_history().len(), 61);
    assert!(
        trainer
            .last_gradient_vector()
            .iter()
            .any(|g| g.abs() > 1e-10),
        "the applied meta-gradient was all zeros"
    );
}

/// The headline behavioural requirement: after meta-training, the learned
/// optimizer must reduce a **held-out** quadratic's loss faster than the same
/// controller did before training.
///
/// The comparison is *paired* — the untrained baseline is a clone of the very
/// network that later gets trained — so the result cannot be an accident of one
/// lucky random initialization.
#[test]
fn trained_controller_beats_its_untrained_self_on_a_held_out_quadratic() {
    let config = controller_config(false);
    let mut trained = LSTMNetwork::<f64>::new(&config).expect("controller");
    let mut untrained = trained.clone();

    let train_tasks = task_family(6);
    let weights = vec![1.0_f64; train_tasks.len()];
    let mut trainer = MetaTrainer::new(BpttConfig {
        unroll_steps: 12,
        meta_learning_rate: 3e-2,
        gradient_clip: 1.0,
    });
    for _ in 0..60 {
        trainer
            .meta_step(&mut trained, &train_tasks, &weights)
            .expect("meta step");
    }

    // Held out: curvature, optimum and start all outside the training family.
    let held_out = quadratic(&[1.35, 0.22], &[-0.45, 0.95], &[2.0, -1.7]);
    let eval = MetaTrainer::new(BpttConfig {
        unroll_steps: 12,
        meta_learning_rate: 0.0,
        gradient_clip: 0.0,
    });

    let final_loss = |network: &mut LSTMNetwork<f64>| -> f64 {
        let rollout = eval
            .capture_rollout(network, &held_out, config.input_features, 1.0)
            .expect("rollout");
        let params = bptt::rollout_final_parameters(network, &held_out, &rollout).expect("params");
        held_out.loss(&params)
    };

    let start_loss = held_out.loss(&held_out.initial_parameters());
    let untrained_loss = final_loss(&mut untrained);
    let trained_loss = final_loss(&mut trained);

    assert!(
        trained_loss < untrained_loss,
        "meta-training did not help on the held-out task: \
         trained {trained_loss} vs untrained {untrained_loss} (start {start_loss})"
    );
    assert!(
        trained_loss < start_loss,
        "the trained controller failed to make progress at all: \
         {start_loss} -> {trained_loss}"
    );
}

#[test]
fn meta_step_rejects_degenerate_batches() {
    let config = controller_config(false);
    let mut network = LSTMNetwork::<f64>::new(&config).expect("controller");
    let mut trainer = MetaTrainer::new(BpttConfig::default());

    let empty: Vec<DiagonalQuadraticTask<f64>> = Vec::new();
    assert!(trainer.meta_step(&mut network, &empty, &[]).is_err());

    let tasks = task_family(2);
    // Fewer weights than tasks.
    assert!(trainer.meta_step(&mut network, &tasks, &[1.0]).is_err());

    // Mixed dimensions.
    let mixed = vec![
        quadratic(&[1.0, 1.0], &[0.0, 0.0], &[1.0, 1.0]),
        quadratic(&[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]),
    ];
    assert!(trainer
        .meta_step(&mut network, &mixed, &[1.0, 1.0])
        .is_err());

    // Zero unroll horizon.
    let mut zero = MetaTrainer::new(BpttConfig {
        unroll_steps: 0,
        meta_learning_rate: 1e-2,
        gradient_clip: 1.0,
    });
    assert!(zero.meta_step(&mut network, &tasks, &[1.0, 1.0]).is_err());
}

// ---------------------------------------------------------------------------
// Trajectory -> differentiable surrogate identification
// ---------------------------------------------------------------------------

fn trajectory_point(step: usize, params: &[f64], grad: &[f64], loss: f64) -> TrajectoryPoint<f64> {
    TrajectoryPoint {
        step,
        gradient: Array1::from_vec(grad.to_vec()),
        parameters: Array1::from_vec(params.to_vec()),
        loss,
        learning_rate: 1e-2,
        update: Array1::from_vec(vec![0.0; params.len()]),
    }
}

#[test]
fn surrogate_identification_recovers_a_known_quadratic() {
    // Ground truth: f(θ) = ½[3(θ₀−1)² + 0.5(θ₁+2)²]
    let curvature = [3.0_f64, 0.5];
    let optimum = [1.0_f64, -2.0];
    let points: Vec<TrajectoryPoint<f64>> = (0..6)
        .map(|i| {
            let p = [1.7 - 0.2 * i as f64, -1.1 - 0.15 * i as f64];
            let g = [
                curvature[0] * (p[0] - optimum[0]),
                curvature[1] * (p[1] - optimum[1]),
            ];
            let loss = 0.5
                * (curvature[0] * (p[0] - optimum[0]).powi(2)
                    + curvature[1] * (p[1] - optimum[1]).powi(2));
            trajectory_point(i, &p, &g, loss)
        })
        .collect();

    let surrogate = DiagonalQuadraticTask::from_trajectory(&points).expect("surrogate");
    for j in 0..2 {
        assert!(
            (surrogate.curvature()[j] - curvature[j]).abs() < 1e-8,
            "curvature {j}: {} vs {}",
            surrogate.curvature()[j],
            curvature[j]
        );
        assert!(
            (surrogate.optimum()[j] - optimum[j]).abs() < 1e-8,
            "optimum {j}: {} vs {}",
            surrogate.optimum()[j],
            optimum[j]
        );
    }

    // And the identified surrogate must reproduce the observed gradients.
    let probe = Array1::from_vec(vec![0.4, 0.9]);
    let g = surrogate.gradient(&probe);
    assert!((g[0] - curvature[0] * (0.4 - optimum[0])).abs() < 1e-8);
    assert!((g[1] - curvature[1] * (0.9 - optimum[1])).abs() < 1e-8);
}

#[test]
fn surrogate_identification_rejects_unusable_trajectories() {
    assert!(DiagonalQuadraticTask::<f64>::from_trajectory(&[]).is_err());
    let single = vec![trajectory_point(0, &[1.0], &[0.5], 1.0)];
    assert!(DiagonalQuadraticTask::from_trajectory(&single).is_err());
    let ragged = vec![
        trajectory_point(0, &[1.0, 1.0], &[0.5, 0.5], 1.0),
        trajectory_point(1, &[1.0], &[0.5], 1.0),
    ];
    assert!(DiagonalQuadraticTask::from_trajectory(&ragged).is_err());
}

// ---------------------------------------------------------------------------
// The public `LSTMOptimizer` surface
// ---------------------------------------------------------------------------

fn meta_task(id: &str, curvature: [f64; 2], optimum: [f64; 2]) -> MetaTask<f64> {
    let trajectory: Vec<TrajectoryPoint<f64>> = (0..6)
        .map(|i| {
            let p = [1.6 - 0.25 * i as f64, -1.4 + 0.3 * i as f64];
            let g = [
                curvature[0] * (p[0] - optimum[0]),
                curvature[1] * (p[1] - optimum[1]),
            ];
            let loss = 0.5
                * (curvature[0] * (p[0] - optimum[0]).powi(2)
                    + curvature[1] * (p[1] - optimum[1]).powi(2));
            trajectory_point(i, &p, &g, loss)
        })
        .collect();
    MetaTask {
        id: id.to_string(),
        task_type: TaskType::SupervisedLearning,
        training_trajectory: trajectory,
        final_performance: 0.1,
        characteristics: TaskCharacteristics {
            dimensionality: 2,
            curvature: curvature[0],
            noise_level: 0.0,
            conditioning: curvature[0] / curvature[1],
            difficulty: 0.5,
            domain_features: Array1::from_vec(vec![0.1, 0.2]),
        },
        weight: 1.0,
    }
}

#[test]
fn meta_learning_step_is_no_longer_a_no_op() {
    let config = controller_config(false);
    let mut optimizer = LSTMOptimizer::<f64>::new(config).expect("optimizer");

    let before = bptt::flatten_parameters(optimizer.network());
    let tasks = vec![
        meta_task("a", [1.2, 0.4], [0.5, -0.5]),
        meta_task("b", [0.7, 0.9], [-0.3, 0.8]),
    ];

    let loss = optimizer.meta_learning_step(&tasks).expect("meta step");
    assert!(
        loss > 0.0,
        "a real meta-loss over a non-degenerate quadratic must be positive, got {loss}"
    );
    assert_eq!(
        optimizer.get_metrics().meta_learning_loss,
        loss,
        "the reported metric must be the measured meta-loss"
    );

    let after = bptt::flatten_parameters(optimizer.network());
    let moved = before
        .iter()
        .zip(after.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        moved > 1e-10,
        "the meta-learning step left every controller weight untouched \
         (max delta {moved})"
    );
}

#[test]
fn meta_learning_step_errors_on_unusable_tasks() {
    let config = controller_config(false);
    let mut optimizer = LSTMOptimizer::<f64>::new(config).expect("optimizer");
    assert!(optimizer.meta_learning_step(&[]).is_err());

    // A task with a one-point trajectory cannot yield a surrogate.
    let mut degenerate = meta_task("d", [1.0, 1.0], [0.0, 0.0]);
    degenerate.training_trajectory.truncate(1);
    assert!(optimizer.meta_learning_step(&[degenerate]).is_err());
}

#[test]
fn transfer_to_domain_reports_measured_numbers() {
    let config = controller_config(false);
    let mut optimizer = LSTMOptimizer::<f64>::new(config).expect("optimizer");
    let targets = vec![
        meta_task("t0", [1.1, 0.6], [0.4, -0.4]),
        meta_task("t1", [0.8, 1.0], [-0.2, 0.6]),
    ];

    let result = optimizer.transfer_to_domain(&targets).expect("transfer");

    // The old implementation returned all zeros.
    assert!(
        result.initial_performance > 0.0,
        "initial performance must be a measured loss, got {}",
        result.initial_performance
    );
    assert!(
        result.final_performance > 0.0,
        "final performance must be a measured loss, got {}",
        result.final_performance
    );
    assert!(result.adaptation_steps > 0, "no adaptation steps were run");
    assert!(
        result.transfer_efficiency.is_finite(),
        "transfer efficiency must be finite"
    );
    // Efficiency must be exactly the documented relative reduction.
    let expected =
        (result.initial_performance - result.final_performance) / result.initial_performance;
    assert!(
        (result.transfer_efficiency - expected.clamp(-1.0, 1.0)).abs() < 1e-12,
        "efficiency {} does not match the documented formula {expected}",
        result.transfer_efficiency
    );

    assert!(optimizer.transfer_to_domain(&[]).is_err());
}

// ---------------------------------------------------------------------------
// The adaptive learning-rate controller and the state tracker
// ---------------------------------------------------------------------------

#[test]
fn adaptive_learning_rate_responds_to_the_gradient_norm() {
    let config = controller_config(false);
    let base = config.learning_rate;

    let rate_after_gradient = |scale: f64| -> f64 {
        let mut optimizer = LSTMOptimizer::<f64>::new(controller_config(false)).expect("optimizer");
        let params = Array1::from_vec(vec![1.0, 1.0]);
        let grads = Array1::from_vec(vec![scale, scale]);
        optimizer
            .lstm_step(&params, &grads, Some(1.0))
            .expect("step");
        optimizer.current_learning_rate()
    };

    let tiny = rate_after_gradient(1e-3);
    let huge = rate_after_gradient(1e3);

    assert!(
        huge < tiny,
        "a huge gradient must shrink the learning rate: {huge} !< {tiny}"
    );
    // The old implementation returned the stored base rate no matter what.
    assert!(
        (huge - base).abs() > 1e-9,
        "the learning rate is still the constant base rate {base}"
    );
}

#[test]
fn state_tracker_records_real_statistics() {
    let mut optimizer = LSTMOptimizer::<f64>::new(controller_config(false)).expect("optimizer");
    let mut params = Array1::from_vec(vec![1.0, 1.0]);

    for i in 0..6 {
        let g = 1.0 / (1.0 + i as f64);
        let grads = Array1::from_vec(vec![g, -g]);
        params = optimizer
            .lstm_step(&params, &grads, Some(1.0 / (1.0 + i as f64)))
            .expect("step");
    }

    let analysis = optimizer.get_state_analysis();
    assert!(
        !analysis
            .convergence_indicators
            .gradient_norm_trend
            .is_empty(),
        "the gradient-norm trend is still empty; update() is a no-op"
    );
    assert!(
        analysis.gradient_analysis.gradient_stats.mean_norm > 0.0,
        "mean gradient norm is still zero: {}",
        analysis.gradient_analysis.gradient_stats.mean_norm
    );
    assert!(
        analysis.convergence_indicators.convergence_probability > 0.0,
        "convergence probability is still zero"
    );
    assert!(
        analysis
            .gradient_analysis
            .gradient_stats
            .direction_consistency
            .abs()
            > 0.0,
        "direction consistency was never computed"
    );
    assert!(
        analysis.stability_metrics.robustness_score > 0.0,
        "stability metrics were never computed"
    );
    assert!(
        !analysis.convergence_indicators.loss_change_trend.is_empty(),
        "the loss-change trend is still empty"
    );
}

#[test]
fn dropout_is_off_by_default_so_rollouts_are_reproducible() {
    let config = controller_config(false);
    let mut network = LSTMNetwork::<f64>::new(&config).expect("controller");
    assert!(
        !network.is_training(),
        "the controller must default to evaluation mode"
    );

    let task = quadratic(&[0.8, 0.5], &[0.2, -0.3], &[1.0, -1.0]);
    let trainer = MetaTrainer::new(BpttConfig {
        unroll_steps: 6,
        meta_learning_rate: 0.0,
        gradient_clip: 0.0,
    });
    let rollout = trainer
        .capture_rollout(&mut network, &task, config.input_features, 1.0)
        .expect("rollout");

    let a = bptt::rollout_loss_frozen(&mut network, &task, &rollout).expect("loss a");
    let b = bptt::rollout_loss_frozen(&mut network, &task, &rollout).expect("loss b");
    let c = bptt::rollout_loss_frozen(&mut network, &task, &rollout).expect("loss c");
    assert_eq!(a, b);
    assert_eq!(b, c);

    network.set_training(true);
    assert!(network.is_training());
}

/// The taped forward inside the BPTT engine re-implements the LSTM stack so it
/// can record intermediates. That makes it possible for the two to drift apart —
/// and if they do, meta-training optimizes a function the deployed optimizer
/// never evaluates, while `bptt_gradient_matches_finite_differences` keeps
/// passing (it is self-consistent by construction and would not notice).
///
/// This pins them together: for the same weights, the same reset state and the
/// same input, the taped forward's `y_t` must equal `LSTMNetwork::forward`'s
/// output to floating-point noise. (Exact equality is not required only because
/// the two associate the four gate terms differently.)
#[test]
fn taped_forward_matches_the_deployed_forward() {
    for use_attention in [false, true] {
        let config = controller_config(use_attention);
        let mut network = LSTMNetwork::<f64>::new(&config).expect("controller");
        let task = quadratic(&[0.7, 0.9], &[0.1, -0.2], &[1.3, -0.9]);

        let trainer = MetaTrainer::new(BpttConfig {
            unroll_steps: 4,
            meta_learning_rate: 0.0,
            gradient_clip: 0.0,
        });
        let rollout = trainer
            .capture_rollout(&mut network, &task, config.input_features, 1.0)
            .expect("rollout");

        let taped =
            bptt::taped_projection_outputs(&mut network, &task, &rollout).expect("taped outputs");
        assert_eq!(taped.len(), rollout.inputs.len());

        // Replay the same input sequence through the deployed forward.
        network.reset_state();
        let mut deployed = Vec::with_capacity(rollout.inputs.len());
        for features in &rollout.inputs {
            deployed.push(network.forward(features).expect("deployed forward"));
        }

        for (t, (a, b)) in taped.iter().zip(deployed.iter()).enumerate() {
            assert_eq!(a.len(), b.len(), "step {t} width mismatch");
            for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
                let scale = x.abs().max(y.abs()).max(1.0);
                assert!(
                    (x - y).abs() / scale < 1e-12,
                    "attention={use_attention} step {t} component {i}: \
                     taped {x} vs deployed {y}"
                );
            }
        }
    }
}

/// The BPTT horizon `MetaLearner::step` uses must be derived from the data.
/// It previously read `adaptation_history.len().clamp(8, 32).max(8)` on a
/// `VecDeque` that had no writers anywhere, so it was always exactly 8 — a knob
/// that configured nothing dressed up as adaptive. It must now track the observed
/// trajectory length, and `adaptation_history` must actually be populated.
#[test]
fn the_unroll_horizon_is_derived_from_the_observed_trajectories() {
    let horizon_for = |trajectory_len: usize| -> usize {
        let mut optimizer = LSTMOptimizer::<f64>::new(controller_config(false)).expect("optimizer");
        let mut task = meta_task("t", [1.1, 0.6], [0.4, -0.4]);
        // Rebuild the trajectory at the requested length.
        task.training_trajectory = (0..trajectory_len)
            .map(|i| {
                let p = [1.6 - 0.05 * i as f64, -1.4 + 0.06 * i as f64];
                let g = [1.1 * (p[0] - 0.4), 0.6 * (p[1] + 0.4)];
                trajectory_point(i, &p, &g, 1.0 / (1.0 + i as f64))
            })
            .collect();
        optimizer
            .meta_learning_step(std::slice::from_ref(&task))
            .expect("meta step");
        let history = optimizer.meta_learner().adaptation_history();
        assert_eq!(history.len(), 1, "no adaptation event was recorded");
        history.back().expect("one event").adaptation_steps
    };

    let short = horizon_for(6);
    let long = horizon_for(20);
    assert!(
        long > short,
        "the horizon does not track trajectory length: {short} vs {long}"
    );
    assert!(
        (4..=32).contains(&short),
        "short horizon {short} out of range"
    );
    assert!((4..=32).contains(&long), "long horizon {long} out of range");
    assert_ne!(
        short, 8,
        "the horizon is still the hardcoded 8 regardless of input"
    );
}

/// A coordinate with no usable convex signal must not manufacture an absurd task.
///
/// The first version clamped a non-positive least-squares slope up to
/// `min_curvature = 1e-6` and kept the intercept relation
/// `θ* = mean_x − mean_y/a`, which places the optimum up to a **million** units
/// from anything the trajectory visited — an enormous fake loss that then gets
/// averaged into the meta-training batch.
#[test]
fn a_degenerate_coordinate_does_not_manufacture_an_absurd_optimum() {
    // Coordinate 0 is a clean quadratic; coordinate 1 has a *negative* slope
    // (gradient falls as the parameter rises), which no convex quadratic produces.
    let points: Vec<TrajectoryPoint<f64>> = (0..6)
        .map(|i| {
            let p = [1.5 - 0.2 * i as f64, 0.5 + 0.2 * i as f64];
            let g = [2.0 * (p[0] - 0.5), -3.0 * p[1] + 1.0];
            trajectory_point(i, &p, &g, 1.0)
        })
        .collect();

    let surrogate = DiagonalQuadraticTask::from_trajectory(&points).expect("surrogate");

    // Both curvatures must be usable, not 1e-6.
    for j in 0..2 {
        assert!(
            surrogate.curvature()[j] >= 1e-3,
            "curvature {j} collapsed to {}",
            surrogate.curvature()[j]
        );
    }
    // The degenerate coordinate's optimum must stay near the visited range
    // [0.5, 1.5], widened by one span — never ~1e6.
    let opt = surrogate.optimum()[1];
    assert!(
        opt.abs() < 10.0,
        "the degenerate coordinate's optimum ran away to {opt}"
    );

    // And the resulting task must have a sane loss at its own starting point.
    let start_loss = surrogate.loss(&surrogate.initial_parameters());
    assert!(
        start_loss.is_finite() && start_loss < 1e3,
        "the surrogate manufactured an absurd starting loss {start_loss}"
    );

    // A well-identified coordinate is still recovered exactly.
    assert!(
        (surrogate.curvature()[0] - 2.0).abs() < 1e-8,
        "clean coordinate curvature {}",
        surrogate.curvature()[0]
    );
    assert!(
        (surrogate.optimum()[0] - 0.5).abs() < 1e-8,
        "clean coordinate optimum {}",
        surrogate.optimum()[0]
    );
}
