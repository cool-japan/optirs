//! F75: the GNN optimizer's weights are now trainable, and training them helps.
//!
//! `gnn_optimizer.rs` implements a real message-passing round, a real GRU node
//! update and a real learned readout — but for the whole life of the module
//! **nothing trained the weights**. They came out of the configured seed at
//! construction and stayed there, so every "learned" update was produced by a
//! randomly-initialised network.
//!
//! These tests exercise the evolution-strategies entry point end to end and, most
//! importantly, assert *generalisation*: a controller meta-trained on one family
//! of quadratics must beat its untrained self on a quadratic it never saw. A test
//! that only checked the training loss went down would not distinguish real
//! meta-learning from memorising the training set.

use optirs_learned::domain_objectives::{MetaObjective, QuadraticObjective};
use optirs_learned::domain_optimizers::AdvancedOptimizer;
use optirs_learned::es_meta_training::{EsMetaTrainer, MetaTrainingConfig};
use optirs_learned::gnn_optimizer::{GnnOptimizer, GnnOptimizerConfig, GraphTopology};
use scirs2_core::ndarray::Array1;

const SEED: u64 = 20_260_817;

fn controller() -> GnnOptimizer<f64> {
    GnnOptimizer::new(GnnOptimizerConfig {
        num_nodes: 4,
        hidden_dim: 6,
        num_message_rounds: 2,
        topology: GraphTopology::Chain,
        base_lr: 0.2,
        seed: SEED,
        ..Default::default()
    })
    .expect("optimizer construction")
}

/// Training family: eight-dimensional bowls with a range of conditioning.
fn training_tasks() -> Vec<QuadraticObjective<f64>> {
    vec![
        QuadraticObjective::isotropic(8, 1.0, 0.8).expect("task"),
        QuadraticObjective::from_curvature(vec![1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0], 0.6)
            .expect("task"),
        QuadraticObjective::from_curvature(vec![0.5, 0.5, 2.0, 2.0, 6.0, 6.0, 0.5, 2.0], -0.7)
            .expect("task"),
    ]
}

/// Held-out task: a curvature pattern, start point and displaced optimum that
/// appear in none of the training tasks.
fn held_out_task() -> QuadraticObjective<f64> {
    QuadraticObjective::from_curvature(vec![4.0, 0.7, 1.6, 9.0, 0.9, 2.5, 5.0, 1.1], 1.1)
        .expect("task")
        .with_optimum(vec![0.2, -0.3, 0.1, 0.0, -0.15, 0.25, -0.05, 0.4])
        .expect("optimum")
        .labelled("held-out")
}

/// Run `steps` optimizer steps on `task` and return the final loss.
fn final_loss(
    mut optimizer: GnnOptimizer<f64>,
    task: &dyn MetaObjective<f64>,
    steps: usize,
) -> f64 {
    optimizer.reset_state();
    let mut params: Array1<f64> = task.initial_parameters();
    for _ in 0..steps {
        let (_, gradient) = task.loss_and_gradient(&params).expect("gradient");
        params = optimizer.step(&params, &gradient).expect("step");
    }
    task.loss_and_gradient(&params).expect("loss").0
}

/// The headline F75 assertion.
#[test]
fn meta_training_improves_held_out_optimization() {
    let tasks = training_tasks();
    let task_refs: Vec<&dyn MetaObjective<f64>> = tasks
        .iter()
        .map(|task| task as &dyn MetaObjective<f64>)
        .collect();

    let trainer = EsMetaTrainer::new(MetaTrainingConfig {
        population: 8,
        iterations: 25,
        sigma: 0.08,
        meta_learning_rate: 0.08,
        horizon: 15,
        seed: 4242,
    })
    .expect("trainer");

    let untrained = controller();
    let mut trained = controller();
    let report = trainer.train(&mut trained, &task_refs).expect("training");

    // 1. Training actually moved the weights.
    assert_eq!(report.weight_count, untrained.weight_count());
    assert!(report.weight_count > 0);
    assert_ne!(
        trained.weight_vector(),
        untrained.weight_vector(),
        "meta-training left the weights untouched"
    );

    // 2. The ES gradient estimate was non-trivial on every iteration; an
    //    all-zero estimate is what a disconnected/no-op trainer would produce.
    assert_eq!(report.gradient_norms.len(), report.iterations);
    assert_eq!(report.loss_history.len(), report.iterations);
    assert!(
        report.gradient_norms.iter().all(|n| n.is_finite()),
        "non-finite meta-gradient norm"
    );
    assert!(
        report.gradient_norms.iter().any(|n| *n > 1e-6),
        "every meta-gradient estimate was ~zero: {:?}",
        report.gradient_norms
    );

    // 3. Training reduced the training meta-loss.
    assert!(
        report.improved(),
        "meta-loss did not improve: {} -> {}",
        report.initial_meta_loss,
        report.final_meta_loss
    );
    assert!(
        report.relative_improvement() > 0.02,
        "improvement {:.4} is too small to be meaningful",
        report.relative_improvement()
    );

    // 4. And — the part that makes it meta-*learning* — it generalises to a task
    //    the trainer never saw.
    let held_out = held_out_task();
    let before = final_loss(untrained, &held_out, 30);
    let after = final_loss(trained, &held_out, 30);
    assert!(
        before.is_finite() && after.is_finite(),
        "held-out rollout diverged: {before} -> {after}"
    );
    assert!(
        after < before,
        "the trained controller ({after:.6e}) did not beat the untrained one \
         ({before:.6e}) on the held-out task"
    );
}

/// Meta-training must be reproducible from its seed: same trainer seed, same
/// controller seed, same tasks -> bit-identical weights.
#[test]
fn meta_training_is_reproducible_from_its_seed() {
    let tasks = training_tasks();
    let task_refs: Vec<&dyn MetaObjective<f64>> = tasks
        .iter()
        .map(|task| task as &dyn MetaObjective<f64>)
        .collect();
    let config = MetaTrainingConfig {
        population: 4,
        iterations: 6,
        sigma: 0.08,
        meta_learning_rate: 0.08,
        horizon: 8,
        seed: 99,
    };

    let trainer = EsMetaTrainer::new(config.clone()).expect("trainer");
    let mut first = controller();
    let report_a = trainer.train(&mut first, &task_refs).expect("run a");

    let trainer = EsMetaTrainer::new(config).expect("trainer");
    let mut second = controller();
    let report_b = trainer.train(&mut second, &task_refs).expect("run b");

    assert_eq!(first.weight_vector(), second.weight_vector());
    assert_eq!(report_a.gradient_norms, report_b.gradient_norms);
    assert!((report_a.final_meta_loss - report_b.final_meta_loss).abs() < 1e-15);
}

/// The trainer must never hand back weights worse than the ones it was given —
/// an ES run on a noisy objective can wander uphill, and returning a degraded
/// controller while reporting "trained" is the failure mode this whole wave is
/// about.
#[test]
fn training_never_returns_worse_weights_than_it_received() {
    let tasks = training_tasks();
    let task_refs: Vec<&dyn MetaObjective<f64>> = tasks
        .iter()
        .map(|task| task as &dyn MetaObjective<f64>)
        .collect();

    // A deliberately hostile configuration: a huge step size, so plain ES would
    // very likely end at a worse point than it started.
    let trainer = EsMetaTrainer::new(MetaTrainingConfig {
        population: 4,
        iterations: 8,
        sigma: 0.5,
        meta_learning_rate: 2.0,
        horizon: 6,
        seed: 5,
    })
    .expect("trainer");

    let mut optimizer = controller();
    let report = trainer.train(&mut optimizer, &task_refs).expect("training");
    assert!(
        report.final_meta_loss <= report.initial_meta_loss,
        "training returned worse weights: {} -> {}",
        report.initial_meta_loss,
        report.final_meta_loss
    );

    // And the weights it kept must really score what it reported.
    let measured = trainer
        .meta_loss(&optimizer, &task_refs)
        .expect("re-measure");
    assert!(
        (measured - report.final_meta_loss).abs() < 1e-12,
        "reported final meta-loss {} does not match the kept weights' {measured}",
        report.final_meta_loss
    );
}

/// The trainer must reject a task whose declared dimension disagrees with the
/// parameters it hands out, instead of rolling out something incoherent.
#[test]
fn an_inconsistent_task_is_rejected() {
    struct Liar;
    impl MetaObjective<f64> for Liar {
        fn dimension(&self) -> usize {
            5
        }
        fn initial_parameters(&self) -> Array1<f64> {
            Array1::zeros(3)
        }
        fn loss_and_gradient(
            &self,
            params: &Array1<f64>,
        ) -> optirs_learned::Result<(f64, Array1<f64>)> {
            Ok((params.iter().map(|v| v * v).sum(), params.mapv(|v| 2.0 * v)))
        }
    }

    let trainer = EsMetaTrainer::new(MetaTrainingConfig::default()).expect("trainer");
    let liar = Liar;
    let refs: Vec<&dyn MetaObjective<f64>> = vec![&liar];
    assert!(trainer.meta_loss(&controller(), &refs).is_err());
    assert!(trainer.train(&mut controller(), &refs).is_err());
}
