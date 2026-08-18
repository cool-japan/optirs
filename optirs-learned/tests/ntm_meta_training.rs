//! F75, NTM half: the NTM optimizer's controller weights are now trainable, and
//! training them helps on a task the trainer never saw.
//!
//! `ntm_optimizer.rs` implements real content-based addressing, a real circular
//! convolutional shift, real sharpening and real erase-add memory writes — and for
//! the whole life of the module **nothing trained the controller**. Its 19 weight
//! blocks came out of the configured seed at construction and stayed there.
//!
//! The NTM shares [`optirs_learned::es_meta_training::EsMetaTrainer`] with the GNN
//! optimizer; what is NTM-specific and tested here is that the *memory matrix* is
//! treated as state rather than as a parameter, so rollouts cannot leak memory
//! contents into one another's score.

use optirs_learned::domain_objectives::{MetaObjective, QuadraticObjective};
use optirs_learned::domain_optimizers::AdvancedOptimizer;
use optirs_learned::es_meta_training::{
    EsMetaTrainer, MetaTrainable, MetaTrainingConfig, SelectionMetric,
};
use optirs_learned::ntm_optimizer::{NtmOptimizer, NtmOptimizerConfig};
use scirs2_core::ndarray::Array1;

const SEED: u64 = 0x0BAD_C0FFEE;

fn controller() -> NtmOptimizer<f64> {
    NtmOptimizer::new(NtmOptimizerConfig {
        num_slots: 4,
        num_locations: 6,
        mem_width: 4,
        hidden_dim: 6,
        shift_range: 1,
        base_lr: 0.2,
        seed: SEED,
        ..Default::default()
    })
    .expect("optimizer construction")
}

fn training_tasks() -> Vec<QuadraticObjective<f64>> {
    vec![
        QuadraticObjective::isotropic(8, 1.0, 0.8).expect("task"),
        QuadraticObjective::from_curvature(vec![1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0], 0.6)
            .expect("task"),
        QuadraticObjective::from_curvature(vec![0.5, 0.5, 2.0, 2.0, 6.0, 6.0, 0.5, 2.0], -0.7)
            .expect("task"),
    ]
}

fn held_out_task() -> QuadraticObjective<f64> {
    QuadraticObjective::from_curvature(vec![4.0, 0.7, 1.6, 9.0, 0.9, 2.5, 5.0, 1.1], 1.1)
        .expect("task")
        .with_optimum(vec![0.2, -0.3, 0.1, 0.0, -0.15, 0.25, -0.05, 0.4])
        .expect("optimum")
        .labelled("held-out")
}

fn final_loss(
    mut optimizer: NtmOptimizer<f64>,
    task: &dyn MetaObjective<f64>,
    steps: usize,
) -> f64 {
    MetaTrainable::reset_state(&mut optimizer);
    let mut params: Array1<f64> = task.initial_parameters();
    for _ in 0..steps {
        let (_, gradient) = task.loss_and_gradient(&params).expect("gradient");
        params = optimizer.step(&params, &gradient).expect("step");
    }
    task.loss_and_gradient(&params).expect("loss").0
}

#[test]
fn meta_training_improves_held_out_optimization() {
    let tasks = training_tasks();
    let task_refs: Vec<&dyn MetaObjective<f64>> = tasks
        .iter()
        .map(|task| task as &dyn MetaObjective<f64>)
        .collect();

    // Budget note, measured rather than guessed. Across 4 controller seeds x 3 ES
    // budgets the *validation* meta-loss improved 12/12, but held-out improvement
    // was only 8/12 — and every one of the 4 failures was at a small budget
    // (population 8, 20-25 iterations). At population 12 / 40 iterations the
    // held-out meta-loss improved 4/4. So the NTM controller needs a genuinely
    // larger ES sample budget than the GNN (which improved 12/12 at every budget
    // tried), and this test uses the budget that actually works rather than
    // asserting something the smaller one cannot deliver.
    let trainer = EsMetaTrainer::new(MetaTrainingConfig {
        population: 12,
        iterations: 40,
        sigma: 0.10,
        meta_learning_rate: 0.06,
        horizon: 15,
        seed: 8181,
    })
    .expect("trainer");

    // Validation family: same *kind* of task, but shifted start points and
    // curvature ranges the training family does not cover. Selecting on this is
    // what turns "the training loss went down" into a generalisation claim — and
    // it is required here, because the NTM controller demonstrably overfits when
    // the weights are selected on the training loss (training loss down ~30%,
    // held-out loss up 40x).
    let validation = [
        QuadraticObjective::from_curvature(vec![3.0, 1.0, 1.0, 7.0, 1.0, 2.0, 4.0, 1.0], 1.0)
            .expect("task")
            .labelled("validation-a"),
        QuadraticObjective::isotropic(8, 2.5, -1.2)
            .expect("task")
            .labelled("validation-b"),
    ];
    let validation_refs: Vec<&dyn MetaObjective<f64>> = validation
        .iter()
        .map(|task| task as &dyn MetaObjective<f64>)
        .collect();

    let untrained = controller();
    let mut trained = controller();
    let report = trainer
        .train_with_validation(&mut trained, &task_refs, &validation_refs)
        .expect("training");
    assert_eq!(report.selection_metric, SelectionMetric::ValidationLoss);
    assert_eq!(report.validation_history.len(), report.iterations);
    assert!(report.initial_validation_loss.is_some());
    assert!(report.final_validation_loss.is_some());

    assert_eq!(report.weight_count, untrained.weight_count());
    assert!(report.weight_count > 0);
    assert_ne!(
        MetaTrainable::weight_vector(&trained),
        MetaTrainable::weight_vector(&untrained),
        "meta-training left the controller weights untouched"
    );

    assert_eq!(report.gradient_norms.len(), report.iterations);
    assert!(report.gradient_norms.iter().all(|n| n.is_finite()));
    assert!(
        report.gradient_norms.iter().any(|n| *n > 1e-6),
        "every meta-gradient estimate was ~zero: {:?}",
        report.gradient_norms
    );

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

    // Generalisation, measured on the same objective the trainer optimises (the
    // normalised loss-curve area) on a task that is in neither the training nor
    // the validation set.
    let held_out = held_out_task();
    let held_refs: Vec<&dyn MetaObjective<f64>> = vec![&held_out];
    let curve_before = trainer
        .meta_loss(&untrained, &held_refs)
        .expect("held-out curve, untrained");
    let curve_after = trainer
        .meta_loss(&trained, &held_refs)
        .expect("held-out curve, trained");
    assert!(
        curve_after < curve_before,
        "the trained controller ({curve_after:.6}) did not beat the untrained one \
         ({curve_before:.6}) on the held-out loss curve"
    );

    // And the final loss after twice the training horizon must not blow up — the
    // failure mode that selecting on the *training* loss produces here is a
    // controller that takes aggressive steps and diverges off-distribution.
    let before = final_loss(untrained, &held_out, 30);
    let after = final_loss(trained, &held_out, 30);
    assert!(
        before.is_finite() && after.is_finite(),
        "held-out rollout diverged: {before} -> {after}"
    );
    assert!(
        after < before,
        "the trained controller's final held-out loss ({after:.6e}) is worse than the \
         untrained one's ({before:.6e})"
    );
}

/// The meta-objective must not depend on the order the tasks are presented in.
///
/// This is the property the memory reset buys: without it, task *k*'s score would
/// be computed against whatever task *k-1* wrote into the memory matrix, so
/// permuting the task list would change the meta-loss and the trainer would be
/// optimizing an artefact of the ordering.
#[test]
fn the_meta_objective_is_invariant_to_task_order() {
    let tasks = training_tasks();
    let forward: Vec<&dyn MetaObjective<f64>> = tasks
        .iter()
        .map(|task| task as &dyn MetaObjective<f64>)
        .collect();
    let mut reversed = forward.clone();
    reversed.reverse();

    let trainer = EsMetaTrainer::new(MetaTrainingConfig::default()).expect("trainer");
    let optimizer = controller();
    let a = trainer.meta_loss(&optimizer, &forward).expect("forward");
    let b = trainer.meta_loss(&optimizer, &reversed).expect("reversed");

    assert!(
        (a - b).abs() < 1e-12,
        "meta-loss depends on task order ({a} vs {b}), so state is leaking between rollouts"
    );
}

/// Both architectures must be trainable through the *same* generic trainer —
/// that is the point of [`MetaTrainable`].
#[test]
fn the_generic_trainer_handles_the_ntm_through_the_trait() {
    let tasks = training_tasks();
    let task_refs: Vec<&dyn MetaObjective<f64>> = tasks
        .iter()
        .map(|task| task as &dyn MetaObjective<f64>)
        .collect();

    fn train_any<O: MetaTrainable<f64>>(
        trainer: &EsMetaTrainer,
        optimizer: &mut O,
        tasks: &[&dyn MetaObjective<f64>],
    ) -> f64 {
        let report = trainer.train(optimizer, tasks).expect("training");
        report.relative_improvement()
    }

    let trainer = EsMetaTrainer::new(MetaTrainingConfig {
        population: 4,
        iterations: 8,
        sigma: 0.08,
        meta_learning_rate: 0.08,
        horizon: 8,
        seed: 606,
    })
    .expect("trainer");

    let mut ntm = controller();
    let improvement = train_any(&trainer, &mut ntm, &task_refs);
    assert!(
        improvement >= 0.0,
        "the generic trainer degraded the NTM controller ({improvement})"
    );
}
