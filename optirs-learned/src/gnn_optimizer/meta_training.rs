//! [`crate::es_meta_training::MetaTrainable`] implementation for [`GnnOptimizer`] (finding F75).
//!
//! The generic evolution-strategies loop lives in [`crate::es_meta_training`],
//! which also explains why ES rather than backpropagation. This file supplies the
//! three architecture-specific pieces the loop needs: flatten the GNN's learned
//! weights into one vector, load them back, and clear the persistent per-rollout
//! state.
//!
//! Before this existed, `GnnOptimizer`'s weights were drawn once from the
//! configured seed and never changed again, so every "learned" update came out of
//! a randomly-initialised network.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::{GnnOptimizer, GnnWeights, GruCell};
use crate::error::Result;
use crate::es_meta_training::{
    expect_fully_consumed, pull_matrix, pull_scalar, pull_vector, push_matrix, push_scalar,
    push_vector, MetaTrainable,
};

/// Convenience alias: the ES trainer, named for the architecture it is usually
/// pointed at here. It is not GNN-specific — see [`crate::es_meta_training`].
pub use crate::es_meta_training::EsMetaTrainer as GnnMetaTrainer;
pub use crate::es_meta_training::{MetaTrainingConfig, MetaTrainingReport, SelectionMetric};

impl<T: Float + Debug + Send + Sync + 'static> GruCell<T> {
    /// Append the cell's nine parameter blocks to `out` in a fixed order.
    fn extend_flat(&self, out: &mut Vec<f64>) {
        push_matrix(&self.w_z, out);
        push_matrix(&self.u_z, out);
        push_vector(&self.b_z, out);
        push_matrix(&self.w_r, out);
        push_matrix(&self.u_r, out);
        push_vector(&self.b_r, out);
        push_matrix(&self.w_h, out);
        push_matrix(&self.u_h, out);
        push_vector(&self.b_h, out);
    }

    /// Load the cell's nine parameter blocks from `flat`, in the same order.
    fn load_flat(&mut self, flat: &[f64], cursor: &mut usize) -> Result<()> {
        pull_matrix(&mut self.w_z, flat, cursor)?;
        pull_matrix(&mut self.u_z, flat, cursor)?;
        pull_vector(&mut self.b_z, flat, cursor)?;
        pull_matrix(&mut self.w_r, flat, cursor)?;
        pull_matrix(&mut self.u_r, flat, cursor)?;
        pull_vector(&mut self.b_r, flat, cursor)?;
        pull_matrix(&mut self.w_h, flat, cursor)?;
        pull_matrix(&mut self.u_h, flat, cursor)?;
        pull_vector(&mut self.b_h, flat, cursor)?;
        Ok(())
    }
}

impl<T: Float + Debug + Send + Sync + 'static> GnnWeights<T> {
    /// Flatten every learned weight into one vector, in a stable order.
    fn to_flat(&self) -> Vec<f64> {
        let mut out = Vec::new();
        push_matrix(&self.w_in, &mut out);
        push_vector(&self.b_in, &mut out);
        push_matrix(&self.w_msg, &mut out);
        push_vector(&self.b_msg, &mut out);
        self.gru.extend_flat(&mut out);
        push_vector(&self.w_out, &mut out);
        push_scalar(self.b_out, &mut out);
        out
    }

    /// Load every learned weight from a vector produced by [`Self::to_flat`].
    fn load_flat(&mut self, flat: &[f64]) -> Result<()> {
        let mut cursor = 0usize;
        pull_matrix(&mut self.w_in, flat, &mut cursor)?;
        pull_vector(&mut self.b_in, flat, &mut cursor)?;
        pull_matrix(&mut self.w_msg, flat, &mut cursor)?;
        pull_vector(&mut self.b_msg, flat, &mut cursor)?;
        self.gru.load_flat(flat, &mut cursor)?;
        pull_vector(&mut self.w_out, flat, &mut cursor)?;
        pull_scalar(&mut self.b_out, flat, &mut cursor)?;
        expect_fully_consumed(flat, cursor)
    }
}

impl<T: Float + Debug + Send + Sync + 'static> GnnOptimizer<T> {
    /// Every learned weight, flattened in a stable order.
    ///
    /// The layout is an implementation detail, but it is *stable across calls on
    /// the same architecture*, which is what a meta-training loop needs:
    /// `set_weight_vector(&weight_vector())` is the identity.
    pub fn weight_vector(&self) -> Vec<f64> {
        self.weights.to_flat()
    }

    /// Number of learned weights.
    pub fn weight_count(&self) -> usize {
        self.weights.to_flat().len()
    }

    /// Overwrite every learned weight from a vector laid out like
    /// [`Self::weight_vector`]'s output.
    ///
    /// # Errors
    /// Returns [`crate::error::OptimError::InvalidConfig`] if `flat` is not
    /// exactly the right length for this architecture — a silent partial load
    /// would leave the network in a state no caller could reason about.
    pub fn set_weight_vector(&mut self, flat: &[f64]) -> Result<()> {
        self.weights.load_flat(flat)
    }

    /// Clear the persistent optimization state (per-node EMAs, GRU hidden
    /// vectors, step count, graph layout) while keeping the learned weights.
    ///
    /// Meta-training evaluates the same weights on several tasks, and a rollout
    /// must not inherit momentum from the previous one. The graph layout is
    /// discarded too because it is derived from the parameter length, which
    /// differs between tasks; it is rebuilt lazily on the next step.
    pub fn reset_state(&mut self) {
        self.param_len = 0;
        self.num_nodes = 0;
        self.chunk_bounds.clear();
        self.adjacency.clear();
        self.grad_ema = Array1::zeros(0);
        self.grad_sq_ema = Array1::zeros(0);
        self.node_hidden.clear();
        self.node_history = Array1::zeros(0);
        self.step_count = 0;
        self.grad_norm_ema = T::zero();
        self.current_lr = self.base_lr;
    }
}

impl<T: Float + Debug + Send + Sync + 'static> MetaTrainable<T> for GnnOptimizer<T> {
    fn weight_vector(&self) -> Vec<f64> {
        GnnOptimizer::weight_vector(self)
    }

    fn set_weight_vector(&mut self, flat: &[f64]) -> Result<()> {
        GnnOptimizer::set_weight_vector(self, flat)
    }

    fn reset_state(&mut self) {
        GnnOptimizer::reset_state(self)
    }

    fn weight_count(&self) -> usize {
        GnnOptimizer::weight_count(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain_objectives::{MetaObjective, QuadraticObjective};
    use crate::domain_optimizers::AdvancedOptimizer;
    use crate::gnn_optimizer::{GnnOptimizerConfig, GraphTopology};

    fn optimizer(seed: u64) -> GnnOptimizer<f64> {
        GnnOptimizer::new(GnnOptimizerConfig {
            num_nodes: 4,
            hidden_dim: 6,
            num_message_rounds: 2,
            topology: GraphTopology::Chain,
            base_lr: 0.2,
            seed,
            ..Default::default()
        })
        .expect("optimizer")
    }

    fn training_tasks() -> Vec<QuadraticObjective<f64>> {
        vec![
            QuadraticObjective::isotropic(8, 1.0, 0.7).expect("task"),
            QuadraticObjective::from_curvature(vec![1.0, 4.0, 1.0, 4.0, 1.0, 4.0, 1.0, 4.0], 0.5)
                .expect("task"),
        ]
    }

    #[test]
    fn weight_vector_round_trips() {
        let mut opt = optimizer(11);
        let flat = opt.weight_vector();
        assert!(!flat.is_empty());
        assert_eq!(flat.len(), opt.weight_count());
        assert!(
            flat.iter().any(|v| v.abs() > 1e-9),
            "the weight vector is all zeros, so it is not reading real weights"
        );

        opt.set_weight_vector(&flat).expect("round trip");
        assert_eq!(opt.weight_vector(), flat);

        // Wrong lengths must be refused, not partially applied.
        assert!(opt.set_weight_vector(&flat[..flat.len() - 1]).is_err());
        let mut too_long = flat.clone();
        too_long.push(0.0);
        assert!(opt.set_weight_vector(&too_long).is_err());
    }

    /// Changing the weights must change what the optimizer does — otherwise
    /// "meta-training" could not possibly matter.
    #[test]
    fn the_weights_actually_drive_the_update() {
        let mut opt = optimizer(3);
        let params = Array1::from_vec(vec![1.0; 8]);
        let gradient = Array1::from_vec(vec![0.4, -0.3, 0.2, 0.9, -0.1, 0.5, -0.7, 0.15]);

        let before = opt.step(&params, &gradient).expect("step");

        let mut flat = opt.weight_vector();
        // Push the readout bias hard negative: the step scale is a logistic of
        // the readout, so this must shrink every update.
        let last = flat.len() - 1;
        flat[last] = -8.0;
        let mut shrunk = optimizer(3);
        shrunk.set_weight_vector(&flat).expect("set");
        let after = shrunk.step(&params, &gradient).expect("step");

        let moved_before = (&before - &params).mapv(f64::abs).sum();
        let moved_after = (&after - &params).mapv(f64::abs).sum();
        assert!(
            moved_after < moved_before * 0.5,
            "a strongly negative readout bias must shrink the update: \
             {moved_before} -> {moved_after}"
        );
    }

    #[test]
    fn meta_loss_is_a_real_measurement() {
        let tasks = training_tasks();
        let refs: Vec<&dyn MetaObjective<f64>> =
            tasks.iter().map(|t| t as &dyn MetaObjective<f64>).collect();
        let trainer = GnnMetaTrainer::new(MetaTrainingConfig::default()).expect("trainer");

        let a = trainer.meta_loss(&optimizer(1), &refs).expect("loss a");
        let b = trainer.meta_loss(&optimizer(2), &refs).expect("loss b");
        assert!(a.is_finite() && b.is_finite());
        assert!(
            a > 0.0 && b > 0.0,
            "normalised loss curves must be positive"
        );
        assert!(
            (a - b).abs() > 1e-9,
            "two different weight draws scored identically ({a} vs {b}), so meta_loss is not \
             reading the weights"
        );

        // Measuring must not mutate the optimizer it measures.
        let opt = optimizer(1);
        let snapshot = opt.weight_vector();
        let repeat = trainer.meta_loss(&opt, &refs).expect("repeat");
        assert_eq!(opt.weight_vector(), snapshot);
        assert!((repeat - a).abs() < 1e-12, "meta_loss is not deterministic");
    }

    #[test]
    fn meta_loss_rejects_an_empty_task_set() {
        let trainer = GnnMetaTrainer::new(MetaTrainingConfig::default()).expect("trainer");
        let empty: [&dyn MetaObjective<f64>; 0] = [];
        assert!(trainer.meta_loss(&optimizer(1), &empty).is_err());
        assert!(trainer.train(&mut optimizer(1), &empty).is_err());
    }

    /// `reset_state` must clear momentum, so the first step after a reset is the
    /// same as the first step of a fresh optimizer.
    #[test]
    fn reset_state_clears_the_persistent_state() {
        let params = Array1::from_vec(vec![1.0; 8]);
        let gradient = Array1::from_vec(vec![0.5; 8]);

        let mut fresh = optimizer(21);
        let baseline = fresh.step(&params, &gradient).expect("step");

        let mut used = optimizer(21);
        for _ in 0..5 {
            used.step(&params, &gradient).expect("warm-up step");
        }
        let warm = used.step(&params, &gradient).expect("step");
        assert!(
            (&warm - &baseline).mapv(f64::abs).sum() > 1e-12,
            "five warm-up steps left no trace, so there is no state to reset"
        );

        used.reset_state();
        let after_reset = used.step(&params, &gradient).expect("step");
        assert!(
            (&after_reset - &baseline).mapv(f64::abs).sum() < 1e-12,
            "reset_state did not restore the fresh-optimizer behaviour"
        );
        // The weights must survive the reset.
        assert_eq!(used.weight_vector(), fresh.weight_vector());
    }

    /// `reset_state` is public and clears the graph layout, the chunk bounds and
    /// the gradient EMAs to zero length. Every *other* public accessor must stay
    /// callable in between, with no intervening `step` to rebuild them — a caller
    /// can write `reset_state(); get_state();` and must not get a panic or a
    /// division by zero.
    #[test]
    fn every_accessor_stays_safe_after_a_bare_reset() {
        let mut opt = optimizer(31);
        // Warm it up first, so the reset really has state to clear.
        let params = Array1::from_vec(vec![1.0; 8]);
        let gradient = Array1::from_vec(vec![0.25; 8]);
        for _ in 0..3 {
            opt.step(&params, &gradient).expect("warm-up");
        }

        opt.reset_state();

        // No `step` in between: these are the reachable read paths.
        let state = opt.get_state();
        assert_eq!(state.step_count, 0);
        assert!(state.current_lr.is_finite());
        assert!(state.grad_norm_ema.is_finite());
        assert_eq!(opt.get_learning_rate(), state.current_lr);
        // Documented to be 0 before the first step.
        assert_eq!(opt.num_nodes(), 0);
        assert!(opt.hidden_dim() > 0);
        assert_eq!(opt.seed(), 31);
        assert_eq!(opt.name(), "GNNOptimizer");
        assert!(!opt.weight_vector().is_empty());
        assert!(opt.weight_count() > 0);
        // A second reset must be idempotent, not a panic.
        opt.reset_state();
        assert_eq!(opt.num_nodes(), 0);
        // And the optimizer must still be usable afterwards.
        assert!(opt.step(&params, &gradient).is_ok());
        assert!(opt.num_nodes() > 0);
    }
}
