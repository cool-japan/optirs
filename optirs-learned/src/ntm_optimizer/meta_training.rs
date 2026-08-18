//! [`crate::es_meta_training::MetaTrainable`] implementation for [`NtmOptimizer`] (finding F75).
//!
//! The generic evolution-strategies loop lives in [`crate::es_meta_training`],
//! which also explains why ES rather than backpropagation. This file supplies the
//! three architecture-specific pieces the loop needs: flatten the NTM controller's
//! 19 learned weight blocks into one vector, load them back, and clear the
//! persistent per-rollout state.
//!
//! # What counts as a weight and what counts as state
//!
//! This distinction matters more for the NTM than for any other optimizer in the
//! crate, because the *memory matrix* looks like a parameter and is not one:
//!
//! * **Learned weights** (trained here): the controller's input encoder, the seven
//!   head projections (key, β, gate, shift, γ, erase, add) and the output-scale
//!   readout. These are what the meta-gradient acts on.
//! * **State** (reset between rollouts): the memory matrix, the previous
//!   addressing weights, the previous read vector, the gradient EMAs, the chunk
//!   layout and the step count. The memory is *written by* the erase-add rule
//!   during a rollout, so carrying it between tasks would let one task's memory
//!   contents leak into another's score — and would make the meta-objective depend
//!   on task ordering. [`NtmOptimizer::reset_state`] restores it to the value the
//!   seed produced at construction, kept for exactly this purpose.
//!
//! Before this existed, the controller weights were drawn once from the configured
//! seed and never changed again, so every "learned" update came out of a
//! randomly-initialised controller.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::{NtmController, NtmOptimizer};
use crate::error::Result;
use crate::es_meta_training::{
    expect_fully_consumed, pull_matrix, pull_scalar, pull_vector, push_matrix, push_scalar,
    push_vector, MetaTrainable,
};

/// Convenience alias: the ES trainer, named for the architecture it is usually
/// pointed at here. It is not NTM-specific — see [`crate::es_meta_training`].
pub use crate::es_meta_training::EsMetaTrainer as NtmMetaTrainer;
pub use crate::es_meta_training::{MetaTrainingConfig, MetaTrainingReport, SelectionMetric};

impl<T: Float + Debug + Send + Sync + 'static> NtmController<T> {
    /// Flatten all 19 learned blocks into one vector, in a stable order.
    fn to_flat(&self) -> Vec<f64> {
        let mut out = Vec::new();
        push_matrix(&self.w_in, &mut out);
        push_vector(&self.b_in, &mut out);
        push_matrix(&self.w_key, &mut out);
        push_vector(&self.b_key, &mut out);
        push_vector(&self.w_beta, &mut out);
        push_scalar(self.b_beta, &mut out);
        push_vector(&self.w_gate, &mut out);
        push_scalar(self.b_gate, &mut out);
        push_matrix(&self.w_shift, &mut out);
        push_vector(&self.b_shift, &mut out);
        push_vector(&self.w_gamma, &mut out);
        push_scalar(self.b_gamma, &mut out);
        push_matrix(&self.w_erase, &mut out);
        push_vector(&self.b_erase, &mut out);
        push_matrix(&self.w_add, &mut out);
        push_vector(&self.b_add, &mut out);
        push_vector(&self.w_out, &mut out);
        push_vector(&self.v_out, &mut out);
        push_scalar(self.b_out, &mut out);
        out
    }

    /// Load all 19 learned blocks from a vector produced by [`Self::to_flat`].
    fn load_flat(&mut self, flat: &[f64]) -> Result<()> {
        let mut cursor = 0usize;
        pull_matrix(&mut self.w_in, flat, &mut cursor)?;
        pull_vector(&mut self.b_in, flat, &mut cursor)?;
        pull_matrix(&mut self.w_key, flat, &mut cursor)?;
        pull_vector(&mut self.b_key, flat, &mut cursor)?;
        pull_vector(&mut self.w_beta, flat, &mut cursor)?;
        pull_scalar(&mut self.b_beta, flat, &mut cursor)?;
        pull_vector(&mut self.w_gate, flat, &mut cursor)?;
        pull_scalar(&mut self.b_gate, flat, &mut cursor)?;
        pull_matrix(&mut self.w_shift, flat, &mut cursor)?;
        pull_vector(&mut self.b_shift, flat, &mut cursor)?;
        pull_vector(&mut self.w_gamma, flat, &mut cursor)?;
        pull_scalar(&mut self.b_gamma, flat, &mut cursor)?;
        pull_matrix(&mut self.w_erase, flat, &mut cursor)?;
        pull_vector(&mut self.b_erase, flat, &mut cursor)?;
        pull_matrix(&mut self.w_add, flat, &mut cursor)?;
        pull_vector(&mut self.b_add, flat, &mut cursor)?;
        pull_vector(&mut self.w_out, flat, &mut cursor)?;
        pull_vector(&mut self.v_out, flat, &mut cursor)?;
        pull_scalar(&mut self.b_out, flat, &mut cursor)?;
        expect_fully_consumed(flat, cursor)
    }
}

impl<T: Float + Debug + Send + Sync + 'static> NtmOptimizer<T> {
    /// Every learned controller weight, flattened in a stable order.
    ///
    /// The **memory matrix is deliberately excluded**: it is written by the
    /// erase-add rule during a rollout, so it is state, not a parameter. Including
    /// it would let the trainer optimize a starting memory that the forward path
    /// immediately overwrites.
    pub fn weight_vector(&self) -> Vec<f64> {
        self.controller.to_flat()
    }

    /// Number of learned controller weights.
    pub fn weight_count(&self) -> usize {
        self.controller.to_flat().len()
    }

    /// Overwrite every learned controller weight from a vector laid out like
    /// [`Self::weight_vector`]'s output.
    ///
    /// # Errors
    /// Returns [`crate::error::OptimError::InvalidConfig`] if `flat` is not
    /// exactly the right length for this architecture — a silent partial load
    /// would leave the controller in a state no caller could reason about.
    pub fn set_weight_vector(&mut self, flat: &[f64]) -> Result<()> {
        self.controller.load_flat(flat)
    }

    /// Clear the persistent state — memory, addressing weights, read vector,
    /// gradient EMAs, chunk layout, step count — while keeping the learned
    /// controller weights.
    ///
    /// The memory is restored to the value the construction seed produced, not
    /// zeroed: a zero memory makes every cosine similarity in
    /// content addressing degenerate, which is a different (and worse) starting
    /// condition than the one the optimizer was built with.
    pub fn reset_state(&mut self) {
        self.memory = self.initial_memory.clone();
        let uniform: T = scirs2_core::numeric::NumCast::from(1.0 / self.num_locations as f64)
            .unwrap_or_else(|| T::zero());
        self.prev_weights = Array1::from_elem(self.num_locations, uniform);
        self.last_weights = self.prev_weights.clone();
        self.prev_read = Array1::zeros(self.mem_width);
        self.param_len = 0;
        self.num_slots = 0;
        self.chunk_bounds.clear();
        self.grad_ema = Array1::zeros(0);
        self.grad_sq_ema = Array1::zeros(0);
        self.step_count = 0;
        self.grad_norm_ema = T::zero();
        self.current_lr = self.base_lr;
    }
}

impl<T: Float + Debug + Send + Sync + 'static> MetaTrainable<T> for NtmOptimizer<T> {
    fn weight_vector(&self) -> Vec<f64> {
        NtmOptimizer::weight_vector(self)
    }

    fn set_weight_vector(&mut self, flat: &[f64]) -> Result<()> {
        NtmOptimizer::set_weight_vector(self, flat)
    }

    fn reset_state(&mut self) {
        NtmOptimizer::reset_state(self)
    }

    fn weight_count(&self) -> usize {
        NtmOptimizer::weight_count(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain_objectives::{MetaObjective, QuadraticObjective};
    use crate::domain_optimizers::AdvancedOptimizer;
    use crate::ntm_optimizer::NtmOptimizerConfig;

    fn optimizer(seed: u64) -> NtmOptimizer<f64> {
        NtmOptimizer::new(NtmOptimizerConfig {
            num_slots: 4,
            num_locations: 6,
            mem_width: 4,
            hidden_dim: 6,
            shift_range: 1,
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
        let mut opt = optimizer(13);
        let flat = opt.weight_vector();
        assert!(!flat.is_empty());
        assert_eq!(flat.len(), opt.weight_count());
        assert!(
            flat.iter().any(|v| v.abs() > 1e-9),
            "the weight vector is all zeros, so it is not reading real weights"
        );

        opt.set_weight_vector(&flat).expect("round trip");
        assert_eq!(opt.weight_vector(), flat);

        assert!(opt.set_weight_vector(&flat[..flat.len() - 1]).is_err());
        let mut too_long = flat.clone();
        too_long.push(0.0);
        assert!(opt.set_weight_vector(&too_long).is_err());
    }

    /// The memory matrix is state, not a weight, and must not appear in the
    /// flattened vector.
    #[test]
    fn the_memory_matrix_is_not_part_of_the_weight_vector() {
        let opt = optimizer(13);
        let weights = opt.weight_count();
        let memory_cells = opt.memory().len();
        assert!(memory_cells > 0);

        // The controller is (hidden × input) + head projections; the memory is
        // (num_locations × mem_width) = 24 cells. If the memory were included the
        // count would change when only the memory does, so compare against a
        // second optimizer whose memory differs (different seed changes both, so
        // instead check the count is invariant to memory *mutation*).
        let mut mutated = opt.clone();
        let params = Array1::from_vec(vec![1.0; 8]);
        let gradient = Array1::from_vec(vec![0.3; 8]);
        for _ in 0..4 {
            mutated.step(&params, &gradient).expect("step");
        }
        assert!(
            (mutated.memory() - opt.memory()).mapv(f64::abs).sum() > 1e-12,
            "stepping did not write to memory, so there is nothing to distinguish"
        );
        assert_eq!(
            mutated.weight_count(),
            weights,
            "the weight count changed when only the memory did"
        );
        assert_eq!(
            mutated.weight_vector(),
            opt.weight_vector(),
            "writing to memory changed the weight vector, so memory is being \
             treated as a learned parameter"
        );
    }

    /// `reset_state` must restore the seeded memory and clear momentum, so the
    /// first step after a reset matches a fresh optimizer's first step.
    #[test]
    fn reset_state_restores_the_seeded_memory_and_clears_momentum() {
        let params = Array1::from_vec(vec![1.0; 8]);
        let gradient = Array1::from_vec(vec![0.5; 8]);

        let mut fresh = optimizer(29);
        let pristine_memory = fresh.memory().clone();
        let baseline = fresh.step(&params, &gradient).expect("step");

        let mut used = optimizer(29);
        for _ in 0..5 {
            used.step(&params, &gradient).expect("warm-up step");
        }
        assert!(
            (used.memory() - &pristine_memory).mapv(f64::abs).sum() > 1e-12,
            "the warm-up left the memory untouched"
        );

        used.reset_state();
        assert!(
            (used.memory() - &pristine_memory).mapv(f64::abs).sum() < 1e-12,
            "reset_state did not restore the seeded memory"
        );
        let after_reset = used.step(&params, &gradient).expect("step");
        assert!(
            (&after_reset - &baseline).mapv(f64::abs).sum() < 1e-12,
            "reset_state did not restore the fresh-optimizer behaviour"
        );
        assert_eq!(used.weight_vector(), fresh.weight_vector());
    }

    /// `reset_state` is public and clears the chunk layout and the gradient EMAs to
    /// zero length. Every *other* public accessor must stay callable in between,
    /// with no intervening `step` to rebuild them — a caller can write
    /// `reset_state(); get_state();` and must not get a panic or a division by
    /// zero. `last_address_weights` is the one that would return an empty array if
    /// the reset zeroed it instead of restoring the uniform distribution.
    #[test]
    fn every_accessor_stays_safe_after_a_bare_reset() {
        let mut opt = optimizer(37);
        let params = Array1::from_vec(vec![1.0; 8]);
        let gradient = Array1::from_vec(vec![0.25; 8]);
        for _ in 0..3 {
            opt.step(&params, &gradient).expect("warm-up");
        }

        opt.reset_state();

        let state = opt.get_state();
        assert_eq!(state.step_count, 0);
        assert!(state.current_lr.is_finite());
        assert!(state.grad_norm_ema.is_finite());
        assert_eq!(opt.get_learning_rate(), state.current_lr);
        // Documented to be 0 before the first step.
        assert_eq!(opt.num_slots(), 0);
        assert_eq!(opt.num_locations(), 6);
        assert_eq!(opt.mem_width(), 4);
        assert!(opt.hidden_dim() > 0);
        assert_eq!(opt.shift_range(), 1);
        assert_eq!(opt.seed(), 37);
        assert_eq!(opt.name(), "NtmOptimizer");
        // The addressing state must be a usable distribution, not an empty array.
        let weights = opt.last_address_weights();
        assert_eq!(weights.len(), opt.num_locations());
        let total: f64 = weights.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "addressing weights should sum to 1 after a reset, got {total}"
        );
        assert_eq!(opt.memory().dim(), (6, 4));
        assert!(!opt.weight_vector().is_empty());

        // Idempotent, and still usable afterwards.
        opt.reset_state();
        assert_eq!(opt.num_slots(), 0);
        assert!(opt.step(&params, &gradient).is_ok());
        assert!(opt.num_slots() > 0);
    }

    #[test]
    fn meta_loss_is_a_real_measurement() {
        let tasks = training_tasks();
        let refs: Vec<&dyn MetaObjective<f64>> =
            tasks.iter().map(|t| t as &dyn MetaObjective<f64>).collect();
        let trainer = NtmMetaTrainer::new(MetaTrainingConfig::default()).expect("trainer");

        let a = trainer.meta_loss(&optimizer(1), &refs).expect("loss a");
        let b = trainer.meta_loss(&optimizer(2), &refs).expect("loss b");
        assert!(a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0);
        assert!(
            (a - b).abs() > 1e-9,
            "two different controller draws scored identically ({a} vs {b})"
        );

        let opt = optimizer(1);
        let snapshot = opt.weight_vector();
        let repeat = trainer.meta_loss(&opt, &refs).expect("repeat");
        assert_eq!(opt.weight_vector(), snapshot);
        assert!((repeat - a).abs() < 1e-12, "meta_loss is not deterministic");
    }
}
