// Hybrid quantum-classical optimizer.
//
// `HybridQuantumClassical` wraps a quantum-inspired exploration phase
// (`QuantumAnnealing`) with a classical refinement phase (`Adam`). The
// optimizer transitions from exploration to refinement after a configurable
// number of steps, optionally seeding the classical optimizer with the best
// parameters discovered during exploration.

use scirs2_core::ndarray::{Array, Dimension, ScalarOperand};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::optimizers::{Adam, Optimizer};

use super::annealing::QuantumAnnealing;

/// Phase of the hybrid quantum-classical optimization process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationPhase {
    /// Exploration via quantum-inspired annealing.
    Exploration,
    /// Refinement via classical (Adam) updates.
    Refinement,
}

/// Hybrid quantum-classical optimizer.
///
/// `HybridQuantumClassical` blends quantum-inspired exploration with classical
/// refinement, much in the spirit of variational hybrid algorithms. The first
/// `switch_step` updates are produced by a [`QuantumAnnealing`] instance that
/// performs broad exploration of the loss landscape. Subsequent updates are
/// produced by an [`Adam`] instance whose state is implicitly warm-started by
/// the best parameters discovered by the annealer.
///
/// # Examples
///
/// ```
/// use optirs_core::quantum_inspired::HybridQuantumClassical;
/// use optirs_core::optimizers::Optimizer;
/// use scirs2_core::ndarray::Array1;
///
/// let mut optimizer: HybridQuantumClassical<f64> =
///     HybridQuantumClassical::new(0.05, 50);
///
/// let params = Array1::from_vec(vec![1.0, -1.0, 0.5]);
/// let grads = Array1::from_vec(vec![0.1, -0.2, 0.05]);
/// let _ = optimizer.step(&params, &grads).expect("step failed");
/// ```
#[derive(Debug)]
pub struct HybridQuantumClassical<A: Float + ScalarOperand + Debug> {
    /// Quantum-inspired exploration optimizer.
    quantum: QuantumAnnealing<A>,
    /// Classical refinement optimizer.
    classical: Adam<A>,
    /// Current phase.
    phase: OptimizationPhase,
    /// Step at which to transition from exploration to refinement.
    switch_step: usize,
    /// Current step counter.
    current_step: usize,
    /// Whether we have performed the warm-start transition yet.
    warmed_up: bool,
}

impl<A> HybridQuantumClassical<A>
where
    A: Float + ScalarOperand + Debug + Send + Sync,
{
    /// Create a new hybrid optimizer with default sub-optimizer settings.
    pub fn new(learning_rate: A, switch_step: usize) -> Self {
        let initial_phase = if switch_step == 0 {
            OptimizationPhase::Refinement
        } else {
            OptimizationPhase::Exploration
        };
        Self {
            quantum: QuantumAnnealing::new(learning_rate),
            classical: Adam::new(learning_rate),
            phase: initial_phase,
            switch_step,
            current_step: 0,
            warmed_up: false,
        }
    }

    /// Replace the underlying quantum annealer.
    pub fn with_quantum(mut self, quantum: QuantumAnnealing<A>) -> Self {
        self.quantum = quantum;
        self
    }

    /// Replace the underlying classical optimizer.
    pub fn with_classical(mut self, classical: Adam<A>) -> Self {
        self.classical = classical;
        self
    }

    /// Configure the quantum annealer via a builder closure.
    pub fn with_quantum_config<F>(mut self, f: F) -> Self
    where
        F: FnOnce(QuantumAnnealing<A>) -> QuantumAnnealing<A>,
    {
        self.quantum = f(self.quantum);
        self
    }

    /// Configure the classical optimizer via a builder closure.
    pub fn with_classical_config<F>(mut self, f: F) -> Self
    where
        F: FnOnce(Adam<A>) -> Adam<A>,
    {
        self.classical = f(self.classical);
        self
    }

    /// Returns the current optimization phase.
    pub fn phase(&self) -> OptimizationPhase {
        self.phase
    }

    /// Returns the configured switch step.
    pub fn switch_step(&self) -> usize {
        self.switch_step
    }

    /// Returns the current step counter.
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Returns whether the warm-start handoff has fired.
    pub fn warmed_up(&self) -> bool {
        self.warmed_up
    }

    /// Returns the current learning rate, choosing the relevant sub-optimizer
    /// based on the current phase.
    pub fn learning_rate(&self) -> A {
        match self.phase {
            OptimizationPhase::Exploration => self.quantum.learning_rate(),
            OptimizationPhase::Refinement => self.classical.learning_rate(),
        }
    }

    /// Set the learning rate on both the quantum annealer and the classical
    /// optimizer simultaneously.
    pub fn set_lr(&mut self, learning_rate: A) {
        self.quantum.set_lr(learning_rate);
        self.classical.set_lr(learning_rate);
    }

    /// Returns an immutable reference to the underlying quantum annealer.
    pub fn quantum(&self) -> &QuantumAnnealing<A> {
        &self.quantum
    }

    /// Returns an immutable reference to the underlying classical optimizer.
    pub fn classical(&self) -> &Adam<A> {
        &self.classical
    }

    /// Returns a mutable reference to the underlying quantum annealer.
    pub fn quantum_mut(&mut self) -> &mut QuantumAnnealing<A> {
        &mut self.quantum
    }

    /// Returns a mutable reference to the underlying classical optimizer.
    pub fn classical_mut(&mut self) -> &mut Adam<A> {
        &mut self.classical
    }

    /// Internal: pick the best params recorded by the annealer (if any) and
    /// warm-start the classical optimizer by stepping it with zero gradient.
    /// Returns the params the classical optimizer should treat as its anchor.
    fn maybe_warm_start<D>(&mut self, params: &Array<A, D>) -> Result<Array<A, D>>
    where
        D: Dimension,
    {
        if self.warmed_up {
            return Ok(params.clone());
        }
        let anchor: Array<A, D> = match self.quantum.best_params::<D>() {
            Some(best) if best.shape() == params.shape() => best,
            _ => params.clone(),
        };
        // Step Adam with zero gradient to register the anchor shape in its
        // internal state without actually changing the value (Adam with grad=0
        // and no weight decay simply leaves params unchanged).
        let zero_grads = Array::<A, D>::zeros(anchor.raw_dim());
        let warmed = self.classical.step(&anchor, &zero_grads)?;
        self.warmed_up = true;
        Ok(warmed)
    }
}

impl<A, D> Optimizer<A, D> for HybridQuantumClassical<A>
where
    A: Float + ScalarOperand + Debug + Send + Sync,
    D: Dimension,
{
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        if params.shape() != gradients.shape() {
            return Err(OptimError::DimensionMismatch(format!(
                "Hybrid optimizer: parameters have shape {:?}, gradients have shape {:?}",
                params.shape(),
                gradients.shape()
            )));
        }

        let in_exploration = self.current_step < self.switch_step;
        if in_exploration {
            self.phase = OptimizationPhase::Exploration;
            let next = self.quantum.step(params, gradients)?;
            self.current_step = self.current_step.saturating_add(1);
            Ok(next)
        } else {
            // Trigger warm-start transition on the first refinement step.
            let anchor = self.maybe_warm_start(params)?;
            self.phase = OptimizationPhase::Refinement;
            let next = self.classical.step(&anchor, gradients)?;
            self.current_step = self.current_step.saturating_add(1);
            Ok(next)
        }
    }

    fn get_learning_rate(&self) -> A {
        match self.phase {
            OptimizationPhase::Exploration => self.quantum.learning_rate(),
            OptimizationPhase::Refinement => self.classical.learning_rate(),
        }
    }

    fn set_learning_rate(&mut self, learning_rate: A) {
        self.quantum.set_lr(learning_rate);
        self.classical.set_lr(learning_rate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array1;

    fn quadratic_grad(params: &Array1<f64>) -> Array1<f64> {
        params.mapv(|x| 2.0 * x)
    }

    #[test]
    fn test_starts_in_exploration_phase() {
        let optimizer: HybridQuantumClassical<f64> = HybridQuantumClassical::new(0.05, 30);
        assert_eq!(optimizer.phase(), OptimizationPhase::Exploration);
        assert_eq!(optimizer.current_step(), 0);
        assert_eq!(optimizer.switch_step(), 30);
        assert!(!optimizer.warmed_up());
    }

    #[test]
    fn test_switches_at_correct_step() {
        let mut optimizer: HybridQuantumClassical<f64> = HybridQuantumClassical::new(0.05, 5)
            .with_quantum_config(|q| {
                q.with_temperature_schedule(1.0, 1e-3)
                    .with_iterations(50)
                    .with_seed(11)
            });
        let mut params = Array1::from_vec(vec![1.0, -1.0]);
        // First 5 steps must be exploration.
        for _ in 0..5 {
            let grads = quadratic_grad(&params);
            params = optimizer.step(&params, &grads).expect("step failed");
            assert_eq!(optimizer.phase(), OptimizationPhase::Exploration);
        }
        // Sixth step must be refinement.
        let grads = quadratic_grad(&params);
        params = optimizer.step(&params, &grads).expect("step failed");
        assert_eq!(optimizer.phase(), OptimizationPhase::Refinement);
        let _ = params; // suppress unused warning if any
    }

    #[test]
    fn test_classical_used_after_switch() {
        let mut optimizer: HybridQuantumClassical<f64> = HybridQuantumClassical::new(0.1, 3)
            .with_quantum_config(|q| {
                q.with_temperature_schedule(1.0, 1e-3)
                    .with_iterations(20)
                    .with_seed(101)
            });
        let mut params = Array1::from_vec(vec![2.0]);
        for _ in 0..3 {
            let grads = quadratic_grad(&params);
            params = optimizer.step(&params, &grads).expect("step failed");
        }
        assert!(!optimizer.warmed_up());
        let grads = quadratic_grad(&params);
        let _ = optimizer.step(&params, &grads).expect("step failed");
        assert!(optimizer.warmed_up());
        assert_eq!(optimizer.phase(), OptimizationPhase::Refinement);
    }

    #[test]
    fn test_phase_transition_preserves_best_params() {
        let mut optimizer: HybridQuantumClassical<f64> = HybridQuantumClassical::new(0.05, 30)
            .with_quantum_config(|q| {
                q.with_temperature_schedule(1.0, 1e-3)
                    .with_tunneling(0.1)
                    .with_iterations(60)
                    .with_seed(2024)
            });
        let mut params = Array1::from_vec(vec![5.0]);
        // Run exploration phase.
        for _ in 0..30 {
            let grads = quadratic_grad(&params);
            params = optimizer.step(&params, &grads).expect("step failed");
        }
        let best_energy_at_switch = optimizer.quantum().best_energy();
        // One refinement step.
        let grads = quadratic_grad(&params);
        let _ = optimizer.step(&params, &grads).expect("step failed");
        // best_energy is non-increasing because we only ever record improvements.
        assert!(
            optimizer.quantum().best_energy() <= best_energy_at_switch + 1e-12,
            "best_energy regressed after switch: before={best_energy_at_switch}, after={}",
            optimizer.quantum().best_energy()
        );
    }

    #[test]
    fn test_convergence_on_quadratic() {
        // Hybrid should converge at least as well as Adam on a convex bowl.
        let mut optimizer: HybridQuantumClassical<f64> = HybridQuantumClassical::new(0.05, 50)
            .with_quantum_config(|q| {
                q.with_temperature_schedule(1.0, 1e-4)
                    .with_tunneling(0.05)
                    .with_iterations(50)
                    .with_seed(7)
            });
        let mut params = Array1::from_vec(vec![5.0]);
        for _ in 0..400 {
            let grads = quadratic_grad(&params);
            params = optimizer.step(&params, &grads).expect("step failed");
        }
        assert!(
            params[0].abs() < 0.5,
            "Hybrid did not converge on quadratic: |x|={}",
            params[0].abs()
        );
    }

    #[test]
    fn test_switch_step_zero_means_pure_classical() {
        let mut optimizer: HybridQuantumClassical<f64> = HybridQuantumClassical::new(0.1, 0);
        // From step 0 we should already be in refinement.
        assert_eq!(optimizer.phase(), OptimizationPhase::Refinement);
        let params = Array1::from_vec(vec![2.0]);
        let grads = quadratic_grad(&params);
        let _ = optimizer.step(&params, &grads).expect("step failed");
        assert_eq!(optimizer.phase(), OptimizationPhase::Refinement);
        assert!(optimizer.warmed_up());
    }

    #[test]
    fn test_switch_step_inf_means_pure_quantum() {
        let mut optimizer: HybridQuantumClassical<f64> =
            HybridQuantumClassical::new(0.05, usize::MAX).with_quantum_config(|q| {
                q.with_temperature_schedule(1.0, 1e-3)
                    .with_iterations(100)
                    .with_seed(5)
            });
        let mut params = Array1::from_vec(vec![1.0, -1.0]);
        for _ in 0..50 {
            let grads = quadratic_grad(&params);
            params = optimizer.step(&params, &grads).expect("step failed");
            assert_eq!(optimizer.phase(), OptimizationPhase::Exploration);
        }
        assert!(!optimizer.warmed_up());
    }

    #[test]
    fn test_builder_pattern() {
        let optimizer: HybridQuantumClassical<f64> = HybridQuantumClassical::new(0.05, 20)
            .with_quantum_config(|q| {
                q.with_temperature_schedule(1.5, 1e-2)
                    .with_tunneling(0.4)
                    .with_iterations(100)
                    .with_seed(42)
            })
            .with_classical_config(|adam| adam.with_beta1(0.95).with_beta2(0.9999));
        assert_eq!(optimizer.switch_step(), 20);
        assert_abs_diff_eq!(optimizer.quantum().initial_temperature(), 1.5);
        assert_abs_diff_eq!(optimizer.quantum().final_temperature(), 1e-2);
        assert_abs_diff_eq!(optimizer.quantum().tunneling_strength(), 0.4);
        assert_eq!(optimizer.quantum().num_iterations(), 100);
        assert_eq!(optimizer.quantum().seed(), 42);
        assert_abs_diff_eq!(optimizer.classical().get_beta1(), 0.95);
        assert_abs_diff_eq!(optimizer.classical().get_beta2(), 0.9999);
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let mut optimizer: HybridQuantumClassical<f64> = HybridQuantumClassical::new(0.05, 10);
        let params = Array1::from_vec(vec![1.0, 2.0]);
        let grads = Array1::from_vec(vec![0.1]);
        let result = optimizer.step(&params, &grads);
        assert!(result.is_err(), "expected dimension mismatch error");
    }

    #[test]
    fn test_set_learning_rate_applies_to_both() {
        let mut optimizer: HybridQuantumClassical<f64> = HybridQuantumClassical::new(0.05, 5);
        optimizer.set_lr(0.2);
        assert_abs_diff_eq!(optimizer.quantum().learning_rate(), 0.2);
        assert_abs_diff_eq!(optimizer.classical().learning_rate(), 0.2);
    }
}
