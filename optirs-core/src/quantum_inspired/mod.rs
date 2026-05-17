// Quantum-inspired optimization algorithms
//
// This module provides optimization algorithms inspired by quantum computing
// concepts, implemented on classical hardware. These methods leverage ideas
// from quantum annealing, variational quantum eigensolvers (VQE), and hybrid
// quantum-classical optimization to explore non-convex loss landscapes more
// effectively than purely gradient-based methods.
//
// # Features
//
// - **Quantum Annealing**: Simulated quantum annealing with temperature-driven
//   exploration and an optional tunneling term that lets the optimizer escape
//   local minima.
// - **Variational Quantum Optimizer (VQE)**: A SPSA-based optimizer with a
//   quantum-inspired ansatz update rule that mimics rotation gate semantics.
// - **Hybrid Quantum-Classical**: A two-phase optimizer that performs broad
//   exploration with quantum annealing followed by fine-grained Adam-based
//   refinement once the search has localised.
//
// # Mathematical Background
//
// Classical Metropolis acceptance in quantum annealing uses the rule
//
// ```text
//     P(accept) = min(1, exp(-ΔE / (k * T) + Γ * K(δ)))
// ```
//
// where `Γ` is the tunneling strength and `K(δ) = exp(-‖δ‖²)` is a kernel that
// boosts the acceptance probability of nearby candidate moves to model
// quantum tunneling on classical hardware.
//
// # Examples
//
// ```ignore
// use optirs_core::quantum_inspired::{QuantumAnnealing, QuantumOptimizerConfig};
// use optirs_core::optimizers::Optimizer;
// use scirs2_core::ndarray::Array1;
//
// let mut optimizer: QuantumAnnealing<f64> = QuantumAnnealing::new(0.05)
//     .with_temperature_schedule(2.0, 0.01)
//     .with_tunneling(0.5)
//     .with_iterations(500)
//     .with_seed(42);
//
// let params = Array1::from_vec(vec![3.0, -2.0, 1.5]);
// let gradients = params.mapv(|x| 2.0 * x);
// let next = optimizer.step(&params, &gradients).expect("step failed");
// assert_eq!(next.len(), params.len());
// ```

mod annealing;
mod hybrid;
mod vqe;

pub use annealing::QuantumAnnealing;
pub use hybrid::{HybridQuantumClassical, OptimizationPhase};
pub use vqe::VariationalQuantumOptimizer;

/// Default initial temperature for quantum annealing schedules.
pub const DEFAULT_INITIAL_TEMP: f64 = 1.0;
/// Default final temperature for quantum annealing schedules.
pub const DEFAULT_FINAL_TEMP: f64 = 1.0e-3;
/// Default number of cooling iterations.
pub const DEFAULT_NUM_ITERATIONS: usize = 1000;
/// Default tunneling strength for the quantum-inspired Metropolis kernel.
pub const DEFAULT_TUNNELING_STRENGTH: f64 = 0.1;
/// Default RNG seed used when none is supplied.
pub const DEFAULT_SEED: u64 = 0xC001_5EED_F00D_BABE;

/// Shared configuration for quantum-inspired optimizers.
///
/// `QuantumOptimizerConfig` collects the high level knobs that drive every
/// quantum-inspired optimizer in this module. Sensible defaults are provided
/// via [`QuantumOptimizerConfig::default`] so users can opt into only the
/// parameters they care about.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantumOptimizerConfig {
    /// Initial temperature used at the start of the annealing schedule.
    pub initial_temperature: f64,
    /// Final temperature used at the end of the annealing schedule.
    pub final_temperature: f64,
    /// Total number of cooling iterations the schedule should span.
    pub num_iterations: usize,
    /// Strength of the quantum-inspired tunneling kernel.
    pub tunneling_strength: f64,
    /// Seed used to drive the deterministic RNG.
    pub seed: u64,
}

impl Default for QuantumOptimizerConfig {
    fn default() -> Self {
        Self {
            initial_temperature: DEFAULT_INITIAL_TEMP,
            final_temperature: DEFAULT_FINAL_TEMP,
            num_iterations: DEFAULT_NUM_ITERATIONS,
            tunneling_strength: DEFAULT_TUNNELING_STRENGTH,
            seed: DEFAULT_SEED,
        }
    }
}
