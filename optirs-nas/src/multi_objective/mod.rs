// Multi-objective optimization for neural architecture search
//
// Implements various multi-objective optimization algorithms including NSGA-II, NSGA-III,
// MOEA/D, and other state-of-the-art algorithms for finding Pareto-optimal optimizer architectures.

pub mod algorithms;
pub mod core;
pub mod hypervolume;
pub mod metrics;
pub mod moead;
pub mod nsga2;
pub mod nsga3;
pub mod operators;
pub mod preference;
pub mod weighted_sum;

// Re-export all types
pub use algorithms::*;
pub use core::*;
pub use hypervolume::*;
pub use metrics::*;
pub use moead::*;
pub use nsga2::*;
pub use nsga3::*;
pub use operators::*;
pub use preference::*;
pub use weighted_sum::*;

#[cfg(test)]
mod tests;
