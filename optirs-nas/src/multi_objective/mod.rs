// Multi-objective optimization for neural architecture search
//
// Implements NSGA-II, NSGA-III, MOEA/D and a weighted-sum scalarizer for finding
// Pareto-optimal optimizer architectures, together with the shared machinery they
// need (Das-Dennis decomposition, hypervolume, genetic operators, front metrics).
//
// Two type-only modules were removed rather than kept as a zombie API:
//
// * `algorithms` declared `IBEA` and `SmsEmoa` structs with no constructor, no
//   methods and no `MultiObjectiveAlgorithm` variant that could select them, so
//   nothing inside or outside the crate could build or run either one.
// * `preference` declared `ConstraintHandler` / `PreferenceHandler` and their
//   parameter types the same way. `MultiObjectiveConfig::constraint_handling` and
//   `::user_preferences` are still declared and still read by nobody; that gap is
//   documented on those fields instead of being implied by empty types.

pub mod core;
pub mod decomposition;
pub mod hypervolume;
pub mod metrics;
pub mod moead;
pub mod nsga2;
pub mod nsga3;
pub mod operators;
pub mod weighted_sum;

// Re-export all types
pub use core::*;
pub use decomposition::*;
pub use hypervolume::*;
pub use metrics::*;
pub use moead::*;
pub use nsga2::*;
pub use nsga3::*;
pub use operators::*;
pub use weighted_sum::*;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_nsga3;
