//! # MultiObjectiveSettings - Trait Implementations
//!
//! This module contains trait implementations for `MultiObjectiveSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::{MultiObjectiveSettings, ParetoApproximationMethod, ScalarizationMethod};

impl<T: Float + Debug + Send + Sync + 'static> Default for MultiObjectiveSettings<T> {
    fn default() -> Self {
        Self {
            objectives: vec![],
            scalarization: ScalarizationMethod::WeightedSum,
            pareto_approximation: ParetoApproximationMethod::NonDominatedSorting,
            reference_point: None,
            weights: None,
        }
    }
}

