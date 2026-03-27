//! # DatabaseStatistics - Trait Implementations
//!
//! This module contains trait implementations for `DatabaseStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::collections::{HashMap, VecDeque, BTreeMap};

use super::types::DatabaseStatistics;

impl<T: Float + Debug + Send + Sync + 'static> Default for DatabaseStatistics<T> {
    fn default() -> Self {
        Self {
            total_evaluations: 0,
            successful_evaluations: 0,
            best_objectives: HashMap::new(),
            average_objectives: HashMap::new(),
            objective_distributions: HashMap::new(),
        }
    }
}

