//! # TaskDataset - Trait Implementations
//!
//! This module contains trait implementations for `TaskDataset`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::{DatasetMetadata, TaskDataset};

impl<T: Float + Debug + Send + Sync + 'static> Default for TaskDataset<T> {
    fn default() -> Self {
        Self {
            features: Vec::new(),
            targets: Vec::new(),
            weights: Vec::new(),
            metadata: DatasetMetadata::default(),
        }
    }
}
